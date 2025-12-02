#!/usr/bin/env python3
"""
Unified TRM-ARC Training Script

Supports:
- Training on all puzzles from ARC-AGI-1 or ARC-AGI-2
- Week-long training runs without memory/storage issues
- Visualization at each eval interval
- Resume from checkpoint capability

Usage:
    # Train on ARC-AGI-1:
    python train.py --dataset arc-agi-1

    # Train on ARC-AGI-2:
    python train.py --dataset arc-agi-2

    # Resume from checkpoint:
    python train.py --dataset arc-agi-1 --resume checkpoints/step_10000

    # Disable WandB:
    python train.py --dataset arc-agi-1 --no-wandb
"""

import argparse
import gc
import json
import math
import os
import random
import shutil
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False


def get_memory_usage_mb():
    """Get current process memory usage in MB."""
    if HAS_PSUTIL:
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024
    return None

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Determine device and dtype
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    DTYPE = torch.bfloat16
else:
    if torch.backends.mps.is_available():
        DEVICE = torch.device("mps")
    else:
        DEVICE = torch.device("cpu")
    DTYPE = torch.float32


# =============================================================================
# Dataset Configuration
# =============================================================================

DATASET_CONFIGS = {
    "arc-agi-1": {
        "json_prefix": "kaggle/combined/arc-agi",
        "subsets": ["training", "evaluation"],
        "test_set_name": "evaluation",
        "output_dir_suffix": "arc-agi-1-aug",
    },
    "arc-agi-2": {
        "json_prefix": "kaggle/combined/arc-agi",
        "subsets": ["training2", "evaluation2"],
        "test_set_name": "evaluation2",
        "output_dir_suffix": "arc-agi-2-aug",
    },
}


# =============================================================================
# Checkpoint Manager
# =============================================================================

@dataclass
class CheckpointMetadata:
    step: int
    epoch: int
    train_loss: float
    val_accuracy: Optional[float]
    val_exact_accuracy: Optional[float]
    timestamp: str


class CheckpointManager:
    """Manages checkpoint saving with rotation to prevent disk fill-up."""

    def __init__(
        self,
        checkpoint_dir: str,
        keep_recent: int = 5,
        keep_best: int = 3,
        metric_key: str = "val_exact_accuracy",
        higher_is_better: bool = True,
    ):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.keep_recent = keep_recent
        self.keep_best = keep_best
        self.metric_key = metric_key
        self.higher_is_better = higher_is_better

        # Track checkpoints: list of (step, path)
        self.recent_checkpoints: List[Tuple[int, str]] = []
        # Track best: list of (metric, step, path)
        self.best_checkpoints: List[Tuple[float, int, str]] = []

        self._load_history()

    def _load_history(self):
        history_file = self.checkpoint_dir / "checkpoint_history.json"
        if history_file.exists():
            with open(history_file, "r") as f:
                data = json.load(f)
                self.recent_checkpoints = [(c["step"], c["path"]) for c in data.get("recent", [])]
                self.best_checkpoints = [(c["metric"], c["step"], c["path"]) for c in data.get("best", [])]

    def _save_history(self):
        history_file = self.checkpoint_dir / "checkpoint_history.json"
        data = {
            "recent": [{"step": s, "path": p} for s, p in self.recent_checkpoints],
            "best": [{"metric": m, "step": s, "path": p} for m, s, p in self.best_checkpoints],
        }
        with open(history_file, "w") as f:
            json.dump(data, f, indent=2)

    def save(self, model: nn.Module, metadata: CheckpointMetadata) -> str:
        """Save checkpoint with automatic rotation."""
        checkpoint_name = f"step_{metadata.step}"
        checkpoint_path = self.checkpoint_dir / checkpoint_name

        # Save checkpoint atomically (write to temp, then rename)
        temp_path = checkpoint_path.with_suffix(".tmp")

        # Save model state dict only (minimal checkpoint)
        torch.save({
            "model_state_dict": model.state_dict(),
            "step": metadata.step,
        }, temp_path)

        # Atomic rename
        temp_path.rename(checkpoint_path)

        # Save metadata separately for quick inspection
        with open(str(checkpoint_path) + ".json", "w") as f:
            json.dump(asdict(metadata), f, indent=2)

        # Update recent checkpoints
        self.recent_checkpoints.append((metadata.step, str(checkpoint_path)))
        self._rotate_recent()

        # Update best checkpoints if metric available
        metric_value = getattr(metadata, self.metric_key.replace("val_", ""), None)
        if metric_value is not None:
            self.best_checkpoints.append((metric_value, metadata.step, str(checkpoint_path)))
            self._rotate_best()

        self._save_history()

        return str(checkpoint_path)

    def _rotate_recent(self):
        """Remove old recent checkpoints beyond keep_recent limit."""
        self.recent_checkpoints.sort(key=lambda x: x[0], reverse=True)

        best_paths = {p for _, _, p in self.best_checkpoints}

        while len(self.recent_checkpoints) > self.keep_recent:
            step, path = self.recent_checkpoints.pop()
            if path not in best_paths:
                self._delete_checkpoint(path)

    def _rotate_best(self):
        """Keep only top keep_best checkpoints by metric."""
        self.best_checkpoints.sort(key=lambda x: x[0], reverse=self.higher_is_better)

        recent_paths = {p for _, p in self.recent_checkpoints}

        while len(self.best_checkpoints) > self.keep_best:
            metric, step, path = self.best_checkpoints.pop()
            if path not in recent_paths:
                self._delete_checkpoint(path)

    def _delete_checkpoint(self, path: str):
        """Safely delete a checkpoint file."""
        try:
            Path(path).unlink(missing_ok=True)
            Path(path + ".json").unlink(missing_ok=True)
        except Exception as e:
            print(f"Warning: Could not delete checkpoint {path}: {e}")

    def get_latest(self) -> Optional[str]:
        """Get path to latest checkpoint."""
        if self.recent_checkpoints:
            return self.recent_checkpoints[0][1]
        return None


# =============================================================================
# Memory Manager
# =============================================================================

class MemoryManager:
    """Manages memory for long training runs."""

    def __init__(self, empty_cache_interval: int = 100, gc_interval: int = 1000):
        self.empty_cache_interval = empty_cache_interval
        self.gc_interval = gc_interval
        self.step = 0

    def step_update(self, step: int):
        """Called after each training step."""
        self.step = step

        if step % self.empty_cache_interval == 0 and torch.cuda.is_available():
            torch.cuda.empty_cache()

        if step % self.gc_interval == 0:
            gc.collect()


# =============================================================================
# Visualization Functions
# =============================================================================

def find_grid_bounds(grid: np.ndarray) -> Tuple[int, int, int, int]:
    """Find actual grid boundaries (non-padding area)."""
    # Tokens: PAD=0, EOS=1, digits=2-11
    mask = (grid >= 2) & (grid <= 11)
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not rows.any():
        return 0, 5, 0, 5
    r_min, r_max = np.where(rows)[0][[0, -1]]
    c_min, c_max = np.where(cols)[0][[0, -1]]
    return r_min, r_max + 1, c_min, c_max + 1


def grid_to_str(grid: np.ndarray, r1: int, r2: int, c1: int, c2: int) -> str:
    """Convert grid to string for display."""
    lines = []
    for r in range(r1, min(r2, r1 + 12)):
        row = []
        for c in range(c1, min(c2, c1 + 12)):
            val = grid[r, c]
            if val == 0:
                row.append(".")
            elif val == 1:
                row.append("|")
            else:
                row.append(str(val - 2))
        lines.append(" ".join(row))
    return "\n".join(lines)


def visualize_predictions(
    model: nn.Module,
    eval_loader: DataLoader,
    step: int,
    metrics: Dict[str, float],
    num_samples: int = 3,
    identifier_map: Optional[List[str]] = None,
):
    """Visualize predictions at eval interval."""
    model.eval()

    print(f"\n{'='*60}")
    print(f"STEP {step} | Accuracy: {metrics.get('val/accuracy', 0):.2%} | "
          f"Exact: {metrics.get('val/exact_accuracy', 0):.2%}")
    print(f"{'='*60}")

    samples_shown = 0

    with torch.inference_mode():
        for set_name, batch, global_batch_size in eval_loader:
            if samples_shown >= num_samples:
                break

            batch = {k: v.to(DEVICE) for k, v in batch.items()}

            # Get initial carry and run inference
            carry = model.initial_carry(batch)
            while True:
                carry, loss, batch_metrics, preds, all_finish = model(
                    carry=carry, batch=batch, return_keys=["preds"]
                )
                if all_finish:
                    break

            # Show samples from this batch
            batch_size = batch["inputs"].shape[0]
            indices = list(range(min(num_samples - samples_shown, batch_size)))

            for idx in indices:
                inputs = batch["inputs"][idx].cpu().numpy().reshape(30, 30)
                labels = batch["labels"][idx].cpu().numpy().reshape(30, 30)
                pred = preds["preds"][idx].cpu().numpy().reshape(30, 30)
                puzzle_id = batch["puzzle_identifiers"][idx].item()

                puzzle_name = f"ID:{puzzle_id}"
                if identifier_map and puzzle_id < len(identifier_map):
                    puzzle_name = identifier_map[puzzle_id][:40]

                r1, r2, c1, c2 = find_grid_bounds(labels)

                # Compute accuracy
                mask = (labels != 0) & (labels != -100)
                correct = ((pred == labels) & mask).sum()
                total = mask.sum()
                acc = correct / total if total > 0 else 0

                print(f"\n{'-'*60}")
                print(f"Sample: {puzzle_name}")
                print(f"Token Accuracy: {acc:.2%} ({correct}/{total})")
                print(f"{'-'*60}")

                gt_lines = grid_to_str(labels, r1, r2, c1, c2).split("\n")
                pred_lines = grid_to_str(pred, r1, r2, c1, c2).split("\n")

                print(f"{'GROUND TRUTH':<25} {'PREDICTION':<25}")
                max_lines = max(len(gt_lines), len(pred_lines))
                for i in range(max_lines):
                    gt = gt_lines[i] if i < len(gt_lines) else ""
                    pr = pred_lines[i] if i < len(pred_lines) else ""
                    print(f"{gt:<25} {pr:<25}")

                samples_shown += 1
                if samples_shown >= num_samples:
                    break

            break  # Only need one batch

    print(f"{'='*60}\n")


# =============================================================================
# Dataset Building
# =============================================================================

def build_dataset_if_needed(args) -> str:
    """Build dataset from raw JSON if it doesn't exist. Returns dataset path."""
    config = DATASET_CONFIGS[args.dataset]
    output_dir = os.path.join(args.data_root, config["output_dir_suffix"])

    # Check if dataset already exists
    train_dataset_json = os.path.join(output_dir, "train", "dataset.json")
    if os.path.exists(train_dataset_json):
        print(f"Dataset already exists at {output_dir}")
        return output_dir

    print(f"Building dataset for {args.dataset}...")
    print(f"This may take a few minutes...")

    # Import dataset builder
    from dataset.build_arc_dataset import DataProcessConfig, convert_dataset

    # Determine the JSON prefix based on dataset
    json_prefix = config["json_prefix"]

    build_config = DataProcessConfig(
        input_file_prefix=json_prefix,
        output_dir=output_dir,
        subsets=config["subsets"],
        test_set_name=config["test_set_name"],
        seed=args.seed,
        num_aug=args.num_augmentations,
        puzzle_identifiers_start=1,
    )

    convert_dataset(build_config)
    print(f"Dataset built successfully at {output_dir}")

    return output_dir


# =============================================================================
# Model Creation
# =============================================================================

def create_model(args, metadata, device):
    """Create TRM model with ACTLossHead."""
    from utils.functions import load_model_class

    dtype_str = "float32" if DTYPE == torch.float32 else "bfloat16"

    config_dict = {
        "batch_size": args.batch_size,
        "seq_len": metadata.seq_len,
        "vocab_size": metadata.vocab_size,
        "num_puzzle_identifiers": metadata.num_puzzle_identifiers,
        "hidden_size": args.hidden_size,
        "num_heads": args.num_heads,
        "L_layers": args.num_layers,
        "H_layers": 0,
        "H_cycles": args.H_cycles,
        "L_cycles": args.L_cycles,
        "expansion": 4.0,
        "puzzle_emb_ndim": args.hidden_size,
        "puzzle_emb_len": 16,
        "halt_max_steps": args.halt_max_steps,
        "halt_exploration_prob": 0.1,
        "pos_encodings": "rope",
        "rms_norm_eps": 1e-5,
        "rope_theta": 10000.0,
        "forward_dtype": dtype_str,
        "mlp_t": False,
        "no_ACT_continue": True,
        "causal": False,
    }

    model_cls = load_model_class("recursive_reasoning.trm@TinyRecursiveReasoningModel_ACTV1")
    loss_cls = load_model_class("losses@ACTLossHead")

    with torch.device(device):
        model = model_cls(config_dict)
        model = loss_cls(model, loss_type="stablemax_cross_entropy")

    if args.compile and "DISABLE_COMPILE" not in os.environ:
        model = torch.compile(model)

    return model


def create_optimizers(args, model):
    """Create optimizers for model."""
    from models.sparse_embedding import CastedSparseEmbeddingSignSGD_Distributed

    optimizers = []
    optimizer_lrs = []

    # Puzzle embedding optimizer
    if args.hidden_size > 0:
        optimizers.append(
            CastedSparseEmbeddingSignSGD_Distributed(
                model.model.puzzle_emb.buffers(),
                lr=0,
                weight_decay=args.puzzle_emb_weight_decay,
                world_size=1,
            )
        )
        optimizer_lrs.append(args.puzzle_emb_lr)

    # Main optimizer
    optimizers.append(
        torch.optim.AdamW(
            model.parameters(),
            lr=0,
            weight_decay=args.weight_decay,
            betas=(args.beta1, args.beta2),
        )
    )
    optimizer_lrs.append(args.lr)

    return optimizers, optimizer_lrs


# =============================================================================
# Learning Rate Schedule
# =============================================================================

def cosine_schedule_with_warmup(
    current_step: int,
    base_lr: float,
    num_warmup_steps: int,
    num_training_steps: int,
    min_ratio: float = 0.0,
) -> float:
    if current_step < num_warmup_steps:
        return base_lr * float(current_step) / float(max(1, num_warmup_steps))

    progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
    return base_lr * (min_ratio + max(0.0, (1 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))))


# =============================================================================
# Training Functions
# =============================================================================

def train_step(
    model: nn.Module,
    optimizers: List,
    optimizer_lrs: List[float],
    batch: Dict[str, torch.Tensor],
    global_batch_size: int,
    carry: Any,
    step: int,
    total_steps: int,
    args,
) -> Tuple[Any, Dict[str, float]]:
    """Single training step."""
    model.train()

    # Init carry if None
    if carry is None:
        with torch.device(DEVICE):
            carry = model.initial_carry(batch)

    # Forward
    carry, loss, metrics, _, _ = model(carry=carry, batch=batch, return_keys=[])

    # Backward
    ((1 / global_batch_size) * loss).backward()

    # Compute LR and step optimizers
    for opt, base_lr in zip(optimizers, optimizer_lrs):
        lr = cosine_schedule_with_warmup(
            step, base_lr, args.lr_warmup_steps, total_steps, args.lr_min_ratio
        )
        for param_group in opt.param_groups:
            param_group["lr"] = lr
        opt.step()
        opt.zero_grad()

    # Extract metrics
    count = max(metrics["count"].item(), 1)
    return carry, {
        "train/loss": loss.item() / global_batch_size,
        "train/accuracy": metrics["accuracy"].item() / count,
        "train/exact_accuracy": metrics["exact_accuracy"].item() / count,
        "train/lr": lr,
    }


def evaluate(
    model: nn.Module,
    eval_loader: DataLoader,
) -> Dict[str, float]:
    """Run evaluation."""
    model.eval()

    total_metrics = {"count": 0, "accuracy": 0, "exact_accuracy": 0}

    with torch.inference_mode():
        for set_name, batch, global_batch_size in eval_loader:
            batch = {k: v.to(DEVICE) for k, v in batch.items()}

            carry = model.initial_carry(batch)
            while True:
                carry, loss, metrics, preds, all_finish = model(
                    carry=carry, batch=batch, return_keys=["preds"]
                )
                if all_finish:
                    break

            count = max(metrics["count"].item(), 1)
            total_metrics["count"] += count
            total_metrics["accuracy"] += metrics["accuracy"].item()
            total_metrics["exact_accuracy"] += metrics["exact_accuracy"].item()

    count = max(total_metrics["count"], 1)
    return {
        "val/accuracy": total_metrics["accuracy"] / count,
        "val/exact_accuracy": total_metrics["exact_accuracy"] / count,
    }


# =============================================================================
# Main Training Loop
# =============================================================================

def train(args):
    print(f"\n{'='*60}")
    print(f"TRM-ARC Training")
    print(f"{'='*60}")
    print(f"Device: {DEVICE}, dtype: {DTYPE}")
    print(f"Dataset: {args.dataset}")
    print(f"{'='*60}\n")

    # Set seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Build dataset if needed
    dataset_path = build_dataset_if_needed(args)

    # Force garbage collection after dataset building to free memory
    gc.collect()
    mem = get_memory_usage_mb()
    if mem:
        print(f"Memory after dataset building: {mem:.0f} MB")

    # Import dataset utilities
    from puzzle_dataset import PuzzleDataset, PuzzleDatasetConfig

    # Create dataloaders
    train_config = PuzzleDatasetConfig(
        seed=args.seed,
        dataset_paths=[dataset_path],
        global_batch_size=args.batch_size,
        test_set_mode=False,
        epochs_per_iter=args.eval_interval,
        rank=0,
        num_replicas=1,
    )
    train_dataset = PuzzleDataset(train_config, split="train")
    train_loader = DataLoader(
        train_dataset,
        batch_size=None,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )

    eval_config = PuzzleDatasetConfig(
        seed=args.seed,
        dataset_paths=[dataset_path],
        global_batch_size=args.batch_size,
        test_set_mode=True,
        epochs_per_iter=1,
        rank=0,
        num_replicas=1,
    )
    eval_dataset = PuzzleDataset(eval_config, split="test")
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=None,
        num_workers=0,
        pin_memory=True,
    )

    metadata = train_dataset.metadata
    print(f"Dataset loaded:")
    print(f"  - Total puzzles: {metadata.total_puzzles}")
    print(f"  - Total groups: {metadata.total_groups}")
    print(f"  - Mean examples per puzzle: {metadata.mean_puzzle_examples:.1f}")
    print(f"  - Vocab size: {metadata.vocab_size}")
    print(f"  - Sequence length: {metadata.seq_len}")
    print(f"  - Puzzle identifiers: {metadata.num_puzzle_identifiers}")

    # Load identifier map for visualization
    identifier_map = None
    identifiers_path = os.path.join(dataset_path, "identifiers.json")
    if os.path.exists(identifiers_path):
        with open(identifiers_path) as f:
            identifier_map = json.load(f)

    # Create model
    print(f"\nCreating model...")
    model = create_model(args, metadata, DEVICE)
    model = model.to(DEVICE)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")
    mem = get_memory_usage_mb()
    if mem:
        print(f"Memory after model creation: {mem:.0f} MB")

    # Create optimizers
    optimizers, optimizer_lrs = create_optimizers(args, model)

    # Compute total steps
    total_steps = int(
        args.epochs * metadata.total_groups * metadata.mean_puzzle_examples / args.batch_size
    )
    num_evals = args.epochs // args.eval_interval

    print(f"\nTraining configuration:")
    print(f"  - Total epochs: {args.epochs}")
    print(f"  - Total steps: {total_steps}")
    print(f"  - Eval interval: {args.eval_interval} epochs")
    print(f"  - Number of evaluations: {num_evals}")

    # Initialize managers
    checkpoint_manager = CheckpointManager(
        checkpoint_dir=args.checkpoint_dir,
        keep_recent=args.checkpoint_keep,
        keep_best=args.checkpoint_best,
    )
    memory_manager = MemoryManager(empty_cache_interval=args.empty_cache_interval)

    # Resume from checkpoint
    start_step = 0
    start_eval = 0
    if args.resume:
        print(f"\nLoading checkpoint from {args.resume}...")
        checkpoint = torch.load(args.resume, map_location=DEVICE)
        model.load_state_dict(checkpoint["model_state_dict"])
        start_step = checkpoint.get("step", 0)
        start_eval = start_step // (args.eval_interval * args.batch_size // (metadata.total_groups * metadata.mean_puzzle_examples))
        print(f"Resumed from step {start_step}")

    # Initialize WandB
    if not args.no_wandb:
        import wandb
        import coolname

        run_name = args.run_name or f"TRM-{args.dataset}-{coolname.generate_slug(2)}"
        wandb.init(
            project=args.project_name,
            name=run_name,
            config=vars(args),
            settings=wandb.Settings(_disable_stats=True),
        )
        wandb.log({"num_params": num_params}, step=0)
        print(f"\nWandB initialized: {run_name}")

    # Training loop
    print(f"\n{'='*60}")
    print("TRAINING START")
    print(f"{'='*60}\n")

    step = start_step
    best_val_acc = 0.0
    training_start_time = time.time()

    for eval_iter in range(start_eval, num_evals):
        epoch_start = eval_iter * args.eval_interval
        print(f"\n[Epoch {epoch_start}-{epoch_start + args.eval_interval}] Training...")

        # Training iteration
        model.train()
        carry = None
        train_loss = 0.0
        train_acc = 0.0
        train_exact_acc = 0.0
        train_count = 0
        interval_loss = 0.0
        interval_acc = 0.0
        interval_exact_acc = 0.0
        interval_count = 0
        interval_start_time = time.time()

        pbar = tqdm(train_loader, desc=f"Training", leave=False)
        for set_name, batch, global_batch_size in pbar:
            step += 1
            if step > total_steps:
                break

            batch = {k: v.to(DEVICE) for k, v in batch.items()}

            carry, metrics = train_step(
                model, optimizers, optimizer_lrs, batch, global_batch_size,
                carry, step, total_steps, args
            )

            train_loss += metrics["train/loss"]
            train_acc += metrics["train/accuracy"]
            train_exact_acc += metrics["train/exact_accuracy"]
            train_count += 1

            interval_loss += metrics["train/loss"]
            interval_acc += metrics["train/accuracy"]
            interval_exact_acc += metrics["train/exact_accuracy"]
            interval_count += 1

            pbar.set_postfix({
                "loss": f"{metrics['train/loss']:.4f}",
                "acc": f"{metrics['train/accuracy']:.2%}",
                "lr": f"{metrics['train/lr']:.2e}",
            })

            memory_manager.step_update(step)

            if not args.no_wandb:
                wandb.log(metrics, step=step)

            # Print detailed stats every log_interval steps
            if step % args.log_interval == 0:
                elapsed = time.time() - training_start_time
                interval_elapsed = time.time() - interval_start_time
                steps_per_sec = interval_count / max(interval_elapsed, 1e-6)
                avg_interval_loss = interval_loss / max(interval_count, 1)
                avg_interval_acc = interval_acc / max(interval_count, 1)
                avg_interval_exact = interval_exact_acc / max(interval_count, 1)

                hours, remainder = divmod(int(elapsed), 3600)
                minutes, seconds = divmod(remainder, 60)

                print(f"\n  Step {step:,}/{total_steps:,} ({100*step/total_steps:.1f}%) | "
                      f"Time: {hours:02d}:{minutes:02d}:{seconds:02d} | "
                      f"{steps_per_sec:.1f} steps/s")
                print(f"    Loss: {avg_interval_loss:.4f} | "
                      f"Acc: {avg_interval_acc:.2%} | "
                      f"Exact: {avg_interval_exact:.2%} | "
                      f"LR: {metrics['train/lr']:.2e}")

                # Reset interval tracking
                interval_loss = 0.0
                interval_acc = 0.0
                interval_exact_acc = 0.0
                interval_count = 0
                interval_start_time = time.time()

        avg_train_loss = train_loss / max(train_count, 1)
        avg_train_acc = train_acc / max(train_count, 1)
        avg_train_exact = train_exact_acc / max(train_count, 1)

        # Print epoch summary
        print(f"\n{'─'*60}")
        print(f"[Epoch {epoch_start + args.eval_interval}] Training Summary:")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Train Accuracy: {avg_train_acc:.2%}")
        print(f"  Train Exact Accuracy: {avg_train_exact:.2%}")
        print(f"{'─'*60}")

        # Evaluation
        print(f"[Epoch {epoch_start + args.eval_interval}] Evaluating...")
        val_metrics = evaluate(model, eval_loader)

        if val_metrics["val/exact_accuracy"] > best_val_acc:
            best_val_acc = val_metrics["val/exact_accuracy"]
            print(f"  *** New best validation accuracy! ***")

        print(f"  Val Accuracy: {val_metrics['val/accuracy']:.2%}")
        print(f"  Val Exact Accuracy: {val_metrics['val/exact_accuracy']:.2%}")
        print(f"  Best Val Exact: {best_val_acc:.2%}")

        if not args.no_wandb:
            wandb.log(val_metrics, step=step)

        # Visualization
        if args.visualize:
            # Need to recreate eval_loader for visualization (it's exhausted)
            viz_loader = DataLoader(
                PuzzleDataset(eval_config, split="test"),
                batch_size=None,
                num_workers=0,
            )
            visualize_predictions(
                model, viz_loader, step, val_metrics,
                num_samples=args.vis_samples,
                identifier_map=identifier_map,
            )

        # Save checkpoint
        ckpt_metadata = CheckpointMetadata(
            step=step,
            epoch=epoch_start + args.eval_interval,
            train_loss=avg_train_loss,
            val_accuracy=val_metrics["val/accuracy"],
            val_exact_accuracy=val_metrics["val/exact_accuracy"],
            timestamp=datetime.now().isoformat(),
        )
        checkpoint_manager.save(model, ckpt_metadata)
        print(f"  Checkpoint saved at step {step}")

    # Finalize
    print(f"\n{'='*60}")
    print("TRAINING COMPLETE")
    print(f"{'='*60}")
    print(f"Best Val Exact Accuracy: {best_val_acc:.2%}")
    print(f"Total steps: {step}")

    if not args.no_wandb:
        wandb.finish()


# =============================================================================
# Main
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="TRM-ARC Training Script")

    # Dataset
    parser.add_argument("--dataset", type=str, default="arc-agi-1",
                        choices=["arc-agi-1", "arc-agi-2"],
                        help="Which ARC dataset to train on")
    parser.add_argument("--data-root", type=str, default="data",
                        help="Root directory for datasets")
    parser.add_argument("--num-augmentations", type=int, default=1000,
                        help="Number of augmentations per puzzle when building dataset")

    # Model
    parser.add_argument("--hidden-size", type=int, default=512)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--H-cycles", type=int, default=3)
    parser.add_argument("--L-cycles", type=int, default=6)
    parser.add_argument("--halt-max-steps", type=int, default=16)
    parser.add_argument("--compile", action="store_true",
                        help="Use torch.compile for optimization")

    # Training
    parser.add_argument("--epochs", type=int, default=100000)
    parser.add_argument("--batch-size", type=int, default=768)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr-min-ratio", type=float, default=1.0)
    parser.add_argument("--lr-warmup-steps", type=int, default=2000)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--puzzle-emb-lr", type=float, default=1e-2)
    parser.add_argument("--puzzle-emb-weight-decay", type=float, default=0.1)

    # Checkpointing
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints",
                        help="Directory for saving checkpoints")
    parser.add_argument("--checkpoint-keep", type=int, default=5,
                        help="Number of recent checkpoints to keep")
    parser.add_argument("--checkpoint-best", type=int, default=3,
                        help="Number of best checkpoints to keep")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")

    # Evaluation & Visualization
    parser.add_argument("--eval-interval", type=int, default=10000,
                        help="Evaluate every N epochs")
    parser.add_argument("--visualize", action="store_true", default=True,
                        help="Show ASCII visualization at each eval")
    parser.add_argument("--no-visualize", action="store_false", dest="visualize",
                        help="Disable visualization")
    parser.add_argument("--vis-samples", type=int, default=3,
                        help="Number of samples to visualize per eval")

    # Logging (WandB enabled by default)
    parser.add_argument("--no-wandb", action="store_true",
                        help="Disable WandB logging")
    parser.add_argument("--project-name", type=str, default="TRM-ARC")
    parser.add_argument("--run-name", type=str, default=None)

    # Memory
    parser.add_argument("--empty-cache-interval", type=int, default=100,
                        help="Clear CUDA cache every N steps")
    parser.add_argument("--num-workers", type=int, default=0,
                        help="DataLoader workers (0 = main process, reduces RAM)")
    parser.add_argument("--prefetch-factor", type=int, default=2,
                        help="DataLoader prefetch factor (only if num-workers > 0)")

    # Misc
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-interval", type=int, default=100,
                        help="Print training stats every N steps")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
