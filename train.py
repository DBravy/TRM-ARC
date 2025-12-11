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

# Import pixel error CNN components for intermittent testing
from train_pixel_error_cnn import (
    CorrespondenceDataset,
    evaluate_binary as evaluate_pixel_error_cnn,
    visualize_predictions as visualize_pixel_error_cnn,
    load_puzzles as load_raw_puzzles,
)

# Import sparse TRM (CNN-gated attention)
from models.recursive_reasoning.trm_sparse import TRMSparse, TRMSparseConfig

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


def run_pixel_error_test(
    cnn_model: nn.Module,
    raw_puzzles: Dict,
    step: int,
    num_negatives: int = 100,
    batch_size: int = 32,
    quiet: bool = True,
    augment: bool = True,
    dihedral_only: bool = False,
) -> Dict[str, float]:
    """
    Run pixel error test with augmentations on training puzzles.
    This verifies the CNN is working correctly by testing on synthetic samples.
    """
    if not quiet:
        print(f"\n{'='*60}")
        print(f"PIXEL ERROR CNN TEST (Step {step})")
        print(f"{'='*60}")

    # Create dataset with the specified negatives
    # Auto-distribute: 35% corrupted, 15% wrong_input, 15% mismatched_aug, 15% color_swap, 20% degenerate
    num_positives = max(1, num_negatives // 4)
    num_corrupted = max(1, int(num_negatives * 0.35))
    num_wrong_input = max(1, int(num_negatives * 0.15))
    # mismatched_aug only makes sense when augmentation is enabled
    num_mismatched_aug = max(1, int(num_negatives * 0.15)) if augment else 0
    num_color_swap = max(1, int(num_negatives * 0.15))
    # Degenerate outputs (all_zeros, constant_fill, random_noise) get remaining ~20%
    remaining = num_negatives - num_corrupted - num_wrong_input - num_mismatched_aug - num_color_swap
    num_all_zeros = max(1, remaining // 3)
    num_constant_fill = max(1, remaining // 3)
    num_random_noise = max(1, remaining - num_all_zeros - num_constant_fill)

    test_dataset = CorrespondenceDataset(
        raw_puzzles,
        num_positives=num_positives,
        num_corrupted=num_corrupted,
        num_wrong_input=num_wrong_input,
        num_mismatched_aug=num_mismatched_aug,
        num_all_zeros=num_all_zeros,
        num_constant_fill=num_constant_fill,
        num_random_noise=num_random_noise,
        num_color_swap=num_color_swap,
        augment=augment,
        dihedral_only=dihedral_only,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    # Run evaluation
    metrics = evaluate_pixel_error_cnn(cnn_model, test_loader, DEVICE)

    if not quiet:
        print(f"\nResults:")
        print(f"  Loss: {metrics['loss']:.4f}")
        print(f"  Pixel Accuracy: {metrics['pixel_accuracy']:.2%}")
        print(f"  Error Precision: {metrics['error_precision']:.2%}")
        print(f"  Error Recall: {metrics['error_recall']:.2%}")
        print(f"  Error IoU: {metrics['error_iou']:.2%}")
        print(f"  Perfect Rate: {metrics['perfect_rate']:.2%}")

        # Show visualization
        visualize_pixel_error_cnn(cnn_model, test_dataset, DEVICE, num_samples=8)

        print(f"{'='*60}\n")

    return {f"pixel_error_test/{k}": v for k, v in metrics.items()}


def visualize_predictions(
    model: nn.Module,
    eval_loader: DataLoader,
    step: int,
    metrics: Dict[str, float],
    num_samples: int = 3,
    identifier_map: Optional[List[str]] = None,
    cnn_model: Optional[nn.Module] = None,
):
    """Visualize predictions at eval interval, optionally with CNN error detection."""
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

                # Get CNN error predictions if available
                cnn_error_mask = None
                if cnn_model is not None:
                    # Convert grids to CNN format (values 0-9, not tokens 2-11)
                    inp_grid = np.clip(inputs - 2, 0, 9).astype(np.int64)
                    pred_grid = np.clip(pred - 2, 0, 9).astype(np.int64)

                    inp_t = torch.from_numpy(inp_grid).long().unsqueeze(0).to(DEVICE)
                    pred_t = torch.from_numpy(pred_grid).long().unsqueeze(0).to(DEVICE)

                    cnn_proba = cnn_model.predict_proba(inp_t, pred_t)[0].cpu().numpy()
                    cnn_error_mask = (cnn_proba < 0.5)  # True where CNN thinks pixel is wrong

                print(f"\n{'-'*80}")
                print(f"Sample: {puzzle_name}")
                print(f"Token Accuracy: {acc:.2%} ({correct}/{total})")
                print(f"{'-'*80}")

                # Build display strings
                gt_lines = grid_to_str(labels, r1, r2, c1, c2).split("\n")
                pred_lines = grid_to_str(pred, r1, r2, c1, c2).split("\n")

                # Build CNN error indicator lines
                if cnn_error_mask is not None:
                    cnn_lines = []
                    for r in range(r1, min(r2, r1 + 12)):
                        row = []
                        for c in range(c1, min(c2, c1 + 12)):
                            if cnn_error_mask[r, c]:
                                row.append("X")  # CNN thinks this is wrong
                            else:
                                row.append(".")
                        cnn_lines.append(" ".join(row))

                    # Also compute actual errors for comparison
                    actual_errors = (pred != labels) & mask.astype(bool)
                    actual_lines = []
                    for r in range(r1, min(r2, r1 + 12)):
                        row = []
                        for c in range(c1, min(c2, c1 + 12)):
                            if actual_errors[r, c]:
                                row.append("X")
                            else:
                                row.append(".")
                        actual_lines.append(" ".join(row))

                    print(f"{'GROUND TRUTH':<25} {'PREDICTION':<25} {'ACTUAL ERRORS':<25} {'CNN PRED ERRORS':<25}")
                    max_lines = max(len(gt_lines), len(pred_lines), len(cnn_lines))
                    for i in range(max_lines):
                        gt = gt_lines[i] if i < len(gt_lines) else ""
                        pr = pred_lines[i] if i < len(pred_lines) else ""
                        ae = actual_lines[i] if i < len(actual_lines) else ""
                        cn = cnn_lines[i] if i < len(cnn_lines) else ""
                        print(f"{gt:<25} {pr:<25} {ae:<25} {cn:<25}")

                    # CNN accuracy stats
                    cnn_pred_errors = cnn_error_mask[r1:r2, c1:c2].sum()
                    actual_error_count = actual_errors[r1:r2, c1:c2].sum()
                    cnn_correct = ((cnn_error_mask == actual_errors) & mask.astype(bool))[r1:r2, c1:c2].sum()
                    cnn_total = mask[r1:r2, c1:c2].sum()
                    print(f"\nCNN: {cnn_pred_errors} predicted errors, {actual_error_count} actual errors, "
                          f"{cnn_correct}/{cnn_total} pixels correct")
                else:
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

    print(f"{'='*80}\n")


# =============================================================================
# Dataset Building
# =============================================================================

def build_dataset_if_needed(args) -> str:
    """Build dataset from raw JSON if it doesn't exist. Returns dataset path."""
    config = DATASET_CONFIGS[args.dataset]

    # Determine augmentation suffix for directory name
    if args.no_augment:
        aug_suffix = "-noaug"
    elif args.dihedral_only:
        aug_suffix = "-dihedral"
    else:
        aug_suffix = ""  # Default full augmentation

    # Use separate directory for single puzzle mode
    if args.single_puzzle:
        output_dir = os.path.join(args.data_root, f"single-puzzle-{args.single_puzzle}{aug_suffix}")
    else:
        output_dir = os.path.join(args.data_root, config["output_dir_suffix"] + aug_suffix)

    # Check if dataset already exists
    train_dataset_json = os.path.join(output_dir, "train", "dataset.json")
    if os.path.exists(train_dataset_json):
        print(f"Dataset already exists at {output_dir}")
        return output_dir

    print(f"Building dataset for {args.dataset}...")
    if args.single_puzzle:
        print(f"Single puzzle mode: {args.single_puzzle}")
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
        test_num_aug=args.test_num_aug,
        puzzle_identifiers_start=1,
        single_puzzle=args.single_puzzle,
        augment=not args.no_augment,
        dihedral_only=args.dihedral_only,
    )

    convert_dataset(build_config)
    print(f"Dataset built successfully at {output_dir}")

    return output_dir


# =============================================================================
# Model Creation
# =============================================================================

def create_model(args, metadata, device):
    """Create TRM model with ACTLossHead.
    
    Supports two modes:
    - Standard TRM (default)
    - Sparse TRM with CNN-gated attention (--sparse-attention)
    """
    from utils.functions import load_model_class

    dtype_str = "float32" if DTYPE == torch.float32 else "bfloat16"
    
    # Determine CNN checkpoint path
    cnn_path = args.cnn_checkpoint if os.path.exists(args.cnn_checkpoint) else None
    
    if args.sparse_attention:
        # =====================================================================
        # Sparse TRM: CNN-gated attention
        # Only positions the CNN thinks are WRONG actively compute
        # =====================================================================
        print(f"\n*** Using SPARSE ATTENTION mode ***")
        print(f"  Sparsity mode: {args.sparsity_mode}")
        print(f"  CNN error threshold: {args.cnn_error_threshold}")
        print(f"  CNN checkpoint: {cnn_path or 'None (will train from scratch)'}")
        
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
            # Sparse attention config
            "cnn_checkpoint_path": cnn_path,
            "cnn_error_threshold": args.cnn_error_threshold,
            "cnn_warmup_steps": args.cnn_warmup_steps,
            "cnn_loss_weight": args.cnn_loss_weight,
            "sparsity_mode": args.sparsity_mode,
            "min_active_ratio": args.min_active_ratio,
        }
        
        loss_cls = load_model_class("losses@ACTLossHead")
        
        with torch.device(device):
            model = TRMSparse(config_dict)
            model = loss_cls(model, loss_type="stablemax_cross_entropy", cnn_loss_weight=config_dict["cnn_loss_weight"])
        
    else:
        # =====================================================================
        # Standard TRM (original behavior)
        # =====================================================================
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
            # CNN-guided pixel freezing (uses pretrained CNN)
            "cnn_checkpoint_path": cnn_path,
            "cnn_freeze_threshold": 0.5,
            "cnn_loss_weight": 0.0,
            "cnn_freeze_warmup_steps": 0,
            # Dynamic iteration mode (CNN-guided stopping)
            "dynamic_iterations": args.dynamic_iterations,
            "dynamic_error_threshold": args.dynamic_error_threshold,
            "dynamic_max_steps": args.dynamic_max_steps,
            "dynamic_min_steps": args.dynamic_min_steps,
            # Force error pixel changes
            "force_error_changes": args.force_error_changes,
            "force_error_scale": args.force_error_scale,
        }

        model_cls = load_model_class("recursive_reasoning.trm@TinyRecursiveReasoningModel_ACTV1")
        loss_cls = load_model_class("losses@ACTLossHead")

        with torch.device(device):
            model = model_cls(config_dict)
            model = loss_cls(model, loss_type="stablemax_cross_entropy", cnn_loss_weight=config_dict["cnn_loss_weight"])

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
    result_metrics = {
        "train/loss": loss.item() / global_batch_size,
        "train/accuracy": metrics["accuracy"].item() / count,
        "train/exact_accuracy": metrics["exact_accuracy"].item() / count,
        "train/lr": lr,
    }

    # Add CNN loss if present
    if "cnn_loss" in metrics:
        result_metrics["train/cnn_loss"] = metrics["cnn_loss"].item()

    # Add sparsity metrics if present (from sparse attention mode)
    if "sparsity/active_ratio" in metrics:
        result_metrics["train/sparsity_active_ratio"] = metrics["sparsity/active_ratio"]
    if "sparsity/inactive_ratio" in metrics:
        result_metrics["train/sparsity_inactive_ratio"] = metrics["sparsity/inactive_ratio"]

    return carry, result_metrics


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


def evaluate_arc(
    model: nn.Module,
    eval_loader: DataLoader,
    data_path: str,
    metadata,
    pass_Ks: tuple = (1, 2),
) -> Dict[str, float]:
    """Run ARC evaluation with voting mechanism.

    This uses the ARC evaluator which:
    1. Collects predictions from all augmented versions of each puzzle
    2. Uses inverse_aug to map predictions back to original space
    3. Votes on the best prediction based on frequency and confidence
    4. Returns pass@K metrics
    """
    from evaluators.arc import ARC

    evaluator = ARC(
        data_path=data_path,
        eval_metadata=metadata,
        submission_K=2,
        pass_Ks=pass_Ks,
        aggregated_voting=True
    )

    evaluator.begin_eval()
    model.eval()

    with torch.inference_mode():
        for set_name, batch, global_batch_size in eval_loader:
            batch = {k: v.to(DEVICE) for k, v in batch.items()}

            carry = model.initial_carry(batch)
            while True:
                carry, loss, metrics, outputs, all_finish = model(
                    carry=carry, batch=batch,
                    return_keys=["preds", "q_halt_logits"]
                )
                if all_finish:
                    break

            # Prepare predictions for evaluator
            preds_dict = {
                "preds": outputs["preds"],
                "q_halt_logits": outputs["q_halt_logits"]
            }

            evaluator.update_batch(batch, preds_dict)

    # Single-process mode: rank=0, world_size=1
    result = evaluator.result(save_path=None, rank=0, world_size=1)
    return result if result is not None else {}


# =============================================================================
# Main Training Loop
# =============================================================================

def train(args):
    print(f"\n{'='*60}")
    print(f"TRM-ARC Training")
    print(f"{'='*60}")
    print(f"Device: {DEVICE}, dtype: {DTYPE}")
    print(f"Dataset: {args.dataset}")

    # Determine augmentation mode
    if args.no_augment:
        aug_mode = "none"
    elif args.dihedral_only:
        aug_mode = "dihedral-only (rotations/flips, no color permutations)"
    else:
        aug_mode = "full (dihedral + color permutations)"
    print(f"Augmentation: {aug_mode}")
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

    # Load raw puzzles for pixel error test
    raw_puzzles = None
    if args.pixel_error_test:
        print(f"\nLoading raw puzzles for pixel error test...")
        raw_puzzles = load_raw_puzzles(args.dataset)
        if args.single_puzzle and args.single_puzzle in raw_puzzles:
            raw_puzzles = {args.single_puzzle: raw_puzzles[args.single_puzzle]}
        print(f"Loaded {len(raw_puzzles)} puzzles for pixel error test")

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

    # Load CNN error detector for visualization (optional)
    cnn_model = None
    if args.cnn_checkpoint and os.path.exists(args.cnn_checkpoint):
        print(f"\nLoading CNN error detector from {args.cnn_checkpoint}...")
        try:
            from train_pixel_error_cnn import PixelErrorCNN
            cnn_model = PixelErrorCNN.from_checkpoint(args.cnn_checkpoint, DEVICE)
            print("CNN error detector loaded successfully")
        except Exception as e:
            print(f"Warning: Could not load CNN model: {e}")
            cnn_model = None

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

            postfix = {
                "loss": f"{metrics['train/loss']:.4f}",
                "acc": f"{metrics['train/accuracy']:.2%}",
                "lr": f"{metrics['train/lr']:.2e}",
            }
            if "train/cnn_loss" in metrics:
                postfix["cnn"] = f"{metrics['train/cnn_loss']:.4f}"
            if "train/sparsity_active_ratio" in metrics:
                postfix["sparse"] = f"{metrics['train/sparsity_active_ratio']:.1%}"
            pbar.set_postfix(postfix)

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
                sparsity_str = ""
                if "train/sparsity_active_ratio" in metrics:
                    sparsity_str = f" | Sparse: {metrics['train/sparsity_active_ratio']:.1%} active"
                print(f"    Loss: {avg_interval_loss:.4f} | "
                      f"Acc: {avg_interval_acc:.2%} | "
                      f"Exact: {avg_interval_exact:.2%} | "
                      f"LR: {metrics['train/lr']:.2e}{sparsity_str}")

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

        # ARC voting evaluation (optional)
        if args.use_arc_eval:
            # Need to recreate eval_loader since it's an iterator
            arc_eval_loader = DataLoader(
                PuzzleDataset(eval_config, split="test"),
                batch_size=None,
                num_workers=0,
            )
            arc_metrics = evaluate_arc(
                model, arc_eval_loader, dataset_path, metadata,
                pass_Ks=(1, 2)
            )
            for k, v in arc_metrics.items():
                print(f"  {k}: {v:.2%}")
            val_metrics.update(arc_metrics)

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
                cnn_model=cnn_model,
            )

        # Pixel error CNN test (runs at same rate as eval visualizations)
        if args.pixel_error_test and cnn_model is not None and raw_puzzles is not None:
            pixel_error_metrics = run_pixel_error_test(
                cnn_model, raw_puzzles, step,
                num_negatives=args.pixel_error_test_negatives,
                quiet=args.quiet_pixel_error_test,
                augment=not args.no_augment,
                dihedral_only=args.dihedral_only,
            )
            if not args.no_wandb:
                wandb.log(pixel_error_metrics, step=step)

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
    parser.add_argument("--test-num-aug", type=int, default=100,
                        help="Number of augmentations for test split (0 = one prediction per puzzle)")
    parser.add_argument("--single-puzzle", type=str, default=None,
                        help="Train on only this puzzle ID (e.g., '00d62c1b')")

    # Model
    parser.add_argument("--hidden-size", type=int, default=512)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--H-cycles", type=int, default=3)
    parser.add_argument("--L-cycles", type=int, default=6)
    parser.add_argument("--halt-max-steps", type=int, default=16)
    parser.add_argument("--compile", action="store_true",
                        help="Use torch.compile for optimization")

    # Dynamic iteration mode (CNN-guided stopping)
    parser.add_argument("--dynamic-iterations", action="store_true",
                        help="Enable dynamic iteration mode (stop when CNN error < threshold)")
    parser.add_argument("--dynamic-error-threshold", type=float, default=0.1,
                        help="Stop iterating when CNN error rate drops below this (0.1 = 10%%)")
    parser.add_argument("--dynamic-max-steps", type=int, default=8,
                        help="Maximum iterations in dynamic mode (keep low for memory)")
    parser.add_argument("--dynamic-min-steps", type=int, default=1,
                        help="Minimum iterations before checking error threshold")

    # Force error pixel changes (CNN-guided)
    parser.add_argument("--force-error-changes", action="store_true",
                        help="Force model to output different values for pixels CNN marks as errors")
    parser.add_argument("--force-error-scale", type=float, default=1.0,
                        help="Scale of perturbation applied to stuck error pixels (default: 1.0)")

    # Sparse attention mode (CNN-gated computation)
    parser.add_argument("--sparse-attention", action="store_true",
                        help="Enable CNN-gated sparse attention (only compute on positions CNN thinks are wrong)")
    parser.add_argument("--sparsity-mode", type=str, default="soft",
                        choices=["soft", "hard"],
                        help="Sparsity mode: 'soft' (mask outputs) or 'hard' (gather/scatter)")
    parser.add_argument("--cnn-error-threshold", type=float, default=0.5,
                        help="CNN confidence below this = active/needs work (default: 0.5)")
    parser.add_argument("--cnn-warmup-steps", type=int, default=0,
                        help="Training steps before enabling sparse attention (default: 0)")
    parser.add_argument("--cnn-loss-weight", type=float, default=0.0,
                        help="Weight for CNN loss in joint training (default: 0.0 = frozen CNN)")
    parser.add_argument("--min-active-ratio", type=float, default=0.01,
                        help="Minimum ratio of positions that must be active (default: 0.01)")

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
    parser.add_argument("--use-arc-eval", action="store_true",
                        help="Use ARC voting evaluator for pass@K metrics (requires --test-num-aug > 0)")
    parser.add_argument("--visualize", action="store_true", default=True,
                        help="Show ASCII visualization at each eval")
    parser.add_argument("--no-visualize", action="store_false", dest="visualize",
                        help="Disable visualization")
    parser.add_argument("--vis-samples", type=int, default=3,
                        help="Number of samples to visualize per eval")
    parser.add_argument("--cnn-checkpoint", type=str, default="checkpoints/pixel_error_cnn.pt",
                        help="Path to CNN error detector checkpoint for visualization")
    parser.add_argument("--pixel-error-test", action="store_true", default=True,
                        help="Run pixel error CNN test with augmentations at each eval")
    parser.add_argument("--no-pixel-error-test", action="store_false", dest="pixel_error_test",
                        help="Disable pixel error CNN test")
    parser.add_argument("--pixel-error-test-negatives", type=int, default=100,
                        help="Number of negatives for pixel error test")
    parser.add_argument("--quiet-pixel-error-test", action="store_true", default=True,
                        help="Suppress pixel error test console output (default)")
    parser.add_argument("--verbose-pixel-error-test", action="store_false", dest="quiet_pixel_error_test",
                        help="Show pixel error test console output")

    # Augmentation options (for pixel error test and dataset building)
    parser.add_argument("--no-augment", action="store_true",
                        help="Disable augmentations (use raw example grids)")
    parser.add_argument("--dihedral-only", action="store_true",
                        help="Only use dihedral transforms (rotations/flips), no color permutations")

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