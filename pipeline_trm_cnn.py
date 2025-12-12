#!/usr/bin/env python3
"""
TRM + CNN Unified Pipeline

This script automates the entire pipeline:
1. Train TRM for N epochs on a single puzzle
2. Train CNN for M epochs on the same puzzle (color mode)
3. Get CNN prediction from test input (starting with all zeros)
4. Initialize TRM's hidden state (z_H) with CNN prediction embeddings
5. Run one TRM forward pass to refine the prediction
6. Report accuracy results

Usage:
    python pipeline_trm_cnn.py --puzzle 00d62c1b --trm-epochs 1000 --cnn-epochs 25

    # Skip TRM training if checkpoint exists:
    python pipeline_trm_cnn.py --puzzle 00d62c1b --trm-checkpoint checkpoints/puzzle_00d62c1b/best.pt

    # Skip CNN training if checkpoint exists:
    python pipeline_trm_cnn.py --puzzle 00d62c1b --cnn-checkpoint checkpoints/cnn_00d62c1b.pt
"""

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# Device Setup
# =============================================================================

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    DTYPE = torch.bfloat16
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
    DTYPE = torch.float32
else:
    DEVICE = torch.device("cpu")
    DTYPE = torch.float32

GRID_SIZE = 30
SEQ_LEN = 900  # 30 * 30


# =============================================================================
# Puzzle Loading
# =============================================================================

def load_puzzle(puzzle_id: str, data_root: str = "kaggle/combined") -> Dict:
    """Load a single puzzle from the ARC dataset."""
    # Try both ARC-AGI-1 and ARC-AGI-2 subsets
    subsets = [
        ("training", "training"),
        ("evaluation", "evaluation"),
        ("training2", "training2"),
        ("evaluation2", "evaluation2"),
    ]

    for subset_name, subset_key in subsets:
        challenges_path = f"{data_root}/arc-agi_{subset_key}_challenges.json"
        solutions_path = f"{data_root}/arc-agi_{subset_key}_solutions.json"

        if not os.path.exists(challenges_path):
            continue

        with open(challenges_path) as f:
            puzzles = json.load(f)

        if puzzle_id not in puzzles:
            continue

        puzzle = puzzles[puzzle_id]

        # Load solutions if available
        if os.path.exists(solutions_path):
            with open(solutions_path) as f:
                solutions = json.load(f)
            if puzzle_id in solutions:
                for i, sol in enumerate(solutions[puzzle_id]):
                    if i < len(puzzle["test"]):
                        puzzle["test"][i]["output"] = sol

        return puzzle

    raise ValueError(f"Puzzle '{puzzle_id}' not found in dataset")


# =============================================================================
# Step 1: Train TRM
# =============================================================================

def train_trm(
    puzzle_id: str,
    epochs: int,
    checkpoint_dir: str,
    hidden_size: int = 256,
    num_heads: int = 4,
    num_layers: int = 2,
    H_cycles: int = 2,
    L_cycles: int = 3,
    batch_size: int = 4,
    lr: float = 1e-4,
    dynamic_iterations: bool = True,
    dynamic_max_steps: int = 20,
    use_arc_eval: bool = True,
) -> str:
    """Train TRM model using train.py subprocess.

    Returns path to best checkpoint.
    """
    print("\n" + "=" * 60)
    print("STEP 1: Training TRM")
    print("=" * 60)

    cmd = [
        sys.executable, "train.py",
        "--dataset", "arc-agi-1",
        "--single-puzzle", puzzle_id,
        "--epochs", str(epochs),
        "--checkpoint-dir", checkpoint_dir,
        "--hidden-size", str(hidden_size),
        "--num-heads", str(num_heads),
        "--num-layers", str(num_layers),
        "--H-cycles", str(H_cycles),
        "--L-cycles", str(L_cycles),
        "--batch-size", str(batch_size),
        "--lr", str(lr),
        "--eval-interval", str(max(1000, epochs)),
    ]

    if dynamic_iterations:
        cmd.extend([
            "--dynamic-iterations",
            "--dynamic-error-threshold", "0.0",
            "--dynamic-max-steps", str(dynamic_max_steps),
        ])

    if use_arc_eval:
        cmd.append("--use-arc-eval")

    print(f"Running: {' '.join(cmd)}")
    print()

    result = subprocess.run(cmd, cwd=os.path.dirname(os.path.abspath(__file__)))

    if result.returncode != 0:
        raise RuntimeError(f"TRM training failed with return code {result.returncode}")

    # Find checkpoint using checkpoint_history.json (preferred)
    history_path = os.path.join(checkpoint_dir, "checkpoint_history.json")
    if os.path.exists(history_path):
        with open(history_path) as f:
            history = json.load(f)

        # Try best checkpoint first
        if history.get("best"):
            best_entry = history["best"][0]  # First is best (sorted by metric)
            if os.path.exists(best_entry["path"]):
                print(f"Found best checkpoint: {best_entry['path']}")
                return best_entry["path"]

        # Fall back to most recent
        if history.get("recent"):
            recent_entry = history["recent"][0]  # First is most recent
            if os.path.exists(recent_entry["path"]):
                print(f"Found recent checkpoint: {recent_entry['path']}")
                return recent_entry["path"]

    # Fall back to scanning directory for step_* files
    if os.path.exists(checkpoint_dir):
        all_files = os.listdir(checkpoint_dir)
        print(f"Files in {checkpoint_dir}: {all_files}")

        checkpoints = [f for f in all_files
                      if f.startswith("step_") and not f.endswith(".json")]
        if checkpoints:
            # Sort by step number
            checkpoints.sort(key=lambda x: int(x.split("_")[1]) if x.split("_")[1].isdigit() else 0, reverse=True)
            checkpoint_path = os.path.join(checkpoint_dir, checkpoints[0])
            print(f"Found checkpoint by scanning: {checkpoint_path}")
            return checkpoint_path

    raise RuntimeError(
        f"No checkpoints found in {checkpoint_dir}. "
        f"This can happen if training was interrupted before the first eval. "
        f"Try reducing --eval-interval or increasing --trm-epochs."
    )


# =============================================================================
# Step 2: Train CNN
# =============================================================================

def train_cnn(
    puzzle_id: str,
    epochs: int,
    checkpoint_path: str,
    hidden_dim: int = 64,
) -> str:
    """Train CNN model using train_pixel_error_cnn.py subprocess.

    Returns path to checkpoint.
    """
    print("\n" + "=" * 60)
    print("STEP 2: Training CNN")
    print("=" * 60)

    cmd = [
        sys.executable, "train_pixel_error_cnn.py",
        "--dataset", "arc-agi-1",
        "--single-puzzle", puzzle_id,
        "--mode", "color",  # Color prediction mode
        "--epochs", str(epochs),
        "--hidden-dim", str(hidden_dim),
        "--save-path", checkpoint_path,
        "--eval-on-test",  # Evaluate on held-out test examples
    ]

    print(f"Running: {' '.join(cmd)}")
    print()

    result = subprocess.run(cmd, cwd=os.path.dirname(os.path.abspath(__file__)))

    if result.returncode != 0:
        raise RuntimeError(f"CNN training failed with return code {result.returncode}")

    return checkpoint_path


# =============================================================================
# Step 3: Load Models
# =============================================================================

def load_trm_model(checkpoint_path: str, device: torch.device):
    """Load trained TRM model from checkpoint."""
    from utils.functions import load_model_class
    from puzzle_dataset import PuzzleDatasetConfig

    print(f"\nLoading TRM from {checkpoint_path}...")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # We need to reconstruct the model with the same config
    # The checkpoint only has model_state_dict and step
    # We need to infer config from state dict shapes

    state_dict = checkpoint["model_state_dict"]

    # Infer dimensions from state dict
    # model.model.inner.embed_tokens.embedding_weight has shape [vocab_size, hidden_size]
    embed_weight_key = None
    for key in state_dict.keys():
        if "embed_tokens" in key and "weight" in key:
            embed_weight_key = key
            break

    if embed_weight_key is None:
        raise ValueError("Could not find embed_tokens weight in checkpoint")

    vocab_size, hidden_size = state_dict[embed_weight_key].shape

    # Infer num_heads from attention weights
    # Look for q_proj weight shape
    num_heads = 8  # Default
    for key in state_dict.keys():
        if "q_proj" in key and "weight" in key:
            # Shape is [hidden_size, hidden_size]
            # We can't directly infer num_heads, use default
            break

    # Count L_level layers
    num_layers = 0
    for key in state_dict.keys():
        if "L_level.layers" in key:
            layer_idx = int(key.split("L_level.layers.")[1].split(".")[0])
            num_layers = max(num_layers, layer_idx + 1)

    # Infer puzzle_emb_ndim
    puzzle_emb_ndim = hidden_size  # Default
    for key in state_dict.keys():
        if "puzzle_emb" in key and "embedding_weight" in key:
            puzzle_emb_ndim = state_dict[key].shape[-1]
            break

    # Get num_puzzle_identifiers from puzzle_emb weights
    # Note: key could be "puzzle_emb.weights" or "puzzle_emb.embedding_weight"
    num_puzzle_identifiers = 1  # Default
    for key in state_dict.keys():
        if "puzzle_emb" in key and ("weights" in key or "embedding_weight" in key):
            num_puzzle_identifiers = state_dict[key].shape[0]
            print(f"  Detected num_puzzle_identifiers={num_puzzle_identifiers} from {key}")
            break

    # Check if checkpoint has embedded CNN (from dynamic_iterations training)
    has_embedded_cnn = any("correctness_cnn" in key for key in state_dict.keys())
    if has_embedded_cnn:
        print(f"  Checkpoint has embedded correctness_cnn, will initialize with 'init'")

    dtype_str = "bfloat16" if DTYPE == torch.bfloat16 else "float32"

    # Use memory-efficient defaults that match training
    config_dict = {
        "batch_size": 1,  # For inference
        "seq_len": SEQ_LEN,
        "vocab_size": vocab_size,
        "num_puzzle_identifiers": num_puzzle_identifiers,
        "hidden_size": hidden_size,
        "num_heads": num_heads,
        "L_layers": num_layers,
        "H_layers": 0,
        "H_cycles": 2,  # Memory-efficient default
        "L_cycles": 3,  # Memory-efficient default
        "expansion": 4.0,
        "puzzle_emb_ndim": puzzle_emb_ndim,
        "puzzle_emb_len": 16,
        "halt_max_steps": 20,  # Match dynamic_max_steps
        "halt_exploration_prob": 0.1,
        "pos_encodings": "rope",
        "rms_norm_eps": 1e-5,
        "rope_theta": 10000.0,
        "forward_dtype": dtype_str,
        "mlp_t": False,
        "no_ACT_continue": True,
        "causal": False,
        # If checkpoint has CNN, we need to create one to load the weights
        "cnn_checkpoint_path": "init" if has_embedded_cnn else None,
        "cnn_freeze_threshold": 0.5,
        "cnn_loss_weight": 0.0,
        "cnn_freeze_warmup_steps": 0,
        "dynamic_iterations": False,  # Disable for inference
        "dynamic_error_threshold": 0.0,
        "dynamic_max_steps": 20,
        "dynamic_min_steps": 1,
        "force_error_changes": False,
        "force_error_scale": 0.1,
    }

    model_cls = load_model_class("recursive_reasoning.trm@TinyRecursiveReasoningModel_ACTV1")
    loss_cls = load_model_class("losses@ACTLossHead")

    with torch.device(device):
        model = model_cls(config_dict)
        model = loss_cls(model, loss_type="stablemax_cross_entropy", cnn_loss_weight=0.0)

    model.load_state_dict(state_dict)
    model.eval()
    model = model.to(device)

    print(f"  Loaded TRM: hidden_size={hidden_size}, num_layers={num_layers}, vocab_size={vocab_size}")

    return model, config_dict


def load_cnn_model(checkpoint_path: str, device: torch.device):
    """Load trained CNN model from checkpoint."""
    from train_pixel_error_cnn import PixelErrorCNN

    print(f"\nLoading CNN from {checkpoint_path}...")

    model = PixelErrorCNN.from_checkpoint(checkpoint_path, device)
    model.eval()

    print(f"  Loaded CNN in color mode")

    return model


# =============================================================================
# Step 4: Get CNN Prediction
# =============================================================================

def get_cnn_prediction(
    cnn_model: nn.Module,
    input_grid: np.ndarray,
    device: torch.device,
    num_iterations: int = 1,
) -> np.ndarray:
    """Get CNN's predicted output starting from all-zeros candidate.

    Args:
        cnn_model: Trained CNN in color mode
        input_grid: Input grid [H, W] with values 0-9
        device: Target device
        num_iterations: Number of recursive refinement iterations

    Returns:
        Predicted output grid [30, 30] with values 0-9
    """
    # Pad input to 30x30
    h, w = input_grid.shape
    padded_input = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.int64)
    padded_input[:h, :w] = input_grid

    input_t = torch.from_numpy(padded_input).long().unsqueeze(0).to(device)

    # Start with all-zeros candidate
    candidate = torch.zeros(1, GRID_SIZE, GRID_SIZE, dtype=torch.long, device=device)

    with torch.no_grad():
        for i in range(num_iterations):
            logits = cnn_model(input_t, candidate)  # [1, 10, 30, 30]
            candidate = logits.argmax(dim=1)  # [1, 30, 30]

    return candidate[0].cpu().numpy()


# =============================================================================
# Step 5: Initialize TRM with CNN Embeddings
# =============================================================================

def initialize_trm_with_cnn(
    trm_model: nn.Module,
    cnn_prediction: np.ndarray,
    puzzle_identifier: int,
    config_dict: Dict,
    device: torch.device,
):
    """Create TRM initial carry with CNN prediction embeddings.

    This is the key innovation: instead of using learned H_init,
    we construct z_H from CNN predictions.

    Returns:
        inner_carry: Custom carry with CNN-initialized z_H
        batch: Input batch dict for TRM
    """
    from models.recursive_reasoning.trm import TinyRecursiveReasoningModel_ACTV1InnerCarry

    # Access inner model: ACTLossHead -> ACTV1 -> Inner
    # ACTLossHead.model -> TinyRecursiveReasoningModel_ACTV1.inner
    inner = trm_model.model.inner

    hidden_size = config_dict["hidden_size"]
    seq_len = config_dict["seq_len"]
    puzzle_emb_len = inner.puzzle_emb_len
    batch_size = 1

    # Step 1: Convert CNN prediction (colors 0-9) to token IDs (2-11)
    # Vocab: PAD=0, EOS=1, colors 0-9 -> tokens 2-11
    cnn_tokens = (cnn_prediction.flatten() + 2).astype(np.int64)
    cnn_tokens_t = torch.from_numpy(cnn_tokens).to(torch.int32).to(device)

    # Step 2: Embed tokens using TRM's embed_tokens layer
    with torch.no_grad():
        token_embeddings = inner.embed_tokens(cnn_tokens_t)  # [900, hidden_size]

    # Step 3: Get puzzle embeddings
    puzzle_id_t = torch.tensor([puzzle_identifier], device=device)

    if config_dict["puzzle_emb_ndim"] > 0:
        puzzle_emb = inner.puzzle_emb(puzzle_id_t)  # [1, puzzle_emb_ndim]

        # Pad and reshape to [puzzle_emb_len, hidden_size]
        pad_count = puzzle_emb_len * hidden_size - puzzle_emb.shape[-1]
        if pad_count > 0:
            puzzle_emb = F.pad(puzzle_emb, (0, pad_count))
        puzzle_emb = puzzle_emb.view(puzzle_emb_len, hidden_size)
    else:
        # No puzzle embeddings, use zeros
        puzzle_emb = torch.zeros(puzzle_emb_len, hidden_size, device=device, dtype=inner.forward_dtype)

    # Step 4: Construct full z_H
    # Shape: [1, puzzle_emb_len + seq_len, hidden_size]
    z_H = torch.cat([
        puzzle_emb.unsqueeze(0),  # [1, puzzle_emb_len, hidden_size]
        token_embeddings.unsqueeze(0)  # [1, seq_len, hidden_size]
    ], dim=1).to(inner.forward_dtype)

    # Scale by embed_scale (like _input_embeddings does)
    z_H = inner.embed_scale * z_H

    # Step 5: Initialize z_L with L_init (standard)
    L_init = inner.L_init.to(device)
    z_L = L_init.unsqueeze(0).expand(batch_size, seq_len + puzzle_emb_len, -1).clone()

    inner_carry = TinyRecursiveReasoningModel_ACTV1InnerCarry(z_H=z_H, z_L=z_L)

    return inner_carry


# =============================================================================
# Step 6: Run TRM Forward Pass
# =============================================================================

def run_trm_forward(
    trm_model: nn.Module,
    inner_carry,
    input_grid: np.ndarray,
    puzzle_identifier: int,
    config_dict: Dict,
    device: torch.device,
) -> np.ndarray:
    """Run TRM forward pass with custom initial carry.

    Returns:
        TRM predictions [30, 30] as colors (0-9)
    """
    # Access inner model
    # ACTLossHead.model -> TinyRecursiveReasoningModel_ACTV1.inner
    inner = trm_model.model.inner

    # Create input batch
    h, w = input_grid.shape
    padded_input = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.int64)
    padded_input[:h, :w] = input_grid + 2  # Convert to token IDs

    inputs = torch.from_numpy(padded_input.flatten()).to(torch.int32).unsqueeze(0).to(device)
    puzzle_ids = torch.tensor([puzzle_identifier], device=device)
    labels = torch.full((1, SEQ_LEN), -100, dtype=torch.long, device=device)  # Ignored

    batch = {
        "inputs": inputs,
        "puzzle_identifiers": puzzle_ids,
        "labels": labels,
    }

    trm_model.eval()

    with torch.no_grad():
        # Run forward pass through inner model directly
        new_carry, logits, q_logits, cnn_loss = inner(inner_carry, batch)

        # logits shape: [B, seq_len, vocab_size]
        predictions = logits.argmax(dim=-1)  # [B, seq_len]

    # Reshape to 2D grid and convert from tokens to colors
    pred_tokens = predictions[0].cpu().numpy().reshape(GRID_SIZE, GRID_SIZE)
    pred_colors = np.clip(pred_tokens - 2, 0, 9)  # Token 2-11 -> Color 0-9

    return pred_colors


# =============================================================================
# Accuracy Computation
# =============================================================================

def compute_accuracy(
    prediction: np.ndarray,
    ground_truth: np.ndarray,
) -> Dict:
    """Compute accuracy metrics.

    Args:
        prediction: Predicted grid [H, W] or [30, 30] with colors 0-9
        ground_truth: Expected output grid [H, W] with colors 0-9

    Returns:
        Dict with metrics
    """
    h, w = ground_truth.shape

    # Extract relevant region from prediction (may be padded)
    pred_region = prediction[:h, :w]

    total_pixels = h * w
    correct_pixels = (pred_region == ground_truth).sum()

    return {
        "pixel_accuracy": correct_pixels / total_pixels,
        "correct_pixels": int(correct_pixels),
        "total_pixels": total_pixels,
        "exact_match": correct_pixels == total_pixels,
    }


# =============================================================================
# Visualization
# =============================================================================

# ANSI color codes for ARC colors
ARC_COLORS = {
    0: "\033[40m",   # Black
    1: "\033[44m",   # Blue
    2: "\033[41m",   # Red
    3: "\033[42m",   # Green
    4: "\033[43m",   # Yellow
    5: "\033[47m",   # Gray (white bg)
    6: "\033[45m",   # Magenta
    7: "\033[48;5;208m",  # Orange
    8: "\033[46m",   # Cyan
    9: "\033[48;5;94m",   # Brown
}
RESET = "\033[0m"


def visualize_grid(grid: np.ndarray, title: str = ""):
    """Print a colored grid to terminal."""
    h, w = grid.shape
    if title:
        print(f"  {title}:")
    for row in range(h):
        line = "    "
        for col in range(w):
            color = int(grid[row, col])
            line += f"{ARC_COLORS.get(color, '')}{color}{RESET}"
        print(line)
    print()


def visualize_comparison(
    input_grid: np.ndarray,
    ground_truth: np.ndarray,
    cnn_prediction: np.ndarray,
    trm_prediction: np.ndarray,
):
    """Visualize input, ground truth, CNN prediction, and TRM prediction."""
    h, w = ground_truth.shape

    print("\n" + "-" * 60)
    print("VISUALIZATION")
    print("-" * 60)

    visualize_grid(input_grid, "Input")
    visualize_grid(ground_truth, "Ground Truth")
    visualize_grid(cnn_prediction[:h, :w], "CNN Prediction")
    visualize_grid(trm_prediction[:h, :w], "TRM+CNN Prediction")


# =============================================================================
# Main Pipeline
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="TRM + CNN Unified Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Puzzle selection
    parser.add_argument("--puzzle", type=str, default="00d62c1b",
                        help="Puzzle ID to train on")
    parser.add_argument("--data-root", type=str, default="kaggle/combined",
                        help="Path to ARC dataset")

    # TRM training (memory-efficient defaults)
    parser.add_argument("--trm-epochs", type=int, default=1000,
                        help="Number of epochs to train TRM")
    parser.add_argument("--trm-checkpoint", type=str, default=None,
                        help="Path to existing TRM checkpoint (skip training)")
    parser.add_argument("--trm-checkpoint-dir", type=str, default=None,
                        help="Directory to save TRM checkpoints")
    parser.add_argument("--hidden-size", type=int, default=256,
                        help="TRM hidden size (default: 256 for low RAM)")
    parser.add_argument("--num-heads", type=int, default=4,
                        help="TRM attention heads (default: 4 for low RAM)")
    parser.add_argument("--num-layers", type=int, default=2,
                        help="TRM transformer layers")
    parser.add_argument("--H-cycles", type=int, default=2,
                        help="TRM H cycles (default: 2 for low RAM)")
    parser.add_argument("--L-cycles", type=int, default=3,
                        help="TRM L cycles (default: 3 for low RAM)")
    parser.add_argument("--batch-size", type=int, default=4,
                        help="TRM batch size (default: 4 for low RAM)")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="TRM learning rate")
    parser.add_argument("--dynamic-iterations", action="store_true", default=True,
                        help="Use dynamic iterations (CNN-guided stopping)")
    parser.add_argument("--no-dynamic-iterations", action="store_false", dest="dynamic_iterations",
                        help="Disable dynamic iterations")
    parser.add_argument("--dynamic-max-steps", type=int, default=20,
                        help="Max steps for dynamic iterations")
    parser.add_argument("--use-arc-eval", action="store_true", default=True,
                        help="Use ARC evaluation with voting")
    parser.add_argument("--no-arc-eval", action="store_false", dest="use_arc_eval",
                        help="Disable ARC evaluation")

    # CNN training
    parser.add_argument("--cnn-epochs", type=int, default=25,
                        help="Number of epochs to train CNN")
    parser.add_argument("--cnn-checkpoint", type=str, default=None,
                        help="Path to existing CNN checkpoint (skip training)")
    parser.add_argument("--cnn-hidden-dim", type=int, default=64,
                        help="CNN hidden dimension")
    parser.add_argument("--cnn-iterations", type=int, default=1,
                        help="CNN recursive iterations for prediction")

    # Output
    parser.add_argument("--visualize", action="store_true",
                        help="Visualize grids in terminal")
    parser.add_argument("--skip-trm-training", action="store_true",
                        help="Skip TRM training (requires --trm-checkpoint)")
    parser.add_argument("--skip-cnn-training", action="store_true",
                        help="Skip CNN training (requires --cnn-checkpoint)")

    args = parser.parse_args()

    print("=" * 60)
    print("TRM + CNN UNIFIED PIPELINE")
    print("=" * 60)
    print(f"Puzzle: {args.puzzle}")
    print(f"Device: {DEVICE}")
    print(f"Dtype: {DTYPE}")
    print()

    # Set default checkpoint paths
    if args.trm_checkpoint_dir is None:
        args.trm_checkpoint_dir = f"checkpoints/puzzle_{args.puzzle}"

    cnn_checkpoint_path = args.cnn_checkpoint or f"checkpoints/cnn_{args.puzzle}.pt"

    # =========================================================================
    # Step 1: Train TRM (or load checkpoint)
    # =========================================================================

    if args.trm_checkpoint and os.path.exists(args.trm_checkpoint):
        trm_checkpoint_path = args.trm_checkpoint
        print(f"\nUsing existing TRM checkpoint: {trm_checkpoint_path}")
    elif args.skip_trm_training:
        raise ValueError("--skip-trm-training requires --trm-checkpoint")
    else:
        trm_checkpoint_path = train_trm(
            puzzle_id=args.puzzle,
            epochs=args.trm_epochs,
            checkpoint_dir=args.trm_checkpoint_dir,
            hidden_size=args.hidden_size,
            num_heads=args.num_heads,
            num_layers=args.num_layers,
            H_cycles=args.H_cycles,
            L_cycles=args.L_cycles,
            batch_size=args.batch_size,
            lr=args.lr,
            dynamic_iterations=args.dynamic_iterations,
            dynamic_max_steps=args.dynamic_max_steps,
            use_arc_eval=args.use_arc_eval,
        )

    # =========================================================================
    # Step 2: Train CNN (or load checkpoint)
    # =========================================================================

    if args.cnn_checkpoint and os.path.exists(args.cnn_checkpoint):
        cnn_checkpoint_path = args.cnn_checkpoint
        print(f"\nUsing existing CNN checkpoint: {cnn_checkpoint_path}")
    elif args.skip_cnn_training:
        raise ValueError("--skip-cnn-training requires --cnn-checkpoint")
    else:
        cnn_checkpoint_path = train_cnn(
            puzzle_id=args.puzzle,
            epochs=args.cnn_epochs,
            checkpoint_path=cnn_checkpoint_path,
            hidden_dim=args.cnn_hidden_dim,
        )

    # =========================================================================
    # Step 3: Load Models
    # =========================================================================

    print("\n" + "=" * 60)
    print("STEP 3: Loading Models")
    print("=" * 60)

    trm_model, config_dict = load_trm_model(trm_checkpoint_path, DEVICE)
    cnn_model = load_cnn_model(cnn_checkpoint_path, DEVICE)

    # =========================================================================
    # Step 4: Load Puzzle and Get Test Example
    # =========================================================================

    print("\n" + "=" * 60)
    print("STEP 4: Loading Puzzle")
    print("=" * 60)

    puzzle = load_puzzle(args.puzzle, args.data_root)

    test_example = puzzle["test"][0]
    input_grid = np.array(test_example["input"], dtype=np.uint8)

    if "output" not in test_example:
        print("WARNING: Test example has no ground truth output!")
        print("Cannot compute accuracy.")
        ground_truth = None
    else:
        ground_truth = np.array(test_example["output"], dtype=np.uint8)

    print(f"  Input shape: {input_grid.shape}")
    if ground_truth is not None:
        print(f"  Output shape: {ground_truth.shape}")

    # =========================================================================
    # Step 5: Get CNN Prediction
    # =========================================================================

    print("\n" + "=" * 60)
    print("STEP 5: Getting CNN Prediction")
    print("=" * 60)

    cnn_prediction = get_cnn_prediction(
        cnn_model, input_grid, DEVICE,
        num_iterations=args.cnn_iterations,
    )

    if ground_truth is not None:
        cnn_metrics = compute_accuracy(cnn_prediction, ground_truth)
        print(f"  CNN-only accuracy: {cnn_metrics['pixel_accuracy']:.2%} "
              f"({cnn_metrics['correct_pixels']}/{cnn_metrics['total_pixels']})")
        print(f"  CNN exact match: {'YES' if cnn_metrics['exact_match'] else 'NO'}")

    # =========================================================================
    # Step 6: Initialize TRM with CNN Embeddings
    # =========================================================================

    print("\n" + "=" * 60)
    print("STEP 6: Initializing TRM with CNN Embeddings")
    print("=" * 60)

    puzzle_identifier = 0  # First puzzle in single-puzzle dataset

    inner_carry = initialize_trm_with_cnn(
        trm_model, cnn_prediction, puzzle_identifier, config_dict, DEVICE
    )

    print("  Initialized z_H from CNN prediction embeddings")
    print(f"  z_H shape: {inner_carry.z_H.shape}")

    # =========================================================================
    # Step 7: Run TRM Forward Pass
    # =========================================================================

    print("\n" + "=" * 60)
    print("STEP 7: Running TRM Forward Pass")
    print("=" * 60)

    trm_prediction = run_trm_forward(
        trm_model, inner_carry, input_grid, puzzle_identifier, config_dict, DEVICE
    )

    if ground_truth is not None:
        trm_metrics = compute_accuracy(trm_prediction, ground_truth)
        print(f"  TRM+CNN accuracy: {trm_metrics['pixel_accuracy']:.2%} "
              f"({trm_metrics['correct_pixels']}/{trm_metrics['total_pixels']})")
        print(f"  TRM exact match: {'YES' if trm_metrics['exact_match'] else 'NO'}")

    # =========================================================================
    # Final Results
    # =========================================================================

    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)

    if ground_truth is not None:
        print(f"\nPuzzle: {args.puzzle}")
        print(f"Grid size: {ground_truth.shape[0]}x{ground_truth.shape[1]} = {ground_truth.shape[0] * ground_truth.shape[1]} pixels")
        print()
        print(f"CNN-only:  {cnn_metrics['pixel_accuracy']:>6.2%}  ({cnn_metrics['correct_pixels']:>3}/{cnn_metrics['total_pixels']:>3} pixels)  "
              f"{'EXACT MATCH!' if cnn_metrics['exact_match'] else ''}")
        print(f"TRM+CNN:   {trm_metrics['pixel_accuracy']:>6.2%}  ({trm_metrics['correct_pixels']:>3}/{trm_metrics['total_pixels']:>3} pixels)  "
              f"{'EXACT MATCH!' if trm_metrics['exact_match'] else ''}")

        # Show improvement
        improvement = trm_metrics['pixel_accuracy'] - cnn_metrics['pixel_accuracy']
        if improvement > 0:
            print(f"\nTRM improved CNN by: +{improvement:.2%}")
        elif improvement < 0:
            print(f"\nTRM degraded CNN by: {improvement:.2%}")
        else:
            print(f"\nNo change from TRM refinement")
    else:
        print("No ground truth available for accuracy computation")

    # Visualization
    if args.visualize and ground_truth is not None:
        visualize_comparison(input_grid, ground_truth, cnn_prediction, trm_prediction)

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)

    # Return results for programmatic use
    return {
        "puzzle_id": args.puzzle,
        "cnn_metrics": cnn_metrics if ground_truth is not None else None,
        "trm_metrics": trm_metrics if ground_truth is not None else None,
        "cnn_prediction": cnn_prediction,
        "trm_prediction": trm_prediction,
        "ground_truth": ground_truth,
    }


if __name__ == "__main__":
    main()
