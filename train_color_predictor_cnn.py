#!/usr/bin/env python3
"""
Color Predictor CNN - Predicts which colors appear in the output grid

This CNN takes an input grid and predicts which of the 10 colors (0-9) will
appear in the corresponding output grid. It's trained on augmented pairs from
a single puzzle (or multiple puzzles).

Task: Multi-label classification (10 binary predictions, one per color)

Usage:
    python train_color_predictor_cnn.py --single-puzzle 00d62c1b
    python train_color_predictor_cnn.py --dataset arc-agi-1
"""

import argparse
import json
import os
import random
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# Determine device
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

GRID_SIZE = 30
NUM_COLORS = 10


# =============================================================================
# Augmentation Utilities (same as pixel error CNN)
# =============================================================================

def dihedral_transform(arr: np.ndarray, tid: int) -> np.ndarray:
    """8 dihedral symmetries"""
    if tid == 0:
        return arr
    elif tid == 1:
        return np.rot90(arr, k=1)
    elif tid == 2:
        return np.rot90(arr, k=2)
    elif tid == 3:
        return np.rot90(arr, k=3)
    elif tid == 4:
        return np.fliplr(arr)
    elif tid == 5:
        return np.flipud(arr)
    elif tid == 6:
        return arr.T
    elif tid == 7:
        return np.fliplr(np.rot90(arr, k=1))
    return arr


def random_color_permutation() -> np.ndarray:
    """Create random color permutation (keeping 0 fixed)"""
    mapping = np.concatenate([
        np.array([0], dtype=np.uint8),
        np.random.permutation(np.arange(1, 10, dtype=np.uint8))
    ])
    return mapping


def apply_augmentation(grid: np.ndarray, trans_id: int, color_map: np.ndarray) -> np.ndarray:
    """Apply dihedral transform and color permutation"""
    return dihedral_transform(color_map[grid], trans_id)


def get_random_augmentation() -> Tuple[int, np.ndarray]:
    """Get random augmentation parameters"""
    trans_id = random.randint(0, 7)
    color_map = random_color_permutation()
    return trans_id, color_map


# =============================================================================
# Model Architecture
# =============================================================================

class DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class Down(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.pool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_ch, out_ch)
        )

    def forward(self, x):
        return self.pool_conv(x)


class ColorPredictorCNN(nn.Module):
    """
    CNN that predicts which colors (0-9) appear in the output grid
    given only the input grid.

    Architecture:
    - Color embedding for input grid
    - Encoder (downsampling convolutions)
    - Global pooling
    - MLP classifier -> 10 binary predictions
    """

    def __init__(self, hidden_dim: int = 64, embed_dim: int = 16):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.embed_dim = embed_dim

        # Color embedding
        self.color_embed = nn.Embedding(NUM_COLORS, embed_dim)

        # Encoder
        self.inc = DoubleConv(embed_dim, hidden_dim)
        self.down1 = Down(hidden_dim, hidden_dim * 2)
        self.down2 = Down(hidden_dim * 2, hidden_dim * 4)
        self.down3 = Down(hidden_dim * 4, hidden_dim * 8)
        self.down4 = Down(hidden_dim * 8, hidden_dim * 8)

        # Global pooling + classifier
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 8, hidden_dim * 4),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim * 4, hidden_dim * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim * 2, NUM_COLORS),
        )

    def forward(self, input_grid: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_grid: (B, 30, 30) tensor of color indices (0-9)

        Returns:
            logits: (B, 10) tensor of logits for each color
        """
        # Embed colors
        x = self.color_embed(input_grid)  # (B, 30, 30, embed_dim)
        x = x.permute(0, 3, 1, 2).contiguous()  # (B, embed_dim, 30, 30)

        # Encode
        x = self.inc(x)
        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)
        x = self.down4(x)

        # Global pool and classify
        x = self.global_pool(x)  # (B, hidden_dim*8, 1, 1)
        x = x.view(x.size(0), -1)  # (B, hidden_dim*8)
        logits = self.classifier(x)  # (B, 10)

        return logits

    def predict_proba(self, input_grid: torch.Tensor) -> torch.Tensor:
        """Return probabilities instead of logits"""
        return torch.sigmoid(self.forward(input_grid))

    def predict_colors(self, input_grid: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """Return predicted color presence (binary)"""
        return (self.predict_proba(input_grid) > threshold).float()

    @classmethod
    def from_checkpoint(cls, checkpoint_path: str, device: torch.device = None):
        checkpoint = torch.load(checkpoint_path, map_location=device or 'cpu', weights_only=False)
        args = checkpoint.get('args', {})
        hidden_dim = args.get('hidden_dim', 64)
        embed_dim = args.get('embed_dim', 16)

        model = cls(hidden_dim=hidden_dim, embed_dim=embed_dim)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

        if device:
            model = model.to(device)
        return model


# =============================================================================
# Dataset
# =============================================================================

class ColorPredictorDataset(Dataset):
    """
    Dataset for training color prediction CNN.

    For each input-output pair, extracts:
    - Input grid (augmented)
    - Binary vector indicating which colors are present in output grid
    """

    def __init__(
        self,
        puzzles: Dict,
        num_augmentations: int = 100,
        augment: bool = True,
    ):
        self.num_augmentations = num_augmentations
        self.augment = augment

        # Extract all (input, output) pairs
        self.examples = []

        for puzzle_id, puzzle in puzzles.items():
            for example in puzzle.get("train", []):
                inp = np.array(example["input"], dtype=np.uint8)
                out = np.array(example["output"], dtype=np.uint8)
                self.examples.append((inp, out, puzzle_id))
            for example in puzzle.get("test", []):
                if "output" in example:
                    inp = np.array(example["input"], dtype=np.uint8)
                    out = np.array(example["output"], dtype=np.uint8)
                    self.examples.append((inp, out, puzzle_id))

        print(f"Loaded {len(self.examples)} examples from {len(puzzles)} puzzles")
        print(f"Augmentations per example: {num_augmentations}")
        print(f"Total samples: {len(self.examples) * num_augmentations}")

    def __len__(self):
        return len(self.examples) * self.num_augmentations

    def _pad_grid(self, grid: np.ndarray) -> np.ndarray:
        h, w = grid.shape
        padded = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.uint8)
        padded[:h, :w] = grid
        return padded

    def _get_color_vector(self, grid: np.ndarray) -> np.ndarray:
        """Get binary vector indicating which colors are present"""
        colors_present = np.zeros(NUM_COLORS, dtype=np.float32)
        unique_colors = np.unique(grid)
        colors_present[unique_colors] = 1.0
        return colors_present

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        example_idx = idx // self.num_augmentations
        input_grid, output_grid, puzzle_id = self.examples[example_idx]

        if self.augment:
            # Apply same augmentation to both input and output
            trans_id, color_map = get_random_augmentation()
            aug_input = apply_augmentation(input_grid, trans_id, color_map)
            aug_output = apply_augmentation(output_grid, trans_id, color_map)
        else:
            aug_input = input_grid
            aug_output = output_grid

        # Pad input grid
        padded_input = self._pad_grid(aug_input)

        # Get color presence vector from output (before padding, since 0 might be added by padding)
        color_vector = self._get_color_vector(aug_output)

        # Ensure contiguous
        padded_input = np.ascontiguousarray(padded_input)

        return (
            torch.from_numpy(padded_input.copy()).long(),
            torch.from_numpy(color_vector.copy()).float()
        )


# =============================================================================
# Training
# =============================================================================

def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Dict[str, float]:
    model.train()

    total_loss = 0.0
    total_correct = 0
    total_predictions = 0
    total_tp = np.zeros(NUM_COLORS)
    total_fp = np.zeros(NUM_COLORS)
    total_fn = np.zeros(NUM_COLORS)
    num_batches = 0

    pbar = tqdm(loader, desc="Training")
    for input_grid, color_target in pbar:
        input_grid = input_grid.to(device)
        color_target = color_target.to(device)

        optimizer.zero_grad()

        logits = model(input_grid)
        loss = F.binary_cross_entropy_with_logits(logits, color_target)

        loss.backward()
        optimizer.step()

        with torch.no_grad():
            preds = (logits > 0).float()

            # Overall accuracy (per-label)
            correct = (preds == color_target).sum().item()
            total = color_target.numel()
            total_correct += correct
            total_predictions += total

            # Per-color metrics
            for c in range(NUM_COLORS):
                target_c = color_target[:, c]
                pred_c = preds[:, c]
                total_tp[c] += ((target_c == 1) & (pred_c == 1)).sum().item()
                total_fp[c] += ((target_c == 0) & (pred_c == 1)).sum().item()
                total_fn[c] += ((target_c == 1) & (pred_c == 0)).sum().item()

            total_loss += loss.item()
            num_batches += 1

        pbar.set_postfix(loss=loss.item())

    # Compute metrics
    accuracy = total_correct / max(total_predictions, 1)

    # Macro-averaged precision, recall, F1
    precisions = total_tp / np.maximum(total_tp + total_fp, 1)
    recalls = total_tp / np.maximum(total_tp + total_fn, 1)
    f1s = 2 * precisions * recalls / np.maximum(precisions + recalls, 1e-8)

    return {
        "loss": total_loss / max(num_batches, 1),
        "accuracy": accuracy,
        "macro_precision": precisions.mean(),
        "macro_recall": recalls.mean(),
        "macro_f1": f1s.mean(),
    }


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_predictions = 0
    total_tp = np.zeros(NUM_COLORS)
    total_fp = np.zeros(NUM_COLORS)
    total_fn = np.zeros(NUM_COLORS)
    total_perfect = 0
    total_samples = 0
    num_batches = 0

    with torch.no_grad():
        for input_grid, color_target in loader:
            input_grid = input_grid.to(device)
            color_target = color_target.to(device)

            logits = model(input_grid)
            loss = F.binary_cross_entropy_with_logits(logits, color_target)

            preds = (logits > 0).float()

            # Overall accuracy
            correct = (preds == color_target).sum().item()
            total = color_target.numel()
            total_correct += correct
            total_predictions += total

            # Per-color metrics
            for c in range(NUM_COLORS):
                target_c = color_target[:, c]
                pred_c = preds[:, c]
                total_tp[c] += ((target_c == 1) & (pred_c == 1)).sum().item()
                total_fp[c] += ((target_c == 0) & (pred_c == 1)).sum().item()
                total_fn[c] += ((target_c == 1) & (pred_c == 0)).sum().item()

            # Perfect predictions (all 10 colors correct)
            batch_perfect = (preds == color_target).all(dim=1).sum().item()
            total_perfect += batch_perfect
            total_samples += input_grid.size(0)

            total_loss += loss.item()
            num_batches += 1

    accuracy = total_correct / max(total_predictions, 1)
    perfect_rate = total_perfect / max(total_samples, 1)

    precisions = total_tp / np.maximum(total_tp + total_fp, 1)
    recalls = total_tp / np.maximum(total_tp + total_fn, 1)
    f1s = 2 * precisions * recalls / np.maximum(precisions + recalls, 1e-8)

    return {
        "loss": total_loss / max(num_batches, 1),
        "accuracy": accuracy,
        "perfect_rate": perfect_rate,
        "macro_precision": precisions.mean(),
        "macro_recall": recalls.mean(),
        "macro_f1": f1s.mean(),
    }


def visualize_predictions(model: nn.Module, dataset: Dataset, device: torch.device, num_samples: int = 8):
    """Visualize color predictions"""
    model.eval()

    print("\n" + "="*80)
    print("Sample Color Predictions")
    print("="*80)

    indices = random.sample(range(len(dataset)), min(num_samples, len(dataset)))

    for idx in indices:
        input_grid, color_target = dataset[idx]

        inp_t = input_grid.unsqueeze(0).to(device)

        with torch.no_grad():
            pred_proba = model.predict_proba(inp_t)[0].cpu().numpy()
            pred_colors = (pred_proba > 0.5).astype(int)

        target_colors = color_target.numpy().astype(int)

        # Find actual grid content bounds
        input_np = input_grid.numpy()
        nonzero = np.where(input_np > 0)
        if len(nonzero[0]) > 0:
            r_max = min(nonzero[0].max() + 1, 8)
            c_max = min(nonzero[1].max() + 1, 8)
        else:
            r_max, c_max = 6, 6

        print(f"\n{'─'*80}")
        print("Input Grid:")
        for r in range(r_max):
            row = " ".join(f"{input_np[r, c]}" for c in range(c_max))
            print(f"  {row}")

        print(f"\nTarget colors:    {' '.join(str(c) for c in range(10))}")
        print(f"                  {' '.join(str(t) for t in target_colors)}")
        print(f"Predicted colors: {' '.join(str(p) for p in pred_colors)}")
        print(f"Confidence:       {' '.join(f'{p:.1f}' for p in pred_proba)}")

        # Show which predictions are correct/wrong
        correct = ["✓" if p == t else "✗" for p, t in zip(pred_colors, target_colors)]
        print(f"Correct:          {' '.join(correct)}")

        accuracy = (pred_colors == target_colors).mean()
        print(f"Accuracy: {accuracy:.0%}")

    print("="*80 + "\n")


# =============================================================================
# Main
# =============================================================================

def load_puzzles(dataset_name: str, data_root: str = "kaggle/combined") -> Dict:
    config = {
        "arc-agi-1": {"subsets": ["training", "evaluation"]},
        "arc-agi-2": {"subsets": ["training2", "evaluation2"]},
    }

    all_puzzles = {}

    for subset in config[dataset_name]["subsets"]:
        challenges_path = f"{data_root}/arc-agi_{subset}_challenges.json"
        solutions_path = f"{data_root}/arc-agi_{subset}_solutions.json"

        if not os.path.exists(challenges_path):
            print(f"Warning: {challenges_path} not found")
            continue

        with open(challenges_path) as f:
            puzzles = json.load(f)

        if os.path.exists(solutions_path):
            with open(solutions_path) as f:
                solutions = json.load(f)
            for puzzle_id in puzzles:
                if puzzle_id in solutions:
                    for i, sol in enumerate(solutions[puzzle_id]):
                        if i < len(puzzles[puzzle_id]["test"]):
                            puzzles[puzzle_id]["test"][i]["output"] = sol

        all_puzzles.update(puzzles)

    return all_puzzles


def main():
    parser = argparse.ArgumentParser(description="Train Color Predictor CNN")

    parser.add_argument("--dataset", type=str, default="arc-agi-1",
                        choices=["arc-agi-1", "arc-agi-2"])
    parser.add_argument("--data-root", type=str, default="kaggle/combined")
    parser.add_argument("--single-puzzle", type=str, default=None,
                        help="Train on only this puzzle ID")

    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--embed-dim", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num-augmentations", type=int, default=100,
                        help="Number of augmentations per example")
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--save-path", type=str, default="checkpoints/color_predictor_cnn.pt")

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    print(f"Device: {DEVICE}")
    print(f"Dataset: {args.dataset}")

    # Load puzzles
    print("\nLoading puzzles...")
    puzzles = load_puzzles(args.dataset, args.data_root)
    print(f"Loaded {len(puzzles)} puzzles")

    if args.single_puzzle:
        if args.single_puzzle not in puzzles:
            print(f"Error: Puzzle '{args.single_puzzle}' not found!")
            return
        puzzles = {args.single_puzzle: puzzles[args.single_puzzle]}
        print(f"Single puzzle mode: {args.single_puzzle}")

    # Split
    puzzle_ids = list(puzzles.keys())
    random.shuffle(puzzle_ids)

    if args.single_puzzle:
        train_ids = puzzle_ids
        val_ids = puzzle_ids
    else:
        split_idx = int(len(puzzle_ids) * (1 - args.val_split))
        train_ids = puzzle_ids[:split_idx]
        val_ids = puzzle_ids[split_idx:]

    train_puzzles = {pid: puzzles[pid] for pid in train_ids}
    val_puzzles = {pid: puzzles[pid] for pid in val_ids}

    print(f"Train puzzles: {len(train_puzzles)}, Val puzzles: {len(val_puzzles)}")

    # Create datasets
    train_dataset = ColorPredictorDataset(
        train_puzzles,
        num_augmentations=args.num_augmentations,
        augment=True,
    )
    val_dataset = ColorPredictorDataset(
        val_puzzles,
        num_augmentations=args.num_augmentations,
        augment=True,
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # Create model
    print("\nCreating model...")
    model = ColorPredictorCNN(hidden_dim=args.hidden_dim, embed_dim=args.embed_dim)
    model = model.to(DEVICE)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)

    # Create checkpoint directory
    os.makedirs(os.path.dirname(args.save_path) or ".", exist_ok=True)

    # Training
    print("\n" + "="*60)
    print("Starting Training")
    print("="*60)

    best_val_f1 = 0.0

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")

        train_metrics = train_epoch(model, train_loader, optimizer, DEVICE)
        val_metrics = evaluate(model, val_loader, DEVICE)

        scheduler.step()

        print(f"  Train Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.2%}, "
              f"F1: {train_metrics['macro_f1']:.2%}")
        print(f"  Val   Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.2%}, "
              f"F1: {val_metrics['macro_f1']:.2%}, Perfect: {val_metrics['perfect_rate']:.2%}")

        if val_metrics['macro_f1'] > best_val_f1:
            best_val_f1 = val_metrics['macro_f1']
            print(f"  [New best F1: {best_val_f1:.2%}]")

            checkpoint = {
                'model_state_dict': model.state_dict(),
                'args': vars(args),
                'epoch': epoch,
                'best_val_f1': best_val_f1,
            }
            torch.save(checkpoint, args.save_path)

    # Final summary
    print("\n" + "="*60)
    print("Training Complete")
    print("="*60)
    print(f"Best F1: {best_val_f1:.2%}")
    print(f"Checkpoint saved to: {args.save_path}")

    # Visualize
    visualize_predictions(model, val_dataset, DEVICE)

    print("\nDone!")


if __name__ == "__main__":
    main()
