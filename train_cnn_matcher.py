#!/usr/bin/env python3
"""
CNN Input-Output Matcher for ARC Puzzles

This trains a CNN to determine whether a given output grid is the correct
output for a given input grid. The model learns to distinguish correct
input-output pairs from incorrect ones.

Usage:
    python train_cnn_matcher.py --dataset arc-agi-1
    python train_cnn_matcher.py --dataset arc-agi-1 --num-negatives 5
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path
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
NUM_COLORS = 10  # ARC has colors 0-9


# =============================================================================
# Augmentation Utilities (from dataset/common.py)
# =============================================================================

def dihedral_transform(arr: np.ndarray, tid: int) -> np.ndarray:
    """8 dihedral symmetries by rotate, flip and mirror"""
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
    else:
        return arr


def random_color_permutation() -> np.ndarray:
    """Create random color permutation (keeping 0 fixed as background)"""
    mapping = np.concatenate([
        np.array([0], dtype=np.uint8),
        np.random.permutation(np.arange(1, 10, dtype=np.uint8))
    ])
    return mapping


def apply_augmentation(grid: np.ndarray, trans_id: int, color_map: np.ndarray) -> np.ndarray:
    """Apply dihedral transform and color permutation to grid"""
    return dihedral_transform(color_map[grid], trans_id)


# =============================================================================
# CNN Model
# =============================================================================

class GridEncoder(nn.Module):
    """Encodes a single 30x30 grid into a feature vector"""

    def __init__(self, hidden_dim: int = 128):
        super().__init__()

        # Embedding for colors (0-9)
        self.color_embed = nn.Embedding(NUM_COLORS, 16)

        # Conv layers
        self.conv1 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, hidden_dim, kernel_size=3, padding=1)

        self.pool = nn.MaxPool2d(2, 2)
        self.norm1 = nn.BatchNorm2d(32)
        self.norm2 = nn.BatchNorm2d(64)
        self.norm3 = nn.BatchNorm2d(128)
        self.norm4 = nn.BatchNorm2d(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, 30, 30) integer tensor with values 0-9
        Returns:
            (batch, hidden_dim) feature vector
        """
        # Embed colors: (batch, 30, 30) -> (batch, 30, 30, 16)
        x = self.color_embed(x)

        # Reshape to (batch, 16, 30, 30) for conv
        x = x.permute(0, 3, 1, 2).contiguous()

        # Conv blocks with pooling
        x = self.pool(F.relu(self.norm1(self.conv1(x))))  # -> (batch, 32, 15, 15)
        x = self.pool(F.relu(self.norm2(self.conv2(x))))  # -> (batch, 64, 7, 7)
        x = self.pool(F.relu(self.norm3(self.conv3(x))))  # -> (batch, 128, 3, 3)
        x = F.relu(self.norm4(self.conv4(x)))             # -> (batch, hidden_dim, 3, 3)

        # Global average pooling
        x = F.adaptive_avg_pool2d(x, 1)  # -> (batch, hidden_dim, 1, 1)
        x = x.reshape(x.size(0), -1)     # -> (batch, hidden_dim)

        return x


class CNNMatcher(nn.Module):
    """
    CNN that determines if an output grid matches an input grid.

    Takes (input_grid, output_grid) and outputs a probability that
    the output is correct for the input.
    """

    def __init__(self, hidden_dim: int = 128):
        super().__init__()

        # Separate encoders for input and output grids
        self.input_encoder = GridEncoder(hidden_dim)
        self.output_encoder = GridEncoder(hidden_dim)

        # Matching network
        # Takes: concat(input_features, output_features, element-wise product, element-wise diff)
        self.matcher = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, input_grid: torch.Tensor, output_grid: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_grid: (batch, 30, 30) integer tensor
            output_grid: (batch, 30, 30) integer tensor
        Returns:
            (batch,) logits indicating match probability
        """
        # Encode both grids
        input_features = self.input_encoder(input_grid)
        output_features = self.output_encoder(output_grid)

        # Combine features in multiple ways
        concat = torch.cat([input_features, output_features], dim=1)
        product = input_features * output_features
        diff = torch.abs(input_features - output_features)

        combined = torch.cat([concat, product, diff], dim=1)

        # Get match score
        logits = self.matcher(combined).squeeze(-1)
        return logits

    def predict_proba(self, input_grid: torch.Tensor, output_grid: torch.Tensor) -> torch.Tensor:
        """Get probability of match"""
        return torch.sigmoid(self.forward(input_grid, output_grid))


# =============================================================================
# Dataset
# =============================================================================

class ARCMatcherDataset(Dataset):
    """
    Dataset for training the CNN matcher.

    For each puzzle example, creates:
    - 1 positive sample (correct input-output pair)
    - N negative samples (wrong outputs)

    Negative outputs are generated by:
    1. Taking outputs from other puzzles
    2. Applying wrong augmentations to the correct output
    """

    def __init__(
        self,
        puzzles: Dict,
        num_negatives: int = 4,
        augment: bool = True,
    ):
        self.num_negatives = num_negatives
        self.augment = augment

        # Extract all (input, output) pairs
        self.examples = []  # List of (input_grid, output_grid, puzzle_id)
        self.outputs_by_puzzle = {}  # puzzle_id -> list of outputs

        for puzzle_id, puzzle in puzzles.items():
            puzzle_outputs = []
            for example in puzzle.get("train", []):
                inp = np.array(example["input"], dtype=np.uint8)
                out = np.array(example["output"], dtype=np.uint8)
                self.examples.append((inp, out, puzzle_id))
                puzzle_outputs.append(out)
            for example in puzzle.get("test", []):
                if "output" in example:
                    inp = np.array(example["input"], dtype=np.uint8)
                    out = np.array(example["output"], dtype=np.uint8)
                    self.examples.append((inp, out, puzzle_id))
                    puzzle_outputs.append(out)

            self.outputs_by_puzzle[puzzle_id] = puzzle_outputs

        # Flatten all outputs for random negative sampling
        self.all_outputs = []
        for outputs in self.outputs_by_puzzle.values():
            self.all_outputs.extend(outputs)

        print(f"Loaded {len(self.examples)} examples from {len(puzzles)} puzzles")

    def __len__(self):
        return len(self.examples) * (1 + self.num_negatives)

    def _pad_grid(self, grid: np.ndarray) -> np.ndarray:
        """Pad grid to 30x30"""
        h, w = grid.shape
        padded = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.uint8)
        padded[:h, :w] = grid
        return padded

    def _generate_negative(self, correct_output: np.ndarray, puzzle_id: str) -> np.ndarray:
        """Generate a negative (wrong) output"""
        # If only one puzzle, skip "other_puzzle" strategy
        if len(self.outputs_by_puzzle) == 1:
            strategy = random.choice(["wrong_augment", "random_noise"])
        else:
            strategy = random.choice(["other_puzzle", "wrong_augment", "random_noise"])

        if strategy == "other_puzzle":
            # Use output from a different puzzle
            other_output = random.choice(self.all_outputs)
            # Make sure it's actually different
            while np.array_equal(other_output, correct_output):
                other_output = random.choice(self.all_outputs)
            return other_output

        elif strategy == "wrong_augment":
            # Apply a different augmentation to the correct output
            # This creates a plausible-looking but incorrect output
            trans_id = random.randint(1, 7)  # Skip identity (0)
            color_map = random_color_permutation()
            return apply_augmentation(correct_output, trans_id, color_map)

        else:  # random_noise
            # Add random noise/modifications to the correct output
            noisy = correct_output.copy()
            num_changes = random.randint(1, max(1, noisy.size // 10))
            for _ in range(num_changes):
                r = random.randint(0, noisy.shape[0] - 1)
                c = random.randint(0, noisy.shape[1] - 1)
                noisy[r, c] = random.randint(0, 9)
            return noisy

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            input_grid: (30, 30) tensor
            output_grid: (30, 30) tensor
            label: scalar (1 for correct match, 0 for incorrect)
        """
        # Determine which example and whether positive/negative
        example_idx = idx // (1 + self.num_negatives)
        sample_type = idx % (1 + self.num_negatives)

        input_grid, correct_output, puzzle_id = self.examples[example_idx]

        if sample_type == 0:
            # Positive sample
            output_grid = correct_output
            label = 1.0
        else:
            # Negative sample
            output_grid = self._generate_negative(correct_output, puzzle_id)
            label = 0.0

        # Apply augmentation to both grids (same augmentation)
        if self.augment and random.random() < 0.5:
            trans_id = random.randint(0, 7)
            color_map = random_color_permutation()
            input_grid = apply_augmentation(input_grid, trans_id, color_map)
            output_grid = apply_augmentation(output_grid, trans_id, color_map)

        # Pad to 30x30
        input_grid = self._pad_grid(input_grid)
        output_grid = self._pad_grid(output_grid)

        # Ensure contiguous arrays (dihedral transforms can create non-contiguous views)
        input_grid = np.ascontiguousarray(input_grid)
        output_grid = np.ascontiguousarray(output_grid)

        return (
            torch.from_numpy(input_grid.copy()).long(),
            torch.from_numpy(output_grid.copy()).long(),
            torch.tensor(label, dtype=torch.float32)
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
    """Train for one epoch"""
    model.train()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    pbar = tqdm(loader, desc="Training")
    for input_grid, output_grid, labels in pbar:
        input_grid = input_grid.to(device)
        output_grid = output_grid.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        logits = model(input_grid, output_grid)
        loss = F.binary_cross_entropy_with_logits(logits, labels)

        loss.backward()
        optimizer.step()

        # Metrics
        total_loss += loss.item() * labels.size(0)
        preds = (logits > 0).float()
        total_correct += (preds == labels).sum().item()
        total_samples += labels.size(0)

        pbar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "acc": f"{total_correct/total_samples:.2%}"
        })

    return {
        "loss": total_loss / total_samples,
        "accuracy": total_correct / total_samples,
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    """Evaluate the model"""
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    # Track true/false positives/negatives
    tp = fp = tn = fn = 0

    for input_grid, output_grid, labels in loader:
        input_grid = input_grid.to(device)
        output_grid = output_grid.to(device)
        labels = labels.to(device)

        logits = model(input_grid, output_grid)
        loss = F.binary_cross_entropy_with_logits(logits, labels)

        total_loss += loss.item() * labels.size(0)
        preds = (logits > 0).float()
        total_correct += (preds == labels).sum().item()
        total_samples += labels.size(0)

        # Confusion matrix
        tp += ((preds == 1) & (labels == 1)).sum().item()
        fp += ((preds == 1) & (labels == 0)).sum().item()
        tn += ((preds == 0) & (labels == 0)).sum().item()
        fn += ((preds == 0) & (labels == 1)).sum().item()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {
        "loss": total_loss / total_samples,
        "accuracy": total_correct / total_samples,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def demo_predictions(
    model: nn.Module,
    dataset: ARCMatcherDataset,
    device: torch.device,
    num_samples: int = 5,
):
    """Show some sample predictions"""
    model.eval()

    print("\n" + "="*60)
    print("Sample Predictions")
    print("="*60)

    indices = random.sample(range(len(dataset)), min(num_samples * 2, len(dataset)))

    for idx in indices[:num_samples]:
        input_grid, output_grid, label = dataset[idx]
        input_grid = input_grid.unsqueeze(0).to(device)
        output_grid = output_grid.unsqueeze(0).to(device)

        with torch.no_grad():
            prob = model.predict_proba(input_grid, output_grid).item()

        status = "CORRECT" if label.item() == 1 else "WRONG"
        pred_status = "Match" if prob > 0.5 else "No Match"
        correct = (prob > 0.5) == (label.item() == 1)

        print(f"\nSample (actual: {status})")
        print(f"  Predicted: {pred_status} (prob={prob:.3f}) {'[OK]' if correct else '[WRONG]'}")


# =============================================================================
# Main
# =============================================================================

def load_puzzles(dataset_name: str, data_root: str = "kaggle/combined") -> Dict:
    """Load puzzle data from JSON files"""
    config = {
        "arc-agi-1": {
            "subsets": ["training", "evaluation"],
        },
        "arc-agi-2": {
            "subsets": ["training2", "evaluation2"],
        },
    }

    if dataset_name not in config:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    all_puzzles = {}

    for subset in config[dataset_name]["subsets"]:
        challenges_path = f"{data_root}/arc-agi_{subset}_challenges.json"
        solutions_path = f"{data_root}/arc-agi_{subset}_solutions.json"

        if not os.path.exists(challenges_path):
            print(f"Warning: {challenges_path} not found")
            continue

        with open(challenges_path) as f:
            puzzles = json.load(f)

        # Load solutions if available
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
    parser = argparse.ArgumentParser(description="Train CNN Matcher for ARC")

    # Dataset
    parser.add_argument("--dataset", type=str, default="arc-agi-1",
                        choices=["arc-agi-1", "arc-agi-2"])
    parser.add_argument("--data-root", type=str, default="kaggle/combined")
    parser.add_argument("--single-puzzle", type=str, default=None,
                        help="Train on only this puzzle ID (e.g., '00d62c1b')")

    # Model
    parser.add_argument("--hidden-dim", type=int, default=128,
                        help="Hidden dimension for CNN encoder")

    # Training
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num-negatives", type=int, default=4,
                        help="Number of negative samples per positive")
    parser.add_argument("--val-split", type=float, default=0.1,
                        help="Fraction of data for validation")

    # Checkpointing
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints_matcher")
    parser.add_argument("--save-every", type=int, default=10,
                        help="Save checkpoint every N epochs")

    # Misc
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)

    args = parser.parse_args()

    # Set seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    print(f"Device: {DEVICE}")
    print(f"Dataset: {args.dataset}")

    # Load puzzles
    print("\nLoading puzzles...")
    puzzles = load_puzzles(args.dataset, args.data_root)
    print(f"Loaded {len(puzzles)} puzzles")

    # Filter to single puzzle if specified
    if args.single_puzzle:
        if args.single_puzzle not in puzzles:
            print(f"Error: Puzzle '{args.single_puzzle}' not found!")
            print(f"Available puzzles (first 10): {list(puzzles.keys())[:10]}")
            return
        puzzles = {args.single_puzzle: puzzles[args.single_puzzle]}
        print(f"Single puzzle mode: {args.single_puzzle}")

    # Split into train/val
    puzzle_ids = list(puzzles.keys())
    random.shuffle(puzzle_ids)

    # For single puzzle mode, use same puzzle for train and val
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
    train_dataset = ARCMatcherDataset(
        train_puzzles,
        num_negatives=args.num_negatives,
        augment=True,
    )
    val_dataset = ARCMatcherDataset(
        val_puzzles,
        num_negatives=args.num_negatives,
        augment=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # Create model
    print("\nCreating model...")
    model = CNNMatcher(hidden_dim=args.hidden_dim)
    model = model.to(DEVICE)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)

    # Checkpoint dir
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # Training loop
    print("\n" + "="*60)
    print("Starting Training")
    print("="*60)

    best_val_f1 = 0.0

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")

        # Train
        train_metrics = train_epoch(model, train_loader, optimizer, DEVICE)

        # Evaluate
        val_metrics = evaluate(model, val_loader, DEVICE)

        scheduler.step()

        # Print metrics
        print(f"  Train Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.2%}")
        print(f"  Val Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.2%}")
        print(f"  Val Precision: {val_metrics['precision']:.2%}, Recall: {val_metrics['recall']:.2%}, F1: {val_metrics['f1']:.2%}")

        # Save best model
        if val_metrics['f1'] > best_val_f1:
            best_val_f1 = val_metrics['f1']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_metrics': val_metrics,
            }, os.path.join(args.checkpoint_dir, 'best_model.pt'))
            print(f"  [New best model saved! F1: {best_val_f1:.2%}]")

        # Periodic checkpoint
        if (epoch + 1) % args.save_every == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, os.path.join(args.checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pt'))

    # Final evaluation
    print("\n" + "="*60)
    print("Final Evaluation")
    print("="*60)

    # Load best model
    checkpoint = torch.load(os.path.join(args.checkpoint_dir, 'best_model.pt'))
    model.load_state_dict(checkpoint['model_state_dict'])

    final_metrics = evaluate(model, val_loader, DEVICE)
    print(f"Best Model Metrics:")
    print(f"  Accuracy: {final_metrics['accuracy']:.2%}")
    print(f"  Precision: {final_metrics['precision']:.2%}")
    print(f"  Recall: {final_metrics['recall']:.2%}")
    print(f"  F1: {final_metrics['f1']:.2%}")

    # Demo predictions
    demo_predictions(model, val_dataset, DEVICE)

    print("\nTraining complete!")


if __name__ == "__main__":
    main()
