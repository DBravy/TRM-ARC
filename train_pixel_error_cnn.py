#!/usr/bin/env python3
"""
Pixel-Level Error Detection CNN for ARC Puzzles

This trains a CNN to predict which specific pixels in a candidate output
are incorrect. The model outputs a 30x30 heatmap where each pixel indicates
the probability that pixel is correct.

Usage:
    python train_pixel_error_cnn.py --dataset arc-agi-1
    python train_pixel_error_cnn.py --single-puzzle 00d62c1b
"""

import argparse
import json
import os
import random
from typing import Dict, List, Tuple

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
# Augmentation Utilities
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


# =============================================================================
# U-Net Style Model for Pixel-Level Prediction
# =============================================================================

class DoubleConv(nn.Module):
    """Two conv layers with batch norm and ReLU"""
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
    """Downscale with maxpool then double conv"""
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.pool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_ch, out_ch)
        )

    def forward(self, x):
        return self.pool_conv(x)


class Up(nn.Module):
    """Upscale then double conv with skip connection"""
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        # Handle size mismatch
        diff_h = x2.size(2) - x1.size(2)
        diff_w = x2.size(3) - x1.size(3)
        x1 = F.pad(x1, [diff_w // 2, diff_w - diff_w // 2,
                       diff_h // 2, diff_h - diff_h // 2])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class PixelErrorCNN(nn.Module):
    """
    U-Net style CNN that predicts pixel-level correctness.

    Takes (input_grid, candidate_output) and outputs a 30x30 heatmap
    where each pixel indicates P(this pixel is correct).
    """

    def __init__(self, hidden_dim: int = 64):
        super().__init__()

        # Color embeddings for both grids
        self.input_embed = nn.Embedding(NUM_COLORS, 16)
        self.output_embed = nn.Embedding(NUM_COLORS, 16)

        # Combined channels: 16 (input) + 16 (output) = 32
        base_ch = hidden_dim

        # Encoder (downsampling)
        self.inc = DoubleConv(32, base_ch)           # 30x30 -> 30x30
        self.down1 = Down(base_ch, base_ch * 2)      # 30x30 -> 15x15
        self.down2 = Down(base_ch * 2, base_ch * 4)  # 15x15 -> 7x7
        self.down3 = Down(base_ch * 4, base_ch * 8)  # 7x7 -> 3x3

        # Decoder (upsampling with skip connections)
        self.up1 = Up(base_ch * 8, base_ch * 4)      # 3x3 -> 7x7
        self.up2 = Up(base_ch * 4, base_ch * 2)      # 7x7 -> 15x15
        self.up3 = Up(base_ch * 2, base_ch)          # 15x15 -> 30x30

        # Output layer
        self.outc = nn.Conv2d(base_ch, 1, kernel_size=1)

    def forward(self, input_grid: torch.Tensor, output_grid: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_grid: (batch, 30, 30) integer tensor
            output_grid: (batch, 30, 30) integer tensor
        Returns:
            (batch, 30, 30) logits for pixel correctness
        """
        # Embed both grids
        inp_emb = self.input_embed(input_grid)   # (B, 30, 30, 16)
        out_emb = self.output_embed(output_grid) # (B, 30, 30, 16)

        # Concatenate and reshape for conv
        x = torch.cat([inp_emb, out_emb], dim=-1)  # (B, 30, 30, 32)
        x = x.permute(0, 3, 1, 2).contiguous()     # (B, 32, 30, 30)

        # Encoder
        x1 = self.inc(x)     # (B, 64, 30, 30)
        x2 = self.down1(x1)  # (B, 128, 15, 15)
        x3 = self.down2(x2)  # (B, 256, 7, 7)
        x4 = self.down3(x3)  # (B, 512, 3, 3)

        # Decoder with skip connections
        x = self.up1(x4, x3)  # (B, 256, 7, 7)
        x = self.up2(x, x2)   # (B, 128, 15, 15)
        x = self.up3(x, x1)   # (B, 64, 30, 30)

        # Output
        logits = self.outc(x)  # (B, 1, 30, 30)
        return logits.squeeze(1)  # (B, 30, 30)

    def predict_proba(self, input_grid: torch.Tensor, output_grid: torch.Tensor) -> torch.Tensor:
        """Get probability map"""
        return torch.sigmoid(self.forward(input_grid, output_grid))

    @classmethod
    def from_checkpoint(cls, checkpoint_path: str, device: torch.device = None):
        """Load model from checkpoint for inference"""
        checkpoint = torch.load(checkpoint_path, map_location=device or 'cpu', weights_only=False)
        
        hidden_dim = checkpoint.get('args', {}).get('hidden_dim', 64)
        
        model = cls(hidden_dim=hidden_dim)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        if device:
            model = model.to(device)
        
        return model


# =============================================================================
# Dataset
# =============================================================================

class PixelErrorDataset(Dataset):
    """
    Dataset for pixel-level error detection.

    Returns:
        input_grid: (30, 30) the puzzle input
        output_grid: (30, 30) candidate output (may be correct or wrong)
        pixel_mask: (30, 30) binary mask where 1 = correct pixel, 0 = incorrect
        is_positive: scalar, 1 if this is a correct output, 0 if wrong
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
        self.examples = []
        self.outputs_by_puzzle = {}

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

    def _generate_negative(self, correct_output: np.ndarray, puzzle_id: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate a wrong output and the corresponding error mask.

        Returns:
            wrong_output: the incorrect output grid
            error_mask: 1 where pixels match correct_output, 0 where they differ
        """
        if len(self.outputs_by_puzzle) == 1:
            strategy = random.choice(["wrong_augment", "random_noise"])
        else:
            strategy = random.choice(["other_puzzle", "wrong_augment", "random_noise"])

        if strategy == "other_puzzle":
            wrong_output = random.choice(self.all_outputs)
            while np.array_equal(wrong_output, correct_output):
                wrong_output = random.choice(self.all_outputs)

        elif strategy == "wrong_augment":
            trans_id = random.randint(1, 7)
            color_map = random_color_permutation()
            wrong_output = apply_augmentation(correct_output, trans_id, color_map)

        else:  # random_noise
            wrong_output = correct_output.copy()
            num_changes = random.randint(1, max(1, wrong_output.size // 10))
            for _ in range(num_changes):
                r = random.randint(0, wrong_output.shape[0] - 1)
                c = random.randint(0, wrong_output.shape[1] - 1)
                wrong_output[r, c] = random.randint(0, 9)

        # Compute error mask: 1 = correct, 0 = wrong
        # Need to pad both to compare
        padded_correct = self._pad_grid(correct_output)
        padded_wrong = self._pad_grid(wrong_output)
        error_mask = (padded_wrong == padded_correct).astype(np.float32)

        return wrong_output, error_mask

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        example_idx = idx // (1 + self.num_negatives)
        sample_type = idx % (1 + self.num_negatives)

        input_grid, correct_output, puzzle_id = self.examples[example_idx]

        if sample_type == 0:
            # Positive sample - all pixels are correct
            output_grid = correct_output
            pixel_mask = np.ones((GRID_SIZE, GRID_SIZE), dtype=np.float32)
            is_positive = 1.0
        else:
            # Negative sample
            output_grid, pixel_mask = self._generate_negative(correct_output, puzzle_id)
            is_positive = 0.0

        # Apply same augmentation to input and output (and mask)
        if self.augment and random.random() < 0.5:
            trans_id = random.randint(0, 7)
            color_map = random_color_permutation()
            input_grid = apply_augmentation(input_grid, trans_id, color_map)
            output_grid = apply_augmentation(output_grid, trans_id, color_map)
            # Transform mask (no color mapping needed)
            pixel_mask = dihedral_transform(pixel_mask, trans_id)

        # Pad grids
        input_grid = self._pad_grid(input_grid)
        output_grid = self._pad_grid(output_grid)

        # Ensure contiguous
        input_grid = np.ascontiguousarray(input_grid)
        output_grid = np.ascontiguousarray(output_grid)
        pixel_mask = np.ascontiguousarray(pixel_mask)

        return (
            torch.from_numpy(input_grid.copy()).long(),
            torch.from_numpy(output_grid.copy()).long(),
            torch.from_numpy(pixel_mask.copy()).float(),
            torch.tensor(is_positive, dtype=torch.float32)
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
    total_pixel_correct = 0
    total_pixels = 0
    total_error_iou = 0.0
    num_batches = 0

    pbar = tqdm(loader, desc="Training")
    for input_grid, output_grid, pixel_mask, is_positive in pbar:
        input_grid = input_grid.to(device)
        output_grid = output_grid.to(device)
        pixel_mask = pixel_mask.to(device)

        optimizer.zero_grad()

        logits = model(input_grid, output_grid)
        loss = F.binary_cross_entropy_with_logits(logits, pixel_mask)

        loss.backward()
        optimizer.step()

        # Metrics
        with torch.no_grad():
            preds = (logits > 0).float()
            total_loss += loss.item() * input_grid.size(0)
            total_pixel_correct += (preds == pixel_mask).sum().item()
            total_pixels += pixel_mask.numel()

            # IoU for error pixels (where mask == 0)
            error_pred = (preds == 0)
            error_true = (pixel_mask == 0)
            intersection = (error_pred & error_true).sum().item()
            union = (error_pred | error_true).sum().item()
            if union > 0:
                total_error_iou += intersection / union
            num_batches += 1

        pbar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "px_acc": f"{total_pixel_correct/total_pixels:.2%}"
        })

    return {
        "loss": total_loss / (total_pixels / (GRID_SIZE * GRID_SIZE)),
        "pixel_accuracy": total_pixel_correct / total_pixels,
        "error_iou": total_error_iou / num_batches if num_batches > 0 else 0,
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
    total_pixel_correct = 0
    total_pixels = 0
    total_error_iou = 0.0
    num_batches = 0

    # Track per-sample accuracy
    total_samples = 0
    perfect_predictions = 0

    for input_grid, output_grid, pixel_mask, is_positive in loader:
        input_grid = input_grid.to(device)
        output_grid = output_grid.to(device)
        pixel_mask = pixel_mask.to(device)

        logits = model(input_grid, output_grid)
        loss = F.binary_cross_entropy_with_logits(logits, pixel_mask)

        preds = (logits > 0).float()
        total_loss += loss.item() * input_grid.size(0)
        total_pixel_correct += (preds == pixel_mask).sum().item()
        total_pixels += pixel_mask.numel()

        # IoU for error pixels
        error_pred = (preds == 0)
        error_true = (pixel_mask == 0)
        intersection = (error_pred & error_true).sum().item()
        union = (error_pred | error_true).sum().item()
        if union > 0:
            total_error_iou += intersection / union
        num_batches += 1

        # Perfect predictions (all pixels correct)
        for i in range(input_grid.size(0)):
            if (preds[i] == pixel_mask[i]).all():
                perfect_predictions += 1
            total_samples += 1

    return {
        "loss": total_loss / (total_pixels / (GRID_SIZE * GRID_SIZE)),
        "pixel_accuracy": total_pixel_correct / total_pixels,
        "error_iou": total_error_iou / num_batches if num_batches > 0 else 0,
        "perfect_rate": perfect_predictions / total_samples if total_samples > 0 else 0,
    }


def visualize_predictions(
    model: nn.Module,
    dataset: PixelErrorDataset,
    device: torch.device,
    num_samples: int = 3,
):
    """
    Visualize pixel-level predictions showing the full task:
    - Input grid
    - Correct output
    - Distractor (wrong) output
    - Where errors are
    - What the model predicts
    """
    model.eval()

    print("\n" + "="*80)
    print("VISUALIZATION: What the model is learning")
    print("="*80)
    print("\nThe model sees: INPUT + DISTRACTOR")
    print("The model predicts: which pixels in DISTRACTOR are WRONG")
    print("="*80)

    # Get raw examples from dataset (before augmentation)
    examples = dataset.examples[:num_samples]

    for i, (input_grid_raw, correct_output_raw, puzzle_id) in enumerate(examples):
        print(f"\n{'━'*80}")
        print(f"Example {i+1}: Puzzle {puzzle_id}")
        print(f"{'━'*80}")

        # Generate a distractor (wrong output)
        wrong_output, _ = dataset._generate_negative(correct_output_raw, puzzle_id)

        # Pad all grids
        input_padded = dataset._pad_grid(input_grid_raw)
        correct_padded = dataset._pad_grid(correct_output_raw)
        wrong_padded = dataset._pad_grid(wrong_output)

        # Compute error mask
        error_mask = (wrong_padded != correct_padded).astype(np.float32)

        # Get model prediction
        inp_t = torch.from_numpy(input_padded.copy()).long().unsqueeze(0).to(device)
        wrong_t = torch.from_numpy(wrong_padded.copy()).long().unsqueeze(0).to(device)

        with torch.no_grad():
            pred_proba = model.predict_proba(inp_t, wrong_t)[0].cpu().numpy()

        # Find bounds based on correct output (the actual content)
        h, w = correct_output_raw.shape
        r_min, c_min = 0, 0
        r_max = min(h, 12)
        c_max = min(w, 12)

        # Print grids side by side
        print(f"\n{'INPUT':<15} {'CORRECT OUTPUT':<15} {'DISTRACTOR':<15} {'ERRORS (X=wrong)':<18} {'MODEL PRED':<15}")
        print(f"{'─'*15} {'─'*15} {'─'*15} {'─'*18} {'─'*15}")

        for r in range(r_max):
            inp_row = ""
            correct_row = ""
            wrong_row = ""
            error_row = ""
            pred_row = ""

            for c in range(c_max):
                # Input
                if r < input_grid_raw.shape[0] and c < input_grid_raw.shape[1]:
                    inp_row += f"{input_grid_raw[r, c]} "
                else:
                    inp_row += ". "

                # Correct output
                if r < correct_output_raw.shape[0] and c < correct_output_raw.shape[1]:
                    correct_row += f"{correct_output_raw[r, c]} "
                else:
                    correct_row += ". "

                # Wrong output (distractor)
                if r < wrong_output.shape[0] and c < wrong_output.shape[1]:
                    wrong_row += f"{wrong_output[r, c]} "
                else:
                    wrong_row += ". "

                # Error mask (X where distractor differs from correct)
                error_row += "X " if error_mask[r, c] == 1 else ". "

                # Model prediction (X where model thinks it's wrong)
                pred_row += "X " if pred_proba[r, c] < 0.5 else ". "

            print(f"{inp_row:<15} {correct_row:<15} {wrong_row:<15} {error_row:<18} {pred_row:<15}")

        # Statistics
        total_errors = int(error_mask.sum())
        pred_errors = int((pred_proba < 0.5).sum())
        true_positives = int(((error_mask == 1) & (pred_proba < 0.5)).sum())

        precision = true_positives / pred_errors if pred_errors > 0 else 0
        recall = true_positives / total_errors if total_errors > 0 else 0

        print(f"\nStats: {total_errors} actual errors, {pred_errors} predicted errors, {true_positives} correctly found")
        print(f"       Precision: {precision:.0%}, Recall: {recall:.0%}")


# =============================================================================
# Main
# =============================================================================

def load_puzzles(dataset_name: str, data_root: str = "kaggle/combined") -> Dict:
    """Load puzzle data"""
    config = {
        "arc-agi-1": {"subsets": ["training", "evaluation"]},
        "arc-agi-2": {"subsets": ["training2", "evaluation2"]},
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
    parser = argparse.ArgumentParser(description="Train Pixel Error CNN")

    parser.add_argument("--dataset", type=str, default="arc-agi-1",
                        choices=["arc-agi-1", "arc-agi-2"])
    parser.add_argument("--data-root", type=str, default="kaggle/combined")
    parser.add_argument("--single-puzzle", type=str, default=None)

    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num-negatives", type=int, default=4)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--save-path", type=str, default="checkpoints/pixel_error_cnn.pt",
                        help="Path to save best model checkpoint")

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
    train_dataset = PixelErrorDataset(train_puzzles, num_negatives=args.num_negatives, augment=True)
    val_dataset = PixelErrorDataset(val_puzzles, num_negatives=args.num_negatives, augment=False)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # Create model
    print("\nCreating model...")
    model = PixelErrorCNN(hidden_dim=args.hidden_dim)
    model = model.to(DEVICE)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)

    # Create checkpoint directory
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)

    # Training
    print("\n" + "="*60)
    print("Starting Training")
    print("="*60)

    best_val_iou = 0.0

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")

        train_metrics = train_epoch(model, train_loader, optimizer, DEVICE)
        val_metrics = evaluate(model, val_loader, DEVICE)

        scheduler.step()

        print(f"  Train Loss: {train_metrics['loss']:.4f}, Pixel Acc: {train_metrics['pixel_accuracy']:.2%}, Error IoU: {train_metrics['error_iou']:.2%}")
        print(f"  Val Loss: {val_metrics['loss']:.4f}, Pixel Acc: {val_metrics['pixel_accuracy']:.2%}, Error IoU: {val_metrics['error_iou']:.2%}")
        print(f"  Val Perfect Rate: {val_metrics['perfect_rate']:.2%}")

        if val_metrics['error_iou'] > best_val_iou:
            best_val_iou = val_metrics['error_iou']
            print(f"  [New best Error IoU: {best_val_iou:.2%}]")

    # Final summary
    print("\n" + "="*60)
    print("Training Complete")
    print("="*60)
    print(f"Best Error IoU: {best_val_iou:.2%}")

    # Save checkpoint
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'args': vars(args),
    }
    torch.save(checkpoint, args.save_path)
    print(f"Checkpoint saved to: {args.save_path}")

    # Visualize
    visualize_predictions(model, val_dataset, DEVICE)

    print("\nDone!")


if __name__ == "__main__":
    main()