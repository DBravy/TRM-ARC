#!/usr/bin/env python3
"""
Pixel-Level Error Detection CNN for ARC Puzzles - HARD NEGATIVES VERSION

This version focuses on hard negatives that are "almost correct" to force
the CNN to learn fine-grained pixel-level correctness detection.

Hard negative strategies:
1. Single pixel corruption (1 pixel wrong)
2. Few pixel corruption (2-5 pixels wrong)  
3. Local region corruption (small contiguous area)
4. Color swap (swap two colors throughout)
5. Single row/column shift
6. Boundary errors (errors only at edges of shapes)
7. Pattern extension errors (extend a repeating pattern incorrectly)

Usage:
    python train_pixel_error_cnn_hard.py --dataset arc-agi-1
    python train_pixel_error_cnn_hard.py --single-puzzle 00d62c1b
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
from scipy import ndimage

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
# Hard Negative Generation
# =============================================================================

def get_nonzero_positions(grid: np.ndarray) -> List[Tuple[int, int]]:
    """Get list of positions with non-zero values"""
    positions = list(zip(*np.where(grid > 0)))
    return positions


def get_boundary_positions(grid: np.ndarray) -> List[Tuple[int, int]]:
    """Get positions at boundaries between different colors"""
    h, w = grid.shape
    boundary = []
    
    for r in range(h):
        for c in range(w):
            if grid[r, c] == 0:
                continue
            # Check if any neighbor is different
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w:
                    if grid[nr, nc] != grid[r, c]:
                        boundary.append((r, c))
                        break
    return boundary


def get_colors_used(grid: np.ndarray) -> List[int]:
    """Get list of colors used in grid (excluding 0)"""
    return [c for c in np.unique(grid) if c > 0]


def corrupt_single_pixel(grid: np.ndarray) -> np.ndarray:
    """Corrupt exactly 1 pixel"""
    result = grid.copy()
    positions = get_nonzero_positions(grid)
    
    if not positions:
        # Fallback: corrupt any position
        r, c = random.randint(0, grid.shape[0]-1), random.randint(0, grid.shape[1]-1)
    else:
        r, c = random.choice(positions)
    
    # Change to a different color
    current = result[r, c]
    new_color = random.choice([x for x in range(10) if x != current])
    result[r, c] = new_color
    
    return result


def corrupt_few_pixels(grid: np.ndarray, num_pixels: int = None) -> np.ndarray:
    """Corrupt 2-5 random pixels"""
    result = grid.copy()
    
    if num_pixels is None:
        num_pixels = random.randint(2, 5)
    
    positions = get_nonzero_positions(grid)
    if len(positions) < num_pixels:
        positions = [(r, c) for r in range(grid.shape[0]) for c in range(grid.shape[1])]
    
    chosen = random.sample(positions, min(num_pixels, len(positions)))
    
    for r, c in chosen:
        current = result[r, c]
        new_color = random.choice([x for x in range(10) if x != current])
        result[r, c] = new_color
    
    return result


def corrupt_local_region(grid: np.ndarray) -> np.ndarray:
    """Corrupt a small contiguous region (2x2 or 3x3)"""
    result = grid.copy()
    h, w = grid.shape
    
    # Choose region size
    size = random.choice([2, 2, 2, 3])  # Bias toward 2x2
    
    # Find a position where we can place the region
    positions = get_nonzero_positions(grid)
    if positions:
        center_r, center_c = random.choice(positions)
    else:
        center_r = random.randint(0, h - size)
        center_c = random.randint(0, w - size)
    
    # Ensure region fits
    start_r = max(0, min(center_r, h - size))
    start_c = max(0, min(center_c, w - size))
    
    # Corrupt the region
    for dr in range(size):
        for dc in range(size):
            r, c = start_r + dr, start_c + dc
            if r < h and c < w:
                current = result[r, c]
                new_color = random.choice([x for x in range(10) if x != current])
                result[r, c] = new_color
    
    return result


def corrupt_color_swap(grid: np.ndarray) -> np.ndarray:
    """Swap two colors throughout the grid"""
    colors = get_colors_used(grid)
    
    if len(colors) < 2:
        # Fallback to single pixel corruption
        return corrupt_single_pixel(grid)
    
    c1, c2 = random.sample(colors, 2)
    
    result = grid.copy()
    mask1 = grid == c1
    mask2 = grid == c2
    result[mask1] = c2
    result[mask2] = c1
    
    return result


def corrupt_partial_color_swap(grid: np.ndarray) -> np.ndarray:
    """Swap two colors but only in part of the grid"""
    colors = get_colors_used(grid)
    
    if len(colors) < 2:
        return corrupt_single_pixel(grid)
    
    c1, c2 = random.sample(colors, 2)
    
    result = grid.copy()
    h, w = grid.shape
    
    # Only swap in top half, bottom half, left half, or right half
    region = random.choice(['top', 'bottom', 'left', 'right'])
    
    if region == 'top':
        mask = np.zeros_like(grid, dtype=bool)
        mask[:h//2, :] = True
    elif region == 'bottom':
        mask = np.zeros_like(grid, dtype=bool)
        mask[h//2:, :] = True
    elif region == 'left':
        mask = np.zeros_like(grid, dtype=bool)
        mask[:, :w//2] = True
    else:
        mask = np.zeros_like(grid, dtype=bool)
        mask[:, w//2:] = True
    
    swap1 = (grid == c1) & mask
    swap2 = (grid == c2) & mask
    result[swap1] = c2
    result[swap2] = c1
    
    return result


def corrupt_shift_row_or_col(grid: np.ndarray) -> np.ndarray:
    """Shift a single row or column by 1 pixel"""
    result = grid.copy()
    h, w = grid.shape
    
    if random.random() < 0.5:
        # Shift a row
        row = random.randint(0, h - 1)
        direction = random.choice([-1, 1])
        result[row, :] = np.roll(grid[row, :], direction)
    else:
        # Shift a column
        col = random.randint(0, w - 1)
        direction = random.choice([-1, 1])
        result[:, col] = np.roll(grid[:, col], direction)
    
    return result


def corrupt_boundary_pixels(grid: np.ndarray) -> np.ndarray:
    """Corrupt 1-3 pixels at shape boundaries"""
    result = grid.copy()
    boundary = get_boundary_positions(grid)
    
    if not boundary:
        return corrupt_single_pixel(grid)
    
    num_corrupt = random.randint(1, min(3, len(boundary)))
    chosen = random.sample(boundary, num_corrupt)
    
    for r, c in chosen:
        current = result[r, c]
        new_color = random.choice([x for x in range(10) if x != current])
        result[r, c] = new_color
    
    return result


def corrupt_extend_wrong(grid: np.ndarray) -> np.ndarray:
    """Add or remove pixels at the edge of shapes"""
    result = grid.copy()
    h, w = grid.shape
    
    # Find edges of non-zero regions
    positions = get_nonzero_positions(grid)
    if not positions:
        return corrupt_single_pixel(grid)
    
    # Try to extend a shape
    attempts = 0
    while attempts < 20:
        r, c = random.choice(positions)
        color = grid[r, c]
        
        # Pick a direction
        dr, dc = random.choice([(-1, 0), (1, 0), (0, -1), (0, 1)])
        nr, nc = r + dr, c + dc
        
        if 0 <= nr < h and 0 <= nc < w:
            if grid[nr, nc] == 0:
                # Extend: add pixel where there was none
                result[nr, nc] = color
                return result
            elif grid[nr, nc] != color:
                # Different color neighbor: change it
                result[nr, nc] = color
                return result
        
        attempts += 1
    
    # Fallback
    return corrupt_single_pixel(grid)


def corrupt_wrong_color_fill(grid: np.ndarray) -> np.ndarray:
    """Change one instance of a color to a different color (like wrong fill)"""
    result = grid.copy()
    
    # Find connected components
    colors = get_colors_used(grid)
    if not colors:
        return corrupt_single_pixel(grid)
    
    target_color = random.choice(colors)
    labeled, num_features = ndimage.label(grid == target_color)
    
    if num_features == 0:
        return corrupt_single_pixel(grid)
    
    # Pick one component to change
    component_id = random.randint(1, num_features)
    mask = labeled == component_id
    
    # Change to a different color
    new_color = random.choice([x for x in range(10) if x != target_color])
    result[mask] = new_color
    
    return result


def corrupt_almost_correct(grid: np.ndarray) -> np.ndarray:
    """
    Generate a hard negative that's almost correct.
    Randomly picks one of the hard corruption strategies.
    """
    strategies = [
        (corrupt_single_pixel, 3),           # Weight 3 - most common
        (corrupt_few_pixels, 2),              # Weight 2
        (corrupt_local_region, 1),            # Weight 1
        (corrupt_boundary_pixels, 2),         # Weight 2
        (corrupt_shift_row_or_col, 1),        # Weight 1
        (corrupt_color_swap, 1),              # Weight 1
        (corrupt_partial_color_swap, 1),      # Weight 1
        (corrupt_extend_wrong, 1),            # Weight 1
        (corrupt_wrong_color_fill, 1),        # Weight 1
    ]
    
    # Build weighted list
    weighted = []
    for func, weight in strategies:
        weighted.extend([func] * weight)
    
    chosen_func = random.choice(weighted)
    
    try:
        return chosen_func(grid)
    except Exception:
        # Fallback to single pixel if anything goes wrong
        return corrupt_single_pixel(grid)


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
# Dataset with Hard Negatives
# =============================================================================

class HardNegativeDataset(Dataset):
    """
    Dataset for pixel-level error detection with HARD negatives.

    All negatives are "almost correct" - only a few pixels wrong.
    This forces the CNN to learn fine-grained correctness detection.

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
        include_easy_negatives: bool = False,
        easy_negative_ratio: float = 0.1,
    ):
        self.num_negatives = num_negatives
        self.augment = augment
        self.include_easy_negatives = include_easy_negatives
        self.easy_negative_ratio = easy_negative_ratio

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
        print(f"Hard negatives mode: num_negatives={num_negatives}")
        if include_easy_negatives:
            print(f"Including {easy_negative_ratio:.0%} easy negatives for diversity")

    def __len__(self):
        return len(self.examples) * (1 + self.num_negatives)

    def _pad_grid(self, grid: np.ndarray) -> np.ndarray:
        """Pad grid to 30x30"""
        h, w = grid.shape
        padded = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.uint8)
        padded[:h, :w] = grid
        return padded

    def _generate_easy_negative(self, correct_output: np.ndarray) -> np.ndarray:
        """Generate an easy negative (for diversity)"""
        strategy = random.choice(["random_noise", "blank", "full_random"])
        
        if strategy == "blank":
            return np.zeros_like(correct_output, dtype=np.uint8)
        elif strategy == "full_random":
            return np.random.randint(0, 10, size=correct_output.shape, dtype=np.uint8)
        else:
            # Heavy random noise
            result = correct_output.copy()
            num_changes = random.randint(correct_output.size // 4, correct_output.size // 2)
            for _ in range(num_changes):
                r = random.randint(0, result.shape[0] - 1)
                c = random.randint(0, result.shape[1] - 1)
                result[r, c] = random.randint(0, 9)
            return result

    def _generate_negative(self, correct_output: np.ndarray, puzzle_id: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate a wrong output and the corresponding error mask.
        Primarily uses HARD negatives (almost correct).

        Returns:
            wrong_output: the incorrect output grid
            error_mask: 1 where pixels match correct_output, 0 where they differ
        """
        # Occasionally include easy negatives for diversity
        if self.include_easy_negatives and random.random() < self.easy_negative_ratio:
            wrong_output = self._generate_easy_negative(correct_output)
        else:
            # HARD negative - almost correct
            wrong_output = corrupt_almost_correct(correct_output)

        # Compute error mask: 1 = correct, 0 = wrong
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
            # Negative sample - HARD negative
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
    total_error_tp = 0
    total_error_fp = 0
    total_error_fn = 0
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
            
            # Pixel accuracy
            correct = (preds == pixel_mask).sum().item()
            total = pixel_mask.numel()
            total_pixel_correct += correct
            total_pixels += total

            # Error detection metrics (where mask = 0, i.e., pixel is wrong)
            error_target = (pixel_mask == 0)
            error_pred = (preds == 0)
            
            total_error_tp += (error_target & error_pred).sum().item()
            total_error_fp += (~error_target & error_pred).sum().item()
            total_error_fn += (error_target & ~error_pred).sum().item()

            total_loss += loss.item()
            num_batches += 1

        pbar.set_postfix(loss=loss.item())

    # Compute aggregate metrics
    pixel_accuracy = total_pixel_correct / max(total_pixels, 1)
    
    precision = total_error_tp / max(total_error_tp + total_error_fp, 1)
    recall = total_error_tp / max(total_error_tp + total_error_fn, 1)
    error_iou = total_error_tp / max(total_error_tp + total_error_fp + total_error_fn, 1)

    return {
        "loss": total_loss / max(num_batches, 1),
        "pixel_accuracy": pixel_accuracy,
        "error_precision": precision,
        "error_recall": recall,
        "error_iou": error_iou,
    }


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    """Evaluate model"""
    model.eval()

    total_loss = 0.0
    total_pixel_correct = 0
    total_pixels = 0
    total_error_tp = 0
    total_error_fp = 0
    total_error_fn = 0
    total_perfect = 0
    total_samples = 0
    num_batches = 0

    with torch.no_grad():
        for input_grid, output_grid, pixel_mask, is_positive in loader:
            input_grid = input_grid.to(device)
            output_grid = output_grid.to(device)
            pixel_mask = pixel_mask.to(device)

            logits = model(input_grid, output_grid)
            loss = F.binary_cross_entropy_with_logits(logits, pixel_mask)

            preds = (logits > 0).float()

            # Pixel accuracy
            correct = (preds == pixel_mask).sum().item()
            total = pixel_mask.numel()
            total_pixel_correct += correct
            total_pixels += total

            # Error detection metrics
            error_target = (pixel_mask == 0)
            error_pred = (preds == 0)
            
            total_error_tp += (error_target & error_pred).sum().item()
            total_error_fp += (~error_target & error_pred).sum().item()
            total_error_fn += (error_target & ~error_pred).sum().item()

            # Perfect predictions
            batch_perfect = (preds == pixel_mask).all(dim=-1).all(dim=-1).sum().item()
            total_perfect += batch_perfect
            total_samples += input_grid.size(0)

            total_loss += loss.item()
            num_batches += 1

    pixel_accuracy = total_pixel_correct / max(total_pixels, 1)
    precision = total_error_tp / max(total_error_tp + total_error_fp, 1)
    recall = total_error_tp / max(total_error_tp + total_error_fn, 1)
    error_iou = total_error_tp / max(total_error_tp + total_error_fp + total_error_fn, 1)
    perfect_rate = total_perfect / max(total_samples, 1)

    return {
        "loss": total_loss / max(num_batches, 1),
        "pixel_accuracy": pixel_accuracy,
        "error_precision": precision,
        "error_recall": recall,
        "error_iou": error_iou,
        "perfect_rate": perfect_rate,
    }


def visualize_predictions(
    model: nn.Module,
    dataset: Dataset,
    device: torch.device,
    num_samples: int = 5
):
    """Visualize model predictions on some examples"""
    model.eval()

    print("\n" + "="*80)
    print("Sample Predictions (Hard Negatives)")
    print("="*80)

    indices = random.sample(range(len(dataset)), min(num_samples, len(dataset)))

    for idx in indices:
        input_grid, output_grid, pixel_mask, is_positive = dataset[idx]

        inp_t = input_grid.unsqueeze(0).to(device)
        out_t = output_grid.unsqueeze(0).to(device)

        with torch.no_grad():
            pred_proba = model.predict_proba(inp_t, out_t)[0].cpu().numpy()

        input_np = input_grid.numpy()
        output_np = output_grid.numpy()
        mask_np = pixel_mask.numpy()

        # Find content bounds
        nonzero = np.where(output_np > 0)
        if len(nonzero[0]) > 0:
            r_min, r_max = nonzero[0].min(), min(nonzero[0].max() + 1, 12)
            c_min, c_max = nonzero[1].min(), min(nonzero[1].max() + 1, 12)
        else:
            r_min, r_max, c_min, c_max = 0, 8, 0, 8

        sample_type = "POSITIVE (correct)" if is_positive > 0.5 else "NEGATIVE (corrupted)"
        num_errors = int((mask_np == 0).sum())
        
        print(f"\n{'─'*80}")
        print(f"Sample {idx}: {sample_type}, {num_errors} error pixels")
        print(f"{'─'*80}")

        # Print grids
        print(f"{'INPUT':<15} {'OUTPUT':<15} {'ACTUAL ERRORS':<18} {'CNN PRED ERRORS':<18}")

        for r in range(r_min, r_max):
            inp_row = " ".join(str(input_np[r, c]) for c in range(c_min, c_max))
            out_row = " ".join(str(output_np[r, c]) for c in range(c_min, c_max))
            actual_row = " ".join("X" if mask_np[r, c] == 0 else "." for c in range(c_min, c_max))
            pred_row = " ".join("X" if pred_proba[r, c] < 0.5 else "." for c in range(c_min, c_max))
            print(f"{inp_row:<15} {out_row:<15} {actual_row:<18} {pred_row:<18}")

        # Stats
        actual_errors = (mask_np == 0)
        pred_errors = (pred_proba < 0.5)
        tp = (actual_errors & pred_errors).sum()
        fp = (~actual_errors & pred_errors).sum()
        fn = (actual_errors & ~pred_errors).sum()
        
        print(f"\nTP={tp}, FP={fp}, FN={fn}")

    print("="*80 + "\n")


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
    parser = argparse.ArgumentParser(description="Train Pixel Error CNN with Hard Negatives")

    parser.add_argument("--dataset", type=str, default="arc-agi-1",
                        choices=["arc-agi-1", "arc-agi-2"])
    parser.add_argument("--data-root", type=str, default="kaggle/combined")
    parser.add_argument("--single-puzzle", type=str, default=None)

    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=100,
                        help="More epochs needed for hard negatives")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num-negatives", type=int, default=8,
                        help="More negatives since they're harder")
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--save-path", type=str, default="checkpoints/pixel_error_cnn_hard.pt")
    
    # Hard negative specific args
    parser.add_argument("--include-easy", action="store_true",
                        help="Include some easy negatives for diversity")
    parser.add_argument("--easy-ratio", type=float, default=0.1,
                        help="Ratio of easy negatives when --include-easy is set")

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    print(f"Device: {DEVICE}")
    print(f"Dataset: {args.dataset}")
    print(f"Mode: HARD NEGATIVES")

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

    # Create datasets with HARD negatives
    train_dataset = HardNegativeDataset(
        train_puzzles, 
        num_negatives=args.num_negatives, 
        augment=True,
        include_easy_negatives=args.include_easy,
        easy_negative_ratio=args.easy_ratio,
    )
    val_dataset = HardNegativeDataset(
        val_puzzles, 
        num_negatives=args.num_negatives, 
        augment=False,
        include_easy_negatives=args.include_easy,
        easy_negative_ratio=args.easy_ratio,
    )

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
    print("Starting Training (Hard Negatives)")
    print("="*60)

    best_val_iou = 0.0

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")

        train_metrics = train_epoch(model, train_loader, optimizer, DEVICE)
        val_metrics = evaluate(model, val_loader, DEVICE)

        scheduler.step()

        print(f"  Train Loss: {train_metrics['loss']:.4f}, Pix Acc: {train_metrics['pixel_accuracy']:.2%}, "
              f"Err P/R: {train_metrics['error_precision']:.2%}/{train_metrics['error_recall']:.2%}, "
              f"IoU: {train_metrics['error_iou']:.2%}")
        print(f"  Val   Loss: {val_metrics['loss']:.4f}, Pix Acc: {val_metrics['pixel_accuracy']:.2%}, "
              f"Err P/R: {val_metrics['error_precision']:.2%}/{val_metrics['error_recall']:.2%}, "
              f"IoU: {val_metrics['error_iou']:.2%}")
        print(f"  Val Perfect Rate: {val_metrics['perfect_rate']:.2%}")

        if val_metrics['error_iou'] > best_val_iou:
            best_val_iou = val_metrics['error_iou']
            print(f"  [New best Error IoU: {best_val_iou:.2%}]")
            
            # Save best model
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'args': vars(args),
                'epoch': epoch,
                'best_val_iou': best_val_iou,
            }
            torch.save(checkpoint, args.save_path)

    # Final summary
    print("\n" + "="*60)
    print("Training Complete")
    print("="*60)
    print(f"Best Error IoU: {best_val_iou:.2%}")
    print(f"Checkpoint saved to: {args.save_path}")

    # Visualize
    visualize_predictions(model, val_dataset, DEVICE)

    print("\nDone!")


if __name__ == "__main__":
    main()