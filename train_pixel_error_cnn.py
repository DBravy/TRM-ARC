#!/usr/bin/env python3
"""
Pixel-Level Error Detection CNN - CORRESPONDENCE VERSION

This version forces the CNN to learn input-output correspondence by:
1. Using augmented PAIRS as positives (same aug applied to both input and output)
2. Using mismatched inputs as negatives (correct output, wrong input)
3. Using mismatched augmentations as negatives

The key insight: the CNN must see the SAME output labeled both "correct" and 
"incorrect" depending on the input. This forces it to actually compare them.

Usage:
    python train_pixel_error_cnn.py --dataset arc-agi-1
    python train_pixel_error_cnn.py --single-puzzle 00d62c1b
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


def get_random_augmentation() -> Tuple[int, np.ndarray]:
    """Get random augmentation parameters"""
    trans_id = random.randint(0, 7)
    color_map = random_color_permutation()
    return trans_id, color_map


# =============================================================================
# Hard Negative Corruption (from previous version)
# =============================================================================

def corrupt_single_pixel(grid: np.ndarray) -> np.ndarray:
    """Corrupt exactly 1 pixel"""
    result = grid.copy()
    h, w = grid.shape
    r, c = random.randint(0, h-1), random.randint(0, w-1)
    current = result[r, c]
    new_color = random.choice([x for x in range(10) if x != current])
    result[r, c] = new_color
    return result


def corrupt_few_pixels(grid: np.ndarray, num_pixels: int = None) -> np.ndarray:
    """Corrupt 2-5 random pixels"""
    result = grid.copy()
    if num_pixels is None:
        num_pixels = random.randint(2, 5)
    h, w = grid.shape
    for _ in range(num_pixels):
        r, c = random.randint(0, h-1), random.randint(0, w-1)
        current = result[r, c]
        new_color = random.choice([x for x in range(10) if x != current])
        result[r, c] = new_color
    return result


def corrupt_output(grid: np.ndarray) -> np.ndarray:
    """Apply random corruption to output"""
    if random.random() < 0.7:
        return corrupt_single_pixel(grid)
    else:
        return corrupt_few_pixels(grid)


# =============================================================================
# Degenerate Output Generators (for detecting obvious garbage)
# =============================================================================

def generate_all_zeros(grid: np.ndarray) -> np.ndarray:
    """Generate output that is all zeros (blank/empty)"""
    return np.zeros_like(grid)


def generate_constant_fill(grid: np.ndarray) -> np.ndarray:
    """Generate output that is all one color (1-9)"""
    color = random.randint(1, 9)
    return np.full_like(grid, color)


def generate_random_noise(grid: np.ndarray) -> np.ndarray:
    """Generate output that is random noise (0-9)"""
    return np.random.randint(0, 10, size=grid.shape, dtype=np.uint8)


def generate_color_swap(grid: np.ndarray) -> np.ndarray:
    """Swap one color with another in the grid, potentially only partially.

    Picks a color that exists in the grid and swaps it with a different color.
    With some probability, only a random fraction of pixels of that color get swapped.
    """
    result = grid.copy()

    # Find colors present in the grid (excluding 0 for more interesting swaps)
    unique_colors = np.unique(grid)
    non_zero_colors = unique_colors[unique_colors > 0]

    if len(non_zero_colors) == 0:
        # No non-zero colors, swap 0 with something
        color_from = 0
        color_to = random.randint(1, 9)
    else:
        # Pick a color to swap
        color_from = random.choice(non_zero_colors)
        # Pick a different color to swap to
        available_colors = [c for c in range(10) if c != color_from]
        color_to = random.choice(available_colors)

    # Get all pixel locations with this color
    swap_locations = np.argwhere(grid == color_from)

    if len(swap_locations) == 0:
        return result

    # Decide whether to do full swap or partial swap
    # 50% chance of partial swap
    if random.random() < 0.5:
        # Full swap - all pixels of that color
        result[grid == color_from] = color_to
    else:
        # Partial swap - random fraction of pixels
        # Use randomish ratios: 25%, 33%, 50%, 67%, 75%
        ratio = random.choice([0.25, 0.33, 0.5, 0.67, 0.75])
        num_to_swap = max(1, int(len(swap_locations) * ratio))

        # Randomly select which pixels to swap
        indices_to_swap = random.sample(range(len(swap_locations)), num_to_swap)
        for idx in indices_to_swap:
            r, c = swap_locations[idx]
            result[r, c] = color_to

    return result


# =============================================================================
# Model Architecture (same as before but with explicit comparison)
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


class Up(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diff_h = x2.size(2) - x1.size(2)
        diff_w = x2.size(3) - x1.size(3)
        x1 = F.pad(x1, [diff_w // 2, diff_w - diff_w // 2,
                       diff_h // 2, diff_h - diff_h // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class PixelErrorCNN(nn.Module):
    """
    U-Net style CNN with EXPLICIT comparison between input and output.
    
    Instead of just concatenating embeddings, we compute:
    - input embedding
    - output embedding  
    - difference (input - output)
    - element-wise product (input * output)
    
    This forces the network to see both and compare them.
    """

    def __init__(self, hidden_dim: int = 64, force_comparison: bool = True):
        super().__init__()
        
        self.force_comparison = force_comparison

        # Color embeddings for both grids
        self.input_embed = nn.Embedding(NUM_COLORS, 16)
        self.output_embed = nn.Embedding(NUM_COLORS, 16)

        # If forcing comparison: 16 + 16 + 16 + 16 = 64 channels
        # Otherwise: 16 + 16 = 32 channels
        if force_comparison:
            in_channels = 64
        else:
            in_channels = 32
            
        base_ch = hidden_dim

        # Encoder
        self.inc = DoubleConv(in_channels, base_ch)
        self.down1 = Down(base_ch, base_ch * 2)
        self.down2 = Down(base_ch * 2, base_ch * 4)
        self.down3 = Down(base_ch * 4, base_ch * 8)

        # Decoder
        self.up1 = Up(base_ch * 8, base_ch * 4)
        self.up2 = Up(base_ch * 4, base_ch * 2)
        self.up3 = Up(base_ch * 2, base_ch)

        # Output
        self.outc = nn.Conv2d(base_ch, 1, kernel_size=1)

    def forward(self, input_grid: torch.Tensor, output_grid: torch.Tensor) -> torch.Tensor:
        # Embed both grids
        inp_emb = self.input_embed(input_grid)   # (B, 30, 30, 16)
        out_emb = self.output_embed(output_grid) # (B, 30, 30, 16)

        if self.force_comparison:
            # Explicit comparison features
            diff = inp_emb - out_emb        # Where are they different?
            prod = inp_emb * out_emb        # Element-wise interaction
            x = torch.cat([inp_emb, out_emb, diff, prod], dim=-1)  # (B, 30, 30, 64)
        else:
            x = torch.cat([inp_emb, out_emb], dim=-1)  # (B, 30, 30, 32)
            
        x = x.permute(0, 3, 1, 2).contiguous()

        # U-Net forward
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)

        logits = self.outc(x)
        return logits.squeeze(1)

    def predict_proba(self, input_grid: torch.Tensor, output_grid: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.forward(input_grid, output_grid))

    @classmethod
    def from_checkpoint(cls, checkpoint_path: str, device: torch.device = None):
        checkpoint = torch.load(checkpoint_path, map_location=device or 'cpu', weights_only=False)
        args = checkpoint.get('args', {})
        hidden_dim = args.get('hidden_dim', 64)
        force_comparison = args.get('force_comparison', True)
        
        model = cls(hidden_dim=hidden_dim, force_comparison=force_comparison)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        if device:
            model = model.to(device)
        return model


# =============================================================================
# Dataset with Correspondence Learning
# =============================================================================

class CorrespondenceDataset(Dataset):
    """
    Dataset that forces learning of input-output CORRESPONDENCE.

    Sample types:
    1. POSITIVE: (input, output) with SAME augmentation applied to both
    2. NEGATIVE - corrupted: (input, corrupted_output)
    3. NEGATIVE - wrong input: (wrong_input, correct_output) ← CRITICAL
    4. NEGATIVE - mismatched aug: (aug1(input), aug2(output)) ← CRITICAL
    5. NEGATIVE - all_zeros: output is all zeros (blank)
    6. NEGATIVE - constant_fill: output is all one color (1-9)
    7. NEGATIVE - random_noise: output is random values
    8. NEGATIVE - color_swap: one color swapped with another

    Types 3-8 force the CNN to detect various failure modes!
    """

    def __init__(
        self,
        puzzles: Dict,
        num_positives: int = 2,
        num_corrupted: int = 2,
        num_wrong_input: int = 2,
        num_mismatched_aug: int = 2,
        num_all_zeros: int = 1,
        num_constant_fill: int = 1,
        num_random_noise: int = 1,
        num_color_swap: int = 1,
        augment: bool = True,
        dihedral_only: bool = False,
    ):
        self.num_positives = num_positives
        self.num_corrupted = num_corrupted
        self.num_wrong_input = num_wrong_input
        self.num_mismatched_aug = num_mismatched_aug
        self.num_all_zeros = num_all_zeros
        self.num_constant_fill = num_constant_fill
        self.num_random_noise = num_random_noise
        self.num_color_swap = num_color_swap
        self.samples_per_example = (
            num_positives + num_corrupted + num_wrong_input + num_mismatched_aug +
            num_all_zeros + num_constant_fill + num_random_noise + num_color_swap
        )
        self.augment = augment
        self.dihedral_only = dihedral_only

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
        print(f"Samples per example: {self.samples_per_example} "
              f"({num_positives} pos, {num_corrupted} corrupt, "
              f"{num_wrong_input} wrong_input, {num_mismatched_aug} mismatch, "
              f"{num_all_zeros} all_zeros, {num_constant_fill} const_fill, "
              f"{num_random_noise} noise, {num_color_swap} color_swap)")

    def __len__(self):
        return len(self.examples) * self.samples_per_example

    def _pad_grid(self, grid: np.ndarray) -> np.ndarray:
        h, w = grid.shape
        padded = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.uint8)
        padded[:h, :w] = grid
        return padded

    def _get_different_example(self, exclude_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """Get a different example (for wrong input/output negatives)"""
        idx = random.randint(0, len(self.examples) - 1)
        while idx == exclude_idx and len(self.examples) > 1:
            idx = random.randint(0, len(self.examples) - 1)
        return self.examples[idx][0], self.examples[idx][1]

    def _get_augmentation(self) -> Tuple[int, np.ndarray]:
        """Get augmentation params based on augment and dihedral_only settings"""
        if not self.augment:
            return 0, np.arange(10, dtype=np.uint8)  # Identity transform
        elif self.dihedral_only:
            # Random dihedral transform, but identity color mapping
            trans_id = random.randint(0, 7)
            color_map = np.arange(10, dtype=np.uint8)
            return trans_id, color_map
        else:
            return get_random_augmentation()  # Full augmentation

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        example_idx = idx // self.samples_per_example
        sample_type_idx = idx % self.samples_per_example

        input_grid, correct_output, puzzle_id = self.examples[example_idx]

        # Determine sample type based on configurable counts
        pos_end = self.num_positives
        corrupt_end = pos_end + self.num_corrupted
        wrong_input_end = corrupt_end + self.num_wrong_input
        mismatch_end = wrong_input_end + self.num_mismatched_aug
        all_zeros_end = mismatch_end + self.num_all_zeros
        const_fill_end = all_zeros_end + self.num_constant_fill
        noise_end = const_fill_end + self.num_random_noise

        if sample_type_idx < pos_end:
            sample_type = "positive"
        elif sample_type_idx < corrupt_end:
            sample_type = "corrupted"
        elif sample_type_idx < wrong_input_end:
            sample_type = "wrong_input"
        elif sample_type_idx < mismatch_end:
            sample_type = "mismatched_aug"
        elif sample_type_idx < all_zeros_end:
            sample_type = "all_zeros"
        elif sample_type_idx < const_fill_end:
            sample_type = "constant_fill"
        elif sample_type_idx < noise_end:
            sample_type = "random_noise"
        else:
            sample_type = "color_swap"

        if sample_type == "positive":
            # Same augmentation to both input and output
            trans_id, color_map = self._get_augmentation()
            aug_input = apply_augmentation(input_grid, trans_id, color_map)
            aug_output = apply_augmentation(correct_output, trans_id, color_map)

            final_input = aug_input
            final_output = aug_output
            # All pixels are correct
            pixel_mask = np.ones((GRID_SIZE, GRID_SIZE), dtype=np.float32)
            is_positive = 1.0

        elif sample_type == "corrupted":
            # Corrupted output (standard negative)
            trans_id, color_map = self._get_augmentation()
            aug_input = apply_augmentation(input_grid, trans_id, color_map)
            aug_output = apply_augmentation(correct_output, trans_id, color_map)
            corrupted = corrupt_output(aug_output)

            final_input = aug_input
            final_output = corrupted
            # Compute which pixels are wrong
            padded_correct = self._pad_grid(aug_output)
            padded_corrupted = self._pad_grid(corrupted)
            pixel_mask = (padded_correct == padded_corrupted).astype(np.float32)
            is_positive = 0.0

        elif sample_type == "wrong_input":
            # CRITICAL: Correct output but WRONG input
            # This forces CNN to check the input!
            wrong_input, _ = self._get_different_example(example_idx)

            # Apply same augmentation to both (so aug isn't the signal)
            trans_id, color_map = self._get_augmentation()
            aug_wrong_input = apply_augmentation(wrong_input, trans_id, color_map)
            aug_output = apply_augmentation(correct_output, trans_id, color_map)

            final_input = aug_wrong_input
            final_output = aug_output
            # Content area is wrong, but padding (0==0) is still correct
            h, w = aug_output.shape
            pixel_mask = np.ones((GRID_SIZE, GRID_SIZE), dtype=np.float32)
            pixel_mask[:h, :w] = 0.0  # Only content region marked as error
            is_positive = 0.0

        elif sample_type == "mismatched_aug":
            # CRITICAL: Input and output have DIFFERENT augmentations
            # Same puzzle, but augmentations don't match
            # Note: This sample type only makes sense when augment=True
            if self.dihedral_only:
                # Only mismatch dihedral transforms, not colors
                identity_colors = np.arange(10, dtype=np.uint8)
                trans_id_1 = random.randint(0, 7)
                trans_id_2 = random.randint(0, 7)
                while trans_id_1 == trans_id_2:
                    trans_id_2 = random.randint(0, 7)
                color_map_1, color_map_2 = identity_colors, identity_colors
            else:
                trans_id_1, color_map_1 = get_random_augmentation()
                trans_id_2, color_map_2 = get_random_augmentation()
                # Ensure they're actually different
                while trans_id_1 == trans_id_2 and np.array_equal(color_map_1, color_map_2):
                    trans_id_2, color_map_2 = get_random_augmentation()

            aug_input = apply_augmentation(input_grid, trans_id_1, color_map_1)
            aug_output = apply_augmentation(correct_output, trans_id_2, color_map_2)

            final_input = aug_input
            final_output = aug_output
            # Content area is wrong, but padding (0==0) is still correct
            # Use max dimensions since augmentations may rotate/transpose
            h1, w1 = aug_input.shape
            h2, w2 = aug_output.shape
            h_max, w_max = max(h1, h2), max(w1, w2)
            pixel_mask = np.ones((GRID_SIZE, GRID_SIZE), dtype=np.float32)
            pixel_mask[:h_max, :w_max] = 0.0  # Only content region marked as error
            is_positive = 0.0

        elif sample_type == "all_zeros":
            # Output is all zeros (blank/empty) - obvious garbage
            trans_id, color_map = self._get_augmentation()
            aug_input = apply_augmentation(input_grid, trans_id, color_map)
            aug_output = apply_augmentation(correct_output, trans_id, color_map)
            garbage_output = generate_all_zeros(correct_output)

            final_input = aug_input
            final_output = garbage_output
            # Compute which pixels are actually wrong (differ from correct)
            padded_correct = self._pad_grid(aug_output)
            padded_garbage = self._pad_grid(garbage_output)
            pixel_mask = (padded_correct == padded_garbage).astype(np.float32)
            is_positive = 0.0

        elif sample_type == "constant_fill":
            # Output is all one color (1-9) - obvious garbage
            trans_id, color_map = self._get_augmentation()
            aug_input = apply_augmentation(input_grid, trans_id, color_map)
            aug_output = apply_augmentation(correct_output, trans_id, color_map)
            garbage_output = generate_constant_fill(correct_output)

            final_input = aug_input
            final_output = garbage_output
            # Compute which pixels are actually wrong (differ from correct)
            padded_correct = self._pad_grid(aug_output)
            padded_garbage = self._pad_grid(garbage_output)
            pixel_mask = (padded_correct == padded_garbage).astype(np.float32)
            is_positive = 0.0

        elif sample_type == "random_noise":
            # Output is random noise - obvious garbage
            trans_id, color_map = self._get_augmentation()
            aug_input = apply_augmentation(input_grid, trans_id, color_map)
            aug_output = apply_augmentation(correct_output, trans_id, color_map)
            garbage_output = generate_random_noise(correct_output)

            final_input = aug_input
            final_output = garbage_output
            # Compute which pixels are actually wrong (differ from correct)
            padded_correct = self._pad_grid(aug_output)
            padded_garbage = self._pad_grid(garbage_output)
            pixel_mask = (padded_correct == padded_garbage).astype(np.float32)
            is_positive = 0.0

        else:  # color_swap
            # One color swapped with another - subtle corruption
            trans_id, color_map = self._get_augmentation()
            aug_input = apply_augmentation(input_grid, trans_id, color_map)
            aug_output = apply_augmentation(correct_output, trans_id, color_map)
            swapped_output = generate_color_swap(aug_output)

            final_input = aug_input
            final_output = swapped_output
            # Compute which pixels are actually wrong (differ from correct)
            padded_correct = self._pad_grid(aug_output)
            padded_swapped = self._pad_grid(swapped_output)
            pixel_mask = (padded_correct == padded_swapped).astype(np.float32)
            is_positive = 0.0

        # Pad grids
        final_input = self._pad_grid(final_input)
        final_output = self._pad_grid(final_output)

        # Ensure contiguous
        final_input = np.ascontiguousarray(final_input)
        final_output = np.ascontiguousarray(final_output)
        pixel_mask = np.ascontiguousarray(pixel_mask)

        return (
            torch.from_numpy(final_input.copy()).long(),
            torch.from_numpy(final_output.copy()).long(),
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

        with torch.no_grad():
            preds = (logits > 0).float()
            
            correct = (preds == pixel_mask).sum().item()
            total = pixel_mask.numel()
            total_pixel_correct += correct
            total_pixels += total

            error_target = (pixel_mask == 0)
            error_pred = (preds == 0)
            
            total_error_tp += (error_target & error_pred).sum().item()
            total_error_fp += (~error_target & error_pred).sum().item()
            total_error_fn += (error_target & ~error_pred).sum().item()

            total_loss += loss.item()
            num_batches += 1

        pbar.set_postfix(loss=loss.item())

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

            correct = (preds == pixel_mask).sum().item()
            total = pixel_mask.numel()
            total_pixel_correct += correct
            total_pixels += total

            error_target = (pixel_mask == 0)
            error_pred = (preds == 0)
            
            total_error_tp += (error_target & error_pred).sum().item()
            total_error_fp += (~error_target & error_pred).sum().item()
            total_error_fn += (error_target & ~error_pred).sum().item()

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


def visualize_predictions(model: nn.Module, dataset: Dataset, device: torch.device, num_samples: int = 8):
    """Visualize predictions across different sample types"""
    model.eval()

    print("\n" + "="*80)
    print("Sample Predictions (Correspondence Learning)")
    print("="*80)

    # Get samples of each type with their counts
    base = 0
    type_info = [
        ("positive", base, dataset.num_positives),
    ]
    base += dataset.num_positives
    type_info.append(("corrupted", base, dataset.num_corrupted))
    base += dataset.num_corrupted
    type_info.append(("wrong_input", base, dataset.num_wrong_input))
    base += dataset.num_wrong_input
    type_info.append(("mismatched_aug", base, dataset.num_mismatched_aug))
    base += dataset.num_mismatched_aug
    type_info.append(("all_zeros", base, dataset.num_all_zeros))
    base += dataset.num_all_zeros
    type_info.append(("constant_fill", base, dataset.num_constant_fill))
    base += dataset.num_constant_fill
    type_info.append(("random_noise", base, dataset.num_random_noise))
    base += dataset.num_random_noise
    type_info.append(("color_swap", base, dataset.num_color_swap))
    
    for sample_type, offset, count in type_info:
        if count == 0:
            continue
            
        print(f"\n{'─'*80}")
        print(f"Sample Type: {sample_type.upper()}")
        print(f"{'─'*80}")
        
        samples_to_show = min(2, count)
        for i in range(samples_to_show):
            # Calculate index for this sample type
            example_idx = random.randint(0, len(dataset.examples) - 1)
            sample_idx = example_idx * dataset.samples_per_example + offset + i
            
            if sample_idx >= len(dataset):
                continue
                
            input_grid, output_grid, pixel_mask, is_positive = dataset[sample_idx]

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
                r_max = min(nonzero[0].max() + 1, 8)
                c_max = min(nonzero[1].max() + 1, 8)
            else:
                r_max, c_max = 6, 6

            num_errors = int((mask_np[:r_max, :c_max] == 0).sum())
            expected = "ALL CORRECT" if is_positive > 0.5 else f"{num_errors} errors"
            
            print(f"\nExpected: {expected}")
            print(f"{'INPUT':<20} {'OUTPUT':<20} {'PREDICTED ERRORS':<20}")

            for r in range(r_max):
                inp_row = " ".join(f"{input_np[r, c]}" for c in range(c_max))
                out_row = " ".join(f"{output_np[r, c]}" for c in range(c_max))
                err_row = " ".join("X" if pred_proba[r, c] < 0.5 else "·" for c in range(c_max))
                print(f"{inp_row:<20} {out_row:<20} {err_row:<20}")

            # Stats (only count within content area)
            num_predicted_errors = (pred_proba[:r_max, :c_max] < 0.5).sum()
            print(f"Predicted errors: {num_predicted_errors}")

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
    parser = argparse.ArgumentParser(description="Train Pixel Error CNN with Correspondence Learning")

    parser.add_argument("--dataset", type=str, default="arc-agi-1",
                        choices=["arc-agi-1", "arc-agi-2"])
    parser.add_argument("--data-root", type=str, default="kaggle/combined")
    parser.add_argument("--single-puzzle", type=str, default=None)

    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num-negatives", type=int, default=8,
                        help="Total negatives per example (auto-distributed across types)")
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--save-path", type=str, default="checkpoints/pixel_error_cnn.pt")
    
    # Architecture options
    parser.add_argument("--no-force-comparison", action="store_true",
                        help="Disable explicit comparison features (for ablation)")
    parser.add_argument("--no-augment", action="store_true",
                        help="Train without augmentations (use raw example grids)")
    parser.add_argument("--dihedral-only", action="store_true",
                        help="Only use dihedral transforms (rotations/flips), no color permutations")

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    print(f"Device: {DEVICE}")
    print(f"Dataset: {args.dataset}")
    print(f"Mode: CORRESPONDENCE LEARNING")
    print(f"Force comparison: {not args.no_force_comparison}")

    # Determine augmentation mode
    if args.no_augment:
        aug_mode = "none"
    elif args.dihedral_only:
        aug_mode = "dihedral-only (rotations/flips, no color permutations)"
    else:
        aug_mode = "full (dihedral + color permutations)"
    print(f"Augmentation: {aug_mode}")

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

    # Auto-distribute negatives across types
    # Ratio: 1 positive per 4 negatives
    # Negatives split: 35% corrupted, 15% wrong_input, 15% mismatched_aug, 15% color_swap, 20% degenerate
    num_negatives = args.num_negatives
    num_positives = max(1, num_negatives // 4)
    use_augment = not args.no_augment

    # Split negatives across 7 types
    num_corrupted = max(1, int(num_negatives * 0.35))
    num_wrong_input = max(1, int(num_negatives * 0.15))
    # mismatched_aug only makes sense when augmentation is enabled
    num_mismatched_aug = max(1, int(num_negatives * 0.15)) if use_augment else 0
    num_color_swap = max(1, int(num_negatives * 0.15))
    # Degenerate outputs (all_zeros, constant_fill, random_noise) get remaining ~20%
    remaining = num_negatives - num_corrupted - num_wrong_input - num_mismatched_aug - num_color_swap
    num_all_zeros = max(1, remaining // 3)
    num_constant_fill = max(1, remaining // 3)
    num_random_noise = max(1, remaining - num_all_zeros - num_constant_fill)

    print(f"Auto-distributed {num_negatives} negatives: "
          f"{num_corrupted} corrupted, {num_wrong_input} wrong_input, {num_mismatched_aug} mismatched_aug, "
          f"{num_color_swap} color_swap, {num_all_zeros} all_zeros, {num_constant_fill} const_fill, {num_random_noise} noise")
    print(f"Plus {num_positives} positives per example")

    # Create datasets
    train_dataset = CorrespondenceDataset(
        train_puzzles,
        num_positives=num_positives,
        num_corrupted=num_corrupted,
        num_wrong_input=num_wrong_input,
        num_mismatched_aug=num_mismatched_aug,
        num_all_zeros=num_all_zeros,
        num_constant_fill=num_constant_fill,
        num_random_noise=num_random_noise,
        num_color_swap=num_color_swap,
        augment=use_augment,
        dihedral_only=args.dihedral_only,
    )
    val_dataset = CorrespondenceDataset(
        val_puzzles,
        num_positives=num_positives,
        num_corrupted=num_corrupted,
        num_wrong_input=num_wrong_input,
        num_mismatched_aug=num_mismatched_aug,
        num_all_zeros=num_all_zeros,
        num_constant_fill=num_constant_fill,
        num_random_noise=num_random_noise,
        num_color_swap=num_color_swap,
        augment=use_augment,
        dihedral_only=args.dihedral_only,
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # Create model with explicit comparison
    print("\nCreating model...")
    force_comparison = not args.no_force_comparison
    model = PixelErrorCNN(hidden_dim=args.hidden_dim, force_comparison=force_comparison)
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
    print("Starting Training (Correspondence Learning)")
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

        if val_metrics['error_iou'] > best_val_iou:
            best_val_iou = val_metrics['error_iou']
            print(f"  [New best Error IoU: {best_val_iou:.2%}]")
            
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'args': vars(args),
                'epoch': epoch,
                'best_val_iou': best_val_iou,
                'force_comparison': force_comparison,
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
    print("\nRun the diagnostic to verify the CNN now uses the input:")
    print(f"  python diagnose_cnn.py --checkpoint {args.save_path}")


if __name__ == "__main__":
    main()