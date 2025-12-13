#!/usr/bin/env python3
"""
Evaluate CNN color prediction generalization across many puzzles.

For each puzzle:
1. Detect color semantics (absolute vs relative) from training examples
2. Train CNN with appropriate augmentation strategy
3. Evaluate on held-out test examples
4. Record whether it generalizes

NEW: Automatic augmentation selection based on palette analysis
- Identical palettes across examples → absolute color semantics → dihedral only
- Varying palettes → relative color semantics → full augmentation (incl. color)

Usage:
    python eval_cnn_generalization.py --dataset arc-agi-1 --epochs 100
    python eval_cnn_generalization.py --dataset arc-agi-1 --epochs 100 --max-puzzles 50
    python eval_cnn_generalization.py --auto-augment  # NEW: automatic detection
"""

import argparse
import json
import os
import random
import time
from typing import Dict, List, Tuple, Set
from dataclasses import dataclass, field

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
# Palette Analysis - Detect Color Semantics
# =============================================================================

def get_palette(grid: np.ndarray, include_zero: bool = False) -> Set[int]:
    """Extract the set of colors used in a grid."""
    unique = set(np.unique(grid).tolist())
    if not include_zero:
        unique.discard(0)
    return unique


def analyze_puzzle_palettes(puzzle: Dict) -> Dict:
    """
    Analyze color palettes across all training examples.
    
    Returns:
        dict with:
            - input_palettes: list of palettes for each input
            - output_palettes: list of palettes for each output
            - combined_palettes: list of input∪output for each example
            - identical_input_palettes: bool - are all input palettes the same?
            - identical_output_palettes: bool - are all output palettes the same?
            - identical_combined_palettes: bool - are all combined palettes the same?
            - colors_preserved: bool - does each output use only colors from its input?
            - recommendation: "dihedral_only" or "full_augment"
    """
    train_examples = puzzle.get("train", [])
    
    if len(train_examples) == 0:
        return {"recommendation": "dihedral_only", "reason": "no_training_examples"}
    
    input_palettes = []
    output_palettes = []
    combined_palettes = []
    
    for ex in train_examples:
        inp = np.array(ex["input"], dtype=np.uint8)
        out = np.array(ex["output"], dtype=np.uint8)
        
        inp_pal = get_palette(inp)
        out_pal = get_palette(out)
        combined = inp_pal | out_pal
        
        input_palettes.append(inp_pal)
        output_palettes.append(out_pal)
        combined_palettes.append(combined)
    
    # Check if palettes are identical across examples
    identical_input = all(p == input_palettes[0] for p in input_palettes)
    identical_output = all(p == output_palettes[0] for p in output_palettes)
    identical_combined = all(p == combined_palettes[0] for p in combined_palettes)
    
    # Check if colors are preserved (output colors ⊆ input colors)
    colors_preserved = all(
        output_palettes[i] <= (input_palettes[i] | {0})  # output colors subset of input + black
        for i in range(len(train_examples))
    )
    
    # NEW: Check if outputs introduce new colors not in inputs
    new_colors_in_output = any(
        output_palettes[i] - input_palettes[i] - {0}  # colors in output but not input
        for i in range(len(train_examples))
    )
    
    # Decision logic
    # If palettes are identical AND outputs introduce specific new colors → absolute semantics
    # If palettes vary OR colors are just preserved from input → relative semantics
    
    if identical_combined:
        # Same colors across all examples - likely absolute color meanings
        recommendation = "dihedral_only"
        reason = "identical_palettes"
    elif colors_preserved and not new_colors_in_output:
        # Colors come from input - relative semantics, need color augmentation
        recommendation = "full_augment"
        reason = "colors_from_input"
    elif not identical_combined and new_colors_in_output:
        # Different palettes AND new colors - could go either way
        # Default to full augmentation to learn color relationships
        recommendation = "full_augment"
        reason = "varying_palettes_new_colors"
    else:
        # Varying palettes - likely relative semantics
        recommendation = "full_augment"
        reason = "varying_palettes"
    
    return {
        "input_palettes": input_palettes,
        "output_palettes": output_palettes,
        "combined_palettes": combined_palettes,
        "identical_input_palettes": identical_input,
        "identical_output_palettes": identical_output,
        "identical_combined_palettes": identical_combined,
        "colors_preserved": colors_preserved,
        "new_colors_in_output": new_colors_in_output,
        "recommendation": recommendation,
        "reason": reason,
    }


# =============================================================================
# Model (copied from train_pixel_error_cnn.py for standalone use)
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
    def __init__(self, hidden_dim: int = 64, num_classes: int = 10):
        super().__init__()
        self.num_classes = num_classes
        
        self.input_embed = nn.Embedding(NUM_COLORS, 16)
        self.output_embed = nn.Embedding(NUM_COLORS, 16)
        
        in_channels = 64  # force_comparison=True: 16+16+16+16
        base_ch = hidden_dim

        self.inc = DoubleConv(in_channels, base_ch)
        self.down1 = Down(base_ch, base_ch * 2)
        self.down2 = Down(base_ch * 2, base_ch * 4)
        self.down3 = Down(base_ch * 4, base_ch * 8)
        self.up1 = Up(base_ch * 8, base_ch * 4)
        self.up2 = Up(base_ch * 4, base_ch * 2)
        self.up3 = Up(base_ch * 2, base_ch)
        self.outc = nn.Conv2d(base_ch, num_classes, kernel_size=1)

    def forward(self, input_grid: torch.Tensor, output_grid: torch.Tensor) -> torch.Tensor:
        inp_emb = self.input_embed(input_grid)
        out_emb = self.output_embed(output_grid)
        
        diff = inp_emb - out_emb
        prod = inp_emb * out_emb
        x = torch.cat([inp_emb, out_emb, diff, prod], dim=-1)
        x = x.permute(0, 3, 1, 2).contiguous()

        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        return self.outc(x)

    def predict_colors(self, input_grid: torch.Tensor, output_grid: torch.Tensor) -> torch.Tensor:
        return self.forward(input_grid, output_grid).argmax(dim=1)


# =============================================================================
# Augmentation
# =============================================================================

def dihedral_transform(arr: np.ndarray, tid: int) -> np.ndarray:
    if tid == 0: return arr
    elif tid == 1: return np.rot90(arr, k=1)
    elif tid == 2: return np.rot90(arr, k=2)
    elif tid == 3: return np.rot90(arr, k=3)
    elif tid == 4: return np.fliplr(arr)
    elif tid == 5: return np.flipud(arr)
    elif tid == 6: return arr.T
    elif tid == 7: return np.fliplr(np.rot90(arr, k=1))
    return arr


def random_color_permutation() -> np.ndarray:
    return np.concatenate([
        np.array([0], dtype=np.uint8),
        np.random.permutation(np.arange(1, 10, dtype=np.uint8))
    ])


# =============================================================================
# Corruption utilities (matching train_pixel_error_cnn.py)
# =============================================================================

def corrupt_single_pixel(grid: np.ndarray) -> np.ndarray:
    result = grid.copy()
    h, w = grid.shape
    r, c = random.randint(0, h-1), random.randint(0, w-1)
    current = result[r, c]
    new_color = random.choice([x for x in range(10) if x != current])
    result[r, c] = new_color
    return result


def corrupt_few_pixels(grid: np.ndarray, num_pixels: int = None) -> np.ndarray:
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
    if random.random() < 0.7:
        return corrupt_single_pixel(grid)
    else:
        return corrupt_few_pixels(grid)


def generate_all_zeros(grid: np.ndarray) -> np.ndarray:
    return np.zeros_like(grid)


def generate_constant_fill(grid: np.ndarray) -> np.ndarray:
    color = random.randint(1, 9)
    return np.full_like(grid, color)


def generate_random_noise(grid: np.ndarray) -> np.ndarray:
    return np.random.randint(0, 10, size=grid.shape, dtype=np.uint8)


def generate_color_swap(grid: np.ndarray) -> np.ndarray:
    result = grid.copy()
    unique_colors = np.unique(grid)
    non_zero_colors = unique_colors[unique_colors > 0]
    
    if len(non_zero_colors) == 0:
        color_from = 0
        color_to = random.randint(1, 9)
    else:
        color_from = random.choice(non_zero_colors)
        available_colors = [c for c in range(10) if c != color_from]
        color_to = random.choice(available_colors)
    
    if random.random() < 0.5:
        result[grid == color_from] = color_to
    else:
        swap_locations = np.argwhere(grid == color_from)
        if len(swap_locations) > 0:
            ratio = random.choice([0.25, 0.33, 0.5, 0.67, 0.75])
            num_to_swap = max(1, int(len(swap_locations) * ratio))
            indices_to_swap = random.sample(range(len(swap_locations)), num_to_swap)
            for idx in indices_to_swap:
                r, c = swap_locations[idx]
                result[r, c] = color_to
    return result


# =============================================================================
# Dataset (matching train_pixel_error_cnn.py CorrespondenceDataset)
# =============================================================================

class SinglePuzzleDataset(Dataset):
    """
    Dataset for a single puzzle's training examples.
    Matches CorrespondenceDataset from train_pixel_error_cnn.py
    
    Includes all sample types:
    - positive: correct candidate
    - corrupted: some pixels wrong
    - wrong_input: mismatched input
    - mismatched_aug: different augmentations
    - all_zeros, constant_fill, random_noise, color_swap
    """
    
    def __init__(
        self, 
        puzzle: Dict, 
        num_positives: int = 2,
        num_corrupted: int = 3,
        num_wrong_input: int = 1,
        num_mismatched_aug: int = 1,
        num_all_zeros: int = 1,
        num_constant_fill: int = 1,
        num_random_noise: int = 1,
        num_color_swap: int = 1,
        dihedral_only: bool = True,
    ):
        self.num_positives = num_positives
        self.num_corrupted = num_corrupted
        self.num_wrong_input = num_wrong_input
        self.num_mismatched_aug = num_mismatched_aug
        self.num_all_zeros = num_all_zeros
        self.num_constant_fill = num_constant_fill
        self.num_random_noise = num_random_noise
        self.num_color_swap = num_color_swap
        self.dihedral_only = dihedral_only
        
        self.samples_per_example = (
            num_positives + num_corrupted + num_wrong_input + num_mismatched_aug +
            num_all_zeros + num_constant_fill + num_random_noise + num_color_swap
        )
        
        self.examples = []
        for example in puzzle.get("train", []):
            inp = np.array(example["input"], dtype=np.uint8)
            out = np.array(example["output"], dtype=np.uint8)
            self.examples.append((inp, out))
    
    def __len__(self):
        return len(self.examples) * self.samples_per_example
    
    def _pad_grid(self, grid: np.ndarray) -> np.ndarray:
        h, w = grid.shape
        padded = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.uint8)
        padded[:h, :w] = grid
        return padded
    
    def _get_augmentation(self):
        trans_id = random.randint(0, 7)
        if self.dihedral_only:
            color_map = np.arange(10, dtype=np.uint8)
        else:
            color_map = random_color_permutation()
        return trans_id, color_map
    
    def _get_different_example(self, exclude_idx: int):
        idx = random.randint(0, len(self.examples) - 1)
        while idx == exclude_idx and len(self.examples) > 1:
            idx = random.randint(0, len(self.examples) - 1)
        return self.examples[idx]
    
    def __getitem__(self, idx):
        example_idx = idx // self.samples_per_example
        sample_type_idx = idx % self.samples_per_example
        
        input_grid, correct_output = self.examples[example_idx]
        
        # Determine sample type
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
        
        # Generate sample based on type
        if sample_type == "positive":
            trans_id, color_map = self._get_augmentation()
            aug_input = dihedral_transform(color_map[input_grid], trans_id)
            aug_output = dihedral_transform(color_map[correct_output], trans_id)
            candidate = aug_output  # Correct!
            target = aug_output
            
        elif sample_type == "corrupted":
            trans_id, color_map = self._get_augmentation()
            aug_input = dihedral_transform(color_map[input_grid], trans_id)
            aug_output = dihedral_transform(color_map[correct_output], trans_id)
            candidate = corrupt_output(aug_output)
            target = aug_output
            
        elif sample_type == "wrong_input":
            wrong_inp, _ = self._get_different_example(example_idx)
            trans_id, color_map = self._get_augmentation()
            aug_input = dihedral_transform(color_map[wrong_inp], trans_id)
            aug_output = dihedral_transform(color_map[correct_output], trans_id)
            candidate = aug_output
            target = aug_output
            
        elif sample_type == "mismatched_aug":
            trans_id_1, color_map_1 = self._get_augmentation()
            trans_id_2, color_map_2 = self._get_augmentation()
            while trans_id_1 == trans_id_2 and np.array_equal(color_map_1, color_map_2):
                trans_id_2, color_map_2 = self._get_augmentation()
            aug_input = dihedral_transform(color_map_1[input_grid], trans_id_1)
            aug_output = dihedral_transform(color_map_2[correct_output], trans_id_2)
            candidate = aug_output
            # Target should match input's augmentation
            target = dihedral_transform(color_map_1[correct_output], trans_id_1)
            
        elif sample_type == "all_zeros":
            trans_id, color_map = self._get_augmentation()
            aug_input = dihedral_transform(color_map[input_grid], trans_id)
            aug_output = dihedral_transform(color_map[correct_output], trans_id)
            candidate = generate_all_zeros(aug_output)
            target = aug_output
            
        elif sample_type == "constant_fill":
            trans_id, color_map = self._get_augmentation()
            aug_input = dihedral_transform(color_map[input_grid], trans_id)
            aug_output = dihedral_transform(color_map[correct_output], trans_id)
            candidate = generate_constant_fill(aug_output)
            target = aug_output
            
        elif sample_type == "random_noise":
            trans_id, color_map = self._get_augmentation()
            aug_input = dihedral_transform(color_map[input_grid], trans_id)
            aug_output = dihedral_transform(color_map[correct_output], trans_id)
            candidate = generate_random_noise(aug_output)
            target = aug_output
            
        else:  # color_swap
            trans_id, color_map = self._get_augmentation()
            aug_input = dihedral_transform(color_map[input_grid], trans_id)
            aug_output = dihedral_transform(color_map[correct_output], trans_id)
            candidate = generate_color_swap(aug_output)
            target = aug_output
        
        return (
            torch.from_numpy(self._pad_grid(aug_input)).long(),
            torch.from_numpy(self._pad_grid(candidate)).long(),
            torch.from_numpy(self._pad_grid(target)).long(),
        )


class MultiPuzzleDataset(Dataset):
    """
    Dataset for training on ALL puzzles' training examples together.
    Each puzzle uses its own augmentation strategy (auto-detected or manual).

    Matches the negative sampling strategy of SinglePuzzleDataset.
    """

    def __init__(
        self,
        puzzles: Dict[str, Dict],
        num_positives: int = 2,
        num_corrupted: int = 3,
        num_wrong_input: int = 1,
        num_mismatched_aug: int = 1,
        num_all_zeros: int = 1,
        num_constant_fill: int = 1,
        num_random_noise: int = 1,
        num_color_swap: int = 1,
        auto_augment: bool = True,
        dihedral_only: bool = True,  # fallback if auto_augment=False
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

        # Collect all training examples with their augmentation strategies
        # Each entry: (puzzle_id, example_idx, input_grid, output_grid, dihedral_only)
        self.examples = []
        self.puzzle_examples = {}  # puzzle_id -> list of (input, output) for wrong_input sampling

        for puzzle_id, puzzle in puzzles.items():
            train_examples = puzzle.get("train", [])
            if not train_examples:
                continue

            # Determine augmentation strategy for this puzzle
            if auto_augment:
                palette_analysis = analyze_puzzle_palettes(puzzle)
                use_dihedral_only = (palette_analysis["recommendation"] == "dihedral_only")
            else:
                use_dihedral_only = dihedral_only

            puzzle_exs = []
            for ex_idx, example in enumerate(train_examples):
                inp = np.array(example["input"], dtype=np.uint8)
                out = np.array(example["output"], dtype=np.uint8)
                self.examples.append((puzzle_id, ex_idx, inp, out, use_dihedral_only))
                puzzle_exs.append((inp, out))

            self.puzzle_examples[puzzle_id] = puzzle_exs

    def __len__(self):
        return len(self.examples) * self.samples_per_example

    def _pad_grid(self, grid: np.ndarray) -> np.ndarray:
        h, w = grid.shape
        padded = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.uint8)
        padded[:h, :w] = grid
        return padded

    def _get_augmentation(self, dihedral_only: bool):
        trans_id = random.randint(0, 7)
        if dihedral_only:
            color_map = np.arange(10, dtype=np.uint8)
        else:
            color_map = random_color_permutation()
        return trans_id, color_map

    def _get_different_example(self, puzzle_id: str, exclude_idx: int):
        """Get a different example from the same puzzle for wrong_input samples."""
        puzzle_exs = self.puzzle_examples[puzzle_id]
        if len(puzzle_exs) <= 1:
            # Only one example, just return it (will be same but that's ok)
            return puzzle_exs[0]
        idx = random.randint(0, len(puzzle_exs) - 1)
        while idx == exclude_idx:
            idx = random.randint(0, len(puzzle_exs) - 1)
        return puzzle_exs[idx]

    def __getitem__(self, idx):
        example_idx = idx // self.samples_per_example
        sample_type_idx = idx % self.samples_per_example

        puzzle_id, ex_idx, input_grid, correct_output, dihedral_only = self.examples[example_idx]

        # Determine sample type (same logic as SinglePuzzleDataset)
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

        # Generate sample based on type
        if sample_type == "positive":
            trans_id, color_map = self._get_augmentation(dihedral_only)
            aug_input = dihedral_transform(color_map[input_grid], trans_id)
            aug_output = dihedral_transform(color_map[correct_output], trans_id)
            candidate = aug_output
            target = aug_output

        elif sample_type == "corrupted":
            trans_id, color_map = self._get_augmentation(dihedral_only)
            aug_input = dihedral_transform(color_map[input_grid], trans_id)
            aug_output = dihedral_transform(color_map[correct_output], trans_id)
            candidate = corrupt_output(aug_output)
            target = aug_output

        elif sample_type == "wrong_input":
            wrong_inp, _ = self._get_different_example(puzzle_id, ex_idx)
            trans_id, color_map = self._get_augmentation(dihedral_only)
            aug_input = dihedral_transform(color_map[wrong_inp], trans_id)
            aug_output = dihedral_transform(color_map[correct_output], trans_id)
            candidate = aug_output
            target = aug_output

        elif sample_type == "mismatched_aug":
            trans_id_1, color_map_1 = self._get_augmentation(dihedral_only)
            trans_id_2, color_map_2 = self._get_augmentation(dihedral_only)
            while trans_id_1 == trans_id_2 and np.array_equal(color_map_1, color_map_2):
                trans_id_2, color_map_2 = self._get_augmentation(dihedral_only)
            aug_input = dihedral_transform(color_map_1[input_grid], trans_id_1)
            aug_output = dihedral_transform(color_map_2[correct_output], trans_id_2)
            candidate = aug_output
            target = dihedral_transform(color_map_1[correct_output], trans_id_1)

        elif sample_type == "all_zeros":
            trans_id, color_map = self._get_augmentation(dihedral_only)
            aug_input = dihedral_transform(color_map[input_grid], trans_id)
            aug_output = dihedral_transform(color_map[correct_output], trans_id)
            candidate = generate_all_zeros(aug_output)
            target = aug_output

        elif sample_type == "constant_fill":
            trans_id, color_map = self._get_augmentation(dihedral_only)
            aug_input = dihedral_transform(color_map[input_grid], trans_id)
            aug_output = dihedral_transform(color_map[correct_output], trans_id)
            candidate = generate_constant_fill(aug_output)
            target = aug_output

        elif sample_type == "random_noise":
            trans_id, color_map = self._get_augmentation(dihedral_only)
            aug_input = dihedral_transform(color_map[input_grid], trans_id)
            aug_output = dihedral_transform(color_map[correct_output], trans_id)
            candidate = generate_random_noise(aug_output)
            target = aug_output

        else:  # color_swap
            trans_id, color_map = self._get_augmentation(dihedral_only)
            aug_input = dihedral_transform(color_map[input_grid], trans_id)
            aug_output = dihedral_transform(color_map[correct_output], trans_id)
            candidate = generate_color_swap(aug_output)
            target = aug_output

        return (
            torch.from_numpy(self._pad_grid(aug_input)).long(),
            torch.from_numpy(self._pad_grid(candidate)).long(),
            torch.from_numpy(self._pad_grid(target)).long(),
        )


# =============================================================================
# Training and Evaluation
# =============================================================================

def train_on_puzzle(
    puzzle: Dict,
    epochs: int = 100,
    hidden_dim: int = 64,
    lr: float = 1e-3,
    batch_size: int = 32,
    num_negatives: int = 8,
    dihedral_only: bool = True,
    negative_type: str = "mixed",
    verbose: bool = False,
) -> nn.Module:
    """Train a CNN on a single puzzle's training examples.
    
    Uses full augmentation strategy matching train_pixel_error_cnn.py:
    - Positive samples (correct candidate)
    - Corrupted samples
    - Wrong input samples
    - Mismatched augmentation samples
    - Degenerate outputs (zeros, constant, noise, color_swap)
    """
    
    # Distribute negatives based on negative_type
    num_positives = max(1, num_negatives // 4)
    
    if negative_type == "mixed":
        num_corrupted = max(1, int(num_negatives * 0.35))
        num_wrong_input = max(1, int(num_negatives * 0.15))
        num_mismatched_aug = max(1, int(num_negatives * 0.15))
        num_color_swap = max(1, int(num_negatives * 0.15))
        remaining = num_negatives - num_corrupted - num_wrong_input - num_mismatched_aug - num_color_swap
        num_all_zeros = max(1, remaining // 3)
        num_constant_fill = max(1, remaining // 3)
        num_random_noise = max(1, remaining - num_all_zeros - num_constant_fill)
    else:
        # Single negative type for ablation
        num_corrupted = 0
        num_wrong_input = 0
        num_mismatched_aug = 0
        num_color_swap = 0
        num_all_zeros = 0
        num_constant_fill = 0
        num_random_noise = 0
        
        if negative_type == "all_zeros":
            num_all_zeros = num_negatives
        elif negative_type == "corrupted":
            num_corrupted = num_negatives
        elif negative_type == "random_noise":
            num_random_noise = num_negatives
        elif negative_type == "constant_fill":
            num_constant_fill = num_negatives
        elif negative_type == "color_swap":
            num_color_swap = num_negatives
        elif negative_type == "wrong_input":
            num_wrong_input = num_negatives
        elif negative_type == "mismatched_aug":
            num_mismatched_aug = num_negatives
    
    dataset = SinglePuzzleDataset(
        puzzle,
        num_positives=num_positives,
        num_corrupted=num_corrupted,
        num_wrong_input=num_wrong_input,
        num_mismatched_aug=num_mismatched_aug,
        num_all_zeros=num_all_zeros,
        num_constant_fill=num_constant_fill,
        num_random_noise=num_random_noise,
        num_color_swap=num_color_swap,
        dihedral_only=dihedral_only,
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    
    model = PixelErrorCNN(hidden_dim=hidden_dim, num_classes=10).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for inp, candidate, target in loader:
            inp = inp.to(DEVICE)
            candidate = candidate.to(DEVICE)
            target = target.to(DEVICE)
            
            optimizer.zero_grad()
            logits = model(inp, candidate)
            loss = F.cross_entropy(logits, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        scheduler.step()
        
        if verbose and (epoch + 1) % 20 == 0:
            print(f"    Epoch {epoch+1}: loss={total_loss/len(loader):.4f}")

    return model


def train_on_all_puzzles(
    puzzles: Dict[str, Dict],
    epochs: int = 100,
    hidden_dim: int = 64,
    lr: float = 1e-3,
    batch_size: int = 32,
    num_negatives: int = 8,
    auto_augment: bool = True,
    dihedral_only: bool = True,
    verbose: bool = False,
) -> nn.Module:
    """Train a single CNN on ALL puzzles' training examples together.

    Uses the same augmentation and negative sampling strategy as train_on_puzzle,
    but combines all puzzles into one dataset.
    """
    # Distribute negatives (same logic as train_on_puzzle)
    num_positives = max(1, num_negatives // 4)
    num_corrupted = max(1, int(num_negatives * 0.35))
    num_wrong_input = max(1, int(num_negatives * 0.15))
    num_mismatched_aug = max(1, int(num_negatives * 0.15))
    num_color_swap = max(1, int(num_negatives * 0.15))
    remaining = num_negatives - num_corrupted - num_wrong_input - num_mismatched_aug - num_color_swap
    num_all_zeros = max(1, remaining // 3)
    num_constant_fill = max(1, remaining // 3)
    num_random_noise = max(1, remaining - num_all_zeros - num_constant_fill)

    dataset = MultiPuzzleDataset(
        puzzles,
        num_positives=num_positives,
        num_corrupted=num_corrupted,
        num_wrong_input=num_wrong_input,
        num_mismatched_aug=num_mismatched_aug,
        num_all_zeros=num_all_zeros,
        num_constant_fill=num_constant_fill,
        num_random_noise=num_random_noise,
        num_color_swap=num_color_swap,
        auto_augment=auto_augment,
        dihedral_only=dihedral_only,
    )

    print(f"Multi-puzzle dataset: {len(dataset.examples)} training examples from {len(dataset.puzzle_examples)} puzzles")
    print(f"Total samples per epoch: {len(dataset)}")

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    model = PixelErrorCNN(hidden_dim=hidden_dim, num_classes=10).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        num_batches = 0
        for inp, candidate, target in tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False):
            inp = inp.to(DEVICE)
            candidate = candidate.to(DEVICE)
            target = target.to(DEVICE)

            optimizer.zero_grad()
            logits = model(inp, candidate)
            loss = F.cross_entropy(logits, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            num_batches += 1

        scheduler.step()
        avg_loss = total_loss / max(num_batches, 1)

        if verbose or (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}: loss={avg_loss:.4f}")

    return model


def visualize_prediction(
    input_grid: np.ndarray,
    target_grid: np.ndarray, 
    pred_grid: np.ndarray,
    h: int, 
    w: int,
    puzzle_id: str
):
    """Visualize input, target output, and prediction stacked vertically."""
    # Color codes for terminal (ANSI) - 256 color mode for better visibility
    COLORS = [
        '\033[48;5;0m',    # 0: black
        '\033[48;5;21m',   # 1: blue
        '\033[48;5;196m',  # 2: red
        '\033[48;5;46m',   # 3: green
        '\033[48;5;226m',  # 4: yellow
        '\033[48;5;250m',  # 5: gray
        '\033[48;5;201m',  # 6: magenta
        '\033[48;5;208m',  # 7: orange
        '\033[48;5;51m',   # 8: cyan
        '\033[48;5;88m',   # 9: maroon
    ]
    RESET = '\033[0m'
    
    tgt = target_grid[:h, :w]
    pred = pred_grid[:h, :w]
    
    # Find actual input dimensions
    inp_nonzero = np.where(input_grid > 0)
    if len(inp_nonzero[0]) > 0:
        inp_h = inp_nonzero[0].max() + 1
        inp_w = inp_nonzero[1].max() + 1
    else:
        inp_h, inp_w = h, w
    inp = input_grid[:inp_h, :inp_w]
    
    # Calculate errors
    errors = (pred != tgt).sum()
    total = h * w
    
    print(f"\n{'═'*50}")
    if errors == 0:
        print(f"\033[92m✓ PERFECT: {puzzle_id}\033[0m")
    else:
        print(f"\033[91m✗ {puzzle_id}: {errors}/{total} errors ({100*errors/total:.1f}% wrong)\033[0m")
    print(f"  Input: {inp_h}×{inp_w} → Output: {h}×{w}")
    print(f"{'═'*50}")
    
    def print_grid(grid, label, rows, cols, highlight_errors=False, target=None):
        print(f"\n{label}:")
        for r in range(rows):
            row_str = "  "
            for c in range(cols):
                val = grid[r, c]
                if highlight_errors and target is not None and grid[r, c] != target[r, c]:
                    # White text on red background for errors
                    row_str += f"\033[41m\033[97m{val}\033[0m "
                else:
                    row_str += f"{COLORS[val]}{val}{RESET} "
            print(row_str)
    
    # Print grids stacked
    print_grid(inp, "INPUT", inp_h, inp_w)
    print_grid(tgt, "TARGET", h, w)
    print_grid(pred, "PREDICTION (errors in red)", h, w, highlight_errors=True, target=tgt)
    
    print()


def evaluate_on_test(model: nn.Module, puzzle: Dict, puzzle_id: str = "", 
                     candidate_mode: str = "zeros", visualize: bool = False) -> Dict:
    """
    Evaluate trained model on puzzle's test examples.
    
    CRITICAL: We must NOT give the model the correct output as input!
    The model was designed for error correction (input, candidate) -> corrected
    But for actual puzzle solving, we need to give it a blank/garbage candidate.
    
    candidate_mode:
        - "zeros": All zeros (blank grid)
        - "random": Random colors 0-9
        - "input_copy": Copy of the input (tests if it learns input->output transform)
    """
    model.eval()
    
    test_examples = puzzle.get("test", [])
    if not test_examples or not any("output" in ex for ex in test_examples):
        return {"error": "no_test_outputs"}
    
    results = []
    total_correct = 0
    total_pixels = 0
    
    with torch.no_grad():
        for ex in test_examples:
            if "output" not in ex:
                continue
            
            inp = np.array(ex["input"], dtype=np.uint8)
            out = np.array(ex["output"], dtype=np.uint8)
            h_out, w_out = out.shape
            h_inp, w_inp = inp.shape
            
            # Pad input
            inp_pad = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.uint8)
            inp_pad[:h_inp, :w_inp] = inp
            
            # Create candidate output - NOT the correct answer!
            if candidate_mode == "zeros":
                candidate_pad = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.uint8)
            elif candidate_mode == "random":
                candidate_pad = np.random.randint(0, 10, (GRID_SIZE, GRID_SIZE), dtype=np.uint8)
            elif candidate_mode == "input_copy":
                candidate_pad = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.uint8)
                candidate_pad[:h_inp, :w_inp] = inp
            else:
                raise ValueError(f"Unknown candidate_mode: {candidate_mode}")
            
            inp_t = torch.from_numpy(inp_pad).long().unsqueeze(0).to(DEVICE)
            candidate_t = torch.from_numpy(candidate_pad).long().unsqueeze(0).to(DEVICE)
            
            # Model predicts what the output SHOULD be
            pred = model.predict_colors(inp_t, candidate_t)[0].cpu().numpy()
            
            # Compare to actual correct output
            correct = (pred[:h_out, :w_out] == out).sum()
            total = h_out * w_out
            
            # Visualize if requested
            if visualize:
                # Pad output for visualization
                out_pad = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.uint8)
                out_pad[:h_out, :w_out] = out
                visualize_prediction(inp_pad, out_pad, pred, h_out, w_out, puzzle_id)
            
            results.append({
                "correct": int(correct),
                "total": total,
                "accuracy": correct / total,
                "perfect": correct == total
            })
            
            total_correct += correct
            total_pixels += total
    
    return {
        "pixel_accuracy": total_correct / max(total_pixels, 1),
        "perfect_rate": sum(r["perfect"] for r in results) / max(len(results), 1),
        "num_test_examples": len(results),
        "all_perfect": all(r["perfect"] for r in results),
        "results": results
    }


def evaluate_all_tests(
    model: nn.Module,
    puzzles: Dict[str, Dict],
    visualize: bool = False,
    verbose: bool = False,
) -> Dict:
    """
    Evaluate a trained model on ALL puzzles' test examples.

    Returns aggregate metrics and per-puzzle breakdown.
    """
    model.eval()

    all_results = []
    per_puzzle_results = {}

    total_correct = 0
    total_pixels = 0
    total_perfect = 0
    total_test_examples = 0

    puzzle_ids = [pid for pid in puzzles.keys()
                  if any("output" in ex for ex in puzzles[pid].get("test", []))]

    for puzzle_id in tqdm(puzzle_ids, desc="Evaluating tests"):
        puzzle = puzzles[puzzle_id]
        eval_result = evaluate_on_test(
            model, puzzle, puzzle_id=puzzle_id,
            candidate_mode="zeros", visualize=visualize
        )

        if "error" in eval_result:
            per_puzzle_results[puzzle_id] = {"error": eval_result["error"]}
            continue

        per_puzzle_results[puzzle_id] = {
            "pixel_accuracy": eval_result["pixel_accuracy"],
            "perfect_rate": eval_result["perfect_rate"],
            "all_perfect": eval_result["all_perfect"],
            "num_test_examples": eval_result["num_test_examples"],
        }

        # Aggregate stats
        for r in eval_result["results"]:
            total_correct += r["correct"]
            total_pixels += r["total"]
            if r["perfect"]:
                total_perfect += 1
            total_test_examples += 1

        all_results.append(eval_result)

        if verbose and eval_result["all_perfect"]:
            print(f"  {puzzle_id}: PERFECT")

    # Compute aggregate metrics
    aggregate = {
        "pixel_accuracy": total_correct / max(total_pixels, 1),
        "perfect_example_rate": total_perfect / max(total_test_examples, 1),
        "total_test_examples": total_test_examples,
        "total_perfect_examples": total_perfect,
        "num_puzzles_evaluated": len(all_results),
        "num_puzzles_all_perfect": sum(1 for r in all_results if r["all_perfect"]),
    }

    return {
        "aggregate": aggregate,
        "per_puzzle": per_puzzle_results,
    }


@dataclass
class PuzzleResult:
    puzzle_id: str
    num_train: int
    num_test: int
    train_time: float
    pixel_accuracy: float
    perfect_rate: float
    all_perfect: bool
    error: str = None
    augmentation_used: str = None  # NEW: "dihedral_only" or "full"
    augmentation_reason: str = None  # NEW: why this augmentation was chosen


def evaluate_puzzle(
    puzzle_id: str,
    puzzle: Dict,
    epochs: int,
    hidden_dim: int,
    num_negatives: int,
    dihedral_only: bool,
    negative_type: str = "mixed",
    visualize: bool = False,
    verbose: bool = False,
    auto_augment: bool = False,  # NEW
) -> PuzzleResult:
    """Train and evaluate on a single puzzle."""
    
    num_train = len(puzzle.get("train", []))
    test_examples = puzzle.get("test", [])
    num_test = sum(1 for ex in test_examples if "output" in ex)
    
    if num_train == 0:
        return PuzzleResult(puzzle_id, 0, 0, 0, 0, 0, False, "no_train")
    if num_test == 0:
        return PuzzleResult(puzzle_id, num_train, 0, 0, 0, 0, False, "no_test_outputs")
    
    # NEW: Auto-detect augmentation strategy
    if auto_augment:
        palette_analysis = analyze_puzzle_palettes(puzzle)
        use_dihedral_only = (palette_analysis["recommendation"] == "dihedral_only")
        augmentation_reason = palette_analysis["reason"]
        augmentation_used = "dihedral_only" if use_dihedral_only else "full"
        
        if verbose:
            print(f"  Palette analysis: {augmentation_used} ({augmentation_reason})")
            print(f"    Identical combined palettes: {palette_analysis['identical_combined_palettes']}")
            print(f"    Colors preserved: {palette_analysis['colors_preserved']}")
    else:
        use_dihedral_only = dihedral_only
        augmentation_used = "dihedral_only" if dihedral_only else "full"
        augmentation_reason = "manual"
    
    start = time.time()
    model = train_on_puzzle(
        puzzle, epochs=epochs, hidden_dim=hidden_dim,
        num_negatives=num_negatives, dihedral_only=use_dihedral_only,
        negative_type=negative_type,
        verbose=verbose
    )
    train_time = time.time() - start
    
    eval_results = evaluate_on_test(model, puzzle, puzzle_id=puzzle_id, 
                                     candidate_mode="zeros", visualize=visualize)
    
    if "error" in eval_results:
        return PuzzleResult(puzzle_id, num_train, num_test, train_time, 0, 0, False, 
                           eval_results["error"], augmentation_used, augmentation_reason)
    
    return PuzzleResult(
        puzzle_id=puzzle_id,
        num_train=num_train,
        num_test=num_test,
        train_time=train_time,
        pixel_accuracy=eval_results["pixel_accuracy"],
        perfect_rate=eval_results["perfect_rate"],
        all_perfect=eval_results["all_perfect"],
        augmentation_used=augmentation_used,
        augmentation_reason=augmentation_reason,
    )


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
            continue
        
        with open(challenges_path) as f:
            puzzles = json.load(f)
        
        if os.path.exists(solutions_path):
            with open(solutions_path) as f:
                solutions = json.load(f)
            for pid in puzzles:
                if pid in solutions:
                    for i, sol in enumerate(solutions[pid]):
                        if i < len(puzzles[pid]["test"]):
                            puzzles[pid]["test"][i]["output"] = sol
        
        all_puzzles.update(puzzles)
    
    return all_puzzles


def main():
    parser = argparse.ArgumentParser(description="Evaluate CNN generalization across ARC puzzles")
    parser.add_argument("--dataset", type=str, default="arc-agi-1", choices=["arc-agi-1", "arc-agi-2"])
    parser.add_argument("--data-root", type=str, default="kaggle/combined")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--num-negatives", type=int, default=8, 
                        help="Negatives per example (distributed across corruption types)")
    parser.add_argument("--max-puzzles", type=int, default=None, help="Max puzzles to evaluate (for quick testing)")
    parser.add_argument("--dihedral-only", action="store_true", help="Only dihedral augmentation (no color permutation)")
    parser.add_argument("--auto-augment", action="store_true",
                        help="NEW: Automatically detect color semantics and choose augmentation strategy")
    parser.add_argument("--multi-puzzle", action="store_true",
                        help="Train single CNN on all puzzles together (vs per-puzzle training)")
    parser.add_argument("--negative-type", type=str, default="mixed",
                        choices=["mixed", "all_zeros", "corrupted", "random_noise", 
                                 "constant_fill", "color_swap", "wrong_input", "mismatched_aug"],
                        help="Type of negatives: mixed=all types (default), or single type for ablation")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--visualize", action="store_true", 
                        help="Show visual comparison of predictions vs targets")
    parser.add_argument("--output", type=str, default="cnn_generalization_results.json")
    args = parser.parse_args()
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    print(f"Device: {DEVICE}")
    print(f"Dataset: {args.dataset}")
    print(f"Training mode: {'MULTI-PUZZLE (single CNN for all)' if args.multi_puzzle else 'PER-PUZZLE (one CNN each)'}")
    print(f"Epochs: {args.epochs}")

    if args.auto_augment:
        print(f"Augmentation: AUTO (palette-based detection)")
    else:
        print(f"Augmentation: {'dihedral-only' if args.dihedral_only else 'full'}")

    print(f"Num negatives: {args.num_negatives}")
    print(f"Negative type: {args.negative_type}")
    print(f"Evaluation: Blank candidate (model must generate from scratch)")
    
    # Load puzzles
    puzzles = load_puzzles(args.dataset, args.data_root)
    puzzle_ids = list(puzzles.keys())

    # Filter to puzzles with test outputs
    puzzle_ids = [pid for pid in puzzle_ids
                  if any("output" in ex for ex in puzzles[pid].get("test", []))]

    if args.max_puzzles:
        random.shuffle(puzzle_ids)
        puzzle_ids = puzzle_ids[:args.max_puzzles]
        # Also filter the puzzles dict for multi-puzzle mode
        puzzles = {pid: puzzles[pid] for pid in puzzle_ids}

    print(f"\nTotal puzzles: {len(puzzle_ids)}")
    print("="*60)

    # =========================================================================
    # MULTI-PUZZLE MODE: Train single CNN on all puzzles, evaluate on all tests
    # =========================================================================
    if args.multi_puzzle:
        print("\n[MULTI-PUZZLE MODE]")
        print("Training single CNN on all puzzles together...")

        start_time = time.time()
        model = train_on_all_puzzles(
            puzzles,
            epochs=args.epochs,
            hidden_dim=args.hidden_dim,
            num_negatives=args.num_negatives,
            auto_augment=args.auto_augment,
            dihedral_only=args.dihedral_only,
            verbose=args.verbose,
        )
        train_time = time.time() - start_time

        print(f"\nTraining completed in {train_time:.1f}s")
        print("\nEvaluating on all test examples...")

        eval_results = evaluate_all_tests(
            model, puzzles,
            visualize=args.visualize,
            verbose=args.verbose,
        )

        # Print summary
        agg = eval_results["aggregate"]
        print("\n" + "="*60)
        print("MULTI-PUZZLE RESULTS SUMMARY")
        print("="*60)
        print(f"Training time: {train_time:.1f}s")
        print(f"Puzzles evaluated: {agg['num_puzzles_evaluated']}")
        print(f"Total test examples: {agg['total_test_examples']}")
        print(f"Perfect test examples: {agg['total_perfect_examples']} ({100*agg['perfect_example_rate']:.1f}%)")
        print(f"Puzzles with all tests perfect: {agg['num_puzzles_all_perfect']} ({100*agg['num_puzzles_all_perfect']/max(agg['num_puzzles_evaluated'],1):.1f}%)")
        print(f"Overall pixel accuracy: {agg['pixel_accuracy']:.2%}")

        # Per-puzzle accuracy distribution
        per_puzzle = eval_results["per_puzzle"]
        valid_puzzle_accs = [p["pixel_accuracy"] for p in per_puzzle.values() if "pixel_accuracy" in p]

        if valid_puzzle_accs:
            print(f"\n{'─'*60}")
            print("Per-puzzle accuracy distribution:")
            bins = [(1.0, 1.0), (0.95, 1.0), (0.9, 0.95), (0.8, 0.9), (0.5, 0.8), (0.0, 0.5)]
            for lo, hi in bins:
                count = sum(1 for acc in valid_puzzle_accs if lo <= acc < hi or (lo == hi == 1.0 and acc == 1.0))
                pct = 100 * count / len(valid_puzzle_accs)
                label = f"{100*lo:.0f}%" if lo == hi else f"{100*lo:.0f}-{100*hi:.0f}%"
                bar = "█" * int(pct / 2)
                print(f"  {label:>10}: {count:>4} ({pct:>5.1f}%) {bar}")

        # List perfect puzzles
        perfect_puzzles = [pid for pid, p in per_puzzle.items()
                          if p.get("all_perfect", False)]
        if perfect_puzzles:
            print(f"\nPerfect puzzles ({len(perfect_puzzles)}):")
            for pid in sorted(perfect_puzzles)[:20]:
                print(f"  {pid}")
            if len(perfect_puzzles) > 20:
                print(f"  ... and {len(perfect_puzzles) - 20} more")

        # Save results
        output_data = {
            "config": vars(args),
            "mode": "multi-puzzle",
            "summary": {
                "train_time": train_time,
                "num_puzzles": agg["num_puzzles_evaluated"],
                "total_test_examples": agg["total_test_examples"],
                "perfect_examples": agg["total_perfect_examples"],
                "perfect_example_rate": agg["perfect_example_rate"],
                "puzzles_all_perfect": agg["num_puzzles_all_perfect"],
                "pixel_accuracy": agg["pixel_accuracy"],
            },
            "per_puzzle": eval_results["per_puzzle"],
        }

        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"\nResults saved to: {args.output}")

    # =========================================================================
    # PER-PUZZLE MODE: Train separate CNN for each puzzle (original behavior)
    # =========================================================================
    else:
        print("\n[PER-PUZZLE MODE]")
        print(f"Training separate CNN for each of {len(puzzle_ids)} puzzles...")

        results: List[PuzzleResult] = []

        for puzzle_id in tqdm(puzzle_ids, desc="Puzzles"):
            result = evaluate_puzzle(
                puzzle_id, puzzles[puzzle_id],
                epochs=args.epochs,
                hidden_dim=args.hidden_dim,
                num_negatives=args.num_negatives,
                dihedral_only=args.dihedral_only,
                negative_type=args.negative_type,
                visualize=args.visualize,
                verbose=args.verbose,
                auto_augment=args.auto_augment,
            )
            results.append(result)

            if args.verbose or result.all_perfect:
                status = "✓ PERFECT" if result.all_perfect else f"✗ {result.pixel_accuracy:.1%}"
                aug_info = f" [{result.augmentation_used}]" if args.auto_augment else ""
                print(f"  {puzzle_id}: {status} ({result.train_time:.1f}s){aug_info}")

        # Summary statistics
        valid_results = [r for r in results if r.error is None]
        perfect_results = [r for r in valid_results if r.all_perfect]
        high_acc_results = [r for r in valid_results if r.pixel_accuracy >= 0.95]

        print("\n" + "="*60)
        print("PER-PUZZLE RESULTS SUMMARY")
        print("="*60)
        print(f"Total puzzles evaluated: {len(puzzle_ids)}")
        print(f"Valid evaluations: {len(valid_results)}")
        print(f"Perfect generalization (100%): {len(perfect_results)} ({100*len(perfect_results)/max(len(valid_results),1):.1f}%)")
        print(f"High accuracy (≥95%): {len(high_acc_results)} ({100*len(high_acc_results)/max(len(valid_results),1):.1f}%)")

        avg_accuracy = 0
        avg_time = 0
        if valid_results:
            avg_accuracy = sum(r.pixel_accuracy for r in valid_results) / len(valid_results)
            avg_time = sum(r.train_time for r in valid_results) / len(valid_results)
            print(f"Average pixel accuracy: {avg_accuracy:.2%}")
            print(f"Average training time: {avg_time:.1f}s")

        # Augmentation strategy breakdown (if auto_augment)
        if args.auto_augment:
            dihedral_only_results = [r for r in valid_results if r.augmentation_used == "dihedral_only"]
            full_aug_results = [r for r in valid_results if r.augmentation_used == "full"]

            print(f"\n{'─'*60}")
            print("AUGMENTATION STRATEGY BREAKDOWN")
            print(f"{'─'*60}")
            print(f"Dihedral-only (identical palettes): {len(dihedral_only_results)} puzzles")
            if dihedral_only_results:
                dihedral_perfect = sum(1 for r in dihedral_only_results if r.all_perfect)
                dihedral_avg_acc = sum(r.pixel_accuracy for r in dihedral_only_results) / len(dihedral_only_results)
                print(f"  Perfect: {dihedral_perfect} ({100*dihedral_perfect/len(dihedral_only_results):.1f}%)")
                print(f"  Avg accuracy: {dihedral_avg_acc:.2%}")

            print(f"\nFull augmentation (varying palettes): {len(full_aug_results)} puzzles")
            if full_aug_results:
                full_perfect = sum(1 for r in full_aug_results if r.all_perfect)
                full_avg_acc = sum(r.pixel_accuracy for r in full_aug_results) / len(full_aug_results)
                print(f"  Perfect: {full_perfect} ({100*full_perfect/len(full_aug_results):.1f}%)")
                print(f"  Avg accuracy: {full_avg_acc:.2%}")

            # Breakdown by reason
            reasons = {}
            for r in valid_results:
                reason = r.augmentation_reason or "unknown"
                if reason not in reasons:
                    reasons[reason] = []
                reasons[reason].append(r)

            print(f"\nBy detection reason:")
            for reason, rs in sorted(reasons.items(), key=lambda x: -len(x[1])):
                perfect_count = sum(1 for r in rs if r.all_perfect)
                avg_acc = sum(r.pixel_accuracy for r in rs) / len(rs)
                print(f"  {reason}: {len(rs)} puzzles, {perfect_count} perfect ({avg_acc:.1%} avg)")

        # Accuracy distribution
        print(f"\n{'─'*60}")
        print("Accuracy distribution:")
        bins = [(1.0, 1.0), (0.95, 1.0), (0.9, 0.95), (0.8, 0.9), (0.5, 0.8), (0.0, 0.5)]
        for lo, hi in bins:
            count = sum(1 for r in valid_results if lo <= r.pixel_accuracy < hi or (lo == hi == 1.0 and r.pixel_accuracy == 1.0))
            pct = 100 * count / max(len(valid_results), 1)
            label = f"{100*lo:.0f}%" if lo == hi else f"{100*lo:.0f}-{100*hi:.0f}%"
            bar = "█" * int(pct / 2)
            print(f"  {label:>10}: {count:>4} ({pct:>5.1f}%) {bar}")

        # List perfect puzzles
        if perfect_results:
            print(f"\nPerfect generalization puzzles ({len(perfect_results)}):")
            for r in sorted(perfect_results, key=lambda x: x.puzzle_id)[:20]:
                aug_info = f" [{r.augmentation_used}]" if args.auto_augment else ""
                print(f"  {r.puzzle_id} (train={r.num_train}, test={r.num_test}, time={r.train_time:.1f}s){aug_info}")
            if len(perfect_results) > 20:
                print(f"  ... and {len(perfect_results) - 20} more")

        # Save results
        output_data = {
            "config": vars(args),
            "mode": "per-puzzle",
            "summary": {
                "total_puzzles": len(puzzle_ids),
                "valid_evaluations": len(valid_results),
                "perfect_count": len(perfect_results),
                "perfect_rate": len(perfect_results) / max(len(valid_results), 1),
                "high_acc_count": len(high_acc_results),
                "avg_pixel_accuracy": avg_accuracy,
                "avg_train_time": avg_time,
            },
            "results": [
                {
                    "puzzle_id": r.puzzle_id,
                    "num_train": r.num_train,
                    "num_test": r.num_test,
                    "train_time": r.train_time,
                    "pixel_accuracy": r.pixel_accuracy,
                    "perfect_rate": r.perfect_rate,
                    "all_perfect": r.all_perfect,
                    "error": r.error,
                    "augmentation_used": r.augmentation_used,
                    "augmentation_reason": r.augmentation_reason,
                }
                for r in results
            ]
        }

        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()