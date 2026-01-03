#!/usr/bin/env python3
"""
TRM-style Correspondence Predictor

Instead of predicting each object's transformation independently,
this processes ALL objects together with iterative self-attention,
allowing objects to condition on each other's predicted positions.

Key difference from PixelTransformerPredictor:
- All objects in a puzzle are processed as a sequence
- Multiple iterations of self-attention let position estimates refine
- Object N can "see" where objects 1..N-1 are predicted to go

Usage:
    python trm_correspondence.py --data-root kaggle/combined --puzzle-id <id>
"""

import argparse
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# =============================================================================
# Constants
# =============================================================================

MAX_OBJECTS = 16  # Maximum objects per puzzle
POSITION_DIM = 4  # (center_y, center_x, height, width) normalized
FEATURE_DIM = 32  # Object feature dimension

# ANSI color codes for console output (256-color mode)
ANSI_COLORS = [
    16,   # 0: black
    27,   # 1: blue
    196,  # 2: red
    46,   # 3: green
    226,  # 4: yellow
    248,  # 5: grey
    201,  # 6: magenta
    208,  # 7: orange
    51,   # 8: cyan
    88,   # 9: brown/maroon
]

COLOR_NAMES = ['blk', 'blu', 'red', 'grn', 'yel', 'gry', 'mag', 'org', 'cyn', 'brn']


def grid_to_console(grid: np.ndarray, use_color: bool = True, use_unicode: bool = True) -> str:
    """Convert a color grid to console-printable string."""
    H, W = grid.shape
    lines = []

    for row in range(H):
        line_parts = []
        for col in range(W):
            val = grid[row, col]

            if val >= 10 or val < 0:  # Padding or invalid
                char = '·' if use_unicode else '.'
                if use_color:
                    line_parts.append(f'\033[38;5;240m{char}\033[0m')
                else:
                    line_parts.append(char)
            else:
                char = '██' if use_unicode else str(val)
                if use_color:
                    ansi_code = ANSI_COLORS[val]
                    line_parts.append(f'\033[38;5;{ansi_code}m{char}\033[0m')
                else:
                    line_parts.append(char)

        lines.append(''.join(line_parts))

    return '\n'.join(lines)


# =============================================================================
# Augmentation Functions
# =============================================================================

def dihedral_transform(arr: np.ndarray, tid: int) -> np.ndarray:
    """Apply one of 8 dihedral symmetries."""
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
    """Create random color permutation (keeping 0/black fixed)."""
    mapping = np.concatenate([
        np.array([0], dtype=np.int64),
        np.random.permutation(np.arange(1, 10, dtype=np.int64))
    ])
    return mapping


def apply_augmentation(grid: np.ndarray, trans_id: int, color_map: np.ndarray) -> np.ndarray:
    """Apply dihedral transform and color permutation."""
    return dihedral_transform(color_map[grid], trans_id)


def get_random_augmentation() -> Tuple[int, np.ndarray]:
    """Get random augmentation parameters."""
    trans_id = np.random.randint(0, 8)
    color_map = random_color_permutation()
    return trans_id, color_map


# =============================================================================
# Object Feature Extraction (simplified from your slot_viz.py)
# =============================================================================

def extract_object_features(mask: np.ndarray, color_grid: np.ndarray, 
                            grid_h: int, grid_w: int) -> np.ndarray:
    """Extract features for a single object.
    
    Returns:
        (FEATURE_DIM,) array with:
        - Normalized position (center_y, center_x, height, width) [4]
        - Color histogram [10]
        - Shape features (area, aspect_ratio, density, compactness) [4]
        - Padding [14]
    """
    features = np.zeros(FEATURE_DIM, dtype=np.float32)
    
    if not mask.any():
        return features
    
    rows, cols = np.where(mask)
    
    # Bounding box
    min_r, max_r = rows.min(), rows.max()
    min_c, max_c = cols.min(), cols.max()
    height = max_r - min_r + 1
    width = max_c - min_c + 1
    
    # Normalized position (0-1 range)
    center_y = (min_r + max_r) / 2 / max(grid_h - 1, 1)
    center_x = (min_c + max_c) / 2 / max(grid_w - 1, 1)
    norm_h = height / grid_h
    norm_w = width / grid_w
    
    features[0:4] = [center_y, center_x, norm_h, norm_w]
    
    # Color histogram
    colors_in_mask = color_grid[mask]
    for c in range(10):
        features[4 + c] = (colors_in_mask == c).sum() / len(colors_in_mask)
    
    # Shape features
    area = mask.sum()
    bbox_area = height * width
    density = area / bbox_area if bbox_area > 0 else 0
    aspect_ratio = min(width, height) / max(width, height) if max(width, height) > 0 else 1
    compactness = area / (height * width) if height * width > 0 else 0
    
    features[14] = np.log1p(area) / 5.0
    features[15] = aspect_ratio
    features[16] = density
    features[17] = compactness
    
    return features


# =============================================================================
# Dataset: Puzzle-level (all objects together)
# =============================================================================

@dataclass
class PuzzleObjectData:
    """All objects from one training example."""
    puzzle_id: str
    example_idx: int
    num_objects: int
    input_features: np.ndarray   # (MAX_OBJECTS, FEATURE_DIM)
    output_positions: np.ndarray  # (MAX_OBJECTS, 4) - target positions
    object_mask: np.ndarray       # (MAX_OBJECTS,) - which slots are valid
    input_grid: np.ndarray        # Original input grid for context
    output_grid: np.ndarray       # Original output grid


class PuzzleLevelDataset(Dataset):
    """Dataset where each item is ALL objects from one example."""

    def __init__(self, puzzles: Dict, device: torch.device,
                 augment: bool = True,
                 num_augments: int = 8,
                 dihedral_only: bool = False,
                 color_only: bool = False):
        self.device = device
        self.augment = augment
        self.num_augments = num_augments if augment else 1
        self.dihedral_only = dihedral_only
        self.color_only = color_only
        self.examples: List[PuzzleObjectData] = []

        print("Extracting puzzle-level object data...")
        for puzzle_id, puzzle in tqdm(puzzles.items()):
            for ex_idx, example in enumerate(puzzle.get('train', [])):
                # Extract original example
                data = self._extract_example(example, puzzle_id, ex_idx)
                if data is not None and data.num_objects > 0:
                    self.examples.append(data)

                # Extract augmented versions
                if self.augment:
                    for aug_idx in range(self.num_augments):
                        aug_example = self._augment_example(example)
                        aug_data = self._extract_example(aug_example, puzzle_id, ex_idx)
                        if aug_data is not None and aug_data.num_objects > 0:
                            self.examples.append(aug_data)

        print(f"Extracted {len(self.examples)} examples")
        if self.augment:
            print(f"  (with {self.num_augments} augmentations per example)")

    def _get_augmentation(self) -> Tuple[int, np.ndarray]:
        """Get augmentation params based on settings."""
        if not self.augment:
            return 0, np.arange(10, dtype=np.int64)
        elif self.dihedral_only:
            # Random dihedral transform, identity color mapping
            trans_id = np.random.randint(0, 8)
            color_map = np.arange(10, dtype=np.int64)
            return trans_id, color_map
        elif self.color_only:
            # Identity dihedral transform, random color mapping
            trans_id = 0
            color_map = random_color_permutation()
            return trans_id, color_map
        else:
            return get_random_augmentation()

    def _augment_example(self, example: Dict) -> Dict:
        """Apply the same random augmentation to both input and output grids."""
        input_grid = np.array(example['input'], dtype=np.int64)
        output_grid = np.array(example['output'], dtype=np.int64)

        # Get random augmentation (same for both input and output)
        trans_id, color_map = self._get_augmentation()

        # Apply augmentation to both grids
        aug_input = apply_augmentation(input_grid, trans_id, color_map)
        aug_output = apply_augmentation(output_grid, trans_id, color_map)

        return {
            'input': aug_input.tolist(),
            'output': aug_output.tolist(),
        }
    
    def _extract_example(self, example: Dict, puzzle_id: str, 
                         ex_idx: int) -> Optional[PuzzleObjectData]:
        """Extract all objects from one input/output pair."""
        input_grid = np.array(example['input'], dtype=np.int64)
        output_grid = np.array(example['output'], dtype=np.int64)
        
        in_h, in_w = input_grid.shape
        out_h, out_w = output_grid.shape
        
        # Simple object extraction: connected components of non-black pixels
        input_objects = self._extract_objects(input_grid)
        output_objects = self._extract_objects(output_grid)
        
        if len(input_objects) == 0 or len(output_objects) == 0:
            return None
        
        # Match objects by color (simplified correspondence)
        correspondences = self._match_by_color(
            input_objects, output_objects, input_grid, output_grid
        )
        
        if len(correspondences) == 0:
            return None
        
        # Build feature arrays
        num_objects = min(len(correspondences), MAX_OBJECTS)
        input_features = np.zeros((MAX_OBJECTS, FEATURE_DIM), dtype=np.float32)
        output_positions = np.zeros((MAX_OBJECTS, 4), dtype=np.float32)
        object_mask = np.zeros(MAX_OBJECTS, dtype=np.float32)
        
        for i, (in_mask, out_mask) in enumerate(correspondences[:MAX_OBJECTS]):
            input_features[i] = extract_object_features(in_mask, input_grid, in_h, in_w)
            
            # Extract output position
            out_rows, out_cols = np.where(out_mask)
            if len(out_rows) > 0:
                min_r, max_r = out_rows.min(), out_rows.max()
                min_c, max_c = out_cols.min(), out_cols.max()
                center_y = (min_r + max_r) / 2 / max(out_h - 1, 1)
                center_x = (min_c + max_c) / 2 / max(out_w - 1, 1)
                norm_h = (max_r - min_r + 1) / out_h
                norm_w = (max_c - min_c + 1) / out_w
                output_positions[i] = [center_y, center_x, norm_h, norm_w]
            
            object_mask[i] = 1.0
        
        return PuzzleObjectData(
            puzzle_id=puzzle_id,
            example_idx=ex_idx,
            num_objects=num_objects,
            input_features=input_features,
            output_positions=output_positions,
            object_mask=object_mask,
            input_grid=input_grid,
            output_grid=output_grid,
        )
    
    def _extract_objects(self, grid: np.ndarray) -> List[np.ndarray]:
        """Extract object masks using connected components."""
        from scipy.ndimage import label
        
        non_black = grid != 0
        labeled, num_features = label(non_black)
        
        masks = []
        for i in range(1, num_features + 1):
            mask = labeled == i
            if mask.sum() > 0:
                masks.append(mask)
        
        return masks
    
    def _match_by_color(self, input_objects: List[np.ndarray],
                        output_objects: List[np.ndarray],
                        input_grid: np.ndarray,
                        output_grid: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Match input/output objects by dominant color."""
        def dominant_color(mask, grid):
            colors = grid[mask]
            if len(colors) == 0:
                return -1
            counts = np.bincount(colors, minlength=10)
            return counts.argmax()
        
        in_colors = [dominant_color(m, input_grid) for m in input_objects]
        out_colors = [dominant_color(m, output_grid) for m in output_objects]
        
        correspondences = []
        used_out = set()
        
        for i, in_mask in enumerate(input_objects):
            for j, out_mask in enumerate(output_objects):
                if j not in used_out and in_colors[i] == out_colors[j]:
                    correspondences.append((in_mask, out_mask))
                    used_out.add(j)
                    break
        
        # Sort by input x-position (left to right)
        def get_x(pair):
            rows, cols = np.where(pair[0])
            return cols.min() if len(cols) > 0 else 0
        
        correspondences.sort(key=get_x)
        
        return correspondences
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        ex = self.examples[idx]
        return {
            'input_features': torch.from_numpy(ex.input_features),
            'output_positions': torch.from_numpy(ex.output_positions),
            'object_mask': torch.from_numpy(ex.object_mask),
            'num_objects': ex.num_objects,
            'puzzle_id': ex.puzzle_id,
        }


# =============================================================================
# TRM-style Model: Iterative Self-Attention
# =============================================================================

class ObjectTRM(nn.Module):
    """
    TRM-style model for object position prediction.
    
    Key features:
    - All objects processed together as a sequence
    - Multiple iterations of self-attention
    - Position estimates refined each iteration
    - Objects can condition on each other's evolving predictions
    """
    
    def __init__(self, 
                 feature_dim: int = FEATURE_DIM,
                 hidden_dim: int = 32,
                 num_heads: int = 4,
                 num_iterations: int = 4,
                 num_layers: int = 2,
                 dropout: float = 0.1):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_iterations = num_iterations
        
        # Input projection
        self.input_proj = nn.Linear(feature_dim, hidden_dim)
        
        # Learnable initial hidden state
        self.h_init = nn.Parameter(torch.randn(hidden_dim) * 0.02)
        
        # Position embedding (inject current position estimate back in)
        self.pos_proj = nn.Linear(4, hidden_dim)
        
        # Transformer layers (shared across iterations)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output head: predict position delta (residual prediction)
        self.output_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 4),  # (dy, dx, dh, dw)
        )
        
        # Initialize output to small values (start near identity)
        nn.init.zeros_(self.output_head[-1].bias)
        nn.init.normal_(self.output_head[-1].weight, std=0.01)
    
    def forward(self, input_features: torch.Tensor, 
                object_mask: torch.Tensor,
                return_all_iterations: bool = False) -> torch.Tensor:
        """
        Args:
            input_features: (B, MAX_OBJECTS, FEATURE_DIM)
            object_mask: (B, MAX_OBJECTS) - 1 for valid objects, 0 for padding
            
        Returns:
            positions: (B, MAX_OBJECTS, 4) predicted output positions
        """
        B, N, _ = input_features.shape
        
        # Create attention mask (True = ignore)
        attn_mask = (object_mask == 0)  # (B, N)
        
        # Initial hidden state
        h = self.input_proj(input_features)  # (B, N, hidden_dim)
        h = h + self.h_init.view(1, 1, -1)
        
        # Current position estimate (start with input positions)
        # Input features[0:4] contains (center_y, center_x, h, w)
        current_pos = input_features[:, :, 0:4].clone()  # (B, N, 4)
        
        all_positions = [current_pos]
        
        # Iterative refinement
        for iteration in range(self.num_iterations):
            # Inject current position estimate
            pos_embed = self.pos_proj(current_pos)
            h_with_pos = h + pos_embed
            
            # Self-attention across all objects
            # Key: objects can see each other's current state
            h_out = self.transformer(
                h_with_pos,
                src_key_padding_mask=attn_mask
            )
            
            # Residual connection on hidden state
            h = h + h_out
            
            # Predict position delta
            delta = self.output_head(h)  # (B, N, 4)
            
            # Update position estimate (residual)
            current_pos = current_pos + delta
            
            all_positions.append(current_pos)
        
        if return_all_iterations:
            return torch.stack(all_positions, dim=1)  # (B, num_iter+1, N, 4)
        
        return current_pos


# =============================================================================
# Training
# =============================================================================

def train_epoch(model: ObjectTRM, dataloader: DataLoader, 
                optimizer: torch.optim.Optimizer, device: torch.device) -> Dict:
    model.train()
    total_loss = 0
    total_samples = 0
    
    for batch in dataloader:
        input_features = batch['input_features'].to(device)
        output_positions = batch['output_positions'].to(device)
        object_mask = batch['object_mask'].to(device)
        
        optimizer.zero_grad()
        
        # Forward
        pred_positions = model(input_features, object_mask)
        
        # Loss: MSE on valid objects only
        mask = object_mask.unsqueeze(-1)  # (B, N, 1)
        loss = F.mse_loss(pred_positions * mask, output_positions * mask, reduction='sum')
        loss = loss / (mask.sum() * 4 + 1e-8)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * input_features.shape[0]
        total_samples += input_features.shape[0]
    
    return {'loss': total_loss / total_samples}


def evaluate(model: ObjectTRM, dataloader: DataLoader, device: torch.device) -> Dict:
    model.eval()
    total_loss = 0
    total_position_error = 0
    total_objects = 0
    
    with torch.no_grad():
        for batch in dataloader:
            input_features = batch['input_features'].to(device)
            output_positions = batch['output_positions'].to(device)
            object_mask = batch['object_mask'].to(device)
            
            pred_positions = model(input_features, object_mask)
            
            mask = object_mask.unsqueeze(-1)
            loss = F.mse_loss(pred_positions * mask, output_positions * mask, reduction='sum')
            loss = loss / (mask.sum() * 4 + 1e-8)
            
            # Position error (just center, ignoring size)
            center_error = torch.sqrt(
                ((pred_positions[:, :, 0:2] - output_positions[:, :, 0:2]) ** 2).sum(dim=-1)
            )
            center_error = (center_error * object_mask).sum()
            
            total_loss += loss.item() * input_features.shape[0]
            total_position_error += center_error.item()
            total_objects += object_mask.sum().item()
    
    n = len(dataloader.dataset)
    return {
        'loss': total_loss / n,
        'center_error': total_position_error / total_objects if total_objects > 0 else 0,
    }


def visualize_predictions(model: ObjectTRM, dataset: PuzzleLevelDataset, 
                          device: torch.device, num_examples: int = 3):
    """Print text visualization of predictions vs targets."""
    model.eval()
    
    indices = np.random.choice(len(dataset), min(num_examples, len(dataset)), replace=False)
    
    for idx in indices:
        item = dataset[idx]
        input_features = item['input_features'].unsqueeze(0).to(device)
        object_mask = item['object_mask'].unsqueeze(0).to(device)
        target_positions = item['output_positions'].numpy()
        num_objects = item['num_objects']
        puzzle_id = item['puzzle_id']
        
        with torch.no_grad():
            # Get all iterations
            all_pos = model(input_features, object_mask, return_all_iterations=True)
            all_pos = all_pos.squeeze(0).cpu().numpy()  # (num_iter+1, N, 4)
        
        print(f"\n{'='*60}")
        print(f"Puzzle: {puzzle_id} | Objects: {num_objects}")
        print(f"{'='*60}")
        
        input_pos = item['input_features'][:, 0:4].numpy()
        
        for obj_idx in range(num_objects):
            in_y, in_x = input_pos[obj_idx, 0:2]
            tgt_y, tgt_x = target_positions[obj_idx, 0:2]
            
            print(f"\nObject {obj_idx}:")
            print(f"  Input:  ({in_y:.3f}, {in_x:.3f})")
            print(f"  Target: ({tgt_y:.3f}, {tgt_x:.3f})")
            print(f"  Iterations:")
            
            for it in range(all_pos.shape[0]):
                pred_y, pred_x = all_pos[it, obj_idx, 0:2]
                err = np.sqrt((pred_y - tgt_y)**2 + (pred_x - tgt_x)**2)
                marker = "  " if it == 0 else "->"
                print(f"    {marker} iter {it}: ({pred_y:.3f}, {pred_x:.3f})  err={err:.4f}")


def visualize_test_prediction(model: ObjectTRM, puzzle: Dict, puzzle_id: str,
                               device: torch.device):
    """
    Visualize model predictions on the test input.

    Trains on training pairs, then predicts where objects in test input should go.
    """
    from scipy.ndimage import label

    model.eval()

    # Get test example
    test_examples = puzzle.get('test', [])
    if not test_examples:
        print("No test examples found")
        return

    test_example = test_examples[0]
    test_input = np.array(test_example.get('input', []), dtype=np.int64)
    test_output = np.array(test_example.get('output', [])) if 'output' in test_example else None

    if test_input.size == 0:
        print("No test input found")
        return

    H_in, W_in = test_input.shape

    print("\n" + "=" * 60)
    print(f"TEST PREDICTION: {puzzle_id}")
    print("=" * 60)

    # Print color legend
    print("\nColor Legend:")
    legend_parts = []
    for i, name in enumerate(COLOR_NAMES):
        ansi_code = ANSI_COLORS[i]
        legend_parts.append(f'\033[38;5;{ansi_code}m{i}={name}\033[0m')
    print("  " + "  ".join(legend_parts))

    # Print test input
    print(f"\n{'─' * 40}")
    print(f"TEST INPUT ({H_in}x{W_in}):")
    print('─' * 40)
    print(grid_to_console(test_input))

    # Extract objects from test input
    non_black = test_input != 0
    labeled, num_features = label(non_black)

    input_objects = []
    for i in range(1, num_features + 1):
        mask = labeled == i
        if mask.sum() > 0:
            input_objects.append(mask)

    if len(input_objects) == 0:
        print("\nNo objects found in test input")
        return

    print(f"\nFound {len(input_objects)} object(s) in test input")

    # Sort by x-position (left to right) - same as training
    def get_x(mask):
        rows, cols = np.where(mask)
        return cols.min() if len(cols) > 0 else 0
    input_objects.sort(key=get_x)

    # Build feature array for test input
    num_objects = min(len(input_objects), MAX_OBJECTS)
    input_features = np.zeros((MAX_OBJECTS, FEATURE_DIM), dtype=np.float32)
    object_mask = np.zeros(MAX_OBJECTS, dtype=np.float32)

    # Store original object info for rendering
    object_info = []

    for i, mask in enumerate(input_objects[:MAX_OBJECTS]):
        input_features[i] = extract_object_features(mask, test_input, H_in, W_in)
        object_mask[i] = 1.0

        # Store object pixels and colors for rendering
        rows, cols = np.where(mask)
        colors = test_input[mask]
        min_r, max_r = rows.min(), rows.max()
        min_c, max_c = cols.min(), cols.max()

        # Relative positions within object's bounding box
        rel_rows = rows - min_r
        rel_cols = cols - min_c
        obj_h = max_r - min_r + 1
        obj_w = max_c - min_c + 1

        object_info.append({
            'rel_rows': rel_rows,
            'rel_cols': rel_cols,
            'colors': colors,
            'height': obj_h,
            'width': obj_w,
            'input_center_y': (min_r + max_r) / 2,
            'input_center_x': (min_c + max_c) / 2,
        })

    # Run model prediction
    with torch.no_grad():
        input_tensor = torch.from_numpy(input_features).unsqueeze(0).to(device)
        mask_tensor = torch.from_numpy(object_mask).unsqueeze(0).to(device)

        pred_positions = model(input_tensor, mask_tensor)
        pred_positions = pred_positions.squeeze(0).cpu().numpy()

    # Determine output grid size
    # Use test output size if available, otherwise estimate from predictions
    if test_output is not None:
        H_out, W_out = test_output.shape
    else:
        # Estimate from predictions - use same as input for now
        H_out, W_out = H_in, W_in

    # Render predicted output
    predicted_grid = np.zeros((H_out, W_out), dtype=np.int64)

    print(f"\n{'─' * 40}")
    print("OBJECT PREDICTIONS:")
    print('─' * 40)

    for i in range(num_objects):
        pred_y, pred_x, pred_h, pred_w = pred_positions[i]
        info = object_info[i]

        # Convert normalized predictions to pixel coordinates
        pred_center_y = pred_y * (H_out - 1)
        pred_center_x = pred_x * (W_out - 1)

        # Calculate top-left corner for placing the object
        top_left_y = int(round(pred_center_y - info['height'] / 2))
        top_left_x = int(round(pred_center_x - info['width'] / 2))

        print(f"\nObject {i}:")
        print(f"  Input pos:  ({info['input_center_y']:.1f}, {info['input_center_x']:.1f})")
        print(f"  Pred pos:   ({pred_center_y:.1f}, {pred_center_x:.1f})")
        print(f"  Size: {info['height']}x{info['width']}")

        # Place object pixels in predicted grid
        for rel_r, rel_c, color in zip(info['rel_rows'], info['rel_cols'], info['colors']):
            out_r = top_left_y + rel_r
            out_c = top_left_x + rel_c
            if 0 <= out_r < H_out and 0 <= out_c < W_out:
                predicted_grid[out_r, out_c] = color

    # Display predicted output
    print(f"\n{'─' * 40}")
    print(f"PREDICTED OUTPUT ({H_out}x{W_out}):")
    print('─' * 40)
    print(grid_to_console(predicted_grid))

    # Display expected output if available
    if test_output is not None:
        print(f"\n{'─' * 40}")
        print(f"EXPECTED OUTPUT ({test_output.shape[0]}x{test_output.shape[1]}):")
        print('─' * 40)
        print(grid_to_console(test_output))

        # Calculate accuracy
        if predicted_grid.shape == test_output.shape:
            correct = (predicted_grid == test_output).sum()
            total = test_output.size
            accuracy = correct / total * 100
            print(f"\nPixel accuracy: {correct}/{total} ({accuracy:.1f}%)")
    else:
        print("\n(No expected output available for comparison)")

    print("\n" + "=" * 60)


# =============================================================================
# Main
# =============================================================================

def load_puzzles(data_root: str) -> Dict:
    """Load all puzzles from the ARC dataset."""
    puzzles = {}

    subsets = [
        ("training", "training"),
        ("evaluation", "evaluation"),
    ]

    for subset_name, subset_key in subsets:
        challenges_path = f"{data_root}/arc-agi_{subset_key}_challenges.json"
        solutions_path = f"{data_root}/arc-agi_{subset_key}_solutions.json"

        if not os.path.exists(challenges_path):
            continue

        with open(challenges_path) as f:
            subset_puzzles = json.load(f)

        # Load solutions if available
        solutions = {}
        if os.path.exists(solutions_path):
            with open(solutions_path) as f:
                solutions = json.load(f)

        for puzzle_id, puzzle in subset_puzzles.items():
            # Add solutions to test examples
            if puzzle_id in solutions:
                for i, sol in enumerate(solutions[puzzle_id]):
                    if i < len(puzzle.get("test", [])):
                        puzzle["test"][i]["output"] = sol
            puzzles[puzzle_id] = puzzle

    return puzzles


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, default="kaggle/combined")
    parser.add_argument("--puzzle-id", type=str, default=None,
                        help="Single puzzle to train on (for debugging)")
    parser.add_argument("--max-puzzles", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--num-iterations", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=4)

    # Augmentation args
    parser.add_argument("--no-augment", action="store_true",
                        help="Disable augmentations (use raw example grids only)")
    parser.add_argument("--num-augments", type=int, default=50,
                        help="Number of augmented versions per example pair")
    parser.add_argument("--dihedral-only", action="store_true",
                        help="Only apply dihedral transforms (no color permutation)")
    parser.add_argument("--color-only", action="store_true",
                        help="Only apply color permutation (no dihedral transforms)")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load puzzles
    puzzles = load_puzzles(args.data_root)
    print(f"Loaded {len(puzzles)} puzzles")
    
    if args.puzzle_id:
        if args.puzzle_id not in puzzles:
            print(f"Puzzle {args.puzzle_id} not found!")
            return
        puzzles = {args.puzzle_id: puzzles[args.puzzle_id]}
    elif args.max_puzzles:
        puzzle_ids = list(puzzles.keys())[:args.max_puzzles]
        puzzles = {k: puzzles[k] for k in puzzle_ids}
    
    # Create dataset with augmentation
    use_augment = not args.no_augment
    dataset = PuzzleLevelDataset(
        puzzles, device,
        augment=use_augment,
        num_augments=args.num_augments,
        dihedral_only=args.dihedral_only,
        color_only=args.color_only,
    )

    if len(dataset) == 0:
        print("No valid examples found!")
        return
    
    # Split
    n = len(dataset)
    indices = np.random.permutation(n)
    train_size = int(0.8 * n)
    
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    
    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    
    # Create model
    model = ObjectTRM(
        hidden_dim=args.hidden_dim,
        num_iterations=args.num_iterations,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    
    best_val_loss = float('inf')
    
    for epoch in range(1, args.epochs + 1):
        train_metrics = train_epoch(model, train_loader, optimizer, device)
        val_metrics = evaluate(model, val_loader, device)
        scheduler.step()
        
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            marker = " *"
        else:
            marker = ""
        
        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d} | Train Loss: {train_metrics['loss']:.4f} | "
                  f"Val Loss: {val_metrics['loss']:.4f} | "
                  f"Val Center Err: {val_metrics['center_error']:.4f}{marker}")
    
    print(f"\nBest val loss: {best_val_loss:.4f}")

    # Visualize some predictions on training data
    print("\n" + "="*60)
    print("SAMPLE PREDICTIONS (TRAINING DATA)")
    print("="*60)
    visualize_predictions(model, dataset, device, num_examples=3)

    # Visualize test predictions
    for puzzle_id, puzzle in puzzles.items():
        visualize_test_prediction(model, puzzle, puzzle_id, device)


if __name__ == "__main__":
    main()