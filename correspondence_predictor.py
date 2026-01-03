#!/usr/bin/env python3
"""
Correspondence Predictor

Uses the CRM architecture (DorsalCNN + SlotRoutedCrossAttention + VentralCNN)
to learn and predict shape correspondences between input and output grids.

The model learns to transform input shapes into output shapes by:
1. Extracting correspondences from training examples using slot-based matching
2. Training a ShapeDorsalCNN to predict output shape colors from input shapes
3. Using SlotRoutedCrossAttention to attend to the full input grid context

Usage:
    python correspondence_predictor.py --data-root kaggle/combined --epochs 100
"""

import argparse
import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from crm import (
    VentralCNN, AffinitySlotAttention, SlotRoutedCrossAttention,
    NUM_COLORS, get_conv_block, Down, Up, UpNoSkip,
    dihedral_transform, random_color_permutation, apply_augmentation, get_random_augmentation
)


# =============================================================================
# Constants
# =============================================================================

SHAPE_SIZE = 16  # Fixed canvas size for shape crops
GRID_SIZE = 30   # Max ARC grid size
PADDING_VALUE = 10  # Sentinel value for padding (color 10 = invalid)


# =============================================================================
# Core Utilities
# =============================================================================

def extract_shape_crop(grid: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    """Extract bounding box crop of a shape, centered in SHAPE_SIZE canvas.

    Args:
        grid: (H, W) integer array with color values 0-9
        mask: (H, W) binary mask for the shape

    Returns:
        padded_crop: (SHAPE_SIZE, SHAPE_SIZE) array with shape centered, padding=10
        bbox: (min_row, min_col, max_row, max_col) original bounding box
    """
    # Find bounding box
    rows, cols = np.where(mask > 0)
    if len(rows) == 0:
        # Empty mask
        return np.full((SHAPE_SIZE, SHAPE_SIZE), PADDING_VALUE, dtype=np.uint8), (0, 0, 0, 0)

    min_r, max_r = rows.min(), rows.max()
    min_c, max_c = cols.min(), cols.max()

    # Extract crop from grid
    crop = grid[min_r:max_r+1, min_c:max_c+1].copy()
    crop_mask = mask[min_r:max_r+1, min_c:max_c+1]

    # Zero out pixels outside the shape mask (use black as background)
    crop = crop.astype(np.uint8)
    crop[~crop_mask.astype(bool)] = 0

    # Create padded canvas and center the crop
    h, w = crop.shape

    # Handle shapes larger than SHAPE_SIZE by resizing
    if h > SHAPE_SIZE or w > SHAPE_SIZE:
        # Scale to fit while preserving aspect ratio
        scale = min(SHAPE_SIZE / h, SHAPE_SIZE / w)
        new_h, new_w = int(h * scale), int(w * scale)
        # Use nearest neighbor to preserve discrete colors
        from scipy.ndimage import zoom
        crop = zoom(crop, (new_h / h, new_w / w), order=0)
        h, w = new_h, new_w

    padded = np.full((SHAPE_SIZE, SHAPE_SIZE), PADDING_VALUE, dtype=np.uint8)

    start_r = (SHAPE_SIZE - h) // 2
    start_c = (SHAPE_SIZE - w) // 2
    padded[start_r:start_r+h, start_c:start_c+w] = crop

    return padded, (min_r, min_c, max_r, max_c)


def grid_to_onehot(grid: np.ndarray) -> torch.Tensor:
    """Convert a color grid to one-hot encoding.

    Args:
        grid: (H, W) integer array with values 0-9

    Returns:
        (1, 10, H, W) one-hot encoded tensor
    """
    H, W = grid.shape
    onehot = np.zeros((10, H, W), dtype=np.float32)
    for c in range(10):
        onehot[c] = (grid == c).astype(np.float32)
    return torch.from_numpy(onehot).unsqueeze(0)


@dataclass
class ShapeCorrespondence:
    """A single shape correspondence from one example pair."""
    puzzle_id: str
    example_idx: int
    input_shape_crop: np.ndarray      # (SHAPE_SIZE, SHAPE_SIZE)
    output_shape_crop: np.ndarray     # (SHAPE_SIZE, SHAPE_SIZE)
    input_slot_idx: int
    output_slot_idx: int
    input_bbox: Tuple[int, int, int, int]
    output_bbox: Tuple[int, int, int, int]
    similarity_score: float
    # For cross-attention context
    full_input_grid: np.ndarray = field(default=None)  # Original input grid
    input_slot_mask: np.ndarray = field(default=None)  # Mask for this slot in input


# =============================================================================
# Dataset
# =============================================================================

class ShapeCorrespondenceDataset(Dataset):
    """Dataset of shape correspondences for training."""

    def __init__(self, puzzles: Dict, ir_encoder: VentralCNN,
                 slot_attention: AffinitySlotAttention, device: torch.device,
                 similarity_threshold: float = 0.3,
                 store_grid_context: bool = True,
                 augment: bool = True,
                 num_augments: int = 8,
                 dihedral_only: bool = False,
                 color_only: bool = False):
        """
        Args:
            puzzles: Dict mapping puzzle_id -> puzzle data
            ir_encoder: VentralCNN for feature extraction
            slot_attention: AffinitySlotAttention for slot discovery
            device: torch device
            similarity_threshold: Minimum similarity for a correspondence
            store_grid_context: Whether to store full grid for cross-attention
            augment: Whether to apply augmentations to example pairs
            num_augments: Number of augmented versions per example (only used if augment=True)
            dihedral_only: Only apply dihedral transforms (no color permutation)
            color_only: Only apply color permutation (no dihedral transforms)
        """
        self.device = device
        self.store_grid_context = store_grid_context
        self.augment = augment
        self.num_augments = num_augments if augment else 1
        self.dihedral_only = dihedral_only
        self.color_only = color_only
        self.correspondences: List[ShapeCorrespondence] = []

        print("Extracting correspondences from puzzles...")
        for puzzle_id, puzzle in tqdm(puzzles.items()):
            for ex_idx, example in enumerate(puzzle.get('train', [])):
                # Extract correspondences from original example
                corrs = self._extract_correspondences(
                    example, puzzle_id, ex_idx,
                    ir_encoder, slot_attention, device,
                    similarity_threshold
                )
                self.correspondences.extend(corrs)

                # Extract correspondences from augmented versions
                if self.augment:
                    for aug_idx in range(self.num_augments):
                        aug_example = self._augment_example(example)
                        aug_corrs = self._extract_correspondences(
                            aug_example, puzzle_id, ex_idx,
                            ir_encoder, slot_attention, device,
                            similarity_threshold
                        )
                        self.correspondences.extend(aug_corrs)

        print(f"Extracted {len(self.correspondences)} correspondences from {len(puzzles)} puzzles")
        if self.augment:
            print(f"  (with {self.num_augments} augmentations per example)")

    def _get_augmentation(self) -> Tuple[int, np.ndarray]:
        """Get augmentation params based on augment settings."""
        if not self.augment:
            return 0, np.arange(10, dtype=np.uint8)  # Identity transform
        elif self.dihedral_only:
            # Random dihedral transform, but identity color mapping
            trans_id = np.random.randint(0, 8)
            color_map = np.arange(10, dtype=np.uint8)
            return trans_id, color_map
        elif self.color_only:
            # Identity dihedral transform, but random color mapping
            trans_id = 0
            color_map = random_color_permutation()
            return trans_id, color_map
        else:
            return get_random_augmentation()  # Full augmentation

    def _augment_example(self, example: Dict) -> Dict:
        """Apply the same random augmentation to both input and output grids."""
        input_grid = np.array(example['input'], dtype=np.uint8)
        output_grid = np.array(example['output'], dtype=np.uint8)

        # Get random augmentation (same for both input and output)
        trans_id, color_map = self._get_augmentation()

        # Apply augmentation to both grids
        aug_input = apply_augmentation(input_grid, trans_id, color_map)
        aug_output = apply_augmentation(output_grid, trans_id, color_map)

        return {
            'input': aug_input.tolist(),
            'output': aug_output.tolist()
        }

    def _extract_correspondences(self, example: Dict, puzzle_id: str, ex_idx: int,
                                   ir_encoder: VentralCNN, slot_attention: AffinitySlotAttention,
                                   device: torch.device, threshold: float) -> List[ShapeCorrespondence]:
        """Extract correspondences from a single example pair."""
        input_grid = np.array(example['input'], dtype=np.int64)
        output_grid = np.array(example['output'], dtype=np.int64)

        # Process grids through VentralCNN and AffinitySlotAttention
        with torch.no_grad():
            input_slots, input_masks = self._process_grid(input_grid, ir_encoder, slot_attention, device)
            output_slots, output_masks = self._process_grid(output_grid, ir_encoder, slot_attention, device)

        # Compute similarity and find correspondences
        similarity, valid_in, valid_out = self._compute_slot_similarity(
            input_masks, output_masks, input_grid, output_grid
        )
        correspondences = self._find_correspondences(similarity, valid_in, valid_out, threshold)

        # Extract shape crops for each correspondence
        results = []
        for in_idx, out_idx, score in correspondences:
            in_mask = input_masks[in_idx].cpu().numpy()
            out_mask = output_masks[out_idx].cpu().numpy()

            in_crop, in_bbox = extract_shape_crop(input_grid, in_mask)
            out_crop, out_bbox = extract_shape_crop(output_grid, out_mask)

            corr = ShapeCorrespondence(
                puzzle_id=puzzle_id,
                example_idx=ex_idx,
                input_shape_crop=in_crop,
                output_shape_crop=out_crop,
                input_slot_idx=in_idx,
                output_slot_idx=out_idx,
                input_bbox=in_bbox,
                output_bbox=out_bbox,
                similarity_score=score,
            )

            if self.store_grid_context:
                corr.full_input_grid = input_grid.copy()
                corr.input_slot_mask = in_mask.copy()

            results.append(corr)

        return results

    def _process_grid(self, grid: np.ndarray, encoder: VentralCNN,
                      slot_attention: AffinitySlotAttention,
                      device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """Process a grid through VentralCNN and AffinitySlotAttention."""
        H, W = grid.shape
        onehot = grid_to_onehot(grid).to(device)
        features = encoder(onehot)
        color_grid = torch.from_numpy(grid).unsqueeze(0).to(device)
        slots, masks = slot_attention(features, color_grid)
        return slots.squeeze(0), masks.squeeze(0)

    def _compute_slot_similarity(self, input_masks: torch.Tensor, output_masks: torch.Tensor,
                                  input_grid: np.ndarray, output_grid: np.ndarray) -> Tuple[np.ndarray, List[int], List[int]]:
        """Compute similarity between input and output slots using shape features."""
        from slot_viz import ShapeFeatureExtractor, compute_shape_similarity

        # Find non-empty slots
        input_counts = input_masks.sum(dim=(1, 2))
        output_counts = output_masks.sum(dim=(1, 2))

        valid_input_idx = (input_counts > 0).nonzero(as_tuple=True)[0].tolist()
        valid_output_idx = (output_counts > 0).nonzero(as_tuple=True)[0].tolist()

        if len(valid_input_idx) == 0 or len(valid_output_idx) == 0:
            return np.zeros((len(valid_input_idx), len(valid_output_idx))), valid_input_idx, valid_output_idx

        extractor = ShapeFeatureExtractor(n_fourier_coefficients=32)

        # Extract features
        input_features = [extractor.extract(input_masks[idx].cpu().numpy(), input_grid)
                          for idx in valid_input_idx]
        output_features = [extractor.extract(output_masks[idx].cpu().numpy(), output_grid)
                           for idx in valid_output_idx]

        # Compute similarity matrix
        similarity = np.zeros((len(valid_input_idx), len(valid_output_idx)))
        for i, in_feat in enumerate(input_features):
            for j, out_feat in enumerate(output_features):
                similarity[i, j] = compute_shape_similarity(in_feat, out_feat)

        return similarity, valid_input_idx, valid_output_idx

    def _find_correspondences(self, similarity: np.ndarray, valid_in: List[int],
                               valid_out: List[int], threshold: float,
                               margin: float = 0.1) -> List[Tuple[int, int, float]]:
        """Find correspondences from similarity matrix."""
        if similarity.size == 0:
            return []

        best_per_output = similarity.max(axis=0)
        best_per_input = similarity.max(axis=1)

        correspondences = []
        for in_i, in_idx in enumerate(valid_in):
            for out_i, out_idx in enumerate(valid_out):
                score = similarity[in_i, out_i]
                if score < threshold:
                    continue
                if score < best_per_output[out_i] - margin:
                    continue
                if score < best_per_input[in_i] - margin:
                    continue
                correspondences.append((in_idx, out_idx, float(score)))

        correspondences.sort(key=lambda x: x[2], reverse=True)
        return correspondences

    def __len__(self) -> int:
        return len(self.correspondences)

    def __getitem__(self, idx: int) -> Dict:
        corr = self.correspondences[idx]

        # Compute bbox dimensions (height, width) normalized to SHAPE_SIZE
        in_min_r, in_min_c, in_max_r, in_max_c = corr.input_bbox
        out_min_r, out_min_c, out_max_r, out_max_c = corr.output_bbox

        in_h = in_max_r - in_min_r + 1
        in_w = in_max_c - in_min_c + 1
        out_h = out_max_r - out_min_r + 1
        out_w = out_max_c - out_min_c + 1

        item = {
            'input_shape': torch.from_numpy(corr.input_shape_crop.astype(np.int64)),
            'output_shape': torch.from_numpy(corr.output_shape_crop.astype(np.int64)),
            'puzzle_id': corr.puzzle_id,
            'example_idx': corr.example_idx,
            'similarity': corr.similarity_score,
            # Bounding box info: normalized dimensions (0-1 range based on SHAPE_SIZE)
            'input_bbox_hw': torch.tensor([in_h / SHAPE_SIZE, in_w / SHAPE_SIZE], dtype=torch.float32),
            'output_bbox_hw': torch.tensor([out_h / SHAPE_SIZE, out_w / SHAPE_SIZE], dtype=torch.float32),
            # Raw bbox dimensions for creating masks
            'input_bbox_raw': torch.tensor([in_h, in_w], dtype=torch.int64),
            'output_bbox_raw': torch.tensor([out_h, out_w], dtype=torch.int64),
        }

        if self.store_grid_context and corr.full_input_grid is not None:
            # Pad grid to GRID_SIZE x GRID_SIZE
            H, W = corr.full_input_grid.shape
            padded_grid = np.full((GRID_SIZE, GRID_SIZE), PADDING_VALUE, dtype=np.int64)
            padded_grid[:H, :W] = corr.full_input_grid
            item['full_input_grid'] = torch.from_numpy(padded_grid)

            # Pad mask similarly
            padded_mask = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)
            padded_mask[:H, :W] = corr.input_slot_mask
            item['input_slot_mask'] = torch.from_numpy(padded_mask)

        return item


# =============================================================================
# Model
# =============================================================================

class PixelTransformerPredictor(nn.Module):
    """
    Transformer-based pixel predictor with identity skip connection.

    Each pixel in the shape attends to VentralCNN features from the full input grid,
    then passes through an MLP. The identity skip connection in the OUTPUT SPACE
    (not just feature space) means the model learns to MODIFY the input colors
    rather than predicting from scratch.

    Architecture:
        1. Embed input pixels with learnable color embeddings + 2D positional embeddings
        2. Cross-attention: each pixel queries VentralCNN features from full grid
        3. MLP refinement (with feature-space residuals)
        4. Project to color logits
        5. **Identity skip**: Add scaled input one-hot to logits (output-space residual)
        6. (Optional) Predict output bounding box from pooled features + input bbox

    The identity skip connection is critical: at initialization, output_proj produces
    ~0 logits, so the model defaults to predicting the input color. This gives a
    strong identity prior that the network learns to deviate from when needed.
    """

    def __init__(self, pixel_dim: int = 64, num_heads: int = 4, mlp_ratio: int = 4,
                 ir_encoder: VentralCNN = None, ir_out_dim: int = 64,
                 dropout: float = 0.0, use_bbox_prediction: bool = True,
                 identity_scale_init: float = 5.0):
        super().__init__()

        self.pixel_dim = pixel_dim
        self.ir_encoder = ir_encoder
        self.num_classes = NUM_COLORS
        self.use_bbox_prediction = use_bbox_prediction

        # Learnable identity skip scale - initialized to bias toward identity
        # At init, output_proj produces ~0, so identity_scale controls the identity prior
        self.identity_scale = nn.Parameter(torch.tensor(identity_scale_init))

        # Color embedding (11 entries: 0-9 colors + padding sentinel 10)
        self.color_embed = nn.Embedding(NUM_COLORS + 1, pixel_dim, padding_idx=PADDING_VALUE)

        # Learnable 2D positional embeddings for 16x16 canvas
        self.pos_embed = nn.Parameter(torch.zeros(1, SHAPE_SIZE * SHAPE_SIZE, pixel_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # Project VentralCNN features to pixel_dim if dimensions differ
        if ir_out_dim != pixel_dim:
            self.context_proj = nn.Linear(ir_out_dim, pixel_dim)
        else:
            self.context_proj = nn.Identity()

        # Cross-attention: pixels (query) attend to VentralCNN features (key/value)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=pixel_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.norm1 = nn.LayerNorm(pixel_dim)

        # MLP block
        self.mlp = nn.Sequential(
            nn.Linear(pixel_dim, pixel_dim * mlp_ratio),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(pixel_dim * mlp_ratio, pixel_dim),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(pixel_dim)

        # Output projection to color logits
        self.output_proj = nn.Linear(pixel_dim, NUM_COLORS)

        # Initialize output projection to small values (bias toward identity)
        nn.init.zeros_(self.output_proj.bias)
        nn.init.normal_(self.output_proj.weight, std=0.01)

        # Bounding box prediction head (predicts RESIDUAL: output = input + delta)
        # Zero residual = identity (output bbox = input bbox)
        if use_bbox_prediction:
            # Input: pooled pixel features + input bbox (2 values: h, w normalized)
            bbox_input_dim = pixel_dim + 2
            self.bbox_head = nn.Sequential(
                nn.Linear(bbox_input_dim, pixel_dim),
                nn.GELU(),
                nn.Linear(pixel_dim, pixel_dim // 2),
                nn.GELU(),
                nn.Linear(pixel_dim // 2, 2),  # Output: residual delta for height, width
            )
            # Initialize final layer to output zeros (identity by default)
            nn.init.zeros_(self.bbox_head[-1].weight)
            nn.init.zeros_(self.bbox_head[-1].bias)

    def _create_bbox_mask(self, pred_bbox_hw: torch.Tensor, hard: bool = False) -> torch.Tensor:
        """Create a centered bounding box mask from predicted height/width.

        Args:
            pred_bbox_hw: (B, 2) predicted height and width (normalized 0-1)
            hard: If True, use hard binary mask (for inference). If False, use
                  soft sigmoid mask (for training, allows gradient flow).

        Returns:
            mask: (B, 1, SHAPE_SIZE, SHAPE_SIZE) binary mask, 1 inside bbox
        """
        B = pred_bbox_hw.size(0)
        device = pred_bbox_hw.device

        # Convert normalized to pixel dimensions
        pred_h = (pred_bbox_hw[:, 0] * SHAPE_SIZE).clamp(min=1)  # (B,)
        pred_w = (pred_bbox_hw[:, 1] * SHAPE_SIZE).clamp(min=1)  # (B,)

        # Create coordinate grids
        rows = torch.arange(SHAPE_SIZE, device=device).float()
        cols = torch.arange(SHAPE_SIZE, device=device).float()
        row_grid, col_grid = torch.meshgrid(rows, cols, indexing='ij')  # (H, W)

        # Center of the canvas
        center = SHAPE_SIZE / 2.0

        # Compute start/end positions for each sample (centered bbox)
        start_r = (center - pred_h / 2).unsqueeze(-1).unsqueeze(-1)  # (B, 1, 1)
        end_r = (center + pred_h / 2).unsqueeze(-1).unsqueeze(-1)
        start_c = (center - pred_w / 2).unsqueeze(-1).unsqueeze(-1)
        end_c = (center + pred_w / 2).unsqueeze(-1).unsqueeze(-1)

        # Expand grids for batch
        row_grid = row_grid.unsqueeze(0).expand(B, -1, -1)  # (B, H, W)
        col_grid = col_grid.unsqueeze(0).expand(B, -1, -1)

        # Create soft mask (use sigmoid for differentiability during training)
        steepness = 5.0
        mask_r = torch.sigmoid(steepness * (row_grid - start_r)) * torch.sigmoid(steepness * (end_r - row_grid))
        mask_c = torch.sigmoid(steepness * (col_grid - start_c)) * torch.sigmoid(steepness * (end_c - col_grid))
        mask = mask_r * mask_c  # (B, H, W)

        if hard:
            mask = (mask > 0.5).float()

        return mask.unsqueeze(1)  # (B, 1, H, W)

    def forward(self, input_shape: torch.Tensor, output_shape: torch.Tensor = None,
                full_input_grid: torch.Tensor = None,
                input_slot_mask: torch.Tensor = None,
                input_bbox_hw: torch.Tensor = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass for pixel color prediction.

        Args:
            input_shape: (B, SHAPE_SIZE, SHAPE_SIZE) input shape crop with color indices
            output_shape: ignored (kept for API compatibility with ShapeDorsalCNN)
            full_input_grid: (B, GRID_SIZE, GRID_SIZE) full input grid for VentralCNN context
            input_slot_mask: ignored (could be used for masked attention in future)
            input_bbox_hw: (B, 2) input bbox dimensions (normalized h, w) for bbox prediction

        Returns:
            pixel_logits: (B, NUM_COLORS, SHAPE_SIZE, SHAPE_SIZE) color predictions
            pred_bbox_hw: (B, 2) predicted output bbox dimensions, or None if disabled
        """
        B = input_shape.size(0)
        device = input_shape.device

        # Embed input pixels: (B, 16, 16) -> (B, 16, 16, D) -> (B, 256, D)
        pixel_features = self.color_embed(input_shape.clamp(max=PADDING_VALUE))
        pixel_features = pixel_features.view(B, SHAPE_SIZE * SHAPE_SIZE, self.pixel_dim)

        # Add positional embeddings
        pixel_features = pixel_features + self.pos_embed

        # Get context features from VentralCNN
        if full_input_grid is not None and self.ir_encoder is not None:
            # Prepare input for VentralCNN: one-hot encode with content masking
            content_mask = (full_input_grid < PADDING_VALUE).float()
            inp_clamped = full_input_grid.clamp(max=9)
            inp_onehot = F.one_hot(inp_clamped.long(), NUM_COLORS).float()
            inp_onehot = inp_onehot * content_mask.unsqueeze(-1)
            inp_onehot = inp_onehot.permute(0, 3, 1, 2).contiguous()  # (B, 10, H, W)

            # Get VentralCNN features
            context_features = self.ir_encoder(inp_onehot)  # (B, H, W, ir_out_dim)
            H_ctx, W_ctx = context_features.shape[1], context_features.shape[2]
            context_features = context_features.view(B, H_ctx * W_ctx, -1)  # (B, H*W, D)
            context_features = self.context_proj(context_features)  # (B, H*W, pixel_dim)
        else:
            # Fallback: pixels attend to themselves (degrades to self-attention)
            context_features = pixel_features

        # Cross-attention + residual
        # Pixels query the VentralCNN features to gather relevant context
        attn_out, _ = self.cross_attn(
            query=pixel_features,
            key=context_features,
            value=context_features
        )
        pixel_features = pixel_features + self.norm1(attn_out)

        # MLP + residual
        pixel_features = pixel_features + self.norm2(self.mlp(pixel_features))

        # Bounding box prediction (residual: output = input + delta)
        pred_bbox_hw = None
        if self.use_bbox_prediction and input_bbox_hw is not None:
            # Global average pool the pixel features
            pooled_features = pixel_features.mean(dim=1)  # (B, pixel_dim)

            # Concatenate with input bbox info
            bbox_input = torch.cat([pooled_features, input_bbox_hw], dim=1)  # (B, pixel_dim+2)

            # Predict residual delta and add to input (identity when delta=0)
            bbox_delta = self.bbox_head(bbox_input)  # (B, 2) residual
            pred_bbox_hw = (input_bbox_hw + bbox_delta).clamp(min=0.0, max=1.0)  # (B, 2)

        # Project to color logits: (B, 256, D) -> (B, 256, 10)
        logits = self.output_proj(pixel_features)

        # Reshape to conv-style output: (B, 256, 10) -> (B, 16, 16, 10) -> (B, 10, 16, 16)
        logits = logits.view(B, SHAPE_SIZE, SHAPE_SIZE, NUM_COLORS)
        logits = logits.permute(0, 3, 1, 2).contiguous()

        # Identity skip connection: bias toward predicting input color
        # This gives the model a strong identity prior - it learns to MODIFY the input
        # rather than predict from scratch
        input_clamped = input_shape.clamp(max=NUM_COLORS - 1)  # Clamp padding to valid range
        input_onehot = F.one_hot(input_clamped, NUM_COLORS).float()  # (B, H, W, 10)
        input_onehot = input_onehot.permute(0, 3, 1, 2)  # (B, 10, H, W)

        # Zero out the identity skip for padding pixels (color 10)
        padding_mask = (input_shape >= PADDING_VALUE).unsqueeze(1).float()  # (B, 1, H, W)
        input_onehot = input_onehot * (1.0 - padding_mask)

        logits = logits + self.identity_scale * input_onehot

        # NOTE: We do NOT apply bbox masking here anymore.
        # The bbox mask was replacing identity-skip logits with forced black predictions,
        # which destroyed the identity prior for edge pixels when bbox prediction was imperfect.
        # Instead, bbox cropping should be applied as post-processing after prediction.

        return logits, pred_bbox_hw

    def predict_colors(self, input_shape: torch.Tensor,
                       full_input_grid: torch.Tensor = None,
                       input_slot_mask: torch.Tensor = None,
                       input_bbox_hw: torch.Tensor = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Predict output shape colors from input shape.

        Returns:
            predicted_colors: (B, SHAPE_SIZE, SHAPE_SIZE) predicted color indices
            pred_bbox_hw: (B, 2) predicted bbox dimensions or None
        """
        logits, pred_bbox_hw = self.forward(input_shape, None, full_input_grid, input_slot_mask, input_bbox_hw)
        return logits.argmax(dim=1), pred_bbox_hw


class ShapeDorsalCNN(nn.Module):
    """
    U-Net style CNN for shape-level prediction.

    Takes input and output shape crops (16x16) and predicts output shape colors.
    Optionally integrates with SlotRoutedCrossAttention for grid context.

    With bbox prediction enabled:
    - Receives input bounding box dimensions (height, width)
    - Predicts output bounding box dimensions
    - Masks output to only allow pixels within the predicted bounding box
    """

    def __init__(self, hidden_dim: int = 64, num_layers: int = 2, conv_depth: int = 2,
                 kernel_size: int = 3, use_onehot: bool = False,
                 use_cross_attention: bool = False,
                 ir_encoder: VentralCNN = None,
                 ir_out_dim: int = 64,
                 slot_top_k: int = 2,
                 use_bbox_prediction: bool = False):
        super().__init__()

        self.num_classes = NUM_COLORS
        self.num_layers = num_layers
        self.use_onehot = use_onehot
        self.use_cross_attention = use_cross_attention
        self.use_bbox_prediction = use_bbox_prediction

        if use_onehot:
            # One-hot (10) + content mask (1) = 11 channels
            embed_dim = 11
        else:
            # Learned embeddings (11 entries: 0-9 colors + padding sentinel)
            embed_dim = 16
            self.input_embed = nn.Embedding(NUM_COLORS + 1, 16, padding_idx=PADDING_VALUE)
            self.output_embed = nn.Embedding(NUM_COLORS + 1, 16, padding_idx=PADDING_VALUE)

        self.embed_dim = embed_dim

        # Channel concatenation: embed_dim * 2 (input + output)
        in_channels = embed_dim * 2
        base_ch = hidden_dim

        # Encoder - 16x16 -> 8x8 -> 4x4 for num_layers=2
        self.inc = get_conv_block(in_channels, base_ch, conv_depth, kernel_size)
        if num_layers >= 1:
            self.down1 = Down(base_ch, base_ch * 2, conv_depth, kernel_size)
        if num_layers >= 2:
            self.down2 = Down(base_ch * 2, base_ch * 4, conv_depth, kernel_size)

        # Decoder with skip connections
        if num_layers >= 2:
            self.up1 = Up(base_ch * 4, base_ch * 2, conv_depth, kernel_size)
        if num_layers >= 1:
            self.up2 = Up(base_ch * 2, base_ch, conv_depth, kernel_size)

        # Output: 10 channels for color prediction
        self.outc = nn.Conv2d(base_ch, NUM_COLORS, kernel_size=1, padding=0)

        # Cross-attention components
        self.ir_encoder = ir_encoder
        self.slot_cross_attention = None

        if use_cross_attention and ir_encoder is not None:
            # Slot attention for discovering slots in the input grid
            self.slot_attention = AffinitySlotAttention(
                input_dim=ir_out_dim,
                slot_dim=ir_out_dim,
                max_slots=40,
                merge_threshold=1.0,
                use_background_detection=True
            )

            # Cross-attention: shape decoder queries attend to input grid features
            self.slot_cross_attention = SlotRoutedCrossAttention(
                query_dim=base_ch,
                slot_dim=ir_out_dim,
                value_dim=ir_out_dim,
                proj_dim=ir_out_dim,
                top_k=slot_top_k
            )

        # Bounding box prediction head (predicts RESIDUAL, not absolute value)
        # Zero residual = identity (output bbox = input bbox)
        if use_bbox_prediction:
            # Compute bottleneck channel size based on num_layers
            if num_layers == 0:
                bottleneck_ch = base_ch
            elif num_layers == 1:
                bottleneck_ch = base_ch * 2
            else:
                bottleneck_ch = base_ch * 4

            # Input: bottleneck features (global pooled) + input bbox (2 values: h, w normalized)
            bbox_input_dim = bottleneck_ch + 2
            self.bbox_head = nn.Sequential(
                nn.Linear(bbox_input_dim, hidden_dim * 2),
                nn.ReLU(),
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 2),  # Output: residual delta for height, width
            )
            # Initialize final layer to output zeros (identity by default)
            nn.init.zeros_(self.bbox_head[-1].weight)
            nn.init.zeros_(self.bbox_head[-1].bias)

    def _encode_grid(self, grid: torch.Tensor) -> torch.Tensor:
        """Encode grid to one-hot + content mask."""
        # grid: (B, H, W) with values 0-10 (10 = padding)
        B, H, W = grid.shape

        # One-hot for colors 0-9, clamped to handle padding
        clamped = grid.clamp(max=9)
        onehot = F.one_hot(clamped.long(), NUM_COLORS).float()  # (B, H, W, 10)

        # Content mask: 1 where grid < 10 (valid), 0 for padding
        content_mask = (grid < PADDING_VALUE).float().unsqueeze(-1)  # (B, H, W, 1)

        # Zero out padding positions in one-hot
        onehot = onehot * content_mask

        # Concatenate one-hot with content mask
        return torch.cat([onehot, content_mask], dim=-1)  # (B, H, W, 11)

    def _create_bbox_mask(self, pred_bbox_hw: torch.Tensor, hard: bool = False) -> torch.Tensor:
        """Create a centered bounding box mask from predicted height/width.

        Args:
            pred_bbox_hw: (B, 2) predicted height and width (normalized 0-1)
            hard: If True, use hard binary mask (for inference). If False, use
                  soft sigmoid mask (for training, allows gradient flow).

        Returns:
            mask: (B, 1, SHAPE_SIZE, SHAPE_SIZE) binary mask, 1 inside bbox
        """
        B = pred_bbox_hw.size(0)
        device = pred_bbox_hw.device

        # Convert normalized to pixel dimensions
        pred_h = (pred_bbox_hw[:, 0] * SHAPE_SIZE).clamp(min=1)  # (B,)
        pred_w = (pred_bbox_hw[:, 1] * SHAPE_SIZE).clamp(min=1)  # (B,)

        # Create coordinate grids
        rows = torch.arange(SHAPE_SIZE, device=device).float()
        cols = torch.arange(SHAPE_SIZE, device=device).float()
        row_grid, col_grid = torch.meshgrid(rows, cols, indexing='ij')  # (H, W)

        # Center of the canvas
        center = SHAPE_SIZE / 2.0

        # Compute start/end positions for each sample (centered bbox)
        start_r = (center - pred_h / 2).unsqueeze(-1).unsqueeze(-1)  # (B, 1, 1)
        end_r = (center + pred_h / 2).unsqueeze(-1).unsqueeze(-1)
        start_c = (center - pred_w / 2).unsqueeze(-1).unsqueeze(-1)
        end_c = (center + pred_w / 2).unsqueeze(-1).unsqueeze(-1)

        # Expand grids for batch
        row_grid = row_grid.unsqueeze(0).expand(B, -1, -1)  # (B, H, W)
        col_grid = col_grid.unsqueeze(0).expand(B, -1, -1)

        # Create soft mask (use sigmoid for differentiability during training)
        # Steepness controls how sharp the boundary is
        steepness = 5.0
        mask_r = torch.sigmoid(steepness * (row_grid - start_r)) * torch.sigmoid(steepness * (end_r - row_grid))
        mask_c = torch.sigmoid(steepness * (col_grid - start_c)) * torch.sigmoid(steepness * (end_c - col_grid))
        mask = mask_r * mask_c  # (B, H, W)

        # Hard threshold at inference for clean boundaries
        if hard:
            mask = (mask > 0.5).float()

        return mask.unsqueeze(1)  # (B, 1, H, W)

    def forward(self, input_shape: torch.Tensor, output_shape: torch.Tensor,
                full_input_grid: torch.Tensor = None,
                input_slot_mask: torch.Tensor = None,
                input_bbox_hw: torch.Tensor = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass for shape color prediction.

        Args:
            input_shape: (B, SHAPE_SIZE, SHAPE_SIZE) input shape crop
            output_shape: (B, SHAPE_SIZE, SHAPE_SIZE) output shape crop (target during training)
            full_input_grid: (B, GRID_SIZE, GRID_SIZE) optional full input grid for cross-attention
            input_slot_mask: (B, GRID_SIZE, GRID_SIZE) optional mask for input slot
            input_bbox_hw: (B, 2) optional input bbox dimensions (normalized h, w)

        Returns:
            pixel_logits: (B, 10, SHAPE_SIZE, SHAPE_SIZE)
            pred_bbox_hw: (B, 2) predicted output bbox dimensions, or None if bbox prediction disabled
        """
        # Encode shapes
        if self.use_onehot:
            inp_emb = self._encode_grid(input_shape)   # (B, 16, 16, 11)
            out_emb = self._encode_grid(output_shape)  # (B, 16, 16, 11)
        else:
            inp_emb = self.input_embed(input_shape)    # (B, 16, 16, 16)
            out_emb = self.output_embed(output_shape)  # (B, 16, 16, 16)

        # Concatenate input and output embeddings
        x = torch.cat([inp_emb, out_emb], dim=-1)  # (B, 16, 16, embed_dim*2)
        x = x.permute(0, 3, 1, 2).contiguous()  # (B, C, 16, 16)

        # U-Net encoder
        x1 = self.inc(x)  # (B, base_ch, 16, 16)
        if self.num_layers >= 1:
            x2 = self.down1(x1)  # (B, base_ch*2, 8, 8)
        if self.num_layers >= 2:
            x3 = self.down2(x2)  # (B, base_ch*4, 4, 4)

        # Bottleneck
        if self.num_layers == 0:
            bottleneck = x1
        elif self.num_layers == 1:
            bottleneck = x2
        else:
            bottleneck = x3

        # U-Net decoder with skip connections
        if self.num_layers == 2:
            x = self.up1(bottleneck, x2)  # (B, base_ch*2, 8, 8)
            x = self.up2(x, x1)            # (B, base_ch, 16, 16)
        elif self.num_layers == 1:
            x = self.up2(bottleneck, x1)   # (B, base_ch, 16, 16)
        else:
            x = bottleneck

        # Optional cross-attention to full input grid
        if (self.slot_cross_attention is not None and
            full_input_grid is not None and
            input_slot_mask is not None):

            B = input_shape.size(0)

            # Compute IR features for the full input grid
            input_content_mask = (full_input_grid < PADDING_VALUE).float()
            inp_clamped = full_input_grid.clamp(max=9)
            inp_onehot = F.one_hot(inp_clamped.long(), NUM_COLORS).float()
            inp_onehot = inp_onehot * input_content_mask.unsqueeze(-1)
            inp_onehot = inp_onehot.permute(0, 3, 1, 2).contiguous()

            ir_features = self.ir_encoder(inp_onehot)  # (B, H, W, ir_out_dim)

            # Get slot masks from the provided input_slot_mask
            # We use the single slot mask provided (the correspondence's input slot)
            slot_masks = input_slot_mask.unsqueeze(1)  # (B, 1, H, W)
            slot_embeddings = torch.zeros(B, 1, ir_features.size(-1), device=x.device)

            # Decoder features attend to IR features via the input slot
            x_perm = x.permute(0, 2, 3, 1).contiguous()  # (B, H, W, C)

            attended = self.slot_cross_attention(
                decoder_features=x_perm,
                slot_embeddings=slot_embeddings,
                slot_masks=slot_masks,
                ir_features=ir_features,
                content_mask=input_content_mask
            )

            x_perm = x_perm + attended
            x = x_perm.permute(0, 3, 1, 2).contiguous()

        # Bounding box prediction (residual: output = input + delta)
        pred_bbox_hw = None
        if self.use_bbox_prediction and input_bbox_hw is not None:
            # Global average pool the bottleneck features
            bottleneck_pooled = bottleneck.mean(dim=(2, 3))  # (B, C)

            # Concatenate with input bbox info
            bbox_input = torch.cat([bottleneck_pooled, input_bbox_hw], dim=1)  # (B, C+2)

            # Predict residual delta and add to input (identity when delta=0)
            bbox_delta = self.bbox_head(bbox_input)  # (B, 2) residual
            pred_bbox_hw = (input_bbox_hw + bbox_delta).clamp(min=0.0, max=1.0)  # (B, 2)

        # Output head
        pixel_logits = self.outc(x)  # (B, 10, 16, 16)

        # Apply bbox mask if prediction is enabled
        if self.use_bbox_prediction and pred_bbox_hw is not None:
            # Use hard mask at inference, soft mask during training for gradient flow
            bbox_mask = self._create_bbox_mask(pred_bbox_hw, hard=not self.training)  # (B, 1, H, W)

            # Mask the logits: outside bbox, strongly favor color 0 (black/background)
            # We do this by adding a large negative bias to non-zero color logits outside the mask
            outside_mask = 1.0 - bbox_mask  # (B, 1, H, W)

            # Create bias: for colors 1-9, add large negative value outside bbox
            logit_bias = torch.zeros_like(pixel_logits)
            logit_bias[:, 1:, :, :] = -10.0 * outside_mask  # Penalize non-black outside bbox

            pixel_logits = pixel_logits + logit_bias

        return pixel_logits, pred_bbox_hw

    def predict_colors(self, input_shape: torch.Tensor,
                       full_input_grid: torch.Tensor = None,
                       input_slot_mask: torch.Tensor = None,
                       input_bbox_hw: torch.Tensor = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Predict output shape colors from input shape.

        Uses blank output as initial guess.

        Returns:
            predicted_colors: (B, 16, 16) predicted color indices
            pred_bbox_hw: (B, 2) predicted bbox dimensions or None
        """
        B = input_shape.size(0)
        # Start with blank output (all zeros)
        output_shape = torch.zeros_like(input_shape)

        logits, pred_bbox_hw = self.forward(
            input_shape, output_shape, full_input_grid, input_slot_mask, input_bbox_hw
        )
        return logits.argmax(dim=1), pred_bbox_hw  # (B, 16, 16), (B, 2) or None


# =============================================================================
# Training
# =============================================================================

def masked_cross_entropy(logits: torch.Tensor, targets: torch.Tensor,
                         mask: torch.Tensor) -> torch.Tensor:
    """Compute cross-entropy loss only on content pixels.

    Args:
        logits: (B, 10, H, W) prediction logits
        targets: (B, H, W) target color indices (may contain padding value 10)
        mask: (B, H, W) binary mask, 1 for content pixels

    Returns:
        Scalar loss value
    """
    # Clamp targets to valid range (0-9) to avoid index errors
    # Padding pixels (value 10) will be masked out anyway
    targets_clamped = targets.clamp(max=9).long()

    # Compute per-pixel cross-entropy
    per_pixel_loss = F.cross_entropy(logits, targets_clamped, reduction='none')

    # Apply mask (zeros out loss for padding pixels)
    masked_loss = per_pixel_loss * mask

    # Average over content pixels
    num_pixels = mask.sum()
    if num_pixels > 0:
        return masked_loss.sum() / num_pixels
    return masked_loss.sum()


def bbox_loss(pred_bbox_hw: torch.Tensor, target_bbox_hw: torch.Tensor) -> torch.Tensor:
    """Compute MSE loss for bounding box prediction.

    Args:
        pred_bbox_hw: (B, 2) predicted normalized height, width
        target_bbox_hw: (B, 2) target normalized height, width

    Returns:
        Scalar loss value
    """
    return F.mse_loss(pred_bbox_hw, target_bbox_hw)


def train_epoch(model: ShapeDorsalCNN, loader: DataLoader,
                optimizer: torch.optim.Optimizer, device: torch.device,
                use_cross_attention: bool = False,
                use_bbox_prediction: bool = False,
                bbox_loss_weight: float = 1.0) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    total_pixel_loss = 0.0
    total_bbox_loss = 0.0
    total_correct = 0
    total_pixels = 0
    num_batches = 0

    for batch in tqdm(loader, desc="Training"):
        input_shapes = batch['input_shape'].to(device)
        output_shapes = batch['output_shape'].to(device)

        # Content mask: pixels where output is not padding
        content_mask = (output_shapes < PADDING_VALUE).float()

        optimizer.zero_grad()

        # Prepare bbox inputs if needed
        input_bbox_hw = None
        target_bbox_hw = None
        if use_bbox_prediction and 'input_bbox_hw' in batch:
            input_bbox_hw = batch['input_bbox_hw'].to(device)
            target_bbox_hw = batch['output_bbox_hw'].to(device)

        # Forward pass
        if use_cross_attention and 'full_input_grid' in batch:
            logits, pred_bbox_hw = model(
                input_shapes, output_shapes,
                batch['full_input_grid'].to(device),
                batch['input_slot_mask'].to(device),
                input_bbox_hw
            )
        else:
            logits, pred_bbox_hw = model(input_shapes, output_shapes,
                                          input_bbox_hw=input_bbox_hw)

        # Pixel loss
        pixel_loss = masked_cross_entropy(logits, output_shapes, content_mask)

        # Bbox loss
        bbox_loss_val = torch.tensor(0.0, device=device)
        if use_bbox_prediction and pred_bbox_hw is not None and target_bbox_hw is not None:
            bbox_loss_val = bbox_loss(pred_bbox_hw, target_bbox_hw)

        # Combined loss
        loss = pixel_loss + bbox_loss_weight * bbox_loss_val

        loss.backward()
        optimizer.step()

        # Metrics
        total_loss += loss.item()
        total_pixel_loss += pixel_loss.item()
        total_bbox_loss += bbox_loss_val.item()
        predictions = logits.argmax(dim=1)
        correct = ((predictions == output_shapes) * content_mask).sum().item()
        total_correct += correct
        total_pixels += content_mask.sum().item()
        num_batches += 1

    metrics = {
        'loss': total_loss / max(num_batches, 1),
        'pixel_loss': total_pixel_loss / max(num_batches, 1),
        'pixel_accuracy': total_correct / max(total_pixels, 1),
    }
    if use_bbox_prediction:
        metrics['bbox_loss'] = total_bbox_loss / max(num_batches, 1)

    return metrics


def evaluate(model: ShapeDorsalCNN, loader: DataLoader, device: torch.device,
             use_cross_attention: bool = False,
             use_bbox_prediction: bool = False) -> Dict[str, float]:
    """Evaluate on validation/test set."""
    model.eval()
    total_loss = 0.0
    total_pixel_loss = 0.0
    total_bbox_loss = 0.0
    total_correct = 0
    total_pixels = 0
    exact_matches = 0
    total_samples = 0
    num_batches = 0

    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating"):
            input_shapes = batch['input_shape'].to(device)
            output_shapes = batch['output_shape'].to(device)

            content_mask = (output_shapes < PADDING_VALUE).float()

            # Prepare bbox inputs if needed
            input_bbox_hw = None
            target_bbox_hw = None
            if use_bbox_prediction and 'input_bbox_hw' in batch:
                input_bbox_hw = batch['input_bbox_hw'].to(device)
                target_bbox_hw = batch['output_bbox_hw'].to(device)

            if use_cross_attention and 'full_input_grid' in batch:
                logits, pred_bbox_hw = model(
                    input_shapes, output_shapes,
                    batch['full_input_grid'].to(device),
                    batch['input_slot_mask'].to(device),
                    input_bbox_hw
                )
            else:
                logits, pred_bbox_hw = model(input_shapes, output_shapes,
                                              input_bbox_hw=input_bbox_hw)

            pixel_loss = masked_cross_entropy(logits, output_shapes, content_mask)
            total_pixel_loss += pixel_loss.item()

            # Bbox loss
            bbox_loss_val = 0.0
            if use_bbox_prediction and pred_bbox_hw is not None and target_bbox_hw is not None:
                bbox_loss_val = bbox_loss(pred_bbox_hw, target_bbox_hw).item()
                total_bbox_loss += bbox_loss_val

            total_loss += pixel_loss.item() + bbox_loss_val

            predictions = logits.argmax(dim=1)
            correct = ((predictions == output_shapes) * content_mask).sum().item()
            total_correct += correct
            total_pixels += content_mask.sum().item()

            # Check exact matches per sample
            B = input_shapes.size(0)
            for i in range(B):
                mask_i = content_mask[i]
                if mask_i.sum() == 0:
                    continue
                pred_i = predictions[i][mask_i.bool()]
                target_i = output_shapes[i][mask_i.bool()]
                if torch.all(pred_i == target_i):
                    exact_matches += 1
                total_samples += 1

            num_batches += 1

    metrics = {
        'loss': total_loss / max(num_batches, 1),
        'pixel_loss': total_pixel_loss / max(num_batches, 1),
        'pixel_accuracy': total_correct / max(total_pixels, 1),
        'exact_match_rate': exact_matches / max(total_samples, 1),
    }
    if use_bbox_prediction:
        metrics['bbox_loss'] = total_bbox_loss / max(num_batches, 1)

    return metrics


# =============================================================================
# Test Evaluation
# =============================================================================

def evaluate_test_pair(model: ShapeDorsalCNN, ir_encoder: VentralCNN,
                       slot_attention: AffinitySlotAttention,
                       test_input: np.ndarray, test_output: np.ndarray,
                       device: torch.device) -> List[Dict]:
    """
    Evaluate on a test pair.

    For each input slot, predict output shape and check against actual output shapes.
    """
    model.eval()

    # Process input to get slots
    H_in, W_in = test_input.shape
    with torch.no_grad():
        onehot_in = grid_to_onehot(test_input).to(device)
        features_in = ir_encoder(onehot_in)
        color_grid_in = torch.from_numpy(test_input).unsqueeze(0).to(device)
        _, input_masks = slot_attention(features_in, color_grid_in)
        input_masks = input_masks.squeeze(0)  # (max_slots, H, W)

    # Get valid input slots
    input_counts = input_masks.sum(dim=(1, 2))
    valid_input_idx = (input_counts > 0).nonzero(as_tuple=True)[0].tolist()

    # Process output to get slots (ground truth)
    H_out, W_out = test_output.shape
    with torch.no_grad():
        onehot_out = grid_to_onehot(test_output).to(device)
        features_out = ir_encoder(onehot_out)
        color_grid_out = torch.from_numpy(test_output).unsqueeze(0).to(device)
        _, output_masks = slot_attention(features_out, color_grid_out)
        output_masks = output_masks.squeeze(0)

    # Get valid output slots
    output_counts = output_masks.sum(dim=(1, 2))
    valid_output_idx = (output_counts > 0).nonzero(as_tuple=True)[0].tolist()

    # Extract output shape crops for comparison
    output_crops = []
    for out_idx in valid_output_idx:
        out_mask = output_masks[out_idx].cpu().numpy()
        crop, _ = extract_shape_crop(test_output, out_mask)
        output_crops.append((out_idx, crop))

    results = []

    # Prepare full input grid (padded)
    padded_input = np.full((GRID_SIZE, GRID_SIZE), PADDING_VALUE, dtype=np.int64)
    padded_input[:H_in, :W_in] = test_input
    full_input_tensor = torch.from_numpy(padded_input).unsqueeze(0).to(device)

    with torch.no_grad():
        for in_idx in valid_input_idx:
            in_mask = input_masks[in_idx].cpu().numpy()
            in_crop, in_bbox = extract_shape_crop(test_input, in_mask)

            # Prepare slot mask (padded)
            padded_mask = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)
            padded_mask[:H_in, :W_in] = in_mask
            mask_tensor = torch.from_numpy(padded_mask).unsqueeze(0).to(device)

            # Compute input bbox for model
            in_h = in_bbox[2] - in_bbox[0] + 1
            in_w = in_bbox[3] - in_bbox[1] + 1
            input_bbox_hw = torch.tensor(
                [[in_h / SHAPE_SIZE, in_w / SHAPE_SIZE]], dtype=torch.float32, device=device
            )

            # Predict output shape
            in_tensor = torch.from_numpy(in_crop.astype(np.int64)).unsqueeze(0).to(device)
            pred_colors, _ = model.predict_colors(in_tensor, full_input_tensor, mask_tensor, input_bbox_hw)
            pred_crop = pred_colors[0].cpu().numpy()

            # Check against each output slot
            for out_idx, out_crop in output_crops:
                # Compare (exact match on content pixels)
                content_mask = (out_crop < PADDING_VALUE)
                pred_masked = pred_crop[content_mask]
                out_masked = out_crop[content_mask]

                match_type = 'NONE'
                if len(pred_masked) > 0 and np.array_equal(pred_masked, out_masked):
                    match_type = 'EXACT'
                elif len(pred_masked) > 0:
                    accuracy = (pred_masked == out_masked).mean()
                    if accuracy > 0.8:
                        match_type = 'PARTIAL'

                if match_type != 'NONE':
                    results.append({
                        'input_slot': in_idx,
                        'output_slot': out_idx,
                        'match': match_type,
                        'predicted': pred_crop,
                        'actual': out_crop,
                    })

    return results


# =============================================================================
# Visualization
# =============================================================================

# ARC color palette
ARC_COLORS = [
    '#000000',  # 0: black
    '#0074D9',  # 1: blue
    '#FF4136',  # 2: red
    '#2ECC40',  # 3: green
    '#FFDC00',  # 4: yellow
    '#AAAAAA',  # 5: grey
    '#F012BE',  # 6: magenta
    '#FF851B',  # 7: orange
    '#7FDBFF',  # 8: cyan
    '#870C25',  # 9: brown/maroon
]


def grid_to_rgb(grid: np.ndarray) -> np.ndarray:
    """Convert a color grid to RGB image."""
    H, W = grid.shape
    rgb = np.zeros((H, W, 3), dtype=np.float32)
    for c in range(10):
        color = np.array([int(ARC_COLORS[c][i:i+2], 16) / 255.0 for i in (1, 3, 5)])
        mask = (grid == c)
        rgb[mask] = color
    # Handle padding (value 10) as white
    rgb[grid >= 10] = [0.9, 0.9, 0.9]
    return rgb


def visualize_test_predictions(model: ShapeDorsalCNN, ir_encoder: VentralCNN,
                                slot_attention: AffinitySlotAttention,
                                puzzle: Dict, puzzle_id: str,
                                device: torch.device, save_path: str = None):
    """
    Visualize predictions on the test input of a puzzle.

    Shows:
    - Test input grid with extracted slots
    - For each input slot: the input shape crop and predicted output shape
    - Test output grid (if available) for comparison
    """
    model.eval()

    # Get test example
    test_example = puzzle.get('test', [{}])[0]
    test_input = np.array(test_example.get('input', []), dtype=np.int64)
    test_output = np.array(test_example.get('output', [])) if 'output' in test_example else None

    if test_input.size == 0:
        print("No test input found")
        return

    H_in, W_in = test_input.shape

    # Extract slots from test input
    with torch.no_grad():
        onehot_in = grid_to_onehot(test_input).to(device)
        features_in = ir_encoder(onehot_in)
        color_grid_in = torch.from_numpy(test_input).unsqueeze(0).to(device)
        _, input_masks = slot_attention(features_in, color_grid_in)
        input_masks = input_masks.squeeze(0)

    # Get valid input slots
    input_counts = input_masks.sum(dim=(1, 2))
    valid_input_idx = (input_counts > 0).nonzero(as_tuple=True)[0].tolist()

    # Prepare predictions for each slot
    predictions = []
    padded_input = np.full((GRID_SIZE, GRID_SIZE), PADDING_VALUE, dtype=np.int64)
    padded_input[:H_in, :W_in] = test_input
    full_input_tensor = torch.from_numpy(padded_input).unsqueeze(0).to(device)

    with torch.no_grad():
        for in_idx in valid_input_idx:
            in_mask = input_masks[in_idx].cpu().numpy()
            in_crop, in_bbox = extract_shape_crop(test_input, in_mask)

            # Prepare slot mask
            padded_mask = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)
            padded_mask[:H_in, :W_in] = in_mask
            mask_tensor = torch.from_numpy(padded_mask).unsqueeze(0).to(device)

            # Predict - compute input bbox for model
            in_h = in_bbox[2] - in_bbox[0] + 1
            in_w = in_bbox[3] - in_bbox[1] + 1
            input_bbox_hw = torch.tensor(
                [[in_h / SHAPE_SIZE, in_w / SHAPE_SIZE]], dtype=torch.float32, device=device
            )

            in_tensor = torch.from_numpy(in_crop.astype(np.int64)).unsqueeze(0).to(device)
            pred_colors, pred_bbox_hw = model.predict_colors(
                in_tensor, full_input_tensor, mask_tensor, input_bbox_hw
            )
            pred_crop = pred_colors[0].cpu().numpy()

            pred_info = {
                'slot_idx': in_idx,
                'input_crop': in_crop,
                'predicted_crop': pred_crop,
                'mask': in_mask,
                'bbox': in_bbox,
            }
            if pred_bbox_hw is not None:
                pred_h = int(pred_bbox_hw[0, 0].item() * SHAPE_SIZE)
                pred_w = int(pred_bbox_hw[0, 1].item() * SHAPE_SIZE)
                pred_info['predicted_bbox_hw'] = (pred_h, pred_w)

            predictions.append(pred_info)

    # Create visualization
    n_slots = len(predictions)
    if n_slots == 0:
        print("No valid slots found in test input")
        return

    # Layout: input grid | slot predictions (2 cols each: input crop, predicted) | output grid
    n_pred_cols = min(n_slots, 4)  # Max 4 slot predictions per row
    n_pred_rows = (n_slots + n_pred_cols - 1) // n_pred_cols

    has_output = test_output is not None and test_output.size > 0

    fig_width = 4 + n_pred_cols * 4 + (4 if has_output else 0)
    fig_height = max(4, n_pred_rows * 2.5)

    fig = plt.figure(figsize=(fig_width, fig_height))

    # Grid spec for layout
    if has_output:
        gs = fig.add_gridspec(n_pred_rows, 2 + n_pred_cols * 2 + 2, wspace=0.3, hspace=0.4)
    else:
        gs = fig.add_gridspec(n_pred_rows, 2 + n_pred_cols * 2, wspace=0.3, hspace=0.4)

    # Draw test input (spans all rows on the left)
    ax_input = fig.add_subplot(gs[:, :2])
    ax_input.imshow(grid_to_rgb(test_input), interpolation='nearest')
    ax_input.set_title(f'Test Input\n({H_in}x{W_in})', fontsize=10)
    ax_input.axis('off')

    # Draw slot outlines on input
    for pred in predictions:
        mask = pred['mask']
        rows, cols = np.where(mask > 0)
        if len(rows) > 0:
            min_r, max_r = rows.min(), rows.max()
            min_c, max_c = cols.min(), cols.max()
            rect = plt.Rectangle((min_c - 0.5, min_r - 0.5), max_c - min_c + 1, max_r - min_r + 1,
                                  fill=False, edgecolor='yellow', linewidth=2)
            ax_input.add_patch(rect)
            ax_input.text(min_c, min_r - 0.5, f'S{pred["slot_idx"]}', fontsize=8,
                          color='yellow', va='bottom')

    # Draw slot predictions
    for i, pred in enumerate(predictions):
        row = i // n_pred_cols
        col = i % n_pred_cols

        # Input crop - crop to the actual bbox region
        in_bbox = pred['bbox']
        in_h = in_bbox[2] - in_bbox[0] + 1
        in_w = in_bbox[3] - in_bbox[1] + 1
        # Compute crop region (centered in SHAPE_SIZE canvas)
        in_start_r = int((SHAPE_SIZE - in_h) / 2)
        in_start_c = int((SHAPE_SIZE - in_w) / 2)
        in_cropped = pred['input_crop'][in_start_r:in_start_r+in_h, in_start_c:in_start_c+in_w]

        ax_in = fig.add_subplot(gs[row, 2 + col * 2])
        ax_in.imshow(grid_to_rgb(in_cropped), interpolation='nearest')
        ax_in.set_title(f'S{pred["slot_idx"]} In\n({in_h}x{in_w})', fontsize=8)
        ax_in.axis('off')

        # Predicted output - crop to the predicted bbox region
        ax_pred = fig.add_subplot(gs[row, 2 + col * 2 + 1])

        if 'predicted_bbox_hw' in pred:
            pred_h, pred_w = pred['predicted_bbox_hw']
            # Compute crop region (centered in SHAPE_SIZE canvas)
            pred_start_r = int((SHAPE_SIZE - pred_h) / 2)
            pred_start_c = int((SHAPE_SIZE - pred_w) / 2)
            pred_cropped = pred['predicted_crop'][pred_start_r:pred_start_r+pred_h, pred_start_c:pred_start_c+pred_w]
            pred_title = f'S{pred["slot_idx"]} Pred\n({pred_h}x{pred_w})'
        else:
            # No bbox prediction - show full crop
            pred_cropped = pred['predicted_crop']
            pred_title = f'S{pred["slot_idx"]} Pred'

        ax_pred.imshow(grid_to_rgb(pred_cropped), interpolation='nearest')
        ax_pred.set_title(pred_title, fontsize=8)
        ax_pred.axis('off')

    # Draw test output if available
    if has_output:
        ax_output = fig.add_subplot(gs[:, -2:])
        ax_output.imshow(grid_to_rgb(test_output), interpolation='nearest')
        H_out, W_out = test_output.shape
        ax_output.set_title(f'Test Output\n({H_out}x{W_out})', fontsize=10)
        ax_output.axis('off')

    plt.suptitle(f'Puzzle: {puzzle_id} - Test Predictions', fontsize=12, fontweight='bold')

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to: {save_path}")

    # Always show the plot window
    plt.show()


# =============================================================================
# Puzzle Loading
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

    print(f"Loaded {len(puzzles)} puzzles")
    return puzzles


# =============================================================================
# Main
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Correspondence Predictor Training")

    # Data
    parser.add_argument("--data-root", type=str, default="kaggle/combined",
                        help="Path to ARC dataset")
    parser.add_argument("--puzzle-id", type=str, default=None,
                        help="Run on a single puzzle by ID (e.g., 00d62c1b)")
    parser.add_argument("--max-puzzles", type=int, default=None,
                        help="Maximum number of puzzles to use (for debugging)")

    # Model selection
    parser.add_argument("--model", type=str, default="transformer",
                        choices=["transformer", "cnn"],
                        help="Model type: 'transformer' (PixelTransformerPredictor) or 'cnn' (ShapeDorsalCNN)")

    # Transformer model args
    parser.add_argument("--pixel-dim", type=int, default=16,
                        help="Pixel embedding dimension for transformer model")
    parser.add_argument("--num-heads", type=int, default=4,
                        help="Number of attention heads for transformer model")
    parser.add_argument("--mlp-ratio", type=int, default=4,
                        help="MLP expansion ratio for transformer model")
    parser.add_argument("--dropout", type=float, default=0.0,
                        help="Dropout rate for transformer model")
    parser.add_argument("--identity-scale", type=float, default=500.0,
                        help="Initial scale for identity skip connection (higher = stronger identity prior)")

    # CNN model args (legacy)
    parser.add_argument("--hidden-dim", type=int, default=8,
                        help="Hidden dimension for U-Net (CNN model only)")
    parser.add_argument("--num-layers", type=int, default=0,
                        help="Number of U-Net encoder/decoder layers (CNN model only)")
    parser.add_argument("--use-onehot", action="store_true",
                        help="Use one-hot encoding instead of learned embeddings (CNN model only)")
    parser.add_argument("--use-cross-attention", action="store_true",
                        help="Enable cross-attention to full input grid (CNN model only)")

    # Bbox prediction (works for both models)
    parser.add_argument("--use-bbox-prediction", action="store_true",
                        help="Enable bounding box prediction (predicts output bbox from input bbox)")
    parser.add_argument("--bbox-loss-weight", type=float, default=1.0,
                        help="Weight for bounding box loss")

    # Shared model args
    parser.add_argument("--ir-out-dim", type=int, default=16,
                        help="Output dimension of VentralCNN (IR encoder)")

    # Training
    parser.add_argument("--epochs", type=int, default=100,
                        help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.01,
                        help="Weight decay")
    parser.add_argument("--similarity-threshold", type=float, default=0.3,
                        help="Minimum similarity for correspondences")

    # Augmentation
    parser.add_argument("--no-augment", action="store_true",
                        help="Disable augmentations (use raw example grids only)")
    parser.add_argument("--num-augments", type=int, default=50,
                        help="Number of augmented versions per example pair")
    parser.add_argument("--dihedral-only", action="store_true",
                        help="Only apply dihedral transforms (no color permutation)")
    parser.add_argument("--color-only", action="store_true",
                        help="Only apply color permutation (no dihedral transforms)")

    # Output
    parser.add_argument("--save-dir", type=str, default="checkpoints/correspondence",
                        help="Directory for saving visualizations")

    return parser.parse_args()


def main():
    args = parse_args()

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)

    # Load puzzles
    puzzles = load_puzzles(args.data_root)

    # Single puzzle mode
    if args.puzzle_id:
        if args.puzzle_id not in puzzles:
            print(f"Error: Puzzle '{args.puzzle_id}' not found in dataset")
            return
        puzzles = {args.puzzle_id: puzzles[args.puzzle_id]}
        print(f"Running on single puzzle: {args.puzzle_id}")
        # Use same puzzle for train and val in single-puzzle mode
        train_puzzles = puzzles
        val_puzzles = puzzles
    else:
        if args.max_puzzles:
            puzzle_ids = list(puzzles.keys())[:args.max_puzzles]
            puzzles = {k: puzzles[k] for k in puzzle_ids}

        # Split into train/val
        puzzle_ids = list(puzzles.keys())
        np.random.seed(42)
        np.random.shuffle(puzzle_ids)
        split_idx = int(0.9 * len(puzzle_ids))
        train_ids = puzzle_ids[:split_idx]
        val_ids = puzzle_ids[split_idx:]

        train_puzzles = {k: puzzles[k] for k in train_ids}
        val_puzzles = {k: puzzles[k] for k in val_ids}

    print(f"Train puzzles: {len(train_puzzles)}, Val puzzles: {len(val_puzzles)}")

    # Create IR encoder and slot attention for correspondence extraction
    ir_encoder = VentralCNN(out_dim=args.ir_out_dim).to(device)
    ir_encoder.eval()

    slot_attention = AffinitySlotAttention(
        input_dim=args.ir_out_dim,
        slot_dim=args.ir_out_dim,
        max_slots=40,
        merge_threshold=1.0,
        use_background_detection=True
    ).to(device)
    slot_attention.eval()

    # Determine if we need grid context (transformer always needs it, CNN only if cross-attention)
    use_transformer = args.model == "transformer"
    need_grid_context = use_transformer or args.use_cross_attention

    # Create datasets
    use_augment = not args.no_augment
    train_dataset = ShapeCorrespondenceDataset(
        train_puzzles, ir_encoder, slot_attention, device,
        similarity_threshold=args.similarity_threshold,
        store_grid_context=need_grid_context,
        augment=use_augment,
        num_augments=args.num_augments,
        dihedral_only=args.dihedral_only,
        color_only=args.color_only
    )
    # Validation set: no augmentation for consistent evaluation
    val_dataset = ShapeCorrespondenceDataset(
        val_puzzles, ir_encoder, slot_attention, device,
        similarity_threshold=args.similarity_threshold,
        store_grid_context=need_grid_context,
        augment=False
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # Create model
    if use_transformer:
        model = PixelTransformerPredictor(
            pixel_dim=args.pixel_dim,
            num_heads=args.num_heads,
            mlp_ratio=args.mlp_ratio,
            ir_encoder=ir_encoder,
            ir_out_dim=args.ir_out_dim,
            dropout=args.dropout,
            use_bbox_prediction=args.use_bbox_prediction,
            identity_scale_init=args.identity_scale,
        ).to(device)
        print(f"Using PixelTransformerPredictor (pixel_dim={args.pixel_dim}, heads={args.num_heads}, bbox={args.use_bbox_prediction}, identity_scale={args.identity_scale})")
    else:
        model = ShapeDorsalCNN(
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            use_onehot=args.use_onehot,
            use_cross_attention=args.use_cross_attention,
            ir_encoder=ir_encoder if args.use_cross_attention else None,
            ir_out_dim=args.ir_out_dim,
            use_bbox_prediction=args.use_bbox_prediction,
        ).to(device)
        print(f"Using ShapeDorsalCNN (hidden_dim={args.hidden_dim}, layers={args.num_layers})")

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)

    # Training loop
    best_val_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{args.epochs}")
        print(f"{'='*60}")

        # Transformer always uses grid context; both models can use bbox prediction
        use_grid_context = use_transformer or args.use_cross_attention
        use_bbox = args.use_bbox_prediction

        train_metrics = train_epoch(
            model, train_loader, optimizer, device,
            use_grid_context, use_bbox, args.bbox_loss_weight
        )
        val_metrics = evaluate(
            model, val_loader, device,
            use_grid_context, use_bbox
        )

        scheduler.step()

        # Build training log string
        train_log = f"Train - Loss: {train_metrics['loss']:.4f}, Pixel Acc: {train_metrics['pixel_accuracy']:.4f}"
        if use_bbox:
            train_log += f", BBox Loss: {train_metrics['bbox_loss']:.4f}"
        print(train_log)

        val_log = f"Val   - Loss: {val_metrics['loss']:.4f}, Pixel Acc: {val_metrics['pixel_accuracy']:.4f}, "
        val_log += f"Exact Match: {val_metrics['exact_match_rate']:.4f}"
        if use_bbox:
            val_log += f", BBox Loss: {val_metrics['bbox_loss']:.4f}"
        print(val_log)

        # Track best accuracy (no checkpoint saving to save disk space)
        if val_metrics['pixel_accuracy'] > best_val_acc:
            best_val_acc = val_metrics['pixel_accuracy']
            print(f"  -> New best (val acc: {best_val_acc:.4f})")

    print(f"\nTraining complete! Best val accuracy: {best_val_acc:.4f}")

    # Visualize test predictions for single puzzle mode
    if args.puzzle_id:
        print(f"\nGenerating test predictions visualization...")
        puzzle = puzzles[args.puzzle_id]
        save_path = os.path.join(args.save_dir, f"{args.puzzle_id}_test_predictions.png")
        visualize_test_predictions(
            model, ir_encoder, slot_attention,
            puzzle, args.puzzle_id, device,
            save_path=save_path
        )


if __name__ == "__main__":
    main()