#!/usr/bin/env python3
"""
Pixel-Level Color Prediction CNN - CORRESPONDENCE VERSION

This version forces the CNN to learn input-output correspondence by:
1. Using augmented PAIRS as positives (same aug applied to both input and output)
2. Using mismatched inputs as negatives (correct output, wrong input)
3. Using mismatched augmentations as negatives

The key insight: the CNN must see the SAME output labeled both "correct" and
"incorrect" depending on the input. This forces it to actually compare them.

Predicts what color each pixel SHOULD be (0-9).

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

class NConv(nn.Module):
    """N consecutive convolutions with BatchNorm and ReLU after each.

    Args:
        in_ch: Input channels
        out_ch: Output channels
        depth: Number of conv layers (1=single, 2=double, 3=triple, etc.)
        kernel_size: Kernel size for all convolutions

    Receptive field per block: depth * (kernel_size - 1) + 1
    For kernel_size=3: depth=1 -> 3x3, depth=2 -> 5x5, depth=3 -> 7x7, depth=4 -> 9x9
    """
    def __init__(self, in_ch: int, out_ch: int, depth: int = 2, kernel_size: int = 3):
        super().__init__()
        padding = (kernel_size - 1) // 2

        layers = []
        for i in range(depth):
            layers.extend([
                nn.Conv2d(in_ch if i == 0 else out_ch, out_ch, kernel_size, padding=padding),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        return self.conv(x)


def get_conv_block(in_ch: int, out_ch: int, conv_depth: int = 2, kernel_size: int = 3):
    """Factory function to get a conv block with specified depth."""
    return NConv(in_ch, out_ch, depth=conv_depth, kernel_size=kernel_size)


class Down(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, conv_depth: int = 2, kernel_size: int = 3):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = get_conv_block(in_ch, out_ch, conv_depth, kernel_size)

    def forward(self, x):
        return self.conv(self.pool(x))


class Up(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, conv_depth: int = 2, kernel_size: int = 3):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
        self.conv = get_conv_block(in_ch, out_ch, conv_depth, kernel_size)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diff_h = x2.size(2) - x1.size(2)
        diff_w = x2.size(3) - x1.size(3)
        x1 = F.pad(x1, [diff_w // 2, diff_w - diff_w // 2,
                       diff_h // 2, diff_h - diff_h // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class UpNoSkip(nn.Module):
    """Decoder block without skip connections - forces all info through bottleneck."""
    def __init__(self, in_ch: int, out_ch: int, conv_depth: int = 2, kernel_size: int = 3):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = get_conv_block(out_ch, out_ch, conv_depth, kernel_size)

    def forward(self, x, target_size=None):
        x = self.up(x)
        if target_size is not None:
            diff_h = target_size[0] - x.size(2)
            diff_w = target_size[1] - x.size(3)
            x = F.pad(x, [diff_w // 2, diff_w - diff_w // 2,
                         diff_h // 2, diff_h - diff_h // 2])
        return self.conv(x)


class IREncoder(nn.Module):
    """
    Instance Recognition encoder - produces features per pixel.

    Default architecture matches ir_checkpoints/test.pt:
    - conv1: (10, 128, 3x3) with BatchNorm + ReLU
    - conv2: (128, 128, 3x3) with BatchNorm + ReLU
    - conv3: (128, 128, 3x3) with BatchNorm + ReLU
    - conv4: (128, 64, 1x1) - final projection to out_dim features

    Args:
        hidden_dim: Number of channels in hidden conv layers (default: 128)
        out_dim: Output feature dimension per pixel (default: 64)
        num_layers: Number of 3x3 conv layers before projection (default: 3)
        kernel_size: Kernel size for conv layers (default: 3)
    """

    def __init__(self, hidden_dim: int = 128, out_dim: int = 64,
                 num_layers: int = 3, kernel_size: int = 3):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.num_layers = num_layers
        self.kernel_size = kernel_size
        padding = (kernel_size - 1) // 2

        # Build conv layers dynamically
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for i in range(num_layers):
            in_ch = 10 if i == 0 else hidden_dim
            self.convs.append(nn.Conv2d(in_ch, hidden_dim, kernel_size, padding=padding))
            self.bns.append(nn.BatchNorm2d(hidden_dim))

        # Final 1x1 projection to output dimension
        self.proj = nn.Conv2d(hidden_dim, out_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: One-hot encoded grid (B, 10, H, W)
        Returns:
            Features (B, H, W, out_dim)
        """
        for conv, bn in zip(self.convs, self.bns):
            x = F.relu(bn(conv(x)))
        x = self.proj(x)  # (B, out_dim, H, W)
        return x.permute(0, 2, 3, 1).contiguous()  # (B, H, W, out_dim)

    @classmethod
    def from_checkpoint(cls, path: str, device=None):
        """Load encoder weights from an instance recognition checkpoint.

        Automatically infers architecture (hidden_dim, out_dim, num_layers, kernel_size)
        from the checkpoint's state_dict.
        """
        ckpt = torch.load(path, map_location=device or 'cpu', weights_only=False)
        # Extract encoder weights from full model checkpoint
        state_dict = {k.replace('encoder.', ''): v
                      for k, v in ckpt['model_state_dict'].items()
                      if k.startswith('encoder.')}

        # Handle old checkpoint format (conv1, bn1, ...) vs new format (convs.0, bns.0, ...)
        if 'conv1.weight' in state_dict and 'convs.0.weight' not in state_dict:
            # Map old keys to new keys
            new_state_dict = {}
            for key, value in state_dict.items():
                new_key = key
                # Map convN -> convs.(N-1) for N in 1,2,3
                for i in range(1, 4):
                    new_key = new_key.replace(f'conv{i}.', f'convs.{i-1}.')
                    new_key = new_key.replace(f'bn{i}.', f'bns.{i-1}.')
                # Map conv4 -> proj (the final projection layer)
                new_key = new_key.replace('conv4.', 'proj.')
                new_state_dict[new_key] = value
            state_dict = new_state_dict

        # Infer architecture from state_dict
        # hidden_dim: output channels of first conv layer
        hidden_dim = state_dict['convs.0.weight'].shape[0]
        # out_dim: output channels of projection layer
        out_dim = state_dict['proj.weight'].shape[0]
        # num_layers: count conv layers
        num_layers = sum(1 for k in state_dict if k.startswith('convs.') and k.endswith('.weight'))
        # kernel_size: from first conv layer weight shape
        kernel_size = state_dict['convs.0.weight'].shape[2]

        encoder = cls(hidden_dim=hidden_dim, out_dim=out_dim,
                      num_layers=num_layers, kernel_size=kernel_size)
        encoder.load_state_dict(state_dict)
        return encoder


class SizeClassificationHead(nn.Module):
    """
    Predicts output grid size (height, width) from input grid using IR encoder.

    Uses the IR encoder to extract features from the input grid, then applies
    global average pooling and MLP heads to predict height and width as
    classification tasks (1-30 for each dimension).

    Args:
        ir_encoder: Pre-trained or untrained IREncoder instance
        max_size: Maximum grid dimension (default: 30)
        hidden_dim: Hidden dimension for MLP (default: 128)
        freeze_ir: Whether to freeze IR encoder weights (default: True)
    """

    def __init__(self, ir_encoder: IREncoder, max_size: int = 30,
                 hidden_dim: int = 128, freeze_ir: bool = True):
        super().__init__()
        self.ir_encoder = ir_encoder
        self.max_size = max_size
        self.freeze_ir = freeze_ir

        if freeze_ir:
            for p in self.ir_encoder.parameters():
                p.requires_grad = False

        ir_feature_dim = ir_encoder.out_dim

        # Global average pooling
        self.pool = nn.AdaptiveAvgPool2d(1)

        # Shared trunk
        self.trunk = nn.Sequential(
            nn.Linear(ir_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
        )

        # Separate heads for height and width (classes 0-29 map to sizes 1-30)
        self.height_head = nn.Linear(hidden_dim, max_size)
        self.width_head = nn.Linear(hidden_dim, max_size)

    def forward(self, input_grid: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            input_grid: (B, H, W) integer grid with values 0-9
        Returns:
            height_logits: (B, max_size) - logits for height classes
            width_logits: (B, max_size) - logits for width classes
        """
        # One-hot encode input
        inp_onehot = F.one_hot(input_grid.long(), NUM_COLORS).float()
        inp_onehot = inp_onehot.permute(0, 3, 1, 2).contiguous()  # (B, 10, H, W)

        # Get IR features
        features = self.ir_encoder(inp_onehot)  # (B, H, W, out_dim)
        features = features.permute(0, 3, 1, 2)  # (B, out_dim, H, W)

        # Global average pool
        pooled = self.pool(features).squeeze(-1).squeeze(-1)  # (B, out_dim)

        # Predict sizes through trunk + heads
        trunk_out = self.trunk(pooled)
        height_logits = self.height_head(trunk_out)
        width_logits = self.width_head(trunk_out)

        return height_logits, width_logits

    def predict_size(self, input_grid: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict output size as (height, width) integers.

        Args:
            input_grid: (B, H, W) integer grid
        Returns:
            heights: (B,) predicted heights (1-30)
            widths: (B,) predicted widths (1-30)
        """
        height_logits, width_logits = self.forward(input_grid)
        heights = height_logits.argmax(dim=1) + 1  # Convert from class (0-29) to size (1-30)
        widths = width_logits.argmax(dim=1) + 1
        return heights, widths

    @classmethod
    def from_ir_checkpoint(cls, ir_checkpoint_path: str, max_size: int = 30,
                           hidden_dim: int = 128, freeze_ir: bool = True,
                           device: torch.device = None):
        """Create SizeClassificationHead from an IR encoder checkpoint."""
        ir_encoder = IREncoder.from_checkpoint(ir_checkpoint_path, device)
        model = cls(ir_encoder, max_size=max_size, hidden_dim=hidden_dim, freeze_ir=freeze_ir)
        if device:
            model = model.to(device)
        return model

    def save_checkpoint(self, path: str, args: dict = None):
        """Save model checkpoint."""
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'ir_out_dim': self.ir_encoder.out_dim,
            'ir_hidden_dim': self.ir_encoder.hidden_dim,
            'ir_num_layers': self.ir_encoder.num_layers,
            'ir_kernel_size': self.ir_encoder.kernel_size,
            'max_size': self.max_size,
            'args': args or {},
        }
        torch.save(checkpoint, path)

    @classmethod
    def from_checkpoint(cls, checkpoint_path: str, device: torch.device = None):
        """Load SizeClassificationHead from checkpoint."""
        ckpt = torch.load(checkpoint_path, map_location=device or 'cpu', weights_only=False)

        # Reconstruct IR encoder
        ir_encoder = IREncoder(
            hidden_dim=ckpt.get('ir_hidden_dim', 128),
            out_dim=ckpt.get('ir_out_dim', 64),
            num_layers=ckpt.get('ir_num_layers', 3),
            kernel_size=ckpt.get('ir_kernel_size', 3),
        )

        args = ckpt.get('args', {})
        model = cls(
            ir_encoder,
            max_size=ckpt.get('max_size', 30),
            hidden_dim=args.get('size_hidden_dim', 128),
            freeze_ir=False,  # We're loading full weights
        )
        model.load_state_dict(ckpt['model_state_dict'])
        model.eval()

        if device:
            model = model.to(device)
        return model


class SpatialSelfAttention(nn.Module):
    """
    Single-head spatial self-attention over pixels with learned Q/K/V projections.
    Each pixel attends to all other pixels in the feature map.

    Args:
        dim: Feature dimension (number of channels)

    Set capture_attention=True to store attention weights for visualization.
    Access via .last_attention_weights after forward pass.
    """
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.scale = dim ** -0.5

        # Learned Q/K/V projections
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)

        self.capture_attention = False
        self.last_attention_weights = None  # (B, H*W, H*W) when captured

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        N = H * W

        # Reshape to (B, N, C)
        x_flat = x.reshape(B, C, N).permute(0, 2, 1)  # (B, N, C)

        # Project to Q, K, V
        q = self.q_proj(x_flat)  # (B, N, C) - "what am I looking for?"
        k = self.k_proj(x_flat)  # (B, N, C) - "what do I have to offer?"
        v = self.v_proj(x_flat)  # (B, N, C) - "what info to pass along?"

        # Attention scores
        scores = torch.bmm(q, k.transpose(1, 2)) * self.scale  # (B, N, N)
        attn = F.softmax(scores, dim=-1)

        # Optionally store attention weights for visualization
        if self.capture_attention:
            self.last_attention_weights = attn.detach().cpu()

        # Apply attention to values
        attended = torch.bmm(attn, v)  # (B, N, C)

        # Residual connection
        out = x_flat + attended

        return out.permute(0, 2, 1).reshape(B, C, H, W)


class CrossAttention(nn.Module):
    """
    Cross-attention: output embeddings attend to input embeddings.

    Each output pixel queries all input pixels to gather relevant information.
    This allows the model to learn "what in the input is relevant for predicting
    this output pixel?" before any convolutional processing.

    Args:
        query_dim: Dimension of query embeddings (from output)
        kv_dim: Dimension of key/value embeddings (from input). Defaults to query_dim.
        proj_dim: Dimension for attention computation. Defaults to query_dim.

    Set capture_attention=True to store attention weights for visualization.
    Access via .last_attention_weights after forward pass.
    """
    def __init__(self, query_dim: int, kv_dim: int = None, proj_dim: int = None):
        super().__init__()
        kv_dim = kv_dim if kv_dim is not None else query_dim
        proj_dim = proj_dim if proj_dim is not None else query_dim

        self.query_dim = query_dim
        self.kv_dim = kv_dim
        self.proj_dim = proj_dim
        self.scale = proj_dim ** -0.5

        # Q comes from output, K/V come from input (possibly different dimensions)
        self.q_proj = nn.Linear(query_dim, proj_dim)  # "What am I looking for?" (output)
        self.k_proj = nn.Linear(kv_dim, proj_dim)     # "What's available?" (input)
        self.v_proj = nn.Linear(kv_dim, proj_dim)     # "What to retrieve?" (input)

        # Output projection back to query_dim if proj_dim differs
        self.out_proj = nn.Linear(proj_dim, query_dim) if proj_dim != query_dim else nn.Identity()

        self.capture_attention = False
        self.last_attention_weights = None  # (B, H*W, H*W) when captured

    def forward(self, query: torch.Tensor, key_value: torch.Tensor) -> torch.Tensor:
        """
        Args:
            query: Output embeddings (B, H, W, query_dim) - queries come from here
            key_value: Input embeddings (B, H, W, kv_dim) - keys and values come from here

        Returns:
            Attended output embeddings (B, H, W, query_dim)
        """
        B, H, W, C_q = query.shape
        _, _, _, C_kv = key_value.shape
        N = H * W

        # Flatten spatial dimensions
        q_flat = query.reshape(B, N, C_q)       # (B, N, query_dim)
        kv_flat = key_value.reshape(B, N, C_kv)  # (B, N, kv_dim)

        # Project to proj_dim
        q = self.q_proj(q_flat)   # (B, N, proj_dim) - from output
        k = self.k_proj(kv_flat)  # (B, N, proj_dim) - from input
        v = self.v_proj(kv_flat)  # (B, N, proj_dim) - from input

        # Attention scores: each output pixel attends to all input pixels
        scores = torch.bmm(q, k.transpose(1, 2)) * self.scale  # (B, N, N)
        attn = F.softmax(scores, dim=-1)

        # Optionally store attention weights for visualization
        if self.capture_attention:
            self.last_attention_weights = attn.detach().cpu()

        # Apply attention: gather information from input based on attention
        attended = torch.bmm(attn, v)  # (B, N, proj_dim)

        # Project attended values back to query_dim if needed
        attended = self.out_proj(attended)  # (B, N, query_dim)

        # Residual connection: output embedding + attended input info
        out = q_flat + attended  # (B, N, query_dim)

        return out.reshape(B, H, W, self.query_dim)


class PixelErrorCNN(nn.Module):
    """
    U-Net style CNN for comparing input and output grids.

    Concatenates input and output embeddings along the channel dimension,
    allowing convolutions to see both grids at corresponding spatial positions.

    Predicts what color each pixel SHOULD be (0-9).
    Optionally also predicts output size (height, width) using IR encoder.
    """

    def __init__(self, hidden_dim: int = 64,
                 no_skip: bool = False, num_layers: int = 3, conv_depth: int = 2,
                 kernel_size: int = 3, use_onehot: bool = False, out_kernel_size: int = 1,
                 use_attention: bool = False, use_cross_attention: bool = False,
                 ir_checkpoint: str = None, use_untrained_ir: bool = False,
                 ir_hidden_dim: int = 128, ir_out_dim: int = 64,
                 ir_num_layers: int = 3, ir_kernel_size: int = 3,
                 freeze_ir: bool = True,
                 predict_size: bool = False, size_hidden_dim: int = 128):
        super().__init__()

        self.num_classes = NUM_COLORS  # Always 10 for color prediction
        self.no_skip = no_skip
        self.num_layers = num_layers
        self.conv_depth = conv_depth
        self.kernel_size = kernel_size
        self.use_onehot = use_onehot
        self.out_kernel_size = out_kernel_size
        self.use_attention = use_attention
        self.use_cross_attention = use_cross_attention
        self.ir_checkpoint = ir_checkpoint
        self.use_untrained_ir = use_untrained_ir
        self.ir_hidden_dim = ir_hidden_dim
        self.ir_out_dim = ir_out_dim
        self.ir_num_layers = ir_num_layers
        self.ir_kernel_size = ir_kernel_size
        self.freeze_ir = freeze_ir
        self.predict_size = predict_size
        self.size_hidden_dim = size_hidden_dim

        if use_onehot:
            # One-hot (10) + nonzero mask (1) = 11 channels per grid
            embed_dim = 11
        else:
            # Learned embeddings
            embed_dim = 16
            self.input_embed = nn.Embedding(NUM_COLORS, 16)
            self.output_embed = nn.Embedding(NUM_COLORS, 16)

        self.embed_dim = embed_dim

        # Instance Recognition encoder for cross-attention (optional)
        self.ir_encoder = None
        self.ir_self_attention = None
        if use_cross_attention and (ir_checkpoint or use_untrained_ir):
            if ir_checkpoint:
                self.ir_encoder = IREncoder.from_checkpoint(ir_checkpoint, DEVICE)
                ir_feature_dim = self.ir_encoder.out_dim  # Get from loaded checkpoint
            else:
                # Use randomly initialized IR encoder (for ablation study)
                self.ir_encoder = IREncoder(
                    hidden_dim=ir_hidden_dim,
                    out_dim=ir_out_dim,
                    num_layers=ir_num_layers,
                    kernel_size=ir_kernel_size
                )
                ir_feature_dim = ir_out_dim
            if freeze_ir and ir_checkpoint:
                # Only freeze if using pretrained weights
                for p in self.ir_encoder.parameters():
                    p.requires_grad = False
            # Self-attention on IR features before cross-attention
            # All input pixels attend to each other before being queried by output
            self.ir_self_attention = SpatialSelfAttention(ir_feature_dim)
            # Cross-attention: query_dim=embed_dim (16 or 11), kv_dim=ir_feature_dim
            self.cross_attention = CrossAttention(
                query_dim=embed_dim,
                kv_dim=ir_feature_dim,
                proj_dim=embed_dim  # Project to embedding dimension
            )
        elif use_cross_attention:
            # Original behavior - same dimension for query and kv
            self.cross_attention = CrossAttention(embed_dim)
        else:
            self.cross_attention = None

        # Channel concatenation: embed_dim * 2 (inp, out)
        in_channels = embed_dim * 2

        base_ch = hidden_dim

        # Encoder - create layers based on num_layers
        self.inc = get_conv_block(in_channels, base_ch, conv_depth, kernel_size)
        if num_layers >= 1:
            self.down1 = Down(base_ch, base_ch * 2, conv_depth, kernel_size)
        if num_layers >= 2:
            self.down2 = Down(base_ch * 2, base_ch * 4, conv_depth, kernel_size)
        if num_layers >= 3:
            self.down3 = Down(base_ch * 4, base_ch * 8, conv_depth, kernel_size)

        # Decoder - create layers based on num_layers (no up layers needed for num_layers=0)
        if num_layers >= 1:
            if no_skip:
                # No skip connections - all info must flow through bottleneck
                if num_layers >= 3:
                    self.up1 = UpNoSkip(base_ch * 8, base_ch * 4, conv_depth, kernel_size)
                if num_layers >= 2:
                    self.up2 = UpNoSkip(base_ch * 4, base_ch * 2, conv_depth, kernel_size)
                self.up3 = UpNoSkip(base_ch * 2, base_ch, conv_depth, kernel_size)
            else:
                # Standard U-Net with skip connections
                if num_layers >= 3:
                    self.up1 = Up(base_ch * 8, base_ch * 4, conv_depth, kernel_size)
                if num_layers >= 2:
                    self.up2 = Up(base_ch * 4, base_ch * 2, conv_depth, kernel_size)
                self.up3 = Up(base_ch * 2, base_ch, conv_depth, kernel_size)

        # Output: 10 channels for color prediction
        out_padding = (out_kernel_size - 1) // 2
        self.outc = nn.Conv2d(base_ch, NUM_COLORS, kernel_size=out_kernel_size, padding=out_padding)

        # Optional spatial self-attention before output
        self.attention = SpatialSelfAttention(base_ch) if use_attention else None

        # Size prediction head (uses IR encoder features)
        self.size_head = None
        if predict_size:
            # Ensure we have an IR encoder for size prediction
            if self.ir_encoder is None:
                if ir_checkpoint:
                    self.ir_encoder = IREncoder.from_checkpoint(ir_checkpoint, DEVICE)
                    ir_feature_dim = self.ir_encoder.out_dim
                else:
                    # Create untrained IR encoder for size prediction
                    self.ir_encoder = IREncoder(
                        hidden_dim=ir_hidden_dim,
                        out_dim=ir_out_dim,
                        num_layers=ir_num_layers,
                        kernel_size=ir_kernel_size
                    )
                    ir_feature_dim = ir_out_dim
                if freeze_ir and ir_checkpoint:
                    for p in self.ir_encoder.parameters():
                        p.requires_grad = False
            else:
                ir_feature_dim = self.ir_encoder.out_dim

            # Size prediction MLP (global pool -> trunk -> height/width heads)
            self.size_pool = nn.AdaptiveAvgPool2d(1)
            self.size_trunk = nn.Sequential(
                nn.Linear(ir_feature_dim, size_hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(size_hidden_dim, size_hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
            )
            self.height_head = nn.Linear(size_hidden_dim, GRID_SIZE)  # 30 classes
            self.width_head = nn.Linear(size_hidden_dim, GRID_SIZE)   # 30 classes

    def _encode_grid(self, grid: torch.Tensor) -> torch.Tensor:
        """Encode grid to (B, H, W, 11) one-hot + nonzero mask."""
        # One-hot: (B, H, W) -> (B, H, W, 10)
        onehot = F.one_hot(grid.long(), num_classes=NUM_COLORS).float()
        # Nonzero mask: (B, H, W) -> (B, H, W, 1)
        nonzero = (grid > 0).float().unsqueeze(-1)
        return torch.cat([onehot, nonzero], dim=-1)

    def forward(self, input_grid: torch.Tensor, output_grid: torch.Tensor):
        """
        Forward pass for pixel prediction and optionally size prediction.

        Returns:
            If predict_size=False: pixel_logits (B, 10, H, W)
            If predict_size=True: dict with 'pixel_logits', 'height_logits', 'width_logits'
        """
        # Compute IR features for size prediction (if enabled) or cross-attention
        ir_features_for_size = None
        if self.ir_encoder is not None:
            inp_onehot = F.one_hot(input_grid.long(), NUM_COLORS).float()
            inp_onehot = inp_onehot.permute(0, 3, 1, 2).contiguous()  # (B, 10, H, W)
            ir_features = self.ir_encoder(inp_onehot)  # (B, H, W, out_dim)

            # Save for size prediction before any modifications
            if self.predict_size:
                ir_features_for_size = ir_features.permute(0, 3, 1, 2).contiguous()  # (B, out_dim, H, W)

        # Encode output grid (for queries and concatenation)
        if self.use_onehot:
            out_emb = self._encode_grid(output_grid)  # (B, 30, 30, 11)
        else:
            out_emb = self.output_embed(output_grid)  # (B, 30, 30, 16)

        # Apply cross-attention: output pixels attend to input pixels
        # This happens BEFORE convolutions so global info is available early
        if self.cross_attention is not None:
            if self.ir_encoder is not None:
                # Use IR encoder features for cross-attention keys/values
                # Apply self-attention so all input pixels can attend to each other
                ir_features_attn = ir_features.permute(0, 3, 1, 2).contiguous()  # (B, 64, H, W)
                ir_features_attn = self.ir_self_attention(ir_features_attn)  # (B, 64, H, W)
                ir_features_attn = ir_features_attn.permute(0, 2, 3, 1).contiguous()  # (B, H, W, 64)
                out_emb = self.cross_attention(query=out_emb, key_value=ir_features_attn)
            else:
                # Original behavior - use same embedding type for input
                if self.use_onehot:
                    inp_emb_for_attn = self._encode_grid(input_grid)
                else:
                    inp_emb_for_attn = self.input_embed(input_grid)
                out_emb = self.cross_attention(query=out_emb, key_value=inp_emb_for_attn)

        # Encode input grid for U-Net concatenation (always use original embeddings)
        if self.use_onehot:
            inp_emb = self._encode_grid(input_grid)   # (B, 30, 30, 11)
        else:
            inp_emb = self.input_embed(input_grid)    # (B, 30, 30, 16)

        # Channel concatenation: input and output side by side in channel dim
        x = torch.cat([inp_emb, out_emb], dim=-1)  # (B, 30, 30, 32) or (B, 30, 30, 22)

        x = x.permute(0, 3, 1, 2).contiguous()

        # U-Net forward - encoder
        x1 = self.inc(x)
        if self.num_layers >= 1:
            x2 = self.down1(x1)
        if self.num_layers >= 2:
            x3 = self.down2(x2)
        if self.num_layers >= 3:
            x4 = self.down3(x3)

        # Determine bottleneck based on num_layers
        if self.num_layers == 0:
            bottleneck = x1  # No downsampling, inc output is the "bottleneck"
        elif self.num_layers == 1:
            bottleneck = x2
        elif self.num_layers == 2:
            bottleneck = x3
        else:  # num_layers == 3
            bottleneck = x4

        # Decoder - based on num_layers
        if self.num_layers == 0:
            # No encoder/decoder layers - just use bottleneck (which is x1) directly
            x = bottleneck
        elif self.no_skip:
            # No skip connections - pass target sizes for proper upsampling
            if self.num_layers == 3:
                x = self.up1(bottleneck, target_size=(x3.size(2), x3.size(3)))
                x = self.up2(x, target_size=(x2.size(2), x2.size(3)))
                x = self.up3(x, target_size=(x1.size(2), x1.size(3)))
            elif self.num_layers == 2:
                x = self.up2(bottleneck, target_size=(x2.size(2), x2.size(3)))
                x = self.up3(x, target_size=(x1.size(2), x1.size(3)))
            else:  # num_layers == 1
                x = self.up3(bottleneck, target_size=(x1.size(2), x1.size(3)))
        else:
            # Standard U-Net with skip connections
            if self.num_layers == 3:
                x = self.up1(bottleneck, x3)
                x = self.up2(x, x2)
                x = self.up3(x, x1)
            elif self.num_layers == 2:
                x = self.up2(bottleneck, x2)
                x = self.up3(x, x1)
            else:  # num_layers == 1
                x = self.up3(bottleneck, x1)

        # Apply spatial self-attention if enabled
        if self.attention is not None:
            x = self.attention(x)

        pixel_logits = self.outc(x)  # (B, 10, H, W) for color prediction

        # Size prediction (if enabled)
        if self.predict_size and ir_features_for_size is not None:
            # Global pool IR features and predict size
            pooled = self.size_pool(ir_features_for_size).squeeze(-1).squeeze(-1)  # (B, ir_dim)
            trunk_out = self.size_trunk(pooled)
            height_logits = self.height_head(trunk_out)  # (B, 30)
            width_logits = self.width_head(trunk_out)    # (B, 30)

            return {
                'pixel_logits': pixel_logits,
                'height_logits': height_logits,
                'width_logits': width_logits,
            }

        return pixel_logits

    def predict_proba(self, input_grid: torch.Tensor, output_grid: torch.Tensor) -> torch.Tensor:
        """Return softmax probabilities for color prediction"""
        output = self.forward(input_grid, output_grid)
        logits = output['pixel_logits'] if isinstance(output, dict) else output
        return F.softmax(logits, dim=1)

    def predict_colors(self, input_grid: torch.Tensor, output_grid: torch.Tensor) -> torch.Tensor:
        """Return predicted color per pixel"""
        output = self.forward(input_grid, output_grid)
        logits = output['pixel_logits'] if isinstance(output, dict) else output
        return logits.argmax(dim=1)  # (B, H, W)

    def predict_output_size(self, input_grid: torch.Tensor, output_grid: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict output size (height, width).

        Returns:
            heights: (B,) predicted heights (1-30)
            widths: (B,) predicted widths (1-30)
        """
        if not self.predict_size:
            raise RuntimeError("Model was not initialized with predict_size=True")
        output = self.forward(input_grid, output_grid)
        heights = output['height_logits'].argmax(dim=1) + 1  # Convert 0-29 to 1-30
        widths = output['width_logits'].argmax(dim=1) + 1
        return heights, widths

    @classmethod
    def from_checkpoint(cls, checkpoint_path: str, device: torch.device = None):
        checkpoint = torch.load(checkpoint_path, map_location=device or 'cpu', weights_only=False)
        args = checkpoint.get('args', {})
        hidden_dim = args.get('hidden_dim', 64)
        no_skip = args.get('no_skip', False)
        num_layers = args.get('num_layers', 3)
        kernel_size = args.get('kernel_size', 3)

        # Handle conv_depth with backwards compatibility for old single_conv/triple_conv checkpoints
        if 'conv_depth' in args:
            conv_depth = args['conv_depth']
        else:
            # Old checkpoint format - convert single_conv/triple_conv to conv_depth
            single_conv = args.get('single_conv', False)
            triple_conv = args.get('triple_conv', False)
            if single_conv:
                conv_depth = 1
            elif triple_conv:
                conv_depth = 3
            else:
                conv_depth = 2

        use_onehot = args.get('use_onehot', False)
        out_kernel_size = args.get('out_kernel_size', 1)
        use_attention = args.get('use_attention', False)
        use_cross_attention = args.get('use_cross_attention', False)
        ir_checkpoint = args.get('ir_checkpoint', None)
        use_untrained_ir = args.get('use_untrained_ir', False)
        ir_hidden_dim = args.get('ir_hidden_dim', 128)
        ir_out_dim = args.get('ir_out_dim', 64)
        ir_num_layers = args.get('ir_num_layers', 3)
        ir_kernel_size = args.get('ir_kernel_size', 3)
        freeze_ir = args.get('freeze_ir', True)
        predict_size = args.get('predict_size', False)
        size_hidden_dim = args.get('size_hidden_dim', 128)

        model = cls(
            hidden_dim=hidden_dim,
            no_skip=no_skip,
            num_layers=num_layers,
            conv_depth=conv_depth,
            kernel_size=kernel_size,
            use_onehot=use_onehot,
            out_kernel_size=out_kernel_size,
            use_attention=use_attention,
            use_cross_attention=use_cross_attention,
            ir_checkpoint=ir_checkpoint,
            use_untrained_ir=use_untrained_ir,
            ir_hidden_dim=ir_hidden_dim,
            ir_out_dim=ir_out_dim,
            ir_num_layers=ir_num_layers,
            ir_kernel_size=ir_kernel_size,
            freeze_ir=freeze_ir,
            predict_size=predict_size,
            size_hidden_dim=size_hidden_dim,
        )
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
    3. NEGATIVE - wrong input: (wrong_input, correct_output) â† CRITICAL
    4. NEGATIVE - mismatched aug: (aug1(input), aug2(output)) â† CRITICAL
    5. NEGATIVE - all_zeros: output is all zeros (blank)
    6. NEGATIVE - constant_fill: output is all one color (1-9)
    7. NEGATIVE - random_noise: output is random values
    8. NEGATIVE - color_swap: one color swapped with another

    Types 3-8 force the CNN to detect various failure modes!

    Returns target_colors where each pixel has the correct color (0-9).
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
        color_only: bool = False,
        include_test: bool = True,  # Whether to include test examples
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
        self.color_only = color_only
        self.include_test = include_test

        # Extract all (input, output) pairs
        self.examples = []

        for puzzle_id, puzzle in puzzles.items():
            for example in puzzle.get("train", []):
                inp = np.array(example["input"], dtype=np.uint8)
                out = np.array(example["output"], dtype=np.uint8)
                self.examples.append((inp, out, puzzle_id))
            if include_test:
                for example in puzzle.get("test", []):
                    if "output" in example:
                        inp = np.array(example["input"], dtype=np.uint8)
                        out = np.array(example["output"], dtype=np.uint8)
                        self.examples.append((inp, out, puzzle_id))

        print(f"Loaded {len(self.examples)} examples from {len(puzzles)} puzzles (include_test={include_test})")
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
        """Get augmentation params based on augment, dihedral_only, and color_only settings"""
        if not self.augment:
            return 0, np.arange(10, dtype=np.uint8)  # Identity transform
        elif self.dihedral_only:
            # Random dihedral transform, but identity color mapping
            trans_id = random.randint(0, 7)
            color_map = np.arange(10, dtype=np.uint8)
            return trans_id, color_map
        elif self.color_only:
            # Identity dihedral transform, but random color mapping
            trans_id = 0
            color_map = random_color_permutation()
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
            # Target: the correct output itself (for both binary and color modes)
            target_correct = aug_output
            is_positive = 1.0

        elif sample_type == "corrupted":
            # Corrupted output (standard negative)
            trans_id, color_map = self._get_augmentation()
            aug_input = apply_augmentation(input_grid, trans_id, color_map)
            aug_output = apply_augmentation(correct_output, trans_id, color_map)
            corrupted = corrupt_output(aug_output)

            final_input = aug_input
            final_output = corrupted
            target_correct = aug_output  # What it SHOULD be
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
            # For wrong_input: we don't know what the "correct" output should be
            # since the input doesn't match. Mark entire content as "wrong"
            # For color mode, we can't know the correct colors, so use special handling
            target_correct = aug_output  # Use output as target (will be marked wrong via mask)
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
            elif self.color_only:
                # Only mismatch color permutations, not dihedral transforms
                trans_id_1, trans_id_2 = 0, 0
                color_map_1 = random_color_permutation()
                color_map_2 = random_color_permutation()
                # Ensure they're actually different
                while np.array_equal(color_map_1, color_map_2):
                    color_map_2 = random_color_permutation()
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
            # Similar to wrong_input - mismatch means we can't know correct colors
            target_correct = aug_output
            is_positive = 0.0

        elif sample_type == "all_zeros":
            # Output is all zeros (blank/empty) - obvious garbage
            trans_id, color_map = self._get_augmentation()
            aug_input = apply_augmentation(input_grid, trans_id, color_map)
            aug_output = apply_augmentation(correct_output, trans_id, color_map)
            garbage_output = generate_all_zeros(correct_output)

            final_input = aug_input
            final_output = garbage_output
            target_correct = aug_output  # What it SHOULD be
            is_positive = 0.0

        elif sample_type == "constant_fill":
            # Output is all one color (1-9) - obvious garbage
            trans_id, color_map = self._get_augmentation()
            aug_input = apply_augmentation(input_grid, trans_id, color_map)
            aug_output = apply_augmentation(correct_output, trans_id, color_map)
            garbage_output = generate_constant_fill(correct_output)

            final_input = aug_input
            final_output = garbage_output
            target_correct = aug_output  # What it SHOULD be
            is_positive = 0.0

        elif sample_type == "random_noise":
            # Output is random noise - obvious garbage
            trans_id, color_map = self._get_augmentation()
            aug_input = apply_augmentation(input_grid, trans_id, color_map)
            aug_output = apply_augmentation(correct_output, trans_id, color_map)
            garbage_output = generate_random_noise(correct_output)

            final_input = aug_input
            final_output = garbage_output
            target_correct = aug_output  # What it SHOULD be
            is_positive = 0.0

        else:  # color_swap
            # One color swapped with another - subtle corruption
            trans_id, color_map = self._get_augmentation()
            aug_input = apply_augmentation(input_grid, trans_id, color_map)
            aug_output = apply_augmentation(correct_output, trans_id, color_map)
            swapped_output = generate_color_swap(aug_output)

            final_input = aug_input
            final_output = swapped_output
            target_correct = aug_output  # What it SHOULD be
            is_positive = 0.0

        # Track target output dimensions (after augmentation)
        # For positive samples, use the augmented output shape
        # For negatives, use the correct output shape (what it SHOULD be)
        target_out_h, target_out_w = target_correct.shape

        # Pad grids
        final_input = self._pad_grid(final_input)
        final_output = self._pad_grid(final_output)
        target_correct = self._pad_grid(target_correct)

        # Compute binary mask (always needed for metrics)
        pixel_mask = (final_output == target_correct).astype(np.float32)
        
        # Handle special cases for wrong_input and mismatched_aug
        # These have the entire content area marked as wrong
        if sample_type == "wrong_input":
            h, w = correct_output.shape
            pixel_mask = np.ones((GRID_SIZE, GRID_SIZE), dtype=np.float32)
            pixel_mask[:h, :w] = 0.0  # Only content region marked as error
        elif sample_type == "mismatched_aug":
            h1, w1 = input_grid.shape
            h2, w2 = correct_output.shape
            # After augmentation, dimensions might be swapped
            trans_id_1 = trans_id_1 if 'trans_id_1' in dir() else trans_id
            if trans_id_1 in [1, 3, 6, 7]:  # Transforms that swap dimensions
                h1, w1 = w1, h1
            h_max, w_max = max(h1, h2), max(w1, w2)
            pixel_mask = np.ones((GRID_SIZE, GRID_SIZE), dtype=np.float32)
            pixel_mask[:h_max, :w_max] = 0.0

        # Ensure contiguous
        final_input = np.ascontiguousarray(final_input)
        final_output = np.ascontiguousarray(final_output)
        pixel_mask = np.ascontiguousarray(pixel_mask)
        target_correct = np.ascontiguousarray(target_correct)

        return (
            torch.from_numpy(final_input.copy()).long(),
            torch.from_numpy(final_output.copy()).long(),
            torch.from_numpy(target_correct.copy()).long(),  # Target colors (0-9)
            torch.from_numpy(pixel_mask.copy()).float(),  # Still useful for metrics
            torch.tensor([target_out_h, target_out_w], dtype=torch.long),  # Target output dims
        )


# =============================================================================
# Training
# =============================================================================

class TestEvalDataset(Dataset):
    """
    Simple dataset for evaluating on held-out test examples.
    No augmentation, no corruptions - just raw (input, output) pairs.

    For the critical generalization test: did the CNN learn the rule?
    """

    def __init__(self, puzzles: Dict, test_only: bool = True):
        self.examples = []
        
        for puzzle_id, puzzle in puzzles.items():
            # Get examples from the appropriate split
            if test_only:
                examples = puzzle.get("test", [])
            else:
                examples = puzzle.get("train", [])
            
            for example in examples:
                if "output" in example:
                    inp = np.array(example["input"], dtype=np.uint8)
                    out = np.array(example["output"], dtype=np.uint8)
                    self.examples.append((inp, out, puzzle_id))
        
        split_name = "test" if test_only else "train"
        print(f"TestEvalDataset: {len(self.examples)} {split_name} examples from {len(puzzles)} puzzles")
    
    def __len__(self):
        return len(self.examples)
    
    def _pad_grid(self, grid: np.ndarray) -> np.ndarray:
        h, w = grid.shape
        padded = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.uint8)
        padded[:h, :w] = grid
        return padded
    
    def __getitem__(self, idx: int):
        input_grid, output_grid, puzzle_id = self.examples[idx]
        
        # Pad grids
        input_padded = self._pad_grid(input_grid)
        output_padded = self._pad_grid(output_grid)
        
        # Store original dimensions for accurate evaluation and display
        inp_h, inp_w = input_grid.shape
        out_h, out_w = output_grid.shape
        
        return (
            torch.from_numpy(input_padded).long(),
            torch.from_numpy(output_padded).long(),
            torch.tensor([inp_h, inp_w, out_h, out_w], dtype=torch.long),  # Both input and output dims
            puzzle_id
        )


def visualize_prediction(
    input_grid: np.ndarray,
    target_grid: np.ndarray, 
    pred_grid: np.ndarray,
    inp_h: int,
    inp_w: int,
    out_h: int, 
    out_w: int,
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
    
    # Extract actual regions using provided dimensions
    inp = input_grid[:inp_h, :inp_w]
    tgt = target_grid[:out_h, :out_w]
    pred = pred_grid[:out_h, :out_w]
    
    # Calculate errors
    errors = (pred != tgt).sum()
    total = out_h * out_w
    
    print(f"\n{'â•'*50}")
    if errors == 0:
        print(f"\033[92mâœ“ PERFECT: {puzzle_id}\033[0m")
    else:
        print(f"\033[91mâœ— {puzzle_id}: {errors}/{total} errors ({100*errors/total:.1f}% wrong)\033[0m")
    print(f"  Input: {inp_h}Ã—{inp_w} â†’ Output: {out_h}Ã—{out_w}")
    print(f"{'â•'*50}")
    
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
    print_grid(tgt, "TARGET", out_h, out_w)
    print_grid(pred, "PREDICTION (errors in red)", out_h, out_w, highlight_errors=True, target=tgt)
    
    print()


def evaluate_test_examples(
    model: nn.Module,
    dataset: TestEvalDataset,
    device: torch.device,
    verbose: bool = True,
    visualize: bool = True,
    recursive_iters: int = 1,
) -> Dict[str, float]:
    """
    Evaluate model on held-out test examples.
    This is the critical generalization test.

    With recursive_iters > 1, we feed the model's output back as the candidate
    and iterate, tracking accuracy at each step.
    """
    model.eval()
    
    total_pixels = 0
    total_correct = 0
    total_examples = 0
    perfect_examples = 0
    
    # Track per-iteration stats
    iter_stats = {i: {"correct": 0, "total": 0} for i in range(recursive_iters)}
    
    results = []
    
    with torch.no_grad():
        for i in range(len(dataset)):
            input_grid, output_grid, dims, puzzle_id = dataset[i]
            inp_h, inp_w, out_h, out_w = dims[0].item(), dims[1].item(), dims[2].item(), dims[3].item()
            
            inp_t = input_grid.unsqueeze(0).to(device)
            target_colors = output_grid.numpy()
            
            # Start with blank candidate
            candidate = torch.zeros_like(input_grid).unsqueeze(0).to(device)

            # Recursive iteration
            for iter_idx in range(recursive_iters):
                # Predict colors from (input, current_candidate)
                logits = model(inp_t, candidate)  # (1, 10, H, W)
                pred_colors = logits.argmax(dim=1)  # (1, H, W)

                # Track accuracy at this iteration
                pred_np = pred_colors[0].cpu().numpy()
                pred_region = pred_np[:out_h, :out_w]
                target_region = target_colors[:out_h, :out_w]

                correct = (pred_region == target_region).sum()
                total = out_h * out_w

                iter_stats[iter_idx]["correct"] += correct
                iter_stats[iter_idx]["total"] += total

                # Update candidate for next iteration
                candidate = pred_colors.long()

            # Final results use last iteration
            is_perfect = (correct == total)

            # Visualize final result
            if visualize:
                visualize_prediction(
                    input_grid.numpy(),
                    target_colors,
                    pred_np,
                    inp_h, inp_w,
                    out_h, out_w,
                    puzzle_id
                )

            total_correct += correct
            total_pixels += total
            total_examples += 1
            if is_perfect:
                perfect_examples += 1
            
            results.append({
                'puzzle_id': puzzle_id,
                'correct': correct,
                'total': total,
                'accuracy': correct / total,
                'perfect': is_perfect
            })
            
            if verbose and not visualize:
                status = "âœ“ PERFECT" if is_perfect else f"âœ— {correct}/{total} ({100*correct/total:.1f}%)"
                print(f"  Example {i+1}: {puzzle_id} - {status}")
    
    overall_accuracy = total_correct / max(total_pixels, 1)
    perfect_rate = perfect_examples / max(total_examples, 1)
    
    # Print per-iteration accuracy if recursive
    if recursive_iters > 1 and verbose:
        print(f"\n{'─'*60}")
        print("ACCURACY BY ITERATION:")
        for iter_idx in range(recursive_iters):
            iter_correct = iter_stats[iter_idx]["correct"]
            iter_total = iter_stats[iter_idx]["total"]
            iter_acc = iter_correct / max(iter_total, 1)
            delta = ""
            if iter_idx > 0:
                prev_acc = iter_stats[iter_idx-1]["correct"] / max(iter_stats[iter_idx-1]["total"], 1)
                diff = iter_acc - prev_acc
                delta = f" ({'+' if diff >= 0 else ''}{100*diff:.2f}%)"
            print(f"  Iteration {iter_idx}: {100*iter_acc:.2f}%{delta}")
        print(f"{'─'*60}")
    
    return {
        'pixel_accuracy': overall_accuracy,
        'perfect_rate': perfect_rate,
        'total_examples': total_examples,
        'perfect_examples': perfect_examples,
        'results': results,
        'iter_stats': iter_stats if recursive_iters > 1 else None,
    }


# =============================================================================
# Layer-wise Ablation Analysis
# =============================================================================

def run_layer_ablation(
    model: nn.Module,
    test_dataset,
    device: torch.device,
    verbose: bool = True
) -> Dict[str, Dict]:
    """
    Run layer-wise ablation analysis to understand what each layer contributes.

    For each layer, we temporarily zero out its output and measure the impact
    on test accuracy. Larger drops indicate more critical layers.

    Returns:
        Dict mapping layer names to their ablation results
    """
    model.eval()

    # First, get baseline accuracy without any ablation
    print("\n" + "="*70)
    print("LAYER-WISE ABLATION ANALYSIS")
    print("="*70)
    print("Testing which layers are critical for this transformation...")
    print("Baseline = normal model, then we zero each layer's output one at a time.\n")

    baseline_results = evaluate_test_examples(
        model, test_dataset, device, verbose=False, visualize=False
    )
    baseline_acc = baseline_results['pixel_accuracy']
    baseline_perfect = baseline_results['perfect_rate']
    
    print(f"BASELINE (no ablation):")
    print(f"  Pixel Accuracy: {baseline_acc:.2%}")
    print(f"  Perfect Rate:   {baseline_perfect:.2%}")
    print(f"{'─'*70}\n")

    # PixelErrorCNN (U-Net) layers - build based on num_layers
    encoder_layers = [
        ("inc (initial conv)", model.inc),
    ]
    if model.num_layers >= 1:
        encoder_layers.append(("down1 (encoder level 1)", model.down1))
    if model.num_layers >= 2:
        encoder_layers.append(("down2 (encoder level 2)", model.down2))
    if model.num_layers >= 3:
        encoder_layers.append(("down3 (encoder level 3 / bottleneck)", model.down3))

    decoder_layers = []
    if model.num_layers >= 3:
        decoder_layers.append(("up1 (decoder level 1)", model.up1))
    if model.num_layers >= 2:
        decoder_layers.append(("up2 (decoder level 2)", model.up2))
    if model.num_layers >= 1:
        decoder_layers.append(("up3 (decoder level 3)", model.up3))

    layer_groups = {}
    # Only include embeddings group if model uses learned embeddings (not one-hot)
    if not model.use_onehot:
        layer_groups["Embeddings"] = [
            ("input_embed", model.input_embed),
            ("output_embed", model.output_embed),
        ]
    layer_groups["Encoder"] = encoder_layers
    if decoder_layers:  # Only add Decoder group if there are decoder layers
        layer_groups["Decoder"] = decoder_layers
    
    # Store ablation results
    ablation_results = {}
    
    # Hook storage
    handles = []
    ablation_active = {"enabled": False, "layer": None}
    
    def make_zero_hook(layer_name):
        """Create a hook that zeros out the layer's output when ablation is active."""
        def hook(module, input, output):
            if ablation_active["enabled"] and ablation_active["layer"] == layer_name:
                if isinstance(output, torch.Tensor):
                    return torch.zeros_like(output)
                elif isinstance(output, tuple):
                    return tuple(torch.zeros_like(o) if isinstance(o, torch.Tensor) else o for o in output)
            return output
        return hook
    
    # Register hooks on all layers
    all_layers = []
    for group_name, layers in layer_groups.items():
        for layer_name, layer_module in layers:
            all_layers.append((group_name, layer_name, layer_module))
            handle = layer_module.register_forward_hook(make_zero_hook(layer_name))
            handles.append(handle)
    
    # Test each layer
    print("Testing each layer (zeroing its output):\n")
    
    for group_name, layer_name, layer_module in all_layers:
        # Enable ablation for this layer
        ablation_active["enabled"] = True
        ablation_active["layer"] = layer_name
        
        # Run evaluation
        try:
            results = evaluate_test_examples(
                model, test_dataset, device, verbose=False, visualize=False
            )
            acc = results['pixel_accuracy']
            perfect = results['perfect_rate']
            
            # Calculate drop from baseline
            acc_drop = baseline_acc - acc
            perfect_drop = baseline_perfect - perfect
            
            # Store results
            ablation_results[layer_name] = {
                "group": group_name,
                "accuracy": acc,
                "perfect_rate": perfect,
                "accuracy_drop": acc_drop,
                "perfect_drop": perfect_drop,
                "relative_drop": acc_drop / baseline_acc if baseline_acc > 0 else 0,
            }
            
            # Determine impact level
            if acc_drop > 0.5:
                impact = "██████████ CRITICAL"
            elif acc_drop > 0.3:
                impact = "████████░░ HIGH"
            elif acc_drop > 0.1:
                impact = "█████░░░░░ MEDIUM"
            elif acc_drop > 0.02:
                impact = "██░░░░░░░░ LOW"
            else:
                impact = "░░░░░░░░░░ MINIMAL"
            
            print(f"  [{group_name}] {layer_name}")
            print(f"    Accuracy: {acc:.2%} (drop: {acc_drop:+.2%})  {impact}")
            if baseline_perfect > 0:
                print(f"    Perfect:  {perfect:.2%} (drop: {perfect_drop:+.2%})")
            print()
            
        except Exception as e:
            print(f"  [{group_name}] {layer_name}: ERROR - {e}")
            ablation_results[layer_name] = {"error": str(e)}
        
        # Disable ablation
        ablation_active["enabled"] = False
    
    # Remove all hooks
    for handle in handles:
        handle.remove()
    
    # Print summary
    print("="*70)
    print("ABLATION SUMMARY (sorted by impact)")
    print("="*70)
    
    # Sort by accuracy drop
    sorted_layers = sorted(
        [(name, res) for name, res in ablation_results.items() if "error" not in res],
        key=lambda x: x[1]["accuracy_drop"],
        reverse=True
    )
    
    print("\nMost Critical Layers (largest accuracy drop when ablated):")
    for i, (name, res) in enumerate(sorted_layers[:5], 1):
        print(f"  {i}. {name}: {res['accuracy_drop']:+.2%} drop ({res['accuracy']:.2%} remaining)")
    
    print("\nLeast Critical Layers (smallest accuracy drop when ablated):")
    for i, (name, res) in enumerate(sorted_layers[-3:], 1):
        print(f"  {i}. {name}: {res['accuracy_drop']:+.2%} drop ({res['accuracy']:.2%} remaining)")
    
    # Interpretation
    print("\n" + "─"*70)
    print("INTERPRETATION:")
    
    critical_layers = [name for name, res in sorted_layers if res["accuracy_drop"] > 0.3]
    if critical_layers:
        print(f"  • Critical layers: {', '.join(critical_layers)}")
        print(f"    These layers store essential transformation knowledge.")
    
    minimal_layers = [name for name, res in sorted_layers if res["accuracy_drop"] < 0.02]
    if minimal_layers:
        print(f"  • Minimal-impact layers: {', '.join(minimal_layers)}")
        print(f"    These layers may be redundant or store non-essential features.")
    
    # Check if attention matters
    attention_results = [res for name, res in ablation_results.items() 
                        if "attention" in name.lower() and "error" not in res]
    if attention_results:
        attn_drop = max(res["accuracy_drop"] for res in attention_results)
        if attn_drop > 0.1:
            print(f"  • Attention is important (drop: {attn_drop:+.2%})")
            print(f"    The transformation likely requires global/relational reasoning.")
        else:
            print(f"  • Attention has minimal impact (drop: {attn_drop:+.2%})")
            print(f"    The transformation may be primarily local/convolutional.")
    
    print("─"*70 + "\n")
    
    return {
        "baseline": {"accuracy": baseline_acc, "perfect_rate": baseline_perfect},
        "layers": ablation_results,
    }


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    size_weight: float = 0.0,  # Weight for size prediction loss (0 = pixel only)
) -> Dict[str, float]:
    """Training epoch for color prediction (and optionally size prediction)"""
    model.train()

    total_loss = 0.0
    total_pixel_loss = 0.0
    total_size_loss = 0.0
    total_color_correct = 0
    total_pixels = 0
    total_error_pixels = 0
    total_error_color_correct = 0
    total_height_correct = 0
    total_width_correct = 0
    total_both_correct = 0
    total_samples = 0
    num_batches = 0

    pbar = tqdm(loader, desc="Training")
    for input_grid, output_grid, target_colors, pixel_mask, output_dims in pbar:
        input_grid = input_grid.to(device)
        output_grid = output_grid.to(device)
        target_colors = target_colors.to(device)
        pixel_mask = pixel_mask.to(device)
        output_dims = output_dims.to(device)

        optimizer.zero_grad()

        output = model(input_grid, output_grid)

        # Handle both dict (with size) and tensor (pixel only) outputs
        if isinstance(output, dict):
            pixel_logits = output['pixel_logits']
            height_logits = output['height_logits']
            width_logits = output['width_logits']

            # Pixel loss
            pixel_loss = F.cross_entropy(pixel_logits, target_colors)

            # Size loss (convert sizes 1-30 to classes 0-29)
            target_h = output_dims[:, 0] - 1
            target_w = output_dims[:, 1] - 1
            size_loss = (F.cross_entropy(height_logits, target_h) +
                         F.cross_entropy(width_logits, target_w)) / 2

            # Combined loss
            loss = pixel_loss + size_weight * size_loss
        else:
            pixel_logits = output
            pixel_loss = F.cross_entropy(pixel_logits, target_colors)
            loss = pixel_loss
            size_loss = torch.tensor(0.0)

        loss.backward()
        optimizer.step()

        with torch.no_grad():
            pred_colors = pixel_logits.argmax(dim=1)  # (B, H, W)

            # Overall color accuracy
            correct = (pred_colors == target_colors).sum().item()
            total = target_colors.numel()
            total_color_correct += correct
            total_pixels += total

            # Accuracy on error pixels only (pixels where output != target)
            error_mask = (pixel_mask == 0)  # pixels that are wrong
            if error_mask.sum() > 0:
                error_correct = ((pred_colors == target_colors) & error_mask).sum().item()
                total_error_color_correct += error_correct
                total_error_pixels += error_mask.sum().item()

            # Size accuracy (if predicting size)
            if isinstance(output, dict):
                pred_h = height_logits.argmax(dim=1)
                pred_w = width_logits.argmax(dim=1)
                target_h = output_dims[:, 0] - 1
                target_w = output_dims[:, 1] - 1

                total_height_correct += (pred_h == target_h).sum().item()
                total_width_correct += (pred_w == target_w).sum().item()
                total_both_correct += ((pred_h == target_h) & (pred_w == target_w)).sum().item()
                total_samples += input_grid.size(0)
                total_size_loss += size_loss.item()

            total_loss += loss.item()
            total_pixel_loss += pixel_loss.item()
            num_batches += 1

        pbar.set_postfix(loss=loss.item())

    color_accuracy = total_color_correct / max(total_pixels, 1)
    error_color_accuracy = total_error_color_correct / max(total_error_pixels, 1)

    metrics = {
        "loss": total_loss / max(num_batches, 1),
        "pixel_loss": total_pixel_loss / max(num_batches, 1),
        "color_accuracy": color_accuracy,
        "error_color_accuracy": error_color_accuracy,
    }

    # Add size metrics if applicable
    if total_samples > 0:
        metrics["size_loss"] = total_size_loss / max(num_batches, 1)
        metrics["height_accuracy"] = total_height_correct / total_samples
        metrics["width_accuracy"] = total_width_correct / total_samples
        metrics["both_accuracy"] = total_both_correct / total_samples

    return metrics


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    size_weight: float = 0.0,
) -> Dict[str, float]:
    """Evaluation for color prediction (and optionally size prediction)"""
    model.eval()

    total_loss = 0.0
    total_pixel_loss = 0.0
    total_size_loss = 0.0
    total_color_correct = 0
    total_pixels = 0
    total_error_pixels = 0
    total_error_color_correct = 0
    total_perfect = 0
    total_height_correct = 0
    total_width_correct = 0
    total_both_correct = 0
    total_samples = 0
    num_batches = 0

    with torch.no_grad():
        for input_grid, output_grid, target_colors, pixel_mask, output_dims in loader:
            input_grid = input_grid.to(device)
            output_grid = output_grid.to(device)
            target_colors = target_colors.to(device)
            pixel_mask = pixel_mask.to(device)
            output_dims = output_dims.to(device)

            output = model(input_grid, output_grid)

            # Handle both dict (with size) and tensor (pixel only) outputs
            if isinstance(output, dict):
                pixel_logits = output['pixel_logits']
                height_logits = output['height_logits']
                width_logits = output['width_logits']

                pixel_loss = F.cross_entropy(pixel_logits, target_colors)

                target_h = output_dims[:, 0] - 1
                target_w = output_dims[:, 1] - 1
                size_loss = (F.cross_entropy(height_logits, target_h) +
                             F.cross_entropy(width_logits, target_w)) / 2

                loss = pixel_loss + size_weight * size_loss

                # Size accuracy
                pred_h = height_logits.argmax(dim=1)
                pred_w = width_logits.argmax(dim=1)
                total_height_correct += (pred_h == target_h).sum().item()
                total_width_correct += (pred_w == target_w).sum().item()
                total_both_correct += ((pred_h == target_h) & (pred_w == target_w)).sum().item()
                total_size_loss += size_loss.item()
            else:
                pixel_logits = output
                pixel_loss = F.cross_entropy(pixel_logits, target_colors)
                loss = pixel_loss

            pred_colors = pixel_logits.argmax(dim=1)  # (B, H, W)

            # Overall color accuracy
            correct = (pred_colors == target_colors).sum().item()
            total = target_colors.numel()
            total_color_correct += correct
            total_pixels += total

            # Accuracy on error pixels only
            error_mask = (pixel_mask == 0)
            if error_mask.sum() > 0:
                error_correct = ((pred_colors == target_colors) & error_mask).sum().item()
                total_error_color_correct += error_correct
                total_error_pixels += error_mask.sum().item()

            # Perfect predictions (all pixels correct)
            batch_perfect = (pred_colors == target_colors).all(dim=-1).all(dim=-1).sum().item()
            total_perfect += batch_perfect
            total_samples += input_grid.size(0)

            total_loss += loss.item()
            total_pixel_loss += pixel_loss.item()
            num_batches += 1

    color_accuracy = total_color_correct / max(total_pixels, 1)
    error_color_accuracy = total_error_color_correct / max(total_error_pixels, 1)
    perfect_rate = total_perfect / max(total_samples, 1)

    metrics = {
        "loss": total_loss / max(num_batches, 1),
        "pixel_loss": total_pixel_loss / max(num_batches, 1),
        "color_accuracy": color_accuracy,
        "error_color_accuracy": error_color_accuracy,
        "perfect_rate": perfect_rate,
    }

    # Add size metrics if applicable
    if total_samples > 0 and total_size_loss > 0:
        metrics["size_loss"] = total_size_loss / max(num_batches, 1)
        metrics["height_accuracy"] = total_height_correct / total_samples
        metrics["width_accuracy"] = total_width_correct / total_samples
        metrics["both_accuracy"] = total_both_correct / total_samples

    return metrics


# =============================================================================
# Size Prediction Training
# =============================================================================

def train_size_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Dict[str, float]:
    """Training epoch for output size prediction."""
    model.train()

    total_loss = 0.0
    total_height_correct = 0
    total_width_correct = 0
    total_both_correct = 0
    total_samples = 0
    num_batches = 0

    pbar = tqdm(loader, desc="Training Size")
    for input_grid, _, _, _, output_dims in pbar:
        input_grid = input_grid.to(device)
        target_h = output_dims[:, 0].to(device) - 1  # Convert size (1-30) to class (0-29)
        target_w = output_dims[:, 1].to(device) - 1

        optimizer.zero_grad()

        height_logits, width_logits = model(input_grid)

        # Cross-entropy loss for both dimensions
        loss_h = F.cross_entropy(height_logits, target_h)
        loss_w = F.cross_entropy(width_logits, target_w)
        loss = (loss_h + loss_w) / 2

        loss.backward()
        optimizer.step()

        with torch.no_grad():
            pred_h = height_logits.argmax(dim=1)
            pred_w = width_logits.argmax(dim=1)

            h_correct = (pred_h == target_h).sum().item()
            w_correct = (pred_w == target_w).sum().item()
            both_correct = ((pred_h == target_h) & (pred_w == target_w)).sum().item()

            total_height_correct += h_correct
            total_width_correct += w_correct
            total_both_correct += both_correct
            total_samples += input_grid.size(0)

            total_loss += loss.item()
            num_batches += 1

        pbar.set_postfix(loss=loss.item(), h_acc=h_correct/input_grid.size(0), w_acc=w_correct/input_grid.size(0))

    return {
        "loss": total_loss / max(num_batches, 1),
        "height_accuracy": total_height_correct / max(total_samples, 1),
        "width_accuracy": total_width_correct / max(total_samples, 1),
        "both_accuracy": total_both_correct / max(total_samples, 1),
    }


def evaluate_size(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    """Evaluation for output size prediction."""
    model.eval()

    total_loss = 0.0
    total_height_correct = 0
    total_width_correct = 0
    total_both_correct = 0
    total_samples = 0
    num_batches = 0

    with torch.no_grad():
        for input_grid, _, _, _, output_dims in loader:
            input_grid = input_grid.to(device)
            target_h = output_dims[:, 0].to(device) - 1
            target_w = output_dims[:, 1].to(device) - 1

            height_logits, width_logits = model(input_grid)

            loss_h = F.cross_entropy(height_logits, target_h)
            loss_w = F.cross_entropy(width_logits, target_w)
            loss = (loss_h + loss_w) / 2

            pred_h = height_logits.argmax(dim=1)
            pred_w = width_logits.argmax(dim=1)

            h_correct = (pred_h == target_h).sum().item()
            w_correct = (pred_w == target_w).sum().item()
            both_correct = ((pred_h == target_h) & (pred_w == target_w)).sum().item()

            total_height_correct += h_correct
            total_width_correct += w_correct
            total_both_correct += both_correct
            total_samples += input_grid.size(0)

            total_loss += loss.item()
            num_batches += 1

    return {
        "loss": total_loss / max(num_batches, 1),
        "height_accuracy": total_height_correct / max(total_samples, 1),
        "width_accuracy": total_width_correct / max(total_samples, 1),
        "both_accuracy": total_both_correct / max(total_samples, 1),
    }


def evaluate_size_on_test(
    model: nn.Module,
    dataset,  # TestEvalDataset
    device: torch.device,
    verbose: bool = True,
) -> Dict[str, float]:
    """Evaluate size prediction on held-out test examples."""
    model.eval()

    total_height_correct = 0
    total_width_correct = 0
    total_both_correct = 0
    total_examples = 0

    results = []

    with torch.no_grad():
        for i in range(len(dataset)):
            input_grid, _, dims, puzzle_id = dataset[i]
            out_h, out_w = dims[2].item(), dims[3].item()

            inp_t = input_grid.unsqueeze(0).to(device)

            pred_h, pred_w = model.predict_size(inp_t)
            pred_h, pred_w = pred_h.item(), pred_w.item()

            h_correct = (pred_h == out_h)
            w_correct = (pred_w == out_w)
            both_correct = h_correct and w_correct

            total_height_correct += int(h_correct)
            total_width_correct += int(w_correct)
            total_both_correct += int(both_correct)
            total_examples += 1

            results.append({
                'puzzle_id': puzzle_id,
                'true_h': out_h, 'true_w': out_w,
                'pred_h': pred_h, 'pred_w': pred_w,
                'h_correct': h_correct, 'w_correct': w_correct,
                'both_correct': both_correct,
            })

            if verbose:
                status = "✓" if both_correct else "✗"
                print(f"  {status} {puzzle_id}: true={out_h}×{out_w}, pred={pred_h}×{pred_w}")

    return {
        "height_accuracy": total_height_correct / max(total_examples, 1),
        "width_accuracy": total_width_correct / max(total_examples, 1),
        "both_accuracy": total_both_correct / max(total_examples, 1),
        "total_examples": total_examples,
        "results": results,
    }


def _find_grid_bounds(grid: np.ndarray, max_size: int = 30) -> Tuple[int, int]:
    """Find the bounds of non-zero content in a grid."""
    nonzero = np.where(grid > 0)
    if len(nonzero[0]) > 0:
        r_max = min(nonzero[0].max() + 1, max_size)
        c_max = min(nonzero[1].max() + 1, max_size)
    else:
        # If all zeros, check for any non-padding content by looking at the mask pattern
        # Default to a small size
        r_max, c_max = 1, 1
    return r_max, c_max


def visualize_predictions(model: nn.Module, dataset: Dataset, device: torch.device, num_samples: int = 8):
    """Visualize predictions for color prediction"""
    model.eval()

    print("\n" + "="*80)
    print("Sample Predictions (Color Mode - Color Prediction)")
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

        print(f"\n{'â”€'*80}")
        print(f"Sample Type: {sample_type.upper()}")
        print(f"{'â”€'*80}")

        samples_to_show = min(2, count)
        for i in range(samples_to_show):
            # Calculate index for this sample type
            example_idx = random.randint(0, len(dataset.examples) - 1)
            sample_idx = example_idx * dataset.samples_per_example + offset + i

            if sample_idx >= len(dataset):
                continue

            input_grid, output_grid, target_colors, pixel_mask = dataset[sample_idx]

            inp_t = input_grid.unsqueeze(0).to(device)
            out_t = output_grid.unsqueeze(0).to(device)

            with torch.no_grad():
                pred_colors = model.predict_colors(inp_t, out_t)[0].cpu().numpy()

            input_np = input_grid.numpy()
            output_np = output_grid.numpy()
            target_np = target_colors.numpy()
            mask_np = pixel_mask.numpy()

            # Find content bounds SEPARATELY for input and output
            inp_r, inp_c = _find_grid_bounds(input_np)
            out_r, out_c = _find_grid_bounds(output_np)

            # Also check target bounds (may differ from output for corrupted samples)
            tgt_r, tgt_c = _find_grid_bounds(target_np)
            out_r = max(out_r, tgt_r)
            out_c = max(out_c, tgt_c)

            # For all-zeros output, use mask to find content region
            if out_r == 1 and out_c == 1:
                error_positions = np.where(mask_np == 0)
                if len(error_positions[0]) > 0:
                    out_r = min(error_positions[0].max() + 1, 30)
                    out_c = min(error_positions[1].max() + 1, 30)

            num_errors = int((mask_np[:out_r, :out_c] == 0).sum())
            num_correct_preds = int((pred_colors[:out_r, :out_c] == target_np[:out_r, :out_c]).sum())
            total_pixels = out_r * out_c

            print(f"\nInput: {inp_r}Ã—{inp_c} â†’ Output: {out_r}Ã—{out_c}")
            print(f"Actual errors: {num_errors}, Color prediction accuracy: {num_correct_preds}/{total_pixels}")

            # Print INPUT separately with its own dimensions
            print(f"\nINPUT ({inp_r}Ã—{inp_c}):")
            for r in range(inp_r):
                inp_row = " ".join(f"{input_np[r, c]}" for c in range(inp_c))
                print(f"  {inp_row}")

            # Print OUTPUT, TARGET, PREDICTED with output dimensions
            print(f"\n{'OUTPUT':<25} {'TARGET':<25} {'PREDICTED':<25}")
            for r in range(out_r):
                out_row = " ".join(f"{output_np[r, c]}" for c in range(out_c))
                tgt_row = " ".join(f"{target_np[r, c]}" for c in range(out_c))
                pred_row = " ".join(f"{pred_colors[r, c]}" for c in range(out_c))
                print(f"  {out_row:<23} {tgt_row:<23} {pred_row:<23}")

    print("="*80 + "\n")


def run_ablation_analysis(model, test_loader, verbose: bool = True):
    """
    Run layer-wise ablation using forward hooks (consistent with run_layer_ablation).

    For each layer, we temporarily zero out its output and measure the impact
    on test accuracy. Larger drops indicate more critical layers.

    Args:
        model: The model to analyze
        test_loader: DataLoader for test data
        verbose: Whether to print detailed output

    Returns:
        Dict with 'baseline' and 'layers' keys containing ablation results
    """
    model.eval()

    def evaluate_loader(loader, device):
        """Evaluate model on a DataLoader, returning pixel and grid accuracy."""
        pixel_correct = 0
        pixel_total = 0
        grid_correct = 0
        grid_total = 0

        with torch.no_grad():
            for batch in loader:
                input_grid, output_grid, target, _ = batch
                input_grid = input_grid.to(device)
                output_grid = output_grid.to(device)
                target = target.to(device)

                logits = model(input_grid, output_grid)
                preds = logits.argmax(dim=1)

                pixel_correct += (preds == target).sum().item()
                pixel_total += target.numel()

                # Grid-level accuracy (entire grid must be correct)
                grid_match = (preds == target).all(dim=(1, 2))
                grid_correct += grid_match.sum().item()
                grid_total += grid_match.size(0)

        pixel_acc = pixel_correct / pixel_total if pixel_total > 0 else 0
        grid_acc = grid_correct / grid_total if grid_total > 0 else 0
        return pixel_acc, grid_acc

    # Get baseline accuracy
    if verbose:
        print("\n" + "="*70)
        print("LAYER-WISE ABLATION ANALYSIS")
        print("="*70)
        print("Testing which layers are critical for this task...")
        print("Baseline = normal model, then we zero each layer's output one at a time.\n")

    baseline_acc, baseline_grid_acc = evaluate_loader(test_loader, DEVICE)

    if verbose:
        print(f"BASELINE (no ablation):")
        print(f"  Pixel Accuracy: {baseline_acc:.2%}")
        print(f"  Grid Accuracy:  {baseline_grid_acc:.2%}")
        print(f"{'─'*70}\n")

    # Define layer groups based on architecture (similar to run_layer_ablation)
    layer_groups = {
        "Embeddings": [
            ("input_embed", "input_embed"),
            ("output_embed", "output_embed"),
        ],
        "Encoder": [
            ("inc (initial conv)", "inc"),
            ("down1 (encoder level 1)", "down1"),
            ("down2 (encoder level 2)", "down2"),
            ("down3 (encoder level 3 / bottleneck)", "down3"),
        ],
        "Decoder": [
            ("up1 (decoder level 1)", "up1"),
            ("up2 (decoder level 2)", "up2"),
            ("up3 (decoder level 3)", "up3"),
        ],
        "Output": [
            ("outc (output conv)", "outc"),
        ],
    }

    # Add attention if present
    if hasattr(model, 'bottleneck_attn') and getattr(model, 'use_attention', False):
        layer_groups["Attention"] = [
            ("bottleneck_attn", "bottleneck_attn"),
        ]
        if hasattr(model, 'film') and not getattr(model, 'no_skip', True):
            layer_groups["FiLM"] = [
                ("film (global context)", "film"),
            ]

    # Collect all layers to test
    all_layers = []
    for group_name, layers in layer_groups.items():
        for display_name, attr_name in layers:
            if hasattr(model, attr_name):
                all_layers.append((group_name, display_name, attr_name, getattr(model, attr_name)))

    # Hook storage
    handles = []
    ablation_active = {"enabled": False, "layer": None}

    def make_zero_hook(layer_attr_name):
        """Create a hook that zeros out the layer's output when ablation is active."""
        def hook(module, input, output):
            if ablation_active["enabled"] and ablation_active["layer"] == layer_attr_name:
                if isinstance(output, torch.Tensor):
                    return torch.zeros_like(output)
                elif isinstance(output, tuple):
                    return tuple(torch.zeros_like(o) if isinstance(o, torch.Tensor) else o for o in output)
            return output
        return hook

    # Register hooks on all layers
    for group_name, display_name, attr_name, layer_module in all_layers:
        handle = layer_module.register_forward_hook(make_zero_hook(attr_name))
        handles.append(handle)

    # Store ablation results
    ablation_results = {}

    if verbose:
        print("Testing each layer (zeroing its output):\n")

    # Test each layer
    for group_name, display_name, attr_name, layer_module in all_layers:
        # Enable ablation for this layer
        ablation_active["enabled"] = True
        ablation_active["layer"] = attr_name

        try:
            acc, grid_acc = evaluate_loader(test_loader, DEVICE)

            # Calculate drops from baseline
            acc_drop = baseline_acc - acc
            grid_drop = baseline_grid_acc - grid_acc

            # Calculate max impact (prioritize grid accuracy - a layer that wipes out
            # grid accuracy is critical even if pixel accuracy is only moderately affected)
            max_drop = max(acc_drop, grid_drop)

            # Store results
            ablation_results[display_name] = {
                "group": group_name,
                "attr_name": attr_name,
                "accuracy": acc,
                "grid_accuracy": grid_acc,
                "accuracy_drop": acc_drop,
                "grid_drop": grid_drop,
                "max_drop": max_drop,
                "relative_drop": acc_drop / baseline_acc if baseline_acc > 0 else 0,
                "relative_grid_drop": grid_drop / baseline_grid_acc if baseline_grid_acc > 0 else 0,
            }

            if verbose:
                # Determine impact levels separately for pixel and grid
                def get_impact_bar(drop):
                    if drop > 0.5:
                        return "██████████ CRITICAL"
                    elif drop > 0.3:
                        return "████████░░ HIGH"
                    elif drop > 0.1:
                        return "█████░░░░░ MEDIUM"
                    elif drop > 0.02:
                        return "██░░░░░░░░ LOW"
                    else:
                        return "░░░░░░░░░░ MINIMAL"

                print(f"  [{group_name}] {display_name}")
                if baseline_grid_acc > 0:
                    print(f"    Grid Acc:  {grid_acc:.2%} (drop: {grid_drop:+.2%})  {get_impact_bar(grid_drop)}")
                print(f"    Pixel Acc: {acc:.2%} (drop: {acc_drop:+.2%})  {get_impact_bar(acc_drop)}")
                print()

        except Exception as e:
            if verbose:
                print(f"  [{group_name}] {display_name}: ERROR - {e}")
            ablation_results[display_name] = {"error": str(e), "group": group_name}

        # Disable ablation
        ablation_active["enabled"] = False

    # Remove all hooks
    for handle in handles:
        handle.remove()

    # Print summary if verbose
    if verbose:
        print("="*70)
        print("ABLATION SUMMARY")
        print("="*70)

        # Sort by grid_drop for grid-critical analysis
        sorted_by_grid = sorted(
            [(name, res) for name, res in ablation_results.items() if "error" not in res],
            key=lambda x: x[1]["grid_drop"],
            reverse=True
        )

        # Sort by pixel_drop for pixel-critical analysis
        sorted_by_pixel = sorted(
            [(name, res) for name, res in ablation_results.items() if "error" not in res],
            key=lambda x: x[1]["accuracy_drop"],
            reverse=True
        )

        print("\nGrid-Critical Layers (sorted by grid accuracy drop):")
        for i, (name, res) in enumerate(sorted_by_grid[:5], 1):
            print(f"  {i}. {name}: {res['grid_drop']:+.2%} grid drop")

        print("\nPixel-Critical Layers (sorted by pixel accuracy drop):")
        for i, (name, res) in enumerate(sorted_by_pixel[:5], 1):
            print(f"  {i}. {name}: {res['accuracy_drop']:+.2%} pixel drop")

        # Interpretation
        print("\n" + "─"*70)
        print("INTERPRETATION:")

        # Critical thresholds
        grid_critical = [name for name, res in sorted_by_grid if res["grid_drop"] > 0.3]
        pixel_critical = [name for name, res in sorted_by_pixel if res["accuracy_drop"] > 0.3]

        if grid_critical:
            print(f"  • Grid-critical (>30% drop): {', '.join(grid_critical)}")

        if pixel_critical:
            print(f"  • Pixel-critical (>30% drop): {', '.join(pixel_critical)}")

        # Minimal impact (both metrics)
        minimal_layers = [name for name, res in sorted_by_grid
                        if res["grid_drop"] < 0.02 and res["accuracy_drop"] < 0.02]
        if minimal_layers:
            print(f"  • Minimal-impact layers: {', '.join(minimal_layers)}")

        # Check encoder vs decoder balance
        encoder_grid_drops = [res["grid_drop"] for _, res in ablation_results.items()
                             if res.get("group") == "Encoder" and "error" not in res]
        decoder_grid_drops = [res["grid_drop"] for _, res in ablation_results.items()
                             if res.get("group") == "Decoder" and "error" not in res]

        if encoder_grid_drops and decoder_grid_drops:
            avg_encoder = sum(encoder_grid_drops) / len(encoder_grid_drops)
            avg_decoder = sum(decoder_grid_drops) / len(decoder_grid_drops)
            if avg_encoder > avg_decoder * 1.5:
                print(f"  • Encoder-heavy (grid): avg {avg_encoder:.1%} vs decoder {avg_decoder:.1%}")
            elif avg_decoder > avg_encoder * 1.5:
                print(f"  • Decoder-heavy (grid): avg {avg_decoder:.1%} vs encoder {avg_encoder:.1%}")

        print("─"*70 + "\n")

    return {
        "baseline": {"accuracy": baseline_acc, "grid_accuracy": baseline_grid_acc},
        "layers": ablation_results,
    }


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
    parser.add_argument("--no-augment", action="store_true",
                        help="Train without augmentations (use raw example grids)")
    parser.add_argument("--dihedral-only", action="store_true",
                        help="Only use dihedral transforms (rotations/flips), no color permutations")
    parser.add_argument("--color-only", action="store_true",
                        help="Only use color permutations, no dihedral transforms (rotations/flips)")
    
    # Test-time evaluation (for single-puzzle mode)
    parser.add_argument("--eval-on-test", action="store_true",
                        help="Train only on puzzle's train examples, evaluate on held-out test examples. "
                             "This tests true generalization - did CNN learn the rule or just memorize?")
    
    # Negative type selection (for ablation studies)
    parser.add_argument("--negative-type", type=str, default="mixed",
                        choices=["mixed", "all_zeros", "corrupted", "random_noise", 
                                 "constant_fill", "color_swap", "wrong_input", "mismatched_aug"],
                        help="Type of negatives to use: "
                             "mixed=all types (default), or single type for ablation")
    
    # Visualization
    parser.add_argument("--visualize", action="store_true",
                        help="Show visual comparison of predictions vs targets during eval-on-test")
    
    # Recursive CNN evaluation
    parser.add_argument("--recursive-iters", type=int, default=1,
                        help="Number of recursive iterations for evaluation (1 = no recursion)")

    parser.add_argument("--no-skip", action="store_true",
                        help="Remove skip connections - forces all info through bottleneck")
    parser.add_argument("--num-layers", type=int, default=3, choices=[0, 1, 2, 3],
                        help="Number of encoder/decoder layers (0-3). 0 = just inc+outc, no down/up. Default: 3")
    parser.add_argument("--conv-depth", type=int, default=2,
                        help="Number of conv layers per block (1=single, 2=double, 3=triple, etc.). "
                             "RF per block = depth*(kernel_size-1)+1. Default: 2 (5x5 RF with 3x3 kernels)")
    parser.add_argument("--kernel-size", type=int, default=3,
                        help="Kernel size for convolutional layers (default: 3). Use odd numbers (3, 5, 7, etc.)")
    parser.add_argument("--out-kernel-size", type=int, default=1,
                        help="Kernel size for output conv layer (default: 1). Use odd numbers (1, 3, 5, etc.)")
    parser.add_argument("--use-onehot", action="store_true",
                        help="Use one-hot + nonzero mask encoding (11 ch/grid) instead of learned embeddings (16 ch/grid)")
    parser.add_argument("--use-attention", action="store_true",
                        help="Add spatial self-attention before output layer (each pixel attends to all others)")
    parser.add_argument("--use-cross-attention", action="store_true",
                        help="Add cross-attention on embeddings: output pixels attend to input pixels before convolutions")
    parser.add_argument("--ir-checkpoint", type=str, default=None,
                        help="Path to instance recognition CNN checkpoint for cross-attention keys/values (requires --use-cross-attention)")
    parser.add_argument("--use-untrained-ir", action="store_true",
                        help="Use randomly initialized IR encoder instead of pretrained (for ablation study)")
    parser.add_argument("--ir-hidden-dim", type=int, default=128,
                        help="Hidden dimension for untrained IR encoder (default: 128)")
    parser.add_argument("--ir-out-dim", type=int, default=64,
                        help="Output feature dimension for untrained IR encoder (default: 64)")
    parser.add_argument("--ir-num-layers", type=int, default=3,
                        help="Number of conv layers in untrained IR encoder (default: 3)")
    parser.add_argument("--ir-kernel-size", type=int, default=3,
                        help="Kernel size for untrained IR encoder (default: 3)")
    parser.add_argument("--freeze-ir", action="store_true", default=True,
                        help="Freeze IR encoder weights during training (default: True)")
    parser.add_argument("--no-freeze-ir", dest="freeze_ir", action="store_false",
                        help="Allow IR encoder weights to be fine-tuned")

    # Joint size prediction (integrated into PixelErrorCNN)
    parser.add_argument("--predict-size", action="store_true",
                        help="Enable joint output size prediction alongside pixel prediction. "
                             "Uses IR encoder to predict output height/width (1-30 each).")
    parser.add_argument("--size-weight", type=float, default=1.0,
                        help="Weight for size prediction loss relative to pixel loss (default: 1.0)")

    # Layer-wise ablation analysis
    parser.add_argument("--ablation", action="store_true",
                        help="Run layer-wise ablation analysis after training. "
                             "Zeros each layer's output and measures accuracy drop. "
                             "Requires --eval-on-test to have test data to evaluate against.")

    # Size prediction mode
    parser.add_argument("--train-size", action="store_true",
                        help="Train output size prediction head instead of pixel prediction. "
                             "Requires --ir-checkpoint to provide the IR encoder.")
    parser.add_argument("--size-hidden-dim", type=int, default=128,
                        help="Hidden dimension for size prediction MLP (default: 128)")
    parser.add_argument("--size-save-path", type=str, default="checkpoints/size_predictor.pt",
                        help="Path to save size prediction model checkpoint")

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    print(f"Device: {DEVICE}")
    print(f"Dataset: {args.dataset}")
    print(f"Mode: COLOR (predict correct colors 0-9)")

    # Determine augmentation mode
    if args.no_augment:
        aug_mode = "none"
    elif args.dihedral_only and args.color_only:
        print("Error: Cannot specify both --dihedral-only and --color-only")
        return
    elif args.dihedral_only:
        aug_mode = "dihedral-only (rotations/flips, no color permutations)"
    elif args.color_only:
        aug_mode = "color-only (color permutations, no rotations/flips)"
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
        
        # Check if test examples have outputs
        test_examples = puzzles[args.single_puzzle].get("test", [])
        test_has_outputs = any("output" in ex for ex in test_examples)
        if args.eval_on_test and not test_has_outputs:
            print(f"Warning: --eval-on-test specified but puzzle has no test outputs!")
            print(f"         Will train on all available examples instead.")
            args.eval_on_test = False
    
    if args.eval_on_test and not args.single_puzzle:
        print("Error: --eval-on-test requires --single-puzzle")
        return
    
    # Check ablation requirements
    if args.ablation and not args.eval_on_test:
        print("Warning: --ablation requires --eval-on-test to have test data to evaluate against.")
        print("         Ablation analysis will be skipped. Add --eval-on-test to enable it.")
        args.ablation = False
    
    # For --eval-on-test mode, explain what we're doing
    if args.eval_on_test:
        puzzle = puzzles[args.single_puzzle]
        n_train = len(puzzle.get("train", []))
        n_test = sum(1 for ex in puzzle.get("test", []) if "output" in ex)
        print(f"\n*** GENERALIZATION TEST MODE ***")
        print(f"Training on {n_train} train examples (with augmentation)")
        print(f"Evaluating on {n_test} held-out test examples (no augmentation)")
        print(f"This tests if CNN learned the RULE, not just memorized examples.\n")

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

    # Handle --negative-type for ablation studies
    if args.negative_type == "mixed":
        # Default: distribute across all types
        num_corrupted = max(1, int(num_negatives * 0.35))
        num_wrong_input = max(1, int(num_negatives * 0.15))
        num_mismatched_aug = max(1, int(num_negatives * 0.15)) if use_augment else 0
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
        
        if args.negative_type == "all_zeros":
            num_all_zeros = num_negatives
        elif args.negative_type == "corrupted":
            num_corrupted = num_negatives
        elif args.negative_type == "random_noise":
            num_random_noise = num_negatives
        elif args.negative_type == "constant_fill":
            num_constant_fill = num_negatives
        elif args.negative_type == "color_swap":
            num_color_swap = num_negatives
        elif args.negative_type == "wrong_input":
            num_wrong_input = num_negatives
        elif args.negative_type == "mismatched_aug":
            if not use_augment:
                print("Warning: --negative-type mismatched_aug requires augmentation. Using corrupted instead.")
                num_corrupted = num_negatives
            elif args.color_only:
                print("Note: --negative-type mismatched_aug with --color-only will mismatch color permutations only.")
                num_mismatched_aug = num_negatives
            else:
                num_mismatched_aug = num_negatives

    print(f"Negative type: {args.negative_type}")
    print(f"Distribution: {num_corrupted} corrupted, {num_wrong_input} wrong_input, {num_mismatched_aug} mismatched_aug, "
          f"{num_color_swap} color_swap, {num_all_zeros} all_zeros, {num_constant_fill} const_fill, {num_random_noise} noise")
    print(f"Plus {num_positives} positives per example")

    # Determine whether to include test examples in training
    # For --eval-on-test mode, we train ONLY on train examples
    include_test_in_train = not args.eval_on_test
    
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
        color_only=args.color_only,
        include_test=include_test_in_train,
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
        color_only=args.color_only,
        include_test=include_test_in_train,
    )

    # For --eval-on-test, create separate test evaluation dataset
    test_eval_dataset = None
    if args.eval_on_test:
        test_eval_dataset = TestEvalDataset(train_puzzles, test_only=True)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # =========================================================================
    # SIZE PREDICTION MODE
    # =========================================================================
    if args.train_size:
        if not args.ir_checkpoint and not args.use_untrained_ir:
            print("Error: --train-size requires --ir-checkpoint or --use-untrained-ir")
            return

        print("\n" + "="*60)
        print("SIZE PREDICTION MODE")
        print("="*60)

        # Create IR encoder
        if args.ir_checkpoint:
            print(f"Loading IR encoder from: {args.ir_checkpoint}")
            ir_encoder = IREncoder.from_checkpoint(args.ir_checkpoint, DEVICE)
        else:
            print("Using untrained IR encoder")
            ir_encoder = IREncoder(
                hidden_dim=args.ir_hidden_dim,
                out_dim=args.ir_out_dim,
                num_layers=args.ir_num_layers,
                kernel_size=args.ir_kernel_size,
            )

        # Create size prediction model
        size_model = SizeClassificationHead(
            ir_encoder=ir_encoder,
            max_size=GRID_SIZE,
            hidden_dim=args.size_hidden_dim,
            freeze_ir=args.freeze_ir,
        ).to(DEVICE)

        num_params = sum(p.numel() for p in size_model.parameters())
        trainable_params = sum(p.numel() for p in size_model.parameters() if p.requires_grad)
        print(f"Size model parameters: {num_params:,} total, {trainable_params:,} trainable")
        print(f"IR encoder frozen: {args.freeze_ir}")

        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, size_model.parameters()),
            lr=args.lr
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=10
        )

        best_both_acc = 0.0
        print(f"\nTraining for {args.epochs} epochs...")

        for epoch in range(args.epochs):
            train_metrics = train_size_epoch(size_model, train_loader, optimizer, DEVICE)
            val_metrics = evaluate_size(size_model, val_loader, DEVICE)

            scheduler.step(val_metrics['loss'])

            print(f"Epoch {epoch+1}/{args.epochs}")
            print(f"  Train - Loss: {train_metrics['loss']:.4f}, H: {train_metrics['height_accuracy']:.2%}, "
                  f"W: {train_metrics['width_accuracy']:.2%}, Both: {train_metrics['both_accuracy']:.2%}")
            print(f"  Val   - Loss: {val_metrics['loss']:.4f}, H: {val_metrics['height_accuracy']:.2%}, "
                  f"W: {val_metrics['width_accuracy']:.2%}, Both: {val_metrics['both_accuracy']:.2%}")

            # Save best model
            if val_metrics['both_accuracy'] > best_both_acc:
                best_both_acc = val_metrics['both_accuracy']
                os.makedirs(os.path.dirname(args.size_save_path) or '.', exist_ok=True)
                size_model.save_checkpoint(args.size_save_path, vars(args))
                print(f"  Saved new best model (both_accuracy: {best_both_acc:.2%})")

        print(f"\nBest validation both_accuracy: {best_both_acc:.2%}")

        # Evaluate on test set if available
        if args.eval_on_test and test_eval_dataset is not None:
            print("\n" + "="*60)
            print("SIZE PREDICTION ON TEST EXAMPLES")
            print("="*60)
            test_results = evaluate_size_on_test(size_model, test_eval_dataset, DEVICE, verbose=True)
            print(f"\nTest Results:")
            print(f"  Height accuracy: {test_results['height_accuracy']:.2%}")
            print(f"  Width accuracy:  {test_results['width_accuracy']:.2%}")
            print(f"  Both correct:    {test_results['both_accuracy']:.2%}")

        print("\nSize prediction training complete!")
        return  # Exit early - don't continue to pixel prediction training

    # =========================================================================
    # PIXEL PREDICTION MODE (default)
    # =========================================================================

    # Create model
    print("\nCreating model...")
    rf_per_block = args.conv_depth * (args.kernel_size - 1) + 1
    encoding_str = "one-hot+nonzero (11 ch/grid)" if args.use_onehot else "learned embeddings (16 ch/grid)"
    print(f"Architecture: U-Net ({args.num_layers} encoder/decoder layers, conv_depth={args.conv_depth} ({rf_per_block}x{rf_per_block} RF), {args.kernel_size}x{args.kernel_size} kernels)")
    print(f"Encoding: {encoding_str}")
    model = PixelErrorCNN(
        hidden_dim=args.hidden_dim,
        no_skip=args.no_skip,
        num_layers=args.num_layers,
        conv_depth=args.conv_depth,
        kernel_size=args.kernel_size,
        use_onehot=args.use_onehot,
        out_kernel_size=args.out_kernel_size,
        use_attention=args.use_attention,
        use_cross_attention=args.use_cross_attention,
        ir_checkpoint=args.ir_checkpoint,
        use_untrained_ir=args.use_untrained_ir,
        ir_hidden_dim=args.ir_hidden_dim,
        ir_out_dim=args.ir_out_dim,
        ir_num_layers=args.ir_num_layers,
        ir_kernel_size=args.ir_kernel_size,
        freeze_ir=args.freeze_ir,
        predict_size=args.predict_size,
        size_hidden_dim=args.size_hidden_dim,
    )
    model = model.to(DEVICE)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")
    print(f"Output classes: 10 (color prediction)")
    if args.no_skip:
        print("Skip connections: DISABLED (all info flows through bottleneck)")
    if args.use_attention:
        print("Spatial self-attention: ENABLED (each pixel attends to all others)")
    if args.use_cross_attention:
        if args.ir_checkpoint:
            freeze_str = "frozen" if args.freeze_ir else "fine-tuned"
            print(f"Cross-attention: ENABLED with pretrained IR encoder ({freeze_str}) from {args.ir_checkpoint}")
        elif args.use_untrained_ir:
            print(f"Cross-attention: ENABLED with UNTRAINED IR encoder (random init, for ablation)")
            print(f"  IR encoder config: hidden={args.ir_hidden_dim}, out={args.ir_out_dim}, "
                  f"layers={args.ir_num_layers}, kernel={args.ir_kernel_size}x{args.ir_kernel_size}")
        else:
            print("Cross-attention: ENABLED (output pixels attend to input pixels on embeddings)")
    if args.predict_size:
        print(f"Size prediction: ENABLED (joint training with weight={args.size_weight})")
        if not args.use_cross_attention and not args.ir_checkpoint and not args.use_untrained_ir:
            print("  Note: IR encoder will be created for size prediction")

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)

    # Create checkpoint directory
    os.makedirs(os.path.dirname(args.save_path) or ".", exist_ok=True)

    # Training
    print("\n" + "="*60)
    print("Starting Training (COLOR mode)")
    print("="*60)

    best_val_metric = 0.0

    size_weight = args.size_weight if args.predict_size else 0.0

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")

        train_metrics = train_epoch(model, train_loader, optimizer, DEVICE, size_weight=size_weight)
        val_metrics = evaluate(model, val_loader, DEVICE, size_weight=size_weight)

        # Print pixel metrics
        print(f"  Train Loss: {train_metrics['loss']:.4f}, Color Acc: {train_metrics['color_accuracy']:.2%}, "
              f"Error Color Acc: {train_metrics['error_color_accuracy']:.2%}")
        print(f"  Val   Loss: {val_metrics['loss']:.4f}, Color Acc: {val_metrics['color_accuracy']:.2%}, "
              f"Error Color Acc: {val_metrics['error_color_accuracy']:.2%}, "
              f"Perfect: {val_metrics['perfect_rate']:.2%}")

        # Print size metrics if applicable
        if args.predict_size and 'both_accuracy' in train_metrics and 'both_accuracy' in val_metrics:
            print(f"  Train Size - H: {train_metrics['height_accuracy']:.2%}, "
                  f"W: {train_metrics['width_accuracy']:.2%}, Both: {train_metrics['both_accuracy']:.2%}")
            print(f"  Val   Size - H: {val_metrics['height_accuracy']:.2%}, "
                  f"W: {val_metrics['width_accuracy']:.2%}, Both: {val_metrics['both_accuracy']:.2%}")

        current_metric = val_metrics['color_accuracy']

        scheduler.step()

        if current_metric > best_val_metric:
            best_val_metric = current_metric
            print(f"  [New best color_accuracy: {best_val_metric:.2%}]")

            checkpoint = {
                'model_state_dict': model.state_dict(),
                'args': vars(args),
                'epoch': epoch,
                'best_val_metric': best_val_metric,
            }
            torch.save(checkpoint, args.save_path)

    # Final summary
    print("\n" + "="*60)
    print("Training Complete")
    print("="*60)
    print(f"Best color_accuracy: {best_val_metric:.2%}")
    print(f"Checkpoint saved to: {args.save_path}")

    # Visualize on training data
    visualize_predictions(model, val_dataset, DEVICE)

    # =========================================================================
    # CRITICAL: Evaluate on held-out test examples
    # =========================================================================
    if args.eval_on_test and test_eval_dataset is not None:
        print("\n" + "="*60)
        print("GENERALIZATION TEST: Evaluating on HELD-OUT test examples")
        print("="*60)
        print("These examples were NOT seen during training (even with augmentation).")
        print("This tests if the CNN learned the actual transformation RULE.\n")

        # Load best model
        checkpoint = torch.load(args.save_path, map_location=DEVICE, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

        test_results = evaluate_test_examples(
            model, test_eval_dataset, DEVICE,
            verbose=True, visualize=args.visualize,
            recursive_iters=args.recursive_iters
        )

        print(f"\n{'-'*60}")
        print(f"TEST SET RESULTS:")
        print(f"  Pixel Accuracy: {test_results['pixel_accuracy']:.2%}")
        print(f"  Perfect Examples: {test_results['perfect_examples']}/{test_results['total_examples']} ({test_results['perfect_rate']:.2%})")
        print(f"{'-'*60}")

        if test_results['perfect_rate'] == 1.0:
            print("\nPERFECT GENERALIZATION! The CNN learned the transformation rule!")
            print("   This suggests the CNN can solve this puzzle from training examples alone.")
        elif test_results['pixel_accuracy'] > 0.95:
            print("\nStrong generalization - CNN learned most of the rule.")
            print("  Minor errors may be edge cases or noise.")
        elif test_results['pixel_accuracy'] > 0.8:
            print("\n~ Partial generalization - CNN learned some patterns but not the full rule.")
        else:
            print("\nPoor generalization - CNN may have memorized training examples.")
            print("  The transformation rule was not learned.")

        # Also show comparison with training examples
        print("\n" + "-"*60)
        print("Comparison - Evaluating on TRAINING examples (sanity check):")
        train_eval_dataset = TestEvalDataset(train_puzzles, test_only=False)
        train_results = evaluate_test_examples(
            model, train_eval_dataset, DEVICE,
            verbose=True, visualize=False,  # Don't visualize training examples
            recursive_iters=args.recursive_iters
        )
        # Summary comparison
        print("\n" + "="*60)
        print("GENERALIZATION SUMMARY")
        print("="*60)

        print(f"\nTRAINING SET RESULTS:")
        print(f"  Pixel Accuracy: {train_results['pixel_accuracy']:.2%}")
        print(f"  Perfect Examples: {train_results['perfect_examples']}/{train_results['total_examples']} ({train_results['perfect_rate']:.2%})")

        print(f"\nComparison:")
        print(f"  Training examples: {train_results['pixel_accuracy']:.2%} accuracy, {train_results['perfect_rate']:.2%} perfect")
        print(f"  Test examples:     {test_results['pixel_accuracy']:.2%} accuracy, {test_results['perfect_rate']:.2%} perfect")
        gap = train_results['pixel_accuracy'] - test_results['pixel_accuracy']

        if gap > 0.1:
            print(f"  Gap: {gap:.2%} - significant overfitting to training examples")
        elif gap > 0.02:
            print(f"  Gap: {gap:.2%} - mild overfitting")
        else:
            print(f"  Gap: {gap:.2%} - good generalization!")

        # =========================================================================
        # LAYER-WISE ABLATION ANALYSIS
        # =========================================================================
        if args.ablation:
            ablation_results = run_layer_ablation(
                model, test_eval_dataset, DEVICE
            )

    print("\nDone!")
    print(f"\nTo test generalization on a single puzzle:")
    print(f"  python train_pixel_error_cnn.py --single-puzzle PUZZLE_ID --eval-on-test")


if __name__ == "__main__":
    main()