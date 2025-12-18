#!/usr/bin/env python3
"""
Pixel-Level Error Detection CNN - CORRESPONDENCE VERSION

This version forces the CNN to learn input-output correspondence by:
1. Using augmented PAIRS as positives (same aug applied to both input and output)
2. Using mismatched inputs as negatives (correct output, wrong input)
3. Using mismatched augmentations as negatives

The key insight: the CNN must see the SAME output labeled both "correct" and 
"incorrect" depending on the input. This forces it to actually compare them.

Modes:
- binary: Predict whether each pixel is correct (1) or incorrect (0)
- color: Predict what color each pixel SHOULD be (0-9)

Usage:
    python train_pixel_error_cnn.py --dataset arc-agi-1
    python train_pixel_error_cnn.py --single-puzzle 00d62c1b
    python train_pixel_error_cnn.py --mode color  # Color prediction mode
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


class UpNoSkip(nn.Module):
    """Decoder block without skip connections - forces all info through bottleneck."""
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = DoubleConv(out_ch, out_ch)

    def forward(self, x, target_size=None):
        x = self.up(x)
        if target_size is not None:
            diff_h = target_size[0] - x.size(2)
            diff_w = target_size[1] - x.size(3)
            x = F.pad(x, [diff_w // 2, diff_w - diff_w // 2,
                         diff_h // 2, diff_h - diff_h // 2])
        return self.conv(x)


class BottleneckAttention(nn.Module):
    """
    Self-attention over spatial positions at the U-Net bottleneck.
    
    This allows global reasoning: each spatial position can query all others,
    enabling comparisons like "am I the smallest object?" or "am I symmetric?"
    that require global context rather than local convolutions.
    """
    
    def __init__(self, dim: int, num_heads: int = 8, num_layers: int = 1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.num_layers = num_layers
        
        # Stack multiple attention layers if requested
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                'norm': nn.LayerNorm(dim),
                'qkv': nn.Linear(dim, dim * 3),
                'proj': nn.Linear(dim, dim),
            })
            for _ in range(num_layers)
        ])
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        B, C, H, W = x.shape
        
        # Flatten spatial dims: (B, C, H*W) -> (B, H*W, C)
        x_flat = x.flatten(2).transpose(1, 2)
        
        for layer in self.layers:
            # Self-attention with residual
            residual = x_flat
            x_norm = layer['norm'](x_flat)
            
            # Compute Q, K, V
            qkv = layer['qkv'](x_norm).reshape(B, H*W, 3, self.num_heads, self.head_dim)
            qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, heads, N, head_dim)
            q, k, v = qkv[0], qkv[1], qkv[2]
            
            # Attention
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            
            # Apply attention to values
            out = (attn @ v).transpose(1, 2).reshape(B, H*W, C)
            out = layer['proj'](out)
            
            # Residual connection
            x_flat = residual + out
        
        # Reshape back to spatial
        return x_flat.transpose(1, 2).reshape(B, C, H, W)


class ChannelAttention(nn.Module):
    """
    Channel attention module from CBAM.
    
    Computes channel-wise attention weights using global average and max pooling
    followed by a shared MLP. This asks "which feature channels are important?"
    but does NOT compute relationships between spatial positions.
    """
    
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        # Shared MLP for both pooled features
        reduced_channels = max(channels // reduction, 8)  # Ensure minimum capacity
        self.mlp = nn.Sequential(
            nn.Linear(channels, reduced_channels, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduced_channels, channels, bias=False)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, _, _ = x.shape
        
        # Global pooling: (B, C, H, W) -> (B, C, 1, 1) -> (B, C)
        avg_out = self.mlp(self.avg_pool(x).view(B, C))
        max_out = self.mlp(self.max_pool(x).view(B, C))
        
        # Combine and apply sigmoid: (B, C) -> (B, C, 1, 1)
        attn = torch.sigmoid(avg_out + max_out).view(B, C, 1, 1)
        
        return attn * x


class SpatialAttention(nn.Module):
    """
    Spatial attention module from CBAM.
    
    Computes spatial attention weights using channel-wise average and max pooling
    followed by a convolution. This asks "which spatial locations are important?"
    but each location is evaluated INDEPENDENTLY - no pairwise comparisons.
    """
    
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Channel-wise pooling: (B, C, H, W) -> (B, 1, H, W)
        avg_out = x.mean(dim=1, keepdim=True)
        max_out = x.max(dim=1, keepdim=True)[0]
        
        # Concatenate and convolve: (B, 2, H, W) -> (B, 1, H, W)
        combined = torch.cat([avg_out, max_out], dim=1)
        attn = torch.sigmoid(self.conv(combined))
        
        return attn * x


class CBAM(nn.Module):
    """
    Convolutional Block Attention Module (CBAM).
    
    Applies channel attention followed by spatial attention sequentially.
    
    Key limitation for ARC tasks: CBAM computes "what's important" and "where's important"
    but NEVER computes pairwise relationships between positions. This means it cannot
    perform counting ("how many objects?") or selection ("which is smallest?") tasks
    that require comparing objects to each other.
    
    Use cases where CBAM helps:
    - Tasks where transformation depends on global image statistics
    - Feature refinement / suppressing irrelevant channels
    - When computational efficiency matters (much lighter than self-attention)
    
    Use cases where CBAM won't help:
    - Counting objects
    - Selecting objects by relative properties (smallest, unique, etc.)
    - Any task requiring explicit pairwise comparison between positions
    """
    
    def __init__(self, channels: int, reduction: int = 16, kernel_size: int = 7):
        super().__init__()
        self.channel_attn = ChannelAttention(channels, reduction)
        self.spatial_attn = SpatialAttention(kernel_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.channel_attn(x)
        x = self.spatial_attn(x)
        return x


class FiLMModulation(nn.Module):
    """
    Feature-wise Linear Modulation (FiLM) for injecting global context into skip connections.
    
    Takes a global context vector from the bottleneck and produces scale (γ) and shift (β)
    parameters for each skip connection level. This allows global reasoning (e.g., "the blue
    object is smallest") to directly influence all spatial locations in the decoder.
    """
    
    def __init__(self, bottleneck_dim: int, skip_dims: list):
        """
        Args:
            bottleneck_dim: Channel dimension of bottleneck (e.g., 512)
            skip_dims: List of channel dimensions for skip connections [x3_ch, x2_ch, x1_ch]
                       e.g., [256, 128, 64]
        """
        super().__init__()
        
        self.skip_dims = skip_dims
        
        # MLP to process global vector before generating modulation params
        self.global_mlp = nn.Sequential(
            nn.Linear(bottleneck_dim, bottleneck_dim),
            nn.ReLU(inplace=True),
        )
        
        # Separate heads for each skip level, each producing scale and shift
        self.film_generators = nn.ModuleList([
            nn.Linear(bottleneck_dim, dim * 2)  # *2 for both γ and β
            for dim in skip_dims
        ])
        
    def forward(self, bottleneck: torch.Tensor) -> list:
        """
        Args:
            bottleneck: (B, C, H, W) bottleneck features (after attention)
            
        Returns:
            List of (gamma, beta) tuples for each skip level
        """
        # Global average pool: (B, C, H, W) -> (B, C)
        global_vec = bottleneck.mean(dim=(2, 3))
        
        # Process through MLP
        global_vec = self.global_mlp(global_vec)
        
        # Generate modulation params for each skip level
        modulation_params = []
        for i, generator in enumerate(self.film_generators):
            params = generator(global_vec)  # (B, dim*2)
            gamma, beta = params.chunk(2, dim=-1)  # Each (B, dim)
            
            # Reshape for broadcasting: (B, dim) -> (B, dim, 1, 1)
            gamma = gamma.unsqueeze(-1).unsqueeze(-1)
            beta = beta.unsqueeze(-1).unsqueeze(-1)
            
            # Initialize gamma around 1 (multiplicative identity)
            # The linear layer will learn deviations from this
            gamma = gamma + 1.0
            
            modulation_params.append((gamma, beta))
            
        return modulation_params
    
    @staticmethod
    def apply_film(features: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
        """Apply FiLM modulation: out = gamma * features + beta"""
        return gamma * features + beta


class DilatedConvBlock(nn.Module):
    """Convolutional block with configurable dilation for growing receptive field without downsampling."""
    
    def __init__(self, in_ch: int, out_ch: int, dilation: int = 1):
        super().__init__()
        # Padding must equal dilation for 3x3 kernel to maintain spatial size
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=dilation, dilation=dilation),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=dilation, dilation=dilation),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
        
        # Residual connection if dimensions match
        self.residual = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
    
    def forward(self, x):
        return self.conv(x) + self.residual(x)


class FullResolutionCNN(nn.Module):
    """
    CNN that maintains full 30x30 resolution throughout, using dilated convolutions
    to grow receptive field without spatial downsampling.
    
    This allows attention to operate over 900 pixel positions rather than 9 spatial
    regions, potentially enabling better object-level reasoning.
    
    Architecture:
    - Dilated conv blocks grow receptive field: 1 -> 2 -> 4 -> 8
    - After sufficient RF, apply self-attention at full resolution
    - Final conv layers to produce output
    
    Receptive field growth with dilation:
    - dilation=1: 3x3 kernel sees 3x3 (RF grows by 2 per layer)
    - dilation=2: 3x3 kernel sees 5x5 effective spacing
    - dilation=4: 3x3 kernel sees 9x9 effective spacing
    - etc.
    """
    
    def __init__(self, hidden_dim: int = 64, force_comparison: bool = True, num_classes: int = 10,
                 attention_type: str = "self", attention_heads: int = 8, attention_layers: int = 2):
        super().__init__()
        
        self.force_comparison = force_comparison
        self.num_classes = num_classes
        self.attention_type = attention_type
        
        # Color embeddings for both grids
        self.input_embed = nn.Embedding(NUM_COLORS, 16)
        self.output_embed = nn.Embedding(NUM_COLORS, 16)
        
        # Input channels: 64 if force_comparison, else 32
        if force_comparison:
            in_channels = 64
        else:
            in_channels = 32
        
        # Dilated conv blocks - no downsampling, grow receptive field via dilation
        # RF after each block (cumulative):
        # Block 1 (d=1): ~5x5
        # Block 2 (d=2): ~13x13  
        # Block 3 (d=4): ~29x29 (covers full 30x30 grid)
        # Block 4 (d=8): ~61x61 (redundant but adds depth)
        self.conv1 = DilatedConvBlock(in_channels, hidden_dim, dilation=1)
        self.conv2 = DilatedConvBlock(hidden_dim, hidden_dim, dilation=2)
        self.conv3 = DilatedConvBlock(hidden_dim, hidden_dim, dilation=4)
        self.conv4 = DilatedConvBlock(hidden_dim, hidden_dim, dilation=8)
        
        # Attention at full resolution (30x30 = 900 tokens)
        if attention_type == "cbam":
            self.attention = CBAM(hidden_dim)
        else:  # "self" attention (default)
            self.attention = BottleneckAttention(
                dim=hidden_dim,
                num_heads=attention_heads,
                num_layers=attention_layers
            )
        
        # Post-attention conv blocks
        self.conv5 = DilatedConvBlock(hidden_dim, hidden_dim, dilation=4)
        self.conv6 = DilatedConvBlock(hidden_dim, hidden_dim, dilation=2)
        self.conv7 = DilatedConvBlock(hidden_dim, hidden_dim, dilation=1)
        
        # Output head
        self.outc = nn.Conv2d(hidden_dim, num_classes, kernel_size=1)
    
    def forward(self, input_grid: torch.Tensor, output_grid: torch.Tensor) -> torch.Tensor:
        # Embed both grids
        inp_emb = self.input_embed(input_grid)   # (B, H, W, 16)
        out_emb = self.output_embed(output_grid) # (B, H, W, 16)
        
        if self.force_comparison:
            diff = inp_emb - out_emb
            prod = inp_emb * out_emb
            x = torch.cat([inp_emb, out_emb, diff, prod], dim=-1)  # (B, H, W, 64)
        else:
            x = torch.cat([inp_emb, out_emb], dim=-1)  # (B, H, W, 32)
        
        x = x.permute(0, 3, 1, 2).contiguous()  # (B, C, H, W)
        
        # Dilated convolutions - grow receptive field while maintaining resolution
        x = self.conv1(x)  # (B, hidden_dim, 30, 30)
        x = self.conv2(x)  # (B, hidden_dim, 30, 30)
        x = self.conv3(x)  # (B, hidden_dim, 30, 30)
        x = self.conv4(x)  # (B, hidden_dim, 30, 30)
        
        # Full-resolution attention (900 tokens)
        x = self.attention(x)  # (B, hidden_dim, 30, 30)
        
        # Post-attention processing
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        
        # Output
        logits = self.outc(x)  # (B, num_classes, 30, 30)
        
        if self.num_classes == 1:
            return logits.squeeze(1)  # (B, H, W) for binary
        else:
            return logits  # (B, 10, H, W) for color prediction
    
    def predict_proba(self, input_grid: torch.Tensor, output_grid: torch.Tensor) -> torch.Tensor:
        logits = self.forward(input_grid, output_grid)
        if self.num_classes == 1:
            return torch.sigmoid(logits)
        else:
            return F.softmax(logits, dim=1)
    
    def predict_colors(self, input_grid: torch.Tensor, output_grid: torch.Tensor) -> torch.Tensor:
        logits = self.forward(input_grid, output_grid)
        if self.num_classes == 1:
            raise ValueError("predict_colors only valid for color mode (num_classes=10)")
        return logits.argmax(dim=1)
    
    @classmethod
    def from_checkpoint(cls, checkpoint_path: str, device: torch.device = None):
        checkpoint = torch.load(checkpoint_path, map_location=device or 'cpu', weights_only=False)
        args = checkpoint.get('args', {})
        hidden_dim = args.get('hidden_dim', 64)
        force_comparison = args.get('force_comparison', True)
        mode = args.get('mode', 'color')
        num_classes = 10 if mode == 'color' else 1
        attention_type = args.get('attention_type', 'self')
        attention_heads = args.get('attention_heads', 8)
        attention_layers = args.get('attention_layers', 2)
        
        model = cls(
            hidden_dim=hidden_dim,
            force_comparison=force_comparison,
            num_classes=num_classes,
            attention_type=attention_type,
            attention_heads=attention_heads,
            attention_layers=attention_layers
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        if device:
            model = model.to(device)
        return model


class PixelErrorCNN(nn.Module):
    """
    U-Net style CNN with EXPLICIT comparison between input and output.
    
    Instead of just concatenating embeddings, we compute:
    - input embedding
    - output embedding  
    - difference (input - output)
    - element-wise product (input * output)
    
    This forces the network to see both and compare them.
    
    Modes:
    - binary (num_classes=1): Predict correct (1) vs incorrect (0) per pixel
    - color (num_classes=10): Predict what color each pixel SHOULD be
    """

    def __init__(self, hidden_dim: int = 64, force_comparison: bool = True, num_classes: int = 1,
                 use_attention: bool = False, attention_type: str = "self", 
                 attention_heads: int = 8, attention_layers: int = 1,
                 no_skip: bool = False):
        super().__init__()
        
        self.force_comparison = force_comparison
        self.num_classes = num_classes
        self.use_attention = use_attention
        self.attention_type = attention_type
        self.no_skip = no_skip

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
        if no_skip:
            # No skip connections - all info must flow through bottleneck
            self.up1 = UpNoSkip(base_ch * 8, base_ch * 4)
            self.up2 = UpNoSkip(base_ch * 4, base_ch * 2)
            self.up3 = UpNoSkip(base_ch * 2, base_ch)
        else:
            # Standard U-Net with skip connections
            self.up1 = Up(base_ch * 8, base_ch * 4)
            self.up2 = Up(base_ch * 4, base_ch * 2)
            self.up3 = Up(base_ch * 2, base_ch)

        # Optional bottleneck attention for global reasoning
        if use_attention:
            if attention_type == "cbam":
                self.bottleneck_attn = CBAM(base_ch * 8)
            else:  # "self" attention (default)
                self.bottleneck_attn = BottleneckAttention(
                    dim=base_ch * 8,
                    num_heads=attention_heads,
                    num_layers=attention_layers
                )
            # FiLM modulation to inject global context into skip connections (only if using skips)
            # Note: FiLM is only used with self-attention since CBAM doesn't produce the same
            # kind of global context vector
            if not no_skip and attention_type == "self":
                self.film = FiLMModulation(
                    bottleneck_dim=base_ch * 8,
                    skip_dims=[base_ch * 4, base_ch * 2, base_ch]
                )

        # Output: 1 channel for binary, 10 channels for color prediction
        self.outc = nn.Conv2d(base_ch, num_classes, kernel_size=1)

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

        # Apply bottleneck attention for global reasoning (if enabled)
        if self.use_attention:
            x4 = self.bottleneck_attn(x4)
            
            # Generate FiLM modulation parameters from global context (only if using skip connections)
            # Note: FiLM is only used with self-attention, not CBAM
            if not self.no_skip and self.attention_type == "self":
                film_params = self.film(x4)  # [(γ3, β3), (γ2, β2), (γ1, β1)]
                
                # Modulate skip connections with global context
                x3 = FiLMModulation.apply_film(x3, *film_params[0])
                x2 = FiLMModulation.apply_film(x2, *film_params[1])
                x1 = FiLMModulation.apply_film(x1, *film_params[2])

        # Decoder
        if self.no_skip:
            # No skip connections - pass target sizes for proper upsampling
            x = self.up1(x4, target_size=(x3.size(2), x3.size(3)))
            x = self.up2(x, target_size=(x2.size(2), x2.size(3)))
            x = self.up3(x, target_size=(x1.size(2), x1.size(3)))
        else:
            # Standard U-Net with skip connections
            x = self.up1(x4, x3)
            x = self.up2(x, x2)
            x = self.up3(x, x1)

        logits = self.outc(x)
        
        if self.num_classes == 1:
            return logits.squeeze(1)  # (B, H, W) for binary
        else:
            return logits  # (B, 10, H, W) for color prediction

    def predict_proba(self, input_grid: torch.Tensor, output_grid: torch.Tensor) -> torch.Tensor:
        """For binary mode: return probability of pixel being correct"""
        logits = self.forward(input_grid, output_grid)
        if self.num_classes == 1:
            return torch.sigmoid(logits)
        else:
            # For color mode, return softmax probabilities
            return F.softmax(logits, dim=1)
    
    def predict_colors(self, input_grid: torch.Tensor, output_grid: torch.Tensor) -> torch.Tensor:
        """For color mode: return predicted color per pixel"""
        logits = self.forward(input_grid, output_grid)
        if self.num_classes == 1:
            raise ValueError("predict_colors only valid for color mode (num_classes=10)")
        return logits.argmax(dim=1)  # (B, H, W)

    @classmethod
    def from_checkpoint(cls, checkpoint_path: str, device: torch.device = None):
        checkpoint = torch.load(checkpoint_path, map_location=device or 'cpu', weights_only=False)
        args = checkpoint.get('args', {})
        hidden_dim = args.get('hidden_dim', 64)
        force_comparison = args.get('force_comparison', True)
        # Support both old checkpoints (binary only) and new ones
        mode = args.get('mode', 'binary')
        num_classes = 10 if mode == 'color' else 1
        # Attention parameters
        use_attention = args.get('use_attention', False)
        attention_type = args.get('attention_type', 'self')
        attention_heads = args.get('attention_heads', 8)
        attention_layers = args.get('attention_layers', 1)
        # Skip connection parameter
        no_skip = args.get('no_skip', False)
        
        model = cls(
            hidden_dim=hidden_dim,
            force_comparison=force_comparison,
            num_classes=num_classes,
            use_attention=use_attention,
            attention_type=attention_type,
            attention_heads=attention_heads,
            attention_layers=attention_layers,
            no_skip=no_skip
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
    
    Modes:
    - binary: Returns pixel_mask where 1=correct, 0=incorrect
    - color: Returns target_colors where each pixel has the correct color (0-9)
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
        mode: str = "binary",  # "binary" or "color"
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
        self.mode = mode
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
        print(f"Mode: {mode.upper()}")
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

        if self.mode == "binary":
            return (
                torch.from_numpy(final_input.copy()).long(),
                torch.from_numpy(final_output.copy()).long(),
                torch.from_numpy(pixel_mask.copy()).float(),
                torch.tensor(is_positive, dtype=torch.float32)
            )
        else:  # color mode
            return (
                torch.from_numpy(final_input.copy()).long(),
                torch.from_numpy(final_output.copy()).long(),
                torch.from_numpy(target_correct.copy()).long(),  # Target colors (0-9)
                torch.from_numpy(pixel_mask.copy()).float(),  # Still useful for metrics
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
    
    def __init__(self, puzzles: Dict, mode: str = "color", test_only: bool = True):
        self.mode = mode
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
    mode: str = "color",
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
            
            if mode == "color":
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
                
            else:  # binary mode
                # Binary mode doesn't make sense with recursion in the same way
                out_t = output_grid.unsqueeze(0).to(device)
                logits = model(inp_t, out_t)  # (1, H, W)
                pred_correct = (logits > 0)[0].cpu().numpy()
                
                pred_region = pred_correct[:out_h, :out_w]
                correct = pred_region.sum()
                total = out_h * out_w
                is_perfect = (correct == total)
            
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
    mode: str = "color",
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
        model, test_dataset, device, mode=mode, verbose=False, visualize=False
    )
    baseline_acc = baseline_results['pixel_accuracy']
    baseline_perfect = baseline_results['perfect_rate']
    
    print(f"BASELINE (no ablation):")
    print(f"  Pixel Accuracy: {baseline_acc:.2%}")
    print(f"  Perfect Rate:   {baseline_perfect:.2%}")
    print(f"{'─'*70}\n")
    
    # Determine which model architecture we have
    is_full_res = isinstance(model, FullResolutionCNN)
    
    # Define layers to ablate based on architecture
    if is_full_res:
        # FullResolutionCNN layers
        layer_groups = {
            "Embeddings": [
                ("input_embed", model.input_embed),
                ("output_embed", model.output_embed),
            ],
            "Pre-Attention Convs": [
                ("conv1 (dilation=1)", model.conv1),
                ("conv2 (dilation=2)", model.conv2),
                ("conv3 (dilation=4)", model.conv3),
                ("conv4 (dilation=8)", model.conv4),
            ],
            "Attention": [
                ("attention", model.attention),
            ],
            "Post-Attention Convs": [
                ("conv5 (dilation=4)", model.conv5),
                ("conv6 (dilation=2)", model.conv6),
                ("conv7 (dilation=1)", model.conv7),
            ],
        }
    else:
        # PixelErrorCNN (U-Net) layers
        layer_groups = {
            "Embeddings": [
                ("input_embed", model.input_embed),
                ("output_embed", model.output_embed),
            ],
            "Encoder": [
                ("inc (initial conv)", model.inc),
                ("down1 (encoder level 1)", model.down1),
                ("down2 (encoder level 2)", model.down2),
                ("down3 (encoder level 3 / bottleneck)", model.down3),
            ],
            "Decoder": [
                ("up1 (decoder level 1)", model.up1),
                ("up2 (decoder level 2)", model.up2),
                ("up3 (decoder level 3)", model.up3),
            ],
        }
        # Add attention if present
        if hasattr(model, 'bottleneck_attn') and model.use_attention:
            layer_groups["Bottleneck Attention"] = [
                ("bottleneck_attn", model.bottleneck_attn),
            ]
            if hasattr(model, 'film') and not model.no_skip:
                layer_groups["FiLM Modulation"] = [
                    ("film (global context injection)", model.film),
                ]
    
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
                model, test_dataset, device, mode=mode, verbose=False, visualize=False
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


def train_epoch_binary(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Dict[str, float]:
    """Training epoch for binary mode"""
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


def train_epoch_color(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Dict[str, float]:
    """Training epoch for color prediction mode"""
    model.train()

    total_loss = 0.0
    total_color_correct = 0
    total_pixels = 0
    total_error_pixels = 0
    total_error_color_correct = 0
    num_batches = 0

    pbar = tqdm(loader, desc="Training")
    for input_grid, output_grid, target_colors, pixel_mask in pbar:
        input_grid = input_grid.to(device)
        output_grid = output_grid.to(device)
        target_colors = target_colors.to(device)
        pixel_mask = pixel_mask.to(device)

        optimizer.zero_grad()

        logits = model(input_grid, output_grid)  # (B, 10, H, W)
        
        # Cross-entropy loss for color prediction
        loss = F.cross_entropy(logits, target_colors)

        loss.backward()
        optimizer.step()

        with torch.no_grad():
            pred_colors = logits.argmax(dim=1)  # (B, H, W)
            
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

            total_loss += loss.item()
            num_batches += 1

        pbar.set_postfix(loss=loss.item())

    color_accuracy = total_color_correct / max(total_pixels, 1)
    error_color_accuracy = total_error_color_correct / max(total_error_pixels, 1)

    return {
        "loss": total_loss / max(num_batches, 1),
        "color_accuracy": color_accuracy,
        "error_color_accuracy": error_color_accuracy,
    }


def evaluate_binary(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    """Evaluation for binary mode"""
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


def evaluate_color(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    """Evaluation for color prediction mode"""
    model.eval()

    total_loss = 0.0
    total_color_correct = 0
    total_pixels = 0
    total_error_pixels = 0
    total_error_color_correct = 0
    total_perfect = 0
    total_samples = 0
    num_batches = 0

    with torch.no_grad():
        for input_grid, output_grid, target_colors, pixel_mask in loader:
            input_grid = input_grid.to(device)
            output_grid = output_grid.to(device)
            target_colors = target_colors.to(device)
            pixel_mask = pixel_mask.to(device)

            logits = model(input_grid, output_grid)  # (B, 10, H, W)
            loss = F.cross_entropy(logits, target_colors)

            pred_colors = logits.argmax(dim=1)  # (B, H, W)
            
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
            num_batches += 1

    color_accuracy = total_color_correct / max(total_pixels, 1)
    error_color_accuracy = total_error_color_correct / max(total_error_pixels, 1)
    perfect_rate = total_perfect / max(total_samples, 1)

    return {
        "loss": total_loss / max(num_batches, 1),
        "color_accuracy": color_accuracy,
        "error_color_accuracy": error_color_accuracy,
        "perfect_rate": perfect_rate,
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


def visualize_predictions_binary(model: nn.Module, dataset: Dataset, device: torch.device, num_samples: int = 8):
    """Visualize predictions for binary mode"""
    model.eval()

    print("\n" + "="*80)
    print("Sample Predictions (Binary Mode - Error Detection)")
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

            input_grid, output_grid, pixel_mask, is_positive = dataset[sample_idx]

            inp_t = input_grid.unsqueeze(0).to(device)
            out_t = output_grid.unsqueeze(0).to(device)

            with torch.no_grad():
                pred_proba = model.predict_proba(inp_t, out_t)[0].cpu().numpy()

            input_np = input_grid.numpy()
            output_np = output_grid.numpy()
            mask_np = pixel_mask.numpy()

            # Find content bounds SEPARATELY for input and output
            inp_r, inp_c = _find_grid_bounds(input_np)
            out_r, out_c = _find_grid_bounds(output_np)

            # For all-zeros output, use mask to find content region
            if out_r == 1 and out_c == 1:
                error_positions = np.where(mask_np == 0)
                if len(error_positions[0]) > 0:
                    out_r = min(error_positions[0].max() + 1, 30)
                    out_c = min(error_positions[1].max() + 1, 30)

            num_errors = int((mask_np[:out_r, :out_c] == 0).sum())
            expected = "ALL CORRECT" if is_positive > 0.5 else f"{num_errors} errors"

            print(f"\nInput: {inp_r}Ã—{inp_c} â†’ Output: {out_r}Ã—{out_c}")
            print(f"Expected: {expected}")

            # Print INPUT separately with its own dimensions
            print(f"\nINPUT ({inp_r}Ã—{inp_c}):")
            for r in range(inp_r):
                inp_row = " ".join(f"{input_np[r, c]}" for c in range(inp_c))
                print(f"  {inp_row}")

            # Print OUTPUT and PREDICTED ERRORS with output dimensions
            print(f"\n{'OUTPUT':<30} {'PREDICTED ERRORS':<30}")
            for r in range(out_r):
                out_row = " ".join(f"{output_np[r, c]}" for c in range(out_c))
                err_row = " ".join("X" if pred_proba[r, c] < 0.5 else "Â·" for c in range(out_c))
                print(f"  {out_row:<28} {err_row:<28}")

            # Stats (only count within content area)
            num_predicted_errors = (pred_proba[:out_r, :out_c] < 0.5).sum()
            print(f"Predicted errors: {num_predicted_errors}")

    print("="*80 + "\n")


def visualize_predictions_color(model: nn.Module, dataset: Dataset, device: torch.device, num_samples: int = 8):
    """Visualize predictions for color mode"""
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


def visualize_counting_predictions(model: nn.Module, test_loader: DataLoader,
                                    grid_size: int, device: torch.device,
                                    num_samples: int = 4):
    """
    Visualize predictions for the counting experiment.
    Shows input grid, expected output (3x3), and predicted output (3x3).
    """
    model.eval()

    print(f"\n{'─'*80}")
    print(f"SAMPLE PREDICTIONS (Grid {grid_size}x{grid_size} → Output 3x3)")
    print(f"{'─'*80}")

    # Get a batch of test data
    batch = next(iter(test_loader))
    input_grids, output_grids, targets, _ = batch

    input_grids = input_grids.to(device)
    output_grids = output_grids.to(device)

    with torch.no_grad():
        logits = model(input_grids, output_grids)
        predictions = logits.argmax(dim=1)  # (B, H, W)

    # Show up to num_samples examples
    num_to_show = min(num_samples, input_grids.size(0))

    for i in range(num_to_show):
        inp = input_grids[i].cpu().numpy()
        target = targets[i].cpu().numpy()
        pred = predictions[i].cpu().numpy()

        # Find actual content size for input (exclude padding)
        inp_rows = inp.shape[0]
        inp_cols = inp.shape[1]
        # Find non-padded region
        for r in range(inp_rows - 1, -1, -1):
            if any(inp[r, :] != 0):
                inp_rows = r + 1
                break
        for c in range(inp_cols - 1, -1, -1):
            if any(inp[:inp_rows, c] != 0):
                inp_cols = c + 1
                break
        inp_rows = max(inp_rows, grid_size)
        inp_cols = max(inp_cols, grid_size)

        # Output is always 3x3
        out_rows, out_cols = 3, 3

        # Count colors in input to show analysis
        from collections import Counter
        inp_flat = inp[:grid_size, :grid_size].flatten()
        color_counts = Counter(inp_flat)
        most_common = color_counts.most_common(3)

        # Get expected and predicted winner colors
        expected_color = target[0, 0]  # All same in target
        predicted_color = pred[0, 0]  # Check prediction at (0,0)
        is_correct = (pred[:out_rows, :out_cols] == target[:out_rows, :out_cols]).all()

        status = "✓ CORRECT" if is_correct else "✗ WRONG"

        print(f"\n[Example {i+1}] {status}")
        print(f"  Top colors: {', '.join(f'{c}:{cnt}' for c, cnt in most_common)}")
        print(f"  Expected: {expected_color}, Predicted: {predicted_color}")

        # Print input grid (truncated if large)
        print(f"\n  INPUT ({grid_size}x{grid_size}):")
        max_display = min(grid_size, 10)  # Limit display for large grids
        for r in range(max_display):
            row_str = " ".join(f"{inp[r, c]}" for c in range(max_display))
            suffix = " ..." if grid_size > 10 else ""
            print(f"    {row_str}{suffix}")
        if grid_size > 10:
            print(f"    ... ({grid_size - 10} more rows)")

        # Print output side by side: EXPECTED vs PREDICTED
        print(f"\n  {'EXPECTED (3x3)':<20} {'PREDICTED (3x3)':<20}")
        for r in range(out_rows):
            exp_row = " ".join(f"{target[r, c]}" for c in range(out_cols))
            pred_row = " ".join(f"{pred[r, c]}" for c in range(out_cols))
            print(f"    {exp_row:<18} {pred_row:<18}")

    print(f"\n{'─'*80}\n")


# =============================================================================
# Counting Experiment - Synthetic Puzzle Generation
# =============================================================================

def generate_counting_puzzle(grid_size: int, num_colors: int, rng: np.random.Generator,
                              output_size: int = 3) -> Dict:
    """
    Generate a single 'most frequent color' puzzle example.

    The winner color has exactly 1 more pixel than the runner-up (hardest case).
    Output is always a fixed-size grid (default 3x3) filled with the winner color.
    This decouples input complexity from output size.
    """
    # Pick num_colors random colors from 0-9
    available_colors = list(range(10))
    rng.shuffle(available_colors)
    colors = available_colors[:num_colors]

    total_pixels = grid_size * grid_size

    # Distribute pixels with winner having exactly 1 more than runner-up
    # Strategy: make counts as equal as possible, then give winner +1
    base_count = total_pixels // num_colors
    remainder = total_pixels % num_colors

    # Start with base counts
    counts = [base_count] * num_colors

    # Distribute remainder evenly (but not to winner yet)
    winner_idx = rng.integers(num_colors)
    runner_up_idx = (winner_idx + 1) % num_colors

    # Give remainder to non-winners first
    extra_indices = [i for i in range(num_colors) if i != winner_idx]
    for i in range(remainder):
        counts[extra_indices[i % len(extra_indices)]] += 1

    # Now ensure winner has exactly 1 more than the current max (runner-up)
    current_max = max(counts[i] for i in range(num_colors) if i != winner_idx)
    counts[winner_idx] = current_max + 1

    # Find actual runner-up index (the one with current_max count)
    runner_up_idx = None
    for i in range(num_colors):
        if i != winner_idx and counts[i] == current_max:
            runner_up_idx = i
            break

    # Adjust total if we went over
    current_total = sum(counts)
    if current_total > total_pixels:
        # Remove from non-winner, non-runner-up colors only
        diff = current_total - total_pixels
        for i in range(num_colors):
            if i != winner_idx and i != runner_up_idx and diff > 0 and counts[i] > 1:
                remove = min(diff, counts[i] - 1)
                counts[i] -= remove
                diff -= remove
        # If still over (not enough other colors), we must accept larger margin
        # This shouldn't happen with reasonable num_colors
    elif current_total < total_pixels:
        # Add to winner
        counts[winner_idx] += total_pixels - current_total

    # Build input grid
    pixels = []
    for color, count in zip(colors, counts):
        pixels.extend([color] * count)

    # Ensure exact size
    while len(pixels) < total_pixels:
        pixels.append(colors[winner_idx])
    pixels = pixels[:total_pixels]

    # Shuffle pixel positions
    rng.shuffle(pixels)
    input_grid = np.array(pixels).reshape(grid_size, grid_size).tolist()

    # Output is all winner color (fixed size, default 3x3)
    winner_color = colors[winner_idx]
    output_grid = [[winner_color] * output_size for _ in range(output_size)]

    return {"input": input_grid, "output": output_grid}


def generate_counting_puzzles(grid_size: int, num_train: int, num_test: int,
                               num_colors: int, seed: int = 42) -> Dict:
    """
    Generate synthetic puzzle dict matching ARC format.

    Returns a dict with one puzzle containing train and test examples.
    """
    rng = np.random.default_rng(seed)

    train_examples = [generate_counting_puzzle(grid_size, num_colors, rng)
                      for _ in range(num_train)]
    test_examples = [generate_counting_puzzle(grid_size, num_colors, rng)
                     for _ in range(num_test)]

    puzzle_id = f"counting_{grid_size}x{grid_size}"

    return {
        puzzle_id: {
            "train": train_examples,
            "test": test_examples
        }
    }


def run_counting_experiment(args):
    """
    Run the counting experiment across multiple grid sizes.

    For each grid size:
    1. Generate synthetic 'most frequent color' puzzles
    2. Train PixelErrorCNN from scratch
    3. Evaluate on held-out test examples
    4. Run ablation analysis
    5. Collect and report results
    """
    print("=" * 80)
    print("COUNTING EXPERIMENT: Most Frequent Color Task")
    print("=" * 80)
    print(f"Device: {DEVICE}")
    print(f"Grid sizes to test: {args.counting_grid_sizes}")
    print(f"Colors per grid: {args.counting_num_colors}")
    print(f"Train examples per size: {args.counting_num_train}")
    print(f"Test examples per size: {args.counting_num_test}")
    print(f"Negatives per example: {args.counting_num_negatives}")
    print(f"Epochs per size: {args.epochs}")
    print(f"Winner margin: 1 pixel (hardest case)")
    print("=" * 80)

    # Force color mode for this experiment (we're predicting what color output should be)
    args.mode = "color"
    args.eval_on_test = True
    args.ablation = True

    # Collect results across all grid sizes
    all_results = []

    for grid_size in args.counting_grid_sizes:
        print(f"\n{'='*80}")
        print(f"GRID SIZE: {grid_size}x{grid_size}")
        print(f"{'='*80}")

        # Generate puzzles for this grid size
        puzzles = generate_counting_puzzles(
            grid_size=grid_size,
            num_train=args.counting_num_train,
            num_test=args.counting_num_test,
            num_colors=args.counting_num_colors,
            seed=args.seed + grid_size  # Different seed per size for variety
        )

        puzzle_id = list(puzzles.keys())[0]
        puzzle = puzzles[puzzle_id]

        print(f"Generated {len(puzzle['train'])} train, {len(puzzle['test'])} test examples")

        # Verify puzzle correctness (sanity check)
        sample = puzzle['train'][0]
        input_flat = [c for row in sample['input'] for c in row]
        output_color = sample['output'][0][0]
        from collections import Counter
        counts = Counter(input_flat)
        most_common = counts.most_common(1)[0]
        print(f"Sample puzzle: most frequent color = {most_common[0]} (count={most_common[1]}), output = {output_color}")

        # Setup datasets
        use_augment = not args.no_augment
        dihedral_only = args.dihedral_only
        color_only = args.color_only

        # Create separate puzzle dicts for train and test
        # CorrespondenceDataset pulls from "train" key, so we put our data there
        train_puzzle = {puzzle_id: {"train": puzzle["train"]}}
        test_puzzle = {puzzle_id: {"train": puzzle["test"]}}  # Put test in "train" key for loading

        # For counting experiment: ONLY all_zeros samples
        # Model sees (input, blank) and must predict the correct output
        num_all_zeros = args.counting_num_negatives

        train_dataset = CorrespondenceDataset(
            puzzles=train_puzzle,
            num_positives=0,
            num_corrupted=0,
            num_wrong_input=0,
            num_mismatched_aug=0,
            num_color_swap=0,
            num_all_zeros=num_all_zeros,
            num_constant_fill=0,
            num_random_noise=0,
            augment=use_augment,
            dihedral_only=dihedral_only,
            color_only=color_only,
            mode=args.mode,
            include_test=False
        )

        # For test: give model (input, all_zeros) and see if it predicts correct output
        # This tests actual prediction ability, not just copying
        test_dataset = CorrespondenceDataset(
            puzzles=test_puzzle,
            num_positives=0,  # Don't give correct answer!
            num_corrupted=0,
            num_wrong_input=0,
            num_mismatched_aug=0,
            num_color_swap=0,
            num_all_zeros=1,  # Give blank output, model must predict correct color
            num_constant_fill=0,
            num_random_noise=0,
            augment=False,  # No augmentation for test
            dihedral_only=False,
            color_only=False,
            mode=args.mode,
            include_test=False
        )

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                                  shuffle=True, num_workers=args.num_workers)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                                 shuffle=False, num_workers=args.num_workers)

        print(f"Train samples: {len(train_dataset)}, Test samples: {len(test_dataset)}")

        # Create model
        num_classes = NUM_COLORS if args.mode == "color" else 1
        model = PixelErrorCNN(
            hidden_dim=args.hidden_dim,
            force_comparison=not args.no_force_comparison,
            num_classes=num_classes,
            use_attention=args.use_attention,
            attention_type=args.attention_type,
            attention_heads=args.attention_heads,
            attention_layers=args.attention_layers,
            no_skip=args.no_skip
        ).to(DEVICE)

        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)

        # Training loop
        best_test_acc = 0
        best_epoch = 0

        for epoch in range(args.epochs):
            model.train()
            total_loss = 0
            correct = 0
            total = 0

            for batch in train_loader:
                # CorrespondenceDataset returns tuple: (input, output, target, mask) for color mode
                input_grid, output_grid, target, _ = batch
                input_grid = input_grid.to(DEVICE)
                output_grid = output_grid.to(DEVICE)
                target = target.to(DEVICE)

                optimizer.zero_grad()
                logits = model(input_grid, output_grid)

                if args.mode == "color":
                    # Cross-entropy for color prediction
                    loss = F.cross_entropy(logits.permute(0, 2, 3, 1).reshape(-1, NUM_COLORS),
                                           target.reshape(-1).long())
                    preds = logits.argmax(dim=1)
                    correct += (preds == target).sum().item()
                    total += target.numel()
                else:
                    # Binary cross-entropy
                    loss = F.binary_cross_entropy_with_logits(logits.squeeze(1), target)
                    preds = (logits.squeeze(1) > 0).float()
                    correct += (preds == target).sum().item()
                    total += target.numel()

                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            scheduler.step()

            # Evaluate on test set
            model.eval()
            test_correct = 0
            test_total = 0
            test_grid_correct = 0
            test_grid_total = 0

            with torch.no_grad():
                for batch in test_loader:
                    input_grid, output_grid, target, _ = batch
                    input_grid = input_grid.to(DEVICE)
                    output_grid = output_grid.to(DEVICE)
                    target = target.to(DEVICE)

                    logits = model(input_grid, output_grid)

                    if args.mode == "color":
                        preds = logits.argmax(dim=1)
                        test_correct += (preds == target).sum().item()
                        test_total += target.numel()
                        # Grid-level accuracy (entire grid must be correct)
                        grid_match = (preds == target).all(dim=(1, 2))
                        test_grid_correct += grid_match.sum().item()
                        test_grid_total += grid_match.size(0)
                    else:
                        preds = (logits.squeeze(1) > 0).float()
                        test_correct += (preds == target).sum().item()
                        test_total += target.numel()

            train_acc = correct / total if total > 0 else 0
            test_acc = test_correct / test_total if test_total > 0 else 0
            test_grid_acc = test_grid_correct / test_grid_total if test_grid_total > 0 else 0

            if test_acc > best_test_acc:
                best_test_acc = test_acc
                best_epoch = epoch + 1

            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"  Epoch {epoch+1:3d}: Train={train_acc:.1%}, Test={test_acc:.1%}, GridAcc={test_grid_acc:.1%}")

        print(f"\n  Best test accuracy: {best_test_acc:.1%} (epoch {best_epoch})")
        print(f"  Final grid accuracy: {test_grid_acc:.1%}")

        # Run ablation analysis
        print(f"\n  Running ablation analysis...")
        ablation_results = run_ablation_analysis(model, test_loader, args.mode)

        # Visualize predictions
        visualize_counting_predictions(model, test_loader, grid_size, DEVICE, num_samples=4)

        # Store one sample for final summary visualization
        sample_batch = next(iter(test_loader))
        sample_input, sample_output, sample_target, _ = sample_batch
        with torch.no_grad():
            sample_logits = model(sample_input[:1].to(DEVICE), sample_output[:1].to(DEVICE))
            sample_pred = sample_logits.argmax(dim=1)[0].cpu().numpy()
        sample_viz = {
            "input": sample_input[0].numpy(),
            "target": sample_target[0].numpy(),
            "pred": sample_pred
        }

        # Store results
        result = {
            "grid_size": grid_size,
            "best_test_acc": best_test_acc,
            "final_test_acc": test_acc,
            "final_grid_acc": test_grid_acc,
            "best_epoch": best_epoch,
            "ablation": ablation_results,
            "sample_viz": sample_viz
        }
        all_results.append(result)

        # Print ablation summary (detailed output already printed by run_ablation_analysis)
        # Just show a compact per-grid-size summary here
        print(f"\n  Ablation Summary (drop when layer zeroed):")
        for layer_name, layer_data in ablation_results["layers"].items():
            if "error" not in layer_data:
                grid_drop = layer_data["grid_drop"]
                pixel_drop = layer_data["accuracy_drop"]
                print(f"    {layer_name}: grid {grid_drop:+.1%}, pixel {pixel_drop:+.1%}")

    # Final summary
    print("\n" + "=" * 80)
    print("COUNTING EXPERIMENT SUMMARY")
    print("=" * 80)
    print(f"{'Grid Size':<12} {'Best Acc':<12} {'Grid Acc':<12} {'Best Epoch':<12}")
    print("-" * 48)

    for r in all_results:
        print(f"{r['grid_size']}x{r['grid_size']:<9} {r['best_test_acc']:<12.1%} {r['final_grid_acc']:<12.1%} {r['best_epoch']:<12}")

    print("\n" + "=" * 80)
    print("ABLATION SUMMARY BY GRID SIZE")
    print("=" * 80)

    # Get all layer names from first result (using short attr names for table)
    if all_results and all_results[0]["ablation"].get("layers"):
        # Get short layer names (attr_name) for compact display
        first_layers = all_results[0]["ablation"]["layers"]
        layer_info = [(name, data.get("attr_name", name)) for name, data in first_layers.items()
                      if "error" not in data]

        # Print header with short names
        header = f"{'Grid':<8}"
        for _, short_name in layer_info[:6]:  # Limit to first 6 layers for readability
            header += f"{short_name:<12}"
        print(header)
        print("-" * (8 + 12 * min(6, len(layer_info))))

        # Show grid accuracy drops
        print("\nGrid Accuracy Drop:")
        for r in all_results:
            row = f"{r['grid_size']}x{r['grid_size']:<5}"
            layers_data = r["ablation"].get("layers", {})
            for full_name, short_name in layer_info[:6]:
                layer_data = layers_data.get(full_name, {})
                drop = layer_data.get("grid_drop", 0) if "error" not in layer_data else 0
                row += f"{drop:+.1%}       "
            print(row)

        # Show pixel accuracy drops
        print("\nPixel Accuracy Drop:")
        for r in all_results:
            row = f"{r['grid_size']}x{r['grid_size']:<5}"
            layers_data = r["ablation"].get("layers", {})
            for full_name, short_name in layer_info[:6]:
                layer_data = layers_data.get(full_name, {})
                drop = layer_data.get("accuracy_drop", 0) if "error" not in layer_data else 0
                row += f"{drop:+.1%}       "
            print(row)

        # Also show baseline accuracy trend
        print("\n" + "-" * 60)
        print("Baseline Accuracy by Grid Size:")
        for r in all_results:
            baseline = r["ablation"].get("baseline", {})
            pixel_acc = baseline.get("accuracy", 0)
            grid_acc = baseline.get("grid_accuracy", 0)
            print(f"  {r['grid_size']}x{r['grid_size']}: Pixel={pixel_acc:.1%}, Grid={grid_acc:.1%}")

    # Final visualization: one sample from each grid size
    print("\n" + "=" * 80)
    print("SAMPLE PREDICTIONS BY GRID SIZE")
    print("=" * 80)

    for r in all_results:
        grid_size = r["grid_size"]
        viz = r["sample_viz"]
        inp = viz["input"]
        target = viz["target"]
        pred = viz["pred"]

        # Get expected and predicted winner colors
        expected_color = target[0, 0]
        predicted_color = pred[0, 0]
        is_correct = (pred[:3, :3] == target[:3, :3]).all()
        status = "✓" if is_correct else "✗"

        # Count colors in input
        from collections import Counter
        inp_flat = inp[:grid_size, :grid_size].flatten()
        color_counts = Counter(inp_flat)
        top_colors = color_counts.most_common(3)

        print(f"\n[{grid_size}x{grid_size}] {status} Expected: {expected_color}, Predicted: {predicted_color}")
        print(f"  Top colors: {', '.join(f'{c}:{cnt}' for c, cnt in top_colors)}")

        # Show input (compact for large grids)
        max_display = min(grid_size, 8)
        print(f"  Input ({grid_size}x{grid_size}):", end="")
        if grid_size <= 8:
            print()
            for row in range(grid_size):
                print(f"    {' '.join(str(inp[row, c]) for c in range(grid_size))}")
        else:
            print(f" (showing {max_display}x{max_display})")
            for row in range(max_display):
                print(f"    {' '.join(str(inp[row, c]) for c in range(max_display))} ...")

        # Show expected vs predicted (3x3)
        print(f"  Expected → Predicted:")
        for row in range(3):
            exp_row = ' '.join(str(target[row, c]) for c in range(3))
            pred_row = ' '.join(str(pred[row, c]) for c in range(3))
            print(f"    {exp_row}    {pred_row}")

    print("\n" + "=" * 80)
    print("Done!")


def run_ablation_analysis(model, test_loader, mode, verbose: bool = True):
    """
    Run layer-wise ablation using forward hooks (consistent with run_layer_ablation).

    For each layer, we temporarily zero out its output and measure the impact
    on test accuracy. Larger drops indicate more critical layers.

    Args:
        model: The model to analyze
        test_loader: DataLoader for test data
        mode: "color" or "binary"
        verbose: Whether to print detailed output

    Returns:
        Dict with 'baseline' and 'layers' keys containing ablation results
    """
    model.eval()

    def evaluate_loader(loader, device, mode):
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

                if mode == "color":
                    preds = logits.argmax(dim=1)
                else:
                    preds = (logits.squeeze(1) > 0).float()

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

    baseline_acc, baseline_grid_acc = evaluate_loader(test_loader, DEVICE, mode)

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
            acc, grid_acc = evaluate_loader(test_loader, DEVICE, mode)

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
    
    # Mode selection
    parser.add_argument("--mode", type=str, default="binary", choices=["binary", "color"],
                        help="binary: detect errors (correct/incorrect), color: predict correct colors (0-9)")
    
    # Architecture options
    parser.add_argument("--no-force-comparison", action="store_true",
                        help="Disable explicit comparison features (for ablation)")
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

    # Bottleneck attention for global reasoning
    parser.add_argument("--use-attention", action="store_true",
                        help="Add self-attention at U-Net bottleneck for global reasoning")
    parser.add_argument("--attention-type", type=str, default="self",
                        choices=["self", "cbam"],
                        help="Type of attention: 'self' for multi-head self-attention (pairwise comparisons), "
                             "'cbam' for channel+spatial attention (no pairwise comparisons)")
    parser.add_argument("--attention-heads", type=int, default=8,
                        help="Number of attention heads (default: 8)")
    parser.add_argument("--attention-layers", type=int, default=1,
                        help="Number of attention layers at bottleneck (default: 1)")
    parser.add_argument("--no-skip", action="store_true",
                        help="Remove skip connections - forces all info through bottleneck")
    
    # Full resolution architecture (no downsampling)
    parser.add_argument("--full-resolution", action="store_true",
                        help="Use FullResolutionCNN instead of U-Net (no downsampling, dilated convs, attention at 30x30)")
    
    # Layer-wise ablation analysis
    parser.add_argument("--ablation", action="store_true",
                        help="Run layer-wise ablation analysis after training. "
                             "Zeros each layer's output and measures accuracy drop. "
                             "Requires --eval-on-test to have test data to evaluate against.")

    # Counting experiment mode
    parser.add_argument("--counting-experiment", action="store_true",
                        help="Run synthetic 'most frequent color' counting experiment across grid sizes")
    parser.add_argument("--counting-grid-sizes", type=int, nargs="+",
                        default=[3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20, 25, 30],
                        help="Grid sizes to test in counting experiment")
    parser.add_argument("--counting-num-colors", type=int, default=5,
                        help="Number of distinct colors per grid in counting experiment")
    parser.add_argument("--counting-num-train", type=int, default=100,
                        help="Number of training examples per grid size")
    parser.add_argument("--counting-num-test", type=int, default=20,
                        help="Number of test examples per grid size")
    parser.add_argument("--counting-num-negatives", type=int, default=8,
                        help="Number of negative samples per example in counting experiment")

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Handle counting experiment mode separately
    if args.counting_experiment:
        run_counting_experiment(args)
        return

    print(f"Device: {DEVICE}")
    print(f"Dataset: {args.dataset}")
    print(f"Mode: {args.mode.upper()}")
    print(f"  - binary: Predict if each pixel is correct (1) or incorrect (0)")
    print(f"  - color: Predict what color each pixel SHOULD be (0-9)")
    print(f"Force comparison: {not args.no_force_comparison}")

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
        mode=args.mode,
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
        mode=args.mode,
        include_test=include_test_in_train,
    )
    
    # For --eval-on-test, create separate test evaluation dataset
    test_eval_dataset = None
    if args.eval_on_test:
        test_eval_dataset = TestEvalDataset(train_puzzles, mode=args.mode, test_only=True)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # Create model
    print("\nCreating model...")
    force_comparison = not args.no_force_comparison
    num_classes = 10 if args.mode == "color" else 1
    
    if args.full_resolution:
        # Full resolution architecture - no downsampling, dilated convs, attention at 30x30
        print("Architecture: FullResolutionCNN (no downsampling)")
        model = FullResolutionCNN(
            hidden_dim=args.hidden_dim,
            force_comparison=force_comparison,
            num_classes=num_classes,
            attention_type=args.attention_type,
            attention_heads=args.attention_heads,
            attention_layers=args.attention_layers
        )
    else:
        # Standard U-Net architecture
        print("Architecture: U-Net")
        model = PixelErrorCNN(
            hidden_dim=args.hidden_dim,
            force_comparison=force_comparison,
            num_classes=num_classes,
            use_attention=args.use_attention,
            attention_type=args.attention_type,
            attention_heads=args.attention_heads,
            attention_layers=args.attention_layers,
            no_skip=args.no_skip
        )
    model = model.to(DEVICE)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")
    print(f"Output classes: {num_classes} ({'color prediction' if num_classes == 10 else 'binary error detection'})")
    if args.full_resolution:
        if args.attention_type == "cbam":
            print(f"Full-resolution attention: CBAM (channel + spatial, NO pairwise comparisons)")
        else:
            print(f"Full-resolution attention: Self-attention, {args.attention_layers} layer(s), {args.attention_heads} heads over 900 tokens")
    elif args.use_attention:
        if args.attention_type == "cbam":
            print(f"Bottleneck attention: CBAM (channel + spatial, NO pairwise comparisons)")
        else:
            print(f"Bottleneck attention: Self-attention, {args.attention_layers} layer(s), {args.attention_heads} heads")
    if args.no_skip and not args.full_resolution:
        print("Skip connections: DISABLED (all info flows through bottleneck)")

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)

    # Create checkpoint directory
    os.makedirs(os.path.dirname(args.save_path) or ".", exist_ok=True)

    # Training
    print("\n" + "="*60)
    print(f"Starting Training ({args.mode.upper()} mode)")
    print("="*60)

    best_val_metric = 0.0
    metric_name = "error_iou" if args.mode == "binary" else "color_accuracy"

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")

        if args.mode == "binary":
            train_metrics = train_epoch_binary(model, train_loader, optimizer, DEVICE)
            val_metrics = evaluate_binary(model, val_loader, DEVICE)
            
            print(f"  Train Loss: {train_metrics['loss']:.4f}, Pix Acc: {train_metrics['pixel_accuracy']:.2%}, "
                  f"Err P/R: {train_metrics['error_precision']:.2%}/{train_metrics['error_recall']:.2%}, "
                  f"IoU: {train_metrics['error_iou']:.2%}")
            print(f"  Val   Loss: {val_metrics['loss']:.4f}, Pix Acc: {val_metrics['pixel_accuracy']:.2%}, "
                  f"Err P/R: {val_metrics['error_precision']:.2%}/{val_metrics['error_recall']:.2%}, "
                  f"IoU: {val_metrics['error_iou']:.2%}")
            
            current_metric = val_metrics['error_iou']
        else:
            train_metrics = train_epoch_color(model, train_loader, optimizer, DEVICE)
            val_metrics = evaluate_color(model, val_loader, DEVICE)
            
            print(f"  Train Loss: {train_metrics['loss']:.4f}, Color Acc: {train_metrics['color_accuracy']:.2%}, "
                  f"Error Color Acc: {train_metrics['error_color_accuracy']:.2%}")
            print(f"  Val   Loss: {val_metrics['loss']:.4f}, Color Acc: {val_metrics['color_accuracy']:.2%}, "
                  f"Error Color Acc: {val_metrics['error_color_accuracy']:.2%}, "
                  f"Perfect: {val_metrics['perfect_rate']:.2%}")
            
            current_metric = val_metrics['color_accuracy']

        scheduler.step()

        if current_metric > best_val_metric:
            best_val_metric = current_metric
            print(f"  [New best {metric_name}: {best_val_metric:.2%}]")
            
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'args': vars(args),
                'epoch': epoch,
                f'best_val_{metric_name}': best_val_metric,
                'force_comparison': force_comparison,
                'mode': args.mode,
                'num_classes': num_classes,
            }
            torch.save(checkpoint, args.save_path)

    # Final summary
    print("\n" + "="*60)
    print("Training Complete")
    print("="*60)
    print(f"Best {metric_name}: {best_val_metric:.2%}")
    print(f"Checkpoint saved to: {args.save_path}")

    # Visualize on training data
    if args.mode == "binary":
        visualize_predictions_binary(model, val_dataset, DEVICE)
    else:
        visualize_predictions_color(model, val_dataset, DEVICE)

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
            mode=args.mode, verbose=True, visualize=args.visualize,
            recursive_iters=args.recursive_iters
        )
        
        print(f"\n{'â”€'*60}")
        print(f"TEST SET RESULTS:")
        print(f"  Pixel Accuracy: {test_results['pixel_accuracy']:.2%}")
        print(f"  Perfect Examples: {test_results['perfect_examples']}/{test_results['total_examples']} ({test_results['perfect_rate']:.2%})")
        print(f"{'â”€'*60}")
        
        if test_results['perfect_rate'] == 1.0:
            print("\nðŸŽ‰ PERFECT GENERALIZATION! The CNN learned the transformation rule!")
            print("   This suggests the CNN can solve this puzzle from training examples alone.")
        elif test_results['pixel_accuracy'] > 0.95:
            print("\nâœ“ Strong generalization - CNN learned most of the rule.")
            print("  Minor errors may be edge cases or noise.")
        elif test_results['pixel_accuracy'] > 0.8:
            print("\n~ Partial generalization - CNN learned some patterns but not the full rule.")
        else:
            print("\nâœ— Poor generalization - CNN may have memorized training examples.")
            print("  The transformation rule was not learned.")
        
        # Also show comparison with training examples
        print("\n" + "â”€"*60)
        print("Comparison - Evaluating on TRAINING examples (sanity check):")
        train_eval_dataset = TestEvalDataset(train_puzzles, mode=args.mode, test_only=False)
        train_results = evaluate_test_examples(
            model, train_eval_dataset, DEVICE,
            mode=args.mode, verbose=True, visualize=False,  # Don't visualize training examples
            recursive_iters=args.recursive_iters
        )
        print(f"\nTRAINING SET RESULTS:")
        print(f"  Pixel Accuracy: {train_results['pixel_accuracy']:.2%}")
        print(f"  Perfect Examples: {train_results['perfect_examples']}/{train_results['total_examples']} ({train_results['perfect_rate']:.2%})")
        
        # Summary comparison
        print("\n" + "="*60)
        print("GENERALIZATION SUMMARY")
        print("="*60)
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
                model, test_eval_dataset, DEVICE, mode=args.mode
            )

    print("\nDone!")
    if not args.eval_on_test:
        print(f"\nRun the diagnostic to verify the CNN (binary mode only):")
        print(f"  python diagnose_cnn.py --checkpoint {args.save_path}")
    print(f"\nTo test generalization on a single puzzle:")
    print(f"  python train_pixel_error_cnn.py --single-puzzle PUZZLE_ID --mode color --eval-on-test")


if __name__ == "__main__":
    main()