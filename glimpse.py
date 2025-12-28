#!/usr/bin/env python3
"""
Deep Recurrent Visual Attention Model for ARC Puzzles

This model learns to solve individual ARC puzzles by:
1. Looking at the input grid through a series of learned glimpses (5x5 patches)
2. Predicting the output grid size
3. Iteratively selecting output regions and predicting their contents (3x3 regions)
4. Learning where to look via REINFORCE (no location supervision)
5. Learning what to predict via cross-entropy on cell colors

Single-puzzle training: we train a completely separate model for each puzzle
using only that puzzle's examples and their augmentations.

Usage:
    python glimpse.py --single-puzzle 00d62c1b
    python glimpse.py --single-puzzle 00d62c1b --epochs 500 --visualize
"""

import argparse
import json
import math
import os
import random
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# Determine device
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

# Grid constants
MAX_GRID_SIZE = 30
NUM_COLORS = 10
GLIMPSE_SIZE = 5
REGION_SIZE = 3


# =============================================================================
# Augmentation Utilities (from crm.py)
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
# Location Utilities
# =============================================================================

def normalize_location(x: int, y: int, grid_height: int, grid_width: int) -> Tuple[float, float]:
    """Convert cell indices (x, y) to normalized [-1, 1] coordinates.

    Args:
        x: Column index (0 to width-1)
        y: Row index (0 to height-1)
        grid_height: Height of the grid
        grid_width: Width of the grid

    Returns:
        (norm_x, norm_y) in range [-1, 1]
    """
    if grid_width > 1:
        norm_x = 2.0 * x / (grid_width - 1) - 1.0
    else:
        norm_x = 0.0
    if grid_height > 1:
        norm_y = 2.0 * y / (grid_height - 1) - 1.0
    else:
        norm_y = 0.0
    return norm_x, norm_y


def denormalize_location(norm_x: float, norm_y: float, grid_height: int, grid_width: int) -> Tuple[int, int]:
    """Convert normalized [-1, 1] coordinates to cell indices.

    Args:
        norm_x: Normalized x coordinate in [-1, 1]
        norm_y: Normalized y coordinate in [-1, 1]
        grid_height: Height of the grid
        grid_width: Width of the grid

    Returns:
        (x, y) cell indices
    """
    x = int(round((norm_x + 1.0) * (grid_width - 1) / 2.0))
    y = int(round((norm_y + 1.0) * (grid_height - 1) / 2.0))
    x = max(0, min(x, grid_width - 1))
    y = max(0, min(y, grid_height - 1))
    return x, y


def clamp_location(loc_x: float, loc_y: float, grid_height: int, grid_width: int,
                   patch_size: int) -> Tuple[float, float]:
    """Clamp location so the patch stays within grid bounds.

    Args:
        loc_x, loc_y: Normalized location in [-1, 1]
        grid_height, grid_width: Grid dimensions
        patch_size: Size of the patch (e.g., 5 for 5x5)

    Returns:
        Clamped (loc_x, loc_y) in [-1, 1]
    """
    half_patch = patch_size // 2

    # Convert to cell indices
    x, y = denormalize_location(loc_x, loc_y, grid_height, grid_width)

    # Clamp so patch stays in bounds
    x = max(half_patch, min(x, grid_width - 1 - half_patch))
    y = max(half_patch, min(y, grid_height - 1 - half_patch))

    # Convert back to normalized
    return normalize_location(x, y, grid_height, grid_width)


def extract_patch(grid: torch.Tensor, loc_x: float, loc_y: float,
                  patch_size: int = GLIMPSE_SIZE) -> torch.Tensor:
    """Extract a patch from the grid at the given normalized location.

    Args:
        grid: One-hot encoded grid tensor of shape (H, W, NUM_COLORS)
        loc_x, loc_y: Normalized location in [-1, 1]
        patch_size: Size of the patch to extract

    Returns:
        Patch tensor of shape (patch_size, patch_size, NUM_COLORS)
    """
    H, W, C = grid.shape
    half_patch = patch_size // 2

    # Denormalize location
    x, y = denormalize_location(loc_x, loc_y, H, W)

    # Clamp to ensure patch stays within bounds
    x = max(half_patch, min(x, W - 1 - half_patch))
    y = max(half_patch, min(y, H - 1 - half_patch))

    # Extract patch
    y_start = y - half_patch
    y_end = y + half_patch + 1
    x_start = x - half_patch
    x_end = x + half_patch + 1

    # Handle edge cases for small grids
    if y_end > H:
        y_start = max(0, H - patch_size)
        y_end = H
    if x_end > W:
        x_start = max(0, W - patch_size)
        x_end = W
    if y_start < 0:
        y_start = 0
        y_end = min(patch_size, H)
    if x_start < 0:
        x_start = 0
        x_end = min(patch_size, W)

    patch = grid[y_start:y_end, x_start:x_end, :]

    # Pad if necessary (for very small grids)
    if patch.shape[0] < patch_size or patch.shape[1] < patch_size:
        padded = torch.zeros(patch_size, patch_size, C, device=grid.device, dtype=grid.dtype)
        padded[:patch.shape[0], :patch.shape[1], :] = patch
        patch = padded

    return patch


def region_index_to_location(idx: int, grid_width: int, grid_height: int,
                              region_size: int = REGION_SIZE) -> Tuple[int, int]:
    """Convert region index to top-left corner coordinates.

    Regions are indexed in raster scan order (left-to-right, top-to-bottom).

    Args:
        idx: Region index
        grid_width: Width of the output grid
        grid_height: Height of the output grid
        region_size: Size of each region (3x3 by default)

    Returns:
        (x, y) top-left corner of the region
    """
    num_regions_x = math.ceil(grid_width / region_size)
    region_y = idx // num_regions_x
    region_x = idx % num_regions_x
    x = region_x * region_size
    y = region_y * region_size
    return x, y


def get_total_regions(grid_width: int, grid_height: int, region_size: int = REGION_SIZE) -> int:
    """Calculate total number of regions for a grid."""
    num_regions_x = math.ceil(grid_width / region_size)
    num_regions_y = math.ceil(grid_height / region_size)
    return num_regions_x * num_regions_y


# =============================================================================
# Neural Network Components
# =============================================================================

class ContextNetwork(nn.Module):
    """
    Processes the full input grid to provide initial state for the recurrent network.

    Input: Full input grid (H, W, 10) one-hot encoded
    Output: Initial state vector for the top LSTM layer
    """

    def __init__(self, hidden_size: int = 512):
        super().__init__()
        self.hidden_size = hidden_size

        # 2-3 convolutional layers
        self.conv1 = nn.Conv2d(NUM_COLORS, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)

        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)

        # Global average pooling is applied in forward
        # FC layer to match LSTM hidden size
        self.fc = nn.Linear(128, hidden_size)

    def forward(self, grid: torch.Tensor) -> torch.Tensor:
        """
        Args:
            grid: One-hot encoded grid of shape (B, H, W, NUM_COLORS)

        Returns:
            Initial state for top LSTM layer of shape (B, hidden_size)
        """
        # Permute to (B, C, H, W) for conv layers
        x = grid.permute(0, 3, 1, 2).contiguous()

        # Conv layers
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        # Global average pooling
        x = x.mean(dim=[2, 3])  # (B, 128)

        # FC to hidden size
        x = self.fc(x)  # (B, hidden_size)

        return x


class GlimpseNetwork(nn.Module):
    """
    Extracts features from a 5x5 patch at a specified location.

    Combines a visual stream (Gimage) processing the patch with a location
    stream (Gloc) via element-wise multiplication.
    """

    def __init__(self, feature_size: int = 256):
        super().__init__()
        self.feature_size = feature_size

        # Visual stream: 2-3 conv layers on 5x5 patch
        # Input: (patch_size, patch_size, NUM_COLORS)
        self.conv1 = nn.Conv2d(NUM_COLORS, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)

        # Flatten and FC for visual stream
        # After conv, we still have 5x5 spatial dims
        self.visual_fc = nn.Linear(64 * GLIMPSE_SIZE * GLIMPSE_SIZE, feature_size)

        # Location stream: 2 inputs -> feature_size
        self.loc_fc = nn.Linear(2, feature_size)

    def forward(self, patch: torch.Tensor, location: torch.Tensor) -> torch.Tensor:
        """
        Args:
            patch: Image patch of shape (B, patch_size, patch_size, NUM_COLORS)
            location: Normalized location (x, y) of shape (B, 2)

        Returns:
            Glimpse feature vector of shape (B, feature_size)
        """
        # Visual stream
        x = patch.permute(0, 3, 1, 2).contiguous()  # (B, C, H, W)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = x.view(x.size(0), -1)  # Flatten
        g_image = F.relu(self.visual_fc(x))  # (B, feature_size)

        # Location stream
        g_loc = F.relu(self.loc_fc(location))  # (B, feature_size)

        # Element-wise multiplication
        g = g_image * g_loc  # (B, feature_size)

        return g


class RecurrentNetwork(nn.Module):
    """
    Two-layer LSTM that accumulates information across glimpses.

    Bottom layer (r1): Takes glimpse features, initialized with zeros
    Top layer (r2): Takes bottom layer output, initialized by context network
    """

    def __init__(self, input_size: int = 256, hidden_size: int = 512):
        super().__init__()
        self.hidden_size = hidden_size

        # Bottom LSTM layer
        self.lstm1 = nn.LSTMCell(input_size, hidden_size)

        # Top LSTM layer
        self.lstm2 = nn.LSTMCell(hidden_size, hidden_size)

    def forward(self,
                glimpse: torch.Tensor,
                h1: torch.Tensor, c1: torch.Tensor,
                h2: torch.Tensor, c2: torch.Tensor
               ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            glimpse: Glimpse feature of shape (B, input_size)
            h1, c1: Hidden and cell state for bottom LSTM
            h2, c2: Hidden and cell state for top LSTM

        Returns:
            Updated (h1, c1, h2, c2)
        """
        h1, c1 = self.lstm1(glimpse, (h1, c1))
        h2, c2 = self.lstm2(h1, (h2, c2))
        return h1, c1, h2, c2

    def init_hidden(self, batch_size: int, context: Optional[torch.Tensor] = None,
                    device: torch.device = None) -> Tuple[torch.Tensor, torch.Tensor,
                                                           torch.Tensor, torch.Tensor]:
        """
        Initialize hidden states.

        Args:
            batch_size: Batch size
            context: Optional context vector to initialize top layer (from ContextNetwork)
            device: Device to create tensors on

        Returns:
            (h1, c1, h2, c2) initial states
        """
        if device is None:
            device = torch.device('cpu')

        # Bottom layer initialized with zeros
        h1 = torch.zeros(batch_size, self.hidden_size, device=device)
        c1 = torch.zeros(batch_size, self.hidden_size, device=device)

        # Top layer initialized with context (or zeros if not provided)
        if context is not None:
            h2 = context
        else:
            h2 = torch.zeros(batch_size, self.hidden_size, device=device)
        c2 = torch.zeros(batch_size, self.hidden_size, device=device)

        return h1, c1, h2, c2


class EmissionNetwork(nn.Module):
    """
    Predicts where to look next (glimpse locations) and where to write next
    (output region locations).

    For glimpse locations: Output 2 values (x, y) in normalized [-1, 1]
    For region locations: Output (x, y, finished_logit)
    """

    def __init__(self, hidden_size: int = 512, emission_hidden: int = 256):
        super().__init__()

        # Glimpse location prediction
        self.glimpse_fc1 = nn.Linear(hidden_size, emission_hidden)
        self.glimpse_fc2 = nn.Linear(emission_hidden, 2)  # (x, y)

        # Region location prediction (with finished signal)
        self.region_fc1 = nn.Linear(hidden_size, emission_hidden)
        self.region_fc2 = nn.Linear(emission_hidden, 3)  # (x, y, finished_logit)

    def predict_glimpse_location(self, h2: torch.Tensor) -> torch.Tensor:
        """
        Predict next glimpse location from top LSTM state.

        Args:
            h2: Top LSTM hidden state of shape (B, hidden_size)

        Returns:
            Location (x, y) in [-1, 1] of shape (B, 2)
        """
        x = F.relu(self.glimpse_fc1(h2))
        loc = torch.tanh(self.glimpse_fc2(x))  # Constrain to [-1, 1]
        return loc

    def predict_region_location(self, h2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict next region location and finished signal.

        Args:
            h2: Top LSTM hidden state of shape (B, hidden_size)

        Returns:
            location: (x, y) in [-1, 1] of shape (B, 2)
            finished_logit: Finished signal logit of shape (B, 1)
        """
        x = F.relu(self.region_fc1(h2))
        out = self.region_fc2(x)  # (B, 3)
        location = torch.tanh(out[:, :2])  # (x, y) in [-1, 1]
        finished_logit = out[:, 2:3]  # (B, 1)
        return location, finished_logit


class ClassificationNetworks(nn.Module):
    """
    Classification heads fed by the bottom LSTM state (r1):
    - Grid height classifier (30 classes)
    - Grid width classifier (30 classes)
    - Cell color classifier (9 cells, 10 classes each)
    """

    def __init__(self, hidden_size: int = 512, classifier_hidden: int = 256):
        super().__init__()

        # Grid height classifier
        self.height_fc1 = nn.Linear(hidden_size, classifier_hidden)
        self.height_fc2 = nn.Linear(classifier_hidden, MAX_GRID_SIZE)

        # Grid width classifier
        self.width_fc1 = nn.Linear(hidden_size, classifier_hidden)
        self.width_fc2 = nn.Linear(classifier_hidden, MAX_GRID_SIZE)

        # Cell color classifier: predicts 9 cells, each with 10 possible colors
        # Output shape will be (B, 9, 10)
        self.cell_fc1 = nn.Linear(hidden_size, classifier_hidden)
        self.cell_fc2 = nn.Linear(classifier_hidden, REGION_SIZE * REGION_SIZE * NUM_COLORS)

    def predict_grid_size(self, h1: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict output grid size.

        Args:
            h1: Bottom LSTM hidden state of shape (B, hidden_size)

        Returns:
            height_logits: (B, MAX_GRID_SIZE)
            width_logits: (B, MAX_GRID_SIZE)
        """
        height_logits = self.height_fc2(F.relu(self.height_fc1(h1)))
        width_logits = self.width_fc2(F.relu(self.width_fc1(h1)))
        return height_logits, width_logits

    def predict_cells(self, h1: torch.Tensor) -> torch.Tensor:
        """
        Predict 9 cell colors for a 3x3 region.

        Args:
            h1: Bottom LSTM hidden state of shape (B, hidden_size)

        Returns:
            cell_logits: (B, 9, 10) - 9 cells, 10 classes each
        """
        x = F.relu(self.cell_fc1(h1))
        x = self.cell_fc2(x)  # (B, 90)
        x = x.view(-1, REGION_SIZE * REGION_SIZE, NUM_COLORS)  # (B, 9, 10)
        return x


class BaselineNetwork(nn.Module):
    """
    Learned baseline for REINFORCE variance reduction.
    Predicts expected reward from top LSTM state.
    """

    def __init__(self, hidden_size: int = 512, baseline_hidden: int = 128):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, baseline_hidden)
        self.fc2 = nn.Linear(baseline_hidden, 1)

    def forward(self, h2: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h2: Top LSTM hidden state of shape (B, hidden_size)

        Returns:
            Baseline prediction of shape (B, 1)
        """
        x = F.relu(self.fc1(h2))
        return self.fc2(x)


# =============================================================================
# Full Model
# =============================================================================

class GlimpseModel(nn.Module):
    """
    Complete Deep Recurrent Visual Attention Model for ARC puzzles.

    Combines all components:
    - Context Network
    - Glimpse Network
    - Recurrent Network (2-layer LSTM)
    - Emission Network
    - Classification Networks
    - Baseline Network
    """

    def __init__(self,
                 hidden_size: int = 512,
                 glimpse_feature_size: int = 256,
                 num_glimpses: int = 5,
                 location_std: float = 0.05):
        super().__init__()

        self.hidden_size = hidden_size
        self.glimpse_feature_size = glimpse_feature_size
        self.num_glimpses = num_glimpses
        self.location_std = location_std

        # Components
        self.context_net = ContextNetwork(hidden_size)
        self.glimpse_net = GlimpseNetwork(glimpse_feature_size)
        self.recurrent_net = RecurrentNetwork(glimpse_feature_size, hidden_size)
        self.emission_net = EmissionNetwork(hidden_size)
        self.classification_net = ClassificationNetworks(hidden_size)
        self.baseline_net = BaselineNetwork(hidden_size)

    def _one_hot_encode(self, grid: torch.Tensor) -> torch.Tensor:
        """One-hot encode a grid.

        Args:
            grid: Grid of shape (B, H, W) with values 0-9

        Returns:
            One-hot encoded tensor of shape (B, H, W, 10)
        """
        return F.one_hot(grid.long(), num_classes=NUM_COLORS).float()

    def _sample_location(self, mean_loc: torch.Tensor,
                         training: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample location from Gaussian centered on predicted mean.

        Args:
            mean_loc: Predicted mean location of shape (B, 2)
            training: Whether in training mode (sample) or inference (use mean)

        Returns:
            sampled_loc: Sampled/deterministic location of shape (B, 2)
            log_prob: Log probability of sampled location of shape (B,)
        """
        if training:
            # Sample from Gaussian
            dist = Normal(mean_loc, torch.full_like(mean_loc, self.location_std))
            sampled_loc = dist.sample()
            log_prob = dist.log_prob(sampled_loc).sum(dim=-1)  # Sum over x, y
            # Clamp to [-1, 1]
            sampled_loc = torch.clamp(sampled_loc, -1.0, 1.0)
        else:
            # Use mean directly (deterministic)
            sampled_loc = mean_loc
            log_prob = torch.zeros(mean_loc.size(0), device=mean_loc.device)

        return sampled_loc, log_prob

    def forward_grid_size(self,
                          input_grid: torch.Tensor,
                          training: bool = True
                         ) -> Dict[str, torch.Tensor]:
        """
        Stage 1: Predict output grid size.

        Args:
            input_grid: Input grid of shape (B, H, W) with values 0-9
            training: Whether in training mode

        Returns:
            Dictionary containing:
                - height_logits: (B, 30)
                - width_logits: (B, 30)
                - h1, c1, h2, c2: LSTM states for next stage
                - log_probs: Sum of log probabilities for glimpse locations
                - baselines: Baseline predictions (B, 1)
        """
        B = input_grid.size(0)
        H, W = input_grid.size(1), input_grid.size(2)
        device = input_grid.device

        # One-hot encode input
        input_onehot = self._one_hot_encode(input_grid)  # (B, H, W, 10)

        # Initialize with context
        context = self.context_net(input_onehot)  # (B, hidden_size)
        h1, c1, h2, c2 = self.recurrent_net.init_hidden(B, context, device)

        # Take glimpses
        total_log_prob = torch.zeros(B, device=device)
        baseline_sum = torch.zeros(B, 1, device=device)

        for _ in range(self.num_glimpses):
            # Predict glimpse location from top LSTM
            mean_loc = self.emission_net.predict_glimpse_location(h2)
            sampled_loc, log_prob = self._sample_location(mean_loc, training)
            total_log_prob = total_log_prob + log_prob

            # Accumulate baseline predictions
            baseline = self.baseline_net(h2)
            baseline_sum = baseline_sum + baseline

            # Extract patches for each item in batch
            patches = []
            for b in range(B):
                patch = extract_patch(input_onehot[b],
                                      sampled_loc[b, 0].item(),
                                      sampled_loc[b, 1].item())
                patches.append(patch)
            patches = torch.stack(patches)  # (B, 5, 5, 10)

            # Get glimpse features
            glimpse = self.glimpse_net(patches, sampled_loc)

            # Update LSTM
            h1, c1, h2, c2 = self.recurrent_net(glimpse, h1, c1, h2, c2)

        # Predict grid size from bottom LSTM
        height_logits, width_logits = self.classification_net.predict_grid_size(h1)

        return {
            'height_logits': height_logits,
            'width_logits': width_logits,
            'h1': h1, 'c1': c1, 'h2': h2, 'c2': c2,
            'log_probs': total_log_prob,
            'baselines': baseline_sum / self.num_glimpses,
        }

    def forward_region(self,
                       input_grid: torch.Tensor,
                       h1: torch.Tensor, c1: torch.Tensor,
                       h2: torch.Tensor, c2: torch.Tensor,
                       training: bool = True
                      ) -> Dict[str, torch.Tensor]:
        """
        One iteration of Stage 2: Predict one region of the output.

        Args:
            input_grid: Input grid of shape (B, H, W)
            h1, c1, h2, c2: LSTM states from previous stage/iteration
            training: Whether in training mode

        Returns:
            Dictionary containing:
                - cell_logits: (B, 9, 10) color predictions for 3x3 region
                - region_loc: (B, 2) normalized region location
                - finished_logit: (B, 1) finished signal
                - h1, c1, h2, c2: Updated LSTM states
                - log_probs: Log probabilities for locations
                - baselines: Baseline predictions
        """
        B = input_grid.size(0)
        device = input_grid.device

        # One-hot encode input
        input_onehot = self._one_hot_encode(input_grid)

        # Take glimpses
        total_log_prob = torch.zeros(B, device=device)
        baseline_sum = torch.zeros(B, 1, device=device)

        for _ in range(self.num_glimpses):
            # Predict glimpse location
            mean_loc = self.emission_net.predict_glimpse_location(h2)
            sampled_loc, log_prob = self._sample_location(mean_loc, training)
            total_log_prob = total_log_prob + log_prob

            # Accumulate baseline
            baseline = self.baseline_net(h2)
            baseline_sum = baseline_sum + baseline

            # Extract patches
            patches = []
            for b in range(B):
                patch = extract_patch(input_onehot[b],
                                      sampled_loc[b, 0].item(),
                                      sampled_loc[b, 1].item())
                patches.append(patch)
            patches = torch.stack(patches)

            # Get glimpse features
            glimpse = self.glimpse_net(patches, sampled_loc)

            # Update LSTM
            h1, c1, h2, c2 = self.recurrent_net(glimpse, h1, c1, h2, c2)

        # Predict region location and finished signal
        region_mean, finished_logit = self.emission_net.predict_region_location(h2)
        region_loc, region_log_prob = self._sample_location(region_mean, training)
        total_log_prob = total_log_prob + region_log_prob

        # Predict cell colors from bottom LSTM
        cell_logits = self.classification_net.predict_cells(h1)

        return {
            'cell_logits': cell_logits,
            'region_loc': region_loc,
            'finished_logit': finished_logit,
            'h1': h1, 'c1': c1, 'h2': h2, 'c2': c2,
            'log_probs': total_log_prob,
            'baselines': baseline_sum / self.num_glimpses,
        }

    def inference(self, input_grid: torch.Tensor, max_regions: int = 100) -> torch.Tensor:
        """
        Full inference procedure: predict output grid for a single input.

        Args:
            input_grid: Input grid of shape (H, W) with values 0-9
            max_regions: Maximum number of regions to predict

        Returns:
            Predicted output grid of shape (pred_H, pred_W)
        """
        self.eval()
        device = next(self.parameters()).device

        with torch.no_grad():
            # Add batch dimension
            input_grid = input_grid.unsqueeze(0).to(device)  # (1, H, W)

            # Stage 1: Predict grid size
            result = self.forward_grid_size(input_grid, training=False)

            pred_height = result['height_logits'].argmax(dim=-1).item() + 1  # 1-indexed
            pred_width = result['width_logits'].argmax(dim=-1).item() + 1

            h1, c1, h2, c2 = result['h1'], result['c1'], result['h2'], result['c2']

            # Create output grid
            output_grid = torch.zeros(pred_height, pred_width, dtype=torch.long, device=device)

            # Stage 2: Predict regions
            total_regions = get_total_regions(pred_width, pred_height)
            regions_filled = 0

            for region_idx in range(min(max_regions, total_regions + 1)):
                result = self.forward_region(input_grid, h1, c1, h2, c2, training=False)

                h1, c1, h2, c2 = result['h1'], result['c1'], result['h2'], result['c2']
                finished = torch.sigmoid(result['finished_logit']).item() > 0.5

                if finished:
                    break

                # Get region location
                region_loc = result['region_loc'][0]  # (2,)
                rx, ry = denormalize_location(region_loc[0].item(), region_loc[1].item(),
                                              pred_height, pred_width)

                # Align to region grid
                rx = (rx // REGION_SIZE) * REGION_SIZE
                ry = (ry // REGION_SIZE) * REGION_SIZE

                # Get predicted colors
                cell_preds = result['cell_logits'].argmax(dim=-1)[0]  # (9,)

                # Fill in region
                for i in range(REGION_SIZE):
                    for j in range(REGION_SIZE):
                        py = ry + i
                        px = rx + j
                        if py < pred_height and px < pred_width:
                            output_grid[py, px] = cell_preds[i * REGION_SIZE + j]

                regions_filled += 1

            return output_grid


# =============================================================================
# Dataset
# =============================================================================

class AugmentedPuzzleDataset(Dataset):
    """
    Dataset for a single puzzle with augmentations.

    Generates many augmented versions of the training examples via:
    - Color permutations (keeping 0 fixed)
    - Dihedral transformations (rotations, reflections)
    """

    def __init__(self,
                 puzzle: Dict,
                 num_augmentations: int = 100,
                 include_test: bool = False):
        """
        Args:
            puzzle: Puzzle dictionary with 'train' and optionally 'test' examples
            num_augmentations: Number of augmented versions to generate
            include_test: Whether to include test examples in training
        """
        self.examples = []

        # Collect base examples
        for ex in puzzle.get('train', []):
            inp = np.array(ex['input'], dtype=np.uint8)
            out = np.array(ex['output'], dtype=np.uint8)
            self.examples.append((inp, out))

        if include_test:
            for ex in puzzle.get('test', []):
                if 'output' in ex:
                    inp = np.array(ex['input'], dtype=np.uint8)
                    out = np.array(ex['output'], dtype=np.uint8)
                    self.examples.append((inp, out))

        self.num_augmentations = num_augmentations

    def __len__(self):
        return len(self.examples) * self.num_augmentations

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        example_idx = idx // self.num_augmentations
        inp, out = self.examples[example_idx]

        # Apply random augmentation
        trans_id, color_map = get_random_augmentation()
        aug_inp = apply_augmentation(inp, trans_id, color_map)
        aug_out = apply_augmentation(out, trans_id, color_map)

        return (
            torch.from_numpy(aug_inp.copy()).long(),
            torch.from_numpy(aug_out.copy()).long(),
        )


class TestEvalDataset(Dataset):
    """Dataset for evaluating on held-out test examples (no augmentation)."""

    def __init__(self, puzzle: Dict):
        self.examples = []
        for ex in puzzle.get('test', []):
            if 'output' in ex:
                inp = np.array(ex['input'], dtype=np.uint8)
                out = np.array(ex['output'], dtype=np.uint8)
                self.examples.append((inp, out))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        inp, out = self.examples[idx]
        return (
            torch.from_numpy(inp.copy()).long(),
            torch.from_numpy(out.copy()).long(),
        )


# =============================================================================
# Training
# =============================================================================

def compute_region_target(output_grid: torch.Tensor,
                          region_x: int, region_y: int) -> torch.Tensor:
    """
    Extract target colors for a 3x3 region.

    Args:
        output_grid: Output grid of shape (H, W)
        region_x, region_y: Top-left corner of the region

    Returns:
        Target colors of shape (9,) in raster scan order
    """
    H, W = output_grid.shape
    target = torch.zeros(REGION_SIZE * REGION_SIZE, dtype=torch.long, device=output_grid.device)

    for i in range(REGION_SIZE):
        for j in range(REGION_SIZE):
            py = region_y + i
            px = region_x + j
            if py < H and px < W:
                target[i * REGION_SIZE + j] = output_grid[py, px]

    return target


def train_step(model: GlimpseModel,
               input_grid: torch.Tensor,
               output_grid: torch.Tensor,
               optimizer: torch.optim.Optimizer,
               reinforce_weight: float = 1.0,
               device: torch.device = None) -> Dict[str, float]:
    """
    Single training step for one example.

    Implements the full training procedure with truncation at first mistake.

    Args:
        model: The GlimpseModel
        input_grid: Input grid of shape (H_in, W_in)
        output_grid: Target output grid of shape (H_out, W_out)
        optimizer: Optimizer
        reinforce_weight: Weight for REINFORCE loss
        device: Device

    Returns:
        Dictionary of metrics
    """
    model.train()

    if device is None:
        device = next(model.parameters()).device

    input_grid = input_grid.unsqueeze(0).to(device)  # (1, H, W)
    output_grid = output_grid.to(device)

    true_height = output_grid.size(0)
    true_width = output_grid.size(1)

    optimizer.zero_grad()

    total_loss = 0.0
    total_reinforce_loss = 0.0
    total_baseline_loss = 0.0

    # Stage 1: Grid size prediction
    result = model.forward_grid_size(input_grid, training=True)

    height_logits = result['height_logits']
    width_logits = result['width_logits']

    # Classification loss for grid size
    # Convert to 0-indexed for targets
    height_target = torch.tensor([true_height - 1], device=device)
    width_target = torch.tensor([true_width - 1], device=device)

    height_loss = F.cross_entropy(height_logits, height_target)
    width_loss = F.cross_entropy(width_logits, width_target)
    size_loss = height_loss + width_loss
    total_loss = total_loss + size_loss

    # Check if grid size is correct
    pred_height = height_logits.argmax(dim=-1).item() + 1
    pred_width = width_logits.argmax(dim=-1).item() + 1
    size_correct = (pred_height == true_height and pred_width == true_width)

    # REINFORCE for grid size stage
    reward = 1.0 if size_correct else 0.0
    baseline = result['baselines']
    advantage = reward - baseline.detach()
    reinforce_loss = -advantage * result['log_probs']
    total_reinforce_loss = total_reinforce_loss + reinforce_loss.mean()

    # Baseline MSE loss
    baseline_target = torch.tensor([[reward]], device=device)
    baseline_loss = F.mse_loss(baseline, baseline_target)
    total_baseline_loss = total_baseline_loss + baseline_loss

    # Truncate at first mistake: if size wrong, don't train region stages
    if not size_correct:
        full_loss = total_loss + reinforce_weight * total_reinforce_loss + total_baseline_loss
        full_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        return {
            'loss': total_loss.item(),
            'reinforce_loss': total_reinforce_loss.item(),
            'baseline_loss': total_baseline_loss.item(),
            'size_correct': 0.0,
            'regions_correct': 0.0,
            'total_regions': get_total_regions(true_width, true_height),
        }

    # Stage 2: Region prediction
    h1, c1, h2, c2 = result['h1'], result['c1'], result['h2'], result['c2']

    total_regions = get_total_regions(true_width, true_height)
    regions_correct = 0

    for region_idx in range(total_regions):
        # Get target region location
        target_rx, target_ry = region_index_to_location(region_idx, true_width, true_height)

        # Forward pass for this region
        result = model.forward_region(input_grid, h1, c1, h2, c2, training=True)

        h1, c1, h2, c2 = result['h1'], result['c1'], result['h2'], result['c2']

        # Classification loss for cells
        cell_logits = result['cell_logits']  # (1, 9, 10)
        cell_targets = compute_region_target(output_grid, target_rx, target_ry)  # (9,)
        cell_loss = F.cross_entropy(cell_logits[0], cell_targets)
        total_loss = total_loss + cell_loss

        # Check if region is correct
        pred_cells = cell_logits.argmax(dim=-1)[0]  # (9,)
        region_correct = (pred_cells == cell_targets).all().item()

        if region_correct:
            regions_correct += 1

        # REINFORCE for this stage
        reward = float(regions_correct)  # Cumulative count
        baseline = result['baselines']
        advantage = reward - baseline.detach()
        reinforce_loss = -advantage * result['log_probs']
        total_reinforce_loss = total_reinforce_loss + reinforce_loss.mean()

        # Baseline loss
        baseline_target = torch.tensor([[reward]], device=device)
        baseline_loss = F.mse_loss(baseline, baseline_target)
        total_baseline_loss = total_baseline_loss + baseline_loss

        # Truncate at first region mistake
        if not region_correct:
            break

    # Final loss and backprop
    full_loss = total_loss + reinforce_weight * total_reinforce_loss + total_baseline_loss
    full_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
    optimizer.step()

    return {
        'loss': total_loss.item(),
        'reinforce_loss': total_reinforce_loss.item(),
        'baseline_loss': total_baseline_loss.item(),
        'size_correct': 1.0,
        'regions_correct': float(regions_correct),
        'total_regions': float(total_regions),
    }


def train_epoch(model: GlimpseModel,
                dataset: Dataset,
                optimizer: torch.optim.Optimizer,
                reinforce_weight: float = 1.0,
                device: torch.device = None) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()

    total_loss = 0.0
    total_reinforce_loss = 0.0
    total_baseline_loss = 0.0
    size_correct_count = 0
    regions_correct_total = 0
    total_regions_total = 0

    pbar = tqdm(range(len(dataset)), desc="Training")
    for idx in pbar:
        input_grid, output_grid = dataset[idx]

        metrics = train_step(model, input_grid, output_grid, optimizer,
                            reinforce_weight, device)

        total_loss += metrics['loss']
        total_reinforce_loss += metrics['reinforce_loss']
        total_baseline_loss += metrics['baseline_loss']
        size_correct_count += metrics['size_correct']
        regions_correct_total += metrics['regions_correct']
        total_regions_total += metrics['total_regions']

        pbar.set_postfix(
            loss=metrics['loss'],
            size_acc=size_correct_count / (idx + 1),
        )

    n = len(dataset)
    return {
        'loss': total_loss / n,
        'reinforce_loss': total_reinforce_loss / n,
        'baseline_loss': total_baseline_loss / n,
        'size_accuracy': size_correct_count / n,
        'region_accuracy': regions_correct_total / max(total_regions_total, 1),
    }


def evaluate(model: GlimpseModel,
             dataset: Dataset,
             device: torch.device = None,
             verbose: bool = False) -> Dict[str, float]:
    """
    Evaluate model on a dataset using full inference.
    """
    model.eval()

    if device is None:
        device = next(model.parameters()).device

    size_correct = 0
    pixel_correct = 0
    pixel_total = 0
    perfect_count = 0

    for idx in range(len(dataset)):
        input_grid, output_grid = dataset[idx]

        pred_grid = model.inference(input_grid)

        true_height, true_width = output_grid.shape
        pred_height, pred_width = pred_grid.shape

        # Check size
        if pred_height == true_height and pred_width == true_width:
            size_correct += 1

            # Check pixels
            pred_grid_cpu = pred_grid.cpu()
            correct = (pred_grid_cpu == output_grid).sum().item()
            total = true_height * true_width
            pixel_correct += correct
            pixel_total += total

            if correct == total:
                perfect_count += 1

            if verbose:
                status = "PERFECT" if correct == total else f"{correct}/{total}"
                print(f"  Example {idx}: {status}")
        else:
            if verbose:
                print(f"  Example {idx}: Size mismatch ({pred_height}x{pred_width} vs {true_height}x{true_width})")

    return {
        'size_accuracy': size_correct / len(dataset),
        'pixel_accuracy': pixel_correct / max(pixel_total, 1),
        'perfect_rate': perfect_count / len(dataset),
    }


# =============================================================================
# Visualization
# =============================================================================

def visualize_prediction(input_grid: np.ndarray,
                         target_grid: np.ndarray,
                         pred_grid: np.ndarray,
                         puzzle_id: str = ""):
    """Visualize input, target, and prediction."""
    # Color codes for terminal
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

    inp_h, inp_w = input_grid.shape
    tgt_h, tgt_w = target_grid.shape
    pred_h, pred_w = pred_grid.shape

    # Calculate errors
    if pred_h == tgt_h and pred_w == tgt_w:
        errors = (pred_grid != target_grid).sum()
        total = tgt_h * tgt_w
    else:
        errors = -1
        total = tgt_h * tgt_w

    print(f"\n{'='*50}")
    if errors == 0:
        print(f"\033[92mPERFECT: {puzzle_id}\033[0m")
    elif errors == -1:
        print(f"\033[91mSIZE MISMATCH: {puzzle_id} (pred {pred_h}x{pred_w} vs target {tgt_h}x{tgt_w})\033[0m")
    else:
        print(f"\033[91m{puzzle_id}: {errors}/{total} errors ({100*errors/total:.1f}% wrong)\033[0m")
    print(f"  Input: {inp_h}x{inp_w} -> Output: {tgt_h}x{tgt_w}")
    print(f"{'='*50}")

    def print_grid(grid, label, highlight_errors=False, target=None):
        h, w = grid.shape
        print(f"\n{label}:")
        for r in range(h):
            row_str = "  "
            for c in range(w):
                val = grid[r, c]
                if highlight_errors and target is not None and r < target.shape[0] and c < target.shape[1]:
                    if grid[r, c] != target[r, c]:
                        row_str += f"\033[41m\033[97m{val}\033[0m "
                    else:
                        row_str += f"{COLORS[val]}{val}{RESET} "
                else:
                    row_str += f"{COLORS[val]}{val}{RESET} "
            print(row_str)

    print_grid(input_grid, "INPUT")
    print_grid(target_grid, "TARGET")
    if pred_h == tgt_h and pred_w == tgt_w:
        print_grid(pred_grid, "PREDICTION (errors in red)", highlight_errors=True, target=target_grid)
    else:
        print_grid(pred_grid, "PREDICTION (size mismatch)")
    print()


def evaluate_with_visualization(model: GlimpseModel,
                                 dataset: Dataset,
                                 device: torch.device = None):
    """Evaluate and visualize all predictions."""
    model.eval()

    if device is None:
        device = next(model.parameters()).device

    for idx in range(len(dataset)):
        input_grid, output_grid = dataset[idx]

        pred_grid = model.inference(input_grid)

        visualize_prediction(
            input_grid.numpy(),
            output_grid.numpy(),
            pred_grid.cpu().numpy(),
            f"Example {idx}"
        )


# =============================================================================
# Puzzle Loading
# =============================================================================

def load_puzzles(dataset_name: str, data_root: str = "kaggle/combined") -> Dict:
    """Load puzzles from JSON files."""
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


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Deep Recurrent Visual Attention Model for ARC Puzzles"
    )

    # Data arguments
    parser.add_argument("--dataset", type=str, default="arc-agi-1",
                        choices=["arc-agi-1", "arc-agi-2"])
    parser.add_argument("--data-root", type=str, default="kaggle/combined")
    parser.add_argument("--single-puzzle", type=str, required=True,
                        help="Puzzle ID to train on (required)")

    # Model arguments
    parser.add_argument("--hidden-size", type=int, default=512,
                        help="LSTM hidden size")
    parser.add_argument("--glimpse-size", type=int, default=256,
                        help="Glimpse feature size")
    parser.add_argument("--num-glimpses", type=int, default=5,
                        help="Number of glimpses per stage")
    parser.add_argument("--location-std", type=float, default=0.05,
                        help="Standard deviation for location sampling")

    # Training arguments
    parser.add_argument("--epochs", type=int, default=200,
                        help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate")
    parser.add_argument("--reinforce-weight", type=float, default=1.0,
                        help="Weight for REINFORCE loss")
    parser.add_argument("--num-augmentations", type=int, default=100,
                        help="Number of augmentations per example")
    parser.add_argument("--seed", type=int, default=42)

    # Output arguments
    parser.add_argument("--save-path", type=str, default="checkpoints/glimpse.pt")
    parser.add_argument("--visualize", action="store_true",
                        help="Visualize predictions on test examples")
    parser.add_argument("--eval-on-test", action="store_true",
                        help="Evaluate on held-out test examples")

    args = parser.parse_args()

    # Set seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    print(f"Device: {DEVICE}")
    print(f"Dataset: {args.dataset}")
    print(f"Puzzle: {args.single_puzzle}")

    # Load puzzles
    print("\nLoading puzzles...")
    puzzles = load_puzzles(args.dataset, args.data_root)
    print(f"Loaded {len(puzzles)} puzzles")

    if args.single_puzzle not in puzzles:
        print(f"Error: Puzzle '{args.single_puzzle}' not found!")
        return

    puzzle = puzzles[args.single_puzzle]

    # Print puzzle info
    n_train = len(puzzle.get('train', []))
    n_test = sum(1 for ex in puzzle.get('test', []) if 'output' in ex)
    print(f"Training examples: {n_train}")
    print(f"Test examples (with outputs): {n_test}")

    # Create datasets
    train_dataset = AugmentedPuzzleDataset(
        puzzle,
        num_augmentations=args.num_augmentations,
        include_test=not args.eval_on_test
    )

    test_dataset = None
    if args.eval_on_test and n_test > 0:
        test_dataset = TestEvalDataset(puzzle)

    print(f"Training samples (with augmentation): {len(train_dataset)}")

    # Create model
    print("\nCreating model...")
    model = GlimpseModel(
        hidden_size=args.hidden_size,
        glimpse_feature_size=args.glimpse_size,
        num_glimpses=args.num_glimpses,
        location_std=args.location_std,
    ).to(DEVICE)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)

    # Create checkpoint directory
    os.makedirs(os.path.dirname(args.save_path) or ".", exist_ok=True)

    # Training loop
    print("\n" + "="*60)
    print("Starting Training")
    print("="*60)

    best_size_acc = 0.0
    best_region_acc = 0.0

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")

        metrics = train_epoch(model, train_dataset, optimizer,
                             args.reinforce_weight, DEVICE)

        print(f"  Loss: {metrics['loss']:.4f}")
        print(f"  REINFORCE Loss: {metrics['reinforce_loss']:.4f}")
        print(f"  Baseline Loss: {metrics['baseline_loss']:.4f}")
        print(f"  Size Accuracy: {metrics['size_accuracy']:.2%}")
        print(f"  Region Accuracy: {metrics['region_accuracy']:.2%}")

        scheduler.step()

        # Save if improved
        if metrics['size_accuracy'] > best_size_acc or \
           (metrics['size_accuracy'] == best_size_acc and metrics['region_accuracy'] > best_region_acc):
            best_size_acc = metrics['size_accuracy']
            best_region_acc = metrics['region_accuracy']
            print(f"  [New best: size={best_size_acc:.2%}, region={best_region_acc:.2%}]")

            checkpoint = {
                'model_state_dict': model.state_dict(),
                'args': vars(args),
                'epoch': epoch,
                'best_size_accuracy': best_size_acc,
                'best_region_accuracy': best_region_acc,
            }
            torch.save(checkpoint, args.save_path)

    # Final summary
    print("\n" + "="*60)
    print("Training Complete")
    print("="*60)
    print(f"Best Size Accuracy: {best_size_acc:.2%}")
    print(f"Best Region Accuracy: {best_region_acc:.2%}")
    print(f"Checkpoint saved to: {args.save_path}")

    # Evaluate on test if requested
    if args.eval_on_test and test_dataset is not None:
        print("\n" + "="*60)
        print("Evaluating on held-out test examples")
        print("="*60)

        # Load best model
        checkpoint = torch.load(args.save_path, map_location=DEVICE, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])

        test_metrics = evaluate(model, test_dataset, DEVICE, verbose=True)
        print(f"\nTest Results:")
        print(f"  Size Accuracy: {test_metrics['size_accuracy']:.2%}")
        print(f"  Pixel Accuracy: {test_metrics['pixel_accuracy']:.2%}")
        print(f"  Perfect Rate: {test_metrics['perfect_rate']:.2%}")

        if args.visualize:
            evaluate_with_visualization(model, test_dataset, DEVICE)

    elif args.visualize:
        # Visualize on training examples
        print("\n" + "="*60)
        print("Visualizing predictions on training examples")
        print("="*60)

        # Create a small dataset with just the base examples (no augmentation)
        viz_dataset = AugmentedPuzzleDataset(puzzle, num_augmentations=1, include_test=True)

        # Only show first few
        class SubsetDataset:
            def __init__(self, dataset, n):
                self.dataset = dataset
                self.n = min(n, len(dataset))
            def __len__(self):
                return self.n
            def __getitem__(self, idx):
                return self.dataset[idx]

        evaluate_with_visualization(model, SubsetDataset(viz_dataset, n_train + n_test), DEVICE)


if __name__ == "__main__":
    main()
