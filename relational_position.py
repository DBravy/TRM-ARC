#!/usr/bin/env python3
"""
Relational Position Prediction for ARC

Tests the hypothesis that object position transformations can be predicted
by attending to pairwise relations between objects.

For each pair of objects (i, j), we compute:
- Relative position (centroid_j - centroid_i)
- Relative size (area ratio)
- Same color
- Adjacent (masks touch)
- Row/col aligned

Then we use relation-conditioned attention to predict position deltas.

Usage:
    python relational_position.py --puzzle-id 03560426 --epochs 1000
    python relational_position.py --puzzle-id 03560426 --epochs 1000 --num-augmentations 100
    python relational_position.py --puzzle-id 03560426 --epochs 1000 --object-by-color
    python relational_position.py --puzzle-id 03560426 --epochs 1000 --ordering-strategy left_to_right
    python relational_position.py --puzzle-id 03560426 --epochs 1000 --screen-ordering

Ordering strategies (--ordering-strategy or --screen-ordering):
    Objects can be processed in a specific order to enable sequential
    placement rules. Available strategies: left_to_right, right_to_left,
    top_to_bottom, bottom_to_top, diagonal_tl_br, diagonal_tr_bl,
    largest_first, smallest_first, by_color, adaptive_reading_order,
    quadrant_order.

    - adaptive_reading_order: Clusters objects into "rows" based on vertical
      gaps, orders rows top-to-bottom, and within each row orders left-to-right.
      Finds natural reading structure regardless of object count.

    - quadrant_order: Divides space into quadrants using median object position
      as center, assigns objects to quadrants (TL, TR, BL, BR), and orders in
      reading order. Useful when objects should be grouped by spatial region.

    Use --screen-ordering to auto-discover the best ordering strategy
    for the puzzle based on framing consistency across training examples.
"""

import argparse
import json
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# Object detection and extraction
from object_module import (
    NUM_COLORS,
    MAX_OBJECTS,
    extract_connected_components,
    extract_objects_from_grid,
    labels_to_objects,
    sort_objects_left_to_right,
    sort_objects_by_strategy,
    compute_object_properties,
)

# Anchor point discovery for object-relative positioning
from anchoring_module import (
    AnchorPoint, ALL_ANCHORS,
    get_anchor_position, get_anchor_offset,
    SpatialRelation, DiscoveredRelation,
    TestObject, ExamplePair,
    discover_relation, compute_position_from_relation
)

# Ordering strategies for object processing order
from ordering_module import (
    OrderingStrategy,
    ALL_ORDERINGS,
    OrderingEvaluator,
    screen_orderings_for_puzzle
)

# Object correspondence matching
from correspondence_module import find_object_correspondences

# Selection module for ranking-based object selection
from selection_module import (
    RANKING_CRITERIA, SELECTION_RULES,
    compute_selection_mask,
    SelectionScreener
)

# Determine device
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

GRID_SIZE = 30

# Per-object anchor framing options (from anchor_validation.py)
FRAMINGS = ['delta', 'grid_tl', 'grid_tr', 'grid_bl', 'grid_br', 'object_relative']
NUM_FRAMINGS = 6


# =============================================================================
# Augmentation
# =============================================================================

def apply_dihedral_transform(grid: np.ndarray, transform_id: int) -> np.ndarray:
    """
    Apply one of 8 dihedral transforms to a grid.

    transform_id:
        0: identity
        1: rotate 90° CW
        2: rotate 180°
        3: rotate 270° CW
        4: flip horizontal
        5: flip vertical
    """
    if transform_id == 0:
        return grid.copy()
    elif transform_id == 1:
        return np.rot90(grid, k=-1)  # 90° CW
    elif transform_id == 2:
        return np.rot90(grid, k=2)   # 180°
    elif transform_id == 3:
        return np.rot90(grid, k=1)   # 270° CW (= 90° CCW)
    elif transform_id == 4:
        return np.fliplr(grid)       # horizontal flip
    elif transform_id == 5:
        return np.flipud(grid)       # vertical flip
    elif transform_id == 6:
        return grid.T                # transpose (main diagonal)
    elif transform_id == 7:
        return np.rot90(np.fliplr(grid), k=1)  # anti-diagonal flip
    else:
        return grid.copy()


def apply_inverse_dihedral_transform(grid: np.ndarray, transform_id: int) -> np.ndarray:
    """Apply the inverse of a dihedral transform."""
    if transform_id == 0:
        return grid.copy()
    elif transform_id == 1:
        return np.rot90(grid, k=1)   # inverse of 90° CW is 90° CCW
    elif transform_id == 2:
        return np.rot90(grid, k=2)   # 180° is its own inverse
    elif transform_id == 3:
        return np.rot90(grid, k=-1)  # inverse of 270° CW is 90° CW
    elif transform_id == 4:
        return np.fliplr(grid)       # flip is its own inverse
    elif transform_id == 5:
        return np.flipud(grid)       # flip is its own inverse
    elif transform_id == 6:
        return grid.T                # transpose is its own inverse
    elif transform_id == 7:
        return np.rot90(np.fliplr(grid), k=1)  # its own inverse
    else:
        return grid.copy()


def transform_delta(delta: np.ndarray, transform_id: int) -> np.ndarray:
    """
    Transform a position delta according to a dihedral transform.
    delta is (row_delta, col_delta).
    """
    dr, dc = delta
    if transform_id == 0:
        return np.array([dr, dc])
    elif transform_id == 1:  # 90° CW: (r,c) -> (c, -r)
        return np.array([dc, -dr])
    elif transform_id == 2:  # 180°: (r,c) -> (-r, -c)
        return np.array([-dr, -dc])
    elif transform_id == 3:  # 270° CW: (r,c) -> (-c, r)
        return np.array([-dc, dr])
    elif transform_id == 4:  # flip horizontal: (r,c) -> (r, -c)
        return np.array([dr, -dc])
    elif transform_id == 5:  # flip vertical: (r,c) -> (-r, c)
        return np.array([-dr, dc])
    elif transform_id == 6:  # transpose: (r,c) -> (c, r)
        return np.array([dc, dr])
    elif transform_id == 7:  # anti-diagonal: (r,c) -> (-c, -r)
        return np.array([-dc, -dr])
    else:
        return np.array([dr, dc])


def apply_color_permutation(grid: np.ndarray, perm: np.ndarray) -> np.ndarray:
    """
    Apply a color permutation to a grid.
    perm[old_color] = new_color
    """
    return perm[grid]


def generate_color_permutation(keep_background: bool = True) -> np.ndarray:
    """
    Generate a random permutation of colors.
    If keep_background=True, color 0 stays as 0.
    """
    perm = np.arange(NUM_COLORS)
    if keep_background:
        # Shuffle colors 1-9, keep 0 fixed
        perm[1:] = np.random.permutation(perm[1:])
    else:
        perm = np.random.permutation(perm)
    return perm


# =============================================================================
# Data Loading
# =============================================================================

def load_puzzles(dataset_name: str, data_root: str = "kaggle/combined") -> Dict:
    """Load ARC puzzles from JSON files."""
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
# Ordering Integration (from ordering_module.py)
# =============================================================================

# Use labels_to_objects from object_module.py as the canonical implementation
labels_to_ordering_objects = labels_to_objects


def get_ordering_strategy(strategy_name: str) -> OrderingStrategy:
    """Get an ordering strategy instance by name."""
    strategy_map = {s.name: s for s in ALL_ORDERINGS}
    if strategy_name not in strategy_map:
        raise ValueError(f"Unknown ordering strategy: {strategy_name}. "
                        f"Available: {list(strategy_map.keys())}")
    return strategy_map[strategy_name]


# =============================================================================
# Relation Encoding
# =============================================================================

class RelationEncoder(nn.Module):
    """
    Computes pairwise relation features between objects.
    
    Features per pair (i, j):
        - Relative position (2): (centroid_j - centroid_i)
        - Relative size (1): log(area_j / area_i)
        - Same color (1): binary
        - Adjacent (1): binary (computed from distance)
        - Row-aligned (1): binary
        - Col-aligned (1): binary  
        - Valid pair (1): binary
    Total: 8 features
    """
    
    def __init__(self, adjacency_threshold: float = 0.1, alignment_threshold: float = 0.05):
        super().__init__()
        self.adjacency_threshold = adjacency_threshold
        self.alignment_threshold = alignment_threshold
        self.raw_feature_dim = 8
    
    def forward(self, centroids: torch.Tensor, areas: torch.Tensor, 
                colors: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        """
        Args:
            centroids: (B, K, 2) normalized centroids
            areas: (B, K) normalized areas
            colors: (B, K) color indices
            valid: (B, K) valid mask
        
        Returns:
            relations: (B, K, K, 8) pairwise relation features
        """
        B, K, _ = centroids.shape
        device = centroids.device
        
        relations = torch.zeros(B, K, K, self.raw_feature_dim, device=device)
        
        # 1-2. Relative position
        centroid_i = centroids.unsqueeze(2)  # (B, K, 1, 2)
        centroid_j = centroids.unsqueeze(1)  # (B, 1, K, 2)
        rel_pos = centroid_j - centroid_i    # (B, K, K, 2)
        relations[..., 0:2] = rel_pos
        
        # 3. Relative size
        areas_i = areas.unsqueeze(2).clamp(min=1e-6)  # (B, K, 1)
        areas_j = areas.unsqueeze(1).clamp(min=1e-6)  # (B, 1, K)
        rel_size = torch.log(areas_j / areas_i).clamp(-5, 5)  # (B, K, K)
        relations[..., 2] = rel_size
        
        # 4. Same color
        colors_i = colors.unsqueeze(2)  # (B, K, 1)
        colors_j = colors.unsqueeze(1)  # (B, 1, K)
        same_color = (colors_i == colors_j).float()  # (B, K, K)
        relations[..., 3] = same_color
        
        # 5. Adjacent (based on centroid distance)
        dist = torch.sqrt((rel_pos ** 2).sum(dim=-1) + 1e-8)  # (B, K, K)
        adjacent = (dist < self.adjacency_threshold).float()
        relations[..., 4] = adjacent
        
        # 6. Row-aligned
        row_diff = torch.abs(rel_pos[..., 0])  # (B, K, K)
        row_aligned = (row_diff < self.alignment_threshold).float()
        relations[..., 5] = row_aligned
        
        # 7. Col-aligned
        col_diff = torch.abs(rel_pos[..., 1])  # (B, K, K)
        col_aligned = (col_diff < self.alignment_threshold).float()
        relations[..., 6] = col_aligned
        
        # 8. Valid pair
        valid_i = valid.unsqueeze(2).float()  # (B, K, 1)
        valid_j = valid.unsqueeze(1).float()  # (B, 1, K)
        valid_pair = valid_i * valid_j        # (B, K, K)
        relations[..., 7] = valid_pair
        
        # Zero out invalid pairs
        relations = relations * valid_pair.unsqueeze(-1)

        return relations


# =============================================================================
# Per-Object Anchor Framing Module
# =============================================================================

class AnchorFramingModule(nn.Module):
    """
    Stores per-object-index anchor framing selections.

    Each object index (0, 1, 2, ...) has a selected framing:
    - delta: output - input position
    - grid_tl/tr/bl/br: distance from grid corner
    - object_relative: position relative to a reference object

    For object_relative, also stores which earlier object to reference,
    plus which anchor points to use (e.g., source.BR -> target.TL).

    Framings are determined by the screening phase (trying all framings
    and selecting based on pixel accuracy), NOT learned via gradient descent.
    """

    def __init__(self, max_objects: int = MAX_OBJECTS):
        super().__init__()
        self.max_objects = max_objects

        # Selected framing index for each object (set by screening)
        # Shape: (max_objects,) - values are indices into FRAMINGS
        self.register_buffer('selected_framings', torch.zeros(max_objects, dtype=torch.long))

        # Selected reference object for each object (for object_relative)
        # Shape: (max_objects,) - values are object indices, -1 if N/A
        self.register_buffer('selected_references', torch.full((max_objects,), -1, dtype=torch.long))

        # Pixel accuracy achieved by each object's selected framing
        self.register_buffer('framing_accuracies', torch.zeros(max_objects))

        # Whether each object has been configured
        self.register_buffer('is_configured', torch.zeros(max_objects, dtype=torch.bool))

        # Anchor point indices for object_relative framing (0-8, indexing into ALL_ANCHORS)
        # source_anchor: which anchor point on this object
        # target_anchor: which anchor point on the reference object
        self.register_buffer('source_anchors', torch.zeros(max_objects, dtype=torch.long))
        self.register_buffer('target_anchors', torch.zeros(max_objects, dtype=torch.long))

        # The learned offset between anchor points (in pixels)
        self.register_buffer('anchor_offsets', torch.zeros(max_objects, 2))

        # Grid anchor points for grid-relative framings (grid_tl, grid_tr, grid_bl, grid_br)
        # grid_anchors: which anchor point on this object should be at the target position
        # grid_target_positions: the absolute grid position (row, col) that anchor should be at
        self.register_buffer('grid_anchors', torch.zeros(max_objects, dtype=torch.long))
        self.register_buffer('grid_target_positions', torch.zeros(max_objects, 2))

        # Delta offsets for delta framing (mean delta across training examples)
        # Used in no-train mode to directly apply discovered offsets
        self.register_buffer('delta_offsets', torch.zeros(max_objects, 2))

        # Variance of discovered offsets (lower = more consistent = better)
        # Used to determine if training can be skipped
        self.register_buffer('framing_variances', torch.full((max_objects,), float('inf')))

    def set_framing(self, obj_idx: int, framing: str, reference_idx: int = -1,
                    accuracy: float = 0.0,
                    source_anchor: Optional[AnchorPoint] = None,
                    target_anchor: Optional[AnchorPoint] = None,
                    anchor_offset: Optional[Tuple[float, float]] = None,
                    grid_anchor: Optional[AnchorPoint] = None,
                    grid_target_position: Optional[Tuple[float, float]] = None,
                    variance: float = float('inf')):
        """
        Set the selected framing for an object index.

        Args:
            obj_idx: Which object index
            framing: One of FRAMINGS
            reference_idx: For object_relative, which object to reference (-1 otherwise)
            accuracy: Pixel accuracy achieved with this framing
            source_anchor: For object_relative, which anchor on this object
            target_anchor: For object_relative, which anchor on the reference object
            anchor_offset: For object_relative, the (row, col) offset between anchors
            grid_anchor: For grid framings, which anchor on this object
            grid_target_position: For grid framings, the absolute position (row, col) for that anchor
            variance: Variance of offsets across examples (lower = more consistent)
        """
        framing_idx = FRAMINGS.index(framing)
        self.selected_framings[obj_idx] = framing_idx
        self.selected_references[obj_idx] = reference_idx
        self.framing_accuracies[obj_idx] = float(accuracy)
        self.framing_variances[obj_idx] = float(variance)
        self.is_configured[obj_idx] = True

        # Store anchor points for object_relative framing
        if framing == 'object_relative' and source_anchor is not None:
            self.source_anchors[obj_idx] = ALL_ANCHORS.index(source_anchor)
            self.target_anchors[obj_idx] = ALL_ANCHORS.index(target_anchor)
            if anchor_offset is not None:
                self.anchor_offsets[obj_idx] = torch.tensor(anchor_offset)

        # Store anchor points for grid-relative framings
        if framing in ['grid_tl', 'grid_tr', 'grid_bl', 'grid_br'] and grid_anchor is not None:
            self.grid_anchors[obj_idx] = ALL_ANCHORS.index(grid_anchor)
            if grid_target_position is not None:
                self.grid_target_positions[obj_idx] = torch.tensor(grid_target_position)

    def get_framing(self, obj_idx: int) -> Tuple[str, int]:
        """
        Get the selected framing for an object index.

        Returns:
            (framing_name, reference_idx)
        """
        framing_idx = self.selected_framings[obj_idx].item()
        ref_idx = self.selected_references[obj_idx].item()
        return FRAMINGS[framing_idx], ref_idx

    def get_framing_idx(self, obj_idx: int) -> int:
        """Get the framing index for an object."""
        return self.selected_framings[obj_idx].item()

    def get_reference_idx(self, obj_idx: int) -> int:
        """Get the reference object index for an object."""
        return self.selected_references[obj_idx].item()

    def get_selected_framings(self) -> List[str]:
        """Get all selected framings as strings."""
        return [FRAMINGS[i] for i in self.selected_framings.cpu().numpy()]

    def get_selected_references(self) -> List[int]:
        """Get all selected reference objects."""
        return self.selected_references.cpu().numpy().tolist()

    def num_configured(self) -> int:
        """Return number of configured object indices."""
        return self.is_configured.sum().item()

    def get_anchor_points(self, obj_idx: int) -> Tuple[AnchorPoint, AnchorPoint, Tuple[float, float]]:
        """
        Get the anchor points for object_relative framing.

        Returns:
            (source_anchor, target_anchor, offset) tuple
        """
        src_idx = self.source_anchors[obj_idx].item()
        tgt_idx = self.target_anchors[obj_idx].item()
        offset = tuple(self.anchor_offsets[obj_idx].cpu().numpy().tolist())
        return ALL_ANCHORS[src_idx], ALL_ANCHORS[tgt_idx], offset

    def has_anchor_points(self, obj_idx: int) -> bool:
        """Check if anchor points have been set for this object."""
        framing_idx = self.selected_framings[obj_idx].item()
        return FRAMINGS[framing_idx] == 'object_relative' and self.is_configured[obj_idx].item()

    def get_grid_anchor(self, obj_idx: int) -> Tuple[AnchorPoint, Tuple[float, float]]:
        """
        Get the grid anchor point for grid-relative framings.

        Returns:
            (grid_anchor, target_position) tuple
        """
        anchor_idx = self.grid_anchors[obj_idx].item()
        target_pos = tuple(self.grid_target_positions[obj_idx].cpu().numpy().tolist())
        return ALL_ANCHORS[anchor_idx], target_pos

    def has_grid_anchor(self, obj_idx: int) -> bool:
        """Check if grid anchor has been set for this object."""
        framing_idx = self.selected_framings[obj_idx].item()
        framing = FRAMINGS[framing_idx]
        return framing in ['grid_tl', 'grid_tr', 'grid_bl', 'grid_br'] and self.is_configured[obj_idx].item()

    def set_delta_offset(self, obj_idx: int, offset: Tuple[float, float]):
        """Set the mean delta offset for delta framing (used in no-train mode)."""
        self.delta_offsets[obj_idx] = torch.tensor(offset)

    def get_delta_offset(self, obj_idx: int) -> Tuple[float, float]:
        """Get the mean delta offset for delta framing."""
        return tuple(self.delta_offsets[obj_idx].cpu().numpy().tolist())

    def has_delta_offset(self, obj_idx: int) -> bool:
        """Check if delta offset has been set for this object."""
        framing_idx = self.selected_framings[obj_idx].item()
        return FRAMINGS[framing_idx] == 'delta' and self.is_configured[obj_idx].item()

    def all_low_variance(self, threshold: float = 0.001) -> bool:
        """
        Check if all configured objects have low variance (near-zero).

        Used to determine if training can be skipped since screening found
        perfectly consistent anchor relationships.

        Args:
            threshold: Maximum variance to consider "low" (default 0.001)

        Returns:
            True if all configured objects have variance <= threshold
        """
        for obj_idx in range(self.max_objects):
            if self.is_configured[obj_idx]:
                if self.framing_variances[obj_idx] > threshold:
                    return False
        return self.num_configured() > 0  # Must have at least one configured

    def get_max_variance(self) -> float:
        """Get the maximum variance across all configured objects."""
        max_var = 0.0
        for obj_idx in range(self.max_objects):
            if self.is_configured[obj_idx]:
                var = self.framing_variances[obj_idx].item()
                if var > max_var:
                    max_var = var
        return max_var


def compute_framing_target(
    input_centroid: np.ndarray,
    output_centroid: np.ndarray,
    framing: str,
    grid_height: int,
    grid_width: int,
    reference_output_centroid: Optional[np.ndarray] = None,
    obj_size: float = 2.0,
    # Anchor point parameters for object_relative
    source_output_bbox: Optional[np.ndarray] = None,
    reference_output_bbox: Optional[np.ndarray] = None,
    source_anchor: Optional[AnchorPoint] = None,
    target_anchor: Optional[AnchorPoint] = None,
    # Grid anchor point parameters
    grid_anchor: Optional[AnchorPoint] = None,
    grid_target_position: Optional[Tuple[float, float]] = None
) -> np.ndarray:
    """
    Compute normalized target for a specific framing.

    Mirrors anchor_validation.py compute_targets() logic.

    Args:
        input_centroid: (2,) input [row, col] in pixel coordinates
        output_centroid: (2,) output [row, col] in pixel coordinates
        framing: One of FRAMINGS
        grid_height, grid_width: Grid dimensions
        reference_output_centroid: (2,) reference object output position (for object_relative)
        obj_size: Object size for corner framings
        source_output_bbox: (4,) output bbox [min_r, min_c, max_r, max_c] for anchor-based object_relative
        reference_output_bbox: (4,) reference output bbox for anchor-based object_relative
        source_anchor: which anchor on source object (for anchor-based object_relative)
        target_anchor: which anchor on reference object (for anchor-based object_relative)
        grid_anchor: which anchor on the object (for grid-based framings)
        grid_target_position: the target absolute position for that anchor (for grid-based framings)

    Returns:
        target: (2,) normalized target
    """
    H, W = grid_height, grid_width
    out_row, out_col = output_centroid

    if framing == 'delta':
        # Target: output - input (normalized by grid size)
        delta = output_centroid - input_centroid
        return delta / np.array([H, W])

    elif framing in ['grid_tl', 'grid_tr', 'grid_bl', 'grid_br']:
        # Check if grid anchor is provided for anchor-based computation
        if (grid_anchor is not None and grid_target_position is not None
                and source_output_bbox is not None):
            # Anchor-based: target = (anchor_pos - target_pos) / normalization
            src_top_left = (int(round(source_output_bbox[0])), int(round(source_output_bbox[1])))
            src_height = max(1, int(round(source_output_bbox[2] - source_output_bbox[0])) + 1)
            src_width = max(1, int(round(source_output_bbox[3] - source_output_bbox[1])) + 1)

            anchor_pos = get_anchor_position(src_top_left, (src_height, src_width), grid_anchor)
            offset = np.array([anchor_pos[0] - grid_target_position[0],
                              anchor_pos[1] - grid_target_position[1]])
            return offset / 20.0  # Normalize by 20 for consistency with old grid framings

        # Fallback to centroid-based (legacy behavior)
        if framing == 'grid_tl':
            return np.array([out_row / 20.0, out_col / 20.0])
        elif framing == 'grid_tr':
            dist_right = W - out_col - obj_size
            return np.array([out_row / 20.0, dist_right / 20.0])
        elif framing == 'grid_bl':
            dist_bottom = H - out_row - obj_size
            return np.array([dist_bottom / 20.0, out_col / 20.0])
        elif framing == 'grid_br':
            dist_bottom = H - out_row - obj_size
            dist_right = W - out_col - obj_size
            return np.array([dist_bottom / 20.0, dist_right / 20.0])

    elif framing == 'object_relative':
        # Check if anchor points are provided for anchor-based computation
        if (source_output_bbox is not None and reference_output_bbox is not None
                and source_anchor is not None and target_anchor is not None):
            # Anchor-based: target = (source_anchor_pos - target_anchor_pos) / grid
            src_top_left = (int(round(source_output_bbox[0])), int(round(source_output_bbox[1])))
            src_height = max(1, int(round(source_output_bbox[2] - source_output_bbox[0])) + 1)
            src_width = max(1, int(round(source_output_bbox[3] - source_output_bbox[1])) + 1)

            ref_top_left = (int(round(reference_output_bbox[0])), int(round(reference_output_bbox[1])))
            ref_height = max(1, int(round(reference_output_bbox[2] - reference_output_bbox[0])) + 1)
            ref_width = max(1, int(round(reference_output_bbox[3] - reference_output_bbox[1])) + 1)

            src_anchor_pos = get_anchor_position(src_top_left, (src_height, src_width), source_anchor)
            ref_anchor_pos = get_anchor_position(ref_top_left, (ref_height, ref_width), target_anchor)

            offset = np.array([src_anchor_pos[0] - ref_anchor_pos[0],
                              src_anchor_pos[1] - ref_anchor_pos[1]])
            return offset / np.array([H, W])

        # Centroid-based fallback
        if reference_output_centroid is None:
            # Fallback to delta if no reference
            delta = output_centroid - input_centroid
            return delta / np.array([H, W])
        relative = output_centroid - reference_output_centroid
        return relative / np.array([H, W])

    else:
        raise ValueError(f"Unknown framing: {framing}")


def decode_framing_prediction(
    prediction: np.ndarray,
    framing: str,
    input_centroid: np.ndarray,
    grid_height: int,
    grid_width: int,
    reference_position: Optional[np.ndarray] = None,
    obj_size: float = 2.0,
    # Anchor point parameters for object_relative
    source_size: Optional[Tuple[int, int]] = None,
    reference_bbox: Optional[np.ndarray] = None,
    source_anchor: Optional[AnchorPoint] = None,
    target_anchor: Optional[AnchorPoint] = None,
    # Grid anchor point parameters
    grid_anchor: Optional[AnchorPoint] = None,
    grid_target_position: Optional[Tuple[float, float]] = None
) -> np.ndarray:
    """
    Decode a framing-specific prediction back to pixel position.

    Mirrors anchor_validation.py decode_to_pixels() logic.

    Args:
        prediction: (2,) predicted value in framing space
        framing: One of FRAMINGS
        input_centroid: (2,) input position for delta framing
        grid_height, grid_width: Grid dimensions
        reference_position: (2,) already-decoded reference object position (for object_relative)
        obj_size: Object size for corner framings
        source_size: (height, width) of source object for anchor-based object_relative
        reference_bbox: (4,) reference object's decoded bbox for anchor-based object_relative
        source_anchor: which anchor on source object (for anchor-based object_relative)
        target_anchor: which anchor on reference object (for anchor-based object_relative)
        grid_anchor: which anchor on the object (for grid-based framings)
        grid_target_position: the target absolute position for that anchor (for grid-based framings)

    Returns:
        position: (2,) decoded [row, col] in pixel coordinates (centroid position)
    """
    H, W = grid_height, grid_width
    pred = prediction

    if framing == 'delta':
        # output = input + delta * grid_size
        delta = pred * np.array([H, W])
        return input_centroid + delta

    elif framing in ['grid_tl', 'grid_tr', 'grid_bl', 'grid_br']:
        # Check if grid anchor is provided for anchor-based decoding
        if (grid_anchor is not None and grid_target_position is not None
                and source_size is not None):
            # Anchor-based decoding:
            # 1. Decode the offset
            offset = pred * 20.0

            # 2. Compute anchor position from target position + offset
            anchor_pos = np.array([grid_target_position[0] + offset[0],
                                   grid_target_position[1] + offset[1]])

            # 3. Back-calculate top-left from anchor position
            src_h, src_w = source_size
            anchor_dr, anchor_dc = get_anchor_offset(grid_anchor, src_h, src_w)
            source_top_left = (anchor_pos[0] - anchor_dr, anchor_pos[1] - anchor_dc)

            # 4. Return centroid position
            centroid_row = source_top_left[0] + (src_h - 1) / 2.0
            centroid_col = source_top_left[1] + (src_w - 1) / 2.0
            return np.array([centroid_row, centroid_col])

        # Fallback to centroid-based (legacy behavior)
        if framing == 'grid_tl':
            row = pred[0] * 20.0
            col = pred[1] * 20.0
            return np.array([row, col])
        elif framing == 'grid_tr':
            row = pred[0] * 20.0
            dist_right = pred[1] * 20.0
            col = W - dist_right - obj_size
            return np.array([row, col])
        elif framing == 'grid_bl':
            dist_bottom = pred[0] * 20.0
            col = pred[1] * 20.0
            row = H - dist_bottom - obj_size
            return np.array([row, col])
        else:  # grid_br
            dist_bottom = pred[0] * 20.0
            dist_right = pred[1] * 20.0
            row = H - dist_bottom - obj_size
            col = W - dist_right - obj_size
            return np.array([row, col])

    elif framing == 'object_relative':
        # Check if anchor points are provided for anchor-based decoding
        if (source_size is not None and reference_bbox is not None
                and source_anchor is not None and target_anchor is not None):
            # Anchor-based decoding:
            # 1. Compute target_anchor position from reference bbox
            ref_top_left = (int(round(reference_bbox[0])), int(round(reference_bbox[1])))
            ref_height = max(1, int(round(reference_bbox[2] - reference_bbox[0])) + 1)
            ref_width = max(1, int(round(reference_bbox[3] - reference_bbox[1])) + 1)
            target_anchor_pos = get_anchor_position(ref_top_left, (ref_height, ref_width), target_anchor)

            # 2. source_anchor_pos = target_anchor_pos + prediction * grid
            offset = pred * np.array([H, W])
            source_anchor_pos = np.array([target_anchor_pos[0] + offset[0],
                                          target_anchor_pos[1] + offset[1]])

            # 3. Back-calculate source_top_left from source_anchor_pos
            src_h, src_w = source_size
            anchor_dr, anchor_dc = get_anchor_offset(source_anchor, src_h, src_w)
            source_top_left = (source_anchor_pos[0] - anchor_dr,
                              source_anchor_pos[1] - anchor_dc)

            # 4. Return centroid position
            centroid_row = source_top_left[0] + (src_h - 1) / 2.0
            centroid_col = source_top_left[1] + (src_w - 1) / 2.0
            return np.array([centroid_row, centroid_col])

        # Centroid-based fallback
        if reference_position is None:
            # Fallback to delta-style decoding from input
            offset = pred * np.array([H, W])
            return input_centroid + offset
        offset = pred * np.array([H, W])
        return reference_position + offset

    else:
        raise ValueError(f"Unknown framing: {framing}")


def apply_screening_offset(
    anchor_module: 'AnchorFramingModule',
    obj_idx: int,
    input_centroid: np.ndarray,
    grid_height: int,
    grid_width: int,
    source_size: Optional[Tuple[int, int]] = None,
    reference_position: Optional[np.ndarray] = None,
    reference_bbox: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Apply the screening-discovered offset directly to predict object position.

    This function is used in no-train mode to bypass the neural network and
    directly use the offsets discovered during screening.

    Args:
        anchor_module: AnchorFramingModule with discovered offsets
        obj_idx: Which object index to predict
        input_centroid: (2,) input position [row, col] in pixels
        grid_height, grid_width: Grid dimensions
        source_size: (height, width) of source object for anchor-based framings
        reference_position: (2,) decoded position of reference object (for object_relative)
        reference_bbox: (4,) decoded bbox of reference object (for object_relative with anchors)

    Returns:
        position: (2,) predicted [row, col] in pixel coordinates
    """
    framing_idx = anchor_module.get_framing_idx(obj_idx)
    framing = FRAMINGS[framing_idx]
    H, W = grid_height, grid_width

    if framing == 'delta':
        # Use stored mean delta
        delta_offset = anchor_module.get_delta_offset(obj_idx)
        return input_centroid + np.array(delta_offset)

    elif framing in ['grid_tl', 'grid_tr', 'grid_bl', 'grid_br']:
        if anchor_module.has_grid_anchor(obj_idx) and source_size is not None:
            grid_anchor, target_position = anchor_module.get_grid_anchor(obj_idx)
            # The target_position is where the anchor should be
            # Back-calculate centroid from anchor position
            src_h, src_w = source_size
            anchor_dr, anchor_dc = get_anchor_offset(grid_anchor, src_h, src_w)
            source_top_left = (target_position[0] - anchor_dr, target_position[1] - anchor_dc)
            centroid_row = source_top_left[0] + (src_h - 1) / 2.0
            centroid_col = source_top_left[1] + (src_w - 1) / 2.0
            return np.array([centroid_row, centroid_col])
        else:
            # Fallback: no anchor info, just use input position
            return input_centroid

    elif framing == 'object_relative':
        if anchor_module.has_anchor_points(obj_idx) and reference_bbox is not None and source_size is not None:
            source_anchor, target_anchor, offset = anchor_module.get_anchor_points(obj_idx)

            # Compute target_anchor position from reference bbox
            ref_top_left = (int(round(reference_bbox[0])), int(round(reference_bbox[1])))
            ref_height = max(1, int(round(reference_bbox[2] - reference_bbox[0])) + 1)
            ref_width = max(1, int(round(reference_bbox[3] - reference_bbox[1])) + 1)
            target_anchor_pos = get_anchor_position(ref_top_left, (ref_height, ref_width), target_anchor)

            # source_anchor_pos = target_anchor_pos + stored_offset
            source_anchor_pos = np.array([target_anchor_pos[0] + offset[0],
                                          target_anchor_pos[1] + offset[1]])

            # Back-calculate source_top_left from source_anchor_pos
            src_h, src_w = source_size
            anchor_dr, anchor_dc = get_anchor_offset(source_anchor, src_h, src_w)
            source_top_left = (source_anchor_pos[0] - anchor_dr, source_anchor_pos[1] - anchor_dc)

            # Return centroid position
            centroid_row = source_top_left[0] + (src_h - 1) / 2.0
            centroid_col = source_top_left[1] + (src_w - 1) / 2.0
            return np.array([centroid_row, centroid_col])

        elif reference_position is not None:
            # Fallback to centroid-based offset
            # The anchor_offsets stores the discovered offset in pixels
            offset = anchor_module.get_anchor_points(obj_idx)[2] if anchor_module.has_anchor_points(obj_idx) else (0, 0)
            return reference_position + np.array(offset)

        else:
            # No reference available, use input position
            return input_centroid

    else:
        raise ValueError(f"Unknown framing: {framing}")


# =============================================================================
# Position Predictor
# =============================================================================

class RelationalPositionPredictor(nn.Module):
    """
    Predicts position transformation using relation-conditioned attention.

    Each object attends to other objects, with relations informing attention weights.
    Then predicts position delta from (object features, attended context).
    """

    def __init__(self, relation_dim: int = 8, hidden_dim: int = 64,
                 num_heads: int = 4, pos_dim: int = 2, num_colors: int = 10):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.pos_dim = pos_dim

        # Color embedding
        self.color_embed = nn.Embedding(num_colors, hidden_dim // 2)

        # Position encoding
        self.pos_encoder = nn.Linear(pos_dim, hidden_dim // 2)

        # Combined input: color_embed + pos_encoding
        input_dim = hidden_dim

        # Query projection
        self.query_proj = nn.Linear(input_dim, hidden_dim)

        # Key projection: includes relation features
        self.key_proj = nn.Linear(input_dim + relation_dim, hidden_dim)

        # Value projection
        self.value_proj = nn.Linear(input_dim, hidden_dim)

        # Output projection
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

        # Position delta prediction
        self.delta_mlp = nn.Sequential(
            nn.Linear(input_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, pos_dim)
        )

        self.scale = hidden_dim ** -0.5

    def forward(self, centroids: torch.Tensor, colors: torch.Tensor,
                relations: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        """
        Args:
            centroids: (B, K, 2) normalized centroids
            colors: (B, K) color indices
            relations: (B, K, K, relation_dim) pairwise relations
            valid: (B, K) valid mask

        Returns:
            deltas: (B, K, 2) predicted position changes
        """
        B, K, _ = centroids.shape
        device = centroids.device

        # Build object representations
        color_emb = self.color_embed(colors)  # (B, K, hidden_dim//2)
        pos_enc = self.pos_encoder(centroids)  # (B, K, hidden_dim//2)
        obj_repr = torch.cat([color_emb, pos_enc], dim=-1)  # (B, K, hidden_dim)

        # Queries
        queries = self.query_proj(obj_repr)  # (B, K, hidden_dim)

        # Keys: include relations
        obj_repr_expanded = obj_repr.unsqueeze(1).expand(-1, K, -1, -1)  # (B, K, K, hidden_dim)
        key_input = torch.cat([obj_repr_expanded, relations], dim=-1)    # (B, K, K, hidden_dim + relation_dim)
        keys = self.key_proj(key_input)  # (B, K, K, hidden_dim)

        # Values
        values = self.value_proj(obj_repr)  # (B, K, hidden_dim)

        # Attention scores
        queries_expanded = queries.unsqueeze(2)  # (B, K, 1, hidden_dim)
        attn_logits = (queries_expanded * keys).sum(dim=-1) * self.scale  # (B, K, K)

        # Mask invalid objects
        invalid_mask = ~valid  # (B, K)
        attn_logits = attn_logits.masked_fill(invalid_mask.unsqueeze(1), float('-inf'))

        # Mask self-attention
        self_mask = torch.eye(K, device=device, dtype=torch.bool).unsqueeze(0)
        attn_logits = attn_logits.masked_fill(self_mask, float('-inf'))

        # Softmax
        attn_weights = F.softmax(attn_logits, dim=-1)  # (B, K, K)
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0)

        # Aggregate
        context = torch.bmm(attn_weights, values)  # (B, K, hidden_dim)
        context = self.out_proj(context)

        # Predict deltas
        delta_input = torch.cat([obj_repr, context], dim=-1)  # (B, K, 2*hidden_dim)
        deltas = self.delta_mlp(delta_input)  # (B, K, pos_dim)

        # Zero out invalid
        deltas = deltas * valid.unsqueeze(-1).float()

        return deltas


class PositionalTransformModule(nn.Module):
    """Complete module for predicting object position transformations."""
    
    def __init__(self, hidden_dim: int = 64, num_heads: int = 4):
        super().__init__()
        
        self.relation_encoder = RelationEncoder()
        self.position_predictor = RelationalPositionPredictor(
            relation_dim=self.relation_encoder.raw_feature_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            pos_dim=2
        )
    
    def forward(self, centroids: torch.Tensor, areas: torch.Tensor,
                colors: torch.Tensor, valid: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            centroids: (B, K, 2) normalized centroids
            areas: (B, K) normalized areas  
            colors: (B, K) color indices
            valid: (B, K) valid mask
        
        Returns:
            relations: (B, K, K, relation_dim)
            deltas: (B, K, 2) predicted position deltas
        """
        relations = self.relation_encoder(centroids, areas, colors, valid)
        deltas = self.position_predictor(centroids, colors, relations, valid)
        return relations, deltas


# =============================================================================
# Per-Object Anchor Position Predictor
# =============================================================================

class PerObjectAnchorPositionPredictor(nn.Module):
    """
    Position predictor with per-object-index anchor framing selection.

    Each object index learns:
    1. Which framing to use (delta, grid corners, object_relative)
    2. For object_relative: which earlier object to reference

    For object_relative framing, object i can reference outputs of earlier objects 0..i-1.
    """

    def __init__(self, relation_dim: int = 8, hidden_dim: int = 64,
                 num_heads: int = 4, num_colors: int = 10,
                 max_objects: int = MAX_OBJECTS):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.max_objects = max_objects

        # Anchor framing module for per-object framing selection
        # (Framings are discovered by screening, not learned)
        self.anchor_module = AnchorFramingModule(max_objects)

        # Input object encoding (same as AutoregressivePositionPredictor)
        self.color_embed = nn.Embedding(num_colors, hidden_dim // 4)
        self.input_pos_encoder = nn.Linear(2, hidden_dim // 4)
        self.input_bbox_encoder = nn.Linear(4, hidden_dim // 4)
        self.input_area_encoder = nn.Linear(1, hidden_dim // 4)

        input_obj_dim = hidden_dim

        # Output context encoding (for already-predicted objects)
        self.output_bbox_encoder = nn.Linear(4, hidden_dim // 2)
        self.predicted_flag_embed = nn.Embedding(2, hidden_dim // 2)

        output_ctx_dim = hidden_dim

        # Framing embedding - tells the model which framing we're predicting in
        self.framing_embed = nn.Embedding(NUM_FRAMINGS, hidden_dim // 4)

        # Combine input features + output context + framing
        combined_dim = input_obj_dim + output_ctx_dim

        # Self-attention over all objects
        self.q_proj = nn.Linear(combined_dim, hidden_dim)
        self.k_proj = nn.Linear(combined_dim + relation_dim, hidden_dim)
        self.v_proj = nn.Linear(combined_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

        # Prediction MLP: takes object features + context + framing info
        self.output_mlp = nn.Sequential(
            nn.Linear(combined_dim + hidden_dim + hidden_dim // 4, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2)  # Predicted value in framing space
        )

        self.scale = hidden_dim ** -0.5

    def encode_input_objects(self, centroids: torch.Tensor, bboxes: torch.Tensor,
                              areas: torch.Tensor, colors: torch.Tensor) -> torch.Tensor:
        """Encode input object features."""
        color_emb = self.color_embed(colors)
        pos_enc = self.input_pos_encoder(centroids)

        heights = bboxes[..., 2] - bboxes[..., 0]
        widths = bboxes[..., 3] - bboxes[..., 1]
        bbox_features = torch.stack([heights, widths, bboxes[..., 0], bboxes[..., 1]], dim=-1)
        bbox_enc = self.input_bbox_encoder(bbox_features)

        area_enc = self.input_area_encoder(areas.unsqueeze(-1))

        return torch.cat([color_emb, pos_enc, bbox_enc, area_enc], dim=-1)

    def encode_output_context(self, output_bboxes: torch.Tensor,
                               predicted_mask: torch.Tensor) -> torch.Tensor:
        """Encode already-predicted output bounding boxes."""
        bbox_enc = self.output_bbox_encoder(output_bboxes)
        flag_enc = self.predicted_flag_embed(predicted_mask.long())

        output_repr = torch.cat([bbox_enc, flag_enc], dim=-1)
        output_repr[..., :self.hidden_dim // 2] *= predicted_mask.unsqueeze(-1).float()

        return output_repr

    def forward_single(self, input_repr: torch.Tensor, output_repr: torch.Tensor,
                        relations: torch.Tensor, valid: torch.Tensor,
                        target_idx: int, framing_idx: int) -> torch.Tensor:
        """
        Predict position for a single object in a specific framing.

        Args:
            input_repr: (B, K, hidden_dim) input object features
            output_repr: (B, K, hidden_dim) output context features
            relations: (B, K, K, relation_dim) pairwise relations
            valid: (B, K) valid object mask
            target_idx: which object to predict for
            framing_idx: which framing to predict in (0-5)

        Returns:
            output_val: (B, 2) predicted value in framing space
        """
        B, K, _ = input_repr.shape
        device = input_repr.device

        # Combine input and output representations
        combined = torch.cat([input_repr, output_repr], dim=-1)

        # Get query for target object
        query = self.q_proj(combined[:, target_idx:target_idx+1, :])

        # Keys and values from all objects
        combined_expanded = combined.unsqueeze(1).expand(-1, K, -1, -1)
        key_input = torch.cat([combined_expanded, relations], dim=-1)
        keys = self.k_proj(key_input[:, target_idx, :, :])
        values = self.v_proj(combined)

        # Attention scores
        attn_logits = (query * keys).sum(dim=-1) * self.scale
        attn_logits = attn_logits.masked_fill(~valid, float('-inf'))
        attn_logits[:, target_idx] = float('-inf')  # Mask self

        attn_weights = F.softmax(attn_logits, dim=-1)
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0)

        # Aggregate
        context = (attn_weights.unsqueeze(-1) * values).sum(dim=1)
        context = self.out_proj(context)

        # Get framing embedding
        framing_emb = self.framing_embed(
            torch.tensor([framing_idx], device=device).expand(B)
        )

        # Predict output in framing space
        target_combined = combined[:, target_idx, :]
        mlp_input = torch.cat([target_combined, context, framing_emb], dim=-1)
        output_val = self.output_mlp(mlp_input)

        return output_val

    def forward_teacher_forcing(self, input_repr: torch.Tensor,
                                  target_output_bboxes: torch.Tensor,
                                  relations: torch.Tensor, valid: torch.Tensor
                                  ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass with teacher forcing and per-object framing selection.

        Args:
            input_repr: (B, K, hidden_dim) input object features
            target_output_bboxes: (B, K, 4) ground truth output bboxes
            relations: (B, K, K, relation_dim) pairwise relations
            valid: (B, K) valid mask

        Returns:
            output_predictions: (B, K, 2) predicted values in framing space
            framing_weights: (B, K, num_framings) framing selection weights
            reference_weights: (B, K, max_objects) reference selection weights
        """
        B, K, _ = input_repr.shape
        device = input_repr.device

        output_predictions = torch.zeros(B, K, 2, device=device)
        framing_weights = torch.zeros(B, K, NUM_FRAMINGS, device=device)
        reference_weights = torch.zeros(B, K, self.max_objects, device=device)

        for i in range(K):
            # Build causal mask
            predicted_mask = torch.zeros(B, K, dtype=torch.bool, device=device)
            if i > 0:
                predicted_mask[:, :i] = valid[:, :i]

            # Encode output context
            output_repr = self.encode_output_context(target_output_bboxes, predicted_mask)

            # Get selected framing for this object index (from screening)
            framing_idx = self.anchor_module.get_framing_idx(i)
            ref_idx = self.anchor_module.get_reference_idx(i)

            # Build one-hot framing weights for tracking
            framing_weights[:, i, framing_idx] = 1.0

            # Build reference weights (one-hot if object_relative)
            if ref_idx >= 0:
                reference_weights[:, i, ref_idx] = 1.0

            # Predict in the selected framing
            pred_i = self.forward_single(
                input_repr, output_repr, relations, valid, i, framing_idx
            )
            output_predictions[:, i] = pred_i

        # Zero out invalid
        output_predictions = output_predictions * valid.unsqueeze(-1).float()

        return output_predictions, framing_weights, reference_weights

    def forward_inference(self, input_repr: torch.Tensor,
                          input_bboxes: torch.Tensor,
                          relations: torch.Tensor, valid: torch.Tensor
                          ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Inference without ground truth output bboxes.

        Builds output context iteratively as each object is predicted.

        Args:
            input_repr: (B, K, hidden_dim) input object features
            input_bboxes: (B, K, 4) input bboxes (used as initial output context)
            relations: (B, K, K, relation_dim) pairwise relations
            valid: (B, K) valid mask

        Returns:
            output_predictions: (B, K, 2) predicted values in framing space
            framing_weights: (B, K, num_framings) framing selection weights
            reference_weights: (B, K, max_objects) reference selection weights
        """
        B, K, _ = input_repr.shape
        device = input_repr.device

        output_predictions = torch.zeros(B, K, 2, device=device)
        framing_weights = torch.zeros(B, K, NUM_FRAMINGS, device=device)
        reference_weights = torch.zeros(B, K, self.max_objects, device=device)

        # For inference, use input bboxes as initial output context (before any predictions)
        # Then build predictions iteratively
        predicted_output_bboxes = input_bboxes.clone()  # Start with input positions
        predicted_mask = torch.zeros(B, K, dtype=torch.bool, device=device)

        for i in range(K):
            # Skip invalid objects
            if not valid[:, i].any():
                continue

            # Encode output context (previous predictions)
            output_repr = self.encode_output_context(predicted_output_bboxes, predicted_mask)

            # Get selected framing for this object index (from screening)
            framing_idx = self.anchor_module.get_framing_idx(i)
            ref_idx = self.anchor_module.get_reference_idx(i)

            # Build one-hot framing weights for tracking
            framing_weights[:, i, framing_idx] = 1.0

            # Build reference weights (one-hot if object_relative)
            if ref_idx >= 0:
                reference_weights[:, i, ref_idx] = 1.0

            # Predict in the selected framing
            pred_i = self.forward_single(
                input_repr, output_repr, relations, valid, i, framing_idx
            )
            output_predictions[:, i] = pred_i

            # Mark this object as predicted for next iteration
            predicted_mask[:, i] = valid[:, i]

        # Zero out invalid
        output_predictions = output_predictions * valid.unsqueeze(-1).float()

        return output_predictions, framing_weights, reference_weights


class PerObjectAnchorTransformModule(nn.Module):
    """
    Complete module for position prediction with per-object anchor learning.

    Combines:
    - RelationEncoder for pairwise features
    - PerObjectAnchorPositionPredictor for framing-aware position prediction
    """

    def __init__(self, hidden_dim: int = 64, num_heads: int = 4,
                 max_objects: int = MAX_OBJECTS):
        super().__init__()

        self.relation_encoder = RelationEncoder()
        self.position_predictor = PerObjectAnchorPositionPredictor(
            relation_dim=self.relation_encoder.raw_feature_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            max_objects=max_objects
        )

    @property
    def anchor_module(self) -> AnchorFramingModule:
        """Access the anchor framing module."""
        return self.position_predictor.anchor_module

    def forward(self, centroids: torch.Tensor, bboxes: torch.Tensor,
                areas: torch.Tensor, colors: torch.Tensor, valid: torch.Tensor,
                target_output_bboxes: torch.Tensor = None
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            centroids: (B, K, 2) normalized input centroids
            bboxes: (B, K, 4) normalized input bboxes
            areas: (B, K) normalized areas
            colors: (B, K) color indices
            valid: (B, K) valid mask
            target_output_bboxes: (B, K, 4) ground truth output bboxes (for teacher forcing)

        Returns:
            relations: (B, K, K, relation_dim)
            predictions: (B, K, 2) predicted values in framing space
            framing_weights: (B, K, num_framings)
            reference_weights: (B, K, max_objects)
        """
        # Encode relations
        relations = self.relation_encoder(centroids, areas, colors, valid)

        # Encode input objects
        input_repr = self.position_predictor.encode_input_objects(
            centroids, bboxes, areas, colors
        )

        if target_output_bboxes is not None:
            # Training: use teacher forcing with ground truth output bboxes
            predictions, framing_weights, reference_weights = \
                self.position_predictor.forward_teacher_forcing(
                    input_repr, target_output_bboxes, relations, valid
                )
        else:
            # Inference: build output context iteratively
            predictions, framing_weights, reference_weights = \
                self.position_predictor.forward_inference(
                    input_repr, bboxes, relations, valid
                )

        return relations, predictions, framing_weights, reference_weights

    def get_framing_summary(self, num_objects: int) -> str:
        """Get a human-readable summary of learned framings."""
        framings = self.anchor_module.get_selected_framings()[:num_objects]
        refs = self.anchor_module.get_selected_references()[:num_objects]

        lines = []
        for i, (f, r) in enumerate(zip(framings, refs)):
            if f == 'object_relative':
                lines.append(f"  Object {i}: {f} (ref: object {r})")
            else:
                lines.append(f"  Object {i}: {f}")
        return "\n".join(lines)


# =============================================================================
# Dataset
# =============================================================================

@dataclass
class PositionSample:
    """A single training sample for position prediction."""
    puzzle_id: str
    example_idx: int
    input_centroids: np.ndarray   # (K, 2)
    input_bboxes: np.ndarray      # (K, 4) [min_r, min_c, max_r, max_c] normalized
    input_areas: np.ndarray       # (K,)
    input_colors: np.ndarray      # (K,)
    input_valid: np.ndarray       # (K,)
    target_deltas: np.ndarray     # (K, 2) position deltas
    target_output_bboxes: np.ndarray  # (K, 4) output bboxes (for teacher forcing)
    correspondence: np.ndarray    # (K,) index into output, -1 if no match

    # Per-object anchor framing fields (optional, for per-object-anchor mode)
    target_by_framing: Optional[Dict[str, np.ndarray]] = None  # framing -> (K, 2) targets
    output_centroids: Optional[np.ndarray] = None  # (K, 2) for object_relative
    grid_height: Optional[int] = None
    grid_width: Optional[int] = None


class PositionDataset(torch.utils.data.Dataset):
    """Dataset for position prediction training with optional augmentation."""

    def __init__(self, puzzles: Dict, puzzle_ids: List[str] = None,
                 use_color_only: bool = False, include_test: bool = False,
                 num_augmentations: int = 0, dihedral_only: bool = False,
                 color_only: bool = False, ordering_strategy: Optional[str] = None,
                 predict_absolute: bool = False, per_object_anchor: bool = False,
                 selection_criterion: Optional[str] = None,
                 selection_rule: Optional[str] = None):
        self.samples: List[PositionSample] = []
        self.use_color_only = use_color_only
        self.ordering_strategy = ordering_strategy
        self.predict_absolute = predict_absolute
        self.per_object_anchor = per_object_anchor
        self.selection_criterion = selection_criterion
        self.selection_rule = selection_rule

        if puzzle_ids is None:
            puzzle_ids = list(puzzles.keys())

        base_count = 0
        for puzzle_id in puzzle_ids:
            if puzzle_id not in puzzles:
                print(f"Warning: puzzle {puzzle_id} not found")
                continue

            puzzle = puzzles[puzzle_id]
            examples = puzzle.get('train', [])
            if include_test:
                examples = examples + puzzle.get('test', [])

            for ex_idx, example in enumerate(examples):
                if 'output' not in example:
                    continue

                input_grid = np.array(example['input'], dtype=np.int64)
                output_grid = np.array(example['output'], dtype=np.int64)

                # Always create the original (non-augmented) sample
                sample = self._create_sample(puzzle_id, ex_idx, input_grid, output_grid)
                if sample is not None:
                    self.samples.append(sample)
                    base_count += 1

                # Generate augmented samples by random sampling
                if num_augmentations > 0:
                    seen_hashes = set()
                    orig_hash = self._grid_hash(input_grid, output_grid)
                    seen_hashes.add(orig_hash)

                    max_attempts = num_augmentations * 5
                    for _ in range(max_attempts):
                        if len(seen_hashes) > num_augmentations:
                            break

                        # Sample random augmentation
                        if color_only:
                            d_id = 0
                        else:
                            d_id = np.random.randint(0, 8)

                        if dihedral_only:
                            color_perm = None
                        else:
                            color_perm = generate_color_permutation()

                        # Apply augmentations
                        aug_input = apply_dihedral_transform(input_grid, d_id)
                        aug_output = apply_dihedral_transform(output_grid, d_id)

                        if color_perm is not None:
                            aug_input = apply_color_permutation(aug_input, color_perm)
                            aug_output = apply_color_permutation(aug_output, color_perm)

                        # Check for duplicate
                        aug_hash = self._grid_hash(aug_input, aug_output)
                        if aug_hash in seen_hashes:
                            continue
                        seen_hashes.add(aug_hash)

                        sample = self._create_sample(
                            puzzle_id, ex_idx, aug_input, aug_output,
                            dihedral_id=d_id, color_perm=color_perm
                        )
                        if sample is not None:
                            self.samples.append(sample)

        aug_factor = len(self.samples) / max(base_count, 1)
        print(f"Created {len(self.samples)} position samples "
              f"({base_count} base × {aug_factor:.1f} augmentation factor)")

    def _grid_hash(self, inp: np.ndarray, out: np.ndarray) -> str:
        """Hash a pair of grids for deduplication."""
        return f"{inp.tobytes().hex()}|{out.tobytes().hex()}"
    
    def _create_sample(self, puzzle_id: str, ex_idx: int,
                        input_grid: np.ndarray, output_grid: np.ndarray,
                        dihedral_id: int = 0, color_perm: Optional[np.ndarray] = None
                        ) -> Optional[PositionSample]:
        """Create a training sample from input/output grids."""

        # Extract objects
        input_labels, input_colors, input_bboxes, _ = extract_connected_components(
            input_grid, use_color_only=self.use_color_only
        )
        output_labels, output_colors, output_bboxes, _ = extract_connected_components(
            output_grid, use_color_only=self.use_color_only
        )

        if len(input_colors) == 0:
            return None

        # Apply selection filtering if configured (before ordering)
        selection_mask = None
        if self.selection_criterion is not None and self.selection_rule is not None:
            selection_mask = compute_selection_mask(
                input_labels, input_colors, input_bboxes, input_grid,
                self.selection_criterion, self.selection_rule
            )

        # Apply ordering strategy (after selection)
        if self.ordering_strategy:
            # Compute ordering mapping BEFORE sorting (to update selection_mask)
            if selection_mask is not None:
                objects_before = labels_to_ordering_objects(input_labels, input_colors, input_bboxes)
                strategy = get_ordering_strategy(self.ordering_strategy)
                ordered_objects = strategy.order(objects_before)
                old_to_new = {obj.id: new_idx for new_idx, obj in enumerate(ordered_objects)}
                # Reorder selection mask to match new ordering
                new_mask = np.zeros_like(selection_mask)
                for old_idx, selected in enumerate(selection_mask):
                    if old_idx in old_to_new:
                        new_mask[old_to_new[old_idx]] = selected
                selection_mask = new_mask

            input_labels, input_colors, input_bboxes, _ = sort_objects_by_strategy(
                input_labels, input_colors, input_bboxes, self.ordering_strategy
            )
            output_labels, output_colors, output_bboxes, _ = sort_objects_by_strategy(
                output_labels, output_colors, output_bboxes, self.ordering_strategy
            )

        # Find correspondences (with pattern matching for moved objects)
        matches = find_object_correspondences(
            input_labels, input_colors,
            output_labels, output_colors,
            iou_threshold=0.0,  # Allow any match with same color
            input_grid=input_grid,
            output_grid=output_grid
        )

        if len(matches) == 0:
            return None

        # Compute properties
        grid_size = max(input_grid.shape[0], input_grid.shape[1],
                        output_grid.shape[0], output_grid.shape[1], GRID_SIZE)

        input_props = compute_object_properties(input_labels, input_colors, input_bboxes, grid_size)
        output_props = compute_object_properties(output_labels, output_colors, output_bboxes, grid_size)

        # Apply selection mask to filter objects (mark non-selected as invalid)
        if selection_mask is not None:
            # Pad selection_mask to MAX_OBJECTS size
            padded_mask = np.zeros(MAX_OBJECTS, dtype=bool)
            padded_mask[:len(selection_mask)] = selection_mask
            # Only keep objects that are both valid AND selected
            input_props['valid'] = input_props['valid'] & padded_mask

        # Build correspondence array and targets
        correspondence = np.full(MAX_OBJECTS, -1, dtype=np.int64)
        target_deltas = np.zeros((MAX_OBJECTS, 2), dtype=np.float32)
        target_output_bboxes = np.zeros((MAX_OBJECTS, 4), dtype=np.float32)

        for in_idx, out_idx, score in matches:
            if in_idx < MAX_OBJECTS and out_idx < MAX_OBJECTS:
                correspondence[in_idx] = out_idx
                if self.predict_absolute:
                    # Target = absolute output centroid position
                    target_deltas[in_idx] = output_props['centroids'][out_idx]
                else:
                    # Target delta = output_centroid - input_centroid
                    target_deltas[in_idx] = output_props['centroids'][out_idx] - input_props['centroids'][in_idx]

                # Full output bounding box for teacher forcing
                # IMPORTANT: Use OUTPUT position (TL) but INPUT size
                # Output shapes may be L-shaped due to occlusion, but input shapes are the true rectangles
                output_tl = output_props['bboxes'][out_idx, :2]  # min_r, min_c from output
                input_size = input_props['bboxes'][in_idx, 2:] - input_props['bboxes'][in_idx, :2]  # height, width from input
                target_output_bboxes[in_idx] = np.array([
                    output_tl[0], output_tl[1],  # TL from output
                    output_tl[0] + input_size[0], output_tl[1] + input_size[1]  # BR = TL + input size
                ])

        # Compute per-object anchor framing targets if enabled
        target_by_framing = None
        output_centroids = None
        grid_h = None
        grid_w = None

        if self.per_object_anchor:
            grid_h = max(input_grid.shape[0], output_grid.shape[0])
            grid_w = max(input_grid.shape[1], output_grid.shape[1])

            # Store output centroids for object_relative computation
            output_centroids = np.zeros((MAX_OBJECTS, 2), dtype=np.float32)
            for in_idx, out_idx, _ in matches:
                if in_idx < MAX_OBJECTS and out_idx < MAX_OBJECTS:
                    # Store in pixel coordinates (denormalized)
                    output_centroids[in_idx] = output_props['centroids'][out_idx] * grid_size

            # Compute targets for each non-relative framing
            target_by_framing = {}
            for framing in ['delta', 'grid_tl', 'grid_tr', 'grid_bl', 'grid_br']:
                targets = np.zeros((MAX_OBJECTS, 2), dtype=np.float32)
                for in_idx, out_idx, _ in matches:
                    if in_idx < MAX_OBJECTS and out_idx < MAX_OBJECTS:
                        input_centroid = input_props['centroids'][in_idx] * grid_size
                        output_centroid = output_props['centroids'][out_idx] * grid_size
                        targets[in_idx] = compute_framing_target(
                            input_centroid, output_centroid,
                            framing, grid_h, grid_w
                        )
                target_by_framing[framing] = targets

            # object_relative targets are computed dynamically during training
            # (depends on which reference object is selected)
            target_by_framing['object_relative'] = np.zeros((MAX_OBJECTS, 2), dtype=np.float32)

        return PositionSample(
            puzzle_id=puzzle_id,
            example_idx=ex_idx,
            input_centroids=input_props['centroids'],
            input_bboxes=input_props['bboxes'],
            input_areas=input_props['areas'],
            input_colors=input_props['colors'],
            input_valid=input_props['valid'],
            target_deltas=target_deltas,
            target_output_bboxes=target_output_bboxes,
            correspondence=correspondence,
            target_by_framing=target_by_framing,
            output_centroids=output_centroids,
            grid_height=grid_h,
            grid_width=grid_w
        )
    
    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, ...]:
        sample = self.samples[idx]

        # Compute heights and widths from bboxes
        heights = sample.input_bboxes[:, 2] - sample.input_bboxes[:, 0]
        widths = sample.input_bboxes[:, 3] - sample.input_bboxes[:, 1]

        return (
            torch.from_numpy(sample.input_centroids).float(),
            torch.from_numpy(sample.input_bboxes).float(),
            torch.from_numpy(sample.input_areas).float(),
            torch.from_numpy(sample.input_colors).long(),
            torch.from_numpy(sample.input_valid).bool(),
            torch.from_numpy(sample.target_deltas).float(),
            torch.from_numpy(sample.target_output_bboxes).float(),
            torch.from_numpy(sample.correspondence).long(),
            torch.from_numpy(heights).float(),
            torch.from_numpy(widths).float(),
        )


# =============================================================================
# Training
# =============================================================================

def train_epoch(model: nn.Module, dataloader: torch.utils.data.DataLoader,
                optimizer: torch.optim.Optimizer, device: torch.device) -> Dict:
    """Train for one epoch."""
    model.train()

    total_loss = 0.0
    total_samples = 0
    total_matched = 0
    total_correct_direction = 0

    for batch in dataloader:
        (centroids, bboxes, areas, colors, valid, target_deltas,
         target_output_bboxes, correspondence,
         heights, widths) = [x.to(device) for x in batch]

        optimizer.zero_grad()

        # Forward
        relations, pred_deltas = model(centroids, areas, colors, valid)
        targets = target_deltas
        preds = pred_deltas

        # Loss: MSE only on matched objects (correspondence >= 0)
        matched_mask = (correspondence >= 0) & valid  # (B, K)

        if matched_mask.sum() == 0:
            continue

        # MSE loss
        diff = preds - targets  # (B, K, 2)
        sq_error = (diff ** 2).sum(dim=-1)  # (B, K)
        loss = (sq_error * matched_mask.float()).sum() / matched_mask.sum()

        loss.backward()
        optimizer.step()

        # Metrics
        total_loss += loss.item() * matched_mask.sum().item()
        total_samples += matched_mask.sum().item()

        # Direction accuracy: did we predict the right direction of movement?
        # Use threshold to handle near-zero values
        with torch.no_grad():
            threshold = 0.001
            pred_dir = torch.where(torch.abs(preds) < threshold,
                                   torch.zeros_like(preds), torch.sign(preds))
            target_dir = torch.where(torch.abs(targets) < threshold,
                                     torch.zeros_like(targets), torch.sign(targets))
            correct = ((pred_dir == target_dir).all(dim=-1) & matched_mask).sum()
            total_correct_direction += correct.item()
            total_matched += matched_mask.sum().item()

    return {
        'loss': total_loss / max(total_samples, 1),
        'direction_acc': total_correct_direction / max(total_matched, 1),
    }


def evaluate(model: nn.Module, dataloader: torch.utils.data.DataLoader,
             device: torch.device) -> Dict:
    """Evaluate the model."""
    model.eval()

    total_loss = 0.0
    total_samples = 0
    total_matched = 0
    total_correct_direction = 0
    total_l1_error = 0.0

    with torch.no_grad():
        for batch in dataloader:
            (centroids, bboxes, areas, colors, valid, target_deltas,
             target_output_bboxes, correspondence,
             heights, widths) = [x.to(device) for x in batch]

            _, pred_deltas = model(centroids, areas, colors, valid)
            targets = target_deltas
            preds = pred_deltas

            matched_mask = (correspondence >= 0) & valid

            if matched_mask.sum() == 0:
                continue

            # MSE loss
            diff = preds - targets
            sq_error = (diff ** 2).sum(dim=-1)
            loss = (sq_error * matched_mask.float()).sum() / matched_mask.sum()

            # L1 error (more interpretable)
            l1_error = torch.abs(diff).sum(dim=-1)
            total_l1_error += (l1_error * matched_mask.float()).sum().item()

            total_loss += loss.item() * matched_mask.sum().item()
            total_samples += matched_mask.sum().item()

            # Direction accuracy (use threshold to handle near-zero values)
            threshold = 0.001
            pred_dir = torch.where(torch.abs(preds) < threshold,
                                   torch.zeros_like(preds), torch.sign(preds))
            target_dir = torch.where(torch.abs(targets) < threshold,
                                     torch.zeros_like(targets), torch.sign(targets))
            correct = ((pred_dir == target_dir).all(dim=-1) & matched_mask).sum()
            total_correct_direction += correct.item()
            total_matched += matched_mask.sum().item()

    return {
        'loss': total_loss / max(total_samples, 1),
        'l1_error': total_l1_error / max(total_samples, 1),
        'direction_acc': total_correct_direction / max(total_matched, 1),
    }


def evaluate_per_object_anchor(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    grid_size: int = 20,
    tolerance: float = 0.5
) -> Dict:
    """
    Evaluate per-object anchor model using pixel accuracy.

    Note: Since training uses delta targets (not framing-specific targets),
    we decode predictions as deltas for consistency.

    Args:
        model: PerObjectAnchorTransformModule
        dataloader: DataLoader
        device: Device
        grid_size: Grid dimension for coordinate scaling
        tolerance: Pixels within this distance count as correct

    Returns:
        Dict with metrics including pixel_accuracy
    """
    model.eval()

    total_loss = 0.0
    total_samples = 0
    total_correct = 0
    total_objects = 0
    total_pixel_error = 0.0

    with torch.no_grad():
        for batch in dataloader:
            (centroids, bboxes, areas, colors, valid, target_deltas,
             target_output_bboxes, correspondence,
             heights, widths) = [x.to(device) for x in batch]

            # Inference mode (no teacher forcing)
            _relations, predictions, _framing_weights, _ref_weights = model(
                centroids, bboxes, areas, colors, valid,
                target_output_bboxes=None
            )

            matched_mask = (correspondence >= 0) & valid

            if matched_mask.sum() == 0:
                continue

            # Compute MSE loss against delta targets for consistency
            diff = predictions - target_deltas
            sq_error = (diff ** 2).sum(dim=-1)
            loss = (sq_error * matched_mask.float()).sum() / matched_mask.sum()
            total_loss += loss.item() * matched_mask.sum().item()
            total_samples += matched_mask.sum().item()

            # Compute pixel accuracy
            # Since training uses delta targets, decode predictions as deltas:
            # predicted_position = input_centroid + prediction * grid_size
            input_centroids_pixels = centroids * grid_size  # (B, K, 2)
            predicted_positions = input_centroids_pixels + predictions * grid_size  # (B, K, 2)

            # Get target output centroids from bboxes (center of bbox)
            target_centroids = (target_output_bboxes[..., :2] + target_output_bboxes[..., 2:]) / 2 * grid_size

            # Compute pixel error
            pixel_errors = torch.sqrt(((predicted_positions - target_centroids) ** 2).sum(dim=-1))

            # Count correct predictions (within tolerance)
            correct = ((pixel_errors <= tolerance) & matched_mask).sum().item()
            total_correct += correct

            # Sum pixel errors for matched objects
            total_pixel_error += (pixel_errors * matched_mask.float()).sum().item()
            total_objects += matched_mask.sum().item()

    return {
        'loss': total_loss / max(total_samples, 1),
        'pixel_accuracy': total_correct / max(total_objects, 1),
        'mean_pixel_error': total_pixel_error / max(total_objects, 1),
    }


def compute_object_relative_target(
    obj_idx: int,
    output_centroids: torch.Tensor,
    reference_weights: torch.Tensor,
    grid_height: int,
    grid_width: int
) -> torch.Tensor:
    """
    Compute object_relative target dynamically based on learned reference.

    Args:
        obj_idx: Current object index
        output_centroids: (B, K, 2) output positions in pixel coordinates
        reference_weights: (B, K) soft weights over reference objects
        grid_height, grid_width: Grid dimensions for normalization

    Returns:
        target: (B, 2) normalized target relative to weighted reference
    """
    # Compute weighted reference position
    # reference_weights[:, :obj_idx] are the only non-zero weights (causal)
    ref_pos = (reference_weights.unsqueeze(-1) * output_centroids).sum(dim=1)  # (B, 2)

    # Target: output - reference (normalized)
    target_pos = output_centroids[:, obj_idx]  # (B, 2)
    relative = target_pos - ref_pos
    normalized = relative / torch.tensor([[grid_height, grid_width]],
                                          device=output_centroids.device).float()

    return normalized


def train_epoch_per_object_anchor(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device
) -> Dict:
    """
    Train for one epoch with per-object anchor framing.

    Uses the framings discovered during screening phase.

    Args:
        model: PerObjectAnchorTransformModule
        dataloader: DataLoader with per_object_anchor=True dataset
        optimizer: Optimizer
        device: Device

    Returns:
        Dict with training metrics
    """
    model.train()

    total_loss = 0.0
    total_samples = 0
    framing_counts = {f: 0 for f in FRAMINGS}

    for batch in dataloader:
        (centroids, bboxes, areas, colors, valid, target_deltas,
         target_output_bboxes, correspondence,
         heights, widths) = [x.to(device) for x in batch]

        optimizer.zero_grad()

        # Forward pass with discovered framings
        _relations, predictions, framing_weights, _reference_weights = model(
            centroids, bboxes, areas, colors, valid,
            target_output_bboxes=target_output_bboxes
        )

        # Get samples from dataloader to access multi-framing targets
        # Note: We need to access the dataset's samples through batch indices
        # For now, we'll compute loss based on the best matching target

        # Matched mask
        matched_mask = (correspondence >= 0) & valid  # (B, K)

        if matched_mask.sum() == 0:
            continue

        # Compute proper anchor-based targets for each object
        B, K, _ = predictions.shape
        anchor_module = model.anchor_module

        # Compute targets using discovered anchor points
        proper_targets = torch.zeros_like(predictions)

        # Convert to numpy for compute_framing_target
        centroids_np = centroids.cpu().numpy()
        bboxes_np = bboxes.cpu().numpy()
        target_output_bboxes_np = target_output_bboxes.cpu().numpy()

        with torch.no_grad():
            for obj_idx in range(K):
                framing_idx = anchor_module.get_framing_idx(obj_idx)
                framing = FRAMINGS[framing_idx]
                ref_idx = anchor_module.get_reference_idx(obj_idx)

                for b in range(B):
                    if not valid[b, obj_idx]:
                        continue

                    # Get grid size (assume square or use max)
                    # Use normalized coords * 30 as proxy for grid size
                    grid_size = 30

                    # Input centroid (denormalized)
                    input_centroid = centroids_np[b, obj_idx] * grid_size

                    # Output bbox and centroid (denormalized)
                    out_bbox = target_output_bboxes_np[b, obj_idx] * grid_size
                    output_centroid = np.array([
                        (out_bbox[0] + out_bbox[2]) / 2,
                        (out_bbox[1] + out_bbox[3]) / 2
                    ])

                    # Get anchor point info
                    source_anchor = None
                    target_anchor = None
                    source_output_bbox = out_bbox
                    reference_output_bbox = None
                    reference_output_centroid = None
                    grid_anchor = None
                    grid_target_position = None

                    if framing == 'object_relative' and anchor_module.has_anchor_points(obj_idx):
                        source_anchor, target_anchor, _ = anchor_module.get_anchor_points(obj_idx)
                        if ref_idx >= 0:
                            ref_bbox = target_output_bboxes_np[b, ref_idx] * grid_size
                            reference_output_bbox = ref_bbox
                            reference_output_centroid = np.array([
                                (ref_bbox[0] + ref_bbox[2]) / 2,
                                (ref_bbox[1] + ref_bbox[3]) / 2
                            ])

                    elif framing in ['grid_tl', 'grid_tr', 'grid_bl', 'grid_br']:
                        if anchor_module.has_grid_anchor(obj_idx):
                            grid_anchor, grid_target_position = anchor_module.get_grid_anchor(obj_idx)

                    # Compute proper target
                    target = compute_framing_target(
                        input_centroid, output_centroid, framing,
                        grid_size, grid_size,
                        reference_output_centroid=reference_output_centroid,
                        source_output_bbox=source_output_bbox,
                        reference_output_bbox=reference_output_bbox,
                        source_anchor=source_anchor,
                        target_anchor=target_anchor,
                        grid_anchor=grid_anchor,
                        grid_target_position=grid_target_position
                    )
                    proper_targets[b, obj_idx] = torch.tensor(target, dtype=torch.float32, device=device)

        # Compute loss against proper targets
        diff = predictions - proper_targets
        sq_error = (diff ** 2).sum(dim=-1)  # (B, K)
        loss = (sq_error * matched_mask.float()).sum() / matched_mask.sum()

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * matched_mask.sum().item()
        total_samples += matched_mask.sum().item()

        # Track framing selections
        with torch.no_grad():
            selected = framing_weights.argmax(dim=-1)  # (B, K)
            for b in range(B):
                for k in range(K):
                    if valid[b, k]:
                        framing_counts[FRAMINGS[selected[b, k].item()]] += 1

    # Compute framing distribution
    total_selections = sum(framing_counts.values())
    framing_dist = {f: c / max(total_selections, 1) for f, c in framing_counts.items()}

    return {
        'loss': total_loss / max(total_samples, 1),
        'framing_distribution': framing_dist,
    }


def evaluate_no_train(
    anchor_module: 'AnchorFramingModule',
    samples: List,
    grid_size: int = 30,
    tolerance: float = 0.5
) -> Dict:
    """
    Evaluate using screening-discovered offsets directly (no neural network).

    This function applies the offsets discovered during screening to predict
    object positions, without any gradient-based training.

    Args:
        anchor_module: AnchorFramingModule with discovered offsets
        samples: List of PositionSample objects to evaluate
        grid_size: Grid dimension for coordinate scaling
        tolerance: Pixels within this distance count as correct

    Returns:
        Dict with pixel_accuracy, mean_pixel_error, and per-object metrics
    """
    total_correct = 0
    total_samples = 0
    total_pixel_error = 0.0
    per_object_correct = {}
    per_object_total = {}

    for sample in samples:
        H = sample.grid_height or grid_size
        W = sample.grid_width or grid_size

        # Track decoded positions for reference by later objects
        decoded_positions = {}  # obj_idx -> (centroid, bbox)

        # Process objects in order (object 0 first, then 1, etc.)
        for obj_idx in range(len(sample.input_valid)):
            if not sample.input_valid[obj_idx]:
                continue
            if sample.correspondence[obj_idx] < 0:
                continue
            if not anchor_module.is_configured[obj_idx]:
                continue

            # Get input info
            input_centroid = sample.input_centroids[obj_idx] * grid_size
            input_bbox = sample.input_bboxes[obj_idx] * grid_size

            # Get source size from input bbox
            src_height = max(1, int(round(input_bbox[2] - input_bbox[0])) + 1)
            src_width = max(1, int(round(input_bbox[3] - input_bbox[1])) + 1)
            source_size = (src_height, src_width)

            # Get reference info if needed
            ref_idx = anchor_module.get_reference_idx(obj_idx)
            reference_position = None
            reference_bbox = None
            if ref_idx >= 0 and ref_idx in decoded_positions:
                reference_position, reference_bbox = decoded_positions[ref_idx]

            # Apply screening offset
            predicted_centroid = apply_screening_offset(
                anchor_module, obj_idx, input_centroid,
                H, W, source_size, reference_position, reference_bbox
            )

            # Get ground truth
            output_bbox = sample.target_output_bboxes[obj_idx] * grid_size
            gt_centroid = np.array([
                (output_bbox[0] + output_bbox[2]) / 2,
                (output_bbox[1] + output_bbox[3]) / 2
            ])

            # Compute error
            error = np.sqrt(((predicted_centroid - gt_centroid) ** 2).sum())
            is_correct = error <= tolerance

            # Update metrics
            total_samples += 1
            total_pixel_error += error
            if is_correct:
                total_correct += 1

            # Per-object tracking
            if obj_idx not in per_object_correct:
                per_object_correct[obj_idx] = 0
                per_object_total[obj_idx] = 0
            per_object_total[obj_idx] += 1
            if is_correct:
                per_object_correct[obj_idx] += 1

            # Store decoded position for reference by later objects
            # For the decoded bbox, we need to compute it from the predicted centroid
            pred_top_left = (predicted_centroid[0] - (src_height - 1) / 2.0,
                            predicted_centroid[1] - (src_width - 1) / 2.0)
            pred_bbox = np.array([pred_top_left[0], pred_top_left[1],
                                  pred_top_left[0] + src_height - 1,
                                  pred_top_left[1] + src_width - 1])
            decoded_positions[obj_idx] = (predicted_centroid, pred_bbox)

    # Compute per-object accuracies
    per_object_acc = {}
    for obj_idx in per_object_total:
        per_object_acc[obj_idx] = per_object_correct[obj_idx] / per_object_total[obj_idx]

    return {
        'pixel_accuracy': total_correct / max(total_samples, 1),
        'mean_pixel_error': total_pixel_error / max(total_samples, 1),
        'total_correct': total_correct,
        'total_samples': total_samples,
        'per_object_accuracy': per_object_acc,
    }


class SimpleObjectPredictor(nn.Module):
    """
    Tiny MLP that predicts position for a single object.
    Used during screening to test each framing independently.
    """

    def __init__(self, input_dim: int = 2, hidden_dim: int = 32, output_dim: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)


def compute_per_object_pixel_accuracy(
    predictions: np.ndarray,
    _targets: np.ndarray,
    input_centroids: np.ndarray,
    output_centroids: np.ndarray,
    framing: str,
    grid_height: int,
    grid_width: int,
    reference_output_centroid: Optional[np.ndarray] = None,
    tolerance: float = 1.0
) -> Tuple[float, float]:
    """
    Compute pixel accuracy for a single object across all examples.

    Args:
        predictions: (N,) or (N, 2) predicted values in framing space
        targets: (N, 2) target values in framing space
        input_centroids: (N, 2) input positions in pixels
        output_centroids: (N, 2) ground truth output positions in pixels
        framing: The framing type used
        grid_height, grid_width: Grid dimensions
        reference_output_centroid: (N, 2) reference positions for object_relative
        tolerance: How close (in pixels) counts as correct

    Returns:
        accuracy: fraction of correct placements
        mean_error: mean pixel distance error
    """
    N = len(predictions)
    if predictions.ndim == 1:
        predictions = predictions.reshape(-1, 2)

    correct = 0
    total_error = 0.0

    for i in range(N):
        pred = predictions[i]
        input_centroid = input_centroids[i]

        # Decode prediction to pixel position
        if framing == 'object_relative' and reference_output_centroid is not None:
            ref_pos = reference_output_centroid[i]
        else:
            ref_pos = None

        decoded_pos = decode_framing_prediction(
            pred, framing, input_centroid,
            grid_height, grid_width,
            reference_position=ref_pos
        )

        true_pos = output_centroids[i]
        error = np.sqrt(((decoded_pos - true_pos) ** 2).sum())
        total_error += error

        if error <= tolerance:
            correct += 1

    accuracy = correct / N if N > 0 else 0.0
    mean_error = total_error / N if N > 0 else 0.0

    return accuracy, mean_error


def discover_anchor_points_for_object_relative(
    samples: List,
    obj_idx: int,
    ref_idx: int,
    grid_size: int = 30
) -> Optional[DiscoveredRelation]:
    """
    Discover the best anchor point relationship for object_relative framing.

    After screening determines that object_relative is the best framing for
    obj_idx with reference ref_idx, this function finds WHICH anchor points
    give the most consistent offset across examples.

    Args:
        samples: List of PositionSample objects
        obj_idx: The object index we're positioning
        ref_idx: The reference object index
        grid_size: Grid size for denormalizing bboxes

    Returns:
        DiscoveredRelation with the best anchor pair, or None if insufficient data
    """
    example_pairs = []

    for sample in samples:
        # Check both objects are valid and have correspondences
        if not sample.input_valid[obj_idx] or not sample.input_valid[ref_idx]:
            continue
        if sample.correspondence[obj_idx] < 0 or sample.correspondence[ref_idx] < 0:
            continue

        # Get grid size for denormalization
        # IMPORTANT: Must use GRID_SIZE (30) since that's what was used for normalization
        # Using sample.grid_height/width would cause scaling mismatch
        gs = grid_size  # Use the parameter (defaults to 30, matching normalization)

        # Get OUTPUT bounding boxes (where objects end up)
        # These are stored in target_output_bboxes as normalized [min_r, min_c, max_r, max_c]
        src_bbox = sample.target_output_bboxes[obj_idx] * gs
        ref_bbox = sample.target_output_bboxes[ref_idx] * gs

        # Convert to TestObject format: top_left=(row, col), size=(height, width)
        src_top_left = (int(round(src_bbox[0])), int(round(src_bbox[1])))
        src_height = max(1, int(round(src_bbox[2] - src_bbox[0])) + 1)
        src_width = max(1, int(round(src_bbox[3] - src_bbox[1])) + 1)

        ref_top_left = (int(round(ref_bbox[0])), int(round(ref_bbox[1])))
        ref_height = max(1, int(round(ref_bbox[2] - ref_bbox[0])) + 1)
        ref_width = max(1, int(round(ref_bbox[3] - ref_bbox[1])) + 1)

        source_obj = TestObject(
            top_left=src_top_left,
            size=(src_height, src_width),
            object_id=obj_idx
        )
        target_obj = TestObject(
            top_left=ref_top_left,
            size=(ref_height, ref_width),
            object_id=ref_idx
        )

        example_pairs.append(ExamplePair(source=source_obj, target=target_obj))

    if len(example_pairs) < 2:
        return None

    # Debug: print the bboxes being used
    print(f"    [Anchor Discovery Debug] obj={obj_idx}, ref={ref_idx}")
    for i, pair in enumerate(example_pairs):
        print(f"      Example {i}: src_tl={pair.source.top_left}, src_size={pair.source.size}, "
              f"ref_tl={pair.target.top_left}, ref_size={pair.target.size}")

    # Discover the best anchor point relationship
    discovered = discover_relation(example_pairs, top_k=1)
    return discovered[0] if discovered else None


@dataclass
class GridAnchorDiscovery:
    """Result of grid anchor point discovery."""
    anchor: AnchorPoint
    target_position: Tuple[float, float]  # (row, col) absolute position
    variance: float  # variance across examples (lower is better)


def discover_grid_anchor_point(
    samples: List,
    obj_idx: int,
    grid_size: int = 30
) -> Optional[GridAnchorDiscovery]:
    """
    Discover the best anchor point for grid-relative positioning.

    For each of the 9 anchor points on the object, compute the absolute
    position across all examples. The anchor point with lowest variance
    (most consistent absolute position) is the best choice.

    Args:
        samples: List of PositionSample objects
        obj_idx: The object index we're positioning
        grid_size: Grid size for denormalizing bboxes

    Returns:
        GridAnchorDiscovery with the best anchor and target position, or None if insufficient data
    """
    # Collect output bboxes for this object across all examples
    output_bboxes = []

    for sample in samples:
        if not sample.input_valid[obj_idx]:
            continue
        if sample.correspondence[obj_idx] < 0:
            continue

        # Get grid size for denormalization
        # IMPORTANT: Must use grid_size parameter (30) since that's what was used for normalization
        gs = grid_size

        # Get OUTPUT bounding box (where object ends up)
        bbox = sample.target_output_bboxes[obj_idx] * gs
        top_left = (int(round(bbox[0])), int(round(bbox[1])))
        height = max(1, int(round(bbox[2] - bbox[0])) + 1)
        width = max(1, int(round(bbox[3] - bbox[1])) + 1)

        output_bboxes.append((top_left, (height, width)))

    if len(output_bboxes) < 2:
        return None

    # Test each anchor point
    best_anchor = None
    best_variance = float('inf')
    best_mean_pos = None

    for anchor in ALL_ANCHORS:
        # Compute anchor position for each example
        positions = []
        for top_left, size in output_bboxes:
            anchor_pos = get_anchor_position(top_left, size, anchor)
            positions.append(anchor_pos)

        positions = np.array(positions)

        # Compute variance (sum of row variance + col variance)
        variance = positions[:, 0].var() + positions[:, 1].var()

        if variance < best_variance:
            best_variance = variance
            best_anchor = anchor
            best_mean_pos = (positions[:, 0].mean(), positions[:, 1].mean())

    if best_anchor is None:
        return None

    return GridAnchorDiscovery(
        anchor=best_anchor,
        target_position=best_mean_pos,
        variance=best_variance
    )


def compute_mean_delta_for_object(
    samples: List,
    obj_idx: int,
    grid_size: int = 30
) -> Optional[Tuple[float, float]]:
    """
    Compute the mean delta (output - input) for an object across all training examples.

    Used in no-train mode to directly apply the discovered mean delta for delta-framed objects.

    Args:
        samples: List of PositionSample objects
        obj_idx: The object index to compute mean delta for
        grid_size: Grid size for denormalizing (default 30)

    Returns:
        Mean delta as (row, col) tuple in pixels, or None if insufficient data
    """
    deltas = []

    for sample in samples:
        # Check object is valid and has correspondence
        if not sample.input_valid[obj_idx]:
            continue
        if sample.correspondence[obj_idx] < 0:
            continue

        # target_deltas is already in normalized form (delta / grid_size)
        # We want pixel deltas, so multiply by grid_size
        delta_normalized = sample.target_deltas[obj_idx]
        delta_pixels = delta_normalized * grid_size

        deltas.append(delta_pixels)

    if len(deltas) < 1:
        return None, float('inf')

    deltas = np.array(deltas)
    mean_delta = (deltas[:, 0].mean(), deltas[:, 1].mean())
    # Compute variance (sum of row variance + col variance)
    variance = deltas[:, 0].var() + deltas[:, 1].var()
    return mean_delta, variance


class AnchorScreeningTrainer:
    """
    Screening to discover best framing per object index.

    For each object index:
    1. Try each of the 6 framings independently
    2. Train a simple model for each framing
    3. Measure pixel accuracy
    4. Select the framing with best pixel accuracy

    For object_relative, also tries different reference objects.
    """

    def __init__(self, screening_epochs: int = 100, lr: float = 0.01, verbose: bool = False):
        self.screening_epochs = screening_epochs
        self.lr = lr
        self.verbose = verbose
        self.results = {}  # obj_idx -> {framing -> (accuracy, mean_error)}

    def screen_object(
        self,
        obj_idx: int,
        samples: List,
        device: torch.device
    ) -> Tuple[str, int, float, float, Optional[DiscoveredRelation], Optional[GridAnchorDiscovery]]:
        """
        Screen all framings for a single object index.

        Args:
            obj_idx: Which object index to screen
            samples: List of PositionSample objects
            device: Torch device

        Returns:
            (best_framing, best_reference, best_accuracy, best_error,
             discovered_anchor_relation, discovered_grid_anchor)
            discovered_anchor_relation is set when object_relative is the best framing
            discovered_grid_anchor is set when a grid framing is the best
        """
        # Collect data for this object across all samples
        input_centroids = []
        output_centroids = []
        grid_heights = []
        grid_widths = []
        valid_samples = []

        for sample in samples:
            if not sample.input_valid[obj_idx]:
                continue
            if sample.correspondence[obj_idx] < 0:
                continue

            # Get grid size for denormalization
            # IMPORTANT: Must use GRID_SIZE (30) since that's what was used for normalization
            gs = GRID_SIZE

            # Input centroid in pixels
            in_cent = sample.input_centroids[obj_idx] * gs

            # Output centroid in pixels
            out_idx = sample.correspondence[obj_idx]
            if sample.output_centroids is not None:
                out_cent = sample.output_centroids[obj_idx]
            else:
                # Fallback: compute from target
                out_cent = in_cent + sample.target_deltas[obj_idx] * gs

            input_centroids.append(in_cent)
            output_centroids.append(out_cent)
            grid_heights.append(gs)  # Use consistent GRID_SIZE
            grid_widths.append(gs)   # Use consistent GRID_SIZE
            valid_samples.append(sample)

        if len(input_centroids) == 0:
            return 'delta', -1, 0.0, float('inf'), None, None

        input_centroids = np.array(input_centroids)
        output_centroids = np.array(output_centroids)
        N = len(input_centroids)

        # Use consistent grid size for simplicity
        H = int(np.mean(grid_heights))
        W = int(np.mean(grid_widths))

        best_framing = 'delta'
        best_reference = -1
        best_accuracy = -1.0
        best_error = float('inf')

        # Try each non-relative framing
        for framing in ['delta', 'grid_tl', 'grid_tr', 'grid_bl', 'grid_br']:
            # Compute targets for this framing
            targets = np.zeros((N, 2), dtype=np.float32)
            for i in range(N):
                targets[i] = compute_framing_target(
                    input_centroids[i], output_centroids[i],
                    framing, H, W
                )

            # Train simple predictor
            accuracy, mean_error = self._train_and_evaluate(
                input_centroids, targets, output_centroids,
                framing, H, W, device
            )

            if self.verbose:
                print(f"    {framing}: accuracy={accuracy:.1%}, error={mean_error:.2f}px")

            if accuracy > best_accuracy or (accuracy == best_accuracy and mean_error < best_error):
                best_accuracy = accuracy
                best_error = mean_error
                best_framing = framing
                best_reference = -1

        # Try object_relative with different reference objects
        if obj_idx > 0:
            for ref_idx in range(obj_idx):
                # Collect reference object output positions
                ref_output_centroids = []
                valid_for_ref = True

                for sample in valid_samples:
                    if not sample.input_valid[ref_idx]:
                        valid_for_ref = False
                        break
                    if sample.correspondence[ref_idx] < 0:
                        valid_for_ref = False
                        break

                    # Use GRID_SIZE for consistent denormalization
                    gs = GRID_SIZE

                    if sample.output_centroids is not None:
                        ref_cent = sample.output_centroids[ref_idx]
                    else:
                        ref_in = sample.input_centroids[ref_idx] * gs
                        ref_cent = ref_in + sample.target_deltas[ref_idx] * gs

                    ref_output_centroids.append(ref_cent)

                if not valid_for_ref or len(ref_output_centroids) != N:
                    continue

                ref_output_centroids = np.array(ref_output_centroids)

                # Compute object_relative targets
                targets = np.zeros((N, 2), dtype=np.float32)
                for i in range(N):
                    targets[i] = compute_framing_target(
                        input_centroids[i], output_centroids[i],
                        'object_relative', H, W,
                        reference_output_centroid=ref_output_centroids[i]
                    )

                # Train and evaluate
                accuracy, mean_error = self._train_and_evaluate(
                    input_centroids, targets, output_centroids,
                    'object_relative', H, W, device,
                    reference_output_centroids=ref_output_centroids
                )

                if self.verbose:
                    print(f"    object_relative (ref={ref_idx}): accuracy={accuracy:.1%}, error={mean_error:.2f}px")
                    # Show positions and targets for debugging
                    print(f"      Output centroids: {output_centroids.tolist()}")
                    print(f"      Ref centroids: {ref_output_centroids.tolist()}")
                    print(f"      Targets (centroid-based): {targets.tolist()}")
                    # Also try anchor discovery to see if anchor-based would be better
                    test_discovery = discover_anchor_points_for_object_relative(
                        valid_samples, obj_idx, ref_idx
                    )
                    if test_discovery:
                        print(f"      Anchor discovery: {test_discovery.relation.source_anchor.value.upper()}->"
                              f"{test_discovery.relation.target_anchor.value.upper()}, "
                              f"offset={test_discovery.relation.offset}, var={test_discovery.variance:.4f}")

                if accuracy > best_accuracy or (accuracy == best_accuracy and mean_error < best_error):
                    best_accuracy = accuracy
                    best_error = mean_error
                    best_framing = 'object_relative'
                    best_reference = ref_idx

        # Check if any object_relative has a near-perfect anchor relationship (var ≈ 0)
        # If so, prefer it over centroid-based framings like delta
        ANCHOR_VAR_THRESHOLD = 0.01  # Near-zero variance indicates perfect anchor relationship
        best_anchor_discovery = None
        best_anchor_ref = -1
        best_anchor_variance = float('inf')

        if obj_idx > 0:
            for ref_idx in range(obj_idx):
                discovery = discover_anchor_points_for_object_relative(
                    valid_samples, obj_idx, ref_idx
                )
                if discovery and discovery.variance < best_anchor_variance:
                    best_anchor_variance = discovery.variance
                    best_anchor_discovery = discovery
                    best_anchor_ref = ref_idx

            # If we found a near-perfect anchor relationship, prefer object_relative
            if best_anchor_variance < ANCHOR_VAR_THRESHOLD:
                if self.verbose:
                    print(f"    ** Preferring object_relative (ref={best_anchor_ref}) due to perfect anchor relationship (var={best_anchor_variance:.4f})")
                best_framing = 'object_relative'
                best_reference = best_anchor_ref

        # If object_relative is the best framing, discover which anchor points to use
        discovered_relation = None
        if best_framing == 'object_relative' and best_reference >= 0:
            discovered_relation = discover_anchor_points_for_object_relative(
                samples, obj_idx, best_reference
            )

        # If a grid framing is the best, discover which anchor point to use
        discovered_grid_anchor = None
        if best_framing in ['grid_tl', 'grid_tr', 'grid_bl', 'grid_br']:
            discovered_grid_anchor = discover_grid_anchor_point(samples, obj_idx)

        return best_framing, best_reference, best_accuracy, best_error, discovered_relation, discovered_grid_anchor

    def _train_and_evaluate(
        self,
        input_centroids: np.ndarray,
        targets: np.ndarray,
        output_centroids: np.ndarray,
        framing: str,
        grid_height: int,
        grid_width: int,
        device: torch.device,
        reference_output_centroids: Optional[np.ndarray] = None
    ) -> Tuple[float, float]:
        """
        Train a simple model and evaluate pixel accuracy.
        """
        N = len(input_centroids)

        # Normalize inputs for training
        X = torch.tensor(input_centroids / 20.0, dtype=torch.float32, device=device)
        Y = torch.tensor(targets, dtype=torch.float32, device=device)

        model = SimpleObjectPredictor(input_dim=2, hidden_dim=32, output_dim=2).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)

        # Train
        model.train()
        for _ in range(self.screening_epochs):
            optimizer.zero_grad()
            pred = model(X)
            loss = ((pred - Y) ** 2).mean()
            loss.backward()
            optimizer.step()

        # Evaluate
        model.eval()
        with torch.no_grad():
            predictions = model(X).cpu().numpy()

        accuracy, mean_error = compute_per_object_pixel_accuracy(
            predictions, targets, input_centroids, output_centroids,
            framing, grid_height, grid_width,
            reference_output_centroid=reference_output_centroids
        )

        return accuracy, mean_error

    def run_screening(
        self,
        dataset,
        anchor_module: AnchorFramingModule,
        device: torch.device
    ) -> Dict:
        """
        Run screening for all object indices present in the dataset.

        Args:
            dataset: PositionDataset with samples
            anchor_module: AnchorFramingModule to configure with results
            device: Torch device

        Returns:
            Dict with screening results
        """
        print(f"\n{'='*60}")
        print("ANCHOR FRAMING SCREENING")
        print(f"{'='*60}")
        print(f"Training {self.screening_epochs} epochs per framing to measure pixel accuracy\n")

        # Find max object index used
        max_obj_idx = 0
        for sample in dataset.samples:
            for i in range(MAX_OBJECTS):
                if sample.input_valid[i] and sample.correspondence[i] >= 0:
                    max_obj_idx = max(max_obj_idx, i)

        results = {}

        for obj_idx in range(max_obj_idx + 1):
            # Check if this object index appears in any sample
            has_samples = any(
                sample.input_valid[obj_idx] and sample.correspondence[obj_idx] >= 0
                for sample in dataset.samples
            )

            if not has_samples:
                continue

            print(f"Screening object {obj_idx}...")

            best_framing, best_ref, best_acc, best_err, discovered_rel, discovered_grid = self.screen_object(
                obj_idx, dataset.samples, device
            )

            # Extract anchor point info for object_relative
            source_anchor = None
            target_anchor = None
            anchor_offset = None
            if discovered_rel is not None:
                source_anchor = discovered_rel.relation.source_anchor
                target_anchor = discovered_rel.relation.target_anchor
                anchor_offset = discovered_rel.relation.offset

            # Extract grid anchor info for grid framings
            grid_anchor = None
            grid_target_position = None
            grid_variance = float('inf')
            if discovered_grid is not None:
                grid_anchor = discovered_grid.anchor
                grid_target_position = discovered_grid.target_position
                grid_variance = discovered_grid.variance

            # Determine variance based on framing type
            framing_variance = float('inf')
            if best_framing == 'object_relative' and discovered_rel is not None:
                framing_variance = discovered_rel.variance
            elif best_framing in ['grid_tl', 'grid_tr', 'grid_bl', 'grid_br'] and discovered_grid is not None:
                framing_variance = grid_variance

            # For delta framing, compute and store the mean delta for no-train mode
            mean_delta = None
            delta_variance = float('inf')
            if best_framing == 'delta':
                mean_delta, delta_variance = compute_mean_delta_for_object(dataset.samples, obj_idx)
                if mean_delta is not None:
                    anchor_module.set_delta_offset(obj_idx, mean_delta)
                framing_variance = delta_variance

            # Store in anchor module
            anchor_module.set_framing(
                obj_idx, best_framing, best_ref, best_acc,
                source_anchor=source_anchor,
                target_anchor=target_anchor,
                anchor_offset=anchor_offset,
                grid_anchor=grid_anchor,
                grid_target_position=grid_target_position,
                variance=framing_variance
            )

            results[obj_idx] = {
                'framing': best_framing,
                'reference': best_ref,
                'accuracy': best_acc,
                'mean_error': best_err,
                'discovered_relation': discovered_rel,
                'discovered_grid_anchor': discovered_grid
            }

            if best_framing == 'object_relative':
                anchor_desc = ""
                if discovered_rel is not None:
                    src = discovered_rel.relation.source_anchor.value.upper()
                    tgt = discovered_rel.relation.target_anchor.value.upper()
                    off = discovered_rel.relation.offset
                    anchor_desc = f" [{src}->{tgt}, offset={off}, var={discovered_rel.variance:.4f}]"
                print(f"  -> {best_framing} (ref: object {best_ref}){anchor_desc} "
                      f"accuracy={best_acc:.1%}, error={best_err:.2f}px")
            elif best_framing in ['grid_tl', 'grid_tr', 'grid_bl', 'grid_br']:
                grid_desc = ""
                if discovered_grid is not None:
                    anc = discovered_grid.anchor.value.upper()
                    pos = discovered_grid.target_position
                    grid_desc = f" [anchor={anc} -> pos=({pos[0]:.1f},{pos[1]:.1f}), var={discovered_grid.variance:.4f}]"
                print(f"  -> {best_framing}{grid_desc} accuracy={best_acc:.1%}, error={best_err:.2f}px")
            elif best_framing == 'delta':
                delta_desc = ""
                if mean_delta is not None:
                    delta_desc = f" [mean_delta=({mean_delta[0]:.2f},{mean_delta[1]:.2f})px, var={delta_variance:.4f}]"
                print(f"  -> {best_framing}{delta_desc} accuracy={best_acc:.1%}, error={best_err:.2f}px")
            else:
                print(f"  -> {best_framing} accuracy={best_acc:.1%}, error={best_err:.2f}px")

        # Second pass: inherit pattern for objects with insufficient data
        # If an object couldn't validate anchor relationships (< 2 examples),
        # but the previous object uses object_relative with a good anchor,
        # assume this object follows the same pattern
        for obj_idx in sorted(results.keys()):
            if obj_idx == 0:
                continue

            result = results[obj_idx]
            prev_result = results.get(obj_idx - 1)

            # Check if this object could benefit from pattern inheritance
            if (result['framing'] != 'object_relative' and
                result['discovered_relation'] is None and
                prev_result is not None and
                prev_result['framing'] == 'object_relative' and
                prev_result['discovered_relation'] is not None and
                prev_result['discovered_relation'].variance < 0.01):

                # Inherit the pattern: use object_relative to previous object
                # with the same anchor relationship
                inherited_rel = prev_result['discovered_relation']
                print(f"\n  ** Object {obj_idx}: Inheriting pattern from object {obj_idx-1} "
                      f"(insufficient data to validate, but pattern is consistent)")
                print(f"     Using object_relative (ref={obj_idx-1}) with "
                      f"{inherited_rel.relation.source_anchor.value.upper()}->"
                      f"{inherited_rel.relation.target_anchor.value.upper()}")

                # Update anchor module
                anchor_module.set_framing(
                    obj_idx, 'object_relative', obj_idx - 1, result['accuracy'],
                    source_anchor=inherited_rel.relation.source_anchor,
                    target_anchor=inherited_rel.relation.target_anchor,
                    anchor_offset=inherited_rel.relation.offset
                )

                # Update results
                results[obj_idx]['framing'] = 'object_relative'
                results[obj_idx]['reference'] = obj_idx - 1
                results[obj_idx]['discovered_relation'] = inherited_rel

        self.results = results
        return results


# =============================================================================
# Visualization
# =============================================================================

def visualize_predictions(model: nn.Module, dataset: PositionDataset,
                          device: torch.device, num_samples: int = 5,
                          title: str = "Position Prediction Visualization"):
    """Visualize position predictions."""
    model.eval()

    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)

    indices = random.sample(range(len(dataset)), min(num_samples, len(dataset)))

    for idx in indices:
        sample = dataset.samples[idx]

        # Get predictions
        centroids = torch.from_numpy(sample.input_centroids).float().unsqueeze(0).to(device)
        areas = torch.from_numpy(sample.input_areas).float().unsqueeze(0).to(device)
        colors = torch.from_numpy(sample.input_colors).long().unsqueeze(0).to(device)
        valid = torch.from_numpy(sample.input_valid).bool().unsqueeze(0).to(device)

        with torch.no_grad():
            _, preds = model(centroids, areas, colors, valid)
            preds = preds.squeeze(0).cpu().numpy()
            targets = sample.target_deltas
            label = "delta"

        correspondence = sample.correspondence

        print(f"\nPuzzle: {sample.puzzle_id}, Example: {sample.example_idx}")
        print("-" * 40)

        for i in range(MAX_OBJECTS):
            if not sample.input_valid[i]:
                continue
            if correspondence[i] < 0:
                continue

            pred = preds[i]
            target = targets[i]
            color = sample.input_colors[i]
            pos = sample.input_centroids[i]

            # Direction comparison (use threshold to handle near-zero values)
            threshold = 0.001
            pred_dir = np.where(np.abs(pred) < threshold, 0, np.sign(pred))
            target_dir = np.where(np.abs(target) < threshold, 0, np.sign(target))
            dir_match = "✓" if np.allclose(pred_dir, target_dir) else "✗"

            color_name = ARC_COLOR_NAMES[color] if 0 <= color < len(ARC_COLOR_NAMES) else f"color{color}"
            print(f"  Object {i} ({color_name}, pos=[{pos[0]:.2f}, {pos[1]:.2f}]):")
            print(f"    Target {label}: [{target[0]:+.3f}, {target[1]:+.3f}]")
            print(f"    Pred {label}:   [{pred[0]:+.3f}, {pred[1]:+.3f}] {dir_match}")


# ARC color map
ARC_COLORS = [
    '#000000',  # 0: black
    '#0074D9',  # 1: blue
    '#FF4136',  # 2: red
    '#2ECC40',  # 3: green
    '#FFDC00',  # 4: yellow
    '#AAAAAA',  # 5: gray
    '#F012BE',  # 6: magenta
    '#FF851B',  # 7: orange
    '#7FDBFF',  # 8: cyan
    '#870C25',  # 9: brown
]

ARC_COLOR_NAMES = [
    'black', 'blue', 'red', 'green', 'yellow',
    'gray', 'magenta', 'orange', 'cyan', 'brown'
]


class InteractiveGridViewer:
    """Interactive viewer for ARC puzzle grids with keyboard navigation."""

    def __init__(self, puzzles: Dict, puzzle_id: str, model: nn.Module,
                 device: torch.device, use_color_only: bool = False,
                 ordering_strategy: Optional[str] = None, predict_absolute: bool = False,
                 per_object_anchor: bool = False,
                 verbose: bool = False,
                 selection_criterion: Optional[str] = None,
                 selection_rule: Optional[str] = None,
                 no_train: bool = False):
        self.puzzle_id = puzzle_id
        self.model = model
        self.device = device
        self.use_color_only = use_color_only
        self.ordering_strategy = ordering_strategy
        self.predict_absolute = predict_absolute
        self.per_object_anchor = per_object_anchor
        self.verbose = verbose
        self.selection_criterion = selection_criterion
        self.selection_rule = selection_rule
        self.no_train = no_train

        # Create color map
        self.cmap = mcolors.ListedColormap(ARC_COLORS)
        bounds = np.arange(-0.5, 10.5, 1)
        self.norm = mcolors.BoundaryNorm(bounds, self.cmap.N)

        puzzle = puzzles[puzzle_id]
        train_examples = puzzle.get('train', [])
        test_examples = puzzle.get('test', [])

        # Build list of all examples to display
        self.examples = []
        for i, ex in enumerate(train_examples):
            input_grid = np.array(ex['input'])
            output_grid = np.array(ex['output'])
            # Compute prediction for training examples too
            output_size = output_grid.shape
            predicted = apply_predicted_transformation(
                input_grid, model, device, use_color_only, ordering_strategy,
                self.predict_absolute, self.per_object_anchor,
                verbose=False, output_size=output_size,
                selection_criterion=self.selection_criterion, selection_rule=self.selection_rule,
                no_train=self.no_train
            )
            self.examples.append({
                'type': 'train',
                'index': i + 1,
                'input': input_grid,
                'output': output_grid,
                'predicted': predicted,
            })
        for i, ex in enumerate(test_examples):
            entry = {
                'type': 'test',
                'index': i + 1,
                'input': np.array(ex['input']),
                'output': np.array(ex['output']) if 'output' in ex else None,
            }
            # Compute prediction for test
            if verbose:
                print(f"\n{'#'*60}")
                print(f"# Test Example {i+1}: Computing predicted transformation")
                print(f"{'#'*60}")
            # Use output size if ground truth is available
            output_size = entry['output'].shape if entry['output'] is not None else None
            entry['predicted'] = apply_predicted_transformation(
                entry['input'], model, device, use_color_only, ordering_strategy,
                self.predict_absolute, self.per_object_anchor,
                verbose=verbose, output_size=output_size,
                selection_criterion=self.selection_criterion, selection_rule=self.selection_rule,
                no_train=self.no_train
            )
            self.examples.append(entry)

        self.current_idx = 0
        self.fig = None
        self.axes = None

    def show(self):
        """Display the interactive viewer."""
        if len(self.examples) == 0:
            print("No examples to display")
            return

        self.fig, self.axes = plt.subplots(1, 3, figsize=(12, 5))
        self.fig.canvas.mpl_connect('key_press_event', self._on_key)

        self._update_display()
        plt.show()

    def _update_display(self):
        """Update the display for the current example."""
        for ax in self.axes:
            ax.clear()

        ex = self.examples[self.current_idx]
        is_test = ex['type'] == 'test'

        # Title
        self.fig.suptitle(
            f"Puzzle: {self.puzzle_id}  |  "
            f"{ex['type'].capitalize()} {ex['index']}  |  "
            f"[{self.current_idx + 1}/{len(self.examples)}]  "
            f"(← → to navigate, q to quit)",
            fontsize=12, fontweight='bold'
        )

        # Input
        self.axes[0].imshow(ex['input'], cmap=self.cmap, norm=self.norm)
        self.axes[0].set_title('Input')
        self.axes[0].set_xticks([])
        self.axes[0].set_yticks([])
        edge_color = 'blue' if is_test else 'black'
        for spine in self.axes[0].spines.values():
            spine.set_edgecolor(edge_color)
            spine.set_linewidth(3 if is_test else 2)

        # Output / True Output
        if ex['output'] is not None:
            self.axes[1].imshow(ex['output'], cmap=self.cmap, norm=self.norm)
            self.axes[1].set_title('True Output' if is_test else 'Output')
            edge_color = 'green' if is_test else 'black'
            for spine in self.axes[1].spines.values():
                spine.set_edgecolor(edge_color)
                spine.set_linewidth(3 if is_test else 2)
        else:
            self.axes[1].text(0.5, 0.5, '(No ground truth)',
                              ha='center', va='center', fontsize=12)
            self.axes[1].set_title('True Output')
        self.axes[1].set_xticks([])
        self.axes[1].set_yticks([])

        # Predicted (for both train and test)
        self.axes[2].imshow(ex['predicted'], cmap=self.cmap, norm=self.norm)
        self.axes[2].set_title('Predicted')
        for spine in self.axes[2].spines.values():
            spine.set_edgecolor('orange')
            spine.set_linewidth(3 if is_test else 2)
        self.axes[2].set_xticks([])
        self.axes[2].set_yticks([])

        self.fig.tight_layout()
        self.fig.canvas.draw()

    def _on_key(self, event):
        """Handle keyboard navigation."""
        if event.key in ['right', 'n', ' ']:
            self.current_idx = (self.current_idx + 1) % len(self.examples)
            self._update_display()
        elif event.key in ['left', 'p']:
            self.current_idx = (self.current_idx - 1) % len(self.examples)
            self._update_display()
        elif event.key == 'home':
            self.current_idx = 0
            self._update_display()
        elif event.key == 'end':
            self.current_idx = len(self.examples) - 1
            self._update_display()
        elif event.key in ['q', 'escape']:
            plt.close(self.fig)


def visualize_grids(puzzles: Dict, puzzle_id: str, model: nn.Module,
                    device: torch.device, use_color_only: bool = False,
                    ordering_strategy: Optional[str] = None, predict_absolute: bool = False,
                    per_object_anchor: bool = False,
                    verbose: bool = False,
                    selection_criterion: Optional[str] = None,
                    selection_rule: Optional[str] = None,
                    no_train: bool = False):
    """
    Visualize the training pairs and test prediction with actual grid shapes.
    Uses an interactive viewer with keyboard navigation.
    """
    if puzzle_id not in puzzles:
        print(f"Puzzle {puzzle_id} not found")
        return

    # Show interactive viewer
    print("\nInteractive Viewer Controls:")
    print("  → / n / Space : Next example")
    print("  ← / p         : Previous example")
    print("  Home          : First example")
    print("  End           : Last example")
    print("  q / Escape    : Quit")

    viewer = InteractiveGridViewer(puzzles, puzzle_id, model, device, use_color_only,
                                   ordering_strategy, predict_absolute,
                                   per_object_anchor, verbose=verbose,
                                   selection_criterion=selection_criterion, selection_rule=selection_rule,
                                   no_train=no_train)
    viewer.show()


def save_overview_image(puzzles: Dict, puzzle_id: str, model: nn.Module,
                        device: torch.device, use_color_only: bool = False,
                        ordering_strategy: Optional[str] = None, predict_absolute: bool = False,
                        per_object_anchor: bool = False,
                        selection_criterion: Optional[str] = None,
                        selection_rule: Optional[str] = None,
                        no_train: bool = False):
    """Save a static overview image of all examples."""
    puzzle = puzzles[puzzle_id]
    train_examples = puzzle.get('train', [])
    test_examples = puzzle.get('test', [])

    cmap = mcolors.ListedColormap(ARC_COLORS)
    bounds = np.arange(-0.5, 10.5, 1)
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    n_train = len(train_examples)
    n_test = len(test_examples)
    n_cols = 3
    n_rows = n_train + n_test

    # Use smaller figure size per row
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(9, 3 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)

    fig.suptitle(f'Puzzle: {puzzle_id}', fontsize=12, fontweight='bold')

    model.eval()
    for i, example in enumerate(train_examples):
        input_grid = np.array(example['input'])
        output_grid = np.array(example['output'])

        axes[i, 0].imshow(input_grid, cmap=cmap, norm=norm)
        axes[i, 0].set_title(f'Train {i+1} In', fontsize=10)
        axes[i, 0].set_xticks([])
        axes[i, 0].set_yticks([])

        axes[i, 1].imshow(output_grid, cmap=cmap, norm=norm)
        axes[i, 1].set_title(f'Train {i+1} Out', fontsize=10)
        axes[i, 1].set_xticks([])
        axes[i, 1].set_yticks([])

        # Compute prediction for training example
        output_size = output_grid.shape
        predicted_grid = apply_predicted_transformation(
            input_grid, model, device, use_color_only, ordering_strategy,
            predict_absolute, per_object_anchor,
            output_size=output_size,
            selection_criterion=selection_criterion, selection_rule=selection_rule,
            no_train=no_train
        )
        axes[i, 2].imshow(predicted_grid, cmap=cmap, norm=norm)
        axes[i, 2].set_title(f'Train {i+1} Pred', fontsize=10)
        axes[i, 2].set_xticks([])
        axes[i, 2].set_yticks([])
        for spine in axes[i, 2].spines.values():
            spine.set_edgecolor('orange')
            spine.set_linewidth(2)
    for i, example in enumerate(test_examples):
        row = n_train + i
        input_grid = np.array(example['input'])

        axes[row, 0].imshow(input_grid, cmap=cmap, norm=norm)
        axes[row, 0].set_title(f'Test {i+1} In', fontsize=10)
        axes[row, 0].set_xticks([])
        axes[row, 0].set_yticks([])
        for spine in axes[row, 0].spines.values():
            spine.set_edgecolor('blue')
            spine.set_linewidth(2)

        if 'output' in example:
            output_grid = np.array(example['output'])
            output_size = output_grid.shape
            axes[row, 1].imshow(output_grid, cmap=cmap, norm=norm)
            axes[row, 1].set_title(f'Test {i+1} True', fontsize=10)
            for spine in axes[row, 1].spines.values():
                spine.set_edgecolor('green')
                spine.set_linewidth(2)
        else:
            output_size = None
            axes[row, 1].axis('off')
        axes[row, 1].set_xticks([])
        axes[row, 1].set_yticks([])

        predicted_grid = apply_predicted_transformation(
            input_grid, model, device, use_color_only, ordering_strategy,
            predict_absolute, per_object_anchor,
            output_size=output_size,
            selection_criterion=selection_criterion, selection_rule=selection_rule,
            no_train=no_train
        )
        axes[row, 2].imshow(predicted_grid, cmap=cmap, norm=norm)
        axes[row, 2].set_title(f'Test {i+1} Pred', fontsize=10)
        axes[row, 2].set_xticks([])
        axes[row, 2].set_yticks([])
        for spine in axes[row, 2].spines.values():
            spine.set_edgecolor('orange')
            spine.set_linewidth(2)

    plt.tight_layout()
    plt.savefig(f'puzzle_{puzzle_id}_visualization.png', dpi=150, bbox_inches='tight')
    print(f"\nOverview saved to puzzle_{puzzle_id}_visualization.png")
    plt.close(fig)


def save_placement_logic(
    puzzles: Dict, puzzle_id: str, model: nn.Module,
    device: torch.device, use_color_only: bool = False,
    ordering_strategy: Optional[str] = None, per_object_anchor: bool = False
):
    """
    Save detailed object placement logic to a text file for debugging.

    Shows how each object is positioned using the discovered anchor relationships
    for both training and test examples.
    """
    if not per_object_anchor:
        return  # Only relevant for per-object-anchor mode

    puzzle = puzzles.get(puzzle_id)
    if puzzle is None:
        return

    output_file = f'puzzle_{puzzle_id}_placement_logic.txt'

    with open(output_file, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write(f"OBJECT PLACEMENT LOGIC FOR PUZZLE: {puzzle_id}\n")
        f.write("=" * 70 + "\n\n")

        # Write anchor module configuration
        f.write("ANCHOR MODULE CONFIGURATION\n")
        f.write("-" * 40 + "\n")
        anchor_module = model.anchor_module
        for obj_idx in range(MAX_OBJECTS):
            if not anchor_module.is_configured[obj_idx]:
                continue
            framing = FRAMINGS[anchor_module.selected_framings[obj_idx].item()]
            ref_idx = anchor_module.selected_references[obj_idx].item()
            accuracy = anchor_module.framing_accuracies[obj_idx].item()

            f.write(f"Object {obj_idx}:\n")
            f.write(f"  Framing: {framing}\n")
            f.write(f"  Reference: {ref_idx if ref_idx >= 0 else 'N/A'}\n")
            f.write(f"  Screening accuracy: {accuracy:.1%}\n")

            if framing == 'object_relative' and anchor_module.has_anchor_points(obj_idx):
                src_anchor, tgt_anchor, offset = anchor_module.get_anchor_points(obj_idx)
                f.write(f"  Anchor relationship: {src_anchor.value.upper()} -> {tgt_anchor.value.upper()}\n")
                f.write(f"  Discovered offset: {offset}\n")
            f.write("\n")

        # Process training examples
        train_examples = puzzle.get('train', [])
        f.write("\n" + "=" * 70 + "\n")
        f.write("TRAINING EXAMPLES\n")
        f.write("=" * 70 + "\n")

        for ex_idx, example in enumerate(train_examples):
            f.write(f"\n{'#' * 60}\n")
            f.write(f"# TRAIN EXAMPLE {ex_idx + 1}\n")
            f.write(f"{'#' * 60}\n\n")

            input_grid = np.array(example['input'])
            output_grid = np.array(example['output'])

            _write_example_placement_logic(
                f, input_grid, output_grid, model, device,
                use_color_only, ordering_strategy
            )

        # Process test examples
        test_examples = puzzle.get('test', [])
        f.write("\n" + "=" * 70 + "\n")
        f.write("TEST EXAMPLES\n")
        f.write("=" * 70 + "\n")

        for ex_idx, example in enumerate(test_examples):
            f.write(f"\n{'#' * 60}\n")
            f.write(f"# TEST EXAMPLE {ex_idx + 1}\n")
            f.write(f"{'#' * 60}\n\n")

            input_grid = np.array(example['input'])
            output_grid = np.array(example['output']) if 'output' in example else None

            _write_example_placement_logic(
                f, input_grid, output_grid, model, device,
                use_color_only, ordering_strategy
            )

    print(f"Placement logic saved to {output_file}")


def _write_example_placement_logic(
    f, input_grid: np.ndarray, output_grid: Optional[np.ndarray],
    model: nn.Module, device: torch.device,
    use_color_only: bool, ordering_strategy: Optional[str]
):
    """Helper to write placement logic for a single example."""
    H, W = input_grid.shape
    grid_size = max(H, W, GRID_SIZE)

    f.write(f"Input grid: {H}x{W}\n")
    if output_grid is not None:
        f.write(f"Output grid: {output_grid.shape[0]}x{output_grid.shape[1]}\n")
    f.write(f"Normalization grid_size: {grid_size}\n\n")

    # Extract objects
    input_labels, input_colors, input_bboxes, _ = extract_connected_components(
        input_grid, use_color_only=use_color_only
    )

    if ordering_strategy:
        input_labels, input_colors, input_bboxes, _ = sort_objects_by_strategy(
            input_labels, input_colors, input_bboxes, ordering_strategy
        )

    if len(input_colors) == 0:
        f.write("No objects found in input.\n")
        return

    # Compute properties
    props = compute_object_properties(input_labels, input_colors, input_bboxes, grid_size)

    f.write(f"Found {len(input_colors)} objects:\n")
    for i, color in enumerate(input_colors):
        if i >= MAX_OBJECTS or not props['valid'][i]:
            continue
        bbox = props['bboxes'][i] * grid_size
        centroid = props['centroids'][i] * grid_size
        f.write(f"  Object {i}: color={color}, bbox=[{bbox[0]:.1f},{bbox[1]:.1f},{bbox[2]:.1f},{bbox[3]:.1f}], ")
        f.write(f"centroid=({centroid[0]:.1f},{centroid[1]:.1f})\n")

    # Get model predictions
    centroids = torch.from_numpy(props['centroids']).float().unsqueeze(0).to(device)
    bboxes = torch.from_numpy(props['bboxes']).float().unsqueeze(0).to(device)
    areas = torch.from_numpy(props['areas']).float().unsqueeze(0).to(device)
    colors = torch.from_numpy(props['colors']).long().unsqueeze(0).to(device)
    valid = torch.from_numpy(props['valid']).bool().unsqueeze(0).to(device)

    with torch.no_grad():
        _relations, predictions, _framing_weights, _ref_weights = model(
            centroids, bboxes, areas, colors, valid,
            target_output_bboxes=None
        )
        predictions = predictions.squeeze(0).cpu().numpy()

    f.write(f"\nModel predictions (normalized):\n")
    for i in range(len(input_colors)):
        if i >= MAX_OBJECTS or not props['valid'][i]:
            continue
        f.write(f"  Object {i}: {predictions[i]}\n")

    # Decode step by step
    f.write(f"\nDecoding (step by step):\n")
    f.write("-" * 40 + "\n")

    input_centroids_pixels = props['centroids'] * grid_size
    input_bboxes_pixels = props['bboxes'] * grid_size

    K = len(input_colors)
    decoded = np.zeros((MAX_OBJECTS, 2), dtype=np.float32)
    decoded_bboxes = np.zeros((MAX_OBJECTS, 4), dtype=np.float32)

    anchor_module = model.anchor_module
    with torch.no_grad():
        framing_indices = anchor_module.selected_framings[:MAX_OBJECTS].cpu().numpy()
        reference_indices = anchor_module.selected_references[:MAX_OBJECTS].cpu().numpy()
        source_anchor_indices = anchor_module.source_anchors[:MAX_OBJECTS].cpu().numpy()
        target_anchor_indices = anchor_module.target_anchors[:MAX_OBJECTS].cpu().numpy()

    for obj_idx in range(min(K, MAX_OBJECTS)):
        if not props['valid'][obj_idx]:
            continue

        pred = predictions[obj_idx]
        framing = FRAMINGS[framing_indices[obj_idx]]
        input_centroid = input_centroids_pixels[obj_idx]

        # Get source size
        bbox = input_bboxes_pixels[obj_idx]
        src_h = max(1, int(round(bbox[2] - bbox[0])) + 1)
        src_w = max(1, int(round(bbox[3] - bbox[1])) + 1)
        source_size = (src_h, src_w)

        f.write(f"\nObject {obj_idx}:\n")
        f.write(f"  Input centroid: ({input_centroid[0]:.2f}, {input_centroid[1]:.2f})\n")
        f.write(f"  Input bbox (pixels): [{bbox[0]:.1f}, {bbox[1]:.1f}, {bbox[2]:.1f}, {bbox[3]:.1f}]\n")
        f.write(f"  Size: {source_size}\n")
        f.write(f"  Framing: {framing}\n")
        f.write(f"  Prediction: [{pred[0]:.4f}, {pred[1]:.4f}]\n")

        if framing == 'object_relative':
            ref_idx = reference_indices[obj_idx]
            f.write(f"  Reference object: {ref_idx}\n")

            if ref_idx >= 0 and props['valid'][ref_idx]:
                reference_position = decoded[ref_idx]

                if anchor_module.has_anchor_points(obj_idx):
                    source_anchor = ALL_ANCHORS[source_anchor_indices[obj_idx]]
                    target_anchor = ALL_ANCHORS[target_anchor_indices[obj_idx]]
                    reference_bbox = decoded_bboxes[ref_idx]

                    f.write(f"  Using ANCHOR-BASED decoding:\n")
                    f.write(f"    Source anchor: {source_anchor.value.upper()}\n")
                    f.write(f"    Target anchor: {target_anchor.value.upper()}\n")
                    f.write(f"    Reference decoded bbox: [{reference_bbox[0]:.1f}, {reference_bbox[1]:.1f}, {reference_bbox[2]:.1f}, {reference_bbox[3]:.1f}]\n")

                    # Compute anchor positions
                    ref_tl = (int(round(reference_bbox[0])), int(round(reference_bbox[1])))
                    ref_h = max(1, int(round(reference_bbox[2] - reference_bbox[0])) + 1)
                    ref_w = max(1, int(round(reference_bbox[3] - reference_bbox[1])) + 1)
                    tgt_anchor_pos = get_anchor_position(ref_tl, (ref_h, ref_w), target_anchor)
                    f.write(f"    Target anchor position: ({tgt_anchor_pos[0]}, {tgt_anchor_pos[1]})\n")

                    offset_pixels = pred * np.array([grid_size, grid_size])
                    f.write(f"    Offset (pixels): ({offset_pixels[0]:.2f}, {offset_pixels[1]:.2f})\n")

                    src_anchor_pos = (tgt_anchor_pos[0] + offset_pixels[0], tgt_anchor_pos[1] + offset_pixels[1])
                    f.write(f"    Source anchor position: ({src_anchor_pos[0]:.2f}, {src_anchor_pos[1]:.2f})\n")

                    # Back-calculate top-left
                    anchor_dr, anchor_dc = get_anchor_offset(source_anchor, src_h, src_w)
                    src_tl = (src_anchor_pos[0] - anchor_dr, src_anchor_pos[1] - anchor_dc)
                    f.write(f"    Anchor offset from TL: ({anchor_dr}, {anchor_dc})\n")
                    f.write(f"    Computed top-left: ({src_tl[0]:.2f}, {src_tl[1]:.2f})\n")

                    decoded[obj_idx] = decode_framing_prediction(
                        pred, framing, input_centroid, grid_size, grid_size,
                        reference_position=reference_position,
                        source_size=source_size,
                        reference_bbox=reference_bbox,
                        source_anchor=source_anchor,
                        target_anchor=target_anchor
                    )
                else:
                    f.write(f"  Using CENTROID-BASED decoding (no anchor points)\n")
                    f.write(f"    Reference position: ({reference_position[0]:.2f}, {reference_position[1]:.2f})\n")
                    decoded[obj_idx] = decode_framing_prediction(
                        pred, framing, input_centroid, grid_size, grid_size,
                        reference_position=reference_position
                    )
            else:
                f.write(f"  FALLBACK: using input centroid as reference\n")
                decoded[obj_idx] = decode_framing_prediction(
                    pred, framing, input_centroid, grid_size, grid_size,
                    reference_position=input_centroid
                )
        else:
            # Check for grid anchor points
            if anchor_module.has_grid_anchor(obj_idx):
                grid_anchor, grid_target_position = anchor_module.get_grid_anchor(obj_idx)

                f.write(f"  Using GRID ANCHOR-BASED decoding:\n")
                f.write(f"    Grid anchor: {grid_anchor.value.upper()}\n")
                f.write(f"    Target position: ({grid_target_position[0]}, {grid_target_position[1]})\n")

                offset_pixels = pred * 20.0
                f.write(f"    Prediction * 20: ({offset_pixels[0]:.2f}, {offset_pixels[1]:.2f})\n")

                anchor_pos = (grid_target_position[0] + offset_pixels[0],
                             grid_target_position[1] + offset_pixels[1])
                f.write(f"    Anchor position: ({anchor_pos[0]:.2f}, {anchor_pos[1]:.2f})\n")

                # Back-calculate top-left
                anchor_dr, anchor_dc = get_anchor_offset(grid_anchor, src_h, src_w)
                src_tl = (anchor_pos[0] - anchor_dr, anchor_pos[1] - anchor_dc)
                f.write(f"    Anchor offset from TL: ({anchor_dr}, {anchor_dc})\n")
                f.write(f"    Computed top-left: ({src_tl[0]:.2f}, {src_tl[1]:.2f})\n")

                decoded[obj_idx] = decode_framing_prediction(
                    pred, framing, input_centroid, grid_size, grid_size,
                    source_size=source_size,
                    grid_anchor=grid_anchor,
                    grid_target_position=grid_target_position
                )
            else:
                f.write(f"  Using CENTROID-BASED decoding\n")
                decoded[obj_idx] = decode_framing_prediction(
                    pred, framing, input_centroid, grid_size, grid_size
                )

        # Update decoded bbox
        centroid = decoded[obj_idx]
        top_left_r = centroid[0] - (src_h - 1) / 2.0
        top_left_c = centroid[1] - (src_w - 1) / 2.0
        decoded_bboxes[obj_idx] = [top_left_r, top_left_c,
                                   top_left_r + src_h - 1, top_left_c + src_w - 1]

        f.write(f"  => Decoded centroid: ({decoded[obj_idx][0]:.2f}, {decoded[obj_idx][1]:.2f})\n")
        f.write(f"  => Decoded bbox: [{decoded_bboxes[obj_idx][0]:.1f}, {decoded_bboxes[obj_idx][1]:.1f}, {decoded_bboxes[obj_idx][2]:.1f}, {decoded_bboxes[obj_idx][3]:.1f}]\n")

        # Compare with ground truth if available
        if output_grid is not None:
            output_labels, output_colors, output_bboxes_raw, _ = extract_connected_components(
                output_grid, use_color_only=use_color_only
            )
            if ordering_strategy:
                output_labels, output_colors, output_bboxes_raw, _ = sort_objects_by_strategy(
                    output_labels, output_colors, output_bboxes_raw, ordering_strategy
                )

            if obj_idx < len(output_colors):
                out_bbox = output_bboxes_raw[obj_idx]
                out_centroid = ((out_bbox[0] + out_bbox[2]) / 2, (out_bbox[1] + out_bbox[3]) / 2)
                f.write(f"  Ground truth centroid: ({out_centroid[0]:.2f}, {out_centroid[1]:.2f})\n")
                error = np.sqrt((decoded[obj_idx][0] - out_centroid[0])**2 +
                               (decoded[obj_idx][1] - out_centroid[1])**2)
                f.write(f"  Pixel error: {error:.2f}\n")


def decode_per_object_anchor(
    predictions: np.ndarray,
    anchor_module: AnchorFramingModule,
    input_centroids: np.ndarray,
    grid_height: int,
    grid_width: int,
    valid: np.ndarray,
    input_bboxes: Optional[np.ndarray] = None,
    verbose: bool = False
) -> np.ndarray:
    """
    Decode predictions back to pixel positions using per-object anchor framings.

    Must process objects in order (0, 1, 2...) due to object_relative dependencies.

    Args:
        predictions: (K, 2) predicted values in framing space
        anchor_module: AnchorFramingModule with learned framings
        input_centroids: (K, 2) input centroids in pixel coordinates
        grid_height, grid_width: Grid dimensions
        valid: (K,) validity mask
        input_bboxes: (K, 4) input bounding boxes [min_r, min_c, max_r, max_c] in pixel coords
                      Required for anchor-based object_relative decoding
        verbose: If True, print detailed decoding logic

    Returns:
        decoded_positions: (K, 2) decoded [row, col] positions in pixel coordinates
    """
    K = predictions.shape[0]
    decoded = np.zeros((K, 2), dtype=np.float32)
    decoded_bboxes = np.zeros((K, 4), dtype=np.float32)  # Track decoded bboxes for anchor-based

    # Get framing selections from the anchor module
    with torch.no_grad():
        framing_indices = anchor_module.selected_framings[:K].cpu().numpy()
        reference_indices = anchor_module.selected_references[:K].cpu().numpy()
        source_anchor_indices = anchor_module.source_anchors[:K].cpu().numpy()
        target_anchor_indices = anchor_module.target_anchors[:K].cpu().numpy()

    if verbose:
        print(f"\nDecoding {K} objects (grid: {grid_height}x{grid_width}):")

    for obj_idx in range(K):
        if not valid[obj_idx]:
            continue

        pred = predictions[obj_idx]
        framing = FRAMINGS[framing_indices[obj_idx]]
        input_centroid = input_centroids[obj_idx]

        # Get source object size from input bbox (assume shape preserved)
        source_size = None
        if input_bboxes is not None:
            bbox = input_bboxes[obj_idx]
            src_h = max(1, int(round(bbox[2] - bbox[0])) + 1)
            src_w = max(1, int(round(bbox[3] - bbox[1])) + 1)
            source_size = (src_h, src_w)

        if verbose:
            print(f"\n  Object {obj_idx}:")
            print(f"    Input centroid: ({input_centroid[0]:.1f}, {input_centroid[1]:.1f})")
            print(f"    Input bbox: {input_bboxes[obj_idx] if input_bboxes is not None else 'N/A'}")
            print(f"    Source size: {source_size}")
            print(f"    Framing: {framing}")
            print(f"    Prediction: {pred}")

        if framing == 'object_relative':
            ref_idx = reference_indices[obj_idx]
            reference_position = None
            reference_bbox = None
            source_anchor = None
            target_anchor = None

            if verbose:
                print(f"    Reference object: {ref_idx}")

            if ref_idx >= 0 and valid[ref_idx]:
                reference_position = decoded[ref_idx]

                # Check if we have anchor points for this object
                if anchor_module.has_anchor_points(obj_idx) and input_bboxes is not None:
                    source_anchor = ALL_ANCHORS[source_anchor_indices[obj_idx]]
                    target_anchor = ALL_ANCHORS[target_anchor_indices[obj_idx]]
                    reference_bbox = decoded_bboxes[ref_idx]

                    if verbose:
                        print(f"    Using ANCHOR-BASED decoding:")
                        print(f"      Source anchor: {source_anchor.value.upper()}")
                        print(f"      Target anchor: {target_anchor.value.upper()}")
                        print(f"      Reference bbox (decoded): {reference_bbox}")
                        print(f"      Reference centroid (decoded): {reference_position}")

                        # Show anchor position calculation
                        ref_tl = (int(round(reference_bbox[0])), int(round(reference_bbox[1])))
                        ref_h = max(1, int(round(reference_bbox[2] - reference_bbox[0])) + 1)
                        ref_w = max(1, int(round(reference_bbox[3] - reference_bbox[1])) + 1)
                        tgt_anchor_pos = get_anchor_position(ref_tl, (ref_h, ref_w), target_anchor)
                        print(f"      Target anchor position: {tgt_anchor_pos}")

                        offset_pixels = pred * np.array([grid_height, grid_width])
                        src_anchor_pos = (tgt_anchor_pos[0] + offset_pixels[0],
                                         tgt_anchor_pos[1] + offset_pixels[1])
                        print(f"      Offset (pixels): {offset_pixels}")
                        print(f"      Source anchor position: {src_anchor_pos}")
                else:
                    if verbose:
                        print(f"    Using CENTROID-BASED decoding (no anchor points)")
                        print(f"      Reference position: {reference_position}")
            else:
                # Fallback: use input position as reference
                reference_position = input_centroid
                if verbose:
                    print(f"    FALLBACK: using input centroid as reference")

            decoded[obj_idx] = decode_framing_prediction(
                pred, framing, input_centroid,
                grid_height, grid_width,
                reference_position=reference_position,
                source_size=source_size,
                reference_bbox=reference_bbox,
                source_anchor=source_anchor,
                target_anchor=target_anchor
            )
        else:
            # Check for grid anchor points
            if anchor_module.has_grid_anchor(obj_idx):
                grid_anchor, grid_target_position = anchor_module.get_grid_anchor(obj_idx)

                if verbose:
                    print(f"    Using GRID ANCHOR-BASED decoding:")
                    print(f"      Grid anchor: {grid_anchor.value.upper()}")
                    print(f"      Target position: {grid_target_position}")

                    offset_pixels = pred * 20.0
                    print(f"      Offset (pred * 20): ({offset_pixels[0]:.2f}, {offset_pixels[1]:.2f})")

                    anchor_pos = (grid_target_position[0] + offset_pixels[0],
                                 grid_target_position[1] + offset_pixels[1])
                    print(f"      Anchor position: ({anchor_pos[0]:.2f}, {anchor_pos[1]:.2f})")

                    if source_size is not None:
                        anchor_dr, anchor_dc = get_anchor_offset(grid_anchor, source_size[0], source_size[1])
                        src_tl = (anchor_pos[0] - anchor_dr, anchor_pos[1] - anchor_dc)
                        print(f"      Anchor offset from TL: ({anchor_dr}, {anchor_dc})")
                        print(f"      Computed top-left: ({src_tl[0]:.2f}, {src_tl[1]:.2f})")

                decoded[obj_idx] = decode_framing_prediction(
                    pred, framing, input_centroid,
                    grid_height, grid_width,
                    source_size=source_size,
                    grid_anchor=grid_anchor,
                    grid_target_position=grid_target_position
                )
            else:
                if verbose:
                    print(f"    Using CENTROID-BASED decoding")
                decoded[obj_idx] = decode_framing_prediction(
                    pred, framing, input_centroid,
                    grid_height, grid_width
                )

        # Update decoded bbox for this object (for use by later objects)
        if source_size is not None:
            src_h, src_w = source_size
            centroid = decoded[obj_idx]
            # Back-calculate top-left from centroid
            top_left_r = centroid[0] - (src_h - 1) / 2.0
            top_left_c = centroid[1] - (src_w - 1) / 2.0
            decoded_bboxes[obj_idx] = [top_left_r, top_left_c,
                                       top_left_r + src_h - 1, top_left_c + src_w - 1]

        if verbose:
            print(f"    => Decoded centroid: ({decoded[obj_idx][0]:.1f}, {decoded[obj_idx][1]:.1f})")
            print(f"    => Decoded bbox: {decoded_bboxes[obj_idx]}")

    return decoded


def apply_predicted_transformation(input_grid: np.ndarray, model: nn.Module,
                                     device: torch.device, use_color_only: bool = False,
                                     ordering_strategy: Optional[str] = None,
                                     predict_absolute: bool = False,
                                     per_object_anchor: bool = False,
                                     verbose: bool = False,
                                     output_size: tuple = None,
                                     selection_criterion: Optional[str] = None,
                                     selection_rule: Optional[str] = None,
                                     no_train: bool = False
                                     ) -> np.ndarray:
    """
    Apply the model's predicted position deltas to create a predicted output grid.

    Args:
        input_grid: Input grid to transform
        model: Trained position prediction model
        device: Torch device
        use_color_only: If True, use color-based object extraction
        ordering_strategy: Optional ordering strategy name (e.g., 'left_to_right', 'top_to_bottom')
        predict_absolute: If True, model predicts absolute positions instead of deltas
        per_object_anchor: If True, use per-object anchor framing decoding
        verbose: If True, print detailed placement logic for debugging
        output_size: Optional (H, W) tuple for output grid size. If None, uses input size.
        selection_criterion: Optional ranking criterion for object selection (e.g., 'largest', 'smallest')
        selection_rule: Optional selection rule (e.g., 'top_1', 'top_2')
        no_train: If True, use screening offsets directly instead of model predictions
    """
    H, W = input_grid.shape
    # Use output_size if provided, otherwise default to input size
    out_H, out_W = output_size if output_size is not None else (H, W)
    grid_size = max(H, W, out_H, out_W, GRID_SIZE)

    # Extract objects from input
    input_labels, input_colors, input_bboxes, _ = extract_connected_components(
        input_grid, use_color_only=use_color_only
    )

    if len(input_colors) == 0:
        return input_grid.copy()

    # Apply selection filtering if configured (before ordering, must match training)
    selection_mask = None
    if selection_criterion is not None and selection_rule is not None:
        selection_mask = compute_selection_mask(
            input_labels, input_colors, input_bboxes, input_grid,
            selection_criterion, selection_rule
        )

    # Apply ordering strategy (after selection, must match training)
    if ordering_strategy:
        # Compute ordering mapping BEFORE sorting (to update selection_mask)
        if selection_mask is not None:
            objects_before = labels_to_ordering_objects(input_labels, input_colors, input_bboxes)
            strategy = get_ordering_strategy(ordering_strategy)
            ordered_objects = strategy.order(objects_before)
            old_to_new = {obj.id: new_idx for new_idx, obj in enumerate(ordered_objects)}
            # Reorder selection mask to match new ordering
            new_mask = np.zeros_like(selection_mask)
            for old_idx, selected in enumerate(selection_mask):
                if old_idx in old_to_new:
                    new_mask[old_to_new[old_idx]] = selected
            selection_mask = new_mask

        input_labels, input_colors, input_bboxes, _ = sort_objects_by_strategy(
            input_labels, input_colors, input_bboxes, ordering_strategy
        )

    # Compute properties
    props = compute_object_properties(input_labels, input_colors, input_bboxes, grid_size)

    # Apply selection mask to props['valid']
    if selection_mask is not None:
        padded_mask = np.zeros(MAX_OBJECTS, dtype=bool)
        padded_mask[:len(selection_mask)] = selection_mask
        props['valid'] = props['valid'] & padded_mask

    # Get predictions from model
    centroids = torch.from_numpy(props['centroids']).float().unsqueeze(0).to(device)
    bboxes = torch.from_numpy(props['bboxes']).float().unsqueeze(0).to(device)
    areas = torch.from_numpy(props['areas']).float().unsqueeze(0).to(device)
    colors = torch.from_numpy(props['colors']).long().unsqueeze(0).to(device)
    valid = torch.from_numpy(props['valid']).bool().unsqueeze(0).to(device)

    with torch.no_grad():
        if per_object_anchor and no_train:
            # No-train mode: use screening offsets directly without model prediction
            input_centroids_pixels = props['centroids'] * grid_size
            input_bboxes_pixels = props['bboxes'] * grid_size

            if verbose:
                print(f"\n{'='*60}")
                print("VERBOSE: Object Placement Logic (NO-TRAIN MODE)")
                print(f"{'='*60}")
                print(f"Grid size: {H}x{W}, normalization grid_size: {grid_size}")
                print(f"Number of objects: {len(input_colors)}")
                print("Using screening-discovered offsets directly")

            # Decode positions using screening offsets
            decoded_positions = np.zeros((MAX_OBJECTS, 2))
            decoded_bboxes = {}  # obj_idx -> bbox for reference by later objects

            for obj_idx in range(len(input_colors)):
                if obj_idx >= MAX_OBJECTS or not props['valid'][obj_idx]:
                    continue
                if not model.anchor_module.is_configured[obj_idx]:
                    decoded_positions[obj_idx] = input_centroids_pixels[obj_idx]
                    continue

                input_centroid = input_centroids_pixels[obj_idx]
                input_bbox = input_bboxes_pixels[obj_idx]

                # Get source size from input bbox
                src_height = max(1, int(round(input_bbox[2] - input_bbox[0])) + 1)
                src_width = max(1, int(round(input_bbox[3] - input_bbox[1])) + 1)
                source_size = (src_height, src_width)

                # Get reference info if needed
                ref_idx = model.anchor_module.get_reference_idx(obj_idx)
                reference_position = None
                reference_bbox = None
                if ref_idx >= 0 and ref_idx in decoded_bboxes:
                    reference_bbox = decoded_bboxes[ref_idx]
                    reference_position = np.array([
                        (reference_bbox[0] + reference_bbox[2]) / 2,
                        (reference_bbox[1] + reference_bbox[3]) / 2
                    ])

                # Apply screening offset
                predicted_centroid = apply_screening_offset(
                    model.anchor_module, obj_idx, input_centroid,
                    grid_size, grid_size, source_size, reference_position, reference_bbox
                )
                decoded_positions[obj_idx] = predicted_centroid

                # Store decoded bbox for reference by later objects
                pred_top_left = (predicted_centroid[0] - (src_height - 1) / 2.0,
                                predicted_centroid[1] - (src_width - 1) / 2.0)
                decoded_bboxes[obj_idx] = np.array([pred_top_left[0], pred_top_left[1],
                                                    pred_top_left[0] + src_height - 1,
                                                    pred_top_left[1] + src_width - 1])

                if verbose:
                    framing = FRAMINGS[model.anchor_module.get_framing_idx(obj_idx)]
                    print(f"  Object {obj_idx}: {framing} -> centroid={predicted_centroid}")

        elif per_object_anchor:
            # Per-object anchor mode: get predictions and decode using discovered framings
            _relations, predictions, _framing_weights, _ref_weights = model(
                centroids, bboxes, areas, colors, valid,
                target_output_bboxes=None  # No teacher forcing at inference
            )
            predictions = predictions.squeeze(0).cpu().numpy()

            # Decode using per-object anchor framings
            # Input centroids and bboxes in pixel coordinates
            input_centroids_pixels = props['centroids'] * grid_size
            input_bboxes_pixels = props['bboxes'] * grid_size

            if verbose:
                print(f"\n{'='*60}")
                print("VERBOSE: Object Placement Logic")
                print(f"{'='*60}")
                print(f"Grid size: {H}x{W}, normalization grid_size: {grid_size}")
                print(f"Number of objects: {len(input_colors)}")

                # Print anchor module configuration
                print(f"\nAnchor Module Configuration:")
                for i in range(len(input_colors)):
                    if not props['valid'][i]:
                        continue
                    framing = FRAMINGS[model.anchor_module.selected_framings[i].item()]
                    ref_idx = model.anchor_module.selected_references[i].item()
                    print(f"  Object {i}: framing={framing}, ref={ref_idx}")
                    if framing == 'object_relative' and model.anchor_module.has_anchor_points(i):
                        src_anchor, tgt_anchor, offset = model.anchor_module.get_anchor_points(i)
                        print(f"           anchor: {src_anchor.value.upper()} -> {tgt_anchor.value.upper()}, offset={offset}")

                print(f"\nModel Predictions (normalized):")
                for i in range(len(input_colors)):
                    if props['valid'][i]:
                        print(f"  Object {i}: prediction={predictions[i]}")

            decoded_positions = decode_per_object_anchor(
                predictions, model.anchor_module,
                input_centroids_pixels, grid_size, grid_size, props['valid'],
                input_bboxes=input_bboxes_pixels,
                verbose=verbose
            )
        else:
            _, pred_deltas = model(centroids, areas, colors, valid)
            pred_deltas = pred_deltas.squeeze(0).cpu().numpy()

    # Create output grid by moving objects (use output size if specified)
    output_grid = np.zeros((out_H, out_W), dtype=input_grid.dtype)

    for obj_idx in range(len(input_colors)):
        if obj_idx >= MAX_OBJECTS or not props['valid'][obj_idx]:
            continue

        # Get object mask
        obj_mask = (input_labels == obj_idx + 1)
        obj_color = input_colors[obj_idx]

        # Get object pixels
        rows, cols = np.where(obj_mask)

        # Current bounding box
        current_min_row = rows.min()
        current_min_col = cols.min()
        current_max_row = rows.max()
        current_max_col = cols.max()

        if per_object_anchor:
            # Decoded positions are output centroids in pixel coordinates
            current_centroid_row = rows.mean()
            current_centroid_col = cols.mean()
            pred_centroid_row = decoded_positions[obj_idx, 0]
            pred_centroid_col = decoded_positions[obj_idx, 1]
            delta_row = int(round(pred_centroid_row - current_centroid_row))
            delta_col = int(round(pred_centroid_col - current_centroid_col))
        elif predict_absolute:
            # Model predicts absolute output centroid position
            current_centroid_row = rows.mean()
            current_centroid_col = cols.mean()
            pred_centroid_row = pred_deltas[obj_idx, 0] * grid_size
            pred_centroid_col = pred_deltas[obj_idx, 1] * grid_size
            delta_row = int(round(pred_centroid_row - current_centroid_row))
            delta_col = int(round(pred_centroid_col - current_centroid_col))
        else:
            # Model predicts delta (convert from normalized to pixels)
            delta_row = int(round(pred_deltas[obj_idx, 0] * grid_size))
            delta_col = int(round(pred_deltas[obj_idx, 1] * grid_size))

        # Clamp deltas to keep object within output grid bounds
        # Ensure top-left doesn't go negative
        delta_row = max(delta_row, -current_min_row)
        delta_col = max(delta_col, -current_min_col)
        # Ensure bottom-right doesn't exceed output grid
        delta_row = min(delta_row, out_H - 1 - current_max_row)
        delta_col = min(delta_col, out_W - 1 - current_max_col)

        # Move each pixel of the object
        for r, c in zip(rows, cols):
            new_r = r + delta_row
            new_c = c + delta_col

            # Check bounds against output grid size
            if 0 <= new_r < out_H and 0 <= new_c < out_W:
                output_grid[new_r, new_c] = obj_color

    return output_grid


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Relational Position Prediction for ARC")

    parser.add_argument("--puzzle-id", type=str, required=True,
                        help="Puzzle ID to train on")
    parser.add_argument("--dataset", type=str, default="arc-agi-1",
                        choices=["arc-agi-1", "arc-agi-2"])
    parser.add_argument("--data-root", type=str, default="kaggle/combined")

    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=8)

    parser.add_argument("--object-by-color", action="store_true",
                        help="Use color-based object extraction (one object per color)")

    # Ordering strategy flags
    parser.add_argument("--ordering-strategy", type=str,
                        choices=['left_to_right', 'right_to_left', 'top_to_bottom',
                                'bottom_to_top', 'diagonal_tl_br', 'diagonal_tr_bl',
                                'largest_first', 'smallest_first', 'by_color',
                                'adaptive_reading_order', 'quadrant_order'],
                        help="Object ordering strategy (e.g., left_to_right, adaptive_reading_order)")
    parser.add_argument("--screen-ordering", action="store_true",
                        help="Auto-screen to find best ordering strategy for this puzzle")
    parser.add_argument("--left-to-right", action="store_true",
                        help="DEPRECATED: Use --ordering-strategy left_to_right instead")

    parser.add_argument("--predict-absolute", action="store_true",
                        help="Predict absolute output positions instead of deltas (target = output_centroid, not output_centroid - input_centroid)")

    # Per-object anchor framing flags
    parser.add_argument("--per-object-anchor", action="store_true",
                        help="Enable per-object-index anchor learning. Each object index "
                             "discovers its own framing (delta, grid corners, object-relative) "
                             "via screening phase that measures pixel accuracy.")
    parser.add_argument("--screening-epochs", type=int, default=100,
                        help="Number of epochs per framing during screening phase")

    # Augmentation flags
    parser.add_argument("--num-augmentations", type=int, default=0,
                        help="Number of augmented versions per sample (0 = no augmentation)")
    aug_group = parser.add_mutually_exclusive_group()
    aug_group.add_argument("--dihedral-only", action="store_true",
                           help="Only apply dihedral transforms (no color permutation)")
    aug_group.add_argument("--color-only", action="store_true",
                           help="Only apply color permutations (no dihedral transform)")

    # Selection filtering flags
    parser.add_argument("--selection-criterion", type=str, choices=RANKING_CRITERIA,
                        help="Ranking criterion for object selection (e.g., largest, leftmost)")
    parser.add_argument("--selection-rule", type=str, choices=SELECTION_RULES,
                        help="Selection rule to apply (e.g., top_1, top_2)")
    parser.add_argument("--screen-selection", action="store_true",
                        help="Auto-screen to find best selection criterion and rule for this puzzle")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-visualize", action="store_true",
                        help="Skip visualization after training")
    parser.add_argument("--verbose", action="store_true",
                        help="Print detailed object placement logic during test visualization")
    parser.add_argument("--no-train", action="store_true",
                        help="Skip training, apply screening-discovered offsets directly")

    args = parser.parse_args()

    # Set seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    print(f"Device: {DEVICE}")
    print(f"Puzzle: {args.puzzle_id}")
    print(f"Object extraction: {'by-color' if args.object_by_color else 'connected-components'}")
    if args.per_object_anchor:
        print("Decoding mode: PER-OBJECT ANCHOR (discovers per-object-index framings via screening)")
        print(f"  Screening epochs per framing: {args.screening_epochs}")
    else:
        print(f"Prediction target: {'absolute positions' if args.predict_absolute else 'position deltas'}")
    if args.num_augmentations > 0:
        aug_type = "dihedral-only" if args.dihedral_only else "color-only" if args.color_only else "dihedral+color"
        print(f"Augmentation: {args.num_augmentations} samples ({aug_type})")
    else:
        print("Augmentation: disabled")

    # Load puzzles
    print("\nLoading puzzles...")
    puzzles = load_puzzles(args.dataset, args.data_root)

    if args.puzzle_id not in puzzles:
        print(f"Error: puzzle {args.puzzle_id} not found")
        print(f"Available puzzles: {len(puzzles)}")
        return

    # Selection screening/configuration
    selection_criterion = args.selection_criterion
    selection_rule = args.selection_rule

    if args.screen_selection:
        print("\n" + "=" * 60)
        print("SELECTION SCREENING")
        print("=" * 60)

        screener = SelectionScreener(verbose=args.verbose)
        selection_samples = screener.create_selection_samples(
            puzzles, [args.puzzle_id], use_color_only=args.object_by_color
        )

        if selection_samples:
            # Analyze selection pattern
            pattern = screener.analyze_selection_pattern(selection_samples)
            print(f"Selection pattern: {pattern.get('description', pattern.get('pattern', 'unknown'))}")

            # Screen all criterion+rule combinations
            screen_results = screener.screen(selection_samples)

            selection_criterion = screen_results['best_criterion']
            selection_rule = screen_results['best_rule']

            print(f"\nBest selection rule found:")
            print(f"  Criterion: {selection_criterion}")
            print(f"  Rule: {selection_rule}")
            print(f"  Accuracy: {screen_results['best_accuracy']:.1%}")

            # Show top 5 results
            print("\nTop 5 criterion+rule combinations:")
            sorted_results = sorted(screen_results['all_results'].items(),
                                    key=lambda x: x[1], reverse=True)
            for (criterion, rule), acc in sorted_results[:5]:
                print(f"  {criterion:25s} + {rule:10s}: {acc:.1%}")
        else:
            print("Warning: No selection samples created, skipping selection screening")

    # Print selection config
    if selection_criterion and selection_rule:
        print(f"\nSelection filtering: {selection_criterion} + {selection_rule}")
    else:
        print("\nSelection filtering: disabled (using all objects)")

    # Ordering strategy configuration
    ordering_strategy = args.ordering_strategy
    explicit_ordering = ordering_strategy is not None  # Track if user explicitly provided one

    # Handle deprecated --left-to-right flag
    if args.left_to_right and not ordering_strategy:
        print("Warning: --left-to-right is deprecated. Use --ordering-strategy left_to_right")
        ordering_strategy = 'left_to_right'
        explicit_ordering = True

    if args.screen_ordering:
        print("\n" + "=" * 60)
        print("ORDERING SCREENING")
        print("=" * 60)

        puzzle = puzzles[args.puzzle_id]
        screen_results = screen_orderings_for_puzzle(
            puzzle,
            verbose=args.verbose,
            selection_criterion=selection_criterion,
            selection_rule=selection_rule,
            use_color_only=args.object_by_color
        )

        screened_ordering = screen_results['best_name']

        print(f"\nBest ordering strategy found:")
        print(f"  Strategy: {screened_ordering}")
        print(f"  Consistent framings: {screen_results['is_consistent']}")

        # Show all results if verbose
        if args.verbose:
            print("\nAll ordering results:")
            for name, result in screen_results['all_results'].items():
                status = "CONSISTENT" if result.get('consistent', False) else "inconsistent"
                num_framings = result.get('num_framings', 0)
                print(f"  {name:20s}: {status} ({num_framings} framings)")

        # Only use screened ordering if user didn't explicitly provide one
        if explicit_ordering:
            print(f"\n*** Using explicitly provided ordering: {ordering_strategy} (overriding screening) ***")
        else:
            ordering_strategy = screened_ordering

    # Print ordering config
    if ordering_strategy:
        print(f"\nObject ordering: {ordering_strategy}")
    else:
        print("\nObject ordering: disabled (extraction order)")

    # Create TRAINING dataset (train examples only)
    print("\nCreating training dataset (train pairs only)...")
    train_dataset = PositionDataset(
        puzzles,
        puzzle_ids=[args.puzzle_id],
        use_color_only=args.object_by_color,
        include_test=False,  # Train only on training pairs
        num_augmentations=args.num_augmentations,
        dihedral_only=args.dihedral_only,
        color_only=args.color_only,
        ordering_strategy=ordering_strategy,
        predict_absolute=args.predict_absolute,
        per_object_anchor=args.per_object_anchor,
        selection_criterion=selection_criterion,
        selection_rule=selection_rule
    )

    if len(train_dataset) == 0:
        print("Error: no valid training samples created")
        return

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True
    )

    # Create TEST dataset (test examples only, no augmentation)
    print("Creating test dataset (held-out test input)...")

    # Temporarily modify puzzles to only include test examples
    puzzle = puzzles[args.puzzle_id]
    test_puzzle = {
        args.puzzle_id: {
            'train': puzzle.get('test', []),  # Use test as "train" for dataset creation
            'test': []
        }
    }
    test_dataset = PositionDataset(
        test_puzzle,
        puzzle_ids=[args.puzzle_id],
        use_color_only=args.object_by_color,
        include_test=False,
        num_augmentations=0,  # No augmentation for test
        ordering_strategy=ordering_strategy,
        predict_absolute=args.predict_absolute,
        per_object_anchor=args.per_object_anchor,
        selection_criterion=selection_criterion,
        selection_rule=selection_rule
    )

    has_test_data = len(test_dataset) > 0
    if has_test_data:
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=args.batch_size, shuffle=False
        )
        print(f"Test samples: {len(test_dataset)}")
    else:
        print("Warning: No test samples with ground truth available")

    # Create model
    print("\nCreating model...")
    if args.per_object_anchor:
        model = PerObjectAnchorTransformModule(
            hidden_dim=args.hidden_dim,
            num_heads=4
        ).to(DEVICE)
        print("Using PerObjectAnchorTransformModule")
    else:
        model = PositionalTransformModule(
            hidden_dim=args.hidden_dim,
            num_heads=4
        ).to(DEVICE)
        print("Using PositionalTransformModule")

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)

    # Training loop
    print("\n" + "=" * 60)
    print("Training (on training pairs only)")
    print("=" * 60)

    best_loss = float('inf')
    skip_training = False  # Will be set to True if --no-train or auto-detected low variance

    if args.per_object_anchor:
        # Phase 1: Screening - discover best framing per object index
        # This trains small independent models to measure pixel accuracy for each framing
        trainer = AnchorScreeningTrainer(screening_epochs=args.screening_epochs, lr=0.01, verbose=args.verbose)
        _screening_results = trainer.run_screening(train_dataset, model.anchor_module, DEVICE)

        # Print screening results summary
        print("\nScreening Results:")
        max_obj = max(sum(s.input_valid) for s in train_dataset.samples) if train_dataset.samples else 0
        print(model.get_framing_summary(max_obj))

        # Check if all configured objects have low variance - auto-skip training if so
        auto_no_train = False
        if not args.no_train and model.anchor_module.all_low_variance(threshold=0.001):
            max_var = model.anchor_module.get_max_variance()
            print(f"\n*** All objects have near-zero variance (max={max_var:.6f}) ***")
            print("*** Auto-enabling no-train mode since screening found perfect anchors ***")
            auto_no_train = True

        skip_training = args.no_train or auto_no_train

        if skip_training:
            # Skip training, use screening offsets directly
            print("\n" + "=" * 60)
            if args.no_train:
                print("SKIPPING TRAINING (--no-train enabled)")
            else:
                print("SKIPPING TRAINING (auto-detected: all variances ≈ 0)")
            print("Using screening-discovered offsets directly")
            print("=" * 60)
        else:
            # Phase 2: Full training with discovered framings
            # Now train the actual model using the configured framings
            print("\n" + "=" * 60)
            print("FULL TRAINING (with discovered framings)")
            print("=" * 60)

            for epoch in range(args.epochs):
                # Use per-object anchor training function
                train_metrics = train_epoch_per_object_anchor(
                    model, train_loader, optimizer, DEVICE
                )
                scheduler.step()

                if (epoch + 1) % 100 == 0 or epoch == 0:
                    # Evaluate with pixel accuracy
                    train_eval = evaluate_per_object_anchor(model, train_loader, DEVICE)
                    log_msg = (f"Epoch {epoch+1:4d}: train_loss={train_metrics['loss']:.4f}, "
                               f"train_px_acc={train_eval['pixel_accuracy']:.1%}")

                    if has_test_data:
                        test_eval = evaluate_per_object_anchor(model, test_loader, DEVICE)
                        log_msg += f", test_px_acc={test_eval['pixel_accuracy']:.1%}"

                    print(log_msg)

                    if train_eval['loss'] < best_loss:
                        best_loss = train_eval['loss']
    elif args.no_train:
        # --no-train without --per-object-anchor doesn't make sense
        print("\n" + "=" * 60)
        print("WARNING: --no-train requires --per-object-anchor")
        print("Proceeding with standard training...")
        print("=" * 60)
        for epoch in range(args.epochs):
            train_metrics = train_epoch(model, train_loader, optimizer, DEVICE)
            scheduler.step()

            if (epoch + 1) % 100 == 0 or epoch == 0:
                train_eval = evaluate(model, train_loader, DEVICE)
                log_msg = (f"Epoch {epoch+1:4d}: train_loss={train_metrics['loss']:.4f}, "
                           f"train_dir_acc={train_metrics['direction_acc']:.2%}")

                if has_test_data:
                    test_eval = evaluate(model, test_loader, DEVICE)
                    log_msg += f", test_loss={test_eval['loss']:.4f}, test_dir_acc={test_eval['direction_acc']:.2%}"

                print(log_msg)

                if train_eval['loss'] < best_loss:
                    best_loss = train_eval['loss']
    else:
        for epoch in range(args.epochs):
            train_metrics = train_epoch(model, train_loader, optimizer, DEVICE)
            scheduler.step()

            if (epoch + 1) % 100 == 0 or epoch == 0:
                train_eval = evaluate(model, train_loader, DEVICE)
                log_msg = (f"Epoch {epoch+1:4d}: train_loss={train_metrics['loss']:.4f}, "
                           f"train_dir_acc={train_metrics['direction_acc']:.2%}")

                if has_test_data:
                    test_eval = evaluate(model, test_loader, DEVICE)
                    log_msg += f", test_loss={test_eval['loss']:.4f}, test_dir_acc={test_eval['direction_acc']:.2%}"

                print(log_msg)

                if train_eval['loss'] < best_loss:
                    best_loss = train_eval['loss']

    # Final evaluation
    print("\n" + "=" * 60)
    print("Final Evaluation")
    print("=" * 60)

    if args.per_object_anchor and skip_training:
        # No-train mode: evaluate using screening offsets directly
        print("\n--- Training Set (using screening offsets) ---")
        train_final = evaluate_no_train(model.anchor_module, train_dataset.samples)
        print(f"Pixel Accuracy: {train_final['pixel_accuracy']:.1%}")
        print(f"Mean Pixel Error: {train_final['mean_pixel_error']:.2f}px")
        print(f"Correct/Total: {train_final['total_correct']}/{train_final['total_samples']}")

        # Per-object breakdown
        if train_final['per_object_accuracy']:
            print("\nPer-object accuracy:")
            for obj_idx, acc in sorted(train_final['per_object_accuracy'].items()):
                framing_idx = model.anchor_module.get_framing_idx(obj_idx)
                framing = FRAMINGS[framing_idx]
                print(f"  Object {obj_idx}: {acc:.1%} ({framing})")

        if has_test_data:
            print("\n--- Test Set (using screening offsets) ---")
            test_final = evaluate_no_train(model.anchor_module, test_dataset.samples)
            print(f"Pixel Accuracy: {test_final['pixel_accuracy']:.1%}")
            print(f"Mean Pixel Error: {test_final['mean_pixel_error']:.2f}px")
            print(f"Correct/Total: {test_final['total_correct']}/{test_final['total_samples']}")

            # Per-object breakdown
            if test_final['per_object_accuracy']:
                print("\nPer-object accuracy:")
                for obj_idx, acc in sorted(test_final['per_object_accuracy'].items()):
                    framing_idx = model.anchor_module.get_framing_idx(obj_idx)
                    framing = FRAMINGS[framing_idx]
                    print(f"  Object {obj_idx}: {acc:.1%} ({framing})")

    elif args.per_object_anchor:
        print("\n--- Training Set ---")
        train_final = evaluate_per_object_anchor(model, train_loader, DEVICE)
        print(f"Loss: {train_final['loss']:.4f}")
        print(f"Pixel Accuracy: {train_final['pixel_accuracy']:.1%}")
        print(f"Mean Pixel Error: {train_final['mean_pixel_error']:.2f}px")

        if has_test_data:
            print("\n--- Test Set (Held-out) ---")
            test_final = evaluate_per_object_anchor(model, test_loader, DEVICE)
            print(f"Loss: {test_final['loss']:.4f}")
            print(f"Pixel Accuracy: {test_final['pixel_accuracy']:.1%}")
            print(f"Mean Pixel Error: {test_final['mean_pixel_error']:.2f}px")
    else:
        print("\n--- Training Set ---")
        train_final = evaluate(model, train_loader, DEVICE)
        print(f"Loss: {train_final['loss']:.4f}")
        print(f"L1 Error: {train_final['l1_error']:.4f}")
        print(f"Direction Accuracy: {train_final['direction_acc']:.2%}")

        if has_test_data:
            print("\n--- Test Set (Held-out) ---")
            test_final = evaluate(model, test_loader, DEVICE)
            print(f"Loss: {test_final['loss']:.4f}")
            print(f"L1 Error: {test_final['l1_error']:.4f}")
            print(f"Direction Accuracy: {test_final['direction_acc']:.2%}")

        # Text-based visualization of predictions
        visualize_predictions(model, train_dataset, DEVICE,
                              title="Training Set Predictions")

        if has_test_data:
            visualize_predictions(model, test_dataset, DEVICE,
                                  title="Test Set Predictions")

    # Visualize grids with matplotlib
    if not args.no_visualize:
        print("\n" + "=" * 60)
        print("Grid Visualization")
        print("=" * 60)
        visualize_grids(puzzles, args.puzzle_id, model, DEVICE,
                        args.object_by_color, ordering_strategy,
                        args.predict_absolute,
                        args.per_object_anchor, verbose=args.verbose,
                        selection_criterion=selection_criterion, selection_rule=selection_rule,
                        no_train=skip_training)

    print("\nDone!")


if __name__ == "__main__":
    main()