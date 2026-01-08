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
    # Basic usage (auto-screens ordering and selection)
    python relational_position.py --puzzle-id 03560426

    # Force a specific ordering strategy (skips ordering screening)
    python relational_position.py --puzzle-id 03560426 --ordering-strategy left_to_right

    # Force specific selection (skips selection screening)
    python relational_position.py --puzzle-id 03560426 --selection-criterion largest --selection-rule top_1

    # Use different segmentation modes
    python relational_position.py --puzzle-id 03560426 --segmentation-mode color

    # Screen hierarchy as well
    python relational_position.py --puzzle-id 03560426 --screen-hierarchy

Ordering strategies:
    By default, the system auto-screens to find the best ordering strategy.
    Use --ordering-strategy to force a specific one:

    left_to_right, right_to_left, top_to_bottom, bottom_to_top,
    diagonal_tl_br, diagonal_tr_bl, largest_first, smallest_first,
    by_color, adaptive_reading_order, quadrant_order

Selection:
    By default, the system auto-screens to find the best selection criterion
    and rule. Use --selection-criterion and --selection-rule to force specific ones.
"""

import argparse
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from genesis_module import ObjectSpec

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# Object detection and extraction
from object_module import (
    NUM_COLORS,
    MAX_OBJECTS,
    SegmentationMode,
    SegmentationStrategy,
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
    screen_hierarchy_strategies,
    find_best_ordering,
    OrderingScreenResult,
    PerParentOrdering,
    PerParentOrderingResult,
)

# Object correspondence matching
from correspondence_module import (
    find_correspondences,
    CorrespondenceMode,
    DEFAULT_MARGIN
)

# Selection module for ranking-based object selection
from selection_module import (
    RANKING_CRITERIA, SELECTION_RULES,
    compute_selection_mask,
    SelectionScreener
)

# Genesis module for novel object creation
from genesis_module import (
    find_novel_children,
    ObjectSpec, ColorSpec, ShapeSpec, PositionSpec,
    screen_color_hypotheses, screen_position_hypotheses, screen_shape_hypotheses,
    render_object
)
from aggregation_module import aggregate_regions

# Puzzle loading
from puzzle_loader import load_puzzles, load_puzzle

# Transformation module for color/shape transformation rules
from transformation_module import (
    discover_transformation_rules,
    apply_transformation,
    apply_transformation_to_objects,
    TransformationRule,
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
# 'genesis' added for novel object creation (objects with no input correspondence)
FRAMINGS = ['delta', 'grid_tl', 'grid_tr', 'grid_bl', 'grid_br', 'object_relative', 'genesis']
NUM_FRAMINGS = 7


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

        # Genesis specifications for novel object creation
        # These are stored as a dictionary (not a tensor) since ObjectSpec is complex
        self.genesis_specs: Dict[int, 'ObjectSpec'] = {}

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

    # -------------------------------------------------------------------------
    # Genesis Framing Methods
    # -------------------------------------------------------------------------

    def set_genesis_spec(self, obj_idx: int, spec: 'ObjectSpec'):
        """
        Store a genesis specification for generating a novel object.

        Genesis framing is used for objects that have no input correspondence
        and must be created from scratch based on discovered rules.

        Args:
            obj_idx: Which object index
            spec: The ObjectSpec describing how to generate this object
        """
        self.genesis_specs[obj_idx] = spec
        # Also mark this as using genesis framing
        framing_idx = FRAMINGS.index('genesis')
        self.selected_framings[obj_idx] = framing_idx
        self.is_configured[obj_idx] = True
        self.framing_variances[obj_idx] = spec.variance
        self.framing_accuracies[obj_idx] = spec.confidence

    def get_genesis_spec(self, obj_idx: int) -> Optional['ObjectSpec']:
        """
        Get the genesis specification for an object.

        Args:
            obj_idx: Which object index

        Returns:
            ObjectSpec if this object uses genesis framing, None otherwise
        """
        return self.genesis_specs.get(obj_idx)

    def has_genesis_spec(self, obj_idx: int) -> bool:
        """Check if this object has a genesis specification."""
        framing_idx = self.selected_framings[obj_idx].item()
        return FRAMINGS[framing_idx] == 'genesis' and obj_idx in self.genesis_specs

    def get_all_genesis_specs(self) -> Dict[int, 'ObjectSpec']:
        """Get all genesis specifications."""
        return dict(self.genesis_specs)

    def clear_genesis_specs(self):
        """Clear all genesis specifications."""
        self.genesis_specs.clear()

    def get_framing_summary(self, max_obj: int = None) -> str:
        """
        Get a human-readable summary of discovered framings.

        Args:
            max_obj: Maximum object index to include (default: all configured)

        Returns:
            Multi-line string with framing info for each configured object
        """
        lines = []
        if max_obj is None:
            max_obj = self.max_objects

        for obj_idx in range(min(max_obj, self.max_objects)):
            if not self.is_configured[obj_idx]:
                continue

            framing_idx = self.selected_framings[obj_idx].item()
            framing = FRAMINGS[framing_idx]
            accuracy = self.framing_accuracies[obj_idx].item()
            variance = self.framing_variances[obj_idx].item()

            if framing == 'object_relative':
                ref_idx = self.selected_references[obj_idx].item()
                src_anchor, tgt_anchor, offset = self.get_anchor_points(obj_idx)
                lines.append(
                    f"  Object {obj_idx}: {framing} (ref={ref_idx}) "
                    f"[{src_anchor.value.upper()}->{tgt_anchor.value.upper()}, offset={offset}] "
                    f"acc={accuracy:.1%}, var={variance:.6f}"
                )
            elif framing in ['grid_tl', 'grid_tr', 'grid_bl', 'grid_br']:
                grid_anchor, target_pos = self.get_grid_anchor(obj_idx)
                lines.append(
                    f"  Object {obj_idx}: {framing} "
                    f"[anchor={grid_anchor.value.upper()} -> pos={target_pos}] "
                    f"acc={accuracy:.1%}, var={variance:.6f}"
                )
            elif framing == 'genesis':
                spec = self.genesis_specs.get(obj_idx)
                desc = spec.describe() if spec else "no spec"
                lines.append(f"  Object {obj_idx}: genesis - {desc}")
            else:
                offset = self.get_delta_offset(obj_idx) if framing == 'delta' else (0, 0)
                lines.append(
                    f"  Object {obj_idx}: {framing} "
                    f"[offset={offset}] "
                    f"acc={accuracy:.1%}, var={variance:.6f}"
                )

        return "\n".join(lines) if lines else "  (no objects configured)"


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
    """Dataset for position prediction with screening-based zero-shot evaluation."""

    def __init__(self, puzzles: Dict, puzzle_ids: List[str] = None,
                 use_color_only: bool = False, include_test: bool = False,
                 ordering_strategy: Optional[str] = None,
                 selection_criterion: Optional[str] = None,
                 selection_rule: Optional[str] = None,
                 per_parent_ordering: Optional[PerParentOrderingResult] = None,
                 correspondence_mode: CorrespondenceMode = "one_to_one",
                 correspondence_margin: float = DEFAULT_MARGIN):
        self.samples: List[PositionSample] = []
        self.use_color_only = use_color_only
        self.ordering_strategy = ordering_strategy
        self.selection_criterion = selection_criterion
        self.selection_rule = selection_rule
        self.per_parent_ordering = per_parent_ordering
        self.correspondence_mode = correspondence_mode
        self.correspondence_margin = correspondence_margin

        if puzzle_ids is None:
            puzzle_ids = list(puzzles.keys())

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

                sample = self._create_sample(puzzle_id, ex_idx, input_grid, output_grid)
                if sample is not None:
                    self.samples.append(sample)

        print(f"Created {len(self.samples)} position samples")

    def _create_sample(self, puzzle_id: str, ex_idx: int,
                        input_grid: np.ndarray, output_grid: np.ndarray
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

        # Convert to Object instances for correspondence matching
        input_objects = labels_to_objects(input_labels, input_colors, input_bboxes)
        output_objects = labels_to_objects(output_labels, output_colors, output_bboxes)

        # Find correspondences using canonical shape-based matching
        matches, _, _ = find_correspondences(
            input_grid, output_grid,
            input_objects, output_objects,
            threshold=0.0,  # Allow any match
            use_matchable=False,  # Already have flat object lists
            mode=self.correspondence_mode,
            margin=self.correspondence_margin
        )

        if len(matches) == 0:
            return None

        # Reorder input objects so that input[i] maps to output[i]
        # This ensures consistent object indices across training examples
        # (objects going to the same output slot always get the same index)
        if len(matches) == len(input_colors) and len(matches) == len(output_colors):
            # Sort matches by output index
            matches_sorted = sorted(matches, key=lambda x: x[1])  # sort by out_idx

            # Build old_idx -> new_idx mapping based on output order
            old_to_new_input = {}
            for new_idx, (old_in_idx, out_idx, score) in enumerate(matches_sorted):
                old_to_new_input[old_in_idx] = new_idx

            # Only reorder if this creates a different ordering
            needs_reorder = any(old != new for old, new in old_to_new_input.items())

            if needs_reorder:
                new_input_colors = [None] * len(input_colors)
                new_input_bboxes = np.zeros_like(input_bboxes)
                new_input_labels = np.zeros_like(input_labels)

                for old_idx, new_idx in old_to_new_input.items():
                    new_input_colors[new_idx] = input_colors[old_idx]
                    new_input_bboxes[new_idx] = input_bboxes[old_idx]
                    # Relabel: old label (old_idx+1) -> new label (new_idx+1)
                    new_input_labels[input_labels == old_idx + 1] = new_idx + 1

                input_colors = new_input_colors
                input_bboxes = new_input_bboxes
                input_labels = new_input_labels

                # Update selection_mask if it exists
                if selection_mask is not None:
                    new_selection_mask = np.zeros_like(selection_mask)
                    for old_idx, new_idx in old_to_new_input.items():
                        if old_idx < len(selection_mask):
                            new_selection_mask[new_idx] = selection_mask[old_idx]
                    selection_mask = new_selection_mask

                # Update matches to reflect new ordering (now input[i] -> output[i])
                matches = [(new_idx, new_idx, score)
                           for new_idx, (_, _, score) in enumerate(matches_sorted)]

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

        # Compute per-object anchor framing targets
        # Always compute these for screening
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
# Evaluation
# =============================================================================

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


def evaluate_grid_accuracy(
    puzzles: Dict,
    puzzle_id: str,
    anchor_module: 'AnchorFramingModule' = None,
    use_color_only: bool = False,
    ordering_strategy: Optional[str] = None,
    selection_criterion: Optional[str] = None,
    selection_rule: Optional[str] = None,
    include_test: bool = False,
    segmentation_mode: Optional[SegmentationMode] = None,
    predict_fn: Optional[Callable[[np.ndarray, Tuple[int, int]], np.ndarray]] = None
) -> Dict:
    """
    Evaluate grid-level pixel accuracy by comparing predicted grids to actual output grids.

    This compares the entire reconstructed output grid pixel-by-pixel against the
    ground truth output grid, rather than just checking object positions.

    Args:
        puzzles: Dictionary of puzzles
        puzzle_id: ID of puzzle to evaluate
        anchor_module: AnchorFramingModule with screening offsets (optional if predict_fn provided)
        use_color_only: DEPRECATED - use segmentation_mode instead
        ordering_strategy: Optional ordering strategy name
        selection_criterion: Optional ranking criterion for selection
        selection_rule: Optional selection rule
        include_test: If True, also evaluate on test examples
        segmentation_mode: How to segment objects (CONNECTIVITY, PIXEL, or COLOR)
        predict_fn: Optional callable(input_grid, output_shape) -> predicted_grid.
                    If provided, used instead of apply_predicted_transformation.

    Returns:
        Dict with train_accuracy, test_accuracy (if applicable), and per-example details
    """
    if puzzle_id not in puzzles:
        return {'train_accuracy': 0.0, 'error': 'Puzzle not found'}

    # Validate that we have either anchor_module or predict_fn
    if anchor_module is None and predict_fn is None:
        return {'train_accuracy': 0.0, 'error': 'Either anchor_module or predict_fn must be provided'}

    puzzle = puzzles[puzzle_id]
    train_examples = puzzle.get('train', [])
    test_examples = puzzle.get('test', []) if include_test else []

    results = {
        'train_correct': 0,
        'train_total': len(train_examples),
        'train_pixel_correct': 0,
        'train_pixel_total': 0,
        'train_per_example': [],
    }

    # Evaluate training examples
    for i, example in enumerate(train_examples):
        input_grid = np.array(example['input'])
        output_grid = np.array(example['output'])
        output_size = output_grid.shape

        # Use predict_fn if provided, otherwise use apply_predicted_transformation
        if predict_fn is not None:
            predicted_grid = predict_fn(input_grid, output_size)
        else:
            predicted_grid = apply_predicted_transformation(
                input_grid, anchor_module, use_color_only, ordering_strategy,
                output_size=output_size,
                selection_criterion=selection_criterion,
                selection_rule=selection_rule,
                segmentation_mode=segmentation_mode
            )

        # Compare pixel by pixel
        if predicted_grid.shape == output_grid.shape:
            matching_pixels = np.sum(predicted_grid == output_grid)
            total_pixels = output_grid.size
            is_exact_match = np.array_equal(predicted_grid, output_grid)
        else:
            # Shape mismatch - count as all wrong
            matching_pixels = 0
            total_pixels = output_grid.size
            is_exact_match = False

        results['train_pixel_correct'] += matching_pixels
        results['train_pixel_total'] += total_pixels
        if is_exact_match:
            results['train_correct'] += 1

        results['train_per_example'].append({
            'example_idx': i,
            'exact_match': is_exact_match,
            'pixel_accuracy': matching_pixels / total_pixels if total_pixels > 0 else 0.0,
            'matching_pixels': matching_pixels,
            'total_pixels': total_pixels,
        })

    results['train_accuracy'] = results['train_correct'] / max(results['train_total'], 1)
    results['train_pixel_accuracy'] = results['train_pixel_correct'] / max(results['train_pixel_total'], 1)

    # Evaluate test examples if requested
    if include_test and test_examples:
        results['test_correct'] = 0
        results['test_total'] = 0
        results['test_pixel_correct'] = 0
        results['test_pixel_total'] = 0
        results['test_per_example'] = []

        for i, example in enumerate(test_examples):
            if 'output' not in example:
                continue  # Skip if no ground truth

            input_grid = np.array(example['input'])
            output_grid = np.array(example['output'])
            output_size = output_grid.shape
            results['test_total'] += 1

            # Use predict_fn if provided, otherwise use apply_predicted_transformation
            if predict_fn is not None:
                predicted_grid = predict_fn(input_grid, output_size)
            else:
                predicted_grid = apply_predicted_transformation(
                    input_grid, anchor_module, use_color_only, ordering_strategy,
                    output_size=output_size,
                    selection_criterion=selection_criterion,
                    selection_rule=selection_rule,
                    segmentation_mode=segmentation_mode
                )

            if predicted_grid.shape == output_grid.shape:
                matching_pixels = np.sum(predicted_grid == output_grid)
                total_pixels = output_grid.size
                is_exact_match = np.array_equal(predicted_grid, output_grid)
            else:
                matching_pixels = 0
                total_pixels = output_grid.size
                is_exact_match = False

            results['test_pixel_correct'] += matching_pixels
            results['test_pixel_total'] += total_pixels
            if is_exact_match:
                results['test_correct'] += 1

            results['test_per_example'].append({
                'example_idx': i,
                'exact_match': is_exact_match,
                'pixel_accuracy': matching_pixels / total_pixels if total_pixels > 0 else 0.0,
            })

        results['test_accuracy'] = results['test_correct'] / max(results['test_total'], 1)
        results['test_pixel_accuracy'] = results['test_pixel_correct'] / max(results['test_pixel_total'], 1)

    return results


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
            # Inherit when: no anchor points discovered (either non-object_relative framing,
            # or object_relative with only 1 example so discovery returned None)
            needs_inheritance = (
                result['discovered_relation'] is None and
                prev_result is not None and
                prev_result['framing'] == 'object_relative' and
                prev_result['discovered_relation'] is not None and
                prev_result['discovered_relation'].variance < 0.01
            )
            if needs_inheritance:

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


def discover_genesis_for_puzzle(
    puzzles: Dict,
    puzzle_id: str,
    anchor_module: 'AnchorFramingModule',
    verbose: bool = False,
    segmentation_strategy: Optional[SegmentationStrategy] = None
) -> List[ObjectSpec]:
    """
    Discover genesis rules for novel children in the puzzle.

    This function finds children that appear in output but not input,
    and discovers rules for creating them.

    Args:
        puzzles: Dictionary of all puzzles
        puzzle_id: ID of the puzzle to analyze
        anchor_module: The anchor module to store genesis specs
        verbose: Print detailed info
        segmentation_strategy: How to segment input/output grids

    Returns:
        List of discovered ObjectSpecs
    """
    if segmentation_strategy is None:
        segmentation_strategy = SegmentationStrategy()

    if puzzle_id not in puzzles:
        return []

    puzzle = puzzles[puzzle_id]
    examples = puzzle.get('train', [])

    if not examples:
        return []

    # Process each example to find novel children
    novel_per_ex = []
    input_objs_per_ex = []
    output_objs_per_ex = []
    input_grids = []
    grid_sizes = []
    regions_per_ex = []

    for ex in examples:
        if 'output' not in ex:
            continue

        input_grid = np.array(ex['input'])
        output_grid = np.array(ex['output'])

        input_objects = extract_objects_from_grid(input_grid, segmentation_mode=segmentation_strategy.get_mode("input"))
        output_objects = extract_objects_from_grid(output_grid, segmentation_mode=segmentation_strategy.get_mode("output"))

        novel = find_novel_children(input_objects, output_objects, input_grid, output_grid)
        regions = aggregate_regions(input_grid, input_objects)

        novel_per_ex.append(novel)
        input_objs_per_ex.append(input_objects)
        output_objs_per_ex.append(output_objects)
        input_grids.append(input_grid)
        grid_sizes.append(output_grid.shape)
        regions_per_ex.append(regions)

    # Check if there are novel children
    novel_counts = [len(n) for n in novel_per_ex]
    if not any(c > 0 for c in novel_counts):
        if verbose:
            print("  No novel children found")
        return []

    # Check consistency
    if not all(c == novel_counts[0] for c in novel_counts):
        if verbose:
            print(f"  Warning: Inconsistent novel child counts: {novel_counts}")
        # Use minimum count
        min_count = min(novel_counts)
        for i in range(len(novel_per_ex)):
            novel_per_ex[i] = novel_per_ex[i][:min_count]

    num_novel = novel_counts[0] if novel_counts else 0
    if num_novel == 0:
        return []

    if verbose:
        print(f"  Found {num_novel} novel child(ren) per example")

    # Discover specs for each novel child
    discovered_specs = []

    for novel_idx in range(num_novel):
        # Collect this novel child across all examples
        novel_at_idx = [[ex[novel_idx]] if novel_idx < len(ex) else []
                        for ex in novel_per_ex]

        if any(len(n) == 0 for n in novel_at_idx):
            continue

        # Screen color hypotheses
        color_hyps = screen_color_hypotheses(
            novel_at_idx, input_objs_per_ex, input_grids, regions_per_ex
        )

        # Screen position hypotheses
        pos_hyps = screen_position_hypotheses(
            novel_at_idx, input_objs_per_ex, output_objs_per_ex,
            grid_sizes, regions_per_ex
        )

        # Screen shape hypotheses
        shape_hyps = screen_shape_hypotheses(
            novel_at_idx, input_objs_per_ex, input_grids
        )

        # Get best hypotheses
        best_color = color_hyps[0][0] if color_hyps else ColorSpec.literal(0)
        best_pos = pos_hyps[0][0] if pos_hyps else PositionSpec.grid_relative(
            AnchorPoint.CENTER
        )
        best_shape = shape_hyps[0][0] if shape_hyps else ShapeSpec.pixel()

        # Compute confidence
        color_conf = color_hyps[0][1] if color_hyps else 0.0
        pos_var = pos_hyps[0][1] if pos_hyps else float('inf')
        shape_conf = shape_hyps[0][1] if shape_hyps else 0.0

        confidence = (color_conf + shape_conf) / 2.0 if pos_var < 1e-6 else 0.0

        spec = ObjectSpec(
            color=best_color,
            shape=best_shape,
            position=best_pos,
            confidence=confidence,
            variance=pos_var,
            discovered_from_examples=len(examples)
        )

        discovered_specs.append(spec)

        if verbose:
            print(f"  Novel child {novel_idx}: {spec.describe()}")
            print(f"    Confidence: {confidence:.1%}, Position variance: {pos_var:.6f}")

        # Assign a unique object index for this genesis object
        # Use indices after max regular objects
        genesis_obj_idx = MAX_OBJECTS - 1 - novel_idx
        anchor_module.set_genesis_spec(genesis_obj_idx, spec)

        if verbose:
            print(f"    Stored as genesis object {genesis_obj_idx}")

    return discovered_specs


# =============================================================================
# Visualization
# =============================================================================

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

    def __init__(self, puzzles: Dict, puzzle_id: str,
                 anchor_module: 'AnchorFramingModule' = None,
                 use_color_only: bool = False,
                 ordering_strategy: Optional[str] = None,
                 verbose: bool = False,
                 selection_criterion: Optional[str] = None,
                 selection_rule: Optional[str] = None,
                 segmentation_mode: Optional[SegmentationMode] = None,
                 predict_fn: Optional[Callable[[np.ndarray, Tuple[int, int]], np.ndarray]] = None):
        self.puzzle_id = puzzle_id
        self.anchor_module = anchor_module
        self.use_color_only = use_color_only
        self.ordering_strategy = ordering_strategy
        self.verbose = verbose
        self.selection_criterion = selection_criterion
        self.selection_rule = selection_rule
        self.segmentation_mode = segmentation_mode
        self.predict_fn = predict_fn

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
            if self.predict_fn is not None:
                predicted = self.predict_fn(input_grid, output_size)
            else:
                predicted = apply_predicted_transformation(
                    input_grid, self.anchor_module, use_color_only, ordering_strategy,
                    verbose=False, output_size=output_size,
                    selection_criterion=self.selection_criterion, selection_rule=self.selection_rule,
                    segmentation_mode=self.segmentation_mode
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
            if self.predict_fn is not None:
                entry['predicted'] = self.predict_fn(entry['input'], output_size)
            else:
                entry['predicted'] = apply_predicted_transformation(
                    entry['input'], self.anchor_module, use_color_only, ordering_strategy,
                    verbose=verbose, output_size=output_size,
                    selection_criterion=self.selection_criterion, selection_rule=self.selection_rule,
                    segmentation_mode=self.segmentation_mode
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


def visualize_grids(puzzles: Dict, puzzle_id: str,
                    anchor_module: 'AnchorFramingModule' = None,
                    use_color_only: bool = False,
                    ordering_strategy: Optional[str] = None,
                    verbose: bool = False,
                    selection_criterion: Optional[str] = None,
                    selection_rule: Optional[str] = None,
                    segmentation_mode: Optional[SegmentationMode] = None,
                    predict_fn: Optional[Callable[[np.ndarray, Tuple[int, int]], np.ndarray]] = None):
    """
    Visualize the training pairs and test prediction with actual grid shapes.
    Uses an interactive viewer with keyboard navigation.

    Args:
        predict_fn: Optional callable(input_grid, output_shape) -> predicted_grid.
                    If provided, used instead of apply_predicted_transformation.
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

    viewer = InteractiveGridViewer(puzzles, puzzle_id, anchor_module, use_color_only,
                                   ordering_strategy, verbose=verbose,
                                   selection_criterion=selection_criterion, selection_rule=selection_rule,
                                   segmentation_mode=segmentation_mode, predict_fn=predict_fn)
    viewer.show()


def save_overview_image(puzzles: Dict, puzzle_id: str,
                        anchor_module: 'AnchorFramingModule',
                        use_color_only: bool = False,
                        ordering_strategy: Optional[str] = None,
                        selection_criterion: Optional[str] = None,
                        selection_rule: Optional[str] = None,
                        segmentation_mode: Optional[SegmentationMode] = None):
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
            input_grid, anchor_module, use_color_only, ordering_strategy,
            output_size=output_size,
            selection_criterion=selection_criterion, selection_rule=selection_rule,
            segmentation_mode=segmentation_mode
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
            input_grid, anchor_module, use_color_only, ordering_strategy,
            output_size=output_size,
            selection_criterion=selection_criterion, selection_rule=selection_rule,
            segmentation_mode=segmentation_mode
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


def apply_predicted_transformation(input_grid: np.ndarray,
                                     anchor_module: 'AnchorFramingModule',
                                     use_color_only: bool = False,
                                     ordering_strategy: Optional[str] = None,
                                     verbose: bool = False,
                                     output_size: tuple = None,
                                     selection_criterion: Optional[str] = None,
                                     selection_rule: Optional[str] = None,
                                     segmentation_mode: Optional[SegmentationMode] = None
                                     ) -> np.ndarray:
    """
    Apply screening-discovered offsets to create a predicted output grid.

    Args:
        input_grid: Input grid to transform
        anchor_module: AnchorFramingModule with discovered framings and offsets
        use_color_only: DEPRECATED - use segmentation_mode instead
        ordering_strategy: Optional ordering strategy name (e.g., 'left_to_right', 'top_to_bottom')
        verbose: If True, print detailed placement logic for debugging
        output_size: Optional (H, W) tuple for output grid size. If None, uses input size.
        selection_criterion: Optional ranking criterion for object selection (e.g., 'largest', 'smallest')
        selection_rule: Optional selection rule (e.g., 'top_1', 'top_2')
        segmentation_mode: How to segment objects (CONNECTIVITY, PIXEL, or COLOR)
    """
    H, W = input_grid.shape
    # Use output_size if provided, otherwise default to input size
    out_H, out_W = output_size if output_size is not None else (H, W)
    grid_size = max(H, W, out_H, out_W, GRID_SIZE)

    # Resolve segmentation mode (backward compatibility)
    if segmentation_mode is None:
        segmentation_mode = SegmentationMode.COLOR if use_color_only else SegmentationMode.CONNECTIVITY

    # Extract objects from input
    input_labels, input_colors, input_bboxes, _ = extract_connected_components(
        input_grid, segmentation_mode=segmentation_mode
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

    # Use screening offsets directly
    input_centroids_pixels = props['centroids'] * grid_size
    input_bboxes_pixels = props['bboxes'] * grid_size

    if verbose:
        print(f"\n{'='*60}")
        print("VERBOSE: Object Placement Logic")
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
        if not anchor_module.is_configured[obj_idx]:
            decoded_positions[obj_idx] = input_centroids_pixels[obj_idx]
            continue

        input_centroid = input_centroids_pixels[obj_idx]
        input_bbox = input_bboxes_pixels[obj_idx]

        # Get source size from input bbox
        src_height = max(1, int(round(input_bbox[2] - input_bbox[0])) + 1)
        src_width = max(1, int(round(input_bbox[3] - input_bbox[1])) + 1)
        source_size = (src_height, src_width)

        # Get reference info if needed
        ref_idx = anchor_module.get_reference_idx(obj_idx)
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
            anchor_module, obj_idx, input_centroid,
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
            framing = FRAMINGS[anchor_module.get_framing_idx(obj_idx)]
            print(f"  Object {obj_idx}: {framing} -> centroid={predicted_centroid}")

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

        # Decoded positions are output centroids in pixel coordinates
        current_centroid_row = rows.mean()
        current_centroid_col = cols.mean()
        pred_centroid_row = decoded_positions[obj_idx, 0]
        pred_centroid_col = decoded_positions[obj_idx, 1]
        delta_row = int(round(pred_centroid_row - current_centroid_row))
        delta_col = int(round(pred_centroid_col - current_centroid_col))

        # Clamp deltas to keep object within output grid bounds
        # Ensure top-left doesn't go negative
        delta_row = max(delta_row, -current_min_row)
        delta_col = max(delta_col, -current_min_col)
        # Ensure bottom-right doesn't exceed output grid
        delta_row = min(delta_row, out_H - 1 - current_max_row)
        delta_col = min(delta_col, out_W - 1 - current_max_col)

        # Move each pixel of the object, preserving original colors
        for r, c in zip(rows, cols):
            new_r = r + delta_row
            new_c = c + delta_col

            # Check bounds against output grid size
            if 0 <= new_r < out_H and 0 <= new_c < out_W:
                # Use actual input pixel color, not region's mode color
                output_grid[new_r, new_c] = input_grid[r, c]

    # Render genesis objects (novel children)
    genesis_specs = anchor_module.get_all_genesis_specs()
    if genesis_specs:
        # Get input objects for reference
        input_objects = extract_objects_from_grid(input_grid, segmentation_mode=segmentation_mode)
        regions = aggregate_regions(input_grid, input_objects)

        for obj_idx, spec in genesis_specs.items():
            # Render the genesis object directly onto the output grid
            render_object(spec, output_grid, input_grid, input_objects, None, regions)

            if verbose:
                print(f"Genesis object {obj_idx}: rendered {spec.describe()}")

    return output_grid


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Relational Position Prediction for ARC (Zero-Shot Screening)")

    parser.add_argument("--puzzle-id", type=str, required=True,
                        help="Puzzle ID to evaluate")
    parser.add_argument("--dataset", type=str, default="arc-agi-1",
                        choices=["arc-agi-1", "arc-agi-2"])
    parser.add_argument("--data-root", type=str, default="kaggle/combined")

    parser.add_argument("--object-by-color", action="store_true",
                        help="DEPRECATED: Use --segmentation-mode color instead")
    parser.add_argument("--segmentation-mode", type=str,
                        choices=['connectivity', 'pixel', 'color'],
                        default=None,
                        help="Object segmentation mode for BOTH grids (shorthand). "
                             "Use --input-segmentation-mode and --output-segmentation-mode for different modes.")
    parser.add_argument("--input-segmentation-mode", type=str,
                        choices=['connectivity', 'pixel', 'color'],
                        default=None,
                        help="Segmentation mode for INPUT grids (overrides --segmentation-mode)")
    parser.add_argument("--output-segmentation-mode", type=str,
                        choices=['connectivity', 'pixel', 'color'],
                        default=None,
                        help="Segmentation mode for OUTPUT grids (overrides --segmentation-mode)")

    # Ordering strategy flags
    parser.add_argument("--ordering-strategy", type=str,
                        choices=['left_to_right', 'right_to_left', 'top_to_bottom',
                                'bottom_to_top', 'diagonal_tl_br', 'diagonal_tr_bl',
                                'largest_first', 'smallest_first', 'by_color',
                                'adaptive_reading_order', 'quadrant_order'],
                        help="Force a specific ordering strategy (default: auto-screen to find best)")
    parser.add_argument("--left-to-right", action="store_true",
                        help="DEPRECATED: Use --ordering-strategy left_to_right instead")

    # Screening parameters
    parser.add_argument("--screening-epochs", type=int, default=100,
                        help="Number of epochs per framing during screening phase")

    # Selection filtering flags
    parser.add_argument("--selection-criterion", type=str, choices=RANKING_CRITERIA,
                        help="Force a specific selection criterion (default: auto-screen to find best)")
    parser.add_argument("--selection-rule", type=str, choices=SELECTION_RULES,
                        help="Force a specific selection rule (default: auto-screen to find best)")
    parser.add_argument("--screen-hierarchy", action="store_true",
                        help="Auto-screen to determine if hierarchical object representation is beneficial")

    # Correspondence matching flags
    parser.add_argument("--correspondence-mode", type=str, default="one_to_one",
                        choices=["one_to_one", "many_to_one", "one_to_many"],
                        help="Correspondence matching mode: one_to_one (default), "
                             "many_to_one (multiple inputs to one output), "
                             "one_to_many (one input to multiple outputs)")
    parser.add_argument("--correspondence-margin", type=float, default=DEFAULT_MARGIN,
                        help=f"For non-one_to_one modes, how close to best score to allow secondary matches (default: {DEFAULT_MARGIN})")

    # Transformation mode flag
    parser.add_argument("--transformation", action="store_true",
                        help="Use transformation module instead of object repositioning. "
                             "Useful when outputs are derived from inputs via color/shape rules "
                             "(e.g., fill grid with most common color)")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-visualize", action="store_true",
                        help="Skip visualization")
    parser.add_argument("--verbose", action="store_true",
                        help="Print detailed object placement logic during test visualization")

    args = parser.parse_args()

    # Build segmentation strategy from arguments
    # Priority: specific mode > general mode > default (connectivity)
    # Also handle backward compatibility with --object-by-color
    if args.object_by_color:
        base_mode = SegmentationMode.COLOR
    elif args.segmentation_mode:
        base_mode = SegmentationMode(args.segmentation_mode)
    else:
        base_mode = SegmentationMode.CONNECTIVITY

    input_segmentation_mode = SegmentationMode(args.input_segmentation_mode) if args.input_segmentation_mode else base_mode
    output_segmentation_mode = SegmentationMode(args.output_segmentation_mode) if args.output_segmentation_mode else base_mode

    segmentation_strategy = SegmentationStrategy(
        input_mode=input_segmentation_mode,
        output_mode=output_segmentation_mode
    )

    # For backward compatibility, keep segmentation_mode as input mode (most common use case)
    segmentation_mode = input_segmentation_mode

    # Set seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    print(f"Device: {DEVICE}")
    print(f"Puzzle: {args.puzzle_id}")
    print(f"Object segmentation: {segmentation_strategy}")
    print("Decoding mode: PER-OBJECT ANCHOR (discovers per-object-index framings via screening)")
    print(f"  Screening epochs per framing: {args.screening_epochs}")
    if args.correspondence_mode != "one_to_one":
        print(f"Correspondence mode: {args.correspondence_mode} (margin={args.correspondence_margin})")

    # Load puzzles
    print("\nLoading puzzles...")

    # Handle synthetic puzzles specially
    if args.puzzle_id.startswith("syn_"):
        try:
            puzzle_data = load_puzzle(args.puzzle_id)
            puzzles = {args.puzzle_id: puzzle_data}
        except ValueError as e:
            print(f"Error: {e}")
            return
    else:
        puzzles = load_puzzles(args.dataset, args.data_root)
        if args.puzzle_id not in puzzles:
            print(f"Error: puzzle {args.puzzle_id} not found")
            print(f"Available puzzles: {len(puzzles)}")
            return

    # ==========================================================================
    # TRANSFORMATION MODE - use transformation module instead of repositioning
    # ==========================================================================
    if args.transformation:
        print("\n" + "=" * 60)
        print("TRANSFORMATION MODE")
        print("=" * 60)
        print(f"Input segmentation: {input_segmentation_mode.name}")
        print(f"Output segmentation: {output_segmentation_mode.name}")
        print(f"Correspondence mode: {args.correspondence_mode}")

        # Discover transformation rules
        rule = discover_transformation_rules(
            puzzles, args.puzzle_id,
            input_segmentation_mode=input_segmentation_mode,
            output_segmentation_mode=output_segmentation_mode,
            correspondence_mode=args.correspondence_mode,
            correspondence_margin=args.correspondence_margin,
            verbose=args.verbose
        )

        if rule is None:
            print("Error: No transformation rule discovered")
            return

        print(f"\nDiscovered Rule: {rule.describe()}")
        print(f"Total Variance: {rule.total_variance:.4f}")

        # Create prediction function for use with standard evaluation/visualization
        def transformation_predict_fn(input_grid: np.ndarray, output_shape: Tuple[int, int]) -> np.ndarray:
            """Prediction function that wraps apply_transformation for the discovered rule."""
            return apply_transformation(
                input_grid, rule,
                input_segmentation_mode=input_segmentation_mode,
                output_shape=output_shape
            )

        # Check if test data has ground truth
        puzzle = puzzles[args.puzzle_id]
        has_test_data = puzzle.get('test') and any('output' in ex for ex in puzzle['test'])

        # Use standard grid accuracy evaluation
        print("\n" + "=" * 60)
        print("Grid Pixel Accuracy")
        print("=" * 60)
        grid_results = evaluate_grid_accuracy(
            puzzles, args.puzzle_id,
            include_test=has_test_data,
            predict_fn=transformation_predict_fn
        )
        print(f"\n--- Training Set ---")
        print(f"Grid Accuracy: {grid_results['train_accuracy']:.1%} ({grid_results['train_correct']}/{grid_results['train_total']} exact matches)")
        print(f"Pixel Accuracy: {grid_results['train_pixel_accuracy']:.1%} ({grid_results['train_pixel_correct']}/{grid_results['train_pixel_total']} pixels)")
        for ex in grid_results['train_per_example']:
            status = "PASS" if ex['exact_match'] else "FAIL"
            print(f"  Example {ex['example_idx'] + 1}: {status} ({ex['pixel_accuracy']:.1%} pixels correct)")

        if has_test_data and 'test_accuracy' in grid_results:
            print(f"\n--- Test Set ---")
            print(f"Grid Accuracy: {grid_results['test_accuracy']:.1%} ({grid_results['test_correct']}/{grid_results['test_total']} exact matches)")
            print(f"Pixel Accuracy: {grid_results['test_pixel_accuracy']:.1%} ({grid_results['test_pixel_correct']}/{grid_results['test_pixel_total']} pixels)")
            for ex in grid_results['test_per_example']:
                status = "PASS" if ex['exact_match'] else "FAIL"
                print(f"  Example {ex['example_idx'] + 1}: {status} ({ex['pixel_accuracy']:.1%} pixels correct)")

        # Use standard visualization
        if not args.no_visualize:
            print("\n" + "=" * 60)
            print("Grid Visualization")
            print("=" * 60)
            visualize_grids(puzzles, args.puzzle_id,
                           verbose=args.verbose,
                           predict_fn=transformation_predict_fn)

        print("\nDone!")
        return

    # Selection screening/configuration
    # Screen by default unless user provides explicit criterion/rule
    selection_criterion = args.selection_criterion
    selection_rule = args.selection_rule
    explicit_selection = selection_criterion is not None or selection_rule is not None

    if not explicit_selection:
        print("\n" + "=" * 60)
        print("SELECTION SCREENING")
        print("=" * 60)

        screener = SelectionScreener(verbose=args.verbose)
        selection_samples = screener.create_selection_samples(
            puzzles, [args.puzzle_id], use_color_only=args.object_by_color,
            strategy=segmentation_strategy,
            correspondence_mode=args.correspondence_mode,
            correspondence_margin=args.correspondence_margin
        )

        if selection_samples:
            # Analyze selection pattern
            pattern = screener.analyze_selection_pattern(selection_samples)
            print(f"Selection pattern: {pattern.get('description', pattern.get('pattern', 'unknown'))}")

            # If all objects are selected, skip selection screening entirely
            if pattern.get('pattern') == 'all_selected':
                print("\nAll input objects appear in output - skipping selection filtering")
                # Leave selection_criterion and selection_rule as None (no filtering)
            else:
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
    else:
        print(f"\n*** Using explicit selection: {selection_criterion or 'none'} + {selection_rule or 'none'} ***")

    # Print selection config
    if selection_criterion and selection_rule:
        print(f"\nSelection filtering: {selection_criterion} + {selection_rule}")
    else:
        print("\nSelection filtering: disabled (using all objects)")

    # Ordering strategy configuration
    # Screen by default unless user provides explicit ordering strategy
    ordering_strategy = args.ordering_strategy
    explicit_ordering = ordering_strategy is not None  # Track if user explicitly provided one

    # Handle deprecated --left-to-right flag
    if args.left_to_right and not ordering_strategy:
        print("Warning: --left-to-right is deprecated. Use --ordering-strategy left_to_right")
        ordering_strategy = 'left_to_right'
        explicit_ordering = True

    # Per-parent ordering result (populated if find_best_ordering selects per_parent mode)
    per_parent_ordering_result: Optional[PerParentOrderingResult] = None

    if not explicit_ordering:
        print("\n" + "=" * 60)
        print("ORDERING SCREENING (unified: global + per-parent)")
        print("=" * 60)

        puzzle = puzzles[args.puzzle_id]

        # Use unified find_best_ordering which handles both global and per-parent orderings
        ordering_screen_result = find_best_ordering(
            puzzle,
            verbose=args.verbose,
            selection_criterion=selection_criterion,
            selection_rule=selection_rule,
            use_color_only=args.object_by_color
        )

        print(f"\nBest ordering configuration found:")
        print(f"  Mode: {ordering_screen_result.best_mode}")
        print(f"  Strategy: {ordering_screen_result.best_ordering_name}")
        print(f"  Consistent: {ordering_screen_result.is_consistent}")
        print(f"  Has hierarchy: {ordering_screen_result.has_hierarchy}")

        if ordering_screen_result.best_mode == 'per_parent' and ordering_screen_result.per_parent_result:
            per_parent_result = ordering_screen_result.per_parent_result
            print(f"\nPer-parent ordering details:")
            print(f"  Child ordering: {per_parent_result.child_ordering.name}")
            print(f"  Avg variance: {per_parent_result.avg_variance:.4f}")
            print(f"  Learned offsets by child index:")
            for idx, offset in sorted(per_parent_result.learned_offsets.items()):
                print(f"    Index {idx}: parent + {offset}")

        # Show all global results if verbose
        if args.verbose:
            print("\nAll global ordering results:")
            for name, result in ordering_screen_result.global_results.items():
                status = "CONSISTENT" if result.get('consistent', False) else "inconsistent"
                num_framings = result.get('num_framings', 0)
                print(f"  {name:20s}: {status} ({num_framings} framings)")

        # Use global ordering name (per-parent mode uses child_ordering internally)
        if ordering_screen_result.best_mode == 'per_parent':
            # For per-parent mode, use the child ordering strategy for global sorting
            # but also store the per-parent result for later use
            ordering_strategy = ordering_screen_result.per_parent_result.child_ordering.name
            per_parent_ordering_result = ordering_screen_result.per_parent_result
            print(f"\n*** Using per-parent ordering with child strategy: {ordering_strategy} ***")
        else:
            ordering_strategy = ordering_screen_result.best_ordering_name
    else:
        print(f"\n*** Using explicit ordering strategy: {ordering_strategy} ***")

    # Print ordering config
    if ordering_strategy:
        print(f"\nObject ordering: {ordering_strategy}")
    else:
        print("\nObject ordering: disabled (extraction order)")

    # Hierarchy screening
    use_hierarchy = False
    hierarchy_results = None
    if args.screen_hierarchy:
        print("\n" + "=" * 60)
        print("HIERARCHY SCREENING")
        print("=" * 60)

        puzzle = puzzles[args.puzzle_id]
        hierarchy_results = screen_hierarchy_strategies(puzzle, verbose=args.verbose)

        use_hierarchy = hierarchy_results['use_hierarchy']
        if use_hierarchy:
            print(f"\n*** Hierarchy mode selected ***")
            print(f"  Hierarchy score (variance): {hierarchy_results['hierarchy_score']:.4f}")
            if hierarchy_results['hierarchy_stats']:
                stats = hierarchy_results['hierarchy_stats']
                print(f"  Composite roots: {stats['num_composite']}")
                print(f"  Max depth: {stats['max_depth']}")
            if hierarchy_results['parent_child_relations']:
                rel = hierarchy_results['parent_child_relations'][0]
                print(f"  Best parent-child relation: {rel.relation.describe()}")
        else:
            print(f"\n*** Flat mode selected (hierarchy not beneficial) ***")
            print(f"  Flat score: {hierarchy_results['flat_score']:.4f}")
            print(f"  Hierarchy score: {hierarchy_results['hierarchy_score']:.4f}")

    # Create TRAINING dataset (train examples only)
    print("\nCreating training dataset (train pairs only)...")
    train_dataset = PositionDataset(
        puzzles,
        puzzle_ids=[args.puzzle_id],
        use_color_only=args.object_by_color,
        include_test=False,  # Train only on training pairs
        ordering_strategy=ordering_strategy,
        selection_criterion=selection_criterion,
        selection_rule=selection_rule,
        per_parent_ordering=per_parent_ordering_result,
        correspondence_mode=args.correspondence_mode,
        correspondence_margin=args.correspondence_margin
    )

    if len(train_dataset) == 0:
        print("Error: no valid training samples created")
        return

    # Create TEST dataset (test examples only)
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
        ordering_strategy=ordering_strategy,
        selection_criterion=selection_criterion,
        selection_rule=selection_rule,
        per_parent_ordering=per_parent_ordering_result,
        correspondence_mode=args.correspondence_mode,
        correspondence_margin=args.correspondence_margin
    )

    has_test_data = len(test_dataset) > 0
    if has_test_data:
        print(f"Test samples: {len(test_dataset)}")
    else:
        print("Warning: No test samples with ground truth available")

    # Create anchor framing module
    print("\nCreating anchor framing module...")
    anchor_module = AnchorFramingModule().to(DEVICE)

    # Screening phase - discover best framing per object index
    print("\n" + "=" * 60)
    print("Screening Phase")
    print("=" * 60)

    trainer = AnchorScreeningTrainer(screening_epochs=args.screening_epochs, lr=0.01, verbose=args.verbose)
    _screening_results = trainer.run_screening(train_dataset, anchor_module, DEVICE)

    # Print screening results summary
    print("\nScreening Results:")
    max_obj = max(sum(s.input_valid) for s in train_dataset.samples) if train_dataset.samples else 0
    print(anchor_module.get_framing_summary(max_obj))

    # Genesis discovery - find novel children and create rules for them
    print("\n" + "-" * 40)
    print("GENESIS DISCOVERY (novel children)")
    print("-" * 40)
    genesis_specs = discover_genesis_for_puzzle(
        puzzles, args.puzzle_id, anchor_module, verbose=args.verbose,
        segmentation_strategy=segmentation_strategy
    )
    if genesis_specs:
        print(f"Discovered {len(genesis_specs)} genesis rule(s)")
        for i, spec in enumerate(genesis_specs):
            print(f"  Genesis {i}: {spec.describe()}")
    else:
        print("No genesis rules needed (no novel children)")

    # Report variance status
    if anchor_module.all_low_variance(threshold=0.001):
        max_var = anchor_module.get_max_variance()
        print(f"\n*** All objects have near-zero variance (max={max_var:.6f}) ***")
        print("*** Using screening-discovered offsets directly ***")

    print("\n" + "=" * 60)
    print("Using screening-discovered offsets (zero-shot)")
    print("=" * 60)

    # Final evaluation using screening offsets
    print("\n" + "=" * 60)
    print("Final Evaluation")
    print("=" * 60)

    # Evaluate using screening offsets directly
    print("\n--- Training Set (using screening offsets) ---")
    train_final = evaluate_no_train(anchor_module, train_dataset.samples)
    print(f"Pixel Accuracy: {train_final['pixel_accuracy']:.1%}")
    print(f"Mean Pixel Error: {train_final['mean_pixel_error']:.2f}px")
    print(f"Correct/Total: {train_final['total_correct']}/{train_final['total_samples']}")

    # Per-object breakdown
    if train_final['per_object_accuracy']:
        print("\nPer-object accuracy:")
        for obj_idx, acc in sorted(train_final['per_object_accuracy'].items()):
            framing_idx = anchor_module.get_framing_idx(obj_idx)
            framing = FRAMINGS[framing_idx]
            print(f"  Object {obj_idx}: {acc:.1%} ({framing})")

    if has_test_data:
        print("\n--- Test Set (using screening offsets) ---")
        test_final = evaluate_no_train(anchor_module, test_dataset.samples)
        print(f"Pixel Accuracy: {test_final['pixel_accuracy']:.1%}")
        print(f"Mean Pixel Error: {test_final['mean_pixel_error']:.2f}px")
        print(f"Correct/Total: {test_final['total_correct']}/{test_final['total_samples']}")

        # Per-object breakdown
        if test_final['per_object_accuracy']:
            print("\nPer-object accuracy:")
            for obj_idx, acc in sorted(test_final['per_object_accuracy'].items()):
                framing_idx = anchor_module.get_framing_idx(obj_idx)
                framing = FRAMINGS[framing_idx]
                print(f"  Object {obj_idx}: {acc:.1%} ({framing})")

    # Grid-level pixel accuracy evaluation
    print("\n" + "=" * 60)
    print("Grid Pixel Accuracy")
    print("=" * 60)
    grid_results = evaluate_grid_accuracy(
        puzzles, args.puzzle_id, anchor_module,
        use_color_only=args.object_by_color,
        ordering_strategy=ordering_strategy,
        selection_criterion=selection_criterion,
        selection_rule=selection_rule,
        include_test=has_test_data,
        segmentation_mode=segmentation_mode
    )
    print(f"\n--- Training Set ---")
    print(f"Grid Accuracy: {grid_results['train_accuracy']:.1%} ({grid_results['train_correct']}/{grid_results['train_total']} exact matches)")
    print(f"Pixel Accuracy: {grid_results['train_pixel_accuracy']:.1%} ({grid_results['train_pixel_correct']}/{grid_results['train_pixel_total']} pixels)")
    for ex in grid_results['train_per_example']:
        status = "PASS" if ex['exact_match'] else "FAIL"
        print(f"  Example {ex['example_idx'] + 1}: {status} ({ex['pixel_accuracy']:.1%} pixels correct)")

    if has_test_data and 'test_accuracy' in grid_results:
        print(f"\n--- Test Set ---")
        print(f"Grid Accuracy: {grid_results['test_accuracy']:.1%} ({grid_results['test_correct']}/{grid_results['test_total']} exact matches)")
        print(f"Pixel Accuracy: {grid_results['test_pixel_accuracy']:.1%} ({grid_results['test_pixel_correct']}/{grid_results['test_pixel_total']} pixels)")
        for ex in grid_results['test_per_example']:
            status = "PASS" if ex['exact_match'] else "FAIL"
            print(f"  Example {ex['example_idx'] + 1}: {status} ({ex['pixel_accuracy']:.1%} pixels correct)")

    # Visualize grids with matplotlib
    if not args.no_visualize:
        print("\n" + "=" * 60)
        print("Grid Visualization")
        print("=" * 60)
        visualize_grids(puzzles, args.puzzle_id, anchor_module,
                        args.object_by_color, ordering_strategy,
                        verbose=args.verbose,
                        selection_criterion=selection_criterion, selection_rule=selection_rule,
                        segmentation_mode=segmentation_mode)

    print("\nDone!")


if __name__ == "__main__":
    main()