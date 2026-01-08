#!/usr/bin/env python3
"""
Object Module for ARC Puzzle Solver

Unified object detection and extraction for ARC puzzles.
This module provides the canonical object detection implementation used across
the codebase (anchoring_module, ordering_module, slot_viz, relational_position, crm).

The object detection approach matches AffinitySlotAttention in crm.py:
- Dynamic background detection via edge-connected component heuristic
- 4-connectivity for background color (so diagonal lines can act as barriers)
- 8-connectivity for foreground colors
- Exterior mask computation to identify "true" background vs enclosed regions

Key functions:
    extract_connected_components: Extract objects with background-aware detection
    extract_objects_from_grid: Extract objects as List[Object] format
    detect_background_color: Find background using edge-connected heuristic
    compute_exterior_mask: Find background pixels reachable from edges

Segmentation modes (SegmentationMode enum):
    CONNECTIVITY: Default. Connected component analysis with divider detection.
    PIXEL: Each non-background pixel becomes its own object.
    COLOR: All pixels of the same color form a single object.

Usage:
    from object_module import (
        Object,
        SegmentationMode,
        extract_connected_components,
        extract_objects_from_grid,
        detect_background_color,
    )

    # Command-line visualization:
    python object_module.py --puzzle-id 1990f7a8
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Set, Optional, Any
from enum import Enum
from scipy import ndimage

from puzzle_loader import load_puzzle as _load_puzzle


# =============================================================================
# Constants
# =============================================================================

NUM_COLORS = 10
MAX_OBJECTS = 40


# =============================================================================
# Segmentation Modes
# =============================================================================

class SegmentationMode(Enum):
    """Defines how objects are segmented from the grid.

    CONNECTIVITY: Default mode. Uses connected component analysis with
                  8-connectivity for foreground colors and 4-connectivity
                  for background. Respects divider lines.

    PIXEL: Each non-background pixel becomes its own object. Useful for
           puzzles where individual pixel positions matter.

    COLOR: All pixels of the same color form a single object, regardless
           of connectivity. Useful for color-based transformations.
    """
    CONNECTIVITY = "connectivity"
    PIXEL = "pixel"
    COLOR = "color"


@dataclass
class SegmentationStrategy:
    """Configures how input and output grids are segmented.

    Allows different segmentation strategies for input vs output grids.
    This is useful when input objects are connected components but output
    treats each pixel separately, or vice versa.

    Attributes:
        input_mode: Segmentation mode for input grids
        output_mode: Segmentation mode for output grids

    Usage:
        # Different modes for input vs output
        strategy = SegmentationStrategy(
            input_mode=SegmentationMode.CONNECTIVITY,
            output_mode=SegmentationMode.PIXEL
        )

        # Uniform mode (same for both)
        strategy = SegmentationStrategy.uniform(SegmentationMode.COLOR)

        # From legacy single mode (backward compatible)
        strategy = SegmentationStrategy.from_mode(SegmentationMode.CONNECTIVITY)
    """
    input_mode: SegmentationMode = SegmentationMode.CONNECTIVITY
    output_mode: SegmentationMode = SegmentationMode.CONNECTIVITY

    def get_mode(self, context: str = "input") -> SegmentationMode:
        """Get the segmentation mode for a given context.

        Args:
            context: "input" or "output"

        Returns:
            The appropriate SegmentationMode for the context
        """
        if context == "output":
            return self.output_mode
        return self.input_mode

    @classmethod
    def uniform(cls, mode: SegmentationMode) -> 'SegmentationStrategy':
        """Create a strategy that uses the same mode for input and output.

        Args:
            mode: The segmentation mode to use for both grids

        Returns:
            A SegmentationStrategy with uniform mode
        """
        return cls(input_mode=mode, output_mode=mode)

    @classmethod
    def from_mode(cls, mode: Optional[SegmentationMode]) -> 'SegmentationStrategy':
        """Convert a legacy SegmentationMode to a SegmentationStrategy.

        This provides backward compatibility with code that passes a single
        SegmentationMode instead of a SegmentationStrategy.

        Args:
            mode: A SegmentationMode, or None for default (CONNECTIVITY)

        Returns:
            A SegmentationStrategy with uniform mode
        """
        if mode is None:
            return cls()
        return cls.uniform(mode)

    def __repr__(self) -> str:
        if self.input_mode == self.output_mode:
            return f"SegmentationStrategy(uniform={self.input_mode.value})"
        return f"SegmentationStrategy(input={self.input_mode.value}, output={self.output_mode.value})"


# Connectivity structures
STRUCTURE_8CONN = np.array([[1, 1, 1],
                            [1, 1, 1],
                            [1, 1, 1]], dtype=np.int32)

STRUCTURE_4CONN = np.array([[0, 1, 0],
                            [1, 1, 1],
                            [0, 1, 0]], dtype=np.int32)


# =============================================================================
# Object Data Structure
# =============================================================================

@dataclass
class Object:
    """Represents an extracted object from an ARC grid.

    Attributes:
        id: Unique identifier for this object within the grid
        row: Top-left row coordinate
        col: Top-left column coordinate
        height: Height of bounding box
        width: Width of bounding box
        color: Color value (0-9)
        pixels: Set of (row, col) tuples for all pixels in this object
        is_background: Whether this object is part of the background
        is_divider: Whether this object is a divider line
        parent: Reference to containing parent object (None if top-level)
        children: List of objects contained within this object
    """
    id: int
    row: int          # top-left row
    col: int          # top-left col
    height: int
    width: int
    color: int
    pixels: Set[Tuple[int, int]] = field(default_factory=set)
    is_background: bool = False
    is_divider: bool = False
    # Hierarchy fields (backward-compatible defaults)
    parent: Optional['Object'] = field(default=None, repr=False)
    children: List['Object'] = field(default_factory=list, repr=False)

    @property
    def center(self) -> Tuple[float, float]:
        """Get centroid (row, col) of the object."""
        return (self.row + self.height / 2, self.col + self.width / 2)

    @property
    def area(self) -> int:
        """Get area (number of pixels) of the object."""
        return len(self.pixels) if self.pixels else self.height * self.width

    @property
    def bottom_right(self) -> Tuple[int, int]:
        """Get bottom-right corner coordinates."""
        return (self.row + self.height - 1, self.col + self.width - 1)

    @property
    def bbox(self) -> Tuple[int, int, int, int]:
        """Get bounding box as (min_row, min_col, max_row, max_col)."""
        return (self.row, self.col, self.row + self.height - 1, self.col + self.width - 1)

    # -------------------------------------------------------------------------
    # Hierarchy Properties
    # -------------------------------------------------------------------------

    @property
    def is_atomic(self) -> bool:
        """True if this object is standalone (no parent AND no children).

        Atomic objects are independent entities that are neither containers
        nor contained within other objects.
        """
        return self.parent is None and len(self.children) == 0

    @property
    def is_parent(self) -> bool:
        """True if this object contains other objects (has children)."""
        return len(self.children) > 0

    @property
    def is_composite(self) -> bool:
        """Alias for is_parent. True if this object contains other objects."""
        return len(self.children) > 0

    @property
    def is_child(self) -> bool:
        """True if this object is contained within another object (has a parent)."""
        return self.parent is not None

    @property
    def is_root(self) -> bool:
        """True if this object has no parent (top-level).

        Note: A root object can be either atomic (no children) or a parent (has children).
        """
        return self.parent is None

    @property
    def depth(self) -> int:
        """Nesting depth (0 for root objects)."""
        return 0 if self.parent is None else 1 + self.parent.depth

    # -------------------------------------------------------------------------
    # Hierarchy Methods
    # -------------------------------------------------------------------------

    def flatten(self) -> List['Object']:
        """Returns this object plus all descendants in pre-order traversal.

        Useful for getting all objects in a hierarchy as a flat list.

        Returns:
            List starting with self, then all children recursively.
        """
        result = [self]
        for child in self.children:
            result.extend(child.flatten())
        return result

    def get_atomic_objects(self) -> List['Object']:
        """Returns only the leaf (childless) objects in the hierarchy.

        Returns:
            List of all atomic objects under this object (may include self).
        """
        if self.is_atomic:
            return [self]
        result = []
        for child in self.children:
            result.extend(child.get_atomic_objects())
        return result

    def local_bbox(self) -> Tuple[int, int, int, int]:
        """Get bounding box relative to parent's top-left corner.

        Returns:
            (local_row, local_col, height, width) relative to parent.
            For root objects, returns (row, col, height, width).
        """
        if self.parent is None:
            return (self.row, self.col, self.height, self.width)
        return (
            self.row - self.parent.row,
            self.col - self.parent.col,
            self.height,
            self.width
        )

    def local_center(self) -> Tuple[float, float]:
        """Get center relative to parent's top-left corner.

        Returns:
            (local_row, local_col) center position relative to parent.
        """
        if self.parent is None:
            return self.center
        return (
            self.center[0] - self.parent.row,
            self.center[1] - self.parent.col
        )

    # -------------------------------------------------------------------------
    # Utility Methods for Genesis
    # -------------------------------------------------------------------------

    def contains_point(self, row: int, col: int) -> bool:
        """Check if a point is within this object's bounding box.

        Args:
            row: Row coordinate to check
            col: Column coordinate to check

        Returns:
            True if point is within bounding box
        """
        return (self.row <= row < self.row + self.height and
                self.col <= col < self.col + self.width)

    def get_local_mask(self) -> np.ndarray:
        """Get this object's shape as a local mask (height x width).

        The mask is a boolean array where True indicates pixels
        that belong to this object.

        Returns:
            Boolean numpy array of shape (height, width)
        """
        mask = np.zeros((self.height, self.width), dtype=bool)
        if self.pixels:
            for r, c in self.pixels:
                local_r = r - self.row
                local_c = c - self.col
                if 0 <= local_r < self.height and 0 <= local_c < self.width:
                    mask[local_r, local_c] = True
        else:
            # If no pixels set, assume filled rectangle
            mask[:] = True
        return mask

    def overlaps(self, other: 'Object') -> bool:
        """Check if this object's bounding box overlaps with another's.

        Args:
            other: Another Object to check overlap with

        Returns:
            True if bounding boxes overlap
        """
        # Check for no overlap conditions
        if self.row + self.height <= other.row:  # self is above other
            return False
        if other.row + other.height <= self.row:  # other is above self
            return False
        if self.col + self.width <= other.col:   # self is left of other
            return False
        if other.col + other.width <= self.col:   # other is left of self
            return False
        return True

    def contains_object(self, other: 'Object') -> bool:
        """Check if this object's bounding box fully contains another.

        Args:
            other: Another Object to check containment for

        Returns:
            True if other is fully inside this object's bbox
        """
        return (self.row <= other.row and
                self.col <= other.col and
                self.row + self.height >= other.row + other.height and
                self.col + self.width >= other.col + other.width)


# =============================================================================
# Background Detection (from AffinitySlotAttention)
# =============================================================================

def detect_background_color(
    grid: np.ndarray,
    min_bg_coverage: float = 0.5,
    prefer_black: bool = True,
    black_min_coverage: float = 0.4
) -> Optional[int]:
    """
    Detect the background color using the largest edge-connected component heuristic.

    This matches the approach in AffinitySlotAttention._detect_background_color,
    with an added preference for black (0) since it's the conventional ARC background.

    The background is identified as the color with the largest connected component
    that touches any edge of the grid, provided it covers at least min_bg_coverage
    of the total grid area.

    Args:
        grid: (H, W) integer color values 0-9
        min_bg_coverage: Minimum fraction of grid that must be covered to be
                        considered background (default 0.5 = 50%)
        prefer_black: If True, prefer black (0) as background when it touches
                     edges and covers at least black_min_coverage (default True)
        black_min_coverage: Minimum coverage for black to be preferred (default 0.4 = 40%)

    Returns:
        The background color (0-9), or None if no clear background is detected
    """
    H, W = grid.shape
    total_cells = H * W

    # If the grid has only one color, there's no background - you need
    # foreground objects for something to be considered background
    unique_colors = np.unique(grid)
    if len(unique_colors) <= 1:
        return None

    # Get colors that touch any edge
    edge_colors = set()
    edge_colors.update(grid[0, :].tolist())      # top
    edge_colors.update(grid[H-1, :].tolist())    # bottom
    edge_colors.update(grid[:, 0].tolist())      # left
    edge_colors.update(grid[:, W-1].tolist())    # right

    # Prefer black (0) if it touches edges and covers enough of the grid
    # This handles ARC's convention where black is typically background
    if prefer_black and 0 in edge_colors:
        black_mask = (grid == 0)
        black_coverage = black_mask.sum() / total_cells
        if black_coverage >= black_min_coverage:
            return 0

    best_color = None
    best_size = 0

    for color in edge_colors:
        mask = (grid == color)
        labeled, num_features = ndimage.label(mask)

        # Find components that touch the edge
        for comp_id in range(1, num_features + 1):
            comp_mask = (labeled == comp_id)

            touches_edge = (
                comp_mask[0, :].any() or comp_mask[H-1, :].any() or
                comp_mask[:, 0].any() or comp_mask[:, W-1].any()
            )

            if touches_edge:
                size = comp_mask.sum()
                if size > best_size:
                    best_size = size
                    best_color = color

    # Only return as background if it covers enough of the grid
    if best_color is not None and best_size / total_cells >= min_bg_coverage:
        return best_color

    return None


def detect_background_color_for_pair(
    input_grid: np.ndarray,
    output_grid: np.ndarray,
    min_coverage: float = 0.5
) -> Optional[int]:
    """
    Detect background color consistently for an input/output pair.

    A color is considered background only if it covers more than min_coverage
    of EITHER the input or output grid. This ensures:
    1. Consistency: same background used for both grids in a pair
    2. Conservative: only truly dominant colors are treated as background

    Args:
        input_grid: (H, W) input grid with integer color values 0-9
        output_grid: (H, W) output grid with integer color values 0-9
        min_coverage: Minimum fraction (default 0.5 = 50%) a color must cover
                     in either grid to be considered background

    Returns:
        The background color (0-9), or None if no color is dominant enough
    """
    input_total = input_grid.size
    output_total = output_grid.size

    # Check each color's coverage in both grids
    for color in range(NUM_COLORS):
        input_coverage = (input_grid == color).sum() / input_total
        output_coverage = (output_grid == color).sum() / output_total

        # If this color dominates either grid, it's the background
        if input_coverage > min_coverage or output_coverage > min_coverage:
            return color

    return None


def compute_exterior_mask(grid: np.ndarray, background_color: int) -> np.ndarray:
    """
    Compute which pixels of the background color are "exterior" (reachable from edges)
    using 4-connectivity. This allows diagonal lines to act as enclosing barriers.

    This matches the approach in AffinitySlotAttention._compute_exterior_mask.

    Args:
        grid: (H, W) integer color values 0-9
        background_color: The detected background color

    Returns:
        Boolean mask (H, W) where True = exterior background pixel
    """
    H, W = grid.shape

    # Create mask of background-colored pixels
    bg_mask = (grid == background_color)

    # Start by marking edge pixels of background color
    seed = np.zeros((H, W), dtype=bool)
    seed[0, :] = bg_mask[0, :]       # top edge
    seed[H-1, :] = bg_mask[H-1, :]   # bottom edge
    seed[:, 0] = bg_mask[:, 0]       # left edge
    seed[:, W-1] = bg_mask[:, W-1]   # right edge

    # Flood fill from edge seeds within background mask using 4-connectivity
    # This finds all background pixels reachable from edges without crossing diagonals
    exterior_mask = ndimage.binary_dilation(seed, structure=STRUCTURE_4CONN,
                                             iterations=-1, mask=bg_mask)

    return exterior_mask


# =============================================================================
# Divider Line Detection
# =============================================================================

def detect_divider_lines(
    grid: np.ndarray,
    max_divider_ratio: float = 0.5,
    verbose: bool = False
) -> Tuple[List[Tuple[int, int, int]], List[Tuple[int, int, int]]]:
    """
    Detect horizontal and vertical divider lines in a grid with semantic validation.

    A divider line is a contiguous band of a single color spanning the
    full width (horizontal) or full height (vertical) of the grid.
    The divider must have non-empty regions on both sides.

    Additional semantic checks (improvements over original):
    - Not a stripe pattern (not ALL rows or ALL cols uniform)
    - Divider area should be reasonable (not overwhelming content)
    - Dividers should not split connected objects
    - Separated regions should be meaningfully different

    Args:
        grid: (H, W) integer color values 0-9
        max_divider_ratio: Maximum fraction of grid that single-axis dividers can occupy (default 0.5)
        verbose: Print debug info

    Returns:
        horizontal_dividers: List of (start_row, end_row_exclusive, color) tuples
        vertical_dividers: List of (start_col, end_col_exclusive, color) tuples
    """
    H, W = grid.shape
    total_pixels = H * W
    
    if verbose:
        print(f"\n--- Analyzing {H}x{W} grid ---")
    
    # ==========================================================================
    # Step 1: Find candidate dividers (same as original)
    # ==========================================================================
    
    h_candidates = []
    row = 0
    while row < H:
        row_colors = grid[row, :]
        if np.all(row_colors == row_colors[0]):
            color = int(row_colors[0])
            start_row = row
            while row < H and np.all(grid[row, :] == color):
                row += 1
            end_row = row
            if start_row > 0 and end_row < H:
                h_candidates.append((start_row, end_row, color))
        else:
            row += 1
    
    v_candidates = []
    col = 0
    while col < W:
        col_colors = grid[:, col]
        if np.all(col_colors == col_colors[0]):
            color = int(col_colors[0])
            start_col = col
            while col < W and np.all(grid[:, col] == color):
                col += 1
            end_col = col
            if start_col > 0 and end_col < W:
                v_candidates.append((start_col, end_col, color))
        else:
            col += 1
    
    if verbose:
        print(f"Candidates: {len(h_candidates)} H, {len(v_candidates)} V")
    
    if not h_candidates and not v_candidates:
        return [], []
    
    # ==========================================================================
    # Step 1b: Filter dividers - each side must have meaningful variation
    # ==========================================================================
    # A region has "meaningful variation" if it's not dominated by one color.
    # This handles cases where a mostly-uniform region (95% black) has one
    # different pixel - that's not real variation, just noise.
    
    def region_has_meaningful_variation(region: np.ndarray, 
                                         min_colors: int = 2,
                                         max_dominant_ratio: float = 0.90) -> bool:
        """
        Check if a region has meaningful variation (not dominated by one color).
        
        Args:
            region: The region to check
            min_colors: Minimum number of distinct colors required
            max_dominant_ratio: If one color is more than this ratio, not varied
        """
        if region.size == 0:
            return False
        
        unique, counts = np.unique(region, return_counts=True)
        
        # Must have at least min_colors distinct colors
        if len(unique) < min_colors:
            return False
        
        # The dominant color shouldn't be more than max_dominant_ratio of the region
        dominant_ratio = counts.max() / region.size
        if dominant_ratio > max_dominant_ratio:
            return False
        
        return True
    
    def filter_h_dividers(dividers: list, grid: np.ndarray) -> list:
        """Keep H dividers that separate meaningfully varied regions."""
        H, W = grid.shape
        valid = []
        
        for start, end, color in dividers:
            above = grid[:start, :]
            below = grid[end:, :]
            
            above_varied = region_has_meaningful_variation(above)
            below_varied = region_has_meaningful_variation(below)
            
            # A valid divider has varied content on at least one side,
            # AND the other side is either varied OR a different uniform color
            if above_varied and below_varied:
                valid.append((start, end, color))
            elif above_varied and not below_varied:
                # Below is uniform - check it's different from the divider
                below_colors = np.unique(below)
                if len(below_colors) > 0 and below_colors[0] != color:
                    valid.append((start, end, color))
                elif verbose:
                    print(f"  Rejecting H divider rows {start}-{end-1}: below is uniform same color")
            elif below_varied and not above_varied:
                above_colors = np.unique(above)
                if len(above_colors) > 0 and above_colors[0] != color:
                    valid.append((start, end, color))
                elif verbose:
                    print(f"  Rejecting H divider rows {start}-{end-1}: above is uniform same color")
            else:
                # Neither side is varied - but could still be valid if both are
                # uniform with DIFFERENT colors (separating two filled regions)
                above_colors = np.unique(above)
                below_colors = np.unique(below)
                if (len(above_colors) == 1 and len(below_colors) == 1 and
                    above_colors[0] != below_colors[0] and
                    above_colors[0] != color and below_colors[0] != color):
                    valid.append((start, end, color))
                elif verbose:
                    print(f"  Rejecting H divider rows {start}-{end-1}: neither side varied")

        return valid

    def filter_v_dividers(dividers: list, grid: np.ndarray) -> list:
        """Keep V dividers that separate meaningfully varied regions."""
        H, W = grid.shape
        valid = []
        
        for start, end, color in dividers:
            left = grid[:, :start]
            right = grid[:, end:]
            
            left_varied = region_has_meaningful_variation(left)
            right_varied = region_has_meaningful_variation(right)
            
            if left_varied and right_varied:
                valid.append((start, end, color))
            elif left_varied and not right_varied:
                right_colors = np.unique(right)
                if len(right_colors) > 0 and right_colors[0] != color:
                    valid.append((start, end, color))
                elif verbose:
                    print(f"  Rejecting V divider cols {start}-{end-1}: right is uniform same color")
            elif right_varied and not left_varied:
                left_colors = np.unique(left)
                if len(left_colors) > 0 and left_colors[0] != color:
                    valid.append((start, end, color))
                elif verbose:
                    print(f"  Rejecting V divider cols {start}-{end-1}: left is uniform same color")
            else:
                # Neither side is varied - but could still be valid if both are
                # uniform with DIFFERENT colors (separating two filled regions)
                left_colors = np.unique(left)
                right_colors = np.unique(right)
                if (len(left_colors) == 1 and len(right_colors) == 1 and
                    left_colors[0] != right_colors[0] and
                    left_colors[0] != color and right_colors[0] != color):
                    valid.append((start, end, color))
                elif verbose:
                    print(f"  Rejecting V divider cols {start}-{end-1}: neither side varied")

        return valid
    
    h_candidates = filter_h_dividers(h_candidates, grid)
    v_candidates = filter_v_dividers(v_candidates, grid)
    
    if verbose:
        print(f"After content filter: {len(h_candidates)} H, {len(v_candidates)} V")
    
    # ==========================================================================
    # Step 1c: Reject dividers that would create tiny cells
    # ==========================================================================
    # If accepting multiple dividers would create a cell smaller than min_cell_size,
    # keep only the first one (greedy approach).
    
    min_cell_size = 2  # Minimum rows/cols for a cell to be meaningful
    
    def filter_by_cell_size(dividers: list, total_size: int) -> list:
        """Remove dividers that would create cells smaller than min_cell_size."""
        if len(dividers) <= 1:
            return dividers
        
        sorted_divs = sorted(dividers, key=lambda x: x[0])
        
        # Build cell boundaries: [0, div1_start, div1_end, div2_start, ...]
        kept = []
        prev_end = 0
        
        for start, end, color in sorted_divs:
            # Cell before this divider
            cell_size = start - prev_end
            
            if cell_size >= min_cell_size:
                kept.append((start, end, color))
                prev_end = end
            elif verbose:
                print(f"  Rejecting divider {start}-{end-1}: would create cell of size {cell_size}")
        
        # Check final cell (after last kept divider)
        if kept:
            last_end = kept[-1][1]
            final_cell_size = total_size - last_end
            if final_cell_size < min_cell_size:
                if verbose:
                    print(f"  Removing last divider: final cell size {final_cell_size}")
                kept = kept[:-1]
        
        return kept
    
    h_candidates = filter_by_cell_size(h_candidates, H)
    v_candidates = filter_by_cell_size(v_candidates, W)
    
    if verbose:
        print(f"After cell size filter: {len(h_candidates)} H, {len(v_candidates)} V")
    
    if not h_candidates and not v_candidates:
        return [], []
    
    # ==========================================================================
    # Step 2: Analyze row/column uniformity patterns
    # ==========================================================================
    
    uniform_row_colors = {}
    for r in range(H):
        if np.all(grid[r,:] == grid[r,0]):
            uniform_row_colors[r] = int(grid[r,0])
    
    uniform_col_colors = {}
    for c in range(W):
        if np.all(grid[:,c] == grid[0,c]):
            uniform_col_colors[c] = int(grid[0,c])
    
    h_uniform_colors = set(uniform_row_colors.values())
    v_uniform_colors = set(uniform_col_colors.values())
    
    if verbose:
        print(f"Uniform rows: {len(uniform_row_colors)}/{H}, colors: {h_uniform_colors}")
        print(f"Uniform cols: {len(uniform_col_colors)}/{W}, colors: {v_uniform_colors}")
    
    # ==========================================================================
    # Step 3: Stripe pattern detection
    # ==========================================================================
    
    def is_stripe_pattern(uniform_dict: dict, total: int, candidates: list) -> bool:
        """Detect if this is a stripe pattern rather than a grid."""
        if len(uniform_dict) < total * 0.6:
            return False
        
        div_colors = set(c for _, _, c in candidates)
        non_div_uniform = set()
        for idx, color in uniform_dict.items():
            is_divider = any(start <= idx < end for start, end, c in candidates if c == color)
            if not is_divider:
                non_div_uniform.add(color)
        
        # Valid grid with uniform cells:
        # - one non-div color, one div color, different -> grid not stripes
        # - two non-div colors (filled regions), one div color -> grid not stripes
        if len(div_colors) == 1:
            if len(non_div_uniform) == 1 and non_div_uniform != div_colors:
                return False
            if len(non_div_uniform) == 2:
                # Two uniform regions separated by a divider - this is a valid grid
                return False

        # If 80%+ rows/cols are uniform, it's a stripe pattern
        if len(uniform_dict) >= total * 0.8:
            return True
        
        return False
    
    is_h_stripes = is_stripe_pattern(uniform_row_colors, H, h_candidates)
    is_v_stripes = is_stripe_pattern(uniform_col_colors, W, v_candidates)
    
    if is_h_stripes:
        if verbose:
            print("H stripe pattern detected - rejecting H dividers")
        h_candidates = []
        
    if is_v_stripes:
        if verbose:
            print("V stripe pattern detected - rejecting V dividers")
        v_candidates = []
    
    if not h_candidates and not v_candidates:
        return [], []
    
    # ==========================================================================
    # Step 4: Connected object detection
    # ==========================================================================
    
    def check_connected_split(grid, h_divs, v_divs) -> bool:
        """Check if dividers would split a single connected component."""
        div_colors = set()
        for _, _, c in h_divs + v_divs:
            div_colors.add(c)
        
        for div_color in div_colors:
            mask = (grid == div_color)
            labeled, num_components = ndimage.label(mask)
            
            if num_components == 1:
                rows, cols = np.where(mask)
                
                h_div_rows = set()
                for start, end, c in h_divs:
                    if c == div_color:
                        for r in range(start, end):
                            h_div_rows.add(r)
                
                v_div_cols = set()
                for start, end, c in v_divs:
                    if c == div_color:
                        for c_idx in range(start, end):
                            v_div_cols.add(c_idx)
                
                non_div_pixels = sum(1 for r, c in zip(rows, cols) 
                                    if r not in h_div_rows and c not in v_div_cols)
                
                if non_div_pixels > 0:
                    for h_start, h_end, c in h_divs:
                        if c == div_color:
                            above = any(r < h_start for r in rows)
                            below = any(r >= h_end for r in rows)
                            if above and below:
                                return True
                    
                    for v_start, v_end, c in v_divs:
                        if c == div_color:
                            left = any(col < v_start for col in cols)
                            right = any(col >= v_end for col in cols)
                            if left and right:
                                return True
        
        return False
    
    if check_connected_split(grid, h_candidates, v_candidates):
        if verbose:
            print("Connected object split detected - rejecting dividers")
        return [], []
    
    # ==========================================================================
    # Step 5: Ratio check (for single-axis dividers only)
    # ==========================================================================
    
    h_pixels = sum((end - start) * W for start, end, _ in h_candidates)
    v_pixels = sum((end - start) * H for start, end, _ in v_candidates)
    
    intersect = 0
    for h_start, h_end, _ in h_candidates:
        for v_start, v_end, _ in v_candidates:
            intersect += (h_end - h_start) * (v_end - v_start)
    
    total_div_pixels = h_pixels + v_pixels - intersect
    div_ratio = total_div_pixels / total_pixels if total_pixels > 0 else 0
    
    if verbose:
        print(f"Divider ratio: {div_ratio:.1%}")
    
    is_grid_pattern = len(h_candidates) > 0 and len(v_candidates) > 0
    
    if not is_grid_pattern and div_ratio > max_divider_ratio:
        if verbose:
            print(f"Single-axis dividers exceed ratio ({div_ratio:.1%} > {max_divider_ratio:.0%})")
        return [], []
    
    if is_grid_pattern and div_ratio > 0.75:
        if verbose:
            print(f"Grid divider ratio too high ({div_ratio:.1%} > 75%)")
        return [], []
    
    # ==========================================================================
    # Step 6: Content differentiation check
    # ==========================================================================
    
    def get_region_signature(region: np.ndarray) -> tuple:
        hist = np.zeros(10, dtype=int)
        for c in range(10):
            hist[c] = np.sum(region == c)
        return tuple(hist)
    
    row_bounds = [0]
    for start, end, _ in sorted(h_candidates):
        row_bounds.extend([start, end])
    row_bounds.append(H)
    row_bounds = sorted(set(row_bounds))
    
    col_bounds = [0]
    for start, end, _ in sorted(v_candidates):
        col_bounds.extend([start, end])
    col_bounds.append(W)
    col_bounds = sorted(set(col_bounds))
    
    content_signatures = []
    for i in range(len(row_bounds) - 1):
        r_start, r_end = row_bounds[i], row_bounds[i+1]
        is_h_div = any(start == r_start and end == r_end for start, end, _ in h_candidates)
        
        for j in range(len(col_bounds) - 1):
            c_start, c_end = col_bounds[j], col_bounds[j+1]
            is_v_div = any(start == c_start and end == c_end for start, end, _ in v_candidates)
            
            if not is_h_div and not is_v_div:
                region = grid[r_start:r_end, c_start:c_end]
                content_signatures.append(get_region_signature(region))
    
    if len(content_signatures) > 1:
        unique_sigs = set(content_signatures)
        if len(unique_sigs) == 1 and len(content_signatures) > 2:
            if verbose:
                print(f"All {len(content_signatures)} content regions identical - suspicious")
            return [], []
    
    if verbose:
        print(f"Content regions: {len(content_signatures)}, unique: {len(set(content_signatures))}")
        print(f"Final: {len(h_candidates)} H, {len(v_candidates)} V")
    
    return h_candidates, v_candidates


def segment_by_dividers(
    grid: np.ndarray,
    horizontal_dividers: List[Tuple[int, int, int]],
    vertical_dividers: List[Tuple[int, int, int]],
    skip_background: bool = True,
    background_color: Optional[int] = None
) -> Tuple[np.ndarray, List[int], List[Tuple[int, int, int, int]], List[bool], List[bool]]:
    """
    Segment a grid based on detected divider lines.

    Creates objects for each region separated by dividers, plus the dividers themselves.
    Regions are labeled in reading order (top-to-bottom, left-to-right).

    Args:
        grid: (H, W) integer color values 0-9
        horizontal_dividers: List of (start_row, end_row_exclusive, color)
        vertical_dividers: List of (start_col, end_col_exclusive, color)
        skip_background: If True, skip regions whose dominant color matches background_color
        background_color: The background color to skip (if skip_background is True)

    Returns:
        labels: (H, W) component IDs (0 = none, 1+ = component IDs)
        colors: List of dominant colors for each component
        bboxes: List of (min_row, min_col, max_row, max_col) for each component
        is_background: List of bool indicating if each component is background-colored
        is_divider: List of bool indicating which components are dividers
    """
    H, W = grid.shape
    labels = np.zeros((H, W), dtype=np.int32)
    colors = []
    bboxes = []
    is_background = []
    is_divider = []

    # Build row boundaries from horizontal dividers
    row_boundaries = [0]
    h_divider_rows = set()
    for start_row, end_row, _ in sorted(horizontal_dividers):
        row_boundaries.append(start_row)
        row_boundaries.append(end_row)
        h_divider_rows.add((start_row, end_row))
    row_boundaries.append(H)

    # Build col boundaries from vertical dividers
    col_boundaries = [0]
    v_divider_cols = set()
    for start_col, end_col, _ in sorted(vertical_dividers):
        col_boundaries.append(start_col)
        col_boundaries.append(end_col)
        v_divider_cols.add((start_col, end_col))
    col_boundaries.append(W)

    # Remove duplicates and sort
    row_boundaries = sorted(set(row_boundaries))
    col_boundaries = sorted(set(col_boundaries))

    component_id = 0

    # Create regions for each cell in the grid formed by boundaries
    for i in range(len(row_boundaries) - 1):
        row_start = row_boundaries[i]
        row_end = row_boundaries[i + 1]

        for j in range(len(col_boundaries) - 1):
            col_start = col_boundaries[j]
            col_end = col_boundaries[j + 1]

            # Skip empty regions
            if row_start >= row_end or col_start >= col_end:
                continue

            # Check if this region is a divider
            is_h_divider = (row_start, row_end) in h_divider_rows
            is_v_divider = (col_start, col_end) in v_divider_cols
            region_is_divider = is_h_divider or is_v_divider

            # Find colors in this region
            region = grid[row_start:row_end, col_start:col_end]
            unique, counts = np.unique(region, return_counts=True)

            # Check if region has any non-background content
            if skip_background and background_color is not None:
                # Filter to non-background colors
                non_bg_mask = unique != background_color
                if not np.any(non_bg_mask):
                    # Region contains only background color - skip it
                    continue
                # Use most common non-background color as the region's color
                non_bg_unique = unique[non_bg_mask]
                non_bg_counts = counts[non_bg_mask]
                dominant_color = int(non_bg_unique[np.argmax(non_bg_counts)])
                region_is_background = False
            else:
                # No background filtering - use overall dominant color
                dominant_color = int(unique[np.argmax(counts)])
                region_is_background = (background_color is not None and
                                        dominant_color == background_color)

            component_id += 1

            # Label this region
            labels[row_start:row_end, col_start:col_end] = component_id

            colors.append(dominant_color)
            bboxes.append((row_start, col_start, row_end - 1, col_end - 1))
            is_background.append(region_is_background)
            is_divider.append(region_is_divider)

    return labels, colors, bboxes, is_background, is_divider


def extract_children_in_region(
    sub_grid: np.ndarray,
    parent_row_offset: int,
    parent_col_offset: int,
    start_id: int = 0,
    background_color: Optional[int] = None,
    segmentation_mode: Optional[SegmentationMode] = None
) -> List[Object]:
    """
    Extract child objects within a region (sub-grid).

    This function is used after divider-based segmentation to find the objects
    within each divided region.

    Args:
        sub_grid: The region's grid data (H, W) integer color values 0-9
        parent_row_offset: Row offset to add to child coordinates (parent's top-left row)
        parent_col_offset: Col offset to add to child coordinates (parent's top-left col)
        start_id: Starting ID for child objects
        background_color: Background color to skip (from global grid detection).
                         If None, detects background within the sub-grid.
        segmentation_mode: How to segment children within regions:
                          - CONNECTIVITY (default): Connected component analysis
                          - PIXEL: Each pixel is its own object (useful for mode counting)
                          - COLOR: All pixels of same color form one object

    Returns:
        List of Object instances with coordinates in global (full grid) space.
        Returns empty list if region has only one color (parent is atomic).
    """
    # Default to CONNECTIVITY mode
    if segmentation_mode is None:
        segmentation_mode = SegmentationMode.CONNECTIVITY

    # Check if region has only one color - if so, it's atomic (no children)
    unique_colors = np.unique(sub_grid)
    if len(unique_colors) <= 1:
        return []

    H, W = sub_grid.shape

    # Use provided background color, or detect within this region
    if background_color is None:
        background_color = detect_background_color(sub_grid, min_bg_coverage=0.4)

    labels = np.zeros((H, W), dtype=np.int32)
    colors = []
    bboxes = []
    is_background_list = []
    component_id = 0

    if segmentation_mode == SegmentationMode.PIXEL:
        # PIXEL mode: each non-background pixel is its own object
        for r in range(H):
            for c in range(W):
                color = int(sub_grid[r, c])
                if color == background_color:
                    continue
                component_id += 1
                labels[r, c] = component_id
                colors.append(color)
                bboxes.append((r, c, r, c))
                is_background_list.append(False)

    elif segmentation_mode == SegmentationMode.COLOR:
        # COLOR mode: all pixels of same color form one object
        for color in range(NUM_COLORS):
            if color == background_color:
                continue
            mask = (sub_grid == color)
            if not mask.any():
                continue
            component_id += 1
            labels[mask] = component_id
            colors.append(color)
            is_background_list.append(False)
            rows, cols = np.where(mask)
            bbox = (rows.min(), cols.min(), rows.max(), cols.max())
            bboxes.append(bbox)

    else:
        # CONNECTIVITY mode: connected component analysis
        for c in range(NUM_COLORS):
            # Skip background color
            if c == background_color:
                continue

            mask = (sub_grid == c)
            if not mask.any():
                continue

            # Use 8-connectivity for foreground colors
            structure = STRUCTURE_8CONN
            labeled, num_features = ndimage.label(mask, structure=structure)

            for comp in range(1, num_features + 1):
                comp_mask = (labeled == comp)
                component_id += 1
                labels[comp_mask] = component_id
                colors.append(c)
                is_background_list.append(False)

                rows, cols = np.where(comp_mask)
                bbox = (rows.min(), cols.min(), rows.max(), cols.max())
                bboxes.append(bbox)

    if component_id == 0:
        return []

    # Convert to Object instances with global coordinates
    children = []
    for i, (color, bbox, is_bg) in enumerate(zip(colors, bboxes, is_background_list)):
        local_min_row, local_min_col, local_max_row, local_max_col = bbox
        height = local_max_row - local_min_row + 1
        width = local_max_col - local_min_col + 1

        # Extract pixels and convert to global coordinates
        mask = (labels == i + 1)
        local_pixels = list(zip(*np.where(mask)))
        global_pixels = set(
            (r + parent_row_offset, c + parent_col_offset)
            for r, c in local_pixels
        )

        # Global coordinates
        global_row = local_min_row + parent_row_offset
        global_col = local_min_col + parent_col_offset

        children.append(Object(
            id=start_id + i,
            row=global_row,
            col=global_col,
            height=height,
            width=width,
            color=color,
            pixels=global_pixels,
            is_background=is_bg,
            is_divider=False
        ))

    return children


def segment_by_dividers_hierarchical(
    grid: np.ndarray,
    horizontal_dividers: List[Tuple[int, int, int]],
    vertical_dividers: List[Tuple[int, int, int]],
    children_segmentation_mode: Optional[SegmentationMode] = None
) -> List[Object]:
    """
    Segment a grid by dividers and build hierarchical parent/child structure.

    Each region separated by dividers becomes a parent object. Within each parent,
    child objects are extracted. If a region has only one color, the parent has
    no children (it's atomic).

    Args:
        grid: (H, W) integer color values 0-9
        horizontal_dividers: List of (start_row, end_row_exclusive, color)
        vertical_dividers: List of (start_col, end_col_exclusive, color)
        children_segmentation_mode: How to segment children within regions:
                                   - CONNECTIVITY (default): Connected component analysis
                                   - PIXEL: Each pixel is its own object
                                   - COLOR: All pixels of same color form one object

    Returns:
        List of parent Object instances with children populated.
        Divider objects are included with is_divider=True.
    """
    H, W = grid.shape

    # Detect background at the global level (used for all sub-regions)
    global_background = detect_background_color(grid, min_bg_coverage=0.4)

    # Build row boundaries from horizontal dividers (store color too)
    row_boundaries = [0]
    h_divider_info = {}  # (start, end) -> color
    for start_row, end_row, color in sorted(horizontal_dividers):
        row_boundaries.append(start_row)
        row_boundaries.append(end_row)
        h_divider_info[(start_row, end_row)] = color
    row_boundaries.append(H)

    # Build col boundaries from vertical dividers (store color too)
    col_boundaries = [0]
    v_divider_info = {}  # (start, end) -> color
    for start_col, end_col, color in sorted(vertical_dividers):
        col_boundaries.append(start_col)
        col_boundaries.append(end_col)
        v_divider_info[(start_col, end_col)] = color
    col_boundaries.append(W)

    # Remove duplicates and sort
    row_boundaries = sorted(set(row_boundaries))
    col_boundaries = sorted(set(col_boundaries))

    parents = []
    parent_id = 0
    child_id_counter = 0

    # Create parent objects for each cell in the grid formed by boundaries
    for i in range(len(row_boundaries) - 1):
        row_start = row_boundaries[i]
        row_end = row_boundaries[i + 1]

        for j in range(len(col_boundaries) - 1):
            col_start = col_boundaries[j]
            col_end = col_boundaries[j + 1]

            # Skip empty regions
            if row_start >= row_end or col_start >= col_end:
                continue

            # Check if this region is a divider
            is_h_divider = (row_start, row_end) in h_divider_info
            is_v_divider = (col_start, col_end) in v_divider_info

            # Create divider object if this is a divider region
            if is_h_divider or is_v_divider:
                # Get divider color
                if is_h_divider:
                    divider_color = h_divider_info[(row_start, row_end)]
                else:
                    divider_color = v_divider_info[(col_start, col_end)]

                # Compute all pixels in this divider region
                divider_pixels = set()
                for r in range(row_start, row_end):
                    for c in range(col_start, col_end):
                        divider_pixels.add((r, c))

                # Create divider object
                divider_obj = Object(
                    id=parent_id,
                    row=row_start,
                    col=col_start,
                    height=row_end - row_start,
                    width=col_end - col_start,
                    color=divider_color,
                    pixels=divider_pixels,
                    is_background=False,
                    is_divider=True
                )
                parents.append(divider_obj)
                parent_id += 1
                continue

            # Extract the sub-grid for this region
            sub_grid = grid[row_start:row_end, col_start:col_end]

            # Find dominant color for the parent (excluding background if other colors exist)
            unique, counts = np.unique(sub_grid, return_counts=True)
            # Prefer non-background colors to avoid assigning color=0 to pattern regions
            non_bg_mask = unique != global_background
            if np.any(non_bg_mask):
                # Use most frequent non-background color
                non_bg_unique = unique[non_bg_mask]
                non_bg_counts = counts[non_bg_mask]
                dominant_color = int(non_bg_unique[np.argmax(non_bg_counts)])
            else:
                # All background - use background color
                dominant_color = int(unique[np.argmax(counts)])

            # Compute all pixels in this region (for the parent)
            parent_pixels = set()
            for r in range(row_start, row_end):
                for c in range(col_start, col_end):
                    parent_pixels.add((r, c))

            # Create parent object
            parent = Object(
                id=parent_id,
                row=row_start,
                col=col_start,
                height=row_end - row_start,
                width=col_end - col_start,
                color=dominant_color,
                pixels=parent_pixels,
                is_background=False,
                is_divider=False
            )

            # Extract children within this region
            children = extract_children_in_region(
                sub_grid,
                parent_row_offset=row_start,
                parent_col_offset=col_start,
                start_id=child_id_counter,
                background_color=None,
                segmentation_mode=children_segmentation_mode
            )

            # Check for "fake" hierarchy: parent with single child that has same bbox
            # This happens when divider detection triggers but the region contains
            # only one object that fills the region. In this case, don't create
            # parent-child relationship - just use the child as an atomic object.
            if len(children) == 1:
                child = children[0]
                # Check if child bbox matches parent bbox
                same_bbox = (
                    child.row == parent.row and
                    child.col == parent.col and
                    child.height == parent.height and
                    child.width == parent.width
                )
                if same_bbox:
                    # Collapse hierarchy: use child as standalone object
                    child.parent = None
                    child.id = parent_id
                    parents.append(child)
                    parent_id += 1
                    continue

            # Link parent and children
            for child in children:
                child.parent = parent
            parent.children = children

            child_id_counter += len(children)
            parents.append(parent)
            parent_id += 1

    return parents


# =============================================================================
# Core Object Extraction Functions
# =============================================================================

def extract_connected_components(
    grid: np.ndarray,
    use_color_only: bool = False,
    background_color: Optional[int] = None,
    auto_detect_background: bool = True,
    min_bg_coverage: float = 0.4,
    skip_background: bool = True,
    segmentation_mode: Optional[SegmentationMode] = None
) -> Tuple[np.ndarray, List[int], List[Tuple[int, int, int, int]], List[bool]]:
    """
    Extract connected components from a grid with background-aware detection.

    This is the canonical object detection function, matching the approach in
    AffinitySlotAttention._compute_connected_components.

    Args:
        grid: (H, W) integer color values 0-9
        use_color_only: DEPRECATED - use segmentation_mode=SegmentationMode.COLOR instead.
                       If True, each color is one "object" (no connectivity check).
        background_color: Explicitly specify background color. If None and
                         auto_detect_background is True, will detect automatically.
        auto_detect_background: Whether to automatically detect background color
                               using the edge-connected component heuristic.
        min_bg_coverage: Minimum fraction of grid for background detection.
        skip_background: If True, skip the detected background color entirely
                        (default True for backward compatibility).
        segmentation_mode: How to segment objects. If provided, overrides use_color_only.
                          - CONNECTIVITY (default): Connected component analysis
                          - PIXEL: Each pixel is its own object
                          - COLOR: All pixels of same color form one object

    Returns:
        labels: (H, W) component IDs (0 = no component, 1+ = component IDs)
        colors: List of colors for each component (indexed by component_id - 1)
        bboxes: List of (min_row, min_col, max_row, max_col) for each component
        is_background: List of bool indicating if each component is background

    Example:
        >>> grid = np.array([[0, 1, 1], [0, 1, 0], [2, 2, 0]])
        >>> labels, colors, bboxes, is_bg = extract_connected_components(grid)
        >>> print(f"Found {len(colors)} components")
    """
    H, W = grid.shape

    # Resolve segmentation mode (new parameter takes priority over deprecated use_color_only)
    if segmentation_mode is None:
        if use_color_only:
            segmentation_mode = SegmentationMode.COLOR
        else:
            segmentation_mode = SegmentationMode.CONNECTIVITY

    # Detect background color early (needed for both divider and connectivity paths)
    if background_color is None and auto_detect_background:
        background_color = detect_background_color(grid, min_bg_coverage)

    # Check for divider lines first (takes priority over connectivity-based detection)
    # Only applies to CONNECTIVITY mode
    if segmentation_mode == SegmentationMode.CONNECTIVITY:
        horizontal_dividers, vertical_dividers = detect_divider_lines(grid)
        if horizontal_dividers or vertical_dividers:
            labels, colors, bboxes, is_bg, _ = segment_by_dividers(
                grid, horizontal_dividers, vertical_dividers,
                skip_background=skip_background,
                background_color=background_color
            )
            return labels, colors, bboxes, is_bg

    # Compute exterior mask if we have a background color (for marking, not skipping)
    exterior_mask = None
    if background_color is not None and not skip_background:
        exterior_mask = compute_exterior_mask(grid, background_color)

    # =========================================================================
    # PIXEL mode: Each pixel is its own object
    # =========================================================================
    if segmentation_mode == SegmentationMode.PIXEL:
        labels = np.zeros((H, W), dtype=np.int32)
        colors = []
        bboxes = []
        is_background_list = []
        component_id = 0

        for r in range(H):
            for c in range(W):
                pixel_color = int(grid[r, c])

                # Skip background pixels if requested
                if skip_background and pixel_color == background_color:
                    continue

                # Determine if this pixel is background
                is_bg = False
                if background_color is not None and pixel_color == background_color:
                    if exterior_mask is not None:
                        is_bg = bool(exterior_mask[r, c])

                component_id += 1
                labels[r, c] = component_id
                colors.append(pixel_color)
                is_background_list.append(is_bg)
                bboxes.append((r, c, r, c))  # Single pixel bbox

        return labels, colors, bboxes, is_background_list

    # =========================================================================
    # COLOR mode: Each color is one object (no connectivity check)
    # =========================================================================
    if segmentation_mode == SegmentationMode.COLOR:
        labels = np.zeros((H, W), dtype=np.int32)
        colors = []
        bboxes = []
        is_background_list = []
        component_id = 0

        for c in range(NUM_COLORS):
            # Skip background color entirely if requested
            if skip_background and c == background_color:
                continue

            mask = (grid == c)
            if not mask.any():
                continue

            # Determine if this is a background component (only relevant if not skipping)
            is_bg = False
            if background_color is not None and c == background_color and exterior_mask is not None:
                is_bg = bool((mask & exterior_mask).any())

            component_id += 1
            labels[mask] = component_id
            colors.append(c)
            is_background_list.append(is_bg)

            rows, cols = np.where(mask)
            bbox = (rows.min(), cols.min(), rows.max(), cols.max())
            bboxes.append(bbox)

        return labels, colors, bboxes, is_background_list

    # =========================================================================
    # CONNECTIVITY mode: Connected component analysis (default)
    # =========================================================================
    labels = np.zeros((H, W), dtype=np.int32)
    colors = []
    bboxes = []
    is_background_list = []
    component_id = 0

    for c in range(NUM_COLORS):
        # Skip background color entirely if requested
        if skip_background and c == background_color:
            continue

        mask = (grid == c)
        if not mask.any():
            continue

        # Use 4-connectivity for background color, 8-connectivity for others
        # This allows diagonal lines to act as barriers for background
        if c == background_color:
            structure = STRUCTURE_4CONN
        else:
            structure = STRUCTURE_8CONN

        labeled, num_features = ndimage.label(mask, structure=structure)

        for comp in range(1, num_features + 1):
            comp_mask = (labeled == comp)

            # Determine if this component is background (only relevant if not skipping)
            is_bg = False
            if background_color is not None and c == background_color and exterior_mask is not None:
                is_bg = bool((comp_mask & exterior_mask).any())

            component_id += 1
            labels[comp_mask] = component_id
            colors.append(c)
            is_background_list.append(is_bg)

            rows, cols = np.where(comp_mask)
            bbox = (rows.min(), cols.min(), rows.max(), cols.max())
            bboxes.append(bbox)

    return labels, colors, bboxes, is_background_list


def extract_objects_from_grid(
    grid: np.ndarray,
    skip_background: bool = True,
    auto_detect_background: bool = True,
    build_hierarchy: bool = False,
    segmentation_mode: Optional[SegmentationMode] = None,
    background_color: Optional[int] = None,
    children_segmentation_mode: Optional[SegmentationMode] = None
) -> List[Object]:
    """
    Extract connected components from a grid and return as Object instances.

    This is a convenience wrapper around extract_connected_components that
    returns Object instances instead of the (labels, colors, bboxes) format.

    If divider lines are detected (CONNECTIVITY mode only), segments by dividers
    and automatically builds parent/child hierarchy. Each region separated by
    dividers becomes a parent, with child objects extracted within each region.
    If a region has only one color, the parent has no children (atomic).

    Args:
        grid: (H, W) integer color values 0-9
        skip_background: If True, exclude background objects from results.
                        Default True for backward compatibility.
        auto_detect_background: Whether to detect background automatically.
                               Ignored if background_color is explicitly provided.
        build_hierarchy: If True, also build parent/child relationships
                        based on bounding box containment. When True,
                        returns only root objects (but all objects are
                        accessible via .children). For divider-based grids,
                        hierarchy is already built automatically.
        segmentation_mode: How to segment objects:
                          - CONNECTIVITY (default): Connected component analysis
                          - PIXEL: Each pixel is its own object
                          - COLOR: All pixels of same color form one object
        background_color: Explicitly specify background color. If provided,
                         overrides auto_detect_background. Use with
                         detect_background_color_for_pair() for consistent
                         background across input/output pairs.
        children_segmentation_mode: For divider-based grids, how to segment
                                   children within each region. If None, uses
                                   CONNECTIVITY. Set to PIXEL for accurate
                                   pixel-based mode counting within regions.

    Returns:
        List of Object instances. For divider-based grids, returns parent
        objects with children populated. For non-divider grids, returns
        flat list unless build_hierarchy=True.

    Example:
        >>> grid = np.array([[0, 1, 1], [0, 1, 0], [2, 2, 0]])
        >>> objects = extract_objects_from_grid(grid)
        >>> for obj in objects:
        ...     print(f"Object {obj.id}: color={obj.color}, area={obj.area}")
    """
    # Default to CONNECTIVITY mode
    if segmentation_mode is None:
        segmentation_mode = SegmentationMode.CONNECTIVITY

    # Check for divider lines first (only in CONNECTIVITY mode)
    if segmentation_mode == SegmentationMode.CONNECTIVITY:
        horizontal_dividers, vertical_dividers = detect_divider_lines(grid)
        if horizontal_dividers or vertical_dividers:
            # Use hierarchical segmentation - parents with children populated
            objects = segment_by_dividers_hierarchical(
                grid, horizontal_dividers, vertical_dividers,
                children_segmentation_mode=children_segmentation_mode
            )
            return objects

    # Use standard extraction with the specified segmentation mode
    # If background_color is explicitly provided, disable auto-detection
    effective_auto_detect = auto_detect_background if background_color is None else False
    labels, colors, bboxes, is_background = extract_connected_components(
        grid,
        skip_background=skip_background,
        auto_detect_background=effective_auto_detect,
        background_color=background_color,
        segmentation_mode=segmentation_mode
    )
    objects = labels_to_objects(labels, colors, bboxes, is_background)

    # Optionally build hierarchy based on containment
    if build_hierarchy:
        return build_containment_hierarchy(objects)

    return objects


def labels_to_objects(
    labels: np.ndarray,
    colors: List[int],
    bboxes: List[Tuple[int, int, int, int]],
    is_background: Optional[List[bool]] = None,
    is_divider: Optional[List[bool]] = None
) -> List[Object]:
    """
    Convert label-based representation to Object instances.

    Args:
        labels: (H, W) component IDs (0 = background, 1+ = component IDs)
        colors: List of colors for each component (indexed by component_id - 1)
        bboxes: List of (min_row, min_col, max_row, max_col) for each component
        is_background: Optional list of bool indicating if each component is background
        is_divider: Optional list of bool indicating if each component is a divider

    Returns:
        List of Object instances
    """
    if is_background is None:
        is_background = [False] * len(colors)
    if is_divider is None:
        is_divider = [False] * len(colors)

    objects = []
    for i, (color, bbox, is_bg, is_div) in enumerate(zip(colors, bboxes, is_background, is_divider)):
        min_row, min_col, max_row, max_col = bbox
        height = max_row - min_row + 1
        width = max_col - min_col + 1

        # Extract pixels for this object
        mask = (labels == i + 1)
        pixels = set(zip(*np.where(mask)))

        objects.append(Object(
            id=i,
            row=min_row,
            col=min_col,
            height=height,
            width=width,
            color=color,
            pixels=pixels,
            is_background=is_bg,
            is_divider=is_div
        ))

    return objects


# =============================================================================
# Hierarchy Construction
# =============================================================================

def build_containment_hierarchy(
    objects: List[Object],
    require_full_containment: bool = True,
    skip_background: bool = True,
    skip_dividers: bool = True
) -> List[Object]:
    """Build parent/child relationships based on bounding box containment.

    If object A's bbox fully contains object B's bbox, B becomes a child of A.
    Each object is assigned to its immediate (smallest containing) parent.

    Args:
        objects: List of Object instances (modified in place)
        require_full_containment: If True, bbox must fully contain.
                                  If False, partial overlap (center inside) counts.
        skip_background: Exclude background objects from hierarchy
        skip_dividers: Exclude divider objects from hierarchy

    Returns:
        List of root objects (those with no parent).
        The objects list is modified in place with parent/children set.

    Note:
        Objects that don't contain anything and aren't contained by anything
        remain as root objects with no children (atomic roots).
    """
    # Filter objects to consider for hierarchy
    candidates = [
        obj for obj in objects
        if not (skip_background and obj.is_background)
        and not (skip_dividers and obj.is_divider)
    ]

    if not candidates:
        return list(objects)  # Return original list unchanged

    # Clear any existing hierarchy
    for obj in candidates:
        obj.parent = None
        obj.children = []

    # Sort by area (largest first) - larger objects are potential parents
    sorted_by_area = sorted(candidates, key=lambda o: o.area, reverse=True)

    # Build containment relationships
    for i, potential_child in enumerate(sorted_by_area):
        child_bbox = potential_child.bbox  # (min_row, min_col, max_row, max_col)

        # Find the smallest containing object (immediate parent)
        best_parent = None
        best_parent_area = float('inf')

        for potential_parent in sorted_by_area[:i]:  # Only larger objects
            if potential_parent is potential_child:
                continue

            parent_bbox = potential_parent.bbox

            if require_full_containment:
                # Check if parent_bbox fully contains child_bbox
                contains = (
                    parent_bbox[0] <= child_bbox[0] and  # parent.min_row <= child.min_row
                    parent_bbox[1] <= child_bbox[1] and  # parent.min_col <= child.min_col
                    parent_bbox[2] >= child_bbox[2] and  # parent.max_row >= child.max_row
                    parent_bbox[3] >= child_bbox[3]      # parent.max_col >= child.max_col
                )
            else:
                # Partial overlap - child center is within parent bbox
                child_center = potential_child.center
                contains = (
                    parent_bbox[0] <= child_center[0] <= parent_bbox[2] and
                    parent_bbox[1] <= child_center[1] <= parent_bbox[3]
                )

            if contains and potential_parent.area < best_parent_area:
                best_parent = potential_parent
                best_parent_area = potential_parent.area

        if best_parent is not None:
            potential_child.parent = best_parent
            best_parent.children.append(potential_child)

    # Return only root objects (those with no parent)
    roots = [obj for obj in candidates if obj.parent is None]

    return roots


def get_hierarchy_stats(roots: List[Object]) -> Dict[str, Any]:
    """Get statistics about a hierarchy for debugging.

    Args:
        roots: List of root objects from build_containment_hierarchy()

    Returns:
        Dict with hierarchy statistics including:
            - num_roots: Number of root objects
            - total_objects: Total objects in hierarchy
            - max_depth: Maximum nesting depth
            - num_composite: Number of composite (non-leaf) objects
            - num_atomic_roots: Number of root objects with no children
    """
    def count_nodes(obj: Object) -> Tuple[int, int]:
        """Returns (total_nodes, max_depth)"""
        if obj.is_atomic:
            return (1, 0)
        total = 1
        max_child_depth = 0
        for child in obj.children:
            child_total, child_depth = count_nodes(child)
            total += child_total
            max_child_depth = max(max_child_depth, child_depth + 1)
        return (total, max_child_depth)

    total_objects = 0
    max_depth = 0
    composite_count = 0

    for root in roots:
        nodes, depth = count_nodes(root)
        total_objects += nodes
        max_depth = max(max_depth, depth)
        if root.is_composite:
            composite_count += 1

    return {
        'num_roots': len(roots),
        'total_objects': total_objects,
        'max_depth': max_depth,
        'num_composite': composite_count,
        'num_atomic_roots': len(roots) - composite_count,
    }


def get_matchable_objects(
    objects: List[Object],
    grid_shape: Optional[Tuple[int, int]] = None,
) -> List[Object]:
    """Get all objects at all hierarchy levels for correspondence matching.

    Returns parents, children, and atomic objects together. The correspondence
    module's similarity scoring will naturally select the best matches - parents
    tend to win over children when matching against large output regions due to
    structural features (area, bbox size).

    This approach allows MODE_OF_CHILDREN to work when parents are matched,
    while still supporting child-level matching when children are the best fit.

    Args:
        objects: List of root Object instances (may have parent/child relationships)
        grid_shape: Deprecated, kept for backwards compatibility (unused)

    Returns:
        List of all objects at all hierarchy levels suitable for matching.

    Example:
        >>> matchable = get_matchable_objects(input_objs)
        >>> # Returns: [parent1, child1a, child1b, parent2, child2a, atomic1, ...]
    """
    _ = grid_shape  # Unused, kept for backwards compatibility

    result = []
    for obj in objects:
        # Include the object itself (parent or atomic)
        result.append(obj)
        # Also include children if any
        if obj.children:
            result.extend(obj.children)
    return result


def flatten_hierarchy(objects: List[Object]) -> List[Object]:
    """Flatten a hierarchy into a single list of all objects.

    Unlike get_matchable_objects, this returns ALL objects (parents AND children).
    Useful when you need to iterate over every object regardless of hierarchy level.

    Args:
        objects: List of root Object instances

    Returns:
        Flat list containing all objects in the hierarchy.
    """
    result = []
    for obj in objects:
        result.append(obj)
        if obj.children:
            result.extend(flatten_hierarchy(obj.children))
    return result


# =============================================================================
# Object Sorting Functions
# =============================================================================

def sort_objects_left_to_right(
    labels: np.ndarray,
    colors: List[int],
    bboxes: List[Tuple[int, int, int, int]],
    is_background: Optional[List[bool]] = None
) -> Tuple[np.ndarray, List[int], List[Tuple[int, int, int, int]], List[bool]]:
    """
    Reorder objects by their centroid column (left-to-right).

    This canonicalizes object indices so that object 0 is always the leftmost,
    object 1 is the next, etc. This allows the model to learn rules based on
    spatial ordering.

    Args:
        labels: (H, W) component IDs (0 = background, 1+ = component IDs)
        colors: List of colors for each component
        bboxes: List of (min_row, min_col, max_row, max_col) for each component
        is_background: Optional list of background flags

    Returns:
        Reordered (labels, colors, bboxes, is_background) with leftmost object as index 1
    """
    if len(colors) == 0:
        return labels, colors, bboxes, is_background or []

    if is_background is None:
        is_background = [False] * len(colors)

    # Compute centroid columns for each object
    centroids_col = []
    for i in range(len(colors)):
        mask = (labels == i + 1)
        cols = np.where(mask)[1]
        centroids_col.append(cols.mean())

    # Get sorting order (leftmost first)
    order = np.argsort(centroids_col)

    # Relabel according to new order
    new_labels = np.zeros_like(labels)
    new_colors = []
    new_bboxes = []
    new_is_background = []

    for new_idx, old_idx in enumerate(order):
        old_mask = (labels == old_idx + 1)
        new_labels[old_mask] = new_idx + 1
        new_colors.append(colors[old_idx])
        new_bboxes.append(bboxes[old_idx])
        new_is_background.append(is_background[old_idx])

    return new_labels, new_colors, new_bboxes, new_is_background


def sort_objects_by_strategy(
    labels: np.ndarray,
    colors: List[int],
    bboxes: List[Tuple[int, int, int, int]],
    strategy_name: str,
    ordering_strategies: Optional[List[Any]] = None,
    is_background: Optional[List[bool]] = None
) -> Tuple[np.ndarray, List[int], List[Tuple[int, int, int, int]], List[bool]]:
    """
    Reorder objects according to a named ordering strategy.

    This generalizes sort_objects_left_to_right to support all ordering
    strategies from ordering_module.py.

    Args:
        labels: (H, W) component IDs (0 = background, 1+ = component IDs)
        colors: List of colors for each component
        bboxes: List of (min_row, min_col, max_row, max_col) for each component
        strategy_name: Name of ordering strategy (e.g., 'left_to_right', 'top_to_bottom')
        ordering_strategies: Optional list of ordering strategy instances.
                            If None, will import from ordering_module.
        is_background: Optional list of background flags

    Returns:
        Reordered (labels, colors, bboxes, is_background) according to strategy
    """
    if len(colors) == 0:
        return labels, colors, bboxes, is_background or []

    if is_background is None:
        is_background = [False] * len(colors)

    # Get ordering strategies if not provided
    if ordering_strategies is None:
        from ordering_module import ALL_ORDERINGS
        ordering_strategies = ALL_ORDERINGS

    # Find the named strategy
    strategy = None
    for s in ordering_strategies:
        if s.name == strategy_name:
            strategy = s
            break

    if strategy is None:
        available = [s.name for s in ordering_strategies]
        raise ValueError(f"Unknown ordering strategy: {strategy_name}. "
                        f"Available: {available}")

    # Convert to Object instances
    objects = labels_to_objects(labels, colors, bboxes, is_background)

    # Apply ordering strategy
    ordered_objects = strategy.order(objects)

    # Get new order (mapping from old index to new index)
    old_to_new = {obj.id: new_idx for new_idx, obj in enumerate(ordered_objects)}

    # Relabel according to new order
    new_labels = np.zeros_like(labels)
    new_colors = [None] * len(colors)
    new_bboxes = [None] * len(bboxes)
    new_is_background = [None] * len(is_background)

    for old_idx in range(len(colors)):
        new_idx = old_to_new[old_idx]
        old_mask = (labels == old_idx + 1)
        new_labels[old_mask] = new_idx + 1
        new_colors[new_idx] = colors[old_idx]
        new_bboxes[new_idx] = bboxes[old_idx]
        new_is_background[new_idx] = is_background[old_idx]

    return new_labels, new_colors, new_bboxes, new_is_background


# =============================================================================
# Object Property Computation
# =============================================================================

def compute_object_properties(
    labels: np.ndarray,
    colors: List[int],
    bboxes: List[Tuple[int, int, int, int]],
    grid_size: int = 30,
    is_background: Optional[List[bool]] = None
) -> Dict[str, np.ndarray]:
    """
    Compute properties for each object from labels.

    Returns normalized properties suitable for neural network input.

    Args:
        labels: (H, W) component IDs (0 = background, 1+ = component IDs)
        colors: List of colors for each component
        bboxes: List of (min_row, min_col, max_row, max_col) for each component
        grid_size: Size used for normalization (default 30)
        is_background: Optional list of background flags

    Returns:
        Dict with:
            centroids: (MAX_OBJECTS, 2) row, col of centroid (normalized by grid_size)
            bboxes: (MAX_OBJECTS, 4) min_row, min_col, max_row, max_col (normalized)
            areas: (MAX_OBJECTS,) number of pixels (normalized)
            colors: (MAX_OBJECTS,) color index
            valid: (MAX_OBJECTS,) bool mask indicating valid objects
            is_background: (MAX_OBJECTS,) bool mask indicating background objects
    """
    num_objects = len(colors)

    if is_background is None:
        is_background = [False] * num_objects

    if num_objects == 0:
        return {
            'centroids': np.zeros((MAX_OBJECTS, 2), dtype=np.float32),
            'bboxes': np.zeros((MAX_OBJECTS, 4), dtype=np.float32),
            'areas': np.zeros(MAX_OBJECTS, dtype=np.float32),
            'colors': np.zeros(MAX_OBJECTS, dtype=np.int64),
            'valid': np.zeros(MAX_OBJECTS, dtype=bool),
            'is_background': np.zeros(MAX_OBJECTS, dtype=bool),
        }

    H, W = labels.shape

    centroids = np.zeros((MAX_OBJECTS, 2), dtype=np.float32)
    bboxes_arr = np.zeros((MAX_OBJECTS, 4), dtype=np.float32)
    areas = np.zeros(MAX_OBJECTS, dtype=np.float32)
    colors_arr = np.zeros(MAX_OBJECTS, dtype=np.int64)
    valid = np.zeros(MAX_OBJECTS, dtype=bool)
    is_bg_arr = np.zeros(MAX_OBJECTS, dtype=bool)

    for i in range(min(num_objects, MAX_OBJECTS)):
        mask = (labels == i + 1)
        area = mask.sum()

        if area > 0:
            rows, cols = np.where(mask)
            centroid_row = rows.mean()
            centroid_col = cols.mean()

            centroids[i] = [centroid_row / grid_size, centroid_col / grid_size]
            bboxes_arr[i] = [
                bboxes[i][0] / grid_size,
                bboxes[i][1] / grid_size,
                bboxes[i][2] / grid_size,
                bboxes[i][3] / grid_size
            ]
            areas[i] = area / (grid_size * grid_size)
            colors_arr[i] = colors[i]
            valid[i] = True
            is_bg_arr[i] = is_background[i]

    return {
        'centroids': centroids,
        'bboxes': bboxes_arr,
        'areas': areas,
        'colors': colors_arr,
        'valid': valid,
        'is_background': is_bg_arr,
    }


# =============================================================================
# Utility Functions
# =============================================================================

def compute_iou_from_pixels(pixels1: Set[Tuple[int, int]],
                            pixels2: Set[Tuple[int, int]]) -> float:
    """Compute Intersection over Union between two pixel sets.

    Args:
        pixels1: Set of (row, col) tuples for first object
        pixels2: Set of (row, col) tuples for second object

    Returns:
        IoU score in [0, 1]
    """
    if not pixels1 or not pixels2:
        return 0.0

    intersection = len(pixels1 & pixels2)
    union = len(pixels1 | pixels2)

    return intersection / union if union > 0 else 0.0


def get_object_mask(labels: np.ndarray, object_id: int) -> np.ndarray:
    """Get binary mask for a specific object.

    Args:
        labels: (H, W) component IDs
        object_id: Object ID (1-indexed in labels)

    Returns:
        (H, W) boolean mask for the object
    """
    return labels == object_id


def count_objects(labels: np.ndarray) -> int:
    """Count the number of objects in a label array.

    Args:
        labels: (H, W) component IDs (0 = background)

    Returns:
        Number of distinct objects
    """
    return int(labels.max())


def filter_foreground_objects(objects: List[Object]) -> List[Object]:
    """Filter out background objects from a list.

    Args:
        objects: List of Object instances

    Returns:
        List containing only foreground objects (is_background=False)
    """
    return [obj for obj in objects if not obj.is_background]


# =============================================================================
# Visualization (for --puzzle-id mode)
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

# Distinct colors for object IDs (for segmentation overlay)
OBJECT_COLORS = [
    '#E6194B', '#3CB44B', '#FFE119', '#4363D8', '#F58231',
    '#911EB4', '#46F0F0', '#F032E6', '#BCF60C', '#FABEBE',
    '#008080', '#E6BEFF', '#9A6324', '#FFFAC8', '#800000',
    '#AAFFC3', '#808000', '#FFD8B1', '#000075', '#808080',
]


def _draw_grid(ax, grid: np.ndarray, title: str = ""):
    """Draw an ARC grid on a matplotlib axis."""
    H, W = grid.shape

    # Create RGB image
    rgb_image = np.zeros((H, W, 3), dtype=np.float32)
    for c in range(10):
        color = np.array([int(ARC_COLORS[c][i:i+2], 16) / 255.0 for i in (1, 3, 5)])
        mask = (grid == c)
        rgb_image[mask] = color

    ax.imshow(rgb_image, interpolation='nearest')

    # Draw grid lines
    for i in range(H + 1):
        ax.axhline(y=i - 0.5, color='gray', linewidth=0.5)
    for j in range(W + 1):
        ax.axvline(x=j - 0.5, color='gray', linewidth=0.5)

    ax.set_xlim(-0.5, W - 0.5)
    ax.set_ylim(H - 0.5, -0.5)
    ax.set_aspect('equal')
    ax.set_title(title, fontsize=10)
    ax.axis('off')


def _draw_segmentation(ax, grid: np.ndarray, objects: List[Object], title: str = "",
                       horizontal_dividers: List[Tuple[int, int, int]] = None,
                       vertical_dividers: List[Tuple[int, int, int]] = None):
    """Draw object segmentation overlay on grid with dividers and hierarchy.

    Args:
        ax: Matplotlib axis
        grid: The grid to draw
        objects: List of Object instances
        title: Title for the subplot
        horizontal_dividers: List of (start_row, end_row, color) for H dividers
        vertical_dividers: List of (start_col, end_col, color) for V dividers
    """
    import matplotlib.patches as mpatches

    H, W = grid.shape
    horizontal_dividers = horizontal_dividers or []
    vertical_dividers = vertical_dividers or []

    # Create RGB image (dimmed original)
    rgb_image = np.zeros((H, W, 3), dtype=np.float32)
    for c in range(10):
        color = np.array([int(ARC_COLORS[c][i:i+2], 16) / 255.0 for i in (1, 3, 5)])
        mask = (grid == c)
        rgb_image[mask] = color * 0.5  # Dimmed background

    # Highlight divider regions with a distinct tint
    divider_tint = np.array([1.0, 0.3, 0.3])  # Red tint for dividers
    for start_row, end_row, _ in horizontal_dividers:
        for r in range(start_row, end_row):
            rgb_image[r, :] = rgb_image[r, :] * 0.5 + divider_tint * 0.5
    for start_col, end_col, _ in vertical_dividers:
        for c in range(start_col, end_col):
            rgb_image[:, c] = rgb_image[:, c] * 0.5 + divider_tint * 0.5

    # Collect all objects including children for pixel overlay
    all_objects = []
    for obj in objects:
        all_objects.append(obj)
        if obj.children:
            all_objects.extend(obj.children)

    # Overlay object colors
    for obj in all_objects:
        obj_color_hex = OBJECT_COLORS[obj.id % len(OBJECT_COLORS)]
        obj_color = np.array([int(obj_color_hex[i:i+2], 16) / 255.0 for i in (1, 3, 5)])
        for (r, c) in obj.pixels:
            if 0 <= r < H and 0 <= c < W:
                rgb_image[r, c] = obj_color

    ax.imshow(rgb_image, interpolation='nearest')

    # Draw divider boundary lines (red dashed)
    for start_row, end_row, div_color in horizontal_dividers:
        # Draw lines at top and bottom of divider region
        ax.axhline(y=start_row - 0.5, color='red', linewidth=2, linestyle='--', alpha=0.8)
        ax.axhline(y=end_row - 0.5, color='red', linewidth=2, linestyle='--', alpha=0.8)
        # Label the divider
        mid_row = (start_row + end_row) / 2 - 0.5
        ax.annotate(f'H-DIV c={div_color}', (W/2, mid_row),
                   color='red', fontsize=7, fontweight='bold',
                   ha='center', va='center', alpha=0.9)

    for start_col, end_col, div_color in vertical_dividers:
        # Draw lines at left and right of divider region
        ax.axvline(x=start_col - 0.5, color='red', linewidth=2, linestyle='--', alpha=0.8)
        ax.axvline(x=end_col - 0.5, color='red', linewidth=2, linestyle='--', alpha=0.8)
        # Label the divider
        mid_col = (start_col + end_col) / 2 - 0.5
        ax.annotate(f'V-DIV c={div_color}', (mid_col, H/2),
                   color='red', fontsize=7, fontweight='bold',
                   ha='center', va='center', rotation=90, alpha=0.9)

    # Draw bounding boxes and labels - parents first, then children
    for obj in objects:
        obj_color_hex = OBJECT_COLORS[obj.id % len(OBJECT_COLORS)]

        # Parents get thick solid boxes
        if obj.is_composite:
            rect = mpatches.Rectangle(
                (obj.col - 0.5, obj.row - 0.5),
                obj.width, obj.height,
                linewidth=3, edgecolor=obj_color_hex, facecolor='none',
                linestyle='-'
            )
            ax.add_patch(rect)

            # Parent label at top-left corner (inside box) to avoid overlap with children
            label = f'P{obj.id}'
            ax.annotate(label, (obj.col - 0.3, obj.row - 0.3),
                       color='white', fontsize=9, fontweight='bold',
                       ha='left', va='top',
                       bbox=dict(boxstyle='round', facecolor=obj_color_hex, alpha=0.9,
                                edgecolor='white', linewidth=2))

            # Draw children with dashed boxes
            for child in obj.children:
                child_color_hex = OBJECT_COLORS[child.id % len(OBJECT_COLORS)]
                child_rect = mpatches.Rectangle(
                    (child.col - 0.5, child.row - 0.5),
                    child.width, child.height,
                    linewidth=1.5, edgecolor=child_color_hex, facecolor='none',
                    linestyle='--'
                )
                ax.add_patch(child_rect)

                # Child label at center
                child_center_row = child.row + child.height / 2
                child_center_col = child.col + child.width / 2
                child_label = f'C{child.id}'
                ax.annotate(child_label, (child_center_col, child_center_row),
                           color='white', fontsize=7, fontweight='bold',
                           ha='center', va='center',
                           bbox=dict(boxstyle='circle', facecolor=child_color_hex, alpha=0.85))
        else:
            # Atomic objects (no children) - regular boxes
            rect = mpatches.Rectangle(
                (obj.col - 0.5, obj.row - 0.5),
                obj.width, obj.height,
                linewidth=2, edgecolor=obj_color_hex, facecolor='none'
            )
            ax.add_patch(rect)

            # Regular object label at center
            center_row = obj.row + obj.height / 2
            center_col = obj.col + obj.width / 2
            label = f'{obj.id}'
            ax.annotate(label, (center_col, center_row),
                       color='white', fontsize=8, fontweight='bold',
                       ha='center', va='center',
                       bbox=dict(boxstyle='circle', facecolor=obj_color_hex, alpha=0.9))

    # Draw grid lines
    for i in range(H + 1):
        ax.axhline(y=i - 0.5, color='gray', linewidth=0.3, alpha=0.5)
    for j in range(W + 1):
        ax.axvline(x=j - 0.5, color='gray', linewidth=0.3, alpha=0.5)

    ax.set_xlim(-0.5, W - 0.5)
    ax.set_ylim(H - 0.5, -0.5)
    ax.set_aspect('equal')
    ax.set_title(title, fontsize=10)
    ax.axis('off')


class ObjectSegmentationNavigator:
    """Interactive navigator for viewing object segmentation examples."""

    def __init__(self, puzzle_id: str, examples_data: List[dict]):
        self.puzzle_id = puzzle_id
        self.examples_data = examples_data
        self.current_idx = 0
        self.fig = None

    def draw_example(self, ex_idx: int):
        """Draw a single example (training or test pair)."""
        import matplotlib.pyplot as plt

        if self.fig is not None:
            self.fig.clear()
        else:
            self.fig = plt.figure(figsize=(16, 8))

        data = self.examples_data[ex_idx]
        input_grid = data['input_grid']
        output_grid = data['output_grid']
        input_objects = data['input_objects']
        output_objects = data['output_objects']
        input_h_dividers = data.get('input_h_dividers', [])
        input_v_dividers = data.get('input_v_dividers', [])
        output_h_dividers = data.get('output_h_dividers', [])
        output_v_dividers = data.get('output_v_dividers', [])
        example_type = data['type']
        example_num = data['num']

        # Count children for display
        input_children = sum(len(obj.children) for obj in input_objects)
        output_children = sum(len(obj.children) for obj in output_objects) if output_objects else 0

        # Layout: 2x2 grid (original and segmented for input/output)
        gs = self.fig.add_gridspec(2, 2, wspace=0.15, hspace=0.25)

        # Top-left: Input original
        ax_in_orig = self.fig.add_subplot(gs[0, 0])
        _draw_grid(ax_in_orig, input_grid, f"Input Grid ({input_grid.shape[0]}x{input_grid.shape[1]})")

        # Top-right: Output original
        ax_out_orig = self.fig.add_subplot(gs[0, 1])
        if output_grid is not None:
            _draw_grid(ax_out_orig, output_grid, f"Output Grid ({output_grid.shape[0]}x{output_grid.shape[1]})")
        else:
            ax_out_orig.text(0.5, 0.5, "No output\n(test example)", ha='center', va='center',
                           fontsize=14, transform=ax_out_orig.transAxes)
            ax_out_orig.axis('off')
            ax_out_orig.set_title("Output Grid")

        # Bottom-left: Input segmentation
        ax_in_seg = self.fig.add_subplot(gs[1, 0])
        n_dividers_in = len(input_h_dividers) + len(input_v_dividers)
        title_in = f"Input: {len(input_objects)} obj"
        if input_children > 0:
            title_in += f" ({input_children} children)"
        if n_dividers_in > 0:
            title_in += f", {n_dividers_in} dividers"
        _draw_segmentation(ax_in_seg, input_grid, input_objects, title_in,
                          horizontal_dividers=input_h_dividers,
                          vertical_dividers=input_v_dividers)

        # Bottom-right: Output segmentation
        ax_out_seg = self.fig.add_subplot(gs[1, 1])
        if output_grid is not None and output_objects:
            n_dividers_out = len(output_h_dividers) + len(output_v_dividers)
            title_out = f"Output: {len(output_objects)} obj"
            if output_children > 0:
                title_out += f" ({output_children} children)"
            if n_dividers_out > 0:
                title_out += f", {n_dividers_out} dividers"
            _draw_segmentation(ax_out_seg, output_grid, output_objects, title_out,
                              horizontal_dividers=output_h_dividers,
                              vertical_dividers=output_v_dividers)
        else:
            if output_grid is not None:
                _draw_grid(ax_out_seg, output_grid, "Output (no objects detected)")
            else:
                ax_out_seg.text(0.5, 0.5, "No output", ha='center', va='center',
                               fontsize=14, transform=ax_out_seg.transAxes)
                ax_out_seg.axis('off')
                ax_out_seg.set_title("Output Segmentation")

        n_examples = len(self.examples_data)
        self.fig.suptitle(
            f"Object Segmentation: {self.puzzle_id}\n"
            f"{example_type} Example {example_num} ({ex_idx + 1}/{n_examples})  "
            f"[← / → to navigate, q to quit]",
            fontsize=12, fontweight='bold'
        )
        plt.subplots_adjust(top=0.88)
        self.fig.canvas.draw()

    def on_key(self, event):
        """Handle keyboard navigation."""
        import matplotlib.pyplot as plt
        if event.key == 'right' or event.key == 'n':
            self.current_idx = (self.current_idx + 1) % len(self.examples_data)
            self.draw_example(self.current_idx)
        elif event.key == 'left' or event.key == 'p':
            self.current_idx = (self.current_idx - 1) % len(self.examples_data)
            self.draw_example(self.current_idx)
        elif event.key == 'q':
            plt.close(self.fig)

    def show(self):
        """Display the interactive navigator."""
        import matplotlib.pyplot as plt
        self.draw_example(0)
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        plt.show()


def _print_object_hierarchy(objects: List[Object], indent: int = 0):
    """
    Print objects with their parent/child hierarchy.

    Shows composite objects (parents) with their children indented below them,
    and marks atomic objects, dividers, and background objects.

    Args:
        objects: List of objects to print
        indent: Base indentation level (spaces)
    """
    prefix = " " * indent

    # Separate root objects from nested ones
    root_objects = [obj for obj in objects if obj.is_root]

    def print_obj(obj: Object, level: int = 0):
        obj_prefix = prefix + "  " * level

        # Build status flags
        flags = []
        if obj.is_divider:
            flags.append("DIVIDER")
        if obj.is_background:
            flags.append("BG")
        if obj.is_composite:
            flags.append(f"PARENT: {len(obj.children)} children")
        if obj.is_atomic and not obj.is_divider and not obj.is_background:
            flags.append("atomic")

        flag_str = f" [{', '.join(flags)}]" if flags else ""

        print(f"{obj_prefix}Object {obj.id}: color={obj.color}, size={obj.area}, "
              f"bbox=({obj.row},{obj.col})-({obj.row+obj.height-1},{obj.col+obj.width-1}){flag_str}")

        # Print children indented
        for child in obj.children:
            print_obj(child, level + 1)

    for obj in root_objects:
        print_obj(obj)


def visualize_puzzle_objects(puzzle_id: str, data_root: str = "kaggle/combined"):
    """
    Visualize object segmentation for a specific puzzle.

    Shows a scrollable view of all training and test examples with their
    object segmentations, including divider lines and parent/child hierarchy.

    Args:
        puzzle_id: The ARC puzzle ID (e.g., "1990f7a8")
        data_root: Path to puzzle data
    """
    print(f"Loading puzzle: {puzzle_id}")
    puzzle = _load_puzzle(puzzle_id, data_root)
    print(f"Found {len(puzzle['train'])} training examples, {len(puzzle.get('test', []))} test examples")

    examples_data = []

    # Process training examples
    for i, example in enumerate(puzzle['train']):
        input_grid = np.array(example['input'], dtype=np.int64)
        output_grid = np.array(example['output'], dtype=np.int64) if 'output' in example else None

        print(f"\nTraining example {i + 1}:")
        print(f"  Input shape: {input_grid.shape}")
        if output_grid is not None:
            print(f"  Output shape: {output_grid.shape}")

        # Detect dividers separately for visualization
        input_h_div, input_v_div = detect_divider_lines(input_grid)
        output_h_div, output_v_div = ([], [])
        if output_grid is not None:
            output_h_div, output_v_div = detect_divider_lines(output_grid)

        # Print divider info
        if input_h_div or input_v_div:
            print(f"  Input dividers: {len(input_h_div)} horizontal, {len(input_v_div)} vertical")
            for start, end, color in input_h_div:
                print(f"    H-DIV rows {start}-{end-1}, color={color}")
            for start, end, color in input_v_div:
                print(f"    V-DIV cols {start}-{end-1}, color={color}")

        if output_h_div or output_v_div:
            print(f"  Output dividers: {len(output_h_div)} horizontal, {len(output_v_div)} vertical")
            for start, end, color in output_h_div:
                print(f"    H-DIV rows {start}-{end-1}, color={color}")
            for start, end, color in output_v_div:
                print(f"    V-DIV cols {start}-{end-1}, color={color}")

        # Extract objects with pair-based background detection
        bg_color = None
        if output_grid is not None:
            bg_color = detect_background_color_for_pair(input_grid, output_grid)
            if bg_color is not None:
                print(f"  Pair background color: {bg_color}")
            else:
                print(f"  No dominant background (using pair-based detection)")
        input_objects = extract_objects_from_grid(input_grid, background_color=bg_color)
        output_objects = extract_objects_from_grid(output_grid, background_color=bg_color) if output_grid is not None else []

        print(f"  Input objects: {len(input_objects)}")
        _print_object_hierarchy(input_objects, indent=4)

        if output_objects:
            print(f"  Output objects: {len(output_objects)}")
            _print_object_hierarchy(output_objects, indent=4)

        examples_data.append({
            'input_grid': input_grid,
            'output_grid': output_grid,
            'input_objects': input_objects,
            'output_objects': output_objects,
            'input_h_dividers': input_h_div,
            'input_v_dividers': input_v_div,
            'output_h_dividers': output_h_div,
            'output_v_dividers': output_v_div,
            'type': 'Training',
            'num': i + 1,
        })

    # Process test examples
    for i, example in enumerate(puzzle.get('test', [])):
        input_grid = np.array(example['input'], dtype=np.int64)
        output_grid = np.array(example['output'], dtype=np.int64) if 'output' in example else None

        print(f"\nTest example {i + 1}:")
        print(f"  Input shape: {input_grid.shape}")
        if output_grid is not None:
            print(f"  Output shape: {output_grid.shape}")

        # Detect dividers separately for visualization
        input_h_div, input_v_div = detect_divider_lines(input_grid)
        output_h_div, output_v_div = ([], [])
        if output_grid is not None:
            output_h_div, output_v_div = detect_divider_lines(output_grid)

        # Print divider info
        if input_h_div or input_v_div:
            print(f"  Input dividers: {len(input_h_div)} horizontal, {len(input_v_div)} vertical")
            for start, end, color in input_h_div:
                print(f"    H-DIV rows {start}-{end-1}, color={color}")
            for start, end, color in input_v_div:
                print(f"    V-DIV cols {start}-{end-1}, color={color}")

        if output_h_div or output_v_div:
            print(f"  Output dividers: {len(output_h_div)} horizontal, {len(output_v_div)} vertical")
            for start, end, color in output_h_div:
                print(f"    H-DIV rows {start}-{end-1}, color={color}")
            for start, end, color in output_v_div:
                print(f"    V-DIV cols {start}-{end-1}, color={color}")

        # Extract objects with pair-based background detection
        bg_color = None
        if output_grid is not None:
            input_bg_color = detect_background_color(input_grid)
            output_bg_color = detect_background_color(output_grid)

            if bg_color is not None:
                print(f"  Pair background color: {bg_color}")
            else:
                print(f"  No dominant background (using pair-based detection)")
        input_objects = extract_objects_from_grid(input_grid, background_color=input_bg_color)
        output_objects = extract_objects_from_grid(output_grid, background_color=output_bg_color) if output_grid is not None else []

        print(f"  Input objects: {len(input_objects)}")
        _print_object_hierarchy(input_objects, indent=4)

        if output_objects:
            print(f"  Output objects: {len(output_objects)}")
            _print_object_hierarchy(output_objects, indent=4)

        examples_data.append({
            'input_grid': input_grid,
            'output_grid': output_grid,
            'input_objects': input_objects,
            'output_objects': output_objects,
            'input_h_dividers': input_h_div,
            'input_v_dividers': input_v_div,
            'output_h_dividers': output_h_div,
            'output_v_dividers': output_v_div,
            'type': 'Test',
            'num': i + 1,
        })

    # Show interactive visualization
    print("\nGenerating visualization...")
    print("Legend: Red dashed lines = dividers, P# = parent objects, C# = child objects")
    navigator = ObjectSegmentationNavigator(puzzle_id, examples_data)
    navigator.show()


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Object Module for ARC Puzzles - Visualize object segmentation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python object_module.py --puzzle-id 1990f7a8
    python object_module.py --puzzle-id 009d5c81 --data-root kaggle/combined
        """
    )

    parser.add_argument("--puzzle-id", type=str, required=True,
                        help="ARC puzzle ID to visualize (e.g., 1990f7a8)")
    parser.add_argument("--data-root", type=str, default="kaggle/combined",
                        help="Path to puzzle data")

    args = parser.parse_args()

    visualize_puzzle_objects(
        args.puzzle_id,
        data_root=args.data_root
    )
