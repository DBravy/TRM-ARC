#!/usr/bin/env python3
"""
Regional Correspondence Module for ARC Puzzle Solver

This module handles puzzles where the grid is split into two regions (e.g., a small
"pattern" region and a larger "template" region), and correspondences are established
based on LOGICAL POSITION within each region, potentially with spatial transformations
(rotations, reflections).

Core Insight:
    Some puzzles establish correspondence not by shape similarity, but by POSITION
    within a logical grid. A template cell at position (r,c) corresponds to the
    pattern cell at position transform(r,c), where transform might be a rotation
    or reflection.

Key Concepts:
    - RegionSplit: How to divide the grid into pattern and template regions
    - LogicalCell: A cell with its logical grid position (not just pixel position)
    - SpatialTransform: Rotation/reflection to apply to positions for correspondence
    - RegionalCorrespondence: Position-based matching between regions

Use Cases:
    - Puzzle 103eff5b: Pattern defines colors, template defines shape, correspondence
      via 90° rotation of logical positions

Usage:
    from regional_correspondence_module import (
        discover_regional_rule,
        apply_regional_rule,
        RegionalRule,
    )

    # Discover rule from training examples
    rule = discover_regional_rule(puzzle, verbose=True)

    # Apply to test input
    output_grid = apply_regional_rule(test_input, rule)
"""

import numpy as np
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Set, Tuple, Any
from collections import Counter

from object_module import (
    Object,
    extract_objects_from_grid,
    SegmentationMode,
)


# =============================================================================
# Spatial Transforms
# =============================================================================

class SpatialTransform(Enum):
    """Spatial transformations for position correspondence."""
    IDENTITY = auto()
    ROTATE_90_CW = auto()      # (r, c) -> (c, n-1-r)
    ROTATE_180 = auto()        # (r, c) -> (n-1-r, n-1-c)
    ROTATE_90_CCW = auto()     # (r, c) -> (n-1-c, r)
    FLIP_HORIZONTAL = auto()   # (r, c) -> (r, n-1-c)
    FLIP_VERTICAL = auto()     # (r, c) -> (n-1-r, c)
    FLIP_DIAGONAL = auto()     # (r, c) -> (c, r)
    FLIP_ANTIDIAGONAL = auto() # (r, c) -> (n-1-c, n-1-r)


def apply_transform(
    pos: Tuple[int, int],
    transform: SpatialTransform,
    grid_size: Tuple[int, int]
) -> Tuple[int, int]:
    """Apply a spatial transform to a logical position.

    Args:
        pos: (row, col) position in logical grid
        transform: The transformation to apply
        grid_size: (height, width) of the logical grid

    Returns:
        Transformed (row, col) position
    """
    r, c = pos
    h, w = grid_size

    if transform == SpatialTransform.IDENTITY:
        return (r, c)
    elif transform == SpatialTransform.ROTATE_90_CW:
        # 90° clockwise: (r,c) -> (c, h-1-r)
        return (c, h - 1 - r)
    elif transform == SpatialTransform.ROTATE_180:
        return (h - 1 - r, w - 1 - c)
    elif transform == SpatialTransform.ROTATE_90_CCW:
        # 90° counter-clockwise: (r,c) -> (w-1-c, r)
        return (w - 1 - c, r)
    elif transform == SpatialTransform.FLIP_HORIZONTAL:
        return (r, w - 1 - c)
    elif transform == SpatialTransform.FLIP_VERTICAL:
        return (h - 1 - r, c)
    elif transform == SpatialTransform.FLIP_DIAGONAL:
        return (c, r)
    elif transform == SpatialTransform.FLIP_ANTIDIAGONAL:
        return (w - 1 - c, h - 1 - r)

    return pos


# =============================================================================
# Region Types
# =============================================================================

class RegionType(Enum):
    """Types of regions in the grid."""
    PATTERN = auto()   # Small region with color information
    TEMPLATE = auto()  # Large region to be filled


@dataclass
class LogicalCell:
    """A cell with both pixel and logical grid position."""
    # Pixel-level info
    pixels: Set[Tuple[int, int]]
    color: int
    row: int  # Top-left row in pixel coordinates
    col: int  # Top-left col in pixel coordinates
    height: int
    width: int

    # Logical grid position
    logical_row: int
    logical_col: int

    # Which region this cell belongs to
    region_type: RegionType

    def __hash__(self):
        return hash((self.logical_row, self.logical_col, self.region_type))


@dataclass
class Region:
    """A region of the grid containing cells at logical positions."""
    region_type: RegionType
    cells: List[LogicalCell]
    # Bounding box in pixel coordinates
    row: int
    col: int
    height: int
    width: int
    # Logical grid dimensions
    logical_height: int
    logical_width: int

    def get_cell_at(self, logical_row: int, logical_col: int) -> Optional[LogicalCell]:
        """Get the cell at a logical position, or None if empty."""
        for cell in self.cells:
            if cell.logical_row == logical_row and cell.logical_col == logical_col:
                return cell
        return None

    def get_occupied_positions(self) -> Set[Tuple[int, int]]:
        """Get the set of occupied logical positions."""
        return {(c.logical_row, c.logical_col) for c in self.cells}


# =============================================================================
# Region Splitting
# =============================================================================

class SplitMode(Enum):
    """How to split the grid into regions."""
    BY_COLOR = auto()       # Template is one color (e.g., gray=8), pattern is rest
    SPATIAL_VERTICAL = auto()   # Top half vs bottom half
    SPATIAL_HORIZONTAL = auto() # Left half vs right half
    BY_SIZE = auto()        # Large objects vs small objects


@dataclass
class RegionSplit:
    """Specification for how to split the grid."""
    mode: SplitMode
    template_color: Optional[int] = None  # For BY_COLOR mode

    def describe(self) -> str:
        if self.mode == SplitMode.BY_COLOR:
            return f"by_color(template={self.template_color})"
        return self.mode.name.lower()


def detect_region_split(
    input_grid: np.ndarray,
    output_grid: np.ndarray,
    verbose: bool = False
) -> Optional[RegionSplit]:
    """Detect how to split the grid into pattern and template regions.

    Heuristics:
    1. BY_COLOR: If one color (often 8=gray) forms large connected blocks that
       get filled with other colors in output, that's the template.
    2. SPATIAL: If objects cluster in distinct spatial regions.
    """
    H, W = input_grid.shape

    # Count colors and their pixel counts
    input_colors = Counter(input_grid.flatten())
    output_colors = Counter(output_grid.flatten())

    # Remove background (0) from consideration
    input_colors.pop(0, None)
    output_colors.pop(0, None)

    if verbose:
        print(f"Input colors: {dict(input_colors)}")
        print(f"Output colors: {dict(output_colors)}")

    # Heuristic 1: BY_COLOR
    # Look for a color that has many pixels in input but fewer in output
    # (indicating it gets "replaced" by other colors)
    for color, input_count in input_colors.items():
        output_count = output_colors.get(color, 0)

        # If this color is significantly reduced in output, it might be template
        if input_count > 20 and output_count < input_count * 0.5:
            if verbose:
                print(f"Color {color} reduced from {input_count} to {output_count} - likely template")
            return RegionSplit(mode=SplitMode.BY_COLOR, template_color=color)

    # Heuristic 2: Look for color 8 (gray) which is commonly used as template
    if 8 in input_colors and input_colors[8] > 10:
        if verbose:
            print("Found color 8 (gray) with significant pixels - assuming template")
        return RegionSplit(mode=SplitMode.BY_COLOR, template_color=8)

    # Heuristic 3: Spatial split - check if objects cluster in top/bottom halves
    # Extract objects
    objects = extract_objects_from_grid(input_grid, segmentation_mode=SegmentationMode.CONNECTIVITY)
    objects = [o for o in objects if not o.is_background and o.color > 0]

    if objects:
        # Check vertical clustering
        centroids = [(o.row + o.height/2) for o in objects]
        mid_row = H / 2
        top_objects = sum(1 for c in centroids if c < mid_row)
        bottom_objects = sum(1 for c in centroids if c >= mid_row)

        if top_objects > 0 and bottom_objects > 0:
            # Objects in both halves - check if one half has larger objects
            top_sizes = [len(o.pixels) for o in objects if o.row + o.height/2 < mid_row]
            bottom_sizes = [len(o.pixels) for o in objects if o.row + o.height/2 >= mid_row]

            avg_top = np.mean(top_sizes) if top_sizes else 0
            avg_bottom = np.mean(bottom_sizes) if bottom_sizes else 0

            if avg_bottom > avg_top * 3:
                if verbose:
                    print("Bottom objects much larger - likely template")
                return RegionSplit(mode=SplitMode.SPATIAL_VERTICAL)

    return None


def split_grid_into_regions(
    grid: np.ndarray,
    split: RegionSplit,
    verbose: bool = False
) -> Tuple[Optional[Region], Optional[Region]]:
    """Split the grid into pattern and template regions.

    Returns:
        (pattern_region, template_region) or (None, None) if split fails
    """
    H, W = grid.shape

    if split.mode == SplitMode.BY_COLOR:
        template_color = split.template_color

        # Extract all objects
        objects = extract_objects_from_grid(grid, segmentation_mode=SegmentationMode.CONNECTIVITY)
        objects = [o for o in objects if not o.is_background and o.color > 0]

        # Separate into template objects (gray) and pattern objects (other colors)
        template_objects = [o for o in objects if o.color == template_color]
        pattern_objects = [o for o in objects if o.color != template_color]

        if verbose:
            print(f"Template objects (color {template_color}): {len(template_objects)}")
            print(f"Pattern objects: {len(pattern_objects)}")

        if not template_objects or not pattern_objects:
            return None, None

        # Create regions
        pattern_region = create_region_from_objects(pattern_objects, RegionType.PATTERN, grid, verbose)
        template_region = create_region_from_objects(template_objects, RegionType.TEMPLATE, grid, verbose)

        return pattern_region, template_region

    elif split.mode == SplitMode.SPATIAL_VERTICAL:
        # Split into top and bottom halves
        mid_row = H // 2

        # Find objects in each half
        objects = extract_objects_from_grid(grid, segmentation_mode=SegmentationMode.CONNECTIVITY)
        objects = [o for o in objects if not o.is_background and o.color > 0]

        top_objects = [o for o in objects if o.row + o.height/2 < mid_row]
        bottom_objects = [o for o in objects if o.row + o.height/2 >= mid_row]

        if not top_objects or not bottom_objects:
            return None, None

        # Assume smaller objects are pattern, larger are template
        avg_top = np.mean([len(o.pixels) for o in top_objects])
        avg_bottom = np.mean([len(o.pixels) for o in bottom_objects])

        if avg_top < avg_bottom:
            pattern_objects, template_objects = top_objects, bottom_objects
        else:
            pattern_objects, template_objects = bottom_objects, top_objects

        pattern_region = create_region_from_objects(pattern_objects, RegionType.PATTERN, grid, verbose)
        template_region = create_region_from_objects(template_objects, RegionType.TEMPLATE, grid, verbose)

        return pattern_region, template_region

    return None, None


def detect_cell_size(
    pixels: Set[Tuple[int, int]],
    verbose: bool = False
) -> Tuple[int, int]:
    """Detect the cell size in a template region.

    Strategy: Look at the horizontal structure within rows to find cell width,
    and vertical structure within columns to find cell height.
    The template may be connected but have a regular block structure.

    Returns:
        (cell_height, cell_width) - the detected cell size
    """
    if not pixels:
        return (1, 1)

    rows = sorted(set(r for r, c in pixels))
    cols = sorted(set(c for r, c in pixels))

    if len(rows) < 2 or len(cols) < 2:
        return (len(rows), len(cols))

    min_r, max_r = min(rows), max(rows)
    min_c, max_c = min(cols), max(cols)

    # Build a presence map for quick lookup
    presence = set(pixels)

    # Strategy: For each row, look at which columns are filled
    # Find the "run lengths" of consecutive filled columns
    # The most common run length is the cell width

    col_run_lengths = []
    for r in rows:
        # Get filled columns in this row
        row_cols = sorted(c for (pr, c) in pixels if pr == r)
        if not row_cols:
            continue

        # Find runs of consecutive columns
        run_len = 1
        for i in range(1, len(row_cols)):
            if row_cols[i] == row_cols[i-1] + 1:
                run_len += 1
            else:
                col_run_lengths.append(run_len)
                run_len = 1
        col_run_lengths.append(run_len)

    # Similarly for rows - look at runs within each column
    row_run_lengths = []
    for c in cols:
        col_rows = sorted(r for (r, pc) in pixels if pc == c)
        if not col_rows:
            continue

        run_len = 1
        for i in range(1, len(col_rows)):
            if col_rows[i] == col_rows[i-1] + 1:
                run_len += 1
            else:
                row_run_lengths.append(run_len)
                run_len = 1
        row_run_lengths.append(run_len)

    if verbose:
        col_runs_counter = Counter(col_run_lengths)
        row_runs_counter = Counter(row_run_lengths)
        print(f"    Col run lengths: {dict(col_runs_counter)}")
        print(f"    Row run lengths: {dict(row_runs_counter)}")

    # The most common run length is likely the cell size
    # Filter to runs >= 2 as cells should be at least 2x2
    valid_col_runs = [r for r in col_run_lengths if r >= 2]
    valid_row_runs = [r for r in row_run_lengths if r >= 2]

    def find_cell_size(run_lengths):
        """Find the fundamental cell size from run lengths.

        Prefer smaller sizes that divide larger sizes evenly.
        This handles cases where two adjacent cells form a longer run.
        """
        if not run_lengths:
            return 2

        counter = Counter(run_lengths)
        unique_sizes = sorted(counter.keys())

        if len(unique_sizes) == 1:
            return unique_sizes[0]

        # Check if smaller sizes divide larger sizes
        smallest = unique_sizes[0]
        for size in unique_sizes[1:]:
            if size % smallest == 0:
                # The smallest is likely the fundamental unit
                return smallest

        # Otherwise, return the most common
        return counter.most_common(1)[0][0]

    cell_w = find_cell_size(valid_col_runs) if valid_col_runs else 2
    cell_h = find_cell_size(valid_row_runs) if valid_row_runs else 2

    return (cell_h, cell_w)


def split_template_into_cells(
    template_obj: Object,
    grid: np.ndarray,
    verbose: bool = False
) -> List[LogicalCell]:
    """Split a single template object into logical cells.

    The template is often a single connected component, but it has internal
    structure (blocks arranged in a grid pattern). This function detects
    that structure and splits into logical cells.
    """
    pixels = set(template_obj.pixels)
    if not pixels:
        return []

    # Get bounding box
    rows = sorted(set(r for r, c in pixels))
    cols = sorted(set(c for r, c in pixels))
    min_r, max_r = min(rows), max(rows)
    min_c, max_c = min(cols), max(cols)

    # Detect cell size
    cell_h, cell_w = detect_cell_size(pixels, verbose)

    if verbose:
        print(f"    Template bounding box: ({min_r},{min_c}) to ({max_r},{max_c})")
        print(f"    Detected cell size: {cell_h}x{cell_w}")

    cells = []

    # Scan the template region in cell-sized chunks
    logical_row = 0
    r = min_r
    while r <= max_r:
        logical_col = 0
        c = min_c
        while c <= max_c:
            # Check if this cell position has template pixels
            cell_pixels = set()
            for dr in range(cell_h):
                for dc in range(cell_w):
                    if (r + dr, c + dc) in pixels:
                        cell_pixels.add((r + dr, c + dc))

            # If at least half the cell is filled, consider it occupied
            if len(cell_pixels) >= (cell_h * cell_w) // 2:
                cell = LogicalCell(
                    pixels=cell_pixels,
                    color=template_obj.color,
                    row=r,
                    col=c,
                    height=cell_h,
                    width=cell_w,
                    logical_row=logical_row,
                    logical_col=logical_col,
                    region_type=RegionType.TEMPLATE
                )
                cells.append(cell)
                if verbose:
                    print(f"    Template cell at pixel ({r},{c}) -> logical ({logical_row},{logical_col})")

            c += cell_w
            logical_col += 1
        r += cell_h
        logical_row += 1

    return cells


def create_pattern_region(
    objects: List[Object],
    grid: np.ndarray,
    verbose: bool = False
) -> Optional[Region]:
    """Create a pattern region from individual objects (each object is a cell).

    The pattern is typically a small grid where each colored pixel/object
    represents a logical cell. We need to map pixel positions to logical
    grid positions.
    """
    if not objects:
        return None

    # Compute bounding box of all pattern objects
    min_row = min(o.row for o in objects)
    max_row = max(o.row + o.height for o in objects)
    min_col = min(o.col for o in objects)
    max_col = max(o.col + o.width for o in objects)

    region_height = max_row - min_row
    region_width = max_col - min_col

    # For pattern, each object's position determines its logical position
    # The logical grid is based on the bounding box relative positions

    # Get unique row and col positions (use top-left of each object)
    row_positions = sorted(set(o.row for o in objects))
    col_positions = sorted(set(o.col for o in objects))

    if verbose:
        print(f"  Region type: PATTERN")
        print(f"  Objects: {len(objects)}")
        print(f"  Row positions: {row_positions}")
        print(f"  Col positions: {col_positions}")

    # Map each row/col position to a logical index
    row_to_logical = {r: i for i, r in enumerate(row_positions)}
    col_to_logical = {c: i for i, c in enumerate(col_positions)}

    # Map each object to its logical position
    cells = []
    for obj in objects:
        # Find the closest row/col position
        closest_row = min(row_positions, key=lambda r: abs(r - obj.row))
        closest_col = min(col_positions, key=lambda c: abs(c - obj.col))

        logical_row = row_to_logical[closest_row]
        logical_col = col_to_logical[closest_col]

        cell = LogicalCell(
            pixels=set(obj.pixels),
            color=obj.color,
            row=obj.row,
            col=obj.col,
            height=obj.height,
            width=obj.width,
            logical_row=logical_row,
            logical_col=logical_col,
            region_type=RegionType.PATTERN
        )
        cells.append(cell)

        if verbose:
            print(f"    Pattern cell at pixel ({obj.row},{obj.col}), color={obj.color} -> logical ({logical_row},{logical_col})")

    return Region(
        region_type=RegionType.PATTERN,
        cells=cells,
        row=min_row,
        col=min_col,
        height=region_height,
        width=region_width,
        logical_height=len(row_positions),
        logical_width=len(col_positions)
    )


def create_template_region(
    objects: List[Object],
    grid: np.ndarray,
    verbose: bool = False
) -> Optional[Region]:
    """Create a template region, splitting connected objects into logical cells."""
    if not objects:
        return None

    # If there's only one object (common case), split it into cells
    if len(objects) == 1:
        cells = split_template_into_cells(objects[0], grid, verbose)
        if not cells:
            return None

        # Compute region bounds
        all_pixels = set()
        for cell in cells:
            all_pixels.update(cell.pixels)

        rows = [r for r, c in all_pixels]
        cols = [c for r, c in all_pixels]
        min_row, max_row = min(rows), max(rows)
        min_col, max_col = min(cols), max(cols)

        logical_height = max(c.logical_row for c in cells) + 1
        logical_width = max(c.logical_col for c in cells) + 1

        return Region(
            region_type=RegionType.TEMPLATE,
            cells=cells,
            row=min_row,
            col=min_col,
            height=max_row - min_row + 1,
            width=max_col - min_col + 1,
            logical_height=logical_height,
            logical_width=logical_width
        )

    # Multiple separate objects - each is a cell (less common)
    # Use same logic as pattern but for template
    return create_pattern_region(objects, grid, verbose)


def create_region_from_objects(
    objects: List[Object],
    region_type: RegionType,
    grid: np.ndarray,
    verbose: bool = False
) -> Optional[Region]:
    """Create a Region with LogicalCells from a list of objects.

    This function:
    1. For PATTERN: Each object is a cell, cluster by position
    2. For TEMPLATE: Split connected objects into block-structured cells
    """
    if not objects:
        return None

    if region_type == RegionType.PATTERN:
        return create_pattern_region(objects, grid, verbose)
    else:
        return create_template_region(objects, grid, verbose)


# =============================================================================
# Transform Screening
# =============================================================================

def get_fallback_pattern_color(
    pattern_region: Region,
    template_pos: Tuple[int, int],
    transform: SpatialTransform,
    grid_size: Tuple[int, int]
) -> Optional[int]:
    """Find a fallback color when no direct pattern correspondence exists.

    Strategy: Consider the rotation transform to find pattern cells that would
    map to the same template row or column.

    For rotations:
    - 90° CCW: template_row corresponds to pattern_col (2-template_row)
    - 90° CW: template_row corresponds to pattern_col (template_row)
    - 180°: template_row corresponds to pattern_row (grid_h-1-template_row)

    We look for any pattern cell that would end up in the same template row/col.
    """
    t_row, t_col = template_pos
    h, w = grid_size

    # Find all pattern cells that map to the same template row
    row_colors = []
    for cell in pattern_region.cells:
        mapped_pos = apply_transform(
            (cell.logical_row, cell.logical_col),
            transform, grid_size
        )
        if mapped_pos[0] == t_row:
            row_colors.append(cell.color)

    if row_colors:
        # Return the most common color
        return Counter(row_colors).most_common(1)[0][0]

    # Find all pattern cells that map to the same template column
    col_colors = []
    for cell in pattern_region.cells:
        mapped_pos = apply_transform(
            (cell.logical_row, cell.logical_col),
            transform, grid_size
        )
        if mapped_pos[1] == t_col:
            col_colors.append(cell.color)

    if col_colors:
        return Counter(col_colors).most_common(1)[0][0]

    return None


def screen_spatial_transforms(
    pattern_region: Region,
    template_region: Region,
    input_grid: np.ndarray,
    output_grid: np.ndarray,
    verbose: bool = False
) -> Tuple[Optional[SpatialTransform], float]:
    """Screen spatial transformations to find the one with zero variance.

    For each transform, check if:
        template_cell[pos].output_color == pattern[transform(pos)].color

    When no direct pattern cell exists at the transformed position, we try
    fallback strategies (same row/column) to find a matching color.

    Args:
        pattern_region: Region with pattern cells
        template_region: Region with template cells
        input_grid: Original input grid
        output_grid: Expected output grid

    Returns:
        (best_transform, variance) - transform with lowest variance
    """
    transforms = list(SpatialTransform)

    # Get logical grid size (use max of both regions)
    logical_h = max(pattern_region.logical_height, template_region.logical_height)
    logical_w = max(pattern_region.logical_width, template_region.logical_width)

    best_transform = None
    best_variance = float('inf')

    for transform in transforms:
        errors = 0
        total = 0

        for template_cell in template_region.cells:
            # Get logical position in template
            t_pos = (template_cell.logical_row, template_cell.logical_col)

            # Apply transform to get corresponding pattern position
            p_pos = apply_transform(t_pos, transform, (logical_h, logical_w))

            # Find pattern cell at that position
            pattern_cell = pattern_region.get_cell_at(p_pos[0], p_pos[1])

            # Get actual output color
            sample_r, sample_c = next(iter(template_cell.pixels))
            if 0 <= sample_r < output_grid.shape[0] and 0 <= sample_c < output_grid.shape[1]:
                actual_color = output_grid[sample_r, sample_c]
            else:
                continue

            if pattern_cell is not None:
                expected_color = pattern_cell.color
            else:
                # Try fallback - look for pattern cells that map to same template row/col
                expected_color = get_fallback_pattern_color(
                    pattern_region, t_pos, transform, (logical_h, logical_w)
                )
                if expected_color is None:
                    # No fallback found - if output is 0, that's fine
                    if actual_color == 0:
                        continue
                    else:
                        errors += 1
                        total += 1
                        continue

            if actual_color != expected_color:
                errors += 1
            total += 1

        variance = errors / total if total > 0 else float('inf')

        if verbose:
            print(f"  Transform {transform.name}: errors={errors}/{total}, variance={variance:.4f}")

        if variance < best_variance:
            best_variance = variance
            best_transform = transform

    return best_transform, best_variance


# =============================================================================
# Regional Rule
# =============================================================================

@dataclass
class RegionalRule:
    """Complete rule for regional correspondence transformation."""
    split: RegionSplit
    transform: SpatialTransform

    # Whether pattern region is preserved in output
    preserve_pattern: bool = True

    # Variance metrics
    transform_variance: float = float('inf')

    def describe(self) -> str:
        return (f"RegionalRule(split={self.split.describe()}, "
                f"transform={self.transform.name}, "
                f"preserve_pattern={self.preserve_pattern})")


# =============================================================================
# Discovery Function
# =============================================================================

def discover_regional_rule(
    puzzle: Dict,
    verbose: bool = False
) -> Optional[RegionalRule]:
    """Discover the regional correspondence rule from training examples.

    Args:
        puzzle: Puzzle dict with 'train' examples
        verbose: Print debug info

    Returns:
        RegionalRule if found, None otherwise
    """
    train_examples = puzzle.get('train', [])
    if not train_examples:
        return None

    if verbose:
        print("=" * 60)
        print("Discovering Regional Correspondence Rule")
        print("=" * 60)

    # Use first example to detect split mode
    first_input = np.array(train_examples[0]['input'])
    first_output = np.array(train_examples[0]['output'])

    if verbose:
        print("\nStep 1: Detect region split")

    split = detect_region_split(first_input, first_output, verbose)
    if split is None:
        if verbose:
            print("Could not detect region split")
        return None

    if verbose:
        print(f"  Split mode: {split.describe()}")

    # Screen transforms across all examples
    if verbose:
        print("\nStep 2: Screen spatial transforms")

    transform_votes = Counter()
    transform_variances = {t: [] for t in SpatialTransform}

    for ex_idx, ex in enumerate(train_examples):
        input_grid = np.array(ex['input'])
        output_grid = np.array(ex['output'])

        if verbose:
            print(f"\n  Example {ex_idx + 1}:")

        # Split into regions
        pattern_region, template_region = split_grid_into_regions(input_grid, split, verbose)

        if pattern_region is None or template_region is None:
            if verbose:
                print("    Could not split into regions")
            continue

        if verbose:
            print(f"    Pattern region: {len(pattern_region.cells)} cells, "
                  f"logical grid {pattern_region.logical_height}x{pattern_region.logical_width}")
            print(f"    Template region: {len(template_region.cells)} cells, "
                  f"logical grid {template_region.logical_height}x{template_region.logical_width}")

        # Screen transforms for this example
        best_transform, variance = screen_spatial_transforms(
            pattern_region, template_region, input_grid, output_grid, verbose
        )

        if best_transform is not None:
            transform_votes[best_transform] += 1
            transform_variances[best_transform].append(variance)

    # Find transform with most votes and lowest variance
    if not transform_votes:
        if verbose:
            print("\nNo valid transforms found")
        return None

    best_transform = transform_votes.most_common(1)[0][0]
    avg_variance = np.mean(transform_variances[best_transform]) if transform_variances[best_transform] else float('inf')

    if verbose:
        print(f"\nBest transform: {best_transform.name}")
        print(f"  Votes: {transform_votes[best_transform]}/{len(train_examples)}")
        print(f"  Average variance: {avg_variance:.4f}")

    # Check if pattern is preserved in output
    preserve_pattern = True  # Default assumption

    rule = RegionalRule(
        split=split,
        transform=best_transform,
        preserve_pattern=preserve_pattern,
        transform_variance=avg_variance
    )

    if verbose:
        print(f"\n{rule.describe()}")

    return rule


# =============================================================================
# Application Function
# =============================================================================

def apply_regional_rule(
    input_grid: np.ndarray,
    rule: RegionalRule,
    output_shape: Optional[Tuple[int, int]] = None,
    verbose: bool = False
) -> np.ndarray:
    """Apply a regional rule to produce output grid.

    Args:
        input_grid: The input grid
        rule: The regional rule to apply
        output_shape: Shape of output (defaults to input shape)
        verbose: Print debug info

    Returns:
        Generated output grid
    """
    if output_shape is None:
        output_shape = input_grid.shape

    # Start with copy of input (preserves pattern if needed)
    if rule.preserve_pattern:
        output_grid = input_grid.copy()
    else:
        output_grid = np.zeros(output_shape, dtype=input_grid.dtype)

    # Split into regions
    pattern_region, template_region = split_grid_into_regions(input_grid, rule.split, verbose)

    if pattern_region is None or template_region is None:
        if verbose:
            print("Could not split into regions")
        return output_grid

    # Get logical grid size
    logical_h = max(pattern_region.logical_height, template_region.logical_height)
    logical_w = max(pattern_region.logical_width, template_region.logical_width)

    if verbose:
        print(f"Logical grid size: {logical_h}x{logical_w}")
        print(f"Transform: {rule.transform.name}")

    # For each template cell, find corresponding pattern cell and fill
    for template_cell in template_region.cells:
        t_pos = (template_cell.logical_row, template_cell.logical_col)

        # Apply transform to get pattern position
        p_pos = apply_transform(t_pos, rule.transform, (logical_h, logical_w))

        # Find pattern cell at that position
        pattern_cell = pattern_region.get_cell_at(p_pos[0], p_pos[1])

        if pattern_cell is not None:
            fill_color = pattern_cell.color
        else:
            # Try fallback - look for pattern cells that map to same template row/col
            fill_color = get_fallback_pattern_color(
                pattern_region, t_pos, rule.transform, (logical_h, logical_w)
            )
            if fill_color is None:
                fill_color = 0  # No fallback found, use background

        # Fill template cell with the determined color
        for r, c in template_cell.pixels:
            if 0 <= r < output_shape[0] and 0 <= c < output_shape[1]:
                output_grid[r, c] = fill_color

        if verbose:
            print(f"  Template {t_pos} -> Pattern {p_pos} -> color {fill_color}")

    return output_grid


# =============================================================================
# CLI for Testing
# =============================================================================

def main():
    import argparse
    from puzzle_loader import load_puzzle

    parser = argparse.ArgumentParser(description='Regional Correspondence Module')
    parser.add_argument('--puzzle-id', type=str, required=True, help='Puzzle ID')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')

    args = parser.parse_args()

    # Load puzzle
    puzzle = load_puzzle(args.puzzle_id)

    print(f"Analyzing puzzle {args.puzzle_id} with regional correspondence")
    print("=" * 60)

    # Discover rule
    rule = discover_regional_rule(puzzle, verbose=args.verbose)

    if rule is None:
        print("\nNo regional rule discovered")
        return

    print(f"\n{'='*60}")
    print(f"Discovered Rule: {rule.describe()}")
    print(f"Transform Variance: {rule.transform_variance:.4f}")
    print(f"{'='*60}")

    # Test on training examples
    print("\nTesting on training examples:")

    total_correct = 0
    total_examples = len(puzzle['train'])

    for i, ex in enumerate(puzzle['train']):
        input_grid = np.array(ex['input'])
        expected_output = np.array(ex['output'])

        predicted_output = apply_regional_rule(
            input_grid, rule,
            output_shape=expected_output.shape,
            verbose=args.verbose
        )

        match = np.array_equal(predicted_output, expected_output)
        accuracy = np.mean(predicted_output == expected_output)

        if match:
            total_correct += 1

        status = "PASS" if match else "FAIL"
        print(f"  Example {i+1}: {status} (pixel accuracy: {accuracy:.1%})")

        if args.verbose and not match:
            print(f"    Expected:\n{expected_output}")
            print(f"    Predicted:\n{predicted_output}")

    print(f"\n{'='*60}")
    print(f"Total: {total_correct}/{total_examples} examples correct")

    # Test on test examples if available
    if puzzle.get('test'):
        print(f"\n{'='*60}")
        print("Testing on test examples:")
        print("=" * 60)

        for i, ex in enumerate(puzzle['test']):
            input_grid = np.array(ex['input'])

            # Try to get expected output shape
            if 'output' in ex:
                expected = np.array(ex['output'])
                out_shape = expected.shape
            else:
                out_shape = input_grid.shape
                expected = None

            predicted = apply_regional_rule(input_grid, rule, output_shape=out_shape, verbose=args.verbose)

            if expected is not None:
                match = np.array_equal(predicted, expected)
                accuracy = np.mean(predicted == expected)
                status = "PASS" if match else "FAIL"
                print(f"  Test {i+1}: {status} (pixel accuracy: {accuracy:.1%})")
            else:
                print(f"  Test {i+1}: Prediction shape {predicted.shape}")


if __name__ == "__main__":
    main()
