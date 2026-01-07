#!/usr/bin/env python3
"""
Aggregation Module for ARC Puzzle Solver

Provides region-based pixel counting and color analysis, unified with the
parent/child object hierarchy system. Regions ARE parents - counting within
a parent object uses the same machinery as counting in spatial regions.

Key Concepts:
    - RegionStats: Pixel-level statistics (color counts, mode color)
    - ObjectCount: Object-level statistics (count objects by color)
    - RegionAggregation: Combined stats for a region (parent object)

Usage:
    from aggregation_module import (
        RegionStats,
        ObjectCount,
        RegionAggregation,
        compute_region_stats,
        mode_color,
        count_objects_in_region,
        aggregate_regions,
    )

    # Get mode color of whole grid
    dominant = mode_color(grid)

    # Get mode color within a parent object's region
    dominant = mode_color(grid, region=parent_object)

    # Count objects in a region (uses parent.children)
    counts = count_objects_in_region(objects, region=parent_object)
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from object_module import Object


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class RegionStats:
    """Statistics about pixels in a region.

    Attributes:
        color_counts: Mapping from color (0-9) to pixel count
        total_pixels: Total number of pixels in region
        mode_color: Most common color in the region
        unique_colors: Set of distinct colors present
    """
    color_counts: Dict[int, int]
    total_pixels: int
    mode_color: int
    unique_colors: Set[int]

    @property
    def color_distribution(self) -> Dict[int, float]:
        """Return normalized color distribution (proportions sum to 1)."""
        if self.total_pixels == 0:
            return {}
        return {c: count / self.total_pixels
                for c, count in self.color_counts.items()}

    @property
    def num_colors(self) -> int:
        """Number of distinct colors in region."""
        return len(self.unique_colors)

    def count(self, color: int) -> int:
        """Get count for a specific color."""
        return self.color_counts.get(color, 0)

    def proportion(self, color: int) -> float:
        """Get proportion for a specific color."""
        if self.total_pixels == 0:
            return 0.0
        return self.color_counts.get(color, 0) / self.total_pixels


@dataclass
class ObjectCount:
    """Count of objects by color in a region.

    Attributes:
        object_counts_by_color: Mapping from color to count of objects
        total_objects: Total number of objects
    """
    object_counts_by_color: Dict[int, int]
    total_objects: int

    @classmethod
    def from_objects(cls, objects: List['Object']) -> 'ObjectCount':
        """Count objects by color from a list of Object instances."""
        counts: Dict[int, int] = {}
        for obj in objects:
            counts[obj.color] = counts.get(obj.color, 0) + 1
        return cls(object_counts_by_color=counts, total_objects=len(objects))

    @property
    def colors_present(self) -> Set[int]:
        """Set of colors with at least one object."""
        return set(self.object_counts_by_color.keys())

    def count(self, color: int) -> int:
        """Get object count for a specific color."""
        return self.object_counts_by_color.get(color, 0)

    def mode_color(self) -> Optional[int]:
        """Get the color with the most objects, or None if empty."""
        if not self.object_counts_by_color:
            return None
        return max(self.object_counts_by_color.keys(),
                   key=lambda c: self.object_counts_by_color[c])


@dataclass
class RegionAggregation:
    """Complete aggregation data for a region (region = parent object).

    Attributes:
        region: The parent object defining this region
        pixel_stats: Pixel-level statistics
        object_stats: Object-level statistics (children)
        child_objects: List of child objects in region
    """
    region: 'Object'
    pixel_stats: RegionStats
    object_stats: ObjectCount
    child_objects: List['Object']

    @property
    def pixel_mode_color(self) -> int:
        """Mode color based on pixel counts."""
        return self.pixel_stats.mode_color

    @property
    def object_mode_color(self) -> Optional[int]:
        """Mode color based on object counts."""
        return self.object_stats.mode_color()


# =============================================================================
# Core Functions
# =============================================================================

def compute_region_stats(
    grid: np.ndarray,
    region_mask: Optional[np.ndarray] = None
) -> RegionStats:
    """
    Compute pixel statistics for a region defined by a mask.

    Args:
        grid: (H, W) color grid with values 0-9
        region_mask: (H, W) boolean mask defining the region.
                     If None, uses entire grid.

    Returns:
        RegionStats with color counts, mode color, etc.
    """
    if region_mask is None:
        pixels = grid.flatten()
    else:
        pixels = grid[region_mask]

    total_pixels = len(pixels)

    if total_pixels == 0:
        return RegionStats(
            color_counts={},
            total_pixels=0,
            mode_color=0,
            unique_colors=set()
        )

    # Count colors
    color_counts: Dict[int, int] = {}
    for color in range(10):
        count = int(np.sum(pixels == color))
        if count > 0:
            color_counts[color] = count

    # Find mode (most common color)
    mode = max(color_counts.keys(), key=lambda c: color_counts[c])

    return RegionStats(
        color_counts=color_counts,
        total_pixels=total_pixels,
        mode_color=mode,
        unique_colors=set(color_counts.keys())
    )


def mode_color(
    grid: np.ndarray,
    region: Optional['Object'] = None,
    region_mask: Optional[np.ndarray] = None,
    exclude_background: bool = False,
    background_color: Optional[int] = None
) -> int:
    """
    Find the dominant (most common) color in a region.

    Args:
        grid: (H, W) color grid with values 0-9
        region: Object defining the region (uses its bounding box and pixels)
        region_mask: Alternative: explicit mask for the region
        exclude_background: If True, exclude background color from consideration
        background_color: Color to exclude (required if exclude_background=True)

    Returns:
        The color (0-9) with highest pixel count in the region.

    Note:
        If both region and region_mask are None, uses entire grid.
        If region is provided, uses its bounding box.
    """
    if region is not None:
        # Create mask from region's bounding box
        mask = np.zeros(grid.shape, dtype=bool)
        r1, c1 = region.row, region.col
        r2, c2 = r1 + region.height, c1 + region.width
        mask[r1:r2, c1:c2] = True

        # If region has explicit pixels, use those instead
        if region.pixels:
            mask = np.zeros(grid.shape, dtype=bool)
            for r, c in region.pixels:
                if 0 <= r < grid.shape[0] and 0 <= c < grid.shape[1]:
                    mask[r, c] = True

        region_mask = mask

    stats = compute_region_stats(grid, region_mask)

    if exclude_background and background_color is not None:
        # Remove background from consideration
        filtered_counts = {c: count for c, count in stats.color_counts.items()
                          if c != background_color}
        if filtered_counts:
            return max(filtered_counts.keys(), key=lambda c: filtered_counts[c])

    return stats.mode_color


def count_objects_in_region(
    objects: List['Object'],
    region: 'Object'
) -> ObjectCount:
    """
    Count objects by color within a region (parent's children).

    Uses the parent/child hierarchy - counts region.children.
    This is the "unified" approach where regions ARE parents.

    Args:
        objects: All objects (for context, though not used directly)
        region: The parent object defining the region

    Returns:
        ObjectCount with per-color counts of children
    """
    children = region.children if region.children else []
    return ObjectCount.from_objects(children)


def count_all_objects(objects: List['Object']) -> ObjectCount:
    """
    Count all objects by color (grid-level counting).

    Args:
        objects: List of all objects

    Returns:
        ObjectCount with per-color counts
    """
    return ObjectCount.from_objects(objects)


def aggregate_grid(
    grid: np.ndarray,
    objects: Optional[List['Object']] = None,
    include_background: bool = False,
    background_color: Optional[int] = None
) -> Tuple[RegionStats, ObjectCount]:
    """
    Aggregate statistics for the entire grid.

    Args:
        grid: (H, W) color grid
        objects: Optional list of objects for object-level stats
        include_background: Whether to include background color in pixel stats
        background_color: The background color (for filtering if needed)

    Returns:
        (RegionStats, ObjectCount) for the whole grid
    """
    # Pixel stats
    if not include_background and background_color is not None:
        mask = grid != background_color
        pixel_stats = compute_region_stats(grid, mask)
    else:
        pixel_stats = compute_region_stats(grid)

    # Object stats
    if objects is not None:
        obj_list = objects
        if not include_background:
            obj_list = [o for o in objects if not o.is_background]
        object_stats = ObjectCount.from_objects(obj_list)
    else:
        object_stats = ObjectCount(object_counts_by_color={}, total_objects=0)

    return pixel_stats, object_stats


def aggregate_regions(
    grid: np.ndarray,
    objects: List['Object']
) -> List[RegionAggregation]:
    """
    Compute aggregation for each parent (composite) object.

    For divider-segmented grids, each region becomes a parent with
    child objects. This function aggregates stats for each region.

    Args:
        grid: (H, W) color grid
        objects: List of objects (with parent/child hierarchy)

    Returns:
        List of RegionAggregation, one per parent (composite) object.
        Returns empty list if no composite objects exist.
    """
    results = []

    for obj in objects:
        if obj.is_composite:
            # This is a parent/region - aggregate its children

            # Pixel stats within region's bounding box
            mask = np.zeros(grid.shape, dtype=bool)
            r1, c1 = obj.row, obj.col
            r2, c2 = r1 + obj.height, c1 + obj.width
            mask[r1:r2, c1:c2] = True
            pixel_stats = compute_region_stats(grid, mask)

            # Object stats from children
            object_stats = count_objects_in_region(objects, obj)

            results.append(RegionAggregation(
                region=obj,
                pixel_stats=pixel_stats,
                object_stats=object_stats,
                child_objects=list(obj.children)
            ))

    return results


# =============================================================================
# Spatial Region Helpers
# =============================================================================

def create_region_mask_top_half(grid_shape: Tuple[int, int]) -> np.ndarray:
    """Create mask for top half of grid."""
    H, W = grid_shape
    mask = np.zeros((H, W), dtype=bool)
    mask[:H//2, :] = True
    return mask


def create_region_mask_bottom_half(grid_shape: Tuple[int, int]) -> np.ndarray:
    """Create mask for bottom half of grid."""
    H, W = grid_shape
    mask = np.zeros((H, W), dtype=bool)
    mask[H//2:, :] = True
    return mask


def create_region_mask_left_half(grid_shape: Tuple[int, int]) -> np.ndarray:
    """Create mask for left half of grid."""
    H, W = grid_shape
    mask = np.zeros((H, W), dtype=bool)
    mask[:, :W//2] = True
    return mask


def create_region_mask_right_half(grid_shape: Tuple[int, int]) -> np.ndarray:
    """Create mask for right half of grid."""
    H, W = grid_shape
    mask = np.zeros((H, W), dtype=bool)
    mask[:, W//2:] = True
    return mask


def create_region_mask_top_n_rows(grid_shape: Tuple[int, int], n: int) -> np.ndarray:
    """Create mask for top N rows."""
    H, W = grid_shape
    mask = np.zeros((H, W), dtype=bool)
    mask[:min(n, H), :] = True
    return mask


def create_region_mask_bottom_n_rows(grid_shape: Tuple[int, int], n: int) -> np.ndarray:
    """Create mask for bottom N rows."""
    H, W = grid_shape
    mask = np.zeros((H, W), dtype=bool)
    mask[max(0, H-n):, :] = True
    return mask


def create_region_mask_left_n_cols(grid_shape: Tuple[int, int], n: int) -> np.ndarray:
    """Create mask for left N columns."""
    H, W = grid_shape
    mask = np.zeros((H, W), dtype=bool)
    mask[:, :min(n, W)] = True
    return mask


def create_region_mask_right_n_cols(grid_shape: Tuple[int, int], n: int) -> np.ndarray:
    """Create mask for right N columns."""
    H, W = grid_shape
    mask = np.zeros((H, W), dtype=bool)
    mask[:, max(0, W-n):] = True
    return mask


def create_region_mask_from_bbox(
    grid_shape: Tuple[int, int],
    min_row: int,
    min_col: int,
    max_row: int,
    max_col: int
) -> np.ndarray:
    """Create mask from bounding box coordinates."""
    H, W = grid_shape
    mask = np.zeros((H, W), dtype=bool)
    r1 = max(0, min_row)
    c1 = max(0, min_col)
    r2 = min(H, max_row + 1)
    c2 = min(W, max_col + 1)
    mask[r1:r2, c1:c2] = True
    return mask


# =============================================================================
# Convenience Functions
# =============================================================================

def mode_color_of_region_by_name(
    grid: np.ndarray,
    region_name: str
) -> int:
    """
    Get mode color for a named region.

    Args:
        grid: (H, W) color grid
        region_name: One of 'top_half', 'bottom_half', 'left_half', 'right_half',
                     'full_grid'

    Returns:
        Mode color of the specified region
    """
    H, W = grid.shape

    region_creators = {
        'top_half': lambda: create_region_mask_top_half((H, W)),
        'bottom_half': lambda: create_region_mask_bottom_half((H, W)),
        'left_half': lambda: create_region_mask_left_half((H, W)),
        'right_half': lambda: create_region_mask_right_half((H, W)),
        'full_grid': lambda: None,
    }

    if region_name not in region_creators:
        raise ValueError(f"Unknown region name: {region_name}. "
                        f"Valid names: {list(region_creators.keys())}")

    mask = region_creators[region_name]()
    return mode_color(grid, region_mask=mask)


def colors_not_in_grid(grid: np.ndarray) -> List[int]:
    """
    Find colors (0-9) that are not present in the grid.

    Args:
        grid: (H, W) color grid

    Returns:
        Sorted list of colors not present in the grid
    """
    present = set(np.unique(grid).tolist())
    all_colors = set(range(10))
    return sorted(all_colors - present)


def colors_in_grid(grid: np.ndarray) -> List[int]:
    """
    Find colors that are present in the grid.

    Args:
        grid: (H, W) color grid

    Returns:
        Sorted list of colors present in the grid
    """
    return sorted(set(np.unique(grid).tolist()))


# =============================================================================
# Testing
# =============================================================================

if __name__ == "__main__":
    # Simple test
    grid = np.array([
        [0, 0, 1, 1, 2],
        [0, 0, 1, 1, 2],
        [3, 3, 4, 4, 5],
        [3, 3, 4, 4, 5],
    ])

    print("Test grid:")
    print(grid)
    print()

    # Full grid stats
    stats = compute_region_stats(grid)
    print(f"Full grid mode color: {stats.mode_color}")
    print(f"Color counts: {stats.color_counts}")
    print()

    # Top half
    top_mode = mode_color_of_region_by_name(grid, 'top_half')
    print(f"Top half mode color: {top_mode}")

    # Bottom half
    bottom_mode = mode_color_of_region_by_name(grid, 'bottom_half')
    print(f"Bottom half mode color: {bottom_mode}")

    # Colors not in grid
    absent = colors_not_in_grid(grid)
    print(f"Colors not in grid: {absent}")
