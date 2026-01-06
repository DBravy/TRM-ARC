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

Usage:
    from object_module import (
        Object,
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
from scipy import ndimage


# =============================================================================
# Constants
# =============================================================================

NUM_COLORS = 10
MAX_OBJECTS = 40

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
    """
    id: int
    row: int          # top-left row
    col: int          # top-left col
    height: int
    width: int
    color: int
    pixels: Set[Tuple[int, int]] = field(default_factory=set)
    is_background: bool = False

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


# =============================================================================
# Background Detection (from AffinitySlotAttention)
# =============================================================================

def detect_background_color(grid: np.ndarray, min_bg_coverage: float = 0.1) -> Optional[int]:
    """
    Detect the background color using the largest edge-connected component heuristic.

    This matches the approach in AffinitySlotAttention._detect_background_color.

    The background is identified as the color with the largest connected component
    that touches any edge of the grid, provided it covers at least min_bg_coverage
    of the total grid area.

    Args:
        grid: (H, W) integer color values 0-9
        min_bg_coverage: Minimum fraction of grid that must be covered to be
                        considered background (default 0.1 = 10%)

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
# Core Object Extraction Functions
# =============================================================================

def extract_connected_components(
    grid: np.ndarray,
    use_color_only: bool = False,
    background_color: Optional[int] = None,
    auto_detect_background: bool = True,
    min_bg_coverage: float = 0.1,
    skip_background: bool = True
) -> Tuple[np.ndarray, List[int], List[Tuple[int, int, int, int]], List[bool]]:
    """
    Extract connected components from a grid with background-aware detection.

    This is the canonical object detection function, matching the approach in
    AffinitySlotAttention._compute_connected_components.

    Args:
        grid: (H, W) integer color values 0-9
        use_color_only: If True, each color is one "object" (no connectivity check)
                       If False, use connectivity-based detection
        background_color: Explicitly specify background color. If None and
                         auto_detect_background is True, will detect automatically.
        auto_detect_background: Whether to automatically detect background color
                               using the edge-connected component heuristic.
        min_bg_coverage: Minimum fraction of grid for background detection.
        skip_background: If True, skip the detected background color entirely
                        (default True for backward compatibility).

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

    # Detect background color if requested
    if background_color is None and auto_detect_background:
        background_color = detect_background_color(grid, min_bg_coverage)

    # Compute exterior mask if we have a background color (for marking, not skipping)
    exterior_mask = None
    if background_color is not None and not skip_background:
        exterior_mask = compute_exterior_mask(grid, background_color)

    if use_color_only:
        # Each color is one "object" - no connectivity check
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

    # Connectivity-based detection
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
    auto_detect_background: bool = True
) -> List[Object]:
    """
    Extract connected components from a grid and return as Object instances.

    This is a convenience wrapper around extract_connected_components that
    returns Object instances instead of the (labels, colors, bboxes) format.

    Args:
        grid: (H, W) integer color values 0-9
        skip_background: If True, exclude background objects from results.
                        Default True for backward compatibility.
        auto_detect_background: Whether to detect background automatically.

    Returns:
        List of Object instances, one per connected component

    Example:
        >>> grid = np.array([[0, 1, 1], [0, 1, 0], [2, 2, 0]])
        >>> objects = extract_objects_from_grid(grid)
        >>> for obj in objects:
        ...     print(f"Object {obj.id}: color={obj.color}, area={obj.area}")
    """
    labels, colors, bboxes, is_background = extract_connected_components(
        grid,
        skip_background=skip_background,
        auto_detect_background=auto_detect_background
    )
    return labels_to_objects(labels, colors, bboxes, is_background)


def labels_to_objects(
    labels: np.ndarray,
    colors: List[int],
    bboxes: List[Tuple[int, int, int, int]],
    is_background: Optional[List[bool]] = None
) -> List[Object]:
    """
    Convert label-based representation to Object instances.

    Args:
        labels: (H, W) component IDs (0 = background, 1+ = component IDs)
        colors: List of colors for each component (indexed by component_id - 1)
        bboxes: List of (min_row, min_col, max_row, max_col) for each component
        is_background: Optional list of bool indicating if each component is background

    Returns:
        List of Object instances
    """
    if is_background is None:
        is_background = [False] * len(colors)

    objects = []
    for i, (color, bbox, is_bg) in enumerate(zip(colors, bboxes, is_background)):
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
            is_background=is_bg
        ))

    return objects


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


def _load_puzzle(puzzle_id: str, data_root: str = "kaggle/combined"):
    """Load a single puzzle from the ARC dataset."""
    import json
    import os

    subsets = ["training", "evaluation", "training2", "evaluation2"]

    for subset in subsets:
        challenges_path = f"{data_root}/arc-agi_{subset}_challenges.json"
        solutions_path = f"{data_root}/arc-agi_{subset}_solutions.json"

        if not os.path.exists(challenges_path):
            continue

        with open(challenges_path) as f:
            puzzles = json.load(f)

        if puzzle_id not in puzzles:
            continue

        puzzle = puzzles[puzzle_id]

        # Load solutions if available
        if os.path.exists(solutions_path):
            with open(solutions_path) as f:
                solutions = json.load(f)
            if puzzle_id in solutions:
                for i, sol in enumerate(solutions[puzzle_id]):
                    if i < len(puzzle["test"]):
                        puzzle["test"][i]["output"] = sol

        return puzzle

    raise ValueError(f"Puzzle '{puzzle_id}' not found in dataset")


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


def _draw_segmentation(ax, grid: np.ndarray, objects: List[Object], title: str = ""):
    """Draw object segmentation overlay on grid."""
    import matplotlib.patches as mpatches

    H, W = grid.shape

    # Create RGB image (dimmed original)
    rgb_image = np.zeros((H, W, 3), dtype=np.float32)
    for c in range(10):
        color = np.array([int(ARC_COLORS[c][i:i+2], 16) / 255.0 for i in (1, 3, 5)])
        mask = (grid == c)
        rgb_image[mask] = color * 0.5  # Dimmed background

    # Overlay object colors
    for obj in objects:
        obj_color_hex = OBJECT_COLORS[obj.id % len(OBJECT_COLORS)]
        obj_color = np.array([int(obj_color_hex[i:i+2], 16) / 255.0 for i in (1, 3, 5)])
        for (r, c) in obj.pixels:
            rgb_image[r, c] = obj_color

    ax.imshow(rgb_image, interpolation='nearest')

    # Draw bounding boxes and labels
    for obj in objects:
        obj_color_hex = OBJECT_COLORS[obj.id % len(OBJECT_COLORS)]
        rect = mpatches.Rectangle(
            (obj.col - 0.5, obj.row - 0.5),
            obj.width, obj.height,
            linewidth=2, edgecolor=obj_color_hex, facecolor='none'
        )
        ax.add_patch(rect)

        # Add object ID and color label
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
        example_type = data['type']
        example_num = data['num']

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
        _draw_segmentation(ax_in_seg, input_grid, input_objects,
                          f"Input Segmentation ({len(input_objects)} objects)")

        # Bottom-right: Output segmentation
        ax_out_seg = self.fig.add_subplot(gs[1, 1])
        if output_grid is not None and output_objects:
            _draw_segmentation(ax_out_seg, output_grid, output_objects,
                              f"Output Segmentation ({len(output_objects)} objects)")
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


def visualize_puzzle_objects(puzzle_id: str, data_root: str = "kaggle/combined"):
    """
    Visualize object segmentation for a specific puzzle.

    Shows a scrollable view of all training and test examples with their
    object segmentations.

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

        # Extract objects
        input_objects = extract_objects_from_grid(input_grid)
        output_objects = extract_objects_from_grid(output_grid) if output_grid is not None else []

        print(f"  Input objects: {len(input_objects)}")
        for obj in input_objects:
            print(f"    Object {obj.id}: color={obj.color}, size={obj.area}, "
                  f"bbox=({obj.row},{obj.col})-({obj.row+obj.height-1},{obj.col+obj.width-1})")

        if output_objects:
            print(f"  Output objects: {len(output_objects)}")
            for obj in output_objects:
                print(f"    Object {obj.id}: color={obj.color}, size={obj.area}, "
                      f"bbox=({obj.row},{obj.col})-({obj.row+obj.height-1},{obj.col+obj.width-1})")

        examples_data.append({
            'input_grid': input_grid,
            'output_grid': output_grid,
            'input_objects': input_objects,
            'output_objects': output_objects,
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

        # Extract objects
        input_objects = extract_objects_from_grid(input_grid)
        output_objects = extract_objects_from_grid(output_grid) if output_grid is not None else []

        print(f"  Input objects: {len(input_objects)}")
        for obj in input_objects:
            print(f"    Object {obj.id}: color={obj.color}, size={obj.area}, "
                  f"bbox=({obj.row},{obj.col})-({obj.row+obj.height-1},{obj.col+obj.width-1})")

        if output_objects:
            print(f"  Output objects: {len(output_objects)}")

        examples_data.append({
            'input_grid': input_grid,
            'output_grid': output_grid,
            'input_objects': input_objects,
            'output_objects': output_objects,
            'type': 'Test',
            'num': i + 1,
        })

    # Show interactive visualization
    print("\nGenerating visualization...")
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
