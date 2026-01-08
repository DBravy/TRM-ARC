#!/usr/bin/env python3
"""
Sequential Binding Module for ARC Puzzle Solver

This module unifies multi-head correspondence and procedural execution into a
single framework based on ORDERED CORRESPONDENCE with feature derivation.

Core Insight:
    Many ARC puzzles establish correspondence between input and output objects
    through ORDERING rather than feature similarity. Once ordered correspondence
    is established, each output feature (color, shape, position) can be derived
    from potentially different input sources.

Key Concepts:
    - InputPartition: How to segment the input (by divider, regions, object type)
    - OrderedBinding: Correspondence established by spatial ordering
    - FeatureDerivation: How each output feature derives from input
    - PatternVocabulary: Learned mapping from pattern shapes to parameters

Supported Puzzle Types:
    1. Procedural (136b0064): Patterns encode drawing instructions, cumulative position
    2. Pattern-to-fill (17cae0c1): Pattern shapes map to fill colors
    3. Template replication (12997ef3): Template shape + sequential colors

Usage:
    from sequential_binding_module import (
        discover_sequential_rule,
        apply_sequential_rule,
        SequentialRule,
    )

    # Discover rule from training examples
    rule = discover_sequential_rule(puzzle, verbose=True)

    # Apply to test input
    output_grid = apply_sequential_rule(test_input, rule)
"""

import numpy as np
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, FrozenSet, List, Optional, Set, Tuple, Any, Callable
from collections import Counter

from object_module import (
    Object,
    extract_objects_from_grid,
    SegmentationMode,
)


# =============================================================================
# Enums for Configuration
# =============================================================================

class PartitionMode(Enum):
    """How to partition the input grid."""
    NONE = auto()           # No partition - whole grid
    VERTICAL_DIVIDER = auto()  # Split by vertical line
    HORIZONTAL_DIVIDER = auto()  # Split by horizontal line
    FIXED_REGIONS = auto()   # Fixed-size regions (e.g., 3x3)
    OBJECT_TYPE = auto()     # By object properties (size, color)


class OrderingMode(Enum):
    """How to order objects for correspondence."""
    SPATIAL_ROW_MAJOR = auto()    # Top-to-bottom, left-to-right
    SPATIAL_COL_MAJOR = auto()    # Left-to-right, top-to-bottom
    COLUMN_THEN_ROW = auto()      # Left column top-bottom, then right column
    BY_SIZE = auto()              # Largest to smallest
    BY_COLOR = auto()             # By color value


class ColorDerivation(Enum):
    """How output color is derived."""
    PRESERVE = auto()        # Same as correspondent
    FROM_PATTERN = auto()    # Lookup in pattern vocabulary
    FROM_SEQUENCE = auto()   # From separate color source in order
    CONSTANT = auto()        # Fixed color


class ShapeDerivation(Enum):
    """How output shape is derived."""
    PRESERVE = auto()        # Same as correspondent
    FROM_PATTERN = auto()    # Pattern encodes shape parameters
    FROM_TEMPLATE = auto()   # From a template object
    FILL_REGION = auto()     # Fill a region solid


class PositionDerivation(Enum):
    """How output position is derived."""
    PRESERVE = auto()        # Same as correspondent
    SEQUENTIAL = auto()      # Evenly spaced sequence
    CUMULATIVE = auto()      # Each position depends on previous
    IN_PLACE = auto()        # Replace correspondent position


class Direction(Enum):
    """Cardinal directions for drawing."""
    UP = auto()
    DOWN = auto()
    LEFT = auto()
    RIGHT = auto()

    def delta(self) -> Tuple[int, int]:
        """Get (row_delta, col_delta) for this direction."""
        deltas = {
            Direction.UP: (-1, 0),
            Direction.DOWN: (1, 0),
            Direction.LEFT: (0, -1),
            Direction.RIGHT: (0, 1),
        }
        return deltas.get(self, (0, 0))


# =============================================================================
# Pattern Vocabulary
# =============================================================================

@dataclass
class PatternEntry:
    """An entry in the pattern vocabulary."""
    pattern: FrozenSet[Tuple[int, int]]  # Normalized pixel positions
    direction: Optional[Direction] = None
    length: Optional[int] = None
    color: Optional[int] = None

    def describe(self) -> str:
        parts = []
        if self.direction:
            parts.append(f"dir={self.direction.name}")
        if self.length:
            parts.append(f"len={self.length}")
        if self.color is not None:
            parts.append(f"color={self.color}")
        return f"Pattern({', '.join(parts)})"


class PatternVocabulary:
    """Learned mapping from pattern shapes to parameters.

    This replaces hardcoded BASE_PATTERNS with a learnable vocabulary
    discovered from training examples.
    """

    def __init__(self):
        self.entries: Dict[FrozenSet[Tuple[int, int]], PatternEntry] = {}

    def add(self, pattern: FrozenSet[Tuple[int, int]], **kwargs) -> None:
        """Add a pattern with its parameters."""
        self.entries[pattern] = PatternEntry(pattern=pattern, **kwargs)

    def lookup(self, pattern: FrozenSet[Tuple[int, int]]) -> Optional[PatternEntry]:
        """Look up parameters for a pattern."""
        return self.entries.get(pattern)

    def normalize_pixels(self, pixels: Set[Tuple[int, int]]) -> FrozenSet[Tuple[int, int]]:
        """Normalize pixel positions to origin."""
        if not pixels:
            return frozenset()
        min_r = min(r for r, c in pixels)
        min_c = min(c for r, c in pixels)
        return frozenset((r - min_r, c - min_c) for r, c in pixels)

    def describe(self) -> str:
        lines = ["PatternVocabulary:"]
        for pattern, entry in self.entries.items():
            lines.append(f"  {entry.describe()}")
        return "\n".join(lines)


# Note: Pattern vocabulary is now learned dynamically from training examples
# via discover_pattern_vocabulary_procedural() rather than hardcoded.


# =============================================================================
# Input Partition
# =============================================================================

@dataclass
class InputPartition:
    """Specification for how to partition the input."""
    mode: PartitionMode
    divider_col: Optional[int] = None
    divider_row: Optional[int] = None
    divider_color: Optional[int] = None
    region_size: Optional[Tuple[int, int]] = None
    template_criterion: Optional[str] = None  # "largest", "color_1", etc.


def find_vertical_divider(grid: np.ndarray) -> Tuple[int, int]:
    """Find vertical divider column and color."""
    H, W = grid.shape
    for c in range(W):
        col = grid[:, c]
        unique = set(col)
        if len(unique) == 1 and 0 not in unique:
            return c, int(col[0])
    return W // 2, 0


def partition_grid(
    grid: np.ndarray,
    partition: InputPartition
) -> Tuple[np.ndarray, Optional[np.ndarray], List[Object]]:
    """Partition grid according to specification.

    Returns:
        (instruction_region, canvas_region, instruction_objects)
    """
    if partition.mode == PartitionMode.NONE:
        objects = extract_objects_from_grid(grid, segmentation_mode=SegmentationMode.CONNECTIVITY)
        return grid, None, objects

    elif partition.mode == PartitionMode.VERTICAL_DIVIDER:
        div_col = partition.divider_col
        if div_col is None:
            div_col, _ = find_vertical_divider(grid)

        instruction_region = grid[:, :div_col]
        canvas_region = grid[:, div_col+1:]

        objects = extract_objects_from_grid(
            instruction_region,
            segmentation_mode=SegmentationMode.CONNECTIVITY
        )
        return instruction_region, canvas_region, objects

    elif partition.mode == PartitionMode.FIXED_REGIONS:
        # For pattern-to-fill puzzles like 17cae0c1
        objects = extract_objects_from_grid(grid, segmentation_mode=SegmentationMode.CONNECTIVITY)
        return grid, None, objects

    elif partition.mode == PartitionMode.OBJECT_TYPE:
        # Split objects by type (template vs color sources)
        objects = extract_objects_from_grid(grid, segmentation_mode=SegmentationMode.CONNECTIVITY)
        return grid, None, objects

    return grid, None, []


# =============================================================================
# Ordered Binding
# =============================================================================

@dataclass
class OrderedBinding:
    """Specification for ordered correspondence."""
    ordering_mode: OrderingMode
    color_derivation: ColorDerivation
    shape_derivation: ShapeDerivation
    position_derivation: PositionDerivation

    # For factored correspondence (like 12997ef3)
    color_source: str = "correspondent"  # "correspondent", "template", "sequence"
    shape_source: str = "correspondent"  # "correspondent", "template"

    # For cumulative positioning
    start_position: Optional[Tuple[int, int]] = None
    implicit_step: Optional[Direction] = None  # Step before each instruction


def order_objects(
    objects: List[Object],
    mode: OrderingMode,
    region_width: Optional[int] = None
) -> List[Object]:
    """Order objects according to the specified mode."""
    if not objects:
        return []

    if mode == OrderingMode.SPATIAL_ROW_MAJOR:
        return sorted(objects, key=lambda o: (o.row, o.col))

    elif mode == OrderingMode.SPATIAL_COL_MAJOR:
        return sorted(objects, key=lambda o: (o.col, o.row))

    elif mode == OrderingMode.COLUMN_THEN_ROW:
        # Split into left and right halves, order each by row
        if region_width is None:
            region_width = max(o.col + o.width for o in objects) if objects else 1
        mid = region_width // 2

        left = sorted([o for o in objects if o.col < mid], key=lambda o: o.row)
        right = sorted([o for o in objects if o.col >= mid], key=lambda o: o.row)
        return left + right

    elif mode == OrderingMode.BY_SIZE:
        return sorted(objects, key=lambda o: len(o.pixels), reverse=True)

    elif mode == OrderingMode.BY_COLOR:
        return sorted(objects, key=lambda o: o.color)

    return objects


# =============================================================================
# Feature Derivation Helpers
# =============================================================================

def derive_color(
    obj: Object,
    binding: OrderedBinding,
    pattern_vocab: PatternVocabulary,
    color_sources: Optional[List[Object]] = None,
    index: int = 0
) -> int:
    """Derive output color based on binding specification."""
    if binding.color_derivation == ColorDerivation.PRESERVE:
        return obj.color

    elif binding.color_derivation == ColorDerivation.FROM_PATTERN:
        # Lookup pattern in vocabulary
        normalized = pattern_vocab.normalize_pixels(obj.pixels)
        entry = pattern_vocab.lookup(normalized)
        if entry and entry.color is not None:
            return entry.color
        return obj.color

    elif binding.color_derivation == ColorDerivation.FROM_SEQUENCE:
        if color_sources and index < len(color_sources):
            return color_sources[index].color
        return obj.color

    return obj.color


def derive_shape_params(
    obj: Object,
    binding: OrderedBinding,
    pattern_vocab: PatternVocabulary,
    grid: Optional[np.ndarray] = None
) -> Tuple[Optional[Direction], Optional[int]]:
    """Derive shape parameters (direction, length) from pattern.

    Args:
        obj: The object to derive parameters for
        binding: The binding specification
        pattern_vocab: The learned pattern vocabulary
        grid: Optional grid for filtering pixels by color (handles bounding box issues)
    """
    if binding.shape_derivation != ShapeDerivation.FROM_PATTERN:
        return None, None

    # Normalize pixels, filtering by color if grid is provided
    normalized = normalize_object_shape(obj, grid)
    entry = pattern_vocab.lookup(normalized)

    if entry:
        return entry.direction, entry.length

    # Fallback: infer from shape
    return infer_direction_from_shape(normalized if normalized else obj.pixels)


def infer_direction_from_shape(pixels: Set[Tuple[int, int]]) -> Tuple[Direction, int]:
    """Heuristic direction inference when pattern not in vocabulary.

    Delegates to infer_direction_length_from_shape for the full analysis.
    """
    # Normalize pixels first
    if not pixels:
        return Direction.RIGHT, 1

    min_r = min(r for r, c in pixels)
    min_c = min(c for r, c in pixels)
    normalized = frozenset((r - min_r, c - min_c) for r, c in pixels)

    return infer_direction_length_from_shape(normalized)


# =============================================================================
# Sequential Rule
# =============================================================================

@dataclass
class SequentialRule:
    """Complete rule for sequential binding transformation."""
    partition: InputPartition
    binding: OrderedBinding
    pattern_vocab: PatternVocabulary

    # Output specification
    output_shape: Optional[Tuple[int, int]] = None
    background_color: int = 0

    # For template-based puzzles
    template_pixels: Optional[FrozenSet[Tuple[int, int]]] = None

    # For pattern-to-fill puzzles
    fill_regions: Optional[List[Tuple[int, int, int, int]]] = None  # (r, c, h, w)

    # Metadata
    variance: float = float('inf')

    def describe(self) -> str:
        return (f"SequentialRule(\n"
                f"  partition={self.partition.mode.name},\n"
                f"  ordering={self.binding.ordering_mode.name},\n"
                f"  color={self.binding.color_derivation.name},\n"
                f"  shape={self.binding.shape_derivation.name},\n"
                f"  position={self.binding.position_derivation.name}\n"
                f")")


# =============================================================================
# Execution Engine
# =============================================================================

def execute_procedural(
    instruction_objects: List[Object],
    rule: SequentialRule,
    output_shape: Tuple[int, int],
    start_pos: Tuple[int, int],
    instruction_grid: Optional[np.ndarray] = None,
    start_color: int = 5
) -> np.ndarray:
    """Execute procedural drawing instructions.

    Args:
        instruction_objects: Ordered list of instruction pattern objects
        rule: The sequential rule with pattern vocabulary
        output_shape: Shape of the output grid
        start_pos: Starting position (row, col)
        instruction_grid: Optional grid for filtering pixels by color during shape lookup
        start_color: Color for the start marker (default grey=5)
    """
    grid = np.zeros(output_shape, dtype=np.int64)

    row, col = start_pos

    # Mark start position
    if 0 <= row < output_shape[0] and 0 <= col < output_shape[1]:
        grid[row, col] = start_color

    # Execute each instruction
    for obj in instruction_objects:
        # Get direction and length from pattern
        direction, length = derive_shape_params(obj, rule.binding, rule.pattern_vocab, instruction_grid)
        if direction is None:
            continue

        color = derive_color(obj, rule.binding, rule.pattern_vocab)

        # Implicit step (usually DOWN) before drawing
        if rule.binding.implicit_step:
            dr, dc = rule.binding.implicit_step.delta()
            row, col = row + dr, col + dc

        # Draw at current position
        if 0 <= row < output_shape[0] and 0 <= col < output_shape[1]:
            grid[row, col] = color

        # Draw remaining length-1 pixels in direction
        dr, dc = direction.delta()
        for _ in range(length - 1):
            row, col = row + dr, col + dc
            if 0 <= row < output_shape[0] and 0 <= col < output_shape[1]:
                grid[row, col] = color

    return grid


def execute_template_replication(
    color_sources: List[Object],
    template_pixels: FrozenSet[Tuple[int, int]],
    rule: SequentialRule,
    output_shape: Tuple[int, int]
) -> np.ndarray:
    """Execute template replication with sequential colors."""
    grid = np.zeros(output_shape, dtype=np.int64)

    # Get template dimensions
    if not template_pixels:
        return grid

    rows = [r for r, c in template_pixels]
    cols = [c for r, c in template_pixels]
    template_height = max(rows) - min(rows) + 1
    template_width = max(cols) - min(cols) + 1

    # Normalize template to origin
    min_r, min_c = min(rows), min(cols)
    normalized_template = [(r - min_r, c - min_c) for r, c in template_pixels]

    # Determine arrangement from color source positions
    n = len(color_sources)
    if n >= 2:
        row_span = max(o.row for o in color_sources) - min(o.row for o in color_sources)
        col_span = max(o.col for o in color_sources) - min(o.col for o in color_sources)
        horizontal = col_span >= row_span
    else:
        horizontal = True

    # Place copies
    for i, color_obj in enumerate(color_sources):
        if horizontal:
            base_row, base_col = 0, i * template_width
        else:
            base_row, base_col = i * template_height, 0

        for r, c in normalized_template:
            out_r, out_c = base_row + r, base_col + c
            if 0 <= out_r < output_shape[0] and 0 <= out_c < output_shape[1]:
                grid[out_r, out_c] = color_obj.color

    return grid


def execute_pattern_to_fill(
    input_grid: np.ndarray,
    rule: SequentialRule,
    output_shape: Tuple[int, int],
    verbose: bool = False
) -> np.ndarray:
    """Execute pattern-to-fill transformation (17cae0c1 style).

    Processes each 3x3 region independently, extracting the pattern
    and looking up its fill color in the vocabulary.
    """
    grid = np.zeros(output_shape, dtype=np.int64)
    H, W = input_grid.shape

    # Process each 3x3 region
    for region_r in range(0, H, 3):
        for region_c in range(0, W, 3):
            # Extract pattern from this region
            region = input_grid[region_r:min(region_r+3, H), region_c:min(region_c+3, W)]

            # Get non-zero pixel positions within the region
            pattern_pixels = set()
            for r in range(region.shape[0]):
                for c in range(region.shape[1]):
                    if region[r, c] > 0:
                        pattern_pixels.add((r, c))

            if not pattern_pixels:
                continue

            # Look up pattern in vocabulary
            pattern = frozenset(pattern_pixels)
            entry = rule.pattern_vocab.lookup(pattern)

            if entry and entry.color is not None:
                fill_color = entry.color
                if verbose:
                    print(f"Region ({region_r},{region_c}): pattern found -> color {fill_color}")
            else:
                # Pattern not found - this shouldn't happen if vocabulary was learned correctly
                if verbose:
                    print(f"Region ({region_r},{region_c}): pattern {pattern} NOT found in vocabulary")
                continue

            # Fill the region
            for r in range(region_r, min(region_r + 3, output_shape[0])):
                for c in range(region_c, min(region_c + 3, output_shape[1])):
                    grid[r, c] = fill_color

    return grid


# =============================================================================
# Discovery Functions
# =============================================================================

def discover_partition(
    examples: List[dict],
    verbose: bool = False
) -> InputPartition:
    """Discover how to partition input grids."""
    if not examples:
        return InputPartition(mode=PartitionMode.NONE)

    first_input = np.array(examples[0]['input'])
    first_output = np.array(examples[0]['output'])
    H, W = first_input.shape

    # FIRST: Check for fixed regions (like 17cae0c1 - 3x3 regions)
    # This should come before divider check to avoid false positives
    if first_input.shape == first_output.shape:
        # Check if output has uniform 3x3 regions
        if H % 3 == 0 and W % 3 == 0:
            is_region_based = True
            for r in range(0, H, 3):
                for c in range(0, W, 3):
                    region = first_output[r:r+3, c:c+3]
                    if len(set(region.flatten())) > 1:
                        is_region_based = False
                        break
                if not is_region_based:
                    break
            if is_region_based:
                if verbose:
                    print("Found 3x3 region-based structure")
                return InputPartition(
                    mode=PartitionMode.FIXED_REGIONS,
                    region_size=(3, 3)
                )

    # SECOND: Check for vertical divider
    # Must be: not at edge, single color, spans full height
    for c in range(1, W - 1):  # Exclude edges
        col = first_input[:, c]
        unique = set(col)
        if len(unique) == 1 and 0 not in unique:
            color = int(col[0])
            # Additional check: there should be content on both sides
            left_has_content = np.any(first_input[:, :c] > 0)
            right_has_content = np.any(first_input[:, c+1:] > 0)
            if left_has_content and right_has_content:
                if verbose:
                    print(f"Found vertical divider at column {c}, color {color}")
                return InputPartition(
                    mode=PartitionMode.VERTICAL_DIVIDER,
                    divider_col=c,
                    divider_color=color
                )

    # THIRD: Check for template + color pixels pattern (12997ef3)
    objects = extract_objects_from_grid(first_input, segmentation_mode=SegmentationMode.CONNECTIVITY)
    objects = [o for o in objects if not o.is_background and o.color > 0]

    if len(objects) >= 2:
        sizes = [len(o.pixels) for o in objects]
        if max(sizes) > 3 * min(sizes):  # One object much larger than others
            if verbose:
                print("Found template + color pixels structure")
            return InputPartition(
                mode=PartitionMode.OBJECT_TYPE,
                template_criterion="largest"
            )

    return InputPartition(mode=PartitionMode.NONE)


def extract_output_segments(
    output_grid: np.ndarray,
    start_marker_color: int = 5,
    verbose: bool = False
) -> List[Tuple[int, Direction, int]]:
    """Extract drawn segments from output by tracing from start marker.

    The output follows a cumulative drawing model:
    1. Start at the marker pixel
    2. Move DOWN (implicit step before each instruction)
    3. Draw in some direction for some length
    4. Repeat

    Returns:
        List of (color, direction, length) tuples for each segment
    """
    H, W = output_grid.shape
    segments = []

    # Find start marker
    start_pos = None
    for r in range(H):
        for c in range(W):
            if output_grid[r, c] == start_marker_color:
                start_pos = (r, c)
                break
        if start_pos:
            break

    if start_pos is None:
        return segments

    # Track visited pixels to avoid re-tracing
    visited = set()
    visited.add(start_pos)

    row, col = start_pos

    # Trace through the output following the drawing model
    while True:
        # Implicit DOWN step
        next_row = row + 1
        if next_row >= H:
            break

        # Look for a colored pixel at or near the next position
        # The segment starts at (next_row, col) or nearby
        segment_start = None
        segment_color = None

        # Check current column first, then adjacent
        for dc in [0, -1, 1, -2, 2, -3, 3]:
            check_col = col + dc
            if 0 <= check_col < W:
                if output_grid[next_row, check_col] > 0 and output_grid[next_row, check_col] != start_marker_color:
                    if (next_row, check_col) not in visited:
                        segment_start = (next_row, check_col)
                        segment_color = output_grid[next_row, check_col]
                        break

        if segment_start is None:
            break

        # Trace the segment to find direction and length
        seg_row, seg_col = segment_start
        color = segment_color

        # Find all connected pixels of this color in cardinal directions
        segment_pixels = [(seg_row, seg_col)]
        visited.add((seg_row, seg_col))

        # Try each direction and find which one has more pixels
        directions_found = {}
        for direction in [Direction.RIGHT, Direction.LEFT, Direction.UP, Direction.DOWN]:
            dr, dc = direction.delta()
            length = 1
            r, c = seg_row + dr, seg_col + dc
            while 0 <= r < H and 0 <= c < W and output_grid[r, c] == color and (r, c) not in visited:
                length += 1
                segment_pixels.append((r, c))
                visited.add((r, c))
                r, c = r + dr, c + dc
            if length > 1:
                directions_found[direction] = length

        # Determine the primary direction (the one with most extension)
        if directions_found:
            primary_dir = max(directions_found.keys(), key=lambda d: directions_found[d])
            total_length = directions_found[primary_dir]
        else:
            # Single pixel - need to infer direction from position relative to previous
            primary_dir = Direction.RIGHT  # Default
            total_length = 1

        segments.append((color, primary_dir, total_length))

        if verbose:
            print(f"  Segment: color={color}, dir={primary_dir.name}, len={total_length}")

        # Update position to end of this segment
        dr, dc = primary_dir.delta()
        row = seg_row + dr * (total_length - 1)
        col = seg_col + dc * (total_length - 1)

    return segments


def find_instruction_objects(
    grid: np.ndarray,
    divider_col: int,
    verbose: bool = False
) -> List[Object]:
    """Find instruction objects in the instruction region using connected components.

    This is the correct approach for finding complete patterns - we use
    connected component analysis to find all pixels of each object,
    rather than scanning 3x3 windows which may find partial patterns.

    Returns:
        List of Object instances representing instruction patterns
    """
    instruction_region = grid[:, :divider_col]

    # Extract objects using connected component analysis
    objects = extract_objects_from_grid(
        instruction_region,
        segmentation_mode=SegmentationMode.CONNECTIVITY
    )

    # Filter to non-background objects
    objects = [o for o in objects if not o.is_background and o.color > 0]

    if verbose:
        for obj in objects:
            print(f"  Found object at ({obj.row},{obj.col}): {len(obj.pixels)} pixels, color {obj.color}")

    return objects


def normalize_object_shape(
    obj: Object,
    grid: Optional[np.ndarray] = None
) -> FrozenSet[Tuple[int, int]]:
    """Normalize an object's pixels to origin (0,0) for shape comparison.

    Args:
        obj: The object to normalize
        grid: Optional grid to validate pixel colors. If provided, only pixels
              that actually match the object's color are included. This handles
              cases where object.pixels may include bounding box regions.
    """
    if not obj.pixels:
        return frozenset()

    # If grid is provided, filter pixels to only those matching obj.color
    if grid is not None:
        actual_pixels = {
            (r, c) for r, c in obj.pixels
            if 0 <= r < grid.shape[0] and 0 <= c < grid.shape[1]
            and grid[r, c] == obj.color
        }
    else:
        actual_pixels = obj.pixels

    if not actual_pixels:
        return frozenset()

    min_r = min(r for r, c in actual_pixels)
    min_c = min(c for r, c in actual_pixels)
    return frozenset((r - min_r, c - min_c) for r, c in actual_pixels)


def find_instruction_patterns_for_learning(
    grid: np.ndarray,
    divider_col: int,
    verbose: bool = False
) -> List[Tuple[FrozenSet[Tuple[int, int]], int, int, int]]:
    """Find distinct instruction patterns in instruction region.

    Uses connected component analysis to find complete objects,
    then normalizes their shapes for vocabulary learning.

    Returns:
        List of (normalized_pattern, color, row, col) tuples
    """
    instruction_region = grid[:, :divider_col]
    objects = find_instruction_objects(grid, divider_col, verbose=False)

    patterns = []
    for obj in objects:
        # Pass the instruction region to filter out zero-valued pixels
        normalized = normalize_object_shape(obj, instruction_region)
        patterns.append((normalized, obj.color, obj.row, obj.col))

        if verbose:
            print(f"  Found pattern at ({obj.row},{obj.col}): {len(normalized)} pixels, color {obj.color}")

    return patterns


def infer_direction_length_from_shape(pattern: FrozenSet[Tuple[int, int]]) -> Tuple[Direction, int]:
    """Infer direction and length from pattern shape characteristics.

    Analyzes the geometric properties of the pattern to determine
    what direction and length it encodes.
    """
    if not pattern:
        return Direction.RIGHT, 1

    rows = [r for r, c in pattern]
    cols = [c for r, c in pattern]

    min_r, max_r = min(rows), max(rows)
    min_c, max_c = min(cols), max(cols)
    height = max_r - min_r + 1
    width = max_c - min_c + 1

    # Count pixels in each position
    top_row = sum(1 for r, c in pattern if r == min_r)
    bottom_row = sum(1 for r, c in pattern if r == max_r)
    left_col = sum(1 for r, c in pattern if c == min_c)
    right_col = sum(1 for r, c in pattern if c == max_c)

    # Analyze asymmetry to determine direction
    # The "heavy" side is typically the base, direction points away from it

    vertical_asymmetry = bottom_row - top_row
    horizontal_asymmetry = left_col - right_col

    # Determine primary direction based on asymmetry
    if abs(horizontal_asymmetry) > abs(vertical_asymmetry):
        # Horizontal pattern
        if horizontal_asymmetry > 0:
            # Heavy on left -> points right
            direction = Direction.RIGHT
        else:
            # Heavy on right -> points left
            direction = Direction.LEFT
        length = width
    else:
        # Vertical pattern
        if vertical_asymmetry > 0:
            # Heavy on bottom -> points up
            direction = Direction.UP
        elif vertical_asymmetry < 0:
            # Heavy on top -> points down
            direction = Direction.DOWN
        else:
            # Symmetric - use other heuristics
            # Check for "arrow" shape by looking at pixel distribution
            center_r, center_c = (min_r + max_r) / 2, (min_c + max_c) / 2
            top_half = sum(1 for r, c in pattern if r < center_r)
            bottom_half = sum(1 for r, c in pattern if r > center_r)
            if top_half > bottom_half:
                direction = Direction.DOWN
            else:
                direction = Direction.UP
        length = height

    # Adjust length based on pixel count (more pixels often = longer)
    # Common patterns: 4-5 pixels = short (2-3), 6-7 pixels = medium (3-4)
    n_pixels = len(pattern)
    if n_pixels <= 4:
        length = 2
    elif n_pixels <= 5:
        length = 3
    elif n_pixels <= 6:
        length = 4
    else:
        length = 2  # Dense patterns often = short

    return direction, length


def discover_pattern_vocabulary_procedural(
    examples: List[dict],
    partition: InputPartition,
    verbose: bool = False
) -> PatternVocabulary:
    """Discover pattern vocabulary for procedural puzzles.

    Uses two-phase learning:
    1. Collect all unique patterns from training examples
    2. For each pattern, try correspondence-based learning first
    3. Fall back to shape-based inference if correspondence fails
    """
    vocab = PatternVocabulary()
    pattern_observations = {}  # pattern -> list of (direction, length) observations
    all_patterns = set()  # Collect all unique patterns

    div_col = partition.divider_col
    if div_col is None:
        return vocab

    # Phase 1: Collect patterns and try correspondence-based learning
    for ex_idx, ex in enumerate(examples):
        input_grid = np.array(ex['input'])
        output_grid = np.array(ex['output'])

        if verbose:
            print(f"\nExample {ex_idx + 1}:")

        # Find patterns in instruction region
        raw_patterns = find_instruction_patterns_for_learning(input_grid, div_col, verbose)

        # Collect all unique patterns
        for pattern, color, row, col in raw_patterns:
            all_patterns.add(pattern)

        # Order patterns by column-then-row
        mid_col = div_col // 2
        left_patterns = [(p, c, r, col) for p, c, r, col in raw_patterns if col < mid_col]
        right_patterns = [(p, c, r, col) for p, c, r, col in raw_patterns if col >= mid_col]
        left_patterns.sort(key=lambda x: x[2])
        right_patterns.sort(key=lambda x: x[2])
        ordered_patterns = left_patterns + right_patterns

        if verbose:
            print(f"  Ordered {len(ordered_patterns)} patterns")

        # Extract segments from output
        segments = extract_output_segments(output_grid, verbose=verbose)

        if verbose:
            print(f"  Found {len(segments)} output segments")

        # Try to match by color (more reliable than pure order)
        pattern_by_color = {}
        for pattern, color, row, col in ordered_patterns:
            if color not in pattern_by_color:
                pattern_by_color[color] = []
            pattern_by_color[color].append((pattern, row, col))

        segment_by_color = {}
        for seg_color, seg_dir, seg_len in segments:
            if seg_color not in segment_by_color:
                segment_by_color[seg_color] = []
            segment_by_color[seg_color].append((seg_dir, seg_len))

        # Match patterns to segments by color
        for color in pattern_by_color:
            if color not in segment_by_color:
                continue
            patterns_of_color = pattern_by_color[color]
            segments_of_color = segment_by_color[color]

            # If same count, match by order within color
            for i, (pattern, _, _) in enumerate(patterns_of_color):
                if i < len(segments_of_color):
                    seg_dir, seg_len = segments_of_color[i]
                    if pattern not in pattern_observations:
                        pattern_observations[pattern] = []
                    pattern_observations[pattern].append((seg_dir, seg_len))
                    if verbose:
                        print(f"  Learned: color {color} pattern -> {seg_dir.name}, len={seg_len}")

    # Phase 2: Build vocabulary
    for pattern in all_patterns:
        if pattern in pattern_observations and pattern_observations[pattern]:
            # Use correspondence-based learning
            counts = Counter(pattern_observations[pattern])
            direction, length = counts.most_common(1)[0][0]
            if verbose:
                print(f"Vocab (correspondence): {len(pattern)}px -> {direction.name}, len={length}")
        else:
            # Fall back to shape-based inference
            direction, length = infer_direction_length_from_shape(pattern)
            if verbose:
                print(f"Vocab (shape inference): {len(pattern)}px -> {direction.name}, len={length}")

        vocab.add(pattern, direction=direction, length=length)

    return vocab


def discover_pattern_vocabulary_fill(
    examples: List[dict],
    verbose: bool = False
) -> PatternVocabulary:
    """Discover pattern vocabulary for pattern-to-fill puzzles (17cae0c1)."""
    vocab = PatternVocabulary()

    for ex in examples:
        input_grid = np.array(ex['input'])
        output_grid = np.array(ex['output'])
        H, W = input_grid.shape

        # Process each 3x3 region
        for region_r in range(0, H, 3):
            for region_c in range(0, W, 3):
                # Extract input pattern in this region
                input_region = input_grid[region_r:region_r+3, region_c:region_c+3]
                output_region = output_grid[region_r:region_r+3, region_c:region_c+3]

                # Get pattern pixels (non-zero in input)
                pattern_pixels = set()
                for r in range(min(3, input_region.shape[0])):
                    for c in range(min(3, input_region.shape[1])):
                        if input_region[r, c] > 0:
                            pattern_pixels.add((r, c))

                if not pattern_pixels:
                    continue

                # Get fill color (should be uniform in output)
                fill_colors = set(output_region.flatten())
                if len(fill_colors) == 1:
                    fill_color = int(fill_colors.pop())
                else:
                    fill_color = int(Counter(output_region.flatten()).most_common(1)[0][0])

                # Add to vocabulary
                pattern = frozenset(pattern_pixels)
                if pattern not in vocab.entries:
                    vocab.add(pattern, color=fill_color)
                    if verbose:
                        print(f"Learned pattern {pattern} -> color {fill_color}")

    return vocab


def discover_ordering(
    examples: List[dict],
    partition: InputPartition,
    verbose: bool = False
) -> OrderingMode:
    """Discover the ordering mode for correspondence."""
    if partition.mode == PartitionMode.VERTICAL_DIVIDER:
        # Procedural puzzles typically use column-then-row
        return OrderingMode.COLUMN_THEN_ROW

    elif partition.mode == PartitionMode.FIXED_REGIONS:
        # Pattern-to-fill uses row-major
        return OrderingMode.SPATIAL_ROW_MAJOR

    elif partition.mode == PartitionMode.OBJECT_TYPE:
        # Template replication uses row-major for color sources
        return OrderingMode.SPATIAL_ROW_MAJOR

    return OrderingMode.SPATIAL_ROW_MAJOR


def discover_binding(
    examples: List[dict],
    partition: InputPartition,
    ordering: OrderingMode,
    verbose: bool = False
) -> OrderedBinding:
    """Discover the binding specification."""
    if partition.mode == PartitionMode.VERTICAL_DIVIDER:
        # Procedural: cumulative position, preserve color, pattern encodes shape
        return OrderedBinding(
            ordering_mode=ordering,
            color_derivation=ColorDerivation.PRESERVE,
            shape_derivation=ShapeDerivation.FROM_PATTERN,
            position_derivation=PositionDerivation.CUMULATIVE,
            implicit_step=Direction.DOWN
        )

    elif partition.mode == PartitionMode.FIXED_REGIONS:
        # Pattern-to-fill: in-place position, pattern encodes color, fill shape
        return OrderedBinding(
            ordering_mode=ordering,
            color_derivation=ColorDerivation.FROM_PATTERN,
            shape_derivation=ShapeDerivation.FILL_REGION,
            position_derivation=PositionDerivation.IN_PLACE
        )

    elif partition.mode == PartitionMode.OBJECT_TYPE:
        # Template replication: sequential position, sequential color, template shape
        return OrderedBinding(
            ordering_mode=ordering,
            color_derivation=ColorDerivation.FROM_SEQUENCE,
            shape_derivation=ShapeDerivation.FROM_TEMPLATE,
            position_derivation=PositionDerivation.SEQUENTIAL,
            color_source="sequence",
            shape_source="template"
        )

    return OrderedBinding(
        ordering_mode=ordering,
        color_derivation=ColorDerivation.PRESERVE,
        shape_derivation=ShapeDerivation.PRESERVE,
        position_derivation=PositionDerivation.PRESERVE
    )


def find_start_marker(
    grid: np.ndarray,
    divider_col: int,
    marker_color: int = 5
) -> Tuple[int, int]:
    """Find the start marker position on the right side of divider."""
    H, W = grid.shape
    for r in range(H):
        for c in range(divider_col + 1, W):
            if grid[r, c] == marker_color:
                return r, c - divider_col - 1  # Relative to output grid
    return 0, 0


def find_template_object(objects: List[Object]) -> Optional[Object]:
    """Find the template object (largest non-single-pixel object)."""
    candidates = [o for o in objects if len(o.pixels) > 1 and o.color > 0 and not o.is_background]
    if not candidates:
        return None

    # Prefer color=1 (common ARC pattern)
    for obj in candidates:
        if obj.color == 1:
            return obj

    return max(candidates, key=lambda o: len(o.pixels))


def find_color_sources(objects: List[Object], template: Object) -> List[Object]:
    """Find color source objects (small objects different from template)."""
    sources = []
    for obj in objects:
        if obj.color == template.color or obj.color <= 0 or obj.is_background:
            continue
        if len(obj.pixels) > len(template.pixels) * 0.5:
            continue
        sources.append(obj)
    return sorted(sources, key=lambda o: (o.row, o.col))


def discover_sequential_rule(
    puzzle: Dict,
    verbose: bool = False
) -> Optional[SequentialRule]:
    """Discover the sequential binding rule from training examples.

    This is the main entry point for rule discovery.
    """
    train_examples = puzzle.get('train', [])
    if not train_examples:
        return None

    if verbose:
        print("=" * 60)
        print("Discovering Sequential Binding Rule")
        print("=" * 60)

    # Step 1: Discover partition
    partition = discover_partition(train_examples, verbose)
    if verbose:
        print(f"\nPartition mode: {partition.mode.name}")

    # Step 2: Discover pattern vocabulary
    if partition.mode == PartitionMode.VERTICAL_DIVIDER:
        vocab = discover_pattern_vocabulary_procedural(train_examples, partition, verbose)
    elif partition.mode == PartitionMode.FIXED_REGIONS:
        vocab = discover_pattern_vocabulary_fill(train_examples, verbose)
    else:
        vocab = PatternVocabulary()

    if verbose and vocab.entries:
        print(f"\n{vocab.describe()}")

    # Step 3: Discover ordering
    ordering = discover_ordering(train_examples, partition, verbose)
    if verbose:
        print(f"\nOrdering mode: {ordering.name}")

    # Step 4: Discover binding
    binding = discover_binding(train_examples, partition, ordering, verbose)
    if verbose:
        print(f"\nBinding: color={binding.color_derivation.name}, "
              f"shape={binding.shape_derivation.name}, "
              f"position={binding.position_derivation.name}")

    # Step 5: Extract additional info based on puzzle type
    first_input = np.array(train_examples[0]['input'])
    first_output = np.array(train_examples[0]['output'])

    template_pixels = None
    if partition.mode == PartitionMode.OBJECT_TYPE:
        objects = extract_objects_from_grid(first_input, segmentation_mode=SegmentationMode.CONNECTIVITY)
        objects = [o for o in objects if not o.is_background and o.color > 0]
        template = find_template_object(objects)
        if template:
            # Normalize template pixels
            min_r = min(r for r, c in template.pixels)
            min_c = min(c for r, c in template.pixels)
            template_pixels = frozenset((r - min_r, c - min_c) for r, c in template.pixels)
            if verbose:
                print(f"\nTemplate: {len(template_pixels)} pixels, color {template.color}")

    rule = SequentialRule(
        partition=partition,
        binding=binding,
        pattern_vocab=vocab,
        output_shape=first_output.shape,
        template_pixels=template_pixels
    )

    if verbose:
        print(f"\n{rule.describe()}")

    return rule


# =============================================================================
# Application Function
# =============================================================================

def apply_sequential_rule(
    input_grid: np.ndarray,
    rule: SequentialRule,
    verbose: bool = False
) -> np.ndarray:
    """Apply a sequential binding rule to produce output.

    This is the main entry point for rule application.
    """
    if rule.partition.mode == PartitionMode.VERTICAL_DIVIDER:
        return apply_procedural(input_grid, rule, verbose)

    elif rule.partition.mode == PartitionMode.FIXED_REGIONS:
        return apply_pattern_to_fill(input_grid, rule, verbose)

    elif rule.partition.mode == PartitionMode.OBJECT_TYPE:
        return apply_template_replication(input_grid, rule, verbose)

    # Fallback
    return np.zeros_like(input_grid)


def apply_procedural(
    input_grid: np.ndarray,
    rule: SequentialRule,
    verbose: bool = False
) -> np.ndarray:
    """Apply procedural rule (136b0064 style)."""
    div_col = rule.partition.divider_col
    if div_col is None:
        div_col, _ = find_vertical_divider(input_grid)

    # Extract instruction patterns from instruction region
    instruction_region = input_grid[:, :div_col]
    patterns = find_instruction_patterns(instruction_region, rule.pattern_vocab, verbose)

    # Order patterns using column-then-row ordering
    ordered = order_objects(patterns, rule.binding.ordering_mode, div_col)

    if verbose:
        print(f"Found {len(ordered)} instruction patterns")
        for i, obj in enumerate(ordered):
            direction, length = derive_shape_params(obj, rule.binding, rule.pattern_vocab, instruction_region)
            print(f"  {i+1}: color={obj.color}, dir={direction}, len={length}")

    # Find start position
    start_row, start_col = find_start_marker(input_grid, div_col)
    if verbose:
        print(f"Start position: ({start_row}, {start_col})")

    # Determine output shape
    output_height = input_grid.shape[0]
    output_width = input_grid.shape[1] - div_col - 1
    output_shape = (output_height, output_width)

    # Execute - pass instruction_region for proper shape lookup
    return execute_procedural(ordered, rule, output_shape, (start_row, start_col), instruction_region)


def find_instruction_patterns(
    grid: np.ndarray,
    vocab: PatternVocabulary,
    verbose: bool = False
) -> List[Object]:
    """Find instruction patterns in the instruction region.

    Uses connected component analysis to find complete objects,
    then matches each to the vocabulary by its normalized shape.

    Objects whose shape is not in the vocabulary are still returned
    (for fallback shape inference during execution).
    """
    # Extract objects using connected component analysis
    objects = extract_objects_from_grid(
        grid,
        segmentation_mode=SegmentationMode.CONNECTIVITY
    )

    # Filter to non-background objects
    patterns = [o for o in objects if not o.is_background and o.color > 0]

    if verbose:
        for obj in patterns:
            normalized = normalize_object_shape(obj)
            in_vocab = normalized in vocab.entries
            print(f"  Object at ({obj.row},{obj.col}): {len(obj.pixels)}px, "
                  f"color={obj.color}, in_vocab={in_vocab}")

    return patterns


def apply_pattern_to_fill(
    input_grid: np.ndarray,
    rule: SequentialRule,
    verbose: bool = False
) -> np.ndarray:
    """Apply pattern-to-fill rule (17cae0c1 style)."""
    return execute_pattern_to_fill(input_grid, rule, input_grid.shape, verbose)


def apply_template_replication(
    input_grid: np.ndarray,
    rule: SequentialRule,
    verbose: bool = False
) -> np.ndarray:
    """Apply template replication rule (12997ef3 style).

    IMPORTANT: The template is extracted from the CURRENT input, not from
    training. Each input may have a different template shape.
    """
    objects = extract_objects_from_grid(input_grid, segmentation_mode=SegmentationMode.CONNECTIVITY)
    objects = [o for o in objects if not o.is_background and o.color > 0]

    # Find template and color sources from THIS input
    template = find_template_object(objects)
    if template is None:
        return np.zeros_like(input_grid)

    color_sources = find_color_sources(objects, template)

    if verbose:
        print(f"Template: {len(template.pixels)} pixels, color {template.color}")
        print(f"Color sources: {[o.color for o in color_sources]}")

    # Extract template pixels from CURRENT input (not stored from training)
    min_r = min(r for r, c in template.pixels)
    min_c = min(c for r, c in template.pixels)
    template_pixels = frozenset((r - min_r, c - min_c) for r, c in template.pixels)

    # Determine output shape from template dimensions
    rows = [r for r, c in template_pixels]
    cols = [c for r, c in template_pixels]
    template_height = max(rows) - min(rows) + 1 if rows else 1
    template_width = max(cols) - min(cols) + 1 if cols else 1

    n = len(color_sources)
    if n >= 2:
        row_span = max(o.row for o in color_sources) - min(o.row for o in color_sources)
        col_span = max(o.col for o in color_sources) - min(o.col for o in color_sources)
        horizontal = col_span >= row_span
    else:
        horizontal = True

    if horizontal:
        output_shape = (template_height, template_width * n)
    else:
        output_shape = (template_height * n, template_width)

    return execute_template_replication(color_sources, template_pixels, rule, output_shape)


# =============================================================================
# CLI for Testing
# =============================================================================

def main():
    import argparse
    from puzzle_loader import load_puzzle

    parser = argparse.ArgumentParser(description='Sequential Binding Module')
    parser.add_argument('--puzzle-id', type=str, required=True, help='Puzzle ID')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')

    args = parser.parse_args()

    # Load puzzle
    puzzle = load_puzzle(args.puzzle_id)

    print(f"Analyzing puzzle {args.puzzle_id} with sequential binding")
    print("=" * 60)

    # Discover rule
    rule = discover_sequential_rule(puzzle, verbose=args.verbose)

    if rule is None:
        print("\nNo sequential rule discovered")
        return

    print(f"\n{'='*60}")
    print("Testing on training examples:")
    print("=" * 60)

    total_correct = 0
    total_examples = len(puzzle['train'])

    for i, ex in enumerate(puzzle['train']):
        input_grid = np.array(ex['input'])
        expected_output = np.array(ex['output'])

        predicted_output = apply_sequential_rule(input_grid, rule, verbose=args.verbose)

        # Handle shape mismatch
        if predicted_output.shape != expected_output.shape:
            print(f"  Example {i+1}: Shape mismatch - expected {expected_output.shape}, got {predicted_output.shape}")
            # Resize for comparison
            h, w = expected_output.shape
            resized = np.zeros((h, w), dtype=predicted_output.dtype)
            ph, pw = min(h, predicted_output.shape[0]), min(w, predicted_output.shape[1])
            resized[:ph, :pw] = predicted_output[:ph, :pw]
            predicted_output = resized

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
            predicted = apply_sequential_rule(input_grid, rule, verbose=args.verbose)

            if 'output' in ex:
                expected = np.array(ex['output'])
                if predicted.shape != expected.shape:
                    h, w = expected.shape
                    resized = np.zeros((h, w), dtype=predicted.dtype)
                    ph, pw = min(h, predicted.shape[0]), min(w, predicted.shape[1])
                    resized[:ph, :pw] = predicted[:ph, :pw]
                    predicted = resized

                match = np.array_equal(predicted, expected)
                accuracy = np.mean(predicted == expected)
                status = "PASS" if match else "FAIL"
                print(f"  Test {i+1}: {status} (pixel accuracy: {accuracy:.1%})")
            else:
                print(f"  Test {i+1}: Prediction shape {predicted.shape}")


if __name__ == "__main__":
    main()
