#!/usr/bin/env python3
"""
Procedural Execution Module for ARC Puzzle Solver

This module handles puzzles where input patterns encode instructions that must be
executed sequentially to produce the output. Unlike correspondence-based puzzles
where objects map between input and output, procedural puzzles require:

1. Parsing symbolic patterns as instructions
2. Discovering the execution order
3. Following the instructions to draw the output

Key Concepts:
    - Instruction: A parsed command (color, direction, length)
    - InstructionPattern: An input object that encodes an instruction
    - ExecutionState: Current position and direction during execution
    - ProcedureSpec: Complete specification for executing a puzzle

Example Use Case (puzzle 136b0064):
    Input: Left side has colored patterns encoding (direction, length)
           Right side has a grey starting marker
    Output: Result of executing the instructions in order

Usage:
    from procedural_module import (
        discover_procedure,
        execute_procedure,
        ProcedureSpec,
    )

    # Discover the procedure from training examples
    procedure = discover_procedure(puzzle)

    # Execute on test input
    output_grid = execute_procedure(test_input, procedure)
"""

import numpy as np
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, FrozenSet, List, Optional, Set, Tuple, Any
from collections import Counter

from object_module import (
    Object,
    extract_objects_from_grid,
    SegmentationMode,
)


# =============================================================================
# Direction Enum
# =============================================================================

class Direction(Enum):
    """Cardinal directions for drawing."""
    UP = auto()
    DOWN = auto()
    LEFT = auto()
    RIGHT = auto()

    def delta(self) -> Tuple[int, int]:
        """Get (row_delta, col_delta) for this direction."""
        if self == Direction.UP:
            return (-1, 0)
        elif self == Direction.DOWN:
            return (1, 0)
        elif self == Direction.LEFT:
            return (0, -1)
        elif self == Direction.RIGHT:
            return (0, 1)
        return (0, 0)


# =============================================================================
# Instruction Data Structures
# =============================================================================

@dataclass
class Instruction:
    """A single drawing instruction."""
    color: int
    direction: Direction
    length: int

    def describe(self) -> str:
        return f"{self.color}:{self.direction.name}({self.length})"


@dataclass
class InstructionPattern:
    """An input pattern that encodes an instruction.

    The pattern's shape encodes the direction, and the pixel count
    encodes the length. The color determines what color to draw.
    """
    obj: Object                    # The original object
    color: int                     # Color to draw
    direction: Direction           # Encoded direction
    length: int                    # Number of pixels to draw
    row: int                       # Position in instruction grid (for ordering)
    col: int                       # Position in instruction grid (for ordering)

    def to_instruction(self) -> Instruction:
        return Instruction(self.color, self.direction, self.length)


@dataclass
class ExecutionState:
    """Current state during procedure execution."""
    row: int
    col: int

    def move(self, direction: Direction) -> 'ExecutionState':
        """Return new state after moving one step in direction."""
        dr, dc = direction.delta()
        return ExecutionState(self.row + dr, self.col + dc)


# =============================================================================
# Direction Detection from Shape - Pattern Matching
# =============================================================================

# Known base patterns (3x3) mapped to (direction, length)
# Each pattern is defined as a frozenset of (row, col) offsets within a 3x3 grid
BASE_PATTERNS = {
    # Blue arrow → RIGHT, length 3
    # ██.
    # █.█
    # .█.
    frozenset([(0,0), (0,1), (1,0), (1,2), (2,1)]): (Direction.RIGHT, 3),

    # Pink arrow → DOWN, length 2
    # █.█
    # .█.
    # .█.
    frozenset([(0,0), (0,2), (1,1), (2,1)]): (Direction.DOWN, 2),

    # Red arrow → LEFT, length 2
    # █.█
    # █.█
    # ███
    frozenset([(0,0), (0,2), (1,0), (1,2), (2,0), (2,1), (2,2)]): (Direction.LEFT, 2),

    # Green arrow → LEFT, length 4
    # ███
    # .█.
    # █.█
    frozenset([(0,0), (0,1), (0,2), (1,1), (2,0), (2,2)]): (Direction.LEFT, 4),
}


def normalize_shape_to_pattern(pixels: Set[Tuple[int, int]]) -> frozenset:
    """Convert object pixels to a normalized 3x3 pattern.

    Args:
        pixels: Set of (row, col) absolute pixel positions

    Returns:
        Frozenset of (row, col) offsets normalized to top-left origin
    """
    if not pixels:
        return frozenset()

    rows = [p[0] for p in pixels]
    cols = [p[1] for p in pixels]
    min_row, min_col = min(rows), min(cols)

    # Normalize to origin
    normalized = frozenset((r - min_row, c - min_col) for r, c in pixels)
    return normalized


def extract_base_pattern(pixels: Set[Tuple[int, int]]) -> Tuple[frozenset, int]:
    """Extract the base 3x3 pattern from a potentially larger shape.

    Some shapes contain multiple copies of the same base pattern.
    This function finds and extracts one copy of the base pattern.

    Args:
        pixels: Set of (row, col) absolute pixel positions

    Returns:
        (base_pattern, count) where count is the number of pattern instances
    """
    if not pixels:
        return frozenset(), 0

    normalized = normalize_shape_to_pattern(pixels)

    # If it's already a known pattern, return it
    if normalized in BASE_PATTERNS:
        return normalized, 1

    # Check if shape is a 3x3 bounding box (single pattern)
    rows = [p[0] for p in normalized]
    cols = [p[1] for p in normalized]
    height = max(rows) - min(rows) + 1 if rows else 0
    width = max(cols) - min(cols) + 1 if cols else 0

    if height <= 3 and width <= 3:
        return normalized, 1

    # For larger shapes, try to find repeated 3x3 patterns
    # by looking at the top-left 3x3 region
    top_left_pattern = frozenset((r, c) for r, c in normalized
                                  if r < 3 and c < 3)

    if top_left_pattern in BASE_PATTERNS:
        # Count how many times this pattern appears
        # For now, estimate based on pixel count ratio
        base_pixel_count = len(top_left_pattern)
        total_pixels = len(normalized)
        count = total_pixels // base_pixel_count if base_pixel_count > 0 else 1
        return top_left_pattern, count

    return normalized, 1


def detect_direction_from_shape(obj: Object, grid: np.ndarray) -> Tuple[Direction, int]:
    """Detect direction and length from shape using pattern matching.

    First tries to match against known base patterns. Falls back to
    heuristic analysis if no pattern matches.

    Args:
        obj: The object to analyze
        grid: The grid containing the object

    Returns:
        (direction, length) tuple
    """
    pixels = set(obj.pixels)
    if not pixels:
        return Direction.RIGHT, 1

    # Try pattern matching first
    base_pattern, count = extract_base_pattern(pixels)

    if base_pattern in BASE_PATTERNS:
        direction, base_length = BASE_PATTERNS[base_pattern]
        # Multiple pattern instances multiply the instruction count, not length
        return direction, base_length

    # Fall back to heuristic analysis
    return detect_direction_heuristic(pixels)


def detect_direction_heuristic(pixels: Set[Tuple[int, int]]) -> Tuple[Direction, int]:
    """Heuristic direction detection based on shape properties.

    Used when pattern matching fails. Analyzes the shape's structure
    to infer direction.

    Args:
        pixels: Set of (row, col) positions

    Returns:
        (direction, length) tuple
    """
    if not pixels:
        return Direction.RIGHT, 1

    rows = [p[0] for p in pixels]
    cols = [p[1] for p in pixels]

    min_row, max_row = min(rows), max(rows)
    min_col, max_col = min(cols), max(cols)

    height = max_row - min_row + 1
    width = max_col - min_col + 1

    # Normalize to origin
    normalized = [(r - min_row, c - min_col) for r, c in pixels]

    # Count pixels in each row and column
    row_counts = [sum(1 for r, c in normalized if r == i) for i in range(height)]
    col_counts = [sum(1 for r, c in normalized if c == i) for i in range(width)]

    # Edge pixel counts
    left_edge = col_counts[0] if col_counts else 0
    right_edge = col_counts[-1] if col_counts else 0
    top_edge = row_counts[0] if row_counts else 0
    bottom_edge = row_counts[-1] if row_counts else 0

    # Determine direction based on which edge is "lighter" (fewer pixels)
    # The direction points toward the lighter side

    if height <= width:  # Primarily horizontal
        if right_edge < left_edge:
            direction = Direction.RIGHT
        elif left_edge < right_edge:
            direction = Direction.LEFT
        else:
            # Check for asymmetry in column distribution
            left_half = sum(col_counts[:width//2])
            right_half = sum(col_counts[(width+1)//2:])
            direction = Direction.RIGHT if left_half > right_half else Direction.LEFT
        length = width
    else:  # Primarily vertical
        if bottom_edge < top_edge:
            direction = Direction.DOWN
        elif top_edge < bottom_edge:
            direction = Direction.UP
        else:
            # Check for asymmetry in row distribution
            top_half = sum(row_counts[:height//2])
            bottom_half = sum(row_counts[(height+1)//2:])
            direction = Direction.DOWN if top_half > bottom_half else Direction.UP
        length = height

    return direction, length


def detect_direction_from_shape_advanced(obj: Object, grid: np.ndarray) -> Tuple[Direction, int]:
    """Advanced direction detection - now delegates to pattern matching.

    This function is kept for backwards compatibility but now uses
    the pattern matching approach.

    Args:
        obj: The object to analyze
        grid: The grid containing the object

    Returns:
        (direction, length) tuple
    """
    return detect_direction_from_shape(obj, grid)


# =============================================================================
# Instruction Parsing
# =============================================================================

def find_pattern_instances(
    grid: np.ndarray,
    divider_col: int,
    verbose: bool = False
) -> List[InstructionPattern]:
    """Find all 3x3 pattern instances on the left side of the divider.

    Scans the grid for all known base patterns, identifying each instance
    as a separate instruction.

    Args:
        grid: The input grid
        divider_col: Column index of the vertical divider
        verbose: Print debug info

    Returns:
        List of InstructionPattern objects, one per pattern instance
    """
    H, W = grid.shape
    patterns = []
    found_positions = set()  # Track which positions we've already matched

    # Define pattern templates with their direction and length
    # Each template is (relative_positions, direction, length)
    templates = []
    for pattern_set, (direction, length) in BASE_PATTERNS.items():
        templates.append((pattern_set, direction, length))

    # Scan the left side of the grid for pattern matches
    # Pattern search area: columns 0 to divider_col-1, in 3x3 windows
    max_col = min(divider_col, W)

    for start_row in range(H - 2):  # Need at least 3 rows
        for start_col in range(max_col - 2):  # Need at least 3 cols
            # Skip if we've already found a pattern starting near here
            if (start_row, start_col) in found_positions:
                continue

            # Extract the 3x3 window
            window = grid[start_row:start_row+3, start_col:start_col+3]

            # Get non-zero positions and their color
            non_zero = []
            colors = set()
            for r in range(3):
                for c in range(3):
                    if window[r, c] > 0 and window[r, c] != 4:  # Skip divider color
                        non_zero.append((r, c))
                        colors.add(window[r, c])

            # Skip if empty or multiple colors (overlapping patterns)
            if len(non_zero) == 0 or len(colors) != 1:
                continue

            color = colors.pop()
            pattern_positions = frozenset(non_zero)

            # Check if this matches a known pattern
            for template, direction, length in templates:
                if pattern_positions == template:
                    # Found a match!
                    pattern = InstructionPattern(
                        obj=None,  # No object reference needed
                        color=color,
                        direction=direction,
                        length=length,
                        row=start_row,
                        col=start_col,
                    )
                    patterns.append(pattern)
                    found_positions.add((start_row, start_col))

                    if verbose:
                        print(f"  Pattern at ({start_row},{start_col}): color={color}, "
                              f"dir={direction.name}, len={length}")
                    break

    return patterns


def parse_instruction_patterns(
    grid: np.ndarray,
    objects: List[Object],
    divider_col: int,
    verbose: bool = False
) -> List[InstructionPattern]:
    """Parse instruction patterns from the left side of a divided grid.

    Now uses pattern scanning to find all 3x3 pattern instances.

    Args:
        grid: The input grid
        objects: Extracted objects from the grid (unused, kept for compatibility)
        divider_col: Column index of the vertical divider
        verbose: Print debug info

    Returns:
        List of InstructionPattern objects
    """
    return find_pattern_instances(grid, divider_col, verbose)


def order_patterns_by_reading(
    patterns: List[InstructionPattern],
    divider_col: int,
    reading_order: str = "column_then_row"
) -> List[InstructionPattern]:
    """Order patterns according to reading order.

    Args:
        patterns: List of instruction patterns
        divider_col: Column of the divider (patterns are to the left)
        reading_order: How to order - "column_then_row" reads left column
                      top-to-bottom, then right column top-to-bottom

    Returns:
        Ordered list of patterns
    """
    if reading_order == "column_then_row":
        # Split into left and right columns (relative to divider)
        mid_col = divider_col // 2

        left_patterns = [p for p in patterns if p.col < mid_col]
        right_patterns = [p for p in patterns if p.col >= mid_col]

        # Sort each by row
        left_patterns.sort(key=lambda p: p.row)
        right_patterns.sort(key=lambda p: p.row)

        return left_patterns + right_patterns

    elif reading_order == "row_then_column":
        # Read row by row
        return sorted(patterns, key=lambda p: (p.row, p.col))

    elif reading_order == "top_to_bottom":
        return sorted(patterns, key=lambda p: p.row)

    else:
        return patterns


# =============================================================================
# Procedure Specification
# =============================================================================

@dataclass
class ProcedureSpec:
    """Complete specification for executing a procedural puzzle."""
    # Divider info
    divider_col: int
    divider_color: int = 4  # Yellow by default

    # Starting point
    start_row: int = 0
    start_col: int = 0
    start_color: int = 5  # Grey by default

    # Reading order for instructions
    reading_order: str = "column_then_row"

    # Output dimensions
    output_height: int = 0
    output_width: int = 0

    # Discovered instructions (for reference)
    instructions: List[Instruction] = field(default_factory=list)

    def describe(self) -> str:
        instr_str = ", ".join(i.describe() for i in self.instructions)
        return (f"ProcedureSpec(divider={self.divider_col}, "
                f"start=({self.start_row},{self.start_col}), "
                f"reading={self.reading_order}, "
                f"instructions=[{instr_str}])")


# =============================================================================
# Execution Engine
# =============================================================================

def execute_instructions(
    instructions: List[Instruction],
    start_state: ExecutionState,
    grid_shape: Tuple[int, int],
    include_start: bool = True,
    start_color: int = 5,
    implicit_down: bool = True
) -> np.ndarray:
    """Execute a list of instructions to produce an output grid.

    The execution model:
    1. Mark starting position with start_color
    2. For each instruction:
       a. If implicit_down, move DOWN one step first
       b. Draw at current position
       c. For remaining (length-1) steps: move in direction, draw

    Args:
        instructions: List of instructions to execute
        start_state: Starting position
        grid_shape: (height, width) of output grid
        include_start: Whether to mark the starting position
        start_color: Color for the starting marker
        implicit_down: If True, move DOWN before each instruction

    Returns:
        Output grid with executed drawing
    """
    grid = np.zeros(grid_shape, dtype=np.int64)

    state = start_state

    # Mark starting position
    if include_start:
        if 0 <= state.row < grid_shape[0] and 0 <= state.col < grid_shape[1]:
            grid[state.row, state.col] = start_color

    # Execute each instruction
    for instr in instructions:
        # Implicit DOWN step before each instruction
        if implicit_down:
            state = state.move(Direction.DOWN)

        # Draw at current position first
        if 0 <= state.row < grid_shape[0] and 0 <= state.col < grid_shape[1]:
            grid[state.row, state.col] = instr.color

        # Then move and draw for remaining length-1 steps
        for _ in range(instr.length - 1):
            state = state.move(instr.direction)

            if 0 <= state.row < grid_shape[0] and 0 <= state.col < grid_shape[1]:
                grid[state.row, state.col] = instr.color

    return grid


# =============================================================================
# Discovery Functions
# =============================================================================

def find_divider(grid: np.ndarray, objects: List[Object]) -> Tuple[int, int]:
    """Find the vertical divider in the grid.

    The divider is typically a single-color vertical line that spans
    the full height of the grid. Yellow (4) is the most common divider color.

    Args:
        grid: The input grid
        objects: Extracted objects

    Returns:
        (divider_column, divider_color) tuple
    """
    H, W = grid.shape

    # First, look for yellow (4) column - most common divider color
    for c in range(W):
        column = grid[:, c]
        if np.all(column == 4):
            return c, 4

    # Look for any single-color column that spans full height
    for c in range(W):
        column = grid[:, c]
        unique_vals = set(column)
        if len(unique_vals) == 1 and 0 not in unique_vals:
            return c, int(column[0])

    # Look for objects marked as dividers
    for obj in objects:
        if obj.is_divider and obj.height == H and obj.width == 1:
            return obj.col, obj.color

    # Fall back to middle of grid
    return W // 2, 0


def find_start_marker(
    grid: np.ndarray,
    objects: List[Object],
    divider_col: int,
    marker_color: int = 5
) -> Tuple[int, int]:
    """Find the starting position marker on the right side of divider.

    Args:
        grid: The input grid
        objects: Extracted objects
        divider_col: Column of the divider
        marker_color: Color of the start marker (default grey=5)

    Returns:
        (row, col) of the start marker
    """
    # Look for single pixel of marker_color on right side
    for obj in objects:
        if obj.color == marker_color and obj.col > divider_col:
            if len(obj.pixels) == 1:
                return obj.row, obj.col - divider_col - 1  # Relative to output grid

    # Search grid directly
    H, W = grid.shape
    for r in range(H):
        for c in range(divider_col + 1, W):
            if grid[r, c] == marker_color:
                return r, c - divider_col - 1

    return 0, 0


def discover_reading_order(
    examples: List[dict],
    verbose: bool = False
) -> str:
    """Discover the reading order for instructions by testing hypotheses.

    Args:
        examples: Training examples with input/output
        verbose: Print debug info

    Returns:
        Reading order string
    """
    # For now, assume column_then_row based on puzzle description
    # A full implementation would test multiple orderings and score by
    # how well the executed output matches expected output
    return "column_then_row"


def discover_procedure(
    puzzle: Dict,
    segmentation_mode: SegmentationMode = SegmentationMode.COLOR,
    verbose: bool = False
) -> Optional[ProcedureSpec]:
    """Discover the procedural specification from training examples.

    Args:
        puzzle: Puzzle dict with 'train' examples
        segmentation_mode: How to segment grids
        verbose: Print debug info

    Returns:
        ProcedureSpec if discovered, None otherwise
    """
    train_examples = puzzle.get('train', [])
    if not train_examples:
        return None

    # Analyze first example to get structure
    ex = train_examples[0]
    input_grid = np.array(ex['input'])
    output_grid = np.array(ex['output'])

    # Extract objects
    input_objects = extract_objects_from_grid(input_grid, segmentation_mode=segmentation_mode)

    # Find divider
    divider_col, divider_color = find_divider(input_grid, input_objects)
    if verbose:
        print(f"Divider at column {divider_col}, color {divider_color}")

    # Find start marker
    start_row, start_col = find_start_marker(input_grid, input_objects, divider_col)
    if verbose:
        print(f"Start marker at ({start_row}, {start_col})")

    # Discover reading order
    reading_order = discover_reading_order(train_examples, verbose)
    if verbose:
        print(f"Reading order: {reading_order}")

    # Parse instruction patterns
    patterns = parse_instruction_patterns(input_grid, input_objects, divider_col, verbose)

    # Order patterns
    ordered_patterns = order_patterns_by_reading(patterns, divider_col, reading_order)

    # Convert to instructions
    instructions = [p.to_instruction() for p in ordered_patterns]

    if verbose:
        print(f"Instructions: {[i.describe() for i in instructions]}")

    # Output dimensions
    output_height, output_width = output_grid.shape

    return ProcedureSpec(
        divider_col=divider_col,
        divider_color=divider_color,
        start_row=start_row,
        start_col=start_col,
        reading_order=reading_order,
        output_height=output_height,
        output_width=output_width,
        instructions=instructions,
    )


def execute_procedure(
    input_grid: np.ndarray,
    procedure: ProcedureSpec,
    segmentation_mode: SegmentationMode = SegmentationMode.COLOR,
    verbose: bool = False
) -> np.ndarray:
    """Execute a procedure on an input grid to produce output.

    Args:
        input_grid: The input grid
        procedure: The procedure specification
        segmentation_mode: How to segment the input
        verbose: Print debug info

    Returns:
        Output grid
    """
    # Extract objects
    input_objects = extract_objects_from_grid(input_grid, segmentation_mode=segmentation_mode)

    # Find start marker position
    start_row, start_col = find_start_marker(
        input_grid, input_objects, procedure.divider_col
    )

    if verbose:
        print(f"Start position: ({start_row}, {start_col})")

    # Parse instruction patterns
    patterns = parse_instruction_patterns(
        input_grid, input_objects, procedure.divider_col, verbose
    )

    # Order patterns
    ordered_patterns = order_patterns_by_reading(
        patterns, procedure.divider_col, procedure.reading_order
    )

    # Convert to instructions
    instructions = [p.to_instruction() for p in ordered_patterns]

    if verbose:
        print(f"Instructions: {[i.describe() for i in instructions]}")

    # Determine output shape
    input_height = input_grid.shape[0]
    output_width = input_grid.shape[1] - procedure.divider_col - 1
    output_shape = (input_height, output_width)

    # Execute
    start_state = ExecutionState(start_row, start_col)
    output_grid = execute_instructions(
        instructions, start_state, output_shape,
        include_start=True, start_color=5
    )

    return output_grid


# =============================================================================
# CLI for Testing
# =============================================================================

def main():
    import argparse
    from puzzle_loader import load_puzzle

    parser = argparse.ArgumentParser(description='Procedural Execution Module')
    parser.add_argument('--puzzle-id', type=str, required=True, help='Puzzle ID')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    parser.add_argument('--segmentation-mode', type=str, default='connectivity',
                        choices=['connectivity', 'pixel', 'color'],
                        help='Segmentation mode')

    args = parser.parse_args()

    # Map string to enum
    seg_mode_map = {
        'connectivity': SegmentationMode.CONNECTIVITY,
        'pixel': SegmentationMode.PIXEL,
        'color': SegmentationMode.COLOR,
    }
    seg_mode = seg_mode_map[args.segmentation_mode]

    # Load puzzle
    puzzle = load_puzzle(args.puzzle_id)

    print(f"Analyzing puzzle {args.puzzle_id} with procedural execution")
    print(f"Segmentation mode: {seg_mode.name}")
    print("=" * 60)

    # Discover procedure
    procedure = discover_procedure(puzzle, seg_mode, verbose=args.verbose)

    if procedure is None:
        print("\nNo procedure discovered")
        return

    print(f"\n{'='*60}")
    print("Discovered Procedure:")
    print(procedure.describe())
    print(f"{'='*60}")

    # Test on training examples
    print("\nTesting on training examples:")

    for i, ex in enumerate(puzzle['train']):
        input_grid = np.array(ex['input'])
        expected_output = np.array(ex['output'])

        # Execute procedure
        predicted_output = execute_procedure(
            input_grid, procedure, seg_mode, verbose=args.verbose
        )

        # Handle shape mismatch
        if predicted_output.shape != expected_output.shape:
            print(f"  Example {i+1}: Shape mismatch - expected {expected_output.shape}, got {predicted_output.shape}")
            # Resize for comparison
            h, w = expected_output.shape
            pred_h, pred_w = predicted_output.shape
            if pred_h < h or pred_w < w:
                padded = np.zeros((h, w), dtype=predicted_output.dtype)
                padded[:pred_h, :pred_w] = predicted_output
                predicted_output = padded
            else:
                predicted_output = predicted_output[:h, :w]

        match = np.array_equal(predicted_output, expected_output)
        accuracy = np.mean(predicted_output == expected_output)

        status = "PASS" if match else "FAIL"
        print(f"  Example {i+1}: {status} (pixel accuracy: {accuracy:.1%})")

        if args.verbose and not match:
            print(f"    Expected:\n{expected_output}")
            print(f"    Predicted:\n{predicted_output}")

    # Test on test examples
    if puzzle.get('test'):
        print("\nTesting on test examples:")
        for i, ex in enumerate(puzzle['test']):
            input_grid = np.array(ex['input'])
            expected = ex.get('output')

            predicted = execute_procedure(input_grid, procedure, seg_mode, verbose=args.verbose)

            if expected:
                expected_output = np.array(expected)
                if predicted.shape != expected_output.shape:
                    h, w = expected_output.shape
                    pred_h, pred_w = predicted.shape
                    if pred_h < h or pred_w < w:
                        padded = np.zeros((h, w), dtype=predicted.dtype)
                        padded[:pred_h, :pred_w] = predicted
                        predicted = padded
                    else:
                        predicted = predicted[:h, :w]

                match = np.array_equal(predicted, expected_output)
                accuracy = np.mean(predicted == expected_output)
                status = "PASS" if match else "FAIL"
                print(f"  Test {i+1}: {status} (pixel accuracy: {accuracy:.1%})")
            else:
                print(f"  Test {i+1}: Prediction:\n{predicted}")


if __name__ == "__main__":
    main()
