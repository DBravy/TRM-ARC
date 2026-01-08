#!/usr/bin/env python3
"""
Multi-Head Correspondence Module for ARC Puzzle Solver

This module implements "parallel attention" for correspondence finding - multiple
correspondence heads that focus on different object features (shape, color, position)
and can be combined to solve puzzles requiring factored correspondence.

Key Concepts:
    - CorrespondenceHead: Configuration for one correspondence channel
    - IndexBinding: How output indices map to correspondence indices
    - MultiHeadCorrespondence: Runs multiple correspondence passes
    - CompositeDerivation: Specifies how to derive output features from multiple heads

Example Use Case (puzzle 12997ef3):
    Input: A shape template (color 1) + color pixels (colors 2,3,4)
    Output: Multiple copies of the template, each colored differently

    Solution requires:
    - Shape head: template -> all output shapes (1:N)
    - Color head: each color pixel -> corresponding output (1:1 sequential)
    - Binding: shape from head[constant], color from head[sequential]

Usage:
    from multi_head_correspondence import (
        MultiHeadCorrespondence,
        screen_composite_derivation,
        apply_composite_derivation,
    )

    # Find correspondences from all heads
    mhc = MultiHeadCorrespondence()
    head_results = mhc.find_all(input_grid, output_grid, in_objs, out_objs)

    # Screen to discover binding rules
    derivation = screen_composite_derivation(examples, head_results)

    # Apply to generate output
    output_objects = apply_composite_derivation(input_grid, derivation, mhc)
"""

import numpy as np
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Set, Tuple, Any
from collections import Counter

from correspondence_module import (
    find_correspondences,
    CorrespondenceMode,
    DEFAULT_MARGIN,
)
from object_module import (
    Object,
    extract_objects_from_grid,
    SegmentationMode,
)
from anchoring_module import AnchorPoint, get_anchor_position
from genesis_module import PositionSpec


# =============================================================================
# Correspondence Head Configuration
# =============================================================================

@dataclass
class CorrespondenceHead:
    """Configuration for one correspondence channel.

    Each head focuses on specific object features by adjusting weights
    and can use different matching modes (1:1, 1:N, N:1).
    """
    name: str
    weights: Dict[str, float]
    mode: CorrespondenceMode = "one_to_one"
    margin: float = DEFAULT_MARGIN
    threshold: float = 0.3

    # What feature this head is designed to capture
    feature_focus: str = "mixed"  # 'shape', 'color', 'position', 'size', 'mixed'

    def describe(self) -> str:
        active = [k for k, v in self.weights.items() if v > 0]
        return f"{self.name}(focus={self.feature_focus}, mode={self.mode}, weights={active})"


# Standard head configurations
SHAPE_HEAD = CorrespondenceHead(
    name="shape",
    weights={'structural': 1.0, 'moments': 1.0, 'fourier': 1.0, 'color': 0.0, 'location': 0.0},
    mode="one_to_many",
    margin=0.2,
    feature_focus="shape"
)

COLOR_HEAD = CorrespondenceHead(
    name="color",
    weights={'structural': 0.0, 'moments': 0.0, 'fourier': 0.0, 'color': 1.0, 'location': 0.0},
    mode="one_to_one",
    feature_focus="color"
)

POSITION_HEAD = CorrespondenceHead(
    name="position",
    weights={'structural': 0.0, 'moments': 0.0, 'fourier': 0.0, 'color': 0.0, 'location': 1.0},
    mode="one_to_one",
    feature_focus="position"
)

SIZE_HEAD = CorrespondenceHead(
    name="size",
    weights={'structural': 1.0, 'moments': 0.0, 'fourier': 0.0, 'color': 0.0, 'location': 0.0},
    mode="one_to_one",
    feature_focus="size"
)


# =============================================================================
# Index Binding Types
# =============================================================================

class BindingType(Enum):
    """How output index maps to correspondence index."""
    CONSTANT = auto()      # Always use same index (e.g., always first match)
    IDENTITY = auto()      # Output[i] uses correspondence[i]
    SEQUENCE = auto()      # Output[i] uses correspondence in spatial order
    REVERSE = auto()       # Output[i] uses correspondence[n-1-i]


@dataclass
class IndexBinding:
    """Specification for how output indices map to correspondence indices.

    For a given correspondence head, this describes how each output object
    should select its correspondent from the matches.
    """
    binding_type: BindingType
    constant_value: Optional[int] = None  # For CONSTANT type
    variance: float = float('inf')  # How well this binding explains the data

    def get_correspondent_index(self, output_idx: int, num_correspondents: int) -> int:
        """Get the correspondent index for a given output index."""
        if self.binding_type == BindingType.CONSTANT:
            return self.constant_value if self.constant_value is not None else 0
        elif self.binding_type == BindingType.IDENTITY:
            return min(output_idx, num_correspondents - 1)
        elif self.binding_type == BindingType.SEQUENCE:
            return min(output_idx, num_correspondents - 1)
        elif self.binding_type == BindingType.REVERSE:
            return max(0, num_correspondents - 1 - output_idx)
        return 0

    def describe(self) -> str:
        if self.binding_type == BindingType.CONSTANT:
            return f"constant({self.constant_value})"
        return self.binding_type.name.lower()


# =============================================================================
# Head Results Container
# =============================================================================

@dataclass
class HeadResult:
    """Results from running one correspondence head."""
    head: CorrespondenceHead
    correspondences: List[Tuple[int, int, float]]
    input_objects: List[Object]
    output_objects: List[Object]

    def get_input_correspondents(self) -> List[int]:
        """Get unique input indices that have correspondences."""
        return list(sorted(set(in_idx for in_idx, _, _ in self.correspondences)))

    def get_output_correspondents(self) -> List[int]:
        """Get unique output indices that have correspondences."""
        return list(sorted(set(out_idx for _, out_idx, _ in self.correspondences)))

    def get_inputs_for_output(self, out_idx: int) -> List[Tuple[int, float]]:
        """Get all input indices that correspond to a given output."""
        return [(in_idx, score) for in_idx, o_idx, score in self.correspondences
                if o_idx == out_idx]

    def get_outputs_for_input(self, in_idx: int) -> List[Tuple[int, float]]:
        """Get all output indices that correspond to a given input."""
        return [(out_idx, score) for i_idx, out_idx, score in self.correspondences
                if i_idx == in_idx]

    def describe(self) -> str:
        n_corr = len(self.correspondences)
        n_in = len(self.get_input_correspondents())
        n_out = len(self.get_output_correspondents())
        return f"{self.head.name}: {n_corr} correspondences ({n_in} inputs -> {n_out} outputs)"


# =============================================================================
# Multi-Head Correspondence
# =============================================================================

class MultiHeadCorrespondence:
    """Run multiple correspondence passes with different feature focuses.

    This is the core class for factored correspondence. It maintains multiple
    "heads" that each look at objects through a different lens (shape-only,
    color-only, position-only, etc.).
    """

    def __init__(self, heads: Optional[List[CorrespondenceHead]] = None):
        """Initialize with a list of heads.

        Args:
            heads: List of CorrespondenceHead configurations. If None, uses
                   standard heads (shape, color, position).
        """
        if heads is None:
            heads = [SHAPE_HEAD, COLOR_HEAD, POSITION_HEAD]

        self.heads = {h.name: h for h in heads}

    def find_all(
        self,
        input_grid: np.ndarray,
        output_grid: np.ndarray,
        input_objects: List[Object],
        output_objects: List[Object],
    ) -> Dict[str, HeadResult]:
        """Run all correspondence heads and return results.

        Args:
            input_grid: Input grid array
            output_grid: Output grid array
            input_objects: Extracted input objects
            output_objects: Extracted output objects

        Returns:
            Dict mapping head name to HeadResult
        """
        results = {}

        for name, head in self.heads.items():
            correspondences, in_match, out_match = find_correspondences(
                input_grid, output_grid,
                input_objects, output_objects,
                weights=head.weights,
                mode=head.mode,
                margin=head.margin,
                threshold=head.threshold,
            )

            results[name] = HeadResult(
                head=head,
                correspondences=correspondences,
                input_objects=in_match,
                output_objects=out_match,
            )

        return results

    def describe_results(self, results: Dict[str, HeadResult]) -> str:
        """Get a human-readable summary of head results."""
        lines = ["Multi-Head Correspondence Results:"]
        for name, result in results.items():
            lines.append(f"  {result.describe()}")
        return "\n".join(lines)


# =============================================================================
# Feature Source Specification
# =============================================================================

@dataclass
class FeatureSource:
    """Specifies where a feature (shape, color, position) comes from.

    This connects a feature to a correspondence head and specifies how
    to select the correspondent for each output.
    """
    head_name: str              # Which correspondence head provides this
    binding: IndexBinding       # How to select correspondent index

    def describe(self) -> str:
        return f"{self.head_name}[{self.binding.describe()}]"


# =============================================================================
# Composite Derivation Rule
# =============================================================================

@dataclass
class CompositeDerivation:
    """Complete rule for deriving outputs from multiple correspondence heads.

    This is the "binding" layer that specifies:
    - Which head provides the shape template
    - Which head provides the color
    - How positions are determined
    - How many outputs to generate
    """
    shape_source: FeatureSource      # Where shape comes from
    color_source: FeatureSource      # Where color comes from
    position_source: Optional[FeatureSource] = None  # Optional position from head
    position_spec: Optional[PositionSpec] = None     # Or explicit position rule

    # Output count determination
    count_source: str = "color"      # Which head determines output count

    # Metadata
    total_variance: float = float('inf')

    def describe(self) -> str:
        pos = self.position_source.describe() if self.position_source else str(self.position_spec)
        return (f"CompositeDerivation(\n"
                f"  shape={self.shape_source.describe()},\n"
                f"  color={self.color_source.describe()},\n"
                f"  position={pos},\n"
                f"  count_from={self.count_source})")


# =============================================================================
# Training Example Data
# =============================================================================

@dataclass
class MultiHeadExampleData:
    """Data for one training example with multi-head correspondences."""
    input_grid: np.ndarray
    output_grid: np.ndarray
    input_objects: List[Object]
    output_objects: List[Object]
    head_results: Dict[str, HeadResult]


# =============================================================================
# Screening Functions
# =============================================================================

def _get_ordered_correspondents(
    head_result: HeadResult,
    ordering: str = "spatial"
) -> List[Tuple[int, Object]]:
    """Get correspondent input objects in a consistent order.

    Args:
        head_result: Results from one correspondence head
        ordering: How to order ('spatial' = top-left to bottom-right, 'index' = by index)

    Returns:
        List of (input_idx, input_object) tuples in order
    """
    input_indices = head_result.get_input_correspondents()

    if ordering == "spatial":
        # Sort by position (top-to-bottom, left-to-right)
        def sort_key(idx):
            obj = head_result.input_objects[idx]
            return (obj.row, obj.col)
        input_indices = sorted(input_indices, key=sort_key)

    return [(idx, head_result.input_objects[idx]) for idx in input_indices]


def _get_ordered_outputs(
    output_objects: List[Object],
    ordering: str = "spatial"
) -> List[Tuple[int, Object]]:
    """Get output objects in a consistent order.

    Args:
        output_objects: List of output objects
        ordering: How to order ('spatial' = top-left to bottom-right)

    Returns:
        List of (index, object) tuples in order
    """
    indices = list(range(len(output_objects)))

    if ordering == "spatial":
        def sort_key(idx):
            obj = output_objects[idx]
            return (obj.row, obj.col)
        indices = sorted(indices, key=sort_key)

    return [(idx, output_objects[idx]) for idx in indices]


def screen_shape_source(
    examples: List[MultiHeadExampleData],
    verbose: bool = False
) -> Tuple[str, IndexBinding]:
    """Screen to find which head provides shapes and with what binding.

    Looks for 1:N relationships where one input shape matches multiple outputs.

    Returns:
        (head_name, binding) tuple
    """
    best_head = "shape"
    best_binding = IndexBinding(BindingType.CONSTANT, 0)
    best_score = 0

    for head_name in ["shape", "color", "position"]:
        # Check if this head has 1:N pattern
        for ex in examples:
            if head_name not in ex.head_results:
                continue

            result = ex.head_results[head_name]

            # Count how many outputs each input maps to
            input_to_outputs = {}
            for in_idx, out_idx, score in result.correspondences:
                if in_idx not in input_to_outputs:
                    input_to_outputs[in_idx] = []
                input_to_outputs[in_idx].append(out_idx)

            # Look for 1:N pattern (one input -> many outputs)
            for in_idx, out_indices in input_to_outputs.items():
                if len(out_indices) > 1:
                    # This input maps to multiple outputs - potential shape source
                    # Check if shapes actually match
                    in_obj = result.input_objects[in_idx]
                    shape_matches = 0
                    for out_idx in out_indices:
                        out_obj = result.output_objects[out_idx]
                        # Compare dimensions
                        if (in_obj.height == out_obj.height and
                            in_obj.width == out_obj.width):
                            shape_matches += 1

                    score = shape_matches / len(out_indices)
                    if score > best_score:
                        best_score = score
                        best_head = head_name
                        best_binding = IndexBinding(BindingType.CONSTANT, in_idx)

    if verbose:
        print(f"Shape source: {best_head}[{best_binding.describe()}] (score={best_score:.2f})")

    return best_head, best_binding


def screen_color_binding(
    examples: List[MultiHeadExampleData],
    verbose: bool = False
) -> Tuple[str, IndexBinding]:
    """Screen to find how colors bind between heads and outputs.

    Checks different binding hypotheses:
    - IDENTITY: output[i] gets color from correspondent[i]
    - CONSTANT: all outputs get color from same correspondent
    - SEQUENCE: outputs get colors in spatial order

    Returns:
        (head_name, binding) tuple
    """
    hypotheses = [
        ("color", BindingType.IDENTITY),
        ("color", BindingType.SEQUENCE),
        ("color", BindingType.CONSTANT),
        ("shape", BindingType.IDENTITY),
    ]

    best_head = "color"
    best_binding = IndexBinding(BindingType.IDENTITY)
    best_score = -1

    for head_name, binding_type in hypotheses:
        total_matches = 0
        total_outputs = 0

        for ex in examples:
            if head_name not in ex.head_results:
                continue

            result = ex.head_results[head_name]

            # Get ordered correspondents and outputs
            ordered_corrs = _get_ordered_correspondents(result, "spatial")
            ordered_outs = _get_ordered_outputs(result.output_objects, "spatial")

            # Filter outputs to those with correspondences
            matched_out_indices = result.get_output_correspondents()
            ordered_outs = [(i, o) for i, o in ordered_outs if i in matched_out_indices]

            if not ordered_corrs or not ordered_outs:
                continue

            # Test this binding
            for out_pos, (out_idx, out_obj) in enumerate(ordered_outs):
                # Get expected correspondent based on binding
                if binding_type == BindingType.IDENTITY:
                    corr_pos = out_pos
                elif binding_type == BindingType.SEQUENCE:
                    corr_pos = out_pos
                elif binding_type == BindingType.CONSTANT:
                    corr_pos = 0
                else:
                    corr_pos = 0

                if corr_pos >= len(ordered_corrs):
                    continue

                corr_idx, corr_obj = ordered_corrs[corr_pos]

                # Check if colors match
                if out_obj.color == corr_obj.color:
                    total_matches += 1
                total_outputs += 1

        score = total_matches / total_outputs if total_outputs > 0 else 0

        if verbose:
            print(f"  Color binding {head_name}[{binding_type.name}]: {score:.2f} ({total_matches}/{total_outputs})")

        if score > best_score:
            best_score = score
            best_head = head_name
            best_binding = IndexBinding(binding_type, variance=1-score)

    if verbose:
        print(f"Best color binding: {best_head}[{best_binding.describe()}] (score={best_score:.2f})")

    return best_head, best_binding


def screen_composite_derivation(
    examples: List[MultiHeadExampleData],
    verbose: bool = False
) -> CompositeDerivation:
    """Screen to discover the complete composite derivation rule.

    This is the main screening function that discovers:
    - Which head provides shapes (and with what binding)
    - Which head provides colors (and with what binding)
    - How positions are determined
    - What determines output count

    Args:
        examples: List of training examples with head results
        verbose: Print screening details

    Returns:
        CompositeDerivation rule
    """
    if verbose:
        print("\n=== Screening Composite Derivation ===")

    # Screen shape source
    if verbose:
        print("\nScreening shape source:")
    shape_head, shape_binding = screen_shape_source(examples, verbose)

    # Screen color binding
    if verbose:
        print("\nScreening color binding:")
    color_head, color_binding = screen_color_binding(examples, verbose)

    # Determine count source - which head has N:1 or 1:1 that matches output count
    count_source = "color"  # Default: count determined by color correspondents

    # For now, use simple position spec (sequence along row/column)
    # TODO: Screen position hypotheses
    position_spec = None

    # Compute total variance
    total_variance = shape_binding.variance + color_binding.variance

    derivation = CompositeDerivation(
        shape_source=FeatureSource(shape_head, shape_binding),
        color_source=FeatureSource(color_head, color_binding),
        position_spec=position_spec,
        count_source=count_source,
        total_variance=total_variance,
    )

    if verbose:
        print(f"\n{derivation.describe()}")

    return derivation


# =============================================================================
# Application Functions
# =============================================================================

def _find_template_object(
    input_objects: List[Object],
    examples: Optional[List[MultiHeadExampleData]] = None
) -> Optional[Object]:
    """Find the shape template object from input.

    The template is typically:
    - The object that matches multiple output shapes (1:N in shape head)
    - Often has color=1 (blue) in ARC puzzles
    - Has a non-trivial shape (more than 1 pixel)

    Args:
        input_objects: List of input objects
        examples: Optional training examples for context

    Returns:
        The template Object, or None if not found
    """
    # Filter to objects with actual shapes (more than 1 pixel, non-background)
    candidates = [obj for obj in input_objects
                  if len(obj.pixels) > 1 and obj.color > 0 and not obj.is_background]

    if not candidates:
        # Fall back to any object with pixels
        candidates = [obj for obj in input_objects
                      if len(obj.pixels) > 1 and obj.color >= 0]

    if not candidates:
        return None

    # Prefer color=1 (common ARC pattern for templates)
    for obj in candidates:
        if obj.color == 1:
            return obj

    # Otherwise return largest non-single-pixel object
    return max(candidates, key=lambda o: len(o.pixels))


def _find_color_sources(
    input_objects: List[Object],
    template: Object
) -> List[Object]:
    """Find the color source objects from input.

    Color sources are typically:
    - Single pixels or small objects
    - Different colors from the template
    - Not background

    Args:
        input_objects: List of input objects
        template: The template object (to exclude)

    Returns:
        List of color source Objects in spatial order
    """
    # Filter to small objects with different colors
    sources = []
    for obj in input_objects:
        # Skip template, background, and Grid pseudo-objects
        if obj.color == template.color or obj.color <= 0 or obj.is_background:
            continue
        # Skip objects that are too similar in size to template (they might be shapes)
        if len(obj.pixels) > len(template.pixels) * 0.5:
            continue
        sources.append(obj)

    # Sort by spatial position (top-to-bottom, left-to-right)
    sources.sort(key=lambda o: (o.row, o.col))

    return sources


def apply_composite_derivation(
    input_grid: np.ndarray,
    derivation: CompositeDerivation,
    mhc: MultiHeadCorrespondence,
    output_shape: Optional[Tuple[int, int]] = None,
    segmentation_mode: SegmentationMode = SegmentationMode.CONNECTIVITY,
    examples: Optional[List[MultiHeadExampleData]] = None,
) -> List[Object]:
    """Apply a composite derivation rule to generate output objects.

    This function takes a discovered composite derivation and applies it to
    generate output objects from an input grid.

    The key insight: we don't need to run correspondence at test time.
    Instead, we:
    1. Find the template object (shape source)
    2. Find color source objects
    3. Generate N outputs with template shape and source colors

    Args:
        input_grid: Input grid to transform
        derivation: The composite derivation rule
        mhc: MultiHeadCorrespondence instance (for context)
        output_shape: Shape of output grid (for positioning)
        segmentation_mode: How to segment input
        examples: Optional training examples for context

    Returns:
        List of output Objects
    """
    # Extract input objects
    input_objects = extract_objects_from_grid(input_grid, segmentation_mode=segmentation_mode)

    # Filter out Grid pseudo-objects
    input_objects = [obj for obj in input_objects if obj.color >= 0]

    # Find template object (shape source)
    template = _find_template_object(input_objects, examples)
    if template is None:
        return []

    # Find color source objects
    color_sources = _find_color_sources(input_objects, template)
    if not color_sources:
        return []

    # Determine output shape if not provided
    if output_shape is None:
        # Estimate from color source arrangement
        if len(color_sources) >= 2:
            row_span = max(o.row for o in color_sources) - min(o.row for o in color_sources)
            col_span = max(o.col for o in color_sources) - min(o.col for o in color_sources)
            horizontal = col_span >= row_span
        else:
            horizontal = True

        if horizontal:
            output_shape = (template.height, template.width * len(color_sources))
        else:
            output_shape = (template.height * len(color_sources), template.width)

    # Determine arrangement from color pixel positions
    if len(color_sources) >= 2:
        row_span = max(o.row for o in color_sources) - min(o.row for o in color_sources)
        col_span = max(o.col for o in color_sources) - min(o.col for o in color_sources)
        horizontal = col_span >= row_span
    else:
        horizontal = True

    # Generate output objects
    output_objects = []

    # Get template shape as relative pixels
    template_min_row = min(r for r, c in template.pixels)
    template_min_col = min(c for r, c in template.pixels)
    template_relative = {(r - template_min_row, c - template_min_col)
                        for r, c in template.pixels}

    # Place copies of template with colors from color sources
    for out_idx, color_obj in enumerate(color_sources):
        # Get position for this output based on binding
        if derivation.color_source.binding.binding_type == BindingType.IDENTITY:
            # Sequential placement
            if horizontal:
                target_row = 0
                target_col = out_idx * template.width
            else:
                target_row = out_idx * template.height
                target_col = 0
        else:
            # Default to sequential
            if horizontal:
                target_row = 0
                target_col = out_idx * template.width
            else:
                target_row = out_idx * template.height
                target_col = 0

        # Create pixel set at new position
        new_pixels = set()
        for r, c in template_relative:
            new_r = r + target_row
            new_c = c + target_col
            if 0 <= new_r < output_shape[0] and 0 <= new_c < output_shape[1]:
                new_pixels.add((new_r, new_c))

        new_obj = Object(
            id=out_idx,
            row=target_row,
            col=target_col,
            height=template.height,
            width=template.width,
            color=color_obj.color,  # Use color from color source
            pixels=new_pixels,
            is_background=False,
            is_divider=False,
        )
        output_objects.append(new_obj)

    return output_objects


def render_objects_to_grid(
    objects: List[Object],
    grid_shape: Tuple[int, int],
    background: int = 0
) -> np.ndarray:
    """Render a list of objects to a grid.

    Args:
        objects: List of Object instances
        grid_shape: (height, width) of output grid
        background: Background color value

    Returns:
        Grid array with objects rendered
    """
    grid = np.full(grid_shape, background, dtype=np.int64)

    for obj in objects:
        for r, c in obj.pixels:
            if 0 <= r < grid_shape[0] and 0 <= c < grid_shape[1]:
                grid[r, c] = obj.color

    return grid


# =============================================================================
# High-Level Discovery Function
# =============================================================================

def discover_multi_head_rule(
    puzzle: Dict,
    segmentation_mode: SegmentationMode = SegmentationMode.CONNECTIVITY,
    verbose: bool = False
) -> Optional[CompositeDerivation]:
    """Discover a composite derivation rule from puzzle training examples.

    This is the main entry point for multi-head correspondence discovery.

    Args:
        puzzle: Puzzle dict with 'train' examples
        segmentation_mode: How to segment grids
        verbose: Print screening details

    Returns:
        CompositeDerivation if a consistent rule is found, None otherwise
    """
    train_examples = puzzle.get('train', [])
    if not train_examples:
        return None

    mhc = MultiHeadCorrespondence()
    example_data_list = []

    for ex in train_examples:
        input_grid = np.array(ex['input'])
        output_grid = np.array(ex['output'])

        # Extract objects
        input_objects = extract_objects_from_grid(input_grid, segmentation_mode=segmentation_mode)
        output_objects = extract_objects_from_grid(output_grid, segmentation_mode=segmentation_mode)

        # Run all heads
        head_results = mhc.find_all(input_grid, output_grid, input_objects, output_objects)

        if verbose:
            print(f"\nExample {len(example_data_list) + 1}:")
            print(mhc.describe_results(head_results))

        example_data_list.append(MultiHeadExampleData(
            input_grid=input_grid,
            output_grid=output_grid,
            input_objects=input_objects,
            output_objects=output_objects,
            head_results=head_results,
        ))

    # Screen to find the derivation rule
    derivation = screen_composite_derivation(example_data_list, verbose)

    return derivation


# =============================================================================
# CLI for Testing
# =============================================================================

def main():
    import argparse
    from puzzle_loader import load_puzzle

    parser = argparse.ArgumentParser(description='Multi-Head Correspondence Module')
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

    print(f"Analyzing puzzle {args.puzzle_id} with multi-head correspondence")
    print(f"Segmentation mode: {seg_mode.name}")
    print("=" * 60)

    # Discover rule
    derivation = discover_multi_head_rule(puzzle, seg_mode, verbose=args.verbose)

    if derivation is None:
        print("\nNo composite derivation rule found")
        return

    print(f"\n{'='*60}")
    print("Discovered Composite Derivation:")
    print(derivation.describe())
    print(f"{'='*60}")

    # Test on training examples
    print("\nTesting on training examples:")

    mhc = MultiHeadCorrespondence()

    for i, ex in enumerate(puzzle['train']):
        input_grid = np.array(ex['input'])
        expected_output = np.array(ex['output'])

        # Apply derivation
        output_objects = apply_composite_derivation(
            input_grid, derivation, mhc,
            output_shape=expected_output.shape,
            segmentation_mode=seg_mode
        )

        predicted_output = render_objects_to_grid(output_objects, expected_output.shape)

        match = np.array_equal(predicted_output, expected_output)
        accuracy = np.mean(predicted_output == expected_output)

        status = "PASS" if match else "FAIL"
        print(f"  Example {i+1}: {status} (pixel accuracy: {accuracy:.1%})")

        if args.verbose and not match:
            print(f"    Expected:\n{expected_output}")
            print(f"    Predicted:\n{predicted_output}")


if __name__ == "__main__":
    main()
