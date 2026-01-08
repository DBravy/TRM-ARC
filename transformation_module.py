#!/usr/bin/env python3
"""
Transformation Module for ARC Puzzle Solver

Discovers transformation rules when object repositioning fails. This module
analyzes correspondences between input and output objects to find rules that
explain how outputs are derived from inputs.

This is a FALLBACK module - only used when the standard object repositioning
approach doesn't achieve 100% accuracy.

Key Concepts:
    - TransformationRule: Complete specification for transforming inputs to output
    - ColorDerivation: How output color is derived from correspondent inputs
    - ShapeDerivation: How output shape is derived from correspondent inputs
    - Position is handled by reusing the anchoring_module

Use Cases:
    - Many input pixels of color C correspond to one output filled with color C
    - Pattern → Color mappings
    - Color → Pattern mappings

Usage:
    from transformation_module import (
        discover_transformation_rules,
        TransformationRule,
        apply_transformation,
    )

    # Discover rules from training examples
    rules = discover_transformation_rules(
        train_examples,
        correspondence_mode="many_to_one"
    )

    # Apply rules to generate test output
    output_grid = apply_transformation(test_input, rules)
"""

import numpy as np
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Set, Tuple, Union
from collections import Counter

# Reuse existing modules
from anchoring_module import (
    AnchorPoint, ALL_ANCHORS,
    discover_relation_to_grid,
    TestObject,
    DiscoveredRelation,
    get_anchor_position,
)
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
from genesis_module import (
    PositionSpec,
    resolve_position,
)


# =============================================================================
# Selection Criterion Types (how correspondents are chosen)
# =============================================================================

class SelectionCriterion(Enum):
    """How to select which input objects become correspondents at test time."""
    MODE_COLOR = auto()         # Select all inputs of the most common color
    LEAST_COMMON_COLOR = auto() # Select all inputs of the least common color
    ALL = auto()                # Select all inputs
    # Future: LARGEST, BY_PATTERN, etc.


# =============================================================================
# Color Derivation Types
# =============================================================================

class ColorDerivationType(Enum):
    """How output color is derived from correspondent inputs."""
    COMMON_COLOR = auto()       # All correspondents share one color -> use it
    MODE_OF_CORRESPONDENTS = auto()  # Most common color among correspondents
    LITERAL = auto()            # Fixed color (same across all examples)
    MODE_OF_CHILDREN = auto()   # Mode color of children within hierarchical region


@dataclass
class ColorDerivation:
    """Specification for how to derive output color."""
    derivation_type: ColorDerivationType
    literal_value: Optional[int] = None  # For LITERAL type
    variance: float = float('inf')  # How consistent across examples

    def derive_color(
        self,
        correspondent_colors: List[int],
        correspondent_objects: Optional[List['Object']] = None
    ) -> int:
        """Derive the output color from correspondent input colors.

        Args:
            correspondent_colors: List of colors from correspondent objects
            correspondent_objects: Optional list of Object instances (needed for MODE_OF_CHILDREN)
        """
        if self.derivation_type == ColorDerivationType.LITERAL:
            return self.literal_value
        elif self.derivation_type == ColorDerivationType.COMMON_COLOR:
            # All should be same color
            if correspondent_colors:
                return correspondent_colors[0]
            return 0
        elif self.derivation_type == ColorDerivationType.MODE_OF_CORRESPONDENTS:
            if correspondent_colors:
                counter = Counter(correspondent_colors)
                return counter.most_common(1)[0][0]
            return 0
        elif self.derivation_type == ColorDerivationType.MODE_OF_CHILDREN:
            # Get mode color from children of correspondent objects
            if correspondent_objects:
                child_colors = []
                for obj in correspondent_objects:
                    if obj.children:
                        for child in obj.children:
                            child_colors.append(child.color)
                if child_colors:
                    counter = Counter(child_colors)
                    return counter.most_common(1)[0][0]
            # Fallback to mode of correspondent colors
            if correspondent_colors:
                counter = Counter(correspondent_colors)
                return counter.most_common(1)[0][0]
            return 0
        return 0

    def describe(self) -> str:
        if self.derivation_type == ColorDerivationType.LITERAL:
            return f"literal({self.literal_value})"
        elif self.derivation_type == ColorDerivationType.COMMON_COLOR:
            return "common_color_of_correspondents"
        elif self.derivation_type == ColorDerivationType.MODE_OF_CORRESPONDENTS:
            return "mode_of_correspondents"
        elif self.derivation_type == ColorDerivationType.MODE_OF_CHILDREN:
            return "mode_of_children"
        return str(self.derivation_type)


# =============================================================================
# Shape Derivation Types
# =============================================================================

class ShapeDerivationType(Enum):
    """How output shape is derived from correspondent inputs."""
    FILL_GRID = auto()          # Fill entire output grid
    BOUNDING_BOX = auto()       # Bounding box of all correspondents
    UNION = auto()              # Union of correspondent shapes
    LITERAL_RECT = auto()       # Fixed rectangle dimensions
    SAME_AS_CORRESPONDENT = auto()  # Use correspondent's bbox (for parallel correspondences)


@dataclass
class ShapeDerivation:
    """Specification for how to derive output shape."""
    derivation_type: ShapeDerivationType
    dimensions: Optional[Tuple[int, int]] = None  # For LITERAL_RECT: (height, width)
    variance: float = float('inf')

    def derive_shape(
        self,
        correspondent_pixels: List[Set[Tuple[int, int]]],
        grid_shape: Tuple[int, int]
    ) -> Set[Tuple[int, int]]:
        """Derive output shape (as pixel set) from correspondent inputs.

        Note: This returns pixels at their absolute positions. For anchor-based
        placement, use derive_relative_shape() instead.
        """
        if self.derivation_type == ShapeDerivationType.FILL_GRID:
            h, w = grid_shape
            return {(r, c) for r in range(h) for c in range(w)}

        elif self.derivation_type == ShapeDerivationType.BOUNDING_BOX:
            all_pixels = set()
            for pixels in correspondent_pixels:
                all_pixels.update(pixels)
            if not all_pixels:
                return set()
            rows = [p[0] for p in all_pixels]
            cols = [p[1] for p in all_pixels]
            min_r, max_r = min(rows), max(rows)
            min_c, max_c = min(cols), max(cols)
            return {(r, c) for r in range(min_r, max_r + 1)
                    for c in range(min_c, max_c + 1)}

        elif self.derivation_type == ShapeDerivationType.UNION:
            all_pixels = set()
            for pixels in correspondent_pixels:
                all_pixels.update(pixels)
            return all_pixels

        elif self.derivation_type == ShapeDerivationType.LITERAL_RECT:
            if self.dimensions:
                h, w = self.dimensions
                return {(r, c) for r in range(h) for c in range(w)}
            return set()

        elif self.derivation_type == ShapeDerivationType.SAME_AS_CORRESPONDENT:
            # Use bounding box of correspondents (same as BOUNDING_BOX but semantically different)
            # This is typically used per-correspondence, so there's usually just one correspondent
            all_pixels = set()
            for pixels in correspondent_pixels:
                all_pixels.update(pixels)
            if not all_pixels:
                return set()
            rows = [p[0] for p in all_pixels]
            cols = [p[1] for p in all_pixels]
            min_r, max_r = min(rows), max(rows)
            min_c, max_c = min(cols), max(cols)
            return {(r, c) for r in range(min_r, max_r + 1)
                    for c in range(min_c, max_c + 1)}

        return set()

    def derive_relative_shape(
        self,
        correspondent_pixels: List[Set[Tuple[int, int]]],
        grid_shape: Tuple[int, int]
    ) -> Tuple[Set[Tuple[int, int]], Tuple[int, int], Tuple[int, int]]:
        """Derive output shape normalized to (0,0) with origin info.

        Returns:
            (relative_pixels, origin, size) where:
            - relative_pixels: Set of (r, c) tuples normalized so min_r=min_c=0
            - origin: (row, col) of where the shape was originally located
            - size: (height, width) of the shape's bounding box
        """
        if self.derivation_type == ShapeDerivationType.FILL_GRID:
            h, w = grid_shape
            pixels = {(r, c) for r in range(h) for c in range(w)}
            return pixels, (0, 0), (h, w)

        elif self.derivation_type == ShapeDerivationType.BOUNDING_BOX:
            all_pixels = set()
            for pixels in correspondent_pixels:
                all_pixels.update(pixels)
            if not all_pixels:
                return set(), (0, 0), (0, 0)
            rows = [p[0] for p in all_pixels]
            cols = [p[1] for p in all_pixels]
            min_r, max_r = min(rows), max(rows)
            min_c, max_c = min(cols), max(cols)
            h, w = max_r - min_r + 1, max_c - min_c + 1
            # Normalize to (0,0)
            relative = {(r - min_r, c - min_c) for r in range(min_r, max_r + 1)
                        for c in range(min_c, max_c + 1)}
            return relative, (min_r, min_c), (h, w)

        elif self.derivation_type == ShapeDerivationType.UNION:
            all_pixels = set()
            for pixels in correspondent_pixels:
                all_pixels.update(pixels)
            if not all_pixels:
                return set(), (0, 0), (0, 0)
            rows = [p[0] for p in all_pixels]
            cols = [p[1] for p in all_pixels]
            min_r, min_c = min(rows), min(cols)
            max_r, max_c = max(rows), max(cols)
            h, w = max_r - min_r + 1, max_c - min_c + 1
            # Normalize to (0,0)
            relative = {(r - min_r, c - min_c) for r, c in all_pixels}
            return relative, (min_r, min_c), (h, w)

        elif self.derivation_type == ShapeDerivationType.LITERAL_RECT:
            if self.dimensions:
                h, w = self.dimensions
                pixels = {(r, c) for r in range(h) for c in range(w)}
                return pixels, (0, 0), (h, w)
            return set(), (0, 0), (0, 0)

        elif self.derivation_type == ShapeDerivationType.SAME_AS_CORRESPONDENT:
            # Use bounding box of correspondents, preserving position
            all_pixels = set()
            for pixels in correspondent_pixels:
                all_pixels.update(pixels)
            if not all_pixels:
                return set(), (0, 0), (0, 0)
            rows = [p[0] for p in all_pixels]
            cols = [p[1] for p in all_pixels]
            min_r, max_r = min(rows), max(rows)
            min_c, max_c = min(cols), max(cols)
            h, w = max_r - min_r + 1, max_c - min_c + 1
            # Normalize to (0,0)
            relative = {(r - min_r, c - min_c) for r in range(min_r, max_r + 1)
                        for c in range(min_c, max_c + 1)}
            return relative, (min_r, min_c), (h, w)

        return set(), (0, 0), (0, 0)

    def describe(self) -> str:
        if self.derivation_type == ShapeDerivationType.FILL_GRID:
            return "fill_grid"
        elif self.derivation_type == ShapeDerivationType.BOUNDING_BOX:
            return "bounding_box"
        elif self.derivation_type == ShapeDerivationType.UNION:
            return "union"
        elif self.derivation_type == ShapeDerivationType.LITERAL_RECT:
            return f"rect({self.dimensions})"
        elif self.derivation_type == ShapeDerivationType.SAME_AS_CORRESPONDENT:
            return "same_as_correspondent"
        return str(self.derivation_type)


# =============================================================================
# Transformation Rule
# =============================================================================

@dataclass
class TransformationRule:
    """Complete rule for transforming input correspondents to output."""
    selection_criterion: SelectionCriterion  # How to select correspondents at test time
    color_derivation: ColorDerivation
    shape_derivation: ShapeDerivation
    # Position determined via anchoring system (like genesis objects)
    position_spec: Optional[PositionSpec] = None

    # Per-correspondence mode: apply rule to each input→output pair independently
    # When True: each input object produces its own output at its own position
    # When False: all correspondents are aggregated to produce one output
    per_correspondence: bool = False

    # Metadata
    total_variance: float = float('inf')
    position_variance: float = float('inf')

    def describe(self) -> str:
        pos_desc = self.position_spec.describe() if self.position_spec else "default"
        per_corr = ", per_correspondence=True" if self.per_correspondence else ""
        return (f"TransformationRule(select={self.selection_criterion.name}, "
                f"color={self.color_derivation.describe()}, "
                f"shape={self.shape_derivation.describe()}, pos={pos_desc}{per_corr})")


# =============================================================================
# Screening Functions
# =============================================================================

def screen_selection_criterion(
    examples: List['ExampleData'],
    verbose: bool = False
) -> SelectionCriterion:
    """
    Screen selection criterion hypotheses.

    Analyzes how correspondents relate to all input objects to determine
    the selection rule used at test time.

    Checks:
    - MODE_COLOR: correspondents are all pixels of the most common input color
    - ALL: all inputs are correspondents
    """
    hypotheses = [
        SelectionCriterion.MODE_COLOR,
        SelectionCriterion.ALL,
    ]

    best_criterion = SelectionCriterion.ALL
    best_score = 0

    for criterion in hypotheses:
        matches = 0
        total = 0

        for ex in examples:
            if not ex.correspondences or not ex.input_objects:
                continue

            # Get all input colors
            all_input_colors = [obj.color for obj in ex.input_objects]

            # Get correspondent indices
            correspondent_indices = set(in_idx for in_idx, out_idx, score in ex.correspondences)
            correspondent_colors = [ex.input_objects[i].color for i in correspondent_indices
                                    if i < len(ex.input_objects)]

            if criterion == SelectionCriterion.MODE_COLOR:
                # Check if correspondents are all of the mode color
                if all_input_colors:
                    mode_color = Counter(all_input_colors).most_common(1)[0][0]
                    # Correspondents should all be mode color
                    all_are_mode = all(c == mode_color for c in correspondent_colors)
                    # And they should be ALL inputs of that color
                    mode_count = sum(1 for c in all_input_colors if c == mode_color)
                    if all_are_mode and len(correspondent_colors) == mode_count:
                        matches += 1
                total += 1

            elif criterion == SelectionCriterion.ALL:
                # Check if all inputs are correspondents
                if len(correspondent_indices) == len(ex.input_objects):
                    matches += 1
                total += 1

        score = matches / total if total > 0 else 0

        if verbose:
            print(f"  Selection {criterion.name}: score={score:.2f} ({matches}/{total})")

        if score > best_score:
            best_score = score
            best_criterion = criterion

    return best_criterion


@dataclass
class ExampleData:
    """Data extracted from one training example for screening."""
    input_grid: np.ndarray
    output_grid: np.ndarray
    # Correspondences: list of (input_obj_idx, output_obj_idx, score)
    correspondences: List[Tuple[int, int, float]]
    input_objects: List[Object]
    output_objects: List[Object]


def screen_color_derivation(
    examples: List[ExampleData],
    verbose: bool = False
) -> ColorDerivation:
    """
    Screen color derivation hypotheses to find one with zero variance.

    For each example:
    - Get colors of all input objects that correspond to the output
    - Get the output object's color
    - Check which derivation rule correctly predicts output color

    Returns the derivation with lowest variance (ideally 0).
    """
    hypotheses = [
        ColorDerivationType.COMMON_COLOR,
        ColorDerivationType.MODE_OF_CORRESPONDENTS,
        ColorDerivationType.MODE_OF_CHILDREN,
        ColorDerivationType.LITERAL,
    ]

    best_derivation = None
    best_variance = float('inf')

    for hyp_type in hypotheses:
        errors = []
        literal_values = []  # For LITERAL type

        for ex in examples:
            if not ex.correspondences or not ex.output_objects:
                continue

            # Group correspondences by output object
            output_to_inputs: Dict[int, List[int]] = {}
            for in_idx, out_idx, score in ex.correspondences:
                if out_idx not in output_to_inputs:
                    output_to_inputs[out_idx] = []
                output_to_inputs[out_idx].append(in_idx)

            # For each output object, check the derivation
            for out_idx, in_indices in output_to_inputs.items():
                if out_idx >= len(ex.output_objects):
                    continue

                out_obj = ex.output_objects[out_idx]
                actual_color = out_obj.color

                # Get correspondent colors
                correspondent_colors = []
                for in_idx in in_indices:
                    if in_idx < len(ex.input_objects):
                        correspondent_colors.append(ex.input_objects[in_idx].color)

                if not correspondent_colors:
                    continue

                # Predict color based on hypothesis
                if hyp_type == ColorDerivationType.COMMON_COLOR:
                    # Check if all correspondents have same color
                    if len(set(correspondent_colors)) == 1:
                        predicted = correspondent_colors[0]
                    else:
                        predicted = -1  # Not applicable

                elif hyp_type == ColorDerivationType.MODE_OF_CORRESPONDENTS:
                    counter = Counter(correspondent_colors)
                    predicted = counter.most_common(1)[0][0]

                elif hyp_type == ColorDerivationType.MODE_OF_CHILDREN:
                    # Get mode color from children of correspondent objects
                    child_colors = []
                    for in_idx in in_indices:
                        if in_idx < len(ex.input_objects):
                            obj = ex.input_objects[in_idx]
                            if obj.children:
                                for child in obj.children:
                                    child_colors.append(child.color)
                    if child_colors:
                        counter = Counter(child_colors)
                        predicted = counter.most_common(1)[0][0]
                    else:
                        predicted = -1  # No children, not applicable

                elif hyp_type == ColorDerivationType.LITERAL:
                    literal_values.append(actual_color)
                    predicted = actual_color  # Will check variance separately

                # Compute error
                if predicted == actual_color:
                    errors.append(0)
                else:
                    errors.append(1)

        # Compute variance
        if hyp_type == ColorDerivationType.LITERAL:
            # For LITERAL, variance is whether the color is consistent
            if literal_values:
                variance = np.var(literal_values)
            else:
                variance = float('inf')
        else:
            # For other types, variance is error rate
            if errors:
                variance = np.mean(errors)
            else:
                variance = float('inf')

        if verbose:
            print(f"  Color {hyp_type.name}: variance={variance:.4f}")

        if variance < best_variance:
            best_variance = variance
            if hyp_type == ColorDerivationType.LITERAL and literal_values:
                best_derivation = ColorDerivation(
                    derivation_type=hyp_type,
                    literal_value=int(np.round(np.mean(literal_values))),
                    variance=variance
                )
            else:
                best_derivation = ColorDerivation(
                    derivation_type=hyp_type,
                    variance=variance
                )

    return best_derivation or ColorDerivation(
        derivation_type=ColorDerivationType.MODE_OF_CORRESPONDENTS,
        variance=float('inf')
    )


def screen_shape_derivation(
    examples: List[ExampleData],
    verbose: bool = False
) -> ShapeDerivation:
    """
    Screen shape derivation hypotheses to find one with zero variance.

    Checks if output shape matches:
    - FILL_GRID: output fills entire grid
    - BOUNDING_BOX: output is bbox of correspondent inputs
    - UNION: output is union of correspondent shapes
    - SAME_AS_CORRESPONDENT: output matches correspondent's bbox (for parallel correspondences)
    """
    hypotheses = [
        ShapeDerivationType.FILL_GRID,
        ShapeDerivationType.BOUNDING_BOX,
        ShapeDerivationType.UNION,
        ShapeDerivationType.SAME_AS_CORRESPONDENT,
    ]

    best_derivation = None
    best_variance = float('inf')

    for hyp_type in hypotheses:
        errors = []

        for ex in examples:
            if not ex.correspondences or not ex.output_objects:
                continue

            # Group correspondences by output object
            output_to_inputs: Dict[int, List[int]] = {}
            for in_idx, out_idx, score in ex.correspondences:
                if out_idx not in output_to_inputs:
                    output_to_inputs[out_idx] = []
                output_to_inputs[out_idx].append(in_idx)

            grid_shape = ex.output_grid.shape

            for out_idx, in_indices in output_to_inputs.items():
                if out_idx >= len(ex.output_objects):
                    continue

                out_obj = ex.output_objects[out_idx]
                actual_pixels = set(out_obj.pixels)

                # Get correspondent pixel sets
                correspondent_pixels = []
                for in_idx in in_indices:
                    if in_idx < len(ex.input_objects):
                        correspondent_pixels.append(set(ex.input_objects[in_idx].pixels))

                # Predict shape based on hypothesis
                derivation = ShapeDerivation(derivation_type=hyp_type)
                predicted_pixels = derivation.derive_shape(correspondent_pixels, grid_shape)

                # Compute IoU as similarity metric
                intersection = len(actual_pixels & predicted_pixels)
                union = len(actual_pixels | predicted_pixels)
                iou = intersection / union if union > 0 else 0

                # Error is 1 - IoU
                errors.append(1 - iou)

        # Variance is mean error
        if errors:
            variance = np.mean(errors)
        else:
            variance = float('inf')

        if verbose:
            print(f"  Shape {hyp_type.name}: variance={variance:.4f}")

        if variance < best_variance:
            best_variance = variance
            best_derivation = ShapeDerivation(
                derivation_type=hyp_type,
                variance=variance
            )

    return best_derivation or ShapeDerivation(
        derivation_type=ShapeDerivationType.FILL_GRID,
        variance=float('inf')
    )


def screen_position_hypotheses(
    examples: List['ExampleData'],
    shape_derivation: ShapeDerivation,
    verbose: bool = False
) -> Tuple[Optional[PositionSpec], float]:
    """
    Screen position hypotheses for the transformed object.

    Similar to genesis_module.screen_position_hypotheses(), this tests different
    anchor-based position hypotheses to find the one with lowest variance.

    Args:
        examples: List of ExampleData with correspondences and grids
        shape_derivation: The shape derivation being used (needed to compute shape)
        verbose: Print screening details

    Returns:
        (best_position_spec, variance) tuple
    """
    if not examples:
        return None, float('inf')

    # Collect transformed object positions and sizes for each example
    positions = []  # List of (row, col) top-left positions
    sizes = []      # List of (height, width)
    grid_shapes = []
    input_objects_list = []

    for ex in examples:
        # Get correspondent pixels
        correspondent_pixels = []
        for in_idx, out_idx, score in ex.correspondences:
            if in_idx < len(ex.input_objects):
                correspondent_pixels.append(set(ex.input_objects[in_idx].pixels))

        if not correspondent_pixels:
            continue

        # Derive relative shape to get origin (where output object should be)
        _, origin, size = shape_derivation.derive_relative_shape(
            correspondent_pixels, ex.output_grid.shape
        )

        positions.append(origin)
        sizes.append(size)
        grid_shapes.append(ex.output_grid.shape)
        input_objects_list.append(ex.input_objects)

    if not positions:
        return None, float('inf')

    num_examples = len(positions)
    hypotheses = []

    # Hypothesis 1: GRID_ANCHOR - position relative to grid corners/edges/center
    for grid_anchor in ALL_ANCHORS:
        for obj_anchor in ALL_ANCHORS:
            offsets = []

            for ex_idx in range(num_examples):
                pos = positions[ex_idx]
                size = sizes[ex_idx]
                grid_h, grid_w = grid_shapes[ex_idx]

                # Get object anchor position
                obj_anchor_pos = get_anchor_position(
                    (pos[0], pos[1]), (size[0], size[1]), obj_anchor
                )

                # Get grid anchor position
                grid_anchor_pos = get_anchor_position(
                    (0, 0), (grid_h, grid_w), grid_anchor
                )

                # Offset = object_anchor - grid_anchor
                offset = (obj_anchor_pos[0] - grid_anchor_pos[0],
                          obj_anchor_pos[1] - grid_anchor_pos[1])
                offsets.append(offset)

            # Compute variance
            if offsets:
                row_offsets = [o[0] for o in offsets]
                col_offsets = [o[1] for o in offsets]
                variance = np.var(row_offsets) + np.var(col_offsets)
                mean_offset = (int(round(np.mean(row_offsets))),
                              int(round(np.mean(col_offsets))))

                spec = PositionSpec.grid_relative(
                    grid_anchor=grid_anchor,
                    object_anchor=obj_anchor,
                    offset=mean_offset
                )
                hypotheses.append((spec, variance))

    # Hypothesis 2: OBJECT_RELATIVE - position relative to input objects
    if input_objects_list and input_objects_list[0]:
        max_obj_idx = min(len(objs) for objs in input_objects_list)

        for ref_idx in range(max_obj_idx):
            for src_anchor in ALL_ANCHORS:
                for tgt_anchor in ALL_ANCHORS:
                    offsets = []

                    for ex_idx in range(num_examples):
                        pos = positions[ex_idx]
                        size = sizes[ex_idx]
                        input_objs = input_objects_list[ex_idx]

                        if ref_idx >= len(input_objs):
                            continue

                        ref_obj = input_objs[ref_idx]

                        # Get object anchor position (on the transformed object)
                        obj_anchor_pos = get_anchor_position(
                            (pos[0], pos[1]), (size[0], size[1]), src_anchor
                        )

                        # Get reference object anchor position
                        ref_anchor_pos = get_anchor_position(
                            (ref_obj.row, ref_obj.col), (ref_obj.height, ref_obj.width), tgt_anchor
                        )

                        offset = (obj_anchor_pos[0] - ref_anchor_pos[0],
                                  obj_anchor_pos[1] - ref_anchor_pos[1])
                        offsets.append(offset)

                    if len(offsets) == num_examples:
                        row_offsets = [o[0] for o in offsets]
                        col_offsets = [o[1] for o in offsets]
                        variance = np.var(row_offsets) + np.var(col_offsets)
                        mean_offset = (int(round(np.mean(row_offsets))),
                                      int(round(np.mean(col_offsets))))

                        spec = PositionSpec.object_relative(
                            ref_idx=ref_idx,
                            source_anchor=src_anchor,
                            target_anchor=tgt_anchor,
                            offset=mean_offset
                        )
                        hypotheses.append((spec, variance))

    # Sort by variance and return best
    hypotheses.sort(key=lambda x: x[1])

    if verbose and hypotheses:
        print(f"\nPosition screening results (top 5):")
        for spec, var in hypotheses[:5]:
            print(f"  {spec.describe()}: variance={var:.6f}")

    if hypotheses:
        return hypotheses[0]

    return None, float('inf')


# =============================================================================
# Main Discovery Function
# =============================================================================

def discover_transformation_rules(
    puzzles: Dict,
    puzzle_id: str,
    input_segmentation_mode: SegmentationMode = SegmentationMode.PIXEL,
    output_segmentation_mode: SegmentationMode = SegmentationMode.CONNECTIVITY,
    correspondence_mode: CorrespondenceMode = "many_to_one",
    correspondence_margin: float = DEFAULT_MARGIN,
    children_segmentation_mode: Optional[SegmentationMode] = None,
    verbose: bool = False
) -> Optional[TransformationRule]:
    """
    Discover transformation rules from training examples.

    This function:
    1. Extracts objects from input/output grids with specified segmentation
    2. Finds correspondences between input and output objects
    3. Screens color derivation hypotheses
    4. Screens shape derivation hypotheses
    5. Returns the best transformation rule

    Args:
        puzzles: Dict of puzzles
        puzzle_id: Which puzzle to analyze
        input_segmentation_mode: How to segment input grids
        output_segmentation_mode: How to segment output grids
        correspondence_mode: How to match objects
        correspondence_margin: Margin for correspondence matching
        children_segmentation_mode: For divider-based grids, how to segment
                                   children within regions. Use PIXEL for
                                   accurate pixel-based mode counting.
        verbose: Print debug info

    Returns:
        TransformationRule if a consistent rule is found, None otherwise
    """
    if puzzle_id not in puzzles:
        return None

    puzzle = puzzles[puzzle_id]
    train_examples = puzzle.get('train', [])

    if not train_examples:
        return None

    # Extract data from each training example
    example_data_list: List[ExampleData] = []

    for ex in train_examples:
        input_grid = np.array(ex['input'])
        output_grid = np.array(ex['output'])

        # Extract objects with specified segmentation modes
        input_objects = extract_objects_from_grid(
            input_grid,
            segmentation_mode=input_segmentation_mode,
            children_segmentation_mode=children_segmentation_mode
        )
        output_objects = extract_objects_from_grid(
            output_grid,
            segmentation_mode=output_segmentation_mode
        )

        if verbose:
            print(f"Example: {len(input_objects)} input objects, {len(output_objects)} output objects")

        # Find correspondences
        correspondences, in_matchable, out_matchable = find_correspondences(
            input_grid, output_grid,
            input_objects, output_objects,
            mode=correspondence_mode,
            margin=correspondence_margin
        )

        if verbose:
            print(f"  Correspondences: {correspondences}")

        example_data_list.append(ExampleData(
            input_grid=input_grid,
            output_grid=output_grid,
            correspondences=correspondences,
            input_objects=in_matchable,
            output_objects=out_matchable
        ))

    if not example_data_list:
        return None

    # Screen selection criterion (how to choose correspondents at test time)
    if verbose:
        print("\nScreening selection criterion:")
    selection_criterion = screen_selection_criterion(example_data_list, verbose)

    # Screen color derivation
    if verbose:
        print("\nScreening color derivation:")
    color_derivation = screen_color_derivation(example_data_list, verbose)

    # Screen shape derivation
    if verbose:
        print("\nScreening shape derivation:")
    shape_derivation = screen_shape_derivation(example_data_list, verbose)

    # Screen position hypotheses using the anchoring system
    if verbose:
        print("\nScreening position hypotheses:")
    position_spec, position_variance = screen_position_hypotheses(
        example_data_list, shape_derivation, verbose
    )

    # For FILL_GRID with zero variance, default to grid top-left
    if position_spec is None:
        position_spec = PositionSpec.grid_relative(
            grid_anchor=AnchorPoint.TOP_LEFT,
            object_anchor=AnchorPoint.TOP_LEFT,
            offset=(0, 0)
        )
        position_variance = 0.0

    if verbose:
        print(f"  Best position: {position_spec.describe()} (variance={position_variance:.6f})")

    # Always use per_correspondence mode: each input object produces its own output
    per_correspondence = True

    # Compute total variance
    total_variance = color_derivation.variance + shape_derivation.variance + position_variance

    rule = TransformationRule(
        selection_criterion=selection_criterion,
        color_derivation=color_derivation,
        shape_derivation=shape_derivation,
        position_spec=position_spec,
        per_correspondence=per_correspondence,
        total_variance=total_variance,
        position_variance=position_variance
    )

    if verbose:
        print(f"\nDiscovered rule: {rule.describe()}")
        print(f"Total variance: {total_variance:.4f}")

    return rule


# =============================================================================
# Application Functions
# =============================================================================

def apply_transformation_to_objects(
    input_grid: np.ndarray,
    rule: TransformationRule,
    input_segmentation_mode: SegmentationMode = SegmentationMode.PIXEL,
    output_shape: Optional[Tuple[int, int]] = None,
    children_segmentation_mode: Optional[SegmentationMode] = None
) -> List[Object]:
    """
    Apply a transformation rule to generate output objects (not grid).

    This function returns Object instances that can be placed on a grid by the
    standard placement/evaluation pipeline, enabling integration with the
    existing visualization and testing system.

    The object's position is determined by the rule's position_spec using the
    same anchoring system as genesis objects.

    When per_correspondence is True, the rule is applied to each input object
    independently, producing one output object per input object.

    Args:
        input_grid: The input grid to transform
        rule: The transformation rule to apply
        input_segmentation_mode: How to segment the input
        output_shape: Shape of output grid (for FILL_GRID derivation)
        children_segmentation_mode: For divider-based grids, how to segment
                                   children within regions.

    Returns:
        List of Object instances representing the transformed output.
    """
    if output_shape is None:
        output_shape = input_grid.shape

    # Segment input
    input_objects = extract_objects_from_grid(
        input_grid,
        segmentation_mode=input_segmentation_mode,
        children_segmentation_mode=children_segmentation_mode
    )

    # Apply selection criterion to determine which inputs are "correspondents"
    all_colors = [obj.color for obj in input_objects]

    if rule.selection_criterion == SelectionCriterion.MODE_COLOR:
        # Select all inputs of the most common color
        if all_colors:
            mode_color = Counter(all_colors).most_common(1)[0][0]
            selected_objects = [obj for obj in input_objects if obj.color == mode_color]
        else:
            selected_objects = input_objects
    elif rule.selection_criterion == SelectionCriterion.ALL:
        selected_objects = input_objects
    else:
        selected_objects = input_objects

    # Handle per_correspondence mode: apply rule to each input object independently
    if rule.per_correspondence:
        output_objects = []
        for obj_idx, input_obj in enumerate(selected_objects):
            # Derive color for this specific object
            correspondent_colors = [input_obj.color]
            correspondent_objects = [input_obj]
            output_color = rule.color_derivation.derive_color(
                correspondent_colors, correspondent_objects
            )

            # For SAME_AS_CORRESPONDENT, use input object's position and size
            if rule.shape_derivation.derivation_type == ShapeDerivationType.SAME_AS_CORRESPONDENT:
                # Output fills the same region as the input object
                target_row = input_obj.row
                target_col = input_obj.col
                shape_size = (input_obj.height, input_obj.width)
                # Create filled rectangle at same position
                output_pixels = {
                    (r, c)
                    for r in range(target_row, target_row + input_obj.height)
                    for c in range(target_col, target_col + input_obj.width)
                }
            else:
                # Use standard shape derivation for single object
                correspondent_pixels = [set(input_obj.pixels)]
                relative_pixels, origin, shape_size = rule.shape_derivation.derive_relative_shape(
                    correspondent_pixels, output_shape
                )
                if not relative_pixels:
                    continue
                target_row, target_col = origin
                output_pixels = {(r + target_row, c + target_col) for r, c in relative_pixels}

            transformed_obj = Object(
                id=obj_idx,
                row=target_row,
                col=target_col,
                height=shape_size[0],
                width=shape_size[1],
                color=output_color,
                pixels=output_pixels,
                is_background=False,
                is_divider=False,
            )
            output_objects.append(transformed_obj)

        return output_objects

    # Standard mode: aggregate all correspondents into one output
    # Get colors and pixels from selected objects
    correspondent_colors = [obj.color for obj in selected_objects]
    correspondent_pixels = [set(obj.pixels) for obj in selected_objects]

    # Derive color
    output_color = rule.color_derivation.derive_color(correspondent_colors, selected_objects)

    # Derive relative shape (normalized to (0,0))
    relative_pixels, _, shape_size = rule.shape_derivation.derive_relative_shape(
        correspondent_pixels, output_shape
    )

    # If no pixels, return empty list
    if not relative_pixels:
        return []

    # Resolve position using the anchoring system
    if rule.position_spec is not None:
        # Use resolve_position from genesis_module
        target_row, target_col = resolve_position(
            rule.position_spec,
            shape_size=shape_size,
            grid_height=output_shape[0],
            grid_width=output_shape[1],
            input_objects=input_objects,
            output_objects_so_far=None,
            regions=None
        )
    else:
        # Default to (0, 0) if no position spec
        target_row, target_col = 0, 0

    # Shift relative pixels to the resolved position
    output_pixels = {(r + target_row, c + target_col) for r, c in relative_pixels}

    # Create Object instance at the resolved position
    transformed_obj = Object(
        id=0,
        row=target_row,
        col=target_col,
        height=shape_size[0],
        width=shape_size[1],
        color=output_color,
        pixels=output_pixels,
        is_background=False,
        is_divider=False,
    )

    return [transformed_obj]


def apply_transformation(
    input_grid: np.ndarray,
    rule: TransformationRule,
    input_segmentation_mode: SegmentationMode = SegmentationMode.PIXEL,
    output_segmentation_mode: SegmentationMode = SegmentationMode.CONNECTIVITY,
    correspondence_mode: CorrespondenceMode = "many_to_one",
    correspondence_margin: float = DEFAULT_MARGIN,
    output_shape: Optional[Tuple[int, int]] = None,
    children_segmentation_mode: Optional[SegmentationMode] = None
) -> np.ndarray:
    """
    Apply a transformation rule to generate output grid from input.

    This is a convenience wrapper that calls apply_transformation_to_objects()
    and renders the resulting objects to a grid.

    Args:
        input_grid: The input grid to transform
        rule: The transformation rule to apply
        input_segmentation_mode: How to segment the input
        output_segmentation_mode: (unused for generation, but kept for API consistency)
        correspondence_mode: (unused for generation)
        correspondence_margin: (unused for generation)
        output_shape: Shape of output grid (defaults to input shape)
        children_segmentation_mode: For divider-based grids, how to segment children

    Returns:
        Generated output grid
    """
    if output_shape is None:
        output_shape = input_grid.shape

    # Get transformed objects
    output_objects = apply_transformation_to_objects(
        input_grid, rule, input_segmentation_mode, output_shape,
        children_segmentation_mode=children_segmentation_mode
    )

    # Create output grid (start with zeros/background)
    output_grid = np.zeros(output_shape, dtype=input_grid.dtype)

    # Render each object onto the grid
    for obj in output_objects:
        for r, c in obj.pixels:
            if 0 <= r < output_shape[0] and 0 <= c < output_shape[1]:
                output_grid[r, c] = obj.color

    return output_grid


# =============================================================================
# CLI for testing
# =============================================================================

def main():
    import argparse
    from puzzle_loader import load_puzzles, load_puzzle

    parser = argparse.ArgumentParser(description='Transformation Module')
    parser.add_argument('--puzzle-id', type=str, required=True, help='Puzzle ID')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    parser.add_argument('--input-segmentation-mode', type=str, default='connectivity',
                        choices=['connectivity', 'pixel', 'color'],
                        help='Segmentation mode for input')
    parser.add_argument('--output-segmentation-mode', type=str, default='connectivity',
                        choices=['connectivity', 'pixel', 'color'],
                        help='Segmentation mode for output')
    parser.add_argument('--children-segmentation-mode', type=str, default=None,
                        choices=['connectivity', 'pixel', 'color'],
                        help='Segmentation mode for children within regions (for divider-based grids)')
    parser.add_argument('--correspondence-mode', type=str, default='one_to_one',
                        choices=['one_to_one', 'many_to_one', 'one_to_many'],
                        help='Correspondence mode')

    args = parser.parse_args()

    # Map string to enum
    seg_mode_map = {
        'connectivity': SegmentationMode.CONNECTIVITY,
        'pixel': SegmentationMode.PIXEL,
        'color': SegmentationMode.COLOR,
    }
    input_seg = seg_mode_map[args.input_segmentation_mode]
    output_seg = seg_mode_map[args.output_segmentation_mode]
    children_seg = seg_mode_map.get(args.children_segmentation_mode) if args.children_segmentation_mode else None

    # Load puzzle (handle synthetic puzzles with syn_ prefix)
    if args.puzzle_id.startswith('syn_'):
        puzzle = load_puzzle(args.puzzle_id)
        puzzles = {args.puzzle_id: puzzle}
    else:
        puzzles = load_puzzles()
        puzzle = puzzles.get(args.puzzle_id)
        if puzzle is None:
            print(f"Puzzle {args.puzzle_id} not found")
            return

    print(f"Discovering transformation rules for puzzle {args.puzzle_id}")
    print(f"Input segmentation: {input_seg.name}")
    print(f"Output segmentation: {output_seg.name}")
    if children_seg:
        print(f"Children segmentation: {children_seg.name}")
    print(f"Correspondence mode: {args.correspondence_mode}")
    print()

    # Discover rules
    rule = discover_transformation_rules(
        puzzles, args.puzzle_id,
        input_segmentation_mode=input_seg,
        output_segmentation_mode=output_seg,
        correspondence_mode=args.correspondence_mode,
        children_segmentation_mode=children_seg,
        verbose=args.verbose
    )

    if rule is None:
        print("No transformation rule discovered")
        return

    print(f"\n{'='*60}")
    print(f"Discovered Rule: {rule.describe()}")
    print(f"Total Variance: {rule.total_variance:.4f}")
    print(f"{'='*60}")

    # Test on training examples
    print("\nTesting on training examples:")

    for i, ex in enumerate(puzzle['train']):
        input_grid = np.array(ex['input'])
        expected_output = np.array(ex['output'])

        predicted_output = apply_transformation(
            input_grid, rule,
            input_segmentation_mode=input_seg,
            output_shape=expected_output.shape,
            children_segmentation_mode=children_seg
        )

        match = np.array_equal(predicted_output, expected_output)
        accuracy = np.mean(predicted_output == expected_output)

        status = "PASS" if match else "FAIL"
        print(f"  Example {i+1}: {status} (pixel accuracy: {accuracy:.1%})")

        if args.verbose and not match:
            print(f"    Input:\n{input_grid}")
            print(f"    Expected:\n{expected_output}")
            print(f"    Predicted:\n{predicted_output}")

    # Test on test examples
    if puzzle.get('test'):
        print("\nTesting on test examples:")
        for i, ex in enumerate(puzzle['test']):
            input_grid = np.array(ex['input'])
            expected_output = ex.get('output')

            # For test, use expected shape if available, else use input shape
            out_shape = np.array(expected_output).shape if expected_output else input_grid.shape
            predicted_output = apply_transformation(
                input_grid, rule,
                input_segmentation_mode=input_seg,
                output_shape=out_shape,
                children_segmentation_mode=children_seg
            )

            if expected_output:
                expected = np.array(expected_output)
                match = np.array_equal(predicted_output, expected)
                accuracy = np.mean(predicted_output == expected)
                status = "PASS" if match else "FAIL"
                print(f"  Test {i+1}: {status} (pixel accuracy: {accuracy:.1%})")
                if args.verbose and not match:
                    print(f"    Expected:\n{expected}")
                    print(f"    Predicted:\n{predicted_output}")
            else:
                print(f"  Test {i+1} prediction:")
                print(predicted_output)


if __name__ == "__main__":
    main()
