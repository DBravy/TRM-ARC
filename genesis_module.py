#!/usr/bin/env python3
"""
Genesis Module for ARC Puzzle Solver

Handles detection and specification of novel objects - objects that appear in
outputs but have no corresponding input object. This module provides:

1. Novel object detection via correspondence analysis
2. Object specifications (color, shape, position)
3. Hypothesis screening to discover genesis rules
4. Object rendering from specifications

Key Concepts:
    - NovelObject: An output object with no input correspondence
    - ObjectSpec: Complete specification for generating an object
    - ColorSpec: How to derive the object's color
    - ShapeSpec: What shape the object should have
    - PositionSpec: Where to place the object

Usage:
    from genesis_module import (
        find_novel_objects,
        discover_genesis_rules,
        ObjectSpec,
        ColorSpec,
        ShapeSpec,
        PositionSpec,
    )

    # Detect novel objects
    novel = find_novel_objects(input_objects, output_objects, input_grid, output_grid)

    # Discover genesis rules from training examples
    result = discover_genesis_rules(examples)
"""

import numpy as np
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Set, Tuple, Union, TYPE_CHECKING

if TYPE_CHECKING:
    from object_module import Object
    from aggregation_module import RegionAggregation

# Import anchor system from existing module
from anchoring_module import AnchorPoint, get_anchor_offset, get_anchor_position, ALL_ANCHORS
from correspondence_module import find_unmatched_objects, CorrespondenceMode, DEFAULT_MARGIN
from object_module import SegmentationMode, SegmentationStrategy
from puzzle_loader import load_puzzle as _load_puzzle


# =============================================================================
# Shape Specifications
# =============================================================================

class ShapeType(Enum):
    """Types of shapes that can be generated."""
    PIXEL = auto()         # Single pixel
    HLINE = auto()         # Horizontal line
    VLINE = auto()         # Vertical line
    RECT = auto()          # Filled rectangle
    RECT_OUTLINE = auto()  # Rectangle outline only
    TEMPLATE = auto()      # Copy shape from input object


@dataclass
class ShapeSpec:
    """Specification for the shape of a generated object.

    Attributes:
        shape_type: The type of shape to generate
        dimensions: For HLINE: (1, width); for VLINE: (height, 1);
                    for RECT/RECT_OUTLINE: (height, width)
        template_source_idx: For TEMPLATE, the index of source object
        template_mask: For TEMPLATE, the actual mask (populated at runtime)
    """
    shape_type: ShapeType
    dimensions: Optional[Tuple[int, int]] = None
    template_source_idx: Optional[int] = None
    template_mask: Optional[np.ndarray] = None

    @classmethod
    def pixel(cls) -> 'ShapeSpec':
        """Single pixel shape."""
        return cls(shape_type=ShapeType.PIXEL, dimensions=(1, 1))

    @classmethod
    def hline(cls, width: int) -> 'ShapeSpec':
        """Horizontal line of given width."""
        return cls(shape_type=ShapeType.HLINE, dimensions=(1, width))

    @classmethod
    def vline(cls, height: int) -> 'ShapeSpec':
        """Vertical line of given height."""
        return cls(shape_type=ShapeType.VLINE, dimensions=(height, 1))

    @classmethod
    def rect(cls, height: int, width: int) -> 'ShapeSpec':
        """Filled rectangle."""
        return cls(shape_type=ShapeType.RECT, dimensions=(height, width))

    @classmethod
    def rect_outline(cls, height: int, width: int) -> 'ShapeSpec':
        """Rectangle outline (hollow)."""
        return cls(shape_type=ShapeType.RECT_OUTLINE, dimensions=(height, width))

    @classmethod
    def template(cls, source_idx: int, mask: Optional[np.ndarray] = None) -> 'ShapeSpec':
        """Copy shape from another object."""
        return cls(
            shape_type=ShapeType.TEMPLATE,
            template_source_idx=source_idx,
            template_mask=mask
        )

    def get_size(self) -> Tuple[int, int]:
        """Get (height, width) of this shape."""
        if self.dimensions is not None:
            return self.dimensions
        if self.template_mask is not None:
            return self.template_mask.shape
        return (1, 1)


# =============================================================================
# Color Specifications
# =============================================================================

class ColorSourceType(Enum):
    """How to determine the color of a generated object."""
    LITERAL = auto()         # Explicit color value (0-9)
    MODE_OF_REGION = auto()  # Most common color in a region
    FROM_OBJECT = auto()     # Color of a specific object
    NOT_IN_INPUT = auto()    # A color not present in input grid
    COMPLEMENT = auto()      # Different from a reference color


@dataclass
class ColorSpec:
    """Specification for determining object color.

    Attributes:
        source_type: How to derive the color
        literal_value: For LITERAL, the color (0-9)
        region_idx: For MODE_OF_REGION, index of region (-1 = whole grid)
        region_name: For MODE_OF_REGION, name like 'top_half', 'bottom_half'
        object_idx: For FROM_OBJECT, index of object to copy color from
        not_in_input_rank: For NOT_IN_INPUT, which absent color (0=first)
        reference_color: For COMPLEMENT, the color to differ from
    """
    source_type: ColorSourceType
    literal_value: Optional[int] = None
    region_idx: Optional[int] = None
    region_name: Optional[str] = None
    object_idx: Optional[int] = None
    not_in_input_rank: int = 0
    reference_color: Optional[int] = None

    @classmethod
    def literal(cls, color: int) -> 'ColorSpec':
        """Explicit color value."""
        return cls(source_type=ColorSourceType.LITERAL, literal_value=color)

    @classmethod
    def mode_of_region(cls, region_idx: int) -> 'ColorSpec':
        """Mode color of a region (by index)."""
        return cls(source_type=ColorSourceType.MODE_OF_REGION, region_idx=region_idx)

    @classmethod
    def mode_of_region_by_name(cls, name: str) -> 'ColorSpec':
        """Mode color of a named region (top_half, bottom_half, etc.)."""
        return cls(source_type=ColorSourceType.MODE_OF_REGION, region_name=name)

    @classmethod
    def mode_of_grid(cls) -> 'ColorSpec':
        """Mode color of the entire grid."""
        return cls(source_type=ColorSourceType.MODE_OF_REGION, region_idx=-1)

    @classmethod
    def from_object(cls, obj_idx: int) -> 'ColorSpec':
        """Copy color from a specific object."""
        return cls(source_type=ColorSourceType.FROM_OBJECT, object_idx=obj_idx)

    @classmethod
    def not_in_input(cls, rank: int = 0) -> 'ColorSpec':
        """A color not present in input (rank 0 = first absent color)."""
        return cls(source_type=ColorSourceType.NOT_IN_INPUT, not_in_input_rank=rank)

    @classmethod
    def complement(cls, ref_color: int) -> 'ColorSpec':
        """A color different from the reference."""
        return cls(source_type=ColorSourceType.COMPLEMENT, reference_color=ref_color)


# =============================================================================
# Position Specifications
# =============================================================================

class PositionType(Enum):
    """How to determine the position of a generated object."""
    GRID_ANCHOR = auto()      # Relative to grid corner/edge/center
    OBJECT_RELATIVE = auto()  # Relative to another object's anchor
    AT_INTERSECTION = auto()  # At intersection of row/col from objects
    ABSOLUTE = auto()         # Fixed (row, col) position
    PARENT_RELATIVE = auto()  # Relative to a parent region's anchor


@dataclass
class GridLocation:
    """Location that can be literal or derived from context.

    Attributes:
        row: Literal row coordinate
        col: Literal column coordinate
        row_from_object_idx: Get row from this object's center
        col_from_object_idx: Get col from this object's center
    """
    row: Optional[int] = None
    col: Optional[int] = None
    row_from_object_idx: Optional[int] = None
    col_from_object_idx: Optional[int] = None

    def is_literal(self) -> bool:
        """True if this is a literal (row, col) location."""
        return self.row is not None and self.col is not None

    def resolve(
        self,
        objects: List['Object'],
        grid_height: int,
        grid_width: int
    ) -> Tuple[int, int]:
        """Resolve to actual (row, col) coordinates."""
        row = self.row
        col = self.col

        if self.row_from_object_idx is not None:
            ref_obj = objects[self.row_from_object_idx]
            row = int(ref_obj.center[0])

        if self.col_from_object_idx is not None:
            ref_obj = objects[self.col_from_object_idx]
            col = int(ref_obj.center[1])

        return (row if row is not None else 0,
                col if col is not None else 0)


@dataclass
class PositionSpec:
    """Specification for determining object position.

    Attributes:
        position_type: How to compute the position

        # For GRID_ANCHOR:
        object_anchor: Which anchor on the generated object
        grid_anchor: Which anchor point on the grid (TL, BC, etc.)
        offset: (row, col) offset from grid anchor

        # For OBJECT_RELATIVE:
        reference_object_idx: Which object to position relative to
        source_anchor: Which anchor on this object
        target_anchor: Which anchor on reference object
        anchor_offset: Offset between anchors

        # For AT_INTERSECTION:
        row_from_object_idx: Get row from this object's center
        col_from_object_idx: Get col from this object's center

        # For ABSOLUTE:
        absolute_location: Fixed (row, col) location

        # For PARENT_RELATIVE:
        parent_region_idx: Which parent region (from divider segmentation)
        parent_anchor: Which anchor point within the parent region
    """
    position_type: PositionType

    # GRID_ANCHOR fields
    object_anchor: Optional[AnchorPoint] = None
    grid_anchor: Optional[AnchorPoint] = None
    offset: Tuple[int, int] = (0, 0)

    # OBJECT_RELATIVE fields
    reference_object_idx: Optional[int] = None
    source_anchor: Optional[AnchorPoint] = None
    target_anchor: Optional[AnchorPoint] = None
    anchor_offset: Tuple[int, int] = (0, 0)

    # AT_INTERSECTION fields
    row_from_object_idx: Optional[int] = None
    col_from_object_idx: Optional[int] = None

    # ABSOLUTE fields
    absolute_location: Optional[GridLocation] = None

    # PARENT_RELATIVE fields
    parent_region_idx: Optional[int] = None
    parent_anchor: Optional[AnchorPoint] = None

    @classmethod
    def grid_relative(
        cls,
        grid_anchor: AnchorPoint,
        object_anchor: AnchorPoint = AnchorPoint.TOP_LEFT,
        offset: Tuple[int, int] = (0, 0)
    ) -> 'PositionSpec':
        """Position relative to a grid anchor point."""
        return cls(
            position_type=PositionType.GRID_ANCHOR,
            grid_anchor=grid_anchor,
            object_anchor=object_anchor,
            offset=offset
        )

    @classmethod
    def object_relative(
        cls,
        ref_idx: int,
        source_anchor: AnchorPoint,
        target_anchor: AnchorPoint,
        offset: Tuple[int, int] = (0, 0)
    ) -> 'PositionSpec':
        """Position relative to another object."""
        return cls(
            position_type=PositionType.OBJECT_RELATIVE,
            reference_object_idx=ref_idx,
            source_anchor=source_anchor,
            target_anchor=target_anchor,
            anchor_offset=offset
        )

    @classmethod
    def at_intersection(
        cls,
        row_from_idx: int,
        col_from_idx: int
    ) -> 'PositionSpec':
        """Position at intersection of row/col from two objects."""
        return cls(
            position_type=PositionType.AT_INTERSECTION,
            row_from_object_idx=row_from_idx,
            col_from_object_idx=col_from_idx
        )

    @classmethod
    def absolute(cls, row: int, col: int) -> 'PositionSpec':
        """Fixed absolute position."""
        return cls(
            position_type=PositionType.ABSOLUTE,
            absolute_location=GridLocation(row=row, col=col)
        )

    @classmethod
    def parent_relative(
        cls,
        parent_idx: int,
        parent_anchor: AnchorPoint,
        object_anchor: AnchorPoint = AnchorPoint.CENTER,
        offset: Tuple[int, int] = (0, 0)
    ) -> 'PositionSpec':
        """Position relative to a parent region's anchor point.

        Used for divider-segmented grids where the novel object should be
        placed at a specific location within a parent region.

        Args:
            parent_idx: Index of the parent region (from divider segmentation)
            parent_anchor: Which anchor point within the parent (e.g., CENTER)
            object_anchor: Which anchor on the generated object
            offset: Additional offset from the anchor point
        """
        return cls(
            position_type=PositionType.PARENT_RELATIVE,
            parent_region_idx=parent_idx,
            parent_anchor=parent_anchor,
            object_anchor=object_anchor,
            offset=offset
        )

    def describe(self) -> str:
        """Return human-readable description of the position spec."""
        if self.position_type == PositionType.GRID_ANCHOR:
            return f"grid_relative({self.grid_anchor.name}→{self.object_anchor.name}, offset={self.offset})"
        elif self.position_type == PositionType.OBJECT_RELATIVE:
            return f"object_relative(ref={self.reference_object_idx}, {self.target_anchor.name}→{self.source_anchor.name}, offset={self.anchor_offset})"
        elif self.position_type == PositionType.AT_INTERSECTION:
            return f"at_intersection(row_from={self.row_from_object_idx}, col_from={self.col_from_object_idx})"
        elif self.position_type == PositionType.ABSOLUTE:
            return f"absolute({self.absolute_location})"
        elif self.position_type == PositionType.PARENT_RELATIVE:
            return f"parent_relative(parent={self.parent_region_idx}, {self.parent_anchor.name})"
        return f"unknown({self.position_type})"


# =============================================================================
# Complete Object Specification
# =============================================================================

@dataclass
class ObjectSpec:
    """Complete specification for generating a novel object.

    Attributes:
        color: How to determine the color
        shape: What shape to create
        position: Where to place it
        anchor: Optional anchor point for positioning
        discovered_from_examples: How many examples this was discovered from
        confidence: Confidence score (0-1)
        variance: Variance across examples (lower = better)
    """
    color: ColorSpec
    shape: ShapeSpec
    position: PositionSpec
    anchor: Optional[AnchorPoint] = None
    discovered_from_examples: int = 0
    confidence: float = 1.0
    variance: float = 0.0

    def describe(self) -> str:
        """Human-readable description of this spec."""
        parts = []

        # Color
        if self.color.source_type == ColorSourceType.LITERAL:
            parts.append(f"color={self.color.literal_value}")
        elif self.color.source_type == ColorSourceType.MODE_OF_REGION:
            if self.color.region_idx == -1:
                parts.append("color=mode_of_grid")
            elif self.color.region_name:
                parts.append(f"color=mode_of_{self.color.region_name}")
            else:
                parts.append(f"color=mode_of_region[{self.color.region_idx}]")
        elif self.color.source_type == ColorSourceType.FROM_OBJECT:
            parts.append(f"color=from_object[{self.color.object_idx}]")
        elif self.color.source_type == ColorSourceType.NOT_IN_INPUT:
            parts.append(f"color=not_in_input[{self.color.not_in_input_rank}]")

        # Shape
        if self.shape.shape_type == ShapeType.PIXEL:
            parts.append("shape=pixel")
        elif self.shape.shape_type == ShapeType.HLINE:
            parts.append(f"shape=hline({self.shape.dimensions[1]})")
        elif self.shape.shape_type == ShapeType.VLINE:
            parts.append(f"shape=vline({self.shape.dimensions[0]})")
        elif self.shape.shape_type == ShapeType.RECT:
            parts.append(f"shape=rect{self.shape.dimensions}")
        elif self.shape.shape_type == ShapeType.TEMPLATE:
            parts.append(f"shape=template[{self.shape.template_source_idx}]")

        # Position
        if self.position.position_type == PositionType.GRID_ANCHOR:
            parts.append(f"pos=grid.{self.position.grid_anchor.value}")
        elif self.position.position_type == PositionType.OBJECT_RELATIVE:
            parts.append(f"pos=relative_to[{self.position.reference_object_idx}]")
        elif self.position.position_type == PositionType.AT_INTERSECTION:
            parts.append(f"pos=intersection(row[{self.position.row_from_object_idx}], col[{self.position.col_from_object_idx}])")
        elif self.position.position_type == PositionType.PARENT_RELATIVE:
            anchor = self.position.parent_anchor.value if self.position.parent_anchor else 'c'
            parts.append(f"pos=parent[{self.position.parent_region_idx}].{anchor}")

        return ", ".join(parts)


# =============================================================================
# Novel Object Detection
# =============================================================================

@dataclass
class NovelObject:
    """An output object with no input correspondence.

    Attributes:
        object: The output object
        output_idx: Index in output object list
        parent_idx: Index of parent region (for child-level novel objects)
        candidate_specs: Potential ObjectSpecs that could generate this object
    """
    object: 'Object'
    output_idx: int
    parent_idx: Optional[int] = None
    candidate_specs: List[ObjectSpec] = field(default_factory=list)


def find_novel_objects(
    input_objects: List['Object'],
    output_objects: List['Object'],
    input_grid: np.ndarray,
    output_grid: np.ndarray,
    threshold: float = 0.3
) -> List[NovelObject]:
    """
    Find output objects that have no corresponding input object.

    Uses the canonical shape-based matching from correspondence_module.
    Handles flat object lists and hierarchical objects uniformly.

    Args:
        input_objects: Objects extracted from input grid
        output_objects: Objects extracted from output grid
        input_grid: The input grid
        output_grid: The output grid
        threshold: Minimum similarity for correspondence (default 0.3)

    Returns:
        List of NovelObject for each output object without correspondence
    """
    return find_novel_children(
        input_objects, output_objects,
        input_grid, output_grid,
        threshold=threshold
    )


def find_novel_children(
    input_objects: List['Object'],
    output_objects: List['Object'],
    input_grid: np.ndarray,
    output_grid: np.ndarray,
    threshold: float = 0.3,
) -> List[NovelObject]:
    """
    Find novel children - output objects that don't correspond to any input object.

    Uses a TWO-LEVEL matching strategy:
    1. Match top-level regions/parents between input and output
    2. For each matched region pair, detect novel children:
       - If input region is atomic (no children) but output region has children
         → those children are NOVEL
       - If both have children → match children, report unmatched output children

    This correctly handles the asymmetric case where:
    - Input has an empty region (atomic, no children)
    - Output has content in that region (children)
    - The content is NOVEL because the input region was empty

    Args:
        input_objects: Objects extracted from input grid (may have hierarchy)
        output_objects: Objects extracted from output grid (may have hierarchy)
        input_grid: The input grid array
        output_grid: The output grid array
        threshold: Minimum similarity for a valid correspondence (default 0.3)

    Returns:
        List of NovelObject for each output object without correspondence.
        Each NovelObject includes:
        - object: The novel Object instance
        - output_idx: Object ID
        - parent_idx: ID of parent object (if object is a child), else None
    """
    if not output_objects:
        return []

    if not input_objects:
        # All output objects are novel if no input objects
        novel = []
        for obj in output_objects:
            if hasattr(obj, 'is_background') and obj.is_background:
                continue
            parent = getattr(obj, 'parent', None)
            parent_idx = parent.id if parent is not None else None
            novel.append(NovelObject(
                object=obj,
                output_idx=obj.id if hasattr(obj, 'id') else 0,
                parent_idx=parent_idx
            ))
        return novel

    # Check if we have hierarchical objects (parents with children)
    has_hierarchy = any(
        hasattr(obj, 'children') and obj.children
        for obj in input_objects + output_objects
    )

    if has_hierarchy:
        # TWO-LEVEL MATCHING for hierarchical objects
        return _find_novel_children_hierarchical(
            input_objects, output_objects,
            input_grid, output_grid,
            threshold
        )
    else:
        # FLAT MATCHING for non-hierarchical objects
        return _find_novel_children_flat(
            input_objects, output_objects,
            input_grid, output_grid,
            threshold
        )


def _find_novel_children_flat(
    input_objects: List['Object'],
    output_objects: List['Object'],
    input_grid: np.ndarray,
    output_grid: np.ndarray,
    threshold: float = 0.3,
) -> List[NovelObject]:
    """Find novel objects in flat (non-hierarchical) object lists."""
    _, unmatched_output, _ = find_unmatched_objects(
        input_grid, output_grid,
        input_objects, output_objects,
        threshold=threshold,
        use_matchable=False  # No hierarchy to handle
    )

    novel = []
    for obj in unmatched_output:
        if hasattr(obj, 'is_background') and obj.is_background:
            continue

        novel_obj = NovelObject(
            object=obj,
            output_idx=obj.id if hasattr(obj, 'id') else 0,
            parent_idx=None
        )
        novel.append(novel_obj)

    return novel


def _find_novel_children_hierarchical(
    input_objects: List['Object'],
    output_objects: List['Object'],
    input_grid: np.ndarray,
    output_grid: np.ndarray,
    threshold: float = 0.3,
) -> List[NovelObject]:
    """
    Find novel children using two-level hierarchical matching.

    Level 1: Match top-level regions (parents/atomic objects)
    Level 2: For each matched region pair, detect novel children
    """
    from correspondence_module import find_correspondences

    novel = []

    # Filter out background objects for top-level matching
    input_toplevel = [o for o in input_objects if not getattr(o, 'is_background', False)]
    output_toplevel = [o for o in output_objects if not getattr(o, 'is_background', False)]

    if not output_toplevel:
        return []

    # LEVEL 1: Match top-level regions
    # Use shape matching at the region level (NOT using get_matchable_objects)
    correspondences, _, _ = find_correspondences(
        input_grid, output_grid,
        input_toplevel, output_toplevel,
        threshold=threshold,
        use_matchable=False  # Match regions directly, not their children
    )

    # Build mapping: output region index -> matched input region index
    output_to_input = {}
    matched_input_idxs = set()
    for in_idx, out_idx, _score in correspondences:
        output_to_input[out_idx] = in_idx
        matched_input_idxs.add(in_idx)

    # LEVEL 2: For each output region, detect novel children
    for out_idx, out_region in enumerate(output_toplevel):
        out_children = getattr(out_region, 'children', None) or []

        if out_idx in output_to_input:
            # This output region matches an input region
            in_idx = output_to_input[out_idx]
            in_region = input_toplevel[in_idx]
            in_children = getattr(in_region, 'children', None) or []

            if not in_children and out_children:
                # Input region was atomic (empty), output region has children
                # ALL output children are novel
                for child in out_children:
                    novel.append(NovelObject(
                        object=child,
                        output_idx=child.id if hasattr(child, 'id') else 0,
                        parent_idx=out_region.id if hasattr(out_region, 'id') else out_idx
                    ))
            elif in_children and out_children:
                # Both have children - match children and find unmatched
                child_correspondences, _, _ = find_correspondences(
                    input_grid, output_grid,
                    in_children, out_children,
                    threshold=threshold,
                    use_matchable=False
                )
                matched_out_child_idxs = {out_c_idx for _, out_c_idx, _ in child_correspondences}

                for c_idx, child in enumerate(out_children):
                    if c_idx not in matched_out_child_idxs:
                        novel.append(NovelObject(
                            object=child,
                            output_idx=child.id if hasattr(child, 'id') else 0,
                            parent_idx=out_region.id if hasattr(out_region, 'id') else out_idx
                        ))
        else:
            # Output region has no matching input region - region itself is novel
            # (or we treat all its children as novel if it has any)
            if out_children:
                for child in out_children:
                    novel.append(NovelObject(
                        object=child,
                        output_idx=child.id if hasattr(child, 'id') else 0,
                        parent_idx=out_region.id if hasattr(out_region, 'id') else out_idx
                    ))
            else:
                # Atomic output region with no input match
                novel.append(NovelObject(
                    object=out_region,
                    output_idx=out_region.id if hasattr(out_region, 'id') else out_idx,
                    parent_idx=None
                ))

    return novel


def find_novel_pixels(
    input_grid: np.ndarray,
    output_grid: np.ndarray,
    input_objects: List['Object'],
    output_objects: List['Object'],
    correspondence_threshold: float = 0.3
) -> np.ndarray:
    """
    Find pixels in output that are not covered by matched objects.

    Returns:
        Boolean mask where True = novel pixel
    """
    from correspondence_module import find_correspondences

    # Start with all output pixels as potentially novel
    novel_mask = np.ones(output_grid.shape, dtype=bool)

    if not input_objects or not output_objects:
        return novel_mask

    # Find correspondences using canonical shape-based matching
    correspondences, _, _ = find_correspondences(
        input_grid, output_grid,
        input_objects, output_objects,
        threshold=correspondence_threshold,
        use_matchable=False
    )

    # Mark matched output object pixels as not novel
    matched_output_idxs = {out_idx for _, out_idx, _ in correspondences}
    for out_idx in matched_output_idxs:
        obj = output_objects[out_idx]
        for r, c in obj.pixels:
            if 0 <= r < output_grid.shape[0] and 0 <= c < output_grid.shape[1]:
                novel_mask[r, c] = False

    return novel_mask


# =============================================================================
# Rendering Functions
# =============================================================================

def resolve_color(
    color_spec: ColorSpec,
    input_grid: np.ndarray,
    input_objects: List['Object'],
    regions: Optional[List['RegionAggregation']] = None
) -> int:
    """
    Resolve a ColorSpec to an actual color value (0-9).

    Args:
        color_spec: The color specification
        input_grid: The input grid
        input_objects: List of input objects
        regions: Optional list of region aggregations

    Returns:
        The resolved color (0-9)
    """
    from aggregation_module import mode_color, mode_color_of_region_by_name, colors_not_in_grid

    if color_spec.source_type == ColorSourceType.LITERAL:
        return color_spec.literal_value if color_spec.literal_value is not None else 0

    elif color_spec.source_type == ColorSourceType.MODE_OF_REGION:
        if color_spec.region_name is not None:
            return mode_color_of_region_by_name(input_grid, color_spec.region_name)
        elif color_spec.region_idx == -1 or color_spec.region_idx is None:
            # Mode of whole grid
            return mode_color(input_grid)
        elif regions is not None and color_spec.region_idx < len(regions):
            return regions[color_spec.region_idx].pixel_stats.mode_color
        else:
            return mode_color(input_grid)

    elif color_spec.source_type == ColorSourceType.FROM_OBJECT:
        if color_spec.object_idx is not None and color_spec.object_idx < len(input_objects):
            return input_objects[color_spec.object_idx].color
        return 0

    elif color_spec.source_type == ColorSourceType.NOT_IN_INPUT:
        absent = colors_not_in_grid(input_grid)
        rank = color_spec.not_in_input_rank
        if rank < len(absent):
            return absent[rank]
        return 0  # Fallback

    elif color_spec.source_type == ColorSourceType.COMPLEMENT:
        # Return a color different from reference
        ref = color_spec.reference_color if color_spec.reference_color is not None else 0
        for c in range(10):
            if c != ref:
                return c
        return 0

    return 0


def render_shape(
    shape_spec: ShapeSpec,
    input_objects: Optional[List['Object']] = None,
    input_grid: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Render a shape to a boolean mask.

    Args:
        shape_spec: The shape specification
        input_objects: Input objects (for TEMPLATE shapes)
        input_grid: Input grid (for context)

    Returns:
        2D boolean array representing the shape
    """
    if shape_spec.shape_type == ShapeType.PIXEL:
        return np.ones((1, 1), dtype=bool)

    elif shape_spec.shape_type == ShapeType.HLINE:
        h, w = shape_spec.dimensions or (1, 1)
        return np.ones((h, w), dtype=bool)

    elif shape_spec.shape_type == ShapeType.VLINE:
        h, w = shape_spec.dimensions or (1, 1)
        return np.ones((h, w), dtype=bool)

    elif shape_spec.shape_type == ShapeType.RECT:
        h, w = shape_spec.dimensions or (1, 1)
        return np.ones((h, w), dtype=bool)

    elif shape_spec.shape_type == ShapeType.RECT_OUTLINE:
        h, w = shape_spec.dimensions or (1, 1)
        mask = np.zeros((h, w), dtype=bool)
        if h > 0 and w > 0:
            mask[0, :] = True      # Top
            mask[-1, :] = True     # Bottom
            mask[:, 0] = True      # Left
            mask[:, -1] = True     # Right
        return mask

    elif shape_spec.shape_type == ShapeType.TEMPLATE:
        # Use provided mask or extract from source object
        if shape_spec.template_mask is not None:
            return shape_spec.template_mask.copy()

        if input_objects is not None and shape_spec.template_source_idx is not None:
            idx = shape_spec.template_source_idx
            if idx < len(input_objects):
                src_obj = input_objects[idx]
                mask = np.zeros((src_obj.height, src_obj.width), dtype=bool)
                for r, c in src_obj.pixels:
                    local_r = r - src_obj.row
                    local_c = c - src_obj.col
                    if 0 <= local_r < src_obj.height and 0 <= local_c < src_obj.width:
                        mask[local_r, local_c] = True
                return mask

        return np.ones((1, 1), dtype=bool)

    return np.ones((1, 1), dtype=bool)


def resolve_position(
    position_spec: PositionSpec,
    shape_size: Tuple[int, int],
    grid_height: int,
    grid_width: int,
    input_objects: List['Object'],
    output_objects_so_far: Optional[List['Object']] = None,
    regions: Optional[List['RegionAggregation']] = None
) -> Tuple[int, int]:
    """
    Resolve a PositionSpec to (row, col) top-left coordinates.

    Args:
        position_spec: The position specification
        shape_size: (height, width) of the shape being placed
        grid_height: Height of the output grid
        grid_width: Width of the output grid
        input_objects: Input objects for reference
        output_objects_so_far: Output objects already generated (for chaining)
        regions: Optional parent regions from divider segmentation

    Returns:
        (row, col) top-left position for the new object
    """
    obj_h, obj_w = shape_size

    if position_spec.position_type == PositionType.GRID_ANCHOR:
        # Get position of grid anchor point
        grid_anchor = position_spec.grid_anchor or AnchorPoint.TOP_LEFT
        grid_anchor_pos = get_anchor_offset(grid_anchor, grid_height, grid_width)

        # Apply offset
        target_pos = (
            grid_anchor_pos[0] + position_spec.offset[0],
            grid_anchor_pos[1] + position_spec.offset[1]
        )

        # Back-calculate top-left from object anchor
        obj_anchor = position_spec.object_anchor or AnchorPoint.TOP_LEFT
        anchor_dr, anchor_dc = get_anchor_offset(obj_anchor, obj_h, obj_w)

        return (target_pos[0] - anchor_dr, target_pos[1] - anchor_dc)

    elif position_spec.position_type == PositionType.OBJECT_RELATIVE:
        # Get reference object
        ref_idx = position_spec.reference_object_idx or 0

        # Check output objects first, then input
        if output_objects_so_far and ref_idx < len(output_objects_so_far):
            ref_obj = output_objects_so_far[ref_idx]
        elif ref_idx < len(input_objects):
            ref_obj = input_objects[ref_idx]
        else:
            return (0, 0)

        # Get reference object's anchor position
        target_anchor = position_spec.target_anchor or AnchorPoint.CENTER
        ref_anchor_pos = get_anchor_position(
            (ref_obj.row, ref_obj.col),
            (ref_obj.height, ref_obj.width),
            target_anchor
        )

        # Apply offset
        source_anchor_pos = (
            ref_anchor_pos[0] + position_spec.anchor_offset[0],
            ref_anchor_pos[1] + position_spec.anchor_offset[1]
        )

        # Back-calculate top-left from source anchor
        source_anchor = position_spec.source_anchor or AnchorPoint.TOP_LEFT
        anchor_dr, anchor_dc = get_anchor_offset(source_anchor, obj_h, obj_w)

        return (source_anchor_pos[0] - anchor_dr, source_anchor_pos[1] - anchor_dc)

    elif position_spec.position_type == PositionType.AT_INTERSECTION:
        row_idx = position_spec.row_from_object_idx
        col_idx = position_spec.col_from_object_idx

        row = 0
        col = 0

        if row_idx is not None and row_idx < len(input_objects):
            row = int(input_objects[row_idx].center[0])

        if col_idx is not None and col_idx < len(input_objects):
            col = int(input_objects[col_idx].center[1])

        # Center the shape at this point
        return (row - obj_h // 2, col - obj_w // 2)

    elif position_spec.position_type == PositionType.ABSOLUTE:
        if position_spec.absolute_location is not None:
            return position_spec.absolute_location.resolve(
                input_objects, grid_height, grid_width
            )
        return (0, 0)

    elif position_spec.position_type == PositionType.PARENT_RELATIVE:
        # Position relative to a parent region from divider segmentation
        parent_idx = position_spec.parent_region_idx or 0

        if regions is None or parent_idx >= len(regions):
            # Fall back to grid center if regions not available
            return (grid_height // 2 - obj_h // 2, grid_width // 2 - obj_w // 2)

        parent_region = regions[parent_idx].region
        parent_anchor = position_spec.parent_anchor or AnchorPoint.CENTER

        # Get parent region's anchor position
        parent_anchor_pos = get_anchor_position(
            (parent_region.row, parent_region.col),
            (parent_region.height, parent_region.width),
            parent_anchor
        )

        # Apply offset
        target_pos = (
            parent_anchor_pos[0] + position_spec.offset[0],
            parent_anchor_pos[1] + position_spec.offset[1]
        )

        # Back-calculate top-left from object anchor
        obj_anchor = position_spec.object_anchor or AnchorPoint.CENTER
        anchor_dr, anchor_dc = get_anchor_offset(obj_anchor, obj_h, obj_w)

        return (target_pos[0] - anchor_dr, target_pos[1] - anchor_dc)

    return (0, 0)


def render_object(
    spec: ObjectSpec,
    grid: np.ndarray,
    input_grid: np.ndarray,
    input_objects: List['Object'],
    output_objects_so_far: Optional[List['Object']] = None,
    regions: Optional[List['RegionAggregation']] = None
) -> np.ndarray:
    """
    Render an object onto a grid according to its specification.

    Args:
        spec: The object specification
        grid: The grid to render onto (modified in place)
        input_grid: The input grid (for color derivation)
        input_objects: Input objects
        output_objects_so_far: Already-rendered output objects
        regions: Region aggregations for color derivation

    Returns:
        The modified grid
    """
    # Resolve color
    color = resolve_color(spec.color, input_grid, input_objects, regions)

    # Render shape
    shape_mask = render_shape(spec.shape, input_objects, input_grid)
    h, w = shape_mask.shape

    # Resolve position
    row, col = resolve_position(
        spec.position,
        (h, w),
        grid.shape[0], grid.shape[1],
        input_objects,
        output_objects_so_far,
        regions
    )

    # Paint onto grid
    for dr in range(h):
        for dc in range(w):
            if shape_mask[dr, dc]:
                r, c = row + dr, col + dc
                if 0 <= r < grid.shape[0] and 0 <= c < grid.shape[1]:
                    grid[r, c] = color

    return grid


# =============================================================================
# Hypothesis Screening
# =============================================================================

def screen_color_hypotheses(
    novel_objects: List[List[NovelObject]],
    input_objects_per_example: List[List['Object']],
    input_grids: List[np.ndarray],
    regions_per_example: Optional[List[List['RegionAggregation']]] = None
) -> List[Tuple[ColorSpec, float]]:
    """
    Screen color hypotheses for novel objects across examples.

    Args:
        novel_objects: Novel objects per example [example_idx][novel_idx]
        input_objects_per_example: Input objects per example
        input_grids: Input grids per example
        regions_per_example: Optional regions per example

    Returns:
        List of (ColorSpec, accuracy) sorted by accuracy (higher = better)
    """
    from aggregation_module import mode_color, mode_color_of_region_by_name, colors_not_in_grid

    if not novel_objects or not novel_objects[0]:
        return []

    num_examples = len(novel_objects)
    hypotheses = []

    # Get actual colors from novel objects
    actual_colors = [novel_objects[ex][0].object.color for ex in range(num_examples)]

    # Hypothesis 1: LITERAL - is color consistent?
    if len(set(actual_colors)) == 1:
        hypotheses.append((ColorSpec.literal(actual_colors[0]), 1.0))

    # Hypothesis 2: MODE_OF_GRID
    predicted = [mode_color(input_grids[ex]) for ex in range(num_examples)]
    accuracy = sum(p == a for p, a in zip(predicted, actual_colors)) / num_examples
    if accuracy > 0:
        hypotheses.append((ColorSpec.mode_of_grid(), accuracy))

    # Hypothesis 3: MODE_OF_REGION (named regions)
    for region_name in ['top_half', 'bottom_half', 'left_half', 'right_half']:
        predicted = [mode_color_of_region_by_name(input_grids[ex], region_name)
                     for ex in range(num_examples)]
        accuracy = sum(p == a for p, a in zip(predicted, actual_colors)) / num_examples
        if accuracy > 0:
            hypotheses.append((ColorSpec.mode_of_region_by_name(region_name), accuracy))

    # Hypothesis 3b: MODE_OF_PARENT_REGION (from divider segmentation)
    # This tests each parent region created by divider detection
    if regions_per_example and regions_per_example[0]:
        max_regions = min(len(regs) for regs in regions_per_example)
        for region_idx in range(max_regions):
            predicted = []
            for ex in range(num_examples):
                if region_idx < len(regions_per_example[ex]):
                    predicted.append(regions_per_example[ex][region_idx].pixel_stats.mode_color)
                else:
                    predicted.append(-1)  # Invalid

            accuracy = sum(p == a for p, a in zip(predicted, actual_colors)) / num_examples
            if accuracy > 0:
                hypotheses.append((ColorSpec.mode_of_region(region_idx), accuracy))

    # Hypothesis 4: FROM_OBJECT
    if input_objects_per_example and input_objects_per_example[0]:
        max_obj_idx = min(len(objs) for objs in input_objects_per_example)
        for obj_idx in range(max_obj_idx):
            predicted = [input_objects_per_example[ex][obj_idx].color
                         for ex in range(num_examples)]
            accuracy = sum(p == a for p, a in zip(predicted, actual_colors)) / num_examples
            if accuracy > 0:
                hypotheses.append((ColorSpec.from_object(obj_idx), accuracy))

    # Hypothesis 5: NOT_IN_INPUT
    not_in_input_match = True
    for ex in range(num_examples):
        absent = colors_not_in_grid(input_grids[ex])
        if actual_colors[ex] not in absent:
            not_in_input_match = False
            break
    if not_in_input_match:
        hypotheses.append((ColorSpec.not_in_input(0), 1.0))

    # Sort by accuracy
    hypotheses.sort(key=lambda x: -x[1])
    return hypotheses


def screen_position_hypotheses(
    novel_objects: List[List[NovelObject]],
    input_objects_per_example: List[List['Object']],
    output_objects_per_example: List[List['Object']],
    grid_sizes: List[Tuple[int, int]],
    regions_per_example: Optional[List[List['RegionAggregation']]] = None
) -> List[Tuple[PositionSpec, float]]:
    """
    Screen position hypotheses for novel objects across examples.

    Uses anchor-based discovery similar to anchoring_module.

    Args:
        novel_objects: Novel objects per example
        input_objects_per_example: Input objects per example
        output_objects_per_example: Output objects per example
        grid_sizes: (height, width) per example
        regions_per_example: Optional parent regions from divider segmentation

    Returns:
        List of (PositionSpec, variance) sorted by variance (lower = better)
    """
    if not novel_objects or not novel_objects[0]:
        return []

    num_examples = len(novel_objects)
    hypotheses = []

    # Get novel object positions
    novel_positions = []
    novel_sizes = []
    for ex in range(num_examples):
        obj = novel_objects[ex][0].object
        novel_positions.append((obj.row, obj.col))
        novel_sizes.append((obj.height, obj.width))

    # Hypothesis 1: GRID_ANCHOR - try all 81 combinations
    for grid_anchor in ALL_ANCHORS:
        for obj_anchor in ALL_ANCHORS:
            offsets = []
            for ex in range(num_examples):
                H, W = grid_sizes[ex]
                obj_h, obj_w = novel_sizes[ex]
                obj_row, obj_col = novel_positions[ex]

                # Get where grid anchor is
                grid_anchor_pos = get_anchor_offset(grid_anchor, H, W)

                # Get where object's anchor is
                obj_anchor_offset = get_anchor_offset(obj_anchor, obj_h, obj_w)
                obj_anchor_pos = (obj_row + obj_anchor_offset[0],
                                  obj_col + obj_anchor_offset[1])

                # Offset = object_anchor_pos - grid_anchor_pos
                offset = (obj_anchor_pos[0] - grid_anchor_pos[0],
                          obj_anchor_pos[1] - grid_anchor_pos[1])
                offsets.append(offset)

            # Check variance
            if offsets:
                row_offsets = [o[0] for o in offsets]
                col_offsets = [o[1] for o in offsets]
                variance = np.var(row_offsets) + np.var(col_offsets)

                if variance < 1e-6:  # Zero variance = consistent
                    mean_offset = (int(np.mean(row_offsets)),
                                   int(np.mean(col_offsets)))
                    spec = PositionSpec.grid_relative(
                        grid_anchor=grid_anchor,
                        object_anchor=obj_anchor,
                        offset=mean_offset
                    )
                    hypotheses.append((spec, variance))

    # Hypothesis 2: OBJECT_RELATIVE - try each input object
    if input_objects_per_example and input_objects_per_example[0]:
        max_obj_idx = min(len(objs) for objs in input_objects_per_example)

        for ref_idx in range(max_obj_idx):
            for src_anchor in ALL_ANCHORS:
                for tgt_anchor in ALL_ANCHORS:
                    offsets = []

                    for ex in range(num_examples):
                        ref_obj = input_objects_per_example[ex][ref_idx]
                        novel_obj = novel_objects[ex][0].object

                        # Get reference anchor position
                        ref_anchor_pos = get_anchor_position(
                            (ref_obj.row, ref_obj.col),
                            (ref_obj.height, ref_obj.width),
                            tgt_anchor
                        )

                        # Get novel object's anchor position
                        novel_anchor_pos = get_anchor_position(
                            (novel_obj.row, novel_obj.col),
                            (novel_obj.height, novel_obj.width),
                            src_anchor
                        )

                        offset = (novel_anchor_pos[0] - ref_anchor_pos[0],
                                  novel_anchor_pos[1] - ref_anchor_pos[1])
                        offsets.append(offset)

                    # Check variance
                    if offsets:
                        row_offsets = [o[0] for o in offsets]
                        col_offsets = [o[1] for o in offsets]
                        variance = np.var(row_offsets) + np.var(col_offsets)

                        if variance < 1e-6:
                            mean_offset = (int(np.mean(row_offsets)),
                                           int(np.mean(col_offsets)))
                            spec = PositionSpec.object_relative(
                                ref_idx=ref_idx,
                                source_anchor=src_anchor,
                                target_anchor=tgt_anchor,
                                offset=mean_offset
                            )
                            hypotheses.append((spec, variance))

    # Hypothesis 3: PARENT_RELATIVE - position within parent regions
    # This tests placing the novel object at an anchor point within a parent region
    if regions_per_example and regions_per_example[0]:
        max_regions = min(len(regs) for regs in regions_per_example)

        for parent_idx in range(max_regions):
            for parent_anchor in ALL_ANCHORS:
                for obj_anchor in ALL_ANCHORS:
                    offsets = []

                    for ex in range(num_examples):
                        if parent_idx >= len(regions_per_example[ex]):
                            continue

                        parent_region = regions_per_example[ex][parent_idx].region
                        novel_obj = novel_objects[ex][0].object

                        # Get parent region's anchor position
                        parent_anchor_pos = get_anchor_position(
                            (parent_region.row, parent_region.col),
                            (parent_region.height, parent_region.width),
                            parent_anchor
                        )

                        # Get novel object's anchor position
                        novel_anchor_pos = get_anchor_position(
                            (novel_obj.row, novel_obj.col),
                            (novel_obj.height, novel_obj.width),
                            obj_anchor
                        )

                        offset = (novel_anchor_pos[0] - parent_anchor_pos[0],
                                  novel_anchor_pos[1] - parent_anchor_pos[1])
                        offsets.append(offset)

                    # Check variance
                    if len(offsets) == num_examples:
                        row_offsets = [o[0] for o in offsets]
                        col_offsets = [o[1] for o in offsets]
                        variance = np.var(row_offsets) + np.var(col_offsets)

                        if variance < 1e-6:
                            mean_offset = (int(np.mean(row_offsets)),
                                           int(np.mean(col_offsets)))
                            spec = PositionSpec.parent_relative(
                                parent_idx=parent_idx,
                                parent_anchor=parent_anchor,
                                object_anchor=obj_anchor,
                                offset=mean_offset
                            )
                            hypotheses.append((spec, variance))

    # Sort by variance (lower = better)
    hypotheses.sort(key=lambda x: x[1])
    return hypotheses


def screen_shape_hypotheses(
    novel_objects: List[List[NovelObject]],
    input_objects_per_example: List[List['Object']],
    input_grids: List[np.ndarray]
) -> List[Tuple[ShapeSpec, float]]:
    """
    Screen shape hypotheses for novel objects across examples.

    Args:
        novel_objects: Novel objects per example
        input_objects_per_example: Input objects per example
        input_grids: Input grids per example

    Returns:
        List of (ShapeSpec, score) sorted by score (higher = better)
    """
    if not novel_objects or not novel_objects[0]:
        return []

    num_examples = len(novel_objects)
    hypotheses = []

    # Get novel object shapes
    novel_shapes = []
    for ex in range(num_examples):
        obj = novel_objects[ex][0].object
        novel_shapes.append({
            'height': obj.height,
            'width': obj.width,
            'area': obj.area,
            'pixels': obj.pixels
        })

    # Hypothesis 1: PIXEL - all are single pixels?
    all_pixels = all(s['area'] == 1 for s in novel_shapes)
    if all_pixels:
        hypotheses.append((ShapeSpec.pixel(), 1.0))

    # Hypothesis 2: HLINE - all are horizontal lines (height=1)?
    all_hline = all(s['height'] == 1 for s in novel_shapes)
    if all_hline:
        widths = [s['width'] for s in novel_shapes]
        if len(set(widths)) == 1:
            hypotheses.append((ShapeSpec.hline(widths[0]), 1.0))
        else:
            # Variable width - still an hline pattern
            hypotheses.append((ShapeSpec.hline(int(np.mean(widths))), 0.8))

    # Hypothesis 3: VLINE - all are vertical lines (width=1)?
    all_vline = all(s['width'] == 1 for s in novel_shapes)
    if all_vline:
        heights = [s['height'] for s in novel_shapes]
        if len(set(heights)) == 1:
            hypotheses.append((ShapeSpec.vline(heights[0]), 1.0))
        else:
            hypotheses.append((ShapeSpec.vline(int(np.mean(heights))), 0.8))

    # Hypothesis 4: RECT - all are filled rectangles?
    all_rect = all(
        s['area'] == s['height'] * s['width']
        for s in novel_shapes
    )
    if all_rect:
        sizes = [(s['height'], s['width']) for s in novel_shapes]
        if len(set(sizes)) == 1:
            h, w = sizes[0]
            hypotheses.append((ShapeSpec.rect(h, w), 1.0))
        else:
            mean_h = int(np.mean([s[0] for s in sizes]))
            mean_w = int(np.mean([s[1] for s in sizes]))
            hypotheses.append((ShapeSpec.rect(mean_h, mean_w), 0.7))

    # Hypothesis 5: TEMPLATE - matches an input object's shape?
    if input_objects_per_example and input_objects_per_example[0]:
        max_obj_idx = min(len(objs) for objs in input_objects_per_example)

        for src_idx in range(max_obj_idx):
            matches = 0
            for ex in range(num_examples):
                src_obj = input_objects_per_example[ex][src_idx]
                novel_shape = novel_shapes[ex]

                # Check if dimensions match
                if (src_obj.height == novel_shape['height'] and
                    src_obj.width == novel_shape['width'] and
                    src_obj.area == novel_shape['area']):
                    matches += 1

            accuracy = matches / num_examples
            if accuracy > 0.5:
                hypotheses.append((ShapeSpec.template(src_idx), accuracy))

    # Sort by score
    hypotheses.sort(key=lambda x: -x[1])
    return hypotheses


# =============================================================================
# Genesis Discovery Pipeline
# =============================================================================

@dataclass
class GenesisDiscoveryResult:
    """Result of running genesis discovery on multiple examples.

    Attributes:
        novel_object_count: Number of novel objects detected per example
        consistent_specs: ObjectSpecs consistent across all examples
        best_spec: The single best specification found
        per_novel_object_specs: Best spec for each novel object position
    """
    novel_object_count: int
    consistent_specs: List[ObjectSpec]
    best_spec: Optional[ObjectSpec]
    per_novel_object_specs: Dict[int, ObjectSpec] = field(default_factory=dict)


def discover_genesis_rules(
    examples: List[Dict],
    correspondence_threshold: float = 0.3
) -> GenesisDiscoveryResult:
    """
    Main genesis discovery pipeline.

    For each training example:
    1. Extract input/output objects
    2. Find correspondences and identify novel objects
    3. Screen color, position, and shape hypotheses
    4. Combine best hypotheses into ObjectSpecs

    Args:
        examples: List of {'input': grid, 'output': grid} dicts
        correspondence_threshold: Similarity threshold for correspondence

    Returns:
        GenesisDiscoveryResult with discovered rules
    """
    from object_module import extract_objects_from_grid
    from aggregation_module import aggregate_regions

    if not examples:
        return GenesisDiscoveryResult(
            novel_object_count=0,
            consistent_specs=[],
            best_spec=None
        )

    # Process each example
    novel_objects_per_example = []
    input_objects_per_example = []
    output_objects_per_example = []
    input_grids = []
    grid_sizes = []
    regions_per_example = []

    for example in examples:
        input_grid = np.array(example['input'])
        output_grid = np.array(example['output'])

        input_objects = extract_objects_from_grid(input_grid)
        output_objects = extract_objects_from_grid(output_grid)

        # Filter out background objects
        input_objects = [o for o in input_objects if not o.is_background]
        output_objects = [o for o in output_objects if not o.is_background]

        novel = find_novel_objects(
            input_objects, output_objects,
            input_grid, output_grid,
            correspondence_threshold
        )

        novel_objects_per_example.append(novel)
        input_objects_per_example.append(input_objects)
        output_objects_per_example.append(output_objects)
        input_grids.append(input_grid)
        grid_sizes.append(output_grid.shape)

        # Compute regions
        regions = aggregate_regions(input_grid, input_objects)
        regions_per_example.append(regions)

    # Check consistency of novel object count
    novel_counts = [len(n) for n in novel_objects_per_example]
    if not all(c == novel_counts[0] for c in novel_counts):
        # Variable number of novel objects - more complex case
        # For now, just use minimum
        min_count = min(novel_counts)
        for i, novels in enumerate(novel_objects_per_example):
            novel_objects_per_example[i] = novels[:min_count]

    num_novel = novel_counts[0] if novel_counts else 0

    if num_novel == 0:
        return GenesisDiscoveryResult(
            novel_object_count=0,
            consistent_specs=[],
            best_spec=None
        )

    # Discover specs for each novel object position
    consistent_specs = []
    per_novel_specs = {}

    for novel_idx in range(num_novel):
        # Collect this novel object across all examples
        novel_at_idx = [[ex[novel_idx]] if novel_idx < len(ex) else []
                        for ex in novel_objects_per_example]

        # Skip if not present in all examples
        if any(len(n) == 0 for n in novel_at_idx):
            continue

        # Screen hypotheses
        color_candidates = screen_color_hypotheses(
            novel_at_idx,
            input_objects_per_example,
            input_grids,
            regions_per_example
        )

        position_candidates = screen_position_hypotheses(
            novel_at_idx,
            input_objects_per_example,
            output_objects_per_example,
            grid_sizes,
            regions_per_example
        )

        shape_candidates = screen_shape_hypotheses(
            novel_at_idx,
            input_objects_per_example,
            input_grids
        )

        # Combine best hypotheses
        best_color = color_candidates[0][0] if color_candidates else ColorSpec.literal(0)
        best_position = position_candidates[0][0] if position_candidates else PositionSpec.grid_relative(AnchorPoint.CENTER)
        best_shape = shape_candidates[0][0] if shape_candidates else ShapeSpec.pixel()

        # Compute overall confidence
        color_conf = color_candidates[0][1] if color_candidates else 0.0
        pos_variance = position_candidates[0][1] if position_candidates else float('inf')
        shape_conf = shape_candidates[0][1] if shape_candidates else 0.0

        confidence = (color_conf + shape_conf) / 2.0 if pos_variance < 1e-6 else 0.0

        spec = ObjectSpec(
            color=best_color,
            shape=best_shape,
            position=best_position,
            confidence=confidence,
            variance=pos_variance,
            discovered_from_examples=len(examples)
        )

        consistent_specs.append(spec)
        per_novel_specs[novel_idx] = spec

    return GenesisDiscoveryResult(
        novel_object_count=num_novel,
        consistent_specs=consistent_specs,
        best_spec=consistent_specs[0] if consistent_specs else None,
        per_novel_object_specs=per_novel_specs
    )


# =============================================================================
# Visualization and CLI
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
    ax.set_title(title)
    ax.axis('off')


def _highlight_novel_objects(ax, novel_objects: List[NovelObject], color='lime'):
    """Highlight novel objects with circles."""
    import matplotlib.patches as mpatches

    for novel in novel_objects:
        obj = novel.object
        # Draw a rectangle around the novel object
        rect = mpatches.Rectangle(
            (obj.col - 0.5, obj.row - 0.5),
            obj.width, obj.height,
            linewidth=3, edgecolor=color, facecolor='none',
            linestyle='--'
        )
        ax.add_patch(rect)

        # Add label
        ax.annotate(
            f'NOVEL\nc={obj.color}',
            (obj.col + obj.width / 2, obj.row + obj.height / 2),
            color='white', fontsize=8, fontweight='bold',
            ha='center', va='center',
            bbox=dict(boxstyle='round', facecolor=color, alpha=0.8)
        )


def trace_genesis_for_puzzle(puzzle_id: str, data_root: str = "kaggle/combined",
                             threshold: float = 0.3, verbose: bool = True,
                             strategy: Optional[SegmentationStrategy] = None,
                             correspondence_mode: CorrespondenceMode = "one_to_one",
                             correspondence_margin: float = DEFAULT_MARGIN):
    """
    Detailed tracing of genesis (novel object) detection for a puzzle.

    This function provides comprehensive output about:
    1. Object extraction from input/output grids
    2. Shape feature extraction and comparison
    3. Correspondence matching scores and decisions
    4. Why specific objects were marked as novel (unmatched)

    Args:
        puzzle_id: The ARC puzzle ID (e.g., "27a77e38")
        data_root: Path to puzzle data
        threshold: Similarity threshold for correspondence matching
        verbose: If True, print detailed shape features
        strategy: Optional segmentation strategy for input/output grids
        correspondence_mode: Matching mode ('one_to_one', 'many_to_one', 'one_to_many')
        correspondence_margin: Margin for non-one_to_one modes

    Returns:
        Dict with tracing results for programmatic analysis
    """
    from object_module import extract_objects_from_grid, get_matchable_objects
    from correspondence_module import (
        find_correspondences, find_unmatched_objects,
        ShapeFeatureExtractor, compute_shape_similarity
    )

    print(f"\n{'='*80}")
    print(f"GENESIS TRACE: Puzzle {puzzle_id}")
    print(f"{'='*80}")
    print(f"Correspondence threshold: {threshold}")
    if correspondence_mode != "one_to_one":
        print(f"Correspondence mode: {correspondence_mode} (margin={correspondence_margin})")

    puzzle = _load_puzzle(puzzle_id, data_root)
    print(f"Training examples: {len(puzzle['train'])}")

    trace_results = {
        'puzzle_id': puzzle_id,
        'threshold': threshold,
        'examples': []
    }

    extractor = ShapeFeatureExtractor(n_fourier_coefficients=32)

    for ex_idx, example in enumerate(puzzle['train']):
        if 'output' not in example:
            continue

        input_grid = np.array(example['input'], dtype=np.int64)
        output_grid = np.array(example['output'], dtype=np.int64)

        print(f"\n{'='*80}")
        print(f"EXAMPLE {ex_idx}")
        print(f"{'='*80}")
        print(f"Input grid:  {input_grid.shape[0]}x{input_grid.shape[1]}")
        print(f"Output grid: {output_grid.shape[0]}x{output_grid.shape[1]}")

        # =================================================================
        # STEP 1: Object Extraction
        # =================================================================
        print(f"\n{'-'*40}")
        print("STEP 1: Object Extraction")
        print(f"{'-'*40}")

        # Determine segmentation modes
        input_mode = strategy.input_mode if strategy else SegmentationMode.CONNECTIVITY
        output_mode = strategy.output_mode if strategy else SegmentationMode.CONNECTIVITY
        if strategy and (input_mode != SegmentationMode.CONNECTIVITY or output_mode != SegmentationMode.CONNECTIVITY):
            print(f"Input segmentation mode: {input_mode.value}")
            print(f"Output segmentation mode: {output_mode.value}")

        input_objects_raw = extract_objects_from_grid(input_grid, segmentation_mode=input_mode)
        output_objects_raw = extract_objects_from_grid(output_grid, segmentation_mode=output_mode)

        # Filter background
        input_objects = [o for o in input_objects_raw if not o.is_background]
        output_objects = [o for o in output_objects_raw if not o.is_background]

        print(f"\nInput objects (excluding background): {len(input_objects)}")
        for i, obj in enumerate(input_objects):
            children_info = ""
            if hasattr(obj, 'children') and obj.children:
                children_info = f", children={len(obj.children)}"
            print(f"  [{i}] id={obj.id}, color={obj.color}, pos=({obj.row},{obj.col}), "
                  f"size={obj.height}x{obj.width}, area={obj.area}{children_info}")

        print(f"\nOutput objects (excluding background): {len(output_objects)}")
        for i, obj in enumerate(output_objects):
            children_info = ""
            if hasattr(obj, 'children') and obj.children:
                children_info = f", children={len(obj.children)}"
            print(f"  [{i}] id={obj.id}, color={obj.color}, pos=({obj.row},{obj.col}), "
                  f"size={obj.height}x{obj.width}, area={obj.area}{children_info}")

        # =================================================================
        # STEP 2: Matchable Objects (Hierarchy Handling)
        # =================================================================
        print(f"\n{'-'*40}")
        print("STEP 2: Matchable Objects (Hierarchy Handling)")
        print(f"{'-'*40}")

        input_matchable = get_matchable_objects(input_objects, grid_shape=input_grid.shape)
        output_matchable = get_matchable_objects(output_objects, grid_shape=output_grid.shape)

        print(f"\nInput matchable objects: {len(input_matchable)}")
        if len(input_matchable) != len(input_objects):
            print("  (Different from raw count - hierarchy flattening occurred)")
        for i, obj in enumerate(input_matchable):
            parent_info = ""
            if hasattr(obj, 'parent') and obj.parent is not None:
                parent_info = f", parent_id={obj.parent.id}"
            print(f"  [{i}] id={obj.id}, color={obj.color}, pos=({obj.row},{obj.col}), "
                  f"size={obj.height}x{obj.width}, area={obj.area}{parent_info}")

        print(f"\nOutput matchable objects: {len(output_matchable)}")
        if len(output_matchable) != len(output_objects):
            print("  (Different from raw count - hierarchy flattening occurred)")
        for i, obj in enumerate(output_matchable):
            parent_info = ""
            if hasattr(obj, 'parent') and obj.parent is not None:
                parent_info = f", parent_id={obj.parent.id}"
            print(f"  [{i}] id={obj.id}, color={obj.color}, pos=({obj.row},{obj.col}), "
                  f"size={obj.height}x{obj.width}, area={obj.area}{parent_info}")

        # =================================================================
        # STEP 3: Shape Feature Extraction
        # =================================================================
        print(f"\n{'-'*40}")
        print("STEP 3: Shape Feature Extraction")
        print(f"{'-'*40}")

        # Create masks for feature extraction
        H_in, W_in = input_grid.shape
        H_out, W_out = output_grid.shape

        input_features = []
        print("\nInput object features:")
        for i, obj in enumerate(input_matchable):
            mask = np.zeros((H_in, W_in), dtype=bool)
            for r, c in obj.pixels:
                if 0 <= r < H_in and 0 <= c < W_in:
                    mask[r, c] = True
            features = extractor.extract(mask, input_grid)
            input_features.append(features)
            if verbose:
                print(f"  [{i}] area={features.area}, density={features.density:.3f}, "
                      f"aspect={features.aspect_ratio:.3f}, compact={features.compactness:.3f}, "
                      f"euler={features.euler_number}, colors={features.num_colors}, "
                      f"dominant={features.dominant_color}")

        output_features = []
        print("\nOutput object features:")
        for i, obj in enumerate(output_matchable):
            mask = np.zeros((H_out, W_out), dtype=bool)
            for r, c in obj.pixels:
                if 0 <= r < H_out and 0 <= c < W_out:
                    mask[r, c] = True
            features = extractor.extract(mask, output_grid)
            output_features.append(features)
            if verbose:
                print(f"  [{i}] area={features.area}, density={features.density:.3f}, "
                      f"aspect={features.aspect_ratio:.3f}, compact={features.compactness:.3f}, "
                      f"euler={features.euler_number}, colors={features.num_colors}, "
                      f"dominant={features.dominant_color}")

        # =================================================================
        # STEP 4: Similarity Matrix
        # =================================================================
        print(f"\n{'-'*40}")
        print("STEP 4: Similarity Matrix (Shape-based)")
        print(f"{'-'*40}")

        # Compute full similarity matrix
        n_in = len(input_matchable)
        n_out = len(output_matchable)

        similarity_matrix = np.zeros((n_in, n_out))
        component_matrices = {
            'structural': np.zeros((n_in, n_out)),
            'moments': np.zeros((n_in, n_out)),
            'color': np.zeros((n_in, n_out)),
            'fourier': np.zeros((n_in, n_out)),
            'location': np.zeros((n_in, n_out)),
        }

        shape_only_weights = {
            'structural': 1.0,
            'moments': 1.0,
            'color': 1.0,
            'fourier': 1.0,
            'location': 0.0,  # Shape matching ignores location
        }

        for i in range(n_in):
            for j in range(n_out):
                # Overall similarity (shape-only for matching)
                similarity_matrix[i, j] = compute_shape_similarity(
                    input_features[i], output_features[j], shape_only_weights
                )

                # Component-wise similarities for debugging
                for comp in ['structural', 'moments', 'color', 'fourier', 'location']:
                    comp_weights = {k: 1.0 if k == comp else 0.0 for k in shape_only_weights}
                    if comp_weights[comp] > 0:
                        component_matrices[comp][i, j] = compute_shape_similarity(
                            input_features[i], output_features[j], comp_weights
                        )

        print("\nSimilarity matrix (rows=input, cols=output):")
        print("        ", end="")
        for j in range(n_out):
            print(f"Out[{j}]  ", end="")
        print()

        for i in range(n_in):
            print(f"In[{i}]   ", end="")
            for j in range(n_out):
                score = similarity_matrix[i, j]
                marker = "*" if score >= threshold else " "
                print(f"{score:.3f}{marker} ", end="")
            print()

        print(f"\n(* = above threshold {threshold})")

        # =================================================================
        # STEP 5: Correspondence Matching (Greedy)
        # =================================================================
        print(f"\n{'-'*40}")
        print("STEP 5: Correspondence Matching (Greedy)")
        print(f"{'-'*40}")

        correspondences, _, _ = find_correspondences(
            input_grid, output_grid,
            input_objects, output_objects,
            threshold=threshold,
            use_matchable=True,
            mode=correspondence_mode,
            margin=correspondence_margin
        )

        print(f"\nMatched pairs: {len(correspondences)}")
        matched_input_idxs = set()
        matched_output_idxs = set()

        for in_idx, out_idx, score in correspondences:
            matched_input_idxs.add(in_idx)
            matched_output_idxs.add(out_idx)
            in_obj = input_matchable[in_idx]
            out_obj = output_matchable[out_idx]
            print(f"  Input[{in_idx}] -> Output[{out_idx}], score={score:.4f}")
            print(f"    Input:  id={in_obj.id}, color={in_obj.color}, "
                  f"pos=({in_obj.row},{in_obj.col}), size={in_obj.height}x{in_obj.width}")
            print(f"    Output: id={out_obj.id}, color={out_obj.color}, "
                  f"pos=({out_obj.row},{out_obj.col}), size={out_obj.height}x{out_obj.width}")

        # =================================================================
        # STEP 6: Unmatched Objects Analysis
        # =================================================================
        print(f"\n{'-'*40}")
        print("STEP 6: Unmatched Objects (Novel Detection)")
        print(f"{'-'*40}")

        unmatched_input = [i for i in range(n_in) if i not in matched_input_idxs]
        unmatched_output = [j for j in range(n_out) if j not in matched_output_idxs]

        print(f"\nUnmatched input objects: {len(unmatched_input)}")
        for i in unmatched_input:
            obj = input_matchable[i]
            print(f"  Input[{i}]: id={obj.id}, color={obj.color}, "
                  f"pos=({obj.row},{obj.col}), size={obj.height}x{obj.width}")
            # Show why it didn't match any output
            print(f"    Best output scores:")
            scores_for_input = [(j, similarity_matrix[i, j]) for j in range(n_out)]
            scores_for_input.sort(key=lambda x: -x[1])
            for j, score in scores_for_input[:3]:
                out_obj = output_matchable[j]
                status = "MATCHED to other" if j in matched_output_idxs else "AVAILABLE"
                print(f"      -> Output[{j}] (color={out_obj.color}): {score:.4f} [{status}]")

        print(f"\nUnmatched output objects (NOVEL): {len(unmatched_output)}")
        for j in unmatched_output:
            obj = output_matchable[j]
            print(f"\n  *** NOVEL: Output[{j}] ***")
            print(f"    id={obj.id}, color={obj.color}, pos=({obj.row},{obj.col}), "
                  f"size={obj.height}x{obj.width}, area={obj.area}")

            # Detailed analysis of why this object is novel
            print(f"    WHY NOVEL? Similarity to all input objects:")
            scores_for_output = [(i, similarity_matrix[i, j]) for i in range(n_in)]
            scores_for_output.sort(key=lambda x: -x[1])

            for i, score in scores_for_output:
                in_obj = input_matchable[i]
                status = "MATCHED to other" if i in matched_input_idxs else "AVAILABLE"
                below_thresh = f"BELOW threshold ({threshold})" if score < threshold else "above threshold"
                print(f"      <- Input[{i}] (color={in_obj.color}, size={in_obj.height}x{in_obj.width}): "
                      f"{score:.4f} [{status}] [{below_thresh}]")

                # Show component breakdown for best candidates
                if score == scores_for_output[0][1] or score >= threshold - 0.1:
                    print(f"         Component breakdown:")
                    print(f"           structural: {component_matrices['structural'][i, j]:.4f}")
                    print(f"           moments:    {component_matrices['moments'][i, j]:.4f}")
                    print(f"           color:      {component_matrices['color'][i, j]:.4f}")
                    print(f"           fourier:    {component_matrices['fourier'][i, j]:.4f}")

        # Store results
        example_trace = {
            'example_idx': ex_idx,
            'input_objects': len(input_matchable),
            'output_objects': len(output_matchable),
            'correspondences': correspondences,
            'unmatched_input': unmatched_input,
            'unmatched_output': unmatched_output,
            'novel_count': len(unmatched_output),
            'similarity_matrix': similarity_matrix,
        }
        trace_results['examples'].append(example_trace)

    # =================================================================
    # SUMMARY
    # =================================================================
    print(f"\n{'='*80}")
    print("GENESIS SUMMARY")
    print(f"{'='*80}")

    novel_counts = [ex['novel_count'] for ex in trace_results['examples']]
    print(f"Novel objects per example: {novel_counts}")
    print(f"Total novel objects: {sum(novel_counts)}")

    if sum(novel_counts) > 0:
        print("\nGenesis is TRIGGERED for this puzzle")
        print("Novel objects will be created according to discovered rules")
    else:
        print("\nNo genesis triggered - all output objects have input correspondences")

    return trace_results


def visualize_genesis_for_puzzle(puzzle_id: str, data_root: str = "kaggle/combined",
                                  strategy: Optional[SegmentationStrategy] = None):
    """
    Visualize novel object detection for a puzzle.

    Shows input/output grids with novel objects highlighted.

    Args:
        puzzle_id: The ARC puzzle ID (e.g., "27a77e38")
        data_root: Path to puzzle data
        strategy: Optional segmentation strategy for input/output grids
    """
    import matplotlib.pyplot as plt
    from object_module import extract_objects_from_grid

    print(f"Loading puzzle: {puzzle_id}")
    puzzle = _load_puzzle(puzzle_id, data_root)
    print(f"Found {len(puzzle['train'])} training examples")

    # Determine segmentation modes
    input_mode = strategy.input_mode if strategy else SegmentationMode.CONNECTIVITY
    output_mode = strategy.output_mode if strategy else SegmentationMode.CONNECTIVITY
    if strategy and (input_mode != SegmentationMode.CONNECTIVITY or output_mode != SegmentationMode.CONNECTIVITY):
        print(f"Input segmentation mode: {input_mode.value}")
        print(f"Output segmentation mode: {output_mode.value}")

    # Process each example
    for ex_idx, example in enumerate(puzzle['train']):
        if 'output' not in example:
            continue

        input_grid = np.array(example['input'], dtype=np.int64)
        output_grid = np.array(example['output'], dtype=np.int64)

        print(f"\n{'='*60}")
        print(f"Example {ex_idx}")
        print(f"{'='*60}")
        print(f"Input shape: {input_grid.shape}, Output shape: {output_grid.shape}")

        # Extract objects
        input_objects = extract_objects_from_grid(input_grid, segmentation_mode=input_mode)
        output_objects = extract_objects_from_grid(output_grid, segmentation_mode=output_mode)

        # Filter background
        input_nobg = [o for o in input_objects if not o.is_background]
        output_nobg = [o for o in output_objects if not o.is_background]

        print(f"Input objects (no bg): {len(input_nobg)}")
        print(f"Output objects (no bg): {len(output_nobg)}")

        # Find novel objects
        novel = find_novel_objects(
            input_nobg, output_nobg,
            input_grid, output_grid
        )

        print(f"\nNovel objects found: {len(novel)}")
        for n in novel:
            obj = n.object
            print(f"  - Object at ({obj.row}, {obj.col}): color={obj.color}, "
                  f"size=({obj.height}x{obj.width}), area={obj.area}")

        # Visualize
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))

        _draw_grid(axes[0], input_grid, f"Input ({input_grid.shape[0]}x{input_grid.shape[1]})")
        _draw_grid(axes[1], output_grid, f"Output ({output_grid.shape[0]}x{output_grid.shape[1]})")

        # Highlight novel objects on output grid
        _highlight_novel_objects(axes[1], novel)

        fig.suptitle(f"Puzzle {puzzle_id} - Example {ex_idx}\n"
                     f"Novel objects: {len(novel)}",
                     fontsize=12, fontweight='bold')
        plt.tight_layout()
        plt.show()

    # Run genesis discovery
    print(f"\n{'='*60}")
    print("GENESIS DISCOVERY")
    print(f"{'='*60}")

    result = discover_genesis_rules(puzzle['train'])

    print(f"Novel object count per example: {result.novel_object_count}")
    print(f"Consistent specs found: {len(result.consistent_specs)}")

    if result.best_spec:
        print(f"\nBest specification:")
        print(f"  {result.best_spec.describe()}")
        print(f"  Confidence: {result.best_spec.confidence:.2f}")
        print(f"  Variance: {result.best_spec.variance:.4f}")
    else:
        print("\nNo genesis rules discovered (no novel objects)")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Genesis Module for ARC Puzzles - Novel Object Detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python genesis_module.py --puzzle-id 27a77e38
    python genesis_module.py --puzzle-id 03560426 --trace
    python genesis_module.py --puzzle-id 03560426 --trace --threshold 0.5

    # Different segmentation for input vs output:
    python genesis_module.py --puzzle-id 27a77e38 --input-segmentation-mode connectivity --output-segmentation-mode pixel

    # Different correspondence modes:
    python genesis_module.py --puzzle-id 27a77e38 --trace --correspondence-mode many_to_one --correspondence-margin 0.1

This module detects novel objects (objects that appear in output but have no
corresponding input object) and discovers rules for generating them.

Use --trace for detailed debugging output showing:
  - Object extraction from input/output grids
  - Shape feature comparison between all object pairs
  - Similarity scores and why objects did/didn't match
  - Detailed analysis of why objects were marked as novel
        """
    )

    parser.add_argument("--puzzle-id", type=str,
                        help="ARC puzzle ID to analyze (e.g., 27a77e38)")
    parser.add_argument("--data-root", type=str, default="kaggle/combined",
                        help="Path to puzzle data (default: kaggle/combined)")
    parser.add_argument("--trace", action="store_true",
                        help="Enable detailed tracing of genesis detection")
    parser.add_argument("--threshold", type=float, default=0.3,
                        help="Similarity threshold for correspondence matching (default: 0.3)")
    parser.add_argument("--quiet", action="store_true",
                        help="Less verbose output (skip detailed feature printing)")
    parser.add_argument("--segmentation-mode", type=str,
                        choices=['connectivity', 'pixel', 'color'],
                        default=None,
                        help="Object segmentation mode for BOTH input and output (shorthand). "
                             "Use --input-segmentation-mode and --output-segmentation-mode for different modes.")
    parser.add_argument("--input-segmentation-mode", type=str,
                        choices=['connectivity', 'pixel', 'color'],
                        default=None,
                        help="Segmentation mode for INPUT grid (overrides --segmentation-mode)")
    parser.add_argument("--output-segmentation-mode", type=str,
                        choices=['connectivity', 'pixel', 'color'],
                        default=None,
                        help="Segmentation mode for OUTPUT grid (overrides --segmentation-mode)")
    parser.add_argument("--correspondence-mode", type=str,
                        choices=['one_to_one', 'many_to_one', 'one_to_many'],
                        default='one_to_one',
                        help="Correspondence matching mode: one_to_one (default), "
                             "many_to_one (multiple inputs to one output), "
                             "one_to_many (one input to multiple outputs)")
    parser.add_argument("--correspondence-margin", type=float, default=DEFAULT_MARGIN,
                        help=f"For non-one_to_one modes, how close to best score to allow secondary matches (default: {DEFAULT_MARGIN})")
    parser.add_argument("--test", action="store_true",
                        help="Run basic module tests")

    args = parser.parse_args()

    # Build segmentation strategy from arguments
    # Priority: specific mode > general mode > default (connectivity)
    base_mode = SegmentationMode(args.segmentation_mode) if args.segmentation_mode else SegmentationMode.CONNECTIVITY
    input_mode = SegmentationMode(args.input_segmentation_mode) if args.input_segmentation_mode else base_mode
    output_mode = SegmentationMode(args.output_segmentation_mode) if args.output_segmentation_mode else base_mode
    strategy = SegmentationStrategy(input_mode=input_mode, output_mode=output_mode)

    if args.puzzle_id:
        if args.trace:
            trace_genesis_for_puzzle(
                args.puzzle_id,
                args.data_root,
                threshold=args.threshold,
                verbose=not args.quiet,
                strategy=strategy,
                correspondence_mode=args.correspondence_mode,
                correspondence_margin=args.correspondence_margin
            )
        else:
            visualize_genesis_for_puzzle(args.puzzle_id, args.data_root, strategy=strategy)
    elif args.test:
        # Run basic tests
        print("Genesis Module loaded successfully")
        print(f"Shape types: {[s.name for s in ShapeType]}")
        print(f"Color source types: {[c.name for c in ColorSourceType]}")
        print(f"Position types: {[p.name for p in PositionType]}")

        # Test shape spec
        pixel = ShapeSpec.pixel()
        print(f"\nPixel spec: {pixel}")

        hline = ShapeSpec.hline(5)
        print(f"HLine spec: {hline}")

        # Test color spec
        literal = ColorSpec.literal(3)
        print(f"\nLiteral color: {literal}")

        mode_grid = ColorSpec.mode_of_grid()
        print(f"Mode of grid: {mode_grid}")

        # Test position spec
        grid_pos = PositionSpec.grid_relative(AnchorPoint.BOTTOM_CENTER)
        print(f"\nGrid position: {grid_pos}")

        # Test full object spec
        spec = ObjectSpec(
            color=ColorSpec.mode_of_region_by_name('top_half'),
            shape=ShapeSpec.pixel(),
            position=PositionSpec.grid_relative(AnchorPoint.BOTTOM_CENTER)
        )
        print(f"\nFull spec description: {spec.describe()}")
    else:
        parser.print_help()