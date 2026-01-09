#!/usr/bin/env python3
"""
Multi-Head Correspondence Module for ARC Puzzle Solver

This module unifies sequential_binding_module and regional_correspondence_module
into a single framework based on MULTI-HEAD CORRESPONDENCE, where output features
(color, shape, position) can be derived from different input sources.

Core Insight:
    Many ARC puzzles establish correspondence between input and output through
    ORDERING or SPATIAL TRANSFORMS, with each output feature potentially derived
    from a different input source (multi-head).

Key Concepts:
    - InputPartition: How to segment the input (divider, regions, object type, color)
    - CorrespondenceMode: How to establish correspondence (ordering, transform, pattern-lookup)
    - FeatureDerivation: How each output feature is derived from input sources
    - ExecutionMode: How to generate output (procedural, fill, replication, regional)

Supported Puzzle Types:
    1. Procedural (136b0064): Patterns encode drawing instructions, cumulative position
    2. Pattern-to-fill (17cae0c1): Pattern shapes map to fill colors
    3. Template replication (12997ef3): Template shape + sequential colors
    4. Regional correspondence (103eff5b): Pattern colors + template shape via transform

Usage:
    from multihead_correspondence_module import (
        discover_multihead_rule,
        apply_multihead_rule,
        MultiHeadRule,
    )

    # Discover rule from training examples
    rule = discover_multihead_rule(puzzle, verbose=True)

    # Apply to test input
    output_grid = apply_multihead_rule(test_input, rule)
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

from anchoring_module import (
    AnchorPoint,
    SpatialRelation,
    DiscoveredRelation,
    TestObject,
    ExamplePair,
    discover_relation,
    compute_position_from_relation,
)

from ordering_module import AdaptiveReadingOrder


# =============================================================================
# Enums
# =============================================================================

class PartitionMode(Enum):
    """How to partition the input grid."""
    NONE = auto()                # No partition
    VERTICAL_DIVIDER = auto()    # Split by vertical line (136b0064)
    FIXED_REGIONS = auto()       # Fixed-size regions like 3x3 (17cae0c1)
    OBJECT_TYPE = auto()         # By object properties - template vs sources (12997ef3)
    BY_COLOR = auto()            # Template is one color, pattern is others (103eff5b)


class CorrespondenceMode(Enum):
    """How to establish correspondence between input and output."""
    ORDERED = auto()             # Same position in ordered sequences
    SPATIAL_TRANSFORM = auto()   # Apply transform to logical positions
    PATTERN_LOOKUP = auto()      # Pattern shape maps to features
    DIRECT = auto()              # Direct pixel-to-pixel (color remapping)


class SpatialTransform(Enum):
    """Spatial transformations for position correspondence."""
    IDENTITY = auto()
    ROTATE_90_CW = auto()
    ROTATE_180 = auto()
    ROTATE_90_CCW = auto()
    FLIP_HORIZONTAL = auto()
    FLIP_VERTICAL = auto()
    FLIP_DIAGONAL = auto()       # (r,c) -> (c,r)
    FLIP_ANTIDIAGONAL = auto()


class OrderingMode(Enum):
    """How to order objects for correspondence."""
    ROW_MAJOR = auto()           # Top-to-bottom, left-to-right
    COL_MAJOR = auto()           # Left-to-right, top-to-bottom
    COLUMN_THEN_ROW = auto()     # Left column top-bottom, then right column
    BY_SIZE = auto()
    BY_COLOR = auto()


class ColorDerivation(Enum):
    """How output color is derived."""
    PRESERVE = auto()            # Same as correspondent
    FROM_PATTERN = auto()        # Lookup in pattern vocabulary
    FROM_SEQUENCE = auto()       # From separate color sources in order
    FROM_CORRESPONDENT = auto()  # From corresponding cell via transform
    CONSTANT = auto()
    TRANSFORMED = auto()         # Color is transformed via mapping


class ShapeDerivation(Enum):
    """How output shape is derived."""
    PRESERVE = auto()
    FROM_PATTERN = auto()        # Pattern encodes shape parameters
    FROM_TEMPLATE = auto()       # From a template object
    FILL_REGION = auto()         # Fill a region solid
    FROM_CORRESPONDENT = auto()  # From template cell shape


class PositionDerivation(Enum):
    """How output position is derived."""
    PRESERVE = auto()
    SEQUENTIAL = auto()          # Evenly spaced sequence
    CUMULATIVE = auto()          # Each depends on previous
    IN_PLACE = auto()            # Replace correspondent position


class Direction(Enum):
    """Cardinal directions for drawing."""
    UP = auto()
    DOWN = auto()
    LEFT = auto()
    RIGHT = auto()

    def delta(self) -> Tuple[int, int]:
        return {
            Direction.UP: (-1, 0),
            Direction.DOWN: (1, 0),
            Direction.LEFT: (0, -1),
            Direction.RIGHT: (0, 1),
        }.get(self, (0, 0))


class ExecutionMode(Enum):
    """How to generate output."""
    PROCEDURAL = auto()          # Cumulative drawing (136b0064)
    PATTERN_FILL = auto()        # Fill regions by pattern (17cae0c1)
    TEMPLATE_REPLICATION = auto() # Replicate template with colors (12997ef3)
    REGIONAL_FILL = auto()       # Fill template with pattern colors (103eff5b)
    COLOR_REMAPPING = auto()     # Remap colors based on a legend (0becf7df)
    SEQUENTIAL_PLACEMENT = auto() # Place objects sequentially using anchor relations (03560426)
    GRID_PACKING = auto()        # Pack objects into grid with separators (1990f7a8)


class ColorMappingType(Enum):
    """Type of color mapping derived from a 2x2 legend."""
    ROW_SWAP = auto()       # Swap colors within each row: (0,0)↔(0,1), (1,0)↔(1,1)
    COLUMN_SWAP = auto()    # Swap colors within each column: (0,0)↔(1,0), (0,1)↔(1,1)
    DIAGONAL_SWAP = auto()  # Swap along main diagonal: (0,0)↔(1,1), (0,1)↔(1,0)


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class LogicalCell:
    """A cell with both pixel and logical grid position."""
    pixels: Set[Tuple[int, int]]
    color: int
    row: int  # Pixel row
    col: int  # Pixel col
    height: int
    width: int
    logical_row: int
    logical_col: int

    def __hash__(self):
        return hash((self.logical_row, self.logical_col))


@dataclass
class Region:
    """A region containing cells at logical positions."""
    cells: List[LogicalCell]
    row: int
    col: int
    height: int
    width: int
    logical_height: int
    logical_width: int

    def get_cell_at(self, logical_row: int, logical_col: int) -> Optional[LogicalCell]:
        for cell in self.cells:
            if cell.logical_row == logical_row and cell.logical_col == logical_col:
                return cell
        return None


@dataclass
class PatternEntry:
    """An entry in the pattern vocabulary."""
    pattern: FrozenSet[Tuple[int, int]]
    direction: Optional[Direction] = None
    length: Optional[int] = None
    color: Optional[int] = None


class PatternVocabulary:
    """Learned mapping from pattern shapes to parameters."""

    def __init__(self):
        self.entries: Dict[FrozenSet[Tuple[int, int]], PatternEntry] = {}

    def add(self, pattern: FrozenSet[Tuple[int, int]], **kwargs) -> None:
        self.entries[pattern] = PatternEntry(pattern=pattern, **kwargs)

    def lookup(self, pattern: FrozenSet[Tuple[int, int]]) -> Optional[PatternEntry]:
        return self.entries.get(pattern)

    def normalize_pixels(self, pixels: Set[Tuple[int, int]]) -> FrozenSet[Tuple[int, int]]:
        if not pixels:
            return frozenset()
        min_r = min(r for r, c in pixels)
        min_c = min(c for r, c in pixels)
        return frozenset((r - min_r, c - min_c) for r, c in pixels)


@dataclass
class ColorRemappingSpec:
    """Specification for color remapping based on a 2x2 legend.

    The legend is a 2x2 grid in a corner of the input that defines
    color swap pairs. The mapping_type determines how pairs are formed:
    - ROW_SWAP: (0,0)↔(0,1), (1,0)↔(1,1)
    - COLUMN_SWAP: (0,0)↔(1,0), (0,1)↔(1,1)
    - DIAGONAL_SWAP: (0,0)↔(1,1), (0,1)↔(1,0)
    """
    legend_row: int  # Top-left row of the legend
    legend_col: int  # Top-left col of the legend
    mapping_type: ColorMappingType
    color_map: Dict[int, int] = field(default_factory=dict)  # Derived color mapping

    def get_mapped_color(self, color: int) -> int:
        """Return the mapped color, or the original if not in mapping."""
        return self.color_map.get(color, color)


@dataclass
class InputPartition:
    """Specification for how to partition the input."""
    mode: PartitionMode
    divider_col: Optional[int] = None
    divider_row: Optional[int] = None
    divider_color: Optional[int] = None
    region_size: Optional[Tuple[int, int]] = None
    template_color: Optional[int] = None
    template_criterion: Optional[str] = None


@dataclass
class CorrespondenceSpec:
    """Specification for how correspondence is established."""
    mode: CorrespondenceMode
    ordering: Optional[OrderingMode] = None
    transform: Optional[SpatialTransform] = None


@dataclass
class FeatureSpec:
    """Specification for how features are derived."""
    color_derivation: ColorDerivation
    shape_derivation: ShapeDerivation
    position_derivation: PositionDerivation
    implicit_step: Optional[Direction] = None  # For procedural


@dataclass
class MultiHeadRule:
    """Complete rule for multi-head correspondence transformation."""
    partition: InputPartition
    correspondence: CorrespondenceSpec
    features: FeatureSpec
    execution_mode: ExecutionMode
    pattern_vocab: PatternVocabulary = field(default_factory=PatternVocabulary)

    # For template-based puzzles
    template_pixels: Optional[FrozenSet[Tuple[int, int]]] = None

    # For color remapping puzzles
    color_remapping: Optional[ColorRemappingSpec] = None

    # For sequential placement puzzles
    anchor_relation: Optional[SpatialRelation] = None

    # Output specification
    output_shape: Optional[Tuple[int, int]] = None
    background_color: int = 0

    # Metadata
    variance: float = float('inf')

    def describe(self) -> str:
        return (f"MultiHeadRule(\n"
                f"  partition={self.partition.mode.name},\n"
                f"  correspondence={self.correspondence.mode.name},\n"
                f"  execution={self.execution_mode.name},\n"
                f"  color={self.features.color_derivation.name},\n"
                f"  shape={self.features.shape_derivation.name}\n"
                f")")


# =============================================================================
# Pre-Extracted Data Structures
# =============================================================================

@dataclass
class ByColorExtraction:
    """Extracted data for BY_COLOR partition mode."""
    template_color: int
    pattern_regions: List[Region]  # One per training example
    template_regions: List[Region]  # One per training example
    best_transforms: Dict[SpatialTransform, float]  # transform -> variance


@dataclass
class ProceduralExtraction:
    """Extracted data for PROCEDURAL execution mode."""
    divider_col: int
    divider_color: int
    vocab: PatternVocabulary


@dataclass
class TemplateExtraction:
    """Extracted data for TEMPLATE_REPLICATION execution mode."""
    template_object: Object
    template_pixels: FrozenSet[Tuple[int, int]]
    color_sources: List[Object]  # From first example as reference


@dataclass
class LegendExtraction:
    """Extracted data for COLOR_REMAPPING execution mode."""
    legend_row: int
    legend_col: int
    best_mapping_type: ColorMappingType
    mapping_accuracy: float


@dataclass
class SequentialPlacementExtraction:
    """Extracted data for SEQUENTIAL_PLACEMENT execution mode.

    Objects are placed sequentially, with each object's position determined
    by an anchor relationship to the previous object. The relationship is
    discovered automatically using the anchoring module.

    Example relationships this can represent:
    - Diagonal chain: source.TL at target.BR + (0,0)
    - Horizontal stack: source.TL at target.TR + (0,1)
    - Vertical stack: source.TL at target.BL + (1,0)
    """
    ordering_mode: OrderingMode  # How objects are ordered (e.g., left-to-right)
    anchor_relation: SpatialRelation  # Relationship between consecutive objects
    accuracy: float  # How well this rule explains the outputs


@dataclass
class GridPackingExtraction:
    """Extracted data for GRID_PACKING execution mode (1990f7a8 style).

    Objects scattered in the input are packed into a compact grid in the output.
    Uses adaptive reading order to determine the 2D grid arrangement:
    - Objects are clustered into rows based on vertical gaps
    - Within each row, objects are ordered left-to-right
    - Output grid has separators between rows/columns

    Example: 4 objects scattered in input → 2x2 grid with separator in output
    """
    grid_rows: int  # Number of rows in the output grid
    grid_cols: int  # Number of columns in the output grid
    cell_height: int  # Height of each cell (object bounding box)
    cell_width: int  # Width of each cell (object bounding box)
    separator_size: int  # Size of separator between cells (usually 1)
    accuracy: float  # How well this rule explains the outputs


@dataclass
class PreExtractedData:
    """All potentially useful structures extracted from training examples."""

    # BY_COLOR data: keyed by potential template color
    by_color: Dict[int, ByColorExtraction]

    # FIXED_REGIONS data
    fill_vocab: Optional[PatternVocabulary]
    region_size: Optional[Tuple[int, int]]

    # VERTICAL_DIVIDER / PROCEDURAL data
    procedural: Optional[ProceduralExtraction]

    # OBJECT_TYPE / TEMPLATE_REPLICATION data
    template: Optional[TemplateExtraction]

    # COLOR_REMAPPING data
    legend: Optional[LegendExtraction]

    # Common
    output_shape: Tuple[int, int]

    # Fields with defaults must come last
    sequential_placement: Optional[SequentialPlacementExtraction] = None
    grid_packing: Optional[GridPackingExtraction] = None
    background_color: int = 0


# =============================================================================
# Pre-Extraction Functions
# =============================================================================

def pre_extract_structures(
    examples: List[dict],
    verbose: bool = False
) -> PreExtractedData:
    """Extract all potentially useful structures from training examples.

    This function extracts everything we might need, regardless of which
    partition/execution mode ends up being used. The screening phase will
    determine which extractions are actually useful.
    """
    if not examples:
        return PreExtractedData(
            by_color={},
            fill_vocab=None,
            region_size=None,
            procedural=None,
            template=None,
            legend=None,
            output_shape=(1, 1)
        )

    first_input = np.array(examples[0]['input'])
    first_output = np.array(examples[0]['output'])
    output_shape = first_output.shape

    if verbose:
        print("Pre-extracting structures...")

    # Extract BY_COLOR data for all potential template colors
    by_color_data = extract_by_color_structures(examples, verbose)

    # Extract FIXED_REGIONS data
    fill_vocab, region_size = extract_fixed_regions_structures(examples, verbose)

    # Extract PROCEDURAL data
    procedural_data = extract_procedural_structures(examples, verbose)

    # Extract TEMPLATE_REPLICATION data
    template_data = extract_template_structures(examples, verbose)

    # Extract COLOR_REMAPPING data
    legend_data = extract_legend_structures(examples, verbose)

    # Extract SEQUENTIAL_PLACEMENT data
    sequential_placement_data = extract_sequential_placement_structures(examples, verbose)

    # Extract GRID_PACKING data
    grid_packing_data = extract_grid_packing_structures(examples, verbose)

    return PreExtractedData(
        by_color=by_color_data,
        fill_vocab=fill_vocab,
        region_size=region_size,
        procedural=procedural_data,
        template=template_data,
        legend=legend_data,
        sequential_placement=sequential_placement_data,
        grid_packing=grid_packing_data,
        output_shape=output_shape
    )


def extract_by_color_structures(
    examples: List[dict],
    verbose: bool = False
) -> Dict[int, ByColorExtraction]:
    """Extract BY_COLOR partition data for all potential template colors."""
    result = {}

    first_input = np.array(examples[0]['input'])
    first_output = np.array(examples[0]['output'])

    # Find colors that are significantly reduced in output (potential templates)
    input_colors = Counter(first_input.flatten())
    output_colors = Counter(first_output.flatten())
    input_colors.pop(0, None)
    output_colors.pop(0, None)

    candidate_colors = []
    for color, input_count in input_colors.items():
        output_count = output_colors.get(color, 0)
        # Color significantly reduced = potential template
        if input_count > 10 and output_count < input_count * 0.5:
            candidate_colors.append(color)

    if verbose and candidate_colors:
        print(f"  BY_COLOR candidates: {candidate_colors}")

    for template_color in candidate_colors:
        pattern_regions = []
        template_regions = []

        for ex in examples:
            input_grid = np.array(ex['input'])
            output_grid = np.array(ex['output'])

            pattern_region = extract_pattern_region_by_color(input_grid, template_color, verbose=False)
            template_region = extract_template_region_by_color(input_grid, template_color, verbose=False)

            if pattern_region and template_region:
                pattern_regions.append(pattern_region)
                template_regions.append(template_region)

        if len(pattern_regions) == len(examples):
            # Screen all transforms for this template color
            transform_variances = {}
            for transform in SpatialTransform:
                total_errors = 0
                total_cells = 0

                for i, ex in enumerate(examples):
                    output_grid = np.array(ex['output'])
                    pattern_region = pattern_regions[i]
                    template_region = template_regions[i]

                    logical_h = max(pattern_region.logical_height, template_region.logical_height)
                    logical_w = max(pattern_region.logical_width, template_region.logical_width)

                    for template_cell in template_region.cells:
                        t_pos = (template_cell.logical_row, template_cell.logical_col)
                        p_pos = apply_spatial_transform(t_pos, transform, (logical_h, logical_w))
                        pattern_cell = pattern_region.get_cell_at(p_pos[0], p_pos[1])

                        sample_r, sample_c = next(iter(template_cell.pixels))
                        if 0 <= sample_r < output_grid.shape[0] and 0 <= sample_c < output_grid.shape[1]:
                            actual_color = output_grid[sample_r, sample_c]
                        else:
                            continue

                        if pattern_cell is not None:
                            expected_color = pattern_cell.color
                        else:
                            expected_color = get_fallback_color(pattern_region, t_pos, transform, (logical_h, logical_w))
                            if expected_color is None:
                                if actual_color == 0:
                                    continue
                                total_errors += 1
                                total_cells += 1
                                continue

                        if actual_color != expected_color:
                            total_errors += 1
                        total_cells += 1

                variance = total_errors / total_cells if total_cells > 0 else float('inf')
                transform_variances[transform] = variance

            result[template_color] = ByColorExtraction(
                template_color=template_color,
                pattern_regions=pattern_regions,
                template_regions=template_regions,
                best_transforms=transform_variances
            )

            if verbose:
                best = min(transform_variances.items(), key=lambda x: x[1])
                print(f"    Color {template_color}: best transform {best[0].name} (var={best[1]:.4f})")

    return result


def extract_fixed_regions_structures(
    examples: List[dict],
    verbose: bool = False
) -> Tuple[Optional[PatternVocabulary], Optional[Tuple[int, int]]]:
    """Extract FIXED_REGIONS partition data."""
    first_input = np.array(examples[0]['input'])
    first_output = np.array(examples[0]['output'])
    H, W = first_input.shape

    # Check if 3x3 region structure applies
    if first_input.shape != first_output.shape or H % 3 != 0 or W % 3 != 0:
        return None, None

    # Check if output has uniform regions
    is_region_based = True
    for r in range(0, H, 3):
        for c in range(0, W, 3):
            region = first_output[r:r+3, c:c+3]
            if len(set(region.flatten())) > 1:
                is_region_based = False
                break
        if not is_region_based:
            break

    if not is_region_based:
        return None, None

    if verbose:
        print("  FIXED_REGIONS: 3x3 structure detected")

    # Learn pattern vocabulary
    vocab = learn_pattern_vocabulary_fill(examples, verbose=False)

    return vocab, (3, 3)


def extract_procedural_structures(
    examples: List[dict],
    verbose: bool = False
) -> Optional[ProceduralExtraction]:
    """Extract VERTICAL_DIVIDER / PROCEDURAL data."""
    first_input = np.array(examples[0]['input'])
    _, W = first_input.shape

    # Find vertical divider
    divider_col = None
    divider_color = None
    for c in range(1, W - 1):
        col = first_input[:, c]
        unique = set(col)
        if len(unique) == 1 and 0 not in unique:
            color = int(col[0])
            left_has_content = np.any(first_input[:, :c] > 0)
            right_has_content = np.any(first_input[:, c+1:] > 0)
            if left_has_content and right_has_content:
                divider_col = c
                divider_color = color
                break

    if divider_col is None:
        return None

    if verbose:
        print(f"  PROCEDURAL: divider at col {divider_col}, color {divider_color}")

    # Build partition for vocab learning
    partition = InputPartition(
        mode=PartitionMode.VERTICAL_DIVIDER,
        divider_col=divider_col,
        divider_color=divider_color
    )
    vocab = learn_pattern_vocabulary_procedural(examples, partition, verbose=False)

    return ProceduralExtraction(
        divider_col=divider_col,
        divider_color=divider_color,
        vocab=vocab
    )


def extract_template_structures(
    examples: List[dict],
    verbose: bool = False
) -> Optional[TemplateExtraction]:
    """Extract OBJECT_TYPE / TEMPLATE_REPLICATION data."""
    first_input = np.array(examples[0]['input'])

    objects = extract_objects_from_grid(first_input, segmentation_mode=SegmentationMode.CONNECTIVITY)
    objects = [o for o in objects if not o.is_background and o.color > 0]

    if len(objects) < 2:
        return None

    # Find template (largest non-single-pixel object)
    template = find_template_object(objects)
    if template is None:
        return None

    # Check size disparity
    sizes = [len(o.pixels) for o in objects]
    if max(sizes) <= 3 * min(sizes):
        return None

    # Normalize template pixels
    min_r = min(r for r, c in template.pixels)
    min_c = min(c for r, c in template.pixels)
    template_pixels = frozenset((r - min_r, c - min_c) for r, c in template.pixels)

    # Find color sources
    color_sources = find_color_sources(objects, template)

    if verbose:
        print(f"  TEMPLATE: {len(template.pixels)} pixels, {len(color_sources)} color sources")

    return TemplateExtraction(
        template_object=template,
        template_pixels=template_pixels,
        color_sources=color_sources
    )


def extract_legend_structures(
    examples: List[dict],
    verbose: bool = False
) -> Optional[LegendExtraction]:
    """Extract COLOR_REMAPPING data."""
    # Check if all examples have a consistent legend
    legends = []
    for ex in examples:
        input_grid = np.array(ex['input'])
        legend_info = detect_legend(input_grid, verbose=False)
        if legend_info is None:
            return None
        legends.append(legend_info)

    # Check legends are at consistent positions
    legend_rows = set(l[0] for l in legends)
    legend_cols = set(l[1] for l in legends)
    if len(legend_rows) != 1 or len(legend_cols) != 1:
        return None

    legend_row, legend_col = legends[0][0], legends[0][1]

    # Try each mapping type and find best
    best_mapping_type = None
    best_accuracy = -1

    for mapping_type in ColorMappingType:
        total_pixels = 0
        correct_pixels = 0

        for i, ex in enumerate(examples):
            input_grid = np.array(ex['input'])
            output_grid = np.array(ex['output'])

            legend = legends[i][2]
            color_map = build_color_map_from_legend(legend, mapping_type)

            remapping = ColorRemappingSpec(
                legend_row=legend_row,
                legend_col=legend_col,
                mapping_type=mapping_type,
                color_map=color_map
            )
            predicted = apply_color_remapping(input_grid, remapping, verbose=False)

            total_pixels += output_grid.size
            correct_pixels += np.sum(predicted == output_grid)

        accuracy = correct_pixels / total_pixels if total_pixels > 0 else 0

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_mapping_type = mapping_type

    if best_mapping_type is None or best_accuracy < 0.5:
        return None

    if verbose:
        print(f"  LEGEND: at ({legend_row},{legend_col}), {best_mapping_type.name} ({best_accuracy:.1%})")

    return LegendExtraction(
        legend_row=legend_row,
        legend_col=legend_col,
        best_mapping_type=best_mapping_type,
        mapping_accuracy=best_accuracy
    )


def extract_sequential_placement_structures(
    examples: List[dict],
    verbose: bool = False
) -> Optional[SequentialPlacementExtraction]:
    """Extract SEQUENTIAL_PLACEMENT data using the anchoring module.

    Detects puzzles where objects are placed sequentially with a consistent
    anchor relationship between consecutive objects. Uses the anchoring module
    to discover the relationship automatically.

    This generalizes diagonal chains, horizontal stacks, vertical stacks, etc.
    """
    if not examples:
        return None

    # Check basic structure: input and output same size, multiple distinct objects
    first_input = np.array(examples[0]['input'])
    first_output = np.array(examples[0]['output'])

    if first_input.shape != first_output.shape:
        return None

    # Extract objects from first input
    objects = extract_objects_from_grid(first_input, segmentation_mode=SegmentationMode.CONNECTIVITY)
    objects = [o for o in objects if not o.is_background and o.color > 0]

    # Need at least 2 objects
    if len(objects) < 2:
        return None

    # All objects should have distinct colors
    colors = set(o.color for o in objects)
    if len(colors) != len(objects):
        return None

    # Try different ordering strategies
    best_ordering = None
    best_relation = None
    best_accuracy = 0.0

    for ordering_mode in [OrderingMode.COL_MAJOR, OrderingMode.ROW_MAJOR]:
        # Collect consecutive object pairs from output grids across all examples
        example_pairs = []

        for ex in examples:
            input_grid = np.array(ex['input'])
            output_grid = np.array(ex['output'])

            # Extract objects from input (to determine ordering)
            in_objs = extract_objects_from_grid(input_grid, segmentation_mode=SegmentationMode.CONNECTIVITY)
            in_objs = [o for o in in_objs if not o.is_background and o.color > 0]

            # Order input objects
            if ordering_mode == OrderingMode.COL_MAJOR:
                ordered_in = sorted(in_objs, key=lambda o: (o.col, o.row))
            else:
                ordered_in = sorted(in_objs, key=lambda o: (o.row, o.col))

            # Find corresponding objects in output by color
            out_objs = extract_objects_from_grid(output_grid, segmentation_mode=SegmentationMode.CONNECTIVITY)
            out_objs = [o for o in out_objs if not o.is_background and o.color > 0]
            color_to_out = {o.color: o for o in out_objs}

            # Get ordered output objects (by matching colors from ordered input)
            ordered_out = []
            for in_obj in ordered_in:
                if in_obj.color in color_to_out:
                    ordered_out.append(color_to_out[in_obj.color])

            if len(ordered_out) < 2:
                continue

            # Create ExamplePairs for consecutive objects
            # source = object[i], target = object[i-1]
            for i in range(1, len(ordered_out)):
                source_obj = ordered_out[i]
                target_obj = ordered_out[i - 1]

                source_test = TestObject(
                    top_left=(source_obj.row, source_obj.col),
                    size=(source_obj.height, source_obj.width),
                    object_id=source_obj.id
                )
                target_test = TestObject(
                    top_left=(target_obj.row, target_obj.col),
                    size=(target_obj.height, target_obj.width),
                    object_id=target_obj.id
                )
                example_pairs.append(ExamplePair(source=source_test, target=target_test))

        if not example_pairs:
            continue

        # Use anchoring module to discover the best relationship
        discovered = discover_relation(example_pairs, top_k=1)

        if not discovered:
            continue

        best_discovered = discovered[0]

        # Only accept if variance is low (consistent relationship)
        if best_discovered.variance > 0.5:
            continue

        # Evaluate accuracy by generating outputs
        total_correct = 0
        total_pixels = 0

        for ex in examples:
            input_grid = np.array(ex['input'])
            output_grid = np.array(ex['output'])

            predicted = generate_sequential_placement_output(
                input_grid, ordering_mode, best_discovered.relation, output_grid.shape
            )

            total_correct += np.sum(predicted == output_grid)
            total_pixels += output_grid.size

        accuracy = total_correct / total_pixels if total_pixels > 0 else 0

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_ordering = ordering_mode
            best_relation = best_discovered.relation

    if best_accuracy < 0.8 or best_relation is None:
        return None

    if verbose:
        print(f"  SEQUENTIAL_PLACEMENT: {best_ordering.name} ordering, "
              f"{best_relation.source_anchor.value.upper()}->{best_relation.target_anchor.value.upper()} "
              f"offset={best_relation.offset}, accuracy={best_accuracy:.1%}")

    return SequentialPlacementExtraction(
        ordering_mode=best_ordering,
        anchor_relation=best_relation,
        accuracy=best_accuracy
    )


def generate_sequential_placement_output(
    input_grid: np.ndarray,
    ordering_mode: OrderingMode,
    relation: SpatialRelation,
    output_shape: Tuple[int, int]
) -> np.ndarray:
    """Generate output by placing objects sequentially using anchor relationship.

    Uses compute_position_from_relation from anchoring module.
    """
    output = np.zeros(output_shape, dtype=np.int64)

    # Extract and order objects from input
    objects = extract_objects_from_grid(input_grid, segmentation_mode=SegmentationMode.CONNECTIVITY)
    objects = [o for o in objects if not o.is_background and o.color > 0]

    if not objects:
        return output

    if ordering_mode == OrderingMode.COL_MAJOR:
        ordered = sorted(objects, key=lambda o: (o.col, o.row))
    elif ordering_mode == OrderingMode.ROW_MAJOR:
        ordered = sorted(objects, key=lambda o: (o.row, o.col))
    else:
        ordered = sorted(objects, key=lambda o: (o.col, o.row))

    # Place first object at (0, 0)
    prev_pos = (0, 0)
    prev_size = None

    for i, obj in enumerate(ordered):
        # Get object dimensions
        rows = [r for r, c in obj.pixels]
        cols = [c for r, c in obj.pixels]
        obj_height = max(rows) - min(rows) + 1
        obj_width = max(cols) - min(cols) + 1
        obj_size = (obj_height, obj_width)

        # Normalize pixels to origin
        min_r = min(rows)
        min_c = min(cols)
        normalized_pixels = [(r - min_r, c - min_c) for r, c in obj.pixels]

        if i == 0:
            # First object goes at (0, 0)
            place_pos = (0, 0)
        else:
            # Use anchor relationship to compute position relative to previous object
            place_pos = compute_position_from_relation(
                relation,
                obj_size,
                prev_pos,
                prev_size
            )

        # Place object at computed position
        for dr, dc in normalized_pixels:
            out_r = place_pos[0] + dr
            out_c = place_pos[1] + dc
            if 0 <= out_r < output_shape[0] and 0 <= out_c < output_shape[1]:
                output[out_r, out_c] = obj.color

        # Update for next iteration
        prev_pos = place_pos
        prev_size = obj_size

    return output


def extract_grid_packing_structures(
    examples: List[dict],
    verbose: bool = False
) -> Optional[GridPackingExtraction]:
    """Extract GRID_PACKING data using adaptive reading order (1990f7a8 style).

    Detects puzzles where objects scattered in input are packed into a compact
    grid in the output with separators between cells.

    Detection criteria:
    - Multiple objects of the same color in input
    - Output is smaller than input
    - Output has a regular grid structure with separators
    - Objects can be ordered using adaptive reading order
    """
    if not examples:
        return None

    first_input = np.array(examples[0]['input'])
    first_output = np.array(examples[0]['output'])

    # Output should be smaller than input (compaction)
    if first_output.size >= first_input.size:
        return None

    # Extract objects from first input
    objects = extract_objects_from_grid(first_input, segmentation_mode=SegmentationMode.CONNECTIVITY)
    objects = [o for o in objects if not o.is_background and o.color > 0]

    # Need at least 2 objects
    if len(objects) < 2:
        return None

    # Use adaptive reading order to determine grid structure
    adaptive_order = AdaptiveReadingOrder()
    ordered_objects = adaptive_order.order(objects)
    row_structure = adaptive_order.get_row_structure(objects)

    # Determine grid dimensions from row structure
    grid_rows = len(row_structure)
    grid_cols = max(len(row) for row in row_structure) if row_structure else 1

    # Need at least 2x2 structure (or 1xN / Nx1 with N > 2)
    n_objects = len(ordered_objects)
    if n_objects < 2:
        return None

    # Compute max object bounding box
    max_height = max(o.height for o in objects)
    max_width = max(o.width for o in objects)

    # Try different separator sizes (0, 1, 2)
    best_sep = 1
    best_accuracy = 0.0

    for sep_size in [0, 1, 2]:
        expected_height = grid_rows * max_height + (grid_rows - 1) * sep_size
        expected_width = grid_cols * max_width + (grid_cols - 1) * sep_size

        # Check if output dimensions match
        if first_output.shape[0] != expected_height or first_output.shape[1] != expected_width:
            continue

        # Evaluate accuracy
        total_correct = 0
        total_pixels = 0

        for ex in examples:
            input_grid = np.array(ex['input'])
            output_grid = np.array(ex['output'])

            predicted = generate_grid_packing_output(
                input_grid, grid_rows, grid_cols, max_height, max_width, sep_size
            )

            if predicted.shape == output_grid.shape:
                total_correct += np.sum(predicted == output_grid)
                total_pixels += output_grid.size
            else:
                total_pixels += output_grid.size

        accuracy = total_correct / total_pixels if total_pixels > 0 else 0

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_sep = sep_size

    if best_accuracy < 0.8:
        return None

    if verbose:
        print(f"  GRID_PACKING: {grid_rows}x{grid_cols} grid, cell={max_height}x{max_width}, "
              f"sep={best_sep}, accuracy={best_accuracy:.1%}")

    return GridPackingExtraction(
        grid_rows=grid_rows,
        grid_cols=grid_cols,
        cell_height=max_height,
        cell_width=max_width,
        separator_size=best_sep,
        accuracy=best_accuracy
    )


def generate_grid_packing_output(
    input_grid: np.ndarray,
    grid_rows: int,
    grid_cols: int,
    cell_height: int,
    cell_width: int,
    separator_size: int
) -> np.ndarray:
    """Generate output by packing objects into a grid with separators.

    Uses adaptive reading order to determine which object goes where.
    """
    # Calculate output dimensions
    output_height = grid_rows * cell_height + (grid_rows - 1) * separator_size
    output_width = grid_cols * cell_width + (grid_cols - 1) * separator_size
    output = np.zeros((output_height, output_width), dtype=np.int64)

    # Extract and order objects
    objects = extract_objects_from_grid(input_grid, segmentation_mode=SegmentationMode.CONNECTIVITY)
    objects = [o for o in objects if not o.is_background and o.color > 0]

    if not objects:
        return output

    adaptive_order = AdaptiveReadingOrder()
    ordered_objects = adaptive_order.order(objects)

    # Place each object in its grid cell
    for idx, obj in enumerate(ordered_objects):
        # Determine grid position for this object
        grid_row = idx // grid_cols
        grid_col = idx % grid_cols

        if grid_row >= grid_rows:
            break  # More objects than cells

        # Calculate output position for this cell
        out_row = grid_row * (cell_height + separator_size)
        out_col = grid_col * (cell_width + separator_size)

        # Normalize object pixels to origin
        min_r = min(r for r, c in obj.pixels)
        min_c = min(c for r, c in obj.pixels)

        # Place object pixels
        for r, c in obj.pixels:
            dr = r - min_r
            dc = c - min_c
            if dr < cell_height and dc < cell_width:
                out_r = out_row + dr
                out_c = out_col + dc
                if 0 <= out_r < output_height and 0 <= out_c < output_width:
                    output[out_r, out_c] = obj.color

    return output


# =============================================================================
# Spatial Transform Helpers
# =============================================================================

def apply_spatial_transform(
    pos: Tuple[int, int],
    transform: SpatialTransform,
    grid_size: Tuple[int, int]
) -> Tuple[int, int]:
    """Apply a spatial transform to a logical position."""
    r, c = pos
    h, w = grid_size

    if transform == SpatialTransform.IDENTITY:
        return (r, c)
    elif transform == SpatialTransform.ROTATE_90_CW:
        return (c, h - 1 - r)
    elif transform == SpatialTransform.ROTATE_180:
        return (h - 1 - r, w - 1 - c)
    elif transform == SpatialTransform.ROTATE_90_CCW:
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
# Partitioning Functions
# =============================================================================

def find_vertical_divider(grid: np.ndarray) -> Tuple[int, int]:
    """Find vertical divider column and color."""
    H, W = grid.shape
    for c in range(1, W - 1):
        col = grid[:, c]
        unique = set(col)
        if len(unique) == 1 and 0 not in unique:
            return c, int(col[0])
    return W // 2, 0


def detect_by_color_partition(
    examples: List[dict],
    verbose: bool = False
) -> Optional[InputPartition]:
    """Detect BY_COLOR partition (103eff5b style)."""
    if not examples:
        return None

    first_input = np.array(examples[0]['input'])
    first_output = np.array(examples[0]['output'])

    input_colors = Counter(first_input.flatten())
    output_colors = Counter(first_output.flatten())
    input_colors.pop(0, None)
    output_colors.pop(0, None)

    for color, input_count in input_colors.items():
        output_count = output_colors.get(color, 0)
        # Color significantly reduced = template that gets filled
        if input_count > 20 and output_count < input_count * 0.3:
            if verbose:
                print(f"BY_COLOR: color {color} reduced from {input_count} to {output_count}")
            return InputPartition(mode=PartitionMode.BY_COLOR, template_color=color)

    return None


def detect_fixed_regions_partition(
    examples: List[dict],
    verbose: bool = False
) -> Optional[InputPartition]:
    """Detect FIXED_REGIONS partition (17cae0c1 style)."""
    if not examples:
        return None

    first_input = np.array(examples[0]['input'])
    first_output = np.array(examples[0]['output'])
    H, W = first_input.shape

    if first_input.shape == first_output.shape and H % 3 == 0 and W % 3 == 0:
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
                print("FIXED_REGIONS: 3x3 region structure detected")
            return InputPartition(mode=PartitionMode.FIXED_REGIONS, region_size=(3, 3))

    return None


def detect_vertical_divider_partition(
    examples: List[dict],
    verbose: bool = False
) -> Optional[InputPartition]:
    """Detect VERTICAL_DIVIDER partition (136b0064 style)."""
    if not examples:
        return None

    first_input = np.array(examples[0]['input'])
    _, W = first_input.shape

    for c in range(1, W - 1):
        col = first_input[:, c]
        unique = set(col)
        if len(unique) == 1 and 0 not in unique:
            color = int(col[0])
            left_has_content = np.any(first_input[:, :c] > 0)
            right_has_content = np.any(first_input[:, c+1:] > 0)
            if left_has_content and right_has_content:
                if verbose:
                    print(f"VERTICAL_DIVIDER: column {c}, color {color}")
                return InputPartition(
                    mode=PartitionMode.VERTICAL_DIVIDER,
                    divider_col=c,
                    divider_color=color
                )

    return None


def detect_object_type_partition(
    examples: List[dict],
    verbose: bool = False
) -> Optional[InputPartition]:
    """Detect OBJECT_TYPE partition (12997ef3 style)."""
    if not examples:
        return None

    first_input = np.array(examples[0]['input'])

    objects = extract_objects_from_grid(first_input, segmentation_mode=SegmentationMode.CONNECTIVITY)
    objects = [o for o in objects if not o.is_background and o.color > 0]

    if len(objects) >= 2:
        sizes = [len(o.pixels) for o in objects]
        if max(sizes) > 3 * min(sizes):
            if verbose:
                print("OBJECT_TYPE: template + color sources")
            return InputPartition(mode=PartitionMode.OBJECT_TYPE, template_criterion="largest")

    return None


def detect_partition(
    examples: List[dict],
    verbose: bool = False
) -> InputPartition:
    """Detect how to partition input grids (legacy function for compatibility)."""
    # Try each partition type in order
    partition = detect_by_color_partition(examples, verbose)
    if partition:
        return partition

    partition = detect_fixed_regions_partition(examples, verbose)
    if partition:
        return partition

    partition = detect_vertical_divider_partition(examples, verbose)
    if partition:
        return partition

    partition = detect_object_type_partition(examples, verbose)
    if partition:
        return partition

    return InputPartition(mode=PartitionMode.NONE)


# =============================================================================
# Region Extraction
# =============================================================================

def extract_pattern_region_by_color(
    grid: np.ndarray,
    template_color: int,
    verbose: bool = False
) -> Optional[Region]:
    """Extract pattern region (non-template colored pixels as a logical grid).

    For BY_COLOR mode, the pattern is a grid of individual colored cells.
    We extract at pixel level to preserve the logical grid structure,
    rather than using connected components which would merge adjacent cells.
    """
    H, W = grid.shape

    # Find all pattern pixels (non-0, non-template)
    pattern_pixels = []
    for r in range(H):
        for c in range(W):
            if grid[r, c] > 0 and grid[r, c] != template_color:
                pattern_pixels.append((r, c, grid[r, c]))

    if not pattern_pixels:
        return None

    # Get unique row/col positions
    row_positions = sorted(set(r for r, c, _ in pattern_pixels))
    col_positions = sorted(set(c for r, c, _ in pattern_pixels))

    row_to_logical = {r: i for i, r in enumerate(row_positions)}
    col_to_logical = {c: i for i, c in enumerate(col_positions)}

    # Group pixels into logical cells
    # Key: (logical_row, logical_col), Value: (pixels, color)
    cell_data = {}
    for r, c, color in pattern_pixels:
        logical_r = row_to_logical[r]
        logical_c = col_to_logical[c]
        key = (logical_r, logical_c)

        if key not in cell_data:
            cell_data[key] = (set(), color)
        cell_data[key][0].add((r, c))

    cells = []
    for (logical_r, logical_c), (pixels, color) in cell_data.items():
        min_r = min(r for r, c in pixels)
        min_c = min(c for r, c in pixels)
        max_r = max(r for r, c in pixels)
        max_c = max(c for r, c in pixels)

        cell = LogicalCell(
            pixels=pixels,
            color=color,
            row=min_r,
            col=min_c,
            height=max_r - min_r + 1,
            width=max_c - min_c + 1,
            logical_row=logical_r,
            logical_col=logical_c
        )
        cells.append(cell)

        if verbose:
            print(f"  Pattern cell at ({min_r},{min_c}), color={color} -> logical ({logical_r},{logical_c})")

    min_row = min(row_positions)
    max_row = max(row_positions)
    min_col = min(col_positions)
    max_col = max(col_positions)

    return Region(
        cells=cells,
        row=min_row,
        col=min_col,
        height=max_row - min_row + 1,
        width=max_col - min_col + 1,
        logical_height=len(row_positions),
        logical_width=len(col_positions)
    )


def detect_cell_size(pixels: Set[Tuple[int, int]]) -> Tuple[int, int]:
    """Detect cell size in a template region."""
    if not pixels:
        return (1, 1)

    rows = sorted(set(r for r, c in pixels))
    cols = sorted(set(c for r, c in pixels))

    if len(rows) < 2 or len(cols) < 2:
        return (len(rows), len(cols))

    # Find run lengths
    col_run_lengths = []
    for r in rows:
        row_cols = sorted(c for (pr, c) in pixels if pr == r)
        if not row_cols:
            continue
        run_len = 1
        for i in range(1, len(row_cols)):
            if row_cols[i] == row_cols[i-1] + 1:
                run_len += 1
            else:
                col_run_lengths.append(run_len)
                run_len = 1
        col_run_lengths.append(run_len)

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

    def find_fundamental_size(run_lengths):
        if not run_lengths:
            return 2
        valid_runs = [r for r in run_lengths if r >= 2]
        if not valid_runs:
            return 2
        counter = Counter(valid_runs)
        unique_sizes = sorted(counter.keys())
        if len(unique_sizes) == 1:
            return unique_sizes[0]
        smallest = unique_sizes[0]
        for size in unique_sizes[1:]:
            if size % smallest == 0:
                return smallest
        return counter.most_common(1)[0][0]

    return (find_fundamental_size(row_run_lengths), find_fundamental_size(col_run_lengths))


def extract_template_region_by_color(
    grid: np.ndarray,
    template_color: int,
    verbose: bool = False
) -> Optional[Region]:
    """Extract template region (single-color object split into logical cells)."""
    objects = extract_objects_from_grid(grid, segmentation_mode=SegmentationMode.CONNECTIVITY)
    template_objects = [o for o in objects if o.color == template_color and not o.is_background]

    if not template_objects:
        return None

    # Usually one connected template object
    if len(template_objects) == 1:
        template_obj = template_objects[0]
        pixels = set(template_obj.pixels)

        rows = sorted(set(r for r, c in pixels))
        cols = sorted(set(c for r, c in pixels))
        min_r, max_r = min(rows), max(rows)
        min_c, max_c = min(cols), max(cols)

        cell_h, cell_w = detect_cell_size(pixels)

        if verbose:
            print(f"  Template bounding box: ({min_r},{min_c}) to ({max_r},{max_c})")
            print(f"  Detected cell size: {cell_h}x{cell_w}")

        cells = []
        logical_row = 0
        r = min_r
        while r <= max_r:
            logical_col = 0
            c = min_c
            while c <= max_c:
                cell_pixels = set()
                for dr in range(cell_h):
                    for dc in range(cell_w):
                        if (r + dr, c + dc) in pixels:
                            cell_pixels.add((r + dr, c + dc))

                if len(cell_pixels) >= (cell_h * cell_w) // 2:
                    cell = LogicalCell(
                        pixels=cell_pixels,
                        color=template_color,
                        row=r,
                        col=c,
                        height=cell_h,
                        width=cell_w,
                        logical_row=logical_row,
                        logical_col=logical_col
                    )
                    cells.append(cell)
                    if verbose:
                        print(f"    Template cell at pixel ({r},{c}) -> logical ({logical_row},{logical_col})")

                c += cell_w
                logical_col += 1
            r += cell_h
            logical_row += 1

        if not cells:
            return None

        logical_height = max(c.logical_row for c in cells) + 1
        logical_width = max(c.logical_col for c in cells) + 1

        return Region(
            cells=cells,
            row=min_r,
            col=min_c,
            height=max_r - min_r + 1,
            width=max_c - min_c + 1,
            logical_height=logical_height,
            logical_width=logical_width
        )

    return None


# =============================================================================
# Correspondence Discovery
# =============================================================================

def screen_spatial_transforms(
    pattern_region: Region,
    template_region: Region,
    output_grid: np.ndarray,
    verbose: bool = False
) -> Tuple[Optional[SpatialTransform], float]:
    """Screen transforms to find one with zero variance."""
    transforms = list(SpatialTransform)

    logical_h = max(pattern_region.logical_height, template_region.logical_height)
    logical_w = max(pattern_region.logical_width, template_region.logical_width)

    best_transform = None
    best_variance = float('inf')

    for transform in transforms:
        errors = 0
        total = 0

        for template_cell in template_region.cells:
            t_pos = (template_cell.logical_row, template_cell.logical_col)
            p_pos = apply_spatial_transform(t_pos, transform, (logical_h, logical_w))

            pattern_cell = pattern_region.get_cell_at(p_pos[0], p_pos[1])

            sample_r, sample_c = next(iter(template_cell.pixels))
            if 0 <= sample_r < output_grid.shape[0] and 0 <= sample_c < output_grid.shape[1]:
                actual_color = output_grid[sample_r, sample_c]
            else:
                continue

            if pattern_cell is not None:
                expected_color = pattern_cell.color
            else:
                # Fallback: look for pattern cells in same row/col
                expected_color = get_fallback_color(pattern_region, t_pos, transform, (logical_h, logical_w))
                if expected_color is None:
                    if actual_color == 0:
                        continue
                    errors += 1
                    total += 1
                    continue

            if actual_color != expected_color:
                errors += 1
            total += 1

        variance = errors / total if total > 0 else float('inf')

        if verbose:
            print(f"  Transform {transform.name}: errors={errors}/{total}, var={variance:.4f}")

        if variance < best_variance:
            best_variance = variance
            best_transform = transform

    return best_transform, best_variance


def get_fallback_color(
    pattern_region: Region,
    template_pos: Tuple[int, int],
    transform: SpatialTransform,
    grid_size: Tuple[int, int]
) -> Optional[int]:
    """Find fallback color when no direct correspondence exists."""
    t_row, t_col = template_pos

    # Find pattern cells mapping to same template row
    row_colors = []
    for cell in pattern_region.cells:
        mapped = apply_spatial_transform((cell.logical_row, cell.logical_col), transform, grid_size)
        if mapped[0] == t_row:
            row_colors.append(cell.color)

    if row_colors:
        return Counter(row_colors).most_common(1)[0][0]

    # Try column
    col_colors = []
    for cell in pattern_region.cells:
        mapped = apply_spatial_transform((cell.logical_row, cell.logical_col), transform, grid_size)
        if mapped[1] == t_col:
            col_colors.append(cell.color)

    if col_colors:
        return Counter(col_colors).most_common(1)[0][0]

    return None


# =============================================================================
# Pattern Vocabulary Learning (for procedural/pattern-to-fill)
# =============================================================================

def normalize_object_shape(obj: Object, grid: Optional[np.ndarray] = None) -> FrozenSet[Tuple[int, int]]:
    """Normalize object pixels to origin."""
    if not obj.pixels:
        return frozenset()

    if grid is not None:
        actual_pixels = {(r, c) for r, c in obj.pixels
                        if 0 <= r < grid.shape[0] and 0 <= c < grid.shape[1]
                        and grid[r, c] == obj.color}
    else:
        actual_pixels = obj.pixels

    if not actual_pixels:
        return frozenset()

    min_r = min(r for r, c in actual_pixels)
    min_c = min(c for r, c in actual_pixels)
    return frozenset((r - min_r, c - min_c) for r, c in actual_pixels)


def infer_direction_length(pattern: FrozenSet[Tuple[int, int]]) -> Tuple[Direction, int]:
    """Infer direction and length from pattern shape."""
    if not pattern:
        return Direction.RIGHT, 1

    rows = [r for r, c in pattern]
    cols = [c for r, c in pattern]
    min_r, max_r = min(rows), max(rows)
    min_c, max_c = min(cols), max(cols)
    height = max_r - min_r + 1
    width = max_c - min_c + 1

    top_row = sum(1 for r, c in pattern if r == min_r)
    bottom_row = sum(1 for r, c in pattern if r == max_r)
    left_col = sum(1 for r, c in pattern if c == min_c)
    right_col = sum(1 for r, c in pattern if c == max_c)

    vertical_asymmetry = bottom_row - top_row
    horizontal_asymmetry = left_col - right_col

    if abs(horizontal_asymmetry) > abs(vertical_asymmetry):
        direction = Direction.RIGHT if horizontal_asymmetry > 0 else Direction.LEFT
        length = width
    else:
        if vertical_asymmetry > 0:
            direction = Direction.UP
        elif vertical_asymmetry < 0:
            direction = Direction.DOWN
        else:
            direction = Direction.UP
        length = height

    n_pixels = len(pattern)
    if n_pixels <= 4:
        length = 2
    elif n_pixels <= 5:
        length = 3
    elif n_pixels <= 6:
        length = 4
    else:
        length = 2

    return direction, length


def learn_pattern_vocabulary_procedural(
    examples: List[dict],
    partition: InputPartition,
    verbose: bool = False
) -> PatternVocabulary:
    """Learn pattern vocabulary for procedural puzzles."""
    vocab = PatternVocabulary()

    div_col = partition.divider_col
    if div_col is None:
        return vocab

    all_patterns = set()
    pattern_observations = {}

    for ex in examples:
        input_grid = np.array(ex['input'])
        output_grid = np.array(ex['output'])

        instruction_region = input_grid[:, :div_col]
        objects = extract_objects_from_grid(instruction_region, segmentation_mode=SegmentationMode.CONNECTIVITY)
        objects = [o for o in objects if not o.is_background and o.color > 0]

        for obj in objects:
            normalized = normalize_object_shape(obj, instruction_region)
            all_patterns.add(normalized)

        # Extract output segments for correspondence learning
        segments = extract_output_segments(output_grid)

        # Match patterns to segments by color
        pattern_by_color = {}
        for obj in objects:
            normalized = normalize_object_shape(obj, instruction_region)
            if obj.color not in pattern_by_color:
                pattern_by_color[obj.color] = []
            pattern_by_color[obj.color].append(normalized)

        segment_by_color = {}
        for seg_color, seg_dir, seg_len in segments:
            if seg_color not in segment_by_color:
                segment_by_color[seg_color] = []
            segment_by_color[seg_color].append((seg_dir, seg_len))

        for color in pattern_by_color:
            if color not in segment_by_color:
                continue
            for i, pattern in enumerate(pattern_by_color[color]):
                if i < len(segment_by_color[color]):
                    seg_dir, seg_len = segment_by_color[color][i]
                    if pattern not in pattern_observations:
                        pattern_observations[pattern] = []
                    pattern_observations[pattern].append((seg_dir, seg_len))

    for pattern in all_patterns:
        if pattern in pattern_observations and pattern_observations[pattern]:
            counts = Counter(pattern_observations[pattern])
            direction, length = counts.most_common(1)[0][0]
        else:
            direction, length = infer_direction_length(pattern)
        vocab.add(pattern, direction=direction, length=length)

    return vocab


def extract_output_segments(
    output_grid: np.ndarray,
    start_marker_color: int = 5
) -> List[Tuple[int, Direction, int]]:
    """Extract drawn segments from output."""
    H, W = output_grid.shape
    segments = []

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

    visited = {start_pos}
    row, col = start_pos

    while True:
        next_row = row + 1
        if next_row >= H:
            break

        segment_start = None
        segment_color = None

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

        seg_row, seg_col = segment_start
        color = segment_color
        visited.add((seg_row, seg_col))

        directions_found = {}
        for direction in [Direction.RIGHT, Direction.LEFT, Direction.UP, Direction.DOWN]:
            dr, dc = direction.delta()
            length = 1
            r, c = seg_row + dr, seg_col + dc
            while 0 <= r < H and 0 <= c < W and output_grid[r, c] == color and (r, c) not in visited:
                length += 1
                visited.add((r, c))
                r, c = r + dr, c + dc
            if length > 1:
                directions_found[direction] = length

        if directions_found:
            primary_dir = max(directions_found.keys(), key=lambda d: directions_found[d])
            total_length = directions_found[primary_dir]
        else:
            primary_dir = Direction.RIGHT
            total_length = 1

        segments.append((color, primary_dir, total_length))

        dr, dc = primary_dir.delta()
        row = seg_row + dr * (total_length - 1)
        col = seg_col + dc * (total_length - 1)

    return segments


def learn_pattern_vocabulary_fill(
    examples: List[dict],
    verbose: bool = False
) -> PatternVocabulary:
    """Learn pattern vocabulary for pattern-to-fill puzzles."""
    vocab = PatternVocabulary()

    for ex in examples:
        input_grid = np.array(ex['input'])
        output_grid = np.array(ex['output'])
        H, W = input_grid.shape

        for region_r in range(0, H, 3):
            for region_c in range(0, W, 3):
                input_region = input_grid[region_r:region_r+3, region_c:region_c+3]
                output_region = output_grid[region_r:region_r+3, region_c:region_c+3]

                pattern_pixels = set()
                for r in range(min(3, input_region.shape[0])):
                    for c in range(min(3, input_region.shape[1])):
                        if input_region[r, c] > 0:
                            pattern_pixels.add((r, c))

                if not pattern_pixels:
                    continue

                fill_colors = set(output_region.flatten())
                if len(fill_colors) == 1:
                    fill_color = int(fill_colors.pop())
                else:
                    fill_color = int(Counter(output_region.flatten()).most_common(1)[0][0])

                pattern = frozenset(pattern_pixels)
                if pattern not in vocab.entries:
                    vocab.add(pattern, color=fill_color)
                    if verbose:
                        print(f"Learned pattern {pattern} -> color {fill_color}")

    return vocab


# =============================================================================
# Discovery Function
# =============================================================================

def evaluate_rule(rule: MultiHeadRule, examples: List[dict]) -> float:
    """Evaluate rule accuracy on training examples.

    Returns accuracy as a float between 0.0 and 1.0.
    """
    total_pixels = 0
    correct_pixels = 0

    for ex in examples:
        input_grid = np.array(ex['input'])
        expected = np.array(ex['output'])

        try:
            predicted = apply_multihead_rule(input_grid, rule)

            # Shape mismatch = bad rule
            if predicted.shape != expected.shape:
                # Count all pixels as wrong for shape mismatch
                total_pixels += expected.size
                continue

            correct_pixels += np.sum(predicted == expected)
            total_pixels += expected.size
        except Exception:
            # Exception during application = bad rule
            total_pixels += expected.size
            continue

    return correct_pixels / total_pixels if total_pixels > 0 else 0.0


def build_rule_from_extraction(
    extracted: PreExtractedData,
    execution_mode: ExecutionMode,
    template_color: Optional[int] = None,
    transform: Optional[SpatialTransform] = None,
) -> Optional[MultiHeadRule]:
    """Build a rule from pre-extracted data for a specific execution mode.

    This replaces the discover_xxx_rule functions with a unified builder
    that assembles rules from pre-extracted components.
    """
    if execution_mode == ExecutionMode.REGIONAL_FILL:
        if template_color is None or template_color not in extracted.by_color:
            return None
        if transform is None:
            return None

        by_color = extracted.by_color[template_color]

        return MultiHeadRule(
            partition=InputPartition(mode=PartitionMode.BY_COLOR, template_color=template_color),
            correspondence=CorrespondenceSpec(
                mode=CorrespondenceMode.SPATIAL_TRANSFORM,
                transform=transform
            ),
            features=FeatureSpec(
                color_derivation=ColorDerivation.FROM_CORRESPONDENT,
                shape_derivation=ShapeDerivation.FROM_CORRESPONDENT,
                position_derivation=PositionDerivation.IN_PLACE
            ),
            execution_mode=ExecutionMode.REGIONAL_FILL,
            output_shape=extracted.output_shape,
            variance=by_color.best_transforms.get(transform, float('inf'))
        )

    elif execution_mode == ExecutionMode.PATTERN_FILL:
        if extracted.fill_vocab is None or extracted.region_size is None:
            return None

        return MultiHeadRule(
            partition=InputPartition(mode=PartitionMode.FIXED_REGIONS, region_size=extracted.region_size),
            correspondence=CorrespondenceSpec(mode=CorrespondenceMode.PATTERN_LOOKUP),
            features=FeatureSpec(
                color_derivation=ColorDerivation.FROM_PATTERN,
                shape_derivation=ShapeDerivation.FILL_REGION,
                position_derivation=PositionDerivation.IN_PLACE
            ),
            execution_mode=ExecutionMode.PATTERN_FILL,
            pattern_vocab=extracted.fill_vocab,
            output_shape=extracted.output_shape
        )

    elif execution_mode == ExecutionMode.PROCEDURAL:
        if extracted.procedural is None:
            return None

        return MultiHeadRule(
            partition=InputPartition(
                mode=PartitionMode.VERTICAL_DIVIDER,
                divider_col=extracted.procedural.divider_col,
                divider_color=extracted.procedural.divider_color
            ),
            correspondence=CorrespondenceSpec(
                mode=CorrespondenceMode.ORDERED,
                ordering=OrderingMode.COLUMN_THEN_ROW
            ),
            features=FeatureSpec(
                color_derivation=ColorDerivation.PRESERVE,
                shape_derivation=ShapeDerivation.FROM_PATTERN,
                position_derivation=PositionDerivation.CUMULATIVE,
                implicit_step=Direction.DOWN
            ),
            execution_mode=ExecutionMode.PROCEDURAL,
            pattern_vocab=extracted.procedural.vocab,
            output_shape=extracted.output_shape
        )

    elif execution_mode == ExecutionMode.TEMPLATE_REPLICATION:
        if extracted.template is None:
            return None

        return MultiHeadRule(
            partition=InputPartition(mode=PartitionMode.OBJECT_TYPE, template_criterion="largest"),
            correspondence=CorrespondenceSpec(
                mode=CorrespondenceMode.ORDERED,
                ordering=OrderingMode.ROW_MAJOR
            ),
            features=FeatureSpec(
                color_derivation=ColorDerivation.FROM_SEQUENCE,
                shape_derivation=ShapeDerivation.FROM_TEMPLATE,
                position_derivation=PositionDerivation.SEQUENTIAL
            ),
            execution_mode=ExecutionMode.TEMPLATE_REPLICATION,
            template_pixels=extracted.template.template_pixels,
            output_shape=extracted.output_shape
        )

    elif execution_mode == ExecutionMode.COLOR_REMAPPING:
        if extracted.legend is None:
            return None

        # Build color map from first example (will be rebuilt during application)
        return MultiHeadRule(
            partition=InputPartition(mode=PartitionMode.NONE),
            correspondence=CorrespondenceSpec(mode=CorrespondenceMode.DIRECT),
            features=FeatureSpec(
                color_derivation=ColorDerivation.TRANSFORMED,
                shape_derivation=ShapeDerivation.PRESERVE,
                position_derivation=PositionDerivation.PRESERVE
            ),
            execution_mode=ExecutionMode.COLOR_REMAPPING,
            color_remapping=ColorRemappingSpec(
                legend_row=extracted.legend.legend_row,
                legend_col=extracted.legend.legend_col,
                mapping_type=extracted.legend.best_mapping_type,
                color_map={}  # Will be built per-input during application
            ),
            output_shape=extracted.output_shape,
            variance=1.0 - extracted.legend.mapping_accuracy
        )

    elif execution_mode == ExecutionMode.SEQUENTIAL_PLACEMENT:
        if extracted.sequential_placement is None:
            return None

        return MultiHeadRule(
            partition=InputPartition(mode=PartitionMode.NONE),
            correspondence=CorrespondenceSpec(
                mode=CorrespondenceMode.ORDERED,
                ordering=extracted.sequential_placement.ordering_mode
            ),
            features=FeatureSpec(
                color_derivation=ColorDerivation.PRESERVE,
                shape_derivation=ShapeDerivation.PRESERVE,
                position_derivation=PositionDerivation.CUMULATIVE
            ),
            execution_mode=ExecutionMode.SEQUENTIAL_PLACEMENT,
            output_shape=extracted.output_shape,
            variance=1.0 - extracted.sequential_placement.accuracy,
            anchor_relation=extracted.sequential_placement.anchor_relation
        )

    elif execution_mode == ExecutionMode.GRID_PACKING:
        if extracted.grid_packing is None:
            return None

        gp = extracted.grid_packing
        # Calculate output shape from grid packing params
        out_height = gp.grid_rows * gp.cell_height + (gp.grid_rows - 1) * gp.separator_size
        out_width = gp.grid_cols * gp.cell_width + (gp.grid_cols - 1) * gp.separator_size

        return MultiHeadRule(
            partition=InputPartition(mode=PartitionMode.NONE),
            correspondence=CorrespondenceSpec(
                mode=CorrespondenceMode.ORDERED,
                ordering=OrderingMode.ROW_MAJOR  # Uses adaptive reading order internally
            ),
            features=FeatureSpec(
                color_derivation=ColorDerivation.PRESERVE,
                shape_derivation=ShapeDerivation.PRESERVE,
                position_derivation=PositionDerivation.SEQUENTIAL
            ),
            execution_mode=ExecutionMode.GRID_PACKING,
            output_shape=(out_height, out_width),
            variance=1.0 - gp.accuracy
        )

    return None


def discover_multihead_rule(
    puzzle: Dict,
    verbose: bool = False
) -> Optional[MultiHeadRule]:
    """Discover the multi-head correspondence rule by screening all combinations.

    Uses pre-extraction to gather all potentially useful structures, then
    systematically screens all valid combinations of execution modes and
    parameters. Returns the rule with highest accuracy.
    """
    train_examples = puzzle.get('train', [])
    if not train_examples:
        return None

    if verbose:
        print("=" * 60)
        print("Discovering Multi-Head Correspondence Rule")
        print("=" * 60)

    # Phase 1: Pre-extract all potentially useful structures
    extracted = pre_extract_structures(train_examples, verbose)

    if verbose:
        print("\nPre-extraction complete:")
        print(f"  BY_COLOR candidates: {list(extracted.by_color.keys())}")
        print(f"  FIXED_REGIONS: {'yes' if extracted.fill_vocab else 'no'}")
        print(f"  PROCEDURAL: {'yes' if extracted.procedural else 'no'}")
        print(f"  TEMPLATE: {'yes' if extracted.template else 'no'}")
        print(f"  LEGEND: {'yes' if extracted.legend else 'no'}")
        print(f"  SEQUENTIAL_PLACEMENT: {'yes' if extracted.sequential_placement else 'no'}")
        print(f"  GRID_PACKING: {'yes' if extracted.grid_packing else 'no'}")
        print()

    # Phase 2: Screen all valid combinations
    candidates = []

    # Screen REGIONAL_FILL with all template colors and transforms
    for template_color, by_color_data in extracted.by_color.items():
        for transform, variance in by_color_data.best_transforms.items():
            rule = build_rule_from_extraction(
                extracted,
                ExecutionMode.REGIONAL_FILL,
                template_color=template_color,
                transform=transform
            )
            if rule:
                accuracy = evaluate_rule(rule, train_examples)
                rule.variance = 1.0 - accuracy

                if verbose:
                    print(f"  REGIONAL_FILL (color={template_color}, {transform.name}): {accuracy:.1%}")

                if accuracy == 1.0:
                    if verbose:
                        print("\n  Perfect rule found! Early terminating.")
                    return rule

                candidates.append((accuracy, rule))

    # Screen PATTERN_FILL
    rule = build_rule_from_extraction(extracted, ExecutionMode.PATTERN_FILL)
    if rule:
        accuracy = evaluate_rule(rule, train_examples)
        rule.variance = 1.0 - accuracy

        if verbose:
            print(f"  PATTERN_FILL: {accuracy:.1%}")

        if accuracy == 1.0:
            if verbose:
                print("\n  Perfect rule found! Early terminating.")
            return rule

        candidates.append((accuracy, rule))

    # Screen PROCEDURAL
    rule = build_rule_from_extraction(extracted, ExecutionMode.PROCEDURAL)
    if rule:
        accuracy = evaluate_rule(rule, train_examples)
        rule.variance = 1.0 - accuracy

        if verbose:
            print(f"  PROCEDURAL: {accuracy:.1%}")

        if accuracy == 1.0:
            if verbose:
                print("\n  Perfect rule found! Early terminating.")
            return rule

        candidates.append((accuracy, rule))

    # Screen TEMPLATE_REPLICATION
    rule = build_rule_from_extraction(extracted, ExecutionMode.TEMPLATE_REPLICATION)
    if rule:
        accuracy = evaluate_rule(rule, train_examples)
        rule.variance = 1.0 - accuracy

        if verbose:
            print(f"  TEMPLATE_REPLICATION: {accuracy:.1%}")

        if accuracy == 1.0:
            if verbose:
                print("\n  Perfect rule found! Early terminating.")
            return rule

        candidates.append((accuracy, rule))

    # Screen COLOR_REMAPPING
    rule = build_rule_from_extraction(extracted, ExecutionMode.COLOR_REMAPPING)
    if rule:
        accuracy = evaluate_rule(rule, train_examples)
        rule.variance = 1.0 - accuracy

        if verbose:
            print(f"  COLOR_REMAPPING: {accuracy:.1%}")

        if accuracy == 1.0:
            if verbose:
                print("\n  Perfect rule found! Early terminating.")
            return rule

        candidates.append((accuracy, rule))

    # Screen SEQUENTIAL_PLACEMENT
    rule = build_rule_from_extraction(extracted, ExecutionMode.SEQUENTIAL_PLACEMENT)
    if rule:
        accuracy = evaluate_rule(rule, train_examples)
        rule.variance = 1.0 - accuracy

        if verbose:
            print(f"  SEQUENTIAL_PLACEMENT: {accuracy:.1%}")

        if accuracy == 1.0:
            if verbose:
                print("\n  Perfect rule found! Early terminating.")
            return rule

        candidates.append((accuracy, rule))

    # Screen GRID_PACKING
    rule = build_rule_from_extraction(extracted, ExecutionMode.GRID_PACKING)
    if rule:
        accuracy = evaluate_rule(rule, train_examples)
        rule.variance = 1.0 - accuracy

        if verbose:
            print(f"  GRID_PACKING: {accuracy:.1%}")

        if accuracy == 1.0:
            if verbose:
                print("\n  Perfect rule found! Early terminating.")
            return rule

        candidates.append((accuracy, rule))

    # Phase 3: Pick best rule
    if not candidates:
        if verbose:
            print("\nNo valid rules discovered")
        return None

    best_accuracy, best_rule = max(candidates, key=lambda x: x[0])

    if verbose:
        print(f"\nBest rule: {best_rule.execution_mode.name}")
        print(f"Accuracy: {best_accuracy:.1%}")

    return best_rule


def find_template_object(objects: List[Object]) -> Optional[Object]:
    """Find the template object (largest non-single-pixel)."""
    candidates = [o for o in objects if len(o.pixels) > 1 and o.color > 0 and not o.is_background]
    if not candidates:
        return None

    for obj in candidates:
        if obj.color == 1:
            return obj

    return max(candidates, key=lambda o: len(o.pixels))


def find_color_sources(objects: List[Object], template: Object) -> List[Object]:
    """Find color source objects."""
    sources = []
    for obj in objects:
        if obj.color == template.color or obj.color <= 0 or obj.is_background:
            continue
        if len(obj.pixels) > len(template.pixels) * 0.5:
            continue
        sources.append(obj)
    return sorted(sources, key=lambda o: (o.row, o.col))


# =============================================================================
# Color Remapping Functions
# =============================================================================

def detect_legend(
    input_grid: np.ndarray,
    verbose: bool = False
) -> Optional[Tuple[int, int, np.ndarray]]:
    """Detect a 2x2 legend in the top-left corner of the grid.

    Returns (row, col, legend_array) if found, None otherwise.
    The legend must have 4 distinct non-zero colors.
    """
    H, W = input_grid.shape
    if H < 2 or W < 2:
        return None

    # Check top-left corner
    legend = input_grid[0:2, 0:2]
    colors = set(legend.flatten())

    # Must have 4 distinct non-zero colors
    if 0 in colors:
        colors.discard(0)
    if len(colors) == 4 and 0 not in legend.flatten():
        if verbose:
            print(f"  Found legend at (0, 0): {legend.tolist()}")
        return (0, 0, legend)

    return None


def build_color_map_from_legend(
    legend: np.ndarray,
    mapping_type: ColorMappingType
) -> Dict[int, int]:
    """Build a bidirectional color mapping from a 2x2 legend."""
    color_map = {}

    if mapping_type == ColorMappingType.ROW_SWAP:
        # Swap within rows: (0,0)↔(0,1), (1,0)↔(1,1)
        color_map[legend[0, 0]] = legend[0, 1]
        color_map[legend[0, 1]] = legend[0, 0]
        color_map[legend[1, 0]] = legend[1, 1]
        color_map[legend[1, 1]] = legend[1, 0]
    elif mapping_type == ColorMappingType.COLUMN_SWAP:
        # Swap within columns: (0,0)↔(1,0), (0,1)↔(1,1)
        color_map[legend[0, 0]] = legend[1, 0]
        color_map[legend[1, 0]] = legend[0, 0]
        color_map[legend[0, 1]] = legend[1, 1]
        color_map[legend[1, 1]] = legend[0, 1]
    elif mapping_type == ColorMappingType.DIAGONAL_SWAP:
        # Swap along diagonals: (0,0)↔(1,1), (0,1)↔(1,0)
        color_map[legend[0, 0]] = legend[1, 1]
        color_map[legend[1, 1]] = legend[0, 0]
        color_map[legend[0, 1]] = legend[1, 0]
        color_map[legend[1, 0]] = legend[0, 1]

    return color_map


def apply_color_remapping(
    input_grid: np.ndarray,
    remapping: ColorRemappingSpec,
    verbose: bool = False
) -> np.ndarray:
    """Apply color remapping to the input grid.

    The legend itself is preserved; all other pixels are remapped.
    """
    output_grid = input_grid.copy()
    H, W = input_grid.shape

    legend_rows = {remapping.legend_row, remapping.legend_row + 1}
    legend_cols = {remapping.legend_col, remapping.legend_col + 1}

    for r in range(H):
        for c in range(W):
            # Skip the legend itself
            if r in legend_rows and c in legend_cols:
                continue

            old_color = input_grid[r, c]
            new_color = remapping.get_mapped_color(old_color)
            output_grid[r, c] = new_color

    return output_grid


# =============================================================================
# Application Functions
# =============================================================================

def apply_multihead_rule(
    input_grid: np.ndarray,
    rule: MultiHeadRule,
    verbose: bool = False
) -> np.ndarray:
    """Apply a multi-head rule to produce output."""
    if rule.execution_mode == ExecutionMode.REGIONAL_FILL:
        return apply_regional_fill(input_grid, rule, verbose)
    elif rule.execution_mode == ExecutionMode.PATTERN_FILL:
        return apply_pattern_fill(input_grid, rule, verbose)
    elif rule.execution_mode == ExecutionMode.PROCEDURAL:
        return apply_procedural(input_grid, rule, verbose)
    elif rule.execution_mode == ExecutionMode.TEMPLATE_REPLICATION:
        return apply_template_replication(input_grid, rule, verbose)
    elif rule.execution_mode == ExecutionMode.COLOR_REMAPPING:
        return apply_multihead_color_remapping(input_grid, rule, verbose)
    elif rule.execution_mode == ExecutionMode.SEQUENTIAL_PLACEMENT:
        return apply_sequential_placement(input_grid, rule, verbose)
    elif rule.execution_mode == ExecutionMode.GRID_PACKING:
        return apply_grid_packing(input_grid, rule, verbose)

    return np.zeros_like(input_grid)


def apply_multihead_color_remapping(
    input_grid: np.ndarray,
    rule: MultiHeadRule,
    verbose: bool = False
) -> np.ndarray:
    """Apply color remapping rule (0becf7df style).

    Rebuilds the color map from this input's legend, then applies the mapping.
    """
    if rule.color_remapping is None:
        return input_grid.copy()

    # Detect legend in this input and build color map
    legend_info = detect_legend(input_grid, verbose)
    if legend_info is None:
        # Fall back to stored color map
        return apply_color_remapping(input_grid, rule.color_remapping, verbose)

    # Build color map for this input's legend
    legend = legend_info[2]
    color_map = build_color_map_from_legend(legend, rule.color_remapping.mapping_type)

    remapping = ColorRemappingSpec(
        legend_row=rule.color_remapping.legend_row,
        legend_col=rule.color_remapping.legend_col,
        mapping_type=rule.color_remapping.mapping_type,
        color_map=color_map
    )

    return apply_color_remapping(input_grid, remapping, verbose)


def apply_regional_fill(
    input_grid: np.ndarray,
    rule: MultiHeadRule,
    verbose: bool = False
) -> np.ndarray:
    """Apply regional fill rule (103eff5b style)."""
    output_grid = input_grid.copy()

    pattern_region = extract_pattern_region_by_color(input_grid, rule.partition.template_color, verbose)
    template_region = extract_template_region_by_color(input_grid, rule.partition.template_color, verbose)

    if pattern_region is None or template_region is None:
        return output_grid

    logical_h = max(pattern_region.logical_height, template_region.logical_height)
    logical_w = max(pattern_region.logical_width, template_region.logical_width)

    transform = rule.correspondence.transform

    for template_cell in template_region.cells:
        t_pos = (template_cell.logical_row, template_cell.logical_col)
        p_pos = apply_spatial_transform(t_pos, transform, (logical_h, logical_w))

        pattern_cell = pattern_region.get_cell_at(p_pos[0], p_pos[1])

        if pattern_cell is not None:
            fill_color = pattern_cell.color
        else:
            fill_color = get_fallback_color(pattern_region, t_pos, transform, (logical_h, logical_w))
            if fill_color is None:
                fill_color = 0

        for r, c in template_cell.pixels:
            if 0 <= r < output_grid.shape[0] and 0 <= c < output_grid.shape[1]:
                output_grid[r, c] = fill_color

    return output_grid


def apply_pattern_fill(
    input_grid: np.ndarray,
    rule: MultiHeadRule,
    verbose: bool = False
) -> np.ndarray:
    """Apply pattern fill rule (17cae0c1 style)."""
    output_shape = rule.output_shape if rule.output_shape else input_grid.shape
    output_grid = np.zeros(output_shape, dtype=np.int64)
    H, W = input_grid.shape

    for region_r in range(0, H, 3):
        for region_c in range(0, W, 3):
            region = input_grid[region_r:min(region_r+3, H), region_c:min(region_c+3, W)]

            pattern_pixels = set()
            for r in range(region.shape[0]):
                for c in range(region.shape[1]):
                    if region[r, c] > 0:
                        pattern_pixels.add((r, c))

            if not pattern_pixels:
                continue

            pattern = frozenset(pattern_pixels)
            entry = rule.pattern_vocab.lookup(pattern)

            if entry and entry.color is not None:
                fill_color = entry.color
            else:
                continue

            for r in range(region_r, min(region_r + 3, output_shape[0])):
                for c in range(region_c, min(region_c + 3, output_shape[1])):
                    output_grid[r, c] = fill_color

    return output_grid


def apply_procedural(
    input_grid: np.ndarray,
    rule: MultiHeadRule,
    verbose: bool = False
) -> np.ndarray:
    """Apply procedural rule (136b0064 style)."""
    div_col = rule.partition.divider_col
    if div_col is None:
        div_col, _ = find_vertical_divider(input_grid)

    instruction_region = input_grid[:, :div_col]

    objects = extract_objects_from_grid(instruction_region, segmentation_mode=SegmentationMode.CONNECTIVITY)
    objects = [o for o in objects if not o.is_background and o.color > 0]

    # Order by column-then-row
    mid_col = div_col // 2
    left = sorted([o for o in objects if o.col < mid_col], key=lambda o: o.row)
    right = sorted([o for o in objects if o.col >= mid_col], key=lambda o: o.row)
    ordered = left + right

    # Find start position
    start_row, start_col = 0, 0
    H, W = input_grid.shape
    for r in range(H):
        for c in range(div_col + 1, W):
            if input_grid[r, c] == 5:
                start_row, start_col = r, c - div_col - 1
                break

    output_height = H
    output_width = W - div_col - 1
    output_grid = np.zeros((output_height, output_width), dtype=np.int64)

    row, col = start_row, start_col

    if 0 <= row < output_height and 0 <= col < output_width:
        output_grid[row, col] = 5

    for obj in ordered:
        normalized = normalize_object_shape(obj, instruction_region)
        entry = rule.pattern_vocab.lookup(normalized)

        if entry and entry.direction:
            direction = entry.direction
            length = entry.length or 1
        else:
            direction, length = infer_direction_length(normalized)

        color = obj.color

        # Implicit step down
        if rule.features.implicit_step:
            dr, dc = rule.features.implicit_step.delta()
            row, col = row + dr, col + dc

        if 0 <= row < output_height and 0 <= col < output_width:
            output_grid[row, col] = color

        dr, dc = direction.delta()
        for _ in range(length - 1):
            row, col = row + dr, col + dc
            if 0 <= row < output_height and 0 <= col < output_width:
                output_grid[row, col] = color

    return output_grid


def apply_template_replication(
    input_grid: np.ndarray,
    rule: MultiHeadRule,
    verbose: bool = False
) -> np.ndarray:
    """Apply template replication rule (12997ef3 style)."""
    objects = extract_objects_from_grid(input_grid, segmentation_mode=SegmentationMode.CONNECTIVITY)
    objects = [o for o in objects if not o.is_background and o.color > 0]

    template = find_template_object(objects)
    if template is None:
        return np.zeros_like(input_grid)

    color_sources = find_color_sources(objects, template)

    # Extract template from current input
    min_r = min(r for r, c in template.pixels)
    min_c = min(c for r, c in template.pixels)
    template_pixels = [(r - min_r, c - min_c) for r, c in template.pixels]

    rows = [r for r, c in template_pixels]
    cols = [c for r, c in template_pixels]
    template_height = max(rows) + 1 if rows else 1
    template_width = max(cols) + 1 if cols else 1

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

    output_grid = np.zeros(output_shape, dtype=np.int64)

    for i, color_obj in enumerate(color_sources):
        if horizontal:
            base_row, base_col = 0, i * template_width
        else:
            base_row, base_col = i * template_height, 0

        for r, c in template_pixels:
            out_r, out_c = base_row + r, base_col + c
            if 0 <= out_r < output_shape[0] and 0 <= out_c < output_shape[1]:
                output_grid[out_r, out_c] = color_obj.color

    return output_grid


def apply_sequential_placement(
    input_grid: np.ndarray,
    rule: MultiHeadRule,
    verbose: bool = False
) -> np.ndarray:
    """Apply sequential placement rule using anchor relationships.

    Places objects sequentially, with each object's position determined
    by an anchor relationship to the previous object.
    """
    output_shape = rule.output_shape if rule.output_shape else input_grid.shape

    if rule.anchor_relation is None:
        return np.zeros(output_shape, dtype=np.int64)

    ordering = rule.correspondence.ordering

    if verbose:
        rel = rule.anchor_relation
        print(f"  Sequential placement: {ordering.name if ordering else 'default'} ordering, "
              f"{rel.source_anchor.value.upper()}->{rel.target_anchor.value.upper()} offset={rel.offset}")

    return generate_sequential_placement_output(
        input_grid, ordering, rule.anchor_relation, output_shape
    )


def apply_grid_packing(
    input_grid: np.ndarray,
    rule: MultiHeadRule,
    verbose: bool = False
) -> np.ndarray:
    """Apply grid packing rule (1990f7a8 style).

    Packs objects into a compact grid with separators using adaptive reading order.
    """
    output_shape = rule.output_shape if rule.output_shape else input_grid.shape

    # Extract grid parameters from output shape
    # Output shape was computed as: grid_rows * cell_h + (grid_rows - 1) * sep
    # We need to reverse-engineer the parameters

    # Extract objects to determine cell dimensions
    objects = extract_objects_from_grid(input_grid, segmentation_mode=SegmentationMode.CONNECTIVITY)
    objects = [o for o in objects if not o.is_background and o.color > 0]

    if not objects:
        return np.zeros(output_shape, dtype=np.int64)

    # Use adaptive reading order
    adaptive_order = AdaptiveReadingOrder()
    ordered_objects = adaptive_order.order(objects)
    row_structure = adaptive_order.get_row_structure(objects)

    # Determine grid dimensions
    grid_rows = len(row_structure)
    grid_cols = max(len(row) for row in row_structure) if row_structure else 1

    # Compute cell dimensions
    cell_height = max(o.height for o in objects)
    cell_width = max(o.width for o in objects)

    # Compute separator size from output shape
    # output_height = grid_rows * cell_height + (grid_rows - 1) * sep_size
    # sep_size = (output_height - grid_rows * cell_height) / (grid_rows - 1)
    if grid_rows > 1:
        sep_size_h = (output_shape[0] - grid_rows * cell_height) // (grid_rows - 1)
    else:
        sep_size_h = 0

    if grid_cols > 1:
        sep_size_w = (output_shape[1] - grid_cols * cell_width) // (grid_cols - 1)
    else:
        sep_size_w = 0

    # Use the smaller of the two (they should be equal for proper grid)
    sep_size = min(sep_size_h, sep_size_w) if grid_rows > 1 or grid_cols > 1 else 0

    if verbose:
        print(f"  Grid packing: {grid_rows}x{grid_cols} grid, cell={cell_height}x{cell_width}, sep={sep_size}")

    return generate_grid_packing_output(
        input_grid, grid_rows, grid_cols, cell_height, cell_width, sep_size
    )


# =============================================================================
# CLI
# =============================================================================

def main():
    import argparse
    from puzzle_loader import load_puzzle

    parser = argparse.ArgumentParser(description='Multi-Head Correspondence Module')
    parser.add_argument('--puzzle-id', type=str, required=True, help='Puzzle ID')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')

    args = parser.parse_args()

    puzzle = load_puzzle(args.puzzle_id)

    print(f"Analyzing puzzle {args.puzzle_id} with multi-head correspondence")
    print("=" * 60)

    rule = discover_multihead_rule(puzzle, verbose=args.verbose)

    if rule is None:
        print("\nNo multi-head rule discovered")
        return

    print(f"\n{'='*60}")
    print(f"Discovered Rule:")
    print(rule.describe())
    print(f"{'='*60}")

    print("\nTesting on training examples:")

    total_correct = 0
    total_examples = len(puzzle['train'])

    for i, ex in enumerate(puzzle['train']):
        input_grid = np.array(ex['input'])
        expected_output = np.array(ex['output'])

        predicted_output = apply_multihead_rule(input_grid, rule, verbose=args.verbose)

        if predicted_output.shape != expected_output.shape:
            print(f"  Example {i+1}: Shape mismatch - expected {expected_output.shape}, got {predicted_output.shape}")
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

    if puzzle.get('test'):
        print(f"\n{'='*60}")
        print("Testing on test examples:")
        print("=" * 60)

        for i, ex in enumerate(puzzle['test']):
            input_grid = np.array(ex['input'])

            if 'output' in ex:
                expected = np.array(ex['output'])
            else:
                expected = None

            predicted = apply_multihead_rule(input_grid, rule, verbose=args.verbose)

            if expected is not None:
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
