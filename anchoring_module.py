#!/usr/bin/env python3
"""
Anchor Point Relationship Discovery

Tests the hypothesis that spatial relationships between objects in ARC puzzles
can be discovered by finding which anchor point pairs have the most consistent
offset across examples.

For two objects A and B, there are 9 anchor points on each, giving 81 possible
anchor-to-anchor relationships. The correct relationship should have near-zero
variance in the offset across all training examples.

Example relationships this can discover:
- "B's bottom-right corner touches A's bottom-left corner" (offset = 0,0)
- "B is placed 1 row below A, left-aligned" (TL-to-BL with offset 1,0)
- "B is centered horizontally relative to A" (TC-to-BC with offset)
"""

import numpy as np
import json
import os
import argparse
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
from enum import Enum
from itertools import product
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap

# Object detection and extraction
from object_module import (
    Object as OrderingObject,
    extract_objects_from_grid,
)

# Import shared utilities from ordering_module
from ordering_module import (
    load_puzzles,
    find_object_correspondences_simple,
    Correspondence,
    ALL_ORDERINGS,
    AdaptiveReadingOrder,
)

# Object correspondence matching
from correspondence_module import (
    find_correspondences_by_pattern as _find_correspondences_by_pattern,
    extract_pattern_from_bbox,
)


# =============================================================================
# Anchor Point Definitions
# =============================================================================

class AnchorPoint(Enum):
    """
    Nine anchor points on a rectangular object.
    
    Visual layout:
        TL --- TC --- TR
        |             |
        ML --- C  --- MR
        |             |
        BL --- BC --- BR
    """
    TOP_LEFT = 'tl'
    TOP_CENTER = 'tc'
    TOP_RIGHT = 'tr'
    MIDDLE_LEFT = 'ml'
    CENTER = 'c'
    MIDDLE_RIGHT = 'mr'
    BOTTOM_LEFT = 'bl'
    BOTTOM_CENTER = 'bc'
    BOTTOM_RIGHT = 'br'


# All anchor points for iteration
ALL_ANCHORS = list(AnchorPoint)


def get_anchor_offset(anchor: AnchorPoint, height: int, width: int) -> Tuple[int, int]:
    """
    Get the (row, col) offset from top-left to reach this anchor point.
    
    For an object of size (height, width), returns how many rows/cols
    from the top-left corner to reach the specified anchor.
    """
    # Row offset: 0 for top, h//2 for middle, h-1 for bottom
    # Col offset: 0 for left, w//2 for center, w-1 for right
    
    row_offsets = {
        'top': 0,
        'middle': (height - 1) // 2,
        'bottom': height - 1,
    }
    
    col_offsets = {
        'left': 0,
        'center': (width - 1) // 2,
        'right': width - 1,
    }
    
    anchor_map = {
        AnchorPoint.TOP_LEFT:      ('top', 'left'),
        AnchorPoint.TOP_CENTER:    ('top', 'center'),
        AnchorPoint.TOP_RIGHT:     ('top', 'right'),
        AnchorPoint.MIDDLE_LEFT:   ('middle', 'left'),
        AnchorPoint.CENTER:        ('middle', 'center'),
        AnchorPoint.MIDDLE_RIGHT:  ('middle', 'right'),
        AnchorPoint.BOTTOM_LEFT:   ('bottom', 'left'),
        AnchorPoint.BOTTOM_CENTER: ('bottom', 'center'),
        AnchorPoint.BOTTOM_RIGHT:  ('bottom', 'right'),
    }
    
    row_key, col_key = anchor_map[anchor]
    return (row_offsets[row_key], col_offsets[col_key])


def get_anchor_position(obj_top_left: Tuple[int, int],
                        obj_size: Tuple[int, int],
                        anchor: AnchorPoint) -> Tuple[int, int]:
    """
    Get the absolute pixel position of an anchor point on an object.
    
    Args:
        obj_top_left: (row, col) of object's top-left corner
        obj_size: (height, width) of the object
        anchor: which anchor point to compute
        
    Returns:
        (row, col) absolute position of the anchor point
    """
    row, col = obj_top_left
    height, width = obj_size
    dr, dc = get_anchor_offset(anchor, height, width)
    return (row + dr, col + dc)


# =============================================================================
# Spatial Relation Representation
# =============================================================================

@dataclass
class SpatialRelation:
    """
    Describes how one object is positioned relative to another.
    
    The relation means: source_anchor aligns with target_anchor + offset
    
    Examples:
        - B's BR corner at A's BL corner:
          source_anchor=BR, target_anchor=BL, offset=(0, 0)
        
        - B is 1 row below A, left edges aligned:
          source_anchor=TL, target_anchor=BL, offset=(1, 0)
        
        - B is directly right of A with 2-cell gap:
          source_anchor=ML, target_anchor=MR, offset=(0, 2)
    """
    source_anchor: AnchorPoint
    target_anchor: AnchorPoint
    offset: Tuple[int, int]  # (row_offset, col_offset)
    
    def __repr__(self):
        return (f"SpatialRelation({self.source_anchor.value} -> "
                f"{self.target_anchor.value}, offset={self.offset})")
    
    def describe(self) -> str:
        """Human-readable description of this relation."""
        src = self.source_anchor.value.upper()
        tgt = self.target_anchor.value.upper()
        dr, dc = self.offset
        
        if dr == 0 and dc == 0:
            return f"source.{src} coincides with target.{tgt}"
        else:
            return f"source.{src} is at target.{tgt} + ({dr}, {dc})"


@dataclass 
class DiscoveredRelation:
    """Result of relation discovery, including confidence metrics."""
    relation: SpatialRelation
    variance: float  # Lower is better - more consistent across examples
    mean_offset: Tuple[float, float]  # The discovered mean offset
    offsets: List[Tuple[int, int]]  # Individual offsets per example
    
    def __repr__(self):
        return (f"DiscoveredRelation({self.relation}, "
                f"var={self.variance:.4f})")


# =============================================================================
# Object Representation for Testing
# =============================================================================

@dataclass
class TestObject:
    """An object with position and size."""
    top_left: Tuple[int, int]  # (row, col)
    size: Tuple[int, int]      # (height, width)
    object_id: int = 0
    
    def get_anchor(self, anchor: AnchorPoint) -> Tuple[int, int]:
        """Get absolute position of an anchor point."""
        return get_anchor_position(self.top_left, self.size, anchor)


@dataclass
class ExamplePair:
    """A pair of objects in one example (input/output state)."""
    source: TestObject  # The object being positioned
    target: TestObject  # The reference object


# =============================================================================
# Relation Discovery Algorithm
# =============================================================================

def compute_offset_for_anchor_pair(
    examples: List[ExamplePair],
    source_anchor: AnchorPoint,
    target_anchor: AnchorPoint
) -> Tuple[List[Tuple[int, int]], float, Tuple[float, float]]:
    """
    For a given anchor pair, compute the offset in each example.
    
    Returns:
        offsets: List of (dr, dc) offsets for each example
        variance: Sum of row variance + col variance (lower = more consistent)
        mean_offset: Mean (dr, dc) across examples
    """
    offsets = []
    
    for ex in examples:
        # Get anchor positions
        src_pos = ex.source.get_anchor(source_anchor)
        tgt_pos = ex.target.get_anchor(target_anchor)
        
        # Offset = source_anchor_pos - target_anchor_pos
        dr = src_pos[0] - tgt_pos[0]
        dc = src_pos[1] - tgt_pos[1]
        offsets.append((dr, dc))
    
    # Compute variance
    offsets_array = np.array(offsets)
    row_var = np.var(offsets_array[:, 0])
    col_var = np.var(offsets_array[:, 1])
    total_var = row_var + col_var
    
    # Mean offset
    mean_dr = np.mean(offsets_array[:, 0])
    mean_dc = np.mean(offsets_array[:, 1])
    
    return offsets, total_var, (mean_dr, mean_dc)


def discover_relation(
    examples: List[ExamplePair],
    top_k: int = 5
) -> List[DiscoveredRelation]:
    """
    Discover the best spatial relation between source and target objects.
    
    Tries all 81 anchor point combinations and returns the ones with
    lowest variance (most consistent offset across examples).
    
    Args:
        examples: List of ExamplePair, each containing source and target objects
        top_k: Number of best relations to return
        
    Returns:
        List of DiscoveredRelation, sorted by variance (best first)
    """
    results = []
    
    # Try all 81 combinations
    for src_anchor, tgt_anchor in product(ALL_ANCHORS, ALL_ANCHORS):
        offsets, variance, mean_offset = compute_offset_for_anchor_pair(
            examples, src_anchor, tgt_anchor
        )
        
        # Round mean offset to nearest integer for the relation
        int_offset = (round(mean_offset[0]), round(mean_offset[1]))
        
        relation = SpatialRelation(
            source_anchor=src_anchor,
            target_anchor=tgt_anchor,
            offset=int_offset
        )
        
        discovered = DiscoveredRelation(
            relation=relation,
            variance=variance,
            mean_offset=mean_offset,
            offsets=offsets
        )
        results.append(discovered)
    
    # Sort by variance (lowest first)
    results.sort(key=lambda x: x.variance)
    
    return results[:top_k]


def discover_relation_to_grid(
    examples: List[Tuple[TestObject, Tuple[int, int]]],  # (object, grid_size)
    top_k: int = 5
) -> List[DiscoveredRelation]:
    """
    Discover spatial relation between an object and the grid.
    
    Args:
        examples: List of (object, grid_size) tuples
        top_k: Number of best relations to return
        
    Returns:
        List of DiscoveredRelation, sorted by variance (best first)
    """
    results = []
    
    for src_anchor, grid_anchor in product(ALL_ANCHORS, ALL_ANCHORS):
        offsets = []
        
        for obj, (grid_h, grid_w) in examples:
            # Get source anchor position
            src_pos = obj.get_anchor(src_anchor)
            
            # Get grid anchor position (treat grid as an object at 0,0)
            grid_anchor_offset = get_anchor_offset(grid_anchor, grid_h, grid_w)
            
            # Offset from grid anchor to object anchor
            dr = src_pos[0] - grid_anchor_offset[0]
            dc = src_pos[1] - grid_anchor_offset[1]
            offsets.append((dr, dc))
        
        # Compute variance
        offsets_array = np.array(offsets)
        row_var = np.var(offsets_array[:, 0])
        col_var = np.var(offsets_array[:, 1])
        total_var = row_var + col_var
        
        mean_dr = np.mean(offsets_array[:, 0])
        mean_dc = np.mean(offsets_array[:, 1])
        
        int_offset = (round(mean_dr), round(mean_dc))
        
        relation = SpatialRelation(
            source_anchor=src_anchor,
            target_anchor=grid_anchor,
            offset=int_offset
        )
        
        discovered = DiscoveredRelation(
            relation=relation,
            variance=total_var,
            mean_offset=(mean_dr, mean_dc),
            offsets=offsets
        )
        results.append(discovered)
    
    results.sort(key=lambda x: x.variance)
    return results[:top_k]


# =============================================================================
# Position Computation from Relations
# =============================================================================

def compute_position_from_relation(
    relation: SpatialRelation,
    source_size: Tuple[int, int],
    target_pos: Tuple[int, int],
    target_size: Tuple[int, int]
) -> Tuple[int, int]:
    """
    Compute where source object's top-left should be, given a relation.
    
    The relation says: source.anchor = target.anchor + offset
    We need to find source.top_left.
    
    Args:
        relation: The spatial relation to apply
        source_size: (height, width) of source object
        target_pos: (row, col) top-left of target object
        target_size: (height, width) of target object
        
    Returns:
        (row, col) top-left position for source object
    """
    # Get target anchor position
    target_anchor_pos = get_anchor_position(
        target_pos, target_size, relation.target_anchor
    )
    
    # Source anchor should be at: target_anchor + offset
    source_anchor_row = target_anchor_pos[0] + relation.offset[0]
    source_anchor_col = target_anchor_pos[1] + relation.offset[1]
    
    # Back-calculate source top-left from source anchor position
    src_h, src_w = source_size
    anchor_dr, anchor_dc = get_anchor_offset(relation.source_anchor, src_h, src_w)
    
    source_top_left = (
        source_anchor_row - anchor_dr,
        source_anchor_col - anchor_dc
    )
    
    return source_top_left


def compute_position_from_grid_relation(
    relation: SpatialRelation,
    source_size: Tuple[int, int],
    grid_size: Tuple[int, int]
) -> Tuple[int, int]:
    """
    Compute where source object's top-left should be, relative to grid.
    
    Args:
        relation: The spatial relation (target_anchor refers to grid)
        source_size: (height, width) of source object
        grid_size: (height, width) of the grid
        
    Returns:
        (row, col) top-left position for source object
    """
    grid_h, grid_w = grid_size
    
    # Grid anchor position
    grid_anchor_pos = get_anchor_offset(relation.target_anchor, grid_h, grid_w)
    
    # Source anchor should be at: grid_anchor + offset
    source_anchor_row = grid_anchor_pos[0] + relation.offset[0]
    source_anchor_col = grid_anchor_pos[1] + relation.offset[1]
    
    # Back-calculate source top-left
    src_h, src_w = source_size
    anchor_dr, anchor_dc = get_anchor_offset(relation.source_anchor, src_h, src_w)
    
    source_top_left = (
        source_anchor_row - anchor_dr,
        source_anchor_col - anchor_dc
    )
    
    return source_top_left


# =============================================================================
# Synthetic Test Cases
# =============================================================================

def create_corner_touch_examples(num_examples: int = 5) -> List[ExamplePair]:
    """
    Create examples where B's bottom-right touches A's bottom-left.
    
    Ground truth relation: source.BR -> target.BL, offset=(0, 0)
    
    The objects have varying sizes and positions, but the relationship
    is always the same.
    """
    examples = []
    
    for _ in range(num_examples):
        # Random target (A) position and size
        target_h = np.random.randint(2, 5)
        target_w = np.random.randint(2, 5)
        target_row = np.random.randint(5, 15)
        target_col = np.random.randint(5, 15)
        
        target = TestObject(
            top_left=(target_row, target_col),
            size=(target_h, target_w),
            object_id=0
        )
        
        # Source (B): its BR should touch target's BL
        source_h = np.random.randint(2, 5)
        source_w = np.random.randint(2, 5)
        
        # Target's BL is at (target_row + target_h - 1, target_col)
        # Source's BR should be there
        # Source's BR is at (source_row + source_h - 1, source_col + source_w - 1)
        # So: source_row + source_h - 1 = target_row + target_h - 1
        #     source_col + source_w - 1 = target_col
        source_row = target_row + target_h - source_h
        source_col = target_col - source_w + 1
        
        source = TestObject(
            top_left=(source_row, source_col),
            size=(source_h, source_w),
            object_id=1
        )
        
        examples.append(ExamplePair(source=source, target=target))
    
    return examples


def create_stacked_below_examples(
    num_examples: int = 5,
    gap: int = 1
) -> List[ExamplePair]:
    """
    Create examples where B is stacked below A with a gap.
    
    Ground truth relation: source.TL -> target.BL, offset=(gap, 0)
    (B's top-left is 'gap' rows below A's bottom-left)
    """
    examples = []
    
    for _ in range(num_examples):
        # Random target (A)
        target_h = np.random.randint(2, 5)
        target_w = np.random.randint(3, 7)
        target_row = np.random.randint(2, 10)
        target_col = np.random.randint(2, 15)
        
        target = TestObject(
            top_left=(target_row, target_col),
            size=(target_h, target_w),
            object_id=0
        )
        
        # Source (B): placed below with left edges aligned
        source_h = np.random.randint(2, 5)
        source_w = np.random.randint(3, 7)  # Width can differ
        
        # B's TL should be at A's BL + (gap, 0)
        source_row = target_row + target_h - 1 + gap
        source_col = target_col
        
        source = TestObject(
            top_left=(source_row, source_col),
            size=(source_h, source_w),
            object_id=1
        )
        
        examples.append(ExamplePair(source=source, target=target))
    
    return examples


def create_centered_below_examples(
    num_examples: int = 5,
    gap: int = 1
) -> List[ExamplePair]:
    """
    Create examples where B is centered below A.
    
    Ground truth relation: source.TC -> target.BC, offset=(gap, 0)
    """
    examples = []
    
    for _ in range(num_examples):
        # Random target (A)
        target_h = np.random.randint(2, 5)
        target_w = np.random.randint(4, 8)
        target_row = np.random.randint(2, 10)
        target_col = np.random.randint(2, 15)
        
        target = TestObject(
            top_left=(target_row, target_col),
            size=(target_h, target_w),
            object_id=0
        )
        
        # Source (B): centered horizontally below A
        source_h = np.random.randint(2, 5)
        source_w = np.random.randint(2, 6)
        
        # Target's BC is at (target_row + target_h - 1, target_col + target_w//2)
        # Source's TC should be at BC + (gap, 0)
        # Source's TC is at (source_row, source_col + source_w//2)
        target_bc_row = target_row + target_h - 1
        target_bc_col = target_col + (target_w - 1) // 2
        
        source_row = target_bc_row + gap
        source_col = target_bc_col - (source_w - 1) // 2
        
        source = TestObject(
            top_left=(source_row, source_col),
            size=(source_h, source_w),
            object_id=1
        )
        
        examples.append(ExamplePair(source=source, target=target))
    
    return examples


def create_right_adjacent_examples(
    num_examples: int = 5,
    gap: int = 1
) -> List[ExamplePair]:
    """
    Create examples where B is to the right of A, vertically centered.
    
    Ground truth relation: source.ML -> target.MR, offset=(0, gap)
    """
    examples = []
    
    for _ in range(num_examples):
        # Random target (A)
        target_h = np.random.randint(3, 7)
        target_w = np.random.randint(2, 5)
        target_row = np.random.randint(2, 12)
        target_col = np.random.randint(2, 10)
        
        target = TestObject(
            top_left=(target_row, target_col),
            size=(target_h, target_w),
            object_id=0
        )
        
        # Source (B): to the right, vertically centered
        source_h = np.random.randint(2, 6)
        source_w = np.random.randint(2, 5)
        
        # Target's MR is at (target_row + target_h//2, target_col + target_w - 1)
        # Source's ML should be at MR + (0, gap)
        target_mr_row = target_row + (target_h - 1) // 2
        target_mr_col = target_col + target_w - 1
        
        source_row = target_mr_row - (source_h - 1) // 2
        source_col = target_mr_col + gap
        
        source = TestObject(
            top_left=(source_row, source_col),
            size=(source_h, source_w),
            object_id=1
        )
        
        examples.append(ExamplePair(source=source, target=target))
    
    return examples


def create_grid_corner_examples(
    num_examples: int = 5,
    corner: str = 'bottom_right'
) -> List[Tuple[TestObject, Tuple[int, int]]]:
    """
    Create examples where object is placed at a grid corner.
    
    Ground truth relations:
    - bottom_right: source.BR -> grid.BR, offset=(0, 0)
    - top_left: source.TL -> grid.TL, offset=(0, 0)
    etc.
    """
    examples = []
    
    corner_to_anchor = {
        'top_left': AnchorPoint.TOP_LEFT,
        'top_right': AnchorPoint.TOP_RIGHT,
        'bottom_left': AnchorPoint.BOTTOM_LEFT,
        'bottom_right': AnchorPoint.BOTTOM_RIGHT,
    }
    anchor = corner_to_anchor[corner]
    
    for _ in range(num_examples):
        # Random grid size
        grid_h = np.random.randint(15, 25)
        grid_w = np.random.randint(15, 25)
        
        # Random object size
        obj_h = np.random.randint(2, 5)
        obj_w = np.random.randint(2, 5)
        
        # Place object so its corner matches grid corner
        grid_anchor_pos = get_anchor_offset(anchor, grid_h, grid_w)
        obj_anchor_offset = get_anchor_offset(anchor, obj_h, obj_w)
        
        obj_row = grid_anchor_pos[0] - obj_anchor_offset[0]
        obj_col = grid_anchor_pos[1] - obj_anchor_offset[1]
        
        obj = TestObject(
            top_left=(obj_row, obj_col),
            size=(obj_h, obj_w),
            object_id=0
        )
        
        examples.append((obj, (grid_h, grid_w)))
    
    return examples


# =============================================================================
# Main Test Runner
# =============================================================================

def run_discovery_tests():
    """Run all discovery tests and report results."""
    print("=" * 70)
    print("ANCHOR POINT RELATIONSHIP DISCOVERY TESTS")
    print("=" * 70)
    
    test_cases = [
        (
            "Corner Touch (B.BR -> A.BL)",
            create_corner_touch_examples(10),
            SpatialRelation(AnchorPoint.BOTTOM_RIGHT, AnchorPoint.BOTTOM_LEFT, (0, 0))
        ),
        (
            "Stacked Below with gap=1 (B.TL -> A.BL + (1,0))",
            create_stacked_below_examples(10, gap=1),
            SpatialRelation(AnchorPoint.TOP_LEFT, AnchorPoint.BOTTOM_LEFT, (1, 0))
        ),
        (
            "Stacked Below with gap=2 (B.TL -> A.BL + (2,0))",
            create_stacked_below_examples(10, gap=2),
            SpatialRelation(AnchorPoint.TOP_LEFT, AnchorPoint.BOTTOM_LEFT, (2, 0))
        ),
        (
            "Centered Below (B.TC -> A.BC + (1,0))",
            create_centered_below_examples(10, gap=1),
            SpatialRelation(AnchorPoint.TOP_CENTER, AnchorPoint.BOTTOM_CENTER, (1, 0))
        ),
        (
            "Right Adjacent (B.ML -> A.MR + (0,1))",
            create_right_adjacent_examples(10, gap=1),
            SpatialRelation(AnchorPoint.MIDDLE_LEFT, AnchorPoint.MIDDLE_RIGHT, (0, 1))
        ),
    ]
    
    all_passed = True
    
    for name, examples, expected in test_cases:
        print(f"\nTest: {name}")
        print("-" * 50)
        
        # Discover relations
        discovered = discover_relation(examples, top_k=5)
        
        best = discovered[0]
        print(f"Expected: {expected}")
        print(f"Best found: {best.relation}")
        print(f"  Variance: {best.variance:.6f}")
        print(f"  Mean offset: ({best.mean_offset[0]:.2f}, {best.mean_offset[1]:.2f})")
        
        # Check if best matches expected
        match = (
            best.relation.source_anchor == expected.source_anchor and
            best.relation.target_anchor == expected.target_anchor and
            best.relation.offset == expected.offset
        )
        
        if match:
            print("  ✓ CORRECT")
        else:
            print("  ✗ WRONG")
            all_passed = False
            
            # Show top 5 for debugging
            print("\n  Top 5 candidates:")
            for i, d in enumerate(discovered):
                print(f"    {i+1}. {d.relation} (var={d.variance:.6f})")
    
    # Grid-relative tests
    print("\n" + "=" * 70)
    print("GRID-RELATIVE DISCOVERY TESTS")
    print("=" * 70)
    
    grid_tests = [
        ("Bottom-Right Corner", "bottom_right", 
         SpatialRelation(AnchorPoint.BOTTOM_RIGHT, AnchorPoint.BOTTOM_RIGHT, (0, 0))),
        ("Top-Left Corner", "top_left",
         SpatialRelation(AnchorPoint.TOP_LEFT, AnchorPoint.TOP_LEFT, (0, 0))),
    ]
    
    for name, corner, expected in grid_tests:
        print(f"\nTest: {name}")
        print("-" * 50)
        
        examples = create_grid_corner_examples(10, corner)
        discovered = discover_relation_to_grid(examples, top_k=5)
        
        best = discovered[0]
        print(f"Expected: {expected}")
        print(f"Best found: {best.relation}")
        print(f"  Variance: {best.variance:.6f}")
        
        match = (
            best.relation.source_anchor == expected.source_anchor and
            best.relation.target_anchor == expected.target_anchor and
            best.relation.offset == expected.offset
        )
        
        if match:
            print("  ✓ CORRECT")
        else:
            print("  ✗ WRONG")
            all_passed = False
    
    # Position reconstruction test
    print("\n" + "=" * 70)
    print("POSITION RECONSTRUCTION TEST")
    print("=" * 70)
    
    print("\nUsing discovered relation to predict positions...")
    
    # Use corner touch examples
    examples = create_corner_touch_examples(5)
    discovered = discover_relation(examples, top_k=1)
    relation = discovered[0].relation
    
    print(f"Discovered relation: {relation}")
    
    errors = []
    for i, ex in enumerate(examples):
        # Predict source position using relation
        predicted_pos = compute_position_from_relation(
            relation,
            ex.source.size,
            ex.target.top_left,
            ex.target.size
        )
        
        actual_pos = ex.source.top_left
        error = abs(predicted_pos[0] - actual_pos[0]) + abs(predicted_pos[1] - actual_pos[1])
        errors.append(error)
        
        status = "✓" if error == 0 else "✗"
        print(f"  Example {i+1}: predicted={predicted_pos}, actual={actual_pos} {status}")
    
    print(f"\nMean reconstruction error: {np.mean(errors):.2f}")
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    if all_passed:
        print("✓ All discovery tests passed!")
        print("  The algorithm correctly identifies anchor point relationships")
        print("  by finding the combination with lowest variance across examples.")
    else:
        print("✗ Some tests failed - see above for details.")
    
    return all_passed


def demonstrate_ambiguity():
    """
    Show cases where multiple relations could explain the data.
    
    This happens when objects have the same width/height, making
    multiple anchor pairs equivalent.
    """
    print("\n" + "=" * 70)
    print("AMBIGUITY DEMONSTRATION")
    print("=" * 70)
    print("\nWhen objects have equal width, horizontal anchors become ambiguous.")
    print("For example, if all objects are 3 wide, then TL and TC and TR")
    print("differ by consistent offsets, so multiple relations have zero variance.")
    
    # Create examples with same-width objects
    examples = []
    fixed_width = 3
    
    for _ in range(5):
        target = TestObject(
            top_left=(np.random.randint(5, 10), np.random.randint(5, 10)),
            size=(np.random.randint(2, 5), fixed_width),  # Fixed width
            object_id=0
        )
        
        # Stack B below A, left-aligned
        source = TestObject(
            top_left=(target.top_left[0] + target.size[0], target.top_left[1]),
            size=(np.random.randint(2, 5), fixed_width),  # Same fixed width
            object_id=1
        )
        
        examples.append(ExamplePair(source=source, target=target))
    
    discovered = discover_relation(examples, top_k=10)
    
    print("\nTop 10 relations (all with zero variance expected for horizontal anchors):")
    for i, d in enumerate(discovered):
        print(f"  {i+1}. {d.relation} (var={d.variance:.6f})")
    
    print("\nNote: Multiple relations with var=0 are all equally valid!")
    print("This is expected and correct - they all describe the same physical relationship.")


# =============================================================================
# Real Puzzle Analysis
# =============================================================================

def ordering_object_to_test_object(obj: OrderingObject) -> TestObject:
    """Convert an ordering_module Object to a TestObject for anchor analysis."""
    return TestObject(
        top_left=(obj.row, obj.col),
        size=(obj.height, obj.width),
        object_id=obj.id
    )


def analyze_puzzle_anchors(
    puzzle_id: str,
    ordering_strategy: str = "adaptive_reading_order",
    top_k: int = 5,
    verbose: bool = True
) -> Dict:
    """
    Analyze anchor point relationships for a specific ARC puzzle.

    This function:
    1. Loads the puzzle
    2. Extracts objects from input/output grids
    3. Finds correspondences between input and output objects
    4. For each object pair (in ordering sequence), discovers the best anchor relationships

    Args:
        puzzle_id: The ARC puzzle ID (e.g., "1990f7a8")
        ordering_strategy: Which ordering to use for sequencing objects
        top_k: Number of top anchor relations to show per object pair
        verbose: Print detailed output

    Returns:
        Dict with analysis results
    """
    # Load puzzle
    puzzles = load_puzzles("arc-agi-1")
    if puzzle_id not in puzzles:
        puzzles.update(load_puzzles("arc-agi-2"))

    if puzzle_id not in puzzles:
        print(f"Error: Puzzle '{puzzle_id}' not found")
        return {}

    puzzle = puzzles[puzzle_id]

    if verbose:
        print("=" * 70)
        print(f"ANCHOR POINT ANALYSIS: Puzzle {puzzle_id}")
        print("=" * 70)
        print(f"\nOrdering strategy: {ordering_strategy}")
        print(f"Training examples: {len(puzzle['train'])}")

    # Get ordering strategy
    ordering = None
    for strat in ALL_ORDERINGS:
        if strat.name == ordering_strategy:
            ordering = strat
            break

    if ordering is None:
        print(f"Warning: Unknown ordering '{ordering_strategy}', using adaptive_reading_order")
        ordering = AdaptiveReadingOrder()

    # Collect data across all training examples
    # Structure: per_position_examples[position_index] = list of ExamplePair
    # where position_index is the object's position in the ordering sequence

    all_examples_data = []  # List of per-example data

    for ex_idx, pair in enumerate(puzzle['train']):
        if 'output' not in pair:
            continue

        input_grid = np.array(pair['input'])
        output_grid = np.array(pair['output'])

        # Extract objects
        input_objects = extract_objects_from_grid(input_grid)
        output_objects = extract_objects_from_grid(output_grid)

        if not input_objects or not output_objects:
            continue

        # Find correspondences
        correspondences = find_object_correspondences_simple(input_objects, output_objects)

        if not correspondences:
            continue

        # Apply ordering to input objects
        ordered_inputs = ordering.order(input_objects)
        input_id_to_order = {obj.id: idx for idx, obj in enumerate(ordered_inputs)}

        # Sort correspondences by input ordering
        ordered_corrs = sorted(
            correspondences,
            key=lambda c: input_id_to_order.get(c.input_obj.id, 999)
        )

        # Convert to TestObjects (using OUTPUT positions - that's what we're predicting)
        output_test_objects = []
        for corr in ordered_corrs:
            out_obj = corr.output_obj
            test_obj = TestObject(
                top_left=(out_obj.row, out_obj.col),
                size=(out_obj.height, out_obj.width),
                object_id=out_obj.id
            )
            output_test_objects.append(test_obj)

        # Also store input positions for delta analysis
        input_test_objects = []
        for corr in ordered_corrs:
            in_obj = corr.input_obj
            test_obj = TestObject(
                top_left=(in_obj.row, in_obj.col),
                size=(in_obj.height, in_obj.width),
                object_id=in_obj.id
            )
            input_test_objects.append(test_obj)

        grid_shape = (output_grid.shape[0], output_grid.shape[1])

        all_examples_data.append({
            'example_idx': ex_idx,
            'output_objects': output_test_objects,
            'input_objects': input_test_objects,
            'grid_shape': grid_shape,
            'num_objects': len(output_test_objects),
        })

    if not all_examples_data:
        print("Error: No valid training examples found")
        return {}

    # Check object count consistency
    object_counts = [d['num_objects'] for d in all_examples_data]
    if len(set(object_counts)) > 1:
        if verbose:
            print(f"\nWarning: Inconsistent object counts across examples: {object_counts}")

    num_objects = min(object_counts)

    if verbose:
        print(f"\nAnalyzing {num_objects} objects per example")
        print("-" * 70)

    results = {
        'puzzle_id': puzzle_id,
        'ordering_strategy': ordering_strategy,
        'num_objects': num_objects,
        'num_examples': len(all_examples_data),
        'position_analyses': [],
    }

    # Analyze each position in the sequence
    for pos in range(num_objects):
        if verbose:
            print(f"\n{'='*60}")
            print(f"POSITION {pos} (Object #{pos} in {ordering_strategy} order)")
            print(f"{'='*60}")

        pos_result = {
            'position': pos,
            'grid_relations': None,
            'object_relations': {},
            'input_delta': None,
        }

        # 1. Analyze relation to GRID (absolute positioning)
        if verbose:
            print(f"\n--- Relation to Grid ---")

        grid_examples = []
        for data in all_examples_data:
            if pos < len(data['output_objects']):
                obj = data['output_objects'][pos]
                grid_examples.append((obj, data['grid_shape']))

        if grid_examples:
            grid_relations = discover_relation_to_grid(grid_examples, top_k=top_k)
            pos_result['grid_relations'] = grid_relations

            if verbose:
                print(f"  Top {top_k} grid-relative anchors:")
                for i, rel in enumerate(grid_relations):
                    var_str = f"var={rel.variance:.4f}" if rel.variance > 0 else "var=0 (PERFECT)"
                    print(f"    {i+1}. {rel.relation.source_anchor.value.upper()}->{rel.relation.target_anchor.value.upper()} "
                          f"offset={rel.relation.offset} ({var_str})")

        # 2. Analyze relation to PREVIOUS OBJECTS (object-relative positioning)
        for ref_pos in range(pos):
            if verbose:
                print(f"\n--- Relation to Object at Position {ref_pos} ---")

            object_examples = []
            for data in all_examples_data:
                if pos < len(data['output_objects']) and ref_pos < len(data['output_objects']):
                    source_obj = data['output_objects'][pos]
                    target_obj = data['output_objects'][ref_pos]
                    object_examples.append(ExamplePair(source=source_obj, target=target_obj))

            if object_examples:
                obj_relations = discover_relation(object_examples, top_k=top_k)
                pos_result['object_relations'][ref_pos] = obj_relations

                if verbose:
                    print(f"  Top {top_k} object-relative anchors (pos {pos} -> pos {ref_pos}):")
                    for i, rel in enumerate(obj_relations):
                        var_str = f"var={rel.variance:.4f}" if rel.variance > 0 else "var=0 (PERFECT)"
                        print(f"    {i+1}. {rel.relation.source_anchor.value.upper()}->{rel.relation.target_anchor.value.upper()} "
                              f"offset={rel.relation.offset} ({var_str})")

        # 3. Analyze INPUT-OUTPUT delta (traditional framing)
        if verbose:
            print(f"\n--- Input-to-Output Delta ---")

        deltas = []
        for data in all_examples_data:
            if pos < len(data['output_objects']) and pos < len(data['input_objects']):
                out_obj = data['output_objects'][pos]
                in_obj = data['input_objects'][pos]
                delta_row = out_obj.top_left[0] - in_obj.top_left[0]
                delta_col = out_obj.top_left[1] - in_obj.top_left[1]
                deltas.append((delta_row, delta_col))

        if deltas:
            deltas_array = np.array(deltas)
            delta_var = np.var(deltas_array[:, 0]) + np.var(deltas_array[:, 1])
            delta_mean = (np.mean(deltas_array[:, 0]), np.mean(deltas_array[:, 1]))

            pos_result['input_delta'] = {
                'deltas': deltas,
                'variance': delta_var,
                'mean': delta_mean,
            }

            if verbose:
                var_str = f"var={delta_var:.4f}" if delta_var > 0 else "var=0 (PERFECT)"
                print(f"  Delta (output - input): mean=({delta_mean[0]:.1f}, {delta_mean[1]:.1f}) ({var_str})")
                print(f"  Per-example deltas: {deltas}")

        results['position_analyses'].append(pos_result)

    # Summary
    if verbose:
        print("\n" + "=" * 70)
        print("SUMMARY: Best Framing Candidates")
        print("=" * 70)

        for pos_result in results['position_analyses']:
            pos = pos_result['position']
            print(f"\nPosition {pos}:")

            # Find best option across all framing types
            best_options = []

            # Check grid relations
            if pos_result['grid_relations']:
                best_grid = pos_result['grid_relations'][0]
                best_options.append(('grid', best_grid.variance,
                    f"GRID: {best_grid.relation.source_anchor.value.upper()}->{best_grid.relation.target_anchor.value.upper()} "
                    f"offset={best_grid.relation.offset}"))

            # Check object relations
            for ref_pos, rels in pos_result['object_relations'].items():
                if rels:
                    best_obj = rels[0]
                    best_options.append(('object', best_obj.variance,
                        f"OBJ[{ref_pos}]: {best_obj.relation.source_anchor.value.upper()}->{best_obj.relation.target_anchor.value.upper()} "
                        f"offset={best_obj.relation.offset}"))

            # Check input delta
            if pos_result['input_delta']:
                delta_var = pos_result['input_delta']['variance']
                delta_mean = pos_result['input_delta']['mean']
                best_options.append(('delta', delta_var,
                    f"DELTA: ({delta_mean[0]:.0f}, {delta_mean[1]:.0f})"))

            # Sort by variance
            best_options.sort(key=lambda x: x[1])

            for _, var, desc in best_options[:3]:
                marker = "★" if var == 0 else " "
                print(f"  {marker} {desc} (var={var:.4f})")

    return results


# =============================================================================
# Visualization Functions
# =============================================================================

# ARC color palette (indices 0-9)
ARC_COLORS = [
    '#000000',  # 0: black
    '#0074D9',  # 1: blue
    '#FF4136',  # 2: red
    '#2ECC40',  # 3: green
    '#FFDC00',  # 4: yellow
    '#AAAAAA',  # 5: gray
    '#F012BE',  # 6: magenta
    '#FF851B',  # 7: orange
    '#7FDBFF',  # 8: cyan
    '#870C25',  # 9: brown
]

ARC_CMAP = ListedColormap(ARC_COLORS)


def extract_object_pattern(grid: np.ndarray, obj) -> np.ndarray:
    """Extract the pixel pattern within an object's bounding box.

    This is a thin wrapper around correspondence_module.extract_pattern_from_bbox
    for backwards compatibility.
    """
    return extract_pattern_from_bbox(grid, obj.row, obj.col, obj.height, obj.width)


def find_correspondences_by_pattern(
    input_grid: np.ndarray,
    output_grid: np.ndarray,
    input_objects: list,
    output_objects: list,
    threshold: float = 0.5
) -> list:
    """
    Find correspondences by comparing actual pixel patterns.

    This is a thin wrapper around correspondence_module.find_correspondences_by_pattern
    that returns Correspondence objects for backwards compatibility.

    This is more robust than color+IoU+area when objects move but maintain
    their internal pattern.
    """
    if not input_objects or not output_objects:
        return []

    # Use the shared correspondence function
    matches = _find_correspondences_by_pattern(
        input_grid, output_grid, input_objects, output_objects, threshold
    )

    # Convert (input_idx, output_idx, score) tuples to Correspondence objects
    return [
        Correspondence(input_objects[in_idx], output_objects[out_idx])
        for in_idx, out_idx, _ in matches
    ]


def visualize_anchor_analysis(
    puzzle_id: str,
    ordering_strategy: str = "adaptive_reading_order",
    save_path: Optional[str] = None,
    show: bool = True
) -> None:
    """
    Create a comprehensive visualization for debugging anchor point discovery.

    Shows:
    - Each training example's input and output grids
    - Detected objects with bounding boxes
    - Anchor points on objects
    - Best discovered relationships with connecting lines
    - Variance heatmap for anchor combinations

    Args:
        puzzle_id: The ARC puzzle ID
        ordering_strategy: Which ordering to use
        save_path: If provided, save figure to this path
        show: Whether to display the figure
    """
    # Load puzzle
    puzzles = load_puzzles("arc-agi-1")
    if puzzle_id not in puzzles:
        puzzles.update(load_puzzles("arc-agi-2"))

    if puzzle_id not in puzzles:
        print(f"Error: Puzzle '{puzzle_id}' not found")
        return

    puzzle = puzzles[puzzle_id]
    num_examples = len(puzzle['train'])

    # Get ordering strategy
    ordering = None
    for strat in ALL_ORDERINGS:
        if strat.name == ordering_strategy:
            ordering = strat
            break
    if ordering is None:
        ordering = AdaptiveReadingOrder()

    # Create figure: rows = examples, cols = input, output, variance heatmap
    fig = plt.figure(figsize=(16, 4 * num_examples + 2))

    # Grid spec for layout
    gs = fig.add_gridspec(num_examples + 1, 3, height_ratios=[1] * num_examples + [0.8],
                          hspace=0.3, wspace=0.3)

    # Collect all example data for aggregate analysis
    all_example_pairs = []  # For computing variance heatmap

    for ex_idx, pair in enumerate(puzzle['train']):
        if 'output' not in pair:
            continue

        input_grid = np.array(pair['input'])
        output_grid = np.array(pair['output'])

        # Extract objects
        input_objects = extract_objects_from_grid(input_grid)
        output_objects = extract_objects_from_grid(output_grid)

        # Find correspondences using pattern matching (more robust than color+IoU)
        correspondences = find_correspondences_by_pattern(
            input_grid, output_grid, input_objects, output_objects
        )

        # Order input objects
        ordered_inputs = ordering.order(input_objects) if input_objects else []
        input_id_to_order = {obj.id: idx for idx, obj in enumerate(ordered_inputs)}

        # Sort correspondences by input ordering
        ordered_corrs = sorted(
            correspondences,
            key=lambda c: input_id_to_order.get(c.input_obj.id, 999)
        )

        # Plot input grid
        ax_in = fig.add_subplot(gs[ex_idx, 0])
        _plot_grid_with_objects(ax_in, input_grid, input_objects,
                               f"Example {ex_idx + 1} - Input ({len(input_objects)} objects)")

        # Plot output grid with correspondences
        ax_out = fig.add_subplot(gs[ex_idx, 1])
        _plot_grid_with_objects(ax_out, output_grid, output_objects,
                               f"Example {ex_idx + 1} - Output ({len(output_objects)} objects)")

        # Show order numbers on INPUT grid (based on ordering strategy)
        for i, obj in enumerate(ordered_inputs):
            center_row = obj.row + obj.height / 2
            center_col = obj.col + obj.width / 2
            ax_in.annotate(f'{i}', (center_col, center_row),
                          color='white', fontsize=12, fontweight='bold',
                          ha='center', va='center',
                          bbox=dict(boxstyle='circle', facecolor='blue', alpha=0.7))

        # Debug: print correspondence details
        print(f"\n--- Example {ex_idx + 1} Correspondences ---")
        print(f"Input objects: {len(input_objects)}, Output objects: {len(output_objects)}")
        print(f"Correspondences found: {len(correspondences)}")

        # Print the ordering of input objects
        print(f"\nOrdered inputs (what gets blue numbers):")
        for i, obj in enumerate(ordered_inputs):
            print(f"  Blue '{i}' -> IN(id={obj.id}, pos=({obj.row},{obj.col}), size={obj.height}x{obj.width})")

        # Print what will get red numbers
        print(f"\nOrdered correspondences (what gets red numbers):")
        for i, corr in enumerate(ordered_corrs):
            in_obj = corr.input_obj
            out_obj = corr.output_obj
            order_idx = input_id_to_order.get(in_obj.id, '?')
            print(f"  Red '{i}' -> OUT(id={out_obj.id}, pos=({out_obj.row},{out_obj.col})) "
                  f"[corresponds to IN(id={in_obj.id}, order={order_idx})]")

        # Summary: show expected visual matching
        print(f"\nExpected matching (same number = same shape):")
        for i, corr in enumerate(ordered_corrs):
            in_obj = corr.input_obj
            out_obj = corr.output_obj
            print(f"  Number '{i}': Input@({in_obj.row},{in_obj.col}) <-> Output@({out_obj.row},{out_obj.col})")

        # Draw anchor relationships if we have correspondences
        if len(ordered_corrs) >= 2:
            # Collect example pairs for the first two corresponding objects
            out_objs = [corr.output_obj for corr in ordered_corrs]
            test_objs = [TestObject(
                top_left=(o.row, o.col),
                size=(o.height, o.width),
                object_id=o.id
            ) for o in out_objs]

            if len(test_objs) >= 2:
                all_example_pairs.append(ExamplePair(source=test_objs[1], target=test_objs[0]))

        # Show object order annotation on OUTPUT (based on correspondence to ordered inputs)
        for i, corr in enumerate(ordered_corrs):
            out_obj = corr.output_obj
            center_row = out_obj.row + out_obj.height / 2
            center_col = out_obj.col + out_obj.width / 2
            ax_out.annotate(f'{i}', (center_col, center_row),
                           color='white', fontsize=12, fontweight='bold',
                           ha='center', va='center',
                           bbox=dict(boxstyle='circle', facecolor='red', alpha=0.7))

    # Variance heatmap for object-to-object relationships (if we have pairs)
    ax_heatmap = fig.add_subplot(gs[num_examples, :])

    if all_example_pairs:
        _plot_variance_heatmap(ax_heatmap, all_example_pairs,
                              "Anchor Pair Variance (Object 1 → Object 0)")
    else:
        ax_heatmap.text(0.5, 0.5, "Not enough corresponding objects for heatmap",
                       ha='center', va='center', transform=ax_heatmap.transAxes)
        ax_heatmap.set_title("Anchor Pair Variance")

    # Title
    fig.suptitle(f"Anchor Point Analysis: {puzzle_id}\nOrdering: {ordering_strategy}",
                 fontsize=14, fontweight='bold', y=0.98)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")

    if show:
        plt.show()
    else:
        plt.close()


def _plot_grid_with_objects(ax, grid: np.ndarray, objects: List, title: str):
    """Plot a grid with object bounding boxes and anchor points."""
    # Plot grid
    ax.imshow(grid, cmap=ARC_CMAP, vmin=0, vmax=9)

    # Add grid lines
    h, w = grid.shape
    for i in range(h + 1):
        ax.axhline(i - 0.5, color='gray', linewidth=0.5, alpha=0.5)
    for j in range(w + 1):
        ax.axvline(j - 0.5, color='gray', linewidth=0.5, alpha=0.5)

    # Draw bounding boxes and anchor points for each object
    colors = plt.cm.Set1(np.linspace(0, 1, max(len(objects), 1)))

    for idx, obj in enumerate(objects):
        color = colors[idx % len(colors)]

        # Bounding box
        rect = mpatches.Rectangle(
            (obj.col - 0.5, obj.row - 0.5),
            obj.width, obj.height,
            linewidth=2, edgecolor=color, facecolor='none'
        )
        ax.add_patch(rect)

        # Draw 9 anchor points
        test_obj = TestObject(
            top_left=(obj.row, obj.col),
            size=(obj.height, obj.width),
            object_id=obj.id
        )

        for anchor in ALL_ANCHORS:
            r, c = test_obj.get_anchor(anchor)
            ax.plot(c, r, 'o', color=color, markersize=4, alpha=0.7)

    ax.set_title(title, fontsize=10)
    ax.set_xlim(-0.5, w - 0.5)
    ax.set_ylim(h - 0.5, -0.5)
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])


def _plot_variance_heatmap(ax, examples: List[ExamplePair], title: str):
    """
    Plot a 9x9 heatmap showing variance for each anchor pair combination.

    X-axis: target anchor (Object 0)
    Y-axis: source anchor (Object 1)
    """
    # Compute variance for all 81 combinations
    variance_matrix = np.zeros((9, 9))
    anchor_names = [a.value.upper() for a in ALL_ANCHORS]

    for i, src_anchor in enumerate(ALL_ANCHORS):
        for j, tgt_anchor in enumerate(ALL_ANCHORS):
            _, variance, _ = compute_offset_for_anchor_pair(examples, src_anchor, tgt_anchor)
            variance_matrix[i, j] = variance

    # Find best (minimum variance)
    min_idx = np.unravel_index(np.argmin(variance_matrix), variance_matrix.shape)

    # Plot heatmap
    im = ax.imshow(variance_matrix, cmap='RdYlGn_r', aspect='auto')

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Variance (lower = better)')

    # Highlight minimum
    ax.add_patch(mpatches.Rectangle(
        (min_idx[1] - 0.5, min_idx[0] - 0.5), 1, 1,
        linewidth=3, edgecolor='blue', facecolor='none'
    ))

    # Labels
    ax.set_xticks(range(9))
    ax.set_xticklabels(anchor_names, fontsize=8)
    ax.set_yticks(range(9))
    ax.set_yticklabels(anchor_names, fontsize=8)
    ax.set_xlabel("Target Anchor (Reference Object)", fontsize=9)
    ax.set_ylabel("Source Anchor (Positioned Object)", fontsize=9)

    # Annotate values
    for i in range(9):
        for j in range(9):
            val = variance_matrix[i, j]
            text_color = 'white' if val > np.median(variance_matrix) else 'black'
            ax.text(j, i, f'{val:.1f}', ha='center', va='center',
                   fontsize=6, color=text_color)

    # Title with best result
    best_src = ALL_ANCHORS[min_idx[0]].value.upper()
    best_tgt = ALL_ANCHORS[min_idx[1]].value.upper()
    ax.set_title(f"{title}\nBest: {best_src} → {best_tgt} (var={variance_matrix[min_idx]:.4f})",
                fontsize=10)


def visualize_per_position_analysis(
    puzzle_id: str,
    position: int = 1,
    ordering_strategy: str = "adaptive_reading_order",
    save_path: Optional[str] = None,
    show: bool = True
) -> None:
    """
    Detailed visualization for a specific object position.

    Shows:
    - All training examples side by side
    - The specific object at this position highlighted
    - Variance heatmaps for: grid-relative, and each prior object
    - Per-example offset breakdown

    Args:
        puzzle_id: The ARC puzzle ID
        position: Which object position to analyze (0-indexed)
        ordering_strategy: Which ordering to use
        save_path: If provided, save figure to this path
        show: Whether to display the figure
    """
    # Load puzzle and run analysis
    results = analyze_puzzle_anchors(
        puzzle_id,
        ordering_strategy=ordering_strategy,
        top_k=5,
        verbose=False
    )

    if not results or position >= len(results.get('position_analyses', [])):
        print(f"Error: Position {position} not available")
        return

    pos_result = results['position_analyses'][position]
    num_refs = len(pos_result['object_relations']) + 1  # +1 for grid

    # Create figure
    fig, axes = plt.subplots(1, num_refs + 1, figsize=(4 * (num_refs + 1), 4))
    if num_refs == 0:
        axes = [axes]

    # Grid-relative heatmap
    if pos_result['grid_relations']:
        _plot_discovered_relations_heatmap(
            axes[0],
            pos_result['grid_relations'],
            f"Position {position} → Grid"
        )

    # Object-relative heatmaps
    for ref_idx, (ref_pos, rels) in enumerate(pos_result['object_relations'].items()):
        if rels:
            _plot_discovered_relations_heatmap(
                axes[ref_idx + 1],
                rels,
                f"Position {position} → Object {ref_pos}"
            )

    # Input-delta summary
    if pos_result['input_delta']:
        ax_delta = axes[-1]
        deltas = pos_result['input_delta']['deltas']
        delta_var = pos_result['input_delta']['variance']

        # Plot deltas as scatter
        rows = [d[0] for d in deltas]
        cols = [d[1] for d in deltas]
        ax_delta.scatter(cols, rows, s=100, c=range(len(deltas)), cmap='viridis')

        # Add example labels
        for i, (r, c) in enumerate(deltas):
            ax_delta.annotate(f'Ex{i+1}', (c, r), xytext=(5, 5),
                            textcoords='offset points', fontsize=8)

        ax_delta.axhline(0, color='gray', linestyle='--', alpha=0.5)
        ax_delta.axvline(0, color='gray', linestyle='--', alpha=0.5)
        ax_delta.set_xlabel('Column Delta')
        ax_delta.set_ylabel('Row Delta')
        ax_delta.set_title(f'Input→Output Delta\nvar={delta_var:.4f}')
        ax_delta.set_aspect('equal')
        ax_delta.grid(True, alpha=0.3)

    fig.suptitle(f"Position {position} Analysis: {puzzle_id}", fontsize=12, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")

    if show:
        plt.show()
    else:
        plt.close()


def _plot_discovered_relations_heatmap(ax, relations: List[DiscoveredRelation], title: str):
    """Plot the top discovered relations as a mini heatmap."""
    # Create variance matrix from relations (they're sorted by variance)
    variance_matrix = np.full((9, 9), np.nan)

    for rel in relations:
        src_idx = ALL_ANCHORS.index(rel.relation.source_anchor)
        tgt_idx = ALL_ANCHORS.index(rel.relation.target_anchor)
        variance_matrix[src_idx, tgt_idx] = rel.variance

    # Show only the top relations
    anchor_names = [a.value.upper() for a in ALL_ANCHORS]

    # Plot with masking for NaN
    masked = np.ma.masked_invalid(variance_matrix)
    ax.imshow(masked, cmap='RdYlGn_r', aspect='auto')

    # Highlight top result
    if relations:
        best = relations[0]
        best_src = ALL_ANCHORS.index(best.relation.source_anchor)
        best_tgt = ALL_ANCHORS.index(best.relation.target_anchor)
        ax.add_patch(mpatches.Rectangle(
            (best_tgt - 0.5, best_src - 0.5), 1, 1,
            linewidth=3, edgecolor='blue', facecolor='none'
        ))

        # Title with best
        src_name = best.relation.source_anchor.value.upper()
        tgt_name = best.relation.target_anchor.value.upper()
        ax.set_title(f"{title}\nBest: {src_name}→{tgt_name} off={best.relation.offset}\nvar={best.variance:.4f}",
                    fontsize=9)
    else:
        ax.set_title(title, fontsize=9)

    ax.set_xticks(range(9))
    ax.set_xticklabels(anchor_names, fontsize=6, rotation=45)
    ax.set_yticks(range(9))
    ax.set_yticklabels(anchor_names, fontsize=6)


def compare_orderings_for_anchors(puzzle_id: str, top_k: int = 3) -> Dict:
    """
    Compare how different orderings affect anchor relationship discovery.

    This helps identify which ordering produces the most consistent
    anchor relationships.
    """
    print("=" * 70)
    print(f"COMPARING ORDERINGS FOR ANCHOR CONSISTENCY: {puzzle_id}")
    print("=" * 70)

    orderings_to_test = [
        'left_to_right',
        'top_to_bottom',
        'adaptive_reading_order',
        'quadrant_order',
    ]

    results = {}

    for ordering_name in orderings_to_test:
        print(f"\n{'='*50}")
        print(f"Testing ordering: {ordering_name}")
        print(f"{'='*50}")

        result = analyze_puzzle_anchors(
            puzzle_id,
            ordering_strategy=ordering_name,
            top_k=top_k,
            verbose=False  # Suppress detailed output
        )

        if not result:
            continue

        # Compute aggregate score: sum of best variances across all positions
        total_variance = 0
        num_perfect = 0

        for pos_result in result.get('position_analyses', []):
            # Find minimum variance across all framing options
            min_var = float('inf')

            if pos_result['grid_relations']:
                min_var = min(min_var, pos_result['grid_relations'][0].variance)

            for rels in pos_result['object_relations'].values():
                if rels:
                    min_var = min(min_var, rels[0].variance)

            if pos_result['input_delta']:
                min_var = min(min_var, pos_result['input_delta']['variance'])

            if min_var < float('inf'):
                total_variance += min_var
                if min_var == 0:
                    num_perfect += 1

        results[ordering_name] = {
            'total_variance': total_variance,
            'num_perfect': num_perfect,
            'num_positions': len(result.get('position_analyses', [])),
            'details': result,
        }

        print(f"  Total variance: {total_variance:.4f}")
        print(f"  Perfect (var=0) positions: {num_perfect}/{len(result.get('position_analyses', []))}")

    # Summary
    print("\n" + "=" * 70)
    print("ORDERING COMPARISON SUMMARY")
    print("=" * 70)

    sorted_orderings = sorted(results.items(), key=lambda x: x[1]['total_variance'])

    for i, (name, data) in enumerate(sorted_orderings):
        marker = "→ BEST" if i == 0 else ""
        print(f"  {name:25s}: total_var={data['total_variance']:.4f}, "
              f"perfect={data['num_perfect']}/{data['num_positions']} {marker}")

    return results


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Anchor Point Relationship Discovery for ARC Puzzles",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python anchoring_module.py --puzzle-id 1990f7a8
    python anchoring_module.py --puzzle-id 1990f7a8 --ordering left_to_right
    python anchoring_module.py --puzzle-id 1990f7a8 --compare-orderings
    python anchoring_module.py --puzzle-id 1990f7a8 --visualize
    python anchoring_module.py --puzzle-id 1990f7a8 --visualize-position 1
    python anchoring_module.py --puzzle-id 1990f7a8 --visualize --save output.png
    python anchoring_module.py --test   # Run synthetic tests
        """
    )

    parser.add_argument("--puzzle-id", type=str,
                        help="ARC puzzle ID to analyze (e.g., 1990f7a8)")
    parser.add_argument("--ordering", type=str, default="adaptive_reading_order",
                        help="Ordering strategy to use (default: adaptive_reading_order)")
    parser.add_argument("--top-k", type=int, default=5,
                        help="Number of top anchor relations to show (default: 5)")
    parser.add_argument("--compare-orderings", action="store_true",
                        help="Compare multiple orderings for anchor consistency")
    parser.add_argument("--test", action="store_true",
                        help="Run synthetic discovery tests")
    parser.add_argument("--ambiguity", action="store_true",
                        help="Demonstrate ambiguity cases")
    parser.add_argument("--visualize", action="store_true",
                        help="Show visual analysis of puzzle")
    parser.add_argument("--visualize-position", type=int, default=None,
                        help="Show detailed visual for specific object position")
    parser.add_argument("--save", type=str, default=None,
                        help="Save visualization to file path")

    args = parser.parse_args()

    if args.puzzle_id:
        if args.visualize:
            visualize_anchor_analysis(
                args.puzzle_id,
                ordering_strategy=args.ordering,
                save_path=args.save,
                show=True
            )
        elif args.visualize_position is not None:
            visualize_per_position_analysis(
                args.puzzle_id,
                position=args.visualize_position,
                ordering_strategy=args.ordering,
                save_path=args.save,
                show=True
            )
        elif args.compare_orderings:
            compare_orderings_for_anchors(args.puzzle_id, top_k=args.top_k)
        else:
            analyze_puzzle_anchors(
                args.puzzle_id,
                ordering_strategy=args.ordering,
                top_k=args.top_k,
                verbose=True
            )
    elif args.test:
        run_discovery_tests()
    elif args.ambiguity:
        demonstrate_ambiguity()
    else:
        parser.print_help()