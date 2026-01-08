"""
Ordering Module for ARC Puzzle Solver

Key insight: Ordering quality can only be evaluated through framing accuracy.
An ordering is "good" if it enables object_relative framings that predict positions correctly.

Usage:
    python ordering_module.py [--demo | --test | --adaptive]
    python ordering_module.py --puzzle-id 1990f7a8   # Visualize ordering for a specific puzzle
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Callable, Dict, Optional, Any
from enum import Enum, auto
from itertools import permutations
from abc import ABC, abstractmethod
from collections import defaultdict

# Object detection and extraction
from object_module import (
    Object,
    extract_objects_from_grid,
    extract_connected_components,
    labels_to_objects,
    compute_iou_from_pixels,
)

# Object correspondence matching
from correspondence_module import find_correspondences

# Puzzle loading
from puzzle_loader import load_puzzle

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap

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

# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class Correspondence:
    """Maps input object to output object."""
    input_obj: Object
    output_obj: Object


@dataclass 
class FramingResult:
    """Result of applying a framing to predict position."""
    framing_name: str
    predicted_pos: Tuple[int, int]
    actual_pos: Tuple[int, int]
    params: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def is_correct(self) -> bool:
        return self.predicted_pos == self.actual_pos
    
    @property
    def error(self) -> Tuple[int, int]:
        return (self.actual_pos[0] - self.predicted_pos[0],
                self.actual_pos[1] - self.predicted_pos[1])


@dataclass
class OrderingEvaluation:
    """Evaluation results for a particular ordering."""
    ordering_name: str
    object_order: List[int]  # list of object ids in order
    framing_results: List[List[FramingResult]]  # per object, list of tried framings
    best_framings: List[Optional[FramingResult]]  # best framing per object
    
    @property
    def success_count(self) -> int:
        return sum(1 for f in self.best_framings if f and f.is_correct)
    
    @property
    def success_rate(self) -> float:
        if not self.best_framings:
            return 0.0
        return self.success_count / len(self.best_framings)
    
    @property
    def all_correct(self) -> bool:
        return all(f and f.is_correct for f in self.best_framings)


# =============================================================================
# Ordering Strategies
# =============================================================================

class OrderingStrategy(ABC):
    """Base class for ordering strategies."""
    
    @property
    @abstractmethod
    def name(self) -> str:
        pass
    
    @abstractmethod
    def order(self, objects: List[Object]) -> List[Object]:
        """Return objects in the strategy's order."""
        pass


class LeftToRight(OrderingStrategy):
    name = "left_to_right"
    
    def order(self, objects: List[Object]) -> List[Object]:
        return sorted(objects, key=lambda o: (o.col, o.row))


class RightToLeft(OrderingStrategy):
    name = "right_to_left"
    
    def order(self, objects: List[Object]) -> List[Object]:
        return sorted(objects, key=lambda o: (-o.col, o.row))


class TopToBottom(OrderingStrategy):
    name = "top_to_bottom"
    
    def order(self, objects: List[Object]) -> List[Object]:
        return sorted(objects, key=lambda o: (o.row, o.col))


class BottomToTop(OrderingStrategy):
    name = "bottom_to_top"
    
    def order(self, objects: List[Object]) -> List[Object]:
        return sorted(objects, key=lambda o: (-o.row, o.col))


class DiagonalTLBR(OrderingStrategy):
    """Diagonal sweep from top-left to bottom-right."""
    name = "diagonal_tl_br"
    
    def order(self, objects: List[Object]) -> List[Object]:
        return sorted(objects, key=lambda o: (o.row + o.col, o.row))


class DiagonalTRBL(OrderingStrategy):
    """Diagonal sweep from top-right to bottom-left."""
    name = "diagonal_tr_bl"
    
    def order(self, objects: List[Object]) -> List[Object]:
        return sorted(objects, key=lambda o: (o.row - o.col, o.row))


class LargestFirst(OrderingStrategy):
    name = "largest_first"
    
    def order(self, objects: List[Object]) -> List[Object]:
        return sorted(objects, key=lambda o: -o.area)


class SmallestFirst(OrderingStrategy):
    name = "smallest_first"
    
    def order(self, objects: List[Object]) -> List[Object]:
        return sorted(objects, key=lambda o: o.area)


class ByColor(OrderingStrategy):
    """Order by color value (useful when colors encode sequence)."""
    name = "by_color"
    
    def order(self, objects: List[Object]) -> List[Object]:
        return sorted(objects, key=lambda o: o.color)


class AdaptiveReadingOrder(OrderingStrategy):
    """
    Adaptive reading order that works with any number of objects.
    
    Approach:
    1. Cluster objects into "rows" based on vertical gaps
    2. Order rows top-to-bottom
    3. Within each row, order left-to-right
    
    This finds the natural reading structure regardless of object count.
    """
    name = "adaptive_reading_order"
    
    def __init__(self, row_gap_threshold: float = None):
        # If None, automatically detect gaps
        self.row_gap_threshold = row_gap_threshold
    
    def order(self, objects: List[Object]) -> List[Object]:
        if len(objects) <= 1:
            return objects
        
        # Get centers and sort by row
        sorted_by_row = sorted(objects, key=lambda o: o.center[0])
        
        # Find row clusters by detecting vertical gaps
        row_clusters = self._cluster_into_rows(sorted_by_row)
        
        # Build final order: rows top-to-bottom, within each row left-to-right
        result = []
        for row_cluster in row_clusters:
            row_sorted = sorted(row_cluster, key=lambda o: o.center[1])
            result.extend(row_sorted)
        
        return result
    
    def _cluster_into_rows(self, objects_by_row: List[Object]) -> List[List[Object]]:
        """Group objects into rows based on vertical gaps."""
        if not objects_by_row:
            return []

        # Calculate gaps between consecutive objects (by row position)
        centers = [o.center[0] for o in objects_by_row]
        gaps = [centers[i+1] - centers[i] for i in range(len(centers)-1)]

        if not gaps:
            return [objects_by_row]

        # Determine threshold: use provided value or find natural breaks
        if self.row_gap_threshold is not None:
            threshold = self.row_gap_threshold
        else:
            # Adaptive threshold - look for natural breaks in the gap distribution
            #
            # Key insight: a "row break" gap should be significantly larger than
            # "within-row" gaps. We need to find the PRIMARY row break, not
            # treat every moderate gap as a break.

            sorted_gaps = sorted(gaps)
            total_span = centers[-1] - centers[0] if len(centers) > 1 else 1

            if len(gaps) == 1:
                # Only 2 objects - any significant gap means different rows
                threshold = max(sorted_gaps[0] * 0.5, 2)
            else:
                # Strategy: Find gaps that are clearly "between-row" gaps vs "within-row"
                #
                # A true row break should satisfy BOTH:
                # 1. Be significantly larger than smaller gaps (ratio test)
                # 2. Represent a substantial portion of total vertical span

                max_gap = sorted_gaps[-1]

                # Check if max gap is dominant (significantly larger than others)
                # Compare max gap to the second-largest gap
                second_max_gap = sorted_gaps[-2] if len(sorted_gaps) > 1 else 0

                # The max gap should be at least 1.3x the second largest to be
                # considered the "true" row break
                max_gap_is_dominant = max_gap >= second_max_gap * 1.3

                # Also check that max gap represents a meaningful portion of span
                # (at least 25% of total vertical distance)
                max_gap_is_substantial = max_gap >= total_span * 0.25

                if max_gap_is_dominant and max_gap_is_substantial:
                    # Clear single row break - set threshold just below max gap
                    # This ensures only the largest gap(s) trigger row breaks
                    threshold = (second_max_gap + max_gap) / 2
                else:
                    # No clear dominant gap - fall back to original method
                    # but be more conservative
                    gap_jumps = [(sorted_gaps[i+1] - sorted_gaps[i], i)
                                 for i in range(len(sorted_gaps)-1)]

                    if gap_jumps:
                        max_jump, max_jump_idx = max(gap_jumps, key=lambda x: x[0])

                        # More conservative: require larger jump relative to gaps
                        if max_jump > max(sorted_gaps[max_jump_idx], 2):
                            threshold = (sorted_gaps[max_jump_idx] + sorted_gaps[max_jump_idx + 1]) / 2
                        else:
                            # No clear separation - all objects likely on same row
                            threshold = sorted_gaps[-1] + 1
                    else:
                        threshold = sorted_gaps[-1] + 1
        
        # Split into clusters at large gaps
        clusters = []
        current_cluster = [objects_by_row[0]]
        
        for i, gap in enumerate(gaps):
            if gap > threshold:
                clusters.append(current_cluster)
                current_cluster = []
            current_cluster.append(objects_by_row[i + 1])
        
        clusters.append(current_cluster)
        return clusters
    
    def get_row_structure(self, objects: List[Object]) -> List[List[int]]:
        """Return the detected row structure as lists of object IDs (for debugging)."""
        if len(objects) <= 1:
            return [[o.id for o in objects]]
        
        sorted_by_row = sorted(objects, key=lambda o: o.center[0])
        row_clusters = self._cluster_into_rows(sorted_by_row)
        
        result = []
        for row_cluster in row_clusters:
            row_sorted = sorted(row_cluster, key=lambda o: o.center[1])
            result.append([o.id for o in row_sorted])
        return result


class QuadrantOrder(OrderingStrategy):
    """
    Assign objects to quadrants based on median position, then order in reading order.

    Unlike AdaptiveReadingOrder which clusters by vertical gaps, this strategy
    divides space into quadrants using the median object position as the center.
    Objects are assigned to quadrants (TL, TR, BL, BR) and ordered accordingly.

    This is useful when objects should be grouped by spatial region regardless
    of their exact vertical positions (e.g., when "top-left" conceptually means
    "in the top-left area" not "in the topmost row").
    """
    name = "quadrant_order"

    def order(self, objects: List[Object]) -> List[Object]:
        if len(objects) <= 1:
            return objects

        # Find the dividing point using medians of object positions
        centers = [o.center for o in objects]
        mid_row = np.median([c[0] for c in centers])
        mid_col = np.median([c[1] for c in centers])

        def quadrant_key(obj: Object) -> Tuple[int, float, float]:
            cr, cc = obj.center
            # Determine quadrant: 0=TL, 1=TR, 2=BL, 3=BR (reading order)
            is_bottom = cr >= mid_row
            is_right = cc >= mid_col
            quadrant = is_bottom * 2 + is_right
            # Within quadrant, use reading order as tiebreaker
            return (quadrant, cr, cc)

        return sorted(objects, key=quadrant_key)

    def get_quadrant_structure(self, objects: List[Object]) -> Dict[str, List[int]]:
        """Return objects grouped by quadrant (for debugging)."""
        if not objects:
            return {}

        centers = [o.center for o in objects]
        mid_row = np.median([c[0] for c in centers])
        mid_col = np.median([c[1] for c in centers])

        quadrants = {'TL': [], 'TR': [], 'BL': [], 'BR': []}

        for obj in objects:
            cr, cc = obj.center
            if cr < mid_row:
                q = 'TL' if cc < mid_col else 'TR'
            else:
                q = 'BL' if cc < mid_col else 'BR'
            quadrants[q].append(obj.id)

        return quadrants


# =============================================================================
# Per-Parent Ordering (for hierarchical objects)
# =============================================================================

class PerParentOrdering(OrderingStrategy):
    """
    Orders children within each parent using a consistent strategy.

    This is an OrderingStrategy that:
    1. Separates parents from children
    2. Orders parents using a parent ordering strategy
    3. Orders children within each parent using a child ordering strategy
    4. Returns objects in order: [parent1, child1a, child1b, parent2, child2a, ...]

    For hierarchical objects, this enables rules like:
    "the 1st child (by left-to-right) in each parent goes to position X"

    Example:
        Parent A has children [c1, c2, c3] at positions [(2,5), (2,10), (2,15)]
        Parent B has children [c4, c5] at positions [(8,3), (8,9)]

        With left_to_right child ordering:
        - Parent A's children ordered: [c1, c2, c3] (indices 0, 1, 2)
        - Parent B's children ordered: [c4, c5] (indices 0, 1)

        If the rule is "child at index 0 goes to parent.top_left + (1, 1)",
        this applies consistently to c1 and c4.
    """

    def __init__(self, child_ordering: OrderingStrategy, parent_ordering: Optional[OrderingStrategy] = None):
        """
        Args:
            child_ordering: Strategy to order children within each parent
            parent_ordering: Strategy to order parents (default: left_to_right)
        """
        self.child_ordering = child_ordering
        self.parent_ordering = parent_ordering or LeftToRight()

    @property
    def name(self) -> str:
        return f"per_parent({self.child_ordering.name})"

    def order(self, objects: List[Object]) -> List[Object]:
        """
        Order objects: parents first (in parent_ordering), then children within each parent.

        For objects without hierarchy (no parent/children), falls back to child_ordering.

        Args:
            objects: List of objects (may include parents with children)

        Returns:
            Ordered list of objects
        """
        if not objects:
            return []

        # Separate parents (objects with children) and children (objects with parent)
        parents = [o for o in objects if o.children]
        children = [o for o in objects if o.parent is not None]
        standalone = [o for o in objects if not o.children and o.parent is None]

        # If no hierarchy, fall back to child ordering on all objects
        if not parents and not children:
            return self.child_ordering.order(objects)

        # Order parents
        ordered_parents = self.parent_ordering.order(parents)

        # Build result: for each parent, add parent then its ordered children
        result = []
        for parent in ordered_parents:
            result.append(parent)
            parent_children = [c for c in children if c.parent and c.parent.id == parent.id]
            ordered_children = self.child_ordering.order(parent_children)
            result.extend(ordered_children)

        # Add any standalone objects at the end
        result.extend(self.child_ordering.order(standalone))

        return result

    def order_children_by_parent(self, objects: List[Object]) -> Dict[int, List[Object]]:
        """
        Group objects by parent and order children within each group.

        Args:
            objects: List of objects (should be children with parent references)

        Returns:
            Dict mapping parent_id -> ordered list of children
        """
        # Group by parent
        by_parent: Dict[Optional[int], List[Object]] = defaultdict(list)
        for obj in objects:
            parent_id = obj.parent.id if obj.parent else None
            by_parent[parent_id].append(obj)

        # Order children within each parent
        result = {}
        for parent_id, children in by_parent.items():
            if parent_id is not None:  # Only process actual children
                result[parent_id] = self.child_ordering.order(children)

        return result

    def get_child_index(self, obj: Object, all_children: List[Object]) -> Optional[int]:
        """
        Get the index of an object within its parent's ordered children.

        Args:
            obj: The child object to find
            all_children: All children across all parents

        Returns:
            Index within parent (0-based), or None if obj has no parent
        """
        if obj.parent is None:
            return None

        # Get siblings (children of same parent)
        siblings = [o for o in all_children if o.parent and o.parent.id == obj.parent.id]
        ordered_siblings = self.child_ordering.order(siblings)

        for idx, sibling in enumerate(ordered_siblings):
            if sibling.id == obj.id:
                return idx
        return None

    def get_indexed_children(self, objects: List[Object]) -> Dict[int, List[Tuple[Object, int]]]:
        """
        Get all children with their per-parent indices.

        Args:
            objects: List of child objects

        Returns:
            Dict mapping parent_id -> list of (child, index) tuples
        """
        ordered_by_parent = self.order_children_by_parent(objects)
        result = {}
        for parent_id, ordered_children in ordered_by_parent.items():
            result[parent_id] = [(child, idx) for idx, child in enumerate(ordered_children)]
        return result

    def group_by_child_index(self, objects: List[Object]) -> Dict[int, List[Object]]:
        """
        Group children across all parents by their index within parent.

        This is useful for finding patterns like "all index-0 children
        have the same offset from their parent".

        Args:
            objects: List of child objects

        Returns:
            Dict mapping child_index -> list of children at that index
        """
        by_index: Dict[int, List[Object]] = defaultdict(list)
        ordered_by_parent = self.order_children_by_parent(objects)

        for parent_id, ordered_children in ordered_by_parent.items():
            for idx, child in enumerate(ordered_children):
                by_index[idx].append(child)

        return dict(by_index)


@dataclass
class PerParentOrderingResult:
    """Result of screening per-parent orderings."""
    ordering_name: str
    child_ordering: OrderingStrategy
    per_parent_ordering: PerParentOrdering
    # For each child index, the variance of parent-relative offsets
    index_variances: Dict[int, float]
    # Average variance across all indices
    avg_variance: float
    # Whether all indices have zero (or near-zero) variance
    is_consistent: bool
    # Learned offsets per index (if consistent)
    learned_offsets: Dict[int, Tuple[int, int]]


@dataclass
class OrderingScreenResult:
    """Comprehensive result from unified ordering screening.

    This is the return type of find_best_ordering(), providing all information
    needed to apply the best ordering strategy to a puzzle.
    """
    # Best overall configuration
    best_mode: str  # 'global' or 'per_parent'
    best_ordering_name: str
    is_consistent: bool

    # For global ordering (always populated)
    global_ordering: Optional[OrderingStrategy]
    global_results: Dict[str, Dict]  # Per-strategy results

    # For per-parent ordering (populated if hierarchy exists)
    per_parent_result: Optional[PerParentOrderingResult]
    has_hierarchy: bool

    def get_ordering_strategy(self) -> Optional[OrderingStrategy]:
        """Get the OrderingStrategy instance for the best global ordering."""
        return self.global_ordering

    def describe(self) -> str:
        """Human-readable description of the best ordering."""
        if self.best_mode == 'per_parent' and self.per_parent_result:
            return (f"Per-parent ordering using {self.per_parent_result.child_ordering.name} "
                    f"(consistent: {self.is_consistent})")
        else:
            return f"Global ordering: {self.best_ordering_name} (consistent: {self.is_consistent})"


def screen_per_parent_orderings(
    output_roots: List[Object],
    verbose: bool = False
) -> Optional[PerParentOrderingResult]:
    """
    Screen ordering strategies to find one where children at the same index
    across different parents have consistent parent-relative positions.

    Args:
        output_roots: List of root objects from output (with children)
        verbose: Print detailed results

    Returns:
        PerParentOrderingResult for best ordering, or None if no hierarchy
    """
    # Collect all children from output roots
    all_output_children = []
    for root in output_roots:
        all_output_children.extend(root.children)

    if len(all_output_children) < 2:
        return None

    # Check we have multiple parents with children
    parents_with_children = [r for r in output_roots if r.children]
    if len(parents_with_children) < 2:
        return None

    if verbose:
        print("\n" + "=" * 60)
        print("PER-PARENT ORDERING SCREENING")
        print("=" * 60)
        print(f"Parents with children: {len(parents_with_children)}")
        print(f"Total children: {len(all_output_children)}")

    best_result = None
    best_variance = float('inf')

    # Try each base ordering strategy
    for base_ordering in ALL_ORDERINGS:
        per_parent = PerParentOrdering(base_ordering)

        # Group children by index
        by_index = per_parent.group_by_child_index(all_output_children)

        if not by_index:
            continue

        # For each index, compute variance of parent-relative offsets
        index_variances = {}
        learned_offsets = {}

        for idx, children in by_index.items():
            if len(children) < 2:
                # Can't compute variance with single sample
                index_variances[idx] = 0.0
                if children:
                    child = children[0]
                    offset = (child.row - child.parent.row, child.col - child.parent.col)
                    learned_offsets[idx] = offset
                continue

            # Compute parent-relative offsets
            offsets = []
            for child in children:
                offset_row = child.row - child.parent.row
                offset_col = child.col - child.parent.col
                offsets.append((offset_row, offset_col))

            # Compute variance (sum of row variance + col variance)
            rows = [o[0] for o in offsets]
            cols = [o[1] for o in offsets]
            var_row = np.var(rows)
            var_col = np.var(cols)
            total_var = var_row + var_col

            index_variances[idx] = total_var

            # Learn offset (use mean, rounded)
            mean_row = int(round(np.mean(rows)))
            mean_col = int(round(np.mean(cols)))
            learned_offsets[idx] = (mean_row, mean_col)

        # Average variance across indices
        if index_variances:
            avg_variance = np.mean(list(index_variances.values()))
        else:
            avg_variance = float('inf')

        is_consistent = avg_variance < 0.1  # Near-zero variance threshold

        if verbose:
            status = "CONSISTENT" if is_consistent else "inconsistent"
            print(f"\n  {base_ordering.name}: {status} (avg_var={avg_variance:.4f})")
            for idx, var in sorted(index_variances.items()):
                offset = learned_offsets.get(idx, (0, 0))
                print(f"    Index {idx}: variance={var:.4f}, offset={offset}")

        if avg_variance < best_variance:
            best_variance = avg_variance
            best_result = PerParentOrderingResult(
                ordering_name=per_parent.name,
                child_ordering=base_ordering,
                per_parent_ordering=per_parent,
                index_variances=index_variances,
                avg_variance=avg_variance,
                is_consistent=is_consistent,
                learned_offsets=learned_offsets,
            )

    if verbose and best_result:
        print("\n" + "-" * 60)
        print(f"Best per-parent ordering: {best_result.ordering_name}")
        print(f"Consistent: {best_result.is_consistent}")
        print(f"Learned offsets by child index:")
        for idx, offset in sorted(best_result.learned_offsets.items()):
            print(f"  Index {idx}: parent + {offset}")

    return best_result


def screen_per_parent_orderings_for_puzzle(
    puzzle: Dict,
    verbose: bool = False
) -> Optional[PerParentOrderingResult]:
    """
    Screen per-parent orderings across all training examples of a puzzle.

    Aggregates children from all training examples to find an ordering where
    children at the same index (across all parents in all examples) have
    consistent parent-relative positions.

    Args:
        puzzle: Puzzle dict with 'train' examples
        verbose: Print detailed results

    Returns:
        PerParentOrderingResult for best ordering, or None if no hierarchy
    """
    from object_module import extract_objects_from_grid

    # Collect all output children across all training examples
    all_output_children = []
    total_parents = 0

    for pair in puzzle.get('train', []):
        if 'output' not in pair:
            continue

        output_grid = np.array(pair['output'])
        output_roots = extract_objects_from_grid(output_grid, build_hierarchy=True)

        for root in output_roots:
            if root.children:
                total_parents += 1
                all_output_children.extend(root.children)

    if len(all_output_children) < 2 or total_parents < 2:
        if verbose:
            print("Not enough hierarchical structure for per-parent ordering")
        return None

    if verbose:
        print("\n" + "=" * 60)
        print("PER-PARENT ORDERING SCREENING (across all training examples)")
        print("=" * 60)
        print(f"Total parents with children: {total_parents}")
        print(f"Total children: {len(all_output_children)}")

    best_result = None
    best_variance = float('inf')

    # Try each base ordering strategy
    for base_ordering in ALL_ORDERINGS:
        per_parent = PerParentOrdering(base_ordering)

        # Group children by index
        by_index = per_parent.group_by_child_index(all_output_children)

        if not by_index:
            continue

        # For each index, compute variance of parent-relative offsets
        index_variances = {}
        learned_offsets = {}

        for idx, children in by_index.items():
            if len(children) < 2:
                index_variances[idx] = 0.0
                if children:
                    child = children[0]
                    offset = (child.row - child.parent.row, child.col - child.parent.col)
                    learned_offsets[idx] = offset
                continue

            # Compute parent-relative offsets
            offsets = []
            for child in children:
                offset_row = child.row - child.parent.row
                offset_col = child.col - child.parent.col
                offsets.append((offset_row, offset_col))

            # Compute variance
            rows = [o[0] for o in offsets]
            cols = [o[1] for o in offsets]
            var_row = np.var(rows)
            var_col = np.var(cols)
            total_var = var_row + var_col

            index_variances[idx] = total_var

            mean_row = int(round(np.mean(rows)))
            mean_col = int(round(np.mean(cols)))
            learned_offsets[idx] = (mean_row, mean_col)

        if index_variances:
            avg_variance = np.mean(list(index_variances.values()))
        else:
            avg_variance = float('inf')

        is_consistent = avg_variance < 0.1

        if verbose:
            status = "CONSISTENT" if is_consistent else "inconsistent"
            print(f"\n  {base_ordering.name}: {status} (avg_var={avg_variance:.4f})")
            for idx, var in sorted(index_variances.items()):
                offset = learned_offsets.get(idx, (0, 0))
                print(f"    Index {idx}: variance={var:.4f}, offset={offset}")

        if avg_variance < best_variance:
            best_variance = avg_variance
            best_result = PerParentOrderingResult(
                ordering_name=per_parent.name,
                child_ordering=base_ordering,
                per_parent_ordering=per_parent,
                index_variances=index_variances,
                avg_variance=avg_variance,
                is_consistent=is_consistent,
                learned_offsets=learned_offsets,
            )

    if verbose and best_result:
        print("\n" + "-" * 60)
        print(f"Best per-parent ordering: {best_result.ordering_name}")
        print(f"Consistent: {best_result.is_consistent}")
        print(f"Learned offsets by child index:")
        for idx, offset in sorted(best_result.learned_offsets.items()):
            print(f"  Index {idx}: parent + {offset}")

    return best_result


# Registry of all ordering strategies (global orderings)
ALL_ORDERINGS: List[OrderingStrategy] = [
    LeftToRight(),
    RightToLeft(),
    TopToBottom(),
    BottomToTop(),
    DiagonalTLBR(),
    DiagonalTRBL(),
    LargestFirst(),
    SmallestFirst(),
    ByColor(),
    AdaptiveReadingOrder(),
    QuadrantOrder(),
]

# Per-parent orderings (for hierarchical objects)
# These use different child ordering strategies within each parent
PER_PARENT_ORDERINGS: List[PerParentOrdering] = [
    PerParentOrdering(LeftToRight()),
    PerParentOrdering(RightToLeft()),
    PerParentOrdering(TopToBottom()),
    PerParentOrdering(BottomToTop()),
    PerParentOrdering(LargestFirst()),
    PerParentOrdering(SmallestFirst()),
    PerParentOrdering(ByColor()),
]


# =============================================================================
# Unified Ordering Discovery (like find_correspondences)
# =============================================================================

def find_best_ordering(
    puzzle: Dict,
    verbose: bool = False,
    selection_criterion: Optional[str] = None,
    selection_rule: Optional[str] = None,
    use_color_only: bool = False
) -> OrderingScreenResult:
    """
    Unified ordering screening - THE canonical function for discovering the best ordering.

    Similar to find_correspondences() for matching, this is the single entry point
    for ordering discovery. It automatically handles:
    - All global orderings (left_to_right, top_to_bottom, adaptive_reading_order, etc.)
    - Per-parent orderings (if hierarchy exists in the puzzle)

    All orderings go through the same ConsistencyChecker system, which evaluates
    whether framings produce consistent (zero-variance) parameters across training examples.

    Strategy:
    1. Try all global orderings first
    2. If hierarchy exists, also try per-parent orderings
    3. Select the best based on: consistency first, then number of framings, then preference

    Args:
        puzzle: Puzzle dict with 'train' examples (each having 'input' and 'output')
        verbose: Print detailed screening results
        selection_criterion: Optional criterion for pre-filtering objects
        selection_rule: Optional rule for pre-filtering objects
        use_color_only: Object extraction mode

    Returns:
        OrderingScreenResult with:
        - best_mode: 'global' or 'per_parent'
        - best_ordering_name: Name of the best strategy
        - is_consistent: Whether the best strategy has consistent framings
        - global_ordering: The OrderingStrategy instance
        - global_results: Per-strategy results for all orderings
        - per_parent_result: None (deprecated, per-parent now in global_results)
        - has_hierarchy: Whether hierarchical structure was detected

    Usage:
        result = find_best_ordering(puzzle, verbose=True)
        ordering = result.global_ordering
        # Apply ordering to objects
    """
    if verbose:
        print("\n" + "=" * 60)
        print("UNIFIED ORDERING SCREENING")
        print("=" * 60)

    # Screen all orderings (global + per-parent if hierarchy exists)
    screen_result = screen_orderings_for_puzzle(
        puzzle,
        verbose=verbose,
        selection_criterion=selection_criterion,
        selection_rule=selection_rule,
        use_color_only=use_color_only,
        include_per_parent=True
    )

    best_ordering = screen_result['best_ordering']
    best_name = screen_result['best_name']
    is_consistent = screen_result['is_consistent']
    all_results = screen_result['all_results']
    has_hierarchy = screen_result.get('has_hierarchy', False)

    # Determine if best is per-parent
    best_mode = 'per_parent' if best_name.startswith('per_parent(') else 'global'

    if verbose:
        print(f"\n*** Best: {best_mode} - {best_name} (consistent: {is_consistent}) ***")

    return OrderingScreenResult(
        best_mode=best_mode,
        best_ordering_name=best_name,
        is_consistent=is_consistent,
        global_ordering=best_ordering,
        global_results=all_results,
        per_parent_result=None,  # Deprecated - per-parent now goes through same system
        has_hierarchy=has_hierarchy,
    )


# =============================================================================
# Framings (Position Predictors)
# =============================================================================

class Framing(ABC):
    """Base class for position prediction framings."""
    
    @property
    @abstractmethod
    def name(self) -> str:
        pass
    
    @abstractmethod
    def predict(self, 
                input_obj: Object, 
                earlier_outputs: List[Object],
                grid_shape: Tuple[int, int],
                **learned_params) -> Tuple[int, int]:
        """Predict output position given input object and context."""
        pass
    
    @abstractmethod
    def learn(self,
              input_obj: Object,
              output_obj: Object,
              earlier_outputs: List[Object],
              grid_shape: Tuple[int, int]) -> Dict[str, Any]:
        """Learn parameters from a single example."""
        pass


class DeltaFraming(Framing):
    """Predict position as offset from input position."""
    name = "delta"
    
    def predict(self, input_obj, earlier_outputs, grid_shape, **params) -> Tuple[int, int]:
        delta_row = params.get('delta_row', 0)
        delta_col = params.get('delta_col', 0)
        return (input_obj.row + delta_row, input_obj.col + delta_col)
    
    def learn(self, input_obj, output_obj, earlier_outputs, grid_shape) -> Dict:
        return {
            'delta_row': output_obj.row - input_obj.row,
            'delta_col': output_obj.col - input_obj.col
        }


class GridAbsoluteFraming(Framing):
    """Predict absolute position relative to grid corner."""
    
    def __init__(self, corner: str = 'tl'):
        self.corner = corner
        
    @property
    def name(self) -> str:
        return f"grid_{self.corner}"
    
    def predict(self, input_obj, earlier_outputs, grid_shape, **params) -> Tuple[int, int]:
        abs_row = params.get('abs_row', 0)
        abs_col = params.get('abs_col', 0)
        h, w = grid_shape
        
        if self.corner == 'tl':
            return (abs_row, abs_col)
        elif self.corner == 'tr':
            return (abs_row, w - 1 - abs_col)
        elif self.corner == 'bl':
            return (h - 1 - abs_row, abs_col)
        elif self.corner == 'br':
            return (h - 1 - abs_row, w - 1 - abs_col)
        return (abs_row, abs_col)
    
    def learn(self, input_obj, output_obj, earlier_outputs, grid_shape) -> Dict:
        h, w = grid_shape
        
        if self.corner == 'tl':
            return {'abs_row': output_obj.row, 'abs_col': output_obj.col}
        elif self.corner == 'tr':
            return {'abs_row': output_obj.row, 'abs_col': w - 1 - output_obj.col}
        elif self.corner == 'bl':
            return {'abs_row': h - 1 - output_obj.row, 'abs_col': output_obj.col}
        elif self.corner == 'br':
            return {'abs_row': h - 1 - output_obj.row, 'abs_col': w - 1 - output_obj.col}
        return {}


class ObjectRelativeFraming(Framing):
    """Predict position relative to an earlier object in the sequence."""

    def __init__(self, ref_index: int):
        self.ref_index = ref_index  # index in the ordered sequence (must be < current)

    @property
    def name(self) -> str:
        return f"object_relative(ref={self.ref_index})"

    def predict(self, input_obj, earlier_outputs, grid_shape, **params) -> Tuple[int, int]:
        if self.ref_index >= len(earlier_outputs):
            # Reference doesn't exist yet - this framing is invalid
            return (-999, -999)

        ref_obj = earlier_outputs[self.ref_index]
        rel_row = params.get('rel_row', 0)
        rel_col = params.get('rel_col', 0)
        return (ref_obj.row + rel_row, ref_obj.col + rel_col)

    def learn(self, input_obj, output_obj, earlier_outputs, grid_shape) -> Dict:
        if self.ref_index >= len(earlier_outputs):
            return {'rel_row': -999, 'rel_col': -999}  # invalid

        ref_obj = earlier_outputs[self.ref_index]
        return {
            'rel_row': output_obj.row - ref_obj.row,
            'rel_col': output_obj.col - ref_obj.col
        }


class ParentRelativeFraming(Framing):
    """Predict position relative to parent object (for hierarchical objects).

    This framing is used when objects have been organized into a hierarchy
    using build_containment_hierarchy(). It predicts a child's position
    based on its offset from its parent container.

    Note: This framing only works for objects with a parent. For root objects
    (those with no parent), it returns an invalid position (-999, -999).
    """
    name = "parent_relative"

    def predict(self, input_obj, earlier_outputs, grid_shape, **params) -> Tuple[int, int]:
        # Check if the object has a parent
        if not hasattr(input_obj, 'parent') or input_obj.parent is None:
            return (-999, -999)  # Invalid for non-hierarchical objects

        parent = input_obj.parent
        rel_row = params.get('rel_row', 0)
        rel_col = params.get('rel_col', 0)
        return (parent.row + rel_row, parent.col + rel_col)

    def learn(self, input_obj, output_obj, earlier_outputs, grid_shape) -> Dict:
        # Check if the output object has a parent
        if not hasattr(output_obj, 'parent') or output_obj.parent is None:
            return {'rel_row': -999, 'rel_col': -999}  # invalid

        parent = output_obj.parent
        return {
            'rel_row': output_obj.row - parent.row,
            'rel_col': output_obj.col - parent.col
        }


def get_all_framings(max_object_refs: int = 3, include_parent: bool = False) -> List[Framing]:
    """Generate all framings to try.

    Args:
        max_object_refs: Maximum number of object-relative framings to include
        include_parent: If True, include ParentRelativeFraming for hierarchical objects

    Returns:
        List of Framing instances to evaluate
    """
    framings = [
        DeltaFraming(),
        GridAbsoluteFraming('tl'),
        GridAbsoluteFraming('tr'),
        GridAbsoluteFraming('bl'),
        GridAbsoluteFraming('br'),
    ]

    # Add object-relative framings for each possible reference
    for i in range(max_object_refs):
        framings.append(ObjectRelativeFraming(ref_index=i))

    # Add parent-relative framing for hierarchical objects
    if include_parent:
        framings.append(ParentRelativeFraming())

    return framings


# =============================================================================
# Evaluation Engine
# =============================================================================

class OrderingEvaluator:
    """Evaluates orderings by testing framing accuracy."""
    
    def __init__(self, framings: List[Framing] = None):
        self.framings = framings or get_all_framings()
    
    def evaluate_ordering(self,
                         correspondences: List[Correspondence],
                         ordering: OrderingStrategy,
                         grid_shape: Tuple[int, int]) -> OrderingEvaluation:
        """
        Evaluate an ordering by:
        1. Reorder correspondences according to the strategy
        2. For each object, try all framings
        3. Track which framings succeed
        """
        # Sort correspondences by the ordering strategy applied to input objects
        input_objects = [c.input_obj for c in correspondences]
        ordered_inputs = ordering.order(input_objects)
        
        # Map input objects to correspondences
        input_to_corr = {c.input_obj.id: c for c in correspondences}
        ordered_corrs = [input_to_corr[obj.id] for obj in ordered_inputs]
        
        all_framing_results = []
        best_framings = []
        earlier_outputs = []
        
        for i, corr in enumerate(ordered_corrs):
            input_obj = corr.input_obj
            output_obj = corr.output_obj
            actual_pos = (output_obj.row, output_obj.col)
            
            obj_results = []
            best_result = None
            
            for framing in self.framings:
                # Skip object_relative framings that reference non-existent objects
                if isinstance(framing, ObjectRelativeFraming):
                    if framing.ref_index >= i:
                        continue
                
                # Learn parameters from this example
                params = framing.learn(input_obj, output_obj, earlier_outputs, grid_shape)
                
                # Predict position
                predicted = framing.predict(input_obj, earlier_outputs, grid_shape, **params)
                
                result = FramingResult(
                    framing_name=framing.name,
                    predicted_pos=predicted,
                    actual_pos=actual_pos,
                    params=params
                )
                obj_results.append(result)
                
                if result.is_correct:
                    if best_result is None:
                        best_result = result
            
            all_framing_results.append(obj_results)
            best_framings.append(best_result)
            earlier_outputs.append(output_obj)
        
        return OrderingEvaluation(
            ordering_name=ordering.name,
            object_order=[c.input_obj.id for c in ordered_corrs],
            framing_results=all_framing_results,
            best_framings=best_framings
        )
    
    def evaluate_all_orderings(self,
                               correspondences: List[Correspondence],
                               grid_shape: Tuple[int, int],
                               orderings: List[OrderingStrategy] = None) -> List[OrderingEvaluation]:
        """Evaluate all ordering strategies and return results sorted by success."""
        orderings = orderings or ALL_ORDERINGS
        
        results = []
        for ordering in orderings:
            eval_result = self.evaluate_ordering(correspondences, ordering, grid_shape)
            results.append(eval_result)
        
        # Sort by success rate (descending)
        results.sort(key=lambda e: (-e.success_rate, -e.success_count))
        return results
    
    def find_best_ordering(self,
                          correspondences: List[Correspondence],
                          grid_shape: Tuple[int, int]) -> OrderingEvaluation:
        """Find the ordering that enables the best framing accuracy."""
        results = self.evaluate_all_orderings(correspondences, grid_shape)
        return results[0] if results else None


# =============================================================================
# Cross-Example Consistency Checker
# =============================================================================

class ConsistencyChecker:
    """
    Checks if an ordering + framing combination works consistently 
    across multiple training examples.
    
    Key insight: we need to check that the SAME learned parameters work,
    not just that some framing works per example.
    """
    
    def __init__(self):
        self.evaluator = OrderingEvaluator()
    
    def check_consistency(self,
                         examples: List[Tuple[List[Correspondence], Tuple[int, int]]],
                         ordering: OrderingStrategy) -> Dict:
        """
        Check if a framing with CONSISTENT PARAMETERS exists across examples.
        
        For a framing to be truly consistent:
        - Same framing type must work at same position across all examples
        - The learned parameters must be identical
        
        Returns dict with:
        - consistent: bool - whether a consistent framing+params exists for each position
        - framings: list of (framing_name, params) per position that are consistent
        - details: per-example breakdown
        """
        if not examples:
            return {'consistent': False, 'framings': [], 'details': []}
        
        # Collect (framing_name, params) for each position in each example
        per_example_results = []
        
        for correspondences, grid_shape in examples:
            eval_result = self.evaluator.evaluate_ordering(correspondences, ordering, grid_shape)
            
            # For each position, get dict of framing_name -> params for working framings
            working_per_pos = []
            for obj_results in eval_result.framing_results:
                working = {}
                for r in obj_results:
                    if r.is_correct:
                        # Convert params to hashable tuple for comparison
                        param_key = tuple(sorted(r.params.items()))
                        working[r.framing_name] = param_key
                working_per_pos.append(working)
            
            per_example_results.append(working_per_pos)
        
        # Find framings with IDENTICAL parameters across all examples
        num_positions = len(per_example_results[0]) if per_example_results else 0
        consistent_framings = []
        
        for pos in range(num_positions):
            pos_results = [ex[pos] if pos < len(ex) else {} for ex in per_example_results]
            
            # Find framings present in ALL examples at this position
            if not pos_results:
                consistent_framings.append([])
                continue
                
            common_framings = set(pos_results[0].keys())
            for pr in pos_results[1:]:
                common_framings &= set(pr.keys())
            
            # Now check which have identical params
            truly_consistent = []
            for framing_name in common_framings:
                params_across_examples = [pr[framing_name] for pr in pos_results]
                # Check if all params are identical
                if len(set(params_across_examples)) == 1:
                    # Convert back to readable format
                    params_dict = dict(params_across_examples[0])
                    truly_consistent.append((framing_name, params_dict))
            
            consistent_framings.append(truly_consistent)
        
        # Check if every position has at least one truly consistent framing
        all_consistent = all(len(cf) > 0 for cf in consistent_framings)
        
        return {
            'consistent': all_consistent,
            'framings': consistent_framings,
            'details': per_example_results
        }


# =============================================================================
# Object Extraction Helpers
# =============================================================================

def compute_iou(pixels1: set, pixels2: set) -> float:
    """Compute Intersection over Union between two pixel sets.

    This is a thin wrapper around object_module.compute_iou_from_pixels
    for backwards compatibility.
    """
    return compute_iou_from_pixels(pixels1, pixels2)


def find_object_correspondences_simple(
    input_objects: List[Object],
    output_objects: List[Object],
    input_grid: np.ndarray,
    output_grid: np.ndarray,
    threshold: float = 0.1
) -> List[Correspondence]:
    """
    Find correspondences between input and output objects using canonical shape-based matching.

    Uses the shared find_correspondences() function with:
    - Rich shape features (structural, Hu moments, color distribution, Fourier descriptors)
    - Hierarchy awareness via get_matchable_objects()

    Args:
        input_objects: List of input Object instances
        output_objects: List of output Object instances
        input_grid: The input grid array
        output_grid: The output grid array
        threshold: Minimum similarity threshold for a valid match

    Returns:
        List of Correspondence instances
    """
    if not input_objects or not output_objects:
        return []

    # Use canonical correspondence function with hierarchy handling
    matches, matchable_in, matchable_out = find_correspondences(
        input_grid, output_grid,
        input_objects, output_objects,
        threshold=threshold,
        use_matchable=True
    )

    if not matchable_in or not matchable_out:
        return []

    # Convert (input_idx, output_idx, score) tuples to Correspondence objects
    # Indices are into the matchable lists
    return [
        Correspondence(matchable_in[in_idx], matchable_out[out_idx])
        for in_idx, out_idx, _ in matches
    ]


def screen_orderings_for_puzzle(
    puzzle: Dict,
    verbose: bool = False,
    selection_criterion: Optional[str] = None,
    selection_rule: Optional[str] = None,
    use_color_only: bool = False,
    include_per_parent: bool = True
) -> Dict:
    """
    Screen all ordering strategies for a puzzle and find the best one.

    This includes both global orderings (left_to_right, top_to_bottom, etc.)
    and per-parent orderings (if hierarchy exists in the puzzle).

    If selection_criterion and selection_rule are provided, applies selection
    filtering BEFORE ordering evaluation (selection first, then ordering).

    Args:
        puzzle: Puzzle dict with 'train' examples (each having 'input' and 'output')
        verbose: Print detailed results
        selection_criterion: Optional ranking criterion for pre-filtering objects
        selection_rule: Optional selection rule for pre-filtering objects
        use_color_only: Object extraction mode (passed to extract_connected_components)
        include_per_parent: Whether to also try per-parent orderings (default: True)

    Returns:
        Dict with:
            - best_ordering: OrderingStrategy instance
            - best_name: str
            - is_consistent: bool
            - all_results: Dict[str, Dict] with per-ordering results
            - has_hierarchy: bool (whether hierarchy was detected)
    """
    checker = ConsistencyChecker()

    # Check if we need selection filtering
    use_selection = selection_criterion is not None and selection_rule is not None
    if use_selection:
        from selection_module import apply_object_selection

    # Build examples list: (correspondences, grid_shape) for each training pair
    # We build two sets: flat examples (no hierarchy) and hierarchical examples
    flat_examples = []
    hierarchical_examples = []
    has_hierarchy = False

    for pair in puzzle.get('train', []):
        if 'output' not in pair:
            continue

        input_grid = np.array(pair['input'])
        output_grid = np.array(pair['output'])

        if use_selection:
            # Use labels-based workflow for selection filtering
            input_labels, input_colors, input_bboxes, _ = extract_connected_components(
                input_grid, use_color_only=use_color_only
            )
            output_labels, output_colors, output_bboxes, _ = extract_connected_components(
                output_grid, use_color_only=use_color_only
            )

            if len(input_colors) == 0:
                continue

            # Apply selection filtering
            _, input_labels, input_colors, input_bboxes = apply_object_selection(
                input_labels, input_colors, input_bboxes, input_grid,
                selection_criterion, selection_rule
            )

            # Convert to Object instances (flat, no hierarchy)
            input_objects = labels_to_objects(input_labels, input_colors, input_bboxes)
            output_objects = labels_to_objects(output_labels, output_colors, output_bboxes)
        else:
            # Use simple extraction (no selection filtering)
            input_objects = extract_objects_from_grid(input_grid)
            output_objects = extract_objects_from_grid(output_grid)

        if not input_objects:
            continue

        # Use canonical find_correspondences for shape-based matching
        matches, matchable_in, matchable_out = find_correspondences(
            input_grid, output_grid, input_objects, output_objects
        )
        correspondences = [
            Correspondence(matchable_in[in_idx], matchable_out[out_idx])
            for in_idx, out_idx, _ in matches
        ]

        grid_shape = (max(input_grid.shape[0], output_grid.shape[0]),
                     max(input_grid.shape[1], output_grid.shape[1]))

        flat_examples.append((correspondences, grid_shape))

        # Also extract with hierarchy for per-parent orderings
        if include_per_parent:
            input_objects_hier = extract_objects_from_grid(input_grid, build_hierarchy=True)
            output_objects_hier = extract_objects_from_grid(output_grid, build_hierarchy=True)

            # Check if any object has children
            if any(o.children for o in output_objects_hier):
                has_hierarchy = True

            matches_hier, matchable_in_hier, matchable_out_hier = find_correspondences(
                input_grid, output_grid, input_objects_hier, output_objects_hier
            )
            correspondences_hier = [
                Correspondence(matchable_in_hier[in_idx], matchable_out_hier[out_idx])
                for in_idx, out_idx, _ in matches_hier
            ]
            hierarchical_examples.append((correspondences_hier, grid_shape))

    if not flat_examples:
        # No valid examples - return default
        default_ordering = AdaptiveReadingOrder()
        return {
            'best_ordering': default_ordering,
            'best_name': default_ordering.name,
            'is_consistent': False,
            'all_results': {},
            'has_hierarchy': False
        }

    # Evaluate all orderings
    results = {}

    if verbose:
        print("\nScreening ordering strategies...")
        print("-" * 60)
        print("Global orderings:")

    # Evaluate global orderings on flat examples
    for ordering in ALL_ORDERINGS:
        consistency_result = checker.check_consistency(flat_examples, ordering)
        results[ordering.name] = {
            'consistent': consistency_result['consistent'],
            'framings': consistency_result['framings'],
            'num_framings': sum(len(f) for f in consistency_result.get('framings', []))
        }

        if verbose:
            status = "CONSISTENT" if consistency_result['consistent'] else "inconsistent"
            num_framings = results[ordering.name]['num_framings']
            print(f"  {ordering.name:25s}: {status} ({num_framings} framings)")

    # Evaluate per-parent orderings on hierarchical examples (if hierarchy exists)
    if include_per_parent and has_hierarchy and hierarchical_examples:
        if verbose:
            print("\nPer-parent orderings (hierarchy detected):")

        for ordering in PER_PARENT_ORDERINGS:
            consistency_result = checker.check_consistency(hierarchical_examples, ordering)
            results[ordering.name] = {
                'consistent': consistency_result['consistent'],
                'framings': consistency_result['framings'],
                'num_framings': sum(len(f) for f in consistency_result.get('framings', []))
            }

            if verbose:
                status = "CONSISTENT" if consistency_result['consistent'] else "inconsistent"
                num_framings = results[ordering.name]['num_framings']
                print(f"  {ordering.name:25s}: {status} ({num_framings} framings)")

    # Find best ordering: prefer consistent, then by number of framings
    # When scores tie, prefer reading-order strategies over arbitrary ones
    PREFERRED_ORDERINGS = [
        'adaptive_reading_order',
        'quadrant_order',
        'top_to_bottom',
        'left_to_right',
        'diagonal_tl_br',
        # Per-parent orderings have lower preference than global
        'per_parent(left_to_right)',
        'per_parent(top_to_bottom)',
    ]

    best_name = None
    best_score = -1
    best_preference = 999

    for name, result in results.items():
        # Score = (is_consistent * 1000) + num_framings
        score = (1000 if result['consistent'] else 0) + result['num_framings']

        # Get preference rank (lower is better)
        try:
            preference = PREFERRED_ORDERINGS.index(name)
        except ValueError:
            preference = 100  # Not in preferred list

        # Update best if: higher score, or same score with better preference
        if score > best_score or (score == best_score and preference < best_preference):
            best_score = score
            best_name = name
            best_preference = preference

    # Fallback
    if best_name is None:
        best_name = 'adaptive_reading_order'

    # Get the ordering instance
    best_ordering = None
    for ordering in ALL_ORDERINGS + PER_PARENT_ORDERINGS:
        if ordering.name == best_name:
            best_ordering = ordering
            break

    if verbose:
        print("-" * 60)
        print(f"Best ordering: {best_name}")
        print(f"Consistent: {results.get(best_name, {}).get('consistent', False)}")

    return {
        'best_ordering': best_ordering,
        'best_name': best_name,
        'is_consistent': results.get(best_name, {}).get('consistent', False),
        'all_results': results,
        'has_hierarchy': has_hierarchy
    }


def screen_hierarchy_strategies(
    puzzle: Dict,
    verbose: bool = False
) -> Dict:
    """
    Screen flat vs hierarchical object representations for a puzzle.

    Compares:
    1. Flat mode: Objects as independent peers (existing behavior)
    2. Hierarchy mode: Objects with containment-based parent/child relationships

    The hierarchy mode wins when:
    - Parent-child anchor relations have lower variance than peer relations
    - Scoped operations (within containers) produce more consistent results

    Args:
        puzzle: Puzzle dict with 'train' examples

    Returns:
        Dict with:
            - best_mode: 'flat' or 'hierarchy'
            - use_hierarchy: bool
            - flat_score: float (lower is better - based on anchor variance)
            - hierarchy_score: float
            - hierarchy_stats: dict with hierarchy info (if hierarchy detected)
            - parent_child_relations: list of best relations (if hierarchy)
    """
    from object_module import (
        extract_objects_from_grid,
        get_hierarchy_stats,
    )
    from anchoring_module import discover_parent_child_relation

    results = {
        'best_mode': 'flat',
        'use_hierarchy': False,
        'flat_score': float('inf'),
        'hierarchy_score': float('inf'),
        'hierarchy_stats': None,
        'parent_child_relations': None,
    }

    if verbose:
        print("\n" + "=" * 60)
        print("HIERARCHY SCREENING")
        print("=" * 60)

    # Collect data for both modes across all training examples
    flat_examples = []
    hierarchy_examples = []
    all_child_parent_pairs = []
    has_hierarchy = False

    for pair in puzzle.get('train', []):
        if 'output' not in pair:
            continue

        input_grid = np.array(pair['input'])
        output_grid = np.array(pair['output'])

        # === FLAT MODE ===
        input_objects_flat = extract_objects_from_grid(input_grid, build_hierarchy=False)
        output_objects_flat = extract_objects_from_grid(output_grid, build_hierarchy=False)
        flat_examples.append({
            'input_objects': input_objects_flat,
            'output_objects': output_objects_flat,
            'input_grid': input_grid,
            'output_grid': output_grid,
        })

        # === HIERARCHY MODE ===
        input_roots = extract_objects_from_grid(input_grid, build_hierarchy=True)
        output_roots = extract_objects_from_grid(output_grid, build_hierarchy=True)

        # Check if hierarchy was actually detected
        input_has_composite = any(r.is_composite for r in input_roots)
        output_has_composite = any(r.is_composite for r in output_roots)

        if input_has_composite or output_has_composite:
            has_hierarchy = True
            hierarchy_examples.append({
                'input_roots': input_roots,
                'output_roots': output_roots,
                'input_grid': input_grid,
                'output_grid': output_grid,
            })

            # Collect child-parent pairs from output
            for root in output_roots:
                for child in root.children:
                    all_child_parent_pairs.append((child, root))

    if verbose:
        print(f"\nFlat mode: {len(flat_examples)} examples")
        print(f"Hierarchy detected: {has_hierarchy}")
        if has_hierarchy:
            print(f"  - Examples with composites: {len(hierarchy_examples)}")
            print(f"  - Total child-parent pairs: {len(all_child_parent_pairs)}")

    # If no hierarchy detected, flat wins by default
    if not has_hierarchy:
        results['best_mode'] = 'flat'
        results['use_hierarchy'] = False
        if verbose:
            print("\nNo containment hierarchy detected - using flat mode")
        return results

    # === EVALUATE FLAT MODE ===
    # Score based on consistency of object-to-object relations
    # (We use a simple heuristic: count how many objects have consistent positions)
    flat_consistent_count = 0
    flat_total_count = 0
    for ex in flat_examples:
        # Simple check: do objects maintain relative positions?
        for in_obj in ex['input_objects']:
            for out_obj in ex['output_objects']:
                if in_obj.color == out_obj.color:
                    flat_total_count += 1
                    # Check if position delta is consistent (very simplified)
                    flat_consistent_count += 1  # Placeholder
    flat_score = 1.0  # Baseline score for flat mode

    # === EVALUATE HIERARCHY MODE ===
    hierarchy_score = float('inf')
    best_relations = []

    if all_child_parent_pairs and len(all_child_parent_pairs) >= 2:
        # Discover parent-child anchor relations
        relations = discover_parent_child_relation(all_child_parent_pairs, top_k=5)
        best_relations = relations

        if relations:
            # Use variance of best relation as score (lower is better)
            hierarchy_score = relations[0].variance

            if verbose:
                print(f"\nParent-child anchor analysis:")
                for i, rel in enumerate(relations[:3]):
                    print(f"  {i+1}. {rel.relation.describe()} (variance: {rel.variance:.4f})")

    # Get hierarchy stats from first example
    if hierarchy_examples:
        stats = get_hierarchy_stats(hierarchy_examples[0]['input_roots'])
        results['hierarchy_stats'] = stats

    results['flat_score'] = flat_score
    results['hierarchy_score'] = hierarchy_score
    results['parent_child_relations'] = best_relations

    # Decision: use hierarchy if it has low variance (< 1.0 is good)
    # and lower than a threshold indicating consistent parent-child positioning
    HIERARCHY_VARIANCE_THRESHOLD = 0.1

    if hierarchy_score < HIERARCHY_VARIANCE_THRESHOLD:
        results['best_mode'] = 'hierarchy'
        results['use_hierarchy'] = True
        if verbose:
            print(f"\n*** Hierarchy mode selected (variance {hierarchy_score:.4f} < {HIERARCHY_VARIANCE_THRESHOLD}) ***")
    else:
        results['best_mode'] = 'flat'
        results['use_hierarchy'] = False
        if verbose:
            print(f"\n*** Flat mode selected (hierarchy variance {hierarchy_score:.4f} too high) ***")

    return results


def visualize_ordering(grid: np.ndarray, objects: List[Object],
                       ordering: OrderingStrategy, ax: plt.Axes,
                       title: str = None) -> None:
    """
    Visualize objects on a grid with ordering annotations.

    Args:
        grid: (H, W) integer color grid
        objects: List of Object instances
        ordering: OrderingStrategy to apply
        ax: matplotlib axes to draw on
        title: Optional title for the subplot
    """
    H, W = grid.shape

    # Create colormap
    cmap = ListedColormap(ARC_COLORS)

    # Draw the grid
    ax.imshow(grid, cmap=cmap, vmin=0, vmax=9)

    # Apply ordering
    ordered_objects = ordering.order(objects)

    # Draw bounding boxes and order numbers
    for order_idx, obj in enumerate(ordered_objects):
        # Draw bounding box
        rect = mpatches.Rectangle(
            (obj.col - 0.5, obj.row - 0.5),
            obj.width, obj.height,
            linewidth=2, edgecolor='white', facecolor='none', linestyle='--'
        )
        ax.add_patch(rect)

        # Draw order number at center of object
        center_row, center_col = obj.center
        ax.text(
            center_col, center_row, str(order_idx),
            ha='center', va='center',
            fontsize=14, fontweight='bold',
            color='white',
            bbox=dict(boxstyle='circle', facecolor='black', alpha=0.7)
        )

    # Set title
    if title:
        ax.set_title(title, fontsize=10)

    # Clean up axes
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(-0.5, W - 0.5)
    ax.set_ylim(H - 0.5, -0.5)


def visualize_puzzle_ordering(puzzle_id: str, ordering_name: str = "auto"):
    """
    Visualize object ordering for a specific puzzle.

    Args:
        puzzle_id: The ARC puzzle ID (e.g., "1990f7a8") or synthetic puzzle ID (e.g., "syn_dual_fill")
        ordering_name: Name of ordering strategy to use, or "auto" to find best
    """
    # Load puzzle (supports synthetic puzzles with syn_ prefix)
    print(f"Loading puzzle {puzzle_id}...")
    try:
        puzzle = load_puzzle(puzzle_id)
    except ValueError as e:
        print(f"Error: {e}")
        return

    # Get the ordering strategy
    if ordering_name == "auto":
        # Auto-detect best ordering using unified screening
        screen_result = find_best_ordering(puzzle, verbose=True)
        ordering = screen_result.global_ordering
        ordering_name = screen_result.best_ordering_name

        print(f"\nBest ordering configuration:")
        print(f"  Mode: {screen_result.best_mode}")
        print(f"  Strategy: {ordering_name}")
        print(f"  Consistent: {screen_result.is_consistent}")
        print(f"  Has hierarchy: {screen_result.has_hierarchy}")

        if screen_result.best_mode == 'per_parent' and screen_result.per_parent_result:
            # For per-parent mode, use the child ordering for visualization
            ordering = screen_result.per_parent_result.child_ordering
            print(f"\nPer-parent ordering details:")
            print(f"  Child ordering: {ordering.name}")
            for idx, offset in sorted(screen_result.per_parent_result.learned_offsets.items()):
                print(f"    Index {idx}: parent + {offset}")
        print()
    else:
        ordering = None
        for strat in ALL_ORDERINGS:
            if strat.name == ordering_name:
                ordering = strat
                break

        if ordering is None:
            print(f"Error: Ordering strategy '{ordering_name}' not found")
            print(f"Available strategies: {[s.name for s in ALL_ORDERINGS]}")
            print("Use 'auto' to automatically find the best ordering")
            return

    # Count total grids: training pairs + test pair(s)
    num_train = len(puzzle["train"])
    num_test = len(puzzle["test"])
    total_grids = num_train + num_test

    # Create figure
    fig, axes = plt.subplots(1, total_grids, figsize=(4 * total_grids, 5))
    if total_grids == 1:
        axes = [axes]

    fig.suptitle(f"Puzzle {puzzle_id} - Object Ordering ({ordering.name})", fontsize=14, fontweight='bold')

    # Process training pairs
    for i, pair in enumerate(puzzle["train"]):
        input_grid = np.array(pair["input"])
        objects = extract_objects_from_grid(input_grid)

        print(f"\nTraining pair {i+1}: {len(objects)} objects detected")
        ordered = ordering.order(objects)
        for idx, obj in enumerate(ordered):
            print(f"  [{idx}] Object id={obj.id}, pos=({obj.row},{obj.col}), "
                  f"size={obj.height}x{obj.width}, color={obj.color}, center={obj.center}")

        visualize_ordering(input_grid, objects, ordering, axes[i],
                          title=f"Train {i+1} Input")

    # Process test pair(s)
    for i, pair in enumerate(puzzle["test"]):
        input_grid = np.array(pair["input"])
        objects = extract_objects_from_grid(input_grid)

        print(f"\nTest pair {i+1}: {len(objects)} objects detected")
        ordered = ordering.order(objects)
        for idx, obj in enumerate(ordered):
            print(f"  [{idx}] Object id={obj.id}, pos=({obj.row},{obj.col}), "
                  f"size={obj.height}x{obj.width}, color={obj.color}, center={obj.center}")

        ax_idx = num_train + i
        visualize_ordering(input_grid, objects, ordering, axes[ax_idx],
                          title=f"Test {i+1} Input")

    plt.tight_layout()
    plt.show()

    # Also show row structure for adaptive ordering
    if isinstance(ordering, AdaptiveReadingOrder):
        print(f"\n{'='*60}")
        print("Adaptive Reading Order - Row Structure Details")
        print('='*60)

        for i, pair in enumerate(puzzle["train"]):
            input_grid = np.array(pair["input"])
            objects = extract_objects_from_grid(input_grid)
            row_structure = ordering.get_row_structure(objects)
            print(f"\nTraining pair {i+1} row structure: {row_structure}")

        for i, pair in enumerate(puzzle["test"]):
            input_grid = np.array(pair["input"])
            objects = extract_objects_from_grid(input_grid)
            row_structure = ordering.get_row_structure(objects)
            print(f"\nTest pair {i+1} row structure: {row_structure}")


def compare_orderings(puzzle_id: str):
    """
    Compare all ordering strategies for a specific puzzle.

    Args:
        puzzle_id: The ARC puzzle ID (e.g., "1990f7a8") or synthetic puzzle ID (e.g., "syn_dual_fill")
    """
    # Load puzzle (supports synthetic puzzles with syn_ prefix)
    print(f"Loading puzzle {puzzle_id}...")
    try:
        puzzle = load_puzzle(puzzle_id)
    except ValueError as e:
        print(f"Error: {e}")
        return

    # Select key ordering strategies to compare
    key_orderings = [
        "left_to_right",
        "top_to_bottom",
        "adaptive_reading_order",
        "quadrant_order",
    ]

    orderings = [s for s in ALL_ORDERINGS if s.name in key_orderings]

    # Get first training input for comparison
    input_grid = np.array(puzzle["train"][0]["input"])
    objects = extract_objects_from_grid(input_grid)

    if len(objects) == 0:
        print("No objects found in first training input")
        return

    # Create comparison figure
    num_orderings = len(orderings)
    fig, axes = plt.subplots(1, num_orderings, figsize=(4 * num_orderings, 5))
    if num_orderings == 1:
        axes = [axes]

    fig.suptitle(f"Puzzle {puzzle_id} - Ordering Strategy Comparison (Train 1 Input)",
                 fontsize=14, fontweight='bold')

    for i, ordering in enumerate(orderings):
        visualize_ordering(input_grid, objects, ordering, axes[i],
                          title=ordering.name)

        print(f"\n{ordering.name}:")
        ordered = ordering.order(objects)
        print(f"  Order: {[obj.id for obj in ordered]}")

    plt.tight_layout()
    plt.show()


# =============================================================================
# Demo / Test Functions
# =============================================================================

def create_demo_objects() -> Tuple[List[Object], List[Object]]:
    """Create demo input/output objects for testing."""
    # Scenario: objects arranged in a line, output shifts them relative to each other
    input_objects = [
        Object(id=0, row=2, col=1, height=2, width=2, color=1),   # red, left
        Object(id=1, row=2, col=5, height=2, width=2, color=2),   # blue, middle  
        Object(id=2, row=2, col=9, height=2, width=2, color=3),   # green, right
    ]
    
    # Output: objects stack vertically, each 3 below the previous
    output_objects = [
        Object(id=0, row=0, col=5, height=2, width=2, color=1),   # red at top
        Object(id=1, row=3, col=5, height=2, width=2, color=2),   # blue 3 below red
        Object(id=2, row=6, col=5, height=2, width=2, color=3),   # green 3 below blue
    ]
    
    return input_objects, output_objects


def create_correspondences(input_objs: List[Object], 
                          output_objs: List[Object]) -> List[Correspondence]:
    """Create correspondences (assumes same IDs map to each other)."""
    output_by_id = {o.id: o for o in output_objs}
    return [Correspondence(inp, output_by_id[inp.id]) for inp in input_objs]


def demo():
    """Run a demonstration of the ordering module."""
    print("=" * 70)
    print("ORDERING MODULE DEMO")
    print("=" * 70)
    
    # Create demo data
    input_objs, output_objs = create_demo_objects()
    correspondences = create_correspondences(input_objs, output_objs)
    grid_shape = (10, 15)
    
    print("\nScenario: 3 objects arranged horizontally in input")
    print("Output: objects stack vertically, each 3 pixels below previous")
    print("\nInput objects:")
    for obj in input_objs:
        print(f"  ID={obj.id}, pos=({obj.row},{obj.col}), color={obj.color}")
    
    print("\nOutput objects:")
    for obj in output_objs:
        print(f"  ID={obj.id}, pos=({obj.row},{obj.col}), color={obj.color}")
    
    # Evaluate orderings
    evaluator = OrderingEvaluator()
    print("\n" + "-" * 70)
    print("EVALUATING ORDERINGS")
    print("-" * 70)
    
    results = evaluator.evaluate_all_orderings(correspondences, grid_shape)
    
    for eval_result in results:
        status = "✓ ALL CORRECT" if eval_result.all_correct else f"✗ {eval_result.success_count}/{len(eval_result.best_framings)}"
        print(f"\n{eval_result.ordering_name}: {status}")
        print(f"  Object order: {eval_result.object_order}")
        
        for i, best in enumerate(eval_result.best_framings):
            if best:
                print(f"  Position {i}: {best.framing_name} → {best.predicted_pos} (correct)")
            else:
                print(f"  Position {i}: No working framing found")
    
    # Show best ordering details
    best = results[0]
    print("\n" + "-" * 70)
    print(f"BEST ORDERING: {best.ordering_name}")
    print("-" * 70)
    
    print("\nDetailed framing analysis:")
    for i, (obj_results, best_framing) in enumerate(zip(best.framing_results, best.best_framings)):
        print(f"\nObject {best.object_order[i]} (position {i}):")
        for result in obj_results:
            mark = "✓" if result.is_correct else "✗"
            print(f"  {mark} {result.framing_name}: predicted={result.predicted_pos}, actual={result.actual_pos}")


def test_ordering_matters():
    """
    Test case where ordering ACTUALLY matters.
    
    Scenario: 
    - Object A (anchor) always stays at its input position (delta=0,0)
    - Object B always goes to (+1 row, +2 col) relative to OUTPUT position of A
    
    With left-to-right ordering (A first):
    - A: delta works (0,0)
    - B: object_relative(ref=0) works (+1, +2)
    
    With right-to-left ordering (B first):  
    - B: can't reference A (comes later!)
    - The rule is inexpressible with available framings
    """
    print("=" * 70)
    print("TEST: ORDERING ACTUALLY MATTERS")
    print("=" * 70)
    
    print("\nScenario:")
    print("  - Object A (left) stays at its input position (delta = 0,0)")
    print("  - Object B (right) goes to A's output position + (1 row, 2 cols)")
    print("\nWith left-to-right: A processed first, B can reference A → works!")
    print("With right-to-left: B processed first, can't reference A yet → fails!\n")
    
    # Example 1
    ex1_input = [
        Object(id=0, row=2, col=1, height=1, width=1, color=1),  # A at (2,1)
        Object(id=1, row=2, col=6, height=1, width=1, color=2),  # B at (2,6)
    ]
    ex1_output = [
        Object(id=0, row=2, col=1, height=1, width=1, color=1),  # A stays at (2,1)
        Object(id=1, row=3, col=3, height=1, width=1, color=2),  # B = A + (1,2) = (3,3)
    ]
    
    # Example 2: Different starting positions, same rules
    ex2_input = [
        Object(id=0, row=5, col=3, height=1, width=1, color=1),  # A at (5,3)
        Object(id=1, row=5, col=8, height=1, width=1, color=2),  # B at (5,8)
    ]
    ex2_output = [
        Object(id=0, row=5, col=3, height=1, width=1, color=1),  # A stays at (5,3)
        Object(id=1, row=6, col=5, height=1, width=1, color=2),  # B = A + (1,2) = (6,5)
    ]
    
    # Example 3
    ex3_input = [
        Object(id=0, row=0, col=0, height=1, width=1, color=1),  # A at (0,0)
        Object(id=1, row=0, col=4, height=1, width=1, color=2),  # B at (0,4)
    ]
    ex3_output = [
        Object(id=0, row=0, col=0, height=1, width=1, color=1),  # A stays at (0,0)
        Object(id=1, row=1, col=2, height=1, width=1, color=2),  # B = A + (1,2) = (1,2)
    ]
    
    examples = [
        (create_correspondences(ex1_input, ex1_output), (10, 10)),
        (create_correspondences(ex2_input, ex2_output), (10, 10)),
        (create_correspondences(ex3_input, ex3_output), (10, 10)),
    ]
    
    checker = ConsistencyChecker()
    
    # Left-to-right: A comes first, B can reference A
    print("LEFT-TO-RIGHT ordering (A at position 0, B at position 1):")
    ltr_result = checker.check_consistency(examples, LeftToRight())
    print(f"  Fully consistent: {ltr_result['consistent']}")
    print(f"  Framings with same params across all examples:")
    for i, cf in enumerate(ltr_result['framings']):
        obj_name = "A" if i == 0 else "B"
        if cf:
            print(f"    Position {i} ({obj_name}): {cf}")
        else:
            print(f"    Position {i} ({obj_name}): NONE - no consistent framing!")
    
    # Right-to-left: B comes first, can't reference A
    print("\nRIGHT-TO-LEFT ordering (B at position 0, A at position 1):")
    rtl_result = checker.check_consistency(examples, RightToLeft())
    print(f"  Fully consistent: {rtl_result['consistent']}")
    print(f"  Framings with same params across all examples:")
    for i, cf in enumerate(rtl_result['framings']):
        obj_name = "B" if i == 0 else "A"
        if cf:
            print(f"    Position {i} ({obj_name}): {cf}")
        else:
            print(f"    Position {i} ({obj_name}): NONE - no consistent framing!")
    
    # Summary
    print("\n" + "=" * 50)
    print("CONCLUSION:")
    print("=" * 50)
    
    if ltr_result['consistent'] and not rtl_result['consistent']:
        print("✓ LEFT-TO-RIGHT enables consistent rules")
        print("✗ RIGHT-TO-LEFT cannot express the dependency")
        print("\nThis proves: ordering determines which rules are expressible!")
    elif ltr_result['consistent'] and rtl_result['consistent']:
        print("Both orderings work (both have consistent framings)")
    else:
        print(f"LTR consistent: {ltr_result['consistent']}")
        print(f"RTL consistent: {rtl_result['consistent']}")
    
    print()


def test():
    """Run tests on the ordering module."""
    # First run the important test
    test_ordering_matters()
    
    print("\n")
    print("=" * 70)
    print("ADDITIONAL ORDERING MODULE TESTS")
    print("=" * 70)
    
    # Test 1: Object relative framing depends on ordering
    print("\nTest 1: Basic object-relative framing")
    
    # Two objects: one should be positioned relative to the other
    input_objs = [
        Object(id=0, row=0, col=0, height=1, width=1, color=1),
        Object(id=1, row=0, col=5, height=1, width=1, color=2),
    ]
    output_objs = [
        Object(id=0, row=2, col=2, height=1, width=1, color=1),
        Object(id=1, row=2, col=4, height=1, width=1, color=2),  # 2 right of obj 0
    ]
    
    correspondences = create_correspondences(input_objs, output_objs)
    evaluator = OrderingEvaluator()
    
    # With left-to-right: obj0 first, obj1 can reference obj0
    ltr_result = evaluator.evaluate_ordering(correspondences, LeftToRight(), (10, 10))
    print(f"  Left-to-right ordering: success={ltr_result.success_rate:.0%}")
    
    # With right-to-left: obj1 first, obj0 second - different dependencies
    rtl_result = evaluator.evaluate_ordering(correspondences, RightToLeft(), (10, 10))
    print(f"  Right-to-left ordering: success={rtl_result.success_rate:.0%}")
    
    # Test 2: Consistency across examples
    print("\nTest 2: Cross-example consistency")
    
    examples = []
    
    # Example 1: same pattern as above
    examples.append((correspondences, (10, 10)))
    
    # Example 2: same relative positioning, different absolute positions
    input_objs2 = [
        Object(id=0, row=1, col=1, height=1, width=1, color=1),
        Object(id=1, row=1, col=6, height=1, width=1, color=2),
    ]
    output_objs2 = [
        Object(id=0, row=5, col=3, height=1, width=1, color=1),
        Object(id=1, row=5, col=5, height=1, width=1, color=2),  # still 2 right of obj 0
    ]
    examples.append((create_correspondences(input_objs2, output_objs2), (10, 10)))
    
    checker = ConsistencyChecker()
    
    ltr_consistency = checker.check_consistency(examples, LeftToRight())
    print(f"  Left-to-right consistent: {ltr_consistency['consistent']}")
    print(f"  Consistent framings per position: {ltr_consistency['framings']}")
    
    print("\n" + "=" * 70)
    print("TESTS COMPLETE")
    print("=" * 70)


def test_adaptive_reading_order():
    """
    Test AdaptiveReadingOrder with various object configurations.
    
    Demonstrates that it correctly identifies row structure and orders
    objects in reading order regardless of how many objects there are.
    """
    print("=" * 70)
    print("TEST: ADAPTIVE READING ORDER")
    print("=" * 70)
    
    strategy = AdaptiveReadingOrder()
    
    # Helper to visualize object positions
    def show_objects(objects: List[Object], grid_h: int = 20, grid_w: int = 30):
        grid = [['·' for _ in range(grid_w)] for _ in range(grid_h)]
        for obj in objects:
            cr, cc = int(obj.center[0]), int(obj.center[1])
            if 0 <= cr < grid_h and 0 <= cc < grid_w:
                grid[cr][cc] = str(obj.id)
        return '\n'.join(''.join(row) for row in grid)
    
    # Test 1: 4 objects in quadrant layout (like the puzzle image)
    print("\n" + "-" * 50)
    print("Test 1: 4 objects in quadrant layout")
    print("-" * 50)
    objects_4quad = [
        Object(id=0, row=2, col=3, height=3, width=3, color=1),   # top-left
        Object(id=1, row=4, col=15, height=4, width=4, color=2),  # top-right
        Object(id=2, row=12, col=5, height=2, width=2, color=3),  # bottom-left
        Object(id=3, row=13, col=18, height=2, width=3, color=4), # bottom-right
    ]
    print("\nObject positions (center points on grid):")
    print(show_objects(objects_4quad))
    
    ordered = strategy.order(objects_4quad)
    row_structure = strategy.get_row_structure(objects_4quad)
    
    print(f"\nDetected row structure: {row_structure}")
    print(f"Final order: {[o.id for o in ordered]}")
    print(f"Expected: [0, 1, 2, 3] (TL, TR, BL, BR)")
    assert [o.id for o in ordered] == [0, 1, 2, 3], "4-quadrant test failed!"
    print("✓ Passed")
    
    # Test 2: 3 objects in L-shape (2 top, 1 bottom-left)
    print("\n" + "-" * 50)
    print("Test 2: 3 objects in L-shape")
    print("-" * 50)
    objects_L = [
        Object(id=0, row=2, col=3, height=2, width=2, color=1),   # top-left
        Object(id=1, row=3, col=15, height=2, width=2, color=2),  # top-right
        Object(id=2, row=14, col=4, height=2, width=2, color=3),  # bottom-left
    ]
    print("\nObject positions:")
    print(show_objects(objects_L))
    
    ordered = strategy.order(objects_L)
    row_structure = strategy.get_row_structure(objects_L)
    
    print(f"\nDetected row structure: {row_structure}")
    print(f"Final order: {[o.id for o in ordered]}")
    print(f"Expected: [0, 1, 2] (top-left, top-right, then bottom)")
    assert [o.id for o in ordered] == [0, 1, 2], "L-shape test failed!"
    print("✓ Passed")
    
    # Test 3: 5 objects - 2 rows, top has 2, bottom has 3
    print("\n" + "-" * 50)
    print("Test 3: 5 objects (2 top, 3 bottom)")
    print("-" * 50)
    objects_5 = [
        Object(id=0, row=2, col=5, height=2, width=2, color=1),   # top row, left
        Object(id=1, row=3, col=20, height=2, width=2, color=2),  # top row, right
        Object(id=2, row=14, col=3, height=2, width=2, color=3),  # bottom row, left
        Object(id=3, row=15, col=12, height=2, width=2, color=4), # bottom row, middle
        Object(id=4, row=14, col=22, height=2, width=2, color=5), # bottom row, right
    ]
    print("\nObject positions:")
    print(show_objects(objects_5))
    
    ordered = strategy.order(objects_5)
    row_structure = strategy.get_row_structure(objects_5)
    
    print(f"\nDetected row structure: {row_structure}")
    print(f"Final order: {[o.id for o in ordered]}")
    print(f"Expected: [0, 1, 2, 3, 4]")
    assert [o.id for o in ordered] == [0, 1, 2, 3, 4], "5-object test failed!"
    print("✓ Passed")
    
    # Test 4: 6 objects in 3x2 grid
    print("\n" + "-" * 50)
    print("Test 4: 6 objects in 3 rows × 2 cols")
    print("-" * 50)
    objects_6 = [
        Object(id=0, row=2, col=5, height=2, width=2, color=1),   # row 1, left
        Object(id=1, row=2, col=20, height=2, width=2, color=2),  # row 1, right
        Object(id=2, row=8, col=4, height=2, width=2, color=3),   # row 2, left
        Object(id=3, row=9, col=21, height=2, width=2, color=4),  # row 2, right
        Object(id=4, row=15, col=5, height=2, width=2, color=5),  # row 3, left
        Object(id=5, row=16, col=19, height=2, width=2, color=6), # row 3, right
    ]
    print("\nObject positions:")
    print(show_objects(objects_6))
    
    ordered = strategy.order(objects_6)
    row_structure = strategy.get_row_structure(objects_6)
    
    print(f"\nDetected row structure: {row_structure}")
    print(f"Final order: {[o.id for o in ordered]}")
    print(f"Expected: [0, 1, 2, 3, 4, 5]")
    assert [o.id for o in ordered] == [0, 1, 2, 3, 4, 5], "6-object (3x2) test failed!"
    print("✓ Passed")
    
    # Test 5: 2 objects (simple case)
    print("\n" + "-" * 50)
    print("Test 5: 2 objects (diagonal)")
    print("-" * 50)
    objects_2 = [
        Object(id=0, row=3, col=5, height=2, width=2, color=1),   # top-left
        Object(id=1, row=12, col=18, height=2, width=2, color=2), # bottom-right
    ]
    print("\nObject positions:")
    print(show_objects(objects_2))
    
    ordered = strategy.order(objects_2)
    row_structure = strategy.get_row_structure(objects_2)
    
    print(f"\nDetected row structure: {row_structure}")
    print(f"Final order: {[o.id for o in ordered]}")
    print(f"Expected: [0, 1]")
    assert [o.id for o in ordered] == [0, 1], "2-object test failed!"
    print("✓ Passed")
    
    # Test 6: All objects on same row
    print("\n" + "-" * 50)
    print("Test 6: 4 objects all on same row")
    print("-" * 50)
    objects_row = [
        Object(id=0, row=8, col=2, height=2, width=2, color=1),
        Object(id=1, row=9, col=8, height=2, width=2, color=2),
        Object(id=2, row=8, col=15, height=2, width=2, color=3),
        Object(id=3, row=9, col=22, height=2, width=2, color=4),
    ]
    print("\nObject positions:")
    print(show_objects(objects_row))
    
    ordered = strategy.order(objects_row)
    row_structure = strategy.get_row_structure(objects_row)
    
    print(f"\nDetected row structure: {row_structure}")
    print(f"Final order: {[o.id for o in ordered]}")
    print(f"Expected: [0, 1, 2, 3] (all same row, sorted by column)")
    assert [o.id for o in ordered] == [0, 1, 2, 3], "Same-row test failed!"
    print("✓ Passed")
    
    # Test 7: Compare strategies on tricky layout
    print("\n" + "-" * 50)
    print("Test 7: Comparing strategies on overlapping positions")
    print("-" * 50)
    print("Objects where vertical positions overlap between quadrants:")
    
    # Tricky case: bottom-left object is higher than top-right
    objects_tricky = [
        Object(id=0, row=5, col=3, height=3, width=3, color=1),   # top-left (center ~6.5)
        Object(id=1, row=8, col=20, height=4, width=4, color=2),  # top-right (center ~10)
        Object(id=2, row=7, col=4, height=2, width=2, color=3),   # bottom-left (center ~8) - ABOVE top-right!
        Object(id=3, row=15, col=18, height=2, width=3, color=4), # bottom-right
    ]
    print("\nObject positions:")
    print(show_objects(objects_tricky))
    print("\nObject centers:")
    for obj in objects_tricky:
        print(f"  {obj.id}: center=({obj.center[0]:.1f}, {obj.center[1]:.1f})")
    
    print("\nNote: Object 2 (bottom-left area) is vertically ABOVE Object 1 (top-right area)")
    
    # TopToBottom ordering
    ttb = TopToBottom()
    ttb_ordered = ttb.order(objects_tricky)
    print(f"\n1. TopToBottom (pure row sort): {[o.id for o in ttb_ordered]}")
    print("   → Orders by vertical position only")
    
    # AdaptiveReadingOrder
    adaptive = AdaptiveReadingOrder()
    adaptive_ordered = adaptive.order(objects_tricky)
    row_structure = adaptive.get_row_structure(objects_tricky)
    print(f"\n2. AdaptiveReadingOrder: {[o.id for o in adaptive_ordered]}")
    print(f"   → Detected rows: {row_structure}")
    print("   → Groups by vertical proximity (gap detection)")
    
    # QuadrantOrder  
    quadrant = QuadrantOrder()
    quadrant_ordered = quadrant.order(objects_tricky)
    quad_structure = quadrant.get_quadrant_structure(objects_tricky)
    print(f"\n3. QuadrantOrder: {[o.id for o in quadrant_ordered]}")
    print(f"   → Quadrants: {quad_structure}")
    print("   → Groups by spatial region (median-based)")
    
    print("\n" + "=" * 70)
    print("ADAPTIVE READING ORDER TESTS COMPLETE")
    print("=" * 70)


def test_per_parent_ordering():
    """
    Test PerParentOrdering with hierarchical objects.

    Demonstrates ordering children within each parent consistently,
    enabling rules like "the 1st child in each parent goes to position X".
    """
    print("=" * 70)
    print("TEST: PER-PARENT ORDERING")
    print("=" * 70)

    # Create hierarchical objects: 2 parents, each with 3 children
    # Parent A at (0, 0) with children at various positions
    parent_a = Object(id=100, row=0, col=0, height=10, width=15, color=5)
    child_a1 = Object(id=1, row=2, col=2, height=2, width=2, color=1)
    child_a2 = Object(id=2, row=2, col=8, height=2, width=2, color=2)
    child_a3 = Object(id=3, row=6, col=5, height=2, width=2, color=3)

    # Set parent references
    child_a1.parent = parent_a
    child_a2.parent = parent_a
    child_a3.parent = parent_a
    parent_a.children = [child_a1, child_a2, child_a3]

    # Parent B at (12, 0) with children at various positions
    parent_b = Object(id=200, row=12, col=0, height=10, width=15, color=5)
    child_b1 = Object(id=4, row=14, col=3, height=2, width=2, color=1)
    child_b2 = Object(id=5, row=14, col=10, height=2, width=2, color=2)
    child_b3 = Object(id=6, row=18, col=6, height=2, width=2, color=3)

    child_b1.parent = parent_b
    child_b2.parent = parent_b
    child_b3.parent = parent_b
    parent_b.children = [child_b1, child_b2, child_b3]

    all_children = [child_a1, child_a2, child_a3, child_b1, child_b2, child_b3]

    print("\nScenario: 2 parent objects, each with 3 children")
    print("\nParent A children:")
    for c in parent_a.children:
        print(f"  id={c.id}, pos=({c.row},{c.col}), color={c.color}")
    print("\nParent B children:")
    for c in parent_b.children:
        print(f"  id={c.id}, pos=({c.row},{c.col}), color={c.color}")

    # Test different orderings
    print("\n" + "-" * 50)
    print("Testing per-parent orderings:")
    print("-" * 50)

    for base_ordering in [LeftToRight(), TopToBottom(), ByColor()]:
        per_parent = PerParentOrdering(base_ordering)

        print(f"\n{per_parent.name}:")

        # Order children within each parent
        ordered_by_parent = per_parent.order_children_by_parent(all_children)
        for parent_id, ordered_children in ordered_by_parent.items():
            parent_name = "A" if parent_id == 100 else "B"
            child_ids = [c.id for c in ordered_children]
            print(f"  Parent {parent_name}: {child_ids}")

        # Group by child index
        by_index = per_parent.group_by_child_index(all_children)
        print("  Grouped by index:")
        for idx, children in sorted(by_index.items()):
            child_ids = [c.id for c in children]
            print(f"    Index {idx}: {child_ids}")

    # Test variance computation
    print("\n" + "-" * 50)
    print("Testing parent-relative offset consistency:")
    print("-" * 50)

    # Children have consistent offsets from their parents
    # child_a1 at (2,2) in parent at (0,0) -> offset (2,2)
    # child_b1 at (14,3) in parent at (12,0) -> offset (2,3)

    for base_ordering in [LeftToRight(), TopToBottom()]:
        per_parent = PerParentOrdering(base_ordering)
        by_index = per_parent.group_by_child_index(all_children)

        print(f"\n{per_parent.name}:")
        for idx, children in sorted(by_index.items()):
            offsets = [(c.row - c.parent.row, c.col - c.parent.col) for c in children]
            print(f"  Index {idx}: offsets = {offsets}")

    print("\n" + "=" * 70)
    print("PER-PARENT ORDERING TESTS COMPLETE")
    print("=" * 70)


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="ARC Puzzle Object Ordering Module",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python ordering_module.py --puzzle-id 1990f7a8            # Auto-find best ordering
    python ordering_module.py --puzzle-id 1990f7a8 --ordering adaptive_reading_order
    python ordering_module.py --puzzle-id 1990f7a8 --compare  # Compare all strategies
    python ordering_module.py --demo                          # Run demo
    python ordering_module.py --test                          # Run tests
    python ordering_module.py --adaptive                      # Test adaptive reading order

Available ordering strategies:
    auto (default) - automatically find best ordering based on framing consistency
    left_to_right, right_to_left, top_to_bottom, bottom_to_top,
    diagonal_tl_br, diagonal_tr_bl, largest_first, smallest_first,
    by_color, adaptive_reading_order, quadrant_order
        """
    )

    parser.add_argument("--demo", action="store_true", help="Run demonstration")
    parser.add_argument("--test", action="store_true", help="Run tests")
    parser.add_argument("--adaptive", action="store_true", help="Test adaptive reading order")
    parser.add_argument("--per-parent", action="store_true", help="Test per-parent ordering")
    parser.add_argument("--puzzle-id", type=str, help="ARC puzzle ID to visualize (e.g., 1990f7a8)")
    parser.add_argument("--ordering", type=str, default="auto",
                        help="Ordering strategy to use (default: auto)")
    parser.add_argument("--compare", action="store_true",
                        help="Compare multiple ordering strategies for the puzzle")
    parser.add_argument("--screen-per-parent", action="store_true",
                        help="Screen per-parent orderings only (requires --puzzle-id)")
    parser.add_argument("--screen-all", action="store_true",
                        help="Unified screening: global + per-parent orderings (requires --puzzle-id)")

    args = parser.parse_args()

    if args.puzzle_id:
        # Load single puzzle (supports synthetic puzzles with syn_ prefix)
        try:
            puzzle = load_puzzle(args.puzzle_id)
        except ValueError as e:
            print(f"Error: {e}")
            exit(1)

        if args.screen_all:
            # Unified ordering screening (the canonical approach)
            result = find_best_ordering(puzzle, verbose=True)
            print("\n" + "=" * 60)
            print("SUMMARY")
            print("=" * 60)
            print(f"Best configuration: {result.describe()}")
            print(f"Has hierarchy: {result.has_hierarchy}")
        elif args.screen_per_parent:
            # Screen per-parent orderings only
            result = screen_per_parent_orderings_for_puzzle(puzzle, verbose=True)
            if result is None:
                print("\nNo hierarchical structure found for per-parent ordering")
        elif args.compare:
            compare_orderings(args.puzzle_id)
        else:
            visualize_puzzle_ordering(args.puzzle_id, args.ordering)
    elif args.demo:
        demo()
    elif args.test:
        test()
    elif args.adaptive:
        test_adaptive_reading_order()
    elif args.per_parent:
        test_per_parent_ordering()
    else:
        # Default: show help
        parser.print_help()