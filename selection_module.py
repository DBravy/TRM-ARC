#!/usr/bin/env python3
"""
Selection Module for ARC Object Processing

Provides ranking-based object selection functionality:
- Ranking criteria (largest, leftmost, etc.)
- Selection rules (top_1, top_2, all, etc.)
- Selection screening to find best predictor for object selection

Usage:
    from selection_module import (
        RANKING_CRITERIA, SELECTION_RULES,
        ObjectPropertiesForRanking,
        compute_object_properties_for_ranking,
        compute_ranks, apply_selection_rule,
        compute_selection_mask,   # Returns mask only (keeps all objects)
        apply_object_selection,   # Returns mask + filtered objects
        SelectionSample, SelectionScreener
    )

    # Command-line usage:
    python selection_module.py --puzzle-id 6fa7a44f
"""

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
from scipy import ndimage

from object_module import extract_connected_components, labels_to_objects, SegmentationMode, SegmentationStrategy
from correspondence_module import find_correspondences, CorrespondenceMode, DEFAULT_MARGIN
from puzzle_loader import load_all_puzzles as _load_puzzles


# =============================================================================
# Ranking Criteria and Selection Rules
# =============================================================================

# Ranking criteria for object selection
# Each defines how to rank/sort objects for selection
RANKING_CRITERIA = [
    # Size-based
    'largest',
    'smallest',
    'tallest',
    'shortest_height',
    'widest',
    'narrowest',

    # Position-based (single axis)
    'leftmost',
    'rightmost',
    'topmost',
    'bottommost',

    # Position-based (distance to center)
    'closest_to_center',
    'farthest_from_center',

    # Position-based (distance to corners)
    'closest_to_tl',
    'closest_to_tr',
    'closest_to_bl',
    'closest_to_br',
    'farthest_from_tl',
    'farthest_from_tr',
    'farthest_from_bl',
    'farthest_from_br',

    # Position-based (distance to edges)
    'closest_to_top',
    'closest_to_bottom',
    'closest_to_left',
    'closest_to_right',
    'farthest_from_top',
    'farthest_from_bottom',
    'farthest_from_left',
    'farthest_from_right',

    # Color-based
    'most_common_color',
    'least_common_color',

    # Relational
    'most_neighbors',
    'most_isolated',
]

# Selection rules - how to use ranking to select objects
SELECTION_RULES = [
    'top_1',      # Keep only rank 0
    'top_2',      # Keep rank 0 and 1
    'top_3',      # Keep rank 0, 1, 2
    'bottom_1',   # Keep only highest rank (last place)
    'bottom_2',   # Keep last two
    'all',        # Keep all (baseline)
    'all_best',   # Keep ALL objects with the best (lowest) rank
]

# Rule preference order for tie-breaking: prefer more permissive rules
# Higher value = more preferred when accuracy and num_selected are tied
RULE_PREFERENCE = {
    'all_best': 6,   # Most preferred - semantically clearest "all tied for best"
    'all': 5,        # All objects
    'top_3': 4,
    'top_2': 3,
    'top_1': 2,
    'bottom_2': 1,
    'bottom_1': 0,
}


# =============================================================================
# Object Properties for Ranking
# =============================================================================

@dataclass
class ObjectPropertiesForRanking:
    """Properties of objects in a grid, used for ranking-based selection."""
    centroids: np.ndarray      # (N, 2) row, col in pixels
    bboxes: np.ndarray         # (N, 4) min_row, min_col, max_row, max_col
    areas: np.ndarray          # (N,) pixel count
    heights: np.ndarray        # (N,) bbox height
    widths: np.ndarray         # (N,) bbox width
    colors: np.ndarray         # (N,) color index
    valid: np.ndarray          # (N,) bool mask
    grid_height: int
    grid_width: int
    color_counts: Dict[int, int]  # color -> count of objects with that color
    adjacency: np.ndarray      # (N, N) bool - whether objects are adjacent


def compute_object_properties_for_ranking(
    labels: np.ndarray, colors: List[int], bboxes: List[Tuple], grid: np.ndarray
) -> ObjectPropertiesForRanking:
    """Compute detailed properties for each object, used for ranking."""
    num_objects = len(colors)
    H, W = grid.shape

    centroids = np.zeros((num_objects, 2), dtype=np.float32)
    bboxes_arr = np.zeros((num_objects, 4), dtype=np.float32)
    areas = np.zeros(num_objects, dtype=np.float32)
    heights = np.zeros(num_objects, dtype=np.float32)
    widths = np.zeros(num_objects, dtype=np.float32)
    colors_arr = np.zeros(num_objects, dtype=np.int64)
    valid = np.zeros(num_objects, dtype=bool)

    # Count colors
    color_counts = defaultdict(int)
    for c in colors:
        color_counts[c] += 1

    for i in range(num_objects):
        mask = (labels == i + 1)
        area = mask.sum()

        if area > 0:
            rows, cols = np.where(mask)
            centroids[i] = [rows.mean(), cols.mean()]
            bboxes_arr[i] = [bboxes[i][0], bboxes[i][1], bboxes[i][2], bboxes[i][3]]
            areas[i] = area
            heights[i] = bboxes[i][2] - bboxes[i][0] + 1
            widths[i] = bboxes[i][3] - bboxes[i][1] + 1
            colors_arr[i] = colors[i]
            valid[i] = True

    # Compute adjacency (objects are adjacent if their masks touch after dilation)
    adjacency = np.zeros((num_objects, num_objects), dtype=bool)
    for i in range(num_objects):
        if not valid[i]:
            continue
        mask_i = (labels == i + 1)
        # Dilate mask by 1 pixel
        dilated = ndimage.binary_dilation(mask_i, iterations=1)

        for j in range(num_objects):
            if i == j or not valid[j]:
                continue
            mask_j = (labels == j + 1)
            if (dilated & mask_j).any():
                adjacency[i, j] = True

    return ObjectPropertiesForRanking(
        centroids=centroids,
        bboxes=bboxes_arr,
        areas=areas,
        heights=heights,
        widths=widths,
        colors=colors_arr,
        valid=valid,
        grid_height=H,
        grid_width=W,
        color_counts=dict(color_counts),
        adjacency=adjacency
    )


def compute_ranks(props: ObjectPropertiesForRanking, criterion: str) -> np.ndarray:
    """
    Compute ranks for each object based on criterion.

    Rank 0 = best/first according to criterion.
    Invalid objects get rank -1.

    Returns:
        ranks: (N,) array of ranks, -1 for invalid objects
    """
    n = len(props.valid)
    if n == 0:
        return np.array([], dtype=np.int32)

    # Compute sort key for each object (lower = better rank)
    keys = np.full(n, np.inf, dtype=np.float64)

    center_row = props.grid_height / 2
    center_col = props.grid_width / 2

    for i in range(n):
        if not props.valid[i]:
            continue

        row, col = props.centroids[i]
        area = props.areas[i]
        height = props.heights[i]
        width = props.widths[i]
        color = props.colors[i]

        if criterion == 'largest':
            keys[i] = -area  # Negative so larger = smaller key = better rank
        elif criterion == 'smallest':
            keys[i] = area
        elif criterion == 'tallest':
            keys[i] = -height
        elif criterion == 'shortest_height':
            keys[i] = height
        elif criterion == 'widest':
            keys[i] = -width
        elif criterion == 'narrowest':
            keys[i] = width
        elif criterion == 'leftmost':
            keys[i] = col
        elif criterion == 'rightmost':
            keys[i] = -col
        elif criterion == 'topmost':
            keys[i] = row
        elif criterion == 'bottommost':
            keys[i] = -row
        elif criterion == 'closest_to_center':
            dist = np.sqrt((row - center_row)**2 + (col - center_col)**2)
            keys[i] = dist
        elif criterion == 'farthest_from_center':
            dist = np.sqrt((row - center_row)**2 + (col - center_col)**2)
            keys[i] = -dist
        elif criterion == 'closest_to_tl':
            dist = np.sqrt(row**2 + col**2)
            keys[i] = dist
        elif criterion == 'closest_to_tr':
            dist = np.sqrt(row**2 + (col - props.grid_width)**2)
            keys[i] = dist
        elif criterion == 'closest_to_bl':
            dist = np.sqrt((row - props.grid_height)**2 + col**2)
            keys[i] = dist
        elif criterion == 'closest_to_br':
            dist = np.sqrt((row - props.grid_height)**2 + (col - props.grid_width)**2)
            keys[i] = dist
        elif criterion == 'farthest_from_tl':
            dist = np.sqrt(row**2 + col**2)
            keys[i] = -dist
        elif criterion == 'farthest_from_tr':
            dist = np.sqrt(row**2 + (col - props.grid_width)**2)
            keys[i] = -dist
        elif criterion == 'farthest_from_bl':
            dist = np.sqrt((row - props.grid_height)**2 + col**2)
            keys[i] = -dist
        elif criterion == 'farthest_from_br':
            dist = np.sqrt((row - props.grid_height)**2 + (col - props.grid_width)**2)
            keys[i] = -dist
        elif criterion == 'closest_to_top':
            keys[i] = row
        elif criterion == 'closest_to_bottom':
            keys[i] = props.grid_height - row
        elif criterion == 'closest_to_left':
            keys[i] = col
        elif criterion == 'closest_to_right':
            keys[i] = props.grid_width - col
        elif criterion == 'farthest_from_top':
            keys[i] = -row
        elif criterion == 'farthest_from_bottom':
            keys[i] = -(props.grid_height - row)
        elif criterion == 'farthest_from_left':
            keys[i] = -col
        elif criterion == 'farthest_from_right':
            keys[i] = -(props.grid_width - col)
        elif criterion == 'most_common_color':
            # Rank by color frequency (most common = rank 0)
            keys[i] = -props.color_counts.get(color, 0)
        elif criterion == 'least_common_color':
            keys[i] = props.color_counts.get(color, 0)
        elif criterion == 'most_neighbors':
            num_neighbors = props.adjacency[i].sum()
            keys[i] = -num_neighbors
        elif criterion == 'most_isolated':
            num_neighbors = props.adjacency[i].sum()
            keys[i] = num_neighbors
        else:
            raise ValueError(f"Unknown criterion: {criterion}")

    # Convert keys to ranks (ties get the same rank)
    ranks = np.full(n, -1, dtype=np.int32)
    valid_indices = np.where(props.valid)[0]

    if len(valid_indices) > 0:
        valid_keys = keys[valid_indices]
        sorted_order = np.argsort(valid_keys)

        # Assign ranks with tie handling - equal keys get equal ranks
        current_rank = 0
        for i, idx in enumerate(sorted_order):
            if i > 0:
                prev_idx = sorted_order[i - 1]
                # If key differs from previous, increment rank
                if valid_keys[idx] != valid_keys[prev_idx]:
                    current_rank = i  # Use position as rank (leaves gaps for ties)
            ranks[valid_indices[idx]] = current_rank

    return ranks


def apply_selection_rule(ranks: np.ndarray, valid: np.ndarray,
                         rule: str, num_objects: int) -> np.ndarray:
    """
    Apply a selection rule based on ranks.

    Returns:
        selected: (N,) bool array indicating which objects are selected
    """
    selected = np.zeros(len(ranks), dtype=bool)

    if rule == 'all':
        selected = valid.copy()
    elif rule == 'top_1':
        selected = (ranks == 0)
    elif rule == 'top_2':
        selected = (ranks >= 0) & (ranks <= 1)
    elif rule == 'top_3':
        selected = (ranks >= 0) & (ranks <= 2)
    elif rule == 'bottom_1':
        max_rank = ranks[valid].max() if valid.any() else -1
        selected = (ranks == max_rank)
    elif rule == 'bottom_2':
        max_rank = ranks[valid].max() if valid.any() else -1
        selected = (ranks >= 0) & (ranks >= max_rank - 1)
    elif rule == 'all_best':
        # Select ALL objects that share the best (lowest) rank
        if valid.any():
            best_rank = ranks[valid].min()
            selected = (ranks == best_rank)
        else:
            selected = np.zeros(len(ranks), dtype=bool)
    else:
        raise ValueError(f"Unknown selection rule: {rule}")

    return selected & valid


def compute_selection_mask(
    input_labels: np.ndarray,
    input_colors: List[int],
    input_bboxes: List[Tuple],
    input_grid: np.ndarray,
    criterion: str,
    rule: str
) -> np.ndarray:
    """
    Compute selection mask without filtering objects.

    This is a convenience wrapper that combines compute_object_properties_for_ranking,
    compute_ranks, and apply_selection_rule into a single call.

    Use this when you need to know which objects are selected but want to keep
    all objects in memory (e.g., for neural network training where the mask
    indicates which objects to focus on).

    Args:
        input_labels: Label mask for input objects
        input_colors: Colors for each object
        input_bboxes: Bounding boxes for each object
        input_grid: Original input grid
        criterion: Ranking criterion (e.g., 'largest', 'leftmost')
        rule: Selection rule (e.g., 'top_1', 'top_2', 'all')

    Returns:
        selection_mask: (N,) bool array indicating which objects are selected
    """
    ranking_props = compute_object_properties_for_ranking(
        input_labels, input_colors, input_bboxes, input_grid
    )
    ranks = compute_ranks(ranking_props, criterion)
    return apply_selection_rule(
        ranks, ranking_props.valid, rule, len(input_colors)
    )


def apply_object_selection(
    input_labels: np.ndarray,
    input_colors: List[int],
    input_bboxes: List[Tuple],
    input_grid: np.ndarray,
    criterion: str,
    rule: str
) -> Tuple[np.ndarray, np.ndarray, List[int], List[Tuple]]:
    """
    Apply selection filtering to objects in a single call.

    This is a high-level wrapper that combines:
    1. compute_object_properties_for_ranking()
    2. compute_ranks()
    3. apply_selection_rule()
    4. Filtering and relabeling

    Args:
        input_labels: Label mask for input objects (from extract_connected_components)
        input_colors: Colors for each object
        input_bboxes: Bounding boxes for each object
        input_grid: Original input grid
        criterion: Ranking criterion (e.g., 'largest', 'leftmost')
        rule: Selection rule (e.g., 'top_1', 'top_2', 'all')

    Returns:
        Tuple of:
        - selection_mask: (N,) bool array indicating which objects are selected
        - filtered_labels: Labels with non-selected objects zeroed and relabeled consecutively
        - filtered_colors: List of colors for selected objects only
        - filtered_bboxes: List of bboxes for selected objects only
    """
    # Compute ranking properties
    ranking_props = compute_object_properties_for_ranking(
        input_labels, input_colors, input_bboxes, input_grid
    )

    # Compute ranks
    ranks = compute_ranks(ranking_props, criterion)

    # Apply selection rule
    selection_mask = apply_selection_rule(
        ranks, ranking_props.valid, rule, len(input_colors)
    )

    # Filter labels (zero out non-selected)
    filtered_labels = input_labels.copy()
    for i in range(len(input_colors)):
        if not selection_mask[i]:
            filtered_labels[filtered_labels == i + 1] = 0

    # Filter colors and bboxes
    filtered_colors = [c for i, c in enumerate(input_colors) if selection_mask[i]]
    filtered_bboxes = [b for i, b in enumerate(input_bboxes) if selection_mask[i]]

    # Relabel to be consecutive (1, 2, 3, ...)
    new_labels = np.zeros_like(filtered_labels)
    for new_idx, old_idx in enumerate(np.where(selection_mask)[0]):
        new_labels[filtered_labels == old_idx + 1] = new_idx + 1
    filtered_labels = new_labels

    return selection_mask, filtered_labels, filtered_colors, filtered_bboxes


# =============================================================================
# Selection Screening
# =============================================================================

@dataclass
class SelectionSample:
    """A single example for selection screening."""
    puzzle_id: str
    example_idx: int
    props: ObjectPropertiesForRanking
    actual_selected: np.ndarray  # (N,) bool - from correspondence
    num_input: int
    num_selected: int


class SelectionScreener:
    """
    Screens ranking criteria + selection rules to find the best predictor
    for which objects are selected (appear in output).

    For each criterion (e.g., 'largest', 'leftmost') and rule (e.g., 'top_1'),
    computes accuracy of predicted selection vs actual selection from correspondences.
    """

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.results: Dict[Tuple[str, str], float] = {}

    def create_selection_samples(
        self,
        puzzles: Dict,
        puzzle_ids: List[str],
        use_color_only: bool = False,
        strategy: Optional[SegmentationStrategy] = None,
        correspondence_mode: CorrespondenceMode = "one_to_one",
        correspondence_margin: float = DEFAULT_MARGIN
    ) -> List[SelectionSample]:
        """Create selection samples from puzzles for screening.

        Args:
            puzzles: Dictionary of puzzles
            puzzle_ids: List of puzzle IDs to process
            use_color_only: DEPRECATED - use strategy instead
            strategy: Optional segmentation strategy for input/output grids
            correspondence_mode: Matching mode for correspondences
            correspondence_margin: Margin for non-one_to_one modes
        """
        samples = []

        # Determine segmentation modes
        if strategy:
            input_mode = strategy.input_mode
            output_mode = strategy.output_mode
        elif use_color_only:
            input_mode = SegmentationMode.COLOR
            output_mode = SegmentationMode.COLOR
        else:
            input_mode = SegmentationMode.CONNECTIVITY
            output_mode = SegmentationMode.CONNECTIVITY

        for puzzle_id in puzzle_ids:
            if puzzle_id not in puzzles:
                continue

            puzzle = puzzles[puzzle_id]
            examples = puzzle.get('train', [])

            for ex_idx, example in enumerate(examples):
                if 'output' not in example:
                    continue

                input_grid = np.array(example['input'], dtype=np.int64)
                output_grid = np.array(example['output'], dtype=np.int64)

                # Extract objects
                input_labels, input_colors, input_bboxes, _ = extract_connected_components(
                    input_grid, segmentation_mode=input_mode
                )
                output_labels, output_colors, output_bboxes, _ = extract_connected_components(
                    output_grid, segmentation_mode=output_mode
                )

                if len(input_colors) == 0:
                    continue

                # Convert to Object instances for correspondence matching
                input_objects = labels_to_objects(input_labels, input_colors, input_bboxes)
                output_objects = labels_to_objects(output_labels, output_colors, output_bboxes)

                # Find correspondences using canonical shape-based matching
                matches, _, _ = find_correspondences(
                    input_grid, output_grid,
                    input_objects, output_objects,
                    threshold=0.0,
                    use_matchable=False,
                    mode=correspondence_mode,
                    margin=correspondence_margin
                )

                # Build actual_selected from correspondences
                actual_selected = np.zeros(len(input_colors), dtype=bool)
                for in_idx, out_idx, score in matches:
                    actual_selected[in_idx] = True

                # Compute object properties for ranking
                props = compute_object_properties_for_ranking(
                    input_labels, input_colors, input_bboxes, input_grid
                )

                samples.append(SelectionSample(
                    puzzle_id=puzzle_id,
                    example_idx=ex_idx,
                    props=props,
                    actual_selected=actual_selected,
                    num_input=len(input_colors),
                    num_selected=int(actual_selected.sum())
                ))

        return samples

    def screen(
        self,
        samples: List[SelectionSample]
    ) -> Dict:
        """
        Screen all criterion + rule combinations to find best predictor.

        Returns dict with:
            - best_criterion: str
            - best_rule: str
            - best_accuracy: float
            - all_results: Dict[(criterion, rule), accuracy]
        """
        results = {}
        # Track how many objects each combination selects (for tie-breaking)
        num_selected = {}

        for criterion in RANKING_CRITERIA:
            for rule in SELECTION_RULES:
                correct = 0
                total = 0
                total_selected = 0

                for sample in samples:
                    ranks = compute_ranks(sample.props, criterion)
                    predicted = apply_selection_rule(
                        ranks, sample.props.valid, rule, sample.num_input
                    )

                    # Compare predicted vs actual selection
                    match = (predicted == sample.actual_selected)
                    correct += match.sum()
                    total += len(match)
                    total_selected += predicted.sum()

                accuracy = correct / total if total > 0 else 0.0
                results[(criterion, rule)] = accuracy
                num_selected[(criterion, rule)] = total_selected

                if self.verbose and accuracy > 0.8:
                    print(f"  {criterion:25s} + {rule:10s}: {accuracy:.1%}")

        # Find best
        if not results:
            return {
                'best_criterion': 'all',
                'best_rule': 'all',
                'best_accuracy': 0.0,
                'all_results': results
            }

        # Find best combo: highest accuracy, then most objects selected, then rule preference
        best_combo = max(results, key=lambda k: (results[k], num_selected[k], RULE_PREFERENCE.get(k[1], 0)))
        best_accuracy = results[best_combo]

        self.results = results

        return {
            'best_criterion': best_combo[0],
            'best_rule': best_combo[1],
            'best_accuracy': best_accuracy,
            'all_results': results
        }

    def analyze_selection_pattern(self, samples: List[SelectionSample]) -> Dict:
        """Analyze what kind of selection pattern exists in the samples."""
        if not samples:
            return {'pattern': 'no_samples'}

        # Check if all objects are always selected (no filtering)
        all_selected_counts = [s.num_selected for s in samples]
        all_input_counts = [s.num_input for s in samples]

        if all(sel == inp for sel, inp in zip(all_selected_counts, all_input_counts)):
            return {'pattern': 'all_selected', 'description': 'All input objects appear in output'}

        # Check if exactly one object is always selected
        if all(sel == 1 for sel in all_selected_counts):
            return {'pattern': 'single_selection', 'description': 'Exactly one object is selected'}

        # Check if a fixed number is always selected
        if len(set(all_selected_counts)) == 1:
            k = all_selected_counts[0]
            return {'pattern': f'fixed_k', 'k': k, 'description': f'Exactly {k} objects always selected'}

        # Check if selection count varies
        return {
            'pattern': 'variable',
            'min_selected': min(all_selected_counts),
            'max_selected': max(all_selected_counts),
            'description': 'Variable number of objects selected'
        }


# =============================================================================
# Visualization
# =============================================================================

def visualize_selection(samples: List[SelectionSample], criterion: str, rule: str):
    """Print detailed visualization of selection predictions."""
    print(f"\nSelection visualization: {criterion} + {rule}")
    print("=" * 70)

    for sample in samples[:5]:  # Show first 5 examples
        ranks = compute_ranks(sample.props, criterion)
        predicted = apply_selection_rule(
            ranks, sample.props.valid, rule, sample.num_input
        )

        print(f"\nPuzzle {sample.puzzle_id}, Example {sample.example_idx}")
        print(f"  {'Obj':>3} {'Color':>5} {'Area':>6} {'Pos(r,c)':>12} {'Rank':>4} {'Pred':>5} {'Actual':>6} {'Match':>5}")
        print(f"  {'-'*3} {'-'*5} {'-'*6} {'-'*12} {'-'*4} {'-'*5} {'-'*6} {'-'*5}")

        for i in range(sample.num_input):
            if not sample.props.valid[i]:
                continue
            color = sample.props.colors[i]
            area = sample.props.areas[i]
            row, col = sample.props.centroids[i]
            rank = ranks[i]
            pred = "YES" if predicted[i] else "no"
            actual = "YES" if sample.actual_selected[i] else "no"
            match = "✓" if predicted[i] == sample.actual_selected[i] else "✗"

            print(f"  {i:>3} {color:>5} {area:>6.0f} ({row:>4.1f},{col:>4.1f}) {rank:>4} {pred:>5} {actual:>6} {match:>5}")


def analyze_puzzle_selection(puzzle_id: str, puzzles: Dict,
                             use_color_only: bool = False,
                             verbose: bool = True,
                             visualize: bool = False,
                             strategy: Optional[SegmentationStrategy] = None,
                             correspondence_mode: CorrespondenceMode = "one_to_one",
                             correspondence_margin: float = DEFAULT_MARGIN):
    """
    Analyze selection patterns for a single puzzle.

    Args:
        puzzle_id: The puzzle ID to analyze
        puzzles: Dictionary of all puzzles
        use_color_only: DEPRECATED - use strategy instead
        verbose: Whether to print detailed output
        visualize: Whether to show detailed visualization
        strategy: Optional segmentation strategy for input/output grids
        correspondence_mode: Matching mode for correspondences
        correspondence_margin: Margin for non-one_to_one modes

    Returns:
        Dictionary with analysis results
    """
    if puzzle_id not in puzzles:
        print(f"Error: Puzzle {puzzle_id} not found")
        return None

    screener = SelectionScreener(verbose=False)
    samples = screener.create_selection_samples(
        puzzles, [puzzle_id], use_color_only,
        strategy=strategy,
        correspondence_mode=correspondence_mode,
        correspondence_margin=correspondence_margin
    )

    if not samples:
        print("No valid samples created")
        return None

    print(f"\nAnalyzing puzzle {puzzle_id}")
    print(f"Created {len(samples)} samples from training examples")

    # Analyze pattern
    pattern = screener.analyze_selection_pattern(samples)
    print(f"\nSelection pattern: {pattern['description']}")

    # Quick stats
    print("\nPer-example statistics:")
    for sample in samples:
        print(f"  Example {sample.example_idx}: {sample.num_input} objects, "
              f"{sample.num_selected} selected ({sample.num_selected}/{sample.num_input})")

    # Screen all criteria
    print("\nScreening selection rules...")
    results = screener.screen(samples)

    print(f"\nBest selection rule:")
    print(f"  Criterion: {results['best_criterion']}")
    print(f"  Rule: {results['best_rule']}")
    print(f"  Accuracy: {results['best_accuracy']:.1%}")

    # Show top 10 results
    print("\nTop 10 criterion+rule combinations:")
    sorted_results = sorted(results['all_results'].items(),
                            key=lambda x: x[1], reverse=True)
    for (criterion, rule), acc in sorted_results[:10]:
        marker = "  " if acc < results['best_accuracy'] else "->"
        print(f"  {marker} {criterion:<25} + {rule:<10}: {acc:>6.1%}")

    # Show what each criterion means
    if verbose:
        print("\nCriterion explanations:")
        criterion_desc = {
            'largest': 'Objects with most pixels',
            'smallest': 'Objects with fewest pixels',
            'leftmost': 'Objects nearest left edge',
            'rightmost': 'Objects nearest right edge',
            'topmost': 'Objects nearest top edge',
            'bottommost': 'Objects nearest bottom edge',
            'closest_to_center': 'Objects nearest grid center',
            'farthest_from_center': 'Objects farthest from grid center',
            'most_common_color': 'Objects with most frequently occurring color',
            'least_common_color': 'Objects with least frequently occurring color',
            'most_neighbors': 'Objects adjacent to most other objects',
            'most_isolated': 'Objects with fewest adjacent objects',
        }
        best_crit = results['best_criterion']
        if best_crit in criterion_desc:
            print(f"  {best_crit}: {criterion_desc[best_crit]}")

    # Visualize if requested
    if visualize:
        visualize_selection(samples, results['best_criterion'], results['best_rule'])

    return {
        'puzzle_id': puzzle_id,
        'pattern': pattern,
        'best_criterion': results['best_criterion'],
        'best_rule': results['best_rule'],
        'best_accuracy': results['best_accuracy'],
        'samples': samples,
        'all_results': results['all_results']
    }


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Selection Module - Analyze object selection patterns in ARC puzzles",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Analyze a single puzzle
    python selection_module.py --puzzle-id 6fa7a44f

    # Analyze with visualization
    python selection_module.py --puzzle-id 6fa7a44f --visualize

    # Scan all puzzles for selection patterns
    python selection_module.py --scan-all --min-accuracy 0.9

    # Use color-based object extraction (DEPRECATED - use --segmentation-mode color)
    python selection_module.py --puzzle-id 6fa7a44f --object-by-color

    # Different segmentation for input vs output:
    python selection_module.py --puzzle-id 6fa7a44f --input-segmentation-mode connectivity --output-segmentation-mode pixel

    # Different correspondence modes:
    python selection_module.py --puzzle-id 6fa7a44f --correspondence-mode many_to_one --correspondence-margin 0.1
        """
    )

    parser.add_argument("--puzzle-id", type=str,
                        help="Specific puzzle to analyze (e.g., 6fa7a44f)")
    parser.add_argument("--scan-all", action="store_true",
                        help="Scan all puzzles for selection patterns")
    parser.add_argument("--min-accuracy", type=float, default=0.9,
                        help="Minimum accuracy to report (for --scan-all)")
    parser.add_argument("--object-by-color", action="store_true",
                        help="DEPRECATED: Extract objects by color only. Use --segmentation-mode color instead.")
    parser.add_argument("--data-root", type=str, default="kaggle/combined",
                        help="Path to puzzle data")
    parser.add_argument("--verbose", action="store_true",
                        help="Show detailed output")
    parser.add_argument("--visualize", action="store_true",
                        help="Show selection visualization for each example")
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

    args = parser.parse_args()

    # Build segmentation strategy from arguments
    # Priority: specific mode > general mode > object-by-color flag > default (connectivity)
    if args.segmentation_mode:
        base_mode = SegmentationMode(args.segmentation_mode)
    elif args.object_by_color:
        base_mode = SegmentationMode.COLOR
    else:
        base_mode = SegmentationMode.CONNECTIVITY
    input_mode = SegmentationMode(args.input_segmentation_mode) if args.input_segmentation_mode else base_mode
    output_mode = SegmentationMode(args.output_segmentation_mode) if args.output_segmentation_mode else base_mode
    strategy = SegmentationStrategy(input_mode=input_mode, output_mode=output_mode)

    # Load puzzles
    print("Loading puzzles...")
    puzzles = _load_puzzles(args.data_root)
    print(f"Loaded {len(puzzles)} puzzles")

    if args.puzzle_id:
        # Analyze single puzzle
        analyze_puzzle_selection(
            args.puzzle_id,
            puzzles,
            use_color_only=args.object_by_color,
            verbose=args.verbose,
            visualize=args.visualize,
            strategy=strategy,
            correspondence_mode=args.correspondence_mode,
            correspondence_margin=args.correspondence_margin
        )

    elif args.scan_all:
        # Scan all puzzles for selection patterns
        print(f"\nScanning all puzzles for selection patterns...")
        print(f"Looking for accuracy >= {args.min_accuracy:.0%}")

        screener = SelectionScreener(verbose=False)
        interesting_puzzles = []

        for puzzle_id in puzzles:
            samples = screener.create_selection_samples(
                puzzles, [puzzle_id], args.object_by_color,
                strategy=strategy,
                correspondence_mode=args.correspondence_mode,
                correspondence_margin=args.correspondence_margin
            )

            if not samples:
                continue

            # Skip puzzles where all objects are selected (no filtering)
            pattern = screener.analyze_selection_pattern(samples)
            if pattern['pattern'] == 'all_selected':
                continue

            # Screen
            results = screener.screen(samples)

            if results['best_accuracy'] >= args.min_accuracy:
                interesting_puzzles.append({
                    'puzzle_id': puzzle_id,
                    'criterion': results['best_criterion'],
                    'rule': results['best_rule'],
                    'accuracy': results['best_accuracy'],
                    'pattern': pattern,
                    'num_samples': len(samples)
                })

        # Sort by accuracy
        interesting_puzzles.sort(key=lambda x: x['accuracy'], reverse=True)

        print(f"\nFound {len(interesting_puzzles)} puzzles with selection patterns")
        print("\nTop puzzles with clear selection rules:")
        print(f"{'Puzzle':<12} {'Criterion':<25} {'Rule':<10} {'Acc':>6} {'Pattern':<25}")
        print("-" * 85)

        for p in interesting_puzzles[:30]:
            pattern_str = p['pattern'].get('description', str(p['pattern'].get('pattern', '')))[:25]
            print(f"{p['puzzle_id']:<12} {p['criterion']:<25} {p['rule']:<10} "
                  f"{p['accuracy']:>5.0%} {pattern_str:<25}")

        # Summarize by criterion
        print("\nCriterion frequency in top results:")
        criterion_counts = defaultdict(int)
        for p in interesting_puzzles:
            criterion_counts[p['criterion']] += 1

        for criterion, count in sorted(criterion_counts.items(), key=lambda x: -x[1]):
            print(f"  {criterion:<25}: {count}")

    else:
        print("Please specify --puzzle-id or --scan-all")
        parser.print_help()
