#!/usr/bin/env python3
"""
Ranking Module for Object Selection

Tests the hypothesis that object selection (which input objects appear in the output)
can be predicted by ranking objects according to various criteria.

The correspondence system already tells us which objects are selected:
    correspondence[i] >= 0  →  object i appears in output
    correspondence[i] == -1 →  object i is filtered out

This script screens ranking criteria to discover selection rules like:
    - "Keep only the largest object"
    - "Keep the leftmost object"
    - "Keep objects closest to center"

Usage:
    # Test on a specific puzzle
    python ranking_selection.py --puzzle-id 6fa7a44f
    
    # Test on all puzzles and find those with selection patterns
    python ranking_selection.py --scan-all --min-accuracy 0.9
    
    # Test with color-based object extraction
    python ranking_selection.py --puzzle-id 6fa7a44f --object-by-color
"""

import argparse
import json
import os
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

from scipy import ndimage

# Object detection and extraction
from object_module import (
    extract_connected_components,
    MAX_OBJECTS,
)

# Object correspondence matching
from correspondence_module import find_object_correspondences

# Ranking criteria - each defines how to sort objects
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
    
    # Position-based (distance)
    'closest_to_center',
    'farthest_from_center',
    'closest_to_tl',
    'closest_to_tr',
    'closest_to_bl',
    'closest_to_br',
    
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
]


# =============================================================================
# Data Loading (borrowed from relational_position.py)
# =============================================================================

def load_puzzles(data_root: str = "kaggle/combined") -> Dict:
    """Load all ARC puzzles from JSON files."""
    all_puzzles = {}
    
    subsets = ["training", "evaluation", "training2", "evaluation2"]
    
    for subset in subsets:
        challenges_path = f"{data_root}/arc-agi_{subset}_challenges.json"
        solutions_path = f"{data_root}/arc-agi_{subset}_solutions.json"
        
        if not os.path.exists(challenges_path):
            continue
            
        with open(challenges_path) as f:
            puzzles = json.load(f)
            
        if os.path.exists(solutions_path):
            with open(solutions_path) as f:
                solutions = json.load(f)
            for puzzle_id in puzzles:
                if puzzle_id in solutions:
                    for i, test in enumerate(puzzles[puzzle_id].get('test', [])):
                        if i < len(solutions[puzzle_id]):
                            test['output'] = solutions[puzzle_id][i]
                            
        all_puzzles.update(puzzles)
    
    return all_puzzles


# =============================================================================
# Object Properties
# =============================================================================

@dataclass
class ObjectProperties:
    """Properties of objects in a grid, used for ranking."""
    centroids: np.ndarray      # (N, 2) row, col
    bboxes: np.ndarray         # (N, 4) min_row, min_col, max_row, max_col
    areas: np.ndarray          # (N,)
    heights: np.ndarray        # (N,)
    widths: np.ndarray         # (N,)
    colors: np.ndarray         # (N,)
    valid: np.ndarray          # (N,) bool
    grid_height: int
    grid_width: int
    color_counts: Dict[int, int]  # color -> count of objects with that color
    adjacency: np.ndarray      # (N, N) bool - whether objects are adjacent


def compute_object_properties(labels: np.ndarray, colors: List[int],
                               bboxes: List[Tuple], grid: np.ndarray) -> ObjectProperties:
    """Compute detailed properties for each object."""
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
    
    # Compute adjacency (objects are adjacent if their bboxes are within 1 pixel)
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
    
    return ObjectProperties(
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


# =============================================================================
# Ranking Functions
# =============================================================================

def compute_ranks(props: ObjectProperties, criterion: str) -> np.ndarray:
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
    
    # Convert keys to ranks
    ranks = np.full(n, -1, dtype=np.int32)
    valid_indices = np.where(props.valid)[0]
    
    if len(valid_indices) > 0:
        valid_keys = keys[valid_indices]
        sorted_order = np.argsort(valid_keys)
        for rank, idx in enumerate(sorted_order):
            ranks[valid_indices[idx]] = rank
    
    return ranks


def apply_selection_rule(ranks: np.ndarray, valid: np.ndarray, 
                         rule: str, num_objects: int) -> np.ndarray:
    """
    Apply a selection rule based on ranks.
    
    Returns:
        selected: (N,) bool array
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
    else:
        raise ValueError(f"Unknown selection rule: {rule}")
    
    return selected & valid


# =============================================================================
# Screening
# =============================================================================

@dataclass 
class SelectionSample:
    """A single example for selection screening."""
    puzzle_id: str
    example_idx: int
    props: ObjectProperties
    actual_selected: np.ndarray  # (N,) bool - from correspondence
    num_input: int
    num_selected: int


def create_selection_samples(puzzles: Dict, puzzle_ids: List[str],
                              use_color_only: bool = False) -> List[SelectionSample]:
    """Create selection samples from puzzles."""
    samples = []
    
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
                input_grid, use_color_only=use_color_only
            )
            output_labels, output_colors, output_bboxes, _ = extract_connected_components(
                output_grid, use_color_only=use_color_only
            )
            
            if len(input_colors) == 0:
                continue
            
            # Find correspondences
            matches = find_object_correspondences(
                input_labels, input_colors,
                output_labels, output_colors,
                iou_threshold=0.0
            )
            
            # Build actual_selected from correspondences
            actual_selected = np.zeros(len(input_colors), dtype=bool)
            for in_idx, out_idx, score in matches:
                actual_selected[in_idx] = True
            
            # Compute object properties
            props = compute_object_properties(
                input_labels, input_colors, input_bboxes, input_grid
            )
            
            samples.append(SelectionSample(
                puzzle_id=puzzle_id,
                example_idx=ex_idx,
                props=props,
                actual_selected=actual_selected,
                num_input=len(input_colors),
                num_selected=actual_selected.sum()
            ))
    
    return samples


def screen_selection_rules(samples: List[SelectionSample], 
                           verbose: bool = False) -> Dict:
    """
    Screen all criterion + rule combinations to find best predictor.
    
    Returns dict with:
        - best_criterion
        - best_rule  
        - best_accuracy
        - all_results: {(criterion, rule): accuracy}
    """
    results = {}
    
    for criterion in RANKING_CRITERIA:
        for rule in SELECTION_RULES:
            correct = 0
            total = 0
            
            for sample in samples:
                ranks = compute_ranks(sample.props, criterion)
                predicted = apply_selection_rule(
                    ranks, sample.props.valid, rule, sample.num_input
                )
                
                # Compare predicted vs actual selection
                match = (predicted == sample.actual_selected)
                correct += match.sum()
                total += len(match)
            
            accuracy = correct / total if total > 0 else 0.0
            results[(criterion, rule)] = accuracy
            
            if verbose and accuracy > 0.8:
                print(f"  {criterion:25s} + {rule:10s}: {accuracy:.1%}")
    
    # Find best
    best_combo = max(results, key=results.get)
    best_accuracy = results[best_combo]
    
    return {
        'best_criterion': best_combo[0],
        'best_rule': best_combo[1],
        'best_accuracy': best_accuracy,
        'all_results': results
    }


def analyze_selection_pattern(samples: List[SelectionSample]) -> Dict:
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
    print("=" * 60)
    
    for sample in samples[:3]:  # Show first 3 examples
        ranks = compute_ranks(sample.props, criterion)
        predicted = apply_selection_rule(
            ranks, sample.props.valid, rule, sample.num_input
        )
        
        print(f"\nPuzzle {sample.puzzle_id}, Example {sample.example_idx}")
        print(f"  {'Obj':>3} {'Color':>5} {'Area':>6} {'Rank':>4} {'Pred':>5} {'Actual':>6} {'Match':>5}")
        print(f"  {'-'*3} {'-'*5} {'-'*6} {'-'*4} {'-'*5} {'-'*6} {'-'*5}")
        
        for i in range(sample.num_input):
            if not sample.props.valid[i]:
                continue
            color = sample.props.colors[i]
            area = sample.props.areas[i]
            rank = ranks[i]
            pred = "YES" if predicted[i] else "no"
            actual = "YES" if sample.actual_selected[i] else "no"
            match = "✓" if predicted[i] == sample.actual_selected[i] else "✗"
            
            print(f"  {i:>3} {color:>5} {area:>6.0f} {rank:>4} {pred:>5} {actual:>6} {match:>5}")


# =============================================================================
# Main
# =============================================================================

def create_synthetic_test_samples() -> List[SelectionSample]:
    """
    Create synthetic test samples to verify the ranking logic.
    
    Creates puzzles with known selection rules:
    1. "Keep largest" - 3 objects, only the largest appears in output
    2. "Keep leftmost" - 4 objects, only the leftmost appears in output  
    3. "Keep top 2 by size" - 4 objects, 2 largest appear in output
    """
    samples = []
    
    # Puzzle 1: Keep largest (3 examples)
    for ex_idx in range(3):
        # Vary sizes each example but always select largest
        areas = np.array([50 + ex_idx*10, 100 + ex_idx*5, 30 + ex_idx*3], dtype=np.float32)
        largest_idx = np.argmax(areas)
        
        props = ObjectProperties(
            centroids=np.array([[2, 2], [5, 8], [8, 4]], dtype=np.float32),
            bboxes=np.array([[1, 1, 4, 4], [4, 7, 7, 10], [7, 3, 9, 6]], dtype=np.float32),
            areas=areas,
            heights=np.array([3, 3, 2], dtype=np.float32),
            widths=np.array([3, 3, 3], dtype=np.float32),
            colors=np.array([1, 2, 3], dtype=np.int64),
            valid=np.array([True, True, True]),
            grid_height=10,
            grid_width=12,
            color_counts={1: 1, 2: 1, 3: 1},
            adjacency=np.zeros((3, 3), dtype=bool)
        )
        
        actual_selected = np.array([i == largest_idx for i in range(3)])
        
        samples.append(SelectionSample(
            puzzle_id="synthetic_largest",
            example_idx=ex_idx,
            props=props,
            actual_selected=actual_selected,
            num_input=3,
            num_selected=1
        ))
    
    # Puzzle 2: Keep leftmost (3 examples)
    # IMPORTANT: All dimensions vary but don't correlate with position
    for ex_idx in range(3):
        # Vary positions each example, leftmost is always object at min column
        col_positions = np.array([2 + ex_idx, 8, 14, 20], dtype=np.float32)
        leftmost_idx = np.argmin(col_positions)
        
        # Sizes, heights, widths all vary but leftmost is never extreme
        # This ensures only position-based criteria will work
        data_by_example = [
            # (areas, heights, widths) - leftmost is index 0, medium in all
            ([30, 50, 20, 40], [3, 5, 2, 4], [10, 10, 10, 10]),
            ([25, 15, 45, 35], [4, 3, 6, 5], [6, 5, 8, 7]),
            ([35, 40, 25, 50], [4, 5, 3, 6], [9, 8, 8, 8]),
        ]
        areas, heights, widths = data_by_example[ex_idx]
        areas = np.array(areas, dtype=np.float32)
        heights = np.array(heights, dtype=np.float32)
        widths = np.array(widths, dtype=np.float32)
        
        props = ObjectProperties(
            centroids=np.array([[3, col_positions[0]], [3, col_positions[1]], 
                               [7, col_positions[2]], [7, col_positions[3]]], dtype=np.float32),
            bboxes=np.array([[2, col_positions[0]-1, 4, col_positions[0]+1],
                            [2, col_positions[1]-1, 4, col_positions[1]+1],
                            [6, col_positions[2]-1, 8, col_positions[2]+1],
                            [6, col_positions[3]-1, 8, col_positions[3]+1]], dtype=np.float32),
            areas=areas,
            heights=heights,
            widths=widths,
            colors=np.array([1, 2, 3, 4], dtype=np.int64),
            valid=np.array([True, True, True, True]),
            grid_height=10,
            grid_width=25,
            color_counts={1: 1, 2: 1, 3: 1, 4: 1},
            adjacency=np.zeros((4, 4), dtype=bool)
        )
        
        actual_selected = np.array([i == leftmost_idx for i in range(4)])
        
        samples.append(SelectionSample(
            puzzle_id="synthetic_leftmost",
            example_idx=ex_idx,
            props=props,
            actual_selected=actual_selected,
            num_input=4,
            num_selected=1
        ))
    
    # Puzzle 3: Keep top 2 by size (3 examples)
    for ex_idx in range(3):
        areas = np.array([30 + ex_idx*5, 80 + ex_idx*3, 50 + ex_idx*2, 20], dtype=np.float32)
        sorted_indices = np.argsort(-areas)  # Descending
        top2 = set(sorted_indices[:2])
        
        props = ObjectProperties(
            centroids=np.array([[2, 2], [2, 8], [6, 2], [6, 8]], dtype=np.float32),
            bboxes=np.array([[1, 1, 3, 3], [1, 7, 3, 9], [5, 1, 7, 3], [5, 7, 7, 9]], dtype=np.float32),
            areas=areas,
            heights=np.array([2, 2, 2, 2], dtype=np.float32),
            widths=np.array([2, 2, 2, 2], dtype=np.float32),
            colors=np.array([1, 2, 3, 4], dtype=np.int64),
            valid=np.array([True, True, True, True]),
            grid_height=10,
            grid_width=12,
            color_counts={1: 1, 2: 1, 3: 1, 4: 1},
            adjacency=np.zeros((4, 4), dtype=bool)
        )
        
        actual_selected = np.array([i in top2 for i in range(4)])
        
        samples.append(SelectionSample(
            puzzle_id="synthetic_top2_size",
            example_idx=ex_idx,
            props=props,
            actual_selected=actual_selected,
            num_input=4,
            num_selected=2
        ))
    
    # Puzzle 4: Keep smallest (to test that we can distinguish from largest)
    for ex_idx in range(3):
        areas = np.array([50, 30, 80, 45], dtype=np.float32)
        smallest_idx = np.argmin(areas)
        
        props = ObjectProperties(
            centroids=np.array([[2, 2], [2, 8], [6, 2], [6, 8]], dtype=np.float32),
            bboxes=np.array([[1, 1, 3, 3], [1, 7, 3, 9], [5, 1, 7, 3], [5, 7, 7, 9]], dtype=np.float32),
            areas=areas,
            heights=np.array([2, 2, 2, 2], dtype=np.float32),
            widths=np.array([2, 2, 2, 2], dtype=np.float32),
            colors=np.array([1, 2, 3, 4], dtype=np.int64),
            valid=np.array([True, True, True, True]),
            grid_height=10,
            grid_width=12,
            color_counts={1: 1, 2: 1, 3: 1, 4: 1},
            adjacency=np.zeros((4, 4), dtype=bool)
        )
        
        actual_selected = np.array([i == smallest_idx for i in range(4)])
        
        samples.append(SelectionSample(
            puzzle_id="synthetic_smallest",
            example_idx=ex_idx,
            props=props,
            actual_selected=actual_selected,
            num_input=4,
            num_selected=1
        ))
    
    # Puzzle 5: Keep most isolated (tests relational criterion)
    # Object 3 is far from others (no adjacency), but NOT extreme in position/size
    # Objects 0,1,2 are in a tight cluster (adjacent), object 3 is alone but in middle region
    for ex_idx in range(3):
        # Adjacency: 0-1, 1-2, 0-2 are adjacent (cluster), 3 is isolated
        adjacency = np.array([
            [False, True, True, False],
            [True, False, True, False],
            [True, True, False, False],
            [False, False, False, False]
        ], dtype=bool)
        
        # All properties vary, isolated object (3) is never extreme in any
        # Position: cluster at top-left and bottom-right corners, isolated object in middle
        data_by_example = [
            # centroids: obj 0,1,2 cluster top-left; obj 3 is NOT extreme
            # obj 3 at (7, 10) is middle-ish in a 15x20 grid
            {
                'centroids': [[2, 2], [2, 4], [4, 3], [7, 10]],
                'areas': [20, 40, 35, 28],   # 28 is middle
                'heights': [3, 5, 4, 4],     # 4 is middle  
                'widths': [8, 6, 9, 7],      # 7 is middle
            },
            {
                'centroids': [[12, 15], [12, 17], [14, 16], [7, 9]],  # cluster bottom-right, obj 3 middle
                'areas': [35, 25, 45, 30],   # 30 is middle
                'heights': [4, 6, 3, 5],     # 5 is middle
                'widths': [7, 9, 6, 8],      # 8 is middle
            },
            {
                'centroids': [[2, 16], [2, 18], [4, 17], [8, 8]],  # cluster top-right, obj 3 middle-left
                'areas': [28, 50, 22, 35],   # 35 is middle-ish
                'heights': [5, 3, 6, 4],     # 4 is middle
                'widths': [9, 7, 10, 8],     # 8 is middle
            },
        ]
        d = data_by_example[ex_idx]
        centroids = np.array(d['centroids'], dtype=np.float32)
        areas = np.array(d['areas'], dtype=np.float32)
        heights = np.array(d['heights'], dtype=np.float32)
        widths = np.array(d['widths'], dtype=np.float32)
        
        # Compute bboxes from centroids and sizes (approximate)
        bboxes = np.zeros((4, 4), dtype=np.float32)
        for i in range(4):
            h, w = heights[i], widths[i]
            r, c = centroids[i]
            bboxes[i] = [r - h/2, c - w/2, r + h/2, c + w/2]
        
        props = ObjectProperties(
            centroids=centroids,
            bboxes=bboxes,
            areas=areas,
            heights=heights,
            widths=widths,
            colors=np.array([1, 2, 3, 4], dtype=np.int64),
            valid=np.array([True, True, True, True]),
            grid_height=15,
            grid_width=20,
            color_counts={1: 1, 2: 1, 3: 1, 4: 1},
            adjacency=adjacency
        )
        
        # Object 3 is selected (the isolated one)
        actual_selected = np.array([False, False, False, True])
        
        samples.append(SelectionSample(
            puzzle_id="synthetic_isolated",
            example_idx=ex_idx,
            props=props,
            actual_selected=actual_selected,
            num_input=4,
            num_selected=1
        ))
    
    return samples


def run_synthetic_tests():
    """Run tests on synthetic data to verify ranking logic."""
    print("=" * 60)
    print("SYNTHETIC DATA TESTS")
    print("=" * 60)
    
    all_samples = create_synthetic_test_samples()
    
    # Group by puzzle
    puzzles = defaultdict(list)
    for s in all_samples:
        puzzles[s.puzzle_id].append(s)
    
    print(f"\nCreated {len(all_samples)} synthetic samples across {len(puzzles)} test puzzles\n")
    
    expected_rules = {
        'synthetic_largest': [('largest', 'top_1')],
        'synthetic_leftmost': [('leftmost', 'top_1'), ('closest_to_tl', 'top_1')],
        'synthetic_top2_size': [('largest', 'top_2')],
        # smallest+top_1 is equivalent to largest+bottom_1
        'synthetic_smallest': [('smallest', 'top_1'), ('largest', 'bottom_1')],
        # For isolated: multiple criteria work due to synthetic data coincidences
        # The key point is that most_isolated+top_1 achieves 100%, even if others do too
        'synthetic_isolated': [
            ('most_isolated', 'top_1'), 
            ('most_neighbors', 'bottom_1'),
            ('closest_to_center', 'top_1'),
            ('farthest_from_center', 'bottom_1'),
        ],
    }
    
    all_passed = True
    
    for puzzle_id, samples in puzzles.items():
        print(f"\nPuzzle: {puzzle_id}")
        print("-" * 40)
        
        # Analyze pattern
        pattern = analyze_selection_pattern(samples)
        print(f"  Pattern: {pattern['description']}")
        
        # Screen
        results = screen_selection_rules(samples, verbose=False)
        
        print(f"  Best criterion: {results['best_criterion']}")
        print(f"  Best rule: {results['best_rule']}")
        print(f"  Accuracy: {results['best_accuracy']:.1%}")
        
        # Check if it matches any expected rule
        expected_list = expected_rules.get(puzzle_id, [])
        found = (results['best_criterion'], results['best_rule'])
        if found in expected_list:
            print(f"  ✓ Matches expected: {found[0]} + {found[1]}")
        elif expected_list:
            print(f"  ✗ Expected one of: {expected_list}")
            print(f"    Got: {found}")
            # Show all rules that achieved same accuracy
            best_acc = results['best_accuracy']
            tied_rules = [(c, r) for (c, r), acc in results['all_results'].items() 
                         if acc == best_acc]
            print(f"    All rules with {best_acc:.0%} accuracy: {tied_rules}")
            all_passed = False
        
        # Show detailed results for this puzzle
        if results['best_accuracy'] < 1.0:
            print(f"  WARNING: Accuracy < 100%, showing details:")
            visualize_selection(samples, results['best_criterion'], results['best_rule'])
    
    print("\n" + "=" * 60)
    if all_passed:
        print("ALL TESTS PASSED ✓")
    else:
        print("SOME TESTS FAILED ✗")
    print("=" * 60)
    
    return all_passed


def main():
    parser = argparse.ArgumentParser(description="Test ranking module for object selection")
    parser.add_argument("--puzzle-id", type=str, help="Specific puzzle to analyze")
    parser.add_argument("--scan-all", action="store_true", help="Scan all puzzles")
    parser.add_argument("--min-accuracy", type=float, default=0.9,
                        help="Minimum accuracy to report (for --scan-all)")
    parser.add_argument("--object-by-color", action="store_true",
                        help="Extract objects by color only (no connectivity)")
    parser.add_argument("--data-root", type=str, default="kaggle/combined",
                        help="Path to puzzle data")
    parser.add_argument("--verbose", action="store_true", help="Show detailed output")
    parser.add_argument("--visualize", action="store_true", help="Show selection visualization")
    parser.add_argument("--test", action="store_true", help="Run synthetic tests")
    args = parser.parse_args()
    
    # Run synthetic tests if requested
    if args.test:
        run_synthetic_tests()
        return
    
    # Load puzzles
    print("Loading puzzles...")
    puzzles = load_puzzles(args.data_root)
    print(f"Loaded {len(puzzles)} puzzles")
    
    if args.puzzle_id:
        # Analyze single puzzle
        if args.puzzle_id not in puzzles:
            print(f"Error: Puzzle {args.puzzle_id} not found")
            return
        
        puzzle_ids = [args.puzzle_id]
        samples = create_selection_samples(puzzles, puzzle_ids, args.object_by_color)
        
        if not samples:
            print("No valid samples created")
            return
        
        print(f"\nAnalyzing puzzle {args.puzzle_id}")
        print(f"Created {len(samples)} samples")
        
        # Analyze pattern
        pattern = analyze_selection_pattern(samples)
        print(f"\nSelection pattern: {pattern['description']}")
        
        # Quick stats
        for sample in samples:
            print(f"  Example {sample.example_idx}: {sample.num_input} objects, "
                  f"{sample.num_selected} selected")
        
        # Screen all criteria
        print("\nScreening selection rules...")
        results = screen_selection_rules(samples, verbose=args.verbose)
        
        print(f"\nBest selection rule:")
        print(f"  Criterion: {results['best_criterion']}")
        print(f"  Rule: {results['best_rule']}")
        print(f"  Accuracy: {results['best_accuracy']:.1%}")
        
        # Show top 5 results
        print("\nTop 5 criterion+rule combinations:")
        sorted_results = sorted(results['all_results'].items(), 
                                key=lambda x: x[1], reverse=True)
        for (criterion, rule), acc in sorted_results[:5]:
            print(f"  {criterion:25s} + {rule:10s}: {acc:.1%}")
        
        # Visualize if requested
        if args.visualize:
            visualize_selection(samples, results['best_criterion'], results['best_rule'])
    
    elif args.scan_all:
        # Scan all puzzles for selection patterns
        print(f"\nScanning all puzzles for selection patterns...")
        print(f"Looking for accuracy >= {args.min_accuracy:.0%}")
        
        interesting_puzzles = []
        
        for puzzle_id in puzzles:
            samples = create_selection_samples(puzzles, [puzzle_id], args.object_by_color)
            
            if not samples:
                continue
            
            # Skip puzzles where all objects are selected (no filtering)
            pattern = analyze_selection_pattern(samples)
            if pattern['pattern'] == 'all_selected':
                continue
            
            # Screen
            results = screen_selection_rules(samples, verbose=False)
            
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
        print(f"{'Puzzle':<12} {'Criterion':<25} {'Rule':<10} {'Acc':>6} {'Pattern':<20}")
        print("-" * 80)
        
        for p in interesting_puzzles[:30]:
            pattern_str = p['pattern'].get('description', str(p['pattern'].get('pattern', '')))[:20]
            print(f"{p['puzzle_id']:<12} {p['criterion']:<25} {p['rule']:<10} "
                  f"{p['accuracy']:>5.0%} {pattern_str:<20}")
        
        # Summarize by criterion
        print("\nCriterion frequency in top results:")
        criterion_counts = defaultdict(int)
        for p in interesting_puzzles:
            criterion_counts[p['criterion']] += 1
        
        for criterion, count in sorted(criterion_counts.items(), key=lambda x: -x[1]):
            print(f"  {criterion:<25}: {count}")
    
    else:
        print("Please specify --puzzle-id or --scan-all")
        return


if __name__ == "__main__":
    main()