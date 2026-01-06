"""
Correspondence Module for ARC Puzzle Solver

This module provides functions for finding correspondences between input and output
objects in ARC puzzles. It uses a combination of color matching, IoU overlap,
area similarity, and pattern matching to establish object correspondences.

Usage:
    from correspondence_module import find_object_correspondences, compute_iou

    # Command-line visualization:
    python correspondence_module.py --puzzle-id 1990f7a8
"""

from typing import List, Tuple, Optional
import numpy as np


# =============================================================================
# IoU (Intersection over Union)
# =============================================================================

def compute_iou(mask1: np.ndarray, mask2: np.ndarray) -> float:
    """
    Compute intersection over union between two binary masks.

    Handles different-sized masks by padding the smaller one to match the larger.

    Args:
        mask1: First binary mask (H1, W1)
        mask2: Second binary mask (H2, W2)

    Returns:
        IoU score in [0, 1]
    """
    # Handle different-sized masks (e.g., input/output grids have different dimensions)
    if mask1.shape != mask2.shape:
        h1, w1 = mask1.shape
        h2, w2 = mask2.shape
        max_h, max_w = max(h1, h2), max(w1, w2)

        # Create padded versions
        padded1 = np.zeros((max_h, max_w), dtype=bool)
        padded2 = np.zeros((max_h, max_w), dtype=bool)
        padded1[:h1, :w1] = mask1
        padded2[:h2, :w2] = mask2

        mask1, mask2 = padded1, padded2

    intersection = (mask1 & mask2).sum()
    union = (mask1 | mask2).sum()
    if union == 0:
        return 0.0
    return intersection / union


def compute_iou_from_pixels(pixels1: set, pixels2: set) -> float:
    """
    Compute Intersection over Union between two pixel sets.

    This is an alternative to mask-based IoU when objects are represented
    as sets of (row, col) tuples.

    Args:
        pixels1: Set of (row, col) tuples for first object
        pixels2: Set of (row, col) tuples for second object

    Returns:
        IoU score in [0, 1]
    """
    if not pixels1 or not pixels2:
        return 0.0
    intersection = len(pixels1 & pixels2)
    union = len(pixels1 | pixels2)
    if union == 0:
        return 0.0
    return intersection / union


# =============================================================================
# Pattern Matching
# =============================================================================

def _extract_pattern_from_mask(grid: np.ndarray, mask: np.ndarray) -> Optional[np.ndarray]:
    """
    Extract the pixel pattern within a mask's bounding box.

    Args:
        grid: Original grid with color values
        mask: Binary mask indicating object pixels

    Returns:
        Pattern array (colors within bounding box) or None if mask is empty
    """
    rows, cols = np.where(mask)
    if len(rows) == 0:
        return None
    r_min, r_max = rows.min(), rows.max()
    c_min, c_max = cols.min(), cols.max()
    # Extract the bounding box region from the original grid
    return grid[r_min:r_max+1, c_min:c_max+1].copy()


def extract_pattern_from_bbox(grid: np.ndarray, row: int, col: int,
                               height: int, width: int) -> np.ndarray:
    """
    Extract the pixel pattern within a bounding box.

    This is useful when working with Object instances that have row, col,
    height, and width properties.

    Args:
        grid: Original grid with color values
        row: Top-left row of bounding box
        col: Top-left column of bounding box
        height: Height of bounding box
        width: Width of bounding box

    Returns:
        Pattern array (colors within bounding box)
    """
    return grid[row:row + height, col:col + width].copy()


def pattern_similarity(pattern1: Optional[np.ndarray], pattern2: Optional[np.ndarray]) -> float:
    """
    Compute similarity between two patterns (0 to 1, higher = more similar).

    Args:
        pattern1: First pattern array
        pattern2: Second pattern array

    Returns:
        Similarity score in [0, 1]
    """
    if pattern1 is None or pattern2 is None:
        return 0.0
    # Must be same size for exact match
    if pattern1.shape != pattern2.shape:
        return 0.0
    # Count matching pixels
    matches = np.sum(pattern1 == pattern2)
    total = pattern1.size
    return matches / total if total > 0 else 0.0


# Keep private alias for backwards compatibility within this module
_pattern_similarity = pattern_similarity


# =============================================================================
# Object Correspondence Finding
# =============================================================================

def find_object_correspondences(
    input_labels: np.ndarray,
    input_colors: List[int],
    output_labels: np.ndarray,
    output_colors: List[int],
    iou_threshold: float = 0.1,
    input_grid: Optional[np.ndarray] = None,
    output_grid: Optional[np.ndarray] = None
) -> List[Tuple[int, int, float]]:
    """
    Find correspondences between input and output objects.

    Uses a combination of:
    1. Same color (required)
    2. IoU overlap (for objects that don't move much)
    3. Area similarity
    4. Pattern matching (when grids provided) - compares actual pixel patterns

    When objects move (IoU=0), pattern matching becomes the primary discriminator.

    Args:
        input_labels: Label mask for input (each object has unique label 1, 2, ...)
        input_colors: Dominant color for each input object (indexed by label - 1)
        output_labels: Label mask for output
        output_colors: Dominant color for each output object
        iou_threshold: Minimum score threshold for a match
        input_grid: Original input grid (optional, enables pattern matching)
        output_grid: Original output grid (optional, enables pattern matching)

    Returns:
        List of (input_idx, output_idx, score) tuples, where indices are 0-based.
    """
    num_input = len(input_colors)
    num_output = len(output_colors)

    if num_input == 0 or num_output == 0:
        return []

    # Pre-extract patterns if grids provided
    use_pattern_matching = input_grid is not None and output_grid is not None
    input_patterns = []
    output_patterns = []

    if use_pattern_matching:
        for i in range(num_input):
            mask = (input_labels == i + 1)
            input_patterns.append(_extract_pattern_from_mask(input_grid, mask))
        for j in range(num_output):
            mask = (output_labels == j + 1)
            output_patterns.append(_extract_pattern_from_mask(output_grid, mask))

    # Compute similarity matrix
    similarity = np.zeros((num_input, num_output), dtype=np.float32)

    for i in range(num_input):
        input_mask = (input_labels == i + 1)
        input_color = input_colors[i]

        for j in range(num_output):
            output_mask = (output_labels == j + 1)
            output_color = output_colors[j]

            # Must be same color
            if input_color != output_color:
                continue

            # IoU
            iou = compute_iou(input_mask, output_mask)

            # Area similarity
            input_area = input_mask.sum()
            output_area = output_mask.sum()
            if input_area > 0 and output_area > 0:
                area_ratio = min(input_area, output_area) / max(input_area, output_area)
            else:
                area_ratio = 0.0

            # Pattern similarity (if grids provided)
            if use_pattern_matching:
                pattern_sim = _pattern_similarity(input_patterns[i], output_patterns[j])
                # When IoU is high, objects overlap - use IoU + area
                # When IoU is low (objects moved), pattern matching is critical
                if iou > 0.3:
                    # Objects overlap significantly, use traditional approach
                    similarity[i, j] = 0.4 * iou + 0.3 * area_ratio + 0.3 * pattern_sim
                else:
                    # Objects moved - pattern matching is primary discriminator
                    similarity[i, j] = 0.1 * iou + 0.2 * area_ratio + 0.7 * pattern_sim
            else:
                # No grids provided - fall back to original behavior
                similarity[i, j] = 0.5 * iou + 0.5 * area_ratio

    # Greedy matching
    matches = []
    used_input = set()
    used_output = set()

    while True:
        # Find best remaining match
        best_score = -1
        best_i, best_j = -1, -1

        for i in range(num_input):
            if i in used_input:
                continue
            for j in range(num_output):
                if j in used_output:
                    continue
                if similarity[i, j] > best_score:
                    best_score = similarity[i, j]
                    best_i, best_j = i, j

        if best_score < iou_threshold:
            break

        matches.append((best_i, best_j, best_score))
        used_input.add(best_i)
        used_output.add(best_j)

    return matches


def find_object_correspondences_from_objects(
    input_objects: List,
    output_objects: List,
    iou_threshold: float = 0.1
) -> List[Tuple[int, int, float]]:
    """
    Find correspondences between input and output Object instances.

    This is a convenience wrapper for use with ordering_module.Object instances
    or similar dataclasses that have 'color' and 'pixels' attributes.

    Args:
        input_objects: List of input Object instances with color and pixels attributes
        output_objects: List of output Object instances
        iou_threshold: Minimum score threshold for a valid match

    Returns:
        List of (input_idx, output_idx, score) tuples
    """
    if not input_objects or not output_objects:
        return []

    num_input = len(input_objects)
    num_output = len(output_objects)

    # Compute similarity matrix
    similarity = np.zeros((num_input, num_output), dtype=np.float32)

    for i, in_obj in enumerate(input_objects):
        for j, out_obj in enumerate(output_objects):
            # Must be same color
            if in_obj.color != out_obj.color:
                continue

            # IoU using pixel sets
            iou = compute_iou_from_pixels(in_obj.pixels, out_obj.pixels)

            # Area similarity
            in_area = in_obj.area if hasattr(in_obj, 'area') else len(in_obj.pixels)
            out_area = out_obj.area if hasattr(out_obj, 'area') else len(out_obj.pixels)
            if in_area > 0 and out_area > 0:
                area_ratio = min(in_area, out_area) / max(in_area, out_area)
            else:
                area_ratio = 0.0

            # Combined score
            similarity[i, j] = 0.5 * iou + 0.5 * area_ratio

    # Greedy matching
    matches = []
    used_input = set()
    used_output = set()

    while True:
        # Find best remaining match
        best_score = -1
        best_i, best_j = -1, -1

        for i in range(num_input):
            if i in used_input:
                continue
            for j in range(num_output):
                if j in used_output:
                    continue
                if similarity[i, j] > best_score:
                    best_score = similarity[i, j]
                    best_i, best_j = i, j

        if best_score < iou_threshold:
            break

        matches.append((best_i, best_j, best_score))
        used_input.add(best_i)
        used_output.add(best_j)

    return matches


def find_correspondences_from_similarity_matrix(
    similarity_matrix: np.ndarray,
    valid_input_idx: List[int],
    valid_output_idx: List[int],
    threshold: float = 0.3,
    margin: float = 0.1
) -> List[Tuple[int, int, float]]:
    """
    Find matches from a pre-computed similarity matrix.

    Allows many-to-many correspondences when scores are close to the best match.
    For each output slot, includes the best matching input slot plus any others
    whose score is within `margin` of the best. Similarly for each input slot.

    Args:
        similarity_matrix: (num_input, num_output) similarity scores
        valid_input_idx: indices of valid input slots
        valid_output_idx: indices of valid output slots
        threshold: minimum similarity to consider a match
        margin: how close to the best score a match must be to be included
                (e.g., 0.1 means include if score >= best_score - 0.1)

    Returns:
        List of (input_slot_idx, output_slot_idx, similarity_score)
        sorted by similarity score (highest first)
    """
    if similarity_matrix.size == 0:
        return []

    # Compute best scores for each row (input) and column (output)
    best_per_output = similarity_matrix.max(axis=0)  # Best input score for each output
    best_per_input = similarity_matrix.max(axis=1)   # Best output score for each input

    correspondences_set = set()

    # Include a correspondence only if it passes the margin check from BOTH directions:
    # 1. Score is within margin of the best input for this output
    # 2. Score is within margin of the best output for this input
    for in_i, in_idx in enumerate(valid_input_idx):
        for out_i, out_idx in enumerate(valid_output_idx):
            score = similarity_matrix[in_i, out_i]
            if score < threshold:
                continue

            # Check margin from output's perspective (best input for this output)
            if score < best_per_output[out_i] - margin:
                continue

            # Check margin from input's perspective (best output for this input)
            if score < best_per_input[in_i] - margin:
                continue

            correspondences_set.add((in_idx, out_idx, float(score)))

    # Convert to list and sort by similarity score (highest first)
    correspondences = list(correspondences_set)
    correspondences.sort(key=lambda x: x[2], reverse=True)

    return correspondences


def find_correspondences_by_pattern(
    input_grid: np.ndarray,
    output_grid: np.ndarray,
    input_objects: List,
    output_objects: List,
    threshold: float = 0.5
) -> List[Tuple[int, int, float]]:
    """
    Find correspondences by comparing actual pixel patterns.

    This is more robust than color+IoU+area when objects move but maintain
    their internal pattern. Uses pure pattern matching without color or IoU checks.

    Args:
        input_grid: The input grid array
        output_grid: The output grid array
        input_objects: List of input objects with row, col, height, width attributes
        output_objects: List of output objects with row, col, height, width attributes
        threshold: Minimum pattern similarity threshold for a match

    Returns:
        List of (input_idx, output_idx, score) tuples
    """
    if not input_objects or not output_objects:
        return []

    # Extract patterns for all objects
    input_patterns = [
        extract_pattern_from_bbox(input_grid, obj.row, obj.col, obj.height, obj.width)
        for obj in input_objects
    ]
    output_patterns = [
        extract_pattern_from_bbox(output_grid, obj.row, obj.col, obj.height, obj.width)
        for obj in output_objects
    ]

    # Build similarity matrix
    num_in = len(input_objects)
    num_out = len(output_objects)
    similarity = np.zeros((num_in, num_out))

    for i, in_pat in enumerate(input_patterns):
        for j, out_pat in enumerate(output_patterns):
            similarity[i, j] = pattern_similarity(in_pat, out_pat)

    # Greedy matching (best match first)
    matches = []
    used_input = set()
    used_output = set()

    while True:
        # Find best remaining match
        best_score = -1
        best_i, best_j = -1, -1

        for i in range(num_in):
            if i in used_input:
                continue
            for j in range(num_out):
                if j in used_output:
                    continue
                if similarity[i, j] > best_score:
                    best_score = similarity[i, j]
                    best_i, best_j = i, j

        if best_score < threshold:
            break

        matches.append((best_i, best_j, best_score))
        used_input.add(best_i)
        used_output.add(best_j)

    return matches


# =============================================================================
# Visualization (for --puzzle-id mode)
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


def _load_puzzle(puzzle_id: str, data_root: str = "kaggle/combined"):
    """Load a single puzzle from the ARC dataset."""
    import json
    import os

    subsets = ["training", "evaluation", "training2", "evaluation2"]

    for subset in subsets:
        challenges_path = f"{data_root}/arc-agi_{subset}_challenges.json"
        solutions_path = f"{data_root}/arc-agi_{subset}_solutions.json"

        if not os.path.exists(challenges_path):
            continue

        with open(challenges_path) as f:
            puzzles = json.load(f)

        if puzzle_id not in puzzles:
            continue

        puzzle = puzzles[puzzle_id]

        # Load solutions if available
        if os.path.exists(solutions_path):
            with open(solutions_path) as f:
                solutions = json.load(f)
            if puzzle_id in solutions:
                for i, sol in enumerate(solutions[puzzle_id]):
                    if i < len(puzzle["test"]):
                        puzzle["test"][i]["output"] = sol

        return puzzle

    raise ValueError(f"Puzzle '{puzzle_id}' not found in dataset")


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


def _draw_object_outlines(ax, objects: List, color_map: dict, linewidth: float = 2):
    """Draw outlines around objects."""
    for obj in objects:
        if obj.id not in color_map:
            continue

        color = color_map[obj.id]
        # Draw bounding box
        import matplotlib.patches as mpatches
        rect = mpatches.Rectangle(
            (obj.col - 0.5, obj.row - 0.5),
            obj.width, obj.height,
            linewidth=linewidth, edgecolor=color, facecolor='none'
        )
        ax.add_patch(rect)

        # Add object ID label
        center_row = obj.row + obj.height / 2
        center_col = obj.col + obj.width / 2
        ax.annotate(f'{obj.id}', (center_col, center_row),
                   color='white', fontsize=10, fontweight='bold',
                   ha='center', va='center',
                   bbox=dict(boxstyle='circle', facecolor=color, alpha=0.8))


class CorrespondenceNavigator:
    """Interactive navigator for viewing correspondence examples."""

    def __init__(self, puzzle_id: str, examples_data: List[dict]):
        self.puzzle_id = puzzle_id
        self.examples_data = examples_data
        self.current_idx = 0
        self.fig = None

    def draw_example(self, ex_idx: int):
        """Draw a single example."""
        import matplotlib.pyplot as plt
        from matplotlib.patches import FancyArrowPatch

        if self.fig is not None:
            self.fig.clear()
        else:
            self.fig = plt.figure(figsize=(14, 8))

        data = self.examples_data[ex_idx]
        input_grid = data['input_grid']
        output_grid = data['output_grid']
        input_objects = data['input_objects']
        output_objects = data['output_objects']
        correspondences = data['correspondences']

        H_in, W_in = input_grid.shape
        H_out, W_out = output_grid.shape

        # Assign colors to correspondence pairs based on input object color
        input_color_map = {}
        output_color_map = {}
        for in_idx, out_idx, score in correspondences:
            in_obj = input_objects[in_idx]
            color = ARC_COLORS[in_obj.color]
            input_color_map[in_idx] = color
            output_color_map[out_idx] = color

        # Layout: input grid, arrows, output grid
        gs = self.fig.add_gridspec(1, 3, width_ratios=[1, 0.3, 1], wspace=0.15)

        # Input grid
        ax_in = self.fig.add_subplot(gs[0, 0])
        _draw_grid(ax_in, input_grid, f"Input ({H_in}x{W_in}) - {len(input_objects)} objects")
        _draw_object_outlines(ax_in, input_objects, input_color_map, linewidth=3)

        # Arrow space
        ax_arrows = self.fig.add_subplot(gs[0, 1])
        ax_arrows.set_xlim(0, 1)
        ax_arrows.set_ylim(0, 1)
        ax_arrows.axis('off')

        # Draw arrows for correspondences
        for i, (in_idx, out_idx, score) in enumerate(correspondences):
            in_obj = input_objects[in_idx]
            color = ARC_COLORS[in_obj.color]

            y_pos = 0.9 - (i * 0.12) % 0.8

            arrow = FancyArrowPatch(
                (0.1, y_pos),
                (0.9, y_pos),
                connectionstyle="arc3,rad=0.0",
                arrowstyle="->,head_width=0.1,head_length=0.08",
                color=color,
                linewidth=2,
                alpha=0.8
            )
            ax_arrows.add_patch(arrow)
            ax_arrows.text(0.5, y_pos + 0.03, f"In{in_idx}→Out{out_idx}: {score:.2f}",
                          ha='center', va='bottom', fontsize=8, color=color)

        # Output grid
        ax_out = self.fig.add_subplot(gs[0, 2])
        _draw_grid(ax_out, output_grid, f"Output ({H_out}x{W_out}) - {len(output_objects)} objects")
        _draw_object_outlines(ax_out, output_objects, output_color_map, linewidth=3)

        n_examples = len(self.examples_data)
        self.fig.suptitle(
            f"Correspondence: {self.puzzle_id} - Example {ex_idx + 1}/{n_examples}\n"
            f"Found {len(correspondences)} correspondences  [← / → to navigate, q to quit]",
            fontsize=12, fontweight='bold'
        )
        plt.subplots_adjust(top=0.88)
        self.fig.canvas.draw()

    def on_key(self, event):
        """Handle keyboard navigation."""
        import matplotlib.pyplot as plt
        if event.key == 'right' or event.key == 'n':
            self.current_idx = (self.current_idx + 1) % len(self.examples_data)
            self.draw_example(self.current_idx)
        elif event.key == 'left' or event.key == 'p':
            self.current_idx = (self.current_idx - 1) % len(self.examples_data)
            self.draw_example(self.current_idx)
        elif event.key == 'q':
            plt.close(self.fig)

    def show(self):
        """Display the interactive navigator."""
        import matplotlib.pyplot as plt
        self.draw_example(0)
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        plt.show()


def visualize_puzzle_correspondences(puzzle_id: str, data_root: str = "kaggle/combined",
                                      iou_threshold: float = 0.1):
    """
    Visualize object correspondences for a specific puzzle.

    Args:
        puzzle_id: The ARC puzzle ID (e.g., "1990f7a8")
        data_root: Path to puzzle data
        iou_threshold: Minimum score for correspondence matching
    """
    from object_module import extract_objects_from_grid

    print(f"Loading puzzle: {puzzle_id}")
    puzzle = _load_puzzle(puzzle_id, data_root)
    print(f"Found {len(puzzle['train'])} training examples")

    examples_data = []

    for i, example in enumerate(puzzle['train']):
        if 'output' not in example:
            continue

        input_grid = np.array(example['input'], dtype=np.int64)
        output_grid = np.array(example['output'], dtype=np.int64)

        print(f"\nProcessing example {i + 1}...")
        print(f"  Input shape: {input_grid.shape}, Output shape: {output_grid.shape}")

        # Extract objects
        input_objects = extract_objects_from_grid(input_grid)
        output_objects = extract_objects_from_grid(output_grid)

        print(f"  Input objects: {len(input_objects)}, Output objects: {len(output_objects)}")

        # Find correspondences using pattern matching
        correspondences = find_correspondences_by_pattern(
            input_grid, output_grid, input_objects, output_objects,
            threshold=iou_threshold
        )

        print(f"  Found {len(correspondences)} correspondences:")
        for in_idx, out_idx, score in correspondences:
            in_obj = input_objects[in_idx]
            out_obj = output_objects[out_idx]
            print(f"    In[{in_idx}] (color={in_obj.color}, pos=({in_obj.row},{in_obj.col})) -> "
                  f"Out[{out_idx}] (color={out_obj.color}, pos=({out_obj.row},{out_obj.col})) "
                  f"score={score:.3f}")

        examples_data.append({
            'input_grid': input_grid,
            'output_grid': output_grid,
            'input_objects': input_objects,
            'output_objects': output_objects,
            'correspondences': correspondences,
        })

    # Show interactive visualization
    print("\nGenerating visualization...")
    navigator = CorrespondenceNavigator(puzzle_id, examples_data)
    navigator.show()


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Correspondence Module for ARC Puzzles",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python correspondence_module.py --puzzle-id 1990f7a8
    python correspondence_module.py --puzzle-id 009d5c81 --threshold 0.3
        """
    )

    parser.add_argument("--puzzle-id", type=str, required=True,
                        help="ARC puzzle ID to visualize (e.g., 1990f7a8)")
    parser.add_argument("--data-root", type=str, default="kaggle/combined",
                        help="Path to puzzle data")
    parser.add_argument("--threshold", type=float, default=0.1,
                        help="Minimum score threshold for correspondences (default: 0.1)")

    args = parser.parse_args()

    visualize_puzzle_correspondences(
        args.puzzle_id,
        data_root=args.data_root,
        iou_threshold=args.threshold
    )
