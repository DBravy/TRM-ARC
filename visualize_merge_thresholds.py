#!/usr/bin/env python3
"""
Visualize how different merge thresholds affect component grouping in ARC puzzles.

Usage:
    python visualize_merge_thresholds.py --single-puzzle 00d62c1b
    python visualize_merge_thresholds.py --single-puzzle 00d62c1b --thresholds 0.5 1.0 1.5 2.0 2.5 3.0
    python visualize_merge_thresholds.py --single-puzzle 00d62c1b --no-background-detection
"""

import argparse
import json
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
from scipy import ndimage

# ARC color palette (same as used in the main code)
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
    '#870C25',  # 9: brown/maroon
]
NUM_COLORS = 10


def load_puzzles(dataset_name: str, data_root: str = "kaggle/combined") -> dict:
    """Load puzzles from the dataset."""
    config = {
        "arc-agi-1": {"subsets": ["training", "evaluation"]},
        "arc-agi-2": {"subsets": ["training2", "evaluation2"]},
    }

    all_puzzles = {}

    for subset in config[dataset_name]["subsets"]:
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
                    for i, sol in enumerate(solutions[puzzle_id]):
                        if i < len(puzzles[puzzle_id]["test"]):
                            puzzles[puzzle_id]["test"][i]["output"] = sol

        all_puzzles.update(puzzles)

    return all_puzzles


def detect_background_color(grid: np.ndarray, min_coverage: float = 0.1) -> int | None:
    """
    Detect the background color using the largest edge-connected component heuristic.
    
    Args:
        grid: The color grid (H, W) with integer color values
        min_coverage: Minimum fraction of grid that must be covered to be considered background
        
    Returns:
        The background color (0-9), or None if no clear background is detected
    """
    H, W = grid.shape
    total_cells = H * W
    
    # Get colors that touch any edge
    edge_colors = set()
    edge_colors.update(grid[0, :].tolist())      # top
    edge_colors.update(grid[H-1, :].tolist())    # bottom
    edge_colors.update(grid[:, 0].tolist())      # left
    edge_colors.update(grid[:, W-1].tolist())    # right
    
    best_color = None
    best_size = 0
    
    for color in edge_colors:
        mask = (grid == color)
        labeled, num_features = ndimage.label(mask)
        
        # Find components that touch the edge
        for comp_id in range(1, num_features + 1):
            comp_mask = (labeled == comp_id)
            
            touches_edge = (
                comp_mask[0, :].any() or comp_mask[H-1, :].any() or
                comp_mask[:, 0].any() or comp_mask[:, W-1].any()
            )
            
            if touches_edge:
                size = comp_mask.sum()
                if size > best_size:
                    best_size = size
                    best_color = color
    
    # Only return as background if it covers enough of the grid
    if best_color is not None and best_size / total_cells >= min_coverage:
        return best_color
    
    return None


def compute_exterior_mask(grid: np.ndarray, background_color: int) -> np.ndarray:
    """
    Compute which pixels of the background color are "exterior" (reachable from edges)
    using 4-connectivity. This allows diagonal lines to act as enclosing barriers.
    
    Args:
        grid: The color grid (H, W)
        background_color: The detected background color
        
    Returns:
        Boolean mask (H, W) where True = exterior background pixel
    """
    H, W = grid.shape
    
    # Create mask of background-colored pixels
    bg_mask = (grid == background_color)
    
    # Use 4-connectivity to find what's reachable from edges
    # Start by marking edge pixels of background color
    seed = np.zeros((H, W), dtype=bool)
    seed[0, :] = bg_mask[0, :]       # top edge
    seed[H-1, :] = bg_mask[H-1, :]   # bottom edge
    seed[:, 0] = bg_mask[:, 0]       # left edge
    seed[:, W-1] = bg_mask[:, W-1]   # right edge
    
    # 4-connectivity structure (no diagonals)
    structure_4conn = np.array([[0, 1, 0],
                                 [1, 1, 1],
                                 [0, 1, 0]], dtype=np.int32)
    
    # Flood fill from edge seeds within background mask using 4-connectivity
    # This finds all background pixels reachable from edges without crossing diagonals
    exterior_mask = ndimage.binary_dilation(seed, structure=structure_4conn, 
                                             iterations=-1, mask=bg_mask)
    
    return exterior_mask


def compute_connected_components(color_grid: np.ndarray, background_color: int | None = None):
    """
    Compute connected components for a color grid.

    Args:
        color_grid: The color grid (H, W)
        background_color: If provided, this color uses 4-connectivity (so diagonal 
                         enclosures create separate regions), while other colors 
                         use 8-connectivity (so diagonal lines are solid shapes).
                         Components of this color are marked as background only if 
                         they are "exterior" (reachable from edges via 4-connectivity).
        
    Returns:
        component_labels: (H, W) component IDs (0 = no component, 1+ = component IDs)
        component_colors: list of colors for each component
        component_bboxes: list of (min_row, min_col, max_row, max_col) for each component
        component_is_background: list of booleans indicating if each component is background
    """
    H, W = color_grid.shape
    all_labels = np.zeros((H, W), dtype=np.int32)
    component_colors = []
    component_bboxes = []
    component_is_background = []
    component_id = 0

    # 8-connectivity structure (includes diagonals) for foreground objects
    structure_8conn = np.array([[1, 1, 1],
                                 [1, 1, 1],
                                 [1, 1, 1]], dtype=np.int32)
    
    # 4-connectivity structure (no diagonals) for background color
    structure_4conn = np.array([[0, 1, 0],
                                 [1, 1, 1],
                                 [0, 1, 0]], dtype=np.int32)
    
    # Compute exterior mask if we have a background color
    # This uses 4-connectivity to determine what's reachable from edges
    exterior_mask = None
    if background_color is not None:
        exterior_mask = compute_exterior_mask(color_grid, background_color)

    for c in range(NUM_COLORS):
        mask = (color_grid == c)
        if not mask.any():
            continue

        # Use 4-connectivity for background color, 8-connectivity for others
        if c == background_color:
            labeled, num_features = ndimage.label(mask, structure=structure_4conn)
        else:
            labeled, num_features = ndimage.label(mask, structure=structure_8conn)

        for comp in range(1, num_features + 1):
            component_id += 1
            comp_mask = (labeled == comp)
            all_labels[comp_mask] = component_id
            component_colors.append(c)
            
            # Check if this component is part of the background
            # A background component must be the background color AND be exterior
            # (reachable from edges via 4-connectivity)
            is_bg = False
            if background_color is not None and c == background_color and exterior_mask is not None:
                # Check if any pixel of this component is in the exterior mask
                is_bg = (comp_mask & exterior_mask).any()
            component_is_background.append(is_bg)

            rows, cols = np.where(comp_mask)
            bbox = (rows.min(), cols.min(), rows.max(), cols.max())
            component_bboxes.append(bbox)

    return all_labels, component_colors, component_bboxes, component_is_background


def compute_pairwise_affinity(component_labels: np.ndarray,
                               component_colors: list,
                               component_bboxes: list,
                               component_is_background: list = None,
                               w_color: float = 1.0,
                               w_adjacent: float = 1.0,
                               w_bbox: float = 1.0):
    """
    Compute pairwise affinity matrix between components.

    Affinity is based on:
    - Same color: +w_color
    - Spatial adjacency (8-connected): +w_adjacent
    - Bounding box containment: +w_bbox
    
    Background components only have affinity with other background components.
    """
    H, W = component_labels.shape
    num_components = len(component_colors)

    if num_components == 0:
        return np.zeros((0, 0), dtype=np.float32)
    
    # Default: no components are background
    if component_is_background is None:
        component_is_background = [False] * num_components

    affinity = np.zeros((num_components, num_components), dtype=np.float32)

    # Precompute adjacency pairs
    adjacent_pairs = set()
    for row in range(H):
        for col in range(W):
            comp_id = component_labels[row, col]
            if comp_id == 0:
                continue
            comp_idx = comp_id - 1

            for dr, dc in [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]:
                nr, nc = row + dr, col + dc
                if 0 <= nr < H and 0 <= nc < W:
                    neighbor_id = component_labels[nr, nc]
                    if neighbor_id != 0 and neighbor_id != comp_id:
                        neighbor_idx = neighbor_id - 1
                        adjacent_pairs.add((comp_idx, neighbor_idx))

    for i in range(num_components):
        for j in range(i + 1, num_components):
            # Key change: no affinity between background and non-background components
            i_is_bg = component_is_background[i]
            j_is_bg = component_is_background[j]
            
            if i_is_bg != j_is_bg:
                # One is background, one is not - no affinity
                affinity[i, j] = 0.0
                affinity[j, i] = 0.0
                continue
            
            score = 0.0

            # Same color
            if component_colors[i] == component_colors[j]:
                score += w_color

            # Adjacency
            if (i, j) in adjacent_pairs or (j, i) in adjacent_pairs:
                score += w_adjacent

            # Bounding box containment
            bbox_i = component_bboxes[i]
            bbox_j = component_bboxes[j]

            i_contains_j = (bbox_i[0] <= bbox_j[0] and bbox_i[1] <= bbox_j[1] and
                           bbox_i[2] >= bbox_j[2] and bbox_i[3] >= bbox_j[3])
            j_contains_i = (bbox_j[0] <= bbox_i[0] and bbox_j[1] <= bbox_i[1] and
                           bbox_j[2] >= bbox_i[2] and bbox_j[3] >= bbox_i[3])

            if i_contains_j or j_contains_i:
                score += w_bbox

            affinity[i, j] = score
            affinity[j, i] = score

    return affinity


def merge_components(affinity: np.ndarray, num_components: int, merge_threshold: float):
    """
    Merge components based on affinity using Union-Find.

    Returns:
        group_ids: array mapping component index to group ID
    """
    if num_components == 0:
        return np.array([], dtype=np.int32)

    parent = list(range(num_components))

    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py

    # Merge components where affinity > threshold
    for i in range(num_components):
        for j in range(i + 1, num_components):
            if affinity[i, j] > merge_threshold:
                union(i, j)

    # Compress paths and assign group IDs
    root_to_group = {}
    group_ids = np.zeros(num_components, dtype=np.int32)
    next_group = 0

    for i in range(num_components):
        root = find(i)
        if root not in root_to_group:
            root_to_group[root] = next_group
            next_group += 1
        group_ids[i] = root_to_group[root]

    return group_ids


def create_group_visualization(component_labels: np.ndarray, group_ids: np.ndarray):
    """Create a visualization grid showing merged groups."""
    H, W = component_labels.shape
    group_grid = np.full((H, W), -1, dtype=np.int32)

    for row in range(H):
        for col in range(W):
            comp_id = component_labels[row, col]
            if comp_id > 0:
                comp_idx = comp_id - 1
                group_grid[row, col] = group_ids[comp_idx]

    return group_grid


def plot_grid(ax, grid, title, cmap=None, show_grid_lines=True):
    """Plot a single grid with optional colormap."""
    H, W = grid.shape

    if cmap is None:
        # Use ARC colors
        cmap = mcolors.ListedColormap(ARC_COLORS)
        vmin, vmax = 0, 9
    else:
        vmin, vmax = grid.min(), grid.max()

    ax.imshow(grid, cmap=cmap, vmin=vmin, vmax=vmax, interpolation='nearest')
    ax.set_title(title, fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])

    if show_grid_lines:
        ax.set_xticks(np.arange(-0.5, W, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, H, 1), minor=True)
        ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.5)


def visualize_puzzle_thresholds(puzzle_id: str, puzzle_data: dict, thresholds: list, 
                                 output_path: str = None, use_background_detection: bool = True):
    """
    Visualize component grouping at different merge thresholds for a puzzle.
    """
    examples = puzzle_data.get("test", [])
    if not examples:
        print(f"No test examples found for puzzle {puzzle_id}")
        return

    # Use first test example
    example = examples[0]
    input_grid = np.array(example["input"], dtype=np.int32)
    output_grid = np.array(example["output"], dtype=np.int32) if "output" in example else None

    # Detect background color
    background_color = None
    if use_background_detection:
        background_color = detect_background_color(input_grid)
        if background_color is not None:
            print(f"Detected background color: {background_color}")
        else:
            print("No clear background detected")

    # Compute components and affinity for input grid
    component_labels, component_colors, component_bboxes, component_is_background = \
        compute_connected_components(input_grid, background_color)
    affinity = compute_pairwise_affinity(component_labels, component_colors, component_bboxes, 
                                          component_is_background)
    num_components = len(component_colors)
    num_background_components = sum(component_is_background)

    # Create figure
    n_thresholds = len(thresholds)
    n_cols = n_thresholds + 2  # Original + Components + thresholds
    n_rows = 2 if output_grid is not None else 1

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 3 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)

    bg_str = f" | bg={background_color}" if background_color is not None else ""
    fig.suptitle(f"Puzzle: {puzzle_id} (test input) | {num_components} components ({num_background_components} bg){bg_str}", 
                 fontsize=14)

    # Create a colormap for groups (distinct colors)
    group_cmap = plt.cm.get_cmap('tab20', 20)

    # Row 0: Test input grid analysis
    plot_grid(axes[0, 0], input_grid, "Test Input (colors)")

    # Connected components
    comp_display = component_labels.copy()
    plot_grid(axes[0, 1], comp_display, f"Components ({num_components})",
              cmap=group_cmap, show_grid_lines=True)

    # Different thresholds
    for i, threshold in enumerate(thresholds):
        group_ids = merge_components(affinity, num_components, threshold)
        group_grid = create_group_visualization(component_labels, group_ids)
        num_groups = len(np.unique(group_ids)) if len(group_ids) > 0 else 0

        # Create masked array for visualization (background = -1)
        masked_grid = np.ma.masked_where(group_grid < 0, group_grid)

        ax = axes[0, i + 2]
        ax.imshow(masked_grid, cmap=group_cmap, vmin=0, vmax=19, interpolation='nearest')
        ax.set_title(f"Threshold {threshold}\n({num_groups} groups)", fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])

        H, W = group_grid.shape
        ax.set_xticks(np.arange(-0.5, W, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, H, 1), minor=True)
        ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.5)

    # Row 1: Test output grid analysis (if available)
    if output_grid is not None:
        # Detect background for output grid separately
        out_background_color = None
        if use_background_detection:
            out_background_color = detect_background_color(output_grid)
            
        out_component_labels, out_component_colors, out_component_bboxes, out_component_is_background = \
            compute_connected_components(output_grid, out_background_color)
        out_affinity = compute_pairwise_affinity(out_component_labels, out_component_colors, 
                                                  out_component_bboxes, out_component_is_background)
        out_num_components = len(out_component_colors)

        plot_grid(axes[1, 0], output_grid, "Test Output (colors)")
        plot_grid(axes[1, 1], out_component_labels, f"Components ({out_num_components})",
                  cmap=group_cmap, show_grid_lines=True)

        for i, threshold in enumerate(thresholds):
            group_ids = merge_components(out_affinity, out_num_components, threshold)
            group_grid = create_group_visualization(out_component_labels, group_ids)
            num_groups = len(np.unique(group_ids)) if len(group_ids) > 0 else 0

            masked_grid = np.ma.masked_where(group_grid < 0, group_grid)

            ax = axes[1, i + 2]
            ax.imshow(masked_grid, cmap=group_cmap, vmin=0, vmax=19, interpolation='nearest')
            ax.set_title(f"Threshold {threshold}\n({num_groups} groups)", fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])

            H, W = group_grid.shape
            ax.set_xticks(np.arange(-0.5, W, 1), minor=True)
            ax.set_yticks(np.arange(-0.5, H, 1), minor=True)
            ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.5)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {output_path}")

    plt.show()

    # Print affinity matrix summary
    print(f"\n=== Affinity Analysis for {puzzle_id} ===")
    print(f"Number of connected components: {num_components}")
    print(f"Background components: {num_background_components}")
    if background_color is not None:
        print(f"Background color: {background_color}")
    
    if num_components > 0 and num_components <= 20:
        print("\nAffinity matrix (non-zero values):")
        for i in range(num_components):
            for j in range(i + 1, num_components):
                if affinity[i, j] > 0:
                    bg_i = " [BG]" if component_is_background[i] else ""
                    bg_j = " [BG]" if component_is_background[j] else ""
                    print(f"  Component {i+1} (color {component_colors[i]}{bg_i}) <-> "
                          f"Component {j+1} (color {component_colors[j]}{bg_j}): {affinity[i, j]:.1f}")

    print("\nGroups at each threshold:")
    for threshold in thresholds:
        group_ids = merge_components(affinity, num_components, threshold)
        num_groups = len(np.unique(group_ids)) if len(group_ids) > 0 else 0
        print(f"  Threshold {threshold}: {num_groups} groups")


def main():
    parser = argparse.ArgumentParser(description="Visualize merge threshold effects on ARC puzzles")
    parser.add_argument("--single-puzzle", type=str, required=True,
                        help="Puzzle ID to visualize")
    parser.add_argument("--dataset", type=str, default="arc-agi-1",
                        choices=["arc-agi-1", "arc-agi-2"])
    parser.add_argument("--data-root", type=str, default="kaggle/combined")
    parser.add_argument("--thresholds", type=float, nargs="+",
                        default=[0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
                        help="Merge thresholds to visualize")
    parser.add_argument("--output", type=str, default=None,
                        help="Output path to save the visualization")
    parser.add_argument("--no-background-detection", action="store_true",
                        help="Disable automatic background detection")

    args = parser.parse_args()

    # Load puzzles
    print(f"Loading puzzles from {args.dataset}...")
    puzzles = load_puzzles(args.dataset, args.data_root)

    if args.single_puzzle not in puzzles:
        print(f"Error: Puzzle '{args.single_puzzle}' not found!")
        print(f"Available puzzles: {len(puzzles)}")
        return

    puzzle_data = puzzles[args.single_puzzle]
    print(f"Loaded puzzle {args.single_puzzle}")
    print(f"Thresholds to visualize: {args.thresholds}")
    print(f"Background detection: {'disabled' if args.no_background_detection else 'enabled'}")

    visualize_puzzle_thresholds(
        args.single_puzzle, 
        puzzle_data, 
        args.thresholds, 
        args.output,
        use_background_detection=not args.no_background_detection
    )


if __name__ == "__main__":
    main()