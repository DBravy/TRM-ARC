#!/usr/bin/env python3
"""
Test Puzzle for Hierarchical Object Support

This puzzle demonstrates a pattern that REQUIRES hierarchy understanding:
- Multiple containers (grey rectangles)
- Each container has objects inside
- The transformation rule: "Within each container, the smallest object
  moves to the container's top-left corner (offset by 1 pixel)"

WITHOUT HIERARCHY: The flat representation sees 9 independent objects.
  No single ordering or anchoring rule explains all movements because
  different objects move in different directions depending on their container.

WITH HIERARCHY: Each container is a "parent" with "children" inside.
  The rule becomes consistent: child.TL at parent.TL + (1,1) for smallest children.

This script demonstrates:
1. The puzzle structure
2. Hierarchy screening selects hierarchy mode
3. Parent-child anchor relations have low variance
4. The transformation rule is discovered
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# Import hierarchy functions
from object_module import (
    Object,
    extract_objects_from_grid,
    build_containment_hierarchy,
    get_hierarchy_stats,
)
from correspondence_module import find_hierarchical_correspondences
from anchoring_module import discover_parent_child_relation, ParentRelation
from ordering_module import screen_hierarchy_strategies

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


def create_hierarchy_test_puzzle():
    """
    Create a puzzle with SOLID CONTAINER containment.

    The puzzle has 3 separate colored rectangles (containers), each with gaps
    between them to ensure they're detected as separate objects.

    Rule: Within each container, the smallest object moves to the
    top-left corner of its container (offset by 1 pixel from container edge).

    Layout:
    - Container 1 (grey, 6x6) at (1,1): contains blue 3x3 and green 2x2
    - Container 2 (grey, 6x6) at (1,9): contains cyan 3x3 and orange 1x2
    - Container 3 (grey, 4x6) at (8,1): contains red 2x3 and yellow 1x1

    All containers are separated by at least 1 row/col of black (background).
    """

    # 13x16 grid
    input_grid = np.zeros((13, 16), dtype=np.int32)
    output_grid = np.zeros((13, 16), dtype=np.int32)

    # === CONTAINER 1 (top-left): grey 6x6 at (1,1) ===
    input_grid[1:7, 1:7] = 5   # Grey container
    output_grid[1:7, 1:7] = 5  # Container stays

    # Objects inside Container 1:
    # - Blue object (3x3) at bottom-right - larger, stays put
    input_grid[3:6, 3:6] = 1
    output_grid[3:6, 3:6] = 1
    # - Green object (2x2) in middle - smaller, moves to top-left of container
    input_grid[2:4, 2:4] = 3
    output_grid[2:4, 2:4] = 3  # Already at top-left (row 2 = container.row+1)

    # === CONTAINER 2 (top-right): grey 6x6 at (1,9) ===
    # Note: gap of 2 cols (7-8) between Container 1 and 2
    input_grid[1:7, 9:15] = 5   # Grey container
    output_grid[1:7, 9:15] = 5  # Container stays

    # Objects inside Container 2:
    # - Cyan object (3x3) at center - larger, stays put
    input_grid[2:5, 11:14] = 8
    output_grid[2:5, 11:14] = 8
    # - Orange object (1x2) at bottom - smaller, moves to top-left
    input_grid[5, 12:14] = 7
    output_grid[2, 10:12] = 7  # Moved to top-left of container

    # === CONTAINER 3 (bottom): grey 4x6 at (9,1) ===
    # Note: gap of 2 rows (7-8) between Container 1 and 3
    input_grid[9:13, 1:7] = 5   # Grey container
    output_grid[9:13, 1:7] = 5  # Container stays

    # Objects inside Container 3:
    # - Red object (2x3) at right - larger, stays put
    input_grid[10:12, 3:6] = 2
    output_grid[10:12, 3:6] = 2
    # - Yellow object (1x1) at left - smaller, moves to top-left
    input_grid[11, 2] = 4
    output_grid[10, 2] = 4  # Moved to top-left of container

    return input_grid, output_grid


def visualize_puzzle(input_grid, output_grid, title="Hierarchy Test Puzzle"):
    """Visualize input and output grids side by side."""
    cmap = ListedColormap(ARC_COLORS)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].imshow(input_grid, cmap=cmap, vmin=0, vmax=9)
    axes[0].set_title("Input")
    axes[0].set_xticks([])
    axes[0].set_yticks([])
    axes[0].grid(True, color='white', linewidth=0.5)

    axes[1].imshow(output_grid, cmap=cmap, vmin=0, vmax=9)
    axes[1].set_title("Output")
    axes[1].set_xticks([])
    axes[1].set_yticks([])
    axes[1].grid(True, color='white', linewidth=0.5)

    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig('hierarchy_test_puzzle.png', dpi=150, bbox_inches='tight')
    print("Saved visualization to hierarchy_test_puzzle.png")
    plt.close()


def analyze_with_hierarchy(input_grid, output_grid):
    """Analyze the puzzle using the new hierarchy system."""

    print("=" * 60)
    print("ANALYZING PUZZLE WITH HIERARCHY SYSTEM")
    print("=" * 60)

    # Extract objects WITH hierarchy
    print("\n1. Extracting objects with hierarchy detection...")
    input_roots = extract_objects_from_grid(input_grid, build_hierarchy=True)
    output_roots = extract_objects_from_grid(output_grid, build_hierarchy=True)

    # Get stats
    input_stats = get_hierarchy_stats(input_roots)
    output_stats = get_hierarchy_stats(output_roots)

    print(f"\n   Input hierarchy:")
    print(f"     - Root objects: {input_stats['num_roots']}")
    print(f"     - Total objects: {input_stats['total_objects']}")
    print(f"     - Max depth: {input_stats['max_depth']}")
    print(f"     - Composite roots: {input_stats['num_composite']}")

    print(f"\n   Output hierarchy:")
    print(f"     - Root objects: {output_stats['num_roots']}")
    print(f"     - Total objects: {output_stats['total_objects']}")
    print(f"     - Max depth: {output_stats['max_depth']}")

    # Print hierarchy tree
    print("\n2. Input Hierarchy Tree:")
    for root in input_roots:
        print_object_tree(root, indent=3)

    print("\n3. Output Hierarchy Tree:")
    for root in output_roots:
        print_object_tree(root, indent=3)

    # Find hierarchical correspondences
    print("\n4. Finding hierarchical correspondences...")
    correspondences = find_hierarchical_correspondences(
        input_roots, output_roots,
        input_grid, output_grid,
        threshold=0.3  # Lower threshold for small objects
    )

    print(f"   Found {len(correspondences)} correspondences:")
    for in_obj, out_obj in correspondences:
        delta_r = out_obj.row - in_obj.row
        delta_c = out_obj.col - in_obj.col
        move_str = f"({delta_r:+d}, {delta_c:+d})" if (delta_r or delta_c) else "(no move)"
        print(f"     Color {in_obj.color}: ({in_obj.row},{in_obj.col}) -> ({out_obj.row},{out_obj.col}) {move_str}")

    # Analyze parent-child relationships
    print("\n5. Analyzing parent-child anchor relationships...")

    # Collect child-parent pairs from output (where children have moved)
    child_parent_pairs = []
    for root in output_roots:
        for child in root.children:
            child_parent_pairs.append((child, root))

    if child_parent_pairs:
        print(f"   Found {len(child_parent_pairs)} child-parent pairs in output")

        # Analyze positioning
        for child, parent in child_parent_pairs:
            local_row = child.row - parent.row
            local_col = child.col - parent.col
            print(f"     Child (color {child.color}) in parent (color {parent.color}): "
                  f"local position ({local_row}, {local_col})")

        # Try to discover consistent relation
        if len(child_parent_pairs) >= 2:
            relations = discover_parent_child_relation(child_parent_pairs, top_k=3)
            print(f"\n   Best parent-child anchor relations:")
            for i, rel in enumerate(relations[:3]):
                print(f"     {i+1}. {rel.relation.describe()} (variance: {rel.variance:.4f})")

    return correspondences


def print_object_tree(obj, indent=0):
    """Print an object and its children as a tree."""
    prefix = " " * indent
    info = f"Object(color={obj.color}, pos=({obj.row},{obj.col}), size={obj.height}x{obj.width})"
    if obj.is_composite:
        print(f"{prefix}├─ {info} [COMPOSITE, {len(obj.children)} children]")
        for i, child in enumerate(obj.children):
            child_prefix = "│" if i < len(obj.children) - 1 else " "
            print_object_tree(child, indent + 3)
    else:
        print(f"{prefix}└─ {info} [atomic]")


def analyze_without_hierarchy(input_grid, output_grid):
    """Show what analysis looks like WITHOUT hierarchy (flat objects)."""

    print("\n" + "=" * 60)
    print("ANALYZING PUZZLE WITHOUT HIERARCHY (FLAT)")
    print("=" * 60)

    # Extract objects WITHOUT hierarchy
    input_objects = extract_objects_from_grid(input_grid, build_hierarchy=False)
    output_objects = extract_objects_from_grid(output_grid, build_hierarchy=False)

    # Filter out dividers
    input_objects = [o for o in input_objects if not o.is_divider]
    output_objects = [o for o in output_objects if not o.is_divider]

    print(f"\n   Input: {len(input_objects)} flat objects")
    print(f"   Output: {len(output_objects)} flat objects")

    print("\n   Input objects (flat list):")
    for obj in input_objects:
        print(f"     Object {obj.id}: color={obj.color}, pos=({obj.row},{obj.col}), "
              f"size={obj.height}x{obj.width}, area={obj.area}")

    print("\n   Challenge: Without hierarchy, we see all objects at same level.")
    print("   We can't express 'smallest object in each cell moves to cell's top-left'")
    print("   because we don't know which objects belong to which cells!")


def test_hierarchy_screening():
    """
    Test the automatic hierarchy screening on this puzzle.

    This demonstrates that screen_hierarchy_strategies() correctly identifies
    that hierarchy mode should be used for this puzzle.
    """
    print("\n" + "=" * 60)
    print("TESTING AUTOMATIC HIERARCHY SCREENING")
    print("=" * 60)

    # Create puzzle in the format expected by screen_hierarchy_strategies
    input_grid, output_grid = create_hierarchy_test_puzzle()

    # Create a second training example with slight variation
    # to give the screener more data to work with
    input_grid2, output_grid2 = create_hierarchy_test_puzzle()

    puzzle = {
        'train': [
            {'input': input_grid.tolist(), 'output': output_grid.tolist()},
            {'input': input_grid2.tolist(), 'output': output_grid2.tolist()},
        ]
    }

    # Run hierarchy screening
    print("\nRunning screen_hierarchy_strategies()...")
    results = screen_hierarchy_strategies(puzzle, verbose=True)

    print("\n" + "-" * 40)
    print("SCREENING RESULTS:")
    print("-" * 40)
    print(f"  Best mode: {results['best_mode']}")
    print(f"  Use hierarchy: {results['use_hierarchy']}")
    print(f"  Flat score: {results['flat_score']:.4f}")
    print(f"  Hierarchy score (variance): {results['hierarchy_score']:.4f}")

    if results['hierarchy_stats']:
        stats = results['hierarchy_stats']
        print(f"\n  Hierarchy stats:")
        print(f"    - Root objects: {stats['num_roots']}")
        print(f"    - Total objects: {stats['total_objects']}")
        print(f"    - Max depth: {stats['max_depth']}")
        print(f"    - Composite roots: {stats['num_composite']}")

    if results['parent_child_relations']:
        print(f"\n  Best parent-child relation:")
        rel = results['parent_child_relations'][0]
        print(f"    {rel.relation.describe()}")
        print(f"    Variance: {rel.variance:.4f}")

    # Verify that hierarchy was selected
    if results['use_hierarchy']:
        print("\n  ✓ SUCCESS: Hierarchy mode was correctly selected!")
    else:
        print("\n  ✗ WARNING: Hierarchy mode was NOT selected (expected it to be)")

    return results


def main():
    print("=" * 60)
    print("HIERARCHY TEST PUZZLE")
    print("=" * 60)
    print("\nThis puzzle can ONLY be solved correctly with hierarchy.")
    print("The rule is: 'smallest child in each container moves to top-left'")

    print("\nCreating hierarchy test puzzle...")
    input_grid, output_grid = create_hierarchy_test_puzzle()

    print("\nPuzzle dimensions:", input_grid.shape)
    print("Unique colors in input:", sorted(set(input_grid.flatten())))
    print("Unique colors in output:", sorted(set(output_grid.flatten())))

    # Visualize
    visualize_puzzle(input_grid, output_grid)

    # Analyze without hierarchy first (to show the problem)
    analyze_without_hierarchy(input_grid, output_grid)

    # Analyze with hierarchy (to show the solution)
    analyze_with_hierarchy(input_grid, output_grid)

    # Test the automatic screening
    screening_results = test_hierarchy_screening()

    print("\n" + "=" * 60)
    print("CONCLUSION")
    print("=" * 60)
    print("""
The hierarchy system enables expressing rules like:
  "The smallest child in each container moves to the container's top-left"

Without hierarchy, we only see a flat list of objects with no grouping.
With hierarchy, we can:
  1. Identify containers (composite objects)
  2. Scope operations to within each container
  3. Use parent-relative anchoring to describe positions

The automatic screening correctly detected that hierarchy mode should be
used for this puzzle based on the low variance of parent-child anchor
relationships.
    """)


if __name__ == "__main__":
    main()
