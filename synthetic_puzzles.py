"""
Synthetic Puzzles Module

Generates synthetic puzzles for testing and development.
Synthetic puzzles use IDs prefixed with "syn_" to distinguish them from ARC puzzles.

Usage:
    # Generate and register a synthetic puzzle
    from synthetic_puzzles import register_synthetic_puzzle, get_synthetic_puzzle

    # Command-line:
    python synthetic_puzzles.py --list                    # List all synthetic puzzles
    python synthetic_puzzles.py --generate dual_fill      # Generate dual_fill puzzle
    python synthetic_puzzles.py --view syn_dual_fill      # View a specific puzzle
"""

import json
import os
import random
from typing import Dict, List, Tuple, Optional
from collections import Counter
import numpy as np

# Directory to store synthetic puzzles
SYNTHETIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "synthetic_puzzles")


def ensure_synthetic_dir():
    """Ensure the synthetic puzzles directory exists."""
    os.makedirs(SYNTHETIC_DIR, exist_ok=True)


def get_synthetic_puzzle_path(puzzle_id: str) -> str:
    """Get the file path for a synthetic puzzle."""
    # Remove 'syn_' prefix if present for the filename
    name = puzzle_id[4:] if puzzle_id.startswith("syn_") else puzzle_id
    return os.path.join(SYNTHETIC_DIR, f"{name}.json")


def save_synthetic_puzzle(puzzle_id: str, puzzle_data: Dict):
    """Save a synthetic puzzle to disk."""
    ensure_synthetic_dir()
    path = get_synthetic_puzzle_path(puzzle_id)
    with open(path, 'w') as f:
        json.dump(puzzle_data, f, indent=2)
    print(f"Saved synthetic puzzle to {path}")


def load_synthetic_puzzle(puzzle_id: str) -> Optional[Dict]:
    """Load a synthetic puzzle from disk."""
    path = get_synthetic_puzzle_path(puzzle_id)
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


def list_synthetic_puzzles() -> List[str]:
    """List all available synthetic puzzle IDs."""
    if not os.path.exists(SYNTHETIC_DIR):
        return []
    puzzles = []
    for f in os.listdir(SYNTHETIC_DIR):
        if f.endswith('.json'):
            puzzles.append(f"syn_{f[:-5]}")
    return sorted(puzzles)


# =============================================================================
# Dual Region Fill Puzzle Generator
# =============================================================================

def generate_dual_fill_example(seed: Optional[int] = None) -> Tuple[List[List[int]], List[List[int]]]:
    """
    Generate a single input/output pair for the dual region fill puzzle.

    Input: 7x3 grid with:
        - Top 3x3 region with random colors
        - 1-row divider with a consistent color
        - Bottom 3x3 region with random colors

    Output:
        - Top 3x3 filled with most common color from top region
        - Divider unchanged
        - Bottom 3x3 filled with most common color from bottom region

    Returns:
        Tuple of (input_grid, output_grid)
    """
    if seed is not None:
        random.seed(seed)

    # Available ARC colors (0-9)
    colors = list(range(10))

    # Generate divider color
    divider_color = random.choice(colors)

    # Generate top region (3x3) - ensure there's a unique most common color
    top_region = []
    for _ in range(3):
        row = [random.choice(colors) for _ in range(3)]
        top_region.append(row)

    # Generate bottom region (3x3) - ensure there's a unique most common color
    bottom_region = []
    for _ in range(3):
        row = [random.choice(colors) for _ in range(3)]
        bottom_region.append(row)

    # Find most common colors
    top_flat = [c for row in top_region for c in row]
    bottom_flat = [c for row in bottom_region for c in row]

    top_mode = Counter(top_flat).most_common(1)[0][0]
    bottom_mode = Counter(bottom_flat).most_common(1)[0][0]

    # Construct input grid (7 rows x 3 cols)
    input_grid = (
        top_region +
        [[divider_color] * 3] +
        bottom_region
    )

    # Construct output grid
    output_grid = (
        [[top_mode] * 3 for _ in range(3)] +
        [[divider_color] * 3] +
        [[bottom_mode] * 3 for _ in range(3)]
    )

    return input_grid, output_grid


def generate_dual_fill_puzzle(num_train: int = 3, num_test: int = 1, base_seed: int = 42) -> Dict:
    """
    Generate a complete dual-region fill puzzle.

    Args:
        num_train: Number of training examples
        num_test: Number of test examples
        base_seed: Base random seed for reproducibility

    Returns:
        Puzzle dictionary in ARC format
    """
    puzzle = {
        "train": [],
        "test": []
    }

    # Generate training examples
    for i in range(num_train):
        input_grid, output_grid = generate_dual_fill_example(seed=base_seed + i)
        puzzle["train"].append({
            "input": input_grid,
            "output": output_grid
        })

    # Generate test examples
    for i in range(num_test):
        input_grid, output_grid = generate_dual_fill_example(seed=base_seed + num_train + i)
        puzzle["test"].append({
            "input": input_grid,
            "output": output_grid
        })

    return puzzle


def generate_and_save_dual_fill(seed: int = 42):
    """Generate and save the dual fill puzzle."""
    puzzle = generate_dual_fill_puzzle(num_train=3, num_test=1, base_seed=seed)
    save_synthetic_puzzle("syn_dual_fill", puzzle)
    return puzzle


# =============================================================================
# Puzzle Registry
# =============================================================================

# Registry of puzzle generators
PUZZLE_GENERATORS = {
    "dual_fill": generate_and_save_dual_fill,
}


def generate_puzzle(puzzle_type: str, **kwargs) -> Dict:
    """Generate a puzzle of the specified type."""
    if puzzle_type not in PUZZLE_GENERATORS:
        raise ValueError(f"Unknown puzzle type: {puzzle_type}. Available: {list(PUZZLE_GENERATORS.keys())}")
    return PUZZLE_GENERATORS[puzzle_type](**kwargs)


# =============================================================================
# Visualization
# =============================================================================

ARC_COLORS = {
    0: '\033[40m  \033[0m',   # Black
    1: '\033[44m  \033[0m',   # Blue
    2: '\033[41m  \033[0m',   # Red
    3: '\033[42m  \033[0m',   # Green
    4: '\033[43m  \033[0m',   # Yellow
    5: '\033[100m  \033[0m',  # Gray
    6: '\033[45m  \033[0m',   # Magenta
    7: '\033[48;5;208m  \033[0m',  # Orange
    8: '\033[46m  \033[0m',   # Cyan
    9: '\033[48;5;88m  \033[0m',   # Maroon
}


def print_grid(grid: List[List[int]], label: str = ""):
    """Print a grid using ANSI colors."""
    if label:
        print(f"{label}:")
    for row in grid:
        print("".join(ARC_COLORS.get(c, f'[{c}]') for c in row))
    print()


def view_puzzle(puzzle_id: str):
    """View a synthetic puzzle."""
    puzzle = load_synthetic_puzzle(puzzle_id)
    if puzzle is None:
        print(f"Puzzle {puzzle_id} not found")
        return

    print(f"\n{'='*50}")
    print(f"Puzzle: {puzzle_id}")
    print(f"{'='*50}\n")

    print("TRAINING EXAMPLES:")
    for i, example in enumerate(puzzle["train"]):
        print(f"\n--- Train {i+1} ---")
        print_grid(example["input"], "Input")
        print_grid(example["output"], "Output")

    print("\nTEST EXAMPLES:")
    for i, example in enumerate(puzzle["test"]):
        print(f"\n--- Test {i+1} ---")
        print_grid(example["input"], "Input")
        if "output" in example:
            print_grid(example["output"], "Output (expected)")


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Synthetic Puzzle Generator")
    parser.add_argument("--list", action="store_true", help="List all synthetic puzzles")
    parser.add_argument("--generate", type=str, help="Generate a puzzle type (e.g., 'dual_fill')")
    parser.add_argument("--view", type=str, help="View a synthetic puzzle by ID")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for generation")

    args = parser.parse_args()

    if args.list:
        puzzles = list_synthetic_puzzles()
        if puzzles:
            print("Available synthetic puzzles:")
            for p in puzzles:
                print(f"  {p}")
        else:
            print("No synthetic puzzles found. Generate one with --generate")
    elif args.generate:
        print(f"Generating puzzle type: {args.generate}")
        generate_puzzle(args.generate, seed=args.seed)
        print("Done!")
    elif args.view:
        view_puzzle(args.view)
    else:
        parser.print_help()
