"""
Puzzle Loader Module

Unified puzzle loading functions for the ARC Relational Position System.
Consolidates duplicate loading logic from multiple modules.

Synthetic puzzles: IDs starting with "syn_" are loaded from synthetic_puzzles/
"""

import json
import os
from typing import Dict, Optional

# Default data root relative to this file
DEFAULT_DATA_ROOT = "kaggle/combined"
SYNTHETIC_DIR = "synthetic_puzzles"


def _get_default_data_root() -> str:
    """Get the default data root directory relative to this file."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(script_dir, DEFAULT_DATA_ROOT)


def _get_synthetic_dir() -> str:
    """Get the synthetic puzzles directory relative to this file."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(script_dir, SYNTHETIC_DIR)


def _load_synthetic_puzzle(puzzle_id: str) -> Optional[Dict]:
    """
    Load a synthetic puzzle by ID.

    Synthetic puzzles are stored in synthetic_puzzles/ directory.
    IDs should start with 'syn_' prefix.

    Returns:
        Puzzle data if found, None otherwise.
    """
    synthetic_dir = _get_synthetic_dir()
    if not os.path.exists(synthetic_dir):
        return None

    # Remove 'syn_' prefix for filename lookup
    name = puzzle_id[4:] if puzzle_id.startswith("syn_") else puzzle_id
    path = os.path.join(synthetic_dir, f"{name}.json")

    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


def load_puzzle(puzzle_id: str, data_root: Optional[str] = None) -> Dict:
    """
    Load a single puzzle from the ARC dataset by ID.

    Searches through all available subsets (training, evaluation, training2, evaluation2)
    to find the puzzle. Also checks synthetic puzzles (IDs starting with "syn_").

    Args:
        puzzle_id: The unique identifier for the puzzle
        data_root: Path to the data directory. If None, uses kaggle/combined relative to this file.

    Returns:
        Dictionary containing the puzzle data with 'train' and 'test' keys.
        Test outputs are included if solutions are available.

    Raises:
        ValueError: If the puzzle is not found in any subset
    """
    # Check for synthetic puzzles first
    if puzzle_id.startswith("syn_"):
        synthetic = _load_synthetic_puzzle(puzzle_id)
        if synthetic is not None:
            return synthetic
        raise ValueError(f"Synthetic puzzle '{puzzle_id}' not found in synthetic_puzzles/")

    if data_root is None:
        data_root = _get_default_data_root()

    subsets = ["training", "evaluation", "training2", "evaluation2"]

    for subset in subsets:
        challenges_path = os.path.join(data_root, f"arc-agi_{subset}_challenges.json")
        solutions_path = os.path.join(data_root, f"arc-agi_{subset}_solutions.json")

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


def load_puzzles(dataset_name: str = "arc-agi-1", data_root: Optional[str] = None) -> Dict:
    """
    Load all puzzles for a specific ARC dataset.

    Args:
        dataset_name: Which dataset to load. Options:
            - "arc-agi-1": Original ARC training and evaluation sets
            - "arc-agi-2": Second set (training2 and evaluation2)
        data_root: Path to the data directory. If None, uses kaggle/combined relative to this file.

    Returns:
        Dictionary mapping puzzle_id -> puzzle data
    """
    if data_root is None:
        data_root = _get_default_data_root()

    config = {
        "arc-agi-1": {"subsets": ["training", "evaluation"]},
        "arc-agi-2": {"subsets": ["training2", "evaluation2"]},
    }

    if dataset_name not in config:
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(config.keys())}")

    all_puzzles = {}

    for subset in config[dataset_name]["subsets"]:
        challenges_path = os.path.join(data_root, f"arc-agi_{subset}_challenges.json")
        solutions_path = os.path.join(data_root, f"arc-agi_{subset}_solutions.json")

        if not os.path.exists(challenges_path):
            print(f"Warning: {challenges_path} not found")
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


def load_all_puzzles(data_root: Optional[str] = None) -> Dict:
    """
    Load all puzzles from all available subsets.

    Args:
        data_root: Path to the data directory. If None, uses kaggle/combined relative to this file.

    Returns:
        Dictionary mapping puzzle_id -> puzzle data
    """
    if data_root is None:
        data_root = _get_default_data_root()

    all_puzzles = {}

    subsets = ["training", "evaluation", "training2", "evaluation2"]

    for subset in subsets:
        challenges_path = os.path.join(data_root, f"arc-agi_{subset}_challenges.json")
        solutions_path = os.path.join(data_root, f"arc-agi_{subset}_solutions.json")

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
                        if i < len(puzzles[puzzle_id].get("test", [])):
                            puzzles[puzzle_id]["test"][i]["output"] = sol

        all_puzzles.update(puzzles)

    return all_puzzles
