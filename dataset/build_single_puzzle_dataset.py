"""
Build a dataset with a single ARC puzzle for testing whether the model
can learn a puzzle embedding from just the demonstration examples.

Usage:
    python -m dataset.build_single_puzzle_dataset \
        --puzzle-id 007bbfb7 \
        --input-file-prefix kaggle/combined/arc-agi \
        --source-subset training \
        --output-dir data/single_puzzle_test
"""

from typing import List, Tuple
from dataclasses import dataclass
import os
import json
import numpy as np

from argdantic import ArgParser
from pydantic import BaseModel

from dataset.common import PuzzleDatasetMetadata
from dataset.build_arc_dataset import (
    ARCPuzzle, ARCMaxGridSize,
    arc_grid_to_np, np_grid_to_seq_translational_augment, np_grid_to_raw_grid_augment,
    aug, puzzle_hash, ARCAugmentRetriesFactor
)

cli = ArgParser()


class SinglePuzzleConfig(BaseModel):
    puzzle_id: str
    input_file_prefix: str
    source_subset: str  # which subset the puzzle is in (training, evaluation, etc.)
    output_dir: str
    seed: int = 42
    num_aug: int = 100  # fewer augmentations for single puzzle


def build_single_puzzle_dataset(config: SinglePuzzleConfig):
    np.random.seed(config.seed)

    # Load the puzzle
    challenges_file = f"{config.input_file_prefix}_{config.source_subset}_challenges.json"
    solutions_file = f"{config.input_file_prefix}_{config.source_subset}_solutions.json"

    with open(challenges_file, "r") as f:
        all_puzzles = json.load(f)

    if config.puzzle_id not in all_puzzles:
        raise ValueError(f"Puzzle {config.puzzle_id} not found in {challenges_file}")

    puzzle = all_puzzles[config.puzzle_id]

    # Load solution
    with open(solutions_file, "r") as f:
        all_solutions = json.load(f)

    if config.puzzle_id in all_solutions:
        for idx, sol_grid in enumerate(all_solutions[config.puzzle_id]):
            puzzle["test"][idx]["output"] = sol_grid
    else:
        raise ValueError(f"Solution for {config.puzzle_id} not found")

    # Print puzzle info
    print(f"Puzzle: {config.puzzle_id}")
    print(f"  Train examples (demonstrations): {len(puzzle['train'])}")
    print(f"  Test examples: {len(puzzle['test'])}")

    for i, ex in enumerate(puzzle['train']):
        inp = np.array(ex['input'])
        out = np.array(ex['output'])
        print(f"    Demo {i+1}: input {inp.shape} -> output {out.shape}")

    for i, ex in enumerate(puzzle['test']):
        inp = np.array(ex['input'])
        out = np.array(ex['output'])
        print(f"    Test {i+1}: input {inp.shape} -> output {out.shape}")

    # Convert to ARCPuzzle format
    train_puzzle = ARCPuzzle(
        id=config.puzzle_id,
        examples=[(arc_grid_to_np(ex["input"]), arc_grid_to_np(ex["output"]))
                  for ex in puzzle["train"]]
    )

    test_puzzle = ARCPuzzle(
        id=config.puzzle_id,  # SAME ID - this is the key!
        examples=[(arc_grid_to_np(ex["input"]), arc_grid_to_np(ex["output"]))
                  for ex in puzzle["test"]]
    )

    # Generate augmentations for training data only
    train_puzzles = [train_puzzle]

    if config.num_aug > 0:
        hashes = {puzzle_hash({"train": train_puzzle})}

        for _trial in range(ARCAugmentRetriesFactor * config.num_aug):
            aug_name, _map_grid = aug(config.puzzle_id)

            augmented = ARCPuzzle(
                aug_name,
                [(_map_grid(inp), _map_grid(out)) for inp, out in train_puzzle.examples]
            )

            h = puzzle_hash({"train": augmented})
            if h not in hashes:
                hashes.add(h)
                train_puzzles.append(augmented)

            if len(train_puzzles) >= config.num_aug + 1:
                break

        print(f"  Generated {len(train_puzzles)} augmented versions for training")

    # Create output directories
    os.makedirs(os.path.join(config.output_dir, "train"), exist_ok=True)
    os.makedirs(os.path.join(config.output_dir, "test"), exist_ok=True)

    # Puzzle identifier: 1 for the puzzle, 0 for blank
    puzzle_identifier = 1
    identifier_map = {config.puzzle_id: puzzle_identifier}

    # Also map augmented versions to the same identifier
    for p in train_puzzles:
        if p.id not in identifier_map:
            identifier_map[p.id] = puzzle_identifier  # All augments share same ID

    # Build training data
    print("\nBuilding training data (from demonstrations)...")
    train_results = {
        "inputs": [], "labels": [],
        "input_grids": [], "label_grids": [],
        "puzzle_identifiers": [], "puzzle_indices": [], "group_indices": []
    }
    train_results["puzzle_indices"].append(0)
    train_results["group_indices"].append(0)

    example_id = 0
    puzzle_id_counter = 0

    for p in train_puzzles:
        for inp, out in p.examples:
            # Tokenized sequences
            inp_seq, out_seq = np_grid_to_seq_translational_augment(inp, out, do_translation=True)
            train_results["inputs"].append(inp_seq)
            train_results["labels"].append(out_seq)

            # Raw grids
            inp_grid, out_grid = np_grid_to_raw_grid_augment(inp, out, do_translation=True)
            train_results["input_grids"].append(inp_grid)
            train_results["label_grids"].append(out_grid)

            example_id += 1

        train_results["puzzle_indices"].append(example_id)
        train_results["puzzle_identifiers"].append(puzzle_identifier)
        puzzle_id_counter += 1

    train_results["group_indices"].append(puzzle_id_counter)

    print(f"  Total training examples: {example_id}")

    # Build test data
    print("\nBuilding test data (from test examples)...")
    test_results = {
        "inputs": [], "labels": [],
        "input_grids": [], "label_grids": [],
        "puzzle_identifiers": [], "puzzle_indices": [], "group_indices": []
    }
    test_results["puzzle_indices"].append(0)
    test_results["group_indices"].append(0)

    example_id = 0

    for inp, out in test_puzzle.examples:
        # Tokenized sequences (no augmentation for test)
        inp_seq, out_seq = np_grid_to_seq_translational_augment(inp, out, do_translation=False)
        test_results["inputs"].append(inp_seq)
        test_results["labels"].append(out_seq)

        # Raw grids
        inp_grid, out_grid = np_grid_to_raw_grid_augment(inp, out, do_translation=False)
        test_results["input_grids"].append(inp_grid)
        test_results["label_grids"].append(out_grid)

        example_id += 1

    test_results["puzzle_indices"].append(example_id)
    test_results["puzzle_identifiers"].append(puzzle_identifier)  # SAME ID as training!
    test_results["group_indices"].append(1)

    print(f"  Total test examples: {example_id}")

    # Save data
    for split_name, results in [("train", train_results), ("test", test_results)]:
        for k, v in results.items():
            if k in {"inputs", "labels"}:
                v = np.stack(v, 0)
            elif k in {"input_grids", "label_grids"}:
                v = np.stack(v, 0).astype(np.uint8)
            else:
                v = np.array(v, dtype=np.int32)

            np.save(os.path.join(config.output_dir, split_name, f"all__{k}.npy"), v)

        # Metadata
        total_examples = len(results["inputs"])
        total_puzzles = len(results["puzzle_indices"]) - 1

        metadata = PuzzleDatasetMetadata(
            seq_len=ARCMaxGridSize * ARCMaxGridSize,
            vocab_size=10 + 2,
            pad_id=0,
            ignore_label_id=0,
            blank_identifier_id=0,
            num_puzzle_identifiers=2,  # 0=blank, 1=our puzzle
            total_groups=len(results["group_indices"]) - 1,
            mean_puzzle_examples=total_examples / total_puzzles if total_puzzles > 0 else 0,
            total_puzzles=total_puzzles,
            sets=["all"],
            grid_height=ARCMaxGridSize,
            grid_width=ARCMaxGridSize
        )

        with open(os.path.join(config.output_dir, split_name, "dataset.json"), "w") as f:
            json.dump(metadata.model_dump(), f, indent=2)

    # Save identifiers mapping
    ids_mapping = ["<blank>", config.puzzle_id]
    with open(os.path.join(config.output_dir, "identifiers.json"), "w") as f:
        json.dump(ids_mapping, f)

    # Save test puzzles (for evaluator)
    test_puzzles = {
        config.puzzle_id: puzzle
    }
    with open(os.path.join(config.output_dir, "test_puzzles.json"), "w") as f:
        json.dump(test_puzzles, f)

    print(f"\nDataset saved to {config.output_dir}")
    print(f"\nKey insight:")
    print(f"  - Training data has {len(train_puzzles)} puzzle variants (original + augments)")
    print(f"  - All share puzzle_identifier = {puzzle_identifier}")
    print(f"  - Test data uses the SAME puzzle_identifier = {puzzle_identifier}")
    print(f"  - Model must learn embedding from demos, apply to test input")


@cli.command(singleton=True)
def main(config: SinglePuzzleConfig):
    build_single_puzzle_dataset(config)


if __name__ == "__main__":
    cli()
