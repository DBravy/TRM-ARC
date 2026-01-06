#!/usr/bin/env python3
"""
Benchmark all ARC-AGI-1 puzzles through relational_position.py

Tracks:
- Puzzles solved immediately (no training, via screening)
- Puzzles solved with training (up to 100 epochs max)
- Puzzles that can't be solved
- Accuracy percentages for train/test sets
- Other useful metrics

Results exported to JSON.
"""

import argparse
import json
import os
import sys
import time
import random
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

# Import from relational_position.py
from relational_position import (
    DEVICE, GRID_SIZE, FRAMINGS,
    load_puzzles,
    PositionDataset,
    PerObjectAnchorTransformModule,
    AnchorScreeningTrainer,
    evaluate_no_train,
    evaluate_per_object_anchor,
    train_epoch_per_object_anchor,
    screen_orderings_for_puzzle,
)

# Selection module imports
from selection_module import SelectionScreener


def run_puzzle_benchmark(
    puzzle_id: str,
    puzzles: Dict,
    max_epochs: int = 100,
    seed: int = 42,
    hidden_dim: int = 64,
    batch_size: int = 8,
    lr: float = 1e-3,
    screening_epochs: int = 100,
    verbose: bool = False
) -> Dict:
    """
    Run benchmark for a single puzzle.

    Returns a dict with:
    - puzzle_id
    - solved_immediately: bool (100% test accuracy without training)
    - solved_with_training: bool (100% test accuracy after training)
    - epochs_needed: int (0 if immediate, N if trained, -1 if unsolved)
    - train_accuracy: float
    - test_accuracy: float
    - train_mean_pixel_error: float
    - test_mean_pixel_error: float
    - num_train_examples: int
    - num_test_examples: int
    - num_objects: int (max objects across examples)
    - ordering_strategy: str
    - framings_used: list of framings per object
    - error: str or None (if puzzle failed to process)
    - runtime_seconds: float
    """
    start_time = time.time()

    result = {
        'puzzle_id': puzzle_id,
        'solved_immediately': False,
        'solved_with_training': False,
        'epochs_needed': -1,
        'train_accuracy': 0.0,
        'test_accuracy': 0.0,
        'train_mean_pixel_error': float('inf'),
        'test_mean_pixel_error': float('inf'),
        'train_correct': 0,
        'train_total': 0,
        'test_correct': 0,
        'test_total': 0,
        'num_train_examples': 0,
        'num_test_examples': 0,
        'num_objects': 0,
        'ordering_strategy': None,
        'framings_used': [],
        'error': None,
        'runtime_seconds': 0.0,
        'auto_skipped_training': False,  # True if screening found perfect anchors
    }

    # Set seeds
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    try:
        if puzzle_id not in puzzles:
            result['error'] = f"Puzzle {puzzle_id} not found"
            return result

        puzzle = puzzles[puzzle_id]
        result['num_train_examples'] = len(puzzle.get('train', []))
        result['num_test_examples'] = len(puzzle.get('test', []))

        # Screen for best ordering strategy
        screen_results = screen_orderings_for_puzzle(
            puzzle,
            verbose=False,
            selection_criterion=None,
            selection_rule=None,
            use_color_only=False
        )
        ordering_strategy = screen_results['best_name']
        result['ordering_strategy'] = ordering_strategy

        # Create training dataset
        train_dataset = PositionDataset(
            puzzles,
            puzzle_ids=[puzzle_id],
            use_color_only=False,
            include_test=False,
            num_augmentations=0,
            ordering_strategy=ordering_strategy,
            predict_absolute=False,
            per_object_anchor=True,
            selection_criterion=None,
            selection_rule=None
        )

        if len(train_dataset) == 0:
            result['error'] = "No valid training samples"
            return result

        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )

        # Create test dataset
        test_puzzle = {
            puzzle_id: {
                'train': puzzle.get('test', []),
                'test': []
            }
        }
        test_dataset = PositionDataset(
            test_puzzle,
            puzzle_ids=[puzzle_id],
            use_color_only=False,
            include_test=False,
            num_augmentations=0,
            ordering_strategy=ordering_strategy,
            predict_absolute=False,
            per_object_anchor=True,
            selection_criterion=None,
            selection_rule=None
        )

        has_test_data = len(test_dataset) > 0
        if has_test_data:
            test_loader = torch.utils.data.DataLoader(
                test_dataset, batch_size=batch_size, shuffle=False
            )

        # Get max objects
        max_obj = max(sum(s.input_valid) for s in train_dataset.samples) if train_dataset.samples else 0
        result['num_objects'] = max_obj

        # Create model
        model = PerObjectAnchorTransformModule(
            hidden_dim=hidden_dim,
            num_heads=4
        ).to(DEVICE)

        # Phase 1: Screening
        trainer = AnchorScreeningTrainer(screening_epochs=screening_epochs, lr=0.01, verbose=False)
        _screening_results = trainer.run_screening(train_dataset, model.anchor_module, DEVICE)

        # Collect framings used
        for obj_idx in range(max_obj):
            framing_idx = model.anchor_module.get_framing_idx(obj_idx)
            result['framings_used'].append(FRAMINGS[framing_idx])

        # Check if all objects have low variance (can skip training)
        auto_no_train = model.anchor_module.all_low_variance(threshold=0.001)
        result['auto_skipped_training'] = auto_no_train

        # Evaluate no-train (screening offsets only)
        train_eval = evaluate_no_train(model.anchor_module, train_dataset.samples)
        result['train_accuracy'] = train_eval['pixel_accuracy']
        result['train_mean_pixel_error'] = train_eval['mean_pixel_error']
        result['train_correct'] = train_eval.get('total_correct', 0)
        result['train_total'] = train_eval.get('total_samples', 0)

        if has_test_data:
            test_eval = evaluate_no_train(model.anchor_module, test_dataset.samples)
            result['test_accuracy'] = test_eval['pixel_accuracy']
            result['test_mean_pixel_error'] = test_eval['mean_pixel_error']
            result['test_correct'] = test_eval.get('total_correct', 0)
            result['test_total'] = test_eval.get('total_samples', 0)

            # Check if solved immediately
            if test_eval['pixel_accuracy'] >= 1.0:
                result['solved_immediately'] = True
                result['epochs_needed'] = 0
        else:
            # No test data - use train accuracy
            if train_eval['pixel_accuracy'] >= 1.0:
                result['solved_immediately'] = True
                result['epochs_needed'] = 0

        # If not solved immediately and not auto-skipped, try training
        if not result['solved_immediately'] and not auto_no_train:
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, max_epochs)

            for epoch in range(max_epochs):
                train_metrics = train_epoch_per_object_anchor(
                    model, train_loader, optimizer, DEVICE
                )
                scheduler.step()

                # Evaluate periodically (every 10 epochs) and at the end
                if (epoch + 1) % 10 == 0 or epoch == max_epochs - 1:
                    train_eval = evaluate_per_object_anchor(model, train_loader, DEVICE)
                    result['train_accuracy'] = train_eval['pixel_accuracy']
                    result['train_mean_pixel_error'] = train_eval['mean_pixel_error']

                    if has_test_data:
                        test_eval = evaluate_per_object_anchor(model, test_loader, DEVICE)
                        result['test_accuracy'] = test_eval['pixel_accuracy']
                        result['test_mean_pixel_error'] = test_eval['mean_pixel_error']

                        if test_eval['pixel_accuracy'] >= 1.0:
                            result['solved_with_training'] = True
                            result['epochs_needed'] = epoch + 1
                            break
                    else:
                        if train_eval['pixel_accuracy'] >= 1.0:
                            result['solved_with_training'] = True
                            result['epochs_needed'] = epoch + 1
                            break

    except Exception as e:
        result['error'] = str(e)

    result['runtime_seconds'] = time.time() - start_time
    return result


def main():
    parser = argparse.ArgumentParser(description="Benchmark all ARC-AGI-1 puzzles")
    parser.add_argument("--dataset", type=str, default="arc-agi-1",
                        choices=["arc-agi-1", "arc-agi-2"])
    parser.add_argument("--data-root", type=str, default="kaggle/combined")
    parser.add_argument("--max-epochs", type=int, default=100,
                        help="Max training epochs per puzzle")
    parser.add_argument("--screening-epochs", type=int, default=100,
                        help="Epochs for screening phase")
    parser.add_argument("--output", type=str, default="benchmark_results.json",
                        help="Output JSON file")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of puzzles (for testing)")
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume from existing results file")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--puzzle-ids", type=str, nargs='+', default=None,
                        help="Specific puzzle IDs to run (space-separated)")

    args = parser.parse_args()

    print(f"Device: {DEVICE}")
    print(f"Dataset: {args.dataset}")
    print(f"Max epochs per puzzle: {args.max_epochs}")
    print(f"Output: {args.output}")

    # Load puzzles
    print("\nLoading puzzles...")
    puzzles = load_puzzles(args.dataset, args.data_root)
    print(f"Total puzzles: {len(puzzles)}")

    # Get puzzle IDs to process
    if args.puzzle_ids:
        puzzle_ids = args.puzzle_ids
    else:
        puzzle_ids = sorted(puzzles.keys())

    if args.limit:
        puzzle_ids = puzzle_ids[:args.limit]

    print(f"Puzzles to process: {len(puzzle_ids)}")

    # Load existing results if resuming
    existing_results = {}
    if args.resume and os.path.exists(args.resume):
        with open(args.resume) as f:
            data = json.load(f)
            existing_results = {r['puzzle_id']: r for r in data.get('results', [])}
        print(f"Resuming from {args.resume} ({len(existing_results)} existing results)")

    # Run benchmark
    results = []
    start_time = time.time()

    solved_immediate = 0
    solved_training = 0
    unsolved = 0
    errors = 0

    for i, puzzle_id in enumerate(puzzle_ids):
        # Skip if already processed
        if puzzle_id in existing_results:
            result = existing_results[puzzle_id]
            results.append(result)
            if result.get('error'):
                errors += 1
            elif result['solved_immediately']:
                solved_immediate += 1
            elif result['solved_with_training']:
                solved_training += 1
            else:
                unsolved += 1
            continue

        print(f"\n[{i+1}/{len(puzzle_ids)}] Processing {puzzle_id}...", end=" ", flush=True)

        result = run_puzzle_benchmark(
            puzzle_id=puzzle_id,
            puzzles=puzzles,
            max_epochs=args.max_epochs,
            seed=args.seed,
            screening_epochs=args.screening_epochs,
            verbose=args.verbose
        )

        results.append(result)

        # Update counters
        if result.get('error'):
            print(f"ERROR: {result['error']}")
            errors += 1
        elif result['solved_immediately']:
            print(f"IMMEDIATE (test_acc={result['test_accuracy']:.1%}, {result['runtime_seconds']:.1f}s)")
            solved_immediate += 1
        elif result['solved_with_training']:
            print(f"TRAINED (epochs={result['epochs_needed']}, test_acc={result['test_accuracy']:.1%}, {result['runtime_seconds']:.1f}s)")
            solved_training += 1
        else:
            print(f"UNSOLVED (test_acc={result['test_accuracy']:.1%}, {result['runtime_seconds']:.1f}s)")
            unsolved += 1

        # Save intermediate results every 10 puzzles
        if (i + 1) % 10 == 0:
            _save_results(args.output, results, start_time, solved_immediate,
                         solved_training, unsolved, errors, len(puzzle_ids))

    # Final save
    _save_results(args.output, results, start_time, solved_immediate,
                 solved_training, unsolved, errors, len(puzzle_ids))

    # Print summary
    print("\n" + "=" * 60)
    print("BENCHMARK COMPLETE")
    print("=" * 60)
    total = len(results)
    print(f"Total puzzles: {total}")
    print(f"Solved immediately: {solved_immediate} ({100*solved_immediate/total:.1f}%)")
    print(f"Solved with training: {solved_training} ({100*solved_training/total:.1f}%)")
    print(f"Unsolved: {unsolved} ({100*unsolved/total:.1f}%)")
    print(f"Errors: {errors} ({100*errors/total:.1f}%)")
    print(f"Total runtime: {time.time() - start_time:.1f}s")
    print(f"\nResults saved to: {args.output}")


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def _convert_numpy(obj):
    """Recursively convert numpy types to Python types."""
    if isinstance(obj, dict):
        return {k: _convert_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_convert_numpy(v) for v in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif obj == float('inf'):
        return "inf"
    elif obj == float('-inf'):
        return "-inf"
    return obj


def _save_results(output_path: str, results: List[Dict], start_time: float,
                  solved_immediate: int, solved_training: int, unsolved: int,
                  errors: int, total: int):
    """Save results to JSON file."""
    # Calculate aggregate stats
    valid_results = [r for r in results if not r.get('error')]

    avg_train_acc = float(np.mean([r['train_accuracy'] for r in valid_results])) if valid_results else 0
    avg_test_acc = float(np.mean([r['test_accuracy'] for r in valid_results])) if valid_results else 0
    avg_runtime = float(np.mean([r['runtime_seconds'] for r in valid_results])) if valid_results else 0

    output = {
        'metadata': {
            'timestamp': datetime.now().isoformat(),
            'total_puzzles': total,
            'processed_puzzles': len(results),
            'total_runtime_seconds': time.time() - start_time,
        },
        'summary': {
            'solved_immediately': solved_immediate,
            'solved_immediately_pct': 100 * solved_immediate / max(len(results), 1),
            'solved_with_training': solved_training,
            'solved_with_training_pct': 100 * solved_training / max(len(results), 1),
            'unsolved': unsolved,
            'unsolved_pct': 100 * unsolved / max(len(results), 1),
            'errors': errors,
            'errors_pct': 100 * errors / max(len(results), 1),
            'avg_train_accuracy': avg_train_acc,
            'avg_test_accuracy': avg_test_acc,
            'avg_runtime_per_puzzle': avg_runtime,
        },
        'results': _convert_numpy(results)
    }

    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2, cls=NumpyEncoder)


if __name__ == "__main__":
    main()
