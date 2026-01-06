#!/usr/bin/env python3
"""
Anchor Selection Validation Script

Tests the hypothesis that the "correct" reference frame for position prediction
learns faster than incorrect ones.

We create synthetic puzzles with known ground-truth rules:
1. "Move to grid corner" - grid-relative framing should win
2. "Shift by constant delta" - delta framing should win  
3. "Stack relative to anchor object" - object-relative framing should win

For each puzzle, we train small models with different anchor framings and
compare learning curves. The correct framing should converge faster and
achieve lower loss.

Usage:
    python anchor_validation.py
"""

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Tuple, Dict
from collections import defaultdict


# =============================================================================
# Synthetic Puzzle Generation
# =============================================================================

@dataclass
class SyntheticObject:
    """An object with input and output positions."""
    input_pos: np.ndarray   # (row, col) top-left corner
    output_pos: np.ndarray  # (row, col) top-left corner
    size: Tuple[int, int]   # (height, width)
    color: int


@dataclass
class SyntheticPuzzle:
    """A synthetic puzzle with known transformation rule."""
    name: str
    rule_type: str  # 'grid_corner', 'constant_delta', 'relative_to_object'
    grid_sizes: List[Tuple[int, int]]  # Grid size per example (allows variation)
    examples: List[List[SyntheticObject]]  # List of examples, each with objects


def generate_grid_corner_puzzle(
    num_examples: int = 5,
    num_objects: int = 3,
    grid_size_range: Tuple[int, int] = (15, 25),
    corner: str = 'top_left'
) -> SyntheticPuzzle:
    """
    Generate puzzle where all objects move to a grid corner.
    The grid-relative framing should learn this easily (constant target).
    """
    examples = []
    grid_sizes = []

    for _ in range(num_examples):
        # Vary grid size per example
        H = np.random.randint(grid_size_range[0], grid_size_range[1] + 1)
        W = np.random.randint(grid_size_range[0], grid_size_range[1] + 1)
        grid_sizes.append((H, W))

        # Corner positions (computed per example since grid size varies)
        corners = {
            'top_left': (0, 0),
            'top_right': (0, W - 3),
            'bottom_left': (H - 3, 0),
            'bottom_right': (H - 3, W - 3),
        }
        target_corner = corners[corner]

        objects = []
        # Stack objects at the corner
        stack_offset = 0
        for obj_idx in range(num_objects):
            # Random input position
            input_row = np.random.randint(0, H - 3)
            input_col = np.random.randint(0, W - 3)

            # Output position: at corner, stacked vertically
            output_row = target_corner[0] + stack_offset
            output_col = target_corner[1]

            obj = SyntheticObject(
                input_pos=np.array([input_row, input_col]),
                output_pos=np.array([output_row, output_col]),
                size=(2, 2),
                color=obj_idx + 1
            )
            objects.append(obj)
            stack_offset += 3  # Stack with gap

        examples.append(objects)

    return SyntheticPuzzle(
        name=f"move_to_{corner}",
        rule_type='grid_corner',
        grid_sizes=grid_sizes,
        examples=examples
    )


def generate_constant_delta_puzzle(
    num_examples: int = 5,
    num_objects: int = 3,
    grid_size_range: Tuple[int, int] = (15, 25),
    delta: Tuple[int, int] = (0, 5)
) -> SyntheticPuzzle:
    """
    Generate puzzle where all objects shift by a constant delta.
    The delta framing should learn this easily (constant target).
    """
    dr, dc = delta

    examples = []
    grid_sizes = []

    for _ in range(num_examples):
        # Vary grid size per example
        H = np.random.randint(grid_size_range[0], grid_size_range[1] + 1)
        W = np.random.randint(grid_size_range[0], grid_size_range[1] + 1)
        grid_sizes.append((H, W))

        objects = []
        for obj_idx in range(num_objects):
            # Random input position (ensure output stays in bounds)
            max_row = H - 3 - max(0, dr)
            max_col = W - 3 - max(0, dc)
            min_row = max(0, -dr)
            min_col = max(0, -dc)

            input_row = np.random.randint(min_row, max_row)
            input_col = np.random.randint(min_col, max_col)

            obj = SyntheticObject(
                input_pos=np.array([input_row, input_col]),
                output_pos=np.array([input_row + dr, input_col + dc]),
                size=(2, 2),
                color=obj_idx + 1
            )
            objects.append(obj)

        examples.append(objects)

    return SyntheticPuzzle(
        name=f"shift_by_{delta}",
        rule_type='constant_delta',
        grid_sizes=grid_sizes,
        examples=examples
    )


def generate_relative_to_anchor_puzzle(
    num_examples: int = 5,
    num_objects: int = 3,
    grid_size_range: Tuple[int, int] = (20, 30),  # Needs larger grids for stacking
    anchor_object: int = 0,  # Object index that serves as anchor
    relative_offset: Tuple[int, int] = (3, 0)  # Where others go relative to anchor
) -> SyntheticPuzzle:
    """
    Generate puzzle where objects are placed relative to an anchor object.
    The anchor object STAYS IN PLACE (output = input).
    Other objects are placed at fixed offsets from the anchor.
    Object-relative framing should learn this easily.
    """
    dr, dc = relative_offset

    examples = []
    grid_sizes = []

    # Ensure minimum grid size can fit stacked objects
    min_h = max(grid_size_range[0], 7 + num_objects * 3)

    for _ in range(num_examples):
        # Vary grid size per example
        H = np.random.randint(min_h, grid_size_range[1] + 1)
        W = np.random.randint(grid_size_range[0], grid_size_range[1] + 1)
        grid_sizes.append((H, W))

        objects = []

        # Anchor object: stays in place (output = input)
        # Place it so there's room for stacked objects below
        anchor_row = np.random.randint(3, H - 3 - num_objects * 3)
        anchor_col = np.random.randint(0, W - 3)

        for obj_idx in range(num_objects):
            if obj_idx == anchor_object:
                # Anchor stays in place: output = input
                obj = SyntheticObject(
                    input_pos=np.array([anchor_row, anchor_col]),
                    output_pos=np.array([anchor_row, anchor_col]),  # Same position!
                    size=(2, 2),
                    color=obj_idx + 1
                )
            else:
                # Random input position
                input_row = np.random.randint(0, H - 3)
                input_col = np.random.randint(0, W - 3)

                # Output is relative to anchor's position (which stays in place)
                stack_num = obj_idx if obj_idx < anchor_object else obj_idx - 1
                output_row = anchor_row + dr * (stack_num + 1)
                output_col = anchor_col + dc * (stack_num + 1)

                obj = SyntheticObject(
                    input_pos=np.array([input_row, input_col]),
                    output_pos=np.array([output_row, output_col]),
                    size=(2, 2),
                    color=obj_idx + 1
                )
            objects.append(obj)

        examples.append(objects)

    return SyntheticPuzzle(
        name=f"relative_to_obj{anchor_object}",
        rule_type='relative_to_object',
        grid_sizes=grid_sizes,
        examples=examples
    )


# =============================================================================
# Target Computation for Different Framings
# =============================================================================

def compute_targets(
    puzzle: SyntheticPuzzle,
    framing: str,
    anchor_object: int = 0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute inputs and targets for a given framing.

    Args:
        puzzle: The synthetic puzzle
        framing: One of 'delta', 'grid_relative', 'object_relative'
        anchor_object: Which object to use as anchor (for object_relative)

    Returns:
        inputs: (N, num_objects, input_features)
        targets: (N, num_objects, 2) - all framings use 2D targets
    """
    num_examples = len(puzzle.examples)
    num_objects = len(puzzle.examples[0])
    obj_size = 2  # Object size (height/width)

    # Input features: normalized (row, col) from top-left
    inputs = np.zeros((num_examples, num_objects, 2), dtype=np.float32)

    # Target dimension: all framings use 2D targets
    target_dim = 2
    targets = np.zeros((num_examples, num_objects, target_dim), dtype=np.float32)

    for ex_idx, objects in enumerate(puzzle.examples):
        # Get grid size for this example
        H, W = puzzle.grid_sizes[ex_idx]

        for obj_idx, obj in enumerate(objects):
            # Input: normalized position from top-left
            inputs[ex_idx, obj_idx] = obj.input_pos / np.array([H, W])

            if framing == 'delta':
                # Target: output_pos - input_pos (normalized)
                delta = obj.output_pos - obj.input_pos
                targets[ex_idx, obj_idx] = delta / np.array([H, W])

            elif framing == 'grid_tl':
                # Target: distance from top-left corner
                row, col = obj.output_pos
                targets[ex_idx, obj_idx] = np.array([row / 20.0, col / 20.0])

            elif framing == 'grid_tr':
                # Target: distance from top-right corner
                row, col = obj.output_pos
                dist_right = W - col - obj_size
                targets[ex_idx, obj_idx] = np.array([row / 20.0, dist_right / 20.0])

            elif framing == 'grid_bl':
                # Target: distance from bottom-left corner
                row, col = obj.output_pos
                dist_bottom = H - row - obj_size
                targets[ex_idx, obj_idx] = np.array([dist_bottom / 20.0, col / 20.0])

            elif framing == 'grid_br':
                # Target: distance from bottom-right corner
                row, col = obj.output_pos
                dist_bottom = H - row - obj_size
                dist_right = W - col - obj_size
                targets[ex_idx, obj_idx] = np.array([dist_bottom / 20.0, dist_right / 20.0])

            elif framing == 'object_relative':
                # Target: output_pos relative to anchor object's output
                anchor_output = objects[anchor_object].output_pos
                relative = obj.output_pos - anchor_output
                targets[ex_idx, obj_idx] = relative / np.array([H, W])

            else:
                raise ValueError(f"Unknown framing: {framing}")

    return inputs, targets


# =============================================================================
# Decode Predictions to Pixel Positions
# =============================================================================

def decode_to_pixels(
    predictions: np.ndarray,
    puzzle: SyntheticPuzzle,
    framing: str,
    anchor_object: int = 0
) -> np.ndarray:
    """
    Decode framing-specific predictions back to actual pixel positions.

    For object_relative: anchor stays in place (anchor_output = anchor_input).
    This makes decoding possible without knowing ground truth.

    Args:
        predictions: (N, num_objects, 2) predicted targets
        puzzle: The puzzle (for grid sizes and input positions)
        framing: The framing type
        anchor_object: Which object is the anchor

    Returns:
        pixel_positions: (N, num_objects, 2) decoded row, col positions
    """
    num_examples = len(puzzle.examples)
    num_objects = len(puzzle.examples[0])
    obj_size = 2

    decoded = np.zeros((num_examples, num_objects, 2), dtype=np.float32)

    for ex_idx, objects in enumerate(puzzle.examples):
        H, W = puzzle.grid_sizes[ex_idx]

        for obj_idx, obj in enumerate(objects):
            pred = predictions[ex_idx, obj_idx]

            if framing == 'delta':
                # output = input + delta * grid_size
                delta = pred * np.array([H, W])
                decoded[ex_idx, obj_idx] = obj.input_pos + delta

            elif framing == 'grid_tl':
                # Decode from top-left: row = dist_top, col = dist_left
                row = pred[0] * 20.0
                col = pred[1] * 20.0
                decoded[ex_idx, obj_idx] = np.array([row, col])

            elif framing == 'grid_tr':
                # Decode from top-right
                row = pred[0] * 20.0
                dist_right = pred[1] * 20.0
                col = W - dist_right - obj_size
                decoded[ex_idx, obj_idx] = np.array([row, col])

            elif framing == 'grid_bl':
                # Decode from bottom-left
                dist_bottom = pred[0] * 20.0
                col = pred[1] * 20.0
                row = H - dist_bottom - obj_size
                decoded[ex_idx, obj_idx] = np.array([row, col])

            elif framing == 'grid_br':
                # Decode from bottom-right
                dist_bottom = pred[0] * 20.0
                dist_right = pred[1] * 20.0
                row = H - dist_bottom - obj_size
                col = W - dist_right - obj_size
                decoded[ex_idx, obj_idx] = np.array([row, col])

            elif framing == 'object_relative':
                # Anchor stays in place: anchor_output = anchor_input
                anchor_input = objects[anchor_object].input_pos
                # output = anchor_input + offset * grid_size
                offset = pred * np.array([H, W])
                decoded[ex_idx, obj_idx] = anchor_input + offset

    return decoded


def compute_pixel_accuracy(
    predictions: np.ndarray,
    puzzle: SyntheticPuzzle,
    framing: str,
    anchor_object: int = 0,
    tolerance: float = 1.0
) -> Tuple[float, float]:
    """
    Compute pixel accuracy: what fraction of objects are placed correctly?

    Args:
        predictions: (N, num_objects, 2) predicted targets
        puzzle: The puzzle with ground truth
        framing: The framing type
        anchor_object: Which object is anchor
        tolerance: How close (in pixels) counts as correct

    Returns:
        accuracy: fraction of correct placements
        mean_error: mean pixel distance error
    """
    decoded = decode_to_pixels(predictions, puzzle, framing, anchor_object)

    correct = 0
    total = 0
    total_error = 0.0

    for ex_idx, objects in enumerate(puzzle.examples):
        for obj_idx, obj in enumerate(objects):
            pred_pos = decoded[ex_idx, obj_idx]
            true_pos = obj.output_pos

            error = np.sqrt(((pred_pos - true_pos) ** 2).sum())
            total_error += error
            total += 1

            if error <= tolerance:
                correct += 1

    accuracy = correct / total if total > 0 else 0.0
    mean_error = total_error / total if total > 0 else 0.0

    return accuracy, mean_error


# =============================================================================
# Simple Predictor Model
# =============================================================================

class SimplePredictor(nn.Module):
    """
    Tiny MLP that predicts position from input features.
    Same architecture for all framings - only targets differ.
    """
    
    def __init__(self, input_dim: int = 2, hidden_dim: int = 32, output_dim: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        return self.net(x)


# =============================================================================
# Training
# =============================================================================

def train_with_framing(
    puzzle: SyntheticPuzzle,
    framing: str,
    epochs: int = 500,
    lr: float = 1e-2,
    anchor_object: int = 0,
    verbose: bool = False
) -> Tuple[List[float], nn.Module, np.ndarray]:
    """
    Train a model with a given framing and return loss history, model, and final predictions.
    """
    inputs, targets = compute_targets(puzzle, framing, anchor_object)

    # Get dimensions from data
    input_dim = inputs.shape[-1]
    output_dim = targets.shape[-1]
    num_examples = len(puzzle.examples)
    num_objects = len(puzzle.examples[0])

    # Flatten across examples and objects
    # Shape: (num_examples * num_objects, features)
    X = torch.tensor(inputs.reshape(-1, input_dim), dtype=torch.float32)
    Y = torch.tensor(targets.reshape(-1, output_dim), dtype=torch.float32)

    model = SimplePredictor(input_dim=input_dim, hidden_dim=32, output_dim=output_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    loss_history = []

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        pred = model(X)
        loss = ((pred - Y) ** 2).mean()

        loss.backward()
        optimizer.step()

        loss_history.append(loss.item())

        if verbose and (epoch + 1) % 100 == 0:
            print(f"  Epoch {epoch+1}: loss = {loss.item():.6f}")

    # Get final predictions reshaped back to (num_examples, num_objects, output_dim)
    model.eval()
    with torch.no_grad():
        final_preds = model(X).numpy().reshape(num_examples, num_objects, output_dim)

    return loss_history, model, final_preds


def train_grid_two_phase(
    puzzle: SyntheticPuzzle,
    screening_epochs: int = 2,
    full_epochs: int = 500,
    lr: float = 1e-2,
    verbose: bool = False
) -> Tuple[List[float], nn.Module, np.ndarray, str]:
    """
    Two-phase grid-relative training:
    1. Screen all 4 corners for a few epochs
    2. Pick the best corner by loss
    3. Train fully with that corner

    Returns:
        loss_history, model, predictions, chosen_corner
    """
    grid_corners = ['grid_tl', 'grid_tr', 'grid_bl', 'grid_br']

    # Phase 1: Quick screening
    corner_losses = {}
    for corner in grid_corners:
        inputs, targets = compute_targets(puzzle, corner)
        input_dim = inputs.shape[-1]
        output_dim = targets.shape[-1]

        X = torch.tensor(inputs.reshape(-1, input_dim), dtype=torch.float32)
        Y = torch.tensor(targets.reshape(-1, output_dim), dtype=torch.float32)

        model = SimplePredictor(input_dim=input_dim, hidden_dim=32, output_dim=output_dim)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        for _ in range(screening_epochs):
            model.train()
            optimizer.zero_grad()
            pred = model(X)
            loss = ((pred - Y) ** 2).mean()
            loss.backward()
            optimizer.step()

        corner_losses[corner] = loss.item()

    # Pick the best corner (lowest loss after screening)
    best_corner = min(corner_losses, key=corner_losses.get)

    if verbose:
        print(f"    Screening results (after {screening_epochs} epochs):")
        for corner, loss in corner_losses.items():
            marker = " <-- BEST" if corner == best_corner else ""
            print(f"      {corner}: {loss:.6f}{marker}")

    # Phase 2: Full training with best corner
    loss_history, model, predictions = train_with_framing(
        puzzle, best_corner, epochs=full_epochs, lr=lr, verbose=verbose
    )

    return loss_history, model, predictions, best_corner


# =============================================================================
# Validation Experiment
# =============================================================================

def run_validation_experiment():
    """
    Run the full validation experiment comparing framings across puzzle types.
    Uses PIXEL ACCURACY as the primary metric (not target MSE).
    """
    print("=" * 70)
    print("ANCHOR FRAMING VALIDATION EXPERIMENT")
    print("=" * 70)
    print()
    print("Hypothesis: The 'correct' framing for each puzzle type should")
    print("achieve better PIXEL ACCURACY (actual object placement).")
    print()
    print("Note: For object_relative, anchor stays in place for decoding.")
    print("Note: grid_auto uses 2-epoch screening to pick the best corner.")
    print()

    # Create puzzles with RANDOM corner for grid puzzle
    random_corner = np.random.choice(['top_left', 'top_right', 'bottom_left', 'bottom_right'])
    corner_to_framing = {
        'top_left': 'grid_tl',
        'top_right': 'grid_tr',
        'bottom_left': 'grid_bl',
        'bottom_right': 'grid_br',
    }
    expected_grid_framing = corner_to_framing[random_corner]

    puzzles = [
        (f"Grid Corner ({random_corner})",
         generate_grid_corner_puzzle(num_examples=10, num_objects=1, corner=random_corner),
         expected_grid_framing),  # Expected best: matching corner

        ("Constant Delta (+0, +5)",
         generate_constant_delta_puzzle(num_examples=10, delta=(0, 5)),
         'delta'),  # Expected best framing

        ("Relative to Object 0",
         generate_relative_to_anchor_puzzle(num_examples=10, num_objects=2, anchor_object=0),
         'object_relative'),  # Expected best framing
    ]

    # Framings to test: delta, grid_auto (two-phase), object_relative
    framings = ['delta', 'grid_auto', 'object_relative']
    epochs = 1000

    results = {}

    for puzzle_name, puzzle, expected_best in puzzles:
        print("-" * 70)
        print(f"PUZZLE: {puzzle_name}")
        print(f"Rule type: {puzzle.rule_type}")
        print(f"Expected best framing: {expected_best}")
        print("-" * 70)

        results[puzzle_name] = {}

        for framing in framings:
            print(f"\n  Training with '{framing}' framing...")

            if framing == 'grid_auto':
                # Two-phase: screen all corners, pick best, train fully
                loss_history, _, predictions, chosen_corner = train_grid_two_phase(
                    puzzle, screening_epochs=2, full_epochs=epochs, verbose=True
                )
                # Store which corner was chosen
                actual_framing = chosen_corner
                print(f"    Two-phase chose: {chosen_corner}")
            else:
                loss_history, _, predictions = train_with_framing(puzzle, framing, epochs=epochs)
                actual_framing = framing

            final_loss = loss_history[-1]
            min_loss = min(loss_history)

            # Compute convergence speed (epochs to reach 10% of initial loss)
            initial_loss = loss_history[0]
            threshold = initial_loss * 0.1
            convergence_epoch = epochs  # Default if never converges
            for i, loss in enumerate(loss_history):
                if loss < threshold:
                    convergence_epoch = i
                    break

            # Compute PIXEL ACCURACY - the real metric
            pixel_acc, mean_error = compute_pixel_accuracy(predictions, puzzle, actual_framing)

            results[puzzle_name][framing] = {
                'loss_history': loss_history,
                'final_loss': final_loss,
                'min_loss': min_loss,
                'convergence_epoch': convergence_epoch,
                'pixel_accuracy': pixel_acc,
                'mean_pixel_error': mean_error,
                'actual_framing': actual_framing,  # Track what was actually used
            }

            print(f"    Target loss: {final_loss:.6f}")
            print(f"    Convergence: epoch {convergence_epoch}")
            print(f"    PIXEL ACCURACY: {pixel_acc*100:.1f}% (mean error: {mean_error:.2f} px)")

        # Determine actual best by PIXEL ACCURACY (primary), then mean error (tiebreaker)
        def score_framing(f):
            r = results[puzzle_name][f]
            # Higher accuracy is better, lower error is better
            return (-r['pixel_accuracy'], r['mean_pixel_error'])
        actual_best = min(framings, key=score_framing)

        # For grid puzzles, check if grid_auto picked the right corner
        if expected_best.startswith('grid_'):
            chosen = results[puzzle_name]['grid_auto']['actual_framing']
            corner_match = "✓" if chosen == expected_best else "✗"
            print(f"\n  Grid corner selection: {corner_match} (expected {expected_best}, got {chosen})")
            match = "✓ CORRECT" if actual_best == 'grid_auto' else "✗ WRONG"
        else:
            match = "✓ CORRECT" if actual_best == expected_best else "✗ WRONG"

        print(f"  RESULT (by pixel accuracy): Best = '{actual_best}' {match}")

        # Show target variance for each framing
        print(f"\n  Target statistics:")
        for framing in framings:
            if framing == 'grid_auto':
                actual = results[puzzle_name][framing]['actual_framing']
                _, targets = compute_targets(puzzle, actual)
                print(f"    {framing} ({actual}): target variance = {targets.var():.6f}")
            else:
                _, targets = compute_targets(puzzle, framing)
                print(f"    {framing}: target variance = {targets.var():.6f}")
    
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    # Plot results - bar chart of pixel accuracy
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for idx, (puzzle_name, puzzle, expected_best) in enumerate(puzzles):
        ax = axes[idx]

        accuracies = [results[puzzle_name][f]['pixel_accuracy'] * 100 for f in framings]

        # Color logic: green if this is the expected best framing
        # For grid puzzles, grid_auto is expected to win
        colors = []
        for f in framings:
            if expected_best.startswith('grid_') and f == 'grid_auto':
                colors.append('green')
            elif f == expected_best:
                colors.append('green')
            else:
                colors.append('steelblue')

        # Build labels showing chosen corner for grid_auto
        labels = []
        for f in framings:
            if f == 'grid_auto':
                chosen = results[puzzle_name][f]['actual_framing']
                labels.append(f'grid_auto\n({chosen})')
            else:
                labels.append(f)

        bars = ax.bar(labels, accuracies, color=colors)
        ax.set_ylabel('Pixel Accuracy (%)')
        ax.set_title(puzzle_name)
        ax.set_ylim(0, 105)
        ax.axhline(100, color='gray', linestyle='--', alpha=0.5)

        # Add value labels on bars
        for bar, acc in zip(bars, accuracies):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                   f'{acc:.1f}%', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig('anchor_validation_results.png', dpi=150, bbox_inches='tight')
    print("\nPlot saved to: anchor_validation_results.png")
    plt.show()
    
    # Final verification using PIXEL ACCURACY
    print("\n" + "=" * 70)
    print("VALIDATION VERDICT (by Pixel Accuracy)")
    print("=" * 70)

    all_correct = True
    grid_corner_correct = True

    for puzzle_name, puzzle, expected_best in puzzles:
        def score_framing(f):
            r = results[puzzle_name][f]
            return (-r['pixel_accuracy'], r['mean_pixel_error'])
        actual_best = min(framings, key=score_framing)

        best_acc = results[puzzle_name][actual_best]['pixel_accuracy']

        if expected_best.startswith('grid_'):
            # For grid puzzles: check if grid_auto won AND picked correct corner
            chosen = results[puzzle_name]['grid_auto']['actual_framing']
            framing_correct = (actual_best == 'grid_auto')
            corner_correct = (chosen == expected_best)
            correct = framing_correct and corner_correct
            grid_corner_correct = grid_corner_correct and corner_correct

            symbol = "✓" if correct else "✗"
            corner_symbol = "✓" if corner_correct else "✗"
            print(f"  {symbol} {puzzle_name}:")
            print(f"      Framing winner: {'grid_auto' if framing_correct else actual_best} ({best_acc*100:.1f}%)")
            print(f"      Corner selection: {corner_symbol} expected {expected_best}, got {chosen}")
        else:
            correct = (actual_best == expected_best)
            symbol = "✓" if correct else "✗"
            print(f"  {symbol} {puzzle_name}: expected '{expected_best}', got '{actual_best}' ({best_acc*100:.1f}%)")

        all_correct = all_correct and correct

    print()
    if all_correct:
        print("SUCCESS: All framings correct, including grid corner auto-selection!")
        print("The 2-epoch screening reliably identifies the correct corner.")
    elif grid_corner_correct:
        print("PARTIAL: Grid corner selection worked, but some other framings failed.")
    else:
        print("PARTIAL: Grid corner auto-selection failed.")
        print("May need more screening epochs or different selection criteria.")
    
    return results


# =============================================================================
# Additional Analysis: Target Distribution Visualization
# =============================================================================

def visualize_target_distributions():
    """
    Show why certain framings are easier - their targets are more concentrated.
    Shows all 4 grid corners to demonstrate why screening works.
    """
    print("\n" + "=" * 70)
    print("TARGET DISTRIBUTION ANALYSIS")
    print("=" * 70)

    puzzle = generate_grid_corner_puzzle(num_examples=20, corner='top_left')

    # Show delta, all 4 grid corners, and object_relative
    framings = ['delta', 'grid_tl', 'grid_tr', 'grid_bl', 'grid_br', 'object_relative']
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    axes = axes.flatten()

    for idx, framing in enumerate(framings):
        _, targets = compute_targets(puzzle, framing)
        targets_flat = targets.reshape(-1, 2)

        ax = axes[idx]
        ax.scatter(targets_flat[:, 1], targets_flat[:, 0], alpha=0.6, s=50)
        ax.set_xlabel('Column target')
        ax.set_ylabel('Row target')
        # Highlight the correct framing for this puzzle
        title = f'{framing}\nvar={targets.var():.4f}'
        if framing == 'grid_tl':
            title += ' ← BEST'
            ax.set_facecolor('#e8f5e9')
        ax.set_title(title)
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
        ax.axvline(0, color='gray', linestyle='--', alpha=0.5)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')

    plt.suptitle('Target distributions for "Move to top-left corner" puzzle\n'
                 '(Lower variance = easier to learn; grid_tl should cluster tightly)', fontsize=12)
    plt.tight_layout()
    plt.savefig('target_distributions.png', dpi=150, bbox_inches='tight')
    print("Target distribution plot saved to: target_distributions.png")
    plt.show()


if __name__ == "__main__":
    results = run_validation_experiment()
    visualize_target_distributions()