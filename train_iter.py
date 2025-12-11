#!/usr/bin/env python3
"""
Train CNN with iterative refinement during training.

Instead of pre-generating candidate types (zeros, corrupted, etc.),
we train iteratively:
1. Start with blank candidate
2. Predict, compute loss, backprop
3. Detach prediction, use as next candidate
4. Repeat for N iterations

This teaches the model to improve from any state.

Usage:
    python train_iterative.py --puzzle 00d62c1b --epochs 50 --iterations 5
    python train_iterative.py --puzzle 00d62c1b --epochs 50 --iterations 5 --visualize
"""

import argparse
import json
import os
import random
from typing import Dict, List, Set

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# Determine device
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

GRID_SIZE = 30
NUM_COLORS = 10


# =============================================================================
# Model
# =============================================================================

class DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class Down(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.pool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_ch, out_ch)
        )

    def forward(self, x):
        return self.pool_conv(x)


class Up(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diff_h = x2.size(2) - x1.size(2)
        diff_w = x2.size(3) - x1.size(3)
        x1 = F.pad(x1, [diff_w // 2, diff_w - diff_w // 2,
                       diff_h // 2, diff_h - diff_h // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class PixelErrorCNN(nn.Module):
    def __init__(self, hidden_dim: int = 64, num_classes: int = 10):
        super().__init__()
        self.num_classes = num_classes
        input_channels = num_classes * 2
        
        self.inc = DoubleConv(input_channels, hidden_dim)
        self.down1 = Down(hidden_dim, hidden_dim * 2)
        self.down2 = Down(hidden_dim * 2, hidden_dim * 4)
        self.up1 = Up(hidden_dim * 4, hidden_dim * 2)
        self.up2 = Up(hidden_dim * 2, hidden_dim)
        self.outc = nn.Conv2d(hidden_dim, num_classes, 1)

    def forward(self, input_grid, candidate_output):
        input_onehot = F.one_hot(input_grid, self.num_classes).float()
        input_onehot = input_onehot.permute(0, 3, 1, 2)
        candidate_onehot = F.one_hot(candidate_output, self.num_classes).float()
        candidate_onehot = candidate_onehot.permute(0, 3, 1, 2)
        
        x = torch.cat([input_onehot, candidate_onehot], dim=1)
        
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x = self.up1(x3, x2)
        x = self.up2(x, x1)
        return self.outc(x)

    def predict_colors(self, input_grid, candidate_output):
        logits = self.forward(input_grid, candidate_output)
        return logits.argmax(dim=1)


# =============================================================================
# Augmentation
# =============================================================================

def dihedral_transform(grid: np.ndarray, trans_id: int) -> np.ndarray:
    if trans_id == 0:
        return grid
    elif trans_id == 1:
        return np.rot90(grid, 1)
    elif trans_id == 2:
        return np.rot90(grid, 2)
    elif trans_id == 3:
        return np.rot90(grid, 3)
    elif trans_id == 4:
        return np.fliplr(grid)
    elif trans_id == 5:
        return np.flipud(grid)
    elif trans_id == 6:
        return np.fliplr(np.rot90(grid, 1))
    elif trans_id == 7:
        return np.flipud(np.rot90(grid, 1))
    return grid


def random_color_permutation() -> np.ndarray:
    perm = np.arange(10, dtype=np.uint8)
    non_zero = perm[1:].copy()
    np.random.shuffle(non_zero)
    perm[1:] = non_zero
    return perm


def get_palette(grid: np.ndarray, include_zero: bool = False) -> Set[int]:
    unique = set(np.unique(grid).tolist())
    if not include_zero:
        unique.discard(0)
    return unique


def analyze_puzzle_palettes(puzzle: Dict) -> Dict:
    train_examples = puzzle.get("train", [])
    
    if len(train_examples) == 0:
        return {"recommendation": "dihedral_only", "reason": "no_training_examples"}
    
    input_palettes = []
    output_palettes = []
    combined_palettes = []
    
    for ex in train_examples:
        inp = np.array(ex["input"], dtype=np.uint8)
        out = np.array(ex["output"], dtype=np.uint8)
        
        inp_pal = get_palette(inp)
        out_pal = get_palette(out)
        combined = inp_pal | out_pal
        
        input_palettes.append(inp_pal)
        output_palettes.append(out_pal)
        combined_palettes.append(combined)
    
    identical_combined = all(p == combined_palettes[0] for p in combined_palettes)
    colors_preserved = all(
        output_palettes[i] <= (input_palettes[i] | {0})
        for i in range(len(train_examples))
    )
    new_colors_in_output = any(
        output_palettes[i] - input_palettes[i] - {0}
        for i in range(len(train_examples))
    )
    
    if identical_combined:
        return {"recommendation": "dihedral_only", "reason": "identical_palettes"}
    elif colors_preserved and not new_colors_in_output:
        return {"recommendation": "full_augment", "reason": "colors_from_input"}
    else:
        return {"recommendation": "full_augment", "reason": "varying_palettes"}


# =============================================================================
# Dataset - Simplified for iterative training
# =============================================================================

class IterativeTrainingDataset(Dataset):
    """
    Simple dataset that provides (input, target) pairs with augmentation.
    The iterative refinement happens in the training loop, not here.
    """
    
    def __init__(
        self, 
        puzzle: Dict, 
        samples_per_example: int = 50,
        dihedral_only: bool = True,
    ):
        self.samples_per_example = samples_per_example
        self.dihedral_only = dihedral_only
        
        self.examples = []
        for example in puzzle.get("train", []):
            inp = np.array(example["input"], dtype=np.uint8)
            out = np.array(example["output"], dtype=np.uint8)
            self.examples.append((inp, out))
    
    def __len__(self):
        return len(self.examples) * self.samples_per_example
    
    def _pad_grid(self, grid: np.ndarray) -> np.ndarray:
        h, w = grid.shape
        padded = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.uint8)
        padded[:h, :w] = grid
        return padded
    
    def _get_augmentation(self):
        trans_id = random.randint(0, 7)
        if self.dihedral_only:
            color_map = np.arange(10, dtype=np.uint8)
        else:
            color_map = random_color_permutation()
        return trans_id, color_map
    
    def __getitem__(self, idx):
        example_idx = idx % len(self.examples)
        input_grid, target_output = self.examples[example_idx]
        
        # Apply augmentation
        trans_id, color_map = self._get_augmentation()
        aug_input = dihedral_transform(color_map[input_grid], trans_id)
        aug_target = dihedral_transform(color_map[target_output], trans_id)
        
        return (
            torch.from_numpy(self._pad_grid(aug_input)).long(),
            torch.from_numpy(self._pad_grid(aug_target)).long(),
        )


# =============================================================================
# Iterative Training
# =============================================================================

def train_iterative(
    puzzle: Dict,
    epochs: int = 50,
    iterations: int = 5,
    hidden_dim: int = 64,
    lr: float = 1e-3,
    batch_size: int = 32,
    samples_per_example: int = 50,
    dihedral_only: bool = True,
    verbose: bool = False,
) -> nn.Module:
    """
    Train with iterative refinement.
    
    Each training step:
    1. Start with blank candidate
    2. For each iteration:
       - Forward pass: logits = model(input, candidate)
       - Compute loss against target
       - Backward pass
       - Detach and argmax to get next candidate
    3. Optimizer step after all iterations
    """
    
    dataset = IterativeTrainingDataset(
        puzzle,
        samples_per_example=samples_per_example,
        dihedral_only=dihedral_only,
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    
    model = PixelErrorCNN(hidden_dim=hidden_dim, num_classes=10).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        iteration_losses = [0.0] * iterations
        
        for inp, target in loader:
            inp = inp.to(DEVICE)
            target = target.to(DEVICE)
            
            optimizer.zero_grad()
            
            # Start with blank candidate
            candidate = torch.zeros_like(target)
            
            batch_loss = 0
            for it in range(iterations):
                # Forward pass
                logits = model(inp, candidate)
                
                # Loss against target
                loss = F.cross_entropy(logits, target)
                
                # Backward pass for this iteration
                loss.backward()
                
                batch_loss += loss.item()
                iteration_losses[it] += loss.item()
                
                # Prepare next candidate (detached)
                candidate = logits.argmax(dim=1).detach()
            
            # Single optimizer step after all iterations
            optimizer.step()
            total_loss += batch_loss
        
        scheduler.step()
        
        if verbose and (epoch + 1) % 10 == 0:
            avg_loss = total_loss / len(loader)
            iter_avg = [l / len(loader) for l in iteration_losses]
            iter_str = ", ".join([f"it{i}={l:.3f}" for i, l in enumerate(iter_avg)])
            print(f"  Epoch {epoch+1:3d}: loss={avg_loss:.4f} [{iter_str}]")
    
    return model


# =============================================================================
# Evaluation
# =============================================================================

def evaluate_iterative(
    model: nn.Module, 
    puzzle: Dict, 
    iterations: int = 5,
    use_train: bool = False,
    visualize: bool = False,
) -> Dict:
    """Evaluate with iterative refinement."""
    model.eval()
    
    examples = puzzle.get("train" if use_train else "test", [])
    
    results = []
    iteration_accuracies = [[] for _ in range(iterations)]
    
    with torch.no_grad():
        for ex in examples:
            if "output" not in ex:
                continue
            
            inp = np.array(ex["input"], dtype=np.uint8)
            out = np.array(ex["output"], dtype=np.uint8)
            h_out, w_out = out.shape
            h_inp, w_inp = inp.shape
            
            # Pad input
            inp_pad = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.uint8)
            inp_pad[:h_inp, :w_inp] = inp
            
            inp_t = torch.from_numpy(inp_pad).long().unsqueeze(0).to(DEVICE)
            
            # Start with blank candidate
            candidate = torch.zeros(1, GRID_SIZE, GRID_SIZE, dtype=torch.long, device=DEVICE)
            
            # Iterate
            for it in range(iterations):
                pred = model.predict_colors(inp_t, candidate)
                candidate = pred
                
                # Calculate accuracy at this iteration
                pred_np = pred[0].cpu().numpy()
                correct = (pred_np[:h_out, :w_out] == out).sum()
                total = h_out * w_out
                acc = correct / total
                iteration_accuracies[it].append(acc)
            
            # Final result
            final_pred = candidate[0].cpu().numpy()
            correct = (final_pred[:h_out, :w_out] == out).sum()
            total = h_out * w_out
            
            results.append({
                "correct": int(correct),
                "total": total,
                "accuracy": correct / total,
                "perfect": correct == total,
            })
            
            if visualize:
                visualize_result(inp_pad, out, final_pred, h_out, w_out)
    
    model.train()
    
    total_correct = sum(r["correct"] for r in results)
    total_pixels = sum(r["total"] for r in results)
    
    return {
        "pixel_accuracy": total_correct / max(total_pixels, 1),
        "perfect_rate": sum(r["perfect"] for r in results) / max(len(results), 1),
        "all_perfect": all(r["perfect"] for r in results),
        "num_examples": len(results),
        "iteration_accuracies": [np.mean(accs) if accs else 0 for accs in iteration_accuracies],
    }


def visualize_result(inp, target, pred, h, w):
    """Simple text visualization."""
    COLORS = ['⬛', '🔵', '🔴', '🟢', '🟡', '⬜', '🟣', '🟠', '🔷', '🟤']
    
    print("\nInput:")
    for row in inp[:h, :w]:
        print("".join(COLORS[c] for c in row))
    
    print("\nTarget:")
    for row in target[:h, :w]:
        print("".join(COLORS[c] for c in row))
    
    print("\nPrediction:")
    for row in pred[:h, :w]:
        print("".join(COLORS[c] for c in row))
    
    # Show errors
    errors = (pred[:h, :w] != target[:h, :w])
    if errors.any():
        print("\nErrors (X = wrong):")
        for i in range(h):
            row_str = ""
            for j in range(w):
                if errors[i, j]:
                    row_str += "❌"
                else:
                    row_str += "✅"
            print(row_str)


# =============================================================================
# Data Loading
# =============================================================================

def load_puzzles(dataset_name: str, data_root: str = "kaggle/combined") -> Dict:
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
            for pid in puzzles:
                if pid in solutions:
                    for i, sol in enumerate(solutions[pid]):
                        if i < len(puzzles[pid]["test"]):
                            puzzles[pid]["test"][i]["output"] = sol
        
        all_puzzles.update(puzzles)
    
    return all_puzzles


def main():
    parser = argparse.ArgumentParser(description="Train CNN with iterative refinement")
    parser.add_argument("--puzzle", type=str, required=True, help="Puzzle ID to train on")
    parser.add_argument("--dataset", type=str, default="arc-agi-1")
    parser.add_argument("--data-root", type=str, default="kaggle/combined")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--iterations", type=int, default=5, 
                        help="Number of refinement iterations during training")
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--samples-per-example", type=int, default=50)
    parser.add_argument("--auto-augment", action="store_true", default=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--visualize", action="store_true")
    args = parser.parse_args()
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    print(f"Device: {DEVICE}")
    print(f"Puzzle: {args.puzzle}")
    print(f"Epochs: {args.epochs}")
    print(f"Training iterations: {args.iterations}")
    
    # Load puzzle
    puzzles = load_puzzles(args.dataset, args.data_root)
    if args.puzzle not in puzzles:
        print(f"Puzzle {args.puzzle} not found!")
        return
    
    puzzle = puzzles[args.puzzle]
    print(f"Training examples: {len(puzzle.get('train', []))}")
    print(f"Test examples: {len(puzzle.get('test', []))}")
    
    # Determine augmentation
    if args.auto_augment:
        palette_info = analyze_puzzle_palettes(puzzle)
        dihedral_only = (palette_info["recommendation"] == "dihedral_only")
        print(f"Augmentation: {'dihedral-only' if dihedral_only else 'full'} ({palette_info['reason']})")
    else:
        dihedral_only = True
        print("Augmentation: dihedral-only (manual)")
    
    # Train
    print(f"\n{'='*60}")
    print("TRAINING")
    print(f"{'='*60}")
    
    model = train_iterative(
        puzzle,
        epochs=args.epochs,
        iterations=args.iterations,
        hidden_dim=args.hidden_dim,
        samples_per_example=args.samples_per_example,
        dihedral_only=dihedral_only,
        verbose=True,
    )
    
    # Evaluate on test
    print(f"\n{'='*60}")
    print("TEST EVALUATION")
    print(f"{'='*60}")
    
    test_results = evaluate_iterative(
        model, puzzle, 
        iterations=args.iterations,
        use_train=False,
        visualize=args.visualize,
    )
    
    print(f"\nAccuracy by iteration:")
    for i, acc in enumerate(test_results["iteration_accuracies"]):
        delta = ""
        if i > 0:
            diff = acc - test_results["iteration_accuracies"][i-1]
            delta = f" ({diff:+.2%})"
        print(f"  Iteration {i}: {acc:.2%}{delta}")
    
    print(f"\nFinal Results:")
    print(f"  Pixel accuracy: {test_results['pixel_accuracy']:.2%}")
    print(f"  Perfect examples: {int(test_results['perfect_rate'] * test_results['num_examples'])}/{test_results['num_examples']}")
    print(f"  All perfect: {test_results['all_perfect']}")
    
    # Evaluate on train (sanity check)
    print(f"\n{'='*60}")
    print("TRAIN EVALUATION (sanity check)")
    print(f"{'='*60}")
    
    train_results = evaluate_iterative(
        model, puzzle,
        iterations=args.iterations, 
        use_train=True,
        visualize=False,
    )
    
    print(f"\nAccuracy by iteration:")
    for i, acc in enumerate(train_results["iteration_accuracies"]):
        delta = ""
        if i > 0:
            diff = acc - train_results["iteration_accuracies"][i-1]
            delta = f" ({diff:+.2%})"
        print(f"  Iteration {i}: {acc:.2%}{delta}")
    
    print(f"\nFinal Results:")
    print(f"  Pixel accuracy: {train_results['pixel_accuracy']:.2%}")
    print(f"  Perfect examples: {int(train_results['perfect_rate'] * train_results['num_examples'])}/{train_results['num_examples']}")
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    gap = train_results['pixel_accuracy'] - test_results['pixel_accuracy']
    print(f"  Train: {train_results['pixel_accuracy']:.2%}")
    print(f"  Test:  {test_results['pixel_accuracy']:.2%}")
    print(f"  Gap:   {gap:.2%}")


if __name__ == "__main__":
    main()