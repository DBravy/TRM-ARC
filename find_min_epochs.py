#!/usr/bin/env python3
"""
Find minimum epochs needed for perfect generalization.

Takes puzzles that achieved perfect generalization in a previous run,
re-trains them, and evaluates at multiple epoch checkpoints to determine
the minimum epochs needed.

Usage:
    python find_min_epochs.py --results cnn_generalization_results.json
"""

import argparse
import json
import os
import random
import time
from typing import Dict, List, Set

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

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
# Model (copied from eval_cnn_generalization.py)
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


class DorsalCNN(nn.Module):
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
# Augmentation utilities
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
        recommendation = "dihedral_only"
        reason = "identical_palettes"
    elif colors_preserved and not new_colors_in_output:
        recommendation = "full_augment"
        reason = "colors_from_input"
    else:
        recommendation = "full_augment"
        reason = "varying_palettes"
    
    return {"recommendation": recommendation, "reason": reason}


# =============================================================================
# Dataset
# =============================================================================

class SinglePuzzleDataset(Dataset):
    def __init__(
        self, 
        puzzle: Dict, 
        num_positives: int = 2,
        num_negatives: int = 8,
        dihedral_only: bool = True,
    ):
        self.num_positives = num_positives
        self.num_negatives = num_negatives
        self.dihedral_only = dihedral_only
        self.samples_per_example = num_positives + num_negatives
        
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
        example_idx = idx // self.samples_per_example
        sample_type_idx = idx % self.samples_per_example
        
        input_grid, correct_output = self.examples[example_idx]
        
        is_positive = sample_type_idx < self.num_positives
        
        if is_positive:
            trans_id, color_map = self._get_augmentation()
            aug_input = dihedral_transform(color_map[input_grid], trans_id)
            aug_output = dihedral_transform(color_map[correct_output], trans_id)
            candidate = aug_output
            target = aug_output
        else:
            # Negative: all zeros candidate
            trans_id, color_map = self._get_augmentation()
            aug_input = dihedral_transform(color_map[input_grid], trans_id)
            aug_output = dihedral_transform(color_map[correct_output], trans_id)
            candidate = np.zeros_like(aug_output)
            target = aug_output
        
        inp_padded = self._pad_grid(aug_input)
        cand_padded = self._pad_grid(candidate)
        tgt_padded = self._pad_grid(target)
        
        return (
            torch.from_numpy(inp_padded).long(),
            torch.from_numpy(cand_padded).long(),
            torch.from_numpy(tgt_padded).long(),
        )


# =============================================================================
# Training with checkpoint evaluation
# =============================================================================

def train_and_evaluate_checkpoints(
    puzzle: Dict,
    checkpoints: List[int],
    hidden_dim: int = 64,
    num_negatives: int = 200,
    dihedral_only: bool = True,
    batch_size: int = 32,
    lr: float = 1e-3,
) -> Dict[int, Dict]:
    """
    Train on a puzzle and evaluate at multiple epoch checkpoints.
    
    Returns:
        Dict mapping epoch -> evaluation results
    """
    max_epochs = max(checkpoints)
    
    dataset = SinglePuzzleDataset(
        puzzle,
        num_positives=2,
        num_negatives=num_negatives,
        dihedral_only=dihedral_only,
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    
    model = DorsalCNN(hidden_dim=hidden_dim, num_classes=10).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, max_epochs)
    
    results = {}
    
    model.train()
    for epoch in range(1, max_epochs + 1):
        for inp, candidate, target in loader:
            inp = inp.to(DEVICE)
            candidate = candidate.to(DEVICE)
            target = target.to(DEVICE)
            
            optimizer.zero_grad()
            logits = model(inp, candidate)
            loss = F.cross_entropy(logits, target)
            loss.backward()
            optimizer.step()
        
        scheduler.step()
        
        # Evaluate at checkpoints
        if epoch in checkpoints:
            eval_result = evaluate_on_test(model, puzzle)
            results[epoch] = eval_result
    
    return results


def evaluate_on_test(model: nn.Module, puzzle: Dict) -> Dict:
    """Evaluate model on test examples."""
    model.eval()
    test_examples = puzzle.get("test", [])
    
    total_correct = 0
    total_pixels = 0
    all_perfect = True
    
    with torch.no_grad():
        for ex in test_examples:
            if "output" not in ex:
                continue
            
            inp = np.array(ex["input"], dtype=np.uint8)
            out = np.array(ex["output"], dtype=np.uint8)
            h_out, w_out = out.shape
            h_inp, w_inp = inp.shape
            
            # Pad input
            inp_pad = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.uint8)
            inp_pad[:h_inp, :w_inp] = inp
            
            # Blank candidate
            candidate_pad = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.uint8)
            
            inp_t = torch.from_numpy(inp_pad).long().unsqueeze(0).to(DEVICE)
            candidate_t = torch.from_numpy(candidate_pad).long().unsqueeze(0).to(DEVICE)
            
            pred = model.predict_colors(inp_t, candidate_t)[0].cpu().numpy()
            
            correct = (pred[:h_out, :w_out] == out).sum()
            total = h_out * w_out
            
            total_correct += correct
            total_pixels += total
            
            if correct != total:
                all_perfect = False
    
    model.train()
    
    return {
        "pixel_accuracy": total_correct / max(total_pixels, 1),
        "all_perfect": all_perfect,
    }


# =============================================================================
# Data loading
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
    parser = argparse.ArgumentParser(description="Find minimum epochs for perfect generalization")
    parser.add_argument("--results", type=str, required=True, 
                        help="Path to previous results JSON file")
    parser.add_argument("--data-root", type=str, default="kaggle/combined")
    parser.add_argument("--checkpoints", type=str, default="5,10,15,20,25,30,35,40,45,50",
                        help="Comma-separated list of epoch checkpoints to evaluate")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="min_epochs_results.json")
    args = parser.parse_args()
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    checkpoints = [int(x) for x in args.checkpoints.split(",")]
    checkpoints.sort()
    
    print(f"Device: {DEVICE}")
    print(f"Checkpoints: {checkpoints}")
    
    # Load previous results
    with open(args.results) as f:
        prev_results = json.load(f)
    
    config = prev_results["config"]
    print(f"\nPrevious config:")
    print(f"  Dataset: {config['dataset']}")
    print(f"  Epochs: {config['epochs']}")
    print(f"  Num negatives: {config['num_negatives']}")
    print(f"  Auto augment: {config['auto_augment']}")
    
    # Get puzzles that achieved perfect generalization
    perfect_puzzles = [
        r for r in prev_results["results"] 
        if r["all_perfect"]
    ]
    print(f"\nPerfect puzzles from previous run: {len(perfect_puzzles)}")
    
    # Load puzzle data
    puzzles = load_puzzles(config["dataset"], args.data_root)
    
    # Track results per checkpoint
    checkpoint_results = {cp: {"perfect": 0, "accuracies": []} for cp in checkpoints}
    puzzle_results = []
    
    print(f"\nEvaluating {len(perfect_puzzles)} puzzles at {len(checkpoints)} checkpoints...")
    print("=" * 60)
    
    for puzzle_info in tqdm(perfect_puzzles, desc="Puzzles"):
        puzzle_id = puzzle_info["puzzle_id"]
        puzzle = puzzles[puzzle_id]
        
        # Determine augmentation strategy
        if config["auto_augment"]:
            palette_analysis = analyze_puzzle_palettes(puzzle)
            dihedral_only = (palette_analysis["recommendation"] == "dihedral_only")
        else:
            dihedral_only = config.get("dihedral_only", False)
        
        # Train and evaluate at checkpoints
        results = train_and_evaluate_checkpoints(
            puzzle,
            checkpoints=checkpoints,
            hidden_dim=config["hidden_dim"],
            num_negatives=config["num_negatives"],
            dihedral_only=dihedral_only,
        )
        
        # Record results
        puzzle_checkpoint_data = {"puzzle_id": puzzle_id}
        first_perfect_epoch = None
        
        for cp in checkpoints:
            r = results[cp]
            checkpoint_results[cp]["accuracies"].append(r["pixel_accuracy"])
            if r["all_perfect"]:
                checkpoint_results[cp]["perfect"] += 1
                if first_perfect_epoch is None:
                    first_perfect_epoch = cp
            puzzle_checkpoint_data[f"epoch_{cp}"] = {
                "accuracy": r["pixel_accuracy"],
                "perfect": r["all_perfect"]
            }
        
        puzzle_checkpoint_data["first_perfect_epoch"] = first_perfect_epoch
        puzzle_results.append(puzzle_checkpoint_data)
    
    # Print summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"\nPerfect generalization by epoch (out of {len(perfect_puzzles)} puzzles):")
    print("-" * 40)
    
    for cp in checkpoints:
        perfect_count = checkpoint_results[cp]["perfect"]
        avg_acc = np.mean(checkpoint_results[cp]["accuracies"])
        pct = 100 * perfect_count / len(perfect_puzzles)
        bar = "█" * int(pct / 2)
        print(f"  Epoch {cp:3d}: {perfect_count:3d} ({pct:5.1f}%) avg_acc={avg_acc:.3f} {bar}")
    
    # Analyze first perfect epoch distribution
    first_perfect_epochs = [r["first_perfect_epoch"] for r in puzzle_results if r["first_perfect_epoch"]]
    never_perfect = sum(1 for r in puzzle_results if r["first_perfect_epoch"] is None)
    
    print(f"\nFirst epoch achieving perfect generalization:")
    print("-" * 40)
    for cp in checkpoints:
        count = sum(1 for e in first_perfect_epochs if e == cp)
        if count > 0:
            print(f"  Epoch {cp:3d}: {count:3d} puzzles first achieved perfect")
    if never_perfect > 0:
        print(f"  Never:     {never_perfect:3d} puzzles (didn't achieve perfect in this run)")
    
    # Find recommended epochs
    target_pcts = [90, 95, 99, 100]
    print(f"\nRecommended epochs to match target % of original perfect puzzles:")
    print("-" * 40)
    for target_pct in target_pcts:
        target_count = int(len(perfect_puzzles) * target_pct / 100)
        for cp in checkpoints:
            if checkpoint_results[cp]["perfect"] >= target_count:
                print(f"  {target_pct:3d}% ({target_count:2d} puzzles): epoch {cp}")
                break
        else:
            print(f"  {target_pct:3d}% ({target_count:2d} puzzles): >{checkpoints[-1]} epochs needed")
    
    # Save results
    output_data = {
        "config": {
            "previous_results": args.results,
            "checkpoints": checkpoints,
            "num_perfect_puzzles": len(perfect_puzzles),
            "seed": args.seed,
        },
        "checkpoint_summary": {
            cp: {
                "perfect_count": checkpoint_results[cp]["perfect"],
                "perfect_rate": checkpoint_results[cp]["perfect"] / len(perfect_puzzles),
                "avg_accuracy": float(np.mean(checkpoint_results[cp]["accuracies"])),
            }
            for cp in checkpoints
        },
        "puzzle_results": puzzle_results,
    }
    
    with open(args.output, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()