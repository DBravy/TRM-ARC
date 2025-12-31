#!/usr/bin/env python3
"""
Test script for the Output Autoencoder.

Visualizes how the autoencoder learns to reconstruct output grids over training.
"""

import argparse
import json
import os
import random
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

# ARC color palette (same as in crm.py)
ARC_COLORS = [
    (0, 0, 0),        # 0: black
    (0, 116, 217),    # 1: blue
    (255, 65, 54),    # 2: red
    (46, 204, 64),    # 3: green
    (255, 220, 0),    # 4: yellow
    (170, 170, 170),  # 5: gray
    (240, 18, 190),   # 6: magenta
    (255, 133, 27),   # 7: orange
    (127, 219, 255),  # 8: cyan
    (135, 12, 37),    # 9: maroon
]

NUM_COLORS = 10


def grid_to_ansi(grid: np.ndarray, max_h: int = None, max_w: int = None) -> str:
    """Convert grid to ANSI colored string."""
    h, w = grid.shape
    if max_h:
        h = min(h, max_h)
    if max_w:
        w = min(w, max_w)

    lines = []
    for r in range(h):
        row = []
        for c in range(w):
            color = int(grid[r, c])
            r_val, g_val, b_val = ARC_COLORS[color]
            row.append(f"\033[48;2;{r_val};{g_val};{b_val}m  \033[0m")
        lines.append("".join(row))
    return "\n".join(lines)


def dihedral_transform(arr: np.ndarray, tid: int) -> np.ndarray:
    """8 dihedral symmetries"""
    if tid == 0:
        return arr
    elif tid == 1:
        return np.rot90(arr, k=1)
    elif tid == 2:
        return np.rot90(arr, k=2)
    elif tid == 3:
        return np.rot90(arr, k=3)
    elif tid == 4:
        return np.fliplr(arr)
    elif tid == 5:
        return np.flipud(arr)
    elif tid == 6:
        return arr.T
    elif tid == 7:
        return np.fliplr(np.rot90(arr, k=1))
    return arr


def random_color_permutation() -> np.ndarray:
    """Create random color permutation (keeping 0 fixed)"""
    mapping = np.concatenate([
        np.array([0], dtype=np.uint8),
        np.random.permutation(np.arange(1, 10, dtype=np.uint8))
    ])
    return mapping


def apply_augmentation(grid: np.ndarray, trans_id: int, color_map: np.ndarray) -> np.ndarray:
    """Apply dihedral transform and color permutation"""
    return dihedral_transform(color_map[grid], trans_id)


def get_random_augmentation():
    """Get random augmentation parameters"""
    trans_id = random.randint(0, 7)
    color_map = random_color_permutation()
    return trans_id, color_map


class OutputAutoencoder(nn.Module):
    """Small convolutional autoencoder for learning valid output structure."""

    def __init__(self, hidden_dim: int = 32, bottleneck_size: int = 4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.bottleneck_size = bottleneck_size

        self.encoder_convs = nn.Sequential(
            nn.Conv2d(NUM_COLORS, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim * 2),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d(bottleneck_size)

        self.decoder = nn.Sequential(
            nn.Conv2d(hidden_dim * 2, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, NUM_COLORS, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor, target_size=None) -> torch.Tensor:
        if target_size is None:
            target_size = (x.size(2), x.size(3))

        z = self.encoder_convs(x)
        z = self.pool(z)
        z_upsampled = F.interpolate(z, size=target_size, mode='bilinear', align_corners=False)
        logits = self.decoder(z_upsampled)
        return logits

    def reconstruct(self, grid: torch.Tensor) -> torch.Tensor:
        onehot = F.one_hot(grid.long(), num_classes=NUM_COLORS).float()
        onehot = onehot.permute(0, 3, 1, 2)
        logits = self.forward(onehot)
        return logits.argmax(dim=1)


def load_puzzle(puzzle_id: str, data_root: str = "kaggle/combined") -> dict:
    """Load a single puzzle by ID."""
    for subset in ["training", "evaluation", "training2", "evaluation2"]:
        challenges_path = f"{data_root}/arc-agi_{subset}_challenges.json"
        if os.path.exists(challenges_path):
            with open(challenges_path) as f:
                puzzles = json.load(f)
            if puzzle_id in puzzles:
                return puzzles[puzzle_id]
    raise ValueError(f"Puzzle {puzzle_id} not found")


def visualize_reconstruction(autoencoder, grid: np.ndarray, device: torch.device, title: str = ""):
    """Visualize original vs reconstructed grid."""
    autoencoder.eval()
    with torch.no_grad():
        grid_t = torch.from_numpy(grid).long().unsqueeze(0).to(device)
        recon = autoencoder.reconstruct(grid_t)[0].cpu().numpy()

    # Compute accuracy
    correct = (grid == recon).sum()
    total = grid.size
    acc = correct / total

    print(f"\n{title}")
    print(f"Accuracy: {correct}/{total} = {acc:.1%}")
    print()
    print("Original:".ljust(grid.shape[1] * 2 + 5) + "Reconstructed:")

    orig_lines = grid_to_ansi(grid).split("\n")
    recon_lines = grid_to_ansi(recon).split("\n")

    for o, r in zip(orig_lines, recon_lines):
        print(o + "     " + r)

    return acc


def train_and_visualize(
    output_grids: List[np.ndarray],
    num_steps: int = 200,
    hidden_dim: int = 32,
    bottleneck_size: int = 4,
    lr: float = 0.001,
    viz_interval: int = 25,
    use_augment: bool = True,
    dihedral_only: bool = False,
):
    """Train autoencoder and visualize progress."""
    device = torch.device('cpu')  # Use CPU for compatibility

    autoencoder = OutputAutoencoder(hidden_dim=hidden_dim, bottleneck_size=bottleneck_size).to(device)
    optimizer = torch.optim.AdamW(autoencoder.parameters(), lr=lr, weight_decay=0.01)

    param_count = sum(p.numel() for p in autoencoder.parameters())
    print(f"Autoencoder parameters: {param_count:,}")
    print(f"Bottleneck: {bottleneck_size}x{bottleneck_size} spatial")
    print(f"Training on {len(output_grids)} output grids")
    print(f"Augmentation: {'dihedral-only' if dihedral_only else 'full' if use_augment else 'none'}")
    print()

    # Pick a fixed test grid for visualization
    test_grid = output_grids[0]

    # Initial visualization
    print("=" * 60)
    visualize_reconstruction(autoencoder, test_grid, device, "BEFORE TRAINING (Step 0)")
    print("=" * 60)

    # Training loop
    losses = []
    autoencoder.train()

    pbar = tqdm(range(num_steps), desc="Training")
    for step in pbar:
        # Sample and augment
        grid = random.choice(output_grids)
        grid = np.array(grid, dtype=np.uint8)

        if use_augment:
            if dihedral_only:
                trans_id = random.randint(0, 7)
                color_map = np.arange(10, dtype=np.uint8)
            else:
                trans_id, color_map = get_random_augmentation()
            aug_grid = apply_augmentation(grid, trans_id, color_map)
        else:
            aug_grid = grid

        # Forward pass
        grid_t = torch.from_numpy(aug_grid.copy()).long().unsqueeze(0).to(device)
        onehot = F.one_hot(grid_t, num_classes=NUM_COLORS).float()
        onehot = onehot.permute(0, 3, 1, 2)

        optimizer.zero_grad()
        logits = autoencoder(onehot)
        loss = F.cross_entropy(logits, grid_t)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        pbar.set_postfix(loss=loss.item())

        # Periodic visualization
        if (step + 1) % viz_interval == 0:
            print("\n" + "=" * 60)
            acc = visualize_reconstruction(autoencoder, test_grid, device, f"Step {step + 1} (loss={loss.item():.4f})")
            print("=" * 60)
            autoencoder.train()

    # Final visualization
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)

    total_acc = 0
    for i, grid in enumerate(output_grids):
        grid = np.array(grid, dtype=np.uint8)
        acc = visualize_reconstruction(autoencoder, grid, device, f"Output Grid {i + 1}")
        total_acc += acc

    print(f"\nAverage reconstruction accuracy: {total_acc / len(output_grids):.1%}")
    print(f"Final loss: {losses[-1]:.4f}")

    return autoencoder, losses


def main():
    parser = argparse.ArgumentParser(description="Test Output Autoencoder")
    parser.add_argument("--puzzle", type=str, required=True, help="Puzzle ID to test on")
    parser.add_argument("--data-root", type=str, default="kaggle/combined")
    parser.add_argument("--steps", type=int, default=150000, help="Training steps")
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--bottleneck-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--viz-interval", type=int, default=50, help="Steps between visualizations")
    parser.add_argument("--no-augment", action="store_true")
    parser.add_argument("--dihedral-only", action="store_true")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    print(f"Loading puzzle: {args.puzzle}")
    puzzle = load_puzzle(args.puzzle, args.data_root)

    output_grids = [np.array(ex['output'], dtype=np.uint8) for ex in puzzle['train']]
    print(f"Found {len(output_grids)} training examples")

    for i, grid in enumerate(output_grids):
        print(f"\nTraining Output {i + 1}: {grid.shape}")
        print(grid_to_ansi(grid))

    print("\n" + "=" * 60)
    print("TRAINING AUTOENCODER")
    print("=" * 60)

    train_and_visualize(
        output_grids,
        num_steps=args.steps,
        hidden_dim=args.hidden_dim,
        bottleneck_size=args.bottleneck_size,
        lr=args.lr,
        viz_interval=args.viz_interval,
        use_augment=not args.no_augment,
        dihedral_only=args.dihedral_only,
    )


if __name__ == "__main__":
    main()
