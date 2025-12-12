#!/usr/bin/env python3
"""
Output Grid Size Prediction

Given an input grid, predict the height and width of the output grid.
This is a diagnostic to see if the network can learn size-related rules.

Usage:
    python predict_grid_size.py --puzzle-id 00d62c1b
    python predict_grid_size.py --puzzle-id 00d62c1b --epochs 200
"""

import argparse
import json
import os
import random
from typing import Dict, List, Tuple

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
MAX_DIM = 30  # Maximum grid dimension


# =============================================================================
# Augmentation
# =============================================================================

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


def get_output_size_after_transform(h: int, w: int, tid: int) -> Tuple[int, int]:
    """Get output dimensions after dihedral transform"""
    if tid in [0, 2, 4, 5]:  # No rotation or 180 or flips
        return h, w
    else:  # 90, 270 rotations or transpose
        return w, h


# =============================================================================
# Model
# =============================================================================

class GridSizePredictor(nn.Module):
    """
    CNN that predicts output grid dimensions from input grid.
    
    Two classification heads:
    - Height predictor (1-30)
    - Width predictor (1-30)
    """
    
    def __init__(self, hidden_dim: int = 64):
        super().__init__()
        
        # Color embedding
        self.embed = nn.Embedding(NUM_COLORS, 16)
        
        # Encoder - standard CNN with pooling
        self.conv1 = nn.Sequential(
            nn.Conv2d(16, hidden_dim, 3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)  # 30 -> 15
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim * 2, 3, padding=1),
            nn.BatchNorm2d(hidden_dim * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim * 2, hidden_dim * 2, 3, padding=1),
            nn.BatchNorm2d(hidden_dim * 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)  # 15 -> 7
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(hidden_dim * 2, hidden_dim * 4, 3, padding=1),
            nn.BatchNorm2d(hidden_dim * 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim * 4, hidden_dim * 4, 3, padding=1),
            nn.BatchNorm2d(hidden_dim * 4),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1)  # Global average pooling
        )
        
        # Classification heads
        self.height_head = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim * 2, MAX_DIM)  # Predict 1-30 (0-indexed: 0-29)
        )
        
        self.width_head = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim * 2, MAX_DIM)  # Predict 1-30 (0-indexed: 0-29)
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input grid (B, H, W) with values 0-9
        
        Returns:
            height_logits: (B, 30) logits for height prediction
            width_logits: (B, 30) logits for width prediction
        """
        # Embed colors
        x = self.embed(x)  # (B, H, W, 16)
        x = x.permute(0, 3, 1, 2)  # (B, 16, H, W)
        
        # CNN forward
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        
        # Flatten
        x = x.view(x.size(0), -1)  # (B, hidden_dim * 4)
        
        # Predict dimensions
        height_logits = self.height_head(x)
        width_logits = self.width_head(x)
        
        return height_logits, width_logits
    
    def predict(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return predicted height and width (1-indexed)"""
        h_logits, w_logits = self.forward(x)
        h_pred = h_logits.argmax(dim=1) + 1  # Convert to 1-indexed
        w_pred = w_logits.argmax(dim=1) + 1
        return h_pred, w_pred


# =============================================================================
# Dataset
# =============================================================================

class GridSizeDataset(Dataset):
    """Dataset for grid size prediction with augmentation."""
    
    def __init__(self, examples: List[Dict], augment: bool = True, num_augments: int = 8):
        self.examples = examples
        self.augment = augment
        self.num_augments = num_augments
        
        # Build samples
        self.samples = []
        for ex in examples:
            inp = np.array(ex['input'], dtype=np.uint8)
            out = np.array(ex['output'], dtype=np.uint8)
            out_h, out_w = out.shape
            
            if augment:
                # Add augmented versions
                for tid in range(8):
                    for _ in range(num_augments // 8 + 1):
                        color_map = random_color_permutation()
                        aug_inp = dihedral_transform(color_map[inp], tid)
                        aug_h, aug_w = get_output_size_after_transform(out_h, out_w, tid)
                        self.samples.append((aug_inp, aug_h, aug_w))
            else:
                self.samples.append((inp, out_h, out_w))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        inp, out_h, out_w = self.samples[idx]
        
        # Pad to 30x30
        padded = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.uint8)
        h, w = inp.shape
        padded[:h, :w] = inp
        
        return {
            'input': torch.tensor(padded, dtype=torch.long),
            'height': torch.tensor(out_h - 1, dtype=torch.long),  # 0-indexed for CE loss
            'width': torch.tensor(out_w - 1, dtype=torch.long),
        }


# =============================================================================
# Training
# =============================================================================

def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    correct_h = 0
    correct_w = 0
    correct_both = 0
    total = 0
    
    for batch in loader:
        inp = batch['input'].to(device)
        h_target = batch['height'].to(device)
        w_target = batch['width'].to(device)
        
        optimizer.zero_grad()
        h_logits, w_logits = model(inp)
        
        loss_h = F.cross_entropy(h_logits, h_target)
        loss_w = F.cross_entropy(w_logits, w_target)
        loss = loss_h + loss_w
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        h_pred = h_logits.argmax(dim=1)
        w_pred = w_logits.argmax(dim=1)
        
        correct_h += (h_pred == h_target).sum().item()
        correct_w += (w_pred == w_target).sum().item()
        correct_both += ((h_pred == h_target) & (w_pred == w_target)).sum().item()
        total += inp.size(0)
    
    return {
        'loss': total_loss / len(loader),
        'height_acc': correct_h / total,
        'width_acc': correct_w / total,
        'both_acc': correct_both / total,
    }


def evaluate(model, examples, device):
    """Evaluate on a list of examples (no augmentation)."""
    model.eval()
    
    results = []
    with torch.no_grad():
        for ex in examples:
            inp = np.array(ex['input'], dtype=np.uint8)
            out = np.array(ex['output'], dtype=np.uint8)
            true_h, true_w = out.shape
            
            # Pad and predict
            padded = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.uint8)
            h, w = inp.shape
            padded[:h, :w] = inp
            
            inp_tensor = torch.tensor(padded, dtype=torch.long).unsqueeze(0).to(device)
            pred_h, pred_w = model.predict(inp_tensor)
            pred_h = pred_h.item()
            pred_w = pred_w.item()
            
            results.append({
                'true_h': true_h,
                'true_w': true_w,
                'pred_h': pred_h,
                'pred_w': pred_w,
                'h_correct': pred_h == true_h,
                'w_correct': pred_w == true_w,
                'both_correct': pred_h == true_h and pred_w == true_w,
            })
    
    return results


# =============================================================================
# Main
# =============================================================================

def load_puzzle(puzzle_id: str, data_root: str = "kaggle/combined") -> Dict:
    """Load a puzzle by ID from combined dataset."""
    for dataset in ["arc-agi-1", "arc-agi-2", "arc-agi-3"]:
        for split in ["training", "evaluation"]:
            path = os.path.join(data_root, dataset, split, f"{puzzle_id}.json")
            if os.path.exists(path):
                with open(path) as f:
                    return json.load(f)
    raise FileNotFoundError(f"Puzzle {puzzle_id} not found in {data_root}")


def main():
    parser = argparse.ArgumentParser(description="Predict output grid size from input")
    parser.add_argument("--puzzle-id", type=str, required=True, help="Puzzle ID to train on")
    parser.add_argument("--data-root", type=str, default="kaggle/combined")
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num-augments", type=int, default=64, help="Augmentation multiplier")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    print(f"Device: {DEVICE}")
    print(f"Puzzle: {args.puzzle_id}")
    
    # Load puzzle
    puzzle = load_puzzle(args.puzzle_id, args.data_root)
    train_examples = puzzle['train']
    test_examples = puzzle['test']
    
    print(f"\nTraining examples: {len(train_examples)}")
    print(f"Test examples: {len(test_examples)}")
    
    # Show ground truth sizes
    print("\nGround truth output sizes:")
    print("  Training:")
    for i, ex in enumerate(train_examples):
        out = np.array(ex['output'])
        inp = np.array(ex['input'])
        print(f"    Example {i}: input {inp.shape} -> output {out.shape}")
    print("  Test:")
    for i, ex in enumerate(test_examples):
        out = np.array(ex['output'])
        inp = np.array(ex['input'])
        print(f"    Example {i}: input {inp.shape} -> output {out.shape}")
    
    # Create dataset and loader
    train_dataset = GridSizeDataset(train_examples, augment=True, num_augments=args.num_augments)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    
    print(f"\nTraining samples (with augmentation): {len(train_dataset)}")
    
    # Create model
    model = GridSizePredictor(hidden_dim=args.hidden_dim).to(DEVICE)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    
    # Training loop
    print("\n" + "=" * 60)
    print("Training")
    print("=" * 60)
    
    best_train_acc = 0
    for epoch in range(args.epochs):
        metrics = train_epoch(model, train_loader, optimizer, DEVICE)
        scheduler.step()
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch + 1:3d}: Loss={metrics['loss']:.4f}, "
                  f"H_Acc={metrics['height_acc']:.2%}, W_Acc={metrics['width_acc']:.2%}, "
                  f"Both={metrics['both_acc']:.2%}")
        
        if metrics['both_acc'] > best_train_acc:
            best_train_acc = metrics['both_acc']
    
    # Evaluate on training examples (no augmentation)
    print("\n" + "=" * 60)
    print("Evaluation on Training Examples (no augmentation)")
    print("=" * 60)
    
    train_results = evaluate(model, train_examples, DEVICE)
    for i, r in enumerate(train_results):
        status = "✓" if r['both_correct'] else "✗"
        print(f"  Example {i}: true=({r['true_h']}, {r['true_w']}), "
              f"pred=({r['pred_h']}, {r['pred_w']}) {status}")
    
    train_correct = sum(r['both_correct'] for r in train_results)
    print(f"\nTraining accuracy: {train_correct}/{len(train_results)}")
    
    # Evaluate on test examples
    print("\n" + "=" * 60)
    print("Evaluation on TEST Examples (generalization)")
    print("=" * 60)
    
    test_results = evaluate(model, test_examples, DEVICE)
    for i, r in enumerate(test_results):
        status = "✓" if r['both_correct'] else "✗"
        print(f"  Example {i}: true=({r['true_h']}, {r['true_w']}), "
              f"pred=({r['pred_h']}, {r['pred_w']}) {status}")
    
    test_correct = sum(r['both_correct'] for r in test_results)
    print(f"\nTest accuracy: {test_correct}/{len(test_results)}")
    
    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Training: {train_correct}/{len(train_results)} correct")
    print(f"Test:     {test_correct}/{len(test_results)} correct")
    
    if test_correct == len(test_results):
        print("\n✓ Perfect generalization! The network learned the size rule.")
    elif train_correct == len(train_results) and test_correct == 0:
        print("\n✗ Memorized training but failed to generalize.")
    elif train_correct < len(train_results):
        print("\n✗ Couldn't even fit training data.")
    else:
        print("\n~ Partial generalization.")


if __name__ == "__main__":
    main()