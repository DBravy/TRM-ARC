#!/usr/bin/env python3
"""
Probe CNN Pattern Storage Experiment

This script investigates HOW a U-Net stores a spatial transformation (180° rotation).

Experiments:
1. Train baseline model to perfect accuracy
2. Layer-wise gradient importance: Which layers have highest gradient magnitude?
3. Weight pruning: Does performance degrade gracefully or collapse suddenly?
4. Skip connection ablation: Zero out each skip individually - which matters most?
5. Activation statistics: What do activations look like at each layer?

Usage:
    python probe_pattern_storage.py --puzzle 3c9b0459 --epochs 50
"""

import argparse
import json
import os
import copy
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


# =============================================================================
# Model Architecture (simplified U-Net for probing)
# =============================================================================

class DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, name: str = ""):
        super().__init__()
        self.name = name
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
    def __init__(self, in_ch: int, out_ch: int, name: str = ""):
        super().__init__()
        self.name = name
        self.pool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_ch, out_ch, name=name)
        )

    def forward(self, x):
        return self.pool_conv(x)


class Up(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, name: str = ""):
        super().__init__()
        self.name = name
        self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_ch, out_ch, name=name)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diff_h = x2.size(2) - x1.size(2)
        diff_w = x2.size(3) - x1.size(3)
        x1 = F.pad(x1, [diff_w // 2, diff_w - diff_w // 2,
                       diff_h // 2, diff_h - diff_h // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class ProbeableUNet(nn.Module):
    """
    U-Net with hooks for probing internal activations and gradients.
    """

    def __init__(self, hidden_dim: int = 64, num_classes: int = 10):
        super().__init__()
        
        self.num_classes = num_classes
        
        # Color embeddings
        self.input_embed = nn.Embedding(NUM_COLORS, 16)
        self.output_embed = nn.Embedding(NUM_COLORS, 16)
        
        in_channels = 64  # With force_comparison
        base_ch = hidden_dim

        # Encoder - named for probing
        self.inc = DoubleConv(in_channels, base_ch, name="enc_input")
        self.down1 = Down(base_ch, base_ch * 2, name="enc_down1")
        self.down2 = Down(base_ch * 2, base_ch * 4, name="enc_down2")
        self.down3 = Down(base_ch * 4, base_ch * 8, name="enc_down3_bottleneck")

        # Decoder - named for probing
        self.up1 = Up(base_ch * 8, base_ch * 4, name="dec_up1")
        self.up2 = Up(base_ch * 4, base_ch * 2, name="dec_up2")
        self.up3 = Up(base_ch * 2, base_ch, name="dec_up3")

        self.outc = nn.Conv2d(base_ch, num_classes, kernel_size=1)
        
        # Storage for activations and gradients during probing
        self.activations = {}
        self.gradients = {}
        
        # Register hooks
        self._register_hooks()
    
    def _register_hooks(self):
        """Register forward and backward hooks on key layers."""
        
        def get_activation(name):
            def hook(module, input, output):
                self.activations[name] = output.detach()
            return hook
        
        def get_gradient(name):
            def hook(module, grad_input, grad_output):
                self.gradients[name] = grad_output[0].detach()
            return hook
        
        # Hook encoder
        self.inc.register_forward_hook(get_activation('enc_input'))
        self.inc.register_full_backward_hook(get_gradient('enc_input'))
        
        self.down1.register_forward_hook(get_activation('enc_down1'))
        self.down1.register_full_backward_hook(get_gradient('enc_down1'))
        
        self.down2.register_forward_hook(get_activation('enc_down2'))
        self.down2.register_full_backward_hook(get_gradient('enc_down2'))
        
        self.down3.register_forward_hook(get_activation('bottleneck'))
        self.down3.register_full_backward_hook(get_gradient('bottleneck'))
        
        # Hook decoder
        self.up1.register_forward_hook(get_activation('dec_up1'))
        self.up1.register_full_backward_hook(get_gradient('dec_up1'))
        
        self.up2.register_forward_hook(get_activation('dec_up2'))
        self.up2.register_full_backward_hook(get_gradient('dec_up2'))
        
        self.up3.register_forward_hook(get_activation('dec_up3'))
        self.up3.register_full_backward_hook(get_gradient('dec_up3'))

    def forward(self, input_grid: torch.Tensor, output_grid: torch.Tensor,
                skip_ablation: str = None) -> torch.Tensor:
        """
        Forward pass with optional skip connection ablation.
        
        skip_ablation: If set to 'skip1', 'skip2', or 'skip3', zeros that skip connection.
        """
        # Embed both grids
        inp_emb = self.input_embed(input_grid)
        out_emb = self.output_embed(output_grid)

        # Explicit comparison features
        diff = inp_emb - out_emb
        prod = inp_emb * out_emb
        x = torch.cat([inp_emb, out_emb, diff, prod], dim=-1)
        x = x.permute(0, 3, 1, 2).contiguous()

        # Encoder
        x1 = self.inc(x)      # (B, 64, 30, 30)
        x2 = self.down1(x1)   # (B, 128, 15, 15)
        x3 = self.down2(x2)   # (B, 256, 7, 7)
        x4 = self.down3(x3)   # (B, 512, 3, 3) - bottleneck
        
        # Apply skip ablation if requested
        if skip_ablation == 'skip3':
            x3 = torch.zeros_like(x3)
        if skip_ablation == 'skip2':
            x2 = torch.zeros_like(x2)
        if skip_ablation == 'skip1':
            x1 = torch.zeros_like(x1)

        # Decoder
        x = self.up1(x4, x3)  # (B, 256, 7, 7)
        x = self.up2(x, x2)   # (B, 128, 15, 15)
        x = self.up3(x, x1)   # (B, 64, 30, 30)

        logits = self.outc(x)
        return logits

    def predict_colors(self, input_grid: torch.Tensor, output_grid: torch.Tensor,
                       skip_ablation: str = None) -> torch.Tensor:
        logits = self.forward(input_grid, output_grid, skip_ablation=skip_ablation)
        return logits.argmax(dim=1)


# =============================================================================
# Dataset
# =============================================================================

def pad_grid(grid: np.ndarray, size: int = GRID_SIZE) -> np.ndarray:
    """Pad grid to fixed size."""
    h, w = grid.shape
    padded = np.zeros((size, size), dtype=np.uint8)
    padded[:h, :w] = grid
    return padded


class SinglePuzzleDataset(Dataset):
    """Dataset for a single puzzle with augmentation.
    
    CRITICAL: The model receives (input, candidate_output) and predicts the correct output.
    During training, we show it ZEROS as the candidate, so it learns to predict 
    the correct output from the input alone. This is the "color prediction" task.
    """
    
    def __init__(self, puzzle: Dict, include_test: bool = False, augment: bool = True):
        self.examples = []
        self.augment = augment
        
        for ex in puzzle.get("train", []):
            inp = np.array(ex["input"], dtype=np.uint8)
            out = np.array(ex["output"], dtype=np.uint8)
            self.examples.append((inp, out))
        
        if include_test:
            for ex in puzzle.get("test", []):
                if "output" in ex:
                    inp = np.array(ex["input"], dtype=np.uint8)
                    out = np.array(ex["output"], dtype=np.uint8)
                    self.examples.append((inp, out))
    
    def __len__(self):
        return len(self.examples) * (72 if self.augment else 1)
    
    def __getitem__(self, idx):
        base_idx = idx % len(self.examples)
        inp, out = self.examples[base_idx]
        
        if self.augment:
            # Random dihedral transform
            trans_id = random.randint(0, 7)
            inp = self._dihedral_transform(inp, trans_id)
            out = self._dihedral_transform(out, trans_id)
            
            # Random color permutation
            color_map = self._random_color_perm()
            inp = color_map[inp]
            out = color_map[out]
        
        inp_padded = pad_grid(inp)
        out_padded = pad_grid(out)
        
        # CRITICAL: Model sees ZEROS as candidate output during training.
        # It must learn to predict the correct output from the input alone.
        zeros_padded = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.uint8)
        
        return (
            torch.tensor(inp_padded, dtype=torch.long),
            torch.tensor(zeros_padded, dtype=torch.long),  # Model sees zeros
            torch.tensor(out_padded, dtype=torch.long)     # Target to predict
        )
    
    def _dihedral_transform(self, arr: np.ndarray, tid: int) -> np.ndarray:
        if tid == 0: return arr
        elif tid == 1: return np.rot90(arr, k=1)
        elif tid == 2: return np.rot90(arr, k=2)
        elif tid == 3: return np.rot90(arr, k=3)
        elif tid == 4: return np.fliplr(arr)
        elif tid == 5: return np.flipud(arr)
        elif tid == 6: return arr.T
        elif tid == 7: return np.fliplr(np.rot90(arr, k=1))
        return arr
    
    def _random_color_perm(self) -> np.ndarray:
        mapping = np.concatenate([
            np.array([0], dtype=np.uint8),
            np.random.permutation(np.arange(1, 10, dtype=np.uint8))
        ])
        return mapping


class TestDataset(Dataset):
    """Test dataset without augmentation.
    
    IMPORTANT: For evaluation, we pass ZEROS as the output_grid to the model.
    The model must predict the correct output without seeing it.
    This simulates the actual test condition where we don't know the answer.
    """
    
    def __init__(self, puzzle: Dict, test_only: bool = True):
        self.examples = []
        
        source = puzzle.get("test", []) if test_only else puzzle.get("train", [])
        for ex in source:
            if "output" in ex:
                inp = np.array(ex["input"], dtype=np.uint8)
                out = np.array(ex["output"], dtype=np.uint8)
                self.examples.append((inp, out))
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        inp, out = self.examples[idx]
        inp_padded = pad_grid(inp)
        out_padded = pad_grid(out)
        
        # CRITICAL: Pass zeros as output_grid, not the correct answer!
        # The model must predict the output without seeing it.
        zeros_padded = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.uint8)
        
        return (
            torch.tensor(inp_padded, dtype=torch.long),
            torch.tensor(zeros_padded, dtype=torch.long),  # Model sees zeros
            torch.tensor(out_padded, dtype=torch.long)     # Target for comparison
        )


# =============================================================================
# Training
# =============================================================================

def visualize_grid(grid: np.ndarray, title: str = ""):
    """Print a grid to console using numbers as colors."""
    print(f"\n{title}")
    h, w = grid.shape
    # Find actual content bounds (non-zero region)
    rows_with_content = np.any(grid != 0, axis=1)
    cols_with_content = np.any(grid != 0, axis=0)

    if rows_with_content.any() and cols_with_content.any():
        row_min, row_max = np.where(rows_with_content)[0][[0, -1]]
        col_min, col_max = np.where(cols_with_content)[0][[0, -1]]
        # Add 1 padding
        row_min = max(0, row_min - 1)
        row_max = min(h - 1, row_max + 1)
        col_min = max(0, col_min - 1)
        col_max = min(w - 1, col_max + 1)
    else:
        row_min, row_max = 0, min(10, h - 1)
        col_min, col_max = 0, min(10, w - 1)

    # Print grid
    for i in range(row_min, row_max + 1):
        row_str = ""
        for j in range(col_min, col_max + 1):
            val = grid[i, j]
            row_str += str(val)
        print(row_str)


def visualize_prediction(model, loader, device, num_examples: int = 1):
    """Visualize model predictions vs ground truth."""
    model.eval()

    with torch.no_grad():
        for inp, out, target in loader:
            inp, out, target = inp.to(device), out.to(device), target.to(device)
            pred = model.predict_colors(inp, out)

            for i in range(min(num_examples, pred.size(0))):
                inp_grid = inp[i].cpu().numpy()
                pred_grid = pred[i].cpu().numpy()
                target_grid = target[i].cpu().numpy()

                print("\n" + "=" * 40)
                visualize_grid(inp_grid, "INPUT:")
                visualize_grid(target_grid, "GROUND TRUTH:")
                visualize_grid(pred_grid, "PREDICTION:")

                # Show match status
                match = np.array_equal(pred_grid, target_grid)
                print(f"\nMatch: {'✓ CORRECT' if match else '✗ WRONG'}")
                print("=" * 40)

            break  # Only show first batch


def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0

    for inp, out, target in loader:
        inp, out, target = inp.to(device), out.to(device), target.to(device)

        optimizer.zero_grad()
        logits = model(inp, out)
        loss = F.cross_entropy(logits, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def get_content_mask(grid: torch.Tensor) -> torch.Tensor:
    """Get a mask for the content region (bounding box of non-zero pixels)."""
    # grid shape: (H, W)
    nonzero = grid != 0
    if not nonzero.any():
        # If all zeros, return full grid mask
        return torch.ones_like(grid, dtype=torch.bool)

    rows = nonzero.any(dim=1)
    cols = nonzero.any(dim=0)
    row_indices = torch.where(rows)[0]
    col_indices = torch.where(cols)[0]

    row_min, row_max = row_indices[0], row_indices[-1]
    col_min, col_max = col_indices[0], col_indices[-1]

    mask = torch.zeros_like(grid, dtype=torch.bool)
    mask[row_min:row_max + 1, col_min:col_max + 1] = True
    return mask


def evaluate(model, loader, device, skip_ablation: str = None):
    """Evaluate model, optionally with skip ablation."""
    model.eval()

    total_pixels = 0
    correct_pixels = 0
    perfect_examples = 0
    total_examples = 0

    with torch.no_grad():
        for inp, out, target in loader:
            inp, out, target = inp.to(device), out.to(device), target.to(device)

            pred = model.predict_colors(inp, out, skip_ablation=skip_ablation)

            # Per-example accuracy (only in content region)
            for i in range(pred.size(0)):
                total_examples += 1
                # Get content mask from target (ground truth defines the region)
                mask = get_content_mask(target[i])
                pred_content = pred[i][mask]
                target_content = target[i][mask]

                if torch.equal(pred_content, target_content):
                    perfect_examples += 1

                # Pixel accuracy only in content region
                correct_pixels += (pred_content == target_content).sum().item()
                total_pixels += target_content.numel()

    return {
        'pixel_acc': correct_pixels / total_pixels if total_pixels > 0 else 0,
        'perfect_rate': perfect_examples / total_examples if total_examples > 0 else 0,
        'perfect': perfect_examples,
        'total': total_examples
    }


# =============================================================================
# Probing Experiments
# =============================================================================

def probe_gradient_importance(model, loader, device):
    """Compute gradient magnitude at each layer to assess importance."""
    model.train()
    
    # Accumulate gradient magnitudes
    grad_magnitudes = {name: 0.0 for name in model.gradients.keys()}
    num_batches = 0
    
    for inp, out, target in loader:
        inp, out, target = inp.to(device), out.to(device), target.to(device)
        
        model.zero_grad()
        logits = model(inp, out)
        loss = F.cross_entropy(logits, target)
        loss.backward()
        
        # Record gradient magnitudes
        for name, grad in model.gradients.items():
            grad_magnitudes[name] += grad.abs().mean().item()
        
        num_batches += 1
        if num_batches >= 10:  # Sample 10 batches
            break
    
    # Average
    for name in grad_magnitudes:
        grad_magnitudes[name] /= num_batches
    
    return grad_magnitudes


def probe_activation_statistics(model, loader, device):
    """Compute activation statistics at each layer."""
    model.eval()
    
    stats = {}
    
    with torch.no_grad():
        for inp, out, target in loader:
            inp, out, target = inp.to(device), out.to(device), target.to(device)
            
            _ = model(inp, out)
            
            # Record stats for each activation
            for name, act in model.activations.items():
                if name not in stats:
                    stats[name] = {
                        'mean': [],
                        'std': [],
                        'sparsity': [],  # Fraction of zeros/near-zeros
                        'shape': act.shape
                    }
                
                stats[name]['mean'].append(act.mean().item())
                stats[name]['std'].append(act.std().item())
                stats[name]['sparsity'].append((act.abs() < 0.01).float().mean().item())
            
            break  # Just one batch for stats
    
    # Summarize
    for name in stats:
        stats[name]['mean'] = np.mean(stats[name]['mean'])
        stats[name]['std'] = np.mean(stats[name]['std'])
        stats[name]['sparsity'] = np.mean(stats[name]['sparsity'])
    
    return stats


def probe_weight_pruning(model, test_loader, device, prune_fractions=[0.1, 0.2, 0.3, 0.5, 0.7, 0.9]):
    """Test how performance degrades as weights are pruned."""
    results = []
    
    # Get baseline
    baseline = evaluate(model, test_loader, device)
    results.append({'prune_fraction': 0.0, **baseline})
    
    for frac in prune_fractions:
        # Create a copy of the model
        pruned_model = copy.deepcopy(model)
        
        # Prune smallest weights globally
        all_weights = []
        for name, param in pruned_model.named_parameters():
            if 'weight' in name and param.dim() > 1:
                all_weights.append(param.data.abs().flatten())
        
        all_weights = torch.cat(all_weights)
        threshold = torch.quantile(all_weights, frac)
        
        # Apply pruning
        with torch.no_grad():
            for name, param in pruned_model.named_parameters():
                if 'weight' in name and param.dim() > 1:
                    mask = param.data.abs() >= threshold
                    param.data *= mask.float()
        
        # Evaluate
        metrics = evaluate(pruned_model, test_loader, device)
        results.append({'prune_fraction': frac, **metrics})
        
        del pruned_model
    
    return results


def probe_skip_ablation(model, test_loader, device):
    """Test performance when each skip connection is zeroed."""
    results = {}
    
    # Baseline (no ablation)
    results['baseline'] = evaluate(model, test_loader, device)
    
    # Ablate each skip
    for skip in ['skip1', 'skip2', 'skip3']:
        results[skip] = evaluate(model, test_loader, device, skip_ablation=skip)
    
    return results


def probe_layer_wise_freezing(model, train_loader, test_loader, device, epochs=10):
    """Freeze all but one layer group, retrain, see which matters most."""
    
    layer_groups = {
        'encoder_input': ['inc'],
        'encoder_down1': ['down1'],
        'encoder_down2': ['down2'],
        'bottleneck': ['down3'],
        'decoder_up1': ['up1'],
        'decoder_up2': ['up2'],
        'decoder_up3': ['up3'],
        'output': ['outc']
    }
    
    results = {}
    
    for group_name, layer_names in layer_groups.items():
        # Create fresh model
        probe_model = ProbeableUNet(hidden_dim=64, num_classes=10).to(device)
        
        # Freeze everything except this group
        for name, param in probe_model.named_parameters():
            should_train = any(ln in name for ln in layer_names)
            param.requires_grad = should_train
        
        # Count trainable params
        trainable = sum(p.numel() for p in probe_model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in probe_model.parameters())
        
        # Train briefly
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, probe_model.parameters()),
            lr=1e-3
        )
        
        for epoch in range(epochs):
            train_epoch(probe_model, train_loader, optimizer, device)
        
        # Evaluate
        metrics = evaluate(probe_model, test_loader, device)
        results[group_name] = {
            'trainable_params': trainable,
            'total_params': total,
            'trainable_fraction': trainable / total,
            **metrics
        }
        
        del probe_model
    
    return results


# =============================================================================
# Main
# =============================================================================

def load_puzzle(puzzle_id: str, data_root: str = "kaggle/combined") -> Dict:
    """Load a single puzzle."""
    for subset in ["training", "evaluation"]:
        challenges_path = f"{data_root}/arc-agi_training_challenges.json" if subset == "training" else f"{data_root}/arc-agi_evaluation_challenges.json"
        solutions_path = challenges_path.replace("challenges", "solutions")
        
        if not os.path.exists(challenges_path):
            continue
        
        with open(challenges_path) as f:
            puzzles = json.load(f)
        
        if puzzle_id in puzzles:
            puzzle = puzzles[puzzle_id]
            
            # Load solutions if available
            if os.path.exists(solutions_path):
                with open(solutions_path) as f:
                    solutions = json.load(f)
                if puzzle_id in solutions:
                    for i, sol in enumerate(solutions[puzzle_id]):
                        if i < len(puzzle.get("test", [])):
                            puzzle["test"][i]["output"] = sol
            
            return puzzle
    
    raise ValueError(f"Puzzle {puzzle_id} not found")


def main():
    parser = argparse.ArgumentParser(description="Probe CNN Pattern Storage")
    parser.add_argument("--puzzle", type=str, default="3c9b0459", help="Puzzle ID")
    parser.add_argument("--data-root", type=str, default="kaggle/combined")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    print(f"Device: {DEVICE}")
    print(f"Puzzle: {args.puzzle}")
    print("="*60)
    
    # Load puzzle
    puzzle = load_puzzle(args.puzzle, args.data_root)
    n_train = len(puzzle.get("train", []))
    n_test = len([ex for ex in puzzle.get("test", []) if "output" in ex])
    print(f"Train examples: {n_train}, Test examples: {n_test}")
    
    # Create datasets
    train_dataset = SinglePuzzleDataset(puzzle, include_test=False, augment=True)
    test_dataset = TestDataset(puzzle, test_only=True)
    train_eval_dataset = TestDataset(puzzle, test_only=False)  # For evaluating on train examples
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    train_eval_loader = DataLoader(train_eval_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Create model
    model = ProbeableUNet(hidden_dim=64, num_classes=10).to(DEVICE)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    
    # ==========================================================================
    # Phase 1: Train the model
    # ==========================================================================
    print("\n" + "="*60)
    print("PHASE 1: Training")
    print("="*60)
    
    training_history = []
    
    for epoch in range(args.epochs):
        train_loss = train_epoch(model, train_loader, optimizer, DEVICE)
        scheduler.step()
        
        # Evaluate every 10 epochs
        if (epoch + 1) % 10 == 0 or epoch == 0:
            train_metrics = evaluate(model, train_eval_loader, DEVICE)
            test_metrics = evaluate(model, test_loader, DEVICE)
            
            print(f"Epoch {epoch+1:3d} | Loss: {train_loss:.4f} | "
                  f"Train: {train_metrics['pixel_acc']:.2%} ({train_metrics['perfect']}/{train_metrics['total']} perfect) | "
                  f"Test: {test_metrics['pixel_acc']:.2%} ({test_metrics['perfect']}/{test_metrics['total']} perfect)")

            # # Visualize a prediction
            # visualize_prediction(model, test_loader, DEVICE, num_examples=1)
            
            training_history.append({
                'epoch': epoch + 1,
                'loss': train_loss,
                'train_pixel_acc': train_metrics['pixel_acc'],
                'train_perfect_rate': train_metrics['perfect_rate'],
                'test_pixel_acc': test_metrics['pixel_acc'],
                'test_perfect_rate': test_metrics['perfect_rate']
            })
    
    # Final evaluation
    print("\n" + "-"*60)
    final_train = evaluate(model, train_eval_loader, DEVICE)
    final_test = evaluate(model, test_loader, DEVICE)
    print(f"FINAL - Train: {final_train['pixel_acc']:.2%} ({final_train['perfect']}/{final_train['total']} perfect)")
    print(f"FINAL - Test:  {final_test['pixel_acc']:.2%} ({final_test['perfect']}/{final_test['total']} perfect)")
    
    if final_test['perfect_rate'] < 1.0:
        print("\n⚠️  Model did not achieve perfect accuracy. Probing results may not reflect fully learned pattern.")
    
    # ==========================================================================
    # Phase 2: Probing Experiments
    # ==========================================================================
    print("\n" + "="*60)
    print("PHASE 2: Probing Experiments")
    print("="*60)
    
    # --------------------------------------------------------------------------
    # Experiment 2.1: Gradient Importance
    # --------------------------------------------------------------------------
    print("\n--- 2.1 Gradient Importance (higher = more important) ---")
    grad_importance = probe_gradient_importance(model, train_loader, DEVICE)
    
    # Sort by importance
    sorted_grads = sorted(grad_importance.items(), key=lambda x: x[1], reverse=True)
    for name, magnitude in sorted_grads:
        bar = "█" * int(magnitude * 100 / max(grad_importance.values()))
        print(f"  {name:20s}: {magnitude:.6f} {bar}")
    
    # --------------------------------------------------------------------------
    # Experiment 2.2: Activation Statistics
    # --------------------------------------------------------------------------
    print("\n--- 2.2 Activation Statistics ---")
    act_stats = probe_activation_statistics(model, test_loader, DEVICE)
    
    print(f"  {'Layer':<20s} {'Shape':<20s} {'Mean':>10s} {'Std':>10s} {'Sparsity':>10s}")
    print(f"  {'-'*70}")
    for name, stats in act_stats.items():
        shape_str = str(tuple(stats['shape']))
        print(f"  {name:<20s} {shape_str:<20s} {stats['mean']:>10.4f} {stats['std']:>10.4f} {stats['sparsity']:>10.2%}")
    
    # --------------------------------------------------------------------------
    # Experiment 2.3: Weight Pruning
    # --------------------------------------------------------------------------
    print("\n--- 2.3 Weight Pruning (does performance collapse suddenly?) ---")
    pruning_results = probe_weight_pruning(model, test_loader, DEVICE)
    
    print(f"  {'Pruned %':<10s} {'Pixel Acc':>12s} {'Perfect':>10s}")
    print(f"  {'-'*35}")
    for r in pruning_results:
        print(f"  {r['prune_fraction']:>8.0%}   {r['pixel_acc']:>10.2%}   {r['perfect']}/{r['total']}")
    
    # --------------------------------------------------------------------------
    # Experiment 2.4: Skip Connection Ablation
    # --------------------------------------------------------------------------
    print("\n--- 2.4 Skip Connection Ablation ---")
    skip_results = probe_skip_ablation(model, test_loader, DEVICE)
    
    print(f"  {'Condition':<15s} {'Pixel Acc':>12s} {'Perfect':>10s} {'Drop':>10s}")
    print(f"  {'-'*50}")
    baseline_acc = skip_results['baseline']['pixel_acc']
    for condition, metrics in skip_results.items():
        drop = baseline_acc - metrics['pixel_acc']
        print(f"  {condition:<15s} {metrics['pixel_acc']:>10.2%}   {metrics['perfect']}/{metrics['total']}      {drop:>+.2%}")
    
    # --------------------------------------------------------------------------
    # Experiment 2.5: Layer-wise Freezing (which layer learns the pattern?)
    # --------------------------------------------------------------------------
    print("\n--- 2.5 Layer-wise Freezing (train only one layer group from scratch) ---")
    print("  This shows which layers can learn the pattern independently.")
    print("  Training each configuration for 10 epochs...")
    
    freeze_results = probe_layer_wise_freezing(model, train_loader, test_loader, DEVICE, epochs=10)
    
    print(f"\n  {'Layer Group':<20s} {'Trainable':>12s} {'Pixel Acc':>12s} {'Perfect':>10s}")
    print(f"  {'-'*60}")
    for group, metrics in freeze_results.items():
        trainable_pct = metrics['trainable_fraction'] * 100
        print(f"  {group:<20s} {trainable_pct:>10.1f}%   {metrics['pixel_acc']:>10.2%}   {metrics['perfect']}/{metrics['total']}")
    
    # ==========================================================================
    # Summary
    # ==========================================================================
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    print(f"""
Model trained for {args.epochs} epochs on puzzle {args.puzzle}

Final Performance:
  - Train: {final_train['pixel_acc']:.2%} pixel accuracy, {final_train['perfect_rate']:.0%} perfect
  - Test:  {final_test['pixel_acc']:.2%} pixel accuracy, {final_test['perfect_rate']:.0%} perfect

Key Findings:

1. GRADIENT IMPORTANCE:
   Highest gradient layer: {sorted_grads[0][0]} ({sorted_grads[0][1]:.6f})
   Lowest gradient layer:  {sorted_grads[-1][0]} ({sorted_grads[-1][1]:.6f})

2. PRUNING ROBUSTNESS:
   At 50% pruned: {pruning_results[3]['pixel_acc']:.2%} accuracy
   At 90% pruned: {pruning_results[5]['pixel_acc']:.2%} accuracy
   Pattern is {'ROBUST (dispersed)' if pruning_results[5]['pixel_acc'] > 0.5 else 'FRAGILE (localized)'}

3. SKIP CONNECTIONS:
   Most critical skip: {max([(k, skip_results['baseline']['pixel_acc'] - v['pixel_acc']) for k, v in skip_results.items() if k != 'baseline'], key=lambda x: x[1])[0]}
   
4. LAYER INDEPENDENCE:
   Best single-layer-group learning: {max(freeze_results.items(), key=lambda x: x[1]['pixel_acc'])[0]}
""")

    # Save results
    results = {
        'puzzle': args.puzzle,
        'epochs': args.epochs,
        'training_history': training_history,
        'final_train': final_train,
        'final_test': final_test,
        'gradient_importance': grad_importance,
        'activation_stats': {k: {kk: vv for kk, vv in v.items() if kk != 'shape'} for k, v in act_stats.items()},
        'pruning_results': pruning_results,
        'skip_ablation': skip_results,
        'layer_freezing': freeze_results
    }
    
    output_path = f"probe_results_{args.puzzle}.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()