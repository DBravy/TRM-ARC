#!/usr/bin/env python3
"""
CNN Counting Experiment

Tests how well a U-Net encoder can count occurrences of a specific color
as grid size increases. This isolates the "counting problem" from the
complexity of full ARC puzzles.

Hypothesis: Counting accuracy will degrade as grid size exceeds the
receptive field of the early layers, because the deeper layers (which
have global RF) compress information too aggressively.
"""

import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import defaultdict

# Device setup
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

NUM_COLORS = 10
TARGET_COLOR = 1  # "Blue" - the color we're counting


# =============================================================================
# Model Architecture (from your code)
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
    """Downsampling block that skips pooling if spatial size is too small."""
    def __init__(self, in_ch: int, out_ch: int, min_size: int = 2):
        super().__init__()
        self.min_size = min_size
        self.pool = nn.MaxPool2d(2)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x):
        # Only pool if we won't go below min_size
        if x.size(2) >= self.min_size * 2 and x.size(3) >= self.min_size * 2:
            x = self.pool(x)
        return self.conv(x)


class Up(nn.Module):
    """Upsampling block with skip connection (matches PixelErrorCNN).

    Handles cases where spatial sizes already match (small grids where
    downsampling was skipped).
    """
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.in_ch = in_ch
        self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
        # 1x1 conv for channel reduction when spatial upsample not needed
        self.channel_reduce = nn.Conv2d(in_ch, in_ch // 2, kernel_size=1)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x1, x2):
        # x1: deeper features (more channels, possibly smaller spatial)
        # x2: skip connection (fewer channels, possibly larger spatial)

        need_spatial_upsample = x1.size(2) < x2.size(2) or x1.size(3) < x2.size(3)

        if need_spatial_upsample:
            # Normal case: use transposed conv to upsample and reduce channels
            x1 = self.up(x1)
            # Handle any remaining size mismatches due to odd dimensions
            diff_h = x2.size(2) - x1.size(2)
            diff_w = x2.size(3) - x1.size(3)
            if diff_h != 0 or diff_w != 0:
                x1 = F.pad(x1, [diff_w // 2, diff_w - diff_w // 2,
                               diff_h // 2, diff_h - diff_h // 2])
        else:
            # Small grid case: spatial sizes match, just reduce channels
            x1 = self.channel_reduce(x1)
            # Ensure spatial sizes match exactly
            if x1.size(2) != x2.size(2) or x1.size(3) != x2.size(3):
                x1 = F.interpolate(x1, size=(x2.size(2), x2.size(3)),
                                  mode='bilinear', align_corners=False)

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class CountingUNet(nn.Module):
    """
    Full U-Net with skip connections for per-pixel target detection.

    Architecture matches PixelErrorCNN: encoder-decoder with skip connections.
    Output: per-pixel binary prediction (is this pixel the target color?).
    We evaluate by counting predicted target pixels vs actual count.

    This tests whether the U-Net can accurately identify all instances of
    the target color at different grid sizes.
    """

    def __init__(self, hidden_dim: int = 64):
        super().__init__()

        base_ch = hidden_dim

        # Color embedding
        self.embed = nn.Embedding(NUM_COLORS, 16)

        # Encoder (same as PixelErrorCNN)
        self.inc = DoubleConv(16, base_ch)
        self.down1 = Down(base_ch, base_ch * 2)
        self.down2 = Down(base_ch * 2, base_ch * 4)
        self.down3 = Down(base_ch * 4, base_ch * 8)

        # Decoder with skip connections (same as PixelErrorCNN)
        self.up1 = Up(base_ch * 8, base_ch * 4)
        self.up2 = Up(base_ch * 4, base_ch * 2)
        self.up3 = Up(base_ch * 2, base_ch)

        # Output: 1 channel for binary target detection
        self.outc = nn.Conv2d(base_ch, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Embed colors
        x = self.embed(x)  # (B, H, W, 16)
        x = x.permute(0, 3, 1, 2).contiguous()  # (B, 16, H, W)

        # Store original size for decoder
        orig_h, orig_w = x.size(2), x.size(3)

        # Encoder forward
        x1 = self.inc(x)      # (B, base_ch, H, W)
        x2 = self.down1(x1)   # (B, base_ch*2, H/2, W/2) or same if too small
        x3 = self.down2(x2)   # (B, base_ch*4, H/4, W/4) or same if too small
        x4 = self.down3(x3)   # (B, base_ch*8, H/8, W/8) or same if too small

        # Decoder with skip connections
        # Use the Up modules' forward method which handles upsample + concat + conv
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)

        # Ensure output matches input size
        if x.size(2) != orig_h or x.size(3) != orig_w:
            x = F.interpolate(x, size=(orig_h, orig_w), mode='bilinear', align_corners=False)

        return self.outc(x)  # (B, 1, H, W)


class CountingCNN(nn.Module):
    """
    U-Net encoder + classification head for counting (direct count prediction).

    Alternative to CountingUNet that predicts count directly instead of
    per-pixel segmentation. Useful for comparison.
    """

    def __init__(self, hidden_dim: int = 64, num_classes: int = 10,
                 use_all_layers: bool = True):
        super().__init__()

        self.use_all_layers = use_all_layers
        base_ch = hidden_dim

        # Color embedding
        self.embed = nn.Embedding(NUM_COLORS, 16)

        # Encoder (same as U-Net)
        self.inc = DoubleConv(16, base_ch)
        self.down1 = Down(base_ch, base_ch * 2)
        self.down2 = Down(base_ch * 2, base_ch * 4)
        self.down3 = Down(base_ch * 4, base_ch * 8)

        # Global pooling + classifier
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        if use_all_layers:
            # Aggregate features from all levels
            total_channels = base_ch + base_ch * 2 + base_ch * 4 + base_ch * 8
        else:
            # Only use bottleneck features
            total_channels = base_ch * 8

        self.classifier = nn.Sequential(
            nn.Linear(total_channels, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Embed colors
        x = self.embed(x)  # (B, H, W, 16)
        x = x.permute(0, 3, 1, 2).contiguous()  # (B, 16, H, W)

        # Encoder forward
        x1 = self.inc(x)      # (B, 64, H, W)
        x2 = self.down1(x1)   # (B, 128, H/2, W/2)
        x3 = self.down2(x2)   # (B, 256, H/4, W/4)
        x4 = self.down3(x3)   # (B, 512, H/8, W/8)

        if self.use_all_layers:
            # Pool each level and concatenate
            p1 = self.global_pool(x1).flatten(1)  # (B, 64)
            p2 = self.global_pool(x2).flatten(1)  # (B, 128)
            p3 = self.global_pool(x3).flatten(1)  # (B, 256)
            p4 = self.global_pool(x4).flatten(1)  # (B, 512)
            pooled = torch.cat([p1, p2, p3, p4], dim=1)  # (B, 960)
        else:
            # Only use bottleneck
            pooled = self.global_pool(x4).flatten(1)  # (B, 512)

        return self.classifier(pooled)


class ShallowCountingCNN(nn.Module):
    """
    Shallow CNN that only uses the first conv block.
    Tests if counting can happen with just local features + global pooling.
    """
    
    def __init__(self, hidden_dim: int = 64, num_classes: int = 10):
        super().__init__()
        
        self.embed = nn.Embedding(NUM_COLORS, 16)
        self.inc = DoubleConv(16, hidden_dim)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embed(x)
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.inc(x)
        pooled = self.global_pool(x).flatten(1)
        return self.classifier(pooled)


# =============================================================================
# Dataset
# =============================================================================

class CountingDataset(Dataset):
    """
    Dataset of random grids where we count occurrences of TARGET_COLOR.

    Each grid has a random number of TARGET_COLOR pixels (within count_range),
    with remaining pixels filled with other random colors.

    Returns:
    - grid: The input grid (H, W) with color indices
    - mask: Binary mask (H, W) where 1 = target color pixel
    - count: The count class (0-indexed)
    """

    def __init__(self, grid_size: int, num_samples: int,
                 count_range: tuple = (1, 10), seed: int = None):
        self.grid_size = grid_size
        self.num_samples = num_samples
        self.count_range = count_range  # (min_count, max_count) exclusive of max

        if seed is not None:
            np.random.seed(seed)

        self.grids = []
        self.masks = []
        self.counts = []

        for _ in range(num_samples):
            # Random count of target color
            count = np.random.randint(count_range[0], count_range[1])

            # Create grid with random non-target colors
            other_colors = [c for c in range(NUM_COLORS) if c != TARGET_COLOR]
            grid = np.random.choice(other_colors, size=(grid_size, grid_size))

            # Place exactly 'count' target color pixels at random positions
            total_pixels = grid_size * grid_size
            if count > total_pixels:
                count = total_pixels

            positions = np.random.choice(total_pixels, size=count, replace=False)
            for pos in positions:
                r, c = pos // grid_size, pos % grid_size
                grid[r, c] = TARGET_COLOR

            # Create binary mask
            mask = (grid == TARGET_COLOR).astype(np.float32)

            self.grids.append(grid.astype(np.int64))
            self.masks.append(mask)
            self.counts.append(count - count_range[0])  # 0-indexed class

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return (torch.tensor(self.grids[idx]),
                torch.tensor(self.masks[idx]),
                torch.tensor(self.counts[idx]))


# =============================================================================
# Training and Evaluation
# =============================================================================

def train_epoch_unet(model, loader, optimizer, device):
    """Train U-Net model for per-pixel target detection."""
    model.train()
    total_loss = 0
    total_pixel_acc = 0
    total_count_acc = 0
    total = 0

    for grids, masks, counts in loader:
        grids = grids.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()
        logits = model(grids)  # (B, 1, H, W)

        # Binary cross-entropy loss for per-pixel prediction
        loss = F.binary_cross_entropy_with_logits(logits.squeeze(1), masks)
        loss.backward()
        optimizer.step()

        # Compute metrics
        preds = (logits.squeeze(1) > 0).float()  # Threshold at 0 (logit space)
        pixel_acc = (preds == masks).float().mean().item()

        # Count accuracy: predicted count vs actual count
        pred_counts = preds.sum(dim=(1, 2)).round().int()
        actual_counts = masks.sum(dim=(1, 2)).int()
        count_acc = (pred_counts == actual_counts).float().mean().item()

        total_loss += loss.item() * grids.size(0)
        total_pixel_acc += pixel_acc * grids.size(0)
        total_count_acc += count_acc * grids.size(0)
        total += grids.size(0)

    return total_loss / total, total_pixel_acc / total, total_count_acc / total


def train_epoch_classifier(model, loader, optimizer, device):
    """Train classifier model for direct count prediction."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for grids, masks, counts in loader:
        grids, counts = grids.to(device), counts.to(device)

        optimizer.zero_grad()
        logits = model(grids)
        loss = F.cross_entropy(logits, counts)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * grids.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == counts).sum().item()
        total += grids.size(0)

    return total_loss / total, correct / total


def evaluate_unet(model, loader, device, count_range):
    """Evaluate U-Net model by comparing predicted vs actual counts."""
    model.eval()
    total_pixel_acc = 0
    total_count_acc = 0
    total = 0

    # Track accuracy by count value
    correct_by_count = defaultdict(int)
    total_by_count = defaultdict(int)

    # Track count errors
    count_errors = []

    with torch.no_grad():
        for grids, masks, counts in loader:
            grids = grids.to(device)
            masks = masks.to(device)

            logits = model(grids)
            preds = (logits.squeeze(1) > 0).float()

            # Pixel accuracy
            pixel_acc = (preds == masks).float().mean().item()
            total_pixel_acc += pixel_acc * grids.size(0)

            # Count from predictions
            pred_counts = preds.sum(dim=(1, 2)).round().int()
            actual_counts = masks.sum(dim=(1, 2)).int()

            for pc, ac, c in zip(pred_counts.cpu().numpy(),
                                 actual_counts.cpu().numpy(),
                                 counts.cpu().numpy()):
                total_by_count[c] += 1
                if pc == ac:
                    correct_by_count[c] += 1
                    total_count_acc += 1
                count_errors.append(abs(pc - ac))
                total += 1

    acc_by_count = {k: correct_by_count[k] / total_by_count[k]
                    for k in sorted(total_by_count.keys())}

    return (total_count_acc / total, total_pixel_acc / (total / grids.size(0)),
            acc_by_count, np.mean(count_errors))


def evaluate_classifier(model, loader, device):
    """Evaluate classifier model."""
    model.eval()
    correct = 0
    total = 0

    # Track accuracy by count value
    correct_by_count = defaultdict(int)
    total_by_count = defaultdict(int)

    with torch.no_grad():
        for grids, masks, counts in loader:
            grids, counts = grids.to(device), counts.to(device)
            logits = model(grids)
            preds = logits.argmax(dim=1)

            correct += (preds == counts).sum().item()
            total += grids.size(0)

            for p, t in zip(preds.cpu().numpy(), counts.cpu().numpy()):
                total_by_count[t] += 1
                if p == t:
                    correct_by_count[t] += 1

    acc_by_count = {k: correct_by_count[k] / total_by_count[k]
                    for k in sorted(total_by_count.keys())}

    return correct / total, acc_by_count


def run_single_experiment(grid_size: int, count_range: tuple,
                          hidden_dim: int = 64, epochs: int = 50,
                          model_type: str = "unet", verbose: bool = True):
    """
    Train and evaluate a counting model on a specific grid size.

    Model types:
    - unet: Full U-Net with skip connections (matches PixelErrorCNN architecture)
    - encoder: Encoder + global pooling classifier
    - shallow: Single conv block + classifier

    Returns: (train_acc, test_acc, best_acc, acc_by_count, model)
    """
    num_classes = count_range[1] - count_range[0]

    # Create datasets
    train_dataset = CountingDataset(grid_size, num_samples=5000,
                                    count_range=count_range, seed=42)
    test_dataset = CountingDataset(grid_size, num_samples=1000,
                                   count_range=count_range, seed=123)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64)

    # Create model
    is_unet = model_type == "unet"

    if model_type == "unet":
        model = CountingUNet(hidden_dim=hidden_dim)
    elif model_type == "encoder":
        model = CountingCNN(hidden_dim=hidden_dim, num_classes=num_classes,
                           use_all_layers=True)
    elif model_type == "shallow":
        model = ShallowCountingCNN(hidden_dim=hidden_dim, num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
        
    model = model.to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

    # Training loop
    best_test_acc = 0

    iterator = range(epochs)
    if verbose:
        iterator = tqdm(iterator, desc=f"Grid {grid_size}x{grid_size}")

    for epoch in iterator:
        if is_unet:
            train_loss, pixel_acc, train_acc = train_epoch_unet(
                model, train_loader, optimizer, DEVICE)
            test_acc, _, _, _ = evaluate_unet(model, test_loader, DEVICE, count_range)
        else:
            train_loss, train_acc = train_epoch_classifier(
                model, train_loader, optimizer, DEVICE)
            test_acc, _ = evaluate_classifier(model, test_loader, DEVICE)

        scheduler.step()

        best_test_acc = max(best_test_acc, test_acc)

        if verbose and (epoch + 1) % 10 == 0:
            tqdm.write(f"  Epoch {epoch+1}: Train {train_acc:.1%}, Test {test_acc:.1%}")

    # Final evaluation
    if is_unet:
        final_train_acc, _, _, _ = evaluate_unet(model, train_loader, DEVICE, count_range)
        final_test_acc, pixel_acc, acc_by_count, mean_error = evaluate_unet(
            model, test_loader, DEVICE, count_range)
    else:
        final_train_acc, _ = evaluate_classifier(model, train_loader, DEVICE)
        final_test_acc, acc_by_count = evaluate_classifier(model, test_loader, DEVICE)

    return final_train_acc, final_test_acc, best_test_acc, acc_by_count, model


def compute_receptive_field(grid_size: int) -> dict:
    """
    Compute the receptive field at each layer of the U-Net encoder.
    
    For a 3x3 conv with padding=1, RF grows by 2 per layer.
    For MaxPool2d(2), the RF is effectively doubled.
    """
    # DoubleConv = 2 conv layers, each adds 2 to RF
    # So DoubleConv adds 4 to RF (starting from 1)
    # After inc: RF = 1 + 4 = 5
    
    # Down = MaxPool2d(2) + DoubleConv
    # MaxPool doubles effective RF, then DoubleConv adds 4
    
    rf = 1  # Initial RF
    
    results = {}
    
    # inc: DoubleConv
    rf = rf + 4  # 5
    results['inc'] = {
        'rf': rf,
        'spatial': grid_size,
        'coverage': min(1.0, (rf / grid_size) ** 2)
    }
    
    # down1: MaxPool + DoubleConv
    rf = rf * 2 + 4  # 14
    spatial = grid_size // 2
    results['down1'] = {
        'rf': rf,
        'spatial': spatial,
        'coverage': min(1.0, (rf / grid_size) ** 2)
    }
    
    # down2: MaxPool + DoubleConv
    rf = rf * 2 + 4  # 32
    spatial = spatial // 2
    results['down2'] = {
        'rf': rf,
        'spatial': spatial,
        'coverage': min(1.0, (rf / grid_size) ** 2)
    }
    
    # down3: MaxPool + DoubleConv
    rf = rf * 2 + 4  # 68
    spatial = spatial // 2
    results['down3'] = {
        'rf': rf,
        'spatial': spatial,
        'coverage': min(1.0, (rf / grid_size) ** 2)
    }
    
    return results


def main():
    parser = argparse.ArgumentParser(description="CNN Counting Experiment")
    parser.add_argument("--grid-sizes", type=int, nargs="+",
                        default=[3, 5, 7, 10, 15, 20, 25, 30],
                        help="Grid sizes to test")
    parser.add_argument("--count-range", type=int, nargs=2, default=[1, 10],
                        help="Range of counts (min, max)")
    parser.add_argument("--hidden-dim", type=int, default=64,
                        help="Hidden dimension for CNN")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Training epochs per grid size")
    parser.add_argument("--model-type", type=str, default="unet",
                        choices=["unet", "encoder", "shallow"],
                        help="Model architecture: unet (full U-Net with skip connections), "
                             "encoder (encoder + classifier), shallow (single conv block)")
    parser.add_argument("--compare-models", action="store_true",
                        help="Compare all model types")
    args = parser.parse_args()
    
    count_range = tuple(args.count_range)
    num_classes = count_range[1] - count_range[0]
    
    print("=" * 70)
    print("CNN COUNTING EXPERIMENT")
    print("=" * 70)
    print(f"Target color: {TARGET_COLOR} (blue)")
    print(f"Count range: {count_range[0]} to {count_range[1]-1} ({num_classes} classes)")
    print(f"Grid sizes: {args.grid_sizes}")
    print(f"Device: {DEVICE}")
    print()
    
    # Print receptive field analysis
    print("RECEPTIVE FIELD ANALYSIS")
    print("-" * 70)
    print(f"{'Grid':<8} {'inc (RF)':<12} {'down1 (RF)':<12} {'down2 (RF)':<12} {'down3 (RF)':<12}")
    print("-" * 70)
    
    for gs in args.grid_sizes:
        rf_info = compute_receptive_field(gs)
        print(f"{gs}x{gs:<5} "
              f"{rf_info['inc']['rf']:>2} ({rf_info['inc']['coverage']:>5.1%})   "
              f"{rf_info['down1']['rf']:>2} ({rf_info['down1']['coverage']:>5.1%})   "
              f"{rf_info['down2']['rf']:>2} ({rf_info['down2']['coverage']:>5.1%})   "
              f"{rf_info['down3']['rf']:>2} ({rf_info['down3']['coverage']:>5.1%})")
    print()
    
    if args.compare_models:
        # Compare all model types
        model_types = ["unet", "encoder", "shallow"]
        results = {mt: {} for mt in model_types}
        
        for model_type in model_types:
            print(f"\n{'=' * 70}")
            print(f"MODEL TYPE: {model_type.upper()}")
            print("=" * 70)
            
            for grid_size in args.grid_sizes:
                print(f"\nTraining on {grid_size}x{grid_size} grid...")
                train_acc, test_acc, best_acc, acc_by_count, _ = run_single_experiment(
                    grid_size, count_range, args.hidden_dim, args.epochs,
                    model_type=model_type, verbose=True
                )
                results[model_type][grid_size] = {
                    'train': train_acc,
                    'test': test_acc,
                    'best': best_acc,
                    'by_count': acc_by_count
                }
                print(f"  Final: Train {train_acc:.1%}, Test {test_acc:.1%}, Best {best_acc:.1%}")
        
        # Summary comparison
        print("\n" + "=" * 70)
        print("COMPARISON SUMMARY")
        print("=" * 70)
        print(f"{'Grid':<8}", end="")
        for mt in model_types:
            print(f"{mt:<20}", end="")
        print()
        print("-" * 70)
        
        for gs in args.grid_sizes:
            print(f"{gs}x{gs:<5}", end="")
            for mt in model_types:
                acc = results[mt][gs]['test']
                print(f"{acc:>6.1%}              ", end="")
            print()
            
    else:
        # Single model type experiment
        results = {}
        
        print(f"MODEL TYPE: {args.model_type.upper()}")
        print("-" * 70)
        
        for grid_size in args.grid_sizes:
            print(f"\nTraining on {grid_size}x{grid_size} grid...")
            train_acc, test_acc, best_acc, acc_by_count, model = run_single_experiment(
                grid_size, count_range, args.hidden_dim, args.epochs,
                model_type=args.model_type, verbose=True
            )
            
            results[grid_size] = {
                'train': train_acc,
                'test': test_acc,
                'best': best_acc,
                'by_count': acc_by_count
            }
            
            print(f"\n  Results for {grid_size}x{grid_size}:")
            print(f"    Train accuracy: {train_acc:.1%}")
            print(f"    Test accuracy:  {test_acc:.1%}")
            print(f"    Best accuracy:  {best_acc:.1%}")
            print(f"    Accuracy by count:")
            for count, acc in acc_by_count.items():
                actual_count = count + count_range[0]
                print(f"      Count {actual_count}: {acc:.1%}")
        
        # Final summary
        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)
        print(f"{'Grid Size':<12} {'Train Acc':<12} {'Test Acc':<12} {'Best Acc':<12}")
        print("-" * 70)
        
        for gs in args.grid_sizes:
            r = results[gs]
            print(f"{gs}x{gs:<9} {r['train']:<12.1%} {r['test']:<12.1%} {r['best']:<12.1%}")
        
        # Create visualization
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Accuracy vs grid size
        ax1 = axes[0]
        grid_sizes = list(results.keys())
        test_accs = [results[gs]['test'] for gs in grid_sizes]
        train_accs = [results[gs]['train'] for gs in grid_sizes]
        
        ax1.plot(grid_sizes, train_accs, 'b-o', label='Train', linewidth=2)
        ax1.plot(grid_sizes, test_accs, 'r-s', label='Test', linewidth=2)
        ax1.axhline(y=1/num_classes, color='gray', linestyle='--', label='Random chance')
        ax1.set_xlabel('Grid Size')
        ax1.set_ylabel('Accuracy')
        ax1.set_title(f'Counting Accuracy vs Grid Size\n({args.model_type} model)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 1.05)
        
        # Receptive field coverage overlay
        ax2 = axes[1]
        
        layers = ['inc', 'down1', 'down2', 'down3']
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        
        for layer, color in zip(layers, colors):
            coverages = []
            for gs in grid_sizes:
                rf_info = compute_receptive_field(gs)
                coverages.append(rf_info[layer]['coverage'])
            ax2.plot(grid_sizes, coverages, '-o', color=color, label=layer, linewidth=2)
        
        # Overlay test accuracy
        ax2_twin = ax2.twinx()
        ax2_twin.plot(grid_sizes, test_accs, 'k--s', label='Test Acc', linewidth=2, alpha=0.7)
        ax2_twin.set_ylabel('Test Accuracy', color='black')
        ax2_twin.set_ylim(0, 1.05)
        
        ax2.set_xlabel('Grid Size')
        ax2.set_ylabel('Receptive Field Coverage')
        ax2.set_title('RF Coverage by Layer vs Grid Size')
        ax2.legend(loc='center right')
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 1.05)
        
        plt.tight_layout()
        plt.savefig('/mnt/user-data/outputs/counting_experiment_results.png', dpi=150)
        print(f"\nVisualization saved to: /mnt/user-data/outputs/counting_experiment_results.png")
        
    print("\nDone!")


if __name__ == "__main__":
    main()