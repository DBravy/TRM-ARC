#!/usr/bin/env python3
"""
CNN Kernel and Feature Map Visualization for DorsalCNN.

Visualizes:
1. Convolutional kernel weights as grids + histograms
2. Feature map activations at each layer during inference

Usage:
    python visualize_cnn.py --checkpoint checkpoints/pixel_error_cnn.pt
    python visualize_cnn.py --checkpoint checkpoints/pixel_error_cnn.pt --kernels-only
    python visualize_cnn.py --checkpoint checkpoints/pixel_error_cnn.pt --activations-only
"""

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import torch
import torch.nn as nn

# Import model from training script
from crm import DorsalCNN, load_puzzles, GRID_SIZE, NUM_COLORS

# ARC color palette (0-9)
ARC_COLORS = [
    "#000000",  # 0: black
    "#0074D9",  # 1: blue
    "#FF4136",  # 2: red
    "#2ECC40",  # 3: green
    "#FFDC00",  # 4: yellow
    "#AAAAAA",  # 5: gray
    "#F012BE",  # 6: magenta
    "#FF851B",  # 7: orange
    "#7FDBFF",  # 8: cyan
    "#870C25",  # 9: brown
]


def get_device():
    """Get the best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# =============================================================================
# Kernel Visualization
# =============================================================================

def get_conv_layers(model: nn.Module) -> Dict[str, nn.Module]:
    """
    Find all Conv2d and ConvTranspose2d layers in the model.

    Returns:
        Dict mapping layer name to module
    """
    conv_layers = {}
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
            conv_layers[name] = module
    return conv_layers


def visualize_kernels(
    layer: nn.Module,
    layer_name: str,
    max_filters: int = 64,
    output_dir: Path = None
) -> plt.Figure:
    """
    Visualize convolutional kernel weights as a grid.

    Args:
        layer: Conv2d or ConvTranspose2d layer
        layer_name: Name for title and filename
        max_filters: Maximum number of output filters to display
        output_dir: Directory to save the figure

    Returns:
        matplotlib Figure
    """
    weights = layer.weight.data.cpu().numpy()
    # weights shape: (out_ch, in_ch, H, W) for Conv2d
    # For ConvTranspose2d: (in_ch, out_ch, H, W)

    if isinstance(layer, nn.ConvTranspose2d):
        # Swap to make consistent: (out_ch, in_ch, H, W)
        weights = weights.transpose(1, 0, 2, 3)

    out_ch, in_ch, kH, kW = weights.shape

    # Average across input channels for visualization
    weights_viz = weights.mean(axis=1)  # (out_ch, H, W)

    n_filters = min(max_filters, out_ch)
    grid_size = int(np.ceil(np.sqrt(n_filters)))

    fig, axes = plt.subplots(grid_size, grid_size, figsize=(12, 12))
    if grid_size == 1:
        axes = np.array([[axes]])
    axes = axes.flatten()

    # Symmetric color limits for diverging colormap
    vmax = max(abs(weights_viz.min()), abs(weights_viz.max()))
    vmin = -vmax

    for i in range(n_filters):
        axes[i].imshow(weights_viz[i], cmap='RdBu_r', vmin=vmin, vmax=vmax)
        axes[i].axis('off')
        axes[i].set_title(f'{i}', fontsize=8)

    # Hide unused subplots
    for i in range(n_filters, len(axes)):
        axes[i].axis('off')

    fig.suptitle(
        f'{layer_name}\n({out_ch} out, {in_ch} in, {kH}x{kW})',
        fontsize=12
    )
    plt.tight_layout()

    if output_dir:
        safe_name = layer_name.replace('.', '_')
        fig.savefig(output_dir / f'{safe_name}_filters.png', dpi=150, bbox_inches='tight')

    return fig


def visualize_kernel_statistics(
    layer: nn.Module,
    layer_name: str,
    output_dir: Path = None
) -> plt.Figure:
    """
    Visualize kernel weight distribution as histogram.

    Args:
        layer: Conv2d or ConvTranspose2d layer
        layer_name: Name for title and filename
        output_dir: Directory to save the figure

    Returns:
        matplotlib Figure
    """
    weights = layer.weight.data.cpu().numpy().flatten()

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.hist(weights, bins=50, color='steelblue', edgecolor='white', alpha=0.8)

    # Add statistics
    mean = weights.mean()
    std = weights.std()
    min_val = weights.min()
    max_val = weights.max()

    stats_text = f'Mean: {mean:.4f}\nStd: {std:.4f}\nMin: {min_val:.4f}\nMax: {max_val:.4f}'
    ax.axvline(mean, color='red', linestyle='--', linewidth=2, label=f'Mean = {mean:.4f}')
    ax.axvline(mean - std, color='orange', linestyle=':', linewidth=1.5)
    ax.axvline(mean + std, color='orange', linestyle=':', linewidth=1.5)

    ax.set_xlabel('Weight Value', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title(f'{layer_name} - Weight Distribution', fontsize=14)

    # Add text box with statistics
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.98, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', horizontalalignment='right', bbox=props)

    ax.legend(loc='upper left')
    plt.tight_layout()

    if output_dir:
        safe_name = layer_name.replace('.', '_')
        fig.savefig(output_dir / f'{safe_name}_histogram.png', dpi=150, bbox_inches='tight')

    return fig


# =============================================================================
# Activation Capture
# =============================================================================

class ActivationCapture:
    """
    Capture intermediate activations using forward hooks.

    Usage:
        capture = ActivationCapture(model, ['inc', 'down1', 'up3'])
        output = model(input_grid, output_grid)
        activations = capture.get_activations()
        capture.remove_hooks()
    """

    def __init__(self, model: nn.Module, layer_names: List[str]):
        self.activations: Dict[str, torch.Tensor] = {}
        self.hooks = []

        for name, module in model.named_modules():
            if name in layer_names:
                hook = module.register_forward_hook(self._create_hook(name))
                self.hooks.append(hook)

    def _create_hook(self, name: str):
        def hook(module, input, output):
            # Store detached copy
            self.activations[name] = output.detach().cpu()
        return hook

    def get_activations(self) -> Dict[str, torch.Tensor]:
        return self.activations

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def clear(self):
        self.activations = {}


# =============================================================================
# Feature Map Visualization
# =============================================================================

def visualize_feature_maps(
    activation: torch.Tensor,
    layer_name: str,
    max_channels: int = 64,
    sample_idx: int = 0,
    output_dir: Path = None
) -> plt.Figure:
    """
    Visualize feature maps for a single layer.

    Args:
        activation: Tensor of shape (B, C, H, W)
        layer_name: Name for title and filename
        max_channels: Maximum channels to display
        sample_idx: Which sample in batch to visualize
        output_dir: Directory to save the figure

    Returns:
        matplotlib Figure
    """
    # Get single sample
    fmaps = activation[sample_idx].numpy()  # (C, H, W)
    n_channels = min(max_channels, fmaps.shape[0])

    grid_size = int(np.ceil(np.sqrt(n_channels)))
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(14, 14))
    if grid_size == 1:
        axes = np.array([[axes]])
    axes = axes.flatten()

    for i in range(n_channels):
        im = axes[i].imshow(fmaps[i], cmap='viridis')
        axes[i].axis('off')
        axes[i].set_title(f'Ch{i}', fontsize=7)

    for i in range(n_channels, len(axes)):
        axes[i].axis('off')

    C, H, W = fmaps.shape
    fig.suptitle(f'{layer_name}\nShape: ({C}, {H}, {W})', fontsize=12)
    plt.tight_layout()

    if output_dir:
        safe_name = layer_name.replace('.', '_')
        fig.savefig(output_dir / f'{safe_name}.png', dpi=150, bbox_inches='tight')

    return fig


def visualize_activation_overlay(
    grid: np.ndarray,
    activation: torch.Tensor,
    layer_name: str,
    sample_idx: int = 0,
    output_dir: Path = None
) -> plt.Figure:
    """
    Overlay mean activation heatmap on input grid.

    Args:
        grid: Input grid of shape (H, W) with values 0-9
        activation: Tensor of shape (B, C, H, W)
        layer_name: Name for title and filename
        sample_idx: Which sample in batch to visualize
        output_dir: Directory to save the figure

    Returns:
        matplotlib Figure
    """
    fmaps = activation[sample_idx].numpy()  # (C, H, W)

    # Mean activation across channels
    mean_activation = fmaps.mean(axis=0)  # (H, W)

    # Resize activation to grid size if different
    if mean_activation.shape != grid.shape:
        from scipy.ndimage import zoom
        zoom_factor = (grid.shape[0] / mean_activation.shape[0],
                       grid.shape[1] / mean_activation.shape[1])
        mean_activation = zoom(mean_activation, zoom_factor, order=1)

    # Normalize for overlay
    act_norm = (mean_activation - mean_activation.min()) / (mean_activation.max() - mean_activation.min() + 1e-8)

    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # ARC colormap
    arc_cmap = mcolors.ListedColormap(ARC_COLORS)

    # Input grid
    axes[0].imshow(grid, cmap=arc_cmap, vmin=0, vmax=9)
    axes[0].set_title('Input Grid', fontsize=12)
    axes[0].axis('off')

    # Activation heatmap
    im = axes[1].imshow(mean_activation, cmap='hot')
    axes[1].set_title(f'{layer_name} Mean Activation', fontsize=12)
    axes[1].axis('off')
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    # Overlay
    axes[2].imshow(grid, cmap=arc_cmap, vmin=0, vmax=9)
    axes[2].imshow(act_norm, cmap='hot', alpha=0.5)
    axes[2].set_title('Overlay', fontsize=12)
    axes[2].axis('off')

    plt.tight_layout()

    if output_dir:
        safe_name = layer_name.replace('.', '_')
        fig.savefig(output_dir / f'overlay_{safe_name}.png', dpi=150, bbox_inches='tight')

    return fig


def visualize_activation_statistics(
    activations: Dict[str, torch.Tensor],
    output_dir: Path = None
) -> plt.Figure:
    """
    Show mean activation, std, and sparsity across layers.

    Args:
        activations: Dict mapping layer name to activation tensor
        output_dir: Directory to save the figure

    Returns:
        matplotlib Figure
    """
    layer_names = list(activations.keys())
    means = []
    stds = []
    sparsities = []

    for name in layer_names:
        act = activations[name].float()
        means.append(act.mean().item())
        stds.append(act.std().item())
        # ReLU sparsity (fraction of zeros)
        sparsities.append((act <= 0).float().mean().item())

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    x = range(len(layer_names))

    axes[0].bar(x, means, color='steelblue')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(layer_names, rotation=45, ha='right')
    axes[0].set_title('Mean Activation', fontsize=12)
    axes[0].set_ylabel('Mean')

    axes[1].bar(x, stds, color='coral')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(layer_names, rotation=45, ha='right')
    axes[1].set_title('Activation Std Dev', fontsize=12)
    axes[1].set_ylabel('Std')

    axes[2].bar(x, sparsities, color='seagreen')
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(layer_names, rotation=45, ha='right')
    axes[2].set_title('Sparsity (% <= 0)', fontsize=12)
    axes[2].set_ylabel('Fraction')

    plt.tight_layout()

    if output_dir:
        fig.savefig(output_dir / 'statistics.png', dpi=150, bbox_inches='tight')

    return fig


# =============================================================================
# Helper Functions
# =============================================================================

def pad_to_grid_size(grid: np.ndarray, size: int = GRID_SIZE) -> np.ndarray:
    """Pad grid to standard size."""
    h, w = grid.shape
    padded = np.zeros((size, size), dtype=grid.dtype)
    padded[:h, :w] = grid
    return padded


def get_first_example(
    puzzles: Dict,
    puzzle_id: Optional[str] = None,
    use_test: bool = False
) -> Tuple[np.ndarray, Optional[np.ndarray], str]:
    """Get the first input/output pair from puzzles.

    Args:
        puzzles: Dict of puzzle_id -> puzzle data
        puzzle_id: Optional specific puzzle ID to use. If None, uses first puzzle.
        use_test: If True, use test split instead of train split.

    Returns:
        (input_grid, output_grid, puzzle_id) - output_grid may be None for test examples
    """
    split = "test" if use_test else "train"

    if puzzle_id is not None:
        if puzzle_id not in puzzles:
            raise ValueError(f"Puzzle '{puzzle_id}' not found in dataset. Available: {list(puzzles.keys())[:10]}...")
        puzzle = puzzles[puzzle_id]
        for example in puzzle.get(split, []):
            inp = np.array(example["input"], dtype=np.uint8)
            out = np.array(example["output"], dtype=np.uint8) if "output" in example else None
            return inp, out, puzzle_id
        raise ValueError(f"Puzzle '{puzzle_id}' has no {split} examples")

    for pid, puzzle in puzzles.items():
        for example in puzzle.get(split, []):
            inp = np.array(example["input"], dtype=np.uint8)
            out = np.array(example["output"], dtype=np.uint8) if "output" in example else None
            return inp, out, pid
    raise ValueError(f"No {split} examples found in puzzles")


def get_layer_names_for_model(model: DorsalCNN) -> List[str]:
    """Get appropriate layer names based on model configuration."""
    num_layers = model.num_layers

    # Always have inc
    names = ['inc']

    # Encoder
    if num_layers >= 1:
        names.append('down1')
    if num_layers >= 2:
        names.append('down2')
    if num_layers >= 3:
        names.append('down3')

    # Decoder
    if num_layers >= 3:
        names.append('up1')
    if num_layers >= 2:
        names.append('up2')
    if num_layers >= 1:
        names.append('up3')

    return names


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Visualize CNN kernels and feature maps"
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to CNN checkpoint"
    )
    parser.add_argument(
        "--output-dir", type=str, default="viz_output",
        help="Directory to save visualizations"
    )
    parser.add_argument(
        "--kernels-only", action="store_true",
        help="Only visualize kernel weights"
    )
    parser.add_argument(
        "--activations-only", action="store_true",
        help="Only visualize activations"
    )
    parser.add_argument(
        "--max-channels", type=int, default=64,
        help="Maximum channels to display per layer"
    )
    parser.add_argument(
        "--dataset", type=str, default="arc-agi-1",
        choices=["arc-agi-1", "arc-agi-2"],
        help="Dataset to load example from"
    )
    parser.add_argument(
        "--data-root", type=str, default="kaggle/combined",
        help="Path to data root"
    )
    parser.add_argument(
        "--puzzle-id", type=str, default=None,
        help="Specific puzzle ID to use for activation visualization (defaults to first puzzle)"
    )
    parser.add_argument(
        "--use-test", action="store_true",
        help="Use test example instead of training example"
    )

    args = parser.parse_args()

    device = get_device()
    print(f"Using device: {device}")

    # Setup output directories (clear old files first)
    output_dir = Path(args.output_dir)
    kernels_dir = output_dir / "kernels"
    activations_dir = output_dir / "activations"

    # Clear existing directories to remove stale files from previous runs
    if kernels_dir.exists() and not args.activations_only:
        shutil.rmtree(kernels_dir)
    if activations_dir.exists() and not args.kernels_only:
        shutil.rmtree(activations_dir)

    kernels_dir.mkdir(parents=True, exist_ok=True)
    activations_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    print(f"Loading model from {args.checkpoint}")
    model = DorsalCNN.from_checkpoint(args.checkpoint, device=device)
    model.eval()

    print(f"Model configuration:")
    print(f"  num_layers: {model.num_layers}")
    print(f"  single_conv: {getattr(model, 'single_conv', False)}")
    print(f"  num_classes: {model.num_classes}")
    print(f"  force_comparison: {model.force_comparison}")

    # =========================================================================
    # Kernel Visualization
    # =========================================================================
    if not args.activations_only:
        print("\n" + "="*60)
        print("KERNEL VISUALIZATION")
        print("="*60)

        conv_layers = get_conv_layers(model)
        print(f"Found {len(conv_layers)} convolutional layers:")
        for name, layer in conv_layers.items():
            if isinstance(layer, nn.Conv2d):
                print(f"  {name}: Conv2d({layer.in_channels}, {layer.out_channels}, {layer.kernel_size})")
            else:
                print(f"  {name}: ConvTranspose2d({layer.in_channels}, {layer.out_channels}, {layer.kernel_size})")

        print("\nGenerating kernel visualizations...")
        for name, layer in conv_layers.items():
            print(f"  Processing {name}...")
            visualize_kernels(layer, name, max_filters=args.max_channels, output_dir=kernels_dir)
            visualize_kernel_statistics(layer, name, output_dir=kernels_dir)
            plt.close('all')

        print(f"Kernel visualizations saved to {kernels_dir}")

    # =========================================================================
    # Activation Visualization
    # =========================================================================
    if not args.kernels_only:
        print("\n" + "="*60)
        print("ACTIVATION VISUALIZATION")
        print("="*60)

        # Load puzzles and get first example
        print(f"Loading puzzles from {args.dataset}...")
        puzzles = load_puzzles(args.dataset, args.data_root)

        split_name = "test" if args.use_test else "train"
        inp_grid, out_grid, puzzle_id = get_first_example(puzzles, args.puzzle_id, args.use_test)
        print(f"Using {split_name} example from puzzle: {puzzle_id}")
        print(f"  Input shape: {inp_grid.shape}")
        if out_grid is not None:
            print(f"  Output shape: {out_grid.shape}")
        else:
            print(f"  Output: None (using input as stand-in for visualization)")
            out_grid = inp_grid  # Use input as stand-in when no output available

        # Pad to standard size
        inp_padded = pad_to_grid_size(inp_grid)
        out_padded = pad_to_grid_size(out_grid)

        # Convert to tensors
        inp_tensor = torch.from_numpy(inp_padded).long().unsqueeze(0).to(device)
        out_tensor = torch.from_numpy(out_padded).long().unsqueeze(0).to(device)

        # Get layer names for this model
        layer_names = get_layer_names_for_model(model)
        print(f"Capturing activations for layers: {layer_names}")

        # Capture activations
        capture = ActivationCapture(model, layer_names)

        with torch.no_grad():
            _ = model(inp_tensor, out_tensor)

        activations = capture.get_activations()
        capture.remove_hooks()

        print("\nGenerating activation visualizations...")
        for name in layer_names:
            if name in activations:
                print(f"  Processing {name}: shape {tuple(activations[name].shape)}")
                visualize_feature_maps(
                    activations[name], name,
                    max_channels=args.max_channels,
                    output_dir=activations_dir
                )
                visualize_activation_overlay(
                    inp_padded, activations[name], name,
                    output_dir=activations_dir
                )
                plt.close('all')

        # Statistics
        print("  Generating statistics...")
        visualize_activation_statistics(activations, output_dir=activations_dir)
        plt.close('all')

        print(f"Activation visualizations saved to {activations_dir}")

    print("\n" + "="*60)
    print("VISUALIZATION COMPLETE")
    print("="*60)
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()
