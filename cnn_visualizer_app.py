#!/usr/bin/env python3
"""
CNN Learning Visualization Web App for ARC Patterns.

Provides interactive visualization of how PixelErrorCNN learns ARC patterns:
1. Per-color kernel visualization - How each kernel responds to each color
2. Layer flow animation - Step through layers to see transformations
3. Puzzle browser - Select puzzles from the ARC dataset

Usage:
    python cnn_visualizer_app.py --checkpoint checkpoints/pixel_error_cnn.pt
    python cnn_visualizer_app.py --checkpoint checkpoints/pixel_error_cnn.pt --port 5005
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from flask import Flask, render_template, jsonify, request

# Import from existing modules
from train_pixel_error_cnn import PixelErrorCNN, load_puzzles, GRID_SIZE, NUM_COLORS
from visualize_cnn import (
    ActivationCapture,
    get_layer_names_for_model,
    ARC_COLORS,
    pad_to_grid_size,
    get_conv_layers
)

app = Flask(__name__,
            template_folder='templates',
            static_folder='static')

# Global state
model: Optional[PixelErrorCNN] = None
puzzles: Optional[Dict] = None
device: Optional[torch.device] = None
checkpoint_path: str = ""


def get_device():
    """Get the best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def init_app(ckpt_path: str, dataset: str = "arc-agi-1", data_root: str = "kaggle/combined"):
    """Initialize the model and load puzzle data."""
    global model, puzzles, device, checkpoint_path

    device = get_device()
    checkpoint_path = ckpt_path

    print(f"Loading model from {ckpt_path}")
    model = PixelErrorCNN.from_checkpoint(ckpt_path, device=device)
    model.eval()

    print(f"Model loaded: {model.num_layers} layers, hidden_dim={model.inc.conv[0].out_channels if hasattr(model.inc, 'conv') else 'unknown'}")

    print(f"Loading puzzles from {dataset}")
    puzzles = load_puzzles(dataset, data_root)
    print(f"Loaded {len(puzzles)} puzzles")


# =============================================================================
# Per-Color Kernel Response Computation
# =============================================================================

def compute_per_color_kernel_response(model: PixelErrorCNN) -> Dict:
    """
    Compute effective kernel response for each of the 10 ARC colors.

    The model uses 16-dim embeddings for each color. The first conv layer receives
    64 input channels: [inp_emb(16), out_emb(16), diff(16), prod(16)].

    For each color, we compute the "effective kernel" - the kernel's response
    when that color is present in both input and output (correct match).

    Returns:
        Dict with 'kernels' (10 x out_ch x 3 x 3), 'num_filters', and metadata
    """
    inp_embed = model.input_embed.weight.detach()  # (10, 16)
    out_embed = model.output_embed.weight.detach()  # (10, 16)

    # Get first conv layer kernel
    if hasattr(model.inc, 'conv'):
        # DoubleConv or SingleConv case
        kernel = model.inc.conv[0].weight.detach()  # (out_ch, 64, 3, 3)
    else:
        kernel = model.inc[0].weight.detach()

    out_ch, in_ch, kH, kW = kernel.shape

    responses = []
    for c in range(NUM_COLORS):
        ie = inp_embed[c]  # (16,)
        oe = out_embed[c]  # (16,)

        # When input color == output color (correct match)
        diff = ie - oe  # Should be near 0 for same embedding
        prod = ie * oe  # Element-wise product

        # Combine embeddings in same order as model forward pass
        combined = torch.cat([ie, oe, diff, prod])  # (64,)

        # Compute weighted sum across input channels
        # kernel: (out_ch, 64, 3, 3), combined: (64,)
        # Result: (out_ch, 3, 3) - effective kernel for this color
        effective = (kernel * combined.view(1, -1, 1, 1)).sum(dim=1)
        responses.append(effective.cpu().numpy())

    responses = np.array(responses)  # (10, out_ch, 3, 3)

    return {
        'kernels': responses.tolist(),
        'num_filters': out_ch,
        'kernel_size': [kH, kW],
        'colors': ARC_COLORS
    }


def compute_all_kernel_responses(model: PixelErrorCNN) -> Dict:
    """
    Compute per-color responses for all conv layers.

    For deeper layers, we can't directly compute per-color response since
    they operate on transformed features. Instead, we show the raw kernels.

    Returns:
        Dict with layer-wise kernel information
    """
    result = {
        'first_layer': compute_per_color_kernel_response(model),
        'all_layers': {}
    }

    conv_layers = get_conv_layers(model)
    for name, layer in conv_layers.items():
        weights = layer.weight.data.cpu().numpy()
        out_ch, in_ch, kH, kW = weights.shape

        # For deeper layers, just show averaged kernels (or first few input channels)
        result['all_layers'][name] = {
            'shape': [out_ch, in_ch, kH, kW],
            'weights_mean': weights.mean(axis=1).tolist(),  # Average across input channels
            'weights_stats': {
                'mean': float(weights.mean()),
                'std': float(weights.std()),
                'min': float(weights.min()),
                'max': float(weights.max())
            }
        }

    return result


# =============================================================================
# Layer Flow Capture
# =============================================================================

def capture_layer_flow(
    input_grid: np.ndarray,
    output_grid: np.ndarray,
    use_zeros_as_candidate: bool = True
) -> Dict:
    """
    Run inference and capture activations at each layer.

    IMPORTANT: For proper test evaluation (matching training behavior), the model
    should receive (input, zeros) as the candidate, NOT (input, correct_output).
    The model then predicts what each pixel's color should be.

    Args:
        input_grid: Input puzzle grid (H, W) with values 0-9
        output_grid: Expected output puzzle grid (H, W) with values 0-9
        use_zeros_as_candidate: If True, pass zeros as candidate (proper test mode).
                                If False, pass the correct output (shows perfect match).

    Returns:
        Dict with input/output grids, prediction, and layer-wise activations
    """
    # Get all layer names including the output conv
    layer_names = get_layer_names_for_model(model)
    layer_names_with_outc = layer_names + ['outc']

    capture = ActivationCapture(model, layer_names_with_outc)

    # Pad to standard size and convert
    inp_pad = pad_to_grid_size(input_grid)
    out_pad = pad_to_grid_size(output_grid)

    inp_tensor = torch.from_numpy(inp_pad).long().unsqueeze(0).to(device)
    expected_tensor = torch.from_numpy(out_pad).long().unsqueeze(0).to(device)

    # For proper evaluation: use zeros as candidate (model must predict from scratch)
    # This matches how training evaluates test examples
    if use_zeros_as_candidate:
        candidate_tensor = torch.zeros_like(inp_tensor).to(device)
    else:
        candidate_tensor = expected_tensor

    # Run inference with candidate (zeros for real test, or expected for comparison)
    with torch.no_grad():
        logits = model(inp_tensor, candidate_tensor)

        # Get prediction based on model type
        if model.num_classes == 1:
            # Binary mode - predict correct (1) vs incorrect (0)
            probs = torch.sigmoid(logits)
            prediction = (probs > 0.5).squeeze(1).long()
        else:
            # Color mode - predict what color each pixel should be
            probs = torch.softmax(logits, dim=1)  # (1, 10, 30, 30)
            prediction = logits.argmax(dim=1)

        prediction = prediction.squeeze(0).cpu().numpy()
        logits_np = logits.squeeze(0).cpu().numpy()  # (10, 30, 30) or (1, 30, 30)
        probs_np = probs.squeeze(0).cpu().numpy()  # (10, 30, 30) or (1, 30, 30)

    activations = capture.get_activations()
    capture.remove_hooks()

    # Build result with layer information
    layers = []
    for name in layer_names_with_outc:
        if name in activations:
            act = activations[name][0]  # Remove batch dimension
            C, H, W = act.shape

            # Normalize activations for visualization
            act_np = act.numpy()
            act_min, act_max = act_np.min(), act_np.max()
            if act_max - act_min > 1e-8:
                act_normalized = (act_np - act_min) / (act_max - act_min)
            else:
                act_normalized = np.zeros_like(act_np)

            layer_info = {
                'name': name,
                'shape': [C, H, W],
                'activations': act_normalized.tolist(),
                'mean_activation': act_normalized.mean(axis=0).tolist(),
                'stats': {
                    'min': float(act_min),
                    'max': float(act_max),
                    'mean': float(act_np.mean()),
                    'std': float(act_np.std())
                }
            }

            # For the output layer, include raw logits and probabilities
            if name == 'outc':
                layer_info['is_output'] = True
                layer_info['logits'] = logits_np.tolist()
                layer_info['probabilities'] = probs_np.tolist()
                layer_info['color_names'] = [
                    'Black', 'Blue', 'Red', 'Green', 'Yellow',
                    'Gray', 'Magenta', 'Orange', 'Cyan', 'Brown'
                ]

            layers.append(layer_info)

    # Also get embedding visualization (show expected output embeddings for comparison)
    with torch.no_grad():
        inp_emb = model.input_embed(inp_tensor).squeeze(0).cpu().numpy()  # (30, 30, 16)
        out_emb = model.output_embed(expected_tensor).squeeze(0).cpu().numpy()  # (30, 30, 16)

    # What candidate was used for inference
    candidate_grid = candidate_tensor.squeeze(0).cpu().numpy()

    return {
        'input_grid': inp_pad.tolist(),
        'output_grid': out_pad.tolist(),  # This is the EXPECTED output
        'candidate_grid': candidate_grid.tolist(),  # This is what was passed to model (zeros or expected)
        'used_zeros_as_candidate': use_zeros_as_candidate,
        'input_shape': list(input_grid.shape),
        'output_shape': list(output_grid.shape),
        'prediction': prediction.tolist(),
        'layers': layers,
        'logits': logits_np.tolist(),
        'probabilities': probs_np.tolist(),
        'embeddings': {
            'input': inp_emb.tolist(),
            'output': out_emb.tolist()
        }
    }


# =============================================================================
# API Routes
# =============================================================================

@app.route('/')
def index():
    """Serve the main visualization page."""
    return render_template('cnn_viz.html')


@app.route('/api/model-info')
def api_model_info():
    """Return model configuration and layer structure."""
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500

    layer_names = get_layer_names_for_model(model)
    conv_layers = get_conv_layers(model)

    # Get hidden dimension from first conv
    if hasattr(model.inc, 'conv'):
        hidden_dim = model.inc.conv[0].out_channels
    else:
        hidden_dim = model.inc[0].out_channels

    return jsonify({
        'checkpoint': checkpoint_path,
        'num_layers': model.num_layers,
        'hidden_dim': hidden_dim,
        'single_conv': model.single_conv,
        'force_comparison': model.force_comparison,
        'num_classes': model.num_classes,
        'use_attention': model.use_attention,
        'layer_names': layer_names,
        'conv_layers': {
            name: {
                'type': type(layer).__name__,
                'in_channels': layer.in_channels,
                'out_channels': layer.out_channels,
                'kernel_size': list(layer.kernel_size)
            }
            for name, layer in conv_layers.items()
        },
        'arc_colors': ARC_COLORS
    })


@app.route('/api/kernels')
def api_kernels():
    """Return per-color kernel responses."""
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500

    try:
        responses = compute_all_kernel_responses(model)
        return jsonify(responses)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/puzzles')
def api_puzzles():
    """Return list of all puzzle IDs with metadata."""
    if puzzles is None:
        return jsonify({'error': 'Puzzles not loaded'}), 500

    puzzle_list = []
    for pid, puzzle in puzzles.items():
        train_examples = puzzle.get('train', [])
        test_examples = puzzle.get('test', [])

        puzzle_list.append({
            'id': pid,
            'num_train': len(train_examples),
            'num_test': len(test_examples),
            'train_sizes': [
                [len(ex['input']), len(ex['input'][0]) if ex['input'] else 0]
                for ex in train_examples
            ],
            'test_sizes': [
                [len(ex['input']), len(ex['input'][0]) if ex['input'] else 0]
                for ex in test_examples
            ]
        })

    return jsonify({
        'puzzles': puzzle_list,
        'total': len(puzzle_list)
    })


@app.route('/api/puzzle/<puzzle_id>')
def api_puzzle(puzzle_id: str):
    """Return full puzzle data for a specific puzzle."""
    if puzzles is None:
        return jsonify({'error': 'Puzzles not loaded'}), 500

    if puzzle_id not in puzzles:
        return jsonify({'error': f'Puzzle {puzzle_id} not found'}), 404

    puzzle = puzzles[puzzle_id]
    return jsonify({
        'id': puzzle_id,
        'train': puzzle.get('train', []),
        'test': puzzle.get('test', [])
    })


@app.route('/api/flow/<puzzle_id>/<int:example_idx>')
def api_flow(puzzle_id: str, example_idx: int):
    """
    Run inference and return layer activations for a specific puzzle example.

    Args:
        puzzle_id: The puzzle ID
        example_idx: Index of the example (0-based, train examples first, then test)
    """
    if model is None or puzzles is None:
        return jsonify({'error': 'Model or puzzles not loaded'}), 500

    if puzzle_id not in puzzles:
        return jsonify({'error': f'Puzzle {puzzle_id} not found'}), 404

    puzzle = puzzles[puzzle_id]
    train_examples = puzzle.get('train', [])
    test_examples = puzzle.get('test', [])

    # Determine which example to use
    if example_idx < len(train_examples):
        example = train_examples[example_idx]
        split = 'train'
    elif example_idx < len(train_examples) + len(test_examples):
        example = test_examples[example_idx - len(train_examples)]
        split = 'test'
    else:
        return jsonify({'error': f'Example index {example_idx} out of range'}), 404

    input_grid = np.array(example['input'], dtype=np.uint8)
    output_grid = np.array(example.get('output', example['input']), dtype=np.uint8)

    try:
        result = capture_layer_flow(input_grid, output_grid)
        result['puzzle_id'] = puzzle_id
        result['example_idx'] = example_idx
        result['split'] = split
        return jsonify(result)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/compare-colors')
def api_compare_colors():
    """
    Return visualization of how the model compares different color pairs.

    Shows the embedding difference/product for each pair of input/output colors.
    """
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500

    inp_embed = model.input_embed.weight.detach().cpu().numpy()  # (10, 16)
    out_embed = model.output_embed.weight.detach().cpu().numpy()  # (10, 16)

    # Compute pairwise comparisons
    comparisons = {}
    for ic in range(NUM_COLORS):
        for oc in range(NUM_COLORS):
            ie = inp_embed[ic]
            oe = out_embed[oc]

            diff = ie - oe
            prod = ie * oe

            comparisons[f'{ic}_{oc}'] = {
                'input_color': ic,
                'output_color': oc,
                'diff_norm': float(np.linalg.norm(diff)),
                'prod_norm': float(np.linalg.norm(prod)),
                'diff_mean': float(diff.mean()),
                'prod_mean': float(prod.mean())
            }

    return jsonify({
        'comparisons': comparisons,
        'input_embed': inp_embed.tolist(),
        'output_embed': out_embed.tolist()
    })


# =============================================================================
# Single Pixel Computation Trace
# =============================================================================

def compute_pixel_trace(
    input_grid: np.ndarray,
    output_grid: np.ndarray,
    row: int,
    col: int
) -> Dict:
    """
    Compute a complete breakdown of how the model arrives at a prediction
    for a specific pixel location.

    For num_layers=0 models (inc -> outc), this traces:
    1. Receptive field patch from input and candidate grids
    2. Embedding vectors for center pixel
    3. Feature vector from inc layer at this position
    4. Final calculation: outc weights, dot product, bias, logits
    5. Softmax probabilities and prediction

    Args:
        input_grid: Input puzzle grid (H, W)
        output_grid: Expected output grid (H, W)
        row: Row index of the pixel to trace
        col: Column index of the pixel to trace

    Returns:
        Dict with complete computation trace
    """
    # Pad grids to standard size
    inp_pad = pad_to_grid_size(input_grid)
    out_pad = pad_to_grid_size(output_grid)

    inp_tensor = torch.from_numpy(inp_pad).long().unsqueeze(0).to(device)
    expected_tensor = torch.from_numpy(out_pad).long().unsqueeze(0).to(device)

    # Use zeros as candidate (test mode)
    candidate_tensor = torch.zeros_like(inp_tensor).to(device)

    # Determine receptive field size based on model architecture
    # Get kernel size from the first conv layer
    if hasattr(model.inc, 'conv'):
        kernel_size = model.inc.conv[0].kernel_size[0]  # Assumes square kernels
    else:
        kernel_size = model.inc[0].kernel_size[0]

    # For DoubleConv (two convs stacked), effective RF expands
    # RF = k + (k-1) for two stacked convs of size k
    if model.single_conv:
        rf_size = kernel_size
    else:
        rf_size = kernel_size + (kernel_size - 1)  # e.g., 3+2=5 for k=3, 5+4=9 for k=5
    rf_half = rf_size // 2

    # Calculate receptive field bounds (clamped to grid boundaries)
    rf_row_start = max(0, row - rf_half)
    rf_row_end = min(GRID_SIZE, row + rf_half + 1)
    rf_col_start = max(0, col - rf_half)
    rf_col_end = min(GRID_SIZE, col + rf_half + 1)

    # Extract receptive field patches
    input_rf_patch = inp_pad[rf_row_start:rf_row_end, rf_col_start:rf_col_end].tolist()
    candidate_rf_patch = candidate_tensor[0, rf_row_start:rf_row_end, rf_col_start:rf_col_end].cpu().numpy().tolist()

    # Get colors present in receptive field
    input_colors_in_rf = sorted(set(inp_pad[rf_row_start:rf_row_end, rf_col_start:rf_col_end].flatten().tolist()))

    # Get embedding vectors for the center pixel
    with torch.no_grad():
        # Embeddings for the entire grid
        inp_emb_full = model.input_embed(inp_tensor)  # (1, 30, 30, 16)
        out_emb_full = model.output_embed(candidate_tensor)  # (1, 30, 30, 16)

        # Extract embedding at the target position
        inp_emb_pixel = inp_emb_full[0, row, col].cpu().numpy()  # (16,)
        out_emb_pixel = out_emb_full[0, row, col].cpu().numpy()  # (16,)

        # Compute comparison features
        diff_emb = inp_emb_pixel - out_emb_pixel
        prod_emb = inp_emb_pixel * out_emb_pixel

        # Build combined input (same as forward pass)
        if model.force_comparison:
            combined = np.concatenate([inp_emb_pixel, out_emb_pixel, diff_emb, prod_emb])  # (64,)
        else:
            combined = np.concatenate([inp_emb_pixel, out_emb_pixel])  # (32,)

        # Run forward pass through inc layer to get feature vector
        if model.force_comparison:
            x = torch.cat([inp_emb_full, out_emb_full,
                          inp_emb_full - out_emb_full,
                          inp_emb_full * out_emb_full], dim=-1)
        else:
            x = torch.cat([inp_emb_full, out_emb_full], dim=-1)

        x = x.permute(0, 3, 1, 2).contiguous()  # (1, C, 30, 30)

        # Pass through inc layer
        inc_output = model.inc(x)  # (1, hidden_dim, 30, 30)

        # Extract feature vector at the target position
        feature_vector = inc_output[0, :, row, col].cpu().numpy()  # (hidden_dim,)
        hidden_dim = feature_vector.shape[0]

        # Get outc layer weights and bias
        outc_weight = model.outc.weight.data.cpu().numpy()  # (num_classes, hidden_dim, 1, 1)
        outc_weight = outc_weight.squeeze(-1).squeeze(-1)  # (num_classes, hidden_dim)
        outc_bias = model.outc.bias.data.cpu().numpy() if model.outc.bias is not None else np.zeros(model.num_classes)

        # Compute logits manually for this pixel
        logits_manual = outc_weight @ feature_vector + outc_bias  # (num_classes,)

        # Get actual model output for verification
        logits_full = model(inp_tensor, candidate_tensor)  # (1, num_classes, 30, 30)
        logits_at_pixel = logits_full[0, :, row, col].cpu().numpy()

        # Compute probabilities
        probs = torch.softmax(torch.from_numpy(logits_at_pixel), dim=0).numpy()

        # Get prediction
        predicted_color = int(np.argmax(logits_at_pixel))
        expected_color = int(out_pad[row, col])
        input_color = int(inp_pad[row, col])
        is_correct = predicted_color == expected_color

    # Build result
    result = {
        'pixel': {
            'row': row,
            'col': col,
            'input_color': input_color,
            'expected_color': expected_color,
            'predicted_color': predicted_color,
            'is_correct': is_correct
        },
        'receptive_field': {
            'size': rf_size,
            'bounds': {
                'row_start': rf_row_start,
                'row_end': rf_row_end,
                'col_start': rf_col_start,
                'col_end': rf_col_end
            },
            'input_patch': input_rf_patch,
            'candidate_patch': candidate_rf_patch,
            'colors_in_input_rf': input_colors_in_rf
        },
        'embeddings': {
            'input_embedding': inp_emb_pixel.tolist(),
            'candidate_embedding': out_emb_pixel.tolist(),
            'difference': diff_emb.tolist(),
            'product': prod_emb.tolist(),
            'combined_input': combined.tolist()
        },
        'feature_vector': {
            'values': feature_vector.tolist(),
            'hidden_dim': hidden_dim,
            'stats': {
                'min': float(feature_vector.min()),
                'max': float(feature_vector.max()),
                'mean': float(feature_vector.mean()),
                'std': float(feature_vector.std())
            },
            # Per-feature contributions: how each feature dim contributes to each color's logit
            # contribution[color][dim] = feature_vector[dim] * outc_weight[color][dim]
            'contributions': (outc_weight * feature_vector).tolist(),  # (num_classes, hidden_dim)
        },
        'output_layer': {
            'weights_shape': list(outc_weight.shape),
            'per_class_calculation': []
        },
        'prediction': {
            'logits': logits_at_pixel.tolist(),
            'probabilities': probs.tolist(),
            'predicted_color': predicted_color,
            'expected_color': expected_color,
            'is_correct': is_correct,
            'color_names': [
                'Black', 'Blue', 'Red', 'Green', 'Yellow',
                'Gray', 'Magenta', 'Orange', 'Cyan', 'Brown'
            ]
        }
    }

    # Add per-class calculation breakdown
    for c in range(model.num_classes):
        weight_vec = outc_weight[c]
        dot_product = float(np.dot(weight_vec, feature_vector))
        bias_val = float(outc_bias[c])
        logit = dot_product + bias_val

        result['output_layer']['per_class_calculation'].append({
            'color': c,
            'color_name': result['prediction']['color_names'][c],
            'weight_vector': weight_vec.tolist(),
            'dot_product': dot_product,
            'bias': bias_val,
            'logit': logit,
            'probability': float(probs[c])
        })

    return result


@app.route('/api/pixel-trace/<puzzle_id>/<int:example_idx>/<int:row>/<int:col>')
def api_pixel_trace(puzzle_id: str, example_idx: int, row: int, col: int):
    """
    Compute detailed trace of how the model predicts a specific pixel.

    Returns complete breakdown including:
    - Receptive field visualization
    - Embedding vectors
    - Feature vector from inc layer
    - Output layer calculation (weights, dot product, bias, logits)
    - Softmax probabilities
    """
    if model is None or puzzles is None:
        return jsonify({'error': 'Model or puzzles not loaded'}), 500

    if puzzle_id not in puzzles:
        return jsonify({'error': f'Puzzle {puzzle_id} not found'}), 404

    # Validate coordinates
    if not (0 <= row < GRID_SIZE and 0 <= col < GRID_SIZE):
        return jsonify({'error': f'Invalid coordinates ({row}, {col})'}), 400

    puzzle = puzzles[puzzle_id]
    train_examples = puzzle.get('train', [])
    test_examples = puzzle.get('test', [])

    # Determine which example to use
    if example_idx < len(train_examples):
        example = train_examples[example_idx]
    elif example_idx < len(train_examples) + len(test_examples):
        example = test_examples[example_idx - len(train_examples)]
    else:
        return jsonify({'error': f'Example index {example_idx} out of range'}), 404

    input_grid = np.array(example['input'], dtype=np.uint8)
    output_grid = np.array(example.get('output', example['input']), dtype=np.uint8)

    try:
        result = compute_pixel_trace(input_grid, output_grid, row, col)
        result['puzzle_id'] = puzzle_id
        result['example_idx'] = example_idx
        result['model_info'] = {
            'num_layers': model.num_layers,
            'single_conv': model.single_conv,
            'force_comparison': model.force_comparison,
            'num_classes': model.num_classes
        }
        return jsonify(result)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


# =============================================================================
# Main
# =============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="CNN Learning Visualization Web App"
    )
    parser.add_argument(
        '--checkpoint', type=str,
        default='checkpoints/pixel_error_cnn.pt',
        help='Path to CNN checkpoint'
    )
    parser.add_argument(
        '--dataset', type=str,
        default='arc-agi-1',
        choices=['arc-agi-1', 'arc-agi-2'],
        help='Dataset to load puzzles from'
    )
    parser.add_argument(
        '--data-root', type=str,
        default='kaggle/combined',
        help='Path to data root'
    )
    parser.add_argument(
        '--port', type=int,
        default=5005,
        help='Port to run the server on'
    )
    parser.add_argument(
        '--debug', action='store_true',
        help='Run in debug mode'
    )

    args = parser.parse_args()

    # Initialize model and data
    init_app(args.checkpoint, args.dataset, args.data_root)

    print(f"\n{'='*50}")
    print("CNN Learning Visualizer")
    print('='*50)
    print(f"\nOpen http://localhost:{args.port} in your browser")
    print("Press Ctrl+C to stop the server\n")

    app.run(host='0.0.0.0', port=args.port, debug=args.debug, threaded=True)
