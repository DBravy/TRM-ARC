#!/usr/bin/env python3
"""
CNN Learning Visualization Web App for ARC Patterns.

Provides interactive visualization of how DorsalCNN learns ARC patterns:
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
import torch.nn.functional as F
from flask import Flask, render_template, jsonify, request

# Import from existing modules
from crm import (
    DorsalCNN, load_puzzles, GRID_SIZE, NUM_COLORS,
    SpatialSelfAttention, CrossAttention, SlotAttention, SlotRoutedCrossAttention
)
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
model: Optional[DorsalCNN] = None
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
    model = DorsalCNN.from_checkpoint(ckpt_path, device=device)
    model.eval()

    print(f"Model loaded: {model.num_layers} layers, hidden_dim={model.inc.conv[0].out_channels if hasattr(model.inc, 'conv') else 'unknown'}")

    print(f"Loading puzzles from {dataset}")
    puzzles = load_puzzles(dataset, data_root)
    print(f"Loaded {len(puzzles)} puzzles")


# =============================================================================
# Per-Color Kernel Response Computation
# =============================================================================

def compute_per_color_kernel_response(model: DorsalCNN) -> Dict:
    """
    Compute effective kernel response for each of the 10 ARC colors.

    For embedding models: Uses 16-dim learned embeddings for each color.
    For one-hot models: Uses fixed 11-dim one-hot + content mask encoding.

    For each color, we compute the "effective kernel" - the kernel's response
    when that color is present in both input and output (correct match).

    Returns:
        Dict with 'kernels' (10 x out_ch x 3 x 3), 'num_filters', and metadata
    """
    # Get first conv layer kernel
    if hasattr(model.inc, 'conv'):
        # DoubleConv or SingleConv case
        kernel = model.inc.conv[0].weight.detach()  # (out_ch, in_ch, 3, 3)
    else:
        kernel = model.inc[0].weight.detach()

    out_ch, in_ch, kH, kW = kernel.shape

    responses = []
    for c in range(NUM_COLORS):
        if model.use_onehot:
            # One-hot encoding: 10 one-hot + 1 content mask = 11 dims per grid
            # Content mask is 1 for all real colors (0-9), 0 for padding (10)
            # Input encoding
            ie = torch.zeros(11, device=kernel.device)
            ie[c] = 1.0  # one-hot
            ie[10] = 1.0  # content mask (always 1 for actual colors 0-9)
            # Output encoding (same for matching colors)
            oe = ie.clone()
            # Combined: [input(11), output(11)] = 22 channels
            combined = torch.cat([ie, oe])  # (22,)
        else:
            # Learned embeddings
            inp_embed = model.input_embed.weight.detach()  # (10, 16)
            out_embed = model.output_embed.weight.detach()  # (10, 16)
            ie = inp_embed[c]  # (16,)
            oe = out_embed[c]  # (16,)
            # Combined: [input(16), output(16)] = 32 channels
            combined = torch.cat([ie, oe])  # (32,)

        # Compute weighted sum across input channels
        # kernel: (out_ch, in_ch, 3, 3), combined: (in_ch,)
        # Result: (out_ch, 3, 3) - effective kernel for this color
        effective = (kernel * combined.view(1, -1, 1, 1)).sum(dim=1)
        responses.append(effective.cpu().numpy())

    responses = np.array(responses)  # (10, out_ch, 3, 3)

    return {
        'kernels': responses.tolist(),
        'num_filters': out_ch,
        'kernel_size': [kH, kW],
        'colors': ARC_COLORS,
        'encoding': 'onehot' if model.use_onehot else 'embedding'
    }


def compute_all_kernel_responses(model: DorsalCNN) -> Dict:
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

    # Check if model has size prediction enabled
    has_size_prediction = getattr(model, 'predict_size', False)
    predicted_height = None
    predicted_width = None

    # Run inference with candidate (zeros for real test, or expected for comparison)
    with torch.no_grad():
        logits = model(inp_tensor, candidate_tensor)

        # Handle dict output from size-predicting models
        if isinstance(logits, dict):
            pixel_logits = logits['pixel_logits']
            height_logits = logits['height_logits']
            width_logits = logits['width_logits']
            # Get predicted size (convert from 0-29 class to 1-30 size)
            predicted_height = int(height_logits.argmax(dim=1).item()) + 1
            predicted_width = int(width_logits.argmax(dim=1).item()) + 1
            logits = pixel_logits  # Use pixel logits for the rest of processing

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
        if model.use_onehot:
            inp_emb = model._encode_grid(inp_tensor).squeeze(0).cpu().numpy()  # (30, 30, 11)
            out_emb = model._encode_grid(expected_tensor).squeeze(0).cpu().numpy()  # (30, 30, 11)
        else:
            inp_emb = model.input_embed(inp_tensor).squeeze(0).cpu().numpy()  # (30, 30, 16)
            out_emb = model.output_embed(expected_tensor).squeeze(0).cpu().numpy()  # (30, 30, 16)

    # What candidate was used for inference
    candidate_grid = candidate_tensor.squeeze(0).cpu().numpy()

    result = {
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

    # Add size prediction info if model has it enabled
    if has_size_prediction and predicted_height is not None:
        result['has_size_prediction'] = True
        result['predicted_height'] = predicted_height
        result['predicted_width'] = predicted_width
    else:
        result['has_size_prediction'] = False

    return result


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

    # Slot attention info
    use_slot_cross_attention = getattr(model, 'use_slot_cross_attention', False)
    slot_info = {}
    if use_slot_cross_attention and model.slot_attention is not None:
        slot_info = {
            'num_slots': getattr(model, 'num_slots', getattr(model.slot_attention, 'num_slots', 10)),
            'slot_dim': model.slot_dim,
            'slot_iterations': getattr(model, 'slot_iterations', None),  # None for color/affinity slots
            'slot_type': 'affinity' if getattr(model, 'use_affinity_slots', False) else ('color' if getattr(model, 'use_color_slots', False) else 'learned'),
        }

    # Cross-attention info
    cross_attention_info = {}
    if getattr(model, 'use_cross_attention', False) and model.cross_attention is not None:
        cross_attention_info = {
            'num_heads': model.cross_attention.num_heads,
            'head_dim': model.cross_attention.head_dim,
            'proj_dim': model.cross_attention.proj_dim,
            'no_softmax': getattr(model.cross_attention, 'no_softmax', False),
        }

    # Per-layer cross-attention info
    per_layer_cross_attention_info = {}
    cross_attention_per_layer = getattr(model, 'cross_attention_per_layer', False)
    use_per_layer_slot = getattr(model, 'use_per_layer_slot', False)
    if cross_attention_per_layer or use_per_layer_slot:
        per_layer_cross_attention_info = {
            'enabled': True,
            'is_slot_attention': use_per_layer_slot,
            'shared': getattr(model, 'cross_attention_shared', False),
            'conv_depth': model.conv_depth,
            'num_layers': model.num_layers,
        }
        # Count total cross-attention applications
        num_blocks = 1  # inc always exists
        if model.num_layers >= 1:
            num_blocks += 2  # down1 + up3
        if model.num_layers >= 2:
            num_blocks += 2  # down2 + up2
        if model.num_layers >= 3:
            num_blocks += 2  # down3 + up1
        per_layer_cross_attention_info['total_applications'] = num_blocks * model.conv_depth

    return jsonify({
        'checkpoint': checkpoint_path,
        'num_layers': model.num_layers,
        'hidden_dim': hidden_dim,
        'conv_depth': model.conv_depth,
        'kernel_size': model.kernel_size,
        'force_comparison': getattr(model, 'force_comparison', False),
        'use_onehot': model.use_onehot,
        'use_attention': getattr(model, 'use_attention', False),
        'use_cross_attention': getattr(model, 'use_cross_attention', False),
        'cross_attention_info': cross_attention_info,
        'cross_attention_per_layer': cross_attention_per_layer or use_per_layer_slot,
        'per_layer_cross_attention_info': per_layer_cross_attention_info,
        'use_slot_cross_attention': use_slot_cross_attention,
        'slot_info': slot_info,
        'predict_size': getattr(model, 'predict_size', False),
        'encoding': 'one-hot (11 ch/grid)' if model.use_onehot else 'learned embeddings (16 ch/grid)',
        'num_classes': model.num_classes,
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

    For embedding models: Shows learned embedding difference/product for each pair.
    For one-hot models: Shows fixed one-hot encoding comparisons.
    """
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500

    if model.use_onehot:
        # One-hot encoding: fixed 11-dim vectors (10 one-hot + 1 content mask)
        # Content mask is 1 for all real colors (0-9), 0 for padding (10)
        inp_embed = np.zeros((NUM_COLORS, 11), dtype=np.float32)
        out_embed = np.zeros((NUM_COLORS, 11), dtype=np.float32)
        for c in range(NUM_COLORS):
            inp_embed[c, c] = 1.0  # one-hot
            inp_embed[c, 10] = 1.0  # content mask (always 1 for actual colors 0-9)
            out_embed[c, c] = 1.0
            out_embed[c, 10] = 1.0  # content mask
    else:
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
        'output_embed': out_embed.tolist(),
        'encoding': 'onehot' if model.use_onehot else 'embedding'
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
    # Get kernel size and conv depth from model
    kernel_size = model.kernel_size
    conv_depth = model.conv_depth

    # RF = conv_depth * (kernel_size - 1) + 1
    # e.g., depth=2, k=3: 2*2+1=5, depth=3, k=3: 3*2+1=7, depth=4, k=3: 4*2+1=9
    rf_size = conv_depth * (kernel_size - 1) + 1
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

    # Get embedding/encoding vectors for the center pixel
    with torch.no_grad():
        # Embeddings for the entire grid (or one-hot encodings)
        if model.use_onehot:
            inp_emb_full = model._encode_grid(inp_tensor)  # (1, 30, 30, 11)
            out_emb_full = model._encode_grid(candidate_tensor)  # (1, 30, 30, 11)
        else:
            inp_emb_full = model.input_embed(inp_tensor)  # (1, 30, 30, 16)
            out_emb_full = model.output_embed(candidate_tensor)  # (1, 30, 30, 16)

        # Extract embedding at the target position
        inp_emb_pixel = inp_emb_full[0, row, col].cpu().numpy()
        out_emb_pixel = out_emb_full[0, row, col].cpu().numpy()

        # Build combined input (same as forward pass)
        # Current model just concatenates [inp, out], no diff/prod
        combined = np.concatenate([inp_emb_pixel, out_emb_pixel])

        # Run forward pass through inc layer to get feature vector
        x = torch.cat([inp_emb_full, out_emb_full], dim=-1)

        x = x.permute(0, 3, 1, 2).contiguous()  # (1, C, 30, 30)

        # Pass through inc layer
        inc_output = model.inc(x)  # (1, hidden_dim, 30, 30)

        # Get outc layer weights and bias
        outc_weight = model.outc.weight.data.cpu().numpy()  # (num_classes, hidden_dim, kH, kW)
        outc_bias = model.outc.bias.data.cpu().numpy() if model.outc.bias is not None else np.zeros(model.num_classes)

        # Get kernel size from the weight shape
        kH, kW = outc_weight.shape[2], outc_weight.shape[3]
        pad_h, pad_w = kH // 2, kW // 2

        # Extract the patch of features needed for this pixel's output
        # For a 3x3 kernel with padding=1, we need a 3x3 patch centered at (row, col)
        inc_output_np = inc_output[0].cpu().numpy()  # (hidden_dim, 30, 30)
        H, W = inc_output_np.shape[1], inc_output_np.shape[2]

        # Pad the feature map to handle edge cases
        inc_output_padded = np.pad(inc_output_np, ((0, 0), (pad_h, pad_h), (pad_w, pad_w)), mode='constant')

        # Extract the patch (accounting for the padding offset)
        feature_patch = inc_output_padded[:, row:row + kH, col:col + kW]  # (hidden_dim, kH, kW)

        # Compute logits manually for this pixel using convolution formula
        # logits[c] = sum over (c_in, kh, kw) of weight[c, c_in, kh, kw] * feature_patch[c_in, kh, kw] + bias[c]
        logits_manual = np.einsum('oihw,ihw->o', outc_weight, feature_patch) + outc_bias  # (num_classes,)

        # For feature_vector, we'll use the center of the patch for display purposes
        feature_vector = inc_output_np[:, row, col]  # (hidden_dim,)
        hidden_dim = feature_vector.shape[0]

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
            'combined_input': combined.tolist(),
            'encoding_type': 'onehot' if model.use_onehot else 'learned',
            'embed_dim': 11 if model.use_onehot else 16
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
            # For 3x3 kernels, we sum contributions across spatial dimensions
            'contributions': np.einsum('oihw,ihw->oi', outc_weight, feature_patch).tolist(),  # (num_classes, hidden_dim)
        },
        'output_layer': {
            'weights_shape': list(outc_weight.shape),
            'kernel_size': [kH, kW],
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
        weight_tensor = outc_weight[c]  # (hidden_dim, kH, kW)
        # Compute the weighted sum across the feature patch
        weighted_sum = float(np.sum(weight_tensor * feature_patch))
        bias_val = float(outc_bias[c])
        logit = weighted_sum + bias_val

        result['output_layer']['per_class_calculation'].append({
            'color': c,
            'color_name': result['prediction']['color_names'][c],
            'weight_shape': list(weight_tensor.shape),
            'weighted_sum': weighted_sum,
            'bias': bias_val,
            'logit': logit,
            'probability': float(probs[c])
        })

    # Add attention visualization if model has attention
    # Cross-attention: output pixels attend to INPUT pixels (more useful for ARC)
    if getattr(model, 'use_cross_attention', False) and model.cross_attention is not None:
        # Enable cross-attention capture and run inference again
        model.cross_attention.capture_attention = True
        with torch.no_grad():
            _ = model(inp_tensor, candidate_tensor)

        attn_weights = model.cross_attention.last_attention_weights  # (B, num_heads, H*W, H*W)
        model.cross_attention.capture_attention = False
        num_heads = model.cross_attention.num_heads

        if attn_weights is not None:
            # attn_weights shape: (B, num_heads, N, N) where N = H*W
            attn_np = attn_weights.squeeze(0).numpy()  # (num_heads, 900, 900)
            H, W = GRID_SIZE, GRID_SIZE
            pixel_idx = row * W + col

            # For multi-head: compute per-head attention and averaged attention
            if num_heads > 1:
                # Per-head attention for this pixel
                per_head_from = []  # (num_heads, H, W)
                per_head_to = []    # (num_heads, H, W)
                per_head_stats = []
                per_head_top_attended = []

                for h in range(num_heads):
                    head_attn = attn_np[h]  # (N, N)
                    # Attention FROM this output pixel to all INPUT pixels
                    head_from_pixel = head_attn[pixel_idx, :]  # (900,)
                    head_from_grid = head_from_pixel.reshape(H, W)
                    per_head_from.append(head_from_grid.tolist())

                    # Attention TO this pixel position from all OUTPUT pixels
                    head_to_pixel = head_attn[:, pixel_idx]  # (900,)
                    head_to_grid = head_to_pixel.reshape(H, W)
                    per_head_to.append(head_to_grid.tolist())

                    # Top INPUT pixels this OUTPUT pixel attends to (per head)
                    top_from_indices = np.argsort(head_from_pixel)[-5:][::-1]
                    per_head_top_attended.append([
                        {
                            'row': int(idx // W),
                            'col': int(idx % W),
                            'weight': float(head_from_pixel[idx])
                        }
                        for idx in top_from_indices
                    ])

                    # Entropy only valid with softmax (normalized positive values)
                    no_softmax = getattr(model.cross_attention, 'no_softmax', False)
                    if no_softmax:
                        from_entropy = None
                    else:
                        from_entropy = float(-np.sum(head_from_pixel * np.log(head_from_pixel + 1e-10)))

                    per_head_stats.append({
                        'from_max': float(head_from_pixel.max()),
                        'from_mean': float(head_from_pixel.mean()),
                        'from_entropy': from_entropy,
                        'to_max': float(head_to_pixel.max()),
                        'to_mean': float(head_to_pixel.mean()),
                        'self_attention_weight': float(head_from_pixel[pixel_idx]),
                    })

                # Average attention across heads for summary view
                attn_avg = attn_np.mean(axis=0)  # (N, N)
            else:
                attn_avg = attn_np[0]  # (N, N) - single head
                per_head_from = None
                per_head_to = None
                per_head_stats = None
                per_head_top_attended = None

            # Compute averaged attention metrics
            attn_from_pixel = attn_avg[pixel_idx, :]  # (900,)
            attn_from_grid = attn_from_pixel.reshape(H, W)
            attn_to_pixel = attn_avg[:, pixel_idx]  # (900,)
            attn_to_grid = attn_to_pixel.reshape(H, W)

            # Top INPUT pixels this OUTPUT pixel attends to (averaged)
            top_from_indices = np.argsort(attn_from_pixel)[-10:][::-1]
            top_attended = [
                {
                    'row': int(idx // W),
                    'col': int(idx % W),
                    'weight': float(attn_from_pixel[idx])
                }
                for idx in top_from_indices
            ]

            # Top OUTPUT pixels that attend to this INPUT position (averaged)
            top_to_indices = np.argsort(attn_to_pixel)[-10:][::-1]
            top_attending = [
                {
                    'row': int(idx // W),
                    'col': int(idx % W),
                    'weight': float(attn_to_pixel[idx])
                }
                for idx in top_to_indices
            ]

            # Check if no_softmax is enabled
            no_softmax = getattr(model.cross_attention, 'no_softmax', False)
            if no_softmax:
                from_entropy = None
            else:
                from_entropy = float(-np.sum(attn_from_pixel * np.log(attn_from_pixel + 1e-10)))

            result['attention'] = {
                'has_attention': True,
                'attention_type': 'cross',  # Distinguish from self-attention
                'num_heads': num_heads,
                'no_softmax': no_softmax,
                'attention_from_pixel': attn_from_grid.tolist(),  # Averaged: this output pixel attends to these input pixels
                'attention_to_pixel': attn_to_grid.tolist(),      # Averaged: these output pixels attend to this input position
                'top_attended': top_attended,   # Top 10 input pixels this output attends to (averaged)
                'top_attending': top_attending, # Top 10 output pixels that attend to this input position (averaged)
                'self_attention_weight': float(attn_from_pixel[pixel_idx]),  # Attention to same position in input (averaged)
                'stats': {
                    'from_max': float(attn_from_pixel.max()),
                    'from_min': float(attn_from_pixel.min()),  # Useful for no_softmax case
                    'from_mean': float(attn_from_pixel.mean()),
                    'from_entropy': from_entropy,
                    'to_max': float(attn_to_pixel.max()),
                    'to_mean': float(attn_to_pixel.mean()),
                }
            }

            # Add per-head data if multi-head
            if num_heads > 1:
                result['attention']['per_head'] = {
                    'attention_from_pixel': per_head_from,  # List of (H, W) grids per head
                    'attention_to_pixel': per_head_to,      # List of (H, W) grids per head
                    'top_attended': per_head_top_attended,  # List of top-5 per head
                    'stats': per_head_stats,                # List of stats per head
                }
        else:
            result['attention'] = {'has_attention': False}

    # Self-attention: pixels attend to each other after CNN processing
    elif getattr(model, 'use_attention', False) and model.attention is not None:
        # Enable attention capture and run inference again
        model.attention.capture_attention = True
        with torch.no_grad():
            _ = model(inp_tensor, candidate_tensor)

        attn_weights = model.attention.last_attention_weights  # (1, H*W, H*W)
        model.attention.capture_attention = False

        if attn_weights is not None:
            attn_np = attn_weights.squeeze(0).numpy()  # (900, 900)
            H, W = GRID_SIZE, GRID_SIZE
            pixel_idx = row * W + col

            # Attention FROM this pixel to all others
            attn_from_pixel = attn_np[pixel_idx, :]  # (900,)
            attn_from_grid = attn_from_pixel.reshape(H, W)

            # Attention TO this pixel from all others
            attn_to_pixel = attn_np[:, pixel_idx]  # (900,)
            attn_to_grid = attn_to_pixel.reshape(H, W)

            # Top pixels this pixel attends to
            top_from_indices = np.argsort(attn_from_pixel)[-10:][::-1]
            top_attended = [
                {
                    'row': int(idx // W),
                    'col': int(idx % W),
                    'weight': float(attn_from_pixel[idx])
                }
                for idx in top_from_indices
            ]

            # Top pixels that attend to this pixel
            top_to_indices = np.argsort(attn_to_pixel)[-10:][::-1]
            top_attending = [
                {
                    'row': int(idx // W),
                    'col': int(idx % W),
                    'weight': float(attn_to_pixel[idx])
                }
                for idx in top_to_indices
            ]

            result['attention'] = {
                'has_attention': True,
                'attention_type': 'self',  # Distinguish from cross-attention
                'attention_from_pixel': attn_from_grid.tolist(),  # This pixel attends to...
                'attention_to_pixel': attn_to_grid.tolist(),      # Others attend to this pixel...
                'top_attended': top_attended,   # Top 10 pixels this one attends to
                'top_attending': top_attending, # Top 10 pixels that attend to this one
                'self_attention_weight': float(attn_from_pixel[pixel_idx]),  # Attention to self
                'stats': {
                    'from_max': float(attn_from_pixel.max()),
                    'from_mean': float(attn_from_pixel.mean()),
                    'from_entropy': float(-np.sum(attn_from_pixel * np.log(attn_from_pixel + 1e-10))),
                    'to_max': float(attn_to_pixel.max()),
                    'to_mean': float(attn_to_pixel.mean()),
                }
            }
        else:
            result['attention'] = {'has_attention': False}
    else:
        result['attention'] = {'has_attention': False}

    # Per-layer cross-attention visualization (NConvWithCrossAttention blocks)
    cross_attention_per_layer = getattr(model, 'cross_attention_per_layer', False)
    use_per_layer_slot = getattr(model, 'use_per_layer_slot', False)
    if cross_attention_per_layer or use_per_layer_slot:
        # Collect all NConvWithCrossAttention blocks
        per_layer_blocks = []
        block_names = []

        # Check inc block
        if hasattr(model.inc, 'capture_attention'):
            per_layer_blocks.append(model.inc)
            block_names.append('inc')

        # Check down blocks
        for i in range(1, 4):
            down_name = f'down{i}'
            if hasattr(model, down_name):
                down_block = getattr(model, down_name)
                if hasattr(down_block, 'conv') and hasattr(down_block.conv, 'capture_attention'):
                    per_layer_blocks.append(down_block.conv)
                    block_names.append(down_name)

        # Check up blocks
        for i in range(1, 4):
            up_name = f'up{i}'
            if hasattr(model, up_name):
                up_block = getattr(model, up_name)
                if hasattr(up_block, 'conv') and hasattr(up_block.conv, 'capture_attention'):
                    per_layer_blocks.append(up_block.conv)
                    block_names.append(up_name)

        if per_layer_blocks:
            # Enable attention capture on all blocks
            for block in per_layer_blocks:
                block.capture_attention = True

            # Run inference to capture attention
            with torch.no_grad():
                _ = model(inp_tensor, candidate_tensor)

            # Collect attention weights from all blocks
            per_layer_attention = []
            for block, block_name in zip(per_layer_blocks, block_names):
                if block.last_attention_weights is not None:
                    for layer_attn in block.last_attention_weights:
                        layer_idx = layer_attn['layer_idx']
                        attn_weights = layer_attn['attention_weights']
                        spatial_size = layer_attn['spatial_size']
                        is_slot = layer_attn['is_slot_attention']

                        # Process attention weights for visualization
                        H, W = spatial_size
                        pixel_idx = row * W + col if row < H and col < W else None

                        if attn_weights is not None and pixel_idx is not None:
                            # Handle different attention weight shapes
                            if isinstance(attn_weights, torch.Tensor):
                                attn_np = attn_weights.squeeze(0).cpu().numpy()
                            else:
                                attn_np = attn_weights

                            # For cross-attention: (num_heads, N_q, N_kv) or (N_q, N_kv)
                            if len(attn_np.shape) == 3:
                                # Multi-head: average across heads for display
                                attn_avg = attn_np.mean(axis=0)  # (N_q, N_kv)
                                num_heads = attn_np.shape[0]
                            elif len(attn_np.shape) == 2:
                                attn_avg = attn_np
                                num_heads = 1
                            else:
                                continue

                            N_q = attn_avg.shape[0]
                            N_kv = attn_avg.shape[1]

                            # Get attention from this pixel to all key-value positions
                            if pixel_idx < N_q:
                                attn_from_pixel = attn_avg[pixel_idx, :]  # (N_kv,)

                                # Reshape to grid (assuming square or use original IR size)
                                H_kv = int(np.sqrt(N_kv))
                                W_kv = N_kv // H_kv if H_kv > 0 else 1
                                attn_from_grid = attn_from_pixel.reshape(H_kv, W_kv)

                                # Top attended positions
                                top_indices = np.argsort(attn_from_pixel)[-5:][::-1]
                                top_attended = [
                                    {
                                        'row': int(idx // W_kv),
                                        'col': int(idx % W_kv),
                                        'weight': float(attn_from_pixel[idx])
                                    }
                                    for idx in top_indices
                                ]

                                per_layer_attention.append({
                                    'block_name': block_name,
                                    'layer_idx': layer_idx,
                                    'display_name': f'{block_name}_conv{layer_idx}',
                                    'is_slot_attention': is_slot,
                                    'spatial_size': list(spatial_size),
                                    'kv_size': [H_kv, W_kv],
                                    'num_heads': num_heads,
                                    'attention_from_pixel': attn_from_grid.tolist(),
                                    'top_attended': top_attended,
                                    'stats': {
                                        'max': float(attn_from_pixel.max()),
                                        'min': float(attn_from_pixel.min()),
                                        'mean': float(attn_from_pixel.mean()),
                                    }
                                })

                # Disable capture after grabbing weights
                block.capture_attention = False

            result['per_layer_attention'] = {
                'has_per_layer_attention': True,
                'is_slot_attention': use_per_layer_slot,
                'shared': getattr(model, 'cross_attention_shared', False),
                'num_blocks': len(per_layer_blocks),
                'block_names': block_names,
                'conv_depth': model.conv_depth,
                'layers': per_layer_attention
            }
    else:
        result['per_layer_attention'] = {'has_per_layer_attention': False}

    # Slot-routed cross-attention visualization
    if getattr(model, 'use_slot_cross_attention', False) and model.slot_attention is not None:
        with torch.no_grad():
            # Compute IR features
            inp_onehot_slot = F.one_hot(inp_tensor.long(), NUM_COLORS).float()
            inp_onehot_slot = inp_onehot_slot.permute(0, 3, 1, 2).contiguous()  # (B, 10, H, W)
            ir_features_slot = model.ir_encoder(inp_onehot_slot)  # (B, H, W, D)

            # Get slot embeddings and masks
            slot_embeddings, slot_masks = model.slot_attention(ir_features_slot, color_grid=inp_tensor)
            # slot_embeddings: (B, K, slot_dim)
            # slot_masks: (B, K, H, W) - soft assignment of input pixels to slots

            slot_masks_np = slot_masks.squeeze(0).cpu().numpy()  # (K, H, W)
            slot_embeddings_np = slot_embeddings.squeeze(0).cpu().numpy()  # (K, slot_dim)
            K = slot_masks_np.shape[0]
            H_in, W_in = slot_masks_np.shape[1], slot_masks_np.shape[2]

            # Get decoder features (need to run partial forward)
            # Re-run the U-Net to get decoder output features
            if model.use_onehot:
                out_emb_slot = model._encode_grid(candidate_tensor)
            else:
                out_emb_slot = model.output_embed(candidate_tensor)

            if model.cross_attention is not None and model.ir_encoder is not None:
                ir_features_attn = ir_features_slot.permute(0, 3, 1, 2).contiguous()
                ir_features_attn = model.ir_self_attention(ir_features_attn)
                ir_features_attn = ir_features_attn.permute(0, 2, 3, 1).contiguous()
                out_emb_slot = model.cross_attention(query=out_emb_slot, key_value=ir_features_attn)

            if model.use_onehot:
                inp_emb_slot = model._encode_grid(inp_tensor)
            else:
                inp_emb_slot = model.input_embed(inp_tensor)

            x = torch.cat([inp_emb_slot, out_emb_slot], dim=-1)
            x = x.permute(0, 3, 1, 2).contiguous()

            # U-Net encoder
            x1 = model.inc(x)
            if model.num_layers >= 1:
                x2 = model.down1(x1)
            if model.num_layers >= 2:
                x3 = model.down2(x2)
            if model.num_layers >= 3:
                x4 = model.down3(x3)

            # Determine bottleneck
            if model.num_layers == 0:
                bottleneck = x1
            elif model.num_layers == 1:
                bottleneck = x2
            elif model.num_layers == 2:
                bottleneck = x3
            else:
                bottleneck = x4

            # U-Net decoder
            if model.num_layers == 0:
                decoder_out = bottleneck
            elif model.no_skip:
                if model.num_layers == 3:
                    decoder_out = model.up1(bottleneck, target_size=(x3.size(2), x3.size(3)))
                    decoder_out = model.up2(decoder_out, target_size=(x2.size(2), x2.size(3)))
                    decoder_out = model.up3(decoder_out, target_size=(x1.size(2), x1.size(3)))
                elif model.num_layers == 2:
                    decoder_out = model.up2(bottleneck, target_size=(x2.size(2), x2.size(3)))
                    decoder_out = model.up3(decoder_out, target_size=(x1.size(2), x1.size(3)))
                else:
                    decoder_out = model.up3(bottleneck, target_size=(x1.size(2), x1.size(3)))
            else:
                if model.num_layers == 3:
                    decoder_out = model.up1(bottleneck, x3)
                    decoder_out = model.up2(decoder_out, x2)
                    decoder_out = model.up3(decoder_out, x1)
                elif model.num_layers == 2:
                    decoder_out = model.up2(bottleneck, x2)
                    decoder_out = model.up3(decoder_out, x1)
                else:
                    decoder_out = model.up3(bottleneck, x1)

            # Apply spatial self-attention if enabled
            if model.attention is not None:
                decoder_out = model.attention(decoder_out)

            # Now compute slot-routed cross-attention manually to capture intermediate values
            decoder_features = decoder_out.permute(0, 2, 3, 1).contiguous()  # (B, H, W, C)
            B, H_out, W_out, C = decoder_features.shape
            N_out = H_out * W_out
            N_in = H_in * W_in

            # Flatten IR features for attention computation
            ir_flat = ir_features_slot.reshape(B, N_in, -1)  # (B, N_in, D)

            # Normalize slot masks (sum to 1 over spatial dims)
            masks_normalized = slot_masks / (slot_masks.sum(dim=(-2, -1), keepdim=True) + 1e-8)
            masks_flat = masks_normalized.reshape(B, K, N_in)  # (B, K, N_in)

            # Step 1: Compute slot content by mean-pooling IR features within each slot
            # This is what the actual SlotRoutedCrossAttention does
            slot_content = torch.bmm(masks_flat, ir_flat)  # (B, K, D)

            # Create binary masks for hard slot boundaries
            masks_binary = (slot_masks > 0.1).float()  # (B, K, H_in, W_in)
            masks_binary_flat = masks_binary.reshape(B, K, N_in)  # (B, K, N_in)

            # Detect empty slots (no pixels belong to this slot)
            slot_pixel_counts = masks_binary.sum(dim=(-2, -1))  # (B, K)
            empty_slots = (slot_pixel_counts == 0)  # (B, K)

            # Compute slot attention weights using the correct projections
            q = model.slot_cross_attention.q_proj(decoder_features.reshape(B, N_out, C))  # (B, N_out, proj_dim)
            k_slots = model.slot_cross_attention.k_slot_proj(slot_content)  # (B, K, proj_dim)
            scale = model.slot_cross_attention.scale

            slot_attn_logits = torch.bmm(q, k_slots.transpose(1, 2)) * scale  # (B, N_out, K)

            # Mask out empty slots so they're never selected
            slot_attn_logits = slot_attn_logits.masked_fill(empty_slots.unsqueeze(1), float('-inf'))

            # Top-k slot selection (matching forward pass)
            top_k_value = getattr(model.slot_cross_attention, 'top_k', 2)
            num_valid_slots = (~empty_slots).sum(dim=-1).min().item()
            top_k = min(top_k_value, K, max(1, num_valid_slots))
            topk_logits, topk_indices = torch.topk(slot_attn_logits, top_k, dim=-1)  # (B, N_out, top_k)

            # Softmax over only the top-k slots (renormalized)
            topk_attn = F.softmax(topk_logits, dim=-1)  # (B, N_out, top_k)
            topk_attn = torch.nan_to_num(topk_attn, nan=1.0 / top_k)

            # For visualization, expand back to full K slots (with zeros for non-top-k)
            slot_attn = torch.zeros(B, N_out, K, device=slot_attn_logits.device)
            slot_attn.scatter_(2, topk_indices, topk_attn)
            slot_attn_np = slot_attn.squeeze(0).cpu().numpy()  # (N_out, K)

            # Get slot attention for the selected pixel
            pixel_idx = row * W_out + col
            slot_attn_for_pixel = slot_attn_np[pixel_idx, :]  # (K,)

            # Compute per-pixel attention within slots (matching forward pass)
            k_pixels = model.slot_cross_attention.k_pixel_proj(ir_flat)  # (B, N_in, proj_dim)
            pixel_logits = torch.bmm(q, k_pixels.transpose(1, 2)) * scale  # (B, N_out, N_in)

            # Gather top-k masks
            topk_indices_expanded = topk_indices.unsqueeze(-1).expand(-1, -1, -1, N_in)  # (B, N_out, top_k, N_in)
            masks_expanded_full = masks_binary_flat.unsqueeze(1).expand(-1, N_out, -1, -1)  # (B, N_out, K, N_in)
            topk_masks = torch.gather(masks_expanded_full, 2, topk_indices_expanded)  # (B, N_out, top_k, N_in)

            # Compute per-slot pixel attention (only for top-k slots)
            pixel_logits_expanded = pixel_logits.unsqueeze(2)  # (B, N_out, 1, N_in)
            masked_logits = pixel_logits_expanded.masked_fill(topk_masks == 0, float('-inf'))  # (B, N_out, top_k, N_in)
            topk_per_slot_pixel_attn = F.softmax(masked_logits, dim=-1)  # (B, N_out, top_k, N_in)
            topk_per_slot_pixel_attn = torch.nan_to_num(topk_per_slot_pixel_attn, nan=0.0)

            # Expand back to full K slots for visualization
            per_slot_pixel_attn = torch.zeros(B, N_out, K, N_in, device=pixel_logits.device)
            topk_indices_for_scatter = topk_indices.unsqueeze(-1).expand(-1, -1, -1, N_in)
            per_slot_pixel_attn.scatter_(2, topk_indices_for_scatter, topk_per_slot_pixel_attn)

            # Combine slot selection with per-slot pixel attention (matching forward pass)
            combined_attn = (topk_attn.unsqueeze(-1) * topk_per_slot_pixel_attn).sum(dim=2)  # (B, N_out, N_in)
            combined_attn_np = combined_attn.squeeze(0).cpu().numpy()  # (N_out, N_in)

            # Get combined attention for the selected pixel
            combined_attn_for_pixel = combined_attn_np[pixel_idx, :]  # (N_in,)
            combined_attn_grid = combined_attn_for_pixel.reshape(H_in, W_in)

            # Top input pixels this output pixel attends to (via slot routing)
            top_input_indices = np.argsort(combined_attn_for_pixel)[-10:][::-1]
            top_attended_inputs = [
                {
                    'row': int(idx // W_in),
                    'col': int(idx % W_in),
                    'weight': float(combined_attn_for_pixel[idx])
                }
                for idx in top_input_indices
            ]

            # Per-slot pixel attention for this output pixel (shows attention within each slot)
            per_slot_pixel_attn_for_pixel = per_slot_pixel_attn.squeeze(0)[pixel_idx].cpu().numpy()  # (K, N_in)
            per_slot_pixel_attn_grids = per_slot_pixel_attn_for_pixel.reshape(K, H_in, W_in)  # (K, H_in, W_in)

            # Get which slots are empty and which are in top-k for this pixel
            empty_slots_np = empty_slots.squeeze(0).cpu().numpy()  # (K,)
            topk_indices_for_pixel = topk_indices.squeeze(0)[pixel_idx].cpu().numpy()  # (top_k,)

            # Determine slot type
            slot_type = 'affinity' if getattr(model, 'use_affinity_slots', False) else ('color' if getattr(model, 'use_color_slots', False) else 'learned')

            result['slot_attention'] = {
                'has_slot_attention': True,
                'slot_type': slot_type,
                'num_slots': K,
                'top_k': top_k,
                'top_k_indices': topk_indices_for_pixel.tolist(),  # Which slots are in top-k for this pixel
                'empty_slots': empty_slots_np.tolist(),  # (K,) - True if slot has no pixels
                'num_valid_slots': int(num_valid_slots),
                'slot_dim': int(slot_embeddings_np.shape[1]),
                'slot_masks': slot_masks_np.tolist(),  # (K, H, W)
                'slot_masks_normalized': masks_normalized.squeeze(0).cpu().numpy().tolist(),
                'slot_attention_weights': slot_attn_for_pixel.tolist(),  # (K,) - which slots this pixel attends to (0 for non-top-k)
                'per_slot_pixel_attention': per_slot_pixel_attn_grids.tolist(),  # (K, H_in, W_in) - per-slot pixel attention
                'combined_attention': combined_attn_grid.tolist(),  # (H_in, W_in) - final attention over input
                'top_attended_inputs': top_attended_inputs,
                'slot_embeddings_norm': [float(np.linalg.norm(slot_embeddings_np[k])) for k in range(K)],
                'slot_content_norm': [float(np.linalg.norm(slot_content[0, k].cpu().numpy())) for k in range(K)],
                'stats': {
                    'slot_attn_max': float(slot_attn_for_pixel.max()),
                    'slot_attn_entropy': float(-np.sum(slot_attn_for_pixel * np.log(slot_attn_for_pixel + 1e-10))),
                    'combined_attn_max': float(combined_attn_for_pixel.max()),
                    'combined_attn_entropy': float(-np.sum(combined_attn_for_pixel * np.log(combined_attn_for_pixel + 1e-10))),
                }
            }
    else:
        result['slot_attention'] = {'has_slot_attention': False}

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
            'conv_depth': model.conv_depth,
            'kernel_size': model.kernel_size,
            'force_comparison': getattr(model, 'force_comparison', False),
            'use_onehot': model.use_onehot,
            'use_attention': getattr(model, 'use_attention', False),
            'use_cross_attention': getattr(model, 'use_cross_attention', False),
            'cross_attention_no_softmax': getattr(model.cross_attention, 'no_softmax', False) if model.cross_attention is not None else False,
            'num_classes': model.num_classes
        }
        return jsonify(result)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


# =============================================================================
# Attention Visualization
# =============================================================================

def capture_attention_weights(
    input_grid: np.ndarray,
    output_grid: np.ndarray,
    selected_pixel: Optional[Tuple[int, int]] = None
) -> Dict:
    """
    Capture attention weights for visualization.

    Args:
        input_grid: Input puzzle grid (H, W) with values 0-9
        output_grid: Expected output puzzle grid (H, W) with values 0-9
        selected_pixel: Optional (row, col) to get attention FROM this pixel to all others

    Returns:
        Dict with attention weights and metadata
    """
    if not getattr(model, 'use_attention', False) or model.attention is None:
        return {'error': 'Model does not have attention enabled', 'has_attention': False}

    # Enable attention capture
    model.attention.capture_attention = True

    # Pad grids and convert to tensors
    inp_pad = pad_to_grid_size(input_grid)
    out_pad = pad_to_grid_size(output_grid)

    inp_tensor = torch.from_numpy(inp_pad).long().unsqueeze(0).to(device)
    # Use zeros as candidate (standard test mode)
    candidate_tensor = torch.zeros_like(inp_tensor).to(device)

    # Run inference to capture attention
    with torch.no_grad():
        logits = model(inp_tensor, candidate_tensor)
        prediction = logits.argmax(dim=1).squeeze(0).cpu().numpy()

    # Get attention weights
    attn_weights = model.attention.last_attention_weights  # (1, H*W, H*W)
    model.attention.capture_attention = False

    if attn_weights is None:
        return {'error': 'Failed to capture attention weights', 'has_attention': True}

    attn_np = attn_weights.squeeze(0).numpy()  # (900, 900) for 30x30 grid
    H, W = GRID_SIZE, GRID_SIZE
    N = H * W  # 900

    result = {
        'has_attention': True,
        'grid_size': [H, W],
        'num_pixels': N,
        'input_grid': inp_pad.tolist(),
        'output_grid': out_pad.tolist(),
        'prediction': prediction.tolist(),
        'input_shape': list(input_grid.shape),
        'output_shape': list(output_grid.shape),
    }

    # If a specific pixel is selected, return attention FROM that pixel
    if selected_pixel is not None:
        row, col = selected_pixel
        if 0 <= row < H and 0 <= col < W:
            pixel_idx = row * W + col
            # Attention from selected pixel to all other pixels
            attn_from_pixel = attn_np[pixel_idx, :]  # (900,)
            # Reshape to grid for visualization
            attn_grid = attn_from_pixel.reshape(H, W)
            result['selected_pixel'] = {'row': row, 'col': col}
            result['attention_from_pixel'] = attn_grid.tolist()
            # Also include which pixels this pixel attends to most (top 10)
            top_indices = np.argsort(attn_from_pixel)[-10:][::-1]
            result['top_attended'] = [
                {
                    'row': int(idx // W),
                    'col': int(idx % W),
                    'weight': float(attn_from_pixel[idx])
                }
                for idx in top_indices
            ]
    else:
        # Return summary statistics (full matrix is too large: 900x900)
        # Compute mean attention per pixel (how much each pixel attends on average)
        mean_attn_from = attn_np.mean(axis=1).reshape(H, W)  # avg attention FROM each pixel
        mean_attn_to = attn_np.mean(axis=0).reshape(H, W)    # avg attention TO each pixel
        max_attn_from = attn_np.max(axis=1).reshape(H, W)    # max attention FROM each pixel
        max_attn_to = attn_np.max(axis=0).reshape(H, W)      # max attention TO each pixel

        result['attention_summary'] = {
            'mean_from': mean_attn_from.tolist(),
            'mean_to': mean_attn_to.tolist(),
            'max_from': max_attn_from.tolist(),
            'max_to': max_attn_to.tolist(),
        }

        # Entropy of attention distribution (high entropy = uniform, low = focused)
        entropy = -np.sum(attn_np * np.log(attn_np + 1e-10), axis=1).reshape(H, W)
        result['attention_summary']['entropy'] = entropy.tolist()

    return result


@app.route('/api/attention/<puzzle_id>/<int:example_idx>')
def api_attention(puzzle_id: str, example_idx: int):
    """
    Get attention weights visualization for a specific puzzle example.

    Query params:
        row, col: Optional pixel coordinates to show attention FROM that pixel
    """
    if model is None or puzzles is None:
        return jsonify({'error': 'Model or puzzles not loaded'}), 500

    if not getattr(model, 'use_attention', False):
        return jsonify({
            'error': 'Model does not have attention enabled',
            'has_attention': False
        }), 400

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

    # Check for selected pixel in query params
    selected_pixel = None
    row = request.args.get('row', type=int)
    col = request.args.get('col', type=int)
    if row is not None and col is not None:
        selected_pixel = (row, col)

    try:
        result = capture_attention_weights(input_grid, output_grid, selected_pixel)
        result['puzzle_id'] = puzzle_id
        result['example_idx'] = example_idx
        result['split'] = split
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
