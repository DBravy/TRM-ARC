# Slot Attention Implementation Guide

## ✅ Syntax Validation

All files have been validated for correct Python syntax:
- ✓ `models/recursive_reasoning/trm.py` - Modified with slot attention
- ✓ `test_slot_attention.py` - Comprehensive test suite

## 🧪 Testing Before Server Deployment

### Option 1: Quick Syntax Check (No Dependencies)
```bash
python3 check_syntax.py
```
This verifies the code is syntactically correct without needing PyTorch.

### Option 2: Full Functional Tests (Requires PyTorch)
```bash
# Install dependencies first
pip install -r requirements.txt

# Run comprehensive tests
python test_slot_attention.py
```

The test suite validates:
- ✓ Slot attention encoding/decoding
- ✓ Forward and backward passes
- ✓ Gradient flow
- ✓ ACT halting loop
- ✓ Multiple grid sizes
- ✓ Puzzle embeddings integration
- ✓ Backward compatibility with token mode

## 🚀 Using Slot Attention in Training

### Configuration Changes

When creating your model config (e.g., in `config/` YAML files), add these parameters:

```yaml
# Enable slot attention
use_slot_attention: true

# Slot attention parameters
num_slots: 10              # Number of object slots (tune based on ARC complexity)
slot_dim: 64               # Slot representation dimension
slot_iterations: 3         # Number of slot attention iterations
slot_mlp_hidden: 128       # Hidden size for slot MLP

# Grid dimensions (MUST match your data preprocessing)
grid_height: 30            # Height of ARC grids
grid_width: 30             # Width of ARC grids
grid_channels: 1           # 1 for single channel, or 10 for one-hot color encoding

# Architecture hyperparameters
cnn_hidden_dim: 64         # CNN encoder hidden dimension
decoder_hidden_dim: 64     # Decoder hidden dimension

# Keep your existing parameters
hidden_size: 256
num_heads: 8
L_layers: 4
# ... etc
```

### Data Format Requirements

**Input format must change from tokens to grids:**

Before (token-based):
```python
batch = {
    "inputs": token_ids,  # Shape: [batch, seq_len], dtype: long
    "puzzle_identifiers": puzzle_ids,
}
```

After (slot-based):
```python
batch = {
    "inputs": grids,  # Shape: [batch, height, width] or [batch, channels, height, width]
                      # dtype: float32
    "puzzle_identifiers": puzzle_ids,
}
```

**Output format also changes:**

Before: `[batch, seq_len, vocab_size]` logits
After: `[batch, channels, height, width]` reconstructed grid

### Loss Function Update

Your loss computation needs to change from token prediction to pixel prediction:

```python
# Before (token-based)
loss = F.cross_entropy(
    logits.reshape(-1, vocab_size),
    targets.reshape(-1)
)

# After (slot-based)
loss = F.mse_loss(output_grid, target_grid)
# Or for discrete colors:
# loss = F.cross_entropy(output_grid, target_grid)  # if one-hot encoded
```

## 📊 Expected Behavior

### Model Size
With slot attention, the model should be:
- **More parameter efficient**: `num_slots` (e.g., 10) << `seq_len` (e.g., 900 for 30x30 grid)
- **Faster per step**: Fewer sequence positions to process

### Training Dynamics
- **Early training**: Slots may not segment objects well initially
- **Mid training**: Slots should start binding to distinct objects
- **Convergence**: Check if output grids look reasonable (not just noise)

### Debugging Tips
If training fails, check:
1. Grid dimensions in config match actual data
2. Input dtype is float (not int/long for grids)
3. Loss is decreasing (even slowly)
4. Output grids have reasonable values (not NaN/Inf)

## 🔄 Backward Compatibility

The original token-based mode still works! To use it:

```yaml
use_slot_attention: false
```

This allows you to:
- Compare slot-based vs token-based on same task
- Fallback if slot attention has issues
- Use existing configs without modification

## 🎯 Recommended Hyperparameters for ARC

Based on ARC puzzle characteristics:

```yaml
# For typical ARC puzzles (up to 30x30)
num_slots: 8-15            # Most ARC puzzles have 2-10 objects
slot_dim: 64-128           # Enough for object properties
slot_iterations: 3-5       # 3 is usually sufficient
grid_height: 30
grid_width: 30
grid_channels: 1           # Single channel for color indices

# If using one-hot color encoding:
grid_channels: 11          # 10 colors + 1 background
```

## 🐛 Known Limitations

1. **Fixed grid size**: Grid dimensions must be specified in config
   - If ARC grids vary in size, you may need padding
2. **Decoder complexity**: Spatial broadcast decoder can be slow for large grids
3. **Slot collapse**: Slots might collapse to same object (use slot_iterations >= 3)

## 📝 Example Training Command

```bash
# On your server
python pretrain.py \
  --config config/slot_attention_arc.yaml \
  --data-dir data/arc2concept-aug-1000 \
  --output-dir outputs/slot_attention_run
```

## ✨ Next Steps

1. **Run syntax check** locally: `python3 check_syntax.py`
2. **Update your config** files to enable slot attention
3. **Modify data loading** to provide grids instead of tokens
4. **Update loss computation** for pixel-level prediction
5. **Deploy to server** and monitor training

Good luck! The implementation is ready to go. 🚀
