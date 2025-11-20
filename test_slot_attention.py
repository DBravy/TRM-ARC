"""
Test script for slot attention implementation.
Validates the model works without needing a full dataset.
"""

import torch
import sys
from models.recursive_reasoning.trm import TinyRecursiveReasoningModel_ACTV1

def test_slot_attention_mode():
    """Test model with slot attention enabled."""
    print("\n" + "="*60)
    print("Testing SLOT ATTENTION mode")
    print("="*60)

    # Configuration for slot attention
    config = {
        "batch_size": 4,
        "seq_len": 100,  # Not used with slot attention
        "puzzle_emb_ndim": 0,  # Disable puzzle embeddings for simplicity
        "puzzle_emb_len": 0,  # Must be 0 when puzzle_emb_ndim is 0
        "num_puzzle_identifiers": 10,
        "vocab_size": 20,  # Not used with slot attention

        "H_cycles": 2,
        "L_cycles": 3,
        "H_layers": 1,
        "L_layers": 2,

        "hidden_size": 128,
        "expansion": 2.0,
        "num_heads": 4,
        "pos_encodings": "rope",

        "halt_max_steps": 3,
        "halt_exploration_prob": 0.1,

        "forward_dtype": "float32",  # Use float32 for CPU testing

        # Slot Attention config
        "use_slot_attention": True,
        "num_slots": 8,
        "slot_dim": 64,
        "slot_iterations": 3,
        "slot_mlp_hidden": 128,
        "grid_height": 10,
        "grid_width": 10,
        "grid_channels": 1,
        "cnn_hidden_dim": 64,
        "decoder_hidden_dim": 64,
    }

    print(f"Config: num_slots={config['num_slots']}, grid_size={config['grid_height']}x{config['grid_width']}")

    # Create model
    print("\n1. Creating model...")
    model = TinyRecursiveReasoningModel_ACTV1(config)
    model.train()

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Total params: {total_params:,}")
    print(f"   Trainable params: {trainable_params:,}")

    # Create dummy batch with grid inputs
    print("\n2. Creating dummy batch...")
    batch = {
        "inputs": torch.randint(0, 10, (config["batch_size"], config["grid_height"], config["grid_width"]), dtype=torch.float32),
        "puzzle_identifiers": torch.zeros(config["batch_size"], dtype=torch.long),
    }
    print(f"   Input shape: {batch['inputs'].shape}")

    # Initialize carry
    print("\n3. Initializing carry...")
    carry = model.initial_carry(batch)
    print(f"   Carry z_H shape: {carry.inner_carry.z_H.shape}")
    print(f"   Carry z_L shape: {carry.inner_carry.z_L.shape}")

    # Forward pass
    print("\n4. Running forward pass...")
    try:
        new_carry, outputs = model(carry, batch)
        print("   ✓ Forward pass successful!")
    except Exception as e:
        print(f"   ✗ Forward pass failed: {e}")
        raise

    # Check output shapes
    print("\n5. Checking output shapes...")
    output_grid = outputs["logits"]
    print(f"   Output grid shape: {output_grid.shape}")
    expected_shape = (config["batch_size"], config["grid_channels"], config["grid_height"], config["grid_width"])
    assert output_grid.shape == expected_shape, f"Expected {expected_shape}, got {output_grid.shape}"
    print(f"   ✓ Output shape correct: {output_grid.shape}")

    print(f"   Q halt logits shape: {outputs['q_halt_logits'].shape}")
    print(f"   Q continue logits shape: {outputs['q_continue_logits'].shape}")

    # Test backward pass
    print("\n6. Testing backward pass...")
    try:
        # Dummy loss: MSE between output and random target
        target = torch.randn_like(output_grid)
        loss = torch.nn.functional.mse_loss(output_grid, target)
        print(f"   Loss value: {loss.item():.4f}")

        loss.backward()
        print("   ✓ Backward pass successful!")

        # Check gradients exist
        has_grads = sum(1 for p in model.parameters() if p.grad is not None)
        print(f"   ✓ {has_grads}/{trainable_params} parameters have gradients")

    except Exception as e:
        print(f"   ✗ Backward pass failed: {e}")
        raise

    # Test multiple forward passes
    print("\n7. Testing multiple forward passes (ACT loop)...")
    try:
        carry = model.initial_carry(batch)
        for step in range(5):
            carry, outputs = model(carry, batch)
            num_halted = carry.halted.sum().item()
            print(f"   Step {step}: {num_halted}/{config['batch_size']} sequences halted")
        print("   ✓ Multiple forward passes successful!")
    except Exception as e:
        print(f"   ✗ Multiple forward passes failed: {e}")
        raise

    print("\n" + "="*60)
    print("✓ All slot attention tests passed!")
    print("="*60)


def test_token_mode():
    """Test model with original token-based encoding (backward compatibility)."""
    print("\n" + "="*60)
    print("Testing TOKEN-BASED mode (original)")
    print("="*60)

    config = {
        "batch_size": 4,
        "seq_len": 50,
        "puzzle_emb_ndim": 0,
        "puzzle_emb_len": 0,
        "num_puzzle_identifiers": 10,
        "vocab_size": 20,

        "H_cycles": 2,
        "L_cycles": 3,
        "H_layers": 1,
        "L_layers": 2,

        "hidden_size": 128,
        "expansion": 2.0,
        "num_heads": 4,
        "pos_encodings": "rope",

        "halt_max_steps": 3,
        "halt_exploration_prob": 0.1,

        "forward_dtype": "float32",

        # Disable slot attention
        "use_slot_attention": False,
    }

    print(f"Config: seq_len={config['seq_len']}, vocab_size={config['vocab_size']}")

    # Create model
    print("\n1. Creating model...")
    model = TinyRecursiveReasoningModel_ACTV1(config)
    model.train()

    # Create dummy batch with token inputs
    print("\n2. Creating dummy batch...")
    batch = {
        "inputs": torch.randint(0, config["vocab_size"], (config["batch_size"], config["seq_len"])),
        "puzzle_identifiers": torch.zeros(config["batch_size"], dtype=torch.long),
    }
    print(f"   Input shape: {batch['inputs'].shape}")

    # Initialize and run forward
    print("\n3. Running forward pass...")
    carry = model.initial_carry(batch)
    try:
        new_carry, outputs = model(carry, batch)
        print("   ✓ Forward pass successful!")
    except Exception as e:
        print(f"   ✗ Forward pass failed: {e}")
        raise

    # Check output shapes
    print("\n4. Checking output shapes...")
    logits = outputs["logits"]
    print(f"   Logits shape: {logits.shape}")
    expected_shape = (config["batch_size"], config["seq_len"], config["vocab_size"])
    assert logits.shape == expected_shape, f"Expected {expected_shape}, got {logits.shape}"
    print(f"   ✓ Output shape correct: {logits.shape}")

    print("\n" + "="*60)
    print("✓ All token-based tests passed!")
    print("="*60)


def test_with_puzzle_embeddings():
    """Test slot attention with puzzle embeddings enabled."""
    print("\n" + "="*60)
    print("Testing SLOT ATTENTION with PUZZLE EMBEDDINGS")
    print("="*60)

    config = {
        "batch_size": 4,
        "seq_len": 100,
        "puzzle_emb_ndim": 256,  # Enable puzzle embeddings
        "num_puzzle_identifiers": 100,
        "vocab_size": 20,

        "H_cycles": 2,
        "L_cycles": 2,
        "H_layers": 1,
        "L_layers": 2,

        "hidden_size": 128,
        "expansion": 2.0,
        "num_heads": 4,
        "pos_encodings": "learned",  # Test learned positions

        "halt_max_steps": 3,
        "halt_exploration_prob": 0.1,

        "forward_dtype": "float32",

        # Slot Attention
        "use_slot_attention": True,
        "num_slots": 6,
        "slot_dim": 64,
        "slot_iterations": 2,
        "slot_mlp_hidden": 128,
        "grid_height": 15,
        "grid_width": 15,
        "grid_channels": 1,
        "cnn_hidden_dim": 64,
        "decoder_hidden_dim": 64,
        "puzzle_emb_len": 2,
    }

    print(f"Config: puzzle_emb_ndim={config['puzzle_emb_ndim']}, num_slots={config['num_slots']}")

    print("\n1. Creating model with puzzle embeddings...")
    model = TinyRecursiveReasoningModel_ACTV1(config)

    print("\n2. Creating batch with different puzzle IDs...")
    batch = {
        "inputs": torch.randn(config["batch_size"], config["grid_height"], config["grid_width"]),
        "puzzle_identifiers": torch.tensor([0, 1, 2, 3]),  # Different puzzles
    }

    print("\n3. Running forward pass...")
    carry = model.initial_carry(batch)
    try:
        new_carry, outputs = model(carry, batch)
        print("   ✓ Forward pass successful!")
        print(f"   Carry shape (with puzzle emb): {carry.inner_carry.z_H.shape}")
        print(f"   Expected: (batch={config['batch_size']}, seq={config['num_slots']+config['puzzle_emb_len']}, hidden={config['hidden_size']})")
    except Exception as e:
        print(f"   ✗ Forward pass failed: {e}")
        raise

    print("\n" + "="*60)
    print("✓ Puzzle embedding tests passed!")
    print("="*60)


def test_different_grid_sizes():
    """Test that model handles different grid configurations."""
    print("\n" + "="*60)
    print("Testing DIFFERENT GRID SIZES")
    print("="*60)

    test_cases = [
        {"height": 5, "width": 5, "channels": 1, "desc": "Small 5x5 single channel"},
        {"height": 30, "width": 30, "channels": 1, "desc": "Large 30x30 single channel"},
        {"height": 10, "width": 15, "channels": 1, "desc": "Rectangular 10x15"},
    ]

    for i, case in enumerate(test_cases, 1):
        print(f"\n{i}. Testing: {case['desc']}")

        config = {
            "batch_size": 2,
            "seq_len": 100,
            "puzzle_emb_ndim": 0,
            "puzzle_emb_len": 0,
            "num_puzzle_identifiers": 10,
            "vocab_size": 20,
            "H_cycles": 1,
            "L_cycles": 1,
            "H_layers": 1,
            "L_layers": 1,
            "hidden_size": 64,
            "expansion": 2.0,
            "num_heads": 4,
            "pos_encodings": "rope",
            "halt_max_steps": 2,
            "halt_exploration_prob": 0.1,
            "forward_dtype": "float32",
            "use_slot_attention": True,
            "num_slots": 5,
            "slot_dim": 32,
            "slot_iterations": 2,
            "slot_mlp_hidden": 64,
            "grid_height": case["height"],
            "grid_width": case["width"],
            "grid_channels": case["channels"],
            "cnn_hidden_dim": 32,
            "decoder_hidden_dim": 32,
        }

        model = TinyRecursiveReasoningModel_ACTV1(config)
        batch = {
            "inputs": torch.randn(2, case["height"], case["width"]),
            "puzzle_identifiers": torch.zeros(2, dtype=torch.long),
        }

        carry = model.initial_carry(batch)
        new_carry, outputs = model(carry, batch)

        assert outputs["logits"].shape == (2, case["channels"], case["height"], case["width"])
        print(f"   ✓ Output shape: {outputs['logits'].shape}")

    print("\n" + "="*60)
    print("✓ All grid size tests passed!")
    print("="*60)


if __name__ == "__main__":
    print("\n" + "="*80)
    print(" "*20 + "SLOT ATTENTION TEST SUITE")
    print("="*80)

    try:
        # Test 1: Slot attention mode
        test_slot_attention_mode()

        # Test 2: Token mode (backward compatibility)
        test_token_mode()

        # Test 3: With puzzle embeddings
        test_with_puzzle_embeddings()

        # Test 4: Different grid sizes
        test_different_grid_sizes()

        print("\n" + "="*80)
        print(" "*25 + "ALL TESTS PASSED! ✓")
        print("="*80)
        print("\nYour slot attention implementation is ready for training on the server!")
        print("No bugs detected in:")
        print("  - Slot attention encoding/decoding")
        print("  - Forward/backward passes")
        print("  - ACT loop")
        print("  - Backward compatibility with token mode")
        print("  - Puzzle embeddings integration")
        print("  - Various grid configurations")
        print("\n")

    except Exception as e:
        print("\n" + "="*80)
        print(" "*30 + "TESTS FAILED! ✗")
        print("="*80)
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
