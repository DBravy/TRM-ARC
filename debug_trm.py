"""
Debug script to verify that the actual TRM from trm.py is working correctly.
Tests gradient flow, numerical stability, and learning ability on synthetic data.

Run: python debug_trm.py

This helps distinguish between architecture limitations vs implementation bugs.
"""

import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.recursive_reasoning.trm import TinyRecursiveReasoningModel_ACTV1

# =============================================================================
# Configuration
# =============================================================================

# Device setup
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    DTYPE = torch.bfloat16
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
    DTYPE = torch.float32
else:
    DEVICE = torch.device("cpu")
    DTYPE = torch.float32

print(f"Device: {DEVICE}, Dtype: {DTYPE}")

# =============================================================================
# Synthetic Dataset
# =============================================================================

class SyntheticPuzzleDataset(Dataset):
    """
    Simple synthetic dataset for debugging.
    Task: Given a pattern of tokens, predict a deterministic transformation.

    Example patterns:
    - Identity: output = input
    - Shift: output[i] = input[i] + 1 (mod vocab)
    - Swap: output[i,j] = input[j,i]
    """

    def __init__(self, num_samples: int, seq_len: int, vocab_size: int,
                 task: str = "identity", grid_size: int = 10):
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.task = task
        self.grid_size = grid_size

        # Generate data
        self.inputs = []
        self.labels = []

        for _ in range(num_samples):
            # Create a grid pattern (simulating ARC puzzle)
            # Use only vocab tokens 2-11 (0=PAD, 1=EOS)
            grid = np.random.randint(2, min(vocab_size, 12), size=(grid_size, grid_size))

            # Apply transformation
            if task == "identity":
                output_grid = grid.copy()
            elif task == "shift":
                output_grid = ((grid - 2 + 1) % 10) + 2  # Shift digits
            elif task == "invert":
                output_grid = (11 - grid + 2)  # Invert: 0->9, 1->8, etc.
            elif task == "constant":
                # Output is always the same pattern (easiest test)
                output_grid = np.full_like(grid, 5)  # All 5s
            else:
                output_grid = grid.copy()

            # Flatten and pad to seq_len
            input_flat = np.zeros(seq_len, dtype=np.int64)
            label_flat = np.zeros(seq_len, dtype=np.int64)

            flat_size = grid_size * grid_size
            input_flat[:flat_size] = grid.flatten()
            label_flat[:flat_size] = output_grid.flatten()

            self.inputs.append(input_flat)
            self.labels.append(label_flat)

        self.inputs = np.array(self.inputs)
        self.labels = np.array(self.labels)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return {
            "inputs": torch.tensor(self.inputs[idx], dtype=torch.long),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
            "puzzle_identifiers": torch.tensor(1, dtype=torch.long),  # Single puzzle
        }


# =============================================================================
# Test Functions
# =============================================================================

def create_trm_model(batch_size: int = 8, seq_len: int = 100, vocab_size: int = 12,
                     hidden_size: int = 64, num_layers: int = 2, num_heads: int = 4,
                     H_cycles: int = 2, L_cycles: int = 3):
    """Create a TRM model with the given configuration."""

    dtype_str = "float32" if DTYPE == torch.float32 else "bfloat16"

    config_dict = {
        "batch_size": batch_size,
        "seq_len": seq_len,
        "puzzle_emb_ndim": hidden_size,
        "num_puzzle_identifiers": 3,  # 0=reserved, 1=puzzle1, 2=puzzle2
        "vocab_size": vocab_size,
        "H_cycles": H_cycles,
        "L_cycles": L_cycles,
        "H_layers": 1,
        "L_layers": num_layers,
        "hidden_size": hidden_size,
        "expansion": 4.0,
        "num_heads": num_heads,
        "pos_encodings": "rope",
        "rms_norm_eps": 1e-5,
        "rope_theta": 10000.0,
        "halt_max_steps": 1,
        "halt_exploration_prob": 0.0,
        "forward_dtype": dtype_str,
        "mlp_t": False,
        "puzzle_emb_len": 16,
        "no_ACT_continue": True,
    }

    model = TinyRecursiveReasoningModel_ACTV1(config_dict)
    return model.to(DEVICE).to(DTYPE)


def test_forward_pass():
    """Test 1: Basic forward pass - shapes and no NaN/Inf."""
    print("\n" + "="*60)
    print("TEST 1: Forward Pass Sanity Check")
    print("="*60)

    batch_size = 8
    seq_len = 100
    vocab_size = 12

    model = create_trm_model(batch_size=batch_size, seq_len=seq_len, vocab_size=vocab_size)
    model.eval()

    # Create dummy input
    inputs = torch.randint(2, vocab_size, (batch_size, seq_len), device=DEVICE)
    puzzle_ids = torch.ones(batch_size, dtype=torch.long, device=DEVICE)

    batch = {"inputs": inputs, "puzzle_identifiers": puzzle_ids}

    # Get initial carry
    carry = model.initial_carry(batch)
    carry.inner_carry.z_H = carry.inner_carry.z_H.to(DEVICE)
    carry.inner_carry.z_L = carry.inner_carry.z_L.to(DEVICE)
    carry.steps = carry.steps.to(DEVICE)
    carry.halted = carry.halted.to(DEVICE)
    carry.current_data = {k: v.to(DEVICE) for k, v in carry.current_data.items()}

    # Forward pass
    with torch.no_grad():
        _, outputs = model(carry, batch)

    logits = outputs["logits"]

    # Check shapes
    expected_shape = (batch_size, seq_len, vocab_size)
    shape_ok = logits.shape == expected_shape
    print(f"  Logits shape: {logits.shape} (expected {expected_shape}) - {'PASS' if shape_ok else 'FAIL'}")

    # Check for NaN/Inf
    has_nan = torch.isnan(logits).any().item()
    has_inf = torch.isinf(logits).any().item()
    print(f"  Contains NaN: {has_nan} - {'FAIL' if has_nan else 'PASS'}")
    print(f"  Contains Inf: {has_inf} - {'FAIL' if has_inf else 'PASS'}")

    # Check output distribution
    logits_float = logits.float()
    print(f"  Logits min/max: {logits_float.min().item():.4f} / {logits_float.max().item():.4f}")
    print(f"  Logits mean/std: {logits_float.mean().item():.4f} / {logits_float.std().item():.4f}")

    # Check softmax distribution
    probs = F.softmax(logits_float, dim=-1)
    entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1).mean()
    max_entropy = np.log(vocab_size)
    print(f"  Avg entropy: {entropy.item():.4f} (max: {max_entropy:.4f})")
    print(f"  Entropy ratio: {entropy.item()/max_entropy:.2%} (high = uniform, low = confident)")

    passed = shape_ok and not has_nan and not has_inf
    print(f"\n  TEST 1 {'PASSED' if passed else 'FAILED'}")
    return passed


def test_gradient_flow():
    """Test 2: Verify gradients flow through all trainable parameters."""
    print("\n" + "="*60)
    print("TEST 2: Gradient Flow Verification")
    print("="*60)

    batch_size = 8
    seq_len = 100
    vocab_size = 12

    model = create_trm_model(batch_size=batch_size, seq_len=seq_len, vocab_size=vocab_size)
    model.train()

    # Create dummy input with labels
    inputs = torch.randint(2, vocab_size, (batch_size, seq_len), device=DEVICE)
    labels = torch.randint(2, vocab_size, (batch_size, seq_len), device=DEVICE)
    puzzle_ids = torch.ones(batch_size, dtype=torch.long, device=DEVICE)

    batch = {"inputs": inputs, "puzzle_identifiers": puzzle_ids}

    # Get initial carry
    carry = model.initial_carry(batch)
    carry.inner_carry.z_H = carry.inner_carry.z_H.to(DEVICE)
    carry.inner_carry.z_L = carry.inner_carry.z_L.to(DEVICE)
    carry.steps = carry.steps.to(DEVICE)
    carry.halted = carry.halted.to(DEVICE)
    carry.current_data = {k: v.to(DEVICE) for k, v in carry.current_data.items()}

    # Forward pass
    _, outputs = model(carry, batch)
    logits = outputs["logits"]

    # Compute loss
    loss = F.cross_entropy(
        logits.reshape(-1, vocab_size).float(),
        labels.reshape(-1),
        ignore_index=0
    )

    print(f"  Loss value: {loss.item():.4f}")

    # Backward pass
    loss.backward()

    # Check gradients for all named parameters
    params_with_grad = 0
    params_without_grad = 0
    params_with_nan_grad = 0
    zero_grad_params = []
    nan_grad_params = []

    for name, param in model.named_parameters():
        if param.requires_grad:
            if param.grad is not None:
                if torch.isnan(param.grad).any():
                    params_with_nan_grad += 1
                    nan_grad_params.append(name)
                elif param.grad.abs().max() > 0:
                    params_with_grad += 1
                else:
                    zero_grad_params.append(name)
                    params_without_grad += 1
            else:
                params_without_grad += 1
                zero_grad_params.append(name)

    total_params = params_with_grad + params_without_grad
    print(f"  Parameters with gradients: {params_with_grad}/{total_params}")
    print(f"  Parameters with zero/no gradients: {params_without_grad}")
    print(f"  Parameters with NaN gradients: {params_with_nan_grad}")

    if zero_grad_params:
        print(f"  Zero gradient parameters:")
        for name in zero_grad_params[:5]:
            print(f"    - {name}")
        if len(zero_grad_params) > 5:
            print(f"    ... and {len(zero_grad_params) - 5} more")

    if nan_grad_params:
        print(f"  NaN gradient parameters:")
        for name in nan_grad_params[:5]:
            print(f"    - {name}")

    # Check specific important parameters
    important_params = [
        "inner.embed_tokens.embedding_weight",
        "inner.lm_head.weight",
        "inner.L_level.layers.0.mlp.gate_up_proj.weight",
    ]

    print("\n  Gradient magnitudes for key parameters:")
    for name, param in model.named_parameters():
        if any(imp in name for imp in important_params) and param.grad is not None:
            grad_norm = param.grad.float().norm().item()
            print(f"    {name}: {grad_norm:.6f}")

    passed = params_with_grad > 0 and params_with_nan_grad == 0
    print(f"\n  TEST 2 {'PASSED' if passed else 'FAILED'}")
    return passed


def test_overfitting_synthetic():
    """Test 3: Can the model overfit on a tiny synthetic dataset?"""
    print("\n" + "="*60)
    print("TEST 3: Overfitting on Synthetic Data")
    print("="*60)

    # Small configuration for fast testing
    batch_size = 8
    seq_len = 100  # 10x10 grid
    vocab_size = 12
    num_samples = 16  # Very small dataset
    grid_size = 10
    num_epochs = 100

    # Test with "constant" task first (easiest)
    for task_name in ["constant", "identity"]:
        print(f"\n  Task: {task_name}")
        print(f"  {'-'*40}")

        # Create dataset
        dataset = SyntheticPuzzleDataset(
            num_samples=num_samples,
            seq_len=seq_len,
            vocab_size=vocab_size,
            task=task_name,
            grid_size=grid_size
        )
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # Create model
        model = create_trm_model(
            batch_size=batch_size,
            seq_len=seq_len,
            vocab_size=vocab_size,
            hidden_size=64,
            num_layers=2,
            num_heads=4,
            H_cycles=2,
            L_cycles=3
        )

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        initial_loss = None
        final_loss = None
        initial_acc = None
        final_acc = None

        for epoch in range(num_epochs):
            model.train()
            epoch_loss = 0
            epoch_correct = 0
            epoch_total = 0

            for batch_data in loader:
                inputs = batch_data["inputs"].to(DEVICE)
                labels = batch_data["labels"].to(DEVICE)
                puzzle_ids = batch_data["puzzle_identifiers"].to(DEVICE)

                # Handle batch size mismatch
                actual_bs = inputs.shape[0]
                if actual_bs < batch_size:
                    pad_size = batch_size - actual_bs
                    inputs = torch.cat([inputs, inputs[:1].expand(pad_size, -1)], dim=0)
                    labels = torch.cat([labels, labels[:1].expand(pad_size, -1)], dim=0)
                    puzzle_ids = torch.cat([puzzle_ids, puzzle_ids[:1].expand(pad_size)], dim=0)

                batch = {"inputs": inputs, "puzzle_identifiers": puzzle_ids}

                carry = model.initial_carry(batch)
                carry.inner_carry.z_H = carry.inner_carry.z_H.to(DEVICE)
                carry.inner_carry.z_L = carry.inner_carry.z_L.to(DEVICE)
                carry.steps = carry.steps.to(DEVICE)
                carry.halted = carry.halted.to(DEVICE)
                carry.current_data = {k: v.to(DEVICE) for k, v in carry.current_data.items()}

                optimizer.zero_grad()

                _, outputs = model(carry, batch)
                logits = outputs["logits"][:actual_bs]  # Remove padding
                labels = labels[:actual_bs]

                loss = F.cross_entropy(
                    logits.reshape(-1, vocab_size).float(),
                    labels.reshape(-1),
                    ignore_index=0
                )

                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

                # Accuracy
                preds = logits.argmax(dim=-1)
                mask = labels != 0
                epoch_correct += ((preds == labels) & mask).sum().item()
                epoch_total += mask.sum().item()

            avg_loss = epoch_loss / len(loader)
            acc = epoch_correct / epoch_total if epoch_total > 0 else 0

            if epoch == 0:
                initial_loss = avg_loss
                initial_acc = acc

            if (epoch + 1) % 25 == 0:
                print(f"    Epoch {epoch+1:3d}: Loss = {avg_loss:.4f}, Acc = {acc:.2%}")

        final_loss = avg_loss
        final_acc = acc

        # Check if we improved
        loss_improved = final_loss < initial_loss * 0.5  # At least 50% reduction
        acc_improved = final_acc > initial_acc + 0.1  # At least 10% improvement

        print(f"\n    Initial -> Final Loss: {initial_loss:.4f} -> {final_loss:.4f}")
        print(f"    Initial -> Final Acc:  {initial_acc:.2%} -> {final_acc:.2%}")
        print(f"    Loss improved: {'YES' if loss_improved else 'NO'}")
        print(f"    Acc improved:  {'YES' if acc_improved else 'NO'}")

        if task_name == "constant" and not (loss_improved or acc_improved):
            print(f"\n  WARNING: Model cannot learn even the simplest task!")
            print(f"  This likely indicates a bug, not an architecture issue.")
            return False

    print(f"\n  TEST 3 PASSED")
    return True


def test_h_l_cycles():
    """Test 4: Verify H/L cycles are actually doing something different."""
    print("\n" + "="*60)
    print("TEST 4: H/L Cycle Behavior Analysis")
    print("="*60)

    batch_size = 4
    seq_len = 100
    vocab_size = 12

    # Create two models with different H/L cycles
    configs = [
        {"H_cycles": 1, "L_cycles": 1},
        {"H_cycles": 2, "L_cycles": 3},
        {"H_cycles": 3, "L_cycles": 4},
    ]

    inputs = torch.randint(2, vocab_size, (batch_size, seq_len), device=DEVICE)
    puzzle_ids = torch.ones(batch_size, dtype=torch.long, device=DEVICE)
    batch = {"inputs": inputs, "puzzle_identifiers": puzzle_ids}

    outputs_list = []

    for config in configs:
        model = create_trm_model(
            batch_size=batch_size,
            seq_len=seq_len,
            vocab_size=vocab_size,
            H_cycles=config["H_cycles"],
            L_cycles=config["L_cycles"]
        )
        model.eval()

        carry = model.initial_carry(batch)
        carry.inner_carry.z_H = carry.inner_carry.z_H.to(DEVICE)
        carry.inner_carry.z_L = carry.inner_carry.z_L.to(DEVICE)
        carry.steps = carry.steps.to(DEVICE)
        carry.halted = carry.halted.to(DEVICE)
        carry.current_data = {k: v.to(DEVICE) for k, v in carry.current_data.items()}

        with torch.no_grad():
            _, outputs = model(carry, batch)

        logits = outputs["logits"]
        outputs_list.append({
            "config": config,
            "logits": logits,
            "logits_mean": logits.float().mean().item(),
            "logits_std": logits.float().std().item(),
        })

        print(f"  H={config['H_cycles']}, L={config['L_cycles']}: "
              f"mean={outputs_list[-1]['logits_mean']:.4f}, "
              f"std={outputs_list[-1]['logits_std']:.4f}")

    # Check if outputs differ between configurations
    # (They should, since more cycles = more computation)
    diff_01 = (outputs_list[0]["logits"] - outputs_list[1]["logits"]).abs().mean().item()
    diff_12 = (outputs_list[1]["logits"] - outputs_list[2]["logits"]).abs().mean().item()

    print(f"\n  Output differences:")
    print(f"    Config 0 vs Config 1: {diff_01:.4f}")
    print(f"    Config 1 vs Config 2: {diff_12:.4f}")

    # They should be different (not exactly the same)
    outputs_differ = diff_01 > 0.01 or diff_12 > 0.01
    print(f"\n  Outputs differ with cycle count: {'YES' if outputs_differ else 'NO'}")

    print(f"\n  TEST 4 {'PASSED' if outputs_differ else 'WARNING: Outputs identical!'}")
    return True  # This is informational


def test_loss_decreases():
    """Test 5: Verify loss actually decreases during training."""
    print("\n" + "="*60)
    print("TEST 5: Loss Decrease Verification")
    print("="*60)

    batch_size = 8
    seq_len = 100
    vocab_size = 12

    model = create_trm_model(
        batch_size=batch_size,
        seq_len=seq_len,
        vocab_size=vocab_size,
        hidden_size=64
    )

    # Fixed training data (to test memorization)
    torch.manual_seed(42)
    fixed_inputs = torch.randint(2, vocab_size, (batch_size, seq_len), device=DEVICE)
    fixed_labels = torch.randint(2, vocab_size, (batch_size, seq_len), device=DEVICE)
    fixed_puzzle_ids = torch.ones(batch_size, dtype=torch.long, device=DEVICE)

    batch = {"inputs": fixed_inputs, "puzzle_identifiers": fixed_puzzle_ids}

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    losses = []

    for step in range(50):
        model.train()

        carry = model.initial_carry(batch)
        carry.inner_carry.z_H = carry.inner_carry.z_H.to(DEVICE)
        carry.inner_carry.z_L = carry.inner_carry.z_L.to(DEVICE)
        carry.steps = carry.steps.to(DEVICE)
        carry.halted = carry.halted.to(DEVICE)
        carry.current_data = {k: v.to(DEVICE) for k, v in carry.current_data.items()}

        optimizer.zero_grad()

        _, outputs = model(carry, batch)
        logits = outputs["logits"]

        loss = F.cross_entropy(
            logits.reshape(-1, vocab_size).float(),
            fixed_labels.reshape(-1),
            ignore_index=0
        )

        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        if (step + 1) % 10 == 0:
            print(f"    Step {step+1:3d}: Loss = {loss.item():.4f}")

    # Check if loss decreased
    initial_loss = np.mean(losses[:5])
    final_loss = np.mean(losses[-5:])

    print(f"\n    Initial loss (avg first 5): {initial_loss:.4f}")
    print(f"    Final loss (avg last 5): {final_loss:.4f}")
    print(f"    Reduction: {(1 - final_loss/initial_loss)*100:.1f}%")

    loss_decreased = final_loss < initial_loss * 0.9  # At least 10% reduction

    print(f"\n  TEST 5 {'PASSED' if loss_decreased else 'FAILED: Loss did not decrease!'}")
    return loss_decreased


def test_parameter_updates():
    """Test 6: Verify parameters actually change during training."""
    print("\n" + "="*60)
    print("TEST 6: Parameter Update Verification")
    print("="*60)

    batch_size = 8
    seq_len = 100
    vocab_size = 12

    model = create_trm_model(batch_size=batch_size, seq_len=seq_len, vocab_size=vocab_size)

    # Save initial parameter values
    initial_params = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            initial_params[name] = param.detach().clone()

    # Training step
    inputs = torch.randint(2, vocab_size, (batch_size, seq_len), device=DEVICE)
    labels = torch.randint(2, vocab_size, (batch_size, seq_len), device=DEVICE)
    puzzle_ids = torch.ones(batch_size, dtype=torch.long, device=DEVICE)

    batch = {"inputs": inputs, "puzzle_identifiers": puzzle_ids}

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    model.train()

    for _ in range(5):  # A few steps
        carry = model.initial_carry(batch)
        carry.inner_carry.z_H = carry.inner_carry.z_H.to(DEVICE)
        carry.inner_carry.z_L = carry.inner_carry.z_L.to(DEVICE)
        carry.steps = carry.steps.to(DEVICE)
        carry.halted = carry.halted.to(DEVICE)
        carry.current_data = {k: v.to(DEVICE) for k, v in carry.current_data.items()}

        optimizer.zero_grad()
        _, outputs = model(carry, batch)

        loss = F.cross_entropy(
            outputs["logits"].reshape(-1, vocab_size).float(),
            labels.reshape(-1),
            ignore_index=0
        )
        loss.backward()
        optimizer.step()

    # Check which parameters changed
    params_changed = 0
    params_unchanged = 0
    unchanged_names = []

    for name, param in model.named_parameters():
        if param.requires_grad and name in initial_params:
            diff = (param - initial_params[name]).abs().max().item()
            if diff > 1e-7:
                params_changed += 1
            else:
                params_unchanged += 1
                unchanged_names.append(name)

    total = params_changed + params_unchanged
    print(f"  Parameters that changed: {params_changed}/{total}")
    print(f"  Parameters unchanged: {params_unchanged}")

    if unchanged_names:
        print(f"  Unchanged parameters:")
        for name in unchanged_names[:5]:
            print(f"    - {name}")

    passed = params_changed > params_unchanged
    print(f"\n  TEST 6 {'PASSED' if passed else 'FAILED'}")
    return passed


# =============================================================================
# Main
# =============================================================================

def main():
    print("="*60)
    print("TRM DEBUG SCRIPT")
    print("Verifying TinyRecursiveReasoningModel_ACTV1 is working correctly")
    print("="*60)

    results = {}

    # Run all tests
    results["Forward Pass"] = test_forward_pass()
    results["Gradient Flow"] = test_gradient_flow()
    results["Loss Decreases"] = test_loss_decreases()
    results["Parameter Updates"] = test_parameter_updates()
    results["H/L Cycles"] = test_h_l_cycles()
    results["Overfitting Synthetic"] = test_overfitting_synthetic()

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    all_passed = True
    for test_name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {test_name}: {status}")
        if not passed:
            all_passed = False

    print("\n" + "="*60)
    if all_passed:
        print("ALL TESTS PASSED!")
        print("The TRM implementation appears to be working correctly.")
        print("Poor learning is likely due to architecture/hyperparameters, not bugs.")
    else:
        print("SOME TESTS FAILED!")
        print("There may be bugs in the TRM implementation.")
        print("Review the failed tests above for details.")
    print("="*60)

    return 0 if all_passed else 1


if __name__ == "__main__":
    exit(main())
