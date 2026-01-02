#!/usr/bin/env python3
"""
CNN Diagnostic: Is the CNN actually using the input?

This script tests whether the pretrained CNN is conditioning on the input
or just learning output statistics.

Tests:
1. (correct_input, correct_output) → expect HIGH confidence (all correct)
2. (correct_input, all_1s)         → expect LOW confidence (garbage output)
3. (random_input, correct_output)  → if HIGH, CNN ignores input!
4. (wrong_puzzle_input, correct_output) → if HIGH, CNN ignores input!

If tests 3 and 4 show high confidence, the CNN learned to recognize
"valid-looking outputs" without comparing to the input.
"""

import argparse
import json
import os
import numpy as np
import torch

# Determine device
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

GRID_SIZE = 30


def load_cnn(checkpoint_path: str):
    """Load the CNN model"""
    # Import here to avoid dependency issues
    from train_pixel_error_cnn import DorsalCNN
    return DorsalCNN.from_checkpoint(checkpoint_path, device=DEVICE)


def load_puzzles(dataset_name: str, data_root: str = "kaggle/combined"):
    """Load puzzle data"""
    config = {
        "arc-agi-1": {"subsets": ["training", "evaluation"]},
        "arc-agi-2": {"subsets": ["training2", "evaluation2"]},
    }

    all_puzzles = {}

    for subset in config[dataset_name]["subsets"]:
        challenges_path = f"{data_root}/arc-agi_{subset}_challenges.json"
        solutions_path = f"{data_root}/arc-agi_{subset}_solutions.json"

        if not os.path.exists(challenges_path):
            continue

        with open(challenges_path) as f:
            puzzles = json.load(f)

        if os.path.exists(solutions_path):
            with open(solutions_path) as f:
                solutions = json.load(f)
            for puzzle_id in puzzles:
                if puzzle_id in solutions:
                    for i, sol in enumerate(solutions[puzzle_id]):
                        if i < len(puzzles[puzzle_id]["test"]):
                            puzzles[puzzle_id]["test"][i]["output"] = sol

        all_puzzles.update(puzzles)

    return all_puzzles


def pad_grid(grid: np.ndarray) -> np.ndarray:
    """Pad grid to 30x30"""
    h, w = grid.shape
    padded = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.uint8)
    padded[:h, :w] = grid
    return padded


def to_tensor(grid: np.ndarray) -> torch.Tensor:
    """Convert to tensor and add batch dimension"""
    return torch.from_numpy(pad_grid(grid).copy()).long().unsqueeze(0).to(DEVICE)


def get_confidence(model, input_grid: np.ndarray, output_grid: np.ndarray) -> dict:
    """Get CNN confidence statistics"""
    inp_t = to_tensor(input_grid)
    out_t = to_tensor(output_grid)
    
    with torch.no_grad():
        proba = model.predict_proba(inp_t, out_t)[0].cpu().numpy()
    
    # Find content region (where output is non-zero)
    padded_out = pad_grid(output_grid)
    content_mask = padded_out > 0
    
    if content_mask.sum() > 0:
        content_confidence = proba[content_mask].mean()
    else:
        content_confidence = proba.mean()
    
    return {
        "mean": proba.mean(),
        "content_mean": content_confidence,
        "min": proba.min(),
        "max": proba.max(),
        "pct_high": (proba > 0.5).mean(),  # % of pixels CNN thinks are correct
    }


def print_grid_comparison(input_grid: np.ndarray, output_grid: np.ndarray, 
                          proba: np.ndarray, title: str, max_size: int = 10):
    """Print grids side by side with CNN confidence"""
    inp_padded = pad_grid(input_grid)
    out_padded = pad_grid(output_grid)
    
    # Find bounds
    h = min(max(input_grid.shape[0], output_grid.shape[0]), max_size)
    w = min(max(input_grid.shape[1], output_grid.shape[1]), max_size)
    
    print(f"\n{'─'*70}")
    print(f"{title}")
    print(f"{'─'*70}")
    print(f"{'INPUT':<25} {'OUTPUT':<25} {'CNN CONFIDENCE':<20}")
    
    for r in range(h):
        inp_row = " ".join(f"{inp_padded[r, c]}" for c in range(w))
        out_row = " ".join(f"{out_padded[r, c]}" for c in range(w))
        conf_row = " ".join(f"{proba[r, c]:.1f}"[1:] for c in range(w))  # Remove leading 0
        print(f"{inp_row:<25} {out_row:<25} {conf_row:<20}")


def run_diagnostic(model, puzzle_id: str, puzzle: dict, all_puzzles: dict):
    """Run all diagnostic tests for a puzzle"""
    
    # Get first training example
    example = puzzle["train"][0]
    correct_input = np.array(example["input"], dtype=np.uint8)
    correct_output = np.array(example["output"], dtype=np.uint8)
    
    # Get a different puzzle's input/output
    other_puzzle_ids = [pid for pid in all_puzzles.keys() if pid != puzzle_id]
    if other_puzzle_ids:
        other_id = other_puzzle_ids[0]
        other_example = all_puzzles[other_id]["train"][0]
        wrong_input = np.array(other_example["input"], dtype=np.uint8)
        wrong_output = np.array(other_example["output"], dtype=np.uint8)
    else:
        wrong_input = np.random.randint(0, 10, correct_input.shape, dtype=np.uint8)
        wrong_output = np.random.randint(0, 10, correct_output.shape, dtype=np.uint8)
    
    # Create test cases
    all_1s = np.ones_like(correct_output, dtype=np.uint8)
    all_0s = np.zeros_like(correct_output, dtype=np.uint8)
    random_grid = np.random.randint(0, 10, correct_output.shape, dtype=np.uint8)
    random_input = np.random.randint(0, 10, correct_input.shape, dtype=np.uint8)
    
    print(f"\n{'='*70}")
    print(f"DIAGNOSTIC: Puzzle {puzzle_id}")
    print(f"{'='*70}")
    print(f"Input shape: {correct_input.shape}, Output shape: {correct_output.shape}")
    
    tests = [
        ("TEST 1: correct_input + correct_output", correct_input, correct_output, 
         "EXPECT: HIGH confidence (this is ground truth)"),
        
        ("TEST 2: correct_input + all_1s", correct_input, all_1s,
         "EXPECT: LOW confidence (obvious garbage)"),
        
        ("TEST 3: correct_input + all_0s", correct_input, all_0s,
         "EXPECT: LOW confidence (blank output)"),
        
        ("TEST 4: correct_input + random_noise", correct_input, random_grid,
         "EXPECT: LOW confidence (random garbage)"),
        
        ("TEST 5: RANDOM_input + correct_output", random_input, correct_output,
         "CRITICAL: If HIGH, CNN ignores input!"),
        
        ("TEST 6: WRONG_PUZZLE_input + correct_output", wrong_input, correct_output,
         "CRITICAL: If HIGH, CNN ignores input!"),
        
        ("TEST 7: correct_input + WRONG_PUZZLE_output", correct_input, wrong_output,
         "EXPECT: LOW confidence (wrong puzzle's output)"),
    ]
    
    print(f"\n{'Test':<45} {'Mean':>8} {'Content':>8} {'%High':>8}")
    print(f"{'-'*45} {'-'*8} {'-'*8} {'-'*8}")
    
    results = []
    for name, inp, out, expectation in tests:
        conf = get_confidence(model, inp, out)
        results.append((name, inp, out, conf, expectation))
        
        # Highlight critical tests
        is_critical = "CRITICAL" in expectation
        prefix = ">>>" if is_critical else "   "
        
        print(f"{prefix}{name:<42} {conf['mean']:>8.2%} {conf['content_mean']:>8.2%} {conf['pct_high']:>8.2%}")
    
    # Print interpretations
    print(f"\n{'─'*70}")
    print("INTERPRETATION:")
    print(f"{'─'*70}")
    
    test1_conf = results[0][3]['content_mean']
    test5_conf = results[4][3]['content_mean']
    test6_conf = results[5][3]['content_mean']
    
    if test5_conf > 0.7 or test6_conf > 0.7:
        print("⚠️  CNN IS NOT USING THE INPUT!")
        print(f"   Test 5 (random input): {test5_conf:.2%} confidence")
        print(f"   Test 6 (wrong puzzle input): {test6_conf:.2%} confidence")
        print("   The CNN learned to recognize 'valid-looking outputs'")
        print("   without comparing to the input.")
    elif test5_conf > 0.4 or test6_conf > 0.4:
        print("⚠️  CNN may be partially ignoring input")
        print(f"   Test 5 (random input): {test5_conf:.2%} confidence")
        print(f"   Test 6 (wrong puzzle input): {test6_conf:.2%} confidence")
    else:
        print("✓  CNN appears to be using the input")
        print(f"   Test 5 (random input): {test5_conf:.2%} confidence")
        print(f"   Test 6 (wrong puzzle input): {test6_conf:.2%} confidence")
    
    # Show visual for critical tests
    if True:  # Always show for debugging
        inp_t = to_tensor(correct_input)
        
        # Test 1 visual
        out_t = to_tensor(correct_output)
        with torch.no_grad():
            proba1 = model.predict_proba(inp_t, out_t)[0].cpu().numpy()
        print_grid_comparison(correct_input, correct_output, proba1, 
                             "TEST 1: Correct input + Correct output")
        
        # Test 5 visual
        rand_t = to_tensor(random_input)
        with torch.no_grad():
            proba5 = model.predict_proba(rand_t, out_t)[0].cpu().numpy()
        print_grid_comparison(random_input, correct_output, proba5,
                             "TEST 5: RANDOM input + Correct output (CRITICAL)")
        
        # Test 6 visual
        wrong_t = to_tensor(wrong_input)
        with torch.no_grad():
            proba6 = model.predict_proba(wrong_t, out_t)[0].cpu().numpy()
        print_grid_comparison(wrong_input, correct_output, proba6,
                             "TEST 6: WRONG PUZZLE input + Correct output (CRITICAL)")

    return results


def main():
    parser = argparse.ArgumentParser(description="Diagnose CNN input usage")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/pixel_error_cnn.pt",
                        help="Path to CNN checkpoint")
    parser.add_argument("--dataset", type=str, default="arc-agi-1")
    parser.add_argument("--data-root", type=str, default="kaggle/combined")
    parser.add_argument("--puzzle", type=str, default=None,
                        help="Specific puzzle ID to test (default: first puzzle)")
    parser.add_argument("--num-puzzles", type=int, default=3,
                        help="Number of puzzles to test if --puzzle not specified")
    
    args = parser.parse_args()
    
    print(f"Device: {DEVICE}")
    print(f"Loading CNN from: {args.checkpoint}")
    
    # Load model
    model = load_cnn(args.checkpoint)
    model.eval()
    
    # Load puzzles
    puzzles = load_puzzles(args.dataset, args.data_root)
    print(f"Loaded {len(puzzles)} puzzles")
    
    if args.puzzle:
        if args.puzzle not in puzzles:
            print(f"Error: Puzzle '{args.puzzle}' not found!")
            return
        puzzle_ids = [args.puzzle]
    else:
        puzzle_ids = list(puzzles.keys())[:args.num_puzzles]
    
    # Run diagnostics
    for pid in puzzle_ids:
        run_diagnostic(model, pid, puzzles[pid], puzzles)
    
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print("""
If Tests 5 and 6 show HIGH confidence (>70%), the CNN learned to 
recognize "valid-looking outputs" without using the input at all.

This would explain why:
- Pretraining works (correct vs corrupted outputs look different)
- TRM evaluation fails (TRM's garbage looks "uncorrupted" to CNN)

Potential fixes:
1. Explicit comparison: input_embed XOR output_embed, or difference
2. Contrastive loss: (correct_input, correct_output) vs (wrong_input, correct_output)  
3. Input-shuffled negatives during training
4. Cross-attention between input and output before U-Net
""")


if __name__ == "__main__":
    main()