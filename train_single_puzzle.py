"""
Training script for single or multi-puzzle experiments (1-3 puzzles).
Works on CPU/MPS (Mac) without CUDA.

Usage:
    # Random single puzzle (auto-discovers from data root):
    python train_single_puzzle.py --epochs 1000

    # Random 2 puzzles:
    python train_single_puzzle.py --num-puzzles 2 --epochs 1000

    # Specific puzzles:
    python train_single_puzzle.py --data-dirs data/single_puzzle_007bbfb7 --epochs 1000

    # With actual TRM:
    python train_single_puzzle.py --num-puzzles 2 --model actual_trm --epochs 1000
"""

import argparse
import json
import os
import random
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Determine device and dtype
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    DTYPE = torch.bfloat16  # CUDA supports bfloat16
else:
    if torch.backends.mps.is_available():
        DEVICE = torch.device("mps")
    else:
        DEVICE = torch.device("cpu")
    DTYPE = torch.float32  # MPS/CPU need float32

print(f"Using device: {DEVICE}, dtype: {DTYPE}")


def discover_puzzle_dirs(data_root: str) -> list:
    """Find all valid puzzle directories under data_root.

    A valid puzzle directory has train/ and test/ subdirs with all__inputs.npy files.
    """
    valid_dirs = []

    if not os.path.isdir(data_root):
        return valid_dirs

    for name in os.listdir(data_root):
        candidate = os.path.join(data_root, name)
        if not os.path.isdir(candidate):
            continue

        # Check for required structure
        train_inputs = os.path.join(candidate, "train", "all__inputs.npy")
        test_inputs = os.path.join(candidate, "test", "all__inputs.npy")

        if os.path.exists(train_inputs) and os.path.exists(test_inputs):
            valid_dirs.append(candidate)

    return sorted(valid_dirs)


class MultiPuzzleDataset(Dataset):
    """Dataset that can load from 1-3 puzzle directories."""

    def __init__(self, data_dirs: list, split: str, puzzle_id_offset: int = 0):
        """
        Args:
            data_dirs: List of data directories (1-3 puzzles)
            split: 'train' or 'test'
            puzzle_id_offset: Starting puzzle ID (usually 1, since 0 is reserved)
        """
        all_inputs = []
        all_labels = []
        all_puzzle_ids = []

        for puzzle_idx, data_dir in enumerate(data_dirs):
            inputs = np.load(os.path.join(data_dir, split, "all__inputs.npy"))
            labels = np.load(os.path.join(data_dir, split, "all__labels.npy"))
            puzzle_indices = np.load(os.path.join(data_dir, split, "all__puzzle_indices.npy"))

            # Expand puzzle_ids to match examples
            example_puzzle_ids = np.zeros(len(inputs), dtype=np.int32)
            for i in range(len(puzzle_indices) - 1):
                start, end = puzzle_indices[i], puzzle_indices[i + 1]
                # Assign unique puzzle ID: offset + puzzle_idx + 1
                # Each puzzle directory gets a unique ID
                example_puzzle_ids[start:end] = puzzle_id_offset + puzzle_idx + 1

            all_inputs.append(inputs)
            all_labels.append(labels)
            all_puzzle_ids.append(example_puzzle_ids)

        # Concatenate all puzzles
        self.inputs = np.concatenate(all_inputs, axis=0)
        self.labels = np.concatenate(all_labels, axis=0)
        self.example_puzzle_ids = np.concatenate(all_puzzle_ids, axis=0)

        # Track which examples belong to which puzzle (for per-puzzle metrics)
        self.puzzle_boundaries = []
        offset = 0
        for inputs in all_inputs:
            self.puzzle_boundaries.append((offset, offset + len(inputs)))
            offset += len(inputs)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return {
            "inputs": torch.tensor(self.inputs[idx], dtype=torch.long),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
            "puzzle_identifiers": torch.tensor(self.example_puzzle_ids[idx], dtype=torch.long),
        }

    def get_puzzle_indices(self, puzzle_num: int):
        """Get indices for examples from a specific puzzle (0-indexed)."""
        if puzzle_num >= len(self.puzzle_boundaries):
            return []
        start, end = self.puzzle_boundaries[puzzle_num]
        return list(range(start, end))


class SimpleTRM(nn.Module):
    """Simplified TRM for single-puzzle experiment."""

    def __init__(self, vocab_size=12, hidden_size=128, num_puzzle_ids=2, seq_len=900,
                 num_layers=2, num_heads=4, H_cycles=2, L_cycles=3):
        super().__init__()

        self.hidden_size = hidden_size
        self.seq_len = seq_len
        self.H_cycles = H_cycles
        self.L_cycles = L_cycles

        # Embeddings
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.embed_puzzle = nn.Embedding(num_puzzle_ids, hidden_size)
        self.embed_pos = nn.Embedding(seq_len, hidden_size)

        # Transformer layers (shared for both H and L updates)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output head
        self.lm_head = nn.Linear(hidden_size, vocab_size)

        # Initial states
        self.H_init = nn.Parameter(torch.randn(hidden_size) * 0.02)
        self.L_init = nn.Parameter(torch.randn(hidden_size) * 0.02)

    def forward(self, inputs, puzzle_identifiers):
        B, L = inputs.shape

        # Embeddings
        tok_emb = self.embed_tokens(inputs)  # (B, L, H)
        pos_emb = self.embed_pos(torch.arange(L, device=inputs.device))  # (L, H)
        puzzle_emb = self.embed_puzzle(puzzle_identifiers)  # (B, H)

        # Input embedding = tokens + position + puzzle (broadcast)
        input_emb = tok_emb + pos_emb.unsqueeze(0) + puzzle_emb.unsqueeze(1)

        # Initialize latent states
        z_H = self.H_init.unsqueeze(0).unsqueeze(0).expand(B, L, -1)
        z_L = self.L_init.unsqueeze(0).unsqueeze(0).expand(B, L, -1)

        # Recursive reasoning
        for _h in range(self.H_cycles):
            for _l in range(self.L_cycles):
                # L update: z_L = f(z_L + z_H + input)
                z_L = self.transformer(z_L + z_H + input_emb)
            # H update: z_H = f(z_H + z_L)
            z_H = self.transformer(z_H + z_L)

        # Output
        logits = self.lm_head(z_H)
        return logits


class SimpleMLP(nn.Module):
    """Simple MLP baseline - can it memorize a single puzzle?"""

    def __init__(self, vocab_size=12, hidden_size=128, num_puzzle_ids=2, seq_len=900, **kwargs):
        super().__init__()
        self.vocab_size = vocab_size
        self.seq_len = seq_len

        # Simple embedding + MLP approach
        self.embed_tokens = nn.Embedding(vocab_size, 32)

        # MLP: takes flattened embedded input, outputs logits for each position
        self.mlp = nn.Sequential(
            nn.Linear(seq_len * 32, hidden_size * 4),
            nn.ReLU(),
            nn.Linear(hidden_size * 4, hidden_size * 4),
            nn.ReLU(),
            nn.Linear(hidden_size * 4, seq_len * vocab_size),
        )

    def forward(self, inputs, puzzle_identifiers):
        B, L = inputs.shape

        # Embed and flatten
        x = self.embed_tokens(inputs)  # (B, L, 32)
        x = x.view(B, -1)  # (B, L*32)

        # MLP
        x = self.mlp(x)  # (B, L*vocab_size)

        # Reshape to (B, L, vocab_size)
        logits = x.view(B, L, self.vocab_size)
        return logits


class SimpleCNN(nn.Module):
    """Simple CNN baseline - uses 2D spatial structure."""

    def __init__(self, vocab_size=12, hidden_size=128, num_puzzle_ids=2, seq_len=900, **kwargs):
        super().__init__()
        self.vocab_size = vocab_size
        self.grid_size = 30  # 30x30 grid

        # Embedding for input tokens
        self.embed_tokens = nn.Embedding(vocab_size, 32)

        # CNN layers
        self.conv = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, vocab_size, kernel_size=1),  # 1x1 conv to get per-pixel logits
        )

    def forward(self, inputs, puzzle_identifiers):
        B, L = inputs.shape

        # Embed tokens
        x = self.embed_tokens(inputs)  # (B, 900, 32)

        # Reshape to 2D grid: (B, 32, 30, 30)
        x = x.view(B, self.grid_size, self.grid_size, -1)
        x = x.permute(0, 3, 1, 2)  # (B, 32, 30, 30)

        # Apply CNN
        x = self.conv(x)  # (B, vocab_size, 30, 30)

        # Reshape back to sequence
        x = x.permute(0, 2, 3, 1)  # (B, 30, 30, vocab_size)
        logits = x.reshape(B, L, self.vocab_size)

        return logits


class ActualTRMWrapper(nn.Module):
    """Wrapper around the actual TRM from trm.py for simple interface."""

    def __init__(self, vocab_size=12, hidden_size=128, num_puzzle_ids=2, seq_len=900,
                 num_layers=2, num_heads=4, H_cycles=3, L_cycles=4, batch_size=16, **kwargs):
        super().__init__()

        # Import the actual TRM
        from models.recursive_reasoning.trm import TinyRecursiveReasoningModel_ACTV1

        # Determine dtype string based on global DTYPE
        dtype_str = "float32" if DTYPE == torch.float32 else "bfloat16"

        # Build config for actual TRM
        config_dict = {
            "batch_size": batch_size,
            "seq_len": seq_len,
            "puzzle_emb_ndim": hidden_size,  # Use puzzle embeddings
            "num_puzzle_identifiers": num_puzzle_ids,
            "vocab_size": vocab_size,
            "H_cycles": H_cycles,
            "L_cycles": L_cycles,
            "H_layers": 1,  # Ignored per docs
            "L_layers": num_layers,
            "hidden_size": hidden_size,
            "expansion": 4.0,
            "num_heads": num_heads,
            "pos_encodings": "rope",  # Use RoPE embeddings
            "rms_norm_eps": 1e-5,
            "rope_theta": 10000.0,
            "halt_max_steps": 1,  # No halting for simplicity
            "halt_exploration_prob": 0.0,
            "forward_dtype": dtype_str,
            "mlp_t": False,  # Use attention, not MLP transpose
            "puzzle_emb_len": 16,
            "no_ACT_continue": True,
        }

        self.trm = TinyRecursiveReasoningModel_ACTV1(config_dict)
        self.trm = self.trm.to(DTYPE)

    def forward(self, inputs, puzzle_identifiers):
        """Simple forward interface matching other models."""
        actual_batch_size = inputs.shape[0]
        expected_batch_size = self.trm.config.batch_size

        # Pad batch if smaller than expected (sparse embedding requires fixed size)
        if actual_batch_size < expected_batch_size:
            pad_size = expected_batch_size - actual_batch_size
            inputs = torch.cat([inputs, inputs[:1].expand(pad_size, -1)], dim=0)
            puzzle_identifiers = torch.cat([puzzle_identifiers, puzzle_identifiers[:1].expand(pad_size)], dim=0)

        batch = {
            "inputs": inputs,
            "puzzle_identifiers": puzzle_identifiers,
        }

        # Get initial carry
        carry = self.trm.initial_carry(batch)

        # Move carry to device
        carry.inner_carry.z_H = carry.inner_carry.z_H.to(DEVICE)
        carry.inner_carry.z_L = carry.inner_carry.z_L.to(DEVICE)
        carry.steps = carry.steps.to(DEVICE)
        carry.halted = carry.halted.to(DEVICE)
        carry.current_data = {k: v.to(DEVICE) for k, v in carry.current_data.items()}

        # Forward pass
        _, outputs = self.trm(carry, batch)

        # Return only the actual batch (remove padding)
        return outputs["logits"][:actual_batch_size]


def find_grid_bounds(grid):
    """Find actual grid boundaries (non-padding area)."""
    # Tokens: PAD=0, EOS=1, digits=2-11
    mask = (grid >= 2) & (grid <= 11)
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not rows.any():
        return 0, 5, 0, 5
    r_min, r_max = np.where(rows)[0][[0, -1]]
    c_min, c_max = np.where(cols)[0][[0, -1]]
    return r_min, r_max + 1, c_min, c_max + 1


def grid_to_str(grid, r1, r2, c1, c2):
    """Convert grid to string for display."""
    lines = []
    for r in range(r1, min(r2, r1 + 12)):  # Limit display
        row = []
        for c in range(c1, min(c2, c1 + 12)):
            val = grid[r, c]
            if val == 0:
                row.append(".")
            elif val == 1:
                row.append("|")  # EOS
            else:
                row.append(str(val - 2))  # Convert back to 0-9
        lines.append(" ".join(row))
    return "\n".join(lines)


def visualize_prediction(model, test_loader, epoch, test_acc, debug=False):
    """Show side-by-side comparison of prediction vs ground truth."""
    model.eval()
    with torch.no_grad():
        test_batch = next(iter(test_loader))
        inputs = test_batch["inputs"][:1].to(DEVICE)
        labels = test_batch["labels"][:1].to(DEVICE)
        puzzle_ids = test_batch["puzzle_identifiers"][:1].to(DEVICE)

        logits = model(inputs, puzzle_ids)
        preds = logits.argmax(dim=-1)

        # Debug info
        if debug or epoch <= 5000:
            tqdm.write(f"\n[DEBUG] Logits shape: {logits.shape}")
            tqdm.write(f"[DEBUG] Logits min/max: {logits.min().item():.3f} / {logits.max().item():.3f}")
            tqdm.write(f"[DEBUG] Logits mean/std: {logits.mean().item():.3f} / {logits.std().item():.3f}")

            # Show distribution of predictions
            pred_flat = preds[0].cpu().numpy().flatten()
            unique, counts = np.unique(pred_flat, return_counts=True)
            tqdm.write(f"[DEBUG] Prediction distribution: {dict(zip(unique.tolist(), counts.tolist()))}")

            # Show softmax probabilities for a non-padding position
            label_flat = labels[0].cpu().numpy().flatten()
            non_pad_idx = np.where(label_flat != 0)[0]
            if len(non_pad_idx) > 0:
                idx = non_pad_idx[0]
                probs = torch.softmax(logits[0, idx].float(), dim=-1).cpu().numpy()
                tqdm.write(f"[DEBUG] Softmax at pos {idx}: {probs.round(3)}")
            if len(non_pad_idx) > 20:
                idx = non_pad_idx[20]
                probs = torch.softmax(logits[0, idx].float(), dim=-1).cpu().numpy()
                tqdm.write(f"[DEBUG] Softmax at pos {idx}: {probs.round(3)}")

        # Convert to grids (30x30)
        label_grid = labels[0].cpu().numpy().reshape(30, 30)
        pred_grid = preds[0].cpu().numpy().reshape(30, 30)

        r1, r2, c1, c2 = find_grid_bounds(label_grid)

        # Token-level comparison
        match = (pred_grid == label_grid)
        valid = (label_grid != 0)
        correct = (match & valid).sum()
        total = valid.sum()

        # Build side-by-side display
        gt_lines = grid_to_str(label_grid, r1, r2, c1, c2).split("\n")
        pred_lines = grid_to_str(pred_grid, r1, r2, c1, c2).split("\n")

        tqdm.write(f"\n{'─'*50}")
        tqdm.write(f"EPOCH {epoch} | Test Acc: {test_acc:.2%} | Token: {correct}/{total}")
        tqdm.write(f"{'─'*50}")
        tqdm.write(f"{'GROUND TRUTH':<25} {'MODEL PREDICTION':<25}")
        tqdm.write(f"{'─'*50}")

        max_lines = max(len(gt_lines), len(pred_lines))
        for i in range(max_lines):
            gt = gt_lines[i] if i < len(gt_lines) else ""
            pr = pred_lines[i] if i < len(pred_lines) else ""
            # Mark differences
            tqdm.write(f"{gt:<25} {pr:<25}")

        tqdm.write(f"{'─'*50}\n")


def train(args):
    num_puzzles = len(args.data_dirs)
    print(f"\nLoading {num_puzzles} puzzle(s)...")

    # Load metadata from each puzzle directory
    all_metadata = []
    puzzle_names = []
    for data_dir in args.data_dirs:
        with open(os.path.join(data_dir, "train", "dataset.json")) as f:
            metadata = json.load(f)
            all_metadata.append(metadata)
        # Extract puzzle name from directory
        puzzle_names.append(os.path.basename(data_dir.rstrip("/")))

    # Combine metadata (use max values for compatibility)
    combined_metadata = {
        "vocab_size": max(m["vocab_size"] for m in all_metadata),
        "seq_len": max(m["seq_len"] for m in all_metadata),
        # Number of puzzle IDs = 1 (reserved) + num_puzzles
        "num_puzzle_identifiers": num_puzzles + 1,
    }

    print(f"Puzzles: {puzzle_names}")
    print(f"Combined metadata: {combined_metadata}")

    # Datasets
    train_dataset = MultiPuzzleDataset(args.data_dirs, "train")
    test_dataset = MultiPuzzleDataset(args.data_dirs, "test")

    print(f"Training examples: {len(train_dataset)} (across {num_puzzles} puzzle(s))")
    print(f"Test examples: {len(test_dataset)}")
    for i, name in enumerate(puzzle_names):
        train_count = len(train_dataset.get_puzzle_indices(i))
        test_count = len(test_dataset.get_puzzle_indices(i))
        print(f"  Puzzle {i+1} ({name}): {train_count} train, {test_count} test")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # Model selection
    model_classes = {
        "trm": SimpleTRM,
        "mlp": SimpleMLP,
        "cnn": SimpleCNN,
        "actual_trm": ActualTRMWrapper,
    }

    if args.model not in model_classes:
        raise ValueError(f"Unknown model: {args.model}. Choose from {list(model_classes.keys())}")

    model = model_classes[args.model](
        vocab_size=combined_metadata["vocab_size"],
        hidden_size=args.hidden_size,
        num_puzzle_ids=combined_metadata["num_puzzle_identifiers"],
        seq_len=combined_metadata["seq_len"],
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        H_cycles=args.H_cycles,
        L_cycles=args.L_cycles,
        batch_size=args.batch_size,  # Needed for actual TRM
    ).to(DEVICE)

    print(f"Model: {args.model.upper()}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # Training loop
    best_test_acc = 0.0
    best_test_epoch = 0

    print("\n" + "="*60)
    print("TRAINING START")
    print("="*60)
    print(f"Training on {len(train_dataset)} examples from {num_puzzles} puzzle(s)")
    print(f"Testing on {len(test_dataset)} examples (held-out test inputs)")
    print(f"Puzzle IDs: 1-{num_puzzles} (0 reserved)")
    print("="*60)

    # Debug: Check data
    print("\n[DEBUG] Checking training data...")
    sample = train_dataset[0]
    print(f"  Input shape: {sample['inputs'].shape}")
    print(f"  Label shape: {sample['labels'].shape}")
    print(f"  Puzzle ID: {sample['puzzle_identifiers']}")
    print(f"  Input unique values: {torch.unique(sample['inputs']).tolist()}")
    print(f"  Label unique values: {torch.unique(sample['labels']).tolist()}")
    print(f"  Non-zero labels: {(sample['labels'] != 0).sum().item()} / {sample['labels'].numel()}")

    print("\n[DEBUG] Checking test data...")
    test_sample = test_dataset[0]
    print(f"  Input shape: {test_sample['inputs'].shape}")
    print(f"  Label shape: {test_sample['labels'].shape}")
    print(f"  Puzzle ID: {test_sample['puzzle_identifiers']}")
    print(f"  Input unique values: {torch.unique(test_sample['inputs']).tolist()}")
    print(f"  Label unique values: {torch.unique(test_sample['labels']).tolist()}")
    print(f"  Non-zero labels: {(test_sample['labels'] != 0).sum().item()} / {test_sample['labels'].numel()}")
    print("="*60 + "\n")

    pbar = tqdm(range(args.epochs), desc="Training", unit="epoch")

    for epoch in pbar:
        model.train()
        total_loss = 0.0
        total_correct = 0
        total_tokens = 0

        for batch in train_loader:
            inputs = batch["inputs"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)
            puzzle_ids = batch["puzzle_identifiers"].to(DEVICE)

            optimizer.zero_grad()

            logits = model(inputs, puzzle_ids)

            # Loss (ignore padding = 0)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                labels.reshape(-1),
                ignore_index=0
            )

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            # Accuracy (non-padding tokens)
            preds = logits.argmax(dim=-1)
            mask = labels != 0
            total_correct += ((preds == labels) & mask).sum().item()
            total_tokens += mask.sum().item()

        train_acc = total_correct / total_tokens if total_tokens > 0 else 0
        avg_loss = total_loss / len(train_loader)

        # Update progress bar
        pbar.set_postfix({
            "loss": f"{avg_loss:.4f}",
            "train_acc": f"{train_acc:.2%}",
            "best_test": f"{best_test_acc:.2%}"
        })

        # Evaluate on test set
        if (epoch + 1) % args.eval_interval == 0 or epoch == args.epochs - 1:
            model.eval()
            test_correct = 0
            test_tokens = 0

            with torch.no_grad():
                for batch in test_loader:
                    inputs = batch["inputs"].to(DEVICE)
                    labels = batch["labels"].to(DEVICE)
                    puzzle_ids = batch["puzzle_identifiers"].to(DEVICE)

                    logits = model(inputs, puzzle_ids)
                    preds = logits.argmax(dim=-1)

                    mask = labels != 0
                    test_correct += ((preds == labels) & mask).sum().item()
                    test_tokens += mask.sum().item()

            test_acc = test_correct / test_tokens if test_tokens > 0 else 0

            if test_acc > best_test_acc:
                best_test_acc = test_acc
                best_test_epoch = epoch + 1

            # Show visual comparison
            visualize_prediction(model, test_loader, epoch + 1, test_acc)

    print(f"\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    print(f"Best Test Accuracy: {best_test_acc:.2%} (at epoch {best_test_epoch})")

    if best_test_acc > 0.95:
        print("\n✓ SUCCESS: Model learned the puzzle from demonstrations!")
        print("  This confirms the transductive learning hypothesis.")
    elif best_test_acc > 0.7:
        print("\n~ PARTIAL: Model learned something but not perfectly.")
    else:
        print("\n✗ FAILURE: Model did not learn the puzzle.")

    print("="*60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dirs", type=str, nargs="+", default=None,
                        help="Data directories for puzzles (optional - will auto-discover if not provided)")
    parser.add_argument("--data-root", type=str, default="data",
                        help="Root directory to search for puzzles (default: data/)")
    parser.add_argument("--num-puzzles", type=int, default=1,
                        help="Number of random puzzles to select (1-3, default: 1)")
    parser.add_argument("--model", type=str, default="trm", choices=["trm", "mlp", "cnn", "actual_trm"],
                        help="Model architecture: trm (simplified), mlp (simple), cnn (spatial), actual_trm (real TRM)")
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--H-cycles", type=int, default=2)
    parser.add_argument("--L-cycles", type=int, default=3)
    parser.add_argument("--eval-interval", type=int, default=1)
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for puzzle selection (for reproducibility)")

    args = parser.parse_args()

    # Validate num-puzzles
    if args.num_puzzles < 1 or args.num_puzzles > 3:
        parser.error("--num-puzzles must be between 1 and 3")

    # If data-dirs not provided, auto-discover and randomly select
    if args.data_dirs is None:
        available = discover_puzzle_dirs(args.data_root)
        if not available:
            parser.error(f"No valid puzzle directories found in '{args.data_root}'. "
                        "Use --data-dirs to specify manually or --data-root to change search path.")

        if len(available) < args.num_puzzles:
            parser.error(f"Requested {args.num_puzzles} puzzles but only {len(available)} available in '{args.data_root}'")

        # Set seed if provided
        if args.seed is not None:
            random.seed(args.seed)

        # Randomly select puzzles
        args.data_dirs = random.sample(available, args.num_puzzles)
        print(f"Auto-selected {args.num_puzzles} puzzle(s) from {len(available)} available:")
        for d in args.data_dirs:
            print(f"  - {os.path.basename(d)}")
    else:
        # Validate manually specified dirs
        if len(args.data_dirs) > 3:
            parser.error("Maximum 3 puzzle directories supported")

    train(args)
