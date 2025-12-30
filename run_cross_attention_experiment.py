#!/usr/bin/env python3
"""
Cross-Attention and Color Embeddings Experiment Script

Tests how cross-attention configuration and color embedding choices interact
in the TRM-ARC model on puzzle 009d5c81.

Combinations tested:
- cross-attention-position: early, late
- cross-attention-heads: 1, 4, 8
- color embeddings: learned (default), one-hot

Constraint: For one-hot + early attention, only 1 head is valid.

Each combination runs 5 times with seeds 42-46.
"""

import subprocess
import re
import os
import sys
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import statistics

# =============================================================================
# Configuration
# =============================================================================

COMBINATIONS = [
    {"position": "late", "heads": 1, "onehot": False, "name": "late-1-learned"},
    {"position": "late", "heads": 4, "onehot": False, "name": "late-4-learned"},
    {"position": "late", "heads": 8, "onehot": False, "name": "late-8-learned"},
    {"position": "late", "heads": 1, "onehot": True, "name": "late-1-onehot"},
    {"position": "late", "heads": 4, "onehot": True, "name": "late-4-onehot"},
    {"position": "late", "heads": 8, "onehot": True, "name": "late-8-onehot"},
    {"position": "early", "heads": 1, "onehot": False, "name": "early-1-learned"},
    {"position": "early", "heads": 4, "onehot": False, "name": "early-4-learned"},
    {"position": "early", "heads": 8, "onehot": False, "name": "early-8-learned"},
    {"position": "early", "heads": 1, "onehot": True, "name": "early-1-onehot"},
]

SEEDS = [42, 43, 44, 45, 46]
OUTPUT_DIR = "experiment_results"
PUZZLE_ID = "009d5c81"


# =============================================================================
# Helper Functions
# =============================================================================

def strip_ansi(text: str) -> str:
    """Remove ANSI escape codes from text."""
    ansi_pattern = re.compile(r'\x1b\[[0-9;]*m')
    return ansi_pattern.sub('', text)


def build_command(combo: Dict, seed: int) -> List[str]:
    """Build the crm.py command for a given combination and seed."""
    cmd = [
        sys.executable, "crm.py",
        "--single-puzzle", PUZZLE_ID,
        "--epochs", "50",
        "--num-negatives", "200",
        "--eval-on-test",
        "--visualize",
        "--negative-type", "all_zeros",
        "--ablation",
        "--num-layers", "0",
        "--hidden-dim", "8",
        "--kernel-size", "3",
        "--conv-depth", "3",
        "--out-kernel-size", "3",
        "--use-cross-attention",
        "--use-untrained-ir",
        "--ir-hidden-dim", "8",
        "--ir-out-dim", "16",
        "--ir-num-layers", "3",
        "--dihedral-only",
        "--cross-attention-position", combo["position"],
        "--cross-attention-heads", str(combo["heads"]),
        "--seed", str(seed),
    ]

    if combo["onehot"]:
        cmd.append("--use-onehot")

    return cmd


def parse_results(output: str) -> Dict:
    """Parse test results from crm.py output."""
    results = {
        "pixel_accuracy": None,
        "perfect_count": None,
        "total_examples": None,
        "perfect_rate": None,
        "final_epoch": None,
        "early_stopped": False,
    }

    # Find pixel accuracy: "  Pixel Accuracy: 94.56%"
    acc_match = re.search(r"Pixel Accuracy:\s*([\d.]+)%", output)
    if acc_match:
        results["pixel_accuracy"] = float(acc_match.group(1))

    # Find perfect examples: "  Perfect Examples: 1/1 (100.00%)"
    perfect_match = re.search(r"Perfect Examples:\s*(\d+)/(\d+)", output)
    if perfect_match:
        results["perfect_count"] = int(perfect_match.group(1))
        results["total_examples"] = int(perfect_match.group(2))
        if results["total_examples"] > 0:
            results["perfect_rate"] = results["perfect_count"] / results["total_examples"]

    # Check for early stopping: "Early stopping: reached 100% validation accuracy at epoch 5"
    early_stop_match = re.search(r"Early stopping.*epoch\s*(\d+)", output)
    if early_stop_match:
        results["early_stopped"] = True
        results["final_epoch"] = int(early_stop_match.group(1))
    else:
        # Find last epoch: "Epoch 50/50"
        epoch_matches = re.findall(r"Epoch\s*(\d+)/\d+", output)
        if epoch_matches:
            results["final_epoch"] = int(epoch_matches[-1])

    return results


def run_trial(combo: Dict, seed: int, trial_num: int, total_trials: int) -> Tuple[str, Dict]:
    """Run a single trial and capture output."""
    cmd = build_command(combo, seed)

    print(f"  [{trial_num}/{total_trials}] Running {combo['name']} with seed {seed}...", end=" ", flush=True)

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,  # 10 minute timeout per trial
        )
        output = result.stdout + "\n" + result.stderr
        parsed = parse_results(output)

        # Print quick status
        if parsed["pixel_accuracy"] is not None:
            perfect = "PERFECT" if parsed["perfect_rate"] == 1.0 else f"{parsed['pixel_accuracy']:.1f}%"
            epoch_info = f"epoch {parsed['final_epoch']}"
            if parsed["early_stopped"]:
                epoch_info += " (early stop)"
            print(f"{perfect} @ {epoch_info}")
        else:
            print("PARSE ERROR")

        return output, parsed

    except subprocess.TimeoutExpired:
        print("TIMEOUT")
        return "TIMEOUT: Trial exceeded 10 minutes", {
            "pixel_accuracy": None,
            "perfect_count": None,
            "total_examples": None,
            "perfect_rate": None,
            "final_epoch": None,
            "early_stopped": False,
            "error": "timeout"
        }
    except Exception as e:
        print(f"ERROR: {e}")
        return f"ERROR: {e}", {
            "pixel_accuracy": None,
            "perfect_count": None,
            "total_examples": None,
            "perfect_rate": None,
            "final_epoch": None,
            "early_stopped": False,
            "error": str(e)
        }


def classify_stability(accuracies: List[float]) -> str:
    """Classify stability based on standard deviation of pixel accuracies."""
    if len(accuracies) < 2:
        return "Unknown"

    # Filter out None values
    valid_accs = [a for a in accuracies if a is not None]
    if len(valid_accs) < 2:
        return "Unknown"

    std_dev = statistics.stdev(valid_accs)

    if std_dev < 2.0:
        return "Stable"
    elif std_dev < 10.0:
        return "Variable"
    else:
        return "Oscillates wildly"


def generate_detailed_report(all_results: Dict, output_path: str):
    """Generate detailed report with all trial outputs."""
    with open(output_path, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("CROSS-ATTENTION EXPERIMENT - DETAILED RESULTS\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Puzzle: {PUZZLE_ID}\n")
        f.write("=" * 80 + "\n\n")

        for combo in COMBINATIONS:
            name = combo["name"]
            f.write("\n" + "=" * 80 + "\n")
            f.write(f"COMBINATION: {name}\n")
            f.write(f"  Position: {combo['position']}\n")
            f.write(f"  Heads: {combo['heads']}\n")
            f.write(f"  Embedding: {'one-hot' if combo['onehot'] else 'learned'}\n")
            f.write("=" * 80 + "\n")

            if name not in all_results:
                f.write("NO RESULTS\n")
                continue

            for seed in SEEDS:
                if seed not in all_results[name]:
                    continue

                trial = all_results[name][seed]
                f.write(f"\n{'-' * 60}\n")
                f.write(f"TRIAL: seed={seed}\n")
                f.write(f"{'-' * 60}\n")

                # Write parsed results
                parsed = trial["parsed"]
                f.write(f"Pixel Accuracy: {parsed['pixel_accuracy']}%\n")
                f.write(f"Perfect: {parsed['perfect_count']}/{parsed['total_examples']}\n")
                f.write(f"Final Epoch: {parsed['final_epoch']}\n")
                f.write(f"Early Stopped: {parsed['early_stopped']}\n\n")

                # Write full output (ANSI stripped)
                f.write("--- FULL OUTPUT ---\n")
                f.write(strip_ansi(trial["output"]))
                f.write("\n")


def generate_summary_report(all_results: Dict, output_path: str):
    """Generate concise summary report."""

    # Calculate statistics for each combination
    combo_stats = {}
    for combo in COMBINATIONS:
        name = combo["name"]
        if name not in all_results:
            continue

        accuracies = []
        perfect_trials = 0
        total_trials = 0
        epochs = []

        for seed in SEEDS:
            if seed not in all_results[name]:
                continue

            parsed = all_results[name][seed]["parsed"]
            total_trials += 1

            if parsed["pixel_accuracy"] is not None:
                accuracies.append(parsed["pixel_accuracy"])

            if parsed["perfect_rate"] == 1.0:
                perfect_trials += 1

            if parsed["final_epoch"] is not None:
                epochs.append(parsed["final_epoch"])

        combo_stats[name] = {
            "combo": combo,
            "perfect_trials": perfect_trials,
            "total_trials": total_trials,
            "accuracies": accuracies,
            "avg_accuracy": statistics.mean(accuracies) if accuracies else None,
            "std_accuracy": statistics.stdev(accuracies) if len(accuracies) > 1 else 0,
            "stability": classify_stability(accuracies),
            "avg_epoch": statistics.mean(epochs) if epochs else None,
        }

    # Categorize combinations
    perfect_always = []
    perfect_sometimes = []
    perfect_never = []

    for name, stats in combo_stats.items():
        if stats["perfect_trials"] == stats["total_trials"] and stats["total_trials"] > 0:
            perfect_always.append(name)
        elif stats["perfect_trials"] > 0:
            perfect_sometimes.append((name, stats["perfect_trials"], stats["total_trials"]))
        else:
            perfect_never.append(name)

    # Categorize by stability
    stable = []
    variable = []
    oscillates = []

    for name, stats in combo_stats.items():
        if stats["stability"] == "Stable":
            stable.append(name)
        elif stats["stability"] == "Variable":
            variable.append(name)
        elif stats["stability"] == "Oscillates wildly":
            oscillates.append(name)

    # Write report
    with open(output_path, "w") as f:
        f.write("# Cross-Attention Experiment Summary\n\n")
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  \n")
        f.write(f"**Puzzle:** {PUZZLE_ID}  \n")
        f.write(f"**Trials per combination:** {len(SEEDS)}  \n")
        f.write(f"**Seeds:** {', '.join(map(str, SEEDS))}\n\n")

        # Quick results table
        f.write("## Quick Results\n\n")
        f.write("| Position | Heads | Embedding | Perfect Rate | Avg Accuracy | Std Dev | Stability | Avg Epoch |\n")
        f.write("|----------|-------|-----------|--------------|--------------|---------|-----------|----------|\n")

        for combo in COMBINATIONS:
            name = combo["name"]
            if name not in combo_stats:
                continue

            stats = combo_stats[name]
            perfect_str = f"{stats['perfect_trials']}/{stats['total_trials']}"
            avg_acc = f"{stats['avg_accuracy']:.1f}%" if stats['avg_accuracy'] else "N/A"
            std_acc = f"{stats['std_accuracy']:.1f}%" if stats['std_accuracy'] else "N/A"
            avg_ep = f"{stats['avg_epoch']:.1f}" if stats['avg_epoch'] else "N/A"

            f.write(f"| {combo['position']} | {combo['heads']} | {'one-hot' if combo['onehot'] else 'learned'} | ")
            f.write(f"{perfect_str} | {avg_acc} | {std_acc} | {stats['stability']} | {avg_ep} |\n")

        # Categories
        f.write("\n## Categories\n\n")

        f.write("### Perfect Every Time (5/5)\n")
        if perfect_always:
            for name in perfect_always:
                f.write(f"- {name}\n")
        else:
            f.write("- *None*\n")

        f.write("\n### Perfect Sometimes (1-4/5)\n")
        if perfect_sometimes:
            for name, count, total in perfect_sometimes:
                f.write(f"- {name} ({count}/{total})\n")
        else:
            f.write("- *None*\n")

        f.write("\n### Never Perfect (0/5)\n")
        if perfect_never:
            for name in perfect_never:
                f.write(f"- {name}\n")
        else:
            f.write("- *None*\n")

        # Stability analysis
        f.write("\n## Stability Analysis\n\n")

        f.write("### Stable (std < 2%)\n")
        if stable:
            for name in stable:
                stats = combo_stats[name]
                f.write(f"- {name} (std: {stats['std_accuracy']:.2f}%)\n")
        else:
            f.write("- *None*\n")

        f.write("\n### Variable (std 2-10%)\n")
        if variable:
            for name in variable:
                stats = combo_stats[name]
                f.write(f"- {name} (std: {stats['std_accuracy']:.2f}%)\n")
        else:
            f.write("- *None*\n")

        f.write("\n### Oscillates Wildly (std > 10%)\n")
        if oscillates:
            for name in oscillates:
                stats = combo_stats[name]
                f.write(f"- {name} (std: {stats['std_accuracy']:.2f}%)\n")
        else:
            f.write("- *None*\n")

        # Insights
        f.write("\n## Insights\n\n")

        # Position comparison
        late_perfect = sum(1 for n in perfect_always if n.startswith("late"))
        early_perfect = sum(1 for n in perfect_always if n.startswith("early"))
        f.write(f"**Position:** Late attention has {late_perfect} always-perfect configs, ")
        f.write(f"early attention has {early_perfect}.\n\n")

        # Head count comparison
        for h in [1, 4, 8]:
            h_perfect = sum(1 for n in perfect_always if f"-{h}-" in n)
            f.write(f"**{h} head(s):** {h_perfect} always-perfect configs\n")
        f.write("\n")

        # Embedding comparison
        learned_perfect = sum(1 for n in perfect_always if n.endswith("learned"))
        onehot_perfect = sum(1 for n in perfect_always if n.endswith("onehot"))
        f.write(f"**Embeddings:** Learned has {learned_perfect} always-perfect, ")
        f.write(f"one-hot has {onehot_perfect}.\n\n")

        # Best and worst
        if combo_stats:
            best_avg = max(combo_stats.items(), key=lambda x: x[1]['avg_accuracy'] or 0)
            worst_avg = min(combo_stats.items(), key=lambda x: x[1]['avg_accuracy'] or 100)
            f.write(f"**Best avg accuracy:** {best_avg[0]} ({best_avg[1]['avg_accuracy']:.1f}%)\n")
            f.write(f"**Worst avg accuracy:** {worst_avg[0]} ({worst_avg[1]['avg_accuracy']:.1f}%)\n")


def main():
    """Main experiment runner."""
    print("=" * 60)
    print("CROSS-ATTENTION EXPERIMENT")
    print(f"Puzzle: {PUZZLE_ID}")
    print(f"Combinations: {len(COMBINATIONS)}")
    print(f"Seeds: {SEEDS}")
    print(f"Total trials: {len(COMBINATIONS) * len(SEEDS)}")
    print("=" * 60)

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Store all results
    all_results = {}

    total_trials = len(COMBINATIONS) * len(SEEDS)
    trial_num = 0

    # Run all trials
    for combo in COMBINATIONS:
        name = combo["name"]
        print(f"\n--- {name} ---")
        all_results[name] = {}

        for seed in SEEDS:
            trial_num += 1
            output, parsed = run_trial(combo, seed, trial_num, total_trials)
            all_results[name][seed] = {
                "output": output,
                "parsed": parsed,
            }

    # Generate reports
    print("\n" + "=" * 60)
    print("Generating reports...")

    detailed_path = os.path.join(OUTPUT_DIR, "detailed_results.txt")
    summary_path = os.path.join(OUTPUT_DIR, "summary.md")

    generate_detailed_report(all_results, detailed_path)
    print(f"  Detailed report: {detailed_path}")

    generate_summary_report(all_results, summary_path)
    print(f"  Summary report: {summary_path}")

    print("\nExperiment complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
