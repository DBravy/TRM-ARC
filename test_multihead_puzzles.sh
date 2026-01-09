#!/bin/bash
# Regression test for multihead correspondence module puzzles
# All puzzles should achieve 100% accuracy on training examples

set -e

cd "$(dirname "$0")"

PUZZLES=(
    "136b0064"    # Procedural drawing
    "17cae0c1"    # Pattern fill
    "12997ef3"    # Template replication
    "103eff5b"    # Regional fill
    "0becf7df"    # Color remapping
)

FAILED=()

echo "Running multihead correspondence module tests..."
echo "================================================="

for puzzle in "${PUZZLES[@]}"; do
    echo ""
    echo "Testing puzzle: $puzzle"
    echo "----------------------------------------"

    # Run the test and capture output
    output=$(/opt/anaconda3/envs/TRM/bin/python multihead_correspondence_module.py \
        --puzzle-id "$puzzle" 2>&1)

    # Print output
    echo "$output"

    # Check for perfect accuracy on training examples
    # Look for "Total: N/N examples correct" where both numbers are the same
    total_line=$(echo "$output" | grep "Total:" | head -1)

    if [ -z "$total_line" ]; then
        echo "✗ FAILED: $puzzle (no Total line found)"
        FAILED+=("$puzzle")
        continue
    fi

    # Extract the numbers from "Total: X/Y examples correct"
    correct=$(echo "$total_line" | sed -n 's/.*Total: \([0-9]*\)\/\([0-9]*\).*/\1/p')
    total=$(echo "$total_line" | sed -n 's/.*Total: \([0-9]*\)\/\([0-9]*\).*/\2/p')

    if [ "$correct" = "$total" ] && [ "$total" != "0" ]; then
        echo "✓ PASSED: $puzzle ($correct/$total training examples correct)"
    else
        echo "✗ FAILED: $puzzle ($correct/$total training examples correct)"
        FAILED+=("$puzzle")
    fi
done

echo ""
echo "================================================="
if [ ${#FAILED[@]} -eq 0 ]; then
    echo "All multihead tests passed!"
    exit 0
else
    echo "FAILED puzzles: ${FAILED[*]}"
    exit 1
fi
