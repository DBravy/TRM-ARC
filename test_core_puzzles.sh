#!/bin/bash
# Quick regression test for core puzzles
# All 4 puzzles should achieve 100% grid pixel accuracy

set -e

PUZZLES=("1990f7a8" "03560426" "23b5c85d" "27a77e38")
FAILED=()

echo "Running core puzzle regression tests..."
echo "========================================"

for puzzle in "${PUZZLES[@]}"; do
    echo ""
    echo "Testing puzzle: $puzzle"
    echo "----------------------------------------"

    # Run the test and capture output (MPLBACKEND=Agg to avoid GUI blocking)
    output=$(MPLBACKEND=Agg /opt/anaconda3/envs/TRM/bin/python relational_position.py \
        --puzzle-id "$puzzle" \
        --per-object-anchor \
        --screen-ordering \
        --no-train \
        --screen-selection \
        --screen-hierarchy 2>&1)

    # Print output
    echo "$output"

    # Check for 100% grid accuracy in the "Grid Pixel Accuracy" section
    # Look for "Grid Accuracy: 100.0%" pattern in the training set output
    if echo "$output" | grep -A6 "Grid Pixel Accuracy" | grep -q "Grid Accuracy: 100.0%"; then
        echo "✓ PASSED: $puzzle (100% grid accuracy)"
    else
        # Check if all examples show PASS
        if echo "$output" | grep -A20 "Grid Pixel Accuracy" | grep -E "Example [0-9]+:" | grep -qv "PASS"; then
            echo "✗ FAILED: $puzzle (not all grids match)"
            FAILED+=("$puzzle")
        elif echo "$output" | grep -A6 "Grid Pixel Accuracy" | grep -qE "Grid Accuracy:"; then
            # Grid Accuracy line exists but not 100%
            echo "✗ FAILED: $puzzle"
            FAILED+=("$puzzle")
        else
            echo "? UNKNOWN (Grid Pixel Accuracy section not found): $puzzle"
            FAILED+=("$puzzle")
        fi
    fi
done

echo ""
echo "========================================"
if [ ${#FAILED[@]} -eq 0 ]; then
    echo "All tests passed!"
    exit 0
else
    echo "FAILED puzzles: ${FAILED[*]}"
    exit 1
fi
