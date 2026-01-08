#!/bin/bash
# Quick regression test for core puzzles
# All 6 puzzles should achieve 100% grid pixel accuracy

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
    # Note: ordering and selection screening are now enabled by default
    output=$(MPLBACKEND=Agg /opt/anaconda3/envs/TRM/bin/python relational_position.py \
        --puzzle-id "$puzzle" 2>&1)

    # Print output
    echo "$output"

    # Check for 100% grid accuracy in BOTH Training Set AND Test Set sections
    # Extract the Grid Pixel Accuracy section
    accuracy_section=$(echo "$output" | sed -n '/Grid Pixel Accuracy/,/Grid Visualization/p')

    # Check Training Set: find "--- Training Set ---" and the next "Grid Accuracy:" line
    train_accuracy=$(echo "$accuracy_section" | sed -n '/--- Training Set ---/,/--- Test Set ---/p' | grep "Grid Accuracy:" | head -1)

    # Check Test Set: find "--- Test Set ---" and the next "Grid Accuracy:" line
    test_accuracy=$(echo "$accuracy_section" | sed -n '/--- Test Set ---/,/$/p' | grep "Grid Accuracy:" | head -1)

    # Both must show 100.0%
    train_pass=false
    test_pass=false

    if echo "$train_accuracy" | grep -q "100.0%"; then
        train_pass=true
    fi

    if echo "$test_accuracy" | grep -q "100.0%"; then
        test_pass=true
    fi

    if $train_pass && $test_pass; then
        echo "✓ PASSED: $puzzle (100% grid accuracy on both train and test)"
    else
        if ! $train_pass && ! $test_pass; then
            echo "✗ FAILED: $puzzle (train: $train_accuracy, test: $test_accuracy)"
        elif ! $train_pass; then
            echo "✗ FAILED: $puzzle (training set not 100%: $train_accuracy)"
        else
            echo "✗ FAILED: $puzzle (test set not 100%: $test_accuracy)"
        fi
        FAILED+=("$puzzle")
    fi
done

# Special test: 5582e5ca with transformation mode
echo ""
echo "Testing puzzle: 5582e5ca (transformation mode)"
echo "----------------------------------------"

output=$(MPLBACKEND=Agg /opt/anaconda3/envs/TRM/bin/python relational_position.py \
    --puzzle-id 5582e5ca \
    --verbose \
    --input-segmentation-mode pixel \
    --output-segmentation-mode connectivity \
    --correspondence-mode many_to_one \
    --correspondence-margin 0.0 \
    --transformation 2>&1)

echo "$output"

accuracy_section=$(echo "$output" | sed -n '/Grid Pixel Accuracy/,/Grid Visualization/p')
train_accuracy=$(echo "$accuracy_section" | sed -n '/--- Training Set ---/,/--- Test Set ---/p' | grep "Grid Accuracy:" | head -1)
test_accuracy=$(echo "$accuracy_section" | sed -n '/--- Test Set ---/,/$/p' | grep "Grid Accuracy:" | head -1)

train_pass=false
test_pass=false

if echo "$train_accuracy" | grep -q "100.0%"; then
    train_pass=true
fi

if echo "$test_accuracy" | grep -q "100.0%"; then
    test_pass=true
fi

if $train_pass && $test_pass; then
    echo "✓ PASSED: 5582e5ca (100% grid accuracy on both train and test)"
else
    if ! $train_pass && ! $test_pass; then
        echo "✗ FAILED: 5582e5ca (train: $train_accuracy, test: $test_accuracy)"
    elif ! $train_pass; then
        echo "✗ FAILED: 5582e5ca (training set not 100%: $train_accuracy)"
    else
        echo "✗ FAILED: 5582e5ca (test set not 100%: $test_accuracy)"
    fi
    FAILED+=("5582e5ca")
fi

# Special test: syn_dual_fill with transformation mode (divider preservation)
echo ""
echo "Testing puzzle: syn_dual_fill (transformation mode)"
echo "----------------------------------------"

output=$(MPLBACKEND=Agg /opt/anaconda3/envs/TRM/bin/python relational_position.py \
    --puzzle-id syn_dual_fill \
    --verbose \
    --transformation 2>&1)

echo "$output"

accuracy_section=$(echo "$output" | sed -n '/Grid Pixel Accuracy/,/Grid Visualization/p')
train_accuracy=$(echo "$accuracy_section" | sed -n '/--- Training Set ---/,/--- Test Set ---/p' | grep "Grid Accuracy:" | head -1)
test_accuracy=$(echo "$accuracy_section" | sed -n '/--- Test Set ---/,/$/p' | grep "Grid Accuracy:" | head -1)

train_pass=false
test_pass=false

if echo "$train_accuracy" | grep -q "100.0%"; then
    train_pass=true
fi

if echo "$test_accuracy" | grep -q "100.0%"; then
    test_pass=true
fi

if $train_pass && $test_pass; then
    echo "✓ PASSED: syn_dual_fill (100% grid accuracy on both train and test)"
else
    if ! $train_pass && ! $test_pass; then
        echo "✗ FAILED: syn_dual_fill (train: $train_accuracy, test: $test_accuracy)"
    elif ! $train_pass; then
        echo "✗ FAILED: syn_dual_fill (training set not 100%: $train_accuracy)"
    else
        echo "✗ FAILED: syn_dual_fill (test set not 100%: $test_accuracy)"
    fi
    FAILED+=("syn_dual_fill")
fi

echo ""
echo "========================================"
if [ ${#FAILED[@]} -eq 0 ]; then
    echo "All tests passed!"
    exit 0
else
    echo "FAILED puzzles: ${FAILED[*]}"
    exit 1
fi
