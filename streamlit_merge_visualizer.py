#!/usr/bin/env python3
"""
Streamlit app to visualize merge thresholds for all ARC AGI 1 test inputs.

Run with:
    streamlit run streamlit_merge_visualizer.py
"""

import json
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import streamlit as st
from scipy import ndimage

# Import the core functions from the original visualization script
from visualize_merge_thresholds import (
    ARC_COLORS,
    NUM_COLORS,
    detect_background_color,
    compute_connected_components,
    compute_pairwise_affinity,
    merge_components,
    create_group_visualization,
)


def load_puzzles(dataset_name: str, data_root: str = "kaggle/combined") -> dict:
    """Load puzzles from the dataset."""
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


def create_visualization_figure(input_grid, output_grid, thresholds, use_background_detection):
    """Create matplotlib figure for a puzzle's test input/output."""
    # Detect background color
    background_color = None
    if use_background_detection:
        background_color = detect_background_color(input_grid)

    # Compute components and affinity for input grid
    component_labels, component_colors, component_bboxes, component_is_background = \
        compute_connected_components(input_grid, background_color)
    affinity = compute_pairwise_affinity(
        component_labels, component_colors, component_bboxes, component_is_background
    )
    num_components = len(component_colors)
    num_background_components = sum(component_is_background)

    # Create figure
    n_thresholds = len(thresholds)
    n_cols = n_thresholds + 2  # Original + Components + thresholds
    n_rows = 2 if output_grid is not None else 1

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.5 * n_cols, 2.5 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)

    # Colormaps
    arc_cmap = mcolors.ListedColormap(ARC_COLORS)
    group_cmap = plt.cm.get_cmap('tab20', 20)

    def plot_arc_grid(ax, grid, title):
        H, W = grid.shape
        ax.imshow(grid, cmap=arc_cmap, vmin=0, vmax=9, interpolation='nearest')
        ax.set_title(title, fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xticks(np.arange(-0.5, W, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, H, 1), minor=True)
        ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.5)

    def plot_group_grid(ax, grid, title):
        H, W = grid.shape
        masked_grid = np.ma.masked_where(grid < 0, grid)
        ax.imshow(masked_grid, cmap=group_cmap, vmin=0, vmax=19, interpolation='nearest')
        ax.set_title(title, fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xticks(np.arange(-0.5, W, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, H, 1), minor=True)
        ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.5)

    # Row 0: Test input grid analysis
    plot_arc_grid(axes[0, 0], input_grid, "Test Input")
    plot_group_grid(axes[0, 1], component_labels, f"Components ({num_components})")

    for i, threshold in enumerate(thresholds):
        group_ids = merge_components(affinity, num_components, threshold)
        group_grid = create_group_visualization(component_labels, group_ids)
        num_groups = len(np.unique(group_ids)) if len(group_ids) > 0 else 0
        plot_group_grid(axes[0, i + 2], group_grid, f"t={threshold} ({num_groups} grps)")

    # Row 1: Test output grid analysis (if available)
    if output_grid is not None:
        out_background_color = None
        if use_background_detection:
            out_background_color = detect_background_color(output_grid)

        out_component_labels, out_component_colors, out_component_bboxes, out_component_is_background = \
            compute_connected_components(output_grid, out_background_color)
        out_affinity = compute_pairwise_affinity(
            out_component_labels, out_component_colors, out_component_bboxes, out_component_is_background
        )
        out_num_components = len(out_component_colors)

        plot_arc_grid(axes[1, 0], output_grid, "Test Output")
        plot_group_grid(axes[1, 1], out_component_labels, f"Components ({out_num_components})")

        for i, threshold in enumerate(thresholds):
            group_ids = merge_components(out_affinity, out_num_components, threshold)
            group_grid = create_group_visualization(out_component_labels, group_ids)
            num_groups = len(np.unique(group_ids)) if len(group_ids) > 0 else 0
            plot_group_grid(axes[1, i + 2], group_grid, f"t={threshold} ({num_groups} grps)")

    plt.tight_layout()
    return fig, background_color, num_components, num_background_components


def main():
    st.set_page_config(
        page_title="ARC Merge Threshold Visualizer",
        layout="wide",
    )

    st.title("ARC AGI 1 - Merge Threshold Visualizer")

    # Sidebar controls
    st.sidebar.header("Settings")

    dataset = st.sidebar.selectbox(
        "Dataset",
        ["arc-agi-1", "arc-agi-2"],
        index=0
    )

    use_background_detection = st.sidebar.checkbox(
        "Enable Background Detection",
        value=True
    )

    # Threshold selection
    st.sidebar.subheader("Merge Thresholds")
    default_thresholds = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    threshold_str = st.sidebar.text_input(
        "Thresholds (comma-separated)",
        value=", ".join(map(str, default_thresholds))
    )
    try:
        thresholds = [float(t.strip()) for t in threshold_str.split(",")]
    except ValueError:
        thresholds = default_thresholds
        st.sidebar.error("Invalid thresholds, using defaults")

    # Load puzzles
    @st.cache_data
    def cached_load_puzzles(dataset_name):
        return load_puzzles(dataset_name)

    with st.spinner(f"Loading {dataset} puzzles..."):
        puzzles = cached_load_puzzles(dataset)

    puzzle_ids = sorted(puzzles.keys())
    st.sidebar.info(f"Loaded {len(puzzle_ids)} puzzles")

    # Puzzle selection
    st.sidebar.subheader("Navigation")

    # Initialize session state for puzzle index
    if "puzzle_idx" not in st.session_state:
        st.session_state.puzzle_idx = 0

    # Search by ID
    search_id = st.sidebar.text_input("Search puzzle ID")
    if search_id:
        matching = [pid for pid in puzzle_ids if search_id.lower() in pid.lower()]
        if matching:
            st.sidebar.write(f"Found {len(matching)} matches")
            selected_match = st.sidebar.selectbox("Select match", matching)
            if selected_match:
                st.session_state.puzzle_idx = puzzle_ids.index(selected_match)

    # Navigation buttons
    col1, col2, col3 = st.sidebar.columns(3)
    with col1:
        if st.button("⏮ First"):
            st.session_state.puzzle_idx = 0
    with col2:
        if st.button("◀ Prev"):
            st.session_state.puzzle_idx = max(0, st.session_state.puzzle_idx - 1)
    with col3:
        if st.button("Next ▶"):
            st.session_state.puzzle_idx = min(len(puzzle_ids) - 1, st.session_state.puzzle_idx + 1)

    # Slider for puzzle selection
    st.session_state.puzzle_idx = st.sidebar.slider(
        "Puzzle index",
        0, len(puzzle_ids) - 1,
        st.session_state.puzzle_idx
    )

    # Get current puzzle
    current_puzzle_id = puzzle_ids[st.session_state.puzzle_idx]
    puzzle_data = puzzles[current_puzzle_id]

    st.sidebar.markdown(f"**Current:** `{current_puzzle_id}`")

    # Main content
    st.header(f"Puzzle: {current_puzzle_id}")

    # Get test examples
    test_examples = puzzle_data.get("test", [])
    train_examples = puzzle_data.get("train", [])

    st.write(f"Training examples: {len(train_examples)} | Test examples: {len(test_examples)}")

    if not test_examples:
        st.warning("No test examples found for this puzzle")
        return

    # Test example selector (if multiple)
    test_idx = 0
    if len(test_examples) > 1:
        test_idx = st.selectbox(
            "Test example",
            range(len(test_examples)),
            format_func=lambda x: f"Test {x + 1}"
        )

    example = test_examples[test_idx]
    input_grid = np.array(example["input"], dtype=np.int32)
    output_grid = np.array(example["output"], dtype=np.int32) if "output" in example else None

    # Create and display visualization
    fig, background_color, num_components, num_bg_components = create_visualization_figure(
        input_grid, output_grid, thresholds, use_background_detection
    )

    # Display info
    info_cols = st.columns(4)
    with info_cols[0]:
        st.metric("Input Size", f"{input_grid.shape[0]}×{input_grid.shape[1]}")
    with info_cols[1]:
        st.metric("Components", num_components)
    with info_cols[2]:
        st.metric("Background Components", num_bg_components)
    with info_cols[3]:
        bg_display = str(background_color) if background_color is not None else "None"
        st.metric("Background Color", bg_display)

    # Display the figure
    st.pyplot(fig)
    plt.close(fig)

    # Show training examples
    with st.expander("View Training Examples"):
        for i, train_ex in enumerate(train_examples):
            st.subheader(f"Training Example {i + 1}")
            train_input = np.array(train_ex["input"], dtype=np.int32)
            train_output = np.array(train_ex["output"], dtype=np.int32)

            train_fig, _, _, _ = create_visualization_figure(
                train_input, train_output, thresholds, use_background_detection
            )
            st.pyplot(train_fig)
            plt.close(train_fig)


if __name__ == "__main__":
    main()
