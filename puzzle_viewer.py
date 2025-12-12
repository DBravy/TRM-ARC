import streamlit as st
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path

# ARC color palette (0-9)
ARC_COLORS = [
    "#000000",  # 0: black
    "#0074D9",  # 1: blue
    "#FF4136",  # 2: red
    "#2ECC40",  # 3: green
    "#FFDC00",  # 4: yellow
    "#AAAAAA",  # 5: gray
    "#F012BE",  # 6: magenta
    "#FF851B",  # 7: orange
    "#7FDBFF",  # 8: cyan
    "#870C25",  # 9: brown
]

@st.cache_data
def load_results(path: str) -> dict:
    """Load CNN generalization results."""
    with open(path) as f:
        return json.load(f)

@st.cache_data
def load_puzzles(data_root: str) -> dict:
    """Load all puzzle data from Kaggle files."""
    puzzles = {}
    data_path = Path(data_root)

    # Load all challenge files
    challenge_files = list(data_path.glob("*_challenges.json"))
    solution_files = list(data_path.glob("*_solutions.json"))

    for cf in challenge_files:
        with open(cf) as f:
            challenges = json.load(f)

        # Find matching solutions file
        sf = cf.with_name(cf.name.replace("_challenges", "_solutions"))
        solutions = {}
        if sf.exists():
            with open(sf) as f:
                solutions = json.load(f)

        for puzzle_id, data in challenges.items():
            puzzles[puzzle_id] = {
                "train": data.get("train", []),
                "test": data.get("test", []),
                "solutions": solutions.get(puzzle_id, [])
            }

    return puzzles

def render_grid(grid: list, title: str = "", ax=None):
    """Render an ARC grid using matplotlib."""
    grid = np.array(grid)

    # Create custom colormap
    cmap = mcolors.ListedColormap(ARC_COLORS)
    bounds = np.arange(-0.5, 10.5, 1)
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    if ax is None:
        fig, ax = plt.subplots(figsize=(3, 3))
    else:
        fig = ax.figure

    ax.imshow(grid, cmap=cmap, norm=norm)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=10)

    # Add grid lines
    for i in range(grid.shape[0] + 1):
        ax.axhline(i - 0.5, color='white', linewidth=0.5)
    for j in range(grid.shape[1] + 1):
        ax.axvline(j - 0.5, color='white', linewidth=0.5)

    return fig

def render_example(example: dict, title_prefix: str = ""):
    """Render an input-output pair."""
    has_output = "output" in example
    cols = 2 if has_output else 1

    fig, axes = plt.subplots(1, cols, figsize=(3 * cols, 3))
    if cols == 1:
        axes = [axes]

    render_grid(example["input"], f"{title_prefix}Input", axes[0])
    if has_output:
        render_grid(example["output"], f"{title_prefix}Output", axes[1])

    plt.tight_layout()
    return fig

def accuracy_color(acc: float) -> str:
    """Get color based on accuracy."""
    if acc >= 1.0:
        return "green"
    elif acc >= 0.9:
        return "lightgreen"
    elif acc >= 0.7:
        return "orange"
    else:
        return "red"

def main():
    st.set_page_config(
        page_title="ARC Puzzle Viewer",
        page_icon="🧩",
        layout="wide"
    )

    st.title("🧩 CNN Generalization Results Viewer")

    # Load data
    base_path = Path(__file__).parent
    results_path = base_path / "cnn_generalization_results.json"
    data_root = base_path / "kaggle" / "combined"

    if not results_path.exists():
        st.error(f"Results file not found: {results_path}")
        return

    results = load_results(str(results_path))
    puzzles = load_puzzles(str(data_root))

    puzzle_results = results["results"]
    summary = results["summary"]

    # Sidebar filters
    st.sidebar.header("Filters")

    # Accuracy filter
    min_acc, max_acc = st.sidebar.slider(
        "Pixel Accuracy Range",
        0.0, 1.0, (0.0, 1.0), 0.01
    )

    # Perfect filter
    perfect_filter = st.sidebar.radio(
        "Perfect Score",
        ["All", "Perfect Only", "Imperfect Only"]
    )

    # Augmentation filter
    aug_types = list(set(r.get("augmentation_used", "unknown") for r in puzzle_results))
    selected_augs = st.sidebar.multiselect(
        "Augmentation Type",
        aug_types,
        default=aug_types
    )

    # Sort options
    sort_by = st.sidebar.selectbox(
        "Sort By",
        ["puzzle_id", "pixel_accuracy", "train_time", "num_train"]
    )
    sort_order = st.sidebar.radio("Sort Order", ["Ascending", "Descending"])

    # Apply filters
    filtered = [
        r for r in puzzle_results
        if min_acc <= r.get("pixel_accuracy", 0) <= max_acc
        and (perfect_filter == "All"
             or (perfect_filter == "Perfect Only" and r.get("all_perfect", False))
             or (perfect_filter == "Imperfect Only" and not r.get("all_perfect", False)))
        and r.get("augmentation_used", "unknown") in selected_augs
    ]

    # Sort
    filtered.sort(
        key=lambda x: x.get(sort_by, 0) or 0,
        reverse=(sort_order == "Descending")
    )

    # Summary stats
    st.sidebar.markdown("---")
    st.sidebar.header("Summary")
    st.sidebar.metric("Total Puzzles", summary["total_puzzles"])
    st.sidebar.metric("Perfect Rate", f"{summary['perfect_rate']:.1%}")
    st.sidebar.metric("Avg Accuracy", f"{summary['avg_pixel_accuracy']:.1%}")
    st.sidebar.markdown("---")
    st.sidebar.metric("Filtered Puzzles", len(filtered))

    if not filtered:
        st.warning("No puzzles match the current filters.")
        return

    # Pagination
    puzzles_per_page = st.sidebar.slider("Puzzles per page", 1, 10, 5)
    total_pages = (len(filtered) + puzzles_per_page - 1) // puzzles_per_page

    # Page navigation
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        page = st.number_input(
            f"Page (1-{total_pages})",
            min_value=1,
            max_value=total_pages,
            value=1
        )

    start_idx = (page - 1) * puzzles_per_page
    end_idx = min(start_idx + puzzles_per_page, len(filtered))

    st.markdown(f"Showing puzzles **{start_idx + 1}** to **{end_idx}** of **{len(filtered)}**")

    # Display puzzles
    for i, result in enumerate(filtered[start_idx:end_idx]):
        puzzle_id = result["puzzle_id"]
        puzzle_data = puzzles.get(puzzle_id, {})

        with st.expander(f"🧩 {puzzle_id}", expanded=True):
            # Stats row
            stat_cols = st.columns(6)

            acc = result.get("pixel_accuracy", 0)
            with stat_cols[0]:
                st.metric("Pixel Accuracy", f"{acc:.1%}")
            with stat_cols[1]:
                perfect = "✅" if result.get("all_perfect") else "❌"
                st.metric("Perfect", perfect)
            with stat_cols[2]:
                st.metric("Train Examples", result.get("num_train", "?"))
            with stat_cols[3]:
                st.metric("Test Examples", result.get("num_test", "?"))
            with stat_cols[4]:
                st.metric("Train Time", f"{result.get('train_time', 0):.1f}s")
            with stat_cols[5]:
                aug = result.get("augmentation_used", "unknown")
                st.metric("Augmentation", aug)

            # Accuracy bar
            color = accuracy_color(acc)
            st.markdown(
                f"""<div style="background: linear-gradient(to right, {color} {acc*100}%, #333 {acc*100}%);
                    height: 10px; border-radius: 5px; margin-bottom: 10px;"></div>""",
                unsafe_allow_html=True
            )

            if result.get("augmentation_reason"):
                st.caption(f"Augmentation reason: {result['augmentation_reason']}")

            # Show puzzle grids
            if puzzle_data:
                # Training examples
                train_examples = puzzle_data.get("train", [])
                if train_examples:
                    st.subheader("Training Examples")
                    train_cols = st.columns(min(len(train_examples), 3))
                    for j, ex in enumerate(train_examples[:3]):
                        with train_cols[j]:
                            fig = render_example(ex, f"Train {j+1}: ")
                            st.pyplot(fig)
                            plt.close(fig)

                    if len(train_examples) > 3:
                        st.caption(f"... and {len(train_examples) - 3} more training examples")

                # Test examples
                test_examples = puzzle_data.get("test", [])
                solutions = puzzle_data.get("solutions", [])

                if test_examples:
                    st.subheader("Test Examples")
                    for j, test_ex in enumerate(test_examples):
                        test_cols = st.columns(2)
                        with test_cols[0]:
                            fig, ax = plt.subplots(figsize=(3, 3))
                            render_grid(test_ex["input"], f"Test {j+1}: Input", ax)
                            st.pyplot(fig)
                            plt.close(fig)

                        with test_cols[1]:
                            if j < len(solutions):
                                fig, ax = plt.subplots(figsize=(3, 3))
                                render_grid(solutions[j], f"Test {j+1}: Expected Output", ax)
                                st.pyplot(fig)
                                plt.close(fig)
                            else:
                                st.info("Solution not available")
            else:
                st.warning(f"Puzzle data not found for {puzzle_id}")

            st.markdown("---")

if __name__ == "__main__":
    main()
