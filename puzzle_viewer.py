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

def normalize_results(results: dict) -> tuple[list, dict, str]:
    """
    Normalize results from either per-puzzle or multi-puzzle format.
    Returns (puzzle_results_list, summary_dict, mode).
    """
    mode = results.get("mode", "per-puzzle")

    if mode == "multi-puzzle":
        # Convert per_puzzle dict to list format
        per_puzzle = results.get("per_puzzle", {})
        puzzle_results = []
        for puzzle_id, data in per_puzzle.items():
            if "error" in data:
                continue
            puzzle_results.append({
                "puzzle_id": puzzle_id,
                "pixel_accuracy": data.get("pixel_accuracy", 0),
                "perfect_rate": data.get("perfect_rate", 0),
                "all_perfect": data.get("all_perfect", False),
                "num_test": data.get("num_test_examples", 1),
                # These fields aren't available in multi-puzzle mode
                "num_train": None,
                "train_time": None,
                "augmentation_used": None,
                "augmentation_reason": None,
            })

        # Normalize summary
        raw_summary = results.get("summary", {})
        summary = {
            "total_puzzles": raw_summary.get("num_puzzles", len(puzzle_results)),
            "perfect_count": raw_summary.get("puzzles_all_perfect", 0),
            "perfect_rate": raw_summary.get("puzzles_all_perfect", 0) / max(raw_summary.get("num_puzzles", 1), 1),
            "avg_pixel_accuracy": raw_summary.get("pixel_accuracy", 0),
            "train_time": raw_summary.get("train_time", 0),
            "total_test_examples": raw_summary.get("total_test_examples", 0),
            "perfect_examples": raw_summary.get("perfect_examples", 0),
            "perfect_example_rate": raw_summary.get("perfect_example_rate", 0),
        }
    else:
        # Per-puzzle format - already a list
        puzzle_results = results.get("results", [])
        summary = results.get("summary", {})

    return puzzle_results, summary, mode


def main():
    st.set_page_config(
        page_title="ARC Puzzle Viewer",
        page_icon="🧩",
        layout="wide"
    )

    st.title("🧩 CNN Generalization Results Viewer")

    # Load data
    base_path = Path(__file__).parent
    data_root = base_path / "kaggle" / "combined"

    # Find available results files
    results_files = list(base_path.glob("cnn_generalization_results*.json"))
    if not results_files:
        st.error("No results files found matching 'cnn_generalization_results*.json'")
        return

    # Let user select which results file to view
    results_file_names = [f.name for f in results_files]
    selected_file = st.sidebar.selectbox("Results File", results_file_names)
    results_path = base_path / selected_file

    results = load_results(str(results_path))
    puzzles = load_puzzles(str(data_root))

    puzzle_results, summary, mode = normalize_results(results)

    # Show mode indicator
    mode_label = "🔀 Multi-Puzzle" if mode == "multi-puzzle" else "🎯 Per-Puzzle"
    st.sidebar.markdown(f"**Mode:** {mode_label}")

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

    # Augmentation filter (only for per-puzzle mode)
    selected_augs = None
    if mode == "per-puzzle":
        aug_types = list(set(r.get("augmentation_used", "unknown") for r in puzzle_results))
        if len(aug_types) > 1 or (len(aug_types) == 1 and aug_types[0] is not None):
            selected_augs = st.sidebar.multiselect(
                "Augmentation Type",
                aug_types,
                default=aug_types
            )

    # Sort options - adjust based on mode
    if mode == "multi-puzzle":
        sort_options = ["puzzle_id", "pixel_accuracy"]
    else:
        sort_options = ["puzzle_id", "pixel_accuracy", "train_time", "num_train"]

    sort_by = st.sidebar.selectbox("Sort By", sort_options)
    sort_order = st.sidebar.radio("Sort Order", ["Ascending", "Descending"])

    # Apply filters
    filtered = []
    for r in puzzle_results:
        acc = r.get("pixel_accuracy", 0)
        if not (min_acc <= acc <= max_acc):
            continue
        if perfect_filter == "Perfect Only" and not r.get("all_perfect", False):
            continue
        if perfect_filter == "Imperfect Only" and r.get("all_perfect", False):
            continue
        if selected_augs is not None and r.get("augmentation_used", "unknown") not in selected_augs:
            continue
        filtered.append(r)

    # Sort
    filtered.sort(
        key=lambda x: (x.get(sort_by) is None, x.get(sort_by, 0) or 0),
        reverse=(sort_order == "Descending")
    )

    # Summary stats
    st.sidebar.markdown("---")
    st.sidebar.header("Summary")
    st.sidebar.metric("Total Puzzles", summary.get("total_puzzles", len(puzzle_results)))
    st.sidebar.metric("Perfect Rate", f"{summary.get('perfect_rate', 0):.1%}")
    st.sidebar.metric("Avg Accuracy", f"{summary.get('avg_pixel_accuracy', 0):.1%}")

    if mode == "multi-puzzle":
        train_time = summary.get("train_time", 0)
        st.sidebar.metric("Total Train Time", f"{train_time:.1f}s")

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
            # Stats row - adjust columns based on available data
            if mode == "multi-puzzle":
                stat_cols = st.columns(3)
            else:
                stat_cols = st.columns(6)

            acc = result.get("pixel_accuracy", 0)
            col_idx = 0

            with stat_cols[col_idx]:
                st.metric("Pixel Accuracy", f"{acc:.1%}")
            col_idx += 1

            with stat_cols[col_idx]:
                perfect = "✅" if result.get("all_perfect") else "❌"
                st.metric("Perfect", perfect)
            col_idx += 1

            if mode == "per-puzzle":
                with stat_cols[col_idx]:
                    num_train = result.get("num_train")
                    st.metric("Train Examples", num_train if num_train is not None else "?")
                col_idx += 1

            with stat_cols[col_idx % len(stat_cols)]:
                num_test = result.get("num_test")
                st.metric("Test Examples", num_test if num_test is not None else "?")

            if mode == "per-puzzle":
                col_idx += 1
                with stat_cols[col_idx]:
                    train_time = result.get("train_time")
                    if train_time is not None:
                        st.metric("Train Time", f"{train_time:.1f}s")
                    else:
                        st.metric("Train Time", "N/A")
                col_idx += 1

                with stat_cols[col_idx]:
                    aug = result.get("augmentation_used")
                    st.metric("Augmentation", aug if aug else "N/A")

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
