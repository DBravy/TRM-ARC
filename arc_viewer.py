import streamlit as st
import json
import numpy as np
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

st.set_page_config(
    page_title="ARC Puzzle Viewer",
    page_icon="◼",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inject custom CSS for sleek black/white UI
st.markdown("""
<style>
    /* Main background */
    .stApp {
        background-color: #0a0a0a;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #111111;
        border-right: 1px solid #222;
    }

    section[data-testid="stSidebar"] .stMarkdown {
        color: #e0e0e0;
    }

    /* Headers */
    h1, h2, h3, h4, h5, h6 {
        color: #ffffff !important;
        font-weight: 300 !important;
        letter-spacing: 0.5px;
    }

    /* Text */
    p, span, label, .stMarkdown {
        color: #b0b0b0;
    }

    /* Input fields */
    .stTextInput input, .stSelectbox select {
        background-color: #1a1a1a !important;
        color: #ffffff !important;
        border: 1px solid #333 !important;
        border-radius: 4px;
    }

    .stTextInput input:focus {
        border-color: #555 !important;
        box-shadow: none !important;
    }

    /* Selectbox styling */
    div[data-baseweb="select"] {
        background-color: #1a1a1a;
    }

    div[data-baseweb="select"] > div {
        background-color: #1a1a1a !important;
        border-color: #333 !important;
    }

    /* Buttons */
    .stButton button {
        background-color: #1a1a1a !important;
        color: #ffffff !important;
        border: 1px solid #333 !important;
        border-radius: 4px;
        transition: all 0.2s ease;
    }

    .stButton button:hover {
        background-color: #2a2a2a !important;
        border-color: #555 !important;
    }

    /* Cards/containers */
    .puzzle-card {
        background-color: #111111;
        border: 1px solid #222;
        border-radius: 8px;
        padding: 20px;
        margin-bottom: 20px;
    }

    /* Grid container */
    .grid-container {
        display: inline-block;
        border: 2px solid #333;
        border-radius: 4px;
        overflow: hidden;
        margin: 5px;
    }

    /* Example pair container */
    .example-pair {
        display: flex;
        align-items: center;
        gap: 20px;
        margin: 15px 0;
        padding: 15px;
        background-color: #0d0d0d;
        border-radius: 6px;
    }

    /* Arrow between input/output */
    .arrow {
        color: #444;
        font-size: 24px;
        font-weight: 300;
    }

    /* Section labels */
    .section-label {
        color: #666;
        font-size: 12px;
        text-transform: uppercase;
        letter-spacing: 2px;
        margin-bottom: 10px;
    }

    /* Grid label */
    .grid-label {
        color: #555;
        font-size: 11px;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 5px;
        text-align: center;
    }

    /* Divider */
    hr {
        border: none;
        border-top: 1px solid #222;
        margin: 30px 0;
    }

    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }

    ::-webkit-scrollbar-track {
        background: #0a0a0a;
    }

    ::-webkit-scrollbar-thumb {
        background: #333;
        border-radius: 4px;
    }

    ::-webkit-scrollbar-thumb:hover {
        background: #444;
    }

    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* Expander styling */
    .streamlit-expanderHeader {
        background-color: #111 !important;
        color: #fff !important;
    }

    /* Metric styling */
    [data-testid="stMetricValue"] {
        color: #ffffff !important;
    }

    [data-testid="stMetricLabel"] {
        color: #666 !important;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: transparent;
    }

    .stTabs [data-baseweb="tab"] {
        background-color: #1a1a1a;
        border-radius: 4px;
        color: #888;
        border: 1px solid #222;
    }

    .stTabs [aria-selected="true"] {
        background-color: #2a2a2a !important;
        color: #fff !important;
    }

    /* Number input */
    .stNumberInput input {
        background-color: #1a1a1a !important;
        color: #fff !important;
        border-color: #333 !important;
    }
</style>
""", unsafe_allow_html=True)


def render_grid_html(grid: list, cell_size: int = 20) -> str:
    """Render an ARC grid as HTML table."""
    grid = np.array(grid)
    rows, cols = grid.shape

    html = f'<div class="grid-container">'
    html += '<table style="border-collapse: collapse; border-spacing: 0;">'

    for i in range(rows):
        html += '<tr>'
        for j in range(cols):
            color = ARC_COLORS[int(grid[i, j])]
            html += f'''<td style="
                width: {cell_size}px;
                height: {cell_size}px;
                background-color: {color};
                border: 1px solid #222;
                padding: 0;
            "></td>'''
        html += '</tr>'

    html += '</table></div>'
    return html


def render_example_pair(input_grid: list, output_grid: list = None, label: str = "") -> str:
    """Render an input-output pair as HTML."""
    cell_size = calculate_cell_size(input_grid, output_grid)

    html = '<div class="example-pair">'

    # Input
    html += '<div>'
    html += f'<div class="grid-label">{label} Input</div>'
    html += render_grid_html(input_grid, cell_size)
    dims = f'{len(input_grid)}×{len(input_grid[0])}'
    html += f'<div style="color: #444; font-size: 10px; text-align: center; margin-top: 4px;">{dims}</div>'
    html += '</div>'

    if output_grid is not None:
        # Arrow
        html += '<div class="arrow">→</div>'

        # Output
        html += '<div>'
        html += f'<div class="grid-label">{label} Output</div>'
        html += render_grid_html(output_grid, cell_size)
        dims = f'{len(output_grid)}×{len(output_grid[0])}'
        html += f'<div style="color: #444; font-size: 10px; text-align: center; margin-top: 4px;">{dims}</div>'
        html += '</div>'

    html += '</div>'
    return html


def calculate_cell_size(input_grid: list, output_grid: list = None) -> int:
    """Calculate appropriate cell size based on grid dimensions."""
    max_dim = max(len(input_grid), len(input_grid[0]))
    if output_grid:
        max_dim = max(max_dim, len(output_grid), len(output_grid[0]))

    if max_dim <= 5:
        return 28
    elif max_dim <= 10:
        return 22
    elif max_dim <= 15:
        return 18
    elif max_dim <= 20:
        return 14
    else:
        return 10


@st.cache_data
def load_puzzles(data_root: str) -> dict:
    """Load all puzzle data from Kaggle files."""
    puzzles = {}
    data_path = Path(data_root)

    challenge_files = list(data_path.glob("*_challenges.json"))

    for cf in challenge_files:
        # Determine dataset name
        name = cf.stem.replace("arc-agi_", "").replace("_challenges", "")

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
                "solutions": solutions.get(puzzle_id, []),
                "dataset": name
            }

    return puzzles


def main():
    # Header
    st.markdown("""
        <div style="margin-bottom: 30px;">
            <h1 style="margin-bottom: 5px; font-size: 28px;">ARC Puzzle Viewer</h1>
            <p style="color: #555; font-size: 14px;">Abstraction and Reasoning Corpus</p>
        </div>
    """, unsafe_allow_html=True)

    # Load puzzles
    base_path = Path(__file__).parent
    data_root = base_path / "kaggle" / "combined"

    if not data_root.exists():
        st.error(f"Data directory not found: {data_root}")
        return

    puzzles = load_puzzles(str(data_root))
    puzzle_ids = sorted(puzzles.keys())

    # Sidebar
    with st.sidebar:
        st.markdown('<p class="section-label">Navigation</p>', unsafe_allow_html=True)

        # Search
        search = st.text_input("Search puzzle ID", placeholder="e.g., 007bbfb7")

        # Dataset filter
        datasets = sorted(set(p["dataset"] for p in puzzles.values()))
        selected_datasets = st.multiselect(
            "Filter by dataset",
            datasets,
            default=datasets,
            help="Select which datasets to include"
        )

        # Filter puzzles
        filtered_ids = [
            pid for pid in puzzle_ids
            if puzzles[pid]["dataset"] in selected_datasets
            and (not search or search.lower() in pid.lower())
        ]

        st.markdown("---")
        st.markdown(f'<p style="color: #555;">{len(filtered_ids)} puzzles</p>', unsafe_allow_html=True)

        # Puzzle selector
        if filtered_ids:
            selected_idx = st.selectbox(
                "Select puzzle",
                range(len(filtered_ids)),
                format_func=lambda i: filtered_ids[i]
            )
            selected_id = filtered_ids[selected_idx]

            # Quick navigation
            st.markdown("---")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("← Prev", use_container_width=True) and selected_idx > 0:
                    st.session_state.nav_to = selected_idx - 1
                    st.rerun()
            with col2:
                if st.button("Next →", use_container_width=True) and selected_idx < len(filtered_ids) - 1:
                    st.session_state.nav_to = selected_idx + 1
                    st.rerun()

            # Jump to index
            st.markdown("---")
            jump_idx = st.number_input(
                "Jump to index",
                min_value=0,
                max_value=len(filtered_ids) - 1,
                value=selected_idx
            )
            if jump_idx != selected_idx:
                st.session_state.nav_to = jump_idx
                st.rerun()
        else:
            st.warning("No puzzles match your filters")
            return

    # Main content
    puzzle = puzzles[selected_id]

    # Puzzle header
    col1, col2, col3 = st.columns([3, 1, 1])
    with col1:
        st.markdown(f"""
            <div style="margin-bottom: 20px;">
                <h2 style="margin: 0; font-family: monospace; letter-spacing: 1px;">{selected_id}</h2>
                <span style="color: #555; font-size: 12px; text-transform: uppercase;">{puzzle['dataset']}</span>
            </div>
        """, unsafe_allow_html=True)
    with col2:
        st.metric("Train", len(puzzle["train"]))
    with col3:
        st.metric("Test", len(puzzle["test"]))

    st.markdown("---")

    # Training examples
    st.markdown('<p class="section-label">Training Examples</p>', unsafe_allow_html=True)

    for i, example in enumerate(puzzle["train"]):
        html = render_example_pair(
            example["input"],
            example.get("output"),
            f"#{i+1}"
        )
        st.markdown(html, unsafe_allow_html=True)

    st.markdown("---")

    # Test examples
    st.markdown('<p class="section-label">Test Examples</p>', unsafe_allow_html=True)

    solutions = puzzle.get("solutions", [])

    for i, test in enumerate(puzzle["test"]):
        output = solutions[i] if i < len(solutions) else None
        label = f"Test #{i+1}"

        if output:
            html = render_example_pair(test["input"], output, label)
        else:
            # Show input only with placeholder for output
            cell_size = calculate_cell_size(test["input"])
            html = '<div class="example-pair">'
            html += '<div>'
            html += f'<div class="grid-label">{label} Input</div>'
            html += render_grid_html(test["input"], cell_size)
            dims = f'{len(test["input"])}×{len(test["input"][0])}'
            html += f'<div style="color: #444; font-size: 10px; text-align: center; margin-top: 4px;">{dims}</div>'
            html += '</div>'
            html += '<div class="arrow">→</div>'
            html += '<div style="color: #444; font-style: italic; padding: 20px;">Solution hidden</div>'
            html += '</div>'

        st.markdown(html, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
