/**
 * CNN Learning Visualizer - Frontend JavaScript
 *
 * Handles rendering of ARC grids, kernel heatmaps, and layer activations.
 */

// ARC color palette (0-9)
const ARC_COLORS = [
    '#000000',  // 0: black
    '#0074D9',  // 1: blue
    '#FF4136',  // 2: red
    '#2ECC40',  // 3: green
    '#FFDC00',  // 4: yellow
    '#AAAAAA',  // 5: gray
    '#F012BE',  // 6: magenta
    '#FF851B',  // 7: orange
    '#7FDBFF',  // 8: cyan
    '#870C25'   // 9: brown
];

// Color names for display
const COLOR_NAMES = [
    'Black', 'Blue', 'Red', 'Green', 'Yellow',
    'Gray', 'Magenta', 'Orange', 'Cyan', 'Brown'
];

// RdBu diverging colormap for kernel visualization
function rdBuColor(val) {
    // val should be in [-1, 1], map to RGB
    // -1 = blue (#2166ac), 0 = white (#f7f7f7), 1 = red (#b2182b)
    val = Math.max(-1, Math.min(1, val));

    if (val < 0) {
        // Blue to white
        const t = val + 1; // 0 to 1
        const r = Math.round(33 + t * (247 - 33));
        const g = Math.round(102 + t * (247 - 102));
        const b = Math.round(172 + t * (247 - 172));
        return `rgb(${r}, ${g}, ${b})`;
    } else {
        // White to red
        const t = val; // 0 to 1
        const r = Math.round(247 + t * (178 - 247));
        const g = Math.round(247 + t * (24 - 247));
        const b = Math.round(247 + t * (43 - 247));
        return `rgb(${r}, ${g}, ${b})`;
    }
}

// Viridis-like colormap for activations
function viridisColor(val) {
    // val should be in [0, 1]
    val = Math.max(0, Math.min(1, val));

    // Simplified viridis approximation
    const r = Math.round(68 + val * (253 - 68));
    const g = Math.round(1 + val * (231 - 1));
    const b = Math.round(84 + val * (37 - 84));
    return `rgb(${r}, ${g}, ${b})`;
}

// State
let modelInfo = null;
let kernelData = null;
let puzzleList = [];
let filteredPuzzles = [];
let currentPuzzle = null;
let currentExampleIdx = 0;
let flowData = null;
let currentLayerIdx = 0;
let isPlaying = false;
let playInterval = null;
let selectedPixel = null;  // {row, col} of currently selected pixel for trace

// =============================================================================
// Tab Navigation
// =============================================================================

function showTab(tabName) {
    // Hide all tabs
    document.querySelectorAll('.tab-content').forEach(el => {
        el.classList.remove('active');
    });
    document.querySelectorAll('.tab-btn').forEach(el => {
        el.classList.remove('active');
    });

    // Show selected tab
    document.getElementById(`${tabName}-tab`).classList.add('active');
    event.target.classList.add('active');

    // Load data for tab if needed
    if (tabName === 'kernels' && !kernelData) {
        loadKernels();
    }
    if (tabName === 'puzzles' && puzzleList.length === 0) {
        loadPuzzles();
    }
}

// =============================================================================
// Rendering Functions
// =============================================================================

function renderArcGrid(grid, container, cellSize = 15, options = {}) {
    container.innerHTML = '';

    const fullRows = grid.length;
    const fullCols = grid[0] ? grid[0].length : 0;
    const {
        clickable = false,
        onCellClick = null,
        highlightBounds = null,
        cropHeight = null,  // Crop to this height (null = show full grid)
        cropWidth = null    // Crop to this width (null = show full grid)
    } = options;

    // Use cropped dimensions if specified
    const rows = cropHeight !== null ? Math.min(cropHeight, fullRows) : fullRows;
    const cols = cropWidth !== null ? Math.min(cropWidth, fullCols) : fullCols;

    const gridEl = document.createElement('div');
    gridEl.className = 'arc-grid';
    gridEl.style.gridTemplateColumns = `repeat(${cols}, ${cellSize}px)`;

    for (let r = 0; r < rows; r++) {
        for (let c = 0; c < cols; c++) {
            const cell = document.createElement('div');
            cell.className = 'arc-cell';
            cell.style.backgroundColor = ARC_COLORS[grid[r][c] || 0];
            cell.style.width = `${cellSize}px`;
            cell.style.height = `${cellSize}px`;
            cell.dataset.row = r;
            cell.dataset.col = c;

            // Add clickable functionality
            if (clickable && onCellClick) {
                cell.classList.add('clickable');
                cell.onclick = () => onCellClick(r, c);
            }

            // Highlight cells within receptive field bounds
            if (highlightBounds) {
                const { row_start, row_end, col_start, col_end, center_row, center_col } = highlightBounds;
                if (r >= row_start && r < row_end && c >= col_start && c < col_end) {
                    cell.style.boxShadow = '0 0 0 1px rgba(0, 212, 255, 0.5)';
                    if (r === center_row && c === center_col) {
                        cell.classList.add('selected');
                    }
                }
            }

            gridEl.appendChild(cell);
        }
    }

    container.appendChild(gridEl);
}

function renderKernelCanvas(data, canvasSize = 60) {
    // data is a kH x kW array (any kernel size)
    const canvas = document.createElement('canvas');
    canvas.width = canvasSize;
    canvas.height = canvasSize;
    const ctx = canvas.getContext('2d');

    const kH = data.length;
    const kW = data[0] ? data[0].length : 0;
    if (kH === 0 || kW === 0) return canvas;

    const cellH = canvasSize / kH;
    const cellW = canvasSize / kW;

    // Find max absolute value for normalization
    let maxAbs = 0;
    for (let r = 0; r < kH; r++) {
        for (let c = 0; c < kW; c++) {
            maxAbs = Math.max(maxAbs, Math.abs(data[r][c]));
        }
    }
    if (maxAbs === 0) maxAbs = 1;

    for (let r = 0; r < kH; r++) {
        for (let c = 0; c < kW; c++) {
            const val = data[r][c] / maxAbs;
            ctx.fillStyle = rdBuColor(val);
            ctx.fillRect(c * cellW, r * cellH, cellW, cellH);
        }
    }

    return canvas;
}

function renderActivationCanvas(data, canvasSize = 100) {
    // data is a 2D array (H x W)
    const canvas = document.createElement('canvas');
    const rows = data.length;
    const cols = data[0] ? data[0].length : 0;

    // Keep aspect ratio but fit within canvasSize
    const scale = Math.min(canvasSize / rows, canvasSize / cols);
    canvas.width = Math.round(cols * scale);
    canvas.height = Math.round(rows * scale);

    const ctx = canvas.getContext('2d');

    for (let r = 0; r < rows; r++) {
        for (let c = 0; c < cols; c++) {
            const val = data[r][c];  // Assume already normalized 0-1
            ctx.fillStyle = viridisColor(val);
            ctx.fillRect(
                Math.round(c * scale),
                Math.round(r * scale),
                Math.ceil(scale),
                Math.ceil(scale)
            );
        }
    }

    return canvas;
}

function renderOverlayCanvas(activation, grid, canvasSize = 150) {
    // Overlay mean activation on ARC grid
    const canvas = document.createElement('canvas');
    const rows = grid.length;
    const cols = grid[0] ? grid[0].length : 0;

    const scale = Math.min(canvasSize / rows, canvasSize / cols);
    canvas.width = Math.round(cols * scale);
    canvas.height = Math.round(rows * scale);

    const ctx = canvas.getContext('2d');

    // Resize activation to match grid if needed
    const actRows = activation.length;
    const actCols = activation[0] ? activation[0].length : 0;

    for (let r = 0; r < rows; r++) {
        for (let c = 0; c < cols; c++) {
            // Draw base color
            ctx.fillStyle = ARC_COLORS[grid[r][c] || 0];
            ctx.fillRect(
                Math.round(c * scale),
                Math.round(r * scale),
                Math.ceil(scale),
                Math.ceil(scale)
            );

            // Get activation value (interpolate if sizes differ)
            const ar = Math.min(Math.floor(r * actRows / rows), actRows - 1);
            const ac = Math.min(Math.floor(c * actCols / cols), actCols - 1);
            const actVal = activation[ar] ? (activation[ar][ac] || 0) : 0;

            // Overlay with transparency
            ctx.fillStyle = `rgba(255, 255, 0, ${actVal * 0.6})`;
            ctx.fillRect(
                Math.round(c * scale),
                Math.round(r * scale),
                Math.ceil(scale),
                Math.ceil(scale)
            );
        }
    }

    return canvas;
}

// =============================================================================
// Model Info & Kernels
// =============================================================================

async function loadModelInfo() {
    try {
        const response = await fetch('/api/model-info');
        modelInfo = await response.json();

        const container = document.getElementById('model-info');
        // Build slot attention info string
        let slotInfo = 'No';
        if (modelInfo.use_slot_cross_attention && modelInfo.slot_info) {
            slotInfo = `${modelInfo.slot_info.num_slots} slots`;
        }

        // Build cross-attention info string
        let crossAttnInfo = 'No';
        if (modelInfo.use_cross_attention) {
            if (modelInfo.cross_attention_info && modelInfo.cross_attention_info.num_heads > 1) {
                crossAttnInfo = `${modelInfo.cross_attention_info.num_heads} heads`;
            } else {
                crossAttnInfo = 'Yes';
            }
        }

        container.innerHTML = `
            <div class="info-item">
                <div class="info-label">Layers</div>
                <div class="info-value">${modelInfo.num_layers}</div>
            </div>
            <div class="info-item">
                <div class="info-label">Hidden Dim</div>
                <div class="info-value">${modelInfo.hidden_dim}</div>
            </div>
            <div class="info-item">
                <div class="info-label">Mode</div>
                <div class="info-value">${modelInfo.num_classes === 1 ? 'Binary' : 'Color'}</div>
            </div>
            <div class="info-item">
                <div class="info-label">Attention</div>
                <div class="info-value">${modelInfo.use_attention ? 'Yes' : 'No'}</div>
            </div>
            <div class="info-item">
                <div class="info-label">Cross Attn</div>
                <div class="info-value" style="color: ${modelInfo.use_cross_attention ? '#2ECC40' : '#888'};">${crossAttnInfo}</div>
            </div>
            <div class="info-item">
                <div class="info-label">Slot Attn</div>
                <div class="info-value" style="color: ${modelInfo.use_slot_cross_attention ? '#2ECC40' : '#888'};">${slotInfo}</div>
            </div>
            <div class="info-item">
                <div class="info-label">Encoding</div>
                <div class="info-value">${modelInfo.use_onehot ? 'One-Hot' : 'Learned'}</div>
            </div>
            <div class="info-item">
                <div class="info-label">Size Pred</div>
                <div class="info-value" style="color: ${modelInfo.predict_size ? '#2ECC40' : '#888'};">${modelInfo.predict_size ? 'Yes' : 'No'}</div>
            </div>
        `;
    } catch (err) {
        console.error('Failed to load model info:', err);
    }
}

async function loadKernels() {
    try {
        const response = await fetch('/api/kernels');
        kernelData = await response.json();
        renderKernels();
    } catch (err) {
        console.error('Failed to load kernels:', err);
        document.getElementById('kernel-grid').innerHTML =
            '<p style="color: #ff6b6b;">Failed to load kernel data</p>';
    }
}

function renderKernels() {
    const container = document.getElementById('kernel-grid');
    container.innerHTML = '';

    const kernels = kernelData.first_layer.kernels;  // (10, num_filters, 3, 3)
    const numFilters = kernelData.first_layer.num_filters;

    for (let c = 0; c < 10; c++) {
        const row = document.createElement('div');
        row.className = 'kernel-row';

        // Color swatch
        const swatch = document.createElement('div');
        swatch.className = 'color-swatch';
        swatch.style.backgroundColor = ARC_COLORS[c];
        swatch.style.color = c === 0 || c === 9 ? '#fff' : '#000';
        swatch.textContent = c;
        swatch.title = COLOR_NAMES[c];
        row.appendChild(swatch);

        // Kernels for this color
        for (let f = 0; f < numFilters; f++) {
            const cell = document.createElement('div');
            cell.className = 'kernel-cell';

            const canvas = renderKernelCanvas(kernels[c][f], 50);
            cell.appendChild(canvas);

            const label = document.createElement('span');
            label.className = 'kernel-label';
            label.textContent = `F${f}`;
            cell.appendChild(label);

            row.appendChild(cell);
        }

        container.appendChild(row);
    }
}

// =============================================================================
// Puzzle Browser
// =============================================================================

async function loadPuzzles() {
    try {
        const response = await fetch('/api/puzzles');
        const data = await response.json();
        puzzleList = data.puzzles;
        filteredPuzzles = [...puzzleList];
        renderPuzzleGrid();
    } catch (err) {
        console.error('Failed to load puzzles:', err);
        document.getElementById('puzzle-grid').innerHTML =
            '<p style="color: #ff6b6b;">Failed to load puzzles</p>';
    }
}

function filterPuzzles() {
    const searchTerm = document.getElementById('puzzle-search').value.toLowerCase();
    const exampleType = document.getElementById('example-type').value;

    filteredPuzzles = puzzleList.filter(p => {
        // Filter by search term
        if (searchTerm && !p.id.toLowerCase().includes(searchTerm)) {
            return false;
        }

        // Filter by example type
        if (exampleType === 'train' && p.num_train === 0) return false;
        if (exampleType === 'test' && p.num_test === 0) return false;

        return true;
    });

    renderPuzzleGrid();
}

function renderPuzzleGrid() {
    const container = document.getElementById('puzzle-grid');
    container.innerHTML = '';

    // Limit to first 50 for performance
    const displayPuzzles = filteredPuzzles.slice(0, 50);

    for (const puzzle of displayPuzzles) {
        const card = document.createElement('div');
        card.className = 'puzzle-card';
        if (currentPuzzle && currentPuzzle.id === puzzle.id) {
            card.classList.add('selected');
        }
        card.onclick = () => selectPuzzle(puzzle.id);

        card.innerHTML = `
            <div class="puzzle-card-header">
                <span class="puzzle-id">${puzzle.id}</span>
                <span class="puzzle-meta">${puzzle.num_train} train, ${puzzle.num_test} test</span>
            </div>
            <div class="puzzle-preview" id="preview-${puzzle.id}"></div>
        `;

        container.appendChild(card);

        // Load puzzle preview asynchronously
        loadPuzzlePreview(puzzle.id);
    }

    if (filteredPuzzles.length > 50) {
        const more = document.createElement('p');
        more.style.cssText = 'color: #888; text-align: center; grid-column: 1 / -1; padding: 20px;';
        more.textContent = `Showing 50 of ${filteredPuzzles.length} puzzles. Use search to narrow down.`;
        container.appendChild(more);
    }
}

async function loadPuzzlePreview(puzzleId) {
    try {
        const response = await fetch(`/api/puzzle/${puzzleId}`);
        const puzzle = await response.json();

        const container = document.getElementById(`preview-${puzzleId}`);
        if (!container) return;

        // Show first training example preview
        if (puzzle.train && puzzle.train.length > 0) {
            const ex = puzzle.train[0];

            // Get actual dimensions
            const inputH = ex.input.length;
            const inputW = ex.input[0] ? ex.input[0].length : 0;
            const outputH = ex.output.length;
            const outputW = ex.output[0] ? ex.output[0].length : 0;

            const inputContainer = document.createElement('div');
            inputContainer.className = 'arc-grid-container';
            renderArcGrid(ex.input, inputContainer, 8, {
                cropHeight: inputH,
                cropWidth: inputW
            });

            const arrow = document.createElement('span');
            arrow.textContent = '\u2192';
            arrow.style.color = '#888';

            const outputContainer = document.createElement('div');
            outputContainer.className = 'arc-grid-container';
            renderArcGrid(ex.output, outputContainer, 8, {
                cropHeight: outputH,
                cropWidth: outputW
            });

            container.appendChild(inputContainer);
            container.appendChild(arrow);
            container.appendChild(outputContainer);
        }
    } catch (err) {
        console.error(`Failed to load preview for ${puzzleId}:`, err);
    }
}

async function selectPuzzle(puzzleId) {
    try {
        const response = await fetch(`/api/puzzle/${puzzleId}`);
        currentPuzzle = await response.json();
        currentExampleIdx = 0;

        // Update UI
        document.querySelectorAll('.puzzle-card').forEach(card => {
            card.classList.remove('selected');
        });
        const selectedCard = document.querySelector(`[onclick="selectPuzzle('${puzzleId}')"]`);
        if (selectedCard) selectedCard.classList.add('selected');

        // Update puzzle ID display
        document.getElementById('selected-puzzle-id').textContent = puzzleId;

        // Populate example selector
        populateExampleSelector();

        // Switch to flow tab and load flow data
        showTabDirect('flow');
        await loadFlowData();

    } catch (err) {
        console.error('Failed to select puzzle:', err);
    }
}

function populateExampleSelector() {
    const select = document.getElementById('example-select');
    select.innerHTML = '';
    select.disabled = false;

    const trainExamples = currentPuzzle.train || [];
    const testExamples = currentPuzzle.test || [];

    // Add training examples
    trainExamples.forEach((ex, idx) => {
        const option = document.createElement('option');
        option.value = idx;
        const inputSize = `${ex.input.length}x${ex.input[0]?.length || 0}`;
        const outputSize = `${ex.output.length}x${ex.output[0]?.length || 0}`;
        option.textContent = `Train ${idx + 1}: ${inputSize} → ${outputSize}`;
        select.appendChild(option);
    });

    // Add test examples
    testExamples.forEach((ex, idx) => {
        const option = document.createElement('option');
        option.value = trainExamples.length + idx;
        const inputSize = `${ex.input.length}x${ex.input[0]?.length || 0}`;
        const hasOutput = ex.output && ex.output.length > 0;
        const outputSize = hasOutput ? `${ex.output.length}x${ex.output[0]?.length || 0}` : 'hidden';
        option.textContent = `Test ${idx + 1}: ${inputSize} → ${outputSize}`;
        select.appendChild(option);
    });

    // Set current selection
    select.value = currentExampleIdx;
}

async function changeExample() {
    const select = document.getElementById('example-select');
    currentExampleIdx = parseInt(select.value);
    await loadFlowData();
}

function showTabDirect(tabName) {
    document.querySelectorAll('.tab-content').forEach(el => {
        el.classList.remove('active');
    });
    document.querySelectorAll('.tab-btn').forEach(el => {
        el.classList.remove('active');
        if (el.textContent.toLowerCase().includes(tabName)) {
            el.classList.add('active');
        }
    });
    document.getElementById(`${tabName}-tab`).classList.add('active');
}

// =============================================================================
// Layer Flow Animation
// =============================================================================

async function loadFlowData() {
    if (!currentPuzzle) return;

    try {
        document.getElementById('layer-info').textContent = 'Loading...';

        const response = await fetch(`/api/flow/${currentPuzzle.id}/${currentExampleIdx}`);
        flowData = await response.json();

        // Enable controls
        document.getElementById('prev-layer').disabled = false;
        document.getElementById('next-layer').disabled = false;
        document.getElementById('play-btn').disabled = false;

        // Update slider
        const slider = document.getElementById('layer-slider');
        slider.max = flowData.layers.length - 1;
        slider.value = 0;

        // Display input/output
        renderIOPanel();

        // Display first layer
        currentLayerIdx = 0;
        displayLayer(0);

    } catch (err) {
        console.error('Failed to load flow data:', err);
        document.getElementById('layer-info').textContent = 'Error loading data';
    }
}

function renderIOPanel() {
    const container = document.getElementById('io-panel');
    container.innerHTML = '';

    // Note about evaluation mode
    if (flowData.used_zeros_as_candidate) {
        const note = document.createElement('div');
        note.style.cssText = 'font-size: 0.75em; color: #888; margin-bottom: 10px; text-align: center; padding: 8px; background: #0a1428; border-radius: 4px;';
        note.innerHTML = '<strong>Test Mode:</strong> Model receives (input, zeros) and must predict output from scratch';
        container.appendChild(note);
    }

    // Get actual output size
    const actualH = flowData.output_shape[0];
    const actualW = flowData.output_shape[1];

    // Get predicted size if available
    const hasSizePrediction = flowData.has_size_prediction || false;
    const predH = flowData.predicted_height || actualH;
    const predW = flowData.predicted_width || actualW;

    // Input grid (cropped to actual input size)
    const inputDiv = document.createElement('div');
    inputDiv.className = 'arc-grid-container';
    const inputLabel = document.createElement('div');
    inputLabel.className = 'grid-label';
    inputLabel.textContent = `Input (${flowData.input_shape[0]}x${flowData.input_shape[1]})`;
    inputDiv.appendChild(inputLabel);
    renderArcGrid(flowData.input_grid, inputDiv, 12, {
        cropHeight: flowData.input_shape[0],
        cropWidth: flowData.input_shape[1]
    });
    container.appendChild(inputDiv);

    // Expected output grid (cropped to actual output size)
    const outputDiv = document.createElement('div');
    outputDiv.className = 'arc-grid-container';
    const outputLabel = document.createElement('div');
    outputLabel.className = 'grid-label';
    outputLabel.textContent = `Expected (${actualH}x${actualW})`;
    outputDiv.appendChild(outputLabel);
    renderArcGrid(flowData.output_grid, outputDiv, 12, {
        cropHeight: actualH,
        cropWidth: actualW
    });
    container.appendChild(outputDiv);

    // Prediction grid - crop to predicted size if model has size prediction, else actual size
    const predDiv = document.createElement('div');
    predDiv.className = 'arc-grid-container';
    const predLabel = document.createElement('div');
    predLabel.className = 'grid-label';

    if (hasSizePrediction) {
        // Show predicted size with correctness indicator
        const sizeCorrect = (predH === actualH && predW === actualW);
        const sizeColor = sizeCorrect ? '#2ECC40' : '#FF851B';
        predLabel.innerHTML = `Prediction <span style="color: ${sizeColor};">(${predH}x${predW})</span> <span style="font-size: 0.8em; color: #00d4ff;">(click to trace)</span>`;
    } else {
        predLabel.innerHTML = `Prediction (${actualH}x${actualW}) <span style="font-size: 0.8em; color: #00d4ff;">(click to trace)</span>`;
    }
    predDiv.appendChild(predLabel);

    // Crop prediction to predicted size (model's output) or actual size
    const displayH = hasSizePrediction ? predH : actualH;
    const displayW = hasSizePrediction ? predW : actualW;
    renderArcGrid(flowData.prediction, predDiv, 12, {
        clickable: true,
        onCellClick: handlePixelClick,
        cropHeight: displayH,
        cropWidth: displayW
    });
    container.appendChild(predDiv);

    // Calculate accuracy on the actual output region
    const expected = flowData.output_grid;
    const predicted = flowData.prediction;

    let correct = 0;
    let total = actualH * actualW;
    for (let r = 0; r < actualH; r++) {
        for (let c = 0; c < actualW; c++) {
            if (expected[r][c] === predicted[r][c]) {
                correct++;
            }
        }
    }

    const accuracy = (100 * correct / total).toFixed(1);
    const isPerfect = correct === total;

    // Accuracy display
    const accDiv = document.createElement('div');
    accDiv.style.cssText = `text-align: center; margin-top: 10px; padding: 8px; border-radius: 4px; background: ${isPerfect ? '#0a3020' : '#301a0a'};`;
    accDiv.innerHTML = `
        <span style="color: ${isPerfect ? '#2ECC40' : '#FF851B'}; font-weight: bold;">
            ${isPerfect ? '✓ PERFECT' : `${accuracy}%`}
        </span>
        <span style="color: #888; font-size: 0.85em;"> (${correct}/${total} pixels)</span>
    `;
    container.appendChild(accDiv);

    // Size prediction accuracy if model has it enabled
    if (hasSizePrediction) {
        const sizeCorrect = (predH === actualH && predW === actualW);
        const heightCorrect = (predH === actualH);
        const widthCorrect = (predW === actualW);

        const sizeDiv = document.createElement('div');
        sizeDiv.style.cssText = 'margin-top: 8px; padding: 8px; border-radius: 4px; background: #0a1428; font-size: 0.85em;';
        sizeDiv.innerHTML = `
            <div style="color: #888; margin-bottom: 4px;">Size Prediction:</div>
            <div style="display: flex; gap: 15px; justify-content: center;">
                <span>Height: <span style="color: ${heightCorrect ? '#2ECC40' : '#FF4136'};">${predH}</span> / ${actualH} ${heightCorrect ? '✓' : '✗'}</span>
                <span>Width: <span style="color: ${widthCorrect ? '#2ECC40' : '#FF4136'};">${predW}</span> / ${actualW} ${widthCorrect ? '✓' : '✗'}</span>
            </div>
        `;
        container.appendChild(sizeDiv);
    }
}

function displayLayer(idx) {
    idx = parseInt(idx);
    currentLayerIdx = idx;

    const layer = flowData.layers[idx];
    if (!layer) return;

    // Update slider
    document.getElementById('layer-slider').value = idx;

    // Update layer info
    const [C, H, W] = layer.shape;
    const isOutput = layer.is_output || false;
    document.getElementById('layer-info').textContent =
        `${layer.name} (${C} channels, ${H}x${W})${isOutput ? ' - OUTPUT' : ''}`;

    // Render activations
    const actPanel = document.getElementById('activations-panel');
    actPanel.innerHTML = '';

    // For output layer, show special header
    if (isOutput) {
        const header = document.createElement('div');
        header.style.cssText = 'color: #00d4ff; margin-bottom: 10px; font-size: 0.9em;';
        header.textContent = 'Per-Color Logits (higher = model predicts this color)';
        actPanel.appendChild(header);
    }

    const grid = document.createElement('div');
    grid.className = 'activations-grid';

    // For output layer, show all 10 color channels with labels
    // For other layers, show up to 16 channels
    const numToShow = isOutput ? C : Math.min(C, 16);

    for (let c = 0; c < numToShow; c++) {
        const cell = document.createElement('div');
        cell.className = 'activation-cell';

        // For output layer, use probabilities for visualization
        const data = isOutput && layer.probabilities ? layer.probabilities[c] : layer.activations[c];
        const canvas = renderActivationCanvas(data, 80);
        cell.appendChild(canvas);

        const label = document.createElement('span');
        label.className = 'channel-label';

        if (isOutput && layer.color_names) {
            // Show color name with swatch
            const swatch = document.createElement('span');
            swatch.style.cssText = `display: inline-block; width: 10px; height: 10px; background: ${ARC_COLORS[c]}; margin-right: 4px; border-radius: 2px;`;
            label.appendChild(swatch);
            label.appendChild(document.createTextNode(layer.color_names[c]));
        } else {
            label.textContent = `Ch ${c}`;
        }
        cell.appendChild(label);

        grid.appendChild(cell);
    }

    actPanel.appendChild(grid);

    // Stats
    const stats = document.createElement('div');
    stats.className = 'stats-grid';
    stats.innerHTML = `
        <div class="stat-item">
            <div class="stat-label">Min</div>
            <div class="stat-value">${layer.stats.min.toFixed(3)}</div>
        </div>
        <div class="stat-item">
            <div class="stat-label">Max</div>
            <div class="stat-value">${layer.stats.max.toFixed(3)}</div>
        </div>
        <div class="stat-item">
            <div class="stat-label">Mean</div>
            <div class="stat-value">${layer.stats.mean.toFixed(3)}</div>
        </div>
        <div class="stat-item">
            <div class="stat-label">Std</div>
            <div class="stat-value">${layer.stats.std.toFixed(3)}</div>
        </div>
    `;
    actPanel.appendChild(stats);

    // Render overlay
    const overlayPanel = document.getElementById('overlay-panel');

    // Get actual and predicted sizes for cropping
    const actualH = flowData.output_shape[0];
    const actualW = flowData.output_shape[1];
    const hasSizePrediction = flowData.has_size_prediction || false;
    const predH = flowData.predicted_height || actualH;
    const predW = flowData.predicted_width || actualW;
    const displayH = hasSizePrediction ? predH : actualH;
    const displayW = hasSizePrediction ? predW : actualW;

    if (isOutput) {
        // For output layer, show prediction with color legend
        const sizeInfo = hasSizePrediction ? ` (${displayH}x${displayW})` : ` (${actualH}x${actualW})`;
        overlayPanel.innerHTML = `<h4 style="color: #888; margin-bottom: 10px;">Model Prediction${sizeInfo}</h4>`;
        const predContainer = document.createElement('div');
        predContainer.className = 'arc-grid-container';
        renderArcGrid(flowData.prediction, predContainer, 6, {
            cropHeight: displayH,
            cropWidth: displayW
        });
        overlayPanel.appendChild(predContainer);
    } else {
        overlayPanel.innerHTML = '<h4 style="color: #888; margin-bottom: 10px;">Mean Activation</h4>';
        const overlayCanvas = renderOverlayCanvas(layer.mean_activation, flowData.input_grid, 180);
        overlayPanel.appendChild(overlayCanvas);
    }
}

function prevLayer() {
    if (currentLayerIdx > 0) {
        displayLayer(currentLayerIdx - 1);
    }
}

function nextLayer() {
    if (flowData && currentLayerIdx < flowData.layers.length - 1) {
        displayLayer(currentLayerIdx + 1);
    }
}

function togglePlay() {
    isPlaying = !isPlaying;
    const btn = document.getElementById('play-btn');

    if (isPlaying) {
        btn.textContent = 'Pause';
        animate();
    } else {
        btn.textContent = 'Play';
        if (playInterval) {
            clearTimeout(playInterval);
            playInterval = null;
        }
    }
}

function animate() {
    if (!isPlaying || !flowData) return;

    const speed = parseInt(document.getElementById('speed-slider').value);

    if (currentLayerIdx < flowData.layers.length - 1) {
        displayLayer(currentLayerIdx + 1);
        playInterval = setTimeout(animate, speed);
    } else {
        // Loop back to start
        displayLayer(0);
        playInterval = setTimeout(animate, speed);
    }
}

// =============================================================================
// Pixel Trace Functions
// =============================================================================

async function handlePixelClick(row, col) {
    if (!currentPuzzle || !flowData) return;

    selectedPixel = { row, col };

    // Show the pixel trace panel
    const panel = document.getElementById('pixel-trace-panel');
    panel.style.display = 'block';

    // Show loading state
    const content = document.getElementById('pixel-trace-content');
    content.innerHTML = '<div class="loading">Loading pixel trace</div>';

    try {
        const response = await fetch(`/api/pixel-trace/${currentPuzzle.id}/${currentExampleIdx}/${row}/${col}`);
        const traceData = await response.json();

        if (traceData.error) {
            content.innerHTML = `<p style="color: #ff6b6b;">Error: ${traceData.error}</p>`;
            return;
        }

        renderPixelTrace(traceData);

        // Scroll to the trace panel
        panel.scrollIntoView({ behavior: 'smooth', block: 'start' });
    } catch (err) {
        console.error('Failed to load pixel trace:', err);
        content.innerHTML = '<p style="color: #ff6b6b;">Failed to load pixel trace</p>';
    }
}

function closePixelTrace() {
    const panel = document.getElementById('pixel-trace-panel');
    panel.style.display = 'none';
    selectedPixel = null;
}

function renderPixelTrace(trace) {
    const content = document.getElementById('pixel-trace-content');
    content.innerHTML = '';

    const grid = document.createElement('div');
    grid.className = 'trace-grid';

    // 1. Pixel Summary Section
    grid.appendChild(createPixelSummarySection(trace));

    // 2. Receptive Field Section
    grid.appendChild(createReceptiveFieldSection(trace));

    // 3. Embeddings Section
    grid.appendChild(createEmbeddingsSection(trace));

    // 4. Feature Vector Section
    grid.appendChild(createFeatureVectorSection(trace));

    // 5. Attention Section (if model has attention)
    if (trace.attention && trace.attention.has_attention) {
        grid.appendChild(createAttentionSection(trace));
    }

    // 5b. Slot Attention Section (if model has slot cross-attention)
    if (trace.slot_attention && trace.slot_attention.has_slot_attention) {
        grid.appendChild(createSlotAttentionSection(trace));
    }

    // 6. Final Calculation Section
    grid.appendChild(createFinalCalculationSection(trace));

    // 7. Prediction Summary Section
    grid.appendChild(createPredictionSummarySection(trace));

    content.appendChild(grid);
}

function createPixelSummarySection(trace) {
    const section = document.createElement('div');
    section.className = 'trace-section';

    const pixel = trace.pixel;
    const isCorrect = pixel.is_correct;

    section.innerHTML = `
        <h4>Pixel Location</h4>
        <div class="trace-row">
            <span class="trace-label">Position</span>
            <span class="trace-value">(${pixel.row}, ${pixel.col})</span>
        </div>
        <div class="trace-row">
            <span class="trace-label">Input Color</span>
            <span class="trace-value">
                <span class="color-indicator">
                    <span class="color-swatch-small" style="background: ${ARC_COLORS[pixel.input_color]}"></span>
                    ${COLOR_NAMES[pixel.input_color]} (${pixel.input_color})
                </span>
            </span>
        </div>
        <div class="trace-row">
            <span class="trace-label">Expected Color</span>
            <span class="trace-value">
                <span class="color-indicator">
                    <span class="color-swatch-small" style="background: ${ARC_COLORS[pixel.expected_color]}"></span>
                    ${COLOR_NAMES[pixel.expected_color]} (${pixel.expected_color})
                </span>
            </span>
        </div>
        <div class="trace-row">
            <span class="trace-label">Predicted Color</span>
            <span class="trace-value ${isCorrect ? 'correct' : 'incorrect'}">
                <span class="color-indicator">
                    <span class="color-swatch-small" style="background: ${ARC_COLORS[pixel.predicted_color]}"></span>
                    ${COLOR_NAMES[pixel.predicted_color]} (${pixel.predicted_color})
                    ${isCorrect ? ' ✓' : ' ✗'}
                </span>
            </span>
        </div>
    `;

    return section;
}

function createReceptiveFieldSection(trace) {
    const section = document.createElement('div');
    section.className = 'trace-section';

    const rf = trace.receptive_field;
    const bounds = rf.bounds;

    let html = `
        <h4>Receptive Field (${rf.size}×${rf.size})</h4>
        <p style="font-size: 0.8em; color: #888; margin-bottom: 10px;">
            The model looks at this patch of pixels to make its prediction.
        </p>
        <div class="rf-container">
    `;

    // Input patch
    html += `
        <div class="rf-grid-wrapper">
            <span class="rf-grid-label">Input Patch</span>
            <div class="rf-grid" style="grid-template-columns: repeat(${rf.input_patch[0].length}, 24px);">
    `;

    const centerLocalRow = trace.pixel.row - bounds.row_start;
    const centerLocalCol = trace.pixel.col - bounds.col_start;

    for (let r = 0; r < rf.input_patch.length; r++) {
        for (let c = 0; c < rf.input_patch[r].length; c++) {
            const colorVal = rf.input_patch[r][c];
            const isCenter = (r === centerLocalRow && c === centerLocalCol);
            const textColor = colorVal === 0 || colorVal === 9 ? '#fff' : '#000';
            html += `<div class="rf-cell ${isCenter ? 'center' : ''}"
                         style="background: ${ARC_COLORS[colorVal]}; color: ${textColor};">
                         ${colorVal}
                     </div>`;
        }
    }

    html += `</div></div>`;

    // Candidate patch (zeros in test mode)
    html += `
        <div class="rf-grid-wrapper">
            <span class="rf-grid-label">Candidate Patch</span>
            <div class="rf-grid" style="grid-template-columns: repeat(${rf.candidate_patch[0].length}, 24px);">
    `;

    for (let r = 0; r < rf.candidate_patch.length; r++) {
        for (let c = 0; c < rf.candidate_patch[r].length; c++) {
            const colorVal = rf.candidate_patch[r][c];
            const isCenter = (r === centerLocalRow && c === centerLocalCol);
            const textColor = colorVal === 0 || colorVal === 9 ? '#fff' : '#000';
            html += `<div class="rf-cell ${isCenter ? 'center' : ''}"
                         style="background: ${ARC_COLORS[colorVal]}; color: ${textColor};">
                         ${colorVal}
                     </div>`;
        }
    }

    html += `</div></div></div>`;

    // Colors in receptive field
    html += `
        <div style="margin-top: 10px;">
            <span style="font-size: 0.8em; color: #888;">Colors in input RF: </span>
            ${rf.colors_in_input_rf.map(c =>
                `<span class="color-swatch-small" style="background: ${ARC_COLORS[c]}; display: inline-block; margin: 0 2px;" title="${COLOR_NAMES[c]}"></span>`
            ).join('')}
        </div>
    `;

    section.innerHTML = html;
    return section;
}

function createEmbeddingsSection(trace) {
    const section = document.createElement('div');
    section.className = 'trace-section';

    const emb = trace.embeddings;

    const isOnehot = emb.encoding_type === 'onehot';
    const embedDim = emb.embed_dim || (isOnehot ? 11 : 16);

    let html = `
        <h4>${isOnehot ? 'One-Hot Encoding' : 'Embedding Vectors'} (${embedDim}-dim each)</h4>
        <p style="font-size: 0.8em; color: #888; margin-bottom: 10px;">
            ${isOnehot ? 'Fixed one-hot + nonzero mask encoding for colors.' : 'Learned vector representations for the colors at this position.'}
        </p>
    `;

    // Helper to render embedding as colored cells
    function renderEmbedding(values, label) {
        if (!values) return '';
        const maxAbs = Math.max(...values.map(Math.abs)) || 1;
        let embHtml = `<div style="margin-bottom: 8px;">
            <span style="font-size: 0.8em; color: #aaa;">${label}:</span>
            <div class="embedding-row">`;
        for (const val of values) {
            const normalized = val / maxAbs;
            const color = rdBuColor(normalized);
            embHtml += `<div class="embedding-cell" style="background: ${color};" title="${val.toFixed(3)}"></div>`;
        }
        embHtml += `</div></div>`;
        return embHtml;
    }

    html += renderEmbedding(emb.input_embedding, 'Input');
    html += renderEmbedding(emb.candidate_embedding, 'Candidate');

    // Only show difference/product if they exist (older models with force_comparison)
    if (emb.difference) {
        html += renderEmbedding(emb.difference, 'Difference');
    }
    if (emb.product) {
        html += renderEmbedding(emb.product, 'Product');
    }

    html += `
        <div style="margin-top: 10px; font-size: 0.8em; color: #888;">
            Combined input to conv: ${emb.combined_input.length} dimensions
        </div>
    `;

    section.innerHTML = html;
    return section;
}

function createFeatureVectorSection(trace) {
    const section = document.createElement('div');
    section.className = 'trace-section';

    const feat = trace.feature_vector;
    const pred = trace.prediction;
    const values = feat.values;
    const contributions = feat.contributions;  // (10, hidden_dim)
    const maxAbs = Math.max(...values.map(Math.abs)) || 1;

    // Find max absolute contribution for normalization
    let maxContribAbs = 0;
    for (const row of contributions) {
        for (const val of row) {
            maxContribAbs = Math.max(maxContribAbs, Math.abs(val));
        }
    }
    if (maxContribAbs === 0) maxContribAbs = 1;

    let html = `
        <h4>Feature Vector & Contributions</h4>
        <p style="font-size: 0.8em; color: #888; margin-bottom: 10px;">
            ${feat.hidden_dim}-dim features. Heatmap shows how each feature contributes to each color's logit.
        </p>
    `;

    // Feature value bars (original visualization, compact)
    html += `<div style="margin-bottom: 12px;">
        <span style="font-size: 0.75em; color: #666;">Feature values:</span>
        <div class="feature-bars" style="height: 30px;">`;
    for (const val of values) {
        const height = Math.abs(val) / maxAbs * 25;
        const isNegative = val < 0;
        html += `<div class="feature-bar ${isNegative ? 'negative' : ''}"
                     style="height: ${height}px;"
                     title="${val.toFixed(4)}"></div>`;
    }
    html += `</div></div>`;

    // Contribution heatmap: rows = colors, columns = feature dimensions
    html += `
        <div style="margin-bottom: 8px;">
            <span style="font-size: 0.75em; color: #666;">Per-feature contributions (feature × weight):</span>
        </div>
        <div style="overflow-x: auto;">
            <div style="display: flex; flex-direction: column; gap: 2px; min-width: fit-content;">
    `;

    // Header row with feature indices (show every 4th for readability)
    html += `<div style="display: flex; align-items: center; gap: 2px;">
        <div style="width: 70px; flex-shrink: 0;"></div>`;
    for (let d = 0; d < feat.hidden_dim; d++) {
        if (d % 4 === 0) {
            html += `<div style="width: 10px; font-size: 0.6em; color: #555; text-align: center;">${d}</div>`;
        } else {
            html += `<div style="width: 10px;"></div>`;
        }
    }
    html += `</div>`;

    // One row per color
    for (let c = 0; c < 10; c++) {
        const isPredicted = c === pred.predicted_color;
        const isExpected = c === pred.expected_color;
        const rowContribs = contributions[c];
        const rowSum = rowContribs.reduce((a, b) => a + b, 0);

        // Row label with color swatch
        const labelStyle = isPredicted ? 'color: #2ECC40; font-weight: bold;' :
                          (isExpected ? 'color: #FFDC00; font-weight: bold;' : 'color: #888;');
        const marker = isPredicted ? ' ✓' : (isExpected ? ' ◎' : '');

        html += `<div style="display: flex; align-items: center; gap: 2px;">
            <div style="width: 70px; flex-shrink: 0; display: flex; align-items: center; gap: 4px;">
                <div style="width: 12px; height: 12px; background: ${ARC_COLORS[c]}; border-radius: 2px; flex-shrink: 0;"></div>
                <span style="font-size: 0.7em; ${labelStyle}">${COLOR_NAMES[c]}${marker}</span>
            </div>`;

        // Contribution cells
        for (let d = 0; d < feat.hidden_dim; d++) {
            const contrib = rowContribs[d];
            const normalized = contrib / maxContribAbs;
            const color = rdBuColor(normalized);
            html += `<div style="width: 10px; height: 12px; background: ${color}; border-radius: 1px;"
                         title="${COLOR_NAMES[c]} dim${d}: ${contrib.toFixed(4)}"></div>`;
        }

        // Row sum (total contribution to this color's logit before bias)
        html += `<div style="width: 45px; font-size: 0.65em; color: #888; text-align: right; padding-left: 4px;"
                     title="Sum of feature contributions (before bias)">${rowSum.toFixed(2)}</div>`;
        html += `</div>`;
    }

    html += `</div></div>`;

    // Legend
    html += `
        <div style="display: flex; align-items: center; gap: 8px; margin-top: 8px; font-size: 0.75em; color: #666;">
            <span>Contribution:</span>
            <span style="color: #2166ac;">−</span>
            <div style="width: 60px; height: 10px; background: linear-gradient(to right, #2166ac, #f7f7f7, #b2182b); border-radius: 2px;"></div>
            <span style="color: #b2182b;">+</span>
            <span style="margin-left: 10px;">✓ = predicted</span>
            <span>◎ = expected</span>
        </div>
    `;

    section.innerHTML = html;
    return section;
}

function createAttentionSection(trace) {
    const section = document.createElement('div');
    section.className = 'trace-section';

    const attn = trace.attention;
    const pixel = trace.pixel;
    const isCrossAttention = attn.attention_type === 'cross';
    const numHeads = attn.num_heads || 1;
    const hasMultiHead = numHeads > 1 && attn.per_head;

    const title = isCrossAttention
        ? `Cross-Attention (Output → Input)${hasMultiHead ? ` - ${numHeads} Heads` : ''}`
        : 'Spatial Self-Attention';
    const description = isCrossAttention
        ? 'Output pixel queries input pixels for global context. Brighter = higher attention weight.'
        : 'This pixel attends to all other pixels. Brighter = higher attention weight.';

    let html = `
        <h4>${title}</h4>
        <p style="font-size: 0.8em; color: #888; margin-bottom: 10px;">
            ${description}
        </p>
    `;

    // Self-attention weight and entropy (averaged if multi-head)
    const avgLabel = hasMultiHead ? ' (averaged)' : '';
    html += `
        <div class="trace-row">
            <span class="trace-label">Self-attention weight${avgLabel}</span>
            <span class="trace-value">${(attn.self_attention_weight * 100).toFixed(2)}%</span>
        </div>
        <div class="trace-row">
            <span class="trace-label">Attention entropy${avgLabel}</span>
            <span class="trace-value">${attn.stats.from_entropy.toFixed(2)} (higher = more uniform)</span>
        </div>
    `;

    // For multi-head: add head selector tabs
    if (hasMultiHead) {
        const tabStyle = 'padding: 4px 10px; border: 1px solid #444; border-radius: 4px; background: #1a2a3a; color: #aaa; cursor: pointer; font-size: 0.8em;';
        const activeTabStyle = 'padding: 4px 10px; border: 1px solid #00d4ff; border-radius: 4px; background: #0a3040; color: #00d4ff; cursor: pointer; font-size: 0.8em;';
        html += `
            <div style="margin: 15px 0 10px 0;">
                <div style="font-size: 0.85em; color: #aaa; margin-bottom: 8px;">Select attention head:</div>
                <div class="head-tabs" style="display: flex; gap: 4px; flex-wrap: wrap;">
                    <button class="head-tab active" data-head="avg" style="${activeTabStyle}" onclick="selectAttentionHead(this, 'avg')">Averaged</button>
        `;
        for (let h = 0; h < numHeads; h++) {
            html += `<button class="head-tab" data-head="${h}" style="${tabStyle}" onclick="selectAttentionHead(this, ${h})">Head ${h + 1}</button>`;
        }
        html += `</div></div>`;
    }

    // Store attention data globally for head tab selection
    window.currentAttentionData = { attn, pixel, isCrossAttention };

    // Attention heatmaps container (will be updated by head selection)
    html += `<div id="attention-heatmaps-container">`;
    html += renderAttentionHeatmaps(attn, pixel, isCrossAttention, 'avg');
    html += `</div>`;

    // Legend
    const legendFrom = isCrossAttention ? 'Queries (orange)' : 'Attends to (orange)';
    const legendTo = isCrossAttention ? 'Queried by (blue)' : 'Attended by (blue)';
    html += `
        <div style="margin-top: 12px; font-size: 0.75em; color: #666; display: flex; gap: 15px;">
            <span><span style="display: inline-block; width: 12px; height: 12px; background: rgb(255, 128, 0); border-radius: 2px;"></span> ${legendFrom}</span>
            <span><span style="display: inline-block; width: 12px; height: 12px; background: rgb(0, 128, 255); border-radius: 2px;"></span> ${legendTo}</span>
            <span><span style="display: inline-block; width: 12px; height: 12px; border: 1px solid #ff0; border-radius: 2px;"></span> Selected pixel</span>
        </div>
    `;

    section.innerHTML = html;
    return section;
}

// Helper function to render attention heatmaps (used for both averaged and per-head)
function renderAttentionHeatmaps(attn, pixel, isCrossAttention, headIdx) {
    let fromGrid, toGrid, maxFrom, maxTo, topAttended, stats;

    if (headIdx === 'avg' || !attn.per_head) {
        // Use averaged attention
        fromGrid = attn.attention_from_pixel;
        toGrid = attn.attention_to_pixel;
        maxFrom = attn.stats.from_max;
        maxTo = attn.stats.to_max;
        topAttended = attn.top_attended;
        stats = attn.stats;
    } else {
        // Use per-head attention
        const h = parseInt(headIdx);
        fromGrid = attn.per_head.attention_from_pixel[h];
        toGrid = attn.per_head.attention_to_pixel[h];
        stats = attn.per_head.stats[h];
        maxFrom = stats.from_max;
        maxTo = stats.to_max;
        topAttended = attn.per_head.top_attended[h];
    }

    let html = `<div style="display: flex; gap: 20px; margin-top: 15px; flex-wrap: wrap;">`;

    // "Attention FROM this pixel" heatmap
    const fromLabel = isCrossAttention ? 'This output pixel queries input:' : 'This pixel attends to:';
    html += `
        <div>
            <div style="font-size: 0.85em; color: #aaa; margin-bottom: 8px;">${fromLabel}</div>
            <div class="attention-grid" style="display: grid; grid-template-columns: repeat(30, 8px); gap: 1px;">
    `;

    for (let r = 0; r < 30; r++) {
        for (let c = 0; c < 30; c++) {
            const weight = fromGrid[r][c];
            const normalized = maxFrom > 0 ? weight / maxFrom : 0;
            const intensity = Math.floor(normalized * 255);
            const isSelected = (r === pixel.row && c === pixel.col);
            const border = isSelected ? 'border: 1px solid #ff0;' : '';
            html += `<div style="width: 8px; height: 8px; background: rgb(${intensity}, ${Math.floor(intensity*0.5)}, 0); ${border}"
                         title="(${r},${c}): ${(weight * 100).toFixed(2)}%"></div>`;
        }
    }
    html += `</div></div>`;

    // "Attention TO this pixel" heatmap
    const toLabel = isCrossAttention ? 'Input queried by output pixels:' : 'Others attend to this pixel:';
    html += `
        <div>
            <div style="font-size: 0.85em; color: #aaa; margin-bottom: 8px;">${toLabel}</div>
            <div class="attention-grid" style="display: grid; grid-template-columns: repeat(30, 8px); gap: 1px;">
    `;

    for (let r = 0; r < 30; r++) {
        for (let c = 0; c < 30; c++) {
            const weight = toGrid[r][c];
            const normalized = maxTo > 0 ? weight / maxTo : 0;
            const intensity = Math.floor(normalized * 255);
            const isSelected = (r === pixel.row && c === pixel.col);
            const border = isSelected ? 'border: 1px solid #ff0;' : '';
            html += `<div style="width: 8px; height: 8px; background: rgb(0, ${Math.floor(intensity*0.5)}, ${intensity}); ${border}"
                         title="(${r},${c}): ${(weight * 100).toFixed(2)}%"></div>`;
        }
    }
    html += `</div></div>`;
    html += `</div>`;

    // Top attended pixels
    const topLabel = isCrossAttention ? 'Top input pixels queried:' : 'Top pixels this one attends to:';
    html += `
        <div style="margin-top: 15px;">
            <div style="font-size: 0.85em; color: #aaa; margin-bottom: 8px;">${topLabel}</div>
            <div style="display: flex; gap: 10px; flex-wrap: wrap;">
    `;
    const numTop = topAttended ? Math.min(5, topAttended.length) : 0;
    for (let i = 0; i < numTop; i++) {
        const p = topAttended[i];
        html += `
            <div style="padding: 4px 8px; background: #1a2a3a; border-radius: 4px; font-size: 0.8em;">
                (${p.row}, ${p.col}): <span style="color: #f90;">${(p.weight * 100).toFixed(1)}%</span>
            </div>
        `;
    }
    html += `</div></div>`;

    // Stats for this head/average
    html += `
        <div style="margin-top: 10px; font-size: 0.8em; color: #666;">
            Max: ${(stats.from_max * 100).toFixed(1)}% | Mean: ${(stats.from_mean * 100).toFixed(2)}% | Entropy: ${stats.from_entropy.toFixed(2)}
        </div>
    `;

    return html;
}

// Global function to handle head tab selection
function selectAttentionHead(button, headIdx) {
    const tabStyle = 'padding: 4px 10px; border: 1px solid #444; border-radius: 4px; background: #1a2a3a; color: #aaa; cursor: pointer; font-size: 0.8em;';
    const activeTabStyle = 'padding: 4px 10px; border: 1px solid #00d4ff; border-radius: 4px; background: #0a3040; color: #00d4ff; cursor: pointer; font-size: 0.8em;';

    // Update active tab styling
    document.querySelectorAll('.head-tab').forEach(tab => {
        tab.classList.remove('active');
        tab.style.cssText = tabStyle;
    });
    button.classList.add('active');
    button.style.cssText = activeTabStyle;

    // Get the container for heatmaps
    const container = document.getElementById('attention-heatmaps-container');
    if (!container) return;

    // Get the trace data from the globally stored attention data
    if (window.currentAttentionData) {
        const attn = window.currentAttentionData.attn;
        const pixel = window.currentAttentionData.pixel;
        const isCrossAttention = window.currentAttentionData.isCrossAttention;
        container.innerHTML = renderAttentionHeatmaps(attn, pixel, isCrossAttention, headIdx);
    }
}

function createSlotAttentionSection(trace) {
    const section = document.createElement('div');
    section.className = 'trace-section';

    const slot = trace.slot_attention;
    const pixel = trace.pixel;
    const K = slot.num_slots;

    let html = `
        <h4>Slot-Routed Cross-Attention (${K} slots)</h4>
        <p style="font-size: 0.8em; color: #888; margin-bottom: 10px;">
            Output pixel queries objects via slots. Each slot represents a discovered object/pattern in the input.
        </p>
    `;

    // Slot attention weights bar chart
    html += `
        <div style="margin-bottom: 15px;">
            <div style="font-size: 0.85em; color: #aaa; margin-bottom: 8px;">Slot Attention Weights (which objects this pixel attends to):</div>
            <div style="display: flex; align-items: end; gap: 4px; height: 60px;">
    `;

    const maxSlotWeight = Math.max(...slot.slot_attention_weights);
    for (let k = 0; k < K; k++) {
        const weight = slot.slot_attention_weights[k];
        const height = (weight / maxSlotWeight) * 50;
        const isMax = weight === maxSlotWeight;
        const color = isMax ? '#2ECC40' : '#0074D9';
        html += `
            <div style="display: flex; flex-direction: column; align-items: center; gap: 2px;">
                <div style="width: 30px; height: ${height}px; background: ${color}; border-radius: 2px 2px 0 0;"
                     title="Slot ${k}: ${(weight * 100).toFixed(1)}%"></div>
                <span style="font-size: 0.7em; color: #888;">S${k}</span>
                <span style="font-size: 0.65em; color: ${isMax ? '#2ECC40' : '#666'};">${(weight * 100).toFixed(0)}%</span>
            </div>
        `;
    }
    html += `</div></div>`;

    // Stats
    html += `
        <div class="trace-row">
            <span class="trace-label">Slot attention entropy</span>
            <span class="trace-value">${slot.stats.slot_attn_entropy.toFixed(2)} (higher = more uniform)</span>
        </div>
        <div class="trace-row">
            <span class="trace-label">Combined attention entropy</span>
            <span class="trace-value">${slot.stats.combined_attn_entropy.toFixed(2)}</span>
        </div>
    `;

    // Slot masks visualization
    html += `
        <div style="margin-top: 15px;">
            <div style="font-size: 0.85em; color: #aaa; margin-bottom: 8px;">Slot Masks (what input pixels each slot covers):</div>
            <div style="display: flex; gap: 10px; flex-wrap: wrap;">
    `;

    for (let k = 0; k < K; k++) {
        const mask = slot.slot_masks[k];
        const weight = slot.slot_attention_weights[k];
        const isTopSlot = weight === maxSlotWeight;
        const borderStyle = isTopSlot ? 'border: 2px solid #2ECC40;' : 'border: 1px solid #333;';

        // Find max value in mask for normalization
        let maxMask = 0;
        for (const row of mask) {
            for (const val of row) {
                maxMask = Math.max(maxMask, val);
            }
        }
        if (maxMask === 0) maxMask = 1;

        html += `
            <div style="${borderStyle} border-radius: 4px; padding: 4px;">
                <div style="font-size: 0.7em; color: ${isTopSlot ? '#2ECC40' : '#888'}; text-align: center; margin-bottom: 2px;">
                    Slot ${k} (${(weight * 100).toFixed(0)}%)
                </div>
                <div style="display: grid; grid-template-columns: repeat(30, 4px); gap: 0;">
        `;

        for (let r = 0; r < 30; r++) {
            for (let c = 0; c < 30; c++) {
                const val = mask[r] ? (mask[r][c] || 0) : 0;
                const normalized = val / maxMask;
                const intensity = Math.floor(normalized * 255);
                const isSelected = (r === pixel.row && c === pixel.col);
                const border = isSelected ? 'border: 1px solid #ff0;' : '';
                html += `<div style="width: 4px; height: 4px; background: rgb(${intensity}, ${Math.floor(intensity * 0.6)}, 0); ${border}"
                             title="(${r},${c}): ${(val * 100).toFixed(1)}%"></div>`;
            }
        }

        html += `</div></div>`;
    }
    html += `</div></div>`;

    // Combined attention map (final attention over input pixels)
    html += `
        <div style="margin-top: 15px;">
            <div style="font-size: 0.85em; color: #aaa; margin-bottom: 8px;">Combined Attention (final attention over input after slot routing):</div>
            <div style="display: flex; gap: 20px; align-items: start;">
                <div>
                    <div style="display: grid; grid-template-columns: repeat(30, 8px); gap: 1px;">
    `;

    const combinedAttn = slot.combined_attention;
    const maxCombined = slot.stats.combined_attn_max;
    for (let r = 0; r < 30; r++) {
        for (let c = 0; c < 30; c++) {
            const weight = combinedAttn[r] ? (combinedAttn[r][c] || 0) : 0;
            const normalized = weight / maxCombined;
            const intensity = Math.floor(normalized * 255);
            const isSelected = (r === pixel.row && c === pixel.col);
            const border = isSelected ? 'border: 1px solid #ff0;' : '';
            html += `<div style="width: 8px; height: 8px; background: rgb(${intensity}, ${Math.floor(intensity * 0.5)}, 0); ${border}"
                         title="(${r},${c}): ${(weight * 100).toFixed(2)}%"></div>`;
        }
    }

    html += `</div></div>`;

    // Top attended input pixels
    html += `
                <div>
                    <div style="font-size: 0.8em; color: #aaa; margin-bottom: 8px;">Top input pixels:</div>
                    <div style="display: flex; flex-direction: column; gap: 4px;">
    `;

    for (let i = 0; i < Math.min(5, slot.top_attended_inputs.length); i++) {
        const p = slot.top_attended_inputs[i];
        html += `
            <div style="padding: 4px 8px; background: #1a2a3a; border-radius: 4px; font-size: 0.8em;">
                (${p.row}, ${p.col}): <span style="color: #f90;">${(p.weight * 100).toFixed(2)}%</span>
            </div>
        `;
    }

    html += `</div></div></div></div>`;

    // Legend
    html += `
        <div style="margin-top: 12px; font-size: 0.75em; color: #666; display: flex; gap: 15px;">
            <span><span style="display: inline-block; width: 12px; height: 12px; background: rgb(255, 128, 0); border-radius: 2px;"></span> High attention</span>
            <span><span style="display: inline-block; width: 12px; height: 12px; background: rgb(0, 0, 0); border-radius: 2px; border: 1px solid #444;"></span> Low attention</span>
            <span><span style="display: inline-block; width: 12px; height: 12px; border: 2px solid #2ECC40; border-radius: 2px;"></span> Top slot</span>
            <span><span style="display: inline-block; width: 12px; height: 12px; border: 1px solid #ff0; border-radius: 2px;"></span> Selected pixel</span>
        </div>
    `;

    section.innerHTML = html;
    return section;
}

function createFinalCalculationSection(trace) {
    const section = document.createElement('div');
    section.className = 'trace-section';

    const out = trace.output_layer;
    const pred = trace.prediction;

    // Find the top 3 classes by probability
    const sortedClasses = out.per_class_calculation
        .map((c, i) => ({ ...c, idx: i }))
        .sort((a, b) => b.probability - a.probability)
        .slice(0, 3);

    let html = `
        <h4>Output Layer Calculation</h4>
        <p style="font-size: 0.8em; color: #888; margin-bottom: 10px;">
            The outc layer computes: logit = weight · features + bias
        </p>
    `;

    // Show calculation for top 3 classes
    for (const cls of sortedClasses) {
        const isExpected = cls.color === pred.expected_color;
        const isPredicted = cls.color === pred.predicted_color;
        let label = '';
        if (isPredicted && isExpected) label = ' (Predicted & Expected)';
        else if (isPredicted) label = ' (Predicted)';
        else if (isExpected) label = ' (Expected)';

        html += `
            <div style="margin-bottom: 12px; padding: 8px; background: #0a1428; border-radius: 4px;">
                <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 6px;">
                    <span class="color-swatch-small" style="background: ${ARC_COLORS[cls.color]}"></span>
                    <strong style="color: ${isPredicted ? '#2ECC40' : (isExpected ? '#FFDC00' : '#eee')}">${cls.color_name}${label}</strong>
                </div>
                <div class="calc-breakdown">
                    <div class="calc-line">
                        <span class="calc-operator">weighted sum:</span> ${cls.weighted_sum.toFixed(4)}
                    </div>
                    <div class="calc-line">
                        <span class="calc-operator">+ bias:</span> ${cls.bias.toFixed(4)}
                    </div>
                    <div class="calc-line">
                        <span class="calc-operator">= logit:</span> <span class="calc-result">${cls.logit.toFixed(4)}</span>
                    </div>
                    <div class="calc-line">
                        <span class="calc-operator">→ prob:</span> <span class="calc-result">${(cls.probability * 100).toFixed(1)}%</span>
                    </div>
                </div>
            </div>
        `;
    }

    section.innerHTML = html;
    return section;
}

function createPredictionSummarySection(trace) {
    const section = document.createElement('div');
    section.className = 'trace-section';

    const pred = trace.prediction;
    const probs = pred.probabilities;

    let html = `
        <h4>Prediction Summary</h4>
        <p style="font-size: 0.8em; color: #888; margin-bottom: 10px;">
            Softmax probabilities for all 10 colors.
        </p>
        <div class="prob-bars">
    `;

    for (let c = 0; c < 10; c++) {
        const prob = probs[c];
        const isPredicted = c === pred.predicted_color;
        const isExpected = c === pred.expected_color;
        const barClass = isPredicted ? 'predicted' : (isExpected ? 'expected' : '');
        const textColor = c === 0 || c === 9 ? '#fff' : '#000';

        html += `
            <div class="prob-row">
                <div class="prob-color" style="background: ${ARC_COLORS[c]}; color: ${textColor};">${c}</div>
                <div class="prob-bar-container">
                    <div class="prob-bar ${barClass}" style="width: ${prob * 100}%;"></div>
                </div>
                <span class="prob-value ${isPredicted ? 'highlight' : ''}">${(prob * 100).toFixed(1)}%</span>
            </div>
        `;
    }

    html += `</div>`;

    // Legend
    html += `
        <div style="margin-top: 12px; font-size: 0.8em; color: #888; display: flex; gap: 15px;">
            <span><span style="display: inline-block; width: 12px; height: 12px; background: #2ECC40; border-radius: 2px;"></span> Predicted</span>
            <span><span style="display: inline-block; width: 12px; height: 12px; background: #FFDC00; border-radius: 2px;"></span> Expected</span>
        </div>
    `;

    section.innerHTML = html;
    return section;
}


// =============================================================================
// Initialization
// =============================================================================

document.addEventListener('DOMContentLoaded', () => {
    loadModelInfo();
    loadKernels();
    loadPuzzles();
});
