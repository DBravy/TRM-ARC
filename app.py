#!/usr/bin/env python3
"""
TRM-ARC Training Web App

Run with: python app.py
Then open http://localhost:5000 in your browser.
"""

import os
import re
import sys
import json
import time
import signal
import subprocess
import threading
from datetime import datetime
from collections import deque
from flask import Flask, render_template_string, jsonify, request, Response

app = Flask(__name__)

# Global state for tracking training processes
training_state = {
    "trm": {
        "process": None,
        "status": "stopped",  # stopped, running, stopping
        "logs": deque(maxlen=1000),
        "start_time": None,
        "metrics": {},
        "metrics_history": []  # List of {x, loss, accuracy, ...}
    },
    "cnn": {
        "process": None,
        "status": "stopped",
        "logs": deque(maxlen=1000),
        "start_time": None,
        "metrics": {},
        "metrics_history": []  # List of {x, loss, accuracy, ...}
    }
}

lock = threading.Lock()


def stream_output(process, trainer_type):
    """Stream process output to logs."""
    global training_state

    try:
        for line in iter(process.stdout.readline, ''):
            if not line:
                break
            line = line.rstrip('\n\r')
            timestamp = datetime.now().strftime("%H:%M:%S")
            with lock:
                training_state[trainer_type]["logs"].append(f"[{timestamp}] {line}")
                # Parse metrics from output
                parse_metrics(line, trainer_type)
    except Exception as e:
        with lock:
            training_state[trainer_type]["logs"].append(f"[ERROR] Stream error: {e}")
    finally:
        with lock:
            training_state[trainer_type]["status"] = "stopped"
            training_state[trainer_type]["process"] = None


def parse_metrics(line, trainer_type):
    """Parse training metrics from output lines."""
    global training_state

    try:
        # Parse TRM metrics
        if trainer_type == "trm":
            if "Loss:" in line and "Acc:" in line:
                parts = line.split("|")
                for part in parts:
                    if "Loss:" in part:
                        val = part.split("Loss:")[1].strip().split()[0]
                        training_state[trainer_type]["metrics"]["loss"] = val
                    if "Acc:" in part:
                        val = part.split("Acc:")[1].strip().split()[0]
                        training_state[trainer_type]["metrics"]["accuracy"] = val
                    if "Exact:" in part:
                        val = part.split("Exact:")[1].strip().split()[0]
                        training_state[trainer_type]["metrics"]["exact_accuracy"] = val
            if "Step" in line and "/" in line:
                try:
                    step_part = line.split("Step")[1].split("|")[0].strip()
                    if "/" in step_part:
                        current, total = step_part.replace(",", "").split("/")
                        training_state[trainer_type]["metrics"]["step"] = current.strip()
                        training_state[trainer_type]["metrics"]["total_steps"] = total.strip().split()[0]
                        # Record to history
                        m = training_state[trainer_type]["metrics"]
                        if m.get("loss") and m.get("accuracy"):
                            history = training_state[trainer_type]["metrics_history"]
                            history.append({
                                "x": int(current.strip()),
                                "loss": float(m["loss"]),
                                "accuracy": float(m["accuracy"].rstrip('%')) / 100 if '%' in m["accuracy"] else float(m["accuracy"]),
                            })
                            # Keep last 500 points
                            if len(history) > 500:
                                training_state[trainer_type]["metrics_history"] = history[-500:]
                except:
                    pass
            if "Val Accuracy:" in line:
                val = line.split("Val Accuracy:")[1].strip().split()[0]
                training_state[trainer_type]["metrics"]["val_accuracy"] = val
            if "Val Exact Accuracy:" in line:
                val = line.split("Val Exact Accuracy:")[1].strip().split()[0]
                training_state[trainer_type]["metrics"]["val_exact_accuracy"] = val

        # Parse CNN metrics
        elif trainer_type == "cnn":
            if "Epoch" in line and "/" in line and "Training" not in line:
                try:
                    epoch_part = line.split("Epoch")[1].strip().split("/")
                    training_state[trainer_type]["metrics"]["epoch"] = epoch_part[0].strip()
                    training_state[trainer_type]["metrics"]["total_epochs"] = epoch_part[1].strip().split()[0]
                except:
                    pass
            # Parse tqdm progress bar for step-by-step tracking
            # Format: "Training: 45%|████ | 450/1000 [00:30<00:37, loss=0.1234]"
            if "Training:" in line and "loss=" in line:
                try:
                    # Extract step from "450/1000" pattern
                    step_match = re.search(r'\|\s*(\d+)/(\d+)\s*\[', line)
                    # Extract loss from "loss=0.1234" pattern
                    loss_match = re.search(r'loss=([0-9.]+)', line)
                    if step_match and loss_match:
                        current_step = int(step_match.group(1))
                        total_steps = int(step_match.group(2))
                        loss_val = float(loss_match.group(1))

                        # Calculate global step (epoch * steps_per_epoch + current_step)
                        epoch = int(training_state[trainer_type]["metrics"].get("epoch", 1))
                        global_step = (epoch - 1) * total_steps + current_step

                        training_state[trainer_type]["metrics"]["step"] = str(global_step)
                        training_state[trainer_type]["metrics"]["total_steps"] = str(int(training_state[trainer_type]["metrics"].get("total_epochs", 100)) * total_steps)
                        training_state[trainer_type]["metrics"]["train_loss"] = f"{loss_val:.4f}"

                        # Record to history (sample every 10 steps to avoid too many points)
                        if current_step % 10 == 0 or current_step == total_steps:
                            history = training_state[trainer_type]["metrics_history"]
                            history.append({
                                "x": global_step,
                                "loss": loss_val,
                                "accuracy": 0,  # Not available per-step
                            })
                            # Keep last 500 points
                            if len(history) > 500:
                                training_state[trainer_type]["metrics_history"] = history[-500:]
                except:
                    pass
            if "Train Loss:" in line:
                parts = line.split(",")
                for part in parts:
                    if "Train Loss:" in part:
                        val = part.split("Train Loss:")[1].strip().split()[0]
                        training_state[trainer_type]["metrics"]["train_loss"] = val
                    if "Pix Acc:" in part:
                        val = part.split("Pix Acc:")[1].strip().split()[0]
                        training_state[trainer_type]["metrics"]["pixel_accuracy"] = val
                    if "IoU:" in part:
                        val = part.split("IoU:")[1].strip().split()[0]
                        training_state[trainer_type]["metrics"]["iou"] = val
            if "Val   Loss:" in line:
                parts = line.split(",")
                for part in parts:
                    if "Val   Loss:" in part:
                        val = part.split("Val   Loss:")[1].strip().split()[0]
                        training_state[trainer_type]["metrics"]["val_loss"] = val
            if "New best Error IoU:" in line:
                val = line.split("New best Error IoU:")[1].strip().split("]")[0]
                training_state[trainer_type]["metrics"]["best_iou"] = val
    except Exception:
        pass  # Ignore parsing errors


def start_training(trainer_type, extra_args=None):
    """Start a training process."""
    global training_state

    with lock:
        if training_state[trainer_type]["status"] == "running":
            return False, "Training already running"

        # Clear previous logs and history
        training_state[trainer_type]["logs"].clear()
        training_state[trainer_type]["metrics"] = {}
        training_state[trainer_type]["metrics_history"] = []
        training_state[trainer_type]["status"] = "running"
        training_state[trainer_type]["start_time"] = datetime.now().isoformat()

    # Determine script to run
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if trainer_type == "trm":
        script = os.path.join(script_dir, "train.py")
        default_args = ["--no-wandb", "--visualize"]
    else:
        script = os.path.join(script_dir, "train_pixel_error_cnn.py")
        default_args = []

    cmd = [sys.executable, "-u", script] + default_args
    if extra_args:
        cmd.extend(extra_args)

    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            cwd=script_dir,
            env={**os.environ, "PYTHONUNBUFFERED": "1"}
        )

        with lock:
            training_state[trainer_type]["process"] = process
            training_state[trainer_type]["logs"].append(f"[INFO] Started {trainer_type.upper()} training")
            training_state[trainer_type]["logs"].append(f"[INFO] Command: {' '.join(cmd)}")

        # Start output streaming thread
        thread = threading.Thread(target=stream_output, args=(process, trainer_type), daemon=True)
        thread.start()

        return True, "Training started"
    except Exception as e:
        with lock:
            training_state[trainer_type]["status"] = "stopped"
        return False, str(e)


def stop_training(trainer_type):
    """Stop a training process."""
    global training_state

    with lock:
        if training_state[trainer_type]["status"] != "running":
            return False, "Training not running"

        process = training_state[trainer_type]["process"]
        training_state[trainer_type]["status"] = "stopping"
        training_state[trainer_type]["logs"].append(f"[INFO] Stopping {trainer_type.upper()} training...")

    if process:
        try:
            # Try graceful termination first
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()

            with lock:
                training_state[trainer_type]["status"] = "stopped"
                training_state[trainer_type]["process"] = None
                training_state[trainer_type]["logs"].append(f"[INFO] {trainer_type.upper()} training stopped")

            return True, "Training stopped"
        except Exception as e:
            return False, str(e)

    return False, "No process to stop"


# HTML Template
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>TRM-ARC Training Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            background: #1a1a2e;
            color: #eee;
            min-height: 100vh;
            padding: 20px;
        }
        h1 {
            text-align: center;
            margin-bottom: 30px;
            color: #00d4ff;
        }
        .container {
            max-width: 1400px;
            margin: 0 auto;
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
        }
        @media (max-width: 1000px) {
            .container {
                grid-template-columns: 1fr;
            }
        }
        .panel {
            background: #16213e;
            border-radius: 12px;
            padding: 20px;
            border: 1px solid #0f3460;
        }
        .panel-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 15px;
            padding-bottom: 15px;
            border-bottom: 1px solid #0f3460;
        }
        .panel-title {
            font-size: 1.4em;
            font-weight: 600;
        }
        .status {
            padding: 6px 12px;
            border-radius: 20px;
            font-size: 0.85em;
            font-weight: 500;
        }
        .status.stopped { background: #333; color: #888; }
        .status.running { background: #0a5; color: #fff; }
        .status.stopping { background: #a50; color: #fff; }

        .controls {
            display: flex;
            gap: 10px;
            margin-bottom: 15px;
        }
        button {
            padding: 10px 20px;
            border: none;
            border-radius: 6px;
            font-size: 1em;
            cursor: pointer;
            transition: all 0.2s;
        }
        button:disabled {
            opacity: 0.5;
            cursor: not-allowed;
        }
        .btn-start {
            background: #00d4ff;
            color: #000;
        }
        .btn-start:hover:not(:disabled) {
            background: #00a8cc;
        }
        .btn-stop {
            background: #ff4757;
            color: #fff;
        }
        .btn-stop:hover:not(:disabled) {
            background: #cc3a47;
        }
        .btn-clear {
            background: #444;
            color: #fff;
        }
        .btn-clear:hover:not(:disabled) {
            background: #555;
        }

        .metrics {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
            gap: 10px;
            margin-bottom: 15px;
        }
        .metric {
            background: #0f3460;
            padding: 12px;
            border-radius: 8px;
            text-align: center;
        }
        .metric-label {
            font-size: 0.75em;
            color: #888;
            text-transform: uppercase;
            margin-bottom: 4px;
        }
        .metric-value {
            font-size: 1.2em;
            font-weight: 600;
            color: #00d4ff;
        }

        .log-container {
            background: #0d1b2a;
            border-radius: 8px;
            height: 400px;
            overflow-y: auto;
            font-family: 'Monaco', 'Menlo', 'Courier New', monospace;
            font-size: 0.85em;
            padding: 10px;
        }
        .log-line {
            padding: 2px 0;
            white-space: pre-wrap;
            word-break: break-all;
        }
        .log-line:hover {
            background: #1a2a3a;
        }

        .options {
            margin-bottom: 15px;
        }
        .options label {
            display: block;
            margin-bottom: 8px;
            font-size: 0.9em;
            color: #aaa;
        }
        .options input, .options select {
            width: 100%;
            padding: 8px 12px;
            border: 1px solid #0f3460;
            border-radius: 6px;
            background: #0d1b2a;
            color: #fff;
            font-size: 0.9em;
        }
        .options-row {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 10px;
        }

        .graph-panel {
            max-width: 1400px;
            margin: 20px auto 0;
            background: #16213e;
            border-radius: 12px;
            padding: 20px;
            border: 1px solid #0f3460;
        }
        .graph-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 15px;
            flex-wrap: wrap;
            gap: 10px;
        }
        .graph-title {
            font-size: 1.2em;
            font-weight: 600;
        }
        .graph-controls {
            display: flex;
            gap: 10px;
            align-items: center;
        }
        .graph-controls select {
            padding: 8px 12px;
            border: 1px solid #0f3460;
            border-radius: 6px;
            background: #0d1b2a;
            color: #fff;
            font-size: 0.9em;
        }
        .graph-container {
            background: #0d1b2a;
            border-radius: 8px;
            padding: 15px;
            height: 300px;
        }
    </style>
</head>
<body>
    <h1>TRM-ARC Training Dashboard</h1>

    <div class="container">
        <!-- TRM Training Panel -->
        <div class="panel">
            <div class="panel-header">
                <span class="panel-title">TRM Model Training</span>
                <span id="trm-status" class="status stopped">Stopped</span>
            </div>

            <div class="options">
                <div class="options-row">
                    <div>
                        <label>Dataset</label>
                        <select id="trm-dataset">
                            <option value="arc-agi-1">ARC-AGI-1</option>
                            <option value="arc-agi-2">ARC-AGI-2</option>
                        </select>
                    </div>
                    <div>
                        <label>Epochs</label>
                        <input type="number" id="trm-epochs" value="10000" min="1">
                    </div>
                </div>
                <div class="options-row" style="margin-top: 10px;">
                    <div>
                        <label>Batch Size</label>
                        <input type="number" id="trm-batch" value="768" min="1">
                    </div>
                    <div>
                        <label>Eval Interval</label>
                        <input type="number" id="trm-eval-interval" value="1000" min="1">
                    </div>
                </div>
            </div>

            <div class="controls">
                <button id="trm-start" class="btn-start" onclick="startTraining('trm')">Start Training</button>
                <button id="trm-stop" class="btn-stop" onclick="stopTraining('trm')" disabled>Stop</button>
                <button class="btn-clear" onclick="clearLogs('trm')">Clear Logs</button>
            </div>

            <div id="trm-metrics" class="metrics">
                <div class="metric">
                    <div class="metric-label">Step</div>
                    <div class="metric-value" id="trm-metric-step">-</div>
                </div>
                <div class="metric">
                    <div class="metric-label">Loss</div>
                    <div class="metric-value" id="trm-metric-loss">-</div>
                </div>
                <div class="metric">
                    <div class="metric-label">Accuracy</div>
                    <div class="metric-value" id="trm-metric-accuracy">-</div>
                </div>
                <div class="metric">
                    <div class="metric-label">Val Exact</div>
                    <div class="metric-value" id="trm-metric-val-exact">-</div>
                </div>
            </div>

            <div id="trm-logs" class="log-container"></div>
        </div>

        <!-- CNN Training Panel -->
        <div class="panel">
            <div class="panel-header">
                <span class="panel-title">Pixel Error CNN Training</span>
                <span id="cnn-status" class="status stopped">Stopped</span>
            </div>

            <div class="options">
                <div class="options-row">
                    <div>
                        <label>Dataset</label>
                        <select id="cnn-dataset">
                            <option value="arc-agi-1">ARC-AGI-1</option>
                            <option value="arc-agi-2">ARC-AGI-2</option>
                        </select>
                    </div>
                    <div>
                        <label>Epochs</label>
                        <input type="number" id="cnn-epochs" value="100" min="1">
                    </div>
                </div>
                <div class="options-row" style="margin-top: 10px;">
                    <div>
                        <label>Batch Size</label>
                        <input type="number" id="cnn-batch" value="32" min="1">
                    </div>
                    <div>
                        <label>Hidden Dim</label>
                        <input type="number" id="cnn-hidden" value="64" min="8">
                    </div>
                </div>
                <div class="options-row" style="margin-top: 10px;">
                    <div>
                        <label>Num Negatives</label>
                        <input type="number" id="cnn-negatives" value="8" min="1" title="Total negatives per example (auto-distributed across types)">
                    </div>
                    <div></div>
                </div>
            </div>

            <div class="controls">
                <button id="cnn-start" class="btn-start" onclick="startTraining('cnn')">Start Training</button>
                <button id="cnn-stop" class="btn-stop" onclick="stopTraining('cnn')" disabled>Stop</button>
                <button class="btn-clear" onclick="clearLogs('cnn')">Clear Logs</button>
            </div>

            <div id="cnn-metrics" class="metrics">
                <div class="metric">
                    <div class="metric-label">Step</div>
                    <div class="metric-value" id="cnn-metric-step">-</div>
                </div>
                <div class="metric">
                    <div class="metric-label">Train Loss</div>
                    <div class="metric-value" id="cnn-metric-loss">-</div>
                </div>
                <div class="metric">
                    <div class="metric-label">Pixel Acc</div>
                    <div class="metric-value" id="cnn-metric-acc">-</div>
                </div>
                <div class="metric">
                    <div class="metric-label">Best IoU</div>
                    <div class="metric-value" id="cnn-metric-iou">-</div>
                </div>
            </div>

            <div id="cnn-logs" class="log-container"></div>
        </div>
    </div>

    <!-- Graph Panel -->
    <div class="graph-panel">
        <div class="graph-header">
            <span class="graph-title">Training Metrics</span>
            <div class="graph-controls">
                <select id="graph-source">
                    <option value="trm">TRM Model</option>
                    <option value="cnn">CNN Model</option>
                </select>
                <select id="graph-metric">
                    <option value="loss">Loss</option>
                    <option value="accuracy">Accuracy</option>
                </select>
            </div>
        </div>
        <div class="graph-container">
            <canvas id="metrics-chart"></canvas>
        </div>
    </div>

    <script>
        let lastLogIndex = { trm: 0, cnn: 0 };
        let metricsHistory = { trm: [], cnn: [] };
        let chart = null;

        // Initialize chart
        function initChart() {
            const ctx = document.getElementById('metrics-chart').getContext('2d');
            chart = new Chart(ctx, {
                type: 'line',
                data: {
                    datasets: [{
                        label: 'Loss',
                        data: [],
                        borderColor: '#00d4ff',
                        backgroundColor: 'rgba(0, 212, 255, 0.1)',
                        tension: 0.1,
                        fill: true,
                        pointRadius: 0,
                        borderWidth: 2
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    animation: { duration: 0 },
                    scales: {
                        x: {
                            type: 'linear',
                            title: { display: true, text: 'Step/Epoch', color: '#888' },
                            ticks: { color: '#888' },
                            grid: { color: '#333' }
                        },
                        y: {
                            title: { display: true, text: 'Value', color: '#888' },
                            ticks: { color: '#888' },
                            grid: { color: '#333' }
                        }
                    },
                    plugins: {
                        legend: { labels: { color: '#eee' } }
                    }
                }
            });
        }

        function updateChart() {
            if (!chart) return;

            const source = document.getElementById('graph-source').value;
            const metric = document.getElementById('graph-metric').value;
            const history = metricsHistory[source] || [];

            const data = history.map(h => ({
                x: h.x,
                y: metric === 'loss' ? h.loss : h.accuracy
            }));

            chart.data.datasets[0].data = data;
            chart.data.datasets[0].label = metric === 'loss' ? 'Loss' : 'Accuracy';
            chart.data.datasets[0].borderColor = metric === 'loss' ? '#ff4757' : '#00d4ff';
            chart.data.datasets[0].backgroundColor = metric === 'loss' ? 'rgba(255, 71, 87, 0.1)' : 'rgba(0, 212, 255, 0.1)';
            chart.options.scales.x.title.text = 'Step';
            chart.update('none');
        }

        function updateStatus() {
            fetch('/api/status')
                .then(response => response.json())
                .then(data => {
                    ['trm', 'cnn'].forEach(type => {
                        const state = data[type];

                        // Update status badge
                        const statusEl = document.getElementById(`${type}-status`);
                        statusEl.textContent = state.status.charAt(0).toUpperCase() + state.status.slice(1);
                        statusEl.className = `status ${state.status}`;

                        // Update buttons
                        const isRunning = state.status === 'running';
                        document.getElementById(`${type}-start`).disabled = isRunning;
                        document.getElementById(`${type}-stop`).disabled = !isRunning;

                        // Update logs
                        const logsEl = document.getElementById(`${type}-logs`);
                        const newLogs = state.logs.slice(lastLogIndex[type]);
                        if (newLogs.length > 0) {
                            newLogs.forEach(log => {
                                const line = document.createElement('div');
                                line.className = 'log-line';
                                line.textContent = log;
                                logsEl.appendChild(line);
                            });
                            lastLogIndex[type] = state.logs.length;
                            logsEl.scrollTop = logsEl.scrollHeight;
                        }

                        // Update metrics history
                        metricsHistory[type] = state.metrics_history || [];

                        // Update metrics
                        if (type === 'trm') {
                            const m = state.metrics;
                            document.getElementById('trm-metric-step').textContent =
                                m.step ? `${m.step}/${m.total_steps || '?'}` : '-';
                            document.getElementById('trm-metric-loss').textContent = m.loss || '-';
                            document.getElementById('trm-metric-accuracy').textContent = m.accuracy || '-';
                            document.getElementById('trm-metric-val-exact').textContent = m.val_exact_accuracy || '-';
                        } else {
                            const m = state.metrics;
                            document.getElementById('cnn-metric-step').textContent =
                                m.step ? `${m.step}/${m.total_steps || '?'}` : (m.epoch ? `Epoch ${m.epoch}/${m.total_epochs || '?'}` : '-');
                            document.getElementById('cnn-metric-loss').textContent = m.train_loss || '-';
                            document.getElementById('cnn-metric-acc').textContent = m.pixel_accuracy || '-';
                            document.getElementById('cnn-metric-iou').textContent = m.best_iou || '-';
                        }
                    });

                    // Update chart
                    updateChart();
                })
                .catch(err => console.error('Status update error:', err));
        }

        function startTraining(type) {
            let args = [];

            if (type === 'trm') {
                args = [
                    '--dataset', document.getElementById('trm-dataset').value,
                    '--epochs', document.getElementById('trm-epochs').value,
                    '--batch-size', document.getElementById('trm-batch').value,
                    '--eval-interval', document.getElementById('trm-eval-interval').value
                ];
            } else {
                args = [
                    '--dataset', document.getElementById('cnn-dataset').value,
                    '--epochs', document.getElementById('cnn-epochs').value,
                    '--batch-size', document.getElementById('cnn-batch').value,
                    '--hidden-dim', document.getElementById('cnn-hidden').value,
                    '--num-negatives', document.getElementById('cnn-negatives').value
                ];
            }

            fetch(`/api/start/${type}`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ args: args })
            })
            .then(response => response.json())
            .then(data => {
                if (!data.success) {
                    alert('Failed to start: ' + data.message);
                }
            })
            .catch(err => alert('Error: ' + err));
        }

        function stopTraining(type) {
            fetch(`/api/stop/${type}`, { method: 'POST' })
                .then(response => response.json())
                .then(data => {
                    if (!data.success) {
                        alert('Failed to stop: ' + data.message);
                    }
                })
                .catch(err => alert('Error: ' + err));
        }

        function clearLogs(type) {
            document.getElementById(`${type}-logs`).innerHTML = '';
            lastLogIndex[type] = 0;
            metricsHistory[type] = [];
            fetch(`/api/clear/${type}`, { method: 'POST' });
            updateChart();
        }

        // Event listeners for chart controls
        document.getElementById('graph-source').addEventListener('change', updateChart);
        document.getElementById('graph-metric').addEventListener('change', updateChart);

        // Initialize chart and start polling
        initChart();
        setInterval(updateStatus, 1000);
        updateStatus();
    </script>
</body>
</html>
"""


@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)


@app.route('/api/status')
def get_status():
    with lock:
        return jsonify({
            "trm": {
                "status": training_state["trm"]["status"],
                "logs": list(training_state["trm"]["logs"]),
                "start_time": training_state["trm"]["start_time"],
                "metrics": training_state["trm"]["metrics"],
                "metrics_history": training_state["trm"]["metrics_history"]
            },
            "cnn": {
                "status": training_state["cnn"]["status"],
                "logs": list(training_state["cnn"]["logs"]),
                "start_time": training_state["cnn"]["start_time"],
                "metrics": training_state["cnn"]["metrics"],
                "metrics_history": training_state["cnn"]["metrics_history"]
            }
        })


@app.route('/api/start/<trainer_type>', methods=['POST'])
def api_start(trainer_type):
    if trainer_type not in ['trm', 'cnn']:
        return jsonify({"success": False, "message": "Invalid trainer type"})

    data = request.get_json() or {}
    extra_args = data.get('args', [])

    success, message = start_training(trainer_type, extra_args)
    return jsonify({"success": success, "message": message})


@app.route('/api/stop/<trainer_type>', methods=['POST'])
def api_stop(trainer_type):
    if trainer_type not in ['trm', 'cnn']:
        return jsonify({"success": False, "message": "Invalid trainer type"})

    success, message = stop_training(trainer_type)
    return jsonify({"success": success, "message": message})


@app.route('/api/clear/<trainer_type>', methods=['POST'])
def api_clear(trainer_type):
    if trainer_type not in ['trm', 'cnn']:
        return jsonify({"success": False, "message": "Invalid trainer type"})

    with lock:
        training_state[trainer_type]["logs"].clear()
        training_state[trainer_type]["metrics"] = {}
        training_state[trainer_type]["metrics_history"] = []

    return jsonify({"success": True})


if __name__ == '__main__':
    print("\n" + "="*50)
    print("TRM-ARC Training Dashboard")
    print("="*50)
    print("\nOpen http://localhost:5000 in your browser")
    print("Press Ctrl+C to stop the server\n")

    app.run(host='0.0.0.0', port=5004, debug=False, threaded=True)
