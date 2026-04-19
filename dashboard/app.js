/**
 * app.js — TrialNet Dashboard Logic
 *
 * Handles data fetching, chart rendering, and real-time updates.
 * Uses Chart.js for all visualizations.
 */

// ── Configuration ─────────────────────────────────────────────
const API_BASE = window.location.origin;
const POLL_INTERVAL = 3000;
let currentMode = 'hybrid';
let charts = {};
let pollTimer = null;

// Chart.js global defaults
Chart.defaults.color = '#8b8ba3';
Chart.defaults.borderColor = 'rgba(99, 102, 241, 0.08)';
Chart.defaults.font.family = "'Inter', sans-serif";
Chart.defaults.font.size = 11;
Chart.defaults.plugins.legend.labels.usePointStyle = true;
Chart.defaults.plugins.legend.labels.pointStyleWidth = 10;
Chart.defaults.plugins.legend.labels.padding = 16;

// ── Color Palette ─────────────────────────────────────────────
const COLORS = {
    indigo: { solid: '#6366f1', alpha: 'rgba(99,102,241,0.15)' },
    violet: { solid: '#8b5cf6', alpha: 'rgba(139,92,246,0.15)' },
    cyan: { solid: '#06b6d4', alpha: 'rgba(6,182,212,0.15)' },
    emerald: { solid: '#10b981', alpha: 'rgba(16,185,129,0.15)' },
    amber: { solid: '#f59e0b', alpha: 'rgba(245,158,11,0.15)' },
    rose: { solid: '#f43f5e', alpha: 'rgba(244,63,94,0.15)' },
};

// ── Initialization ────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
    initModeButtons();
    initCharts();
    loadData(currentMode);
    startPolling();
});

function initModeButtons() {
    document.querySelectorAll('.mode-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            document.querySelectorAll('.mode-btn').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            currentMode = btn.dataset.mode;

            if (currentMode === 'compare') {
                showComparison();
            } else {
                hideComparison();
                loadData(currentMode);
            }
        });
    });
}

// ── Chart Initialization ──────────────────────────────────────
function initCharts() {
    // Loss chart
    charts.loss = new Chart(document.getElementById('lossChart'), {
        type: 'line',
        data: {
            labels: [],
            datasets: [
                createDataset('Train Loss', COLORS.indigo),
                createDataset('Val Loss', COLORS.rose),
            ]
        },
        options: chartOptions('Loss')
    });

    // Accuracy chart
    charts.accuracy = new Chart(document.getElementById('accuracyChart'), {
        type: 'line',
        data: {
            labels: [],
            datasets: [
                createDataset('Train Accuracy', COLORS.emerald),
                createDataset('Val Accuracy', COLORS.cyan),
            ]
        },
        options: chartOptions('Accuracy', { suggestedMin: 0, suggestedMax: 1 })
    });

    // Memory chart
    charts.memory = new Chart(document.getElementById('memoryChart'), {
        type: 'line',
        data: {
            labels: [],
            datasets: [
                createDataset('Stored Mistakes', COLORS.violet),
                { ...createDataset('Correction Rate', COLORS.emerald), yAxisID: 'y1' },
            ]
        },
        options: {
            ...chartOptions('Count'),
            scales: {
                x: { grid: { color: 'rgba(99,102,241,0.05)' }, ticks: { color: '#5a5a72' } },
                y: {
                    position: 'left',
                    grid: { color: 'rgba(99,102,241,0.05)' },
                    ticks: { color: '#5a5a72' },
                    title: { display: true, text: 'Stored Mistakes', color: '#8b8ba3' }
                },
                y1: {
                    position: 'right',
                    grid: { display: false },
                    ticks: { color: '#5a5a72' },
                    title: { display: true, text: 'Correction Rate', color: '#8b8ba3' },
                    suggestedMin: 0,
                    suggestedMax: 1,
                }
            }
        }
    });

    // Learning rate chart
    charts.lr = new Chart(document.getElementById('lrChart'), {
        type: 'line',
        data: {
            labels: [],
            datasets: [createDataset('Learning Rate', COLORS.amber)]
        },
        options: chartOptions('LR')
    });

    // Comparison chart
    charts.comparison = new Chart(document.getElementById('comparisonChart'), {
        type: 'bar',
        data: {
            labels: ['Traditional', 'Trial Only', 'Hybrid'],
            datasets: [{
                label: 'Test Accuracy',
                data: [0, 0, 0],
                backgroundColor: [COLORS.amber.alpha, COLORS.cyan.alpha, COLORS.indigo.alpha],
                borderColor: [COLORS.amber.solid, COLORS.cyan.solid, COLORS.indigo.solid],
                borderWidth: 2,
                borderRadius: 8,
                barPercentage: 0.5,
            }]
        },
        options: {
            ...chartOptions('Accuracy'),
            scales: {
                x: { grid: { display: false }, ticks: { color: '#8b8ba3', font: { weight: 600 } } },
                y: {
                    grid: { color: 'rgba(99,102,241,0.05)' },
                    suggestedMin: 0,
                    suggestedMax: 1,
                    ticks: { color: '#5a5a72' },
                }
            }
        }
    });
}

function createDataset(label, color) {
    return {
        label,
        data: [],
        borderColor: color.solid,
        backgroundColor: color.alpha,
        borderWidth: 2,
        pointRadius: 3,
        pointHoverRadius: 6,
        pointBackgroundColor: color.solid,
        pointBorderColor: 'transparent',
        fill: true,
        tension: 0.4,
    };
}

function chartOptions(yLabel, yOptions = {}) {
    return {
        responsive: true,
        maintainAspectRatio: false,
        interaction: {
            mode: 'index',
            intersect: false,
        },
        plugins: {
            legend: {
                display: true,
                position: 'top',
            },
            tooltip: {
                backgroundColor: 'rgba(10,10,15,0.9)',
                titleColor: '#f0f0f5',
                bodyColor: '#8b8ba3',
                borderColor: 'rgba(99,102,241,0.2)',
                borderWidth: 1,
                cornerRadius: 8,
                padding: 12,
            }
        },
        scales: {
            x: {
                grid: { color: 'rgba(99,102,241,0.05)' },
                ticks: { color: '#5a5a72' },
            },
            y: {
                grid: { color: 'rgba(99,102,241,0.05)' },
                ticks: { color: '#5a5a72' },
                title: { display: true, text: yLabel, color: '#8b8ba3' },
                ...yOptions,
            }
        },
        animation: {
            duration: 600,
            easing: 'easeOutQuart',
        }
    };
}

// ── Data Loading ──────────────────────────────────────────────
async function loadData(mode) {
    try {
        const res = await fetch(`${API_BASE}/api/history/${mode}`);
        if (!res.ok) {
            setStatus('No data', true);
            clearCharts();
            return;
        }
        const data = await res.json();
        updateDashboard(data);
        setStatus('Data loaded', false);
    } catch (err) {
        setStatus('Offline', true);
        console.error('Failed to load data:', err);
    }
}

async function loadComparison() {
    try {
        const res = await fetch(`${API_BASE}/api/comparison`);
        if (!res.ok) return;
        const data = await res.json();
        updateComparison(data);
    } catch (err) {
        console.error('Failed to load comparison:', err);
    }
}

// ── Dashboard Updates ─────────────────────────────────────────
function updateDashboard(data) {
    if (!data || data.error) return;

    const epochs = data.train_loss ? data.train_loss.map((_, i) => `Epoch ${i + 1}`) : [];

    // Stat cards
    if (data.val_accuracy && data.val_accuracy.length > 0) {
        const lastAcc = data.val_accuracy[data.val_accuracy.length - 1];
        document.getElementById('statAccuracy').textContent = (lastAcc * 100).toFixed(1) + '%';
    }
    if (data.val_loss && data.val_loss.length > 0) {
        const lastLoss = data.val_loss[data.val_loss.length - 1];
        document.getElementById('statLoss').textContent = lastLoss.toFixed(4);
    }
    if (data.mistake_count && data.mistake_count.length > 0) {
        document.getElementById('statMemory').textContent = data.mistake_count[data.mistake_count.length - 1];
    }
    if (data.correction_rate && data.correction_rate.length > 0) {
        const rate = data.correction_rate[data.correction_rate.length - 1];
        document.getElementById('statCorrections').textContent = (rate * 100).toFixed(1) + '%';
    }

    // Loss chart
    charts.loss.data.labels = epochs;
    charts.loss.data.datasets[0].data = data.train_loss || [];
    charts.loss.data.datasets[1].data = data.val_loss || [];
    charts.loss.update('none');

    // Accuracy chart
    charts.accuracy.data.labels = epochs;
    charts.accuracy.data.datasets[0].data = data.train_accuracy || [];
    charts.accuracy.data.datasets[1].data = data.val_accuracy || [];
    charts.accuracy.update('none');

    // Memory chart
    charts.memory.data.labels = epochs;
    charts.memory.data.datasets[0].data = data.mistake_count || [];
    charts.memory.data.datasets[1].data = data.correction_rate || [];
    charts.memory.update('none');

    // LR chart
    charts.lr.data.labels = epochs;
    charts.lr.data.datasets[0].data = data.learning_rate || [];
    charts.lr.update('none');

    // Trial details
    updateTrialPanel(data.trial_metrics);
}

function updateTrialPanel(trialMetrics) {
    if (!trialMetrics || trialMetrics.length === 0) {
        document.getElementById('explorationRate').textContent = 'N/A';
        document.getElementById('severityValue').textContent = 'N/A';
        document.getElementById('trialWeight').textContent = 'N/A';
        document.getElementById('patternsFound').textContent = 'N/A';
        return;
    }

    const latest = trialMetrics[trialMetrics.length - 1];
    if (!latest || Object.keys(latest).length === 0) return;

    // Exploration rate
    if (latest.exploration) {
        const rate = latest.exploration.success_rate || 0;
        document.getElementById('explorationRate').textContent = (rate * 100).toFixed(1) + '%';
        document.getElementById('explorationBar').style.width = (rate * 100) + '%';
    }

    // Severity
    if (latest.latest_report) {
        const severity = latest.latest_report.severity || 0;
        document.getElementById('severityValue').textContent = severity.toFixed(2);
        document.getElementById('severityBar').style.width = (severity * 100) + '%';
        document.getElementById('patternsFound').textContent = latest.latest_report.patterns || 0;
    }

    // Trial weight
    if (latest.trial_weight !== undefined) {
        document.getElementById('trialWeight').textContent = latest.trial_weight.toFixed(3);
    }

    // Patterns
    updatePatterns(latest);
}

function updatePatterns(metrics) {
    const list = document.getElementById('patternsList');
    if (!metrics || !metrics.latest_report || !metrics.latest_report.hardest_classes) {
        return;
    }

    const report = metrics.latest_report;
    let html = '';

    if (report.hardest_classes && report.hardest_classes.length > 0) {
        report.hardest_classes.forEach(cls => {
            html += `
                <div class="pattern-item">
                    <span class="pattern-type class_failure">hard class</span>
                    <span>Class ${cls} has high mistake rate</span>
                    <span class="pattern-severity" style="color: ${COLORS.amber.solid}">●</span>
                </div>`;
        });
    }

    if (metrics.memory) {
        const mem = metrics.memory;
        html += `
            <div class="pattern-item">
                <span class="pattern-type confusion">memory</span>
                <span>${mem.total_stored || 0} mistakes stored, ${mem.total_corrected || 0} corrected</span>
                <span class="pattern-severity" style="color: ${COLORS.cyan.solid}">${((mem.correction_rate || 0) * 100).toFixed(0)}%</span>
            </div>`;
    }

    if (html) {
        list.innerHTML = html;
    }
}

function updateComparison(data) {
    const modes = ['traditional', 'trial', 'hybrid'];
    const accs = modes.map(m => data[m] ? data[m].test_accuracy : 0);

    charts.comparison.data.datasets[0].data = accs;
    charts.comparison.update();

    // Find winner
    const maxIdx = accs.indexOf(Math.max(...accs));
    const winner = modes[maxIdx];
    document.getElementById('statAccuracy').textContent = (accs[maxIdx] * 100).toFixed(1) + '%';
}

// ── UI Helpers ────────────────────────────────────────────────
function showComparison() {
    document.getElementById('comparisonCard').classList.remove('hidden');
    loadComparison();
}

function hideComparison() {
    document.getElementById('comparisonCard').classList.add('hidden');
}

function clearCharts() {
    Object.values(charts).forEach(chart => {
        chart.data.datasets.forEach(ds => { ds.data = []; });
        chart.data.labels = [];
        chart.update('none');
    });
    document.getElementById('statAccuracy').textContent = '—';
    document.getElementById('statLoss').textContent = '—';
    document.getElementById('statMemory').textContent = '—';
    document.getElementById('statCorrections').textContent = '—';
}

function setStatus(text, offline = false) {
    document.getElementById('statusText').textContent = text;
    const dot = document.querySelector('.status-dot');
    if (offline) {
        dot.classList.add('offline');
    } else {
        dot.classList.remove('offline');
    }
}

// ── Polling ───────────────────────────────────────────────────
function startPolling() {
    pollTimer = setInterval(() => {
        if (currentMode === 'compare') {
            loadComparison();
        } else {
            loadData(currentMode);
        }
    }, POLL_INTERVAL);
}

function stopPolling() {
    if (pollTimer) {
        clearInterval(pollTimer);
        pollTimer = null;
    }
}
