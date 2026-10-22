/**
 * VN-Index AI Stock Predictor - Frontend Client Logic
 */

let currentModel = 'DeepStackedLSTM';
let stockChartInstance = null;
let historicalData = [];

document.addEventListener('DOMContentLoaded', () => {
    initApp();
});

function initApp() {
    setupModelSelectors();
    setupButtons();
    setupCustomForm();
    setupMobileNav();

    // Load initial data
    loadHistoryAndRenderChart(30);
    fetchPrediction(currentModel);
    loadComparisonTable();
}

// -------------------------------------------------------------
// 1. Prediction Handler
// -------------------------------------------------------------
async function fetchPrediction(modelName) {
    const badge = document.getElementById('prediction-direction-badge');
    const predText = document.getElementById('pred-direction-text');
    const predIcon = document.getElementById('pred-icon');
    const actionBadge = document.getElementById('action-badge');

    // Set loading state
    badge.className = 'prediction-badge';
    predIcon.className = 'fa-solid fa-spinner fa-spin me-2';
    predText.textContent = 'Đang phân tích...';

    try {
        const res = await fetch(`/api/predict_latest?model=${modelName}`);
        const data = await res.json();

        if (data.status === 'success') {
            updatePredictionUI(data);
        } else {
            predText.textContent = 'Lỗi dự đoán';
            console.error(data.message);
        }
    } catch (err) {
        console.error('Error fetching prediction:', err);
        predText.textContent = 'Lỗi kết nối';
    }
}

function updatePredictionUI(data) {
    const badge = document.getElementById('prediction-direction-badge');
    const predText = document.getElementById('pred-direction-text');
    const predIcon = document.getElementById('pred-icon');
    const actionBadge = document.getElementById('action-badge');

    const isUp = data.direction === 'UP';

    // Update Badge
    badge.className = `prediction-badge ${isUp ? 'badge-up' : 'badge-down'}`;
    predIcon.className = isUp ? 'fa-solid fa-arrow-trend-up me-2' : 'fa-solid fa-arrow-trend-down me-2';
    predText.textContent = isUp ? 'XU HƯỚNG TĂNG' : 'XU HƯỚNG GIẢM';

    // Action Recommendation
    actionBadge.textContent = data.recommendation;
    actionBadge.style.backgroundColor = data.recommendation_color;
    actionBadge.style.color = '#ffffff';

    // Probability Bars
    document.getElementById('prob-up-text').textContent = `${data.probability_up}%`;
    document.getElementById('prob-down-text').textContent = `${data.probability_down}%`;
    document.getElementById('prob-bar-up').style.width = `${data.probability_up}%`;
    document.getElementById('prob-bar-down').style.width = `${data.probability_down}%`;
    document.getElementById('confidence-text').textContent = `${data.confidence}%`;

    // Ticker & Header Info
    document.getElementById('header-vnindex').textContent = `${data.latest_close.toLocaleString()} đ`;
    document.getElementById('ticker-close').textContent = `${data.latest_close.toLocaleString()}`;
    document.getElementById('ticker-date').textContent = data.latest_date;
    document.getElementById('card-date-badge').textContent = data.latest_date;

    // Technical Indicators
    const ind = data.indicators;
    if (ind) {
        document.getElementById('ticker-rsi').textContent = ind.rsi_14;
        document.getElementById('ticker-macd').textContent = ind.macd;
        document.getElementById('ticker-vol').textContent = `${ind.volatility}%`;

        document.getElementById('val-rsi').textContent = ind.rsi_14;
        const statusRsi = document.getElementById('status-rsi');
        if (ind.rsi_14 > 70) {
            statusRsi.textContent = 'Quá mua (Cảnh báo)';
            statusRsi.className = 'metric-status text-danger';
        } else if (ind.rsi_14 < 30) {
            statusRsi.textContent = 'Quá bán (Cơ hội)';
            statusRsi.className = 'metric-status text-success';
        } else {
            statusRsi.textContent = 'Vùng trung lập';
            statusRsi.className = 'metric-status text-warning';
        }

        document.getElementById('val-macd').textContent = `${ind.macd} / ${ind.macd_signal}`;
        const statusMacd = document.getElementById('status-macd');
        if (ind.macd > ind.macd_signal) {
            statusMacd.textContent = 'Bullish Crossover 🟢';
            statusMacd.className = 'metric-status text-success';
        } else {
            statusMacd.textContent = 'Bearish Crossover 🔴';
            statusMacd.className = 'metric-status text-danger';
        }

        document.getElementById('val-bb').textContent = `${ind.bb_pct}%`;
        const statusBb = document.getElementById('status-bb');
        if (ind.bb_pct > 80) {
            statusBb.textContent = 'Tiệm cận dải trên';
            statusBb.className = 'metric-status text-warning';
        } else if (ind.bb_pct < 20) {
            statusBb.textContent = 'Tiệm cận dải dưới';
            statusBb.className = 'metric-status text-info';
        } else {
            statusBb.textContent = 'Giữa dải BB';
            statusBb.className = 'metric-status text-muted';
        }

        const statusEma = document.getElementById('status-ema');
        if (ind.ema_5 > ind.ema_20) {
            statusEma.textContent = 'Xu hướng ngắn hạn TĂNG';
            statusEma.className = 'metric-status text-success';
        } else {
            statusEma.textContent = 'Xu hướng ngắn hạn GIẢM';
            statusEma.className = 'metric-status text-danger';
        }

        // Summary counts
        let buyCnt = (isUp ? 1 : 0) + (ind.macd > ind.macd_signal ? 1 : 0) + (ind.rsi_14 < 45 ? 1 : 0) + (ind.ema_5 > ind.ema_20 ? 1 : 0);
        let sellCnt = (!isUp ? 1 : 0) + (ind.macd < ind.macd_signal ? 1 : 0) + (ind.rsi_14 > 65 ? 1 : 0) + (ind.ema_5 < ind.ema_20 ? 1 : 0);
        let neutralCnt = 4 - (buyCnt > 4 ? 4 : buyCnt) - (sellCnt > 4 ? 4 : sellCnt);
        if (neutralCnt < 0) neutralCnt = 1;

        document.getElementById('cnt-buy').textContent = buyCnt;
        document.getElementById('cnt-neutral').textContent = neutralCnt;
        document.getElementById('cnt-sell').textContent = sellCnt;

        const conBadge = document.getElementById('consensus-badge');
        if (buyCnt > sellCnt) {
            conBadge.textContent = 'TÍN HIỆU TÍCH CỰC (BUY)';
            conBadge.className = 'badge bg-success';
        } else if (sellCnt > buyCnt) {
            conBadge.textContent = 'TÍN HIỆU TIÊU CỰC (SELL)';
            conBadge.className = 'badge bg-danger';
        } else {
            conBadge.textContent = 'TÍN HIỆU GIẰNG CO (NEUTRAL)';
            conBadge.className = 'badge bg-warning text-dark';
        }
    }
}

// -------------------------------------------------------------
// 2. Interactive Price Chart
// -------------------------------------------------------------
async function loadHistoryAndRenderChart(days = 30) {
    try {
        const res = await fetch('/api/history');
        const json = await res.json();
        if (json.status === 'success') {
            historicalData = json.data;
            renderChart(days);
        }
    } catch (e) {
        console.error('Error loading history:', e);
    }
}

function renderChart(days = 30) {
    const dataSlice = historicalData.slice(-days);
    const labels = dataSlice.map(d => d.date);
    const closePrices = dataSlice.map(d => d.close);
    const volumes = dataSlice.map(d => d.Volume / 1e6); // Triệu CP

    const ctx = document.getElementById('stockChart').getContext('2d');

    if (stockChartInstance) {
        stockChartInstance.destroy();
    }

    stockChartInstance = new Chart(ctx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [
                {
                    label: 'VN-Index Đóng cửa',
                    data: closePrices,
                    borderColor: '#38bdf8',
                    backgroundColor: 'rgba(56, 189, 248, 0.12)',
                    borderWidth: 2.5,
                    fill: true,
                    tension: 0.25,
                    yAxisID: 'y'
                },
                {
                    label: 'Khối lượng (Tr. CP)',
                    data: volumes,
                    type: 'bar',
                    backgroundColor: 'rgba(148, 163, 184, 0.25)',
                    yAxisID: 'y1'
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: {
                mode: 'index',
                intersect: false,
            },
            plugins: {
                legend: {
                    labels: { color: '#94a3b8', font: { family: 'Plus Jakarta Sans' } }
                },
                tooltip: {
                    backgroundColor: '#1e293b',
                    titleColor: '#f8fafc',
                    bodyColor: '#cbd5e1',
                    borderColor: '#334155',
                    borderWidth: 1
                }
            },
            scales: {
                x: {
                    grid: { color: 'rgba(255, 255, 255, 0.05)' },
                    ticks: { color: '#64748b', maxTicksLimit: 10 }
                },
                y: {
                    type: 'linear',
                    display: true,
                    position: 'left',
                    grid: { color: 'rgba(255, 255, 255, 0.05)' },
                    ticks: { color: '#38bdf8' }
                },
                y1: {
                    type: 'linear',
                    display: true,
                    position: 'right',
                    grid: { drawOnChartArea: false },
                    ticks: { color: '#64748b' }
                }
            }
        }
    });
}

// -------------------------------------------------------------
// 3. Comparison Table Handler
// -------------------------------------------------------------
async function loadComparisonTable() {
    try {
        const res = await fetch('/api/metadata');
        const meta = await res.json();

        const tbody = document.querySelector('#comparison-table tbody');
        if (!meta.eval_results) return;

        const layerCountMap = {
            'VanillaRNN': '2 Layers',
            'LSTM': '2 Layers',
            'GRU': '2 Layers',
            'BiLSTM': '2 Layers',
            'DeepStackedLSTM': '7 Layers (>5L)'
        };

        let html = '';
        for (const [name, stats] of Object.entries(meta.eval_results)) {
            const isDeep = name === 'DeepStackedLSTM';
            html += `
                <tr class="${isDeep ? 'table-active border-primary' : ''}">
                    <td class="text-start ps-3 fw-bold">
                        ${isDeep ? '<i class="fa-solid fa-star text-warning me-1"></i>' : ''}
                        ${name}
                        ${isDeep ? '<span class="badge bg-primary ms-1">Mô hình chính</span>' : ''}
                    </td>
                    <td><span class="badge bg-dark-subtle border border-secondary">${layerCountMap[name] || '2L'}</span></td>
                    <td class="text-info fw-bold">${(stats.accuracy * 100).toFixed(2)}%</td>
                    <td class="text-success fw-bold">${stats.auc.toFixed(4)}</td>
                    <td class="text-warning fw-bold">${stats.f1.toFixed(4)}</td>
                    <td class="text-muted">${stats.train_time.toFixed(1)}s</td>
                    <td id="table-pred-${name}"><span class="badge bg-secondary">Chưa chạy</span></td>
                </tr>
            `;
        }
        tbody.innerHTML = html;
    } catch (e) {
        console.error('Error loading comparison table:', e);
    }
}

// -------------------------------------------------------------
// 4. Compare All Models Concurrently
// -------------------------------------------------------------
async function runAllModels() {
    const btn = document.getElementById('btn-run-all-models');
    btn.innerHTML = '<i class="fa-solid fa-spinner fa-spin me-1"></i> Đang chạy...';
    btn.disabled = true;

    try {
        const res = await fetch('/api/predict_compare_all');
        const data = await res.json();

        if (data.status === 'success') {
            data.comparison.forEach(item => {
                const td = document.getElementById(`table-pred-${item.model}`);
                if (td) {
                    const isUp = item.direction === 'UP';
                    td.innerHTML = `
                        <span class="badge ${isUp ? 'bg-success' : 'bg-danger'}">
                            <i class="fa-solid fa-arrow-trend-${isUp ? 'up' : 'down'} me-1"></i>
                            ${item.pred} (${item.prob_up}%)
                        </span>
                    `;
                }
            });
        }
    } catch (e) {
        console.error('Error comparing all models:', e);
    } finally {
        btn.innerHTML = '<i class="fa-solid fa-bolt me-1"></i> Dự đoán đồng thời cả 5 model';
        btn.disabled = false;
    }
}

// -------------------------------------------------------------
// 5. Custom Simulation Form
// -------------------------------------------------------------
function setupCustomForm() {
    const form = document.getElementById('form-custom-predict');
    form.addEventListener('submit', async (e) => {
        e.preventDefault();

        const payload = {
            model: currentModel,
            open: parseFloat(document.getElementById('inp-open').value),
            high: parseFloat(document.getElementById('inp-high').value),
            low: parseFloat(document.getElementById('inp-low').value),
            close: parseFloat(document.getElementById('inp-close').value),
            volume: parseFloat(document.getElementById('inp-vol').value)
        };

        const resBox = document.getElementById('custom-result-box');
        const predText = document.getElementById('custom-pred-text');
        const probText = document.getElementById('custom-prob-text');

        resBox.classList.remove('d-none');
        predText.textContent = 'Đang tính toán kịch bản...';

        try {
            const res = await fetch('/api/predict_custom', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });
            const data = await res.json();

            if (data.status === 'success') {
                const isUp = data.prediction === 'TĂNG';
                predText.innerHTML = `<span class="${isUp ? 'text-success' : 'text-danger'}">
                    <i class="fa-solid fa-arrow-trend-${isUp ? 'up' : 'down'} me-1"></i>
                    Dự đoán ngày kế tiếp: ${data.prediction}
                </span>`;
                probText.textContent = `P(Tăng) = ${data.probability_up}% | P(Giảm) = ${data.probability_down}%`;
            }
        } catch (err) {
            predText.textContent = 'Lỗi tính toán kịch bản';
        }
    });
}

// -------------------------------------------------------------
// 6. UI Event Listeners
// -------------------------------------------------------------
function setupModelSelectors() {
    document.querySelectorAll('.model-select').forEach(item => {
        item.addEventListener('click', (e) => {
            e.preventDefault();
            document.querySelectorAll('.model-select').forEach(el => el.classList.remove('active'));
            item.classList.add('active');

            currentModel = item.getAttribute('data-model');
            document.getElementById('current-model-label').textContent = item.textContent.trim();
            document.getElementById('header-model-name').textContent = item.textContent.trim();
            document.getElementById('badge-model-tag').textContent = currentModel;

            fetchPrediction(currentModel);
        });
    });
}

function setupButtons() {
    document.getElementById('btn-refresh').addEventListener('click', () => {
        fetchPrediction(currentModel);
        loadHistoryAndRenderChart(30);
    });

    document.getElementById('btn-predict-now').addEventListener('click', () => {
        fetchPrediction(currentModel);
    });

    document.getElementById('btn-compare-all').addEventListener('click', () => {
        document.getElementById('compare-section').scrollIntoView({ behavior: 'smooth' });
        runAllModels();
    });

    document.getElementById('btn-run-all-models').addEventListener('click', runAllModels);

    document.querySelectorAll('.chart-range').forEach(btn => {
        btn.addEventListener('click', () => {
            document.querySelectorAll('.chart-range').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            renderChart(parseInt(btn.getAttribute('data-days')));
        });
    });
}

function setupMobileNav() {
    const navItems = document.querySelectorAll('.mobile-nav-item');
    navItems.forEach(item => {
        item.addEventListener('click', () => {
            navItems.forEach(i => i.classList.remove('active'));
            item.classList.add('active');
        });
    });
}
