/**
 * LLM Admin Dashboard JavaScript
 * Handles llama.cpp server management, monitoring, and configuration
 */

// ============================================================================
// CONSTANTS
// ============================================================================

const SERVERS = {
    sql: { port: 8080, name: 'SQL Model Server', serviceName: 'LlamaCppSql' },
    general: { port: 8081, name: 'General Model Server', serviceName: 'LlamaCppGeneral' },
    code: { port: 8082, name: 'Code Model Server', serviceName: 'LlamaCppCode' }
};

// Refresh interval loaded from server config (default 60 seconds)
let REFRESH_INTERVAL_SECONDS = 60;

// Chart configuration
const CHART_MAX_POINTS = 60; // Keep 60 data points for charts

// ============================================================================
// STATE
// ============================================================================

let refreshTimer = null;
let refreshCountdown = 60;
let tpsChart = null;
let requestsChart = null;
let miniCharts = {
    sql: { tps: null, requests: null },
    general: { tps: null, requests: null },
    code: { tps: null, requests: null }
};
let metricsHistory = {
    sql: { tps: [], requests: [], timestamps: [] },
    general: { tps: [], requests: [], timestamps: [] },
    code: { tps: [], requests: [], timestamps: [] }
};
let availableModels = [];
let currentConfig = {};
let runningModels = {
    sql: null,
    general: null,
    code: null
};

// ============================================================================
// INITIALIZATION
// ============================================================================

document.addEventListener('DOMContentLoaded', async () => {
    console.log('LLM Admin Dashboard initializing...');

    // Load client configuration from server first
    await loadClientConfig();

    // Initialize sidebar
    if (typeof initSidebar === 'function') {
        await initSidebar();
    }

    // Initialize charts
    initializeCharts();
    initializeMiniCharts();

    // Load initial data
    await Promise.all([
        refreshAllData(),
        loadAvailableModels(),
        loadConfiguration()
    ]);

    // Start auto-refresh
    startAutoRefresh();
});

/**
 * Load client configuration from server
 */
async function loadClientConfig() {
    try {
        const response = await fetch('/api/client-config');
        if (response.ok) {
            const config = await response.json();
            REFRESH_INTERVAL_SECONDS = config.adminRefreshInterval || 60;
            refreshCountdown = REFRESH_INTERVAL_SECONDS;
            console.log(`Loaded refresh interval: ${REFRESH_INTERVAL_SECONDS}s`);
        }
    } catch (error) {
        console.warn('Failed to load client config, using defaults:', error);
    }
}

// ============================================================================
// TAB NAVIGATION
// ============================================================================

function switchTab(tabName) {
    // Update tab buttons
    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.classList.remove('active');
        if (btn.textContent.toLowerCase().includes(tabName.toLowerCase()) ||
            btn.onclick.toString().includes(`'${tabName}'`)) {
            btn.classList.add('active');
        }
    });

    // Update tab content
    document.querySelectorAll('.tab-content').forEach(content => {
        content.classList.remove('active');
    });

    const targetTab = document.getElementById(`tab-${tabName}`);
    if (targetTab) {
        targetTab.classList.add('active');
    }

    // Update button states
    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.classList.remove('active');
    });
    event.target.classList.add('active');
}

// ============================================================================
// AUTO-REFRESH
// ============================================================================

function startAutoRefresh() {
    if (refreshTimer) {
        clearInterval(refreshTimer);
    }

    refreshCountdown = REFRESH_INTERVAL_SECONDS;
    updateRefreshCountdown();

    refreshTimer = setInterval(() => {
        refreshCountdown--;
        updateRefreshCountdown();

        if (refreshCountdown <= 0) {
            refreshAllData();
            refreshCountdown = REFRESH_INTERVAL_SECONDS;
        }
    }, 1000);
}

function updateRefreshCountdown() {
    const el = document.getElementById('refreshCountdown');
    if (el) {
        el.textContent = `${refreshCountdown}s`;
    }
}

// ============================================================================
// DATA REFRESH
// ============================================================================

async function refreshAllData() {
    const indicator = document.getElementById('refreshIndicator');
    const svg = indicator?.querySelector('svg');

    if (svg) {
        svg.classList.add('spin');
    }

    try {
        // Use backend API to avoid CORS issues with direct llama.cpp server requests
        const response = await fetch('/api/admin/llm/all-models', {
            signal: AbortSignal.timeout(5000)
        });

        if (response.ok) {
            const data = await response.json();
            if (data.success && data.models) {
                // Update each server's status from the API response
                for (const serverInfo of data.models) {
                    const serverType = serverInfo.server.toLowerCase();
                    updateServerFromApiData(serverType, serverInfo);
                }
            }
        } else {
            // API failed, mark all as offline
            ['sql', 'general', 'code'].forEach(serverType => {
                const statusEl = document.getElementById(`${serverType}Status`);
                updateServerStatusUI(statusEl, 'offline', 'API Error');
                resetServerMetrics(serverType);
            });
        }

        updateCharts();
    } catch (error) {
        console.error('Error refreshing data:', error);
        // On error, mark all as offline
        ['sql', 'general', 'code'].forEach(serverType => {
            const statusEl = document.getElementById(`${serverType}Status`);
            updateServerStatusUI(statusEl, 'offline', 'Offline');
            resetServerMetrics(serverType);
        });
    } finally {
        if (svg) {
            svg.classList.remove('spin');
        }
    }
}

/**
 * Update server UI from API response data
 */
function updateServerFromApiData(serverType, serverInfo) {
    const statusEl = document.getElementById(`${serverType}Status`);
    const modelNameEl = document.getElementById(`${serverType}ModelName`);
    const modelQuantEl = document.getElementById(`${serverType}ModelQuant`);
    const tpsEl = document.getElementById(`${serverType}TPS`);
    const contextEl = document.getElementById(`${serverType}Context`);
    const requestsEl = document.getElementById(`${serverType}Requests`);
    const uptimeEl = document.getElementById(`${serverType}Uptime`);

    if (serverInfo.status === 'running') {
        updateServerStatusUI(statusEl, 'online', 'Online');

        // Update advanced tab status too
        const advStatusEl = document.getElementById(`adv${serverType.charAt(0).toUpperCase() + serverType.slice(1)}Status`);
        if (advStatusEl) {
            updateServerStatusUI(advStatusEl, 'online', 'Online');
        }

        // Update model info
        const modelName = serverInfo.model || 'Unknown Model';
        runningModels[serverType] = modelName;

        if (modelNameEl) {
            const pipelineLabel = serverType.toUpperCase();
            const pipelineColor = serverType === 'sql' ? 'sql' : (serverType === 'general' ? 'general' : 'code');
            modelNameEl.innerHTML = `
                ${modelName}
                <span class="pipeline-badge ${pipelineColor}">${pipelineLabel} Pipeline</span>
            `;
        }

        // Extract quantization from model name
        const quantMatch = modelName.match(/Q\d+[_A-Z]*/i);
        if (modelQuantEl) {
            modelQuantEl.textContent = quantMatch ? quantMatch[0] : '--';
        }

        // Get metrics from API response
        const metrics = serverInfo.metrics || {};
        const tps = metrics.tps || 0;
        const requests = metrics.requests || metrics.tokensProcessed || 0;
        const context = metrics.context || 0;
        const promptTokens = metrics.promptTokensTotal || metrics.promptTokens || 0;
        const generatedTokens = metrics.generatedTokensTotal || metrics.predictedTokens || 0;
        const totalTokens = promptTokens + generatedTokens;

        // Update UI with real metrics
        if (tpsEl) {
            tpsEl.textContent = tps > 0 ? tps.toFixed(1) : '--';
        }
        if (contextEl) {
            contextEl.textContent = context > 0 ? formatNumber(context) : '--';
        }
        if (requestsEl) {
            // Show total tokens processed as a more meaningful metric
            requestsEl.textContent = totalTokens > 0 ? formatNumber(totalTokens) : (requests > 0 ? formatNumber(requests) : '0');
        }
        if (uptimeEl) {
            const processing = metrics.requestsProcessing || metrics.slotsProcessing || 0;
            uptimeEl.textContent = processing > 0 ? `Processing (${processing})` : 'Running';
        }

        // Update metrics history with real data
        const now = new Date().toLocaleTimeString();
        metricsHistory[serverType].tps.push(tps);
        metricsHistory[serverType].requests.push(totalTokens);
        metricsHistory[serverType].timestamps.push(now);

        if (metricsHistory[serverType].tps.length > CHART_MAX_POINTS) {
            metricsHistory[serverType].tps.shift();
            metricsHistory[serverType].requests.shift();
            metricsHistory[serverType].timestamps.shift();
        }

        // Update Optimization tab's current slots/context display
        const slotsTotal = (metrics.slotsIdle || 0) + (metrics.slotsProcessing || 0);
        const currentSlotsEl = document.getElementById(`${serverType}CurrentSlots`);
        const currentCtxEl = document.getElementById(`${serverType}CurrentCtx`);

        if (currentSlotsEl) {
            currentSlotsEl.textContent = slotsTotal > 0 ? slotsTotal : '--';
        }
        if (currentCtxEl) {
            currentCtxEl.textContent = context > 0 ? formatNumber(context) : '--';
        }

        // Update RAM usage display
        const ramUsageEl = document.getElementById(`${serverType}RamUsage`);
        const ramBarEl = document.getElementById(`${serverType}RamBar`);

        // Try to get RAM info from metrics (llamacpp_kv_cache_used_bytes, etc.)
        const kvCacheUsed = metrics.kvCacheUsed || metrics.llamacpp_kv_cache_used || 0;
        const ramUsed = metrics.memoryUsed || metrics.ramUsed || kvCacheUsed;
        const totalRam = metrics.memoryTotal || 64 * 1024 * 1024 * 1024; // Default 64GB if not provided

        if (ramUsageEl && ramUsed > 0) {
            const ramGB = ramUsed / (1024 * 1024 * 1024);
            ramUsageEl.textContent = ramGB >= 1 ? `${ramGB.toFixed(1)} GB` : `${(ramUsed / (1024 * 1024)).toFixed(0)} MB`;
        } else if (ramUsageEl) {
            // Estimate RAM from model size (rough heuristic based on quantization)
            const modelSize = serverInfo.modelSize || serverInfo.parameters;
            if (modelSize) {
                ramUsageEl.textContent = modelSize;
            } else {
                ramUsageEl.textContent = '--';
            }
        }

        if (ramBarEl && ramUsed > 0) {
            const percentage = Math.min(100, (ramUsed / totalRam) * 100);
            ramBarEl.style.width = `${percentage}%`;
        } else if (ramBarEl) {
            ramBarEl.style.width = '30%'; // Default visual indicator when running
        }
    } else {
        updateServerStatusUI(statusEl, 'offline', 'Offline');
        resetServerMetrics(serverType);

        const advStatusEl = document.getElementById(`adv${serverType.charAt(0).toUpperCase() + serverType.slice(1)}Status`);
        if (advStatusEl) {
            updateServerStatusUI(advStatusEl, 'offline', 'Offline');
        }

        // Reset Optimization tab displays
        const currentSlotsEl = document.getElementById(`${serverType}CurrentSlots`);
        const currentCtxEl = document.getElementById(`${serverType}CurrentCtx`);
        if (currentSlotsEl) currentSlotsEl.textContent = '--';
        if (currentCtxEl) currentCtxEl.textContent = '--';
    }
}

/**
 * Format large numbers with K/M suffix for display
 */
function formatNumber(num) {
    if (num >= 1000000) {
        return (num / 1000000).toFixed(1) + 'M';
    } else if (num >= 1000) {
        return (num / 1000).toFixed(1) + 'K';
    }
    return num.toString();
}

// ============================================================================
// SERVER STATUS
// ============================================================================

async function checkServerStatus(serverType) {
    const server = SERVERS[serverType];
    const statusEl = document.getElementById(`${serverType}Status`);
    const modelNameEl = document.getElementById(`${serverType}ModelName`);
    const modelSizeEl = document.getElementById(`${serverType}ModelSize`);
    const modelQuantEl = document.getElementById(`${serverType}ModelQuant`);
    const tpsEl = document.getElementById(`${serverType}TPS`);
    const contextEl = document.getElementById(`${serverType}Context`);
    const requestsEl = document.getElementById(`${serverType}Requests`);
    const uptimeEl = document.getElementById(`${serverType}Uptime`);

    try {
        // Check server by calling /v1/models endpoint (OpenAI-compatible API)
        const modelsResponse = await fetch(`http://localhost:${server.port}/v1/models`, {
            method: 'GET',
            signal: AbortSignal.timeout(2000)
        });

        if (modelsResponse.ok) {
            const modelsData = await modelsResponse.json();

            updateServerStatusUI(statusEl, 'online', 'Online');

            // Update advanced tab status too
            const advStatusEl = document.getElementById(`adv${serverType.charAt(0).toUpperCase() + serverType.slice(1)}Status`);
            if (advStatusEl) {
                updateServerStatusUI(advStatusEl, 'online', 'Online');
            }

            // Get model info from /v1/models response
            if (modelsData.data && modelsData.data.length > 0) {
                const modelPath = modelsData.data[0].id || 'Unknown Model';
                // Extract just the filename from the full path
                const modelName = modelPath.split(/[/\\]/).pop();
                runningModels[serverType] = modelName;

                if (modelNameEl) {
                    // Add pipeline badge
                    const pipelineLabel = serverType.toUpperCase();
                    const pipelineColor = serverType === 'sql' ? 'sql' : (serverType === 'general' ? 'general' : 'code');
                    modelNameEl.innerHTML = `
                        ${modelName}
                        <span class="pipeline-badge ${pipelineColor}">${pipelineLabel} Pipeline</span>
                    `;
                }

                // Extract quantization from model name if available
                const quantMatch = modelName.match(/Q\d+[_A-Z]*/i);
                if (modelQuantEl) {
                    modelQuantEl.textContent = quantMatch ? quantMatch[0] : '--';
                }
            }

            // OpenAI-compatible API doesn't have /slots or /props endpoints
            // Set default values for metrics that aren't available
            if (contextEl) {
                contextEl.textContent = '--';
            }
            if (requestsEl) {
                requestsEl.textContent = '0';
            }
            if (tpsEl) {
                tpsEl.textContent = '--';
            }

            // Update metrics history with placeholder values
            const now = new Date().toLocaleTimeString();
            metricsHistory[serverType].tps.push(0);
            metricsHistory[serverType].requests.push(0);
            metricsHistory[serverType].timestamps.push(now);

            // Keep only last N points
            if (metricsHistory[serverType].tps.length > CHART_MAX_POINTS) {
                metricsHistory[serverType].tps.shift();
                metricsHistory[serverType].requests.shift();
                metricsHistory[serverType].timestamps.shift();
            }

            // Set uptime (we don't have actual uptime, so show "Running")
            if (uptimeEl) {
                uptimeEl.textContent = 'Running';
            }

        } else {
            updateServerStatusUI(statusEl, 'offline', 'Offline');
            resetServerMetrics(serverType);

            // Update advanced tab status too
            const advStatusEl = document.getElementById(`adv${serverType.charAt(0).toUpperCase() + serverType.slice(1)}Status`);
            if (advStatusEl) {
                updateServerStatusUI(advStatusEl, 'offline', 'Offline');
            }
        }
    } catch (error) {
        updateServerStatusUI(statusEl, 'offline', 'Offline');
        resetServerMetrics(serverType);

        // Update advanced tab status too
        const advStatusEl = document.getElementById(`adv${serverType.charAt(0).toUpperCase() + serverType.slice(1)}Status`);
        if (advStatusEl) {
            updateServerStatusUI(advStatusEl, 'offline', 'Offline');
        }
    }
}

function updateServerStatusUI(statusEl, status, text) {
    if (!statusEl) return;

    statusEl.className = `server-status ${status}`;
    const dot = statusEl.querySelector('.status-dot');
    const textEl = statusEl.querySelector('span:last-child');

    if (dot) {
        dot.className = `status-dot ${status}`;
    }
    if (textEl) {
        textEl.textContent = text;
    }
}

function resetServerMetrics(serverType) {
    const elements = ['ModelName', 'ModelSize', 'ModelQuant', 'TPS', 'Context', 'Requests', 'Uptime', 'RamUsage'];
    elements.forEach(el => {
        const elem = document.getElementById(`${serverType}${el}`);
        if (elem) {
            elem.textContent = el === 'ModelName' ? 'Not Running' : '--';
        }
    });
    // Reset RAM bar
    const ramBar = document.getElementById(`${serverType}RamBar`);
    if (ramBar) {
        ramBar.style.width = '0%';
    }
}

// ============================================================================
// CHARTS
// ============================================================================

function initializeMiniCharts() {
    const miniChartOptions = {
        responsive: true,
        maintainAspectRatio: false,
        animation: { duration: 200 },
        scales: {
            x: { display: false },
            y: {
                beginAtZero: true,
                display: false
            }
        },
        plugins: {
            legend: { display: false },
            tooltip: { enabled: false }
        }
    };

    // Initialize mini charts for each server
    Object.keys(SERVERS).forEach(serverType => {
        // TPS mini chart
        const tpsCtx = document.getElementById(`${serverType}TpsChart`)?.getContext('2d');
        if (tpsCtx) {
            const color = serverType === 'sql' ? '#3b82f6' : (serverType === 'general' ? '#10b981' : '#8b5cf6');
            miniCharts[serverType].tps = new Chart(tpsCtx, {
                type: 'line',
                data: {
                    labels: [],
                    datasets: [{
                        data: [],
                        borderColor: color,
                        backgroundColor: `${color}33`,
                        fill: true,
                        tension: 0.4,
                        borderWidth: 2,
                        pointRadius: 0
                    }]
                },
                options: miniChartOptions
            });
        }

        // Requests mini chart
        const reqCtx = document.getElementById(`${serverType}RequestsChart`)?.getContext('2d');
        if (reqCtx) {
            const color = serverType === 'sql' ? '#3b82f6' : (serverType === 'general' ? '#10b981' : '#8b5cf6');
            miniCharts[serverType].requests = new Chart(reqCtx, {
                type: 'bar',
                data: {
                    labels: [],
                    datasets: [{
                        data: [],
                        backgroundColor: color,
                        borderWidth: 0
                    }]
                },
                options: miniChartOptions
            });
        }
    });
}

function initializeCharts() {
    const chartOptions = {
        responsive: true,
        maintainAspectRatio: false,
        animation: {
            duration: 300
        },
        scales: {
            x: {
                display: false
            },
            y: {
                beginAtZero: true,
                grid: {
                    color: 'rgba(255, 255, 255, 0.1)'
                },
                ticks: {
                    color: 'rgba(255, 255, 255, 0.6)'
                }
            }
        },
        plugins: {
            legend: {
                display: false
            }
        }
    };

    // TPS Chart
    const tpsCtx = document.getElementById('tpsChart')?.getContext('2d');
    if (tpsCtx) {
        tpsChart = new Chart(tpsCtx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [
                    {
                        label: 'SQL',
                        data: [],
                        borderColor: '#3b82f6',
                        backgroundColor: 'rgba(59, 130, 246, 0.1)',
                        fill: true,
                        tension: 0.4
                    },
                    {
                        label: 'General',
                        data: [],
                        borderColor: '#10b981',
                        backgroundColor: 'rgba(16, 185, 129, 0.1)',
                        fill: true,
                        tension: 0.4
                    },
                    {
                        label: 'Code',
                        data: [],
                        borderColor: '#8b5cf6',
                        backgroundColor: 'rgba(139, 92, 246, 0.1)',
                        fill: true,
                        tension: 0.4
                    }
                ]
            },
            options: chartOptions
        });
    }

    // Requests Chart
    const reqCtx = document.getElementById('requestsChart')?.getContext('2d');
    if (reqCtx) {
        requestsChart = new Chart(reqCtx, {
            type: 'bar',
            data: {
                labels: ['SQL', 'General', 'Code'],
                datasets: [{
                    data: [0, 0, 0],
                    backgroundColor: ['#3b82f6', '#10b981', '#8b5cf6']
                }]
            },
            options: {
                ...chartOptions,
                scales: {
                    ...chartOptions.scales,
                    x: {
                        display: true,
                        grid: {
                            display: false
                        },
                        ticks: {
                            color: 'rgba(255, 255, 255, 0.6)'
                        }
                    }
                }
            }
        });
    }
}

function updateCharts() {
    // Update main TPS chart
    if (tpsChart) {
        const labels = metricsHistory.sql.timestamps.slice(-30);
        tpsChart.data.labels = labels;
        tpsChart.data.datasets[0].data = metricsHistory.sql.tps.slice(-30);
        tpsChart.data.datasets[1].data = metricsHistory.general.tps.slice(-30);
        tpsChart.data.datasets[2].data = metricsHistory.code.tps.slice(-30);
        tpsChart.update('none');
    }

    // Update main Requests chart
    if (requestsChart) {
        const latestRequests = [
            metricsHistory.sql.requests[metricsHistory.sql.requests.length - 1] || 0,
            metricsHistory.general.requests[metricsHistory.general.requests.length - 1] || 0,
            metricsHistory.code.requests[metricsHistory.code.requests.length - 1] || 0
        ];
        requestsChart.data.datasets[0].data = latestRequests;
        requestsChart.update('none');
    }

    // Update mini charts for each server
    Object.keys(SERVERS).forEach(serverType => {
        const history = metricsHistory[serverType];
        const charts = miniCharts[serverType];
        const labels = history.timestamps.slice(-20);

        if (charts.tps) {
            charts.tps.data.labels = labels;
            charts.tps.data.datasets[0].data = history.tps.slice(-20);
            charts.tps.update('none');
        }

        if (charts.requests) {
            charts.requests.data.labels = labels;
            charts.requests.data.datasets[0].data = history.requests.slice(-20);
            charts.requests.update('none');
        }
    });
}

// ============================================================================
// MODEL MANAGEMENT
// ============================================================================

async function loadAvailableModels() {
    try {
        const response = await fetch('/api/admin/llm/models');
        const data = await response.json();

        if (data.success && data.models) {
            availableModels = data.models;
            renderModelList(availableModels);
        } else {
            // Try to get models from local model directory
            availableModels = [];
            renderModelList([]);
        }
    } catch (error) {
        console.error('Error loading models:', error);
        showToast('Error loading models list', 'error');
        renderModelList([]);
    }
}

function renderModelList(models) {
    const container = document.getElementById('modelList');
    if (!container) return;

    // Merge available models with running models
    const allModels = [...models];
    Object.values(runningModels).forEach(modelName => {
        if (modelName && !allModels.some(m => (m.name || m) === modelName)) {
            allModels.push({ name: modelName, size: 'In Use', quantization: '' });
        }
    });

    if (!allModels || allModels.length === 0) {
        container.innerHTML = `
            <div style="padding: 40px; text-align: center; color: var(--text-muted);">
                No models found. Download a model to get started.
            </div>
        `;
        return;
    }

    container.innerHTML = allModels.map(model => {
        const modelName = model.name || model;
        const isInUse = Object.values(runningModels).includes(modelName);
        const usedBy = isInUse ? Object.keys(runningModels).find(k => runningModels[k] === modelName) : null;

        // Don't show quantization if it's "Unknown"
        const quantText = (model.quantization && model.quantization !== 'Unknown')
            ? `<span>${model.quantization}</span>`
            : '';

        return `
            <div class="model-item" data-model="${modelName}">
                <div class="model-item-info">
                    <div class="model-item-icon">
                        ${getModelIcon(modelName)}
                    </div>
                    <div class="model-item-details">
                        <h4 onclick="showModelInfo('${modelName}')">${modelName}</h4>
                        <div class="model-item-meta">
                            ${model.size ? `<span>${model.size}</span>` : ''}
                            ${quantText}
                            ${model.modified ? `<span>${formatDate(model.modified)}</span>` : ''}
                            ${isInUse ? `<span class="model-in-use-badge">In Use: ${usedBy?.toUpperCase()}</span>` : ''}
                        </div>
                        ${model.path ? `<div class="file-path">${model.path}</div>` : ''}
                    </div>
                </div>
                <div class="model-item-actions">
                    <button class="btn btn-sm btn-danger" onclick="deleteModel('${modelName}')" ${isInUse ? 'disabled' : ''}>Delete</button>
                </div>
            </div>
        `;
    }).join('');
}

function getModelIcon(modelName) {
    const name = modelName.toLowerCase();
    if (name.includes('sql') || name.includes('nsql')) return 'SQL';
    if (name.includes('code') || name.includes('coder')) return '{ }';
    if (name.includes('llama')) return 'L';
    if (name.includes('qwen')) return 'Q';
    if (name.includes('mistral')) return 'M';
    return 'AI';
}

function filterModels() {
    const search = document.getElementById('modelSearch')?.value.toLowerCase() || '';
    const filtered = availableModels.filter(model => {
        const name = (model.name || model).toLowerCase();
        return name.includes(search);
    });
    renderModelList(filtered);
}

async function deleteModel(modelName) {
    if (!confirm(`Are you sure you want to delete "${modelName}"?\n\nThis cannot be undone.`)) {
        return;
    }

    try {
        const response = await fetch(`/api/admin/llm/model/${encodeURIComponent(modelName)}`, {
            method: 'DELETE'
        });

        const data = await response.json();

        if (data.success) {
            showToast(`Model "${modelName}" deleted`, 'success');
            await loadAvailableModels();
        } else {
            showToast(`Error deleting model: ${data.error}`, 'error');
        }
    } catch (error) {
        showToast(`Error deleting model: ${error.message}`, 'error');
    }
}

function openDownloadModal() {
    const modelName = prompt('Enter model name or HuggingFace repo:\n\nExamples:\n- TheBloke/Llama-2-7B-GGUF\n- Qwen/Qwen2.5-Coder-7B-Instruct-GGUF');

    if (!modelName) return;

    downloadModel(modelName);
}

async function downloadModel(modelName) {
    showToast(`Starting download of "${modelName}"...`, 'info');

    try {
        const response = await fetch('/api/admin/llm/download', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ model: modelName })
        });

        const data = await response.json();

        if (data.success) {
            showToast(`Model "${modelName}" download started`, 'success');
            // Poll for completion
            pollDownloadStatus(modelName);
        } else {
            showToast(`Error starting download: ${data.error}`, 'error');
        }
    } catch (error) {
        showToast(`Error starting download: ${error.message}`, 'error');
    }
}

async function pollDownloadStatus(modelName) {
    // Simple polling - in production, use WebSocket or SSE
    let attempts = 0;
    const maxAttempts = 60; // 5 minutes

    const poll = async () => {
        attempts++;
        if (attempts > maxAttempts) {
            showToast('Download timeout. Check model directory manually.', 'info');
            return;
        }

        try {
            await loadAvailableModels();
            const found = availableModels.some(m =>
                (m.name || m).toLowerCase().includes(modelName.toLowerCase().split('/').pop())
            );

            if (found) {
                showToast(`Model "${modelName}" downloaded successfully!`, 'success');
                return;
            }
        } catch (e) {
            // Continue polling
        }

        setTimeout(poll, 5000);
    };

    setTimeout(poll, 5000);
}

// ============================================================================
// CONFIGURATION
// ============================================================================

async function loadConfiguration() {
    // Default configuration values
    const defaultConfig = {
        temperature: 0.7,
        top_p: 0.9,
        top_k: 40,
        repeat_penalty: 1.1,
        num_ctx: 4096,
        num_batch: 2048,
        num_gpu: -1,
        num_thread: 0
    };

    try {
        const response = await fetch('/api/llm/config');
        if (!response.ok) {
            // Endpoint doesn't exist or returned error - use defaults
            console.info('LLM config endpoint not available, using defaults');
            currentConfig = defaultConfig;
            applyConfigToUI(currentConfig);
            return;
        }
        const data = await response.json();

        if (data.success) {
            currentConfig = data.config;
            applyConfigToUI(currentConfig);
        } else {
            currentConfig = defaultConfig;
            applyConfigToUI(currentConfig);
        }
    } catch (error) {
        // Network error or JSON parse error - use defaults silently
        console.info('Using default LLM configuration');
        currentConfig = defaultConfig;
        applyConfigToUI(currentConfig);
    }
}

function applyConfigToUI(config) {
    // Generation parameters
    setSliderValue('temperature', config.temperature || 0.7);
    setSliderValue('topP', config.top_p || 0.9);
    setInputValue('topK', config.top_k || 40);
    setSliderValue('repeatPenalty', config.repeat_penalty || 1.1);

    // Optimization settings
    setSelectValue('contextSize', config.num_ctx || 4096);
    setInputValue('batchSize', config.num_batch || 2048);
    setInputValue('gpuLayers', config.num_gpu || -1);
    setInputValue('cpuThreads', config.num_thread || 0);
}

async function applyOptimization() {
    const config = {
        num_ctx: parseInt(document.getElementById('contextSize')?.value) || 4096,
        num_batch: parseInt(document.getElementById('batchSize')?.value) || 2048,
        num_gpu: parseInt(document.getElementById('gpuLayers')?.value) || -1,
        num_thread: parseInt(document.getElementById('cpuThreads')?.value) || 0,
        flash_attn: document.getElementById('flashAttention')?.checked || false,
        cont_batching: document.getElementById('continuousBatching')?.checked || true,
        cache_type_k: document.getElementById('kvCacheTypeK')?.value || 'q8_0',
        cache_type_v: document.getElementById('kvCacheTypeV')?.value || 'q8_0'
    };

    try {
        const response = await fetch('/api/llm/config', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(config)
        });

        const data = await response.json();

        if (data.success) {
            showToast('Optimization settings saved! Restart services to apply.', 'success');
            currentConfig = { ...currentConfig, ...config };
        } else {
            showToast(`Error saving settings: ${data.error}`, 'error');
        }
    } catch (error) {
        showToast(`Error saving settings: ${error.message}`, 'error');
    }
}

function resetOptimization() {
    document.getElementById('contextSize').value = '4096';
    document.getElementById('batchSize').value = '2048';
    document.getElementById('gpuLayers').value = '-1';
    document.getElementById('cpuThreads').value = '0';
    document.getElementById('flashAttention').checked = true;
    document.getElementById('continuousBatching').checked = true;
    document.getElementById('kvCacheTypeK').value = 'q8_0';
    document.getElementById('kvCacheTypeV').value = 'q8_0';

    showToast('Settings reset to defaults', 'info');
}

async function saveAdvancedSettings() {
    const config = {
        temperature: parseFloat(document.getElementById('temperature')?.value) || 0.7,
        top_p: parseFloat(document.getElementById('topP')?.value) || 0.9,
        top_k: parseInt(document.getElementById('topK')?.value) || 40,
        repeat_penalty: parseFloat(document.getElementById('repeatPenalty')?.value) || 1.1
    };

    try {
        const response = await fetch('/api/llm/config', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(config)
        });

        const data = await response.json();

        if (data.success) {
            showToast('Advanced settings saved!', 'success');
            currentConfig = { ...currentConfig, ...config };
        } else {
            showToast(`Error saving settings: ${data.error}`, 'error');
        }
    } catch (error) {
        showToast(`Error saving settings: ${error.message}`, 'error');
    }
}

// ============================================================================
// SERVICE MANAGEMENT
// ============================================================================

async function restartServer(serverType) {
    const server = SERVERS[serverType];
    showToast(`Restarting ${server.name}...`, 'info');

    try {
        const response = await fetch(`/api/admin/service/${server.serviceName}/restart`, {
            method: 'POST'
        });

        const data = await response.json();

        if (data.success) {
            showToast(`${server.name} restart initiated`, 'success');
            // Wait a bit and refresh status
            setTimeout(() => refreshAllData(), 3000);
        } else {
            showToast(`Error restarting server: ${data.error}`, 'error');
        }
    } catch (error) {
        showToast(`Error restarting server: ${error.message}`, 'error');
    }
}

async function startService(serverType) {
    const server = SERVERS[serverType];
    showToast(`Starting ${server.name}...`, 'info');

    try {
        const response = await fetch(`/api/admin/service/${server.serviceName}/start`, {
            method: 'POST'
        });

        const data = await response.json();

        if (data.success) {
            showToast(`${server.name} started`, 'success');
            setTimeout(() => refreshAllData(), 3000);
        } else {
            showToast(`Error starting service: ${data.error}`, 'error');
        }
    } catch (error) {
        showToast(`Error starting service: ${error.message}`, 'error');
    }
}

async function stopService(serverType) {
    const server = SERVERS[serverType];

    if (!confirm(`Are you sure you want to stop ${server.name}?\n\nThis will interrupt any active requests.`)) {
        return;
    }

    showToast(`Stopping ${server.name}...`, 'info');

    try {
        const response = await fetch(`/api/admin/service/${server.serviceName}/stop`, {
            method: 'POST'
        });

        const data = await response.json();

        if (data.success) {
            showToast(`${server.name} stopped`, 'success');
            setTimeout(() => refreshAllData(), 1000);
        } else {
            showToast(`Error stopping service: ${data.error}`, 'error');
        }
    } catch (error) {
        showToast(`Error stopping service: ${error.message}`, 'error');
    }
}

async function startAllServices() {
    showToast('Starting all services...', 'info');

    for (const serverType of Object.keys(SERVERS)) {
        await startService(serverType);
        await new Promise(resolve => setTimeout(resolve, 1000));
    }
}

async function stopAllServices() {
    if (!confirm('Are you sure you want to stop ALL llama.cpp services?\n\nThis will interrupt any active requests.')) {
        return;
    }

    showToast('Stopping all services...', 'info');

    for (const serverType of Object.keys(SERVERS)) {
        await stopService(serverType);
        await new Promise(resolve => setTimeout(resolve, 500));
    }
}

function openServerSettings(serverType) {
    // For now, just switch to optimization tab
    // In future, could open a modal with per-server settings
    showToast(`Configure ${SERVERS[serverType].name} settings in the Optimization tab`, 'info');
    switchTab('optimization');
}

// ============================================================================
// HELPERS
// ============================================================================

function updateSliderValue(sliderId) {
    const slider = document.getElementById(sliderId);
    const display = document.getElementById(`${sliderId}Value`);

    if (slider && display) {
        let value = parseFloat(slider.value);
        if (slider.step && parseFloat(slider.step) < 1) {
            const decimals = (slider.step.split('.')[1] || '').length;
            value = value.toFixed(decimals);
        }
        display.textContent = value;
    }
}

function setSliderValue(id, value) {
    const slider = document.getElementById(id);
    const display = document.getElementById(`${id}Value`);

    if (slider) {
        slider.value = value;
        if (display) {
            let displayValue = parseFloat(value);
            if (slider.step && parseFloat(slider.step) < 1) {
                const decimals = (slider.step.split('.')[1] || '').length;
                displayValue = displayValue.toFixed(decimals);
            }
            display.textContent = displayValue;
        }
    }
}

function setInputValue(id, value) {
    const input = document.getElementById(id);
    if (input) {
        input.value = value;
    }
}

function setSelectValue(id, value) {
    const select = document.getElementById(id);
    if (select) {
        select.value = value;
    }
}

function formatDate(dateString) {
    if (!dateString) return '';
    const date = new Date(dateString);
    return date.toLocaleDateString();
}

// ============================================================================
// TOAST NOTIFICATIONS
// ============================================================================

function showToast(message, type = 'info') {
    const container = document.getElementById('toastContainer');
    if (!container) return;

    const toast = document.createElement('div');
    toast.className = `toast toast-${type}`;
    toast.textContent = message;

    container.appendChild(toast);

    // Auto-remove after 5 seconds
    setTimeout(() => {
        toast.style.opacity = '0';
        toast.style.transform = 'translateX(100%)';
        setTimeout(() => toast.remove(), 300);
    }, 5000);
}

// ============================================================================
// LOGOUT
// ============================================================================

function logout() {
    if (typeof AuthClient !== 'undefined') {
        const auth = new AuthClient();
        auth.logout();
    } else {
        window.location.href = '/';
    }
}

// ============================================================================
// UI INTERACTIONS
// ============================================================================

function showModelInfo(modelName) {
    // Create modal overlay
    const overlay = document.createElement('div');
    overlay.className = 'modal-overlay';
    overlay.onclick = (e) => {
        if (e.target === overlay) closeModal();
    };

    // Create modal
    const modal = document.createElement('div');
    modal.className = 'modal';
    modal.innerHTML = `
        <div class="modal-header">
            <div class="modal-title">Model Information</div>
            <button class="modal-close" onclick="closeModal()">×</button>
        </div>
        <div class="modal-body">
            <div class="modal-loader">
                <div class="spinner"></div>
                <div style="color: var(--text-muted);">Loading model information...</div>
            </div>
        </div>
    `;

    overlay.appendChild(modal);
    document.body.appendChild(overlay);

    // Load model info
    loadModelInfo(modelName, modal);
}

async function loadModelInfo(modelName, modal) {
    try {
        // Try to fetch from our API first
        const response = await fetch(`/api/admin/llm/model-info/${encodeURIComponent(modelName)}`);

        let modelInfo = {};
        if (response.ok) {
            const data = await response.json();
            modelInfo = data.info || {};
        }

        // Check if model is in use
        const isInUse = Object.values(runningModels).includes(modelName);
        const usedBy = isInUse ? Object.keys(runningModels).find(k => runningModels[k] === modelName) : null;

        // Build info grid
        const infoGrid = `
            <div class="model-info-grid">
                <div class="model-info-row">
                    <div class="model-info-label">Model Name</div>
                    <div class="model-info-value">${modelName}</div>
                </div>
                ${modelInfo.size ? `
                    <div class="model-info-row">
                        <div class="model-info-label">File Size</div>
                        <div class="model-info-value">${modelInfo.size}</div>
                    </div>
                ` : ''}
                ${modelInfo.quantization ? `
                    <div class="model-info-row">
                        <div class="model-info-label">Quantization</div>
                        <div class="model-info-value">${modelInfo.quantization}</div>
                    </div>
                ` : ''}
                ${modelInfo.path ? `
                    <div class="model-info-row">
                        <div class="model-info-label">Path</div>
                        <div class="model-info-value" style="font-family: monospace; font-size: 12px; word-break: break-all;">${modelInfo.path}</div>
                    </div>
                ` : ''}
                ${modelInfo.modified ? `
                    <div class="model-info-row">
                        <div class="model-info-label">Last Modified</div>
                        <div class="model-info-value">${new Date(modelInfo.modified).toLocaleString()}</div>
                    </div>
                ` : ''}
                ${isInUse ? `
                    <div class="model-info-row">
                        <div class="model-info-label">Status</div>
                        <div class="model-info-value">
                            <span class="model-in-use-badge">In Use: ${usedBy?.toUpperCase()} Pipeline</span>
                        </div>
                    </div>
                ` : ''}
            </div>
        `;

        // Update modal body
        const modalBody = modal.querySelector('.modal-body');
        if (modalBody) {
            modalBody.innerHTML = infoGrid;
        }

    } catch (error) {
        console.error('Error loading model info:', error);
        const modalBody = modal.querySelector('.modal-body');
        if (modalBody) {
            modalBody.innerHTML = `
                <div style="padding: 40px; text-align: center; color: var(--text-muted);">
                    Error loading model information
                </div>
            `;
        }
    }
}

function closeModal() {
    const overlay = document.querySelector('.modal-overlay');
    if (overlay) {
        overlay.style.opacity = '0';
        setTimeout(() => overlay.remove(), 200);
    }
}

async function restartAllServers() {
    if (!confirm('Restart all llama.cpp servers?\n\nThis will briefly interrupt service.')) {
        return;
    }

    showToast('Restarting all servers...', 'info');

    try {
        for (const serverType of Object.keys(SERVERS)) {
            await restartServer(serverType);
            await new Promise(resolve => setTimeout(resolve, 1000));
        }
        showToast('All servers restarted', 'success');
        setTimeout(() => refreshAllData(), 3000);
    } catch (error) {
        showToast(`Error restarting servers: ${error.message}`, 'error');
    }
}

// ============================================================================
// CLEANUP
// ============================================================================

window.addEventListener('beforeunload', () => {
    if (refreshTimer) {
        clearInterval(refreshTimer);
    }
});

// ============================================================================
// HUGGING FACE MODEL BROWSER
// ============================================================================

// State for HF downloads
let activeDownloads = {};
let downloadPollers = {};

/**
 * Search Hugging Face for GGUF models
 */
async function searchHuggingFaceModels() {
    const searchInput = document.getElementById('hfSearchInput');
    const quantFilter = document.getElementById('hfQuantFilter');
    const sortBy = document.getElementById('hfSortBy');
    const loadingIndicator = document.getElementById('hfLoadingIndicator');
    const modelList = document.getElementById('hfModelList');

    const query = searchInput?.value.trim() || 'gguf';
    const quant = quantFilter?.value || '';
    const sort = sortBy?.value || 'downloads';

    // Show loading
    if (loadingIndicator) loadingIndicator.style.display = 'block';
    if (modelList) {
        modelList.innerHTML = `
            <div style="padding: 40px; text-align: center; color: var(--text-muted);">
                Searching Hugging Face...
            </div>
        `;
    }

    try {
        // Search HuggingFace API for GGUF models
        const searchUrl = `https://huggingface.co/api/models?search=${encodeURIComponent(query + ' gguf')}&filter=text-generation&sort=${sort}&direction=-1&limit=20`;

        const response = await fetch(searchUrl, {
            headers: {
                'Accept': 'application/json'
            }
        });

        if (!response.ok) {
            throw new Error(`HuggingFace API error: ${response.status}`);
        }

        const models = await response.json();

        // Filter for models with GGUF files
        const ggufModels = models.filter(model => {
            const id = model.modelId || model.id || '';
            return id.toLowerCase().includes('gguf') ||
                   (model.tags && model.tags.some(t => t.toLowerCase().includes('gguf')));
        });

        // Render results
        await renderHuggingFaceResults(ggufModels, quant);

    } catch (error) {
        console.error('Error searching Hugging Face:', error);
        if (modelList) {
            modelList.innerHTML = `
                <div style="padding: 40px; text-align: center; color: #ef4444;">
                    Error searching Hugging Face: ${error.message}
                </div>
            `;
        }
    } finally {
        if (loadingIndicator) loadingIndicator.style.display = 'none';
    }
}

/**
 * Render Hugging Face search results
 */
async function renderHuggingFaceResults(models, quantFilter) {
    const modelList = document.getElementById('hfModelList');
    if (!modelList) return;

    if (!models || models.length === 0) {
        modelList.innerHTML = `
            <div style="padding: 40px; text-align: center; color: var(--text-muted);">
                No GGUF models found. Try a different search term.
            </div>
        `;
        return;
    }

    // Create table structure for results
    let html = `
        <table style="width: 100%; border-collapse: collapse;">
            <thead>
                <tr style="border-bottom: 1px solid var(--card-border);">
                    <th style="text-align: left; padding: 12px; color: var(--text-muted); font-weight: 600;">Model</th>
                    <th style="text-align: left; padding: 12px; color: var(--text-muted); font-weight: 600;">Downloads</th>
                    <th style="text-align: left; padding: 12px; color: var(--text-muted); font-weight: 600;">Likes</th>
                    <th style="text-align: right; padding: 12px; color: var(--text-muted); font-weight: 600;">Action</th>
                </tr>
            </thead>
            <tbody>
    `;

    for (const model of models) {
        const modelId = model.modelId || model.id;
        const downloads = formatLargeNumber(model.downloads || 0);
        const likes = formatLargeNumber(model.likes || 0);
        const isDownloading = activeDownloads[modelId];

        html += `
            <tr style="border-bottom: 1px solid var(--card-border);" data-model-id="${modelId}">
                <td style="padding: 12px;">
                    <div style="font-weight: 600; color: var(--text-primary);">${modelId}</div>
                    <div style="font-size: 12px; color: var(--text-muted);">
                        ${model.tags?.slice(0, 3).join(', ') || 'GGUF'}
                    </div>
                </td>
                <td style="padding: 12px; color: var(--accent-cyan);">${downloads}</td>
                <td style="padding: 12px; color: #f59e0b;">${likes}</td>
                <td style="padding: 12px; text-align: right;">
                    <button
                        class="btn btn-sm"
                        style="background: #166534; color: white; padding: 6px 12px; font-size: 12px;"
                        onclick="browseModelFiles('${modelId}')"
                        ${isDownloading ? 'disabled' : ''}
                    >
                        ${isDownloading ? 'Downloading...' : 'Browse Files'}
                    </button>
                </td>
            </tr>
        `;
    }

    html += `
            </tbody>
        </table>
    `;

    modelList.innerHTML = html;
}

/**
 * Browse files for a specific model
 */
async function browseModelFiles(modelId) {
    try {
        showToast('Loading model files...', 'info');

        // Fetch file tree from HuggingFace
        const treeUrl = `https://huggingface.co/api/models/${modelId}/tree/main`;
        const response = await fetch(treeUrl);

        if (!response.ok) {
            throw new Error(`Failed to fetch model files: ${response.status}`);
        }

        const files = await response.json();

        // Filter for GGUF files
        const ggufFiles = files.filter(f => f.path && f.path.toLowerCase().endsWith('.gguf'));

        if (ggufFiles.length === 0) {
            showToast('No GGUF files found in this model repository', 'error');
            return;
        }

        // Show file selection modal
        showFileSelectionModal(modelId, ggufFiles);

    } catch (error) {
        console.error('Error fetching model files:', error);
        showToast(`Error fetching model files: ${error.message}`, 'error');
    }
}

/**
 * Show modal for selecting a specific GGUF file to download
 */
function showFileSelectionModal(modelId, files) {
    // Remove existing modal if any
    const existingModal = document.getElementById('hfFileModal');
    if (existingModal) existingModal.remove();

    // Create modal
    const modal = document.createElement('div');
    modal.id = 'hfFileModal';
    modal.className = 'modal-overlay';
    modal.innerHTML = `
        <div class="modal" style="max-width: 700px;">
            <div class="modal-header">
                <div class="modal-title">Select GGUF File - ${modelId}</div>
                <button class="modal-close" onclick="closeHfFileModal()">&times;</button>
            </div>
            <div class="modal-body" style="max-height: 400px; overflow-y: auto;">
                <table style="width: 100%; border-collapse: collapse;">
                    <thead>
                        <tr style="border-bottom: 1px solid var(--card-border);">
                            <th style="text-align: left; padding: 8px; color: var(--text-muted);">File</th>
                            <th style="text-align: right; padding: 8px; color: var(--text-muted);">Size</th>
                            <th style="text-align: right; padding: 8px; color: var(--text-muted);">Action</th>
                        </tr>
                    </thead>
                    <tbody>
                        ${files.map(file => {
                            const sizeGB = (file.size / (1024 * 1024 * 1024)).toFixed(2);
                            const quantMatch = file.path.match(/[QqIi]\d+[_A-Z0-9]*/);
                            const quant = quantMatch ? quantMatch[0] : '';
                            return `
                                <tr style="border-bottom: 1px solid var(--card-border);">
                                    <td style="padding: 8px;">
                                        <div style="font-weight: 500; color: var(--text-primary); word-break: break-all;">${file.path}</div>
                                        ${quant ? `<span style="background: rgba(0, 212, 255, 0.2); color: var(--accent-cyan); padding: 2px 6px; border-radius: 4px; font-size: 11px;">${quant}</span>` : ''}
                                    </td>
                                    <td style="padding: 8px; text-align: right; color: var(--text-muted);">${sizeGB} GB</td>
                                    <td style="padding: 8px; text-align: right;">
                                        <button
                                            class="btn btn-sm"
                                            style="background: #166534; color: white; padding: 4px 10px; font-size: 11px;"
                                            onclick="startHuggingFaceDownload('${modelId}', '${file.path}')"
                                        >
                                            Download
                                        </button>
                                    </td>
                                </tr>
                            `;
                        }).join('')}
                    </tbody>
                </table>
            </div>
        </div>
    `;

    document.body.appendChild(modal);
}

/**
 * Close the file selection modal
 */
function closeHfFileModal() {
    const modal = document.getElementById('hfFileModal');
    if (modal) modal.remove();
}

/**
 * Start downloading a model file from Hugging Face
 */
async function startHuggingFaceDownload(modelId, filename) {
    closeHfFileModal();

    const downloadId = `${modelId}/${filename}`;
    activeDownloads[downloadId] = { progress: 0, status: 'starting' };

    // Show download progress section
    showDownloadProgress(downloadId, filename);

    showToast(`Starting download: ${filename}`, 'info');

    try {
        const response = await fetch('/api/admin/llm/download-hf', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                modelId: modelId,
                filename: filename
            })
        });

        const data = await response.json();

        if (data.success) {
            showToast(`Download started: ${filename}`, 'success');
            // Start polling for progress
            startDownloadProgressPolling(downloadId, data.downloadId || downloadId);
        } else {
            showToast(`Download failed: ${data.error}`, 'error');
            delete activeDownloads[downloadId];
            updateDownloadProgressUI(downloadId, 'error', data.error);
        }
    } catch (error) {
        showToast(`Download error: ${error.message}`, 'error');
        delete activeDownloads[downloadId];
        updateDownloadProgressUI(downloadId, 'error', error.message);
    }
}

/**
 * Show download progress section
 */
function showDownloadProgress(downloadId, filename) {
    const section = document.getElementById('downloadProgressSection');
    const list = document.getElementById('downloadProgressList');

    if (!section || !list) return;

    section.style.display = 'block';

    // Add progress item if not exists
    const itemIdSafe = downloadId.replace(/[/\\]/g, '-');
    const existingItem = document.getElementById(`download-${itemIdSafe}`);
    if (!existingItem) {
        const item = document.createElement('div');
        item.id = `download-${itemIdSafe}`;
        item.style.cssText = 'padding: 12px; border-bottom: 1px solid var(--card-border);';
        item.innerHTML = `
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                <span style="font-weight: 600; color: var(--text-primary);">${filename}</span>
                <span class="download-status" style="font-size: 12px; color: var(--text-muted);">Starting...</span>
            </div>
            <div class="progress-bar">
                <div class="progress-fill" style="width: 0%;"></div>
            </div>
            <div class="download-speed" style="font-size: 11px; color: var(--text-muted); margin-top: 4px;"></div>
        `;
        list.appendChild(item);
    }
}

/**
 * Update download progress UI
 */
function updateDownloadProgressUI(downloadId, status, message, progress = 0, speed = '') {
    const itemIdSafe = downloadId.replace(/[/\\]/g, '-');
    const item = document.getElementById(`download-${itemIdSafe}`);

    if (!item) return;

    const statusEl = item.querySelector('.download-status');
    const progressBar = item.querySelector('.progress-fill');
    const speedEl = item.querySelector('.download-speed');

    if (statusEl) {
        if (status === 'complete') {
            statusEl.textContent = 'Complete!';
            statusEl.style.color = '#10b981';
        } else if (status === 'error') {
            statusEl.textContent = `Error: ${message}`;
            statusEl.style.color = '#ef4444';
        } else if (status === 'downloading') {
            statusEl.textContent = `${progress}%`;
            statusEl.style.color = 'var(--accent-cyan)';
        } else {
            statusEl.textContent = message || 'Starting...';
        }
    }

    if (progressBar) {
        progressBar.style.width = `${progress}%`;
        if (status === 'complete') {
            progressBar.style.background = 'linear-gradient(90deg, #10b981, #059669)';
        } else if (status === 'error') {
            progressBar.style.background = '#ef4444';
        }
    }

    if (speedEl && speed) {
        speedEl.textContent = speed;
    }
}

/**
 * Poll for download progress
 */
function startDownloadProgressPolling(downloadId, serverDownloadId) {
    // Clear existing poller
    if (downloadPollers[downloadId]) {
        clearInterval(downloadPollers[downloadId]);
    }

    downloadPollers[downloadId] = setInterval(async () => {
        try {
            const response = await fetch(`/api/admin/llm/download-status/${encodeURIComponent(serverDownloadId)}`);
            const data = await response.json();

            if (data.success) {
                const { status, progress, speed, error } = data;

                updateDownloadProgressUI(downloadId, status, error, progress, speed);

                if (status === 'complete') {
                    clearInterval(downloadPollers[downloadId]);
                    delete downloadPollers[downloadId];
                    delete activeDownloads[downloadId];
                    showToast('Download complete! Refreshing model list...', 'success');
                    await loadAvailableModels();
                } else if (status === 'error') {
                    clearInterval(downloadPollers[downloadId]);
                    delete downloadPollers[downloadId];
                    delete activeDownloads[downloadId];
                }
            }
        } catch (error) {
            console.error('Error polling download status:', error);
        }
    }, 2000);
}

/**
 * Format large numbers with K/M suffix
 */
function formatLargeNumber(num) {
    if (num >= 1000000) {
        return (num / 1000000).toFixed(1) + 'M';
    } else if (num >= 1000) {
        return (num / 1000).toFixed(1) + 'K';
    }
    return num.toString();
}

// Allow Enter key to trigger HF search
document.addEventListener('DOMContentLoaded', () => {
    const hfSearchInput = document.getElementById('hfSearchInput');
    if (hfSearchInput) {
        hfSearchInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                searchHuggingFaceModels();
            }
        });
    }

    // Initialize backend services section
    checkBackendServiceStatuses();
    loadMongoDBUri();
});

// ============================================================================
// BACKEND SERVICES (Python Service & MongoDB)
// ============================================================================

/**
 * Check status of Python service and MongoDB
 */
async function checkBackendServiceStatuses() {
    await Promise.all([
        checkPythonServiceStatus(),
        checkMongoDBStatus()
    ]);

    // Refresh every 30 seconds
    setTimeout(checkBackendServiceStatuses, 30000);
}

/**
 * Check Python service status
 */
async function checkPythonServiceStatus() {
    const statusEl = document.getElementById('pythonServiceStatus');
    if (!statusEl) return;

    try {
        const response = await fetch('/api/admin/service/python/status', {
            headers: getAuthHeaders()
        });

        if (response.ok) {
            const data = await response.json();
            updateServiceStatusUI(statusEl, data.status === 'running' ? 'online' : 'offline',
                data.status === 'running' ? 'Running' : 'Stopped');
        } else {
            updateServiceStatusUI(statusEl, 'offline', 'Offline');
        }
    } catch (error) {
        console.error('Failed to check Python service status:', error);
        updateServiceStatusUI(statusEl, 'offline', 'Offline');
    }
}

/**
 * Check MongoDB connection status from Python service
 */
async function checkMongoDBStatus() {
    const statusEl = document.getElementById('mongodbServiceStatus');
    const uriEl = document.getElementById('mongoDBUri');
    if (!statusEl) return;

    try {
        // Use consolidated /status endpoint
        const response = await fetch('http://localhost:8001/status', {
            signal: AbortSignal.timeout(5000)
        });

        if (response.ok) {
            const data = await response.json();
            const isConnected = data.mongodb?.connected === true;

            updateServiceStatusUI(statusEl, isConnected ? 'online' : 'offline',
                isConnected ? 'Connected' : 'Disconnected');

            // Update the URI field
            if (uriEl && data.mongodb?.uri) {
                uriEl.value = data.mongodb.uri;
            }
        } else {
            updateServiceStatusUI(statusEl, 'offline', 'Disconnected');
        }
    } catch (error) {
        console.error('Failed to check MongoDB status:', error);
        updateServiceStatusUI(statusEl, 'offline', 'Unavailable');
    }
}

/**
 * Update service status UI element
 */
function updateServiceStatusUI(statusEl, status, text) {
    const dot = statusEl.querySelector('.status-dot');
    const span = statusEl.querySelector('span:last-child');

    statusEl.classList.remove('online', 'offline', 'loading');
    statusEl.classList.add(status);

    if (dot) {
        dot.classList.remove('online', 'offline');
        dot.classList.add(status);
    }

    if (span) {
        span.textContent = text;
    }
}

/**
 * Restart Python service
 */
async function restartPythonService() {
    if (!confirm('Are you sure you want to restart the Python service?')) return;

    const statusEl = document.getElementById('pythonServiceStatus');
    updateServiceStatusUI(statusEl, 'loading', 'Restarting...');

    try {
        const response = await fetch('/api/admin/service/python/restart', {
            method: 'POST',
            headers: getAuthHeaders()
        });

        const data = await response.json();

        if (data.success) {
            showToast(data.message + (data.instructions ? '\n\n' + data.instructions : ''), 'success');
            setTimeout(checkPythonServiceStatus, 3000);
        } else {
            showToast('Failed to restart: ' + data.error, 'error');
            updateServiceStatusUI(statusEl, 'offline', 'Error');
        }
    } catch (error) {
        showToast('Failed to restart service: ' + error.message, 'error');
        updateServiceStatusUI(statusEl, 'offline', 'Error');
    }
}

/**
 * Stop Python service
 */
async function stopPythonService() {
    if (!confirm('Are you sure you want to stop the Python service?')) return;

    const statusEl = document.getElementById('pythonServiceStatus');
    updateServiceStatusUI(statusEl, 'loading', 'Stopping...');

    try {
        const response = await fetch('/api/admin/service/python/stop', {
            method: 'POST',
            headers: getAuthHeaders()
        });

        const data = await response.json();

        if (data.success) {
            showToast(data.message, 'success');
            setTimeout(checkPythonServiceStatus, 2000);
        } else {
            showToast(data.message || 'Failed to stop: ' + data.error, 'error');
            updateServiceStatusUI(statusEl, 'offline', 'Error');
        }
    } catch (error) {
        showToast('Failed to stop service: ' + error.message, 'error');
        updateServiceStatusUI(statusEl, 'offline', 'Error');
    }
}

/**
 * Load MongoDB URI from Python service
 */
async function loadMongoDBUri() {
    // MongoDB URI is loaded via checkMongoDBStatus() which pulls from Python service
    // This function is kept for backwards compatibility but delegates to the status check
    await checkMongoDBStatus();
}

/**
 * Test MongoDB connection via Python service
 */
async function testMongoDBConnection() {
    const resultEl = document.getElementById('mongoTestResult');
    const testBtn = document.getElementById('btnTestMongo');

    testBtn.disabled = true;
    testBtn.textContent = 'Testing...';

    try {
        // Test connection using consolidated /status endpoint
        const response = await fetch('http://localhost:8001/status', {
            signal: AbortSignal.timeout(10000)
        });

        resultEl.style.display = 'block';
        if (response.ok) {
            const data = await response.json();
            if (data.mongodb?.connected) {
                resultEl.style.background = 'rgba(34, 197, 94, 0.1)';
                resultEl.style.border = '1px solid rgba(34, 197, 94, 0.3)';
                resultEl.style.color = '#22c55e';
                resultEl.textContent = `Connected! Version: ${data.mongodb.version || 'N/A'}`;
            } else {
                resultEl.style.background = 'rgba(239, 68, 68, 0.1)';
                resultEl.style.border = '1px solid rgba(239, 68, 68, 0.3)';
                resultEl.style.color = '#ef4444';
                resultEl.textContent = 'MongoDB is not connected';
            }
        } else {
            resultEl.style.background = 'rgba(239, 68, 68, 0.1)';
            resultEl.style.border = '1px solid rgba(239, 68, 68, 0.3)';
            resultEl.style.color = '#ef4444';
            resultEl.textContent = 'Failed to check MongoDB status';
        }
    } catch (error) {
        resultEl.style.display = 'block';
        resultEl.style.background = 'rgba(239, 68, 68, 0.1)';
        resultEl.style.border = '1px solid rgba(239, 68, 68, 0.3)';
        resultEl.style.color = '#ef4444';
        resultEl.textContent = 'Python service unavailable: ' + error.message;
    } finally {
        testBtn.disabled = false;
        testBtn.textContent = 'Test';
    }
}

/**
 * Reconnect to MongoDB via Python service
 */
async function reconnectMongoDB() {
    const resultEl = document.getElementById('mongoTestResult');
    const reconnectBtn = document.getElementById('btnReconnectMongo');

    reconnectBtn.disabled = true;
    reconnectBtn.textContent = 'Reconnecting...';

    try {
        // Call Python service reconnect endpoint
        const response = await fetch('http://localhost:8001/admin/mongodb-reconnect', {
            method: 'POST',
            signal: AbortSignal.timeout(15000)
        });

        resultEl.style.display = 'block';
        if (response.ok) {
            const data = await response.json();
            if (data.success || data.connected) {
                resultEl.style.background = 'rgba(34, 197, 94, 0.1)';
                resultEl.style.border = '1px solid rgba(34, 197, 94, 0.3)';
                resultEl.style.color = '#22c55e';
                resultEl.textContent = 'MongoDB reconnected successfully!';
                // Refresh status
                setTimeout(checkMongoDBStatus, 1000);
            } else {
                resultEl.style.background = 'rgba(239, 68, 68, 0.1)';
                resultEl.style.border = '1px solid rgba(239, 68, 68, 0.3)';
                resultEl.style.color = '#ef4444';
                resultEl.textContent = 'Reconnect failed: ' + (data.error || data.message || 'Unknown error');
            }
        } else {
            const errorData = await response.json().catch(() => ({}));
            resultEl.style.background = 'rgba(239, 68, 68, 0.1)';
            resultEl.style.border = '1px solid rgba(239, 68, 68, 0.3)';
            resultEl.style.color = '#ef4444';
            resultEl.textContent = 'Reconnect failed: ' + (errorData.detail || errorData.error || response.statusText);
        }
    } catch (error) {
        resultEl.style.display = 'block';
        resultEl.style.background = 'rgba(239, 68, 68, 0.1)';
        resultEl.style.border = '1px solid rgba(239, 68, 68, 0.3)';
        resultEl.style.color = '#ef4444';
        resultEl.textContent = 'Reconnect failed: ' + error.message;
    } finally {
        reconnectBtn.disabled = false;
        reconnectBtn.textContent = 'Reconnect';
    }
}

/**
 * Get authorization headers
 */
function getAuthHeaders() {
    if (typeof AuthClient !== 'undefined') {
        const auth = new AuthClient();
        const token = auth.getAccessToken();
        if (token) {
            return { 'Authorization': `Bearer ${token}` };
        }
    }
    return {};
}
