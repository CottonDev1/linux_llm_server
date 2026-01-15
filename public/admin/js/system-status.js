/**
 * System Status Checker
 * Fetches system status from Python service (single endpoint)
 *
 * Status checks are performed server-side every 2 minutes.
 * Browser fetches cached status at the same interval.
 */

// Configuration
const SERVER_HOST = window.location.hostname || 'localhost';
const PYTHON_SERVICE_URL = `http://${SERVER_HOST}:8001`;
const STATUS_CHECK_INTERVAL = 120000; // 2 minutes - matches server-side check interval

// Status state
let currentStatus = null;
let statusCheckTimer = null;

/**
 * Fetch system status from Python service
 * @returns {Promise<object>} Complete system status
 */
async function fetchSystemStatus() {
    try {
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), 5000);

        const response = await fetch(`${PYTHON_SERVICE_URL}/status`, {
            method: 'GET',
            signal: controller.signal,
            headers: { 'Accept': 'application/json' }
        });

        clearTimeout(timeoutId);

        if (!response.ok) {
            return { error: `Service returned status ${response.status}` };
        }

        return await response.json();
    } catch (error) {
        if (error.name === 'AbortError') {
            return { error: 'Timeout' };
        }
        return { error: 'Service unavailable' };
    }
}

/**
 * Update the status display UI
 */
function updateStatusDisplay(status) {
    currentStatus = status;

    // Update Python/MongoDB status
    const pythonStatus = document.getElementById('pythonServiceStatus');
    if (pythonStatus) {
        if (status.error) {
            updateStatusElement(pythonStatus, false, status.error);
        } else {
            const mongoConnected = status.mongodb?.connected === true;
            updateStatusElement(pythonStatus, mongoConnected, mongoConnected ? 'Connected' : 'Disconnected');
        }
    }

    // Update LLM status indicators
    const llmEndpoints = status.llm?.endpoints || {};

    ['sql', 'general', 'code'].forEach(name => {
        const el = document.getElementById(`${name}LlmStatus`);
        if (el) {
            const endpoint = llmEndpoints[name];
            const isHealthy = endpoint?.healthy === true;
            updateStatusElement(el, isHealthy, isHealthy ? 'Online' : 'Offline');
        }
    });

    // Update vector search status
    const vectorEl = document.getElementById('vectorSearchStatus');
    if (vectorEl) {
        const vectorAvailable = status.vector_search?.native_available === true;
        updateStatusElement(vectorEl, vectorAvailable, vectorAvailable ? 'Available' : 'Unavailable');
    }

    // Update embeddings status
    const embeddingsEl = document.getElementById('embeddingsStatus');
    if (embeddingsEl) {
        const embeddingsLoaded = status.embeddings?.loaded === true;
        updateStatusElement(embeddingsEl, embeddingsLoaded, embeddingsLoaded ? 'Loaded' : 'Not loaded');
    }

    // Update overall status
    const overallEl = document.getElementById('overallStatus');
    if (overallEl) {
        updateStatusElement(overallEl, status.status === 'healthy', status.status || 'Unknown');
    }

    // Update last checked time
    const checkedEl = document.getElementById('lastChecked');
    if (checkedEl && status.checked_at) {
        const checkedTime = new Date(status.checked_at).toLocaleTimeString();
        checkedEl.textContent = `Last checked: ${checkedTime}`;
    }
}

/**
 * Update a status element's appearance
 */
function updateStatusElement(el, isOnline, text) {
    el.classList.remove('online', 'offline', 'loading');
    el.classList.add(isOnline ? 'online' : 'offline');

    const dot = el.querySelector('.status-dot');
    if (dot) {
        dot.classList.remove('online', 'offline');
        dot.classList.add(isOnline ? 'online' : 'offline');
    }

    const span = el.querySelector('span:last-child');
    if (span) {
        span.textContent = text;
    }
}

/**
 * Perform status check and update UI
 */
async function performStatusCheck() {
    const status = await fetchSystemStatus();
    updateStatusDisplay(status);
    return status;
}

/**
 * Start periodic status checks
 */
function startStatusChecks() {
    // Stop any existing timer
    if (statusCheckTimer) {
        clearInterval(statusCheckTimer);
    }

    // Initial check
    performStatusCheck();

    // Schedule periodic checks
    statusCheckTimer = setInterval(performStatusCheck, STATUS_CHECK_INTERVAL);
}

/**
 * Stop status checks
 */
function stopStatusChecks() {
    if (statusCheckTimer) {
        clearInterval(statusCheckTimer);
        statusCheckTimer = null;
    }
}

/**
 * Force refresh status (calls server-side refresh)
 */
async function forceRefreshStatus() {
    try {
        const response = await fetch(`${PYTHON_SERVICE_URL}/status/refresh`, {
            method: 'POST',
            headers: { 'Accept': 'application/json' }
        });

        if (response.ok) {
            const status = await response.json();
            updateStatusDisplay(status);
            return status;
        }
    } catch (error) {
        console.error('Failed to refresh status:', error);
    }
    return null;
}

// Initialize on page load
document.addEventListener('DOMContentLoaded', () => {
    startStatusChecks();
});

// Handle visibility changes - refresh when page becomes visible
document.addEventListener('visibilitychange', () => {
    if (document.visibilityState === 'visible') {
        performStatusCheck();
    }
});

// Export for external use
window.SystemStatus = {
    fetch: fetchSystemStatus,
    check: performStatusCheck,
    refresh: forceRefreshStatus,
    start: startStatusChecks,
    stop: stopStatusChecks,
    getCurrent: () => currentStatus
};
