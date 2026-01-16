/**
 * System Status Checker
 * Fetches system status from Python service (single endpoint)
 *
 * Server-side health checks run every 2 minutes and cache results.
 * Browser polls cached status every 5 seconds for responsive UI updates.
 */

// Configuration
// Use relative URL to go through Node.js proxy - works from any browser
const PYTHON_SERVICE_URL = '/api/python';
const STATUS_CHECK_INTERVAL = 5000; // 5 seconds - polls cached server status

// Status state
let lastStatusKey = null;
let statusCheckTimer = null;
let currentStatus = null;

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
 * Get overall status key for comparison
 */
function getStatusKey(status) {
    if (status.error) return 'error';

    const mongoOk = status.mongodb?.connected === true;
    const vectorOk = status.vector_search?.native_available === true;
    const llmOk = status.llm?.healthy === true;

    if (mongoOk && vectorOk && llmOk) return 'healthy';
    if (mongoOk && vectorOk) return 'degraded-llm';
    if (mongoOk) return 'degraded';
    return 'offline';
}

/**
 * Update status indicator UI
 */
function updateStatusIndicator(status) {
    currentStatus = status;
    const statusKey = getStatusKey(status);

    // Only update if status changed
    if (statusKey !== lastStatusKey) {
        lastStatusKey = statusKey;

        // Update legacy .status-indicator element
        const indicator = document.querySelector('.status-indicator');
        if (indicator) {
            indicator.classList.remove('healthy', 'degraded', 'offline', 'error');

            switch (statusKey) {
                case 'healthy':
                    indicator.classList.add('healthy');
                    indicator.title = 'All systems operational';
                    break;
                case 'degraded-llm':
                    indicator.classList.add('degraded');
                    indicator.title = 'LLM service unavailable';
                    break;
                case 'degraded':
                    indicator.classList.add('degraded');
                    indicator.title = 'Some services degraded';
                    break;
                case 'error':
                    indicator.classList.add('error');
                    indicator.title = status.error || 'Service error';
                    break;
                default:
                    indicator.classList.add('offline');
                    indicator.title = 'Services offline';
            }
        }

        // Update header status elements (statusDot + statusText)
        const statusDot = document.getElementById('statusDot');
        const statusText = document.getElementById('statusText');

        if (statusDot && statusText) {
            // Remove existing status classes
            statusDot.classList.remove('status-healthy', 'status-degraded', 'status-offline', 'status-error');

            switch (statusKey) {
                case 'healthy':
                    statusDot.classList.add('status-healthy');
                    statusDot.style.background = 'var(--accent-green, #10b981)';
                    statusText.textContent = 'System Online';
                    statusText.style.color = 'var(--accent-green, #10b981)';
                    break;
                case 'degraded-llm':
                    statusDot.classList.add('status-degraded');
                    statusDot.style.background = 'var(--accent-yellow, #f59e0b)';
                    statusText.textContent = 'LLM Unavailable';
                    statusText.style.color = 'var(--accent-yellow, #f59e0b)';
                    break;
                case 'degraded':
                    statusDot.classList.add('status-degraded');
                    statusDot.style.background = 'var(--accent-yellow, #f59e0b)';
                    statusText.textContent = 'Degraded';
                    statusText.style.color = 'var(--accent-yellow, #f59e0b)';
                    break;
                case 'error':
                    statusDot.classList.add('status-error');
                    statusDot.style.background = 'var(--accent-red, #ef4444)';
                    statusText.textContent = status.error || 'Error';
                    statusText.style.color = 'var(--accent-red, #ef4444)';
                    break;
                default:
                    statusDot.classList.add('status-offline');
                    statusDot.style.background = 'var(--accent-red, #ef4444)';
                    statusText.textContent = 'Offline';
                    statusText.style.color = 'var(--accent-red, #ef4444)';
            }
        }

        // Update any ewr-header-status components
        document.querySelectorAll('ewr-header-status').forEach(component => {
            if (typeof component.updateStatus === 'function') {
                component.updateStatus(status, statusKey);
            }
        });
    }

    // Dispatch vector search status event for knowledge base and other pages
    const vectorSearchAvailable = status.vector_search?.native_available === true;
    const mongoUri = status.mongodb?.uri || null;
    const vectorSearchError = status.error || (!vectorSearchAvailable ? 'Vector search unavailable' : null);

    window.dispatchEvent(new CustomEvent('vectorSearchStatusChanged', {
        detail: {
            available: vectorSearchAvailable,
            error: vectorSearchError,
            mongoUri: mongoUri
        }
    }));

    // Also dispatch a general system status event
    window.dispatchEvent(new CustomEvent('systemStatusChanged', {
        detail: {
            status: status,
            statusKey: statusKey
        }
    }));
}

/**
 * Get current vector search status (for pages that need to check synchronously)
 */
function getVectorSearchStatus() {
    if (!currentStatus) {
        return { available: false, error: 'Status not yet loaded', mongoUri: null };
    }
    return {
        available: currentStatus.vector_search?.native_available === true,
        error: currentStatus.error || null,
        mongoUri: currentStatus.mongodb?.uri || null
    };
}

/**
 * Perform status check and update UI
 */
async function performStatusCheck() {
    const status = await fetchSystemStatus();
    updateStatusIndicator(status);
    return status;
}

/**
 * Start periodic status checks
 */
function startStatusChecks() {
    if (statusCheckTimer) {
        clearInterval(statusCheckTimer);
    }

    performStatusCheck();
    statusCheckTimer = setInterval(performStatusCheck, STATUS_CHECK_INTERVAL);
}

// Initialize on page load
document.addEventListener('DOMContentLoaded', startStatusChecks);

// Refresh when page becomes visible
document.addEventListener('visibilitychange', () => {
    if (document.visibilityState === 'visible') {
        performStatusCheck();
    }
});

// Refresh immediately when user logs in
window.addEventListener('userLoggedIn', () => {
    performStatusCheck();
});
