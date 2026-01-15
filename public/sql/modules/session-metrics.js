/**
 * Session Metrics Module
 *
 * Handles tracking and displaying session metrics for the SQL Chat application.
 * Tracks query counts, success rates, response times, row counts, and token usage.
 *
 * @module session-metrics
 */

import { state } from './sql-chat-state.js';

/**
 * Callback function for updating token display.
 * Set via setTokenDisplayCallback() to allow the main module to provide the implementation.
 * @type {Function|null}
 */
let tokenDisplayCallback = null;

/**
 * Set the callback function for updating the token display.
 * This allows modules to provide a custom updateTokenDisplay function.
 *
 * @param {Function} callback - The function to call when token display needs updating
 */
export function setTokenDisplayCallback(callback) {
    tokenDisplayCallback = callback;
}

/**
 * Update session metrics based on query results.
 * Called after each query completes to track statistics.
 *
 * @param {Object} result - The query result object
 * @param {boolean} result.success - Whether the query succeeded
 * @param {Array} [result.rows] - The rows returned by the query
 * @param {Object} [result.tokenUsage] - Token usage information
 * @param {number} [result.tokenUsage.prompt_tokens] - Number of prompt tokens used
 * @param {number} [result.tokenUsage.completion_tokens] - Number of completion tokens used
 * @param {number} [result.tokenUsage.total_tokens] - Total tokens used
 */
export function updateSessionMetrics(result) {
    state.sessionMetrics.totalQueries++;

    if (result && result.success) {
        state.sessionMetrics.successfulQueries++;

        // Track rows returned
        if (result.rows && result.rows.length > 0) {
            state.sessionMetrics.totalRows += result.rows.length;
        }
    }

    // Track response time
    if (state.sessionMetrics.currentQueryStartTime) {
        const responseTime = Date.now() - state.sessionMetrics.currentQueryStartTime;
        state.sessionMetrics.responseTimes.push(responseTime);
        state.sessionMetrics.currentQueryStartTime = null;
    }

    // Track tokens
    if (result && result.tokenUsage) {
        const promptTokens = result.tokenUsage.prompt_tokens || 0;
        const completionTokens = result.tokenUsage.completion_tokens || 0;
        const totalTokens = result.tokenUsage.total_tokens || (promptTokens + completionTokens);

        if (totalTokens > 0) {
            state.sessionMetrics.totalTokens += totalTokens;
            // Update token tracker for display
            state.tokenTracker.promptTokens = promptTokens;
            state.tokenTracker.completionTokens = completionTokens;
            state.tokenTracker.totalTokens = totalTokens;

            // Call the token display callback if set
            if (tokenDisplayCallback) {
                tokenDisplayCallback();
            }
        }
    }

    // Update display
    displaySessionMetrics();
}

/**
 * Display current session metrics in the UI.
 * Updates the metrics container with current values for queries, success rate,
 * average response time, total rows, and tokens used.
 */
export function displaySessionMetrics() {
    const metricsContainer = document.getElementById('chatMetrics');
    if (!metricsContainer) return;

    // Show the metrics container
    metricsContainer.style.display = 'grid';

    // Update individual metrics
    document.getElementById('metricQueries').textContent = state.sessionMetrics.totalQueries;

    const successEl = document.getElementById('metricSuccess');
    successEl.textContent = state.sessionMetrics.successfulQueries;
    successEl.className = 'metric-value' + (state.sessionMetrics.successfulQueries > 0 ? ' success' : '');

    // Calculate average response time
    const avgTime = state.sessionMetrics.responseTimes.length > 0
        ? Math.round(state.sessionMetrics.responseTimes.reduce((a, b) => a + b, 0) / state.sessionMetrics.responseTimes.length)
        : 0;
    document.getElementById('metricAvgTime').textContent = avgTime > 1000
        ? `${(avgTime / 1000).toFixed(1)}s`
        : `${avgTime}ms`;

    document.getElementById('metricTotalRows').textContent = state.sessionMetrics.totalRows.toLocaleString();

    const tokensEl = document.getElementById('metricTokensUsed');
    tokensEl.textContent = state.sessionMetrics.totalTokens > 1000
        ? `${(state.sessionMetrics.totalTokens / 1000).toFixed(1)}k`
        : state.sessionMetrics.totalTokens;
}

/**
 * Reset session metrics to initial state.
 * Called when clearing chat or starting a new session.
 * Hides the metrics container and resets all counters.
 */
export function resetSessionMetrics() {
    state.sessionMetrics = {
        totalQueries: 0,
        successfulQueries: 0,
        totalRows: 0,
        totalTokens: 0,
        responseTimes: [],
        currentQueryStartTime: null
    };

    const metricsContainer = document.getElementById('chatMetrics');
    if (metricsContainer) {
        metricsContainer.style.display = 'none';
    }
}
