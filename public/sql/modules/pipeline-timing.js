/**
 * Pipeline Timing Module
 *
 * Handles real-time display and tracking of SQL query pipeline execution timing.
 * Provides visual feedback for each step of the query processing pipeline,
 * including preprocessing, security checks, rule matching, schema loading,
 * SQL generation, fixing, and execution.
 *
 * @module pipeline-timing
 */

import { state } from './sql-chat-state.js';

/**
 * Module-level tracker for pipeline timing state.
 * Tracks the current step, start times, and accumulated step timings.
 * This is separate from state.pipelineTiming to maintain backward compatibility.
 */
let pipelineTimingTracker = {
    startTime: null,
    lastStepTime: null,
    currentStep: null,
    stepTimings: {}
};

/**
 * Initialize pipeline timing display for a new query.
 * Resets all timing metrics and shows the timing display container.
 * Should be called at the start of each new query.
 */
export function initPipelineTiming() {
    pipelineTimingTracker = {
        startTime: Date.now(),
        lastStepTime: Date.now(),
        currentStep: null,
        stepTimings: {}
    };

    const container = document.getElementById('timingMetrics');
    if (!container) return;

    // Show the display
    container.classList.add('visible');

    // Reset all steps to initial state
    const steps = ['preprocessing', 'security', 'rules', 'schema', 'generating', 'fixing', 'executing'];
    steps.forEach(step => {
        const el = document.getElementById(`step-${step}`);
        if (el) {
            el.classList.remove('active', 'completed');
            const valueEl = el.querySelector('.timing-metric-value');
            if (valueEl) valueEl.textContent = '--';
        }
    });

    // Reset total
    const totalEl = document.getElementById('timingTotal');
    if (totalEl) totalEl.textContent = '--';
}

/**
 * Update pipeline step timing from SSE event.
 * Marks the previous step as completed and the current step as active.
 * Calculates and displays duration for completed steps.
 *
 * @param {Object} data - SSE progress data from the backend
 * @param {string} data.stage - The current pipeline stage name
 * @param {number} [data.elapsed] - Total elapsed time in seconds
 * @param {string} [data.substep] - Sub-step within the generating stage
 * @param {string} [data.message] - Status message for the current step
 */
export function updatePipelineStep(data) {
    if (!data || !data.stage) return;

    const container = document.getElementById('timingMetrics');
    if (!container) return;
    container.classList.add('visible');

    const stage = data.stage;
    const elapsed = data.elapsed || 0;
    const now = Date.now();

    // Map backend stages to display step IDs
    const stageToStep = {
        'preprocessing': 'preprocessing',
        'security': 'security',
        'cache': 'security',  // Cache is part of security/init phase
        'rules': 'rules',
        'schema': 'schema',
        'generating': 'generating',
        'fixing': 'fixing',
        'validating': 'fixing',  // Validate is part of fixing phase
        'executing': 'executing',
        'complete': 'executing'  // Complete marks end of executing
    };

    const stepId = stageToStep[stage];
    if (!stepId) return;

    // Mark previous step as completed with its duration
    if (pipelineTimingTracker.currentStep && pipelineTimingTracker.currentStep !== stepId) {
        const prevStepEl = document.getElementById(`step-${pipelineTimingTracker.currentStep}`);
        if (prevStepEl) {
            prevStepEl.classList.remove('active', 'waiting');
            prevStepEl.classList.add('completed');

            // Calculate step duration
            const stepDuration = now - pipelineTimingTracker.lastStepTime;
            pipelineTimingTracker.stepTimings[pipelineTimingTracker.currentStep] = stepDuration;

            const valueEl = prevStepEl.querySelector('.timing-metric-value');
            if (valueEl) {
                valueEl.textContent = formatDuration(stepDuration);
                valueEl.className = 'timing-metric-value ' + getSpeedClass(stepDuration, stepId);
            }
        }
    }

    // Mark current step as active
    const currentStepEl = document.getElementById(`step-${stepId}`);
    if (currentStepEl && !currentStepEl.classList.contains('completed')) {
        currentStepEl.classList.add('active');

        // Handle detailed generating step updates with substeps
        if (stepId === 'generating' && data.substep) {
            updateGeneratingStep(data);
        } else if (data.message && stepId === 'generating') {
            // Show substep message if provided (for generating step)
            const valueEl = currentStepEl.querySelector('.timing-metric-value');
            if (valueEl) {
                // Show short status message
                const shortMessage = data.message.length > 12 ?
                    data.message.substring(0, 10) + '...' : data.message;
                valueEl.textContent = shortMessage;
                valueEl.title = data.message;  // Full message on hover
            }
        }
    }

    // Update tracking
    if (pipelineTimingTracker.currentStep !== stepId) {
        pipelineTimingTracker.lastStepTime = now;
        pipelineTimingTracker.currentStep = stepId;
    }

    // Update total elapsed time
    const totalEl = document.getElementById('timingTotal');
    if (totalEl) {
        totalEl.textContent = formatDuration(elapsed * 1000);
    }
}

/**
 * Handle detailed generating event with substep information.
 * Updates the generating step display with status messages and animations.
 *
 * @param {Object} data - Generating event data from SSE
 * @param {string} [data.substep] - The current substep ('prompt', 'llm', 'complete')
 * @param {string} [data.message] - Status message for the substep
 */
export function updateGeneratingStep(data) {
    if (!data) return;

    const stepEl = document.getElementById('step-generating');
    if (!stepEl) return;

    const valueEl = stepEl.querySelector('.timing-metric-value');
    if (!valueEl) return;

    // Show status based on substep
    const substepMessages = {
        'prompt': 'Building...',
        'llm': 'AI thinking...',
        'complete': 'Done!'
    };

    const message = substepMessages[data.substep] || data.message || '--';
    valueEl.textContent = message;
    valueEl.title = data.message || '';

    // Add pulsing animation for LLM waiting
    if (data.substep === 'llm') {
        stepEl.classList.add('waiting');
    } else {
        stepEl.classList.remove('waiting');
    }
}

/**
 * Finalize pipeline timing when the result is received.
 * Marks the last step as completed and updates the total time display.
 *
 * @param {number} [totalMs] - Total processing time in milliseconds.
 *                             If not provided, calculated from start time.
 */
export function finalizePipelineTiming(totalMs) {
    const now = Date.now();

    // Complete the last step
    if (pipelineTimingTracker.currentStep) {
        const lastStepEl = document.getElementById(`step-${pipelineTimingTracker.currentStep}`);
        if (lastStepEl) {
            lastStepEl.classList.remove('active', 'waiting');
            lastStepEl.classList.add('completed');

            const stepDuration = now - pipelineTimingTracker.lastStepTime;
            const valueEl = lastStepEl.querySelector('.timing-metric-value');
            if (valueEl) {
                valueEl.textContent = formatDuration(stepDuration);
                valueEl.className = 'timing-metric-value ' + getSpeedClass(stepDuration, pipelineTimingTracker.currentStep);
            }
        }
    }

    // Update total
    const totalEl = document.getElementById('timingTotal');
    if (totalEl) {
        const total = totalMs || (now - pipelineTimingTracker.startTime);
        totalEl.textContent = formatDuration(total);
        totalEl.className = 'timing-metric-value ' + (total < 3000 ? 'fast' : total < 8000 ? 'medium' : 'slow');
    }
}

/**
 * Format a duration in milliseconds to a human-readable string.
 *
 * @param {number} ms - Duration in milliseconds
 * @returns {string} Formatted duration string (e.g., '<1ms', '250ms', '2.5s')
 */
export function formatDuration(ms) {
    if (ms === undefined || ms === null) return '--';
    if (ms < 1) return '<1ms';
    if (ms < 1000) return `${Math.round(ms)}ms`;
    return `${(ms / 1000).toFixed(1)}s`;
}

/**
 * Get CSS speed class based on step type and duration.
 * Different steps have different performance thresholds.
 *
 * @param {number} ms - Duration in milliseconds
 * @param {string} stepId - The pipeline step ID
 * @returns {string} CSS class name ('fast', 'medium', 'slow', or '')
 */
export function getSpeedClass(ms, stepId) {
    if (ms === undefined || ms === null) return '';

    // Thresholds vary by step type (in ms)
    const thresholds = {
        'preprocessing': { fast: 100, slow: 500 },
        'security': { fast: 100, slow: 500 },
        'rules': { fast: 200, slow: 1000 },
        'schema': { fast: 300, slow: 1000 },
        'generating': { fast: 2000, slow: 8000 },  // LLM is slower
        'fixing': { fast: 100, slow: 500 },
        'executing': { fast: 500, slow: 2000 }
    };

    const t = thresholds[stepId] || { fast: 200, slow: 1000 };
    if (ms <= t.fast) return 'fast';
    if (ms >= t.slow) return 'slow';
    return 'medium';
}

/**
 * Clear the timing display and reset all tracking state.
 * Called when clearing the chat or resetting the application.
 */
export function clearTimingDisplay() {
    const container = document.getElementById('timingMetrics');
    if (container) {
        container.classList.remove('visible');

        // Reset all steps
        const steps = ['preprocessing', 'security', 'rules', 'schema', 'generating', 'fixing', 'executing'];
        steps.forEach(step => {
            const el = document.getElementById(`step-${step}`);
            if (el) {
                el.classList.remove('active', 'completed');
                const valueEl = el.querySelector('.timing-metric-value');
                if (valueEl) valueEl.textContent = '--';
            }
        });

        const totalEl = document.getElementById('timingTotal');
        if (totalEl) totalEl.textContent = '--';
    }

    pipelineTimingTracker = {
        startTime: null,
        lastStepTime: null,
        currentStep: null,
        stepTimings: {}
    };
}

/**
 * Update pipeline timing metrics display (legacy/backward compatibility).
 * Delegates to finalizePipelineTiming for actual display updates.
 *
 * @param {Object} timing - Timing metrics from API response
 * @param {number} [timing.totalMs] - Total processing time in milliseconds
 */
export function updateTimingDisplay(timing) {
    // This function is kept for backward compatibility
    // Real-time timing is now handled by updatePipelineStep and finalizePipelineTiming
    if (timing && timing.totalMs) {
        finalizePipelineTiming(timing.totalMs);
    }
}

/**
 * Get the current pipeline timing tracker state.
 * Useful for debugging or accessing timing data externally.
 *
 * @returns {Object} The current pipelineTimingTracker object
 */
export function getPipelineTimingTracker() {
    return { ...pipelineTimingTracker };
}
