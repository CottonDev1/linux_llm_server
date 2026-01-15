/**
 * SQL Chat State Module
 *
 * Central state management for the SQL Chat application.
 * Contains all shared state, constants, and configuration used across modules.
 *
 * @module sql-chat-state
 */

// API Configuration - Python service for SQL pipeline
export const SQL_API_BASE = 'http://localhost:8001';

// Pipeline steps configuration - matches backend stages
export const PIPELINE_STEPS = [
    { id: 'preprocessing', label: 'Analyzing question', icon: '1' },
    { id: 'security', label: 'Checking security', icon: '2' },
    { id: 'cache', label: 'Checking cache', icon: '3' },
    { id: 'rules', label: 'Rules', icon: '★', isRulesStep: true },
    { id: 'schema', label: 'Loading schema', icon: '5' },
    { id: 'generating', label: 'Generating SQL', icon: '6' },
    { id: 'fixing', label: 'Applying fixes', icon: '7' },
    { id: 'column_check', label: 'Checking columns', icon: '8' },
    { id: 'validating', label: 'Security check', icon: '9' },
    { id: 'executing', label: 'Executing query', icon: '10' },
    { id: 'complete', label: 'Complete', icon: '✓' }
];

/**
 * Central application state object
 * Contains all mutable state used across the SQL Chat application
 */
export const state = {
    // Chat state
    chatHistory: [],
    currentSchema: null,
    isProcessing: false,
    conversationContextEnabled: true,

    // Application state
    app: {
        connectionTested: false,
        databaseLoaded: false,
        currentDatabase: null,
        schemaStats: null
    },

    // Session metrics tracking
    sessionMetrics: {
        totalQueries: 0,
        successfulQueries: 0,
        totalRows: 0,
        totalTokens: 0,
        responseTimes: [],
        currentQueryStartTime: null
    },

    // Token usage tracker
    tokenTracker: {
        currentTokens: 0,
        maxTokens: 8192,
        isLimitReached: false,
        promptTokens: 0,
        completionTokens: 0,
        totalTokens: 0
    },

    // Pipeline timing state
    pipelineTiming: {
        startTime: null,
        lastStepTime: null,
        currentStep: null,
        stepTimings: {}
    }
};

/**
 * Reset session metrics to initial state
 * Called when starting a new session or clearing metrics
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
}

/**
 * Reset pipeline timing state
 * Called at the start of each new query
 */
export function resetPipelineTiming() {
    state.pipelineTiming = {
        startTime: null,
        lastStepTime: null,
        currentStep: null,
        stepTimings: {}
    };
}

/**
 * Reset application state to initial values
 * Called when disconnecting or resetting the application
 */
export function resetAppState() {
    state.app = {
        connectionTested: false,
        databaseLoaded: false,
        currentDatabase: null,
        schemaStats: null
    };
}

/**
 * Reset token tracker to initial state
 * Called when clearing context or starting fresh
 */
export function resetTokenTracker() {
    state.tokenTracker = {
        currentTokens: 0,
        maxTokens: 8192,
        isLimitReached: false,
        promptTokens: 0,
        completionTokens: 0,
        totalTokens: 0
    };
}

/**
 * Clear chat history
 * Resets the chat history array to empty
 */
export function clearChatHistory() {
    state.chatHistory = [];
}

/**
 * Add a message to chat history
 * @param {Object} message - The message object to add
 */
export function addToChatHistory(message) {
    state.chatHistory.push(message);
}

/**
 * Update token tracker with new values
 * @param {Object} tokenData - Object containing token counts
 */
export function updateTokenTracker(tokenData) {
    if (tokenData.promptTokens !== undefined) {
        state.tokenTracker.promptTokens = tokenData.promptTokens;
    }
    if (tokenData.completionTokens !== undefined) {
        state.tokenTracker.completionTokens = tokenData.completionTokens;
    }
    if (tokenData.totalTokens !== undefined) {
        state.tokenTracker.totalTokens = tokenData.totalTokens;
        state.tokenTracker.currentTokens = tokenData.totalTokens;
    }
    if (tokenData.maxTokens !== undefined) {
        state.tokenTracker.maxTokens = tokenData.maxTokens;
    }
    state.tokenTracker.isLimitReached = state.tokenTracker.currentTokens >= state.tokenTracker.maxTokens;
}

/**
 * Start pipeline timing for a new query
 */
export function startPipelineTiming() {
    const now = Date.now();
    state.pipelineTiming.startTime = now;
    state.pipelineTiming.lastStepTime = now;
    state.pipelineTiming.stepTimings = {};
}

/**
 * Record timing for a pipeline step
 * @param {string} stepId - The ID of the pipeline step
 */
export function recordStepTiming(stepId) {
    const now = Date.now();
    if (state.pipelineTiming.lastStepTime) {
        state.pipelineTiming.stepTimings[stepId] = now - state.pipelineTiming.lastStepTime;
    }
    state.pipelineTiming.lastStepTime = now;
    state.pipelineTiming.currentStep = stepId;
}

/**
 * Get total pipeline duration
 * @returns {number|null} Total duration in milliseconds or null if not started
 */
export function getPipelineDuration() {
    if (!state.pipelineTiming.startTime) return null;
    return Date.now() - state.pipelineTiming.startTime;
}
