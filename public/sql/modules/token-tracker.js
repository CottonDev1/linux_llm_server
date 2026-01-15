/**
 * Token Tracker Module
 *
 * Manages token usage tracking and context window display for the SQL Chat application.
 * Provides functions to estimate, track, and display token usage across conversations.
 *
 * Features:
 * - Token estimation from text content
 * - Context window usage visualization
 * - Context limit alerts and input disabling
 * - Conversation context management for AI queries
 *
 * @module token-tracker
 */

import { state } from './sql-chat-state.js';

/**
 * Load conversation context setting from localStorage
 * Currently always enables conversation context
 */
export function loadConversationContextSetting() {
    state.conversationContextEnabled = true;
}

/**
 * Toggle conversation context on/off
 * Currently always enables conversation context
 */
export function toggleConversationContext() {
    state.conversationContextEnabled = true;
}

/**
 * Get conversation history formatted for AI context
 * Returns recent messages limited by user settings
 *
 * @returns {Array|null} Array of formatted messages for AI context, or null if disabled/empty
 */
export function getConversationForContext() {
    if (!state.conversationContextEnabled || state.chatHistory.length === 0) {
        return null;
    }

    // Get max messages from admin settings (localStorage) or use default
    const savedMax = localStorage.getItem('sqlMaxConversationMessages');
    const maxMessages = savedMax ? parseInt(savedMax) : 6;

    // Get the last N messages (user questions and assistant responses)
    const recentHistory = state.chatHistory.slice(-maxMessages);

    // Format for the AI
    return recentHistory.map(msg => {
        if (msg.type === 'user') {
            return { role: 'user', content: msg.text };
        } else if (msg.type === 'assistant') {
            // Include SQL if available for context
            let content = msg.text;
            if (msg.sql) {
                content += `\nSQL: ${msg.sql}`;
            }
            return { role: 'assistant', content };
        } else if (msg.type === 'error') {
            return { role: 'assistant', content: `Error: ${msg.text}` };
        }
        return null;
    }).filter(m => m !== null);
}

/**
 * Initialize the context remaining display with default values
 * Sets up the token tracker max tokens from localStorage settings
 */
export function initializeContextRemainingDisplay() {
    const usedValueEl = document.getElementById('contextUsedValue');
    const maxValueEl = document.getElementById('contextMaxValue');

    // Get context window size from settings
    const savedContextWindow = localStorage.getItem('sqlContextWindowSize');
    state.tokenTracker.maxTokens = savedContextWindow ? parseInt(savedContextWindow) : 8192;

    if (usedValueEl) {
        usedValueEl.textContent = '0';
    }
    if (maxValueEl) {
        maxValueEl.textContent = state.tokenTracker.maxTokens.toLocaleString();
    }
}

/**
 * Update the token usage display in the UI
 * Updates context window values, prompt/completion token counts,
 * and applies warning styles when usage exceeds 80%
 */
export function updateTokenDisplay() {
    const usedValueEl = document.getElementById('contextUsedValue');
    const maxValueEl = document.getElementById('contextMaxValue');
    const promptTokensEl = document.getElementById('promptTokens');
    const completionTokensEl = document.getElementById('completionTokens');
    const totalTokensEl = document.getElementById('totalTokens');

    const percentage = (state.tokenTracker.currentTokens / state.tokenTracker.maxTokens) * 100;
    const isWarning = percentage > 80;

    // Update the context window display (used / max format)
    if (usedValueEl) {
        usedValueEl.textContent = state.tokenTracker.currentTokens.toLocaleString();
        usedValueEl.style.color = isWarning ? '#f59e0b' : '#10b981';
    }
    if (maxValueEl) {
        maxValueEl.textContent = state.tokenTracker.maxTokens.toLocaleString();
        maxValueEl.style.color = isWarning ? '#f59e0b' : '#10b981';
    }

    // Update prompt/completion/total token displays
    if (promptTokensEl) {
        promptTokensEl.textContent = state.tokenTracker.promptTokens.toLocaleString();
    }
    if (completionTokensEl) {
        completionTokensEl.textContent = state.tokenTracker.completionTokens.toLocaleString();
    }
    if (totalTokensEl) {
        totalTokensEl.textContent = state.tokenTracker.totalTokens.toLocaleString();
    }

    // Update all labels color based on context usage
    document.querySelectorAll('.context-remaining-label').forEach(label => {
        label.style.color = isWarning ? '#f59e0b' : '';
    });
    document.querySelectorAll('.context-remaining-value').forEach(value => {
        value.style.color = isWarning ? '#f59e0b' : '';
    });

    // Check if limit is reached
    if (percentage >= 100 && !state.tokenTracker.isLimitReached) {
        state.tokenTracker.isLimitReached = true;
        showContextLimitAlert();
        disableChatInput();
    }
}

/**
 * Show the context limit alert banner
 * Called when token usage reaches 100% of the context window
 */
export function showContextLimitAlert() {
    const alert = document.getElementById('contextLimitAlert');
    if (alert) {
        alert.style.display = 'block';
    }
}

/**
 * Hide the context limit alert banner
 * Called when clearing chat or resetting context
 */
export function hideContextLimitAlert() {
    const alert = document.getElementById('contextLimitAlert');
    if (alert) {
        alert.style.display = 'none';
    }
}

/**
 * Disable chat input when context limit is reached
 * Prevents further queries until context is cleared
 */
export function disableChatInput() {
    const chatInput = document.getElementById('chatInput');
    const sendBtn = document.getElementById('sendBtn');

    if (chatInput) {
        chatInput.disabled = true;
        chatInput.placeholder = 'Context window full - click Clear Chat to continue';
    }
    if (sendBtn) {
        sendBtn.disabled = true;
    }
}

/**
 * Estimate token count from text
 * Uses approximate ratio of ~4 characters per token
 *
 * @param {string} text - The text to estimate tokens for
 * @returns {number} Estimated token count
 */
export function estimateTokens(text) {
    if (!text) return 0;
    return Math.ceil(text.length / 4);
}

/**
 * Update token tracker by calculating tokens from conversation history
 * Called after every query to track context usage
 */
export function updateTokenTrackerFromHistory() {
    const savedContextWindow = localStorage.getItem('sqlContextWindowSize');
    state.tokenTracker.maxTokens = savedContextWindow ? parseInt(savedContextWindow) : 8192;

    // Estimate tokens from chat history
    let totalEstimatedTokens = 0;
    state.chatHistory.forEach(msg => {
        const textTokens = estimateTokens(msg.text || '');
        const sqlTokens = msg.sql ? estimateTokens(msg.sql) : 0;
        totalEstimatedTokens += textTokens + sqlTokens;
    });

    state.tokenTracker.currentTokens = totalEstimatedTokens;
    console.log(`[TokenTracker] History: ${state.chatHistory.length} msgs, Estimated tokens: ${totalEstimatedTokens}`);
    updateTokenDisplay();
}

/**
 * Update token tracker from API response
 * Currently delegates to updateTokenTrackerFromHistory for consistency
 *
 * @param {Object} tokenUsage - Token usage data from API (currently unused)
 */
export function updateTokenTracker(tokenUsage) {
    // Always update from history to show context usage
    updateTokenTrackerFromHistory();
}
