/**
 * Token Usage Display Component
 *
 * A reusable JavaScript component for displaying token usage information
 * from LLM responses. Tracks cumulative usage across conversations
 * and provides visual feedback when approaching context limits.
 *
 * Usage:
 *   const tokenTracker = new TokenUsageTracker({
 *     containerId: 'token-display',
 *     maxContextSize: 8192
 *   });
 *
 *   // After each response:
 *   tokenTracker.update({
 *     promptTokens: 156,
 *     responseTokens: 423
 *   });
 */

class TokenUsageTracker {
    /**
     * Default context window sizes for common LLM models
     * These can be overridden by passing maxContextSize to constructor
     */
    static MODEL_CONTEXT_SIZES = {
        'qwen2.5-coder:1.5b': 32768,
        'qwen2.5-coder:7b': 32768,
        'qwen2.5-coder:14b': 32768,
        'qwen2.5:7b': 32768,
        'qwen2.5:14b': 32768,
        'llama3.2:3b': 128000,
        'llama3.1:8b': 128000,
        'llama3.1:70b': 128000,
        'mistral:7b': 32768,
        'codellama:7b': 16384,
        'codellama:13b': 16384,
        'deepseek-coder:6.7b': 16384,
        'phi3:mini': 4096,
        'gemma2:9b': 8192,
        'default': 8192
    };

    /**
     * Create a new TokenUsageTracker
     * @param {Object} options - Configuration options
     * @param {string} options.containerId - ID of container element to render into
     * @param {number} options.maxContextSize - Maximum context window size in tokens
     * @param {string} options.model - Model name to auto-detect context size
     * @param {boolean} options.showCumulative - Whether to track cumulative usage across messages
     * @param {number} options.warningThreshold - Percentage at which to show warning (default 80)
     */
    constructor(options = {}) {
        this.containerId = options.containerId || 'token-usage-display';
        this.model = options.model || 'default';
        this.maxContextSize = options.maxContextSize || this.getContextSizeForModel(this.model);
        this.showCumulative = options.showCumulative !== false;
        this.warningThreshold = options.warningThreshold || 80;

        // Track token usage
        this.currentPromptTokens = 0;
        this.currentResponseTokens = 0;
        this.cumulativeTokens = 0;
        this.messageCount = 0;

        // Bind methods
        this.update = this.update.bind(this);
        this.reset = this.reset.bind(this);
        this.render = this.render.bind(this);
    }

    /**
     * Get context window size for a given model
     * @param {string} model - Model name
     * @returns {number} Context window size
     */
    getContextSizeForModel(model) {
        // Try exact match first
        if (TokenUsageTracker.MODEL_CONTEXT_SIZES[model]) {
            return TokenUsageTracker.MODEL_CONTEXT_SIZES[model];
        }

        // Try partial match (model family)
        const modelLower = model.toLowerCase();
        for (const [key, size] of Object.entries(TokenUsageTracker.MODEL_CONTEXT_SIZES)) {
            if (modelLower.includes(key.split(':')[0])) {
                return size;
            }
        }

        return TokenUsageTracker.MODEL_CONTEXT_SIZES.default;
    }

    /**
     * Update model and recalculate context size
     * @param {string} model - New model name
     */
    setModel(model) {
        this.model = model;
        this.maxContextSize = this.getContextSizeForModel(model);
        this.render();
    }

    /**
     * Update token usage with new values
     * @param {Object} usage - Token usage data
     * @param {number} usage.promptTokens - Tokens in the prompt
     * @param {number} usage.responseTokens - Tokens in the response
     * @param {number} usage.totalTokens - Total tokens (optional, calculated if not provided)
     */
    update(usage) {
        if (!usage) return;

        this.currentPromptTokens = usage.promptTokens || 0;
        this.currentResponseTokens = usage.responseTokens || 0;

        const total = usage.totalTokens || (this.currentPromptTokens + this.currentResponseTokens);

        if (this.showCumulative) {
            this.cumulativeTokens += total;
            this.messageCount++;
        } else {
            this.cumulativeTokens = total;
        }

        this.render();
    }

    /**
     * Reset all token counters
     */
    reset() {
        this.currentPromptTokens = 0;
        this.currentResponseTokens = 0;
        this.cumulativeTokens = 0;
        this.messageCount = 0;
        this.render();
    }

    /**
     * Calculate usage percentage
     * @returns {number} Percentage of context window used
     */
    getUsagePercentage() {
        return Math.min(100, (this.cumulativeTokens / this.maxContextSize) * 100);
    }

    /**
     * Check if usage is above warning threshold
     * @returns {boolean} True if above warning threshold
     */
    isWarning() {
        return this.getUsagePercentage() >= this.warningThreshold;
    }

    /**
     * Check if usage is critical (above 95%)
     * @returns {boolean} True if above 95%
     */
    isCritical() {
        return this.getUsagePercentage() >= 95;
    }

    /**
     * Format number with thousands separator
     * @param {number} num - Number to format
     * @returns {string} Formatted number
     */
    formatNumber(num) {
        return num.toLocaleString();
    }

    /**
     * Get the appropriate status color
     * @returns {string} CSS color variable or hex color
     */
    getStatusColor() {
        if (this.isCritical()) return '#ef4444'; // Red
        if (this.isWarning()) return '#f59e0b'; // Amber
        return '#10b981'; // Green
    }

    /**
     * Get the progress bar gradient based on usage
     * @returns {string} CSS gradient
     */
    getProgressGradient() {
        const percentage = this.getUsagePercentage();

        if (this.isCritical()) {
            return 'linear-gradient(90deg, #ef4444, #dc2626)';
        } else if (this.isWarning()) {
            return 'linear-gradient(90deg, #f59e0b, #d97706)';
        } else {
            return 'linear-gradient(90deg, #10b981, #059669)';
        }
    }

    /**
     * Render the token usage display
     */
    render() {
        const container = document.getElementById(this.containerId);
        if (!container) {
            console.warn(`TokenUsageTracker: Container with ID '${this.containerId}' not found`);
            return;
        }

        const percentage = this.getUsagePercentage();
        const statusColor = this.getStatusColor();
        const progressGradient = this.getProgressGradient();

        // Build warning message if needed
        let warningHtml = '';
        if (this.isCritical()) {
            warningHtml = `
                <div class="token-warning token-critical">
                    <span class="warning-icon">!!</span>
                    <span>Context limit nearly reached. Consider starting a new conversation.</span>
                </div>
            `;
        } else if (this.isWarning()) {
            warningHtml = `
                <div class="token-warning">
                    <span class="warning-icon">!</span>
                    <span>Approaching context limit (${percentage.toFixed(0)}% used)</span>
                </div>
            `;
        }

        container.innerHTML = `
            <div class="token-usage-container">
                <div class="token-stats">
                    <div class="token-stat">
                        <span class="token-label">Prompt:</span>
                        <span class="token-value">${this.formatNumber(this.currentPromptTokens)}</span>
                    </div>
                    <div class="token-stat-divider">|</div>
                    <div class="token-stat">
                        <span class="token-label">Response:</span>
                        <span class="token-value">${this.formatNumber(this.currentResponseTokens)}</span>
                    </div>
                    <div class="token-stat-divider">|</div>
                    <div class="token-stat">
                        <span class="token-label">Total:</span>
                        <span class="token-value token-total" style="color: ${statusColor}">
                            ${this.formatNumber(this.cumulativeTokens)}
                        </span>
                    </div>
                </div>

                <div class="token-progress-container">
                    <div class="token-progress-bar">
                        <div class="token-progress-fill" style="width: ${percentage}%; background: ${progressGradient}"></div>
                    </div>
                    <div class="token-progress-text">
                        ${this.formatNumber(this.cumulativeTokens)} / ${this.formatNumber(this.maxContextSize)}
                        <span class="token-percentage">(${percentage.toFixed(1)}%)</span>
                    </div>
                </div>

                ${warningHtml}
            </div>
        `;
    }

    /**
     * Create the display container if it doesn't exist
     * @param {string} parentId - ID of parent element to append to
     * @returns {HTMLElement} The created or existing container
     */
    createContainer(parentId) {
        let container = document.getElementById(this.containerId);
        if (!container) {
            container = document.createElement('div');
            container.id = this.containerId;
            container.className = 'token-usage-wrapper';

            const parent = document.getElementById(parentId);
            if (parent) {
                parent.appendChild(container);
            } else {
                console.warn(`TokenUsageTracker: Parent element '${parentId}' not found`);
            }
        }
        return container;
    }

    /**
     * Get current state for debugging or persistence
     * @returns {Object} Current token tracker state
     */
    getState() {
        return {
            model: this.model,
            maxContextSize: this.maxContextSize,
            currentPromptTokens: this.currentPromptTokens,
            currentResponseTokens: this.currentResponseTokens,
            cumulativeTokens: this.cumulativeTokens,
            messageCount: this.messageCount,
            percentage: this.getUsagePercentage(),
            isWarning: this.isWarning(),
            isCritical: this.isCritical()
        };
    }
}

// Export for module systems, or make globally available
if (typeof module !== 'undefined' && module.exports) {
    module.exports = TokenUsageTracker;
} else {
    window.TokenUsageTracker = TokenUsageTracker;
}
