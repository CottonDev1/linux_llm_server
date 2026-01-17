/**
 * Feedback Widget Component
 * Reusable component for collecting user feedback on search results
 *
 * Usage:
 * 1. Include this script in your page
 * 2. Create a container element with id="feedback-widget"
 * 3. Call FeedbackWidget.init(options) with your configuration
 *
 * Example:
 * FeedbackWidget.init({
 *     query: "Show me all orders",
 *     database: "CentralData",
 *     documentIds: ["doc1", "doc2"],
 *     onSubmit: (feedback) => console.log('Feedback submitted:', feedback)
 * });
 */

const FeedbackWidget = (function() {
    'use strict';

    const PYTHON_API = '/api/python';

    // CSS styles for the widget
    const STYLES = `
        .feedback-widget {
            background: var(--card-bg, #1a1a2e);
            border: 1px solid var(--card-border, #2a2a4a);
            border-radius: 8px;
            padding: 16px;
            margin-top: 16px;
            font-family: inherit;
        }

        .feedback-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 12px;
        }

        .feedback-title {
            font-size: 14px;
            font-weight: 600;
            color: var(--text-primary, #e0e0e0);
        }

        .feedback-close {
            background: none;
            border: none;
            color: var(--text-muted, #888);
            cursor: pointer;
            font-size: 18px;
            padding: 0;
            line-height: 1;
        }

        .feedback-close:hover {
            color: var(--text-primary, #e0e0e0);
        }

        .feedback-question {
            font-size: 13px;
            color: var(--text-secondary, #b0b0b0);
            margin-bottom: 12px;
        }

        .feedback-rating {
            display: flex;
            gap: 8px;
            margin-bottom: 16px;
        }

        .feedback-rating-btn {
            display: flex;
            align-items: center;
            gap: 6px;
            padding: 8px 16px;
            border: 1px solid var(--card-border, #2a2a4a);
            border-radius: 6px;
            background: transparent;
            color: var(--text-secondary, #b0b0b0);
            cursor: pointer;
            font-size: 13px;
            transition: all 0.2s;
        }

        .feedback-rating-btn:hover {
            border-color: var(--accent-cyan, #00d4ff);
            color: var(--accent-cyan, #00d4ff);
        }

        .feedback-rating-btn.selected {
            background: var(--accent-cyan, #00d4ff);
            color: #000;
            border-color: var(--accent-cyan, #00d4ff);
        }

        .feedback-rating-btn.selected.negative {
            background: #ef4444;
            border-color: #ef4444;
            color: #fff;
        }

        .feedback-rating-btn svg {
            width: 16px;
            height: 16px;
            fill: currentColor;
        }

        .feedback-stars {
            display: flex;
            gap: 4px;
        }

        .feedback-star {
            background: none;
            border: none;
            cursor: pointer;
            padding: 4px;
            color: var(--text-muted, #888);
            transition: color 0.2s;
        }

        .feedback-star:hover,
        .feedback-star.active {
            color: #f59e0b;
        }

        .feedback-star svg {
            width: 24px;
            height: 24px;
            fill: currentColor;
        }

        .feedback-correction {
            margin-top: 16px;
            display: none;
        }

        .feedback-correction.visible {
            display: block;
        }

        .feedback-correction-label {
            font-size: 12px;
            color: var(--text-muted, #888);
            margin-bottom: 8px;
        }

        .feedback-correction-input {
            width: 100%;
            background: var(--input-bg, #0d0d1a);
            border: 1px solid var(--card-border, #2a2a4a);
            border-radius: 6px;
            padding: 10px 12px;
            color: var(--text-primary, #e0e0e0);
            font-family: monospace;
            font-size: 12px;
            resize: vertical;
            min-height: 80px;
        }

        .feedback-correction-input:focus {
            outline: none;
            border-color: var(--accent-cyan, #00d4ff);
        }

        .feedback-comment {
            margin-top: 12px;
        }

        .feedback-comment-input {
            width: 100%;
            background: var(--input-bg, #0d0d1a);
            border: 1px solid var(--card-border, #2a2a4a);
            border-radius: 6px;
            padding: 10px 12px;
            color: var(--text-primary, #e0e0e0);
            font-size: 13px;
            resize: vertical;
            min-height: 60px;
        }

        .feedback-comment-input:focus {
            outline: none;
            border-color: var(--accent-cyan, #00d4ff);
        }

        .feedback-actions {
            display: flex;
            justify-content: flex-end;
            gap: 8px;
            margin-top: 16px;
        }

        .feedback-btn {
            padding: 8px 16px;
            border-radius: 6px;
            font-size: 13px;
            font-weight: 500;
            cursor: pointer;
            transition: all 0.2s;
        }

        .feedback-btn-secondary {
            background: transparent;
            border: 1px solid var(--card-border, #2a2a4a);
            color: var(--text-secondary, #b0b0b0);
        }

        .feedback-btn-secondary:hover {
            border-color: var(--text-muted, #888);
            color: var(--text-primary, #e0e0e0);
        }

        .feedback-btn-primary {
            background: var(--accent-cyan, #00d4ff);
            border: 1px solid var(--accent-cyan, #00d4ff);
            color: #000;
        }

        .feedback-btn-primary:hover {
            background: #22d3ee;
            border-color: #22d3ee;
        }

        .feedback-btn-primary:disabled {
            opacity: 0.5;
            cursor: not-allowed;
        }

        .feedback-success {
            text-align: center;
            padding: 20px;
            color: #22c55e;
        }

        .feedback-success svg {
            width: 48px;
            height: 48px;
            fill: currentColor;
            margin-bottom: 8px;
        }

        .feedback-success-text {
            font-size: 14px;
            font-weight: 500;
        }

        .feedback-inline {
            display: flex;
            align-items: center;
            gap: 12px;
            padding: 8px 0;
        }

        .feedback-inline-label {
            font-size: 12px;
            color: var(--text-muted, #888);
        }

        .feedback-inline-btns {
            display: flex;
            gap: 6px;
        }

        .feedback-inline-btn {
            display: flex;
            align-items: center;
            justify-content: center;
            width: 32px;
            height: 32px;
            border: 1px solid var(--card-border, #2a2a4a);
            border-radius: 6px;
            background: transparent;
            color: var(--text-muted, #888);
            cursor: pointer;
            transition: all 0.2s;
        }

        .feedback-inline-btn:hover {
            border-color: var(--accent-cyan, #00d4ff);
            color: var(--accent-cyan, #00d4ff);
        }

        .feedback-inline-btn.selected.positive {
            background: rgba(34, 197, 94, 0.2);
            border-color: #22c55e;
            color: #22c55e;
        }

        .feedback-inline-btn.selected.negative {
            background: rgba(239, 68, 68, 0.2);
            border-color: #ef4444;
            color: #ef4444;
        }

        .feedback-inline-btn svg {
            width: 16px;
            height: 16px;
            fill: currentColor;
        }

        /* Enhanced Detailed Feedback Form Styles */
        .feedback-detailed-form {
            margin-top: 16px;
            padding: 16px;
            background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
            border: 1px solid #334155;
            border-radius: 8px;
            border-left: 3px solid #ef4444;
        }

        .feedback-detailed-form.hidden {
            display: none;
        }

        .feedback-form-title {
            font-size: 14px;
            font-weight: 600;
            color: #fca5a5;
            margin-bottom: 16px;
            display: flex;
            align-items: center;
            gap: 8px;
        }

        .feedback-form-group {
            margin-bottom: 16px;
        }

        .feedback-form-group:last-child {
            margin-bottom: 0;
        }

        .feedback-form-label {
            display: block;
            font-size: 13px;
            font-weight: 500;
            color: #cbd5e1;
            margin-bottom: 6px;
        }

        .feedback-form-select {
            width: 100%;
            background: #0f172a;
            border: 1px solid #334155;
            border-radius: 6px;
            padding: 10px 12px;
            color: #f1f5f9;
            font-size: 13px;
            cursor: pointer;
            transition: border-color 0.2s;
        }

        .feedback-form-select:focus {
            outline: none;
            border-color: #3b82f6;
        }

        .feedback-form-textarea {
            width: 100%;
            background: #0f172a;
            border: 1px solid #334155;
            border-radius: 6px;
            padding: 10px 12px;
            color: #f1f5f9;
            font-size: 13px;
            resize: vertical;
            min-height: 70px;
            transition: border-color 0.2s;
        }

        .feedback-form-textarea:focus {
            outline: none;
            border-color: #3b82f6;
        }

        .feedback-form-textarea.code-input {
            font-family: 'Courier New', monospace;
            font-size: 12px;
            min-height: 90px;
        }

        .feedback-form-hint {
            font-size: 11px;
            color: #64748b;
            margin-top: 4px;
        }

        .feedback-form-actions {
            display: flex;
            justify-content: flex-end;
            gap: 10px;
            margin-top: 16px;
            padding-top: 16px;
            border-top: 1px solid #334155;
        }

        .feedback-form-btn {
            padding: 10px 18px;
            border-radius: 6px;
            font-size: 13px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.2s;
            border: none;
        }

        .feedback-form-btn-cancel {
            background: #374151;
            color: #9ca3af;
        }

        .feedback-form-btn-cancel:hover {
            background: #4b5563;
            color: #e5e7eb;
        }

        .feedback-form-btn-submit {
            background: #3b82f6;
            color: white;
        }

        .feedback-form-btn-submit:hover {
            background: #2563eb;
        }

        .feedback-form-btn-submit:disabled {
            opacity: 0.5;
            cursor: not-allowed;
        }

        /* Document checkbox styles */
        .feedback-docs-list {
            display: flex;
            flex-direction: column;
            gap: 8px;
            max-height: 200px;
            overflow-y: auto;
            padding: 4px;
        }

        .feedback-doc-checkbox {
            display: flex;
            align-items: flex-start;
            gap: 10px;
            padding: 10px 12px;
            background: #0f172a;
            border: 1px solid #334155;
            border-radius: 6px;
            cursor: pointer;
            transition: all 0.2s;
        }

        .feedback-doc-checkbox:hover {
            border-color: #475569;
            background: #1e293b;
        }

        .feedback-doc-checkbox input[type="checkbox"] {
            width: 18px;
            height: 18px;
            margin-top: 2px;
            accent-color: #3b82f6;
            cursor: pointer;
        }

        .feedback-doc-checkbox input[type="checkbox"]:checked + .feedback-doc-info {
            color: #f1f5f9;
        }

        .feedback-doc-info {
            display: flex;
            flex-direction: column;
            gap: 2px;
            flex: 1;
            min-width: 0;
        }

        .feedback-doc-title {
            font-size: 13px;
            font-weight: 500;
            color: #cbd5e1;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }

        .feedback-doc-meta {
            font-size: 11px;
            color: #64748b;
        }

        .feedback-doc-checkbox:has(input:checked) {
            border-color: #3b82f6;
            background: rgba(59, 130, 246, 0.1);
        }

        .feedback-doc-checkbox:has(input:checked) .feedback-doc-title {
            color: #93c5fd;
        }

        /* Document category tags */
        .feedback-doc-tags {
            display: flex;
            flex-wrap: wrap;
            gap: 4px;
            margin-top: 4px;
        }

        .feedback-doc-tag {
            display: inline-block;
            padding: 2px 6px;
            border-radius: 3px;
            font-size: 10px;
            font-weight: 500;
            text-transform: uppercase;
            letter-spacing: 0.3px;
        }

        .tag-department {
            background: rgba(139, 92, 246, 0.2);
            color: #a78bfa;
            border: 1px solid rgba(139, 92, 246, 0.3);
        }

        .tag-type {
            background: rgba(59, 130, 246, 0.2);
            color: #60a5fa;
            border: 1px solid rgba(59, 130, 246, 0.3);
        }

        .tag-subject {
            background: rgba(16, 185, 129, 0.2);
            color: #34d399;
            border: 1px solid rgba(16, 185, 129, 0.3);
        }
    `;

    // Inject styles
    function injectStyles() {
        if (document.getElementById('feedback-widget-styles')) return;
        const style = document.createElement('style');
        style.id = 'feedback-widget-styles';
        style.textContent = STYLES;
        document.head.appendChild(style);
    }

    // Icons using ewr-icon component (auto-loads Lucide)
    const ICONS = {
        thumbUp: '<ewr-icon name="thumbs-up" size="20"></ewr-icon>',
        thumbDown: '<ewr-icon name="thumbs-down" size="20"></ewr-icon>',
        star: '<ewr-icon name="star" size="20" color="#fbbf24"></ewr-icon>',
        starEmpty: '<ewr-icon name="star" size="20" stroke-width="1.5"></ewr-icon>',
        check: '<ewr-icon name="check-circle" size="20" color="#10b981"></ewr-icon>'
    };

    // Store instances by containerId for multi-instance support
    const instances = {};

    // Get or create instance state
    function getInstance(containerId) {
        if (!instances[containerId]) {
            instances[containerId] = {
                config: {},
                state: {
                    isHelpful: null,
                    rating: 0,
                    correction: '',
                    comment: '',
                    submitted: false
                }
            };
        }
        return instances[containerId];
    }

    // Legacy state for backwards compatibility (default instance)
    let config = {};
    let state = {
        isHelpful: null,
        rating: 0,
        correction: '',
        comment: '',
        submitted: false
    };

    /**
     * Initialize the feedback widget
     * @param {Object} options Configuration options
     * @param {string} options.query The original query
     * @param {string} options.response The response that feedback is about
     * @param {string} options.database Database name (optional)
     * @param {string[]} options.documentIds IDs of documents shown in results
     * @param {string} options.generatedSql Generated SQL (optional, for SQL queries)
     * @param {string} options.containerId ID of container element (default: 'feedback-widget')
     * @param {string} options.mode 'full' or 'inline' (default: 'full')
     * @param {string} options.context 'sql' or 'knowledge_base' (default: auto-detect based on generatedSql)
     * @param {Object} options.metadata Additional metadata (retrieval scores, conversation history, etc.)
     * @param {function} options.onSubmit Callback after successful submission
     */
    function init(options) {
        injectStyles();

        const containerId = options.containerId || 'feedback-widget';

        // Get or create instance for this container
        const instance = getInstance(containerId);

        // Auto-detect context based on whether SQL was generated
        const context = options.context || (options.generatedSql ? 'sql' : 'knowledge_base');

        instance.config = {
            query: options.query || '',
            response: options.response || '',
            database: options.database || null,
            documentIds: options.documentIds || [],
            generatedSql: options.generatedSql || null,
            containerId: containerId,
            mode: options.mode || 'full',
            context: context,
            metadata: options.metadata || {},
            onSubmit: options.onSubmit || function() {}
        };

        instance.state = {
            isHelpful: null,
            rating: 0,
            correction: '',
            expectedAnswer: '',
            comment: '',
            errorType: '',
            submitted: false
        };

        // Also update legacy state for backwards compatibility
        config = instance.config;
        state = instance.state;

        render(containerId);
    }

    /**
     * Render the widget
     */
    function render(containerId) {
        // Get instance or use legacy config
        const cid = containerId || config.containerId;
        const instance = instances[cid] || { config, state };
        const instanceConfig = instance.config;
        const instanceState = instance.state;

        const container = document.getElementById(cid);
        if (!container) {
            console.error('Feedback widget container not found:', cid);
            return;
        }

        if (instanceState.submitted) {
            container.innerHTML = renderSuccess();
            return;
        }

        if (instanceConfig.mode === 'inline') {
            container.innerHTML = renderInline(cid);
        } else {
            container.innerHTML = renderFull(cid);
        }

        attachEventListeners(cid);
    }

    /**
     * Render full feedback form
     */
    function renderFull(containerId) {
        const instance = instances[containerId] || { config, state };
        const instanceConfig = instance.config;
        const instanceState = instance.state;

        return `
            <div class="feedback-widget" data-container-id="${containerId}">
                <div class="feedback-header">
                    <span class="feedback-title">Was this result helpful?</span>
                </div>

                <div class="feedback-rating">
                    <button type="button" class="feedback-rating-btn ${instanceState.isHelpful === true ? 'selected' : ''}" data-helpful="true">
                        ${ICONS.thumbUp}
                        <span>Yes, helpful</span>
                    </button>
                    <button type="button" class="feedback-rating-btn ${instanceState.isHelpful === false ? 'selected negative' : ''}" data-helpful="false">
                        ${ICONS.thumbDown}
                        <span>Not helpful</span>
                    </button>
                </div>

                <div class="feedback-question">Rate the quality (optional):</div>
                <div class="feedback-stars">
                    ${[1, 2, 3, 4, 5].map(i => `
                        <button type="button" class="feedback-star ${i <= instanceState.rating ? 'active' : ''}" data-rating="${i}">
                            ${i <= instanceState.rating ? ICONS.star : ICONS.starEmpty}
                        </button>
                    `).join('')}
                </div>

                ${instanceConfig.generatedSql ? `
                    <div class="feedback-correction ${instanceState.isHelpful === false ? 'visible' : ''}">
                        <div class="feedback-correction-label">Provide the correct SQL (optional):</div>
                        <textarea class="feedback-correction-input" placeholder="Enter the correct SQL query...">${instanceState.correction}</textarea>
                    </div>
                ` : ''}

                <div class="feedback-comment">
                    <textarea class="feedback-comment-input" placeholder="Additional comments (optional)...">${instanceState.comment}</textarea>
                </div>

                <div class="feedback-actions">
                    <button type="button" class="feedback-btn feedback-btn-secondary" onclick="FeedbackWidget.dismiss('${containerId}')">Skip</button>
                    <button type="button" class="feedback-btn feedback-btn-primary" onclick="FeedbackWidget.submit('${containerId}')" ${instanceState.isHelpful === null ? 'disabled' : ''}>
                        Submit Feedback
                    </button>
                </div>
            </div>
        `;
    }

    /**
     * Render inline (compact) feedback
     */
    function renderInline(containerId) {
        const instance = instances[containerId] || { config, state };
        const instanceConfig = instance.config;
        const instanceState = instance.state;

        // Check if detailed form should be shown
        const showDetailedForm = instanceState.isHelpful === false && !instanceState.submitted;

        return `
            <div class="feedback-inline" data-container-id="${containerId}">
                <span class="feedback-inline-label">Was this helpful?</span>
                <div class="feedback-inline-btns">
                    <button type="button" class="feedback-inline-btn ${instanceState.isHelpful === true ? 'selected positive' : ''}" data-helpful="true" title="Yes, helpful">
                        ${ICONS.thumbUp}
                    </button>
                    <button type="button" class="feedback-inline-btn ${instanceState.isHelpful === false ? 'selected negative' : ''}" data-helpful="false" title="Not helpful">
                        ${ICONS.thumbDown}
                    </button>
                </div>
            </div>
            ${showDetailedForm ? renderDetailedFeedbackForm(containerId) : ''}
        `;
    }

    /**
     * Issue categories for structured feedback.
     * Each category routes to different improvement mechanisms.
     */
    const ISSUE_CATEGORIES = {
        sql: [
            { value: 'wrong_columns', label: 'Wrong columns selected', action: 'schema_improvement' },
            { value: 'wrong_tables', label: 'Wrong table(s) used', action: 'schema_improvement' },
            { value: 'wrong_join', label: 'Incorrect join or relationship', action: 'schema_improvement' },
            { value: 'wrong_filter', label: 'Wrong WHERE/filter conditions', action: 'prompt_improvement' },
            { value: 'wrong_aggregation', label: 'Incorrect GROUP BY or aggregation', action: 'prompt_improvement' },
            { value: 'syntax_error', label: 'SQL syntax error', action: 'prompt_improvement' },
            { value: 'missing_data', label: 'Query correct but data missing', action: 'none' },
            { value: 'performance', label: 'Query too slow', action: 'optimization' }
        ],
        knowledge_base: [
            { value: 'irrelevant_documents', label: 'Retrieved irrelevant documents', action: 'retrieval_improvement', showDocs: true },
            { value: 'missing_documents', label: 'Missing relevant documents', action: 'retrieval_improvement', showDocs: false },
            { value: 'misinterpreted_sources', label: 'Misinterpreted the sources', action: 'synthesis_improvement', showDocs: true },
            { value: 'document_content_wrong', label: 'Document content is incorrect', action: 'document_quality', showDocs: true },
            { value: 'document_outdated', label: 'Document content is outdated', action: 'document_quality', showDocs: true },
            { value: 'hallucination', label: 'Made up facts not in sources', action: 'synthesis_improvement', showDocs: false }
        ]
    };

    /**
     * Render detailed feedback form for negative feedback
     */
    function renderDetailedFeedbackForm(containerId) {
        const instance = instances[containerId] || { config, state };
        const instanceConfig = instance.config;
        const instanceState = instance.state;
        const isSqlContext = instanceConfig.context === 'sql';
        const categories = ISSUE_CATEGORIES[instanceConfig.context] || ISSUE_CATEGORIES.knowledge_base;

        return `
            <div class="feedback-detailed-form" data-container-id="${containerId}">
                <div class="feedback-form-title">
                    <span>📝</span>
                    <span>Help us improve - What went wrong?</span>
                </div>

                <div class="feedback-form-group">
                    <label class="feedback-form-label">Issue Type <span style="color: #ef4444;">*</span></label>
                    <select class="feedback-form-select feedback-error-type" onchange="FeedbackWidget.onIssueTypeChange('${containerId}')">
                        <option value="">Select issue type...</option>
                        ${categories.map(cat => `<option value="${cat.value}" data-action="${cat.action}" data-show-docs="${cat.showDocs || false}">${cat.label}</option>`).join('')}
                    </select>
                </div>

                <!-- Document flagging section (shown conditionally) -->
                <div class="feedback-form-group feedback-docs-section" style="display: none;">
                    <label class="feedback-form-label feedback-docs-label">Select the problematic document(s):</label>
                    <div class="feedback-docs-list">
                        ${renderDocumentCheckboxes(instanceConfig)}
                    </div>
                </div>

                ${isSqlContext ? `
                <div class="feedback-form-group feedback-sql-section" style="display: none;">
                    <label class="feedback-form-label">Correct SQL (optional)</label>
                    <textarea class="feedback-form-textarea code-input feedback-correct-sql"
                        placeholder="Paste the correct SQL query here...">${instanceState.correction || ''}</textarea>
                </div>
                ` : ''}

                <div class="feedback-form-actions">
                    <button type="button" class="feedback-form-btn feedback-form-btn-cancel" onclick="FeedbackWidget.cancelDetailedForm('${containerId}')">
                        Cancel
                    </button>
                    <button type="button" class="feedback-form-btn feedback-form-btn-submit" onclick="FeedbackWidget.submitDetailedFeedback('${containerId}')" disabled>
                        Submit Feedback
                    </button>
                </div>
            </div>
        `;
    }

    /**
     * Render document checkboxes for flagging with category tags
     */
    function renderDocumentCheckboxes(instanceConfig) {
        const retrievalScores = instanceConfig.metadata?.retrieval_scores || [];

        if (retrievalScores.length === 0) {
            return '<div style="color: #64748b; font-size: 12px; padding: 8px;">No documents to display</div>';
        }

        return retrievalScores.map((doc, index) => {
            // Build category tags
            const tags = [];
            if (doc.department) {
                tags.push(`<span class="feedback-doc-tag tag-department" title="Department">${escapeHtml(formatCategoryName(doc.department))}</span>`);
            }
            if (doc.type) {
                tags.push(`<span class="feedback-doc-tag tag-type" title="Document Type">${escapeHtml(formatCategoryName(doc.type))}</span>`);
            }
            if (doc.subject) {
                tags.push(`<span class="feedback-doc-tag tag-subject" title="Subject">${escapeHtml(formatCategoryName(doc.subject))}</span>`);
            }

            const tagsHtml = tags.length > 0
                ? `<div class="feedback-doc-tags">${tags.join('')}</div>`
                : '';

            return `
                <label class="feedback-doc-checkbox">
                    <input type="checkbox" name="flagged_doc" value="${doc.id}" data-index="${index}">
                    <span class="feedback-doc-info">
                        <span class="feedback-doc-title">${escapeHtml(doc.title || 'Untitled')}</span>
                        ${tagsHtml}
                        <span class="feedback-doc-meta">relevance: ${(doc.score || 0).toFixed(2)}</span>
                    </span>
                </label>
            `;
        }).join('');
    }

    /**
     * Format category name for display (convert snake_case/kebab-case to Title Case)
     */
    function formatCategoryName(name) {
        if (!name) return '';
        return name
            .replace(/[_-]/g, ' ')
            .replace(/\b\w/g, c => c.toUpperCase());
    }

    /**
     * Escape HTML for safe rendering
     */
    function escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text || '';
        return div.innerHTML;
    }

    /**
     * Handle issue type change - show/hide relevant sections
     */
    function onIssueTypeChange(containerId) {
        const container = document.getElementById(containerId);
        if (!container) return;

        const select = container.querySelector('.feedback-error-type');
        const selectedOption = select.options[select.selectedIndex];
        const showDocs = selectedOption?.dataset?.showDocs === 'true';
        const action = selectedOption?.dataset?.action || '';
        const issueType = select.value;

        // Show/hide document section
        const docsSection = container.querySelector('.feedback-docs-section');
        if (docsSection) {
            docsSection.style.display = showDocs ? 'block' : 'none';

            // Update label based on issue type
            const docsLabel = container.querySelector('.feedback-docs-label');
            if (docsLabel) {
                if (issueType === 'irrelevant_documents') {
                    docsLabel.textContent = 'Which documents were NOT relevant to your question?';
                } else if (issueType === 'misinterpreted_sources') {
                    docsLabel.textContent = 'Which document(s) were misinterpreted?';
                } else if (issueType === 'document_content_wrong' || issueType === 'document_outdated') {
                    docsLabel.textContent = 'Which document(s) have incorrect/outdated content?';
                } else {
                    docsLabel.textContent = 'Select the problematic document(s):';
                }
            }
        }

        // Show/hide SQL correction section
        const sqlSection = container.querySelector('.feedback-sql-section');
        if (sqlSection) {
            const showSql = ['wrong_columns', 'wrong_tables', 'wrong_join', 'wrong_filter', 'wrong_aggregation', 'syntax_error'].includes(issueType);
            sqlSection.style.display = showSql ? 'block' : 'none';
        }

        // Enable/disable submit button
        const submitBtn = container.querySelector('.feedback-form-btn-submit');
        if (submitBtn) {
            submitBtn.disabled = !issueType;
        }
    }

    /**
     * Render success message
     */
    function renderSuccess() {
        return `
            <div class="feedback-widget">
                <div class="feedback-success">
                    ${ICONS.check}
                    <div class="feedback-success-text">Thank you for your feedback!</div>
                </div>
            </div>
        `;
    }

    /**
     * Attach event listeners
     */
    function attachEventListeners(containerId) {
        const cid = containerId || config.containerId;
        const container = document.getElementById(cid);
        if (!container) return;

        const instance = instances[cid] || { config, state };
        const instanceConfig = instance.config;
        const instanceState = instance.state;

        // Helpful buttons
        container.querySelectorAll('[data-helpful]').forEach(btn => {
            btn.addEventListener('click', function() {
                const isHelpful = this.dataset.helpful === 'true';
                instanceState.isHelpful = isHelpful;

                // For inline mode with positive feedback, auto-submit
                // For negative feedback, show detailed form first
                if (instanceConfig.mode === 'inline') {
                    if (isHelpful) {
                        submit(cid);
                    } else {
                        // Show detailed form for negative feedback
                        render(cid);
                    }
                } else {
                    render(cid);
                }
            });
        });

        // Star rating
        container.querySelectorAll('[data-rating]').forEach(btn => {
            btn.addEventListener('click', function() {
                instanceState.rating = parseInt(this.dataset.rating);
                render(cid);
            });
        });

        // Correction input
        const correctionInput = container.querySelector('.feedback-correction-input');
        if (correctionInput) {
            correctionInput.addEventListener('input', function() {
                instanceState.correction = this.value;
            });
        }

        // Comment input
        const commentInput = container.querySelector('.feedback-comment-input');
        if (commentInput) {
            commentInput.addEventListener('input', function() {
                instanceState.comment = this.value;
            });
        }
    }

    /**
     * Submit feedback to the API
     */
    async function submit(containerId) {
        const cid = containerId || config.containerId;
        const instance = instances[cid] || { config, state };
        const instanceConfig = instance.config;
        const instanceState = instance.state;

        if (instanceState.isHelpful === null) {
            console.warn('No feedback selected');
            return;
        }

        const feedbackData = {
            query: instanceConfig.query,
            response: instanceConfig.response,
            database: instanceConfig.database,
            document_ids: instanceConfig.documentIds,
            feedback_type: instanceState.correction ? 'correction' : 'rating',
            metadata: {
                ...instanceConfig.metadata,
                context: instanceConfig.context
            }
        };

        // Add rating data
        if (instanceState.rating > 0 || instanceState.isHelpful !== null) {
            feedbackData.rating = {
                is_helpful: instanceState.isHelpful,
                rating: instanceState.rating || (instanceState.isHelpful ? 5 : 1),
                comment: instanceState.comment || null
            };
        }

        // Add correction data if provided
        if (instanceState.correction) {
            feedbackData.correction = {
                original_response: instanceConfig.generatedSql || instanceConfig.response,
                corrected_response: instanceState.correction,
                error_type: instanceState.errorType || 'user_correction'
            };
        }

        try {
            const response = await fetch(`${PYTHON_API}/feedback`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(feedbackData)
            });

            if (!response.ok) {
                throw new Error('Failed to submit feedback');
            }

            const result = await response.json();
            instanceState.submitted = true;
            render(cid);

            if (typeof instanceConfig.onSubmit === 'function') {
                instanceConfig.onSubmit(result);
            }

            // Auto-hide after success
            setTimeout(() => {
                const container = document.getElementById(cid);
                if (container) {
                    container.innerHTML = '';
                }
                // Clean up instance
                delete instances[cid];
            }, 2000);

        } catch (error) {
            console.error('Failed to submit feedback:', error);
            // Show error in UI instead of alert for better UX
            const container = document.getElementById(cid);
            if (container) {
                const errorEl = document.createElement('div');
                errorEl.style.cssText = 'color: #ef4444; font-size: 12px; margin-top: 8px;';
                errorEl.textContent = 'Failed to submit feedback. Please try again.';
                container.appendChild(errorEl);
            }
        }
    }

    /**
     * Dismiss the widget
     */
    function dismiss(containerId) {
        const cid = containerId || config.containerId;
        const container = document.getElementById(cid);
        if (container) {
            container.innerHTML = '';
        }
        // Clean up instance
        delete instances[cid];
    }

    /**
     * Cancel the detailed feedback form
     */
    function cancelDetailedForm(containerId) {
        const cid = containerId || config.containerId;
        const instance = instances[cid];
        if (instance) {
            // Reset to neutral state
            instance.state.isHelpful = null;
            render(cid);
        }
    }

    /**
     * Submit detailed feedback from the negative feedback form
     */
    async function submitDetailedFeedback(containerId) {
        const cid = containerId || config.containerId;
        const instance = instances[cid];
        if (!instance) return;

        const container = document.getElementById(cid);
        if (!container) return;

        const instanceConfig = instance.config;
        const instanceState = instance.state;
        const isSqlContext = instanceConfig.context === 'sql';

        // Gather form values
        const errorTypeSelect = container.querySelector('.feedback-error-type');
        const selectedOption = errorTypeSelect?.options[errorTypeSelect.selectedIndex];
        const correctSqlInput = container.querySelector('.feedback-correct-sql');

        const issueType = errorTypeSelect?.value || '';
        const action = selectedOption?.dataset?.action || '';
        const correctSql = correctSqlInput?.value || '';

        // Gather flagged documents
        const flaggedDocCheckboxes = container.querySelectorAll('input[name="flagged_doc"]:checked');
        const flaggedDocuments = Array.from(flaggedDocCheckboxes).map(cb => ({
            id: cb.value,
            index: parseInt(cb.dataset.index, 10)
        }));

        // Get full document info for flagged docs
        const retrievalScores = instanceConfig.metadata?.retrieval_scores || [];
        const flaggedDocumentsFull = flaggedDocuments.map(fd => {
            const doc = retrievalScores[fd.index];
            return doc ? { ...doc, flagged: true } : { id: fd.id, flagged: true };
        });

        // Store in state
        instanceState.errorType = issueType;
        instanceState.correction = correctSql;

        // Build structured feedback data
        const feedbackData = {
            query: instanceConfig.query,
            response: instanceConfig.response,
            database: instanceConfig.database,
            document_ids: instanceConfig.documentIds,
            feedback_type: 'rating',
            rating: {
                is_helpful: false,
                rating: 1,
                comment: null
            },
            metadata: {
                context: instanceConfig.context,
                issue_type: issueType,
                action: action,
                flagged_documents: flaggedDocumentsFull,
                all_retrieved_documents: retrievalScores,
                conversation_history: instanceConfig.metadata?.conversation_history || [],
                filters: instanceConfig.metadata?.filters || {}
            }
        };

        // Route feedback based on action type
        // This determines what gets updated in the backend
        if (action === 'document_quality') {
            // Only document_content_wrong and document_outdated affect document quality
            feedbackData.metadata.affects_document_quality = true;
            feedbackData.metadata.documents_to_penalize = flaggedDocumentsFull.map(d => d.id);
        } else if (action === 'retrieval_improvement') {
            // irrelevant_documents and missing_documents - track for retrieval tuning
            feedbackData.metadata.affects_document_quality = false;
            feedbackData.metadata.retrieval_issue = true;
            if (issueType === 'irrelevant_documents') {
                feedbackData.metadata.irrelevant_document_ids = flaggedDocumentsFull.map(d => d.id);
            }
        } else if (action === 'synthesis_improvement') {
            // misinterpreted_sources and hallucination - track for prompt tuning
            feedbackData.metadata.affects_document_quality = false;
            feedbackData.metadata.synthesis_issue = true;
            if (issueType === 'misinterpreted_sources') {
                feedbackData.metadata.misinterpreted_document_ids = flaggedDocumentsFull.map(d => d.id);
            }
        }

        // Add SQL correction if provided (for SQL context)
        if (isSqlContext && correctSql && correctSql.trim()) {
            feedbackData.feedback_type = 'correction';
            feedbackData.correction = {
                original_response: instanceConfig.generatedSql || instanceConfig.response,
                corrected_response: correctSql.trim(),
                error_type: issueType
            };
        }

        // Disable submit button during submission
        const submitBtn = container.querySelector('.feedback-form-btn-submit');
        if (submitBtn) {
            submitBtn.disabled = true;
            submitBtn.textContent = 'Submitting...';
        }

        try {
            const response = await fetch(`${PYTHON_API}/feedback`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(feedbackData)
            });

            if (!response.ok) {
                throw new Error('Failed to submit feedback');
            }

            const result = await response.json();
            instanceState.submitted = true;
            render(cid);

            if (typeof instanceConfig.onSubmit === 'function') {
                instanceConfig.onSubmit(result);
            }

            // Auto-hide after success
            setTimeout(() => {
                const container = document.getElementById(cid);
                if (container) {
                    container.innerHTML = '';
                }
                delete instances[cid];
            }, 2000);

        } catch (error) {
            console.error('Failed to submit detailed feedback:', error);
            // Re-enable submit button
            if (submitBtn) {
                submitBtn.disabled = false;
                submitBtn.textContent = 'Submit Feedback';
            }
            // Show error
            const errorEl = document.createElement('div');
            errorEl.style.cssText = 'color: #ef4444; font-size: 12px; margin-top: 8px; text-align: center;';
            errorEl.textContent = 'Failed to submit feedback. Please try again.';
            const actionsEl = container.querySelector('.feedback-form-actions');
            if (actionsEl) {
                actionsEl.parentNode.insertBefore(errorEl, actionsEl);
            }
        }
    }

    /**
     * Quick feedback helper - creates a simple inline widget
     * @param {HTMLElement} parentElement Element to append widget to
     * @param {Object} options Same as init options
     */
    function createInline(parentElement, options) {
        const widgetId = 'feedback-inline-' + Date.now();
        const wrapper = document.createElement('div');
        wrapper.id = widgetId;
        parentElement.appendChild(wrapper);

        init({
            ...options,
            containerId: widgetId,
            mode: 'inline'
        });
    }

    // Public API
    return {
        init,
        submit,
        dismiss,
        createInline,
        cancelDetailedForm,
        submitDetailedFeedback,
        onIssueTypeChange
    };
})();

// Make available globally
if (typeof window !== 'undefined') {
    window.FeedbackWidget = FeedbackWidget;
}
