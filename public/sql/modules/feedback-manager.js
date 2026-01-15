/**
 * Feedback Manager Module
 *
 * Handles all feedback-related functionality for the SQL Chat application.
 * This includes user feedback submission (positive/negative), feedback review
 * for administrators, and AI-powered rule generation from negative feedback.
 *
 * @module feedback-manager
 */

import { SQL_API_BASE, state } from './sql-chat-state.js';

// ============================================================================
// Module State
// ============================================================================

/**
 * Stores the current feedback context when a user provides negative feedback.
 * This context is used when submitting detailed feedback through the modal.
 */
let currentFeedbackContext = {
    query: null,
    generatedSql: null,
    database: null,
    messageId: null,
    buttonElement: null
};

// ============================================================================
// Utility Functions
// ============================================================================

/**
 * Escape HTML to prevent XSS attacks when rendering user-provided content.
 * Uses DOM methods for safe escaping.
 *
 * @param {string} text - The text to escape
 * @returns {string} HTML-escaped text
 */
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

/**
 * Get the currently selected database from the dropdown.
 *
 * @returns {string|null} The selected database name or null if not found
 */
export function getCurrentDatabase() {
    const databaseSelect = document.getElementById('database');
    return databaseSelect ? databaseSelect.value : null;
}

/**
 * Get the current user's role from localStorage.
 * Defaults to 'user' if no role is found or parsing fails.
 *
 * @returns {string} The user's role (lowercase)
 */
export function getUserRole() {
    const userStr = localStorage.getItem('user');
    if (userStr) {
        try {
            const user = JSON.parse(userStr);
            return user.role ? user.role.toLowerCase() : 'user';
        } catch (e) {
            return 'user';
        }
    }
    return 'user';
}

// ============================================================================
// User Feedback Functions
// ============================================================================

/**
 * Handle positive feedback (thumbs up) from a user.
 * Marks the button as active, deactivates negative feedback, shows toast,
 * and submits positive feedback to the API.
 *
 * @param {string} query - The original natural language query
 * @param {string} sql - The generated SQL
 * @param {string} messageId - The ID of the chat message
 * @param {HTMLElement} buttonElement - The clicked button element
 */
export function handlePositiveFeedback(query, sql, messageId, buttonElement) {
    // Mark button as active
    buttonElement.classList.add('active-positive');

    // Deactivate the thumbs down button
    const thumbsDownBtn = buttonElement.parentElement.querySelector('[data-feedback="negative"]');
    if (thumbsDownBtn) {
        thumbsDownBtn.classList.remove('active-negative');
    }

    // Show toast message
    showFeedbackToast('Thanks for the feedback!');

    // Submit positive feedback to API
    submitFeedback({
        query: query,
        database: getCurrentDatabase(),
        generatedSql: sql,
        isPositive: true,
        reason: null,
        correctedSql: null
    });

    // Log for debugging
    console.log('Positive feedback submitted:', { query, sql, messageId });
}

/**
 * Handle negative feedback (thumbs down) from a user.
 * Marks the button as active, deactivates positive feedback, stores context,
 * and opens the feedback modal for detailed input.
 *
 * @param {string} query - The original natural language query
 * @param {string} sql - The generated SQL
 * @param {string} messageId - The ID of the chat message
 * @param {HTMLElement} buttonElement - The clicked button element
 */
export function handleNegativeFeedback(query, sql, messageId, buttonElement) {
    // Mark button as active
    buttonElement.classList.add('active-negative');

    // Deactivate the thumbs up button
    const thumbsUpBtn = buttonElement.parentElement.querySelector('[data-feedback="positive"]');
    if (thumbsUpBtn) {
        thumbsUpBtn.classList.remove('active-positive');
    }

    // Store context for later use
    currentFeedbackContext = {
        query: query,
        generatedSql: sql,
        database: getCurrentDatabase(),
        messageId: messageId,
        buttonElement: buttonElement
    };

    // Open feedback modal
    openSqlFeedbackModal(query, sql);
}

/**
 * Open the SQL Feedback modal and populate it with query details.
 *
 * @param {string} query - The original natural language query
 * @param {string} generatedSql - The generated SQL to provide feedback on
 */
export function openSqlFeedbackModal(query, generatedSql) {
    const modal = document.getElementById('sqlFeedbackModal');
    if (!modal) return;

    // Populate hidden fields
    document.getElementById('feedbackOriginalQuestion').value = query || '';
    document.getElementById('feedbackGeneratedSql').value = generatedSql || '';
    document.getElementById('feedbackDetails').value = '';

    // Show modal
    modal.classList.remove('hidden');

    // Focus the textarea
    setTimeout(() => {
        document.getElementById('feedbackDetails').focus();
    }, 100);
}

/**
 * Close the SQL Feedback modal and clear the current context.
 *
 * @param {Event} [event] - The click event (optional, used for backdrop clicks)
 */
export function closeSqlFeedbackModal(event) {
    if (event && event.target !== event.currentTarget) return;

    const modal = document.getElementById('sqlFeedbackModal');
    if (modal) {
        modal.classList.add('hidden');
    }

    // Clear current context
    currentFeedbackContext = {
        query: null,
        generatedSql: null,
        database: null,
        messageId: null,
        buttonElement: null
    };
}

/**
 * Save SQL feedback from the modal.
 * Validates the input, prepares feedback data, and submits to the API.
 * Shows success/error messages based on the result.
 */
export async function saveSqlFeedback() {
    const details = document.getElementById('feedbackDetails').value.trim();

    if (!details) {
        alert('Please describe what went wrong with the query.');
        return;
    }

    // Prepare feedback data
    const feedbackData = {
        query: currentFeedbackContext.query,
        database: currentFeedbackContext.database,
        generatedSql: currentFeedbackContext.generatedSql,
        isPositive: false,
        reason: details
    };

    // Submit to API
    const success = await submitFeedback(feedbackData);

    if (success) {
        showFeedbackToast('Thanks for your feedback!');
        closeSqlFeedbackModal();
    } else {
        alert('Failed to save feedback. Please try again.');
    }
}

/**
 * Submit feedback to the API endpoint.
 *
 * @param {Object} feedbackData - The feedback data to submit
 * @param {string} feedbackData.query - The original question
 * @param {string} feedbackData.database - The target database
 * @param {string} feedbackData.generatedSql - The generated SQL
 * @param {boolean} feedbackData.isPositive - Whether this is positive feedback
 * @param {string|null} feedbackData.reason - The reason for negative feedback
 * @param {string|null} [feedbackData.correctedSql] - User-provided corrected SQL
 * @returns {Promise<boolean>} True if submission was successful, false otherwise
 */
export async function submitFeedback(feedbackData) {
    try {
        const response = await fetch(`${SQL_API_BASE}/api/sql/feedback`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(feedbackData)
        });

        if (!response.ok) {
            console.error('Failed to submit feedback:', await response.text());
            return false;
        }

        const result = await response.json();
        console.log('Feedback submitted successfully:', result);
        return true;
    } catch (error) {
        console.error('Error submitting feedback:', error);
        return false;
    }
}

/**
 * Show a temporary toast notification message.
 * The toast fades out and is removed after 3 seconds.
 *
 * @param {string} message - The message to display in the toast
 */
export function showFeedbackToast(message) {
    // Create toast element
    const toast = document.createElement('div');
    toast.className = 'feedback-toast';
    toast.textContent = message;

    // Add to page
    document.body.appendChild(toast);

    // Remove after 3 seconds
    setTimeout(() => {
        toast.style.opacity = '0';
        setTimeout(() => {
            toast.remove();
        }, 300);
    }, 3000);
}

// ============================================================================
// Feedback Review System (Admin)
// ============================================================================

/**
 * Show the feedback review modal and load feedback data.
 * Uses the <ewr-modal> component's open() method.
 */
export function showFeedbackReviewModal() {
    const modal = document.getElementById('feedbackReviewModal');
    if (modal && modal.open) {
        modal.open();
        loadFeedbackList();
    }
}

/**
 * Close the feedback review modal.
 * Uses the <ewr-modal> component's close() method.
 *
 * @param {Event} [event] - The click event (optional)
 */
export function closeFeedbackReviewModal(event) {
    const modal = document.getElementById('feedbackReviewModal');
    if (modal && modal.close) {
        modal.close();
    }
}

/**
 * Load feedback list from the API with current filter settings.
 * Updates the feedback list UI and statistics display.
 */
export async function loadFeedbackList() {
    const feedbackList = document.getElementById('feedbackList');
    const feedbackStats = document.getElementById('feedbackStats');
    const typeFilter = document.getElementById('feedbackTypeFilter').value;
    const processedFilter = document.getElementById('feedbackProcessedFilter').value;

    feedbackList.innerHTML = '<div style="text-align: center; color: #64748b; padding: 40px;">Loading feedback...</div>';

    try {
        // Build query params
        const params = new URLSearchParams();
        if (typeFilter) params.append('feedback_type', typeFilter);
        if (processedFilter !== '') params.append('processed', processedFilter);
        params.append('limit', '50');

        const response = await fetch(`${SQL_API_BASE}/api/sql/feedback?${params.toString()}`);
        if (!response.ok) {
            throw new Error('Failed to load feedback');
        }

        const data = await response.json();

        // Update stats
        if (feedbackStats) {
            feedbackStats.textContent = `${data.unprocessed} unprocessed / ${data.total} total`;
        }

        // Update badge
        updateFeedbackBadge(data.unprocessed);

        // Render feedback items
        if (data.feedback && data.feedback.length > 0) {
            feedbackList.innerHTML = data.feedback.map(item => renderFeedbackItem(item)).join('');
        } else {
            feedbackList.innerHTML = '<div style="text-align: center; color: #64748b; padding: 40px;">No feedback found matching filters.</div>';
        }

    } catch (error) {
        console.error('Error loading feedback:', error);
        feedbackList.innerHTML = `<div style="text-align: center; color: #ef4444; padding: 40px;">Error loading feedback: ${error.message}</div>`;
    }
}

/**
 * Update the feedback badge count in the UI.
 * Hides the badge if count is 0, shows "99+" for counts over 99.
 *
 * @param {number} count - The number of unprocessed feedback items
 */
export function updateFeedbackBadge(count) {
    const badge = document.getElementById('feedbackBadge');
    if (badge) {
        if (count > 0) {
            badge.textContent = count > 99 ? '99+' : count;
            badge.style.display = 'inline-flex';
        } else {
            badge.style.display = 'none';
        }
    }
}

/**
 * Render a single feedback item as HTML.
 * Creates the card layout with feedback details, SQL preview, and action buttons.
 *
 * @param {Object} item - The feedback item to render
 * @param {string} item.id - Unique feedback ID
 * @param {string} item.feedback - 'positive' or 'negative'
 * @param {string} item.question - The original question
 * @param {string} item.sql - The generated SQL
 * @param {string} item.database - The target database
 * @param {string} [item.comment] - User's feedback comment
 * @param {boolean} item.processed - Whether the feedback has been processed
 * @param {string} [item.rule_created] - ID of rule created from this feedback
 * @param {string} item.created_at - ISO timestamp of when feedback was created
 * @returns {string} HTML string for the feedback item
 */
export function renderFeedbackItem(item) {
    const isNegative = item.feedback === 'negative';
    const feedbackIcon = isNegative ? '👎' : '👍';
    const feedbackColor = isNegative ? '#ef4444' : '#10b981';
    const processedBadge = item.processed
        ? '<span style="background: rgba(16, 185, 129, 0.2); color: #10b981; padding: 2px 8px; border-radius: 4px; font-size: 11px; margin-left: 8px;">✓ Processed</span>'
        : '';
    const ruleBadge = item.rule_created
        ? `<span style="background: rgba(59, 130, 246, 0.2); color: #3b82f6; padding: 2px 8px; border-radius: 4px; font-size: 11px; margin-left: 8px;">Rule: ${item.rule_created}</span>`
        : '';

    return `
        <div style="background: #1e293b; border: 1px solid #3f5175; border-radius: 8px; padding: 16px; ${item.processed ? 'opacity: 0.7;' : ''}">
            <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 12px;">
                <div style="flex: 1;">
                    <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 4px;">
                        <span style="font-size: 18px;">${feedbackIcon}</span>
                        <span style="color: ${feedbackColor}; font-weight: 600; font-size: 13px;">${item.feedback.toUpperCase()}</span>
                        <span style="color: #64748b; font-size: 12px;">${item.database}</span>
                        ${processedBadge}
                        ${ruleBadge}
                    </div>
                    <div style="color: #cbd5e1; font-size: 14px; margin-bottom: 8px;">${escapeHtml(item.question)}</div>
                    ${item.comment ? `<div style="color: #94a3b8; font-size: 13px; font-style: italic; background: rgba(139, 92, 246, 0.1); padding: 8px 12px; border-radius: 4px; border-left: 3px solid #a78bfa;">"${escapeHtml(item.comment)}"</div>` : ''}
                </div>
                <div style="color: #64748b; font-size: 11px; white-space: nowrap; margin-left: 16px;">
                    ${new Date(item.created_at).toLocaleString()}
                </div>
            </div>

            <!-- SQL Preview -->
            <details style="margin-bottom: 12px;">
                <summary style="color: #60a5fa; font-size: 12px; cursor: pointer; user-select: none;">Show Generated SQL</summary>
                <pre style="background: #0f172a; padding: 12px; border-radius: 4px; margin-top: 8px; color: #94a3b8; font-size: 12px; font-family: monospace; overflow-x: auto; white-space: pre-wrap;">${escapeHtml(item.sql)}</pre>
            </details>

            <!-- Action Buttons -->
            <div style="display: flex; gap: 8px; flex-wrap: wrap;">
                ${isNegative && !item.processed ? `
                    <button onclick="generateRuleFromFeedback('${item.id}')" class="btn" style="padding: 6px 12px; font-size: 12px; background: linear-gradient(135deg, #8b5cf6 0%, #3b82f6 100%);">
                        ✨ Generate Rule with AI
                    </button>
                ` : ''}
                ${!item.processed ? `
                    <button onclick="markFeedbackProcessed('${item.id}')" class="btn" style="padding: 6px 12px; font-size: 12px; background: #475569;">
                        ✓ Mark Processed
                    </button>
                ` : ''}
                ${item.processed && !item.rule_created ? `
                    <button onclick="markFeedbackProcessed('${item.id}', false)" class="btn" style="padding: 6px 12px; font-size: 12px; background: #475569;">
                        ↺ Reopen
                    </button>
                ` : ''}
            </div>
        </div>
    `;
}

// ============================================================================
// Rule Generation from Feedback
// ============================================================================

/**
 * Generate a SQL rule from feedback using AI.
 * Calls the API to analyze the feedback and generate a suggested rule,
 * then displays it in the rule preview modal for review.
 *
 * @param {string} feedbackId - The ID of the feedback to generate a rule from
 */
export async function generateRuleFromFeedback(feedbackId) {
    // Find the button and show loading state
    const btn = event.target;
    const originalText = btn.innerHTML;
    btn.innerHTML = '⏳ Generating...';
    btn.disabled = true;

    try {
        const response = await fetch(`${SQL_API_BASE}/api/sql/feedback/${feedbackId}/generate-rule`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            }
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Failed to generate rule');
        }

        const data = await response.json();

        if (data.success && data.rule) {
            // Show the rule preview modal
            showRulePreviewModal(data.rule, data.feedback, feedbackId);
        } else {
            throw new Error('No rule data returned');
        }

    } catch (error) {
        console.error('Error generating rule:', error);
        alert(`Failed to generate rule: ${error.message}`);
    } finally {
        btn.innerHTML = originalText;
        btn.disabled = false;
    }
}

/**
 * Mark feedback as processed or unprocessed.
 * Updates the feedback status in the database and refreshes the list.
 *
 * @param {string} feedbackId - The ID of the feedback to update
 * @param {boolean} [processed=true] - Whether to mark as processed or unprocessed
 */
export async function markFeedbackProcessed(feedbackId, processed = true) {
    try {
        const response = await fetch(`${SQL_API_BASE}/api/sql/feedback/${feedbackId}`, {
            method: 'PATCH',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ processed: processed })
        });

        if (!response.ok) {
            throw new Error('Failed to update feedback');
        }

        // Reload the list
        loadFeedbackList();
        showFeedbackToast(processed ? 'Feedback marked as processed' : 'Feedback reopened');

    } catch (error) {
        console.error('Error updating feedback:', error);
        alert(`Failed to update feedback: ${error.message}`);
    }
}

// ============================================================================
// Rule Preview Modal
// ============================================================================

/**
 * Show the rule preview modal with generated rule data.
 * Populates all fields with the AI-generated rule for user review and editing.
 *
 * @param {Object} rule - The generated rule data
 * @param {string} rule.description - Rule description
 * @param {string} rule.type - Rule type ('constraint' or 'assistance')
 * @param {string|string[]} rule.trigger_keywords - Keywords that trigger the rule
 * @param {string} rule.rule_text - The rule text/guidance
 * @param {string} [rule.auto_fix_pattern] - Regex pattern for auto-fix
 * @param {string} [rule.auto_fix_replacement] - Replacement text for auto-fix
 * @param {string} [rule.example_question] - Example question for exact match
 * @param {string} [rule.example_sql] - Example SQL for exact match
 * @param {string} [rule.database] - Target database for the rule
 * @param {Object} feedback - The original feedback data
 * @param {string} feedback.question - The original question
 * @param {string} [feedback.comment] - User's feedback comment
 * @param {string} feedbackId - The ID of the source feedback
 */
export function showRulePreviewModal(rule, feedback, feedbackId) {
    // Populate source feedback info
    document.getElementById('rulePreviewQuestion').textContent = feedback?.question || '';
    document.getElementById('rulePreviewComment').textContent = feedback?.comment ? `"${feedback.comment}"` : '';

    // Populate rule fields
    document.getElementById('previewRuleDescription').value = rule.description || '';
    document.getElementById('previewRuleType').value = rule.type || 'constraint';
    document.getElementById('previewRuleKeywords').value = Array.isArray(rule.trigger_keywords)
        ? rule.trigger_keywords.join(', ')
        : (rule.trigger_keywords || '');
    document.getElementById('previewRuleText').value = rule.rule_text || '';

    // Auto-fix section
    const autoFixSection = document.getElementById('previewAutoFixSection');
    if (rule.auto_fix_pattern || rule.auto_fix_replacement) {
        document.getElementById('previewAutoFixFind').value = rule.auto_fix_pattern || '';
        document.getElementById('previewAutoFixReplace').value = rule.auto_fix_replacement || '';
        autoFixSection.style.display = 'block';
    } else {
        autoFixSection.style.display = 'none';
    }

    // Example section
    const exampleSection = document.getElementById('previewExampleSection');
    if (rule.example_question || rule.example_sql) {
        document.getElementById('previewExampleQuestion').value = rule.example_question || '';
        document.getElementById('previewExampleSql').value = rule.example_sql || '';
        exampleSection.style.display = 'block';
    } else {
        exampleSection.style.display = 'none';
    }

    // Store metadata
    document.getElementById('previewFeedbackId').value = feedbackId;
    document.getElementById('previewDatabase').value = rule.database || getCurrentDatabase() || '_global';

    // Show modal
    const modal = document.getElementById('rulePreviewModal');
    if (modal) {
        modal.classList.remove('hidden');
    }
}

/**
 * Close the rule preview modal.
 *
 * @param {Event} [event] - The click event (optional, used for backdrop clicks)
 */
export function closeRulePreviewModal(event) {
    if (event && event.target !== event.currentTarget) return;
    const modal = document.getElementById('rulePreviewModal');
    if (modal) {
        modal.classList.add('hidden');
    }
}

/**
 * Create a rule from the preview modal data.
 * Validates required fields, builds the rule object, submits to the API,
 * and marks the source feedback as processed with the new rule ID.
 */
export async function createRuleFromPreview() {
    const feedbackId = document.getElementById('previewFeedbackId').value;
    const database = document.getElementById('previewDatabase').value || '_global';
    const description = document.getElementById('previewRuleDescription').value.trim();
    const ruleType = document.getElementById('previewRuleType').value;
    const keywords = document.getElementById('previewRuleKeywords').value;
    const ruleText = document.getElementById('previewRuleText').value.trim();

    if (!description || !ruleText) {
        alert('Description and Rule Text are required.');
        return;
    }

    // Generate rule ID from description
    const ruleId = description.toLowerCase()
        .replace(/[^a-z0-9\s]/g, '')
        .replace(/\s+/g, '-')
        .substring(0, 50) + '-' + Date.now().toString(36);

    // Build rule object
    const ruleData = {
        database: database,
        rule_id: ruleId,
        description: description,
        type: ruleType,
        priority: 'normal',
        enabled: true,
        trigger_keywords: keywords.split(',').map(k => k.trim()).filter(k => k),
        trigger_tables: [],
        trigger_columns: [],
        rule_text: ruleText
    };

    // Add auto-fix if present
    const autoFixFind = document.getElementById('previewAutoFixFind').value.trim();
    const autoFixReplace = document.getElementById('previewAutoFixReplace').value.trim();
    if (autoFixFind && autoFixReplace) {
        ruleData.auto_fix_pattern = autoFixFind;
        ruleData.auto_fix_replacement = autoFixReplace;
    }

    // Add example if present
    const exampleQuestion = document.getElementById('previewExampleQuestion').value.trim();
    const exampleSql = document.getElementById('previewExampleSql').value.trim();
    if (exampleQuestion && exampleSql) {
        ruleData.example_question = exampleQuestion;
        ruleData.example_sql = exampleSql;
    }

    try {
        // Create the rule
        const response = await fetch(`${SQL_API_BASE}/api/sql/rules`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(ruleData)
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Failed to create rule');
        }

        const result = await response.json();

        // Mark feedback as processed with rule ID
        if (feedbackId) {
            await fetch(`${SQL_API_BASE}/api/sql/feedback/${feedbackId}`, {
                method: 'PATCH',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    processed: true,
                    rule_created: ruleId
                })
            });
        }

        // Close modals and refresh
        closeRulePreviewModal();
        loadFeedbackList();
        showFeedbackToast(`Rule "${ruleId}" created successfully!`);

    } catch (error) {
        console.error('Error creating rule:', error);
        alert(`Failed to create rule: ${error.message}`);
    }
}

// ============================================================================
// Initialization
// ============================================================================

/**
 * Load the feedback count on page load for the badge display.
 * Called with a delay to allow other initialization to complete.
 */
export async function loadFeedbackCount() {
    try {
        const response = await fetch(`${SQL_API_BASE}/api/sql/feedback?processed=false&limit=1`);
        if (response.ok) {
            const data = await response.json();
            updateFeedbackBadge(data.unprocessed || 0);
        }
    } catch (error) {
        console.warn('Could not load feedback count:', error);
    }
}

/**
 * Initialize the feedback manager module.
 * Sets up the DOMContentLoaded listener to load the feedback count.
 */
export function initializeFeedbackManager() {
    // Delay to let other init complete
    setTimeout(loadFeedbackCount, 2000);
}

// Auto-initialize when DOM is ready
if (typeof document !== 'undefined') {
    document.addEventListener('DOMContentLoaded', () => {
        initializeFeedbackManager();
    });
}
