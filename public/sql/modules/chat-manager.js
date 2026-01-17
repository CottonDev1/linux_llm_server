/**
 * Chat Manager Module
 *
 * Handles all chat/message-related functionality for the SQL Chat application.
 * Manages sending messages, displaying user/assistant messages, error handling,
 * and loading states during query processing.
 *
 * @module chat-manager
 */

import { SQL_API_BASE, PIPELINE_STEPS, state, resetSessionMetrics, resetTokenTracker, clearChatHistory, addToChatHistory } from './sql-chat-state.js';
import { getConnectionConfig } from './connection-manager.js';
import { scrollToBottom, escapeHtml, updateSendButton, autoResizeTextarea } from './ui-manager.js';
import { updateTokenTrackerFromHistory, updateTokenDisplay, hideContextLimitAlert } from './token-tracker.js';
import { createSqlBlock, createResultsTable } from './results-display.js';
import { initPipelineTiming, updatePipelineStep, finalizePipelineTiming, clearTimingDisplay, getPipelineTimingTracker, formatDuration } from './pipeline-timing.js';
import { updateSessionMetrics, displaySessionMetrics } from './session-metrics.js';
import { handlePositiveFeedback, handleNegativeFeedback } from './feedback-manager.js';

/**
 * Send a sample query by populating the input and triggering send
 * @param {string} query - The sample query text to send
 */
export function sendSampleQuery(query) {
    document.getElementById('chatInput').value = query;
    sendMessage();
}

/**
 * Handle keydown events in the chat input
 * Sends message on Enter, allows new line on Shift+Enter
 * @param {KeyboardEvent} event - The keyboard event
 */
export function handleKeyDown(event) {
    // Send on Enter, new line on Shift+Enter
    if (event.key === 'Enter' && !event.shiftKey) {
        event.preventDefault();
        sendMessage();
    }
}

/**
 * Get conversation history formatted for AI context
 * Returns recent messages to provide context for the next query
 * @returns {Array|null} Formatted conversation history or null if disabled/empty
 */
function getConversationForContext() {
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
 * Main message sending function
 * Handles the complete flow of sending a natural language query to the backend,
 * processing the streaming response, and displaying results
 */
export async function sendMessage() {
    const input = document.getElementById('chatInput');
    const message = input.value.trim();

    if (!message || state.isProcessing) {
        return;
    }

    // Collapse status messages when sending a chat message (function defined in HTML)
    if (typeof window.collapseStatusMessage === 'function') {
        window.collapseStatusMessage();
    }

    // Check if database is loaded
    if (!state.app.databaseLoaded) {
        alert('Please connect to a server and load a database first');
        return;
    }

    const database = document.getElementById('database').value;
    if (!database) {
        alert('Please select a database first');
        return;
    }

    // Clear input and hide empty state
    input.value = '';
    autoResizeTextarea(input);
    const emptyChat = document.getElementById('emptyChat');
    if (emptyChat) {
        emptyChat.classList.add('hidden');
    }

    // Add user message to chat
    addUserMessage(message);

    // Show loading indicator
    const loadingId = addLoadingMessage();

    // Disable send button
    state.isProcessing = true;
    updateSendButton();

    const connectionConfig = getConnectionConfig();

    // Get conversation context if enabled
    const conversationHistory = getConversationForContext();

    // Get max tokens from admin settings (was contextWindowSize)
    const savedMaxTokens = localStorage.getItem('sqlMaxTokens') || localStorage.getItem('sqlContextWindowSize');
    const maxTokens = savedMaxTokens ? Math.min(parseInt(savedMaxTokens), 2048) : 512;

    try {
        // Build credentials object for backend
        const credentials = {
            server: connectionConfig.server,
            database: connectionConfig.database,
            username: connectionConfig.user,
            password: connectionConfig.password,
            use_windows_auth: connectionConfig.authType === 'windows',
            integrated_auth: connectionConfig.integratedAuth || false,
            domain: connectionConfig.domain || null
        };

        // Initialize pipeline timing display
        initPipelineTiming();

        // Use streaming endpoint for real-time progress updates
        const response = await fetch(`${SQL_API_BASE}/api/sql/query-stream`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                naturalLanguage: message,
                database: connectionConfig.database,
                server: connectionConfig.server,
                credentials: credentials,
                conversationHistory: conversationHistory,
                maxTokens: maxTokens,
                options: {
                    execute_sql: true,  // Actually run the query
                    include_schema: true,
                    use_cache: true,
                    max_results: 100
                }
            })
        });

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let result = null;
        let buffer = '';

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            buffer += decoder.decode(value, { stream: true });
            const lines = buffer.split('\n');
            buffer = lines.pop() || ''; // Keep incomplete line in buffer

            for (const line of lines) {
                // Log all SSE lines for debugging
                if (line.trim()) {
                    console.log('[SSE] Line:', line.substring(0, 100));
                }

                if (line.startsWith('data: ')) {
                    try {
                        const data = JSON.parse(line.slice(6));
                        console.log('[SSE] Parsed:', { type: data.type, stage: data.stage, msg: data.message?.substring(0, 40) });

                        if (data.type === 'progress') {
                            // Update loading message - pass data object for direct stage matching
                            console.log('[SSE] Progress stage:', data.stage);
                            updateLoadingMessage(loadingId, data.message || '', data);
                            // Update pipeline timing display
                            updatePipelineStep(data);
                        } else if (data.type === 'result') {
                            console.log('[SSE] Got result, success:', data.success);
                            result = data;
                            // Finalize pipeline timing
                            finalizePipelineTiming(data.processing_time ? data.processing_time * 1000 : null);
                        } else if (data.type === 'error') {
                            console.log('[SSE] Got error:', data.error);
                            // Finalize pipeline timing on error
                            finalizePipelineTiming();
                            // Handle error events
                            result = {
                                success: false,
                                error: data.error || data.message || 'Unknown error'
                            };
                        }
                    } catch (e) {
                        // Ignore parse errors for incomplete chunks
                        console.log('SSE parse error:', e, 'line:', line);
                    }
                }
            }
        }

        // Remove loading indicator
        removeLoadingMessage(loadingId);

        if (result) {
            // Update session metrics
            updateSessionMetrics(result);

            if (result.success) {
                // Add assistant response with results
                addAssistantMessage(result, message);

                // Update token usage display
                if (result.tokenUsage) {
                    updateTokenTrackerFromHistory(result.tokenUsage);
                }

                // Update timing metrics display
                if (result.timing) {
                    // Note: updateTimingDisplay is in pipeline-timing module
                    // updateTimingDisplay(result.timing);
                }
            } else {
                // Add error message with stage info for timeouts
                let errorMsg = result.error || 'Query failed';
                if (result.isTimeout && result.stage) {
                    errorMsg = `Timeout during "${result.stage}"\n\n${result.details || ''}`;
                } else if (result.details) {
                    errorMsg = `${errorMsg}: ${result.details}`;
                }
                addErrorMessage(errorMsg, result.generatedSql, message, result.aiExplanation);
            }
        } else {
            // Update metrics for failed query
            updateSessionMetrics({ success: false });
            addErrorMessage('No response received from server');
        }
    } catch (error) {
        removeLoadingMessage(loadingId);
        // Update metrics for error
        updateSessionMetrics({ success: false });
        addErrorMessage(`Error executing query: ${error.message}`);
    } finally {
        state.isProcessing = false;
        updateSendButton();
    }
}

/**
 * Add a user message to the chat display
 * @param {string} text - The user's message text
 */
export function addUserMessage(text) {
    const chatMessages = document.getElementById('chatMessages');

    const messageDiv = document.createElement('div');
    messageDiv.className = 'chat-message user';
    messageDiv.textContent = text;

    chatMessages.appendChild(messageDiv);
    scrollToBottom();

    // Add to history
    const time = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    addToChatHistory({ type: 'user', text, time });
    updateTokenTrackerFromHistory();
}

/**
 * Build a timing metrics display HTML for inclusion in messages
 * @param {Object} timing - Timing data from result or tracker
 * @returns {string} HTML string for timing metrics display
 */
function buildTimingMetricsHtml(timing) {
    const tracker = getPipelineTimingTracker();
    const stepTimings = tracker.stepTimings || {};

    // If no timing data, return empty
    if (Object.keys(stepTimings).length === 0 && !timing?.processing_time) {
        return '';
    }

    const steps = [
        { id: 'preprocessing', label: 'Analyze' },
        { id: 'security', label: 'Security' },
        { id: 'rules', label: 'Rules' },
        { id: 'schema', label: 'Schema' },
        { id: 'generating', label: 'Generate' },
        { id: 'fixing', label: 'Fix' },
        { id: 'executing', label: 'Execute' }
    ];

    let metricsHtml = '<div class="message-timing-metrics">';

    steps.forEach((step, index) => {
        const duration = stepTimings[step.id];
        const value = duration !== undefined ? formatDuration(duration) : '--';
        const statusClass = duration !== undefined ? 'completed' : '';

        if (index > 0) {
            metricsHtml += '<span class="timing-sep">›</span>';
        }

        metricsHtml += `
            <span class="timing-step ${statusClass}">
                <span class="timing-label">${step.label}</span>
                <span class="timing-value">${value}</span>
            </span>
        `;
    });

    // Add total time
    const totalMs = timing?.processing_time ? timing.processing_time * 1000 :
                    (tracker.startTime ? Date.now() - tracker.startTime : null);
    if (totalMs) {
        metricsHtml += `
            <span class="timing-total-sep">|</span>
            <span class="timing-step timing-total-step">
                <span class="timing-label">Total</span>
                <span class="timing-value">${formatDuration(totalMs)}</span>
            </span>
        `;
    }

    metricsHtml += '</div>';
    return metricsHtml;
}

/**
 * Add an assistant message to the chat display with query results
 * @param {Object} result - The query result object from the backend
 * @param {string} originalQuery - The original natural language query
 */
export function addAssistantMessage(result, originalQuery) {
    const chatMessages = document.getElementById('chatMessages');

    const messageDiv = document.createElement('div');
    messageDiv.className = 'chat-message assistant';
    const messageId = `msg-${Date.now()}`;
    messageDiv.id = messageId;

    let responseText = '';
    // Backend sends 'data', normalize to 'rows' for consistency
    const rows = result.rows || result.data || [];
    let hasResults = rows.length > 0;

    // Use AI-generated response if available, otherwise fall back to generic message
    if (result.aiResponse) {
        responseText = result.aiResponse;
    } else if (hasResults) {
        responseText = `I found ${rows.length} result${rows.length !== 1 ? 's' : ''} for your query.`;
    } else if (result.error) {
        // Query failed with error
        responseText = `Query error: ${result.error}`;
    } else {
        responseText = 'Query executed successfully, but returned no results.';
    }

    // Normalize result to have 'rows' field for createResultsTable
    result.rows = rows;

    const feedbackContainerId = `feedback-${messageId}`;

    // Add the original query to result for the Show SQL modal
    const resultWithQuery = { ...result, naturalLanguage: originalQuery };

    // Show TSQL button if there's generated SQL
    const hasSql = result.generatedSql || result.sql;
    const tsqlButton = hasSql ? `
        <div class="sql-actions" style="margin-top: 8px;">
            <button class="ewr-button-show" onclick="showResultSql(this)" title="View generated T-SQL query"
                    data-sql='${(result.generatedSql || result.sql || '').replace(/'/g, "&#39;")}'
                    data-query='${(originalQuery || '').replace(/'/g, "&#39;")}'>
                TSQL
            </button>
        </div>
    ` : '';

    // Build timing metrics display
    const timingMetricsHtml = buildTimingMetricsHtml(result);

    messageDiv.innerHTML = `
        <div class="message-text">
            ${responseText}
            ${hasResults ? createResultsTable(resultWithQuery) : tsqlButton}
        </div>
        ${timingMetricsHtml}
        <div id="${feedbackContainerId}" class="feedback-buttons">
            <button class="feedback-btn-icon" data-message-id="${messageId}" data-feedback="positive" title="Helpful">👍</button>
            <button class="feedback-btn-icon" data-message-id="${messageId}" data-feedback="negative" title="Not helpful">👎</button>
        </div>
    `;

    chatMessages.appendChild(messageDiv);
    scrollToBottom();

    // Attach feedback button event listeners
    const feedbackButtons = messageDiv.querySelectorAll('.feedback-btn-icon');
    feedbackButtons.forEach(btn => {
        btn.addEventListener('click', function() {
            const feedbackType = this.dataset.feedback;
            const msgId = this.dataset.messageId;

            if (feedbackType === 'positive') {
                handlePositiveFeedback(originalQuery, result.generatedSql, msgId, this);
            } else {
                handleNegativeFeedback(originalQuery, result.generatedSql, msgId, this);
            }
        });
    });

    // Add to history
    const time = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    addToChatHistory({
        type: 'assistant',
        text: responseText,
        sql: result.generatedSql,
        results: result,
        time
    });
    updateTokenTrackerFromHistory();
}

/**
 * Add an error message to the chat display
 * @param {string} errorText - The error message text
 * @param {string|null} failedSql - The SQL that failed (if available)
 * @param {string|null} originalQuestion - The original user question
 * @param {string|null} aiExplanation - AI-generated explanation of the error
 */
export function addErrorMessage(errorText, failedSql = null, originalQuestion = null, aiExplanation = null) {
    const chatMessages = document.getElementById('chatMessages');

    const messageDiv = document.createElement('div');
    messageDiv.className = 'chat-message assistant';
    const errorId = `error-${Date.now()}`;
    messageDiv.id = errorId;

    const time = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });

    // Parse and format the error message for better user understanding
    const formattedError = formatErrorMessage(errorText);

    // Build AI explanation block if available
    let aiExplanationBlock = '';
    if (aiExplanation) {
        aiExplanationBlock = `
            <div style="background: linear-gradient(135deg, #1e3a5f 0%, #1e293b 100%); padding: 16px; border-radius: 8px; margin-bottom: 16px; border-left: 4px solid #3b82f6;">
                <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 8px;">
                    <span style="font-size: 18px;">AI</span>
                    <span style="font-weight: 600; color: #60a5fa;">What happened</span>
                </div>
                <div style="color: #e2e8f0; line-height: 1.6; font-size: 14px;">
                    ${escapeHtml(aiExplanation)}
                </div>
            </div>
        `;
    }

    // Get the original question from chat history if not provided
    const question = originalQuestion || (state.chatHistory.length > 0 ?
        state.chatHistory.filter(h => h.type === 'user').pop()?.text : '');

    let sqlBlock = '';
    if (failedSql) {
        sqlBlock = `
            <div class="sql-block" style="margin-top: 12px;">
                <div class="sql-header" style="color: #ef4444;">
                    <span>X</span>
                    <span>Failed SQL Query</span>
                </div>
                <div class="sql-code" style="color: #fca5a5; border-left: 3px solid #ef4444;">
                    ${escapeHtml(failedSql)}
                </div>
            </div>
        `;
    }

    // Simple feedback buttons for errors
    const feedbackContainerId = `feedback-${errorId}`;

    messageDiv.innerHTML = `
        ${aiExplanationBlock}
        <div class="error-block" style="margin-top: 0;">
            <div style="font-weight: 600; color: #fca5a5; margin-bottom: 8px;">
                ${formattedError.title}
            </div>
            <div class="error-text" style="margin-bottom: 8px;">
                ${formattedError.message}
            </div>
            ${formattedError.suggestion ? `
                <div style="color: #cbd5e1; margin-top: 12px; padding: 12px; background: #1e293b; border-left: 3px solid #3b82f6;">
                    <strong style="color: #3b82f6;">Suggestion:</strong> ${formattedError.suggestion}
                </div>
            ` : ''}
        </div>
        ${sqlBlock}
        <div id="${feedbackContainerId}" class="feedback-buttons">
            <button class="feedback-btn-icon" data-message-id="${errorId}" data-feedback="positive" title="Helpful">👍</button>
            <button class="feedback-btn-icon" data-message-id="${errorId}" data-feedback="negative" title="Not helpful">👎</button>
        </div>
    `;

    chatMessages.appendChild(messageDiv);
    scrollToBottom();

    // Attach feedback button event listeners
    const feedbackButtons = messageDiv.querySelectorAll('.feedback-btn-icon');
    feedbackButtons.forEach(btn => {
        btn.addEventListener('click', function() {
            const feedbackType = this.dataset.feedback;
            const msgId = this.dataset.messageId;

            if (feedbackType === 'positive') {
                handlePositiveFeedback(question, failedSql, msgId, this);
            } else {
                handleNegativeFeedback(question, failedSql, msgId, this);
            }
        });
    });

    // Add to history
    addToChatHistory({ type: 'error', text: errorText, sql: failedSql, question, time });
    updateTokenTrackerFromHistory();
}

/**
 * Format error message to be more user-friendly and actionable
 * Analyzes the error and provides context and suggestions
 *
 * RAG Architecture: This helps users understand SQL errors by providing
 * semantic error classification and actionable remediation steps
 *
 * @param {string} errorText - The raw error message
 * @returns {Object} Formatted error with title, message, and suggestion
 */
export function formatErrorMessage(errorText) {
    const errorLower = errorText.toLowerCase();

    // Invalid column name errors
    if (errorLower.includes('invalid column name')) {
        const columnMatch = errorText.match(/invalid column name '([^']+)'/i);
        const columnName = columnMatch ? columnMatch[1] : 'unknown';

        return {
            title: 'Invalid Column Name',
            message: `The column "${columnName}" does not exist in the selected table.`,
            suggestion: `Check the table schema or try rephrasing your query. Use "Show me all columns in [table name]" to see available columns.`
        };
    }

    // Invalid object name errors (table doesn't exist)
    if (errorLower.includes('invalid object name')) {
        const tableMatch = errorText.match(/invalid object name '([^']+)'/i);
        const tableName = tableMatch ? tableMatch[1] : 'unknown';

        return {
            title: 'Table Not Found',
            message: `The table "${tableName}" does not exist in the database.`,
            suggestion: `Verify the table name is correct. Try asking "What tables are available?" or check the schema in the connection panel.`
        };
    }

    // Syntax errors
    if (errorLower.includes('incorrect syntax') || errorLower.includes('syntax error')) {
        return {
            title: 'SQL Syntax Error',
            message: `The generated SQL query has a syntax error: ${errorText}`,
            suggestion: `Try rephrasing your question more clearly, or check if you meant to query a different table or column.`
        };
    }

    // Ambiguous column name
    if (errorLower.includes('ambiguous column name') || errorLower.includes('ambiguous')) {
        const columnMatch = errorText.match(/column name '([^']+)'/i);
        const columnName = columnMatch ? columnMatch[1] : 'unknown';

        return {
            title: 'Ambiguous Column Reference',
            message: `The column "${columnName}" exists in multiple tables and needs to be specified with a table prefix.`,
            suggestion: `Be more specific about which table you want to query, or ask about specific tables separately.`
        };
    }

    // Permission/authentication errors
    if (errorLower.includes('permission') || errorLower.includes('denied') || errorLower.includes('login failed')) {
        return {
            title: 'Authentication/Permission Error',
            message: errorText,
            suggestion: `Check your database credentials in the connection panel. Verify you have the correct username, password, and permissions for this database.`
        };
    }

    // Conversion errors
    if (errorLower.includes('conversion') || errorLower.includes('cast')) {
        return {
            title: 'Data Type Conversion Error',
            message: errorText,
            suggestion: `There's a data type mismatch in the query. Try being more specific about the data type you're looking for (numbers, dates, text, etc.).`
        };
    }

    // Connection errors
    if (errorLower.includes('connection') || errorLower.includes('timeout') || errorLower.includes('network')) {
        return {
            title: 'Connection Error',
            message: errorText,
            suggestion: `The database connection failed. Check if the server is accessible and your connection settings are correct. Try "Test Connection" first.`
        };
    }

    // Aggregate function errors
    if (errorLower.includes('aggregate') || errorLower.includes('group by')) {
        return {
            title: 'Aggregate Function Error',
            message: errorText,
            suggestion: `When using aggregate functions (COUNT, SUM, AVG, etc.), all non-aggregated columns must be in the GROUP BY clause. Try rephrasing to be clearer about what you want to aggregate.`
        };
    }

    // Foreign key constraint errors
    if (errorLower.includes('foreign key') || errorLower.includes('reference constraint')) {
        return {
            title: 'Foreign Key Constraint Violation',
            message: errorText,
            suggestion: `The operation violates a foreign key relationship. Ensure referenced records exist in the related table before inserting or updating.`
        };
    }

    // Generic error - show the full error text
    return {
        title: 'Query Error',
        message: errorText,
        suggestion: `Review the generated SQL query and your question. Try rephrasing your request more specifically, or ask about the table structure first.`
    };
}

/**
 * Toggle visibility of error details block
 * @param {string} errorId - The ID of the error block element
 */
export function toggleErrorDetails(errorId) {
    const errorBlock = document.getElementById(errorId);
    if (errorBlock) {
        if (errorBlock.style.display === 'none') {
            errorBlock.style.display = 'block';
        } else {
            errorBlock.style.display = 'none';
        }
    }
}

/**
 * Add a loading message to the chat while processing a query
 * @returns {string} The ID of the loading message element
 */
export function addLoadingMessage() {
    const chatMessages = document.getElementById('chatMessages');

    const loadingDiv = document.createElement('div');
    loadingDiv.className = 'chat-message assistant';
    const loadingId = `loading-${Date.now()}`;
    loadingDiv.id = loadingId;

    // Loading indicator with step and detail display
    loadingDiv.innerHTML = `
        <div class="loading-indicator">
            <div class="spinner"></div>
            <div class="loading-content">
                <span class="loading-text" id="${loadingId}-text">Waiting on response</span>
                <span class="loading-detail" id="${loadingId}-detail"></span>
            </div>
        </div>
    `;

    // Store start time for metrics
    state.sessionMetrics.currentQueryStartTime = Date.now();

    chatMessages.appendChild(loadingDiv);
    scrollToBottom();

    return loadingId;
}

/**
 * Update the loading message with current pipeline step information
 * @param {string} loadingId - The ID of the loading message element
 * @param {string} message - The status message to display
 * @param {Object|null} data - Optional data object with stage information
 */
export function updateLoadingMessage(loadingId, message, data = null) {
    const detailSpan = document.getElementById(`${loadingId}-detail`);

    // If we have structured data with stage, use it directly
    if (data && data.stage) {
        const stepIndex = PIPELINE_STEPS.findIndex(s => s.id === data.stage);
        if (stepIndex !== -1) {
            const step = PIPELINE_STEPS[stepIndex];
            const textSpan = document.getElementById(`${loadingId}-text`);
            if (textSpan) {
                if (step.isRulesStep) {
                    textSpan.textContent = `* Rules - ${step.label}`;
                } else {
                    const stepNum = stepIndex + 1;
                    textSpan.textContent = `Step ${stepNum}/${PIPELINE_STEPS.length} - ${step.label}`;
                }
            }
            // Show SSE detail message beneath the step indicator
            if (detailSpan && data.message) {
                detailSpan.textContent = data.message;
            }
            console.log(`[Pipeline] Stage: ${data.stage}, Step: ${stepIndex + 1}/${PIPELINE_STEPS.length}, Detail: ${data.message || ''}`);
            return;
        }
    }

    // Fallback: Map backend message text to pipeline step IDs
    const stepMapping = {
        // Backend stage names (highest priority - check these first)
        'preprocessing:': 'preprocessing',
        'security:': 'security',
        'cache:': 'cache',
        'rules:': 'rules',
        'schema:': 'schema',
        'generating:': 'generating',
        'fixing:': 'fixing',
        'validating:': 'validating',
        'executing:': 'executing',
        'complete:': 'complete',
        // Backend message patterns (fallback)
        'Analyzing question': 'preprocessing',
        'Checking security': 'security',
        'Checking for cached': 'cache',
        'Searching SQL rules': 'rules',
        'Found exact rule match': 'rules',
        'matching rules': 'rules',
        'Using rule': 'rules',
        'Loading database schema': 'schema',
        'relevant tables': 'schema',
        'Generating SQL': 'generating',
        'SQL generated': 'generating',
        'Applying syntax fixes': 'fixing',
        'auto-fixes': 'fixing',
        'Validating': 'validating',
        'validation': 'validating',
        'Executing SQL': 'executing',
        'Executing query': 'executing',
        'Query returned': 'executing',
        'Processing complete': 'complete'
    };

    // Find which step this message corresponds to
    let matchedStepId = null;
    const messageLC = message.toLowerCase();

    for (const [key, stepId] of Object.entries(stepMapping)) {
        if (messageLC.includes(key.toLowerCase())) {
            matchedStepId = stepId;
            break;
        }
    }

    // Find step index and label
    let stepNumber = 1;
    let stepLabel = message;
    let isRulesStep = false;

    if (matchedStepId) {
        const stepIndex = PIPELINE_STEPS.findIndex(s => s.id === matchedStepId);
        if (stepIndex !== -1) {
            stepNumber = stepIndex + 1;
            stepLabel = PIPELINE_STEPS[stepIndex].label;
            isRulesStep = PIPELINE_STEPS[stepIndex].isRulesStep || false;
        }
    }

    // Update the loading text with step number/icon
    const textSpan = document.getElementById(`${loadingId}-text`);
    if (textSpan) {
        if (isRulesStep) {
            textSpan.textContent = `* Rules - ${stepLabel}`;
        } else {
            textSpan.textContent = `Step ${stepNumber}/${PIPELINE_STEPS.length} - ${stepLabel}`;
        }
    }
    // Show detail message beneath the step indicator
    if (detailSpan && message) {
        detailSpan.textContent = message;
    }
    console.log(`[Pipeline] Fallback match: "${message.substring(0, 30)}...", Matched: ${matchedStepId}, Step: ${stepNumber}`);
}

/**
 * Remove the loading message from the chat
 * @param {string} loadingId - The ID of the loading message element to remove
 */
export function removeLoadingMessage(loadingId) {
    const loadingDiv = document.getElementById(loadingId);
    if (loadingDiv) {
        loadingDiv.remove();
    }
}

/**
 * Clear all chat messages and reset the chat to initial state
 * Prompts for confirmation if there is existing chat history
 */
export function clearChat() {
    if (state.chatHistory.length > 0 && !confirm('Clear all chat history?')) {
        return;
    }

    clearChatHistory();
    updateTokenTrackerFromHistory(); // Reset token counter
    const chatMessages = document.getElementById('chatMessages');
    chatMessages.innerHTML = `
        <div class="empty-chat" id="emptyChat">
            <div class="empty-icon">?</div>
            <div class="empty-title">Start a Conversation</div>
            <div class="empty-text">
                Ask questions about your database in natural language. I'll help you write and execute SQL queries.
            </div>
        </div>
    `;

    // Reset token tracker
    resetTokenTracker();
    updateTokenDisplay();
    hideContextLimitAlert();

    // Clear timing display
    clearTimingDisplay();

    // Re-enable chat if it was disabled due to limit
    const chatInput = document.getElementById('chatInput');
    const sendBtn = document.getElementById('sendBtn');
    chatInput.disabled = !state.app.databaseLoaded;
    sendBtn.disabled = !state.app.databaseLoaded || state.isProcessing;

    // Reset session metrics
    resetSessionMetrics();
}
