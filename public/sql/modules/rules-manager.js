/**
 * SQL Rules Manager Module
 *
 * Handles SQL rules management functionality including:
 * - Adding and saving new rules
 * - AI-assisted rule generation
 * - Viewing and managing existing rules
 * - Example modal management for saving SQL examples
 *
 * @module rules-manager
 */

import { SQL_API_BASE, state } from './sql-chat-state.js';
import { getConnectionConfig } from './connection-manager.js';

// ========================
// Utility Functions
// ========================

/**
 * Display a status message in a container element
 * @param {HTMLElement} container - The container element to display the message in
 * @param {string} type - The message type ('success', 'error', 'info')
 * @param {string} message - The message text to display
 */
function showMessage(container, type, message) {
    // Set the base class and type class
    container.className = 'ewr-status-pill ' + type;
    // Set the message text directly
    container.textContent = message;

    // Show the status message container
    const statusContainer = document.getElementById('statusMessageContainer');
    if (statusContainer && message) {
        statusContainer.classList.add('visible');
    }
}

/**
 * Escape HTML special characters to prevent XSS
 * @param {string} text - The text to escape
 * @returns {string} The escaped HTML string
 */
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// ========================
// Example Modal Functions
// ========================

/**
 * Show the Add Example modal for saving SQL query examples
 * Requires a database to be selected first
 */
export function showAddExampleModal() {
    const database = document.getElementById('database').value;

    if (!database) {
        alert('Please select a database first');
        return;
    }

    // Clear form
    document.getElementById('examplePrompt').value = '';
    document.getElementById('exampleSql').value = '';
    document.getElementById('exampleResponse').value = '';
    document.getElementById('exampleMessage').innerHTML = '';

    // Hide test results
    document.getElementById('exampleTestResults').style.display = 'none';
    document.getElementById('exampleTestOutput').innerHTML = '';

    // Display database name
    document.getElementById('exampleDatabaseName').textContent = database;

    // Show modal
    document.getElementById('exampleModal').classList.remove('hidden');
}

/**
 * Close the Add Example modal
 * @param {Event} [event] - Optional click event for backdrop click handling
 */
export function closeExampleModal(event) {
    if (event && event.target !== event.currentTarget) {
        return;
    }
    document.getElementById('exampleModal').classList.add('hidden');
}

/**
 * Save an SQL example to the database
 * Validates required fields and sends to the save-example API endpoint
 */
export async function saveExample() {
    const prompt = document.getElementById('examplePrompt').value.trim();
    const sql = document.getElementById('exampleSql').value.trim();
    const response = document.getElementById('exampleResponse').value.trim();
    const database = document.getElementById('database').value;
    const messageDiv = document.getElementById('exampleMessage');

    if (!prompt || !sql) {
        showMessage(messageDiv, 'error', 'Prompt and SQL query are required');
        return;
    }

    const connectionConfig = getConnectionConfig();

    try {
        const result = await fetch(`${SQL_API_BASE}/api/sql/save-example`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                database: connectionConfig.database,
                prompt,
                sql,
                response: response || null
            })
        });

        const data = await result.json();

        if (data.success) {
            showMessage(messageDiv, 'success', 'Example saved successfully! The AI will use this for similar queries.');
            setTimeout(() => {
                closeExampleModal();
            }, 2000);
        } else {
            showMessage(messageDiv, 'error', `Failed to save example: ${data.error}`);
        }
    } catch (error) {
        showMessage(messageDiv, 'error', `Error saving example: ${error.message}`);
    }
}

/**
 * Test the SQL query before saving as an example
 * Executes the query against the current database connection and displays results
 */
export async function testExampleSql() {
    const sql = document.getElementById('exampleSql').value.trim();
    const resultsDiv = document.getElementById('exampleTestResults');
    const outputDiv = document.getElementById('exampleTestOutput');
    const statusSpan = document.getElementById('exampleTestStatus');
    const messageDiv = document.getElementById('exampleMessage');

    if (!sql) {
        showMessage(messageDiv, 'error', 'Please enter a SQL query to test');
        return;
    }

    // Show results area with loading state
    resultsDiv.style.display = 'block';
    statusSpan.innerHTML = '<span style="color: #3b82f6;">Testing...</span>';
    outputDiv.innerHTML = '<div style="color: #94a3b8;">Executing query...</div>';

    const connectionConfig = getConnectionConfig();

    try {
        // Execute the SQL directly using the execute-sql endpoint
        const response = await fetch(`${SQL_API_BASE}/api/sql/execute`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                ...connectionConfig,
                sql: sql
            })
        });

        const result = await response.json();

        if (result.success) {
            const rowCount = result.rowCount || (result.rows ? result.rows.length : 0);
            statusSpan.innerHTML = `<span style="color: #10b981;">Success - ${rowCount} row${rowCount !== 1 ? 's' : ''}</span>`;

            if (rowCount > 0 && result.rows && result.columns) {
                // Build a simple table preview (first 5 rows)
                const previewRows = result.rows.slice(0, 5);
                let tableHtml = '<table style="width: 100%; border-collapse: collapse; font-size: 12px; margin-top: 8px;">';
                tableHtml += '<thead><tr>';
                result.columns.forEach(col => {
                    tableHtml += `<th style="background: #334155; padding: 6px 8px; text-align: left; color: #e2e8f0; border: 1px solid #475569;">${escapeHtml(col)}</th>`;
                });
                tableHtml += '</tr></thead><tbody>';

                previewRows.forEach(row => {
                    tableHtml += '<tr>';
                    result.columns.forEach(col => {
                        const value = row[col];
                        const displayValue = value === null || value === undefined ? 'NULL' : String(value).substring(0, 50);
                        tableHtml += `<td style="padding: 6px 8px; border: 1px solid #334155; color: #cbd5e1;">${escapeHtml(displayValue)}</td>`;
                    });
                    tableHtml += '</tr>';
                });
                tableHtml += '</tbody></table>';

                if (rowCount > 5) {
                    tableHtml += `<div style="margin-top: 8px; color: #64748b; font-size: 11px;">Showing 5 of ${rowCount} rows</div>`;
                }

                outputDiv.innerHTML = tableHtml;
            } else if (rowCount === 0) {
                outputDiv.innerHTML = '<div style="color: #f59e0b;">Query executed successfully but returned no rows.</div>';
            } else {
                outputDiv.innerHTML = '<div style="color: #10b981;">Query executed successfully.</div>';
            }
        } else {
            statusSpan.innerHTML = '<span style="color: #ef4444;">Error</span>';
            outputDiv.innerHTML = `<div style="color: #fca5a5;">${escapeHtml(result.error || result.details || 'Query execution failed')}</div>`;
        }
    } catch (error) {
        statusSpan.innerHTML = '<span style="color: #ef4444;">Error</span>';
        outputDiv.innerHTML = `<div style="color: #fca5a5;">Test failed: ${escapeHtml(error.message)}</div>`;
    }
}

// ========================
// SQL Rules Management
// ========================

/**
 * Show the Add Rule modal for creating new SQL rules
 * Requires a database to be selected first
 */
export function showAddRuleModal() {
    const database = document.getElementById('database').value;

    if (!database) {
        alert('Please select a database first');
        return;
    }

    const modal = document.getElementById('addRuleModal');
    if (modal) {
        // Clear form
        document.getElementById('ruleDescription').value = '';
        document.getElementById('ruleType').value = 'constraint';
        document.getElementById('ruleTriggerKeywords').value = '';
        document.getElementById('ruleText').value = '';
        document.getElementById('ruleAutoFixPattern').value = '';
        document.getElementById('ruleAutoFixReplacement').value = '';
        document.getElementById('addRuleMessage').innerHTML = '';

        modal.classList.remove('hidden');
    }
}

/**
 * Generate rule fields using AI based on problem description
 * Uses the LLM to analyze the problem and suggest rule configuration
 */
export async function generateRuleWithAI() {
    const problemText = document.getElementById('aiRuleProblem').value.trim();
    const database = document.getElementById('database').value;

    if (!problemText) {
        showAddRuleMessage('Please describe the problem first', true);
        return;
    }

    if (!database) {
        showAddRuleMessage('Please select a database first', true);
        return;
    }

    const btn = document.getElementById('aiGenerateBtn');
    const btnText = document.getElementById('aiGenerateBtnText');
    btn.disabled = true;
    btnText.textContent = 'Generating...';

    try {
        // Build prompt for AI
        const prompt = `You are an expert at creating SQL rules for a text-to-SQL system. Based on the user's problem description, generate a rule to fix/prevent the issue.

Database: ${database}

User's Problem:
${problemText}

Generate a JSON response with these fields:
{
  "description": "Short rule description (under 80 chars)",
  "type": "assistance" or "constraint",
  "trigger_keywords": "comma-separated keywords that would appear in user questions",
  "rule_text": "Detailed guidance for the LLM on what to do, including SQL patterns",
  "auto_fix_pattern": "optional regex pattern to find in generated SQL",
  "auto_fix_replacement": "optional replacement text"
}

Make the rule_text specific with actual SQL syntax examples. Include relevant table and column names.
Respond ONLY with valid JSON, no other text.`;

        const response = await fetch(`${SQL_API_BASE}/api/sql/generate-rule`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                prompt: prompt,
                database: database
            })
        });

        if (!response.ok) {
            // Fallback: Try using the general LLM endpoint
            const fallbackResponse = await fetch(`${SQL_API_BASE}/api/llm/complete`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    prompt: prompt,
                    max_tokens: 1000,
                    temperature: 0.3
                })
            });

            if (!fallbackResponse.ok) {
                throw new Error('AI service unavailable');
            }

            const fallbackData = await fallbackResponse.json();
            const jsonMatch = fallbackData.response?.match(/\{[\s\S]*\}/);
            if (jsonMatch) {
                populateRuleFields(JSON.parse(jsonMatch[0]));
            } else {
                throw new Error('Could not parse AI response');
            }
        } else {
            const data = await response.json();
            if (data.rule) {
                populateRuleFields(data.rule);
            } else {
                throw new Error('Invalid response format');
            }
        }

        showAddRuleMessage('Rule generated! Review and adjust the fields below, then click Save Rule.', false);

    } catch (error) {
        console.error('AI rule generation error:', error);
        showAddRuleMessage(`AI generation failed: ${error.message}. Please fill in the fields manually.`, true);
    } finally {
        btn.disabled = false;
        btnText.textContent = 'Generate Rule with AI';
    }
}

/**
 * Populate rule form fields from AI-generated data
 * @param {Object} rule - The rule object containing field values
 * @param {string} [rule.description] - Rule description
 * @param {string} [rule.type] - Rule type ('assistance' or 'constraint')
 * @param {string} [rule.trigger_keywords] - Comma-separated trigger keywords
 * @param {string} [rule.rule_text] - The rule text/guidance
 * @param {string} [rule.auto_fix_pattern] - Auto-fix regex pattern
 * @param {string} [rule.auto_fix_replacement] - Auto-fix replacement text
 */
export function populateRuleFields(rule) {
    if (rule.description) {
        document.getElementById('ruleDescription').value = rule.description;
    }
    if (rule.type) {
        document.getElementById('ruleType').value = rule.type;
    }
    if (rule.trigger_keywords) {
        document.getElementById('ruleTriggerKeywords').value = rule.trigger_keywords;
    }
    if (rule.rule_text) {
        document.getElementById('ruleText').value = rule.rule_text;
    }
    if (rule.auto_fix_pattern) {
        document.getElementById('ruleAutoFixPattern').value = rule.auto_fix_pattern;
    }
    if (rule.auto_fix_replacement) {
        document.getElementById('ruleAutoFixReplacement').value = rule.auto_fix_replacement;
    }
}

/**
 * Close the Add Rule modal
 * @param {Event} [event] - Optional click event for backdrop click handling
 */
export function closeAddRuleModal(event) {
    if (event && event.target !== event.currentTarget) return;
    const modal = document.getElementById('addRuleModal');
    if (modal) {
        modal.classList.add('hidden');
    }
    // Clear AI problem text
    const aiProblem = document.getElementById('aiRuleProblem');
    if (aiProblem) aiProblem.value = '';
}

/**
 * Show message in Add Rule modal
 * @param {string} message - The message to display
 * @param {boolean} [isError=false] - Whether this is an error message
 */
export function showAddRuleMessage(message, isError = false) {
    const msgDiv = document.getElementById('addRuleMessage');
    if (msgDiv) {
        msgDiv.innerHTML = `<div style="padding: 10px; margin-bottom: 12px; border-radius: 6px; background: ${isError ? 'rgba(239, 68, 68, 0.15)' : 'rgba(16, 185, 129, 0.15)'}; color: ${isError ? '#fca5a5' : '#6ee7b7'}; border: 1px solid ${isError ? '#ef4444' : '#10b981'};">${message}</div>`;
    }
}

/**
 * Save a new rule via API
 * Validates required fields, generates rule ID, and submits to the rules endpoint
 */
export async function saveRule() {
    const database = document.getElementById('database').value;
    const description = document.getElementById('ruleDescription').value.trim();
    const ruleType = document.getElementById('ruleType').value;
    const triggerKeywords = document.getElementById('ruleTriggerKeywords').value.trim();
    const ruleText = document.getElementById('ruleText').value.trim();
    const autoFixPattern = document.getElementById('ruleAutoFixPattern').value.trim();
    const autoFixReplacement = document.getElementById('ruleAutoFixReplacement').value.trim();

    // Validation
    if (!description) {
        showAddRuleMessage('Description is required', true);
        return;
    }
    if (!ruleText) {
        showAddRuleMessage('Rule text is required', true);
        return;
    }

    // Generate rule ID from description
    const ruleId = description
        .toLowerCase()
        .replace(/[^a-z0-9\s-]/g, '')
        .replace(/\s+/g, '-')
        .substring(0, 50);

    // Build rule object
    const rule = {
        database: database,
        rule_id: ruleId,
        description: description,
        type: ruleType,
        priority: 'high',
        enabled: true,
        rule_text: ruleText
    };

    // Add optional trigger keywords
    if (triggerKeywords) {
        rule.trigger_keywords = triggerKeywords.split(',').map(k => k.trim()).filter(k => k);
    }

    // Add optional auto-fix
    if (autoFixPattern && autoFixReplacement) {
        rule.auto_fix_pattern = autoFixPattern;
        rule.auto_fix_replacement = autoFixReplacement;
    }

    try {
        const response = await fetch(`${SQL_API_BASE}/api/sql/rules`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(rule)
        });

        const result = await response.json();

        if (result.success) {
            showAddRuleMessage('Rule saved successfully!', false);
            setTimeout(() => {
                closeAddRuleModal();
            }, 1500);
        } else {
            showAddRuleMessage(result.error || result.detail || 'Failed to save rule', true);
        }
    } catch (error) {
        showAddRuleMessage(`Error: ${error.message}`, true);
    }
}

// ========================
// Rules List Modal
// ========================

/**
 * Show the Rules List modal
 * Fetches all rules from the API and displays them organized by scope
 */
export async function showRulesModal() {
    const modal = document.getElementById('rulesListModal');
    const content = document.getElementById('rulesListContent');

    if (modal) {
        modal.classList.remove('hidden');
    }

    // Show loading
    if (content) {
        content.innerHTML = `<div style="text-align: center; color: #94a3b8; padding: 40px;">
            <div class="spinner" style="margin: 0 auto 16px;"></div>
            Loading rules...
        </div>`;
    }

    try {
        const response = await fetch(`${SQL_API_BASE}/api/sql/rules`);
        const result = await response.json();

        if (result.success) {
            renderRulesList(result);
        } else {
            content.innerHTML = `<div style="color: #fca5a5; padding: 20px;">Failed to load rules: ${result.error || 'Unknown error'}</div>`;
        }
    } catch (error) {
        content.innerHTML = `<div style="color: #fca5a5; padding: 20px;">Error loading rules: ${error.message}</div>`;
    }
}

/**
 * Render the rules list in the modal
 * Organizes rules by global constraints and database-specific rules
 * @param {Object} data - The rules data from the API
 * @param {Array} [data.global_constraints] - Global constraint rules
 * @param {Object} [data.database_rules] - Database-specific rules keyed by database name
 */
export function renderRulesList(data) {
    const content = document.getElementById('rulesListContent');
    if (!content) return;

    let html = '';

    // Global Constraints Section
    if (data.global_constraints && data.global_constraints.length > 0) {
        html += `<div style="margin-bottom: 24px;">
            <h3 style="color: #a78bfa; margin-bottom: 12px; display: flex; align-items: center; gap: 8px;">
                <span style="font-size: 20px;">Global Constraints</span>
                <span style="background: #8b5cf6; color: white; padding: 2px 8px; border-radius: 10px; font-size: 12px;">${data.global_constraints.length}</span>
            </h3>`;

        data.global_constraints.forEach(rule => {
            html += renderRuleCard(rule, 'global');
        });

        html += '</div>';
    }

    // Database-Specific Rules
    if (data.database_rules) {
        Object.entries(data.database_rules).forEach(([dbName, dbData]) => {
            if (dbData.rules && dbData.rules.length > 0) {
                html += `<div style="margin-bottom: 24px;">
                    <h3 style="color: #60a5fa; margin-bottom: 12px; display: flex; align-items: center; gap: 8px;">
                        <span style="font-size: 20px;">${dbName}</span>
                        <span style="background: #3b82f6; color: white; padding: 2px 8px; border-radius: 10px; font-size: 12px;">${dbData.rules.length}</span>
                    </h3>`;

                dbData.rules.forEach(rule => {
                    html += renderRuleCard(rule, dbName);
                });

                html += '</div>';
            }
        });
    }

    if (!html) {
        html = '<div style="color: #94a3b8; padding: 20px; text-align: center;">No rules configured yet. Click "Add Rule" to create your first rule.</div>';
    }

    content.innerHTML = html;
}

/**
 * Render a single rule card
 * @param {Object} rule - The rule object to render
 * @param {string} rule.id - Rule ID
 * @param {string} rule.description - Rule description
 * @param {string} rule.rule_text - Rule guidance text
 * @param {string} [rule.priority] - Rule priority ('critical', 'high', etc.)
 * @param {Array} [rule.trigger_keywords] - Keywords that trigger this rule
 * @param {Object} [rule.example] - Example question and SQL
 * @param {string} scope - The scope/database name for this rule
 * @returns {string} HTML string for the rule card
 */
export function renderRuleCard(rule, scope) {
    const priorityBadge = rule.priority === 'critical'
        ? '<span style="background: #dc2626; color: white; padding: 2px 6px; border-radius: 4px; font-size: 10px; margin-left: 8px;">CRITICAL</span>'
        : '';

    let html = `<div style="background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%); border: 1px solid #334155; border-radius: 8px; padding: 14px; margin-bottom: 10px;">
        <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 8px;">
            <div>
                <span style="color: #94a3b8; font-size: 11px; font-family: monospace;">${escapeHtml(rule.id)}</span>
                ${priorityBadge}
            </div>
        </div>
        <div style="font-weight: 600; color: #e2e8f0; margin-bottom: 6px;">${escapeHtml(rule.description)}</div>
        <div style="color: #94a3b8; font-size: 13px; line-height: 1.5;">${escapeHtml(rule.rule_text)}</div>`;

    // Trigger keywords
    if (rule.trigger_keywords && rule.trigger_keywords.length > 0) {
        html += `<div style="margin-top: 10px;">
            <span style="color: #64748b; font-size: 11px;">Triggers: </span>`;
        rule.trigger_keywords.forEach(kw => {
            html += `<span style="background: #1e40af; color: #93c5fd; padding: 2px 6px; border-radius: 4px; font-size: 11px; margin-right: 4px;">${escapeHtml(kw)}</span>`;
        });
        html += '</div>';
    }

    // Example
    if (rule.example) {
        html += `<div style="margin-top: 10px; background: #0f172a; border-radius: 6px; padding: 10px;">
            <div style="color: #64748b; font-size: 11px; margin-bottom: 4px;">Example:</div>
            <div style="color: #94a3b8; font-size: 12px; margin-bottom: 4px;">Q: ${escapeHtml(rule.example.question)}</div>
            <div style="color: #6ee7b7; font-size: 12px; font-family: monospace; white-space: pre-wrap; word-break: break-all;">${escapeHtml(rule.example.sql)}</div>
        </div>`;
    }

    html += '</div>';
    return html;
}

/**
 * Close the Rules List modal
 * @param {Event} [event] - Optional click event for backdrop click handling
 */
export function closeRulesListModal(event) {
    if (event && event.target !== event.currentTarget) return;
    const modal = document.getElementById('rulesListModal');
    if (modal) {
        modal.classList.add('hidden');
    }
}
