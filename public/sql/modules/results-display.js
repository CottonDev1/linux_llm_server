/**
 * Results Display Module
 *
 * Handles rendering and interaction with SQL query results including:
 * - SQL code block generation with copy functionality
 * - Results table HTML generation
 * - CSV export functionality
 * - SQL preview modal display
 * - Clipboard operations
 *
 * @module results-display
 */

import { escapeHtml } from './ui-manager.js';

/**
 * Create an SQL code block with syntax highlighting wrapper
 * @param {string} sql - The SQL query string to display
 * @returns {string} HTML string for the SQL code block
 */
export function createSqlBlock(sql) {
    return `
        <div class="sql-block">
            <div class="sql-header">
                <span>&#128221;</span>
                <span>Generated SQL</span>
            </div>
            <div class="sql-code">${escapeHtml(sql)}</div>
        </div>
    `;
}

/**
 * Create an HTML table to display query results
 * Includes row count, TSQL view button, and CSV export button
 * @param {Object} result - The query result object
 * @param {string[]} result.columns - Array of column names
 * @param {Object[]} result.rows - Array of row objects
 * @param {string} [result.generatedSql] - The SQL query that generated these results
 * @param {string} [result.naturalLanguage] - The original natural language query
 * @param {string} [result.originalQuery] - Alternative property for original query
 * @returns {string} HTML string for the results table, or empty string if no results
 */
export function createResultsTable(result) {
    if (!result.columns || !result.rows || result.rows.length === 0) {
        return '';
    }

    // Generate unique ID for export button
    const tableId = `table-${Date.now()}`;

    // Store SQL for the Show SQL button
    const sqlForDisplay = result.generatedSql ? escapeHtml(result.generatedSql).replace(/'/g, "&#39;") : '';

    let tableHtml = `
        <div class="results-table-wrapper" data-results='${JSON.stringify(result).replace(/'/g, "&#39;")}' data-sql='${sqlForDisplay}'>
            <div class="results-header">
                <div class="results-count">${result.rows.length} row${result.rows.length !== 1 ? 's' : ''}</div>
                <div style="display: flex; gap: 6px;">
                    <button class="ewr-button-show" onclick="showResultSql(this)" title="View generated T-SQL query">
                        TSQL
                    </button>
                    <button class="ewr-button-show" onclick="exportTableToCSV(this)" title="Export to CSV">
                        &#11015;
                    </button>
                </div>
            </div>
            <div class="results-table-container">
                <table class="results-table" id="${tableId}">
                    <thead>
                        <tr>
    `;

    // Add headers
    result.columns.forEach(col => {
        tableHtml += `<th>${escapeHtml(col)}</th>`;
    });

    tableHtml += '</tr></thead><tbody>';

    // Add rows
    result.rows.forEach(row => {
        tableHtml += '<tr>';
        result.columns.forEach(col => {
            const value = row[col];
            const displayValue = value === null || value === undefined ? 'NULL' : value;
            tableHtml += `<td>${escapeHtml(String(displayValue))}</td>`;
        });
        tableHtml += '</tr>';
    });

    tableHtml += '</tbody></table></div></div>';

    return tableHtml;
}

/**
 * Export table data to a CSV file and trigger download
 * Extracts data from the results-table-wrapper data attribute
 * @param {HTMLElement} button - The export button element that was clicked
 */
export function exportTableToCSV(button) {
    // Get the results data from the parent wrapper
    const wrapper = button.closest('.results-table-wrapper');
    const resultsJson = wrapper.getAttribute('data-results');
    const result = JSON.parse(resultsJson);

    if (!result || !result.columns || !result.rows) {
        alert('No data to export');
        return;
    }

    const { columns, rows } = result;

    // Build CSV content
    let csv = columns.join(',') + '\n';

    rows.forEach(row => {
        const values = columns.map(col => {
            let value = row[col];
            if (value === null || value === undefined) {
                return '';
            }
            // Escape quotes and wrap in quotes if contains comma
            value = String(value).replace(/"/g, '""');
            if (value.includes(',') || value.includes('\n') || value.includes('"')) {
                return `"${value}"`;
            }
            return value;
        });
        csv += values.join(',') + '\n';
    });

    // Download
    const blob = new Blob([csv], { type: 'text/csv' });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `query_results_${Date.now()}.csv`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    window.URL.revokeObjectURL(url);
}

/**
 * Show SQL query in a modal/popup for the result
 * Displays the original natural language query and the generated SQL
 * Uses the EwrModal system for consistent styling
 * @param {HTMLElement} button - The button element that triggered the modal
 */
export function showResultSql(button) {
    let sql = '';
    let originalQuery = '';

    try {
        // First try to get from wrapper (results table case)
        const wrapper = button.closest('.results-table-wrapper');
        if (wrapper) {
            const resultsJson = wrapper.getAttribute('data-results');
            if (resultsJson) {
                const result = JSON.parse(resultsJson);
                sql = result.generatedSql || result.sql || '';
                originalQuery = result.naturalLanguage || result.originalQuery || '';
            }
        }

        // Fallback to button data attributes (standalone button case)
        if (!sql) {
            sql = button.getAttribute('data-sql') || '';
            originalQuery = button.getAttribute('data-query') || '';
        }

        // Decode HTML entities that may have been escaped
        sql = sql.replace(/&#39;/g, "'").replace(/&quot;/g, '"').replace(/&amp;/g, '&');
        originalQuery = originalQuery.replace(/&#39;/g, "'").replace(/&quot;/g, '"').replace(/&amp;/g, '&');

    } catch (parseError) {
        console.error('Error parsing SQL data:', parseError);
        sql = button.getAttribute('data-sql') || '';
        originalQuery = button.getAttribute('data-query') || '';
    }

    // Ensure EwrModal is available
    if (typeof window.EwrModal === 'undefined') {
        console.error('EwrModal not available');
        alert('Modal system not loaded. Please refresh the page.');
        return;
    }

    if (!sql) {
        window.EwrModal.display({
            title: 'Error',
            size: 'small',
            sections: [{ content: 'No SQL data available' }]
        });
        return;
    }

    // Store SQL for copy function
    window._currentSqlForCopy = sql;

    // Use EwrModal display (explicitly via window for module compatibility)
    window.EwrModal.display({
        title: 'T-SQL Query Details',
        size: 'large',
        sections: [
            {
                label: 'Original Question',
                content: originalQuery || '(No query recorded)',
                id: 'sqlPreviewOriginalQuery'
            },
            {
                label: 'Generated SQL',
                content: sql,
                isCode: true,
                id: 'sqlPreviewContent'
            }
        ],
        buttons: [
            {
                text: 'Copy SQL',
                class: 'primary',
                onClick: async () => {
                    try {
                        await navigator.clipboard.writeText(window._currentSqlForCopy);
                        // Brief visual feedback could be added here
                    } catch (e) {
                        // Fallback
                        const textArea = document.createElement('textarea');
                        textArea.value = window._currentSqlForCopy;
                        document.body.appendChild(textArea);
                        textArea.select();
                        document.execCommand('copy');
                        document.body.removeChild(textArea);
                    }
                }
            },
            { text: 'Close', class: 'secondary', close: true }
        ]
    });
}

/**
 * Copy SQL content from the preview modal to clipboard
 * Provides visual feedback on the copy button after successful copy
 * Includes fallback for older browsers that don't support the Clipboard API
 */
export async function copySqlToClipboard() {
    const sqlContent = document.getElementById('sqlPreviewContent');
    if (!sqlContent) return;

    const sql = sqlContent.textContent;
    const btnText = document.getElementById('copySqlBtnText');

    try {
        await navigator.clipboard.writeText(sql);
        // Show feedback
        if (btnText) {
            const originalText = btnText.textContent;
            btnText.textContent = 'Copied!';
            btnText.parentElement.style.background = '#10b981';
            setTimeout(() => {
                btnText.textContent = originalText;
                btnText.parentElement.style.background = '';
            }, 1500);
        }
    } catch (e) {
        // Fallback for older browsers
        const textArea = document.createElement('textarea');
        textArea.value = sql;
        textArea.style.position = 'fixed';
        textArea.style.left = '-9999px';
        document.body.appendChild(textArea);
        textArea.select();
        try {
            document.execCommand('copy');
            if (btnText) {
                const originalText = btnText.textContent;
                btnText.textContent = 'Copied!';
                setTimeout(() => btnText.textContent = originalText, 1500);
            }
        } catch (err) {
            alert('Failed to copy to clipboard');
        }
        document.body.removeChild(textArea);
    }
}
