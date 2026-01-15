// Global variables
let currentResults = null;
let currentSchema = null;

// Connection Management
async function testConnection() {
    const messageDiv = document.getElementById('connectionMessage');
    messageDiv.innerHTML = '';

    const connectionConfig = getConnectionConfig();

    try {
        const response = await fetch('/api/sql/test-connection', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(connectionConfig)
        });

        const result = await response.json();

        if (result.success) {
            showMessage(messageDiv, 'success', `✓ Connection successful! Server: ${result.serverVersion || 'Connected'}`);
        } else {
            showMessage(messageDiv, 'error', `✗ Connection failed: ${result.error}`);
        }
    } catch (error) {
        showMessage(messageDiv, 'error', `✗ Connection error: ${error.message}`);
    }
}

async function loadDatabases() {
    const messageDiv = document.getElementById('connectionMessage');
    const dbSelect = document.getElementById('database');
    messageDiv.innerHTML = '';

    const connectionConfig = getConnectionConfig();

    try {
        const response = await fetch('/api/sql/databases', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(connectionConfig)
        });

        const result = await response.json();

        if (result.success && result.databases) {
            // Clear and populate dropdown
            dbSelect.innerHTML = '<option value="">Select database...</option>';
            result.databases.forEach(db => {
                const option = document.createElement('option');
                option.value = db;
                option.textContent = db;
                dbSelect.appendChild(option);
            });
            showMessage(messageDiv, 'success', `✓ Loaded ${result.databases.length} databases`);
        } else {
            showMessage(messageDiv, 'error', `✗ Failed to load databases: ${result.error}`);
        }
    } catch (error) {
        showMessage(messageDiv, 'error', `✗ Error loading databases: ${error.message}`);
    }
}

// Query Execution
async function executeQuery() {
    const queryInput = document.getElementById('queryInput').value.trim();
    const database = document.getElementById('database').value;

    if (!queryInput) {
        showMessage(document.getElementById('errorMessage'), 'error', 'Please enter a query');
        return;
    }

    if (!database) {
        showMessage(document.getElementById('errorMessage'), 'error', 'Please select a database');
        return;
    }

    // Show loading state
    showLoading(true);
    hideResults();
    document.getElementById('errorMessage').innerHTML = '';
    document.getElementById('emptyState').classList.add('hidden');

    const connectionConfig = getConnectionConfig();

    try {
        const response = await fetch('/api/sql/query', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                ...connectionConfig,
                query: queryInput
            })
        });

        const result = await response.json();

        if (result.success) {
            currentResults = result;
            displayResults(result);

            // Show SQL if available
            if (result.generatedSql) {
                displayGeneratedSQL(result.generatedSql);
            }
        } else {
            showMessage(document.getElementById('errorMessage'), 'error', `Query failed: ${result.error}`);
        }
    } catch (error) {
        showMessage(document.getElementById('errorMessage'), 'error', `Error executing query: ${error.message}`);
    } finally {
        showLoading(false);
    }
}

// Display Results
function displayResults(data) {
    if (!data.columns || !data.rows || data.rows.length === 0) {
        showMessage(document.getElementById('errorMessage'), 'info', 'Query executed successfully but returned no results');
        return;
    }

    // Build table
    const table = document.getElementById('resultsTable');
    table.innerHTML = '';

    // Create header
    const thead = document.createElement('thead');
    const headerRow = document.createElement('tr');
    data.columns.forEach(col => {
        const th = document.createElement('th');
        th.textContent = col;
        headerRow.appendChild(th);
    });
    thead.appendChild(headerRow);
    table.appendChild(thead);

    // Create body
    const tbody = document.createElement('tbody');
    data.rows.forEach(row => {
        const tr = document.createElement('tr');
        data.columns.forEach(col => {
            const td = document.createElement('td');
            const value = row[col];
            td.textContent = value === null || value === undefined ? 'NULL' : value;
            tr.appendChild(td);
        });
        tbody.appendChild(tr);
    });
    table.appendChild(tbody);

    // Update row count
    document.getElementById('rowCount').textContent = `${data.rows.length} rows`;

    // Show results
    document.getElementById('resultsTableContainer').classList.remove('hidden');
    document.getElementById('resultsActions').classList.remove('hidden');
}

function displayGeneratedSQL(sql) {
    document.getElementById('sqlDisplay').textContent = sql;
    document.getElementById('sqlSection').classList.remove('hidden');

    // Expand by default
    const header = document.querySelector('.collapsible-header');
    const content = document.getElementById('sqlContent');
    header.classList.remove('collapsed');
    content.classList.remove('collapsed');
}

// Export to CSV
function exportToCSV() {
    if (!currentResults || !currentResults.rows || currentResults.rows.length === 0) {
        alert('No data to export');
        return;
    }

    const { columns, rows } = currentResults;

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

// Schema Browser
async function showSchema() {
    const database = document.getElementById('database').value;

    if (!database) {
        showMessage(document.getElementById('errorMessage'), 'error', 'Please select a database first');
        return;
    }

    // Show modal
    document.getElementById('schemaModal').classList.remove('hidden');
    document.getElementById('loadingSchema').classList.remove('hidden');
    document.getElementById('schemaMessage').innerHTML = '';
    document.getElementById('tableList').innerHTML = '';

    const connectionConfig = getConnectionConfig();

    try {
        const response = await fetch('/api/sql/schema', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(connectionConfig)
        });

        const result = await response.json();

        if (result.success && result.schema) {
            currentSchema = result.schema;
            displaySchema(result.schema);
        } else {
            showMessage(document.getElementById('schemaMessage'), 'error', `Failed to load schema: ${result.error}`);
        }
    } catch (error) {
        showMessage(document.getElementById('schemaMessage'), 'error', `Error loading schema: ${error.message}`);
    } finally {
        document.getElementById('loadingSchema').classList.add('hidden');
    }
}

function displaySchema(schema) {
    const tableList = document.getElementById('tableList');
    tableList.innerHTML = '';

    if (!schema || schema.length === 0) {
        tableList.innerHTML = '<p style="color: #6b7280; text-align: center; padding: 24px;">No tables found in this database</p>';
        return;
    }

    schema.forEach(table => {
        const tableItem = document.createElement('div');
        tableItem.className = 'table-item';

        // Table header
        const header = document.createElement('div');
        header.className = 'table-header';
        header.innerHTML = `
            <span><strong>${table.tableName}</strong> (${table.columns.length} columns)</span>
            <span style="cursor: pointer;">▼</span>
        `;
        header.onclick = () => toggleTableContent(tableItem);

        // Table content
        const content = document.createElement('div');
        content.className = 'table-content';
        content.style.display = 'none';

        table.columns.forEach(col => {
            const colDiv = document.createElement('div');
            colDiv.className = 'column-item';
            colDiv.innerHTML = `
                <span class="column-name">${col.columnName}</span>
                <span class="column-type">${col.dataType}</span>
                ${col.isNullable ? '<span style="color: #9ca3af; font-size: 11px;">(nullable)</span>' : '<span style="color: #10b981; font-size: 11px;">(required)</span>'}
            `;
            content.appendChild(colDiv);
        });

        tableItem.appendChild(header);
        tableItem.appendChild(content);
        tableList.appendChild(tableItem);
    });
}

function toggleTableContent(tableItem) {
    const content = tableItem.querySelector('.table-content');
    const arrow = tableItem.querySelector('.table-header span:last-child');

    if (content.style.display === 'none') {
        content.style.display = 'block';
        arrow.textContent = '▲';
    } else {
        content.style.display = 'none';
        arrow.textContent = '▼';
    }
}

function filterTables() {
    const searchTerm = document.getElementById('schemaSearch').value.toLowerCase();
    const tables = document.querySelectorAll('.table-item');

    tables.forEach(table => {
        const text = table.textContent.toLowerCase();
        if (text.includes(searchTerm)) {
            table.style.display = 'block';
        } else {
            table.style.display = 'none';
        }
    });
}

function closeSchemaModal(event) {
    if (event && event.target !== event.currentTarget) {
        return;
    }
    document.getElementById('schemaModal').classList.add('hidden');
}

// Sample Queries
function insertSampleQuery(query) {
    document.getElementById('queryInput').value = query;
}

// Clear Query
function clearQuery() {
    document.getElementById('queryInput').value = '';
    hideResults();
    document.getElementById('errorMessage').innerHTML = '';
    document.getElementById('emptyState').classList.remove('hidden');
}

// Toggle SQL Display
function toggleSqlDisplay() {
    const header = document.querySelector('.collapsible-header');
    const content = document.getElementById('sqlContent');

    header.classList.toggle('collapsed');
    content.classList.toggle('collapsed');
}

// Helper Functions
function getConnectionConfig() {
    return {
        server: document.getElementById('server').value,
        database: document.getElementById('database').value,
        username: document.getElementById('username').value,
        password: document.getElementById('password').value,
        trustServerCertificate: document.getElementById('trustCert').checked,
        encrypt: document.getElementById('encrypt').checked
    };
}

function showMessage(container, type, message) {
    container.innerHTML = `<div class="message ${type}">${message}</div>`;
}

function showLoading(show) {
    const spinner = document.getElementById('loadingSpinner');
    if (show) {
        spinner.classList.remove('hidden');
    } else {
        spinner.classList.add('hidden');
    }
}

function hideResults() {
    document.getElementById('resultsTableContainer').classList.add('hidden');
    document.getElementById('resultsActions').classList.add('hidden');
    document.getElementById('sqlSection').classList.add('hidden');
}

// Initialize on page load
document.addEventListener('DOMContentLoaded', () => {
    // Auto-load databases on page load
    const server = document.getElementById('server').value;
    const username = document.getElementById('username').value;

    if (server && username) {
        loadDatabases();
    }
});
