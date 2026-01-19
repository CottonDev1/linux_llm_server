/**
 * Schema Manager Module
 *
 * Handles all database schema-related operations for the SQL Chat application.
 * This includes loading databases, checking schema existence, extracting schemas,
 * and polling for schema analysis status.
 *
 * @module schema-manager
 */

import { SQL_API_BASE, state } from './sql-chat-state.js';
import { getConnectionConfig, disableConnectionFields } from './connection-manager.js';
import { showMessage, updateUIState, showErrorPopup, updateConnectionStatus } from './ui-manager.js';

/**
 * Load database schema - called by the "Load Database" button.
 * Triggers schema check for the selected database and updates UI state accordingly.
 *
 * @async
 * @returns {Promise<void>}
 */
export async function loadDatabase() {
    const database = document.getElementById('database').value;
    const schemaStatus = document.getElementById('schemaStatus');

    if (!database) {
        if (schemaStatus) {
            schemaStatus.textContent = 'Select a database first';
            schemaStatus.className = 'ewr-status-pill info visible';
        }
        return;
    }

    // Show loading status
    if (schemaStatus) {
        schemaStatus.textContent = 'Loading schema...';
        schemaStatus.className = 'ewr-status-pill info visible';
    }

    const connectionConfig = getConnectionConfig();

    try {
        const checkResponse = await fetch(`${SQL_API_BASE}/api/sql/schema/check`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(connectionConfig)
        });

        const checkResult = await checkResponse.json();

        if (checkResult.success && checkResult.exists) {
            // Schema exists - enable chat interface
            state.app.databaseLoaded = true;
            state.app.currentDatabase = database;
            state.app.schemaStats = {
                tables: checkResult.table_count || 0,
                procedures: checkResult.procedure_count || 0
            };

            if (schemaStatus) {
                schemaStatus.textContent = 'Schemas located';
                schemaStatus.className = 'ewr-status-pill success visible';
            }

            // Lock connection fields to prevent changes during active session
            disableConnectionFields(true);

            // Also disable all inputs/selects in the ewr-filter-panel
            const filterPanel = document.getElementById('connectionCollapsible');
            if (filterPanel) {
                filterPanel.querySelectorAll('input, select').forEach(el => {
                    el.disabled = true;
                });
            }

            console.log('Schema check: schemas found, connection fields locked');
            updateUIState();
        } else {
            // Schema doesn't exist - keep chat DISABLED until schema analysis is performed
            state.app.databaseLoaded = false;
            state.app.currentDatabase = database;
            state.app.schemaStats = null;

            if (schemaStatus) {
                schemaStatus.textContent = 'Schema analysis required';
                schemaStatus.className = 'ewr-status-pill warning visible';
            }

            // Show message in chat area explaining why it's disabled
            const chatInput = document.getElementById('chatInput');
            if (chatInput) {
                chatInput.placeholder = 'Schema analysis must be performed prior to querying database';
            }

            // Lock all inputs and selects in the filter panel
            disableConnectionFields(true);

            // Also disable all inputs/selects in the ewr-filter-panel
            const filterPanel = document.getElementById('connectionCollapsible');
            if (filterPanel) {
                filterPanel.querySelectorAll('input, select').forEach(el => {
                    el.disabled = true;
                });
            }

            // Show a message to the user
            showMessage(document.getElementById('connectionMessage'), 'warning',
                'Schema analysis must be performed prior to querying database. Use the Extract Schema feature.');

            console.log('Schema check: schemas NOT found, all fields disabled');
            updateUIState();
        }

    } catch (error) {
        state.app.databaseLoaded = false;
        if (schemaStatus) {
            schemaStatus.textContent = `Error: ${error.message}`;
            schemaStatus.className = 'ewr-status-pill error visible';
        }
        updateUIState();
    }
}

/**
 * Load list of available databases from the server.
 * Populates the database dropdown and auto-selects saved preference if available.
 *
 * @async
 * @returns {Promise<void>}
 */
export async function loadDatabases() {
    const dbSelect = document.getElementById('database');
    const connectionConfig = getConnectionConfig();

    try {
        const response = await fetch(`${SQL_API_BASE}/api/sql/list-databases`, {
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

            // Auto-select saved database preference if available
            if (window._savedDatabasePreference) {
                const savedDb = window._savedDatabasePreference;
                if (result.databases.includes(savedDb)) {
                    dbSelect.value = savedDb;
                    console.log('Auto-selected saved database:', savedDb);
                    // Dispatch change event to trigger schema check and button state update
                    dbSelect.dispatchEvent(new Event('change'));
                }
            }

            // Update button states after database list is loaded
            updateConnectionStatus();
        } else {
            // Show actual error message to user
            const errorMsg = result.error || result.detail || 'Failed to load database list';
            console.error('Failed to load databases:', errorMsg);
            showErrorPopup(errorMsg, 'Database List Error');
        }
    } catch (error) {
        // Show actual error message to user
        const errorMsg = error.message || 'Network error while loading databases';
        console.error('Error loading databases:', errorMsg);
        showErrorPopup(errorMsg, 'Database List Error');
    }
}

/**
 * Check if selected database schema exists in MongoDB.
 * Auto-called when database is selected from dropdown.
 * Does NOT trigger extraction - only checks existence.
 *
 * @async
 * @returns {Promise<void>}
 */
export async function checkDatabaseSchema() {
    const messageDiv = document.getElementById('connectionMessage');
    const database = document.getElementById('database').value;

    if (!database) {
        return;
    }

    // Show checking state
    messageDiv.innerHTML = '<div class="message info">Checking database schema...</div>';

    const connectionConfig = getConnectionConfig();

    try {
        // Check if schema exists in MongoDB (checkOnly mode)
        const checkResponse = await fetch(`${SQL_API_BASE}/api/sql/schema/check`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ ...connectionConfig, checkOnly: true })
        });

        const checkResult = await checkResponse.json();

        if (checkResult.success && checkResult.exists) {
            // Schema exists - enable chat interface
            state.app.databaseLoaded = true;
            state.app.currentDatabase = database;
            state.app.schemaStats = {
                tables: checkResult.tableCount || 0,
                procedures: checkResult.procedureCount || 0
            };

            showMessage(messageDiv, 'success',
                `Database schema loaded: ${checkResult.tableCount || 0} tables, ${checkResult.procedureCount || 0} procedures`
            );
            updateUIState();
        } else {
            // Schema doesn't exist - show error with instructions
            state.app.databaseLoaded = false;
            state.app.currentDatabase = null;
            state.app.schemaStats = null;

            showMessage(messageDiv, 'error',
                `Schema information not found for "${database}". Please run schema extraction from the Admin panel.`
            );
            updateUIState();
        }

    } catch (error) {
        state.app.databaseLoaded = false;
        showMessage(messageDiv, 'error', `Error checking database schema: ${error.message}`);
        updateUIState();
    }
}

/**
 * Extract schema with real-time progress via Server-Sent Events (SSE).
 * Displays a progress bar and status updates during extraction.
 *
 * @async
 * @param {Object} connectionConfig - Database connection configuration
 * @param {string} connectionConfig.server - SQL server hostname
 * @param {string} connectionConfig.database - Database name
 * @param {string} connectionConfig.user - Username for authentication
 * @param {string} connectionConfig.password - Password for authentication
 * @param {string} lookupKey - Unique key for schema storage/lookup
 * @param {HTMLElement} messageDiv - DOM element to display progress messages
 * @returns {Promise<void>} Resolves when extraction is complete
 */
export async function extractSchemaWithProgress(connectionConfig, lookupKey, messageDiv) {
    return new Promise((resolve, reject) => {
        // Create a progress display container
        let progressHtml = `
            <div class="message info" id="extractionProgress">
                <div style="font-weight: 600; margin-bottom: 8px;">Extracting Database Schema</div>
                <div id="progressMessage">Starting extraction...</div>
                <div id="progressBar" style="margin-top: 8px; height: 4px; background: #374151; border-radius: 2px; overflow: hidden;">
                    <div id="progressFill" style="height: 100%; background: #3b82f6; width: 0%; transition: width 0.3s;"></div>
                </div>
                <div id="progressStats" style="margin-top: 8px; font-size: 12px; color: #9ca3af;"></div>
            </div>
        `;
        messageDiv.innerHTML = progressHtml;

        // Use fetch with streaming response for SSE
        fetch(`${SQL_API_BASE}/api/sql/schema/extract-stream`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                ...connectionConfig,
                lookupKey: lookupKey,
                database: connectionConfig.database
            })
        }).then(async response => {
            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            let buffer = '';

            while (true) {
                const { done, value } = await reader.read();
                if (done) break;

                buffer += decoder.decode(value, { stream: true });
                const lines = buffer.split('\n');
                buffer = lines.pop(); // Keep incomplete line in buffer

                for (const line of lines) {
                    if (line.startsWith('event: ')) {
                        // Extract event type (currently unused but available for future use)
                        const eventType = line.substring(7).trim();
                        continue;
                    }
                    if (line.startsWith('data: ')) {
                        try {
                            const data = JSON.parse(line.substring(6));
                            handleExtractionProgress(data, messageDiv);
                        } catch (e) {
                            // Ignore parse errors for incomplete JSON
                        }
                    }
                }
            }

            resolve();
        }).catch(error => {
            showMessage(messageDiv, 'error', `Extraction failed: ${error.message}`);
            reject(error);
        });
    });
}

/**
 * Handle extraction progress events from SSE stream.
 * Updates progress bar, stats, and completion state.
 *
 * @param {Object} data - Progress event data from SSE
 * @param {string} [data.message] - Status message to display
 * @param {number} [data.current] - Current progress count
 * @param {number} [data.total] - Total items to process
 * @param {string} [data.phase] - Current extraction phase (tables, procedures, processing, stored_procedures)
 * @param {number} [data.count] - Count for current phase
 * @param {number} [data.tableCount] - Final table count (on completion)
 * @param {number} [data.procedureCount] - Final procedure count (on completion)
 * @param {HTMLElement} messageDiv - DOM element to display progress messages
 */
export function handleExtractionProgress(data, messageDiv) {
    const progressMessage = document.getElementById('progressMessage');
    const progressFill = document.getElementById('progressFill');
    const progressStats = document.getElementById('progressStats');

    if (!progressMessage) return;

    // Update message
    if (data.message) {
        progressMessage.textContent = data.message;
    }

    // Update progress bar
    if (data.total && data.current !== undefined) {
        const percent = Math.round((data.current / data.total) * 100);
        progressFill.style.width = `${percent}%`;
    }

    // Update stats
    if (data.phase === 'tables' && data.count) {
        progressStats.innerHTML = `Tables: ${data.count}`;
    } else if (data.phase === 'procedures' && data.count) {
        progressStats.innerHTML += ` | Procedures: ${data.count}`;
    } else if (data.phase === 'processing' && data.total && data.current) {
        progressStats.innerHTML = `Tables: ${data.current}/${data.total}`;
    } else if (data.phase === 'stored_procedures' && data.total && data.current) {
        progressStats.innerHTML = `Processing procedures: ${data.current}/${data.total}`;
    }

    // Handle completion
    if (data.tableCount !== undefined && data.procedureCount !== undefined) {
        // Extraction complete
        state.app.databaseLoaded = true;
        state.app.currentDatabase = document.getElementById('database').value;
        state.app.schemaStats = {
            tables: data.tableCount,
            procedures: data.procedureCount
        };

        messageDiv.innerHTML = `
            <div class="message success">
                Schema extraction complete!<br>
                <strong>${data.tableCount} tables</strong>, <strong>${data.procedureCount} stored procedures</strong><br>
                <span style="font-size: 12px; color: #9ca3af;">Ready for intelligent queries</span>
            </div>
        `;
        updateUIState();
    }

    // Handle errors
    if (data.message && data.message.includes('failed')) {
        showMessage(messageDiv, 'error', data.message);
    }
}

/**
 * Poll for schema analysis completion status.
 * Continues polling every 5 seconds until analysis is complete or timeout (2 minutes).
 *
 * @async
 * @param {string} database - Name of the database being analyzed
 * @param {HTMLElement} messageDiv - DOM element to display status messages
 * @param {number} [attempts=0] - Current polling attempt count (internal use)
 * @returns {Promise<void>}
 */
export async function pollSchemaAnalysisStatus(database, messageDiv, attempts = 0) {
    if (attempts >= 24) { // Stop after 2 minutes (24 * 5 seconds)
        showMessage(messageDiv, 'info',
            'Schema analysis is still running. You can start querying, but wait for better results.'
        );
        return;
    }

    setTimeout(async () => {
        try {
            const connectionConfig = getConnectionConfig();
            // IMPORTANT: Use checkOnly to prevent spawning duplicate extraction processes
            const response = await fetch(`${SQL_API_BASE}/api/sql/schema/check`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    ...connectionConfig,
                    checkOnly: true  // Only check status, don't trigger new analysis
                })
            });

            const result = await response.json();

            if (result.exists) {
                // Analysis complete! Update state with schema stats
                state.app.schemaStats = {
                    tables: result.tableCount || 0,
                    procedures: result.procedureCount || 0
                };

                showMessage(messageDiv, 'success',
                    `Database schema analysis complete! (${result.tableCount} tables, ${result.procedureCount} procedures)\n\n` +
                    `Ready for intelligent queries!`
                );
                updateUIState();
            } else if (result.analyzing) {
                // Still analyzing, continue polling
                showMessage(messageDiv, 'info',
                    `Schema analysis in progress... (${attempts + 1} checks)\n\n` +
                    `Analyzing database structure and stored procedures...`
                );
                pollSchemaAnalysisStatus(database, messageDiv, attempts + 1);
            } else {
                // Not analyzing and doesn't exist - something went wrong
                showMessage(messageDiv, 'error',
                    'Schema analysis may have failed. Please try clicking "Load Database" again.'
                );
            }
        } catch (error) {
            // Continue polling even on error
            pollSchemaAnalysisStatus(database, messageDiv, attempts + 1);
        }
    }, 5000); // Check every 5 seconds
}
