/**
 * @deprecated This file is superseded by the modular architecture in ./modules/
 * The HTML page loads modules/index.js which imports from individual modules.
 * This file is kept for reference only and is NOT loaded by index.html.
 *
 * For connection settings, see: modules/connection-manager.js
 * For main entry point, see: modules/index.js
 */

// API Configuration - Python service for SQL pipeline
const SQL_API_BASE = 'http://localhost:8001';  // Python FastAPI service

// Global variables
let chatHistory = [];
let currentSchema = null;
let isProcessing = false;
let conversationContextEnabled = true; // Always enabled, checkbox removed

// State management
let appState = {
    connectionTested: false,
    databaseLoaded: false,
    currentDatabase: null,
    schemaStats: null
};

// Session metrics tracking
let sessionMetrics = {
    totalQueries: 0,
    successfulQueries: 0,
    totalRows: 0,
    totalTokens: 0,
    responseTimes: [],
    currentQueryStartTime: null
};

// Token usage tracker
let tokenTracker = {
    currentTokens: 0,
    maxTokens: 8192,
    isLimitReached: false,
    promptTokens: 0,
    completionTokens: 0,
    totalTokens: 0
};

// Pipeline steps configuration - matches backend stages
const PIPELINE_STEPS = [
    { id: 'preprocessing', label: 'Analyzing question', icon: '1' },
    { id: 'security', label: 'Checking security', icon: '2' },
    { id: 'cache', label: 'Checking cache', icon: '3' },
    { id: 'rules', label: 'Rules', icon: '★', isRulesStep: true },
    { id: 'schema', label: 'Loading schema', icon: '5' },
    { id: 'generating', label: 'Generating SQL', icon: '6' },
    { id: 'fixing', label: 'Applying fixes', icon: '7' },
    { id: 'validating', label: 'Validating SQL', icon: '8' },
    { id: 'executing', label: 'Executing query', icon: '9' },
    { id: 'complete', label: 'Complete', icon: '✓' }
];

// Conversation context is always enabled (checkbox removed)
// Keeping these functions for backward compatibility
function loadConversationContextSetting() {
    conversationContextEnabled = true;
}

function toggleConversationContext() {
    conversationContextEnabled = true;
}

// Get conversation history for AI context (limited to recent messages)
function getConversationForContext() {
    if (!conversationContextEnabled || chatHistory.length === 0) {
        return null;
    }

    // Get max messages from admin settings (localStorage) or use default
    const savedMax = localStorage.getItem('sqlMaxConversationMessages');
    const maxMessages = savedMax ? parseInt(savedMax) : 6;

    // Get the last N messages (user questions and assistant responses)
    const recentHistory = chatHistory.slice(-maxMessages);

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

// Save connection settings to localStorage (per-user)
async function saveConnectionSettings() {
    const userStr = localStorage.getItem('user');
    const userId = userStr ? JSON.parse(userStr).id || 'default' : 'default';
    const connectionMessage = document.getElementById('connectionMessage');

    // Get password and encrypt it
    const password = document.getElementById('password').value;
    let encryptedPassword = null;

    if (password && window.CryptoUtils && CryptoUtils.isSupported()) {
        try {
            encryptedPassword = await CryptoUtils.encrypt(password, userId);
            console.log('Password encrypted successfully');
        } catch (e) {
            console.error('Failed to encrypt password:', e);
        }
    }

    const settings = {
        server: document.getElementById('server').value,
        database: document.getElementById('database').value,
        authType: document.getElementById('authType').value,
        domain: document.getElementById('domain').value,
        username: document.getElementById('username').value,
        encryptedPassword: encryptedPassword, // Encrypted password
        trustCert: document.getElementById('trustCert').checked,
        encrypt: document.getElementById('encrypt').checked
    };

    localStorage.setItem(`sqlConnectionSettings_${userId}`, JSON.stringify(settings));

    // Show success message
    const originalHtml = connectionMessage.innerHTML;
    connectionMessage.innerHTML = '<div class="message success">✓ Connection settings saved (including encrypted password)</div>';
    setTimeout(() => {
        connectionMessage.innerHTML = originalHtml;
    }, 3000);

    console.log('Connection settings saved for user:', userId);
}

// Load saved connection settings from localStorage
async function loadSavedConnectionSettings() {
    const userStr = localStorage.getItem('user');
    const userId = userStr ? JSON.parse(userStr).id || 'default' : 'default';

    const savedSettings = localStorage.getItem(`sqlConnectionSettings_${userId}`);
    if (savedSettings) {
        try {
            const settings = JSON.parse(savedSettings);

            if (settings.server) document.getElementById('server').value = settings.server;
            if (settings.authType) {
                document.getElementById('authType').value = settings.authType;
                toggleAuthFields(); // Update UI based on auth type
            }
            if (settings.domain) document.getElementById('domain').value = settings.domain;
            if (settings.username) document.getElementById('username').value = settings.username;
            if (typeof settings.trustCert === 'boolean') document.getElementById('trustCert').checked = settings.trustCert;
            if (typeof settings.encrypt === 'boolean') document.getElementById('encrypt').checked = settings.encrypt;

            // Decrypt and populate password if available
            if (settings.encryptedPassword && window.CryptoUtils && CryptoUtils.isSupported()) {
                try {
                    const decryptedPassword = await CryptoUtils.decrypt(settings.encryptedPassword, userId);
                    if (decryptedPassword) {
                        document.getElementById('password').value = decryptedPassword;
                        console.log('Password decrypted and loaded successfully');
                    }
                } catch (e) {
                    console.error('Failed to decrypt password:', e);
                    // Password field remains empty - user will need to re-enter
                }
            }

            // Note: Database will be loaded after connection test
            // Store database preference for after connection
            if (settings.database) {
                window._savedDatabasePreference = settings.database;
            }

            console.log('Loaded saved connection settings for user:', userId);
        } catch (e) {
            console.error('Failed to load saved connection settings:', e);
        }
    }
}

// Validate connection fields (Connect button always stays enabled)
function validateConnectionFields() {
    // Connect button is always enabled - validation happens on click
}

// Initialize on page load
document.addEventListener('DOMContentLoaded', async () => {
    // DO NOT auto-load databases - require explicit connection test
    // Initialize UI in disabled state
    updateUIState();

    // Load saved connection settings first (async for password decryption)
    await loadSavedConnectionSettings();

    // Load conversation context setting
    loadConversationContextSetting();

    // Initialize remaining context window display
    initializeContextRemainingDisplay();

    // Auto-resize textarea
    const chatInput = document.getElementById('chatInput');
    chatInput.addEventListener('input', () => {
        autoResizeTextarea(chatInput);
    });

    // Add validation listeners to connection fields
    const connectionFields = ['server', 'username', 'password', 'authType', 'domain'];
    connectionFields.forEach(fieldId => {
        const field = document.getElementById(fieldId);
        if (field) {
            field.addEventListener('input', validateConnectionFields);
            field.addEventListener('change', validateConnectionFields);
        }
    });

    // Initial validation
    validateConnectionFields();

    // Add event listener to database dropdown - just log selection, no auto-load
    const databaseSelect = document.getElementById('database');
    databaseSelect.addEventListener('change', () => {
        const selectedDb = databaseSelect.value;
        console.log('Database dropdown changed:', selectedDb);
        // User must click Load DB button to load the database
    });
});

// UI State Management
function updateUIState() {
    const chatInput = document.getElementById('chatInput');
    const sendBtn = document.getElementById('sendBtn');
    const saveBtn = document.querySelector('.btn-save');

    // Update chat interface state
    const chatEnabled = appState.databaseLoaded;
    chatInput.disabled = !chatEnabled;
    sendBtn.disabled = !chatEnabled || isProcessing;

    // Update save button state - only enable when connected
    if (saveBtn) {
        saveBtn.disabled = !appState.connectionTested;
        if (!appState.connectionTested) {
            saveBtn.title = 'Connect to server first to enable saving';
        } else {
            saveBtn.title = 'Save connection settings';
        }
    }

    // Update Add Example button state - only enable when database is loaded
    const addExampleBtn = document.getElementById('addExampleBtn');
    if (addExampleBtn) {
        addExampleBtn.disabled = !appState.databaseLoaded;
    }

    // Update Add Rule button state - only enable when database is loaded
    const addRuleBtn = document.getElementById('addRuleBtn');
    if (addRuleBtn) {
        addRuleBtn.disabled = !appState.databaseLoaded;
    }

    // Update Load Database button state - stay pressed when database is loaded
    const loadDatabaseBtn = document.getElementById('loadDatabaseBtn');
    if (loadDatabaseBtn) {
        if (appState.databaseLoaded) {
            loadDatabaseBtn.classList.add('connected');
            loadDatabaseBtn.textContent = 'DB Loaded';
        } else {
            loadDatabaseBtn.classList.remove('connected');
            loadDatabaseBtn.textContent = 'Load DB';
        }
    }

    // Update placeholder text
    if (!appState.databaseLoaded) {
        chatInput.placeholder = 'Please connect to server and select a database to begin';
    } else {
        chatInput.placeholder = 'Ask a question about your data...';
    }

    // Update status indicator in connection panel
    updateConnectionStatus();
}

function updateConnectionStatus() {
    const connectionMessage = document.getElementById('connectionMessage');
    const connectionStatusInline = document.getElementById('connectionStatusInline');
    const statusContainer = document.getElementById('statusMessageContainer');
    const databaseSelect = document.getElementById('database');
    const loadDatabaseBtn = document.getElementById('loadDatabaseBtn');

    // Update inline status
    if (appState.connectionTested) {
        connectionStatusInline.textContent = '✓ Connected';
        connectionStatusInline.className = 'ewr-status-pill success';

        // Show the status container
        if (statusContainer) {
            statusContainer.classList.add('visible');
        }

        // Enable database dropdown after connection
        databaseSelect.disabled = false;

        // Enable Load Database button when connected (validation happens on click)
        if (loadDatabaseBtn) {
            loadDatabaseBtn.disabled = false;
        }
    } else {
        connectionStatusInline.textContent = '';

        // Hide the status container if no other message
        if (statusContainer && !connectionMessage.textContent.trim()) {
            statusContainer.classList.remove('visible');
        }

        // Disable database dropdown until connected
        databaseSelect.disabled = true;

        // Disable Load Database button
        if (loadDatabaseBtn) {
            loadDatabaseBtn.disabled = true;
        }
    }

    // Update connection message area with schema info
    if (appState.databaseLoaded && appState.schemaStats) {
        connectionMessage.textContent = `DB: ${appState.currentDatabase} (${appState.schemaStats.tables || 0} tables)`;
        connectionMessage.className = 'ewr-status-pill success';
        if (statusContainer) {
            statusContainer.classList.add('visible');
        }
    }
}

// Panel Toggle
function togglePanel() {
    const panel = document.getElementById('leftPanel');
    const btn = document.getElementById('toggleBtn');

    panel.classList.toggle('collapsed');
    btn.textContent = panel.classList.contains('collapsed') ? '▶' : '◀';
}

// Connection Management
async function testConnection() {
    const messageDiv = document.getElementById('connectionMessage');
    const testBtn = document.querySelector('button[onclick="testConnection()"]');
    const dbSelect = document.getElementById('database');
    messageDiv.innerHTML = '';

    const connectionConfig = getConnectionConfig();
    // Use 'master' as default database for connection testing if no database selected
    if (!connectionConfig.database) {
        connectionConfig.database = 'master';
    }

    // Debug: Log the exact config being sent
    console.log('testConnection - sending config:', JSON.stringify(connectionConfig, null, 2));
    console.log('testConnection - domain:', connectionConfig.domain || '(not set)');
    console.log('testConnection - user:', connectionConfig.user || '(not set)');
    console.log('testConnection - integratedAuth:', connectionConfig.integratedAuth);
    console.log('testConnection - authType:', connectionConfig.authType);

    // Show loading state
    testBtn.disabled = true;
    testBtn.textContent = 'Connecting...';
    showMessage(messageDiv, 'info', 'Connecting to server...');

    try {
        const response = await fetch(`${SQL_API_BASE}/api/sql/test-connection`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(connectionConfig)
        });

        const result = await response.json();

        if (result.success) {
            // Update state
            appState.connectionTested = true;
            console.log('Connection successful, appState.connectionTested =', appState.connectionTested);
            // Clear any error message on success and show connected status
            hideConnectionMessage();
            const connectionStatusInline = document.getElementById('connectionStatusInline');
            if (connectionStatusInline) {
                connectionStatusInline.textContent = `✅ Connected to ${connectionConfig.server}`;
                connectionStatusInline.className = 'ewr-status-pill success visible';
            }

            // Populate database dropdown from response (no separate call needed)
            if (result.databases && result.databases.length > 0) {
                dbSelect.innerHTML = '<option value="">Select database...</option>';
                result.databases.forEach(db => {
                    const option = document.createElement('option');
                    option.value = db;
                    option.textContent = db;
                    dbSelect.appendChild(option);
                });
                console.log(`Loaded ${result.databases.length} databases from connection response`);

                // Auto-select saved database preference if available
                if (window._savedDatabasePreference) {
                    const savedDb = window._savedDatabasePreference;
                    if (result.databases.includes(savedDb)) {
                        dbSelect.value = savedDb;
                        console.log('Auto-selected saved database:', savedDb);
                        // Dispatch change event to trigger schema check
                        dbSelect.dispatchEvent(new Event('change'));
                    }
                }
            }

            // Enable Load DB button immediately after successful connection
            const loadDatabaseBtn = document.getElementById('loadDatabaseBtn');
            if (loadDatabaseBtn) {
                loadDatabaseBtn.disabled = false;
                console.log('Load DB button ENABLED after successful connection');
            }

            // Disable connection fields and button after successful connection
            // User must click Clear to change connection
            disableConnectionFields(true);
            testBtn.disabled = true;
            testBtn.textContent = 'Connected';
            testBtn.classList.add('connected');
        } else {
            appState.connectionTested = false;
            // Show error in popup modal with full details
            const errorMessage = result.error || result.message || 'Unknown error';
            const errorDetails = result.details || null;
            const connectionInfo = result.connection_info || null;
            showErrorPopup(errorMessage, 'Connection Failed', errorDetails, connectionInfo);
            testBtn.disabled = false;
            testBtn.textContent = 'Connect to Server';
        }
    } catch (error) {
        appState.connectionTested = false;
        // Show error in popup modal
        showErrorPopup(error.message, 'Connection Error', 'Network or service error - ensure Python service is running on port 8001');
        testBtn.disabled = false;
        testBtn.textContent = 'Connect to Server';
    } finally {
        updateUIState();
    }
}

/**
 * Helper to disable/enable connection form fields
 * Called when connected (disable) or cleared (enable)
 */
function disableConnectionFields(disable) {
    const fields = ['server', 'authType', 'domain', 'username', 'password', 'trustCert', 'encrypt'];
    fields.forEach(id => {
        const el = document.getElementById(id);
        if (el) el.disabled = disable;
    });
}

/**
 * Load database schema - called by the "Load Database" button
 * Triggers schema check for the selected database
 */
async function loadDatabase() {
    const database = document.getElementById('database').value;
    const schemaStatus = document.getElementById('schemaStatus');

    if (!database) {
        if (schemaStatus) {
            schemaStatus.textContent = '⚠️ Select a database first';
            schemaStatus.className = 'ewr-status-pill info visible';
        }
        return;
    }

    // Show loading status
    if (schemaStatus) {
        schemaStatus.textContent = '⏳ Loading schema...';
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
            appState.databaseLoaded = true;
            appState.currentDatabase = database;
            appState.schemaStats = {
                tables: checkResult.table_count || 0,
                procedures: checkResult.procedure_count || 0
            };

            if (schemaStatus) {
                schemaStatus.textContent = '✅ Schemas located';
                schemaStatus.className = 'ewr-status-pill success visible';
            }
            console.log('Schema check: schemas found, status updated');
            updateUIState();
        } else {
            // Schema doesn't exist - still enable chat but show warning
            appState.databaseLoaded = true;
            appState.currentDatabase = database;
            appState.schemaStats = null;

            if (schemaStatus) {
                schemaStatus.textContent = '⚠️ Schemas not found';
                schemaStatus.className = 'ewr-status-pill warning visible';
            }
            console.log('Schema check: schemas NOT found, warning shown');
            updateUIState();
        }

    } catch (error) {
        appState.databaseLoaded = false;
        if (schemaStatus) {
            schemaStatus.textContent = `❌ Error: ${error.message}`;
            schemaStatus.className = 'ewr-status-pill error visible';
        }
        updateUIState();
    }
}

async function loadDatabases() {
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
 * Check if selected database schema exists in MongoDB
 * Auto-called when database is selected from dropdown
 * Does NOT trigger extraction - only checks existence
 */
async function checkDatabaseSchema() {
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
            appState.databaseLoaded = true;
            appState.currentDatabase = database;
            appState.schemaStats = {
                tables: checkResult.tableCount || 0,
                procedures: checkResult.procedureCount || 0
            };

            showMessage(messageDiv, 'success',
                `✅ Database schema loaded: ${checkResult.tableCount || 0} tables, ${checkResult.procedureCount || 0} procedures`
            );
            updateUIState();
        } else {
            // Schema doesn't exist - show error with instructions
            appState.databaseLoaded = false;
            appState.currentDatabase = null;
            appState.schemaStats = null;

            showMessage(messageDiv, 'error',
                `Schema information not found for "${database}". Please run schema extraction from the Admin panel.`
            );
            updateUIState();
        }

    } catch (error) {
        appState.databaseLoaded = false;
        showMessage(messageDiv, 'error', `Error checking database schema: ${error.message}`);
        updateUIState();
    }
}

/**
 * Extract schema with real-time progress via SSE
 */
async function extractSchemaWithProgress(connectionConfig, lookupKey, messageDiv) {
    return new Promise((resolve, reject) => {
        // Create a progress display container
        let progressHtml = `
            <div class="message info" id="extractionProgress">
                <div style="font-weight: 600; margin-bottom: 8px;">⏳ Extracting Database Schema</div>
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
                        // Extract event type
                        const eventType = line.substring(7).trim();
                        continue;
                    }
                    if (line.startsWith('data: ')) {
                        try {
                            const data = JSON.parse(line.substring(6));
                            handleExtractionProgress(data, messageDiv);
                        } catch (e) {
                            // Ignore parse errors
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
 * Handle extraction progress events
 */
function handleExtractionProgress(data, messageDiv) {
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
        appState.databaseLoaded = true;
        appState.currentDatabase = document.getElementById('database').value;
        appState.schemaStats = {
            tables: data.tableCount,
            procedures: data.procedureCount
        };

        messageDiv.innerHTML = `
            <div class="message success">
                ✅ Schema extraction complete!<br>
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
 * Poll for schema analysis completion status
 */
async function pollSchemaAnalysisStatus(database, messageDiv, attempts = 0) {
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
                appState.schemaStats = {
                    tables: result.tableCount || 0,
                    procedures: result.procedureCount || 0
                };

                showMessage(messageDiv, 'success',
                    `✅ Database schema analysis complete! (${result.tableCount} tables, ${result.procedureCount} procedures)\n\n` +
                    `Ready for intelligent queries!`
                );
                updateUIState();
            } else if (result.analyzing) {
                // Still analyzing, continue polling
                showMessage(messageDiv, 'info',
                    `⏳ Schema analysis in progress... (${attempts + 1} checks)\n\n` +
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

// Chat Functions
function sendSampleQuery(query) {
    document.getElementById('chatInput').value = query;
    sendMessage();
}

function handleKeyDown(event) {
    // Send on Enter, new line on Shift+Enter
    if (event.key === 'Enter' && !event.shiftKey) {
        event.preventDefault();
        sendMessage();
    }
}

async function sendMessage() {
    const input = document.getElementById('chatInput');
    const message = input.value.trim();

    if (!message || isProcessing) {
        return;
    }

    // Collapse status messages when sending a chat message
    if (typeof collapseStatusMessage === 'function') {
        collapseStatusMessage();
    }

    // Check if database is loaded
    if (!appState.databaseLoaded) {
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
    isProcessing = true;
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
                    updateTokenTracker(result.tokenUsage);
                }

                // Update timing metrics display
                if (result.timing) {
                    updateTimingDisplay(result.timing);
                }
            } else {
                // Add error message with stage info for timeouts
                let errorMsg = result.error || 'Query failed';
                if (result.isTimeout && result.stage) {
                    errorMsg = `⏱️ Timeout during "${result.stage}"\n\n${result.details || ''}`;
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
        isProcessing = false;
        updateSendButton();
    }
}

function addUserMessage(text) {
    const chatMessages = document.getElementById('chatMessages');

    const messageDiv = document.createElement('div');
    messageDiv.className = 'chat-message user';

    messageDiv.innerHTML = `
        <div class="message-avatar user-avatar">You</div>
        <div class="message-content">
            <div class="message-text">${escapeHtml(text)}</div>
        </div>
    `;

    chatMessages.appendChild(messageDiv);
    scrollToBottom();

    // Add to history
    const time = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    chatHistory.push({ type: 'user', text, time });
    updateTokenTrackerFromHistory();
}

function addAssistantMessage(result, originalQuery) {
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

    messageDiv.innerHTML = `
        <div class="message-avatar assistant-avatar">EWR</div>
        <div class="message-content">
            <div class="message-text">
                ${responseText}
                ${hasResults ? createResultsTable(resultWithQuery) : ''}
            </div>
            <div id="${feedbackContainerId}" class="feedback-buttons" style="margin-top: 12px; border-top: 1px solid #334155; padding-top: 12px;">
                <span class="feedback-label">Was this helpful?</span>
                <button class="feedback-btn-icon" data-message-id="${messageId}" data-feedback="positive" title="Thumbs up">👍</button>
                <button class="feedback-btn-icon" data-message-id="${messageId}" data-feedback="negative" title="Thumbs down">👎</button>
            </div>
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
    chatHistory.push({
        type: 'assistant',
        text: responseText,
        sql: result.generatedSql,
        results: result,
        time
    });
    updateTokenTrackerFromHistory();
}

function addErrorMessage(errorText, failedSql = null, originalQuestion = null, aiExplanation = null) {
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
                    <span style="font-size: 18px;">🤖</span>
                    <span style="font-weight: 600; color: #60a5fa;">What happened</span>
                </div>
                <div style="color: #e2e8f0; line-height: 1.6; font-size: 14px;">
                    ${escapeHtml(aiExplanation)}
                </div>
            </div>
        `;
    }

    // Get the original question from chat history if not provided
    const question = originalQuestion || (chatHistory.length > 0 ?
        chatHistory.filter(h => h.type === 'user').pop()?.text : '');

    let sqlBlock = '';
    if (failedSql) {
        sqlBlock = `
            <div class="sql-block" style="margin-top: 12px;">
                <div class="sql-header" style="color: #ef4444;">
                    <span>❌</span>
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
        <div class="message-avatar assistant-avatar" style="background: #ef4444;">!</div>
        <div class="message-content">
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
                        <strong style="color: #3b82f6;">💡 Suggestion:</strong> ${formattedError.suggestion}
                    </div>
                ` : ''}
            </div>
            ${sqlBlock}
            <div id="${feedbackContainerId}" class="feedback-buttons" style="margin-top: 12px; border-top: 1px solid #334155; padding-top: 12px;">
                <span class="feedback-label">Was this helpful?</span>
                <button class="feedback-btn-icon" data-message-id="${errorId}" data-feedback="positive" title="Thumbs up">👍</button>
                <button class="feedback-btn-icon" data-message-id="${errorId}" data-feedback="negative" title="Thumbs down">👎</button>
            </div>
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
    chatHistory.push({ type: 'error', text: errorText, sql: failedSql, question, time });
    updateTokenTrackerFromHistory();
}

/**
 * Format error message to be more user-friendly and actionable
 * Analyzes the error and provides context and suggestions
 *
 * RAG Architecture: This helps users understand SQL errors by providing
 * semantic error classification and actionable remediation steps
 */
function formatErrorMessage(errorText) {
    const errorLower = errorText.toLowerCase();

    // Invalid column name errors
    if (errorLower.includes('invalid column name')) {
        const columnMatch = errorText.match(/invalid column name '([^']+)'/i);
        const columnName = columnMatch ? columnMatch[1] : 'unknown';

        return {
            title: '❌ Invalid Column Name',
            message: `The column "${columnName}" does not exist in the selected table.`,
            suggestion: `Check the table schema or try rephrasing your query. Use "Show me all columns in [table name]" to see available columns.`
        };
    }

    // Invalid object name errors (table doesn't exist)
    if (errorLower.includes('invalid object name')) {
        const tableMatch = errorText.match(/invalid object name '([^']+)'/i);
        const tableName = tableMatch ? tableMatch[1] : 'unknown';

        return {
            title: '❌ Table Not Found',
            message: `The table "${tableName}" does not exist in the database.`,
            suggestion: `Verify the table name is correct. Try asking "What tables are available?" or check the schema in the connection panel.`
        };
    }

    // Syntax errors
    if (errorLower.includes('incorrect syntax') || errorLower.includes('syntax error')) {
        return {
            title: '❌ SQL Syntax Error',
            message: `The generated SQL query has a syntax error: ${errorText}`,
            suggestion: `Try rephrasing your question more clearly, or check if you meant to query a different table or column.`
        };
    }

    // Ambiguous column name
    if (errorLower.includes('ambiguous column name') || errorLower.includes('ambiguous')) {
        const columnMatch = errorText.match(/column name '([^']+)'/i);
        const columnName = columnMatch ? columnMatch[1] : 'unknown';

        return {
            title: '❌ Ambiguous Column Reference',
            message: `The column "${columnName}" exists in multiple tables and needs to be specified with a table prefix.`,
            suggestion: `Be more specific about which table you want to query, or ask about specific tables separately.`
        };
    }

    // Permission/authentication errors
    if (errorLower.includes('permission') || errorLower.includes('denied') || errorLower.includes('login failed')) {
        return {
            title: '🔒 Authentication/Permission Error',
            message: errorText,
            suggestion: `Check your database credentials in the connection panel. Verify you have the correct username, password, and permissions for this database.`
        };
    }

    // Conversion errors
    if (errorLower.includes('conversion') || errorLower.includes('cast')) {
        return {
            title: '❌ Data Type Conversion Error',
            message: errorText,
            suggestion: `There's a data type mismatch in the query. Try being more specific about the data type you're looking for (numbers, dates, text, etc.).`
        };
    }

    // Connection errors
    if (errorLower.includes('connection') || errorLower.includes('timeout') || errorLower.includes('network')) {
        return {
            title: '🔌 Connection Error',
            message: errorText,
            suggestion: `The database connection failed. Check if the server is accessible and your connection settings are correct. Try "Test Connection" first.`
        };
    }

    // Aggregate function errors
    if (errorLower.includes('aggregate') || errorLower.includes('group by')) {
        return {
            title: '❌ Aggregate Function Error',
            message: errorText,
            suggestion: `When using aggregate functions (COUNT, SUM, AVG, etc.), all non-aggregated columns must be in the GROUP BY clause. Try rephrasing to be clearer about what you want to aggregate.`
        };
    }

    // Foreign key constraint errors
    if (errorLower.includes('foreign key') || errorLower.includes('reference constraint')) {
        return {
            title: '❌ Foreign Key Constraint Violation',
            message: errorText,
            suggestion: `The operation violates a foreign key relationship. Ensure referenced records exist in the related table before inserting or updating.`
        };
    }

    // Generic error - show the full error text
    return {
        title: '❌ Query Error',
        message: errorText,
        suggestion: `Review the generated SQL query and your question. Try rephrasing your request more specifically, or ask about the table structure first.`
    };
}

function toggleErrorDetails(errorId) {
    const errorBlock = document.getElementById(errorId);
    if (errorBlock) {
        if (errorBlock.style.display === 'none') {
            errorBlock.style.display = 'block';
        } else {
            errorBlock.style.display = 'none';
        }
    }
}

function addLoadingMessage() {
    const chatMessages = document.getElementById('chatMessages');

    const loadingDiv = document.createElement('div');
    loadingDiv.className = 'chat-message assistant';
    const loadingId = `loading-${Date.now()}`;
    loadingDiv.id = loadingId;

    // Loading indicator with step and detail lines
    loadingDiv.innerHTML = `
        <div class="message-avatar assistant-avatar">EWR</div>
        <div class="message-content">
            <div class="loading-indicator">
                <div class="spinner"></div>
                <div class="loading-text-container">
                    <span class="loading-text" id="${loadingId}-text">Step 1/${PIPELINE_STEPS.length} - ${PIPELINE_STEPS[0].label}...</span>
                    <span class="loading-detail" id="${loadingId}-detail"></span>
                </div>
            </div>
        </div>
    `;

    // Store start time for metrics
    sessionMetrics.currentQueryStartTime = Date.now();

    chatMessages.appendChild(loadingDiv);
    scrollToBottom();

    return loadingId;
}

function updateLoadingMessage(loadingId, message, data = null) {
    const detailSpan = document.getElementById(`${loadingId}-detail`);

    // If we have structured data with stage, use it directly
    if (data && data.stage) {
        const stepIndex = PIPELINE_STEPS.findIndex(s => s.id === data.stage);
        if (stepIndex !== -1) {
            const step = PIPELINE_STEPS[stepIndex];
            const textSpan = document.getElementById(`${loadingId}-text`);
            if (textSpan) {
                if (step.isRulesStep) {
                    textSpan.textContent = `★ Rules - ${step.label}`;
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
            textSpan.textContent = `★ Rules - ${stepLabel}`;
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

function removeLoadingMessage(loadingId) {
    const loadingDiv = document.getElementById(loadingId);
    if (loadingDiv) {
        loadingDiv.remove();
    }
}

// Update session metrics display
function updateSessionMetrics(result) {
    sessionMetrics.totalQueries++;

    if (result && result.success) {
        sessionMetrics.successfulQueries++;

        // Track rows returned
        if (result.rows && result.rows.length > 0) {
            sessionMetrics.totalRows += result.rows.length;
        }
    }

    // Track response time
    if (sessionMetrics.currentQueryStartTime) {
        const responseTime = Date.now() - sessionMetrics.currentQueryStartTime;
        sessionMetrics.responseTimes.push(responseTime);
        sessionMetrics.currentQueryStartTime = null;
    }

    // Track tokens
    if (result && result.tokenUsage) {
        const promptTokens = result.tokenUsage.prompt_tokens || 0;
        const completionTokens = result.tokenUsage.completion_tokens || 0;
        const totalTokens = result.tokenUsage.total_tokens || (promptTokens + completionTokens);

        if (totalTokens > 0) {
            sessionMetrics.totalTokens += totalTokens;
            // Update token tracker for display
            tokenTracker.promptTokens = promptTokens;
            tokenTracker.completionTokens = completionTokens;
            tokenTracker.totalTokens = totalTokens;
            updateTokenDisplay();
        }
    }

    // Update display
    displaySessionMetrics();
}

function displaySessionMetrics() {
    const metricsContainer = document.getElementById('chatMetrics');
    if (!metricsContainer) return;

    // Show the metrics container
    metricsContainer.style.display = 'grid';

    // Update individual metrics
    document.getElementById('metricQueries').textContent = sessionMetrics.totalQueries;

    const successEl = document.getElementById('metricSuccess');
    successEl.textContent = sessionMetrics.successfulQueries;
    successEl.className = 'metric-value' + (sessionMetrics.successfulQueries > 0 ? ' success' : '');

    // Calculate average response time
    const avgTime = sessionMetrics.responseTimes.length > 0
        ? Math.round(sessionMetrics.responseTimes.reduce((a, b) => a + b, 0) / sessionMetrics.responseTimes.length)
        : 0;
    document.getElementById('metricAvgTime').textContent = avgTime > 1000
        ? `${(avgTime / 1000).toFixed(1)}s`
        : `${avgTime}ms`;

    document.getElementById('metricTotalRows').textContent = sessionMetrics.totalRows.toLocaleString();

    const tokensEl = document.getElementById('metricTokensUsed');
    tokensEl.textContent = sessionMetrics.totalTokens > 1000
        ? `${(sessionMetrics.totalTokens / 1000).toFixed(1)}k`
        : sessionMetrics.totalTokens;
}

// Reset session metrics (called when clearing chat)
function resetSessionMetrics() {
    sessionMetrics = {
        totalQueries: 0,
        successfulQueries: 0,
        totalRows: 0,
        totalTokens: 0,
        responseTimes: [],
        currentQueryStartTime: null
    };

    const metricsContainer = document.getElementById('chatMetrics');
    if (metricsContainer) {
        metricsContainer.style.display = 'none';
    }
}

function createSqlBlock(sql) {
    return `
        <div class="sql-block">
            <div class="sql-header">
                <span>📝</span>
                <span>Generated SQL</span>
            </div>
            <div class="sql-code">${escapeHtml(sql)}</div>
        </div>
    `;
}

function createResultsTable(result) {
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
                        ⬇
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

function exportTableToCSV(button) {
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
 */
function showResultSql(button) {
    const wrapper = button.closest('.results-table-wrapper');
    const resultsJson = wrapper.getAttribute('data-results');

    if (!resultsJson) {
        EwrModal.display({
            title: 'Error',
            size: 'small',
            sections: [{ content: 'No SQL data available' }]
        });
        return;
    }

    const result = JSON.parse(resultsJson);
    const sql = result.generatedSql || 'No SQL query available';
    const originalQuery = result.naturalLanguage || result.originalQuery || '';

    // Store SQL for copy function
    window._currentSqlForCopy = sql;

    // Use EwrModal display
    EwrModal.display({
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

async function copySqlToClipboard() {
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

function clearChat() {
    if (chatHistory.length > 0 && !confirm('Clear all chat history?')) {
        return;
    }

    chatHistory = [];
    updateTokenTrackerFromHistory(); // Reset token counter
    const chatMessages = document.getElementById('chatMessages');
    chatMessages.innerHTML = `
        <div class="empty-chat" id="emptyChat">
            <div class="empty-icon">💬</div>
            <div class="empty-title">Start a Conversation</div>
            <div class="empty-text">
                Ask questions about your database in natural language. I'll help you write and execute SQL queries.
            </div>
        </div>
    `;

    // Reset token tracker
    tokenTracker.currentTokens = 0;
    tokenTracker.isLimitReached = false;
    updateTokenDisplay();
    hideContextLimitAlert();

    // Clear timing display
    clearTimingDisplay();

    // Re-enable chat if it was disabled due to limit
    const chatInput = document.getElementById('chatInput');
    const sendBtn = document.getElementById('sendBtn');
    chatInput.disabled = !appState.databaseLoaded;
    sendBtn.disabled = !appState.databaseLoaded || isProcessing;

    // Reset session metrics
    resetSessionMetrics();
}

// Helper Functions
function getConnectionConfig() {
    const authType = document.getElementById('authType').value;
    const integratedAuthEl = document.getElementById('integratedAuth');
    const integratedAuth = integratedAuthEl ? integratedAuthEl.checked : false;

    // Always read the field values
    const username = document.getElementById('username').value.trim();
    const password = document.getElementById('password').value;
    const domain = document.getElementById('domain').value.trim();

    const config = {
        server: document.getElementById('server').value,
        database: document.getElementById('database').value,
        trustServerCertificate: document.getElementById('trustCert').checked,
        encrypt: document.getElementById('encrypt').checked,
        authType: authType,
        integratedAuth: integratedAuth,
        user: username,
        password: password
    };

    // Add domain for Windows auth
    if (authType === 'windows' && domain) {
        config.domain = domain;
    }

    return config;
}

function toggleAuthFields() {
    const authType = document.getElementById('authType').value;
    const domainInput = document.getElementById('domain');
    const usernameInput = document.getElementById('username');
    const passwordInput = document.getElementById('password');
    const integratedAuthField = document.getElementById('integratedAuthField');
    const integratedAuthCheckbox = document.getElementById('integratedAuth');

    if (authType === 'windows') {
        // Show integrated auth checkbox for Windows auth
        if (integratedAuthField) {
            integratedAuthField.style.display = 'flex';
        }
        // Enable domain field for Windows auth
        domainInput.disabled = false;
        // Update field states based on integrated auth setting
        toggleIntegratedAuth();
    } else {
        // SQL Server auth - hide integrated auth, disable and clear domain field
        if (integratedAuthField) {
            integratedAuthField.style.display = 'none';
        }
        if (integratedAuthCheckbox) {
            integratedAuthCheckbox.checked = false;
        }
        domainInput.disabled = true;
        domainInput.value = '';
        usernameInput.disabled = false;
        passwordInput.disabled = false;
    }
}

function toggleIntegratedAuth() {
    const integratedAuthCheckbox = document.getElementById('integratedAuth');
    const domainInput = document.getElementById('domain');
    const usernameInput = document.getElementById('username');
    const passwordInput = document.getElementById('password');

    if (integratedAuthCheckbox && integratedAuthCheckbox.checked) {
        // Integrated auth - disable credential fields
        domainInput.disabled = true;
        usernameInput.disabled = true;
        passwordInput.disabled = true;
    } else {
        // Explicit credentials - enable fields (only if Windows auth)
        const authType = document.getElementById('authType').value;
        if (authType === 'windows') {
            domainInput.disabled = false;
        }
        usernameInput.disabled = false;
        passwordInput.disabled = false;
    }
}

/**
 * Reset all connection settings to their default values
 */
async function resetConnectionSettings() {
    // Call disconnect endpoint to close connection pool
    try {
        await fetch(`${SQL_API_BASE}/api/sql/disconnect`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' }
        });
    } catch (error) {
        console.warn('Failed to disconnect:', error.message);
    }

    // Reset server
    document.getElementById('server').value = 'NCSQLTEST';

    // Reset database dropdown
    const dbSelect = document.getElementById('database');
    dbSelect.innerHTML = '<option value="">Select database...</option>';
    dbSelect.disabled = true;

    // Reset authentication type to SQL Server
    document.getElementById('authType').value = 'sql';

    // Reset domain field (disabled for SQL Server auth)
    document.getElementById('domain').value = '';

    // Reset credentials to defaults
    document.getElementById('username').value = 'EWRUser';
    document.getElementById('password').value = '66a3904d69';

    // Reset checkboxes
    document.getElementById('trustCert').checked = true;
    document.getElementById('encrypt').checked = false;

    // Reset application state
    appState.connectionTested = false;
    appState.databaseLoaded = false;
    appState.currentDatabase = null;
    appState.schemaStats = null;

    // Re-enable connection fields
    disableConnectionFields(false);
    // Domain should be disabled for SQL auth
    document.getElementById('domain').disabled = true;

    // Reset Connect button state
    const testBtn = document.querySelector('button[onclick="testConnection()"]');
    if (testBtn) {
        testBtn.disabled = false;
        testBtn.textContent = 'Connect to Server';
        testBtn.classList.remove('connected');
    }

    // Reset Load Database button state
    const loadDatabaseBtn = document.getElementById('loadDatabaseBtn');
    if (loadDatabaseBtn) {
        loadDatabaseBtn.disabled = true;
        loadDatabaseBtn.textContent = 'Load DB';
        loadDatabaseBtn.classList.remove('connected');
    }

    // Clear connection message
    hideConnectionMessage();

    // Update UI state
    updateUIState();
}

/**
 * Clear connection settings - closes connection and resets form
 * Calls the disconnect endpoint to close pooled connections
 */
async function clearConnection() {
    // Collapse status message on clear
    if (typeof collapseStatusMessage === 'function') {
        collapseStatusMessage();
    }

    // Call disconnect endpoint to close connection pool
    try {
        await fetch(`${SQL_API_BASE}/api/sql/disconnect`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' }
        });
        console.log('Connection pool closed via disconnect endpoint');
    } catch (error) {
        console.warn('Failed to disconnect:', error.message);
        // Continue with clearing UI even if disconnect fails
    }

    // Clear server, username, and password
    document.getElementById('server').value = '';
    document.getElementById('username').value = '';
    document.getElementById('password').value = '';

    // Reset database dropdown
    const dbSelect = document.getElementById('database');
    dbSelect.innerHTML = '<option value="">Select database...</option>';
    dbSelect.disabled = true;

    // Reset application state
    appState.connectionTested = false;
    appState.databaseLoaded = false;
    appState.currentDatabase = null;
    appState.schemaStats = null;

    // Re-enable all connection fields
    disableConnectionFields(false);

    // Reset Connect button state
    const testBtn = document.querySelector('button[onclick="testConnection()"]');
    if (testBtn) {
        testBtn.disabled = false;
        testBtn.textContent = 'Connect to Server';
        testBtn.classList.remove('connected');
    }

    // Reset Load Database button state
    const loadDatabaseBtn = document.getElementById('loadDatabaseBtn');
    if (loadDatabaseBtn) {
        loadDatabaseBtn.disabled = true;
        loadDatabaseBtn.textContent = 'Load DB';
        loadDatabaseBtn.classList.remove('connected');
    }

    // Clear all status pills
    const connectionStatusInline = document.getElementById('connectionStatusInline');
    const connectionMessage = document.getElementById('connectionMessage');
    const schemaStatus = document.getElementById('schemaStatus');
    if (connectionStatusInline) {
        connectionStatusInline.textContent = '';
        connectionStatusInline.className = 'ewr-status-pill';
    }
    if (connectionMessage) {
        connectionMessage.textContent = '';
        connectionMessage.className = 'ewr-status-pill';
    }
    if (schemaStatus) {
        schemaStatus.textContent = '';
        schemaStatus.className = 'ewr-status-pill';
    }

    updateUIState();
}

/**
 * Save current connection settings as default
 * Calls the backend API to persist settings
 */
async function saveAsDefault() {
    const connectionMessage = document.getElementById('connectionMessage');

    // Check if connected
    if (!appState.connectionTested) {
        showMessage(connectionMessage, 'error', 'Please connect to the server first before saving settings');
        return;
    }

    try {
        const connectionConfig = getConnectionConfig();

        // Show loading state
        showMessage(connectionMessage, 'info', 'Saving connection settings...');

        // Get current user info
        const userStr = localStorage.getItem('user');
        const userName = userStr ? JSON.parse(userStr).username || 'default' : 'default';

        // Gather all settings from the form
        const screenSettings = {
            server: document.getElementById('server').value,
            database: document.getElementById('database').value,
            authType: document.getElementById('authType').value,
            domain: document.getElementById('domain').value,
            username: document.getElementById('username').value,
            trustCert: document.getElementById('trustCert').checked,
            encrypt: document.getElementById('encrypt').checked
        };

        // Try to save to SQL database first
        try {
            const sqlResponse = await fetch('/api/sql/settings', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    userName,
                    pageName: 'sql-chat',
                    settings: screenSettings,
                    connectionConfig
                })
            });

            const sqlResult = await sqlResponse.json();
            if (sqlResult.success) {
                console.log('Settings saved to SQL database');
            } else {
                console.warn('Could not save to SQL database:', sqlResult.error);
            }
        } catch (sqlError) {
            console.warn('SQL settings save failed, continuing with other methods:', sqlError.message);
        }

        // Also try the general settings endpoint
        const response = await fetch('/api/settings/sql-connection', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(connectionConfig)
        });

        const result = await response.json();

        if (result.success) {
            showMessage(connectionMessage, 'success', 'Connection settings saved as default');

            // Also save to localStorage as backup
            await saveConnectionSettings();

            setTimeout(() => {
                hideConnectionMessage();
            }, 3000);
        } else {
            // Fallback - save to localStorage only
            await saveConnectionSettings();
            showMessage(connectionMessage, 'success', 'Settings saved locally');

            setTimeout(() => {
                hideConnectionMessage();
            }, 3000);
        }
    } catch (error) {
        // Last resort - save to localStorage
        await saveConnectionSettings();
        showMessage(connectionMessage, 'info', 'Settings saved locally (server unavailable)');

        setTimeout(() => {
            hideConnectionMessage();
        }, 3000);
    }
}

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

function hideConnectionMessage() {
    const container = document.getElementById('connectionMessage');
    container.className = 'ewr-status-pill';
    container.textContent = '';

    // Also hide the status container if no other status showing
    const statusContainer = document.getElementById('statusMessageContainer');
    const connectionStatusInline = document.getElementById('connectionStatusInline');
    if (statusContainer && (!connectionStatusInline || !connectionStatusInline.textContent.trim())) {
        statusContainer.classList.remove('visible');
    }
}

// Error Modal Functions
function showErrorPopup(message, title = 'Connection Error', details = null, connectionInfo = null) {
    const modal = document.getElementById('errorModal');
    const messageEl = document.getElementById('errorModalMessage');
    const titleEl = modal.querySelector('.modal-header h2');

    if (titleEl) titleEl.textContent = title;

    // Build comprehensive error message
    let fullMessage = message;

    if (details) {
        fullMessage += '\n\n' + details;
    }

    if (connectionInfo) {
        fullMessage += '\n\n--- Connection Parameters ---\n';
        for (const [key, value] of Object.entries(connectionInfo)) {
            fullMessage += `${key}: ${value}\n`;
        }
    }

    if (messageEl) messageEl.textContent = fullMessage;

    // Use EWR modal API
    if (modal.open) {
        modal.open();
    } else {
        modal.classList.remove('hidden');
    }
}

function closeErrorModal(event) {
    // If called from backdrop click, only close if clicking the backdrop itself
    if (event && event.target !== event.currentTarget) return;

    const modal = document.getElementById('errorModal');
    // Use EWR modal API
    if (modal.close) {
        modal.close();
    } else {
        modal.classList.add('hidden');
    }
}

function updateSendButton() {
    const btn = document.getElementById('sendBtn');
    btn.disabled = !appState.databaseLoaded || isProcessing;
}

function autoResizeTextarea(textarea) {
    textarea.style.height = 'auto';
    textarea.style.height = Math.min(textarea.scrollHeight, 120) + 'px';
}

function scrollToBottom() {
    const chatMessages = document.getElementById('chatMessages');
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

/**
 * Initialize the context window display on page load
 */
function initializeContextRemainingDisplay() {
    const usedValueEl = document.getElementById('contextUsedValue');
    const maxValueEl = document.getElementById('contextMaxValue');

    // Get context window size from settings
    const savedContextWindow = localStorage.getItem('sqlContextWindowSize');
    tokenTracker.maxTokens = savedContextWindow ? parseInt(savedContextWindow) : 8192;

    if (usedValueEl) {
        usedValueEl.textContent = '0';
    }
    if (maxValueEl) {
        maxValueEl.textContent = tokenTracker.maxTokens.toLocaleString();
    }
}

/**
 * Update token usage display
 */
function updateTokenDisplay() {
    const usedValueEl = document.getElementById('contextUsedValue');
    const maxValueEl = document.getElementById('contextMaxValue');
    const promptTokensEl = document.getElementById('promptTokens');
    const completionTokensEl = document.getElementById('completionTokens');
    const totalTokensEl = document.getElementById('totalTokens');

    const percentage = (tokenTracker.currentTokens / tokenTracker.maxTokens) * 100;
    const isWarning = percentage > 80;

    // Update the context window display (used / max format)
    if (usedValueEl) {
        usedValueEl.textContent = tokenTracker.currentTokens.toLocaleString();
        usedValueEl.style.color = isWarning ? '#f59e0b' : '#10b981';
    }
    if (maxValueEl) {
        maxValueEl.textContent = tokenTracker.maxTokens.toLocaleString();
        maxValueEl.style.color = isWarning ? '#f59e0b' : '#10b981';
    }

    // Update prompt/completion/total token displays
    if (promptTokensEl) {
        promptTokensEl.textContent = tokenTracker.promptTokens.toLocaleString();
    }
    if (completionTokensEl) {
        completionTokensEl.textContent = tokenTracker.completionTokens.toLocaleString();
    }
    if (totalTokensEl) {
        totalTokensEl.textContent = tokenTracker.totalTokens.toLocaleString();
    }

    // Update all labels color based on context usage
    document.querySelectorAll('.context-remaining-label').forEach(label => {
        label.style.color = isWarning ? '#f59e0b' : '';
    });
    document.querySelectorAll('.context-remaining-value').forEach(value => {
        value.style.color = isWarning ? '#f59e0b' : '';
    });

    // Check if limit is reached
    if (percentage >= 100 && !tokenTracker.isLimitReached) {
        tokenTracker.isLimitReached = true;
        showContextLimitAlert();
        disableChatInput();
    }
}

/**
 * Show context limit alert
 */
function showContextLimitAlert() {
    const alert = document.getElementById('contextLimitAlert');
    if (alert) {
        alert.style.display = 'block';
    }
}

/**
 * Hide context limit alert
 */
function hideContextLimitAlert() {
    const alert = document.getElementById('contextLimitAlert');
    if (alert) {
        alert.style.display = 'none';
    }
}

/**
 * Disable chat input when context limit is reached
 */
function disableChatInput() {
    const chatInput = document.getElementById('chatInput');
    const sendBtn = document.getElementById('sendBtn');

    chatInput.disabled = true;
    sendBtn.disabled = true;
    chatInput.placeholder = 'Context window full - click Clear Chat to continue';
}

/**
 * Estimate token count from text (~4 chars per token)
 */
function estimateTokens(text) {
    if (!text) return 0;
    return Math.ceil(text.length / 4);
}

/**
 * Update token tracker from conversation history
 * Called after every query to track context usage
 */
function updateTokenTrackerFromHistory() {
    const savedContextWindow = localStorage.getItem('sqlContextWindowSize');
    tokenTracker.maxTokens = savedContextWindow ? parseInt(savedContextWindow) : 8192;

    // Estimate tokens from chat history
    let totalEstimatedTokens = 0;
    chatHistory.forEach(msg => {
        const textTokens = estimateTokens(msg.text || '');
        const sqlTokens = msg.sql ? estimateTokens(msg.sql) : 0;
        totalEstimatedTokens += textTokens + sqlTokens;
    });

    tokenTracker.currentTokens = totalEstimatedTokens;
    console.log(`[TokenTracker] History: ${chatHistory.length} msgs, Estimated tokens: ${totalEstimatedTokens}`);
    updateTokenDisplay();
}

/**
 * Update token tracker from API response
 */
function updateTokenTracker(tokenUsage) {
    // Always update from history to show context usage
    updateTokenTrackerFromHistory();
}

// Pipeline step timing tracker
let pipelineTimingTracker = {
    startTime: null,
    lastStepTime: null,
    currentStep: null,
    stepTimings: {}
};

/**
 * Initialize pipeline timing display
 */
function initPipelineTiming() {
    pipelineTimingTracker = {
        startTime: Date.now(),
        lastStepTime: Date.now(),
        currentStep: null,
        stepTimings: {}
    };

    const container = document.getElementById('timingMetrics');
    if (!container) return;

    // Show the display
    container.classList.add('visible');

    // Reset all steps to initial state
    const steps = ['preprocessing', 'security', 'rules', 'schema', 'generating', 'fixing', 'executing'];
    steps.forEach(step => {
        const el = document.getElementById(`step-${step}`);
        if (el) {
            el.classList.remove('active', 'completed');
            const valueEl = el.querySelector('.timing-metric-value');
            if (valueEl) valueEl.textContent = '--';
        }
    });

    // Reset total
    const totalEl = document.getElementById('timingTotal');
    if (totalEl) totalEl.textContent = '--';
}

/**
 * Update pipeline step timing from SSE event
 * @param {Object} data - SSE progress data with stage and elapsed
 */
function updatePipelineStep(data) {
    if (!data || !data.stage) return;

    const container = document.getElementById('timingMetrics');
    if (!container) return;
    container.classList.add('visible');

    const stage = data.stage;
    const elapsed = data.elapsed || 0;
    const now = Date.now();

    // Map backend stages to display step IDs
    const stageToStep = {
        'preprocessing': 'preprocessing',
        'security': 'security',
        'cache': 'security',  // Cache is part of security/init phase
        'rules': 'rules',
        'schema': 'schema',
        'generating': 'generating',
        'fixing': 'fixing',
        'validating': 'fixing',  // Validate is part of fixing phase
        'executing': 'executing',
        'complete': 'executing'  // Complete marks end of executing
    };

    const stepId = stageToStep[stage];
    if (!stepId) return;

    // Mark previous step as completed with its duration
    if (pipelineTimingTracker.currentStep && pipelineTimingTracker.currentStep !== stepId) {
        const prevStepEl = document.getElementById(`step-${pipelineTimingTracker.currentStep}`);
        if (prevStepEl) {
            prevStepEl.classList.remove('active');
            prevStepEl.classList.add('completed');

            // Calculate step duration
            const stepDuration = now - pipelineTimingTracker.lastStepTime;
            pipelineTimingTracker.stepTimings[pipelineTimingTracker.currentStep] = stepDuration;

            const valueEl = prevStepEl.querySelector('.timing-metric-value');
            if (valueEl) {
                valueEl.textContent = formatDuration(stepDuration);
                valueEl.className = 'timing-metric-value ' + getSpeedClass(stepDuration, stepId);
            }
        }
    }

    // Mark current step as active
    const currentStepEl = document.getElementById(`step-${stepId}`);
    if (currentStepEl && !currentStepEl.classList.contains('completed')) {
        currentStepEl.classList.add('active');
    }

    // Update tracking
    if (pipelineTimingTracker.currentStep !== stepId) {
        pipelineTimingTracker.lastStepTime = now;
        pipelineTimingTracker.currentStep = stepId;
    }

    // Update total elapsed time
    const totalEl = document.getElementById('timingTotal');
    if (totalEl) {
        totalEl.textContent = formatDuration(elapsed * 1000);
    }
}

/**
 * Finalize pipeline timing (when result received)
 * @param {number} totalMs - Total processing time in ms
 */
function finalizePipelineTiming(totalMs) {
    const now = Date.now();

    // Complete the last step
    if (pipelineTimingTracker.currentStep) {
        const lastStepEl = document.getElementById(`step-${pipelineTimingTracker.currentStep}`);
        if (lastStepEl) {
            lastStepEl.classList.remove('active');
            lastStepEl.classList.add('completed');

            const stepDuration = now - pipelineTimingTracker.lastStepTime;
            const valueEl = lastStepEl.querySelector('.timing-metric-value');
            if (valueEl) {
                valueEl.textContent = formatDuration(stepDuration);
                valueEl.className = 'timing-metric-value ' + getSpeedClass(stepDuration, pipelineTimingTracker.currentStep);
            }
        }
    }

    // Update total
    const totalEl = document.getElementById('timingTotal');
    if (totalEl) {
        const total = totalMs || (now - pipelineTimingTracker.startTime);
        totalEl.textContent = formatDuration(total);
        totalEl.className = 'timing-metric-value ' + (total < 3000 ? 'fast' : total < 8000 ? 'medium' : 'slow');
    }
}

/**
 * Format duration in ms to human-readable
 */
function formatDuration(ms) {
    if (ms === undefined || ms === null) return '--';
    if (ms < 1) return '<1ms';
    if (ms < 1000) return `${Math.round(ms)}ms`;
    return `${(ms / 1000).toFixed(1)}s`;
}

/**
 * Get speed class based on step type and duration
 */
function getSpeedClass(ms, stepId) {
    if (ms === undefined || ms === null) return '';

    // Thresholds vary by step type (in ms)
    const thresholds = {
        'preprocessing': { fast: 100, slow: 500 },
        'security': { fast: 100, slow: 500 },
        'rules': { fast: 200, slow: 1000 },
        'schema': { fast: 300, slow: 1000 },
        'generating': { fast: 2000, slow: 8000 },  // LLM is slower
        'fixing': { fast: 100, slow: 500 },
        'executing': { fast: 500, slow: 2000 }
    };

    const t = thresholds[stepId] || { fast: 200, slow: 1000 };
    if (ms <= t.fast) return 'fast';
    if (ms >= t.slow) return 'slow';
    return 'medium';
}

/**
 * Clear timing display (called on chat clear)
 */
function clearTimingDisplay() {
    const container = document.getElementById('timingMetrics');
    if (container) {
        container.classList.remove('visible');

        // Reset all steps
        const steps = ['preprocessing', 'security', 'rules', 'schema', 'generating', 'fixing', 'executing'];
        steps.forEach(step => {
            const el = document.getElementById(`step-${step}`);
            if (el) {
                el.classList.remove('active', 'completed');
                const valueEl = el.querySelector('.timing-metric-value');
                if (valueEl) valueEl.textContent = '--';
            }
        });

        const totalEl = document.getElementById('timingTotal');
        if (totalEl) totalEl.textContent = '--';
    }

    pipelineTimingTracker = {
        startTime: null,
        lastStepTime: null,
        currentStep: null,
        stepTimings: {}
    };
}

/**
 * Legacy: Update pipeline timing metrics display (for backward compatibility)
 * @param {Object} timing - Timing metrics from API response
 */
function updateTimingDisplay(timing) {
    // This function is kept for backward compatibility
    // Real-time timing is now handled by updatePipelineStep and finalizePipelineTiming
    if (timing && timing.totalMs) {
        finalizePipelineTiming(timing.totalMs);
    }
}

// Example Modal Functions
function showAddExampleModal() {
    const databaseEl = document.getElementById('database');
    const database = databaseEl.value !== undefined ? databaseEl.value : (databaseEl.getAttribute && databaseEl.getAttribute('value'));

    if (!database) {
        alert('Please select a database first');
        return;
    }

    // Clear form - use EWR component API if available
    const examplePrompt = document.getElementById('examplePrompt');
    if (examplePrompt.value !== undefined) {
        examplePrompt.value = '';
    }
    document.getElementById('exampleSql').value = '';
    document.getElementById('exampleResponse').value = '';
    document.getElementById('exampleMessage').innerHTML = '';

    // Hide test results
    const testResults = document.getElementById('exampleTestResults');
    if (testResults) testResults.style.display = 'none';
    const testOutput = document.getElementById('exampleTestOutput');
    if (testOutput) testOutput.innerHTML = '';

    // Display database name
    const dbNameEl = document.getElementById('exampleDatabaseName');
    if (dbNameEl) dbNameEl.textContent = database;

    // Show modal - use EWR modal API
    const modal = document.getElementById('exampleModal');
    if (modal.open) {
        modal.open();
    } else {
        modal.classList.remove('hidden');
    }
}

function closeExampleModal(event) {
    if (event && event.target !== event.currentTarget) {
        return;
    }
    const modal = document.getElementById('exampleModal');
    if (modal.close) {
        modal.close();
    } else {
        modal.classList.add('hidden');
    }
}

async function saveExample() {
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
            showMessage(messageDiv, 'success', '✅ Example saved successfully! The AI will use this for similar queries.');
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
 * Executes the query against the current database connection
 */
async function testExampleSql() {
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
    statusSpan.innerHTML = '<span style="color: #3b82f6;">⏳ Testing...</span>';
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
            statusSpan.innerHTML = `<span style="color: #10b981;">✅ Success - ${rowCount} row${rowCount !== 1 ? 's' : ''}</span>`;

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
            statusSpan.innerHTML = '<span style="color: #ef4444;">❌ Error</span>';
            outputDiv.innerHTML = `<div style="color: #fca5a5;">${escapeHtml(result.error || result.details || 'Query execution failed')}</div>`;
        }
    } catch (error) {
        statusSpan.innerHTML = '<span style="color: #ef4444;">❌ Error</span>';
        outputDiv.innerHTML = `<div style="color: #fca5a5;">Test failed: ${escapeHtml(error.message)}</div>`;
    }
}

// ========================
// SQL Rules Management
// ========================

/**
 * Show the Add Rule modal
 */
function showAddRuleModal() {
    const databaseEl = document.getElementById('database');
    const database = databaseEl.value !== undefined ? databaseEl.value : '';

    if (!database) {
        alert('Please select a database first');
        return;
    }

    const modal = document.getElementById('addRuleModal');
    if (modal) {
        // Clear form - use EWR component API if available
        const ruleDesc = document.getElementById('ruleDescription');
        if (ruleDesc) ruleDesc.value = '';
        const ruleType = document.getElementById('ruleType');
        if (ruleType) ruleType.value = 'constraint';
        const ruleKeywords = document.getElementById('ruleTriggerKeywords');
        if (ruleKeywords) ruleKeywords.value = '';
        document.getElementById('ruleText').value = '';
        const autoFixPattern = document.getElementById('ruleAutoFixPattern');
        if (autoFixPattern) autoFixPattern.value = '';
        const autoFixReplacement = document.getElementById('ruleAutoFixReplacement');
        if (autoFixReplacement) autoFixReplacement.value = '';
        document.getElementById('addRuleMessage').innerHTML = '';

        // Use EWR modal API
        if (modal.open) {
            modal.open();
        } else {
            modal.classList.remove('hidden');
        }
    }
}

/**
 * Generate rule fields using AI based on problem description
 */
async function generateRuleWithAI() {
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
    btnText.textContent = '⏳ Generating...';

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

        showAddRuleMessage('✨ Rule generated! Review and adjust the fields below, then click Save Rule.', false);

    } catch (error) {
        console.error('AI rule generation error:', error);
        showAddRuleMessage(`AI generation failed: ${error.message}. Please fill in the fields manually.`, true);
    } finally {
        btn.disabled = false;
        btnText.textContent = '✨ Generate Rule with AI';
    }
}

/**
 * Populate rule form fields from AI-generated data
 */
function populateRuleFields(rule) {
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
 */
function closeAddRuleModal(event) {
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
 */
function showAddRuleMessage(message, isError = false) {
    const msgDiv = document.getElementById('addRuleMessage');
    if (msgDiv) {
        msgDiv.innerHTML = `<div style="padding: 10px; margin-bottom: 12px; border-radius: 6px; background: ${isError ? 'rgba(239, 68, 68, 0.15)' : 'rgba(16, 185, 129, 0.15)'}; color: ${isError ? '#fca5a5' : '#6ee7b7'}; border: 1px solid ${isError ? '#ef4444' : '#10b981'};">${message}</div>`;
    }
}

/**
 * Save a new rule via API
 */
async function saveRule() {
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
            showAddRuleMessage('✅ Rule saved successfully!', false);
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

/**
 * Show the Rules List modal
 */
async function showRulesModal() {
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
 */
function renderRulesList(data) {
    const content = document.getElementById('rulesListContent');
    if (!content) return;

    let html = '';

    // Global Constraints Section
    if (data.global_constraints && data.global_constraints.length > 0) {
        html += `<div style="margin-bottom: 24px;">
            <h3 style="color: #a78bfa; margin-bottom: 12px; display: flex; align-items: center; gap: 8px;">
                <span style="font-size: 20px;">🌐</span> Global Constraints
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
                        <span style="font-size: 20px;">🗄️</span> ${dbName}
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
 */
function renderRuleCard(rule, scope) {
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
 */
function closeRulesListModal(event) {
    if (event && event.target !== event.currentTarget) return;
    const modal = document.getElementById('rulesListModal');
    if (modal) {
        modal.classList.add('hidden');
    }
}

// ============================================================================
// SQL FEEDBACK SYSTEM - Thumbs Up/Down with Modal
// ============================================================================

// Store current feedback context
let currentFeedbackContext = {
    query: null,
    generatedSql: null,
    database: null,
    messageId: null,
    buttonElement: null
};

/**
 * Handle positive feedback (thumbs up)
 */
function handlePositiveFeedback(query, sql, messageId, buttonElement) {
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
 * Handle negative feedback (thumbs down) - opens modal
 */
function handleNegativeFeedback(query, sql, messageId, buttonElement) {
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
 * Open the SQL Feedback modal
 */
function openSqlFeedbackModal(query, generatedSql) {
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
 * Close the SQL Feedback modal
 */
function closeSqlFeedbackModal(event) {
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
 * Save SQL feedback - Simple free-text feedback
 */
async function saveSqlFeedback() {
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
 * Submit feedback to the API
 */
async function submitFeedback(feedbackData) {
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
 * Show a temporary toast message
 */
function showFeedbackToast(message) {
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

/**
 * Get current database from dropdown
 */
function getCurrentDatabase() {
    const databaseSelect = document.getElementById('database');
    return databaseSelect ? databaseSelect.value : null;
}

/**
 * Get current user role
 */
function getUserRole() {
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
// Feedback Review System
// ============================================================================

/**
 * Show the feedback review modal and load feedback data
 * Uses the <ewr-modal> component's open() method
 */
function showFeedbackReviewModal() {
    const modal = document.getElementById('feedbackReviewModal');
    if (modal && modal.open) {
        modal.open();
        loadFeedbackList();
    }
}

/**
 * Close the feedback review modal
 * Uses the <ewr-modal> component's close() method
 */
function closeFeedbackReviewModal(event) {
    const modal = document.getElementById('feedbackReviewModal');
    if (modal && modal.close) {
        modal.close();
    }
}

/**
 * Load feedback list from API
 */
async function loadFeedbackList() {
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
 * Update the feedback badge count
 */
function updateFeedbackBadge(count) {
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
 * Render a single feedback item
 */
function renderFeedbackItem(item) {
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

/**
 * Generate a rule from feedback using AI
 */
async function generateRuleFromFeedback(feedbackId) {
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
 * Mark feedback as processed
 */
async function markFeedbackProcessed(feedbackId, processed = true) {
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

/**
 * Show the rule preview modal with generated rule data
 */
function showRulePreviewModal(rule, feedback, feedbackId) {
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
 * Close the rule preview modal
 */
function closeRulePreviewModal(event) {
    if (event && event.target !== event.currentTarget) return;
    const modal = document.getElementById('rulePreviewModal');
    if (modal) {
        modal.classList.add('hidden');
    }
}

/**
 * Create a rule from the preview modal
 */
async function createRuleFromPreview() {
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

/**
 * Load feedback count on page load (for badge)
 */
async function loadFeedbackCount() {
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

// Load feedback count on page load
document.addEventListener('DOMContentLoaded', () => {
    // Delay to let other init complete
    setTimeout(loadFeedbackCount, 2000);
});

