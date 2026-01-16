/**
 * Connection Manager Module
 *
 * Handles all SQL Server connection-related functionality including:
 * - Connection testing and validation
 * - Credential management (save/load/encrypt/decrypt)
 * - Authentication type switching (SQL Server vs Windows auth)
 * - Connection state management
 *
 * @module connection-manager
 */

import { SQL_API_BASE, state, resetAppState } from './sql-chat-state.js';
import { showMessage, hideConnectionMessage, showErrorPopup, updateUIState } from './ui-manager.js';

/**
 * Save connection settings to localStorage with encrypted password
 * Uses CryptoUtils for secure password storage
 *
 * @async
 * @returns {Promise<void>}
 */
export async function saveConnectionSettings() {
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
    connectionMessage.innerHTML = '<div class="message success">Connection settings saved (including encrypted password)</div>';
    setTimeout(() => {
        connectionMessage.innerHTML = originalHtml;
    }, 3000);

    console.log('Connection settings saved for user:', userId);
}

/**
 * Load saved connection settings from database first, then localStorage as fallback
 * Automatically decrypts saved passwords if CryptoUtils is available
 *
 * @async
 * @returns {Promise<void>}
 */
export async function loadSavedConnectionSettings() {
    const userStr = localStorage.getItem('user');
    const userId = userStr ? JSON.parse(userStr).id || 'default' : 'default';
    let settings = null;

    // Try loading from database first
    try {
        const token = localStorage.getItem('accessToken');
        if (token) {
            const response = await fetch('/api/auth/sql-connection-settings', {
                headers: { 'Authorization': `Bearer ${token}` }
            });
            const result = await response.json();
            if (result.success && result.settings) {
                settings = result.settings;
                console.log('Loaded settings from database');
            }
        }
    } catch (e) {
        console.warn('Failed to load settings from database:', e.message);
    }

    // Fallback to localStorage
    if (!settings) {
        const savedSettings = localStorage.getItem(`sqlConnectionSettings_${userId}`);
        if (savedSettings) {
            try {
                settings = JSON.parse(savedSettings);
                console.log('Loaded settings from localStorage');
            } catch (e) {
                console.error('Failed to parse localStorage settings:', e);
            }
        }
    }

    // Apply settings to form
    if (settings) {
        try {
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
                }
            }

            // Store database preference for after connection
            if (settings.database) {
                window._savedDatabasePreference = settings.database;
            }

            console.log('Applied saved connection settings for user:', userId);
        } catch (e) {
            console.error('Failed to apply saved connection settings:', e);
        }
    }
}

/**
 * Save connection settings to the server database
 * @async
 * @returns {Promise<void>}
 */
export async function saveSettingsToDatabase() {
    const connectionMessage = document.getElementById('connectionMessage');
    const userStr = localStorage.getItem('user');
    const userId = userStr ? JSON.parse(userStr).id || 'default' : 'default';

    // Get password and encrypt it
    const password = document.getElementById('password').value;
    let encryptedPassword = null;

    if (password && window.CryptoUtils && CryptoUtils.isSupported()) {
        try {
            encryptedPassword = await CryptoUtils.encrypt(password, userId);
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
        encryptedPassword: encryptedPassword,
        trustCert: document.getElementById('trustCert').checked,
        encrypt: document.getElementById('encrypt').checked
    };

    try {
        const token = localStorage.getItem('accessToken');
        if (!token) {
            showMessage(connectionMessage, 'error', 'Please log in to save settings');
            return;
        }

        const response = await fetch('/api/auth/sql-connection-settings', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${token}`
            },
            body: JSON.stringify(settings)
        });

        const result = await response.json();

        if (result.success) {
            // Also save to localStorage as backup
            localStorage.setItem(`sqlConnectionSettings_${userId}`, JSON.stringify(settings));
            showMessage(connectionMessage, 'success', 'Settings saved to database');
            setTimeout(() => hideConnectionMessage(), 3000);
        } else {
            showMessage(connectionMessage, 'error', result.error || 'Failed to save settings');
        }
    } catch (error) {
        console.error('Failed to save settings to database:', error);
        // Fallback to localStorage
        await saveConnectionSettings();
        showMessage(connectionMessage, 'info', 'Settings saved locally (server unavailable)');
        setTimeout(() => hideConnectionMessage(), 3000);
    }
}

/**
 * Validate connection fields
 * Currently a no-op as Connect button is always enabled - validation happens on click
 *
 * @returns {void}
 */
export function validateConnectionFields() {
    // Connect button is always enabled - validation happens on click
}

/**
 * Update connection status indicators in the UI
 * Updates inline status pill, database dropdown, and Load DB button states
 *
 * @returns {void}
 */
export function updateConnectionStatus() {
    const connectionMessage = document.getElementById('connectionMessage');
    const connectionStatusInline = document.getElementById('connectionStatusInline');
    const statusContainer = document.getElementById('statusMessageContainer');
    const databaseSelect = document.getElementById('database');
    const loadDatabaseBtn = document.getElementById('loadDatabaseBtn');

    // Update inline status
    if (state.app.connectionTested) {
        connectionStatusInline.textContent = 'Connected';
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
    if (state.app.databaseLoaded && state.app.schemaStats) {
        connectionMessage.textContent = `DB: ${state.app.currentDatabase} (${state.app.schemaStats.tables || 0} tables)`;
        connectionMessage.className = 'ewr-status-pill success';
        if (statusContainer) {
            statusContainer.classList.add('visible');
        }
    }
}

/**
 * Test connection to SQL Server
 * Attempts to connect using current form values, populates database list on success
 *
 * @async
 * @returns {Promise<void>}
 */
export async function testConnection() {
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
            state.app.connectionTested = true;
            console.log('Connection successful, state.app.connectionTested =', state.app.connectionTested);
            // Clear any error message on success and show connected status
            hideConnectionMessage();
            const connectionStatusInline = document.getElementById('connectionStatusInline');
            if (connectionStatusInline) {
                connectionStatusInline.textContent = `Connected to ${connectionConfig.server}`;
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
            state.app.connectionTested = false;
            // Show error in popup modal with full details
            const errorMessage = result.error || result.message || 'Unknown error';
            const errorDetails = result.details || null;
            const connectionInfo = result.connection_info || null;
            showErrorPopup(errorMessage, 'Connection Failed', errorDetails, connectionInfo);
            testBtn.disabled = false;
            testBtn.textContent = 'Connect to Server';
        }
    } catch (error) {
        state.app.connectionTested = false;
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
 *
 * @param {boolean} disable - Whether to disable (true) or enable (false) the fields
 * @returns {void}
 */
export function disableConnectionFields(disable) {
    const fields = ['server', 'authType', 'domain', 'username', 'password', 'trustCert', 'encrypt'];
    fields.forEach(id => {
        const el = document.getElementById(id);
        if (el) el.disabled = disable;
    });
}

/**
 * Get connection configuration from form fields
 * Assembles connection config object from all form inputs
 *
 * @returns {Object} Connection configuration object
 */
export function getConnectionConfig() {
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

    // Always include domain if provided (used for domain\username format)
    if (domain) {
        config.domain = domain;
    }

    return config;
}

/**
 * Toggle authentication fields based on auth type selection
 * Shows/hides domain field and integrated auth checkbox based on SQL vs Windows auth
 *
 * @returns {void}
 */
export function toggleAuthFields() {
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

/**
 * Toggle credential fields based on integrated auth checkbox
 * Disables username/password/domain when using Windows integrated auth
 *
 * @returns {void}
 */
export function toggleIntegratedAuth() {
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
 * Toggle password visibility based on Show Password checkbox
 */
export function toggleShowPassword() {
    const showPasswordCheckbox = document.getElementById('showPassword');
    const passwordInput = document.getElementById('password');

    if (showPasswordCheckbox && passwordInput) {
        passwordInput.type = showPasswordCheckbox.checked ? 'text' : 'password';
    }
}

/**
 * Reset all connection settings to their default values
 * Closes any existing connection and resets form to defaults
 *
 * @async
 * @returns {Promise<void>}
 */
export async function resetConnectionSettings() {
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
    document.getElementById('server').value = 'EWRSQLPROD';

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
    state.app.connectionTested = false;
    state.app.databaseLoaded = false;
    state.app.currentDatabase = null;
    state.app.schemaStats = null;

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
 *
 * @async
 * @returns {Promise<void>}
 */
export async function clearConnection() {
    // Collapse status message on clear (if function exists)
    if (typeof window.collapseStatusMessage === 'function') {
        window.collapseStatusMessage();
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
    state.app.connectionTested = false;
    state.app.databaseLoaded = false;
    state.app.currentDatabase = null;
    state.app.schemaStats = null;

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
 * Clear the SQL query cache for the current database
 * Admin-only function to clear cached queries from agent_learning
 *
 * @async
 * @returns {Promise<void>}
 */
export async function clearQueryCache() {
    const database = document.getElementById('database')?.value;

    if (!database) {
        alert('Please select a database first');
        return;
    }

    if (!confirm(`Clear all cached queries for ${database}?\n\nThis will force the LLM to regenerate SQL for all questions.`)) {
        return;
    }

    try {
        const response = await fetch(`${SQL_API_BASE}/api/sql/cache?question=.*&database=${encodeURIComponent(database)}`, {
            method: 'DELETE',
            headers: { 'Content-Type': 'application/json' }
        });

        const result = await response.json();

        if (result.success) {
            alert(`Cache cleared: ${result.message}`);
            console.log('Cache cleared:', result);
        } else {
            alert('Failed to clear cache');
        }
    } catch (error) {
        console.error('Failed to clear cache:', error);
        alert(`Error clearing cache: ${error.message}`);
    }
}

/**
 * Save current connection settings as default
 * Persists settings to both server-side storage and localStorage backup
 *
 * @async
 * @returns {Promise<void>}
 */
export async function saveAsDefault() {
    const connectionMessage = document.getElementById('connectionMessage');

    // Check if connected
    if (!state.app.connectionTested) {
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
