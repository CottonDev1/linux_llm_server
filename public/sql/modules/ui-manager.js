/**
 * SQL Chat UI Manager Module
 *
 * Handles all UI utility functions for the SQL Chat application including:
 * - UI state management (enabling/disabling elements based on application state)
 * - Panel toggling
 * - Send button state updates
 * - Textarea auto-resizing
 * - Chat scrolling
 * - HTML escaping
 * - Status message display
 * - Error modal management
 *
 * @module ui-manager
 */

import { state } from './sql-chat-state.js';

/**
 * Update the UI state based on current application state.
 * Enables/disables chat input, send button, save button, and other controls
 * based on connection and database loading status.
 */
export function updateUIState() {
    const chatInput = document.getElementById('chatInput');
    const sendBtn = document.getElementById('sendBtn');
    const saveBtn = document.querySelector('.btn-save');

    // Update chat interface state
    const chatEnabled = state.app.databaseLoaded;
    chatInput.disabled = !chatEnabled;
    sendBtn.disabled = !chatEnabled || state.isProcessing;

    // Update save button state - only enable when connected
    if (saveBtn) {
        saveBtn.disabled = !state.app.connectionTested;
        if (!state.app.connectionTested) {
            saveBtn.title = 'Connect to server first to enable saving';
        } else {
            saveBtn.title = 'Save connection settings';
        }
    }

    // Update Add Example button state - only enable when database is loaded
    const addExampleBtn = document.getElementById('addExampleBtn');
    if (addExampleBtn) {
        addExampleBtn.disabled = !state.app.databaseLoaded;
    }

    // Update Add Rule button state - only enable when database is loaded
    const addRuleBtn = document.getElementById('addRuleBtn');
    if (addRuleBtn) {
        addRuleBtn.disabled = !state.app.databaseLoaded;
    }

    // Update Load Database button state - stay pressed when database is loaded
    const loadDatabaseBtn = document.getElementById('loadDatabaseBtn');
    if (loadDatabaseBtn) {
        if (state.app.databaseLoaded) {
            loadDatabaseBtn.classList.add('connected');
            loadDatabaseBtn.textContent = 'DB Loaded';
        } else {
            loadDatabaseBtn.classList.remove('connected');
            loadDatabaseBtn.textContent = 'Load DB';
        }
    }

    // Update placeholder text
    if (!state.app.databaseLoaded) {
        chatInput.placeholder = 'Please connect to server and select a database to begin';
    } else {
        chatInput.placeholder = 'Ask a question about your data...';
    }

    // Update status indicator in connection panel
    updateConnectionStatus();
}

/**
 * Update the connection status display in the UI.
 * Shows/hides connection indicators and updates database dropdown state.
 */
export function updateConnectionStatus() {
    const connectionMessage = document.getElementById('connectionMessage');
    const connectionStatusInline = document.getElementById('connectionStatusInline');
    const statusContainer = document.getElementById('statusMessageContainer');
    const databaseSelect = document.getElementById('database');
    const loadDatabaseBtn = document.getElementById('loadDatabaseBtn');

    // Update inline status
    if (state.app.connectionTested) {
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
    if (state.app.databaseLoaded && state.app.schemaStats) {
        connectionMessage.textContent = `DB: ${state.app.currentDatabase} (${state.app.schemaStats.tables || 0} tables)`;
        connectionMessage.className = 'ewr-status-pill success';
        if (statusContainer) {
            statusContainer.classList.add('visible');
        }
    }
}

/**
 * Toggle the visibility of the left panel.
 * Updates the toggle button text to indicate panel state.
 */
export function togglePanel() {
    const panel = document.getElementById('leftPanel');
    const btn = document.getElementById('toggleBtn');

    panel.classList.toggle('collapsed');
    btn.textContent = panel.classList.contains('collapsed') ? '>' : '<';
}

/**
 * Update the send button's disabled state.
 * Button is disabled when database is not loaded or when processing.
 */
export function updateSendButton() {
    const btn = document.getElementById('sendBtn');
    btn.disabled = !state.app.databaseLoaded || state.isProcessing;
}

/**
 * Auto-resize a textarea based on its content.
 * Limits maximum height to 120 pixels.
 *
 * @param {HTMLTextAreaElement} textarea - The textarea element to resize
 */
export function autoResizeTextarea(textarea) {
    textarea.style.height = 'auto';
    textarea.style.height = Math.min(textarea.scrollHeight, 120) + 'px';
}

/**
 * Scroll the chat messages container to the bottom.
 * Used to keep the latest messages visible.
 */
export function scrollToBottom() {
    const chatMessages = document.getElementById('chatMessages');
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

/**
 * Escape HTML special characters to prevent XSS attacks.
 * Creates a temporary div element to safely escape text.
 *
 * @param {string} text - The text to escape
 * @returns {string} The escaped HTML string
 */
export function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

/**
 * Display a status message in a container element.
 * Sets the appropriate CSS class based on message type.
 *
 * @param {HTMLElement} container - The container element to display the message in
 * @param {string} type - The message type ('success', 'error', 'info', 'warning')
 * @param {string} message - The message text to display
 */
export function showMessage(container, type, message) {
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
 * Hide the connection message and potentially hide the status container.
 * The status container is hidden only if no other status is being shown.
 */
export function hideConnectionMessage() {
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

/**
 * Display an error popup modal with detailed error information.
 * Supports optional details and connection information for debugging.
 *
 * @param {string} message - The main error message
 * @param {string} [title='Connection Error'] - The modal title
 * @param {string} [details=null] - Additional error details
 * @param {Object} [connectionInfo=null] - Connection parameters to display for debugging
 */
export function showErrorPopup(message, title = 'Connection Error', details = null, connectionInfo = null) {
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

    modal.classList.remove('hidden');
}

/**
 * Close the error modal dialog.
 * If called from a backdrop click event, only closes if clicking the backdrop itself.
 *
 * @param {Event} [event] - Optional click event for backdrop click detection
 */
export function closeErrorModal(event) {
    // If called from backdrop click, only close if clicking the backdrop itself
    if (event && event.target !== event.currentTarget) return;

    const modal = document.getElementById('errorModal');
    modal.classList.add('hidden');
}
