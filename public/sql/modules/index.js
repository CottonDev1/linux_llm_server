/**
 * SQL Chat Application - Main Entry Point
 *
 * This module imports all sub-modules and exposes functions to the global window
 * object for HTML onclick handlers and external access.
 *
 * @module sql-chat
 */

// ============================================================================
// Imports
// ============================================================================

// State management (must be first - other modules depend on it)
import {
    SQL_API_BASE,
    PIPELINE_STEPS,
    state,
    resetSessionMetrics as resetSessionMetricsState,
    resetPipelineTiming as resetPipelineTimingState,
    resetAppState,
    resetTokenTracker as resetTokenTrackerState,
    clearChatHistory,
    addToChatHistory,
    updateTokenTracker as updateTokenTrackerState,
    startPipelineTiming,
    recordStepTiming,
    getPipelineDuration
} from './sql-chat-state.js';

// UI Manager
import {
    updateUIState,
    updateConnectionStatus as updateConnectionStatusUI,
    togglePanel,
    updateSendButton,
    autoResizeTextarea,
    scrollToBottom,
    escapeHtml,
    showMessage,
    hideConnectionMessage,
    showErrorPopup,
    closeErrorModal
} from './ui-manager.js';

// Connection Manager
import {
    saveConnectionSettings,
    loadSavedConnectionSettings,
    validateConnectionFields,
    updateConnectionStatus as updateConnectionStatusConnection,
    testConnection,
    disableConnectionFields,
    getConnectionConfig,
    toggleAuthFields,
    toggleIntegratedAuth,
    toggleShowPassword,
    resetConnectionSettings,
    clearConnection,
    clearQueryCache,
    saveAsDefault
} from './connection-manager.js';

// Schema Manager
import {
    loadDatabase,
    loadDatabases,
    checkDatabaseSchema,
    extractSchemaWithProgress,
    handleExtractionProgress,
    pollSchemaAnalysisStatus
} from './schema-manager.js';

// Chat Manager
import {
    sendSampleQuery,
    handleKeyDown,
    sendMessage,
    addUserMessage,
    addAssistantMessage,
    addErrorMessage,
    formatErrorMessage,
    toggleErrorDetails,
    addLoadingMessage,
    updateLoadingMessage,
    removeLoadingMessage,
    clearChat
} from './chat-manager.js';

// Results Display
import {
    createSqlBlock,
    createResultsTable,
    exportTableToCSV,
    showResultSql,
    copySqlToClipboard
} from './results-display.js';

// Token Tracker
import {
    loadConversationContextSetting,
    toggleConversationContext,
    getConversationForContext,
    initializeContextRemainingDisplay,
    updateTokenDisplay,
    showContextLimitAlert,
    hideContextLimitAlert,
    disableChatInput,
    estimateTokens,
    updateTokenTrackerFromHistory,
    updateTokenTracker
} from './token-tracker.js';

// Pipeline Timing
import {
    initPipelineTiming,
    updatePipelineStep,
    updateGeneratingStep,
    finalizePipelineTiming,
    formatDuration,
    getSpeedClass,
    clearTimingDisplay,
    updateTimingDisplay,
    getPipelineTimingTracker
} from './pipeline-timing.js';

// Session Metrics
import {
    setTokenDisplayCallback,
    updateSessionMetrics,
    displaySessionMetrics,
    resetSessionMetrics
} from './session-metrics.js';

// Rules Manager
import {
    showAddExampleModal,
    closeExampleModal,
    saveExample,
    testExampleSql,
    showAddRuleModal,
    generateRuleWithAI,
    populateRuleFields,
    closeAddRuleModal,
    showAddRuleMessage,
    saveRule,
    showRulesModal,
    renderRulesList,
    renderRuleCard,
    closeRulesListModal
} from './rules-manager.js';

// Feedback Manager
import {
    getCurrentDatabase,
    getUserRole,
    handlePositiveFeedback,
    handleNegativeFeedback,
    openSqlFeedbackModal,
    closeSqlFeedbackModal,
    saveSqlFeedback,
    submitFeedback,
    showFeedbackToast,
    showFeedbackReviewModal,
    closeFeedbackReviewModal,
    loadFeedbackList,
    updateFeedbackBadge,
    renderFeedbackItem,
    generateRuleFromFeedback,
    markFeedbackProcessed,
    showRulePreviewModal,
    closeRulePreviewModal,
    createRuleFromPreview,
    loadFeedbackCount,
    initializeFeedbackManager
} from './feedback-manager.js';

// ============================================================================
// Global Exports (for HTML onclick handlers)
// ============================================================================

// State
window.SQL_API_BASE = SQL_API_BASE;
window.PIPELINE_STEPS = PIPELINE_STEPS;
window.state = state;

// Connection management
window.saveConnectionSettings = saveConnectionSettings;
window.loadSavedConnectionSettings = loadSavedConnectionSettings;
window.testConnection = testConnection;
window.toggleAuthFields = toggleAuthFields;
window.toggleIntegratedAuth = toggleIntegratedAuth;
window.toggleShowPassword = toggleShowPassword;
window.resetConnectionSettings = resetConnectionSettings;
window.clearConnection = clearConnection;
window.clearQueryCache = clearQueryCache;
window.saveAsDefault = saveAsDefault;
window.getConnectionConfig = getConnectionConfig;

// Database/Schema management
window.loadDatabase = loadDatabase;
window.loadDatabases = loadDatabases;
window.checkDatabaseSchema = checkDatabaseSchema;

// Chat management
window.sendSampleQuery = sendSampleQuery;
window.handleKeyDown = handleKeyDown;
window.sendMessage = sendMessage;
window.clearChat = clearChat;
window.toggleErrorDetails = toggleErrorDetails;

// Results display
window.exportTableToCSV = exportTableToCSV;
window.showResultSql = showResultSql;
window.copySqlToClipboard = copySqlToClipboard;

// Token/Context management
window.toggleConversationContext = toggleConversationContext;

// UI management
window.togglePanel = togglePanel;
window.autoResizeTextarea = autoResizeTextarea;
window.closeErrorModal = closeErrorModal;

// Rules management
window.showAddExampleModal = showAddExampleModal;
window.closeExampleModal = closeExampleModal;
window.saveExample = saveExample;
window.testExampleSql = testExampleSql;
window.showAddRuleModal = showAddRuleModal;
window.generateRuleWithAI = generateRuleWithAI;
window.closeAddRuleModal = closeAddRuleModal;
window.saveRule = saveRule;
window.showRulesModal = showRulesModal;
window.closeRulesListModal = closeRulesListModal;

// Feedback management
window.handlePositiveFeedback = handlePositiveFeedback;
window.handleNegativeFeedback = handleNegativeFeedback;
window.openSqlFeedbackModal = openSqlFeedbackModal;
window.closeSqlFeedbackModal = closeSqlFeedbackModal;
window.saveSqlFeedback = saveSqlFeedback;
window.showFeedbackReviewModal = showFeedbackReviewModal;
window.closeFeedbackReviewModal = closeFeedbackReviewModal;
window.closeRulePreviewModal = closeRulePreviewModal;
window.createRuleFromPreview = createRuleFromPreview;

// ============================================================================
// Initialization
// ============================================================================

/**
 * Initialize the SQL Chat application
 * Sets up all event listeners and loads initial state
 */
async function initializeApp() {
    console.log('SQL Chat: Initializing application...');

    try {
        // Set up token display callback for session metrics
        setTokenDisplayCallback(updateTokenDisplay);

        // Load saved connection settings
        await loadSavedConnectionSettings();

        // Load conversation context setting
        loadConversationContextSetting();

        // Initialize context remaining display
        initializeContextRemainingDisplay();

        // Update UI state
        updateUIState();
        updateSendButton();

        // Initialize feedback manager
        initializeFeedbackManager();

        // Set up event listeners
        setupEventListeners();

        console.log('SQL Chat: Initialization complete');
    } catch (error) {
        console.error('SQL Chat: Initialization error:', error);
    }
}

/**
 * Set up DOM event listeners
 */
function setupEventListeners() {
    // User input auto-resize
    const userInput = document.getElementById('userInput');
    if (userInput) {
        userInput.addEventListener('input', function() {
            autoResizeTextarea(this);
        });
        userInput.addEventListener('keydown', handleKeyDown);
    }

    // Database select change
    const databaseSelect = document.getElementById('databaseSelect');
    if (databaseSelect) {
        databaseSelect.addEventListener('change', loadDatabase);
    }

    // Conversation context checkbox
    const conversationContextCheckbox = document.getElementById('conversationContext');
    if (conversationContextCheckbox) {
        conversationContextCheckbox.addEventListener('change', toggleConversationContext);
    }

    // Auth type select
    const authTypeSelect = document.getElementById('authType');
    if (authTypeSelect) {
        authTypeSelect.addEventListener('change', toggleAuthFields);
    }

    // Integrated auth checkbox
    const integratedAuthCheckbox = document.getElementById('integratedAuth');
    if (integratedAuthCheckbox) {
        integratedAuthCheckbox.addEventListener('change', toggleIntegratedAuth);
    }

    // Close modals on outside click
    document.addEventListener('click', function(event) {
        // Error modal
        if (event.target.classList.contains('error-modal')) {
            closeErrorModal(event);
        }
        // Example modal
        if (event.target.id === 'exampleModal') {
            closeExampleModal(event);
        }
        // Add rule modal
        if (event.target.id === 'addRuleModal') {
            closeAddRuleModal(event);
        }
        // Rules list modal
        if (event.target.id === 'rulesListModal') {
            closeRulesListModal(event);
        }
        // SQL feedback modal
        if (event.target.id === 'sqlFeedbackModal') {
            closeSqlFeedbackModal(event);
        }
        // Feedback review modal
        if (event.target.id === 'feedbackReviewModal') {
            closeFeedbackReviewModal(event);
        }
        // Rule preview modal
        if (event.target.id === 'rulePreviewModal') {
            closeRulePreviewModal(event);
        }
    });

    // Escape key to close modals
    document.addEventListener('keydown', function(event) {
        if (event.key === 'Escape') {
            closeErrorModal(event);
            closeExampleModal(event);
            closeAddRuleModal(event);
            closeRulesListModal(event);
            closeSqlFeedbackModal(event);
            closeFeedbackReviewModal(event);
            closeRulePreviewModal(event);
        }
    });
}

// Initialize when DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initializeApp);
} else {
    initializeApp();
}

// ============================================================================
// Re-exports for programmatic access
// ============================================================================

export {
    // State
    SQL_API_BASE,
    PIPELINE_STEPS,
    state,
    resetAppState,
    clearChatHistory,
    addToChatHistory,
    startPipelineTiming,
    recordStepTiming,
    getPipelineDuration,

    // UI
    updateUIState,
    togglePanel,
    updateSendButton,
    autoResizeTextarea,
    scrollToBottom,
    escapeHtml,
    showMessage,
    hideConnectionMessage,
    showErrorPopup,
    closeErrorModal,

    // Connection
    saveConnectionSettings,
    loadSavedConnectionSettings,
    validateConnectionFields,
    testConnection,
    disableConnectionFields,
    getConnectionConfig,
    toggleAuthFields,
    toggleIntegratedAuth,
    toggleShowPassword,
    resetConnectionSettings,
    clearConnection,
    clearQueryCache,
    saveAsDefault,

    // Schema
    loadDatabase,
    loadDatabases,
    checkDatabaseSchema,
    extractSchemaWithProgress,
    handleExtractionProgress,
    pollSchemaAnalysisStatus,

    // Chat
    sendSampleQuery,
    handleKeyDown,
    sendMessage,
    addUserMessage,
    addAssistantMessage,
    addErrorMessage,
    formatErrorMessage,
    toggleErrorDetails,
    addLoadingMessage,
    updateLoadingMessage,
    removeLoadingMessage,
    clearChat,

    // Results
    createSqlBlock,
    createResultsTable,
    exportTableToCSV,
    showResultSql,
    copySqlToClipboard,

    // Token Tracker
    loadConversationContextSetting,
    toggleConversationContext,
    getConversationForContext,
    initializeContextRemainingDisplay,
    updateTokenDisplay,
    showContextLimitAlert,
    hideContextLimitAlert,
    disableChatInput,
    estimateTokens,
    updateTokenTrackerFromHistory,
    updateTokenTracker,

    // Pipeline Timing
    initPipelineTiming,
    updatePipelineStep,
    updateGeneratingStep,
    finalizePipelineTiming,
    formatDuration,
    getSpeedClass,
    clearTimingDisplay,
    updateTimingDisplay,
    getPipelineTimingTracker,

    // Session Metrics
    setTokenDisplayCallback,
    updateSessionMetrics,
    displaySessionMetrics,
    resetSessionMetrics,

    // Rules
    showAddExampleModal,
    closeExampleModal,
    saveExample,
    testExampleSql,
    showAddRuleModal,
    generateRuleWithAI,
    populateRuleFields,
    closeAddRuleModal,
    showAddRuleMessage,
    saveRule,
    showRulesModal,
    renderRulesList,
    renderRuleCard,
    closeRulesListModal,

    // Feedback
    getCurrentDatabase,
    getUserRole,
    handlePositiveFeedback,
    handleNegativeFeedback,
    openSqlFeedbackModal,
    closeSqlFeedbackModal,
    saveSqlFeedback,
    submitFeedback,
    showFeedbackToast,
    showFeedbackReviewModal,
    closeFeedbackReviewModal,
    loadFeedbackList,
    updateFeedbackBadge,
    renderFeedbackItem,
    generateRuleFromFeedback,
    markFeedbackProcessed,
    showRulePreviewModal,
    closeRulePreviewModal,
    createRuleFromPreview,
    loadFeedbackCount,
    initializeFeedbackManager
};
