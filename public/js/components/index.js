/**
 * EWR Web Components Registry
 * Central entry point for loading all EWR Web Components
 *
 * Usage in HTML:
 * <script type="module" src="/js/components/index.js"></script>
 *
 * Or import specific components:
 * import { EwrButton, EwrInput } from '/js/components/index.js';
 */

// Base component (not registered, just exported for extension)
export { EwrBaseComponent } from './base/ewr-base-component.js';

// Button components
export { EwrButton } from './buttons/ewr-button.js';

// Form components (Phase 1)
export { EwrInput } from './forms/ewr-input.js';
export { EwrSelect } from './forms/ewr-select.js';
export { EwrCheckbox } from './forms/ewr-checkbox.js';

// Layout components (Phase 2)
export { EwrCard } from './layout/ewr-card.js';
export { EwrCollapsible } from './layout/ewr-collapsible.js';
export { EwrStatusPill } from './layout/ewr-status-pill.js';
export { EwrSpinner } from './layout/ewr-spinner.js';
export { EwrModalComponent } from './layout/ewr-modal.js';

// Chat components (Phase 3)
export { EwrChatMessage } from './chat/ewr-chat-message.js';
export { EwrChatInput } from './chat/ewr-chat-input.js';

// Document components (Phase 4)
export { EwrDocumentRow } from './documents/ewr-document-row.js';

// File upload components (Phase 5)
export { EwrFileDropZone } from './files/ewr-file-drop-zone.js';
export { EwrFileRow } from './files/ewr-file-row.js';
export { EwrFileGrid } from './files/ewr-file-grid.js';

// Audio components (Phase 6)
export { EwrAudioPlayer } from './audio/ewr-audio-player.js';
export { EwrAudioResultRow } from './audio/ewr-audio-result-row.js';
export { EwrAudioResultsGrid } from './audio/ewr-audio-results-grid.js';

// Status components (Phase 7)
export { EwrWebsiteStatus } from './status/ewr-website-status.js';

/**
 * Component registration helper
 * Safely registers a component, logging a warning if already registered
 */
function safeDefine(tagName, componentClass) {
    if (!customElements.get(tagName)) {
        customElements.define(tagName, componentClass);
    } else {
        console.warn(`EWR Components: ${tagName} is already registered`);
    }
}

/**
 * Initialize all EWR components
 * Called automatically when this module is imported
 */
async function initComponents() {
    try {
        // Phase 1: Button components
        const { EwrButton } = await import('./buttons/ewr-button.js');
        safeDefine('ewr-btn', EwrButton);

        // Phase 1: Form components
        const { EwrInput } = await import('./forms/ewr-input.js');
        safeDefine('ewr-input', EwrInput);

        const { EwrSelect } = await import('./forms/ewr-select.js');
        safeDefine('ewr-select', EwrSelect);

        const { EwrCheckbox } = await import('./forms/ewr-checkbox.js');
        safeDefine('ewr-checkbox', EwrCheckbox);

        // Phase 2: Layout components
        const { EwrCard } = await import('./layout/ewr-card.js');
        safeDefine('ewr-card', EwrCard);

        const { EwrCollapsible } = await import('./layout/ewr-collapsible.js');
        safeDefine('ewr-collapsible', EwrCollapsible);

        const { EwrStatusPill } = await import('./layout/ewr-status-pill.js');
        safeDefine('ewr-status-pill', EwrStatusPill);

        const { EwrSpinner } = await import('./layout/ewr-spinner.js');
        safeDefine('ewr-spinner', EwrSpinner);

        const { EwrModalComponent } = await import('./layout/ewr-modal.js');
        safeDefine('ewr-modal', EwrModalComponent);

        // Phase 3: Chat components
        const { EwrChatMessage } = await import('./chat/ewr-chat-message.js');
        safeDefine('ewr-chat-message', EwrChatMessage);

        const { EwrChatInput } = await import('./chat/ewr-chat-input.js');
        safeDefine('ewr-chat-input', EwrChatInput);

        // Phase 4: Document components
        const { EwrDocumentRow } = await import('./documents/ewr-document-row.js');
        safeDefine('ewr-document-row', EwrDocumentRow);

        // Phase 5: File upload components
        const { EwrFileDropZone } = await import('./files/ewr-file-drop-zone.js');
        safeDefine('ewr-file-drop-zone', EwrFileDropZone);

        const { EwrFileRow } = await import('./files/ewr-file-row.js');
        safeDefine('ewr-file-row', EwrFileRow);

        const { EwrFileGrid } = await import('./files/ewr-file-grid.js');
        safeDefine('ewr-file-grid', EwrFileGrid);

        // Phase 6: Audio components
        const { EwrAudioPlayer } = await import('./audio/ewr-audio-player.js');
        safeDefine('ewr-audio-player', EwrAudioPlayer);

        const { EwrAudioResultRow } = await import('./audio/ewr-audio-result-row.js');
        safeDefine('ewr-audio-result-row', EwrAudioResultRow);

        const { EwrAudioResultsGrid } = await import('./audio/ewr-audio-results-grid.js');
        safeDefine('ewr-audio-results-grid', EwrAudioResultsGrid);

        // Phase 7: Status components
        const { EwrWebsiteStatus } = await import('./status/ewr-website-status.js');
        safeDefine('ewr-website-status', EwrWebsiteStatus);

        console.log('EWR Web Components initialized (19 components)');
    } catch (error) {
        console.error('Failed to initialize EWR components:', error);
    }
}

// Auto-initialize when loaded as a module
if (typeof window !== 'undefined') {
    // Use DOMContentLoaded if document not ready, otherwise init immediately
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', initComponents);
    } else {
        initComponents();
    }
}

export { initComponents, safeDefine };
