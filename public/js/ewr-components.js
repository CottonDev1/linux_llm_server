/**
 * EWR Web Components
 * Reusable custom HTML elements for EWR applications
 */

/**
 * EWR-ChatBox Component
 * A full-featured chat interface card with customizable settings panel
 *
 * Usage:
 * <ewr-chatbox
 *     messages-id="chatMessages"
 *     input-id="chatInput"
 *     send-btn-id="sendBtn"
 *     placeholder="Ask a question..."
 *     empty-icon="💬"
 *     empty-title="Start a Conversation"
 *     empty-text="Ask questions in natural language.">
 *
 *     <div slot="settings-panel">
 *         <!-- Your collapsible or non-collapsible settings -->
 *     </div>
 *
 *     <div slot="context-display">
 *         <!-- Context/token display elements -->
 *     </div>
 *
 *     <div slot="timing-metrics">
 *         <!-- Optional timing metrics -->
 *     </div>
 * </ewr-chatbox>
 */
class EwrChatBox extends HTMLElement {
    constructor() {
        super();
    }

    connectedCallback() {
        // Get attributes with defaults
        const messagesId = this.getAttribute('messages-id') || 'chatMessages';
        const inputId = this.getAttribute('input-id') || 'chatInput';
        const sendBtnId = this.getAttribute('send-btn-id') || 'sendBtn';
        const placeholder = this.getAttribute('placeholder') || 'Ask a question...';
        const emptyIcon = this.getAttribute('empty-icon') || '💬';
        const emptyTitle = this.getAttribute('empty-title') || 'Start a Conversation';
        const emptyText = this.getAttribute('empty-text') || 'Ask questions in natural language.';
        const onSend = this.getAttribute('on-send') || 'sendMessage()';
        const onClear = this.getAttribute('on-clear') || 'clearChat()';
        const onKeyDown = this.getAttribute('on-keydown') || 'handleKeyDown(event)';

        // Capture slotted children before replacing innerHTML (slots don't work without Shadow DOM)
        // Clone the entire subtree to preserve nested custom elements
        const settingsPanel = this.querySelector('[slot="settings-panel"]');
        const contextDisplay = this.querySelector('[slot="context-display"]');
        const timingMetrics = this.querySelector('[slot="timing-metrics"]');

        // Extract children arrays before any DOM manipulation
        const settingsChildren = settingsPanel ? Array.from(settingsPanel.children) : [];
        const contextChildren = contextDisplay ? Array.from(contextDisplay.children) : [];

        // Build the component HTML
        this.innerHTML = `
            <div class="card ewr_chat-card-full">
                <!-- Settings Panel Area -->
                <div class="ewr-chatbox-settings"></div>

                <!-- Card Body with Chat -->
                <div class="card-body">
                    <div class="ewr-chat-container">
                        <!-- Chat Messages Area -->
                        <div class="ewr-chat-messages chat-messages" id="${messagesId}">
                            <div class="empty-chat" id="emptyChat">
                                <div class="empty-icon">${emptyIcon}</div>
                                <div class="empty-title">${emptyTitle}</div>
                                <div class="empty-text">${emptyText}</div>
                            </div>
                        </div>

                        <!-- LLM Metrics Display Area -->
                        <div class="ewr-chat-llm-metrics"></div>

                        <!-- Timing Metrics Area -->
                        <div class="ewr-chatbox-timing"></div>

                        <!-- Chat Footer -->
                        <div class="ewr-chat-footer">
                            <div id="contextLimitAlert" class="context-limit-alert" style="display: none;">
                                Context window full - click Clear Chat to continue
                            </div>
                            <div class="ewr-chat-input-row">
                                <textarea
                                    class="ewr-chat-entry"
                                    id="${inputId}"
                                    placeholder="${placeholder}"
                                    rows="1"
                                    onkeydown="${onKeyDown}"
                                ></textarea>
                                <div class="ewr-chat-buttons">
                                    <button class="ewr-button-send" id="${sendBtnId}" onclick="${onSend}" title="Send">&#10148;</button>
                                    <button class="ewr-button-clear" onclick="${onClear}">Clear</button>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        `;

        // Move captured slotted content into appropriate areas
        const settingsArea = this.querySelector('.ewr-chatbox-settings');
        settingsChildren.forEach(child => {
            settingsArea.appendChild(child);
        });

        const metricsArea = this.querySelector('.ewr-chat-llm-metrics');
        contextChildren.forEach(child => {
            metricsArea.appendChild(child);
        });

        if (timingMetrics) {
            const timingArea = this.querySelector('.ewr-chatbox-timing');
            // Move the entire timing metrics element (preserving its classes/id)
            timingArea.parentNode.replaceChild(timingMetrics, timingArea);
            timingMetrics.removeAttribute('slot');
        }
    }
}

// Register the custom element
customElements.define('ewr-chatbox', EwrChatBox);


/**
 * EWR-CollapsiblePanel Component
 * A collapsible settings panel with header, content, and footer
 *
 * Usage:
 * <ewr-collapsible-panel id="myPanel" title="Settings">
 *     <div slot="content">
 *         <!-- Panel content -->
 *     </div>
 *     <div slot="footer">
 *         <!-- Footer buttons -->
 *     </div>
 * </ewr-collapsible-panel>
 */
class EwrCollapsiblePanel extends HTMLElement {
    constructor() {
        super();
    }

    connectedCallback() {
        const panelId = this.getAttribute('id') || 'collapsiblePanel';
        const title = this.getAttribute('title') || 'Settings';
        const collapsed = this.hasAttribute('collapsed');

        // Remove ID from custom element to avoid duplicate IDs
        this.removeAttribute('id');

        this.innerHTML = `
            <div class="ewr-collapsible ${collapsed ? 'collapsed' : ''}" id="${panelId}">
                <div class="ewr-collapsible-header" onclick="toggleCollapsible('${panelId}')">
                    <span class="ewr-collapsible-title">${title}</span>
                    <button class="ewr-collapse-toggle" type="button">
                        <span class="toggle-icon">▼</span>
                    </button>
                </div>
                <div class="ewr-collapsible-content">
                    <slot name="content"></slot>
                </div>
                <div class="ewr-collapsible-footer">
                    <slot name="footer"></slot>
                </div>
            </div>
        `;
    }
}

customElements.define('ewr-collapsible-panel', EwrCollapsiblePanel);


/**
 * EWR-NonCollapsiblePanel Component
 * A non-collapsible settings panel with header and content
 *
 * Usage:
 * <ewr-noncollapsible-panel title="Filter Settings">
 *     <div slot="content">
 *         <!-- Panel content -->
 *     </div>
 * </ewr-noncollapsible-panel>
 */
class EwrNonCollapsiblePanel extends HTMLElement {
    constructor() {
        super();
    }

    connectedCallback() {
        const title = this.getAttribute('title') || 'Settings';

        this.innerHTML = `
            <div class="ewr-noncollapsible-div">
                <div class="ewr-noncollapsible-header">
                    <span class="ewr-noncollapsible-title">${title}</span>
                </div>
                <div class="ewr-noncollapsible-content">
                    <slot name="content"></slot>
                </div>
            </div>
        `;
    }
}

customElements.define('ewr-noncollapsible-panel', EwrNonCollapsiblePanel);


/**
 * EWR-FilterDiv Component
 * A filter container with a title header and customizable filter controls
 * Child elements (ewr-horiz-select, ewr-horiz-input, etc.) are moved into the filter row
 *
 * Usage:
 * <ewr-filter-div title="Filter Settings">
 *     <ewr-horiz-select label="Department" id="deptFilter"></ewr-horiz-select>
 *     <ewr-horiz-input label="Keywords" id="keywordFilter"></ewr-horiz-input>
 * </ewr-filter-div>
 */
class EwrFilterDiv extends HTMLElement {
    constructor() {
        super();
    }

    connectedCallback() {
        const title = this.getAttribute('title') || 'Filter Settings';

        // Capture child elements before replacing innerHTML
        const children = Array.from(this.children);

        // Create the container structure
        this.innerHTML = `
            <div class="ewr-noncollapsible-div">
                <div class="ewr-noncollapsible-header">
                    <span class="ewr-noncollapsible-title">${title}</span>
                </div>
                <div class="ewr-noncollapsible-content">
                    <div class="ewr-filter-row" id="filterRow"></div>
                </div>
            </div>
        `;

        // Move captured children into the filter row
        const filterRow = this.querySelector('.ewr-filter-row');
        children.forEach(child => {
            // Remove the slot attribute as we're not using Shadow DOM
            child.removeAttribute('slot');
            filterRow.appendChild(child);
        });
    }
}

customElements.define('ewr-filter-div', EwrFilterDiv);


/**
 * EWR-HorizSelect Component
 * A horizontal label + select dropdown control for filter rows
 *
 * Usage:
 * <ewr-horiz-select
 *     label="Department"
 *     id="deptFilter"
 *     onchange="handleChange()"
 *     default-text="All Departments">
 *     <option value="sales">Sales</option>
 *     <option value="support">Support</option>
 * </ewr-horiz-select>
 */
class EwrHorizSelect extends HTMLElement {
    constructor() {
        super();
    }

    connectedCallback() {
        const label = this.getAttribute('label') || 'Select';
        const id = this.getAttribute('id') || 'horizSelect';
        const onchange = this.getAttribute('onchange') || '';
        const defaultText = this.getAttribute('default-text') || '';
        const disabled = this.hasAttribute('disabled');

        // Capture existing options before replacing innerHTML
        const existingOptions = Array.from(this.querySelectorAll('option'));

        // Build options HTML
        let optionsHtml = defaultText ? `<option value="">${defaultText}</option>` : '';
        existingOptions.forEach(opt => {
            optionsHtml += opt.outerHTML;
        });

        this.innerHTML = `
            <div class="ewr-edit-div">
                <label class="ewr-label" for="${id}">${label}</label>
                <select id="${id}" class="ewr-select-edit" ${onchange ? `onchange="${onchange}"` : ''} ${disabled ? 'disabled' : ''}>
                    ${optionsHtml}
                </select>
            </div>
        `;
    }

    // Method to populate options programmatically
    setOptions(options, defaultText = '') {
        const select = this.querySelector('select');
        if (select) {
            let html = defaultText ? `<option value="">${defaultText}</option>` : '';
            options.forEach(opt => {
                const value = opt.value || opt.id || opt.name;
                const text = opt.text || opt.name || opt.id;
                html += `<option value="${value}">${text}</option>`;
            });
            select.innerHTML = html;
        }
    }

    // Get current value
    get value() {
        const select = this.querySelector('select');
        return select ? select.value : '';
    }

    // Set current value
    set value(val) {
        const select = this.querySelector('select');
        if (select) select.value = val;
    }
}

customElements.define('ewr-horiz-select', EwrHorizSelect);


/**
 * EWR-HorizInput Component
 * A horizontal label + text input control for filter rows
 *
 * Usage:
 * <ewr-horiz-input
 *     label="Keywords"
 *     id="keywordFilter"
 *     placeholder="e.g., safety, inspection..."
 *     type="text">
 * </ewr-horiz-input>
 */
class EwrHorizInput extends HTMLElement {
    constructor() {
        super();
    }

    connectedCallback() {
        const label = this.getAttribute('label') || 'Input';
        const id = this.getAttribute('id') || 'horizInput';
        const placeholder = this.getAttribute('placeholder') || '';
        const type = this.getAttribute('type') || 'text';
        const disabled = this.hasAttribute('disabled');
        const onchange = this.getAttribute('onchange') || '';
        const oninput = this.getAttribute('oninput') || '';

        this.innerHTML = `
            <div class="ewr-edit-div">
                <label class="ewr-label" for="${id}">${label}</label>
                <input
                    type="${type}"
                    id="${id}"
                    class="ewr-input-edits"
                    placeholder="${placeholder}"
                    ${disabled ? 'disabled' : ''}
                    ${onchange ? `onchange="${onchange}"` : ''}
                    ${oninput ? `oninput="${oninput}"` : ''}>
            </div>
        `;
    }

    // Get current value
    get value() {
        const input = this.querySelector('input');
        return input ? input.value : '';
    }

    // Set current value
    set value(val) {
        const input = this.querySelector('input');
        if (input) input.value = val;
    }
}

customElements.define('ewr-horiz-input', EwrHorizInput);


/**
 * EWR-EditDivVert Component
 * A vertical layout with label above the input/select control
 * Modern styling with rounded corners, subtle shadows, and smooth transitions
 *
 * Usage:
 * <ewr-edit-div-vert
 *     label="Department"
 *     type="select"
 *     id="deptFilter"
 *     default-text="All Departments">
 * </ewr-edit-div-vert>
 *
 * <ewr-edit-div-vert
 *     label="Search"
 *     type="input"
 *     id="searchFilter"
 *     placeholder="Search documents...">
 * </ewr-edit-div-vert>
 */
class EwrEditDivVert extends HTMLElement {
    constructor() {
        super();
    }

    connectedCallback() {
        const label = this.getAttribute('label') || 'Label';
        const type = this.getAttribute('type') || 'input'; // 'input' or 'select'
        const placeholder = this.getAttribute('placeholder') || '';
        const defaultText = this.getAttribute('default-text') || '';
        const disabled = this.hasAttribute('disabled');
        const onchange = this.getAttribute('onchange') || '';
        const oninput = this.getAttribute('oninput') || '';
        const inputType = this.getAttribute('input-type') || 'text';

        // Generate unique internal ID for label association (avoid duplicate IDs)
        const internalId = `ewr-vert-${Math.random().toString(36).substr(2, 9)}`;

        // Capture existing options for select type
        const existingOptions = Array.from(this.querySelectorAll('option'));
        let optionsHtml = '';
        if (defaultText) {
            optionsHtml = `<option value="">${defaultText}</option>`;
        }
        existingOptions.forEach(opt => {
            optionsHtml += opt.outerHTML;
        });

        let controlHtml = '';
        if (type === 'select') {
            controlHtml = `
                <select id="${internalId}" class="ewr-edit-modern" ${onchange ? `onchange="${onchange}"` : ''} ${disabled ? 'disabled' : ''}>
                    ${optionsHtml}
                </select>
            `;
        } else {
            controlHtml = `
                <input
                    type="${inputType}"
                    id="${internalId}"
                    class="ewr-edit-modern"
                    placeholder="${placeholder}"
                    ${disabled ? 'disabled' : ''}
                    ${onchange ? `onchange="${onchange}"` : ''}
                    ${oninput ? `oninput="${oninput}"` : ''}>
            `;
        }

        this.innerHTML = `
            <div class="ewr-edit-div-vert">
                <label class="ewr-edit-label-modern" for="${internalId}">${label}</label>
                ${controlHtml}
            </div>
        `;
    }

    // Method to populate options programmatically (for select type)
    setOptions(options, defaultText = '') {
        const select = this.querySelector('select');
        if (select) {
            let html = defaultText ? `<option value="">${defaultText}</option>` : '';
            options.forEach(opt => {
                const value = opt.value || opt.id || opt.name || opt;
                const text = opt.text || opt.name || opt.id || opt;
                html += `<option value="${value}">${text}</option>`;
            });
            select.innerHTML = html;
        }
    }

    // Get current value
    get value() {
        const control = this.querySelector('select, input');
        return control ? control.value : '';
    }

    // Set current value
    set value(val) {
        const control = this.querySelector('select, input');
        if (control) control.value = val;
    }

    // Get/set disabled state
    get disabled() {
        const control = this.querySelector('select, input');
        return control ? control.disabled : false;
    }

    set disabled(val) {
        const control = this.querySelector('select, input');
        if (control) control.disabled = val;
    }
}

customElements.define('ewr-edit-div-vert', EwrEditDivVert);


/**
 * EWR-EditDivHoriz Component
 * A horizontal layout with label beside the input/select control
 * Modern styling with rounded corners, subtle shadows, and smooth transitions
 *
 * Usage:
 * <ewr-edit-div-horiz
 *     label="Department"
 *     type="select"
 *     id="deptFilter"
 *     default-text="All Departments">
 * </ewr-edit-div-horiz>
 *
 * <ewr-edit-div-horiz
 *     label="Keywords"
 *     type="input"
 *     id="keywordFilter"
 *     placeholder="Enter keywords...">
 * </ewr-edit-div-horiz>
 */
class EwrEditDivHoriz extends HTMLElement {
    constructor() {
        super();
    }

    connectedCallback() {
        const label = this.getAttribute('label') || 'Label';
        const type = this.getAttribute('type') || 'input'; // 'input' or 'select'
        const placeholder = this.getAttribute('placeholder') || '';
        const defaultText = this.getAttribute('default-text') || '';
        const disabled = this.hasAttribute('disabled');
        const onchange = this.getAttribute('onchange') || '';
        const oninput = this.getAttribute('oninput') || '';
        const inputType = this.getAttribute('input-type') || 'text';
        const labelWidth = this.getAttribute('label-width') || 'auto';

        // Generate unique internal ID for label association (avoid duplicate IDs)
        const internalId = `ewr-horiz-${Math.random().toString(36).substr(2, 9)}`;

        // Capture existing options for select type
        const existingOptions = Array.from(this.querySelectorAll('option'));
        let optionsHtml = '';
        if (defaultText) {
            optionsHtml = `<option value="">${defaultText}</option>`;
        }
        existingOptions.forEach(opt => {
            optionsHtml += opt.outerHTML;
        });

        let controlHtml = '';
        if (type === 'select') {
            controlHtml = `
                <select id="${internalId}" class="ewr-edit-modern" ${onchange ? `onchange="${onchange}"` : ''} ${disabled ? 'disabled' : ''}>
                    ${optionsHtml}
                </select>
            `;
        } else {
            controlHtml = `
                <input
                    type="${inputType}"
                    id="${internalId}"
                    class="ewr-edit-modern"
                    placeholder="${placeholder}"
                    ${disabled ? 'disabled' : ''}
                    ${onchange ? `onchange="${onchange}"` : ''}
                    ${oninput ? `oninput="${oninput}"` : ''}>
            `;
        }

        const labelStyle = labelWidth !== 'auto' ? `style="min-width: ${labelWidth};"` : '';

        this.innerHTML = `
            <div class="ewr-edit-div-horiz">
                <label class="ewr-edit-label-modern" for="${internalId}" ${labelStyle}>${label}</label>
                ${controlHtml}
            </div>
        `;
    }

    // Method to populate options programmatically (for select type)
    setOptions(options, defaultText = '') {
        const select = this.querySelector('select');
        if (select) {
            let html = defaultText ? `<option value="">${defaultText}</option>` : '';
            options.forEach(opt => {
                const value = opt.value || opt.id || opt.name || opt;
                const text = opt.text || opt.name || opt.id || opt;
                html += `<option value="${value}">${text}</option>`;
            });
            select.innerHTML = html;
        }
    }

    // Get current value
    get value() {
        const control = this.querySelector('select, input');
        return control ? control.value : '';
    }

    // Set current value
    set value(val) {
        const control = this.querySelector('select, input');
        if (control) control.value = val;
    }

    // Get/set disabled state
    get disabled() {
        const control = this.querySelector('select, input');
        return control ? control.disabled : false;
    }

    set disabled(val) {
        const control = this.querySelector('select, input');
        if (control) control.disabled = val;
    }
}

customElements.define('ewr-edit-div-horiz', EwrEditDivHoriz);


/**
 * EWR-ChatEntry Component
 * A complete chat footer with input textarea, send button, clear button, and context limit alert
 *
 * Usage:
 * <ewr-chat-entry
 *     input-id="chatInput"
 *     send-btn-id="sendButton"
 *     placeholder="Ask a question..."
 *     on-send="sendMessage()"
 *     on-clear="clearChat()"
 *     on-keydown="handleKeydown(event)">
 * </ewr-chat-entry>
 */
class EwrChatEntry extends HTMLElement {
    constructor() {
        super();
    }

    connectedCallback() {
        const inputId = this.getAttribute('input-id') || 'chatInput';
        const sendBtnId = this.getAttribute('send-btn-id') || 'sendButton';
        const placeholder = this.getAttribute('placeholder') || 'Ask a question...';
        const onSend = this.getAttribute('on-send') || 'sendMessage()';
        const onClear = this.getAttribute('on-clear') || 'clearChat()';
        const onKeyDown = this.getAttribute('on-keydown') || 'handleKeyDown(event)';
        const alertId = this.getAttribute('alert-id') || 'contextLimitAlert';
        const alertText = this.getAttribute('alert-text') || 'Context window full - click Clear Chat to continue';
        const sendText = this.getAttribute('send-text') || '&#10148;';
        const clearText = this.getAttribute('clear-text') || 'Clear';

        this.innerHTML = `
            <div class="ewr-chat-footer">
                <div id="${alertId}" class="context-limit-alert" style="display: none;">
                    ${alertText}
                </div>
                <div class="ewr-chat-input-row">
                    <textarea
                        class="ewr-chat-entry"
                        id="${inputId}"
                        placeholder="${placeholder}"
                        rows="1"
                        onkeydown="${onKeyDown}"
                    ></textarea>
                    <div class="ewr-chat-buttons">
                        <button class="ewr-button-send" id="${sendBtnId}" onclick="${onSend}" title="Send">${sendText}</button>
                        <button class="ewr-button-clear" onclick="${onClear}">${clearText}</button>
                    </div>
                </div>
            </div>
        `;
    }

    // Disable/enable the input and send button
    setDisabled(disabled) {
        const input = this.querySelector('textarea');
        const sendBtn = this.querySelector('.ewr-button-send');
        if (input) input.disabled = disabled;
        if (sendBtn) sendBtn.disabled = disabled;
    }

    // Show/hide the context limit alert
    showAlert(show) {
        const alert = this.querySelector('.context-limit-alert');
        if (alert) alert.style.display = show ? 'block' : 'none';
    }

    // Get current input value
    get value() {
        const input = this.querySelector('textarea');
        return input ? input.value : '';
    }

    // Set input value
    set value(val) {
        const input = this.querySelector('textarea');
        if (input) input.value = val;
    }

    // Clear the input
    clear() {
        const input = this.querySelector('textarea');
        if (input) input.value = '';
    }

    // Focus the input
    focus() {
        const input = this.querySelector('textarea');
        if (input) input.focus();
    }

    // Set placeholder
    setPlaceholder(text) {
        const input = this.querySelector('textarea');
        if (input) input.placeholder = text;
    }
}

customElements.define('ewr-chat-entry', EwrChatEntry);


/**
 * EWR-ConnectionSettings Component
 * A collapsible panel specifically for database connection settings
 * Contains header, content area for form controls, and footer for action buttons
 *
 * Usage:
 * <ewr-connection-settings
 *     id="connectionPanel"
 *     title="Connection Settings"
 *     collapsed>
 *     <!-- Content goes here (form controls) -->
 *     <ewr-button-bar slot="footer">
 *         <button class="ewr-button-action-green-gradient">Connect</button>
 *     </ewr-button-bar>
 * </ewr-connection-settings>
 */
class EwrConnectionSettings extends HTMLElement {
    constructor() {
        super();
    }

    connectedCallback() {
        const panelId = this.getAttribute('id') || 'connectionSettings';
        const title = this.getAttribute('title') || 'Connection Settings';
        const collapsed = this.hasAttribute('collapsed');
        const contentClass = this.getAttribute('content-class') || '';

        // Remove ID from custom element to avoid duplicate IDs
        this.removeAttribute('id');

        // Capture child elements before replacing innerHTML
        const children = Array.from(this.children);
        const footerChildren = children.filter(c => c.getAttribute('slot') === 'footer');
        const contentChildren = children.filter(c => c.getAttribute('slot') !== 'footer');

        // Create the container structure
        this.innerHTML = `
            <div class="ewr-collapsible ${collapsed ? 'collapsed' : ''}" id="${panelId}">
                <div class="ewr-collapsible-header" onclick="toggleCollapsible('${panelId}')">
                    <span class="ewr-collapsible-title">${title}</span>
                    <button class="ewr-collapse-toggle" type="button">
                        <span class="toggle-icon">▼</span>
                    </button>
                </div>
                <div class="ewr-collapsible-content ${contentClass}">
                    <div class="ewr-settings-content"></div>
                </div>
                <div class="ewr-collapsible-footer"></div>
            </div>
        `;

        // Move content children into settings content area
        const contentArea = this.querySelector('.ewr-settings-content');
        contentChildren.forEach(child => {
            child.removeAttribute('slot');
            contentArea.appendChild(child);
        });

        // Move footer children into footer area
        const footerArea = this.querySelector('.ewr-collapsible-footer');
        footerChildren.forEach(child => {
            child.removeAttribute('slot');
            footerArea.appendChild(child);
        });
    }

    // Toggle collapsed state
    toggle() {
        const panel = this.querySelector('.ewr-collapsible');
        if (panel) panel.classList.toggle('collapsed');
    }

    // Collapse the panel
    collapse() {
        const panel = this.querySelector('.ewr-collapsible');
        if (panel) panel.classList.add('collapsed');
    }

    // Expand the panel
    expand() {
        const panel = this.querySelector('.ewr-collapsible');
        if (panel) panel.classList.remove('collapsed');
    }

    // Check if collapsed
    get isCollapsed() {
        const panel = this.querySelector('.ewr-collapsible');
        return panel ? panel.classList.contains('collapsed') : false;
    }
}

customElements.define('ewr-connection-settings', EwrConnectionSettings);


/**
 * EWR-LinearInput Component
 * A horizontal label + input control with customizable label text
 * Similar to ewr-edit-div but as a reusable component
 *
 * Usage:
 * <ewr-linear-input
 *     label="Server"
 *     id="serverInput"
 *     type="text"
 *     placeholder="Enter server name"
 *     value="NCSQLTEST"
 *     tabindex="1"
 *     oninput="handleInput()">
 * </ewr-linear-input>
 */
class EwrLinearInput extends HTMLElement {
    constructor() {
        super();
    }

    connectedCallback() {
        const label = this.getAttribute('label') || 'Label';
        const id = this.getAttribute('id') || 'linearInput';
        const type = this.getAttribute('type') || 'text';
        const placeholder = this.getAttribute('placeholder') || '';
        const value = this.getAttribute('value') || '';
        const tabindex = this.getAttribute('tabindex') || '';
        const disabled = this.hasAttribute('disabled');
        const oninput = this.getAttribute('oninput') || '';
        const onchange = this.getAttribute('onchange') || '';
        const inputClass = this.getAttribute('input-class') || 'ewr-input-edits';

        this.innerHTML = `
            <div class="ewr-edit-div">
                <label class="ewr-label" for="${id}">${label}</label>
                <input
                    type="${type}"
                    id="${id}"
                    class="${inputClass}"
                    placeholder="${placeholder}"
                    value="${value}"
                    ${tabindex ? `tabindex="${tabindex}"` : ''}
                    ${disabled ? 'disabled' : ''}
                    ${oninput ? `oninput="${oninput}"` : ''}
                    ${onchange ? `onchange="${onchange}"` : ''}>
            </div>
        `;
    }

    // Get current value
    get value() {
        const input = this.querySelector('input');
        return input ? input.value : '';
    }

    // Set current value
    set value(val) {
        const input = this.querySelector('input');
        if (input) input.value = val;
    }

    // Get/set disabled state
    get disabled() {
        const input = this.querySelector('input');
        return input ? input.disabled : false;
    }

    set disabled(val) {
        const input = this.querySelector('input');
        if (input) input.disabled = val;
    }

    // Focus the input
    focus() {
        const input = this.querySelector('input');
        if (input) input.focus();
    }
}

customElements.define('ewr-linear-input', EwrLinearInput);


/**
 * EWR-LinearSelect Component
 * A horizontal label + select control with customizable label text
 *
 * Usage:
 * <ewr-linear-select
 *     label="Database"
 *     id="databaseSelect"
 *     tabindex="2"
 *     disabled>
 *     <option value="">Select database...</option>
 * </ewr-linear-select>
 */
class EwrLinearSelect extends HTMLElement {
    constructor() {
        super();
    }

    connectedCallback() {
        const label = this.getAttribute('label') || 'Label';
        const id = this.getAttribute('id') || 'linearSelect';
        const tabindex = this.getAttribute('tabindex') || '';
        const disabled = this.hasAttribute('disabled');
        const onchange = this.getAttribute('onchange') || '';
        const selectClass = this.getAttribute('select-class') || 'ewr-select-edit';

        // Capture existing options before replacing innerHTML
        const existingOptions = Array.from(this.querySelectorAll('option'));
        let optionsHtml = '';
        existingOptions.forEach(opt => {
            optionsHtml += opt.outerHTML;
        });

        this.innerHTML = `
            <div class="ewr-edit-div">
                <label class="ewr-label" for="${id}">${label}</label>
                <select
                    id="${id}"
                    class="${selectClass}"
                    ${tabindex ? `tabindex="${tabindex}"` : ''}
                    ${disabled ? 'disabled' : ''}
                    ${onchange ? `onchange="${onchange}"` : ''}>
                    ${optionsHtml}
                </select>
            </div>
        `;
    }

    // Get current value
    get value() {
        const select = this.querySelector('select');
        return select ? select.value : '';
    }

    // Set current value
    set value(val) {
        const select = this.querySelector('select');
        if (select) select.value = val;
    }

    // Get/set disabled state
    get disabled() {
        const select = this.querySelector('select');
        return select ? select.disabled : false;
    }

    set disabled(val) {
        const select = this.querySelector('select');
        if (select) select.disabled = val;
    }

    // Method to populate options programmatically
    setOptions(options, defaultText = '') {
        const select = this.querySelector('select');
        if (select) {
            let html = defaultText ? `<option value="">${defaultText}</option>` : '';
            options.forEach(opt => {
                const value = opt.value || opt.id || opt.name || opt;
                const text = opt.text || opt.name || opt.id || opt;
                html += `<option value="${value}">${text}</option>`;
            });
            select.innerHTML = html;
        }
    }
}

customElements.define('ewr-linear-select', EwrLinearSelect);


/**
 * EWR-ButtonBar Component
 * A horizontal button bar that can be used in footers or standalone
 * Allows buttons to be swapped out by placing them as children
 *
 * Usage:
 * <ewr-button-bar>
 *     <button class="ewr-button-action-green-gradient" onclick="connect()">Connect</button>
 *     <button class="ewr-button-action-green-gradient" onclick="load()">Load</button>
 *     <ewr-status-pill id="statusPill"></ewr-status-pill>
 * </ewr-button-bar>
 */
class EwrButtonBar extends HTMLElement {
    constructor() {
        super();
    }

    connectedCallback() {
        const align = this.getAttribute('align') || 'left'; // left, center, right
        const gap = this.getAttribute('gap') || '8px';

        // Capture child elements before replacing innerHTML
        const children = Array.from(this.children);

        // Create the container structure
        const justifyContent = align === 'center' ? 'center' : align === 'right' ? 'flex-end' : 'flex-start';

        this.innerHTML = `
            <div class="ewr-button-bar" style="display: flex; flex-direction: row; gap: ${gap}; align-items: center; justify-content: ${justifyContent};"></div>
        `;

        // Move captured children into the button bar
        const buttonBar = this.querySelector('.ewr-button-bar');
        children.forEach(child => {
            buttonBar.appendChild(child);
        });
    }

    // Get a button by ID
    getButton(id) {
        return this.querySelector(`#${id}`);
    }

    // Disable all buttons
    disableAll() {
        this.querySelectorAll('button').forEach(btn => btn.disabled = true);
    }

    // Enable all buttons
    enableAll() {
        this.querySelectorAll('button').forEach(btn => btn.disabled = false);
    }
}

customElements.define('ewr-button-bar', EwrButtonBar);


/**
 * EWR-StatusPill Component
 * A small status indicator pill for displaying connection status or messages
 *
 * Usage:
 * <ewr-status-pill id="statusPill" type="success">Connected</ewr-status-pill>
 */
class EwrStatusPill extends HTMLElement {
    constructor() {
        super();
    }

    connectedCallback() {
        const id = this.getAttribute('id') || 'statusPill';
        const type = this.getAttribute('type') || ''; // success, error, info, warning
        const text = this.textContent || '';

        this.innerHTML = `
            <span id="${id}" class="ewr-status-pill ${type}">${text}</span>
        `;
    }

    // Set status type and text
    setStatus(type, text) {
        const pill = this.querySelector('.ewr-status-pill');
        if (pill) {
            pill.className = `ewr-status-pill ${type}`;
            pill.textContent = text;
        }
    }

    // Clear status
    clear() {
        const pill = this.querySelector('.ewr-status-pill');
        if (pill) {
            pill.className = 'ewr-status-pill';
            pill.textContent = '';
        }
    }

    // Show the pill
    show() {
        const pill = this.querySelector('.ewr-status-pill');
        if (pill) pill.style.display = 'inline-block';
    }

    // Hide the pill
    hide() {
        const pill = this.querySelector('.ewr-status-pill');
        if (pill) pill.style.display = 'none';
    }
}

customElements.define('ewr-status-pill', EwrStatusPill);


/**
 * EWR-SystemStatus Component
 * A header status indicator that shows system health from Python service
 * Auto-updates via system-status.js events
 *
 * Usage:
 * <ewr-system-status></ewr-system-status>
 *
 * Attributes:
 * - compact: Show only the dot without text (optional)
 * - show-details: Show additional service details on hover (optional)
 */
class EwrSystemStatus extends HTMLElement {
    constructor() {
        super();
        this._status = null;
        this._statusKey = 'connecting';
    }

    connectedCallback() {
        const compact = this.hasAttribute('compact');
        const id = this.getAttribute('id') || 'systemStatus';

        this.innerHTML = `
            <div class="header-status" id="${id}">
                <div class="status-dot" id="statusDot"></div>
                ${compact ? '' : '<span class="status-text" id="statusText">Connecting...</span>'}
            </div>
        `;

        // Listen for status updates from system-status.js
        window.addEventListener('systemStatusChanged', (e) => {
            this.updateStatus(e.detail.status, e.detail.statusKey);
        });
    }

    updateStatus(status, statusKey) {
        this._status = status;
        this._statusKey = statusKey;

        const dot = this.querySelector('.status-dot');
        const text = this.querySelector('.status-text');

        if (!dot) return;

        // Remove existing status classes
        dot.classList.remove('status-healthy', 'status-degraded', 'status-offline', 'status-error');

        const colors = {
            green: 'var(--accent-green, #10b981)',
            yellow: 'var(--accent-yellow, #f59e0b)',
            red: 'var(--accent-red, #ef4444)'
        };

        switch (statusKey) {
            case 'healthy':
                dot.classList.add('status-healthy');
                dot.style.background = colors.green;
                if (text) {
                    text.textContent = 'System Online';
                    text.style.color = colors.green;
                }
                this.title = 'All systems operational';
                break;
            case 'degraded-llm':
                dot.classList.add('status-degraded');
                dot.style.background = colors.yellow;
                if (text) {
                    text.textContent = 'LLM Unavailable';
                    text.style.color = colors.yellow;
                }
                this.title = 'LLM service unavailable';
                break;
            case 'degraded':
                dot.classList.add('status-degraded');
                dot.style.background = colors.yellow;
                if (text) {
                    text.textContent = 'Degraded';
                    text.style.color = colors.yellow;
                }
                this.title = 'Some services degraded';
                break;
            case 'error':
                dot.classList.add('status-error');
                dot.style.background = colors.red;
                if (text) {
                    text.textContent = status?.error || 'Error';
                    text.style.color = colors.red;
                }
                this.title = status?.error || 'Service error';
                break;
            default:
                dot.classList.add('status-offline');
                dot.style.background = colors.red;
                if (text) {
                    text.textContent = 'Offline';
                    text.style.color = colors.red;
                }
                this.title = 'Services offline';
        }
    }

    // Get current status
    get status() {
        return this._status;
    }

    // Get status key (healthy, degraded, offline, error)
    get statusKey() {
        return this._statusKey;
    }

    // Check if system is healthy
    get isHealthy() {
        return this._statusKey === 'healthy';
    }
}

customElements.define('ewr-system-status', EwrSystemStatus);


/**
 * EWR-LogoutButton Component
 * A styled logout button that handles auth logout
 *
 * Usage:
 * <ewr-logout-button></ewr-logout-button>
 * <ewr-logout-button text="Sign Out"></ewr-logout-button>
 * <ewr-logout-button redirect="/login.html"></ewr-logout-button>
 *
 * Attributes:
 * - text: Button text (default: "Logout")
 * - redirect: URL to redirect after logout (default: auto-detected login page)
 */
class EwrLogoutButton extends HTMLElement {
    constructor() {
        super();
    }

    connectedCallback() {
        const text = this.getAttribute('text') || 'Logout';
        const redirect = this.getAttribute('redirect') || '';

        this.innerHTML = `
            <button class="ewr-logout-button" type="button">${text}</button>
        `;

        const button = this.querySelector('button');
        button.addEventListener('click', () => this.handleLogout(redirect));
    }

    async handleLogout(redirect) {
        try {
            // Use global auth if available
            if (typeof auth !== 'undefined' && auth.logout) {
                await auth.logout();
            } else {
                // Fallback: clear tokens manually
                localStorage.removeItem('accessToken');
                localStorage.removeItem('refreshToken');
                sessionStorage.clear();

                // Try to call logout endpoint
                try {
                    await fetch('/api/auth/logout', {
                        method: 'POST',
                        credentials: 'include'
                    });
                } catch (e) {
                    // Ignore errors
                }
            }

            // Redirect
            if (redirect) {
                window.location.href = redirect;
            } else {
                // Auto-detect login page
                window.location.href = '/login.html';
            }
        } catch (error) {
            console.error('Logout failed:', error);
            // Force redirect anyway
            window.location.href = redirect || '/login.html';
        }
    }
}

customElements.define('ewr-logout-button', EwrLogoutButton);


/**
 * EWR-SearchButton Component
 * A styled search button with silver gradient
 *
 * Usage:
 * <ewr-search-button onclick="performSearch()"></ewr-search-button>
 * <ewr-search-button text="Find" onclick="doFind()"></ewr-search-button>
 * <ewr-search-button icon="true" onclick="search()"></ewr-search-button>
 *
 * Attributes:
 * - text: Button text (default: "Search")
 * - icon: Show search icon (default: false)
 * - disabled: Disable the button
 */
class EwrSearchButton extends HTMLElement {
    constructor() {
        super();
    }

    connectedCallback() {
        const text = this.getAttribute('text') || 'Search';
        const showIcon = this.hasAttribute('icon');
        const disabled = this.hasAttribute('disabled');
        const id = this.getAttribute('id') || '';

        const iconHtml = showIcon ? '<ewr-icon name="search" size="16"></ewr-icon>' : '';

        this.innerHTML = `
            <button class="ewr-search-button" type="button" ${id ? `id="${id}"` : ''} ${disabled ? 'disabled' : ''}>
                ${iconHtml}${showIcon && text ? ' ' : ''}${text}
            </button>
        `;

        // Forward click events
        const button = this.querySelector('button');
        button.addEventListener('click', (e) => {
            if (!disabled) {
                this.dispatchEvent(new CustomEvent('search', { bubbles: true }));
            }
        });
    }

    // Enable/disable button
    set disabled(value) {
        const button = this.querySelector('button');
        if (button) {
            button.disabled = value;
            if (value) {
                this.setAttribute('disabled', '');
            } else {
                this.removeAttribute('disabled');
            }
        }
    }

    get disabled() {
        return this.hasAttribute('disabled');
    }

    // Set loading state
    setLoading(loading) {
        const button = this.querySelector('button');
        if (button) {
            button.disabled = loading;
            if (loading) {
                button.dataset.originalText = button.textContent;
                button.innerHTML = '<ewr-icon name="loader-2" size="16" class="spin"></ewr-icon> Searching...';
            } else if (button.dataset.originalText) {
                button.textContent = button.dataset.originalText;
            }
        }
    }
}

customElements.define('ewr-search-button', EwrSearchButton);


/**
 * EWR-ChatLlmMetrics Component
 * A horizontal status bar for displaying LLM context window metrics and token usage
 * Child elements define what metrics to display
 *
 * Usage:
 * <ewr-chat-llm-metrics>
 *     <ewr-status-metric label="Context:" value-id="contextUsed"></ewr-status-metric>
 *     <ewr-status-metric separator="/"></ewr-status-metric>
 *     <ewr-status-metric value-id="contextMax"></ewr-status-metric>
 *     <ewr-status-metric label="Prompt:" value-id="promptTokens" margin-left="16px"></ewr-status-metric>
 *     <ewr-status-metric label="Completion:" value-id="completionTokens" margin-left="16px"></ewr-status-metric>
 *     <ewr-status-metric label="Total:" value-id="totalTokens" margin-left="16px"></ewr-status-metric>
 * </ewr-chat-llm-metrics>
 */
class EwrChatLlmMetrics extends HTMLElement {
    constructor() {
        super();
    }

    connectedCallback() {
        const id = this.getAttribute('id') || 'chatLlmMetrics';

        // Capture child elements before replacing innerHTML
        const children = Array.from(this.children);

        // Create the container structure
        this.innerHTML = `
            <div class="ewr-chat-llm-metrics" id="${id}"></div>
        `;

        // Move captured children into the metrics bar
        const metricsBar = this.querySelector('.ewr-chat-llm-metrics');
        children.forEach(child => {
            metricsBar.appendChild(child);
        });
    }

    // Update a specific metric value by ID
    updateMetric(valueId, value) {
        const el = this.querySelector(`#${valueId}`);
        if (el) el.textContent = value;
    }

    // Set warning state (for context window nearly full)
    setWarning(isWarning) {
        const values = this.querySelectorAll('.ewr-llm-metric-value');
        values.forEach(el => {
            el.style.color = isWarning ? '#f59e0b' : '#10b981';
        });
    }
}

customElements.define('ewr-chat-llm-metrics', EwrChatLlmMetrics);


/**
 * EWR-StatusMetric Component
 * A single metric display with optional label and value
 * Used inside ewr-chat-llm-metrics
 *
 * Usage:
 * <ewr-status-metric label="Context:" value-id="contextUsed" value="0"></ewr-status-metric>
 * <ewr-status-metric separator="/"></ewr-status-metric>
 */
class EwrStatusMetric extends HTMLElement {
    constructor() {
        super();
    }

    connectedCallback() {
        const label = this.getAttribute('label') || '';
        const valueId = this.getAttribute('value-id') || 'metricValue';
        const value = this.getAttribute('value') || '0';
        const separator = this.getAttribute('separator') || '';
        const marginLeft = this.getAttribute('margin-left') || '';

        let html = '';

        if (separator) {
            // Just a separator (like "/")
            html = `<span class="ewr-llm-metric-label" ${marginLeft ? `style="margin-left: ${marginLeft}"` : ''}>${separator}</span>`;
        } else {
            // Label + value pair
            if (label) {
                html += `<span class="ewr-llm-metric-label" ${marginLeft ? `style="margin-left: ${marginLeft}"` : ''}>${label}</span>`;
            }
            html += `<span class="ewr-llm-metric-value" id="${valueId}">${value}</span>`;
        }

        this.innerHTML = html;
    }

    // Get current value
    get value() {
        const el = this.querySelector('.ewr-llm-metric-value');
        return el ? el.textContent : '';
    }

    // Set current value
    set value(val) {
        const el = this.querySelector('.ewr-llm-metric-value');
        if (el) el.textContent = val;
    }

    // Set warning color
    setWarning(isWarning) {
        const el = this.querySelector('.ewr-llm-metric-value');
        if (el) el.style.color = isWarning ? '#f59e0b' : '#10b981';
    }
}

customElements.define('ewr-status-metric', EwrStatusMetric);


/**
 * EWR-ContentDiv Component
 * A standard card-like content container for non-chat master content areas
 * No header/title - just a clean body for holding child elements
 *
 * Usage:
 * <ewr-content-div id="documentBrowser" padding="24px">
 *     <!-- Your content goes here -->
 * </ewr-content-div>
 */
class EwrContentDiv extends HTMLElement {
    constructor() {
        super();
    }

    connectedCallback() {
        const id = this.getAttribute('id') || 'ewrContent';
        const padding = this.getAttribute('padding') || '24px';

        // Capture child elements before replacing innerHTML
        const children = Array.from(this.children);

        // Create the container structure
        this.innerHTML = `
            <div class="card ewr-content-card" id="${id}">
                <div class="ewr-content-body" style="padding: ${padding};"></div>
            </div>
        `;

        // Move captured children into the content body
        const contentBody = this.querySelector('.ewr-content-body');
        children.forEach(child => {
            contentBody.appendChild(child);
        });
    }

    // Get the content body element for direct manipulation
    get contentBody() {
        return this.querySelector('.ewr-content-body');
    }

    // Clear all content
    clear() {
        const body = this.querySelector('.ewr-content-body');
        if (body) body.innerHTML = '';
    }

    // Set padding dynamically
    setPadding(padding) {
        const body = this.querySelector('.ewr-content-body');
        if (body) body.style.padding = padding;
    }
}

customElements.define('ewr-content-div', EwrContentDiv);


/**
 * EWR-Button Component
 * A customizable button with metallic gradient styling and multiple color themes
 * Based on the ewr-button-action-green-gradient style with extended color and size options
 *
 * Usage:
 * <ewr-button
 *     text="Click Me"
 *     color="primary"
 *     size="medium"
 *     id="myButton"
 *     onclick="handleClick()"
 *     disabled>
 * </ewr-button>
 *
 * Properties:
 * - text: Button text content (default: "Button")
 * - color: primary (green), secondary (gray), danger (red), success (green), warning (orange), info (blue)
 * - size: small, medium (default), large
 * - id: Button element ID
 * - onclick: Click handler function
 * - disabled: Disabled state (boolean attribute)
 */
class EwrButton extends HTMLElement {
    constructor() {
        super();
    }

    connectedCallback() {
        const text = this.getAttribute('text') || 'Button';
        const color = this.getAttribute('color') || 'primary';
        const size = this.getAttribute('size') || 'medium';
        const id = this.getAttribute('id') || '';
        const onclick = this.getAttribute('onclick') || '';
        const disabled = this.hasAttribute('disabled');

        // Build button classes
        const colorClass = `ewr-button-${color}`;
        const sizeClass = `ewr-button-${size}`;

        this.innerHTML = `
            <button
                ${id ? `id="${id}"` : ''}
                class="ewr-button ${colorClass} ${sizeClass}"
                ${onclick ? `onclick="${onclick}"` : ''}
                ${disabled ? 'disabled' : ''}>
                ${text}
            </button>
        `;
    }

    // Get/set disabled state
    get disabled() {
        const button = this.querySelector('button');
        return button ? button.disabled : false;
    }

    set disabled(val) {
        const button = this.querySelector('button');
        if (button) button.disabled = val;
    }

    // Get/set text content
    get text() {
        const button = this.querySelector('button');
        return button ? button.textContent : '';
    }

    set text(val) {
        const button = this.querySelector('button');
        if (button) button.textContent = val;
    }

    // Focus the button
    focus() {
        const button = this.querySelector('button');
        if (button) button.focus();
    }

    // Trigger click programmatically
    click() {
        const button = this.querySelector('button');
        if (button) button.click();
    }
}

customElements.define('ewr-button', EwrButton);


/**
 * EWR-MatchingDocuments Component
 * A container for displaying a list of matching documents in a 3-column grid
 *
 * Usage:
 * <ewr-matching-documents
 *     id="docListBox"
 *     title="Matching Documents"
 *     count-id="docCount"
 *     content-id="docContent"
 *     empty-message="No documents found"
 *     on-item-click="handleDocClick">
 * </ewr-matching-documents>
 */
class EwrMatchingDocuments extends HTMLElement {
    constructor() {
        super();
    }

    connectedCallback() {
        const id = this.getAttribute('id') || 'matchingDocuments';
        const title = this.getAttribute('title') || 'Matching Documents';
        const countId = this.getAttribute('count-id') || 'documentCount';
        const contentId = this.getAttribute('content-id') || 'documentContent';
        const emptyMessage = this.getAttribute('empty-message') || 'Select filters and click "Load Documents" to see matching documents';

        this.innerHTML = `
            <div class="ewr-matching-documents" id="${id}">
                <div class="ewr-matching-documents-header">
                    <span class="ewr-matching-documents-title">${title}</span>
                    <span class="ewr-matching-documents-count" id="${countId}">0 documents</span>
                </div>
                <div class="ewr-matching-documents-content empty" id="${contentId}">
                    ${emptyMessage}
                </div>
            </div>
        `;
    }

    // Method to populate documents (accepts array of titles)
    setDocuments(titles, onItemClick = null) {
        const contentEl = this.querySelector('.ewr-matching-documents-content');
        const countEl = this.querySelector('.ewr-matching-documents-count');
        const onClickAttr = this.getAttribute('on-item-click');

        if (!contentEl || !countEl) return;

        const count = titles.length;
        countEl.textContent = `${count} document${count !== 1 ? 's' : ''}`;

        if (count > 0) {
            const gridHtml = `
                <div class="ewr-matching-documents-grid">
                    ${titles.map(title => {
                        const escapedTitle = title.replace(/'/g, "\\'");
                        const clickHandler = onClickAttr ? `${onClickAttr}('${escapedTitle}')` :
                                           onItemClick ? `(${onItemClick})('${escapedTitle}')` : '';
                        return `
                            <div class="ewr-matching-documents-item"
                                 title="${title}"
                                 ${clickHandler ? `onclick="${clickHandler}"` : ''}>
                                ${title}
                            </div>
                        `;
                    }).join('')}
                </div>
            `;
            contentEl.innerHTML = gridHtml;
            contentEl.classList.remove('empty');
        } else {
            contentEl.innerHTML = this.getAttribute('empty-message') || 'No documents found';
            contentEl.classList.add('empty');
        }
    }

    // Method to show loading state
    showLoading() {
        const contentEl = this.querySelector('.ewr-matching-documents-content');
        if (contentEl) {
            contentEl.innerHTML = 'Loading...';
            contentEl.classList.add('empty');
        }
    }

    // Method to show error state
    showError(message) {
        const contentEl = this.querySelector('.ewr-matching-documents-content');
        if (contentEl) {
            contentEl.innerHTML = `Error: ${message}`;
            contentEl.classList.add('empty');
        }
    }

    // Get content element
    getContentElement() {
        return this.querySelector('.ewr-matching-documents-content');
    }

    // Get count element
    getCountElement() {
        return this.querySelector('.ewr-matching-documents-count');
    }
}

customElements.define('ewr-matching-documents', EwrMatchingDocuments);


/**
 * EWR-DocumentItem Component
 * A single clickable document item for use in lists
 *
 * Usage:
 * <ewr-document-item title="My Document.txt" onclick="handleClick('My Document.txt')"></ewr-document-item>
 */
class EwrDocumentItem extends HTMLElement {
    constructor() {
        super();
    }

    connectedCallback() {
        const title = this.getAttribute('title') || 'Untitled';
        const onclick = this.getAttribute('onclick') || '';

        this.innerHTML = `
            <div class="ewr-matching-documents-item"
                 title="${title}"
                 ${onclick ? `onclick="${onclick}"` : ''}>
                ${title}
            </div>
        `;
    }
}

customElements.define('ewr-document-item', EwrDocumentItem);


/**
 * Helper function for collapsible toggle (must be in global scope)
 * This is called from the onclick handler in the collapsible header
 */
if (typeof window.toggleCollapsible === 'undefined') {
    window.toggleCollapsible = function(panelId) {
        const panel = document.getElementById(panelId);
        if (panel) {
            panel.classList.toggle('collapsed');
        }
    };
}


/**
 * EWR-AI-Assistant Component
 * A reusable AI assistant panel that can be embedded in any form/modal
 * to help users generate content using AI.
 *
 * Usage:
 * <ewr-ai-assistant
 *     title="AI Rule Assistant"
 *     placeholder="Describe your problem..."
 *     button-text="Generate with AI"
 *     api-endpoint="/api/sql/generate-rule"
 *     on-result="handleAIResult">
 * </ewr-ai-assistant>
 *
 * The on-result callback receives the parsed JSON response from the API.
 * You must define a function in global scope that handles the result.
 *
 * Example callback:
 * function handleAIResult(data) {
 *     document.getElementById('field1').value = data.field1;
 *     document.getElementById('field2').value = data.field2;
 * }
 */
class EwrAIAssistant extends HTMLElement {
    constructor() {
        super();
        this._isGenerating = false;
    }

    connectedCallback() {
        // Get attributes with defaults
        const title = this.getAttribute('title') || 'AI Assistant';
        const placeholder = this.getAttribute('placeholder') || 'Describe what you need help with...';
        const buttonText = this.getAttribute('button-text') || 'Generate with AI';
        const apiEndpoint = this.getAttribute('api-endpoint') || '/api/llm/complete';
        const onResult = this.getAttribute('on-result') || null;
        const contextParam = this.getAttribute('context-param') || null; // e.g., "database" to pull from another element
        const promptTemplate = this.getAttribute('prompt-template') || null;

        const uniqueId = `ai-assist-${Date.now()}`;

        this.innerHTML = `
            <div class="ewr-ai-assistant" style="margin-bottom: 20px; padding: 16px; background: linear-gradient(135deg, rgba(139, 92, 246, 0.1) 0%, rgba(59, 130, 246, 0.1) 100%); border-radius: 8px; border: 1px solid rgba(139, 92, 246, 0.3);">
                <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 12px;">
                    <span style="font-size: 18px;">🤖</span>
                    <span style="color: #a78bfa; font-weight: 600; font-size: 14px;">${title}</span>
                </div>
                <div style="margin-bottom: 12px;">
                    <textarea id="${uniqueId}-input" rows="2"
                        placeholder="${placeholder}"
                        style="width: 100%; padding: 10px; background: #0f172a; border: 1px solid rgba(139, 92, 246, 0.3); border-radius: 6px; color: #f1f5f9; font-size: 13px; resize: vertical; box-sizing: border-box;"></textarea>
                </div>
                <div id="${uniqueId}-message" style="margin-bottom: 8px;"></div>
                <button id="${uniqueId}-btn" class="btn" style="background: linear-gradient(135deg, #8b5cf6 0%, #3b82f6 100%); width: 100%; padding: 10px; border: none; border-radius: 6px; color: white; cursor: pointer; font-weight: 600;">
                    <span id="${uniqueId}-btn-text">✨ ${buttonText}</span>
                </button>
            </div>
        `;

        // Store references
        this._inputId = `${uniqueId}-input`;
        this._btnId = `${uniqueId}-btn`;
        this._btnTextId = `${uniqueId}-btn-text`;
        this._messageId = `${uniqueId}-message`;
        this._apiEndpoint = apiEndpoint;
        this._onResult = onResult;
        this._contextParam = contextParam;
        this._promptTemplate = promptTemplate;
        this._buttonText = buttonText;

        // Attach click handler
        const btn = document.getElementById(this._btnId);
        btn.addEventListener('click', () => this._generate());
    }

    _showMessage(message, isError = false) {
        const msgDiv = document.getElementById(this._messageId);
        if (msgDiv) {
            msgDiv.innerHTML = `<div style="padding: 8px; border-radius: 6px; background: ${isError ? 'rgba(239, 68, 68, 0.15)' : 'rgba(16, 185, 129, 0.15)'}; color: ${isError ? '#fca5a5' : '#6ee7b7'}; border: 1px solid ${isError ? '#ef4444' : '#10b981'}; font-size: 12px;">${message}</div>`;
        }
    }

    _clearMessage() {
        const msgDiv = document.getElementById(this._messageId);
        if (msgDiv) msgDiv.innerHTML = '';
    }

    async _generate() {
        if (this._isGenerating) return;

        const input = document.getElementById(this._inputId);
        const btn = document.getElementById(this._btnId);
        const btnText = document.getElementById(this._btnTextId);
        const userText = input?.value?.trim();

        if (!userText) {
            this._showMessage('Please describe your request first', true);
            return;
        }

        // Get context from another element if specified
        let context = {};
        if (this._contextParam) {
            const contextEl = document.getElementById(this._contextParam);
            if (contextEl) {
                context[this._contextParam] = contextEl.value;
            }
        }

        this._isGenerating = true;
        btn.disabled = true;
        btnText.textContent = '⏳ Generating...';
        this._clearMessage();

        try {
            // Build request body
            let requestBody = {
                prompt: userText,
                ...context
            };

            // If a prompt template is provided, use it
            if (this._promptTemplate) {
                const templateFn = window[this._promptTemplate];
                if (typeof templateFn === 'function') {
                    requestBody.prompt = templateFn(userText, context);
                }
            }

            const response = await fetch(this._apiEndpoint, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(requestBody)
            });

            if (!response.ok) {
                throw new Error(`API returned ${response.status}`);
            }

            const data = await response.json();

            // Call the result handler if specified
            if (this._onResult && window[this._onResult]) {
                window[this._onResult](data);
            }

            this._showMessage('✨ Generated! Review the fields below.', false);

        } catch (error) {
            console.error('AI Assistant error:', error);
            this._showMessage(`Generation failed: ${error.message}`, true);
        } finally {
            this._isGenerating = false;
            btn.disabled = false;
            btnText.textContent = `✨ ${this._buttonText}`;
        }
    }

    // Public method to clear the input
    clear() {
        const input = document.getElementById(this._inputId);
        if (input) input.value = '';
        this._clearMessage();
    }
}

customElements.define('ewr-ai-assistant', EwrAIAssistant);


/**
 * EWR-Modal Component
 * A reusable modal dialog with consistent styling and behavior
 *
 * Usage:
 * <ewr-modal id="myModal"
 *     title="Modal Title"
 *     icon="📋"
 *     accent-color="#a78bfa"
 *     max-width="700px"
 *     on-close="handleClose">
 *
 *     <!-- Modal content goes here -->
 *     <p>Your content...</p>
 *
 *     <!-- Optional: footer slot for action buttons -->
 *     <div slot="footer">
 *         <button onclick="closeMyModal()">Cancel</button>
 *         <button onclick="saveData()">Save</button>
 *     </div>
 * </ewr-modal>
 *
 * JavaScript API:
 * - document.getElementById('myModal').open()
 * - document.getElementById('myModal').close()
 * - document.getElementById('myModal').isOpen
 *
 * Attributes:
 * - title: Modal header title
 * - icon: Optional emoji/icon before title
 * - accent-color: Color for header background tint and title (default: #3b82f6)
 * - max-width: Maximum width of modal dialog (default: 600px)
 * - max-height: Maximum height of modal (default: 90vh)
 * - on-close: Name of function to call when modal closes
 * - closeable: If "false", clicking overlay won't close modal
 */
class EwrModal extends HTMLElement {
    constructor() {
        super();
        this._isOpen = false;
    }

    connectedCallback() {
        const title = this.getAttribute('title') || 'Modal';
        const icon = this.getAttribute('icon') || '';
        const accentColor = this.getAttribute('accent-color') || '#3b82f6';
        const maxWidth = this.getAttribute('max-width') || '600px';
        const maxHeight = this.getAttribute('max-height') || '90vh';
        const closeable = this.getAttribute('closeable') !== 'false';
        const onClose = this.getAttribute('on-close');

        // Store for later use
        this._onClose = onClose;
        this._closeable = closeable;

        // Capture body content and footer slot before replacing
        const footerSlot = this.querySelector('[slot="footer"]');
        const bodyContent = Array.from(this.childNodes).filter(node => {
            if (node.nodeType === Node.TEXT_NODE) return node.textContent.trim();
            if (node.nodeType === Node.ELEMENT_NODE) return !node.hasAttribute('slot');
            return false;
        });

        // Calculate header background with accent color tint
        const headerBg = `rgba(${this._hexToRgb(accentColor)}, 0.1)`;
        const headerBorder = `rgba(${this._hexToRgb(accentColor)}, 0.3)`;

        // Build the modal structure
        this.innerHTML = `
            <div class="ewr-modal-overlay hidden" id="${this.id}-overlay">
                <div class="ewr-modal-dialog" style="max-width: ${maxWidth}; max-height: ${maxHeight};">
                    <div class="ewr-modal-header" style="background: ${headerBg}; border-bottom: 1px solid ${headerBorder};">
                        <h2 class="ewr-modal-title" style="color: ${accentColor};">${icon ? icon + ' ' : ''}${title}</h2>
                        <button class="ewr-modal-close" id="${this.id}-close-btn">×</button>
                    </div>
                    <div class="ewr-modal-body" id="${this.id}-body"></div>
                    <div class="ewr-modal-footer" id="${this.id}-footer" style="display: none;"></div>
                </div>
            </div>
        `;

        // Move body content
        const bodyContainer = this.querySelector(`#${this.id}-body`);
        bodyContent.forEach(node => {
            bodyContainer.appendChild(node.cloneNode ? node.cloneNode(true) : document.createTextNode(node.textContent));
        });

        // Move footer content if provided
        if (footerSlot) {
            const footerContainer = this.querySelector(`#${this.id}-footer`);
            footerContainer.style.display = 'flex';
            Array.from(footerSlot.children).forEach(child => {
                footerContainer.appendChild(child.cloneNode(true));
            });
        }

        // Setup event listeners
        this._setupListeners();
    }

    _hexToRgb(hex) {
        // Remove # if present
        hex = hex.replace('#', '');
        // Parse
        const r = parseInt(hex.substring(0, 2), 16);
        const g = parseInt(hex.substring(2, 4), 16);
        const b = parseInt(hex.substring(4, 6), 16);
        return `${r}, ${g}, ${b}`;
    }

    _setupListeners() {
        const overlay = this.querySelector('.ewr-modal-overlay');
        const closeBtn = this.querySelector(`#${this.id}-close-btn`);

        // Close on X button
        closeBtn.addEventListener('click', () => this.close());

        // Close on overlay click (if closeable)
        if (this._closeable) {
            overlay.addEventListener('click', (e) => {
                if (e.target === overlay) {
                    this.close();
                }
            });
        }

        // Close on Escape key
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape' && this._isOpen && this._closeable) {
                this.close();
            }
        });
    }

    // Public API
    open() {
        const overlay = this.querySelector('.ewr-modal-overlay');
        if (overlay) {
            overlay.classList.remove('hidden');
            this._isOpen = true;
            // Dispatch custom event
            this.dispatchEvent(new CustomEvent('modal-open', { bubbles: true }));
        }
    }

    close() {
        const overlay = this.querySelector('.ewr-modal-overlay');
        if (overlay) {
            overlay.classList.add('hidden');
            this._isOpen = false;
            // Call onClose handler if specified
            if (this._onClose && window[this._onClose]) {
                window[this._onClose]();
            }
            // Dispatch custom event
            this.dispatchEvent(new CustomEvent('modal-close', { bubbles: true }));
        }
    }

    get isOpen() {
        return this._isOpen;
    }

    // Allow updating body content dynamically
    setBody(html) {
        const body = this.querySelector(`#${this.id}-body`);
        if (body) body.innerHTML = html;
    }

    // Allow updating title dynamically
    setTitle(newTitle) {
        const titleEl = this.querySelector('.ewr-modal-title');
        if (titleEl) titleEl.textContent = newTitle;
    }

    /**
     * Static method to programmatically display a modal
     * @param {Object} options - Modal configuration
     * @param {string} options.title - Modal title
     * @param {string} [options.size='medium'] - Modal size: 'small', 'medium', 'large'
     * @param {Array} [options.sections] - Array of content sections
     * @param {Array} [options.buttons] - Array of button configs
     */
    static display(options) {
        const { title = 'Modal', size = 'medium', sections = [], buttons = [] } = options;

        // Size mapping
        const sizeMap = {
            small: '400px',
            medium: '600px',
            large: '800px'
        };
        const maxWidth = sizeMap[size] || sizeMap.medium;

        // Generate unique ID
        const modalId = `ewr-modal-${Date.now()}`;

        // Build sections HTML
        let sectionsHtml = sections.map(section => {
            const labelHtml = section.label ? `<div style="font-size: 12px; color: var(--text-muted); margin-bottom: 6px; font-weight: 500;">${section.label}</div>` : '';
            const contentHtml = section.isCode
                ? `<pre style="background: var(--bg-tertiary); padding: 12px; border-radius: 6px; overflow-x: auto; font-family: 'Consolas', monospace; font-size: 13px; margin: 0; white-space: pre-wrap; word-break: break-word;">${section.content}</pre>`
                : `<div style="color: var(--text-secondary);">${section.content}</div>`;
            return `<div style="margin-bottom: 16px;" ${section.id ? `id="${section.id}"` : ''}>${labelHtml}${contentHtml}</div>`;
        }).join('');

        // Build buttons HTML
        let buttonsHtml = buttons.map(btn => {
            const btnClass = btn.class === 'primary' ? 'ewr-button-action-green-gradient' : 'ewr-button-default';
            const closeAttr = btn.close ? `onclick="document.getElementById('${modalId}').close(); document.getElementById('${modalId}').remove();"` : '';
            const clickHandler = btn.onClick ? `data-has-click="true"` : '';
            return `<button class="${btnClass}" ${closeAttr} ${clickHandler}>${btn.text}</button>`;
        }).join('');

        // Create the modal element
        const modal = document.createElement('ewr-modal');
        modal.id = modalId;
        modal.setAttribute('title', title);
        modal.setAttribute('max-width', maxWidth);

        // Set body content
        modal.innerHTML = `
            ${sectionsHtml}
            <div slot="footer">${buttonsHtml}</div>
        `;

        // Add to document
        document.body.appendChild(modal);

        // Wait for connectedCallback then open
        requestAnimationFrame(() => {
            // Attach click handlers for buttons with onClick
            buttons.forEach((btn, index) => {
                if (btn.onClick) {
                    const buttonEls = modal.querySelectorAll('.ewr-modal-footer button');
                    if (buttonEls[index]) {
                        buttonEls[index].addEventListener('click', btn.onClick);
                    }
                }
            });
            modal.open();
        });

        return modal;
    }
}

customElements.define('ewr-modal', EwrModal);
// Expose globally for programmatic access
window.EwrModal = EwrModal;


/**
 * EWR-FormField Component
 * A labeled form field with consistent styling
 *
 * Usage:
 * <ewr-form-field
 *     label="Username"
 *     type="text"
 *     id="username"
 *     placeholder="Enter username"
 *     required>
 * </ewr-form-field>
 *
 * <ewr-form-field
 *     label="Description"
 *     type="textarea"
 *     id="desc"
 *     rows="3">
 * </ewr-form-field>
 *
 * <ewr-form-field
 *     label="Category"
 *     type="select"
 *     id="category">
 *     <option value="a">Option A</option>
 *     <option value="b">Option B</option>
 * </ewr-form-field>
 */
class EwrFormField extends HTMLElement {
    constructor() {
        super();
    }

    connectedCallback() {
        const label = this.getAttribute('label') || '';
        const type = this.getAttribute('type') || 'text';
        const id = this.getAttribute('id');
        const placeholder = this.getAttribute('placeholder') || '';
        const required = this.hasAttribute('required');
        const disabled = this.hasAttribute('disabled');
        const value = this.getAttribute('value') || '';
        const rows = this.getAttribute('rows') || '3';
        const hint = this.getAttribute('hint') || '';
        const onchange = this.getAttribute('onchange') || '';

        // Capture options for select
        const options = Array.from(this.querySelectorAll('option'));

        let inputHtml = '';
        const requiredMark = required ? ' *' : '';
        const disabledAttr = disabled ? 'disabled' : '';
        const onchangeAttr = onchange ? `onchange="${onchange}"` : '';

        if (type === 'textarea') {
            inputHtml = `
                <textarea
                    id="${id}"
                    class="ewr-form-textarea"
                    placeholder="${placeholder}"
                    rows="${rows}"
                    ${disabledAttr}
                    ${onchangeAttr}
                >${value}</textarea>
            `;
        } else if (type === 'select') {
            const optionsHtml = options.map(opt => opt.outerHTML).join('');
            inputHtml = `
                <select id="${id}" class="ewr-form-select" ${disabledAttr} ${onchangeAttr}>
                    ${optionsHtml}
                </select>
            `;
        } else {
            inputHtml = `
                <input
                    type="${type}"
                    id="${id}"
                    class="ewr-form-input"
                    placeholder="${placeholder}"
                    value="${value}"
                    ${disabledAttr}
                    ${onchangeAttr}
                >
            `;
        }

        this.innerHTML = `
            <div class="ewr-form-field">
                ${label ? `<label class="ewr-form-label" for="${id}">${label}${requiredMark}</label>` : ''}
                ${inputHtml}
                ${hint ? `<div class="ewr-form-hint">${hint}</div>` : ''}
            </div>
        `;
    }

    // Public API to get/set value
    get value() {
        const input = this.querySelector('input, textarea, select');
        return input ? input.value : '';
    }

    set value(val) {
        const input = this.querySelector('input, textarea, select');
        if (input) input.value = val;
    }
}

customElements.define('ewr-form-field', EwrFormField);


/**
 * EWR-Button-Group Component
 * A group of action buttons with consistent spacing and alignment
 *
 * Usage:
 * <ewr-button-group align="right">
 *     <button class="btn" onclick="cancel()">Cancel</button>
 *     <button class="btn btn-primary" onclick="save()">Save</button>
 * </ewr-button-group>
 */
class EwrButtonGroup extends HTMLElement {
    constructor() {
        super();
    }

    connectedCallback() {
        const align = this.getAttribute('align') || 'right'; // left, center, right
        const gap = this.getAttribute('gap') || '12px';

        // Capture buttons
        const buttons = Array.from(this.children);

        this.innerHTML = `
            <div class="ewr-button-group" style="display: flex; gap: ${gap}; justify-content: ${align === 'right' ? 'flex-end' : align === 'center' ? 'center' : 'flex-start'};"></div>
        `;

        const container = this.querySelector('.ewr-button-group');
        buttons.forEach(btn => container.appendChild(btn));
    }
}

customElements.define('ewr-button-group', EwrButtonGroup);


/**
 * EWR-RequiredSelectInput Component
 * A required select input with yellow outline and yellow label indicating it's required
 * Used for form fields that MUST be filled out (Department, Type, etc.)
 *
 * Usage:
 * <ewr-required-select-input
 *     label="Department"
 *     id="uploadDepartment"
 *     onchange="handleChange()">
 *     <option value="">Loading...</option>
 * </ewr-required-select-input>
 */
class EwrRequiredSelectInput extends HTMLElement {
    constructor() {
        super();
    }

    connectedCallback() {
        const label = this.getAttribute('label') || 'Select';
        const id = this.getAttribute('id') || 'requiredSelect';
        const onchange = this.getAttribute('onchange') || '';
        const hint = this.getAttribute('hint') || '';

        // Capture existing options before replacing innerHTML
        const existingOptions = Array.from(this.querySelectorAll('option'));
        let optionsHtml = '';
        existingOptions.forEach(opt => {
            optionsHtml += opt.outerHTML;
        });

        this.innerHTML = `
            <div class="form-group">
                <label class="required" for="${id}">
                    ${label} * <span style="font-size: 11px; font-weight: 400;">(Required)</span>
                </label>
                <select id="${id}" class="form-select required-field" required ${onchange ? `onchange="${onchange}"` : ''}>
                    ${optionsHtml}
                </select>
                ${hint ? `<div style="font-size: 12px; color: var(--text-muted); margin-top: 4px;">${hint}</div>` : ''}
            </div>
        `;
    }

    // Public API to get/set value
    get value() {
        const select = this.querySelector('select');
        return select ? select.value : '';
    }

    set value(val) {
        const select = this.querySelector('select');
        if (select) select.value = val;
    }

    // Get the select element
    getSelectElement() {
        return this.querySelector('select');
    }

    // Set options programmatically
    setOptions(options, defaultText = '') {
        const select = this.querySelector('select');
        if (select) {
            let html = defaultText ? `<option value="">${defaultText}</option>` : '';
            options.forEach(opt => {
                const value = opt.value || opt.id || opt.name || opt;
                const text = opt.text || opt.name || opt.id || opt;
                html += `<option value="${value}">${text}</option>`;
            });
            select.innerHTML = html;
        }
    }
}

customElements.define('ewr-required-select-input', EwrRequiredSelectInput);


/**
 * EWR-SelectInput Component
 * A standard (non-required) select input with consistent styling
 * Used for optional form fields (Subject, filters, etc.)
 *
 * Usage:
 * <ewr-select-input
 *     label="Subject/Product"
 *     id="uploadSubject"
 *     onchange="handleChange()">
 *     <option value="">None</option>
 * </ewr-select-input>
 */
class EwrSelectInput extends HTMLElement {
    constructor() {
        super();
    }

    connectedCallback() {
        const label = this.getAttribute('label') || 'Select';
        const id = this.getAttribute('id') || 'selectInput';
        const onchange = this.getAttribute('onchange') || '';
        const hint = this.getAttribute('hint') || '';

        // Capture existing options before replacing innerHTML
        const existingOptions = Array.from(this.querySelectorAll('option'));
        let optionsHtml = '';
        existingOptions.forEach(opt => {
            optionsHtml += opt.outerHTML;
        });

        this.innerHTML = `
            <div class="form-group">
                <label for="${id}">
                    ${label} <span style="font-size: 11px; font-weight: 400;">(Optional)</span>
                </label>
                <select id="${id}" class="form-select" ${onchange ? `onchange="${onchange}"` : ''}>
                    ${optionsHtml}
                </select>
                ${hint ? `<div style="font-size: 12px; color: var(--text-muted); margin-top: 4px;">${hint}</div>` : ''}
            </div>
        `;
    }

    // Public API to get/set value
    get value() {
        const select = this.querySelector('select');
        return select ? select.value : '';
    }

    set value(val) {
        const select = this.querySelector('select');
        if (select) select.value = val;
    }

    // Get the select element
    getSelectElement() {
        return this.querySelector('select');
    }

    // Set options programmatically
    setOptions(options, defaultText = '') {
        const select = this.querySelector('select');
        if (select) {
            let html = defaultText ? `<option value="">${defaultText}</option>` : '';
            options.forEach(opt => {
                const value = opt.value || opt.id || opt.name || opt;
                const text = opt.text || opt.name || opt.id || opt;
                html += `<option value="${value}">${text}</option>`;
            });
            select.innerHTML = html;
        }
    }
}

customElements.define('ewr-select-input', EwrSelectInput);


/**
 * EWR-DocumentList Component
 * A reusable document list container with header, count, and scrollable content area
 * Displays documents in a THREE-ROW grid layout (instead of single column)
 *
 * Usage:
 * <ewr-document-list
 *     id="documentBrowser"
 *     title="Matching Documents"
 *     empty-message="Select filters and click Load Documents"
 *     on-item-click="openDocumentPreview">
 * </ewr-document-list>
 */
class EwrDocumentList extends HTMLElement {
    constructor() {
        super();
    }

    connectedCallback() {
        const id = this.getAttribute('id') || 'documentList';
        const title = this.getAttribute('title') || 'Matching Documents';
        const emptyMessage = this.getAttribute('empty-message') || 'No documents loaded';
        const emptyIcon = this.getAttribute('empty-icon') || '📄';
        const emptyTitle = this.getAttribute('empty-title') || 'No Documents Loaded';

        this.innerHTML = `
            <div class="card-body ewr-document-list-container">
                <div class="ewr-document-list-header" style="position: relative;">
                    <span class="ewr-document-list-title">${title}: <span class="ewr-document-list-count" id="${id}-count">0</span></span>
                    <div class="ewr-doc-tag-legend">
                        <div class="ewr-doc-tag-legend-item">
                            <span class="ewr-doc-tag-legend-swatch swatch-department"></span>
                            <span class="ewr-doc-tag-legend-label">Department</span>
                        </div>
                        <div class="ewr-doc-tag-legend-item">
                            <span class="ewr-doc-tag-legend-swatch swatch-type"></span>
                            <span class="ewr-doc-tag-legend-label">Type</span>
                        </div>
                        <div class="ewr-doc-tag-legend-item">
                            <span class="ewr-doc-tag-legend-swatch swatch-subject"></span>
                            <span class="ewr-doc-tag-legend-label">Subject</span>
                        </div>
                    </div>
                </div>
                <div class="ewr-document-list-filters" id="${id}-filters" style="display: none;"></div>
                <div class="ewr-document-list-content" id="${id}-content">
                    <div class="ewr-document-list-empty">
                        <div class="ewr-document-list-empty-icon">${emptyIcon}</div>
                        <div class="ewr-document-list-empty-title">${emptyTitle}</div>
                        <div class="ewr-document-list-empty-text">${emptyMessage}</div>
                    </div>
                </div>
            </div>
        `;

        // Inject filter tag styles if not already present
        this._injectFilterStyles();
    }

    _injectFilterStyles() {
        if (document.getElementById('ewr-document-list-filter-styles')) return;

        const style = document.createElement('style');
        style.id = 'ewr-document-list-filter-styles';
        style.textContent = `
            .ewr-document-list-filters {
                display: flex;
                flex-wrap: wrap;
                gap: 8px;
                padding: 8px 16px 12px;
                align-items: center;
            }

            .ewr-filter-tag {
                display: inline-flex;
                align-items: center;
                gap: 6px;
                padding: 4px 10px;
                background: rgba(59, 130, 246, 0.15);
                border: 1px solid rgba(59, 130, 246, 0.3);
                border-radius: 4px;
                font-size: 12px;
                color: #93c5fd;
            }

            .ewr-filter-tag-label {
                color: #64748b;
                font-weight: 500;
            }

            .ewr-filter-tag-value {
                color: #e2e8f0;
                font-weight: 600;
            }
        `;
        document.head.appendChild(style);
    }

    // Get content container element
    getContentElement() {
        const id = this.getAttribute('id') || 'documentList';
        return this.querySelector(`#${id}-content`);
    }

    // Get count element
    getCountElement() {
        const id = this.getAttribute('id') || 'documentList';
        return this.querySelector(`#${id}-count`);
    }

    // Get filters container element
    getFiltersElement() {
        const id = this.getAttribute('id') || 'documentList';
        return this.querySelector(`#${id}-filters`);
    }

    // Set filter tags to display active filters
    // filters: { department: 'Value', docType: 'Value', subject: 'Value' }
    setFilterTags(filters) {
        const filtersEl = this.getFiltersElement();
        if (!filtersEl) return;

        const tags = [];

        if (filters.department) {
            tags.push(`<span class="ewr-filter-tag"><span class="ewr-filter-tag-label">Department:</span><span class="ewr-filter-tag-value">${filters.department}</span></span>`);
        }
        if (filters.docType) {
            tags.push(`<span class="ewr-filter-tag"><span class="ewr-filter-tag-label">Doc Type:</span><span class="ewr-filter-tag-value">${filters.docType}</span></span>`);
        }
        if (filters.subject) {
            tags.push(`<span class="ewr-filter-tag"><span class="ewr-filter-tag-label">Subject:</span><span class="ewr-filter-tag-value">${filters.subject}</span></span>`);
        }

        if (tags.length > 0) {
            filtersEl.innerHTML = tags.join('');
            filtersEl.style.display = 'flex';
        } else {
            filtersEl.innerHTML = '';
            filtersEl.style.display = 'none';
        }
    }

    // Clear filter tags
    clearFilterTags() {
        const filtersEl = this.getFiltersElement();
        if (filtersEl) {
            filtersEl.innerHTML = '';
            filtersEl.style.display = 'none';
        }
    }

    // Show loading state
    showLoading() {
        const content = this.getContentElement();
        const count = this.getCountElement();
        if (content) {
            content.innerHTML = `
                <div class="ewr-document-list-loading">
                    <div class="ewr-document-list-spinner"></div>
                    <span>Loading documents...</span>
                </div>
            `;
        }
        if (count) count.textContent = '...';
    }

    // Show error state
    showError(message) {
        const content = this.getContentElement();
        const count = this.getCountElement();
        if (content) {
            content.innerHTML = `
                <div class="ewr-document-list-empty">
                    <div class="ewr-document-list-empty-icon">⚠️</div>
                    <div class="ewr-document-list-empty-title">Error Loading Documents</div>
                    <div class="ewr-document-list-empty-text">${message || 'Failed to load'}</div>
                </div>
            `;
        }
        if (count) count.textContent = '!';
    }

    // Show empty state
    showEmpty(message) {
        const content = this.getContentElement();
        const count = this.getCountElement();
        const emptyIcon = this.getAttribute('empty-icon') || '📄';
        const emptyTitle = this.getAttribute('empty-title') || 'No Documents Found';
        if (content) {
            content.innerHTML = `
                <div class="ewr-document-list-empty">
                    <div class="ewr-document-list-empty-icon">${emptyIcon}</div>
                    <div class="ewr-document-list-empty-title">${emptyTitle}</div>
                    <div class="ewr-document-list-empty-text">${message || 'Try adjusting your filters'}</div>
                </div>
            `;
        }
        if (count) count.textContent = '0';
    }

    // Set documents and display in THREE-ROW grid
    // Accepts either array of strings (titles) or array of document objects
    // Document object format: { title, department?, type?, subject? }
    setDocuments(documents, onItemClick) {
        const content = this.getContentElement();
        const count = this.getCountElement();
        const onClickAttr = this.getAttribute('on-item-click');

        if (!content || !count) return;

        const docCount = documents.length;
        count.textContent = docCount;

        if (docCount > 0) {
            // Build THREE-ROW grid HTML with file type icons and category tags
            const gridHtml = `
                <div class="ewr-document-grid-3row">
                    ${documents.map(doc => {
                        // Support both string (title only) and object format
                        const isObject = typeof doc === 'object' && doc !== null;
                        const title = isObject ? (doc.title || 'Untitled') : doc;
                        const department = isObject ? doc.department : null;
                        const docType = isObject ? doc.type : null;
                        const subject = isObject ? doc.subject : null;

                        const escapedTitle = this._escapeHtml(title);
                        const escapedForClick = title.replace(/'/g, "\\'");
                        const clickHandler = onClickAttr ? `${onClickAttr}('${escapedForClick}')` :
                                           onItemClick ? `(${onItemClick})('${escapedForClick}')` : '';
                        const fileIcon = this._getFileIcon(title);

                        // Build category tags HTML
                        const tags = this._buildCategoryTags(department, docType, subject);

                        return `
                            <div class="ewr-document-grid-item ${tags ? 'has-tags' : ''}"
                                 title="${escapedTitle}"
                                 ${clickHandler ? `onclick="${clickHandler}"` : ''}>
                                <span class="ewr-document-grid-icon">${fileIcon}</span>
                                <div class="ewr-document-grid-info">
                                    <div class="ewr-document-grid-title">${escapedTitle}</div>
                                    ${tags}
                                </div>
                            </div>
                        `;
                    }).join('')}
                </div>
            `;
            content.innerHTML = gridHtml;
        } else {
            this.showEmpty();
        }
    }

    // Build category tags HTML for a document
    _buildCategoryTags(department, type, subject) {
        const tags = [];

        if (department) {
            tags.push(`<span class="ewr-doc-tag tag-department" title="Department">${this._escapeHtml(this._formatCategoryName(department))}</span>`);
        }
        if (type) {
            tags.push(`<span class="ewr-doc-tag tag-type" title="Document Type">${this._escapeHtml(this._formatCategoryName(type))}</span>`);
        }
        if (subject) {
            tags.push(`<span class="ewr-doc-tag tag-subject" title="Subject">${this._escapeHtml(this._formatCategoryName(subject))}</span>`);
        }

        return tags.length > 0 ? `<div class="ewr-doc-tags">${tags.join('')}</div>` : '';
    }

    // Format category name for display (convert snake_case/kebab-case to Title Case)
    _formatCategoryName(name) {
        if (!name) return '';
        return name
            .replace(/[_-]/g, ' ')
            .replace(/\b\w/g, c => c.toUpperCase());
    }

    // Get file type icon based on extension (returns Lucide icon name)
    _getFileIcon(filename) {
        const ext = (filename.split('.').pop() || '').toLowerCase();
        const iconMap = {
            // Documents
            'pdf': 'file-text',
            'doc': 'file-text',
            'docx': 'file-text',
            'odt': 'file-text',
            'rtf': 'file-text',
            // Spreadsheets
            'xls': 'file-spreadsheet',
            'xlsx': 'file-spreadsheet',
            'csv': 'file-spreadsheet',
            'ods': 'file-spreadsheet',
            // Presentations
            'ppt': 'presentation',
            'pptx': 'presentation',
            'odp': 'presentation',
            // Images
            'jpg': 'file-image',
            'jpeg': 'file-image',
            'png': 'file-image',
            'gif': 'file-image',
            'svg': 'file-image',
            'webp': 'file-image',
            'bmp': 'file-image',
            'tiff': 'file-image',
            // Text
            'txt': 'file-text',
            'md': 'file-text',
            'json': 'file-json',
            'xml': 'file-code',
            'html': 'file-code',
            'htm': 'file-code',
            // Code
            'js': 'file-code',
            'ts': 'file-code',
            'py': 'file-code',
            'sql': 'database',
            'css': 'file-code',
            'scss': 'file-code',
            // Archives
            'zip': 'file-archive',
            'rar': 'file-archive',
            '7z': 'file-archive',
            'tar': 'file-archive',
            'gz': 'file-archive',
            // Audio
            'mp3': 'file-audio',
            'wav': 'file-audio',
            'ogg': 'file-audio',
            'm4a': 'file-audio',
            'flac': 'file-audio',
            // Video
            'mp4': 'file-video',
            'avi': 'file-video',
            'mov': 'file-video',
            'mkv': 'file-video',
            'webm': 'file-video'
        };
        const iconName = iconMap[ext] || 'file';
        return `<ewr-icon name="${iconName}" size="18"></ewr-icon>`;
    }

    // Helper to escape HTML
    _escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }
}

customElements.define('ewr-document-list', EwrDocumentList);


/**
 * EWR-Icon Component
 * A simple wrapper for Lucide icons providing a developer-friendly interface
 * Auto-loads Lucide library from CDN on first use - no manual script include needed!
 *
 * Usage:
 * <ewr-icon name="database"></ewr-icon>
 * <ewr-icon name="file-text" size="32"></ewr-icon>
 * <ewr-icon name="alert-circle" size="20" color="#ef4444"></ewr-icon>
 * <ewr-icon name="check-circle" stroke-width="1.5"></ewr-icon>
 *
 * Common icon names:
 * - file-text, file, folder, database, server
 * - search, filter, settings, menu, x (close)
 * - check, check-circle, alert-circle, info, help-circle
 * - user, users, mail, phone, calendar
 * - edit, trash-2, download, upload, plus, minus
 * - arrow-up, arrow-down, arrow-left, arrow-right
 * - chevron-up, chevron-down, chevron-left, chevron-right
 * - refresh-cw, loader, clock, eye, eye-off
 *
 * Full icon list: https://lucide.dev/icons/
 */
class EwrIcon extends HTMLElement {
    // Static property to track Lucide loading state
    static _lucideLoading = null;
    static _lucideLoaded = false;

    constructor() {
        super();
    }

    connectedCallback() {
        this._loadAndRender();
    }

    static get observedAttributes() {
        return ['name', 'size', 'color', 'stroke-width'];
    }

    attributeChangedCallback(name, oldValue, newValue) {
        if (oldValue !== newValue && EwrIcon._lucideLoaded) {
            this.render();
        }
    }

    async _loadAndRender() {
        // If Lucide is already loaded, render immediately
        if (typeof lucide !== 'undefined') {
            EwrIcon._lucideLoaded = true;
            this.render();
            return;
        }

        // Show placeholder while loading
        const name = this.getAttribute('name') || 'circle';
        const size = this.getAttribute('size') || '24';
        this.innerHTML = `<span class="ewr-icon-loading" style="display: inline-block; width: ${size}px; height: ${size}px; opacity: 0.3;"></span>`;

        // Load Lucide if not already loading
        if (!EwrIcon._lucideLoading) {
            EwrIcon._lucideLoading = new Promise((resolve, reject) => {
                const script = document.createElement('script');
                script.src = 'https://unpkg.com/lucide@latest';
                script.onload = () => {
                    EwrIcon._lucideLoaded = true;
                    resolve();
                };
                script.onerror = () => reject(new Error('Failed to load Lucide'));
                document.head.appendChild(script);
            });
        }

        try {
            await EwrIcon._lucideLoading;
            this.render();
        } catch (error) {
            console.error('EWR-Icon: Failed to load Lucide library', error);
            this.innerHTML = `<span style="font-size: ${Math.round(size * 0.4)}px; opacity: 0.5;">[${name}]</span>`;
        }
    }

    render() {
        const name = this.getAttribute('name') || 'circle';
        const size = this.getAttribute('size') || '24';
        const color = this.getAttribute('color') || 'currentColor';
        const strokeWidth = this.getAttribute('stroke-width') || '2';

        // Create the Lucide icon element
        this.innerHTML = `<i data-lucide="${name}" style="display: inline-flex; width: ${size}px; height: ${size}px; color: ${color};"></i>`;

        // Call Lucide to render the icon
        if (typeof lucide !== 'undefined') {
            lucide.createIcons({
                attrs: {
                    'stroke-width': strokeWidth
                }
            });
        }
    }
}

customElements.define('ewr-icon', EwrIcon);

/**
 * EWR-Sidebar Component
 * Complete sidebar with navigation, brand, and footer for public pages
 *
 * Usage:
 * <ewr-sidebar></ewr-sidebar>
 *
 * Note: Requires sidebar.js to be loaded for PUBLIC_NAV_CONFIG
 */
class EwrSidebar extends HTMLElement {
    constructor() {
        super();
    }

    connectedCallback() {
        this.className = 'app-sidebar';
        this.innerHTML = `
            <style>
                /* Sidebar brand layout - horizontal with left and right sections */
                .sidebar-brand {
                    padding: 12px 14px;
                    border-bottom: 1px solid rgba(148, 163, 184, 0.25);
                    display: flex;
                    align-items: stretch;
                    gap: 12px;
                    background: rgba(0, 0, 0, 0.2);
                    flex-shrink: 0;
                }

                .sidebar-brand-left {
                    display: flex;
                    align-items: center;
                    flex-shrink: 0;
                }

                .sidebar-brand-right {
                    flex: 1;
                    display: flex;
                    flex-direction: column;
                    gap: 6px;
                    min-width: 0;
                }

                .sidebar-user-info {
                    background: rgba(255, 255, 255, 0.04);
                    border: 1px solid rgba(255, 255, 255, 0.06);
                    border-radius: 6px;
                    padding: 6px 10px;
                }

                .sidebar-user-row {
                    display: flex;
                    align-items: center;
                    gap: 6px;
                }

                .sidebar-user-row + .sidebar-user-row {
                    margin-top: 2px;
                }

                .sidebar-user-label {
                    font-size: 8px;
                    font-weight: 600;
                    text-transform: uppercase;
                    letter-spacing: 0.5px;
                    color: #64748b;
                    min-width: 24px;
                }

                .sidebar-user-value {
                    font-size: 11px;
                    font-weight: 600;
                    color: #e2e8f0;
                    white-space: nowrap;
                    overflow: hidden;
                    text-overflow: ellipsis;
                }

                .sidebar-status-wrapper {
                    /* Container for ewr-website-status */
                }

                .sidebar-status-wrapper ewr-website-status {
                    display: block;
                }
            </style>
            <div class="sidebar-brand">
                <div class="sidebar-brand-left">
                    <div class="sidebar-logo">EWR</div>
                </div>
                <div class="sidebar-brand-right">
                    <div class="sidebar-user-info">
                        <div class="sidebar-user-row">
                            <span class="sidebar-user-label">User</span>
                            <span class="sidebar-user-value" id="userName">Loading...</span>
                        </div>
                        <div class="sidebar-user-row">
                            <span class="sidebar-user-label">Role</span>
                            <span class="sidebar-user-value" id="userRole">USER</span>
                        </div>
                    </div>
                    <div class="sidebar-status-wrapper">
                        <ewr-website-status></ewr-website-status>
                    </div>
                </div>
            </div>
            <nav class="sidebar-nav"></nav>
            <div class="sidebar-footer">
                <ewr-logout-button></ewr-logout-button>
            </div>
        `;

        // Initialize navigation after render
        this.initNavigation();
        this.updateUserInfo();
    }

    initNavigation() {
        const navContainer = this.querySelector('.sidebar-nav');
        if (!navContainer) return;

        // Use PUBLIC_NAV_CONFIG from sidebar.js if available
        const config = window.PUBLIC_NAV_CONFIG || {
            main: {
                title: 'Main',
                items: [{ id: 'dashboard', name: 'Dashboard', url: '/', icon: '<ewr-icon name="layout-dashboard" size="20"></ewr-icon>' }]
            },
            services: {
                title: 'Services',
                items: [
                    { id: 'sql-query', name: 'SQL Chat', url: '/sql', icon: '<ewr-icon name="database" size="20"></ewr-icon>' },
                    { id: 'kb-assistant', name: 'Knowledge Base Assistant', url: '/knowledge-base/index.html', icon: '<ewr-icon name="message-circle" size="20"></ewr-icon>' },
                    { id: 'document-browser', name: 'Document Browser', url: '/knowledge-base/documents.html', icon: '<ewr-icon name="folder" size="20"></ewr-icon>' }
                ]
            },
            system: {
                title: 'System',
                items: [{ id: 'admin', name: 'Admin', url: '/admin/index.html', icon: '<ewr-icon name="settings" size="20"></ewr-icon>' }]
            }
        };

        let html = '';
        for (const [sectionKey, section] of Object.entries(config)) {
            html += `<div class="nav-section"><div class="nav-section-title">${section.title}</div>`;

            for (const item of section.items) {
                const isActive = this.isActivePage(item.url);
                const url = item.url.startsWith('/') ? item.url : '/' + item.url;
                html += `
                    <a href="${url}" class="nav-item${isActive ? ' active' : ''}" data-nav-id="${item.id}">
                        <span class="nav-icon">${item.icon}</span>
                        <span class="nav-text">${item.name}</span>
                    </a>
                `;
            }
            html += '</div>';
        }

        navContainer.innerHTML = html;
    }

    isActivePage(itemUrl) {
        const currentPath = window.location.pathname;
        const normalizedCurrent = currentPath.replace(/\/index\.html$/, '/').replace(/\/$/, '') || '/';
        let normalizedItem = itemUrl.replace(/\/index\.html$/, '/').replace(/\/$/, '') || '/';
        if (!normalizedItem.startsWith('/')) normalizedItem = '/' + normalizedItem;
        return normalizedCurrent === normalizedItem;
    }

    async updateUserInfo() {
        try {
            if (typeof AuthClient === 'undefined') return;
            const auth = new AuthClient();
            const user = await auth.getUser();
            if (user) {
                const userNameEl = this.querySelector('#userName');
                const userRoleEl = this.querySelector('#userRole');
                if (userNameEl) userNameEl.textContent = user.username;
                if (userRoleEl) userRoleEl.textContent = user.role || 'User';
            }
        } catch (error) {
            console.error('Failed to update sidebar user info:', error);
        }
    }
}

customElements.define('ewr-sidebar', EwrSidebar);

/**
 * EWR-Page-Header Component
 * Standard page header with title, subtitle, status indicator, and logout button
 *
 * Usage:
 * <ewr-page-header
 *     title="SQL Chat"
 *     subtitle="Ask questions about your data in plain English">
 * </ewr-page-header>
 */
class EwrPageHeader extends HTMLElement {
    constructor() {
        super();
    }

    connectedCallback() {
        const title = this.getAttribute('title') || 'Page Title';
        const subtitle = this.getAttribute('subtitle') || '';

        this.className = 'app-header';
        this.innerHTML = `
            <div class="header-content">
                <div class="header-left">
                    <div>
                        <h1 class="header-title">${title}</h1>
                        ${subtitle ? `<p class="header-subtitle">${subtitle}</p>` : ''}
                    </div>
                </div>
                <div class="header-right">
                    <div class="header-status" id="systemStatus">
                        <div class="status-dot" id="statusDot"></div>
                        <span class="status-text" id="statusText" style="color: #e8f2ff;">Connecting...</span>
                    </div>
                </div>
            </div>
        `;
    }

    // Method to update title dynamically
    setTitle(title) {
        const titleEl = this.querySelector('.header-title');
        if (titleEl) titleEl.textContent = title;
    }

    // Method to update subtitle dynamically
    setSubtitle(subtitle) {
        let subtitleEl = this.querySelector('.header-subtitle');
        if (subtitle) {
            if (!subtitleEl) {
                subtitleEl = document.createElement('p');
                subtitleEl.className = 'header-subtitle';
                this.querySelector('.header-left > div').appendChild(subtitleEl);
            }
            subtitleEl.textContent = subtitle;
        } else if (subtitleEl) {
            subtitleEl.remove();
        }
    }
}

customElements.define('ewr-page-header', EwrPageHeader);


// NOTE: ewr-website-status moved to Shadow DOM component at /js/components/status/ewr-website-status.js


/**
 * EWR-Admin-Sidebar Component
 * Admin page sidebar with dynamic navigation loaded from server
 *
 * Usage:
 * <ewr-admin-sidebar></ewr-admin-sidebar>
 *
 * Note: Requires admin/js/sidebar.js to be loaded for initSidebar function
 */
class EwrAdminSidebar extends HTMLElement {
    constructor() {
        super();
    }

    connectedCallback() {
        this.className = 'app-sidebar';
        this.innerHTML = `
            <style>
                /* Sidebar brand layout - horizontal with left and right sections */
                .sidebar-brand {
                    padding: 12px 14px;
                    border-bottom: 1px solid rgba(148, 163, 184, 0.25);
                    display: flex;
                    align-items: stretch;
                    gap: 12px;
                    background: rgba(0, 0, 0, 0.2);
                    flex-shrink: 0;
                }

                .sidebar-brand-left {
                    display: flex;
                    align-items: center;
                    flex-shrink: 0;
                }

                .sidebar-brand-right {
                    flex: 1;
                    display: flex;
                    flex-direction: column;
                    gap: 6px;
                    min-width: 0;
                }

                .sidebar-user-info {
                    background: rgba(255, 255, 255, 0.04);
                    border: 1px solid rgba(255, 255, 255, 0.06);
                    border-radius: 6px;
                    padding: 6px 10px;
                }

                .sidebar-user-row {
                    display: flex;
                    align-items: center;
                    gap: 6px;
                }

                .sidebar-user-row + .sidebar-user-row {
                    margin-top: 2px;
                }

                .sidebar-user-label {
                    font-size: 8px;
                    font-weight: 600;
                    text-transform: uppercase;
                    letter-spacing: 0.5px;
                    color: #64748b;
                    min-width: 24px;
                }

                .sidebar-user-value {
                    font-size: 11px;
                    font-weight: 600;
                    color: #e2e8f0;
                    white-space: nowrap;
                    overflow: hidden;
                    text-overflow: ellipsis;
                }

                .sidebar-status-wrapper {
                    /* Container for ewr-website-status */
                }

                .sidebar-status-wrapper ewr-website-status {
                    display: block;
                }

                /* Override website-status styling for sidebar context */
                .sidebar-status-wrapper .website-status {
                    margin-top: 0;
                    padding: 3px 8px;
                }
            </style>
            <div class="sidebar-brand">
                <div class="sidebar-brand-left">
                    <div class="sidebar-logo">EWR</div>
                </div>
                <div class="sidebar-brand-right">
                    <div class="sidebar-user-info">
                        <div class="sidebar-user-row">
                            <span class="sidebar-user-label">User</span>
                            <span class="sidebar-user-value" id="userName">Loading...</span>
                        </div>
                        <div class="sidebar-user-row">
                            <span class="sidebar-user-label">Role</span>
                            <span class="sidebar-user-value" id="userRole">USER</span>
                        </div>
                    </div>
                    <div class="sidebar-status-wrapper">
                        <ewr-website-status></ewr-website-status>
                    </div>
                </div>
            </div>
            <nav class="sidebar-nav">
                <!-- Navigation will be dynamically generated by admin sidebar.js -->
            </nav>
            <div class="sidebar-footer">
                <ewr-logout-button></ewr-logout-button>
            </div>
        `;

        // Initialize admin sidebar after render
        this.initAdminNavigation();
    }

    async initAdminNavigation() {
        // Wait for admin sidebar.js to be loaded and call its init function
        if (typeof window.initSidebar === 'function') {
            await window.initSidebar();
        } else {
            // If initSidebar not available, set up a watcher
            const checkInterval = setInterval(async () => {
                if (typeof window.initSidebar === 'function') {
                    clearInterval(checkInterval);
                    await window.initSidebar();
                }
            }, 50);
            // Clear after 5 seconds to prevent infinite loop
            setTimeout(() => clearInterval(checkInterval), 5000);
        }
    }
}

customElements.define('ewr-admin-sidebar', EwrAdminSidebar);


/**
 * EWR-Admin-Header Component
 * Admin page header with title, optional refresh controls
 *
 * Usage:
 * <ewr-admin-header title="Server Administration"></ewr-admin-header>
 * <ewr-admin-header title="User Management" show-refresh="false"></ewr-admin-header>
 */
class EwrAdminHeader extends HTMLElement {
    constructor() {
        super();
    }

    connectedCallback() {
        const title = this.getAttribute('title') || 'Admin Dashboard';
        const showRefresh = this.getAttribute('show-refresh') !== 'false';

        this.className = 'app-header';
        this.innerHTML = `
            <div class="header-content">
                <div class="header-left">
                    <div>
                        <h1 class="header-title">${title}</h1>
                    </div>
                </div>
                <div class="header-right" style="display: flex; gap: 12px; align-items: center;">
                    ${showRefresh ? `
                        <div class="refresh-indicator" id="refreshIndicator">
                            <ewr-icon name="refresh-cw" size="16"></ewr-icon>
                            <span>Auto-refresh: <span id="refreshCountdown">5s</span></span>
                        </div>
                        <button class="btn btn-secondary" onclick="refreshAllData()">Refresh Now</button>
                    ` : ''}
                </div>
            </div>
        `;
    }

    // Method to update title dynamically
    setTitle(title) {
        const titleEl = this.querySelector('.header-title');
        if (titleEl) titleEl.textContent = title;
    }
}

customElements.define('ewr-admin-header', EwrAdminHeader);


/**
 * EWR-ChatMessage Component
 * A compact user message component for chat interfaces
 * Styled directly via CSS on the custom element
 *
 * Usage:
 * <ewr-chat-message>Your message text</ewr-chat-message>
 */
class EwrChatMessage extends HTMLElement {
    constructor() {
        super();
    }

    connectedCallback() {
        // Component content is used directly, no wrapper needed
        // Styling is applied to the custom element itself via CSS
    }

    // Set message content dynamically
    setContent(html) {
        this.innerHTML = html;
    }

    // Get message content
    getContent() {
        return this.innerHTML;
    }
}

customElements.define('ewr-chat-message', EwrChatMessage);


/**
 * EWR-ChatMessageAssistant Component
 * A compact assistant message component for chat interfaces
 * Styled directly via CSS on the custom element
 *
 * Usage:
 * <ewr-chat-message-assistant>Response text</ewr-chat-message-assistant>
 */
class EwrChatMessageAssistant extends HTMLElement {
    constructor() {
        super();
    }

    connectedCallback() {
        // Component content is used directly, no wrapper needed
        // Styling is applied to the custom element itself via CSS
    }

    // Set message content dynamically
    setContent(html) {
        this.innerHTML = html;
    }

    // Get message content
    getContent() {
        return this.innerHTML;
    }
}

customElements.define('ewr-chat-message-assistant', EwrChatMessageAssistant);


/**
 * EWR-ChatWindow Component
 * A complete chat window container with messages area, footer, feedback buttons, and timing metrics
 *
 * Usage:
 * <ewr-chat-window
 *     id="myChatWindow"
 *     messages-id="chatMessages"
 *     input-id="chatInput"
 *     send-btn-id="sendBtn"
 *     placeholder="Ask a question..."
 *     empty-icon="💬"
 *     empty-title="Start a Conversation"
 *     empty-text="Ask questions in natural language."
 *     on-send="sendMessage()"
 *     on-clear="clearChat()"
 *     on-keydown="handleKeyDown(event)"
 *     show-timing="true"
 *     show-feedback="true">
 *
 *     <!-- Optional: Timing metrics slot for customization -->
 *     <div slot="timing-metrics">
 *         <!-- Custom timing metrics content -->
 *     </div>
 * </ewr-chat-window>
 *
 * Attributes:
 * - messages-id: ID for the chat messages container (default: 'chatMessages')
 * - input-id: ID for the chat input textarea (default: 'chatInput')
 * - send-btn-id: ID for the send button (default: 'sendBtn')
 * - placeholder: Placeholder text for input (default: 'Ask a question...')
 * - empty-icon: Icon shown in empty state (default: '💬')
 * - empty-title: Title shown in empty state (default: 'Start a Conversation')
 * - empty-text: Text shown in empty state (default: 'Ask questions in natural language.')
 * - on-send: Function to call when send is clicked (default: 'sendMessage()')
 * - on-clear: Function to call when clear is clicked (default: 'clearChat()')
 * - on-keydown: Function to call on keydown (default: 'handleKeyDown(event)')
 * - show-timing: Show timing metrics bar (default: 'false')
 * - show-feedback: Enable feedback buttons on messages (default: 'true')
 */
class EwrChatWindow extends HTMLElement {
    constructor() {
        super();
    }

    connectedCallback() {
        // Get attributes with defaults
        const messagesId = this.getAttribute('messages-id') || 'chatMessages';
        const inputId = this.getAttribute('input-id') || 'chatInput';
        const sendBtnId = this.getAttribute('send-btn-id') || 'sendBtn';
        const placeholder = this.getAttribute('placeholder') || 'Ask a question...';
        const emptyIcon = this.getAttribute('empty-icon') || '💬';
        const emptyTitle = this.getAttribute('empty-title') || 'Start a Conversation';
        const emptyText = this.getAttribute('empty-text') || 'Ask questions in natural language.';
        const onSend = this.getAttribute('on-send') || 'sendMessage()';
        const onClear = this.getAttribute('on-clear') || 'clearChat()';
        const onKeyDown = this.getAttribute('on-keydown') || 'handleKeyDown(event)';
        const showTiming = this.getAttribute('show-timing') === 'true';
        const showFeedback = this.getAttribute('show-feedback') !== 'false';
        const showEmpty = this.getAttribute('show-empty') !== 'false';
        const alertId = this.getAttribute('alert-id') || 'contextLimitAlert';
        const timingId = this.getAttribute('timing-id') || 'timingMetrics';

        // Check for slots
        const customTimingSlot = this.querySelector('[slot="timing-metrics"]');
        const statusBarSlot = this.querySelector('[slot="status-bar"]');
        const initialMessageSlot = this.querySelector('[slot="initial-message"]');

        const timingMetricsHtml = customTimingSlot ? '' : this._getDefaultTimingMetrics(timingId);

        // Build initial message content
        let initialContent = '';
        if (showEmpty) {
            initialContent = `
                <div class="empty-chat" id="emptyChat">
                    <div class="empty-icon">${emptyIcon}</div>
                    <div class="empty-title">${emptyTitle}</div>
                    <div class="empty-text">${emptyText}</div>
                </div>
            `;
        }

        // Build the component HTML
        this.innerHTML = `
            <div class="ewr-chat-container">
                <!-- Chat Messages Area -->
                <div class="ewr-chat-window" id="${messagesId}">
                    ${initialContent}
                </div>

                <!-- Status Bar Area (slot for context-remaining, etc.) -->
                <div class="ewr-chat-status-bar" id="${messagesId}-status-bar"></div>

                <!-- Timing Metrics Area -->
                ${showTiming ? `
                    <div class="timing-metrics" id="${timingId}">
                        ${customTimingSlot ? '' : timingMetricsHtml}
                    </div>
                ` : ''}

                <!-- Chat Footer -->
                <div class="ewr-chat-footer">
                    <div id="${alertId}" class="context-limit-alert" style="display: none;">
                        Context window full - click Clear Chat to continue
                    </div>
                    <div class="ewr-chat-input-row">
                        <textarea
                            class="ewr-chat-entry"
                            id="${inputId}"
                            placeholder="${placeholder}"
                            rows="1"
                            onkeydown="${onKeyDown}"
                        ></textarea>
                        <div class="ewr-chat-buttons">
                            <button class="ewr-button-send" id="${sendBtnId}" onclick="${onSend}" title="Send">&#10148;</button>
                            <button class="ewr-button-clear" onclick="${onClear}">Clear</button>
                        </div>
                    </div>
                </div>
            </div>
        `;

        // Move initial-message slot content into messages area
        if (initialMessageSlot) {
            const messagesEl = this.querySelector(`#${messagesId}`);
            if (messagesEl) {
                initialMessageSlot.removeAttribute('slot');
                messagesEl.appendChild(initialMessageSlot);
            }
        }

        // Move status-bar slot content into status bar area
        if (statusBarSlot) {
            const statusBarArea = this.querySelector('.ewr-chat-status-bar');
            if (statusBarArea) {
                statusBarSlot.removeAttribute('slot');
                // Move all children from slot to status bar
                while (statusBarSlot.firstChild) {
                    statusBarArea.appendChild(statusBarSlot.firstChild);
                }
                statusBarSlot.remove();
            }
        }

        // If custom timing slot was provided, move it into place
        if (customTimingSlot && showTiming) {
            const timingArea = this.querySelector('.timing-metrics');
            if (timingArea) {
                customTimingSlot.removeAttribute('slot');
                timingArea.appendChild(customTimingSlot);
            }
        }

        // Store config for later use
        this._showFeedback = showFeedback;
        this._messagesId = messagesId;
    }

    // Default timing metrics HTML
    _getDefaultTimingMetrics(timingId) {
        return `
            <div class="timing-metric" id="step-preprocessing">
                <span class="timing-metric-label">Analyze</span>
                <span class="timing-metric-value">--</span>
            </div>
            <span class="timing-separator">›</span>
            <div class="timing-metric" id="step-security">
                <span class="timing-metric-label">Security</span>
                <span class="timing-metric-value">--</span>
            </div>
            <span class="timing-separator">›</span>
            <div class="timing-metric" id="step-rules">
                <span class="timing-metric-label">Rules</span>
                <span class="timing-metric-value">--</span>
            </div>
            <span class="timing-separator">›</span>
            <div class="timing-metric" id="step-schema">
                <span class="timing-metric-label">Schema</span>
                <span class="timing-metric-value">--</span>
            </div>
            <span class="timing-separator">›</span>
            <div class="timing-metric" id="step-generating">
                <span class="timing-metric-label">Generate</span>
                <span class="timing-metric-value">--</span>
            </div>
            <span class="timing-separator">›</span>
            <div class="timing-metric" id="step-fixing">
                <span class="timing-metric-label">Fix</span>
                <span class="timing-metric-value">--</span>
            </div>
            <span class="timing-separator">›</span>
            <div class="timing-metric" id="step-executing">
                <span class="timing-metric-label">Execute</span>
                <span class="timing-metric-value">--</span>
            </div>
            <div class="timing-metric timing-total" id="step-total">
                <span class="timing-metric-label">Total:</span>
                <span class="timing-metric-value" id="timingTotal">--</span>
            </div>
        `;
    }

    // Add a user message to the chat
    addUserMessage(text, options = {}) {
        const messagesEl = this.querySelector(`#${this._messagesId}`);
        if (!messagesEl) return null;

        // Hide empty state
        const emptyChat = messagesEl.querySelector('.empty-chat');
        if (emptyChat) emptyChat.style.display = 'none';

        const messageId = options.id || `user-msg-${Date.now()}`;
        const timestamp = options.timestamp || new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
        const avatar = options.avatar || 'U';
        const name = options.name || 'You';

        const messageEl = document.createElement('ewr-chat-message');
        messageEl.id = messageId;
        messageEl.setAttribute('avatar', avatar);
        messageEl.setAttribute('name', name);
        messageEl.setAttribute('timestamp', timestamp);
        messageEl.innerHTML = text;

        messagesEl.appendChild(messageEl);
        this.scrollToBottom();

        return messageEl;
    }

    // Add an assistant message to the chat
    addAssistantMessage(text, options = {}) {
        const messagesEl = this.querySelector(`#${this._messagesId}`);
        if (!messagesEl) return null;

        // Hide empty state
        const emptyChat = messagesEl.querySelector('.empty-chat');
        if (emptyChat) emptyChat.style.display = 'none';

        const messageId = options.id || `asst-msg-${Date.now()}`;
        const timestamp = options.timestamp || new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
        const avatar = options.avatar || 'AI';
        const name = options.name || 'Assistant';

        const messageEl = document.createElement('ewr-chat-message-assistant');
        messageEl.id = messageId;
        messageEl.setAttribute('avatar', avatar);
        messageEl.setAttribute('name', name);
        messageEl.setAttribute('timestamp', timestamp);
        messageEl.setAttribute('message-id', messageId);
        messageEl.innerHTML = text;

        messagesEl.appendChild(messageEl);

        // Add feedback buttons if enabled
        if (this._showFeedback) {
            this._addFeedbackButtons(messageEl, messageId);
        }

        this.scrollToBottom();

        return messageEl;
    }

    // Add feedback buttons to a message
    _addFeedbackButtons(messageEl, messageId) {
        const contentEl = messageEl.querySelector('.ewr-message-content');
        if (!contentEl) return;

        const feedbackHtml = `
            <div class="feedback-buttons" style="margin-top: 12px; border-top: 1px solid #334155; padding-top: 12px; display: flex; align-items: center; gap: 8px;">
                <span class="feedback-label" style="font-size: 12px; color: #64748b;">Was this helpful?</span>
                <button class="feedback-btn-icon" data-message-id="${messageId}" data-feedback="positive" title="Thumbs up" style="background: transparent; border: 1px solid #334155; border-radius: 4px; padding: 4px 8px; cursor: pointer; color: #94a3b8; transition: all 0.2s;">
                    <ewr-icon name="thumbs-up" size="16"></ewr-icon>
                </button>
                <button class="feedback-btn-icon" data-message-id="${messageId}" data-feedback="negative" title="Thumbs down" style="background: transparent; border: 1px solid #334155; border-radius: 4px; padding: 4px 8px; cursor: pointer; color: #94a3b8; transition: all 0.2s;">
                    <ewr-icon name="thumbs-down" size="16"></ewr-icon>
                </button>
            </div>
        `;

        contentEl.insertAdjacentHTML('beforeend', feedbackHtml);

        // Add click handlers
        const feedbackContainer = contentEl.querySelector('.feedback-buttons');
        feedbackContainer.querySelectorAll('.feedback-btn-icon').forEach(btn => {
            btn.addEventListener('click', (e) => {
                const feedback = btn.dataset.feedback;
                const msgId = btn.dataset.messageId;

                // Dispatch custom event for parent to handle
                this.dispatchEvent(new CustomEvent('feedback', {
                    bubbles: true,
                    detail: { messageId: msgId, feedback, buttonElement: btn }
                }));

                // Visual feedback
                if (feedback === 'positive') {
                    btn.style.background = 'rgba(34, 197, 94, 0.2)';
                    btn.style.borderColor = '#22c55e';
                    btn.style.color = '#22c55e';
                } else {
                    btn.style.background = 'rgba(239, 68, 68, 0.2)';
                    btn.style.borderColor = '#ef4444';
                    btn.style.color = '#ef4444';
                }
            });
        });
    }

    // Scroll to bottom of messages
    scrollToBottom() {
        const messagesEl = this.querySelector(`#${this._messagesId}`);
        if (messagesEl) {
            messagesEl.scrollTop = messagesEl.scrollHeight;
        }
    }

    // Clear all messages
    clearMessages() {
        const messagesEl = this.querySelector(`#${this._messagesId}`);
        if (messagesEl) {
            // Remove all messages but keep empty state
            const emptyChat = messagesEl.querySelector('.empty-chat');
            messagesEl.innerHTML = '';
            if (emptyChat) {
                emptyChat.style.display = '';
                messagesEl.appendChild(emptyChat);
            } else {
                messagesEl.innerHTML = `
                    <div class="empty-chat" id="emptyChat">
                        <div class="empty-icon">${this.getAttribute('empty-icon') || '💬'}</div>
                        <div class="empty-title">${this.getAttribute('empty-title') || 'Start a Conversation'}</div>
                        <div class="empty-text">${this.getAttribute('empty-text') || 'Ask questions in natural language.'}</div>
                    </div>
                `;
            }
        }
    }

    // Show timing metrics bar
    showTiming() {
        const timingEl = this.querySelector('.timing-metrics');
        if (timingEl) timingEl.classList.add('visible');
    }

    // Hide timing metrics bar
    hideTiming() {
        const timingEl = this.querySelector('.timing-metrics');
        if (timingEl) timingEl.classList.remove('visible');
    }

    // Update a timing metric
    updateTiming(stepId, value, status = '') {
        const stepEl = this.querySelector(`#${stepId}`);
        if (stepEl) {
            const valueEl = stepEl.querySelector('.timing-metric-value');
            if (valueEl) valueEl.textContent = value;

            // Update status classes
            stepEl.classList.remove('active', 'completed', 'waiting');
            if (status) stepEl.classList.add(status);
        }
    }

    // Reset all timing metrics
    resetTiming() {
        const timingEl = this.querySelector('.timing-metrics');
        if (timingEl) {
            timingEl.querySelectorAll('.timing-metric-value').forEach(el => {
                el.textContent = '--';
            });
            timingEl.querySelectorAll('.timing-metric').forEach(el => {
                el.classList.remove('active', 'completed', 'waiting');
            });
        }
    }

    // Show context limit alert
    showAlert(message) {
        const alertEl = this.querySelector('.context-limit-alert');
        if (alertEl) {
            alertEl.textContent = message || alertEl.textContent;
            alertEl.style.display = 'block';
        }
    }

    // Hide context limit alert
    hideAlert() {
        const alertEl = this.querySelector('.context-limit-alert');
        if (alertEl) alertEl.style.display = 'none';
    }

    // Disable/enable input
    setInputDisabled(disabled) {
        const inputEl = this.querySelector('textarea');
        const sendBtn = this.querySelector('.ewr-button-send');
        if (inputEl) inputEl.disabled = disabled;
        if (sendBtn) sendBtn.disabled = disabled;
    }

    // Get input value
    getInputValue() {
        const inputEl = this.querySelector('textarea');
        return inputEl ? inputEl.value : '';
    }

    // Set input value
    setInputValue(value) {
        const inputEl = this.querySelector('textarea');
        if (inputEl) inputEl.value = value;
    }

    // Clear input
    clearInput() {
        this.setInputValue('');
    }

    // Focus input
    focusInput() {
        const inputEl = this.querySelector('textarea');
        if (inputEl) inputEl.focus();
    }

    // Get messages container element
    getMessagesElement() {
        return this.querySelector(`#${this._messagesId}`);
    }
}

customElements.define('ewr-chat-window', EwrChatWindow);

/**
 * EwrAudioFileRow - Audio file row component for processing tables
 *
 * Attributes:
 * - file-id: Unique identifier for the file
 * - filename: Name of the audio file
 * - size: File size in MB
 * - status: Current processing status
 * - selected: Whether the file is selected (checkbox)
 * - expanded: Whether the row is expanded
 * - audio-src: Source URL for the audio player
 *
 * Events:
 * - selection-change: Fired when checkbox changes { fileId, selected }
 * - expand-toggle: Fired when row expansion toggles { fileId, expanded }
 * - delete-click: Fired when delete button clicked { fileId, filename }
 */
class EwrAudioFileRow extends HTMLElement {
    constructor() {
        super();
        this._statusLog = [];
        this._metrics = { elapsed: 0, gpu: null };
    }

    static get observedAttributes() {
        return ['status', 'selected', 'expanded'];
    }

    connectedCallback() {
        // Use display:contents so the <tr> children become direct children of <tbody> for table layout
        this.style.display = 'contents';
        this.render();
    }

    attributeChangedCallback(name, oldValue, newValue) {
        if (oldValue !== newValue && this.isConnected) {
            this.render();
        }
    }

    get fileId() { return this.getAttribute('file-id') || ''; }
    get filename() { return this.getAttribute('filename') || ''; }
    get size() { return this.getAttribute('size') || '0'; }
    get status() { return this.getAttribute('status') || 'Pending'; }
    get selected() { return this.getAttribute('selected') === 'true'; }
    get expanded() { return this.getAttribute('expanded') === 'true'; }
    get audioSrc() { return this.getAttribute('audio-src') || ''; }

    // Add a status log entry
    addStatusLog(time, message, type = 'info') {
        this._statusLog.push({ time, message, type });
        this._updateStatusDisplay();
    }

    // Update metrics
    setMetrics(elapsed, gpu) {
        this._metrics = { elapsed, gpu };
        this._updateMetricsDisplay();
    }

    // Clear status log
    clearStatusLog() {
        this._statusLog = [];
        this._metrics = { elapsed: 0, gpu: null };
        this._updateStatusDisplay();
        this._updateMetricsDisplay();
    }

    _getLogColor(stepType) {
        const colors = {
            'init': '#94a3b8', 'load': '#94a3b8', 'transcribe': '#60a5fa',
            'chunk': '#a78bfa', 'transcribe_done': '#34d399', 'diarize': '#fbbf24',
            'diarize_done': '#34d399', 'diarize_error': '#f87171', 'summary': '#22d3ee',
            'metadata': '#a78bfa', 'customer': '#34d399', 'content': '#60a5fa',
            'llm': '#60a5fa', 'llm_done': '#34d399', 'save': '#4ade80',
            'error': '#f87171', 'waiting': '#fbbf24', 'info': '#cbd5e1', 'cleanup': '#34d399'
        };
        return colors[stepType] || '#e2e8f0';
    }

    _escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    _updateStatusDisplay() {
        const logContainer = this.querySelector('.ewr-audio-status-log');
        if (!logContainer) return;

        logContainer.innerHTML = this._statusLog.length > 0
            ? this._statusLog.map(log =>
                `<div class="ewr-audio-log-entry" style="color: ${this._getLogColor(log.type)}; white-space: nowrap; line-height: 1.5; font-weight: 500;">${this._escapeHtml(log.time)} ${this._escapeHtml(log.message)}</div>`
            ).join('')
            : `<div style="color: #94a3b8; font-size: 14px;">${this._escapeHtml(this.status)}</div>`;

        // Scroll to bottom
        setTimeout(() => { logContainer.scrollTop = logContainer.scrollHeight; }, 10);
    }

    _updateMetricsDisplay() {
        let metricsContainer = this.querySelector('.ewr-audio-metrics');
        const statusCell = this.querySelector('.ewr-status-cell');

        if (this._metrics.elapsed || this._metrics.gpu) {
            const metricsHtml = `
                ${this._metrics.elapsed ? `<span style="color: #60a5fa;">⏱️ ${this._metrics.elapsed}s</span>` : ''}
                ${this._metrics.gpu ? `<span style="color: #10b981;">🎮 ${this._escapeHtml(this._metrics.gpu)}</span>` : ''}
            `;

            if (metricsContainer) {
                metricsContainer.innerHTML = metricsHtml;
            } else if (statusCell) {
                metricsContainer = document.createElement('div');
                metricsContainer.className = 'ewr-audio-metrics';
                metricsContainer.style.cssText = 'display: flex; gap: 16px; margin-top: 6px; font-size: 12px; padding: 4px 8px; background: rgba(59, 130, 246, 0.1); border-radius: 4px; border: 1px solid rgba(59, 130, 246, 0.2);';
                metricsContainer.innerHTML = metricsHtml;
                statusCell.appendChild(metricsContainer);
            }
        } else if (metricsContainer) {
            metricsContainer.remove();
        }
    }

    render() {
        const fileId = this.fileId;
        const filename = this.filename;
        const size = this.size;
        const selected = this.selected;
        const expanded = this.expanded;
        const audioSrc = this.audioSrc;

        // Build status log HTML
        const statusLogHtml = this._statusLog.length > 0
            ? this._statusLog.map(log =>
                `<div class="ewr-audio-log-entry" style="color: ${this._getLogColor(log.type)}; white-space: nowrap; line-height: 1.5; font-weight: 500;">${this._escapeHtml(log.time)} ${this._escapeHtml(log.message)}</div>`
            ).join('')
            : `<div style="color: #94a3b8; font-size: 14px;">${this._escapeHtml(this.status)}</div>`;

        // Build metrics HTML
        const metricsHtml = (this._metrics.elapsed || this._metrics.gpu) ? `
            <div class="ewr-audio-metrics" style="display: flex; gap: 16px; margin-top: 6px; font-size: 12px; padding: 4px 8px; background: rgba(59, 130, 246, 0.1); border-radius: 4px; border: 1px solid rgba(59, 130, 246, 0.2);">
                ${this._metrics.elapsed ? `<span style="color: #60a5fa;">⏱️ ${this._metrics.elapsed}s</span>` : ''}
                ${this._metrics.gpu ? `<span style="color: #10b981;">🎮 ${this._escapeHtml(this._metrics.gpu)}</span>` : ''}
            </div>
        ` : '';

        this.innerHTML = `
            <tr class="ewr-audio-row ${expanded ? 'expanded' : ''}" data-file-id="${fileId}">
                <td class="ewr-checkbox-cell">
                    <input type="checkbox" class="ewr-file-checkbox" ${selected ? 'checked' : ''}>
                </td>
                <td style="width: 30px;">
                    <span class="ewr-expand-indicator" style="cursor: pointer;">▶</span>
                </td>
                <td class="ewr-filename-cell" style="max-width: 340px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; cursor: pointer;" title="${this._escapeHtml(filename)}">${this._escapeHtml(filename)}</td>
                <td style="width: 70px;">${size}</td>
                <td class="ewr-status-cell" style="min-width: 350px; max-width: 450px;">
                    <div class="ewr-audio-status-log" style="max-height: 90px; overflow-y: auto; font-size: 14px; font-family: monospace; background: rgba(0,0,0,0.3); border-radius: 4px; padding: 6px 10px;">
                        ${statusLogHtml}
                    </div>
                    ${metricsHtml}
                </td>
                <td style="width: 60px;">
                    <button class="ewr-delete-file-button">Delete</button>
                </td>
            </tr>
            <tr class="ewr-expandable-row ${expanded ? 'expanded' : ''}" data-file-id="${fileId}-expanded">
                <td colspan="6">
                    <div class="ewr-expandable-row-content">
                        <div class="ewr-audio-player-card compact">
                            <div class="ewr-audio-player-card-metadata">
                                <div class="ewr-audio-player-card-metadata-item">
                                    <div class="ewr-audio-player-card-metadata-label">Filename</div>
                                    <div class="ewr-audio-player-card-metadata-value">${this._escapeHtml(filename)}</div>
                                </div>
                                <div class="ewr-audio-player-card-metadata-item">
                                    <div class="ewr-audio-player-card-metadata-label">Size</div>
                                    <div class="ewr-audio-player-card-metadata-value">${size} MB</div>
                                </div>
                            </div>
                            <audio controls preload="metadata" src="${audioSrc}" style="width: 100%; margin-top: 12px;"></audio>
                        </div>
                    </div>
                </td>
            </tr>
        `;

        // Set up event listeners
        this._setupEventListeners();
    }

    _setupEventListeners() {
        // Checkbox change
        const checkbox = this.querySelector('.ewr-file-checkbox');
        if (checkbox) {
            checkbox.addEventListener('change', (e) => {
                this.setAttribute('selected', e.target.checked ? 'true' : 'false');
                this.dispatchEvent(new CustomEvent('selection-change', {
                    detail: { fileId: this.fileId, selected: e.target.checked },
                    bubbles: true
                }));
            });
        }

        // Expand indicator click
        const expandIndicator = this.querySelector('.ewr-expand-indicator');
        const filenameCell = this.querySelector('.ewr-filename-cell');
        [expandIndicator, filenameCell].forEach(el => {
            if (el) {
                el.addEventListener('click', () => {
                    const newExpanded = this.expanded ? 'false' : 'true';
                    this.setAttribute('expanded', newExpanded);
                    this.dispatchEvent(new CustomEvent('expand-toggle', {
                        detail: { fileId: this.fileId, expanded: newExpanded === 'true' },
                        bubbles: true
                    }));
                });
            }
        });

        // Delete button click
        const deleteBtn = this.querySelector('.ewr-delete-file-button');
        if (deleteBtn) {
            deleteBtn.addEventListener('click', () => {
                this.dispatchEvent(new CustomEvent('delete-click', {
                    detail: { fileId: this.fileId, filename: this.filename },
                    bubbles: true
                }));
            });
        }
    }
}

customElements.define('ewr-audio-file-row', EwrAudioFileRow);


/**
 * EWR-AudioUpload Component
 * File upload section with drag & drop and directory polling toggle
 *
 * Usage:
 * <ewr-audio-upload
 *     title="Upload Files"
 *     drop-zone-id="dropZone"
 *     file-input-id="fileInput"
 *     directory-input-id="directoryPath"
 *     poll-btn-id="pollBtn">
 * </ewr-audio-upload>
 */
class EwrAudioUpload extends HTMLElement {
    constructor() {
        super();
    }

    connectedCallback() {
        const title = this.getAttribute('title') || 'Upload Files';
        const dropZoneId = this.getAttribute('drop-zone-id') || 'dropZone';
        const fileInputId = this.getAttribute('file-input-id') || 'fileInput';
        const directoryInputId = this.getAttribute('directory-input-id') || 'directoryPath';
        const pollBtnId = this.getAttribute('poll-btn-id') || 'pollBtn';
        const acceptedFormats = this.getAttribute('accepted-formats') || '.wav,.mp3,.m4a';
        const formatDescription = this.getAttribute('format-description') || 'WAV, MP3, M4A supported';
        const uniqueId = `upload-${Date.now()}`;

        this.innerHTML = `
            <div class="ewr-audio-upload-section">
                <div class="ewr-audio-upload-header">
                    <h3 class="ewr-audio-upload-title">${title}</h3>
                </div>
                <div class="ewr-audio-upload-content">
                    <!-- Method Toggle Radio Buttons -->
                    <div class="ewr-audio-method-toggle">
                        <label class="ewr-audio-method-option">
                            <input type="radio" name="${uniqueId}-method" value="dragdrop" checked>
                            <span class="ewr-audio-method-label">Drag & Drop</span>
                        </label>
                        <label class="ewr-audio-method-option">
                            <input type="radio" name="${uniqueId}-method" value="poll">
                            <span class="ewr-audio-method-label">Poll Directory</span>
                        </label>
                    </div>

                    <!-- Method Content Area -->
                    <div class="ewr-audio-method-content">
                        <!-- Drag & Drop Method -->
                        <div class="ewr-audio-method-panel" data-method="dragdrop">
                            <div class="ewr-audio-drop-zone" id="${dropZoneId}">
                                <span class="ewr-audio-drop-zone-icon">
                                    <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                                        <path d="M9 18V5l12-2v13"/>
                                        <circle cx="6" cy="18" r="3"/>
                                        <circle cx="18" cy="16" r="3"/>
                                    </svg>
                                </span>
                                <div>
                                    <div class="ewr-audio-drop-zone-text">Drop audio files or click to browse</div>
                                    <div class="ewr-audio-drop-zone-subtext">${formatDescription}</div>
                                </div>
                            </div>
                        </div>

                        <!-- Poll Directory Method -->
                        <div class="ewr-audio-method-panel" data-method="poll" style="display: none;">
                            <div class="ewr-audio-directory-row">
                                <input type="text" id="${directoryInputId}" class="ewr-audio-directory-input" placeholder="Enter directory path...">
                                <button class="ewr-button ewr-button-small ewr-button-secondary" id="${pollBtnId}">Poll</button>
                            </div>
                        </div>
                    </div>

                    <input type="file" id="${fileInputId}" accept="${acceptedFormats}" multiple style="display: none;">
                </div>
            </div>
        `;

        // Setup radio toggle behavior
        this._setupMethodToggle(uniqueId);
    }

    _setupMethodToggle(uniqueId) {
        const radios = this.querySelectorAll(`input[name="${uniqueId}-method"]`);
        const panels = this.querySelectorAll('.ewr-audio-method-panel');

        radios.forEach(radio => {
            radio.addEventListener('change', (e) => {
                const selectedMethod = e.target.value;
                panels.forEach(panel => {
                    panel.style.display = panel.dataset.method === selectedMethod ? 'block' : 'none';
                });
            });
        });
    }
}

customElements.define('ewr-audio-upload', EwrAudioUpload);


/**
 * EWR-AudioFileList Component
 * A file list section for displaying audio files (unanalyzed or pending)
 *
 * Usage:
 * <ewr-audio-file-list
 *     title="Unanalyzed Files"
 *     list-type="unanalyzed"
 *     table-id="unanalyzedTable"
 *     body-id="unanalyzedBody"
 *     count-id="unanalyzedCount"
 *     section-id="unanalyzedSection"
 *     empty-icon="folder"
 *     empty-text="No files added"
 *     show-checkbox="true"
 *     show-process-btn="true"
 *     process-btn-id="processSelectedBtn">
 * </ewr-audio-file-list>
 */
class EwrAudioFileList extends HTMLElement {
    constructor() {
        super();
    }

    connectedCallback() {
        const title = this.getAttribute('title') || 'Files';
        const listType = this.getAttribute('list-type') || 'unanalyzed';
        const tableId = this.getAttribute('table-id') || `${listType}Table`;
        const bodyId = this.getAttribute('body-id') || `${listType}Body`;
        const countId = this.getAttribute('count-id') || `${listType}Count`;
        const sectionId = this.getAttribute('section-id') || `${listType}Section`;
        const emptyIcon = this.getAttribute('empty-icon') || 'folder';
        const emptyText = this.getAttribute('empty-text') || 'No files';
        const showCheckbox = this.getAttribute('show-checkbox') !== 'false';
        const showProcessBtn = this.getAttribute('show-process-btn') === 'true';
        const processBtnId = this.getAttribute('process-btn-id') || 'processSelectedBtn';
        const showRefreshBtn = this.getAttribute('show-refresh-btn') === 'true';
        const refreshBtnId = this.getAttribute('refresh-btn-id') || 'refreshBtn';
        const refreshHandler = this.getAttribute('refresh-handler') || '';

        // Icon SVG map
        const icons = {
            'folder': '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M22 19a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h5l2 3h9a2 2 0 0 1 2 2z"/></svg>',
            'check': '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="20 6 9 17 4 12"/></svg>',
            'clock': '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/></svg>'
        };

        const iconSvg = icons[emptyIcon] || icons['folder'];

        // Build table headers based on list type
        let tableHeaders = '';
        if (listType === 'unanalyzed') {
            tableHeaders = `
                <tr>
                    ${showCheckbox ? '<th style="width: 30px;"><input type="checkbox" class="ewr-file-checkbox" id="selectAll' + listType.charAt(0).toUpperCase() + listType.slice(1) + '" checked></th>' : ''}
                    <th style="width: 24px;"></th>
                    <th>Filename</th>
                    <th style="width: 50px;">Size</th>
                    <th style="min-width: 200px;">Status</th>
                    <th style="width: 50px;"></th>
                </tr>
            `;
        } else {
            // pending type
            tableHeaders = `
                <tr>
                    <th style="width: 24px;"></th>
                    <th>Filename</th>
                    <th style="width: 60px;">Duration</th>
                    <th style="width: 70px;">Mood</th>
                    <th style="width: 60px;">Status</th>
                    <th style="width: 60px;"></th>
                </tr>
            `;
        }

        // Build footer buttons
        let footerButtons = '';
        if (showProcessBtn) {
            footerButtons += `<button class="ewr-button ewr-button-small ewr-button-primary" id="${processBtnId}" disabled>Process Selected</button>`;
        }
        if (showRefreshBtn) {
            footerButtons += `<button class="ewr-button ewr-button-small ewr-button-secondary" id="${refreshBtnId}" ${refreshHandler ? `onclick="${refreshHandler}"` : ''}>Refresh</button>`;
        }

        this.innerHTML = `
            <div class="ewr-audio-file-section" id="${sectionId}">
                <div class="ewr-audio-file-header">
                    <h3 class="ewr-audio-file-title">${title}</h3>
                    <span class="ewr-audio-file-count" id="${countId}">0 files</span>
                </div>
                <div class="ewr-audio-file-table-container">
                    <table class="ewr-audio-file-table" id="${tableId}">
                        <thead>
                            ${tableHeaders}
                        </thead>
                        <tbody id="${bodyId}">
                            <tr>
                                <td colspan="6">
                                    <div class="ewr-audio-empty-state">
                                        <div class="ewr-audio-empty-state-icon">${iconSvg}</div>
                                        <div class="ewr-audio-empty-state-text">${emptyText}</div>
                                    </div>
                                </td>
                            </tr>
                        </tbody>
                    </table>
                </div>
                <div class="ewr-audio-file-footer">
                    ${footerButtons}
                </div>
            </div>
        `;
    }
}

customElements.define('ewr-audio-file-list', EwrAudioFileList);
