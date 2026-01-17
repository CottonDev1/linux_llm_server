/**
 * EWR Container Components
 * Reusable container/panel components for EWR applications
 * These components hold and organize multiple UI elements
 */


/**
 * EWR-Filter-Panel Component
 * A flexible, collapsible filter panel with configurable columns and optional footer sections
 *
 * Usage:
 * <ewr-filter-panel
 *     id="connectionFilters"
 *     title="Connection Settings"
 *     columns="4"
 *     collapsed="false"
 *     show-status="true"
 *     show-actions="true">
 *
 *     <!-- Filter items (use ewr-filter-item for single inputs or custom divs for stacked) -->
 *     <ewr-filter-item label="Server" type="input" id="server" value="NCSQLTEST"></ewr-filter-item>
 *     <ewr-filter-item label="Database" type="select" id="database"></ewr-filter-item>
 *
 *     <!-- Stacked inputs example -->
 *     <div class="ewr-filter-stack">
 *         <ewr-filter-item label="Server" type="input" id="server"></ewr-filter-item>
 *         <ewr-filter-item label="Database" type="select" id="database"></ewr-filter-item>
 *     </div>
 *
 *     <!-- Status slot -->
 *     <div slot="status">
 *         <span class="ewr-status-pill success">Connected</span>
 *     </div>
 *
 *     <!-- Actions slot -->
 *     <div slot="actions">
 *         <button class="ewr-button-action-green-gradient">Connect</button>
 *         <button class="ewr-button-clear">Clear</button>
 *     </div>
 * </ewr-filter-panel>
 *
 * Attributes:
 * - id: Panel ID for toggle functionality
 * - title: Panel header title
 * - columns: Number of columns (default: 4, responsive)
 * - collapsed: Start collapsed (default: false)
 * - show-status: Show status footer section (default: false)
 * - show-actions: Show actions footer section (default: false)
 * - no-margin: Remove top margin from panel (add class "no-margin")
 *
 * Slots:
 * - default: Filter items/content
 * - status: Status labels section content
 * - actions: Action buttons section content
 *
 * CSS Variables (customizable):
 * - --filter-panel-columns: Override column count
 * - --filter-panel-gap: Gap between items (default: 16px)
 */
class EwrFilterPanel extends HTMLElement {
    constructor() {
        super();
    }

    connectedCallback() {
        const id = this.getAttribute('id') || `filter-panel-${Date.now()}`;
        const title = this.getAttribute('title') || 'Filters';
        const columns = this.getAttribute('columns') || '4';
        const collapsed = this.getAttribute('collapsed') === 'true';
        const showStatus = this.getAttribute('show-status') === 'true';
        const showActions = this.getAttribute('show-actions') === 'true';
        const noMargin = this.hasAttribute('no-margin');

        // Remove ID from custom element to avoid duplicate IDs
        // The inner div will have this ID for toggleCollapsible to work
        this.removeAttribute('id');

        // Capture slotted content before replacing innerHTML
        const statusSlot = this.querySelector('[slot="status"]');
        const actionsSlot = this.querySelector('[slot="actions"]');

        // Capture all non-slot children as filter content
        const filterContent = Array.from(this.children).filter(child => {
            return !child.hasAttribute('slot');
        });

        // Build the component HTML
        const collapsedClass = collapsed ? ' collapsed' : '';
        const noMarginClass = noMargin ? ' no-margin' : '';

        this.innerHTML = `
            <div class="ewr-collapsible${collapsedClass}${noMarginClass}" id="${id}">
                <div class="ewr-collapsible-header" onclick="toggleCollapsible('${id}')">
                    <span class="ewr-collapsible-title">${title}</span>
                    <button class="ewr-collapse-toggle" type="button">
                        <span class="toggle-icon">▼</span>
                    </button>
                </div>
                <div class="ewr-collapsible-content">
                    <div class="ewr-filter-panel-grid" style="--filter-columns: ${columns};">
                        <!-- Filter content will be inserted here -->
                    </div>
                </div>
                <div class="ewr-collapsible-footer ewr-status-footer" id="${id}-status" style="display: ${showStatus ? 'flex' : 'none'};">
                    <!-- Status content will be inserted here -->
                </div>
                <div class="ewr-collapsible-footer" id="${id}-actions" style="display: ${showActions ? 'flex' : 'none'};">
                    <!-- Actions content will be inserted here -->
                </div>
            </div>
        `;

        // Move filter content into the grid
        const grid = this.querySelector('.ewr-filter-panel-grid');
        filterContent.forEach(child => {
            grid.appendChild(child);
        });

        // Move status slot content
        if (statusSlot) {
            const statusContainer = this.querySelector(`#${id}-status`);
            Array.from(statusSlot.children).forEach(child => {
                statusContainer.appendChild(child);
            });
        }

        // Move actions slot content
        if (actionsSlot) {
            const actionsContainer = this.querySelector(`#${id}-actions`);
            Array.from(actionsSlot.children).forEach(child => {
                actionsContainer.appendChild(child);
            });
        }

        // Add CSS for the grid if not already present
        this._injectStyles();
    }

    _injectStyles() {
        if (document.getElementById('ewr-filter-panel-styles')) return;

        const style = document.createElement('style');
        style.id = 'ewr-filter-panel-styles';
        style.textContent = `
            .ewr-filter-panel-grid {
                display: flex;
                flex-wrap: wrap;
                gap: var(--filter-panel-gap, 20px);
                padding: 16px 20px;
                justify-content: flex-start;
                align-items: flex-end;
            }

            .ewr-filter-panel-grid > * {
                flex: 0 1 auto;
                min-width: 150px;
            }

            /* Stack container for vertically stacked inputs */
            .ewr-filter-stack {
                display: flex;
                flex-direction: column;
                gap: 12px;
            }

            .ewr-filter-stack > * {
                flex: none;
                min-width: unset;
            }

            /* Checkbox options container - vertical stack */
            .ewr-filter-options {
                display: flex;
                flex-direction: column;
                gap: 8px;
                margin-top: 4px;
            }

            /* Checkbox options container - horizontal row */
            .ewr-filter-options-horizontal {
                display: flex;
                flex-direction: row;
                flex-wrap: wrap;
                gap: 16px 24px;
                align-items: center;
            }

            .ewr-filter-options label,
            .ewr-filter-options-horizontal label {
                display: flex;
                align-items: center;
                gap: 8px;
                color: var(--text-secondary, #b8d4f0);
                font-size: 13px;
                font-weight: 500;
                cursor: pointer;
            }

            .ewr-filter-options input[type="checkbox"],
            .ewr-filter-options-horizontal input[type="checkbox"] {
                width: 16px;
                height: 16px;
                min-width: 16px;
                cursor: pointer;
                accent-color: var(--accent-primary, #5a9fe6);
            }

            /* Span full width for checkbox groups */
            .ewr-filter-span-full {
                flex-basis: 100% !important;
                max-width: none !important;
            }

            /* Span 2 columns for checkbox groups in grid layouts */
            .ewr-filter-span-2 {
                flex-basis: calc(50% - var(--filter-panel-gap, 16px)) !important;
                max-width: none !important;
            }

            /* Responsive adjustments */
            @media (max-width: 1200px) {
                .ewr-filter-panel-grid > * {
                    flex: 1 1 calc(100% / 3 - var(--filter-panel-gap, 16px));
                }
            }

            @media (max-width: 900px) {
                .ewr-filter-panel-grid > * {
                    flex: 1 1 calc(100% / 2 - var(--filter-panel-gap, 16px));
                }
            }

            @media (max-width: 600px) {
                .ewr-filter-panel-grid > * {
                    flex: 1 1 100%;
                }
            }
        `;
        document.head.appendChild(style);
    }

    // Public API
    toggle() {
        const panel = this.querySelector('.ewr-collapsible');
        if (panel) panel.classList.toggle('collapsed');
    }

    expand() {
        const panel = this.querySelector('.ewr-collapsible');
        if (panel) panel.classList.remove('collapsed');
    }

    collapse() {
        const panel = this.querySelector('.ewr-collapsible');
        if (panel) panel.classList.add('collapsed');
    }

    get isCollapsed() {
        const panel = this.querySelector('.ewr-collapsible');
        return panel ? panel.classList.contains('collapsed') : false;
    }

    // Show/hide status section
    showStatus(show = true) {
        const id = this.getAttribute('id') || this.querySelector('.ewr-collapsible')?.id;
        const statusEl = this.querySelector(`#${id}-status`);
        if (statusEl) statusEl.style.display = show ? 'flex' : 'none';
    }

    // Show/hide actions section
    showActions(show = true) {
        const id = this.getAttribute('id') || this.querySelector('.ewr-collapsible')?.id;
        const actionsEl = this.querySelector(`#${id}-actions`);
        if (actionsEl) actionsEl.style.display = show ? 'flex' : 'none';
    }

    // Set status content
    setStatusContent(html) {
        const id = this.getAttribute('id') || this.querySelector('.ewr-collapsible')?.id;
        const statusEl = this.querySelector(`#${id}-status`);
        if (statusEl) statusEl.innerHTML = html;
    }

    // Set actions content
    setActionsContent(html) {
        const id = this.getAttribute('id') || this.querySelector('.ewr-collapsible')?.id;
        const actionsEl = this.querySelector(`#${id}-actions`);
        if (actionsEl) actionsEl.innerHTML = html;
    }

    // Update title
    setTitle(title) {
        const titleEl = this.querySelector('.ewr-collapsible-title');
        if (titleEl) titleEl.textContent = title;
    }
}

customElements.define('ewr-filter-panel', EwrFilterPanel);


/**
 * EWR-Filter-Item Component
 * A single filter input with label for use inside ewr-filter-panel
 *
 * Usage:
 * <ewr-filter-item
 *     label="Server"
 *     type="input"
 *     id="server"
 *     value="NCSQLTEST"
 *     placeholder="Enter server name"
 *     onchange="handleChange()">
 * </ewr-filter-item>
 *
 * <ewr-filter-item
 *     label="Database"
 *     type="select"
 *     id="database"
 *     disabled>
 *     <option value="">Select database...</option>
 * </ewr-filter-item>
 *
 * Attributes:
 * - label: Label text
 * - type: "input", "select", "textarea", "password", "checkbox"
 * - id: Element ID
 * - value: Initial value
 * - placeholder: Placeholder text
 * - disabled: Disabled state
 * - required: Required field
 * - onchange: Change handler
 * - oninput: Input handler
 * - tabindex: Tab order
 */
class EwrFilterItem extends HTMLElement {
    constructor() {
        super();
    }

    connectedCallback() {
        const label = this.getAttribute('label') || '';
        const type = this.getAttribute('type') || 'input';
        const id = this.getAttribute('id') || `filter-${Date.now()}`;
        const value = this.getAttribute('value') || '';
        const placeholder = this.getAttribute('placeholder') || '';
        const disabled = this.hasAttribute('disabled');
        const required = this.hasAttribute('required');
        const onchange = this.getAttribute('onchange') || '';
        const oninput = this.getAttribute('oninput') || '';
        const tabindex = this.getAttribute('tabindex') || '';

        // Capture child options for select type
        // Use <template> to preserve options (browsers strip orphan <option> elements)
        const template = this.querySelector('template');
        let options = '';
        if (template) {
            // Get options from template content
            options = template.innerHTML.trim();
        } else {
            // Fallback: try to get direct option children (may not work in all browsers)
            options = Array.from(this.querySelectorAll('option')).map(opt => opt.outerHTML).join('');
        }

        let inputHtml = '';
        const disabledAttr = disabled ? ' disabled' : '';
        const requiredAttr = required ? ' required' : '';
        const tabindexAttr = tabindex ? ` tabindex="${tabindex}"` : '';
        const onchangeAttr = onchange ? ` onchange="${onchange}"` : '';
        const oninputAttr = oninput ? ` oninput="${oninput}"` : '';

        switch (type) {
            case 'select':
                inputHtml = `
                    <select id="${id}" class="ewr-select-edit"${disabledAttr}${requiredAttr}${tabindexAttr}${onchangeAttr}>
                        ${options || `<option value="">${placeholder || 'Select...'}</option>`}
                    </select>
                `;
                break;

            case 'textarea':
                inputHtml = `
                    <textarea id="${id}" class="ewr-input-edits" placeholder="${placeholder}"${disabledAttr}${requiredAttr}${tabindexAttr}${onchangeAttr}${oninputAttr}>${value}</textarea>
                `;
                break;

            case 'password':
                inputHtml = `
                    <input type="password" id="${id}" class="ewr-input-edits" value="${value}" placeholder="${placeholder}"${disabledAttr}${requiredAttr}${tabindexAttr}${onchangeAttr}${oninputAttr}>
                `;
                break;

            case 'checkbox':
                inputHtml = `
                    <div class="ewr-checkbox-div">
                        <input type="checkbox" id="${id}"${disabled ? ' disabled' : ''}${value === 'true' ? ' checked' : ''}${tabindexAttr}${onchangeAttr}>
                        <label for="${id}">${label}</label>
                    </div>
                `;
                // For checkbox, we don't want the outer label
                this.innerHTML = `
                    <div class="ewr-filter-item ewr-filter-item-checkbox">
                        ${inputHtml}
                    </div>
                `;
                return;

            case 'input':
            default:
                inputHtml = `
                    <input type="text" id="${id}" class="ewr-input-edits" value="${value}" placeholder="${placeholder}"${disabledAttr}${requiredAttr}${tabindexAttr}${onchangeAttr}${oninputAttr}>
                `;
                break;
        }

        this.innerHTML = `
            <div class="ewr-filter-item ewr-edit-div-vert">
                ${label ? `<label class="ewr-label" for="${id}">${label}</label>` : ''}
                ${inputHtml}
            </div>
        `;
    }

    // Get the input/select element
    getInputElement() {
        return this.querySelector('input, select, textarea');
    }

    // Get/set value
    get value() {
        const input = this.getInputElement();
        if (!input) return '';
        if (input.type === 'checkbox') return input.checked;
        return input.value;
    }

    set value(val) {
        const input = this.getInputElement();
        if (!input) return;
        if (input.type === 'checkbox') {
            input.checked = val;
        } else {
            input.value = val;
        }
    }

    // Enable/disable
    get disabled() {
        const input = this.getInputElement();
        return input ? input.disabled : false;
    }

    set disabled(val) {
        const input = this.getInputElement();
        if (input) input.disabled = val;
    }

    // For select: set options
    setOptions(options, defaultText = '') {
        const select = this.querySelector('select');
        if (!select) return;

        let html = defaultText ? `<option value="">${defaultText}</option>` : '';
        options.forEach(opt => {
            const value = opt.value !== undefined ? opt.value : (opt.id || opt.name || opt);
            const text = opt.text || opt.name || opt.id || opt;
            html += `<option value="${value}">${text}</option>`;
        });
        select.innerHTML = html;
    }
}

customElements.define('ewr-filter-item', EwrFilterItem);


/**
 * EWR-Chat-Panel Component
 * A complete chat interface panel with messages, input, and optional metrics
 *
 * Usage:
 * <ewr-chat-panel
 *     id="supportChat"
 *     placeholder="Ask a question..."
 *     welcome-message="How can I assist you?"
 *     on-send="sendMessage"
 *     on-clear="clearChat"
 *     show-context="true"
 *     show-timing="false">
 * </ewr-chat-panel>
 *
 * Attributes:
 * - id: Panel ID
 * - placeholder: Input placeholder text
 * - welcome-message: Initial assistant message
 * - on-send: Send handler function name
 * - on-clear: Clear handler function name
 * - on-keydown: Keydown handler function name
 * - show-context: Show context/token metrics bar
 * - show-timing: Show pipeline timing metrics
 */
class EwrChatPanel extends HTMLElement {
    constructor() {
        super();
    }

    connectedCallback() {
        const id = this.getAttribute('id') || 'chatPanel';
        const placeholder = this.getAttribute('placeholder') || 'Ask a question...';
        const welcomeMessage = this.getAttribute('welcome-message') || 'How can I assist you?';
        const onSend = this.getAttribute('on-send') || 'sendMessage';
        const onClear = this.getAttribute('on-clear') || 'clearChat';
        const onKeydown = this.getAttribute('on-keydown') || 'handleChatKeydown';
        const showContext = this.getAttribute('show-context') !== 'false';
        const showTiming = this.getAttribute('show-timing') === 'true';

        // Capture any custom content (like timing metrics)
        const customTimingSlot = this.querySelector('[slot="timing"]');

        this.innerHTML = `
            <div class="card-body">
                <div class="ewr-chat-container">
                    <!-- Chat Messages Area -->
                    <div class="ewr-chat-window" id="${id}Messages">
                        <div class="ewr-chat-message-assistant">
                            <div class="ewr-message-avatar assistant">EWR</div>
                            <div class="ewr-message-content">${welcomeMessage}</div>
                        </div>
                    </div>

                    <!-- Context Window & Token Display -->
                    <div class="context-remaining" id="${id}Context" style="display: ${showContext ? 'flex' : 'none'};">
                        <span class="context-remaining-label">Context:</span>
                        <span class="context-remaining-value" id="contextUsedValue">0</span>
                        <span class="context-remaining-label">/</span>
                        <span class="context-remaining-value" id="contextMaxValue">--</span>
                        <span class="context-remaining-label" style="margin-left: 16px;">Prompt:</span>
                        <span class="context-remaining-value" id="promptTokens">0</span>
                        <span class="context-remaining-label" style="margin-left: 16px;">Completion:</span>
                        <span class="context-remaining-value" id="completionTokens">0</span>
                        <span class="context-remaining-label" style="margin-left: 16px;">Total:</span>
                        <span class="context-remaining-value" id="totalTokens">0</span>
                    </div>

                    <!-- Pipeline Timing Metrics (optional) -->
                    <div class="ewr-chat-timing" id="${id}Timing" style="display: ${showTiming ? 'block' : 'none'};"></div>

                    <!-- Chat Footer -->
                    <div class="ewr-chat-footer">
                        <div id="contextLimitAlert" class="context-limit-alert" style="display: none;">
                            Context window full - click Clear Chat to continue
                        </div>
                        <div class="ewr-chat-input-row">
                            <textarea
                                class="ewr-chat-entry"
                                id="${id}Input"
                                placeholder="${placeholder}"
                                rows="1"
                                onkeydown="${onKeydown}(event)"
                            ></textarea>
                            <div class="ewr-chat-buttons">
                                <button class="ewr-button-send" id="${id}SendBtn" onclick="${onSend}()" title="Send">&#10148;</button>
                                <button class="ewr-button-clear" onclick="${onClear}()">Clear</button>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        `;

        // Move custom timing content if provided
        if (customTimingSlot) {
            const timingContainer = this.querySelector(`#${id}Timing`);
            timingContainer.style.display = 'block';
            Array.from(customTimingSlot.children).forEach(child => {
                timingContainer.appendChild(child);
            });
        }
    }

    // Get messages container
    getMessagesElement() {
        const id = this.getAttribute('id') || 'chatPanel';
        return this.querySelector(`#${id}Messages`);
    }

    // Get input element
    getInputElement() {
        const id = this.getAttribute('id') || 'chatPanel';
        return this.querySelector(`#${id}Input`);
    }

    // Get send button
    getSendButton() {
        const id = this.getAttribute('id') || 'chatPanel';
        return this.querySelector(`#${id}SendBtn`);
    }

    // Add a message to the chat
    addMessage(content, type = 'assistant', avatar = 'EWR') {
        const messages = this.getMessagesElement();
        if (!messages) return null;

        const messageDiv = document.createElement('div');
        messageDiv.className = `ewr-chat-message-${type}`;
        messageDiv.innerHTML = `
            <div class="ewr-message-avatar ${type}">${avatar}</div>
            <div class="ewr-message-content">${content}</div>
        `;
        messages.appendChild(messageDiv);
        messages.scrollTop = messages.scrollHeight;
        return messageDiv;
    }

    // Clear all messages (keeping welcome message optional)
    clearMessages(keepWelcome = true) {
        const messages = this.getMessagesElement();
        if (!messages) return;

        if (keepWelcome) {
            const welcomeMessage = this.getAttribute('welcome-message') || 'How can I assist you?';
            messages.innerHTML = `
                <div class="ewr-chat-message-assistant">
                    <div class="ewr-message-avatar assistant">EWR</div>
                    <div class="ewr-message-content">${welcomeMessage}</div>
                </div>
            `;
        } else {
            messages.innerHTML = '';
        }
    }

    // Show/hide context limit alert
    showContextAlert(show = true) {
        const alert = this.querySelector('#contextLimitAlert');
        if (alert) alert.style.display = show ? 'block' : 'none';
    }

    // Update context metrics
    updateContext(used, max) {
        const usedEl = this.querySelector('#contextUsedValue');
        const maxEl = this.querySelector('#contextMaxValue');
        if (usedEl) usedEl.textContent = used;
        if (maxEl) maxEl.textContent = max;
    }

    // Update token metrics
    updateTokens(prompt, completion, total) {
        const promptEl = this.querySelector('#promptTokens');
        const completionEl = this.querySelector('#completionTokens');
        const totalEl = this.querySelector('#totalTokens');
        if (promptEl) promptEl.textContent = prompt;
        if (completionEl) completionEl.textContent = completion;
        if (totalEl) totalEl.textContent = total;
    }

    // Disable/enable send button
    setSendEnabled(enabled) {
        const btn = this.getSendButton();
        if (btn) btn.disabled = !enabled;
    }
}

customElements.define('ewr-chat-panel', EwrChatPanel);


/**
 * EWR-Timing-Metrics Component
 * A pipeline timing display showing steps with timing values and separators
 *
 * Usage:
 * <ewr-timing-metrics
 *     id="timingMetrics"
 *     steps="preprocessing:Analyze,security:Security,rules:Rules,schema:Schema,generating:Generate,fixing:Fix,executing:Execute">
 * </ewr-timing-metrics>
 *
 * Attributes:
 * - id: Component ID
 * - steps: Comma-separated list of "id:label" pairs for each step
 * - separator: Custom separator character (default: "›")
 * - show-total: Show total timing (default: true)
 *
 * Methods:
 * - updateStep(stepId, value, status) - Update a step's timing value and optional status
 * - reset() - Reset all timings to "--"
 * - setStepStatus(stepId, status) - Set step status: 'pending', 'active', 'complete', 'error'
 */
class EwrTimingMetrics extends HTMLElement {
    constructor() {
        super();
        this._steps = [];
    }

    connectedCallback() {
        const id = this.getAttribute('id') || 'timingMetrics';
        const stepsAttr = this.getAttribute('steps') || '';
        const separator = this.getAttribute('separator') || '›';
        const showTotal = this.getAttribute('show-total') !== 'false';

        // Parse steps from attribute
        this._steps = stepsAttr.split(',').filter(s => s.trim()).map(s => {
            const [stepId, label] = s.split(':').map(p => p.trim());
            return { id: stepId, label: label || stepId };
        });

        // Build the HTML
        let html = `<div class="ewr-timing-metrics" id="${id}">`;

        this._steps.forEach((step, index) => {
            html += `
                <div class="ewr-timing-metric" id="step-${step.id}">
                    <span class="ewr-timing-metric-label">${step.label}</span>
                    <span class="ewr-timing-metric-value">--</span>
                </div>
            `;
            if (index < this._steps.length - 1) {
                html += `<span class="ewr-timing-separator">${separator}</span>`;
            }
        });

        if (showTotal) {
            html += `
                <div class="ewr-timing-metric ewr-timing-total" id="step-total">
                    <span class="ewr-timing-metric-label">Total:</span>
                    <span class="ewr-timing-metric-value" id="timingTotal">--</span>
                </div>
            `;
        }

        html += '</div>';
        this.innerHTML = html;

        // Inject styles
        this._injectStyles();
    }

    _injectStyles() {
        if (document.getElementById('ewr-timing-metrics-styles')) return;

        const style = document.createElement('style');
        style.id = 'ewr-timing-metrics-styles';
        style.textContent = `
            .ewr-timing-metrics {
                display: flex;
                align-items: center;
                gap: 8px;
                padding: 8px 16px;
                background: var(--bg-tertiary, #1e3a5f);
                border-radius: 8px;
                flex-wrap: wrap;
            }

            .ewr-timing-metric {
                display: flex;
                flex-direction: column;
                align-items: center;
                gap: 2px;
                min-width: 50px;
                padding: 4px 8px;
                border-radius: 6px;
                transition: all 0.2s ease;
            }

            .ewr-timing-metric-label {
                font-size: 10px;
                font-weight: 600;
                color: var(--text-muted, #6b8cae);
                text-transform: uppercase;
                letter-spacing: 0.5px;
            }

            .ewr-timing-metric-value {
                font-size: 12px;
                font-weight: 600;
                color: var(--text-secondary, #b8d4f0);
                font-family: 'JetBrains Mono', monospace;
            }

            .ewr-timing-separator {
                color: var(--text-muted, #6b8cae);
                font-size: 14px;
                font-weight: 600;
                opacity: 0.5;
            }

            .ewr-timing-total {
                margin-left: auto;
                background: var(--bg-secondary, #142338);
                border: 1px solid var(--border-primary, #1e3a5f);
            }

            .ewr-timing-total .ewr-timing-metric-label {
                color: var(--accent-primary, #5a9fe6);
            }

            .ewr-timing-total .ewr-timing-metric-value {
                color: var(--accent-primary, #5a9fe6);
                font-size: 14px;
            }

            /* Step status states */
            .ewr-timing-metric.pending {
                opacity: 0.5;
            }

            .ewr-timing-metric.active {
                background: rgba(90, 159, 230, 0.15);
                border: 1px solid var(--accent-primary, #5a9fe6);
            }

            .ewr-timing-metric.active .ewr-timing-metric-label {
                color: var(--accent-primary, #5a9fe6);
            }

            .ewr-timing-metric.complete {
                background: rgba(16, 185, 129, 0.1);
            }

            .ewr-timing-metric.complete .ewr-timing-metric-value {
                color: var(--success, #10b981);
            }

            .ewr-timing-metric.error {
                background: rgba(239, 68, 68, 0.1);
            }

            .ewr-timing-metric.error .ewr-timing-metric-value {
                color: var(--error, #ef4444);
            }

            .ewr-timing-metric.skipped {
                opacity: 0.4;
            }

            .ewr-timing-metric.skipped .ewr-timing-metric-value {
                color: var(--text-muted, #6b8cae);
            }
        `;
        document.head.appendChild(style);
    }

    // Update a step's timing value
    updateStep(stepId, value, status = null) {
        const step = this.querySelector(`#step-${stepId}`);
        if (!step) return;

        const valueEl = step.querySelector('.ewr-timing-metric-value');
        if (valueEl) {
            valueEl.textContent = value;
        }

        if (status) {
            this.setStepStatus(stepId, status);
        }
    }

    // Update total timing
    updateTotal(value) {
        const totalEl = this.querySelector('#timingTotal');
        if (totalEl) {
            totalEl.textContent = value;
        }
    }

    // Set step status: 'pending', 'active', 'complete', 'error', 'skipped'
    setStepStatus(stepId, status) {
        const step = this.querySelector(`#step-${stepId}`);
        if (!step) return;

        // Remove all status classes
        step.classList.remove('pending', 'active', 'complete', 'error', 'skipped');

        // Add new status class
        if (status) {
            step.classList.add(status);
        }
    }

    // Reset all timings to default
    reset() {
        this._steps.forEach(step => {
            this.updateStep(step.id, '--');
            this.setStepStatus(step.id, null);
        });
        this.updateTotal('--');
    }

    // Get all step elements
    getSteps() {
        return this._steps.map(s => ({
            id: s.id,
            label: s.label,
            element: this.querySelector(`#step-${s.id}`)
        }));
    }
}

customElements.define('ewr-timing-metrics', EwrTimingMetrics);


/**
 * Helper function for collapsible toggle (ensure it's in global scope)
 */
if (typeof window.toggleCollapsible === 'undefined') {
    window.toggleCollapsible = function(panelId) {
        const panel = document.getElementById(panelId);
        if (panel) {
            panel.classList.toggle('collapsed');
        }
    };
}
