/**
 * EWR Base Component
 * Foundation class for all EWR Web Components with Shadow DOM
 *
 * Features:
 * - Shadow DOM encapsulation
 * - CSS variable inheritance from host document
 * - Event dispatch helpers
 * - Attribute/property reflection
 * - Template rendering utilities
 *
 * @example
 * class MyComponent extends EwrBaseComponent {
 *   static get observedAttributes() { return ['value', 'label']; }
 *
 *   render() {
 *     return `
 *       <style>${this.getBaseStyles()}</style>
 *       <div class="my-component">
 *         <label>${this.getAttribute('label')}</label>
 *       </div>
 *     `;
 *   }
 * }
 */

export class EwrBaseComponent extends HTMLElement {
    constructor() {
        super();
        this._shadowRoot = this.attachShadow({ mode: 'open' });
        this._initialized = false;
    }

    /**
     * Called when the element is added to the DOM
     */
    connectedCallback() {
        if (!this._initialized) {
            this._renderComponent();
            this._initialized = true;
        }
        this.onConnected?.();
    }

    /**
     * Called when the element is removed from the DOM
     */
    disconnectedCallback() {
        this.onDisconnected?.();
    }

    /**
     * Called when an observed attribute changes
     */
    attributeChangedCallback(name, oldValue, newValue) {
        if (oldValue !== newValue) {
            this.onAttributeChanged?.(name, oldValue, newValue);
            if (this._initialized) {
                this._renderComponent();
            }
        }
    }

    /**
     * Render the component's Shadow DOM content
     * @private
     */
    _renderComponent() {
        const template = this.render?.();
        if (template) {
            this._shadowRoot.innerHTML = template;
            this.onRendered?.();
        }
    }

    /**
     * Force a re-render of the component
     */
    refresh() {
        this._renderComponent();
    }

    /**
     * Get the shadow root
     * @returns {ShadowRoot}
     */
    get shadow() {
        return this._shadowRoot;
    }

    /**
     * Query an element in the shadow DOM
     * @param {string} selector - CSS selector
     * @returns {Element|null}
     */
    $(selector) {
        return this._shadowRoot.querySelector(selector);
    }

    /**
     * Query all elements in the shadow DOM
     * @param {string} selector - CSS selector
     * @returns {NodeList}
     */
    $$(selector) {
        return this._shadowRoot.querySelectorAll(selector);
    }

    /**
     * Emit a custom event from this component
     * @param {string} eventName - Name of the event
     * @param {*} detail - Event detail data
     * @param {Object} options - Event options
     */
    emit(eventName, detail = null, options = {}) {
        const event = new CustomEvent(eventName, {
            bubbles: true,
            composed: true, // Allows event to cross shadow DOM boundary
            detail,
            ...options
        });
        this.dispatchEvent(event);
        return event;
    }

    /**
     * Get a boolean attribute value
     * @param {string} name - Attribute name
     * @returns {boolean}
     */
    getBoolAttr(name) {
        return this.hasAttribute(name);
    }

    /**
     * Set a boolean attribute
     * @param {string} name - Attribute name
     * @param {boolean} value - Whether to set or remove
     */
    setBoolAttr(name, value) {
        if (value) {
            this.setAttribute(name, '');
        } else {
            this.removeAttribute(name);
        }
    }

    /**
     * Get an attribute with a default value
     * @param {string} name - Attribute name
     * @param {string} defaultValue - Default value if not set
     * @returns {string}
     */
    getAttr(name, defaultValue = '') {
        return this.getAttribute(name) ?? defaultValue;
    }

    /**
     * Base CSS variables that inherit from the document theme
     * Components should include this in their styles
     * @returns {string}
     */
    getThemeVariables() {
        return `
            :host {
                /* Inherit CSS variables from parent document */
                --bg-primary: var(--bg-primary, #0f172a);
                --bg-secondary: var(--bg-secondary, #1e293b);
                --bg-tertiary: var(--bg-tertiary, #334155);

                --text-primary: var(--text-primary, #f1f5f9);
                --text-secondary: var(--text-secondary, #cbd5e1);
                --text-tertiary: var(--text-tertiary, #94a3b8);

                --border-primary: var(--border-primary, #334155);
                --border-secondary: var(--border-secondary, #475569);

                --accent-primary: var(--accent-primary, #3b82f6);
                --accent-success: var(--accent-success, #10b981);
                --accent-warning: var(--accent-warning, #f59e0b);
                --accent-danger: var(--accent-danger, #ef4444);
                --accent-info: var(--accent-info, #06b6d4);

                --error: var(--error, #ef4444);
                --success: var(--success, #10b981);
                --warning: var(--warning, #f59e0b);
                --info: var(--info, #3b82f6);

                /* Component defaults */
                --ewr-transition: 0.2s ease;
                --ewr-border-radius: 6px;
                --ewr-border-radius-sm: 4px;
                --ewr-border-radius-lg: 8px;

                /* Font settings */
                font-family: inherit;
                box-sizing: border-box;
            }

            *, *::before, *::after {
                box-sizing: border-box;
            }
        `;
    }

    /**
     * Get focus ring styles for accessibility
     * @returns {string}
     */
    getFocusStyles() {
        return `
            .focus-ring:focus {
                outline: none;
                box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.3);
            }

            .focus-ring:focus-visible {
                outline: 2px solid var(--accent-primary);
                outline-offset: 2px;
            }
        `;
    }

    /**
     * Get common button styles
     * @returns {string}
     */
    getButtonBaseStyles() {
        return `
            button {
                font-family: inherit;
                font-size: inherit;
                cursor: pointer;
                border: none;
                background: transparent;
                padding: 0;
                margin: 0;
            }

            button:disabled {
                cursor: not-allowed;
                opacity: 0.5;
            }
        `;
    }

    /**
     * Get common input styles
     * @returns {string}
     */
    getInputBaseStyles() {
        return `
            input, select, textarea {
                font-family: inherit;
                font-size: inherit;
                color: inherit;
            }

            input:disabled, select:disabled, textarea:disabled {
                cursor: not-allowed;
                opacity: 0.6;
            }
        `;
    }

    /**
     * Combine multiple style strings
     * @param {...string} styles - Style strings to combine
     * @returns {string}
     */
    combineStyles(...styles) {
        return styles.join('\n');
    }

    /**
     * Create a property getter/setter that reflects to an attribute
     * Call this in the constructor for each property you want to reflect
     * @param {string} propName - Property name
     * @param {string} attrName - Attribute name (defaults to kebab-case of propName)
     * @param {*} defaultValue - Default value
     */
    reflectProperty(propName, attrName = null, defaultValue = null) {
        attrName = attrName || this._toKebabCase(propName);

        Object.defineProperty(this, propName, {
            get() {
                const value = this.getAttribute(attrName);
                if (value === null) return defaultValue;
                if (typeof defaultValue === 'boolean') return this.hasAttribute(attrName);
                if (typeof defaultValue === 'number') return parseFloat(value);
                return value;
            },
            set(value) {
                if (typeof defaultValue === 'boolean') {
                    this.setBoolAttr(attrName, value);
                } else if (value === null || value === undefined) {
                    this.removeAttribute(attrName);
                } else {
                    this.setAttribute(attrName, String(value));
                }
            }
        });
    }

    /**
     * Convert camelCase to kebab-case
     * @private
     */
    _toKebabCase(str) {
        return str.replace(/([a-z0-9])([A-Z])/g, '$1-$2').toLowerCase();
    }

    /**
     * Escape HTML special characters
     * @param {string} text - Text to escape
     * @returns {string}
     */
    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    /**
     * Parse a slot value that might be JSON
     * @param {string} value - Attribute value
     * @returns {*}
     */
    parseJsonAttr(name, defaultValue = null) {
        const value = this.getAttribute(name);
        if (!value) return defaultValue;
        try {
            return JSON.parse(value);
        } catch {
            return defaultValue;
        }
    }
}

// Export for ES module usage
export default EwrBaseComponent;
