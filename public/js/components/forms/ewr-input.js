/**
 * EWR Input Component
 * A text input with label, validation, and consistent styling
 * Uses Shadow DOM for style encapsulation
 *
 * @element ewr-input
 *
 * @attr {string} label - Input label text
 * @attr {string} type - Input type: text, password, email, number, tel, url, search
 * @attr {string} value - Input value
 * @attr {string} placeholder - Placeholder text
 * @attr {boolean} required - Required field
 * @attr {boolean} disabled - Disabled state
 * @attr {boolean} readonly - Readonly state
 * @attr {string} error - Error message to display
 * @attr {string} hint - Hint text below input
 * @attr {string} size - Input size: sm, md, lg
 * @attr {string} layout - Layout: vertical (default), horizontal
 *
 * @fires ewr-input - When value changes (debounced)
 * @fires ewr-change - When input loses focus with changed value
 * @fires ewr-focus - When input receives focus
 * @fires ewr-blur - When input loses focus
 *
 * @example
 * <ewr-input label="Username" placeholder="Enter username" required></ewr-input>
 * <ewr-input label="Email" type="email" error="Invalid email format"></ewr-input>
 * <ewr-input label="Password" type="password" hint="At least 8 characters"></ewr-input>
 */

import { EwrBaseComponent } from '../base/ewr-base-component.js';

export class EwrInput extends EwrBaseComponent {
    static get observedAttributes() {
        return ['label', 'type', 'value', 'placeholder', 'required', 'disabled', 'readonly', 'error', 'hint', 'size', 'layout'];
    }

    constructor() {
        super();
        this._value = '';
        this._handleInput = this._handleInput.bind(this);
        this._handleChange = this._handleChange.bind(this);
        this._handleFocus = this._handleFocus.bind(this);
        this._handleBlur = this._handleBlur.bind(this);
    }

    onConnected() {
        this._attachInputListeners();
    }

    onDisconnected() {
        this._detachInputListeners();
    }

    onRendered() {
        this._attachInputListeners();
        // Restore value if it was set before render
        if (this._value) {
            const input = this.$('input');
            if (input) input.value = this._value;
        }
    }

    _attachInputListeners() {
        const input = this.$('input');
        if (input) {
            input.addEventListener('input', this._handleInput);
            input.addEventListener('change', this._handleChange);
            input.addEventListener('focus', this._handleFocus);
            input.addEventListener('blur', this._handleBlur);
        }
    }

    _detachInputListeners() {
        const input = this.$('input');
        if (input) {
            input.removeEventListener('input', this._handleInput);
            input.removeEventListener('change', this._handleChange);
            input.removeEventListener('focus', this._handleFocus);
            input.removeEventListener('blur', this._handleBlur);
        }
    }

    _handleInput(event) {
        this._value = event.target.value;
        this.emit('ewr-input', { value: this._value });
    }

    _handleChange(event) {
        this._value = event.target.value;
        this.emit('ewr-change', { value: this._value });
    }

    _handleFocus() {
        this.emit('ewr-focus', { value: this._value });
    }

    _handleBlur() {
        this.emit('ewr-blur', { value: this._value });
    }

    // Public API
    get value() {
        const input = this.$('input');
        return input ? input.value : this._value;
    }

    set value(val) {
        this._value = val || '';
        const input = this.$('input');
        if (input) input.value = this._value;
    }

    get label() {
        return this.getAttr('label', '');
    }

    get type() {
        return this.getAttr('type', 'text');
    }

    get placeholder() {
        return this.getAttr('placeholder', '');
    }

    get required() {
        return this.getBoolAttr('required');
    }

    get disabled() {
        return this.getBoolAttr('disabled');
    }

    set disabled(value) {
        this.setBoolAttr('disabled', value);
    }

    get readonly() {
        return this.getBoolAttr('readonly');
    }

    get error() {
        return this.getAttr('error', '');
    }

    set error(value) {
        if (value) {
            this.setAttribute('error', value);
        } else {
            this.removeAttribute('error');
        }
    }

    get hint() {
        return this.getAttr('hint', '');
    }

    get size() {
        return this.getAttr('size', 'md');
    }

    get layout() {
        return this.getAttr('layout', 'vertical');
    }

    focus() {
        this.$('input')?.focus();
    }

    blur() {
        this.$('input')?.blur();
    }

    select() {
        this.$('input')?.select();
    }

    clear() {
        this.value = '';
        this.error = '';
    }

    render() {
        const label = this.label;
        const type = this.type;
        const placeholder = this.placeholder;
        const required = this.required;
        const disabled = this.disabled;
        const readonly = this.readonly;
        const error = this.error;
        const hint = this.hint;
        const size = this.size;
        const layout = this.layout;
        const value = this.getAttr('value', '');

        const inputClasses = [
            'ewr-input__field',
            `ewr-input__field--${size}`,
            error ? 'ewr-input__field--error' : ''
        ].filter(Boolean).join(' ');

        const containerClasses = [
            'ewr-input',
            `ewr-input--${layout}`,
            `ewr-input--${size}`
        ].join(' ');

        const inputId = `input-${Math.random().toString(36).substr(2, 9)}`;

        return `
            <style>
                ${this.getThemeVariables()}
                ${this.getStyles()}
            </style>
            <div class="${containerClasses}">
                ${label ? `
                    <label class="ewr-input__label ${required ? 'ewr-input__label--required' : ''}" for="${inputId}">
                        ${this.escapeHtml(label)}
                    </label>
                ` : ''}
                <div class="ewr-input__wrapper">
                    <input
                        id="${inputId}"
                        class="${inputClasses}"
                        type="${type}"
                        placeholder="${this.escapeHtml(placeholder)}"
                        value="${this.escapeHtml(value)}"
                        ${required ? 'required' : ''}
                        ${disabled ? 'disabled' : ''}
                        ${readonly ? 'readonly' : ''}
                        part="input"
                    />
                    ${error ? `<div class="ewr-input__error">${this.escapeHtml(error)}</div>` : ''}
                    ${hint && !error ? `<div class="ewr-input__hint">${this.escapeHtml(hint)}</div>` : ''}
                </div>
            </div>
        `;
    }

    getStyles() {
        return `
            :host {
                display: block;
            }

            .ewr-input {
                display: flex;
                flex-direction: column;
                gap: 6px;
            }

            .ewr-input--horizontal {
                flex-direction: row;
                align-items: center;
                gap: 12px;
            }

            .ewr-input--horizontal .ewr-input__label {
                min-width: 100px;
                margin-bottom: 0;
            }

            .ewr-input--horizontal .ewr-input__wrapper {
                flex: 1;
            }

            .ewr-input__label {
                display: block;
                color: var(--text-secondary);
                font-weight: 600;
                font-size: 13px;
            }

            .ewr-input__label--required::after {
                content: ' *';
                color: var(--accent-danger);
            }

            .ewr-input__wrapper {
                display: flex;
                flex-direction: column;
                gap: 4px;
            }

            .ewr-input__field {
                width: 100%;
                background: var(--bg-primary);
                border: 2px solid var(--border-primary);
                border-radius: var(--ewr-border-radius);
                color: var(--text-primary);
                font-size: 14px;
                font-family: inherit;
                transition: border-color var(--ewr-transition), box-shadow var(--ewr-transition);
            }

            /* Sizes */
            .ewr-input__field--sm {
                padding: 6px 10px;
                font-size: 13px;
                height: 32px;
            }

            .ewr-input__field--md {
                padding: 10px 12px;
                font-size: 14px;
                height: 40px;
            }

            .ewr-input__field--lg {
                padding: 12px 16px;
                font-size: 16px;
                height: 48px;
            }

            .ewr-input__field::placeholder {
                color: var(--text-tertiary);
            }

            .ewr-input__field:focus {
                outline: none;
                border-color: var(--accent-primary);
                box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.2);
            }

            .ewr-input__field:disabled {
                opacity: 0.6;
                cursor: not-allowed;
                background: var(--bg-secondary);
            }

            .ewr-input__field:read-only {
                background: var(--bg-secondary);
            }

            .ewr-input__field--error {
                border-color: var(--accent-danger);
            }

            .ewr-input__field--error:focus {
                box-shadow: 0 0 0 3px rgba(239, 68, 68, 0.2);
            }

            .ewr-input__error {
                color: var(--accent-danger);
                font-size: 12px;
            }

            .ewr-input__hint {
                color: var(--text-tertiary);
                font-size: 12px;
            }
        `;
    }
}

// Define the custom element
if (!customElements.get('ewr-input')) {
    customElements.define('ewr-input', EwrInput);
}

export default EwrInput;
