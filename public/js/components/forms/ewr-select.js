/**
 * EWR Select Component
 * A dropdown select with label, validation, and consistent styling
 * Uses Shadow DOM for style encapsulation
 *
 * @element ewr-select
 *
 * @attr {string} label - Select label text
 * @attr {string} value - Selected value
 * @attr {boolean} required - Required field
 * @attr {boolean} disabled - Disabled state
 * @attr {string} error - Error message to display
 * @attr {string} hint - Hint text below select
 * @attr {string} size - Select size: sm, md, lg
 * @attr {string} layout - Layout: vertical (default), horizontal
 * @attr {string} placeholder - Placeholder text for default option
 * @attr {string} options - JSON array of options [{value, text}]
 *
 * @slot - Default slot for <option> elements
 *
 * @fires ewr-change - When selection changes
 *
 * @example
 * <ewr-select label="Category" required>
 *   <option value="">Select...</option>
 *   <option value="a">Option A</option>
 * </ewr-select>
 */

import { EwrBaseComponent } from '../base/ewr-base-component.js';

export class EwrSelect extends EwrBaseComponent {
    static get observedAttributes() {
        return ['label', 'value', 'required', 'disabled', 'error', 'hint', 'size', 'layout', 'placeholder', 'options'];
    }

    constructor() {
        super();
        this._value = '';
        this._options = [];
        this._handleChange = this._handleChange.bind(this);
    }

    connectedCallback() {
        // Capture slotted options before shadow DOM setup
        this._captureOptions();
        super.connectedCallback();
    }

    _captureOptions() {
        const slottedOptions = Array.from(this.querySelectorAll('option'));
        if (slottedOptions.length > 0) {
            this._options = slottedOptions.map(opt => ({
                value: opt.value,
                text: opt.textContent,
                selected: opt.selected
            }));
        }
    }

    onConnected() {
        this._attachListeners();
    }

    onDisconnected() {
        this._detachListeners();
    }

    onRendered() {
        this._attachListeners();
        if (this._value) {
            const select = this.$('select');
            if (select) select.value = this._value;
        }
    }

    _attachListeners() {
        const select = this.$('select');
        if (select) {
            select.addEventListener('change', this._handleChange);
        }
    }

    _detachListeners() {
        const select = this.$('select');
        if (select) {
            select.removeEventListener('change', this._handleChange);
        }
    }

    _handleChange(event) {
        this._value = event.target.value;
        this.emit('ewr-change', { value: this._value });
    }

    get value() {
        const select = this.$('select');
        return select ? select.value : this._value;
    }

    set value(val) {
        this._value = val || '';
        const select = this.$('select');
        if (select) select.value = this._value;
    }

    get label() { return this.getAttr('label', ''); }
    get required() { return this.getBoolAttr('required'); }
    get disabled() { return this.getBoolAttr('disabled'); }
    set disabled(value) { this.setBoolAttr('disabled', value); }
    get error() { return this.getAttr('error', ''); }
    set error(value) { value ? this.setAttribute('error', value) : this.removeAttribute('error'); }
    get hint() { return this.getAttr('hint', ''); }
    get size() { return this.getAttr('size', 'md'); }
    get layout() { return this.getAttr('layout', 'vertical'); }
    get placeholder() { return this.getAttr('placeholder', ''); }

    setOptions(options, placeholder = '') {
        this._options = options.map(opt => {
            if (typeof opt === 'string') return { value: opt, text: opt };
            return { value: opt.value || opt.id, text: opt.text || opt.name || opt.id };
        });
        if (placeholder) {
            this._options.unshift({ value: '', text: placeholder });
        }
        this.refresh();
    }

    render() {
        const label = this.label;
        const required = this.required;
        const disabled = this.disabled;
        const error = this.error;
        const hint = this.hint;
        const size = this.size;
        const layout = this.layout;
        const placeholder = this.placeholder;
        const value = this.getAttr('value', '');

        // Parse options from attribute if provided
        let options = this._options;
        const optionsAttr = this.getAttr('options', '');
        if (optionsAttr) {
            try {
                options = JSON.parse(optionsAttr);
            } catch (e) {}
        }

        if (placeholder && !options.find(o => o.value === '')) {
            options = [{ value: '', text: placeholder }, ...options];
        }

        const selectClasses = [
            'ewr-select__field',
            `ewr-select__field--${size}`,
            error ? 'ewr-select__field--error' : ''
        ].filter(Boolean).join(' ');

        const containerClasses = [
            'ewr-select',
            `ewr-select--${layout}`,
            `ewr-select--${size}`
        ].join(' ');

        const selectId = `select-${Math.random().toString(36).substr(2, 9)}`;

        const optionsHtml = options.map(opt =>
            `<option value="${this.escapeHtml(opt.value)}" ${opt.value === value ? 'selected' : ''}>${this.escapeHtml(opt.text)}</option>`
        ).join('');

        return `
            <style>
                ${this.getThemeVariables()}
                ${this.getStyles()}
            </style>
            <div class="${containerClasses}">
                ${label ? `
                    <label class="ewr-select__label ${required ? 'ewr-select__label--required' : ''}" for="${selectId}">
                        ${this.escapeHtml(label)}
                    </label>
                ` : ''}
                <div class="ewr-select__wrapper">
                    <select
                        id="${selectId}"
                        class="${selectClasses}"
                        ${required ? 'required' : ''}
                        ${disabled ? 'disabled' : ''}
                        part="select"
                    >
                        ${optionsHtml}
                    </select>
                    ${error ? `<div class="ewr-select__error">${this.escapeHtml(error)}</div>` : ''}
                    ${hint && !error ? `<div class="ewr-select__hint">${this.escapeHtml(hint)}</div>` : ''}
                </div>
            </div>
        `;
    }

    getStyles() {
        return `
            :host { display: block; }

            .ewr-select {
                display: flex;
                flex-direction: column;
                gap: 6px;
            }

            .ewr-select--horizontal {
                flex-direction: row;
                align-items: center;
                gap: 12px;
            }

            .ewr-select--horizontal .ewr-select__label {
                min-width: 100px;
                margin-bottom: 0;
            }

            .ewr-select--horizontal .ewr-select__wrapper {
                flex: 1;
            }

            .ewr-select__label {
                display: block;
                color: var(--text-secondary);
                font-weight: 600;
                font-size: 13px;
            }

            .ewr-select__label--required::after {
                content: ' *';
                color: var(--accent-danger);
            }

            .ewr-select__wrapper {
                display: flex;
                flex-direction: column;
                gap: 4px;
            }

            .ewr-select__field {
                width: 100%;
                background: var(--bg-primary);
                border: 2px solid var(--border-primary);
                border-radius: var(--ewr-border-radius);
                color: var(--text-primary);
                font-size: 14px;
                font-family: inherit;
                cursor: pointer;
                transition: border-color var(--ewr-transition), box-shadow var(--ewr-transition);
                appearance: none;
                background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='12' height='12' fill='%2394a3b8' viewBox='0 0 16 16'%3E%3Cpath d='M8 11L3 6h10l-5 5z'/%3E%3C/svg%3E");
                background-repeat: no-repeat;
                background-position: right 12px center;
                padding-right: 36px;
            }

            .ewr-select__field--sm { padding: 6px 36px 6px 10px; font-size: 13px; height: 32px; }
            .ewr-select__field--md { padding: 10px 36px 10px 12px; font-size: 14px; height: 40px; }
            .ewr-select__field--lg { padding: 12px 36px 12px 16px; font-size: 16px; height: 48px; }

            .ewr-select__field:focus {
                outline: none;
                border-color: var(--accent-primary);
                box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.2);
            }

            .ewr-select__field:disabled {
                opacity: 0.6;
                cursor: not-allowed;
                background-color: var(--bg-secondary);
            }

            .ewr-select__field--error { border-color: var(--accent-danger); }
            .ewr-select__field--error:focus { box-shadow: 0 0 0 3px rgba(239, 68, 68, 0.2); }

            .ewr-select__field option {
                background: var(--bg-secondary);
                color: var(--text-primary);
            }

            .ewr-select__error { color: var(--accent-danger); font-size: 12px; }
            .ewr-select__hint { color: var(--text-tertiary); font-size: 12px; }
        `;
    }
}

if (!customElements.get('ewr-select')) {
    customElements.define('ewr-select', EwrSelect);
}

export default EwrSelect;
