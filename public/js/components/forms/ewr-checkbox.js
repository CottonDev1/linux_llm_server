/**
 * EWR Checkbox Component
 * A styled checkbox with label
 * Uses Shadow DOM for style encapsulation
 *
 * @element ewr-checkbox
 *
 * @attr {string} label - Checkbox label text
 * @attr {boolean} checked - Checked state
 * @attr {boolean} disabled - Disabled state
 * @attr {string} value - Value when checked
 *
 * @fires ewr-change - When checkbox state changes
 *
 * @example
 * <ewr-checkbox label="Enable feature" checked></ewr-checkbox>
 */

import { EwrBaseComponent } from '../base/ewr-base-component.js';

export class EwrCheckbox extends EwrBaseComponent {
    static get observedAttributes() {
        return ['label', 'checked', 'disabled', 'value'];
    }

    constructor() {
        super();
        this._handleChange = this._handleChange.bind(this);
    }

    onConnected() {
        this._attachListeners();
    }

    onDisconnected() {
        this._detachListeners();
    }

    onRendered() {
        this._attachListeners();
    }

    _attachListeners() {
        this.$('input')?.addEventListener('change', this._handleChange);
    }

    _detachListeners() {
        this.$('input')?.removeEventListener('change', this._handleChange);
    }

    _handleChange(event) {
        this.emit('ewr-change', {
            checked: event.target.checked,
            value: this.value
        });
    }

    get label() { return this.getAttr('label', ''); }
    get checked() { return this.getBoolAttr('checked'); }
    set checked(value) { this.setBoolAttr('checked', value); }
    get disabled() { return this.getBoolAttr('disabled'); }
    set disabled(value) { this.setBoolAttr('disabled', value); }
    get value() { return this.getAttr('value', 'on'); }

    toggle() {
        if (!this.disabled) {
            this.checked = !this.checked;
            this.emit('ewr-change', { checked: this.checked, value: this.value });
        }
    }

    render() {
        const label = this.label;
        const checked = this.checked;
        const disabled = this.disabled;
        const checkboxId = `checkbox-${Math.random().toString(36).substr(2, 9)}`;

        return `
            <style>
                ${this.getThemeVariables()}
                ${this.getStyles()}
            </style>
            <label class="ewr-checkbox ${disabled ? 'ewr-checkbox--disabled' : ''}">
                <input
                    type="checkbox"
                    id="${checkboxId}"
                    class="ewr-checkbox__input"
                    ${checked ? 'checked' : ''}
                    ${disabled ? 'disabled' : ''}
                    part="input"
                />
                <span class="ewr-checkbox__box">
                    <svg class="ewr-checkbox__check" viewBox="0 0 12 12" fill="none">
                        <path d="M2 6L5 9L10 3" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                    </svg>
                </span>
                ${label ? `<span class="ewr-checkbox__label">${this.escapeHtml(label)}</span>` : ''}
            </label>
        `;
    }

    getStyles() {
        return `
            :host { display: inline-flex; }

            .ewr-checkbox {
                display: inline-flex;
                align-items: center;
                gap: 8px;
                cursor: pointer;
                user-select: none;
            }

            .ewr-checkbox--disabled {
                cursor: not-allowed;
                opacity: 0.6;
            }

            .ewr-checkbox__input {
                position: absolute;
                opacity: 0;
                width: 0;
                height: 0;
            }

            .ewr-checkbox__box {
                width: 20px;
                height: 20px;
                border: 2px solid var(--border-primary);
                border-radius: 4px;
                background: var(--bg-primary);
                display: flex;
                align-items: center;
                justify-content: center;
                transition: all var(--ewr-transition);
            }

            .ewr-checkbox__check {
                width: 12px;
                height: 12px;
                color: white;
                opacity: 0;
                transform: scale(0.5);
                transition: all var(--ewr-transition);
            }

            .ewr-checkbox__input:checked + .ewr-checkbox__box {
                background: var(--accent-primary);
                border-color: var(--accent-primary);
            }

            .ewr-checkbox__input:checked + .ewr-checkbox__box .ewr-checkbox__check {
                opacity: 1;
                transform: scale(1);
            }

            .ewr-checkbox__input:focus + .ewr-checkbox__box {
                box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.3);
            }

            .ewr-checkbox__label {
                color: var(--text-primary);
                font-size: 14px;
            }
        `;
    }
}

if (!customElements.get('ewr-checkbox')) {
    customElements.define('ewr-checkbox', EwrCheckbox);
}

export default EwrCheckbox;
