/**
 * EWR Button Component
 * A customizable button with multiple variants, sizes, and states
 * Uses Shadow DOM for style encapsulation
 *
 * @element ewr-btn
 *
 * @attr {string} variant - Button variant: primary, secondary, success, danger, warning, info, outline, ghost
 * @attr {string} size - Button size: xs, sm, md, lg, xl
 * @attr {boolean} disabled - Disabled state
 * @attr {boolean} loading - Loading state with spinner
 * @attr {boolean} block - Full width button
 * @attr {string} type - Button type: button, submit, reset
 *
 * @slot - Default slot for button content
 * @slot icon-left - Icon to show before text
 * @slot icon-right - Icon to show after text
 *
 * @fires click - When button is clicked (native)
 * @fires ewr-click - Custom click event with component reference
 *
 * @example
 * <ewr-btn variant="primary">Click Me</ewr-btn>
 * <ewr-btn variant="danger" size="lg">Delete</ewr-btn>
 * <ewr-btn variant="success" loading>Saving...</ewr-btn>
 * <ewr-btn variant="outline" disabled>Disabled</ewr-btn>
 */

import { EwrBaseComponent } from '../base/ewr-base-component.js';

export class EwrButton extends EwrBaseComponent {
    static get observedAttributes() {
        return ['variant', 'size', 'disabled', 'loading', 'block', 'type'];
    }

    constructor() {
        super();
        this._handleClick = this._handleClick.bind(this);
    }

    onConnected() {
        this.$('button')?.addEventListener('click', this._handleClick);
    }

    onDisconnected() {
        this.$('button')?.removeEventListener('click', this._handleClick);
    }

    onRendered() {
        this.$('button')?.addEventListener('click', this._handleClick);
    }

    _handleClick(event) {
        if (this.disabled || this.loading) {
            event.preventDefault();
            event.stopPropagation();
            return;
        }
        this.emit('ewr-click', { button: this });
    }

    get variant() {
        return this.getAttr('variant', 'primary');
    }

    get size() {
        return this.getAttr('size', 'md');
    }

    get disabled() {
        return this.getBoolAttr('disabled');
    }

    set disabled(value) {
        this.setBoolAttr('disabled', value);
    }

    get loading() {
        return this.getBoolAttr('loading');
    }

    set loading(value) {
        this.setBoolAttr('loading', value);
    }

    get block() {
        return this.getBoolAttr('block');
    }

    get type() {
        return this.getAttr('type', 'button');
    }

    render() {
        const variant = this.variant;
        const size = this.size;
        const disabled = this.disabled || this.loading;
        const loading = this.loading;
        const block = this.block;
        const type = this.type;

        const classes = [
            'ewr-btn',
            `ewr-btn--${variant}`,
            `ewr-btn--${size}`,
            block ? 'ewr-btn--block' : '',
            loading ? 'ewr-btn--loading' : ''
        ].filter(Boolean).join(' ');

        return `
            <style>
                ${this.getThemeVariables()}
                ${this.getStyles()}
            </style>
            <button
                class="${classes}"
                type="${type}"
                ${disabled ? 'disabled' : ''}
                part="button"
            >
                <span class="ewr-btn__icon-left">
                    <slot name="icon-left"></slot>
                </span>
                <span class="ewr-btn__content">
                    <slot></slot>
                </span>
                <span class="ewr-btn__icon-right">
                    <slot name="icon-right"></slot>
                </span>
                ${loading ? '<span class="ewr-btn__spinner"></span>' : ''}
            </button>
        `;
    }

    getStyles() {
        return `
            :host {
                display: inline-flex;
            }

            :host([block]) {
                display: flex;
                width: 100%;
            }

            .ewr-btn {
                display: inline-flex;
                align-items: center;
                justify-content: center;
                gap: 6px;
                border-radius: var(--ewr-border-radius);
                font-weight: 600;
                font-family: inherit;
                cursor: pointer;
                transition: all var(--ewr-transition);
                border: none;
                text-decoration: none;
                white-space: nowrap;
                user-select: none;
                position: relative;
                box-shadow:
                    0 1px 2px rgba(0, 0, 0, 0.2),
                    inset 0 1px 1px rgba(255, 255, 255, 0.4),
                    inset 0 -1px 1px rgba(0, 0, 0, 0.1);
                text-shadow: 0 1px 0 rgba(0, 0, 0, 0.2);
            }

            .ewr-btn:focus {
                outline: 2px solid var(--accent-primary);
                outline-offset: 2px;
            }

            .ewr-btn:disabled {
                opacity: 0.5;
                cursor: not-allowed;
                transform: none !important;
                box-shadow: none !important;
            }

            /* Sizes */
            .ewr-btn--xs {
                padding: 4px 8px;
                font-size: 11px;
                height: 24px;
            }

            .ewr-btn--sm {
                padding: 6px 12px;
                font-size: 12px;
                height: 28px;
            }

            .ewr-btn--md {
                padding: 8px 16px;
                font-size: 14px;
                height: 36px;
            }

            .ewr-btn--lg {
                padding: 10px 20px;
                font-size: 16px;
                height: 44px;
            }

            .ewr-btn--xl {
                padding: 12px 28px;
                font-size: 18px;
                height: 52px;
            }

            /* Primary (Green metallic gradient) */
            .ewr-btn--primary {
                background: linear-gradient(180deg, #6ee7b7 0%, #34d399 30%, #10b981 70%, #059669 100%);
                color: #ffffff;
            }

            .ewr-btn--primary:hover:not(:disabled) {
                background: linear-gradient(180deg, #a7f3d0 0%, #6ee7b7 30%, #34d399 70%, #10b981 100%);
                transform: translateY(-1px);
                box-shadow:
                    0 4px 12px rgba(16, 185, 129, 0.4),
                    inset 0 1px 1px rgba(255, 255, 255, 0.5),
                    inset 0 -1px 1px rgba(0, 0, 0, 0.1);
            }

            .ewr-btn--primary:active:not(:disabled) {
                background: linear-gradient(180deg, #10b981 0%, #059669 30%, #047857 70%, #065f46 100%);
                transform: translateY(0);
                box-shadow:
                    inset 0 2px 3px rgba(0, 0, 0, 0.2),
                    inset 0 -1px 1px rgba(255, 255, 255, 0.1);
            }

            /* Success (same as primary) */
            .ewr-btn--success {
                background: linear-gradient(180deg, #6ee7b7 0%, #34d399 30%, #10b981 70%, #059669 100%);
                color: #ffffff;
            }

            .ewr-btn--success:hover:not(:disabled) {
                background: linear-gradient(180deg, #a7f3d0 0%, #6ee7b7 30%, #34d399 70%, #10b981 100%);
                transform: translateY(-1px);
                box-shadow: 0 4px 12px rgba(16, 185, 129, 0.4);
            }

            .ewr-btn--success:active:not(:disabled) {
                background: linear-gradient(180deg, #10b981 0%, #059669 30%, #047857 70%, #065f46 100%);
            }

            /* Secondary (Gray metallic) */
            .ewr-btn--secondary {
                background: linear-gradient(180deg, #9ca3af 0%, #6b7280 30%, #4b5563 70%, #374151 100%);
                color: #ffffff;
            }

            .ewr-btn--secondary:hover:not(:disabled) {
                background: linear-gradient(180deg, #d1d5db 0%, #9ca3af 30%, #6b7280 70%, #4b5563 100%);
            }

            .ewr-btn--secondary:active:not(:disabled) {
                background: linear-gradient(180deg, #4b5563 0%, #374151 30%, #1f2937 70%, #111827 100%);
            }

            /* Danger (Red metallic) */
            .ewr-btn--danger {
                background: linear-gradient(180deg, #fca5a5 0%, #f87171 30%, #ef4444 70%, #dc2626 100%);
                color: #ffffff;
            }

            .ewr-btn--danger:hover:not(:disabled) {
                background: linear-gradient(180deg, #fecaca 0%, #fca5a5 30%, #f87171 70%, #ef4444 100%);
                transform: translateY(-1px);
                box-shadow: 0 4px 12px rgba(239, 68, 68, 0.4);
            }

            .ewr-btn--danger:active:not(:disabled) {
                background: linear-gradient(180deg, #ef4444 0%, #dc2626 30%, #b91c1c 70%, #991b1b 100%);
            }

            /* Warning (Orange metallic) */
            .ewr-btn--warning {
                background: linear-gradient(180deg, #fcd34d 0%, #fbbf24 30%, #f59e0b 70%, #d97706 100%);
                color: #ffffff;
            }

            .ewr-btn--warning:hover:not(:disabled) {
                background: linear-gradient(180deg, #fde68a 0%, #fcd34d 30%, #fbbf24 70%, #f59e0b 100%);
                transform: translateY(-1px);
                box-shadow: 0 4px 12px rgba(245, 158, 11, 0.4);
            }

            .ewr-btn--warning:active:not(:disabled) {
                background: linear-gradient(180deg, #f59e0b 0%, #d97706 30%, #b45309 70%, #92400e 100%);
            }

            /* Info (Blue metallic) */
            .ewr-btn--info {
                background: linear-gradient(180deg, #93c5fd 0%, #60a5fa 30%, #3b82f6 70%, #2563eb 100%);
                color: #ffffff;
            }

            .ewr-btn--info:hover:not(:disabled) {
                background: linear-gradient(180deg, #bfdbfe 0%, #93c5fd 30%, #60a5fa 70%, #3b82f6 100%);
                transform: translateY(-1px);
                box-shadow: 0 4px 12px rgba(59, 130, 246, 0.4);
            }

            .ewr-btn--info:active:not(:disabled) {
                background: linear-gradient(180deg, #3b82f6 0%, #2563eb 30%, #1d4ed8 70%, #1e40af 100%);
            }

            /* Light Blue (Send button style) */
            .ewr-btn--send {
                background: linear-gradient(180deg, #e0f2fe 0%, #bae6fd 30%, #7dd3fc 70%, #38bdf8 100%);
                color: #0c4a6e;
                text-shadow: 0 1px 0 rgba(255, 255, 255, 0.4);
                box-shadow:
                    0 1px 2px rgba(0, 0, 0, 0.2),
                    inset 0 1px 1px rgba(255, 255, 255, 0.6),
                    inset 0 -1px 1px rgba(0, 0, 0, 0.1);
            }

            .ewr-btn--send:hover:not(:disabled) {
                background: linear-gradient(180deg, #f0f9ff 0%, #e0f2fe 30%, #bae6fd 70%, #7dd3fc 100%);
            }

            .ewr-btn--send:active:not(:disabled) {
                background: linear-gradient(180deg, #7dd3fc 0%, #38bdf8 30%, #0ea5e9 70%, #0284c7 100%);
                color: #ffffff;
            }

            /* Outline */
            .ewr-btn--outline {
                background: transparent;
                border: 2px solid var(--border-primary);
                color: var(--text-primary);
                box-shadow: none;
                text-shadow: none;
            }

            .ewr-btn--outline:hover:not(:disabled) {
                background: rgba(255, 255, 255, 0.05);
                border-color: var(--text-tertiary);
            }

            /* Ghost */
            .ewr-btn--ghost {
                background: transparent;
                color: var(--text-secondary);
                box-shadow: none;
                text-shadow: none;
            }

            .ewr-btn--ghost:hover:not(:disabled) {
                background: rgba(255, 255, 255, 0.05);
                color: var(--text-primary);
            }

            /* Block */
            .ewr-btn--block {
                display: flex;
                width: 100%;
            }

            /* Loading state */
            .ewr-btn--loading {
                color: transparent !important;
                pointer-events: none;
            }

            .ewr-btn--loading .ewr-btn__content,
            .ewr-btn--loading .ewr-btn__icon-left,
            .ewr-btn--loading .ewr-btn__icon-right {
                visibility: hidden;
            }

            .ewr-btn__spinner {
                position: absolute;
                width: 16px;
                height: 16px;
                border: 2px solid rgba(255, 255, 255, 0.3);
                border-top-color: white;
                border-radius: 50%;
                animation: ewr-btn-spin 0.6s linear infinite;
            }

            @keyframes ewr-btn-spin {
                to { transform: rotate(360deg); }
            }

            /* Icon slots */
            .ewr-btn__icon-left,
            .ewr-btn__icon-right {
                display: inline-flex;
                align-items: center;
            }

            .ewr-btn__icon-left:empty,
            .ewr-btn__icon-right:empty {
                display: none;
            }

            /* Slot content styling */
            ::slotted(ewr-icon) {
                display: inline-flex;
            }
        `;
    }
}

// Define the custom element
if (!customElements.get('ewr-btn')) {
    customElements.define('ewr-btn', EwrButton);
}

export default EwrButton;
