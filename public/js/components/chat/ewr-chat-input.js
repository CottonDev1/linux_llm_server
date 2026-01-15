/**
 * EWR Chat Input Component
 * A multi-line text input with send and clear buttons for chat interfaces
 * Uses Shadow DOM for style encapsulation
 *
 * @element ewr-chat-input
 *
 * @attr {string} placeholder - Input placeholder text
 * @attr {boolean} disabled - Disabled state
 * @attr {boolean} autofocus - Auto-focus on mount
 * @attr {number} max-rows - Maximum rows before scrolling (default: 5)
 *
 * @slot send-text - Text for send button (default: shows icon)
 * @slot clear-text - Text for clear button (default: "Clear")
 *
 * @fires ewr-submit - When message is submitted (Enter or click send)
 * @fires ewr-clear - When clear button is clicked
 * @fires ewr-input - When text changes
 *
 * @example
 * <ewr-chat-input placeholder="Type a message..."></ewr-chat-input>
 */

import { EwrBaseComponent } from '../base/ewr-base-component.js';

export class EwrChatInput extends EwrBaseComponent {
    static get observedAttributes() {
        return ['placeholder', 'disabled', 'autofocus', 'max-rows'];
    }

    constructor() {
        super();
        this._handleInput = this._handleInput.bind(this);
        this._handleKeydown = this._handleKeydown.bind(this);
        this._handleSend = this._handleSend.bind(this);
        this._handleClear = this._handleClear.bind(this);
    }

    onConnected() {
        this._attachListeners();
        if (this.autofocus) {
            setTimeout(() => this.focus(), 0);
        }
    }

    onDisconnected() {
        this._detachListeners();
    }

    onRendered() {
        this._attachListeners();
    }

    _attachListeners() {
        this.$('.ewr-chat-input__textarea')?.addEventListener('input', this._handleInput);
        this.$('.ewr-chat-input__textarea')?.addEventListener('keydown', this._handleKeydown);
        this.$('.ewr-chat-input__send')?.addEventListener('click', this._handleSend);
        this.$('.ewr-chat-input__clear')?.addEventListener('click', this._handleClear);
    }

    _detachListeners() {
        this.$('.ewr-chat-input__textarea')?.removeEventListener('input', this._handleInput);
        this.$('.ewr-chat-input__textarea')?.removeEventListener('keydown', this._handleKeydown);
        this.$('.ewr-chat-input__send')?.removeEventListener('click', this._handleSend);
        this.$('.ewr-chat-input__clear')?.removeEventListener('click', this._handleClear);
    }

    _handleInput(event) {
        this._autoResize();
        this.emit('ewr-input', { value: event.target.value });
    }

    _handleKeydown(event) {
        if (event.key === 'Enter' && !event.shiftKey) {
            event.preventDefault();
            this._handleSend();
        }
    }

    _handleSend() {
        const value = this.value.trim();
        if (value && !this.disabled) {
            this.emit('ewr-submit', { value });
        }
    }

    _handleClear() {
        this.clear();
        this.emit('ewr-clear');
    }

    _autoResize() {
        const textarea = this.$('.ewr-chat-input__textarea');
        if (textarea) {
            textarea.style.height = 'auto';
            const maxHeight = this.maxRows * 24; // ~24px per row
            textarea.style.height = Math.min(textarea.scrollHeight, maxHeight) + 'px';
        }
    }

    get placeholder() { return this.getAttr('placeholder', 'Type a message...'); }
    get disabled() { return this.getBoolAttr('disabled'); }
    set disabled(value) { this.setBoolAttr('disabled', value); }
    get autofocus() { return this.getBoolAttr('autofocus'); }
    get maxRows() { return parseInt(this.getAttr('max-rows', '5'), 10); }

    get value() {
        const textarea = this.$('.ewr-chat-input__textarea');
        return textarea ? textarea.value : '';
    }

    set value(val) {
        const textarea = this.$('.ewr-chat-input__textarea');
        if (textarea) {
            textarea.value = val || '';
            this._autoResize();
        }
    }

    focus() {
        this.$('.ewr-chat-input__textarea')?.focus();
    }

    clear() {
        this.value = '';
        this.focus();
    }

    render() {
        const placeholder = this.placeholder;
        const disabled = this.disabled;

        return `
            <style>
                ${this.getThemeVariables()}
                ${this.getStyles()}
            </style>
            <div class="ewr-chat-input" part="container">
                <div class="ewr-chat-input__row">
                    <textarea
                        class="ewr-chat-input__textarea"
                        placeholder="${this.escapeHtml(placeholder)}"
                        ${disabled ? 'disabled' : ''}
                        rows="1"
                        part="textarea"
                    ></textarea>
                    <div class="ewr-chat-input__buttons">
                        <button
                            class="ewr-chat-input__send"
                            type="button"
                            ${disabled ? 'disabled' : ''}
                            part="send"
                            title="Send message"
                        >
                            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                                <path d="M22 2L11 13M22 2L15 22L11 13L2 9L22 2Z"/>
                            </svg>
                        </button>
                        <button
                            class="ewr-chat-input__clear"
                            type="button"
                            ${disabled ? 'disabled' : ''}
                            part="clear"
                        >
                            <slot name="clear-text">Clear</slot>
                        </button>
                    </div>
                </div>
            </div>
        `;
    }

    getStyles() {
        return `
            :host { display: block; }

            .ewr-chat-input {
                padding: 16px;
                border-top: 1px solid var(--border-primary);
                background: var(--bg-tertiary);
            }

            .ewr-chat-input__row {
                display: flex;
                gap: 12px;
                align-items: flex-end;
            }

            .ewr-chat-input__textarea {
                flex: 1;
                min-height: 48px;
                max-height: 150px;
                padding: 14px 18px;
                font-size: 14px;
                font-family: inherit;
                resize: none;
                overflow-y: auto;
                background: rgba(15, 23, 42, 0.5);
                color: var(--text-primary);
                border: 1px solid var(--border-primary);
                border-radius: 8px;
                box-shadow: inset 0 1px 3px rgba(0, 0, 0, 0.2);
                transition: all 0.15s ease;
            }

            .ewr-chat-input__textarea:focus {
                outline: none;
                border-color: #00d4ff;
                background: rgba(0, 212, 255, 0.05);
                box-shadow: 0 0 0 3px rgba(0, 212, 255, 0.1), inset 0 1px 3px rgba(0, 0, 0, 0.2);
            }

            .ewr-chat-input__textarea::placeholder {
                color: var(--text-tertiary);
            }

            .ewr-chat-input__textarea:disabled {
                background: var(--bg-tertiary);
                color: var(--text-tertiary);
                cursor: not-allowed;
                opacity: 0.6;
            }

            .ewr-chat-input__buttons {
                display: flex;
                gap: 8px;
                flex-shrink: 0;
            }

            .ewr-chat-input__send {
                width: 48px;
                height: 48px;
                border-radius: 8px;
                background: linear-gradient(180deg, #e0f2fe 0%, #bae6fd 30%, #7dd3fc 70%, #38bdf8 100%);
                border: none;
                color: #0c4a6e;
                font-size: 18px;
                cursor: pointer;
                transition: all 0.2s ease;
                display: flex;
                align-items: center;
                justify-content: center;
                box-shadow:
                    0 1px 2px rgba(0, 0, 0, 0.2),
                    inset 0 1px 1px rgba(255, 255, 255, 0.6),
                    inset 0 -1px 1px rgba(0, 0, 0, 0.1);
            }

            .ewr-chat-input__send svg {
                width: 20px;
                height: 20px;
            }

            .ewr-chat-input__send:hover:not(:disabled) {
                background: linear-gradient(180deg, #f0f9ff 0%, #e0f2fe 30%, #bae6fd 70%, #7dd3fc 100%);
                transform: translateY(-1px);
            }

            .ewr-chat-input__send:active:not(:disabled) {
                background: linear-gradient(180deg, #7dd3fc 0%, #38bdf8 30%, #0ea5e9 70%, #0284c7 100%);
                color: white;
                transform: translateY(0);
            }

            .ewr-chat-input__send:disabled {
                opacity: 0.5;
                cursor: not-allowed;
            }

            .ewr-chat-input__clear {
                padding: 12px 16px;
                height: 48px;
                border-radius: 8px;
                background: linear-gradient(180deg, #9ca3af 0%, #6b7280 30%, #4b5563 70%, #374151 100%);
                border: none;
                color: white;
                font-size: 13px;
                font-weight: 600;
                font-family: inherit;
                cursor: pointer;
                transition: all 0.2s ease;
                display: flex;
                align-items: center;
                justify-content: center;
                box-shadow:
                    0 1px 2px rgba(0, 0, 0, 0.2),
                    inset 0 1px 1px rgba(255, 255, 255, 0.3),
                    inset 0 -1px 1px rgba(0, 0, 0, 0.1);
            }

            .ewr-chat-input__clear:hover:not(:disabled) {
                background: linear-gradient(180deg, #d1d5db 0%, #9ca3af 30%, #6b7280 70%, #4b5563 100%);
            }

            .ewr-chat-input__clear:active:not(:disabled) {
                background: linear-gradient(180deg, #4b5563 0%, #374151 30%, #1f2937 70%, #111827 100%);
            }

            .ewr-chat-input__clear:disabled {
                opacity: 0.5;
                cursor: not-allowed;
            }
        `;
    }
}

if (!customElements.get('ewr-chat-input')) {
    customElements.define('ewr-chat-input', EwrChatInput);
}

export default EwrChatInput;
