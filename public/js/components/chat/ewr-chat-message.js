/**
 * EWR Chat Message Component
 * A chat message with avatar and styled text bubble
 * Uses Shadow DOM for style encapsulation
 *
 * @element ewr-chat-message
 *
 * @attr {string} role - Message role: user, assistant, error, system
 * @attr {string} avatar - Avatar text (initials or emoji)
 * @attr {string} name - Sender name
 * @attr {string} time - Timestamp text
 *
 * @slot - Default slot for message content (supports HTML)
 *
 * @example
 * <ewr-chat-message role="user" name="User" time="2:30 PM">
 *   <p>Hello, how can I help?</p>
 * </ewr-chat-message>
 */

import { EwrBaseComponent } from '../base/ewr-base-component.js';

export class EwrChatMessage extends EwrBaseComponent {
    static get observedAttributes() {
        return ['role', 'avatar', 'name', 'time'];
    }

    get role() { return this.getAttr('role', 'user'); }
    get avatar() {
        const custom = this.getAttr('avatar', '');
        if (custom) return custom;
        const role = this.role;
        if (role === 'user') return 'U';
        if (role === 'assistant') return 'AI';
        if (role === 'error') return '!';
        return '?';
    }
    get name() { return this.getAttr('name', ''); }
    get time() { return this.getAttr('time', ''); }

    render() {
        const role = this.role;
        const avatar = this.avatar;
        const name = this.name;
        const time = this.time;

        const classes = [
            'ewr-chat-message',
            `ewr-chat-message--${role}`
        ].join(' ');

        return `
            <style>
                ${this.getThemeVariables()}
                ${this.getStyles()}
            </style>
            <div class="${classes}" part="message">
                <div class="ewr-chat-message__avatar" part="avatar">
                    ${this.escapeHtml(avatar)}
                </div>
                <div class="ewr-chat-message__content">
                    ${name || time ? `
                        <div class="ewr-chat-message__header">
                            ${name ? `<span class="ewr-chat-message__name">${this.escapeHtml(name)}</span>` : ''}
                            ${time ? `<span class="ewr-chat-message__time">${this.escapeHtml(time)}</span>` : ''}
                        </div>
                    ` : ''}
                    <div class="ewr-chat-message__text" part="text">
                        <slot></slot>
                    </div>
                </div>
            </div>
        `;
    }

    getStyles() {
        return `
            :host { display: block; }

            .ewr-chat-message {
                display: flex;
                gap: 12px;
                padding: 12px;
                border-radius: 8px;
                animation: ewr-message-slide-in 0.3s ease;
            }

            @keyframes ewr-message-slide-in {
                from { opacity: 0; transform: translateY(10px); }
                to { opacity: 1; transform: translateY(0); }
            }

            .ewr-chat-message__avatar {
                width: 32px;
                height: 32px;
                border-radius: 50%;
                display: flex;
                align-items: center;
                justify-content: center;
                font-size: 11px;
                font-weight: 700;
                color: white;
                flex-shrink: 0;
                text-transform: uppercase;
            }

            .ewr-chat-message__content {
                flex: 1;
                min-width: 0;
            }

            .ewr-chat-message__header {
                display: flex;
                align-items: center;
                gap: 8px;
                margin-bottom: 6px;
            }

            .ewr-chat-message__name {
                font-size: 13px;
                font-weight: 600;
                color: var(--text-primary);
            }

            .ewr-chat-message__time {
                font-size: 11px;
                color: var(--text-tertiary);
            }

            .ewr-chat-message__text {
                background: var(--bg-secondary);
                padding: 12px 16px;
                border-radius: 0 12px 12px 12px;
                border: 1px solid var(--border-primary);
                font-size: 14px;
                line-height: 1.6;
                color: var(--text-secondary);
            }

            .ewr-chat-message__text ::slotted(p) {
                margin: 0 0 8px 0;
            }

            .ewr-chat-message__text ::slotted(p:last-child) {
                margin-bottom: 0;
            }

            /* User variant */
            .ewr-chat-message--user {
                background: rgba(59, 130, 246, 0.08);
            }

            .ewr-chat-message--user .ewr-chat-message__avatar {
                background: var(--accent-primary);
            }

            .ewr-chat-message--user .ewr-chat-message__text {
                background: var(--bg-tertiary);
                border-left: 3px solid var(--accent-primary);
            }

            /* Assistant variant */
            .ewr-chat-message--assistant {
                background: rgba(16, 185, 129, 0.08);
            }

            .ewr-chat-message--assistant .ewr-chat-message__avatar {
                background: var(--accent-success);
            }

            .ewr-chat-message--assistant .ewr-chat-message__text {
                border-left: 3px solid var(--accent-success);
            }

            /* Error variant */
            .ewr-chat-message--error {
                background: rgba(239, 68, 68, 0.08);
            }

            .ewr-chat-message--error .ewr-chat-message__avatar {
                background: var(--accent-danger);
            }

            .ewr-chat-message--error .ewr-chat-message__text {
                border-left: 3px solid var(--accent-danger);
            }

            /* System variant */
            .ewr-chat-message--system {
                background: rgba(148, 163, 184, 0.08);
            }

            .ewr-chat-message--system .ewr-chat-message__avatar {
                background: var(--text-tertiary);
            }

            .ewr-chat-message--system .ewr-chat-message__text {
                border-left: 3px solid var(--text-tertiary);
                font-style: italic;
            }
        `;
    }
}

if (!customElements.get('ewr-chat-message')) {
    customElements.define('ewr-chat-message', EwrChatMessage);
}

export default EwrChatMessage;
