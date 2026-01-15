/**
 * EWR Card Component
 * A flexible card container with optional header, body, and footer
 * Uses Shadow DOM for style encapsulation
 *
 * @element ewr-card
 *
 * @attr {string} title - Card title in header
 * @attr {string} variant - Card variant: default, chat, content
 * @attr {string} width - Width: auto, full, 75
 * @attr {boolean} no-padding - Remove body padding
 *
 * @slot header - Custom header content
 * @slot - Default slot for body content
 * @slot footer - Footer content
 *
 * @example
 * <ewr-card title="My Card">
 *   <p>Card content here</p>
 * </ewr-card>
 */

import { EwrBaseComponent } from '../base/ewr-base-component.js';

export class EwrCard extends EwrBaseComponent {
    static get observedAttributes() {
        return ['title', 'variant', 'width', 'no-padding'];
    }

    get title() { return this.getAttr('title', ''); }
    get variant() { return this.getAttr('variant', 'default'); }
    get width() { return this.getAttr('width', 'auto'); }
    get noPadding() { return this.getBoolAttr('no-padding'); }

    render() {
        const title = this.title;
        const variant = this.variant;
        const width = this.width;
        const noPadding = this.noPadding;

        const cardClasses = [
            'ewr-card',
            `ewr-card--${variant}`,
            `ewr-card--${width}`,
            noPadding ? 'ewr-card--no-padding' : ''
        ].filter(Boolean).join(' ');

        return `
            <style>
                ${this.getThemeVariables()}
                ${this.getStyles()}
            </style>
            <div class="${cardClasses}" part="card">
                <div class="ewr-card__header">
                    <slot name="header">
                        ${title ? `<h3 class="ewr-card__title">${this.escapeHtml(title)}</h3>` : ''}
                    </slot>
                </div>
                <div class="ewr-card__body">
                    <slot></slot>
                </div>
                <div class="ewr-card__footer">
                    <slot name="footer"></slot>
                </div>
            </div>
        `;
    }

    getStyles() {
        return `
            :host { display: block; }

            .ewr-card {
                display: flex;
                flex-direction: column;
                overflow: hidden;
                background: var(--bg-secondary);
                border-radius: 12px;
                border: 1px solid var(--border-primary);
                box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
                transition: all 0.3s ease;
            }

            .ewr-card--auto { width: auto; }
            .ewr-card--full { width: 100%; flex: 1; min-height: 0; }
            .ewr-card--75 { width: 75%; }

            .ewr-card--chat {
                flex: 1;
                min-height: 0;
            }

            .ewr-card--chat .ewr-card__body {
                flex: 1;
                min-height: 0;
                display: flex;
                flex-direction: column;
                overflow: hidden;
                background: var(--bg-primary);
            }

            .ewr-card__header {
                padding: 16px 20px;
                border-bottom: 1px solid var(--border-primary);
            }

            .ewr-card__header:empty { display: none; }

            .ewr-card__title {
                margin: 0;
                font-size: 18px;
                font-weight: 600;
                color: var(--text-primary);
            }

            .ewr-card__body {
                padding: 20px;
                overflow: auto;
            }

            .ewr-card--no-padding .ewr-card__body {
                padding: 0;
            }

            .ewr-card__footer {
                padding: 16px 20px;
                border-top: 1px solid var(--border-primary);
            }

            .ewr-card__footer:empty { display: none; }

            /* Hide empty header slot */
            ::slotted([slot="header"]:empty) { display: none; }
            ::slotted([slot="footer"]:empty) { display: none; }

            @media (max-width: 768px) {
                .ewr-card--75 { width: 100%; }
            }
        `;
    }
}

if (!customElements.get('ewr-card')) {
    customElements.define('ewr-card', EwrCard);
}

export default EwrCard;
