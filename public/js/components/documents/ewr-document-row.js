/**
 * EWR Document Row Component
 * A clickable document row with icon, title, metadata, and arrow indicator
 * Uses Shadow DOM for style encapsulation
 *
 * @element ewr-document-row
 *
 * @attr {string} title - Document title
 * @attr {string} meta - Metadata text (date, size, etc.)
 * @attr {string} icon - Icon text/emoji (default: file icon)
 * @attr {string} icon-color - Icon background color
 *
 * @fires ewr-click - When row is clicked
 *
 * @example
 * <ewr-document-row
 *   title="Annual Report 2024.pdf"
 *   meta="2.4 MB - Jan 5, 2025"
 *   icon="📄"
 * ></ewr-document-row>
 */

import { EwrBaseComponent } from '../base/ewr-base-component.js';

export class EwrDocumentRow extends EwrBaseComponent {
    static get observedAttributes() {
        return ['title', 'meta', 'icon', 'icon-color'];
    }

    constructor() {
        super();
        this._handleClick = this._handleClick.bind(this);
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
        this.$('.ewr-document-row')?.addEventListener('click', this._handleClick);
    }

    _detachListeners() {
        this.$('.ewr-document-row')?.removeEventListener('click', this._handleClick);
    }

    _handleClick() {
        this.emit('ewr-click', {
            title: this.title,
            meta: this.meta
        });
    }

    get title() { return this.getAttr('title', ''); }
    get meta() { return this.getAttr('meta', ''); }
    get icon() { return this.getAttr('icon', '📄'); }
    get iconColor() { return this.getAttr('icon-color', ''); }

    render() {
        const title = this.title;
        const meta = this.meta;
        const icon = this.icon;
        const iconColor = this.iconColor;

        const iconStyle = iconColor ? `background: ${iconColor};` : '';

        return `
            <style>
                ${this.getThemeVariables()}
                ${this.getStyles()}
            </style>
            <div class="ewr-document-row" part="row">
                <div class="ewr-document-row__icon" style="${iconStyle}" part="icon">
                    ${this.escapeHtml(icon)}
                </div>
                <div class="ewr-document-row__info">
                    <div class="ewr-document-row__title" part="title">${this.escapeHtml(title)}</div>
                    ${meta ? `<div class="ewr-document-row__meta" part="meta">${this.escapeHtml(meta)}</div>` : ''}
                </div>
                <div class="ewr-document-row__arrow" part="arrow">→</div>
            </div>
        `;
    }

    getStyles() {
        return `
            :host { display: block; }

            .ewr-document-row {
                display: flex;
                align-items: center;
                padding: 12px 16px;
                margin-bottom: 8px;
                background: var(--bg-secondary);
                border: 1px solid var(--border-primary);
                border-radius: 8px;
                cursor: pointer;
                transition: all 0.2s ease;
            }

            .ewr-document-row:hover {
                background: var(--bg-tertiary);
                border-color: var(--accent-primary);
                transform: translateX(4px);
            }

            .ewr-document-row__icon {
                width: 36px;
                height: 36px;
                background: var(--accent-primary);
                border-radius: 8px;
                display: flex;
                align-items: center;
                justify-content: center;
                margin-right: 12px;
                font-size: 16px;
                flex-shrink: 0;
            }

            .ewr-document-row__info {
                flex: 1;
                min-width: 0;
            }

            .ewr-document-row__title {
                font-size: 14px;
                font-weight: 500;
                color: var(--text-primary);
                margin-bottom: 2px;
                white-space: nowrap;
                overflow: hidden;
                text-overflow: ellipsis;
            }

            .ewr-document-row__meta {
                font-size: 12px;
                color: var(--text-tertiary);
            }

            .ewr-document-row__arrow {
                color: var(--text-tertiary);
                font-size: 18px;
                margin-left: 12px;
                flex-shrink: 0;
                transition: transform 0.2s ease;
            }

            .ewr-document-row:hover .ewr-document-row__arrow {
                transform: translateX(4px);
                color: var(--accent-primary);
            }
        `;
    }
}

if (!customElements.get('ewr-document-row')) {
    customElements.define('ewr-document-row', EwrDocumentRow);
}

export default EwrDocumentRow;
