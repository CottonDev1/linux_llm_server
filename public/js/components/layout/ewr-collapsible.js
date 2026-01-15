/**
 * EWR Collapsible Component
 * An expandable/collapsible section with header, content, and optional footer
 * Uses Shadow DOM for style encapsulation
 *
 * @element ewr-collapsible
 *
 * @attr {string} title - Section title
 * @attr {boolean} collapsed - Initial collapsed state
 * @attr {boolean} no-margin - Remove outer margin
 *
 * @slot header-actions - Actions in header (right side)
 * @slot - Default slot for content
 * @slot footer - Footer content
 *
 * @fires ewr-toggle - When collapsed state changes
 *
 * @example
 * <ewr-collapsible title="Filters">
 *   <p>Filter content here</p>
 *   <div slot="footer">
 *     <ewr-btn variant="primary">Apply</ewr-btn>
 *   </div>
 * </ewr-collapsible>
 */

import { EwrBaseComponent } from '../base/ewr-base-component.js';

export class EwrCollapsible extends EwrBaseComponent {
    static get observedAttributes() {
        return ['title', 'collapsed', 'no-margin'];
    }

    constructor() {
        super();
        this._handleToggle = this._handleToggle.bind(this);
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
        this.$('.ewr-collapsible__header')?.addEventListener('click', this._handleToggle);
    }

    _detachListeners() {
        this.$('.ewr-collapsible__header')?.removeEventListener('click', this._handleToggle);
    }

    _handleToggle(event) {
        // Don't toggle if clicking on slotted actions
        if (event.target.closest('[slot="header-actions"]')) return;

        this.toggle();
    }

    get title() { return this.getAttr('title', ''); }
    get collapsed() { return this.getBoolAttr('collapsed'); }
    set collapsed(value) { this.setBoolAttr('collapsed', value); }
    get noMargin() { return this.getBoolAttr('no-margin'); }

    toggle() {
        this.collapsed = !this.collapsed;
        this.emit('ewr-toggle', { collapsed: this.collapsed });
    }

    expand() {
        if (this.collapsed) {
            this.collapsed = false;
            this.emit('ewr-toggle', { collapsed: false });
        }
    }

    collapse() {
        if (!this.collapsed) {
            this.collapsed = true;
            this.emit('ewr-toggle', { collapsed: true });
        }
    }

    render() {
        const title = this.title;
        const collapsed = this.collapsed;
        const noMargin = this.noMargin;

        const containerClasses = [
            'ewr-collapsible',
            collapsed ? 'ewr-collapsible--collapsed' : '',
            noMargin ? 'ewr-collapsible--no-margin' : ''
        ].filter(Boolean).join(' ');

        return `
            <style>
                ${this.getThemeVariables()}
                ${this.getStyles()}
            </style>
            <div class="${containerClasses}" part="container">
                <div class="ewr-collapsible__header" part="header">
                    <h3 class="ewr-collapsible__title">${this.escapeHtml(title)}</h3>
                    <div class="ewr-collapsible__actions">
                        <slot name="header-actions"></slot>
                        <button class="ewr-collapsible__toggle" type="button">
                            <svg class="ewr-collapsible__icon" viewBox="0 0 12 12" fill="none">
                                <path d="M3 4.5L6 7.5L9 4.5" stroke="currentColor" stroke-width="2" stroke-linecap="round"/>
                            </svg>
                            <span>${collapsed ? 'Expand' : 'Collapse'}</span>
                        </button>
                    </div>
                </div>
                <div class="ewr-collapsible__content" part="content">
                    <slot></slot>
                </div>
                <div class="ewr-collapsible__footer" part="footer">
                    <slot name="footer"></slot>
                </div>
            </div>
        `;
    }

    getStyles() {
        return `
            :host { display: block; }

            .ewr-collapsible {
                display: flex;
                flex-direction: column;
                background: #3a4556;
                border-radius: 8px;
                margin: 8px;
                overflow: hidden;
            }

            .ewr-collapsible--no-margin {
                margin: 0;
                border-radius: 0;
            }

            .ewr-collapsible__header {
                display: flex;
                justify-content: space-between;
                align-items: center;
                padding: 10px 16px;
                background: #4a5568;
                cursor: pointer;
                user-select: none;
                border-radius: 8px 8px 0 0;
                transition: background 0.2s ease;
            }

            .ewr-collapsible--no-margin .ewr-collapsible__header {
                border-radius: 0;
            }

            .ewr-collapsible--collapsed .ewr-collapsible__header {
                border-radius: 8px;
            }

            .ewr-collapsible--no-margin.ewr-collapsible--collapsed .ewr-collapsible__header {
                border-radius: 0;
            }

            .ewr-collapsible__header:hover {
                background: #5a6578;
            }

            .ewr-collapsible__title {
                font-size: 16px;
                font-weight: 600;
                color: var(--text-primary);
                margin: 0;
            }

            .ewr-collapsible__actions {
                display: flex;
                align-items: center;
                gap: 8px;
            }

            .ewr-collapsible__toggle {
                background: transparent;
                border: 1px solid var(--border-primary);
                color: var(--text-secondary);
                padding: 6px 12px;
                font-size: 12px;
                font-weight: 500;
                font-family: inherit;
                cursor: pointer;
                border-radius: 6px;
                transition: all 0.2s ease;
                display: flex;
                align-items: center;
                gap: 6px;
            }

            .ewr-collapsible__toggle:hover {
                background: rgba(90, 159, 230, 0.1);
                border-color: var(--accent-primary);
                color: var(--accent-primary);
            }

            .ewr-collapsible__icon {
                width: 12px;
                height: 12px;
                transition: transform 0.3s ease;
            }

            .ewr-collapsible--collapsed .ewr-collapsible__icon {
                transform: rotate(-90deg);
            }

            .ewr-collapsible__content {
                background: #3a4556;
                transition: all 0.3s ease;
                overflow: hidden;
            }

            .ewr-collapsible--collapsed .ewr-collapsible__content {
                max-height: 0 !important;
                padding-top: 0 !important;
                padding-bottom: 0 !important;
            }

            .ewr-collapsible__footer {
                display: flex;
                flex-direction: row;
                gap: 8px;
                align-items: center;
                padding: 10px 16px;
                background: #2d3748;
                border-radius: 0 0 8px 8px;
            }

            .ewr-collapsible--no-margin .ewr-collapsible__footer {
                border-radius: 0;
            }

            .ewr-collapsible__footer:empty { display: none; }

            .ewr-collapsible--collapsed .ewr-collapsible__footer {
                display: none;
            }

            ::slotted([slot="footer"]:empty) { display: none; }
        `;
    }
}

if (!customElements.get('ewr-collapsible')) {
    customElements.define('ewr-collapsible', EwrCollapsible);
}

export default EwrCollapsible;
