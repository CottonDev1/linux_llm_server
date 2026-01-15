/**
 * EWR Modal Component
 * A modal dialog with overlay, header, body, and footer
 * Uses Shadow DOM for style encapsulation
 *
 * @element ewr-modal
 *
 * @attr {string} title - Modal title
 * @attr {string} size - Size: sm, md, lg, xl, full
 * @attr {boolean} open - Open state
 * @attr {boolean} no-close-on-overlay - Prevent closing when clicking overlay
 * @attr {boolean} no-close-button - Hide the close button
 * @attr {string} accent-color - Header accent color
 *
 * @slot - Default slot for body content
 * @slot footer - Footer content (buttons)
 *
 * @fires ewr-open - When modal opens
 * @fires ewr-close - When modal closes
 *
 * @example
 * <ewr-modal id="myModal" title="Confirm Action" size="sm">
 *   <p>Are you sure?</p>
 *   <div slot="footer">
 *     <ewr-btn variant="secondary" onclick="myModal.close()">Cancel</ewr-btn>
 *     <ewr-btn variant="danger">Delete</ewr-btn>
 *   </div>
 * </ewr-modal>
 */

import { EwrBaseComponent } from '../base/ewr-base-component.js';

export class EwrModalComponent extends EwrBaseComponent {
    static get observedAttributes() {
        return ['title', 'size', 'open', 'no-close-on-overlay', 'no-close-button', 'accent-color'];
    }

    constructor() {
        super();
        this._handleOverlayClick = this._handleOverlayClick.bind(this);
        this._handleCloseClick = this._handleCloseClick.bind(this);
        this._handleKeydown = this._handleKeydown.bind(this);
    }

    onConnected() {
        this._attachListeners();
    }

    onDisconnected() {
        this._detachListeners();
        document.removeEventListener('keydown', this._handleKeydown);
    }

    onRendered() {
        this._attachListeners();
    }

    _attachListeners() {
        this.$('.ewr-modal__overlay')?.addEventListener('click', this._handleOverlayClick);
        this.$('.ewr-modal__close')?.addEventListener('click', this._handleCloseClick);
    }

    _detachListeners() {
        this.$('.ewr-modal__overlay')?.removeEventListener('click', this._handleOverlayClick);
        this.$('.ewr-modal__close')?.removeEventListener('click', this._handleCloseClick);
    }

    _handleOverlayClick(event) {
        if (event.target === event.currentTarget && !this.noCloseOnOverlay) {
            this.close();
        }
    }

    _handleCloseClick() {
        this.close();
    }

    _handleKeydown(event) {
        if (event.key === 'Escape' && this.isOpen) {
            this.close();
        }
    }

    get title() { return this.getAttr('title', ''); }
    get size() { return this.getAttr('size', 'md'); }
    get isOpen() { return this.getBoolAttr('open'); }
    get noCloseOnOverlay() { return this.getBoolAttr('no-close-on-overlay'); }
    get noCloseButton() { return this.getBoolAttr('no-close-button'); }
    get accentColor() { return this.getAttr('accent-color', ''); }

    open() {
        this.setAttribute('open', '');
        document.addEventListener('keydown', this._handleKeydown);
        document.body.style.overflow = 'hidden';
        this.emit('ewr-open');
    }

    close() {
        this.removeAttribute('open');
        document.removeEventListener('keydown', this._handleKeydown);
        document.body.style.overflow = '';
        this.emit('ewr-close');
    }

    toggle() {
        if (this.isOpen) {
            this.close();
        } else {
            this.open();
        }
    }

    render() {
        const title = this.title;
        const size = this.size;
        const isOpen = this.isOpen;
        const noCloseButton = this.noCloseButton;
        const accentColor = this.accentColor;

        const overlayClasses = [
            'ewr-modal__overlay',
            isOpen ? '' : 'ewr-modal__overlay--hidden'
        ].filter(Boolean).join(' ');

        const dialogClasses = [
            'ewr-modal__dialog',
            `ewr-modal__dialog--${size}`
        ].join(' ');

        const headerStyle = accentColor ? `border-bottom-color: ${accentColor}; color: ${accentColor};` : '';

        return `
            <style>
                ${this.getThemeVariables()}
                ${this.getStyles()}
            </style>
            <div class="${overlayClasses}" part="overlay">
                <div class="${dialogClasses}" part="dialog">
                    <div class="ewr-modal__header" style="${headerStyle}" part="header">
                        <h2 class="ewr-modal__title">${this.escapeHtml(title)}</h2>
                        ${noCloseButton ? '' : `
                            <button class="ewr-modal__close" type="button" part="close" aria-label="Close">
                                &times;
                            </button>
                        `}
                    </div>
                    <div class="ewr-modal__body" part="body">
                        <slot></slot>
                    </div>
                    <div class="ewr-modal__footer" part="footer">
                        <slot name="footer"></slot>
                    </div>
                </div>
            </div>
        `;
    }

    getStyles() {
        return `
            :host { display: contents; }

            .ewr-modal__overlay {
                position: fixed;
                top: 0;
                left: 0;
                right: 0;
                bottom: 0;
                background: rgba(0, 0, 0, 0.7);
                backdrop-filter: blur(4px);
                display: flex;
                align-items: center;
                justify-content: center;
                z-index: 1000;
                padding: 20px;
                animation: ewr-modal-fade-in 0.2s ease;
            }

            .ewr-modal__overlay--hidden {
                display: none;
            }

            @keyframes ewr-modal-fade-in {
                from { opacity: 0; }
                to { opacity: 1; }
            }

            .ewr-modal__dialog {
                background: var(--bg-secondary);
                border-radius: 12px;
                box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.5);
                width: 100%;
                display: flex;
                flex-direction: column;
                overflow: hidden;
                animation: ewr-modal-slide-in 0.3s ease;
                max-height: 90vh;
            }

            @keyframes ewr-modal-slide-in {
                from {
                    opacity: 0;
                    transform: translateY(-20px) scale(0.95);
                }
                to {
                    opacity: 1;
                    transform: translateY(0) scale(1);
                }
            }

            .ewr-modal__dialog--sm { max-width: 400px; }
            .ewr-modal__dialog--md { max-width: 600px; }
            .ewr-modal__dialog--lg { max-width: 800px; }
            .ewr-modal__dialog--xl { max-width: 1000px; }
            .ewr-modal__dialog--full { max-width: 95vw; max-height: 95vh; }

            .ewr-modal__header {
                display: flex;
                align-items: center;
                justify-content: space-between;
                padding: 16px 20px;
                border-bottom: 2px solid var(--border-primary);
            }

            .ewr-modal__title {
                margin: 0;
                font-size: 18px;
                font-weight: 600;
                color: var(--text-primary);
            }

            .ewr-modal__close {
                background: transparent;
                border: none;
                color: var(--text-tertiary);
                font-size: 28px;
                cursor: pointer;
                padding: 0 8px;
                border-radius: 4px;
                transition: all 0.2s ease;
                line-height: 1;
                font-family: inherit;
            }

            .ewr-modal__close:hover {
                background: rgba(255, 255, 255, 0.1);
                color: var(--text-primary);
            }

            .ewr-modal__body {
                padding: 20px;
                overflow-y: auto;
                flex: 1;
            }

            .ewr-modal__footer {
                padding: 16px 20px;
                border-top: 1px solid var(--border-primary);
                display: flex;
                gap: 12px;
                justify-content: flex-end;
            }

            .ewr-modal__footer:empty {
                display: none;
            }

            ::slotted([slot="footer"]:empty) { display: none; }

            @media (max-width: 640px) {
                .ewr-modal__overlay {
                    padding: 10px;
                }

                .ewr-modal__dialog {
                    max-height: 95vh;
                }

                .ewr-modal__header {
                    padding: 12px 16px;
                }

                .ewr-modal__body {
                    padding: 16px;
                }

                .ewr-modal__footer {
                    padding: 12px 16px;
                    flex-wrap: wrap;
                }
            }
        `;
    }
}

if (!customElements.get('ewr-modal')) {
    customElements.define('ewr-modal', EwrModalComponent);
}

export default EwrModalComponent;
