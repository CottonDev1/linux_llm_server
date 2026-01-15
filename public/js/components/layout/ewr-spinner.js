/**
 * EWR Spinner Component
 * A loading spinner indicator
 * Uses Shadow DOM for style encapsulation
 *
 * @element ewr-spinner
 *
 * @attr {string} size - Size: xs, sm, md, lg, xl
 * @attr {string} color - Color: primary, success, danger, white, inherit
 *
 * @example
 * <ewr-spinner></ewr-spinner>
 * <ewr-spinner size="lg" color="primary"></ewr-spinner>
 */

import { EwrBaseComponent } from '../base/ewr-base-component.js';

export class EwrSpinner extends EwrBaseComponent {
    static get observedAttributes() {
        return ['size', 'color'];
    }

    get size() { return this.getAttr('size', 'md'); }
    get color() { return this.getAttr('color', 'primary'); }

    render() {
        const size = this.size;
        const color = this.color;

        const classes = [
            'ewr-spinner',
            `ewr-spinner--${size}`,
            `ewr-spinner--${color}`
        ].join(' ');

        return `
            <style>
                ${this.getThemeVariables()}
                ${this.getStyles()}
            </style>
            <div class="${classes}" part="spinner">
                <div class="ewr-spinner__ring"></div>
            </div>
        `;
    }

    getStyles() {
        return `
            :host { display: inline-flex; }

            .ewr-spinner {
                display: inline-flex;
                align-items: center;
                justify-content: center;
            }

            .ewr-spinner__ring {
                border-radius: 50%;
                border-style: solid;
                border-color: transparent;
                animation: ewr-spin 0.8s linear infinite;
            }

            .ewr-spinner--xs .ewr-spinner__ring { width: 12px; height: 12px; border-width: 2px; }
            .ewr-spinner--sm .ewr-spinner__ring { width: 16px; height: 16px; border-width: 2px; }
            .ewr-spinner--md .ewr-spinner__ring { width: 24px; height: 24px; border-width: 3px; }
            .ewr-spinner--lg .ewr-spinner__ring { width: 32px; height: 32px; border-width: 3px; }
            .ewr-spinner--xl .ewr-spinner__ring { width: 48px; height: 48px; border-width: 4px; }

            .ewr-spinner--primary .ewr-spinner__ring {
                border-top-color: var(--accent-primary);
                border-right-color: rgba(59, 130, 246, 0.3);
            }

            .ewr-spinner--success .ewr-spinner__ring {
                border-top-color: var(--accent-success);
                border-right-color: rgba(16, 185, 129, 0.3);
            }

            .ewr-spinner--danger .ewr-spinner__ring {
                border-top-color: var(--accent-danger);
                border-right-color: rgba(239, 68, 68, 0.3);
            }

            .ewr-spinner--white .ewr-spinner__ring {
                border-top-color: white;
                border-right-color: rgba(255, 255, 255, 0.3);
            }

            .ewr-spinner--inherit .ewr-spinner__ring {
                border-top-color: currentColor;
                border-right-color: currentColor;
                opacity: 0.3;
            }

            @keyframes ewr-spin {
                to { transform: rotate(360deg); }
            }
        `;
    }
}

if (!customElements.get('ewr-spinner')) {
    customElements.define('ewr-spinner', EwrSpinner);
}

export default EwrSpinner;
