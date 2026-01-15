/**
 * EWR Status Pill Component
 * A compact status indicator pill/badge
 * Uses Shadow DOM for style encapsulation
 *
 * @element ewr-status-pill
 *
 * @attr {string} variant - Pill variant: success, error, warning, info, default
 * @attr {string} size - Size: sm, md, lg
 *
 * @slot - Default slot for text content
 *
 * @example
 * <ewr-status-pill variant="success">Active</ewr-status-pill>
 * <ewr-status-pill variant="error">Failed</ewr-status-pill>
 */

import { EwrBaseComponent } from '../base/ewr-base-component.js';

export class EwrStatusPill extends EwrBaseComponent {
    static get observedAttributes() {
        return ['variant', 'size'];
    }

    get variant() { return this.getAttr('variant', 'default'); }
    get size() { return this.getAttr('size', 'md'); }

    render() {
        const variant = this.variant;
        const size = this.size;

        const classes = [
            'ewr-status-pill',
            `ewr-status-pill--${variant}`,
            `ewr-status-pill--${size}`
        ].join(' ');

        return `
            <style>
                ${this.getThemeVariables()}
                ${this.getStyles()}
            </style>
            <span class="${classes}" part="pill">
                <slot></slot>
            </span>
        `;
    }

    getStyles() {
        return `
            :host { display: inline-flex; }

            .ewr-status-pill {
                display: inline-flex;
                align-items: center;
                border-radius: 20px;
                font-weight: 600;
                white-space: nowrap;
            }

            .ewr-status-pill--sm { padding: 3px 8px; font-size: 10px; }
            .ewr-status-pill--md { padding: 6px 12px; font-size: 12px; }
            .ewr-status-pill--lg { padding: 8px 16px; font-size: 14px; }

            .ewr-status-pill--default {
                color: #94a3b8;
                background: rgba(148, 163, 184, 0.15);
                border: 1px solid rgba(148, 163, 184, 0.3);
            }

            .ewr-status-pill--success {
                color: #10b981;
                background: rgba(16, 185, 129, 0.15);
                border: 1px solid rgba(16, 185, 129, 0.4);
            }

            .ewr-status-pill--error {
                color: #ef4444;
                background: rgba(239, 68, 68, 0.15);
                border: 1px solid rgba(239, 68, 68, 0.4);
            }

            .ewr-status-pill--warning {
                color: #f59e0b;
                background: rgba(245, 158, 11, 0.15);
                border: 1px solid rgba(245, 158, 11, 0.4);
            }

            .ewr-status-pill--info {
                color: #3b82f6;
                background: rgba(59, 130, 246, 0.15);
                border: 1px solid rgba(59, 130, 246, 0.4);
            }

            .ewr-status-pill--primary {
                color: #60a5fa;
                background: rgba(59, 130, 246, 0.15);
                border: 1px solid rgba(59, 130, 246, 0.3);
            }
        `;
    }
}

if (!customElements.get('ewr-status-pill')) {
    customElements.define('ewr-status-pill', EwrStatusPill);
}

export default EwrStatusPill;
