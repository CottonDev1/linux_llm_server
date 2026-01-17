/**
 * EWR Website Status Component
 * Displays system status indicator with animated dot and text
 * Uses Shadow DOM for style encapsulation
 * Integrates with system-status.js for automatic updates
 *
 * @element ewr-website-status
 *
 * @example
 * <ewr-website-status></ewr-website-status>
 */

import { EwrBaseComponent } from '../base/ewr-base-component.js';

export class EwrWebsiteStatus extends EwrBaseComponent {
    static get observedAttributes() {
        return ['status'];
    }

    constructor() {
        super();
        this._statusKey = 'connecting';
        this._statusText = 'Connecting';
    }

    onConnected() {
        // Register for system status updates if available
        if (typeof window.registerStatusComponent === 'function') {
            window.registerStatusComponent(this);
        }
    }

    onDisconnected() {
        // Unregister from system status updates if available
        if (typeof window.unregisterStatusComponent === 'function') {
            window.unregisterStatusComponent(this);
        }
    }

    get status() {
        return this._statusKey;
    }

    set status(value) {
        this._statusKey = value;
        this._updateStatusText();
        this.refresh();
    }

    _updateStatusText() {
        switch (this._statusKey) {
            case 'healthy': this._statusText = 'Online'; break;
            case 'degraded-llm': this._statusText = 'LLM Down'; break;
            case 'degraded': this._statusText = 'Degraded'; break;
            case 'error': this._statusText = 'Error'; break;
            case 'offline': this._statusText = 'Offline'; break;
            default: this._statusText = 'Connecting';
        }
    }

    _getDotClass() {
        switch (this._statusKey) {
            case 'healthy': return 'status-healthy';
            case 'degraded-llm':
            case 'degraded': return 'status-degraded';
            case 'error':
            case 'offline': return 'status-error';
            default: return 'status-connecting';
        }
    }

    _getTextColor() {
        switch (this._statusKey) {
            case 'healthy': return '#10b981';
            case 'degraded-llm':
            case 'degraded': return '#f59e0b';
            case 'error':
            case 'offline': return '#ef4444';
            default: return '#64748b';
        }
    }

    /**
     * Update status - called by system-status.js
     * @param {object} status - Full status object from Python service
     * @param {string} statusKey - Status key: healthy, degraded, degraded-llm, error, offline
     */
    updateStatus(status, statusKey) {
        this._statusKey = statusKey;
        this._updateStatusText();
        this.refresh();
    }

    render() {
        return `
            <style>
                ${this.getThemeVariables()}
                ${this.getStyles()}
            </style>
            <div class="website-status" part="container">
                <div class="status-dot ${this._getDotClass()}" part="dot"></div>
                <span class="status-label" style="color: ${this._getTextColor()}" part="label">${this._statusText}</span>
            </div>
        `;
    }

    getStyles() {
        return `
            :host {
                display: block;
            }

            .website-status {
                display: flex;
                align-items: center;
                gap: 6px;
                padding: 4px 8px;
                background: rgba(0, 0, 0, 0.25);
                border-radius: 4px;
                border: 1px solid rgba(255, 255, 255, 0.06);
            }

            .status-dot {
                width: 6px;
                height: 6px;
                border-radius: 50%;
                flex-shrink: 0;
            }

            .status-dot.status-healthy {
                background: #10b981;
                box-shadow: 0 0 4px rgba(16, 185, 129, 0.5);
            }

            .status-dot.status-degraded {
                background: #f59e0b;
                box-shadow: 0 0 4px rgba(245, 158, 11, 0.5);
            }

            .status-dot.status-error {
                background: #ef4444;
                box-shadow: 0 0 4px rgba(239, 68, 68, 0.5);
            }

            .status-dot.status-connecting {
                background: #64748b;
                animation: statusPulse 1.5s ease-in-out infinite;
            }

            @keyframes statusPulse {
                0%, 100% { opacity: 0.4; }
                50% { opacity: 1; }
            }

            .status-label {
                font-size: 9px;
                font-weight: 600;
                text-transform: uppercase;
                letter-spacing: 0.5px;
            }
        `;
    }
}

// Registration handled by components/index.js
// Fallback self-registration for standalone use
if (!customElements.get('ewr-website-status')) {
    customElements.define('ewr-website-status', EwrWebsiteStatus);
}

export default EwrWebsiteStatus;
