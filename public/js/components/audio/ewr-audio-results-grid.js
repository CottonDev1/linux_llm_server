/**
 * EWR Audio Results Grid Component
 * A table/grid container for audio analysis results
 * Uses Shadow DOM for style encapsulation
 *
 * @element ewr-audio-results-grid
 *
 * @attr {string} max-height - Maximum height of the grid body (default: 600px)
 *
 * @slot - Default slot for ewr-audio-result-row elements
 *
 * @fires ewr-view - Bubbled from rows when view is clicked
 * @fires ewr-delete - Bubbled from rows when delete is clicked
 *
 * @example
 * <ewr-audio-results-grid>
 *   <ewr-audio-result-row result-id="123" date="2024-01-15" ...></ewr-audio-result-row>
 * </ewr-audio-results-grid>
 */

import { EwrBaseComponent } from '../base/ewr-base-component.js';

export class EwrAudioResultsGrid extends EwrBaseComponent {
    static get observedAttributes() {
        return ['max-height', 'loading'];
    }

    constructor() {
        super();
        this._resultsCount = 0;
    }

    onConnected() {
        this._observeSlot();
    }

    onRendered() {
        this._observeSlot();
    }

    _observeSlot() {
        const slot = this.$('slot:not([name])');
        slot?.addEventListener('slotchange', () => {
            this._updateResultsCount();
        });
    }

    _updateResultsCount() {
        const rows = this._getResultRows();
        this._resultsCount = rows.length;
        const countEl = this.$('.ewr-audio-grid__count');
        if (countEl) {
            countEl.textContent = this._resultsCount;
        }

        // Show/hide empty state
        const body = this.$('.ewr-audio-grid__body');
        const empty = this.$('.ewr-audio-grid__empty');
        if (body && empty) {
            empty.style.display = rows.length === 0 ? 'flex' : 'none';
        }
    }

    _getResultRows() {
        const slot = this.$('slot:not([name])');
        if (!slot) return [];
        return slot.assignedElements().filter(el => el.tagName === 'EWR-AUDIO-RESULT-ROW');
    }

    get maxHeight() { return this.getAttr('max-height', '600px'); }
    get loading() { return this.getBoolAttr('loading'); }
    set loading(value) { this.setBoolAttr('loading', value); }

    get resultsCount() { return this._resultsCount; }

    showLoading() {
        this.loading = true;
    }

    hideLoading() {
        this.loading = false;
    }

    render() {
        const maxHeight = this.maxHeight;
        const loading = this.loading;

        return `
            <style>
                ${this.getThemeVariables()}
                ${this.getStyles()}
            </style>
            <div class="ewr-audio-grid" part="grid">
                <div class="ewr-audio-grid__info" part="info">
                    <span class="ewr-audio-grid__info-text">
                        Showing <strong class="ewr-audio-grid__count">0</strong> results
                    </span>
                </div>
                <div class="ewr-audio-grid__container" part="container">
                    <div class="ewr-audio-grid__header" part="header">
                        <div class="ewr-audio-grid__header-cell ewr-audio-grid__header-cell--view">View</div>
                        <div class="ewr-audio-grid__header-cell ewr-audio-grid__header-cell--date">Date</div>
                        <div class="ewr-audio-grid__header-cell ewr-audio-grid__header-cell--time">Time</div>
                        <div class="ewr-audio-grid__header-cell ewr-audio-grid__header-cell--staff">Staff</div>
                        <div class="ewr-audio-grid__header-cell ewr-audio-grid__header-cell--customer">Customer</div>
                        <div class="ewr-audio-grid__header-cell ewr-audio-grid__header-cell--mood">Mood</div>
                        <div class="ewr-audio-grid__header-cell ewr-audio-grid__header-cell--outcome">Outcome</div>
                        <div class="ewr-audio-grid__header-cell ewr-audio-grid__header-cell--filename">Filename</div>
                        <div class="ewr-audio-grid__header-cell ewr-audio-grid__header-cell--actions"></div>
                    </div>
                    <div class="ewr-audio-grid__body" part="body" style="max-height: ${maxHeight}">
                        ${loading ? `
                            <div class="ewr-audio-grid__loading">
                                <div class="ewr-audio-grid__spinner"></div>
                                <div>Loading...</div>
                            </div>
                        ` : ''}
                        <slot></slot>
                    </div>
                    <div class="ewr-audio-grid__empty" part="empty" style="display: none;">
                        <slot name="empty"></slot>
                    </div>
                </div>
            </div>
        `;
    }

    getStyles() {
        return `
            :host { display: block; }

            .ewr-audio-grid {
                background: var(--card-bg, #1e293b);
                border: 2px solid var(--card-border, #334155);
                border-radius: 12px;
                overflow: hidden;
            }

            .ewr-audio-grid__info {
                display: flex;
                justify-content: space-between;
                align-items: center;
                padding: 12px 16px;
                background: rgba(0, 0, 0, 0.2);
                border-bottom: 1px solid var(--card-border, #334155);
            }

            .ewr-audio-grid__info-text {
                font-size: 14px;
                color: var(--text-secondary, #94a3b8);
            }

            .ewr-audio-grid__info-text strong {
                color: var(--accent-cyan, #22d3ee);
            }

            .ewr-audio-grid__container {
                border: 1px solid var(--card-border, #334155);
                border-radius: 8px;
                margin: 16px;
                overflow: hidden;
            }

            .ewr-audio-grid__header {
                display: flex;
                align-items: center;
                padding: 14px 16px;
                background: var(--card-bg, #1e293b);
                border-bottom: 2px solid var(--card-border, #334155);
                position: sticky;
                top: 0;
                z-index: 10;
            }

            .ewr-audio-grid__header-cell {
                padding: 0 8px;
                font-weight: 600;
                color: var(--text-primary, #f1f5f9);
                font-size: 14px;
            }

            .ewr-audio-grid__header-cell--view {
                width: 50px;
                flex-shrink: 0;
            }

            .ewr-audio-grid__header-cell--date {
                width: 100px;
                flex-shrink: 0;
            }

            .ewr-audio-grid__header-cell--time {
                width: 80px;
                flex-shrink: 0;
            }

            .ewr-audio-grid__header-cell--staff,
            .ewr-audio-grid__header-cell--customer {
                flex: 1;
                min-width: 100px;
            }

            .ewr-audio-grid__header-cell--mood {
                width: 100px;
                flex-shrink: 0;
            }

            .ewr-audio-grid__header-cell--outcome {
                flex: 1;
                min-width: 120px;
            }

            .ewr-audio-grid__header-cell--filename {
                flex: 1.5;
                min-width: 150px;
            }

            .ewr-audio-grid__header-cell--actions {
                width: 50px;
                flex-shrink: 0;
            }

            .ewr-audio-grid__body {
                overflow-y: auto;
            }

            .ewr-audio-grid__loading {
                display: flex;
                flex-direction: column;
                align-items: center;
                justify-content: center;
                padding: 60px 20px;
                color: var(--text-muted, #64748b);
            }

            .ewr-audio-grid__spinner {
                width: 40px;
                height: 40px;
                border: 3px solid rgba(34, 211, 238, 0.2);
                border-top-color: var(--accent-cyan, #22d3ee);
                border-radius: 50%;
                animation: ewr-spin 1s linear infinite;
                margin-bottom: 16px;
            }

            @keyframes ewr-spin {
                to { transform: rotate(360deg); }
            }

            .ewr-audio-grid__empty {
                display: none;
                align-items: center;
                justify-content: center;
                padding: 60px 20px;
                color: var(--text-muted, #64748b);
                text-align: center;
            }

            ::slotted(ewr-audio-result-row) {
                display: block;
            }

            /* Responsive */
            @media (max-width: 1200px) {
                .ewr-audio-grid__header-cell--time,
                .ewr-audio-grid__header-cell--outcome {
                    display: none;
                }
            }

            @media (max-width: 768px) {
                .ewr-audio-grid__header-cell--customer,
                .ewr-audio-grid__header-cell--filename {
                    display: none;
                }
            }
        `;
    }
}

if (!customElements.get('ewr-audio-results-grid')) {
    customElements.define('ewr-audio-results-grid', EwrAudioResultsGrid);
}

export default EwrAudioResultsGrid;
