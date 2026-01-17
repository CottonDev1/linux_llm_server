/**
 * EWR Audio Result Row Component
 * A row displaying an audio analysis result with view/delete actions
 * Uses Shadow DOM for style encapsulation
 *
 * @element ewr-audio-result-row
 *
 * @attr {string} result-id - Unique ID of the analysis
 * @attr {string} date - Call date
 * @attr {string} time - Call time
 * @attr {string} staff - Staff member name
 * @attr {string} customer - Customer name
 * @attr {string} mood - Primary emotion/mood
 * @attr {string} outcome - Call outcome
 * @attr {string} filename - Audio filename
 *
 * @fires ewr-view - When view button is clicked
 * @fires ewr-delete - When delete button is clicked
 *
 * @example
 * <ewr-audio-result-row
 *   result-id="abc123"
 *   date="2024-01-15"
 *   time="10:30:00"
 *   staff="John Doe"
 *   customer="ACME Corp"
 *   mood="HAPPY"
 *   outcome="Issue Resolved"
 *   filename="call_20240115.mp3">
 * </ewr-audio-result-row>
 */

import { EwrBaseComponent } from '../base/ewr-base-component.js';

export class EwrAudioResultRow extends EwrBaseComponent {
    static get observedAttributes() {
        return ['result-id', 'date', 'time', 'staff', 'customer', 'mood', 'outcome', 'filename'];
    }

    constructor() {
        super();
        this._handleView = this._handleView.bind(this);
        this._handleDelete = this._handleDelete.bind(this);
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
        this.$('.ewr-audio-row__view')?.addEventListener('click', this._handleView);
        this.$('.ewr-audio-row__delete')?.addEventListener('click', this._handleDelete);
        this.$('.ewr-audio-row__filename')?.addEventListener('click', this._handleView);
    }

    _detachListeners() {
        this.$('.ewr-audio-row__view')?.removeEventListener('click', this._handleView);
        this.$('.ewr-audio-row__delete')?.removeEventListener('click', this._handleDelete);
        this.$('.ewr-audio-row__filename')?.removeEventListener('click', this._handleView);
    }

    _handleView() {
        this.emit('ewr-view', {
            id: this.resultId,
            filename: this.filename
        });
    }

    _handleDelete() {
        this.emit('ewr-delete', {
            id: this.resultId,
            filename: this.filename
        });
    }

    get resultId() { return this.getAttr('result-id', ''); }
    get date() { return this.getAttr('date', 'N/A'); }
    get time() { return this.getAttr('time', 'N/A'); }
    get staff() { return this.getAttr('staff', 'N/A'); }
    get customer() { return this.getAttr('customer', 'N/A'); }
    get mood() { return this.getAttr('mood', 'NEUTRAL'); }
    get outcome() { return this.getAttr('outcome', 'N/A'); }
    get filename() { return this.getAttr('filename', 'Unknown'); }

    render() {
        const mood = this.mood.toUpperCase();
        const moodClass = `ewr-audio-row__mood--${mood}`;

        return `
            <style>
                ${this.getThemeVariables()}
                ${this.getStyles()}
            </style>
            <div class="ewr-audio-row" part="row">
                <div class="ewr-audio-row__cell ewr-audio-row__cell--view">
                    <button class="ewr-audio-row__view" type="button" title="View Details">
                        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                            <path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"/>
                            <circle cx="12" cy="12" r="3"/>
                        </svg>
                    </button>
                </div>
                <div class="ewr-audio-row__cell ewr-audio-row__cell--date">${this.escapeHtml(this.date)}</div>
                <div class="ewr-audio-row__cell ewr-audio-row__cell--time">${this.escapeHtml(this.time)}</div>
                <div class="ewr-audio-row__cell ewr-audio-row__cell--staff">${this.escapeHtml(this.staff)}</div>
                <div class="ewr-audio-row__cell ewr-audio-row__cell--customer">${this.escapeHtml(this.customer)}</div>
                <div class="ewr-audio-row__cell ewr-audio-row__cell--mood">
                    <span class="ewr-audio-row__mood ${moodClass}">${this.escapeHtml(mood)}</span>
                </div>
                <div class="ewr-audio-row__cell ewr-audio-row__cell--outcome">${this.escapeHtml(this.outcome)}</div>
                <div class="ewr-audio-row__cell ewr-audio-row__cell--filename">
                    <span class="ewr-audio-row__filename">${this.escapeHtml(this.filename)}</span>
                </div>
                <div class="ewr-audio-row__cell ewr-audio-row__cell--actions">
                    <button class="ewr-audio-row__delete" type="button" title="Delete">
                        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                            <path d="M3 6h18M19 6v14a2 2 0 01-2 2H7a2 2 0 01-2-2V6m3 0V4a2 2 0 012-2h4a2 2 0 012 2v2"/>
                        </svg>
                    </button>
                </div>
            </div>
        `;
    }

    getStyles() {
        return `
            :host { display: block; }

            .ewr-audio-row {
                display: flex;
                align-items: center;
                padding: 12px 16px;
                border-bottom: 1px solid rgba(51, 65, 85, 0.5);
                transition: background 0.2s ease;
            }

            .ewr-audio-row:hover {
                background: rgba(34, 211, 238, 0.05);
            }

            .ewr-audio-row__cell {
                padding: 0 8px;
                color: var(--text-secondary);
                font-size: 14px;
                overflow: hidden;
                text-overflow: ellipsis;
                white-space: nowrap;
            }

            .ewr-audio-row__cell--view {
                width: 50px;
                flex-shrink: 0;
            }

            .ewr-audio-row__cell--date {
                width: 100px;
                flex-shrink: 0;
            }

            .ewr-audio-row__cell--time {
                width: 80px;
                flex-shrink: 0;
            }

            .ewr-audio-row__cell--staff,
            .ewr-audio-row__cell--customer {
                flex: 1;
                min-width: 100px;
            }

            .ewr-audio-row__cell--mood {
                width: 100px;
                flex-shrink: 0;
            }

            .ewr-audio-row__cell--outcome {
                flex: 1;
                min-width: 120px;
            }

            .ewr-audio-row__cell--filename {
                flex: 1.5;
                min-width: 150px;
            }

            .ewr-audio-row__cell--actions {
                width: 50px;
                flex-shrink: 0;
                text-align: right;
            }

            /* View button */
            .ewr-audio-row__view {
                width: 32px;
                height: 32px;
                padding: 6px;
                border: 1px solid rgba(59, 130, 246, 0.3);
                border-radius: 6px;
                background: rgba(59, 130, 246, 0.2);
                color: #60a5fa;
                cursor: pointer;
                transition: all 0.2s ease;
                display: inline-flex;
                align-items: center;
                justify-content: center;
            }

            .ewr-audio-row__view svg {
                width: 16px;
                height: 16px;
            }

            .ewr-audio-row__view:hover {
                background: rgba(59, 130, 246, 0.3);
                border-color: #60a5fa;
            }

            /* Delete button */
            .ewr-audio-row__delete {
                width: 32px;
                height: 32px;
                padding: 6px;
                border: none;
                border-radius: 6px;
                background: transparent;
                color: #f87171;
                cursor: pointer;
                transition: all 0.2s ease;
                display: inline-flex;
                align-items: center;
                justify-content: center;
            }

            .ewr-audio-row__delete svg {
                width: 16px;
                height: 16px;
            }

            .ewr-audio-row__delete:hover {
                background: rgba(239, 68, 68, 0.2);
            }

            /* Filename link */
            .ewr-audio-row__filename {
                color: #60a5fa;
                text-decoration: underline;
                cursor: pointer;
                transition: color 0.2s;
            }

            .ewr-audio-row__filename:hover {
                color: #93c5fd;
            }

            /* Mood badges */
            .ewr-audio-row__mood {
                display: inline-block;
                padding: 4px 10px;
                border-radius: 12px;
                font-size: 12px;
                font-weight: 600;
            }

            .ewr-audio-row__mood--HAPPY { background: rgba(34, 197, 94, 0.2); color: #4ade80; }
            .ewr-audio-row__mood--SAD { background: rgba(59, 130, 246, 0.2); color: #60a5fa; }
            .ewr-audio-row__mood--ANGRY { background: rgba(239, 68, 68, 0.2); color: #f87171; }
            .ewr-audio-row__mood--NEUTRAL { background: rgba(148, 163, 184, 0.2); color: #cbd5e1; }
            .ewr-audio-row__mood--FEARFUL { background: rgba(168, 85, 247, 0.2); color: #c084fc; }
            .ewr-audio-row__mood--DISGUSTED { background: rgba(234, 179, 8, 0.2); color: #facc15; }
            .ewr-audio-row__mood--SURPRISED { background: rgba(236, 72, 153, 0.2); color: #f472b6; }

            /* Responsive */
            @media (max-width: 1200px) {
                .ewr-audio-row__cell--time,
                .ewr-audio-row__cell--outcome {
                    display: none;
                }
            }

            @media (max-width: 768px) {
                .ewr-audio-row__cell--customer,
                .ewr-audio-row__cell--filename {
                    display: none;
                }
            }
        `;
    }
}

if (!customElements.get('ewr-audio-result-row')) {
    customElements.define('ewr-audio-result-row', EwrAudioResultRow);
}

export default EwrAudioResultRow;
