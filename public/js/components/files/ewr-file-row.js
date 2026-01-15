/**
 * EWR File Row Component
 * A table row for displaying file information with selection and actions
 * Uses Shadow DOM for style encapsulation
 *
 * @element ewr-file-row
 *
 * @attr {string} filename - File name to display
 * @attr {string} size - File size (formatted)
 * @attr {string} type - File type/extension
 * @attr {string} date - Date string
 * @attr {string} status - Status text
 * @attr {string} status-variant - Status variant: success, error, warning, info
 * @attr {boolean} selected - Whether the row is selected
 * @attr {boolean} expanded - Whether the row is expanded
 * @attr {boolean} selectable - Show selection checkbox
 * @attr {boolean} expandable - Show expand indicator
 *
 * @fires ewr-select - When selection changes
 * @fires ewr-expand - When row is expanded/collapsed
 * @fires ewr-click - When row is clicked
 * @fires ewr-delete - When delete action is triggered
 *
 * @example
 * <ewr-file-row
 *   filename="recording.mp3"
 *   size="4.2 MB"
 *   type="audio/mpeg"
 *   date="Jan 5, 2025"
 *   status="Ready"
 *   status-variant="success"
 *   selectable
 *   expandable
 * ></ewr-file-row>
 */

import { EwrBaseComponent } from '../base/ewr-base-component.js';

export class EwrFileRow extends EwrBaseComponent {
    static get observedAttributes() {
        return ['filename', 'size', 'type', 'date', 'status', 'status-variant', 'selected', 'expanded', 'selectable', 'expandable'];
    }

    constructor() {
        super();
        this._handleClick = this._handleClick.bind(this);
        this._handleCheckboxChange = this._handleCheckboxChange.bind(this);
        this._handleDeleteClick = this._handleDeleteClick.bind(this);
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
        this.$('.ewr-file-row')?.addEventListener('click', this._handleClick);
        this.$('.ewr-file-row__checkbox')?.addEventListener('change', this._handleCheckboxChange);
        this.$('.ewr-file-row__delete')?.addEventListener('click', this._handleDeleteClick);
    }

    _detachListeners() {
        this.$('.ewr-file-row')?.removeEventListener('click', this._handleClick);
        this.$('.ewr-file-row__checkbox')?.removeEventListener('change', this._handleCheckboxChange);
        this.$('.ewr-file-row__delete')?.removeEventListener('click', this._handleDeleteClick);
    }

    _handleClick(event) {
        // Don't trigger click for checkbox or delete button
        if (event.target.closest('.ewr-file-row__checkbox') ||
            event.target.closest('.ewr-file-row__delete')) {
            return;
        }

        if (this.expandable) {
            this.toggle();
            this.emit('ewr-expand', {
                filename: this.filename,
                expanded: this.expanded
            });
        }

        this.emit('ewr-click', {
            filename: this.filename,
            expanded: this.expanded
        });
    }

    _handleCheckboxChange(event) {
        event.stopPropagation();
        this.selected = event.target.checked;
        this.emit('ewr-select', {
            filename: this.filename,
            selected: this.selected
        });
    }

    _handleDeleteClick(event) {
        event.stopPropagation();
        this.emit('ewr-delete', {
            filename: this.filename
        });
    }

    get filename() { return this.getAttr('filename', ''); }
    get size() { return this.getAttr('size', ''); }
    get type() { return this.getAttr('type', ''); }
    get date() { return this.getAttr('date', ''); }
    get status() { return this.getAttr('status', ''); }
    get statusVariant() { return this.getAttr('status-variant', 'default'); }
    get selected() { return this.getBoolAttr('selected'); }
    set selected(value) { this.setBoolAttr('selected', value); }
    get expanded() { return this.getBoolAttr('expanded'); }
    set expanded(value) { this.setBoolAttr('expanded', value); }
    get selectable() { return this.getBoolAttr('selectable'); }
    get expandable() { return this.getBoolAttr('expandable'); }

    toggle() {
        this.expanded = !this.expanded;
    }

    expand() {
        this.expanded = true;
    }

    collapse() {
        this.expanded = false;
    }

    render() {
        const filename = this.filename;
        const size = this.size;
        const type = this.type;
        const date = this.date;
        const status = this.status;
        const statusVariant = this.statusVariant;
        const selected = this.selected;
        const expanded = this.expanded;
        const selectable = this.selectable;
        const expandable = this.expandable;

        const rowClasses = [
            'ewr-file-row',
            selected ? 'selected' : '',
            expanded ? 'expanded' : ''
        ].filter(Boolean).join(' ');

        return `
            <style>
                ${this.getThemeVariables()}
                ${this.getStyles()}
            </style>
            <div class="${rowClasses}" part="row">
                ${selectable ? `
                    <div class="ewr-file-row__cell ewr-file-row__checkbox-cell">
                        <input type="checkbox" class="ewr-file-row__checkbox" ${selected ? 'checked' : ''} part="checkbox" />
                    </div>
                ` : ''}
                <div class="ewr-file-row__cell ewr-file-row__filename" part="filename">
                    ${expandable ? `<span class="ewr-file-row__expand-indicator">${expanded ? '&#9662;' : '&#9656;'}</span>` : ''}
                    ${this.escapeHtml(filename)}
                </div>
                ${size ? `<div class="ewr-file-row__cell ewr-file-row__size" part="size">${this.escapeHtml(size)}</div>` : ''}
                ${type ? `<div class="ewr-file-row__cell ewr-file-row__type" part="type">${this.escapeHtml(type)}</div>` : ''}
                ${date ? `<div class="ewr-file-row__cell ewr-file-row__date" part="date">${this.escapeHtml(date)}</div>` : ''}
                ${status ? `
                    <div class="ewr-file-row__cell ewr-file-row__status" part="status">
                        <span class="ewr-file-row__status-pill ewr-file-row__status-pill--${statusVariant}">${this.escapeHtml(status)}</span>
                    </div>
                ` : ''}
                <div class="ewr-file-row__cell ewr-file-row__actions" part="actions">
                    <button class="ewr-file-row__delete" type="button" title="Delete file">
                        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                            <path d="M3 6h18M19 6v14a2 2 0 01-2 2H7a2 2 0 01-2-2V6m3 0V4a2 2 0 012-2h4a2 2 0 012 2v2"/>
                        </svg>
                    </button>
                    <slot name="actions"></slot>
                </div>
            </div>
            <div class="ewr-file-row__expanded-content ${expanded ? 'visible' : ''}" part="expanded">
                <slot></slot>
            </div>
        `;
    }

    getStyles() {
        return `
            :host { display: block; }

            .ewr-file-row {
                display: flex;
                align-items: center;
                padding: 12px 16px;
                background: transparent;
                border-bottom: 1px solid var(--bg-secondary);
                cursor: pointer;
                transition: all 0.2s ease;
            }

            .ewr-file-row:hover {
                background: rgba(0, 212, 255, 0.05);
            }

            .ewr-file-row.selected {
                background: rgba(0, 212, 255, 0.1);
            }

            .ewr-file-row.expanded {
                background: rgba(59, 130, 246, 0.05);
            }

            .ewr-file-row__cell {
                padding: 0 8px;
                color: var(--text-secondary);
            }

            .ewr-file-row__checkbox-cell {
                width: 40px;
                text-align: center;
                flex-shrink: 0;
            }

            .ewr-file-row__checkbox {
                width: 18px;
                height: 18px;
                cursor: pointer;
                accent-color: var(--accent-primary);
            }

            .ewr-file-row__filename {
                flex: 2;
                font-weight: 500;
                color: var(--text-primary);
                white-space: nowrap;
                overflow: hidden;
                text-overflow: ellipsis;
            }

            .ewr-file-row__filename:hover {
                color: #00d4ff;
            }

            .ewr-file-row__expand-indicator {
                display: inline-block;
                margin-right: 8px;
                font-size: 12px;
                color: var(--text-tertiary);
                transition: transform 0.3s ease;
            }

            .ewr-file-row.expanded .ewr-file-row__expand-indicator {
                color: var(--accent-primary);
            }

            .ewr-file-row__size,
            .ewr-file-row__type,
            .ewr-file-row__date {
                flex: 1;
                font-size: 13px;
            }

            .ewr-file-row__status {
                flex: 1;
            }

            .ewr-file-row__status-pill {
                display: inline-block;
                padding: 4px 10px;
                font-size: 11px;
                font-weight: 600;
                border-radius: 12px;
                text-transform: uppercase;
                letter-spacing: 0.3px;
            }

            .ewr-file-row__status-pill--success {
                background: rgba(16, 185, 129, 0.15);
                color: #10b981;
            }

            .ewr-file-row__status-pill--error {
                background: rgba(239, 68, 68, 0.15);
                color: #ef4444;
            }

            .ewr-file-row__status-pill--warning {
                background: rgba(245, 158, 11, 0.15);
                color: #f59e0b;
            }

            .ewr-file-row__status-pill--info {
                background: rgba(59, 130, 246, 0.15);
                color: #3b82f6;
            }

            .ewr-file-row__status-pill--default {
                background: rgba(148, 163, 184, 0.15);
                color: #94a3b8;
            }

            .ewr-file-row__actions {
                flex-shrink: 0;
                display: flex;
                gap: 8px;
                justify-content: flex-end;
            }

            .ewr-file-row__delete {
                width: 32px;
                height: 32px;
                padding: 6px;
                border: none;
                border-radius: 6px;
                background: transparent;
                color: var(--text-tertiary);
                cursor: pointer;
                transition: all 0.2s ease;
                display: flex;
                align-items: center;
                justify-content: center;
            }

            .ewr-file-row__delete svg {
                width: 16px;
                height: 16px;
            }

            .ewr-file-row__delete:hover {
                background: rgba(239, 68, 68, 0.15);
                color: var(--accent-danger);
            }

            .ewr-file-row__expanded-content {
                display: none;
                padding: 0 16px 16px;
                background: var(--bg-secondary);
                border-bottom: 1px solid var(--border-primary);
            }

            .ewr-file-row__expanded-content.visible {
                display: block;
            }

            /* Responsive */
            @media (max-width: 768px) {
                .ewr-file-row__type,
                .ewr-file-row__date {
                    display: none;
                }
            }
        `;
    }
}

if (!customElements.get('ewr-file-row')) {
    customElements.define('ewr-file-row', EwrFileRow);
}

export default EwrFileRow;
