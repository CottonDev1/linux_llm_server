/**
 * EWR File Grid Component
 * A table/grid container for file listings with header and selection
 * Uses Shadow DOM for style encapsulation
 *
 * @element ewr-file-grid
 *
 * @attr {boolean} selectable - Show selection column
 * @attr {boolean} expandable - Files can be expanded
 * @attr {string} columns - Comma-separated column names (e.g., "filename,size,type,date,status")
 * @attr {string} empty-message - Message to show when no files
 *
 * @slot - Default slot for ewr-file-row elements
 * @slot header - Custom header content
 *
 * @fires ewr-select-all - When select all is toggled
 * @fires ewr-delete-selected - When delete selected is triggered
 *
 * @example
 * <ewr-file-grid selectable expandable columns="filename,size,date,status">
 *   <ewr-file-row filename="test.mp3" size="4MB" date="Jan 5" status="Ready" selectable expandable></ewr-file-row>
 * </ewr-file-grid>
 */

import { EwrBaseComponent } from '../base/ewr-base-component.js';

export class EwrFileGrid extends EwrBaseComponent {
    static get observedAttributes() {
        return ['selectable', 'expandable', 'columns', 'empty-message'];
    }

    constructor() {
        super();
        this._handleSelectAll = this._handleSelectAll.bind(this);
        this._handleDeleteSelected = this._handleDeleteSelected.bind(this);
        this._allSelected = false;
    }

    onConnected() {
        this._attachListeners();
        this._observeSlot();
    }

    onDisconnected() {
        this._detachListeners();
    }

    onRendered() {
        this._attachListeners();
    }

    _attachListeners() {
        this.$('.ewr-file-grid__select-all')?.addEventListener('change', this._handleSelectAll);
        this.$('.ewr-file-grid__delete-selected')?.addEventListener('click', this._handleDeleteSelected);
    }

    _detachListeners() {
        this.$('.ewr-file-grid__select-all')?.removeEventListener('change', this._handleSelectAll);
        this.$('.ewr-file-grid__delete-selected')?.removeEventListener('click', this._handleDeleteSelected);
    }

    _observeSlot() {
        const slot = this.$('slot:not([name])');
        slot?.addEventListener('slotchange', () => {
            this._updateSelectAllState();
        });
    }

    _handleSelectAll(event) {
        const checked = event.target.checked;
        this._allSelected = checked;

        const rows = this._getFileRows();
        rows.forEach(row => {
            row.selected = checked;
        });

        this.emit('ewr-select-all', {
            selected: checked,
            count: rows.length
        });
    }

    _handleDeleteSelected() {
        const selectedRows = this._getFileRows().filter(row => row.selected);
        const filenames = selectedRows.map(row => row.filename);

        this.emit('ewr-delete-selected', {
            files: filenames,
            count: filenames.length
        });
    }

    _updateSelectAllState() {
        const rows = this._getFileRows();
        const selectedCount = rows.filter(row => row.selected).length;

        const selectAllCheckbox = this.$('.ewr-file-grid__select-all');
        if (selectAllCheckbox) {
            selectAllCheckbox.checked = rows.length > 0 && selectedCount === rows.length;
            selectAllCheckbox.indeterminate = selectedCount > 0 && selectedCount < rows.length;
        }
    }

    _getFileRows() {
        const slot = this.$('slot:not([name])');
        if (!slot) return [];
        return slot.assignedElements().filter(el => el.tagName === 'EWR-FILE-ROW');
    }

    get selectable() { return this.getBoolAttr('selectable'); }
    get expandable() { return this.getBoolAttr('expandable'); }
    get columns() { return this.getAttr('columns', 'filename,size,type,date,status'); }
    get emptyMessage() { return this.getAttr('empty-message', 'No files to display'); }

    getSelectedFiles() {
        return this._getFileRows().filter(row => row.selected);
    }

    selectAll() {
        this._getFileRows().forEach(row => row.selected = true);
        this._updateSelectAllState();
    }

    deselectAll() {
        this._getFileRows().forEach(row => row.selected = false);
        this._updateSelectAllState();
    }

    render() {
        const selectable = this.selectable;
        const columns = this.columns.split(',').map(c => c.trim());

        const columnLabels = {
            filename: 'File Name',
            size: 'Size',
            type: 'Type',
            date: 'Date',
            status: 'Status'
        };

        return `
            <style>
                ${this.getThemeVariables()}
                ${this.getStyles()}
            </style>
            <div class="ewr-file-grid" part="grid">
                <div class="ewr-file-grid__header" part="header">
                    ${selectable ? `
                        <div class="ewr-file-grid__header-cell ewr-file-grid__checkbox-cell">
                            <input type="checkbox" class="ewr-file-grid__select-all" title="Select all" />
                        </div>
                    ` : ''}
                    ${columns.map(col => `
                        <div class="ewr-file-grid__header-cell ewr-file-grid__header-cell--${col}" part="header-${col}">
                            ${this.escapeHtml(columnLabels[col] || col)}
                        </div>
                    `).join('')}
                    <div class="ewr-file-grid__header-cell ewr-file-grid__header-cell--actions" part="header-actions">
                        ${selectable ? `
                            <button class="ewr-file-grid__delete-selected" type="button" title="Delete selected">
                                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                                    <path d="M3 6h18M19 6v14a2 2 0 01-2 2H7a2 2 0 01-2-2V6m3 0V4a2 2 0 012-2h4a2 2 0 012 2v2"/>
                                </svg>
                            </button>
                        ` : ''}
                    </div>
                </div>
                <div class="ewr-file-grid__body" part="body">
                    <slot></slot>
                </div>
                <div class="ewr-file-grid__empty" part="empty">
                    <slot name="empty">
                        <p>${this.escapeHtml(this.emptyMessage)}</p>
                    </slot>
                </div>
            </div>
        `;
    }

    getStyles() {
        return `
            :host { display: block; }

            .ewr-file-grid {
                width: 100%;
                font-size: 14px;
                background: var(--bg-secondary);
                border: 1px solid var(--border-primary);
                border-radius: 8px;
                overflow: hidden;
            }

            .ewr-file-grid__header {
                display: flex;
                align-items: center;
                padding: 12px 16px;
                background: var(--bg-secondary);
                border-bottom: 1px solid var(--border-primary);
                position: sticky;
                top: 0;
                z-index: 10;
            }

            .ewr-file-grid__header-cell {
                padding: 0 8px;
                font-weight: 600;
                color: var(--text-tertiary);
                text-transform: uppercase;
                font-size: 11px;
                letter-spacing: 0.5px;
            }

            .ewr-file-grid__checkbox-cell {
                width: 40px;
                text-align: center;
                flex-shrink: 0;
            }

            .ewr-file-grid__select-all {
                width: 18px;
                height: 18px;
                cursor: pointer;
                accent-color: var(--accent-primary);
            }

            .ewr-file-grid__header-cell--filename {
                flex: 2;
            }

            .ewr-file-grid__header-cell--size,
            .ewr-file-grid__header-cell--type,
            .ewr-file-grid__header-cell--date,
            .ewr-file-grid__header-cell--status {
                flex: 1;
            }

            .ewr-file-grid__header-cell--actions {
                flex-shrink: 0;
                width: 80px;
                text-align: right;
            }

            .ewr-file-grid__delete-selected {
                width: 32px;
                height: 32px;
                padding: 6px;
                border: none;
                border-radius: 6px;
                background: transparent;
                color: var(--text-tertiary);
                cursor: pointer;
                transition: all 0.2s ease;
                display: inline-flex;
                align-items: center;
                justify-content: center;
            }

            .ewr-file-grid__delete-selected svg {
                width: 16px;
                height: 16px;
            }

            .ewr-file-grid__delete-selected:hover {
                background: rgba(239, 68, 68, 0.15);
                color: var(--accent-danger);
            }

            .ewr-file-grid__body {
                max-height: 400px;
                overflow-y: auto;
            }

            .ewr-file-grid__body:empty + .ewr-file-grid__empty {
                display: flex;
            }

            .ewr-file-grid__body:not(:empty) + .ewr-file-grid__empty {
                display: none;
            }

            ::slotted(ewr-file-row) {
                display: block;
            }

            .ewr-file-grid__empty {
                display: none;
                align-items: center;
                justify-content: center;
                padding: 40px;
                color: var(--text-tertiary);
                text-align: center;
            }

            .ewr-file-grid__empty p {
                margin: 0;
            }

            /* Responsive */
            @media (max-width: 768px) {
                .ewr-file-grid__header-cell--type,
                .ewr-file-grid__header-cell--date {
                    display: none;
                }
            }
        `;
    }
}

if (!customElements.get('ewr-file-grid')) {
    customElements.define('ewr-file-grid', EwrFileGrid);
}

export default EwrFileGrid;
