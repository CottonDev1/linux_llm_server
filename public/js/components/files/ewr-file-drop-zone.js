/**
 * EWR File Drop Zone Component
 * Drag-and-drop area for file uploads with visual feedback
 * Uses Shadow DOM for style encapsulation
 *
 * @element ewr-file-drop-zone
 *
 * @attr {string} accept - Accepted file types (e.g., ".mp3,.wav,audio/*")
 * @attr {boolean} multiple - Allow multiple files
 * @attr {boolean} disabled - Disable the drop zone
 * @attr {string} icon - Custom icon/emoji (default: cloud upload)
 * @attr {string} text - Main instruction text
 * @attr {string} subtext - Secondary help text
 *
 * @fires ewr-files-selected - When files are selected via drop or click
 * @fires ewr-dragover - When files are dragged over the zone
 * @fires ewr-dragleave - When files leave the drop zone
 *
 * @example
 * <ewr-file-drop-zone
 *   accept="audio/*"
 *   multiple
 *   text="Drop audio files here"
 *   subtext="or click to browse"
 * ></ewr-file-drop-zone>
 */

import { EwrBaseComponent } from '../base/ewr-base-component.js';

export class EwrFileDropZone extends EwrBaseComponent {
    static get observedAttributes() {
        return ['accept', 'multiple', 'disabled', 'icon', 'text', 'subtext'];
    }

    constructor() {
        super();
        this._handleDragOver = this._handleDragOver.bind(this);
        this._handleDragLeave = this._handleDragLeave.bind(this);
        this._handleDrop = this._handleDrop.bind(this);
        this._handleClick = this._handleClick.bind(this);
        this._handleFileChange = this._handleFileChange.bind(this);
        this._files = [];
    }

    onConnected() {
        this._attachListeners();
    }

    onDisconnected() {
        this._detachListeners();
    }

    onRendered() {
        this._attachListeners();
        this._updateFileState();
    }

    _attachListeners() {
        const zone = this.$('.ewr-file-drop-zone');
        const input = this.$('.ewr-file-drop-zone__input');

        zone?.addEventListener('dragover', this._handleDragOver);
        zone?.addEventListener('dragleave', this._handleDragLeave);
        zone?.addEventListener('drop', this._handleDrop);
        zone?.addEventListener('click', this._handleClick);
        input?.addEventListener('change', this._handleFileChange);
    }

    _detachListeners() {
        const zone = this.$('.ewr-file-drop-zone');
        const input = this.$('.ewr-file-drop-zone__input');

        zone?.removeEventListener('dragover', this._handleDragOver);
        zone?.removeEventListener('dragleave', this._handleDragLeave);
        zone?.removeEventListener('drop', this._handleDrop);
        zone?.removeEventListener('click', this._handleClick);
        input?.removeEventListener('change', this._handleFileChange);
    }

    _handleDragOver(event) {
        event.preventDefault();
        event.stopPropagation();
        if (this.disabled) return;

        this.$('.ewr-file-drop-zone')?.classList.add('dragover');
        this.emit('ewr-dragover');
    }

    _handleDragLeave(event) {
        event.preventDefault();
        event.stopPropagation();

        this.$('.ewr-file-drop-zone')?.classList.remove('dragover');
        this.emit('ewr-dragleave');
    }

    _handleDrop(event) {
        event.preventDefault();
        event.stopPropagation();
        if (this.disabled) return;

        this.$('.ewr-file-drop-zone')?.classList.remove('dragover');

        const files = Array.from(event.dataTransfer.files);
        this._processFiles(files);
    }

    _handleClick() {
        if (this.disabled) return;
        this.$('.ewr-file-drop-zone__input')?.click();
    }

    _handleFileChange(event) {
        const files = Array.from(event.target.files);
        this._processFiles(files);
        // Reset input so same file can be selected again
        event.target.value = '';
    }

    _processFiles(files) {
        if (!files.length) return;

        // Filter by accept if specified
        const accept = this.accept;
        let filteredFiles = files;

        if (accept) {
            const acceptTypes = accept.split(',').map(t => t.trim().toLowerCase());
            filteredFiles = files.filter(file => {
                const ext = '.' + file.name.split('.').pop().toLowerCase();
                const mimeType = file.type.toLowerCase();

                return acceptTypes.some(type => {
                    if (type.startsWith('.')) {
                        return ext === type;
                    }
                    if (type.endsWith('/*')) {
                        return mimeType.startsWith(type.replace('/*', '/'));
                    }
                    return mimeType === type;
                });
            });
        }

        if (!this.multiple && filteredFiles.length > 1) {
            filteredFiles = [filteredFiles[0]];
        }

        this._files = filteredFiles;
        this._updateFileState();

        this.emit('ewr-files-selected', {
            files: filteredFiles,
            count: filteredFiles.length
        });
    }

    _updateFileState() {
        const zone = this.$('.ewr-file-drop-zone');
        if (this._files.length > 0) {
            zone?.classList.add('has-files');
        } else {
            zone?.classList.remove('has-files');
        }
    }

    get accept() { return this.getAttr('accept', ''); }
    get multiple() { return this.getBoolAttr('multiple'); }
    get disabled() { return this.getBoolAttr('disabled'); }
    set disabled(value) { this.setBoolAttr('disabled', value); }
    get icon() { return this.getAttr('icon', ''); }
    get text() { return this.getAttr('text', 'Drop files here'); }
    get subtext() { return this.getAttr('subtext', 'or click to browse'); }

    get files() { return this._files; }

    clear() {
        this._files = [];
        this._updateFileState();
    }

    render() {
        const accept = this.accept;
        const multiple = this.multiple;
        const disabled = this.disabled;
        const text = this.text;
        const subtext = this.subtext;
        const hasFiles = this._files.length > 0;

        const zoneClasses = [
            'ewr-file-drop-zone',
            disabled ? 'disabled' : '',
            hasFiles ? 'has-files' : ''
        ].filter(Boolean).join(' ');

        // SVG cloud upload icon
        const cloudIcon = `
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
                <polyline points="17 8 12 3 7 8"/>
                <line x1="12" y1="3" x2="12" y2="15"/>
            </svg>
        `;

        return `
            <style>
                ${this.getThemeVariables()}
                ${this.getStyles()}
            </style>
            <div class="${zoneClasses}" part="zone">
                <input
                    type="file"
                    class="ewr-file-drop-zone__input"
                    ${accept ? `accept="${this.escapeHtml(accept)}"` : ''}
                    ${multiple ? 'multiple' : ''}
                    ${disabled ? 'disabled' : ''}
                />
                <div class="ewr-file-drop-zone__icon" part="icon">
                    ${this.icon ? this.escapeHtml(this.icon) : cloudIcon}
                </div>
                <div class="ewr-file-drop-zone__text" part="text">${this.escapeHtml(text)}</div>
                ${subtext ? `<div class="ewr-file-drop-zone__subtext" part="subtext">${this.escapeHtml(subtext)}</div>` : ''}
                ${hasFiles ? `
                    <div class="ewr-file-drop-zone__count" part="count">
                        ${this._files.length} file${this._files.length !== 1 ? 's' : ''} selected
                    </div>
                ` : ''}
            </div>
        `;
    }

    getStyles() {
        return `
            :host { display: block; }

            .ewr-file-drop-zone {
                background: var(--bg-secondary);
                border: 2px dashed var(--border-primary);
                border-radius: 8px;
                padding: 40px;
                text-align: center;
                transition: all 0.3s ease;
                cursor: pointer;
                min-height: 150px;
                display: flex;
                flex-direction: column;
                align-items: center;
                justify-content: center;
                gap: 12px;
                position: relative;
            }

            .ewr-file-drop-zone:hover:not(.disabled) {
                border-color: var(--border-secondary);
                background: rgba(30, 41, 59, 0.8);
            }

            .ewr-file-drop-zone.dragover {
                border-color: #00d4ff;
                border-style: solid;
                background: rgba(0, 212, 255, 0.05);
                transform: scale(1.02);
            }

            .ewr-file-drop-zone.has-files {
                border-color: var(--accent-success);
                border-style: solid;
                background: rgba(16, 185, 129, 0.05);
            }

            .ewr-file-drop-zone.disabled {
                opacity: 0.5;
                cursor: not-allowed;
            }

            .ewr-file-drop-zone__input {
                position: absolute;
                width: 0;
                height: 0;
                opacity: 0;
                pointer-events: none;
            }

            .ewr-file-drop-zone__icon {
                font-size: 48px;
                color: var(--accent-primary);
                margin-bottom: 8px;
            }

            .ewr-file-drop-zone__icon svg {
                width: 48px;
                height: 48px;
            }

            .ewr-file-drop-zone.has-files .ewr-file-drop-zone__icon {
                color: var(--accent-success);
            }

            .ewr-file-drop-zone__text {
                font-size: 16px;
                color: var(--text-secondary);
                font-weight: 500;
            }

            .ewr-file-drop-zone__subtext {
                font-size: 14px;
                color: var(--text-tertiary);
            }

            .ewr-file-drop-zone__count {
                font-size: 13px;
                color: var(--accent-success);
                font-weight: 600;
                margin-top: 8px;
            }

            /* Responsive */
            @media (max-width: 480px) {
                .ewr-file-drop-zone {
                    padding: 24px;
                    min-height: 120px;
                }

                .ewr-file-drop-zone__icon {
                    font-size: 36px;
                }

                .ewr-file-drop-zone__icon svg {
                    width: 36px;
                    height: 36px;
                }

                .ewr-file-drop-zone__text {
                    font-size: 14px;
                }
            }
        `;
    }
}

if (!customElements.get('ewr-file-drop-zone')) {
    customElements.define('ewr-file-drop-zone', EwrFileDropZone);
}

export default EwrFileDropZone;
