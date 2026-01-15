/**
 * EWR Audio Player Component
 * An audio player card with metadata display and action buttons
 * Uses Shadow DOM for style encapsulation
 *
 * @element ewr-audio-player
 *
 * @attr {string} src - Audio file URL
 * @attr {string} title - Audio file title
 * @attr {string} info - Additional info text (e.g., duration, file size)
 * @attr {string} metadata - JSON string of metadata key-value pairs
 * @attr {boolean} compact - Compact mode (no footer)
 * @attr {boolean} expandable - Can be expanded/collapsed
 * @attr {boolean} expanded - Current expanded state
 * @attr {boolean} autoplay - Autoplay when loaded
 * @attr {boolean} show-analyze - Show analyze button
 *
 * @slot footer - Footer content (buttons)
 * @slot metadata - Custom metadata display
 *
 * @fires ewr-play - When audio starts playing
 * @fires ewr-pause - When audio is paused
 * @fires ewr-ended - When audio ends
 * @fires ewr-analyze - When analyze button is clicked
 * @fires ewr-error - When audio fails to load
 *
 * @example
 * <ewr-audio-player
 *   src="/audio/recording.mp3"
 *   title="Meeting Recording"
 *   info="4.2 MB - 12:34"
 *   metadata='{"Duration":"12:34","Format":"MP3","Sample Rate":"44.1 kHz"}'
 *   show-analyze
 * ></ewr-audio-player>
 */

import { EwrBaseComponent } from '../base/ewr-base-component.js';

export class EwrAudioPlayer extends EwrBaseComponent {
    static get observedAttributes() {
        return ['src', 'title', 'info', 'metadata', 'compact', 'expandable', 'expanded', 'autoplay', 'show-analyze'];
    }

    constructor() {
        super();
        this._handlePlay = this._handlePlay.bind(this);
        this._handlePause = this._handlePause.bind(this);
        this._handleEnded = this._handleEnded.bind(this);
        this._handleError = this._handleError.bind(this);
        this._handleAnalyze = this._handleAnalyze.bind(this);
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
        const audio = this.$('.ewr-audio-player__audio');
        audio?.addEventListener('play', this._handlePlay);
        audio?.addEventListener('pause', this._handlePause);
        audio?.addEventListener('ended', this._handleEnded);
        audio?.addEventListener('error', this._handleError);

        this.$('.ewr-audio-player__analyze')?.addEventListener('click', this._handleAnalyze);
        this.$('.ewr-audio-player__toggle')?.addEventListener('click', this._handleToggle);
    }

    _detachListeners() {
        const audio = this.$('.ewr-audio-player__audio');
        audio?.removeEventListener('play', this._handlePlay);
        audio?.removeEventListener('pause', this._handlePause);
        audio?.removeEventListener('ended', this._handleEnded);
        audio?.removeEventListener('error', this._handleError);

        this.$('.ewr-audio-player__analyze')?.removeEventListener('click', this._handleAnalyze);
        this.$('.ewr-audio-player__toggle')?.removeEventListener('click', this._handleToggle);
    }

    _handlePlay() {
        this.emit('ewr-play', { src: this.src, title: this.title });
    }

    _handlePause() {
        this.emit('ewr-pause', { src: this.src, title: this.title });
    }

    _handleEnded() {
        this.emit('ewr-ended', { src: this.src, title: this.title });
    }

    _handleError(event) {
        this.emit('ewr-error', {
            src: this.src,
            title: this.title,
            error: event.target?.error?.message || 'Failed to load audio'
        });
    }

    _handleAnalyze() {
        this.emit('ewr-analyze', { src: this.src, title: this.title });
    }

    _handleToggle() {
        this.expanded = !this.expanded;
    }

    get src() { return this.getAttr('src', ''); }
    set src(value) { this.setAttribute('src', value); }
    get title() { return this.getAttr('title', ''); }
    get info() { return this.getAttr('info', ''); }
    get metadata() { return this.parseJsonAttr('metadata', {}); }
    get compact() { return this.getBoolAttr('compact'); }
    get expandable() { return this.getBoolAttr('expandable'); }
    get expanded() { return this.getBoolAttr('expanded'); }
    set expanded(value) { this.setBoolAttr('expanded', value); }
    get autoplay() { return this.getBoolAttr('autoplay'); }
    get showAnalyze() { return this.getBoolAttr('show-analyze'); }

    play() {
        this.$('.ewr-audio-player__audio')?.play();
    }

    pause() {
        this.$('.ewr-audio-player__audio')?.pause();
    }

    toggle() {
        this.expanded = !this.expanded;
    }

    render() {
        const src = this.src;
        const title = this.title;
        const info = this.info;
        const metadata = this.metadata;
        const compact = this.compact;
        const expandable = this.expandable;
        const expanded = this.expanded;
        const autoplay = this.autoplay;
        const showAnalyze = this.showAnalyze;

        const cardClasses = [
            'ewr-audio-player',
            compact ? 'compact' : '',
            expandable ? 'expandable' : '',
            expanded ? 'expanded' : ''
        ].filter(Boolean).join(' ');

        const metadataEntries = Object.entries(metadata);

        return `
            <style>
                ${this.getThemeVariables()}
                ${this.getStyles()}
            </style>
            <div class="${cardClasses}" part="card">
                <div class="ewr-audio-player__header" part="header">
                    <h3 class="ewr-audio-player__title">${this.escapeHtml(title)}</h3>
                    ${info ? `<span class="ewr-audio-player__info">${this.escapeHtml(info)}</span>` : ''}
                    ${expandable ? `
                        <button class="ewr-audio-player__toggle" type="button" title="${expanded ? 'Collapse' : 'Expand'}">
                            ${expanded ? '&#9650;' : '&#9660;'}
                        </button>
                    ` : ''}
                </div>

                <div class="ewr-audio-player__content ${expandable && !expanded ? 'hidden' : ''}">
                    ${metadataEntries.length > 0 ? `
                        <div class="ewr-audio-player__metadata" part="metadata">
                            <slot name="metadata">
                                ${metadataEntries.map(([label, value]) => `
                                    <div class="ewr-audio-player__metadata-item">
                                        <span class="ewr-audio-player__metadata-label">${this.escapeHtml(label)}</span>
                                        <span class="ewr-audio-player__metadata-value">${this.escapeHtml(String(value))}</span>
                                    </div>
                                `).join('')}
                            </slot>
                        </div>
                    ` : ''}

                    ${src ? `
                        <audio
                            class="ewr-audio-player__audio"
                            src="${this.escapeHtml(src)}"
                            controls
                            ${autoplay ? 'autoplay' : ''}
                            part="audio"
                        ></audio>
                    ` : ''}

                    ${!compact ? `
                        <div class="ewr-audio-player__footer" part="footer">
                            <slot name="footer">
                                ${showAnalyze ? `
                                    <button class="ewr-audio-player__analyze ewr-btn-analyze" type="button">
                                        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                                            <path d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z"/>
                                        </svg>
                                        Analyze
                                    </button>
                                ` : ''}
                            </slot>
                        </div>
                    ` : ''}
                </div>
            </div>
        `;
    }

    getStyles() {
        return `
            :host { display: block; }

            .ewr-audio-player {
                background: var(--bg-secondary);
                border: 1px solid var(--border-primary);
                border-radius: 8px;
                padding: 24px;
                margin-bottom: 24px;
            }

            .ewr-audio-player.compact {
                padding: 16px;
                margin-bottom: 0;
            }

            .ewr-audio-player.expandable:not(.expanded) {
                padding-bottom: 16px;
            }

            .ewr-audio-player__header {
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-bottom: 16px;
            }

            .ewr-audio-player.expandable:not(.expanded) .ewr-audio-player__header {
                margin-bottom: 0;
            }

            .ewr-audio-player__title {
                font-size: 16px;
                font-weight: 600;
                color: var(--text-primary);
                margin: 0;
            }

            .ewr-audio-player__info {
                font-size: 13px;
                color: var(--text-tertiary);
            }

            .ewr-audio-player__toggle {
                width: 32px;
                height: 32px;
                padding: 0;
                border: none;
                border-radius: 4px;
                background: transparent;
                color: var(--text-tertiary);
                font-size: 12px;
                cursor: pointer;
                transition: all 0.2s ease;
                display: flex;
                align-items: center;
                justify-content: center;
            }

            .ewr-audio-player__toggle:hover {
                background: var(--bg-tertiary);
                color: var(--text-primary);
            }

            .ewr-audio-player__content {
                transition: all 0.3s ease;
            }

            .ewr-audio-player__content.hidden {
                display: none;
            }

            .ewr-audio-player__metadata {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
                gap: 12px;
                margin-bottom: 16px;
                padding: 12px;
                background: var(--bg-primary);
                border-radius: 6px;
            }

            .ewr-audio-player__metadata-item {
                display: flex;
                flex-direction: column;
                gap: 4px;
            }

            .ewr-audio-player__metadata-label {
                font-size: 11px;
                color: var(--text-tertiary);
                text-transform: uppercase;
                font-weight: 600;
            }

            .ewr-audio-player__metadata-value {
                font-size: 14px;
                color: var(--text-primary);
            }

            .ewr-audio-player__audio {
                width: 100%;
                margin-bottom: 16px;
                border-radius: 4px;
            }

            .ewr-audio-player.compact .ewr-audio-player__audio {
                margin-bottom: 0;
            }

            .ewr-audio-player__footer {
                display: flex;
                justify-content: flex-start;
                gap: 12px;
            }

            .ewr-audio-player__footer:empty {
                display: none;
            }

            .ewr-audio-player.compact .ewr-audio-player__footer {
                display: none;
            }

            .ewr-btn-analyze {
                display: inline-flex;
                align-items: center;
                gap: 8px;
                padding: 10px 16px;
                font-size: 13px;
                font-weight: 600;
                border: none;
                border-radius: 6px;
                cursor: pointer;
                transition: all 0.2s ease;
                background: linear-gradient(180deg, #a5b4fc 0%, #818cf8 30%, #6366f1 70%, #4f46e5 100%);
                color: white;
                font-family: inherit;
                box-shadow:
                    0 1px 2px rgba(0, 0, 0, 0.2),
                    inset 0 1px 1px rgba(255, 255, 255, 0.3),
                    inset 0 -1px 1px rgba(0, 0, 0, 0.1);
            }

            .ewr-btn-analyze svg {
                width: 16px;
                height: 16px;
            }

            .ewr-btn-analyze:hover {
                background: linear-gradient(180deg, #c7d2fe 0%, #a5b4fc 30%, #818cf8 70%, #6366f1 100%);
                transform: translateY(-1px);
            }

            .ewr-btn-analyze:active {
                background: linear-gradient(180deg, #6366f1 0%, #4f46e5 30%, #4338ca 70%, #3730a3 100%);
                transform: translateY(0);
            }

            /* Responsive */
            @media (max-width: 480px) {
                .ewr-audio-player {
                    padding: 16px;
                }

                .ewr-audio-player__metadata {
                    grid-template-columns: repeat(2, 1fr);
                }
            }
        `;
    }
}

if (!customElements.get('ewr-audio-player')) {
    customElements.define('ewr-audio-player', EwrAudioPlayer);
}

export default EwrAudioPlayer;
