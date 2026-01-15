/**
 * Call Analysis Page
 * Displays and allows editing of pending audio analysis data
 */

// Global state
let currentAnalysis = null;
let pendingFilename = null;
let parsedFilenameMetadata = null;

// Initialize on page load
document.addEventListener('DOMContentLoaded', async () => {
    // Get filename from URL parameters
    const urlParams = new URLSearchParams(window.location.search);
    pendingFilename = urlParams.get('file');

    if (!pendingFilename) {
        showError('No file specified');
        return;
    }

    // Load the analysis data
    await loadAnalysis(pendingFilename);
});

/**
 * Load analysis data from the pending file
 */
async function loadAnalysis(filename) {
    try {
        const response = await fetch(`/api/audio/pending/${encodeURIComponent(filename)}`);

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.error || 'Failed to load analysis');
        }

        const data = await response.json();
        currentAnalysis = data.data || data;
        currentAnalysis._pendingFilename = filename;

        // Get the audio filename
        const audioFilename = currentAnalysis.metadata?.filename
            || currentAnalysis.audio_metadata?.original_filename
            || filename.replace('.json', '.mp3');

        // Parse filename metadata
        parsedFilenameMetadata = parseCallFilename(audioFilename);

        // Lookup staff by extension
        await lookupStaff();

        // Populate the page
        populatePage(audioFilename);

        // Show main content, hide loading
        document.getElementById('loadingState').style.display = 'none';
        document.getElementById('mainContent').style.display = 'grid';
        document.getElementById('saveBtn').disabled = false;

    } catch (error) {
        console.error('Failed to load analysis:', error);
        showError(error.message);
    }
}

/**
 * Lookup staff by extension
 */
async function lookupStaff() {
    const callMetadata = currentAnalysis.call_metadata || parsedFilenameMetadata || {};

    if (callMetadata.extension) {
        try {
            const response = await fetch(`/api/audio/lookup-staff/${encodeURIComponent(callMetadata.extension)}`);
            if (response.ok) {
                const staffData = await response.json();
                if (staffData.found && staffData.first_name) {
                    // Store for later use
                    if (!currentAnalysis.metadata) currentAnalysis.metadata = {};
                    currentAnalysis.metadata.customer_support_staff = staffData.first_name;
                }
            }
        } catch (e) {
            console.warn('Staff lookup failed:', e);
        }
    }
}

/**
 * Populate the page with analysis data
 */
function populatePage(audioFilename) {
    const callMetadata = currentAnalysis.call_metadata || parsedFilenameMetadata || {};
    const callContent = currentAnalysis.call_content || {};
    const emotions = currentAnalysis.emotions || {};
    const audioMeta = currentAnalysis.audio_metadata || {};

    // Header
    document.getElementById('filenameDisplay').textContent = audioFilename;

    // Required fields
    document.getElementById('staffInput').value = currentAnalysis.metadata?.customer_support_staff || '';
    document.getElementById('customerInput').value = callContent.customer_name || 'NA';

    // Map primary emotion to mood
    const moodMap = {
        'HAPPY': 'Positive',
        'SURPRISED': 'Positive',
        'SAD': 'Negative',
        'ANGRY': 'Negative',
        'FEARFUL': 'Negative',
        'DISGUSTED': 'Negative',
        'NEUTRAL': 'Neutral'
    };
    const mappedMood = moodMap[emotions.primary?.toUpperCase()] || '';
    document.getElementById('moodSelect').value = mappedMood;

    // Map LLM outcome to select options
    const outcomeMap = {
        'Resolved': 'Issue Resolved',
        'Unresolved': 'Issue Unresolved',
        'Pending Follow-up': 'Issue Logged in Central',
        'Information Provided': 'Issue Resolved',
        'Transferred': 'Issue Unresolved',
        'Unknown': 'Undetermined'
    };
    document.getElementById('outcomeSelect').value = outcomeMap[callContent.outcome] || 'Undetermined';

    // Call Metadata
    if (callMetadata.call_date) {
        const dateParts = callMetadata.call_date.split('-');
        if (dateParts.length === 3) {
            document.getElementById('metaDate').textContent = `${dateParts[1]}/${dateParts[2]}/${dateParts[0].slice(2)}`;
        }
    }

    if (callMetadata.call_time) {
        const timeParts = callMetadata.call_time.split(':');
        if (timeParts.length >= 2) {
            let hour = parseInt(timeParts[0], 10);
            const ampm = hour >= 12 ? 'PM' : 'AM';
            hour = hour % 12 || 12;
            document.getElementById('metaTime').textContent = `${hour}:${timeParts[1]} ${ampm}`;
        }
    }

    document.getElementById('metaExtension').textContent = callMetadata.extension || 'N/A';
    document.getElementById('metaPhone').textContent = callMetadata.phone_number || 'N/A';

    const directionEl = document.getElementById('metaDirection');
    directionEl.textContent = callMetadata.direction || 'N/A';
    directionEl.className = `metadata-value ${callMetadata.direction?.toLowerCase() || ''}`;

    // Duration
    const duration = audioMeta.duration_seconds || 0;
    const minutes = Math.floor(duration / 60);
    const seconds = Math.floor(duration % 60);
    document.getElementById('metaDuration').textContent = `${minutes}m ${seconds}s`;

    // Language
    const langNames = {
        'en': 'English', 'zh': 'Chinese', 'ja': 'Japanese', 'ko': 'Korean',
        'es': 'Spanish', 'fr': 'French', 'de': 'German', 'ru': 'Russian', 'auto': 'Auto-detected'
    };
    document.getElementById('metaLanguage').textContent = langNames[currentAnalysis.language] || currentAnalysis.language || 'Unknown';

    document.getElementById('metaRecordingId').textContent = callMetadata.recording_id || 'N/A';

    // LLM Analysis
    document.getElementById('llmSubject').textContent = callContent.subject || 'Not detected';
    document.getElementById('llmOutcome').textContent = callContent.outcome || 'Not detected';
    document.getElementById('llmCustomerName').textContent = callContent.customer_name || 'Not mentioned';

    // Emotions
    const primaryEl = document.getElementById('primaryEmotion');
    primaryEl.textContent = `${getEmotionIcon(emotions.primary || 'NEUTRAL')} ${emotions.primary || 'NEUTRAL'}`;

    const emotionsContainer = document.getElementById('emotionsContainer');
    const detected = emotions.detected || [];
    if (detected.length > 0) {
        emotionsContainer.innerHTML = detected.map(e => {
            const isPrimary = e === emotions.primary;
            return `<span class="emotion-badge ${isPrimary ? 'primary' : ''}">${getEmotionIcon(e)} ${e}</span>`;
        }).join('');
    } else {
        emotionsContainer.innerHTML = '<span class="emotion-badge">No emotions detected</span>';
    }

    // Audio Info
    document.getElementById('audioFormat').textContent = (audioMeta.format || 'N/A').toUpperCase();
    document.getElementById('audioSize').textContent = formatFileSize(audioMeta.file_size_bytes);
    document.getElementById('audioSampleRate').textContent = audioMeta.sample_rate ? `${audioMeta.sample_rate} Hz` : 'N/A';
    document.getElementById('audioChannels').textContent = audioMeta.channels === 1 ? 'Mono' : (audioMeta.channels === 2 ? 'Stereo' : (audioMeta.channels || 'N/A'));

    // Transcription
    const transcriptionBox = document.getElementById('transcriptionBox');
    transcriptionBox.textContent = currentAnalysis.transcription || 'No transcription available';

    // Summary
    if (currentAnalysis.transcription_summary) {
        document.getElementById('summaryCard').style.display = 'block';
        document.getElementById('summaryBox').textContent = currentAnalysis.transcription_summary;
    }

    // Speaker Diarization
    const diarization = currentAnalysis.speaker_diarization;
    if (diarization && diarization.enabled && diarization.num_speakers > 0) {
        document.getElementById('diarizationCard').style.display = 'block';
        document.getElementById('numSpeakers').textContent = diarization.num_speakers;
        document.getElementById('numSegments').textContent = diarization.segments?.length || 0;

        // Build speaker stats display
        const statsContainer = document.getElementById('speakerStats');
        const stats = diarization.statistics || {};
        const speakers = Object.keys(stats).sort();

        if (speakers.length > 0) {
            let html = '';
            speakers.forEach((speaker, idx) => {
                const stat = stats[speaker];
                const duration = stat.total_duration || 0;
                const percentage = stat.percentage || 0;
                const minutes = Math.floor(duration / 60);
                const seconds = Math.floor(duration % 60);

                html += `
                    <div class="speaker-stat">
                        <div class="speaker-label">${speaker}</div>
                        <div class="speaker-bar-container">
                            <div class="speaker-bar speaker-${idx + 1}" style="width: ${percentage}%"></div>
                        </div>
                        <div class="speaker-info">
                            <span class="speaker-percentage">${percentage.toFixed(1)}%</span>
                            <span>(${minutes}m ${seconds}s)</span>
                        </div>
                    </div>
                `;
            });
            statsContainer.innerHTML = html;
        }
    }
}

/**
 * Save analysis to MongoDB
 */
async function saveToMongoDB() {
    // Validate required fields
    const staff = document.getElementById('staffInput').value.trim();
    const customer = document.getElementById('customerInput').value.trim();
    const mood = document.getElementById('moodSelect').value;
    const outcome = document.getElementById('outcomeSelect').value;

    const missingFields = [];
    if (!staff) missingFields.push('Customer Support Staff');
    if (!customer) missingFields.push('EWR Customer');
    if (!mood) missingFields.push('Mood');
    if (!outcome) missingFields.push('Outcome');

    if (missingFields.length > 0) {
        showMessage(`Please fill in required fields: ${missingFields.join(', ')}`, 'error');
        return;
    }

    const saveBtn = document.getElementById('saveBtn');
    saveBtn.disabled = true;
    saveBtn.innerHTML = '⏳ Saving...';

    try {
        const callMetadata = currentAnalysis.call_metadata || parsedFilenameMetadata || {};
        const callContent = currentAnalysis.call_content || {};

        const audioFilename = currentAnalysis.metadata?.filename
            || currentAnalysis.audio_metadata?.original_filename
            || pendingFilename.replace('.json', '.mp3');

        // Build emotions object
        const emotions = {
            primary: currentAnalysis.emotions?.primary || 'NEUTRAL',
            detected: currentAnalysis.emotions?.detected || [],
            timestamps: currentAnalysis.emotions?.timestamps || []
        };

        const payload = {
            transcription: currentAnalysis.transcription || '',
            raw_transcription: currentAnalysis.raw_transcription || '',
            transcription_summary: currentAnalysis.transcription_summary || null,
            emotions: emotions,
            audio_events: currentAnalysis.audio_events || { detected: [], timestamps: [] },
            language: currentAnalysis.language || 'en',
            audio_metadata: currentAnalysis.audio_metadata || null,
            metadata: {
                customer_support_staff: staff,
                ewr_customer: customer,
                mood: mood,
                outcome: outcome,
                filename: audioFilename
            },
            call_metadata: {
                call_date: callMetadata.call_date || null,
                call_time: callMetadata.call_time || null,
                extension: callMetadata.extension || null,
                phone_number: callMetadata.phone_number || null,
                direction: callMetadata.direction || null,
                auto_flag: callMetadata.auto_flag || null,
                recording_id: callMetadata.recording_id || null,
                parsed: callMetadata.parsed || false
            },
            call_content: {
                subject: callContent.subject || null,
                outcome: callContent.outcome || null,
                customer_name: callContent.customer_name || null,
                confidence: callContent.confidence || 0,
                analysis_model: callContent.analysis_model || ''
            },
            speaker_diarization: currentAnalysis.speaker_diarization || {
                enabled: false,
                segments: [],
                statistics: {},
                num_speakers: 0
            },
            pending_filename: pendingFilename
        };

        const response = await fetch('/api/audio/store', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });

        const result = await response.json();

        if (!response.ok) {
            throw new Error(result.error || 'Save failed');
        }

        showMessage(`Analysis saved successfully! ID: ${result.analysis_id}`, 'success');

        // Redirect back to the list after a brief delay
        setTimeout(() => {
            goBack();
        }, 1500);

    } catch (error) {
        console.error('Save failed:', error);
        showMessage(`Save failed: ${error.message}`, 'error');
        saveBtn.disabled = false;
        saveBtn.innerHTML = '💾 Save to MongoDB';
    }
}

/**
 * Go back to the pending save list
 */
function goBack() {
    window.location.href = 'ProcessAudio.html';
}

/**
 * Copy transcription to clipboard
 */
async function copyTranscription() {
    const text = currentAnalysis?.transcription || '';
    if (!text) {
        showMessage('No transcription to copy', 'error');
        return;
    }

    try {
        await navigator.clipboard.writeText(text);
        showMessage('Transcription copied to clipboard', 'success');
    } catch (err) {
        // Fallback
        const textarea = document.createElement('textarea');
        textarea.value = text;
        document.body.appendChild(textarea);
        textarea.select();
        document.execCommand('copy');
        document.body.removeChild(textarea);
        showMessage('Transcription copied to clipboard', 'success');
    }
}

/**
 * Show error state
 */
function showError(message) {
    document.getElementById('loadingState').style.display = 'none';
    document.getElementById('errorState').style.display = 'block';
    document.getElementById('errorMessage').textContent = message;
}

/**
 * Show message toast
 */
function showMessage(message, type = 'info') {
    const toast = document.getElementById('messageToast');
    toast.textContent = message;
    toast.className = `message-toast ${type}`;
    toast.style.display = 'block';

    setTimeout(() => {
        toast.style.display = 'none';
    }, 4000);
}

/**
 * Parse RingCentral call recording filename
 */
function parseCallFilename(filename) {
    const metadata = {
        call_date: null,
        call_time: null,
        extension: null,
        phone_number: null,
        direction: null,
        auto_flag: null,
        recording_id: null,
        parsed: false
    };

    const basename = filename.split(/[\/\\]/).pop();
    const nameWithoutExt = basename.replace(/\.[^.]+$/, '');

    if (!/^\d{8}-/.test(nameWithoutExt)) {
        return metadata;
    }

    const pattern = /^(\d{4})(\d{2})(\d{2})-(\d{2})(\d{2})(\d{2})_(\d+)_(\([^)]+\)[^_]+|[^_]+)_(Incoming|Outgoing)_([^_]+)_(\d+)$/;
    const match = nameWithoutExt.match(pattern);

    if (match) {
        const [, year, month, day, hour, minute, second, ext, phone, direction, autoFlag, recordingId] = match;
        metadata.call_date = `${year}-${month}-${day}`;
        metadata.call_time = `${hour}:${minute}:${second}`;
        metadata.extension = ext;
        metadata.phone_number = phone;
        metadata.direction = direction;
        metadata.auto_flag = autoFlag;
        metadata.recording_id = recordingId;
        metadata.parsed = true;
    }

    return metadata;
}

/**
 * Get emoji icon for emotion
 */
function getEmotionIcon(emotion) {
    const icons = {
        'HAPPY': '😊',
        'SAD': '😢',
        'ANGRY': '😠',
        'NEUTRAL': '😐',
        'FEARFUL': '😨',
        'DISGUSTED': '🤢',
        'SURPRISED': '😲'
    };
    return icons[emotion?.toUpperCase()] || '😐';
}

/**
 * Format file size
 */
function formatFileSize(bytes) {
    if (!bytes || bytes === 0) return 'N/A';
    const units = ['B', 'KB', 'MB', 'GB'];
    let unitIndex = 0;
    let size = bytes;
    while (size >= 1024 && unitIndex < units.length - 1) {
        size /= 1024;
        unitIndex++;
    }
    return `${size.toFixed(2)} ${units[unitIndex]}`;
}

/**
 * Search for matching tickets in EWRCentral
 */
async function searchTickets() {
    const loadingEl = document.getElementById('ticketsLoading');
    const contentEl = document.getElementById('ticketsContent');

    loadingEl.style.display = 'block';
    contentEl.innerHTML = '';

    try {
        const callMetadata = currentAnalysis.call_metadata || parsedFilenameMetadata || {};
        const callContent = currentAnalysis.call_content || {};

        // Build datetime string from call metadata
        let callDatetime = null;
        if (callMetadata.call_date && callMetadata.call_time) {
            callDatetime = `${callMetadata.call_date}T${callMetadata.call_time}`;
        }

        // Extract keywords from subject
        const subjectKeywords = [];
        if (callContent.subject) {
            // Extract meaningful words (4+ chars, not common words)
            const stopWords = ['the', 'and', 'for', 'with', 'that', 'this', 'from', 'have', 'been', 'will', 'about'];
            const words = callContent.subject.toLowerCase().split(/\s+/);
            words.forEach(word => {
                const clean = word.replace(/[^a-z0-9]/g, '');
                if (clean.length >= 4 && !stopWords.includes(clean)) {
                    subjectKeywords.push(clean);
                }
            });
        }

        const response = await fetch('/api/audio/match-tickets', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                extension: callMetadata.extension,
                phone_number: callMetadata.phone_number,
                call_datetime: callDatetime,
                subject_keywords: subjectKeywords.slice(0, 5),  // Limit to 5 keywords
                customer_name: callContent.customer_name,
                time_window_minutes: 120  // 2 hours after call
            })
        });

        const result = await response.json();

        loadingEl.style.display = 'none';

        if (!result.success) {
            contentEl.innerHTML = `<div class="no-tickets"><div class="no-tickets-icon">⚠️</div><p>Search failed: ${result.error}</p></div>`;
            return;
        }

        if (result.matches.length === 0) {
            contentEl.innerHTML = `
                <div class="no-tickets">
                    <div class="no-tickets-icon">🔍</div>
                    <p>No matching tickets found</p>
                    <p style="font-size: 13px; margin-top: 8px;">
                        Searched for tickets by ${callMetadata.extension ? `extension ${callMetadata.extension}` : 'staff'}
                        ${callDatetime ? `within 2 hours of call` : ''}
                    </p>
                </div>
            `;
            return;
        }

        // Display matching tickets
        let html = '<div class="ticket-list">';

        result.matches.forEach((match, index) => {
            const ticket = match.ticket;
            const isBestMatch = result.best_match && ticket.CentralTicketID === result.best_match.ticket.CentralTicketID;

            // Determine score class
            let scoreClass = 'low';
            if (match.match_score >= 70) scoreClass = 'high';
            else if (match.match_score >= 40) scoreClass = 'medium';

            // Format date
            let dateStr = 'N/A';
            if (ticket.AddTicketDate) {
                const date = new Date(ticket.AddTicketDate);
                dateStr = date.toLocaleString();
            }

            html += `
                <div class="ticket-card ${isBestMatch ? 'best-match' : ''}">
                    <div class="ticket-header">
                        <span class="ticket-id">#${ticket.CentralTicketID}</span>
                        <span class="match-score ${scoreClass}">
                            ${isBestMatch ? '⭐ Best Match' : `Score: ${match.match_score}`}
                        </span>
                    </div>
                    <div class="ticket-title">${escapeHtml(ticket.TicketTitle || 'No title')}</div>
                    <div class="ticket-meta">
                        <span>📅 ${dateStr}</span>
                        <span>👤 ${escapeHtml(ticket.CreatedBy || 'Unknown')}</span>
                        <span>🏢 ${escapeHtml(ticket.CompanyName || 'N/A')}</span>
                        <span>📞 ${escapeHtml(ticket.CustomerContactName || 'N/A')} - ${escapeHtml(ticket.CustomerContactPhoneNumber || 'N/A')}</span>
                    </div>
                    ${match.match_reasons.length > 0 ? `
                        <div class="match-reasons">
                            ${match.match_reasons.map(r => `<span class="match-reason">${formatMatchReason(r)}</span>`).join('')}
                        </div>
                    ` : ''}
                </div>
            `;
        });

        html += '</div>';
        contentEl.innerHTML = html;

    } catch (error) {
        console.error('Ticket search failed:', error);
        loadingEl.style.display = 'none';
        contentEl.innerHTML = `<div class="no-tickets"><div class="no-tickets-icon">❌</div><p>Error: ${error.message}</p></div>`;
    }
}

/**
 * Format match reason for display
 */
function formatMatchReason(reason) {
    const reasonMap = {
        'phone_number_match': '📞 Phone match',
        'staff_match': '👤 Staff match',
        'created_within_30min': '⏱️ Within 30 min',
        'created_within_60min': '⏱️ Within 60 min',
        'customer_name_match': '🏷️ Customer match'
    };

    if (reason.startsWith('keyword_match:')) {
        return `🔑 "${reason.split(':')[1]}"`;
    }

    return reasonMap[reason] || reason;
}

/**
 * Escape HTML to prevent XSS
 */
function escapeHtml(text) {
    if (!text) return '';
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}
