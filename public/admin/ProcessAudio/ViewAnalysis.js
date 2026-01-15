/**
 * View Analysis Page
 * Displays saved audio analysis data from MongoDB
 */

// Global state
let currentAnalysis = null;
let analysisId = null;
let isDirty = false;
let selectedTicketIds = [];

// Icon for save button (using ewr-icon component)
const SAVE_ICON_SVG = '<ewr-icon name="save" size="14"></ewr-icon>';

// Initialize on page load
document.addEventListener('DOMContentLoaded', async () => {
    // Get ID from URL parameters
    const urlParams = new URLSearchParams(window.location.search);
    analysisId = urlParams.get('id');

    if (!analysisId) {
        showError('No analysis ID specified');
        return;
    }

    // Load monitored users for staff dropdown
    await loadMonitoredUsers();

    // Load the analysis data
    await loadAnalysis(analysisId);
});

/**
 * Load monitored users for staff dropdown
 */
async function loadMonitoredUsers() {
    try {
        const auth = new AuthClient();
        const response = await fetch('/api/auth/users', {
            headers: {
                'Authorization': 'Bearer ' + auth.getAccessToken()
            }
        });

        if (!response.ok) {
            console.error('Failed to fetch users for staff dropdown');
            return;
        }

        const users = await response.json();

        // Filter users with monitorAnalysis enabled
        const monitoredUsers = users.filter(user =>
            user.settings?.monitorAnalysis === true
        );

        // Populate staff dropdown
        const staffSelect = document.getElementById('staffInput');
        monitoredUsers.forEach(user => {
            const option = document.createElement('option');
            option.value = user.username;
            option.textContent = user.username;
            staffSelect.appendChild(option);
        });

        console.log(`Loaded ${monitoredUsers.length} monitored users for staff dropdown`);
    } catch (error) {
        console.error('Failed to load monitored users:', error);
    }
}

/**
 * Load analysis data from MongoDB
 */
async function loadAnalysis(id) {
    try {
        const response = await fetch(`/api/audio/${encodeURIComponent(id)}`);

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.error || 'Failed to load analysis');
        }

        currentAnalysis = await response.json();

        // Get the audio filename
        const audioFilename = currentAnalysis.filename
            || currentAnalysis.metadata?.filename
            || 'Unknown';

        // Populate the page
        populatePage(audioFilename);

        // Load any existing related ticket IDs
        selectedTicketIds = currentAnalysis.related_ticket_ids || [];

        // Show main content, hide loading
        document.getElementById('loadingState').style.display = 'none';
        document.getElementById('mainContent').style.display = 'grid';
        document.getElementById('deleteBtn').disabled = false;
        document.getElementById('saveBtn').disabled = false;

    } catch (error) {
        console.error('Failed to load analysis:', error);
        showError(error.message);
    }
}

/**
 * Populate the page with analysis data
 */
function populatePage(audioFilename) {
    const callMetadata = currentAnalysis.call_metadata || {};
    const callContent = currentAnalysis.call_content || {};
    const emotions = currentAnalysis.emotions || {};
    const audioMeta = currentAnalysis.audio_metadata || {};
    const metadata = currentAnalysis.metadata || {};

    // Header
    document.getElementById('filenameDisplay').textContent = audioFilename;

    // Call details
    const staffValue = currentAnalysis.customer_support_staff || metadata.customer_support_staff || '';
    const staffSelect = document.getElementById('staffInput');
    setSelectValue(staffSelect, staffValue);
    document.getElementById('customerInput').value = currentAnalysis.ewr_customer || metadata.ewr_customer || '';

    // Set mood select
    const moodValue = currentAnalysis.mood || metadata.mood || 'N/A';
    const moodSelect = document.getElementById('moodInput');
    setSelectValue(moodSelect, moodValue);

    // Set outcome select
    const outcomeValue = currentAnalysis.outcome || metadata.outcome || 'N/A';
    const outcomeSelect = document.getElementById('outcomeInput');
    setSelectValue(outcomeSelect, outcomeValue);

    // Call Metadata
    if (callMetadata.call_date) {
        const dateParts = callMetadata.call_date.split('-');
        if (dateParts.length === 3) {
            document.getElementById('metaDate').textContent = `${dateParts[1]}/${dateParts[2]}/${dateParts[0].slice(2)}`;
        } else {
            document.getElementById('metaDate').textContent = callMetadata.call_date;
        }
    }

    if (callMetadata.call_time) {
        const timeParts = callMetadata.call_time.split(':');
        if (timeParts.length >= 2) {
            let hour = parseInt(timeParts[0], 10);
            const ampm = hour >= 12 ? 'PM' : 'AM';
            hour = hour % 12 || 12;
            document.getElementById('metaTime').textContent = `${hour}:${timeParts[1]} ${ampm}`;
        } else {
            document.getElementById('metaTime').textContent = callMetadata.call_time;
        }
    }

    document.getElementById('metaExtension').textContent = callMetadata.extension || 'N/A';
    document.getElementById('metaPhone').textContent = callMetadata.phone_number || 'N/A';
    document.getElementById('metaDirection').textContent = callMetadata.direction || 'N/A';
    document.getElementById('metaRecordingId').textContent = callMetadata.recording_id || 'N/A';

    // Duration
    if (audioMeta.duration_seconds) {
        const mins = Math.floor(audioMeta.duration_seconds / 60);
        const secs = Math.floor(audioMeta.duration_seconds % 60);
        document.getElementById('metaDuration').textContent = `${mins}m ${secs}s`;
    }

    // Language
    document.getElementById('metaLanguage').textContent = currentAnalysis.language || 'auto';

    // LLM Analysis
    document.getElementById('llmSubject').textContent = callContent.subject || 'Not detected';
    document.getElementById('llmOutcome').textContent = callContent.outcome || 'Not detected';
    document.getElementById('llmCustomerName').textContent = callContent.customer_name || 'Not mentioned';

    // Emotions
    const primaryEmotion = emotions.primary || 'NEUTRAL';
    document.getElementById('primaryEmotion').textContent = `${getEmotionIcon(primaryEmotion)} ${primaryEmotion}`;

    const emotionsContainer = document.getElementById('emotionsContainer');
    const detectedEmotions = emotions.detected || [];
    if (detectedEmotions.length > 0) {
        emotionsContainer.innerHTML = detectedEmotions.map(emotion => {
            const isPrimary = emotion === primaryEmotion;
            return `<span class="emotion-badge ${isPrimary ? 'primary' : ''}">${getEmotionIcon(emotion)} ${emotion}</span>`;
        }).join('');
    } else {
        emotionsContainer.innerHTML = '<span class="emotion-badge">No emotions detected</span>';
    }

    // Audio info
    document.getElementById('audioFormat').textContent = audioMeta.format || 'N/A';
    document.getElementById('audioSize').textContent = formatFileSize(audioMeta.file_size_bytes);
    document.getElementById('audioSampleRate').textContent = audioMeta.sample_rate ? `${audioMeta.sample_rate} Hz` : 'N/A';
    document.getElementById('audioChannels').textContent = audioMeta.channels || 'N/A';

    // Formatted Transcription
    const transcription = currentAnalysis.transcription || '';
    document.getElementById('transcriptionBox').textContent = transcription || 'No transcription available';

    // Raw Transcription
    const rawTranscription = currentAnalysis.raw_transcription || '';
    document.getElementById('rawTranscriptionBox').textContent = rawTranscription || 'No raw transcription available';

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

        // Build speaker stats HTML
        const speakerStatsEl = document.getElementById('speakerStats');
        const stats = diarization.statistics || {};

        let speakerStatsHtml = '';
        Object.entries(stats).forEach(([speaker, data], index) => {
            const percentage = data.percentage || 0;
            const duration = data.total_duration || 0;
            const segments = data.segment_count || 0;

            // Format duration
            const mins = Math.floor(duration / 60);
            const secs = Math.floor(duration % 60);
            const durationStr = `${mins}m ${secs}s`;

            speakerStatsHtml += `
                <div class="speaker-stat-item">
                    <div class="speaker-stat-header">
                        <span class="speaker-stat-label">${speaker}</span>
                        <span class="speaker-stat-details">${durationStr} (${percentage.toFixed(1)}%) - ${segments} segments</span>
                    </div>
                    <div class="speaker-stat-bar">
                        <div class="speaker-stat-fill speaker-${index + 1}" style="width: ${percentage}%"></div>
                    </div>
                </div>
            `;
        });

        speakerStatsEl.innerHTML = speakerStatsHtml;
    }
}

/**
 * Delete analysis from MongoDB
 */
async function deleteAnalysis() {
    const filename = currentAnalysis?.filename || currentAnalysis?.metadata?.filename || 'this analysis';

    if (!confirm(`Are you sure you want to delete "${filename}"?\n\nThis action cannot be undone.`)) {
        return;
    }

    const deleteBtn = document.getElementById('deleteBtn');
    deleteBtn.disabled = true;
    deleteBtn.innerHTML = '⏳ Deleting...';

    try {
        const response = await fetch(`/api/audio/${analysisId}`, {
            method: 'DELETE'
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.error || 'Delete failed');
        }

        showMessage('Analysis deleted successfully', 'success');

        // Redirect back to search after a brief delay
        setTimeout(() => {
            goBack();
        }, 1000);

    } catch (error) {
        console.error('Delete failed:', error);
        showMessage(`Delete failed: ${error.message}`, 'error');
        deleteBtn.disabled = false;
        deleteBtn.innerHTML = '🗑️ Delete';
    }
}

/**
 * Go back to the search page
 */
function goBack() {
    window.location.href = 'search.html';
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
 * Copy raw transcription to clipboard
 */
async function copyRawTranscription() {
    const text = currentAnalysis?.raw_transcription || '';
    if (!text) {
        showMessage('No raw transcription to copy', 'error');
        return;
    }

    try {
        await navigator.clipboard.writeText(text);
        showMessage('Raw transcription copied to clipboard', 'success');
    } catch (err) {
        // Fallback
        const textarea = document.createElement('textarea');
        textarea.value = text;
        document.body.appendChild(textarea);
        textarea.select();
        document.execCommand('copy');
        document.body.removeChild(textarea);
        showMessage('Raw transcription copied to clipboard', 'success');
    }
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
        const callMetadata = currentAnalysis.call_metadata || {};
        const callContent = currentAnalysis.call_content || {};

        // Build datetime string from call metadata
        let callDatetime = null;
        if (callMetadata.call_date && callMetadata.call_time) {
            callDatetime = `${callMetadata.call_date}T${callMetadata.call_time}`;
        }

        // Extract keywords from subject
        const subjectKeywords = [];
        if (callContent.subject) {
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
                subject_keywords: subjectKeywords.slice(0, 5),
                customer_name: callContent.customer_name,
                time_window_minutes: 120
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

            // Truncate note/reason if too long
            let noteText = ticket.Note || 'No notes';
            if (noteText.length > 200) {
                noteText = noteText.substring(0, 200) + '...';
            }

            const isSelected = selectedTicketIds.includes(ticket.CentralTicketID);
            html += `
                <div class="ticket-card ${isBestMatch ? 'best-match' : ''} ${isSelected ? 'selected' : ''}" data-ticket-id="${ticket.CentralTicketID}">
                    <div class="ticket-header">
                        <label class="ticket-select-label">
                            <input type="checkbox" class="ticket-checkbox" ${isSelected ? 'checked' : ''}
                                   onchange="toggleTicketSelection(${ticket.CentralTicketID})">
                            <span class="ticket-id">#${ticket.CentralTicketID}</span>
                        </label>
                        <span class="match-score ${scoreClass}">
                            ${isBestMatch ? '⭐ Best Match' : `Score: ${match.match_score}`}
                        </span>
                    </div>
                    <div class="ticket-title">${escapeHtml(ticket.TicketTitle || 'No title')}</div>
                    <div class="ticket-details">
                        <div class="ticket-detail-row">
                            <span class="detail-label">Entry Date</span>
                            <span class="detail-value">${dateStr}</span>
                        </div>
                        <div class="ticket-detail-row">
                            <span class="detail-label">User</span>
                            <span class="detail-value">${escapeHtml(ticket.UserFirstName || ticket.CreatedBy || 'Unknown')}</span>
                        </div>
                        <div class="ticket-detail-row">
                            <span class="detail-label">Company</span>
                            <span class="detail-value">${escapeHtml(ticket.CompanyName || 'N/A')}</span>
                        </div>
                        <div class="ticket-detail-row">
                            <span class="detail-label">Customer</span>
                            <span class="detail-value">${escapeHtml(ticket.CustomerContactName || 'N/A')}</span>
                        </div>
                        <div class="ticket-detail-row">
                            <span class="detail-label">Phone</span>
                            <span class="detail-value">${escapeHtml(ticket.CustomerContactPhoneNumber || 'N/A')}</span>
                        </div>
                        <div class="ticket-detail-row">
                            <span class="detail-label">Type</span>
                            <span class="detail-value">${escapeHtml(ticket.TicketType || 'N/A')}</span>
                        </div>
                        <div class="ticket-detail-row">
                            <span class="detail-label">Status</span>
                            <span class="detail-value ticket-status">${escapeHtml(ticket.TicketStatus || 'N/A')}</span>
                        </div>
                        <div class="ticket-detail-row full-width">
                            <span class="detail-label">Reason</span>
                            <span class="detail-value reason-text">${escapeHtml(noteText)}</span>
                        </div>
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
 * Escape HTML to prevent XSS
 */
function escapeHtml(text) {
    if (!text) return '';
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

/**
 * Set select element value, fallback to first option if value not found
 */
function setSelectValue(selectEl, value) {
    // Try to find an exact match
    for (let option of selectEl.options) {
        if (option.value.toLowerCase() === value.toLowerCase()) {
            selectEl.value = option.value;
            return;
        }
    }
    // Try to find a partial match
    for (let option of selectEl.options) {
        if (option.value.toLowerCase().includes(value.toLowerCase()) ||
            value.toLowerCase().includes(option.value.toLowerCase())) {
            selectEl.value = option.value;
            return;
        }
    }
    // Default to N/A
    selectEl.value = 'N/A';
}

/**
 * Mark form as dirty (unsaved changes)
 */
function markDirty() {
    isDirty = true;
    const saveBtn = document.getElementById('saveBtn');
    if (saveBtn) {
        saveBtn.innerHTML = SAVE_ICON_SVG + ' Save *';
        saveBtn.classList.add('btn-warning');
    }
}

/**
 * Save analysis changes to MongoDB
 */
async function saveAnalysis() {
    const saveBtn = document.getElementById('saveBtn');
    saveBtn.disabled = true;
    saveBtn.innerHTML = '<span class="loading-dot"></span> Saving...';

    try {
        // Gather updated field values
        const updateData = {
            customer_support_staff: document.getElementById('staffInput').value || null,
            ewr_customer: document.getElementById('customerInput').value || null,
            mood: document.getElementById('moodInput').value,
            outcome: document.getElementById('outcomeInput').value,
            related_ticket_ids: selectedTicketIds
        };

        const response = await fetch(`/api/audio/${analysisId}`, {
            method: 'PUT',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(updateData)
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.error || 'Update failed');
        }

        const result = await response.json();

        // Update local state
        currentAnalysis.customer_support_staff = updateData.customer_support_staff;
        currentAnalysis.ewr_customer = updateData.ewr_customer;
        currentAnalysis.mood = updateData.mood;
        currentAnalysis.outcome = updateData.outcome;
        currentAnalysis.related_ticket_ids = updateData.related_ticket_ids;

        isDirty = false;
        saveBtn.innerHTML = SAVE_ICON_SVG + ' Save';
        saveBtn.classList.remove('btn-warning');
        saveBtn.disabled = false;

        showMessage('Analysis saved successfully', 'success');

    } catch (error) {
        console.error('Save failed:', error);
        showMessage(`Save failed: ${error.message}`, 'error');
        saveBtn.innerHTML = SAVE_ICON_SVG + ' Save *';
        saveBtn.disabled = false;
    }
}

/**
 * Toggle ticket selection for saving
 */
function toggleTicketSelection(ticketId) {
    const index = selectedTicketIds.indexOf(ticketId);
    if (index > -1) {
        selectedTicketIds.splice(index, 1);
    } else {
        selectedTicketIds.push(ticketId);
    }
    markDirty();
    updateTicketSelectionUI();
}

/**
 * Update ticket card selection UI
 */
function updateTicketSelectionUI() {
    document.querySelectorAll('.ticket-card').forEach(card => {
        const ticketId = parseInt(card.dataset.ticketId, 10);
        const checkbox = card.querySelector('.ticket-checkbox');
        if (checkbox) {
            checkbox.checked = selectedTicketIds.includes(ticketId);
        }
        if (selectedTicketIds.includes(ticketId)) {
            card.classList.add('selected');
        } else {
            card.classList.remove('selected');
        }
    });
}
