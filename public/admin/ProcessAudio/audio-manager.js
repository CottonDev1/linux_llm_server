/**
 * Audio Manager - Unified File Management
 * Handles drag-and-drop, directory polling, file grids, and analysis workflow
 */

// ============================================
// State Management
// ============================================

const state = {
    unanalyzedFiles: [],    // Files waiting to be processed { id, filename, size, file, selected, expanded, statusLog, metrics }
    pendingFiles: [],       // Analyzed files waiting for save { id, filename, duration, mood, status, analysis, expanded }
    expandedRowId: null,    // Currently expanded row ID
    audioBlobs: new Map(),  // Map of file ID -> Blob URL for playback
    currentModal: null,     // Currently open modal data
    isProcessing: false,    // Flag to track if files are being processed
    currentGpu: null        // Current GPU being used for processing
};

// ============================================
// DOM Elements (initialized after DOM ready)
// ============================================

let elements = {};

function initElements() {
    elements = {
        // Input elements
        dropZone: document.getElementById('dropZone'),
        fileInput: document.getElementById('fileInput'),
        directoryPath: document.getElementById('directoryPath'),
        pollBtn: document.getElementById('pollBtn'),
        pollStatus: document.getElementById('pollStatus'),

        // Unanalyzed grid
        selectAllUnanalyzed: document.getElementById('selectAllUnanalyzed'),
        unanalyzedBody: document.getElementById('unanalyzedBody'),
        unanalyzedCount: document.getElementById('unanalyzedCount'),
        selectedCount: document.getElementById('selectedCount'),
        processSelectedBtn: document.getElementById('processSelectedBtn'),

        // Pending grid
        pendingBody: document.getElementById('pendingBody'),
        pendingCount: document.getElementById('pendingCount'),
        pendingSavePath: document.getElementById('pendingSavePath'),

        // Modal
        saveModal: document.getElementById('saveModal'),
        modalTitle: document.getElementById('modalTitle'),
        modalStaff: document.getElementById('modalStaff'),
        modalDetectedName: document.getElementById('modalDetectedName'),
        modalCustomer: document.getElementById('modalCustomer'),
        modalMood: document.getElementById('modalMood'),
        modalOutcome: document.getElementById('modalOutcome'),
        modalSubject: document.getElementById('modalSubject'),
        modalDuration: document.getElementById('modalDuration'),
        modalLanguage: document.getElementById('modalLanguage'),
        modalFilename: document.getElementById('modalFilename'),
        modalSummary: document.getElementById('modalSummary'),
        modalTranscription: document.getElementById('modalTranscription'),
        emotionsGrid: document.getElementById('emotionsGrid'),
        modalAnalysisData: document.getElementById('modalAnalysisData'),
        saveAnalysisBtn: document.getElementById('saveAnalysisBtn')
    };
}

// Store monitored staff names for name detection
let monitoredStaffNames = [];

// ============================================
// Initialization
// ============================================

async function initAudioManager() {
    try {
        console.log('initAudioManager starting...');

        // Initialize DOM elements first (must run after DOM is ready)
        initElements();
        console.log('Elements initialized:', {
            dropZone: !!elements.dropZone,
            fileInput: !!elements.fileInput,
            pollBtn: !!elements.pollBtn
        });

        // Setup event listeners FIRST - this is critical for drop zone to work
        console.log('Setting up event listeners...');
        setupEventListeners();
        console.log('Event listeners set up');

        // Initialize sidebar navigation (can fail without breaking the page)
        console.log('Initializing sidebar...');
        try {
            if (typeof initSidebar === 'function') {
                await initSidebar();
                console.log('Sidebar initialized');
            } else {
                console.error('initSidebar function not found - sidebar.js may not be loaded');
            }
        } catch (sidebarError) {
            console.error('Sidebar initialization failed (non-critical):', sidebarError);
        }

        // Load saved directory path and pending save path
        loadSavedDirectoryPath();
        loadPendingSavePath();

        // Load monitored users for staff dropdown
        loadMonitoredUsers();

        // Load existing pending analysis files from server
        // This ensures pending files persist across page reloads/restarts
        await refreshPendingFiles();

        // Load existing unanalyzed files from server
        // This ensures unanalyzed files persist across page reloads/restarts
        await refreshUnanalyzedFiles();

        console.log('Audio Manager initialization complete');
    } catch (error) {
        console.error('Error in initAudioManager:', error);
    }
}

function setupEventListeners() {
    // Drop zone - must exist
    if (!elements.dropZone || !elements.fileInput) {
        console.error('Drop zone or file input not found!', {
            dropZone: elements.dropZone,
            fileInput: elements.fileInput
        });
        return;
    }

    // Prevent browser from opening files dropped anywhere on page
    document.addEventListener('dragover', preventDefaultDragDrop);
    document.addEventListener('drop', preventDefaultDragDrop);

    // Click on drop zone opens file browser
    elements.dropZone.addEventListener('click', (e) => {
        console.log('Drop zone clicked');
        e.preventDefault();
        e.stopPropagation();
        elements.fileInput.click();
    });

    elements.fileInput.addEventListener('change', handleFileSelection);

    // Drop zone specific handlers (these run BEFORE the document handlers due to bubbling)
    elements.dropZone.addEventListener('dragover', handleDragOver);
    elements.dropZone.addEventListener('dragleave', handleDragLeave);
    elements.dropZone.addEventListener('drop', handleDrop);

    // Directory polling
    if (elements.pollBtn) {
        elements.pollBtn.addEventListener('click', pollDirectory);
    }

    // Select all checkbox
    if (elements.selectAllUnanalyzed) {
        elements.selectAllUnanalyzed.addEventListener('change', handleSelectAll);
    }

    // Process button
    if (elements.processSelectedBtn) {
        elements.processSelectedBtn.addEventListener('click', processSelectedFiles);
    }

    // Modal validation
    [elements.modalStaff, elements.modalCustomer, elements.modalMood].forEach(el => {
        if (el) {
            el.addEventListener('input', validateModalForm);
        }
    });

    console.log('All event listeners attached');
}

async function loadSavedDirectoryPath() {
    try {
        const response = await fetch('/api/audio/settings/folder-path');
        const data = await response.json();
        if (data.success && data.folder_path) {
            elements.directoryPath.value = data.folder_path;
        }
    } catch (error) {
        console.error('Failed to load directory path:', error);
    }
}

async function loadMonitoredUsers() {
    try {
        // Load Customer Support staff from EWRCentral database
        const response = await fetch('/api/audio/customer-support-staff');

        if (!response.ok) {
            console.error('Failed to fetch Customer Support staff from EWRCentral');
            // Fallback to auth users if EWRCentral fails
            await loadMonitoredUsersFromAuth();
            return;
        }

        const data = await response.json();

        if (!data.success || !data.staff) {
            console.error('Invalid response from Customer Support staff endpoint');
            await loadMonitoredUsersFromAuth();
            return;
        }

        // Store staff names for name detection in transcripts
        monitoredStaffNames = data.staff.map(s => s.name);

        // Populate staff dropdown
        if (elements.modalStaff) {
            // Clear existing options except the placeholder
            elements.modalStaff.innerHTML = '<option value="">Select Customer Support Staff...</option>';

            // Add Customer Support staff
            data.staff.forEach(staff => {
                const option = document.createElement('option');
                option.value = staff.name;
                option.textContent = `${staff.name}${staff.extension ? ` (Ext: ${staff.extension})` : ''}`;
                option.dataset.email = staff.email || '';
                option.dataset.extension = staff.extension || '';
                elements.modalStaff.appendChild(option);
            });

            console.log(`Loaded ${data.staff.length} Customer Support staff from EWRCentral`);
        }
    } catch (error) {
        console.error('Failed to load Customer Support staff:', error);
        // Fallback to auth users if EWRCentral fails
        await loadMonitoredUsersFromAuth();
    }
}

// Fallback function to load from auth if EWRCentral is unavailable
async function loadMonitoredUsersFromAuth() {
    try {
        const auth = new AuthClient();
        const response = await fetch('/api/auth/users', {
            headers: {
                'Authorization': 'Bearer ' + auth.getAccessToken()
            }
        });

        if (!response.ok) {
            console.error('Fallback: Failed to fetch users for staff dropdown');
            return;
        }

        const users = await response.json();

        // Filter users with monitorAnalysis enabled
        const monitoredUsers = users.filter(user =>
            user.settings?.monitorAnalysis === true
        );

        // Store staff names for name detection in transcripts
        monitoredStaffNames = monitoredUsers.map(user => user.username);

        // Populate staff dropdown
        if (elements.modalStaff) {
            elements.modalStaff.innerHTML = '<option value="">Select Customer Support Staff...</option>';

            monitoredUsers.forEach(user => {
                const option = document.createElement('option');
                option.value = user.username;
                option.textContent = user.username;
                elements.modalStaff.appendChild(option);
            });

            console.log(`Fallback: Loaded ${monitoredUsers.length} monitored users for staff dropdown`);
        }
    } catch (error) {
        console.error('Fallback: Failed to load monitored users:', error);
    }
}

/**
 * Detect staff names mentioned in the transcription
 * @param {string} transcription - The transcription text to search
 * @returns {string[]} Array of detected staff names
 */
function detectStaffNamesInTranscription(transcription) {
    if (!transcription || !monitoredStaffNames.length) {
        return [];
    }

    const detectedNames = [];
    const transcriptLower = transcription.toLowerCase();

    for (const staffName of monitoredStaffNames) {
        // Check for the full name (case-insensitive)
        if (transcriptLower.includes(staffName.toLowerCase())) {
            detectedNames.push(staffName);
            continue;
        }

        // Also check for first name only (if name has spaces)
        const firstName = staffName.split(' ')[0];
        if (firstName.length >= 3 && transcriptLower.includes(firstName.toLowerCase())) {
            // Use regex to ensure it's a whole word match
            const wordBoundaryRegex = new RegExp(`\\b${firstName}\\b`, 'i');
            if (wordBoundaryRegex.test(transcription)) {
                detectedNames.push(staffName);
            }
        }
    }

    return [...new Set(detectedNames)]; // Remove duplicates
}

async function loadPendingSavePath() {
    try {
        const response = await fetch('/api/audio/settings/pending-path');
        const data = await response.json();
        if (data.success && data.pending_path && elements.pendingSavePath) {
            elements.pendingSavePath.textContent = data.pending_path;
        }
    } catch (error) {
        console.error('Failed to load pending save path:', error);
    }
}

// ============================================
// Drag and Drop Handlers
// ============================================

function handleDragOver(e) {
    e.preventDefault();
    e.stopPropagation();
    // Set the drop effect to show it's a valid drop target
    e.dataTransfer.dropEffect = 'copy';
    elements.dropZone.classList.add('drag-over');
}

function handleDragLeave(e) {
    e.preventDefault();
    e.stopPropagation();
    elements.dropZone.classList.remove('drag-over');
}

function handleDrop(e) {
    e.preventDefault();
    e.stopPropagation();
    elements.dropZone.classList.remove('drag-over');

    console.log('File dropped on drop zone');
    const files = Array.from(e.dataTransfer.files);
    console.log('Files dropped:', files.map(f => f.name));
    addFilesToUnanalyzed(files);
}

// Prevent browser from opening files dropped elsewhere on the page
function preventDefaultDragDrop(e) {
    e.preventDefault();
    e.stopPropagation();
}

function handleFileSelection(e) {
    const files = Array.from(e.target.files);
    addFilesToUnanalyzed(files);
    // Reset input so same file can be selected again
    e.target.value = '';
}

function addFilesToUnanalyzed(files) {
    // Filter audio files only
    const audioFiles = files.filter(f =>
        f.type.match(/audio\/(wav|mpeg|mp3|m4a)/) ||
        f.name.match(/\.(wav|mp3|m4a)$/i)
    );

    if (audioFiles.length === 0) {
        showToast('No valid audio files found. Supported: WAV, MP3, M4A', 'error');
        return;
    }

    // Add files to state
    audioFiles.forEach(file => {
        const id = `file-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
        const blobUrl = URL.createObjectURL(file);

        state.unanalyzedFiles.push({
            id,
            filename: file.name,
            size: (file.size / 1024 / 1024).toFixed(2), // MB
            file: file,
            selected: true,
            expanded: false,
            status: 'Pending',
            statusLog: []  // Array to store all progress messages
        });

        state.audioBlobs.set(id, blobUrl);
    });

    elements.dropZone.classList.add('has-files');
    updateUnanalyzedGrid();
    showToast(`Added ${audioFiles.length} file(s)`, 'success');
}

// ============================================
// Directory Polling
// ============================================

async function pollDirectory() {
    const path = elements.directoryPath.value.trim();

    if (!path) {
        showToast('Please enter a directory path', 'error');
        return;
    }

    elements.pollBtn.disabled = true;
    elements.pollBtn.textContent = 'Polling...';
    elements.pollStatus.textContent = 'Scanning directory...';

    try {
        const response = await fetch('/api/audio/bulk/scan', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                source_directory: path,
                recursive: true,
                max_files: 100
            })
        });

        const data = await response.json();

        if (data.success && data.files && data.files.length > 0) {
            // Add files from directory to unanalyzed list
            data.files.forEach(fileInfo => {
                const id = `dir-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;

                state.unanalyzedFiles.push({
                    id,
                    filename: fileInfo.filename,
                    size: fileInfo.size_mb.toFixed(2),
                    file: null, // No file object, will stream from server
                    filepath: fileInfo.filepath, // Server path
                    selected: true,
                    expanded: false,
                    status: 'Pending',
                    statusLog: []  // Array to store all progress messages
                });
            });

            updateUnanalyzedGrid();
            elements.pollStatus.textContent = `Found ${data.files.length} file(s)`;
            showToast(`Found ${data.files.length} audio file(s)`, 'success');
        } else {
            elements.pollStatus.textContent = 'No files found';
            showToast('No audio files found in directory', 'info');
        }
    } catch (error) {
        console.error('Poll error:', error);
        elements.pollStatus.textContent = 'Error polling directory';
        showToast('Failed to poll directory', 'error');
    } finally {
        elements.pollBtn.disabled = false;
        elements.pollBtn.textContent = 'Poll';
    }
}

// ============================================
// Unanalyzed Grid Management
// ============================================

function updateUnanalyzedGrid() {
    const tbody = elements.unanalyzedBody;
    const section = document.getElementById('unanalyzedSection');

    if (state.unanalyzedFiles.length === 0) {
        tbody.innerHTML = `
            <tr>
                <td colspan="6">
                    <div class="ewr-audio-empty-state">
                        <div class="ewr-audio-empty-state-icon">
                            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M22 19a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h5l2 3h9a2 2 0 0 1 2 2z"/></svg>
                        </div>
                        <div class="ewr-audio-empty-state-text">No files added</div>
                    </div>
                </td>
            </tr>
        `;
        elements.unanalyzedCount.textContent = '0 files';
        elements.processSelectedBtn.disabled = true;
        if (section) section.classList.remove('has-content');
        return;
    }

    if (section) section.classList.add('has-content');

    // Render file rows using inline HTML (custom elements don't work well with tables)
    tbody.innerHTML = state.unanalyzedFiles.map(fileItem => {
        const expanded = fileItem.expanded;
        // Use blob URL for dropped files, or stream-file endpoint with filepath for directory-polled files
        const audioSrc = fileItem.file
            ? state.audioBlobs.get(fileItem.id)
            : `/api/audio/stream-file?filepath=${encodeURIComponent(fileItem.filepath)}`;

        return `
            <tr class="${expanded ? 'expanded' : ''}" data-file-id="${fileItem.id}">
                <td class="ewr-checkbox-cell">
                    <input type="checkbox" class="ewr-file-checkbox"
                           ${fileItem.selected ? 'checked' : ''}
                           onchange="toggleFileSelection('${fileItem.id}', this.checked)">
                </td>
                <td style="width: 30px;">
                    <span class="expand-indicator" onclick="toggleRowExpansion('${fileItem.id}')">▶</span>
                </td>
                <td class="filename-cell" onclick="toggleRowExpansion('${fileItem.id}')" style="max-width: 340px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;" title="${escapeHtml(fileItem.filename)}">${escapeHtml(fileItem.filename)}</td>
                <td style="width: 70px;">${fileItem.size}</td>
                <td class="status-cell" style="min-width: 350px; max-width: 450px;">
                    <div class="status-log-container" style="max-height: 90px; overflow-y: auto; font-size: 14px; font-family: monospace; background: rgba(0,0,0,0.3); border-radius: 4px; padding: 6px 10px;">
                        ${fileItem.statusLog && fileItem.statusLog.length > 0
                            ? fileItem.statusLog.map(log => `<div class="status-log-entry" style="color: ${getLogColorBright(log.type)}; white-space: nowrap; line-height: 1.5; font-weight: 500;">${escapeHtml(log.time)} ${escapeHtml(log.message)}</div>`).join('')
                            : `<div style="color: #94a3b8; font-size: 14px;">${fileItem.status}</div>`
                        }
                    </div>
                    ${fileItem.metrics && (fileItem.metrics.elapsed || fileItem.metrics.gpu) ? `
                    <div class="status-metrics" style="display: flex; gap: 16px; margin-top: 6px; font-size: 12px; padding: 4px 8px; background: rgba(59, 130, 246, 0.1); border-radius: 4px; border: 1px solid rgba(59, 130, 246, 0.2);">
                        ${fileItem.metrics.elapsed ? `<span style="color: #60a5fa;">⏱️ ${fileItem.metrics.elapsed}s</span>` : ''}
                        ${fileItem.metrics.gpu ? `<span style="color: #10b981;">🎮 ${escapeHtml(fileItem.metrics.gpu)}</span>` : ''}
                    </div>
                    ` : ''}
                </td>
                <td style="width: 60px;">
                    <button class="ewr-delete-file-button" onclick="deleteUnanalyzedFile('${fileItem.id}')">Delete</button>
                </td>
            </tr>
            <tr class="ewr-expandable-row ${expanded ? 'expanded' : ''}" data-file-id="${fileItem.id}-expanded">
                <td colspan="6">
                    <div class="ewr-expandable-row-content">
                        <div class="ewr-audio-player-card compact">
                            <div class="ewr-audio-player-card-metadata">
                                <div class="ewr-audio-player-card-metadata-item">
                                    <div class="ewr-audio-player-card-metadata-label">Filename</div>
                                    <div class="ewr-audio-player-card-metadata-value">${escapeHtml(fileItem.filename)}</div>
                                </div>
                                <div class="ewr-audio-player-card-metadata-item">
                                    <div class="ewr-audio-player-card-metadata-label">Size</div>
                                    <div class="ewr-audio-player-card-metadata-value">${fileItem.size} MB</div>
                                </div>
                            </div>
                            <audio controls preload="metadata" src="${audioSrc}" style="width: 100%; margin-top: 12px;"></audio>
                        </div>
                    </div>
                </td>
            </tr>
        `;
    }).join('');

    // Update counters
    elements.unanalyzedCount.textContent = `File count: ${state.unanalyzedFiles.length}`;
    updateSelectedCount();
}

function toggleRowExpansion(fileId) {
    const fileItem = state.unanalyzedFiles.find(f => f.id === fileId);
    if (!fileItem) return;

    // Collapse all other rows
    state.unanalyzedFiles.forEach(f => {
        if (f.id !== fileId) f.expanded = false;
    });

    // Toggle current row
    fileItem.expanded = !fileItem.expanded;

    updateUnanalyzedGrid();
}

function toggleFileSelection(fileId, selected) {
    const fileItem = state.unanalyzedFiles.find(f => f.id === fileId);
    if (fileItem) {
        fileItem.selected = selected;
        updateSelectedCount();
    }
}

function handleSelectAll(e) {
    const checked = e.target.checked;
    state.unanalyzedFiles.forEach(f => f.selected = checked);
    updateUnanalyzedGrid();
}

function updateSelectedCount() {
    const count = state.unanalyzedFiles.filter(f => f.selected).length;
    // Update the button text (no parentheses or count shown)
    if (!state.isProcessing) {
        elements.processSelectedBtn.textContent = 'Process Selected';
    }
    // Keep button disabled if processing or no files selected
    elements.processSelectedBtn.disabled = state.isProcessing || count === 0;
}

async function deleteUnanalyzedFile(fileId) {
    const fileItem = state.unanalyzedFiles.find(f => f.id === fileId);
    if (!fileItem) return;

    if (!confirm(`Delete "${fileItem.filename}"?`)) return;

    try {
        // If file was uploaded to server (has filepath but no local file), delete from server
        if (fileItem.filepath && !fileItem.file) {
            const response = await fetch(`/api/audio/unanalyzed/${encodeURIComponent(fileItem.filename)}`, {
                method: 'DELETE'
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.detail || 'Failed to delete file from server');
            }
        }

        // Remove from state
        const index = state.unanalyzedFiles.findIndex(f => f.id === fileId);
        if (index !== -1) {
            // Revoke blob URL if exists
            if (state.audioBlobs.has(fileId)) {
                URL.revokeObjectURL(state.audioBlobs.get(fileId));
                state.audioBlobs.delete(fileId);
            }

            state.unanalyzedFiles.splice(index, 1);
        }

        updateUnanalyzedGrid();
        showToast('File removed', 'success');

    } catch (error) {
        console.error('Delete error:', error);
        showToast(`Failed to delete: ${error.message}`, 'error');
    }
}

function getStatusColor(status) {
    switch(status) {
        case 'Pending': return '';
        case 'Processing': return 'primary';
        case 'Error': return 'danger';
        case 'Complete': return 'success';
        default: return '';
    }
}

// ============================================
// File Processing
// ============================================

async function processSelectedFiles() {
    const selectedFiles = state.unanalyzedFiles.filter(f => f.selected);

    if (selectedFiles.length === 0) {
        showToast('No files selected', 'error');
        return;
    }

    const totalFiles = selectedFiles.length;
    let processedCount = 0;

    // Set processing flag to keep button disabled
    state.isProcessing = true;
    elements.processSelectedBtn.disabled = true;
    elements.processSelectedBtn.textContent = 'Processing...';

    // Add processing counter to the left of the button
    updateProcessingCounter(1, totalFiles);

    for (const fileItem of selectedFiles) {
        try {
            fileItem.status = 'Processing';
            updateUnanalyzedGrid();

            // Upload or reference file
            let filepath;
            if (fileItem.file) {
                // Uploaded file - need to upload to server
                console.log('Uploading fileItem:', fileItem.id, fileItem.filename, 'file object:', fileItem.file);
                filepath = await uploadFileToServer(fileItem.file);
                console.log('Upload returned filepath:', filepath);
            } else {
                // Directory-polled file - already on server
                console.log('Using existing filepath for polled file:', fileItem.filepath);
                filepath = fileItem.filepath;
            }

            if (!filepath) {
                console.error('filepath is undefined/null after upload');
                throw new Error('Failed to upload file');
            }

            // Analyze file with streaming progress updates
            const analysis = await analyzeFile(filepath, fileItem.filename, fileItem.id);

            // Move to pending grid
            moveToPending(fileItem, analysis);

            // Remove from unanalyzed
            const index = state.unanalyzedFiles.findIndex(f => f.id === fileItem.id);
            if (index !== -1) {
                state.unanalyzedFiles.splice(index, 1);
            }

            processedCount++;
            updateProcessingCounter(processedCount + 1, totalFiles);

        } catch (error) {
            console.error(`Error processing ${fileItem.filename}:`, error);
            fileItem.status = 'Error';
            showToast(`Failed to process ${fileItem.filename}`, 'error');
            processedCount++;
            updateProcessingCounter(processedCount + 1, totalFiles);
        }

        updateUnanalyzedGrid();
    }

    // Clear processing flag and update button state
    state.isProcessing = false;
    clearProcessingCounter();
    updateSelectedCount();

    showToast('Processing complete', 'success');
}

// Update the processing counter display
function updateProcessingCounter(current, total) {
    let counter = document.getElementById('processingCounter');
    if (!counter) {
        // Create counter element if it doesn't exist
        const btnContainer = elements.processSelectedBtn.parentElement;
        counter = document.createElement('span');
        counter.id = 'processingCounter';
        counter.style.cssText = 'color: #10b981; font-size: 12px; font-weight: bold; margin-right: 12px;';
        btnContainer.insertBefore(counter, elements.processSelectedBtn);
    }
    // Show current/total, capped at total
    const displayCurrent = Math.min(current, total);
    counter.textContent = `(${displayCurrent}/${total})`;
}

// Clear the processing counter
function clearProcessingCounter() {
    const counter = document.getElementById('processingCounter');
    if (counter) {
        counter.remove();
    }
}

async function uploadFileToServer(file) {
    try {
        const formData = new FormData();
        formData.append('audio', file);  // Node.js expects 'audio', forwards to Python as 'file'

        console.log('Uploading file:', file.name, 'Size:', file.size, 'Type:', file.type);

        const response = await fetch('/api/audio/upload', {
            method: 'POST',
            body: formData
        });

        console.log('Upload response status:', response.status, response.statusText);

        const text = await response.text();
        console.log('Upload response body:', text);

        let data;
        try {
            data = JSON.parse(text);
        } catch (parseErr) {
            console.error('Failed to parse response as JSON:', parseErr);
            throw new Error(`Server returned non-JSON response: ${text.substring(0, 200)}`);
        }

        if (!response.ok) {
            throw new Error(data.error || `Upload failed with status ${response.status}`);
        }

        if (!data.filepath) {
            console.error('Response missing filepath:', data);
            throw new Error('Server did not return filepath');
        }

        return data.filepath;
    } catch (error) {
        console.error('uploadFileToServer error:', error);
        throw error;
    }
}

async function analyzeFile(filepath, filename, fileId) {
    // Use streaming endpoint for real-time progress updates
    console.log('Starting analysis for:', filepath);
    const response = await fetch('/api/audio/analyze-stream', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            audio_path: filepath,
            original_filename: filename,
            language: 'auto'
        })
    });

    console.log('Analysis response status:', response.status, response.statusText);
    if (!response.ok) {
        const errorText = await response.text();
        console.error('Analysis failed with response:', errorText);
        throw new Error('Analysis failed: ' + errorText);
    }

    // Read SSE stream for progress updates
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let result = null;
    let buffer = ''; // Buffer for incomplete SSE lines
    let eventCount = 0;

    while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        // Append decoded chunk to buffer
        buffer += decoder.decode(value, { stream: true });

        // Process complete lines (SSE events are separated by double newlines)
        const events = buffer.split('\n\n');
        // Keep the last potentially incomplete event in the buffer
        buffer = events.pop() || '';

        for (const event of events) {
            const lines = event.split('\n');
            for (const line of lines) {
                if (line.startsWith('data: ')) {
                    eventCount++;
                    const jsonStr = line.slice(6);
                    console.log(`SSE Event #${eventCount}:`, jsonStr.slice(0, 200) + (jsonStr.length > 200 ? '...' : ''));
                    try {
                        const data = JSON.parse(jsonStr);

                        if (data.type === 'progress') {
                            // Update status in the grid with step type for coloring
                            updateFileStatus(fileId, data.message, data.step || 'info', {
                                elapsed: data.elapsed,
                                gpu: data.gpu
                            });
                        } else if (data.type === 'complete' || data.type === 'result') {
                            // Python sends 'complete', handle both for compatibility
                            console.log('Received complete event, result keys:', Object.keys(data.result || {}));
                            result = data.result;
                            const totalTime = data.total_time || data.result?.processing_info?.total_time_seconds;
                            const gpu = data.gpu || data.result?.processing_info?.gpu;
                            updateFileStatus(fileId, '✓ Analysis complete', 'save', {
                                elapsed: totalTime,
                                gpu: gpu
                            });
                        } else if (data.type === 'error') {
                            console.error('Received error event:', data);
                            updateFileStatus(fileId, `✗ ${data.error || data.message}`, 'error');
                            throw new Error(data.error || data.message);
                        }
                    } catch (e) {
                        // Re-throw actual errors, ignore JSON parse errors
                        if (e.message && (e.message.includes('Analysis') || e.message.includes('error'))) {
                            throw e;
                        }
                        console.warn('SSE parse error:', e.message, 'Line length:', line.length, 'Preview:', line.slice(0, 100));
                    }
                }
            }
        }
    }

    console.log(`SSE stream ended. Total events: ${eventCount}, Result received: ${!!result}`);

    // Process any remaining data in buffer
    if (buffer.trim()) {
        console.log('Processing remaining buffer, length:', buffer.length, 'Preview:', buffer.slice(0, 200));
        const lines = buffer.split('\n');
        for (const line of lines) {
            if (line.startsWith('data: ')) {
                const jsonStr = line.slice(6);
                console.log('Final buffer event, length:', jsonStr.length, 'Preview:', jsonStr.slice(0, 200));
                try {
                    const data = JSON.parse(jsonStr);
                    console.log('Final buffer parsed, type:', data.type);
                    if (data.type === 'complete' || data.type === 'result') {
                        console.log('Found complete event in final buffer!');
                        result = data.result;
                        const totalTime = data.total_time || data.result?.processing_info?.total_time_seconds;
                        const gpu = data.gpu || data.result?.processing_info?.gpu;
                        updateFileStatus(fileId, '✓ Analysis complete', 'save', {
                            elapsed: totalTime,
                            gpu: gpu
                        });
                    } else if (data.type === 'error') {
                        console.error('Found error in final buffer:', data);
                        throw new Error(data.error || data.message);
                    }
                } catch (e) {
                    console.warn('Final buffer parse error:', e.message, 'Line length:', line.length);
                }
            }
        }
    }

    if (!result) {
        throw new Error('No analysis result received');
    }

    return result;
}

// Get color for log entry based on step type
function getLogColor(stepType) {
    const colors = {
        'init': '#64748b',
        'load': '#64748b',
        'transcribe': '#3b82f6',
        'chunk': '#8b5cf6',
        'transcribe_done': '#10b981',
        'diarize': '#f59e0b',
        'diarize_done': '#10b981',
        'diarize_error': '#ef4444',
        'summary': '#06b6d4',
        'metadata': '#8b5cf6',
        'customer': '#10b981',
        'content': '#3b82f6',
        'save': '#10b981',
        'error': '#ef4444',
        'waiting': '#f59e0b'
    };
    return colors[stepType] || '#94a3b8';
}

// Get brighter color for log entry (for larger, more visible status text)
function getLogColorBright(stepType) {
    const colors = {
        'init': '#94a3b8',
        'load': '#94a3b8',
        'transcribe': '#60a5fa',
        'chunk': '#a78bfa',
        'transcribe_done': '#34d399',
        'diarize': '#fbbf24',
        'diarize_done': '#34d399',
        'diarize_error': '#f87171',
        'summary': '#22d3ee',
        'metadata': '#a78bfa',
        'customer': '#34d399',
        'content': '#60a5fa',
        'llm': '#60a5fa',
        'llm_done': '#34d399',
        'save': '#4ade80',
        'error': '#f87171',
        'waiting': '#fbbf24',
        'info': '#cbd5e1'
    };
    return colors[stepType] || '#e2e8f0';
}

// Update status for a specific file in the grid
function updateFileStatus(fileId, statusMessage, stepType = 'info', extraData = {}) {
    const fileItem = state.unanalyzedFiles.find(f => f.id === fileId);
    if (fileItem) {
        // Initialize statusLog and metrics if needed
        if (!fileItem.statusLog) {
            fileItem.statusLog = [];
        }
        if (!fileItem.metrics) {
            fileItem.metrics = { elapsed: 0, gpu: null };
        }

        // Update metrics from extraData
        if (extraData.elapsed !== undefined) {
            fileItem.metrics.elapsed = extraData.elapsed;
        }
        if (extraData.gpu) {
            fileItem.metrics.gpu = extraData.gpu;
            state.currentGpu = extraData.gpu;
        }

        // Add timestamped log entry
        const now = new Date();
        const timeStr = now.toLocaleTimeString('en-US', { hour12: false, hour: '2-digit', minute: '2-digit', second: '2-digit' });

        // Include elapsed time in the log message if available
        const elapsedStr = extraData.elapsed !== undefined ? ` [${extraData.elapsed}s]` : '';

        fileItem.statusLog.push({
            time: timeStr,
            message: statusMessage + elapsedStr,
            type: stepType
        });

        // Also update the simple status
        fileItem.status = statusMessage;

        // Update grid and scroll to bottom of log
        updateUnanalyzedGrid();

        // Scroll the log container to bottom
        setTimeout(() => {
            const logContainers = document.querySelectorAll(`[data-file-id="${fileId}"] .status-log-container`);
            logContainers.forEach(container => {
                container.scrollTop = container.scrollHeight;
            });
        }, 10);
    }
}

function moveToPending(fileItem, analysis) {
    const id = `pending-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;

    // Keep audio blob reference
    if (state.audioBlobs.has(fileItem.id)) {
        state.audioBlobs.set(id, state.audioBlobs.get(fileItem.id));
        state.audioBlobs.delete(fileItem.id);
    }

    // Extract duration from audio_metadata.duration_seconds
    const durationSeconds = analysis.audio_metadata?.duration_seconds || 0;

    // Extract mood from emotions.primary
    const primaryMood = analysis.emotions?.primary || 'Unknown';

    state.pendingFiles.push({
        id,
        filename: fileItem.filename,
        duration: formatDuration(durationSeconds),
        mood: primaryMood,
        status: 'Ready',
        analysis: analysis,
        expanded: false
    });

    updatePendingGrid();
}

function formatDuration(seconds) {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
}

// ============================================
// Pending Grid Management
// ============================================

function updatePendingGrid() {
    const tbody = elements.pendingBody;
    const section = document.getElementById('pendingSection');

    if (state.pendingFiles.length === 0) {
        tbody.innerHTML = `
            <tr>
                <td colspan="6">
                    <div class="ewr-audio-empty-state">
                        <div class="ewr-audio-empty-state-icon">
                            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="20 6 9 17 4 12"/></svg>
                        </div>
                        <div class="ewr-audio-empty-state-text">No pending files</div>
                    </div>
                </td>
            </tr>
        `;
        elements.pendingCount.textContent = '0 files';
        if (section) section.classList.remove('has-content');
        return;
    }

    if (section) section.classList.add('has-content');

    tbody.innerHTML = state.pendingFiles.map(fileItem => {
        const expanded = fileItem.expanded;
        const audioSrc = state.audioBlobs.get(fileItem.id) || `/api/audio/stream/${encodeURIComponent(fileItem.filename)}`;
        const analysis = fileItem.analysis || {};

        // Extract metadata for display
        const language = analysis.language || 'Unknown';
        const subject = analysis.call_content?.subject || 'N/A';
        const outcome = analysis.call_content?.outcome || 'N/A';
        const customerName = analysis.customer_lookup?.name || analysis.call_content?.customer_name || 'N/A';
        const summaryExcerpt = (analysis.transcription_summary || 'No summary available').substring(0, 150);
        const hasSummary = analysis.transcription_summary && analysis.transcription_summary.length > 150;

        return `
            <tr class="${expanded ? 'expanded' : ''}" data-file-id="${fileItem.id}">
                <td style="width: 40px;">
                    <span class="expand-indicator" onclick="togglePendingExpansion('${fileItem.id}')">▶</span>
                </td>
                <td class="filename-cell" onclick="togglePendingExpansion('${fileItem.id}')">${escapeHtml(fileItem.filename)}</td>
                <td>${fileItem.duration}</td>
                <td><span class="ewr-count-badge warning">${fileItem.mood}</span></td>
                <td><span class="ewr-count-badge success">${fileItem.status}</span></td>
                <td style="white-space: nowrap;">
                    <button class="ewr-button ewr-button-small ewr-button-primary" onclick="openSaveModal('${fileItem.id}')">Review</button>
                    <button class="ewr-delete-file-button" onclick="deletePendingFile('${fileItem.id}')" style="margin-left: 8px;">Delete</button>
                </td>
            </tr>
            <tr class="ewr-expandable-row ${expanded ? 'expanded' : ''}" data-file-id="${fileItem.id}-expanded">
                <td colspan="6">
                    <div class="ewr-expandable-row-content">
                        <div class="ewr-audio-player-card compact">
                            <!-- Audio Player -->
                            <audio controls preload="metadata" src="${audioSrc}" style="width: 100%; margin-bottom: 16px;"></audio>

                            <!-- Metadata Grid -->
                            <div class="ewr-audio-player-card-metadata" style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px;">
                                <div class="ewr-audio-player-card-metadata-item">
                                    <div class="ewr-audio-player-card-metadata-label">Duration</div>
                                    <div class="ewr-audio-player-card-metadata-value">${fileItem.duration}</div>
                                </div>
                                <div class="ewr-audio-player-card-metadata-item">
                                    <div class="ewr-audio-player-card-metadata-label">Primary Mood</div>
                                    <div class="ewr-audio-player-card-metadata-value">${fileItem.mood}</div>
                                </div>
                                <div class="ewr-audio-player-card-metadata-item">
                                    <div class="ewr-audio-player-card-metadata-label">Language</div>
                                    <div class="ewr-audio-player-card-metadata-value">${escapeHtml(language)}</div>
                                </div>
                                <div class="ewr-audio-player-card-metadata-item">
                                    <div class="ewr-audio-player-card-metadata-label">Customer</div>
                                    <div class="ewr-audio-player-card-metadata-value">${escapeHtml(customerName)}</div>
                                </div>
                                <div class="ewr-audio-player-card-metadata-item">
                                    <div class="ewr-audio-player-card-metadata-label">Call Subject</div>
                                    <div class="ewr-audio-player-card-metadata-value">${escapeHtml(subject)}</div>
                                </div>
                                <div class="ewr-audio-player-card-metadata-item">
                                    <div class="ewr-audio-player-card-metadata-label">Outcome</div>
                                    <div class="ewr-audio-player-card-metadata-value">${escapeHtml(outcome)}</div>
                                </div>
                            </div>

                            <!-- Summary Preview -->
                            <div style="margin-top: 16px; padding-top: 16px; border-top: 1px solid #334155;">
                                <div class="ewr-audio-player-card-metadata-label" style="margin-bottom: 8px;">Summary Preview</div>
                                <div style="color: #cbd5e1; font-size: 13px; line-height: 1.5;">
                                    ${escapeHtml(summaryExcerpt)}${hasSummary ? '...' : ''}
                                </div>
                            </div>
                        </div>
                    </div>
                </td>
            </tr>
        `;
    }).join('');

    elements.pendingCount.textContent = `File count: ${state.pendingFiles.length}`;
}

function togglePendingExpansion(fileId) {
    const fileItem = state.pendingFiles.find(f => f.id === fileId);
    if (!fileItem) return;

    state.pendingFiles.forEach(f => {
        if (f.id !== fileId) f.expanded = false;
    });

    fileItem.expanded = !fileItem.expanded;
    updatePendingGrid();
}

async function deletePendingFile(fileId) {
    const fileItem = state.pendingFiles.find(f => f.id === fileId);
    if (!fileItem) return;

    if (!confirm(`Delete pending analysis for "${fileItem.filename}"?`)) return;

    try {
        // Use the JSON filename if available, otherwise construct it from the audio filename
        const jsonFilename = fileItem.pendingFilename || `${fileItem.filename}.json`;

        // Call backend API to delete the pending JSON file
        const response = await fetch(`/api/audio/pending/${encodeURIComponent(jsonFilename)}`, {
            method: 'DELETE'
        });

        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.detail || 'Failed to delete pending file');
        }

        // Remove from state
        const index = state.pendingFiles.findIndex(f => f.id === fileId);
        if (index !== -1) {
            // Clean up blob URL if exists
            if (state.audioBlobs.has(fileId)) {
                URL.revokeObjectURL(state.audioBlobs.get(fileId));
                state.audioBlobs.delete(fileId);
            }

            state.pendingFiles.splice(index, 1);
        }

        updatePendingGrid();
        showToast('Pending analysis deleted', 'success');

    } catch (error) {
        console.error('Delete pending file error:', error);
        showToast(`Failed to delete: ${error.message}`, 'error');
    }
}

async function refreshPendingFiles() {
    const refreshBtn = document.getElementById('refreshPendingBtn');
    if (refreshBtn) {
        refreshBtn.disabled = true;
        refreshBtn.textContent = 'Refreshing...';
    }

    try {
        // First, get the list of pending files (metadata only)
        const listResponse = await fetch('/api/audio/pending');

        if (!listResponse.ok) {
            throw new Error('Failed to fetch pending analyses list');
        }

        const listData = await listResponse.json();

        if (listData.success && listData.pending_files && listData.pending_files.length > 0) {
            let loadedCount = 0;

            // For each pending file, fetch its full analysis data
            for (const fileMeta of listData.pending_files) {
                // Check if file is already in pending list (by original filename stored in JSON)
                const jsonFilename = fileMeta.filename; // e.g., "recording.wav.json"
                const originalFilename = jsonFilename.replace(/\.json$/, ''); // e.g., "recording.wav"

                const existingFile = state.pendingFiles.find(f =>
                    f.filename === originalFilename || f.pendingFilename === jsonFilename
                );
                if (existingFile) continue;

                // Fetch full analysis data for this file
                try {
                    const detailResponse = await fetch(`/api/audio/pending/${encodeURIComponent(jsonFilename)}`);
                    if (!detailResponse.ok) {
                        console.error(`Failed to fetch details for ${jsonFilename}`);
                        continue;
                    }

                    const detailData = await detailResponse.json();
                    if (!detailData.success || !detailData.data) continue;

                    const fileData = detailData.data;
                    const id = `pending-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;

                    // Extract duration and mood from the analysis data
                    const durationSeconds = fileData.audio_metadata?.duration_seconds || 0;
                    const primaryMood = fileData.emotions?.primary || 'Unknown';

                    state.pendingFiles.push({
                        id,
                        filename: fileData.original_filename || originalFilename,
                        pendingFilename: jsonFilename, // Store JSON filename for deletion
                        duration: formatDuration(durationSeconds),
                        mood: primaryMood,
                        status: 'Ready',
                        analysis: fileData,
                        expanded: false
                    });

                    loadedCount++;
                } catch (detailError) {
                    console.error(`Error fetching details for ${jsonFilename}:`, detailError);
                }
            }

            updatePendingGrid();
            if (loadedCount > 0) {
                showToast(`Loaded ${loadedCount} pending file(s)`, 'success');
            }
        }
    } catch (error) {
        console.error('Error refreshing pending files:', error);
        showToast('Failed to refresh pending files', 'error');
    } finally {
        if (refreshBtn) {
            refreshBtn.disabled = false;
            refreshBtn.textContent = 'Refresh';
        }
    }
}

/**
 * Load unanalyzed files from the server (uploaded files that haven't been processed yet)
 * This ensures files persist across page reloads and server restarts
 */
async function refreshUnanalyzedFiles() {
    try {
        const response = await fetch('/api/audio/unanalyzed');

        if (!response.ok) {
            throw new Error('Failed to fetch unanalyzed files');
        }

        const data = await response.json();

        if (data.success && data.files && data.files.length > 0) {
            let loadedCount = 0;

            for (const fileInfo of data.files) {
                // Check if file is already in unanalyzed list
                const existingFile = state.unanalyzedFiles.find(f =>
                    f.filename === fileInfo.filename || f.filepath === fileInfo.filepath
                );
                if (existingFile) continue;

                const id = `server-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;

                state.unanalyzedFiles.push({
                    id,
                    filename: fileInfo.filename,
                    filepath: fileInfo.filepath,
                    size: fileInfo.size_mb.toFixed(2),
                    file: null, // No local file object, it's on the server
                    selected: true,
                    expanded: false,
                    status: 'Pending',
                    statusLog: []
                });

                loadedCount++;
            }

            if (loadedCount > 0) {
                updateUnanalyzedGrid();
                showToast(`Loaded ${loadedCount} unanalyzed file(s)`, 'success');
            }
        }
    } catch (error) {
        console.error('Error loading unanalyzed files:', error);
        // Don't show error toast on init - files may just not exist yet
    }
}

// ============================================
// Modal Management
// ============================================

// Emotion icons mapping
const EMOTION_ICONS = {
    'HAPPY': '😊',
    'SAD': '😢',
    'ANGRY': '😠',
    'NEUTRAL': '😐',
    'FEARFUL': '😨',
    'DISGUSTED': '🤢',
    'SURPRISED': '😮'
};

function openSaveModal(fileId) {
    const fileItem = state.pendingFiles.find(f => f.id === fileId);
    if (!fileItem) return;

    state.currentModal = fileItem;
    const analysis = fileItem.analysis;

    // Reset to Details tab
    switchModalTab('details');

    // Set modal title
    elements.modalTitle.textContent = `Audio Analysis: ${fileItem.filename}`;

    // Get transcription text for name detection
    const transcriptionText = typeof analysis.transcription === 'string'
        ? analysis.transcription
        : analysis.transcription?.clean_text || analysis.transcription || '';

    // Detect staff names in transcription
    const detectedNames = detectStaffNamesInTranscription(transcriptionText);
    const detectedNameDisplay = detectedNames.length > 0 ? detectedNames.join(', ') : '';

    // Details tab - editable fields
    // If a staff name was detected, auto-select it in the dropdown
    if (detectedNames.length === 1) {
        elements.modalStaff.value = detectedNames[0];
    } else {
        elements.modalStaff.value = '';
    }

    // Populate detected name field
    if (elements.modalDetectedName) {
        elements.modalDetectedName.value = detectedNameDisplay;
        // Add visual indicator if name was detected
        if (detectedNameDisplay) {
            elements.modalDetectedName.style.borderColor = '#10b981';
            elements.modalDetectedName.title = `Detected in transcript: ${detectedNameDisplay}`;
        } else {
            elements.modalDetectedName.style.borderColor = '#334155';
            elements.modalDetectedName.title = 'No staff name detected in transcription';
        }
    }

    elements.modalCustomer.value = analysis.customer_lookup?.name || analysis.call_content?.customer_name || '';
    elements.modalMood.value = analysis.emotions?.primary || '';
    elements.modalOutcome.value = analysis.call_content?.outcome || '';

    // Details tab - read-only fields
    elements.modalSubject.value = analysis.call_content?.subject || '';
    elements.modalDuration.value = fileItem.duration;
    elements.modalLanguage.value = analysis.language || 'Unknown';
    elements.modalFilename.value = fileItem.filename;

    // Summary tab
    elements.modalSummary.value = analysis.transcription_summary || 'No summary available';

    // Transcription tab - format with speaker separation
    formatTranscriptionDisplay(analysis, transcriptionText);

    // Emotions tab
    populateEmotionsGrid(analysis.emotions);

    // Store full analysis data
    elements.modalAnalysisData.value = JSON.stringify(analysis);

    elements.saveModal.style.display = 'flex';
    validateModalForm();
}

function switchModalTab(tabName) {
    // Update tab buttons
    document.querySelectorAll('.modal-tab').forEach(tab => {
        tab.classList.toggle('active', tab.dataset.tab === tabName);
    });

    // Update tab content
    document.querySelectorAll('.modal-tab-content').forEach(content => {
        content.style.display = content.dataset.tab === tabName ? 'block' : 'none';
        content.classList.toggle('active', content.dataset.tab === tabName);
    });
}

/**
 * Format and display transcription with speaker separation
 * @param {Object} analysis - The full analysis object
 * @param {string} transcriptionText - The raw transcription text
 */
function formatTranscriptionDisplay(analysis, transcriptionText) {
    const container = elements.modalTranscription;
    if (!container) return;

    const diarization = analysis.speaker_diarization || {};
    const segments = diarization.segments || [];
    const emotions = analysis.emotions || {};
    const primaryEmotion = emotions.primary || '';

    // If we have speaker segments, format them nicely
    if (segments.length > 0) {
        let html = '';
        segments.forEach((segment, index) => {
            const speaker = segment.speaker || 'Unknown';
            const text = segment.text || '';
            const speakerClass = speaker.toLowerCase().replace(' ', '-');

            // Add separator between segments (except first)
            if (index > 0) {
                html += '<hr class="transcript-separator">';
            }

            // Check if this segment has an emotion tag in the text
            let emotionTag = '';
            const emotionMatch = text.match(/\[(HAPPY|SAD|ANGRY|NEUTRAL|FEARFUL|DISGUSTED|SURPRISED)\]/i);
            if (emotionMatch) {
                emotionTag = `<span class="transcript-emotion">[${emotionMatch[1]}]</span>`;
            }

            // Clean the text of emotion tags for display
            const cleanText = text.replace(/\[(HAPPY|SAD|ANGRY|NEUTRAL|FEARFUL|DISGUSTED|SURPRISED)\]/gi, '').trim();

            html += `
                <div class="transcript-segment">
                    <span class="transcript-speaker ${speakerClass}">${escapeHtml(speaker)}</span>
                    ${emotionTag}
                    <div class="transcript-text">${escapeHtml(cleanText)}</div>
                </div>
            `;
        });
        container.innerHTML = html;
    } else {
        // Fallback: Try to parse speaker labels from transcription text
        const lines = transcriptionText.split('\n');
        let html = '';
        let lastSpeaker = '';

        lines.forEach((line, index) => {
            line = line.trim();
            if (!line || line === '─'.repeat(40)) return;

            // Check for Caller 1/Caller 2 pattern
            const speakerMatch = line.match(/^(Caller [12]):\s*(.*)/i);
            if (speakerMatch) {
                const speaker = speakerMatch[1];
                const text = speakerMatch[2];
                const speakerClass = speaker.toLowerCase().replace(' ', '-');

                // Add separator when speaker changes
                if (lastSpeaker && lastSpeaker !== speaker) {
                    html += '<hr class="transcript-separator">';
                }
                lastSpeaker = speaker;

                // Check for emotion
                let emotionTag = '';
                const emotionMatch = text.match(/\[(HAPPY|SAD|ANGRY|NEUTRAL|FEARFUL|DISGUSTED|SURPRISED)\]/i);
                if (emotionMatch) {
                    emotionTag = `<span class="transcript-emotion">[${emotionMatch[1]}]</span>`;
                }
                const cleanText = text.replace(/\[(HAPPY|SAD|ANGRY|NEUTRAL|FEARFUL|DISGUSTED|SURPRISED)\]/gi, '').trim();

                html += `
                    <div class="transcript-segment">
                        <span class="transcript-speaker ${speakerClass}">${escapeHtml(speaker)}</span>
                        ${emotionTag}
                        <div class="transcript-text">${escapeHtml(cleanText)}</div>
                    </div>
                `;
            } else if (line) {
                // Plain text without speaker label
                html += `<div class="transcript-segment"><div class="transcript-text">${escapeHtml(line)}</div></div>`;
            }
        });

        // If no speaker formatting was found, just display as plain text
        if (!html) {
            html = `<div class="transcript-text" style="white-space: pre-wrap;">${escapeHtml(transcriptionText)}</div>`;
        }

        container.innerHTML = html;
    }
}

function populateEmotionsGrid(emotions) {
    if (!elements.emotionsGrid) return;

    const allEmotions = ['HAPPY', 'SAD', 'ANGRY', 'NEUTRAL', 'FEARFUL', 'DISGUSTED', 'SURPRISED'];
    const primaryEmotion = emotions?.primary?.toUpperCase() || '';
    const detectedEmotions = (emotions?.detected || []).map(e => e.toUpperCase());

    elements.emotionsGrid.innerHTML = allEmotions.map(emotion => {
        const isPrimary = emotion === primaryEmotion;
        const isDetected = detectedEmotions.includes(emotion);
        const cardClass = isPrimary ? 'primary' : (isDetected ? 'detected' : '');
        const badgeClass = isPrimary ? 'primary' : (isDetected ? 'detected' : '');
        const badgeText = isPrimary ? 'PRIMARY' : (isDetected ? 'DETECTED' : '');

        return `
            <div class="emotion-card ${cardClass}">
                <div class="emotion-icon">${EMOTION_ICONS[emotion] || '❓'}</div>
                <div class="emotion-name">${emotion}</div>
                ${badgeText ? `<span class="emotion-badge ${badgeClass}">${badgeText}</span>` : ''}
            </div>
        `;
    }).join('');
}

function closeSaveModal() {
    elements.saveModal.style.display = 'none';
    state.currentModal = null;
}

function validateModalForm() {
    const staff = elements.modalStaff.value.trim();
    const customer = elements.modalCustomer.value.trim();
    const mood = elements.modalMood.value;

    elements.saveAnalysisBtn.disabled = !(staff && customer && mood);
}

async function saveAnalysis() {
    if (!state.currentModal) return;

    const fileItem = state.currentModal;
    const analysis = JSON.parse(elements.modalAnalysisData.value);

    // Build payload with metadata wrapper for Node.js route compatibility
    const payload = {
        transcription: analysis.transcription || '',
        raw_transcription: analysis.raw_transcription || '',
        transcription_summary: analysis.transcription_summary || null,
        emotions: analysis.emotions || { primary: 'NEUTRAL', detected: [], timestamps: [] },
        audio_events: analysis.audio_events || { detected: [], timestamps: [] },
        language: analysis.language || 'en',
        audio_metadata: analysis.audio_metadata || null,
        metadata: {
            customer_support_staff: elements.modalStaff.value.trim(),
            ewr_customer: elements.modalCustomer.value.trim(),
            mood: elements.modalMood.value,
            outcome: elements.modalOutcome.value || 'Undetermined',
            filename: fileItem.filename
        },
        call_metadata: analysis.call_metadata || null,
        call_content: analysis.call_content || null,
        pending_filename: fileItem.pendingFilename || null,
        speaker_diarization: analysis.speaker_diarization || {
            enabled: false,
            segments: [],
            statistics: {},
            num_speakers: 0
        }
    };

    try {
        elements.saveAnalysisBtn.disabled = true;
        elements.saveAnalysisBtn.textContent = 'Saving...';

        // Validate customer support staff is selected (client-side check)
        if (!elements.modalStaff.value.trim()) {
            showToast('Customer Support Staff is required. Please select a staff member.', 'error');
            return;
        }

        const response = await fetch('/api/audio/store', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });

        if (!response.ok) {
            const errorData = await response.json().catch(() => ({}));
            const errorMsg = errorData.detail || 'Failed to save analysis';
            throw new Error(errorMsg);
        }

        // Remove from pending list
        const index = state.pendingFiles.findIndex(f => f.id === fileItem.id);
        if (index !== -1) {
            // Clean up blob URL
            if (state.audioBlobs.has(fileItem.id)) {
                URL.revokeObjectURL(state.audioBlobs.get(fileItem.id));
                state.audioBlobs.delete(fileItem.id);
            }

            state.pendingFiles.splice(index, 1);
        }

        updatePendingGrid();
        closeSaveModal();
        showToast('Analysis saved successfully', 'success');

    } catch (error) {
        console.error('Save error:', error);
        showToast('Failed to save analysis', 'error');
    } finally {
        elements.saveAnalysisBtn.disabled = false;
        elements.saveAnalysisBtn.textContent = 'Save to MongoDB';
    }
}

// ============================================
// Utilities
// ============================================

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

function showToast(message, type = 'info') {
    // Create toast element
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    toast.style.cssText = `
        position: fixed;
        top: 80px;
        right: 20px;
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 16px 20px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
        z-index: 3000;
        min-width: 300px;
        animation: slideIn 0.3s ease-out;
        border-left: 4px solid ${type === 'success' ? '#10b981' : type === 'error' ? '#ef4444' : '#3b82f6'};
    `;

    toast.innerHTML = `
        <div style="color: #f1f5f9; font-size: 14px;">${escapeHtml(message)}</div>
    `;

    document.body.appendChild(toast);

    setTimeout(() => {
        toast.style.animation = 'slideOut 0.3s ease-in';
        setTimeout(() => toast.remove(), 300);
    }, 3000);
}

// CSS animations
const style = document.createElement('style');
style.textContent = `
    @keyframes slideIn {
        from {
            transform: translateX(400px);
            opacity: 0;
        }
        to {
            transform: translateX(0);
            opacity: 1;
        }
    }
    @keyframes slideOut {
        from {
            transform: translateX(0);
            opacity: 1;
        }
        to {
            transform: translateX(400px);
            opacity: 0;
        }
    }
`;
document.head.appendChild(style);

// ============================================
// Export Global Functions (for onclick handlers)
// ============================================

window.initAudioManager = initAudioManager;
window.toggleRowExpansion = toggleRowExpansion;
window.togglePendingExpansion = togglePendingExpansion;
window.toggleFileSelection = toggleFileSelection;
window.deleteUnanalyzedFile = deleteUnanalyzedFile;
window.deletePendingFile = deletePendingFile;
window.openSaveModal = openSaveModal;
window.closeSaveModal = closeSaveModal;
window.saveAnalysis = saveAnalysis;
window.switchModalTab = switchModalTab;
window.refreshPendingFiles = refreshPendingFiles;
window.refreshUnanalyzedFiles = refreshUnanalyzedFiles;
