// RAG Admin Interface - Client-side JavaScript

// Initialize auth client
const auth = new AuthClient();

// Check authentication on page load
if (!auth.isAuthenticated()) {
    window.location.href = '/login.html?redirect=' + encodeURIComponent(window.location.pathname);
} else {
    // Check if user is admin
    auth.getUser().then(user => {
        if (user.role !== 'admin') {
            alert('Admin access required');
            window.location.href = '/';
        }
    }).catch(error => {
        console.error('Failed to verify admin access:', error);
        window.location.href = '/login.html?redirect=' + encodeURIComponent(window.location.pathname);
    });
}

// Load available LLM models
async function loadModels() {
    try {
        const response = await fetch('/api/llm/models');
        const data = await response.json();

        const select = document.getElementById('llmModel');
        select.innerHTML = '';

        if (data.models && data.models.length > 0) {
            data.models.forEach(model => {
                const option = document.createElement('option');
                option.value = model.name;
                option.textContent = model.name;
                select.appendChild(option);
            });

            // Load saved model selection
            const savedSettings = localStorage.getItem('ragSystemSettings');
            if (savedSettings) {
                try {
                    const settings = JSON.parse(savedSettings);
                    if (settings.model) {
                        select.value = settings.model;
                    }
                } catch (error) {
                    console.error('Failed to load saved model:', error);
                }
            }
        } else {
            select.innerHTML = '<option value="">No models found</option>';
        }
    } catch (error) {
        console.error('Failed to load models:', error);
        const select = document.getElementById('llmModel');
        select.innerHTML = '<option value="">Error loading models</option>';
    }
}

// Refresh models list
async function refreshModels() {
    const select = document.getElementById('llmModel');
    select.innerHTML = '<option value="">Loading models...</option>';
    await loadModels();
}

// Initialize
window.addEventListener('DOMContentLoaded', () => {
    // Load saved settings
    loadSavedSettings();
    // Load available models
    loadModels();
});

// Load projects
async function loadProjects() {
    try {
        const response = await fetch('/api/admin/projects');

        if (!response.ok) {
            throw new Error('Failed to load projects');
        }

        const data = await response.json();
        renderProjects(data.projects);

    } catch (error) {
        console.error('Failed to load projects:', error);
        document.getElementById('projectsList').innerHTML =
            `<p style="color: #ef4444;">Failed to load projects: ${error.message}</p>`;
    }
}

function renderProjects(projects) {
    const projectsList = document.getElementById('projectsList');

    if (projects.length === 0) {
        projectsList.innerHTML = '<p style="color: #6b7280;">No projects yet. Upload documents to create one.</p>';
        return;
    }

    projectsList.innerHTML = projects.map(project => `
        <div class="project-card" onclick="openProjectPage('${project.id}')" style="cursor: pointer;">
            <h3>${project.name}</h3>
            <p>${project.description || 'No description'}</p>
            <span class="badge">${project.documentCount || 0} documents</span>
        </div>
    `).join('');
}

function openProjectPage(projectId) {
    window.location.href = `project.html?project=${projectId}`;
}

function selectProjectForBrowsing(projectId) {
    const browseSelect = document.getElementById('browseProjectSelect');
    browseSelect.value = projectId;
    loadDocuments();
    // Scroll to documents section
    document.getElementById('documentsList').scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

// Document browsing
async function loadDocuments() {
    const documentsList = document.getElementById('documentsList');

    documentsList.innerHTML = '<p style="color: #6b7280;">Loading uploaded documents...</p>';

    try {
        const response = await fetch(`/api/admin/documents?limit=100`);

        if (!response.ok) {
            throw new Error('Failed to load documents');
        }

        const data = await response.json();
        renderDocuments(data.documents);

    } catch (error) {
        console.error('Failed to load documents:', error);
        documentsList.innerHTML = `<p style="color: #ef4444;">Failed to load documents: ${error.message}</p>`;
    }
}

function renderDocuments(documents) {
    const documentsList = document.getElementById('documentsList');

    if (documents.length === 0) {
        documentsList.innerHTML = '<p style="color: #6b7280;">No documents found. Upload documents using the Upload page.</p>';
        return;
    }

    documentsList.innerHTML = '<div class="doc-list">' + documents.map((doc, index) => {
        const fileName = doc.file || doc.metadata?.fileName || 'Untitled Document';
        const category = doc.category || doc.metadata?.category || 'Unknown';
        const documentType = doc.documentType || doc.metadata?.documentType || '';
        const uploadDate = doc.uploadDate ? new Date(doc.uploadDate).toLocaleDateString() : 'Unknown';
        const chunks = doc.chunks || 1;
        const docId = doc.id || doc.metadata?.document_id || index;

        return `
        <div class="doc-item">
            <div class="doc-item-info" onclick="viewDocument(${index})">
                <div class="doc-item-title">📄 ${fileName}</div>
                <div class="doc-item-meta">
                    <strong>Category:</strong> ${category} |
                    <strong>Type:</strong> ${documentType} |
                    <strong>Uploaded:</strong> ${uploadDate} |
                    <strong>Chunks:</strong> ${chunks}
                </div>
            </div>
            <div class="doc-item-actions">
                <button class="btn btn-small btn-view" onclick="viewDocument(${index})">View</button>
                <button class="btn btn-small btn-delete" onclick="deleteDocument('${docId}')">Delete</button>
            </div>
        </div>
        `;
    }).join('') + '</div>';

    // Store documents globally for viewing
    window.currentDocuments = documents;
}

async function viewDocument(index) {
    const doc = window.currentDocuments[index];
    const modal = document.getElementById('docModal');
    const title = document.getElementById('docTitle');
    const meta = document.getElementById('docMeta');
    const content = document.getElementById('docContent');

    title.textContent = doc.file || doc.metadata?.fileName || 'Document';

    // Show loading state
    content.textContent = 'Loading document content...';
    modal.classList.remove('hidden');

    try {
        // Fetch full document content from server
        const response = await fetch(`/api/admin/documents/${doc.id}/content`, {
            headers: {
                // Auth removed
            }
        });

        if (!response.ok) {
            throw new Error('Failed to fetch document content');
        }

        const data = await response.json();
        const fullDoc = data.document;

        // Update metadata with full info
        meta.innerHTML = `
            <strong>File:</strong> ${fullDoc.fileName || 'Unknown'}<br>
            <strong>Category:</strong> ${fullDoc.category || 'Unknown'}<br>
            <strong>Document Type:</strong> ${fullDoc.documentType || 'Unknown'}<br>
            <strong>Upload Date:</strong> ${new Date(fullDoc.uploadDate).toLocaleString()}<br>
            <strong>File Size:</strong> ${formatFileSize(fullDoc.fileSize || 0)}<br>
            <strong>Chunks:</strong> ${fullDoc.chunks || 1}<br>
            <strong>Content Length:</strong> ${fullDoc.content.length} characters
        `;

        // Display full content
        content.textContent = fullDoc.content;

    } catch (error) {
        console.error('Failed to fetch document:', error);
        content.textContent = 'Error loading document content: ' + error.message;
    }
}

function formatFileSize(bytes) {
    if (bytes < 1024) return bytes + ' B';
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
    return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
}

function closeDocModal(event) {
    // Close if clicking backdrop or close button
    if (!event || event.target.id === 'docModal') {
        document.getElementById('docModal').classList.add('hidden');
    }
}

async function deleteDocument(documentId) {
    if (!documentId) {
        alert('Document ID not found');
        return;
    }

    if (!confirm('Are you sure you want to delete this document? This action cannot be undone.')) {
        return;
    }

    try {
        const response = await fetch(`/api/admin/documents/${documentId}`, {
            method: 'DELETE',
            headers: {
                'Authorization': `Bearer ${auth.getAccessToken()}`
            }
        });

        const result = await response.json();

        if (result.success) {
            alert('Document deleted successfully');
            loadDocuments(); // Reload the list
        } else {
            alert('Failed to delete document: ' + (result.message || result.error));
        }

    } catch (error) {
        console.error('Delete error:', error);
        alert('Failed to delete document: ' + error.message);
    }
}

// System Settings Functions
function updateTemperatureValue() {
    const slider = document.getElementById('temperatureSlider');
    const valueDisplay = document.getElementById('temperatureValue');
    valueDisplay.textContent = parseFloat(slider.value).toFixed(1);
}

function updateContextWindowValue() {
    const slider = document.getElementById('contextWindowSlider');
    const valueDisplay = document.getElementById('contextWindowValue');
    const value = parseInt(slider.value);
    // Format large numbers with K suffix
    if (value >= 1000) {
        valueDisplay.textContent = Math.round(value / 1024) + 'K';
    } else {
        valueDisplay.textContent = value;
    }
}

function loadSavedSettings() {
    // Load from localStorage
    const savedSettings = localStorage.getItem('ragSystemSettings');
    if (savedSettings) {
        try {
            const settings = JSON.parse(savedSettings);
            document.getElementById('temperatureSlider').value = settings.temperature || 0.7;
            document.getElementById('contextSize').value = settings.contextSize || 2000;
            document.getElementById('numSources').value = settings.numSources || 3;

            // Load SQL Assistant context settings
            if (document.getElementById('contextWindowSlider')) {
                document.getElementById('contextWindowSlider').value = settings.contextWindowSize || 8192;
                updateContextWindowValue();
            }
            if (document.getElementById('maxConversationMessages')) {
                document.getElementById('maxConversationMessages').value = settings.maxConversationMessages || 6;
            }

            updateTemperatureValue();
        } catch (error) {
            console.error('Failed to load saved settings:', error);
        }
    }
}

async function saveSettings() {
    const temperature = parseFloat(document.getElementById('temperatureSlider').value);
    const contextSize = parseInt(document.getElementById('contextSize').value);
    const numSources = parseInt(document.getElementById('numSources').value);
    const model = document.getElementById('llmModel').value;

    // Get SQL Assistant context settings
    const contextWindowSlider = document.getElementById('contextWindowSlider');
    const maxConvMessages = document.getElementById('maxConversationMessages');
    const contextWindowSize = contextWindowSlider ? parseInt(contextWindowSlider.value) : 8192;
    const maxConversationMessages = maxConvMessages ? parseInt(maxConvMessages.value) : 6;

    const settings = {
        temperature,
        contextSize,
        numSources,
        model,
        contextWindowSize,
        maxConversationMessages
    };

    // Save to localStorage
    localStorage.setItem('ragSystemSettings', JSON.stringify(settings));

    // Also save SQL-specific settings separately for the SQL chat page to access
    localStorage.setItem('sqlContextWindowSize', contextWindowSize);
    localStorage.setItem('sqlMaxConversationMessages', maxConversationMessages);

    // Also save to server
    try {
        const response = await fetch('/api/admin/settings', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                // Auth removed
            },
            body: JSON.stringify(settings)
        });

        const result = await response.json();

        if (result.success) {
            showSettingsMessage('Settings saved successfully', 'success');
        } else {
            showSettingsMessage('Failed to save settings: ' + result.message, 'error');
        }
    } catch (error) {
        // Even if server save fails, local save succeeded
        showSettingsMessage('Settings saved locally (server save failed: ' + error.message + ')', 'success');
    }
}

function showSettingsMessage(message, type) {
    const messageDiv = document.getElementById('settingsMessage');
    messageDiv.innerHTML = `<div class="message ${type}">${message}</div>`;

    setTimeout(() => {
        if (type === 'success') {
            messageDiv.innerHTML = '';
        }
    }, 5000);
}

async function updateContextDatabase() {
    const repoSelect = document.getElementById('repoSelect');
    const repo = repoSelect.value;
    const messageDiv = document.getElementById('contextUpdateMessage');
    const button = event.target;

    if (!repo) {
        showContextUpdateMessage('Please select a repository', 'error');
        return;
    }

    button.disabled = true;
    button.textContent = 'Updating...';
    messageDiv.innerHTML = '<p style="color: #6b7280;">Updating context database. This may take several minutes...</p>';

    try {
        const response = await fetch('/api/admin/git/update-context', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                // Auth removed
            },
            body: JSON.stringify({
                repo: repo === 'all' ? undefined : repo,
                changedOnly: false
            })
        });

        const result = await response.json();

        if (result.success) {
            const summary = `
                <div class="message success">
                    <strong>Context database updated successfully</strong><br>
                    ${result.message || ''}<br>
                    ${result.filesProcessed ? `Files processed: ${result.filesProcessed}` : ''}
                    ${result.duration ? ` (${result.duration})` : ''}
                </div>
            `;
            messageDiv.innerHTML = summary;
        } else {
            showContextUpdateMessage('Update failed: ' + result.message, 'error');
        }
    } catch (error) {
        showContextUpdateMessage('Update error: ' + error.message, 'error');
    } finally {
        button.disabled = false;
        button.textContent = 'Update Context Database';
    }
}

function showContextUpdateMessage(message, type) {
    const messageDiv = document.getElementById('contextUpdateMessage');
    messageDiv.innerHTML = `<div class="message ${type}">${message}</div>`;
}

// Database Schema Extraction Functions
function toggleSchemaAuth() {
    const authType = document.getElementById('schemaAuthType').value;
    const sqlAuthFields = document.getElementById('schemaSqlAuthFields');

    if (authType === 'windows') {
        sqlAuthFields.style.display = 'none';
    } else {
        sqlAuthFields.style.display = 'block';
    }
}

function toggleSchemaComparison() {
    const checkbox = document.getElementById('enableComparison');
    const compareDbGroup = document.getElementById('compareDbGroup');

    if (checkbox.checked) {
        compareDbGroup.style.display = 'block';
    } else {
        compareDbGroup.style.display = 'none';
    }
}

async function testSchemaConnection() {
    const server = document.getElementById('schemaServer').value;
    const authType = document.getElementById('schemaAuthType').value;
    const username = document.getElementById('schemaUsername').value;
    const password = document.getElementById('schemaPassword').value;
    const messageDiv = document.getElementById('schemaMessage');
    const extractBtn = document.getElementById('extractSchemaBtn');
    const databaseSelect = document.getElementById('schemaDatabase');
    const compareDbSelect = document.getElementById('schemaCompareDatabase');

    if (!server) {
        showSchemaMessage('Please enter a server name', 'error');
        return;
    }

    if (authType === 'sql' && (!username || !password)) {
        showSchemaMessage('Please enter username and password', 'error');
        return;
    }

    messageDiv.innerHTML = '<p style="color: var(--text-secondary);">Testing connection...</p>';

    try {
        const response = await fetch('/api/admin/list-databases', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${auth.getAccessToken()}`
            },
            body: JSON.stringify({
                server,
                authType,
                username,
                password
            })
        });

        const result = await response.json();

        if (result.success) {
            // Populate database dropdowns
            databaseSelect.innerHTML = '<option value="">Select a database...</option>';
            compareDbSelect.innerHTML = '<option value="">Select a database to compare...</option>';

            result.databases.forEach(db => {
                const option1 = document.createElement('option');
                option1.value = db;
                option1.textContent = db;
                databaseSelect.appendChild(option1);

                const option2 = document.createElement('option');
                option2.value = db;
                option2.textContent = db;
                compareDbSelect.appendChild(option2);
            });

            databaseSelect.disabled = false;
            compareDbSelect.disabled = false;
            extractBtn.disabled = false;

            showSchemaMessage(`Connection successful! Found ${result.databases.length} databases.`, 'success');
        } else {
            showSchemaMessage('Connection failed: ' + result.error, 'error');
        }

    } catch (error) {
        console.error('Connection test error:', error);
        showSchemaMessage('Connection error: ' + error.message, 'error');
    }
}

async function extractDatabaseSchema() {
    const server = document.getElementById('schemaServer').value;
    const database = document.getElementById('schemaDatabase').value;
    const authType = document.getElementById('schemaAuthType').value;
    const username = document.getElementById('schemaUsername').value;
    const password = document.getElementById('schemaPassword').value;
    const messageDiv = document.getElementById('schemaMessage');
    const button = document.getElementById('extractSchemaBtn');

    if (!database) {
        showSchemaMessage('Please select a database', 'error');
        return;
    }

    button.disabled = true;
    button.textContent = 'Extracting...';

    messageDiv.innerHTML = `
        <div style="color: var(--text-secondary); margin-top: 16px;">
            <p>Starting schema extraction for ${database}...</p>
            <div class="progress">
                <div id="schemaProgressBar" class="progress-bar" style="width: 0%"></div>
            </div>
            <p id="schemaStatusText" style="margin-top: 8px; font-size: 14px;">Initializing...</p>
        </div>
    `;

    try {
        // Use fetch with streaming response
        const response = await fetch('/api/sql/extract-schema-stream', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                server,
                database,
                authType,
                user: username,
                password,
                lookupKey: database.toLowerCase()
            })
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const reader = response.body.getReader();
        const decoder = new TextDecoder();

        let buffer = '';

        while (true) {
            const { done, value } = await reader.read();

            if (done) break;

            buffer += decoder.decode(value, { stream: true });
            const lines = buffer.split('\n');
            buffer = lines.pop(); // Keep incomplete line in buffer

            for (const line of lines) {
                if (line.startsWith('event:')) {
                    const eventType = line.substring(6).trim();
                    continue;
                }

                if (line.startsWith('data:')) {
                    const data = JSON.parse(line.substring(5).trim());

                    const statusText = document.getElementById('schemaStatusText');
                    const progressBar = document.getElementById('schemaProgressBar');

                    if (data.phase === 'connecting') {
                        statusText.textContent = data.message;
                        progressBar.style.width = '15%';
                    } else if (data.phase === 'connected') {
                        statusText.textContent = data.message;
                        progressBar.style.width = '20%';
                    } else if (data.phase === 'tables') {
                        statusText.textContent = data.message;
                        progressBar.style.width = '30%';
                    } else if (data.phase === 'procedures') {
                        statusText.textContent = data.message;
                        progressBar.style.width = '40%';
                    } else if (data.phase === 'processing') {
                        statusText.textContent = data.message;
                        if (data.total && data.current) {
                            const percent = Math.min(90, 40 + (data.current / data.total * 40));
                            progressBar.style.width = percent + '%';
                        }
                    } else if (data.phase === 'stored_procedures') {
                        statusText.textContent = data.message;
                        if (data.total && data.current) {
                            const percent = Math.min(95, 80 + (data.current / data.total * 15));
                            progressBar.style.width = percent + '%';
                        }
                    } else if (data.tableCount !== undefined) {
                        // Complete event
                        progressBar.style.width = '100%';
                        statusText.textContent = data.message;

                        showSchemaMessage(
                            `Schema extraction complete!<br>` +
                            `Tables processed: ${data.tableCount}<br>` +
                            `Procedures processed: ${data.procedureCount}`,
                            'success'
                        );

                        button.disabled = false;
                        button.textContent = 'Extract Schema';
                    } else if (data.message && data.message.includes('failed')) {
                        // Error event
                        showSchemaMessage('Extraction failed: ' + data.message, 'error');
                        button.disabled = false;
                        button.textContent = 'Extract Schema';
                    }
                }
            }
        }

    } catch (error) {
        console.error('Schema extraction error:', error);
        showSchemaMessage('Extraction error: ' + error.message, 'error');
        button.disabled = false;
        button.textContent = 'Extract Schema';
    }
}

function showSchemaMessage(message, type) {
    const messageDiv = document.getElementById('schemaMessage');
    messageDiv.innerHTML = `<div class="message ${type}">${message}</div>`;
}

async function runRoslynAnalysis() {
    const projectSelect = document.getElementById('roslynProjectSelect');
    const modeSelect = document.getElementById('roslynModeSelect');
    const project = projectSelect.value;
    const mode = modeSelect.value;
    const messageDiv = document.getElementById('roslynMessage');
    const progressDiv = document.getElementById('roslynProgress');
    const progressBar = document.getElementById('roslynProgressBar');
    const statusText = document.getElementById('roslynStatus');
    const button = event.target;

    if (!project) {
        showRoslynMessage('Please select a project', 'error');
        return;
    }

    button.disabled = true;
    button.textContent = 'Analyzing...';
    messageDiv.innerHTML = '';
    progressDiv.classList.remove('hidden');
    progressBar.style.width = '10%';
    statusText.textContent = 'Initializing Roslyn analyzer...';

    try {
        // Fetch project paths from API endpoint
        const configResponse = await fetch('/api/admin/git/repositories');
        const configData = await configResponse.json();

        const projectPaths = {};
        if (configData.success && configData.repositories) {
            // Build projectPaths from API data
            configData.repositories.forEach(repo => {
                projectPaths[repo.name.toLowerCase()] = repo.path;
            });
        }

        // Add 'all' option
        projectPaths['all'] = 'all';

        const inputPath = projectPaths[project];
        if (!inputPath) {
            showRoslynMessage('Invalid project selected', 'error');
            return;
        }

        progressBar.style.width = '30%';
        statusText.textContent = 'Analyzing C# code...';

        // Call the new Roslyn API endpoint that proxies to Python service
        const response = await fetch('/api/roslyn/analyze', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${auth.getAccessToken()}`
            },
            body: JSON.stringify({
                input_path: inputPath,
                project: project,
                store: true
            })
        });

        progressBar.style.width = '70%';
        statusText.textContent = 'Storing results in MongoDB...';

        const result = await response.json();

        progressBar.style.width = '100%';

        if (result.success) {
            const summary = `
                <div class="message success">
                    <strong>Roslyn Analysis Complete</strong><br><br>
                    <strong>Project:</strong> ${project}<br>
                    <strong>Files Analyzed:</strong> ${result.files_analyzed || 'N/A'}<br>
                    <strong>Classes Found:</strong> ${result.classes_count || 'N/A'}<br>
                    <strong>Methods Found:</strong> ${result.methods_count || 'N/A'}<br>
                    <strong>Status:</strong> ${result.message || 'Completed successfully'}
                </div>
            `;
            messageDiv.innerHTML = summary;
            statusText.textContent = 'Analysis completed successfully';
        } else {
            showRoslynMessage('Analysis failed: ' + (result.error || result.message || 'Unknown error'), 'error');
            statusText.textContent = 'Analysis failed';
        }
    } catch (error) {
        showRoslynMessage('Analysis error: ' + error.message, 'error');
        statusText.textContent = 'Error occurred';
    } finally {
        setTimeout(() => {
            progressDiv.classList.add('hidden');
            progressBar.style.width = '0%';
        }, 3000);
        button.disabled = false;
        button.textContent = 'Run Roslyn Analysis';
    }
}

function showRoslynMessage(message, type) {
    const messageDiv = document.getElementById('roslynMessage');
    messageDiv.innerHTML = `<div class="message ${type}">${message}</div>`;
}


function showSchemaMessage(message, type) {
    const messageDiv = document.getElementById('schemaMessage');
    messageDiv.innerHTML = `<div class="message ${type}">${message}</div>`;
}

// Toggle authentication fields for schema connection
function toggleSchemaAuth() {
    const authType = document.getElementById('schemaAuthType').value;
    const sqlAuthFields = document.getElementById('schemaSqlAuthFields');

    if (authType === 'windows') {
        sqlAuthFields.style.display = 'none';
    } else {
        sqlAuthFields.style.display = 'block';
    }
}

// Test database connection for schema extraction
async function testSchemaConnection() {
    const server = document.getElementById('schemaServer').value;
    const authType = document.getElementById('schemaAuthType').value;
    const username = authType === 'sql' ? document.getElementById('schemaUsername').value : '';
    const password = authType === 'sql' ? document.getElementById('schemaPassword').value : '';
    const messageDiv = document.getElementById('schemaMessage');
    const button = event.target;

    if (!server) {
        showSchemaMessage('❌ Server is required', 'error');
        return;
    }

    button.disabled = true;
    button.textContent = 'Testing Connection...';
    messageDiv.innerHTML = '<p style="color: #cbd5e1;">Testing connection to server...</p>';

    try {
        const response = await fetch('/api/admin/list-databases', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                // Auth removed
            },
            body: JSON.stringify({
                server,
                authType,
                username,
                password
            })
        });

        const result = await response.json();

        if (result.success) {
            // Populate database dropdowns
            const dbSelect = document.getElementById('schemaDatabase');
            const compareDbSelect = document.getElementById('schemaCompareDatabase');

            dbSelect.innerHTML = '';
            compareDbSelect.innerHTML = '';

            result.databases.forEach(db => {
                const option1 = document.createElement('option');
                option1.value = db;
                option1.textContent = db;
                dbSelect.appendChild(option1);

                const option2 = document.createElement('option');
                option2.value = db;
                option2.textContent = db;
                compareDbSelect.appendChild(option2);
            });

            // Enable dropdowns and buttons
            dbSelect.disabled = false;
            document.getElementById('extractSchemaBtn').disabled = false;

            showSchemaMessage(`Connection successful! Loaded ${result.databases.length} databases from ${server}`, 'success');
        } else {
            showSchemaMessage(`Connection failed: ${result.error}`, 'error');
        }

    } catch (error) {
        showSchemaMessage(`Error: ${error.message}`, 'error');
    } finally {
        button.disabled = false;
        button.textContent = 'Test Connection';
    }
}

// Updated extractDatabaseSchema to use connection settings
async function extractDatabaseSchema() {
    const server = document.getElementById('schemaServer').value;
    const database = document.getElementById('schemaDatabase').value;
    const authType = document.getElementById('schemaAuthType').value;
    const username = authType === 'sql' ? document.getElementById('schemaUsername').value : '';
    const password = authType === 'sql' ? document.getElementById('schemaPassword').value : '';
    const messageDiv = document.getElementById('schemaMessage');
    const button = event.target;

    if (!server || !database) {
        showSchemaMessage('Server and Database are required', 'error');
        return;
    }

    button.disabled = true;
    button.textContent = 'Extracting Schema...';
    messageDiv.innerHTML = '';

    try {
        const response = await fetch('/api/admin/extract-schema', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${auth.getAccessToken()}`
            },
            body: JSON.stringify({
                server,
                database,
                authType,
                username,
                password
            })
        });

        const result = await response.json();

        if (result.success) {
            showSchemaMessage(
                `Schema extraction started successfully!<br>` +
                `<strong>Database:</strong> ${database}<br>` +
                `<strong>Process ID:</strong> ${result.pid}<br>` +
                `<em>Check server logs for progress. This may take 1-5 minutes depending on database size.</em>`,
                'success'
            );
        } else {
            showSchemaMessage(`Schema extraction failed: ${result.error}`, 'error');
        }

    } catch (error) {
        showSchemaMessage(`Error: ${error.message}`, 'error');
    } finally {
        button.disabled = false;
        button.textContent = 'Extract Schema';
    }
}

// Analyze stored procedures
async function analyzeStoredProcedures() {
    const server = document.getElementById('schemaServer').value;
    const database = document.getElementById('schemaDatabase').value;
    const authType = document.getElementById('schemaAuthType').value;
    const username = authType === 'sql' ? document.getElementById('schemaUsername').value : '';
    const password = authType === 'sql' ? document.getElementById('schemaPassword').value : '';
    const messageDiv = document.getElementById('schemaMessage');
    const button = event.target;

    if (!server || !database) {
        showSchemaMessage('❌ Server and Database are required', 'error');
        return;
    }

    button.disabled = true;
    button.textContent = 'Analyzing Procedures...';
    messageDiv.innerHTML = '';

    try {
        const response = await fetch('/api/admin/analyze-stored-procedures', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                // Auth removed
            },
            body: JSON.stringify({
                server,
                database,
                authType,
                username,
                password
            })
        });

        const result = await response.json();

        if (result.success) {
            showSchemaMessage(
                `✅ Stored procedure analysis started successfully!<br>` +
                `<strong>Database:</strong> ${database}<br>` +
                `<strong>Process ID:</strong> ${result.pid}<br>` +
                `<em>Check server logs for progress. This may take 2-10 minutes depending on number of procedures.</em>`,
                'success'
            );
        } else {
            showSchemaMessage(`❌ Analysis failed: ${result.error}`, 'error');
        }

    } catch (error) {
        showSchemaMessage(`❌ Error: ${error.message}`, 'error');
    } finally {
        button.disabled = false;
        button.textContent = '⚙️ Analyze Stored Procedures';
    }
}

// Toggle schema comparison UI
function toggleSchemaComparison() {
    const enabled = document.getElementById('enableComparison').checked;
    const compareDbGroup = document.getElementById('compareDbGroup');
    const compareDbSelect = document.getElementById('schemaCompareDatabase');
    const compareBtn = document.getElementById('compareSchemasBtn');
    const dbSelect = document.getElementById('schemaDatabase');

    if (enabled) {
        compareDbGroup.style.display = 'block';
        compareDbSelect.disabled = dbSelect.disabled; // Enable if main db is enabled
        compareBtn.disabled = dbSelect.disabled; // Enable if main db is enabled
    } else {
        compareDbGroup.style.display = 'none';
        compareDbSelect.disabled = true;
        compareBtn.disabled = true;
    }
}

// Compare schemas between two databases
async function compareSchemas() {
    const server = document.getElementById('schemaServer').value;
    const database1 = document.getElementById('schemaDatabase').value;
    const database2 = document.getElementById('schemaCompareDatabase').value;
    const authType = document.getElementById('schemaAuthType').value;
    const username = authType === 'sql' ? document.getElementById('schemaUsername').value : '';
    const password = authType === 'sql' ? document.getElementById('schemaPassword').value : '';
    const messageDiv = document.getElementById('schemaMessage');
    const button = event.target;

    if (!server || !database1 || !database2) {
        showSchemaMessage('❌ Server and both databases are required', 'error');
        return;
    }

    if (database1 === database2) {
        showSchemaMessage('❌ Please select two different databases to compare', 'error');
        return;
    }

    button.disabled = true;
    button.textContent = 'Comparing...';
    messageDiv.innerHTML = '<p style="color: #cbd5e1;">Comparing schemas...</p>';

    try {
        const response = await fetch('/api/admin/compare-schemas', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                // Auth removed
            },
            body: JSON.stringify({
                server,
                database1,
                database2,
                authType,
                username,
                password
            })
        });

        const result = await response.json();

        if (result.success) {
            showComparisonResults(result.comparison, database1, database2);
            showSchemaMessage('✅ Comparison completed successfully', 'success');
        } else {
            showSchemaMessage(`❌ Comparison failed: ${result.error}`, 'error');
        }

    } catch (error) {
        showSchemaMessage(`❌ Error: ${error.message}`, 'error');
    } finally {
        button.disabled = false;
        button.textContent = '🔍 Compare Schemas';
    }
}

// Display comparison results in modal
function showComparisonResults(comparison, db1, db2) {
    const modal = document.getElementById('comparisonModal');
    const content = document.getElementById('comparisonContent');

    let html = `
        <div style="background: #334155; padding: 16px; margin-bottom: 16px;">
            <h3 style="color: #f1f5f9; margin-bottom: 8px;">Comparing:</h3>
            <p style="margin: 0;"><strong>${db1}</strong> vs <strong>${db2}</strong></p>
        </div>
    `;

    // Tables section
    html += `<div style="margin-bottom: 24px;">
        <h3 style="color: #3b82f6; font-size: 18px; margin-bottom: 12px;">📋 Tables</h3>`;

    if (comparison.tables.onlyInDb1.length > 0) {
        html += `<div style="background: #1e293b; padding: 12px; margin-bottom: 12px;">
            <h4 style="color: #10b981; font-size: 14px; margin-bottom: 8px;">Only in ${db1}:</h4>
            <ul style="margin: 0; padding-left: 20px;">
                ${comparison.tables.onlyInDb1.map(t => `<li>${t}</li>`).join('')}
            </ul>
        </div>`;
    }

    if (comparison.tables.onlyInDb2.length > 0) {
        html += `<div style="background: #1e293b; padding: 12px; margin-bottom: 12px;">
            <h4 style="color: #f59e0b; font-size: 14px; margin-bottom: 8px;">Only in ${db2}:</h4>
            <ul style="margin: 0; padding-left: 20px;">
                ${comparison.tables.onlyInDb2.map(t => `<li>${t}</li>`).join('')}
            </ul>
        </div>`;
    }

    if (comparison.tables.onlyInDb1.length === 0 && comparison.tables.onlyInDb2.length === 0) {
        html += `<p style="color: #cbd5e1;">✅ All tables match</p>`;
    }

    html += `</div>`;

    // Stored Procedures section
    html += `<div style="margin-bottom: 24px;">
        <h3 style="color: #3b82f6; font-size: 18px; margin-bottom: 12px;">⚙️ Stored Procedures</h3>`;

    if (comparison.procedures.onlyInDb1.length > 0) {
        html += `<div style="background: #1e293b; padding: 12px; margin-bottom: 12px;">
            <h4 style="color: #10b981; font-size: 14px; margin-bottom: 8px;">Only in ${db1}:</h4>
            <ul style="margin: 0; padding-left: 20px;">
                ${comparison.procedures.onlyInDb1.map(p => `<li>${p}</li>`).join('')}
            </ul>
        </div>`;
    }

    if (comparison.procedures.onlyInDb2.length > 0) {
        html += `<div style="background: #1e293b; padding: 12px; margin-bottom: 12px;">
            <h4 style="color: #f59e0b; font-size: 14px; margin-bottom: 8px;">Only in ${db2}:</h4>
            <ul style="margin: 0; padding-left: 20px;">
                ${comparison.procedures.onlyInDb2.map(p => `<li>${p}</li>`).join('')}
            </ul>
        </div>`;
    }

    if (comparison.procedures.onlyInDb1.length === 0 && comparison.procedures.onlyInDb2.length === 0) {
        html += `<p style="color: #cbd5e1;">✅ All stored procedures match</p>`;
    }

    html += `</div>`;

    // Functions section
    html += `<div style="margin-bottom: 24px;">
        <h3 style="color: #3b82f6; font-size: 18px; margin-bottom: 12px;">🔧 Functions</h3>`;

    if (comparison.functions.onlyInDb1.length > 0) {
        html += `<div style="background: #1e293b; padding: 12px; margin-bottom: 12px;">
            <h4 style="color: #10b981; font-size: 14px; margin-bottom: 8px;">Only in ${db1}:</h4>
            <ul style="margin: 0; padding-left: 20px;">
                ${comparison.functions.onlyInDb1.map(f => `<li>${f}</li>`).join('')}
            </ul>
        </div>`;
    }

    if (comparison.functions.onlyInDb2.length > 0) {
        html += `<div style="background: #1e293b; padding: 12px; margin-bottom: 12px;">
            <h4 style="color: #f59e0b; font-size: 14px; margin-bottom: 8px;">Only in ${db2}:</h4>
            <ul style="margin: 0; padding-left: 20px;">
                ${comparison.functions.onlyInDb2.map(f => `<li>${f}</li>`).join('')}
            </ul>
        </div>`;
    }

    if (comparison.functions.onlyInDb1.length === 0 && comparison.functions.onlyInDb2.length === 0) {
        html += `<p style="color: #cbd5e1;">✅ All functions match</p>`;
    }

    html += `</div>`;

    content.innerHTML = html;
    modal.classList.remove('hidden');
}

// Close comparison modal
function closeComparisonModal(event) {
    if (!event || event.target.id === 'comparisonModal') {
        document.getElementById('comparisonModal').classList.add('hidden');
    }
}

// ========== User Management Functions ==========

// Load users
async function loadUsers() {
    const tbody = document.getElementById('usersTableBody');
    tbody.innerHTML = '<tr><td colspan="5" style="padding: 20px; text-align: center; color: #6b7280;">Loading users...</td></tr>';

    try {
        // Verify auth is still valid
        if (!auth.isAuthenticated()) {
            throw new Error('Not authenticated');
        }

        const response = await fetch('/api/auth/users', {
            headers: {
                'Authorization': `Bearer ${auth.getAccessToken()}`
            }
        });

        if (!response.ok) {
            const errorData = await response.json().catch(() => ({ error: 'Failed to load users' }));
            throw new Error(errorData.error || `HTTP ${response.status}: ${response.statusText}`);
        }

        const users = await response.json();
        renderUsers(users);
    } catch (error) {
        console.error('Load users error:', error);
        tbody.innerHTML = `<tr><td colspan="5" style="padding: 20px; text-align: center; color: #ef4444;">Failed to load users: ${error.message}</td></tr>`;

        // If auth error, redirect to login
        if (error.message.includes('Not authenticated') || error.message.includes('401') || error.message.includes('403')) {
            setTimeout(() => {
                window.location.href = '/login.html?redirect=' + encodeURIComponent(window.location.pathname);
            }, 2000);
        }
    }
}

// Render users table
function renderUsers(users) {
    const tbody = document.getElementById('usersTableBody');

    if (users.length === 0) {
        tbody.innerHTML = '<tr><td colspan="5" style="padding: 20px; text-align: center; color: #6b7280;">No users found</td></tr>';
        return;
    }

    tbody.innerHTML = users.map(user => `
        <tr style="border-bottom: 1px solid #334155;">
            <td style="padding: 12px; color: #f1f5f9;">${user.username}</td>
            <td style="padding: 12px; color: #cbd5e1;">${user.email || 'N/A'}</td>
            <td style="padding: 12px;">
                <span style="padding: 4px 8px; background: ${getRoleBadgeColor(user.role)}; color: white; border-radius: 0; font-size: 12px;">
                    ${user.role.toUpperCase()}
                </span>
            </td>
            <td style="padding: 12px;">
                <span style="padding: 4px 8px; background: ${user.isActive ? '#10b981' : '#ef4444'}; color: white; border-radius: 0; font-size: 12px;">
                    ${user.isActive ? 'ACTIVE' : 'INACTIVE'}
                </span>
            </td>
            <td style="padding: 12px;">
                <button class="btn btn-small btn-view" onclick="editUser('${user.id}')" style="margin-right: 8px;">Edit</button>
                <button class="btn btn-small btn-delete" onclick="deleteUser('${user.id}', '${user.username}')">Delete</button>
            </td>
        </tr>
    `).join('');
}

// Get role badge color
function getRoleBadgeColor(role) {
    switch(role) {
        case 'admin': return '#dc2626';
        case 'developer': return '#3b82f6';
        case 'user': return '#6b7280';
        default: return '#6b7280';
    }
}

// Open add user modal
function openAddUserModal() {
    document.getElementById('addUserModal').classList.remove('hidden');
    document.getElementById('addUserForm').reset();
    document.getElementById('addUserError').classList.add('hidden');
}

// Close add user modal
function closeAddUserModal(event) {
    if (!event || event.target.id === 'addUserModal' || event.type === 'click') {
        document.getElementById('addUserModal').classList.add('hidden');
    }
}

// Open edit user modal
function openEditUserModal() {
    document.getElementById('editUserModal').classList.remove('hidden');
}

// Close edit user modal
function closeEditUserModal(event) {
    if (!event || event.target.id === 'editUserModal' || event.type === 'click') {
        document.getElementById('editUserModal').classList.add('hidden');
    }
}

// Edit user
async function editUser(userId) {
    try {
        if (!auth.isAuthenticated()) {
            throw new Error('Not authenticated');
        }

        const response = await fetch(`/api/auth/users/${userId}`, {
            headers: {
                'Authorization': `Bearer ${auth.getAccessToken()}`
            }
        });

        if (!response.ok) {
            const errorData = await response.json().catch(() => ({ error: 'Failed to load user' }));
            throw new Error(errorData.error || `HTTP ${response.status}`);
        }

        const user = await response.json();

        document.getElementById('editUserId').value = user.id;
        document.getElementById('editUsername').value = user.username;
        document.getElementById('editEmail').value = user.email || '';
        document.getElementById('editRole').value = user.role;

        openEditUserModal();
    } catch (error) {
        console.error('Edit user error:', error);
        showUserMessage(`Failed to load user: ${error.message}`, 'error');

        if (error.message.includes('Not authenticated') || error.message.includes('401') || error.message.includes('403')) {
            setTimeout(() => {
                window.location.href = '/login.html?redirect=' + encodeURIComponent(window.location.pathname);
            }, 2000);
        }
    }
}

// Delete user
async function deleteUser(userId, username) {
    if (!confirm(`Are you sure you want to delete user "${username}"?`)) {
        return;
    }

    try {
        if (!auth.isAuthenticated()) {
            throw new Error('Not authenticated');
        }

        const response = await fetch(`/api/auth/users/${userId}`, {
            method: 'DELETE',
            headers: {
                'Authorization': `Bearer ${auth.getAccessToken()}`
            }
        });

        if (!response.ok) {
            const error = await response.json().catch(() => ({ error: 'Failed to delete user' }));
            throw new Error(error.error || `HTTP ${response.status}`);
        }

        showUserMessage('User deleted successfully', 'success');
        loadUsers();
    } catch (error) {
        console.error('Delete user error:', error);
        showUserMessage(`Failed to delete user: ${error.message}`, 'error');

        if (error.message.includes('Not authenticated') || error.message.includes('401') || error.message.includes('403')) {
            setTimeout(() => {
                window.location.href = '/login.html?redirect=' + encodeURIComponent(window.location.pathname);
            }, 2000);
        }
    }
}

// Show user message
function showUserMessage(message, type = 'success') {
    const messageDiv = document.getElementById('userMessage');
    messageDiv.className = `message ${type}`;
    messageDiv.textContent = message;
    messageDiv.style.display = 'block';

    setTimeout(() => {
        messageDiv.style.display = 'none';
    }, 5000);
}

// Handle add user form submission
document.getElementById('addUserForm').addEventListener('submit', async (e) => {
    e.preventDefault();

    const username = document.getElementById('newUsername').value;
    const password = document.getElementById('newPassword').value;
    const email = document.getElementById('newEmail').value;
    const role = document.getElementById('newRole').value;

    try {
        // Verify auth before making request
        if (!auth.isAuthenticated()) {
            throw new Error('Not authenticated. Please log in again.');
        }

        const response = await fetch('/api/auth/users', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${auth.getAccessToken()}`
            },
            body: JSON.stringify({
                username,
                password,
                email: email || undefined,
                role
            })
        });

        if (!response.ok) {
            const error = await response.json().catch(() => ({ error: 'Failed to create user' }));
            throw new Error(error.error || `HTTP ${response.status}: ${response.statusText}`);
        }

        closeAddUserModal();
        showUserMessage('User created successfully', 'success');
        loadUsers();
    } catch (error) {
        console.error('Add user error:', error);
        const errorDiv = document.getElementById('addUserError');
        errorDiv.textContent = error.message;
        errorDiv.classList.remove('hidden');

        // If auth error, show specific message
        if (error.message.includes('Invalid token') || error.message.includes('401') || error.message.includes('403')) {
            errorDiv.textContent = 'Authentication error. Please log in again.';
            setTimeout(() => {
                window.location.href = '/login.html?redirect=' + encodeURIComponent(window.location.pathname);
            }, 2000);
        }
    }
});

// Handle edit user form submission
document.getElementById('editUserForm').addEventListener('submit', async (e) => {
    e.preventDefault();

    const userId = document.getElementById('editUserId').value;
    const password = document.getElementById('editPassword').value;
    const email = document.getElementById('editEmail').value;
    const role = document.getElementById('editRole').value;

    const updateData = {
        email: email || undefined,
        role
    };

    if (password) {
        updateData.password = password;
    }

    try {
        if (!auth.isAuthenticated()) {
            throw new Error('Not authenticated. Please log in again.');
        }

        const response = await fetch(`/api/auth/users/${userId}`, {
            method: 'PUT',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${auth.getAccessToken()}`
            },
            body: JSON.stringify(updateData)
        });

        if (!response.ok) {
            const error = await response.json().catch(() => ({ error: 'Failed to update user' }));
            throw new Error(error.error || `HTTP ${response.status}: ${response.statusText}`);
        }

        closeEditUserModal();
        showUserMessage('User updated successfully', 'success');
        loadUsers();
    } catch (error) {
        console.error('Edit user error:', error);
        const errorDiv = document.getElementById('editUserError');
        errorDiv.textContent = error.message;
        errorDiv.classList.remove('hidden');

        if (error.message.includes('Invalid token') || error.message.includes('401') || error.message.includes('403')) {
            errorDiv.textContent = 'Authentication error. Please log in again.';
            setTimeout(() => {
                window.location.href = '/login.html?redirect=' + encodeURIComponent(window.location.pathname);
            }, 2000);
        }
    }
});

// ========== LLM Control Functions ==========

// Get currently running LLM model
async function getCurrentModel() {
    try {
        const response = await fetch('/api/admin/llm/models');
        const data = await response.json();

        const currentModelEl = document.getElementById('currentModel');
        const stopButton = document.getElementById('stopModelBtn');

        if (data.success && data.models && data.models.length > 0) {
            const runningModel = data.models[0];
            currentModelEl.textContent = runningModel.name || 'Unknown';
            currentModelEl.style.color = 'var(--accent-cyan)';
            stopButton.disabled = false;
        } else {
            currentModelEl.textContent = 'No model running';
            currentModelEl.style.color = 'var(--text-secondary)';
            stopButton.disabled = true;
        }
    } catch (error) {
        console.error('Failed to get current model:', error);
        document.getElementById('currentModel').textContent = 'Error';
        document.getElementById('currentModel').style.color = 'var(--accent-red)';
    }
}

// Stop the active LLM model
async function stopModel() {
    try {
        const currentModelEl = document.getElementById('currentModel');
        const modelName = currentModelEl.textContent;

        if (modelName === 'No model running' || modelName === 'Error') {
            return;
        }

        const button = document.getElementById('stopModelBtn');
        button.disabled = true;
        button.textContent = 'Stopping...';

        const response = await fetch('/api/admin/llm/unload', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ model: modelName })
        });

        const result = await response.json();

        if (result.success) {
            showSettingsMessage('Model stopped successfully', 'success');
            await getCurrentModel();
        } else {
            showSettingsMessage('Failed to stop model: ' + result.error, 'error');
        }
    } catch (error) {
        showSettingsMessage('Error stopping model: ' + error.message, 'error');
    } finally {
        document.getElementById('stopModelBtn').textContent = 'Stop Active Model';
    }
}

// Show available models (populate combo box)
async function showModels() {
    try {
        const button = document.getElementById('showModelsBtn');
        button.disabled = true;
        button.textContent = 'Loading...';

        const response = await fetch('/api/admin/llm/models');
        const data = await response.json();

        const select = document.getElementById('llmModel');
        select.innerHTML = '<option value="">Select a model...</option>';

        if (data.success && data.models && data.models.length > 0) {
            data.models.forEach(model => {
                const option = document.createElement('option');
                option.value = model.name;
                option.textContent = model.name;
                select.appendChild(option);
            });

            select.disabled = false;
            showSettingsMessage(`Loaded ${data.models.length} models`, 'success');
        } else {
            select.innerHTML = '<option value="">No models found</option>';
            showSettingsMessage('No models available', 'error');
        }
    } catch (error) {
        console.error('Failed to load models:', error);
        const select = document.getElementById('llmModel');
        select.innerHTML = '<option value="">Error loading models</option>';
        showSettingsMessage('Failed to load models: ' + error.message, 'error');
    } finally {
        document.getElementById('showModelsBtn').textContent = 'Show Models';
        document.getElementById('showModelsBtn').disabled = false;
    }
}

// Start selected LLM model
async function startModel() {
    try {
        const select = document.getElementById('llmModel');
        const modelName = select.value;

        if (!modelName) {
            showSettingsMessage('Please select a model first', 'error');
            return;
        }

        const button = document.getElementById('startModelBtn');
        button.disabled = true;
        button.textContent = 'Starting...';

        const response = await fetch('/api/admin/llm/load', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ model: modelName })
        });

        const result = await response.json();

        if (result.success) {
            showSettingsMessage(`Model ${modelName} started successfully`, 'success');
            await getCurrentModel();

            // Update saved model selection
            const savedSettings = localStorage.getItem('ragSystemSettings');
            let settings = {};
            if (savedSettings) {
                try {
                    settings = JSON.parse(savedSettings);
                } catch (e) {}
            }
            settings.model = modelName;
            localStorage.setItem('ragSystemSettings', JSON.stringify(settings));
        } else {
            showSettingsMessage('Failed to start model: ' + result.error, 'error');
        }
    } catch (error) {
        showSettingsMessage('Error starting model: ' + error.message, 'error');
    } finally {
        document.getElementById('startModelBtn').disabled = false;
        document.getElementById('startModelBtn').textContent = 'Start LLM';
    }
}

// Enable Start LLM button when model is selected
function onModelSelected() {
    const select = document.getElementById('llmModel');
    const startButton = document.getElementById('startModelBtn');
    startButton.disabled = !select.value || select.value === '';
}

// Load users on page load
window.addEventListener('DOMContentLoaded', () => {
    // Small delay to ensure auth is initialized
    setTimeout(() => {
        loadUsers();
        getCurrentModel(); // Load current running model
    }, 100);
});
