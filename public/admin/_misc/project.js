// Project Viewer - Admin Page

// Initialize auth client
const auth = new AuthClient();

let currentProject = '';
let allDocuments = [];
let filteredDocuments = [];
let selectedDocument = null;
let originalDocument = null;

// Initialize
window.addEventListener('DOMContentLoaded', async () => {
    // Check authentication
    if (!auth.isAuthenticated()) {
        window.location.href = '/login.html?redirect=' + encodeURIComponent(window.location.pathname);
        return;
    }

    // Initialize admin sidebar
    if (typeof initSidebar === 'function') {
        await initSidebar();
    }

    // Get project from URL parameter
    const urlParams = new URLSearchParams(window.location.search);
    currentProject = urlParams.get('project');

    if (!currentProject) {
        // No project specified - show project selector
        await showProjectSelector();
    } else {
        // Project specified - show document viewer
        await showDocumentViewer();
    }
});

/**
 * Show project selector when no project is specified
 */
async function showProjectSelector() {
    document.getElementById('pageTitle').textContent = 'Project Management';
    document.getElementById('pageSubtitle').textContent = 'Select a project to view and manage its documents';
    document.getElementById('projectSelector').classList.remove('hidden');
    document.getElementById('documentViewer').classList.add('hidden');

    const projectGrid = document.getElementById('projectGrid');
    projectGrid.innerHTML = '<div class="loading">Loading projects...</div>';

    try {
        const response = await fetch('/api/admin/projects', {
            headers: {
                'Authorization': `Bearer ${auth.getAccessToken()}`
            }
        });

        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }

        const data = await response.json();
        const projects = data.projects || [];

        if (projects.length === 0) {
            projectGrid.innerHTML = `
                <div class="empty-state" style="grid-column: 1 / -1;">
                    <div class="empty-state-icon">📂</div>
                    <p>No projects found in the vector database</p>
                    <p style="font-size: 13px; margin-top: 8px; opacity: 0.7;">Upload documents to create a project</p>
                </div>
            `;
            return;
        }

        projectGrid.innerHTML = projects.map(project => `
            <div class="project-card" onclick="selectProject('${project.name}')">
                <div class="project-card-header">
                    <div class="project-card-icon">📁</div>
                    <div>
                        <div class="project-card-name">${project.name}</div>
                        <div class="project-card-meta">${project.tableCount || 1} table${(project.tableCount || 1) > 1 ? 's' : ''}</div>
                    </div>
                </div>
                <div class="project-card-stats">
                    <div class="project-stat">
                        <div class="project-stat-value">${project.documentCount || 0}</div>
                        <div class="project-stat-label">Documents</div>
                    </div>
                </div>
            </div>
        `).join('');

    } catch (error) {
        console.error('Failed to load projects:', error);
        projectGrid.innerHTML = `
            <div class="empty-state" style="grid-column: 1 / -1;">
                <div class="empty-state-icon">❌</div>
                <p>Failed to load projects</p>
                <p style="font-size: 13px; margin-top: 8px; opacity: 0.7;">${error.message}</p>
            </div>
        `;
    }
}

/**
 * Show document viewer for selected project
 */
async function showDocumentViewer() {
    document.getElementById('pageTitle').textContent = currentProject;
    document.getElementById('pageSubtitle').textContent = 'View and manage project documents';
    document.getElementById('projectSelector').classList.add('hidden');
    document.getElementById('documentViewer').classList.remove('hidden');

    await loadDocuments();
}

/**
 * Select a project and load its documents
 */
function selectProject(projectName) {
    // Update URL with project parameter (without page reload)
    const url = new URL(window.location);
    url.searchParams.set('project', projectName);
    window.history.pushState({}, '', url);

    currentProject = projectName;
    showDocumentViewer();
}

/**
 * Go back to project selector
 */
function backToProjects() {
    // Remove project from URL
    const url = new URL(window.location);
    url.searchParams.delete('project');
    window.history.pushState({}, '', url);

    currentProject = '';
    selectedDocument = null;
    originalDocument = null;
    allDocuments = [];
    filteredDocuments = [];

    // Reset document viewer state
    document.getElementById('emptyDocState').classList.remove('hidden');
    document.getElementById('documentForm').classList.add('hidden');
    document.getElementById('docContentHeader').classList.add('hidden');

    showProjectSelector();
}

async function loadDocuments() {
    const docList = document.getElementById('docList');
    docList.innerHTML = '<div class="loading">Loading documents...</div>';

    try {
        const response = await fetch(`/api/admin/documents?project=${encodeURIComponent(currentProject)}&limit=1000`, {
            headers: {
                'Authorization': `Bearer ${auth.getAccessToken()}`
            }
        });

        if (!response.ok) {
            throw new Error('Failed to load documents');
        }

        const data = await response.json();
        allDocuments = data.documents || [];
        filteredDocuments = [...allDocuments];

        renderDocumentList();

    } catch (error) {
        console.error('Failed to load documents:', error);
        docList.innerHTML = `
            <div class="empty-state">
                <div class="empty-state-icon">❌</div>
                <p>Failed to load documents</p>
            </div>
        `;
    }
}

function renderDocumentList() {
    const docList = document.getElementById('docList');
    const docCount = document.getElementById('docCount');

    docCount.textContent = filteredDocuments.length;

    if (filteredDocuments.length === 0) {
        docList.innerHTML = `
            <div class="empty-state">
                <div class="empty-state-icon">📄</div>
                <p>No documents found</p>
            </div>
        `;
        return;
    }

    docList.innerHTML = filteredDocuments.map((doc, index) => {
        const fileName = doc.file || doc.metadata?.file || 'Untitled Document';
        const category = doc.category || doc.metadata?.category || 'unknown';
        const docId = doc.id || doc.metadata?.document_id || index;
        const isActive = selectedDocument && (selectedDocument.id === docId || selectedDocument.metadata?.document_id === docId) ? 'active' : '';

        return `
            <div class="doc-item ${isActive}" onclick="selectDocument(${index})">
                <div class="doc-item-title">${fileName}</div>
                <div class="doc-item-meta">${category} · ${String(docId).substring(0, 8)}...</div>
            </div>
        `;
    }).join('');
}

function selectDocument(index) {
    const doc = filteredDocuments[index];

    // Store original for comparison
    originalDocument = JSON.parse(JSON.stringify(doc));
    selectedDocument = doc;

    // Update UI
    document.getElementById('emptyDocState').classList.add('hidden');
    document.getElementById('docContentHeader').classList.remove('hidden');
    document.getElementById('documentForm').classList.remove('hidden');

    // Populate form
    document.getElementById('docTitle').textContent = doc.file || doc.metadata?.file || 'Untitled Document';
    document.getElementById('docId').value = doc.id || doc.metadata?.document_id || '';
    document.getElementById('docCategory').value = doc.category || doc.metadata?.category || 'other';
    document.getElementById('docFile').value = doc.file || doc.metadata?.file || '';
    document.getElementById('docProject').value = doc.project || doc.metadata?.project || currentProject;
    document.getElementById('docContent').value = doc.content || doc.metadata?.content || '';

    // Highlight in list
    renderDocumentList();
}

function filterDocuments() {
    const searchTerm = document.getElementById('searchBox').value.toLowerCase();

    if (!searchTerm) {
        filteredDocuments = [...allDocuments];
    } else {
        filteredDocuments = allDocuments.filter(doc => {
            const fileName = (doc.file || doc.metadata?.file || '').toLowerCase();
            const content = (doc.content || doc.metadata?.content || '').toLowerCase();
            const category = (doc.category || doc.metadata?.category || '').toLowerCase();

            return fileName.includes(searchTerm) ||
                   content.includes(searchTerm) ||
                   category.includes(searchTerm);
        });
    }

    renderDocumentList();
}

async function saveDocument() {
    if (!selectedDocument) {
        alert('No document selected');
        return;
    }

    const updatedDoc = {
        id: document.getElementById('docId').value,
        category: document.getElementById('docCategory').value,
        content: document.getElementById('docContent').value
    };

    // Check if anything changed
    const hasChanges = updatedDoc.category !== (originalDocument.category || originalDocument.metadata?.category) ||
                       updatedDoc.content !== (originalDocument.content || originalDocument.metadata?.content);

    if (!hasChanges) {
        alert('No changes to save');
        return;
    }

    try {
        const docId = originalDocument.id || originalDocument.metadata?.document_id;
        const response = await fetch(`/api/admin/documents/${docId}`, {
            method: 'PUT',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${auth.getAccessToken()}`
            },
            body: JSON.stringify(updatedDoc)
        });

        const result = await response.json();

        if (result.success) {
            alert('Document saved successfully!');

            // Update local copy
            selectedDocument.category = updatedDoc.category;
            selectedDocument.content = updatedDoc.content;
            if (selectedDocument.metadata) {
                selectedDocument.metadata.category = updatedDoc.category;
                selectedDocument.metadata.content = updatedDoc.content;
            }
            originalDocument = JSON.parse(JSON.stringify(selectedDocument));

            // Reload documents to reflect changes
            await loadDocuments();

        } else {
            alert('Failed to save: ' + (result.message || result.error));
        }

    } catch (error) {
        console.error('Save error:', error);
        alert('Failed to save document: ' + error.message);
    }
}

async function deleteDocument() {
    if (!selectedDocument) {
        alert('No document selected');
        return;
    }

    const fileName = selectedDocument.file || selectedDocument.metadata?.file || 'this document';

    if (!confirm(`Are you sure you want to delete "${fileName}"? This action cannot be undone.`)) {
        return;
    }

    try {
        const docId = selectedDocument.id || selectedDocument.metadata?.document_id;
        const response = await fetch(`/api/admin/documents/${docId}`, {
            method: 'DELETE',
            headers: {
                'Authorization': `Bearer ${auth.getAccessToken()}`
            }
        });

        const result = await response.json();

        if (result.success) {
            // Clear selection
            selectedDocument = null;
            originalDocument = null;

            document.getElementById('emptyDocState').classList.remove('hidden');
            document.getElementById('docContentHeader').classList.add('hidden');
            document.getElementById('documentForm').classList.add('hidden');

            // Reload documents
            await loadDocuments();

            alert('Document deleted successfully');

        } else {
            alert('Failed to delete: ' + (result.message || result.error));
        }

    } catch (error) {
        console.error('Delete error:', error);
        alert('Failed to delete document: ' + error.message);
    }
}
