// Code Changes Tracker - Client-side JavaScript

// Global state
let currentRepo = null;
let currentRepoPath = null;
let lastAnalysisResult = null;

// Enable pull button when repository is selected
document.getElementById('repoSelect').addEventListener('change', function() {
    const pullBtn = document.getElementById('pullBtn');
    pullBtn.disabled = !this.value;
    currentRepo = this.value;
});

// Main function to pull changes and analyze
async function pullChanges() {
    const repoSelect = document.getElementById('repoSelect');
    const repo = repoSelect.value;

    if (!repo) {
        showMessage('Please select a repository', 'error');
        return;
    }

    const pullBtn = document.getElementById('pullBtn');
    const loadingMessage = document.getElementById('loadingMessage');
    const messageContainer = document.getElementById('messageContainer');
    const resultsContainer = document.getElementById('resultsContainer');

    // Reset UI
    pullBtn.disabled = true;
    loadingMessage.classList.add('active');
    messageContainer.innerHTML = '';
    resultsContainer.classList.add('hidden');

    try {
        console.log(`Pulling changes for repo: ${repo}`);

        const response = await fetch('/api/git/pull-and-analyze', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ repo })
        });

        const result = await response.json();

        if (result.success) {
            lastAnalysisResult = result;
            if (result.filesChanged.length === 0) {
                showMessage('No changes detected. Repository is up to date.', 'info');
            } else {
                showMessage(`Successfully pulled changes. ${result.filesChanged.length} file(s) modified.`, 'success');
                displayResults(result);
                // Check call graph status after displaying results
                await checkCallGraphStatus(repo);
            }
        } else {
            showMessage(`Error: ${result.message || 'Failed to pull changes'}`, 'error');
        }

    } catch (error) {
        console.error('Pull error:', error);
        showMessage(`Error: ${error.message}`, 'error');
    } finally {
        loadingMessage.classList.remove('active');
        pullBtn.disabled = false;
    }
}

// Display results with caller impact analysis
function displayResults(data) {
    const resultsContainer = document.getElementById('resultsContainer');
    const resultsContent = document.getElementById('resultsContent');

    // Build HTML for results
    let html = '';

    // Repo info section
    html += `
        <div class="repo-info">
            <h3>${escapeHtml(data.repo)}</h3>
            <div class="meta">
                ${data.filesChanged.length} file(s) changed
            </div>
            ${data.gitOutput ? `
                <div class="git-output">${escapeHtml(data.gitOutput)}</div>
            ` : ''}
        </div>
    `;

    // Files list
    if (data.filesChanged.length > 0) {
        html += '<div class="files-list">';

        data.filesChanged.forEach(file => {
            const fileExt = getFileExtension(file.file);
            const fileIcon = getFileIcon(fileExt);
            const status = file.status || 'modified';

            html += `
                <div class="file-item">
                    <div class="file-header">
                        <div class="file-icon">${fileIcon}</div>
                        <div class="file-path">${escapeHtml(file.file)}</div>
                        <div class="file-status ${status}">${status}</div>
                    </div>
            `;

            // Show classes if available
            if (file.classes && file.classes.length > 0) {
                html += `
                    <div class="code-elements">
                        <div class="element-section">
                            <h4>Classes (${file.classes.length})</h4>
                            <div class="element-list">
                                ${file.classes.map(cls => `<span class="element-tag class">${escapeHtml(cls)}</span>`).join('')}
                            </div>
                        </div>
                    </div>
                `;
            }

            // Show methods with callers (new impact analysis display)
            if (file.methodsWithClass && file.methodsWithClass.length > 0) {
                html += `<div class="code-elements">`;

                file.methodsWithClass.forEach(methodObj => {
                    const methodKey = `${methodObj.className}.${methodObj.name}`;
                    const callerInfo = file.callersInfo && file.callersInfo[methodKey];

                    html += `
                        <div class="method-impact-box">
                            <div class="method-header">
                                <span class="method-name">${escapeHtml(methodKey)}</span>
                            </div>
                    `;

                    // Show callers if available
                    if (callerInfo && callerInfo.callers && callerInfo.callers.length > 0) {
                        html += `
                            <div class="callers-section">
                                <h5>Called by (${callerInfo.totalCallers} caller${callerInfo.totalCallers !== 1 ? 's' : ''}):</h5>
                                <div class="callers-list">
                        `;

                        callerInfo.callers.forEach(caller => {
                            const externalBadge = caller.isExternalProject ?
                                `<span class="external-badge">${escapeHtml(caller.project)}</span>` : '';

                            html += `
                                <div class="caller-item">
                                    <div class="caller-method">
                                        ${externalBadge}
                                        <span class="caller-class-method">${escapeHtml(caller.callerClass)}.${escapeHtml(caller.callerMethod)}</span>
                                    </div>
                                    <div class="caller-location">
                                        ${caller.filePath ? `<span class="caller-file">${escapeHtml(caller.filePath)}</span>` : ''}
                                        ${caller.callLine ? `<span class="caller-line">Line ${caller.callLine}</span>` : ''}
                                    </div>
                                </div>
                            `;
                        });

                        html += `</div>`; // Close callers-list

                        // Show truncation message if there are more callers
                        if (callerInfo.truncated) {
                            const remaining = callerInfo.totalCallers - callerInfo.callers.length;
                            html += `
                                <div class="truncation-message">
                                    ... and ${remaining} other file${remaining !== 1 ? 's' : ''} affected
                                </div>
                            `;
                        }

                        html += `</div>`; // Close callers-section
                    } else {
                        html += `
                            <div class="no-callers">
                                <span>No known callers in call graph</span>
                            </div>
                        `;
                    }

                    html += `</div>`; // Close method-impact-box
                });

                html += `</div>`; // Close code-elements
            } else if (file.methods && file.methods.length > 0) {
                // Fallback: Show simple methods list if no methodsWithClass data
                html += `
                    <div class="code-elements">
                        <div class="element-section">
                            <h4>Methods/Functions (${file.methods.length})</h4>
                            <div class="element-list">
                                ${file.methods.map(method => {
                                    const tag = fileExt === 'cs' ? 'method' : 'function';
                                    return `<span class="element-tag ${tag}">${escapeHtml(method)}</span>`;
                                }).join('')}
                            </div>
                        </div>
                    </div>
                `;
            }

            // If no classes or methods found
            if ((!file.classes || file.classes.length === 0) &&
                (!file.methods || file.methods.length === 0) &&
                (!file.methodsWithClass || file.methodsWithClass.length === 0)) {
                html += `
                    <div class="code-elements">
                        <p style="color: #6b7280; font-size: 14px;">No classes or methods detected (file may be configuration, data, or not analyzable)</p>
                    </div>
                `;
            }

            html += '</div>'; // Close file-item
        });

        html += '</div>'; // Close files-list
    }

    resultsContent.innerHTML = html;
    resultsContainer.classList.remove('hidden');
}

// Helper functions
function showMessage(message, type) {
    const messageContainer = document.getElementById('messageContainer');
    messageContainer.innerHTML = `<div class="message ${type}">${message}</div>`;

    // Auto-hide success messages after 5 seconds
    if (type === 'success') {
        setTimeout(() => {
            messageContainer.innerHTML = '';
        }, 5000);
    }
}

function getFileExtension(filename) {
    const parts = filename.split('.');
    return parts.length > 1 ? parts[parts.length - 1].toLowerCase() : '';
}

function getFileIcon(ext) {
    const icons = {
        'cs': 'C#',
        'js': 'JS',
        'ts': 'TS',
        'sql': 'SQL',
        'md': 'MD',
        'json': 'JSON',
        'xml': 'XML',
        'html': 'HTML',
        'css': 'CSS',
        'txt': 'TXT'
    };
    return icons[ext] || 'FILE';
}

function escapeHtml(text) {
    if (!text) return '';
    const map = {
        '&': '&amp;',
        '<': '&lt;',
        '>': '&gt;',
        '"': '&quot;',
        "'": '&#039;'
    };
    return String(text).replace(/[&<>"']/g, m => map[m]);
}

// ============================================================================
// Call Graph Status and Analysis Functions
// ============================================================================

// Check call graph status for a project
async function checkCallGraphStatus(project) {
    const statusContainer = document.getElementById('callgraphStatus');
    if (!statusContainer) return;

    try {
        const response = await fetch(`/api/roslyn/callgraph-status/${encodeURIComponent(project)}`);
        if (!response.ok) {
            statusContainer.classList.add('hidden');
            return;
        }

        const status = await response.json();
        displayCallGraphStatus(status, project);

    } catch (error) {
        console.error('Failed to check call graph status:', error);
        statusContainer.classList.add('hidden');
    }
}

// Display call graph status with action buttons
function displayCallGraphStatus(status, project) {
    const statusContainer = document.getElementById('callgraphStatus');

    if (status.needs_analysis || status.callgraph_entries === 0) {
        // No call graph data - show warning with build button
        statusContainer.className = 'callgraph-status warning';
        statusContainer.innerHTML = `
            <div class="callgraph-header">
                <div class="callgraph-title">
                    <span class="callgraph-icon">&#9888;</span>
                    <span>No Call Graph Data Available</span>
                </div>
            </div>
            <div class="callgraph-message">
                The call graph for <strong>${escapeHtml(project)}</strong> has not been built yet.
                Caller information cannot be displayed without analyzing the codebase.
            </div>
            <div style="display: flex; gap: 12px; flex-wrap: wrap;">
                <button class="btn-secondary btn-warning" onclick="runTargetedAnalysis('${escapeHtml(project)}')">
                    Build Call Graph (Targeted)
                </button>
                <button class="btn-secondary" onclick="runFullAnalysis('${escapeHtml(project)}')">
                    Full Repository Analysis
                </button>
            </div>
            <div id="analysisProgress" class="analysis-progress hidden"></div>
        `;
    } else {
        // Call graph exists - show info
        const lastUpdated = status.last_updated ? new Date(status.last_updated).toLocaleString() : 'Unknown';
        statusContainer.className = 'callgraph-status success';
        statusContainer.innerHTML = `
            <div class="callgraph-header">
                <div class="callgraph-title">
                    <span class="callgraph-icon">&#10003;</span>
                    <span>Call Graph Available</span>
                </div>
                <div class="callgraph-stats">
                    ${status.callgraph_entries.toLocaleString()} relationships |
                    ${status.methods_indexed.toLocaleString()} methods indexed
                </div>
            </div>
            <div class="callgraph-message">
                Last updated: ${lastUpdated}
            </div>
            <div style="display: flex; gap: 12px; flex-wrap: wrap;">
                <button class="btn-secondary" onclick="runTargetedAnalysis('${escapeHtml(project)}')">
                    Update Call Graph (Changed Files Only)
                </button>
                <button class="btn-secondary" onclick="runFullAnalysis('${escapeHtml(project)}')">
                    Rebuild Full Call Graph
                </button>
            </div>
            <div id="analysisProgress" class="analysis-progress hidden"></div>
        `;
    }

    statusContainer.classList.remove('hidden');
}

// Run targeted analysis on files containing references to modified methods
async function runTargetedAnalysis(project) {
    const progressDiv = document.getElementById('analysisProgress');
    if (!progressDiv) return;

    // Get the methods from the last analysis result
    if (!lastAnalysisResult || !lastAnalysisResult.filesChanged) {
        showMessage('No changes to analyze. Pull changes first.', 'error');
        return;
    }

    // Collect all methods from changed files
    const methods = [];
    lastAnalysisResult.filesChanged.forEach(file => {
        if (file.methodsWithClass) {
            file.methodsWithClass.forEach(m => {
                methods.push(`${m.className}.${m.name}`);
            });
        } else if (file.methods) {
            file.methods.forEach(m => methods.push(m));
        }
    });

    if (methods.length === 0) {
        showMessage('No methods found in changed files to analyze.', 'info');
        return;
    }

    // Get repository path from API
    let repoPath = null;
    try {
        const reposResponse = await fetch('/api/git/repositories');
        if (reposResponse.ok) {
            const repos = await reposResponse.json();
            const repo = repos.find(r => r.name.toLowerCase() === project.toLowerCase());
            if (repo) {
                repoPath = repo.path;
            }
        }
    } catch (error) {
        console.error('Failed to get repository path:', error);
    }

    if (!repoPath) {
        showMessage('Could not determine repository path.', 'error');
        return;
    }

    // Show progress
    progressDiv.innerHTML = `
        <div class="spinner"></div>
        <span>Running targeted analysis on ${methods.length} method(s)...</span>
    `;
    progressDiv.classList.remove('hidden');

    try {
        const response = await fetch('/api/roslyn/analyze-targeted', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${auth.getAccessToken()}`
            },
            body: JSON.stringify({
                methods: methods,
                repoPath: repoPath,
                project: project
            })
        });

        const result = await response.json();

        if (result.success) {
            progressDiv.innerHTML = `
                <span style="color: #22c55e;">&#10003; Analysis complete!
                Analyzed ${result.files_found || 0} files,
                found ${result.data_summary?.call_graph || 0} call relationships.</span>
            `;

            // Refresh the results to show updated caller info
            showMessage('Call graph updated. Re-pulling to show caller information...', 'success');
            setTimeout(() => pullChanges(), 1500);
        } else {
            progressDiv.innerHTML = `
                <span style="color: #f87171;">&#10007; Analysis failed: ${escapeHtml(result.error || 'Unknown error')}</span>
            `;
        }

    } catch (error) {
        console.error('Targeted analysis failed:', error);
        progressDiv.innerHTML = `
            <span style="color: #f87171;">&#10007; Analysis failed: ${escapeHtml(error.message)}</span>
        `;
    }
}

// Run full repository analysis
async function runFullAnalysis(project) {
    const progressDiv = document.getElementById('analysisProgress');
    if (!progressDiv) return;

    // Get repository path from API
    let repoPath = null;
    try {
        const reposResponse = await fetch('/api/git/repositories');
        if (reposResponse.ok) {
            const repos = await reposResponse.json();
            const repo = repos.find(r => r.name.toLowerCase() === project.toLowerCase());
            if (repo) {
                repoPath = repo.path;
            }
        }
    } catch (error) {
        console.error('Failed to get repository path:', error);
    }

    if (!repoPath) {
        showMessage('Could not determine repository path.', 'error');
        return;
    }

    // Show progress
    progressDiv.innerHTML = `
        <div class="spinner"></div>
        <span>Running full repository analysis... This may take several minutes.</span>
    `;
    progressDiv.classList.remove('hidden');

    try {
        const response = await fetch('/api/roslyn/analyze-full', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${auth.getAccessToken()}`
            },
            body: JSON.stringify({
                repoPath: repoPath,
                project: project,
                generateEmbeddings: true
            })
        });

        const result = await response.json();

        if (result.success) {
            progressDiv.innerHTML = `
                <span style="color: #22c55e;">&#10003; Full analysis complete!
                Analyzed ${result.total_files || 0} files,
                found ${result.data_summary?.methods || 0} methods and
                ${result.data_summary?.call_graph || 0} call relationships.</span>
            `;

            // Refresh the call graph status
            await checkCallGraphStatus(project);

            // Re-pull to show updated caller info
            showMessage('Full analysis complete. Re-pulling to show caller information...', 'success');
            setTimeout(() => pullChanges(), 1500);
        } else {
            progressDiv.innerHTML = `
                <span style="color: #f87171;">&#10007; Analysis failed: ${escapeHtml(result.error || result.detail || 'Unknown error')}</span>
            `;
        }

    } catch (error) {
        console.error('Full analysis failed:', error);
        progressDiv.innerHTML = `
            <span style="color: #f87171;">&#10007; Analysis failed: ${escapeHtml(error.message)}</span>
        `;
    }
}
