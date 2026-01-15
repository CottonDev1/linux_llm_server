// Code Flow Analysis - Client Application

const API_BASE = window.location.origin;
let isQuerying = false;
let currentAbortController = null;
let currentZoom = 1.0;
let currentView = 'timeline';
let flowData = null; // Stores the current analysis results

// Initialize the application
async function init() {
    await checkHealth();
    await loadProjects();
    updateButtonStates();
    setInterval(checkHealth, 30000);
}

// Check system health
async function checkHealth() {
    try {
        const response = await fetch(`${API_BASE}/api/health`);
        const data = await response.json();

        const indicator = document.getElementById('statusIndicator');
        const statusText = document.getElementById('statusText');

        if (data.status === 'healthy') {
            indicator.classList.remove('disconnected');
            statusText.textContent = `System Online - ${data.model}`;
        } else {
            indicator.classList.add('disconnected');
            statusText.textContent = 'System Degraded';
        }
    } catch (error) {
        const indicator = document.getElementById('statusIndicator');
        const statusText = document.getElementById('statusText');
        indicator.classList.add('disconnected');
        statusText.textContent = 'System Offline';
    }
}

// Load available projects
async function loadProjects() {
    try {
        const response = await fetch(`${API_BASE}/api/projects`);
        const data = await response.json();

        const select = document.getElementById('projectFilter');
        data.projects.forEach(project => {
            const option = document.createElement('option');
            option.value = project.id === 'all' ? '' : project.id;
            option.textContent = project.name;
            option.title = project.description;
            select.appendChild(option);
        });
    } catch (error) {
        console.error('Failed to load projects:', error);
    }
}

// Send query to analyze code flow
async function sendQuery() {
    const input = document.getElementById('queryInput');
    const query = input.value.trim();

    if (!query || isQuerying) return;

    const project = document.getElementById('projectFilter').value;

    // Hide other containers, show loading
    document.getElementById('resultsContainer').style.display = 'none';
    document.getElementById('errorContainer').style.display = 'none';
    document.getElementById('loadingContainer').style.display = 'flex';

    // Create abort controller
    currentAbortController = new AbortController();

    try {
        isQuerying = true;
        updateButtonStates();

        // Step 1: Query the RAG system for code flow context
        updateLoadingStatus('Searching for relevant code...');

        const response = await fetch(`${API_BASE}/api/query`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                query: `Code execution flow: ${query}. Include method calls, class names, file paths, and SQL operations.`,
                project,
                limit: 10, // Get more context for flow analysis
                temperature: 0.5,
                maxSourceLength: 3000
            }),
            signal: currentAbortController.signal
        });

        if (!response.ok) {
            throw new Error(`Server error: ${response.statusText}`);
        }

        const data = await response.json();

        updateLoadingStatus('Analyzing execution path...');

        // Step 2: Parse the response to extract flow information
        const flowAnalysis = await analyzeCodeFlow(data.answer, data.sources, query);

        // Step 3: Render the results
        await renderFlowVisualization(flowAnalysis);

        // Show results
        document.getElementById('loadingContainer').style.display = 'none';
        document.getElementById('resultsContainer').style.display = 'grid';

    } catch (error) {
        document.getElementById('loadingContainer').style.display = 'none';
        if (error.name === 'AbortError') {
            showError('Request was cancelled.');
        } else {
            showError(`Failed to analyze code flow: ${error.message}`);
        }
    } finally {
        isQuerying = false;
        currentAbortController = null;
        updateButtonStates();
    }
}

// Stop current query
function stopQuery() {
    if (currentAbortController) {
        currentAbortController.abort();
        currentAbortController = null;
    }
}

// Update button states
function updateButtonStates() {
    const sendButton = document.getElementById('sendButton');
    const stopButton = document.getElementById('stopButton');

    sendButton.disabled = isQuerying;
    stopButton.style.display = isQuerying ? 'inline-flex' : 'none';
}

// Update loading status message
function updateLoadingStatus(message) {
    const statusEl = document.getElementById('loadingStatus');
    if (statusEl) {
        statusEl.textContent = message;
    }
}

// Analyze code flow from LLM response
async function analyzeCodeFlow(answer, sources, originalQuery) {
    // Extract key information from the answer and sources
    const flow = {
        query: originalQuery,
        answer: answer,
        sources: sources,
        steps: [],
        methods: [],
        sqlOperations: []
    };

    // Parse the answer for method calls and execution steps
    const methodPattern = /(\w+)\.(\w+)\s*\(/g;
    const sqlPattern = /(?:stored procedure|sproc|exec|execute)\s+(\w+)/gi;
    const filePattern = /(\w+\.cs|\w+\.js|\w+\.ts)/gi;

    let match;
    const seenMethods = new Set();
    const seenFiles = new Set();

    // Extract methods from answer
    while ((match = methodPattern.exec(answer)) !== null) {
        const className = match[1];
        const methodName = match[2];
        const fullName = `${className}.${methodName}`;

        if (!seenMethods.has(fullName)) {
            seenMethods.add(fullName);
            flow.methods.push({
                class: className,
                method: methodName,
                fullName: fullName,
                type: inferMethodType(className, methodName)
            });
        }
    }

    // Extract SQL operations
    while ((match = sqlPattern.exec(answer)) !== null) {
        const sprocName = match[1];
        if (!flow.sqlOperations.find(s => s.name === sprocName)) {
            flow.sqlOperations.push({
                name: sprocName,
                type: 'stored_procedure'
            });
        }
    }

    // Extract files from sources
    sources.forEach(source => {
        if (source.file && !seenFiles.has(source.file)) {
            seenFiles.add(source.file);
        }
    });

    // Build execution steps by analyzing the answer structure
    flow.steps = buildExecutionSteps(answer, flow.methods, flow.sqlOperations, sources);

    return flow;
}

// Infer method type from class/method name
function inferMethodType(className, methodName) {
    const lowerClass = className.toLowerCase();
    const lowerMethod = methodName.toLowerCase();

    if (lowerClass.includes('form') || lowerClass.includes('ui') || lowerMethod.includes('click') || lowerMethod.includes('button')) {
        return 'UI';
    } else if (lowerClass.includes('datalayer') || lowerClass.includes('repository') || lowerClass.includes('data')) {
        return 'Data Layer';
    } else if (lowerClass.includes('service') || lowerClass.includes('manager') || lowerClass.includes('handler')) {
        return 'Business Logic';
    } else if (lowerClass.includes('controller') || lowerClass.includes('api')) {
        return 'API';
    }
    return 'Business Logic';
}

// Build execution steps from analyzed content
function buildExecutionSteps(answer, methods, sqlOps, sources) {
    const steps = [];
    let stepCounter = 1;

    // Add UI layer step if we have UI methods
    const uiMethods = methods.filter(m => m.type === 'UI');
    if (uiMethods.length > 0) {
        steps.push({
            id: `step${stepCounter++}`,
            layer: 'UI Layer',
            description: 'User interaction triggers event handler',
            methods: uiMethods.map(m => m.fullName),
            details: `User clicks button or interacts with form, triggering ${uiMethods[0].fullName}`
        });
    }

    // Combine all non-UI and non-Database methods into execution flow steps
    const flowMethods = methods.filter(m => m.type !== 'UI');
    if (flowMethods.length > 0) {
        // Group methods by their appearance order in the answer
        flowMethods.forEach((method, idx) => {
            if (idx < 5) { // Limit to top 5 most relevant methods
                steps.push({
                    id: `step${stepCounter++}`,
                    layer: method.fullName,
                    description: `Method execution`,
                    methods: [method.fullName],
                    details: `Executing ${method.fullName}`
                });
            }
        });
    }

    // Add database steps
    if (sqlOps.length > 0) {
        steps.push({
            id: `step${stepCounter++}`,
            layer: 'Database Operations',
            description: 'SQL operations and data manipulation',
            methods: sqlOps.map(s => s.name),
            details: `Stored procedures: ${sqlOps.map(s => s.name).join(', ')}`
        });
    }

    // If we couldn't identify specific steps, create a generic flow
    if (steps.length === 0) {
        // Parse the answer into logical paragraphs
        const paragraphs = answer.split('\n\n').filter(p => p.trim().length > 20);
        paragraphs.slice(0, 5).forEach((para, idx) => {
            steps.push({
                id: `step${stepCounter++}`,
                layer: `Step ${idx + 1}`,
                description: para.substring(0, 100) + '...',
                methods: [],
                details: para
            });
        });
    }

    return steps;
}

// Render flow visualization
async function renderFlowVisualization(flowData) {
    // Store for later use
    window.currentFlowData = flowData;

    // Hide placeholders
    document.getElementById('graphPlaceholder').style.display = 'none';
    document.getElementById('detailsPlaceholder').style.display = 'none';

    // Render Mermaid diagram
    renderMermaidDiagram(flowData);

    // Render details panel
    renderDetailsPanel(flowData);
}

// Render Mermaid flowchart diagram
async function renderMermaidDiagram(flowData) {
    const container = document.getElementById('graphContainer');

    // Build Mermaid syntax
    let mermaidCode = 'flowchart TD\n';

    // Add start node
    mermaidCode += '    Start([User Action])\n';

    // Add step nodes
    flowData.steps.forEach((step, idx) => {
        const nodeId = step.id;
        const label = `${step.layer}`;
        const subtitle = step.methods.slice(0, 2).join('<br/>');

        // Choose node shape based on layer
        let nodeShape = '[]';
        if (step.layer.includes('UI')) {
            nodeShape = `[${label}<br/><small>${subtitle}</small>]`;
        } else if (step.layer.includes('Database')) {
            nodeShape = `[(${label}<br/><small>${subtitle}</small>)]`;
        } else {
            nodeShape = `[${label}<br/><small>${subtitle}</small>]`;
        }

        mermaidCode += `    ${nodeId}${nodeShape}\n`;

        // Add styling classes
        if (step.layer.includes('UI')) {
            mermaidCode += `    class ${nodeId} uiNode\n`;
        } else if (step.layer.includes('Database')) {
            mermaidCode += `    class ${nodeId} dbNode\n`;
        } else {
            mermaidCode += `    class ${nodeId} methodNode\n`;
        }
    });

    // Add end node
    mermaidCode += '    End([Complete])\n';

    // Add connections
    mermaidCode += '    Start --> ' + (flowData.steps[0]?.id || 'End') + '\n';
    for (let i = 0; i < flowData.steps.length - 1; i++) {
        mermaidCode += `    ${flowData.steps[i].id} --> ${flowData.steps[i + 1].id}\n`;
    }
    if (flowData.steps.length > 0) {
        mermaidCode += `    ${flowData.steps[flowData.steps.length - 1].id} --> End\n`;
    }

    // Add styling
    mermaidCode += `
    classDef uiNode fill:#3b82f6,stroke:#60a5fa,color:#fff
    classDef methodNode fill:#8b5cf6,stroke:#a78bfa,color:#fff
    classDef dbNode fill:#f59e0b,stroke:#fbbf24,color:#fff
    `;

    console.log('Mermaid diagram code:', mermaidCode);

    // Render with Mermaid
    try {
        container.innerHTML = ''; // Clear previous
        const { svg } = await mermaid.render('flowDiagram', mermaidCode);
        container.innerHTML = svg;

        // Add click handlers to nodes
        addNodeClickHandlers(container, flowData);
    } catch (error) {
        console.error('Mermaid rendering error:', error);
        container.innerHTML = `<div class="error-message">Failed to render diagram: ${error.message}</div>`;
    }
}

// Add click handlers to diagram nodes
function addNodeClickHandlers(container, flowData) {
    const nodes = container.querySelectorAll('.node');
    nodes.forEach((node, idx) => {
        node.style.cursor = 'pointer';
        node.addEventListener('click', () => {
            // Find corresponding step
            const stepId = node.id?.replace('flowchart-', '').replace('flowDiagram-', '');
            const step = flowData.steps.find(s => s.id === stepId || s.id === `step${idx}`);

            if (step) {
                showMethodDetails(step, flowData);
            }
        });
    });
}

// Render details panel
function renderDetailsPanel(flowData) {
    const container = document.getElementById('detailsContainer');

    let html = `
        <div class="flow-timeline">
    `;

    flowData.steps.forEach((step, idx) => {
        html += `
            <div class="timeline-item" onclick="showMethodDetails(window.currentFlowData.steps[${idx}], window.currentFlowData)">
                <div class="timeline-marker">${idx + 1}</div>
                <div class="timeline-content">
                    <div class="timeline-layer">${step.layer}</div>
                    <div class="timeline-description">${step.description}</div>
                    <div class="timeline-methods">
                        ${step.methods.slice(0, 3).map(m => `<span class="method-badge">${m}</span>`).join('')}
                        ${step.methods.length > 3 ? `<span class="method-badge">+${step.methods.length - 3} more</span>` : ''}
                    </div>
                </div>
            </div>
        `;
    });

    html += `
        </div>

        <div class="flow-sources">
            <h4>Referenced Source Files</h4>
            <div class="source-list">
    `;

    flowData.sources.forEach(source => {
        html += `
            <div class="source-item" onclick="showSourceDetails(${JSON.stringify(source).replace(/"/g, '&quot;')})">
                <div class="source-badge">${source.project}</div>
                <div class="source-file">${source.file}</div>
                <div class="source-similarity">${Math.round(source.similarity * 100)}% match</div>
            </div>
        `;
    });

    html += `
            </div>
        </div>
    `;

    container.innerHTML = html;
}

// Show method details in modal
function showMethodDetails(step, flowData) {
    const modal = document.getElementById('methodModal');
    const title = document.getElementById('methodModalTitle');
    const body = document.getElementById('methodModalBody');

    title.textContent = step.layer;

    let html = `
        <div class="method-detail-section">
            <h4>Description</h4>
            <p>${step.details}</p>
        </div>

        <div class="method-detail-section">
            <h4>Methods Involved</h4>
            <div class="method-list">
    `;

    step.methods.forEach(method => {
        html += `
            <div class="method-item">
                <code>${method}</code>
            </div>
        `;
    });

    html += `
            </div>
        </div>

        <div class="method-detail-section">
            <h4>Related Sources</h4>
            <div class="source-snippet-list">
    `;

    // Find relevant sources
    const relevantSources = flowData.sources.filter(source =>
        step.methods.some(m => source.snippet.includes(m.split('.')[1]))
    );

    if (relevantSources.length > 0) {
        relevantSources.slice(0, 3).forEach(source => {
            html += `
                <div class="source-snippet">
                    <div class="snippet-header">
                        <span class="snippet-file">${source.file}</span>
                        <span class="snippet-project">${source.project}</span>
                    </div>
                    <pre><code>${escapeHtml(source.snippet.substring(0, 400))}...</code></pre>
                </div>
            `;
        });
    } else {
        html += '<p class="no-sources">No direct source matches found.</p>';
    }

    html += `
            </div>
        </div>
    `;

    body.innerHTML = html;
    modal.style.display = 'flex';
}

// Show source file details in modal
function showSourceDetails(source) {
    const modal = document.getElementById('methodModal');
    const title = document.getElementById('methodModalTitle');
    const body = document.getElementById('methodModalBody');

    title.textContent = source.file;

    const html = `
        <div class="method-detail-section">
            <div class="source-meta">
                <span class="meta-item"><strong>Project:</strong> ${source.project}</span>
                <span class="meta-item"><strong>Category:</strong> ${source.category}</span>
                <span class="meta-item"><strong>Relevance:</strong> ${Math.round(source.similarity * 100)}%</span>
            </div>
        </div>

        <div class="method-detail-section">
            <h4>Code Snippet</h4>
            <pre><code>${escapeHtml(source.snippet)}</code></pre>
        </div>
    `;

    body.innerHTML = html;
    modal.style.display = 'flex';
}

// Close method modal
function closeMethodModal(event) {
    if (!event || event.target.id === 'methodModal') {
        document.getElementById('methodModal').style.display = 'none';
    }
}

// Show error message
function showError(message) {
    const errorContainer = document.getElementById('errorContainer');
    const errorMessage = document.getElementById('errorMessage');

    errorMessage.textContent = message;
    errorContainer.style.display = 'flex';
}

// Hide error message
function hideError() {
    document.getElementById('errorContainer').style.display = 'none';
}

// Handle keyboard shortcuts
function handleKeyDown(event) {
    if (event.key === 'Enter' && !event.shiftKey) {
        event.preventDefault();
        sendQuery();
    }
}

// Ask example question
function askExample(query) {
    document.getElementById('queryInput').value = query;
    sendQuery();
}

// Zoom controls
function zoomIn() {
    currentZoom = Math.min(currentZoom + 0.2, 3.0);
    applyZoom();
}

function zoomOut() {
    currentZoom = Math.max(currentZoom - 0.2, 0.5);
    applyZoom();
}

function resetZoom() {
    currentZoom = 1.0;
    applyZoom();
}

function applyZoom() {
    const graphContainer = document.getElementById('graphContainer');
    graphContainer.style.transform = `scale(${currentZoom})`;
    graphContainer.style.transformOrigin = 'top left';
}

// Toggle details view
function toggleDetailsView(view) {
    currentView = view;
    // Could implement different view modes here
    console.log('View mode:', view);
}

// Utility function to escape HTML
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// Initialize on page load
window.addEventListener('DOMContentLoaded', init);
