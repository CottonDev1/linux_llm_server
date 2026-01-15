// API Documentation Common JavaScript

let API_DATA = { categories: [] };
let searchTerm = '';

function logout() {
    const auth = new AuthClient();
    auth.logout();
}

async function initApiDocs(jsonPath) {
    try {
        const response = await fetch(jsonPath);
        if (!response.ok) {
            throw new Error(`Failed to load API documentation: ${response.status}`);
        }
        API_DATA = await response.json();
        console.log(`Loaded API documentation v${API_DATA.version} (${API_DATA.lastUpdated})`);
        renderCategories();
        updateStats();
    } catch (error) {
        console.error('Error loading API documentation:', error);
        document.getElementById('apiCategories').innerHTML = `
            <div style="text-align: center; padding: 40px; color: var(--text-muted);">
                <p>Failed to load API documentation.</p>
                <p style="font-size: 12px;">${error.message}</p>
            </div>
        `;
    }
}

function renderCategories() {
    const container = document.getElementById('apiCategories');
    let html = '';

    const categories = API_DATA.categories || [];

    categories.forEach(category => {
        const endpoints = getFilteredEndpoints(category.endpoints);
        if (endpoints.length === 0) return;

        const isExpanded = searchTerm.length > 0;

        html += `
            <div class="api-category ${isExpanded ? 'expanded' : ''}" data-category-id="${category.id}">
                <div class="api-category-header" onclick="toggleCategory('${category.id}')">
                    <div class="api-category-left">
                        <span class="api-category-icon">${category.icon}</span>
                        <span class="api-category-title">${category.name}</span>
                        <span class="api-category-count">${endpoints.length}</span>
                    </div>
                    <ewr-icon name="chevron-down" size="18" class="api-category-chevron"></ewr-icon>
                </div>
                <div class="api-category-content">
                    <div class="api-category-inner">
                        ${endpoints.map(ep => renderEndpoint(ep)).join('')}
                    </div>
                </div>
            </div>
        `;
    });

    container.innerHTML = html || '<p style="color: var(--text-muted); text-align: center; padding: 40px;">No endpoints match your search.</p>';
}

function renderEndpoint(ep) {
    const methodClass = `method-${ep.method.toLowerCase()}`;
    const endpointId = `ep-${ep.path.replace(/[/:{}]/g, '-')}-${Math.random().toString(36).substr(2, 5)}`;

    return `
        <div class="api-endpoint" id="${endpointId}">
            <div class="api-endpoint-header" onclick="toggleEndpoint('${endpointId}')">
                <span class="method-badge ${methodClass}">${ep.method}</span>
                <span class="api-endpoint-path">${ep.path}</span>
                <span class="api-endpoint-desc">${ep.desc}</span>
                <ewr-icon name="chevron-down" size="18" class="api-endpoint-chevron"></ewr-icon>
            </div>
            <div class="api-endpoint-details">
                <div class="api-endpoint-details-inner">
                    <div class="api-detail-section">
                        <div class="api-detail-label">Description</div>
                        <div class="api-detail-description">${ep.desc}</div>
                    </div>
                    ${ep.params && ep.params.length > 0 ? `
                        <div class="api-detail-section">
                            <div class="api-detail-label">Parameters</div>
                            <table class="param-table">
                                <thead>
                                    <tr>
                                        <th>Name</th>
                                        <th>Type</th>
                                        <th>Required</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    ${ep.params.map(p => `
                                        <tr>
                                            <td><span class="param-name">${p.name}</span></td>
                                            <td><span class="param-type">${p.type}</span></td>
                                            <td>${p.required ? '<span class="param-required">Required</span>' : '<span class="param-optional">Optional</span>'}</td>
                                        </tr>
                                    `).join('')}
                                </tbody>
                            </table>
                        </div>
                    ` : ''}
                    <div class="api-detail-section">
                        <div class="api-detail-label">Response</div>
                        <div class="code-block">
                            <div class="code-block-header">
                                <span class="code-block-label">Response Format</span>
                                <button class="code-copy-btn" onclick="copyCode(this, \`${ep.response.replace(/`/g, '\\`')}\`)">Copy</button>
                            </div>
                            <pre>${ep.response}</pre>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    `;
}

function getFilteredEndpoints(endpoints) {
    if (!searchTerm) return endpoints;

    const term = searchTerm.toLowerCase();
    return endpoints.filter(ep =>
        ep.method.toLowerCase().includes(term) ||
        ep.path.toLowerCase().includes(term) ||
        ep.desc.toLowerCase().includes(term)
    );
}

function filterEndpoints() {
    searchTerm = document.getElementById('apiSearch').value;
    renderCategories();
}

function toggleCategory(categoryId) {
    const category = document.querySelector(`[data-category-id="${categoryId}"]`);
    if (category) {
        category.classList.toggle('expanded');
    }
}

function toggleEndpoint(endpointId) {
    const endpoint = document.getElementById(endpointId);
    if (endpoint) {
        endpoint.classList.toggle('expanded');
    }
}

function copyCode(btn, text) {
    navigator.clipboard.writeText(text).then(() => {
        btn.textContent = 'Copied!';
        btn.classList.add('copied');
        setTimeout(() => {
            btn.textContent = 'Copy';
            btn.classList.remove('copied');
        }, 2000);
    });
}

function updateStats() {
    let totalEndpoints = 0;
    const categories = API_DATA.categories || [];

    categories.forEach(cat => totalEndpoints += cat.endpoints.length);

    document.getElementById('totalEndpoints').textContent = totalEndpoints;
    document.getElementById('totalCategories').textContent = categories.length;
}
