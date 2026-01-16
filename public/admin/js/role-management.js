/**
 * Role Management JavaScript - Drag & Drop Edition
 * Manages category assignments to roles via drag-and-drop interface
 * Supports custom role creation
 */

// Initialize auth client
const auth = new AuthClient();

// Store role-category mappings
let roleCategoryMappings = {};

// Store categories fetched from navigation API, organized by type
let adminCategories = [];
let generalCategories = [];

// Store all available roles (including custom ones)
let availableRoles = [];

// Default role definitions (built-in roles)
const DEFAULT_ROLE_DEFINITIONS = {
    user: {
        name: 'User',
        description: 'Basic access for standard users',
        inherits: null,
        isBuiltIn: true
    },
    developer: {
        name: 'Developer',
        description: 'Extended access including development tools',
        inherits: 'user',
        isBuiltIn: true
    },
    admin: {
        name: 'Admin',
        description: 'Full access to all system features',
        inherits: 'developer',
        isBuiltIn: true
    }
};

// Check authentication
if (!auth.isAuthenticated()) {
    window.location.href = '/login.html?redirect=' + encodeURIComponent(window.location.pathname);
} else {
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

/**
 * Load categories from navigation API and organize by admin vs general
 * Admin categories: Pages in the /admin/ folder
 * General categories: Pages NOT in the /admin/ folder (public pages)
 */
async function loadCategoriesFromNav() {
    try {
        const response = await fetch('/api/admin/navigation');
        if (!response.ok) {
            throw new Error('Failed to fetch navigation');
        }
        const data = await response.json();

        adminCategories = [];
        generalCategories = [];

        if (data.success && data.categories) {
            data.categories.forEach(cat => {
                const category = {
                    id: cat.id,
                    name: cat.name,
                    icon: cat.icon || '<ewr-icon name="folder" size="14"></ewr-icon>',
                    requiredRole: cat.defaultRole || 'admin',
                    pages: cat.pages || [],
                    isAdmin: cat.isAdmin !== false // Default to true if not specified
                };

                // Categorize based on isAdmin flag from API
                if (category.isAdmin) {
                    adminCategories.push(category);
                } else {
                    generalCategories.push(category);
                }
            });
        }
    } catch (error) {
        console.error('Error loading categories:', error);
        adminCategories = [];
        generalCategories = [];
    }
}

/**
 * Get all categories combined
 */
function getAllCategories() {
    return [...adminCategories, ...generalCategories];
}

/**
 * Load available roles from server
 */
async function loadAvailableRoles() {
    try {
        const response = await fetch('/api/auth/roles', {
            headers: {
                'Authorization': `Bearer ${auth.getAccessToken()}`
            }
        });

        if (response.ok) {
            const data = await response.json();
            availableRoles = data.roles || ['user', 'developer', 'admin'];
        } else {
            // Fallback to default roles if endpoint doesn't exist
            availableRoles = ['user', 'developer', 'admin'];
        }
    } catch (error) {
        console.error('Error loading roles:', error);
        availableRoles = ['user', 'developer', 'admin'];
    }
}

/**
 * Get role definition (built-in or custom)
 */
function getRoleDefinition(role) {
    if (DEFAULT_ROLE_DEFINITIONS[role]) {
        return DEFAULT_ROLE_DEFINITIONS[role];
    }
    // Custom role
    return {
        name: role.charAt(0).toUpperCase() + role.slice(1),
        description: 'Custom role',
        inherits: null,
        isBuiltIn: false
    };
}

/**
 * Initialize the page
 */
async function init() {
    await loadCategoriesFromNav();
    await loadAvailableRoles();
    await loadRoleCategoryMappings();
    renderBadgePools();
    renderRoleCards();
}

/**
 * Load role-category mappings from server
 */
async function loadRoleCategoryMappings() {
    try {
        const response = await fetch('/api/auth/role-categories', {
            headers: {
                'Authorization': `Bearer ${auth.getAccessToken()}`
            }
        });

        if (!response.ok) {
            throw new Error('Failed to load role categories');
        }

        const data = await response.json();
        roleCategoryMappings = data || {};

        // Ensure all available roles have an entry
        availableRoles.forEach(role => {
            if (!roleCategoryMappings[role]) {
                roleCategoryMappings[role] = [];
            }
        });

    } catch (error) {
        console.error('Error loading role categories:', error);
        // Initialize with empty arrays for each role
        roleCategoryMappings = {};
        availableRoles.forEach(role => {
            roleCategoryMappings[role] = [];
        });
    }
}

/**
 * Render the badge pools (admin and general screens)
 */
function renderBadgePools() {
    const adminPool = document.getElementById('adminBadgePool');
    const generalPool = document.getElementById('generalBadgePool');
    const adminCount = document.getElementById('adminScreenCount');
    const generalCount = document.getElementById('generalScreenCount');

    // Render admin categories
    if (adminPool) {
        adminPool.innerHTML = adminCategories.map(cat => createCategoryBadge(cat)).join('');
    }
    if (adminCount) {
        adminCount.textContent = adminCategories.length;
    }

    // Render general categories
    if (generalPool) {
        if (generalCategories.length > 0) {
            generalPool.innerHTML = generalCategories.map(cat => createCategoryBadge(cat)).join('');
        } else {
            generalPool.innerHTML = '<span style="color: var(--text-muted); font-size: 12px;">No general screens configured</span>';
        }
    }
    if (generalCount) {
        generalCount.textContent = generalCategories.length;
    }
}

/**
 * Create a category badge HTML
 */
function createCategoryBadge(category) {
    return `
        <div class="screen-badge"
             draggable="true"
             data-category="${category.id}"
             ondragstart="handleDragStart(event)"
             ondragend="handleDragEnd(event)"
             title="${category.name} (${category.pages.length} screens)">
            ${category.icon}
            <span>${category.name}</span>
        </div>
    `;
}

/**
 * Render the role cards including the add role card
 */
function renderRoleCards() {
    const rolesGrid = document.getElementById('rolesGrid');
    const categories = getAllCategories();

    let html = '';

    // Render existing roles
    availableRoles.forEach(role => {
        const roleDef = getRoleDefinition(role);
        const assignedCategories = roleCategoryMappings[role] || [];

        html += `
            <div class="role-card"
                 data-role="${role}"
                 ondrop="handleDrop(event, '${role}')"
                 ondragover="handleDragOver(event)"
                 ondragleave="handleDragLeave(event)">
                <div class="role-card-header">
                    <span class="role-badge ${role}">${roleDef.name}</span>
                    <span class="role-description">${roleDef.description}</span>
                    ${!roleDef.isBuiltIn ? `<button class="remove-role-btn" onclick="deleteRole('${role}')" title="Delete role"><ewr-icon name="trash-2" size="16"></ewr-icon></button>` : ''}
                </div>
                <div class="role-card-body">
                    <div class="role-categories" id="role-${role}">
        `;

        if (assignedCategories.length === 0) {
            html += `
                <div class="empty-state">
                    <ewr-icon name="plus" size="24"></ewr-icon>
                    <div class="empty-state-text">Drag categories here</div>
                </div>
            `;
        } else {
            assignedCategories.forEach(categoryId => {
                const category = categories.find(c => c.id === categoryId);
                if (category) {
                    html += `
                        <div class="role-category-badge" data-category="${categoryId}">
                            ${category.icon}
                            <span>${category.name}</span>
                            <div class="remove-btn" onclick="removeCategoryFromRole('${role}', '${categoryId}')">
                                <ewr-icon name="x" size="16"></ewr-icon>
                            </div>
                        </div>
                    `;
                }
            });
        }

        html += `
                    </div>
                </div>
            </div>
        `;
    });

    // Add the "Add Role" card
    html += `
        <div class="add-role-card" id="addRoleCard" onclick="expandAddRoleCard(event)">
            <div class="add-role-btn">
                <ewr-icon name="plus" size="32"></ewr-icon>
                <span>Add New Role</span>
            </div>
            <div class="add-role-form">
                <div class="add-role-form-header">
                    <input type="text" id="newRoleName" placeholder="Enter role name" onclick="event.stopPropagation()" onkeypress="handleRoleNameKeypress(event)">
                    <div class="add-role-form-actions">
                        <button class="btn-save-role" onclick="saveNewRole(event)">Save</button>
                        <button class="btn-cancel-role" onclick="cancelAddRole(event)">Cancel</button>
                    </div>
                </div>
                <div class="add-role-form-body">
                    <div class="empty-state">
                        <ewr-icon name="info" size="24"></ewr-icon>
                        <div class="empty-state-text">Save role first, then drag categories here</div>
                    </div>
                </div>
            </div>
        </div>
    `;

    rolesGrid.innerHTML = html;
}

/**
 * Expand the add role card
 */
function expandAddRoleCard(event) {
    const card = document.getElementById('addRoleCard');
    if (!card.classList.contains('expanded')) {
        card.classList.add('expanded');
        const input = document.getElementById('newRoleName');
        if (input) {
            setTimeout(() => input.focus(), 100);
        }
    }
}

/**
 * Cancel adding a new role
 */
function cancelAddRole(event) {
    event.stopPropagation();
    const card = document.getElementById('addRoleCard');
    card.classList.remove('expanded');
    const input = document.getElementById('newRoleName');
    if (input) {
        input.value = '';
    }
}

/**
 * Handle keypress in role name input
 */
function handleRoleNameKeypress(event) {
    if (event.key === 'Enter') {
        saveNewRole(event);
    } else if (event.key === 'Escape') {
        cancelAddRole(event);
    }
}

/**
 * Save a new role
 */
async function saveNewRole(event) {
    event.stopPropagation();

    const input = document.getElementById('newRoleName');
    const roleName = input.value.trim().toLowerCase();

    if (!roleName) {
        showMessage('Please enter a role name', 'error');
        return;
    }

    // Validate role name (alphanumeric and underscores only)
    if (!/^[a-z][a-z0-9_]*$/.test(roleName)) {
        showMessage('Role name must start with a letter and contain only lowercase letters, numbers, and underscores', 'error');
        return;
    }

    // Check if role already exists
    if (availableRoles.includes(roleName)) {
        showMessage('A role with this name already exists', 'error');
        return;
    }

    updateSaveIndicator('saving');

    try {
        const response = await fetch('/api/auth/roles', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${auth.getAccessToken()}`
            },
            body: JSON.stringify({ name: roleName })
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.error || 'Failed to create role');
        }

        // Add to local state
        availableRoles.push(roleName);
        roleCategoryMappings[roleName] = [];

        // Reset form and re-render
        input.value = '';
        const card = document.getElementById('addRoleCard');
        card.classList.remove('expanded');

        renderRoleCards();
        updateSaveIndicator('saved');
        showMessage(`Role "${roleName}" created successfully`, 'success');

    } catch (error) {
        console.error('Error creating role:', error);
        updateSaveIndicator('error');
        showMessage(error.message || 'Failed to create role', 'error');
    }
}

/**
 * Delete a custom role
 */
async function deleteRole(role) {
    if (DEFAULT_ROLE_DEFINITIONS[role]) {
        showMessage('Cannot delete built-in roles', 'error');
        return;
    }

    if (!confirm(`Are you sure you want to delete the role "${role}"? This cannot be undone.`)) {
        return;
    }

    updateSaveIndicator('saving');

    try {
        const response = await fetch(`/api/auth/roles/${role}`, {
            method: 'DELETE',
            headers: {
                'Authorization': `Bearer ${auth.getAccessToken()}`
            }
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.error || 'Failed to delete role');
        }

        // Remove from local state
        availableRoles = availableRoles.filter(r => r !== role);
        delete roleCategoryMappings[role];

        renderRoleCards();
        updateSaveIndicator('saved');
        showMessage(`Role "${role}" deleted successfully`, 'success');

    } catch (error) {
        console.error('Error deleting role:', error);
        updateSaveIndicator('error');
        showMessage(error.message || 'Failed to delete role', 'error');
    }
}

/**
 * Handle drag start event
 */
function handleDragStart(event) {
    const categoryId = event.target.dataset.category;
    event.dataTransfer.effectAllowed = 'copy';
    event.dataTransfer.setData('text/plain', categoryId);
    event.target.classList.add('dragging');
}

/**
 * Handle drag end event
 */
function handleDragEnd(event) {
    event.target.classList.remove('dragging');
}

/**
 * Handle drag over event (required to allow drop)
 */
function handleDragOver(event) {
    event.preventDefault();
    event.dataTransfer.dropEffect = 'copy';

    const roleCard = event.currentTarget;
    if (!roleCard.classList.contains('drag-over')) {
        roleCard.classList.add('drag-over');
    }
}

/**
 * Handle drag leave event
 */
function handleDragLeave(event) {
    // Only remove if we're actually leaving the card
    if (event.currentTarget === event.target || !event.currentTarget.contains(event.relatedTarget)) {
        event.currentTarget.classList.remove('drag-over');
    }
}

/**
 * Handle drop event
 */
async function handleDrop(event, role) {
    event.preventDefault();
    event.currentTarget.classList.remove('drag-over');

    const categoryId = event.dataTransfer.getData('text/plain');

    if (!categoryId) return;

    await addCategoryToRole(role, categoryId);
}

/**
 * Add a category to a role
 */
async function addCategoryToRole(role, categoryId) {
    // Check if category already assigned
    if (roleCategoryMappings[role] && roleCategoryMappings[role].includes(categoryId)) {
        showMessage(`${categoryId} is already assigned to ${role}`, 'error');
        return;
    }

    // Initialize array if needed
    if (!roleCategoryMappings[role]) {
        roleCategoryMappings[role] = [];
    }

    // Add category
    roleCategoryMappings[role].push(categoryId);

    // Auto-save
    await saveRoleCategory(role, categoryId);

    // Re-render
    renderRoleCards();
}

/**
 * Remove a category from a role
 */
async function removeCategoryFromRole(role, categoryId) {
    // Remove category
    roleCategoryMappings[role] = roleCategoryMappings[role].filter(id => id !== categoryId);

    // Auto-save (delete)
    await deleteRoleCategory(role, categoryId);

    // Re-render
    renderRoleCards();
}

/**
 * Save a role-category assignment to server
 */
async function saveRoleCategory(role, categoryId) {
    updateSaveIndicator('saving');

    try {
        const response = await fetch(`/api/auth/role-categories/${role}`, {
            method: 'PUT',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${auth.getAccessToken()}`
            },
            body: JSON.stringify({
                categories: roleCategoryMappings[role]
            })
        });

        if (!response.ok) {
            throw new Error('Failed to save category assignment');
        }

        updateSaveIndicator('saved');
        showMessage('Role update saved', 'success');

    } catch (error) {
        console.error('Error saving category assignment:', error);
        updateSaveIndicator('error');
        showMessage('Failed to save: ' + error.message, 'error');

        // Revert on error
        roleCategoryMappings[role] = roleCategoryMappings[role].filter(id => id !== categoryId);
        renderRoleCards();
    }
}

/**
 * Delete a role-category assignment from server
 */
async function deleteRoleCategory(role, categoryId) {
    updateSaveIndicator('saving');

    try {
        const response = await fetch(`/api/auth/role-categories/${role}/${categoryId}`, {
            method: 'DELETE',
            headers: {
                'Authorization': `Bearer ${auth.getAccessToken()}`
            }
        });

        if (!response.ok) {
            throw new Error('Failed to remove category assignment');
        }

        updateSaveIndicator('saved');
        showMessage('Role update saved', 'success');

    } catch (error) {
        console.error('Error removing category assignment:', error);
        updateSaveIndicator('error');
        showMessage('Failed to remove: ' + error.message, 'error');

        // Revert on error
        roleCategoryMappings[role].push(categoryId);
        renderRoleCards();
    }
}

/**
 * Update save indicator
 */
function updateSaveIndicator(state) {
    const indicator = document.getElementById('saveIndicator');
    if (!indicator) {
        console.warn('Save indicator element not found');
        return;
    }

    switch (state) {
        case 'saving':
            indicator.className = 'save-indicator saving';
            indicator.textContent = 'Saving...';
            break;
        case 'saved':
            indicator.className = 'save-indicator saved';
            indicator.textContent = 'All changes saved';
            setTimeout(() => {
                if (indicator.textContent === 'All changes saved') {
                    indicator.textContent = '';
                    indicator.className = 'save-indicator';
                }
            }, 3000);
            break;
        case 'error':
            indicator.className = 'save-indicator error';
            indicator.textContent = 'Save failed';
            setTimeout(() => {
                indicator.className = 'save-indicator';
            }, 3000);
            break;
        default:
            indicator.textContent = '';
            indicator.className = 'save-indicator';
    }
}

/**
 * Show message to user
 */
function showMessage(message, type = 'success') {
    const messageDiv = document.getElementById('roleMessage');
    if (!messageDiv) return;

    messageDiv.className = `message ${type}`;
    messageDiv.textContent = message;
    messageDiv.style.display = 'block';

    setTimeout(() => {
        messageDiv.style.display = 'none';
    }, 5000);
}

// Make functions available globally
window.init = init;
window.handleDragStart = handleDragStart;
window.handleDragEnd = handleDragEnd;
window.handleDragOver = handleDragOver;
window.handleDragLeave = handleDragLeave;
window.handleDrop = handleDrop;
window.removeCategoryFromRole = removeCategoryFromRole;
window.expandAddRoleCard = expandAddRoleCard;
window.cancelAddRole = cancelAddRole;
window.saveNewRole = saveNewRole;
window.deleteRole = deleteRole;
window.handleRoleNameKeypress = handleRoleNameKeypress;

// Initialize when page loads
window.addEventListener('DOMContentLoaded', async () => {
    await initSidebar();
    await init();
});
