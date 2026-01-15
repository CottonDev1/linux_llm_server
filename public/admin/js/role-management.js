/**
 * Role Management JavaScript - Drag & Drop Edition
 * Manages category assignments to roles via drag-and-drop interface
 */

// Initialize auth client
const auth = new AuthClient();

// Store role-category mappings
let roleCategoryMappings = {
    user: [],
    developer: [],
    admin: []
};

// Store categories fetched from navigation API
let navCategories = [];

let hasUnsavedChanges = false;

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
 * Role definitions with descriptions
 */
const ROLE_DEFINITIONS = {
    user: {
        name: 'User',
        description: 'Basic access for standard users',
        inherits: null
    },
    developer: {
        name: 'Developer',
        description: 'Extended access including development tools',
        inherits: 'user'
    },
    admin: {
        name: 'Admin',
        description: 'Full access to all system features',
        inherits: 'developer'
    }
};

/**
 * Load categories from navigation API
 */
async function loadCategoriesFromNav() {
    try {
        const response = await fetch('/api/admin/navigation');
        if (!response.ok) {
            throw new Error('Failed to fetch navigation');
        }
        const data = await response.json();
        if (data.success && data.categories) {
            navCategories = data.categories.map(cat => ({
                id: cat.id,
                name: cat.name,
                icon: cat.icon || '<ewr-icon name="folder" size="20"></ewr-icon>'
            }));
        }
    } catch (error) {
        console.error('Error loading categories:', error);
        navCategories = [];
    }
}

/**
 * Get all categories
 */
function getAllCategories() {
    return navCategories;
}

/**
 * Initialize the page
 */
async function init() {
    await loadCategoriesFromNav();
    await loadRoleCategoryMappings();
    renderBadgePool();
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
        // API returns { user: [], developer: [], admin: [...] } directly
        roleCategoryMappings = data || getDefaultMappings();

    } catch (error) {
        console.error('Error loading role categories:', error);
        // Use defaults if API fails
        roleCategoryMappings = getDefaultMappings();
    }
}

/**
 * Get default role-category mappings
 * Admin gets all categories by default
 */
function getDefaultMappings() {
    const allCategories = getAllCategories();
    return {
        user: [],
        developer: [],
        admin: allCategories.map(cat => cat.id)
    };
}

/**
 * Render the category badge pool at the top
 */
function renderBadgePool() {
    const pool = document.getElementById('badgePool');
    const categories = getAllCategories();

    let html = '';
    categories.forEach(cat => {
        html += `
            <div class="category-badge ${cat.id}"
                 draggable="true"
                 data-category="${cat.id}"
                 ondragstart="handleDragStart(event)"
                 ondragend="handleDragEnd(event)">
                ${cat.icon}
                <span>${cat.name}</span>
            </div>
        `;
    });

    pool.innerHTML = html;
}

/**
 * Render the role cards
 */
function renderRoleCards() {
    const rolesGrid = document.getElementById('rolesGrid');
    const categories = getAllCategories();

    let html = '';

    ['user', 'developer', 'admin'].forEach(role => {
        const roleDef = ROLE_DEFINITIONS[role];
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
                        <div class="role-category-badge ${categoryId}">
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

    rolesGrid.innerHTML = html;
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
    if (roleCategoryMappings[role].includes(categoryId)) {
        showMessage(`${categoryId} is already assigned to ${role}`, 'error');
        return;
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
 * Save all role categories (manual save button)
 */
async function saveAllRoles() {
    updateSaveIndicator('saving');

    try {
        const promises = [];

        for (const [role, categories] of Object.entries(roleCategoryMappings)) {
            promises.push(
                fetch(`/api/auth/role-categories/${role}`, {
                    method: 'PUT',
                    headers: {
                        'Content-Type': 'application/json',
                        'Authorization': `Bearer ${auth.getAccessToken()}`
                    },
                    body: JSON.stringify({ categories })
                })
            );
        }

        const results = await Promise.all(promises);

        const allSuccessful = results.every(r => r.ok);

        if (!allSuccessful) {
            throw new Error('Some role categories failed to save');
        }

        updateSaveIndicator('saved');
        showMessage('All role categories saved successfully', 'success');

    } catch (error) {
        console.error('Error saving all roles:', error);
        updateSaveIndicator('error');
        showMessage('Failed to save all roles: ' + error.message, 'error');
    }
}

/**
 * Update save indicator
 */
function updateSaveIndicator(state) {
    const indicator = document.getElementById('saveIndicator');

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
                }
            }, 3000);
            break;
        case 'error':
            indicator.className = 'save-indicator';
            indicator.style.color = '#ef4444';
            indicator.textContent = 'Save failed';
            setTimeout(() => {
                indicator.style.color = '';
            }, 3000);
            break;
        default:
            indicator.textContent = '';
    }
}

/**
 * Show message to user
 */
function showMessage(message, type = 'success') {
    const messageDiv = document.getElementById('roleMessage');
    messageDiv.className = `message ${type}`;
    messageDiv.textContent = message;
    messageDiv.style.display = 'block';

    setTimeout(() => {
        messageDiv.style.display = 'none';
    }, 5000);
}

// Make functions available globally
window.init = init;
window.saveAllRoles = saveAllRoles;
window.handleDragStart = handleDragStart;
window.handleDragEnd = handleDragEnd;
window.handleDragOver = handleDragOver;
window.handleDragLeave = handleDragLeave;
window.handleDrop = handleDrop;
window.removeCategoryFromRole = removeCategoryFromRole;

// Initialize when page loads
window.addEventListener('DOMContentLoaded', async () => {
    await initSidebar();
    await init();
});
