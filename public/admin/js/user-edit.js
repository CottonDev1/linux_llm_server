/**
 * User Edit Page JavaScript
 * Handles loading and updating a single user's information and permissions
 */

// Initialize auth client
const auth = new AuthClient();

// Store current user info (the one editing)
let currentUser = null;
let editingUser = null;
let rolePermissions = null; // Role permissions from API
let roleCategories = []; // Categories assigned to user's role
let userPermissions = []; // User's enabled pages (from user_permissions table) - DEPRECATED
let userDisabledPages = []; // Pages explicitly disabled for user (inverse logic)

// Check authentication on page load
if (!auth.isAuthenticated()) {
    window.location.href = '/login.html?redirect=' + encodeURIComponent(window.location.pathname);
} else {
    // Check if user is admin
    auth.getUser().then(user => {
        currentUser = user;
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
 * Get user ID from URL parameters
 */
function getUserIdFromUrl() {
    const urlParams = new URLSearchParams(window.location.search);
    return urlParams.get('id');
}

/**
 * Load role categories for a specific role
 */
async function loadRoleCategories(role) {
    try {
        const response = await fetch(`/api/auth/role-categories/${role}`, {
            headers: {
                'Authorization': `Bearer ${auth.getAccessToken()}`
            }
        });

        if (!response.ok) {
            console.warn('Failed to load role categories, using defaults');
            return [];
        }

        const data = await response.json();
        return data.categories || [];
    } catch (error) {
        console.error('Error loading role categories:', error);
        return [];
    }
}

/**
 * Load user permissions (enabled pages)
 */
async function loadUserPermissions(userId) {
    try {
        const response = await fetch(`/api/auth/user-permissions/${userId}`, {
            headers: {
                'Authorization': `Bearer ${auth.getAccessToken()}`
            }
        });

        if (!response.ok) {
            console.warn('Failed to load user permissions, defaulting to all enabled');
            return [];
        }

        const data = await response.json();
        return data.permissions || [];
    } catch (error) {
        console.error('Error loading user permissions:', error);
        return [];
    }
}

/**
 * Load user data from API
 */
async function loadUserData() {
    const userId = getUserIdFromUrl();

    if (!userId) {
        alert('No user ID specified');
        window.location.href = 'users.html';
        return;
    }

    try {
        if (!auth.isAuthenticated()) {
            throw new Error('Not authenticated');
        }

        // Load user data first
        const userResponse = await fetch(`/api/auth/users/${userId}`, {
            headers: {
                'Authorization': `Bearer ${auth.getAccessToken()}`
            }
        });

        if (!userResponse.ok) {
            const errorData = await userResponse.json().catch(() => ({ error: 'Failed to load user' }));
            throw new Error(errorData.error || `HTTP ${userResponse.status}: ${userResponse.statusText}`);
        }

        const user = await userResponse.json();
        console.log('Loaded user data:', JSON.stringify(user, null, 2));
        editingUser = user;

        // Load role categories and user permissions in parallel
        const [categoriesResult, permissionsResult] = await Promise.all([
            loadRoleCategories(user.role),
            loadUserPermissions(userId)
        ]);

        roleCategories = categoriesResult;
        userPermissions = permissionsResult;

        // Get disabledPages from user settings (inverse logic)
        userDisabledPages = user.settings?.disabledPages || [];

        populateForm(user);
        populatePagePermissions(user);

        // Hide loading overlay
        const loadingOverlay = document.getElementById('loadingOverlay');
        if (loadingOverlay) loadingOverlay.style.display = 'none';

    } catch (error) {
        console.error('Load user error:', error);
        alert(`Failed to load user: ${error.message}`);

        if (error.message.includes('Not authenticated') || error.message.includes('401') || error.message.includes('403')) {
            window.location.href = '/login.html?redirect=' + encodeURIComponent(window.location.pathname);
        } else {
            window.location.href = 'users.html';
        }
    }
}

/**
 * Populate form with user data
 */
function populateForm(user) {
    // Update header subtitle
    const headerSubtitle = document.getElementById('headerSubtitle');
    if (headerSubtitle) {
        headerSubtitle.textContent = `Editing user: ${user.username}`;
    }

    // Populate form fields with null checks
    const usernameInput = document.getElementById('username');
    const emailInput = document.getElementById('email');
    const roleSelect = document.getElementById('role');
    const isActiveCheckbox = document.getElementById('isActive');
    const monitorAnalysisCheckbox = document.getElementById('monitorAnalysis');
    const forcePasswordResetCheckbox = document.getElementById('forcePasswordReset');

    if (usernameInput) usernameInput.value = user.username || '';
    if (emailInput) emailInput.value = user.email || '';
    if (roleSelect) roleSelect.value = user.role || 'user';
    if (isActiveCheckbox) isActiveCheckbox.checked = user.isActive !== false;
    if (monitorAnalysisCheckbox) monitorAnalysisCheckbox.checked = user.settings?.monitorAnalysis === true;
    if (forcePasswordResetCheckbox) forcePasswordResetCheckbox.checked = user.forcePasswordReset === true;

    // Set up role change protection for admins editing themselves
    const adminWarning = document.getElementById('adminProtectionWarning');

    if (roleSelect) roleSelect.addEventListener('change', function() {
        if (currentUser && editingUser && currentUser.id === editingUser.id) {
            // Admin editing themselves
            if (this.value !== 'admin') {
                // Prevent changing away from admin
                this.value = 'admin';
                if (adminWarning) adminWarning.style.display = 'block';
            } else {
                if (adminWarning) adminWarning.style.display = 'none';
            }
        }
    });
}

/**
 * Populate page permissions grouped by category
 * Shows ONLY categories assigned to user's role, with per-page checkboxes
 */
function populatePagePermissions(user) {
    const permissionsGrid = document.getElementById('permissionsGrid');

    // Safety check - if element doesn't exist, skip
    if (!permissionsGrid) {
        console.warn('Page permissions element not found');
        return;
    }

    // Get all pages organized by category from sidebar.js
    const NAV_CATEGORIES_LOCAL = getNAVCategories();

    // Build permissions grid grouped by category
    // Only show categories that are assigned to the user's role
    let html = '';

    // Add info text about the permission system
    html += `
        <div class="permissions-info">
            <p>This user inherits access to the following categories from their <strong>${user.role}</strong> role.
            Uncheck pages to restrict access.</p>
        </div>
    `;

    // Filter categories to only those assigned to the user's role
    const assignedCategories = NAV_CATEGORIES_LOCAL.filter(category => {
        return roleCategories.includes(category.id);
    });

    if (assignedCategories.length === 0) {
        html += `
            <div class="permissions-info">
                <p>No categories are assigned to the <strong>${user.role}</strong> role.</p>
            </div>
        `;
        permissionsGrid.innerHTML = html;
        return;
    }

    assignedCategories.forEach(category => {
        html += `
            <div class="permission-category" data-category-id="${category.id}">
                <div class="permission-category-header">
                    <div class="permission-category-title">
                        <span class="permission-category-icon">${category.icon}</span>
                        <span>${category.name}</span>
                    </div>
                    <div class="category-controls">
                        <button type="button" class="btn-cat-all" onclick="selectCategoryPages('${category.id}', true)">All</button>
                        <button type="button" class="btn-cat-none" onclick="selectCategoryPages('${category.id}', false)">None</button>
                    </div>
                </div>
                <div class="permission-items">
                    ${category.pages.map(page => {
                        // Inverse logic: checked by default, unchecked if in disabledPages
                        const isChecked = !userDisabledPages.includes(page.id);
                        return `
                            <div class="permission-item" onclick="togglePermission('${page.id}')">
                                <input type="checkbox"
                                       id="page_${page.id}"
                                       value="${page.id}"
                                       data-category="${category.id}"
                                       ${isChecked ? 'checked' : ''}
                                       onclick="event.stopPropagation()">
                                <label for="page_${page.id}">${page.name}</label>
                            </div>
                        `;
                    }).join('')}
                </div>
            </div>
        `;
    });

    permissionsGrid.innerHTML = html;
}

/**
 * Select/deselect all pages in a category
 */
function selectCategoryPages(categoryId, selected) {
    const checkboxes = document.querySelectorAll(
        `.permission-category[data-category-id="${categoryId}"] input[type="checkbox"]`
    );
    checkboxes.forEach(cb => cb.checked = selected);
}

/**
 * Get NAV_CATEGORIES from sidebar.js
 * We need to access the NAV_CATEGORIES constant defined in sidebar.js
 */
function getNAVCategories() {
    // NAV_CATEGORIES is defined globally in sidebar.js
    if (typeof NAV_CATEGORIES !== 'undefined') {
        return NAV_CATEGORIES;
    }

    // Fallback: return empty array if not found
    console.error('NAV_CATEGORIES not found in sidebar.js');
    return [];
}

/**
 * Toggle permission checkbox when clicking on the container
 */
function togglePermission(pageId) {
    const checkbox = document.getElementById(`page_${pageId}`);
    if (checkbox) {
        checkbox.checked = !checkbox.checked;
    }
}

/**
 * Get selected pages as array of {category, pageId} objects
 * DEPRECATED - use getDisabledPages instead
 */
function getSelectedPages() {
    const checkboxes = document.querySelectorAll('.permission-item input[type="checkbox"]:checked');
    return Array.from(checkboxes).map(cb => ({
        category: cb.dataset.category,
        pageId: cb.value
    }));
}

/**
 * Get disabled pages (unchecked) as array of page IDs
 * Inverse logic: we store what's blocked, not what's allowed
 */
function getDisabledPages() {
    const uncheckedBoxes = document.querySelectorAll('.permission-item input[type="checkbox"]:not(:checked)');
    return Array.from(uncheckedBoxes).map(cb => cb.value);
}

/**
 * Show message to user
 */
function showMessage(message, type = 'success') {
    const messageDiv = document.getElementById('userInfoMessage');
    if (!messageDiv) {
        console.warn('userInfoMessage element not found');
        return;
    }
    messageDiv.className = `message ${type}`;
    messageDiv.textContent = message;
    messageDiv.style.display = 'block';

    setTimeout(() => {
        messageDiv.style.display = 'none';
    }, 5000);
}

/**
 * Handle form submission
 */
function initFormHandler() {
    const form = document.getElementById('userInfoForm');
    if (!form) {
        console.error('userInfoForm not found');
        return;
    }
    form.addEventListener('submit', async (e) => {
        e.preventDefault();

        const userId = getUserIdFromUrl();
        if (!userId) {
            alert('No user ID specified');
            return;
        }

        // Get form values
        const email = document.getElementById('email').value;
        const role = document.getElementById('role').value;
        const password = document.getElementById('password').value;
        const isActive = document.getElementById('isActive').checked;
        const monitorAnalysis = document.getElementById('monitorAnalysis')?.checked || false;
        const forcePasswordReset = document.getElementById('forcePasswordReset')?.checked || false;

        // Build update data for user info
        const updateData = {
            email: email || undefined,
            role,
            isActive,
            forcePasswordReset
        };

        // Add password if provided
        if (password) {
            updateData.password = password;
        }

        // Get disabled pages (inverse logic - store what's blocked)
        const disabledPages = getDisabledPages();

        // Add disabledPages and monitorAnalysis to user settings
        updateData.settings = { disabledPages, monitorAnalysis };

        try {
            if (!auth.isAuthenticated()) {
                throw new Error('Not authenticated. Please log in again.');
            }

            // Update user info (including settings.disabledPages)
            const userResponse = await fetch(`/api/auth/users/${userId}`, {
                method: 'PUT',
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': `Bearer ${auth.getAccessToken()}`
                },
                body: JSON.stringify(updateData)
            });

            if (!userResponse.ok) {
                const error = await userResponse.json().catch(() => ({ error: 'Failed to update user' }));
                throw new Error(error.error || `HTTP ${userResponse.status}: ${userResponse.statusText}`);
            }

            showMessage('User updated successfully', 'success');

            // Redirect back to users list
            setTimeout(() => {
                window.location.href = '/admin/users/index.html';
            }, 1500);

        } catch (error) {
            console.error('Update user error:', error);
            showMessage(`Failed to update user: ${error.message}`, 'error');

            if (error.message.includes('Invalid token') || error.message.includes('401') || error.message.includes('403')) {
                setTimeout(() => {
                    window.location.href = '/login.html?redirect=' + encodeURIComponent(window.location.pathname);
                }, 2000);
            }
        }
    });
}

// Initialize form handler when DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initFormHandler);
} else {
    initFormHandler();
}

// Make functions globally available
window.loadUserData = loadUserData;
window.togglePermission = togglePermission;
window.selectCategoryPages = selectCategoryPages;
