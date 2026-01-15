/**
 * User Management JavaScript
 * Handles CRUD operations for users and page permissions
 */

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

// Load users
async function loadUsers() {
    const tbody = document.getElementById('usersTableBody');
    tbody.innerHTML = '<tr><td colspan="5" style="padding: 20px; text-align: center; color: #6b7280;">Loading users...</td></tr>';

    try {
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
        <tr>
            <td style="color: #f1f5f9;">${user.username}</td>
            <td style="color: #cbd5e1;">${user.email || 'N/A'}</td>
            <td>
                <span class="badge badge-${user.role}">
                    ${user.role.toUpperCase()}
                </span>
            </td>
            <td>
                <span class="badge badge-${user.isActive ? 'active' : 'inactive'}">
                    ${user.isActive ? 'ACTIVE' : 'INACTIVE'}
                </span>
            </td>
            <td>
                <button class="btn-small btn-view" onclick="editUser('${user.id}')" style="margin-right: 8px;">Edit</button>
                <button class="btn-small btn-delete" onclick="deleteUser('${user.id}', '${user.username}')">Delete</button>
            </td>
        </tr>
    `).join('');
}

// Open add user modal
function openAddUserModal() {
    document.getElementById('addUserModal').classList.remove('hidden');
    document.getElementById('addUserForm').reset();
    document.getElementById('addUserError').classList.add('hidden');
    document.getElementById('enablePageRestrictions').checked = false;
    togglePageRestrictions('add');
    populatePagePermissions('add');
}

// Close add user modal
function closeAddUserModal(event) {
    if (!event || event.target.id === 'addUserModal' || event.type === 'click') {
        document.getElementById('addUserModal').classList.add('hidden');
    }
}

// Navigate to edit user page
function editUser(userId) {
    window.location.href = `edit.html?id=${userId}`;
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

// Toggle page restrictions visibility
function togglePageRestrictions(mode) {
    const checkbox = document.getElementById(`${mode}EnablePageRestrictions`);
    const permissionsDiv = document.getElementById(`${mode}PagePermissions`);

    if (checkbox.checked) {
        permissionsDiv.style.display = 'block';
    } else {
        permissionsDiv.style.display = 'none';
    }
}

// Populate page permissions checkboxes
// Uses inverse logic: checked = allowed (default), unchecked = disabled
// disabledPages array contains pages that are UNCHECKED
function populatePagePermissions(mode, disabledPages = []) {
    const container = document.getElementById(`${mode}PagePermissionsList`);
    const allPages = getAllPages();

    container.innerHTML = allPages.map(page => `
        <div class="permission-item">
            <input type="checkbox"
                   id="${mode}_page_${page.id}"
                   value="${page.id}"
                   ${disabledPages.includes(page.id) ? '' : 'checked'}>
            <label for="${mode}_page_${page.id}">${page.name}</label>
        </div>
    `).join('');
}

// Get disabled page permissions (inverse logic)
// Returns array of page IDs that are UNCHECKED (disabled)
function getDisabledPagePermissions(mode) {
    const container = document.getElementById(`${mode}PagePermissionsList`);
    const uncheckedBoxes = container.querySelectorAll('input[type="checkbox"]:not(:checked)');
    return Array.from(uncheckedBoxes).map(cb => cb.value);
}

// Legacy function - kept for compatibility
function getSelectedPagePermissions(mode) {
    const container = document.getElementById(`${mode}PagePermissionsList`);
    const checkboxes = container.querySelectorAll('input[type="checkbox"]:checked');
    return Array.from(checkboxes).map(cb => cb.value);
}

// Handle add user form submission
document.getElementById('addUserForm').addEventListener('submit', async (e) => {
    e.preventDefault();

    const username = document.getElementById('newUsername').value;
    const password = document.getElementById('newPassword').value;
    const email = document.getElementById('newEmail').value;
    const role = document.getElementById('newRole').value;
    const enableRestrictions = document.getElementById('enablePageRestrictions').checked;

    const userData = {
        username,
        password,
        email: email || undefined,
        role
    };

    // Add page permissions if enabled (inverse logic - store disabled pages)
    if (enableRestrictions) {
        const disabledPages = getDisabledPagePermissions('add');
        // It's valid to have all pages enabled (disabledPages = [])
        userData.settings = { disabledPages };
    }

    try {
        if (!auth.isAuthenticated()) {
            throw new Error('Not authenticated. Please log in again.');
        }

        const response = await fetch('/api/auth/users', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${auth.getAccessToken()}`
            },
            body: JSON.stringify(userData)
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

        if (error.message.includes('Invalid token') || error.message.includes('401') || error.message.includes('403')) {
            errorDiv.textContent = 'Authentication error. Please log in again.';
            setTimeout(() => {
                window.location.href = '/login.html?redirect=' + encodeURIComponent(window.location.pathname);
            }, 2000);
        }
    }
});
