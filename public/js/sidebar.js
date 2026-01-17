/**
 * Shared Sidebar Component for Public Pages
 * Generates navigation dynamically from a static configuration
 */

// Navigation configuration - edit this to add/remove navigation items
// Icons from Lucide: https://lucide.dev/icons/
const PUBLIC_NAV_CONFIG = {
    main: {
        title: 'Main',
        items: [
            {
                id: 'dashboard',
                name: 'Dashboard',
                url: '/',
                icon: '<ewr-icon name="home" size="16"></ewr-icon>'
            }
        ]
    },
    services: {
        title: 'Services',
        items: [
            {
                id: 'sql-query',
                name: 'SQL Chat',
                url: '/sql',
                icon: '<ewr-icon name="database" size="16"></ewr-icon>'
            },
            {
                id: 'kb-assistant',
                name: 'Knowledge Base Assistant',
                url: '/knowledge-base/index.html',
                icon: '<ewr-icon name="book-open" size="16"></ewr-icon>'
            },
            {
                id: 'document-browser',
                name: 'Document Browser',
                url: '/knowledge-base/documents.html',
                icon: '<ewr-icon name="folder-open" size="16"></ewr-icon>'
            }
        ]
    }
};

/**
 * Get the correct URL adjusted for the current page location
 */
function getAdjustedUrl(pageUrl) {
    // If URL is absolute or external, return as-is
    if (pageUrl.startsWith('/') || pageUrl.startsWith('http')) {
        return pageUrl;
    }

    const currentPath = window.location.pathname;
    const depth = (currentPath.match(/\//g) || []).length - 1;

    // Generate relative prefix based on depth
    const prefix = '../'.repeat(depth);
    return prefix + pageUrl;
}

/**
 * Check if a nav item matches the current page
 */
function isActivePage(itemUrl) {
    const currentPath = window.location.pathname;

    // Normalize URLs for comparison
    const normalizedCurrent = currentPath.replace(/\/index\.html$/, '/').replace(/\/$/, '') || '/';
    let normalizedItem = itemUrl.replace(/\/index\.html$/, '/').replace(/\/$/, '') || '/';

    // Handle absolute vs relative paths
    if (!normalizedItem.startsWith('/')) {
        normalizedItem = '/' + normalizedItem;
    }

    return normalizedCurrent === normalizedItem;
}

/**
 * Generate the sidebar HTML
 * @param {string|null} userRole - The current user's role
 */
function generateSidebarHTML(userRole = null) {
    // Modern professional styles for public sidebar (matching admin)
    let html = `
        <style>
            /* Public sidebar - modern professional styling */
            .sidebar-nav .nav-section {
                margin-bottom: 8px;
            }

            .sidebar-nav .nav-section-title {
                padding: 8px 20px 6px;
                font-size: 10px;
                font-weight: 700;
                text-transform: uppercase;
                letter-spacing: 0.8px;
                color: #475569;
            }

            .sidebar-nav .nav-item {
                display: flex;
                align-items: center;
                gap: 10px;
                padding: 9px 16px;
                margin: 1px 8px;
                color: #94a3b8;
                text-decoration: none;
                transition: all 0.15s ease;
                position: relative;
                border-radius: 6px;
                font-size: 13px;
            }

            .sidebar-nav .nav-item::before {
                content: '';
                position: absolute;
                left: 0;
                top: 50%;
                transform: translateY(-50%);
                width: 3px;
                height: 0;
                background: linear-gradient(180deg, #00d4ff 0%, #0099cc 100%);
                border-radius: 0 2px 2px 0;
                transition: height 0.15s ease;
            }

            .sidebar-nav .nav-item:hover {
                background: rgba(255, 255, 255, 0.04);
                color: #e2e8f0;
            }

            .sidebar-nav .nav-item:hover::before {
                height: 16px;
            }

            .sidebar-nav .nav-item.active {
                background: rgba(0, 212, 255, 0.08);
                color: #00d4ff;
            }

            .sidebar-nav .nav-item.active::before {
                height: 20px;
            }

            .sidebar-nav .nav-item.active .nav-icon {
                color: #00d4ff;
            }

            .sidebar-nav .nav-icon {
                width: 16px;
                height: 16px;
                display: flex;
                align-items: center;
                justify-content: center;
                flex-shrink: 0;
                opacity: 0.7;
                transition: opacity 0.15s ease;
            }

            .sidebar-nav .nav-item:hover .nav-icon {
                opacity: 1;
            }

            .sidebar-nav .nav-text {
                font-size: 13px;
                font-weight: 500;
                white-space: nowrap;
                overflow: hidden;
                text-overflow: ellipsis;
            }

            /* Section divider between nav sections */
            .sidebar-nav .nav-section + .nav-section::before {
                content: '';
                display: block;
                height: 1px;
                background: linear-gradient(90deg,
                    transparent 0%,
                    rgba(100, 116, 139, 0.4) 20%,
                    rgba(148, 163, 184, 0.5) 50%,
                    rgba(100, 116, 139, 0.4) 80%,
                    transparent 100%
                );
                margin: 12px 16px;
                box-shadow: 0 1px 0 rgba(0, 0, 0, 0.3);
            }
        </style>
    `;

    for (const [sectionKey, section] of Object.entries(PUBLIC_NAV_CONFIG)) {
        html += `
            <div class="nav-section">
                <div class="nav-section-title">${section.title}</div>
        `;

        for (const item of section.items) {
            const isActive = isActivePage(item.url);
            const adjustedUrl = item.url.startsWith('/') ? item.url : getAdjustedUrl(item.url);

            html += `
                <a href="${adjustedUrl}" class="nav-item${isActive ? ' active' : ''}" data-nav-id="${item.id}">
                    <span class="nav-icon">${item.icon}</span>
                    <span class="nav-text">${item.name}</span>
                </a>
            `;
        }

        // Add Admin Dashboard link after Main section for admin users
        if (sectionKey === 'main' && userRole === 'admin') {
            html += `
                <a href="/admin/index.html" class="nav-item" data-nav-id="admin-dashboard">
                    <span class="nav-icon"><ewr-icon name="settings" size="16"></ewr-icon></span>
                    <span class="nav-text">Admin Dashboard</span>
                </a>
            `;
        }

        html += '</div>';
    }

    return html;
}

/**
 * Initialize the sidebar navigation
 */
async function initPublicSidebar() {
    const navContainer = document.querySelector('.sidebar-nav');
    if (!navContainer) {
        console.warn('Sidebar nav container (.sidebar-nav) not found');
        return;
    }

    // Get user role for conditional navigation
    let userRole = null;
    try {
        if (typeof AuthClient !== 'undefined') {
            const auth = new AuthClient();
            const user = await auth.getUser();
            if (user) {
                userRole = user.role;
            }
        }
    } catch (error) {
        console.error('Failed to get user role for sidebar:', error);
    }

    // Generate and insert navigation with user role
    navContainer.innerHTML = generateSidebarHTML(userRole);

    // Add footer with logout button if not already present
    const sidebar = navContainer.closest('.app-sidebar') || navContainer.parentElement;
    if (sidebar && !sidebar.querySelector('.sidebar-footer')) {
        const footer = document.createElement('div');
        footer.className = 'sidebar-footer';
        footer.innerHTML = '<ewr-logout-button></ewr-logout-button>';
        sidebar.appendChild(footer);
    }

    // Update user info if AuthClient is available
    updateSidebarUserInfo();
}

/**
 * Update sidebar user information
 */
async function updateSidebarUserInfo() {
    try {
        // Check if AuthClient is available
        if (typeof AuthClient === 'undefined') {
            console.log('AuthClient not available yet');
            return;
        }

        const auth = new AuthClient();
        const user = await auth.getUser();

        if (user) {
            const userNameEl = document.getElementById('userName');
            const userRoleEl = document.getElementById('userRole');

            if (userNameEl) userNameEl.textContent = user.username;
            if (userRoleEl) userRoleEl.textContent = user.role || 'User';
        }
    } catch (error) {
        console.error('Failed to update sidebar user info:', error);
    }
}

/**
 * Get the sidebar brand HTML (logo and user info)
 */
function getSidebarBrandHTML() {
    return `
        <div class="sidebar-brand">
            <div class="sidebar-logo">EWR</div>
            <div class="user-info">
                <div class="user-name" id="userName">Loading...</div>
                <div class="user-role" id="userRole">User</div>
            </div>
        </div>
    `;
}

/**
 * Get the sidebar footer HTML with logout button
 */
function getSidebarFooterHTML() {
    return `
        <div class="sidebar-footer">
            <ewr-logout-button></ewr-logout-button>
        </div>
    `;
}

/**
 * Get the complete sidebar HTML
 * @param {string|null} userRole - The current user's role
 */
function getCompleteSidebarHTML(userRole = null) {
    return `
        ${getSidebarBrandHTML()}
        <nav class="sidebar-nav">
            ${generateSidebarHTML(userRole)}
        </nav>
        ${getSidebarFooterHTML()}
    `;
}

// Auto-initialize when DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initPublicSidebar);
} else {
    initPublicSidebar();
}

// Export functions for use in other scripts
if (typeof window !== 'undefined') {
    window.initPublicSidebar = initPublicSidebar;
    window.updateSidebarUserInfo = updateSidebarUserInfo;
    window.getCompleteSidebarHTML = getCompleteSidebarHTML;
    window.PUBLIC_NAV_CONFIG = PUBLIC_NAV_CONFIG;
}
