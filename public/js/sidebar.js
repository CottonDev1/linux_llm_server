/**
 * Shared Sidebar Component for Public Pages
 * Generates navigation dynamically from a static configuration
 */

// Navigation configuration - edit this to add/remove navigation items
const PUBLIC_NAV_CONFIG = {
    main: {
        title: 'Main',
        items: [
            {
                id: 'dashboard',
                name: 'Dashboard',
                url: '/',
                icon: '<ewr-icon name="layout-dashboard" size="20"></ewr-icon>'
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
                icon: '<ewr-icon name="database" size="20"></ewr-icon>'
            },
            {
                id: 'kb-assistant',
                name: 'Knowledge Base Assistant',
                url: '/knowledge-base/index.html',
                icon: '<ewr-icon name="message-circle" size="20"></ewr-icon>'
            },
            { id: 'document-browser',
                name: 'Document Browser',
                url: '/knowledge-base/documents.html',
                icon: '<ewr-icon name="folder" size="20"></ewr-icon>'
            }
        ]
    },
    system: {
        title: 'System',
        items: [
            {
                id: 'admin',
                name: 'Admin',
                url: '/admin/index.html',
                icon: '<ewr-icon name="settings" size="20"></ewr-icon>'
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
 */
function generateSidebarHTML() {
    let html = '';

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

        html += '</div>';
    }

    return html;
}

/**
 * Initialize the sidebar navigation
 */
function initPublicSidebar() {
    const navContainer = document.querySelector('.sidebar-nav');
    if (!navContainer) {
        console.warn('Sidebar nav container (.sidebar-nav) not found');
        return;
    }

    // Generate and insert navigation
    navContainer.innerHTML = generateSidebarHTML();

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
 */
function getCompleteSidebarHTML() {
    return `
        ${getSidebarBrandHTML()}
        <nav class="sidebar-nav">
            ${generateSidebarHTML()}
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
