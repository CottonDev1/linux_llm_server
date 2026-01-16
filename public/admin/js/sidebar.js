/**
 * Shared Sidebar Component for Admin Pages
 * Dynamically generates sidebar navigation from server-scanned directory structure
 */

// Cache for navigation data
let navCache = null;
let navCacheTime = 0;
const NAV_CACHE_TTL = 60000; // 1 minute cache

// Role hierarchy for permission checking
const ROLE_HIERARCHY = {
    'user': 0,
    'developer': 1,
    'admin': 2
};

// Dashboard is standalone (not in a category)
const DASHBOARD_PAGE = {
    id: 'dashboard',
    name: 'Dashboard',
    url: 'index.html',
    icon: '<ewr-icon name="layout-dashboard" size="20"></ewr-icon>',
    requiredRole: 'user',
    description: 'Overview and quick access'
};

// Store expanded state in localStorage
function getExpandedState() {
    try {
        const currentPage = window.location.pathname.split('/').pop() || 'index.html';
        // On admin index page, always start with all categories collapsed (ignore saved state)
        if (currentPage === 'index.html') {
            return {};
        }
        const saved = localStorage.getItem('adminSidebarExpanded');
        return saved ? JSON.parse(saved) : {};
    } catch {
        return {};
    }
}

function saveExpandedState(state) {
    try {
        localStorage.setItem('adminSidebarExpanded', JSON.stringify(state));
    } catch {
        // Ignore localStorage errors
    }
}

/**
 * Fetch navigation structure from server
 */
async function fetchNavigation() {
    // Check cache
    if (navCache && (Date.now() - navCacheTime) < NAV_CACHE_TTL) {
        return navCache;
    }

    try {
        const response = await fetch('/api/admin/navigation');
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}`);
        }
        const data = await response.json();
        if (data.success) {
            navCache = data.categories;
            navCacheTime = Date.now();
            return navCache;
        }
        throw new Error(data.error || 'Failed to fetch navigation');
    } catch (error) {
        console.error('Failed to fetch navigation:', error);
        // Return empty array on error - sidebar will show minimal navigation
        return [];
    }
}

/**
 * Initialize the sidebar with user-specific navigation
 */
async function initSidebar() {
    try {
        // Get current user info
        const auth = new AuthClient();
        const user = await auth.getUser();

        if (!user) {
            console.error('No user found for sidebar initialization');
            return;
        }

        // Update sidebar user info
        updateSidebarUserInfo(user);

        // Fetch navigation structure from server
        const categories = await fetchNavigation();

        // Generate navigation items
        generateNavigation(user, categories);

        // Set active page
        setActivePage();

    } catch (error) {
        console.error('Failed to initialize sidebar:', error);
    }
}

/**
 * Update sidebar user information display
 */
function updateSidebarUserInfo(user) {
    const usernameEl = document.getElementById('sidebarUsername');
    const userNameEl = document.getElementById('userName');
    const userAvatarEl = document.getElementById('userAvatar');
    const userRoleEl = document.getElementById('userRole');

    if (usernameEl) usernameEl.textContent = user.username;
    if (userNameEl) userNameEl.textContent = user.username;
    if (userAvatarEl) userAvatarEl.textContent = user.username.charAt(0).toUpperCase();
    if (userRoleEl) userRoleEl.textContent = (user.role || 'User').toUpperCase();
}

/**
 * Check if user has access to a page
 * Uses inverse logic: we store what's DISABLED, not what's enabled
 * By default, users can access everything their role allows
 */
function userHasAccess(user, page, disabledPages) {
    const userRoleLevel = ROLE_HIERARCHY[user.role] || 0;
    const pageRoleLevel = ROLE_HIERARCHY[page.requiredRole] || 0;
    const hasRoleAccess = userRoleLevel >= pageRoleLevel;

    if (!hasRoleAccess) return false;

    // Check if page is explicitly disabled for this user
    if (disabledPages && disabledPages.includes(page.id)) {
        return false;
    }

    return true;
}

/**
 * Toggle category expansion
 */
function toggleCategory(categoryId) {
    const state = getExpandedState();
    state[categoryId] = !state[categoryId];
    saveExpandedState(state);

    const categoryEl = document.querySelector(`[data-category="${categoryId}"]`);
    const pagesEl = document.getElementById(`category-pages-${categoryId}`);
    const chevronEl = categoryEl?.querySelector('.category-chevron');

    if (pagesEl) {
        pagesEl.classList.toggle('collapsed');
    }
    if (chevronEl) {
        chevronEl.classList.toggle('rotated');
    }
}

/**
 * Collapse all category dropdowns
 */
function collapseAllCategories() {
    // Clear expanded state
    saveExpandedState({});

    // Collapse all category pages in the DOM
    document.querySelectorAll('.category-pages').forEach(el => {
        el.classList.add('collapsed');
    });

    // Reset all chevrons
    document.querySelectorAll('.category-chevron').forEach(el => {
        el.classList.add('rotated');
    });
}

/**
 * Get the correct URL for a page, adjusting for current location
 */
function getAdjustedUrl(pageUrl) {
    const currentPath = window.location.pathname;

    // If URL is absolute or external, return as-is
    if (pageUrl.startsWith('/') || pageUrl.startsWith('http')) {
        return pageUrl;
    }

    // Check if we're outside the /admin/ directory
    const isOutsideAdmin = !currentPath.includes('/admin/');
    if (isOutsideAdmin && !pageUrl.startsWith('../')) {
        return '../admin/' + pageUrl;
    }

    // Check if we're in an admin subdirectory (e.g., /admin/sql/, /admin/users/)
    const adminMatch = currentPath.match(/\/admin\/([^/]+)\//);
    if (adminMatch) {
        // We're in a subdirectory like /admin/sql/
        // URLs like "sql/dashboard.html" need adjustment based on current location
        if (pageUrl.startsWith('../')) {
            // Already has relative path prefix, use as-is
            return pageUrl;
        }

        // Check if the URL is for a page in the same subdirectory
        const currentSubdir = adminMatch[1];
        const urlParts = pageUrl.split('/');

        if (urlParts.length > 1 && urlParts[0] === currentSubdir) {
            // Same subdirectory, just use the filename
            return urlParts.slice(1).join('/');
        } else if (urlParts.length > 1) {
            // Different subdirectory, go up one level first
            return '../' + pageUrl;
        } else {
            // Root admin file (like index.html), go up one level
            return '../' + pageUrl;
        }
    }

    return pageUrl;
}

/**
 * Check if current page is in admin area
 */
function isInAdminArea() {
    return window.location.pathname.includes('/admin/');
}

/**
 * Generate navigation items based on user permissions
 */
function generateNavigation(user, categories) {
    const navContainer = document.querySelector('.sidebar-nav');
    if (!navContainer) {
        console.error('Sidebar nav container not found');
        return;
    }

    const userRoleLevel = ROLE_HIERARCHY[user.role] || 0;
    const disabledPages = user.settings?.disabledPages || [];
    const expandedState = getExpandedState();

    // Filter categories based on current location
    // Admin area shows admin categories, non-admin area shows non-admin categories
    const inAdmin = isInAdminArea();
    const filteredCategories = categories.filter(cat => {
        // isAdmin defaults to true if not specified (backwards compatibility)
        const catIsAdmin = cat.isAdmin !== false;
        return inAdmin ? catIsAdmin : !catIsAdmin;
    });

    // Start building HTML
    let navHTML = `
        <style>
            .nav-category {
                margin-bottom: 4px;
            }
            .category-header {
                display: flex;
                align-items: center;
                padding: 10px 16px;
                cursor: pointer;
                color: var(--text-secondary, #94a3b8);
                font-size: 12px;
                font-weight: 600;
                text-transform: uppercase;
                letter-spacing: 0.5px;
                transition: all 0.2s;
                border-radius: 6px;
                margin: 0 8px;
            }
            .category-header:hover {
                background: rgba(255, 255, 255, 0.05);
                color: var(--text-primary, #f1f5f9);
            }
            .category-icon {
                width: 16px;
                height: 16px;
                margin-right: 10px;
                fill: currentColor;
            }
            .category-icon svg {
                width: 100%;
                height: 100%;
                fill: currentColor;
            }
            .category-chevron {
                margin-left: auto;
                width: 16px;
                height: 16px;
                transition: transform 0.2s;
                fill: currentColor;
            }
            .category-chevron.rotated {
                transform: rotate(-90deg);
            }
            .category-pages {
                overflow: hidden;
                transition: max-height 0.3s ease;
                max-height: 500px;
            }
            .category-pages.collapsed {
                max-height: 0;
            }
            .category-pages .nav-item {
                padding-left: 42px;
            }
            .nav-item.disabled {
                opacity: 0.5;
                cursor: not-allowed;
                pointer-events: none;
            }
            .nav-item.restricted {
                opacity: 0.4;
                cursor: not-allowed;
                pointer-events: none;
                position: relative;
            }
            .nav-item.restricted::after {
                content: '';
                position: absolute;
                right: 12px;
                top: 50%;
                transform: translateY(-50%);
                width: 14px;
                height: 14px;
                background: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='%236b7280'%3E%3Cpath d='M18 8h-1V6c0-2.76-2.24-5-5-5S7 3.24 7 6v2H6c-1.1 0-2 .9-2 2v10c0 1.1.9 2 2 2h12c1.1 0 2-.9 2-2V10c0-1.1-.9-2-2-2zm-6 9c-1.1 0-2-.9-2-2s.9-2 2-2 2 .9 2 2-.9 2-2 2zm3.1-9H8.9V6c0-1.71 1.39-3.1 3.1-3.1 1.71 0 3.1 1.39 3.1 3.1v2z'/%3E%3C/svg%3E") no-repeat center;
                background-size: contain;
            }
            .nav-divider {
                height: 1px;
                background: var(--card-border, #1e293b);
                margin: 12px 16px;
            }
            .nav-section-header {
                padding: 10px 16px 6px;
            }
            .nav-section-title {
                font-size: 11px;
                font-weight: 600;
                text-transform: uppercase;
                letter-spacing: 0.5px;
                color: var(--text-muted, #64748b);
            }
            /* Ensure Services section items use normal text color, not link highlight color */
            .sidebar-nav .nav-item {
                color: var(--text-secondary, #94a3b8);
                text-decoration: none;
            }
            .sidebar-nav .nav-item:hover {
                color: var(--text-primary, #f1f5f9);
                background: rgba(255, 255, 255, 0.05);
            }
            .sidebar-nav .nav-item.active {
                color: var(--accent-color, #22d3ee);
                background: rgba(34, 211, 238, 0.1);
            }
            .collapse-all-btn {
                display: block;
                width: calc(100% - 32px);
                margin: 6px 16px 0;
                padding: 6px 14px;
                font-size: 13px;
                font-weight: 600;
                color: #1e293b;
                background: linear-gradient(180deg, #e2e8f0 0%, #cbd5e1 25%, #94a3b8 75%, #64748b 100%);
                border: 1px solid #94a3b8;
                border-radius: 5px;
                cursor: pointer;
                transition: all 0.2s ease;
                box-shadow:
                    0 1px 2px rgba(0, 0, 0, 0.1),
                    inset 0 1px 0 rgba(255, 255, 255, 0.4);
            }
            .collapse-all-btn:hover {
                background: linear-gradient(180deg, #f1f5f9 0%, #e2e8f0 25%, #cbd5e1 75%, #94a3b8 100%);
                transform: translateY(-1px);
                box-shadow:
                    0 3px 8px rgba(0, 0, 0, 0.15),
                    inset 0 1px 0 rgba(255, 255, 255, 0.5);
                border-color: #64748b;
            }
            .collapse-all-btn:active {
                background: linear-gradient(180deg, #94a3b8 0%, #64748b 50%, #475569 100%);
                transform: translateY(0);
                box-shadow: inset 0 2px 3px rgba(0, 0, 0, 0.2);
                color: #f1f5f9;
            }
        </style>
    `;

    // Show context-appropriate home links
    if (inAdmin) {
        // In admin area: show Admin Home and link to User Home
        navHTML += `
            <a href="/admin/index.html" class="nav-item go-to-admin">
                <span class="nav-icon">
                    <ewr-icon name="layout-dashboard" size="20"></ewr-icon>
                </span>
                <span class="nav-text">Admin Home</span>
            </a>
            <a href="/index.html" class="nav-item go-to-main">
                <span class="nav-icon">
                    <ewr-icon name="home" size="20"></ewr-icon>
                </span>
                <span class="nav-text">User Home</span>
            </a>
            <button class="collapse-all-btn" onclick="collapseAllCategories()">Collapse</button>
        `;
    } else {
        // In non-admin area: show User Home and link to Admin Home (for admins)
        navHTML += `
            <a href="/index.html" class="nav-item go-to-main">
                <span class="nav-icon">
                    <ewr-icon name="home" size="20"></ewr-icon>
                </span>
                <span class="nav-text">User Home</span>
            </a>
        `;
        // Only show Admin Home link if user is admin
        if (user.role === 'admin') {
            navHTML += `
                <a href="/admin/index.html" class="nav-item go-to-admin">
                    <span class="nav-icon">
                        <ewr-icon name="layout-dashboard" size="20"></ewr-icon>
                    </span>
                    <span class="nav-text">Admin Home</span>
                </a>
            `;
        }
        navHTML += `<button class="collapse-all-btn" onclick="collapseAllCategories()">Collapse</button>`;
    }

    navHTML += '<div class="nav-divider"></div>';

    // Generate categories from server data (filtered by location)
    filteredCategories.forEach(category => {
        // Handle external links (like Prefect)
        if (category.externalLink) {
            const target = category.openInNewTab ? '_blank' : '_self';
            const rel = category.openInNewTab ? 'rel="noopener noreferrer"' : '';
            navHTML += `
                <a href="${category.externalLink}" target="${target}" ${rel} class="nav-item external-link">
                    <span class="nav-icon">${category.icon}</span>
                    <span class="nav-text">${category.name}</span>
                    ${category.openInNewTab ? '<ewr-icon name="external-link" size="12" class="external-icon" style="margin-left: auto; opacity: 0.5;"></ewr-icon>' : ''}
                </a>
            `;
            return;
        }

        // Filter pages user has role access to (not page restrictions)
        const roleAccessiblePages = category.pages.filter(page => {
            const pageRoleLevel = ROLE_HIERARCHY[page.requiredRole] || 0;
            return userRoleLevel >= pageRoleLevel;
        });

        // Skip category if user has no role-accessible pages
        if (roleAccessiblePages.length === 0) return;

        // If only one page, render as direct link instead of dropdown
        if (roleAccessiblePages.length === 1) {
            const page = roleAccessiblePages[0];
            const isRestricted = disabledPages.includes(page.id);

            if (page.disabled) {
                navHTML += `
                    <span class="nav-item disabled" data-page-id="${page.id}" title="Coming soon">
                        <span class="nav-icon">${category.icon}</span>
                        <span class="nav-text">${category.name}</span>
                    </span>
                `;
            } else if (isRestricted) {
                navHTML += `
                    <span class="nav-item restricted" data-page-id="${page.id}" title="Access restricted by administrator">
                        <span class="nav-icon">${category.icon}</span>
                        <span class="nav-text">${category.name}</span>
                    </span>
                `;
            } else {
                navHTML += `
                    <a href="${getAdjustedUrl(page.url)}" class="nav-item" data-page-id="${page.id}">
                        <span class="nav-icon">${category.icon}</span>
                        <span class="nav-text">${category.name}</span>
                    </a>
                `;
            }
            return;
        }

        // Check if category should be expanded (default to category.expanded setting)
        const isExpanded = expandedState[category.id] !== undefined
            ? expandedState[category.id]
            : category.expanded !== false;

        navHTML += `
            <div class="nav-category">
                <div class="category-header" data-category="${category.id}" onclick="toggleCategory('${category.id}')">
                    <span class="category-icon">${category.icon}</span>
                    <span>${category.name}</span>
                    <ewr-icon name="chevron-down" size="18" class="category-chevron ${isExpanded ? '' : 'rotated'}"></ewr-icon>
                </div>
                <div id="category-pages-${category.id}" class="category-pages ${isExpanded ? '' : 'collapsed'}">
                    ${roleAccessiblePages.map(page => {
                        // Check if page is explicitly disabled for this user (inverse logic)
                        const isRestricted = disabledPages.includes(page.id);

                        if (page.disabled) {
                            return `
                                <span class="nav-item disabled" data-page-id="${page.id}" title="Coming soon">
                                    <span class="nav-icon">${page.icon}</span>
                                    <span class="nav-text">${page.name}</span>
                                </span>
                            `;
                        } else if (isRestricted) {
                            return `
                                <span class="nav-item restricted" data-page-id="${page.id}" title="Access restricted by administrator">
                                    <span class="nav-icon">${page.icon}</span>
                                    <span class="nav-text">${page.name}</span>
                                </span>
                            `;
                        } else {
                            return `
                                <a href="${getAdjustedUrl(page.url)}" class="nav-item" data-page-id="${page.id}">
                                    <span class="nav-icon">${page.icon}</span>
                                    <span class="nav-text">${page.name}</span>
                                </a>
                            `;
                        }
                    }).join('')}
                </div>
            </div>
        `;
    });

    navContainer.innerHTML = navHTML;

    // Add footer with logout button if not already present
    const sidebar = navContainer.closest('.app-sidebar') || navContainer.parentElement;
    if (sidebar && !sidebar.querySelector('.sidebar-footer')) {
        const footer = document.createElement('div');
        footer.className = 'sidebar-footer';
        footer.innerHTML = '<ewr-logout-button></ewr-logout-button>';
        sidebar.appendChild(footer);
    }
}

/**
 * Set the active page in the navigation
 */
function setActivePage() {
    const currentPage = window.location.pathname.split('/').pop() || 'index.html';

    // Don't highlight anything on the admin index/dashboard page
    if (currentPage === 'index.html') {
        return;
    }

    const navItems = document.querySelectorAll('.nav-item');

    navItems.forEach(item => {
        const itemUrl = item.getAttribute('href');
        if (itemUrl && (itemUrl === currentPage || itemUrl.endsWith('/' + currentPage))) {
            item.classList.add('active');

            // Expand parent category if page is in a collapsed category
            const categoryPages = item.closest('.category-pages');
            if (categoryPages && categoryPages.classList.contains('collapsed')) {
                const categoryId = categoryPages.id.replace('category-pages-', '');
                toggleCategory(categoryId);
            }
        }
    });
}

/**
 * Get list of all pages for admin configuration
 */
async function getAllPages() {
    const categories = await fetchNavigation();
    const allPages = [DASHBOARD_PAGE];
    categories.forEach(cat => {
        allPages.push(...cat.pages);
    });
    return allPages;
}

/**
 * Check if user has access to a specific page
 * Uses inverse logic: disabledPages stores what's blocked
 */
async function hasPageAccess(user, pageId) {
    const allPages = await getAllPages();
    const page = allPages.find(p => p.id === pageId);
    if (!page) return false;

    const userRoleLevel = ROLE_HIERARCHY[user.role] || 0;
    const pageRoleLevel = ROLE_HIERARCHY[page.requiredRole] || 0;
    const hasRoleAccess = userRoleLevel >= pageRoleLevel;

    if (!hasRoleAccess) return false;

    // Check if page is explicitly disabled for this user
    const disabledPages = user.settings?.disabledPages || [];
    if (disabledPages.includes(pageId)) {
        return false;
    }

    return true;
}


// Export functions for use in other scripts
if (typeof window !== 'undefined') {
    window.initSidebar = initSidebar;
    window.getAllPages = getAllPages;
    window.hasPageAccess = hasPageAccess;
    window.toggleCategory = toggleCategory;
}
