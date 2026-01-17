/**
 * EWR Admin Sidebar Component
 * Complete admin sidebar with brand, navigation, and footer
 * Uses Shadow DOM for style encapsulation
 *
 * @element ewr-admin-sidebar
 *
 * @example
 * <ewr-admin-sidebar></ewr-admin-sidebar>
 */

import { EwrBaseComponent } from '../base/ewr-base-component.js';

export class EwrAdminSidebar extends EwrBaseComponent {
    constructor() {
        super();
        this._navCache = null;
        this._navCacheTime = 0;
        this._NAV_CACHE_TTL = 60000;
        this._ROLE_HIERARCHY = { 'user': 0, 'developer': 1, 'admin': 2 };
    }

    async onConnected() {
        await this._initializeSidebar();
    }

    async _initializeSidebar() {
        try {
            // Get current user info
            if (typeof AuthClient === 'undefined') {
                console.warn('AuthClient not available for sidebar');
                return;
            }

            const auth = new AuthClient();
            const user = await auth.getUser();

            if (!user) {
                console.error('No user found for sidebar initialization');
                return;
            }

            // Update user info display
            this._updateUserInfo(user);

            // Fetch navigation structure from server
            const categories = await this._fetchNavigation();

            // Generate navigation items
            this._generateNavigation(user, categories);

            // Set active page
            this._setActivePage();

        } catch (error) {
            console.error('Failed to initialize admin sidebar:', error);
        }
    }

    async _fetchNavigation() {
        if (this._navCache && (Date.now() - this._navCacheTime) < this._NAV_CACHE_TTL) {
            return this._navCache;
        }

        try {
            const response = await fetch('/api/admin/navigation');
            if (!response.ok) throw new Error(`HTTP ${response.status}`);
            const data = await response.json();
            if (data.success) {
                this._navCache = data.categories;
                this._navCacheTime = Date.now();
                return this._navCache;
            }
            throw new Error(data.error || 'Failed to fetch navigation');
        } catch (error) {
            console.error('Failed to fetch navigation:', error);
            return [];
        }
    }

    _updateUserInfo(user) {
        const userNameEl = this.$('#userName');
        const userRoleEl = this.$('#userRole');

        if (userNameEl) userNameEl.textContent = user.username;
        if (userRoleEl) userRoleEl.textContent = (user.role || 'User').toUpperCase();
    }

    _getExpandedState() {
        try {
            const currentPage = window.location.pathname.split('/').pop() || 'index.html';
            if (currentPage === 'index.html') return {};
            const saved = localStorage.getItem('adminSidebarExpanded');
            return saved ? JSON.parse(saved) : {};
        } catch {
            return {};
        }
    }

    _saveExpandedState(state) {
        try {
            localStorage.setItem('adminSidebarExpanded', JSON.stringify(state));
        } catch { }
    }

    _toggleCategory(categoryId) {
        const state = this._getExpandedState();
        state[categoryId] = !state[categoryId];
        this._saveExpandedState(state);

        const pagesEl = this.$(`#category-pages-${categoryId}`);
        const chevronEl = this.$(`[data-category="${categoryId}"] .category-chevron`);

        if (pagesEl) pagesEl.classList.toggle('collapsed');
        if (chevronEl) chevronEl.classList.toggle('rotated');
    }

    _collapseAllCategories() {
        this._saveExpandedState({});
        this.$$('.category-pages').forEach(el => el.classList.add('collapsed'));
        this.$$('.category-chevron').forEach(el => el.classList.add('rotated'));
    }

    _getAdjustedUrl(pageUrl) {
        const currentPath = window.location.pathname;

        if (pageUrl.startsWith('/') || pageUrl.startsWith('http')) {
            return pageUrl;
        }

        const isOutsideAdmin = !currentPath.includes('/admin/');
        if (isOutsideAdmin && !pageUrl.startsWith('../')) {
            return '../admin/' + pageUrl;
        }

        const adminMatch = currentPath.match(/\/admin\/([^/]+)\//);
        if (adminMatch) {
            if (pageUrl.startsWith('../')) return pageUrl;
            const currentSubdir = adminMatch[1];
            const urlParts = pageUrl.split('/');
            if (urlParts.length > 1 && urlParts[0] === currentSubdir) {
                return urlParts.slice(1).join('/');
            } else if (urlParts.length > 1) {
                return '../' + pageUrl;
            } else {
                return '../' + pageUrl;
            }
        }

        return pageUrl;
    }

    _generateNavigation(user, categories) {
        const navContainer = this.$('.sidebar-nav');
        if (!navContainer) return;

        const userRoleLevel = this._ROLE_HIERARCHY[user.role] || 0;
        const disabledPages = user.settings?.disabledPages || [];
        const expandedState = this._getExpandedState();
        const inAdmin = window.location.pathname.includes('/admin/');

        const filteredCategories = categories.filter(cat => {
            const catIsAdmin = cat.isAdmin !== false;
            return inAdmin ? catIsAdmin : !catIsAdmin;
        });

        let navHTML = '';

        // Home links
        if (inAdmin) {
            navHTML += `
                <a href="/admin/index.html" class="nav-item go-to-admin">
                    <span class="nav-icon"><ewr-icon name="layout-dashboard" size="20"></ewr-icon></span>
                    <span class="nav-text">Admin Home</span>
                </a>
                <a href="/index.html" class="nav-item go-to-main">
                    <span class="nav-icon"><ewr-icon name="home" size="20"></ewr-icon></span>
                    <span class="nav-text">User Home</span>
                </a>
                <button class="collapse-all-btn" type="button">Collapse</button>
            `;
        } else {
            navHTML += `
                <a href="/index.html" class="nav-item go-to-main">
                    <span class="nav-icon"><ewr-icon name="home" size="20"></ewr-icon></span>
                    <span class="nav-text">User Home</span>
                </a>
            `;
            if (user.role === 'admin') {
                navHTML += `
                    <a href="/admin/index.html" class="nav-item go-to-admin">
                        <span class="nav-icon"><ewr-icon name="layout-dashboard" size="20"></ewr-icon></span>
                        <span class="nav-text">Admin Home</span>
                    </a>
                `;
            }
            navHTML += `<button class="collapse-all-btn" type="button">Collapse</button>`;
        }

        navHTML += '<div class="nav-divider"></div>';

        let externalLinksDividerAdded = false;

        filteredCategories.forEach(category => {
            if (category.externalLink) {
                if (!externalLinksDividerAdded) {
                    navHTML += '<div class="nav-divider"></div>';
                    externalLinksDividerAdded = true;
                }
                const target = category.openInNewTab ? '_blank' : '_self';
                const rel = category.openInNewTab ? 'rel="noopener noreferrer"' : '';
                navHTML += `
                    <a href="${category.externalLink}" target="${target}" ${rel} class="nav-item external-link">
                        <span class="nav-icon">${category.icon}</span>
                        <span class="nav-text">${category.name}</span>
                        ${category.openInNewTab ? '<ewr-icon name="external-link" size="12" class="external-icon"></ewr-icon>' : ''}
                    </a>
                `;
                return;
            }

            const roleAccessiblePages = category.pages.filter(page => {
                const pageRoleLevel = this._ROLE_HIERARCHY[page.requiredRole] || 0;
                return userRoleLevel >= pageRoleLevel;
            });

            if (roleAccessiblePages.length === 0) return;

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
                        <span class="nav-item restricted" data-page-id="${page.id}" title="Access restricted">
                            <span class="nav-icon">${category.icon}</span>
                            <span class="nav-text">${category.name}</span>
                        </span>
                    `;
                } else {
                    navHTML += `
                        <a href="${this._getAdjustedUrl(page.url)}" class="nav-item" data-page-id="${page.id}">
                            <span class="nav-icon">${category.icon}</span>
                            <span class="nav-text">${category.name}</span>
                        </a>
                    `;
                }
                return;
            }

            const isExpanded = expandedState[category.id] !== undefined
                ? expandedState[category.id]
                : category.expanded !== false;

            navHTML += `
                <div class="nav-category">
                    <div class="category-header" data-category="${category.id}">
                        <span class="category-icon">${category.icon}</span>
                        <span>${category.name}</span>
                        <ewr-icon name="chevron-down" size="18" class="category-chevron ${isExpanded ? '' : 'rotated'}"></ewr-icon>
                    </div>
                    <div id="category-pages-${category.id}" class="category-pages ${isExpanded ? '' : 'collapsed'}">
                        ${roleAccessiblePages.map(page => {
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
                                    <span class="nav-item restricted" data-page-id="${page.id}" title="Access restricted">
                                        <span class="nav-icon">${page.icon}</span>
                                        <span class="nav-text">${page.name}</span>
                                    </span>
                                `;
                            } else {
                                return `
                                    <a href="${this._getAdjustedUrl(page.url)}" class="nav-item" data-page-id="${page.id}">
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

        // Add event listeners
        this.$$('.category-header').forEach(header => {
            header.addEventListener('click', () => {
                this._toggleCategory(header.dataset.category);
            });
        });

        this.$('.collapse-all-btn')?.addEventListener('click', () => {
            this._collapseAllCategories();
        });
    }

    _setActivePage() {
        const currentPage = window.location.pathname.split('/').pop() || 'index.html';
        if (currentPage === 'index.html') return;

        this.$$('.nav-item').forEach(item => {
            const itemUrl = item.getAttribute('href');
            if (itemUrl && (itemUrl === currentPage || itemUrl.endsWith('/' + currentPage))) {
                item.classList.add('active');

                const categoryPages = item.closest('.category-pages');
                if (categoryPages && categoryPages.classList.contains('collapsed')) {
                    const categoryId = categoryPages.id.replace('category-pages-', '');
                    this._toggleCategory(categoryId);
                }
            }
        });
    }

    render() {
        return `
            <style>
                ${this.getThemeVariables()}
                ${this.getStyles()}
            </style>
            <aside class="app-sidebar" part="sidebar">
                <div class="sidebar-brand">
                    <div class="sidebar-logo">EWR</div>
                    <div class="user-info-card">
                        <div class="user-info-row">
                            <span class="user-info-label">User</span>
                            <span class="sidebar-username" id="userName">Loading...</span>
                        </div>
                        <div class="user-info-row">
                            <span class="user-info-label">Role</span>
                            <span class="sidebar-username" id="userRole">User</span>
                        </div>
                    </div>
                </div>
                <nav class="sidebar-nav"></nav>
                <div class="sidebar-footer">
                    <ewr-logout-button></ewr-logout-button>
                </div>
            </aside>
        `;
    }

    getStyles() {
        return `
            :host {
                display: block;
                width: 240px;
                height: 100vh;
                position: fixed;
                left: 0;
                top: 0;
                z-index: 1000;
            }

            .app-sidebar {
                width: 240px;
                height: 100vh;
                background: linear-gradient(180deg, #1a2332 0%, #0f1419 100%);
                display: flex;
                flex-direction: column;
                border-right: 1px solid rgba(255, 255, 255, 0.06);
                box-shadow: 4px 0 24px rgba(0, 0, 0, 0.4), 1px 0 0 rgba(255, 255, 255, 0.03);
            }

            .app-sidebar ::-webkit-scrollbar { width: 4px; }
            .app-sidebar ::-webkit-scrollbar-track { background: transparent; }
            .app-sidebar ::-webkit-scrollbar-thumb { background: rgba(255, 255, 255, 0.1); border-radius: 2px; }
            .app-sidebar ::-webkit-scrollbar-thumb:hover { background: rgba(255, 255, 255, 0.2); }

            .sidebar-brand {
                padding: 16px 18px;
                border-bottom: 1px solid rgba(148, 163, 184, 0.25);
                display: flex;
                align-items: center;
                gap: 12px;
                background: rgba(0, 0, 0, 0.2);
                flex-shrink: 0;
            }

            .sidebar-logo {
                width: 36px;
                height: 36px;
                border-radius: 8px;
                display: flex;
                align-items: center;
                justify-content: center;
                font-weight: 800;
                font-size: 13px;
                background: linear-gradient(180deg, #7dd3fc 0%, #38bdf8 15%, #0ea5e9 30%, #0284c7 50%, #0369a1 70%, #075985 85%, #0c4a6e 100%);
                color: #ffffff;
                flex-shrink: 0;
                letter-spacing: -0.5px;
                text-shadow: 0 1px 2px rgba(0, 0, 0, 0.3);
                box-shadow: 0 1px 2px rgba(0, 0, 0, 0.3), inset 0 1px 1px rgba(255, 255, 255, 0.4), inset 0 -1px 1px rgba(0, 0, 0, 0.2);
                border: 1px solid rgba(255, 255, 255, 0.15);
                position: relative;
            }

            .sidebar-logo::before {
                content: '';
                position: absolute;
                top: 0; left: 0; right: 0; bottom: 0;
                border-radius: 7px;
                background: linear-gradient(135deg, transparent 0%, rgba(255, 255, 255, 0.2) 45%, rgba(255, 255, 255, 0.35) 50%, rgba(255, 255, 255, 0.2) 55%, transparent 100%);
                pointer-events: none;
            }

            .user-info-card {
                flex: 1;
                background: rgba(255, 255, 255, 0.04);
                border: 1px solid rgba(255, 255, 255, 0.06);
                border-radius: 8px;
                padding: 8px 12px;
                min-width: 0;
            }

            .user-info-row {
                display: flex;
                align-items: center;
                gap: 8px;
            }

            .user-info-row + .user-info-row { margin-top: 4px; }

            .user-info-label {
                font-size: 9px;
                font-weight: 600;
                text-transform: uppercase;
                letter-spacing: 0.8px;
                color: #64748b;
                min-width: 28px;
            }

            .sidebar-username {
                font-size: 12px;
                font-weight: 600;
                color: #e2e8f0;
                white-space: nowrap;
                overflow: hidden;
                text-overflow: ellipsis;
                max-width: 120px;
            }

            .sidebar-nav {
                flex: 1;
                padding: 12px 0;
                overflow-y: auto;
                overflow-x: hidden;
            }

            .nav-item {
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
                cursor: pointer;
            }

            .nav-item::before {
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

            .nav-item:hover {
                background: rgba(255, 255, 255, 0.04);
                color: #e2e8f0;
            }

            .nav-item:hover::before { height: 16px; }

            .nav-item.active {
                background: rgba(0, 212, 255, 0.08);
                color: #00d4ff;
            }

            .nav-item.active::before { height: 20px; }
            .nav-item.active .nav-icon { color: #00d4ff; }

            .nav-icon {
                width: 18px;
                height: 18px;
                display: flex;
                align-items: center;
                justify-content: center;
                flex-shrink: 0;
                opacity: 0.7;
                transition: opacity 0.15s ease;
            }

            .nav-item:hover .nav-icon { opacity: 1; }

            .nav-text {
                font-size: 13px;
                font-weight: 500;
                white-space: nowrap;
                overflow: hidden;
                text-overflow: ellipsis;
            }

            .nav-divider {
                height: 1px;
                background: linear-gradient(90deg, transparent 0%, rgba(100, 116, 139, 0.4) 20%, rgba(148, 163, 184, 0.5) 50%, rgba(100, 116, 139, 0.4) 80%, transparent 100%);
                margin: 12px 16px;
                box-shadow: 0 1px 0 rgba(0, 0, 0, 0.3);
            }

            /* Category styles */
            .nav-category { margin-bottom: 2px; }

            .category-header {
                display: flex;
                align-items: center;
                padding: 9px 14px;
                cursor: pointer;
                color: #94a3b8;
                font-size: 13px;
                font-weight: 600;
                transition: all 0.15s ease;
                border-radius: 6px;
                margin: 1px 8px;
                user-select: none;
            }

            .category-header:hover {
                background: rgba(255, 255, 255, 0.04);
                color: #e2e8f0;
            }

            .category-icon {
                width: 18px;
                height: 18px;
                margin-right: 10px;
                opacity: 0.8;
            }

            .category-header:hover .category-icon { opacity: 1; }

            .category-chevron {
                margin-left: auto;
                width: 16px;
                height: 16px;
                transition: transform 0.2s cubic-bezier(0.4, 0, 0.2, 1);
                opacity: 0.5;
            }

            .category-header:hover .category-chevron { opacity: 0.8; }
            .category-chevron.rotated { transform: rotate(-90deg); }

            .category-pages {
                overflow: hidden;
                transition: max-height 0.25s cubic-bezier(0.4, 0, 0.2, 1);
                max-height: 500px;
            }

            .category-pages.collapsed { max-height: 0; }

            .category-pages .nav-item {
                padding-left: 42px;
                font-size: 13px;
            }

            /* Disabled/restricted items */
            .nav-item.disabled {
                opacity: 0.4;
                cursor: not-allowed;
                pointer-events: none;
            }

            .nav-item.disabled .nav-text::after {
                content: 'Soon';
                margin-left: 8px;
                font-size: 9px;
                padding: 2px 5px;
                background: rgba(100, 116, 139, 0.3);
                border-radius: 3px;
                color: #64748b;
                font-weight: 600;
                text-transform: uppercase;
            }

            .nav-item.restricted {
                opacity: 0.35;
                cursor: not-allowed;
                pointer-events: none;
            }

            /* External link */
            .external-icon {
                margin-left: auto;
                opacity: 0.4;
            }

            .nav-item.external-link:hover .external-icon { opacity: 0.7; }

            /* Collapse button */
            .collapse-all-btn {
                display: block;
                width: calc(100% - 32px);
                margin: 8px 16px 4px;
                padding: 6px 12px;
                font-size: 11px;
                font-weight: 600;
                color: #94a3b8;
                background: rgba(255, 255, 255, 0.03);
                border: 1px solid rgba(255, 255, 255, 0.08);
                border-radius: 5px;
                cursor: pointer;
                transition: all 0.15s ease;
            }

            .collapse-all-btn:hover {
                background: rgba(255, 255, 255, 0.06);
                border-color: rgba(255, 255, 255, 0.12);
                color: #e2e8f0;
            }

            .sidebar-footer {
                padding: 12px 14px;
                border-top: 1px solid rgba(148, 163, 184, 0.25);
                margin-top: auto;
                background: rgba(0, 0, 0, 0.15);
                flex-shrink: 0;
            }

            .sidebar-footer ewr-logout-button {
                display: block;
                width: 100%;
            }
        `;
    }
}

if (!customElements.get('ewr-admin-sidebar')) {
    customElements.define('ewr-admin-sidebar', EwrAdminSidebar);
}

export default EwrAdminSidebar;
