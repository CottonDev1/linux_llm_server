/**
 * EWR Layout Components
 * Master layout templates that eliminate boilerplate from individual pages
 *
 * Usage:
 * <ewr-admin-page title="Page Title" subtitle="Optional subtitle">
 *     <!-- Your page content here -->
 * </ewr-admin-page>
 *
 * <ewr-page title="Page Title">
 *     <!-- Your page content here -->
 * </ewr-page>
 */

/**
 * EWR Admin Page Layout
 * Complete admin page layout with sidebar, header, and content area
 *
 * Attributes:
 * - title: Page title shown in header (required)
 * - subtitle: Optional subtitle/description
 * - show-refresh: Show auto-refresh indicator (default: false)
 *
 * Slots:
 * - default: Main page content
 * - styles: Additional page-specific styles
 * - scripts: Additional page-specific scripts (loaded after common scripts)
 */
class EwrAdminPage extends HTMLElement {
    constructor() {
        super();
        this._initialized = false;
        this._rendered = false;
    }

    connectedCallback() {
        if (this._initialized) return;
        this._initialized = true;

        // Render the layout structure
        this._render();

        // Wait for dependencies to be loaded before initializing
        if (window.__ewrAdminReady) {
            // Already ready - init immediately
            this._initializePage();
        } else if (document.querySelector('script[src*="ewr-admin-loader"]')) {
            // Using loader - wait for ready event
            document.addEventListener('ewr-admin-ready', () => this._initializePage(), { once: true });
        } else {
            // Direct script includes - init on next tick to ensure DOM is ready
            setTimeout(() => this._initializePage(), 0);
        }
    }

    _render() {
        if (this._rendered) return;
        this._rendered = true;

        const title = this.getAttribute('title') || 'Admin';
        const subtitle = this.getAttribute('subtitle') || '';
        const showRefresh = this.getAttribute('show-refresh') === 'true';

        // Get the original content before we replace it
        const originalContent = this.innerHTML;

        // Get any style tags from the original content
        const styleMatch = originalContent.match(/<style[^>]*>([\s\S]*?)<\/style>/gi) || [];
        const styles = styleMatch.join('\n');

        // Get content without style tags
        const contentWithoutStyles = originalContent.replace(/<style[^>]*>[\s\S]*?<\/style>/gi, '').trim();

        // Build the complete page structure
        this.innerHTML = `
            <div class="app-layout">
                <!-- Admin Sidebar -->
                <aside class="app-sidebar">
                    <div class="sidebar-brand">
                        <div class="sidebar-logo">EWR</div>
                        <div class="user-info-card">
                            <div class="user-info-row">
                                <span class="user-name" id="sidebarUserName">Loading...</span>
                                <span class="user-role-badge" id="sidebarUserRole">...</span>
                            </div>
                        </div>
                    </div>
                    <nav class="sidebar-nav" id="sidebarNav">
                        <!-- Navigation populated by sidebar.js -->
                    </nav>
                </aside>

                <!-- Main Content Area -->
                <div class="app-main">
                    <!-- Header -->
                    <header class="app-header">
                        <div class="header-content">
                            <div class="header-left">
                                <div>
                                    <h1 class="header-title">${title}</h1>
                                    ${subtitle ? `<p class="header-subtitle">${subtitle}</p>` : ''}
                                </div>
                            </div>
                            <div class="header-right" style="display: flex; gap: 12px; align-items: center;">
                                ${showRefresh ? `
                                    <div class="refresh-indicator" id="refreshIndicator">
                                        <span>Auto-refresh: <span id="refreshCountdown">5s</span></span>
                                    </div>
                                    <button class="btn btn-secondary" onclick="refreshAllData()">Refresh Now</button>
                                ` : `
                                    <ewr-header-status></ewr-header-status>
                                `}
                            </div>
                        </div>
                    </header>

                    <!-- Content -->
                    <main class="app-content">
                        <div class="content-wrapper">
                            ${contentWithoutStyles}
                        </div>
                    </main>
                </div>
            </div>
            ${styles}
        `;
    }

    async _initializePage() {
        // Check authentication
        if (typeof AuthClient !== 'undefined') {
            const auth = new AuthClient();
            if (!auth.isAuthenticated()) {
                window.location.href = '/login.html?redirect=' + encodeURIComponent(window.location.pathname);
                return;
            }
        }

        // Ensure sidebar nav exists before initializing
        const sidebarNav = this.querySelector('.sidebar-nav');
        if (!sidebarNav) {
            console.error('EwrAdminPage: sidebar-nav not found, re-rendering');
            this._rendered = false;
            this._render();
        }

        // Initialize sidebar
        if (typeof initSidebar === 'function') {
            await initSidebar();
        }

        // Dispatch event for page-specific initialization
        this.dispatchEvent(new CustomEvent('page-ready', { bubbles: true }));
    }
}

/**
 * EWR Page Layout (Non-Admin)
 * Complete page layout with standard sidebar, header, and content area
 *
 * Attributes:
 * - title: Page title shown in header (required)
 * - subtitle: Optional subtitle/description
 *
 * Slots:
 * - default: Main page content
 */
class EwrPage extends HTMLElement {
    constructor() {
        super();
        this._initialized = false;
    }

    connectedCallback() {
        if (this._initialized) return;
        this._initialized = true;

        const title = this.getAttribute('title') || 'EWR AI Assistant';
        const subtitle = this.getAttribute('subtitle') || '';

        // Get the original content before we replace it
        const originalContent = this.innerHTML;

        // Get any style tags from the original content
        const styleMatch = originalContent.match(/<style[^>]*>([\s\S]*?)<\/style>/gi) || [];
        const styles = styleMatch.join('\n');

        // Get content without style tags
        const contentWithoutStyles = originalContent.replace(/<style[^>]*>[\s\S]*?<\/style>/gi, '').trim();

        // Build the complete page structure
        this.innerHTML = `
            <div class="app-layout">
                <!-- Sidebar -->
                <aside class="app-sidebar">
                    <div class="sidebar-brand">
                        <div class="sidebar-logo">EWR</div>
                        <div class="user-info">
                            <div class="user-name" id="userName">Loading...</div>
                            <div class="user-role" id="userRole">User</div>
                        </div>
                    </div>
                    <nav class="sidebar-nav">
                        <!-- Navigation generated by sidebar.js -->
                    </nav>
                </aside>

                <!-- Main Content Area -->
                <div class="app-main">
                    <!-- Header -->
                    <header class="app-header">
                        <div class="header-content">
                            <div class="header-left">
                                <div>
                                    <h1 class="header-title">${title}</h1>
                                    ${subtitle ? `<p class="header-subtitle">${subtitle}</p>` : ''}
                                </div>
                            </div>
                            <div class="header-right">
                                <ewr-header-status></ewr-header-status>
                            </div>
                        </div>
                    </header>

                    <!-- Content -->
                    <main class="app-content">
                        <div class="content-wrapper">
                            ${contentWithoutStyles}
                        </div>
                    </main>
                </div>
            </div>
            ${styles}
        `;

        // Initialize the page
        this._initializePage();
    }

    async _initializePage() {
        // Check authentication
        if (typeof AuthClient !== 'undefined') {
            const auth = new AuthClient();
            if (!auth.isAuthenticated()) {
                window.location.href = '/login.html?redirect=' + encodeURIComponent(window.location.pathname);
                return;
            }

            // Update user info
            try {
                const user = await auth.getUser();
                if (user) {
                    const userNameEl = document.getElementById('userName');
                    const userRoleEl = document.getElementById('userRole');
                    if (userNameEl) userNameEl.textContent = user.username;
                    if (userRoleEl) userRoleEl.textContent = user.role || 'User';
                }
            } catch (e) {
                console.error('Failed to get user info:', e);
            }
        }

        // Initialize sidebar
        if (typeof initSidebar === 'function') {
            await initSidebar();
        }

        // Dispatch event for page-specific initialization
        this.dispatchEvent(new CustomEvent('page-ready', { bubbles: true }));
    }
}

// Register components
customElements.define('ewr-admin-page', EwrAdminPage);
customElements.define('ewr-page', EwrPage);

/**
 * Global logout function available to all pages
 */
function logout() {
    if (typeof AuthClient !== 'undefined') {
        const auth = new AuthClient();
        auth.logout();
    }
}
