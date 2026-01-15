/**
 * Breadcrumb Navigation Component
 * Generates dynamic breadcrumb navigation based on current page
 */

class Breadcrumb {
    constructor() {
        this.routes = {
            '/admin/': { label: 'Dashboard', icon: '🏠' },
            '/admin/index.html': { label: 'Dashboard', icon: '🏠' },
            '/admin/roslyn.html': { label: 'Roslyn Analysis', icon: '🔬' },
            '/admin/git.html': { label: 'Git Integration', icon: '📦' },
            '/admin/upload.html': { label: 'File Upload', icon: '📤' },
            '/admin/sql.html': { label: 'SQL Database', icon: '🗄️' },
            '/admin/system.html': { label: 'System Settings', icon: '⚙️' }
        };
    }

    /**
     * Get current page information
     */
    getCurrentPage() {
        const path = window.location.pathname;
        const normalized = path.endsWith('/') ? path + 'index.html' : path;
        return this.routes[normalized] || { label: 'Admin', icon: '🔧' };
    }

    /**
     * Render breadcrumb HTML
     */
    render() {
        const currentPage = this.getCurrentPage();
        const isHome = window.location.pathname.includes('index.html') || window.location.pathname.endsWith('/admin/');

        let html = '<nav class="breadcrumb-nav" aria-label="Breadcrumb">';

        // Home link
        html += `
            <div class="breadcrumb-item">
                <a href="../index.html" class="breadcrumb-link">🏠 Home</a>
            </div>
        `;

        // Separator
        html += `<span class="breadcrumb-separator">›</span>`;

        // Admin Dashboard link (if not on dashboard)
        if (!isHome) {
            html += `
                <div class="breadcrumb-item">
                    <a href="/admin/" class="breadcrumb-link">🔧 Admin</a>
                </div>
                <span class="breadcrumb-separator">›</span>
            `;
        }

        // Current page
        html += `
            <div class="breadcrumb-item">
                <span class="breadcrumb-current">${currentPage.icon} ${currentPage.label}</span>
            </div>
        `;

        html += '</nav>';

        return html;
    }

    /**
     * Initialize breadcrumb on page
     */
    init() {
        const breadcrumbContainer = document.querySelector('.breadcrumb');
        if (breadcrumbContainer) {
            breadcrumbContainer.innerHTML = this.render();
        }
    }
}

// Auto-initialize when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    const breadcrumb = new Breadcrumb();
    breadcrumb.init();
});
