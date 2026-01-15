/**
 * EWR Admin Page Loader
 * Single script that loads all dependencies for admin pages
 *
 * Include this ONE script in admin pages:
 * <script src="/js/ewr-admin-loader.js"></script>
 *
 * This automatically loads:
 * - layout.css
 * - ewr-classes.css
 * - ewr-components.js
 * - auth-client.js
 * - admin sidebar.js
 * - system-status.js
 * - sidebar-user.js
 * - ewr-layouts.js
 */

(function() {
    const basePath = '/';

    // CSS files to load
    const cssFiles = [
        'css/layout.css',
        'css/ewr-classes.css'
    ];

    // JS files to load (in order)
    const jsFiles = [
        'js/ewr-components.js',
        'js/auth-client.js',
        'admin/js/sidebar.js',
        'js/system-status.js',
        'js/sidebar-user.js',
        'js/ewr-layouts.js'
    ];

    // Load CSS files
    cssFiles.forEach(file => {
        if (!document.querySelector(`link[href="${basePath}${file}"]`)) {
            const link = document.createElement('link');
            link.rel = 'stylesheet';
            link.href = basePath + file;
            document.head.appendChild(link);
        }
    });

    // Load JS files sequentially
    let loadIndex = 0;

    function loadNextScript() {
        if (loadIndex >= jsFiles.length) {
            // All scripts loaded - set flag and dispatch ready event
            window.__ewrAdminReady = true;
            document.dispatchEvent(new CustomEvent('ewr-admin-ready'));
            return;
        }

        const file = jsFiles[loadIndex];
        if (document.querySelector(`script[src="${basePath}${file}"]`)) {
            // Already loaded, skip
            loadIndex++;
            loadNextScript();
            return;
        }

        const script = document.createElement('script');
        script.src = basePath + file;
        script.onload = () => {
            loadIndex++;
            loadNextScript();
        };
        script.onerror = () => {
            console.error(`Failed to load: ${file}`);
            loadIndex++;
            loadNextScript();
        };
        document.head.appendChild(script);
    }

    // Start loading when DOM is ready
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', loadNextScript);
    } else {
        loadNextScript();
    }
})();
