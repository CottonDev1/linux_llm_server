/**
 * EWR Page Loader (Non-Admin)
 * Single script that loads all dependencies for standard pages
 *
 * Include this ONE script in pages:
 * <script src="/js/ewr-page-loader.js"></script>
 *
 * This automatically loads:
 * - layout.css
 * - ewr-components.js
 * - auth-client.js
 * - sidebar.js
 * - system-status.js
 * - sidebar-user.js
 * - ewr-layouts.js
 */

(function() {
    const basePath = '/';

    // CSS files to load
    const cssFiles = [
        'css/layout.css'
    ];

    // JS files to load (in order)
    const jsFiles = [
        'js/ewr-components.js',
        'js/auth-client.js',
        'js/sidebar.js',
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
            // All scripts loaded, dispatch ready event
            document.dispatchEvent(new CustomEvent('ewr-page-ready'));
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
