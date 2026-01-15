/**
 * Sidebar User Display Utility
 * Loads and displays the logged-in username in the sidebar
 */
(function() {
    function loadSidebarUsername() {
        const usernameEl = document.getElementById('sidebarUsername');
        if (usernameEl) {
            try {
                const userStr = localStorage.getItem('user');
                if (userStr) {
                    const user = JSON.parse(userStr);
                    usernameEl.textContent = user.username || '';
                }
            } catch (e) {
                console.log('Could not load username for sidebar');
            }
        }
    }

    // Run when DOM is ready
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', loadSidebarUsername);
    } else {
        loadSidebarUsername();
    }
})();
