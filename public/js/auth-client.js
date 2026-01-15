/**
 * Authentication Client Library
 * Handles JWT-based authentication with the RAG Server
 */
class AuthClient {
    constructor() {
        this.accessToken = null;
        this.refreshToken = null;
        this.user = null;
        this.refreshTimer = null;

        // Load tokens from storage on initialization
        this.loadTokens();

        // Set up axios-like interceptor for fetch
        this.setupFetchInterceptor();

        // Start auto-refresh if we have tokens
        if (this.accessToken && this.refreshToken) {
            this.startAutoRefresh();
        }
    }

    /**
     * Load tokens from localStorage
     */
    loadTokens() {
        this.accessToken = localStorage.getItem('accessToken');
        this.refreshToken = localStorage.getItem('refreshToken');
        const userStr = localStorage.getItem('user');
        if (userStr) {
            try {
                this.user = JSON.parse(userStr);
            } catch (e) {
                console.error('Failed to parse user data:', e);
                this.user = null;
            }
        }
    }

    /**
     * Save tokens to localStorage
     */
    saveTokens(accessToken, refreshToken, user) {
        this.accessToken = accessToken;
        this.refreshToken = refreshToken;
        this.user = user;

        localStorage.setItem('accessToken', accessToken);
        localStorage.setItem('refreshToken', refreshToken);
        if (user) {
            localStorage.setItem('user', JSON.stringify(user));
        }
    }

    /**
     * Clear tokens from storage
     */
    clearTokens() {
        this.accessToken = null;
        this.refreshToken = null;
        this.user = null;

        localStorage.removeItem('accessToken');
        localStorage.removeItem('refreshToken');
        localStorage.removeItem('user');

        // Stop auto-refresh
        if (this.refreshTimer) {
            clearTimeout(this.refreshTimer);
            this.refreshTimer = null;
        }
    }

    /**
     * Login with username and password
     * @param {string} username
     * @param {string} password
     * @returns {Promise<Object>} User object
     */
    async login(username, password) {
        try {
            const response = await fetch('/api/auth/login', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ username, password })
            });

            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.error || 'Login failed');
            }

            const data = await response.json();

            // Save tokens
            this.saveTokens(data.accessToken, data.refreshToken, data.user);

            // Start auto-refresh
            this.startAutoRefresh();

            return data.user;
        } catch (error) {
            console.error('Login error:', error);
            throw error;
        }
    }

    /**
     * Logout the current user
     */
    async logout() {
        try {
            if (this.refreshToken) {
                // Call logout endpoint to invalidate refresh token on server
                await fetch('/api/auth/logout', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                        'Authorization': `Bearer ${this.accessToken}`
                    },
                    body: JSON.stringify({ refreshToken: this.refreshToken })
                });
            }
        } catch (error) {
            console.error('Logout error:', error);
        } finally {
            // Clear tokens regardless of server response
            this.clearTokens();

            // Redirect to login page
            window.location.href = '/login.html';
        }
    }

    /**
     * Refresh the access token using the refresh token
     * @returns {Promise<string>} New access token
     */
    async refreshAccessToken() {
        if (!this.refreshToken) {
            throw new Error('No refresh token available');
        }

        try {
            const response = await fetch('/api/auth/refresh', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ refreshToken: this.refreshToken })
            });

            if (!response.ok) {
                throw new Error('Token refresh failed');
            }

            const data = await response.json();

            // Update access token
            this.accessToken = data.accessToken;
            localStorage.setItem('accessToken', data.accessToken);

            // Restart auto-refresh
            this.startAutoRefresh();

            return data.accessToken;
        } catch (error) {
            console.error('Token refresh error:', error);
            // Clear tokens and redirect to login
            this.clearTokens();
            window.location.href = '/login.html?redirect=' + encodeURIComponent(window.location.pathname);
            throw error;
        }
    }

    /**
     * Check if user is authenticated
     * @returns {boolean}
     */
    isAuthenticated() {
        return !!(this.accessToken && this.refreshToken);
    }

    /**
     * Get the current access token
     * @returns {string|null}
     */
    getAccessToken() {
        return this.accessToken;
    }

    /**
     * Get current user info
     * @returns {Promise<Object>} User object
     */
    async getUser() {
        if (this.user) {
            return this.user;
        }

        if (!this.accessToken) {
            throw new Error('Not authenticated');
        }

        try {
            const response = await fetch('/api/auth/me', {
                headers: {
                    'Authorization': `Bearer ${this.accessToken}`
                }
            });

            if (!response.ok) {
                throw new Error('Failed to get user info');
            }

            const user = await response.json();
            this.user = user;
            localStorage.setItem('user', JSON.stringify(user));

            return user;
        } catch (error) {
            console.error('Get user error:', error);
            throw error;
        }
    }

    /**
     * Check if user has a specific role
     * @param {string} role - Role to check (user, developer, admin)
     * @returns {boolean}
     */
    hasRole(role) {
        if (!this.user) {
            return false;
        }
        return this.user.role === role;
    }

    /**
     * Check if user is admin
     * @returns {boolean}
     */
    isAdmin() {
        return this.hasRole('admin');
    }

    /**
     * Start auto-refresh timer
     * Refreshes token 5 minutes before expiry
     */
    startAutoRefresh() {
        // Clear existing timer
        if (this.refreshTimer) {
            clearTimeout(this.refreshTimer);
        }

        // Decode token to get expiry
        try {
            const payload = this.parseJWT(this.accessToken);
            const expiresAt = payload.exp * 1000; // Convert to milliseconds
            const now = Date.now();
            const refreshIn = expiresAt - now - (5 * 60 * 1000); // 5 minutes before expiry

            if (refreshIn > 0) {
                this.refreshTimer = setTimeout(() => {
                    this.refreshAccessToken().catch(error => {
                        console.error('Auto-refresh failed:', error);
                    });
                }, refreshIn);
            }
        } catch (error) {
            console.error('Failed to parse token for auto-refresh:', error);
        }
    }

    /**
     * Parse JWT token
     * @param {string} token
     * @returns {Object} Token payload
     */
    parseJWT(token) {
        try {
            const base64Url = token.split('.')[1];
            const base64 = base64Url.replace(/-/g, '+').replace(/_/g, '/');
            const jsonPayload = decodeURIComponent(atob(base64).split('').map(c => {
                return '%' + ('00' + c.charCodeAt(0).toString(16)).slice(-2);
            }).join(''));
            return JSON.parse(jsonPayload);
        } catch (error) {
            throw new Error('Invalid token');
        }
    }

    /**
     * Setup fetch interceptor to add auth headers
     */
    setupFetchInterceptor() {
        const originalFetch = window.fetch;
        const authClient = this;

        window.fetch = async function(...args) {
            let [url, config = {}] = args;

            // Only add auth header for API calls
            if (url.startsWith('/api/') && authClient.accessToken) {
                config.headers = {
                    ...config.headers,
                    'Authorization': `Bearer ${authClient.accessToken}`
                };
            }

            try {
                const response = await originalFetch(url, config);

                // If 401, try to refresh token
                if (response.status === 401 && authClient.refreshToken && !url.includes('/auth/')) {
                    try {
                        await authClient.refreshAccessToken();
                        // Retry original request with new token
                        config.headers['Authorization'] = `Bearer ${authClient.accessToken}`;
                        return originalFetch(url, config);
                    } catch (refreshError) {
                        // Refresh failed, redirect to login
                        authClient.clearTokens();
                        window.location.href = '/login.html?redirect=' + encodeURIComponent(window.location.pathname);
                        return response;
                    }
                }

                return response;
            } catch (error) {
                throw error;
            }
        };
    }

    /**
     * Make authenticated API request
     * @param {string} url
     * @param {Object} options
     * @returns {Promise<Response>}
     */
    async authenticatedFetch(url, options = {}) {
        if (!this.accessToken) {
            throw new Error('Not authenticated');
        }

        return fetch(url, {
            ...options,
            headers: {
                ...options.headers,
                'Authorization': `Bearer ${this.accessToken}`
            }
        });
    }

    /**
     * Check authentication and redirect if needed
     * @param {boolean} requireAdmin - Whether admin role is required
     */
    requireAuth(requireAdmin = false) {
        if (!this.isAuthenticated()) {
            window.location.href = '/login.html?redirect=' + encodeURIComponent(window.location.pathname);
            return false;
        }

        if (requireAdmin && !this.isAdmin()) {
            alert('Admin access required');
            window.location.href = '/';
            return false;
        }

        return true;
    }
}

// Export for use in other scripts
if (typeof module !== 'undefined' && module.exports) {
    module.exports = AuthClient;
}