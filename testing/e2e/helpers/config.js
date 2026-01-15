/**
 * Configuration helper for E2E tests
 * Loads configuration from environment variables
 */

/**
 * Get base URL for the application
 * @returns {string}
 */
export function getBaseUrl() {
  return process.env.BASE_URL || 'http://localhost:3000';
}

/**
 * Get timeout configuration
 * @returns {{default: number, long: number, short: number}}
 */
export function getTimeouts() {
  return {
    short: 5000,
    default: 10000,
    long: 30000,
  };
}

/**
 * Get API endpoint URLs
 * @returns {object}
 */
export function getApiEndpoints() {
  const baseUrl = getBaseUrl();
  return {
    sql: `${baseUrl}/api/sql`,
    audio: `${baseUrl}/api/audio`,
    documents: `${baseUrl}/api/documents`,
    admin: `${baseUrl}/api/admin`,
    git: `${baseUrl}/api/git`,
  };
}
