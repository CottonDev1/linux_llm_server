/**
 * Authentication helper functions for E2E tests
 * Loads credentials from environment variables
 */

/**
 * Get admin credentials from environment
 * @returns {{username: string, password: string}}
 * @throws {Error} If credentials are not configured
 */
export function getAdminCredentials() {
  const username = process.env.ADMIN_USERNAME;
  const password = process.env.ADMIN_PASSWORD;

  if (!username || !password) {
    throw new Error('Admin credentials not configured in .env');
  }

  return { username, password };
}

/**
 * Get SQL server configuration from environment
 * @returns {{server: string, database: string, username: string, password: string}}
 */
export function getSqlConfig() {
  return {
    server: process.env.SQL_SERVER || '',
    database: process.env.SQL_DATABASE || '',
    username: process.env.SQL_USERNAME || '',
    password: process.env.SQL_PASSWORD || '',
  };
}

/**
 * Login as admin user
 * @param {import('@playwright/test').Page} page - Playwright page object
 * @returns {Promise<string>} Access token
 * @throws {Error} If login fails
 */
export async function loginAsAdmin(page) {
  const { username, password } = getAdminCredentials();

  await page.goto('/login.html', { waitUntil: 'networkidle' });
  await page.fill('#username', username);
  await page.fill('#password', password);
  await page.click('#loginButton');

  // Wait for redirect away from login page
  await page.waitForURL(url => !url.toString().includes('login.html'), { timeout: 10000 });

  // Verify access token exists
  const accessToken = await page.evaluate(() => localStorage.getItem('accessToken'));
  if (!accessToken) {
    throw new Error('Login failed - no access token');
  }

  return accessToken;
}
