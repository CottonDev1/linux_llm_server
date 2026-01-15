// @ts-check
import { defineConfig, devices } from '@playwright/test';

/**
 * Playwright configuration with Allure reporting
 * @see https://playwright.dev/docs/test-configuration
 */
export default defineConfig({
  // Test directory
  testDir: './tests/e2e',

  // Run tests in parallel within a file
  fullyParallel: false,

  // Fail the build on CI if you accidentally left test.only in the source code
  forbidOnly: !!process.env.CI,

  // Retry failed tests on CI
  retries: process.env.CI ? 2 : 0,

  // Limit parallel workers
  workers: 1,

  // Reporter configuration - Allure + HTML
  reporter: [
    ['list'],
    ['allure-playwright', {
      outputFolder: 'allure-results',
      suiteTitle: true,
      detail: true,
      categories: [
        {
          name: 'Test failures',
          matchedStatuses: ['failed', 'broken']
        },
        {
          name: 'Flaky tests',
          matchedStatuses: ['flaky']
        }
      ],
      environmentInfo: {
        'Node Version': process.version,
        'OS': process.platform,
        'Environment': process.env.NODE_ENV || 'development'
      }
    }],
    ['html', { outputFolder: 'playwright-report', open: 'never' }]
  ],

  // Global test timeout
  timeout: 60000,

  // Expect timeout
  expect: {
    timeout: 10000
  },

  // Shared settings for all projects
  use: {
    // Base URL for the application
    baseURL: process.env.BASE_URL || 'http://localhost:3000',

    // Capture trace on first retry
    trace: 'on-first-retry',

    // Capture screenshot on failure
    screenshot: 'only-on-failure',

    // Capture video on failure
    video: 'retain-on-failure',

    // Viewport size
    viewport: { width: 1280, height: 720 },

    // Ignore HTTPS errors
    ignoreHTTPSErrors: true,
  },

  // Configure projects for different browsers
  projects: [
    {
      name: 'chromium',
      use: {
        ...devices['Desktop Chrome'],
        headless: process.env.HEADLESS !== 'false'
      },
    },
    // Uncomment to test on other browsers
    // {
    //   name: 'firefox',
    //   use: { ...devices['Desktop Firefox'] },
    // },
    // {
    //   name: 'webkit',
    //   use: { ...devices['Desktop Safari'] },
    // },
  ],

  // Output folder for test artifacts
  outputDir: 'test-results/',

  // Run local server before tests if needed
  // webServer: {
  //   command: 'npm run start',
  //   url: 'http://localhost:3000',
  //   reuseExistingServer: !process.env.CI,
  //   timeout: 120000,
  // },
});
