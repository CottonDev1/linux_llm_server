# Playwright E2E Testing Guide

This is the authoritative guide for Playwright end-to-end (E2E) testing in the LLM Website project.

## Table of Contents

- [What is Playwright?](#what-is-playwright)
- [Why Playwright?](#why-playwright)
- [Project Setup](#project-setup)
- [Running Tests](#running-tests)
- [Test Suites](#test-suites)
- [Prefect Integration](#prefect-integration)
- [Writing New Tests](#writing-new-tests)
- [Test Reports](#test-reports)
- [Troubleshooting](#troubleshooting)

---

## What is Playwright?

Playwright is a modern, open-source browser automation framework developed by Microsoft. It enables reliable end-to-end testing for web applications by automating browser interactions across Chromium, Firefox, and WebKit.

### Key Features

- **Cross-browser Testing**: Test on Chrome, Firefox, and Safari with a single API
- **Auto-wait**: Automatically waits for elements to be actionable before performing actions
- **Headless Mode**: Run tests without a visible browser window (ideal for servers)
- **Network Interception**: Mock API responses and test edge cases
- **Screenshots & Videos**: Capture visual evidence on test failures
- **Parallel Execution**: Run tests concurrently for faster feedback

### How It Works

1. Playwright launches a browser instance (headless or visible)
2. Your test script navigates to pages and interacts with elements
3. Assertions verify the application behaves correctly
4. Results are reported with screenshots/videos on failure

---

## Why Playwright?

We chose Playwright over alternatives (Selenium, Cypress, Puppeteer) for these reasons:

| Feature | Playwright | Selenium | Cypress |
|---------|------------|----------|---------|
| **Headless Support** | Native | Requires config | Native |
| **Auto-wait** | Built-in | Manual | Built-in |
| **Speed** | Fast | Slower | Fast |
| **Cross-browser** | All major | All major | Limited |
| **Language Support** | JS, Python, .NET, Java | Many | JS only |
| **CI/CD Ready** | Excellent | Good | Good |

Playwright's native headless support makes it ideal for our Ubuntu server environment where there's no display.

---

## Project Setup

### Prerequisites

- Node.js 18+ installed
- Project dependencies installed (`npm install`)

### Installation

Playwright is already configured in this project. To verify:

```bash
# Check Playwright is installed
npx playwright --version

# Install browsers if needed
npx playwright install chromium
```

### Configuration

Playwright configuration is in `playwright.config.js`:

```javascript
// Key settings
testDir: './tests/e2e',           // Test file location
timeout: 60000,                    // 60 second timeout per test
baseURL: 'http://localhost:3000',  // Application URL
headless: true,                    // Run without visible browser
```

---

## Running Tests

### From Command Line

```bash
# Run all tests
npx playwright test

# Run specific test file
npx playwright test admin-pages.spec.js

# Run tests with visible browser (debugging)
npx playwright test --headed

# Run tests in specific browser
npx playwright test --project=chromium

# Generate HTML report
npx playwright show-report
```

### From Prefect Dashboard (Recommended)

The easiest way to run tests is through the Prefect dashboard at `http://10.101.20.21:4200`:

1. Navigate to **Deployments**
2. Select a test deployment (e.g., `playwright_quick_test/smoke-test`)
3. Click **Run**
4. Monitor progress in the **Runs** tab

See [Prefect Integration](#prefect-integration) for details.

---

## Test Suites

### Overview

Tests are organized into suites by feature area:

| Suite | File | Category | Description |
|-------|------|----------|-------------|
| admin-pages | `admin-pages.spec.js` | Admin | Admin page loading and RBAC tests |
| sidebar-links | `sidebar-links.spec.js` | Admin | Sidebar navigation and link verification |
| user-management | `user-management.spec.js` | Admin | User creation, roles, staff dashboard |
| sql-query | `sql-query.spec.js` | SQL | SQL query generation tests |
| sql-agent | `sql-agent.spec.js` | SQL | SQL agent functionality tests |
| audio-single | `audio-single.spec.js` | Audio | Single audio file processing |
| audio-bulk | `audio-bulk.spec.js` | Audio | Bulk audio processing (max 2 files) |
| audio-analysis | `audio-analysis.spec.js` | Audio | Staff monitoring, search, view/edit |
| knowledge-base-chat | `knowledge-base-chat.spec.js` | Knowledge Base | Chat interface tests |
| document-agent | `document-agent.spec.js` | Documents | Document agent functionality |
| git-analysis | `git-analysis.spec.js` | Git | Git analysis feature tests |
| semantic-ticket-match | `semantic-ticket-match.spec.js` | Tickets | Semantic ticket matching |
| cotton-provider-doc | `cotton-provider-doc.spec.js` | Knowledge Base | Cotton Provider Q&A validation |

### Detailed Suite Descriptions

#### Admin Pages (`admin-pages.spec.js`)

Tests that admin pages load correctly and enforce role-based access control (RBAC):

- Verifies all admin pages return 200 status
- Tests that unauthorized users cannot access admin-only pages
- Validates page titles and basic content rendering
- Checks authentication redirects for unauthenticated users

#### Sidebar Links (`sidebar-links.spec.js`)

Tests sidebar navigation functionality:

- Verifies all sidebar links are present
- Tests that links navigate to correct pages
- Validates category expansion/collapse
- Checks active page highlighting
- Tests external link behavior (opens in new tab)

#### User Management (`user-management.spec.js`)

Tests user administration features:

- Create new user with email and role
- Assign monitoring categories to users
- Verify staff dashboard displays correct user cards
- Test user deactivation/activation
- Validate role permission inheritance

#### SQL Query (`sql-query.spec.js`)

Tests SQL query generation interface:

- Natural language to SQL conversion
- Query execution and result display
- Error handling for invalid queries
- Query history functionality
- Schema browser interaction

#### SQL Agent (`sql-agent.spec.js`)

Tests the AI-powered SQL agent:

- Multi-turn conversation handling
- Context retention across queries
- Complex query decomposition
- Result explanation generation

#### Audio Single (`audio-single.spec.js`)

Tests single audio file processing:

- File upload interface
- Transcription initiation
- Progress tracking
- Result display and playback
- Error handling for invalid files

#### Audio Bulk (`audio-bulk.spec.js`)

Tests bulk audio processing (limited to 2 files for test speed):

- Multiple file selection
- Batch upload progress
- Parallel processing status
- Aggregate results view

#### Audio Analysis (`audio-analysis.spec.js`)

Tests the audio analysis workflow:

- Staff monitoring dashboard
- Audio search functionality
- View and edit analysis results
- Export capabilities

#### Knowledge Base Chat (`knowledge-base-chat.spec.js`)

Tests the RAG-powered knowledge base chat:

- Question input and submission
- Response streaming display
- Source document citations
- Feedback (thumbs up/down) functionality
- Conversation history

#### Document Agent (`document-agent.spec.js`)

Tests document processing features:

- Document upload and ingestion
- Text extraction verification
- Chunk preview
- Document search

#### Git Analysis (`git-analysis.spec.js`)

Tests git repository analysis features:

- Repository connection
- Commit history analysis
- Code change summaries

#### Semantic Ticket Match (`semantic-ticket-match.spec.js`)

Tests semantic ticket matching:

- Ticket similarity search
- Match confidence scores
- Related ticket suggestions

#### Cotton Provider Doc (`cotton-provider-doc.spec.js`)

Validation tests for Cotton Provider documentation Q&A:

- Known question/answer pairs
- Response accuracy verification
- Source attribution

---

## Prefect Integration

Tests are integrated with Prefect for web-based triggering and monitoring.

### Available Deployments

| Deployment | Description | Duration |
|------------|-------------|----------|
| `playwright_full_test/full-suite` | All tests | ~15-30 min |
| `playwright_quick_test/smoke-test` | Core functionality only | ~3-5 min |
| `playwright_test_flow/custom` | Select specific suites | Varies |
| `playwright_audio_test/audio-tests` | All audio tests | ~10 min |
| `playwright_audio_analysis_test/audio-analysis` | Audio analysis only | ~5 min |
| `playwright_user_management_test/user-management` | User/RBAC tests | ~5 min |

### Running Tests via Prefect

#### Web Dashboard

1. Open `http://10.101.20.21:4200/deployments`
2. Click on a deployment
3. Click **Quick Run** or **Custom Run** (to set parameters)
4. Monitor in the **Runs** section

#### CLI

```bash
# Run smoke test
prefect deployment run 'playwright_quick_test/smoke-test'

# Run full suite
prefect deployment run 'playwright_full_test/full-suite'

# Run custom suites
prefect deployment run 'playwright_test_flow/custom' \
  --param suites='["admin-pages", "sql-query"]'
```

### Starting the Deployment Server

If deployments aren't available, start the server:

```bash
cd /data/projects/llm_website/python_services/prefect_pipelines
source ../venv/bin/activate
python playwright_deployments.py
```

This runs in the foreground. Use `nohup` or `tmux` for background execution.

---

## Writing New Tests

### Test File Location

Place test files in: `/data/projects/llm_website/tests/e2e/`

### Basic Test Structure

```javascript
// tests/e2e/example.spec.js
import { test, expect } from '@playwright/test';

test.describe('Feature Name', () => {
  test.beforeEach(async ({ page }) => {
    // Setup: login, navigate, etc.
    await page.goto('/admin/index.html');
  });

  test('should do something', async ({ page }) => {
    // Arrange
    await page.click('#some-button');

    // Act
    await page.fill('#input-field', 'test value');
    await page.click('#submit');

    // Assert
    await expect(page.locator('#result')).toContainText('Success');
  });

  test('should handle errors', async ({ page }) => {
    await page.click('#invalid-action');
    await expect(page.locator('.error-message')).toBeVisible();
  });
});
```

### Best Practices

1. **Use descriptive test names**: `should display error when login fails`
2. **One assertion per test** (when practical)
3. **Use page objects** for complex pages
4. **Avoid hard-coded waits**: Use `waitForSelector` or auto-wait
5. **Clean up test data** in `afterEach` hooks

### Adding to Prefect

To add a new test suite to Prefect orchestration:

1. Add the suite to `TEST_SUITES` in `playwright_test_flow.py`:

```python
TEST_SUITES = {
    # ... existing suites ...
    "my-new-suite": {
        "file": "my-new-suite.spec.js",
        "description": "Description of what this tests",
        "category": "feature-area"
    },
}
```

2. Restart the deployment server

---

## Test Reports

### Allure Reports

Tests generate Allure reports for detailed analysis:

```bash
# Generate report
npx allure generate allure-results --clean -o allure-report

# View report (starts local server)
npx allure open allure-report
```

Report location: `/data/projects/llm_website/allure-report/`

### Playwright HTML Report

```bash
# View HTML report
npx playwright show-report
```

Report location: `/data/projects/llm_website/playwright-report/`

### Prefect Artifacts

When run via Prefect, test results are saved as markdown artifacts visible in the dashboard:

- Navigate to the flow run
- Click on the **Artifacts** tab
- View the `playwright-test-report` artifact

---

## Troubleshooting

### Common Issues

#### Tests fail with "Browser not found"

```bash
# Install browsers
npx playwright install chromium
```

#### Tests timeout on headless server

Increase timeout in `playwright.config.js`:

```javascript
timeout: 120000,  // 2 minutes
```

#### "Cannot find module" errors

```bash
# Reinstall dependencies
npm install
```

#### Tests pass locally but fail in Prefect

- Check environment variables are set
- Verify the web server is running
- Check network connectivity to `localhost:3000`

### Debug Mode

Run tests with visible browser and slow motion:

```bash
npx playwright test --headed --slowmo=1000
```

### Trace Viewer

On failure, Playwright captures traces. View them:

```bash
npx playwright show-trace test-results/trace.zip
```

### Logs

Prefect logs are available at:
- Dashboard: `http://10.101.20.21:4200` > Flow Runs > Select Run > Logs
- File: `/tmp/playwright_deployments.log`

---

## Quick Reference

### Commands

| Command | Description |
|---------|-------------|
| `npx playwright test` | Run all tests |
| `npx playwright test --headed` | Run with visible browser |
| `npx playwright test <file>` | Run specific file |
| `npx playwright show-report` | View HTML report |
| `npx playwright install` | Install browsers |

### Files

| File | Purpose |
|------|---------|
| `playwright.config.js` | Test configuration |
| `tests/e2e/*.spec.js` | Test files |
| `allure-results/` | Raw test results |
| `allure-report/` | Generated Allure report |
| `playwright-report/` | Generated HTML report |
| `python_services/prefect_pipelines/playwright_test_flow.py` | Prefect orchestration |
| `python_services/prefect_pipelines/playwright_deployments.py` | Prefect deployments |

### URLs

| URL | Description |
|-----|-------------|
| `http://10.101.20.21:4200` | Prefect Dashboard |
| `http://10.101.20.21:3000` | Application under test |

---

## Additional Resources

- [Playwright Documentation](https://playwright.dev/docs/intro)
- [Playwright API Reference](https://playwright.dev/docs/api/class-playwright)
- [Prefect Documentation](https://docs.prefect.io/)
- [Allure Report Documentation](https://docs.qameta.io/allure/)
