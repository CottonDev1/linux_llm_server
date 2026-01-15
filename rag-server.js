#!/usr/bin/env node

/**
 * RAG Web Server - Employee Code Query Interface
 *
 * Provides a web interface for employees to query the codebase using natural language.
 * All data operations are handled by the Python FastAPI service (MongoDB).
 * This server handles HTTP routing, authentication, and LLM integration.
 *
 * Architecture:
 * - Node.js: HTTP routing, authentication, caching
 * - Python Service: MongoDB operations, embeddings, vector search
 * - llama.cpp: LLM for natural language processing
 */

// ============================================================================
// Auto-Install Missing Dependencies
// ============================================================================
import { execSync } from 'child_process';
import { readFileSync, existsSync } from 'fs';
import { join, dirname } from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

/**
 * Check if a package is installed by attempting to resolve it
 */
function isPackageInstalled(packageName) {
  try {
    import.meta.resolve(packageName);
    return true;
  } catch {
    return false;
  }
}

/**
 * Install missing npm packages based on package.json
 */
async function ensureDependencies() {
  console.log('Checking dependencies...');

  const packageJsonPath = join(__dirname, 'package.json');

  if (!existsSync(packageJsonPath)) {
    console.warn('package.json not found, skipping dependency check');
    return;
  }

  try {
    const packageJson = JSON.parse(readFileSync(packageJsonPath, 'utf-8'));
    const dependencies = {
      ...packageJson.dependencies || {},
      ...packageJson.devDependencies || {}
    };

    const missingPackages = [];

    for (const [packageName] of Object.entries(dependencies)) {
      if (!isPackageInstalled(packageName)) {
        missingPackages.push(packageName);
      }
    }

    if (missingPackages.length === 0) {
      console.log('All dependencies are installed');
      return;
    }

    console.log(`Missing ${missingPackages.length} package(s):`);
    missingPackages.forEach(pkg => console.log(`   - ${pkg}`));

    console.log('\nInstalling missing packages...');
    console.log('   This may take a few minutes on first deployment...\n');

    try {
      execSync('npm install', {
        cwd: __dirname,
        stdio: 'inherit',
        encoding: 'utf-8'
      });

      console.log('\nSuccessfully installed all dependencies');
    } catch (installError) {
      console.error('\nFailed to install dependencies');
      console.error('   Error:', installError.message);
      console.error('\n   Please run manually: npm install');
      process.exit(1);
    }
  } catch (error) {
    console.error('Error checking dependencies:', error.message);
    console.error('   Continuing anyway, but some features may not work...');
  }
}

// Run dependency check before importing other modules
await ensureDependencies();

// ============================================================================
// Load Environment Variables
// ============================================================================
const dotenv = await import('dotenv');
dotenv.default.config();

// ============================================================================
// Kill Existing Server Processes
// ============================================================================

/**
 * Kill any existing Node processes running rag-server.js
 * Uses tasklist which is more universally available than wmic
 */
async function killExistingProcesses() {
  try {
    console.log(`Checking for existing rag-server.js processes...`);

    const currentPid = process.pid;

    try {
      // Try using tasklist first (more compatible)
      const stdout = execSync('tasklist /FI "IMAGENAME eq node.exe" /FO CSV /NH',
        { encoding: 'utf8', timeout: 5000 }
      );

      // tasklist output format: "node.exe","1234","Console","1","12,345 K"
      const lines = stdout.trim().split('\n').filter(line => line.includes('node.exe'));

      if (lines.length <= 1) {
        // Only current process or none, no need to kill anything
        return;
      }

      console.log(`Found ${lines.length} node.exe process(es), checking for rag-server instances...`);
      // Note: tasklist doesn't show command line, so we'll skip killing for safety
      // The server will fail to bind to port if another instance is running

    } catch (error) {
      // Silently continue - process cleanup is best-effort
      console.log('Process check skipped (tasklist not available)');
    }
  } catch (error) {
    // Non-critical, continue with startup
    console.log('Process cleanup skipped');
  }
}

// ============================================================================
// Import Core Dependencies
// ============================================================================
import express from 'express';
import cors from 'cors';
import multer from 'multer';
import fs from 'fs';
import path from 'path';
// Note: Using native fetch (Node.js 22+) instead of node-fetch
// import fetch from 'node-fetch';

// Python API Client - all data operations go through Python service
import { SQLKnowledgeDB, DocumentationDB, CodeContextDB } from './src/services/PythonApiClient.js';

// Phase 1: Production Enhancements
// Cache implementations removed - Python services handle caching (SemanticCache, ResponseCache)
import ProductionMonitoring from './src/monitoring/ProductionMonitoring.js';

// Phase 3: Multi-User Authentication
import EWRAIDatabase from './src/db/EWRAIDatabase.js';
import { optionalAuth, requireAuthOrAdmin } from './src/middleware/authMiddleware.js';
// concurrencyMiddleware removed - trackQuery was never used
import createAuthRoutes from './src/routes/authRoutes.js';

// Route modules
import createQueryRoutes from './src/routes/queryRoutes.js';
import createAdminRoutes from './src/routes/adminRoutes.js';
import createCodeFlowRoutes from './src/routes/codeFlowRoutes.js';
import createGitRoutes from './src/routes/gitRoutes.js';
// gitRoutesProxy removed - proxy layer was never enabled
import createRoslynRoutes from './src/routes/roslynRoutes.js';
import createHealthRoutes from './src/routes/healthRoutes.js';
import createTagRoutes from './src/routes/tagRoutes.js';
import createLLMRoutes from './src/routes/llmRoutes.js';
import createLogsRoutes from './src/routes/logsRoutes.js';
import createAudioRoutes from './src/routes/audioRoutes.js';
import createCodeRoutes from './src/routes/codeRoutes.js';
import createDocsRoutes from './src/routes/docsRoutes.js';

// Logging
import { apiAccessLogger } from './src/logging/PipelineLogger.js';
// Note: systemLog and loggers removed - unused. Consider consolidating into ProductionMonitoring.

// ============================================================================
// Configuration & Constants
// ============================================================================
const app = express();
const PORT = process.env.PORT || 3000;

// Python service URL - all LLM calls route through Python
const PYTHON_SERVICE_URL = process.env.PYTHON_SERVICE_URL || 'http://localhost:8001';

// Multi-model llama.cpp endpoints (used by Python service, kept for reference)
// Note: Node.js no longer calls these directly - all LLM requests go through Python
const LLM_SQL_HOST = process.env.LLAMACPP_SQL_HOST || 'http://localhost:8080';
const LLM_GENERAL_HOST = process.env.LLAMACPP_HOST || 'http://localhost:8081';
const LLM_CODE_HOST = process.env.LLAMACPP_CODE_HOST || 'http://localhost:8082';

// Dynamic configuration (can be updated via admin API)
const serverConfig = {
  temperature: 0.7,
  maxSourceLength: 2000,
  defaultLimit: 3
};

// Git repository configuration
const GIT_ROOT = process.env.GIT_ROOT;

// Validate required environment variables
if (!GIT_ROOT) {
  console.error('❌ FATAL: GIT_ROOT environment variable is not set');
  console.error('   Please set GIT_ROOT in your .env file');
  console.error('   Example: GIT_ROOT=C:\\Projects\\Git');
  process.exit(1);
}

// ============================================================================
// Helper Functions for Git Repository Scanning
// ============================================================================

/**
 * Scan for git repositories in the configured root directory
 */
function scanForGitRepositories() {
  const repositories = [];

  function scanDirectory(dirPath, relativePath = '') {
    try {
      const entries = fs.readdirSync(dirPath, { withFileTypes: true });

      for (const entry of entries) {
        if (entry.isDirectory()) {
          const fullPath = path.join(dirPath, entry.name);
          const gitPath = path.join(fullPath, '.git');

          if (fs.existsSync(gitPath)) {
            const repoName = relativePath ? `${relativePath}/${entry.name}` : entry.name;
            repositories.push({
              name: repoName,
              path: fullPath,
              displayName: repoName
            });
          } else {
            const newRelativePath = relativePath ? `${relativePath}/${entry.name}` : entry.name;
            scanDirectory(fullPath, newRelativePath);
          }
        }
      }
    } catch (error) {
      if (error.code !== 'EACCES' && error.code !== 'EPERM') {
        monitoring.warn(`Error scanning directory ${dirPath}:`, error.message);
      }
    }
  }

  try {
    if (!fs.existsSync(GIT_ROOT)) {
      monitoring.warn(`Git root directory not found: ${GIT_ROOT}`);
      return repositories;
    }

    scanDirectory(GIT_ROOT);
    monitoring.info(`Found ${repositories.length} git repositories in ${GIT_ROOT}`);
    return repositories;
  } catch (error) {
    monitoring.error('Error scanning for git repositories', error);
    return repositories;
  }
}

/**
 * Get repository path by name
 */
function getRepositoryPath(repoName) {
  const repos = scanForGitRepositories();
  const repo = repos.find(r => r.name === repoName);
  return repo ? repo.path : null;
}

// ============================================================================
// Express Middleware Setup
// ============================================================================
app.use(cors());
app.use(express.json());
app.use(apiAccessLogger);  // Log all API requests with user/IP tracking
app.use(express.static('public'));

// Configure multer for file uploads
const upload = multer({
  storage: multer.memoryStorage(),
  limits: {
    fileSize: 50 * 1024 * 1024 // 50MB max
  }
});

// ============================================================================
// Initialize Core Components
// ============================================================================

// Initialize monitoring first
const monitoring = new ProductionMonitoring({
  logLevel: process.env.LOG_LEVEL || 'info',
  logDir: process.env.LOG_DIR || './logs'
});

// Caches removed - Python services (SemanticCache, ResponseCache) handle all caching

// Initialize Python API clients (MongoDB access)
const sqlDB = new SQLKnowledgeDB({ pythonServiceUrl: PYTHON_SERVICE_URL });
const documentationDB = new DocumentationDB({ pythonServiceUrl: PYTHON_SERVICE_URL });
const codeContextDB = new CodeContextDB({ pythonServiceUrl: PYTHON_SERVICE_URL });

// Create EWRAIDatabase instance for authentication
const ewraiDatabase = new EWRAIDatabase();

// Initialization flags
let pythonServiceInitialized = false;
let userDBInitialized = false;

/**
 * LLM client wrapper for service layer (routes through Python service)
 * All LLM calls now go through Python which handles multi-model routing
 */
const llmClient = {
  endpoint: PYTHON_SERVICE_URL,

  async generate(prompt, options = {}) {
    const response = await fetch(`${this.endpoint}/llm/generate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        prompt,
        system: options.system || 'You are a helpful assistant.',
        max_tokens: options.max_tokens || 1000,
        temperature: options.temperature || serverConfig.temperature || 0.7,
        use_cache: options.use_cache !== false
      }),
      signal: AbortSignal.timeout(120000)
    });

    if (!response.ok) {
      throw new Error(`LLM API error: ${response.statusText}`);
    }

    const data = await response.json();
    return data.content || data.response;
  },

  async embeddings(text) {
    // Embeddings are handled by Python service's embedding model
    const response = await fetch(`${this.endpoint}/embeddings`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        text: text
      })
    });

    if (!response.ok) {
      throw new Error(`Embeddings error: ${response.statusText}`);
    }

    const data = await response.json();
    return data.embedding;
  }
};

// ============================================================================
// Service Initialization
// ============================================================================

/**
 * Initialize Python service connection with retry logic
 */
async function initializePythonService() {
  const maxRetries = 10;
  const retryDelay = 2000; // 2 seconds between retries

  // Try both localhost and 127.0.0.1 in case of DNS issues
  const urlsToTry = [
    PYTHON_SERVICE_URL,
    PYTHON_SERVICE_URL.replace('localhost', '127.0.0.1')
  ];
  // Dedupe if they're the same
  const uniqueUrls = [...new Set(urlsToTry)];

  monitoring.info(`Connecting to Python service at ${PYTHON_SERVICE_URL}...`);
  monitoring.info(`Will try URLs: ${uniqueUrls.join(', ')}`);

  for (let attempt = 1; attempt <= maxRetries; attempt++) {
    // Try each URL variant
    for (const baseUrl of uniqueUrls) {
      try {
        const statusUrl = `${baseUrl}/status`;
        monitoring.info(`Attempt ${attempt}/${maxRetries}: trying ${statusUrl}`);

        const response = await fetch(statusUrl, {
          signal: AbortSignal.timeout(5000)
        });

        if (!response.ok) {
          throw new Error(`Python service returned ${response.status}`);
        }

        const status = await response.json();

        // Check if MongoDB is connected
        if (status.mongodb?.connected) {
          monitoring.info(`Python service connected at ${baseUrl}: MongoDB connected, Embeddings: ${status.embeddings?.loaded}`);

          // Initialize API clients
          await sqlDB.initialize();
          await documentationDB.initialize();
          await codeContextDB.initialize();

          pythonServiceInitialized = true;
          monitoring.updateHealthStatus('database', true);
          return true;
        } else {
          monitoring.info(`Python service responding but MongoDB not ready (attempt ${attempt}/${maxRetries})`);
        }
      } catch (error) {
        monitoring.info(`Failed ${baseUrl}: ${error.message}`);
      }
    }

    // Wait before next retry (except on last attempt)
    if (attempt < maxRetries) {
      await new Promise(resolve => setTimeout(resolve, retryDelay));
    }
  }

  // All retries exhausted
  monitoring.error('Failed to connect to Python service after ' + maxRetries + ' attempts');
  monitoring.warn('Server will start but database operations will fail');
  monitoring.warn(`Make sure Python service is running: cd python_services && python -m uvicorn main:app --port 8001`);
  pythonServiceInitialized = false;
  return false;
}

// ============================================================================
// LLM (llama.cpp) & Redis Management Functions
// ============================================================================

/**
 * Check if a server is running on a specific host and port
 */
async function checkPortStatus(port, host = 'localhost') {
  const net = await import('net');

  return new Promise((resolve) => {
    const socket = new net.default.Socket();
    socket.setTimeout(2000);

    socket.on('connect', () => {
      socket.destroy();
      resolve(true);
    });

    socket.on('timeout', () => {
      socket.destroy();
      resolve(false);
    });

    socket.on('error', () => {
      socket.destroy();
      resolve(false);
    });

    socket.connect(port, host);
  });
}

/**
 * Check if llama.cpp server is running (multi-model setup)
 * Checks SQL model port (8080) as the primary indicator
 */
async function checkLLMStatus() {
  // Extract host and port from LLAMACPP_SQL_HOST or default to localhost:8080
  const sqlHost = process.env.LLAMACPP_SQL_HOST || 'http://localhost:8080';
  const urlMatch = sqlHost.match(/\/\/([^:]+):(\d+)/);
  const checkHost = urlMatch ? urlMatch[1] : 'localhost';
  const checkPort = urlMatch ? parseInt(urlMatch[2]) : 8080;

  return checkPortStatus(checkPort, checkHost);
}

/**
 * Start llama.cpp server if not running (manual start required)
 */
async function startLLM() {
  console.log('llama.cpp server needs to be started manually.');
  console.log('Please run: scripts/start-llamacpp-multimodel.ps1');
  return false;
}

/**
 * Ensure llama.cpp is running before server starts
 * Multi-model setup: checks ports 8080 (SQL), 8081 (General), 8082 (Code)
 */
async function ensureLLMRunning() {
  // Helper to extract host and port from URL
  const parseHostPort = (url, defaultHost = 'localhost', defaultPort) => {
    const match = url?.match(/\/\/([^:\/]+):(\d+)/);
    return {
      host: match ? match[1] : defaultHost,
      port: match ? parseInt(match[2]) : defaultPort
    };
  };

  const sql = parseHostPort(process.env.LLAMACPP_SQL_HOST, 'localhost', 8080);
  const general = parseHostPort(process.env.LLAMACPP_HOST, 'localhost', 8081);
  const code = parseHostPort(process.env.LLAMACPP_CODE_HOST, 'localhost', 8082);

  console.log('Checking llama.cpp multi-model servers...');

  const sqlRunning = await checkPortStatus(sql.port, sql.host);
  const generalRunning = await checkPortStatus(general.port, general.host);
  const codeRunning = await checkPortStatus(code.port, code.host);

  console.log(`  SQL model (${sql.host}:${sql.port}): ${sqlRunning ? 'RUNNING' : 'NOT RUNNING'}`);
  console.log(`  General model (${general.host}:${general.port}): ${generalRunning ? 'RUNNING' : 'NOT RUNNING'}`);
  console.log(`  Code model (${code.host}:${code.port}): ${codeRunning ? 'RUNNING' : 'NOT RUNNING'}`);

  // Require at least SQL model to be running
  if (sqlRunning) {
    console.log('llama.cpp SQL model is running - proceeding with startup');
    return true;
  }

  console.error('');
  console.error(`llama.cpp SQL model is NOT running on ${sql.host}:${sql.port}`);
  console.error('');
  console.error('Please start llama.cpp multi-model servers:');
  console.error('  .\\scripts\\start-llamacpp-multimodel.ps1');
  console.error('');
  console.error('Or see docs/LLAMACPP_SETUP_GUIDE.md for setup instructions');
  console.error('');

  throw new Error('llama.cpp SQL model is not running. Please start llama.cpp first.');
}

/**
 * Check if Redis server is running
 */
async function checkRedisStatus() {
  const net = await import('net');

  return new Promise((resolve) => {
    const socket = new net.default.Socket();
    const port = parseInt(process.env.REDIS_PORT) || 6379;
    const host = process.env.REDIS_HOST || 'localhost';

    socket.setTimeout(2000);

    socket.on('connect', () => {
      socket.destroy();
      resolve(true);
    });

    socket.on('timeout', () => {
      socket.destroy();
      resolve(false);
    });

    socket.on('error', () => {
      socket.destroy();
      resolve(false);
    });

    socket.connect(port, host);
  });
}

/**
 * Check if Memurai service exists
 */
async function checkRedisBinaryExists() {
  try {
    execSync('sc query Memurai', { stdio: 'ignore' });
    return true;
  } catch (error) {
    return false;
  }
}

/**
 * Start Memurai service if not running
 */
async function startRedis() {
  console.log('Starting Memurai service...');

  try {
    execSync('sc query Memurai', { stdio: 'ignore' });
    execSync('net start Memurai', { stdio: 'ignore' });
    console.log('Memurai service started');

    console.log('Waiting for Memurai to respond...');

    for (let i = 0; i < 10; i++) {
      await new Promise(resolve => setTimeout(resolve, 1000));

      if (await checkRedisStatus()) {
        console.log('Memurai is ready');
        return true;
      }
    }

    console.warn('Memurai started but not responding after 10 seconds');
    return false;

  } catch (error) {
    console.warn('Memurai service not found or could not start');
    console.log('Install Memurai for response caching: choco install memurai-developer');
    return false;
  }
}

/**
 * Ensure Memurai is running
 */
async function ensureRedisRunning() {
  if (process.env.SKIP_REDIS_START === 'true') {
    console.log('Skipping Memurai startup check (SKIP_REDIS_START=true)');
    return true;
  }

  console.log('Checking Memurai service status...');

  const isRunning = await checkRedisStatus();

  if (isRunning) {
    console.log('Memurai service is already running');
    return true;
  }

  console.log('Memurai service is not running');

  const serviceExists = await checkRedisBinaryExists();

  if (!serviceExists) {
    console.log('Memurai service not found');
    console.log('Server will continue without caching');
    return true;
  }

  console.log('Attempting to start Memurai...');
  const started = await startRedis();

  if (!started) {
    console.log('Could not start Memurai automatically');
    console.log('Server will continue without response caching');
  }

  return true;
}

// ============================================================================
// Git Repository Configuration
// ============================================================================

/**
 * Load git repository configuration
 */
async function loadGitRepoConfig() {
  try {
    monitoring.info('Loading git repository configuration...');
    const repos = scanForGitRepositories();

    if (userDBInitialized && ewraiDatabase) {
      for (const repo of repos) {
        ewraiDatabase.saveGitRepository(repo);
      }
      monitoring.info(`Loaded and saved ${repos.length} git repositories to database`);
    } else {
      monitoring.info(`Loaded ${repos.length} git repositories (database not initialized yet)`);
    }
  } catch (error) {
    monitoring.error('Failed to load git repository configuration', error);
  }
}

// ============================================================================
// Mount Route Modules
// ============================================================================

/**
 * Mount all route modules after initialization
 */
function mountRoutes() {
  let routesLoaded = 0;

  // Dependency injection object for routes
  const routeDeps = {
    sqlDB,
    documentationDB,
    codeContextDB,
    monitoring,
    // embeddingCache and responseCache removed - Python services handle caching
    serverConfig,
    llmClient,
    upload,
    // LLM endpoints removed - all LLM calls now route through Python service
    // Python service handles multi-model routing internally
    PORT,
    gitRoot: GIT_ROOT,
    pythonServiceInitialized,
    pythonServiceUrl: PYTHON_SERVICE_URL,
    // helpers removed - utility functions are defined locally in routes where needed
    getRepositoryPath,
    scanForGitRepositories,
    requireAuth: requireAuthOrAdmin,
    requireAuthOrAdmin,
    ewraiDatabase,
    userDBInitialized,
    gitRepositories: scanForGitRepositories(),
    saveGitRepoConfig: async (repo) => {
      if (userDBInitialized && ewraiDatabase) {
        return ewraiDatabase.saveGitRepository(repo);
      }
      console.warn('saveGitRepoConfig: database not initialized');
      return null;
    },
    getRecentCommits: async (repoPath, limit = 5) => {
      try {
        const { execSync } = await import('child_process');
        const output = execSync(
          `git log --no-merges --pretty=format:"%H|%an|%ad|%s" --date=iso -${limit}`,
          { cwd: repoPath, encoding: 'utf8' }
        );

        if (!output || !output.trim()) {
          return [];
        }

        return output.trim().split('\n').map(line => {
          const [hash, author, date, message] = line.split('|');
          return { hash, author, date, message };
        });
      } catch (error) {
        console.error(`Error getting commits from ${repoPath}:`, error.message);
        return [];
      }
    },
    getLastPullTime: async (repoPath) => {
      try {
        const { execSync } = await import('child_process');
        const output = execSync(
          'git log -1 --format=%cd --date=iso',
          { cwd: repoPath, encoding: 'utf8' }
        );
        return output.trim();
      } catch (error) {
        return null;
      }
    }
  };

  // Client configuration endpoint (public - no auth required)
  app.get('/api/client-config', (req, res) => {
    res.json({
      adminRefreshInterval: parseInt(process.env.ADMIN_REFRESH_INTERVAL, 10) || 60
    });
  });

  // Mount all route modules
  try {
    app.use('/api', createHealthRoutes(routeDeps));
    routesLoaded++;
  } catch (error) {
    monitoring.error('Failed to mount health routes', error);
  }

  try {
    app.use('/api', createQueryRoutes(routeDeps));
    routesLoaded++;
  } catch (error) {
    monitoring.error('Failed to mount query routes', error);
  }

  try {
    app.use('/api/admin', createAdminRoutes(routeDeps));
    routesLoaded++;
  } catch (error) {
    monitoring.error('Failed to mount admin routes', error);
  }

  try {
    app.use('/api/admin/tags', createTagRoutes(routeDeps));
    routesLoaded++;
  } catch (error) {
    monitoring.error('Failed to mount tag routes', error);
  }

  try {
    app.use('/api', createCodeFlowRoutes(routeDeps));
    routesLoaded++;
  } catch (error) {
    monitoring.error('Failed to mount code flow routes', error);
  }



  try {
    app.use('/api/git', createGitRoutes(routeDeps));
    routesLoaded++;
  } catch (error) {
    monitoring.error('Failed to mount git routes', error);
  }

  try {
    app.use('/api/roslyn', createRoslynRoutes(routeDeps));
    routesLoaded++;
  } catch (error) {
    monitoring.error('Failed to mount Roslyn routes', error);
  }

  try {
    app.use('/api/llm', createLLMRoutes(routeDeps));
    routesLoaded++;
  } catch (error) {
    monitoring.error('Failed to mount LLM routes', error);
  }

  try {
    app.use('/api/logs', createLogsRoutes(routeDeps));
    routesLoaded++;
  } catch (error) {
    monitoring.error('Failed to mount Logs routes', error);
  }

  try {
    app.use('/api/audio', createAudioRoutes(routeDeps));
    routesLoaded++;
  } catch (error) {
    monitoring.error('Failed to mount Audio routes', error);
  }

  try {
    app.use('/api/code', createCodeRoutes(routeDeps));
    routesLoaded++;
  } catch (error) {
    monitoring.error('Failed to mount Code Assistance routes', error);
  }

  try {
    app.use('/api/docs', createDocsRoutes({ requireAuthOrAdmin }));
    routesLoaded++;
  } catch (error) {
    monitoring.error('Failed to mount Docs routes', error);
  }

  // Proxy all /api/python/* requests to Python service
  app.use('/api/python', async (req, res) => {
    try {
      // Build target URL with query parameters
      const queryString = Object.keys(req.query).length > 0
        ? '?' + new URLSearchParams(req.query).toString()
        : '';
      const targetUrl = `${PYTHON_SERVICE_URL}${req.path}${queryString}`;

      monitoring.info(`Proxying request to Python service: ${req.method} ${targetUrl}`);

      const fetchOptions = {
        method: req.method,
        headers: {
          'Content-Type': 'application/json'
        }
      };

      // Add body for POST/PUT/PATCH requests
      if (['POST', 'PUT', 'PATCH'].includes(req.method) && req.body) {
        fetchOptions.body = JSON.stringify(req.body);
      }

      const response = await fetch(targetUrl, fetchOptions);

      // Handle non-JSON responses
      const contentType = response.headers.get('content-type');
      if (contentType && contentType.includes('application/json')) {
        const data = await response.json();
        res.status(response.status).json(data);
      } else {
        const text = await response.text();
        res.status(response.status).send(text);
      }
    } catch (error) {
      monitoring.error('Python service proxy error', error);
      res.status(500).json({
        success: false,
        error: 'Failed to communicate with Python service',
        details: error.message
      });
    }
  });
  routesLoaded++;

  return routesLoaded;
}

// ============================================================================
// Server Startup
// ============================================================================

/**
 * Main server startup function
 */
async function start() {
  // Kill any existing server processes first
  await killExistingProcesses();

  // Ensure external services are running
  await ensureLLMRunning();
  await ensureRedisRunning();

  // Initialize Python service connection (MongoDB)
  await initializePythonService();

  // Initialize EWR AI Database for multi-user authentication
  let authRouteMounted = false;
  try {
    await ewraiDatabase.initialize();
    userDBInitialized = true;

    app.use('/api/auth', createAuthRoutes(ewraiDatabase));
    authRouteMounted = true;

    const adminExists = await ewraiDatabase.getUserByUsername('admin');
    if (!adminExists) {
      await ewraiDatabase.createUser({
        username: 'admin',
        password: 'Admin123',
        role: 'admin',
        settings: serverConfig
      });
      monitoring.info('Default admin user created (username: admin, password: Admin123)');
    }

    await loadGitRepoConfig();
  } catch (error) {
    monitoring.error('Failed to initialize user database', error);
    userDBInitialized = false;
  }

  // Mount all route modules
  monitoring.info('Loading available API endpoints...');
  const routesLoaded = mountRoutes();
  const totalRoutes = routesLoaded + (authRouteMounted ? 1 : 0);
  monitoring.info(`${totalRoutes}/${totalRoutes} API endpoints loaded`);

  // Start Express server
  app.listen(PORT, '0.0.0.0', () => {
    monitoring.info('\n' + '='.repeat(60));
    monitoring.info('RAG Server Started Successfully!');
    monitoring.info('='.repeat(60));
    monitoring.info(`Server running at: http://localhost:${PORT}`);
    monitoring.info(`Network access: http://<your-ip>:${PORT}`);
    monitoring.info(`Python Service: ${PYTHON_SERVICE_URL}`);
    monitoring.info(`LLM Access: via Python service (multi-model routing)`);
    monitoring.info(`Database: MongoDB (via Python service)`);
    monitoring.info('='.repeat(60));
    monitoring.info('\nReady for employee queries!\n');
  });
}

// ============================================================================
// Graceful Shutdown
// ============================================================================

process.on('SIGINT', async () => {
  monitoring.info('Shutting down gracefully...');

  try {
    await monitoring.shutdown();
    process.exit(0);
  } catch (error) {
    console.error('Error during shutdown:', error);
    process.exit(1);
  }
});

// ============================================================================
// Start Server
// ============================================================================

start().catch(error => {
  monitoring.error('Failed to start server', error, { severity: 'critical' });
  process.exit(1);
});
