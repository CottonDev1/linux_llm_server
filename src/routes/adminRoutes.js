/**
 * Admin Routes Module
 *
 * All admin-related route handlers extracted from rag-server.js
 * Handles:
 * - Settings management
 * - User management (create, read, update, delete users)
 * - Document upload and management
 * - Project management
 * - Git operations (pull, update context, sync repositories)
 * - Server configuration and control
 * - Database operations (list databases, compare schemas, extract schema)
 * - Roslyn code analysis
 * - Vector database statistics and monitoring
 */

import express from 'express';
import multer from 'multer';

// Import extracted route modules
import initUserRoutes from './userRoutes.js';
import initDocumentRoutes from './documentRoutes.js';
import initGitRoutes from './gitRoutes.js';
import initDatabaseRoutes from './databaseRoutes.js';
import initLlmRoutes from './llmRoutes.js';
import initNavigationRoutes from './navigationRoutes.js';

const router = express.Router();

// Multer configuration for file uploads
const storage = multer.memoryStorage();
const upload = multer({
  storage,
  limits: {
    fileSize: 50 * 1024 * 1024 // 50MB limit
  }
});

/**
 * Initialize admin routes with required dependencies
 *
 * @param {Object} dependencies - Service and middleware dependencies
 * @param {Function} dependencies.requireAuth - Authentication middleware (admin password)
 * @param {Function} dependencies.requireAuthOrAdmin - Auth middleware (JWT or admin password)
 * @param {Object} dependencies.serverConfig - Server configuration object
 * @param {Object} dependencies.db - Vector database instance
 * @param {Object} dependencies.multiTableSearch - MultiTableSearch instance
 * @param {Object} dependencies.documentationDB - DocumentationDatabase instance
 * @param {Object} dependencies.ewraiDatabase - EWRAIDatabase instance
 * @param {Object} dependencies.adminService - AdminService instance (Phase 2)
 * @param {Object} dependencies.codeAnalysisService - CodeAnalysisService instance (Phase 2)
 * @param {Object} dependencies.monitoring - ProductionMonitoring instance
 * @param {Object} dependencies.gitRepositories - Git repositories configuration
 * @param {Function} dependencies.saveGitRepoConfig - Function to save git repo config
 * @param {Function} dependencies.getRecentCommits - Function to get recent commits
 * @param {Function} dependencies.getLastPullTime - Function to get last pull time
 * @param {boolean} dependencies.userDBInitialized - Flag if user DB is initialized
 * @param {number} dependencies.PORT - Server port
 * @param {string} dependencies.pythonServiceUrl - Python service URL (all LLM calls route through Python)
 */
export default function createAdminRoutes(dependencies) {
  const {
    requireAuth,
    requireAuthOrAdmin,
    serverConfig,
    db,
    multiTableSearch,
    documentationDB,
    ewraiDatabase,
    adminService,
    codeAnalysisService,
    monitoring,
    gitRepositories,
    scanForGitRepositories,
    saveGitRepoConfig,
    getRecentCommits,
    getLastPullTime,
    userDBInitialized,
    PORT,
    pythonServiceUrl
  } = dependencies;

  // ============================================================================
  // Mount Extracted Route Modules
  // ============================================================================

  // Initialize and mount user routes at /users
  const userRouter = initUserRoutes({
    requireAuth,
    requireAuthOrAdmin,
    ewraiDatabase,
    serverConfig
  });
  router.use('/users', userRouter);

  // Initialize and mount document routes (at root level for /upload, /validate, /documents, /stats)
  const documentRouter = initDocumentRoutes({
    requireAuth,
    documentationDB,
    db,
    multiTableSearch,
    adminService,
    monitoring
  });
  router.use('/', documentRouter);

  // Initialize and mount git routes at /git
  const gitRouter = initGitRoutes({
    requireAuth,
    ewraiDatabase,
    gitRepositories,
    scanForGitRepositories,
    saveGitRepoConfig,
    getRecentCommits,
    getLastPullTime,
    userDBInitialized
  });
  router.use('/git', gitRouter);

  // Initialize and mount database routes (at root level for /list-databases, etc.)
  const databaseRouter = initDatabaseRoutes({
    requireAuth
  });
  router.use('/', databaseRouter);

  // Initialize and mount LLM and service routes (at root level for /llm/*, /service-config, etc.)
  const llmRouter = initLlmRoutes({
    requireAuth,
    ewraiDatabase
  });
  router.use('/', llmRouter);

  // Initialize and mount navigation routes (at root level for /navigation)
  const navigationRouter = initNavigationRoutes({});
  router.use('/', navigationRouter);

  // ============================================================================
  // Settings Management
  // ============================================================================

  /**
   * POST /api/admin/settings
   * Update LLM settings (supports both JWT auth and legacy admin password)
   */
  router.post('/settings', requireAuthOrAdmin, async (req, res) => {
    try {
      const { temperature, contextSize, numSources, model } = req.body;

      // Build settings object
      const newSettings = {};
      if (temperature !== undefined) newSettings.temperature = temperature;
      if (contextSize !== undefined) newSettings.maxSourceLength = contextSize;
      if (numSources !== undefined) newSettings.defaultLimit = numSources;

      // If authenticated user, update their personal settings
      if (req.user && userDBInitialized) {
        const updatedUser = await ewraiDatabase.updateUserSettings(req.user.id, newSettings);

        // If admin, also update global defaults
        if (req.user.role === 'admin') {
          if (temperature !== undefined) serverConfig.temperature = temperature;
          if (contextSize !== undefined) serverConfig.maxSourceLength = contextSize;
          if (numSources !== undefined) serverConfig.defaultLimit = numSources;
          // Note: model changes are handled by Python service configuration
        }

        res.json({
          success: true,
          settings: {
            temperature: updatedUser.settings.temperature,
            contextSize: updatedUser.settings.maxSourceLength,
            numSources: updatedUser.settings.defaultLimit,
            model: 'via-python-service',
            user: updatedUser.username
          }
        });
      } else {
        // Legacy behavior - update global settings only (admin password auth)
        if (temperature !== undefined) serverConfig.temperature = temperature;
        if (contextSize !== undefined) serverConfig.maxSourceLength = contextSize;
        if (numSources !== undefined) serverConfig.defaultLimit = numSources;
        // Note: model changes are handled by Python service configuration

        res.json({
          success: true,
          settings: {
            temperature: serverConfig.temperature,
            contextSize: serverConfig.maxSourceLength,
            numSources: serverConfig.defaultLimit,
            model: 'via-python-service'
          }
        });
      }
    } catch (error) {
      res.status(500).json({ success: false, error: error.message });
    }
  });

  /**
   * GET /api/admin/settings/max-doc-size
   * Get maximum document upload size in MB
   */
  router.get('/settings/max-doc-size', requireAuthOrAdmin, async (req, res) => {
    try {
      let maxDocSize = ewraiDatabase.getSetting('MaxDocSize');

      // If setting doesn't exist, create it with default value of 20
      if (maxDocSize === null || maxDocSize === undefined) {
        ewraiDatabase.updateSetting('MaxDocSize', '20');
        maxDocSize = '20';
      }

      res.json({
        success: true,
        maxDocSize: parseInt(maxDocSize, 10) || 20
      });
    } catch (error) {
      res.status(500).json({ success: false, error: error.message });
    }
  });

  /**
   * PUT /api/admin/settings/max-doc-size
   * Update maximum document upload size in MB
   */
  router.put('/settings/max-doc-size', requireAuthOrAdmin, async (req, res) => {
    try {
      const { maxDocSize } = req.body;

      if (maxDocSize === undefined || maxDocSize === null) {
        return res.status(400).json({ success: false, error: 'maxDocSize is required' });
      }

      const sizeValue = parseInt(maxDocSize, 10);
      if (isNaN(sizeValue) || sizeValue < 1 || sizeValue > 100) {
        return res.status(400).json({ success: false, error: 'maxDocSize must be between 1 and 100 MB' });
      }

      ewraiDatabase.updateSetting('MaxDocSize', sizeValue.toString());

      res.json({
        success: true,
        maxDocSize: sizeValue,
        message: `Maximum document size updated to ${sizeValue}MB`
      });
    } catch (error) {
      res.status(500).json({ success: false, error: error.message });
    }
  });

  // ============================================================================
  // ============================================================================
  // Project Management
  // ============================================================================

  /**
   * GET /api/admin/projects
   * Get list of all projects with stats from MongoDB via Python service
   */
  router.get('/projects', requireAuth, async (req, res) => {
    try {
      // Query Python service for projects from MongoDB
      const pythonServiceUrl = process.env.PYTHON_SERVICE_URL || 'http://localhost:8001';

      const response = await fetch(`${pythonServiceUrl}/api/projects`, {
        method: 'GET',
        headers: { 'Content-Type': 'application/json' }
      });

      if (!response.ok) {
        // If Python service doesn't have a projects endpoint yet, return empty list
        if (response.status === 404) {
          console.log('📊 Projects endpoint not available on Python service, returning empty list');
          return res.json({
            success: true,
            projects: [],
            totalProjects: 0,
            message: 'Projects endpoint not configured on Python service'
          });
        }
        throw new Error(`Python service returned ${response.status}`);
      }

      const data = await response.json();

      console.log(`📊 Found ${data.projects?.length || 0} projects from MongoDB`);

      res.json({
        success: true,
        projects: data.projects || [],
        totalProjects: data.projects?.length || 0
      });
    } catch (error) {
      console.error('❌ Failed to get projects:', error.message);
      // Return empty list instead of error to avoid breaking the UI
      res.json({
        success: true,
        projects: [],
        totalProjects: 0,
        error: error.message
      });
    }
  });

  /**
   * POST /api/admin/projects
   * Create new project
   */
  router.post('/projects', requireAuth, async (req, res) => {
    try {
      const { name, description } = req.body;

      if (!name) {
        return res.status(400).json({ error: 'Project name is required' });
      }

      // Project is created implicitly when first document is uploaded
      res.json({
        success: true,
        message: `Project "${name}" ready. Upload documents to populate it.`,
        project: { id: name, name, description }
      });

    } catch (error) {
      res.status(500).json({ error: error.message });
    }
  });

  // ============================================================================
  // Server Configuration and Control
  // ============================================================================

  /**
   * POST /api/admin/server/restart
   * Restart server (triggers process exit, should be managed by process manager)
   */
  router.post('/server/restart', requireAuth, async (req, res) => {
    try {
      console.log('🔄 Server restart requested...');

      res.json({
        success: true,
        message: 'Server will restart in 2 seconds. Use a process manager (pm2, nodemon) for automatic restart.'
      });

      // Give time for response to send
      setTimeout(() => {
        console.log('👋 Restarting server...');
        process.exit(0);
      }, 2000);

    } catch (error) {
      res.status(500).json({ error: error.message });
    }
  });

  /**
   * GET /api/admin/config
   * Get server configuration
   */
  router.get('/config', requireAuth, async (req, res) => {
    try {
      res.json({
        // Dynamic configuration
        temperature: serverConfig.temperature,
        maxSourceLength: serverConfig.maxSourceLength,
        defaultLimit: serverConfig.defaultLimit,
        // Static configuration
        port: PORT,
        pythonServiceUrl: pythonServiceUrl || 'http://localhost:8001',
        llmAccess: 'via-python-service',
        dbPath: db.dbPath,
        nodeVersion: process.version,
        platform: process.platform,
        uptime: process.uptime()
      });
    } catch (error) {
      res.status(500).json({ error: error.message });
    }
  });

  /**
   * PUT /api/admin/config
   * Update server configuration
   */
  router.put('/config', requireAuth, async (req, res) => {
    try {
      const { temperature, maxSourceLength, defaultLimit } = req.body;

      // Validate and update temperature (0.0 - 2.0)
      if (temperature !== undefined) {
        const temp = parseFloat(temperature);
        if (isNaN(temp) || temp < 0 || temp > 2.0) {
          return res.status(400).json({
            error: 'Invalid temperature value. Must be between 0.0 and 2.0'
          });
        }
        serverConfig.temperature = temp;
      }

      // Validate and update maxSourceLength (100 - 10000)
      if (maxSourceLength !== undefined) {
        const length = parseInt(maxSourceLength);
        if (isNaN(length) || length < 100 || length > 10000) {
          return res.status(400).json({
            error: 'Invalid maxSourceLength value. Must be between 100 and 10000'
          });
        }
        serverConfig.maxSourceLength = length;
      }

      // Validate and update defaultLimit (1 - 20)
      if (defaultLimit !== undefined) {
        const limit = parseInt(defaultLimit);
        if (isNaN(limit) || limit < 1 || limit > 20) {
          return res.status(400).json({
            error: 'Invalid defaultLimit value. Must be between 1 and 20'
          });
        }
        serverConfig.defaultLimit = limit;
      }

      console.log('✅ Configuration updated:', serverConfig);

      res.json({
        success: true,
        message: 'Configuration updated successfully',
        config: {
          temperature: serverConfig.temperature,
          maxSourceLength: serverConfig.maxSourceLength,
          defaultLimit: serverConfig.defaultLimit
        }
      });

    } catch (error) {
      console.error('❌ Config update error:', error.message);
      res.status(500).json({
        error: 'Failed to update configuration',
        details: error.message
      });
    }
  });

  // ============================================================================
  // Roslyn Code Analysis
  // ============================================================================

  /**
   * POST /api/admin/roslyn/analyze
   * Roslyn C# Code Analysis Endpoint
   */
  router.post('/roslyn/analyze', requireAuth, async (req, res) => {
    try {
      const { project, mode = 'all', incremental = false } = req.body;

      if (!project) {
        return res.status(400).json({
          success: false,
          error: 'Project parameter is required'
        });
      }

      console.log(`🔬 Starting Roslyn analysis for project: ${project} (mode: ${mode})`);

      // Try Phase 2 Service Layer if available
      if (codeAnalysisService) {
        try {
          const result = await codeAnalysisService.analyzeProject(project, {
            incremental
          });

          res.json(result);
          return;
        } catch (serviceError) {
          monitoring.warn('Phase 2 CodeAnalysisService failed, falling back to direct implementation', { error: serviceError.message });
          // Fall through to original implementation
        }
      }

      // Original implementation
      const { exec } = await import('child_process');
      const { promisify } = await import('util');
      const { promises: fs } = await import('fs');
      const { tmpdir } = await import('os');
      const { join } = await import('path');
      const execAsync = promisify(exec);

      // Project configuration mapping
      const GIT_ROOT = process.env.GIT_ROOT;
      const projectConfig = {
        gin: {
          path: `${GIT_ROOT}\\Gin`,
          dbName: 'gin',
          name: 'Gin'
        },
        marketing: {
          path: `${GIT_ROOT}\\Marketing`,
          dbName: 'marketing',
          name: 'Marketing'
        },
        warehouse: {
          path: `${GIT_ROOT}\\Warehouse`,
          dbName: 'warehouse',
          name: 'Warehouse'
        },
        ewrlibrary: {
          path: `${GIT_ROOT}\\EWR Library`,
          dbName: 'ewrlibrary',
          name: 'EWR Library'
        }
      };

      // Handle "all" projects
      if (project === 'all') {
        const allProjects = ['gin', 'marketing', 'warehouse', 'ewrlibrary'];
        const results = [];

        for (const proj of allProjects) {
          try {
            const config = projectConfig[proj];
            const outputFile = join(tmpdir(), `roslyn-analysis-${config.dbName}-${Date.now()}.json`);

            console.log(`  📊 Analyzing ${config.name}...`);

            // Run Roslyn analyzer
            const analyzeCmd = `dotnet roslyn-analyzer/RoslynCodeAnalyzer/bin/Debug/net8.0/RoslynCodeAnalyzer.dll "${config.path}" "${outputFile}"`;
            await execAsync(analyzeCmd, { maxBuffer: 50 * 1024 * 1024, cwd: process.cwd() });

            // Import to vector database
            const importCmd = `node code-import-pipeline.js import "${outputFile}" "${config.dbName}"`;
            await execAsync(importCmd, { maxBuffer: 50 * 1024 * 1024, cwd: process.cwd() });

            // Clean up temp file
            await fs.unlink(outputFile).catch(() => {});

            results.push(`✅ ${config.name}: Analysis complete`);
          } catch (error) {
            results.push(`❌ ${projectConfig[proj].name}: ${error.message}`);
          }
        }

        return res.json({
          success: true,
          summary: results.join('<br>')
        });
      }

      // Single project analysis
      const config = projectConfig[project];
      if (!config) {
        return res.status(400).json({
          success: false,
          error: `Unknown project: ${project}`
        });
      }

      // Create temporary output file
      const outputFile = join(tmpdir(), `roslyn-analysis-${config.dbName}-${Date.now()}.json`);

      // Count C# files
      const countCmd = `find "${config.path}" -name "*.cs" 2>/dev/null | wc -l`;
      const { stdout: countOutput } = await execAsync(countCmd);
      const fileCount = parseInt(countOutput.trim());

      console.log(`  📊 Found ${fileCount} C# files`);

      // Run Roslyn analyzer
      console.log(`  🔍 Running Roslyn analyzer...`);
      const analyzerDll = 'roslyn-analyzer/RoslynCodeAnalyzer/bin/Debug/net8.0/RoslynCodeAnalyzer.dll';
      const analyzeCmd = `dotnet ${analyzerDll} "${config.path}" "${outputFile}"`;

      await execAsync(analyzeCmd, {
        maxBuffer: 50 * 1024 * 1024,
        cwd: process.cwd(),
        timeout: 300000 // 5 minutes
      });

      // Verify output file was created
      const stats = await fs.stat(outputFile);
      console.log(`  📦 Output file created: ${stats.size} bytes`);

      // Import to vector database
      console.log(`  💾 Importing to vector database...`);
      const importCmd = `node code-import-pipeline.js import "${outputFile}" "${config.dbName}"`;
      const { stdout: importOutput } = await execAsync(importCmd, {
        maxBuffer: 50 * 1024 * 1024,
        cwd: process.cwd(),
        timeout: 300000
      });

      // Parse import output for statistics
      const classMatch = importOutput.match(/(\d+)\s+classes/i);
      const methodMatch = importOutput.match(/(\d+)\s+methods/i);
      const callMatch = importOutput.match(/(\d+)\s+call/i);
      const eventMatch = importOutput.match(/(\d+)\s+event/i);

      const summary = `
        <strong>Project:</strong> ${config.name}<br>
        <strong>C# Files:</strong> ${fileCount}<br>
        ${classMatch ? `<strong>Classes:</strong> ${classMatch[1]}<br>` : ''}
        ${methodMatch ? `<strong>Methods:</strong> ${methodMatch[1]}<br>` : ''}
        ${callMatch ? `<strong>Call Relationships:</strong> ${callMatch[1]}<br>` : ''}
        ${eventMatch ? `<strong>Event Handlers:</strong> ${eventMatch[1]}<br>` : ''}
        <strong>Output File:</strong> ${Math.round(stats.size / 1024)} KB
      `;

      // Clean up temp file
      await fs.unlink(outputFile).catch(() => {});

      console.log(`✅ Roslyn analysis complete for ${config.name}`);

      res.json({
        success: true,
        summary: summary,
        project: config.name,
        filesAnalyzed: fileCount
      });

    } catch (error) {
      console.error('❌ Roslyn analysis failed:', error.message);
      res.status(500).json({
        success: false,
        error: 'Roslyn analysis failed',
        message: error.message
      });
    }
  });


  return router;
}
