/**
 * Roslyn Routes Proxy
 *
 * Thin HTTP layer that proxies all Roslyn C# code analysis operations to the Python FastAPI service.
 *
 * The Python service handles:
 * - C# code analysis using Roslyn compiler APIs
 * - Storage in MongoDB with vector embeddings
 * - Semantic search for classes, methods, and event handlers
 * - Call chain analysis (who calls what)
 * - Database operation tracking
 *
 * Node.js only handles:
 * - HTTP routing
 * - Authentication middleware
 * - Request/response formatting
 */

import express from 'express';

const PYTHON_SERVICE_URL = process.env.PYTHON_SERVICE_URL || 'http://localhost:8001';

/**
 * Factory function to create Roslyn proxy routes
 *
 * @param {Object} deps - Dependencies
 * @param {Object} deps.monitoring - Production monitoring instance
 * @param {Function} deps.requireAuth - Authentication middleware
 * @returns {express.Router} Express router with all Roslyn proxy routes
 */
export default function createRoslynRoutes(deps) {
  const router = express.Router();

  const {
    monitoring,
    requireAuth
  } = deps;

  // Health check for Python service connection
  let pythonServiceAvailable = false;

  async function checkPythonService() {
    try {
      const response = await fetch(`${PYTHON_SERVICE_URL}/status`);
      pythonServiceAvailable = response.ok;
    } catch (error) {
      pythonServiceAvailable = false;
      monitoring.warn('Python Roslyn Service not available:', error.message);
    }
  }

  // Check on startup
  checkPythonService();

  // ============================================================================
  // Roslyn Analysis Routes (Proxied to Python)
  // ============================================================================

  /**
   * POST /api/roslyn/analyze
   * Analyze C# code using Roslyn and store results in MongoDB
   */
  router.post('/analyze', requireAuth, async (req, res) => {
    try {
      if (!pythonServiceAvailable) {
        await checkPythonService();
      }

      const { input_path, project, store = true } = req.body;

      if (!input_path) {
        return res.status(400).json({
          success: false,
          error: 'input_path is required'
        });
      }

      monitoring.info(`Analyzing C# code: ${input_path} (project: ${project || 'Unknown'})`);

      const response = await fetch(`${PYTHON_SERVICE_URL}/roslyn/analyze`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ input_path, project, store })
      });

      const result = await response.json();

      if (!response.ok) {
        return res.status(response.status).json(result);
      }

      res.json(result);

    } catch (error) {
      monitoring.error('Failed to analyze C# code:', error.message);
      res.status(500).json({
        success: false,
        error: 'Failed to analyze C# code',
        details: error.message
      });
    }
  });

  /**
   * POST /api/roslyn/call-chain
   * Get the call chain for a method (who calls it, what it calls)
   */
  router.post('/call-chain', async (req, res) => {
    try {
      const { className, methodName, project, direction = 'callers', max_depth = 3 } = req.body;

      if (!className || !methodName) {
        return res.status(400).json({
          success: false,
          error: 'className and methodName are required'
        });
      }

      monitoring.info(`Getting call chain for ${className}.${methodName}`);

      const response = await fetch(`${PYTHON_SERVICE_URL}/roslyn/call-chain`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ className, methodName, project, direction, max_depth })
      });

      const result = await response.json();

      if (!response.ok) {
        return res.status(response.status).json(result);
      }

      res.json(result);

    } catch (error) {
      monitoring.error('Failed to get call chain:', error.message);
      res.status(500).json({
        success: false,
        error: 'Failed to get call chain',
        details: error.message
      });
    }
  });

  /**
   * GET /api/roslyn/call-chain
   * Get the call chain for a method (query parameter version)
   */
  router.get('/call-chain', async (req, res) => {
    try {
      const { method_name, class_name, project, direction = 'both', max_depth = 3 } = req.query;

      if (!class_name || !method_name) {
        return res.status(400).json({
          success: false,
          error: 'class_name and method_name query parameters are required'
        });
      }

      const params = new URLSearchParams({
        method_name,
        class_name,
        direction,
        max_depth: max_depth.toString()
      });

      if (project) {
        params.append('project', project);
      }

      const response = await fetch(`${PYTHON_SERVICE_URL}/roslyn/call-chain?${params}`);
      const result = await response.json();

      if (!response.ok) {
        return res.status(response.status).json(result);
      }

      res.json(result);

    } catch (error) {
      monitoring.error('Failed to get call chain:', error.message);
      res.status(500).json({
        success: false,
        error: error.message
      });
    }
  });

  /**
   * POST /api/roslyn/search-methods
   * Search for C# methods using semantic similarity
   */
  router.post('/search-methods', async (req, res) => {
    try {
      const { query, project, limit = 10, includeSqlOnly = false } = req.body;

      if (!query) {
        return res.status(400).json({
          success: false,
          error: 'query is required'
        });
      }

      monitoring.info(`Searching methods: "${query}"`);

      const response = await fetch(`${PYTHON_SERVICE_URL}/roslyn/search-methods`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query, project, limit, includeSqlOnly })
      });

      const result = await response.json();

      if (!response.ok) {
        return res.status(response.status).json(result);
      }

      res.json(result);

    } catch (error) {
      monitoring.error('Failed to search methods:', error.message);
      res.status(500).json({
        success: false,
        error: error.message
      });
    }
  });

  /**
   * GET /api/roslyn/search/methods
   * Search for C# methods using semantic similarity (GET version)
   */
  router.get('/search/methods', async (req, res) => {
    try {
      const { query, project, limit = 10, sql_only = false } = req.query;

      if (!query) {
        return res.status(400).json({
          success: false,
          error: 'query parameter is required'
        });
      }

      const params = new URLSearchParams({
        query,
        limit: limit.toString(),
        sql_only: sql_only.toString()
      });

      if (project) {
        params.append('project', project);
      }

      const response = await fetch(`${PYTHON_SERVICE_URL}/roslyn/search/methods?${params}`);
      const result = await response.json();

      if (!response.ok) {
        return res.status(response.status).json(result);
      }

      res.json(result);

    } catch (error) {
      monitoring.error('Failed to search methods:', error.message);
      res.status(500).json({
        success: false,
        error: error.message
      });
    }
  });

  /**
   * GET /api/roslyn/search/classes
   * Search for C# classes using semantic similarity
   */
  router.get('/search/classes', async (req, res) => {
    try {
      const { query, project, limit = 10 } = req.query;

      if (!query) {
        return res.status(400).json({
          success: false,
          error: 'query parameter is required'
        });
      }

      const params = new URLSearchParams({
        query,
        limit: limit.toString()
      });

      if (project) {
        params.append('project', project);
      }

      const response = await fetch(`${PYTHON_SERVICE_URL}/roslyn/search/classes?${params}`);
      const result = await response.json();

      if (!response.ok) {
        return res.status(response.status).json(result);
      }

      res.json(result);

    } catch (error) {
      monitoring.error('Failed to search classes:', error.message);
      res.status(500).json({
        success: false,
        error: error.message
      });
    }
  });

  /**
   * GET /api/roslyn/search/event-handlers
   * Search for event handlers (UI to code mapping)
   */
  router.get('/search/event-handlers', async (req, res) => {
    try {
      const { query, project, limit = 10 } = req.query;

      if (!query) {
        return res.status(400).json({
          success: false,
          error: 'query parameter is required'
        });
      }

      const params = new URLSearchParams({
        query,
        limit: limit.toString()
      });

      if (project) {
        params.append('project', project);
      }

      const response = await fetch(`${PYTHON_SERVICE_URL}/roslyn/search/event-handlers?${params}`);
      const result = await response.json();

      if (!response.ok) {
        return res.status(response.status).json(result);
      }

      res.json(result);

    } catch (error) {
      monitoring.error('Failed to search event handlers:', error.message);
      res.status(500).json({
        success: false,
        error: error.message
      });
    }
  });

  /**
   * GET /api/roslyn/stats
   * Get statistics for all Roslyn collections
   */
  router.get('/stats', async (req, res) => {
    try {
      const response = await fetch(`${PYTHON_SERVICE_URL}/roslyn/stats`);
      const result = await response.json();

      if (!response.ok) {
        return res.status(response.status).json(result);
      }

      res.json(result);

    } catch (error) {
      monitoring.error('Failed to get Roslyn stats:', error.message);
      res.status(500).json({
        success: false,
        error: error.message
      });
    }
  });

  // ============================================================================
  // Targeted and Full Analysis Routes
  // ============================================================================

  /**
   * POST /api/roslyn/analyze-targeted
   * Analyze specific files containing references to modified methods.
   * Finds files via string search, then runs Roslyn analysis only on those files.
   */
  router.post('/analyze-targeted', requireAuth, async (req, res) => {
    try {
      const { methods, repoPath, project } = req.body;

      if (!methods || !Array.isArray(methods) || methods.length === 0) {
        return res.status(400).json({
          success: false,
          error: 'methods array is required'
        });
      }

      if (!repoPath || !project) {
        return res.status(400).json({
          success: false,
          error: 'repoPath and project are required'
        });
      }

      monitoring.info(`Targeted analysis: ${methods.length} methods in ${project}`);

      // Proxy to Python service which will:
      // 1. Search for files containing method references
      // 2. Run Roslyn analyzer on just those files
      // 3. Store call graph results
      const response = await fetch(`${PYTHON_SERVICE_URL}/roslyn/analyze-targeted`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ methods, repo_path: repoPath, project })
      });

      const result = await response.json();

      if (!response.ok) {
        return res.status(response.status).json(result);
      }

      res.json(result);

    } catch (error) {
      monitoring.error('Failed targeted analysis:', error.message);
      res.status(500).json({
        success: false,
        error: 'Failed to run targeted analysis',
        details: error.message
      });
    }
  });

  /**
   * POST /api/roslyn/analyze-full
   * Full repository analysis using Roslyn.
   * Analyzes all C# files and stores complete call graph in MongoDB.
   */
  router.post('/analyze-full', requireAuth, async (req, res) => {
    try {
      const { repoPath, project, generateEmbeddings = true } = req.body;

      if (!repoPath || !project) {
        return res.status(400).json({
          success: false,
          error: 'repoPath and project are required'
        });
      }

      monitoring.info(`Full repository analysis: ${project} at ${repoPath}`);

      // Proxy to Python service for full analysis
      const response = await fetch(`${PYTHON_SERVICE_URL}/roslyn/analyze-full`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          repo_path: repoPath,
          project,
          generate_embeddings: generateEmbeddings
        })
      });

      const result = await response.json();

      if (!response.ok) {
        return res.status(response.status).json(result);
      }

      res.json(result);

    } catch (error) {
      monitoring.error('Failed full analysis:', error.message);
      res.status(500).json({
        success: false,
        error: 'Failed to run full repository analysis',
        details: error.message
      });
    }
  });

  /**
   * GET /api/roslyn/callgraph-status/:project
   * Check if call graph data exists for a project
   */
  router.get('/callgraph-status/:project', async (req, res) => {
    try {
      const { project } = req.params;

      const response = await fetch(`${PYTHON_SERVICE_URL}/roslyn/callgraph-status/${encodeURIComponent(project)}`);
      const result = await response.json();

      if (!response.ok) {
        return res.status(response.status).json(result);
      }

      res.json(result);

    } catch (error) {
      monitoring.error('Failed to get call graph status:', error.message);
      res.status(500).json({
        success: false,
        error: error.message
      });
    }
  });

  // ============================================================================
  // Health Check
  // ============================================================================

  /**
   * GET /api/roslyn/health
   * Check Python Roslyn Service availability
   */
  router.get('/health', async (req, res) => {
    try {
      await checkPythonService();
      res.json({
        success: true,
        pythonServiceAvailable,
        message: pythonServiceAvailable
          ? 'Roslyn service connected to Python backend'
          : 'Python backend unavailable - Roslyn features may not work'
      });
    } catch (error) {
      res.status(503).json({
        success: false,
        pythonServiceAvailable: false,
        error: error.message
      });
    }
  });

  return router;
}
