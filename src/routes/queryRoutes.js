/**
 * Query Routes Module - Thin Proxy Layer
 * ======================================
 *
 * This module serves as a lightweight HTTP proxy to the Python FastAPI service.
 * ALL business logic for document querying and RAG has been moved to Python.
 *
 * Responsibilities (Node.js layer):
 * - HTTP routing and request validation
 * - SSE streaming setup and connection management
 * - Authentication/authorization checks (if configured)
 * - Proxying requests to Python service
 *
 * Business Logic (Python service at localhost:8001):
 * - Vector search and retrieval
 * - LLM query generation and orchestration
 * - Context building and prompt engineering
 * - Response caching
 * - Document processing and embedding
 *
 * Routes:
 * - POST /search - Direct vector search without LLM
 * - POST /query - RAG query with LLM generation
 * - POST /query/stream - Streaming RAG query (SSE)
 * - GET /projects - Available projects list
 *
 * Migration Notes:
 * - Reduced from 624 lines to ~150 lines
 * - Removed all LLM calls, MongoDB operations, context building
 * - Maintained API contract for backward compatibility
 * - All business logic delegated to Python FastAPI service
 */

import express from 'express';

const router = express.Router();

/**
 * Create query routes with dependency injection
 * @param {Object} deps - Dependencies
 * @param {string} deps.pythonServiceUrl - Python service URL (default: http://localhost:8001)
 * @param {Object} deps.monitoring - Monitoring instance (optional)
 * @returns {express.Router} Express router
 */
export default function createQueryRoutes(deps) {
  const {
    pythonServiceUrl = 'http://localhost:8001',
    monitoring
  } = deps;

  /**
   * POST /api/search
   * Direct vector search without LLM processing
   * Proxies to: POST /api/documents/search
   */
  router.post('/search', async (req, res) => {
    try {
      const { query, project, limit = 10 } = req.body;

      // Basic validation
      if (!query) {
        return res.status(400).json({ error: 'Query is required' });
      }

      monitoring?.info?.(`Direct search proxy: "${query}"`, { project, limit });

      // Proxy to Python service
      const pythonResponse = await fetch(`${pythonServiceUrl}/api/documents/search`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query, project, limit })
      });

      if (!pythonResponse.ok) {
        const errorText = await pythonResponse.text();
        throw new Error(`Python service error: ${pythonResponse.status} - ${errorText}`);
      }

      const result = await pythonResponse.json();
      res.json(result);

    } catch (error) {
      monitoring?.error?.('Search proxy failed', error, { type: 'search', severity: 'medium' });
      res.status(500).json({
        error: 'Search failed',
        details: error.message
      });
    }
  });

  /**
   * POST /api/query
   * Main RAG query endpoint: vector search + LLM generation
   * Proxies to: POST /api/documents/query
   */
  router.post('/query', async (req, res) => {
    try {
      const {
        query,
        project,
        limit = 10,
        includeEWRLibrary = false,
        history = [],
        model
      } = req.body;

      // Basic validation
      if (!query) {
        return res.status(400).json({ error: 'Query is required' });
      }

      monitoring?.info?.(`Query proxy: "${query}"`, {
        project,
        includeEWRLibrary,
        historyLength: history.length
      });

      // Proxy to Python service with all parameters
      const pythonResponse = await fetch(`${pythonServiceUrl}/api/documents/query`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query,
          project,
          limit,
          includeEWRLibrary,
          history,
          model
        })
      });

      if (!pythonResponse.ok) {
        const errorText = await pythonResponse.text();
        throw new Error(`Python service error: ${pythonResponse.status} - ${errorText}`);
      }

      const result = await pythonResponse.json();
      res.json(result);

    } catch (error) {
      monitoring?.error?.('Query proxy failed', error, {
        query: req.body.query,
        project: req.body.project,
        type: 'query',
        severity: 'high'
      });
      res.status(500).json({
        error: 'Query failed',
        details: error.message
      });
    }
  });

  /**
   * POST /api/query/stream
   * Streaming RAG query endpoint using Server-Sent Events (SSE)
   * Proxies to: POST /api/documents/query-stream
   */
  router.post('/query/stream', async (req, res) => {
    let pythonResponse;

    try {
      const { query, project, limit = 10, model, history = [] } = req.body;

      // Basic validation
      if (!query) {
        return res.status(400).json({ error: 'Query is required' });
      }

      monitoring?.info?.(`Streaming query proxy: "${query}"`, { project, model });

      // Set up SSE headers
      res.setHeader('Content-Type', 'text/event-stream');
      res.setHeader('Cache-Control', 'no-cache');
      res.setHeader('Connection', 'keep-alive');

      // Handle client disconnect
      let clientDisconnected = false;
      req.on('close', () => {
        clientDisconnected = true;
        monitoring?.info?.('Client disconnected from stream');
      });

      // Proxy to Python service (streaming endpoint)
      pythonResponse = await fetch(`${pythonServiceUrl}/api/documents/query-stream`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query,
          project,
          limit,
          model,
          history
        })
      });

      if (!pythonResponse.ok) {
        const errorText = await pythonResponse.text();
        throw new Error(`Python service error: ${pythonResponse.status} - ${errorText}`);
      }

      // Stream the response from Python to client
      // Python service sends SSE-formatted data, we just pass it through
      for await (const chunk of pythonResponse.body) {
        if (clientDisconnected) {
          monitoring?.info?.('Stopping stream due to client disconnect');
          break;
        }

        // Pass through the SSE data chunk from Python
        res.write(chunk);
      }

      res.end();

    } catch (error) {
      monitoring?.error?.('Stream proxy failed', error, {
        query: req.body?.query,
        type: 'stream',
        severity: 'high'
      });

      if (!res.headersSent) {
        res.writeHead(500, { 'Content-Type': 'text/event-stream' });
      }

      // Send error as SSE event
      res.write(`data: ${JSON.stringify({
        type: 'error',
        error: error.message
      })}\n\n`);
      res.end();
    }
  });

  /**
   * GET /api/projects
   * Get available projects list
   * Proxies to: GET /api/documents/projects
   * Falls back to static list if Python endpoint doesn't exist
   */
  router.get('/projects', async (req, res) => {
    try {
      // Try to proxy to Python service first
      const pythonResponse = await fetch(`${pythonServiceUrl}/api/documents/projects`);

      if (!pythonResponse.ok) {
        // If Python service doesn't have this endpoint, return static list
        if (pythonResponse.status === 404) {
          monitoring?.info?.('Using static projects list (Python endpoint not found)');
          return res.json({
            projects: [
              { id: 'all', name: 'All Projects', description: 'Search across all projects' },
              { id: 'gin', name: 'Gin', description: 'Cotton Gin application' },
              { id: 'EWRLibrary', name: 'EWR Library', description: 'EWR shared library' },
              { id: 'warehouse', name: 'Warehouse', description: 'Warehouse management' },
              { id: 'marketing', name: 'Marketing', description: 'Marketing application' },
              { id: 'knowledge_base', name: 'Knowledge Base', description: 'EWR Documentation' }
            ]
          });
        }

        const errorText = await pythonResponse.text();
        throw new Error(`Python service error: ${pythonResponse.status} - ${errorText}`);
      }

      const result = await pythonResponse.json();
      res.json(result);

    } catch (error) {
      monitoring?.error?.('Projects proxy failed', error, { type: 'projects', severity: 'low' });
      res.status(500).json({
        error: 'Failed to fetch projects',
        details: error.message
      });
    }
  });

  return router;
}
