/**
 * Health Check and Monitoring Routes
 *
 * This module contains all health check and monitoring endpoints for the RAG Server.
 * Routes are exported as a factory function that accepts dependencies via dependency injection.
 *
 * Extracted Routes:
 * - GET /api/health - Basic health check (fetches from Python /status endpoint)
 * - GET /metrics - Prometheus metrics endpoint
 * - GET /api/health/detailed - Detailed health check with service layer status
 *
 * Note: All status is fetched from Python service's single /status endpoint
 */

import express from 'express';

/**
 * Create health routes with injected dependencies
 *
 * @param {Object} deps - Dependencies object
 * @param {Object} deps.monitoring - MonitoringService instance
 * @param {boolean} deps.pythonServiceInitialized - Python service initialization status
 * @param {string} deps.pythonServiceUrl - Python service URL
 * @returns {express.Router} Express router with health routes
 */
export default function createHealthRoutes(deps) {
  const router = express.Router();

  const {
    monitoring,
    pythonServiceInitialized,
    pythonServiceUrl
  } = deps;

  /**
   * Basic health check endpoint
   *
   * Fetches complete status from Python service's /status endpoint
   *
   * @route GET /api/health
   */
  router.get('/health', async (req, res) => {
    try {
      // Get base health check from monitoring system
      const health = await monitoring.checkHealth();

      // Fetch complete status from Python service's single /status endpoint
      let dbHealth = pythonServiceInitialized;
      let mongodbStatus = 'disconnected';
      let llmHealth = false;

      try {
        const controller = new AbortController();
        const timeout = setTimeout(() => controller.abort(), 5000);

        const response = await fetch(`${pythonServiceUrl || 'http://localhost:8001'}/status`, {
          signal: controller.signal
        });
        clearTimeout(timeout);

        if (response.ok) {
          const status = await response.json();

          // MongoDB status
          dbHealth = status.mongodb?.connected === true;
          mongodbStatus = dbHealth ? 'connected' : 'disconnected';
          monitoring.updateHealthStatus('database', dbHealth);

          // LLM status
          llmHealth = status.llm?.healthy === true;
          monitoring.updateHealthStatus('llm', llmHealth);

          // Add Python status details
          health.pythonStatus = status;
        }
      } catch (error) {
        if (error.name === 'AbortError') {
          monitoring.warn('Python service status check timeout after 5s');
        } else {
          monitoring.warn('Python service status check failed', { error: error.message });
        }
        monitoring.updateHealthStatus('database', false);
        monitoring.updateHealthStatus('llm', false);
      }

      health.checks.pythonService = {
        status: dbHealth ? 'up' : 'down',
        url: pythonServiceUrl || 'http://localhost:8001'
      };

      health.database = mongodbStatus;
      health.llm = llmHealth ? 'connected' : 'disconnected';
      health.pythonServiceUrl = pythonServiceUrl || 'http://localhost:8001';
      health.llmBackend = llmHealth ? 'connected' : 'offline';

      res.json(health);
    } catch (error) {
      monitoring.error('Health check failed', error);
      res.status(500).json({ status: 'error', message: error.message });
    }
  });

  /**
   * Prometheus metrics endpoint
   *
   * Returns metrics in Prometheus format for monitoring and alerting.
   *
   * @route GET /metrics
   */
  router.get('/metrics', async (req, res) => {
    try {
      res.set('Content-Type', 'text/plain');
      res.send(await monitoring.getMetrics());
    } catch (error) {
      monitoring.error('Failed to generate metrics', error);
      res.status(500).send('# Error generating metrics');
    }
  });

  /**
   * Detailed system health endpoint with service layer status
   *
   * Provides comprehensive health information including:
   * - Overall system status
   * - Python service (MongoDB) status
   * - LLM backend status (llama.cpp)
   * - Cache status
   *
   * @route GET /api/health/detailed
   */
  router.get('/health/detailed', async (req, res) => {
    try {
      let health = {
        status: 'operational',
        timestamp: new Date().toISOString(),
        services: {}
      };

      // Fetch complete status from Python service's /status endpoint
      try {
        const response = await fetch(`${pythonServiceUrl || 'http://localhost:8001'}/status`);
        if (response.ok) {
          const status = await response.json();
          health.services.pythonService = 'connected';
          health.services.mongodb = status.mongodb?.connected ? 'connected' : 'disconnected';
          health.services.embeddingModel = status.embeddings?.loaded ? 'loaded' : 'not-loaded';
          health.services.vectorSearch = status.vector_search?.native_available ? 'available' : 'unavailable';
          health.services.llm = status.llm?.healthy ? 'connected' : 'disconnected';
          health.collections = status.collections;
          health.llmEndpoints = status.llm?.endpoints;
          health.checkedAt = status.checked_at;
        } else {
          health.services.pythonService = 'error';
          health.services.mongodb = 'unknown';
          health.services.llm = 'unknown';
        }
      } catch (error) {
        health.services.pythonService = 'disconnected';
        health.services.mongodb = 'unknown';
        health.services.llm = 'unknown';
      }

      res.json(health);
    } catch (error) {
      res.status(500).json({
        status: 'degraded',
        error: error.message,
        timestamp: new Date().toISOString()
      });
    }
  });

  return router;
}
