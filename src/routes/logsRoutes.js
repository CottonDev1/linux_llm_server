/**
 * logsRoutes.js
 *
 * API endpoints for accessing and managing log files.
 * Provides read, search, and clear functionality for admin UI.
 */

import express from 'express';
import { getLogFiles, readLogFile, clearLogFile, LOG_CATEGORIES } from '../logging/PipelineLogger.js';

/**
 * Create logs routes
 * @param {Object} dependencies - Route dependencies
 * @returns {express.Router} Express router
 */
export default function createLogsRoutes(dependencies = {}) {
  const router = express.Router();
  const { requireAuthOrAdmin } = dependencies;

  // Apply auth middleware if provided
  if (requireAuthOrAdmin) {
    router.use(requireAuthOrAdmin);
  }

  /**
   * GET /api/logs
   * Get list of available log files
   */
  router.get('/', (req, res) => {
    try {
      const files = getLogFiles();
      const categories = Object.entries(LOG_CATEGORIES).map(([key, filename]) => ({
        key: key.toLowerCase(),
        name: key,
        filename,
        displayName: key.charAt(0) + key.slice(1).toLowerCase()
      }));

      res.json({
        success: true,
        categories,
        files
      });
    } catch (error) {
      res.status(500).json({
        success: false,
        error: error.message
      });
    }
  });

  /**
   * GET /api/logs/:category
   * Read log file contents with optional filtering
   *
   * Query params:
   * - lines: Number of lines to return (default: 500)
   * - search: Search term to filter by
   * - level: Log level filter (ERROR, WARN, INFO)
   */
  router.get('/:category', (req, res) => {
    try {
      const { category } = req.params;
      const {
        lines = 500,
        search = '',
        level = ''
      } = req.query;

      const result = readLogFile(category, {
        lines: parseInt(lines, 10),
        search,
        level
      });

      res.json(result);
    } catch (error) {
      res.status(500).json({
        success: false,
        error: error.message
      });
    }
  });

  /**
   * DELETE /api/logs/:category
   * Clear a log file
   */
  router.delete('/:category', (req, res) => {
    try {
      const { category } = req.params;
      const result = clearLogFile(category);
      res.json(result);
    } catch (error) {
      res.status(500).json({
        success: false,
        error: error.message
      });
    }
  });

  /**
   * GET /api/logs/:category/download
   * Download a log file
   */
  router.get('/:category/download', (req, res) => {
    try {
      const { category } = req.params;
      const result = readLogFile(category, { lines: 100000 });

      if (!result.success) {
        return res.status(404).json(result);
      }

      const content = result.lines.join('\n');
      res.setHeader('Content-Type', 'text/plain');
      res.setHeader('Content-Disposition', `attachment; filename="${category}.log"`);
      res.send(content);
    } catch (error) {
      res.status(500).json({
        success: false,
        error: error.message
      });
    }
  });

  /**
   * POST /api/logs/:category/search
   * Advanced search in log file
   *
   * Body:
   * - query: Search query
   * - startDate: Start date filter
   * - endDate: End date filter
   * - pipeline: Pipeline name filter
   * - user: User/IP filter
   */
  router.post('/:category/search', (req, res) => {
    try {
      const { category } = req.params;
      const { query, startDate, endDate, pipeline, user } = req.body;

      let result = readLogFile(category, { lines: 10000 });

      if (!result.success) {
        return res.json(result);
      }

      let filtered = result.lines;

      // Filter by query
      if (query) {
        const queryLower = query.toLowerCase();
        filtered = filtered.filter(line => line.toLowerCase().includes(queryLower));
      }

      // Filter by pipeline
      if (pipeline) {
        filtered = filtered.filter(line => line.includes(`[${pipeline}]`));
      }

      // Filter by user
      if (user) {
        filtered = filtered.filter(line => line.includes(`[${user}]`));
      }

      // Filter by date range
      if (startDate || endDate) {
        filtered = filtered.filter(line => {
          const match = line.match(/^(\d{4}-\d{2}-\d{2})/);
          if (!match) return true;

          const lineDate = match[1];
          if (startDate && lineDate < startDate) return false;
          if (endDate && lineDate > endDate) return false;
          return true;
        });
      }

      res.json({
        success: true,
        category,
        totalMatches: filtered.length,
        lines: filtered.slice(-500)
      });
    } catch (error) {
      res.status(500).json({
        success: false,
        error: error.message
      });
    }
  });

  return router;
}
