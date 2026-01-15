/**
 * Deprecated: SQL pipeline is handled by the Python FastAPI service.
 * This module is intentionally minimal to avoid any Node-based SQL operations.
 */
import express from 'express';

const router = express.Router();

export default function initDatabaseRoutes() {
  // No-op: all database and SQL-related functionality lives in Python.
  return router;
}
