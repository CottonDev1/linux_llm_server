/**
 * PipelineLogger.js
 *
 * Unified logging system for all pipelines, services, and API calls.
 * Logs are written to files in the logs/ subdirectory with consistent formatting.
 *
 * Format: [PipelineName][User/IP/System] : message
 *
 * Examples:
 * [DocumentPipeline][Admin] : Processing document upload
 * [QueryPipeline][10.1.2.5] : Executing semantic search
 * [SystemStartup][System] : Server initialized
 */

import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Log directory path
const LOG_DIR = path.resolve(__dirname, '../../logs');

// Ensure logs directory exists
if (!fs.existsSync(LOG_DIR)) {
  fs.mkdirSync(LOG_DIR, { recursive: true });
}

/**
 * Available log categories/files
 */
const LOG_CATEGORIES = {
  PIPELINE: 'pipeline.log',
  WEBSERVER: 'webserver.log',
  PYTHON: 'python-service.log',
  API: 'api-access.log',
  ERROR: 'error.log',
  SYSTEM: 'system.log'
};

/**
 * Pipeline names for consistent logging
 */
const PIPELINES = {
  DOCUMENT: 'DocumentPipeline',
  QUERY: 'QueryPipeline',
  EMBEDDING: 'EmbeddingPipeline',
  ROSLYN: 'RoslynPipeline',
  GIT: 'GitPipeline',
  SQL: 'SQLPipeline',
  LLM: 'LLMPipeline',
  AUTH: 'AuthPipeline',
  ADMIN: 'AdminPipeline',
  SYSTEM: 'System',
  WEBSERVER: 'WebServer',
  PYTHON: 'PythonService'
};

/**
 * Format timestamp for log entries
 */
function formatTimestamp() {
  const now = new Date();
  return now.toISOString().replace('T', ' ').substring(0, 19);
}

/**
 * Get user identifier from request or context
 * @param {Object} context - Request object or context with user info
 * @returns {string} User identifier (username, IP, or 'System')
 */
function getUserIdentifier(context) {
  if (!context) return 'System';

  // If it's a request object
  if (context.user && context.user.username) {
    return context.user.username;
  }

  // If it's an Express request, get IP
  if (context.ip || context.connection) {
    let ip = context.ip ||
             context.headers?.['x-forwarded-for']?.split(',')[0] ||
             context.connection?.remoteAddress ||
             context.socket?.remoteAddress;

    // Clean up IPv6 localhost
    if (ip === '::1' || ip === '::ffff:127.0.0.1') {
      ip = '127.0.0.1';
    }

    return ip || 'Unknown';
  }

  // If context has a user string directly
  if (typeof context === 'string') {
    return context;
  }

  return 'System';
}

/**
 * Write log entry to file
 * @param {string} category - Log category (from LOG_CATEGORIES)
 * @param {string} message - Formatted log message
 */
function writeToFile(category, message) {
  const logFile = path.join(LOG_DIR, category);
  const timestamp = formatTimestamp();
  const logEntry = `${timestamp} ${message}\n`;

  fs.appendFile(logFile, logEntry, (err) => {
    if (err) {
      console.error(`Failed to write to ${category}:`, err.message);
    }
  });
}

/**
 * PipelineLogger class
 */
class PipelineLogger {
  constructor(pipelineName) {
    this.pipelineName = pipelineName || 'Unknown';
  }

  /**
   * Format a log message with pipeline and user info
   * @param {string} user - User identifier
   * @param {string} message - Log message
   * @returns {string} Formatted message
   */
  formatMessage(user, message) {
    return `[${this.pipelineName}][${user}] : ${message}`;
  }

  /**
   * Log an info message
   * @param {string} message - Log message
   * @param {Object} context - Request or user context
   */
  info(message, context = null) {
    const user = getUserIdentifier(context);
    const formatted = this.formatMessage(user, message);

    console.log(formatted);
    writeToFile(LOG_CATEGORIES.PIPELINE, formatted);

    // Also write to webserver log if it's a web-related pipeline
    if (['WebServer', 'QueryPipeline', 'AdminPipeline'].includes(this.pipelineName)) {
      writeToFile(LOG_CATEGORIES.WEBSERVER, formatted);
    }
  }

  /**
   * Log an error message
   * @param {string} message - Error message
   * @param {Error} error - Error object
   * @param {Object} context - Request or user context
   */
  error(message, error = null, context = null) {
    const user = getUserIdentifier(context);
    const errorDetails = error ? ` | Error: ${error.message}` : '';
    const formatted = this.formatMessage(user, `ERROR: ${message}${errorDetails}`);

    console.error(formatted);
    writeToFile(LOG_CATEGORIES.PIPELINE, formatted);
    writeToFile(LOG_CATEGORIES.ERROR, formatted);

    // Log stack trace to error log
    if (error && error.stack) {
      writeToFile(LOG_CATEGORIES.ERROR, `  Stack: ${error.stack}`);
    }
  }

  /**
   * Log a warning message
   * @param {string} message - Warning message
   * @param {Object} context - Request or user context
   */
  warn(message, context = null) {
    const user = getUserIdentifier(context);
    const formatted = this.formatMessage(user, `WARN: ${message}`);

    console.warn(formatted);
    writeToFile(LOG_CATEGORIES.PIPELINE, formatted);
  }

  /**
   * Log an API access
   * @param {string} method - HTTP method
   * @param {string} path - Request path
   * @param {number} statusCode - Response status code
   * @param {number} duration - Request duration in ms
   * @param {Object} req - Express request object
   */
  logApiAccess(method, path, statusCode, duration, req) {
    const user = getUserIdentifier(req);
    const formatted = `[API][${user}] : ${method} ${path} ${statusCode} ${duration}ms`;

    writeToFile(LOG_CATEGORIES.API, formatted);
    writeToFile(LOG_CATEGORIES.WEBSERVER, formatted);
  }
}

/**
 * Create a logger for a specific pipeline
 * @param {string} pipelineName - Name of the pipeline
 * @returns {PipelineLogger} Logger instance
 */
export function createPipelineLogger(pipelineName) {
  return new PipelineLogger(pipelineName);
}

/**
 * Pre-configured loggers for common pipelines
 */
export const loggers = {
  document: new PipelineLogger(PIPELINES.DOCUMENT),
  query: new PipelineLogger(PIPELINES.QUERY),
  embedding: new PipelineLogger(PIPELINES.EMBEDDING),
  roslyn: new PipelineLogger(PIPELINES.ROSLYN),
  git: new PipelineLogger(PIPELINES.GIT),
  sql: new PipelineLogger(PIPELINES.SQL),
  llm: new PipelineLogger(PIPELINES.LLM),
  auth: new PipelineLogger(PIPELINES.AUTH),
  admin: new PipelineLogger(PIPELINES.ADMIN),
  system: new PipelineLogger(PIPELINES.SYSTEM),
  webserver: new PipelineLogger(PIPELINES.WEBSERVER),
  python: new PipelineLogger(PIPELINES.PYTHON)
};

/**
 * System-level logging functions
 */
export const systemLog = {
  startup: (message) => {
    const formatted = `[SystemStartup][System] : ${message}`;
    console.log(formatted);
    writeToFile(LOG_CATEGORIES.SYSTEM, formatted);
    writeToFile(LOG_CATEGORIES.WEBSERVER, formatted);
  },

  shutdown: (message) => {
    const formatted = `[SystemShutdown][System] : ${message}`;
    console.log(formatted);
    writeToFile(LOG_CATEGORIES.SYSTEM, formatted);
    writeToFile(LOG_CATEGORIES.WEBSERVER, formatted);
  },

  error: (message, error = null) => {
    const errorDetails = error ? ` | ${error.message}` : '';
    const formatted = `[SystemError][System] : ${message}${errorDetails}`;
    console.error(formatted);
    writeToFile(LOG_CATEGORIES.SYSTEM, formatted);
    writeToFile(LOG_CATEGORIES.ERROR, formatted);
  }
};

/**
 * Express middleware for API access logging
 */
export function apiAccessLogger(req, res, next) {
  const startTime = Date.now();

  res.on('finish', () => {
    const duration = Date.now() - startTime;
    const logger = new PipelineLogger('API');
    logger.logApiAccess(req.method, req.originalUrl, res.statusCode, duration, req);
  });

  next();
}

/**
 * Get available log files
 * @returns {Array} List of log file info
 */
export function getLogFiles() {
  try {
    const files = fs.readdirSync(LOG_DIR);
    return files
      .filter(f => f.endsWith('.log'))
      .map(f => {
        const filePath = path.join(LOG_DIR, f);
        const stats = fs.statSync(filePath);
        return {
          name: f,
          size: stats.size,
          modified: stats.mtime,
          category: f.replace('.log', '')
        };
      });
  } catch (error) {
    return [];
  }
}

/**
 * Read log file contents with optional filtering
 * @param {string} category - Log category name
 * @param {Object} options - Read options
 * @param {number} options.lines - Number of lines to return (default: 1000)
 * @param {string} options.search - Search term to filter by
 * @param {string} options.level - Log level to filter by (ERROR, WARN, INFO)
 * @returns {Object} Log contents and metadata
 */
export function readLogFile(category, options = {}) {
  const { lines = 1000, search = '', level = '' } = options;
  const logFile = path.join(LOG_DIR, `${category}.log`);

  try {
    if (!fs.existsSync(logFile)) {
      return { success: false, error: 'Log file not found', lines: [] };
    }

    const content = fs.readFileSync(logFile, 'utf8');
    let logLines = content.split('\n').filter(line => line.trim());

    // Filter by search term
    if (search) {
      const searchLower = search.toLowerCase();
      logLines = logLines.filter(line => line.toLowerCase().includes(searchLower));
    }

    // Filter by level
    if (level) {
      logLines = logLines.filter(line => line.includes(level.toUpperCase()));
    }

    // Get last N lines
    const totalLines = logLines.length;
    logLines = logLines.slice(-lines);

    return {
      success: true,
      category,
      totalLines,
      returnedLines: logLines.length,
      lines: logLines
    };
  } catch (error) {
    return { success: false, error: error.message, lines: [] };
  }
}

/**
 * Clear a log file
 * @param {string} category - Log category name
 * @returns {Object} Result
 */
export function clearLogFile(category) {
  const logFile = path.join(LOG_DIR, `${category}.log`);

  try {
    fs.writeFileSync(logFile, '');
    return { success: true, message: `${category}.log cleared` };
  } catch (error) {
    return { success: false, error: error.message };
  }
}

export { LOG_CATEGORIES, PIPELINES, LOG_DIR };
export default PipelineLogger;
