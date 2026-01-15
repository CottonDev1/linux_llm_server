/**
 * ProductionMonitoring.js
 *
 * Comprehensive monitoring system for the RAG server using Prometheus metrics
 * and Winston structured logging. Tracks query performance, cache efficiency,
 * database connections, and system health.
 *
 * Architecture:
 * - Prometheus metrics (prom-client): Exposes metrics for scraping
 * - Winston logging: Structured JSON logs with file rotation
 * - Health checks: System component health monitoring
 * - Error tracking: Structured error logging with stack traces
 *
 * Key Metrics:
 * - rag_query_duration_seconds: Query processing time histogram
 * - rag_query_total: Total queries processed counter
 * - rag_embedding_cache_hits_total: Embedding cache hit/miss counter
 * - rag_response_cache_hits_total: Response cache hit/miss counter
 * - rag_db_active_connections: Active database connections gauge
 * - rag_relevance_score: Result relevance score histogram
 * - rag_llm_request_duration_seconds: LLM request time histogram
 * - rag_llm_tokens_total: Token usage counter
 * - rag_vector_search_ops_total: Vector search operations counter
 * - rag_documents_total: Document count gauge
 * - rag_errors_total: Error counter by type/severity
 *
 * @module ProductionMonitoring
 */

import promClient from 'prom-client';
import winston from 'winston';
import path from 'path';
import { fileURLToPath } from 'url';
import fs from 'fs';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

/**
 * ProductionMonitoring class
 *
 * Provides comprehensive monitoring capabilities for the RAG system including:
 * - Prometheus metrics collection and exposure
 * - Structured logging with Winston
 * - Health check endpoints
 * - Performance tracking
 * - Error tracking and alerting
 */
class ProductionMonitoring {
  /**
   * Initialize the monitoring system
   *
   * @param {Object} config - Configuration options
   * @param {string} config.logLevel - Winston log level (default: 'info')
   * @param {string} config.logDir - Directory for log files (default: './logs')
   * @param {boolean} config.metricsEnabled - Enable Prometheus metrics (default: true)
   * @param {number} config.metricsPort - Port for metrics endpoint (default: 9090)
   * @param {boolean} config.consoleLogging - Enable console logging (default: true)
   * @param {number} config.logMaxSize - Max log file size before rotation (default: 10MB)
   * @param {number} config.logMaxFiles - Max log files to keep (default: 5)
   */
  constructor(config = {}) {
    this.config = {
      logLevel: process.env.LOG_LEVEL || 'info',
      logDir: process.env.LOG_DIR || './logs',
      metricsEnabled: process.env.METRICS_ENABLED !== 'false',
      metricsPort: parseInt(process.env.METRICS_PORT || '9090', 10),
      consoleLogging: process.env.CONSOLE_LOGGING !== 'false',
      logMaxSize: 10 * 1024 * 1024, // 10MB
      logMaxFiles: 5,
      ...config
    };

    // Ensure log directory exists
    this.ensureLogDirectory();

    // Initialize components
    this.metrics = {};
    this.startTime = Date.now();

    // System health status (initialize BEFORE logging)
    this.healthStatus = {
      logging: false,
      metrics: this.config.metricsEnabled,
      database: false,
      llm: false
    };

    if (this.config.metricsEnabled) {
      this.initializeMetrics();
    }

    this.initializeLogging();
  }

  /**
   * Ensure log directory exists
   * @private
   */
  ensureLogDirectory() {
    const logDir = path.resolve(this.config.logDir);
    if (!fs.existsSync(logDir)) {
      fs.mkdirSync(logDir, { recursive: true });
    }
  }

  /**
   * Initialize Prometheus metrics
   *
   * Creates histograms, counters, and gauges for tracking:
   * - Query performance (duration, success/failure rates)
   * - Cache efficiency (hit/miss rates)
   * - Database connections
   * - Result quality (relevance scores)
   * - System resources (CPU, memory via default metrics)
   *
   * @private
   */
  initializeMetrics() {
    try {
      // Clear existing registry to prevent duplicate metrics
      promClient.register.clear();

      // Query duration histogram
      // Buckets: 100ms, 500ms, 1s, 2s, 5s, 10s
      this.metrics.queryDuration = new promClient.Histogram({
        name: 'rag_query_duration_seconds',
        help: 'Query processing time in seconds',
        labelNames: ['project', 'status', 'query_type'],
        buckets: [0.1, 0.5, 1, 2, 5, 10]
      });

      // Total queries counter
      this.metrics.queryTotal = new promClient.Counter({
        name: 'rag_query_total',
        help: 'Total number of queries processed',
        labelNames: ['project', 'status', 'query_type']
      });

      // Embedding cache performance
      this.metrics.embeddingCacheHits = new promClient.Counter({
        name: 'rag_embedding_cache_hits_total',
        help: 'Number of embedding cache hits/misses',
        labelNames: ['type'] // 'hit' or 'miss'
      });

      // Response cache performance
      this.metrics.responseCacheHits = new promClient.Counter({
        name: 'rag_response_cache_hits_total',
        help: 'Number of response cache hits/misses',
        labelNames: ['type'] // 'hit' or 'miss'
      });

      // Active database connections
      this.metrics.dbActiveConnections = new promClient.Gauge({
        name: 'rag_db_active_connections',
        help: 'Number of active database connections'
      });

      // Result relevance quality
      this.metrics.relevanceScore = new promClient.Histogram({
        name: 'rag_relevance_score',
        help: 'Average relevance score of search results',
        labelNames: ['project'],
        buckets: [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
      });

      // LLM request duration
      this.metrics.llmDuration = new promClient.Histogram({
        name: 'rag_llm_request_duration_seconds',
        help: 'LLM request processing time in seconds',
        labelNames: ['model', 'status'],
        buckets: [0.5, 1, 2, 5, 10, 20, 30]
      });

      // LLM token usage
      this.metrics.llmTokens = new promClient.Counter({
        name: 'rag_llm_tokens_total',
        help: 'Total tokens consumed by LLM requests',
        labelNames: ['model', 'type'] // 'input' or 'output'
      });

      // Vector search operations
      this.metrics.vectorSearchOps = new promClient.Counter({
        name: 'rag_vector_search_ops_total',
        help: 'Total vector search operations',
        labelNames: ['table', 'status']
      });

      // Document count in vector DB
      this.metrics.documentCount = new promClient.Gauge({
        name: 'rag_documents_total',
        help: 'Total number of documents in vector database',
        labelNames: ['table', 'project']
      });

      // Error counter
      this.metrics.errors = new promClient.Counter({
        name: 'rag_errors_total',
        help: 'Total number of errors by type',
        labelNames: ['type', 'severity']
      });

      // Register default metrics (CPU, memory, event loop lag, etc.)
      promClient.collectDefaultMetrics({
        prefix: 'rag_',
        gcDurationBuckets: [0.001, 0.01, 0.1, 1, 2, 5]
      });

      if (this.logger) {
        this.info('Prometheus metrics initialized successfully', {
          metricsCount: Object.keys(this.metrics).length
        });
      }
    } catch (error) {
      console.error('Failed to initialize Prometheus metrics:', error);
      this.healthStatus.metrics = false;
    }
  }

  /**
   * Initialize Winston logging
   *
   * Creates structured JSON logging with:
   * - Separate error log file (error.log)
   * - Combined log file (combined.log)
   * - Console output with colorization
   * - Timestamp and stack trace capture
   * - Log rotation based on size
   *
   * @private
   */
  initializeLogging() {
    try {
      const transports = [];

      // File transport for errors only
      transports.push(
        new winston.transports.File({
          filename: path.join(this.config.logDir, 'error.log'),
          level: 'error',
          maxsize: this.config.logMaxSize,
          maxFiles: this.config.logMaxFiles,
          tailable: true
        })
      );

      // File transport for all logs
      transports.push(
        new winston.transports.File({
          filename: path.join(this.config.logDir, 'combined.log'),
          maxsize: this.config.logMaxSize,
          maxFiles: this.config.logMaxFiles,
          tailable: true
        })
      );

      // Console transport (plain text only)
      if (this.config.consoleLogging) {
        transports.push(
          new winston.transports.Console({
            format: winston.format.printf(({ message }) => message)
          })
        );
      }

      this.logger = winston.createLogger({
        level: this.config.logLevel,
        format: winston.format.combine(
          winston.format.timestamp({ format: 'YYYY-MM-DD HH:mm:ss' }),
          winston.format.errors({ stack: true }),
          winston.format.metadata(),
          winston.format.json()
        ),
        transports,
        exitOnError: false
      });

      this.healthStatus.logging = true;
      this.logger.info('Winston logging initialized successfully', {
        logLevel: this.config.logLevel,
        logDir: this.config.logDir,
        consoleLogging: this.config.consoleLogging
      });
    } catch (error) {
      console.error('Failed to initialize Winston logging:', error);
      this.healthStatus.logging = false;

      // Fallback to console logging
      this.logger = {
        info: (...args) => console.log('[INFO]', ...args),
        warn: (...args) => console.warn('[WARN]', ...args),
        error: (...args) => console.error('[ERROR]', ...args),
        debug: (...args) => console.debug('[DEBUG]', ...args)
      };
    }
  }

  /**
   * Track query execution metrics
   *
   * @param {string} project - Project name (gin, warehouse, etc.)
   * @param {number} duration - Query duration in seconds
   * @param {string} status - Query status ('success' or 'error')
   * @param {string} queryType - Type of query ('semantic', 'code_flow', 'method_lookup', etc.)
   */
  trackQuery(project, duration, status = 'success', queryType = 'semantic') {
    if (!this.config.metricsEnabled) return;

    try {
      this.metrics.queryDuration.labels(project, status, queryType).observe(duration);
      this.metrics.queryTotal.labels(project, status, queryType).inc();

      this.debug('Query tracked', {
        project,
        duration,
        status,
        queryType
      });
    } catch (error) {
      this.error('Failed to track query metric', error);
    }
  }

  /**
   * Track cache hit/miss events
   *
   * @param {string} cacheType - Type of cache ('embedding' or 'response')
   * @param {boolean} hit - Whether it was a cache hit (true) or miss (false)
   */
  trackCacheHit(cacheType, hit) {
    if (!this.config.metricsEnabled) return;

    try {
      const label = hit ? 'hit' : 'miss';

      if (cacheType === 'embedding') {
        this.metrics.embeddingCacheHits.labels(label).inc();
      } else if (cacheType === 'response') {
        this.metrics.responseCacheHits.labels(label).inc();
      }

      this.debug('Cache event tracked', {
        cacheType,
        result: label
      });
    } catch (error) {
      this.error('Failed to track cache metric', error);
    }
  }

  /**
   * Update active database connections count
   *
   * @param {number} count - Current number of active connections
   */
  updateConnectionCount(count) {
    if (!this.config.metricsEnabled) return;

    try {
      this.metrics.dbActiveConnections.set(count);

      this.debug('Connection count updated', { count });
    } catch (error) {
      this.error('Failed to update connection count', error);
    }
  }

  /**
   * Track result relevance score
   *
   * @param {number} score - Relevance score (0.0 to 1.0)
   * @param {string} project - Project name
   */
  trackRelevance(score, project = 'unknown') {
    if (!this.config.metricsEnabled) return;

    try {
      this.metrics.relevanceScore.labels(project).observe(score);

      this.debug('Relevance score tracked', {
        score,
        project
      });
    } catch (error) {
      this.error('Failed to track relevance score', error);
    }
  }

  /**
   * Track LLM request metrics
   *
   * @param {string} model - Model name (e.g., 'qwen2.5-coder:1.5b')
   * @param {number} duration - Request duration in seconds
   * @param {string} status - Request status ('success' or 'error')
   * @param {number} inputTokens - Number of input tokens
   * @param {number} outputTokens - Number of output tokens
   */
  trackLLMRequest(model, duration, status = 'success', inputTokens = 0, outputTokens = 0) {
    if (!this.config.metricsEnabled) return;

    try {
      this.metrics.llmDuration.labels(model, status).observe(duration);
      this.metrics.llmTokens.labels(model, 'input').inc(inputTokens);
      this.metrics.llmTokens.labels(model, 'output').inc(outputTokens);

      this.debug('LLM request tracked', {
        model,
        duration,
        status,
        inputTokens,
        outputTokens
      });
    } catch (error) {
      this.error('Failed to track LLM request', error);
    }
  }

  /**
   * Track vector search operation
   *
   * @param {string} table - Table name (classes, methods, etc.)
   * @param {string|number} durationOrStatus - Duration in ms or status string ('success' or 'error')
   * @param {number} resultCount - Number of results returned (optional)
   */
  trackVectorSearch(table, durationOrStatus = 'success', resultCount = 0) {
    if (!this.config.metricsEnabled) return;

    try {
      // Handle both old API (table, status) and new API (table, duration, count)
      const status = typeof durationOrStatus === 'string' ? durationOrStatus : 'success';
      this.metrics.vectorSearchOps.labels(table, status).inc();

      this.debug('Vector search tracked', {
        table,
        status,
        resultCount
      });
    } catch (error) {
      this.error('Failed to track vector search', error);
    }
  }

  /**
   * Track intent classification operation
   *
   * @param {Object} data - Classification data
   * @param {string} data.query - Original query
   * @param {Array} data.intents - Classified intents
   * @param {number} data.confidence - Overall confidence score
   * @param {number} data.duration - Classification duration in ms
   * @param {string} data.method - Classification method ('cache', 'zero-shot', 'fallback')
   */
  trackIntentClassification(data) {
    if (!this.config.metricsEnabled) return;

    try {
      this.debug('Intent classification tracked', {
        query: data.query?.substring(0, 100),
        intents: data.intents?.map(i => i.name || i).join(', '),
        confidence: data.confidence,
        duration: data.duration,
        method: data.method
      });
    } catch (error) {
      this.error('Failed to track intent classification', error);
    }
  }

  /**
   * Track entity extraction operation
   *
   * @param {Object} data - Extraction data
   * @param {string} data.query - Original query
   * @param {Array} data.entities - Extracted entities
   * @param {number} data.duration - Extraction duration in ms
   * @param {string} data.method - Extraction method ('cache', 'llm', 'pattern')
   */
  trackEntityExtraction(data) {
    if (!this.config.metricsEnabled) return;

    try {
      this.debug('Entity extraction tracked', {
        query: data.query?.substring(0, 100),
        entityCount: data.entities?.length || 0,
        duration: data.duration,
        method: data.method
      });
    } catch (error) {
      this.error('Failed to track entity extraction', error);
    }
  }

  /**
   * Track table routing decision
   *
   * @param {Object} data - Routing data
   * @param {string} data.query - Original query
   * @param {Array} data.selectedTables - Tables selected for search
   * @param {string} data.strategy - Routing strategy ('parallel', 'sequential', 'cascading')
   * @param {number} data.duration - Routing duration in ms
   */
  trackTableRouting(data) {
    if (!this.config.metricsEnabled) return;

    try {
      this.debug('Table routing tracked', {
        query: data.query?.substring(0, 100),
        tables: data.selectedTables?.join(', '),
        strategy: data.strategy,
        duration: data.duration
      });
    } catch (error) {
      this.error('Failed to track table routing', error);
    }
  }

  /**
   * Track embedding generation
   *
   * @param {boolean} fromCache - Whether embedding came from cache
   * @param {number} duration - Generation duration in ms
   */
  trackEmbeddingGeneration(fromCache, duration) {
    if (!this.config.metricsEnabled) return;

    try {
      this.trackCacheHit('embedding', fromCache);
      this.debug('Embedding generation tracked', {
        fromCache,
        duration
      });
    } catch (error) {
      this.error('Failed to track embedding generation', error);
    }
  }

  /**
   * Track user feedback
   *
   * @param {Object} feedback - Feedback data
   * @param {string} feedback.type - Feedback type
   * @param {string} feedback.query - Original query
   * @param {string} feedback.correctIntent - Correct intent (if applicable)
   */
  trackFeedback(feedback) {
    if (!this.config.metricsEnabled) return;

    try {
      this.debug('Feedback tracked', {
        type: feedback.type,
        query: feedback.query?.substring(0, 100)
      });
    } catch (error) {
      this.error('Failed to track feedback', error);
    }
  }

  /**
   * Update document count in vector database
   *
   * @param {string} table - Table name
   * @param {string} project - Project name
   * @param {number} count - Number of documents
   */
  updateDocumentCount(table, project, count) {
    if (!this.config.metricsEnabled) return;

    try {
      this.metrics.documentCount.labels(table, project).set(count);

      this.debug('Document count updated', {
        table,
        project,
        count
      });
    } catch (error) {
      this.error('Failed to update document count', error);
    }
  }

  /**
   * Track error occurrence
   *
   * @param {string} type - Error type (e.g., 'database', 'llm', 'cache')
   * @param {string} severity - Error severity ('low', 'medium', 'high', 'critical')
   */
  trackError(type, severity = 'medium') {
    if (!this.config.metricsEnabled) return;

    try {
      this.metrics.errors.labels(type, severity).inc();

      this.debug('Error tracked', {
        type,
        severity
      });
    } catch (error) {
      console.error('Failed to track error metric:', error);
    }
  }

  /**
   * Get Prometheus metrics in text format for scraping
   *
   * @returns {Promise<string>} Prometheus-formatted metrics
   */
  async getMetrics() {
    if (!this.config.metricsEnabled) {
      return '# Metrics disabled';
    }

    try {
      return await promClient.register.metrics();
    } catch (error) {
      this.error('Failed to get Prometheus metrics', error);
      return '# Error retrieving metrics';
    }
  }

  /**
   * Get metrics in JSON format for API responses
   *
   * @returns {Promise<Object>} Metrics as JSON object
   */
  async getMetricsJSON() {
    if (!this.config.metricsEnabled) {
      return { enabled: false };
    }

    try {
      const metrics = await promClient.register.getMetricsAsJSON();
      return {
        enabled: true,
        metrics,
        timestamp: new Date().toISOString()
      };
    } catch (error) {
      this.error('Failed to get metrics as JSON', error);
      return { enabled: true, error: error.message };
    }
  }

  /**
   * Perform comprehensive health check
   *
   * @returns {Promise<Object>} Health check results
   */
  async checkHealth() {
    const uptime = (Date.now() - this.startTime) / 1000;

    return {
      status: this.healthStatus.logging && this.healthStatus.metrics ? 'healthy' : 'degraded',
      timestamp: new Date().toISOString(),
      uptime: `${Math.floor(uptime)}s`,
      checks: {
        logging: {
          status: this.healthStatus.logging ? 'up' : 'down',
          level: this.config.logLevel,
          directory: this.config.logDir
        },
        metrics: {
          status: this.healthStatus.metrics ? 'up' : 'down',
          enabled: this.config.metricsEnabled,
          port: this.config.metricsPort
        },
        database: {
          status: this.healthStatus.database ? 'up' : 'unknown'
        },
        llm: {
          status: this.healthStatus.llm ? 'up' : 'unknown'
        }
      },
      config: {
        logLevel: this.config.logLevel,
        metricsEnabled: this.config.metricsEnabled,
        consoleLogging: this.config.consoleLogging
      }
    };
  }

  /**
   * Update health status for external components
   *
   * @param {string} component - Component name ('database', 'llm')
   * @param {boolean} healthy - Whether component is healthy
   */
  updateHealthStatus(component, healthy) {
    if (this.healthStatus.hasOwnProperty(component)) {
      this.healthStatus[component] = healthy;

      this.debug('Health status updated', {
        component,
        healthy
      });
    }
  }

  /**
   * Log info message
   *
   * @param {string} message - Log message
   * @param {Object} meta - Additional metadata
   */
  info(message, meta = {}) {
    this.logger.info(message, meta);
  }

  /**
   * Log error message with stack trace
   *
   * @param {string} message - Error message
   * @param {Error} error - Error object
   * @param {Object} meta - Additional metadata
   */
  error(message, error, meta = {}) {
    const errorData = {
      ...meta,
      error: error?.message || String(error),
      stack: error?.stack
    };

    this.logger.error(message, errorData);

    // Track error in metrics
    if (this.config.metricsEnabled) {
      const errorType = meta.type || 'unknown';
      const severity = meta.severity || 'medium';
      this.trackError(errorType, severity);
    }
  }

  /**
   * Log warning message
   *
   * @param {string} message - Warning message
   * @param {Object} meta - Additional metadata
   */
  warn(message, meta = {}) {
    this.logger.warn(message, meta);
  }

  /**
   * Log debug message
   *
   * @param {string} message - Debug message
   * @param {Object} meta - Additional metadata
   */
  debug(message, meta = {}) {
    this.logger.debug(message, meta);
  }

  /**
   * Create a timer for measuring operation duration
   *
   * @param {string} operation - Operation name
   * @returns {Function} Stop function that returns elapsed time in seconds
   */
  startTimer(operation) {
    const startTime = Date.now();

    return () => {
      const duration = (Date.now() - startTime) / 1000;
      this.debug(`Timer: ${operation}`, { duration });
      return duration;
    };
  }

  /**
   * Get cache statistics
   *
   * @returns {Object} Cache performance statistics
   */
  getCacheStats() {
    if (!this.config.metricsEnabled) {
      return { enabled: false };
    }

    try {
      return {
        enabled: true,
        embedding: {
          note: 'Query Prometheus at /metrics for hit/miss counts'
        },
        response: {
          note: 'Query Prometheus at /metrics for hit/miss counts'
        },
        endpoint: '/metrics'
      };
    } catch (error) {
      this.error('Failed to get cache stats', error);
      return { enabled: true, error: error.message };
    }
  }

  /**
   * Shutdown monitoring system gracefully
   */
  async shutdown() {
    this.info('Shutting down monitoring system');

    try {
      // Close Winston transports
      this.info('Monitoring system shutdown complete');

      await new Promise((resolve) => {
        this.logger.on('finish', resolve);
        this.logger.end();
      });
    } catch (error) {
      console.error('Error during monitoring shutdown:', error);
    }
  }
}

// Singleton instance
let monitoringInstance = null;

/**
 * Get or create monitoring instance
 *
 * @param {Object} config - Configuration options
 * @returns {ProductionMonitoring} Monitoring instance
 */
export function getMonitoring(config = {}) {
  if (!monitoringInstance) {
    monitoringInstance = new ProductionMonitoring(config);
  }
  return monitoringInstance;
}

export default ProductionMonitoring;
