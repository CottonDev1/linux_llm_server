/**
 * LLM Routes Module
 *
 * Handles LLM control and service configuration endpoints
 * Extracted from adminRoutes.js for better organization
 *
 * Endpoints:
 * - LLM Control: Get models, check running models, start/stop models
 * - Service Configuration: Get/update service config, MongoDB operations
 * - Service Management: Restart/stop/status for node and python services
 */

import express from 'express';

const router = express.Router();

/**
 * Initialize LLM routes with required dependencies
 *
 * @param {Object} dependencies - Service and middleware dependencies
 * @param {Function} dependencies.requireAuth - Authentication middleware
 * @param {Object} dependencies.ewraiDatabase - EWRAIDatabase instance
 */
export default function initLlmRoutes(dependencies) {
  const {
    requireAuth,
    ewraiDatabase
  } = dependencies;

  // ============================================================================
  // LLM Control Endpoints (llama.cpp)
  // ============================================================================

  /**
   * GET /api/admin/llm/models
   * Get list of available LLM models via Python service
   */
  router.get('/llm/models', async (req, res) => {
    try {
      const pythonUrl = ewraiDatabase.getSetting('PythonServiceUrl') || 'http://localhost:8001';
      const response = await fetch(`${pythonUrl}/status`);
      if (!response.ok) {
        throw new Error('Failed to fetch status from Python service');
      }
      const data = await response.json();
      // Extract models from LLM endpoints
      const models = [];
      if (data.llm?.endpoints) {
        for (const [key, endpoint] of Object.entries(data.llm.endpoints)) {
          if (endpoint.models) {
            endpoint.models.forEach(m => {
              const name = m.split(/[/\\]/).pop();
              models.push({ name, endpoint: key, modified_at: new Date().toISOString() });
            });
          }
        }
      }
      res.json({
        success: true,
        models: models
      });
    } catch (error) {
      console.error('❌ Failed to fetch LLM models:', error.message);
      res.status(500).json({
        success: false,
        error: error.message,
        models: []
      });
    }
  });

  /**
   * GET /api/admin/llm/all-models
   * Get list of all active llama.cpp models from all servers with metrics
   * Proxies through Python service which has direct access to llama.cpp servers
   */
  router.get('/llm/all-models', async (req, res) => {
    try {
      const pythonUrl = ewraiDatabase.getSetting('PythonServiceUrl') || 'http://localhost:8001';

      // Fetch health info from Python service (which can reach llama.cpp servers)
      const response = await fetch(`${pythonUrl}/status`, {
        signal: AbortSignal.timeout(5000)
      });

      if (!response.ok) {
        throw new Error(`Python service returned ${response.status}`);
      }

      const healthData = await response.json();

      // Transform Python health response to frontend format
      const serverConfig = {
        sql: { name: 'SQL', port: 8080, description: 'Text-to-SQL Generation' },
        general: { name: 'General', port: 8081, description: 'General Chat & Summarization' },
        code: { name: 'Code', port: 8082, description: 'Code Analysis' }
      };

      const activeModels = [];
      const endpoints = healthData.llm?.endpoints || {};

      for (const [key, config] of Object.entries(serverConfig)) {
        const endpointData = endpoints[key];

        if (endpointData) {
          // Extract model name from the path
          let modelName = null;
          if (endpointData.models && endpointData.models.length > 0) {
            const modelPath = endpointData.models[0];
            modelName = modelPath.split(/[/\\]/).pop().replace('.gguf', '');
          }

          // Pass through metrics from Python service (which fetches from llama.cpp /metrics)
          const metrics = endpointData.metrics || null;

          activeModels.push({
            server: config.name,
            port: config.port,
            model: modelName,
            description: config.description,
            status: endpointData.healthy ? 'running' : 'offline',
            metrics: metrics
          });
        } else {
          activeModels.push({
            server: config.name,
            port: config.port,
            model: null,
            description: config.description,
            status: 'offline',
            metrics: null
          });
        }
      }

      res.json({
        success: true,
        models: activeModels,
        activeCount: activeModels.filter(m => m.status === 'running').length
      });
    } catch (error) {
      console.error('❌ Failed to fetch all LLM models:', error.message);
      res.status(500).json({
        success: false,
        error: error.message,
        models: []
      });
    }
  });

  /**
   * Parse Prometheus metrics format text into an object
   */
  function parsePrometheusMetrics(metricsText) {
    const result = {
      tps: 0,
      requests: 0,
      tokensProcessed: 0,
      promptTokensTotal: 0,
      generatedTokensTotal: 0,
      requestsProcessing: 0
    };

    const lines = metricsText.split('\n');
    for (const line of lines) {
      if (line.startsWith('#') || !line.trim()) continue;

      // Parse key metrics
      // llamacpp_tokens_second - tokens per second
      if (line.startsWith('llamacpp_tokens_second')) {
        const match = line.match(/llamacpp_tokens_second\s+([\d.]+)/);
        if (match) result.tps = parseFloat(match[1]);
      }
      // llamacpp_requests_processing - active requests
      else if (line.startsWith('llamacpp_requests_processing')) {
        const match = line.match(/llamacpp_requests_processing\s+(\d+)/);
        if (match) result.requestsProcessing = parseInt(match[1]);
      }
      // llamacpp_prompt_tokens_total - total prompt tokens processed
      else if (line.startsWith('llamacpp_prompt_tokens_total')) {
        const match = line.match(/llamacpp_prompt_tokens_total\s+(\d+)/);
        if (match) result.promptTokensTotal = parseInt(match[1]);
      }
      // llamacpp_tokens_predicted_total - total tokens generated
      else if (line.startsWith('llamacpp_tokens_predicted_total')) {
        const match = line.match(/llamacpp_tokens_predicted_total\s+(\d+)/);
        if (match) result.generatedTokensTotal = parseInt(match[1]);
      }
      // llamacpp_kv_cache_tokens - current KV cache usage
      else if (line.startsWith('llamacpp_kv_cache_tokens')) {
        const match = line.match(/llamacpp_kv_cache_tokens\s+(\d+)/);
        if (match) result.kvCacheTokens = parseInt(match[1]);
      }
      // llamacpp_kv_cache_used_cells - KV cache cells used
      else if (line.startsWith('llamacpp_kv_cache_used_cells')) {
        const match = line.match(/llamacpp_kv_cache_used_cells\s+(\d+)/);
        if (match) result.kvCacheUsed = parseInt(match[1]);
      }
      // llamacpp_requests_total - total requests processed
      else if (line.startsWith('llamacpp_requests_total')) {
        const match = line.match(/llamacpp_requests_total\s+(\d+)/);
        if (match) result.requests = parseInt(match[1]);
      }
    }

    result.tokensProcessed = result.promptTokensTotal + result.generatedTokensTotal;
    return result;
  }

  /**
   * GET /api/admin/llm/ps
   * Get currently running LLM models via Python service
   */
  router.get('/llm/ps', async (req, res) => {
    try {
      const pythonUrl = ewraiDatabase.getSetting('PythonServiceUrl') || 'http://localhost:8001';
      const response = await fetch(`${pythonUrl}/status`);
      if (!response.ok) {
        throw new Error('Failed to fetch running models from Python service');
      }
      const data = await response.json();
      // Convert to running models format from all endpoints
      const models = [];
      if (data.llm?.endpoints) {
        for (const [key, endpoint] of Object.entries(data.llm?.endpoints)) {
          if (endpoint.healthy && endpoint.models) {
            endpoint.models.forEach(m => {
              models.push({
                name: m.split(/[/\\]/).pop(),
                endpoint: key,
                size: 0,
                digest: '',
                details: { family: 'llama-cpp' }
              });
            });
          }
        }
      }
      res.json({
        success: true,
        models: models
      });
    } catch (error) {
      console.error('❌ Failed to fetch running models:', error.message);
      res.status(500).json({
        success: false,
        error: error.message,
        models: []
      });
    }
  });

  /**
   * POST /api/admin/llm/start
   * Check LLM model availability via Python service
   */
  router.post('/llm/start', requireAuth, async (req, res) => {
    try {
      const { model } = req.body;
      console.log(`🚀 Checking LLM model availability: ${model || 'default'}`);

      const pythonUrl = ewraiDatabase.getSetting('PythonServiceUrl') || 'http://localhost:8001';
      const response = await fetch(`${pythonUrl}/status`);
      if (!response.ok) {
        throw new Error('Python service not available');
      }
      const data = await response.json();
      const loadedModels = [];
      if (data.llm?.endpoints) {
        for (const [key, endpoint] of Object.entries(data.llm?.endpoints)) {
          if (endpoint.healthy && endpoint.models?.[0]) {
            loadedModels.push(`${key}: ${endpoint.models[0].split(/[/\\]/).pop()}`);
          }
        }
      }

      console.log(`✅ LLM models available: ${loadedModels.join(', ')}`);

      res.json({
        success: true,
        message: `Models loaded (llama-cpp-python loads models at startup)`,
        models: loadedModels
      });
    } catch (error) {
      console.error('❌ Failed to check model:', error.message);
      res.status(500).json({
        success: false,
        error: error.message
      });
    }
  });

  /**
   * POST /api/admin/llm/stop
   * Stop/unload an LLM model (llama-cpp-python - requires server restart)
   */
  router.post('/llm/stop', requireAuth, async (req, res) => {
    try {
      const { model } = req.body;
      console.log(`🛑 Stop model requested: ${model || 'default'}`);

      // llama-cpp-python doesn't support dynamic model unloading
      // Model is loaded at server startup and remains loaded
      console.log(`ℹ️ llama-cpp-python requires server restart to unload model`);

      res.json({
        success: true,
        message: `Model cannot be dynamically unloaded in llama-cpp-python. Restart the server to change models.`,
        model: model
      });
    } catch (error) {
      console.error('❌ Failed to stop model:', error.message);
      res.status(500).json({
        success: false,
        error: error.message
      });
    }
  });

  // ============================================================================
  // Hugging Face Model Download
  // ============================================================================

  // Track active downloads
  const activeHFDownloads = new Map();

  /**
   * POST /api/admin/llm/download-hf
   * Start downloading a model from Hugging Face
   */
  router.post('/llm/download-hf', requireAuth, async (req, res) => {
    try {
      const { modelId, filename } = req.body;

      if (!modelId || !filename) {
        return res.status(400).json({
          success: false,
          error: 'modelId and filename are required'
        });
      }

      const downloadId = `${modelId}/${filename}`.replace(/[/\\]/g, '-');

      // Check if already downloading
      if (activeHFDownloads.has(downloadId)) {
        return res.json({
          success: true,
          message: 'Download already in progress',
          downloadId
        });
      }

      console.log(`📥 Starting Hugging Face download: ${modelId}/${filename}`);

      // Initialize download state
      activeHFDownloads.set(downloadId, {
        status: 'starting',
        progress: 0,
        speed: '',
        error: null,
        startTime: Date.now()
      });

      // Start download in background
      downloadHuggingFaceModel(modelId, filename, downloadId);

      res.json({
        success: true,
        message: 'Download started',
        downloadId
      });

    } catch (error) {
      console.error('❌ Failed to start HF download:', error.message);
      res.status(500).json({
        success: false,
        error: error.message
      });
    }
  });

  /**
   * GET /api/admin/llm/download-status/:downloadId
   * Get status of a Hugging Face download
   */
  router.get('/llm/download-status/:downloadId', async (req, res) => {
    try {
      const { downloadId } = req.params;
      const decodedId = decodeURIComponent(downloadId);

      const status = activeHFDownloads.get(decodedId);

      if (!status) {
        return res.json({
          success: true,
          status: 'not_found',
          progress: 0
        });
      }

      res.json({
        success: true,
        ...status
      });

    } catch (error) {
      res.status(500).json({
        success: false,
        error: error.message
      });
    }
  });

  /**
   * Download a model file from Hugging Face
   * @param {string} modelId - HuggingFace model ID (e.g., "TheBloke/Llama-2-7B-GGUF")
   * @param {string} filename - File to download (e.g., "llama-2-7b.Q4_K_M.gguf")
   * @param {string} downloadId - Unique download identifier
   */
  async function downloadHuggingFaceModel(modelId, filename, downloadId) {
    const { createWriteStream, promises: fs } = await import('fs');
    const { pipeline } = await import('stream/promises');
    const path = await import('path');

    // Models directory - use the llamacpp models path from script
    const modelsDir = process.env.MODELS_PATH || 'C:\\Projects\\LLM_Website\\models\\llamacpp';

    try {
      // Ensure models directory exists
      await fs.mkdir(modelsDir, { recursive: true });

      // Construct download URL
      const downloadUrl = `https://huggingface.co/${modelId}/resolve/main/${filename}`;
      const outputPath = path.join(modelsDir, filename);

      console.log(`📥 Downloading from: ${downloadUrl}`);
      console.log(`📁 Saving to: ${outputPath}`);

      // Update status
      activeHFDownloads.set(downloadId, {
        ...activeHFDownloads.get(downloadId),
        status: 'downloading',
        progress: 0
      });

      // Start download with fetch
      const response = await fetch(downloadUrl, {
        headers: {
          'User-Agent': 'EWR-RAG-Server/1.0'
        }
      });

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }

      const contentLength = parseInt(response.headers.get('content-length') || '0', 10);
      let downloadedBytes = 0;
      let lastProgressUpdate = Date.now();
      let lastBytes = 0;

      // Create transform stream to track progress
      const { Transform } = await import('stream');

      const progressStream = new Transform({
        transform(chunk, encoding, callback) {
          downloadedBytes += chunk.length;

          // Update progress every 500ms
          const now = Date.now();
          if (now - lastProgressUpdate > 500) {
            const progress = contentLength > 0 ? Math.round((downloadedBytes / contentLength) * 100) : 0;
            const bytesPerSecond = (downloadedBytes - lastBytes) / ((now - lastProgressUpdate) / 1000);
            const speed = formatBytes(bytesPerSecond) + '/s';

            activeHFDownloads.set(downloadId, {
              ...activeHFDownloads.get(downloadId),
              status: 'downloading',
              progress,
              speed,
              downloadedBytes,
              totalBytes: contentLength
            });

            lastProgressUpdate = now;
            lastBytes = downloadedBytes;
          }

          callback(null, chunk);
        }
      });

      // Pipe response to file
      const fileStream = createWriteStream(outputPath);
      const { Readable } = await import('stream');
      const readable = Readable.fromWeb(response.body);

      await pipeline(readable, progressStream, fileStream);

      console.log(`✅ Download complete: ${filename}`);

      // Update status to complete
      activeHFDownloads.set(downloadId, {
        status: 'complete',
        progress: 100,
        speed: '',
        downloadedBytes: contentLength,
        totalBytes: contentLength
      });

      // Clean up after 5 minutes
      setTimeout(() => {
        activeHFDownloads.delete(downloadId);
      }, 5 * 60 * 1000);

    } catch (error) {
      console.error(`❌ Download failed: ${error.message}`);

      activeHFDownloads.set(downloadId, {
        status: 'error',
        progress: 0,
        error: error.message
      });

      // Clean up after 5 minutes
      setTimeout(() => {
        activeHFDownloads.delete(downloadId);
      }, 5 * 60 * 1000);
    }
  }

  /**
   * Format bytes to human-readable string
   */
  function formatBytes(bytes) {
    if (bytes === 0) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + ' ' + sizes[i];
  }

  // ============================================================================
  // Service Configuration & Management
  // ============================================================================

  /**
   * GET /api/admin/service-config
   * Get current service configuration from SQLite settings
   */
  router.get('/service-config', requireAuth, async (req, res) => {
    try {
      const config = {
        mongoDBUri: ewraiDatabase.getSetting('MongoDBUri') || 'mongodb://localhost:27017',
        mongoDBDatabase: ewraiDatabase.getSetting('MongoDBDatabase') || 'rag_server',
        pythonServiceUrl: ewraiDatabase.getSetting('PythonServiceUrl') || 'http://localhost:8001',
        llmHost: ewraiDatabase.getSetting('LLMHost') || 'http://localhost:11434',
        nodeServerPort: ewraiDatabase.getSetting('NodeServerPort') || '3000'
      };

      res.json({
        success: true,
        config
      });
    } catch (error) {
      console.error('❌ Failed to get service config:', error.message);
      res.status(500).json({
        success: false,
        error: error.message
      });
    }
  });

  /**
   * PUT /api/admin/service-config
   * Update service configuration in SQLite settings
   */
  router.put('/service-config', requireAuth, async (req, res) => {
    try {
      const { mongoDBUri, mongoDBDatabase, pythonServiceUrl, llmHost, nodeServerPort } = req.body;

      if (mongoDBUri !== undefined) {
        ewraiDatabase.updateSetting('MongoDBUri', mongoDBUri);
      }
      if (mongoDBDatabase !== undefined) {
        ewraiDatabase.updateSetting('MongoDBDatabase', mongoDBDatabase);
      }
      if (pythonServiceUrl !== undefined) {
        ewraiDatabase.updateSetting('PythonServiceUrl', pythonServiceUrl);
      }
      if (llmHost !== undefined) {
        ewraiDatabase.updateSetting('LLMHost', llmHost);
      }
      if (nodeServerPort !== undefined) {
        ewraiDatabase.updateSetting('NodeServerPort', nodeServerPort);
      }

      res.json({
        success: true,
        message: 'Service configuration updated'
      });
    } catch (error) {
      console.error('❌ Failed to update service config:', error.message);
      res.status(500).json({
        success: false,
        error: error.message
      });
    }
  });

  /**
   * POST /api/admin/test-mongodb
   * Test MongoDB connectivity with a given URI
   */
  router.post('/test-mongodb', requireAuth, async (req, res) => {
    try {
      const { uri, database } = req.body;

      if (!uri) {
        return res.status(400).json({
          success: false,
          error: 'MongoDB URI is required'
        });
      }

      // Use Python service to test MongoDB connection
      const pythonUrl = ewraiDatabase.getSetting('PythonServiceUrl') || 'http://localhost:8001';
      const testResponse = await fetch(`${pythonUrl}/admin/test-mongodb`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ uri, database: database || 'rag_server' })
      });

      if (!testResponse.ok) {
        const errorData = await testResponse.json().catch(() => ({ error: 'Connection test failed' }));
        return res.status(400).json({
          success: false,
          error: errorData.error || 'MongoDB connection test failed'
        });
      }

      const result = await testResponse.json();
      res.json({
        success: true,
        message: 'MongoDB connection successful',
        details: result
      });

    } catch (error) {
      console.error('❌ MongoDB connection test failed:', error.message);
      res.status(500).json({
        success: false,
        error: `Connection test failed: ${error.message}`
      });
    }
  });

  /**
   * POST /api/admin/connect-mongodb
   * Update Python service to use new MongoDB connection
   */
  router.post('/connect-mongodb', requireAuth, async (req, res) => {
    try {
      const { uri, database } = req.body;

      if (!uri) {
        return res.status(400).json({
          success: false,
          error: 'MongoDB URI is required'
        });
      }

      // Save to SQLite first
      ewraiDatabase.updateSetting('MongoDBUri', uri);
      if (database) {
        ewraiDatabase.updateSetting('MongoDBDatabase', database);
      }

      // Tell Python service to reconnect with new URI
      const pythonUrl = ewraiDatabase.getSetting('PythonServiceUrl') || 'http://localhost:8001';
      const connectResponse = await fetch(`${pythonUrl}/admin/reconnect-mongodb`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ uri, database: database || 'rag_server' })
      });

      if (!connectResponse.ok) {
        const errorData = await connectResponse.json().catch(() => ({ error: 'Reconnection failed' }));
        return res.status(400).json({
          success: false,
          error: errorData.error || 'Failed to update Python service MongoDB connection'
        });
      }

      res.json({
        success: true,
        message: 'MongoDB connection updated successfully'
      });

    } catch (error) {
      console.error('❌ Failed to connect MongoDB:', error.message);
      res.status(500).json({
        success: false,
        error: error.message
      });
    }
  });

  /**
   * POST /api/admin/service/:service/restart
   * Restart a service (node or python)
   */
  router.post('/service/:service/restart', requireAuth, async (req, res) => {
    try {
      const { service } = req.params;

      if (!['node', 'python'].includes(service)) {
        return res.status(400).json({
          success: false,
          error: 'Invalid service. Must be "node" or "python"'
        });
      }

      if (service === 'python') {
        // Use Python service's own restart API
        try {
          const response = await fetch('http://localhost:8001/admin/service/restart', {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json'
            }
          });

          const data = await response.json();

          if (data.success) {
            res.json({
              success: true,
              message: 'Python service restart initiated',
              pid: data.pid,
              estimated_restart_time_seconds: data.estimated_restart_time_seconds
            });
          } else {
            res.status(500).json({
              success: false,
              error: data.error || 'Failed to restart Python service'
            });
          }
        } catch (fetchError) {
          // Python service is not responding
          console.error('❌ Python service not responding:', fetchError.message);
          res.status(503).json({
            success: false,
            error: 'Python service is not responding. Start it manually or check if it crashed.',
            instructions: 'Run: python_services\\venv\\Scripts\\python.exe -m uvicorn main:app --host 0.0.0.0 --port 8001'
          });
        }

      } else if (service === 'node') {
        // For Node, we can't easily restart ourselves, so return instructions
        res.json({
          success: true,
          message: 'Node service restart requires manual intervention or process manager',
          instructions: 'Use PM2 or run: .\\stop-all-services.ps1 && .\\start-all-services.ps1'
        });
      }

    } catch (error) {
      console.error(`❌ Failed to restart ${req.params.service} service:`, error.message);
      res.status(500).json({
        success: false,
        error: error.message
      });
    }
  });

  /**
   * POST /api/admin/service/:service/stop
   * Stop a service (node or python)
   *
   * Note: Service stop requires manual intervention or external process manager.
   */
  router.post('/service/:service/stop', requireAuth, async (req, res) => {
    try {
      const { service } = req.params;

      if (!['node', 'python'].includes(service)) {
        return res.status(400).json({
          success: false,
          error: 'Invalid service. Must be "node" or "python"'
        });
      }

      if (service === 'python') {
        res.json({
          success: false,
          message: 'Python service stop requires manual intervention',
          instructions: 'Use process manager or manually stop the uvicorn process'
        });

      } else if (service === 'node') {
        res.json({
          success: false,
          message: 'Cannot stop Node service from within itself',
          instructions: 'Use process manager or manually stop the node process'
        });
      }

    } catch (error) {
      console.error(`❌ Failed to stop ${req.params.service} service:`, error.message);
      res.status(500).json({
        success: false,
        error: error.message
      });
    }
  });

  /**
   * GET /api/admin/service/:service/status
   * Get status of a service
   */
  router.get('/service/:service/status', requireAuth, async (req, res) => {
    try {
      const { service } = req.params;

      if (!['node', 'python', 'mongodb', 'llm'].includes(service)) {
        return res.status(400).json({
          success: false,
          error: 'Invalid service'
        });
      }

      // Special handling for Python service - use its own status API
      if (service === 'python') {
        try {
          const response = await fetch('http://localhost:8001/admin/service/status', {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json'
            }
          });

          const data = await response.json();

          if (data.success) {
            res.json({
              success: true,
              service: 'python',
              status: data.healthy ? 'running' : 'unhealthy',
              port: data.port,
              pid: data.pid,
              uptime_seconds: data.uptime_seconds,
              uptime_formatted: data.uptime_formatted,
              memory_mb: data.memory_mb,
              cpu_percent: data.cpu_percent,
              threads: data.threads,
              connections: data.connections,
              mongodb_connected: data.mongodb_connected,
              llm_healthy: data.llm_healthy
            });
          } else {
            res.json({
              success: true,
              service: 'python',
              status: 'unhealthy',
              error: data.error
            });
          }
        } catch (fetchError) {
          // Python service is down or unreachable
          res.json({
            success: true,
            service: 'python',
            status: 'stopped',
            error: 'Python service not responding'
          });
        }
        return;
      }

      // For other services, use HTTP health check
      let port;
      let healthUrl;
      switch (service) {
        case 'node':
          port = 3000;
          healthUrl = 'http://localhost:3000/health';
          break;
        case 'mongodb':
          port = 27017;
          // MongoDB status checked via Python service
          healthUrl = null;
          break;
        case 'llm':
          port = 8080;
          healthUrl = 'http://localhost:8080/health';
          break;
      }

      if (service === 'mongodb') {
        // Check MongoDB via Python service status
        try {
          const pythonUrl = ewraiDatabase.getSetting('PythonServiceUrl') || 'http://localhost:8001';
          const response = await fetch(`${pythonUrl}/status`, { signal: AbortSignal.timeout(3000) });
          const data = await response.json();
          res.json({
            success: true,
            service: 'mongodb',
            status: data.mongodb?.connected ? 'running' : 'stopped',
            port
          });
        } catch {
          res.json({
            success: true,
            service: 'mongodb',
            status: 'unknown',
            error: 'Cannot check MongoDB status - Python service unavailable'
          });
        }
        return;
      }

      // For node and llm, try HTTP health check
      try {
        const response = await fetch(healthUrl, { signal: AbortSignal.timeout(3000) });
        res.json({
          success: true,
          service,
          status: response.ok ? 'running' : 'unhealthy',
          port
        });
      } catch {
        res.json({
          success: true,
          service,
          status: 'stopped',
          port
        });
      }

    } catch (error) {
      res.json({
        success: true,
        service: req.params.service,
        status: 'unknown',
        error: error.message
      });
    }
  });

  return router;
}
