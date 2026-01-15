/**
 * Audio Routes Module
 * Handles all audio analysis pipeline endpoints
 *
 * This module proxies audio-related requests to the Python FastAPI service
 * which performs SenseVoice analysis and MongoDB storage with embeddings.
 *
 * Routes:
 * - POST /audio/upload - Handle multipart file upload and forward to Python
 * - POST /audio/analyze-stream - Trigger SenseVoice analysis with streaming progress
 * - POST /audio/store - Store analysis with embeddings to MongoDB
 * - GET /audio/search - Vector search on audio analysis results
 * - GET /audio/:id - Retrieve specific analysis by ID
 * - DELETE /audio/:id - Delete analysis by ID
 * - GET /audio/stats - Get audio analysis statistics
 * - GET /audio/pending - List pending analysis files
 * - GET /audio/pending/:filename - Get specific pending analysis
 * - DELETE /audio/pending/:filename - Delete pending analysis file
 */

import express from 'express';
import multer from 'multer';
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const router = express.Router();

// Configure multer for audio file uploads
// Store files temporarily before forwarding to Python service
const storage = multer.diskStorage({
  destination: (req, file, cb) => {
    const uploadDir = path.join(__dirname, '../../uploads/audio');
    // Ensure upload directory exists
    if (!fs.existsSync(uploadDir)) {
      fs.mkdirSync(uploadDir, { recursive: true });
    }
    cb(null, uploadDir);
  },
  filename: (req, file, cb) => {
    // Generate unique filename with timestamp
    const uniqueSuffix = Date.now() + '-' + Math.round(Math.random() * 1E9);
    const ext = path.extname(file.originalname);
    cb(null, `audio-${uniqueSuffix}${ext}`);
  }
});

// File filter to accept only audio files
const fileFilter = (req, file, cb) => {
  const allowedMimes = [
    'audio/mpeg',
    'audio/mp3',
    'audio/wav',
    'audio/wave',
    'audio/x-wav',
    'audio/flac',
    'audio/ogg',
    'audio/webm',
    'audio/aac',
    'audio/m4a',
    'audio/x-m4a'
  ];

  if (allowedMimes.includes(file.mimetype)) {
    cb(null, true);
  } else {
    cb(new Error(`Invalid file type. Allowed types: ${allowedMimes.join(', ')}`), false);
  }
};

const upload = multer({
  storage: storage,
  fileFilter: fileFilter,
  limits: {
    fileSize: 100 * 1024 * 1024 // 100MB max file size
  }
});

/**
 * Create audio routes with dependency injection
 * @param {Object} deps - Dependencies needed by routes
 * @param {Object} deps.monitoring - Monitoring instance
 * @param {string} deps.pythonServiceUrl - Python service URL
 * @param {Function} deps.requireAuth - Authentication middleware (optional)
 * @param {Function} deps.requireAuthOrAdmin - Admin authentication middleware (optional)
 * @param {Object} deps.ewraiDatabase - EWRAIDatabase instance for settings
 * @returns {express.Router} Express router
 */
export default function createAudioRoutes(deps) {
  const {
    monitoring,
    pythonServiceUrl,
    requireAuth,
    requireAuthOrAdmin,
    ewraiDatabase
  } = deps;

  /**
   * POST /audio/upload
   * Handle multipart audio file upload and forward to Python service
   */
  router.post('/upload', upload.single('audio'), async (req, res) => {
    try {
      if (!req.file) {
        return res.status(400).json({
          success: false,
          error: 'No audio file uploaded'
        });
      }

      monitoring?.info(`Audio file uploaded: ${req.file.filename}`, {
        originalName: req.file.originalname,
        size: req.file.size,
        mimetype: req.file.mimetype
      });

      // Read the file and forward to Python service
      // Use native File class (Node.js 20+) which properly includes filename
      const fileBuffer = fs.readFileSync(req.file.path);
      const formData = new FormData();
      const fileObj = new File([fileBuffer], req.file.originalname, { type: req.file.mimetype });
      formData.append('file', fileObj);

      // Forward to Python service
      const response = await fetch(`${pythonServiceUrl}/audio/upload`, {
        method: 'POST',
        body: formData
      });

      // Clean up temporary file
      fs.unlinkSync(req.file.path);

      if (!response.ok) {
        throw new Error(`Python service error: ${response.statusText}`);
      }

      const result = await response.json();

      monitoring?.info(`Audio file uploaded successfully to Python service`, {
        filepath: result.filepath
      });

      res.json({
        success: true,
        filepath: result.filepath,
        filename: result.filename,
        size_mb: result.size_mb
      });

    } catch (error) {
      monitoring?.error('Audio upload failed', error);

      // Clean up temporary file on error
      if (req.file && fs.existsSync(req.file.path)) {
        try {
          fs.unlinkSync(req.file.path);
        } catch (cleanupError) {
          monitoring?.error('Failed to clean up temporary file', cleanupError);
        }
      }

      res.status(500).json({
        success: false,
        error: error.message
      });
    }
  });

  /**
   * POST /audio/analyze-stream
   * Trigger SenseVoice analysis with streaming progress updates
   * Returns Server-Sent Events (SSE) with progress messages
   * Body: { audio_path|filepath: string, original_filename|filename?: string, language?: string }
   */
  router.post('/analyze-stream', async (req, res) => {
    try {
      // Accept both field name formats for backward compatibility
      // Frontend sends: audio_path, original_filename
      // Legacy format: filepath, filename
      const filepath = req.body.audio_path || req.body.filepath;
      const filename = req.body.original_filename || req.body.filename;
      const language = req.body.language || 'auto';

      if (!filepath) {
        return res.status(400).json({
          success: false,
          error: 'audio_path (or filepath) is required'
        });
      }

      monitoring?.info(`Starting streaming SenseVoice analysis for file: ${filepath}`);

      const requestBody = {
        audio_path: filepath,
        original_filename: filename,
        language
      };

      // Set up SSE headers
      res.setHeader('Content-Type', 'text/event-stream');
      res.setHeader('Cache-Control', 'no-cache');
      res.setHeader('Connection', 'keep-alive');
      res.setHeader('X-Accel-Buffering', 'no');

      // Stream the response from Python service
      const response = await fetch(`${pythonServiceUrl}/audio/analyze-stream`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(requestBody)
      });

      if (!response.ok) {
        const errorText = await response.text();
        res.write(`data: ${JSON.stringify({ type: 'error', message: `Python service error: ${response.statusText}` })}\n\n`);
        res.end();
        return;
      }

      // Pipe the SSE stream from Python to the client
      const reader = response.body.getReader();
      const decoder = new TextDecoder();

      try {
        while (true) {
          const { done, value } = await reader.read();
          if (done) break;
          const chunk = decoder.decode(value, { stream: true });
          res.write(chunk);
        }
      } catch (streamError) {
        monitoring?.error('Stream error', streamError);
        res.write(`data: ${JSON.stringify({ type: 'error', message: streamError.message })}\n\n`);
      }

      res.end();
      monitoring?.info(`Streaming analysis completed for file: ${filepath}`);

    } catch (error) {
      monitoring?.error('Streaming audio analysis failed', error);
      // If headers not sent yet, send JSON error
      if (!res.headersSent) {
        res.status(500).json({
          success: false,
          error: error.message
        });
      } else {
        res.write(`data: ${JSON.stringify({ type: 'error', message: error.message })}\n\n`);
        res.end();
      }
    }
  });

  /**
   * POST /audio/store
   * Store analysis results with embeddings to MongoDB
   * Body: { transcription, raw_transcription, emotions, audio_events, language, audio_metadata, metadata: { customer_support_staff, ewr_customer, mood, outcome, filename }, call_metadata, call_content, pending_filename }
   */
  router.post('/store', async (req, res) => {
    try {
      const {
        transcription,
        raw_transcription,
        transcription_summary,
        emotions,
        audio_events,
        language,
        audio_metadata,
        metadata,
        call_metadata,
        call_content,
        pending_filename
      } = req.body;

      // Validate required metadata fields
      if (!metadata || !metadata.customer_support_staff || !metadata.ewr_customer || !metadata.mood || !metadata.filename) {
        return res.status(400).json({
          success: false,
          error: 'Missing required metadata fields (customer_support_staff, ewr_customer, mood, filename)'
        });
      }

      monitoring?.info(`Storing audio analysis for: ${metadata.filename}`, {
        customer_support_staff: metadata.customer_support_staff,
        ewr_customer: metadata.ewr_customer,
        mood: metadata.mood,
        has_call_metadata: !!call_metadata?.parsed,
        has_call_content: !!call_content?.subject,
        pending_filename: pending_filename
      });

      // Build the payload for Python service using the correct field names
      const storePayload = {
        customer_support_staff: metadata.customer_support_staff,
        ewr_customer: metadata.ewr_customer,
        mood: metadata.mood,
        outcome: metadata.outcome || '',
        filename: metadata.filename,
        transcription: transcription || '',
        raw_transcription: raw_transcription || '',
        transcription_summary: transcription_summary || null,
        emotions: emotions || { primary: 'NEUTRAL', detected: [], timestamps: [] },
        audio_events: audio_events || { detected: [], timestamps: [] },
        language: language || 'en',
        audio_metadata: audio_metadata || { duration_seconds: 0, sample_rate: 16000, channels: 1, format: 'wav', file_size_bytes: 0 },
        // New fields for call metadata (parsed from filename)
        call_metadata: call_metadata || {
          call_date: null,
          call_time: null,
          extension: null,
          phone_number: null,
          direction: null,
          auto_flag: null,
          recording_id: null,
          parsed: false
        },
        // New fields for LLM-analyzed content
        call_content: call_content || {
          subject: null,
          outcome: null,
          customer_name: null,
          confidence: 0,
          analysis_model: ''
        }
      };

      // Include pending_filename if provided (for cleanup after successful save)
      if (pending_filename) {
        storePayload.pending_filename = pending_filename;
      }

      const response = await fetch(`${pythonServiceUrl}/audio/store`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(storePayload)
      });

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`Python service error: ${response.statusText} - ${errorText}`);
      }

      const result = await response.json();

      monitoring?.info(`Audio analysis stored successfully`, {
        analysisId: result.analysis_id,
        filename: metadata.filename
      });

      res.json({
        success: true,
        ...result
      });

    } catch (error) {
      monitoring?.error('Audio storage failed', error);
      res.status(500).json({
        success: false,
        error: error.message
      });
    }
  });

  /**
   * GET /audio/search
   * Search on audio analysis results with date range filtering
   * Query params: date_from, date_to (YYYY-MM-DD), mood, outcome, customer_support_staff, ewr_customer, limit
   */
  router.get('/search', async (req, res) => {
    try {
      const {
        date_from,
        date_to,
        mood,
        outcome,
        customer_support_staff,
        ewr_customer,
        limit = 50
      } = req.query;

      monitoring?.info(`Searching audio analysis`, {
        date_from,
        date_to,
        mood,
        outcome,
        customer_support_staff,
        ewr_customer,
        limit
      });

      // Build request body for POST
      const requestBody = {
        limit: parseInt(limit)
      };

      if (date_from) requestBody.date_from = date_from;
      if (date_to) requestBody.date_to = date_to;
      if (mood) requestBody.mood = mood;
      if (outcome) requestBody.outcome = outcome;
      if (customer_support_staff) requestBody.customer_support_staff = customer_support_staff;
      if (ewr_customer) requestBody.ewr_customer = ewr_customer;

      const response = await fetch(`${pythonServiceUrl}/audio/search`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(requestBody)
      });

      if (!response.ok) {
        throw new Error(`Python service error: ${response.statusText}`);
      }

      const data = await response.json();
      // Python service returns { success: true, results: [...], count: N }
      const results = data.results || [];

      monitoring?.info(`Found ${results.length || 0} audio analysis results`);

      res.json({
        success: true,
        results: results,
        total: results.length,
        query: {
          date_from,
          date_to,
          mood,
          outcome,
          customer_support_staff,
          ewr_customer,
          limit: parseInt(limit)
        }
      });

    } catch (error) {
      monitoring?.error('Audio search failed', error);
      res.status(500).json({
        success: false,
        error: error.message,
        results: []
      });
    }
  });

  // ============================================================================
  // Pending Analysis Routes
  // ============================================================================

  /**
   * GET /audio/pending
   * List all pending analysis files
   */
  router.get('/pending', async (req, res) => {
    try {
      monitoring?.info('Retrieving pending analysis files');

      const response = await fetch(`${pythonServiceUrl}/audio/pending`);

      if (!response.ok) {
        throw new Error(`Python service error: ${response.statusText}`);
      }

      const result = await response.json();

      monitoring?.info('Retrieved pending analysis files', {
        count: result.pending_files?.length || 0
      });

      res.json({
        success: true,
        ...result
      });

    } catch (error) {
      monitoring?.error('Failed to retrieve pending analysis files', error);
      res.status(500).json({
        success: false,
        error: error.message
      });
    }
  });

  /**
   * GET /audio/pending/:filename
   * Get specific pending analysis file by filename
   */
  router.get('/pending/:filename', async (req, res) => {
    try {
      const { filename } = req.params;

      monitoring?.info(`Retrieving pending analysis: ${filename}`);

      const response = await fetch(`${pythonServiceUrl}/audio/pending/${encodeURIComponent(filename)}`);

      if (!response.ok) {
        if (response.status === 404) {
          return res.status(404).json({
            success: false,
            error: 'Pending analysis file not found'
          });
        }
        throw new Error(`Python service error: ${response.statusText}`);
      }

      const result = await response.json();

      monitoring?.info(`Retrieved pending analysis: ${filename}`);

      res.json({
        success: true,
        ...result
      });

    } catch (error) {
      monitoring?.error(`Failed to retrieve pending analysis: ${req.params.filename}`, error);
      res.status(500).json({
        success: false,
        error: error.message
      });
    }
  });

  /**
   * DELETE /audio/pending/:filename
   * Delete pending analysis file by filename
   */
  router.delete('/pending/:filename', requireAuthOrAdmin || ((req, res, next) => next()), async (req, res) => {
    try {
      const { filename } = req.params;

      monitoring?.info(`Deleting pending analysis: ${filename}`);

      const response = await fetch(`${pythonServiceUrl}/audio/pending/${encodeURIComponent(filename)}`, {
        method: 'DELETE'
      });

      if (!response.ok) {
        if (response.status === 404) {
          return res.status(404).json({
            success: false,
            error: 'Pending analysis file not found'
          });
        }
        throw new Error(`Python service error: ${response.statusText}`);
      }

      const result = await response.json();

      monitoring?.info(`Deleted pending analysis: ${filename}`);

      res.json({
        success: true,
        message: 'Pending analysis file deleted successfully',
        ...result
      });

    } catch (error) {
      monitoring?.error(`Failed to delete pending analysis: ${req.params.filename}`, error);
      res.status(500).json({
        success: false,
        error: error.message
      });
    }
  });

  /**
   * GET /audio/stats
   * Get audio analysis statistics
   * NOTE: This route MUST be defined before /:id to prevent "stats" being captured as an ID
   */
  router.get('/stats', async (req, res) => {
    try {
      monitoring?.info('Retrieving audio analysis statistics');

      const response = await fetch(`${pythonServiceUrl}/audio/stats/summary`);

      if (!response.ok) {
        throw new Error(`Python service error: ${response.statusText}`);
      }

      const stats = await response.json();

      monitoring?.info('Retrieved audio analysis statistics', {
        total: stats.total_analyses
      });

      res.json({
        success: true,
        ...stats
      });

    } catch (error) {
      monitoring?.error('Failed to retrieve audio statistics', error);
      res.status(500).json({
        success: false,
        error: error.message
      });
    }
  });

  /**
   * GET /audio/lookup-staff/:extension
   * Lookup customer support staff name by phone extension from EWRCentral database
   */
  router.get('/lookup-staff/:extension', async (req, res) => {
    try {
      const { extension } = req.params;

      monitoring?.info(`Looking up staff by extension: ${extension}`);

      const response = await fetch(`${pythonServiceUrl}/audio/lookup-staff/${encodeURIComponent(extension)}`);

      if (!response.ok) {
        throw new Error(`Python service error: ${response.statusText}`);
      }

      const result = await response.json();

      monitoring?.info('Staff lookup completed', {
        extension,
        found: result.found,
        staff: result.staff_name
      });

      res.json(result);

    } catch (error) {
      monitoring?.error(`Failed to lookup staff by extension: ${req.params.extension}`, error);
      res.status(500).json({
        success: false,
        error: error.message
      });
    }
  });

  /**
   * POST /audio/match-tickets
   * Find tickets in EWRCentral that might correspond to a phone call
   * Uses call metadata to search for matching tickets created by staff
   */
  router.post('/match-tickets', async (req, res) => {
    try {
      const { extension, phone_number, call_datetime, subject_keywords, customer_name, time_window_minutes } = req.body;

      monitoring?.info('Searching for matching tickets', {
        extension,
        phone_number,
        call_datetime
      });

      const response = await fetch(`${pythonServiceUrl}/audio/match-tickets`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          extension,
          phone_number,
          call_datetime,
          subject_keywords,
          customer_name,
          time_window_minutes: time_window_minutes || 60
        })
      });

      if (!response.ok) {
        throw new Error(`Python service error: ${response.statusText}`);
      }

      const result = await response.json();

      monitoring?.info('Ticket search completed', {
        total_matches: result.total_matches,
        best_match: result.best_match?.ticket?.CentralTicketID
      });

      res.json(result);

    } catch (error) {
      monitoring?.error('Failed to search for matching tickets', error);
      res.status(500).json({
        success: false,
        error: error.message
      });
    }
  });

  /**
   * GET /audio/staff-metrics/:staffName
   * Get comprehensive metrics for a staff member
   * Combines MongoDB call data and EWRCentral ticket data
   */
  router.get('/staff-metrics/:staffName', async (req, res) => {
    try {
      const { staffName } = req.params;

      monitoring?.info('Fetching staff metrics', { staffName });

      const response = await fetch(`${pythonServiceUrl}/audio/staff-metrics/${encodeURIComponent(staffName)}`);

      if (!response.ok) {
        throw new Error(`Python service error: ${response.statusText}`);
      }

      const result = await response.json();

      monitoring?.info('Staff metrics retrieved', {
        staffName,
        success: result.success,
        calls_analyzed: result.calls_analyzed
      });

      res.json(result);

    } catch (error) {
      monitoring?.error('Failed to fetch staff metrics', error);
      res.status(500).json({
        success: false,
        error: error.message
      });
    }
  });

  // ============================================================================
  // Bulk Audio Processing Routes
  // ============================================================================

  /**
   * GET /audio/settings/folder-path
   * Get the configured bulk audio folder path from settings
   */
  router.get('/settings/folder-path', requireAuthOrAdmin || ((req, res, next) => next()), async (req, res) => {
    try {
      if (!ewraiDatabase) {
        return res.status(500).json({
          success: false,
          error: 'Database not initialized'
        });
      }

      const folderPath = ewraiDatabase.getAudioBulkFolderPath();

      monitoring?.info('Retrieved audio bulk folder path', { folderPath });

      res.json({
        success: true,
        folder_path: folderPath
      });

    } catch (error) {
      monitoring?.error('Failed to get audio folder path', error);
      res.status(500).json({
        success: false,
        error: error.message
      });
    }
  });

  /**
   * PUT /audio/settings/folder-path
   * Update the bulk audio folder path in settings
   */
  router.put('/settings/folder-path', requireAuthOrAdmin || ((req, res, next) => next()), async (req, res) => {
    try {
      const { folder_path } = req.body;

      if (!folder_path) {
        return res.status(400).json({
          success: false,
          error: 'folder_path is required'
        });
      }

      if (!ewraiDatabase) {
        return res.status(500).json({
          success: false,
          error: 'Database not initialized'
        });
      }

      ewraiDatabase.setAudioBulkFolderPath(folder_path);

      monitoring?.info('Updated audio bulk folder path', { folder_path });

      res.json({
        success: true,
        folder_path: folder_path,
        message: 'Folder path updated successfully'
      });

    } catch (error) {
      monitoring?.error('Failed to update audio folder path', error);
      res.status(500).json({
        success: false,
        error: error.message
      });
    }
  });

  /**
   * GET /audio/settings/pending-path
   * Get the path where pending analysis files are saved (from Python service)
   */
  router.get('/settings/pending-path', async (req, res) => {
    try {
      const response = await fetch(`${pythonServiceUrl}/audio/settings/pending-path`);
      const data = await response.json();
      res.json(data);
    } catch (error) {
      monitoring?.error('Failed to get pending path', error);
      res.status(500).json({
        success: false,
        error: error.message
      });
    }
  });

  /**
   * GET /audio/settings/summary-threshold
   * Get the audio summary threshold setting (in seconds)
   * When audio exceeds this duration, only summary is generated (not full transcription)
   */
  router.get('/settings/summary-threshold', async (req, res) => {
    try {
      if (!ewraiDatabase) {
        return res.status(500).json({
          success: false,
          error: 'Database not initialized'
        });
      }

      const threshold = ewraiDatabase.getSetting('AudioSummaryThreshold');
      const thresholdSeconds = threshold ? parseInt(threshold, 10) : 0;

      monitoring?.info('Retrieved audio summary threshold', { thresholdSeconds });

      res.json({
        success: true,
        threshold_seconds: thresholdSeconds,
        enabled: thresholdSeconds > 0,
        // Also return formatted time for display
        threshold_formatted: thresholdSeconds > 0 ? {
          minutes: Math.floor(thresholdSeconds / 60),
          seconds: thresholdSeconds % 60
        } : null
      });

    } catch (error) {
      monitoring?.error('Failed to get audio summary threshold', error);
      res.status(500).json({
        success: false,
        error: error.message
      });
    }
  });

  /**
   * PUT /audio/settings/summary-threshold
   * Update the audio summary threshold setting
   * @body {number} threshold_seconds - Threshold in seconds (0 to disable)
   * @body {number} minutes - Optional: minutes component
   * @body {number} seconds - Optional: seconds component
   */
  router.put('/settings/summary-threshold', requireAuthOrAdmin || ((req, res, next) => next()), async (req, res) => {
    try {
      let thresholdSeconds;

      // Accept either threshold_seconds directly or minutes/seconds
      if (req.body.threshold_seconds !== undefined) {
        thresholdSeconds = parseInt(req.body.threshold_seconds, 10);
      } else if (req.body.minutes !== undefined || req.body.seconds !== undefined) {
        const minutes = parseInt(req.body.minutes, 10) || 0;
        const seconds = parseInt(req.body.seconds, 10) || 0;
        thresholdSeconds = (minutes * 60) + seconds;
      } else {
        return res.status(400).json({
          success: false,
          error: 'threshold_seconds or minutes/seconds is required'
        });
      }

      // Validate - must be non-negative
      if (thresholdSeconds < 0) {
        return res.status(400).json({
          success: false,
          error: 'Threshold must be non-negative'
        });
      }

      if (!ewraiDatabase) {
        return res.status(500).json({
          success: false,
          error: 'Database not initialized'
        });
      }

      // Save to database
      ewraiDatabase.updateSetting('AudioSummaryThreshold', thresholdSeconds.toString());

      monitoring?.info('Updated audio summary threshold', { thresholdSeconds });

      res.json({
        success: true,
        threshold_seconds: thresholdSeconds,
        enabled: thresholdSeconds > 0,
        threshold_formatted: thresholdSeconds > 0 ? {
          minutes: Math.floor(thresholdSeconds / 60),
          seconds: thresholdSeconds % 60
        } : null,
        message: thresholdSeconds > 0
          ? `Summary threshold set to ${Math.floor(thresholdSeconds / 60)}:${(thresholdSeconds % 60).toString().padStart(2, '0')}`
          : 'Summary threshold disabled (full transcription for all durations)'
      });

    } catch (error) {
      monitoring?.error('Failed to update audio summary threshold', error);
      res.status(500).json({
        success: false,
        error: error.message
      });
    }
  });

  /**
   * POST /audio/bulk/scan
   * Scan a directory for audio files (proxy to Python)
   *
   * Python manages its own configured directory. Node.js just proxies the request.
   */
  router.post('/bulk/scan', requireAuthOrAdmin || ((req, res, next) => next()), async (req, res) => {
    try {
      const { source_directory = null, recursive = true, max_files = 100 } = req.body;

      monitoring?.info('Requesting directory scan from Python service', { source_directory, recursive, max_files });

      // Python service manages its own directory configuration
      // If source_directory is not provided, Python uses its configured AUDIO_BULK_DIR
      const queryParams = new URLSearchParams();
      if (source_directory) {
        queryParams.set('source_directory', source_directory);
      }
      queryParams.set('recursive', recursive.toString());
      queryParams.set('max_files', max_files.toString());

      const response = await fetch(`${pythonServiceUrl}/audio/bulk/scan?${queryParams}`, {
        method: 'POST'
      });

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`Python service error: ${response.statusText} - ${errorText}`);
      }

      const result = await response.json();

      monitoring?.info('Directory scan complete', {
        total_files: result.total_files,
        directory: result.source_directory
      });

      res.json(result);

    } catch (error) {
      monitoring?.error('Failed to scan audio directory', error);
      res.status(500).json({
        success: false,
        error: error.message
      });
    }
  });

  /**
   * POST /audio/bulk/start
   * Start bulk audio processing (proxy to Python)
   */
  router.post('/bulk/start', requireAuthOrAdmin || ((req, res, next) => next()), async (req, res) => {
    try {
      const bulkRequest = req.body;

      // Use settings folder path if not provided
      if (!bulkRequest.source_directory && ewraiDatabase) {
        bulkRequest.source_directory = ewraiDatabase.getAudioBulkFolderPath();
      }

      if (!bulkRequest.source_directory) {
        return res.status(400).json({
          success: false,
          error: 'source_directory is required'
        });
      }

      monitoring?.info('Starting bulk audio processing', {
        source_directory: bulkRequest.source_directory,
        max_files: bulkRequest.max_files
      });

      const response = await fetch(`${pythonServiceUrl}/audio/bulk/start`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(bulkRequest)
      });

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`Python service error: ${response.statusText} - ${errorText}`);
      }

      const result = await response.json();

      monitoring?.info('Bulk audio processing started', result);

      res.json(result);

    } catch (error) {
      monitoring?.error('Failed to start bulk audio processing', error);
      res.status(500).json({
        success: false,
        error: error.message
      });
    }
  });

  /**
   * GET /audio/bulk/status
   * Get bulk processing status (proxy to Python)
   */
  router.get('/bulk/status', async (req, res) => {
    try {
      const response = await fetch(`${pythonServiceUrl}/audio/bulk/status`);

      if (!response.ok) {
        throw new Error(`Python service error: ${response.statusText}`);
      }

      const result = await response.json();

      res.json(result);

    } catch (error) {
      monitoring?.error('Failed to get bulk processing status', error);
      res.status(500).json({
        success: false,
        error: error.message
      });
    }
  });

  /**
   * GET /audio/stats/by-staff
   * Get audio analysis statistics grouped by staff member
   * Shows all users with monitorAnalysis=true, merged with their audio stats from MongoDB
   */
  router.get('/stats/by-staff', async (req, res) => {
    try {
      monitoring?.info('Retrieving audio stats by staff');

      // Get audio stats from Python/MongoDB
      let audioStats = [];
      try {
        const response = await fetch(`${pythonServiceUrl}/audio/stats/by-staff`);
        if (response.ok) {
          const result = await response.json();
          audioStats = result.staff || [];
        }
      } catch (pythonError) {
        monitoring?.warn('Could not fetch audio stats from Python service', pythonError);
      }

      // Get all users with monitorAnalysis enabled from SQLite
      let monitoredUsers = [];
      console.log('[stats/by-staff] ewraiDatabase exists:', !!ewraiDatabase);
      if (ewraiDatabase) {
        try {
          const allUsers = await ewraiDatabase.getAllUsers();
          console.log('[stats/by-staff] Got', allUsers.length, 'users from database');
          const usersWithSettings = allUsers.map(u => ({
            username: u.username,
            monitorAnalysis: u.settings?.monitorAnalysis
          }));
          console.log('[stats/by-staff] Users with settings:', JSON.stringify(usersWithSettings));
          monitoredUsers = allUsers.filter(u => u.settings?.monitorAnalysis === true);
          console.log('[stats/by-staff] Monitored users:', monitoredUsers.length);
        } catch (dbError) {
          console.error('[stats/by-staff] Database error:', dbError);
          monitoring?.warn('Could not fetch monitored users from database', dbError);
        }
      } else {
        console.log('[stats/by-staff] ewraiDatabase is not available!');
      }

      // Create a map of audio stats by staff name (case-insensitive)
      const statsMap = new Map();
      for (const stat of audioStats) {
        const key = (stat.staff_name || '').toLowerCase();
        statsMap.set(key, stat);
      }

      // Build combined staff list: all monitored users + any unmatched audio stats
      const staffList = [];
      const processedNames = new Set();

      // First, add all monitored users (with their stats if available)
      for (const user of monitoredUsers) {
        const displayName = user.displayName || user.username;
        const key = displayName.toLowerCase();
        processedNames.add(key);

        const existingStats = statsMap.get(key);
        if (existingStats) {
          // User has audio stats - use them
          staffList.push(existingStats);
        } else {
          // User has no audio stats yet - create empty entry
          staffList.push({
            staff_name: displayName,
            total_calls: 0,
            mood_counts: {
              HAPPY: 0,
              SAD: 0,
              ANGRY: 0,
              NEUTRAL: 0,
              FEARFUL: 0,
              DISGUSTED: 0,
              SURPRISED: 0
            }
          });
        }
      }

      // Optionally: add any audio stats for staff not in the monitored list
      // (This keeps backward compatibility - shows all staff with recordings)
      for (const stat of audioStats) {
        const key = (stat.staff_name || '').toLowerCase();
        if (!processedNames.has(key)) {
          staffList.push(stat);
        }
      }

      monitoring?.info('Retrieved staff audio statistics', {
        monitored_users: monitoredUsers.length,
        audio_stats_count: audioStats.length,
        combined_staff_count: staffList.length
      });

      res.json({
        success: true,
        staff: staffList
      });

    } catch (error) {
      monitoring?.error('Failed to get audio stats by staff', error);
      res.status(500).json({
        success: false,
        error: error.message
      });
    }
  });

  /**
   * GET /audio/:id
   * Retrieve specific audio analysis by ID
   */
  router.get('/:id', async (req, res) => {
    try {
      const { id } = req.params;

      monitoring?.info(`Retrieving audio analysis: ${id}`);

      const response = await fetch(`${pythonServiceUrl}/audio/${id}`);

      if (!response.ok) {
        if (response.status === 404) {
          return res.status(404).json({
            success: false,
            error: 'Audio analysis not found'
          });
        }
        throw new Error(`Python service error: ${response.statusText}`);
      }

      const result = await response.json();

      monitoring?.info(`Retrieved audio analysis: ${id}`);

      res.json({
        success: true,
        ...result
      });

    } catch (error) {
      monitoring?.error(`Failed to retrieve audio analysis: ${req.params.id}`, error);
      res.status(500).json({
        success: false,
        error: error.message
      });
    }
  });

  /**
   * PUT /audio/:id
   * Update audio analysis metadata by ID
   */
  router.put('/:id', requireAuthOrAdmin || ((req, res, next) => next()), async (req, res) => {
    try {
      const { id } = req.params;
      const updateData = req.body;

      monitoring?.info(`Updating audio analysis: ${id}`);

      const response = await fetch(`${pythonServiceUrl}/audio/${id}`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(updateData)
      });

      if (!response.ok) {
        if (response.status === 404) {
          return res.status(404).json({
            success: false,
            error: 'Audio analysis not found'
          });
        }
        throw new Error(`Python service error: ${response.statusText}`);
      }

      const result = await response.json();

      monitoring?.info(`Updated audio analysis: ${id}`);

      res.json({
        success: true,
        message: 'Audio analysis updated successfully',
        ...result
      });

    } catch (error) {
      monitoring?.error(`Failed to update audio analysis: ${req.params.id}`, error);
      res.status(500).json({
        success: false,
        error: error.message
      });
    }
  });

  /**
   * GET /audio/stream/:filename
   * Stream an audio file from the configured audio directory
   * Proxies to Python service which handles security and Range requests
   */
  router.get('/stream/:filename', async (req, res) => {
    try {
      const { filename } = req.params;
      const rangeHeader = req.headers.range;

      monitoring?.info(`Streaming audio file: ${filename}`, { hasRange: !!rangeHeader });

      // Build headers for Python service request
      const headers = {};
      if (rangeHeader) {
        headers['Range'] = rangeHeader;
      }

      const response = await fetch(`${pythonServiceUrl}/audio/stream/${encodeURIComponent(filename)}`, {
        method: 'GET',
        headers
      });

      if (!response.ok) {
        if (response.status === 400) {
          return res.status(400).json({
            success: false,
            error: 'Invalid filename or unsupported file type'
          });
        }
        if (response.status === 403) {
          return res.status(403).json({
            success: false,
            error: 'Access denied'
          });
        }
        if (response.status === 404) {
          return res.status(404).json({
            success: false,
            error: 'Audio file not found'
          });
        }
        throw new Error(`Python service error: ${response.statusText}`);
      }

      // Forward response headers from Python
      const contentType = response.headers.get('content-type');
      const contentLength = response.headers.get('content-length');
      const contentRange = response.headers.get('content-range');
      const acceptRanges = response.headers.get('accept-ranges');
      const cacheControl = response.headers.get('cache-control');

      if (contentType) res.setHeader('Content-Type', contentType);
      if (contentLength) res.setHeader('Content-Length', contentLength);
      if (contentRange) res.setHeader('Content-Range', contentRange);
      if (acceptRanges) res.setHeader('Accept-Ranges', acceptRanges);
      if (cacheControl) res.setHeader('Cache-Control', cacheControl);

      // Set status code (200 for full content, 206 for partial/range)
      res.status(response.status);

      // Stream the audio data
      const reader = response.body.getReader();

      const pump = async () => {
        try {
          while (true) {
            const { done, value } = await reader.read();
            if (done) {
              res.end();
              break;
            }
            res.write(Buffer.from(value));
          }
        } catch (streamError) {
          monitoring?.error('Audio stream error', streamError);
          if (!res.headersSent) {
            res.status(500).json({ success: false, error: 'Stream error' });
          }
        }
      };

      await pump();

    } catch (error) {
      monitoring?.error(`Failed to stream audio file: ${req.params.filename}`, error);
      if (!res.headersSent) {
        res.status(500).json({
          success: false,
          error: error.message
        });
      }
    }
  });

  /**
   * DELETE /audio/file
   * Delete audio file from source folder (delegated to Python service)
   * Body: { file_path: string }
   * Note: This route MUST be defined before /:id to avoid matching "file" as an ID
   */
  router.delete('/file', requireAuthOrAdmin || ((req, res, next) => next()), async (req, res) => {
    try {
      const { file_path } = req.body;

      // Validate file_path is provided
      if (!file_path) {
        return res.status(400).json({
          success: false,
          error: 'file_path is required'
        });
      }

      monitoring?.info(`Requesting file deletion from Python service: ${file_path}`);

      // Delegate to Python service - Python manages its own file system
      const response = await fetch(`${pythonServiceUrl}/audio/delete-file`, {
        method: 'DELETE',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ file_path })
      });

      if (!response.ok) {
        if (response.status === 404) {
          return res.status(404).json({
            success: false,
            error: 'File not found'
          });
        }
        if (response.status === 403) {
          return res.status(403).json({
            success: false,
            error: 'Permission denied: Cannot delete file'
          });
        }
        throw new Error(`Python service error: ${response.statusText}`);
      }

      const result = await response.json();

      monitoring?.info(`File deletion successful: ${file_path}`);

      res.json({
        success: true,
        message: 'File deleted successfully',
        ...result
      });

    } catch (error) {
      monitoring?.error(`Failed to delete audio file: ${req.body.file_path}`, error);
      res.status(500).json({
        success: false,
        error: error.message
      });
    }
  });

  /**
   * DELETE /audio/:id
   * Delete audio analysis by ID
   */
  router.delete('/:id', requireAuthOrAdmin || ((req, res, next) => next()), async (req, res) => {
    try {
      const { id } = req.params;

      monitoring?.info(`Deleting audio analysis: ${id}`);

      const response = await fetch(`${pythonServiceUrl}/audio/${id}`, {
        method: 'DELETE'
      });

      if (!response.ok) {
        if (response.status === 404) {
          return res.status(404).json({
            success: false,
            error: 'Audio analysis not found'
          });
        }
        throw new Error(`Python service error: ${response.statusText}`);
      }

      const result = await response.json();

      monitoring?.info(`Deleted audio analysis: ${id}`);

      res.json({
        success: true,
        message: 'Audio analysis deleted successfully',
        ...result
      });

    } catch (error) {
      monitoring?.error(`Failed to delete audio analysis: ${req.params.id}`, error);
      res.status(500).json({
        success: false,
        error: error.message
      });
    }
  });

  return router;
}
