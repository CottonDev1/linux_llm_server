/**
 * Document Routes Module
 *
 * All document-related route handlers extracted from adminRoutes.js
 * Handles:
 * - Document upload and validation
 * - Document browsing and retrieval
 * - Document updates and deletion
 * - Database statistics
 */

import express from 'express';
import multer from 'multer';

const router = express.Router();

// Allowed MIME types for document upload
const ALLOWED_MIME_TYPES = new Set([
  // Documents
  'application/pdf',
  'application/msword',
  'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
  'application/vnd.ms-excel',
  'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
  // Text
  'text/plain',
  'text/markdown',
  'text/csv',
  'text/x-markdown',
  // Code/Data
  'application/json',
  'application/xml',
  'text/xml',
]);

// Allowed file extensions (backup check)
const ALLOWED_EXTENSIONS = new Set([
  '.pdf', '.doc', '.docx', '.xls', '.xlsx',
  '.txt', '.md', '.csv', '.json', '.xml'
]);

// Multer configuration for file uploads
const storage = multer.memoryStorage();

// File filter to validate MIME type and extension
const fileFilter = (req, file, cb) => {
  // Check MIME type
  const mimeAllowed = ALLOWED_MIME_TYPES.has(file.mimetype);

  // Check extension (defense in depth)
  const ext = file.originalname.toLowerCase().match(/\.[^.]+$/)?.[0] || '';
  const extAllowed = ALLOWED_EXTENSIONS.has(ext);

  if (mimeAllowed || extAllowed) {
    cb(null, true);
  } else {
    cb(new Error(`File type not allowed: ${file.mimetype} (${ext})`), false);
  }
};

const upload = multer({
  storage,
  fileFilter,
  limits: {
    fileSize: 50 * 1024 * 1024 // 50MB limit
  }
});

// Error handling middleware for multer errors
const handleUploadError = (err, req, res, next) => {
  if (err instanceof multer.MulterError) {
    // Multer-specific errors
    if (err.code === 'LIMIT_FILE_SIZE') {
      return res.status(413).json({ error: 'File too large. Maximum size is 50MB.' });
    }
    return res.status(400).json({ error: `Upload error: ${err.message}` });
  } else if (err) {
    // Custom errors (from fileFilter)
    return res.status(415).json({ error: err.message });
  }
  next();
};

/**
 * Initialize document routes with required dependencies
 *
 * @param {Object} dependencies - Service and middleware dependencies
 * @param {Function} dependencies.requireAuth - Authentication middleware
 * @param {Object} dependencies.documentationDB - DocumentationDatabase instance
 * @param {Object} dependencies.db - Vector database instance
 * @param {Object} dependencies.multiTableSearch - MultiTableSearch instance
 * @param {Object} dependencies.adminService - AdminService instance (Phase 2, optional)
 * @param {Object} dependencies.monitoring - ProductionMonitoring instance (optional)
 */
export default function initDocumentRoutes(dependencies) {
  const {
    requireAuth,
    documentationDB,
    db,
    multiTableSearch,
    adminService,
    monitoring
  } = dependencies;

  // ============================================================================
  // Document Upload and Management
  // ============================================================================

  /**
   * POST /api/admin/upload
   * Upload document(s) to a project via Python service
   */
  router.post('/upload', requireAuth, upload.array('files', 10), handleUploadError, async (req, res) => {
    try {
      const {
        project,
        // Three-tier hierarchy
        department,
        type,
        subject,
        // Legacy support
        category
      } = req.body;
      const files = req.files;

      // Use legacy 'category' or new 'type' field
      const docType = type || category || 'documentation';
      const docDepartment = department || 'general';
      const docSubject = subject || null;

      if (!files || files.length === 0) {
        return res.status(400).json({ error: 'No files uploaded' });
      }

      console.log(`📤 Upload request: ${files.length} file(s)`);
      console.log(`   Department: ${docDepartment}, Type: ${docType}, Subject: ${docSubject || 'none'}`);

      // Get Python service URL from environment or default
      const pythonServiceUrl = process.env.PYTHON_SERVICE_URL || 'http://localhost:8001';

      // Use batch endpoint for multiple files (parallel processing)
      // Use single endpoint for single file (simpler)
      if (files.length > 1) {
        // Batch upload - processes files in parallel on Python side
        console.log(`📤 Using batch endpoint for ${files.length} files...`);

        const formData = new FormData();
        for (const file of files) {
          const fileObj = new File([file.buffer], file.originalname, { type: file.mimetype });
          formData.append('files', fileObj);
        }
        formData.append('department', docDepartment);
        formData.append('type', docType);
        if (docSubject) {
          formData.append('subject', docSubject);
        }

        const response = await fetch(`${pythonServiceUrl}/documents/upload-batch`, {
          method: 'POST',
          body: formData
        });

        if (!response.ok) {
          const responseText = await response.text();
          console.error(`❌ Python error response: ${responseText}`);
          let errorData;
          try {
            errorData = JSON.parse(responseText);
          } catch {
            errorData = { detail: responseText || response.statusText };
          }
          throw new Error(errorData.detail || `Python service error: ${response.status}`);
        }

        const batchResult = await response.json();

        // Log individual results
        for (const r of batchResult.results || []) {
          if (r.success) {
            console.log(`   ✅ ${r.filename}: ${r.chunks_created || 0} chunks created`);
          } else {
            console.error(`   ❌ ${r.filename}: ${r.error}`);
          }
        }

        const results = (batchResult.results || []).filter(r => r.success);
        const errors = (batchResult.results || []).filter(r => !r.success);

        res.json({
          success: batchResult.success,
          message: batchResult.message,
          processed: batchResult.successful_count,
          failed: batchResult.failed_count,
          total: batchResult.total_files,
          total_chunks_created: batchResult.total_chunks_created,
          results,
          errors: errors.length > 0 ? errors : undefined
        });

      } else {
        // Single file upload
        const file = files[0];
        const formData = new FormData();
        const fileObj = new File([file.buffer], file.originalname, { type: file.mimetype });
        formData.append('file', fileObj);
        formData.append('department', docDepartment);
        formData.append('type', docType);
        if (docSubject) {
          formData.append('subject', docSubject);
        }

        console.log(`📤 Forwarding ${file.originalname} to Python service...`);
        const response = await fetch(`${pythonServiceUrl}/documents/upload`, {
          method: 'POST',
          body: formData
        });

        console.log(`📥 Python response status: ${response.status}`);

        if (!response.ok) {
          const responseText = await response.text();
          console.error(`❌ Python error response: ${responseText}`);
          let errorData;
          try {
            errorData = JSON.parse(responseText);
          } catch {
            errorData = { detail: responseText || response.statusText };
          }
          throw new Error(errorData.detail || `Python service error: ${response.status}`);
        }

        const result = await response.json();
        console.log(`   ✅ ${file.originalname}: ${result.chunks_created || 0} chunks created`);

        res.json({
          success: true,
          message: `Successfully processed 1 file`,
          processed: 1,
          failed: 0,
          total: 1,
          results: [{
            filename: file.originalname,
            success: true,
            document_id: result.document_id,
            chunks_created: result.chunks_created
          }]
        });
      }

    } catch (error) {
      console.error('❌ Upload error:', error.message);
      res.status(500).json({
        error: 'Upload failed',
        details: error.message
      });
    }
  });

  /**
   * POST /api/admin/validate
   * Validate file before upload (for preview)
   */
  router.post('/validate', requireAuth, upload.single('file'), async (req, res) => {
    try {
      const file = req.file;

      if (!file) {
        return res.status(400).json({ error: 'No file provided' });
      }

      // Validate based on file extension and size
      const ext = file.originalname.split('.').pop()?.toLowerCase() || '';
      const supportedExtensions = ['pdf', 'docx', 'doc', 'txt', 'md', 'csv', 'sql', 'cs', 'js', 'ts', 'jsx', 'tsx', 'json', 'xml'];
      const isValid = supportedExtensions.includes(ext);
      const maxSize = ext === 'csv' ? 50 * 1024 * 1024 : 20 * 1024 * 1024;

      res.json({
        valid: isValid && file.size <= maxSize,
        error: !isValid ? `Unsupported file type: .${ext}` : (file.size > maxSize ? `File too large (max ${maxSize / (1024 * 1024)}MB)` : null),
        fileInfo: {
          name: file.originalname,
          size: file.size,
          type: file.mimetype,
          extension: ext
        }
      });

    } catch (error) {
      res.status(500).json({ error: error.message });
    }
  });

  /**
   * GET /api/admin/documents
   * Browse database contents (no auth required for now)
   */
  router.get('/documents', async (req, res) => {
    try {
      const { project, category, limit = 50, offset = 0 } = req.query;

      // Query documentation database for uploaded documents only
      console.log(`📄 Fetching uploaded documents (limit: ${limit}, offset: ${offset})`);

      // Initialize documentation DB if needed
      if (!documentationDB.isInitialized) {
        await documentationDB.initialize();
      }

      // Get documents from documentation database
      const documents = await documentationDB.listDocuments(parseInt(limit), parseInt(offset));

      // Filter by category if specified
      let filteredDocs = documents;
      if (category) {
        filteredDocs = documents.filter(doc => doc.category === category);
      }

      // Group by category for summary
      const summary = filteredDocs.reduce((acc, doc) => {
        const cat = doc.category || 'unknown';
        if (!acc[cat]) acc[cat] = 0;
        acc[cat]++;
        return acc;
      }, {});

      res.json({
        documents: filteredDocs.map(doc => ({
          id: doc.id,
          project: project || 'uploaded', // Mark as uploaded documents
          category: doc.category,
          documentType: doc.documentType,
          file: doc.fileName,
          preview: '', // No preview in list view for performance
          uploadDate: doc.uploadDate,
          chunks: doc.chunks,
          tags: doc.tags,
          metadata: {
            fileName: doc.fileName,
            fileSize: doc.fileSize,
            category: doc.category,
            documentType: doc.documentType
          }
        })),
        summary,
        total: filteredDocs.length,
        offset: parseInt(offset),
        limit: parseInt(limit)
      });

    } catch (error) {
      console.error('❌ Failed to fetch documents:', error);
      res.status(500).json({ error: error.message });
    }
  });

  /**
   * GET /api/admin/documents/:documentId/content
   * Get a single document with full content
   */
  router.get('/documents/:documentId/content', requireAuth, async (req, res) => {
    try {
      const { documentId } = req.params;

      console.log(`📄 Fetching full content for document: ${documentId}`);

      // Initialize documentation DB if needed
      if (!documentationDB.isInitialized) {
        await documentationDB.initialize();
      }

      // Get full document with all chunks
      const document = await documentationDB.getDocument(documentId);

      if (!document) {
        return res.status(404).json({ error: 'Document not found' });
      }

      res.json({
        success: true,
        document: {
          id: document.id,
          title: document.title,
          content: document.content,
          category: document.category,
          documentType: document.documentType,
          fileName: document.fileName,
          fileSize: document.fileSize,
          uploadDate: document.uploadDate,
          tags: document.tags,
          metadata: document.metadata,
          chunks: document.chunks
        }
      });

    } catch (error) {
      console.error('❌ Failed to fetch document content:', error);
      res.status(500).json({ error: error.message });
    }
  });

  /**
   * GET /api/admin/documents/table/:tableName
   * Get all documents from a specific table
   */
  router.get('/documents/table/:tableName', requireAuth, async (req, res) => {
    try {
      const { tableName } = req.params;
      const { limit = 1000 } = req.query;

      console.log(`📋 Fetching documents from table: ${tableName}`);

      // Initialize databases if needed
      if (!db.isInitialized) await db.initialize();
      if (!multiTableSearch.isInitialized) await multiTableSearch.initialize();

      let documents = [];
      let table = null;

      // Determine which table to query
      if (tableName === 'code_context' && db.table) {
        table = db.table;
      } else if (multiTableSearch.tables && multiTableSearch.tables[tableName]) {
        table = multiTableSearch.tables[tableName];
      }

      if (!table) {
        return res.status(404).json({
          success: false,
          error: `Table '${tableName}' not found`
        });
      }

      // Query all documents from the table using vector search with dummy vector
      const dummyVector = new Array(384).fill(0.001);
      const results = await table
        .search(dummyVector)
        .limit(parseInt(limit))
        .toArray();

      // Format the results
      documents = results.map(row => ({
        id: row.id || row.documentId || row.document_id,
        document: row.document || row.content || row.definition || '',
        project: row.project,
        database: row.database,
        tableName: row.tableName,
        procedureName: row.procedureName,
        type: row.type,
        timestamp: row.timestamp,
        ...row // Include all other fields
      }));

      console.log(`✅ Retrieved ${documents.length} documents from ${tableName}`);

      res.json({
        success: true,
        tableName,
        documents,
        total: documents.length
      });

    } catch (error) {
      console.error(`❌ Failed to fetch documents from table:`, error.message);
      res.status(500).json({
        success: false,
        error: error.message
      });
    }
  });

  /**
   * PUT /api/admin/documents/:documentId
   * Update document with partial or full data
   * Preserves existing metadata for fields not provided in request
   * Handles re-chunking if content changes
   */
  router.put('/documents/:documentId', requireAuth, async (req, res) => {
    try {
      const { documentId } = req.params;
      const {
        title,
        content,
        category,
        documentType,
        fileName,
        tags,
        metadata
      } = req.body;

      console.log(`📝 Updating document: ${documentId}`);

      // Initialize documentation DB if needed
      if (!documentationDB.isInitialized) {
        await documentationDB.initialize();
      }

      // Get existing document to preserve metadata
      const existingDoc = await documentationDB.getDocument(documentId);

      if (!existingDoc) {
        return res.status(404).json({
          success: false,
          error: 'Document not found'
        });
      }

      // Build updated document data - preserve existing values if not provided
      const updatedDocData = {
        title: title !== undefined ? title : existingDoc.title,
        content: content !== undefined ? content : existingDoc.content,
        category: category !== undefined ? category : existingDoc.category,
        documentType: documentType !== undefined ? documentType : existingDoc.documentType,
        fileName: fileName !== undefined ? fileName : existingDoc.fileName,
        fileSize: content !== undefined ? Buffer.byteLength(content, 'utf8') : existingDoc.fileSize,
        tags: tags !== undefined ? tags : existingDoc.tags,
        metadata: metadata !== undefined ? { ...existingDoc.metadata, ...metadata } : existingDoc.metadata
      };

      console.log(`   🗑️  Deleting existing document and chunks...`);

      // Delete existing document and all its chunks
      await documentationDB.deleteDocument(documentId);

      console.log(`   ✅ Deleted old version`);
      console.log(`   📝 Adding updated document...`);

      // Store document - Python service handles chunking and embedding
      const uploadDate = new Date().toISOString();
      await documentationDB.store(
        updatedDocData.title,
        updatedDocData.content,
        {
          department: updatedDocData.category || 'general',
          type: updatedDocData.documentType || 'documentation',
          fileName: updatedDocData.fileName,
          fileSize: updatedDocData.fileSize,
          tags: updatedDocData.tags && updatedDocData.tags.length > 0 ? updatedDocData.tags : ['untagged'],
          ...updatedDocData.metadata
        }
      );

      console.log(`   ✅ Updated document "${updatedDocData.title}" (${updatedDocData.content.length} chars)`);

      res.json({
        success: true,
        message: 'Document updated successfully',
        document: {
          id: documentId,
          title: updatedDocData.title,
          category: updatedDocData.category,
          documentType: updatedDocData.documentType,
          fileName: updatedDocData.fileName,
          uploadDate: uploadDate
        }
      });

    } catch (error) {
      console.error('❌ Update error:', error.message);
      res.status(500).json({
        success: false,
        error: 'Failed to update document',
        message: error.message
      });
    }
  });

  /**
   * DELETE /api/admin/documents/:documentId
   * Delete document (requires authentication)
   */
  router.delete('/documents/:documentId', requireAuth, async (req, res) => {
    try {
      const { documentId } = req.params;

      console.log(`🗑️  Deleting document: ${documentId}`);

      // Use documentationDB for uploaded documents
      await documentationDB.deleteDocument(documentId);

      res.json({
        success: true,
        message: 'Document deleted successfully'
      });

    } catch (error) {
      console.error('❌ Delete error:', error.message);
      res.status(500).json({
        success: false,
        error: 'Failed to delete document',
        message: error.message
      });
    }
  });

  /**
   * GET /api/admin/stats
   * Get database statistics
   */
  router.get('/stats', requireAuth, async (req, res) => {
    try {
      const stats = await db.getStats();

      // TODO: Query for project/category breakdown
      res.json({
        totalDocuments: stats.totalDocuments,
        cacheSize: stats.cacheSize,
        // Will be enhanced with actual project/category counts
      });

    } catch (error) {
      res.status(500).json({ error: error.message });
    }
  });

  return router;
}
