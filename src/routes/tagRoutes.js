/**
 * Tag Routes Module
 *
 * Handles tag management operations for document categorization
 * Tags are stored in SQLite database and linked to documents
 */

import express from 'express';

const router = express.Router();

/**
 * Initialize tag routes with required dependencies
 *
 * @param {Object} dependencies - Service dependencies
 * @param {Function} dependencies.requireAuth - Authentication middleware
 * @param {Object} dependencies.ewraiDatabase - EWRAIDatabase instance
 */
export default function createTagRoutes(dependencies) {
  const { requireAuth, ewraiDatabase } = dependencies;

  /**
   * GET /api/admin/tags
   * Get all tags with document counts
   */
  router.get('/', requireAuth, async (req, res) => {
    try {
      if (!ewraiDatabase || !ewraiDatabase.db) {
        return res.status(500).json({
          success: false,
          error: 'Database not initialized'
        });
      }

      // Query all tags with document counts
      const tags = ewraiDatabase.db.prepare(`
        SELECT
          t.id,
          t.name,
          t.description,
          t.created_at as created,
          COUNT(DISTINCT dt.document_id) as documentCount
        FROM tags t
        LEFT JOIN document_tags dt ON t.id = dt.tag_id
        GROUP BY t.id, t.name, t.description, t.created_at
        ORDER BY t.name ASC
      `).all();

      res.json({
        success: true,
        tags: tags.map(tag => ({
          id: tag.id,
          name: tag.name,
          description: tag.description || '',
          created: tag.created,
          documentCount: tag.documentCount || 0
        }))
      });

    } catch (error) {
      console.error('Failed to get tags:', error);
      res.status(500).json({
        success: false,
        error: error.message
      });
    }
  });

  /**
   * POST /api/admin/tags
   * Create a new tag
   */
  router.post('/', requireAuth, async (req, res) => {
    try {
      const { name, description } = req.body;

      if (!name || typeof name !== 'string') {
        return res.status(400).json({
          success: false,
          error: 'Tag name is required'
        });
      }

      // Validate tag name (lowercase, alphanumeric, hyphens, underscores)
      const tagName = name.trim().toLowerCase();
      if (!/^[a-z0-9-_]+$/.test(tagName)) {
        return res.status(400).json({
          success: false,
          error: 'Tag name must contain only lowercase letters, numbers, hyphens, and underscores'
        });
      }

      if (tagName.length < 2 || tagName.length > 50) {
        return res.status(400).json({
          success: false,
          error: 'Tag name must be between 2 and 50 characters'
        });
      }

      if (!ewraiDatabase || !ewraiDatabase.db) {
        return res.status(500).json({
          success: false,
          error: 'Database not initialized'
        });
      }

      // Check if tag already exists
      const existing = ewraiDatabase.db.prepare(`
        SELECT id FROM tags WHERE name = ?
      `).get(tagName);

      if (existing) {
        return res.status(409).json({
          success: false,
          error: `Tag "${tagName}" already exists`
        });
      }

      // Insert new tag
      const result = ewraiDatabase.db.prepare(`
        INSERT INTO tags (name, description)
        VALUES (?, ?)
      `).run(tagName, description || null);

      const tag = ewraiDatabase.db.prepare(`
        SELECT id, name, description, created_at as created
        FROM tags
        WHERE id = ?
      `).get(result.lastInsertRowid);

      console.log(`Created tag: ${tagName} (ID: ${tag.id})`);

      res.json({
        success: true,
        tag: {
          id: tag.id,
          name: tag.name,
          description: tag.description || '',
          created: tag.created,
          documentCount: 0
        }
      });

    } catch (error) {
      console.error('Failed to create tag:', error);
      res.status(500).json({
        success: false,
        error: error.message
      });
    }
  });

  /**
   * PUT /api/admin/tags/:id
   * Update a tag
   */
  router.put('/:id', requireAuth, async (req, res) => {
    try {
      const { id } = req.params;
      const { name, description } = req.body;

      if (!ewraiDatabase || !ewraiDatabase.db) {
        return res.status(500).json({
          success: false,
          error: 'Database not initialized'
        });
      }

      // Check if tag exists
      const existing = ewraiDatabase.db.prepare(`
        SELECT id FROM tags WHERE id = ?
      `).get(id);

      if (!existing) {
        return res.status(404).json({
          success: false,
          error: 'Tag not found'
        });
      }

      // Build update query
      const updates = [];
      const params = [];

      if (name !== undefined) {
        const tagName = name.trim().toLowerCase();
        if (!/^[a-z0-9-_]+$/.test(tagName)) {
          return res.status(400).json({
            success: false,
            error: 'Tag name must contain only lowercase letters, numbers, hyphens, and underscores'
          });
        }
        updates.push('name = ?');
        params.push(tagName);
      }

      if (description !== undefined) {
        updates.push('description = ?');
        params.push(description || null);
      }

      if (updates.length === 0) {
        return res.status(400).json({
          success: false,
          error: 'No updates provided'
        });
      }

      params.push(id);

      ewraiDatabase.db.prepare(`
        UPDATE tags
        SET ${updates.join(', ')}
        WHERE id = ?
      `).run(...params);

      const tag = ewraiDatabase.db.prepare(`
        SELECT
          t.id,
          t.name,
          t.description,
          t.created_at as created,
          COUNT(DISTINCT dt.document_id) as documentCount
        FROM tags t
        LEFT JOIN document_tags dt ON t.id = dt.tag_id
        WHERE t.id = ?
        GROUP BY t.id, t.name, t.description, t.created_at
      `).get(id);

      res.json({
        success: true,
        tag: {
          id: tag.id,
          name: tag.name,
          description: tag.description || '',
          created: tag.created,
          documentCount: tag.documentCount || 0
        }
      });

    } catch (error) {
      console.error('Failed to update tag:', error);
      res.status(500).json({
        success: false,
        error: error.message
      });
    }
  });

  /**
   * DELETE /api/admin/tags/:id
   * Delete a tag
   */
  router.delete('/:id', requireAuth, async (req, res) => {
    try {
      const { id } = req.params;

      if (!ewraiDatabase || !ewraiDatabase.db) {
        return res.status(500).json({
          success: false,
          error: 'Database not initialized'
        });
      }

      // Check if tag exists
      const tag = ewraiDatabase.db.prepare(`
        SELECT id, name FROM tags WHERE id = ?
      `).get(id);

      if (!tag) {
        return res.status(404).json({
          success: false,
          error: 'Tag not found'
        });
      }

      // Delete tag (document_tags will be deleted via CASCADE)
      ewraiDatabase.db.prepare(`
        DELETE FROM tags WHERE id = ?
      `).run(id);

      console.log(`Deleted tag: ${tag.name} (ID: ${id})`);

      res.json({
        success: true,
        message: `Tag "${tag.name}" deleted successfully`
      });

    } catch (error) {
      console.error('Failed to delete tag:', error);
      res.status(500).json({
        success: false,
        error: error.message
      });
    }
  });

  /**
   * GET /api/admin/tags/:id/documents
   * Get all documents with a specific tag
   */
  router.get('/:id/documents', requireAuth, async (req, res) => {
    try {
      const { id } = req.params;

      if (!ewraiDatabase || !ewraiDatabase.db) {
        return res.status(500).json({
          success: false,
          error: 'Database not initialized'
        });
      }

      // Check if tag exists
      const tag = ewraiDatabase.db.prepare(`
        SELECT id, name FROM tags WHERE id = ?
      `).get(id);

      if (!tag) {
        return res.status(404).json({
          success: false,
          error: 'Tag not found'
        });
      }

      // Get documents with this tag
      const documents = ewraiDatabase.db.prepare(`
        SELECT
          dt.document_id,
          dt.added_at
        FROM document_tags dt
        WHERE dt.tag_id = ?
        ORDER BY dt.added_at DESC
      `).all(id);

      res.json({
        success: true,
        tag: {
          id: tag.id,
          name: tag.name
        },
        documents: documents.map(doc => ({
          documentId: doc.document_id,
          addedAt: doc.added_at
        }))
      });

    } catch (error) {
      console.error('Failed to get tag documents:', error);
      res.status(500).json({
        success: false,
        error: error.message
      });
    }
  });

  return router;
}
