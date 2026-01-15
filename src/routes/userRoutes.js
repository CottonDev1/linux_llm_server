/**
 * User Management Routes Module
 *
 * Handles all user-related operations:
 * - Create new users (admin only)
 * - List all users (admin only)
 * - Update user details (admin only)
 * - Delete/deactivate users (admin only)
 */

import express from 'express';

const router = express.Router();

/**
 * Initialize user routes with required dependencies
 *
 * @param {Object} dependencies - Service and middleware dependencies
 * @param {Function} dependencies.requireAuth - Authentication middleware (admin password)
 * @param {Function} dependencies.requireAuthOrAdmin - Auth middleware (JWT or admin password)
 * @param {Object} dependencies.ewraiDatabase - EWRAIDatabase instance
 * @param {Object} dependencies.serverConfig - Server configuration object
 */
export default function initUserRoutes(dependencies) {
  const {
    requireAuth,
    requireAuthOrAdmin,
    ewraiDatabase,
    serverConfig
  } = dependencies;

  // ============================================================================
  // User Management Endpoints (Admin Only)
  // ============================================================================

  /**
   * POST /api/users
   * Create new user (admin only)
   */
  router.post('/', requireAuthOrAdmin, async (req, res) => {
    try {
      // Check if requester is admin
      if (!req.user || req.user.role !== 'admin') {
        return res.status(403).json({ error: 'Admin privileges required' });
      }

      const { username, password, role = 'user', settings } = req.body;

      if (!username || !password) {
        return res.status(400).json({ error: 'Username and password are required' });
      }

      // Create user with default or provided settings
      const newUser = await ewraiDatabase.createUser({
        username,
        password,
        role,
        settings: settings || serverConfig
      });

      res.json({
        success: true,
        user: {
          id: newUser.id,
          username: newUser.username,
          role: newUser.role,
          created: newUser.createdAt
        }
      });
    } catch (error) {
      if (error.message.includes('already exists')) {
        res.status(409).json({ error: error.message });
      } else {
        res.status(500).json({ error: error.message });
      }
    }
  });

  /**
   * GET /api/users
   * List all users (admin only)
   */
  router.get('/', requireAuthOrAdmin, async (req, res) => {
    try {
      // Check if requester is admin
      if (!req.user || req.user.role !== 'admin') {
        return res.status(403).json({ error: 'Admin privileges required' });
      }

      const users = await ewraiDatabase.getAllUsers();

      res.json({
        success: true,
        users: users.map(user => ({
          id: user.id,
          username: user.username,
          role: user.role,
          active: user.active,
          created: user.createdAt,
          lastLogin: user.lastLogin,
          queryCount: user.queryCount || 0
        }))
      });
    } catch (error) {
      res.status(500).json({ error: error.message });
    }
  });

  /**
   * PUT /api/users/:id
   * Update user (admin only)
   */
  router.put('/:id', requireAuthOrAdmin, async (req, res) => {
    try {
      // Check if requester is admin
      if (!req.user || req.user.role !== 'admin') {
        return res.status(403).json({ error: 'Admin privileges required' });
      }

      const { id } = req.params;
      const { password, role, active, settings } = req.body;

      // Build update object
      const updates = {};
      if (password) updates.password = password;
      if (role !== undefined) updates.role = role;
      if (active !== undefined) updates.active = active;
      if (settings) updates.settings = settings;

      const updatedUser = await ewraiDatabase.updateUser(id, updates);

      if (!updatedUser) {
        return res.status(404).json({ error: 'User not found' });
      }

      res.json({
        success: true,
        user: {
          id: updatedUser.id,
          username: updatedUser.username,
          role: updatedUser.role,
          active: updatedUser.active
        }
      });
    } catch (error) {
      res.status(500).json({ error: error.message });
    }
  });

  /**
   * DELETE /api/users/:id
   * Delete/deactivate user (admin only)
   */
  router.delete('/:id', requireAuthOrAdmin, async (req, res) => {
    try {
      // Check if requester is admin
      if (!req.user || req.user.role !== 'admin') {
        return res.status(403).json({ error: 'Admin privileges required' });
      }

      const { id } = req.params;
      const { permanent = false } = req.query;

      // Don't allow deleting the last admin user
      const user = await ewraiDatabase.getUserById(id);
      if (!user) {
        return res.status(404).json({ error: 'User not found' });
      }

      if (user.role === 'admin') {
        const allUsers = await ewraiDatabase.getAllUsers();
        const activeAdmins = allUsers.filter(u => u.role === 'admin' && u.active && u.id !== id);
        if (activeAdmins.length === 0) {
          return res.status(400).json({ error: 'Cannot delete the last admin user' });
        }
      }

      if (permanent) {
        // Permanently delete user (not recommended)
        await ewraiDatabase.deleteUser(id);
        res.json({ success: true, message: 'User permanently deleted' });
      } else {
        // Soft delete - just deactivate the user
        await ewraiDatabase.updateUser(id, { active: false });
        res.json({ success: true, message: 'User deactivated' });
      }
    } catch (error) {
      res.status(500).json({ error: error.message });
    }
  });

  return router;
}
