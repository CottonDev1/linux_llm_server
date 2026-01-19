/**
 * Authentication Routes
 * Handles user authentication, registration, and token management
 */

import express from 'express';
import { authenticateToken, generateAccessToken, generateRefreshToken } from '../middleware/authMiddleware.js';
import SettingsManager from '../db/SettingsManager.js';

/**
 * Create authentication routes with ewraiDatabase instance
 */
export default function createAuthRoutes(ewraiDatabase) {
  const router = express.Router();

/**
 * POST /api/auth/login
 * Authenticate user and return JWT tokens with Windows/SQL auth fallback
 */
router.post('/login', async (req, res) => {
  try {
    const { username, password } = req.body;

    if (!username || !password) {
      return res.status(400).json({ error: 'Username and password are required' });
    }

    // Create SettingsManager on-demand to ensure database is initialized
    const settingsManager = new SettingsManager(ewraiDatabase);

    let user = null;
    const authMode = settingsManager.getAuthMode();
    const authMethods = [];

    // Try Windows authentication first if enabled
    if (authMode === 'Windows') {
      // Check if Windows auth settings are properly configured
      const { valid, missingFields } = settingsManager.validateWindowsAuthSettings();

      if (valid) {
        try {
          user = await ewraiDatabase.authenticateWindowsUser(username, password);
          if (user) {
            authMethods.push('Windows');
          }
        } catch (winAuthError) {
          console.error('Windows authentication failed:', winAuthError.message);
          // Continue to SQL fallback
        }
      } else {
        console.warn('Windows auth enabled but not configured. Missing fields:', missingFields);
      }
    }

    // Fallback to SQL authentication if Windows auth failed or not enabled
    if (!user) {
      try {
        user = await ewraiDatabase.authenticateUser(username, password);
        if (user) {
          authMethods.push('SQL');
        }
      } catch (sqlAuthError) {
        console.error('SQL authentication failed:', sqlAuthError.message);
      }
    }

    if (!user) {
      console.log(`Authentication failed for user: ${username}. Attempted methods: ${authMethods.join(', ') || 'none'}`);
      return res.status(401).json({ error: 'Invalid credentials' });
    }

    // Log successful authentication method
    console.log(`User ${username} authenticated successfully via ${authMethods[0]}`);

    // Generate tokens
    const accessToken = generateAccessToken(user);
    const refreshTokenData = await ewraiDatabase.createRefreshToken(user.id);

    res.json({
      accessToken,
      refreshToken: refreshTokenData.token,
      user: {
        id: user.id,
        username: user.username,
        email: user.email,
        role: user.role,
        settings: user.settings
      },
      authMethod: authMethods[0], // Include which method was successful
      forcePasswordReset: user.forcePasswordReset || false // Include force password reset flag
    });
  } catch (error) {
    console.error('Login error:', error);
    res.status(500).json({ error: error.message });
  }
});

/**
 * POST /api/auth/test-credentials
 * Test authentication credentials without creating a session (admin only)
 */
router.post('/test-credentials', authenticateToken, async (req, res) => {
  try {
    // Check admin access
    if (req.user.role !== 'admin') {
      return res.status(403).json({ error: 'Admin access required to test credentials' });
    }

    const { username, password, authType = 'auto' } = req.body;

    if (!username || !password) {
      return res.status(400).json({ error: 'Username and password are required' });
    }

    // Create SettingsManager on-demand to ensure database is initialized
    const settingsManager = new SettingsManager(ewraiDatabase);

    const results = {
      username,
      testedMethods: [],
      windowsAuth: {
        tested: false,
        success: false,
        error: null,
        configured: false,
        missingFields: []
      },
      sqlAuth: {
        tested: false,
        success: false,
        error: null
      },
      systemSettings: {
        authMode: settingsManager.getAuthMode(),
        windowsAuthEnabled: settingsManager.isWindowsAuthEnabled()
      }
    };

    // Test Windows authentication if requested or in auto mode with Windows enabled
    if (authType === 'windows' || (authType === 'auto' && settingsManager.isWindowsAuthEnabled())) {
      results.windowsAuth.tested = true;
      results.testedMethods.push('Windows');

      // Check if Windows auth settings are properly configured
      const { valid, missingFields } = settingsManager.validateWindowsAuthSettings();
      results.windowsAuth.configured = valid;
      results.windowsAuth.missingFields = missingFields;

      if (valid) {
        try {
          const user = await ewraiDatabase.authenticateWindowsUser(username, password);
          if (user) {
            results.windowsAuth.success = true;
            results.windowsAuth.userInfo = {
              id: user.id,
              username: user.username,
              email: user.email,
              role: user.role,
              displayName: user.displayName,
              groups: user.groups || []
            };
          } else {
            results.windowsAuth.error = 'Authentication failed - invalid credentials';
          }
        } catch (winAuthError) {
          results.windowsAuth.error = winAuthError.message;
        }
      } else {
        results.windowsAuth.error = `Windows auth not properly configured. Missing: ${missingFields.join(', ')}`;
      }
    }

    // Test SQL authentication if requested or in auto mode
    if (authType === 'sql' || authType === 'auto') {
      results.sqlAuth.tested = true;
      results.testedMethods.push('SQL');

      try {
        const user = await ewraiDatabase.authenticateUser(username, password);
        if (user) {
          results.sqlAuth.success = true;
          results.sqlAuth.userInfo = {
            id: user.id,
            username: user.username,
            email: user.email,
            role: user.role,
            displayName: user.displayName || user.username
          };
        } else {
          results.sqlAuth.error = 'Authentication failed - invalid credentials';
        }
      } catch (sqlAuthError) {
        results.sqlAuth.error = sqlAuthError.message;
      }
    }

    // Determine overall success and provide recommendations
    const anySuccess = results.windowsAuth.success || results.sqlAuth.success;
    results.overallSuccess = anySuccess;

    if (anySuccess) {
      const successfulMethods = [];
      if (results.windowsAuth.success) successfulMethods.push('Windows');
      if (results.sqlAuth.success) successfulMethods.push('SQL');
      results.message = `Authentication successful via: ${successfulMethods.join(' and ')}`;
    } else {
      results.message = 'Authentication failed for all tested methods';
    }

    // Add recommendations
    results.recommendations = [];
    if (results.windowsAuth.tested && !results.windowsAuth.configured) {
      results.recommendations.push('Configure Windows authentication settings before enabling Windows auth mode');
    }
    if (results.windowsAuth.success && results.systemSettings.authMode !== 'Windows') {
      results.recommendations.push('Consider enabling Windows authentication mode since Windows auth succeeded');
    }
    if (!results.sqlAuth.success && results.sqlAuth.tested) {
      results.recommendations.push('Ensure user has a local SQL account for fallback authentication');
    }

    res.json(results);
  } catch (error) {
    console.error('Test credentials error:', error);
    res.status(500).json({ error: error.message });
  }
});

/**
 * POST /api/auth/refresh
 * Refresh access token using refresh token
 */
router.post('/refresh', async (req, res) => {
  try {
    const { refreshToken } = req.body;

    if (!refreshToken) {
      return res.status(400).json({ error: 'Refresh token is required' });
    }

    // Verify refresh token
    const user = await ewraiDatabase.verifyRefreshToken(refreshToken);

    if (!user) {
      return res.status(403).json({ error: 'Invalid or expired refresh token' });
    }

    // Generate new access token
    const accessToken = generateAccessToken(user);

    res.json({ accessToken });
  } catch (error) {
    console.error('Refresh token error:', error);
    res.status(500).json({ error: error.message });
  }
});

/**
 * POST /api/auth/logout
 * Revoke refresh token
 */
router.post('/logout', authenticateToken, async (req, res) => {
  try {
    const { refreshToken } = req.body;

    if (refreshToken) {
      await ewraiDatabase.revokeRefreshToken(refreshToken);
    }

    res.json({ message: 'Logged out successfully' });
  } catch (error) {
    console.error('Logout error:', error);
    res.status(500).json({ error: error.message });
  }
});

/**
 * GET /api/auth/me
 * Get current user information
 */
router.get('/me', authenticateToken, async (req, res) => {
  try {
    const user = await ewraiDatabase.getUserById(req.user.id);

    if (!user) {
      return res.status(404).json({ error: 'User not found' });
    }

    res.json({
      id: user.id,
      username: user.username,
      email: user.email,
      role: user.role,
      permissions: user.permissions,
      settings: user.settings,
      createdAt: user.createdAt,
      lastLogin: user.lastLogin
    });
  } catch (error) {
    console.error('Get user error:', error);
    res.status(500).json({ error: error.message });
  }
});

/**
 * PUT /api/auth/settings
 * Update user settings
 */
router.put('/settings', authenticateToken, async (req, res) => {
  try {
    const settings = req.body;

    // Update user settings in database
    const updatedSettings = await ewraiDatabase.updateUserSettings(req.user.id, settings);

    res.json({
      message: 'Settings updated successfully',
      settings: updatedSettings
    });
  } catch (error) {
    console.error('Update settings error:', error);
    res.status(500).json({ error: error.message });
  }
});

/**
 * PUT /api/auth/password
 * Change user password
 */
router.put('/password', authenticateToken, async (req, res) => {
  try {
    const { currentPassword, newPassword } = req.body;

    if (!currentPassword || !newPassword) {
      return res.status(400).json({ error: 'Current and new password are required' });
    }

    if (newPassword.length < 8) {
      return res.status(400).json({ error: 'Password must be at least 8 characters' });
    }

    await ewraiDatabase.updatePassword(req.user.id, currentPassword, newPassword);

    res.json({ message: 'Password updated successfully' });
  } catch (error) {
    console.error('Update password error:', error);
    res.status(400).json({ error: error.message });
  }
});

/**
 * POST /api/auth/sql-connection-settings
 * Save SQL connection settings for the authenticated user
 */
router.post('/sql-connection-settings', authenticateToken, async (req, res) => {
  try {
    const userId = req.user.id;
    const settings = req.body;

    ewraiDatabase.saveSqlConnectionSettings(userId, settings);

    res.json({ success: true, message: 'SQL connection settings saved' });
  } catch (error) {
    console.error('Error saving SQL connection settings:', error);
    res.status(500).json({ success: false, error: error.message });
  }
});

/**
 * GET /api/auth/sql-connection-settings
 * Get SQL connection settings for the authenticated user
 */
router.get('/sql-connection-settings', authenticateToken, async (req, res) => {
  try {
    const userId = req.user.id;
    const settings = ewraiDatabase.getSqlConnectionSettings(userId);

    res.json({ success: true, settings: settings || null });
  } catch (error) {
    console.error('Error getting SQL connection settings:', error);
    res.status(500).json({ success: false, error: error.message });
  }
});

  // Admin middleware
  const requireAdmin = (req, res, next) => {
    if (req.user.role !== 'admin') {
      return res.status(403).json({ error: 'Admin access required' });
    }
    next();
  };

  /**
   * GET /api/auth/users
   * Get all users (admin only)
   */
  router.get('/users', authenticateToken, requireAdmin, async (req, res) => {
    try {
      const users = await ewraiDatabase.getAllUsers();
      // Remove sensitive data but include settings for filtering
      const sanitizedUsers = users.map(user => ({
        id: user.id,
        username: user.username,
        email: user.email,
        role: user.role,
        isActive: user.isActive,
        createdAt: user.createdAt,
        lastLogin: user.lastLogin,
        settings: user.settings || {}
      }));
      res.json(sanitizedUsers);
    } catch (error) {
      console.error('Error fetching users:', error);
      res.status(500).json({ error: 'Failed to fetch users' });
    }
  });

  /**
   * GET /api/auth/users/:userId
   * Get single user (admin only)
   */
  router.get('/users/:userId', authenticateToken, requireAdmin, async (req, res) => {
    try {
      const user = await ewraiDatabase.getUserById(req.params.userId);
      if (!user) {
        return res.status(404).json({ error: 'User not found' });
      }

      // Remove sensitive data
      delete user.password;
      delete user.refreshTokens;

      res.json(user);
    } catch (error) {
      console.error('Error fetching user:', error);
      res.status(500).json({ error: 'Failed to fetch user' });
    }
  });

  /**
   * POST /api/auth/users
   * Create new user (admin only)
   */
  router.post('/users', authenticateToken, requireAdmin, async (req, res) => {
    try {
      const { username, password, email, role = 'user' } = req.body;

      if (!username || !password) {
        return res.status(400).json({ error: 'Username and password are required' });
      }

      if (password.length < 8) {
        return res.status(400).json({ error: 'Password must be at least 8 characters' });
      }

      // Create user
      const newUser = await ewraiDatabase.createUser({
        username,
        password,
        email,
        role,
        displayName: username // Use username as display name by default
      });

      // Remove sensitive data
      delete newUser.password;
      delete newUser.refreshTokens;

      res.status(201).json(newUser);
    } catch (error) {
      console.error('Error creating user:', error);
      if (error.message.includes('already exists')) {
        res.status(400).json({ error: 'Username already exists' });
      } else {
        res.status(500).json({ error: 'Failed to create user' });
      }
    }
  });

  /**
   * PUT /api/auth/users/:userId
   * Update user (admin only)
   */
  router.put('/users/:userId', authenticateToken, requireAdmin, async (req, res) => {
    try {
      const { userId } = req.params;
      const { password, email, role, isActive, forcePasswordReset, settings } = req.body;

      const user = await ewraiDatabase.getUserById(userId);
      if (!user) {
        return res.status(404).json({ error: 'User not found' });
      }

      // Update fields if provided
      const updates = {};

      if (email !== undefined) {
        updates.email = email;
      }

      if (role !== undefined) {
        updates.role = role;
      }

      if (isActive !== undefined) {
        updates.isActive = isActive;
      }

      if (forcePasswordReset !== undefined) {
        updates.forcePasswordReset = forcePasswordReset;
      }

      // Update user
      await ewraiDatabase.updateUser(userId, updates);

      // Update password separately if provided
      if (password) {
        if (password.length < 8) {
          return res.status(400).json({ error: 'Password must be at least 8 characters' });
        }
        await ewraiDatabase.setUserPassword(userId, password);
      }

      // Update user settings (page restrictions) if provided
      if (settings !== undefined) {
        await ewraiDatabase.updateUserSettings(userId, settings);
      }

      // Get updated user
      const updatedUser = await ewraiDatabase.getUserById(userId);

      // Remove sensitive data
      delete updatedUser.password;
      delete updatedUser.refreshTokens;

      res.json(updatedUser);
    } catch (error) {
      console.error('Error updating user:', error);
      res.status(500).json({ error: 'Failed to update user' });
    }
  });

  /**
   * DELETE /api/auth/users/:userId
   * Delete user (admin only)
   */
  router.delete('/users/:userId', authenticateToken, requireAdmin, async (req, res) => {
    try {
      const { userId } = req.params;

      // Prevent deleting yourself
      if (userId === req.user.id) {
        return res.status(400).json({ error: 'Cannot delete your own account' });
      }

      const user = await ewraiDatabase.getUserById(userId);
      if (!user) {
        return res.status(404).json({ error: 'User not found' });
      }

      await ewraiDatabase.deleteUser(userId);
      res.json({ message: 'User deleted successfully' });
    } catch (error) {
      console.error('Error deleting user:', error);
      res.status(500).json({ error: 'Failed to delete user' });
    }
  });

  /**
   * POST /api/auth/users/:userId/reset-password
   * Reset user password and optionally force password change on next login (admin only)
   */
  router.post('/users/:userId/reset-password', authenticateToken, requireAdmin, async (req, res) => {
    try {
      const { userId } = req.params;
      const { newPassword, forceChangeOnLogin = true } = req.body;

      if (!newPassword) {
        return res.status(400).json({ error: 'New password is required' });
      }

      if (newPassword.length < 8) {
        return res.status(400).json({ error: 'Password must be at least 8 characters' });
      }

      const user = await ewraiDatabase.getUserById(userId);
      if (!user) {
        return res.status(404).json({ error: 'User not found' });
      }

      // Set the new password
      await ewraiDatabase.setUserPassword(userId, newPassword);

      // Set force password reset flag if requested
      if (forceChangeOnLogin) {
        await ewraiDatabase.updateUser(userId, { forcePasswordReset: true });
      }

      console.log(`Password reset for user ${user.username} by admin ${req.user.username}`);

      res.json({
        message: 'Password reset successfully',
        forceChangeOnLogin
      });
    } catch (error) {
      console.error('Error resetting password:', error);
      res.status(500).json({ error: 'Failed to reset password' });
    }
  });

  /**
   * POST /api/auth/force-change-password
   * Change password for user who was required to change password on login
   */
  router.post('/force-change-password', authenticateToken, async (req, res) => {
    try {
      const { newPassword } = req.body;

      if (!newPassword) {
        return res.status(400).json({ error: 'New password is required' });
      }

      if (newPassword.length < 8) {
        return res.status(400).json({ error: 'Password must be at least 8 characters' });
      }

      // Validate at least 1 special character
      const specialCharRegex = /[!@#$%^&*()_+\-=\[\]{};':"\\|,.<>\/?]+/;
      if (!specialCharRegex.test(newPassword)) {
        return res.status(400).json({ error: 'Password must contain at least 1 special character' });
      }

      // Validate at least 2 numbers
      const numberCount = (newPassword.match(/\d/g) || []).length;
      if (numberCount < 2) {
        return res.status(400).json({ error: 'Password must contain at least 2 numbers' });
      }

      const user = await ewraiDatabase.getUserById(req.user.id);
      if (!user) {
        return res.status(404).json({ error: 'User not found' });
      }

      // Verify user is required to change password
      if (!user.forcePasswordReset) {
        return res.status(400).json({ error: 'Password change is not required' });
      }

      // Set the new password
      await ewraiDatabase.setUserPassword(req.user.id, newPassword);

      // Clear the force password reset flag
      await ewraiDatabase.updateUser(req.user.id, { forcePasswordReset: false });

      console.log(`User ${user.username} completed forced password change`);

      res.json({ message: 'Password changed successfully' });
    } catch (error) {
      console.error('Error changing password:', error);
      res.status(500).json({ error: 'Failed to change password' });
    }
  });

  /**
   * GET /api/auth/role-permissions
   * Get all role permissions (admin only)
   */
  router.get('/role-permissions', authenticateToken, requireAdmin, async (req, res) => {
    try {
      const permissions = await ewraiDatabase.getRolePermissions();
      res.json({ permissions });
    } catch (error) {
      console.error('Error getting role permissions:', error);
      res.status(500).json({ error: 'Failed to get role permissions' });
    }
  });

  /**
   * PUT /api/auth/role-permissions
   * Update all role permissions (admin only)
   */
  router.put('/role-permissions', authenticateToken, requireAdmin, async (req, res) => {
    try {
      const { permissions } = req.body;

      if (!permissions || typeof permissions !== 'object') {
        return res.status(400).json({ error: 'Invalid permissions data' });
      }

      // Validate structure - each key should be a valid role with an array of permissions
      const availableRoles = ewraiDatabase.getAllRoles();
      for (const role of Object.keys(permissions)) {
        if (!availableRoles.includes(role)) {
          return res.status(400).json({ error: `Invalid role: ${role}` });
        }
        if (!Array.isArray(permissions[role])) {
          return res.status(400).json({ error: `Invalid permissions for role: ${role}` });
        }
      }

      ewraiDatabase.setRolePermissions(permissions);
      console.log(`Role permissions updated by admin ${req.user.username}`);

      res.json({ message: 'Role permissions updated successfully', permissions });
    } catch (error) {
      console.error('Error updating role permissions:', error);
      res.status(500).json({ error: 'Failed to update role permissions' });
    }
  });

  /**
   * GET /api/auth/role-permissions/:role
   * Get permissions for a specific role (authenticated users)
   */
  router.get('/role-permissions/:role', authenticateToken, async (req, res) => {
    try {
      const { role } = req.params;

      // Check if role exists
      if (!ewraiDatabase.roleExists(role)) {
        return res.status(400).json({ error: 'Invalid role' });
      }

      const allPermissions = ewraiDatabase.getRolePermissions();
      const rolePages = allPermissions[role] || [];

      res.json({ role, pages: rolePages });
    } catch (error) {
      console.error('Error getting role permissions:', error);
      res.status(500).json({ error: 'Failed to get role permissions' });
    }
  });

  /**
   * GET /api/auth/roles
   * Get all available roles
   */
  router.get('/roles', authenticateToken, async (req, res) => {
    try {
      const roles = ewraiDatabase.getAllRoles();
      res.json({ roles });
    } catch (error) {
      console.error('Error getting roles:', error);
      res.status(500).json({ error: 'Failed to get roles' });
    }
  });

  /**
   * POST /api/auth/roles
   * Create a new custom role (admin only)
   */
  router.post('/roles', authenticateToken, requireAdmin, async (req, res) => {
    try {
      const { name, description } = req.body;

      if (!name) {
        return res.status(400).json({ error: 'Role name is required' });
      }

      // Validate role name format
      if (!/^[a-z][a-z0-9_]*$/.test(name)) {
        return res.status(400).json({
          error: 'Role name must start with a lowercase letter and contain only lowercase letters, numbers, and underscores'
        });
      }

      // Check if role already exists
      if (ewraiDatabase.roleExists(name)) {
        return res.status(400).json({ error: 'A role with this name already exists' });
      }

      const role = ewraiDatabase.createRole(name, description);
      console.log(`Role "${name}" created by admin ${req.user.username}`);

      res.status(201).json({ role });
    } catch (error) {
      console.error('Error creating role:', error);
      res.status(500).json({ error: 'Failed to create role' });
    }
  });

  /**
   * DELETE /api/auth/roles/:roleName
   * Delete a custom role (admin only)
   */
  router.delete('/roles/:roleName', authenticateToken, requireAdmin, async (req, res) => {
    try {
      const { roleName } = req.params;

      ewraiDatabase.deleteRole(roleName);
      console.log(`Role "${roleName}" deleted by admin ${req.user.username}`);

      res.json({ message: 'Role deleted successfully' });
    } catch (error) {
      console.error('Error deleting role:', error);
      if (error.message === 'Role not found') {
        return res.status(404).json({ error: 'Role not found' });
      }
      if (error.message === 'Cannot delete built-in roles') {
        return res.status(400).json({ error: 'Cannot delete built-in roles' });
      }
      res.status(500).json({ error: 'Failed to delete role' });
    }
  });

  /**
   * GET /api/auth/role-categories
   * Get all role->category mappings
   * Returns: { user: [], developer: [], admin: ['documentation', 'documents', ...] }
   */
  router.get('/role-categories', authenticateToken, async (req, res) => {
    try {
      const roles = ewraiDatabase.getAllRoles();
      const rows = ewraiDatabase.getRoleCategoriesAll();

      // Initialize mappings for all roles
      const mappings = {};
      for (const role of roles) {
        mappings[role] = [];
      }

      // Fill in categories
      for (const row of rows) {
        if (mappings[row.role] !== undefined) {
          mappings[row.role].push(row.category);
        } else {
          // Role exists in permissions but not in roles table - add it
          mappings[row.role] = [row.category];
        }
      }

      res.json(mappings);
    } catch (error) {
      console.error('Error getting role categories:', error);
      res.status(500).json({ error: 'Failed to get role categories' });
    }
  });

  /**
   * GET /api/auth/role-categories/:role
   * Get categories for a specific role
   */
  router.get('/role-categories/:role', authenticateToken, async (req, res) => {
    try {
      const { role } = req.params;

      // Check if role exists
      if (!ewraiDatabase.roleExists(role)) {
        return res.status(400).json({ error: 'Invalid role' });
      }

      const categories = await ewraiDatabase.getRoleCategories(role);
      res.json({ role, categories });
    } catch (error) {
      console.error('Error getting role categories:', error);
      res.status(500).json({ error: 'Failed to get role categories' });
    }
  });

  /**
   * PUT /api/auth/role-categories/:role
   * Replace all categories for a role
   * Body: { categories: ['Documentation', 'SQL', ...] }
   */
  router.put('/role-categories/:role', authenticateToken, requireAdmin, async (req, res) => {
    try {
      const { role } = req.params;
      const { categories } = req.body;

      // Check if role exists
      if (!ewraiDatabase.roleExists(role)) {
        return res.status(400).json({ error: 'Invalid role' });
      }

      if (!Array.isArray(categories)) {
        return res.status(400).json({ error: 'Categories must be an array' });
      }

      ewraiDatabase.setRoleCategories(role, categories);
      console.log(`Role categories for ${role} updated by admin ${req.user.username}`);

      res.json({ message: 'Role categories updated successfully', role, categories });
    } catch (error) {
      console.error('Error updating role categories:', error);
      res.status(500).json({ error: 'Failed to update role categories' });
    }
  });

  /**
   * POST /api/auth/role-categories/:role/:category
   * Add a single category to a role
   */
  router.post('/role-categories/:role/:category', authenticateToken, requireAdmin, async (req, res) => {
    try {
      const { role, category } = req.params;

      // Check if role exists
      if (!ewraiDatabase.roleExists(role)) {
        return res.status(400).json({ error: 'Invalid role' });
      }

      if (!category || category.trim() === '') {
        return res.status(400).json({ error: 'Category cannot be empty' });
      }

      ewraiDatabase.addRoleCategory(role, category);
      console.log(`Category ${category} added to role ${role} by admin ${req.user.username}`);

      res.json({ message: 'Category added successfully', role, category });
    } catch (error) {
      console.error('Error adding role category:', error);
      res.status(500).json({ error: 'Failed to add role category' });
    }
  });

  /**
   * DELETE /api/auth/role-categories/:role/:category
   * Remove a category from a role and delete all user permissions for users with this role and category
   */
  router.delete('/role-categories/:role/:category', authenticateToken, requireAdmin, async (req, res) => {
    try {
      const { role, category } = req.params;

      // Check if role exists
      if (!ewraiDatabase.roleExists(role)) {
        return res.status(400).json({ error: 'Invalid role' });
      }

      if (!category || category.trim() === '') {
        return res.status(400).json({ error: 'Category cannot be empty' });
      }

      ewraiDatabase.removeRoleCategory(role, category);
      console.log(`Category ${category} removed from role ${role} by admin ${req.user.username}`);

      res.json({ message: 'Category removed successfully', role, category });
    } catch (error) {
      console.error('Error removing role category:', error);
      res.status(500).json({ error: 'Failed to remove role category' });
    }
  });

  /**
   * GET /api/auth/user-permissions/:userId
   * Get user's enabled pages grouped by category
   * Returns: { category1: ['page1', 'page2'], category2: [...] }
   * Admin only or requesting own permissions
   */
  router.get('/user-permissions/:userId', authenticateToken, async (req, res) => {
    try {
      const { userId } = req.params;

      // Check if admin or requesting own permissions
      if (req.user.role !== 'admin' && req.user.id !== userId) {
        return res.status(403).json({ error: 'Admin access required or can only view own permissions' });
      }

      const permissions = await ewraiDatabase.getUserPermissions(userId);
      res.json(permissions);
    } catch (error) {
      console.error('Error getting user permissions:', error);
      res.status(500).json({ error: 'Failed to get user permissions' });
    }
  });

  /**
   * PUT /api/auth/user-permissions/:userId
   * Replace all page permissions for the user
   * Body: { permissions: { category1: ['page1', 'page2'], category2: [...] } }
   */
  router.put('/user-permissions/:userId', authenticateToken, requireAdmin, async (req, res) => {
    try {
      const { userId } = req.params;
      const { permissions } = req.body;

      if (!permissions || typeof permissions !== 'object') {
        return res.status(400).json({ error: 'Invalid permissions data' });
      }

      // Verify user exists
      const user = await ewraiDatabase.getUserById(userId);
      if (!user) {
        return res.status(404).json({ error: 'User not found' });
      }

      await ewraiDatabase.setUserPermissions(userId, permissions);
      console.log(`User permissions for user ${user.username} updated by admin ${req.user.username}`);

      res.json({ message: 'User permissions updated successfully', userId, permissions });
    } catch (error) {
      console.error('Error updating user permissions:', error);
      res.status(500).json({ error: 'Failed to update user permissions' });
    }
  });

  /**
   * GET /api/auth/user-effective-permissions/:userId
   * Get the effective permissions (role categories + individual pages)
   * Returns what the user can actually access
   * Admin only or requesting own permissions
   */
  router.get('/user-effective-permissions/:userId', authenticateToken, async (req, res) => {
    try {
      const { userId } = req.params;

      // Check if admin or requesting own permissions
      if (req.user.role !== 'admin' && req.user.id !== userId) {
        return res.status(403).json({ error: 'Admin access required or can only view own permissions' });
      }

      // Get user to determine role
      const user = await ewraiDatabase.getUserById(userId);
      if (!user) {
        return res.status(404).json({ error: 'User not found' });
      }

      // Get role categories
      const roleCategories = await ewraiDatabase.getRoleCategories(user.role);

      // Get user-specific permissions
      const userPermissions = await ewraiDatabase.getUserPermissions(userId);

      // Combine for effective permissions
      const effectivePermissions = {
        role: user.role,
        roleCategories: roleCategories,
        userPermissions: userPermissions,
        combinedAccess: {
          categories: roleCategories,
          pages: userPermissions
        }
      };

      res.json(effectivePermissions);
    } catch (error) {
      console.error('Error getting user effective permissions:', error);
      res.status(500).json({ error: 'Failed to get user effective permissions' });
    }
  });

  return router;
}
