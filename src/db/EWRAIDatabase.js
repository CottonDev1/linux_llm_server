/**
 * EWR AI Database Service
 * Handles all database operations for EWR AI system using SQLite
 */

import Database from 'better-sqlite3';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';
import { existsSync, mkdirSync } from 'fs';
import bcrypt from 'bcryptjs';
import { randomUUID } from 'crypto';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

class EWRAIDatabase {
  constructor(dbPath = null) {
    this.dbPath = dbPath || join(__dirname, '../../data/EWR_AI.db');
    this.db = null;
  }

  /**
   * Initialize database and create tables
   */
  async initialize() {
    try {
      // Ensure data directory exists
      const dataDir = dirname(this.dbPath);
      if (!existsSync(dataDir)) {
        mkdirSync(dataDir, { recursive: true });
      }

      this.db = new Database(this.dbPath);
      this.db.pragma('journal_mode = WAL');

      // Create users table
      this.db.exec(`
        CREATE TABLE IF NOT EXISTS users (
          user_id TEXT PRIMARY KEY,
          username TEXT UNIQUE NOT NULL,
          display_name TEXT NOT NULL,
          email TEXT,
          password_hash TEXT,
          auth_type TEXT DEFAULT 'local',
          role TEXT DEFAULT 'user',
          department TEXT,
          is_active INTEGER DEFAULT 1,
          force_password_reset INTEGER DEFAULT 0,
          created_at TEXT DEFAULT CURRENT_TIMESTAMP,
          last_login_at TEXT,
          UNIQUE(username)
        )
      `);

      // Add force_password_reset column if it doesn't exist (for existing databases)
      try {
        this.db.exec(`ALTER TABLE users ADD COLUMN force_password_reset INTEGER DEFAULT 0`);
      } catch (e) {
        // Column already exists, ignore
      }

      // Create user_settings table
      this.db.exec(`
        CREATE TABLE IF NOT EXISTS user_settings (
          user_id TEXT PRIMARY KEY,
          default_model TEXT DEFAULT 'qwen2.5-coder:1.5b',
          temperature REAL DEFAULT 0.7,
          context_size INTEGER DEFAULT 2000,
          num_sources INTEGER DEFAULT 3,
          preferred_interface TEXT DEFAULT 'code',
          theme TEXT DEFAULT 'dark',
          enable_streaming INTEGER DEFAULT 1,
          enable_code_flow INTEGER DEFAULT 1,
          enable_sql_mode INTEGER DEFAULT 1,
          max_tokens_per_query INTEGER DEFAULT 4096,
          max_queries_per_hour INTEGER DEFAULT 100,
          max_concurrent_queries INTEGER DEFAULT 3,
          enabled_pages TEXT DEFAULT '[]',
          disabled_pages TEXT DEFAULT '[]',
          updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
          FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE
        )
      `);

      // Add enabled_pages column if it doesn't exist (for existing databases)
      try {
        this.db.exec(`ALTER TABLE user_settings ADD COLUMN enabled_pages TEXT DEFAULT '[]'`);
      } catch (e) {
        // Column already exists, ignore
      }

      // Add disabled_pages column if it doesn't exist (for existing databases)
      // This is the inverse approach - store what's blocked, not what's allowed
      try {
        this.db.exec(`ALTER TABLE user_settings ADD COLUMN disabled_pages TEXT DEFAULT '[]'`);
      } catch (e) {
        // Column already exists, ignore
      }

      // Add monitor_analysis column if it doesn't exist (for existing databases)
      // Controls whether user's audio analysis results appear on Staff Results page
      try {
        this.db.exec(`ALTER TABLE user_settings ADD COLUMN monitor_analysis INTEGER DEFAULT 0`);
      } catch (e) {
        // Column already exists, ignore
      }

      // Drop old user_permissions table and recreate with new schema
      this.db.exec(`DROP TABLE IF EXISTS user_permissions`);

      // Create user_permissions table for page-level permissions
      this.db.exec(`
        CREATE TABLE IF NOT EXISTS user_permissions (
          user_id TEXT NOT NULL,
          category TEXT NOT NULL,
          page_id TEXT NOT NULL,
          PRIMARY KEY (user_id, category, page_id),
          FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE
        )
      `);

      // Create refresh_tokens table
      this.db.exec(`
        CREATE TABLE IF NOT EXISTS refresh_tokens (
          token_id TEXT PRIMARY KEY,
          user_id TEXT NOT NULL,
          token_hash TEXT NOT NULL,
          expires_at TEXT NOT NULL,
          created_at TEXT DEFAULT CURRENT_TIMESTAMP,
          FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE
        )
      `);

      // Create session_tracking table for audit
      this.db.exec(`
        CREATE TABLE IF NOT EXISTS session_tracking (
          session_id TEXT PRIMARY KEY,
          user_id TEXT NOT NULL,
          ip_address TEXT,
          user_agent TEXT,
          login_at TEXT DEFAULT CURRENT_TIMESTAMP,
          logout_at TEXT,
          FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE
        )
      `);

      // Create settings_history for audit trail
      this.db.exec(`
        CREATE TABLE IF NOT EXISTS settings_history (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          user_id TEXT NOT NULL,
          setting_key TEXT NOT NULL,
          old_value TEXT,
          new_value TEXT,
          changed_at TEXT DEFAULT CURRENT_TIMESTAMP,
          FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE
        )
      `);

      // Create settings table for system configuration
      this.db.exec(`
        CREATE TABLE IF NOT EXISTS settings (
          section TEXT NOT NULL,
          setting_value TEXT NOT NULL,
          PRIMARY KEY (section)
        )
      `);

      // Create git_repositories table for repository management
      this.db.exec(`
        CREATE TABLE IF NOT EXISTS git_repositories (
          repo_id TEXT PRIMARY KEY,
          name TEXT UNIQUE NOT NULL,
          display_name TEXT NOT NULL,
          path TEXT NOT NULL,
          branch TEXT DEFAULT 'master',
          project_name TEXT,
          access_token TEXT,
          last_pull_time TEXT,
          last_commit_hash TEXT,
          last_commit_message TEXT,
          last_commit_author TEXT,
          last_analysis_date TEXT,
          created_at TEXT DEFAULT CURRENT_TIMESTAMP,
          updated_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
      `);

      // Migrate existing git_repositories table to add new columns if they don't exist
      try {
        this.db.exec(`
          ALTER TABLE git_repositories ADD COLUMN project_name TEXT;
        `);
      } catch (e) {
        // Column already exists, ignore
      }

      try {
        this.db.exec(`
          ALTER TABLE git_repositories ADD COLUMN access_token TEXT;
        `);
      } catch (e) {
        // Column already exists, ignore
      }

      try {
        this.db.exec(`
          ALTER TABLE git_repositories ADD COLUMN last_analysis_date TEXT;
        `);
      } catch (e) {
        // Column already exists, ignore
      }

      try {
        this.db.exec(`
          ALTER TABLE git_repositories ADD COLUMN sync_interval INTEGER DEFAULT 300000;
        `);
      } catch (e) {
        // Column already exists, ignore
      }

      try {
        this.db.exec(`
          ALTER TABLE git_repositories ADD COLUMN auto_sync INTEGER DEFAULT 0;
        `);
      } catch (e) {
        // Column already exists, ignore
      }

      // Create tags table for document categorization
      this.db.exec(`
        CREATE TABLE IF NOT EXISTS tags (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          name TEXT UNIQUE NOT NULL,
          description TEXT,
          created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
      `);

      // Create document_tags junction table
      this.db.exec(`
        CREATE TABLE IF NOT EXISTS document_tags (
          document_id TEXT NOT NULL,
          tag_id INTEGER NOT NULL,
          added_at TEXT DEFAULT CURRENT_TIMESTAMP,
          PRIMARY KEY (document_id, tag_id),
          FOREIGN KEY (tag_id) REFERENCES tags(id) ON DELETE CASCADE
        )
      `);

      // Create role_permissions table for new permission system
      this.db.exec(`
        CREATE TABLE IF NOT EXISTS role_permissions (
          role TEXT NOT NULL,
          category TEXT NOT NULL,
          PRIMARY KEY (role, category)
        )
      `);

      // user_permissions table already created above with new schema

      // Initialize default settings
      const defaultSettings = [
        { section: 'AuthMode', setting_value: 'SQL' },
        { section: 'WindowsDomain', setting_value: '' },
        { section: 'WindowsDC', setting_value: '' },
        { section: 'LDAPBaseDN', setting_value: '' },
        { section: 'LDAPBindUser', setting_value: '' },
        { section: 'LDAPBindPassword', setting_value: '' },
        { section: 'ADAdminGroups', setting_value: 'Domain Admins,IT Admins' },
        { section: 'ADDeveloperGroups', setting_value: 'Developers,Engineering' },
        // Service configuration settings
        { section: 'MongoDBUri', setting_value: 'mongodb://localhost:27017' },
        { section: 'MongoDBDatabase', setting_value: 'rag_server' },
        { section: 'PythonServiceUrl', setting_value: 'http://localhost:8001' },
        { section: 'LLMHost', setting_value: 'http://localhost:11434' },
        { section: 'NodeServerPort', setting_value: '3000' }
      ];

      const insertSetting = this.db.prepare('INSERT OR IGNORE INTO settings (section, setting_value) VALUES (?, ?)');
      for (const setting of defaultSettings) {
        insertSetting.run(setting.section, setting.setting_value);
      }

      // Create indexes
      this.db.exec(`
        CREATE INDEX IF NOT EXISTS idx_users_username ON users(username);
        CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);
        CREATE INDEX IF NOT EXISTS idx_refresh_tokens_user ON refresh_tokens(user_id);
        CREATE INDEX IF NOT EXISTS idx_session_tracking_user ON session_tracking(user_id);
        CREATE INDEX IF NOT EXISTS idx_git_repositories_name ON git_repositories(name);
        CREATE INDEX IF NOT EXISTS idx_tags_name ON tags(name);
        CREATE INDEX IF NOT EXISTS idx_document_tags_document ON document_tags(document_id);
        CREATE INDEX IF NOT EXISTS idx_document_tags_tag ON document_tags(tag_id);
        CREATE INDEX IF NOT EXISTS idx_role_permissions_role ON role_permissions(role);
        CREATE INDEX IF NOT EXISTS idx_user_permissions_user ON user_permissions(user_id);
        CREATE INDEX IF NOT EXISTS idx_user_permissions_category ON user_permissions(category);
      `);

      // Create default admin user if no users exist
      await this.ensureDefaultAdmin();

      // Initialize default role permissions (Admin gets all categories)
      await this.ensureDefaultRolePermissions();

      console.log('✅ EWR_AI database initialized successfully');
    } catch (error) {
      console.error('❌ Failed to initialize EWR_AI database:', error);
      throw error;
    }
  }

  /**
   * Ensure default admin user exists
   */
  async ensureDefaultAdmin() {
    try {
      const stmt = this.db.prepare('SELECT COUNT(*) as count FROM users WHERE role = ?');
      const result = stmt.get('admin');

      if (result.count === 0) {
        console.log('📌 Creating default admin user...');

        // Create admin user with simple password for development
        await this.createUser({
          username: 'Admin',
          displayName: 'Administrator',
          email: 'admin@localhost',
          password: '123', // Simple password as specified
          role: 'admin',
          department: 'IT'
        });

        console.log('✅ Default admin user created:');
        console.log('   Username: Admin');
        console.log('   Password: 123');
      }
    } catch (error) {
      console.error('❌ Failed to create default admin user:', error);
      throw error;
    }
  }

  /**
   * Ensure default role permissions exist
   * Admin role gets all categories by default
   * User and Developer roles start with no categories (empty)
   */
  async ensureDefaultRolePermissions() {
    try {
      // Check if admin role has any categories assigned
      const stmt = this.db.prepare('SELECT COUNT(*) as count FROM role_permissions WHERE role = ?');
      const result = stmt.get('admin');

      if (result.count === 0) {
        console.log('📌 Initializing default role permissions...');

        // All available categories from NAV_CATEGORIES
        const allCategories = ['documentation', 'documents', 'sql', 'git', 'audio', 'system'];

        // Admin gets ALL categories
        const insertStmt = this.db.prepare('INSERT OR IGNORE INTO role_permissions (role, category) VALUES (?, ?)');

        for (const category of allCategories) {
          insertStmt.run('admin', category);
        }

        console.log('✅ Admin role initialized with all categories:', allCategories.join(', '));
        console.log('   User and Developer roles start empty (no category access)');
      }
    } catch (error) {
      console.error('❌ Failed to initialize default role permissions:', error);
      throw error;
    }
  }

  /**
   * Create a new user
   */
  async createUser({ username, displayName, email, password, role = 'user', department = null }) {
    try {
      const userId = randomUUID();
      const passwordHash = await bcrypt.hash(password, 12);

      const stmt = this.db.prepare(`
        INSERT INTO users (
          user_id, username, display_name, email, password_hash,
          role, department
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
      `);

      stmt.run(userId, username, displayName, email, passwordHash, role, department);

      // Create default settings
      await this.createDefaultSettings(userId);

      // Create default permissions
      await this.createDefaultPermissions(userId, role);

      return this.getUserById(userId);
    } catch (error) {
      if (error.code === 'SQLITE_CONSTRAINT') {
        throw new Error('Username already exists');
      }
      throw error;
    }
  }

  /**
   * Create default settings for a user
   */
  async createDefaultSettings(userId) {
    const stmt = this.db.prepare(`
      INSERT INTO user_settings (user_id) VALUES (?)
    `);
    stmt.run(userId);
  }

  /**
   * Create default permissions for a user
   * Note: With the new permission system, user permissions are derived from role_permissions.
   * Individual page permissions (user_permissions table) are only created when customizing access.
   * New users start with full access to their role's categories.
   */
  async createDefaultPermissions(userId, role) {
    // No default user_permissions needed - users inherit from role_permissions
    // User-specific restrictions are added via the user-edit page
  }

  /**
   * Authenticate user with username and password
   */
  async authenticateUser(username, password) {
    try {
      const user = this.getUserByUsername(username);

      if (!user || !user.isActive) {
        return null;
      }

      const isValid = await bcrypt.compare(password, user.passwordHash);

      if (!isValid) {
        return null;
      }

      // Update last login time
      this.updateLastLogin(user.userId);

      // Remove password hash from returned user object
      delete user.passwordHash;

      return user;
    } catch (error) {
      console.error('Authentication error:', error);
      return null;
    }
  }

  /**
   * Get user by ID
   */
  getUser(userId) {
    const stmt = this.db.prepare(`
      SELECT
        u.*,
        s.*
      FROM users u
      LEFT JOIN user_settings s ON u.user_id = s.user_id
      WHERE u.user_id = ?
    `);

    const row = stmt.get(userId);
    return row ? this.formatUser(row) : null;
  }

  /**
   * Get user by ID (alias for getUser)
   */
  getUserById(userId) {
    return this.getUser(userId);
  }

  /**
   * Get user by username (includes password hash for authentication)
   */
  getUserByUsername(username) {
    const stmt = this.db.prepare(`
      SELECT
        u.*,
        s.*
      FROM users u
      LEFT JOIN user_settings s ON u.user_id = s.user_id
      WHERE u.username = ?
    `);

    const row = stmt.get(username);
    return row ? this.formatUser(row, true) : null; // Include password hash
  }

  /**
   * Verify user password
   */
  async verifyPassword(userId, password) {
    const stmt = this.db.prepare(`SELECT password_hash FROM users WHERE user_id = ?`);
    const row = stmt.get(userId);

    if (!row) return false;
    return await bcrypt.compare(password, row.password_hash);
  }

  /**
   * Update user's last login time
   */
  updateLastLogin(userId) {
    const stmt = this.db.prepare(`
      UPDATE users SET last_login_at = CURRENT_TIMESTAMP WHERE user_id = ?
    `);
    stmt.run(userId);
  }

  /**
   * Update user settings (alias for updateSettings)
   */
  updateUserSettings(userId, settings) {
    this.updateSettings(userId, settings);
    return this.getSettings(userId);
  }

  /**
   * Update user settings
   */
  updateSettings(userId, settings) {
    // Ensure user_settings row exists (for users created before settings system)
    const ensureRow = this.db.prepare(`
      INSERT OR IGNORE INTO user_settings (user_id) VALUES (?)
    `);
    ensureRow.run(userId);

    const allowedFields = [
      'default_model', 'temperature', 'context_size', 'num_sources',
      'preferred_interface', 'theme', 'enable_streaming', 'enable_code_flow',
      'enable_sql_mode', 'max_tokens_per_query'
    ];

    const updates = [];
    const values = [];

    for (const [key, value] of Object.entries(settings)) {
      if (allowedFields.includes(key)) {
        updates.push(`${key} = ?`);
        values.push(value);
      }
    }

    // Handle enabledPages specially (stored as JSON) - DEPRECATED, use disabledPages
    if (settings.enabledPages !== undefined) {
      updates.push('enabled_pages = ?');
      values.push(JSON.stringify(settings.enabledPages));
    }

    // Handle disabledPages (inverse approach - store what's blocked)
    if (settings.disabledPages !== undefined) {
      updates.push('disabled_pages = ?');
      values.push(JSON.stringify(settings.disabledPages));
    }

    // Handle monitorAnalysis (boolean for Staff Results page)
    if (settings.monitorAnalysis !== undefined) {
      updates.push('monitor_analysis = ?');
      values.push(settings.monitorAnalysis ? 1 : 0);
    }

    if (updates.length === 0) return;

    values.push(userId);

    const stmt = this.db.prepare(`
      UPDATE user_settings
      SET ${updates.join(', ')}, updated_at = CURRENT_TIMESTAMP
      WHERE user_id = ?
    `);

    stmt.run(...values);
  }

  /**
   * Get user settings
   */
  getSettings(userId) {
    const stmt = this.db.prepare(`SELECT * FROM user_settings WHERE user_id = ?`);
    const row = stmt.get(userId);

    if (!row) return this.getDefaultSettings();

    return {
      // LLM settings
      llm: {
        model: row.default_model || 'qwen2.5-coder:1.5b',
        temperature: row.temperature || 0.7
      },
      // RAG settings
      rag: {
        maxSourceLength: row.context_size || 2000,
        numSources: row.num_sources || 3
      },
      // Rate limits
      limits: {
        maxConcurrentRequests: row.max_concurrent_queries || 3
      },
      // UI preferences
      interface: {
        preferred: row.preferred_interface || 'code',
        theme: row.theme || 'dark',
        enableStreaming: Boolean(row.enable_streaming),
        enableCodeFlow: Boolean(row.enable_code_flow),
        enableSQLMode: Boolean(row.enable_sql_mode)
      },
      // Legacy flat structure for compatibility
      defaultModel: row.default_model,
      temperature: row.temperature,
      contextSize: row.context_size,
      numSources: row.num_sources,
      preferredInterface: row.preferred_interface,
      theme: row.theme,
      enableStreaming: Boolean(row.enable_streaming),
      enableCodeFlow: Boolean(row.enable_code_flow),
      enableSQLMode: Boolean(row.enable_sql_mode),
      maxTokensPerQuery: row.max_tokens_per_query,
      maxQueriesPerHour: row.max_queries_per_hour,
      maxConcurrentQueries: row.max_concurrent_queries
    };
  }

  /**
   * Get default settings
   */
  getDefaultSettings() {
    return {
      llm: {
        model: 'qwen2.5-coder:1.5b',
        temperature: 0.7
      },
      rag: {
        maxSourceLength: 2000,
        numSources: 3
      },
      limits: {
        maxConcurrentRequests: 3
      },
      interface: {
        preferred: 'code',
        theme: 'dark',
        enableStreaming: true,
        enableCodeFlow: true,
        enableSQLMode: true
      }
    };
  }

  /**
   * Update user password
   */
  async updatePassword(userId, currentPassword, newPassword) {
    try {
      // First verify the current password
      const stmt = this.db.prepare('SELECT password_hash FROM users WHERE user_id = ?');
      const row = stmt.get(userId);

      if (!row) {
        throw new Error('User not found');
      }

      const isValid = await bcrypt.compare(currentPassword, row.password_hash);

      if (!isValid) {
        throw new Error('Current password is incorrect');
      }

      // Hash new password and update
      const newPasswordHash = await bcrypt.hash(newPassword, 12);
      const updateStmt = this.db.prepare('UPDATE users SET password_hash = ? WHERE user_id = ?');
      updateStmt.run(newPasswordHash, userId);

      return true;
    } catch (error) {
      throw error;
    }
  }

  /**
   * Store refresh token
   */
  async storeRefreshToken(userId, token, expiresAt) {
    const tokenId = randomUUID();
    const tokenHash = await bcrypt.hash(token, 10);

    const stmt = this.db.prepare(`
      INSERT INTO refresh_tokens (token_id, user_id, token_hash, expires_at)
      VALUES (?, ?, ?, ?)
    `);

    stmt.run(tokenId, userId, tokenHash, expiresAt);
    return tokenId;
  }

  /**
   * Create refresh token for user
   */
  async createRefreshToken(userId) {
    try {
      const token = randomUUID();
      const expiresAt = new Date(Date.now() + 30 * 24 * 60 * 60 * 1000).toISOString(); // 30 days

      const tokenId = await this.storeRefreshToken(userId, token, expiresAt);

      return {
        tokenId,
        token,
        expiresAt
      };
    } catch (error) {
      throw error;
    }
  }

  /**
   * Verify refresh token (accepts token only, returns user object)
   */
  async verifyRefreshToken(token) {
    try {
      // Get all non-expired tokens and check against the provided token
      const stmt = this.db.prepare(`
        SELECT user_id, token_hash, expires_at
        FROM refresh_tokens
        WHERE expires_at > datetime('now')
      `);

      const rows = stmt.all();

      for (const row of rows) {
        const isValid = await bcrypt.compare(token, row.token_hash);
        if (isValid) {
          // Return the user object
          return this.getUser(row.user_id);
        }
      }

      return null;
    } catch (error) {
      console.error('Error verifying refresh token:', error);
      return null;
    }
  }

  /**
   * Revoke refresh token
   */
  revokeRefreshToken(token) {
    // Delete all tokens matching this token hash
    // In production, we'd hash and match properly
    const stmt = this.db.prepare(`DELETE FROM refresh_tokens WHERE token_id = ? OR token_hash = ?`);
    stmt.run(token, token);
  }

  /**
   * Track user session
   */
  trackSession(userId, sessionId, ipAddress, userAgent) {
    const stmt = this.db.prepare(`
      INSERT INTO session_tracking (session_id, user_id, ip_address, user_agent)
      VALUES (?, ?, ?, ?)
    `);
    stmt.run(sessionId, userId, ipAddress, userAgent);
  }

  /**
   * End user session
   */
  endSession(sessionId) {
    const stmt = this.db.prepare(`
      UPDATE session_tracking SET logout_at = CURRENT_TIMESTAMP WHERE session_id = ?
    `);
    stmt.run(sessionId);
  }

  /**
   * Get all users
   */
  async getAllUsers() {
    try {
      const stmt = this.db.prepare(`
        SELECT u.*, s.*
        FROM users u
        LEFT JOIN user_settings s ON u.user_id = s.user_id
        ORDER BY u.created_at DESC
      `);
      const rows = stmt.all();
      return rows.map(row => this.formatUser(row));
    } catch (error) {
      console.error('Error getting all users:', error);
      return [];
    }
  }

  /**
   * Get user by ID (async version with settings)
   */
  async getUserById(userId) {
    try {
      const stmt = this.db.prepare(`
        SELECT u.*, s.*
        FROM users u
        LEFT JOIN user_settings s ON u.user_id = s.user_id
        WHERE u.user_id = ?
      `);
      const row = stmt.get(userId);
      return row ? this.formatUser(row) : null;
    } catch (error) {
      console.error('Error getting user by ID:', error);
      return null;
    }
  }

  /**
   * Update user
   */
  async updateUser(userId, updates) {
    try {
      const fields = [];
      const values = [];

      if (updates.email !== undefined) {
        fields.push('email = ?');
        values.push(updates.email);
      }

      if (updates.role !== undefined) {
        fields.push('role = ?');
        values.push(updates.role);
      }

      if (updates.isActive !== undefined) {
        fields.push('is_active = ?');
        values.push(updates.isActive ? 1 : 0);
      }

      if (updates.displayName !== undefined) {
        fields.push('display_name = ?');
        values.push(updates.displayName);
      }

      if (updates.department !== undefined) {
        fields.push('department = ?');
        values.push(updates.department);
      }

      if (updates.forcePasswordReset !== undefined) {
        fields.push('force_password_reset = ?');
        values.push(updates.forcePasswordReset ? 1 : 0);
      }

      // Update user_settings table if settings are provided
      if (updates.settings) {
        this.updateSettings(userId, updates.settings);
      }

      // Update users table if there are fields to update
      if (fields.length > 0) {
        values.push(userId);
        const sql = `UPDATE users SET ${fields.join(', ')} WHERE user_id = ?`;
        const stmt = this.db.prepare(sql);
        stmt.run(...values);
      }

      return true;
    } catch (error) {
      console.error('Error updating user:', error);
      throw error;
    }
  }

  /**
   * Set user password (admin function)
   */
  async setUserPassword(userId, newPassword) {
    try {
      const hashedPassword = await bcrypt.hash(newPassword, 10);
      const stmt = this.db.prepare(`
        UPDATE users
        SET password_hash = ?
        WHERE user_id = ?
      `);
      stmt.run(hashedPassword, userId);
      return true;
    } catch (error) {
      console.error('Error setting user password:', error);
      throw error;
    }
  }

  /**
   * Delete user
   */
  async deleteUser(userId) {
    try {
      // Delete refresh tokens first
      const deleteTokens = this.db.prepare(`DELETE FROM refresh_tokens WHERE user_id = ?`);
      deleteTokens.run(userId);

      // Delete session tracking
      const deleteSessions = this.db.prepare(`DELETE FROM session_tracking WHERE user_id = ?`);
      deleteSessions.run(userId);

      // Delete user
      const deleteUser = this.db.prepare(`DELETE FROM users WHERE user_id = ?`);
      deleteUser.run(userId);

      return true;
    } catch (error) {
      console.error('Error deleting user:', error);
      throw error;
    }
  }

  /**
   * Format user object
   */
  formatUser(row, includePasswordHash = false) {
    const user = {
      userId: row.user_id,
      id: row.user_id, // Also include 'id' for compatibility
      username: row.username,
      displayName: row.display_name,
      email: row.email,
      authType: row.auth_type,
      role: row.role,
      department: row.department,
      isActive: Boolean(row.is_active),
      forcePasswordReset: Boolean(row.force_password_reset),
      createdAt: row.created_at,
      lastLoginAt: row.last_login_at,
      lastLogin: row.last_login_at, // Also include 'lastLogin' for compatibility
      settings: {
        defaultModel: row.default_model,
        temperature: row.temperature,
        contextSize: row.context_size,
        numSources: row.num_sources,
        preferredInterface: row.preferred_interface,
        theme: row.theme,
        enableStreaming: Boolean(row.enable_streaming),
        enableCodeFlow: Boolean(row.enable_code_flow),
        enableSQLMode: Boolean(row.enable_sql_mode),
        maxTokensPerQuery: row.max_tokens_per_query,
        maxQueriesPerHour: row.max_queries_per_hour,
        maxConcurrentQueries: row.max_concurrent_queries,
        enabledPages: JSON.parse(row.enabled_pages || '[]'),  // DEPRECATED
        disabledPages: JSON.parse(row.disabled_pages || '[]'), // Pages user cannot access
        monitorAnalysis: Boolean(row.monitor_analysis) // Show user's audio results on Staff Results page
      },
      permissions: {
        // Legacy permissions removed - now using role_permissions and user_permissions tables
      }
    };

    // Include password hash only when needed for authentication
    if (includePasswordHash) {
      user.passwordHash = row.password_hash;
    }

    return user;
  }

  /**
   * List all users (admin function)
   */
  listUsers() {
    const stmt = this.db.prepare(`
      SELECT
        u.user_id, u.username, u.display_name, u.email, u.role,
        u.department, u.is_active, u.created_at, u.last_login_at
      FROM users u
      ORDER BY u.created_at DESC
    `);

    return stmt.all();
  }

  /**
   * Get system setting
   */
  getSetting(section) {
    const stmt = this.db.prepare('SELECT setting_value FROM settings WHERE section = ?');
    const row = stmt.get(section);
    return row ? row.setting_value : null;
  }

  /**
   * Get all system settings
   */
  getAllSettings() {
    const stmt = this.db.prepare('SELECT section, setting_value FROM settings');
    const rows = stmt.all();
    const settings = {};
    for (const row of rows) {
      settings[row.section] = row.setting_value;
    }
    return settings;
  }

  /**
   * Update system setting
   */
  updateSetting(section, settingValue) {
    const stmt = this.db.prepare('INSERT OR REPLACE INTO settings (section, setting_value) VALUES (?, ?)');
    stmt.run(section, settingValue);
  }

  /**
   * Update multiple system settings
   * NOTE: This method was renamed from updateSettings to avoid collision with
   * the user settings method updateSettings(userId, settings) at line 533
   */
  updateSystemSettings(settings) {
    const stmt = this.db.prepare('INSERT OR REPLACE INTO settings (section, setting_value) VALUES (?, ?)');
    for (const [section, value] of Object.entries(settings)) {
      stmt.run(section, value);
    }
  }

  /**
   * Get role permissions configuration
   * Returns object with user, developer, admin arrays of enabled page IDs
   */
  getRolePermissions() {
    const stored = this.getSetting('role_permissions');
    if (stored) {
      try {
        return JSON.parse(stored);
      } catch (e) {
        console.error('Error parsing role permissions:', e);
      }
    }
    // Return default permissions if none stored
    return this.getDefaultRolePermissions();
  }

  /**
   * Set role permissions configuration
   */
  setRolePermissions(permissions) {
    this.updateSetting('role_permissions', JSON.stringify(permissions));
  }

  /**
   * Get default role permissions based on page definitions
   * This provides sensible defaults when no permissions have been configured
   */
  getDefaultRolePermissions() {
    // Default pages for each role level
    // These match the NAV_CATEGORIES page definitions with requiredRole
    return {
      user: [
        'dashboard', 'sql-query', 'semantic-search', 'document-qa',
        'code-assistance', 'git-analysis', 'audio-transcription'
      ],
      developer: [
        'dashboard', 'sql-query', 'semantic-search', 'document-qa',
        'code-assistance', 'git-analysis', 'audio-transcription',
        'document-management', 'roslyn-analysis', 'database-schemas',
        'api-orchestration', 'api-python', 'stack-overview', 'sql-rules'
      ],
      admin: [
        'dashboard', 'sql-query', 'semantic-search', 'document-qa',
        'code-assistance', 'git-analysis', 'audio-transcription',
        'document-management', 'roslyn-analysis', 'database-schemas',
        'api-orchestration', 'api-python', 'stack-overview', 'sql-rules',
        'users', 'role-management', 'system-config', 'workflows', 'jobs', 'audio-staff'
      ]
    };
  }

  /**
   * Save or update git repository
   */
  saveGitRepository(repo) {
    const { name, displayName, path, branch = 'master', projectName = null, accessToken = null } = repo;
    const repoId = repo.repoId || randomUUID();

    // Encrypt access token if provided
    const encryptedToken = accessToken ? this.encryptToken(accessToken) : null;

    const stmt = this.db.prepare(`
      INSERT INTO git_repositories (repo_id, name, display_name, path, branch, project_name, access_token)
      VALUES (?, ?, ?, ?, ?, ?, ?)
      ON CONFLICT(name) DO UPDATE SET
        display_name = excluded.display_name,
        path = excluded.path,
        branch = excluded.branch,
        project_name = excluded.project_name,
        access_token = excluded.access_token,
        updated_at = CURRENT_TIMESTAMP
    `);

    stmt.run(repoId, name, displayName, path, branch, projectName, encryptedToken);
    return this.getGitRepository(name);
  }

  /**
   * Get git repository by name
   */
  getGitRepository(name) {
    const stmt = this.db.prepare(`
      SELECT * FROM git_repositories WHERE name = ?
    `);
    return stmt.get(name);
  }

  /**
   * Get all git repositories
   */
  getAllGitRepositories() {
    const stmt = this.db.prepare(`
      SELECT * FROM git_repositories ORDER BY name ASC
    `);
    return stmt.all();
  }

  /**
   * Update git repository pull information
   */
  updateGitRepositoryPullTime(name, pullInfo = {}) {
    const { lastCommitHash, lastCommitMessage, lastCommitAuthor, branch } = pullInfo;

    const updates = ['last_pull_time = CURRENT_TIMESTAMP', 'updated_at = CURRENT_TIMESTAMP'];
    const values = [];

    if (lastCommitHash) {
      updates.push('last_commit_hash = ?');
      values.push(lastCommitHash);
    }

    if (lastCommitMessage) {
      updates.push('last_commit_message = ?');
      values.push(lastCommitMessage);
    }

    if (lastCommitAuthor) {
      updates.push('last_commit_author = ?');
      values.push(lastCommitAuthor);
    }

    if (branch) {
      updates.push('branch = ?');
      values.push(branch);
    }

    values.push(name);

    const stmt = this.db.prepare(`
      UPDATE git_repositories
      SET ${updates.join(', ')}
      WHERE name = ?
    `);

    stmt.run(...values);
    return this.getGitRepository(name);
  }

  /**
   * Delete git repository
   */
  deleteGitRepository(name) {
    const stmt = this.db.prepare(`
      DELETE FROM git_repositories WHERE name = ?
    `);
    stmt.run(name);
  }

  /**
   * Encrypt access token using simple XOR cipher with key from environment
   * Note: For production use, consider using a more robust encryption library like crypto-js
   */
  encryptToken(token) {
    if (!token) return null;

    // Use a key from environment or a default (should be overridden in production)
    const key = process.env.TOKEN_ENCRYPTION_KEY || 'CHANGE_THIS_KEY_IN_PRODUCTION_123456';

    let encrypted = '';
    for (let i = 0; i < token.length; i++) {
      const charCode = token.charCodeAt(i) ^ key.charCodeAt(i % key.length);
      encrypted += String.fromCharCode(charCode);
    }

    // Convert to base64 for safe storage
    return Buffer.from(encrypted, 'binary').toString('base64');
  }

  /**
   * Decrypt access token
   */
  decryptToken(encryptedToken) {
    if (!encryptedToken) return null;

    try {
      const key = process.env.TOKEN_ENCRYPTION_KEY || 'CHANGE_THIS_KEY_IN_PRODUCTION_123456';

      // Convert from base64
      const encrypted = Buffer.from(encryptedToken, 'base64').toString('binary');

      let decrypted = '';
      for (let i = 0; i < encrypted.length; i++) {
        const charCode = encrypted.charCodeAt(i) ^ key.charCodeAt(i % key.length);
        decrypted += String.fromCharCode(charCode);
      }

      return decrypted;
    } catch (error) {
      console.error('Error decrypting token:', error);
      return null;
    }
  }

  /**
   * Get repository access token (decrypted)
   */
  getRepositoryAccessToken(name) {
    const repo = this.getGitRepository(name);
    if (!repo || !repo.access_token) {
      return null;
    }
    return this.decryptToken(repo.access_token);
  }

  /**
   * Update repository access token
   */
  updateRepositoryAccessToken(name, accessToken) {
    const encryptedToken = accessToken ? this.encryptToken(accessToken) : null;

    const stmt = this.db.prepare(`
      UPDATE git_repositories
      SET access_token = ?, updated_at = CURRENT_TIMESTAMP
      WHERE name = ?
    `);

    stmt.run(encryptedToken, name);
    return this.getGitRepository(name);
  }

  /**
   * Update repository project name
   */
  updateRepositoryProjectName(name, projectName) {
    const stmt = this.db.prepare(`
      UPDATE git_repositories
      SET project_name = ?, updated_at = CURRENT_TIMESTAMP
      WHERE name = ?
    `);

    stmt.run(projectName, name);
    return this.getGitRepository(name);
  }

  /**
   * Get repository by project name
   */
  getRepositoryByProjectName(projectName) {
    const stmt = this.db.prepare(`
      SELECT * FROM git_repositories WHERE project_name = ?
    `);
    return stmt.get(projectName);
  }

  /**
   * Update repository analysis date
   */
  updateRepositoryAnalysisDate(name) {
    const stmt = this.db.prepare(`
      UPDATE git_repositories
      SET last_analysis_date = CURRENT_TIMESTAMP, updated_at = CURRENT_TIMESTAMP
      WHERE name = ?
    `);
    stmt.run(name);
    return this.getGitRepository(name);
  }

  /**
   * Get repositories needing analysis
   */
  getRepositoriesNeedingAnalysis() {
    const stmt = this.db.prepare(`
      SELECT * FROM git_repositories
      WHERE last_analysis_date IS NULL
      OR last_analysis_date < last_pull_time
      ORDER BY name ASC
    `);
    return stmt.all();
  }

  /**
   * Update repository sync configuration
   * @param {string} name - Repository name
   * @param {Object} syncConfig - Sync configuration
   * @param {number} syncConfig.syncInterval - Polling interval in milliseconds
   * @param {boolean} syncConfig.autoSync - Whether to auto-sync this repo
   */
  updateRepositorySyncConfig(name, syncConfig) {
    const { syncInterval, autoSync } = syncConfig;
    const updates = ['updated_at = CURRENT_TIMESTAMP'];
    const values = [];

    if (syncInterval !== undefined) {
      updates.push('sync_interval = ?');
      values.push(syncInterval);
    }

    if (autoSync !== undefined) {
      updates.push('auto_sync = ?');
      values.push(autoSync ? 1 : 0);
    }

    values.push(name);

    const stmt = this.db.prepare(`
      UPDATE git_repositories
      SET ${updates.join(', ')}
      WHERE name = ?
    `);

    stmt.run(...values);
    return this.getGitRepository(name);
  }

  /**
   * Get repositories with auto-sync enabled
   */
  getAutoSyncRepositories() {
    const stmt = this.db.prepare(`
      SELECT * FROM git_repositories
      WHERE auto_sync = 1
      ORDER BY name ASC
    `);
    return stmt.all();
  }

  /**
   * Get audio bulk folder path from settings
   * Returns the stored path or inserts and returns the default if not exists
   */
  getAudioBulkFolderPath() {
    const setting = this.getSetting('AudioBulkFolderPath');
    if (!setting) {
      // Insert default from environment variable
      const defaultPath = process.env.AUDIO_BULK_FOLDER_PATH;
      if (defaultPath) {
        this.updateSetting('AudioBulkFolderPath', defaultPath);
        return defaultPath;
      }
      return null;
    }
    return setting;
  }

  /**
   * Set audio bulk folder path in settings
   */
  setAudioBulkFolderPath(path) {
    this.updateSetting('AudioBulkFolderPath', path);
    return path;
  }

  // ========================================================================
  // NEW PERMISSION SYSTEM METHODS
  // ========================================================================

  /**
   * Get all role->category mappings
   * Returns array of {role, category} objects
   */
  getRoleCategoriesAll() {
    const stmt = this.db.prepare(`
      SELECT role, category
      FROM role_permissions
      ORDER BY role, category
    `);
    return stmt.all();
  }

  /**
   * Get categories for a specific role
   * @param {string} role - Role name ('user', 'developer', 'admin')
   * @returns {string[]} Array of category names
   */
  getRoleCategories(role) {
    const stmt = this.db.prepare(`
      SELECT category
      FROM role_permissions
      WHERE role = ?
      ORDER BY category
    `);
    const rows = stmt.all(role);
    return rows.map(row => row.category);
  }

  /**
   * Replace all categories for a role
   * @param {string} role - Role name ('user', 'developer', 'admin')
   * @param {string[]} categories - Array of category names
   */
  setRoleCategories(role, categories) {
    // Use transaction for atomicity
    const transaction = this.db.transaction(() => {
      // First, get existing categories to determine what's being removed
      const existingCategories = this.getRoleCategories(role);
      const removedCategories = existingCategories.filter(cat => !categories.includes(cat));

      // Delete existing role categories
      const deleteStmt = this.db.prepare(`DELETE FROM role_permissions WHERE role = ?`);
      deleteStmt.run(role);

      // Insert new categories
      const insertStmt = this.db.prepare(`
        INSERT INTO role_permissions (role, category)
        VALUES (?, ?)
      `);
      for (const category of categories) {
        insertStmt.run(role, category);
      }

      // Delete user permissions for removed categories
      if (removedCategories.length > 0) {
        for (const category of removedCategories) {
          this.deleteUserPermissionsByRoleCategory(role, category);
        }
      }
    });

    transaction();
  }

  /**
   * Add a single category to a role
   * @param {string} role - Role name ('user', 'developer', 'admin')
   * @param {string} category - Category name
   */
  addRoleCategory(role, category) {
    const stmt = this.db.prepare(`
      INSERT OR IGNORE INTO role_permissions (role, category)
      VALUES (?, ?)
    `);
    stmt.run(role, category);
  }

  /**
   * Remove a category from a role
   * Also deletes all user permissions for users with this role in this category
   * @param {string} role - Role name ('user', 'developer', 'admin')
   * @param {string} category - Category name
   */
  removeRoleCategory(role, category) {
    const transaction = this.db.transaction(() => {
      // Delete the role category
      const deleteRoleStmt = this.db.prepare(`
        DELETE FROM role_permissions
        WHERE role = ? AND category = ?
      `);
      deleteRoleStmt.run(role, category);

      // Delete user permissions for this role/category combination
      this.deleteUserPermissionsByRoleCategory(role, category);
    });

    transaction();
  }

  /**
   * Get all page permissions for a user
   * @param {string} userId - User ID
   * @returns {Object[]} Array of {category, pageId} objects
   */
  getUserPermissions(userId) {
    const stmt = this.db.prepare(`
      SELECT category, page_id
      FROM user_permissions
      WHERE user_id = ?
      ORDER BY category, page_id
    `);
    return stmt.all(userId);
  }

  /**
   * Get pages enabled for user in a specific category
   * @param {string} userId - User ID
   * @param {string} category - Category name
   * @returns {string[]} Array of page IDs
   */
  getUserCategoryPermissions(userId, category) {
    const stmt = this.db.prepare(`
      SELECT page_id
      FROM user_permissions
      WHERE user_id = ? AND category = ?
      ORDER BY page_id
    `);
    const rows = stmt.all(userId, category);
    return rows.map(row => row.page_id);
  }

  /**
   * Enable or disable a page for a user
   * @param {string} userId - User ID
   * @param {string} category - Category name
   * @param {string} pageId - Page ID
   * @param {boolean} enabled - True to enable, false to disable
   */
  setUserPagePermission(userId, category, pageId, enabled) {
    if (enabled) {
      // Insert or ignore if already exists
      const stmt = this.db.prepare(`
        INSERT OR IGNORE INTO user_permissions (user_id, category, page_id)
        VALUES (?, ?, ?)
      `);
      stmt.run(userId, category, pageId);
    } else {
      // Delete the permission
      const stmt = this.db.prepare(`
        DELETE FROM user_permissions
        WHERE user_id = ? AND category = ? AND page_id = ?
      `);
      stmt.run(userId, category, pageId);
    }
  }

  /**
   * Remove all permissions for a user in a specific category
   * @param {string} userId - User ID
   * @param {string} category - Category name
   */
  clearUserCategoryPermissions(userId, category) {
    const stmt = this.db.prepare(`
      DELETE FROM user_permissions
      WHERE user_id = ? AND category = ?
    `);
    stmt.run(userId, category);
  }

  /**
   * Delete user permissions for all users with a specific role in a specific category
   * Called when a category is removed from a role
   * @param {string} role - Role name ('user', 'developer', 'admin')
   * @param {string} category - Category name
   */
  deleteUserPermissionsByRoleCategory(role, category) {
    const stmt = this.db.prepare(`
      DELETE FROM user_permissions
      WHERE category = ?
      AND user_id IN (
        SELECT user_id FROM users WHERE role = ?
      )
    `);
    stmt.run(category, role);
  }

  /**
   * Close database connection
   */
  close() {
    if (this.db) {
      this.db.close();
    }
  }
}

export default EWRAIDatabase;
