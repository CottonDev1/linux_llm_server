/**
 * Authentication Middleware
 * Handles JWT token validation
 */

import jwt from 'jsonwebtoken';

const JWT_SECRET = process.env.JWT_SECRET || 'your-secret-key-change-in-production';

// Check DISABLE_AUTH at runtime (not module load time) to allow dotenv to load first
const isAuthDisabled = () => process.env.DISABLE_AUTH === 'true';

// Default user context when auth is disabled
const DEFAULT_DEV_USER = {
  id: 'dev-user',
  username: 'developer',
  role: 'admin',
  permissions: ['all'],
  settings: {
    llm: {
      model: 'via-python-service',
      temperature: 0.7
    },
    rag: {
      maxSourceLength: 2000,
      numSources: 3
    }
  }
};

/**
 * JWT Authentication Middleware
 * Validates JWT tokens and attaches user context to request
 * Bypasses auth when DISABLE_AUTH=true (development mode)
 */
export const authenticateToken = (req, res, next) => {
  // Bypass auth in development mode
  if (isAuthDisabled()) {
    req.user = DEFAULT_DEV_USER;
    return next();
  }

  const authHeader = req.headers['authorization'];
  const token = authHeader && authHeader.split(' ')[1]; // Bearer TOKEN

  if (!token) {
    return res.status(401).json({ error: 'Authentication required' });
  }

  try {
    const user = jwt.verify(token, JWT_SECRET);
    req.user = user;
    next();
  } catch (err) {
    if (err.name === 'TokenExpiredError') {
      return res.status(401).json({ error: 'Token expired', code: 'TOKEN_EXPIRED' });
    }
    return res.status(403).json({ error: 'Invalid token' });
  }
};

/**
 * Optional Authentication Middleware
 * Validates token if present, but allows request to continue if not
 */
export const optionalAuth = (req, res, next) => {
  const authHeader = req.headers['authorization'];
  const token = authHeader && authHeader.split(' ')[1];

  if (token) {
    try {
      const user = jwt.verify(token, JWT_SECRET);
      req.user = user;
    } catch (err) {
      // Invalid token, but we continue without user context
      console.warn('Invalid token provided:', err.message);
    }
  }

  next();
};

/**
 * Require Authentication (Admin)
 * Requires valid JWT token
 * Bypasses auth when DISABLE_AUTH=true (development mode)
 */
export const requireAuthOrAdmin = (req, res, next) => {
  // Bypass auth in development mode
  if (isAuthDisabled()) {
    req.user = DEFAULT_DEV_USER;
    return next();
  }

  // Use JWT authentication
  return authenticateToken(req, res, next);
};

/**
 * Role-based Access Control
 * Requires user to have a specific role
 */
export const requireRole = (requiredRole) => {
  return (req, res, next) => {
    if (!req.user) {
      return res.status(401).json({ error: 'Authentication required' });
    }

    const roleHierarchy = { 'user': 1, 'developer': 2, 'admin': 3 };
    const userLevel = roleHierarchy[req.user.role] || 0;
    const requiredLevel = roleHierarchy[requiredRole] || 999;

    if (userLevel < requiredLevel) {
      return res.status(403).json({ error: 'Insufficient permissions' });
    }

    next();
  };
};

/**
 * Permission-based Access Control
 * Requires user to have a specific permission
 */
export const requirePermission = (permission) => {
  return (req, res, next) => {
    if (!req.user) {
      return res.status(401).json({ error: 'Authentication required' });
    }

    // Admin has all permissions
    if (req.user.role === 'admin' || (req.user.permissions && req.user.permissions.includes('all'))) {
      return next();
    }

    if (!req.user.permissions || !req.user.permissions.includes(permission)) {
      return res.status(403).json({ error: `Missing permission: ${permission}` });
    }

    next();
  };
};

/**
 * Generate JWT Access Token
 */
export const generateAccessToken = (user) => {
  return jwt.sign(
    {
      id: user.id,
      username: user.username,
      email: user.email,
      role: user.role,
      permissions: user.permissions,
      settings: user.settings
    },
    JWT_SECRET,
    { expiresIn: '1h' }
  );
};

/**
 * Generate JWT Refresh Token
 */
export const generateRefreshToken = (user) => {
  return jwt.sign(
    {
      id: user.id,
      type: 'refresh'
    },
    JWT_SECRET,
    { expiresIn: '7d' }
  );
};

export default {
  authenticateToken,
  optionalAuth,
  requireAuthOrAdmin,
  requireRole,
  requirePermission,
  generateAccessToken,
  generateRefreshToken,
  JWT_SECRET
};
