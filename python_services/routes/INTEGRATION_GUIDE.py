"""
SQLite Routes Integration Guide

This file shows how to integrate the new SQLite-based auth/user/settings routes
into the existing main.py FastAPI application.

Add the following imports and route registrations to main.py:
"""

# ==============================================================================
# IMPORTS TO ADD (near the top of main.py with other imports)
# ==============================================================================

# Add these imports after the existing router imports:
"""
# SQLite-based auth and user management routes
from routes.auth_routes import router as auth_router, users_router
from routes.git_routes import router as sqlite_git_router
from routes.settings_routes import settings_router, permissions_router
from routes.tag_routes import router as tag_router
"""

# ==============================================================================
# INITIALIZATION TO ADD (in the lifespan function)
# ==============================================================================

# Add this after "Initialize category service" section in the lifespan function:
"""
    # Initialize SQLite user database
    print("Initializing SQLite user database...")
    try:
        from services.sqlite_service import get_sqlite_service
        sqlite_service = get_sqlite_service()
        # Database is auto-initialized on first access
        print("SQLite user database initialized")
    except Exception as e:
        print(f"SQLite initialization warning: {e}")
"""

# ==============================================================================
# ROUTE REGISTRATION TO ADD (after existing router registrations)
# ==============================================================================

# Add these after the existing app.include_router() calls:
"""
# SQLite-based authentication routes: /api/auth/login, /api/auth/logout, /api/auth/refresh, /api/auth/me
app.include_router(auth_router)

# User management routes: /api/users, /api/users/{id}, /api/users/{id}/settings
app.include_router(users_router)

# System settings routes: /api/settings, /api/settings/{section}
app.include_router(settings_router)

# Permission routes: /api/permissions/roles, /api/permissions/users/{id}
app.include_router(permissions_router)

# Tag routes: /api/tags, /api/tags/{id}, /api/tags/documents/{doc_id}
app.include_router(tag_router)

# Note: The sqlite_git_router provides duplicate git repository endpoints.
# Only include if you want to replace the JavaScript-based git repo storage with SQLite:
# app.include_router(sqlite_git_router)
"""

# ==============================================================================
# COMPLETE EXAMPLE - main.py snippet
# ==============================================================================

EXAMPLE_MAIN_PY = '''
# At the top of main.py, add these imports:
from routes.auth_routes import router as auth_router, users_router
from routes.settings_routes import settings_router, permissions_router
from routes.tag_routes import router as tag_router

# In the lifespan function, after category service initialization:
@asynccontextmanager
async def lifespan(app: FastAPI):
    # ... existing startup code ...
    
    # Initialize SQLite user database
    print("Initializing SQLite user database...")
    try:
        from services.sqlite_service import get_sqlite_service
        sqlite_service = get_sqlite_service()
        print("SQLite user database initialized")
    except Exception as e:
        print(f"SQLite initialization warning: {e}")
    
    # ... rest of startup code ...

# After the existing app.include_router() calls:
app.include_router(auth_router)
app.include_router(users_router)
app.include_router(settings_router)
app.include_router(permissions_router)
app.include_router(tag_router)
'''

# ==============================================================================
# API ENDPOINT SUMMARY
# ==============================================================================

"""
New endpoints provided by the SQLite routes:

Authentication (/api/auth):
  POST /api/auth/login          - User login, returns JWT + refresh token
  POST /api/auth/refresh        - Refresh access token
  POST /api/auth/logout         - Revoke refresh token
  GET  /api/auth/me             - Get current user info
  POST /api/auth/change-password - Change own password

Users (/api/users):
  GET    /api/users             - List all users (admin)
  POST   /api/users             - Create user (admin)
  GET    /api/users/{id}        - Get user by ID (admin)
  PATCH  /api/users/{id}        - Update user (admin)
  DELETE /api/users/{id}        - Delete user (admin)
  POST   /api/users/{id}/reset-password - Reset password (admin)
  GET    /api/users/{id}/settings  - Get user settings
  PATCH  /api/users/{id}/settings  - Update user settings
  GET    /api/users/{id}/sessions  - Get user sessions

System Settings (/api/settings):
  GET  /api/settings            - Get all settings (admin)
  GET  /api/settings/{section}  - Get setting (admin)
  PUT  /api/settings/{section}  - Update setting (admin)
  PUT  /api/settings            - Bulk update settings (admin)

Permissions (/api/permissions):
  GET  /api/permissions/roles           - Get all role permissions (admin)
  GET  /api/permissions/roles/{role}    - Get role permissions (admin)
  PUT  /api/permissions/roles/{role}    - Update role permissions (admin)
  POST /api/permissions/roles/{role}/categories/{cat} - Add category to role
  DELETE /api/permissions/roles/{role}/categories/{cat} - Remove category
  GET  /api/permissions/users/{id}      - Get user permissions
  PUT  /api/permissions/users/{id}      - Update user permission (admin)

Tags (/api/tags):
  GET    /api/tags              - List all tags
  POST   /api/tags              - Create tag
  GET    /api/tags/{id}         - Get tag
  PATCH  /api/tags/{id}         - Update tag
  DELETE /api/tags/{id}         - Delete tag (admin)
  GET    /api/tags/documents/{doc_id}        - Get document tags
  POST   /api/tags/documents/{doc_id}/{tag}  - Tag document
  DELETE /api/tags/documents/{doc_id}/{tag}  - Untag document
  GET    /api/tags/{id}/documents            - Get documents by tag

Git Repositories (/api/git-repositories) - Optional SQLite-based:
  GET    /api/git-repositories              - List repositories
  POST   /api/git-repositories              - Create repository (admin)
  GET    /api/git-repositories/auto-sync    - List auto-sync repos
  GET    /api/git-repositories/needs-analysis - List repos needing analysis
  GET    /api/git-repositories/{name}       - Get repository
  PATCH  /api/git-repositories/{name}       - Update repository (admin)
  DELETE /api/git-repositories/{name}       - Delete repository (admin)
  POST   /api/git-repositories/{name}/pull  - Record pull info
  POST   /api/git-repositories/{name}/analyzed - Mark as analyzed
"""

if __name__ == "__main__":
    print(__doc__)
    print("\nSee the constants above for integration instructions.")
