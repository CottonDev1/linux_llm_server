"""
System settings and permissions API routes.

Provides endpoints for:
- System settings management
- Role permission management
- User permission management
- SQL connection settings management
"""
import logging
import json
from typing import Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models import (
    SystemSettingUpdate, SystemSettingResponse, SystemSettingsResponse,
    RolePermissionsResponse, RolePermissionsUpdate, AllRolePermissionsResponse,
    UserPermissionUpdate, UserPermissionsResponse,
    SuccessResponse
)
from services.sqlite_service import get_sqlite_service, SQLiteService
from routes.auth_routes import get_current_user, require_admin


# ============================================================================
# SQL Connection Settings Models
# ============================================================================

class SQLConnectionSettings(BaseModel):
    """SQL Server connection settings for the SQL Query page."""
    server: str = Field(..., description="SQL Server hostname or IP address")
    database: str = Field(..., description="Database name")
    auth_type: str = Field(
        default="sql",
        pattern="^(sql|windows)$",
        description="Authentication type: 'sql' or 'windows'"
    )
    username: Optional[str] = Field(None, description="Username for SQL authentication")
    password: Optional[str] = Field(None, description="Password for SQL authentication (stored encrypted)")

    class Config:
        json_schema_extra = {
            "example": {
                "server": "NCSQLTEST",
                "database": "EWRCentral",
                "auth_type": "sql",
                "username": "EWRUser",
                "password": "mypassword"
            }
        }


class SQLConnectionSettingsResponse(BaseModel):
    """Response model for SQL connection settings."""
    server: str
    database: str
    auth_type: str
    username: Optional[str] = None
    has_password: bool = Field(description="Indicates if a password is stored (password itself not returned)")

    class Config:
        json_schema_extra = {
            "example": {
                "server": "NCSQLTEST",
                "database": "EWRCentral",
                "auth_type": "sql",
                "username": "EWRUser",
                "has_password": True
            }
        }

logger = logging.getLogger(__name__)

# Create routers
settings_router = APIRouter(prefix="/api/settings", tags=["System Settings"])
permissions_router = APIRouter(prefix="/api/permissions", tags=["Permissions"])


def get_db() -> SQLiteService:
    """Dependency to get database service."""
    return get_sqlite_service()


# ============================================================================
# System Settings Routes
# ============================================================================

@settings_router.get("", response_model=SystemSettingsResponse)
async def get_all_settings(
    admin: dict = Depends(require_admin),
    db: SQLiteService = Depends(get_db)
):
    """
    Get all system settings (admin only).
    """
    settings = db.get_all_system_settings()
    return SystemSettingsResponse(settings=settings)


@settings_router.get("/{section}", response_model=SystemSettingResponse)
async def get_setting(
    section: str,
    admin: dict = Depends(require_admin),
    db: SQLiteService = Depends(get_db)
):
    """
    Get a specific system setting (admin only).
    """
    value = db.get_system_setting(section)
    if value is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Setting '{section}' not found"
        )
    return SystemSettingResponse(section=section, value=value)


@settings_router.put("/{section}", response_model=SystemSettingResponse)
async def update_setting(
    section: str,
    update: SystemSettingUpdate,
    admin: dict = Depends(require_admin),
    db: SQLiteService = Depends(get_db)
):
    """
    Update a system setting (admin only).
    """
    db.set_system_setting(section, update.value)
    return SystemSettingResponse(section=section, value=update.value)


@settings_router.put("", response_model=SuccessResponse)
async def update_multiple_settings(
    settings: Dict[str, str],
    admin: dict = Depends(require_admin),
    db: SQLiteService = Depends(get_db)
):
    """
    Update multiple system settings at once (admin only).
    """
    db.update_system_settings(settings)
    return SuccessResponse(message=f"Updated {len(settings)} settings")


# ============================================================================
# Role Permissions Routes
# ============================================================================

@permissions_router.get("/roles", response_model=AllRolePermissionsResponse)
async def get_all_role_permissions(
    admin: dict = Depends(require_admin),
    db: SQLiteService = Depends(get_db)
):
    """
    Get all role permissions (admin only).
    
    Returns categories assigned to each role.
    """
    permissions = db.get_all_role_permissions()
    return AllRolePermissionsResponse(**permissions)


@permissions_router.get("/roles/{role}", response_model=RolePermissionsResponse)
async def get_role_permissions(
    role: str,
    admin: dict = Depends(require_admin),
    db: SQLiteService = Depends(get_db)
):
    """
    Get permissions for a specific role (admin only).
    """
    if role not in ["user", "developer", "admin"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid role. Must be 'user', 'developer', or 'admin'"
        )
    
    categories = db.get_role_categories(role)
    return RolePermissionsResponse(role=role, categories=categories)


@permissions_router.put("/roles/{role}", response_model=RolePermissionsResponse)
async def update_role_permissions(
    role: str,
    update: RolePermissionsUpdate,
    admin: dict = Depends(require_admin),
    db: SQLiteService = Depends(get_db)
):
    """
    Update permissions for a role (admin only).
    
    Replaces all categories for the role.
    """
    if role not in ["user", "developer", "admin"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid role. Must be 'user', 'developer', or 'admin'"
        )
    
    db.set_role_categories(role, update.categories)
    return RolePermissionsResponse(role=role, categories=update.categories)


@permissions_router.post("/roles/{role}/categories/{category}", response_model=SuccessResponse)
async def add_role_category(
    role: str,
    category: str,
    admin: dict = Depends(require_admin),
    db: SQLiteService = Depends(get_db)
):
    """
    Add a category to a role (admin only).
    """
    if role not in ["user", "developer", "admin"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid role"
        )
    
    db.add_role_category(role, category)
    return SuccessResponse(message=f"Added '{category}' to '{role}' role")


@permissions_router.delete("/roles/{role}/categories/{category}", response_model=SuccessResponse)
async def remove_role_category(
    role: str,
    category: str,
    admin: dict = Depends(require_admin),
    db: SQLiteService = Depends(get_db)
):
    """
    Remove a category from a role (admin only).
    
    Also removes user permissions for users with this role in this category.
    """
    if role not in ["user", "developer", "admin"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid role"
        )
    
    db.remove_role_category(role, category)
    return SuccessResponse(message=f"Removed '{category}' from '{role}' role")


# ============================================================================
# User Permissions Routes
# ============================================================================

@permissions_router.get("/users/{user_id}", response_model=UserPermissionsResponse)
async def get_user_permissions(
    user_id: str,
    current_user: dict = Depends(get_current_user),
    db: SQLiteService = Depends(get_db)
):
    """
    Get permissions for a specific user.
    
    Users can view their own permissions; admins can view any user's permissions.
    """
    if current_user["user_id"] != user_id and current_user["role"] != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied"
        )
    
    permissions = db.get_user_permissions(user_id)
    return UserPermissionsResponse(user_id=user_id, permissions=permissions)


@permissions_router.get("/users/{user_id}/categories/{category}")
async def get_user_category_permissions(
    user_id: str,
    category: str,
    current_user: dict = Depends(get_current_user),
    db: SQLiteService = Depends(get_db)
):
    """
    Get pages enabled for a user in a specific category.
    """
    if current_user["user_id"] != user_id and current_user["role"] != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied"
        )
    
    pages = db.get_user_category_permissions(user_id, category)
    return {"user_id": user_id, "category": category, "pages": pages}


@permissions_router.put("/users/{user_id}", response_model=SuccessResponse)
async def update_user_permission(
    user_id: str,
    update: UserPermissionUpdate,
    admin: dict = Depends(require_admin),
    db: SQLiteService = Depends(get_db)
):
    """
    Enable or disable a page for a user (admin only).
    """
    db.set_user_page_permission(
        user_id,
        update.category,
        update.page_id,
        update.enabled
    )
    
    action = "enabled" if update.enabled else "disabled"
    return SuccessResponse(message=f"Page '{update.page_id}' {action} for user")


@permissions_router.delete("/users/{user_id}/categories/{category}", response_model=SuccessResponse)
async def clear_user_category_permissions(
    user_id: str,
    category: str,
    admin: dict = Depends(require_admin),
    db: SQLiteService = Depends(get_db)
):
    """
    Clear all permissions for a user in a category (admin only).
    """
    db.clear_user_category_permissions(user_id, category)
    return SuccessResponse(message=f"Cleared '{category}' permissions for user")


# ============================================================================
# SQL Connection Settings Routes
# ============================================================================

# Constants for SQL connection settings storage
SQL_CONNECTION_SETTING_PREFIX = "SQLConnection:"
SQL_CONNECTION_GLOBAL_KEY = "SQLConnection:global"


def _encrypt_password(password: str) -> str:
    """
    Encrypt password for storage using XOR cipher with base64 encoding.
    This matches the pattern used in SQLiteService._encrypt_token().
    """
    import os
    import base64

    if not password:
        return ""

    key = os.getenv("TOKEN_ENCRYPTION_KEY", "CHANGE_THIS_KEY_IN_PRODUCTION_123456")
    encrypted = ''.join(
        chr(ord(c) ^ ord(key[i % len(key)]))
        for i, c in enumerate(password)
    )
    return base64.b64encode(encrypted.encode('latin-1')).decode('utf-8')


def _decrypt_password(encrypted: str) -> str:
    """
    Decrypt password from storage.
    """
    import os
    import base64

    if not encrypted:
        return ""

    try:
        key = os.getenv("TOKEN_ENCRYPTION_KEY", "CHANGE_THIS_KEY_IN_PRODUCTION_123456")
        decoded = base64.b64decode(encrypted.encode('utf-8')).decode('latin-1')
        return ''.join(
            chr(ord(c) ^ ord(key[i % len(key)]))
            for i, c in enumerate(decoded)
        )
    except Exception as e:
        logger.error(f"Error decrypting password: {e}")
        return ""


@settings_router.get("/sql-connection", response_model=SQLConnectionSettingsResponse)
async def get_sql_connection_settings(
    current_user: dict = Depends(get_current_user),
    db: SQLiteService = Depends(get_db)
):
    """
    Get SQL connection default settings.

    First checks for user-specific settings, then falls back to global settings.
    Password is never returned - only a flag indicating if one is stored.

    Returns 404 if no settings are found.
    """
    user_id = current_user["user_id"]

    # Try user-specific settings first
    user_key = f"{SQL_CONNECTION_SETTING_PREFIX}{user_id}"
    settings_json = db.get_system_setting(user_key)

    # Fall back to global settings
    if not settings_json:
        settings_json = db.get_system_setting(SQL_CONNECTION_GLOBAL_KEY)

    if not settings_json:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No SQL connection settings found"
        )

    try:
        settings = json.loads(settings_json)
        return SQLConnectionSettingsResponse(
            server=settings.get("server", ""),
            database=settings.get("database", ""),
            auth_type=settings.get("auth_type", "sql"),
            username=settings.get("username"),
            has_password=bool(settings.get("password_encrypted"))
        )
    except json.JSONDecodeError as e:
        logger.error(f"Error parsing SQL connection settings: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error parsing stored settings"
        )


@settings_router.post("/sql-connection", response_model=SuccessResponse)
async def save_sql_connection_settings(
    settings: SQLConnectionSettings,
    global_setting: bool = False,
    current_user: dict = Depends(get_current_user),
    db: SQLiteService = Depends(get_db)
):
    """
    Save SQL connection default settings.

    By default, saves per-user settings. Set global_setting=true to save
    as global defaults (requires admin role).

    Password is encrypted before storage for security.
    """
    user_id = current_user["user_id"]

    # Global settings require admin role
    if global_setting and current_user["role"] != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only admins can set global SQL connection settings"
        )

    # Determine storage key
    storage_key = SQL_CONNECTION_GLOBAL_KEY if global_setting else f"{SQL_CONNECTION_SETTING_PREFIX}{user_id}"

    # Build settings dict with encrypted password
    settings_dict = {
        "server": settings.server,
        "database": settings.database,
        "auth_type": settings.auth_type,
        "username": settings.username,
        "password_encrypted": _encrypt_password(settings.password) if settings.password else ""
    }

    # Store as JSON
    db.set_system_setting(storage_key, json.dumps(settings_dict))

    scope = "global" if global_setting else "user"
    logger.info(f"SQL connection settings saved for {scope} (user: {user_id})")

    return SuccessResponse(message=f"SQL connection settings saved as {scope} default")


@settings_router.delete("/sql-connection", response_model=SuccessResponse)
async def delete_sql_connection_settings(
    global_setting: bool = False,
    current_user: dict = Depends(get_current_user),
    db: SQLiteService = Depends(get_db)
):
    """
    Delete SQL connection settings.

    By default, deletes user-specific settings. Set global_setting=true to delete
    global settings (requires admin role).
    """
    user_id = current_user["user_id"]

    # Global settings require admin role
    if global_setting and current_user["role"] != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only admins can delete global SQL connection settings"
        )

    # Determine storage key
    storage_key = SQL_CONNECTION_GLOBAL_KEY if global_setting else f"{SQL_CONNECTION_SETTING_PREFIX}{user_id}"

    # Delete by setting to empty string (SQLite doesn't have true delete for settings)
    db.set_system_setting(storage_key, "")

    scope = "global" if global_setting else "user"
    logger.info(f"SQL connection settings deleted for {scope} (user: {user_id})")

    return SuccessResponse(message=f"SQL connection settings deleted")


@settings_router.get("/sql-connection/with-credentials")
async def get_sql_connection_with_credentials(
    current_user: dict = Depends(get_current_user),
    db: SQLiteService = Depends(get_db)
):
    """
    Get SQL connection settings including the decrypted password.

    This endpoint is for internal use by the SQL query system to build
    connection strings. The frontend should use the regular endpoint
    that doesn't expose passwords.

    Returns the full settings including decrypted password.
    """
    user_id = current_user["user_id"]

    # Try user-specific settings first
    user_key = f"{SQL_CONNECTION_SETTING_PREFIX}{user_id}"
    settings_json = db.get_system_setting(user_key)

    # Fall back to global settings
    if not settings_json:
        settings_json = db.get_system_setting(SQL_CONNECTION_GLOBAL_KEY)

    if not settings_json:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No SQL connection settings found"
        )

    try:
        settings = json.loads(settings_json)
        return {
            "server": settings.get("server", ""),
            "database": settings.get("database", ""),
            "auth_type": settings.get("auth_type", "sql"),
            "username": settings.get("username"),
            "password": _decrypt_password(settings.get("password_encrypted", ""))
        }
    except json.JSONDecodeError as e:
        logger.error(f"Error parsing SQL connection settings: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error parsing stored settings"
        )
