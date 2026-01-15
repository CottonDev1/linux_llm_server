"""
Pydantic models for API requests and responses.

This module defines all the request/response schemas used by the auth,
user management, and settings routes.
"""

from datetime import datetime
from typing import Optional, List, Any
from pydantic import BaseModel, Field, EmailStr, ConfigDict


# =============================================================================
# User Models
# =============================================================================

class UserBase(BaseModel):
    """Base user fields."""
    username: str = Field(..., min_length=3, max_length=50)
    email: Optional[str] = None
    role: str = Field(default="user")
    is_active: bool = Field(default=True)


class UserCreate(UserBase):
    """Request model for creating a new user."""
    password: str = Field(..., min_length=6)


class UserUpdate(BaseModel):
    """Request model for updating a user."""
    username: Optional[str] = Field(None, min_length=3, max_length=50)
    email: Optional[str] = None
    role: Optional[str] = None
    is_active: Optional[bool] = None


class UserResponse(BaseModel):
    """Response model for user data."""
    model_config = ConfigDict(from_attributes=True)

    user_id: str
    username: str
    email: Optional[str] = None
    role: str
    is_active: bool
    created_at: Optional[datetime] = None
    last_login: Optional[datetime] = None


class UserWithSettingsResponse(UserResponse):
    """Response model for user with settings."""
    settings: Optional["UserSettingsResponse"] = None


class UserListResponse(BaseModel):
    """Response model for list of users."""
    users: List[UserResponse]
    total: int


# =============================================================================
# User Settings Models
# =============================================================================

class UserSettingsBase(BaseModel):
    """Base user settings fields."""
    theme: Optional[str] = "light"
    notifications_enabled: Optional[bool] = True
    default_database: Optional[str] = None
    preferred_llm_model: Optional[str] = None
    sidebar_collapsed: Optional[bool] = False


class UserSettingsUpdate(BaseModel):
    """Request model for updating user settings."""
    theme: Optional[str] = None
    notifications_enabled: Optional[bool] = None
    default_database: Optional[str] = None
    preferred_llm_model: Optional[str] = None
    sidebar_collapsed: Optional[bool] = None
    custom_settings: Optional[dict] = None


class UserSettingsResponse(UserSettingsBase):
    """Response model for user settings."""
    model_config = ConfigDict(from_attributes=True)

    settings_id: Optional[str] = None
    user_id: str
    custom_settings: Optional[dict] = None
    updated_at: Optional[datetime] = None


# =============================================================================
# Authentication Models
# =============================================================================

class LoginRequest(BaseModel):
    """Request model for user login."""
    username: str
    password: str


class LoginResponse(BaseModel):
    """Response model for successful login."""
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int  # seconds
    user: UserResponse


class TokenRefreshRequest(BaseModel):
    """Request model for token refresh."""
    refresh_token: str


class TokenRefreshResponse(BaseModel):
    """Response model for token refresh."""
    access_token: str
    token_type: str = "bearer"
    expires_in: int  # seconds


# =============================================================================
# Password Models
# =============================================================================

class PasswordChangeRequest(BaseModel):
    """Request model for changing password."""
    current_password: str
    new_password: str = Field(..., min_length=6)


class PasswordResetRequest(BaseModel):
    """Request model for resetting password (admin)."""
    new_password: str = Field(..., min_length=6)


# =============================================================================
# Session Models
# =============================================================================

class SessionResponse(BaseModel):
    """Response model for user session."""
    model_config = ConfigDict(from_attributes=True)

    session_id: str
    user_id: str
    created_at: datetime
    last_active: Optional[datetime] = None
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    is_active: bool = True


class SessionListResponse(BaseModel):
    """Response model for list of sessions."""
    sessions: List[SessionResponse]
    total: int


# =============================================================================
# Generic Response Models
# =============================================================================

class SuccessResponse(BaseModel):
    """Generic success response."""
    success: bool = True
    message: str


class ErrorResponse(BaseModel):
    """Generic error response."""
    success: bool = False
    error: str
    detail: Optional[str] = None


# =============================================================================
# Permission Models
# =============================================================================

class PermissionBase(BaseModel):
    """Base permission fields."""
    name: str
    description: Optional[str] = None
    resource: str
    action: str


class PermissionResponse(PermissionBase):
    """Response model for permission."""
    model_config = ConfigDict(from_attributes=True)

    permission_id: str
    created_at: Optional[datetime] = None


class RolePermissionsResponse(BaseModel):
    """Response model for role permissions."""
    role: str
    permissions: List[PermissionResponse]


# =============================================================================
# Git Repository Models
# =============================================================================

class GitRepositoryBase(BaseModel):
    """Base git repository fields."""
    name: str
    path: str
    description: Optional[str] = None
    is_active: bool = True


class GitRepositoryCreate(GitRepositoryBase):
    """Request model for creating a git repository."""
    pass


class GitRepositoryUpdate(BaseModel):
    """Request model for updating a git repository."""
    name: Optional[str] = None
    path: Optional[str] = None
    description: Optional[str] = None
    is_active: Optional[bool] = None


class GitRepositoryPullUpdate(BaseModel):
    """Request model for updating pull settings."""
    auto_pull: Optional[bool] = None
    pull_schedule: Optional[str] = None


class GitRepositoryResponse(GitRepositoryBase):
    """Response model for git repository."""
    model_config = ConfigDict(from_attributes=True)

    repo_id: str
    last_pull: Optional[datetime] = None
    last_commit: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


class GitRepositoryListResponse(BaseModel):
    """Response model for list of git repositories."""
    repositories: List[GitRepositoryResponse]
    total: int


# =============================================================================
# Tag Models
# =============================================================================

class TagBase(BaseModel):
    """Base tag fields."""
    name: str
    color: Optional[str] = None
    description: Optional[str] = None


class TagCreate(TagBase):
    """Request model for creating a tag."""
    pass


class TagUpdate(BaseModel):
    """Request model for updating a tag."""
    name: Optional[str] = None
    color: Optional[str] = None
    description: Optional[str] = None


class TagResponse(TagBase):
    """Response model for tag."""
    model_config = ConfigDict(from_attributes=True)

    tag_id: str
    created_at: Optional[datetime] = None


class TagListResponse(BaseModel):
    """Response model for list of tags."""
    tags: List[TagResponse]
    total: int


class DocumentTagCreate(BaseModel):
    """Request model for tagging a document."""
    document_id: str
    tag_ids: List[str]


class DocumentTagsResponse(BaseModel):
    """Response model for document tags."""
    document_id: str
    tags: List[TagResponse]


# =============================================================================
# System Settings Models
# =============================================================================

class SystemSettingUpdate(BaseModel):
    """Request model for updating a system setting."""
    value: Any
    description: Optional[str] = None


class SystemSettingResponse(BaseModel):
    """Response model for system setting."""
    model_config = ConfigDict(from_attributes=True)

    setting_key: str
    value: Any
    description: Optional[str] = None
    updated_at: Optional[datetime] = None
    updated_by: Optional[str] = None


class SystemSettingsResponse(BaseModel):
    """Response model for all system settings."""
    settings: List[SystemSettingResponse]


# =============================================================================
# Role Permissions Models
# =============================================================================

class RolePermissionsUpdate(BaseModel):
    """Request model for updating role permissions."""
    permissions: List[str]  # List of permission IDs


class AllRolePermissionsResponse(BaseModel):
    """Response model for all roles and their permissions."""
    roles: List[RolePermissionsResponse]


class UserPermissionUpdate(BaseModel):
    """Request model for updating user-specific permissions."""
    permissions: List[str]  # List of permission IDs


class UserPermissionsResponse(BaseModel):
    """Response model for user permissions."""
    user_id: str
    role: str
    role_permissions: List[PermissionResponse]
    user_permissions: List[PermissionResponse]
    effective_permissions: List[PermissionResponse]


# Update forward references
UserWithSettingsResponse.model_rebuild()
