"""
Authentication and user management API routes.

Provides endpoints for:
- User authentication (login, logout, token refresh)
- User CRUD operations
- User settings management
- Password management
- Session management
"""
from datetime import datetime, timedelta
from typing import Optional
import logging
import jwt
import os

from fastapi import APIRouter, Depends, HTTPException, status, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

# Import models and schemas
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models import (
    UserCreate, UserUpdate, UserResponse, UserWithSettingsResponse, UserListResponse,
    UserSettingsUpdate, UserSettingsResponse,
    LoginRequest, LoginResponse, TokenRefreshRequest, TokenRefreshResponse,
    PasswordChangeRequest, PasswordResetRequest,
    SessionResponse, SessionListResponse,
    SuccessResponse, ErrorResponse
)
from services.sqlite_service import get_sqlite_service, SQLiteService

logger = logging.getLogger(__name__)

# JWT Configuration
JWT_SECRET = os.getenv("JWT_SECRET", "your-secret-key-change-in-production")
JWT_ALGORITHM = "HS256"
JWT_EXPIRATION_HOURS = 24

# Security
security = HTTPBearer(auto_error=False)

# Create routers
router = APIRouter(prefix="/api/auth", tags=["Authentication"])
users_router = APIRouter(prefix="/api/users", tags=["Users"])


# ============================================================================
# Dependencies
# ============================================================================

def get_db() -> SQLiteService:
    """Dependency to get database service."""
    return get_sqlite_service()


def create_access_token(user_id: str, username: str, role: str) -> str:
    """Create a JWT access token."""
    payload = {
        "sub": user_id,
        "username": username,
        "role": role,
        "exp": datetime.utcnow() + timedelta(hours=JWT_EXPIRATION_HOURS),
        "iat": datetime.utcnow()
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def decode_token(token: str) -> Optional[dict]:
    """Decode and verify a JWT token."""
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        return None
    except jwt.InvalidTokenError:
        return None


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: SQLiteService = Depends(get_db)
) -> dict:
    """
    Dependency to get current authenticated user.
    
    Raises HTTPException if not authenticated.
    """
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"}
        )
    
    payload = decode_token(credentials.credentials)
    if not payload:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"}
        )
    
    user = db.get_user_by_id(payload["sub"])
    if not user or not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found or inactive"
        )
    
    return {
        "user_id": user.user_id,
        "username": user.username,
        "role": user.role,
        "user": user
    }


async def get_current_user_optional(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: SQLiteService = Depends(get_db)
) -> Optional[dict]:
    """
    Dependency to get current user if authenticated, None otherwise.
    """
    if not credentials:
        return None
    
    payload = decode_token(credentials.credentials)
    if not payload:
        return None
    
    user = db.get_user_by_id(payload["sub"])
    if not user or not user.is_active:
        return None
    
    return {
        "user_id": user.user_id,
        "username": user.username,
        "role": user.role,
        "user": user
    }


async def require_admin(current_user: dict = Depends(get_current_user)) -> dict:
    """Dependency to require admin role."""
    if current_user["role"] != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required"
        )
    return current_user


# ============================================================================
# Authentication Routes
# ============================================================================

@router.post("/login", response_model=LoginResponse)
async def login(
    request: LoginRequest,
    db: SQLiteService = Depends(get_db)
):
    """
    Authenticate user and return tokens.
    
    Returns access token (JWT) and refresh token.
    """
    user = db.authenticate_user(request.username, request.password)
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password"
        )
    
    # Create tokens
    access_token = create_access_token(user.user_id, user.username, user.role)
    refresh_token_obj = db.create_refresh_token(user.user_id)
    
    return LoginResponse(
        access_token=access_token,
        refresh_token=refresh_token_obj._plain_token,
        expires_in=JWT_EXPIRATION_HOURS * 3600,
        user=UserResponse.model_validate(user)
    )


@router.post("/refresh", response_model=TokenRefreshResponse)
async def refresh_token(
    request: TokenRefreshRequest,
    db: SQLiteService = Depends(get_db)
):
    """
    Refresh access token using refresh token.
    """
    user = db.verify_refresh_token(request.refresh_token)
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired refresh token"
        )
    
    access_token = create_access_token(user.user_id, user.username, user.role)
    
    return TokenRefreshResponse(
        access_token=access_token,
        expires_in=JWT_EXPIRATION_HOURS * 3600
    )


@router.post("/logout", response_model=SuccessResponse)
async def logout(
    request: TokenRefreshRequest,
    current_user: dict = Depends(get_current_user),
    db: SQLiteService = Depends(get_db)
):
    """
    Logout user by revoking refresh token.
    """
    db.revoke_refresh_token(request.refresh_token)
    return SuccessResponse(message="Logged out successfully")


@router.get("/me", response_model=UserWithSettingsResponse)
async def get_current_user_info(
    current_user: dict = Depends(get_current_user),
    db: SQLiteService = Depends(get_db)
):
    """
    Get current authenticated user's information.
    """
    user = current_user["user"]
    settings = db.get_user_settings(user.user_id)
    
    response = UserWithSettingsResponse.model_validate(user)
    if settings:
        response.settings = UserSettingsResponse.model_validate(settings)
    
    return response


@router.post("/change-password", response_model=SuccessResponse)
async def change_password(
    request: PasswordChangeRequest,
    current_user: dict = Depends(get_current_user),
    db: SQLiteService = Depends(get_db)
):
    """
    Change current user's password.
    
    Requires current password verification.
    """
    success = db.change_user_password(
        current_user["user_id"],
        request.current_password,
        request.new_password
    )
    
    if not success:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Current password is incorrect"
        )
    
    return SuccessResponse(message="Password changed successfully")


# ============================================================================
# User Management Routes (Admin)
# ============================================================================

@users_router.get("", response_model=UserListResponse)
async def list_users(
    admin: dict = Depends(require_admin),
    db: SQLiteService = Depends(get_db)
):
    """
    List all users (admin only).
    """
    users = db.get_all_users()
    return UserListResponse(
        users=[UserResponse.model_validate(u) for u in users],
        total=len(users)
    )


@users_router.post("", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def create_user(
    user_data: UserCreate,
    admin: dict = Depends(require_admin),
    db: SQLiteService = Depends(get_db)
):
    """
    Create a new user (admin only).
    """
    try:
        user = db.create_user(user_data)
        return UserResponse.model_validate(user)
    except Exception as e:
        if "UNIQUE constraint" in str(e):
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Username already exists"
            )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@users_router.get("/{user_id}", response_model=UserWithSettingsResponse)
async def get_user(
    user_id: str,
    admin: dict = Depends(require_admin),
    db: SQLiteService = Depends(get_db)
):
    """
    Get user by ID (admin only).
    """
    user = db.get_user_by_id(user_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    settings = db.get_user_settings(user_id)
    response = UserWithSettingsResponse.model_validate(user)
    if settings:
        response.settings = UserSettingsResponse.model_validate(settings)
    
    return response


@users_router.patch("/{user_id}", response_model=UserResponse)
async def update_user(
    user_id: str,
    updates: UserUpdate,
    admin: dict = Depends(require_admin),
    db: SQLiteService = Depends(get_db)
):
    """
    Update user (admin only).
    """
    user = db.update_user(user_id, updates)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    return UserResponse.model_validate(user)


@users_router.delete("/{user_id}", response_model=SuccessResponse)
async def delete_user(
    user_id: str,
    admin: dict = Depends(require_admin),
    db: SQLiteService = Depends(get_db)
):
    """
    Delete user (admin only).
    """
    # Prevent self-deletion
    if user_id == admin["user_id"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot delete your own account"
        )
    
    success = db.delete_user(user_id)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    return SuccessResponse(message="User deleted successfully")


@users_router.post("/{user_id}/reset-password", response_model=SuccessResponse)
async def reset_user_password(
    user_id: str,
    request: PasswordResetRequest,
    admin: dict = Depends(require_admin),
    db: SQLiteService = Depends(get_db)
):
    """
    Reset user password (admin only).
    
    Does not require current password.
    """
    success = db.set_user_password(user_id, request.new_password)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    return SuccessResponse(message="Password reset successfully")


# ============================================================================
# User Settings Routes
# ============================================================================

@users_router.get("/{user_id}/settings", response_model=UserSettingsResponse)
async def get_user_settings(
    user_id: str,
    current_user: dict = Depends(get_current_user),
    db: SQLiteService = Depends(get_db)
):
    """
    Get user settings.
    
    Users can only access their own settings unless admin.
    """
    if current_user["user_id"] != user_id and current_user["role"] != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied"
        )
    
    settings = db.get_user_settings(user_id)
    if not settings:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Settings not found"
        )
    
    return UserSettingsResponse.model_validate(settings)


@users_router.patch("/{user_id}/settings", response_model=UserSettingsResponse)
async def update_user_settings(
    user_id: str,
    updates: UserSettingsUpdate,
    current_user: dict = Depends(get_current_user),
    db: SQLiteService = Depends(get_db)
):
    """
    Update user settings.
    
    Users can only update their own settings unless admin.
    """
    if current_user["user_id"] != user_id and current_user["role"] != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied"
        )
    
    settings = db.update_user_settings(user_id, updates)
    if not settings:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    return UserSettingsResponse.model_validate(settings)


# ============================================================================
# Session Routes
# ============================================================================

@users_router.get("/{user_id}/sessions", response_model=SessionListResponse)
async def get_user_sessions(
    user_id: str,
    active_only: bool = False,
    current_user: dict = Depends(get_current_user),
    db: SQLiteService = Depends(get_db)
):
    """
    Get user sessions.
    
    Users can only access their own sessions unless admin.
    """
    if current_user["user_id"] != user_id and current_user["role"] != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied"
        )
    
    sessions = db.get_user_sessions(user_id, active_only=active_only)
    return SessionListResponse(
        sessions=[SessionResponse.model_validate(s) for s in sessions],
        total=len(sessions)
    )
