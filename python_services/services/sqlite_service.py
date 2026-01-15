"""
SQLite database service using SQLAlchemy.

Provides typed, safe database operations for:
- User management (CRUD, authentication)
- User settings and preferences
- Session and token management
- Git repository configuration
- Role and permission management
- System settings
- Tags and document tagging

This service is designed to replace the JavaScript EWRAIDatabase.js
with a cleaner, type-safe implementation.
"""
import os
import logging
from contextlib import contextmanager
from core.log_utils import log_info
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any, Generator
from pathlib import Path

from sqlalchemy import create_engine, select, delete, update, func, and_, text
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.exc import IntegrityError
import bcrypt

# Import models
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from .models.database import (
    Base, User, UserSettings, RefreshToken, SessionTracking,
    SystemSettings, SettingsHistory, GitRepository, RolePermission,
    UserPermission, Tag, DocumentTag,
    DEFAULT_SYSTEM_SETTINGS, DEFAULT_ADMIN_CATEGORIES, ALL_CATEGORIES
)
from .models.schemas import (
    UserCreate, UserUpdate, UserSettingsUpdate,
    GitRepositoryCreate, GitRepositoryUpdate, GitRepositoryPullUpdate,
    TagCreate, TagUpdate
)

logger = logging.getLogger(__name__)


class SQLiteService:
    """
    Service for SQLite database operations.
    
    Thread-safe through session-per-request pattern.
    Uses SQLAlchemy ORM for type-safe queries.
    """

    def __init__(self, db_path: str = None):
        """
        Initialize SQLite service.
        
        Args:
            db_path: Path to SQLite database file. 
                     Defaults to data/EWR_AI.db relative to python_services.
        """
        if db_path is None:
            base_dir = Path(__file__).parent.parent.parent
            db_path = str(base_dir / "data" / "EWR_AI.db")
        
        self.db_path = db_path
        self._engine = None
        self._session_factory = None
        self._initialized = False

    @property
    def engine(self):
        """Lazy-load database engine."""
        if self._engine is None:
            # Ensure directory exists
            os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
            
            # Create engine with WAL mode for better concurrency
            self._engine = create_engine(
                f"sqlite:///{self.db_path}",
                echo=False,
                connect_args={"check_same_thread": False}
            )
            
            # Enable WAL mode
            with self._engine.connect() as conn:
                conn.execute(text("PRAGMA journal_mode=WAL"))
                conn.commit()
        
        return self._engine

    @property
    def session_factory(self):
        """Lazy-load session factory."""
        if self._session_factory is None:
            self._session_factory = sessionmaker(
                bind=self.engine,
                expire_on_commit=False
            )
        return self._session_factory

    @contextmanager
    def get_session(self) -> Generator[Session, None, None]:
        """
        Get a database session with automatic cleanup.
        
        Usage:
            with service.get_session() as session:
                user = session.query(User).first()
        """
        session = self.session_factory()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def initialize(self):
        """
        Initialize database: create tables and seed default data.
        
        Safe to call multiple times - only creates missing tables/data.
        """
        if self._initialized:
            return

        log_info("SQLite Service", f"Initializing database at {self.db_path}")

        # Create all tables
        Base.metadata.create_all(self.engine)

        # Seed default data
        with self.get_session() as session:
            self._ensure_default_admin(session)
            self._ensure_default_settings(session)
            self._ensure_default_role_permissions(session)

        self._initialized = True
        log_info("SQLite Service", "Database initialized successfully")

    # ========================================================================
    # User Operations
    # ========================================================================

    def create_user(self, user_data: UserCreate) -> User:
        """
        Create a new user with default settings.
        
        Args:
            user_data: User creation data
            
        Returns:
            Created User object
            
        Raises:
            IntegrityError: If username already exists
        """
        with self.get_session() as session:
            # Hash password
            password_hash = bcrypt.hashpw(
                user_data.password.encode('utf-8'),
                bcrypt.gensalt(rounds=12)
            ).decode('utf-8')

            # Create user
            user = User(
                username=user_data.username,
                display_name=user_data.display_name,
                email=user_data.email,
                password_hash=password_hash,
                role=user_data.role,
                department=user_data.department
            )
            session.add(user)
            session.flush()  # Get user_id

            # Create default settings
            settings = UserSettings(user_id=user.user_id)
            session.add(settings)

            session.commit()
            
            # Refresh to get all relationships
            session.refresh(user)
            logger.info(f"Created user: {user.username} ({user.role})")
            return user

    def get_user_by_id(self, user_id: str) -> Optional[User]:
        """Get user by ID with settings loaded."""
        with self.get_session() as session:
            stmt = select(User).where(User.user_id == user_id)
            return session.execute(stmt).scalar_one_or_none()

    def get_user_by_username(self, username: str) -> Optional[User]:
        """Get user by username with settings loaded."""
        with self.get_session() as session:
            stmt = select(User).where(User.username == username)
            return session.execute(stmt).scalar_one_or_none()

    def authenticate_user(self, username: str, password: str) -> Optional[User]:
        """
        Authenticate user with username and password.
        
        Args:
            username: User's username
            password: Plain text password
            
        Returns:
            User object if authentication successful, None otherwise
        """
        with self.get_session() as session:
            stmt = select(User).where(User.username == username)
            user = session.execute(stmt).scalar_one_or_none()

            if not user or not user.is_active:
                return None

            if not user.password_hash:
                return None

            if not bcrypt.checkpw(password.encode('utf-8'), user.password_hash.encode('utf-8')):
                return None

            # Update last login
            user.last_login_at = datetime.utcnow()
            session.commit()
            
            logger.info(f"User authenticated: {username}")
            return user

    def get_all_users(self) -> List[User]:
        """Get all users ordered by creation date."""
        with self.get_session() as session:
            stmt = select(User).order_by(User.created_at.desc())
            return list(session.execute(stmt).scalars().all())

    def update_user(self, user_id: str, updates: UserUpdate) -> Optional[User]:
        """
        Update user fields.
        
        Args:
            user_id: User ID to update
            updates: Fields to update (only non-None values are applied)
            
        Returns:
            Updated User object or None if not found
        """
        with self.get_session() as session:
            user = session.execute(
                select(User).where(User.user_id == user_id)
            ).scalar_one_or_none()
            
            if not user:
                return None

            update_data = updates.model_dump(exclude_unset=True)
            for key, value in update_data.items():
                setattr(user, key, value)

            session.commit()
            session.refresh(user)
            logger.info(f"Updated user: {user.username}")
            return user

    def delete_user(self, user_id: str) -> bool:
        """
        Delete user and all related data (cascades).
        
        Args:
            user_id: User ID to delete
            
        Returns:
            True if deleted, False if not found
        """
        with self.get_session() as session:
            result = session.execute(
                delete(User).where(User.user_id == user_id)
            )
            deleted = result.rowcount > 0
            if deleted:
                logger.info(f"Deleted user: {user_id}")
            return deleted

    def set_user_password(self, user_id: str, new_password: str) -> bool:
        """
        Set user password (admin function, no current password required).
        
        Args:
            user_id: User ID
            new_password: New plain text password
            
        Returns:
            True if successful
        """
        with self.get_session() as session:
            password_hash = bcrypt.hashpw(
                new_password.encode('utf-8'),
                bcrypt.gensalt(rounds=12)
            ).decode('utf-8')

            result = session.execute(
                update(User)
                .where(User.user_id == user_id)
                .values(password_hash=password_hash, force_password_reset=False)
            )
            return result.rowcount > 0

    def change_user_password(
        self, user_id: str, current_password: str, new_password: str
    ) -> bool:
        """
        Change user password (requires current password verification).
        
        Args:
            user_id: User ID
            current_password: Current plain text password
            new_password: New plain text password
            
        Returns:
            True if successful, False if current password incorrect
        """
        with self.get_session() as session:
            user = session.execute(
                select(User).where(User.user_id == user_id)
            ).scalar_one_or_none()

            if not user or not user.password_hash:
                return False

            if not bcrypt.checkpw(
                current_password.encode('utf-8'),
                user.password_hash.encode('utf-8')
            ):
                return False

            password_hash = bcrypt.hashpw(
                new_password.encode('utf-8'),
                bcrypt.gensalt(rounds=12)
            ).decode('utf-8')

            user.password_hash = password_hash
            user.force_password_reset = False
            session.commit()
            
            logger.info(f"Password changed for user: {user.username}")
            return True

    # ========================================================================
    # User Settings Operations
    # ========================================================================

    def get_user_settings(self, user_id: str) -> Optional[UserSettings]:
        """Get user settings."""
        with self.get_session() as session:
            stmt = select(UserSettings).where(UserSettings.user_id == user_id)
            return session.execute(stmt).scalar_one_or_none()

    def update_user_settings(
        self, user_id: str, updates: UserSettingsUpdate
    ) -> Optional[UserSettings]:
        """
        Update user settings.
        
        Args:
            user_id: User ID
            updates: Settings to update (only non-None values are applied)
            
        Returns:
            Updated UserSettings or None if not found
        """
        with self.get_session() as session:
            settings = session.execute(
                select(UserSettings).where(UserSettings.user_id == user_id)
            ).scalar_one_or_none()

            if not settings:
                # Create settings if they don't exist
                settings = UserSettings(user_id=user_id)
                session.add(settings)

            update_data = updates.model_dump(exclude_unset=True)
            
            # Handle disabled_pages JSON serialization
            if "disabled_pages" in update_data:
                import json
                update_data["disabled_pages"] = json.dumps(update_data["disabled_pages"])

            for key, value in update_data.items():
                setattr(settings, key, value)

            settings.updated_at = datetime.utcnow()
            session.commit()
            session.refresh(settings)
            return settings

    # ========================================================================
    # Token Operations
    # ========================================================================

    def create_refresh_token(
        self, user_id: str, expires_days: int = 30
    ) -> RefreshToken:
        """
        Create a new refresh token for user.
        
        Args:
            user_id: User ID
            expires_days: Token validity in days
            
        Returns:
            RefreshToken with unhashed token in token_hash field
                      (caller should hash before storing, or use store_refresh_token)
        """
        import uuid
        
        token = str(uuid.uuid4())
        token_hash = bcrypt.hashpw(
            token.encode('utf-8'),
            bcrypt.gensalt(rounds=10)
        ).decode('utf-8')

        with self.get_session() as session:
            refresh_token = RefreshToken(
                user_id=user_id,
                token_hash=token_hash,
                expires_at=datetime.utcnow() + timedelta(days=expires_days)
            )
            session.add(refresh_token)
            session.commit()
            session.refresh(refresh_token)
            
            # Return with plain token for caller to use
            # (token_hash in DB is hashed)
            refresh_token._plain_token = token
            return refresh_token

    def verify_refresh_token(self, token: str) -> Optional[User]:
        """
        Verify refresh token and return associated user.
        
        Args:
            token: Plain text refresh token
            
        Returns:
            User if token valid and not expired, None otherwise
        """
        with self.get_session() as session:
            # Get all non-expired tokens
            stmt = select(RefreshToken).where(
                RefreshToken.expires_at > datetime.utcnow()
            )
            tokens = session.execute(stmt).scalars().all()

            for refresh_token in tokens:
                if bcrypt.checkpw(
                    token.encode('utf-8'),
                    refresh_token.token_hash.encode('utf-8')
                ):
                    # Found matching token, return user
                    user = session.execute(
                        select(User).where(User.user_id == refresh_token.user_id)
                    ).scalar_one_or_none()
                    return user

            return None

    def revoke_refresh_token(self, token: str) -> bool:
        """
        Revoke a refresh token.
        
        Args:
            token: Plain text refresh token
            
        Returns:
            True if token found and revoked
        """
        with self.get_session() as session:
            stmt = select(RefreshToken).where(
                RefreshToken.expires_at > datetime.utcnow()
            )
            tokens = session.execute(stmt).scalars().all()

            for refresh_token in tokens:
                if bcrypt.checkpw(
                    token.encode('utf-8'),
                    refresh_token.token_hash.encode('utf-8')
                ):
                    session.delete(refresh_token)
                    session.commit()
                    return True

            return False

    def revoke_user_tokens(self, user_id: str) -> int:
        """
        Revoke all refresh tokens for a user.
        
        Args:
            user_id: User ID
            
        Returns:
            Number of tokens revoked
        """
        with self.get_session() as session:
            result = session.execute(
                delete(RefreshToken).where(RefreshToken.user_id == user_id)
            )
            return result.rowcount

    def cleanup_expired_tokens(self) -> int:
        """
        Remove all expired refresh tokens.
        
        Returns:
            Number of tokens removed
        """
        with self.get_session() as session:
            result = session.execute(
                delete(RefreshToken).where(
                    RefreshToken.expires_at <= datetime.utcnow()
                )
            )
            count = result.rowcount
            if count > 0:
                logger.info(f"Cleaned up {count} expired tokens")
            return count

    # ========================================================================
    # Session Tracking Operations
    # ========================================================================

    def create_session(
        self,
        user_id: str,
        session_id: str,
        ip_address: str = None,
        user_agent: str = None
    ) -> SessionTracking:
        """Create a new session tracking record."""
        with self.get_session() as session:
            tracking = SessionTracking(
                session_id=session_id,
                user_id=user_id,
                ip_address=ip_address,
                user_agent=user_agent
            )
            session.add(tracking)
            session.commit()
            session.refresh(tracking)
            return tracking

    def end_session(self, session_id: str) -> bool:
        """Mark a session as ended."""
        with self.get_session() as session:
            result = session.execute(
                update(SessionTracking)
                .where(SessionTracking.session_id == session_id)
                .values(logout_at=datetime.utcnow())
            )
            return result.rowcount > 0

    def get_user_sessions(self, user_id: str, active_only: bool = False) -> List[SessionTracking]:
        """Get all sessions for a user."""
        with self.get_session() as session:
            stmt = select(SessionTracking).where(
                SessionTracking.user_id == user_id
            )
            if active_only:
                stmt = stmt.where(SessionTracking.logout_at.is_(None))
            stmt = stmt.order_by(SessionTracking.login_at.desc())
            return list(session.execute(stmt).scalars().all())

    # ========================================================================
    # System Settings Operations
    # ========================================================================

    def get_system_setting(self, section: str) -> Optional[str]:
        """Get a system setting value."""
        with self.get_session() as session:
            stmt = select(SystemSettings).where(SystemSettings.section == section)
            result = session.execute(stmt).scalar_one_or_none()
            return result.setting_value if result else None

    def set_system_setting(self, section: str, value: str) -> None:
        """Set a system setting value (upsert)."""
        with self.get_session() as session:
            existing = session.execute(
                select(SystemSettings).where(SystemSettings.section == section)
            ).scalar_one_or_none()

            if existing:
                existing.setting_value = value
            else:
                session.add(SystemSettings(section=section, setting_value=value))

    def get_all_system_settings(self) -> Dict[str, str]:
        """Get all system settings as a dictionary."""
        with self.get_session() as session:
            stmt = select(SystemSettings)
            results = session.execute(stmt).scalars().all()
            return {s.section: s.setting_value for s in results}

    def update_system_settings(self, settings: Dict[str, str]) -> None:
        """Update multiple system settings."""
        with self.get_session() as session:
            for section, value in settings.items():
                existing = session.execute(
                    select(SystemSettings).where(SystemSettings.section == section)
                ).scalar_one_or_none()

                if existing:
                    existing.setting_value = value
                else:
                    session.add(SystemSettings(section=section, setting_value=value))

    # ========================================================================
    # Git Repository Operations
    # ========================================================================

    def create_git_repository(self, repo_data: GitRepositoryCreate) -> GitRepository:
        """Create a new git repository configuration."""
        with self.get_session() as session:
            repo = GitRepository(
                name=repo_data.name,
                display_name=repo_data.display_name,
                path=repo_data.path,
                branch=repo_data.branch,
                project_name=repo_data.project_name,
                access_token=self._encrypt_token(repo_data.access_token) if repo_data.access_token else None,
                sync_interval=repo_data.sync_interval,
                auto_sync=repo_data.auto_sync
            )
            session.add(repo)
            session.commit()
            session.refresh(repo)
            logger.info(f"Created git repository: {repo.name}")
            return repo

    def get_git_repository(self, name: str) -> Optional[GitRepository]:
        """Get git repository by name."""
        with self.get_session() as session:
            stmt = select(GitRepository).where(GitRepository.name == name)
            return session.execute(stmt).scalar_one_or_none()

    def get_git_repository_by_id(self, repo_id: str) -> Optional[GitRepository]:
        """Get git repository by ID."""
        with self.get_session() as session:
            stmt = select(GitRepository).where(GitRepository.repo_id == repo_id)
            return session.execute(stmt).scalar_one_or_none()

    def get_all_git_repositories(self) -> List[GitRepository]:
        """Get all git repositories."""
        with self.get_session() as session:
            stmt = select(GitRepository).order_by(GitRepository.name)
            return list(session.execute(stmt).scalars().all())

    def update_git_repository(
        self, name: str, updates: GitRepositoryUpdate
    ) -> Optional[GitRepository]:
        """Update git repository configuration."""
        with self.get_session() as session:
            repo = session.execute(
                select(GitRepository).where(GitRepository.name == name)
            ).scalar_one_or_none()

            if not repo:
                return None

            update_data = updates.model_dump(exclude_unset=True)
            
            # Encrypt access token if provided
            if "access_token" in update_data and update_data["access_token"]:
                update_data["access_token"] = self._encrypt_token(update_data["access_token"])

            for key, value in update_data.items():
                setattr(repo, key, value)

            repo.updated_at = datetime.utcnow()
            session.commit()
            session.refresh(repo)
            logger.info(f"Updated git repository: {name}")
            return repo

    def update_git_repository_pull(
        self, name: str, pull_data: GitRepositoryPullUpdate
    ) -> Optional[GitRepository]:
        """Update git repository after a pull operation."""
        with self.get_session() as session:
            repo = session.execute(
                select(GitRepository).where(GitRepository.name == name)
            ).scalar_one_or_none()

            if not repo:
                return None

            repo.last_pull_time = datetime.utcnow()
            
            if pull_data.last_commit_hash:
                repo.last_commit_hash = pull_data.last_commit_hash
            if pull_data.last_commit_message:
                repo.last_commit_message = pull_data.last_commit_message
            if pull_data.last_commit_author:
                repo.last_commit_author = pull_data.last_commit_author
            if pull_data.branch:
                repo.branch = pull_data.branch

            repo.updated_at = datetime.utcnow()
            session.commit()
            session.refresh(repo)
            return repo

    def update_git_repository_analysis_date(self, name: str) -> Optional[GitRepository]:
        """Update the last analysis date for a repository."""
        with self.get_session() as session:
            repo = session.execute(
                select(GitRepository).where(GitRepository.name == name)
            ).scalar_one_or_none()

            if not repo:
                return None

            repo.last_analysis_date = datetime.utcnow()
            repo.updated_at = datetime.utcnow()
            session.commit()
            session.refresh(repo)
            return repo

    def delete_git_repository(self, name: str) -> bool:
        """Delete a git repository configuration."""
        with self.get_session() as session:
            result = session.execute(
                delete(GitRepository).where(GitRepository.name == name)
            )
            deleted = result.rowcount > 0
            if deleted:
                logger.info(f"Deleted git repository: {name}")
            return deleted

    def get_auto_sync_repositories(self) -> List[GitRepository]:
        """Get repositories with auto-sync enabled."""
        with self.get_session() as session:
            stmt = select(GitRepository).where(
                GitRepository.auto_sync == True
            ).order_by(GitRepository.name)
            return list(session.execute(stmt).scalars().all())

    def get_repositories_needing_analysis(self) -> List[GitRepository]:
        """Get repositories that need re-analysis."""
        with self.get_session() as session:
            stmt = select(GitRepository).where(
                (GitRepository.last_analysis_date.is_(None)) |
                (GitRepository.last_analysis_date < GitRepository.last_pull_time)
            ).order_by(GitRepository.name)
            return list(session.execute(stmt).scalars().all())

    def get_repository_access_token(self, name: str) -> Optional[str]:
        """Get decrypted access token for a repository."""
        repo = self.get_git_repository(name)
        if not repo or not repo.access_token:
            return None
        return self._decrypt_token(repo.access_token)

    # ========================================================================
    # Role Permission Operations
    # ========================================================================

    def get_role_categories(self, role: str) -> List[str]:
        """Get categories assigned to a role."""
        with self.get_session() as session:
            stmt = select(RolePermission.category).where(
                RolePermission.role == role
            ).order_by(RolePermission.category)
            results = session.execute(stmt).scalars().all()
            return list(results)

    def get_all_role_permissions(self) -> Dict[str, List[str]]:
        """Get all role permissions as a dictionary."""
        with self.get_session() as session:
            stmt = select(RolePermission).order_by(
                RolePermission.role, RolePermission.category
            )
            results = session.execute(stmt).scalars().all()
            
            permissions = {"user": [], "developer": [], "admin": []}
            for rp in results:
                if rp.role in permissions:
                    permissions[rp.role].append(rp.category)
            return permissions

    def set_role_categories(self, role: str, categories: List[str]) -> None:
        """Replace all categories for a role."""
        with self.get_session() as session:
            # Delete existing
            session.execute(
                delete(RolePermission).where(RolePermission.role == role)
            )
            
            # Insert new
            for category in categories:
                session.add(RolePermission(role=role, category=category))
            
            logger.info(f"Updated {role} role categories: {categories}")

    def add_role_category(self, role: str, category: str) -> None:
        """Add a category to a role."""
        with self.get_session() as session:
            existing = session.execute(
                select(RolePermission).where(
                    and_(RolePermission.role == role, RolePermission.category == category)
                )
            ).scalar_one_or_none()
            
            if not existing:
                session.add(RolePermission(role=role, category=category))

    def remove_role_category(self, role: str, category: str) -> None:
        """Remove a category from a role."""
        with self.get_session() as session:
            session.execute(
                delete(RolePermission).where(
                    and_(RolePermission.role == role, RolePermission.category == category)
                )
            )
            
            # Also remove user permissions for users with this role in this category
            subquery = select(User.user_id).where(User.role == role)
            session.execute(
                delete(UserPermission).where(
                    and_(
                        UserPermission.category == category,
                        UserPermission.user_id.in_(subquery)
                    )
                )
            )

    # ========================================================================
    # User Permission Operations
    # ========================================================================

    def get_user_permissions(self, user_id: str) -> List[Dict[str, str]]:
        """Get all page permissions for a user."""
        with self.get_session() as session:
            stmt = select(UserPermission).where(
                UserPermission.user_id == user_id
            ).order_by(UserPermission.category, UserPermission.page_id)
            results = session.execute(stmt).scalars().all()
            return [{"category": p.category, "page_id": p.page_id} for p in results]

    def get_user_category_permissions(self, user_id: str, category: str) -> List[str]:
        """Get page IDs enabled for user in a category."""
        with self.get_session() as session:
            stmt = select(UserPermission.page_id).where(
                and_(
                    UserPermission.user_id == user_id,
                    UserPermission.category == category
                )
            ).order_by(UserPermission.page_id)
            return list(session.execute(stmt).scalars().all())

    def set_user_page_permission(
        self, user_id: str, category: str, page_id: str, enabled: bool
    ) -> None:
        """Enable or disable a page for a user."""
        with self.get_session() as session:
            if enabled:
                existing = session.execute(
                    select(UserPermission).where(
                        and_(
                            UserPermission.user_id == user_id,
                            UserPermission.category == category,
                            UserPermission.page_id == page_id
                        )
                    )
                ).scalar_one_or_none()
                
                if not existing:
                    session.add(UserPermission(
                        user_id=user_id,
                        category=category,
                        page_id=page_id
                    ))
            else:
                session.execute(
                    delete(UserPermission).where(
                        and_(
                            UserPermission.user_id == user_id,
                            UserPermission.category == category,
                            UserPermission.page_id == page_id
                        )
                    )
                )

    def clear_user_category_permissions(self, user_id: str, category: str) -> None:
        """Remove all permissions for a user in a category."""
        with self.get_session() as session:
            session.execute(
                delete(UserPermission).where(
                    and_(
                        UserPermission.user_id == user_id,
                        UserPermission.category == category
                    )
                )
            )

    # ========================================================================
    # Tag Operations
    # ========================================================================

    def create_tag(self, tag_data: TagCreate) -> Tag:
        """Create a new tag."""
        with self.get_session() as session:
            tag = Tag(name=tag_data.name, description=tag_data.description)
            session.add(tag)
            session.commit()
            session.refresh(tag)
            logger.info(f"Created tag: {tag.name}")
            return tag

    def get_tag_by_id(self, tag_id: int) -> Optional[Tag]:
        """Get tag by ID."""
        with self.get_session() as session:
            return session.execute(
                select(Tag).where(Tag.id == tag_id)
            ).scalar_one_or_none()

    def get_tag_by_name(self, name: str) -> Optional[Tag]:
        """Get tag by name."""
        with self.get_session() as session:
            return session.execute(
                select(Tag).where(Tag.name == name)
            ).scalar_one_or_none()

    def get_all_tags(self) -> List[Tag]:
        """Get all tags."""
        with self.get_session() as session:
            stmt = select(Tag).order_by(Tag.name)
            return list(session.execute(stmt).scalars().all())

    def update_tag(self, tag_id: int, updates: TagUpdate) -> Optional[Tag]:
        """Update a tag."""
        with self.get_session() as session:
            tag = session.execute(
                select(Tag).where(Tag.id == tag_id)
            ).scalar_one_or_none()

            if not tag:
                return None

            update_data = updates.model_dump(exclude_unset=True)
            for key, value in update_data.items():
                setattr(tag, key, value)

            session.commit()
            session.refresh(tag)
            return tag

    def delete_tag(self, tag_id: int) -> bool:
        """Delete a tag (cascades to document_tags)."""
        with self.get_session() as session:
            result = session.execute(
                delete(Tag).where(Tag.id == tag_id)
            )
            return result.rowcount > 0

    def tag_document(self, document_id: str, tag_id: int) -> DocumentTag:
        """Add a tag to a document."""
        with self.get_session() as session:
            existing = session.execute(
                select(DocumentTag).where(
                    and_(
                        DocumentTag.document_id == document_id,
                        DocumentTag.tag_id == tag_id
                    )
                )
            ).scalar_one_or_none()

            if existing:
                return existing

            doc_tag = DocumentTag(document_id=document_id, tag_id=tag_id)
            session.add(doc_tag)
            session.commit()
            session.refresh(doc_tag)
            return doc_tag

    def untag_document(self, document_id: str, tag_id: int) -> bool:
        """Remove a tag from a document."""
        with self.get_session() as session:
            result = session.execute(
                delete(DocumentTag).where(
                    and_(
                        DocumentTag.document_id == document_id,
                        DocumentTag.tag_id == tag_id
                    )
                )
            )
            return result.rowcount > 0

    def get_document_tags(self, document_id: str) -> List[Tag]:
        """Get all tags for a document."""
        with self.get_session() as session:
            stmt = (
                select(Tag)
                .join(DocumentTag)
                .where(DocumentTag.document_id == document_id)
                .order_by(Tag.name)
            )
            return list(session.execute(stmt).scalars().all())

    def get_documents_by_tag(self, tag_id: int) -> List[str]:
        """Get all document IDs with a specific tag."""
        with self.get_session() as session:
            stmt = select(DocumentTag.document_id).where(
                DocumentTag.tag_id == tag_id
            )
            return list(session.execute(stmt).scalars().all())

    # ========================================================================
    # Private Helpers
    # ========================================================================

    def _ensure_default_admin(self, session: Session) -> None:
        """Create default admin user if none exists."""
        # Use first() instead of scalar_one_or_none() since multiple admins can exist
        admin = session.execute(
            select(User).where(User.role == "admin")
        ).first()

        if not admin:
            logger.info("Creating default admin user...")
            password_hash = bcrypt.hashpw(
                b"123",
                bcrypt.gensalt(rounds=12)
            ).decode('utf-8')

            admin = User(
                username="Admin",
                display_name="Administrator",
                email="admin@localhost",
                password_hash=password_hash,
                role="admin",
                department="IT"
            )
            session.add(admin)
            session.flush()

            # Create settings
            session.add(UserSettings(user_id=admin.user_id))
            
            logger.info("Default admin created: Admin / 123")

    def _ensure_default_settings(self, session: Session) -> None:
        """Initialize default system settings."""
        for section, value in DEFAULT_SYSTEM_SETTINGS.items():
            existing = session.execute(
                select(SystemSettings).where(SystemSettings.section == section)
            ).scalar_one_or_none()

            if not existing:
                session.add(SystemSettings(section=section, setting_value=value))

    def _ensure_default_role_permissions(self, session: Session) -> None:
        """Initialize default role permissions (admin gets all categories)."""
        admin_perms = session.execute(
            select(RolePermission).where(RolePermission.role == "admin")
        ).first()

        if not admin_perms:
            logger.info("Initializing default role permissions...")
            for category in DEFAULT_ADMIN_CATEGORIES:
                session.add(RolePermission(role="admin", category=category))
            logger.info(f"Admin role initialized with categories: {DEFAULT_ADMIN_CATEGORIES}")

    def _encrypt_token(self, token: str) -> str:
        """Encrypt access token using XOR cipher."""
        if not token:
            return ""
        
        key = os.getenv("TOKEN_ENCRYPTION_KEY", "CHANGE_THIS_KEY_IN_PRODUCTION_123456")
        encrypted = ''.join(
            chr(ord(c) ^ ord(key[i % len(key)]))
            for i, c in enumerate(token)
        )
        import base64
        return base64.b64encode(encrypted.encode('latin-1')).decode('utf-8')

    def _decrypt_token(self, encrypted: str) -> str:
        """Decrypt access token."""
        if not encrypted:
            return ""
        
        try:
            key = os.getenv("TOKEN_ENCRYPTION_KEY", "CHANGE_THIS_KEY_IN_PRODUCTION_123456")
            import base64
            decoded = base64.b64decode(encrypted.encode('utf-8')).decode('latin-1')
            return ''.join(
                chr(ord(c) ^ ord(key[i % len(key)]))
                for i, c in enumerate(decoded)
            )
        except Exception as e:
            logger.error(f"Error decrypting token: {e}")
            return ""


# ============================================================================
# Singleton Instance
# ============================================================================

_sqlite_service: Optional[SQLiteService] = None


def get_sqlite_service() -> SQLiteService:
    """
    Get singleton SQLite service instance.
    
    Initializes the database on first call.
    """
    global _sqlite_service
    if _sqlite_service is None:
        _sqlite_service = SQLiteService()
        _sqlite_service.initialize()
    return _sqlite_service


def reset_sqlite_service() -> None:
    """Reset the singleton instance (for testing)."""
    global _sqlite_service
    _sqlite_service = None
