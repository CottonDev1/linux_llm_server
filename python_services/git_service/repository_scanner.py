"""
Repository Scanner

Scans directories for Git repositories and manages repository configuration.
Migrated from JavaScript GitService.scanForGitRepositories().
"""

import os
import logging
from pathlib import Path
from typing import List, Optional, Dict

from .models import Repository, RepositoryConfig

logger = logging.getLogger(__name__)


class RepositoryScanner:
    """
    Scans for Git repositories and manages repository configuration.

    This class handles:
    - Scanning a directory for Git repositories
    - Loading repository configurations
    - Validating repository paths
    """

    # Default repository configurations matching the EWR setup
    DEFAULT_CONFIGS: Dict[str, RepositoryConfig] = {
        "gin": RepositoryConfig(
            name="gin",
            path=r"C:\Projects\Git\Gin",
            display_name="Gin",
            file_extensions=[".sql", ".cs", ".js", ".config", ".xml"],
            enabled=True
        ),
        "warehouse": RepositoryConfig(
            name="warehouse",
            path=r"C:\Projects\Git\Warehouse",
            display_name="Warehouse",
            file_extensions=[".sql", ".cs", ".js", ".config", ".xml"],
            enabled=True
        ),
        "marketing": RepositoryConfig(
            name="marketing",
            path=r"C:\Projects\Git\Marketing",
            display_name="Marketing",
            file_extensions=[".sql", ".cs", ".js", ".html", ".css"],
            enabled=True
        ),
        "ewr-library": RepositoryConfig(
            name="ewr-library",
            path=r"C:\Projects\Git\EWR Library",
            display_name="EWR Library",
            file_extensions=[".cs", ".config", ".xml"],
            enabled=True
        ),
    }

    def __init__(
        self,
        git_root: str = r"C:\Projects\Git",
        configs: Optional[Dict[str, RepositoryConfig]] = None
    ):
        """
        Initialize the repository scanner.

        Args:
            git_root: Root directory containing git repositories
            configs: Optional custom repository configurations
        """
        self.git_root = Path(git_root)
        self.configs = configs or self.DEFAULT_CONFIGS.copy()
        self._cached_repositories: List[Repository] = []

    def scan_for_repositories(self, force_refresh: bool = False) -> List[Repository]:
        """
        Scan the git root directory for git repositories.

        This method scans subdirectories of git_root for .git folders
        to identify valid Git repositories.

        Args:
            force_refresh: If True, bypass cache and rescan

        Returns:
            List of Repository objects found
        """
        if self._cached_repositories and not force_refresh:
            return self._cached_repositories

        repositories: List[Repository] = []

        if not self.git_root.exists():
            logger.warning(f"Git root directory not found: {self.git_root}")
            return repositories

        try:
            for entry in self.git_root.iterdir():
                if entry.is_dir():
                    git_path = entry / ".git"

                    # Check if this directory contains a .git folder
                    if git_path.exists():
                        repositories.append(Repository(
                            name=entry.name,
                            path=str(entry),
                            display_name=entry.name
                        ))

            logger.info(f"Found {len(repositories)} git repositories in {self.git_root}")
            self._cached_repositories = repositories
            return repositories

        except PermissionError as e:
            logger.error(f"Permission denied scanning {self.git_root}: {e}")
            return repositories
        except Exception as e:
            logger.error(f"Error scanning for git repositories: {e}")
            return repositories

    def get_repository_path(self, repo_name: str) -> Optional[str]:
        """
        Get the full path to a repository by name.

        First checks configured repositories, then scans for matches.

        Args:
            repo_name: Name of the repository (case-insensitive)

        Returns:
            Full path to repository or None if not found
        """
        # Normalize the name for comparison
        repo_name_lower = repo_name.lower().replace(" ", "-").replace("_", "-")

        # Check configured repositories first
        if repo_name_lower in self.configs:
            config = self.configs[repo_name_lower]
            if Path(config.path).exists():
                return config.path

        # Scan and check by name
        repos = self.scan_for_repositories()
        for repo in repos:
            if repo.name.lower() == repo_name.lower():
                return repo.path
            # Also check display name
            if repo.display_name.lower() == repo_name.lower():
                return repo.path

        return None

    def get_repository_config(self, repo_name: str) -> Optional[RepositoryConfig]:
        """
        Get the configuration for a repository by name.

        Args:
            repo_name: Name of the repository

        Returns:
            RepositoryConfig or None if not configured
        """
        repo_name_lower = repo_name.lower().replace(" ", "-").replace("_", "-")
        return self.configs.get(repo_name_lower)

    def verify_repository(self, repo_path: str) -> bool:
        """
        Verify that a path is a valid Git repository.

        Args:
            repo_path: Path to check

        Returns:
            True if valid Git repository
        """
        path = Path(repo_path)
        git_path = path / ".git"
        return path.exists() and git_path.exists()

    def add_repository_config(self, config: RepositoryConfig) -> bool:
        """
        Add or update a repository configuration.

        Args:
            config: Repository configuration to add

        Returns:
            True if added successfully
        """
        key = config.name.lower().replace(" ", "-")
        self.configs[key] = config
        # Clear cache to force rescan
        self._cached_repositories = []
        logger.info(f"Added repository config: {config.name}")
        return True

    def remove_repository_config(self, repo_name: str) -> bool:
        """
        Remove a repository configuration.

        Args:
            repo_name: Name of repository to remove

        Returns:
            True if removed, False if not found
        """
        key = repo_name.lower().replace(" ", "-")
        if key in self.configs:
            del self.configs[key]
            self._cached_repositories = []
            logger.info(f"Removed repository config: {repo_name}")
            return True
        return False

    def get_configured_repositories(self) -> List[RepositoryConfig]:
        """
        Get all configured repositories.

        Returns:
            List of repository configurations
        """
        return list(self.configs.values())

    def get_enabled_repositories(self) -> List[RepositoryConfig]:
        """
        Get only enabled repository configurations.

        Returns:
            List of enabled repository configurations
        """
        return [c for c in self.configs.values() if c.enabled]

    def get_file_extensions_for_repo(self, repo_name: str) -> List[str]:
        """
        Get the tracked file extensions for a repository.

        Args:
            repo_name: Name of the repository

        Returns:
            List of file extensions or defaults
        """
        config = self.get_repository_config(repo_name)
        if config:
            return config.file_extensions
        # Default extensions
        return [".cs", ".js", ".sql", ".config", ".xml"]
