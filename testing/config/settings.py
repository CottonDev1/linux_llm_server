"""
Centralized test configuration using Pydantic Settings.
Loads all values from .env file - NO HARDCODED VALUES.
"""
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, field_validator
from typing import Literal
from pathlib import Path
import os


class TestSettings(BaseSettings):
    """
    Test configuration loaded from environment variables.

    The .env file is NOT gitignored - it's committed for consistency.
    Override sensitive values on production servers via environment variables.
    """

    model_config = SettingsConfigDict(
        env_file=Path(__file__).parent.parent / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # Environment
    test_environment: Literal["development", "staging", "production"] = "development"

    # MongoDB
    mongodb_uri: str = Field(default="mongodb://localhost:27017")
    mongodb_database: str = Field(default="rag_server")

    # Local LLM Endpoints (MUST be localhost)
    llm_sql_endpoint: str = Field(default="http://localhost:8080")
    llm_general_endpoint: str = Field(default="http://localhost:8081")
    llm_code_endpoint: str = Field(default="http://localhost:8082")

    # Frontend
    frontend_url: str = Field(default="http://localhost:3000")

    # Credentials
    admin_username: str = Field(default="")
    admin_password: str = Field(default="")
    sql_server: str = Field(default="")
    sql_database: str = Field(default="")
    sql_username: str = Field(default="")
    sql_password: str = Field(default="")

    # Playwright Settings
    playwright_headless: bool = Field(default=True)
    playwright_screenshot: Literal["off", "on", "only-on-failure"] = Field(default="off")
    playwright_video: Literal["off", "on", "retain-on-failure", "on-first-retry"] = Field(default="off")
    playwright_trace: Literal["off", "on", "retain-on-failure", "on-first-retry"] = Field(default="off")
    playwright_timeout: int = Field(default=60000)
    playwright_navigation_timeout: int = Field(default=30000)
    playwright_workers: int = Field(default=1)

    # Pytest Settings
    pytest_timeout: int = Field(default=300)
    pytest_verbose: bool = Field(default=True)

    # Test Execution
    test_cleanup_after: bool = Field(default=True)
    test_use_prefix: bool = Field(default=True)
    test_run_id_format: str = Field(default="%Y%m%d_%H%M%S")

    # Prefect
    prefect_api_url: str = Field(default="http://localhost:4200/api")
    prefect_work_pool: str = Field(default="default-agent-pool")

    # Reporting
    reports_dir: str = Field(default="reports")

    # Computed paths
    @property
    def testing_root(self) -> Path:
        return Path(__file__).parent.parent

    @property
    def project_root(self) -> Path:
        return self.testing_root.parent

    @property
    def fixtures_dir(self) -> Path:
        return self.testing_root / "fixtures"

    @property
    def reports_path(self) -> Path:
        return self.testing_root / self.reports_dir

    @field_validator("llm_sql_endpoint", "llm_general_endpoint", "llm_code_endpoint")
    @classmethod
    def validate_local_only(cls, v: str) -> str:
        """Ensure LLM endpoints are local - no external APIs allowed."""
        if not v.startswith(("http://localhost", "http://127.0.0.1")):
            raise ValueError(
                f"Only local LLM endpoints allowed. Got: {v}. "
                "External APIs (OpenAI, Anthropic, etc.) are not permitted."
            )
        return v

    def get_playwright_config(self) -> dict:
        """Get Playwright configuration as a dictionary."""
        return {
            "headless": self.playwright_headless,
            "screenshot": self.playwright_screenshot,
            "video": self.playwright_video,
            "trace": self.playwright_trace,
            "timeout": self.playwright_timeout,
            "navigationTimeout": self.playwright_navigation_timeout,
            "workers": self.playwright_workers,
        }

    def to_env_dict(self) -> dict:
        """Export settings as environment variables for subprocess."""
        return {
            "MONGODB_URI": self.mongodb_uri,
            "MONGODB_DATABASE": self.mongodb_database,
            "LLM_SQL_ENDPOINT": self.llm_sql_endpoint,
            "LLM_GENERAL_ENDPOINT": self.llm_general_endpoint,
            "LLM_CODE_ENDPOINT": self.llm_code_endpoint,
            "FRONTEND_URL": self.frontend_url,
            "ADMIN_USERNAME": self.admin_username,
            "ADMIN_PASSWORD": self.admin_password,
            "SQL_SERVER": self.sql_server,
            "SQL_DATABASE": self.sql_database,
            "SQL_USERNAME": self.sql_username,
            "SQL_PASSWORD": self.sql_password,
            "PLAYWRIGHT_HEADLESS": str(self.playwright_headless).lower(),
            "PLAYWRIGHT_SCREENSHOT": self.playwright_screenshot,
            "PLAYWRIGHT_VIDEO": self.playwright_video,
            "PLAYWRIGHT_TRACE": self.playwright_trace,
            "PLAYWRIGHT_TIMEOUT": str(self.playwright_timeout),
            "TEST_CLEANUP_AFTER": str(self.test_cleanup_after).lower(),
        }


# Singleton instance
settings = TestSettings()
