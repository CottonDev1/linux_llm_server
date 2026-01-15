"""
SQL Pipeline Test Configuration
===============================

Provides SQL connection fixtures that read from Prefect variables.
Variables can be modified in the Prefect UI at http://localhost:4200/variables

Prefect Variables:
    - sql_server: SQL Server hostname
    - sql_database: Database name
    - sql_username: Username for SQL auth
    - sql_password: Password for SQL auth
    - sql_domain: Domain for Windows auth
    - sql_auth_type: Authentication type (sql or windows)
"""

import pytest
from typing import Dict, Any, Optional
from dataclasses import dataclass

# Try to import Prefect variables, fall back to defaults if not available
try:
    from prefect.variables import Variable
    PREFECT_AVAILABLE = True
except ImportError:
    PREFECT_AVAILABLE = False


@dataclass
class SQLConnectionConfig:
    """SQL connection configuration."""
    server: str
    database: str
    username: str
    password: str
    domain: str
    auth_type: str

    def to_request_dict(self, auth_type: Optional[str] = None) -> Dict[str, Any]:
        """Convert to request dictionary for API calls."""
        use_auth = auth_type or self.auth_type

        if use_auth == "windows":
            return {
                "server": self.server,
                "database": self.database,
                "username": self.username,
                "password": self.password,
                "domain": self.domain,
                "authType": "windows",
            }
        else:
            return {
                "server": self.server,
                "database": self.database,
                "username": self.username,
                "password": self.password,
                "authType": "sql",
            }


def get_prefect_variable(name: str, default: str) -> str:
    """Get a Prefect variable value, falling back to default if not available."""
    if not PREFECT_AVAILABLE:
        return default

    try:
        value = Variable.get(name)
        return value if value is not None else default
    except Exception:
        return default


# Default values (used if Prefect variables not set)
DEFAULTS = {
    "sql_server": "NCSQLTEST",
    "sql_database": "EWRCentral",
    "sql_username": "EWRUser",
    "sql_password": "66a3904d69",
    "sql_domain": "EWR",
    "sql_auth_type": "sql",
}


@pytest.fixture(scope="session")
def sql_connection_config() -> SQLConnectionConfig:
    """
    Get SQL connection configuration from Prefect variables.

    These values can be changed in the Prefect UI under Variables:
    - sql_server, sql_database, sql_username, sql_password, sql_domain, sql_auth_type

    Falls back to default test values if Prefect is not available.
    """
    return SQLConnectionConfig(
        server=get_prefect_variable("sql_server", DEFAULTS["sql_server"]),
        database=get_prefect_variable("sql_database", DEFAULTS["sql_database"]),
        username=get_prefect_variable("sql_username", DEFAULTS["sql_username"]),
        password=get_prefect_variable("sql_password", DEFAULTS["sql_password"]),
        domain=get_prefect_variable("sql_domain", DEFAULTS["sql_domain"]),
        auth_type=get_prefect_variable("sql_auth_type", DEFAULTS["sql_auth_type"]),
    )


@pytest.fixture
def sql_connection_dict(sql_connection_config: SQLConnectionConfig) -> Dict[str, Any]:
    """Get SQL connection as a dictionary for API requests."""
    return sql_connection_config.to_request_dict()


@pytest.fixture
def sql_connection_windows(sql_connection_config: SQLConnectionConfig) -> Dict[str, Any]:
    """Get SQL connection with Windows auth for API requests."""
    return sql_connection_config.to_request_dict(auth_type="windows")


@pytest.fixture
def valid_query_request(sql_connection_config: SQLConnectionConfig) -> Dict[str, Any]:
    """Standard valid query request using Prefect variables."""
    return {
        "question": "Show me all tables",
        "connectionInfo": sql_connection_config.to_request_dict(),
        "options": {
            "includeSchema": True,
            "maxResults": 100,
        }
    }


@pytest.fixture
def valid_stream_request(sql_connection_config: SQLConnectionConfig) -> Dict[str, Any]:
    """Standard valid streaming request using Prefect variables."""
    return {
        "question": "Show me all tables",
        "connectionInfo": sql_connection_config.to_request_dict(),
        "conversationHistory": [],
        "options": {
            "trustServerCertificate": True,
            "encrypt": False,
            "maxTokens": 2000,
        }
    }
