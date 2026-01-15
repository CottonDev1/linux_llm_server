"""
SQL Pipeline Tests - Shared Fixtures
=====================================

Provides fixtures for SQL pipeline testing with Prefect integration.

Environment Variables (set by Prefect test flow):
- MONGODB_URI: MongoDB connection URI
- MONGODB_DATABASE: Database name (default: llm_website)
- LLAMACPP_SQL_HOST: SQL LLM endpoint (default: http://localhost:8080)
- TEST_CLEANUP: Whether to clean up test data (default: true)
"""

import os
import sys
import pytest
import asyncio
from typing import Dict, Any, List, Optional
from unittest.mock import MagicMock, AsyncMock

# Add python_services to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))


# Configuration from environment
@pytest.fixture
def mongodb_uri() -> str:
    """Get MongoDB URI from environment.

    Note: Uses directConnection=true for MongoDB Atlas Local which runs as a replica set
    with internal Docker hostnames that aren't resolvable from outside the container.
    """
    default_uri = "mongodb://EWRSPT-AI:27018/?directConnection=true"
    return os.environ.get("MONGODB_URI", default_uri)


@pytest.fixture
def mongodb_database() -> str:
    """Get MongoDB database name from environment.

    Note: Default is 'rag_server' which matches the actual MongoDB database
    where SQL schema context is stored.
    """
    return os.environ.get("MONGODB_DATABASE", "rag_server")


@pytest.fixture
def llm_sql_endpoint() -> str:
    """Get SQL LLM endpoint from environment."""
    return os.environ.get("LLAMACPP_SQL_HOST", "http://localhost:8080")


@pytest.fixture
def should_cleanup() -> bool:
    """Check if test data should be cleaned up."""
    return os.environ.get("TEST_CLEANUP", "true").lower() == "true"


# Mock fixtures for isolated testing
@pytest.fixture
def mock_mongodb():
    """Create a mock MongoDB service."""
    mock = MagicMock()
    mock.connect = AsyncMock(return_value=True)
    mock.get_schema_by_table = AsyncMock(return_value={
        "table_name": "CentralTickets",
        "columns": [
            {"name": "CentralTicketID", "type": "int", "nullable": False},
            {"name": "AddTicketDate", "type": "datetime", "nullable": True},
            {"name": "TicketTypeID", "type": "int", "nullable": True},
        ],
        "foreign_keys": [
            {"column": "TicketTypeID", "references_table": "Types", "references_column": "TypeID"}
        ],
        "related_tables": ["Types", "CentralUsers"],
        "sample_values": {"AddTicketDate": ["2024-01-15", "2024-02-20"]}
    })
    mock.hybrid_schema_retrieval = AsyncMock(return_value=[
        {
            "table_name": "CentralTickets",
            "score": 0.95,
            "columns": [
                {"name": "CentralTicketID", "type": "int"},
                {"name": "AddTicketDate", "type": "datetime"},
            ]
        }
    ])
    return mock


@pytest.fixture
def mock_llm_client():
    """Create a mock LLM client."""
    mock = MagicMock()
    mock.generate = AsyncMock(return_value={
        "sql": "SELECT * FROM CentralTickets WHERE CAST(AddTicketDate AS DATE) = CAST(GETDATE() AS DATE)",
        "confidence": 0.85
    })
    return mock


@pytest.fixture
def sample_schema_context() -> Dict[str, Any]:
    """Sample schema context for testing."""
    return {
        "tables": [
            {
                "table_name": "CentralTickets",
                "schema": "dbo",
                "columns": [
                    {"name": "CentralTicketID", "type": "int", "nullable": False, "is_primary": True},
                    {"name": "AddTicketDate", "type": "datetime", "nullable": True},
                    {"name": "TicketTypeID", "type": "int", "nullable": True},
                    {"name": "AddCentralUserID", "type": "int", "nullable": True},
                ],
                "foreign_keys": [
                    {"column": "TicketTypeID", "references_table": "Types", "references_column": "TypeID"},
                    {"column": "AddCentralUserID", "references_table": "CentralUsers", "references_column": "CentralUserID"},
                ],
                "related_tables": ["Types", "CentralUsers"],
                "sample_values": {
                    "AddTicketDate": ["2024-01-15", "2024-02-20", "2024-03-10"]
                },
                "summary": "Central ticket tracking table"
            },
            {
                "table_name": "Types",
                "schema": "dbo",
                "columns": [
                    {"name": "TypeID", "type": "int", "nullable": False, "is_primary": True},
                    {"name": "TypeName", "type": "varchar", "nullable": True},
                    {"name": "TypeDescription", "type": "varchar", "nullable": True},
                ],
                "foreign_keys": [],
                "related_tables": [],
                "sample_values": {}
            },
            {
                "table_name": "CentralUsers",
                "schema": "dbo",
                "columns": [
                    {"name": "CentralUserID", "type": "int", "nullable": False, "is_primary": True},
                    {"name": "UserName", "type": "varchar", "nullable": True},
                    {"name": "Email", "type": "varchar", "nullable": True},
                ],
                "foreign_keys": [],
                "related_tables": [],
                "sample_values": {}
            }
        ],
        "stored_procedures": [
            {
                "name": "RecapGet",
                "parameters": [
                    {"name": "@StartDate", "type": "datetime"},
                    {"name": "@EndDate", "type": "datetime"}
                ],
                "description": "Get recap report for date range"
            }
        ],
        "database": "EWRCentral"
    }


@pytest.fixture
def sample_sql_rules() -> List[Dict[str, Any]]:
    """Sample SQL rules for testing."""
    return [
        {
            "id": "ticket-date-column",
            "description": "Use AddTicketDate for ticket creation date",
            "type": "constraint",
            "priority": "critical",
            "trigger_keywords": ["ticket", "created", "date"],
            "trigger_tables": ["CentralTickets"],
            "rule_text": "Use AddTicketDate (NOT CreateDate) for ticket creation date",
            "auto_fix": {
                "pattern": r"CreateDate",
                "replacement": "AddTicketDate"
            }
        },
        {
            "id": "tickets-today",
            "description": "Query for today's tickets",
            "type": "assistance",
            "trigger_keywords": ["today", "tickets"],
            "example": {
                "question": "Show tickets created today",
                "sql": "SELECT * FROM CentralTickets WHERE CAST(AddTicketDate AS DATE) = CAST(GETDATE() AS DATE)"
            }
        }
    ]


@pytest.fixture
def test_questions() -> List[Dict[str, str]]:
    """Test questions with expected SQL patterns."""
    return [
        {
            "question": "Show tickets created today",
            "expected_pattern": "CAST(AddTicketDate AS DATE) = CAST(GETDATE() AS DATE)",
            "database": "EWRCentral"
        },
        {
            "question": "Count tickets by type",
            "expected_pattern": "GROUP BY",
            "database": "EWRCentral"
        },
        {
            "question": "Show tickets with their status",
            "expected_pattern": "JOIN",
            "database": "EWRCentral"
        }
    ]


# Event loop fixture for async tests
@pytest.fixture(scope="session")
def event_loop():
    """Create an event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()
