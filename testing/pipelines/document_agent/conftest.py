"""
Document Pipeline Test Configuration
=====================================

Provides document test fixtures that read from Prefect variables.
Variables can be modified in the Prefect UI at http://localhost:4200/variables

Prefect Variables:
    - doc_upload_path: Path to document file for upload tests
    - doc_categories: JSON array of categories to apply to documents
"""

import json
import pytest
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from pathlib import Path

# Try to import Prefect variables, fall back to defaults if not available
try:
    from prefect.variables import Variable
    PREFECT_AVAILABLE = True
except ImportError:
    PREFECT_AVAILABLE = False


@dataclass
class DocumentTestConfig:
    """Document test configuration."""
    upload_path: str
    categories: List[str]

    def get_upload_request(self, question: str = "Summarize this document") -> Dict[str, Any]:
        """Get a document upload request dictionary."""
        return {
            "filePath": self.upload_path,
            "categories": self.categories,
            "question": question,
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
    "doc_upload_path": "/data/projects/llm_website/testing_data/documents/sample.pdf",
    "doc_categories": '["General", "Technical"]',
}


@pytest.fixture(scope="session")
def document_test_config() -> DocumentTestConfig:
    """
    Get document test configuration from Prefect variables.

    These values can be changed in the Prefect UI under Variables:
    - doc_upload_path: Path to the document to upload
    - doc_categories: JSON array of categories

    Falls back to default test values if Prefect is not available.
    """
    categories_json = get_prefect_variable("doc_categories", DEFAULTS["doc_categories"])
    try:
        categories = json.loads(categories_json)
    except json.JSONDecodeError:
        categories = ["General"]

    return DocumentTestConfig(
        upload_path=get_prefect_variable("doc_upload_path", DEFAULTS["doc_upload_path"]),
        categories=categories,
    )


@pytest.fixture
def doc_upload_path(document_test_config: DocumentTestConfig) -> str:
    """Get document upload path from Prefect variables."""
    return document_test_config.upload_path


@pytest.fixture
def doc_categories(document_test_config: DocumentTestConfig) -> List[str]:
    """Get document categories from Prefect variables."""
    return document_test_config.categories


@pytest.fixture
def valid_document_request(document_test_config: DocumentTestConfig) -> Dict[str, Any]:
    """Standard valid document request using Prefect variables."""
    return document_test_config.get_upload_request()


@pytest.fixture
def document_query_request(document_test_config: DocumentTestConfig) -> Dict[str, Any]:
    """Document query request with categories from Prefect variables."""
    return {
        "question": "What is this document about?",
        "categories": document_test_config.categories,
        "options": {
            "includeContext": True,
            "maxResults": 5,
        }
    }
