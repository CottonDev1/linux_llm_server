from .settings import settings, TestSettings
from .test_config import (
    TestConfig,
    PipelineTestConfig,
    MongoDBConfig,
    LLMConfig,
    get_test_config,
    get_pipeline_config,
)

# Re-export python_services config values that may be imported by shared code
# This allows testing/config to take precedence in sys.path while still
# providing the config values that python_services modules expect
import sys
from pathlib import Path

_python_services_path = str(Path(__file__).parent.parent.parent / "python_services")
if _python_services_path not in sys.path:
    sys.path.append(_python_services_path)

# Import config values from python_services/config.py using the module's full path
# to avoid circular import issues
import importlib.util
_config_spec = importlib.util.spec_from_file_location(
    "python_services_config",
    Path(__file__).parent.parent.parent / "python_services" / "config.py"
)
_python_config = importlib.util.module_from_spec(_config_spec)
_config_spec.loader.exec_module(_python_config)

# Re-export all config values needed by python_services modules
MONGODB_URI = _python_config.MONGODB_URI
MONGODB_DATABASE = _python_config.MONGODB_DATABASE
MONGODB_REPLICA_SET = _python_config.MONGODB_REPLICA_SET
VECTOR_SEARCH_ENABLED = _python_config.VECTOR_SEARCH_ENABLED
VECTOR_SEARCH_NUM_CANDIDATES_MULTIPLIER = _python_config.VECTOR_SEARCH_NUM_CANDIDATES_MULTIPLIER
COLLECTION_DOCUMENTS = _python_config.COLLECTION_DOCUMENTS
COLLECTION_CODE_CONTEXT = _python_config.COLLECTION_CODE_CONTEXT
COLLECTION_SQL_KNOWLEDGE = _python_config.COLLECTION_SQL_KNOWLEDGE
COLLECTION_CODE_METHODS = _python_config.COLLECTION_CODE_METHODS
COLLECTION_CODE_CLASSES = _python_config.COLLECTION_CODE_CLASSES
COLLECTION_CODE_CALLGRAPH = _python_config.COLLECTION_CODE_CALLGRAPH
COLLECTION_CODE_EVENTHANDLERS = _python_config.COLLECTION_CODE_EVENTHANDLERS
COLLECTION_CODE_DBOPERATIONS = _python_config.COLLECTION_CODE_DBOPERATIONS
COLLECTION_SQL_EXAMPLES = _python_config.COLLECTION_SQL_EXAMPLES
COLLECTION_SQL_FAILED_QUERIES = _python_config.COLLECTION_SQL_FAILED_QUERIES
COLLECTION_SQL_SCHEMA_CONTEXT = _python_config.COLLECTION_SQL_SCHEMA_CONTEXT
COLLECTION_SQL_STORED_PROCEDURES = _python_config.COLLECTION_SQL_STORED_PROCEDURES
COLLECTION_SQL_CORRECTIONS = _python_config.COLLECTION_SQL_CORRECTIONS
COLLECTION_FEEDBACK = _python_config.COLLECTION_FEEDBACK
COLLECTION_QUERY_SESSIONS = _python_config.COLLECTION_QUERY_SESSIONS
COLLECTION_TICKET_MATCH_HISTORY = _python_config.COLLECTION_TICKET_MATCH_HISTORY
COLLECTION_AUDIO_ANALYSIS = _python_config.COLLECTION_AUDIO_ANALYSIS
COLLECTION_PHONE_CUSTOMER_MAP = _python_config.COLLECTION_PHONE_CUSTOMER_MAP
EMBEDDING_MODEL = _python_config.EMBEDDING_MODEL
EMBEDDING_DIMENSIONS = _python_config.EMBEDDING_DIMENSIONS
EMBEDDING_SERVICE_URL = _python_config.EMBEDDING_SERVICE_URL
DEFAULT_SEARCH_LIMIT = _python_config.DEFAULT_SEARCH_LIMIT
SIMILARITY_THRESHOLD = _python_config.SIMILARITY_THRESHOLD

__all__ = [
    "settings",
    "TestSettings",
    "TestConfig",
    "PipelineTestConfig",
    "MongoDBConfig",
    "LLMConfig",
    "get_test_config",
    "get_pipeline_config",
    # Re-exported python_services config values
    "MONGODB_URI",
    "MONGODB_DATABASE",
    "MONGODB_REPLICA_SET",
    "VECTOR_SEARCH_ENABLED",
    "VECTOR_SEARCH_NUM_CANDIDATES_MULTIPLIER",
    "COLLECTION_DOCUMENTS",
    "COLLECTION_CODE_CONTEXT",
    "COLLECTION_SQL_KNOWLEDGE",
    "COLLECTION_CODE_METHODS",
    "COLLECTION_CODE_CLASSES",
    "COLLECTION_CODE_CALLGRAPH",
    "COLLECTION_CODE_EVENTHANDLERS",
    "COLLECTION_CODE_DBOPERATIONS",
    "COLLECTION_SQL_EXAMPLES",
    "COLLECTION_SQL_FAILED_QUERIES",
    "COLLECTION_SQL_SCHEMA_CONTEXT",
    "COLLECTION_SQL_STORED_PROCEDURES",
    "COLLECTION_SQL_CORRECTIONS",
    "COLLECTION_FEEDBACK",
    "COLLECTION_QUERY_SESSIONS",
    "COLLECTION_TICKET_MATCH_HISTORY",
    "COLLECTION_AUDIO_ANALYSIS",
    "COLLECTION_PHONE_CUSTOMER_MAP",
    "EMBEDDING_MODEL",
    "EMBEDDING_DIMENSIONS",
    "EMBEDDING_SERVICE_URL",
    "DEFAULT_SEARCH_LIMIT",
    "SIMILARITY_THRESHOLD",
]
