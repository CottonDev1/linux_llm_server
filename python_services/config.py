"""
Configuration settings for Python Data Services
"""
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from project root (parent of python_services)
# This ensures single source of truth for configuration
# override=True ensures .env values take precedence over system environment variables
# This is critical for cross-platform deployment (Windows vs Linux)
_script_dir = Path(__file__).parent
_project_root = _script_dir.parent
load_dotenv(_project_root / ".env", override=True)

# Platform detection
IS_WINDOWS = sys.platform == "win32"
IS_LINUX = sys.platform.startswith("linux")

# MongoDB Configuration
# Default to local MongoDB on port 2017 unless overridden by .env
MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:2017")
MONGODB_DATABASE = os.getenv("MONGODB_DATABASE", "rag_server")
MONGODB_REPLICA_SET = os.getenv("MONGODB_REPLICA_SET", "")  # Set to "rs0" for replica set

# Vector Search Configuration (MongoDB 8.2+ or Atlas)
VECTOR_SEARCH_ENABLED = os.getenv("VECTOR_SEARCH_ENABLED", "true").lower() == "true"
# Tune down candidates for faster searches; override via .env if needed
VECTOR_SEARCH_NUM_CANDIDATES_MULTIPLIER = int(os.getenv("VECTOR_SEARCH_NUM_CANDIDATES_MULTIPLIER", "10"))

# Collections
COLLECTION_DOCUMENTS = "documents"
COLLECTION_CODE_CONTEXT = "code_context"
COLLECTION_SQL_KNOWLEDGE = "sql_knowledge"
COLLECTION_CODE_METHODS = "code_methods"
COLLECTION_CODE_CLASSES = "code_classes"
COLLECTION_CODE_CALLGRAPH = "code_callgraph"
COLLECTION_CODE_EVENTHANDLERS = "code_eventhandlers"
COLLECTION_CODE_DBOPERATIONS = "code_dboperations"
COLLECTION_AUDIO_ANALYSIS = "audio_analysis"

# SQL Knowledge specialized collections
COLLECTION_SQL_EXAMPLES = "sql_examples"
COLLECTION_SQL_FAILED_QUERIES = "sql_failed_queries"
COLLECTION_SQL_SCHEMA_CONTEXT = "sql_schema_context"
COLLECTION_SQL_STORED_PROCEDURES = "sql_stored_procedures"
COLLECTION_SQL_CORRECTIONS = "sql_corrections"  # User-provided corrections for RAG improvement

# Feedback system collections
COLLECTION_FEEDBACK = "feedback"
COLLECTION_QUERY_SESSIONS = "query_sessions"

# Ticket matching collections
COLLECTION_TICKET_MATCH_HISTORY = "ticket_match_history"

# Phone to customer mapping collection
COLLECTION_PHONE_CUSTOMER_MAP = "phone_customer_map"

# SQL Rules collection (migrated from config/sql_rules.json)
COLLECTION_SQL_RULES = "sql_rules"

# Embedding Configuration
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
# Dimensions: 384 for all-MiniLM-L6-v2, 768 for nomic-embed-text-v1.5
EMBEDDING_DIMENSIONS = int(os.getenv("EMBEDDING_DIMENSIONS", "384"))
# Remote embedding service URL (if set, uses remote API instead of local model)
EMBEDDING_SERVICE_URL = os.getenv("EMBEDDING_SERVICE_URL", "")

# Server Configuration
HOST = os.getenv("PYTHON_SERVICE_HOST", "0.0.0.0")
PORT = int(os.getenv("PYTHON_SERVICE_PORT", "8001"))

# Search Configuration
DEFAULT_SEARCH_LIMIT = 10
SIMILARITY_THRESHOLD = 0.4

# Git Configuration - OS-aware defaults
_DEFAULT_GIT_ROOT = r"C:\Projects\Git" if IS_WINDOWS else "/data/projects/git"
GIT_ROOT = os.getenv("GIT_ROOT", _DEFAULT_GIT_ROOT)

# Repository Configurations
# Each repository has a key, path, display name, and tracked file extensions
# Paths are computed from GIT_ROOT for cross-platform compatibility
def _repo_path(name: str) -> str:
    """Build repository path from GIT_ROOT"""
    return os.path.join(GIT_ROOT, name)

REPO_CONFIG = {
    "gin": {
        "name": "gin",
        "path": _repo_path("Gin"),
        "display_name": "Gin",
        "file_extensions": [".sql", ".cs", ".js", ".config", ".xml"],
        "enabled": True
    },
    "warehouse": {
        "name": "warehouse",
        "path": _repo_path("Warehouse"),
        "display_name": "Warehouse",
        "file_extensions": [".sql", ".cs", ".js", ".config", ".xml"],
        "enabled": True
    },
    "marketing": {
        "name": "marketing",
        "path": _repo_path("Marketing"),
        "display_name": "Marketing",
        "file_extensions": [".sql", ".cs", ".js", ".html", ".css"],
        "enabled": True
    },
    "ewr-library": {
        "name": "ewr-library",
        "path": _repo_path("EWR Library"),
        "display_name": "EWR Library",
        "file_extensions": [".cs", ".config", ".xml"],
        "enabled": True
    },
}

# Git Command Timeouts (seconds)
GIT_COMMAND_TIMEOUT = 60
GIT_PULL_TIMEOUT = 120
GIT_ANALYSIS_TIMEOUT = 300

# Logging Configuration
# Use shared logs directory with Node.js for unified logging
LOG_DIR = os.getenv("LOG_DIR", os.path.join(os.path.dirname(__file__), "..", "logs"))
LOG_SERVICE_FILE = "python-service.log"  # Match Node.js LOG_CATEGORIES
LOG_PIPELINE_FILE = "pipeline.log"
# Daily rotation settings - delete old file and create new one each day
LOG_ROTATION_WHEN = "midnight"  # Rotate at midnight
LOG_ROTATION_INTERVAL = 1  # Every 1 day
LOG_BACKUP_COUNT = 0  # Keep 0 backups (delete old file each day)
# Format: [PipelineName][User/IP] : message (timestamp added separately)
LOG_FORMAT = "%(asctime)s %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Audio Analysis Configuration
AUDIO_ANALYSIS_CONFIG = {
    "collection": COLLECTION_AUDIO_ANALYSIS,
    "sensevoice_model": "FunAudioLLM/SenseVoiceSmall",
    "device": os.getenv("SENSEVOICE_DEVICE", "cuda:0"),
    "vad_model": "fsmn-vad",
    "max_segment_time": 30000,  # milliseconds
    "batch_size_s": 60,  # seconds
    "supported_formats": [".wav", ".mp3", ".flac", ".m4a", ".ogg"],
    "max_file_size_mb": 100,
}

# Audio Analysis Options
MOOD_OPTIONS = ["Negative", "Positive", "Neutral"]
OUTCOME_OPTIONS = ["Issue Resolved", "Issue Unresolved", "Issue Logged in Central"]
