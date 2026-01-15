"""
Log Service Module
Handles logging configuration, file rotation, and log parsing for admin API endpoints.

This module provides:
1. Centralized logging configuration with file rotation
2. Separate loggers for service and pipeline operations
3. Log file parsing with filtering by level and timestamp
4. JSON-formatted log output for API responses
5. PipelineLogger class for unified format: [PipelineName][User/IP/System] : message
"""
import os
import re
import logging
import threading
from logging.handlers import TimedRotatingFileHandler
from datetime import datetime
from typing import Optional, List, Dict, Any, Literal, Union
from dataclasses import dataclass
from fastapi import Request

from config import (
    LOG_DIR, LOG_SERVICE_FILE, LOG_PIPELINE_FILE,
    LOG_ROTATION_WHEN, LOG_ROTATION_INTERVAL, LOG_BACKUP_COUNT,
    LOG_FORMAT, LOG_DATE_FORMAT
)


# ============================================================================
# Pipeline Names (matching Node.js PIPELINES)
# ============================================================================

PIPELINES = {
    'DOCUMENT': 'DocumentPipeline',
    'QUERY': 'QueryPipeline',
    'EMBEDDING': 'EmbeddingPipeline',
    'ROSLYN': 'RoslynPipeline',
    'GIT': 'GitPipeline',
    'SQL': 'SQLPipeline',
    'LLM': 'LLMPipeline',
    'AUTH': 'AuthPipeline',
    'ADMIN': 'AdminPipeline',
    'SYSTEM': 'System',
    'WEBSERVER': 'WebServer',
    'PYTHON': 'PythonService',
    'EXTRACTION': 'ExtractionPipeline',
    'SUMMARIZATION': 'SummarizationPipeline',
    'MONGODB': 'MongoDBService'
}


# ============================================================================
# PipelineLogger Class - Unified Logging Format
# Format: HH:MM:SS AM/PM | PipelineName      |  LEVEL: [User] message
# ============================================================================

class PipelineLogger:
    """
    Pipeline-specific logger with unified format matching log_utils.py.

    Format: HH:MM:SS AM/PM | PipelineName      |  LEVEL: [User] message

    Thread-safe: Uses a lock to prevent concurrent stdout writes.

    Usage:
        logger = PipelineLogger('DocumentPipeline')
        logger.info('Processing document upload', request)
        logger.error('Failed to process', error, request)
    """

    # Tag column width for alignment (matching log_utils.py)
    TAG_WIDTH = 18

    # Thread lock for safe concurrent logging (shared across all instances)
    _log_lock = threading.Lock()

    def __init__(self, pipeline_name: str):
        """
        Initialize logger for a specific pipeline.

        Args:
            pipeline_name: Name of the pipeline (e.g., 'DocumentPipeline', 'SQLPipeline')
        """
        self.pipeline_name = pipeline_name
        self._logger = logging.getLogger('pipeline')
        self._service_logger = logging.getLogger('service')

    def _get_user_identifier(self, context: Optional[Union[Request, str, Dict[str, Any]]] = None) -> str:
        """
        Extract user identifier from request or context.

        Args:
            context: FastAPI Request, username string, or dict with user info

        Returns:
            User identifier (username, IP, or 'System')
        """
        if context is None:
            return 'System'

        # If it's a string, use it directly
        if isinstance(context, str):
            return context

        # If it's a dict with user info
        if isinstance(context, dict):
            if 'username' in context:
                return context['username']
            if 'user' in context and isinstance(context['user'], dict):
                return context['user'].get('username', 'System')
            if 'ip' in context:
                return context['ip']
            return 'System'

        # If it's a FastAPI Request object
        if hasattr(context, 'client'):
            try:
                # Try to get username from request state
                if hasattr(context, 'state') and hasattr(context.state, 'user'):
                    user = context.state.user
                    if isinstance(user, dict) and 'username' in user:
                        return user['username']

                # Fall back to client IP
                if context.client:
                    ip = context.client.host
                    # Check for X-Forwarded-For header
                    forwarded = context.headers.get('x-forwarded-for')
                    if forwarded:
                        ip = forwarded.split(',')[0].strip()
                    # Clean up localhost IP
                    if ip in ('::1', '::ffff:127.0.0.1'):
                        ip = '127.0.0.1'
                    return ip
            except Exception:
                pass

        return 'System'

    def _format_message(self, user: str, message: str, level: str = "INFO") -> str:
        """Format message with pipeline and user info in new format."""
        timestamp = datetime.now().strftime("%I:%M:%S %p")
        padded_tag = self.pipeline_name[:self.TAG_WIDTH].ljust(self.TAG_WIDTH)
        return f"{timestamp} | {padded_tag}|  {level}: [{user}] {message}"

    def _format_console(self, user: str, message: str, level: str = "INFO") -> str:
        """Format message for console output with colors."""
        timestamp = datetime.now().strftime("%I:%M:%S %p")
        padded_tag = self.pipeline_name[:self.TAG_WIDTH].ljust(self.TAG_WIDTH)
        # Use cyan for INFO, yellow for WARN, red for ERROR
        if level == "ERROR":
            color = "\033[91m"  # Red
        elif level == "WARN":
            color = "\033[93m"  # Yellow
        else:
            color = "\033[96m"  # Cyan
        reset = "\033[0m"
        return f"{timestamp} {color}| {padded_tag}|{reset}  [{user}] {message}"

    # Track the last date we wrote to each file for daily rotation
    _last_write_dates: Dict[str, Any] = {}

    def _check_daily_rotation(self, log_path: str) -> None:
        """Clear log file if it's from a previous day."""
        try:
            if os.path.exists(log_path):
                file_mtime = datetime.fromtimestamp(os.path.getmtime(log_path))
                today = datetime.now().date()
                if file_mtime.date() < today:
                    # File is from a previous day, clear it
                    with open(log_path, 'w', encoding='utf-8') as f:
                        f.write("")
        except Exception:
            pass  # Silently fail to avoid log loops

    def _write_to_file(self, log_path: str, message: str) -> None:
        """Write a log entry directly to file. Clears file if from previous day."""
        ensure_log_directory()
        try:
            # Check if we need to rotate (clear) the file for a new day
            today = datetime.now().date()
            last_date = PipelineLogger._last_write_dates.get(log_path)
            if last_date != today:
                self._check_daily_rotation(log_path)
                PipelineLogger._last_write_dates[log_path] = today

            with open(log_path, 'a', encoding='utf-8') as f:
                f.write(f"{message}\n")
        except Exception:
            pass  # Silently fail to avoid log loops


    def info(self, message: str, context: Optional[Union[Request, str, Dict[str, Any]]] = None) -> None:
        """
        Log an info message.

        Args:
            message: Log message
            context: Request object, username, or dict with user info
        """
        user = self._get_user_identifier(context)
        console_msg = self._format_console(user, message, "INFO")
        file_msg = self._format_message(user, message, "INFO")

        # Use lock for thread-safe output
        with self._log_lock:
            print(console_msg)

            # Write to pipeline.log
            self._write_to_file(os.path.join(LOG_DIR, LOG_PIPELINE_FILE), file_msg)

            # Also write to python-service.log for Python-related pipelines
            if self.pipeline_name in [PIPELINES['PYTHON'], PIPELINES['MONGODB'],
                                       PIPELINES['EXTRACTION'], PIPELINES['SUMMARIZATION']]:
                self._write_to_file(os.path.join(LOG_DIR, LOG_SERVICE_FILE), file_msg)

    def error(self, message: str, error: Optional[Exception] = None,
              context: Optional[Union[Request, str, Dict[str, Any]]] = None) -> None:
        """
        Log an error message.

        Args:
            message: Error message
            error: Optional exception object
            context: Request object, username, or dict with user info
        """
        user = self._get_user_identifier(context)
        error_details = f" | {str(error)}" if error else ""
        full_message = f"{message}{error_details}"
        console_msg = self._format_console(user, full_message, "ERROR")
        file_msg = self._format_message(user, full_message, "ERROR")

        # Use lock for thread-safe output
        with self._log_lock:
            print(console_msg)

            # Write to both pipeline.log and error.log
            self._write_to_file(os.path.join(LOG_DIR, LOG_PIPELINE_FILE), file_msg)
            self._write_to_file(os.path.join(LOG_DIR, 'error.log'), file_msg)

            # Log stack trace if available
            if error and hasattr(error, '__traceback__'):
                import traceback
                stack = ''.join(traceback.format_tb(error.__traceback__))
                self._write_to_file(os.path.join(LOG_DIR, 'error.log'), f"  Stack: {stack}")

    def warn(self, message: str, context: Optional[Union[Request, str, Dict[str, Any]]] = None) -> None:
        """
        Log a warning message.

        Args:
            message: Warning message
            context: Request object, username, or dict with user info
        """
        user = self._get_user_identifier(context)
        console_msg = self._format_console(user, message, "WARN")
        file_msg = self._format_message(user, message, "WARN")

        # Use lock for thread-safe output
        with self._log_lock:
            print(console_msg)
            self._write_to_file(os.path.join(LOG_DIR, LOG_PIPELINE_FILE), file_msg)


# Pre-configured loggers for common pipelines
pipeline_loggers = {
    'document': PipelineLogger(PIPELINES['DOCUMENT']),
    'query': PipelineLogger(PIPELINES['QUERY']),
    'embedding': PipelineLogger(PIPELINES['EMBEDDING']),
    'roslyn': PipelineLogger(PIPELINES['ROSLYN']),
    'git': PipelineLogger(PIPELINES['GIT']),
    'sql': PipelineLogger(PIPELINES['SQL']),
    'llm': PipelineLogger(PIPELINES['LLM']),
    'auth': PipelineLogger(PIPELINES['AUTH']),
    'admin': PipelineLogger(PIPELINES['ADMIN']),
    'system': PipelineLogger(PIPELINES['SYSTEM']),
    'python': PipelineLogger(PIPELINES['PYTHON']),
    'extraction': PipelineLogger(PIPELINES['EXTRACTION']),
    'summarization': PipelineLogger(PIPELINES['SUMMARIZATION']),
    'mongodb': PipelineLogger(PIPELINES['MONGODB']),
}


def create_pipeline_logger(pipeline_name: str) -> PipelineLogger:
    """
    Create a logger for a specific pipeline.

    Args:
        pipeline_name: Name of the pipeline

    Returns:
        PipelineLogger instance
    """
    return PipelineLogger(pipeline_name)


# ============================================================================
# Data Classes for Log Entries
# ============================================================================

@dataclass
class LogEntry:
    """Represents a parsed log entry."""
    timestamp: str
    level: str
    source: str
    message: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "timestamp": self.timestamp,
            "level": self.level,
            "source": self.source,
            "message": self.message
        }


@dataclass
class LogResponse:
    """Response model for log API endpoints."""
    logs: List[Dict[str, Any]]
    total: int
    filtered: int

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "logs": self.logs,
            "total": self.total,
            "filtered": self.filtered
        }


# ============================================================================
# Logging Setup Functions
# ============================================================================

def ensure_log_directory() -> str:
    """
    Ensure the log directory exists.
    Creates the directory if it doesn't exist.

    Returns:
        Path to the log directory
    """
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR, exist_ok=True)
    return LOG_DIR


def get_log_file_path(log_type: Literal["service", "pipeline"]) -> str:
    """
    Get the full path to a log file.

    Args:
        log_type: Either "service" or "pipeline"

    Returns:
        Full path to the log file
    """
    ensure_log_directory()
    filename = LOG_SERVICE_FILE if log_type == "service" else LOG_PIPELINE_FILE
    return os.path.join(LOG_DIR, filename)


def setup_file_handler(log_type: Literal["service", "pipeline"]) -> TimedRotatingFileHandler:
    """
    Create a daily rotating file handler for a specific log type.
    Rotates at midnight and deletes old files (keeps 0 backups).

    Args:
        log_type: Either "service" or "pipeline"

    Returns:
        Configured TimedRotatingFileHandler
    """
    log_path = get_log_file_path(log_type)

    handler = TimedRotatingFileHandler(
        log_path,
        when=LOG_ROTATION_WHEN,
        interval=LOG_ROTATION_INTERVAL,
        backupCount=LOG_BACKUP_COUNT,
        encoding='utf-8'
    )

    formatter = logging.Formatter(LOG_FORMAT, datefmt=LOG_DATE_FORMAT)
    handler.setFormatter(formatter)

    return handler


def setup_logging() -> None:
    """
    Configure logging for the entire application.

    Sets up:
    - Service logger: General application logs (uvicorn, FastAPI, etc.)
    - Pipeline logger: RAG pipeline execution logs
    - Root logger: Capture all Python logs

    All logs are written to rotating log files and python-service.log.
    """
    ensure_log_directory()

    # Service logger (for general service operations)
    service_logger = logging.getLogger("service")
    service_logger.setLevel(logging.INFO)
    service_handler = setup_file_handler("service")
    service_logger.addHandler(service_handler)

    # Pipeline logger (for RAG pipeline operations)
    pipeline_logger = logging.getLogger("pipeline")
    pipeline_logger.setLevel(logging.INFO)
    pipeline_handler = setup_file_handler("pipeline")
    pipeline_logger.addHandler(pipeline_handler)

    # Also capture uvicorn logs to service log
    uvicorn_access = logging.getLogger("uvicorn.access")
    uvicorn_error = logging.getLogger("uvicorn.error")
    uvicorn_access.addHandler(service_handler)
    uvicorn_error.addHandler(service_handler)

    # Capture FastAPI logs
    fastapi_logger = logging.getLogger("fastapi")
    fastapi_logger.addHandler(service_handler)

    # Configure root logger to capture ALL Python logs to python-service.log
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    # Add service handler to root to capture everything
    root_logger.addHandler(service_handler)

    # Also add console handler for visibility
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)


def get_service_logger() -> logging.Logger:
    """Get the service logger instance."""
    return logging.getLogger("service")


def get_pipeline_logger() -> logging.Logger:
    """Get the pipeline logger instance."""
    return logging.getLogger("pipeline")


# ============================================================================
# Log Parsing Functions
# ============================================================================

# Regex pattern to parse log lines in format: timestamp|level|source|message
LOG_LINE_PATTERN = re.compile(
    r'^(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})\|(\w+)\|([^|]+)\|(.*)$'
)

# Alternative pattern for standard log format
ALT_LOG_PATTERN = re.compile(
    r'^(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}(?:,\d{3})?)\s+-\s+(\w+)\s+-\s+(.*)$'
)


def parse_log_line(line: str, default_source: str = "unknown") -> Optional[LogEntry]:
    """
    Parse a single log line into a LogEntry.

    Supports two formats:
    1. Custom format: timestamp|level|source|message
    2. Standard format: timestamp - level - message

    Args:
        line: Raw log line
        default_source: Source to use if not in log line

    Returns:
        LogEntry if successfully parsed, None otherwise
    """
    line = line.strip()
    if not line:
        return None

    # Try custom pipe-delimited format first
    match = LOG_LINE_PATTERN.match(line)
    if match:
        timestamp, level, source, message = match.groups()
        return LogEntry(
            timestamp=timestamp,
            level=level.upper(),
            source=source.strip(),
            message=message.strip()
        )

    # Try standard format
    alt_match = ALT_LOG_PATTERN.match(line)
    if alt_match:
        timestamp, level, message = alt_match.groups()
        # Convert timestamp format if needed
        timestamp = timestamp.replace(' ', 'T').replace(',', '.')
        return LogEntry(
            timestamp=timestamp,
            level=level.upper(),
            source=default_source,
            message=message.strip()
        )

    # If no pattern matches, treat entire line as message
    return LogEntry(
        timestamp=datetime.now().strftime(LOG_DATE_FORMAT),
        level="INFO",
        source=default_source,
        message=line
    )


def parse_timestamp(timestamp_str: str) -> Optional[datetime]:
    """
    Parse a timestamp string into a datetime object.

    Args:
        timestamp_str: ISO format timestamp string

    Returns:
        datetime object or None if parsing fails
    """
    formats = [
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%dT%H:%M:%S.%f",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M:%S,%f"
    ]

    for fmt in formats:
        try:
            return datetime.strptime(timestamp_str, fmt)
        except ValueError:
            continue

    return None


def read_log_file(
    log_path: str,
    limit: int = 100,
    level: Optional[str] = None,
    since: Optional[str] = None,
    source_name: str = "service"
) -> LogResponse:
    """
    Read and parse a log file with optional filtering.

    Args:
        log_path: Path to the log file
        limit: Maximum number of log entries to return (default 100)
        level: Filter by log level (INFO, WARNING, ERROR, DEBUG)
        since: ISO timestamp - only return logs after this time
        source_name: Default source name for log entries

    Returns:
        LogResponse with parsed and filtered log entries
    """
    if not os.path.exists(log_path):
        return LogResponse(logs=[], total=0, filtered=0)

    # Parse the 'since' timestamp if provided
    since_dt: Optional[datetime] = None
    if since:
        since_dt = parse_timestamp(since)

    # Normalize level filter
    level_filter: Optional[str] = None
    if level:
        level_filter = level.upper()

    all_entries: List[LogEntry] = []
    filtered_entries: List[LogEntry] = []

    try:
        with open(log_path, 'r', encoding='utf-8', errors='replace') as f:
            # Read all lines (for accurate total count)
            lines = f.readlines()
    except IOError as e:
        get_service_logger().error(f"Failed to read log file {log_path}: {e}")
        return LogResponse(logs=[], total=0, filtered=0)

    # Parse all lines
    for line in lines:
        entry = parse_log_line(line, default_source=source_name)
        if entry:
            all_entries.append(entry)

    total_count = len(all_entries)

    # Apply filters
    for entry in all_entries:
        # Filter by level
        if level_filter and entry.level != level_filter:
            continue

        # Filter by timestamp
        if since_dt:
            entry_dt = parse_timestamp(entry.timestamp)
            if entry_dt and entry_dt < since_dt:
                continue

        filtered_entries.append(entry)

    # Get the most recent entries (last N entries)
    # Reverse to show newest first, then limit
    filtered_entries.reverse()
    limited_entries = filtered_entries[:limit]

    return LogResponse(
        logs=[e.to_dict() for e in limited_entries],
        total=total_count,
        filtered=len(filtered_entries)
    )


def get_service_logs(
    limit: int = 100,
    level: Optional[str] = None,
    since: Optional[str] = None
) -> LogResponse:
    """
    Get service logs with optional filtering.

    Args:
        limit: Maximum number of log entries
        level: Filter by log level
        since: Filter by timestamp (ISO format)

    Returns:
        LogResponse with service logs
    """
    log_path = get_log_file_path("service")
    return read_log_file(log_path, limit, level, since, source_name="service")


def get_pipeline_logs(
    limit: int = 100,
    level: Optional[str] = None,
    since: Optional[str] = None
) -> LogResponse:
    """
    Get pipeline logs with optional filtering.

    Args:
        limit: Maximum number of log entries
        level: Filter by log level
        since: Filter by timestamp (ISO format)

    Returns:
        LogResponse with pipeline logs
    """
    log_path = get_log_file_path("pipeline")
    return read_log_file(log_path, limit, level, since, source_name="pipeline")


def get_all_logs(
    limit: int = 100,
    level: Optional[str] = None,
    since: Optional[str] = None
) -> LogResponse:
    """
    Get combined logs from all sources with optional filtering.

    Merges service and pipeline logs, sorts by timestamp (newest first).

    Args:
        limit: Maximum number of log entries
        level: Filter by log level
        since: Filter by timestamp (ISO format)

    Returns:
        LogResponse with combined logs
    """
    # Get logs from both sources with higher limit
    # (we'll merge and re-limit)
    service_response = get_service_logs(limit=limit * 2, level=level, since=since)
    pipeline_response = get_pipeline_logs(limit=limit * 2, level=level, since=since)

    # Combine all logs
    all_logs = service_response.logs + pipeline_response.logs

    # Sort by timestamp (newest first)
    all_logs.sort(key=lambda x: x.get("timestamp", ""), reverse=True)

    # Limit
    limited_logs = all_logs[:limit]

    return LogResponse(
        logs=limited_logs,
        total=service_response.total + pipeline_response.total,
        filtered=service_response.filtered + pipeline_response.filtered
    )


def clear_logs(log_type: Optional[Literal["service", "pipeline"]] = None) -> Dict[str, Any]:
    """
    Clear log files.

    Args:
        log_type: Specific log type to clear, or None for all

    Returns:
        Status dictionary
    """
    cleared = []

    types_to_clear = ["service", "pipeline"] if log_type is None else [log_type]

    for lt in types_to_clear:
        log_path = get_log_file_path(lt)
        if os.path.exists(log_path):
            try:
                # Clear by opening in write mode
                with open(log_path, 'w', encoding='utf-8') as f:
                    f.write("")
                cleared.append(lt)
            except IOError as e:
                get_service_logger().error(f"Failed to clear {lt} logs: {e}")

    return {
        "success": True,
        "cleared": cleared,
        "message": f"Cleared {len(cleared)} log file(s)"
    }


def get_log_stats() -> Dict[str, Any]:
    """
    Get statistics about log files.

    Returns:
        Dictionary with log file stats (size, line count, etc.)
    """
    stats = {}

    for log_type in ["service", "pipeline"]:
        log_path = get_log_file_path(log_type)

        if os.path.exists(log_path):
            file_stat = os.stat(log_path)

            # Count lines
            try:
                with open(log_path, 'r', encoding='utf-8', errors='replace') as f:
                    line_count = sum(1 for _ in f)
            except IOError:
                line_count = 0

            stats[log_type] = {
                "path": log_path,
                "size_bytes": file_stat.st_size,
                "size_human": _format_size(file_stat.st_size),
                "line_count": line_count,
                "modified": datetime.fromtimestamp(file_stat.st_mtime).isoformat()
            }
        else:
            stats[log_type] = {
                "path": log_path,
                "exists": False
            }

    return stats


def _format_size(size_bytes: int) -> str:
    """Format byte size to human readable string."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.2f} TB"


# Initialize logging when module is imported
# (but allow explicit setup as well)
_logging_initialized = False


def initialize_logging() -> None:
    """
    Initialize logging system.
    Should be called once at application startup.
    """
    global _logging_initialized
    if not _logging_initialized:
        setup_logging()
        _logging_initialized = True


# ============================================================================
# Convenience Functions for Single-Line Logging
# ============================================================================

def log_pipeline(
    pipeline: str,
    user: str,
    action: str,
    message: str = "",
    details: Optional[Dict[str, Any]] = None
) -> None:
    """
    Single-line pipeline logging function.

    Args:
        pipeline: Pipeline name key (e.g., 'SQL', 'DOCUMENT', 'GIT') or full name
        user: Username, IP address, or 'System'
        action: Action being performed (e.g., 'Query submitted', 'Upload started')
        message: Optional message or query text
        details: Optional dict with additional context

    Usage:
        log_pipeline("SQL", "chad.walker", "Query submitted", "How many tickets?")
        log_pipeline("SQL", "chad.walker", "Query completed", details={"tokens": 150})
    """
    # Normalize pipeline name
    pipeline_key = pipeline.upper()
    pipeline_name = PIPELINES.get(pipeline_key, pipeline)

    # Get or create logger
    logger_key = pipeline_key.lower()
    if logger_key in pipeline_loggers:
        logger = pipeline_loggers[logger_key]
    else:
        logger = PipelineLogger(pipeline_name)

    # Format message with action and details
    full_message = action
    if message:
        full_message += f": {message}"
    if details:
        import json
        full_message += f" | {json.dumps(details)}"

    logger.info(full_message, user)


def log_service(
    service: str,
    user: str,
    message: str,
    details: Optional[Dict[str, Any]] = None
) -> None:
    """
    Single-line service logging function.

    Args:
        service: Service name (e.g., 'MongoDB', 'Embedding', 'LLM')
        user: Username, IP address, or 'System'
        message: Log message
        details: Optional dict with additional context

    Usage:
        log_service("MongoDB", "System", "Connected to replica set")
        log_service("LLM", "chad.walker", "Tokens used", details={"tokens": 500})
    """
    logger = PipelineLogger(service)

    full_message = message
    if details:
        import json
        full_message += f" | {json.dumps(details)}"

    logger.info(full_message, user)


def log_error(
    pipeline: str,
    user: str,
    action: str,
    exception: Optional[Exception] = None,
    details: Optional[Dict[str, Any]] = None
) -> None:
    """
    Single-line error logging function with exception support.

    Args:
        pipeline: Pipeline name key (e.g., 'SQL', 'DOCUMENT') or full name
        user: Username, IP address, or 'System'
        action: Action that failed (e.g., 'Query failed', 'Upload failed')
        exception: Optional exception object
        details: Optional dict with additional context

    Usage:
        log_error("SQL", "chad.walker", "Query failed", exception=e)
        log_error("SQL", "chad.walker", "Query failed", details={"database": "CentralData"})
    """
    # Normalize pipeline name
    pipeline_key = pipeline.upper()
    pipeline_name = PIPELINES.get(pipeline_key, pipeline)

    # Get or create logger
    logger_key = pipeline_key.lower()
    if logger_key in pipeline_loggers:
        logger = pipeline_loggers[logger_key]
    else:
        logger = PipelineLogger(pipeline_name)

    # Format message with details
    full_message = action
    if details:
        import json
        full_message += f" | {json.dumps(details)}"

    logger.error(full_message, exception, user)


# Convenience exports for the new logging system
__all__ = [
    # New PipelineLogger system
    'PipelineLogger',
    'create_pipeline_logger',
    'pipeline_loggers',
    'PIPELINES',
    # Single-line logging functions
    'log_pipeline',
    'log_service',
    'log_error',
    # Legacy Python logging
    'initialize_logging',
    'get_service_logger',
    'get_pipeline_logger',
    # Log reading/management
    'get_service_logs',
    'get_pipeline_logs',
    'get_all_logs',
    'get_log_stats',
    'clear_logs',
]
