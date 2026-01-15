"""
Execution Service Module

This service handles safe SQL query execution against databases.
"""

from typing import Optional, Any
import logging
import time
import subprocess
from core.log_utils import log_info
import json
import sys
import os
import pymssql

from sql_pipeline.models.query_models import SQLCredentials
from sql_pipeline.models.validation_models import ExecutionResult

logger = logging.getLogger(__name__)

# Path to Windows Python for domain authentication
# When running from Windows (via PowerShell), use Windows path
# When running from WSL, use WSL path
import platform
import os

def _get_windows_python_path():
    """Get the correct path to Windows Python based on environment."""
    if platform.system() == "Windows":
        # Running on Windows - use Windows path
        # Get the directory where this script is located
        script_dir = os.path.dirname(os.path.abspath(__file__))
        # Navigate up to python_services, then to venv
        base_dir = os.path.dirname(os.path.dirname(script_dir))  # python_services
        return os.path.join(base_dir, "venv", "Scripts", "python.exe")
    else:
        # Running from WSL - use WSL path
        return "/mnt/c/Projects/llm_website/python_services/venv/Scripts/python.exe"

WINDOWS_PYTHON_PATH = _get_windows_python_path()


class ExecutionService:
    """
    Service for executing SQL queries safely.

    This service provides:
    - Safe query execution with timeouts
    - Result formatting and limiting
    - Connection pooling
    - Transaction management

    Attributes:
        default_timeout: Default query timeout in seconds
        max_results: Maximum rows to return
        connection_pool: Dictionary of active connections
    """

    def __init__(
        self,
        default_timeout: int = 30,
        max_results: int = 1000,
    ):
        """
        Initialize the execution service.

        Args:
            default_timeout: Default query timeout in seconds
            max_results: Maximum rows to return by default
        """
        self.default_timeout = default_timeout
        self.max_results = max_results
        self._connection_pool: dict[str, Any] = {}

        log_info("Execution Service", "Initialized")

    def _needs_windows_python(self, credentials: SQLCredentials) -> bool:
        """
        Check if connection requires Windows Python for domain authentication.

        This method always returns False as Windows authentication is not supported.
        """
        return False

    def _execute_via_windows_python(
        self,
        server: str,
        database: str,
        user: str,
        password: str,
        query: str = "SELECT 1",
        timeout: int = 30
    ) -> tuple[bool, Optional[str], Optional[list]]:
        """
        Execute a query using Windows Python's pymssql.

        This is used for Windows domain authentication which doesn't work
        from WSL's Python due to domain trust issues.

        Returns:
            Tuple of (success, error_message, results)
        """
        # Python script to execute via Windows Python
        script = f'''
import pymssql
import json
import sys

try:
    conn = pymssql.connect(
        server={repr(server)},
        database={repr(database)},
        user={repr(user)},
        password={repr(password)},
        port='1433',
        login_timeout={timeout}
    )
    cursor = conn.cursor(as_dict=True)
    cursor.execute({repr(query)})

    if cursor.description:
        results = cursor.fetchall()
        columns = [col[0] for col in cursor.description]
        # Convert results to serializable format
        serializable_results = []
        for row in results:
            serializable_row = {{}}
            for col in columns:
                val = row[col]
                if hasattr(val, 'isoformat'):
                    val = val.isoformat()
                elif isinstance(val, bytes):
                    val = val.decode('utf-8', errors='replace')
                serializable_row[col] = val
            serializable_results.append(serializable_row)
        print(json.dumps({{"success": True, "columns": columns, "results": serializable_results, "row_count": len(results)}}))
    else:
        print(json.dumps({{"success": True, "columns": [], "results": [], "row_count": cursor.rowcount}}))

    conn.close()
except Exception as e:
    print(json.dumps({{"success": False, "error": str(e)}}))
    sys.exit(1)
'''

        try:
            logger.debug(f"Executing query via Windows Python: {query[:100]}...")
            result = subprocess.run(
                [WINDOWS_PYTHON_PATH, "-c", script],
                capture_output=True,
                text=True,
                timeout=timeout + 10
            )

            logger.debug(f"Windows Python returncode: {result.returncode}")
            logger.debug(f"Windows Python stdout length: {len(result.stdout) if result.stdout else 0}")
            logger.debug(f"Windows Python stderr: {result.stderr[:200] if result.stderr else 'None'}")

            if result.returncode == 0:
                data = json.loads(result.stdout.strip())
                logger.info(f"Windows Python execution result: success={data.get('success')}, row_count={data.get('row_count')}, columns={data.get('columns')}")
                if data.get("success"):
                    results = data.get("results", [])
                    logger.debug(f"Returning {len(results)} results from Windows Python")
                    return (True, None, results)
                else:
                    return (False, data.get("error", "Unknown error"), None)
            else:
                error = result.stderr or result.stdout or "Windows Python execution failed"
                logger.error(f"Windows Python execution failed: {error}")
                return (False, error, None)

        except subprocess.TimeoutExpired:
            return (False, f"Query timed out after {timeout} seconds", None)
        except json.JSONDecodeError as e:
            return (False, f"Failed to parse Windows Python output: {e}", None)
        except Exception as e:
            return (False, f"Windows Python execution failed: {e}", None)

    async def execute(
        self,
        sql: str,
        credentials: SQLCredentials,
        max_results: Optional[int] = None,
        timeout: Optional[int] = None,
    ) -> ExecutionResult:
        """
        Execute a SQL query and return results.

        Args:
            sql: The SQL query to execute
            credentials: Database credentials
            max_results: Maximum rows to return
            timeout: Query timeout in seconds

        Returns:
            ExecutionResult with query results or error
        """
        max_results = max_results or self.max_results
        timeout = timeout or self.default_timeout

        start_time = time.time()

        try:
            # Check if we need Windows Python for domain authentication
            if self._needs_windows_python(credentials):
                logger.info(f"Using Windows Python for domain auth query execution on {credentials.server}")
                domain = getattr(credentials, 'domain', '')
                user = f"{domain}\\{credentials.username}" if domain else credentials.username
                password = credentials.password.get_secret_value() if credentials.password else ""

                # Add TOP clause if not present and max_results is set
                query_sql = sql
                if max_results and "TOP" not in sql.upper():
                    # Try to insert TOP after SELECT
                    if sql.strip().upper().startswith("SELECT"):
                        query_sql = sql.replace("SELECT", f"SELECT TOP {max_results}", 1)

                success, error_msg, results = self._execute_via_windows_python(
                    server=credentials.server,
                    database=credentials.database,
                    user=user,
                    password=password,
                    query=query_sql,
                    timeout=timeout
                )

                execution_time = time.time() - start_time
                logger.info(f"Windows Python execution completed: success={success}, results_count={len(results) if results else 0}, error={error_msg}")

                if success:
                    columns = list(results[0].keys()) if results else []
                    logger.info(f"Returning ExecutionResult with {len(results) if results else 0} rows, columns={columns}")
                    return ExecutionResult(
                        success=True,
                        row_count=len(results),
                        columns=columns,
                        data=results,
                        execution_time=execution_time,
                    )
                else:
                    return ExecutionResult(
                        success=False,
                        row_count=0,
                        columns=[],
                        data=None,
                        error=error_msg,
                        execution_time=execution_time,
                    )

            # Standard execution (SQL auth or SSPI)
            conn = await self.get_connection(credentials)
            cursor = conn.cursor()

            # Set query timeout
            cursor.execute(f"SET LOCK_TIMEOUT {timeout * 1000}")

            # Execute query
            cursor.execute(sql)

            # Get column names
            columns = [desc[0] for desc in cursor.description] if cursor.description else []

            # Fetch rows up to max_results
            rows = cursor.fetchmany(max_results)

            # Convert rows to list of dictionaries with JSON-serializable values
            data = []
            for row in rows:
                row_dict = {}
                for i, col in enumerate(columns):
                    val = row[i]
                    # Convert datetime objects to ISO format strings for JSON serialization
                    if hasattr(val, 'isoformat'):
                        val = val.isoformat()
                    elif isinstance(val, bytes):
                        val = val.decode('utf-8', errors='replace')
                    row_dict[col] = val
                data.append(row_dict)

            execution_time = time.time() - start_time

            cursor.close()

            return ExecutionResult(
                success=True,
                row_count=len(data),
                columns=columns,
                data=data,
                execution_time=execution_time,
            )

        except pymssql.Error as e:
            execution_time = time.time() - start_time
            error_msg = str(e)
            logger.error(f"SQL execution error: {error_msg}", exc_info=True)

            return ExecutionResult(
                success=False,
                row_count=0,
                columns=[],
                data=None,
                error=error_msg,
                execution_time=execution_time,
            )

        except Exception as e:
            execution_time = time.time() - start_time
            error_msg = f"Unexpected error: {str(e)}"
            logger.error(error_msg, exc_info=True)

            return ExecutionResult(
                success=False,
                row_count=0,
                columns=[],
                data=None,
                error=error_msg,
                execution_time=execution_time,
            )

    async def execute_scalar(
        self,
        sql: str,
        credentials: SQLCredentials,
        timeout: Optional[int] = None,
    ) -> Any:
        """
        Execute a query and return a single scalar value.

        Args:
            sql: The SQL query to execute
            credentials: Database credentials
            timeout: Query timeout in seconds

        Returns:
            Single value result
        """
        timeout = timeout or self.default_timeout

        try:
            conn = await self.get_connection(credentials)
            cursor = conn.cursor()

            # Set query timeout
            cursor.execute(f"SET LOCK_TIMEOUT {timeout * 1000}")

            # Execute query
            cursor.execute(sql)

            # Fetch first row
            row = cursor.fetchone()

            cursor.close()

            # Return first column of first row
            return row[0] if row else None

        except pymssql.Error as e:
            error_msg = str(e)
            logger.error(f"SQL scalar execution error: {error_msg}", exc_info=True)
            raise

        except Exception as e:
            error_msg = f"Unexpected error during scalar execution: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise

    async def execute_non_query(
        self,
        sql: str,
        credentials: SQLCredentials,
        timeout: Optional[int] = None,
    ) -> int:
        """
        Execute a non-query SQL statement (INSERT, UPDATE, DELETE).

        Args:
            sql: The SQL statement to execute
            credentials: Database credentials
            timeout: Query timeout in seconds

        Returns:
            Number of rows affected
        """
        timeout = timeout or self.default_timeout

        try:
            conn = await self.get_connection(credentials)
            cursor = conn.cursor()

            # Set query timeout
            cursor.execute(f"SET LOCK_TIMEOUT {timeout * 1000}")

            # Execute statement
            cursor.execute(sql)

            # Get rows affected
            rows_affected = cursor.rowcount

            # Commit the transaction
            conn.commit()

            cursor.close()

            return rows_affected

        except pymssql.Error as e:
            error_msg = str(e)
            logger.error(f"SQL non-query execution error: {error_msg}", exc_info=True)
            # Rollback on error
            try:
                conn.rollback()
            except:
                pass
            raise

        except Exception as e:
            error_msg = f"Unexpected error during non-query execution: {str(e)}"
            logger.error(error_msg, exc_info=True)
            # Rollback on error
            try:
                conn.rollback()
            except:
                pass
            raise

    async def test_connection(
        self,
        credentials: SQLCredentials,
        raise_on_error: bool = False,
    ) -> tuple[bool, Optional[str]]:
        """
        Test database connection.

        Args:
            credentials: Database credentials to test
            raise_on_error: If True, raise exception on failure

        Returns:
            Tuple of (success, error_message)
        """
        try:
            # Check if we need Windows Python for domain authentication
            if self._needs_windows_python(credentials):
                logger.info(f"Using Windows Python for domain auth connection test to {credentials.server}")
                domain = getattr(credentials, 'domain', '')
                user = f"{domain}\\{credentials.username}" if domain else credentials.username
                password = credentials.password.get_secret_value() if credentials.password else ""

                success, error_msg, _ = self._execute_via_windows_python(
                    server=credentials.server,
                    database=credentials.database,
                    user=user,
                    password=password,
                    query="SELECT 1",
                    timeout=10
                )

                if not success and raise_on_error:
                    raise Exception(error_msg)
                return (success, error_msg)

            # Standard connection test (SQL auth or SSPI)
            conn = await self.get_connection(credentials)
            cursor = conn.cursor()

            # Execute simple test query
            cursor.execute("SELECT 1")
            result = cursor.fetchone()

            cursor.close()

            success = result is not None and result[0] == 1
            return (success, None if success else "Test query returned unexpected result")

        except pymssql.Error as e:
            error_msg = str(e)
            logger.error(f"Connection test failed: {error_msg}", exc_info=True)
            if raise_on_error:
                raise
            return (False, error_msg)

        except Exception as e:
            error_msg = f"Unexpected error: {str(e)}"
            logger.error(f"Connection test failed: {error_msg}", exc_info=True)
            if raise_on_error:
                raise
            return (False, error_msg)

    async def get_connection(
        self,
        credentials: SQLCredentials,
    ) -> Any:
        """
        Get or create a database connection.

        Args:
            credentials: Database credentials

        Returns:
            Database connection object
        """
        # Create connection key from server, database, and qualified username
        qualified_user = credentials.get_qualified_username()
        conn_key = f"{credentials.server}|{credentials.database}|{qualified_user}"

        # Check if connection already exists in pool
        if conn_key in self._connection_pool:
            conn = self._connection_pool[conn_key]
            # Test if connection is still alive
            try:
                cursor = conn.cursor()
                cursor.execute("SELECT 1")
                cursor.close()
                return conn
            except:
                # Connection dead, remove from pool
                logger.warning(f"Removing dead connection from pool: {conn_key}")
                try:
                    conn.close()
                except:
                    pass
                del self._connection_pool[conn_key]

        # Create new connection
        try:
            # Handle SecretStr password
            password = credentials.password.get_secret_value()

            # Get qualified username (includes domain if provided, e.g., EWR\EWRUser)
            username = credentials.get_qualified_username()

            # Log full connection parameters for debugging
            logger.info(f"=== CONNECTION PARAMETERS ===")
            logger.info(f"  Server: {credentials.server}")
            logger.info(f"  Database: {credentials.database}")
            logger.info(f"  Domain: {credentials.domain}")
            logger.info(f"  Raw Username: {credentials.username}")
            logger.info(f"  Qualified Username: {username}")
            logger.info(f"  Password: {password}")
            logger.info(f"=============================")

            # SQL Server authentication with optional domain prefix
            logger.info(f"Creating connection to {credentials.server}/{credentials.database} as {username}")
            conn = pymssql.connect(
                server=credentials.server,
                database=credentials.database,
                user=username,
                password=password,
            )

            # Add to connection pool
            self._connection_pool[conn_key] = conn

            logger.info(f"Connection established: {conn_key}")

            return conn

        except pymssql.Error as e:
            error_msg = f"Failed to connect to {credentials.server}/{credentials.database}: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise

        except Exception as e:
            error_msg = f"Unexpected error creating connection: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise

    def _format_results(
        self,
        rows: list,
        columns: list[str],
        max_results: int,
    ) -> dict:
        """
        Format query results for return.

        Args:
            rows: Raw query results
            columns: Column names
            max_results: Maximum rows to include

        Returns:
            Formatted result dictionary
        """
        # Limit rows to max_results
        limited_rows = rows[:max_results]

        # Convert rows to list of dictionaries with JSON-serializable values
        data = []
        for row in limited_rows:
            row_dict = {}
            for i, col in enumerate(columns):
                val = row[i]
                # Convert datetime objects to ISO format strings for JSON serialization
                if hasattr(val, 'isoformat'):
                    val = val.isoformat()
                elif isinstance(val, bytes):
                    val = val.decode('utf-8', errors='replace')
                row_dict[col] = val
            data.append(row_dict)

        return {
            "row_count": len(data),
            "columns": columns,
            "data": data,
        }

    async def close(self):
        """Close all database connections."""
        for conn in self._connection_pool.values():
            try:
                conn.close()
            except Exception as e:
                logger.warning(f"Error closing connection: {e}")
        self._connection_pool.clear()
        logger.info("ExecutionService connections closed")
