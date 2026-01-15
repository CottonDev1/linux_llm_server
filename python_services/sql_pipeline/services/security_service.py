"""
Security Service Module

This service provides security checks for SQL queries to prevent
injection attacks and unauthorized operations.
"""

from typing import Optional
from enum import Enum
import logging
import re
from core.log_utils import log_info
from datetime import datetime

logger = logging.getLogger(__name__)


class RiskLevel(str, Enum):
    """Risk level classification for SQL queries."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class SecurityCheckResult:
    """Result of a security check."""

    def __init__(
        self,
        passed: bool,
        risk_level: RiskLevel,
        issues: list[str],
        blocked: bool = False,
    ):
        self.passed = passed
        self.risk_level = risk_level
        self.issues = issues
        self.blocked = blocked


class SecurityService:
    """
    Service for SQL security validation.

    This service provides:
    - SQL injection detection
    - Dangerous operation detection (DROP, TRUNCATE, etc.)
    - Parameter validation
    - Query risk assessment

    Attributes:
        allow_ddl: Whether to allow DDL statements
        allow_dml_writes: Whether to allow INSERT/UPDATE/DELETE
        blocked_patterns: Regex patterns to block
    """

    def __init__(
        self,
        allow_ddl: bool = False,
        allow_dml_writes: bool = True,
        custom_blocked_patterns: Optional[list[str]] = None,
    ):
        """
        Initialize the security service.

        Args:
            allow_ddl: Whether to allow DDL statements (CREATE, ALTER, DROP)
            allow_dml_writes: Whether to allow write operations
            custom_blocked_patterns: Additional patterns to block
        """
        self.allow_ddl = allow_ddl
        self.allow_dml_writes = allow_dml_writes

        # Default blocked patterns
        # Note: r";\s*--" was removed as it's too broad - blocks legitimate SQL comments
        # The injection_patterns list has r"'\s*;\s*--" which catches actual injection
        self.blocked_patterns = [
            r"WAITFOR\s+DELAY",  # Time-based attacks
            r"xp_cmdshell",  # Command execution
            r"sp_executesql.*@",  # Dynamic SQL with parameters
            r"OPENROWSET",  # External data access
            r"OPENDATASOURCE",  # External data access
            r"BULK\s+INSERT",  # Bulk operations
        ]

        if custom_blocked_patterns:
            self.blocked_patterns.extend(custom_blocked_patterns)

        # SQL injection detection patterns
        self.injection_patterns = [
            r"'\s*OR\s+'?\d+'\s*=\s*'?\d+",  # ' OR '1'='1
            r"'\s*OR\s+1\s*=\s*1",  # ' OR 1=1
            r"'\s*;\s*--",  # '; --
            r"UNION\s+ALL\s+SELECT",  # UNION attack
            r"UNION\s+SELECT",  # UNION attack
            r"EXEC\s*\(",  # EXEC injection
            r";\s*DROP\s+",  # Stacked DROP
            r";\s*DELETE\s+",  # Stacked DELETE
            r";\s*UPDATE\s+",  # Stacked UPDATE
            r";\s*INSERT\s+",  # Stacked INSERT
        ]

        log_info("Security Service", "Initialized")

    def check_query(
        self,
        sql: str,
        allow_writes: Optional[bool] = None,
    ) -> SecurityCheckResult:
        """
        Perform security checks on a SQL query.

        Args:
            sql: The SQL query to check
            allow_writes: Override for write permission

        Returns:
            SecurityCheckResult with check status
        """
        issues = []
        blocked = False

        # Use override if provided, otherwise use instance setting
        writes_allowed = allow_writes if allow_writes is not None else self.allow_dml_writes

        # Check for SQL injection patterns
        injection_issues = self.detect_injection(sql)
        if injection_issues:
            issues.extend(injection_issues)
            blocked = True
            logger.warning(f"SQL injection detected: {injection_issues}")

        # Check for dangerous operations
        dangerous_ops = self.detect_dangerous_operations(sql)
        if dangerous_ops:
            issues.extend(dangerous_ops)

            # Check if DDL is blocked
            ddl_keywords = ['DROP', 'TRUNCATE', 'CREATE', 'ALTER']
            has_ddl = any(keyword in op for op in dangerous_ops for keyword in ddl_keywords)
            if has_ddl and not self.allow_ddl:
                blocked = True
                logger.warning(f"DDL operation blocked: {dangerous_ops}")

            # Check if writes are blocked
            write_keywords = ['DELETE', 'UPDATE', 'INSERT']
            has_writes = any(keyword in op for op in dangerous_ops for keyword in write_keywords)
            if has_writes and not writes_allowed:
                blocked = True
                logger.warning(f"Write operation blocked: {dangerous_ops}")

        # Assess overall risk level
        risk_level = self.assess_risk(sql)

        # Determine if check passed
        passed = not blocked and len(issues) == 0

        logger.info(
            f"Security check: passed={passed}, risk={risk_level}, "
            f"issues={len(issues)}, blocked={blocked}"
        )

        return SecurityCheckResult(
            passed=passed,
            risk_level=risk_level,
            issues=issues,
            blocked=blocked,
        )

    def detect_injection(
        self,
        sql: str,
    ) -> list[str]:
        """
        Detect potential SQL injection patterns.

        Args:
            sql: The SQL query to check

        Returns:
            List of detected injection patterns
        """
        issues = []

        # Check injection patterns
        for pattern in self.injection_patterns:
            if re.search(pattern, sql, re.IGNORECASE):
                issues.append(f"Potential SQL injection pattern detected: {pattern}")

        # Check blocked patterns
        for pattern in self.blocked_patterns:
            if re.search(pattern, sql, re.IGNORECASE):
                issues.append(f"Blocked pattern detected: {pattern}")

        return issues

    def detect_dangerous_operations(
        self,
        sql: str,
    ) -> list[str]:
        """
        Detect dangerous SQL operations.

        Args:
            sql: The SQL query to check

        Returns:
            List of detected dangerous operations
        """
        dangerous_ops = []

        # DDL operations
        ddl_patterns = {
            'DROP': r'\bDROP\s+(TABLE|DATABASE|SCHEMA|VIEW|INDEX|PROCEDURE|FUNCTION)',
            'TRUNCATE': r'\bTRUNCATE\s+TABLE',
            'CREATE': r'\bCREATE\s+(TABLE|DATABASE|SCHEMA|VIEW|INDEX|PROCEDURE|FUNCTION)',
            'ALTER': r'\bALTER\s+(TABLE|DATABASE|SCHEMA|VIEW|PROCEDURE|FUNCTION)',
        }

        for op_name, pattern in ddl_patterns.items():
            if re.search(pattern, sql, re.IGNORECASE):
                dangerous_ops.append(f"DDL operation detected: {op_name}")

        # Dangerous DML - DELETE without WHERE
        if re.search(r'\bDELETE\s+FROM\s+\w+\s*(?!WHERE)', sql, re.IGNORECASE):
            dangerous_ops.append("DELETE without WHERE clause detected")

        # Dangerous DML - UPDATE without WHERE
        if re.search(r'\bUPDATE\s+\w+\s+SET\s+.*?(?!WHERE)', sql, re.IGNORECASE):
            # More sophisticated check to avoid false positives
            if not re.search(r'\bWHERE\b', sql, re.IGNORECASE):
                dangerous_ops.append("UPDATE without WHERE clause detected")

        # System stored procedures
        system_procedures = {
            'xp_cmdshell': r'\bxp_cmdshell\b',
            'sp_executesql': r'\bsp_executesql\b',
            'sp_OACreate': r'\bsp_OACreate\b',
            'sp_OAMethod': r'\bsp_OAMethod\b',
            'sp_configure': r'\bsp_configure\b',
        }

        for proc_name, pattern in system_procedures.items():
            if re.search(pattern, sql, re.IGNORECASE):
                dangerous_ops.append(f"System procedure detected: {proc_name}")

        return dangerous_ops

    def assess_risk(
        self,
        sql: str,
    ) -> RiskLevel:
        """
        Assess the risk level of a SQL query.

        Args:
            sql: The SQL query to assess

        Returns:
            RiskLevel classification
        """
        # CRITICAL: Has injection patterns or blocked patterns
        injection_issues = self.detect_injection(sql)
        if injection_issues:
            return RiskLevel.CRITICAL

        # Check for dangerous operations
        dangerous_ops = self.detect_dangerous_operations(sql)

        # HIGH: Has DDL operations
        ddl_keywords = ['DROP', 'TRUNCATE', 'CREATE', 'ALTER']
        has_ddl = any(keyword in op for op in dangerous_ops for keyword in ddl_keywords)
        if has_ddl:
            return RiskLevel.HIGH

        # HIGH: System procedures
        if any('System procedure' in op for op in dangerous_ops):
            return RiskLevel.HIGH

        # HIGH: DELETE or UPDATE without WHERE
        if any('without WHERE' in op for op in dangerous_ops):
            return RiskLevel.HIGH

        # MEDIUM: Has DML writes (INSERT/UPDATE/DELETE)
        if re.search(r'\b(INSERT|UPDATE|DELETE)\b', sql, re.IGNORECASE):
            return RiskLevel.MEDIUM

        # LOW: Read-only SELECT
        if self.is_read_only(sql):
            return RiskLevel.LOW

        # Default to MEDIUM if we can't determine
        return RiskLevel.MEDIUM

    def sanitize_parameter(
        self,
        value: str,
        param_type: str = "string",
    ) -> str:
        """
        Sanitize a parameter value for safe SQL inclusion.

        Args:
            value: The value to sanitize
            param_type: Type of parameter (string, int, date, etc.)

        Returns:
            Sanitized value

        Raises:
            ValueError: If the value is invalid for the specified type
        """
        if value is None:
            return "NULL"

        # Convert to string if not already
        value_str = str(value)

        # Remove null bytes from all types
        value_str = value_str.replace('\x00', '')

        if param_type.lower() == "string":
            # Escape single quotes by doubling them (SQL standard)
            value_str = value_str.replace("'", "''")
            # Remove dangerous characters
            value_str = value_str.replace('\x00', '')
            value_str = value_str.replace('\\', '\\\\')
            return value_str

        elif param_type.lower() in ("int", "integer", "number"):
            # Validate is numeric
            try:
                # Try to convert to int
                int_val = int(value_str)
                return str(int_val)
            except ValueError:
                raise ValueError(f"Invalid integer value: {value_str}")

        elif param_type.lower() in ("float", "decimal", "numeric"):
            # Validate is numeric
            try:
                # Try to convert to float
                float_val = float(value_str)
                return str(float_val)
            except ValueError:
                raise ValueError(f"Invalid numeric value: {value_str}")

        elif param_type.lower() == "date":
            # Validate date format
            try:
                # Try common date formats
                date_formats = [
                    '%Y-%m-%d',
                    '%m/%d/%Y',
                    '%d/%m/%Y',
                    '%Y-%m-%d %H:%M:%S',
                    '%m/%d/%Y %H:%M:%S',
                ]

                parsed_date = None
                for fmt in date_formats:
                    try:
                        parsed_date = datetime.strptime(value_str, fmt)
                        break
                    except ValueError:
                        continue

                if parsed_date is None:
                    raise ValueError(f"Invalid date format: {value_str}")

                # Return in SQL Server format
                return parsed_date.strftime('%Y-%m-%d %H:%M:%S')

            except Exception as e:
                raise ValueError(f"Invalid date value: {value_str} - {str(e)}")

        elif param_type.lower() == "bool":
            # Convert to SQL boolean
            if value_str.lower() in ('true', '1', 'yes', 't', 'y'):
                return "1"
            elif value_str.lower() in ('false', '0', 'no', 'f', 'n'):
                return "0"
            else:
                raise ValueError(f"Invalid boolean value: {value_str}")

        else:
            # Default to string sanitization
            value_str = value_str.replace("'", "''")
            return value_str

    def is_read_only(
        self,
        sql: str,
    ) -> bool:
        """
        Check if a SQL query is read-only.

        Args:
            sql: The SQL query to check

        Returns:
            True if the query is read-only
        """
        # Normalize SQL - remove comments and extra whitespace
        sql_normalized = re.sub(r'--.*$', '', sql, flags=re.MULTILINE)  # Remove line comments
        sql_normalized = re.sub(r'/\*.*?\*/', '', sql_normalized, flags=re.DOTALL)  # Remove block comments
        sql_normalized = sql_normalized.strip()

        # List of write/DDL operations
        write_operations = [
            r'\bINSERT\b',
            r'\bUPDATE\b',
            r'\bDELETE\b',
            r'\bDROP\b',
            r'\bCREATE\b',
            r'\bALTER\b',
            r'\bTRUNCATE\b',
            r'\bMERGE\b',
            r'\bEXEC\b',
            r'\bEXECUTE\b',
        ]

        # Check for any write operations
        for operation in write_operations:
            if re.search(operation, sql_normalized, re.IGNORECASE):
                return False

        # Check if it starts with SELECT (common read-only case)
        if re.match(r'^\s*SELECT\b', sql_normalized, re.IGNORECASE):
            return True

        # Check for other read-only statements
        read_only_operations = [
            r'^\s*WITH\b.*\bSELECT\b',  # CTE
            r'^\s*DECLARE\b.*\bSELECT\b',  # Variable declaration with SELECT
        ]

        for operation in read_only_operations:
            if re.search(operation, sql_normalized, re.IGNORECASE | re.DOTALL):
                return True

        # If we can't determine, assume it's not read-only (safer)
        return False
