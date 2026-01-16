"""
Syntax Fixer Module

This service handles fixing common SQL syntax issues and formatting problems
that may occur in LLM-generated SQL.
"""

from typing import Optional, List, Tuple
from dataclasses import dataclass
import logging
import re
from core.log_utils import log_info

from sql_pipeline.services.rules_service import RulesService

logger = logging.getLogger(__name__)


@dataclass
class FixApplied:
    """Represents a fix that was applied to SQL."""
    rule_id: str  # or "common-xxx" for common fixes
    pattern: str
    original: str  # matched text
    replacement: str
    description: str


class SyntaxFixer:
    """
    Service for fixing SQL syntax issues.

    This service provides:
    - Rule-based auto-fixes from RulesService
    - Common T-SQL syntax error fixes
    - Identifier quoting (backticks to square brackets)
    - MySQL to T-SQL conversions (LIMIT to TOP, IFNULL to ISNULL, etc.)

    Attributes:
        rules_service: RulesService instance for loading auto-fix rules
    """

    def __init__(self, rules_service: Optional[RulesService] = None):
        """
        Initialize the syntax fixer.

        Args:
            rules_service: Optional RulesService instance (for dependency injection)
        """
        self.rules_service = rules_service
        log_info("Syntax Fixer", "Initialized")

    async def apply_all_fixes(self, sql: str, database: str) -> Tuple[str, List[FixApplied]]:
        """
        Apply all available fixes to SQL query.

        This is the main entry point for syntax fixing. It applies:
        1. Rule-based auto-fixes from RulesService (global and database-specific)
        2. Common T-SQL fixes (LIMIT to TOP, backticks, etc.)

        Args:
            sql: The SQL query to fix
            database: Target database name

        Returns:
            Tuple of (fixed_sql, list of FixApplied objects)
        """
        all_fixes = []
        current_sql = sql

        # Apply rule-based fixes first
        current_sql, rule_fixes = await self.apply_rule_fixes(current_sql, database)
        all_fixes.extend(rule_fixes)

        # Apply common T-SQL fixes
        current_sql, common_fixes = self.apply_common_fixes(current_sql)
        all_fixes.extend(common_fixes)

        if all_fixes:
            logger.info(f"Applied {len(all_fixes)} total fixes to SQL")
            for fix in all_fixes:
                logger.debug(f"  - {fix.description}: '{fix.original}' -> '{fix.replacement}'")

        return current_sql, all_fixes

    async def apply_rule_fixes(self, sql: str, database: str) -> Tuple[str, List[FixApplied]]:
        """
        Apply auto-fix patterns from RulesService.

        This loads auto-fix rules from MongoDB and applies their regex patterns.

        Args:
            sql: The SQL query to fix
            database: Target database name

        Returns:
            Tuple of (fixed_sql, list of FixApplied objects)
        """
        if not self.rules_service:
            logger.debug("No rules_service available, skipping rule-based fixes")
            return sql, []

        # Get auto-fix patterns from rules
        auto_fixes = await self.rules_service.get_auto_fixes(database)

        if not auto_fixes:
            logger.debug(f"No auto-fix rules found for database '{database}'")
            return sql, []

        applied_fixes = []
        current_sql = sql

        for auto_fix in auto_fixes:
            # Check if fix applies to specific table
            if auto_fix.applies_to_table:
                # Only apply if table is mentioned in SQL
                if auto_fix.applies_to_table.lower() not in current_sql.lower():
                    continue

            # Try to apply the fix
            pattern = auto_fix.pattern
            replacement = auto_fix.replacement

            try:
                # Find all matches before replacement (for tracking)
                matches = list(re.finditer(pattern, current_sql))

                if matches:
                    # Apply replacement
                    fixed_sql = re.sub(pattern, replacement, current_sql)

                    # Track each match
                    for match in matches:
                        applied_fixes.append(FixApplied(
                            rule_id=f"rule-autofix-{len(applied_fixes)}",
                            pattern=pattern,
                            original=match.group(0),
                            replacement=replacement,
                            description=f"Auto-fix pattern: {pattern}"
                        ))

                    current_sql = fixed_sql
                    logger.info(f"Applied auto-fix pattern: {pattern} -> {replacement}")

            except re.error as e:
                logger.warning(f"Invalid regex pattern in auto-fix: {pattern} - {e}")
                continue

        return current_sql, applied_fixes

    def apply_common_fixes(self, sql: str) -> Tuple[str, List[FixApplied]]:
        """
        Apply hardcoded common T-SQL syntax fixes.

        These are fixes for common MySQL/PostgreSQL syntax that needs conversion
        to T-SQL, plus general LLM mistakes.

        Fixes applied:
        - dbo.dbo. -> dbo.
        - LIMIT N -> TOP N (moved to after SELECT)
        - IFNULL -> ISNULL
        - Backtick identifiers -> square brackets
        - NOW() -> GETDATE()
        - MySQL-style comments (#) -> T-SQL comments (--)

        Args:
            sql: The SQL query to fix

        Returns:
            Tuple of (fixed_sql, list of FixApplied objects)
        """
        applied_fixes = []
        current_sql = sql

        # Fix 1: Replace dbo.dbo. with dbo.
        pattern = r'\bdbo\.dbo\.'
        matches = list(re.finditer(pattern, current_sql, re.IGNORECASE))
        if matches:
            for match in matches:
                applied_fixes.append(FixApplied(
                    rule_id="common-dbo-double",
                    pattern=pattern,
                    original=match.group(0),
                    replacement="dbo.",
                    description="Fixed double dbo.dbo. prefix"
                ))
            current_sql = re.sub(pattern, 'dbo.', current_sql, flags=re.IGNORECASE)
            logger.debug("Fixed dbo.dbo. -> dbo.")

        # Fix 2: LIMIT to TOP conversion
        current_sql, limit_fixes = self.fix_limit_to_top(current_sql)
        applied_fixes.extend(limit_fixes)

        # Fix 3: IFNULL -> ISNULL
        pattern = r'\bIFNULL\b'
        matches = list(re.finditer(pattern, current_sql, re.IGNORECASE))
        if matches:
            for match in matches:
                applied_fixes.append(FixApplied(
                    rule_id="common-ifnull",
                    pattern=pattern,
                    original=match.group(0),
                    replacement="ISNULL",
                    description="Converted IFNULL to ISNULL (T-SQL)"
                ))
            current_sql = re.sub(pattern, 'ISNULL', current_sql, flags=re.IGNORECASE)
            logger.debug("Fixed IFNULL -> ISNULL")

        # Fix 4: Backtick identifiers -> square brackets
        pattern = r'`([^`]+)`'
        matches = list(re.finditer(pattern, current_sql))
        if matches:
            for match in matches:
                applied_fixes.append(FixApplied(
                    rule_id="common-backticks",
                    pattern=pattern,
                    original=match.group(0),
                    replacement=f"[{match.group(1)}]",
                    description="Converted backtick identifier to square brackets"
                ))
            current_sql = re.sub(pattern, r'[\1]', current_sql)
            logger.debug("Fixed backticks -> square brackets")

        # Fix 5: NOW() -> GETDATE()
        pattern = r'\bNOW\s*\(\s*\)'
        matches = list(re.finditer(pattern, current_sql, re.IGNORECASE))
        if matches:
            for match in matches:
                applied_fixes.append(FixApplied(
                    rule_id="common-now",
                    pattern=pattern,
                    original=match.group(0),
                    replacement="GETDATE()",
                    description="Converted NOW() to GETDATE() (T-SQL)"
                ))
            current_sql = re.sub(pattern, 'GETDATE()', current_sql, flags=re.IGNORECASE)
            logger.debug("Fixed NOW() -> GETDATE()")

        # Fix 6: MySQL-style comments (#) -> T-SQL comments (--)
        pattern = r'#(.*)$'
        matches = list(re.finditer(pattern, current_sql, re.MULTILINE))
        if matches:
            for match in matches:
                applied_fixes.append(FixApplied(
                    rule_id="common-mysql-comment",
                    pattern=pattern,
                    original=match.group(0),
                    replacement=f"--{match.group(1)}",
                    description="Converted MySQL comment (#) to T-SQL (--)"
                ))
            current_sql = re.sub(pattern, r'--\1', current_sql, flags=re.MULTILINE)
            logger.debug("Fixed MySQL comments -> T-SQL comments")


        # Fix 7: NULLS FIRST/LAST -> Remove (not valid in T-SQL)
        pattern = r"\s+NULLS\s+(FIRST|LAST)"
        matches = list(re.finditer(pattern, current_sql, re.IGNORECASE))
        if matches:
            for match in matches:
                applied_fixes.append(FixApplied(
                    rule_id="common-nulls-ordering",
                    pattern=pattern,
                    original=match.group(0),
                    replacement="",
                    description="Removed NULLS FIRST/LAST (not valid in T-SQL)"
                ))
            current_sql = re.sub(pattern, "", current_sql, flags=re.IGNORECASE)
            logger.debug("Removed NULLS FIRST/LAST")

        # Fix 8: ILIKE -> LIKE (T-SQL uses LIKE with CI collation)
        pattern = r"\bILIKE\b"
        matches = list(re.finditer(pattern, current_sql, re.IGNORECASE))
        if matches:
            for match in matches:
                applied_fixes.append(FixApplied(
                    rule_id="common-ilike",
                    pattern=pattern,
                    original=match.group(0),
                    replacement="LIKE",
                    description="Converted ILIKE to LIKE (T-SQL)"
                ))
            current_sql = re.sub(pattern, "LIKE", current_sql, flags=re.IGNORECASE)
            logger.debug("Fixed ILIKE -> LIKE")

        # Fix 9: CURRENT_DATE -> CAST(GETDATE() AS DATE)
        pattern = r"\bCURRENT_DATE\b"
        matches = list(re.finditer(pattern, current_sql, re.IGNORECASE))
        if matches:
            for match in matches:
                applied_fixes.append(FixApplied(
                    rule_id="common-current-date",
                    pattern=pattern,
                    original=match.group(0),
                    replacement="CAST(GETDATE() AS DATE)",
                    description="Converted CURRENT_DATE to CAST(GETDATE() AS DATE) (T-SQL)"
                ))
            current_sql = re.sub(pattern, "CAST(GETDATE() AS DATE)", current_sql, flags=re.IGNORECASE)
            logger.debug("Fixed CURRENT_DATE -> CAST(GETDATE() AS DATE)")

        # Fix 10: PostgreSQL interval syntax -> DATEADD (T-SQL)
        # Pattern: date - interval 'N unit' or date + interval 'N unit'
        interval_pattern = r"(\w+|\))\s*(-|\+)\s*interval\s+'(\d+)\s+(year|month|day|week|hour|minute|second)s?'"
        interval_matches = list(re.finditer(interval_pattern, current_sql, re.IGNORECASE))
        if interval_matches:
            for match in interval_matches:
                expr = match.group(1)
                operator = match.group(2)
                amount = match.group(3)
                unit = match.group(4).upper()

                # For subtraction, negate the amount
                if operator == '-':
                    amount = f"-{amount}"

                replacement = f"DATEADD({unit}, {amount}, {expr})"
                applied_fixes.append(FixApplied(
                    rule_id="common-interval",
                    pattern=interval_pattern,
                    original=match.group(0),
                    replacement=replacement,
                    description=f"Converted interval to DATEADD({unit}, {amount}, ...)"
                ))
                current_sql = current_sql[:match.start()] + replacement + current_sql[match.end():]
            logger.debug("Fixed interval -> DATEADD")

        # Fix 11: :: type casting -> CAST (PostgreSQL to T-SQL)
        cast_pattern = r"(\w+)::(\w+)"
        cast_matches = list(re.finditer(cast_pattern, current_sql))
        if cast_matches:
            for match in reversed(cast_matches):  # Process in reverse to maintain positions
                expr = match.group(1)
                type_name = match.group(2)
                # Map PostgreSQL types to T-SQL types
                type_mapping = {
                    'text': 'VARCHAR(MAX)',
                    'integer': 'INT',
                    'bigint': 'BIGINT',
                    'date': 'DATE',
                    'timestamp': 'DATETIME2',
                    'boolean': 'BIT',
                    'float': 'FLOAT',
                    'numeric': 'DECIMAL',
                }
                tsql_type = type_mapping.get(type_name.lower(), type_name.upper())
                replacement = f"CAST({expr} AS {tsql_type})"
                applied_fixes.append(FixApplied(
                    rule_id="common-cast",
                    pattern=cast_pattern,
                    original=match.group(0),
                    replacement=replacement,
                    description=f"Converted :: cast to CAST({expr} AS {tsql_type})"
                ))
                current_sql = current_sql[:match.start()] + replacement + current_sql[match.end():]
            logger.debug("Fixed :: -> CAST")

        return current_sql, applied_fixes

    def fix_limit_to_top(self, sql: str) -> Tuple[str, List[FixApplied]]:
        """
        Convert MySQL LIMIT clause to T-SQL TOP clause.

        Transforms:
            SELECT * FROM table LIMIT 10
        To:
            SELECT TOP 10 * FROM table

        Handles:
        - Simple LIMIT N
        - LIMIT N OFFSET M (converts to OFFSET-FETCH if present)
        - Preserves subqueries

        Args:
            sql: The SQL query to fix

        Returns:
            Tuple of (fixed_sql, list of FixApplied objects)
        """
        applied_fixes = []

        # Pattern to match LIMIT at end of query (not in subquery)
        # This is simplified - a full implementation would need proper SQL parsing
        pattern = r'\bLIMIT\s+(\d+)(?:\s+OFFSET\s+(\d+))?\s*$'
        match = re.search(pattern, sql, re.IGNORECASE)

        if match:
            limit_value = match.group(1)
            offset_value = match.group(2)

            # Find the SELECT keyword to insert TOP after it
            # Look for the first SELECT that's not in a subquery (simplified approach)
            select_pattern = r'\bSELECT\b'
            select_match = re.search(select_pattern, sql, re.IGNORECASE)

            if select_match:
                original_text = match.group(0)

                if offset_value:
                    # If OFFSET is present, use OFFSET-FETCH syntax
                    # Remove LIMIT clause
                    sql_without_limit = sql[:match.start()] + sql[match.end():]

                    # Add OFFSET-FETCH at the end
                    replacement_text = f" OFFSET {offset_value} ROWS FETCH NEXT {limit_value} ROWS ONLY"
                    fixed_sql = sql_without_limit.rstrip() + replacement_text

                    applied_fixes.append(FixApplied(
                        rule_id="common-limit-offset",
                        pattern=pattern,
                        original=original_text.strip(),
                        replacement=replacement_text.strip(),
                        description=f"Converted LIMIT {limit_value} OFFSET {offset_value} to OFFSET-FETCH"
                    ))
                else:
                    # Simple LIMIT -> TOP conversion
                    # Remove LIMIT clause
                    sql_without_limit = sql[:match.start()] + sql[match.end():]

                    # Insert TOP after SELECT
                    top_clause = f"TOP {limit_value} "
                    fixed_sql = sql_without_limit[:select_match.end()] + " " + top_clause + sql_without_limit[select_match.end():]

                    applied_fixes.append(FixApplied(
                        rule_id="common-limit-top",
                        pattern=pattern,
                        original=original_text.strip(),
                        replacement=f"TOP {limit_value}",
                        description=f"Converted LIMIT {limit_value} to TOP {limit_value}"
                    ))

                logger.debug(f"Converted LIMIT to TOP: {original_text.strip()}")
                return fixed_sql, applied_fixes

        return sql, applied_fixes
