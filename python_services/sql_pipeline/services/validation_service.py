"""
Validation Service Module

This service handles SQL validation using execution testing and LLM-as-Judge.
"""

from typing import Optional, List, Dict, Any
import logging
import json
import re

from sql_pipeline.models.validation_models import ValidationResult, ExecutionResult
from sql_pipeline.models.query_models import SQLCredentials

logger = logging.getLogger(__name__)


class ValidationService:
    """
    Service for validating SQL queries using execution testing and LLM-as-Judge.

    This service provides:
    - Syntax validation (T-SQL keyword checking)
    - Auto-fix application via SyntaxFixer
    - Execution validation via ExecutionService
    - LLM-based quality scoring (LLM-as-Judge)
    - Auto-fix and revalidation with retry logic

    Attributes:
        execution_service: Service for SQL execution
        syntax_fixer: Service for auto-fixing common SQL issues
        llm_service: LLM service for LLM-as-Judge validation
    """

    def __init__(
        self,
        execution_service=None,
        syntax_fixer=None,
    ):
        """
        Initialize the validation service.

        Args:
            execution_service: ExecutionService instance for execution validation
            syntax_fixer: SyntaxFixer instance for auto-corrections
        """
        self.execution_service = execution_service
        self.syntax_fixer = syntax_fixer
        self.llm_service = None  # Lazy loaded

        logger.info("ValidationService initialized")

    async def _get_llm_service(self):
        """Lazy load LLM service."""
        if self.llm_service is None:
            from services.llm_service import get_llm_service
            self.llm_service = await get_llm_service()
        return self.llm_service

    async def validate(
        self,
        sql: str,
        question: str,
        database: str,
        credentials: Optional[SQLCredentials] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> ValidationResult:
        """
        Main validation entry point.

        This method orchestrates the complete validation process:
        1. Apply syntax fixes first
        2. Optionally execute query (if credentials provided)
        3. Optionally run LLM-as-Judge

        Args:
            sql: The SQL query to validate
            question: Original natural language question
            database: Target database name
            credentials: Optional credentials for execution validation
            options: Optional validation options (execute, llm_judge, schema_context)

        Returns:
            ValidationResult with is_valid, errors, warnings, fixes applied, score
        """
        options = options or {}
        execute = options.get("execute", False)
        use_llm_judge = options.get("llm_judge", False)
        schema_context = options.get("schema_context", "")

        errors = []
        warnings = []
        auto_fixes_applied = []
        execution_result = None
        llm_judge_score = None
        llm_judge_explanation = None

        # Step 1: Apply syntax fixes
        fixed_sql = sql
        if self.syntax_fixer:
            try:
                fixed_sql, fixes = await self.syntax_fixer.apply_all_fixes(sql, database)
                if fixes:
                    auto_fixes_applied = [f.description for f in fixes]
                    logger.info(f"Applied {len(fixes)} syntax fixes")
            except Exception as e:
                logger.warning(f"Syntax fixer failed: {e}", exc_info=True)
                # Continue with original SQL
                fixed_sql = sql

        # Step 2: Validate syntax
        syntax_errors = await self.validate_syntax(fixed_sql)
        if syntax_errors:
            errors.extend(syntax_errors)

        # Step 3: Optionally execute query
        if execute and credentials:
            try:
                execution_result = await self.validate_with_execution(
                    fixed_sql,
                    credentials,
                    timeout=options.get("timeout", 30)
                )
                if not execution_result.success:
                    errors.append(f"Execution failed: {execution_result.error}")
            except Exception as e:
                logger.error(f"Execution validation failed: {e}", exc_info=True)
                errors.append(f"Execution error: {str(e)}")

        # Step 4: Optionally run LLM-as-Judge
        if use_llm_judge and not errors:
            try:
                llm_judge_score, llm_judge_explanation = await self.validate_with_llm_judge(
                    fixed_sql,
                    question,
                    schema_context
                )
                if llm_judge_score < 0.5:
                    warnings.append(f"LLM judge scored low ({llm_judge_score:.2f}): {llm_judge_explanation}")
            except Exception as e:
                logger.warning(f"LLM judge validation failed: {e}", exc_info=True)
                warnings.append(f"LLM judge error: {str(e)}")

        # Determine if valid
        is_valid = len(errors) == 0

        return ValidationResult(
            is_valid=is_valid,
            errors=errors,
            warnings=warnings,
            auto_fixes_applied=auto_fixes_applied,
            execution_result=execution_result,
            llm_judge_score=llm_judge_score,
            llm_judge_explanation=llm_judge_explanation,
        )

    async def validate_syntax(self, sql: str) -> List[str]:
        """
        Check for obvious syntax errors.

        Validates:
        - T-SQL keywords and structure
        - Balanced parentheses and quotes
        - Common SQL mistakes

        Args:
            sql: The SQL query to validate

        Returns:
            List of syntax error messages (empty if valid)
        """
        errors = []

        if not sql or not sql.strip():
            errors.append("SQL query is empty")
            return errors

        sql_upper = sql.upper()

        # Check for basic SELECT/INSERT/UPDATE/DELETE structure
        has_select = "SELECT" in sql_upper
        has_insert = "INSERT" in sql_upper
        has_update = "UPDATE" in sql_upper
        has_delete = "DELETE" in sql_upper
        has_exec = "EXEC" in sql_upper or "EXECUTE" in sql_upper

        if not (has_select or has_insert or has_update or has_delete or has_exec):
            errors.append("SQL must contain SELECT, INSERT, UPDATE, DELETE, or EXEC")

        # Check for balanced parentheses
        if sql.count("(") != sql.count(")"):
            errors.append("Unbalanced parentheses in SQL")

        # Check for balanced single quotes
        # Count non-escaped single quotes
        quote_count = len(re.findall(r"(?<!\\)'", sql))
        if quote_count % 2 != 0:
            errors.append("Unbalanced single quotes in SQL")

        # Check for dangerous keywords (should be blocked by other layers, but double-check)
        dangerous_keywords = ["DROP TABLE", "DROP DATABASE", "TRUNCATE TABLE", "ALTER TABLE DROP"]
        for keyword in dangerous_keywords:
            if keyword in sql_upper:
                errors.append(f"Dangerous operation detected: {keyword}")

        # Check for common mistakes
        if has_select:
            # Check for SELECT without FROM (unless using literals/functions)
            if "FROM" not in sql_upper and not re.search(r"SELECT\s+\d+|SELECT\s+GETDATE\(\)|SELECT\s+@@", sql_upper):
                warnings_msg = "SELECT without FROM clause (may be intentional)"
                # This is a warning, not an error, but we'll add it as an error for now
                # In production, you might want a separate warnings list
                # For now, skip this check to avoid false positives

        return errors

    async def validate_with_execution(
        self,
        sql: str,
        credentials: SQLCredentials,
        timeout: int = 30,
    ) -> ExecutionResult:
        """
        Execute query with timeout to check for execution errors.

        Args:
            sql: The SQL query to execute
            credentials: Database credentials
            timeout: Query timeout in seconds

        Returns:
            ExecutionResult with success/failure status
        """
        if not self.execution_service:
            raise ValueError("ExecutionService not available for execution validation")

        try:
            # Execute with minimal results (1 row) for validation
            result = await self.execution_service.execute(
                sql=sql,
                credentials=credentials,
                max_results=1,
                timeout=timeout,
            )
            return result

        except Exception as e:
            logger.error(f"Execution validation error: {e}", exc_info=True)
            return ExecutionResult(
                success=False,
                error=str(e),
                row_count=0,
                columns=[],
                data=None,
                execution_time=0.0,
            )

    async def validate_with_llm_judge(
        self,
        sql: str,
        question: str,
        schema_context: str = "",
    ) -> tuple[float, str]:
        """
        Use LLM to evaluate if SQL answers the question correctly.

        This implements the LLM-as-Judge pattern where the LLM evaluates
        the quality and correctness of the generated SQL.

        Args:
            sql: The SQL query to evaluate
            question: Original natural language question
            schema_context: Relevant database schema context

        Returns:
            Tuple of (score 0.0-1.0, explanation)
        """
        llm_service = await self._get_llm_service()

        # Build the evaluation prompt
        prompt = f"""You are a SQL validation expert. Evaluate if this SQL query correctly answers the question.

Question: {question}

SQL Query:
```sql
{sql}
```

Database Schema (relevant tables):
{schema_context if schema_context else "Schema context not provided"}

Evaluate:
1. Does the SQL use correct table/column names?
2. Does the SQL logic match the question intent?
3. Are there any obvious errors or problems?

Return your evaluation as JSON:
{{"score": 0.0-1.0, "valid": true/false, "explanation": "detailed explanation"}}

Only return the JSON, nothing else."""

        system_prompt = "You are a SQL validation expert. Return only valid JSON."

        try:
            # Use the general model for SQL evaluation
            result = await llm_service.generate(
                prompt=prompt,
                system=system_prompt,
                temperature=0.0,
                use_cache=False,
            )

            if not result.success:
                logger.error(f"LLM judge generation failed: {result.error}")
                return 0.5, f"LLM judge error: {result.error}"

            # Parse JSON response
            response_text = result.response.strip()

            # Try to extract JSON from markdown code blocks if present
            json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", response_text, re.DOTALL)
            if json_match:
                response_text = json_match.group(1)

            # Try to find JSON object in response
            json_match = re.search(r"\{[^}]+\}", response_text, re.DOTALL)
            if json_match:
                response_text = json_match.group(0)

            evaluation = json.loads(response_text)

            score = float(evaluation.get("score", 0.5))
            explanation = evaluation.get("explanation", "No explanation provided")

            # Clamp score to 0.0-1.0
            score = max(0.0, min(1.0, score))

            logger.info(f"LLM judge score: {score:.2f}")
            return score, explanation

        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse LLM judge response: {e}", exc_info=True)
            logger.warning(f"Response was: {result.response}")
            return 0.5, f"Failed to parse LLM response: {result.response[:200]}"

        except Exception as e:
            logger.error(f"LLM judge error: {e}", exc_info=True)
            return 0.5, f"LLM judge error: {str(e)}"

    async def auto_fix_and_revalidate(
        self,
        sql: str,
        database: str,
        credentials: Optional[SQLCredentials] = None,
        max_attempts: int = 3,
    ) -> ValidationResult:
        """
        Try to fix SQL errors automatically and revalidate.

        This method attempts to:
        1. Apply syntax fixes
        2. Execute and check for errors
        3. If errors found, attempt additional fixes
        4. Revalidate after fixes
        5. Return best result after max attempts

        Args:
            sql: The SQL query to fix
            database: Target database name
            credentials: Optional credentials for execution testing
            max_attempts: Maximum number of fix attempts

        Returns:
            Best ValidationResult after all attempts
        """
        best_result = None
        current_sql = sql

        for attempt in range(max_attempts):
            logger.info(f"Auto-fix attempt {attempt + 1}/{max_attempts}")

            # Apply syntax fixes
            if self.syntax_fixer:
                try:
                    current_sql, fixes = await self.syntax_fixer.apply_all_fixes(current_sql, database)
                    if fixes:
                        logger.info(f"Applied {len(fixes)} fixes on attempt {attempt + 1}")
                except Exception as e:
                    logger.warning(f"Syntax fixer failed on attempt {attempt + 1}: {e}")

            # Validate
            result = await self.validate(
                sql=current_sql,
                question="",  # Not needed for auto-fix validation
                database=database,
                credentials=credentials,
                options={"execute": credentials is not None},
            )

            # Track best result (fewest errors)
            if best_result is None or len(result.errors) < len(best_result.errors):
                best_result = result

            # If valid, we're done
            if result.is_valid:
                logger.info(f"Auto-fix succeeded on attempt {attempt + 1}")
                return result

            # If we have execution errors, we might be able to extract more info
            if result.execution_result and result.execution_result.error:
                error_msg = result.execution_result.error
                logger.info(f"Execution error on attempt {attempt + 1}: {error_msg}")

                # Could add more sophisticated error-based fixing here
                # For example, detecting "Invalid column name 'X'" and trying alternatives

        logger.warning(f"Auto-fix failed after {max_attempts} attempts")
        return best_result

    async def close(self):
        """Clean up resources."""
        logger.info("ValidationService closed")
