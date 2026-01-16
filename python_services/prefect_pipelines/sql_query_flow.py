"""
Prefect SQL Query Pipeline

Processes natural language questions through:
1. Security Validation - Check input for security issues
2. Cache Check - Look for cached responses in agent_learning
3. Rule Matching - Find applicable SQL rules
4. Schema Retrieval - Get relevant database schema context
5. Query Generation - Generate SQL using LLM
6. Syntax Fixing - Apply rule-based syntax fixes
7. Column Validation - Validate column names against schema
8. Query Execution - Execute query if enabled
9. Result Formatting - Format results for response
10. Learning Storage - Store successful queries for learning

Features:
- Full Prefect tracking with artifacts
- User tracking through all tasks
- Timing metrics at each step
- Built-in retries for resilience
- Fallback to direct pipeline if Prefect fails
"""

import asyncio
import time
import os
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime

from prefect import flow, task, get_run_logger
from prefect.artifacts import create_markdown_artifact
from prefect.variables import Variable


# =============================================================================
# Prefect Variables Support
# =============================================================================

async def get_prefect_variable(name: str, default: Any = None) -> Any:
    """
    Get a Prefect Variable value with fallback to default.

    Args:
        name: Variable name in Prefect
        default: Default value if variable not found

    Returns:
        Variable value or default
    """
    try:
        value = await Variable.get(name, default=default)
        return value
    except Exception:
        return default


def get_prefect_variable_sync(name: str, default: Any = None) -> Any:
    """
    Synchronous wrapper for getting Prefect Variables.

    Args:
        name: Variable name in Prefect
        default: Default value if variable not found

    Returns:
        Variable value or default
    """
    try:
        import asyncio
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If we're already in an async context, create a new thread
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(asyncio.run, Variable.get(name, default=default))
                    return future.result(timeout=5)
            else:
                return asyncio.run(Variable.get(name, default=default))
        except RuntimeError:
            return asyncio.run(Variable.get(name, default=default))
    except Exception:
        return default


# Prefect Variable names for SQL pipeline configuration
SQL_PIPELINE_VARIABLES = {
    # Connection settings
    "server": "sql_server",
    "database": "sql_database",
    "username": "sql_username",
    "password": "sql_password",
    "domain": "sql_domain",
    "auth_type": "sql_auth_type",  # "windows" or "sql"
    # Pipeline settings
    "max_tokens": "default_max_tokens",
    "max_results": "max_results",
    "query_timeout": "query_timeout_ms",
}


async def load_sql_credentials_from_variables() -> Dict[str, Any]:
    """
    Load SQL credentials from Prefect Variables.

    Handles auth_type logic:
    - If auth_type is "windows": username becomes "domain\\username"
    - If auth_type is "sql": username is used as-is (no domain)

    Returns:
        Dict with server, database, username, password, domain, auth_type
    """
    server = await get_prefect_variable("sql_server", "NCSQLTEST")
    database = await get_prefect_variable("sql_database", "EWRCentral")
    username = await get_prefect_variable("sql_username", "")
    password = await get_prefect_variable("sql_password", "")
    domain = await get_prefect_variable("sql_domain", "")
    auth_type = await get_prefect_variable("sql_auth_type", "windows")

    # Build the effective username based on auth_type
    if auth_type.lower() == "windows" and domain:
        effective_username = f"{domain}\\{username}"
    else:
        # SQL auth - no domain prefix
        effective_username = username

    return {
        "server": server,
        "database": database,
        "username": effective_username,
        "password": password,
        "domain": domain if auth_type.lower() == "windows" else None,
        "auth_type": auth_type,
        "raw_username": username,  # Original username without domain
    }


def load_sql_credentials_from_variables_sync() -> Dict[str, Any]:
    """
    Synchronous version of load_sql_credentials_from_variables.

    Returns:
        Dict with server, database, username, password, domain, auth_type
    """
    try:
        return asyncio.run(load_sql_credentials_from_variables())
    except RuntimeError:
        # Already in async context, use thread pool
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(asyncio.run, load_sql_credentials_from_variables())
            return future.result(timeout=10)


async def load_pipeline_settings_from_variables() -> Dict[str, Any]:
    """
    Load pipeline settings from Prefect Variables.

    Returns:
        Dict with max_tokens, max_results, timeout settings
    """
    return {
        "max_tokens": int(await get_prefect_variable("default_max_tokens", 512)),
        "max_results": int(await get_prefect_variable("max_results", 50)),
        "query_timeout_ms": int(await get_prefect_variable("query_timeout_ms", 60000)),
        "default_temperature": float(await get_prefect_variable("default_temperature", 0.3)),
        "similarity_threshold": float(await get_prefect_variable("similarity_threshold", 0.5)),
    }


# =============================================================================
# Data Classes for Task Results
# =============================================================================

@dataclass
class SecurityCheckResult:
    """Result from security validation task"""
    question: str
    is_safe: bool = True
    risk_level: str = "low"
    issues: List[str] = field(default_factory=list)
    blocked: bool = False
    duration_ms: float = 0.0


@dataclass
class CacheCheckResult:
    """Result from cache check task"""
    question: str
    database: str
    cache_hit: bool = False
    cached_sql: Optional[str] = None
    duration_ms: float = 0.0


@dataclass
class RuleMatchResult:
    """Result from rule matching task"""
    question: str
    database: str
    matched_rules: List[Dict[str, Any]] = field(default_factory=list)
    exact_match: Optional[Dict[str, Any]] = None
    is_exact_match: bool = False
    duration_ms: float = 0.0


@dataclass
class SchemaRetrievalResult:
    """Result from schema retrieval task"""
    database: str
    question: str
    schema_text: str = ""
    tables: List[str] = field(default_factory=list)
    table_count: int = 0
    duration_ms: float = 0.0
    success: bool = True


@dataclass
class QueryGenerationResult:
    """Result from query generation task"""
    question: str
    database: str
    generated_sql: str = ""
    token_usage: Dict[str, int] = field(default_factory=dict)
    duration_ms: float = 0.0
    success: bool = True
    error: str = ""


@dataclass
class SyntaxFixResult:
    """Result from syntax fixing task"""
    original_sql: str
    fixed_sql: str = ""
    fixes_applied: List[str] = field(default_factory=list)
    fix_count: int = 0
    duration_ms: float = 0.0


@dataclass
class ColumnValidationResult:
    """Result from column validation task"""
    sql: str
    database: str
    is_valid: bool = True
    invalid_columns: List[str] = field(default_factory=list)
    suggestions: Dict[str, str] = field(default_factory=dict)
    tables_checked: List[str] = field(default_factory=list)
    duration_ms: float = 0.0


@dataclass
class QueryExecutionResult:
    """Result from query execution task"""
    sql: str
    success: bool = False
    data: List[Dict[str, Any]] = field(default_factory=list)
    row_count: int = 0
    column_names: List[str] = field(default_factory=list)
    execution_time_ms: float = 0.0
    error: str = ""
    duration_ms: float = 0.0


@dataclass
class ResultFormattingResult:
    """Result from result formatting task"""
    formatted_response: Dict[str, Any] = field(default_factory=dict)
    duration_ms: float = 0.0


# =============================================================================
# Prefect Tasks
# =============================================================================

@task(
    name="security_validation",
    description="Validate input question for security issues",
    retries=1,
    retry_delay_seconds=2,
    tags=["sql", "security", "validation"]
)
async def security_validation_task(
    question: str,
    user_id: str
) -> SecurityCheckResult:
    """
    Validate the input question for security issues.

    Args:
        question: Natural language question
        user_id: User ID for tracking

    Returns:
        SecurityCheckResult with validation details
    """
    logger = get_run_logger()
    start_time = time.time()

    logger.info(f"[{user_id}] Security validation for question: {question[:100]}...")

    result = SecurityCheckResult(question=question)

    try:
        from sql_pipeline.services.security_service import SecurityService

        security_service = SecurityService()
        check = security_service.check_query(question)

        result.is_safe = not check.blocked
        result.risk_level = check.risk_level.value
        result.issues = check.issues
        result.blocked = check.blocked

        logger.info(f"[{user_id}] Security check: safe={result.is_safe}, risk={result.risk_level}")

    except Exception as e:
        error_msg = f"Security validation error: {str(e)}"
        logger.error(f"[{user_id}] {error_msg}")
        result.issues.append(error_msg)
        result.blocked = True
        result.is_safe = False

    result.duration_ms = (time.time() - start_time) * 1000

    # Create artifact
    await create_markdown_artifact(
        key=f"security-check-{user_id[:8]}",
        markdown=f"""
## Security Validation
- **User**: {user_id}
- **Safe**: {result.is_safe}
- **Risk Level**: {result.risk_level}
- **Blocked**: {result.blocked}
- **Duration**: {result.duration_ms:.1f}ms
{"### Issues" + chr(10) + chr(10).join(f"- {i}" for i in result.issues) if result.issues else ""}
        """,
        description=f"Security validation for user {user_id}"
    )

    return result


@task(
    name="cache_check",
    description="Check for cached SQL responses in agent_learning",
    retries=1,
    retry_delay_seconds=2,
    tags=["sql", "cache"]
)
async def cache_check_task(
    question: str,
    database: str,
    user_id: str,
    use_cache: bool = True
) -> CacheCheckResult:
    """
    Check agent_learning collection for cached SQL responses.

    Args:
        question: Natural language question
        database: Target database
        user_id: User ID for tracking
        use_cache: Whether to use cache

    Returns:
        CacheCheckResult with cache hit details
    """
    logger = get_run_logger()
    start_time = time.time()

    logger.info(f"[{user_id}] Cache check for: {question[:50]}...")

    result = CacheCheckResult(question=question, database=database)

    if not use_cache:
        logger.info(f"[{user_id}] Cache disabled, skipping check")
        result.duration_ms = (time.time() - start_time) * 1000
        return result

    try:
        from mongodb import MongoDBService

        mongo_service = MongoDBService.get_instance()
        if not mongo_service.is_initialized:
            await mongo_service.initialize()

        collection = mongo_service.db["agent_learning"]

        # Normalize question for matching
        question_normalized = question.lower().strip()

        # Query for similar questions with successful results
        cached = await collection.find_one({
            "database": database,
            "question_normalized": question_normalized,
            "success": True
        })

        if cached and cached.get("sql"):
            result.cache_hit = True
            result.cached_sql = cached["sql"]
            logger.info(f"[{user_id}] Cache HIT: Found cached SQL")
        else:
            logger.info(f"[{user_id}] Cache MISS")

    except Exception as e:
        logger.warning(f"[{user_id}] Cache check error: {e}")

    result.duration_ms = (time.time() - start_time) * 1000
    return result


@task(
    name="rule_matching",
    description="Find applicable SQL rules for the question",
    retries=2,
    retry_delay_seconds=5,
    tags=["sql", "rules"]
)
async def rule_matching_task(
    question: str,
    database: str,
    user_id: str
) -> RuleMatchResult:
    """
    Find matching SQL rules using cascading strategy.

    Args:
        question: Natural language question
        database: Target database
        user_id: User ID for tracking

    Returns:
        RuleMatchResult with matched rules
    """
    logger = get_run_logger()
    start_time = time.time()

    logger.info(f"[{user_id}] Rule matching for database: {database}")

    result = RuleMatchResult(question=question, database=database)

    try:
        from sql_pipeline.services.rules_service import RulesService

        rules_service = await RulesService.get_instance()

        # Check for exact match first
        exact_match = await rules_service.find_exact_match(question, database)
        if exact_match and exact_match.example:
            result.is_exact_match = True
            result.exact_match = {
                "rule_id": exact_match.rule_id,
                "description": exact_match.description,
                "sql": exact_match.example.sql
            }
            logger.info(f"[{user_id}] EXACT MATCH found: {exact_match.rule_id}")
        else:
            # Try similarity match
            similar_rules = await rules_service.find_similar_rules(question, database, threshold=0.8)
            if similar_rules:
                result.matched_rules = [
                    {"rule_id": r.rule_id, "description": r.description, "score": score}
                    for r, score in similar_rules
                ]
                logger.info(f"[{user_id}] Found {len(similar_rules)} similar rules")
            else:
                # Fall back to keyword matches
                keyword_rules = await rules_service.find_keyword_matches(question, database)
                if keyword_rules:
                    result.matched_rules = [
                        {"rule_id": r.rule_id, "description": r.description}
                        for r in keyword_rules
                    ]
                    logger.info(f"[{user_id}] Found {len(keyword_rules)} keyword-matched rules")
                else:
                    logger.info(f"[{user_id}] No matching rules found")

    except Exception as e:
        logger.warning(f"[{user_id}] Rule matching error: {e}")

    result.duration_ms = (time.time() - start_time) * 1000

    # Create artifact
    rule_summary = ""
    if result.is_exact_match:
        rule_summary = f"**Exact Match**: {result.exact_match['rule_id']}"
    elif result.matched_rules:
        rule_summary = "**Matched Rules**:\n" + "\n".join(
            f"- {r['rule_id']}: {r['description']}" for r in result.matched_rules[:5]
        )
    else:
        rule_summary = "No matching rules found"

    await create_markdown_artifact(
        key=f"rule-match-{user_id[:8]}",
        markdown=f"""
## Rule Matching
- **User**: {user_id}
- **Database**: {database}
- **Exact Match**: {result.is_exact_match}
- **Rules Found**: {len(result.matched_rules)}
- **Duration**: {result.duration_ms:.1f}ms

{rule_summary}
        """,
        description=f"Rule matching for user {user_id}"
    )

    return result


@task(
    name="schema_retrieval",
    description="Get relevant database schema context",
    retries=2,
    retry_delay_seconds=5,
    tags=["sql", "schema"]
)
async def schema_retrieval_task(
    question: str,
    database: str,
    user_id: str,
    matched_rules: List[Dict[str, Any]],
    include_schema: bool = True,
    max_tables: int = 4
) -> SchemaRetrievalResult:
    """
    Get relevant schema context for the question.

    Args:
        question: Natural language question
        database: Target database
        user_id: User ID for tracking
        matched_rules: Rules that matched (may contain required tables)
        include_schema: Whether to include schema
        max_tables: Maximum tables to include

    Returns:
        SchemaRetrievalResult with schema text
    """
    logger = get_run_logger()
    start_time = time.time()

    logger.info(f"[{user_id}] Schema retrieval for database: {database}")

    result = SchemaRetrievalResult(database=database, question=question)

    if not include_schema:
        logger.info(f"[{user_id}] Schema inclusion disabled")
        result.duration_ms = (time.time() - start_time) * 1000
        return result

    try:
        from sql_pipeline.services.schema_service import SchemaService

        schema_service = await SchemaService.get_instance()

        # Extract required tables from matched rules
        required_tables = set()
        for rule in matched_rules:
            if "trigger_tables" in rule:
                required_tables.update(rule["trigger_tables"])

        schema_info = await schema_service.get_relevant_schema(
            database=database,
            question=question,
            max_tables=max_tables,
            required_tables=list(required_tables) if required_tables else None
        )

        result.schema_text = schema_service.format_schema_for_prompt(schema_info.tables)
        result.tables = [t.full_name for t in schema_info.tables]
        result.table_count = len(schema_info.tables)
        result.success = True

        logger.info(f"[{user_id}] Loaded schema with {result.table_count} tables")

    except Exception as e:
        error_msg = f"Schema retrieval error: {str(e)}"
        logger.error(f"[{user_id}] {error_msg}")
        result.success = False

    result.duration_ms = (time.time() - start_time) * 1000

    # Create artifact
    await create_markdown_artifact(
        key=f"schema-retrieval-{user_id[:8]}",
        markdown=f"""
## Schema Retrieval
- **User**: {user_id}
- **Database**: {database}
- **Tables Found**: {result.table_count}
- **Duration**: {result.duration_ms:.1f}ms

### Tables
{chr(10).join(f"- {t}" for t in result.tables) if result.tables else "No tables found"}
        """,
        description=f"Schema retrieval for user {user_id}"
    )

    return result


@task(
    name="query_generation",
    description="Generate SQL from natural language using LLM",
    retries=2,
    retry_delay_seconds=30,
    tags=["sql", "llm", "generation"]
)
async def query_generation_task(
    question: str,
    database: str,
    user_id: str,
    schema_text: str,
    matched_rules: List[Dict[str, Any]],
    max_tokens: int = 512
) -> QueryGenerationResult:
    """
    Generate SQL using LLM with schema and rules context.

    Args:
        question: Natural language question
        database: Target database
        user_id: User ID for tracking
        schema_text: Formatted schema context
        matched_rules: Matched rules for guidance
        max_tokens: Maximum tokens for response

    Returns:
        QueryGenerationResult with generated SQL
    """
    logger = get_run_logger()
    start_time = time.time()

    logger.info(f"[{user_id}] Generating SQL for: {question[:50]}...")

    result = QueryGenerationResult(question=question, database=database)

    try:
        from services.llm_service import get_llm_service

        llm_service = await get_llm_service()

        # Build prompt
        prompt_parts = [
            "You are an expert SQL Server query generator. Generate a T-SQL query for the following question.",
            f"\nDATABASE: {database}",
        ]

        # Add schema if available
        if schema_text:
            prompt_parts.append("\nDATABASE SCHEMA (CRITICAL - USE ONLY THESE EXACT COLUMN NAMES):")
            prompt_parts.append(schema_text)
            prompt_parts.append("\nIMPORTANT: You MUST use ONLY the column names shown above.")

        # Add rules if available
        if matched_rules:
            prompt_parts.append("\nRULES AND GUIDELINES:")
            for rule in matched_rules[:10]:
                prompt_parts.append(f"- {rule.get('description', '')}")

        prompt_parts.append(f"\nQUESTION: {question}")
        prompt_parts.append("\nGenerate only the raw SQL query. No markdown, no explanations.")
        prompt_parts.append("\nSQL:")

        prompt = "\n".join(prompt_parts)

        # Call LLM
        llm_result = await llm_service.generate(
            prompt=prompt,
            system="You are a Microsoft SQL Server T-SQL expert. Output only raw T-SQL queries without markdown or backticks.",
            use_sql_model=True,
            temperature=0.0,
            max_tokens=max_tokens,
            use_cache=True
        )

        if not llm_result.success:
            result.error = llm_result.error or "LLM generation failed"
            result.success = False
            logger.error(f"[{user_id}] LLM error: {result.error}")
        else:
            # Extract SQL from response
            sql = _extract_sql_from_response(llm_result.response)
            result.generated_sql = sql
            result.token_usage = llm_result.token_usage if hasattr(llm_result, 'token_usage') else {}
            result.success = True
            logger.info(f"[{user_id}] Generated SQL: {sql[:100]}...")

    except Exception as e:
        result.error = str(e)
        result.success = False
        logger.error(f"[{user_id}] Generation error: {e}")

    result.duration_ms = (time.time() - start_time) * 1000

    # Create artifact
    await create_markdown_artifact(
        key=f"query-generation-{user_id[:8]}",
        markdown=f"""
## Query Generation
- **User**: {user_id}
- **Database**: {database}
- **Success**: {result.success}
- **Duration**: {result.duration_ms:.1f}ms
- **Tokens**: {result.token_usage}

### Generated SQL
```sql
{result.generated_sql or result.error}
```
        """,
        description=f"Query generation for user {user_id}"
    )

    return result


@task(
    name="syntax_fixing",
    description="Apply rule-based syntax fixes to generated SQL",
    retries=1,
    retry_delay_seconds=2,
    tags=["sql", "validation", "fixing"]
)
async def syntax_fixing_task(
    sql: str,
    database: str,
    user_id: str
) -> SyntaxFixResult:
    """
    Apply rule-based syntax fixes to the generated SQL.

    Args:
        sql: Generated SQL query
        database: Target database
        user_id: User ID for tracking

    Returns:
        SyntaxFixResult with fixed SQL and applied fixes
    """
    logger = get_run_logger()
    start_time = time.time()

    logger.info(f"[{user_id}] Applying syntax fixes")

    result = SyntaxFixResult(original_sql=sql)

    try:
        from sql_pipeline.services.syntax_fixer import SyntaxFixer
        from sql_pipeline.services.rules_service import RulesService

        rules_service = await RulesService.get_instance()
        syntax_fixer = SyntaxFixer(rules_service=rules_service)

        fixed_sql, fixes = await syntax_fixer.apply_all_fixes(sql, database)

        result.fixed_sql = fixed_sql
        result.fixes_applied = [f.description for f in fixes]
        result.fix_count = len(fixes)

        if fixes:
            logger.info(f"[{user_id}] Applied {len(fixes)} syntax fixes")
        else:
            logger.info(f"[{user_id}] No syntax fixes needed")

    except Exception as e:
        logger.warning(f"[{user_id}] Syntax fixing error: {e}")
        result.fixed_sql = sql  # Return original if fixing fails

    result.duration_ms = (time.time() - start_time) * 1000

    return result


@task(
    name="column_validation",
    description="Validate column names against database schema",
    retries=1,
    retry_delay_seconds=2,
    tags=["sql", "validation"]
)
async def column_validation_task(
    sql: str,
    database: str,
    user_id: str
) -> ColumnValidationResult:
    """
    Validate that column names in SQL exist in the schema.

    Args:
        sql: SQL query to validate
        database: Target database
        user_id: User ID for tracking

    Returns:
        ColumnValidationResult with validation details
    """
    logger = get_run_logger()
    start_time = time.time()

    logger.info(f"[{user_id}] Validating column names")

    result = ColumnValidationResult(sql=sql, database=database)

    try:
        from sql_pipeline.services.schema_service import SchemaService

        schema_service = await SchemaService.get_instance()
        validation = await schema_service.validate_columns(sql, database)

        result.is_valid = validation["valid"]
        result.invalid_columns = validation.get("invalid_columns", [])
        result.suggestions = validation.get("suggestions", {})
        result.tables_checked = validation.get("tables_checked", [])

        if result.is_valid:
            logger.info(f"[{user_id}] Column validation passed")
        else:
            logger.warning(f"[{user_id}] Invalid columns: {result.invalid_columns}")

    except Exception as e:
        logger.warning(f"[{user_id}] Column validation error: {e}")
        result.is_valid = True  # Assume valid on error to avoid blocking

    result.duration_ms = (time.time() - start_time) * 1000

    return result


@task(
    name="security_sql_validation",
    description="Validate generated SQL for security issues",
    retries=1,
    retry_delay_seconds=2,
    tags=["sql", "security"]
)
async def security_sql_validation_task(
    sql: str,
    user_id: str
) -> SecurityCheckResult:
    """
    Validate the generated SQL for security issues.

    Args:
        sql: Generated SQL query
        user_id: User ID for tracking

    Returns:
        SecurityCheckResult with validation details
    """
    logger = get_run_logger()
    start_time = time.time()

    logger.info(f"[{user_id}] SQL security validation")

    result = SecurityCheckResult(question=sql)

    try:
        from sql_pipeline.services.security_service import SecurityService

        security_service = SecurityService()
        check = security_service.check_query(sql)

        result.is_safe = not check.blocked
        result.risk_level = check.risk_level.value
        result.issues = check.issues
        result.blocked = check.blocked

        if result.blocked:
            logger.warning(f"[{user_id}] SQL security check BLOCKED: {result.issues}")
        else:
            logger.info(f"[{user_id}] SQL security check passed, risk={result.risk_level}")

    except Exception as e:
        logger.error(f"[{user_id}] SQL security validation error: {e}")
        result.issues.append(str(e))

    result.duration_ms = (time.time() - start_time) * 1000

    return result


@task(
    name="query_execution",
    description="Execute SQL query against database",
    retries=1,
    retry_delay_seconds=5,
    tags=["sql", "execution"]
)
async def query_execution_task(
    sql: str,
    credentials: Dict[str, Any],
    user_id: str,
    max_results: int = 100
) -> QueryExecutionResult:
    """
    Execute SQL query against the database.

    Args:
        sql: SQL query to execute
        credentials: Database credentials dict
        user_id: User ID for tracking
        max_results: Maximum rows to return

    Returns:
        QueryExecutionResult with execution details
    """
    logger = get_run_logger()
    start_time = time.time()

    logger.info(f"[{user_id}] Executing SQL query")

    result = QueryExecutionResult(sql=sql)

    try:
        from sql_pipeline.services.execution_service import ExecutionService
        from sql_pipeline.models.query_models import SQLCredentials

        execution_service = ExecutionService()

        # Build credentials object
        creds = SQLCredentials(
            server=credentials.get("server", ""),
            database=credentials.get("database", ""),
            username=credentials.get("username", ""),
            password=credentials.get("password", ""),
            domain=credentials.get("domain")
        )

        exec_result = await execution_service.execute(sql, creds, max_results=max_results)

        result.success = exec_result.success
        result.data = exec_result.data or []
        result.row_count = exec_result.row_count
        result.column_names = exec_result.column_names or []
        result.execution_time_ms = exec_result.execution_time_ms
        result.error = exec_result.error or ""

        if result.success:
            logger.info(f"[{user_id}] Query executed: {result.row_count} rows in {result.execution_time_ms:.1f}ms")
        else:
            logger.error(f"[{user_id}] Query execution failed: {result.error}")

    except Exception as e:
        result.error = str(e)
        result.success = False
        logger.error(f"[{user_id}] Execution error: {e}")

    result.duration_ms = (time.time() - start_time) * 1000

    # Create artifact
    await create_markdown_artifact(
        key=f"query-execution-{user_id[:8]}",
        markdown=f"""
## Query Execution
- **User**: {user_id}
- **Success**: {result.success}
- **Rows Returned**: {result.row_count}
- **Execution Time**: {result.execution_time_ms:.1f}ms
- **Task Duration**: {result.duration_ms:.1f}ms

### Query
```sql
{sql}
```

{f"### Error{chr(10)}{result.error}" if result.error else ""}
        """,
        description=f"Query execution for user {user_id}"
    )

    return result


@task(
    name="result_formatting",
    description="Format results for response",
    retries=1,
    retry_delay_seconds=2,
    tags=["sql", "formatting"]
)
async def result_formatting_task(
    question: str,
    database: str,
    user_id: str,
    generated_sql: str,
    matched_rules: List[Dict[str, Any]],
    execution_result: Optional[QueryExecutionResult],
    is_exact_match: bool,
    exact_match_rule_id: Optional[str],
    token_usage: Dict[str, int],
    total_duration_ms: float,
    timing_breakdown: Dict[str, float]
) -> ResultFormattingResult:
    """
    Format all results into final response structure.

    Args:
        question: Original question
        database: Target database
        user_id: User ID for tracking
        generated_sql: The SQL that was generated
        matched_rules: Rules that were matched
        execution_result: Optional execution result
        is_exact_match: Whether this was an exact rule match
        exact_match_rule_id: Rule ID if exact match
        token_usage: LLM token usage
        total_duration_ms: Total processing time
        timing_breakdown: Timing for each step

    Returns:
        ResultFormattingResult with formatted response
    """
    logger = get_run_logger()
    start_time = time.time()

    result = ResultFormattingResult()

    success = True
    if execution_result and not execution_result.success:
        success = False

    result.formatted_response = {
        "success": success,
        "sql": generated_sql,
        "explanation": f"Generated SQL query for: {question}",
        "database": database,
        "user_id": user_id,
        "matched_rules": [r.get("rule_id") for r in matched_rules],
        "is_exact_match": is_exact_match,
        "rule_id": exact_match_rule_id,
        "confidence": 1.0 if is_exact_match else 0.8,
        "token_usage": token_usage,
        "processing_time_ms": total_duration_ms,
        "timing": timing_breakdown
    }

    if execution_result:
        result.formatted_response["execution"] = {
            "success": execution_result.success,
            "row_count": execution_result.row_count,
            "column_names": execution_result.column_names,
            "data": execution_result.data,
            "execution_time_ms": execution_result.execution_time_ms,
            "error": execution_result.error
        }

    result.duration_ms = (time.time() - start_time) * 1000

    logger.info(f"[{user_id}] Result formatted, success={success}")

    return result


@task(
    name="learning_storage",
    description="Store successful queries in agent_learning for future cache hits",
    retries=2,
    retry_delay_seconds=5,
    tags=["sql", "learning", "storage"]
)
async def learning_storage_task(
    question: str,
    sql: str,
    database: str,
    user_id: str,
    success: bool
) -> bool:
    """
    Store successful queries in agent_learning collection.

    Args:
        question: Natural language question
        sql: Generated SQL
        database: Target database
        user_id: User ID for tracking
        success: Whether the query was successful

    Returns:
        Whether storage was successful
    """
    logger = get_run_logger()

    # Only cache successful queries
    if not success:
        logger.info(f"[{user_id}] Skipping learning storage for failed query")
        return False

    try:
        from mongodb import MongoDBService

        mongo_service = MongoDBService.get_instance()
        if not mongo_service.is_initialized:
            await mongo_service.initialize()

        collection = mongo_service.db["agent_learning"]

        document = {
            "question": question,
            "question_normalized": question.lower().strip(),
            "sql": sql,
            "database": database,
            "user_id": user_id,
            "success": success,
            "created_at": time.time(),
            "source": "prefect_flow"
        }

        await collection.insert_one(document)
        logger.info(f"[{user_id}] Learning entry stored")
        return True

    except Exception as e:
        logger.warning(f"[{user_id}] Failed to store learning: {e}")
        return False


# =============================================================================
# Helper Functions
# =============================================================================

def _extract_sql_from_response(response: str) -> str:
    """
    Extract SQL from LLM response, handling various formats.
    """
    import re

    if not response:
        return ""

    # Remove markdown code blocks
    if "```sql" in response:
        match = re.search(r"```sql\s*(.*?)\s*```", response, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()

    if "```" in response:
        match = re.search(r"```\s*(.*?)\s*```", response, re.DOTALL)
        if match:
            return match.group(1).strip()
        response = response.replace("```sql", "").replace("```", "").strip()

    # Remove SQL: prefix
    if response.upper().startswith("SQL:"):
        response = response[4:].strip()
    elif "SQL:" in response:
        parts = response.split("SQL:", 1)
        if parts[1].strip():
            response = parts[1].strip()
        else:
            response = parts[0].strip()

    # Stop at duplicate SQL: prefix
    if "SQL:" in response:
        response = response.split("SQL:", 1)[0].strip()
        response = response.rstrip(";").strip()

    # Take first statement
    if "\n\n" in response:
        response = response.split("\n\n")[0].strip()

    # Handle duplicates after semicolon
    if ";" in response:
        statements = response.split(";")
        non_empty = [s.strip() for s in statements if s.strip()]
        if non_empty:
            first = non_empty[0].upper().strip()
            if len(non_empty) > 1 and all(s.upper().strip() == first for s in non_empty):
                response = non_empty[0]
            else:
                response = "; ".join(non_empty)

    # Remove stray backticks
    response = response.replace("`", "").strip()

    return response.strip()


# =============================================================================
# Main Flow
# =============================================================================

@flow(
    name="sql-query-pipeline",
    description="SQL Query Pipeline - Convert natural language to SQL with Prefect tracking",
    retries=1,
    retry_delay_seconds=60
)
async def sql_query_flow(
    question: str,
    database: str,
    server: str,
    user_id: str,
    credentials: Optional[Dict[str, Any]] = None,
    execute_sql: bool = False,
    include_schema: bool = True,
    use_cache: bool = True,
    max_results: int = 100,
    max_tokens: int = 512
) -> Dict[str, Any]:
    """
    SQL Query Pipeline - Process natural language questions to SQL.

    This flow orchestrates:
    1. Security validation on input
    2. Cache check for previous responses
    3. Rule matching (exact -> similarity -> keyword)
    4. Schema retrieval for context
    5. LLM SQL generation
    6. Syntax fixing
    7. Column validation
    8. SQL security validation
    9. Query execution (optional)
    10. Learning storage

    Args:
        question: Natural language question
        database: Target database name
        server: SQL Server hostname
        user_id: User ID for tracking (required)
        credentials: Database credentials (required if execute_sql=True)
        execute_sql: Whether to execute the generated SQL
        include_schema: Whether to include schema context
        use_cache: Whether to use cache
        max_results: Maximum rows to return
        max_tokens: Maximum LLM tokens

    Returns:
        Dict with SQL, execution results, and metadata
    """
    logger = get_run_logger()
    flow_start = time.time()
    timing = {}

    logger.info(f"[{user_id}] Starting SQL Query Pipeline")
    logger.info(f"[{user_id}] Question: {question}")
    logger.info(f"[{user_id}] Database: {database}, Server: {server}")

    # Step 1: Security validation
    security_result = await security_validation_task(question=question, user_id=user_id)
    timing["security_validation_ms"] = security_result.duration_ms

    if security_result.blocked:
        return {
            "success": False,
            "error": f"Security check failed: {', '.join(security_result.issues)}",
            "blocked": True,
            "user_id": user_id,
            "timing": timing,
            "processing_time_ms": (time.time() - flow_start) * 1000
        }

    # Step 2: Cache check
    cache_result = await cache_check_task(
        question=question,
        database=database,
        user_id=user_id,
        use_cache=use_cache
    )
    timing["cache_check_ms"] = cache_result.duration_ms

    if cache_result.cache_hit:
        logger.info(f"[{user_id}] Returning cached SQL")

        # Execute cached SQL if requested
        execution_result = None
        if execute_sql and credentials:
            execution_result = await query_execution_task(
                sql=cache_result.cached_sql,
                credentials=credentials,
                user_id=user_id,
                max_results=max_results
            )
            timing["execution_ms"] = execution_result.duration_ms

        # Store in learning
        await learning_storage_task(
            question=question,
            sql=cache_result.cached_sql,
            database=database,
            user_id=user_id,
            success=True
        )

        return {
            "success": True,
            "sql": cache_result.cached_sql,
            "explanation": "Cached response from previous successful query",
            "database": database,
            "user_id": user_id,
            "cached": True,
            "confidence": 1.0,
            "execution": execution_result.data if execution_result and execution_result.success else None,
            "row_count": execution_result.row_count if execution_result else 0,
            "timing": timing,
            "processing_time_ms": (time.time() - flow_start) * 1000
        }

    # Step 3: Rule matching
    rule_result = await rule_matching_task(
        question=question,
        database=database,
        user_id=user_id
    )
    timing["rule_matching_ms"] = rule_result.duration_ms

    # Check for exact match bypass
    if rule_result.is_exact_match and rule_result.exact_match:
        sql = rule_result.exact_match["sql"]
        logger.info(f"[{user_id}] Using exact match SQL from rule: {rule_result.exact_match['rule_id']}")

        # Execute if requested
        execution_result = None
        if execute_sql and credentials:
            execution_result = await query_execution_task(
                sql=sql,
                credentials=credentials,
                user_id=user_id,
                max_results=max_results
            )
            timing["execution_ms"] = execution_result.duration_ms

        # Store in learning
        await learning_storage_task(
            question=question,
            sql=sql,
            database=database,
            user_id=user_id,
            success=execution_result.success if execution_result else True
        )

        total_duration = (time.time() - flow_start) * 1000

        # Create final summary artifact
        await create_markdown_artifact(
            key=f"pipeline-summary-{user_id[:8]}",
            markdown=f"""
# SQL Query Pipeline Complete (Exact Match)

## Overview
- **User**: {user_id}
- **Database**: {database}
- **Processing Time**: {total_duration:.1f}ms
- **Status**: Exact Match

## Matched Rule
- **Rule ID**: {rule_result.exact_match['rule_id']}
- **Description**: {rule_result.exact_match['description']}

## Generated SQL
```sql
{sql}
```

## Timing Breakdown
{chr(10).join(f"- **{k}**: {v:.1f}ms" for k, v in timing.items())}
            """,
            description=f"Pipeline summary for {user_id}"
        )

        return {
            "success": execution_result.success if execution_result else True,
            "sql": sql,
            "explanation": f"Exact match from rule: {rule_result.exact_match['description']}",
            "database": database,
            "user_id": user_id,
            "is_exact_match": True,
            "rule_id": rule_result.exact_match['rule_id'],
            "confidence": 1.0,
            "execution": {
                "success": execution_result.success,
                "data": execution_result.data,
                "row_count": execution_result.row_count,
                "error": execution_result.error
            } if execution_result else None,
            "timing": timing,
            "processing_time_ms": total_duration
        }

    # Step 4: Schema retrieval
    schema_result = await schema_retrieval_task(
        question=question,
        database=database,
        user_id=user_id,
        matched_rules=rule_result.matched_rules,
        include_schema=include_schema
    )
    timing["schema_retrieval_ms"] = schema_result.duration_ms

    # Step 5: Query generation
    generation_result = await query_generation_task(
        question=question,
        database=database,
        user_id=user_id,
        schema_text=schema_result.schema_text,
        matched_rules=rule_result.matched_rules,
        max_tokens=max_tokens
    )
    timing["query_generation_ms"] = generation_result.duration_ms

    if not generation_result.success:
        return {
            "success": False,
            "error": generation_result.error,
            "user_id": user_id,
            "timing": timing,
            "processing_time_ms": (time.time() - flow_start) * 1000
        }

    # Step 6: Syntax fixing
    syntax_result = await syntax_fixing_task(
        sql=generation_result.generated_sql,
        database=database,
        user_id=user_id
    )
    timing["syntax_fixing_ms"] = syntax_result.duration_ms

    sql = syntax_result.fixed_sql or generation_result.generated_sql

    # Step 7: Column validation
    column_result = await column_validation_task(
        sql=sql,
        database=database,
        user_id=user_id
    )
    timing["column_validation_ms"] = column_result.duration_ms

    if not column_result.is_valid:
        return {
            "success": False,
            "error": f"Invalid columns: {', '.join(column_result.invalid_columns)}",
            "suggestions": column_result.suggestions,
            "sql": sql,
            "user_id": user_id,
            "timing": timing,
            "processing_time_ms": (time.time() - flow_start) * 1000
        }

    # Step 8: SQL security validation
    sql_security_result = await security_sql_validation_task(sql=sql, user_id=user_id)
    timing["sql_security_ms"] = sql_security_result.duration_ms

    if sql_security_result.blocked:
        return {
            "success": False,
            "error": f"Generated SQL failed security check: {', '.join(sql_security_result.issues)}",
            "sql": sql,
            "user_id": user_id,
            "timing": timing,
            "processing_time_ms": (time.time() - flow_start) * 1000
        }

    # Step 9: Query execution (optional)
    execution_result = None
    exec_success = True
    if execute_sql and credentials:
        execution_result = await query_execution_task(
            sql=sql,
            credentials=credentials,
            user_id=user_id,
            max_results=max_results
        )
        timing["execution_ms"] = execution_result.duration_ms
        exec_success = execution_result.success

    # Step 10: Learning storage
    await learning_storage_task(
        question=question,
        sql=sql,
        database=database,
        user_id=user_id,
        success=exec_success
    )

    total_duration = (time.time() - flow_start) * 1000

    # Create final summary artifact
    await create_markdown_artifact(
        key=f"pipeline-summary-{user_id[:8]}",
        markdown=f"""
# SQL Query Pipeline Complete

## Overview
- **User**: {user_id}
- **Database**: {database}
- **Processing Time**: {total_duration:.1f}ms
- **Status**: {"Success" if exec_success else "Execution Failed"}

## Input Question
{question}

## Generated SQL
```sql
{sql}
```

## Metrics
| Metric | Value |
|--------|-------|
| Matched Rules | {len(rule_result.matched_rules)} |
| Tables Used | {schema_result.table_count} |
| Syntax Fixes | {syntax_result.fix_count} |
| Token Usage | {generation_result.token_usage} |
| Rows Returned | {execution_result.row_count if execution_result else "N/A"} |

## Timing Breakdown
{chr(10).join(f"- **{k}**: {v:.1f}ms" for k, v in timing.items())}
        """,
        description=f"Pipeline summary for {user_id}"
    )

    # Format final result
    result = {
        "success": exec_success,
        "sql": sql,
        "explanation": f"Generated SQL query for: {question}",
        "database": database,
        "user_id": user_id,
        "is_exact_match": False,
        "matched_rules": [r.get("rule_id") for r in rule_result.matched_rules],
        "confidence": 0.8,
        "token_usage": generation_result.token_usage,
        "syntax_fixes": syntax_result.fixes_applied,
        "tables_used": schema_result.tables,
        "timing": timing,
        "processing_time_ms": total_duration
    }

    if execution_result:
        result["execution"] = {
            "success": execution_result.success,
            "data": execution_result.data,
            "row_count": execution_result.row_count,
            "column_names": execution_result.column_names,
            "execution_time_ms": execution_result.execution_time_ms,
            "error": execution_result.error
        }

    return result


# =============================================================================
# Public Wrapper Functions
# =============================================================================

def run_sql_query_flow(
    question: str,
    database: Optional[str] = None,
    server: Optional[str] = None,
    user_id: str = "anonymous",
    credentials: Optional[Dict[str, Any]] = None,
    execute_sql: bool = False,
    include_schema: bool = True,
    use_cache: bool = True,
    max_results: Optional[int] = None,
    max_tokens: Optional[int] = None,
    use_prefect: bool = True,
    load_from_variables: bool = True
) -> Dict[str, Any]:
    """
    Run the SQL query flow and return structured results.

    This is the main entry point for the API to use Prefect-tracked SQL query processing.
    Automatically loads defaults from Prefect Variables when load_from_variables=True.
    Includes fallback to direct pipeline if Prefect fails.

    Prefect Variables used (when load_from_variables=True):
        - sql_server: Default SQL Server hostname
        - sql_database: Default database name
        - sql_username: SQL username
        - sql_password: SQL password
        - sql_domain: Windows domain (only used if auth_type="windows")
        - sql_auth_type: "windows" or "sql" (determines if domain is prefixed)
        - default_max_tokens: Default max LLM tokens
        - max_results: Default max results

    Args:
        question: Natural language question
        database: Target database name (loaded from sql_database if None)
        server: SQL Server hostname (loaded from sql_server if None)
        user_id: User ID for tracking
        credentials: Database credentials (auto-loaded from variables if None and execute_sql=True)
        execute_sql: Whether to execute the generated SQL
        include_schema: Whether to include schema context
        use_cache: Whether to use cache
        max_results: Maximum rows to return (loaded from max_results variable if None)
        max_tokens: Maximum LLM tokens (loaded from default_max_tokens if None)
        use_prefect: If True, run through Prefect flow for tracking
        load_from_variables: If True, load defaults from Prefect Variables

    Returns:
        Dict with SQL, execution results, and metadata
    """
    async def _run():
        nonlocal database, server, credentials, max_results, max_tokens

        # Load defaults from Prefect Variables if enabled
        if load_from_variables:
            sql_creds = await load_sql_credentials_from_variables()
            settings = await load_pipeline_settings_from_variables()

            # Use Prefect variables as defaults if not provided
            if database is None:
                database = sql_creds["database"]
            if server is None:
                server = sql_creds["server"]
            if max_tokens is None:
                max_tokens = settings["max_tokens"]
            if max_results is None:
                max_results = settings["max_results"]

            # Auto-load credentials if executing SQL and none provided
            if execute_sql and credentials is None:
                credentials = {
                    "server": sql_creds["server"],
                    "database": sql_creds["database"],
                    "username": sql_creds["username"],  # Already has domain if windows auth
                    "password": sql_creds["password"],
                    "domain": sql_creds["domain"],
                    "auth_type": sql_creds["auth_type"],
                }
        else:
            # Use hardcoded defaults
            database = database or "EWRCentral"
            server = server or "NCSQLTEST"
            max_tokens = max_tokens or 512
            max_results = max_results or 100

        if use_prefect:
            try:
                return await sql_query_flow(
                    question=question,
                    database=database,
                    server=server,
                    user_id=user_id,
                    credentials=credentials,
                    execute_sql=execute_sql,
                    include_schema=include_schema,
                    use_cache=use_cache,
                    max_results=max_results,
                    max_tokens=max_tokens
                )
            except Exception as e:
                import logging
                logging.getLogger(__name__).warning(f"Prefect flow failed, falling back to direct pipeline: {e}")

        # Fallback to direct pipeline execution
        from sql_pipeline.query_pipeline import get_query_pipeline
        from sql_pipeline.models.query_models import (
            SQLQueryRequest,
            SQLCredentials,
            QueryOptions
        )

        pipeline = await get_query_pipeline()

        creds = None
        if credentials:
            creds = SQLCredentials(
                server=credentials.get("server", server),
                database=credentials.get("database", database),
                username=credentials.get("username", ""),
                password=credentials.get("password", ""),
                domain=credentials.get("domain")
            )

        options = QueryOptions(
            execute_sql=execute_sql,
            include_schema=include_schema,
            use_cache=use_cache,
            max_results=max_results
        )

        request = SQLQueryRequest(
            natural_language=question,
            database=database,
            server=server,
            credentials=creds,
            options=options,
            max_tokens=max_tokens
        )

        result = await pipeline.process_query(request)
        return result.model_dump()

    try:
        return asyncio.run(_run())
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "user_id": user_id
        }


async def run_sql_query_flow_async(
    question: str,
    database: Optional[str] = None,
    server: Optional[str] = None,
    user_id: str = "anonymous",
    credentials: Optional[Dict[str, Any]] = None,
    execute_sql: bool = False,
    include_schema: bool = True,
    use_cache: bool = True,
    max_results: Optional[int] = None,
    max_tokens: Optional[int] = None,
    use_prefect: bool = True,
    load_from_variables: bool = True
) -> Dict[str, Any]:
    """
    Async version of run_sql_query_flow for use in async contexts.

    Automatically loads defaults from Prefect Variables when load_from_variables=True.
    See run_sql_query_flow for full documentation.

    Args:
        question: Natural language question
        database: Target database (loaded from Prefect variables if None)
        server: SQL Server (loaded from Prefect variables if None)
        user_id: User ID for tracking
        credentials: Database credentials (auto-loaded if None and execute_sql=True)
        execute_sql: Whether to execute the generated SQL
        include_schema: Whether to include schema context
        use_cache: Whether to use cache
        max_results: Maximum rows (loaded from Prefect variables if None)
        max_tokens: Maximum LLM tokens (loaded from Prefect variables if None)
        use_prefect: If True, run through Prefect flow for tracking
        load_from_variables: If True, load defaults from Prefect Variables

    Returns:
        Dict with SQL, execution results, and metadata
    """
    # Load defaults from Prefect Variables if enabled
    if load_from_variables:
        sql_creds = await load_sql_credentials_from_variables()
        settings = await load_pipeline_settings_from_variables()

        # Use Prefect variables as defaults if not provided
        if database is None:
            database = sql_creds["database"]
        if server is None:
            server = sql_creds["server"]
        if max_tokens is None:
            max_tokens = settings["max_tokens"]
        if max_results is None:
            max_results = settings["max_results"]

        # Auto-load credentials if executing SQL and none provided
        if execute_sql and credentials is None:
            credentials = {
                "server": sql_creds["server"],
                "database": sql_creds["database"],
                "username": sql_creds["username"],  # Already has domain if windows auth
                "password": sql_creds["password"],
                "domain": sql_creds["domain"],
                "auth_type": sql_creds["auth_type"],
            }
    else:
        # Use hardcoded defaults
        database = database or "EWRCentral"
        server = server or "NCSQLTEST"
        max_tokens = max_tokens or 512
        max_results = max_results or 100

    if use_prefect:
        try:
            return await sql_query_flow(
                question=question,
                database=database,
                server=server,
                user_id=user_id,
                credentials=credentials,
                execute_sql=execute_sql,
                include_schema=include_schema,
                use_cache=use_cache,
                max_results=max_results,
                max_tokens=max_tokens
            )
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(f"Prefect flow failed, falling back to direct pipeline: {e}")

    # Fallback to direct pipeline
    from sql_pipeline.query_pipeline import get_query_pipeline
    from sql_pipeline.models.query_models import (
        SQLQueryRequest,
        SQLCredentials,
        QueryOptions
    )

    pipeline = await get_query_pipeline()

    creds = None
    if credentials:
        creds = SQLCredentials(
            server=credentials.get("server", server),
            database=credentials.get("database", database),
            username=credentials.get("username", ""),
            password=credentials.get("password", ""),
            domain=credentials.get("domain")
        )

    options = QueryOptions(
        execute_sql=execute_sql,
        include_schema=include_schema,
        use_cache=use_cache,
        max_results=max_results
    )

    request = SQLQueryRequest(
        natural_language=question,
        database=database,
        server=server,
        credentials=creds,
        options=options,
        max_tokens=max_tokens
    )

    result = await pipeline.process_query(request)
    return result.model_dump()


# =============================================================================
# CLI Entry Point
# =============================================================================

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: python sql_query_flow.py <question> <database> [server]")
        print("Example: python sql_query_flow.py 'show all tickets' EWRCentral NCSQLTEST")
        sys.exit(1)

    question = sys.argv[1]
    database = sys.argv[2]
    server = sys.argv[3] if len(sys.argv) > 3 else "NCSQLTEST"

    result = run_sql_query_flow(
        question=question,
        database=database,
        server=server,
        user_id="cli_user"
    )

    print(f"\nResult: {result}")
