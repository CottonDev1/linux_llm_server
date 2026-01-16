"""
Query Pipeline Module

This module provides the main query processing pipeline for converting
natural language questions to SQL queries using LLM-powered generation
with rule-based validation and correction.
"""

from typing import AsyncGenerator, Optional, List
import asyncio
import logging
import time
import json
import re
from core.log_utils import log_info

from sql_pipeline.models.query_models import (
    SQLCredentials,
    QueryOptions,
    SQLQueryRequest,
    SSEEvent,
    SQLQueryResult,
)
from sql_pipeline.models.rule_models import SQLRule
from sql_pipeline.models.validation_models import ValidationResult, ExecutionResult
from services.question_rewriter import QuestionRewriter

logger = logging.getLogger(__name__)


class QueryPipeline:
    """
    Main pipeline for processing natural language to SQL queries.

    This pipeline orchestrates:
    1. Security check on input
    2. Preprocess question
    3. Cache check (agent learning)
    4. Rule matching (exact -> similarity -> keyword)
    5. Schema loading (compressed)
    6. LLM generation
    7. Syntax fixing
    8. Validation + auto-fix
    9. Execution (optional)
    10. Learning feedback storage

    Attributes:
        rules_service: Service for loading and matching SQL rules
        schema_service: Service for retrieving database schema
        security_service: Service for security checks
        validation_service: Service for SQL validation
        execution_service: Service for SQL execution
        syntax_fixer: Service for fixing SQL syntax issues
        preprocessor: Service for preprocessing natural language input
        llm_service: Service for LLM operations
    """

    def __init__(
        self,
        mongodb_uri: str = "mongodb://localhost:27017",
        llm_endpoint: str = "http://localhost:11434",
        cache_enabled: bool = True,
    ):
        """
        Initialize the query pipeline.

        Args:
            mongodb_uri: Connection string for MongoDB
            llm_endpoint: Endpoint for the LLM service (Ollama/OpenAI compatible)
            cache_enabled: Whether to enable caching for schema and rules
        """
        self.mongodb_uri = mongodb_uri
        self.llm_endpoint = llm_endpoint
        self.cache_enabled = cache_enabled

        # Services will be initialized lazily
        self._rules_service = None
        self._schema_service = None
        self._security_service = None
        self._validation_service = None
        self._execution_service = None
        self._syntax_fixer = None
        self._preprocessor = None
        self._llm_service = None
        self._mongodb_service = None
        self._question_rewriter = None

        log_info("Query Pipeline", "Initialized")

    async def _get_services(self):
        """Initialize all services lazily."""
        if self._rules_service is None:
            from sql_pipeline.services.rules_service import RulesService
            self._rules_service = await RulesService.get_instance()

        if self._schema_service is None:
            from sql_pipeline.services.schema_service import SchemaService
            self._schema_service = await SchemaService.get_instance()

        if self._security_service is None:
            from sql_pipeline.services.security_service import SecurityService
            self._security_service = SecurityService()

        if self._execution_service is None:
            from sql_pipeline.services.execution_service import ExecutionService
            self._execution_service = ExecutionService()

        if self._syntax_fixer is None:
            from sql_pipeline.services.syntax_fixer import SyntaxFixer
            self._syntax_fixer = SyntaxFixer(rules_service=self._rules_service)

        if self._preprocessor is None:
            from sql_pipeline.services.preprocessor import Preprocessor
            self._preprocessor = Preprocessor()

        if self._llm_service is None:
            from services.llm_service import get_llm_service
            self._llm_service = await get_llm_service()

        if self._mongodb_service is None:
            from mongodb import MongoDBService
            self._mongodb_service = MongoDBService.get_instance()
            if not self._mongodb_service.is_initialized:
                await self._mongodb_service.initialize()

        if self._question_rewriter is None:
            self._question_rewriter = QuestionRewriter(self._llm_service)

    async def process_query(
        self,
        request: SQLQueryRequest,
    ) -> SQLQueryResult:
        """
        Process a natural language query and return SQL result.

        Args:
            request: The SQL query request containing natural language and options

        Returns:
            SQLQueryResult with generated SQL, explanation, and optional execution results
        """
        await self._get_services()

        start_time = time.time()

        # Call generate with the appropriate parameters
        result = await self.generate(
            question=request.natural_language,
            database=request.database,
            server=request.server,
            credentials=request.credentials,
            options=request.options,
        )

        return result

    async def generate(
        self,
        question: str,
        database: str,
        server: str,
        credentials: Optional[SQLCredentials] = None,
        options: Optional[QueryOptions] = None,
    ) -> SQLQueryResult:
        """
        Main non-streaming entry point for SQL generation.

        Args:
            question: Natural language question
            database: Target database name
            server: SQL Server hostname
            credentials: Database credentials (required for execution)
            options: Query options

        Returns:
            SQLQueryResult with generated SQL and optional execution
        """
        await self._get_services()

        start_time = time.time()
        options = options or QueryOptions()

        # 1. Security check on input
        logger.info(f"Processing question: {question}")
        security_check = self._security_service.check_query(question)
        if security_check.blocked:
            raise ValueError(f"Security check failed: {', '.join(security_check.issues)}")

        # 2. Cache check (agent learning)
        cached_sql = await self._check_cache(question, database)
        if cached_sql and options.use_cache:
            logger.info("Cache hit - returning cached SQL")
            return SQLQueryResult(
                sql=cached_sql,
                explanation="Cached response from previous successful query",
                confidence=1.0,
                processing_time=time.time() - start_time
            )

        # 3. Rule matching (exact -> similarity -> keyword)
        matched_rules = await self._find_matching_rules(question, database)
        logger.info(f"Matched {len(matched_rules)} rules")

        # Check for exact match bypass
        exact_match = await self._rules_service.find_exact_match(question, database)
        if exact_match and exact_match.example:
            logger.info(f"EXACT MATCH BYPASS: Using SQL from rule '{exact_match.rule_id}'")
            sql = exact_match.example.sql
            explanation = f"Used exact match from rule: {exact_match.description}"

            # Still validate and optionally execute
            execution_result = None
            exec_success = True
            if options.execute_sql and credentials:
                execution_result = await self._execution_service.execute(
                    sql, credentials, max_results=options.max_results
                )
                exec_success = execution_result.success

            return SQLQueryResult(
                sql=sql,
                explanation=explanation,
                execution_result=execution_result.model_dump() if execution_result else None,
                matched_rules=[exact_match.rule_id],
                confidence=1.0,
                processing_time=time.time() - start_time,
                success=exec_success,
                is_exact_match=True,
                rule_id=exact_match.rule_id,
            )

        # 4. Schema loading (compressed)
        schema = None
        if options.include_schema:
            schema_info = await self._schema_service.get_relevant_schema(
                database=database,
                question=question,
                max_tables=4
            )
            schema = self._schema_service.format_schema_for_prompt(schema_info.tables)
            logger.info(f"Loaded schema with {len(schema_info.tables)} relevant tables")

        # 5. LLM generation
        sql, token_usage = await self._generate_sql(question, database, schema, matched_rules)
        logger.info(f"Generated SQL: {sql[:100]}...")
        logger.info(f"Token usage: {token_usage}")

        # 6. Syntax fixing
        fixed_sql, fixes = await self._syntax_fixer.apply_all_fixes(sql, database)
        if fixes:
            logger.info(f"Applied {len(fixes)} syntax fixes")
            sql = fixed_sql

        # 7. Validation (TODO: implement validation service methods)
        # For now, just check security
        sql_security_check = self._security_service.check_query(sql)
        if sql_security_check.blocked:
            raise ValueError(f"Generated SQL failed security check: {', '.join(sql_security_check.issues)}")

        # 8. Execution (optional)
        execution_result = None
        success = True
        if options.execute_sql and credentials:
            try:
                execution_result = await self._execution_service.execute(
                    sql, credentials, max_results=options.max_results
                )
                success = execution_result.success
                logger.info(f"Execution result: {execution_result.row_count} rows")
            except Exception as e:
                logger.error(f"Execution failed: {e}")
                success = False

        # 9. Learning feedback storage
        await self._store_learning(question, sql, database, success)

        processing_time = time.time() - start_time

        return SQLQueryResult(
            sql=sql,
            explanation=f"Generated SQL query for: {question}",
            execution_result=execution_result.model_dump() if execution_result else None,
            matched_rules=[r.rule_id for r in matched_rules],
            confidence=0.8,  # TODO: Calculate actual confidence
            processing_time=processing_time,
            success=success,
            is_exact_match=False,
            token_usage=token_usage,
        )

    async def process_query_streaming(
        self,
        request: SQLQueryRequest,
    ) -> AsyncGenerator[SSEEvent, None]:
        """
        Process a natural language query with streaming SSE events.

        Args:
            request: The SQL query request

        Yields:
            SSEEvent objects representing pipeline progress and results
        """
        await self._get_services()

        async for event in self.generate_stream(
            question=request.natural_language,
            database=request.database,
            server=request.server,
            credentials=request.credentials,
            options=request.options,
            max_tokens=request.max_tokens,
            conversation_history=request.conversation_history,
        ):
            yield event

    async def generate_stream(
        self,
        question: str,
        database: str,
        server: str,
        credentials: Optional[SQLCredentials] = None,
        options: Optional[QueryOptions] = None,
        max_tokens: int = 512,
        conversation_history: Optional[List] = None,
    ) -> AsyncGenerator[SSEEvent, None]:
        """
        Streaming entry point - yields SSE events for each stage.

        Args:
            question: Natural language question
            database: Target database name
            server: SQL Server hostname
            credentials: Database credentials
            options: Query options
            max_tokens: Maximum tokens for LLM response
            conversation_history: Previous conversation for context

        Yields:
            SSEEvent objects for each pipeline stage
        """
        await self._get_services()

        start_time = time.time()
        options = options or QueryOptions()

        # Initialize token usage tracking
        token_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

        def progress_event(stage: str, message: str, step: int = 0, total_steps: int = 10) -> SSEEvent:
            """Create a properly formatted progress event for frontend consumption."""
            return SSEEvent(
                event="status",
                data=json.dumps({
                    "type": "progress",
                    "stage": stage,
                    "message": message,
                    "step": step,
                    "totalSteps": total_steps,
                    "elapsed": round(time.time() - start_time, 2)
                })
            )

        try:
            # Log conversation history for debugging
            if conversation_history:
                logger.info(f"Conversation history received: {len(conversation_history)} messages")
                for i, msg in enumerate(conversation_history[:2]):
                    logger.info(f"  History[{i}]: {msg.get('role', 'unknown')}: {str(msg.get('content', ''))[:100]}...")
            else:
                logger.info("No conversation history provided")

            # 1. Preprocessing
            yield progress_event("preprocessing", "Analyzing question...", 1, 10)

            # 1b. Question rewriting (for follow-up questions with conversation history)
            original_question = question
            if conversation_history:
                rewrite_result = await self._question_rewriter.rewrite(
                    current_question=question,
                    history=conversation_history
                )
                if rewrite_result.was_rewritten:
                    question = rewrite_result.rewritten_question
                    logger.info(f"Question rewritten: '{original_question}' -> '{question}'")
                    yield SSEEvent(event="rewrite", data=json.dumps({
                        "type": "progress",
                        "original_question": original_question,
                        "rewritten_question": question,
                        "confidence": rewrite_result.confidence,
                        "message": f"Question rewritten for context"
                    }))

            # 2. Security check
            yield progress_event("security", "Checking security...", 2, 10)
            security_check = self._security_service.check_query(question)
            if security_check.blocked:
                yield SSEEvent(event="error", data=json.dumps({"type": "error", "error": f"Security check failed: {', '.join(security_check.issues)}"}))
                return

            # 3. Cache check
            yield progress_event("cache", "Checking for cached queries...", 3, 10)
            cached_sql = await self._check_cache(question, database)
            if cached_sql and options.use_cache:
                yield SSEEvent(event="sql", data=json.dumps({"type": "result", "sql": cached_sql, "cached": True, "success": True}))
                yield SSEEvent(event="done", data=json.dumps({"type": "done", "processing_time": time.time() - start_time}))
                return

            # 4. Rule matching
            yield progress_event("rules", "Searching SQL rules...", 4, 10)

            # Check exact match first
            exact_match = await self._rules_service.find_exact_match(question, database)
            if exact_match and exact_match.example:
                yield progress_event("rules", "Found exact rule match!", 5, 10)
                yield SSEEvent(event="rules", data=json.dumps({
                    "type": "progress",
                    "matched": True,
                    "rule_id": exact_match.rule_id,
                    "matchType": "exact",
                    "description": exact_match.description,
                    "message": f"Using rule: {exact_match.description}"
                }))

                # Execute if requested
                if options.execute_sql and credentials:
                    yield progress_event("executing", "Executing SQL query...", 9, 10)
                    result = await self._execution_service.execute(
                        exact_match.example.sql, credentials, max_results=options.max_results
                    )
                    yield SSEEvent(event="execution", data=json.dumps({
                        "type": "result",
                        "success": result.success,
                        "generatedSql": exact_match.example.sql,
                        "explanation": f"Exact match from rule: {exact_match.description}",
                        "isExactMatch": True,
                        "ruleId": exact_match.rule_id,
                        **result.model_dump()
                    }))
                else:
                    yield SSEEvent(event="sql", data=json.dumps({
                        "type": "result",
                        "success": True,
                        "generatedSql": exact_match.example.sql,
                        "explanation": f"Exact match from rule: {exact_match.description}",
                        "isExactMatch": True,
                        "ruleId": exact_match.rule_id
                    }))

                yield SSEEvent(event="done", data=json.dumps({"type": "done", "processing_time": time.time() - start_time}))
                return

            # Find matching rules
            matched_rules = await self._find_matching_rules(question, database)
            if matched_rules:
                yield progress_event("rules", f"Found {len(matched_rules)} matching rules", 5, 10)
                yield SSEEvent(event="rules", data=json.dumps({
                    "type": "progress",
                    "matched": True,
                    "count": len(matched_rules),
                    "rules": [{"id": r.rule_id, "description": r.description} for r in matched_rules[:5]],
                    "message": f"Using {len(matched_rules)} rules for context"
                }))

            # 5. Schema loading - include tables from matched rules
            yield progress_event("schema", "Loading database schema...", 6, 10)
            schema = None
            if options.include_schema:
                # Extract trigger_tables from matched rules to ensure they're included
                required_tables = set()
                for rule in matched_rules:
                    if rule.trigger_tables:
                        required_tables.update(rule.trigger_tables)

                schema_info = await self._schema_service.get_relevant_schema(
                    database=database,
                    question=question,
                    max_tables=4,
                    required_tables=list(required_tables) if required_tables else None
                )
                schema = self._schema_service.format_schema_for_prompt(schema_info.tables)
                yield progress_event("schema", f"Loaded {len(schema_info.tables)} relevant tables", 6, 10)
                yield SSEEvent(event="schema", data=json.dumps({
                    "type": "progress",
                    "table_count": len(schema_info.tables),
                    "tables": [t.full_name for t in schema_info.tables],
                    "message": f"Found {len(schema_info.tables)} relevant tables"
                }))

            # 6. LLM generation with progress polling
            yield progress_event("generating", "Building prompt...", 7, 10)

            yield SSEEvent(event="generating", data=json.dumps({
                "type": "progress",
                "stage": "generating",
                "substep": "prompt",
                "message": "Building prompt with schema and rules..."
            }))

            # Start LLM generation as background task
            generation_task = asyncio.create_task(
                self._generate_sql(
                    question, database, schema, matched_rules,
                    max_tokens=max_tokens,
                    conversation_history=conversation_history
                )
            )

            # Poll for progress while waiting for generation
            last_progress_msg = ""
            poll_interval = 0.5  # Poll every 500ms
            generation_start = time.time()

            while not generation_task.done():
                try:
                    # Get progress from llama.cpp /slots endpoint
                    progress = await self._llm_service.get_generation_progress(use_sql_model=True)

                    if progress["is_processing"]:
                        n_decoded = progress["n_decoded"]
                        max_tok = progress["max_tokens"]
                        progress_pct = progress["progress_pct"]
                        elapsed = round(time.time() - generation_start, 1)

                        # Calculate tokens per second
                        tps = round(n_decoded / elapsed, 1) if elapsed > 0 and n_decoded > 0 else 0

                        # Build progress message
                        if n_decoded == 0:
                            progress_msg = f"Processing prompt... ({elapsed}s)"
                        else:
                            progress_msg = f"Generating: {n_decoded} tokens @ {tps} tok/s"

                        # Only yield if message changed
                        if progress_msg != last_progress_msg:
                            last_progress_msg = progress_msg
                            yield SSEEvent(event="generating", data=json.dumps({
                                "type": "progress",
                                "stage": "generating",
                                "substep": "llm",
                                "message": progress_msg,
                                "tokens_generated": n_decoded,
                                "tokens_per_second": tps,
                                "elapsed": elapsed
                            }))
                    else:
                        # Not processing yet, show waiting message
                        elapsed = round(time.time() - generation_start, 1)
                        progress_msg = f"Waiting for LLM... ({elapsed}s)"
                        if progress_msg != last_progress_msg:
                            last_progress_msg = progress_msg
                            yield SSEEvent(event="generating", data=json.dumps({
                                "type": "progress",
                                "stage": "generating",
                                "substep": "llm",
                                "message": progress_msg
                            }))

                except Exception as e:
                    logger.debug(f"Progress poll error: {e}")

                # Wait before next poll, but check if task is done frequently
                try:
                    await asyncio.wait_for(asyncio.shield(generation_task), timeout=poll_interval)
                    break  # Task completed
                except asyncio.TimeoutError:
                    pass  # Continue polling

            # Get result from completed task
            sql, llm_token_usage = await generation_task
            token_usage.update(llm_token_usage)
            logger.info(f"Token usage: {token_usage}")

            # Calculate final stats
            total_elapsed = round(time.time() - generation_start, 1)
            completion_tokens = llm_token_usage.get("completion_tokens", 0)
            final_tps = round(completion_tokens / total_elapsed, 1) if total_elapsed > 0 and completion_tokens > 0 else 0

            yield SSEEvent(event="generating", data=json.dumps({
                "type": "progress",
                "stage": "generating",
                "substep": "complete",
                "message": f"Generated {completion_tokens} tokens in {total_elapsed}s ({final_tps} tok/s)",
                "tokens": llm_token_usage
            }))
            yield progress_event("generating", "SQL generated successfully", 7, 10)

            # 7. Syntax fixing
            yield progress_event("fixing", "Applying syntax fixes...", 7, 11)
            fixed_sql, fixes = await self._syntax_fixer.apply_all_fixes(sql, database)
            if fixes:
                yield progress_event("fixing", f"Applied {len(fixes)} syntax fixes", 7, 11)
                yield SSEEvent(event="validation", data=json.dumps({
                    "type": "progress",
                    "fixes_applied": len(fixes),
                    "fixes": [f.description for f in fixes[:5]],
                    "message": f"Applied {len(fixes)} auto-fixes"
                }))
                sql = fixed_sql

            # 8. Column validation - check columns exist in schema
            yield progress_event("column_check", "Validating column names...", 8, 11)
            column_validation = await self._schema_service.validate_columns(sql, database)
            yield SSEEvent(event="validation", data=json.dumps({
                "type": "column_validation",
                "valid": column_validation["valid"],
                "invalid_columns": column_validation["invalid_columns"],
                "suggestions": column_validation["suggestions"],
                "tables_checked": column_validation["tables_checked"],
                "validation_time_ms": column_validation["validation_time_ms"],
                "message": column_validation["message"]
            }))

            if not column_validation["valid"]:
                # Build helpful error message
                invalid_list = ", ".join(column_validation["invalid_columns"][:5])
                suggestion_text = ""
                if column_validation["suggestions"]:
                    suggestions = [f"{k} → {v}" for k, v in column_validation["suggestions"].items()]
                    suggestion_text = f" Did you mean: {', '.join(suggestions[:3])}?"
                yield SSEEvent(event="error", data=json.dumps({
                    "type": "column_validation_error",
                    "error": f"Invalid column(s): {invalid_list}.{suggestion_text}",
                    "invalid_columns": column_validation["invalid_columns"],
                    "suggestions": column_validation["suggestions"],
                    "failed_sql": sql
                }))
                return

            yield progress_event("column_check", f"Column validation passed ({column_validation['validation_time_ms']}ms)", 8, 11)

            # 9. Security Validation
            yield progress_event("validating", "Validating SQL security...", 9, 11)
            sql_security_check = self._security_service.check_query(sql)
            yield SSEEvent(event="validation", data=json.dumps({
                "type": "security_validation",
                "valid": not sql_security_check.blocked,
                "risk_level": sql_security_check.risk_level.value,
                "issues": sql_security_check.issues,
                "message": "Security validation " + ("passed" if not sql_security_check.blocked else "failed")
            }))

            if sql_security_check.blocked:
                yield SSEEvent(event="error", data=json.dumps({"type": "error", "error": "Generated SQL failed security check"}))
                return

            # 10. Execution (optional)
            success = True
            execution_data = None
            if options.execute_sql and credentials:
                yield progress_event("executing", "Executing SQL query...", 10, 11)
                try:
                    result = await self._execution_service.execute(
                        sql, credentials, max_results=options.max_results
                    )
                    success = result.success
                    execution_data = result.model_dump()
                    yield progress_event("executing", f"Query returned {result.row_count} rows" if result.success else "Query failed", 10, 11)
                except Exception as e:
                    success = False
                    yield SSEEvent(event="error", data=json.dumps({"type": "error", "error": str(e)}))

            # 11. Store learning
            await self._store_learning(question, sql, database, success)

            # 12. Final result
            yield progress_event("complete", "Processing complete", 11, 11)

            # Send the final result with all data
            final_result = {
                "type": "result",
                "success": success,
                "generatedSql": sql,
                "explanation": f"Generated SQL query for: {question}",
                "isExactMatch": False,
                "matchedRules": [r.rule_id for r in matched_rules] if matched_rules else [],
                "processing_time": round(time.time() - start_time, 2),
                "tokenUsage": {
                    "prompt_tokens": token_usage.get("prompt_tokens", 0),
                    "completion_tokens": token_usage.get("completion_tokens", token_usage.get("response_tokens", 0)),
                    "total_tokens": token_usage.get("total_tokens", 0)
                }
            }
            if execution_data:
                final_result.update(execution_data)

            yield SSEEvent(event="result", data=json.dumps(final_result))
            yield SSEEvent(event="done", data=json.dumps({"type": "done", "processing_time": time.time() - start_time}))

        except Exception as e:
            logger.error(f"Error in streaming pipeline: {e}", exc_info=True)
            yield SSEEvent(event="error", data=json.dumps({
                "type": "error",
                "error": str(e),
                "message": f"Pipeline error: {str(e)}"
            }))

    async def validate_sql(
        self,
        sql: str,
        database: str,
        credentials: Optional[SQLCredentials] = None,
    ) -> ValidationResult:
        """
        Validate a SQL query against rules and optionally execute.

        Args:
            sql: The SQL query to validate
            database: Target database name
            credentials: Optional credentials for execution validation

        Returns:
            ValidationResult with validation status and any fixes applied
        """
        await self._get_services()

        # Security check
        security_check = self._security_service.check_query(sql)

        errors = []
        warnings = []

        if security_check.blocked:
            errors.extend(security_check.issues)

        # Apply auto-fixes
        fixed_sql, fixes = await self._syntax_fixer.apply_all_fixes(sql, database)
        auto_fixes_applied = [f.description for f in fixes]

        # Execution validation (optional)
        execution_result = None
        if credentials:
            try:
                execution_result = await self._execution_service.execute(
                    fixed_sql, credentials, max_results=1
                )
                if not execution_result.success:
                    errors.append(execution_result.error)
            except Exception as e:
                errors.append(str(e))

        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            auto_fixes_applied=auto_fixes_applied,
            execution_result=execution_result
        )

    async def get_matched_rules(
        self,
        query: str,
        database: str,
    ) -> list[SQLRule]:
        """
        Get rules that match a given query and database.

        Args:
            query: The natural language query
            database: Target database name

        Returns:
            List of matching SQLRule objects
        """
        await self._get_services()
        return await self._find_matching_rules(query, database)

    async def execute_sql(
        self,
        sql: str,
        credentials: SQLCredentials,
        max_results: int = 100,
    ) -> ExecutionResult:
        """
        Execute a SQL query and return results.

        Args:
            sql: The SQL query to execute
            credentials: Database credentials
            max_results: Maximum number of rows to return

        Returns:
            ExecutionResult with query results or error information
        """
        await self._get_services()
        return await self._execution_service.execute(sql, credentials, max_results)

    # =========================================================================
    # Private Helper Methods
    # =========================================================================

    async def _check_cache(self, question: str, database: str) -> Optional[str]:
        """
        Check agent_learning collection for cached SQL responses.

        Args:
            question: Natural language question
            database: Target database

        Returns:
            Cached SQL if found with high confidence, None otherwise
        """
        try:
            collection = self._mongodb_service.db["agent_learning"]

            # Normalize question for matching
            question_normalized = question.lower().strip()

            # Query for similar questions with successful results
            result = await collection.find_one({
                "database": database,
                "question_normalized": question_normalized,
                "success": True
            })

            if result and result.get("sql"):
                logger.info("Found cached SQL in agent_learning")
                return result["sql"]

        except Exception as e:
            logger.warning(f"Cache check failed: {e}")

        return None

    async def _find_matching_rules(self, question: str, database: str) -> List[SQLRule]:
        """
        Find matching rules using cascading strategy.

        Strategy:
        1. Try exact match first (returns immediately if found)
        2. Try similarity match (threshold 0.8)
        3. Fall back to keyword matches

        Args:
            question: Natural language question
            database: Target database

        Returns:
            List of matched SQLRule objects
        """
        # Try similarity match
        similar_rules = await self._rules_service.find_similar_rules(question, database, threshold=0.8)
        if similar_rules:
            logger.info(f"Found {len(similar_rules)} similar rules")
            return [rule for rule, score in similar_rules]

        # Fall back to keyword matches
        keyword_rules = await self._rules_service.find_keyword_matches(question, database)
        if keyword_rules:
            logger.info(f"Found {len(keyword_rules)} keyword-matched rules")
            return keyword_rules

        logger.info("No matching rules found")
        return []

    async def _generate_sql(
        self,
        question: str,
        database: str,
        schema: Optional[str],
        rules: List[SQLRule],
        max_tokens: int = 512,
        conversation_history: Optional[List] = None,
    ) -> tuple:
        """
        Generate SQL using LLM with schema and rules context.

        Args:
            question: Natural language question
            database: Target database
            schema: Formatted schema markdown
            rules: Matched rules for guidance
            max_tokens: Maximum tokens for LLM response
            conversation_history: Previous conversation for context

        Returns:
            Tuple of (Generated SQL query, token_usage dict)
        """
        # Build prompt
        prompt_parts = [
            "You are an expert SQL Server query generator. Generate a T-SQL query for the following question.",
            f"\nDATABASE: {database}",
        ]

        # Add conversation history for context
        if conversation_history:
            prompt_parts.append("\nCONVERSATION CONTEXT:")
            for msg in conversation_history[-4:]:  # Last 4 messages
                role = msg.get('role', 'user')
                content = msg.get('content', '')[:500]  # Limit length
                prompt_parts.append(f"  {role}: {content}")

        # Add schema if available - with strong instruction to use ONLY these columns
        if schema:
            prompt_parts.append("\nDATABASE SCHEMA (CRITICAL - USE ONLY THESE EXACT COLUMN NAMES):")
            prompt_parts.append(schema)
            prompt_parts.append("\nIMPORTANT: You MUST use ONLY the column names shown above. Do NOT guess, invent, or assume column names. If a column is not listed in the schema, it does not exist.")

        # Add business rules (critical domain-specific rules)
        business_rules = self._get_business_rules(database)
        if business_rules:
            prompt_parts.append("\nBUSINESS RULES (MUST FOLLOW):")
            for rule in business_rules:
                prompt_parts.append(f"- {rule}")

        # Add rules if available
        if rules:
            prompt_parts.append("\nRULES AND GUIDELINES:")
            for rule in rules[:10]:  # Limit to top 10 rules
                prompt_parts.append(f"- {rule.to_prompt_text()}")

        prompt_parts.append(f"\nQUESTION: {question}")
        prompt_parts.append("\nGenerate only the raw SQL query using ONLY column names from the schema above. Do NOT include markdown formatting, code blocks, backticks, or explanations. Output plain T-SQL only.")
        prompt_parts.append("\nSQL:")

        prompt = "\n".join(prompt_parts)

        # Call LLM service
        logger.debug(f"LLM prompt (first 500 chars): {prompt[:500]}...")
        result = await self._llm_service.generate(
            prompt=prompt,
            system="You are a Microsoft SQL Server T-SQL expert. Output only raw T-SQL queries without markdown or backticks. CRITICAL RULES: 1) Use ONLY tables and columns from the provided schema - never invent table names (especially no uvw_ or vw_ prefixes). 2) Use T-SQL syntax only - no PostgreSQL syntax like NULLS FIRST/LAST. 3) For user-related queries, use Login column from CentralUsers, not UserName.",
            use_sql_model=True,
            temperature=0.0,
            max_tokens=max_tokens,
            use_cache=True
        )

        if not result.success:
            raise ValueError(f"LLM generation failed: {result.error}")

        # Log raw LLM response for debugging
        logger.info(f"LLM response success={result.success}, response_length={len(result.response) if result.response else 0}")
        logger.info(f"LLM raw response: {repr(result.response[:500]) if result.response else 'EMPTY'}")

        # Extract SQL from response
        sql = self._extract_sql_from_response(result.response)
        logger.info(f"Extracted SQL length: {len(sql) if sql else 0}")

        # Get token usage from result
        token_usage = result.token_usage if hasattr(result, 'token_usage') else {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0
        }

        return sql, token_usage

    def _extract_sql_from_response(self, response: str) -> str:
        """
        Extract SQL from LLM response.

        Handles cases where LLM includes:
        - Markdown code blocks (```sql)
        - "SQL:" prefix
        - Extra explanations
        - Duplicated SQL statements (LLM repeating itself)

        Args:
            response: Raw LLM response

        Returns:
            Cleaned SQL query
        """
        if not response:
            return ""

        # Remove markdown code blocks
        if "```sql" in response:
            match = re.search(r"```sql\s*(.*?)\s*```", response, re.DOTALL | re.IGNORECASE)
            if match:
                return match.group(1).strip()

        if "```" in response:
            # Try to match complete code block first
            match = re.search(r"```\s*(.*?)\s*```", response, re.DOTALL)
            if match:
                return match.group(1).strip()
            # If no complete match, strip any backticks at start/end
            response = response.replace("```sql", "").replace("```", "").strip()

        # Remove first "SQL:" prefix if present
        if response.upper().startswith("SQL:"):
            response = response[4:].strip()
        elif "SQL:" in response:
            parts = response.split("SQL:", 1)
            # If SQL: is at the end (part after is empty), take the part before
            # If SQL: is in the middle/start, take the part after
            if parts[1].strip():
                response = parts[1].strip()
            else:
                response = parts[0].strip()

        # Stop at duplicate "SQL:" prefix (LLM repeating itself)
        # This handles cases like: "SELECT ... ; SQL: SELECT ... ; SQL: SELECT ..."
        if "SQL:" in response:
            response = response.split("SQL:", 1)[0].strip()
            # Remove trailing semicolon if present before the split
            response = response.rstrip(";").strip()

        # Take first statement (before double newline)
        if "\n\n" in response:
            response = response.split("\n\n")[0].strip()

        # If there's still duplication (same statement repeated after semicolon),
        # just take the first statement
        if ";" in response:
            statements = response.split(";")
            # Filter to only non-empty statements
            non_empty = [s.strip() for s in statements if s.strip()]
            if non_empty:
                # Check if statements are duplicates
                first = non_empty[0].upper().strip()
                if len(non_empty) > 1 and all(s.upper().strip() == first for s in non_empty):
                    # All statements are the same, return just the first one
                    response = non_empty[0]
                else:
                    # Statements are different (maybe CTEs or multi-statement), keep them
                    response = "; ".join(non_empty)

        # Final cleanup - remove any stray backticks that might remain
        response = response.replace("`", "").strip()

        return response.strip()

    async def _store_learning(
        self,
        question: str,
        sql: str,
        database: str,
        success: bool,
        feedback: Optional[str] = None
    ) -> None:
        """
        Store query in agent_learning collection for future cache hits.

        Only stores successful queries to avoid caching failed SQL that would
        block the user from trying again with a fresh LLM response.

        Args:
            question: Original question
            sql: Generated SQL
            database: Target database
            success: Whether the query succeeded
            feedback: Optional user feedback
        """
        # Only cache successful queries - failed queries should not be cached
        # so users can retry and get fresh LLM-generated SQL
        if not success:
            logger.debug("Skipping cache storage for failed query")
            return

        try:
            collection = self._mongodb_service.db["agent_learning"]

            document = {
                "question": question,
                "question_normalized": question.lower().strip(),
                "sql": sql,
                "database": database,
                "success": success,
                "feedback": feedback,
                "created_at": time.time(),
            }

            await collection.insert_one(document)
            logger.debug("Stored learning entry in agent_learning collection")

        except Exception as e:
            logger.warning(f"Failed to store learning: {e}")

    async def generate_response_summary(
        self,
        question: str,
        sql: str,
        result_summary: str,
    ) -> Optional[str]:
        """
        Generate a natural language summary of query results.

        Args:
            question: Original user question
            sql: Executed SQL query
            result_summary: Summary of results (row count, sample data)

        Returns:
            Natural language response or None if generation fails
        """
        await self._get_services()

        prompt = f"""The user asked: "{question}"

I executed this SQL query:
{sql}

Result: {result_summary}

Please provide a brief, friendly 1-2 sentence response that answers the user's question based on these results. Be conversational and directly address what they asked. If there's a specific number or value they asked about, state it clearly. Don't mention SQL or technical details."""

        system = "You are a helpful data assistant. Provide brief, natural responses that directly answer the user's question. Be conversational but concise."

        result = await self._llm_service.generate(
            prompt=prompt,
            system=system,
            temperature=0.3,
            max_tokens=2048,
            use_sql_model=False,  # Use general model for natural language
        )

        if result.success:
            return result.response
        return None

    async def generate_error_explanation(
        self,
        question: str,
        sql: str,
        error_message: str,
    ) -> Optional[str]:
        """
        Generate a user-friendly explanation of SQL errors.

        Args:
            question: Original user question
            sql: Failed SQL query
            error_message: Error message from database

        Returns:
            Friendly error explanation or None
        """
        await self._get_services()

        prompt = f"""The user asked: "{question}"

I generated this SQL query:
{sql}

But it failed with this error:
{error_message}

Please explain in 2-3 sentences what went wrong and suggest how the user could rephrase their question to get better results. Be friendly and helpful. Don't include any SQL code."""

        system = "You are a helpful assistant explaining SQL errors to non-technical users. Be concise, friendly, and focus on what the user can do differently."

        result = await self._llm_service.generate(
            prompt=prompt,
            system=system,
            temperature=0.3,
            max_tokens=2048,
            use_sql_model=False,  # Use general model for error explanations
        )

        if result.success:
            return result.response
        return None

    def _get_business_rules(self, database: str) -> List[str]:
        """
        Get business rules for a database.

        These are critical domain-specific rules that the LLM must follow
        to generate correct SQL.

        Args:
            database: Target database name

        Returns:
            List of business rule strings
        """
        rules = []

        # EWRCentral specific rules (hardcoded critical rules)
        if database.lower() == 'ewrcentral':
            rules.extend([
                # View prioritization - CRITICAL
                "PREFER uvw_ views over base tables - views have pre-joined human-readable columns",
                "For ticket queries: USE uvw_CentralTickets which already has Status, Priority, TicketType, AssignedTo, CompanyName as readable text instead of TypeIDs",
                "For user queries: USE uvw_CentralUsers which has human-readable roles and assignments",
                # TypeID handling - NEVER use hardcoded values
                "NEVER use hardcoded TypeID values (like 1, 2, 3) - always JOIN the Types table or use a view that already has the translation",
                "To get type descriptions: LEFT JOIN Types ON TypeIDColumn = Types.TypeID and SELECT Types.Description",
                "When joining Types multiple times, use aliases: Types_TicketType, Types_Status, etc.",
                # Column name rules
                "Use AddTicketDate (NOT CreateDate) for ticket creation date",
                "Use ExpireDate (NOT ExpirationDate) for license expiration",
                "Use AddCentralUserID for ticket creator (NOT CreateUserID)",
                # Query patterns
                "When counting tickets, use COUNT(*) not SELECT TOP with a number",
                "For date comparisons, use CAST(AddTicketDate AS DATE) = CAST(GETDATE() AS DATE) for 'today'"
            ])

        # EWR Gin database rules
        if 'gin' in database.lower():
            rules.extend([
                "Bale records are in the Bales table",
                "Use GinTicketNumber for gin ticket lookups",
                "Module data is in Modules table with ModuleNumber as key"
            ])

        # Future: Load from sql_knowledge MongoDB collection dynamically

        return rules

    async def close(self):
        """Close all service connections."""
        if self._execution_service:
            await self._execution_service.close()

        if self._llm_service:
            await self._llm_service.close()

        logger.info("QueryPipeline closed")


# Singleton accessor
_query_pipeline: Optional[QueryPipeline] = None


async def get_query_pipeline() -> QueryPipeline:
    """Get or create the global QueryPipeline instance."""
    global _query_pipeline
    if _query_pipeline is None:
        _query_pipeline = QueryPipeline()
        await _query_pipeline._get_services()
    return _query_pipeline
