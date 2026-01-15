"""
SQL Generator Service - Text-to-SQL generation with self-correction.

This service provides:
- Natural language to SQL query conversion
- Schema-aware prompt building
- Self-correction loop for validation errors
- Integration with SQL rules and security services
- Support for conversation history

Architecture Overview:
---------------------
The SQL generation pipeline follows a multi-stage process:

1. Security Check: Validate user prompt for injection/malicious intent
2. Schema Loading: Retrieve relevant table schemas from MongoDB
3. Rule Matching: Check for exact matches in SQL rules database
4. Prompt Building: Construct LLM prompt with schema and examples
5. LLM Generation: Generate SQL using llama.cpp
6. Validation: Validate generated SQL against schema
7. Self-Correction: If validation fails, provide feedback and regenerate
8. Security Validation: Final check that SQL is safe to execute

Design Rationale:
-----------------
- Separation of concerns: Each stage is a distinct method for testability
- Self-correction improves accuracy by up to 30% for complex queries
- Schema injection prevents hallucinated table/column names
- Security validation provides defense-in-depth

The service uses dependency injection for MongoDB and LLM services,
making it easy to test with mocks.
"""

import asyncio
import json
import logging
import os
import re
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple

from services.relationship_graph import build_relationship_context

logger = logging.getLogger(__name__)

# DEPRECATION WARNING
warnings.warn(
    "services.sql_generator is deprecated. Use sql_pipeline.QueryPipeline instead. "
    "Import with: from sql_pipeline import get_query_pipeline",
    DeprecationWarning,
    stacklevel=2
)


@dataclass
class SQLGenerationResult:
    """Result from SQL generation."""
    success: bool
    sql: str = ""
    error: Optional[str] = None
    is_exact_match: bool = False
    rule_id: Optional[str] = None
    validation_attempts: int = 0
    schema_errors: List[Dict[str, Any]] = field(default_factory=list)
    token_usage: Dict[str, int] = field(default_factory=lambda: {
        "prompt_tokens": 0,
        "response_tokens": 0,
        "total_tokens": 0
    })
    generation_time_ms: int = 0
    model_used: str = ""
    security_blocked: bool = False
    security_violations: List[str] = field(default_factory=list)


@dataclass
class SchemaContext:
    """Schema context for SQL generation with rich MongoDB metadata."""
    tables: List[Dict[str, Any]] = field(default_factory=list)
    views: List[Dict[str, Any]] = field(default_factory=list)  # Pre-joined views (prioritize these!)
    formatted_schema: str = ""
    formatted_views: str = ""  # Separate formatted views section
    table_names: Set[str] = field(default_factory=set)
    view_names: Set[str] = field(default_factory=set)
    examples: List[Dict[str, Any]] = field(default_factory=list)
    failed_queries: List[Dict[str, Any]] = field(default_factory=list)
    # Rich metadata from MongoDB (NEW)
    stored_procedures: List[Dict[str, Any]] = field(default_factory=list)
    foreign_keys: Dict[str, List[Dict]] = field(default_factory=dict)  # table -> FK list
    related_tables: Dict[str, List[str]] = field(default_factory=dict)  # table -> related tables
    sample_values: Dict[str, Dict[str, List]] = field(default_factory=dict)  # table -> column -> samples
    table_summaries: Dict[str, str] = field(default_factory=dict)  # table -> LLM summary


class SQLGeneratorService:
    """
    Service for generating SQL from natural language.

    This service orchestrates the full text-to-SQL pipeline:
    1. Security validation of user input
    2. Schema context retrieval
    3. LLM-based SQL generation
    4. Schema validation with self-correction
    5. Security validation of generated SQL

    Usage:
        service = await get_sql_generator_service()
        result = await service.generate_sql(
            question="How many tickets were created today?",
            database="EWRCentral"
        )

        if result.success:
            print(f"Generated SQL: {result.sql}")
        else:
            print(f"Error: {result.error}")
    """

    _instance: Optional["SQLGeneratorService"] = None

    def __init__(self):
        self._mongodb = None
        self._llm_service = None
        self._security_service = None
        self._schema_validator = None
        self._rules_service = None  # MongoDB-based rules service
        self._initialized = False
        self._max_correction_attempts = 2

    @classmethod
    async def get_instance(cls) -> "SQLGeneratorService":
        """Get or create singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
            await cls._instance.initialize()
        return cls._instance

    async def initialize(self):
        """Initialize service dependencies."""
        if self._initialized:
            return

        # Import here to avoid circular imports
        from mongodb import get_mongodb_service
        from sql_security_service import get_security_service
        from schema_validator import get_schema_validator
        from services.llm_service import get_llm_service
        from services.sql_rules_service import get_sql_rules_service

        self._mongodb = get_mongodb_service()
        self._llm_service = await get_llm_service()
        self._security_service = get_security_service()
        self._schema_validator = await get_schema_validator(self._mongodb)
        self._rules_service = await get_sql_rules_service()

        self._initialized = True
        logger.info("SQLGeneratorService initialized")

    async def _load_schema_context(
        self,
        question: str,
        database: str,
    ) -> SchemaContext:
        """
        Load relevant schema context for the query.

        This method:
        1. Extracts explicitly mentioned table names from question
        2. Fetches explicit tables by exact match
        3. Performs semantic search for additional relevant tables
        4. Loads similar examples from MongoDB
        """
        context = SchemaContext()

        if not self._mongodb or not self._mongodb.is_initialized:
            await self._mongodb.initialize()

        # Extract explicitly mentioned tables from question
        table_patterns = [
            r'\bfrom\s+(?:the\s+)?(\w+)\b',
            r'\bjoin\s+(\w+)\b',
            r'\bthe\s+(\w+)\s+table\b',
            r'\b(\w+)\s+table\b',
            r'\binto\s+(\w+)\b',
            r'\bupdate\s+(\w+)\b'
        ]

        mentioned_tables: Set[str] = set()
        question_lower = question.lower()

        for pattern in table_patterns:
            for match in re.finditer(pattern, question, re.IGNORECASE):
                table_name = match.group(1)
                # Filter out common non-table words
                if table_name.lower() not in [
                    'the', 'a', 'an', 'this', 'that', 'select', 'where',
                    'and', 'or', 'top', 'all', 'from'
                ]:
                    mentioned_tables.add(table_name)

        if mentioned_tables:
            logger.debug(f"Explicitly mentioned tables: {mentioned_tables}")

        # Fetch explicit tables first
        explicit_schemas = []
        for table_name in mentioned_tables:
            try:
                schema = await self._mongodb.get_schema_by_table(database, table_name)
                if schema:
                    formatted = self._format_schema(schema)
                    if formatted:
                        explicit_schemas.append(formatted)
                        context.table_names.add(schema.get("table_name", table_name))
                        context.tables.append(schema)
            except Exception as e:
                logger.debug(f"Table {table_name} not found: {e}")

        # Semantic search for additional context
        try:
            semantic_results = await self._mongodb.search_schema_context(
                query=question,
                database=database,
                limit=20
            )

            for schema in semantic_results:
                table_name = schema.get("table_name", "")
                if table_name and table_name not in context.table_names:
                    formatted = self._format_schema(schema)
                    if formatted:
                        explicit_schemas.append(formatted)
                        context.table_names.add(table_name)
                        context.tables.append(schema)
        except Exception as e:
            logger.warning(f"Semantic schema search failed: {e}")

        # Combine schemas
        if explicit_schemas:
            context.formatted_schema = "\n---\n".join(explicit_schemas)
            logger.info(f"Loaded schema for {len(context.table_names)} tables")
        else:
            context.formatted_schema = "Schema information not available."

        # Load similar examples
        try:
            context.examples = await self._mongodb.search_sql_examples(
                query=question,
                database=database,
                limit=5
            )
            if context.examples:
                logger.debug(f"Found {len(context.examples)} similar examples")
        except Exception as e:
            logger.warning(f"Example search failed: {e}")

        # Load failed queries (for negative examples)
        try:
            context.failed_queries = await self._mongodb.search_failed_queries(
                query=question,
                database=database,
                limit=2
            )
        except Exception as e:
            logger.debug(f"Failed query search failed: {e}")

        # Load stored procedures (NEW)
        try:
            context.stored_procedures = await self._mongodb.search_stored_procedures(
                query=question,
                database=database,
                limit=5
            )
            if context.stored_procedures:
                logger.debug(f"Found {len(context.stored_procedures)} relevant stored procedures")
        except Exception as e:
            logger.debug(f"Stored procedure search failed: {e}")

        # Load views (PRIORITIZED - views have pre-joined human-readable columns)
        try:
            view_results = await self._mongodb.search_views(
                query=question,
                database=database,
                limit=10
            )
            for view in view_results:
                view_name = view.get("table_name", "")
                if view_name and view_name not in context.view_names:
                    context.views.append(view)
                    context.view_names.add(view_name)

            if context.views:
                # Format views separately for prominent placement in prompt
                view_formats = []
                for v in context.views:
                    formatted = self._format_schema(v, is_view=True)
                    if formatted:
                        view_formats.append(formatted)
                context.formatted_views = "\n---\n".join(view_formats)
                logger.info(f"Found {len(context.views)} relevant views (prioritized)")
        except Exception as e:
            logger.debug(f"View search failed: {e}")

        # Populate rich metadata dictionaries (NEW)
        for table in context.tables:
            table_name = table.get("table_name", "")
            if table_name:
                # Store foreign keys by table
                fks = table.get("foreign_keys", [])
                if fks:
                    context.foreign_keys[table_name] = fks

                # Store related tables
                related = table.get("related_tables", [])
                if related:
                    context.related_tables[table_name] = related

                # Store sample values
                samples = table.get("sample_values", {})
                if samples:
                    context.sample_values[table_name] = samples

                # Store table summaries
                summary = table.get("summary", "")
                if summary:
                    context.table_summaries[table_name] = summary

        return context

    def _format_schema(self, schema: Dict[str, Any], is_view: bool = False) -> str:
        """
        Format a schema dictionary into readable text for the prompt.

        NOW INCLUDES (from MongoDB metadata):
        - Foreign key relationships
        - Related tables
        - Sample values for key columns
        - VIEW indicator for pre-joined views
        """
        if not schema:
            return ""

        parts = []
        table_name = schema.get("table_name", schema.get("tableName", "Unknown"))
        schema_name = schema.get("schema", "dbo")

        # Check if this is a view (from MongoDB metadata or parameter)
        is_view_obj = is_view or schema.get("is_view", False) or schema.get("object_type") == "VIEW"

        if is_view_obj:
            parts.append(f"VIEW: {schema_name}.{table_name} (RECOMMENDED - pre-joined with human-readable columns)")
        else:
            parts.append(f"Table: {schema_name}.{table_name}")

        # Get sample values dict for this table
        sample_values = schema.get("sample_values", {})

        columns = schema.get("columns", [])
        if columns:
            parts.append("Columns:")
            for col in columns:
                if isinstance(col, dict):
                    col_name = col.get("name", col.get("column_name", ""))
                    col_type = col.get("type", col.get("data_type", ""))
                    nullable = " NOT NULL" if col.get("nullable") is False else ""
                    col_desc = f"  - {col_name} ({col_type}){nullable}"

                    # Add sample values if available (NEW)
                    samples = sample_values.get(col_name)
                    if samples:
                        sample_str = ", ".join(str(s) for s in samples[:3])
                        col_desc += f" [Examples: {sample_str}]"

                    parts.append(col_desc)
                else:
                    parts.append(f"  - {col}")

        primary_keys = schema.get("primary_keys", [])
        if primary_keys:
            parts.append(f"Primary Key: {', '.join(primary_keys)}")

        # Add foreign key relationships (NEW)
        fks = schema.get("foreign_keys", [])
        if fks:
            fk_lines = []
            for fk in fks:
                fk_col = fk.get("column", fk.get("from_column", ""))
                ref_table = fk.get("references_table", fk.get("to_table", ""))
                ref_col = fk.get("references_column", fk.get("to_column", ""))
                if fk_col and ref_table:
                    fk_lines.append(f"  - {fk_col} -> {ref_table}.{ref_col}")
            if fk_lines:
                parts.append("Foreign Keys:")
                parts.extend(fk_lines)

        # Add related tables (NEW)
        related = schema.get("related_tables", [])
        if related:
            parts.append(f"Related Tables: {', '.join(related)}")

        summary = schema.get("summary", "")
        if summary:
            parts.append(f"Description: {summary}")

        return "\n".join(parts)

    def _build_system_prompt(
        self,
        database: str,
        schema_context: SchemaContext,
        rules_context: str,
        conversation_history: Optional[List[Dict]] = None,
        exact_match: Optional[Dict] = None,
    ) -> str:
        """
        Build the system prompt for SQL generation.

        The prompt structure prioritizes:
        1. Exact rule matches
        2. Critical SQL rules
        3. Database schema
        4. Similar examples
        5. Conversation history
        """
        parts = [
            "You are a SQL Server T-SQL query expert. Generate ONLY valid T-SQL queries for Microsoft SQL Server.",
            "",
            "PRIORITY ORDER (follow strictly):",
            "1. CRITICAL SQL RULES - These are database-specific constraints. If a rule has an \"Example SQL\" that matches the user's question, USE THAT EXACT SQL.",
            "2. Database schema constraints",
            "3. Reference examples from similar queries (use for patterns only if no exact rule match)",
            "",
            f"Database: {database}",
            ""
        ]

        # Add exact match if found
        if exact_match:
            parts.extend([
                "=== EXACT MATCH FOUND - USE THIS SQL ===",
                f"Your question: \"{exact_match['question']}\"",
                f"Exact SQL to use: {exact_match['sql']}",
                "",
                "IMPORTANT: Return the EXACT SQL above without modification.",
                ""
            ])

        # Add rules context
        if rules_context:
            parts.append(rules_context)

        # Add aggregate query guidance
        question_lower = schema_context.formatted_schema.lower() if schema_context else ""
        # This will be populated by the caller based on the actual question

        # Add VIEWS first (PRIORITIZED - pre-joined with human-readable columns)
        if schema_context.formatted_views:
            parts.extend([
                "=== RECOMMENDED VIEWS (USE THESE FIRST) ===",
                "Views contain pre-joined data with human-readable column names.",
                "PREFER views over base tables to avoid complex JOINs and TypeID lookups.",
                "",
                schema_context.formatted_views,
                ""
            ])

        # Add schema (tables)
        parts.extend([
            "Available Tables Schema:",
            schema_context.formatted_schema,
            ""
        ])

        # Add table relationships from FK graph
        if schema_context.tables:
            relationship_context = build_relationship_context(schema_context.tables)
            if relationship_context:
                parts.append(relationship_context)
                parts.append("")

        # Add conversation history
        if conversation_history:
            parts.append("CONVERSATION HISTORY (use this context for follow-up questions):")
            for msg in conversation_history:
                if msg.get("role") == "user":
                    parts.append(f"User: {msg.get('content', '')}")
                elif msg.get("role") == "assistant":
                    parts.append(f"Assistant: {msg.get('content', '')}")
            parts.extend([
                "",
                "IMPORTANT: The current question may reference the conversation above. Use context from prior exchanges to understand follow-up questions.",
                ""
            ])

        # Add similar examples
        if schema_context.examples:
            parts.append("REFERENCE EXAMPLES (for patterns):")
            for ex in schema_context.examples[:3]:
                prompt = ex.get("metadata", {}).get("prompt", ex.get("prompt", ""))
                sql = ex.get("metadata", {}).get("sql", ex.get("sql", ""))
                if prompt and sql:
                    parts.append(f"Q: {prompt}")
                    parts.append(f"SQL: {sql}")
                    parts.append("")

        # Add stored procedures context (NEW)
        if schema_context.stored_procedures:
            parts.append("AVAILABLE STORED PROCEDURES:")
            parts.append("(Prefer stored procedures over raw SQL when they match the query intent)")
            parts.append("")
            for sp in schema_context.stored_procedures[:3]:
                proc_name = sp.get("procedure_name", "")
                params = sp.get("parameters", [])
                summary = sp.get("summary", "No description")
                tables = sp.get("tables_affected", [])

                # Format parameters
                if params:
                    param_strs = []
                    for p in params:
                        p_name = p.get("name", p.get("parameter_name", ""))
                        p_type = p.get("type", p.get("data_type", ""))
                        has_default = p.get("hasDefault", p.get("has_default", False))
                        if has_default:
                            param_strs.append(f"{p_name} {p_type} = <optional>")
                        else:
                            param_strs.append(f"{p_name} {p_type}")
                    param_str = ", ".join(param_strs)
                else:
                    param_str = ""

                parts.append(f"- {proc_name}({param_str})")
                parts.append(f"  Purpose: {summary}")
                if tables:
                    parts.append(f"  Tables: {', '.join(tables)}")
                parts.append("")

        # Add business rules injection (NEW)
        business_rules = self._get_business_rules(database, schema_context)
        if business_rules:
            parts.append("BUSINESS RULES (MUST FOLLOW):")
            for rule in business_rules:
                parts.append(f"- {rule}")
            parts.append("")

        parts.append("Generate ONLY the SQL query. Do not include explanations.")

        return "\n".join(parts)

    def _get_business_rules(self, database: str, schema_context: SchemaContext) -> List[str]:
        """
        Get business rules for a database.

        These are critical domain-specific rules that the LLM must follow
        to generate correct SQL.
        """
        rules = []

        # EWRCentral specific rules (hardcoded critical rules)
        if database.lower() == 'ewrcentral':
            rules.extend([
                "PREFER uvw_ views over base tables - views have pre-joined human-readable columns (e.g., uvw_CentralTickets has Status, AssignedTo, CompanyName instead of TypeIDs)",
                "For ticket queries: USE uvw_CentralTickets which already has Status, Priority, TicketType, AssignedTo, CompanyName as readable text",
                "For user queries: USE uvw_CentralUsers which has human-readable roles and assignments",
                "Use AddTicketDate (NOT CreateDate) for ticket creation date",
                "Use ExpireDate (NOT ExpirationDate) for license expiration",
                "Use AddCentralUserID for ticket creator (NOT CreateUserID)",
                "Only JOIN Types table if a view is not available for your query",
                "When counting tickets, use COUNT(*) not SELECT TOP with a number"
            ])

        # EWR Gin database rules
        if 'gin' in database.lower():
            rules.extend([
                "Bale records are in the Bales table",
                "Use GinTicketNumber for gin ticket lookups",
                "Module data is in Modules table with ModuleNumber as key"
            ])

        # Add any dynamically loaded rules from schema context
        # Future: Load from sql_knowledge MongoDB collection

        return rules

    def _extract_sql_from_response(self, response: str) -> str:
        """Extract clean SQL from LLM response."""
        sql = response.strip()

        # Remove markdown code blocks
        sql = re.sub(r'```sql\n?', '', sql)
        sql = re.sub(r'```\n?', '', sql)

        # Remove any text after common stop patterns that might slip through
        stop_patterns = [
            r'\n\nUser:.*',
            r'\n\nuser:.*',
            r'\nUser:.*',
            r'\nuser:.*',
            r'\n\nQuestion:.*',
            r'\nExplanation:.*',
            r'\nNote:.*',
            r'\n---.*',
        ]
        for pattern in stop_patterns:
            sql = re.sub(pattern, '', sql, flags=re.DOTALL | re.IGNORECASE)

        # Remove trailing semicolons (we add them if needed)
        sql = re.sub(r';+\s*$', '', sql)

        return sql.strip()

    async def generate_sql(
        self,
        question: str,
        database: str,
        conversation_history: Optional[List[Dict]] = None,
        max_tokens: int = 512,
        skip_validation: bool = False,
    ) -> SQLGenerationResult:
        """
        Generate SQL from natural language question.

        Args:
            question: Natural language question
            database: Target database name
            conversation_history: Previous conversation for context
            max_tokens: Maximum tokens for LLM response
            skip_validation: Skip schema validation (for trusted rules)

        Returns:
            SQLGenerationResult with generated SQL or error

        Pipeline:
        1. Security check on user prompt
        2. Check for exact rule match
        3. Load schema context
        4. Build LLM prompt
        5. Generate SQL
        6. Validate and self-correct if needed
        7. Security check on generated SQL
        """
        import time
        start_time = time.time()

        result = SQLGenerationResult(success=False)

        if not self._initialized:
            await self.initialize()

        # Step 1: Security validation of user prompt
        security_check = self._security_service.check_request(question)
        if not security_check.allowed:
            result.error = security_check.error
            result.security_blocked = True
            result.security_violations = [security_check.reason or "unknown"]
            logger.warning(f"Security blocked prompt: {security_check.reason}")
            return result

        sanitized_question = security_check.sanitized_prompt

        # Normalize database name
        from database_name_parser import normalize_database_name
        master_database = normalize_database_name(database)

        # Step 2: Check for exact rule match (from MongoDB)
        exact_match = await self._rules_service.find_exact_match(sanitized_question, master_database)
        if exact_match and exact_match.get("exact_match"):
            result.success = True
            result.sql = exact_match["sql"]
            result.is_exact_match = True
            result.rule_id = exact_match.get("rule_id")
            result.generation_time_ms = int((time.time() - start_time) * 1000)
            logger.info(f"Exact rule match: {result.rule_id}")
            return result

        # Step 3: Load schema context
        schema_context = await self._load_schema_context(sanitized_question, master_database)

        # Step 4: Build prompts (rules from MongoDB)
        rules_context = await self._rules_service.build_rules_context(
            sanitized_question,
            master_database,
            schema_context.table_names
        )

        system_prompt = self._build_system_prompt(
            database=database,
            schema_context=schema_context,
            rules_context=rules_context,
            conversation_history=conversation_history,
            exact_match=exact_match,
        )

        # Add aggregate query hints
        question_lower = sanitized_question.lower()
        user_prompt_extras = []

        if re.search(r'how many|count|total|number of', question_lower):
            user_prompt_extras.append(
                "AGGREGATE QUERY: Use COUNT(*) or COUNT(column), do NOT add TOP clause."
            )

        if re.search(r'most|least|highest|lowest|average|sum|maximum|minimum', question_lower):
            user_prompt_extras.append(
                "RANKING QUERY: Use GROUP BY with ORDER BY and aggregate functions."
            )

        user_prompt = f"User Question: {sanitized_question}\n\n"
        if user_prompt_extras:
            user_prompt += "\n".join(user_prompt_extras) + "\n\n"
        user_prompt += "SQL Query:"

        # Step 5: Generate SQL
        # Stop sequences to prevent LLM from generating past the SQL query
        sql_stop_sequences = [
            "\n\n",           # Double newline indicates end of SQL
            "\nUser:",        # Prevent chat-style continuation
            "\nuser:",        # Lowercase variant
            "User:",          # Without leading newline
            "user:",          # Lowercase without newline
            "\nQuestion:",    # Prevent repeating question format
            "\n---",          # Section separator
            "Explanation:",   # Prevent explanations
            "Note:",          # Prevent notes
        ]

        llm_result = await self._llm_service.generate(
            prompt=user_prompt,
            system=system_prompt,
            use_sql_model=True,
            max_tokens=max_tokens,
            use_cache=False,  # Don't cache SQL generation
            stop_sequences=sql_stop_sequences,
        )

        if not llm_result.success:
            result.error = llm_result.error
            result.generation_time_ms = int((time.time() - start_time) * 1000)
            return result

        # Extract SQL from response
        generated_sql = self._extract_sql_from_response(llm_result.response)
        result.token_usage = llm_result.token_usage
        result.model_used = llm_result.model

        # Apply auto-fixes from rules (from MongoDB)
        fixed_sql, applied_fixes = await self._rules_service.apply_auto_fixes(generated_sql, master_database)
        if applied_fixes:
            logger.info(f"Applied {len(applied_fixes)} auto-fixes")
            generated_sql = fixed_sql

        # Step 6: Schema validation with self-correction
        if not skip_validation and self._schema_validator:
            validation = self._schema_validator.validate_sql(generated_sql, master_database)

            correction_attempt = 0
            while not validation.get("valid") and correction_attempt < self._max_correction_attempts:
                correction_attempt += 1
                result.validation_attempts = correction_attempt

                # Get feedback for correction
                feedback = self._schema_validator.format_validation_feedback(
                    validation, master_database
                )

                if not feedback:
                    logger.warning("No correction feedback available")
                    break

                logger.info(f"Correction attempt {correction_attempt}: {len(validation.get('errors', []))} errors")

                # Regenerate with feedback
                correction_prompt = f"{user_prompt}\n\n{feedback}\n\nPlease generate the corrected SQL query:"

                correction_result = await self._llm_service.generate(
                    prompt=correction_prompt,
                    system=system_prompt,
                    use_sql_model=True,
                    temperature=0.1,  # Slightly higher temp for correction
                    use_cache=False,
                    stop_sequences=sql_stop_sequences,
                )

                if correction_result.success:
                    generated_sql = self._extract_sql_from_response(correction_result.response)

                    # Update token usage
                    result.token_usage["prompt_tokens"] += correction_result.token_usage.get("prompt_tokens", 0)
                    result.token_usage["response_tokens"] += correction_result.token_usage.get("response_tokens", 0)
                    result.token_usage["total_tokens"] = (
                        result.token_usage["prompt_tokens"] + result.token_usage["response_tokens"]
                    )

                    # Re-validate
                    validation = self._schema_validator.validate_sql(generated_sql, master_database)
                else:
                    logger.warning(f"Correction generation failed: {correction_result.error}")
                    break

            # Store final validation errors
            result.schema_errors = validation.get("errors", [])

        # Step 7: Security validation of generated SQL
        sql_security = self._security_service.validate_generated_sql(generated_sql)
        if not sql_security.safe:
            result.error = sql_security.error
            result.security_blocked = True
            result.security_violations = sql_security.violations
            result.sql = generated_sql  # Include for debugging
            logger.warning(f"Security blocked SQL: {sql_security.violations}")
            return result

        # Success
        result.success = True
        result.sql = generated_sql
        result.generation_time_ms = int((time.time() - start_time) * 1000)

        return result

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


# Singleton accessor
_sql_generator: Optional[SQLGeneratorService] = None


async def get_sql_generator_service() -> SQLGeneratorService:
    """Get or create the global SQL generator service instance."""
    global _sql_generator
    if _sql_generator is None:
        _sql_generator = SQLGeneratorService()
        await _sql_generator.initialize()
    return _sql_generator
