"""
Schema Service Module

This service handles retrieving and caching database schema information
for use in SQL generation prompts.
"""

import time
import re
from typing import Optional, Dict, List, Tuple
import logging
from core.log_utils import log_info

from motor.motor_asyncio import AsyncIOMotorDatabase
from sql_pipeline.models.query_models import (
    SchemaInfo,
    TableInfo,
    ColumnInfo,
    ForeignKeyInfo,
)

from database_name_parser import normalize_database_name

logger = logging.getLogger(__name__)


class SchemaService:
    """
    Service for retrieving and caching database schema information from MongoDB.

    This service provides:
    - Loading full schema from MongoDB (sql_schema_context collection)
    - Intelligent schema compression based on question keywords
    - Table relevance scoring and filtering
    - Schema-to-prompt formatting for LLM context
    - Caching with TTL for performance

    Attributes:
        _instance: Singleton instance
        _schema_cache: Cached schemas by database
        _cache_timestamps: Cache timestamps for TTL
        _cache_ttl: Time-to-live for cache entries (seconds)
    """

    _instance: Optional["SchemaService"] = None
    _schema_cache: Dict[str, Tuple[SchemaInfo, float]] = {}
    _cache_ttl: int = 300  # 5 minutes
    _initialized: bool = False

    def __init__(self):
        """Initialize the schema service (use get_instance instead)."""
        self.db: Optional[AsyncIOMotorDatabase] = None
        self.collection_name = "sql_schema_context"

    @classmethod
    async def get_instance(cls) -> "SchemaService":
        """
        Get or create the singleton instance of SchemaService.

        Returns:
            SchemaService: The singleton instance
        """
        if cls._instance is None:
            cls._instance = cls()
            await cls._instance._initialize()
        return cls._instance

    async def _initialize(self):
        """Initialize the MongoDB connection."""
        if self._initialized:
            return

        # Import here to avoid circular dependency
        from mongodb import MongoDBService

        # Get MongoDB service instance
        mongo_service = MongoDBService.get_instance()
        if not mongo_service.is_initialized:
            await mongo_service.initialize()

        self.db = mongo_service.db
        self._initialized = True
        log_info("Schema Service", f"Initialized with collection: {self.collection_name}")

    async def get_full_schema(self, database: str) -> SchemaInfo:
        """
        Get complete schema for a database with caching.

        For EWR databases (e.g., EWR.Gin.CustomerName), normalizes to base product
        (e.g., EWR.Gin) since all customer databases share the same schema.

        Args:
            database: Database name (e.g., "EWRCentral" or "EWR.Gin.Chad")

        Returns:
            SchemaInfo object with all tables and their metadata
        """
        # Normalize database name (EWR databases use base product name)
        database_normalized = normalize_database_name(database)
        logger.debug(f"Schema lookup: '{database}' -> '{database_normalized}'")

        # Check cache freshness
        now = time.time()
        if database_normalized in self._schema_cache:
            cached_schema, cache_time = self._schema_cache[database_normalized]
            cache_age = now - cache_time
            if cache_age < self._cache_ttl:
                logger.debug(f"Cache hit for database '{database}' (age: {cache_age:.1f}s)")
                return cached_schema

        # Query MongoDB
        schema = await self._fetch_schema_from_db(database_normalized)

        # Cache results
        self._schema_cache[database_normalized] = (schema, now)
        logger.info(f"Cached schema for '{database}' ({len(schema.tables)} tables)")

        return schema

    async def _fetch_schema_from_db(self, database: str) -> SchemaInfo:
        """
        Fetch schema from MongoDB.

        Args:
            database: Normalized database name

        Returns:
            SchemaInfo object
        """
        collection = self.db[self.collection_name]

        # Query all tables for this database
        cursor = collection.find({"database": database})
        documents = await cursor.to_list(length=None)

        tables = []
        for doc in documents:
            try:
                # Convert MongoDB document to TableInfo
                table = self._document_to_table_info(doc)
                tables.append(table)
            except Exception as e:
                logger.error(f"Failed to parse table {doc.get('table_name', 'unknown')}: {e}")

        schema = SchemaInfo(
            database=database,
            tables=tables,
            updated_at=doc.get("updated_at") if documents else None
        )

        logger.info(f"Loaded {len(tables)} tables for database '{database}'")
        return schema

    def _document_to_table_info(self, doc: dict) -> TableInfo:
        """
        Convert MongoDB document to TableInfo model.

        Args:
            doc: MongoDB document from sql_schema_context collection

        Returns:
            TableInfo object
        """
        # Parse columns
        columns = []
        for col in doc.get("columns", []):
            columns.append(ColumnInfo(
                name=col["name"],
                type=col["type"],
                nullable=col.get("nullable", True),
                is_pk=col["name"] in doc.get("primary_keys", []),
                default_value=col.get("defaultValue")
            ))

        # Parse foreign keys
        foreign_keys = []
        for fk in doc.get("foreign_keys", []):
            foreign_keys.append(ForeignKeyInfo(
                name=fk.get("name", ""),
                column=fk.get("column", ""),
                referenced_table=fk.get("referencedTable", ""),
                referenced_column=fk.get("referencedColumn", "")
            ))

        # Extract schema and table name from full name (e.g., "dbo.CentralTickets")
        table_name = doc.get("table_name", "")
        schema = doc.get("schema", "dbo")
        if "." in table_name and not schema:
            parts = table_name.split(".")
            schema = parts[0]
            table_name = parts[1] if len(parts) > 1 else table_name

        return TableInfo(
            name=table_name.split(".")[-1],  # Just the table name
            db_schema=schema,
            full_name=doc.get("table_name", ""),
            columns=columns,
            primary_keys=doc.get("primary_keys", []),
            foreign_keys=foreign_keys,
            related_tables=doc.get("related_tables", []),
            sample_values=doc.get("sample_values", {}),
            summary=doc.get("summary"),
            purpose=doc.get("purpose"),
            keywords=doc.get("keywords", [])
        )

    async def get_table_schema(self, database: str, table_name: str) -> Optional[TableInfo]:
        """
        Get schema for a specific table.

        Args:
            database: Database name
            table_name: Table name (with or without schema prefix)

        Returns:
            TableInfo if found, None otherwise
        """
        schema = await self.get_full_schema(database)

        # Normalize table name for comparison
        table_name_normalized = table_name.lower()

        for table in schema.tables:
            if (table.full_name.lower() == table_name_normalized or
                table.name.lower() == table_name_normalized):
                return table

        logger.warning(f"Table '{table_name}' not found in database '{database}'")
        return None

    async def get_relevant_schema(
        self,
        database: str,
        question: str,
        max_tables: int = 10,
        use_hybrid_retrieval: bool = True,
        required_tables: Optional[List[str]] = None
    ) -> SchemaInfo:
        """
        Get compressed schema with only relevant tables based on question.

        This is the key method for reducing context size while maintaining relevance.

        When use_hybrid_retrieval=True (default):
        - Uses Reciprocal Rank Fusion (RRF) combining semantic and keyword search
        - Better matches for CamelCase table/column names
        - Leverages MongoDB vector search and regex matching

        When use_hybrid_retrieval=False (fallback):
        - Uses local keyword extraction and scoring
        - Scores tables based on keyword matches in table names, column names, etc.

        Args:
            database: Database name
            question: Natural language question
            max_tables: Maximum number of tables to include
            use_hybrid_retrieval: Use RRF-based hybrid retrieval (recommended)
            required_tables: Tables that MUST be included (from rule trigger_tables)

        Returns:
            SchemaInfo with only relevant tables
        """
        if use_hybrid_retrieval:
            try:
                return await self._get_schema_hybrid(database, question, max_tables, required_tables)
            except Exception as e:
                logger.warning(f"Hybrid retrieval failed, falling back to keyword: {e}")
                # Fall through to keyword-based approach

        # Fallback: local keyword extraction and scoring
        full_schema = await self.get_full_schema(database)

        # Extract keywords from question
        keywords = self._extract_keywords(question)

        # Score and filter tables
        relevant_tables = self.compress_schema(full_schema, keywords, max_tables)

        return SchemaInfo(
            database=database,
            tables=relevant_tables,
            updated_at=full_schema.updated_at
        )

    async def _get_schema_hybrid(
        self,
        database: str,
        question: str,
        max_tables: int = 10,
        required_tables: Optional[List[str]] = None
    ) -> SchemaInfo:
        """
        Get relevant schema using hybrid retrieval (semantic + keyword search).

        Uses MongoDB's hybrid_schema_retrieval which combines:
        1. Semantic search via vector embeddings
        2. Keyword search via regex on table/column names
        3. Reciprocal Rank Fusion (RRF) for result merging

        Args:
            database: Database name
            question: Natural language question
            max_tables: Maximum number of tables to include
            required_tables: Tables that MUST be included (from rule trigger_tables)

        Returns:
            SchemaInfo with relevant tables
        """
        from mongodb import MongoDBService

        # Get MongoDB service instance
        mongo_service = MongoDBService.get_instance()
        if not mongo_service.is_initialized:
            await mongo_service.initialize()

        # Use hybrid retrieval from MongoDB service
        results = await mongo_service.hybrid_schema_retrieval(
            query=question,
            database=database,
            limit=max_tables
        )

        # Convert results to TableInfo objects
        tables = []
        table_names_lower = set()
        for doc in results:
            try:
                table = self._document_to_table_info(doc)
                tables.append(table)
                # Track names (without dbo. prefix) for deduplication
                table_names_lower.add(table.name.lower())
                table_names_lower.add(table.full_name.lower())
            except Exception as e:
                table_name = doc.get("table_name", "unknown")
                logger.warning(f"Failed to convert table {table_name}: {e}")

        # PRIORITY: Put required tables from rules at the TOP of the list
        if required_tables:
            priority_tables = []
            for table_name in required_tables:
                # Check if already in results
                already_present = False
                for existing in tables:
                    if existing.name.lower() == table_name.lower() or existing.full_name.lower() == f"dbo.{table_name}".lower():
                        # Move to priority list and remove from regular list
                        priority_tables.append(existing)
                        tables.remove(existing)
                        already_present = True
                        break

                if not already_present:
                    # Fetch the required table
                    try:
                        table = await self.get_table_schema(database, table_name)
                        if table:
                            priority_tables.append(table)
                            logger.info(f"Added priority table: {table.full_name}")
                    except Exception as e:
                        logger.warning(f"Failed to fetch required table {table_name}: {e}")

            # Put priority tables FIRST, then the rest
            if priority_tables:
                logger.info(f"Priority tables (first {len(priority_tables)}): {[t.full_name for t in priority_tables]}")
                tables = priority_tables + tables

        # Log ALL tables being sent to LLM for debugging
        all_tables = [t.full_name for t in tables]
        logger.info(
            f"Hybrid retrieval: {len(tables)} tables for database '{database}': {all_tables}"
        )

        return SchemaInfo(
            database=database,
            tables=tables,
            updated_at=None  # Hybrid results don't have a single timestamp
        )

    def _extract_keywords(self, question: str) -> List[str]:
        """
        Extract meaningful keywords from question.

        Removes common SQL words and short words.

        Args:
            question: Natural language question

        Returns:
            List of keywords (lowercase)
        """
        # Common SQL/question words to ignore
        stop_words = {
            "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
            "of", "with", "by", "from", "as", "is", "was", "are", "were", "be",
            "has", "have", "had", "do", "does", "did", "will", "would", "could",
            "should", "may", "might", "can", "show", "get", "find", "list", "what",
            "when", "where", "how", "why", "who", "which", "all", "any", "some",
            "select", "from", "where", "order", "group", "by", "me", "my"
        }

        # Tokenize and clean
        words = re.findall(r'\b[a-z]+\b', question.lower())

        # Filter stop words and short words
        keywords = [w for w in words if w not in stop_words and len(w) > 2]

        return keywords

    def compress_schema(
        self,
        full_schema: SchemaInfo,
        keywords: List[str],
        max_tables: int = 10
    ) -> List[TableInfo]:
        """
        Compress schema by scoring and filtering tables based on keyword relevance.

        Scoring factors:
        - Table name matches (weight: 10)
        - Column name matches (weight: 5)
        - Keyword matches (weight: 8)
        - Summary/purpose matches (weight: 3)
        - Foreign key relationships to other relevant tables (weight: 2)

        Args:
            full_schema: Complete schema
            keywords: List of keywords from question
            max_tables: Maximum tables to return

        Returns:
            List of most relevant TableInfo objects
        """
        if not keywords:
            # No keywords - return top N tables by column count (most important tables)
            sorted_tables = sorted(
                full_schema.tables,
                key=lambda t: len(t.columns),
                reverse=True
            )
            return sorted_tables[:max_tables]

        # Score each table
        scored_tables: List[Tuple[TableInfo, float]] = []

        for table in full_schema.tables:
            score = 0.0

            # Score table name matches
            table_name_lower = table.full_name.lower()

            # NOTE: View prioritization disabled - only base tables exist in schema
            # If views are added later, uncomment the following:
            # if table_name_lower.startswith('uvw_') or table_name_lower.startswith('dbo.uvw_'):
            #     score += 25.0  # Strong bonus for uvw_ views
            # elif table_name_lower.startswith('vw_') or table_name_lower.startswith('dbo.vw_'):
            #     score += 20.0  # Bonus for vw_ views

            for keyword in keywords:
                if keyword in table_name_lower:
                    score += 10.0

            # Score column name matches
            for column in table.columns:
                column_name_lower = column.name.lower()
                for keyword in keywords:
                    if keyword in column_name_lower:
                        score += 5.0

            # Score keyword matches (from LLM-generated keywords)
            for table_keyword in table.keywords:
                table_keyword_lower = table_keyword.lower()
                for keyword in keywords:
                    if keyword in table_keyword_lower or table_keyword_lower in keyword:
                        score += 8.0

            # Score summary/purpose matches
            searchable_text = " ".join([
                table.summary or "",
                table.purpose or ""
            ]).lower()
            for keyword in keywords:
                if keyword in searchable_text:
                    score += 3.0

            if score > 0:
                scored_tables.append((table, score))

        # Sort by score descending
        scored_tables.sort(key=lambda x: x[1], reverse=True)

        # Get top N tables
        top_tables = [table for table, score in scored_tables[:max_tables]]

        # Add related tables via FK if we have room
        if len(top_tables) < max_tables:
            related_table_names = set()
            for table in top_tables:
                for fk in table.foreign_keys:
                    related_table_names.add(fk.referenced_table.lower())

            # Find related tables not already included
            remaining_slots = max_tables - len(top_tables)
            for table in full_schema.tables:
                if remaining_slots <= 0:
                    break
                if table not in top_tables and table.full_name.lower() in related_table_names:
                    top_tables.append(table)
                    remaining_slots -= 1

        logger.info(f"Compressed schema from {len(full_schema.tables)} to {len(top_tables)} tables")
        return top_tables

    def format_schema_for_prompt(
        self,
        tables: List[TableInfo],
        include_samples: bool = False
    ) -> str:
        """
        Format schema as concise markdown for LLM prompt context.

        Example format:
        ```
        ## dbo.CentralTickets
        Summary: Tracks customer support tickets and requests
        - CentralTicketID (int, PK) - Unique ticket identifier
        - Subject (nvarchar(500), nullable) - Ticket subject line
        - AddTicketDate (datetimeoffset) - Creation timestamp
        - AssignedCentralUserID (int, FK → CentralUsers.CentralUserID) - Assigned user

        Related Tables: CentralUsers, CentralTicketNotes
        ```

        Args:
            tables: List of TableInfo objects
            include_samples: Whether to include sample values

        Returns:
            Markdown-formatted schema string
        """
        if not tables:
            return "No schema information available."

        lines = []
        lines.append("# Database Schema\n")

        for table in tables:
            # Table header with summary
            lines.append(f"## {table.full_name}")
            if table.summary:
                lines.append(f"*{table.summary}*")
            lines.append("")

            # Columns
            for column in table.columns:
                col_parts = [f"- **{column.name}**"]
                col_parts.append(f"({column.type}")

                # Add PK/FK markers
                markers = []
                if column.is_pk:
                    markers.append("PK")

                # Check if column is part of FK
                fk_refs = [fk for fk in table.foreign_keys if fk.column == column.name]
                if fk_refs:
                    for fk in fk_refs:
                        markers.append(f"FK → {fk.referenced_table}.{fk.referenced_column}")

                if markers:
                    col_parts.append(", " + ", ".join(markers))

                col_parts.append(")")

                # Add nullable
                if column.nullable and not column.is_pk:
                    col_parts.append(" - nullable")

                lines.append("".join(col_parts))

            # Sample values if requested
            if include_samples and table.sample_values:
                lines.append("\n*Sample Values:*")
                for col_name, values in list(table.sample_values.items())[:3]:  # Max 3 columns
                    values_str = ", ".join([f"`{v}`" for v in values[:5]])  # Max 5 values
                    lines.append(f"  - {col_name}: {values_str}")

            # Related tables
            if table.related_tables:
                related = ", ".join(table.related_tables[:5])  # Max 5
                lines.append(f"\n*Related Tables:* {related}")

            lines.append("")  # Blank line between tables

        return "\n".join(lines)

    def invalidate_cache(self, database: Optional[str] = None):
        """
        Invalidate cache for specific database or all databases.

        Args:
            database: Database name to invalidate, or None for all
        """
        if database is None:
            # Clear all cache
            self._schema_cache.clear()
            logger.info("Cleared all schema cache")
        else:
            # Clear specific database cache (use same normalization as get_full_schema)
            database_normalized = normalize_database_name(database)
            self._schema_cache.pop(database_normalized, None)
            logger.info(f"Cleared schema cache for database '{database}'")

    async def validate_columns(
        self,
        sql: str,
        database: str
    ) -> Dict[str, any]:
        """
        Validate that all columns referenced in SQL exist in the database schema.

        Args:
            sql: The SQL query to validate
            database: Database name to validate against

        Returns:
            Dict with validation results:
            - valid: bool - True if all columns are valid
            - invalid_columns: List of columns that don't exist
            - suggestions: Dict mapping invalid columns to suggested corrections
            - tables_checked: List of tables that were validated
        """
        start_time = time.time()

        # Get full schema for this database
        schema = await self.get_full_schema(database)
        if not schema:
            return {
                "valid": True,  # Can't validate without schema
                "invalid_columns": [],
                "suggestions": {},
                "tables_checked": [],
                "validation_time_ms": round((time.time() - start_time) * 1000, 2),
                "message": "No schema available for validation"
            }

        # Build lookup of table -> columns
        table_columns: Dict[str, set] = {}
        all_columns: set = set()

        for table in schema.tables:
            table_name_lower = table.name.lower()
            full_name_lower = table.full_name.lower() if table.full_name else table_name_lower

            columns_set = {col.name.lower() for col in table.columns}
            table_columns[table_name_lower] = columns_set
            table_columns[full_name_lower] = columns_set

            # Also map with schema prefix
            if table.db_schema:
                schema_table = f"{table.db_schema.lower()}.{table_name_lower}"
                table_columns[schema_table] = columns_set

            all_columns.update(columns_set)

        # Extract table aliases from SQL
        alias_pattern = r'(?:FROM|JOIN)\s+(?:dbo\.)?(\w+)(?:\s+(?:AS\s+)?(\w+))?'
        aliases: Dict[str, str] = {}  # alias -> table_name
        for match in re.finditer(alias_pattern, sql, re.IGNORECASE):
            table_name = match.group(1).lower()
            alias = match.group(2).lower() if match.group(2) else table_name
            aliases[alias] = table_name

        # Extract column references (alias.column or just column)
        # Pattern: word.word or just word in SELECT, WHERE, ON, GROUP BY, ORDER BY contexts
        column_pattern = r'(?:SELECT|WHERE|ON|AND|OR|GROUP\s+BY|ORDER\s+BY|,)\s+(?:DISTINCT\s+)?(?:COUNT|SUM|AVG|MAX|MIN|CAST)?\s*\(?\s*(\w+)\.(\w+)'

        invalid_columns = []
        suggestions = {}
        tables_checked = set()

        # Find table.column patterns
        for match in re.finditer(column_pattern, sql, re.IGNORECASE):
            table_or_alias = match.group(1).lower()
            column_name = match.group(2).lower()

            # Resolve alias to table name
            actual_table = aliases.get(table_or_alias, table_or_alias)
            tables_checked.add(actual_table)

            # Check if table exists in schema
            if actual_table in table_columns:
                valid_columns = table_columns[actual_table]
                if column_name not in valid_columns:
                    # Column doesn't exist - find suggestions
                    invalid_columns.append(f"{table_or_alias}.{match.group(2)}")

                    # Find similar column names (Levenshtein-ish matching)
                    best_match = None
                    best_score = 0
                    for valid_col in valid_columns:
                        # Simple similarity: common characters ratio
                        common = len(set(column_name) & set(valid_col))
                        score = common / max(len(column_name), len(valid_col))
                        if score > best_score and score > 0.5:
                            best_score = score
                            best_match = valid_col

                    if best_match:
                        suggestions[f"{table_or_alias}.{match.group(2)}"] = best_match

        validation_time_ms = round((time.time() - start_time) * 1000, 2)

        result = {
            "valid": len(invalid_columns) == 0,
            "invalid_columns": invalid_columns,
            "suggestions": suggestions,
            "tables_checked": list(tables_checked),
            "validation_time_ms": validation_time_ms,
            "message": "All columns validated" if not invalid_columns else f"Found {len(invalid_columns)} invalid column(s)"
        }

        if invalid_columns:
            logger.warning(f"Column validation failed: {invalid_columns}")
        else:
            logger.info(f"Column validation passed for {len(tables_checked)} tables in {validation_time_ms}ms")

        return result

    async def close(self):
        """Clean up resources."""
        self._schema_cache.clear()
        logger.info("SchemaService closed")
