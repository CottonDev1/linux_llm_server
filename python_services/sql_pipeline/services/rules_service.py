"""
Rules Service Module

This service handles loading and matching SQL rules from MongoDB.
Rules are used to guide SQL generation and enforce database-specific constraints.
"""

import time
import re
from typing import Optional, List, Dict, Tuple
import logging
from core.log_utils import log_info

from motor.motor_asyncio import AsyncIOMotorDatabase
from sql_pipeline.models.rule_models import SQLRule, AutoFix, RuleExample

logger = logging.getLogger(__name__)


class RulesService:
    """
    Service for managing SQL rules stored in MongoDB.

    This service provides:
    - Loading rules by database and type
    - Matching rules based on keywords and table/column references
    - Caching rules for performance with TTL
    - Exact question matching for LLM bypass
    - Similarity-based rule matching

    Attributes:
        _instance: Singleton instance
        _cache: Cached rules by database
        _cache_timestamps: Cache timestamps for TTL
        _cache_ttl: Time-to-live for cache entries (seconds)
    """

    _instance: Optional["RulesService"] = None
    _cache: Dict[str, List[SQLRule]] = {}
    _cache_timestamps: Dict[str, float] = {}
    _cache_ttl: int = 60  # seconds
    _initialized: bool = False

    def __init__(self):
        """Initialize the rules service (use get_instance instead)."""
        self.db: Optional[AsyncIOMotorDatabase] = None
        self.collection_name = "sql_rules"

    @classmethod
    async def get_instance(cls) -> "RulesService":
        """
        Get or create the singleton instance of RulesService.

        Returns:
            RulesService: The singleton instance
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
        log_info("Rules Service", f"Initialized with collection: {self.collection_name}")

    async def get_rules(self, database: str, include_global: bool = True) -> List[SQLRule]:
        """
        Get all enabled rules for a database with caching.

        Args:
            database: Database name (e.g., "EWRCentral")
            include_global: Whether to include global rules (database="_global")

        Returns:
            List of SQLRule objects sorted by priority
        """
        # Normalize database name (case-insensitive)
        database_normalized = database.lower()
        cache_key = f"{database_normalized}:{'global' if include_global else 'local'}"

        # Check cache freshness
        now = time.time()
        if cache_key in self._cache:
            cache_age = now - self._cache_timestamps.get(cache_key, 0)
            if cache_age < self._cache_ttl:
                return self._cache[cache_key]

        # Query MongoDB
        rules = await self._fetch_rules_from_db(database, include_global)

        # Cache results
        self._cache[cache_key] = rules
        self._cache_timestamps[cache_key] = now

        return rules

    async def _fetch_rules_from_db(self, database: str, include_global: bool) -> List[SQLRule]:
        """
        Fetch rules from MongoDB.

        Args:
            database: Database name
            include_global: Whether to include global rules

        Returns:
            List of SQLRule objects
        """
        collection = self.db[self.collection_name]

        # Build query
        query = {"enabled": True}

        # Match specific database or global
        if include_global:
            query["database"] = {"$in": [database, "_global"]}
        else:
            query["database"] = database

        # Execute query
        cursor = collection.find(query).sort("priority", -1)
        documents = await cursor.to_list(length=None)

        # Convert to SQLRule objects
        rules = []
        for doc in documents:
            try:
                # SQLRule model handles _id conversion via model_validator
                rule = SQLRule(**doc)
                rules.append(rule)
            except Exception as e:
                logger.error(f"Failed to parse rule {doc.get('rule_id', doc.get('_id', 'unknown'))}: {e}")

        # Sort by priority: critical > high > normal
        priority_order = {"critical": 3, "high": 2, "normal": 1}
        rules.sort(key=lambda r: priority_order.get(r.priority, 0), reverse=True)

        logger.info(f"Loaded {len(rules)} rules for database '{database}' (include_global={include_global})")
        return rules

    async def find_exact_match(self, question: str, database: str) -> Optional[SQLRule]:
        """
        Find 100% match using example questions (case-insensitive).

        This allows bypassing the LLM for known questions.

        Args:
            question: Natural language question
            database: Target database

        Returns:
            SQLRule if exact match found, None otherwise
        """
        rules = await self.get_rules(database, include_global=True)

        # Normalize question (remove punctuation, lowercase)
        question_normalized = question.lower().strip()
        question_normalized = re.sub(r'[?!.,]', '', question_normalized)

        for rule in rules:
            if rule.example and rule.example.question:
                example_normalized = rule.example.question.lower().strip()
                example_normalized = re.sub(r'[?!.,]', '', example_normalized)

                if question_normalized == example_normalized:
                    logger.info(f"EXACT MATCH found! Rule '{rule.rule_id}': \"{rule.example.question}\"")
                    return rule

        return None

    async def find_similar_rules(
        self, question: str, database: str, threshold: float = 0.7
    ) -> List[Tuple[SQLRule, float]]:
        """
        Find rules with high keyword overlap using Jaccard similarity.

        Args:
            question: Natural language question
            database: Target database
            threshold: Minimum similarity score (0.0 to 1.0)

        Returns:
            List of (rule, similarity_score) tuples sorted by score DESC
        """
        rules = await self.get_rules(database, include_global=True)
        matches = []

        for rule in rules:
            if not rule.trigger_keywords:
                continue

            similarity = self._calculate_keyword_overlap(question, rule.trigger_keywords)
            if similarity >= threshold:
                matches.append((rule, similarity))

        # Sort by similarity score descending
        matches.sort(key=lambda x: x[1], reverse=True)

        if matches:
            logger.info(f"Found {len(matches)} similar rules above threshold {threshold}")

        return matches

    async def find_keyword_matches(self, question: str, database: str) -> List[SQLRule]:
        """
        Find rules by keyword overlap (any keyword match).

        Args:
            question: Natural language question
            database: Target database

        Returns:
            List of rules sorted by number of keyword matches
        """
        rules = await self.get_rules(database, include_global=True)
        matches = []

        question_lower = question.lower()

        for rule in rules:
            match_count = 0

            # Check trigger keywords
            if rule.trigger_keywords:
                for keyword in rule.trigger_keywords:
                    if keyword.lower() in question_lower:
                        match_count += 1

            # Check trigger columns
            if rule.trigger_columns:
                for column in rule.trigger_columns:
                    if column.lower() in question_lower:
                        match_count += 1

            if match_count > 0:
                matches.append((rule, match_count))

        # Sort by match count descending
        matches.sort(key=lambda x: x[1], reverse=True)

        logger.info(f"Found {len(matches)} rules with keyword matches")
        return [rule for rule, count in matches]

    async def get_auto_fixes(self, database: str) -> List[AutoFix]:
        """
        Get all auto-fix patterns for database (includes global rules).

        Args:
            database: Target database

        Returns:
            List of AutoFix objects
        """
        rules = await self.get_rules(database, include_global=True)

        auto_fixes = []
        for rule in rules:
            if rule.auto_fix:
                auto_fixes.append(rule.auto_fix)

        logger.info(f"Found {len(auto_fixes)} auto-fix patterns for '{database}'")
        return auto_fixes

    async def get_relevant_rules_for_tables(self, tables: List[str], database: str) -> List[SQLRule]:
        """
        Get rules that apply to specific tables.

        Args:
            tables: List of table names mentioned in query
            database: Target database

        Returns:
            List of rules that match the tables
        """
        if not tables:
            return []

        rules = await self.get_rules(database, include_global=True)
        matches = []

        # Normalize table names for comparison
        tables_lower = [t.lower() for t in tables]

        for rule in rules:
            if not rule.trigger_tables:
                continue

            # Check if any trigger table matches
            for trigger_table in rule.trigger_tables:
                if trigger_table.lower() in tables_lower:
                    matches.append(rule)
                    break

        logger.info(f"Found {len(matches)} rules for tables: {tables}")
        return matches

    def _calculate_keyword_overlap(self, question: str, keywords: List[str]) -> float:
        """
        Calculate Jaccard similarity between question words and keywords.

        Jaccard similarity = |intersection| / |union|

        Args:
            question: Natural language question
            keywords: List of keywords to match

        Returns:
            Similarity score between 0.0 and 1.0
        """
        # Tokenize question (lowercase, split on whitespace)
        question_words = set(question.lower().split())

        # Normalize keywords
        keyword_set = set(k.lower() for k in keywords)

        # Calculate Jaccard similarity
        intersection = question_words & keyword_set
        union = question_words | keyword_set

        if not union:
            return 0.0

        return len(intersection) / len(union)

    def invalidate_cache(self, database: Optional[str] = None):
        """
        Invalidate cache for specific database or all databases.

        Args:
            database: Database name to invalidate, or None for all
        """
        if database is None:
            # Clear all cache
            self._cache.clear()
            self._cache_timestamps.clear()
            logger.info("Cleared all cache")
        else:
            # Clear specific database cache
            database_normalized = database.lower()
            keys_to_remove = [k for k in self._cache.keys() if k.startswith(database_normalized)]
            for key in keys_to_remove:
                self._cache.pop(key, None)
                self._cache_timestamps.pop(key, None)
            logger.info(f"Cleared cache for database '{database}'")
