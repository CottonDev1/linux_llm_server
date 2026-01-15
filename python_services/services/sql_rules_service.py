"""
SQL Rules Service - MongoDB-based rule management for text-to-SQL generation.

This service provides:
- Exact match shortcuts (bypass LLM for known queries)
- Contextual guidance for LLM prompts
- Auto-fix patterns for common SQL errors

Rules are stored in MongoDB 'sql_rules' collection with the schema:
{
    "rule_id": "unique-rule-id",
    "scope": "global" | "EWRCentral" | "Gin" | etc.,
    "type": "constraint" | "assistance",
    "priority": "critical" | "normal",
    "description": "Human readable description",
    "rule_text": "Guidance for LLM",
    "trigger_keywords": ["keyword1", "keyword2"],
    "trigger_tables": ["Table1", "Table2"],
    "trigger_columns": ["Column1", "Column2"],
    "auto_fix": { "pattern": "regex", "replacement": "string" },
    "example": { "question": "...", "sql": "..." },
    "is_active": true
}
"""

import logging
import re
from typing import Any, Dict, List, Optional, Set, Tuple

from motor.motor_asyncio import AsyncIOMotorDatabase

logger = logging.getLogger(__name__)


class SQLRulesService:
    """
    Async service for managing SQL rules from MongoDB.

    SQL rules provide:
    - Exact match shortcuts (bypass LLM for known queries)
    - Contextual guidance for LLM
    - Auto-fix patterns for common errors
    """

    def __init__(self, db: AsyncIOMotorDatabase, collection_name: str = "sql_rules"):
        self.db = db
        self.collection_name = collection_name
        self._collection = None

    @property
    def collection(self):
        """Get the rules collection."""
        if self._collection is None:
            self._collection = self.db[self.collection_name]
        return self._collection

    async def find_exact_match(
        self, question: str, database: str
    ) -> Optional[Dict[str, Any]]:
        """
        Find exact match for question in rules.

        Checks database-specific rules first, then global rules.
        Returns rule with SQL if found, None otherwise.
        """
        question_lower = question.lower().strip()

        # Search for exact match in database-specific rules first
        rule = await self.collection.find_one(
            {
                "scope": {"$regex": f"^{re.escape(database)}$", "$options": "i"},
                "example.question": {"$regex": f"^{re.escape(question_lower)}$", "$options": "i"},
                "is_active": True,
            }
        )

        if rule and rule.get("example", {}).get("sql"):
            logger.info(f"Found exact match: {rule['rule_id']} for database {database}")
            return {
                "rule_id": rule.get("rule_id", "unknown"),
                "sql": rule["example"]["sql"],
                "question": rule["example"].get("question", ""),
                "exact_match": True,
            }

        # Check global rules
        rule = await self.collection.find_one(
            {
                "scope": "global",
                "example.question": {"$regex": f"^{re.escape(question_lower)}$", "$options": "i"},
                "is_active": True,
            }
        )

        if rule and rule.get("example", {}).get("sql"):
            logger.info(f"Found global exact match: {rule['rule_id']}")
            return {
                "rule_id": rule.get("rule_id", "unknown"),
                "sql": rule["example"]["sql"],
                "question": rule["example"].get("question", ""),
                "exact_match": True,
            }

        return None

    async def build_rules_context(
        self,
        question: str,
        database: str,
        table_names: Optional[Set[str]] = None,
        max_rules: int = 10,
    ) -> str:
        """
        Build rules context for LLM prompt.

        Finds relevant rules based on:
        - Trigger keywords matching the question
        - Trigger tables matching the detected tables

        Returns formatted string with relevant rules for the LLM prompt.
        """
        question_lower = question.lower()
        table_names_lower = {t.lower() for t in (table_names or set())}

        # Build query for relevant rules
        # Match rules for this database or global rules
        base_query = {
            "$or": [
                {"scope": {"$regex": f"^{re.escape(database)}$", "$options": "i"}},
                {"scope": "global"},
            ],
            "is_active": True,
        }

        # Fetch all potentially relevant rules
        relevant_rules = []
        async for rule in self.collection.find(base_query):
            # Check if rule matches by keywords or tables
            keywords = rule.get("trigger_keywords", [])
            trigger_tables = rule.get("trigger_tables", [])

            keyword_match = any(kw.lower() in question_lower for kw in keywords)
            table_match = any(t.lower() in table_names_lower for t in trigger_tables)

            if keyword_match or table_match:
                # Add priority score for sorting (critical rules first)
                priority_score = 0 if rule.get("priority") == "critical" else 1
                relevant_rules.append((priority_score, rule))

        if not relevant_rules:
            logger.debug(f"No relevant rules found for question: {question[:50]}...")
            return ""

        # Sort by priority and limit
        relevant_rules.sort(key=lambda x: x[0])
        relevant_rules = [r[1] for r in relevant_rules[:max_rules]]

        logger.info(f"Found {len(relevant_rules)} relevant rules for prompt context")

        # Format rules for prompt
        lines = ["CRITICAL SQL RULES (follow these strictly):\n"]

        for i, rule in enumerate(relevant_rules, 1):
            lines.append(f"{i}. {rule.get('description', 'No description')}")

            rule_text = rule.get("rule_text")
            if rule_text:
                lines.append(f"   Rule: {rule_text}")

            example = rule.get("example", {})
            if example.get("sql"):
                lines.append(f"   Example SQL: {example['sql']}")

            lines.append("")

        return "\n".join(lines)

    async def apply_auto_fixes(
        self, sql: str, database: str
    ) -> Tuple[str, List[Dict]]:
        """
        Apply auto-fix patterns to SQL.

        Fetches rules with auto_fix patterns for this database and global,
        then applies each regex pattern.

        Returns (fixed_sql, list of applied fixes).
        """
        fixed_sql = sql
        applied_fixes = []

        # Fetch rules with auto_fix patterns
        query = {
            "$or": [
                {"scope": {"$regex": f"^{re.escape(database)}$", "$options": "i"}},
                {"scope": "global"},
            ],
            "auto_fix": {"$ne": None},
            "is_active": True,
        }

        async for rule in self.collection.find(query):
            auto_fix = rule.get("auto_fix", {})
            pattern = auto_fix.get("pattern")
            replacement = auto_fix.get("replacement", "")

            if pattern:
                try:
                    new_sql, count = re.subn(
                        pattern, replacement, fixed_sql, flags=re.IGNORECASE
                    )
                    if count > 0:
                        fixed_sql = new_sql
                        applied_fixes.append(
                            {
                                "rule_id": rule.get("rule_id", "unknown"),
                                "pattern": pattern,
                                "count": count,
                            }
                        )
                        logger.debug(
                            f"Applied auto-fix {rule['rule_id']}: {count} replacements"
                        )
                except re.error as e:
                    logger.warning(
                        f"Invalid regex in rule {rule.get('rule_id')}: {e}"
                    )

        if applied_fixes:
            logger.info(f"Applied {len(applied_fixes)} auto-fixes to SQL")

        return fixed_sql, applied_fixes

    async def get_all_rules(
        self, scope: Optional[str] = None, include_inactive: bool = False
    ) -> List[Dict]:
        """
        Get all rules, optionally filtered by scope.

        Args:
            scope: Filter by scope (e.g., "EWRCentral", "global")
            include_inactive: Include inactive rules

        Returns list of rule documents.
        """
        query = {}
        if scope:
            query["scope"] = {"$regex": f"^{re.escape(scope)}$", "$options": "i"}
        if not include_inactive:
            query["is_active"] = True

        rules = []
        async for rule in self.collection.find(query).sort("rule_id", 1):
            rules.append(rule)

        return rules

    async def get_rule_by_id(self, rule_id: str) -> Optional[Dict]:
        """Get a specific rule by its ID."""
        return await self.collection.find_one({"rule_id": rule_id})

    async def update_rule(self, rule_id: str, updates: Dict) -> bool:
        """
        Update a rule by ID.

        Args:
            rule_id: The rule ID to update
            updates: Dict of fields to update

        Returns True if rule was updated, False if not found.
        """
        from datetime import datetime, timezone

        updates["updated_at"] = datetime.now(timezone.utc)

        result = await self.collection.update_one(
            {"rule_id": rule_id}, {"$set": updates}
        )

        if result.modified_count > 0:
            logger.info(f"Updated rule: {rule_id}")
            return True
        return False

    async def add_rule(self, rule: Dict) -> str:
        """
        Add a new rule.

        Args:
            rule: Rule document (must include rule_id, scope, type, rule_text)

        Returns the inserted rule_id.
        Raises ValueError if rule_id already exists.
        """
        from datetime import datetime, timezone

        now = datetime.now(timezone.utc)
        rule["created_at"] = now
        rule["updated_at"] = now
        rule["is_active"] = rule.get("is_active", True)
        rule["version"] = 1

        try:
            await self.collection.insert_one(rule)
            logger.info(f"Added new rule: {rule['rule_id']}")
            return rule["rule_id"]
        except Exception as e:
            if "duplicate key" in str(e).lower():
                raise ValueError(f"Rule with ID '{rule['rule_id']}' already exists")
            raise

    async def delete_rule(self, rule_id: str, hard_delete: bool = False) -> bool:
        """
        Delete a rule by ID.

        Args:
            rule_id: The rule ID to delete
            hard_delete: If True, permanently delete. If False, soft delete.

        Returns True if rule was deleted, False if not found.
        """
        if hard_delete:
            result = await self.collection.delete_one({"rule_id": rule_id})
            if result.deleted_count > 0:
                logger.info(f"Hard deleted rule: {rule_id}")
                return True
        else:
            return await self.update_rule(rule_id, {"is_active": False})

        return False

    async def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the rules collection."""
        pipeline = [
            {"$facet": {
                "by_scope": [
                    {"$group": {"_id": "$scope", "count": {"$sum": 1}}}
                ],
                "by_type": [
                    {"$group": {"_id": "$type", "count": {"$sum": 1}}}
                ],
                "with_examples": [
                    {"$match": {"example": {"$ne": None}}},
                    {"$count": "count"}
                ],
                "with_autofix": [
                    {"$match": {"auto_fix": {"$ne": None}}},
                    {"$count": "count"}
                ],
                "active": [
                    {"$match": {"is_active": True}},
                    {"$count": "count"}
                ],
                "total": [
                    {"$count": "count"}
                ]
            }}
        ]

        result = await self.collection.aggregate(pipeline).to_list(1)
        if not result:
            return {"total": 0}

        stats = result[0]
        return {
            "total": stats["total"][0]["count"] if stats["total"] else 0,
            "active": stats["active"][0]["count"] if stats["active"] else 0,
            "by_scope": {r["_id"]: r["count"] for r in stats["by_scope"]},
            "by_type": {r["_id"]: r["count"] for r in stats["by_type"]},
            "with_examples": stats["with_examples"][0]["count"] if stats["with_examples"] else 0,
            "with_autofix": stats["with_autofix"][0]["count"] if stats["with_autofix"] else 0,
        }


# Singleton instance
_rules_service: Optional[SQLRulesService] = None


async def get_sql_rules_service(db: Optional[AsyncIOMotorDatabase] = None) -> SQLRulesService:
    """
    Get the SQLRulesService singleton instance.

    Args:
        db: Optional database instance. If not provided, will use MongoDBService.

    Returns SQLRulesService instance.
    """
    global _rules_service

    if _rules_service is None:
        if db is None:
            # Get from MongoDBService
            from mongodb import get_mongodb_service
            mongodb = get_mongodb_service()
            if not mongodb.is_initialized:
                await mongodb.initialize()
            db = mongodb.db

        from config import COLLECTION_SQL_RULES
        _rules_service = SQLRulesService(db, COLLECTION_SQL_RULES)
        logger.info("SQLRulesService initialized with MongoDB backend")

    return _rules_service
