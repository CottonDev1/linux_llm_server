"""
T-SQL Syntax Guidance Service

Provides pre-generation guidance to help LLMs generate valid T-SQL from the start,
rather than relying solely on post-generation fixes.

This service runs in parallel with schema and rules loading to provide:
1. Common T-SQL patterns and idioms
2. Syntax to avoid (PostgreSQL, MySQL patterns)
3. Best practices for SQL Server

The guidance is included in the LLM prompt BEFORE generation to prevent
invalid syntax from being generated in the first place.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class SyntaxGuideline:
    """A syntax guideline for T-SQL generation."""
    category: str  # e.g., "ordering", "limiting", "dates"
    avoid: str  # What to avoid
    use_instead: str  # What to use instead
    example_bad: Optional[str] = None
    example_good: Optional[str] = None
    priority: int = 5  # 1-10, higher = more important


# Pre-defined T-SQL syntax guidelines
TSQL_SYNTAX_GUIDELINES: List[SyntaxGuideline] = [
    # High priority - common LLM mistakes
    SyntaxGuideline(
        category="ordering",
        avoid="NULLS FIRST or NULLS LAST",
        use_instead="T-SQL sorts NULLs first by default for ASC, last for DESC. Do not use NULLS FIRST/LAST.",
        example_bad="ORDER BY column NULLS LAST",
        example_good="ORDER BY column",
        priority=10
    ),
    SyntaxGuideline(
        category="limiting",
        avoid="LIMIT clause",
        use_instead="Use TOP N after SELECT",
        example_bad="SELECT * FROM table LIMIT 10",
        example_good="SELECT TOP 10 * FROM table",
        priority=10
    ),
    SyntaxGuideline(
        category="limiting",
        avoid="LIMIT with OFFSET",
        use_instead="Use OFFSET-FETCH with ORDER BY",
        example_bad="SELECT * FROM table LIMIT 10 OFFSET 20",
        example_good="SELECT * FROM table ORDER BY id OFFSET 20 ROWS FETCH NEXT 10 ROWS ONLY",
        priority=9
    ),
    SyntaxGuideline(
        category="functions",
        avoid="IFNULL function",
        use_instead="Use ISNULL or COALESCE",
        example_bad="IFNULL(column, 'default')",
        example_good="ISNULL(column, 'default') or COALESCE(column, 'default')",
        priority=8
    ),
    SyntaxGuideline(
        category="functions",
        avoid="NOW() function",
        use_instead="Use GETDATE() or SYSDATETIME()",
        example_bad="WHERE created_at > NOW()",
        example_good="WHERE created_at > GETDATE()",
        priority=8
    ),
    SyntaxGuideline(
        category="dates",
        avoid="CURRENT_DATE",
        use_instead="Use CAST(GETDATE() AS DATE)",
        example_bad="WHERE date_col = CURRENT_DATE",
        example_good="WHERE date_col = CAST(GETDATE() AS DATE)",
        priority=7
    ),
    SyntaxGuideline(
        category="dates",
        avoid="PostgreSQL interval syntax",
        use_instead="Use DATEADD function",
        example_bad="WHERE date > NOW() - INTERVAL '7 days'",
        example_good="WHERE date > DATEADD(DAY, -7, GETDATE())",
        priority=8
    ),
    SyntaxGuideline(
        category="identifiers",
        avoid="Backtick quotes for identifiers",
        use_instead="Use square brackets [identifier]",
        example_bad="SELECT `column name` FROM `table`",
        example_good="SELECT [column name] FROM [table]",
        priority=7
    ),
    SyntaxGuideline(
        category="casting",
        avoid="PostgreSQL :: cast syntax",
        use_instead="Use CAST(expr AS type) or CONVERT",
        example_bad="SELECT value::integer",
        example_good="SELECT CAST(value AS INT)",
        priority=7
    ),
    SyntaxGuideline(
        category="strings",
        avoid="ILIKE for case-insensitive matching",
        use_instead="Use LIKE (T-SQL is case-insensitive by default with CI collation)",
        example_bad="WHERE name ILIKE '%john%'",
        example_good="WHERE name LIKE '%john%'",
        priority=6
    ),
    SyntaxGuideline(
        category="booleans",
        avoid="TRUE/FALSE boolean literals",
        use_instead="Use 1/0 or bit values",
        example_bad="WHERE active = TRUE",
        example_good="WHERE active = 1",
        priority=6
    ),
    SyntaxGuideline(
        category="strings",
        avoid="|| for string concatenation",
        use_instead="Use + operator or CONCAT function",
        example_bad="SELECT first_name || ' ' || last_name",
        example_good="SELECT first_name + ' ' + last_name",
        priority=6
    ),
    SyntaxGuideline(
        category="schema",
        avoid="Inventing table or column names",
        use_instead="Use ONLY tables and columns from the provided schema",
        example_bad="SELECT * FROM uvw_Users (if not in schema)",
        example_good="Use exact table names from schema context",
        priority=10
    ),
]


class SyntaxGuidanceService:
    """
    Service for providing T-SQL syntax guidance to LLM prompts.

    This service should be called in parallel with schema and rules loading
    to provide syntax guidance that prevents invalid SQL from being generated.
    """

    _instance = None

    def __init__(self):
        """Initialize the syntax guidance service."""
        self.guidelines = TSQL_SYNTAX_GUIDELINES
        logger.info("SyntaxGuidanceService initialized with %d guidelines", len(self.guidelines))

    @classmethod
    async def get_instance(cls) -> "SyntaxGuidanceService":
        """Get singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def get_all_guidelines(self, min_priority: int = 1) -> List[SyntaxGuideline]:
        """
        Get all guidelines filtered by minimum priority.

        Args:
            min_priority: Minimum priority level (1-10)

        Returns:
            List of guidelines sorted by priority (highest first)
        """
        filtered = [g for g in self.guidelines if g.priority >= min_priority]
        return sorted(filtered, key=lambda g: g.priority, reverse=True)

    def get_guidelines_by_category(self, category: str) -> List[SyntaxGuideline]:
        """Get guidelines for a specific category."""
        return [g for g in self.guidelines if g.category == category]

    def format_for_prompt(self, min_priority: int = 6) -> str:
        """
        Format guidelines as text for inclusion in LLM prompt.

        Args:
            min_priority: Only include guidelines with this priority or higher

        Returns:
            Formatted string for prompt inclusion
        """
        guidelines = self.get_all_guidelines(min_priority)

        if not guidelines:
            return ""

        lines = ["T-SQL SYNTAX RULES (MUST FOLLOW):"]

        for g in guidelines:
            lines.append(f"- AVOID: {g.avoid}")
            lines.append(f"  USE: {g.use_instead}")
            if g.example_good:
                lines.append(f"  Example: {g.example_good}")

        return "\n".join(lines)

    def format_compact(self, min_priority: int = 7) -> str:
        """
        Format guidelines in a compact form for system prompts.

        Args:
            min_priority: Only include guidelines with this priority or higher

        Returns:
            Compact formatted string
        """
        guidelines = self.get_all_guidelines(min_priority)

        if not guidelines:
            return ""

        rules = []
        for g in guidelines:
            rules.append(f"NO {g.avoid} -> {g.use_instead}")

        return "T-SQL Rules: " + "; ".join(rules)

    def get_avoidance_list(self) -> List[str]:
        """Get list of patterns to avoid for validation."""
        return [g.avoid for g in self.guidelines]

    async def load_guidelines(self) -> Dict[str, Any]:
        """
        Async method for parallel loading compatibility.

        Returns guidance data that can be used during prompt construction.
        This method is designed to be called with asyncio.gather alongside
        schema and rules loading.
        """
        return {
            "prompt_text": self.format_for_prompt(min_priority=6),
            "compact_text": self.format_compact(min_priority=7),
            "guideline_count": len(self.guidelines),
            "categories": list(set(g.category for g in self.guidelines)),
        }


# Convenience function for parallel loading
async def load_syntax_guidance() -> Dict[str, Any]:
    """
    Load T-SQL syntax guidance for parallel execution.

    This function is designed to be called with asyncio.gather:

        syntax_guidance, schema, rules = await asyncio.gather(
            load_syntax_guidance(),
            load_schema(...),
            load_rules(...)
        )

    Returns:
        Dict with formatted guidance text and metadata
    """
    service = await SyntaxGuidanceService.get_instance()
    return await service.load_guidelines()
