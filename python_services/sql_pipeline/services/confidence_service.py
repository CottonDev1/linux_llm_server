"""
Confidence Scoring Service for SQL Generation.

Extracted from ewr_sql_agent Phase 3 implementation.
Provides 4-factor confidence scoring with abstention support.

Usage:
    service = ConfidenceService(threshold=0.6)
    result = await service.calculate_confidence(
        question="Show tickets created today",
        sql="SELECT * FROM CentralTickets WHERE ...",
        schema_info=schema_dict,
        rules=rules_list
    )

    if result.should_abstain:
        print(result.clarifying_questions)
"""

import re
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


@dataclass
class ConfidenceScores:
    """
    Breakdown of confidence scoring components.

    4-factor model:
    - schema_coverage (30%): Do all tables/columns exist?
    - example_similarity (30%): How similar to known examples?
    - complexity_match (20%): Does SQL complexity match question?
    - syntax_valid (20%): Is SQL syntactically valid?
    """
    schema_coverage: float = 0.0
    example_similarity: float = 0.0
    complexity_match: float = 0.0
    syntax_valid: float = 0.0

    @property
    def total(self) -> float:
        """Calculate weighted total confidence score."""
        return (
            self.schema_coverage * 0.3 +
            self.example_similarity * 0.3 +
            self.complexity_match * 0.2 +
            self.syntax_valid * 0.2
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "schema_coverage": self.schema_coverage,
            "example_similarity": self.example_similarity,
            "complexity_match": self.complexity_match,
            "syntax_valid": self.syntax_valid,
            "total": self.total
        }


@dataclass
class ConfidenceResult:
    """Result from confidence calculation."""
    confidence: float
    scores: ConfidenceScores
    should_abstain: bool = False
    abstention_reason: Optional[str] = None
    clarifying_questions: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "confidence": self.confidence,
            "scores": self.scores.to_dict(),
            "should_abstain": self.should_abstain,
            "abstention_reason": self.abstention_reason,
            "clarifying_questions": self.clarifying_questions
        }


class ConfidenceService:
    """
    Service for calculating SQL generation confidence scores.

    Implements 4-factor confidence scoring:
    1. Schema coverage - Are all referenced objects in schema?
    2. Example similarity - How close to known good examples?
    3. Complexity match - Does SQL complexity match question?
    4. Syntax validation - Is the SQL syntactically valid?
    """

    # Question complexity indicators
    QUESTION_COMPLEXITY_INDICATORS = [
        "grouped by", "for each", "per", "by",          # Grouping
        "and", "also", "along with", "with",            # Multiple conditions
        "more than", "less than", "between",            # Comparisons
        "top", "highest", "lowest", "most", "least",    # Ranking
        "average", "total", "count", "sum",             # Aggregations
        "join", "related", "associated",                # Relationships
    ]

    def __init__(self, threshold: float = 0.6):
        """
        Initialize the confidence service.

        Args:
            threshold: Confidence threshold for abstention (default 0.6)
        """
        self.threshold = threshold

    async def calculate_confidence(
        self,
        question: str,
        sql: str,
        schema_info: Optional[Dict[str, Any]] = None,
        rules: Optional[List[Dict[str, Any]]] = None,
        validate_syntax_callback: Optional[callable] = None
    ) -> ConfidenceResult:
        """
        Calculate confidence scores for generated SQL.

        Args:
            question: The natural language question
            sql: The generated SQL
            schema_info: Schema information with tables/columns
            rules: List of SQL rules with example questions
            validate_syntax_callback: Optional async callback to validate SQL syntax

        Returns:
            ConfidenceResult with scores and abstention info
        """
        scores = ConfidenceScores()

        # 1. Schema coverage (30%)
        scores.schema_coverage = self._check_schema_coverage(sql, schema_info)

        # 2. Example similarity (30%)
        scores.example_similarity = self._check_example_similarity(question, rules)

        # 3. Complexity match (20%)
        scores.complexity_match = self._check_complexity_match(question, sql)

        # 4. Syntax validation (20%)
        if validate_syntax_callback:
            try:
                is_valid = await validate_syntax_callback(sql)
                scores.syntax_valid = 1.0 if is_valid else 0.0
            except Exception as e:
                logger.warning(f"Syntax validation failed: {e}")
                scores.syntax_valid = 0.5  # Unknown
        else:
            # Basic heuristic validation
            scores.syntax_valid = self._basic_syntax_check(sql)

        total = scores.total

        # Check for abstention
        if total < self.threshold:
            clarifying = self._generate_clarifying_questions(scores)
            return ConfidenceResult(
                confidence=total,
                scores=scores,
                should_abstain=True,
                abstention_reason=f"Confidence ({total:.2f}) below threshold ({self.threshold})",
                clarifying_questions=clarifying
            )

        return ConfidenceResult(
            confidence=total,
            scores=scores,
            should_abstain=False
        )

    def _check_schema_coverage(
        self,
        sql: str,
        schema_info: Optional[Dict[str, Any]]
    ) -> float:
        """
        Check if all referenced tables/columns exist in schema.

        Returns score from 0.0 to 1.0.
        """
        tables_in_sql = self._extract_tables_from_sql(sql)

        if not tables_in_sql:
            return 0.3  # No tables found = low confidence

        if not schema_info:
            return 0.5  # No schema available = medium baseline

        # Build set of known table names
        known_tables = set()
        tables_list = schema_info.get("tables", [])

        for t in tables_list:
            if isinstance(t, dict):
                name = t.get("name", "").lower()
                schema_name = t.get("schema_name", "dbo").lower()
                known_tables.add(name)
                known_tables.add(f"{schema_name}.{name}")
            elif isinstance(t, str):
                known_tables.add(t.lower())

        # Count how many tables match
        matched = sum(1 for t in tables_in_sql if t.lower() in known_tables)

        return matched / len(tables_in_sql) if tables_in_sql else 0.5

    def _check_example_similarity(
        self,
        question: str,
        rules: Optional[List[Dict[str, Any]]]
    ) -> float:
        """
        Check similarity to known good examples in rules.

        Uses Jaccard similarity on word sets.
        Returns score from 0.0 to 1.0.
        """
        if not rules:
            return 0.5  # No rules = baseline confidence

        best_similarity = 0.0
        question_words = set(question.lower().split())

        for rule in rules:
            example_question = rule.get("example_question") or rule.get("example", {}).get("question")

            if example_question:
                example_words = set(example_question.lower().split())

                # Jaccard similarity
                intersection = len(question_words & example_words)
                union = len(question_words | example_words)

                if union > 0:
                    similarity = intersection / union
                    best_similarity = max(best_similarity, similarity)

        return best_similarity

    def _check_complexity_match(self, question: str, sql: str) -> float:
        """
        Compare question complexity to SQL complexity.

        Returns score from 0.0 to 1.0.
        """
        # Question complexity
        q_lower = question.lower()
        q_complexity = sum(
            1 for ind in self.QUESTION_COMPLEXITY_INDICATORS
            if ind in q_lower
        )

        # SQL complexity
        sql_upper = sql.upper()
        s_complexity = 0

        if "GROUP BY" in sql_upper:
            s_complexity += 1
        if "JOIN" in sql_upper:
            s_complexity += sql_upper.count("JOIN")
        if "WHERE" in sql_upper:
            s_complexity += 1
        if "ORDER BY" in sql_upper:
            s_complexity += 1
        if "HAVING" in sql_upper:
            s_complexity += 1
        if any(agg in sql_upper for agg in ["COUNT(", "SUM(", "AVG(", "MAX(", "MIN("]):
            s_complexity += 1

        # Both simple = good match
        if q_complexity == 0 and s_complexity == 0:
            return 1.0

        # One complex, other simple = mismatch
        if q_complexity == 0 or s_complexity == 0:
            return 0.5

        # Compare complexities
        return min(q_complexity, s_complexity) / max(q_complexity, s_complexity)

    def _basic_syntax_check(self, sql: str) -> float:
        """
        Basic heuristic SQL syntax check.

        Returns 1.0 if looks valid, 0.0 if obviously broken.
        """
        if not sql or len(sql) < 10:
            return 0.0

        sql_upper = sql.upper().strip()

        # Must start with valid SQL keyword
        valid_starts = ['SELECT', 'INSERT', 'UPDATE', 'DELETE', 'WITH', 'EXEC', 'DECLARE']
        if not any(sql_upper.startswith(kw) for kw in valid_starts):
            return 0.0

        # Basic bracket matching
        if sql.count('(') != sql.count(')'):
            return 0.3

        # Check for common truncation issues
        if sql_upper.rstrip().endswith(('SELECT', 'FROM', 'WHERE', 'AND', 'OR')):
            return 0.3

        return 0.8  # Looks okay but not validated

    def _extract_tables_from_sql(self, sql: str) -> List[str]:
        """Extract table names from SQL query."""
        tables = set()

        patterns = [
            r"FROM\s+(\[?[\w\.]+\]?)",
            r"JOIN\s+(\[?[\w\.]+\]?)",
            r"INTO\s+(\[?[\w\.]+\]?)",
            r"UPDATE\s+(\[?[\w\.]+\]?)",
        ]

        for pattern in patterns:
            matches = re.findall(pattern, sql, re.IGNORECASE)
            for match in matches:
                # Clean up brackets and take first word (remove aliases)
                table = match.replace('[', '').replace(']', '').split()[0]
                tables.add(table)

        return list(tables)

    def _generate_clarifying_questions(
        self,
        scores: ConfidenceScores
    ) -> List[str]:
        """
        Generate clarifying questions based on low-scoring factors.

        Returns at most 3 questions.
        """
        questions = []

        if scores.schema_coverage < 0.5:
            questions.append("Which specific table or data source should I query?")

        if scores.example_similarity < 0.3:
            questions.append("Can you provide an example of what the output should look like?")

        if scores.complexity_match < 0.5:
            questions.append("Should the results be grouped or aggregated in any way?")

        if scores.syntax_valid < 0.8:
            questions.append("Can you rephrase your question with more specific column or table names?")

        if not questions:
            questions.append("Can you provide more details about what you're looking for?")

        return questions[:3]


# Convenience function for quick confidence checks
async def check_confidence(
    question: str,
    sql: str,
    schema_info: Optional[Dict[str, Any]] = None,
    rules: Optional[List[Dict[str, Any]]] = None,
    threshold: float = 0.6
) -> ConfidenceResult:
    """
    Quick confidence check for generated SQL.

    Args:
        question: Natural language question
        sql: Generated SQL
        schema_info: Optional schema information
        rules: Optional SQL rules
        threshold: Confidence threshold for abstention

    Returns:
        ConfidenceResult with scores and abstention info
    """
    service = ConfidenceService(threshold=threshold)
    return await service.calculate_confidence(
        question=question,
        sql=sql,
        schema_info=schema_info,
        rules=rules
    )
