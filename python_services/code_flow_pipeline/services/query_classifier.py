"""
Query Classifier Service
========================

Classifies code flow queries into types to determine the optimal retrieval strategy.

Design Rationale:
-----------------
Query classification is the first step in the code flow analysis pipeline.
By classifying the query type, we can:
1. Select appropriate retrieval stages (methods, classes, UI events, etc.)
2. Adjust result weights and limits per stage
3. Customize the LLM synthesis prompt

The classifier uses a combination of regex pattern matching and keyword
analysis. This approach is:
- Fast (no LLM required for classification)
- Deterministic (same query always gets same classification)
- Extensible (easy to add new patterns)

For production at scale, consider adding an ML-based classifier that learns
from user feedback on classification accuracy.
"""

import re
import logging
from typing import List, Tuple
from dataclasses import dataclass, field

from code_flow_pipeline.models.query_models import (
    QueryClassification,
    QueryType,
    RetrievalStage,
    RetrievalStageType,
)

logger = logging.getLogger(__name__)


@dataclass
class ClassificationPattern:
    """A pattern for classifying queries."""
    pattern: str
    query_type: QueryType
    confidence: float
    description: str


class QueryClassifier:
    """
    Classifies code flow queries to determine retrieval strategy.

    The classifier analyzes the query text to determine what type of
    information the user is seeking. This drives the multi-stage
    retrieval process.

    Usage:
        classifier = QueryClassifier()
        result = classifier.classify("How are bales committed to purchase?")
        # result.type == QueryType.BUSINESS_PROCESS
        # result.confidence == 0.9
    """

    # Pattern definitions ordered by specificity (most specific first)
    # Each pattern is a tuple of (regex, QueryType, confidence)
    PATTERNS: List[ClassificationPattern] = [
        # Business process patterns
        ClassificationPattern(
            r"how\s+(?:do|does|is|are|can|could).*(?:process|workflow|work|happen|flow|handle|commit|manage)",
            QueryType.BUSINESS_PROCESS,
            0.9,
            "Process/workflow questions"
        ),
        ClassificationPattern(
            r"how\s+(?:do|does|is|are)\s+.*\s+(?:get|gets)\s+",
            QueryType.BUSINESS_PROCESS,
            0.85,
            "How does X get Y questions"
        ),
        ClassificationPattern(
            r"what\s+(?:is|are)\s+(?:the\s+)?(?:step|process|workflow|procedure)",
            QueryType.BUSINESS_PROCESS,
            0.85,
            "Step/procedure questions"
        ),
        ClassificationPattern(
            r"(?:explain|describe)\s+(?:the\s+)?(?:process|workflow|flow|procedure)",
            QueryType.BUSINESS_PROCESS,
            0.85,
            "Explain process questions"
        ),

        # Method search patterns
        ClassificationPattern(
            r"(?:what|which)\s+method[s]?\s+(?:update|insert|delete|modify|change|create|save)",
            QueryType.METHOD_SEARCH_BY_ACTION,
            0.85,
            "Method by action questions"
        ),
        ClassificationPattern(
            r"(?:find|show|list)\s+(?:\w+\s+)?method[s]?\s+(?:that|which)",
            QueryType.METHOD_SEARCH_BY_ACTION,
            0.8,
            "Find methods questions"
        ),
        ClassificationPattern(
            r"(?:show|find|list)\s+(?:\w+\s+)?method[s]?\s+.*(?:update|insert|delete|modify|change|create|save)",
            QueryType.METHOD_SEARCH_BY_ACTION,
            0.8,
            "Show methods that modify questions"
        ),
        ClassificationPattern(
            r"where\s+(?:is|does)\s+.*(?:called|invoked|executed)",
            QueryType.METHOD_SEARCH_BY_ACTION,
            0.75,
            "Where is called questions"
        ),

        # Call chain patterns
        ClassificationPattern(
            r"(?:call|execution)\s+(?:chain|graph|tree|path|flow|stack)",
            QueryType.CALL_CHAIN,
            0.9,
            "Call chain explicit"
        ),
        ClassificationPattern(
            r"(?:trace|follow|track)\s+(?:the\s+)?(?:call|execution|path)",
            QueryType.CALL_CHAIN,
            0.85,
            "Trace call questions"
        ),
        ClassificationPattern(
            r"(?:from|starting\s+from).*(?:to|until|through)",
            QueryType.CALL_CHAIN,
            0.7,
            "From-to path questions"
        ),

        # UI interaction patterns
        ClassificationPattern(
            r"(?:when|what)\s+(?:happens|occurs|runs|executes)\s+(?:when|if)\s+.*(?:click|press|select|check|submit)",
            QueryType.UI_INTERACTION,
            0.9,
            "What happens on click/submit"
        ),
        ClassificationPattern(
            r"what\s+code\s+runs\s+when\s+.*(?:click|press|select|check|submit)",
            QueryType.UI_INTERACTION,
            0.9,
            "What code runs when"
        ),
        ClassificationPattern(
            r"(?:button|control|form|window|dialog|panel|grid).*(?:click|press|event|handler|submit)",
            QueryType.UI_INTERACTION,
            0.85,
            "UI control event questions"
        ),
        ClassificationPattern(
            r"(?:ui|user\s+interface|front\s*end).*(?:event|handler|trigger)",
            QueryType.UI_INTERACTION,
            0.8,
            "UI event questions"
        ),
        ClassificationPattern(
            r"(?:event|click|press|select|change|submit)\s+handler",
            QueryType.UI_INTERACTION,
            0.75,
            "Event handler questions"
        ),

        # Class responsibility patterns
        ClassificationPattern(
            r"(?:what|which)\s+class(?:es)?\s+(?:handle|manage|responsible|deal\s+with)",
            QueryType.CLASS_RESPONSIBILITY,
            0.85,
            "Class responsibility questions"
        ),
        ClassificationPattern(
            r"class\s+.*(?:do|handle|manage|responsible|purpose)",
            QueryType.CLASS_RESPONSIBILITY,
            0.8,
            "Class purpose questions"
        ),
        ClassificationPattern(
            r"(?:purpose|role|responsibility)\s+(?:of|for)\s+.*class",
            QueryType.CLASS_RESPONSIBILITY,
            0.8,
            "Role of class questions"
        ),

        # Data operation patterns
        ClassificationPattern(
            r"(?:database|db|table|sql|query)\s+.*(?:access|read|write|update|insert|delete)",
            QueryType.DATA_OPERATION,
            0.85,
            "Database operation questions"
        ),
        ClassificationPattern(
            r"(?:which|what)\s+(?:table|column|field|data)",
            QueryType.DATA_OPERATION,
            0.75,
            "Which table questions"
        ),
        ClassificationPattern(
            r"(?:where|how)\s+(?:is|are)\s+.*(?:stored|saved|persisted|read|loaded)",
            QueryType.DATA_OPERATION,
            0.8,
            "Where data stored questions"
        ),
    ]

    # Stage configurations per query type
    STAGE_CONFIGS: dict[QueryType, List[RetrievalStageType]] = {
        QueryType.BUSINESS_PROCESS: [
            RetrievalStageType.BUSINESS_PROCESS,
            RetrievalStageType.METHODS,
            RetrievalStageType.UI_EVENTS,
            RetrievalStageType.CALL_GRAPH,
        ],
        QueryType.METHOD_LOOKUP: [
            RetrievalStageType.METHODS,
        ],
        QueryType.METHOD_SEARCH_BY_ACTION: [
            RetrievalStageType.METHODS,
            RetrievalStageType.CALL_GRAPH,
        ],
        QueryType.CALL_CHAIN: [
            RetrievalStageType.METHODS,
            RetrievalStageType.CALL_GRAPH,
            RetrievalStageType.UI_EVENTS,
        ],
        QueryType.UI_INTERACTION: [
            RetrievalStageType.UI_EVENTS,
            RetrievalStageType.METHODS,
            RetrievalStageType.CALL_GRAPH,
        ],
        QueryType.UI_EVENT: [
            RetrievalStageType.UI_EVENTS,
            RetrievalStageType.METHODS,
        ],
        QueryType.CLASS_RESPONSIBILITY: [
            RetrievalStageType.CLASSES,
            RetrievalStageType.METHODS,
        ],
        QueryType.DATA_OPERATION: [
            RetrievalStageType.METHODS,
            RetrievalStageType.CALL_GRAPH,
        ],
        QueryType.DATA_FLOW: [
            RetrievalStageType.METHODS,
            RetrievalStageType.CALL_GRAPH,
            RetrievalStageType.DATABASE_ACCESSORS,
        ],
        QueryType.CLASS_STRUCTURE: [
            RetrievalStageType.CLASSES,
            RetrievalStageType.METHODS,
        ],
        QueryType.GENERAL: [
            RetrievalStageType.BUSINESS_PROCESS,
            RetrievalStageType.METHODS,
            RetrievalStageType.CLASSES,
            RetrievalStageType.UI_EVENTS,
        ],
    }

    # Default limits per stage type
    STAGE_LIMITS: dict[RetrievalStageType, int] = {
        RetrievalStageType.BUSINESS_PROCESS: 3,
        RetrievalStageType.METHODS: 15,
        RetrievalStageType.CLASSES: 10,
        RetrievalStageType.UI_EVENTS: 5,
        RetrievalStageType.CALL_GRAPH: 20,
    }

    # Filter configurations per stage type
    STAGE_FILTERS: dict[RetrievalStageType, Tuple[str, str]] = {
        # (category, type)
        RetrievalStageType.BUSINESS_PROCESS: ("business-process", None),
        RetrievalStageType.METHODS: ("code", "method"),
        RetrievalStageType.CLASSES: ("code", "class"),
        RetrievalStageType.UI_EVENTS: ("ui-mapping", None),
        RetrievalStageType.CALL_GRAPH: ("relationship", "method-call"),
    }

    def __init__(self):
        """Initialize the classifier with compiled regex patterns."""
        self._compiled_patterns: List[Tuple[re.Pattern, QueryType, float, str]] = []
        for p in self.PATTERNS:
            try:
                compiled = re.compile(p.pattern, re.IGNORECASE)
                self._compiled_patterns.append(
                    (compiled, p.query_type, p.confidence, p.pattern)
                )
            except re.error as e:
                logger.warning(f"Invalid regex pattern '{p.pattern}': {e}")

    def classify(self, query: str) -> QueryClassification:
        """
        Classify a code flow query.

        Args:
            query: The natural language query to classify

        Returns:
            QueryClassification with type, confidence, and matched patterns

        Algorithm:
        1. Try each pattern in order (most specific first)
        2. Return first match with its confidence
        3. If no match, return GENERAL with low confidence
        """
        query_lower = query.lower().strip()
        matched_patterns = []

        for pattern, query_type, confidence, pattern_str in self._compiled_patterns:
            if pattern.search(query_lower):
                matched_patterns.append(pattern_str)
                logger.debug(
                    f"Query '{query[:50]}...' classified as {query_type.value} "
                    f"(confidence={confidence}, pattern='{pattern_str}')"
                )
                return QueryClassification(
                    type=query_type,
                    confidence=confidence,
                    matched_patterns=matched_patterns
                )

        # No pattern matched - classify as GENERAL
        logger.debug(f"Query '{query[:50]}...' classified as GENERAL (no pattern match)")
        return QueryClassification(
            type=QueryType.GENERAL,
            confidence=0.5,
            matched_patterns=[]
        )

    def get_retrieval_stages(
        self,
        classification: QueryClassification,
        include_call_graph: bool = True
    ) -> List[RetrievalStage]:
        """
        Get the retrieval stages for a classified query.

        Args:
            classification: The query classification result
            include_call_graph: Whether to include call graph stage

        Returns:
            List of RetrievalStage configurations

        Design Note:
        The stages are returned in priority order. If time or resource
        constraints exist, earlier stages should be prioritized.
        """
        stage_types = self.STAGE_CONFIGS.get(
            classification.type,
            self.STAGE_CONFIGS[QueryType.GENERAL]
        )

        stages = []
        for stage_type in stage_types:
            # Skip call graph if not requested
            if stage_type == RetrievalStageType.CALL_GRAPH and not include_call_graph:
                continue

            # Get filter configuration
            category, doc_type = self.STAGE_FILTERS.get(stage_type, (None, None))

            stage = RetrievalStage(
                type=stage_type,
                enabled=True,
                limit=self.STAGE_LIMITS.get(stage_type, 10),
                filter_category=category,
                filter_type=doc_type,
            )
            stages.append(stage)

        return stages

    def suggest_search_terms(self, query: str) -> List[str]:
        """
        Extract key search terms from the query.

        This is a simple implementation that extracts potential
        identifiers (method names, class names, etc.) from the query.

        Args:
            query: The natural language query

        Returns:
            List of suggested search terms

        Note:
        A more sophisticated implementation could use NLP techniques
        to extract entities and expand synonyms.
        """
        terms = []

        # Extract quoted strings (explicit terms)
        quoted = re.findall(r'["\']([^"\']+)["\']', query)
        terms.extend(quoted)

        # Extract PascalCase or camelCase identifiers
        identifiers = re.findall(r'\b([A-Z][a-z]+(?:[A-Z][a-z]+)+)\b', query)
        terms.extend(identifiers)

        # Extract potential method names (word_word or wordWord patterns)
        methods = re.findall(r'\b([a-z]+(?:_[a-z]+)+)\b', query, re.IGNORECASE)
        terms.extend(methods)

        # Remove duplicates while preserving order
        seen = set()
        unique_terms = []
        for term in terms:
            if term.lower() not in seen:
                seen.add(term.lower())
                unique_terms.append(term)

        return unique_terms


# Singleton instance
_classifier_instance: QueryClassifier | None = None


def get_query_classifier() -> QueryClassifier:
    """Get or create the singleton QueryClassifier instance."""
    global _classifier_instance
    if _classifier_instance is None:
        _classifier_instance = QueryClassifier()
    return _classifier_instance
