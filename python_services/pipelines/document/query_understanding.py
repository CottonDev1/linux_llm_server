"""
Query Understanding Step - Analyzes and transforms user queries for optimal retrieval

This step is the first in the document retrieval pipeline and performs:
1. Query Classification - Categorizes query type (SIMPLE, FACTUAL, ANALYTICAL, TEMPORAL)
2. Query Rewriting - Optimizes the query for vector retrieval
3. Entity Extraction - Identifies key entities, concepts, and filters
4. Multi-Query Expansion - Generates 3-5 query variants for better recall

Design Rationale:
-----------------
Query understanding is critical because user queries are often:
- Ambiguous: "What about the report?" - needs context
- Informal: "show me ticket stuff" - needs formalization
- Incomplete: Missing context that affects retrieval

By analyzing and expanding queries before retrieval, we can:
- Improve recall by searching with multiple phrasings
- Improve precision by extracting filters (department, date, etc.)
- Route simple queries to cache, complex ones to full pipeline

Implementation Notes:
--------------------
This step uses a lightweight LLM call for classification and expansion.
For production, consider:
- Caching frequent query patterns
- Using a smaller fine-tuned model for classification
- Batching expansion with classification to reduce latency
"""

import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Set
import logging

from .base import PipelineStep, PipelineContext, StepResult, QueryType


# Patterns for quick classification without LLM
SIMPLE_PATTERNS = [
    r'^(what is|define|explain)\s+\w+$',  # Definition queries
    r'^(hi|hello|hey)\b',  # Greetings
    r'^(yes|no|ok|okay|thanks|thank you)\b',  # Acknowledgments
]

TEMPORAL_PATTERNS = [
    r'\b(today|yesterday|this week|last week|this month|last month)\b',
    r'\b(recent|latest|newest|current)\b',
    r'\b(20\d{2})\b',  # Year mentions
    r'\b(\d{1,2}/\d{1,2}/\d{2,4})\b',  # Date patterns
]

ANALYTICAL_PATTERNS = [
    r'\b(compare|difference|versus|vs\.?)\b',
    r'\b(analyze|analysis|evaluate|assessment)\b',
    r'\b(why|how come|reason)\b',
    r'\b(trend|pattern|correlation)\b',
    r'\b(best|worst|most|least)\b',
]


class QueryUnderstandingStep(PipelineStep):
    """
    Analyzes user queries to optimize retrieval.

    This step performs:
    1. Query type classification
    2. Query rewriting for semantic search
    3. Entity extraction (people, concepts, filters)
    4. Multi-query expansion for improved recall

    The step can work with or without an LLM:
    - With LLM: Uses structured prompts for accurate analysis
    - Without LLM: Falls back to pattern matching (less accurate)
    """

    def __init__(
        self,
        llm_service: Optional[Any] = None,
        expansion_count: int = 4,
        enable_entity_extraction: bool = True,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the query understanding step.

        Args:
            llm_service: Optional LLM service for advanced analysis.
                        If None, uses pattern-based classification.
            expansion_count: Number of query variants to generate (3-5 recommended)
            enable_entity_extraction: Whether to extract entities from queries
            logger: Optional logger instance
        """
        super().__init__(logger)
        self.llm_service = llm_service
        self.expansion_count = min(max(expansion_count, 2), 7)  # Clamp to 2-7
        self.enable_entity_extraction = enable_entity_extraction

    @property
    def name(self) -> str:
        return "QueryUnderstanding"

    @property
    def requires(self) -> Set[str]:
        return {"original_query"}

    @property
    def produces(self) -> Set[str]:
        return {
            "query_type",
            "rewritten_query",
            "expanded_queries",
            "extracted_entities",
        }

    async def execute(self, context: PipelineContext) -> StepResult:
        """
        Execute query understanding on the input query.

        Steps:
        1. Classify query type using patterns or LLM
        2. Rewrite query for better retrieval
        3. Extract entities if enabled
        4. Generate query expansions
        """
        query = context.original_query.strip()

        if not query:
            return StepResult(
                success=False,
                errors=["Empty query provided"],
            )

        try:
            # Step 1: Classify query type
            query_type = await self._classify_query(query)
            self.logger.debug(f"Query classified as: {query_type}")

            # Step 2: Rewrite query for retrieval
            rewritten_query = await self._rewrite_query(query, query_type)

            # Step 3: Extract entities
            entities = {}
            if self.enable_entity_extraction:
                entities = await self._extract_entities(query)

            # Step 4: Generate query expansions
            expanded_queries = await self._expand_query(
                query,
                rewritten_query,
                query_type,
                entities
            )

            return StepResult(
                success=True,
                data={
                    "query_type": query_type,
                    "rewritten_query": rewritten_query,
                    "expanded_queries": expanded_queries,
                    "extracted_entities": entities,
                },
                metadata={
                    "expansion_count": len(expanded_queries),
                    "entities_found": sum(len(v) for v in entities.values()),
                    "used_llm": self.llm_service is not None,
                }
            )

        except Exception as e:
            self.logger.exception("Query understanding failed")
            # Return a degraded result rather than failing completely
            return StepResult(
                success=True,  # Partial success - we can still proceed
                data={
                    "query_type": QueryType.FACTUAL,
                    "rewritten_query": query,
                    "expanded_queries": [query],
                    "extracted_entities": {},
                },
                warnings=[f"Query understanding degraded: {str(e)}"],
            )

    async def _classify_query(self, query: str) -> QueryType:
        """
        Classify the query type using patterns or LLM.

        Classification Types:
        - SIMPLE: Can be answered directly or from cache
        - FACTUAL: Requires retrieval of specific information
        - ANALYTICAL: Needs synthesis across multiple sources
        - TEMPORAL: Time-sensitive, requires recent information
        """
        query_lower = query.lower()

        # Quick pattern-based classification (fast path)
        for pattern in SIMPLE_PATTERNS:
            if re.search(pattern, query_lower):
                return QueryType.SIMPLE

        for pattern in TEMPORAL_PATTERNS:
            if re.search(pattern, query_lower):
                return QueryType.TEMPORAL

        for pattern in ANALYTICAL_PATTERNS:
            if re.search(pattern, query_lower):
                return QueryType.ANALYTICAL

        # If LLM is available, use it for ambiguous cases
        if self.llm_service is not None:
            return await self._classify_with_llm(query)

        # Default to FACTUAL for most queries
        return QueryType.FACTUAL

    async def _classify_with_llm(self, query: str) -> QueryType:
        """
        Use LLM for accurate query classification.

        The prompt is designed to be:
        - Concise: Minimal tokens for fast inference
        - Structured: Clear output format
        - Robust: Handles edge cases gracefully
        """
        prompt = f"""Classify this query into exactly one category.

Query: {query}

Categories:
- SIMPLE: Greetings, acknowledgments, single-word definitions
- FACTUAL: Questions seeking specific information or facts
- ANALYTICAL: Questions requiring comparison, analysis, or synthesis
- TEMPORAL: Questions about recent events or time-specific information

Respond with only the category name (SIMPLE, FACTUAL, ANALYTICAL, or TEMPORAL)."""

        try:
            response = await self.llm_service.generate(
                prompt=prompt,
                max_tokens=10,
                temperature=0.0,  # Deterministic for classification
            )

            # Parse response - extract the category
            response_upper = response.strip().upper()

            if "SIMPLE" in response_upper:
                return QueryType.SIMPLE
            elif "TEMPORAL" in response_upper:
                return QueryType.TEMPORAL
            elif "ANALYTICAL" in response_upper:
                return QueryType.ANALYTICAL
            else:
                return QueryType.FACTUAL

        except Exception as e:
            self.logger.warning(f"LLM classification failed: {e}")
            return QueryType.FACTUAL

    async def _rewrite_query(self, query: str, query_type: QueryType) -> str:
        """
        Rewrite the query for better semantic retrieval.

        Techniques applied:
        1. Remove filler words and pleasantries
        2. Expand abbreviations and acronyms
        3. Add implicit context for better matching

        For SIMPLE queries, we keep the original (likely just a lookup).
        For others, we optimize for vector similarity.
        """
        if query_type == QueryType.SIMPLE:
            return query

        # Apply basic transformations without LLM
        rewritten = query

        # Remove common filler phrases
        filler_patterns = [
            r'^(can you|could you|please|i need to|i want to|show me|tell me)\s+',
            r'^(hi|hello|hey|um|uh)\s*,?\s*',
            r'\s*(please|thanks|thank you)\s*$',
        ]

        for pattern in filler_patterns:
            rewritten = re.sub(pattern, '', rewritten, flags=re.IGNORECASE)

        # Expand common abbreviations
        abbreviations = {
            r'\bdocs?\b': 'documentation',
            r'\binfo\b': 'information',
            r'\bconfig\b': 'configuration',
            r'\bauth\b': 'authentication',
            r'\badmin\b': 'administrator',
            r'\bdb\b': 'database',
            r'\bapi\b': 'API interface',
        }

        for pattern, expansion in abbreviations.items():
            rewritten = re.sub(pattern, expansion, rewritten, flags=re.IGNORECASE)

        # If LLM available and query is complex, use it
        if self.llm_service is not None and query_type == QueryType.ANALYTICAL:
            rewritten = await self._rewrite_with_llm(query, rewritten)

        return rewritten.strip() or query

    async def _rewrite_with_llm(self, original: str, preprocessed: str) -> str:
        """
        Use LLM to rewrite complex queries for better retrieval.
        """
        prompt = f"""Rewrite this query to be optimal for semantic search in a knowledge base.
Keep the core meaning but make it clearer and more specific.

Original: {original}

Provide only the rewritten query, nothing else."""

        try:
            response = await self.llm_service.generate(
                prompt=prompt,
                max_tokens=100,
                temperature=0.3,
            )
            return response.strip()
        except Exception as e:
            self.logger.warning(f"LLM rewrite failed: {e}")
            return preprocessed

    async def _extract_entities(self, query: str) -> Dict[str, List[str]]:
        """
        Extract named entities and concepts from the query.

        Categories:
        - departments: IT, HR, Finance, etc.
        - document_types: policy, guide, manual, etc.
        - people: Names mentioned
        - concepts: Technical terms, products, etc.
        - temporal: Date/time references

        These can be used for metadata filtering during retrieval.
        """
        entities: Dict[str, List[str]] = {
            "departments": [],
            "document_types": [],
            "people": [],
            "concepts": [],
            "temporal": [],
        }

        query_lower = query.lower()

        # Department detection
        departments = {
            "it": "IT",
            "hr": "HR",
            "human resources": "HR",
            "finance": "Finance",
            "accounting": "Finance",
            "sales": "Sales",
            "marketing": "Marketing",
            "engineering": "Engineering",
            "support": "Support",
            "customer service": "Support",
            "operations": "Operations",
            "legal": "Legal",
        }

        for pattern, dept in departments.items():
            if pattern in query_lower:
                if dept not in entities["departments"]:
                    entities["departments"].append(dept)

        # Document type detection
        doc_types = {
            "policy": "policy",
            "policies": "policy",
            "guide": "guide",
            "manual": "manual",
            "procedure": "procedure",
            "handbook": "handbook",
            "template": "template",
            "form": "form",
            "report": "report",
            "documentation": "documentation",
            "spec": "specification",
            "specification": "specification",
        }

        for pattern, doc_type in doc_types.items():
            if pattern in query_lower:
                if doc_type not in entities["document_types"]:
                    entities["document_types"].append(doc_type)

        # Temporal extraction
        temporal_matches = []
        for pattern in TEMPORAL_PATTERNS:
            matches = re.findall(pattern, query_lower)
            temporal_matches.extend(matches)

        entities["temporal"] = list(set(temporal_matches))

        # If LLM available, use it for more accurate entity extraction
        if self.llm_service is not None:
            entities = await self._extract_with_llm(query, entities)

        # Filter out empty lists
        return {k: v for k, v in entities.items() if v}

    async def _extract_with_llm(
        self,
        query: str,
        pattern_entities: Dict[str, List[str]]
    ) -> Dict[str, List[str]]:
        """
        Use LLM for more accurate entity extraction.
        """
        prompt = f"""Extract entities from this query for document search.

Query: {query}

Extract into these categories (JSON format):
- departments: Company departments mentioned
- document_types: Types of documents mentioned
- people: Names of people
- concepts: Technical terms, products, or key concepts

Respond with JSON only, no explanation."""

        try:
            import json
            response = await self.llm_service.generate(
                prompt=prompt,
                max_tokens=200,
                temperature=0.0,
            )

            # Try to parse JSON from response
            # Handle potential markdown code blocks
            response = response.strip()
            if response.startswith("```"):
                response = re.sub(r'^```\w*\n?', '', response)
                response = re.sub(r'\n?```$', '', response)

            llm_entities = json.loads(response)

            # Merge with pattern-based entities
            for key in pattern_entities:
                if key in llm_entities:
                    combined = list(set(pattern_entities[key] + llm_entities[key]))
                    pattern_entities[key] = combined

            return pattern_entities

        except Exception as e:
            self.logger.warning(f"LLM entity extraction failed: {e}")
            return pattern_entities

    async def _expand_query(
        self,
        original_query: str,
        rewritten_query: str,
        query_type: QueryType,
        entities: Dict[str, List[str]]
    ) -> List[str]:
        """
        Generate multiple query variants for improved recall.

        Multi-query expansion is one of the most effective RAG techniques,
        improving recall by 20-30% on average. Each variant captures
        different aspects of the user's intent.

        Expansion strategies:
        1. Synonym substitution
        2. Perspective shift (what vs how vs why)
        3. Specificity variation (broad to narrow)
        4. Entity-focused variants
        """
        expansions = [rewritten_query]  # Always include the rewritten query

        # Skip expansion for simple queries
        if query_type == QueryType.SIMPLE:
            return expansions

        # If LLM available, use it for high-quality expansions
        if self.llm_service is not None:
            llm_expansions = await self._expand_with_llm(
                original_query,
                query_type,
                entities
            )
            expansions.extend(llm_expansions)
        else:
            # Pattern-based expansion fallback
            rule_expansions = self._expand_with_rules(
                original_query,
                rewritten_query,
                entities
            )
            expansions.extend(rule_expansions)

        # Deduplicate while preserving order
        seen = set()
        unique_expansions = []
        for exp in expansions:
            exp_normalized = exp.lower().strip()
            if exp_normalized not in seen:
                seen.add(exp_normalized)
                unique_expansions.append(exp)

        # Limit to configured count
        return unique_expansions[:self.expansion_count]

    async def _expand_with_llm(
        self,
        query: str,
        query_type: QueryType,
        entities: Dict[str, List[str]]
    ) -> List[str]:
        """
        Use LLM to generate diverse query expansions.
        """
        prompt = f"""Generate {self.expansion_count - 1} alternative phrasings of this query for document search.

Original query: {query}
Query type: {query_type.value}

Each alternative should:
- Capture the same information need
- Use different words or phrasing
- Be suitable for semantic search

Provide each alternative on a new line, no numbering or bullets."""

        try:
            response = await self.llm_service.generate(
                prompt=prompt,
                max_tokens=300,
                temperature=0.7,  # Higher temperature for diversity
            )

            # Parse lines as separate expansions
            lines = [line.strip() for line in response.strip().split('\n')]
            expansions = [line for line in lines if line and len(line) > 5]

            return expansions[:self.expansion_count - 1]

        except Exception as e:
            self.logger.warning(f"LLM expansion failed: {e}")
            return []

    def _expand_with_rules(
        self,
        original: str,
        rewritten: str,
        entities: Dict[str, List[str]]
    ) -> List[str]:
        """
        Rule-based query expansion fallback.

        Generates variants using:
        - Question word variations
        - Synonym substitutions
        - Entity-focused queries
        """
        expansions = []

        # Add question word variants
        question_starters = ["how to", "what is", "explain", "describe"]
        base = re.sub(r'^(what|how|why|when|where|who)\s+', '', original, flags=re.IGNORECASE)

        for starter in question_starters[:2]:  # Limit to avoid too many
            variant = f"{starter} {base}"
            if variant.lower() != original.lower():
                expansions.append(variant)

        # Add entity-focused variant
        if entities.get("concepts"):
            concept_query = f"{entities['concepts'][0]} information"
            expansions.append(concept_query)

        if entities.get("document_types"):
            doc_type = entities['document_types'][0]
            expansions.append(f"{doc_type} for {base}")

        return expansions[:self.expansion_count - 1]
