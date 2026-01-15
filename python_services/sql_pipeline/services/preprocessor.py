"""
Preprocessor Module

This service handles preprocessing of natural language input before
SQL generation, including entity extraction, normalization, and enhancement.
"""

from typing import Optional, List, Dict, Any
from dataclasses import dataclass
import logging
import re
from core.log_utils import log_info

logger = logging.getLogger(__name__)


@dataclass
class PreprocessedQuery:
    """Result of query preprocessing."""
    original: str
    normalized: str
    keywords: List[str]
    entities: Dict[str, List[str]]
    time_context: Optional[Dict[str, Any]]
    inferred_tables: List[str]


class Preprocessor:
    """
    Service for preprocessing natural language queries.

    This service provides:
    - Entity extraction (dates, numbers, names)
    - Query normalization
    - Synonym expansion
    - Abbreviation resolution
    - Context enhancement

    Attributes:
        synonym_map: Dictionary of synonyms for expansion
        abbreviation_map: Dictionary of abbreviations to expand
    """

    # Common stop words to remove from keyword extraction
    STOP_WORDS = {
        "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
        "has", "he", "in", "is", "it", "its", "of", "on", "that", "the",
        "to", "was", "were", "will", "with", "me", "my", "show", "get",
        "give", "find", "list", "what", "when", "where", "who", "how",
        "many", "much", "all", "some", "any", "do", "does", "did"
    }

    # Contractions to expand
    CONTRACTIONS = {
        "don't": "do not",
        "doesn't": "does not",
        "didn't": "did not",
        "won't": "will not",
        "wouldn't": "would not",
        "can't": "cannot",
        "cannot": "can not",
        "i'm": "i am",
        "you're": "you are",
        "we're": "we are",
        "they're": "they are",
        "it's": "it is",
        "that's": "that is",
        "what's": "what is",
        "who's": "who is",
        "there's": "there is",
        "here's": "here is",
    }

    # Time patterns for natural language date parsing
    TIME_PATTERNS = {
        r"\btoday\b": {"type": "relative", "value": "today", "sql": "CAST(GETDATE() AS DATE)"},
        r"\byesterday\b": {"type": "relative", "value": "yesterday", "sql": "DATEADD(DAY, -1, CAST(GETDATE() AS DATE))"},
        r"\btomorrow\b": {"type": "relative", "value": "tomorrow", "sql": "DATEADD(DAY, 1, CAST(GETDATE() AS DATE))"},
        r"\blast\s+week\b": {"type": "range", "value": "last week", "start": "DATEADD(DAY, -7, GETDATE())", "end": "GETDATE()"},
        r"\bthis\s+week\b": {"type": "week", "value": "this week", "sql": "DATEPART(WEEK, GETDATE())"},
        r"\bthis\s+month\b": {"type": "month", "value": "this month", "sql": "MONTH(GETDATE())"},
        r"\blast\s+month\b": {"type": "month", "value": "last month", "sql": "MONTH(DATEADD(MONTH, -1, GETDATE()))"},
        r"\bthis\s+year\b": {"type": "year", "value": "this year", "sql": "YEAR(GETDATE())"},
        r"\blast\s+year\b": {"type": "year", "value": "last year", "sql": "YEAR(DATEADD(YEAR, -1, GETDATE()))"},
        r"\blast\s+(\d+)\s+days?\b": {"type": "dynamic_days", "template": "DATEADD(DAY, -{n}, GETDATE())"},
        r"\blast\s+(\d+)\s+weeks?\b": {"type": "dynamic_weeks", "template": "DATEADD(WEEK, -{n}, GETDATE())"},
        r"\blast\s+(\d+)\s+months?\b": {"type": "dynamic_months", "template": "DATEADD(MONTH, -{n}, GETDATE())"},
    }

    # Common domain abbreviations
    DOMAIN_ABBREVIATIONS = {
        "tix": "tickets",
        "db": "database",
        "qty": "quantity",
        "amt": "amount",
        "num": "number",
        "avg": "average",
        "min": "minimum",
        "max": "maximum",
        "cnt": "count",
        "yr": "year",
        "mo": "month",
        "dt": "date",
        "cust": "customer",
        "prod": "product",
        "inv": "inventory",
        "ord": "order",
        "acct": "account",
    }

    # Table keyword mappings for inference
    TABLE_KEYWORDS = {
        "ticket": ["CentralTickets"],
        "tickets": ["CentralTickets"],
        "user": ["CentralUsers"],
        "users": ["CentralUsers"],
        "company": ["CentralCompanies"],
        "companies": ["CentralCompanies"],
        "customer": ["CentralCompanies", "CentralCustomers"],
        "customers": ["CentralCompanies", "CentralCustomers"],
        "bale": ["Bales"],
        "bales": ["Bales"],
        "product": ["Products"],
        "products": ["Products"],
        "order": ["Orders"],
        "orders": ["Orders"],
        "invoice": ["Invoices"],
        "invoices": ["Invoices"],
    }

    def __init__(
        self,
        synonym_map: Optional[dict[str, list[str]]] = None,
        abbreviation_map: Optional[dict[str, str]] = None,
    ):
        """
        Initialize the preprocessor.

        Args:
            synonym_map: Map of terms to their synonyms
            abbreviation_map: Map of abbreviations to full forms
        """
        self.synonym_map = synonym_map or {}
        # Merge default abbreviations with custom ones
        self.abbreviation_map = {**self.DOMAIN_ABBREVIATIONS, **(abbreviation_map or {})}

        log_info("Preprocessor", "Initialized")

    async def preprocess(
        self,
        question: str,
        database: str,
    ) -> PreprocessedQuery:
        """
        Main entry point for preprocessing a natural language query.

        Args:
            question: The natural language query
            database: Target database name for table inference

        Returns:
            PreprocessedQuery with all preprocessing results
        """
        logger.info(f"Preprocessing question: {question}")

        # Store original
        original = question

        # Step 1: Normalize the question
        normalized = self.normalize_question(question)

        # Step 2: Extract keywords
        keywords = self.extract_keywords(normalized)

        # Step 3: Extract entities
        entities = self.extract_entities(normalized)

        # Step 4: Parse time context
        time_context = self.parse_time_context(normalized)

        # Step 5: Infer relevant tables
        inferred_tables = await self.infer_tables(normalized, database)

        result = PreprocessedQuery(
            original=original,
            normalized=normalized,
            keywords=keywords,
            entities=entities,
            time_context=time_context,
            inferred_tables=inferred_tables,
        )

        logger.info(f"Preprocessing complete: {len(keywords)} keywords, {len(inferred_tables)} tables inferred")
        return result

    def normalize_question(self, question: str) -> str:
        """
        Normalize a question by cleaning and standardizing the text.

        Args:
            question: The natural language question

        Returns:
            Normalized question
        """
        # Lowercase
        normalized = question.lower()

        # Expand contractions
        for contraction, expansion in self.CONTRACTIONS.items():
            normalized = re.sub(r'\b' + contraction + r'\b', expansion, normalized)

        # Expand abbreviations
        normalized = self.expand_abbreviations(normalized)

        # Remove extra whitespace
        normalized = re.sub(r'\s+', ' ', normalized).strip()

        # Common typo fixes
        typo_fixes = {
            r'\bteh\b': 'the',
            r'\bfrom\s+form\b': 'from',
            r'\bselcet\b': 'select',
            r'\bwher\b': 'where',
        }
        for typo, correction in typo_fixes.items():
            normalized = re.sub(typo, correction, normalized)

        return normalized

    def extract_keywords(self, question: str) -> List[str]:
        """
        Extract meaningful keywords from the question.

        Args:
            question: The normalized question

        Returns:
            List of keywords (stop words removed)
        """
        # Split into words
        words = re.findall(r'\b\w+\b', question.lower())

        # Remove stop words and short words
        keywords = [
            word for word in words
            if word not in self.STOP_WORDS and len(word) > 2
        ]

        # Remove duplicates while preserving order
        seen = set()
        unique_keywords = []
        for keyword in keywords:
            if keyword not in seen:
                seen.add(keyword)
                unique_keywords.append(keyword)

        return unique_keywords

    def extract_entities(self, question: str) -> Dict[str, List[str]]:
        """
        Extract entities from the question.

        Args:
            question: The normalized question

        Returns:
            Dictionary with keys: "quoted", "numbers", "dates"
        """
        entities = {
            "quoted": [],
            "numbers": [],
            "dates": [],
        }

        # Extract quoted strings (company names, user names, etc.)
        quoted_pattern = r'["\']([^"\']+)["\']'
        entities["quoted"] = re.findall(quoted_pattern, question)

        # Extract numbers (including decimals)
        number_pattern = r'\b\d+\.?\d*\b'
        entities["numbers"] = re.findall(number_pattern, question)

        # Extract explicit dates (YYYY-MM-DD, MM/DD/YYYY, etc.)
        date_patterns = [
            r'\b\d{4}-\d{2}-\d{2}\b',  # YYYY-MM-DD
            r'\b\d{1,2}/\d{1,2}/\d{4}\b',  # MM/DD/YYYY
            r'\b\d{1,2}-\d{1,2}-\d{4}\b',  # MM-DD-YYYY
        ]
        for pattern in date_patterns:
            entities["dates"].extend(re.findall(pattern, question))

        return entities

    def parse_time_context(self, question: str) -> Optional[Dict[str, Any]]:
        """
        Parse natural language time references into SQL expressions.

        Args:
            question: The normalized question

        Returns:
            Time context dictionary or None if no time reference found
        """
        # Check each time pattern
        for pattern, context in self.TIME_PATTERNS.items():
            match = re.search(pattern, question, re.IGNORECASE)
            if match:
                result = dict(context)  # Copy the context

                # Handle dynamic patterns (e.g., "last 7 days")
                if result["type"].startswith("dynamic_"):
                    n = match.group(1)  # Extract the number
                    result["sql"] = result["template"].replace("{n}", n)
                    result["value"] = match.group(0)  # The full matched text
                    del result["template"]

                logger.info(f"Parsed time context: {result}")
                return result

        return None

    async def infer_tables(self, question: str, database: str) -> List[str]:
        """
        Infer relevant table names based on keywords in the question.

        Args:
            question: The normalized question
            database: Database name (for future schema-based inference)

        Returns:
            List of likely relevant table names
        """
        inferred = []
        question_lower = question.lower()

        # Check each keyword mapping
        for keyword, tables in self.TABLE_KEYWORDS.items():
            if re.search(r'\b' + keyword + r'\b', question_lower):
                for table in tables:
                    if table not in inferred:
                        inferred.append(table)

        logger.info(f"Inferred tables: {inferred}")
        return inferred

    def expand_abbreviations(self, question: str) -> str:
        """
        Expand common abbreviations in the question.

        Args:
            question: The natural language question

        Returns:
            Question with abbreviations expanded
        """
        expanded = question

        # Replace each abbreviation with its full form
        for abbr, full in self.abbreviation_map.items():
            # Use word boundaries to avoid partial matches
            pattern = r'\b' + re.escape(abbr) + r'\b'
            expanded = re.sub(pattern, full, expanded, flags=re.IGNORECASE)

        return expanded

    # Keep old methods for backwards compatibility
    def preprocess_old(
        self,
        query: str,
        expand_synonyms: bool = True,
        expand_abbreviations_flag: bool = True,
    ) -> dict:
        """
        Legacy preprocess method for backwards compatibility.

        Args:
            query: The natural language query
            expand_synonyms: Whether to expand synonyms
            expand_abbreviations_flag: Whether to expand abbreviations

        Returns:
            Dictionary containing preprocessed query and extracted entities
        """
        normalized = self.normalize_question(query)

        if expand_abbreviations_flag:
            normalized = self.expand_abbreviations(normalized)

        if expand_synonyms:
            normalized = self.expand_synonyms(normalized)

        entities = self.extract_entities(normalized)

        return {
            "original": query,
            "normalized": normalized,
            "entities": entities,
        }

    def extract_dates(
        self,
        query: str,
    ) -> list[dict]:
        """
        Extract date references from a query.

        Args:
            query: The natural language query

        Returns:
            List of date entity dictionaries
        """
        dates = []

        # Extract explicit dates
        date_patterns = [
            (r'\b\d{4}-\d{2}-\d{2}\b', 'YYYY-MM-DD'),
            (r'\b\d{1,2}/\d{1,2}/\d{4}\b', 'MM/DD/YYYY'),
            (r'\b\d{1,2}-\d{1,2}-\d{4}\b', 'MM-DD-YYYY'),
        ]

        for pattern, format_type in date_patterns:
            matches = re.findall(pattern, query)
            for match in matches:
                dates.append({
                    "value": match,
                    "format": format_type,
                })

        # Extract relative dates
        time_context = self.parse_time_context(query)
        if time_context:
            dates.append({
                "value": time_context.get("value", ""),
                "type": "relative",
                "sql": time_context.get("sql", ""),
            })

        return dates

    def extract_numbers(
        self,
        query: str,
    ) -> list[dict]:
        """
        Extract numeric values from a query.

        Args:
            query: The natural language query

        Returns:
            List of number entity dictionaries
        """
        numbers = []

        # Extract integers and decimals
        number_pattern = r'\b(\d+\.?\d*)\b'
        matches = re.findall(number_pattern, query)

        for match in matches:
            is_decimal = '.' in match
            numbers.append({
                "value": match,
                "type": "decimal" if is_decimal else "integer",
            })

        return numbers

    def extract_table_references(
        self,
        query: str,
        known_tables: Optional[list[str]] = None,
    ) -> list[str]:
        """
        Extract potential table references from a query.

        Args:
            query: The natural language query
            known_tables: Optional list of known table names

        Returns:
            List of potential table references
        """
        references = []
        query_lower = query.lower()

        # Check against known tables if provided
        if known_tables:
            for table in known_tables:
                # Check for exact match or plural forms
                table_lower = table.lower()
                if re.search(r'\b' + re.escape(table_lower) + r's?\b', query_lower):
                    references.append(table)

        # Also check our keyword mappings
        for keyword, tables in self.TABLE_KEYWORDS.items():
            if re.search(r'\b' + keyword + r'\b', query_lower):
                for table in tables:
                    if table not in references:
                        references.append(table)

        return references

    def expand_synonyms(
        self,
        query: str,
    ) -> str:
        """
        Expand synonyms in a query.

        Args:
            query: The natural language query

        Returns:
            Query with synonyms expanded
        """
        expanded = query

        # Replace each term with its synonyms appended
        for term, synonyms in self.synonym_map.items():
            pattern = r'\b' + re.escape(term) + r'\b'
            if re.search(pattern, expanded, re.IGNORECASE):
                # Add synonyms in parentheses
                replacement = f"{term} ({', '.join(synonyms)})"
                expanded = re.sub(pattern, replacement, expanded, flags=re.IGNORECASE)

        return expanded

    def normalize_query(
        self,
        query: str,
    ) -> str:
        """
        Normalize a query (lowercase, remove extra whitespace, etc.).
        Alias for normalize_question for backwards compatibility.

        Args:
            query: The natural language query

        Returns:
            Normalized query
        """
        return self.normalize_question(query)

    def enhance_with_context(
        self,
        query: str,
        context: dict,
    ) -> str:
        """
        Enhance a query with additional context.

        Args:
            query: The natural language query
            context: Context dictionary with additional information

        Returns:
            Enhanced query
        """
        enhanced = query

        # Add database context if provided
        if "database" in context:
            enhanced = f"[Database: {context['database']}] {enhanced}"

        # Add table hints if provided
        if "tables" in context and context["tables"]:
            tables_str = ", ".join(context["tables"])
            enhanced = f"{enhanced} [Relevant tables: {tables_str}]"

        # Add time context if provided
        if "time_period" in context:
            enhanced = f"{enhanced} [Time period: {context['time_period']}]"

        return enhanced
