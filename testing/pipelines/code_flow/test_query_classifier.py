"""
Query Classifier Tests
======================

Test the QueryClassifier service including:
- Pattern matching for different query types
- Confidence scoring
- Stage selection logic
- Search term extraction
- All query categories
"""

import pytest
from code_flow_pipeline.services.query_classifier import (
    QueryClassifier,
    get_query_classifier,
    ClassificationPattern,
)
from code_flow_pipeline.models.query_models import (
    QueryType,
    QueryClassification,
    RetrievalStageType,
)


class TestQueryClassifierInitialization:
    """Test classifier initialization."""

    def test_classifier_singleton(self):
        """Test get_query_classifier returns singleton."""
        classifier1 = get_query_classifier()
        classifier2 = get_query_classifier()

        assert classifier1 is classifier2

    def test_classifier_patterns_compiled(self):
        """Test all patterns are compiled on init."""
        classifier = QueryClassifier()

        assert len(classifier._compiled_patterns) > 0
        assert len(classifier._compiled_patterns) == len(QueryClassifier.PATTERNS)

    def test_stage_configs_defined(self):
        """Test stage configurations exist for all query types."""
        classifier = QueryClassifier()

        # Check key query types have stage configs
        assert QueryType.BUSINESS_PROCESS in classifier.STAGE_CONFIGS
        assert QueryType.METHOD_SEARCH_BY_ACTION in classifier.STAGE_CONFIGS
        assert QueryType.CALL_CHAIN in classifier.STAGE_CONFIGS
        assert QueryType.UI_INTERACTION in classifier.STAGE_CONFIGS
        assert QueryType.CLASS_RESPONSIBILITY in classifier.STAGE_CONFIGS
        assert QueryType.DATA_OPERATION in classifier.STAGE_CONFIGS
        assert QueryType.GENERAL in classifier.STAGE_CONFIGS


class TestBusinessProcessClassification:
    """Test classification of business process queries."""

    @pytest.fixture
    def classifier(self):
        return QueryClassifier()

    @pytest.mark.parametrize("query,expected_confidence_min", [
        ("How do bales get committed to purchase contracts?", 0.8),
        ("How does the system process incoming shipments?", 0.8),
        ("How is inventory managed in the warehouse?", 0.8),
        ("What is the workflow for order fulfillment?", 0.8),
        ("Explain the process of ticket creation", 0.8),
        ("Describe the workflow for load assignment", 0.8),
    ])
    def test_business_process_queries(self, classifier, query, expected_confidence_min):
        """Test various business process query patterns."""
        result = classifier.classify(query)

        assert result.type == QueryType.BUSINESS_PROCESS
        assert result.confidence >= expected_confidence_min
        assert len(result.matched_patterns) > 0

    def test_how_does_pattern(self, classifier):
        """Test 'how does' pattern matching."""
        result = classifier.classify("How does the system handle payments?")

        assert result.type == QueryType.BUSINESS_PROCESS
        assert result.confidence >= 0.85

    def test_what_is_the_process_pattern(self, classifier):
        """Test 'what is the process' pattern matching."""
        result = classifier.classify("What is the process for approving orders?")

        assert result.type == QueryType.BUSINESS_PROCESS
        assert result.confidence >= 0.8

    def test_explain_workflow_pattern(self, classifier):
        """Test 'explain workflow' pattern matching."""
        result = classifier.classify("Explain the workflow for ticket escalation")

        assert result.type == QueryType.BUSINESS_PROCESS


class TestMethodSearchClassification:
    """Test classification of method search queries."""

    @pytest.fixture
    def classifier(self):
        return QueryClassifier()

    @pytest.mark.parametrize("query,expected_confidence_min", [
        ("What methods update the bale status?", 0.7),
        ("Which methods insert records into the database?", 0.7),
        ("Find all methods that delete order items", 0.7),
        ("Show me methods that modify user profiles", 0.7),
        ("Where is the save operation called?", 0.7),
    ])
    def test_method_search_queries(self, classifier, query, expected_confidence_min):
        """Test various method search query patterns."""
        result = classifier.classify(query)

        assert result.type == QueryType.METHOD_SEARCH_BY_ACTION
        assert result.confidence >= expected_confidence_min

    def test_what_methods_pattern(self, classifier):
        """Test 'what methods' pattern matching."""
        result = classifier.classify("What methods update inventory?")

        assert result.type == QueryType.METHOD_SEARCH_BY_ACTION

    def test_find_methods_pattern(self, classifier):
        """Test 'find methods' pattern matching."""
        result = classifier.classify("Find methods that validate input")

        assert result.type == QueryType.METHOD_SEARCH_BY_ACTION


class TestCallChainClassification:
    """Test classification of call chain queries."""

    @pytest.fixture
    def classifier(self):
        return QueryClassifier()

    @pytest.mark.parametrize("query,expected_confidence_min", [
        ("Show me the call chain from UI to database", 0.7),
        ("What is the execution path from button click to save?", 0.65),
        ("Trace the call flow from ProcessOrder to database", 0.7),
        ("Build call graph starting from btnSave_Click", 0.85),
        ("Follow the execution from UI to data layer", 0.7),
    ])
    def test_call_chain_queries(self, classifier, query, expected_confidence_min):
        """Test various call chain query patterns."""
        result = classifier.classify(query)

        assert result.type == QueryType.CALL_CHAIN
        assert result.confidence >= expected_confidence_min

    def test_call_chain_explicit_pattern(self, classifier):
        """Test explicit call chain pattern."""
        result = classifier.classify("Show me the call chain for this feature")

        assert result.type == QueryType.CALL_CHAIN
        assert result.confidence >= 0.85

    def test_execution_path_pattern(self, classifier):
        """Test execution path pattern."""
        result = classifier.classify("What is the execution path from A to B?")

        assert result.type == QueryType.CALL_CHAIN


class TestUIInteractionClassification:
    """Test classification of UI interaction queries."""

    @pytest.fixture
    def classifier(self):
        return QueryClassifier()

    @pytest.mark.parametrize("query,expected_confidence_min", [
        ("What happens when user clicks the Save button?", 0.85),
        ("What occurs when the submit button is pressed?", 0.8),
        ("Show me the button click handler for btnProcess", 0.7),
        ("What code runs when the form submits?", 0.7),
        ("UI event handler for grid selection", 0.7),
    ])
    def test_ui_interaction_queries(self, classifier, query, expected_confidence_min):
        """Test various UI interaction query patterns."""
        result = classifier.classify(query)

        assert result.type == QueryType.UI_INTERACTION
        assert result.confidence >= expected_confidence_min

    def test_button_click_pattern(self, classifier):
        """Test button click pattern."""
        result = classifier.classify("What happens when user clicks btnSave?")

        assert result.type == QueryType.UI_INTERACTION

    def test_event_handler_pattern(self, classifier):
        """Test event handler pattern."""
        result = classifier.classify("Show the click handler for Save button")

        assert result.type == QueryType.UI_INTERACTION


class TestClassResponsibilityClassification:
    """Test classification of class responsibility queries."""

    @pytest.fixture
    def classifier(self):
        return QueryClassifier()

    @pytest.mark.parametrize("query,expected_confidence_min", [
        ("What classes handle user authentication?", 0.8),
        ("Which class manages database connections?", 0.8),
        ("What is the purpose of BaleService class?", 0.75),
        ("What is the role of OrderValidator class?", 0.75),
    ])
    def test_class_responsibility_queries(self, classifier, query, expected_confidence_min):
        """Test various class responsibility query patterns."""
        result = classifier.classify(query)

        assert result.type == QueryType.CLASS_RESPONSIBILITY
        assert result.confidence >= expected_confidence_min

    def test_what_classes_handle_pattern(self, classifier):
        """Test 'what classes handle' pattern."""
        result = classifier.classify("What classes handle data validation?")

        assert result.type == QueryType.CLASS_RESPONSIBILITY


class TestDataOperationClassification:
    """Test classification of data operation queries."""

    @pytest.fixture
    def classifier(self):
        return QueryClassifier()

    @pytest.mark.parametrize("query,expected_confidence_min", [
        ("Which tables does this feature access?", 0.7),
        ("What database operations update inventory?", 0.8),
        ("Where is order data stored in the database?", 0.75),
        ("How is user information persisted?", 0.75),
        ("What SQL queries read from the Orders table?", 0.8),
    ])
    def test_data_operation_queries(self, classifier, query, expected_confidence_min):
        """Test various data operation query patterns."""
        result = classifier.classify(query)

        assert result.type == QueryType.DATA_OPERATION
        assert result.confidence >= expected_confidence_min

    def test_database_access_pattern(self, classifier):
        """Test database access pattern."""
        result = classifier.classify("Which database tables are accessed?")

        assert result.type == QueryType.DATA_OPERATION


class TestGeneralClassification:
    """Test classification of general/unmatched queries."""

    @pytest.fixture
    def classifier(self):
        return QueryClassifier()

    @pytest.mark.parametrize("query", [
        "Tell me about the code",
        "What is this?",
        "Help me understand",
        "Show me something",
        "random text here",
    ])
    def test_general_queries(self, classifier, query):
        """Test queries that don't match specific patterns."""
        result = classifier.classify(query)

        assert result.type == QueryType.GENERAL
        assert result.confidence == 0.5
        assert len(result.matched_patterns) == 0


class TestConfidenceScoring:
    """Test confidence scoring behavior."""

    @pytest.fixture
    def classifier(self):
        return QueryClassifier()

    def test_high_confidence_for_explicit_patterns(self, classifier):
        """Test high confidence for explicit pattern matches."""
        result = classifier.classify("Show me the call chain from UI to database")

        assert result.confidence >= 0.85

    def test_medium_confidence_for_partial_matches(self, classifier):
        """Test medium confidence for partial matches."""
        result = classifier.classify("Where is data saved?")

        # Should match but with lower confidence
        assert result.confidence >= 0.7
        assert result.confidence <= 0.9

    def test_low_confidence_for_general_queries(self, classifier):
        """Test low confidence for unmatched queries."""
        result = classifier.classify("something completely different")

        assert result.confidence == 0.5
        assert result.type == QueryType.GENERAL

    def test_confidence_range(self, classifier):
        """Test confidence values are in valid range."""
        queries = [
            "How does bale processing work?",
            "What methods update inventory?",
            "Show the call chain",
            "What happens on button click?",
            "random query",
        ]

        for query in queries:
            result = classifier.classify(query)
            assert 0.0 <= result.confidence <= 1.0


class TestStageSelection:
    """Test retrieval stage selection based on classification."""

    @pytest.fixture
    def classifier(self):
        return QueryClassifier()

    def test_business_process_stages(self, classifier):
        """Test stages for business process queries."""
        classification = QueryClassification(
            type=QueryType.BUSINESS_PROCESS,
            confidence=0.9,
        )

        stages = classifier.get_retrieval_stages(classification)

        stage_types = [s.stage_type for s in stages]
        assert RetrievalStageType.BUSINESS_PROCESS in stage_types
        assert RetrievalStageType.METHODS in stage_types

    def test_method_search_stages(self, classifier):
        """Test stages for method search queries."""
        classification = QueryClassification(
            type=QueryType.METHOD_SEARCH_BY_ACTION,
            confidence=0.85,
        )

        stages = classifier.get_retrieval_stages(classification)

        stage_types = [s.stage_type for s in stages]
        assert RetrievalStageType.METHODS in stage_types
        assert RetrievalStageType.CALL_GRAPH in stage_types

    def test_ui_interaction_stages(self, classifier):
        """Test stages for UI interaction queries."""
        classification = QueryClassification(
            type=QueryType.UI_INTERACTION,
            confidence=0.9,
        )

        stages = classifier.get_retrieval_stages(classification)

        stage_types = [s.stage_type for s in stages]
        assert RetrievalStageType.UI_EVENTS in stage_types
        assert RetrievalStageType.METHODS in stage_types

    def test_call_chain_excluded_when_not_requested(self, classifier):
        """Test call graph stage excluded when not requested."""
        classification = QueryClassification(
            type=QueryType.METHOD_SEARCH_BY_ACTION,
            confidence=0.85,
        )

        stages = classifier.get_retrieval_stages(
            classification,
            include_call_graph=False
        )

        stage_types = [s.stage_type for s in stages]
        assert RetrievalStageType.CALL_GRAPH not in stage_types

    def test_stages_have_limits(self, classifier):
        """Test all stages have configured limits."""
        classification = QueryClassification(
            type=QueryType.GENERAL,
            confidence=0.5,
        )

        stages = classifier.get_retrieval_stages(classification)

        for stage in stages:
            assert stage.limit > 0
            assert stage.limit <= 50  # Reasonable upper bound

    def test_stages_have_filter_configs(self, classifier):
        """Test stages have filter configurations."""
        classification = QueryClassification(
            type=QueryType.BUSINESS_PROCESS,
            confidence=0.9,
        )

        stages = classifier.get_retrieval_stages(classification)

        for stage in stages:
            # Each stage should have filter configuration
            assert stage.filter_category is not None or stage.stage_type == RetrievalStageType.CALL_GRAPH


class TestSearchTermExtraction:
    """Test search term extraction from queries."""

    @pytest.fixture
    def classifier(self):
        return QueryClassifier()

    def test_extract_quoted_terms(self, classifier):
        """Test extraction of quoted terms."""
        query = 'Find methods related to "CommitBale" operation'
        terms = classifier.suggest_search_terms(query)

        assert "CommitBale" in terms

    def test_extract_pascal_case_identifiers(self, classifier):
        """Test extraction of PascalCase identifiers."""
        query = "How does BaleCommitmentService work?"
        terms = classifier.suggest_search_terms(query)

        assert "BaleCommitmentService" in terms

    def test_extract_snake_case_identifiers(self, classifier):
        """Test extraction of snake_case identifiers."""
        query = "What does process_order do?"
        terms = classifier.suggest_search_terms(query)

        assert "process_order" in terms

    def test_extract_multiple_identifiers(self, classifier):
        """Test extraction of multiple identifiers."""
        query = 'Find "SaveBale" in BaleService class'
        terms = classifier.suggest_search_terms(query)

        assert "SaveBale" in terms
        assert "BaleService" in terms

    def test_no_duplicates_in_terms(self, classifier):
        """Test no duplicate terms returned."""
        query = "BaleService calls BaleService methods"
        terms = classifier.suggest_search_terms(query)

        # Should not have duplicates (case-insensitive)
        lower_terms = [t.lower() for t in terms]
        assert len(lower_terms) == len(set(lower_terms))

    def test_empty_query_returns_empty_terms(self, classifier):
        """Test empty query returns empty terms list."""
        terms = classifier.suggest_search_terms("")

        assert terms == []

    def test_no_identifiers_returns_empty(self, classifier):
        """Test query without identifiers returns empty list."""
        query = "how does this work"
        terms = classifier.suggest_search_terms(query)

        assert terms == []


class TestPatternPriority:
    """Test pattern matching priority (most specific first)."""

    @pytest.fixture
    def classifier(self):
        return QueryClassifier()

    def test_more_specific_pattern_matches_first(self, classifier):
        """Test more specific patterns take precedence."""
        # This query could match both UI_INTERACTION and BUSINESS_PROCESS
        # but UI_INTERACTION patterns should be more specific
        query = "What happens when the save button is clicked?"

        result = classifier.classify(query)

        # Should match UI_INTERACTION due to specific pattern
        assert result.type == QueryType.UI_INTERACTION

    def test_pattern_order_respected(self, classifier):
        """Test patterns are checked in defined order."""
        # The classifier should stop at first match
        query = "How does the workflow process orders?"

        result = classifier.classify(query)
        assert result.type == QueryType.BUSINESS_PROCESS

        # Only one pattern should match
        assert len(result.matched_patterns) == 1


class TestCaseInsensitivity:
    """Test case-insensitive matching."""

    @pytest.fixture
    def classifier(self):
        return QueryClassifier()

    @pytest.mark.parametrize("query", [
        "HOW DOES THE SYSTEM PROCESS ORDERS?",
        "how does the system process orders?",
        "How Does The System Process Orders?",
        "hOw DoEs ThE sYsTeM pRoCeSs OrDeRs?",
    ])
    def test_case_insensitive_classification(self, classifier, query):
        """Test classification is case-insensitive."""
        result = classifier.classify(query)

        assert result.type == QueryType.BUSINESS_PROCESS


class TestQueryTypeEnum:
    """Test QueryType enum completeness."""

    def test_all_query_types_have_stage_configs(self):
        """Test all QueryType values have stage configurations."""
        classifier = QueryClassifier()

        for query_type in QueryType:
            assert query_type in classifier.STAGE_CONFIGS, \
                f"Missing stage config for {query_type}"

    def test_query_type_values(self):
        """Test QueryType enum has expected values."""
        expected_types = [
            "business_process",
            "method_lookup",
            "method_search_by_action",
            "call_chain",
            "ui_event",
            "ui_interaction",
            "data_flow",
            "data_operation",
            "class_structure",
            "class_responsibility",
            "general",
        ]

        actual_values = [t.value for t in QueryType]

        for expected in expected_types:
            assert expected in actual_values


class TestRetrievalStageConfig:
    """Test RetrievalStage configuration."""

    @pytest.fixture
    def classifier(self):
        return QueryClassifier()

    def test_stage_limits_are_reasonable(self, classifier):
        """Test stage limits are within reasonable bounds."""
        for stage_type, limit in classifier.STAGE_LIMITS.items():
            assert 1 <= limit <= 50, f"Unreasonable limit for {stage_type}: {limit}"

    def test_stage_filters_are_valid(self, classifier):
        """Test stage filter configurations are valid."""
        for stage_type, (category, doc_type) in classifier.STAGE_FILTERS.items():
            # Category should be a non-empty string or None
            assert category is None or (isinstance(category, str) and len(category) > 0)
            # Doc type can be None or string
            assert doc_type is None or isinstance(doc_type, str)
