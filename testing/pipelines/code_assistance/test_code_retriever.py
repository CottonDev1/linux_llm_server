"""
Code Retriever Service Tests
============================

Test the CodeRetriever service including:
- Method retrieval by name
- Class retrieval
- Project filtering
- to_source_info() method
- Relevance scoring
- Query classification
- Comprehensive retrieval
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Dict, Any, List


# =============================================================================
# Test Data Fixtures
# =============================================================================

@pytest.fixture
def sample_method_data() -> Dict[str, Any]:
    """Create sample method data from MongoDB."""
    return {
        "method_name": "SaveBale",
        "class_name": "BaleService",
        "file_path": "/src/Services/BaleService.cs",
        "line_number": 42,
        "signature": "public void SaveBale(Bale bale)",
        "summary": "Saves a bale record to the database",
        "sql_calls": ["INSERT INTO Bales"],
        "similarity": 0.95,
        "project": "Gin",
        "namespace": "Gin.Services",
    }


@pytest.fixture
def sample_class_data() -> Dict[str, Any]:
    """Create sample class data from MongoDB."""
    return {
        "class_name": "BaleService",
        "namespace": "Gin.Services",
        "file_path": "/src/Services/BaleService.cs",
        "base_class": "BaseService",
        "interfaces": ["IBaleService"],
        "methods": ["SaveBale", "ValidateBale", "GetBales"],
        "similarity": 0.88,
        "project": "Gin",
    }


@pytest.fixture
def sample_event_handler_data() -> Dict[str, Any]:
    """Create sample event handler data from MongoDB."""
    return {
        "event_name": "Click",
        "handler_method": "btnSave_Click",
        "element_name": "btnSave",
        "ui_element_type": "Button",
        "handler_class": "BaleEntryForm",
        "file_path": "/src/Forms/BaleEntryForm.cs",
        "line_number": 150,
        "similarity": 0.75,
        "project": "Gin",
    }


@pytest.fixture
def multiple_methods_data() -> List[Dict[str, Any]]:
    """Create multiple method records for testing."""
    return [
        {
            "method_name": "SaveBale",
            "class_name": "BaleService",
            "file_path": "/src/Services/BaleService.cs",
            "line_number": 42,
            "signature": "public void SaveBale(Bale bale)",
            "similarity": 0.95,
            "project": "Gin",
        },
        {
            "method_name": "ValidateBale",
            "class_name": "BaleService",
            "file_path": "/src/Services/BaleService.cs",
            "line_number": 28,
            "signature": "private bool ValidateBale(Bale bale)",
            "similarity": 0.82,
            "project": "Gin",
        },
        {
            "method_name": "ProcessOrder",
            "class_name": "OrderService",
            "file_path": "/src/Services/OrderService.cs",
            "line_number": 55,
            "signature": "public void ProcessOrder(Order order)",
            "similarity": 0.45,
            "project": "Warehouse",
        },
    ]


# =============================================================================
# Query Classification Tests
# =============================================================================

class TestCodeRetrieverQueryClassification:
    """Test query classification for search strategy."""

    @pytest.fixture
    def retriever(self):
        """Create a CodeRetriever instance for testing."""
        from code_assistance_pipeline.services.code_retriever import CodeRetriever
        return CodeRetriever()

    def test_classify_query_always_searches_methods(self, retriever, pipeline_config):
        """Test that methods are always searched."""
        classification = retriever.classify_query("random query")
        assert classification["search_methods"] is True

    def test_classify_query_detects_event_keywords(self, retriever, pipeline_config):
        """Test detection of event-related keywords."""
        queries = [
            "what happens when save button is clicked",
            "button click handler",
            "event handler for selection changed",
            "mouse hover event",
        ]

        for query in queries:
            classification = retriever.classify_query(query)
            assert classification["search_events"] is True, f"Failed for: {query}"

    def test_classify_query_detects_class_keywords(self, retriever, pipeline_config):
        """Test detection of class-related keywords."""
        queries = [
            "what is the BaleService class",
            "show me the view model for orders",
            "repository pattern implementation",
            "base class for services",
            "interface definitions",
        ]

        for query in queries:
            classification = retriever.classify_query(query)
            assert classification["search_classes"] is True, f"Failed for: {query}"

    def test_classify_query_detects_sql_keywords(self, retriever, pipeline_config):
        """Test detection of SQL-related keywords."""
        queries = [
            "methods that use sql queries",
            "database operations",
            "stored procedure calls",
            "select from orders table",
            "insert statement",
        ]

        for query in queries:
            classification = retriever.classify_query(query)
            assert classification["search_sql_only"] is True, f"Failed for: {query}"

    def test_classify_query_no_special_keywords(self, retriever, pipeline_config):
        """Test classification with no special keywords."""
        query = "how do I save a bale"
        classification = retriever.classify_query(query)

        assert classification["search_methods"] is True
        assert classification["search_events"] is False
        assert classification["search_sql_only"] is False

    def test_classify_query_multiple_keywords(self, retriever, pipeline_config):
        """Test classification with multiple keyword types."""
        query = "database stored procedure called by button click handler"
        classification = retriever.classify_query(query)

        assert classification["search_methods"] is True
        assert classification["search_events"] is True  # "click", "handler"
        assert classification["search_sql_only"] is True  # "database", "stored procedure"


# =============================================================================
# to_source_info Tests
# =============================================================================

class TestCodeRetrieverToSourceInfo:
    """Test conversion of entities to SourceInfo."""

    @pytest.fixture
    def retriever(self):
        """Create a CodeRetriever instance for testing."""
        from code_assistance_pipeline.services.code_retriever import CodeRetriever
        return CodeRetriever()

    def test_to_source_info_method(self, retriever, sample_method_data, pipeline_config):
        """Test converting method data to SourceInfo."""
        from code_assistance_pipeline.models.query_models import SourceType

        source_info = retriever.to_source_info(sample_method_data, SourceType.METHOD)

        assert source_info.type == SourceType.METHOD
        assert source_info.name == "BaleService.SaveBale"
        # Check both field names work (file_path is primary, file is alias)
        assert source_info.file_path == "/src/Services/BaleService.cs"
        assert source_info.line_number == 42
        assert source_info.relevance_score == 0.95
        assert "SaveBale" in source_info.content or "SaveBale" in source_info.name

    def test_to_source_info_class(self, retriever, sample_class_data, pipeline_config):
        """Test converting class data to SourceInfo."""
        from code_assistance_pipeline.models.query_models import SourceType

        source_info = retriever.to_source_info(sample_class_data, SourceType.CLASS)

        assert source_info.type == SourceType.CLASS
        assert "BaleService" in source_info.name
        assert source_info.file_path == "/src/Services/BaleService.cs"
        assert source_info.relevance_score == 0.88
        # Snippet should contain class definition
        assert "class" in source_info.content.lower() or "BaleService" in source_info.content

    def test_to_source_info_event_handler(
        self, retriever, sample_event_handler_data, pipeline_config
    ):
        """Test converting event handler data to SourceInfo."""
        from code_assistance_pipeline.models.query_models import SourceType

        source_info = retriever.to_source_info(
            sample_event_handler_data, SourceType.EVENT_HANDLER
        )

        assert source_info.type == SourceType.EVENT_HANDLER
        assert "Click" in source_info.name
        assert "btnSave_Click" in source_info.name
        assert source_info.file_path == "/src/Forms/BaleEntryForm.cs"
        assert source_info.line_number == 150
        assert source_info.relevance_score == 0.75

    def test_to_source_info_missing_fields(self, retriever, pipeline_config):
        """Test handling of missing fields in entity data."""
        from code_assistance_pipeline.models.query_models import SourceType

        incomplete_data = {
            "method_name": "TestMethod",
            # Missing class_name, file_path, etc.
        }

        source_info = retriever.to_source_info(incomplete_data, SourceType.METHOD)

        # Should not raise, should use defaults
        assert source_info.type == SourceType.METHOD
        assert "TestMethod" in source_info.name
        assert source_info.line_number == 0  # Default
        assert source_info.relevance_score == 0.0  # Default

    def test_to_source_info_with_base_class(self, retriever, pipeline_config):
        """Test class SourceInfo includes base class in snippet."""
        from code_assistance_pipeline.models.query_models import SourceType

        class_data = {
            "class_name": "BaleService",
            "namespace": "Gin.Services",
            "base_class": "BaseService",
            "file_path": "/src/BaleService.cs",
            "similarity": 0.9,
        }

        source_info = retriever.to_source_info(class_data, SourceType.CLASS)

        # Snippet should include base class
        assert "BaseService" in source_info.content


# =============================================================================
# Search Methods Tests
# =============================================================================

class TestCodeRetrieverSearchMethods:
    """Test method search functionality."""

    @pytest.mark.asyncio
    async def test_search_methods_basic(self, pipeline_config, multiple_methods_data):
        """Test basic method search."""
        from code_assistance_pipeline.services.code_retriever import CodeRetriever

        with patch('roslyn_mongodb_service.get_roslyn_mongodb_service') as mock_get_service:
            mock_service = AsyncMock()
            mock_service.search_methods.return_value = multiple_methods_data[:2]
            mock_get_service.return_value = mock_service

            retriever = CodeRetriever()
            await retriever.initialize()

            results = await retriever.search_methods("how to save bale")

            assert len(results) == 2
            mock_service.search_methods.assert_called_once()

    @pytest.mark.asyncio
    async def test_search_methods_with_project_filter(self, pipeline_config, multiple_methods_data):
        """Test method search with project filter."""
        from code_assistance_pipeline.services.code_retriever import CodeRetriever

        with patch('roslyn_mongodb_service.get_roslyn_mongodb_service') as mock_get_service:
            mock_service = AsyncMock()
            # Filter to only Gin project methods
            gin_methods = [m for m in multiple_methods_data if m["project"] == "Gin"]
            mock_service.search_methods.return_value = gin_methods
            mock_get_service.return_value = mock_service

            retriever = CodeRetriever()
            await retriever.initialize()

            results = await retriever.search_methods("save bale", project="Gin")

            mock_service.search_methods.assert_called_once_with(
                query="save bale",
                project="Gin",
                limit=10,
                include_sql_only=False,
            )
            assert all(r["project"] == "Gin" for r in results)

    @pytest.mark.asyncio
    async def test_search_methods_with_limit(self, pipeline_config, multiple_methods_data):
        """Test method search respects limit parameter."""
        from code_assistance_pipeline.services.code_retriever import CodeRetriever

        with patch('roslyn_mongodb_service.get_roslyn_mongodb_service') as mock_get_service:
            mock_service = AsyncMock()
            mock_service.search_methods.return_value = [multiple_methods_data[0]]
            mock_get_service.return_value = mock_service

            retriever = CodeRetriever()
            await retriever.initialize()

            await retriever.search_methods("save", limit=1)

            mock_service.search_methods.assert_called_once()
            call_kwargs = mock_service.search_methods.call_args.kwargs
            assert call_kwargs["limit"] == 1

    @pytest.mark.asyncio
    async def test_search_methods_sql_only(self, pipeline_config, multiple_methods_data):
        """Test method search with sql_only filter."""
        from code_assistance_pipeline.services.code_retriever import CodeRetriever

        with patch('roslyn_mongodb_service.get_roslyn_mongodb_service') as mock_get_service:
            mock_service = AsyncMock()
            mock_service.search_methods.return_value = []
            mock_get_service.return_value = mock_service

            retriever = CodeRetriever()
            await retriever.initialize()

            await retriever.search_methods("database query", sql_only=True)

            call_kwargs = mock_service.search_methods.call_args.kwargs
            assert call_kwargs["include_sql_only"] is True


# =============================================================================
# Search Classes Tests
# =============================================================================

class TestCodeRetrieverSearchClasses:
    """Test class search functionality."""

    @pytest.mark.asyncio
    async def test_search_classes_basic(self, pipeline_config, sample_class_data):
        """Test basic class search."""
        from code_assistance_pipeline.services.code_retriever import CodeRetriever

        with patch('roslyn_mongodb_service.get_roslyn_mongodb_service') as mock_get_service:
            mock_service = AsyncMock()
            mock_service.search_classes.return_value = [sample_class_data]
            mock_get_service.return_value = mock_service

            retriever = CodeRetriever()
            await retriever.initialize()

            results = await retriever.search_classes("service for bales")

            assert len(results) == 1
            assert results[0]["class_name"] == "BaleService"

    @pytest.mark.asyncio
    async def test_search_classes_with_project_filter(self, pipeline_config, sample_class_data):
        """Test class search with project filter."""
        from code_assistance_pipeline.services.code_retriever import CodeRetriever

        with patch('roslyn_mongodb_service.get_roslyn_mongodb_service') as mock_get_service:
            mock_service = AsyncMock()
            mock_service.search_classes.return_value = [sample_class_data]
            mock_get_service.return_value = mock_service

            retriever = CodeRetriever()
            await retriever.initialize()

            await retriever.search_classes("bale service", project="Gin")

            mock_service.search_classes.assert_called_once_with(
                query="bale service",
                project="Gin",
                limit=5,
            )


# =============================================================================
# Search Event Handlers Tests
# =============================================================================

class TestCodeRetrieverSearchEventHandlers:
    """Test event handler search functionality."""

    @pytest.mark.asyncio
    async def test_search_event_handlers_basic(
        self, pipeline_config, sample_event_handler_data
    ):
        """Test basic event handler search."""
        from code_assistance_pipeline.services.code_retriever import CodeRetriever

        with patch('roslyn_mongodb_service.get_roslyn_mongodb_service') as mock_get_service:
            mock_service = AsyncMock()
            mock_service.search_event_handlers.return_value = [sample_event_handler_data]
            mock_get_service.return_value = mock_service

            retriever = CodeRetriever()
            await retriever.initialize()

            results = await retriever.search_event_handlers("save button click")

            assert len(results) == 1
            assert results[0]["handler_method"] == "btnSave_Click"


# =============================================================================
# Call Chain Tests
# =============================================================================

class TestCodeRetrieverCallChain:
    """Test call chain retrieval functionality."""

    @pytest.mark.asyncio
    async def test_get_call_chain_both_directions(self, pipeline_config):
        """Test getting call chain in both directions."""
        from code_assistance_pipeline.services.code_retriever import CodeRetriever

        with patch('roslyn_mongodb_service.get_roslyn_mongodb_service') as mock_get_service:
            mock_service = AsyncMock()
            mock_service.get_call_chain.return_value = {
                "callers": [
                    {"class": "BaleEntryForm", "method": "btnSave_Click"},
                ],
                "callees": [
                    {"class": "BaleRepository", "method": "Insert"},
                ],
            }
            mock_get_service.return_value = mock_service

            retriever = CodeRetriever()
            await retriever.initialize()

            result = await retriever.get_call_chain(
                method_name="SaveBale",
                class_name="BaleService",
                direction="both",
            )

            assert "callers" in result
            assert "callees" in result
            assert len(result["callers"]) == 1
            assert len(result["callees"]) == 1

    @pytest.mark.asyncio
    async def test_get_call_chain_with_depth(self, pipeline_config):
        """Test call chain with custom depth."""
        from code_assistance_pipeline.services.code_retriever import CodeRetriever

        with patch('roslyn_mongodb_service.get_roslyn_mongodb_service') as mock_get_service:
            mock_service = AsyncMock()
            mock_service.get_call_chain.return_value = {"callers": [], "callees": []}
            mock_get_service.return_value = mock_service

            retriever = CodeRetriever()
            await retriever.initialize()

            await retriever.get_call_chain(
                method_name="SaveBale",
                class_name="BaleService",
                max_depth=3,
            )

            mock_service.get_call_chain.assert_called_once()
            call_kwargs = mock_service.get_call_chain.call_args.kwargs
            assert call_kwargs["max_depth"] == 3


# =============================================================================
# Comprehensive Retrieval Tests
# =============================================================================

class TestCodeRetrieverComprehensive:
    """Test comprehensive retrieval across all entity types."""

    @pytest.mark.asyncio
    async def test_retrieve_comprehensive_parallel_execution(
        self, pipeline_config, multiple_methods_data, sample_class_data
    ):
        """Test that comprehensive retrieval executes searches in parallel."""
        from code_assistance_pipeline.services.code_retriever import CodeRetriever

        with patch('roslyn_mongodb_service.get_roslyn_mongodb_service') as mock_get_service:
            mock_service = AsyncMock()
            mock_service.search_methods.return_value = multiple_methods_data[:2]
            mock_service.search_classes.return_value = [sample_class_data]
            mock_service.search_event_handlers.return_value = []
            mock_service.get_call_chain.return_value = {"callers": [], "callees": []}
            mock_get_service.return_value = mock_service

            retriever = CodeRetriever()
            await retriever.initialize()

            methods, classes, events, call_chain = await retriever.retrieve_comprehensive(
                query="how to save bale data",
            )

            # Verify all searches were made
            assert len(methods) == 2
            assert len(classes) == 1
            mock_service.search_methods.assert_called()
            mock_service.search_classes.assert_called()

    @pytest.mark.asyncio
    async def test_retrieve_comprehensive_with_event_query(
        self, pipeline_config, multiple_methods_data, sample_event_handler_data
    ):
        """Test comprehensive retrieval includes events for event-related queries."""
        from code_assistance_pipeline.services.code_retriever import CodeRetriever

        with patch('roslyn_mongodb_service.get_roslyn_mongodb_service') as mock_get_service:
            mock_service = AsyncMock()
            mock_service.search_methods.return_value = multiple_methods_data[:1]
            mock_service.search_classes.return_value = []
            mock_service.search_event_handlers.return_value = [sample_event_handler_data]
            mock_service.get_call_chain.return_value = {"callers": [], "callees": []}
            mock_get_service.return_value = mock_service

            retriever = CodeRetriever()
            await retriever.initialize()

            methods, classes, events, call_chain = await retriever.retrieve_comprehensive(
                query="what happens when save button is clicked",
            )

            # Should include event handler search
            mock_service.search_event_handlers.assert_called()
            assert len(events) == 1

    @pytest.mark.asyncio
    async def test_retrieve_comprehensive_builds_call_chain(
        self, pipeline_config, multiple_methods_data
    ):
        """Test that comprehensive retrieval builds call chain from top method."""
        from code_assistance_pipeline.services.code_retriever import CodeRetriever

        with patch('roslyn_mongodb_service.get_roslyn_mongodb_service') as mock_get_service:
            mock_service = AsyncMock()
            mock_service.search_methods.return_value = multiple_methods_data[:2]
            mock_service.search_classes.return_value = []
            mock_service.search_event_handlers.return_value = []
            mock_service.get_call_chain.return_value = {
                "callers": [
                    {"class": "BaleEntryForm", "method": "btnSave_Click"},
                ],
                "callees": [
                    {"class": "BaleRepository", "method": "Insert"},
                ],
            }
            mock_get_service.return_value = mock_service

            retriever = CodeRetriever()
            await retriever.initialize()

            methods, classes, events, call_chain = await retriever.retrieve_comprehensive(
                query="save bale",
                include_call_chains=True,
            )

            # Should have call chain built from top method
            assert len(call_chain) > 0
            # Call chain should be formatted as "Class.Method"
            assert any("." in item for item in call_chain)

    @pytest.mark.asyncio
    async def test_retrieve_comprehensive_handles_search_error(self, pipeline_config):
        """Test that comprehensive retrieval handles individual search errors."""
        from code_assistance_pipeline.services.code_retriever import CodeRetriever

        with patch('roslyn_mongodb_service.get_roslyn_mongodb_service') as mock_get_service:
            mock_service = AsyncMock()
            mock_service.search_methods.return_value = []
            mock_service.search_classes.side_effect = Exception("Class search failed")
            mock_service.search_event_handlers.return_value = []
            mock_service.get_call_chain.return_value = {"callers": [], "callees": []}
            mock_get_service.return_value = mock_service

            retriever = CodeRetriever()
            await retriever.initialize()

            # Should not raise, should return empty for failed search
            methods, classes, events, call_chain = await retriever.retrieve_comprehensive(
                query="test query",
            )

            # Classes should be empty due to error
            assert classes == []


# =============================================================================
# Initialization and Stats Tests
# =============================================================================

class TestCodeRetrieverInitialization:
    """Test CodeRetriever initialization."""

    @pytest.mark.asyncio
    async def test_lazy_initialization(self, pipeline_config):
        """Test that retriever initializes lazily."""
        from code_assistance_pipeline.services.code_retriever import CodeRetriever

        retriever = CodeRetriever()

        assert retriever._roslyn_service is None
        assert retriever._initialized is False

    @pytest.mark.asyncio
    async def test_double_initialization_safe(self, pipeline_config):
        """Test that double initialization is safe."""
        from code_assistance_pipeline.services.code_retriever import CodeRetriever

        with patch('roslyn_mongodb_service.get_roslyn_mongodb_service') as mock_get_service:
            mock_service = AsyncMock()
            mock_get_service.return_value = mock_service

            retriever = CodeRetriever()
            await retriever.initialize()
            await retriever.initialize()  # Should be no-op

            # Service should only be created once
            assert mock_get_service.call_count == 1

    @pytest.mark.asyncio
    async def test_get_stats(self, pipeline_config):
        """Test getting retriever statistics."""
        from code_assistance_pipeline.services.code_retriever import CodeRetriever

        with patch('roslyn_mongodb_service.get_roslyn_mongodb_service') as mock_get_service:
            mock_service = AsyncMock()
            mock_service.get_stats.return_value = {
                "methods": 1500,
                "classes": 300,
                "event_handlers": 200,
            }
            mock_get_service.return_value = mock_service

            retriever = CodeRetriever()
            await retriever.initialize()

            stats = await retriever.get_stats()

            assert "methods" in stats
            assert stats["methods"] == 1500
