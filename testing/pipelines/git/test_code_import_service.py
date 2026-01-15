"""
Code Import Service Tests
=========================

Comprehensive tests for the CodeImportService class providing vector import.

Tests cover:
- Import analysis (sync and async)
- Embedding generation for code
- Batch processing
- MongoDB upsert operations
- Repository data deletion
- Entity to document conversion
- Searchable text generation
"""

import os
import pytest
import asyncio
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from datetime import datetime
from typing import List, Dict, Any

# Setup environment before imports
os.environ.setdefault("GIT_ROOT", r"C:\Projects\Git")
os.environ.setdefault("PROJECT_ROOT", r"C:\Projects\llm_website")

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "python_services"))

try:
    from git_pipeline.services.code_import_service import CodeImportService
    from git_pipeline.models.pipeline_models import (
        PipelineConfig,
        ImportResult,
        RoslynAnalysisOutput,
        CodeEntity,
    )
    HAS_IMPORT_SERVICE = True
except ImportError as e:
    HAS_IMPORT_SERVICE = False
    IMPORT_ERROR = str(e)


@pytest.fixture
def pipeline_config():
    """Create a pipeline configuration for testing."""
    return PipelineConfig(
        git_root=r"C:\Projects\Git",
        project_root=r"C:\Projects\llm_website"
    )


@pytest.fixture
def mock_mongodb_service():
    """Create a mock MongoDB service."""
    mock_collection = AsyncMock()
    mock_collection.update_one = AsyncMock(return_value=Mock(
        upserted_id="new_id_123",
        modified_count=0
    ))
    mock_collection.delete_many = AsyncMock(return_value=Mock(deleted_count=10))

    mock_db = Mock()
    mock_db.__getitem__ = Mock(return_value=mock_collection)

    mock_service = Mock()
    mock_service.db = mock_db
    mock_service.is_initialized = True
    mock_service.initialize = AsyncMock()

    return mock_service


@pytest.fixture
def mock_embedding_service():
    """Create a mock embedding service."""
    mock_service = Mock()
    mock_service.is_initialized = True
    mock_service.initialize = AsyncMock()
    mock_service.generate_embedding = AsyncMock(return_value=[0.1] * 384)
    mock_service.generate_embeddings_batch = AsyncMock(
        side_effect=lambda texts: [[0.1] * 384 for _ in texts]
    )
    return mock_service


@pytest.fixture
def sample_analysis_output():
    """Create sample Roslyn analysis output."""
    return RoslynAnalysisOutput(
        repository="TestRepo",
        file_count=5,
        classes=[
            CodeEntity(
                type="class",
                name="UserService",
                full_name="MyApp.Services.UserService",
                file_path="/src/Services/UserService.cs",
                line_number=10,
                namespace="MyApp.Services",
                modifiers=["public"],
                doc_comment="User management service"
            )
        ],
        methods=[
            CodeEntity(
                type="method",
                name="GetUserById",
                full_name="MyApp.Services.UserService.GetUserById",
                file_path="/src/Services/UserService.cs",
                line_number=25,
                namespace="MyApp.Services",
                parent_class="UserService",
                signature="public User GetUserById(int userId)",
                return_type="User",
                parameters=[{"name": "userId", "type": "int"}],
                modifiers=["public"],
                doc_comment="Retrieves user by ID",
                body_preview="return _repository.Get(userId);"
            )
        ],
        properties=[
            CodeEntity(
                type="property",
                name="ConnectionString",
                full_name="MyApp.Config.ConnectionString",
                file_path="/src/Config/Config.cs",
                line_number=5,
                parent_class="Config",
                return_type="string",
                modifiers=["public", "static"]
            )
        ],
        interfaces=[
            CodeEntity(
                type="interface",
                name="IUserRepository",
                full_name="MyApp.Interfaces.IUserRepository",
                file_path="/src/Interfaces/IUserRepository.cs",
                line_number=3,
                namespace="MyApp.Interfaces",
                modifiers=["public"]
            )
        ],
        enums=[
            CodeEntity(
                type="enum",
                name="UserStatus",
                full_name="MyApp.Models.UserStatus",
                file_path="/src/Models/UserStatus.cs",
                line_number=1,
                namespace="MyApp.Models",
                modifiers=["public"]
            )
        ],
        database_operations=[
            {
                "operationType": "SELECT",
                "tableName": "Users",
                "sqlCommand": "SELECT * FROM Users WHERE UserID = @Id",
                "containingClass": "UserRepository",
                "containingMethod": "GetById",
                "filePath": "/src/Repositories/UserRepository.cs",
                "lineNumber": 20
            }
        ],
        event_handlers=[
            {
                "eventName": "Click",
                "handlerName": "btnSave_Click",
                "controlName": "btnSave",
                "controlType": "Button",
                "containingClass": "UserForm",
                "filePath": "/src/Forms/UserForm.cs",
                "lineNumber": 50
            }
        ]
    )


@pytest.mark.skipif(not HAS_IMPORT_SERVICE, reason=f"CodeImportService not available: {IMPORT_ERROR if not HAS_IMPORT_SERVICE else ''}")
class TestCodeImportServiceInit:
    """Test CodeImportService initialization."""

    def test_initialization_with_default_config(self):
        """Test CodeImportService initializes with default configuration."""
        service = CodeImportService()

        assert service.config is not None
        assert isinstance(service.config, PipelineConfig)
        assert service._mongodb is None  # Lazy loaded
        assert service._embedding_service is None  # Lazy loaded

    def test_initialization_with_custom_config(self, pipeline_config):
        """Test CodeImportService initializes with custom configuration."""
        service = CodeImportService(config=pipeline_config)

        assert service.config == pipeline_config

    def test_batch_size_constant(self):
        """Test BATCH_SIZE constant is defined."""
        assert CodeImportService.BATCH_SIZE == 100


@pytest.mark.skipif(not HAS_IMPORT_SERVICE, reason="CodeImportService not available")
class TestLazyLoading:
    """Test lazy loading of dependencies."""

    @pytest.mark.asyncio
    async def test_lazy_load_mongodb(self, mock_mongodb_service):
        """Test lazy loading of MongoDB service."""
        service = CodeImportService()

        with patch("git_pipeline.services.code_import_service.MongoDBService") as MockMongo:
            MockMongo.get_instance.return_value = mock_mongodb_service

            mongodb = await service._get_mongodb()

            assert mongodb is mock_mongodb_service
            MockMongo.get_instance.assert_called_once()

    @pytest.mark.asyncio
    async def test_lazy_load_embedding_service(self, mock_embedding_service):
        """Test lazy loading of embedding service."""
        service = CodeImportService()

        with patch("git_pipeline.services.code_import_service.EmbeddingService") as MockEmbed:
            MockEmbed.get_instance.return_value = mock_embedding_service

            embedding_svc = await service._get_embedding_service()

            assert embedding_svc is mock_embedding_service
            MockEmbed.get_instance.assert_called_once()

    @pytest.mark.asyncio
    async def test_cached_mongodb_instance(self, mock_mongodb_service):
        """Test MongoDB instance is cached after first load."""
        service = CodeImportService()

        with patch("git_pipeline.services.code_import_service.MongoDBService") as MockMongo:
            MockMongo.get_instance.return_value = mock_mongodb_service

            mongodb1 = await service._get_mongodb()
            mongodb2 = await service._get_mongodb()

            assert mongodb1 is mongodb2
            # Should only be called once due to caching
            assert MockMongo.get_instance.call_count == 1


@pytest.mark.skipif(not HAS_IMPORT_SERVICE, reason="CodeImportService not available")
class TestImportAnalysis:
    """Test analysis import functionality."""

    @pytest.mark.asyncio
    async def test_import_analysis_success(
        self,
        pipeline_config,
        sample_analysis_output,
        mock_mongodb_service,
        mock_embedding_service
    ):
        """Test successful analysis import."""
        service = CodeImportService(config=pipeline_config)

        with patch.object(service, '_get_mongodb', return_value=mock_mongodb_service):
            with patch.object(service, '_get_embedding_service', return_value=mock_embedding_service):
                result = await service.import_analysis(
                    analysis=sample_analysis_output,
                    db_name="testrepo"
                )

                assert isinstance(result, ImportResult)
                assert result.success is True
                assert result.repository == "TestRepo"
                assert result.documents_imported > 0
                assert result.duration_seconds > 0

    @pytest.mark.asyncio
    async def test_import_analysis_empty_output(
        self,
        pipeline_config,
        mock_mongodb_service,
        mock_embedding_service
    ):
        """Test importing empty analysis output."""
        empty_output = RoslynAnalysisOutput(
            repository="EmptyRepo",
            file_count=0,
            classes=[],
            methods=[],
            properties=[],
            interfaces=[],
            enums=[]
        )

        service = CodeImportService(config=pipeline_config)

        with patch.object(service, '_get_mongodb', return_value=mock_mongodb_service):
            with patch.object(service, '_get_embedding_service', return_value=mock_embedding_service):
                result = await service.import_analysis(
                    analysis=empty_output,
                    db_name="emptyrepo"
                )

                assert result.success is True
                assert result.documents_imported == 0

    @pytest.mark.asyncio
    async def test_import_analysis_exception_handling(
        self,
        pipeline_config,
        sample_analysis_output
    ):
        """Test import handles exceptions gracefully."""
        service = CodeImportService(config=pipeline_config)

        with patch.object(service, '_get_mongodb', side_effect=Exception("MongoDB connection failed")):
            result = await service.import_analysis(
                analysis=sample_analysis_output,
                db_name="testrepo"
            )

            assert result.success is False
            assert "MongoDB connection failed" in result.error


@pytest.mark.skipif(not HAS_IMPORT_SERVICE, reason="CodeImportService not available")
class TestImportAnalysisSync:
    """Test synchronous analysis import."""

    def test_import_analysis_sync(
        self,
        pipeline_config,
        sample_analysis_output
    ):
        """Test sync wrapper for analysis import."""
        service = CodeImportService(config=pipeline_config)

        async_result = ImportResult(
            success=True,
            documents_imported=10,
            documents_updated=5,
            duration_seconds=1.5
        )

        with patch.object(service, 'import_analysis', new=AsyncMock(return_value=async_result)):
            result = service.import_analysis_sync(
                analysis=sample_analysis_output,
                db_name="testrepo"
            )

            assert isinstance(result, ImportResult)
            assert result.success is True


@pytest.mark.skipif(not HAS_IMPORT_SERVICE, reason="CodeImportService not available")
class TestImportEntities:
    """Test entity import functionality."""

    @pytest.mark.asyncio
    async def test_import_entities_success(
        self,
        pipeline_config,
        mock_mongodb_service,
        mock_embedding_service
    ):
        """Test successful entity import."""
        entities = [
            CodeEntity(type="class", name="TestClass1", file_path="/test1.cs"),
            CodeEntity(type="class", name="TestClass2", file_path="/test2.cs"),
        ]

        service = CodeImportService(config=pipeline_config)

        stats = await service._import_entities(
            entities=entities,
            collection_name="code_classes",
            repository="TestRepo",
            db_name="testrepo",
            mongodb=mock_mongodb_service,
            embedding_service=mock_embedding_service
        )

        assert stats["imported"] > 0 or stats["updated"] >= 0
        # Verify update_one was called for each entity
        assert mock_mongodb_service.db["code_classes"].update_one.call_count == 2

    @pytest.mark.asyncio
    async def test_import_entities_empty_list(
        self,
        pipeline_config,
        mock_mongodb_service,
        mock_embedding_service
    ):
        """Test importing empty entity list."""
        service = CodeImportService(config=pipeline_config)

        stats = await service._import_entities(
            entities=[],
            collection_name="code_classes",
            repository="TestRepo",
            db_name="testrepo",
            mongodb=mock_mongodb_service,
            embedding_service=mock_embedding_service
        )

        assert stats["imported"] == 0
        assert stats["updated"] == 0

    @pytest.mark.asyncio
    async def test_import_entities_batch_processing(
        self,
        pipeline_config,
        mock_mongodb_service,
        mock_embedding_service
    ):
        """Test batch processing of large entity lists."""
        # Create more entities than batch size
        entities = [
            CodeEntity(type="class", name=f"Class{i}", file_path=f"/src/class{i}.cs")
            for i in range(150)  # More than BATCH_SIZE (100)
        ]

        service = CodeImportService(config=pipeline_config)

        stats = await service._import_entities(
            entities=entities,
            collection_name="code_classes",
            repository="TestRepo",
            db_name="testrepo",
            mongodb=mock_mongodb_service,
            embedding_service=mock_embedding_service
        )

        # All entities should be processed
        assert mock_mongodb_service.db["code_classes"].update_one.call_count == 150


@pytest.mark.skipif(not HAS_IMPORT_SERVICE, reason="CodeImportService not available")
class TestImportDbOperations:
    """Test database operations import."""

    @pytest.mark.asyncio
    async def test_import_db_operations_success(
        self,
        pipeline_config,
        mock_mongodb_service,
        mock_embedding_service
    ):
        """Test successful database operations import."""
        operations = [
            {
                "operationType": "SELECT",
                "tableName": "Users",
                "sqlCommand": "SELECT * FROM Users",
                "containingClass": "UserRepository",
                "containingMethod": "GetAll",
                "filePath": "/src/UserRepository.cs",
                "lineNumber": 10
            }
        ]

        service = CodeImportService(config=pipeline_config)

        stats = await service._import_db_operations(
            operations=operations,
            repository="TestRepo",
            db_name="testrepo",
            mongodb=mock_mongodb_service,
            embedding_service=mock_embedding_service
        )

        assert stats["imported"] >= 0
        mock_mongodb_service.db["code_dboperations"].update_one.assert_called()

    @pytest.mark.asyncio
    async def test_import_db_operations_empty(
        self,
        pipeline_config,
        mock_mongodb_service,
        mock_embedding_service
    ):
        """Test importing empty database operations list."""
        service = CodeImportService(config=pipeline_config)

        stats = await service._import_db_operations(
            operations=[],
            repository="TestRepo",
            db_name="testrepo",
            mongodb=mock_mongodb_service,
            embedding_service=mock_embedding_service
        )

        assert stats["imported"] == 0


@pytest.mark.skipif(not HAS_IMPORT_SERVICE, reason="CodeImportService not available")
class TestImportEventHandlers:
    """Test event handlers import."""

    @pytest.mark.asyncio
    async def test_import_event_handlers_success(
        self,
        pipeline_config,
        mock_mongodb_service,
        mock_embedding_service
    ):
        """Test successful event handlers import."""
        handlers = [
            {
                "eventName": "Click",
                "handlerName": "btnSave_Click",
                "controlName": "btnSave",
                "controlType": "Button",
                "containingClass": "UserForm",
                "filePath": "/src/UserForm.cs",
                "lineNumber": 50
            }
        ]

        service = CodeImportService(config=pipeline_config)

        stats = await service._import_event_handlers(
            handlers=handlers,
            repository="TestRepo",
            db_name="testrepo",
            mongodb=mock_mongodb_service,
            embedding_service=mock_embedding_service
        )

        assert stats["imported"] >= 0
        mock_mongodb_service.db["code_eventhandlers"].update_one.assert_called()

    @pytest.mark.asyncio
    async def test_import_event_handlers_empty(
        self,
        pipeline_config,
        mock_mongodb_service,
        mock_embedding_service
    ):
        """Test importing empty event handlers list."""
        service = CodeImportService(config=pipeline_config)

        stats = await service._import_event_handlers(
            handlers=[],
            repository="TestRepo",
            db_name="testrepo",
            mongodb=mock_mongodb_service,
            embedding_service=mock_embedding_service
        )

        assert stats["imported"] == 0


@pytest.mark.skipif(not HAS_IMPORT_SERVICE, reason="CodeImportService not available")
class TestEntityToSearchableText:
    """Test searchable text generation."""

    def test_entity_to_searchable_text_method(self, pipeline_config):
        """Test searchable text generation for method."""
        entity = CodeEntity(
            type="method",
            name="GetUserById",
            parent_class="UserService",
            namespace="MyApp.Services",
            signature="public User GetUserById(int userId)",
            return_type="User",
            doc_comment="<summary>Retrieves user by ID</summary>",
            body_preview="return _repo.Get(userId);"
        )

        service = CodeImportService(config=pipeline_config)
        text = service._entity_to_searchable_text(entity)

        assert "method" in text
        assert "GetUserById" in text
        assert "UserService" in text
        assert "User" in text
        assert "Retrieves user by ID" in text

    def test_entity_to_searchable_text_class(self, pipeline_config):
        """Test searchable text generation for class."""
        entity = CodeEntity(
            type="class",
            name="UserService",
            namespace="MyApp.Services",
            doc_comment="User management service"
        )

        service = CodeImportService(config=pipeline_config)
        text = service._entity_to_searchable_text(entity)

        assert "class" in text
        assert "UserService" in text
        assert "MyApp.Services" in text
        assert "User management service" in text

    def test_entity_to_searchable_text_minimal(self, pipeline_config):
        """Test searchable text generation with minimal entity."""
        entity = CodeEntity(type="property", name="Id")

        service = CodeImportService(config=pipeline_config)
        text = service._entity_to_searchable_text(entity)

        assert "property" in text
        assert "Id" in text


@pytest.mark.skipif(not HAS_IMPORT_SERVICE, reason="CodeImportService not available")
class TestDbOperationToSearchableText:
    """Test database operation searchable text generation."""

    def test_db_operation_to_searchable_text_full(self, pipeline_config):
        """Test searchable text for full database operation."""
        operation = {
            "operationType": "SELECT",
            "tableName": "Users",
            "sqlCommand": "SELECT * FROM Users WHERE Id = @Id",
            "storedProcedure": None,
            "containingClass": "UserRepository",
            "containingMethod": "GetById"
        }

        service = CodeImportService(config=pipeline_config)
        text = service._db_operation_to_searchable_text(operation)

        assert "SELECT" in text
        assert "Users" in text
        assert "UserRepository" in text
        assert "GetById" in text

    def test_db_operation_to_searchable_text_stored_proc(self, pipeline_config):
        """Test searchable text for stored procedure operation."""
        operation = {
            "operationType": "EXEC",
            "storedProcedure": "sp_GetUserById",
            "containingClass": "UserRepository"
        }

        service = CodeImportService(config=pipeline_config)
        text = service._db_operation_to_searchable_text(operation)

        assert "EXEC" in text
        assert "sp_GetUserById" in text


@pytest.mark.skipif(not HAS_IMPORT_SERVICE, reason="CodeImportService not available")
class TestEventHandlerToSearchableText:
    """Test event handler searchable text generation."""

    def test_event_handler_to_searchable_text_full(self, pipeline_config):
        """Test searchable text for full event handler."""
        handler = {
            "eventName": "Click",
            "handlerName": "btnSave_Click",
            "controlName": "btnSave",
            "controlType": "Button",
            "containingClass": "UserForm"
        }

        service = CodeImportService(config=pipeline_config)
        text = service._event_handler_to_searchable_text(handler)

        assert "btnSave_Click" in text
        assert "Click" in text
        assert "btnSave" in text
        assert "Button" in text
        assert "UserForm" in text


@pytest.mark.skipif(not HAS_IMPORT_SERVICE, reason="CodeImportService not available")
class TestEntityToDocument:
    """Test entity to document conversion."""

    def test_entity_to_document_full(self, pipeline_config):
        """Test full entity to document conversion."""
        entity = CodeEntity(
            type="method",
            name="GetUserById",
            full_name="MyApp.Services.UserService.GetUserById",
            file_path="/src/Services/UserService.cs",
            line_number=25,
            namespace="MyApp.Services",
            parent_class="UserService",
            signature="public User GetUserById(int userId)",
            return_type="User",
            parameters=[{"name": "userId", "type": "int"}],
            modifiers=["public"],
            doc_comment="Gets user by ID",
            body_preview="return _repo.Get(userId);"
        )

        service = CodeImportService(config=pipeline_config)
        doc = service._entity_to_document(
            entity=entity,
            repository="TestRepo",
            db_name="testrepo",
            searchable_text="method GetUserById",
            embedding=[0.1] * 384
        )

        assert doc["repository"] == "TestRepo"
        assert doc["db_name"] == "testrepo"
        assert doc["entity_type"] == "method"
        assert doc["name"] == "GetUserById"
        assert doc["full_name"] == "MyApp.Services.UserService.GetUserById"
        assert doc["file_path"] == "/src/Services/UserService.cs"
        assert doc["line_number"] == 25
        assert doc["namespace"] == "MyApp.Services"
        assert doc["parent_class"] == "UserService"
        assert doc["signature"] == "public User GetUserById(int userId)"
        assert doc["return_type"] == "User"
        assert doc["parameters"] == [{"name": "userId", "type": "int"}]
        assert doc["modifiers"] == ["public"]
        assert doc["doc_comment"] == "Gets user by ID"
        assert doc["body_preview"] == "return _repo.Get(userId);"
        assert doc["searchable_text"] == "method GetUserById"
        assert len(doc["vector"]) == 384
        assert "updated_at" in doc

    def test_entity_to_document_minimal(self, pipeline_config):
        """Test minimal entity to document conversion."""
        entity = CodeEntity(type="property", name="Id")

        service = CodeImportService(config=pipeline_config)
        doc = service._entity_to_document(
            entity=entity,
            repository="TestRepo",
            db_name="testrepo",
            searchable_text="property Id",
            embedding=[0.5] * 384
        )

        assert doc["repository"] == "TestRepo"
        assert doc["entity_type"] == "property"
        assert doc["name"] == "Id"
        assert doc["full_name"] == ""  # Default empty string


@pytest.mark.skipif(not HAS_IMPORT_SERVICE, reason="CodeImportService not available")
class TestDeleteRepositoryData:
    """Test repository data deletion."""

    @pytest.mark.asyncio
    async def test_delete_repository_data_all_collections(
        self,
        pipeline_config,
        mock_mongodb_service
    ):
        """Test deleting data from all collections."""
        service = CodeImportService(config=pipeline_config)

        with patch.object(service, '_get_mongodb', return_value=mock_mongodb_service):
            results = await service.delete_repository_data("TestRepo")

            # Should delete from all default collections
            assert len(results) == 5  # 5 collections
            mock_mongodb_service.db["code_classes"].delete_many.assert_called_once()
            mock_mongodb_service.db["code_methods"].delete_many.assert_called_once()

    @pytest.mark.asyncio
    async def test_delete_repository_data_specific_collections(
        self,
        pipeline_config,
        mock_mongodb_service
    ):
        """Test deleting data from specific collections."""
        service = CodeImportService(config=pipeline_config)

        with patch.object(service, '_get_mongodb', return_value=mock_mongodb_service):
            results = await service.delete_repository_data(
                repository="TestRepo",
                collections=["code_methods", "code_classes"]
            )

            assert len(results) == 2
            mock_mongodb_service.db["code_methods"].delete_many.assert_called_once()
            mock_mongodb_service.db["code_classes"].delete_many.assert_called_once()

    @pytest.mark.asyncio
    async def test_delete_repository_data_returns_counts(
        self,
        pipeline_config,
        mock_mongodb_service
    ):
        """Test that deletion returns correct counts."""
        mock_mongodb_service.db["code_methods"].delete_many.return_value = Mock(deleted_count=50)
        mock_mongodb_service.db["code_classes"].delete_many.return_value = Mock(deleted_count=10)

        service = CodeImportService(config=pipeline_config)

        with patch.object(service, '_get_mongodb', return_value=mock_mongodb_service):
            results = await service.delete_repository_data(
                repository="TestRepo",
                collections=["code_methods", "code_classes"]
            )

            assert results["code_methods"] == 50
            assert results["code_classes"] == 10


@pytest.mark.skipif(not HAS_IMPORT_SERVICE, reason="CodeImportService not available")
class TestImportResultModel:
    """Test ImportResult model."""

    def test_import_result_defaults(self):
        """Test ImportResult default values."""
        result = ImportResult()

        assert result.success is True
        assert result.documents_imported == 0
        assert result.documents_updated == 0
        assert result.documents_deleted == 0
        assert result.errors == []
        assert result.duration_seconds == 0.0

    def test_import_result_custom_values(self):
        """Test ImportResult with custom values."""
        result = ImportResult(
            success=True,
            documents_imported=100,
            documents_updated=25,
            documents_deleted=5,
            errors=["Warning: duplicate key"],
            duration_seconds=5.5
        )

        assert result.success is True
        assert result.documents_imported == 100
        assert result.documents_updated == 25
        assert result.documents_deleted == 5
        assert len(result.errors) == 1
        assert result.duration_seconds == 5.5


@pytest.mark.skipif(not HAS_IMPORT_SERVICE, reason="CodeImportService not available")
class TestCodeImportServiceIntegration:
    """Integration tests for CodeImportService."""

    @pytest.mark.e2e
    @pytest.mark.asyncio
    async def test_full_import_workflow(
        self,
        pipeline_config,
        sample_analysis_output,
        mock_mongodb_service,
        mock_embedding_service
    ):
        """Test complete import workflow."""
        service = CodeImportService(config=pipeline_config)

        with patch.object(service, '_get_mongodb', return_value=mock_mongodb_service):
            with patch.object(service, '_get_embedding_service', return_value=mock_embedding_service):
                # Import analysis
                result = await service.import_analysis(
                    analysis=sample_analysis_output,
                    db_name="testrepo"
                )

                assert result.success is True
                assert result.documents_imported > 0

                # Verify collections were accessed
                # Classes and interfaces go to code_classes
                assert mock_mongodb_service.db["code_classes"].update_one.called
                # Methods go to code_methods
                assert mock_mongodb_service.db["code_methods"].update_one.called

    @pytest.mark.e2e
    @pytest.mark.asyncio
    async def test_import_and_delete_workflow(
        self,
        pipeline_config,
        sample_analysis_output,
        mock_mongodb_service,
        mock_embedding_service
    ):
        """Test import followed by delete workflow."""
        service = CodeImportService(config=pipeline_config)

        with patch.object(service, '_get_mongodb', return_value=mock_mongodb_service):
            with patch.object(service, '_get_embedding_service', return_value=mock_embedding_service):
                # Import analysis
                import_result = await service.import_analysis(
                    analysis=sample_analysis_output,
                    db_name="testrepo"
                )
                assert import_result.success is True

                # Delete repository data
                delete_results = await service.delete_repository_data("TestRepo")

                # All collections should have been cleaned
                assert len(delete_results) == 5

    @pytest.mark.e2e
    @pytest.mark.asyncio
    async def test_batch_import_large_analysis(
        self,
        pipeline_config,
        mock_mongodb_service,
        mock_embedding_service
    ):
        """Test importing large analysis with batch processing."""
        # Create large analysis output
        large_output = RoslynAnalysisOutput(
            repository="LargeRepo",
            file_count=100,
            classes=[
                CodeEntity(type="class", name=f"Class{i}", file_path=f"/src/class{i}.cs")
                for i in range(150)  # More than batch size
            ],
            methods=[
                CodeEntity(type="method", name=f"Method{i}", file_path=f"/src/method{i}.cs")
                for i in range(200)
            ],
            properties=[],
            interfaces=[],
            enums=[]
        )

        service = CodeImportService(config=pipeline_config)

        with patch.object(service, '_get_mongodb', return_value=mock_mongodb_service):
            with patch.object(service, '_get_embedding_service', return_value=mock_embedding_service):
                result = await service.import_analysis(
                    analysis=large_output,
                    db_name="largerepo"
                )

                assert result.success is True
                # 150 classes + 200 methods = 350 total
                assert result.documents_imported == 350

    @pytest.mark.e2e
    @pytest.mark.asyncio
    async def test_upsert_behavior(
        self,
        pipeline_config,
        mock_mongodb_service,
        mock_embedding_service
    ):
        """Test upsert behavior - updates existing, inserts new."""
        # Configure mock to return upserted_id for first call, modified_count for second
        call_count = [0]

        async def mock_update_one(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] % 2 == 1:
                return Mock(upserted_id="new_id", modified_count=0)
            else:
                return Mock(upserted_id=None, modified_count=1)

        mock_mongodb_service.db["code_classes"].update_one = mock_update_one

        entities = [
            CodeEntity(type="class", name="NewClass", file_path="/src/new.cs"),
            CodeEntity(type="class", name="ExistingClass", file_path="/src/existing.cs"),
        ]

        service = CodeImportService(config=pipeline_config)

        stats = await service._import_entities(
            entities=entities,
            collection_name="code_classes",
            repository="TestRepo",
            db_name="testrepo",
            mongodb=mock_mongodb_service,
            embedding_service=mock_embedding_service
        )

        # One new, one updated
        assert stats["imported"] == 1
        assert stats["updated"] == 1
