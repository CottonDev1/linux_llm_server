"""
Query/RAG Storage Tests for document and code context collections.

Tests document and embedding storage operations including:
- Storing documents with embeddings
- Storing code context (procedures, snippets)
- Storing knowledge base documents
- Vector embeddings storage
- Metadata and search fields

All tests use MongoDB only (no external dependencies).
All test data is prefixed with 'test_' for isolation.
"""

import pytest
from datetime import datetime
from typing import List

from config.test_config import PipelineTestConfig
from fixtures.mongodb_fixtures import (
    create_mock_document,
    create_mock_document_chunk,
    create_mock_code_method,
    insert_test_documents,
    cleanup_test_documents,
)
from utils import (
    assert_document_stored,
    assert_mongodb_document,
    generate_test_id,
)


class TestDocumentStorage:
    """Test document storage in knowledge base collection."""

    @pytest.mark.requires_mongodb
    def test_store_document_chunk(
        self, mongodb_database, pipeline_config: PipelineTestConfig
    ):
        """Test storing a document chunk with embedding."""
        collection = mongodb_database["documents"]

        # Create document chunk with embedding
        doc = create_mock_document_chunk(
            title="EWR System Overview",
            content="The EWR system manages cotton processing operations...",
            doc_type="pdf",
            chunk_index=0,
        )
        doc["test_run_id"] = pipeline_config.test_run_id
        doc["embedding"] = [0.1, 0.2, 0.3] * 128  # Mock 384-dim embedding
        doc["department"] = "IT"
        doc["relevance_score"] = 0.0

        collection.insert_one(doc)

        # Verify stored
        stored_doc = assert_document_stored(
            collection,
            doc["_id"],
            expected_fields=["title", "content", "embedding", "department"],
        )

        assert stored_doc["title"] == "EWR System Overview"
        assert stored_doc["source_type"] == "pdf"
        assert len(stored_doc["embedding"]) == 384
        assert stored_doc["department"] == "IT"

        # Cleanup
        cleanup_test_documents(mongodb_database, pipeline_config.test_run_id)

    @pytest.mark.requires_mongodb
    def test_store_multi_chunk_document(
        self, mongodb_database, pipeline_config: PipelineTestConfig
    ):
        """Test storing multiple chunks from same document."""
        collection = mongodb_database["documents"]

        # Create multiple chunks
        chunks = []
        for i in range(5):
            chunk = create_mock_document_chunk(
                title="Large Document",
                content=f"This is chunk {i} of the document...",
                doc_type="pdf",
                chunk_index=i,
            )
            chunk["test_run_id"] = pipeline_config.test_run_id
            chunk["total_chunks"] = 5
            chunk["embedding"] = [float(i) / 10] * 384
            chunks.append(chunk)

        insert_test_documents(collection, chunks, pipeline_config.test_run_id)

        # Verify all chunks stored
        stored_chunks = list(
            collection.find({"test_run_id": pipeline_config.test_run_id}).sort(
                "chunk_index", 1
            )
        )

        assert len(stored_chunks) == 5
        for i, chunk in enumerate(stored_chunks):
            assert chunk["chunk_index"] == i
            assert chunk["total_chunks"] == 5

        # Cleanup
        cleanup_test_documents(mongodb_database, pipeline_config.test_run_id)

    @pytest.mark.requires_mongodb
    def test_store_document_with_metadata(
        self, mongodb_database, pipeline_config: PipelineTestConfig
    ):
        """Test storing document with rich metadata."""
        collection = mongodb_database["documents"]

        doc = create_mock_document_chunk(
            title="Safety Procedures Manual",
            content="Safety guidelines for warehouse operations...",
            doc_type="pdf",
            chunk_index=0,
        )
        doc["test_run_id"] = pipeline_config.test_run_id
        doc["embedding"] = [0.5] * 384
        doc["metadata"] = {
            "author": "Safety Department",
            "version": "2.1",
            "last_updated": "2024-01-15",
            "tags": ["safety", "warehouse", "procedures"],
            "classification": "internal",
        }
        doc["department"] = "Safety"
        doc["file_name"] = "safety_procedures_v2.1.pdf"

        collection.insert_one(doc)

        # Verify metadata stored
        stored_doc = assert_document_stored(collection, doc["_id"])
        assert "metadata" in stored_doc
        assert stored_doc["metadata"]["author"] == "Safety Department"
        assert stored_doc["metadata"]["version"] == "2.1"
        assert "safety" in stored_doc["metadata"]["tags"]

        # Cleanup
        cleanup_test_documents(mongodb_database, pipeline_config.test_run_id)


class TestCodeContextStorage:
    """Test code context storage for code snippets and procedures."""

    @pytest.mark.requires_mongodb
    def test_store_code_method(
        self, mongodb_database, pipeline_config: PipelineTestConfig
    ):
        """Test storing a code method/function."""
        collection = mongodb_database["code_context"]

        doc = create_mock_code_method(
            method_name="ProcessBale",
            class_name="BaleProcessor",
            project="gin",
            code="public void ProcessBale(Bale bale) { /* implementation */ }",
        )
        doc["test_run_id"] = pipeline_config.test_run_id
        doc["embedding"] = [0.7] * 384

        collection.insert_one(doc)

        # Verify stored
        stored_doc = assert_document_stored(
            collection,
            doc["_id"],
            expected_fields=["method_name", "class_name", "project", "content"],
        )

        assert stored_doc["method_name"] == "ProcessBale"
        assert stored_doc["class_name"] == "BaleProcessor"
        assert stored_doc["project"] == "gin"
        assert "void ProcessBale" in stored_doc["content"]

        # Cleanup
        cleanup_test_documents(mongodb_database, pipeline_config.test_run_id)

    @pytest.mark.requires_mongodb
    def test_store_stored_procedure(
        self, mongodb_database, pipeline_config: PipelineTestConfig
    ):
        """Test storing SQL stored procedure."""
        collection = mongodb_database["code_context"]

        proc_code = """CREATE PROCEDURE RecapGet
    @GinID INT
AS
BEGIN
    SELECT * FROM Recap WHERE GinID = @GinID
END"""

        doc = create_mock_document(
            doc_type="stored_procedure",
            pipeline="sql",
            content=proc_code,
        )
        doc["test_run_id"] = pipeline_config.test_run_id
        doc["procedure_name"] = "RecapGet"
        doc["database"] = "EWR.Gin.Bobby:B-STREET"
        doc["project"] = "gin"
        doc["parameters"] = ["@GinID INT"]
        doc["embedding"] = [0.3] * 384

        collection.insert_one(doc)

        # Verify stored
        stored_doc = assert_document_stored(collection, doc["_id"])
        assert stored_doc["procedure_name"] == "RecapGet"
        assert stored_doc["database"] == "EWR.Gin.Bobby:B-STREET"
        assert "@GinID" in stored_doc["content"]

        # Cleanup
        cleanup_test_documents(mongodb_database, pipeline_config.test_run_id)

    @pytest.mark.requires_mongodb
    def test_store_configuration_snippet(
        self, mongodb_database, pipeline_config: PipelineTestConfig
    ):
        """Test storing configuration file snippet."""
        collection = mongodb_database["code_context"]

        config_content = """{
  "database": {
    "server": "NCSQLTEST",
    "name": "EWRCentral"
  }
}"""

        doc = create_mock_document(
            doc_type="configuration",
            pipeline="git",
            content=config_content,
        )
        doc["test_run_id"] = pipeline_config.test_run_id
        doc["file_path"] = "/config/database.json"
        doc["project"] = "warehouse"
        doc["language"] = "json"
        doc["embedding"] = [0.6] * 384

        collection.insert_one(doc)

        # Verify stored
        stored_doc = assert_document_stored(collection, doc["_id"])
        assert stored_doc["type"] == "configuration"
        assert stored_doc["file_path"] == "/config/database.json"
        assert "NCSQLTEST" in stored_doc["content"]

        # Cleanup
        cleanup_test_documents(mongodb_database, pipeline_config.test_run_id)

    @pytest.mark.requires_mongodb
    def test_store_multiple_projects(
        self, mongodb_database, pipeline_config: PipelineTestConfig
    ):
        """Test storing code from multiple projects."""
        collection = mongodb_database["code_context"]

        projects = ["gin", "warehouse", "marketing", "EWRLibrary"]
        docs = []

        for project in projects:
            doc = create_mock_code_method(
                method_name=f"{project}Method",
                class_name=f"{project}Class",
                project=project,
                code=f"public void {project}Method() {{ }}",
            )
            doc["test_run_id"] = pipeline_config.test_run_id
            doc["embedding"] = [float(len(project)) / 10] * 384
            docs.append(doc)

        insert_test_documents(collection, docs, pipeline_config.test_run_id)

        # Verify each project stored
        for project in projects:
            result = collection.find_one(
                {"project": project, "test_run_id": pipeline_config.test_run_id}
            )
            assert result is not None
            assert result["project"] == project

        # Cleanup
        cleanup_test_documents(mongodb_database, pipeline_config.test_run_id)


class TestEmbeddingStorage:
    """Test embedding vector storage and validation."""

    @pytest.mark.requires_mongodb
    def test_embedding_dimension_validation(
        self, mongodb_database, pipeline_config: PipelineTestConfig
    ):
        """Test that embeddings have correct dimensions."""
        collection = mongodb_database["documents"]

        # Standard embedding dimension for all-MiniLM-L6-v2 is 384
        expected_dim = 384

        doc = create_mock_document_chunk(
            title="Test Doc", content="Test content", chunk_index=0
        )
        doc["test_run_id"] = pipeline_config.test_run_id
        doc["embedding"] = [0.5] * expected_dim

        collection.insert_one(doc)

        # Verify embedding dimension
        stored = collection.find_one({"_id": doc["_id"]})
        assert len(stored["embedding"]) == expected_dim

        # Cleanup
        cleanup_test_documents(mongodb_database, pipeline_config.test_run_id)

    @pytest.mark.requires_mongodb
    def test_embedding_normalization(
        self, mongodb_database, pipeline_config: PipelineTestConfig
    ):
        """Test that embeddings are stored as normalized vectors."""
        collection = mongodb_database["documents"]

        # Create normalized embedding (L2 norm = 1)
        import math

        raw_vector = [1.0, 2.0, 3.0, 4.0] * 96  # 384 dims
        norm = math.sqrt(sum(x * x for x in raw_vector))
        normalized = [x / norm for x in raw_vector]

        doc = create_mock_document_chunk(
            title="Normalized Test", content="Content", chunk_index=0
        )
        doc["test_run_id"] = pipeline_config.test_run_id
        doc["embedding"] = normalized

        collection.insert_one(doc)

        # Verify normalization
        stored = collection.find_one({"_id": doc["_id"]})
        stored_norm = math.sqrt(sum(x * x for x in stored["embedding"]))
        assert abs(stored_norm - 1.0) < 0.001, "Embedding should be normalized"

        # Cleanup
        cleanup_test_documents(mongodb_database, pipeline_config.test_run_id)

    @pytest.mark.requires_mongodb
    def test_batch_embedding_storage(
        self, mongodb_database, pipeline_config: PipelineTestConfig
    ):
        """Test efficient batch storage of embeddings."""
        collection = mongodb_database["code_context"]

        # Create batch of documents with embeddings
        batch_size = 50
        docs = []

        for i in range(batch_size):
            doc = create_mock_code_method(
                method_name=f"Method{i}",
                class_name="TestClass",
                project="gin",
                code=f"public void Method{i}() {{ }}",
            )
            doc["test_run_id"] = pipeline_config.test_run_id
            doc["embedding"] = [float(i) / batch_size] * 384
            docs.append(doc)

        # Batch insert
        doc_ids = insert_test_documents(
            collection, docs, pipeline_config.test_run_id
        )

        assert len(doc_ids) == batch_size

        # Verify all stored with embeddings
        count = collection.count_documents(
            {
                "test_run_id": pipeline_config.test_run_id,
                "embedding": {"$exists": True},
            }
        )
        assert count == batch_size

        # Cleanup
        cleanup_test_documents(mongodb_database, pipeline_config.test_run_id)


class TestStorageSchemaValidation:
    """Test schema validation for stored documents."""

    @pytest.mark.requires_mongodb
    def test_document_schema_validation(
        self, mongodb_database, pipeline_config: PipelineTestConfig
    ):
        """Test document conforms to expected schema."""
        collection = mongodb_database["documents"]

        doc = create_mock_document_chunk(
            title="Schema Test", content="Content", chunk_index=0
        )
        doc["test_run_id"] = pipeline_config.test_run_id
        doc["embedding"] = [0.5] * 384
        doc["department"] = "IT"
        collection.insert_one(doc)

        stored = collection.find_one({"_id": doc["_id"]})

        # Define expected schema
        schema = {
            "title": str,
            "content": str,
            "embedding": list,
            "department": str,
            "chunk_index": int,
            "is_test": bool,
        }

        assert_mongodb_document(stored, schema, allow_extra=True)

        # Cleanup
        cleanup_test_documents(mongodb_database, pipeline_config.test_run_id)

    @pytest.mark.requires_mongodb
    def test_code_context_schema_validation(
        self, mongodb_database, pipeline_config: PipelineTestConfig
    ):
        """Test code context conforms to expected schema."""
        collection = mongodb_database["code_context"]

        doc = create_mock_code_method(
            method_name="TestMethod",
            class_name="TestClass",
            project="gin",
            code="public void TestMethod() { }",
        )
        doc["test_run_id"] = pipeline_config.test_run_id
        doc["embedding"] = [0.5] * 384
        collection.insert_one(doc)

        stored = collection.find_one({"_id": doc["_id"]})

        # Define expected schema
        schema = {
            "method_name": str,
            "class_name": str,
            "project": str,
            "content": str,
            "embedding": list,
            "is_test": bool,
        }

        assert_mongodb_document(stored, schema, allow_extra=True)

        # Cleanup
        cleanup_test_documents(mongodb_database, pipeline_config.test_run_id)
