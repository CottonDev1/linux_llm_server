"""
Query/RAG Storage Tests - Using Real Data
==========================================

Tests document and embedding storage validation using REAL data from the database.
"""

import pytest
from typing import Dict, Any, List

from config.test_config import get_test_config, PipelineTestConfig
from fixtures.mongodb_fixtures import (
    get_test_database,
    get_real_document,
    get_real_documents,
    get_real_code_method,
    get_real_code_methods,
)


class TestDocumentStorageExists:
    """Verify documents collection has data."""

    @pytest.mark.requires_mongodb
    def test_documents_collection_has_data(self, mongodb_database):
        """Verify documents collection exists and has documents."""
        collection = mongodb_database["documents"]
        count = collection.count_documents({})
        assert count > 0, "documents collection should have documents"

    @pytest.mark.requires_mongodb
    def test_documents_have_content(self, mongodb_database):
        """Verify documents have content field."""
        doc = get_real_document(mongodb_database)
        assert doc is not None, "Should have at least one document"
        assert "content" in doc, "Document should have content field"
        assert len(doc["content"]) > 0, "Content should not be empty"

    @pytest.mark.requires_mongodb
    def test_documents_have_embeddings(self, mongodb_database):
        """Verify documents have embedding/vector field."""
        collection = mongodb_database["documents"]

        # Find document with vector field
        doc = collection.find_one({"vector": {"$exists": True}})

        if doc:
            assert "vector" in doc
            assert isinstance(doc["vector"], list)
            assert len(doc["vector"]) > 0, "Vector should not be empty"

    @pytest.mark.requires_mongodb
    def test_documents_have_metadata(self, mongodb_database):
        """Verify documents have metadata fields."""
        doc = get_real_document(mongodb_database)
        assert doc is not None

        # Check for common metadata fields
        assert "_id" in doc
        # At least one metadata field should exist
        metadata_fields = ["title", "file_name", "department", "type", "tags"]
        has_metadata = any(f in doc for f in metadata_fields)
        assert has_metadata, f"Document should have metadata. Found: {list(doc.keys())}"


class TestDocumentStructure:
    """Test structure of stored documents."""

    @pytest.mark.requires_mongodb
    def test_document_has_title(self, mongodb_database):
        """Test that documents have title field."""
        collection = mongodb_database["documents"]
        doc = collection.find_one({"title": {"$exists": True}})

        if doc:
            assert isinstance(doc["title"], str)
            assert len(doc["title"]) > 0

    @pytest.mark.requires_mongodb
    def test_document_has_chunk_info(self, mongodb_database):
        """Test that chunked documents have chunk metadata."""
        collection = mongodb_database["documents"]
        doc = collection.find_one({"chunk_index": {"$exists": True}})

        if doc:
            assert "chunk_index" in doc
            assert "total_chunks" in doc or "parent_id" in doc

    @pytest.mark.requires_mongodb
    def test_documents_have_department(self, mongodb_database):
        """Test that documents have department field."""
        collection = mongodb_database["documents"]
        doc = collection.find_one({"department": {"$exists": True}})

        if doc:
            assert isinstance(doc["department"], str)


class TestCodeContextStorage:
    """Test code context storage."""

    @pytest.mark.requires_mongodb
    def test_code_methods_collection_has_data(self, mongodb_database):
        """Verify code_methods collection exists and has documents."""
        collection = mongodb_database["code_methods"]
        count = collection.count_documents({})
        assert count > 0, "code_methods collection should have documents"

    @pytest.mark.requires_mongodb
    def test_code_method_has_required_fields(self, mongodb_database):
        """Test that code methods have required fields."""
        method = get_real_code_method(mongodb_database)
        assert method is not None, "Should have at least one code method"
        assert "_id" in method

    @pytest.mark.requires_mongodb
    def test_code_method_has_name(self, mongodb_database):
        """Test that code methods have method_name field."""
        collection = mongodb_database["code_methods"]
        method = collection.find_one({"method_name": {"$exists": True}})

        if method:
            assert isinstance(method["method_name"], str)
            assert len(method["method_name"]) > 0

    @pytest.mark.requires_mongodb
    def test_code_method_has_class(self, mongodb_database):
        """Test that code methods have class_name field."""
        collection = mongodb_database["code_methods"]
        method = collection.find_one({"class_name": {"$exists": True}})

        if method:
            assert isinstance(method["class_name"], str)

    @pytest.mark.requires_mongodb
    def test_code_method_has_project(self, mongodb_database):
        """Test that code methods have project field."""
        collection = mongodb_database["code_methods"]
        method = collection.find_one({"project": {"$exists": True}})

        if method:
            assert isinstance(method["project"], str)


class TestEmbeddingStorage:
    """Test embedding vector storage."""

    @pytest.mark.requires_mongodb
    def test_documents_have_vectors(self, mongodb_database):
        """Test that documents have vector embeddings."""
        collection = mongodb_database["documents"]
        count = collection.count_documents({"vector": {"$exists": True}})

        # At least some documents should have vectors
        assert count > 0, "Some documents should have vector embeddings"

    @pytest.mark.requires_mongodb
    def test_vector_dimension(self, mongodb_database):
        """Test that vectors have consistent dimensions."""
        collection = mongodb_database["documents"]
        doc = collection.find_one({"vector": {"$exists": True}})

        if doc and "vector" in doc:
            # Common embedding dimensions: 384 (MiniLM), 768 (BERT), 1536 (OpenAI)
            vector_len = len(doc["vector"])
            assert vector_len in [384, 768, 1024, 1536], f"Unexpected vector dimension: {vector_len}"

    @pytest.mark.requires_mongodb
    def test_vectors_are_float_lists(self, mongodb_database):
        """Test that vectors are lists of floats."""
        collection = mongodb_database["documents"]
        doc = collection.find_one({"vector": {"$exists": True}})

        if doc and "vector" in doc:
            vector = doc["vector"]
            assert isinstance(vector, list)
            # Check first few elements are numeric
            for val in vector[:10]:
                assert isinstance(val, (int, float))


class TestStorageQueryability:
    """Test that stored documents can be queried effectively."""

    @pytest.mark.requires_mongodb
    def test_query_documents_by_department(self, mongodb_database):
        """Test querying documents by department."""
        collection = mongodb_database["documents"]
        departments = collection.distinct("department")

        if departments:
            dept = departments[0]
            results = list(collection.find({"department": dept}).limit(5))
            assert len(results) > 0

    @pytest.mark.requires_mongodb
    def test_query_code_methods_by_project(self, mongodb_database):
        """Test querying code methods by project."""
        collection = mongodb_database["code_methods"]
        projects = collection.distinct("project")

        if projects:
            project = projects[0]
            results = list(collection.find({"project": project}).limit(5))
            assert len(results) > 0

    @pytest.mark.requires_mongodb
    def test_aggregation_documents_by_type(self, mongodb_database):
        """Test aggregation of documents by type."""
        collection = mongodb_database["documents"]

        pipeline = [
            {"$group": {"_id": "$type", "count": {"$sum": 1}}},
            {"$sort": {"count": -1}},
            {"$limit": 5}
        ]

        results = list(collection.aggregate(pipeline))
        assert len(results) > 0


class TestStorageCounts:
    """Verify expected data counts."""

    @pytest.mark.requires_mongodb
    def test_documents_count(self, mongodb_database):
        """Verify documents has expected minimum count."""
        collection = mongodb_database["documents"]
        count = collection.count_documents({})
        assert count >= 10, f"Expected at least 10 documents, got {count}"

    @pytest.mark.requires_mongodb
    def test_code_methods_count(self, mongodb_database):
        """Verify code_methods has expected minimum count."""
        collection = mongodb_database["code_methods"]
        count = collection.count_documents({})
        assert count >= 1000, f"Expected at least 1000 code methods, got {count}"
