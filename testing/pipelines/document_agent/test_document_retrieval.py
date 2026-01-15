"""
Document Retrieval Tests
========================

Tests for querying and retrieving documents and chunks from MongoDB.
Tests include searching by filename, content, metadata, and vector similarity.
"""

import pytest
from datetime import datetime, timedelta
from typing import List, Dict

from config.test_config import get_test_config
from utils import generate_test_id


class TestDocumentRetrieval:
    """Test document retrieval operations."""

    @pytest.fixture(autouse=True)
    def setup_test_data(self, mongodb_database, pipeline_config):
        """Set up test documents and chunks for retrieval tests."""
        self.docs_collection = mongodb_database["documents"]
        self.chunks_collection = mongodb_database["document_chunks"]
        self.test_doc_ids = []
        self.test_chunk_ids = []

        # Create varied test documents
        test_documents = [
            {
                "_id": f"test_{generate_test_id('doc_pdf')}",
                "filename": "technical_manual.pdf",
                "file_type": "pdf",
                "content": "This is a technical manual about software architecture.",
                "metadata": {
                    "title": "Software Architecture Manual",
                    "author": "Engineering Team",
                    "page_count": 50,
                    "has_tables": True,
                    "file_hash": "pdf_hash_1"
                },
                "file_hash": "pdf_hash_1",
                "file_size_bytes": 500000,
                "chunks": [],
                "created_at": datetime.utcnow() - timedelta(hours=2)
            },
            {
                "_id": f"test_{generate_test_id('doc_docx')}",
                "filename": "quarterly_report.docx",
                "file_type": "docx",
                "content": "Quarterly business report with financial tables and charts.",
                "metadata": {
                    "title": "Q4 Business Report",
                    "author": "Finance Team",
                    "page_count": 25,
                    "has_tables": True,
                    "table_count": 5,
                    "file_hash": "docx_hash_1"
                },
                "file_hash": "docx_hash_1",
                "file_size_bytes": 150000,
                "chunks": [],
                "created_at": datetime.utcnow() - timedelta(hours=1)
            },
            {
                "_id": f"test_{generate_test_id('doc_txt')}",
                "filename": "meeting_notes.txt",
                "file_type": "text",
                "content": "Meeting notes from team standup discussing project status.",
                "metadata": {
                    "title": "",
                    "author": "",
                    "page_count": 0,
                    "has_tables": False,
                    "file_hash": "txt_hash_1"
                },
                "file_hash": "txt_hash_1",
                "file_size_bytes": 2000,
                "chunks": [],
                "created_at": datetime.utcnow()
            }
        ]

        # Add common fields and insert
        for doc in test_documents:
            doc.update({
                "is_test": True,
                "test_run_id": pipeline_config.test_run_id
            })
            self.test_doc_ids.append(doc["_id"])

        self.docs_collection.insert_many(test_documents)

        # Create some test chunks
        for i, doc_id in enumerate(self.test_doc_ids[:2]):  # Only first 2 docs
            for j in range(3):
                chunk_id = f"test_{generate_test_id(f'chunk_{i}_{j}')}"
                self.test_chunk_ids.append(chunk_id)

                self.chunks_collection.insert_one({
                    "_id": chunk_id,
                    "document_id": doc_id,
                    "chunk_index": j,
                    "content": f"Chunk {j} content from document {i}",
                    "embedding": [0.1 * (i + j)] * 384,
                    "metadata": {
                        "page_number": j + 1,
                        "start_char": j * 500,
                        "end_char": (j + 1) * 500
                    },
                    "created_at": datetime.utcnow(),
                    "is_test": True,
                    "test_run_id": pipeline_config.test_run_id
                })

        yield

        # Cleanup
        self.docs_collection.delete_many({"_id": {"$in": self.test_doc_ids}})
        self.chunks_collection.delete_many({"_id": {"$in": self.test_chunk_ids}})

    @pytest.mark.requires_mongodb
    def test_retrieve_by_filename(self, mongodb_database, pipeline_config):
        """Test retrieving document by exact filename."""
        result = self.docs_collection.find_one({
            "filename": "technical_manual.pdf",
            "is_test": True
        })

        assert result is not None
        assert result["filename"] == "technical_manual.pdf"
        assert result["file_type"] == "pdf"

    @pytest.mark.requires_mongodb
    def test_retrieve_by_file_type(self, mongodb_database, pipeline_config):
        """Test retrieving documents by file type."""
        # Find all PDFs
        pdfs = list(self.docs_collection.find({
            "file_type": "pdf",
            "is_test": True,
            "test_run_id": pipeline_config.test_run_id
        }))

        assert len(pdfs) >= 1
        assert all(doc["file_type"] == "pdf" for doc in pdfs)

        # Find all DOCx
        docx_files = list(self.docs_collection.find({
            "file_type": "docx",
            "is_test": True,
            "test_run_id": pipeline_config.test_run_id
        }))

        assert len(docx_files) >= 1

    @pytest.mark.requires_mongodb
    def test_retrieve_by_title(self, mongodb_database, pipeline_config):
        """Test retrieving document by title in metadata."""
        result = self.docs_collection.find_one({
            "metadata.title": "Software Architecture Manual",
            "is_test": True
        })

        assert result is not None
        assert result["metadata"]["title"] == "Software Architecture Manual"

    @pytest.mark.requires_mongodb
    def test_retrieve_by_author(self, mongodb_database, pipeline_config):
        """Test retrieving documents by author."""
        results = list(self.docs_collection.find({
            "metadata.author": "Engineering Team",
            "is_test": True
        }))

        assert len(results) >= 1
        assert results[0]["metadata"]["author"] == "Engineering Team"

    @pytest.mark.requires_mongodb
    def test_retrieve_documents_with_tables(self, mongodb_database, pipeline_config):
        """Test retrieving documents that contain tables."""
        results = list(self.docs_collection.find({
            "metadata.has_tables": True,
            "is_test": True,
            "test_run_id": pipeline_config.test_run_id
        }))

        assert len(results) >= 2
        for doc in results:
            assert doc["metadata"]["has_tables"] is True

    @pytest.mark.requires_mongodb
    def test_retrieve_by_file_hash(self, mongodb_database, pipeline_config):
        """Test retrieving document by file hash (deduplication check)."""
        result = self.docs_collection.find_one({
            "file_hash": "pdf_hash_1",
            "is_test": True
        })

        assert result is not None
        assert result["file_hash"] == "pdf_hash_1"

    @pytest.mark.requires_mongodb
    def test_text_search_in_content(self, mongodb_database, pipeline_config):
        """Test searching document content using regex."""
        # Search for "software"
        results = list(self.docs_collection.find({
            "content": {"$regex": "software", "$options": "i"},
            "is_test": True,
            "test_run_id": pipeline_config.test_run_id
        }))

        assert len(results) >= 1
        assert "software" in results[0]["content"].lower()

        # Search for "financial"
        financial_docs = list(self.docs_collection.find({
            "content": {"$regex": "financial", "$options": "i"},
            "is_test": True
        }))

        assert len(financial_docs) >= 1

    @pytest.mark.requires_mongodb
    def test_retrieve_by_page_count_range(self, mongodb_database, pipeline_config):
        """Test retrieving documents by page count range."""
        # Find documents with >20 pages
        large_docs = list(self.docs_collection.find({
            "metadata.page_count": {"$gt": 20},
            "is_test": True,
            "test_run_id": pipeline_config.test_run_id
        }))

        assert len(large_docs) >= 2
        for doc in large_docs:
            assert doc["metadata"]["page_count"] > 20

    @pytest.mark.requires_mongodb
    def test_retrieve_by_file_size(self, mongodb_database, pipeline_config):
        """Test retrieving documents by file size."""
        # Find large files (>100KB)
        large_files = list(self.docs_collection.find({
            "file_size_bytes": {"$gt": 100000},
            "is_test": True,
            "test_run_id": pipeline_config.test_run_id
        }))

        assert len(large_files) >= 2

    @pytest.mark.requires_mongodb
    def test_retrieve_recent_documents(self, mongodb_database, pipeline_config):
        """Test retrieving recently added documents."""
        one_hour_ago = datetime.utcnow() - timedelta(hours=1, minutes=30)

        recent = list(self.docs_collection.find({
            "created_at": {"$gte": one_hour_ago},
            "is_test": True,
            "test_run_id": pipeline_config.test_run_id
        }).sort("created_at", -1))

        assert len(recent) >= 2

    @pytest.mark.requires_mongodb
    def test_retrieve_chunks_by_document(self, mongodb_database, pipeline_config):
        """Test retrieving all chunks for a specific document."""
        doc_id = self.test_doc_ids[0]

        chunks = list(self.chunks_collection.find({
            "document_id": doc_id,
            "is_test": True
        }).sort("chunk_index", 1))

        assert len(chunks) >= 3
        # Verify ordering
        for i, chunk in enumerate(chunks):
            assert chunk["chunk_index"] == i

    @pytest.mark.requires_mongodb
    def test_retrieve_chunk_by_index(self, mongodb_database, pipeline_config):
        """Test retrieving a specific chunk by index."""
        doc_id = self.test_doc_ids[0]

        chunk = self.chunks_collection.find_one({
            "document_id": doc_id,
            "chunk_index": 1,
            "is_test": True
        })

        assert chunk is not None
        assert chunk["chunk_index"] == 1

    @pytest.mark.requires_mongodb
    def test_search_chunks_by_content(self, mongodb_database, pipeline_config):
        """Test searching chunk content."""
        results = list(self.chunks_collection.find({
            "content": {"$regex": "Chunk.*content", "$options": "i"},
            "is_test": True,
            "test_run_id": pipeline_config.test_run_id
        }))

        assert len(results) >= 3

    @pytest.mark.requires_mongodb
    def test_retrieve_with_projection(self, mongodb_database, pipeline_config):
        """Test retrieving specific fields only."""
        results = list(self.docs_collection.find(
            {"is_test": True, "test_run_id": pipeline_config.test_run_id},
            {
                "filename": 1,
                "file_type": 1,
                "metadata.title": 1,
                "_id": 0
            }
        ))

        assert len(results) >= 3
        for doc in results:
            assert "filename" in doc
            assert "file_type" in doc
            assert "_id" not in doc
            assert "content" not in doc

    @pytest.mark.requires_mongodb
    def test_aggregate_by_file_type(self, mongodb_database, pipeline_config):
        """Test aggregating documents by file type."""
        pipeline = [
            {"$match": {"is_test": True, "test_run_id": pipeline_config.test_run_id}},
            {
                "$group": {
                    "_id": "$file_type",
                    "count": {"$sum": 1},
                    "total_size": {"$sum": "$file_size_bytes"}
                }
            }
        ]

        results = list(self.docs_collection.aggregate(pipeline))

        assert len(results) >= 2
        file_types = {r["_id"]: r["count"] for r in results}
        assert "pdf" in file_types or "docx" in file_types

    @pytest.mark.requires_mongodb
    def test_count_documents_by_criteria(self, mongodb_database, pipeline_config):
        """Test counting documents matching criteria."""
        # Count PDFs
        pdf_count = self.docs_collection.count_documents({
            "file_type": "pdf",
            "is_test": True,
            "test_run_id": pipeline_config.test_run_id
        })

        assert pdf_count >= 1

        # Count documents with tables
        tables_count = self.docs_collection.count_documents({
            "metadata.has_tables": True,
            "is_test": True,
            "test_run_id": pipeline_config.test_run_id
        })

        assert tables_count >= 2

    @pytest.mark.requires_mongodb
    def test_retrieve_sorted_by_size(self, mongodb_database, pipeline_config):
        """Test retrieving documents sorted by file size."""
        results = list(self.docs_collection.find({
            "is_test": True,
            "test_run_id": pipeline_config.test_run_id
        }).sort("file_size_bytes", -1))

        assert len(results) >= 3
        # Verify descending order
        for i in range(len(results) - 1):
            assert results[i]["file_size_bytes"] >= results[i + 1]["file_size_bytes"]

    @pytest.mark.requires_mongodb
    def test_retrieve_sorted_by_date(self, mongodb_database, pipeline_config):
        """Test retrieving documents sorted by creation date."""
        results = list(self.docs_collection.find({
            "is_test": True,
            "test_run_id": pipeline_config.test_run_id
        }).sort("created_at", -1))

        assert len(results) >= 3
        # Verify most recent first
        for i in range(len(results) - 1):
            assert results[i]["created_at"] >= results[i + 1]["created_at"]

    @pytest.mark.requires_mongodb
    def test_paginated_retrieval(self, mongodb_database, pipeline_config):
        """Test paginated document retrieval."""
        page_size = 2
        page_1 = list(self.docs_collection.find({
            "is_test": True,
            "test_run_id": pipeline_config.test_run_id
        }).sort("created_at", -1).limit(page_size))

        page_2 = list(self.docs_collection.find({
            "is_test": True,
            "test_run_id": pipeline_config.test_run_id
        }).sort("created_at", -1).skip(page_size).limit(page_size))

        assert len(page_1) == 2
        assert len(page_2) >= 1

        # Verify no overlap
        page_1_ids = {doc["_id"] for doc in page_1}
        page_2_ids = {doc["_id"] for doc in page_2}
        assert len(page_1_ids & page_2_ids) == 0

    @pytest.mark.requires_mongodb
    def test_retrieve_chunk_with_embedding(self, mongodb_database, pipeline_config):
        """Test retrieving chunk with embedding vector."""
        chunk = self.chunks_collection.find_one({
            "is_test": True,
            "test_run_id": pipeline_config.test_run_id
        })

        assert chunk is not None
        assert "embedding" in chunk
        assert isinstance(chunk["embedding"], list)
        assert len(chunk["embedding"]) == 384

    @pytest.mark.requires_mongodb
    def test_find_similar_chunks_by_embedding(self, mongodb_database, pipeline_config):
        """Test simulating vector similarity search."""
        # Get a reference chunk
        reference_chunk = self.chunks_collection.find_one({
            "is_test": True,
            "test_run_id": pipeline_config.test_run_id
        })

        assert reference_chunk is not None
        reference_embedding = reference_chunk["embedding"]

        # In a real implementation, you would use MongoDB vector search
        # Here we just verify we can retrieve chunks for comparison
        all_chunks = list(self.chunks_collection.find({
            "is_test": True,
            "test_run_id": pipeline_config.test_run_id
        }))

        assert len(all_chunks) >= 3
        for chunk in all_chunks:
            assert "embedding" in chunk
            assert len(chunk["embedding"]) == len(reference_embedding)
