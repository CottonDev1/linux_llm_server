"""
Document Storage Tests
======================

Tests for storing documents and chunks in MongoDB collections.

Collections:
- documents: Main document metadata and content
- document_chunks: Individual chunks for retrieval

Document Schema:
- _id: unique identifier
- filename: original file name
- file_type: pdf, docx, xlsx, txt, etc.
- content: full extracted text
- metadata: DocumentMetadata (title, author, page_count, etc.)
- file_hash: SHA256 hash for deduplication
- chunks: list of chunk IDs
- created_at: timestamp
- updated_at: timestamp

Chunk Schema:
- _id: unique identifier
- document_id: reference to parent document
- chunk_index: position in document
- content: chunk text
- embedding: vector for similarity search
- metadata: chunk-specific metadata
- created_at: timestamp
"""

import pytest
from datetime import datetime
from typing import Dict, List

from config.test_config import get_test_config
from utils import generate_test_id, assert_document_stored, assert_mongodb_document


class TestDocumentStorage:
    """Test document storage operations."""

    @pytest.mark.requires_mongodb
    def test_store_basic_document(self, mongodb_database, pipeline_config):
        """Test storing a basic document with metadata."""
        collection = mongodb_database["documents"]

        doc_id = f"test_{generate_test_id('doc')}"
        test_doc = {
            "_id": doc_id,
            "filename": "test_document.pdf",
            "file_type": "pdf",
            "content": "This is a test document content for PDF processing.",
            "metadata": {
                "title": "Test Document",
                "author": "Test Author",
                "created_date": "2025-01-15",
                "page_count": 5,
                "word_count": 250,
                "has_tables": False,
                "has_images": True,
                "table_count": 0,
                "language": "en",
                "file_hash": "abc123hash"
            },
            "file_hash": "abc123hash",
            "file_size_bytes": 15000,
            "chunks": [],
            "created_at": datetime.utcnow(),
            "is_test": True,
            "test_run_id": pipeline_config.test_run_id
        }

        result = collection.insert_one(test_doc)
        assert result.inserted_id == doc_id

        # Verify storage
        stored_doc = assert_document_stored(
            collection,
            doc_id,
            expected_fields=["filename", "file_type", "content", "metadata", "file_hash"]
        )

        assert stored_doc["filename"] == "test_document.pdf"
        assert stored_doc["file_type"] == "pdf"
        assert stored_doc["metadata"]["page_count"] == 5

        # Cleanup
        collection.delete_one({"_id": doc_id})

    @pytest.mark.requires_mongodb
    def test_store_document_with_chunks(self, mongodb_database, pipeline_config):
        """Test storing a document with associated chunks."""
        docs_collection = mongodb_database["documents"]
        chunks_collection = mongodb_database["document_chunks"]

        doc_id = f"test_{generate_test_id('doc_chunks')}"
        chunk_ids = []

        # Create chunks
        for i in range(3):
            chunk_id = f"test_{generate_test_id(f'chunk_{i}')}"
            chunk_ids.append(chunk_id)

            chunk_doc = {
                "_id": chunk_id,
                "document_id": doc_id,
                "chunk_index": i,
                "content": f"This is chunk {i} of the document content.",
                "embedding": [0.1 * i] * 384,  # Mock embedding vector
                "metadata": {
                    "page_number": i + 1,
                    "start_char": i * 100,
                    "end_char": (i + 1) * 100
                },
                "created_at": datetime.utcnow(),
                "is_test": True,
                "test_run_id": pipeline_config.test_run_id
            }
            chunks_collection.insert_one(chunk_doc)

        # Create parent document
        doc = {
            "_id": doc_id,
            "filename": "chunked_document.pdf",
            "file_type": "pdf",
            "content": "Full document content split into chunks.",
            "metadata": {
                "title": "Chunked Doc",
                "page_count": 3,
                "word_count": 100,
                "file_hash": "def456hash"
            },
            "file_hash": "def456hash",
            "file_size_bytes": 5000,
            "chunks": chunk_ids,
            "chunk_count": len(chunk_ids),
            "created_at": datetime.utcnow(),
            "is_test": True,
            "test_run_id": pipeline_config.test_run_id
        }
        docs_collection.insert_one(doc)

        # Verify document
        stored_doc = assert_document_stored(docs_collection, doc_id)
        assert len(stored_doc["chunks"]) == 3
        assert stored_doc["chunk_count"] == 3

        # Verify chunks
        for chunk_id in chunk_ids:
            chunk = assert_document_stored(chunks_collection, chunk_id)
            assert chunk["document_id"] == doc_id
            assert len(chunk["embedding"]) == 384

        # Cleanup
        docs_collection.delete_one({"_id": doc_id})
        chunks_collection.delete_many({"_id": {"$in": chunk_ids}})

    @pytest.mark.requires_mongodb
    def test_store_docx_document(self, mongodb_database, pipeline_config):
        """Test storing a Word document."""
        collection = mongodb_database["documents"]

        doc_id = f"test_{generate_test_id('docx')}"
        test_doc = {
            "_id": doc_id,
            "filename": "report.docx",
            "file_type": "docx",
            "content": "This is a Word document content with tables.\n\nTable 1:\nHeader1 | Header2\nValue1 | Value2",
            "metadata": {
                "title": "Monthly Report",
                "author": "Business Team",
                "created_date": "2025-01-10",
                "page_count": 10,
                "word_count": 500,
                "has_tables": True,
                "has_images": False,
                "table_count": 2,
                "language": "en",
                "file_hash": "ghi789hash"
            },
            "file_hash": "ghi789hash",
            "file_size_bytes": 25000,
            "chunks": [],
            "created_at": datetime.utcnow(),
            "is_test": True,
            "test_run_id": pipeline_config.test_run_id
        }

        collection.insert_one(test_doc)

        stored_doc = assert_document_stored(collection, doc_id)
        assert stored_doc["file_type"] == "docx"
        assert stored_doc["metadata"]["has_tables"] is True
        assert stored_doc["metadata"]["table_count"] == 2

        # Cleanup
        collection.delete_one({"_id": doc_id})

    @pytest.mark.requires_mongodb
    def test_store_xlsx_document(self, mongodb_database, pipeline_config):
        """Test storing an Excel spreadsheet."""
        collection = mongodb_database["documents"]

        doc_id = f"test_{generate_test_id('xlsx')}"
        test_doc = {
            "_id": doc_id,
            "filename": "data_analysis.xlsx",
            "file_type": "xlsx",
            "content": "=== Sheet: Sales ===\nMonth | Revenue | Expenses\nJan | 10000 | 5000\nFeb | 12000 | 5500",
            "metadata": {
                "title": "Sales Data",
                "author": "",
                "created_date": "",
                "page_count": 0,
                "word_count": 50,
                "has_tables": True,
                "has_images": False,
                "table_count": 2,
                "sheet_names": ["Sales", "Expenses"],
                "language": "en",
                "file_hash": "jkl012hash"
            },
            "file_hash": "jkl012hash",
            "file_size_bytes": 8000,
            "chunks": [],
            "created_at": datetime.utcnow(),
            "is_test": True,
            "test_run_id": pipeline_config.test_run_id
        }

        collection.insert_one(test_doc)

        stored_doc = assert_document_stored(collection, doc_id)
        assert stored_doc["file_type"] == "xlsx"
        assert "Sales" in stored_doc["metadata"]["sheet_names"]
        assert stored_doc["metadata"]["has_tables"] is True

        # Cleanup
        collection.delete_one({"_id": doc_id})

    @pytest.mark.requires_mongodb
    def test_store_text_document(self, mongodb_database, pipeline_config):
        """Test storing a plain text document."""
        collection = mongodb_database["documents"]

        doc_id = f"test_{generate_test_id('txt')}"
        test_doc = {
            "_id": doc_id,
            "filename": "notes.txt",
            "file_type": "text",
            "content": "These are plain text notes.\nLine 2 of notes.\nLine 3 of notes.",
            "metadata": {
                "title": "",
                "author": "",
                "created_date": "",
                "page_count": 0,
                "word_count": 15,
                "has_tables": False,
                "has_images": False,
                "table_count": 0,
                "language": "en",
                "file_hash": "mno345hash"
            },
            "file_hash": "mno345hash",
            "file_size_bytes": 500,
            "chunks": [],
            "created_at": datetime.utcnow(),
            "is_test": True,
            "test_run_id": pipeline_config.test_run_id
        }

        collection.insert_one(test_doc)

        stored_doc = assert_document_stored(collection, doc_id)
        assert stored_doc["file_type"] == "text"
        assert "notes" in stored_doc["content"]

        # Cleanup
        collection.delete_one({"_id": doc_id})

    @pytest.mark.requires_mongodb
    def test_prevent_duplicate_documents(self, mongodb_database, pipeline_config):
        """Test preventing duplicate document storage using file_hash."""
        collection = mongodb_database["documents"]

        file_hash = "unique_hash_" + generate_test_id("hash")

        # Insert first document
        doc_id_1 = f"test_{generate_test_id('dup1')}"
        doc1 = {
            "_id": doc_id_1,
            "filename": "duplicate.pdf",
            "file_type": "pdf",
            "content": "Duplicate content",
            "metadata": {"file_hash": file_hash},
            "file_hash": file_hash,
            "file_size_bytes": 1000,
            "chunks": [],
            "created_at": datetime.utcnow(),
            "is_test": True,
            "test_run_id": pipeline_config.test_run_id
        }
        collection.insert_one(doc1)

        # Check for duplicate before inserting second
        existing = collection.find_one({
            "file_hash": file_hash,
            "is_test": True
        })

        assert existing is not None
        assert existing["_id"] == doc_id_1

        # Don't insert duplicate - just verify we can detect it
        duplicate_count = collection.count_documents({
            "file_hash": file_hash,
            "is_test": True
        })
        assert duplicate_count == 1

        # Cleanup
        collection.delete_one({"_id": doc_id_1})

    @pytest.mark.requires_mongodb
    def test_update_document_metadata(self, mongodb_database, pipeline_config):
        """Test updating document metadata."""
        collection = mongodb_database["documents"]

        doc_id = f"test_{generate_test_id('update')}"

        # Initial document
        initial_doc = {
            "_id": doc_id,
            "filename": "update_test.pdf",
            "file_type": "pdf",
            "content": "Original content",
            "metadata": {
                "title": "Original Title",
                "page_count": 5,
                "file_hash": "update_hash"
            },
            "file_hash": "update_hash",
            "file_size_bytes": 2000,
            "chunks": [],
            "created_at": datetime.utcnow(),
            "is_test": True,
            "test_run_id": pipeline_config.test_run_id
        }
        collection.insert_one(initial_doc)

        # Update metadata
        collection.update_one(
            {"_id": doc_id},
            {
                "$set": {
                    "metadata.title": "Updated Title",
                    "metadata.author": "New Author",
                    "updated_at": datetime.utcnow()
                }
            }
        )

        # Verify update
        updated_doc = assert_document_stored(collection, doc_id)
        assert updated_doc["metadata"]["title"] == "Updated Title"
        assert updated_doc["metadata"]["author"] == "New Author"
        assert "updated_at" in updated_doc

        # Cleanup
        collection.delete_one({"_id": doc_id})

    @pytest.mark.requires_mongodb
    def test_batch_document_storage(self, mongodb_database, pipeline_config):
        """Test storing multiple documents in batch."""
        collection = mongodb_database["documents"]

        docs = []
        doc_ids = []

        for i in range(5):
            doc_id = f"test_{generate_test_id(f'batch_{i}')}"
            doc_ids.append(doc_id)

            docs.append({
                "_id": doc_id,
                "filename": f"batch_doc_{i}.pdf",
                "file_type": "pdf",
                "content": f"Batch document {i} content",
                "metadata": {
                    "title": f"Doc {i}",
                    "page_count": i + 1,
                    "file_hash": f"hash_{i}"
                },
                "file_hash": f"hash_{i}",
                "file_size_bytes": 1000 * (i + 1),
                "chunks": [],
                "created_at": datetime.utcnow(),
                "is_test": True,
                "test_run_id": pipeline_config.test_run_id
            })

        # Batch insert
        result = collection.insert_many(docs)
        assert len(result.inserted_ids) == 5

        # Verify all
        for doc_id in doc_ids:
            stored = assert_document_stored(collection, doc_id)
            assert "Batch document" in stored["content"]

        # Cleanup
        collection.delete_many({"_id": {"$in": doc_ids}})

    @pytest.mark.requires_mongodb
    def test_document_schema_validation(self, mongodb_database, pipeline_config):
        """Test that documents match expected schema."""
        collection = mongodb_database["documents"]

        doc_id = f"test_{generate_test_id('schema')}"
        test_doc = {
            "_id": doc_id,
            "filename": "schema_test.pdf",
            "file_type": "pdf",
            "content": "Content",
            "metadata": {
                "title": "Schema Test",
                "page_count": 1,
                "file_hash": "schema_hash"
            },
            "file_hash": "schema_hash",
            "file_size_bytes": 1000,
            "chunks": [],
            "created_at": datetime.utcnow(),
            "is_test": True,
            "test_run_id": pipeline_config.test_run_id
        }

        collection.insert_one(test_doc)
        stored_doc = assert_document_stored(collection, doc_id)

        # Validate schema
        expected_schema = {
            "filename": str,
            "file_type": str,
            "content": str,
            "metadata": dict,
            "file_hash": str,
            "file_size_bytes": int,
            "chunks": list,
            "created_at": datetime,
        }

        assert_mongodb_document(stored_doc, expected_schema)

        # Cleanup
        collection.delete_one({"_id": doc_id})

    @pytest.mark.requires_mongodb
    def test_chunk_schema_validation(self, mongodb_database, pipeline_config):
        """Test that chunks match expected schema."""
        collection = mongodb_database["document_chunks"]

        chunk_id = f"test_{generate_test_id('chunk_schema')}"
        test_chunk = {
            "_id": chunk_id,
            "document_id": "parent_doc_id",
            "chunk_index": 0,
            "content": "Chunk content",
            "embedding": [0.1] * 384,
            "metadata": {
                "page_number": 1,
                "start_char": 0,
                "end_char": 100
            },
            "created_at": datetime.utcnow(),
            "is_test": True,
            "test_run_id": pipeline_config.test_run_id
        }

        collection.insert_one(test_chunk)
        stored_chunk = assert_document_stored(collection, chunk_id)

        # Validate schema
        expected_schema = {
            "document_id": str,
            "chunk_index": int,
            "content": str,
            "embedding": list,
            "metadata": dict,
            "created_at": datetime,
        }

        assert_mongodb_document(stored_chunk, expected_schema)

        # Cleanup
        collection.delete_one({"_id": chunk_id})
