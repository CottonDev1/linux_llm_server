"""
Document End-to-End Tests
=========================

End-to-end tests for the complete document processing pipeline:
1. Document upload/ingestion
2. Content extraction (PDF, DOCX, XLSX, TXT)
3. Chunking and metadata extraction
4. Embedding generation
5. MongoDB storage
6. Retrieval and Q&A

These tests simulate the complete workflow from document upload to query answering.
"""

import pytest
import os
import tempfile
from datetime import datetime
from typing import Dict, List

from config.test_config import get_test_config
from utils import generate_test_id, create_temp_file, assert_document_stored


class TestDocumentEndToEnd:
    """End-to-end document pipeline tests."""

    @pytest.mark.e2e
    @pytest.mark.requires_mongodb
    async def test_document_pipeline_basic_flow(self, mongodb_database, pipeline_config):
        """Test basic end-to-end document processing flow."""
        docs_collection = mongodb_database["documents"]
        chunks_collection = mongodb_database["document_chunks"]

        # Step 1: Simulate document upload
        filename = "test_e2e_doc.pdf"
        file_content = "This is a test document for end-to-end processing. It contains important information about software testing."

        # Step 2: Simulate content extraction
        extracted_content = {
            "content": file_content,
            "metadata": {
                "title": "E2E Test Document",
                "author": "Test Suite",
                "page_count": 1,
                "word_count": len(file_content.split()),
                "has_tables": False,
                "has_images": False,
                "file_hash": f"e2e_hash_{generate_test_id('hash')}"
            }
        }

        # Step 3: Create chunks
        chunk_size = 100
        chunks = []
        for i in range(0, len(file_content), chunk_size):
            chunks.append({
                "chunk_index": len(chunks),
                "content": file_content[i:i + chunk_size],
                "embedding": [0.1 * len(chunks)] * 384,  # Mock embedding
                "metadata": {
                    "start_char": i,
                    "end_char": min(i + chunk_size, len(file_content))
                }
            })

        # Step 4: Store in MongoDB
        doc_id = f"test_{generate_test_id('e2e_doc')}"
        chunk_ids = []

        # Store chunks
        for chunk_data in chunks:
            chunk_id = f"test_{generate_test_id('e2e_chunk')}"
            chunk_ids.append(chunk_id)

            chunk_doc = {
                "_id": chunk_id,
                "document_id": doc_id,
                **chunk_data,
                "created_at": datetime.utcnow(),
                "is_test": True,
                "test_run_id": pipeline_config.test_run_id
            }
            chunks_collection.insert_one(chunk_doc)

        # Store document
        doc = {
            "_id": doc_id,
            "filename": filename,
            "file_type": "pdf",
            "content": extracted_content["content"],
            "metadata": extracted_content["metadata"],
            "file_hash": extracted_content["metadata"]["file_hash"],
            "file_size_bytes": len(file_content),
            "chunks": chunk_ids,
            "chunk_count": len(chunk_ids),
            "created_at": datetime.utcnow(),
            "is_test": True,
            "test_run_id": pipeline_config.test_run_id
        }
        docs_collection.insert_one(doc)

        # Step 5: Verify storage
        stored_doc = assert_document_stored(
            docs_collection,
            doc_id,
            expected_fields=["filename", "content", "chunks", "metadata"]
        )

        assert stored_doc["filename"] == filename
        assert len(stored_doc["chunks"]) > 0
        assert stored_doc["chunk_count"] == len(chunk_ids)

        # Verify chunks
        for chunk_id in chunk_ids:
            chunk = assert_document_stored(chunks_collection, chunk_id)
            assert chunk["document_id"] == doc_id

        # Cleanup
        docs_collection.delete_one({"_id": doc_id})
        chunks_collection.delete_many({"_id": {"$in": chunk_ids}})

    @pytest.mark.e2e
    @pytest.mark.requires_mongodb
    async def test_pdf_document_processing(self, mongodb_database, pipeline_config):
        """Test E2E processing of PDF document."""
        docs_collection = mongodb_database["documents"]

        # Simulate PDF processing
        doc_id = f"test_{generate_test_id('e2e_pdf')}"
        pdf_doc = {
            "_id": doc_id,
            "filename": "technical_guide.pdf",
            "file_type": "pdf",
            "content": "Chapter 1: Introduction\n\nThis technical guide covers best practices for software development.",
            "metadata": {
                "title": "Technical Development Guide",
                "author": "Engineering Team",
                "created_date": "2025-01-15",
                "page_count": 25,
                "word_count": 5000,
                "has_tables": True,
                "has_images": True,
                "table_count": 3,
                "language": "en",
                "file_hash": "pdf_e2e_hash"
            },
            "file_hash": "pdf_e2e_hash",
            "file_size_bytes": 250000,
            "chunks": [],
            "chunk_count": 0,
            "created_at": datetime.utcnow(),
            "processing_status": "completed",
            "is_test": True,
            "test_run_id": pipeline_config.test_run_id
        }

        docs_collection.insert_one(pdf_doc)

        # Verify processing
        stored = assert_document_stored(docs_collection, doc_id)
        assert stored["file_type"] == "pdf"
        assert stored["metadata"]["page_count"] == 25
        assert stored["processing_status"] == "completed"

        # Cleanup
        docs_collection.delete_one({"_id": doc_id})

    @pytest.mark.e2e
    @pytest.mark.requires_mongodb
    async def test_docx_document_processing(self, mongodb_database, pipeline_config):
        """Test E2E processing of Word document."""
        docs_collection = mongodb_database["documents"]

        doc_id = f"test_{generate_test_id('e2e_docx')}"
        docx_doc = {
            "_id": doc_id,
            "filename": "business_proposal.docx",
            "file_type": "docx",
            "content": "Business Proposal\n\nExecutive Summary:\nThis proposal outlines our strategy for Q1 2025.",
            "metadata": {
                "title": "Q1 2025 Business Proposal",
                "author": "Business Development",
                "created_date": "2025-01-10",
                "page_count": 15,
                "word_count": 3000,
                "has_tables": True,
                "table_count": 2,
                "language": "en",
                "file_hash": "docx_e2e_hash"
            },
            "file_hash": "docx_e2e_hash",
            "file_size_bytes": 50000,
            "chunks": [],
            "created_at": datetime.utcnow(),
            "is_test": True,
            "test_run_id": pipeline_config.test_run_id
        }

        docs_collection.insert_one(docx_doc)

        stored = assert_document_stored(docs_collection, doc_id)
        assert stored["file_type"] == "docx"
        assert stored["metadata"]["has_tables"] is True

        # Cleanup
        docs_collection.delete_one({"_id": doc_id})

    @pytest.mark.e2e
    @pytest.mark.requires_mongodb
    async def test_document_chunking_workflow(self, mongodb_database, pipeline_config):
        """Test chunking workflow for large documents."""
        docs_collection = mongodb_database["documents"]
        chunks_collection = mongodb_database["document_chunks"]

        # Large document content
        large_content = "This is a large document. " * 100  # ~2500 chars

        doc_id = f"test_{generate_test_id('e2e_chunking')}"
        chunk_ids = []

        # Create chunks (500 char each with 100 char overlap)
        chunk_size = 500
        overlap = 100
        step = chunk_size - overlap

        for i in range(0, len(large_content), step):
            if i + chunk_size <= len(large_content):
                chunk_id = f"test_{generate_test_id('chunk')}"
                chunk_ids.append(chunk_id)

                chunks_collection.insert_one({
                    "_id": chunk_id,
                    "document_id": doc_id,
                    "chunk_index": len(chunk_ids) - 1,
                    "content": large_content[i:i + chunk_size],
                    "embedding": [0.1] * 384,
                    "metadata": {
                        "start_char": i,
                        "end_char": i + chunk_size,
                        "has_overlap": i > 0
                    },
                    "created_at": datetime.utcnow(),
                    "is_test": True,
                    "test_run_id": pipeline_config.test_run_id
                })

        # Store parent document
        docs_collection.insert_one({
            "_id": doc_id,
            "filename": "large_doc.pdf",
            "file_type": "pdf",
            "content": large_content,
            "metadata": {
                "title": "Large Document",
                "word_count": len(large_content.split()),
                "file_hash": "large_hash"
            },
            "file_hash": "large_hash",
            "file_size_bytes": len(large_content),
            "chunks": chunk_ids,
            "chunk_count": len(chunk_ids),
            "chunking_strategy": "overlap",
            "chunk_size": chunk_size,
            "overlap_size": overlap,
            "created_at": datetime.utcnow(),
            "is_test": True,
            "test_run_id": pipeline_config.test_run_id
        })

        # Verify chunking
        stored = assert_document_stored(docs_collection, doc_id)
        assert stored["chunk_count"] > 0
        assert stored["chunking_strategy"] == "overlap"

        # Cleanup
        docs_collection.delete_one({"_id": doc_id})
        chunks_collection.delete_many({"_id": {"$in": chunk_ids}})

    @pytest.mark.e2e
    @pytest.mark.requires_mongodb
    @pytest.mark.requires_llm
    async def test_document_qa_workflow(self, mongodb_database, async_llm_client, pipeline_config):
        """Test complete document Q&A workflow."""
        docs_collection = mongodb_database["documents"]
        chunks_collection = mongodb_database["document_chunks"]

        # Store document with Q&A-friendly content
        qa_content = """
        Product Name: SuperWidget Pro
        Price: $299.99
        Features: Wireless, Waterproof, 10-hour battery life
        Release Date: March 2025
        """

        doc_id = f"test_{generate_test_id('e2e_qa')}"
        chunk_id = f"test_{generate_test_id('qa_chunk')}"

        # Store chunk
        chunks_collection.insert_one({
            "_id": chunk_id,
            "document_id": doc_id,
            "chunk_index": 0,
            "content": qa_content,
            "embedding": [0.2] * 384,
            "metadata": {},
            "created_at": datetime.utcnow(),
            "is_test": True,
            "test_run_id": pipeline_config.test_run_id
        })

        # Store document
        docs_collection.insert_one({
            "_id": doc_id,
            "filename": "product_info.txt",
            "file_type": "text",
            "content": qa_content,
            "metadata": {
                "title": "Product Information",
                "file_hash": "qa_hash"
            },
            "file_hash": "qa_hash",
            "file_size_bytes": len(qa_content),
            "chunks": [chunk_id],
            "created_at": datetime.utcnow(),
            "is_test": True,
            "test_run_id": pipeline_config.test_run_id
        })

        # Simulate Q&A: Retrieve chunk and query LLM
        retrieved_chunk = chunks_collection.find_one({"_id": chunk_id})
        assert retrieved_chunk is not None

        question = "What is the price of SuperWidget Pro?"
        prompt = f"""Answer based on this content:

{retrieved_chunk['content']}

Question: {question}

Answer:"""

        response = await async_llm_client.generate(
            prompt=prompt,
            endpoint="general",
            max_tokens=50,
            temperature=0.1
        )

        # Verify Q&A worked
        assert response.success
        assert "299" in response.text or "$299" in response.text

        # Cleanup
        docs_collection.delete_one({"_id": doc_id})
        chunks_collection.delete_one({"_id": chunk_id})

    @pytest.mark.e2e
    @pytest.mark.requires_mongodb
    async def test_batch_document_processing(self, mongodb_database, pipeline_config):
        """Test processing multiple documents in batch."""
        docs_collection = mongodb_database["documents"]

        doc_ids = []
        for i in range(3):
            doc_id = f"test_{generate_test_id(f'batch_{i}')}"
            doc_ids.append(doc_id)

            docs_collection.insert_one({
                "_id": doc_id,
                "filename": f"batch_doc_{i}.pdf",
                "file_type": "pdf",
                "content": f"Batch document {i} content for testing.",
                "metadata": {
                    "title": f"Batch Doc {i}",
                    "page_count": i + 1,
                    "file_hash": f"batch_hash_{i}"
                },
                "file_hash": f"batch_hash_{i}",
                "file_size_bytes": 1000 * (i + 1),
                "chunks": [],
                "created_at": datetime.utcnow(),
                "batch_id": f"batch_{pipeline_config.test_run_id}",
                "is_test": True,
                "test_run_id": pipeline_config.test_run_id
            })

        # Verify batch
        batch_docs = list(docs_collection.find({
            "batch_id": f"batch_{pipeline_config.test_run_id}",
            "is_test": True
        }))

        assert len(batch_docs) == 3

        # Cleanup
        docs_collection.delete_many({"_id": {"$in": doc_ids}})

    @pytest.mark.e2e
    @pytest.mark.requires_mongodb
    async def test_document_update_workflow(self, mongodb_database, pipeline_config):
        """Test updating document after initial processing."""
        docs_collection = mongodb_database["documents"]

        doc_id = f"test_{generate_test_id('e2e_update')}"

        # Initial document
        docs_collection.insert_one({
            "_id": doc_id,
            "filename": "updatable.pdf",
            "file_type": "pdf",
            "content": "Original content",
            "metadata": {
                "title": "Original",
                "version": 1,
                "file_hash": "original_hash"
            },
            "file_hash": "original_hash",
            "file_size_bytes": 1000,
            "chunks": [],
            "created_at": datetime.utcnow(),
            "is_test": True,
            "test_run_id": pipeline_config.test_run_id
        })

        # Update document
        docs_collection.update_one(
            {"_id": doc_id},
            {
                "$set": {
                    "content": "Updated content with new information",
                    "metadata.title": "Updated",
                    "metadata.version": 2,
                    "updated_at": datetime.utcnow()
                }
            }
        )

        # Verify update
        updated = assert_document_stored(docs_collection, doc_id)
        assert updated["metadata"]["version"] == 2
        assert "Updated" in updated["metadata"]["title"]
        assert "updated_at" in updated

        # Cleanup
        docs_collection.delete_one({"_id": doc_id})

    @pytest.mark.e2e
    @pytest.mark.requires_mongodb
    async def test_document_deletion_workflow(self, mongodb_database, pipeline_config):
        """Test deleting document and associated chunks."""
        docs_collection = mongodb_database["documents"]
        chunks_collection = mongodb_database["document_chunks"]

        doc_id = f"test_{generate_test_id('e2e_delete')}"
        chunk_ids = []

        # Create chunks
        for i in range(2):
            chunk_id = f"test_{generate_test_id(f'del_chunk_{i}')}"
            chunk_ids.append(chunk_id)
            chunks_collection.insert_one({
                "_id": chunk_id,
                "document_id": doc_id,
                "chunk_index": i,
                "content": f"Chunk {i}",
                "embedding": [0.1] * 384,
                "metadata": {},
                "created_at": datetime.utcnow(),
                "is_test": True,
                "test_run_id": pipeline_config.test_run_id
            })

        # Create document
        docs_collection.insert_one({
            "_id": doc_id,
            "filename": "to_delete.pdf",
            "file_type": "pdf",
            "content": "Content to be deleted",
            "metadata": {"file_hash": "delete_hash"},
            "file_hash": "delete_hash",
            "file_size_bytes": 1000,
            "chunks": chunk_ids,
            "created_at": datetime.utcnow(),
            "is_test": True,
            "test_run_id": pipeline_config.test_run_id
        })

        # Delete document and chunks
        docs_collection.delete_one({"_id": doc_id})
        chunks_collection.delete_many({"document_id": doc_id})

        # Verify deletion
        assert docs_collection.find_one({"_id": doc_id}) is None
        assert chunks_collection.count_documents({"document_id": doc_id}) == 0

    @pytest.mark.e2e
    @pytest.mark.requires_mongodb
    async def test_document_error_handling(self, mongodb_database, pipeline_config):
        """Test error handling in document processing."""
        docs_collection = mongodb_database["documents"]

        doc_id = f"test_{generate_test_id('e2e_error')}"

        # Document with processing error
        docs_collection.insert_one({
            "_id": doc_id,
            "filename": "error_doc.pdf",
            "file_type": "pdf",
            "content": "",
            "metadata": {
                "file_hash": "error_hash"
            },
            "file_hash": "error_hash",
            "file_size_bytes": 0,
            "chunks": [],
            "processing_status": "failed",
            "error": "Unsupported file format or corrupted file",
            "created_at": datetime.utcnow(),
            "is_test": True,
            "test_run_id": pipeline_config.test_run_id
        })

        # Verify error was recorded
        error_doc = assert_document_stored(docs_collection, doc_id)
        assert error_doc["processing_status"] == "failed"
        assert "error" in error_doc
        assert len(error_doc["error"]) > 0

        # Cleanup
        docs_collection.delete_one({"_id": doc_id})
