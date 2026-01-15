"""
MongoDB Test Fixtures
=====================

Fixtures for MongoDB testing with proper isolation and cleanup.
All test data is prefixed with 'test_' for easy identification and cleanup.
"""

import uuid
from datetime import datetime
from typing import List, Dict, Any, Optional
from pymongo import MongoClient
from pymongo.database import Database
from pymongo.collection import Collection


def get_test_mongodb_client(
    uri: str = "mongodb://EWRSPT-AI:27018",
    timeout_ms: int = 30000
) -> MongoClient:
    """
    Get a MongoDB client for testing.

    Args:
        uri: MongoDB connection URI
        timeout_ms: Connection timeout in milliseconds

    Returns:
        MongoClient instance
    """
    return MongoClient(uri, serverSelectionTimeoutMS=timeout_ms)


def create_test_collection(
    db: Database,
    base_name: str,
    indexes: Optional[List[Dict]] = None
) -> Collection:
    """
    Create a test collection with unique name.

    Args:
        db: MongoDB database
        base_name: Base name for the collection
        indexes: Optional list of index specifications

    Returns:
        Collection instance
    """
    collection_name = f"test_{base_name}_{uuid.uuid4().hex[:8]}"
    collection = db[collection_name]

    if indexes:
        for index_spec in indexes:
            collection.create_index(**index_spec)

    return collection


def insert_test_documents(
    collection: Collection,
    documents: List[Dict[str, Any]],
    test_run_id: Optional[str] = None
) -> List[str]:
    """
    Insert test documents with proper markers.

    Args:
        collection: MongoDB collection
        documents: List of documents to insert
        test_run_id: Optional test run identifier

    Returns:
        List of inserted document IDs
    """
    for doc in documents:
        # Ensure test markers
        if "_id" not in doc:
            doc["_id"] = f"test_{uuid.uuid4().hex}"
        elif not str(doc["_id"]).startswith("test_"):
            doc["_id"] = f"test_{doc['_id']}"

        doc["is_test"] = True
        doc["test_marker"] = True
        doc["test_created_at"] = datetime.utcnow()

        if test_run_id:
            doc["test_run_id"] = test_run_id

    result = collection.insert_many(documents)
    return [str(id) for id in result.inserted_ids]


def cleanup_test_documents(
    db: Database,
    test_run_id: Optional[str] = None,
    collection_name: Optional[str] = None
) -> Dict[str, int]:
    """
    Clean up test documents from MongoDB.

    Args:
        db: MongoDB database
        test_run_id: Optional test run ID to filter cleanup
        collection_name: Optional specific collection to clean

    Returns:
        Dict mapping collection names to deleted counts
    """
    results = {}

    collections = [collection_name] if collection_name else db.list_collection_names()

    for col_name in collections:
        collection = db[col_name]

        # Build query for test documents
        query = {
            "$or": [
                {"is_test": True},
                {"test_marker": True},
                {"_id": {"$regex": "^test_"}},
            ]
        }

        if test_run_id:
            query["$or"].append({"test_run_id": test_run_id})

        result = collection.delete_many(query)
        if result.deleted_count > 0:
            results[col_name] = result.deleted_count

    return results


def create_mock_document(
    doc_type: str = "generic",
    pipeline: str = "test",
    content: str = "Test content",
    **extra_fields
) -> Dict[str, Any]:
    """
    Create a mock document for testing.

    Args:
        doc_type: Type of document
        pipeline: Pipeline name
        content: Document content
        **extra_fields: Additional fields to include

    Returns:
        Mock document dictionary
    """
    doc = {
        "_id": f"test_{uuid.uuid4().hex}",
        "type": doc_type,
        "pipeline": pipeline,
        "content": content,
        "is_test": True,
        "test_marker": True,
        "created_at": datetime.utcnow(),
    }
    doc.update(extra_fields)
    return doc


# Collection-specific mock data generators

def create_mock_sql_query(
    question: str = "How many tickets are there?",
    sql: str = "SELECT COUNT(*) FROM CentralTickets",
    database: str = "EWRCentral",
    success: bool = True,
) -> Dict[str, Any]:
    """Create mock SQL query document for agent_learning collection."""
    return create_mock_document(
        doc_type="sql_query",
        pipeline="sql",
        content=sql,
        question=question,
        question_normalized=question.lower().strip(),
        sql=sql,
        database=database,
        success=success,
    )


def create_mock_audio_analysis(
    transcription: str = "Test transcription",
    emotions: List[str] = None,
    duration: float = 60.0,
) -> Dict[str, Any]:
    """Create mock audio analysis document."""
    return create_mock_document(
        doc_type="audio_analysis",
        pipeline="audio",
        content=transcription,
        transcription=transcription,
        emotions=emotions or ["NEUTRAL"],
        duration_seconds=duration,
        audio_events=[],
    )


def create_mock_code_method(
    method_name: str = "TestMethod",
    class_name: str = "TestClass",
    project: str = "TestProject",
    code: str = "public void TestMethod() { }",
    **extra_fields,
) -> Dict[str, Any]:
    """Create mock code method document."""
    return create_mock_document(
        doc_type="code_method",
        pipeline="git",
        content=code,
        method_name=method_name,
        class_name=class_name,
        project=project,
        file_path=f"/src/{class_name}.cs",
        language="csharp",
        **extra_fields,
    )


def create_mock_document_chunk(
    title: str = "Test Document",
    content: str = "Test document content",
    doc_type: str = "pdf",
    chunk_index: int = 0,
) -> Dict[str, Any]:
    """Create mock document chunk."""
    return create_mock_document(
        doc_type="document_chunk",
        pipeline="document_agent",
        content=content,
        title=title,
        source_type=doc_type,
        chunk_index=chunk_index,
        total_chunks=1,
    )
