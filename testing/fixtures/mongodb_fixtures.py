"""
MongoDB Test Fixtures
=====================

Fixtures for MongoDB testing using REAL data from the database.
Tests should validate against actual production data to ensure accuracy.
"""

import os
import random
from typing import List, Dict, Any, Optional
from pymongo import MongoClient
from pymongo.database import Database
from pymongo.collection import Collection


# Default configuration
DEFAULT_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017/?directConnection=true")
DEFAULT_DATABASE = os.getenv("MONGODB_DATABASE", "rag_server")


def get_test_mongodb_client(
    uri: str = DEFAULT_URI,
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


def get_test_database(
    uri: str = DEFAULT_URI,
    database: str = DEFAULT_DATABASE
) -> Database:
    """
    Get a MongoDB database for testing.

    Args:
        uri: MongoDB connection URI
        database: Database name

    Returns:
        Database instance
    """
    client = get_test_mongodb_client(uri)
    return client[database]


def get_collection_count(db: Database, collection_name: str) -> int:
    """Get document count for a collection."""
    return db[collection_name].count_documents({})


def verify_database_has_data(db: Database) -> Dict[str, int]:
    """
    Verify database has real data for testing.

    Returns:
        Dict mapping collection names to document counts

    Raises:
        ValueError: If database is empty or missing required collections
    """
    required_collections = ["sql_examples", "code_methods", "documents"]
    counts = {}

    for col_name in db.list_collection_names():
        count = db[col_name].count_documents({})
        if count > 0:
            counts[col_name] = count

    missing = [col for col in required_collections if col not in counts]
    if missing:
        raise ValueError(
            f"Database missing required collections: {missing}. "
            "Run scripts/mongodb_restore.sh to restore data."
        )

    return counts


# =============================================================================
# Real Data Retrieval Functions
# =============================================================================

def get_real_sql_example(
    db: Database,
    database_filter: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    """
    Get a real SQL example from the database.

    Args:
        db: MongoDB database
        database_filter: Optional database name to filter by

    Returns:
        Real SQL example document or None
    """
    collection = db["sql_examples"]
    query = {}
    if database_filter:
        query["database"] = database_filter

    # Get a random sample
    pipeline = [{"$match": query}, {"$sample": {"size": 1}}]
    results = list(collection.aggregate(pipeline))
    return results[0] if results else None


def get_real_sql_examples(
    db: Database,
    limit: int = 10,
    database_filter: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Get multiple real SQL examples from the database.

    Args:
        db: MongoDB database
        limit: Maximum number to return
        database_filter: Optional database name to filter by

    Returns:
        List of SQL example documents
    """
    collection = db["sql_examples"]
    query = {}
    if database_filter:
        query["database"] = database_filter

    pipeline = [{"$match": query}, {"$sample": {"size": limit}}]
    return list(collection.aggregate(pipeline))


def get_real_sql_rule(db: Database) -> Optional[Dict[str, Any]]:
    """Get a real SQL rule from the database."""
    collection = db["sql_rules"]
    pipeline = [{"$sample": {"size": 1}}]
    results = list(collection.aggregate(pipeline))
    return results[0] if results else None


def get_real_sql_rules(db: Database, limit: int = 10) -> List[Dict[str, Any]]:
    """Get multiple real SQL rules from the database."""
    collection = db["sql_rules"]
    pipeline = [{"$sample": {"size": limit}}]
    return list(collection.aggregate(pipeline))


def get_real_code_method(
    db: Database,
    project_filter: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    """
    Get a real code method from the database.

    Args:
        db: MongoDB database
        project_filter: Optional project name to filter by

    Returns:
        Real code method document or None
    """
    collection = db["code_methods"]
    query = {}
    if project_filter:
        query["project"] = project_filter

    pipeline = [{"$match": query}, {"$sample": {"size": 1}}]
    results = list(collection.aggregate(pipeline))
    return results[0] if results else None


def get_real_code_methods(
    db: Database,
    limit: int = 10,
    project_filter: Optional[str] = None
) -> List[Dict[str, Any]]:
    """Get multiple real code methods from the database."""
    collection = db["code_methods"]
    query = {}
    if project_filter:
        query["project"] = project_filter

    pipeline = [{"$match": query}, {"$sample": {"size": limit}}]
    return list(collection.aggregate(pipeline))


def get_real_code_class(db: Database) -> Optional[Dict[str, Any]]:
    """Get a real code class from the database."""
    collection = db["code_classes"]
    pipeline = [{"$sample": {"size": 1}}]
    results = list(collection.aggregate(pipeline))
    return results[0] if results else None


def get_real_audio_analysis(db: Database) -> Optional[Dict[str, Any]]:
    """
    Get a real audio analysis from the database.

    Returns:
        Real audio analysis document or None
    """
    collection = db["audio_analysis"]
    pipeline = [{"$sample": {"size": 1}}]
    results = list(collection.aggregate(pipeline))
    return results[0] if results else None


def get_real_audio_analyses(db: Database, limit: int = 10) -> List[Dict[str, Any]]:
    """Get multiple real audio analyses from the database."""
    collection = db["audio_analysis"]
    pipeline = [{"$sample": {"size": limit}}]
    return list(collection.aggregate(pipeline))


def get_real_document(db: Database) -> Optional[Dict[str, Any]]:
    """
    Get a real document from the database.

    Returns:
        Real document or None
    """
    collection = db["documents"]
    pipeline = [{"$sample": {"size": 1}}]
    results = list(collection.aggregate(pipeline))
    return results[0] if results else None


def get_real_documents(db: Database, limit: int = 10) -> List[Dict[str, Any]]:
    """Get multiple real documents from the database."""
    collection = db["documents"]
    pipeline = [{"$sample": {"size": limit}}]
    return list(collection.aggregate(pipeline))


def get_real_stored_procedure(db: Database) -> Optional[Dict[str, Any]]:
    """Get a real stored procedure from the database."""
    collection = db["sql_stored_procedures"]
    pipeline = [{"$sample": {"size": 1}}]
    results = list(collection.aggregate(pipeline))
    return results[0] if results else None


def get_real_stored_procedures(db: Database, limit: int = 10) -> List[Dict[str, Any]]:
    """Get multiple real stored procedures from the database."""
    collection = db["sql_stored_procedures"]
    pipeline = [{"$sample": {"size": limit}}]
    return list(collection.aggregate(pipeline))


# =============================================================================
# Test Question Generators (using real data patterns)
# =============================================================================

def get_sql_test_questions(db: Database) -> List[Dict[str, str]]:
    """
    Get real SQL questions for testing based on actual examples.

    Returns:
        List of {question, expected_tables, database} dicts
    """
    examples = get_real_sql_examples(db, limit=20)
    questions = []

    for ex in examples:
        if "sql" in ex:
            # Extract table names from SQL
            sql = ex.get("sql", "")
            tables = []
            if "FROM" in sql.upper():
                # Simple extraction - real tests should use proper parsing
                parts = sql.upper().split("FROM")
                if len(parts) > 1:
                    table_part = parts[1].split()[0] if parts[1].split() else ""
                    tables.append(table_part.strip("[]"))

            questions.append({
                "question": ex.get("question", f"Query for {tables}"),
                "expected_sql_pattern": sql[:100],
                "database": ex.get("database", "unknown"),
                "tables": tables,
            })

    return questions


def get_code_test_queries(db: Database) -> List[Dict[str, str]]:
    """
    Get real code search queries based on actual methods.

    Returns:
        List of {query, expected_method, expected_class} dicts
    """
    methods = get_real_code_methods(db, limit=20)
    queries = []

    for method in methods:
        method_name = method.get("method_name", "")
        class_name = method.get("class_name", "")
        project = method.get("project", "")

        if method_name:
            queries.append({
                "query": f"Find {method_name} method",
                "expected_method": method_name,
                "expected_class": class_name,
                "project": project,
            })

    return queries


def get_document_test_queries(db: Database) -> List[Dict[str, str]]:
    """
    Get real document search queries based on actual documents.

    Returns:
        List of {query, expected_title} dicts
    """
    docs = get_real_documents(db, limit=10)
    queries = []

    for doc in docs:
        title = doc.get("title", "")
        content = doc.get("content", "")[:100]

        if title:
            # Create a query based on content keywords
            words = content.split()[:5]
            query = " ".join(words) if words else title
            queries.append({
                "query": query,
                "expected_title": title,
                "doc_id": str(doc.get("_id")),
            })

    return queries
