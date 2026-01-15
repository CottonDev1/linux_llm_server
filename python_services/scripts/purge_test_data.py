#!/usr/bin/env python3
"""
Purge Test Data Script

Removes all test-prefixed data from MongoDB collections.
Run this before starting fresh test runs to ensure clean state.
"""

import os
import sys
from typing import List

# Add python_services to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def purge_test_data(mongodb_uri: str = None) -> dict:
    """
    Purge all test data from MongoDB collections.

    Args:
        mongodb_uri: MongoDB connection URI (reads from MONGODB_URI env var)

    Returns:
        Dict with collection names and deleted counts
    """
    from pymongo import MongoClient

    mongodb_uri = mongodb_uri or os.getenv("MONGODB_URI", "mongodb://localhost:27017")

    print(f"Connecting to MongoDB: {mongodb_uri}")
    client = MongoClient(mongodb_uri)
    db = client['llm_website']

    results = {}
    collections = db.list_collection_names()

    print(f"Found {len(collections)} collections")

    for collection_name in collections:
        collection = db[collection_name]

        # Delete documents with test markers
        result = collection.delete_many({
            "$or": [
                {"_id": {"$regex": "^test_"}},
                {"name": {"$regex": "^test_"}},
                {"is_test": True},
                {"test_run_id": {"$exists": True}},
                {"test_marker": {"$exists": True}}
            ]
        })

        if result.deleted_count > 0:
            results[collection_name] = result.deleted_count
            print(f"  Purged {result.deleted_count} test documents from {collection_name}")

    client.close()

    if not results:
        print("No test data found to purge")
    else:
        total = sum(results.values())
        print(f"\nTotal: Purged {total} test documents from {len(results)} collections")

    return results


def verify_clean_state(mongodb_uri: str = None) -> bool:
    """
    Verify that no test data remains in MongoDB.

    Args:
        mongodb_uri: MongoDB connection URI

    Returns:
        True if no test data found, False otherwise
    """
    from pymongo import MongoClient

    mongodb_uri = mongodb_uri or os.getenv("MONGODB_URI", "mongodb://localhost:27017")

    client = MongoClient(mongodb_uri)
    db = client['llm_website']

    test_data_found = False
    collections = db.list_collection_names()

    for collection_name in collections:
        collection = db[collection_name]

        count = collection.count_documents({
            "$or": [
                {"_id": {"$regex": "^test_"}},
                {"name": {"$regex": "^test_"}},
                {"is_test": True},
                {"test_run_id": {"$exists": True}},
                {"test_marker": {"$exists": True}}
            ]
        })

        if count > 0:
            print(f"Warning: Found {count} test documents in {collection_name}")
            test_data_found = True

    client.close()

    if not test_data_found:
        print("Clean state verified: No test data found")
        return True

    return False


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Purge test data from MongoDB")
    parser.add_argument(
        "--uri",
        default=None,
        help="MongoDB URI (reads from MONGODB_URI env var, defaults to localhost:27017)"
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Only verify clean state, don't purge"
    )

    args = parser.parse_args()

    if args.verify_only:
        success = verify_clean_state(args.uri)
        sys.exit(0 if success else 1)
    else:
        purge_test_data(args.uri)
        print("\nVerifying clean state...")
        verify_clean_state(args.uri)
