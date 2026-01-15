"""
Migration script: Move SQL rules from JSON file to MongoDB.

This script:
1. Reads rules from config/sql_rules.json
2. Transforms them into MongoDB documents
3. Inserts into the sql_rules collection
4. Creates indexes for efficient querying

Run with: python -m scripts.migrate_sql_rules_to_mongodb
"""

import asyncio
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from motor.motor_asyncio import AsyncIOMotorClient
from pymongo import IndexModel, ASCENDING, TEXT
from pymongo.errors import DuplicateKeyError

from config import MONGODB_URI, MONGODB_DATABASE, COLLECTION_SQL_RULES


async def migrate_rules():
    """Migrate SQL rules from JSON to MongoDB."""

    # Path to JSON file
    json_path = Path(__file__).parent.parent.parent / "config" / "sql_rules.json"

    if not json_path.exists():
        print(f"ERROR: JSON file not found: {json_path}")
        return False

    # Load JSON
    print(f"Loading rules from: {json_path}")
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Connect to MongoDB
    print(f"Connecting to MongoDB: {MONGODB_URI}")
    client = AsyncIOMotorClient(MONGODB_URI)
    db = client[MONGODB_DATABASE]
    collection = db[COLLECTION_SQL_RULES]

    # Test connection
    await client.admin.command('ping')
    print("Connected to MongoDB successfully")

    # Track statistics
    stats = {
        "global_rules": 0,
        "database_rules": 0,
        "total_inserted": 0,
        "duplicates_skipped": 0
    }

    now = datetime.now(timezone.utc)
    documents = []

    # Process global constraints
    print("\nProcessing global constraints...")
    for rule in data.get("global_constraints", []):
        doc = {
            "rule_id": rule.get("id"),
            "scope": "global",
            "type": rule.get("type", "constraint"),
            "priority": rule.get("priority", "normal"),
            "description": rule.get("description", ""),
            "rule_text": rule.get("rule_text", ""),
            "applies_to": rule.get("applies_to"),
            "trigger_keywords": rule.get("trigger_keywords", []),
            "trigger_tables": rule.get("trigger_tables", []),
            "trigger_columns": rule.get("trigger_columns", []),
            "auto_fix": rule.get("auto_fix"),
            "example": rule.get("example"),
            "is_active": True,
            "created_at": now,
            "updated_at": now,
            "version": 1,
            "source": "migration_v1.3"  # Track migration source
        }
        documents.append(doc)
        stats["global_rules"] += 1

    # Process database-specific rules
    print("Processing database-specific rules...")
    for db_name, db_config in data.get("database_rules", {}).items():
        db_description = db_config.get("description", "")

        for rule in db_config.get("rules", []):
            doc = {
                "rule_id": rule.get("id"),
                "scope": db_name,  # e.g., "EWRCentral", "Gin", "Warehouse"
                "database_description": db_description,
                "type": rule.get("type", "assistance"),
                "priority": rule.get("priority", "normal"),
                "description": rule.get("description", ""),
                "rule_text": rule.get("rule_text", ""),
                "applies_to": rule.get("applies_to"),
                "trigger_keywords": rule.get("trigger_keywords", []),
                "trigger_tables": rule.get("trigger_tables", []),
                "trigger_columns": rule.get("trigger_columns", []),
                "auto_fix": rule.get("auto_fix"),
                "example": rule.get("example"),
                "is_active": True,
                "created_at": now,
                "updated_at": now,
                "version": 1,
                "source": "migration_v1.3"
            }
            documents.append(doc)
            stats["database_rules"] += 1

    print(f"\nPrepared {len(documents)} documents for insertion")
    print(f"  - Global rules: {stats['global_rules']}")
    print(f"  - Database rules: {stats['database_rules']}")

    # Create indexes first
    print("\nCreating indexes...")
    indexes = [
        IndexModel([("rule_id", ASCENDING)], unique=True, name="rule_id_unique"),
        IndexModel([("scope", ASCENDING)], name="scope_idx"),
        IndexModel([("type", ASCENDING)], name="type_idx"),
        IndexModel([("is_active", ASCENDING)], name="is_active_idx"),
        IndexModel([("scope", ASCENDING), ("is_active", ASCENDING)], name="scope_active_idx"),
        IndexModel([("trigger_keywords", ASCENDING)], name="trigger_keywords_idx"),
        IndexModel([("trigger_tables", ASCENDING)], name="trigger_tables_idx"),
    ]

    await collection.create_indexes(indexes)
    print("Indexes created successfully")

    # Insert documents
    print("\nInserting documents...")
    for doc in documents:
        try:
            await collection.insert_one(doc)
            stats["total_inserted"] += 1
        except DuplicateKeyError:
            print(f"  Skipping duplicate: {doc['rule_id']} (scope: {doc['scope']})")
            stats["duplicates_skipped"] += 1

    # Summary
    print("\n" + "=" * 50)
    print("MIGRATION COMPLETE")
    print("=" * 50)
    print(f"Total rules processed: {len(documents)}")
    print(f"Successfully inserted: {stats['total_inserted']}")
    print(f"Duplicates skipped: {stats['duplicates_skipped']}")

    # Verify
    count = await collection.count_documents({})
    print(f"\nTotal documents in collection: {count}")

    # Show sample
    print("\nSample documents:")
    async for doc in collection.find().limit(3):
        print(f"  - {doc['rule_id']} (scope: {doc['scope']}, type: {doc['type']})")

    client.close()
    return True


async def verify_migration():
    """Verify the migration was successful."""
    print("\n" + "=" * 50)
    print("VERIFICATION")
    print("=" * 50)

    client = AsyncIOMotorClient(MONGODB_URI)
    db = client[MONGODB_DATABASE]
    collection = db[COLLECTION_SQL_RULES]

    # Count by scope
    pipeline = [
        {"$group": {"_id": "$scope", "count": {"$sum": 1}}},
        {"$sort": {"_id": 1}}
    ]

    print("\nRules by scope:")
    async for result in collection.aggregate(pipeline):
        print(f"  {result['_id']}: {result['count']} rules")

    # Count by type
    pipeline = [
        {"$group": {"_id": "$type", "count": {"$sum": 1}}},
        {"$sort": {"_id": 1}}
    ]

    print("\nRules by type:")
    async for result in collection.aggregate(pipeline):
        print(f"  {result['_id']}: {result['count']} rules")

    # Count with examples
    with_examples = await collection.count_documents({"example": {"$ne": None}})
    print(f"\nRules with examples: {with_examples}")

    # Count with auto_fix
    with_autofix = await collection.count_documents({"auto_fix": {"$ne": None}})
    print(f"Rules with auto_fix: {with_autofix}")

    client.close()


async def main():
    """Main entry point."""
    print("=" * 50)
    print("SQL Rules Migration: JSON -> MongoDB")
    print("=" * 50)

    success = await migrate_rules()

    if success:
        await verify_migration()
        print("\n Migration successful!")
        print("You can now update SQLRulesManager to use MongoDB.")
    else:
        print("\n Migration failed!")
        return 1

    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
