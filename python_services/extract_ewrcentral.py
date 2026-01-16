#!/usr/bin/env python3
"""
Extract schema from EWRCentral on CHAD-PC
Replaces existing data in MongoDB
Phase 1: Extract schema (no summaries)
Phase 2: Generate summaries in parallel batches
"""
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import config first, then override the MongoDB URI
import config

import asyncio
from sql_pipeline.extraction.schema_extractor import SchemaExtractor
from sql_pipeline.extraction.config_parser import DatabaseConfig


async def clear_existing_data(database_key: str):
    """Clear existing schema and stored procedure data for a database"""
    from mongodb import get_mongodb_service
    from config import COLLECTION_SQL_SCHEMA_CONTEXT, COLLECTION_SQL_STORED_PROCEDURES

    mongo = get_mongodb_service()
    await mongo.initialize()

    schema_result = await mongo.db[COLLECTION_SQL_SCHEMA_CONTEXT].delete_many(
        {"database": database_key}
    )
    print(f"Deleted {schema_result.deleted_count} existing schema records", flush=True)

    sp_result = await mongo.db[COLLECTION_SQL_STORED_PROCEDURES].delete_many(
        {"database": database_key}
    )
    print(f"Deleted {sp_result.deleted_count} existing stored procedure records", flush=True)


async def main():
    db_config = DatabaseConfig(
        name='ewrcentral',
        server='CHAD-PC',
        database='EWRCentral',
        lookup_key='ewrcentral',
        user='EWRUser',
        password='66a3904d69',
        port=1433
    )

    print(f"\n{'='*60}", flush=True)
    print(f"PHASE 1: Extracting schema from {db_config.database}", flush=True)
    print(f"MongoDB: localhost:27017", flush=True)
    print(f"Mode: REPLACE existing data (no summaries - will add later)", flush=True)
    print(f"{'='*60}\n", flush=True)

    print("Clearing existing data from MongoDB...", flush=True)
    await clear_existing_data(db_config.lookup_key)
    print("", flush=True)

    # Phase 1: Extract WITHOUT summarization (fast)
    extractor = SchemaExtractor(
        verbose=True,
        enable_summarization=False,  # Skip summaries for now
        skip_existing=False
    )

    try:
        stats = await extractor.extract_single_database(db_config)
        print(f"\n{'='*60}", flush=True)
        print(f"PHASE 1 COMPLETE: {stats.tables} tables/views, {stats.procedures} procedures", flush=True)
        print(f"Duration: {stats.duration_ms}ms", flush=True)
        print(f"{'='*60}", flush=True)
        print(f"\nRun generate_summaries.py to add LLM summaries in parallel", flush=True)
    except Exception as e:
        print(f"\nERROR: {e}", flush=True)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    asyncio.run(main())
