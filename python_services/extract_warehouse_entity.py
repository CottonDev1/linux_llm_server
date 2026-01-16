#!/usr/bin/env python3
"""
Extract schema from EWR.Warehouse.Entity on NCSQLTEST
"""
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import config first, then override the MongoDB URI
# (config uses dotenv with override=True, so we must patch after import)
import config

import asyncio
from sql_pipeline.extraction.schema_extractor import SchemaExtractor
from sql_pipeline.extraction.config_parser import DatabaseConfig


async def main():
    # Configure database connection
    db_config = DatabaseConfig(
        name='warehouse_entity',
        server='NCSQLTEST',
        database='EWR.Warehouse.Entity',
        lookup_key='ewr_warehouse_entity',
        user='EWRUser',
        password='66a3904d69',
        port=1433
    )

    # Create extractor with general LLM for summaries
    extractor = SchemaExtractor(
        verbose=True,
        enable_summarization=True,
        llm_url='http://localhost:8080',  # SQL/Code model
        llm_model='qwen2.5-coder-1.5b-instruct',
        skip_existing=False  # Extract fresh
    )

    print(f"\n{'='*60}", flush=True)
    print(f"Extracting schema from {db_config.database} on {db_config.server}", flush=True)
    print(f"MongoDB: localhost:27017", flush=True)
    print(f"LLM: localhost:8080 (qwen2.5-coder-1.5b)", flush=True)
    print(f"{'='*60}\n", flush=True)

    try:
        stats = await extractor.extract_single_database(db_config)
        print(f"\n{'='*60}")
        print(f"COMPLETED: {stats.tables} tables/views, {stats.procedures} procedures, {stats.errors} errors")
        print(f"Duration: {stats.duration_ms}ms")
        print(f"{'='*60}")
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    asyncio.run(main())
