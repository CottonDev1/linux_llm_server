"""
SP Analysis Pipeline Usage Examples

This script demonstrates how to use the SP Analysis Prefect pipeline
for generating training data from stored procedures.

Prerequisites:
- MongoDB Atlas Local running on EWRSPT-AI:27018
- LLM service (llama.cpp) running on localhost:11434
- Prefect server running (optional, for dashboard visibility)
- Stored procedures already extracted and in MongoDB

Usage:
    python examples/sp_analysis_example.py
"""

import asyncio
import os
import sys
from pathlib import Path
sys.path.insert(0, '..')

# Load .env from project root (single source of truth)
from dotenv import load_dotenv
_services_dir = Path(__file__).parent.parent.parent  # python_services
_project_root = _services_dir.parent  # llm_website
load_dotenv(_project_root / ".env", override=True)

from prefect_pipelines.sp_analysis_flow import (
    analyze_single_sp,
    analyze_sp_batch,
    scheduled_sp_analysis
)


async def example_1_single_sp():
    """
    Example 1: Analyze a single stored procedure by ID

    Use this when you want to test the pipeline on a specific SP
    or regenerate training data for one procedure.
    """
    print("=" * 60)
    print("Example 1: Single SP Analysis")
    print("=" * 60)

    # You'll need to replace this with an actual SP _id from your MongoDB
    sp_id = "507f1f77bcf86cd799439011"  # Example ObjectId

    result = await analyze_single_sp(
        sp_id=sp_id,
        database="EWRCentral",
        question_count=3,
        llm_host="http://localhost:11434",
        model="qwen2.5-coder:7b"
    )

    print(f"\nResult:")
    print(f"  Success: {result['success']}")
    print(f"  SP Name: {result.get('sp_name', 'N/A')}")
    print(f"  Questions Generated: {result.get('questions_generated', 0)}")
    print(f"  Training Examples Created: {result.get('training_examples_created', 0)}")
    print(f"  Validation Score: {result.get('validation_score', 0):.2f}")
    print(f"  Duration: {result.get('duration_seconds', 0):.1f}s")

    if not result['success']:
        print(f"  Error: {result.get('error', 'Unknown error')}")


async def example_2_batch_analysis():
    """
    Example 2: Batch process stored procedures

    Use this for processing multiple SPs efficiently.
    Good for initial dataset creation or periodic updates.
    """
    print("\n" + "=" * 60)
    print("Example 2: Batch SP Analysis")
    print("=" * 60)

    result = await analyze_sp_batch(
        database="EWRCentral",
        batch_size=5,           # Process 5 SPs per batch
        max_sps=20,             # Limit to 20 total SPs (0 = all)
        question_count=3,       # Generate 3 questions per SP
        llm_host="http://localhost:11434",
        model="qwen2.5-coder:7b"
    )

    print(f"\nBatch Results:")
    print(f"  Success: {result['success']}")
    print(f"  Database: {result['database']}")
    print(f"  Total Processed: {result['total_processed']}")
    print(f"  Successful: {result['successful']}")
    print(f"  Failed: {result['failed']}")
    print(f"  Questions Generated: {result['total_questions']}")
    print(f"  Training Examples: {result['total_training_examples']}")
    print(f"  Avg Validation Score: {result['avg_validation_score']:.2f}")
    print(f"  Duration: {result['duration_seconds']:.1f}s")
    print(f"  Started: {result['started_at']}")
    print(f"  Completed: {result['completed_at']}")


async def example_3_multi_database():
    """
    Example 3: Scheduled analysis across multiple databases

    Use this for production scheduled runs that process
    all your databases automatically.
    """
    print("\n" + "=" * 60)
    print("Example 3: Multi-Database Scheduled Analysis")
    print("=" * 60)

    result = await scheduled_sp_analysis(
        databases=["EWRCentral", "EWRReporting"],
        batch_size=10,
        max_sps_per_db=50,      # 50 SPs per database max
        question_count=3
    )

    print(f"\nScheduled Analysis Results:")
    print(f"  Overall Success: {result['success']}")
    print(f"  Databases Processed: {result['databases_processed']}")
    print(f"  Total SPs: {result['total_sps']}")
    print(f"  Total Training Examples: {result['total_training_examples']}")
    print(f"  Overall Avg Score: {result['overall_avg_score']:.2f}")
    print(f"  Total Duration: {result['total_duration_seconds']:.1f}s")

    print("\n  Per-Database Breakdown:")
    for db_result in result['results']:
        print(f"    {db_result['database']}:")
        print(f"      SPs: {db_result['total_processed']}")
        print(f"      Examples: {db_result['total_training_examples']}")
        print(f"      Avg Score: {db_result['avg_validation_score']:.2f}")


async def example_4_custom_configuration():
    """
    Example 4: Custom configuration for specific needs

    Demonstrates advanced configuration options.
    """
    print("\n" + "=" * 60)
    print("Example 4: Custom Configuration")
    print("=" * 60)

    # For faster processing with smaller model
    mongodb_uri = os.environ.get("MONGODB_URI", "mongodb://localhost:27019")
    result = await analyze_sp_batch(
        database="EWRCentral",
        batch_size=20,          # Larger batches for faster throughput
        max_sps=100,
        question_count=2,       # Fewer questions for speed
        llm_host="http://localhost:11434",
        model="qwen2.5-coder:1.5b",  # Smaller, faster model
        mongodb_uri=mongodb_uri
    )

    print(f"\nCustom Configuration Results:")
    print(f"  Model: qwen2.5-coder:1.5b (optimized for speed)")
    print(f"  Questions per SP: 2")
    print(f"  Total Examples: {result['total_training_examples']}")
    print(f"  Duration: {result['duration_seconds']:.1f}s")
    print(f"  Throughput: {result['total_processed'] / result['duration_seconds']:.2f} SPs/second")


async def example_5_query_training_data():
    """
    Example 5: Query the generated training data

    Shows how to access and use the training data created by the pipeline.
    """
    print("\n" + "=" * 60)
    print("Example 5: Query Generated Training Data")
    print("=" * 60)

    from mongodb import MongoDBService
    from config import COLLECTION_SQL_EXAMPLES

    mongodb = MongoDBService()
    await mongodb.connect()

    collection = mongodb.db[COLLECTION_SQL_EXAMPLES]

    # Count total training examples
    total = await collection.count_documents({"active": True})
    print(f"\nTotal Active Training Examples: {total}")

    # Get examples for a specific database
    db_examples = await collection.count_documents({
        "database": "ewrcentral",
        "active": True
    })
    print(f"EWRCentral Examples: {db_examples}")

    # Get high-quality examples (score >= 0.8)
    high_quality = await collection.count_documents({
        "validation_score": {"$gte": 0.8},
        "active": True
    })
    print(f"High Quality Examples (>=0.8): {high_quality}")

    # Show a sample example
    sample = await collection.find_one(
        {"active": True},
        sort=[("generated_at", -1)]  # Most recent
    )

    if sample:
        print(f"\nSample Training Example:")
        print(f"  SP: {sample.get('sp_name')}")
        print(f"  Question: {sample.get('question')}")
        print(f"  SQL: {sample.get('sql')}")
        print(f"  Score: {sample.get('validation_score', 0):.2f}")
        print(f"  Generated: {sample.get('generated_at')}")


async def run_all_examples():
    """Run all examples sequentially."""
    print("\n" + "=" * 60)
    print("SP ANALYSIS PIPELINE - USAGE EXAMPLES")
    print("=" * 60)

    # Note: Example 1 requires a valid SP ID from your database
    # Uncomment to run:
    # await example_1_single_sp()

    # Example 2: Batch processing (small batch for demo)
    await example_2_batch_analysis()

    # Example 3: Multi-database (can be slow, comment out if needed)
    # await example_3_multi_database()

    # Example 4: Custom configuration
    await example_4_custom_configuration()

    # Example 5: Query results
    await example_5_query_training_data()

    print("\n" + "=" * 60)
    print("Examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    # Run examples
    asyncio.run(run_all_examples())

    print("\n\nNext Steps:")
    print("1. Review generated training data in MongoDB (sql_examples collection)")
    print("2. Check Prefect dashboard for execution details and artifacts")
    print("3. Integrate training data into your RAG SQL generation system")
    print("4. Set up scheduled deployment for continuous training data generation")
    print("\nFor scheduled deployment, see SP_ANALYSIS_README.md")
