"""
SQL Pipeline Test Flow
======================

Prefect flow for testing the SQL generation pipeline.

IMPORTANT:
- NO automatic scheduling - manual trigger only
- NO external LLM APIs - local llama.cpp only (port 8080)
- All configuration via Prefect Variables - no hardcoded values

Tests:
1. Storage: agent_learning collection operations
2. Retrieval: Cache hits, rule matching
3. Generation: SQL generation using local llama.cpp
4. E2E: Question → Cache → Rules → Schema → LLM → Validation
"""

import os
import sys
from typing import Dict, Any

from prefect import flow, task, get_run_logger
from prefect.artifacts import create_markdown_artifact
from prefect.variables import Variable

# Add python_services to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from prefect_pipelines.test_flows.base_test_flow import (
    TestFlowConfig,
    run_pytest_module,
    run_test_category,
    create_test_artifact,
)
from prefect_pipelines.test_flows.test_flow_utils import (
    TestStatus,
    TestMetrics,
    TestResult,
    ProgressTracker,
    create_test_report_markdown,
)


async def load_sql_test_config() -> Dict[str, Any]:
    """
    Load SQL test configuration from Prefect Variables.

    Returns:
        Dict with SQL connection configuration including:
        - sql_server, sql_database, sql_username, sql_password
        - sql_domain, sql_auth_type
        - default_max_tokens, sample queries
        - formatted_username (domain\\username for windows auth)
    """
    # Load SQL connection variables
    sql_server = await Variable.get("sql_server", default="localhost")
    sql_database = await Variable.get("sql_database", default="master")
    sql_username = await Variable.get("sql_username", default="sa")
    sql_password = await Variable.get("sql_password", default="")
    sql_domain = await Variable.get("sql_domain", default="")
    sql_auth_type = await Variable.get("sql_auth_type", default="sql")

    # Load LLM settings
    default_max_tokens = await Variable.get("default_max_tokens", default="2048")

    # Load sample queries
    sample_query_gin = await Variable.get("sample_query_gin", default="SELECT * FROM gin_table LIMIT 10")
    sample_query_bale = await Variable.get("sample_query_bale", default="SELECT * FROM bale_table LIMIT 10")

    # Handle auth_type: windows uses domain\username, sql uses username only
    if sql_auth_type == "windows" and sql_domain:
        formatted_username = f"{sql_domain}\\{sql_username}"
    else:
        formatted_username = sql_username

    return {
        "sql_server": sql_server,
        "sql_database": sql_database,
        "sql_username": sql_username,
        "sql_password": sql_password,
        "sql_domain": sql_domain,
        "sql_auth_type": sql_auth_type,
        "formatted_username": formatted_username,
        "default_max_tokens": int(default_max_tokens),
        "sample_query_gin": sample_query_gin,
        "sample_query_bale": sample_query_bale,
    }


@flow(
    name="sql-pipeline-tests",
    description="Run SQL pipeline tests - manual trigger only",
    log_prints=True,
)
async def sql_pipeline_test_flow(
    mongodb_uri: str = "mongodb://localhost:27017",
    llm_endpoint: str = "http://localhost:8080",
    timeout_seconds: int = 300,
    run_storage: bool = True,
    run_retrieval: bool = True,
    run_generation: bool = True,
    run_e2e: bool = True,
    cleanup_after_test: bool = True,
) -> Dict[str, Any]:
    """
    Run SQL pipeline tests.

    Args:
        mongodb_uri: MongoDB connection URI (default: mongodb://localhost:27017)
        llm_endpoint: Local SQL LLM endpoint (llama.cpp port 8080)
        timeout_seconds: Test timeout per category
        run_storage: Run storage tests
        run_retrieval: Run retrieval tests
        run_generation: Run generation tests
        run_e2e: Run end-to-end tests
        cleanup_after_test: Clean up test data

    Returns:
        Dict with test results and metrics
    """
    logger = get_run_logger()
    logger.info("Starting SQL Pipeline Tests")

    # Load SQL configuration from Prefect Variables
    sql_config = await load_sql_test_config()
    logger.info(f"Loaded SQL config: server={sql_config['sql_server']}, "
                f"database={sql_config['sql_database']}, "
                f"auth_type={sql_config['sql_auth_type']}")

    # Validate local endpoint
    if not llm_endpoint.startswith(("http://localhost", "http://127.0.0.1")):
        raise ValueError(f"Only local LLM endpoints allowed. Got: {llm_endpoint}")

    # Create config
    config = TestFlowConfig(
        mongodb_uri=mongodb_uri,
        llm_sql_endpoint=llm_endpoint,
        timeout_seconds=timeout_seconds,
        cleanup_after_test=cleanup_after_test,
        run_storage_tests=run_storage,
        run_retrieval_tests=run_retrieval,
        run_generation_tests=run_generation,
        run_e2e_tests=run_e2e,
    )

    results = []
    metrics = TestMetrics()

    # Run enabled test categories
    if run_storage:
        logger.info("Running SQL storage tests...")
        result = run_test_category("sql", "storage", config)
        results.append(result)
        metrics.add_result(result)
        logger.info(f"  Storage: {result.status.value}")

    if run_retrieval:
        logger.info("Running SQL retrieval tests...")
        result = run_test_category("sql", "retrieval", config)
        results.append(result)
        metrics.add_result(result)
        logger.info(f"  Retrieval: {result.status.value}")

    if run_generation:
        logger.info("Running SQL generation tests...")
        result = run_test_category("sql", "generation", config)
        results.append(result)
        metrics.add_result(result)
        logger.info(f"  Generation: {result.status.value}")

    if run_e2e:
        logger.info("Running SQL E2E tests...")
        result = run_test_category("sql", "e2e", config)
        results.append(result)
        metrics.add_result(result)
        logger.info(f"  E2E: {result.status.value}")

    # Create artifact
    await create_test_artifact(metrics, results, "sql")

    success = metrics.failed == 0 and metrics.errors == 0
    logger.info(f"SQL Pipeline Tests Complete: {metrics.passed}/{metrics.total} passed")

    return {
        "pipeline": "sql",
        "success": success,
        "metrics": metrics.to_dict(),
        "results": [r.to_dict() for r in results],
        "sql_config": {
            "server": sql_config["sql_server"],
            "database": sql_config["sql_database"],
            "auth_type": sql_config["sql_auth_type"],
        },
    }


if __name__ == "__main__":
    # For local testing only
    import argparse
    import asyncio

    parser = argparse.ArgumentParser(description="Run SQL Pipeline Tests")
    parser.add_argument("--mongodb-uri", default="mongodb://localhost:27017", help="MongoDB URI")
    parser.add_argument("--llm-endpoint", default="http://localhost:8080")
    parser.add_argument("--timeout", type=int, default=300)

    args = parser.parse_args()

    result = asyncio.run(sql_pipeline_test_flow(
        mongodb_uri=args.mongodb_uri,
        llm_endpoint=args.llm_endpoint,
        timeout_seconds=args.timeout,
    ))

    print(f"Result: {result}")
