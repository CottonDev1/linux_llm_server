"""
Prefect Agent Learning Pipeline

Orchestrates the agent learning workflow:
1. Stats Collection - Gather learning statistics
2. Pattern Analysis - Analyze learned patterns for quality
3. Correction Processing - Process pending user corrections
4. Accuracy Reporting - Generate accuracy reports

Features:
- Built-in retries with exponential backoff
- Detailed logging and artifact tracking
- Async-native execution
- Visual progress in Prefect dashboard
"""

import asyncio
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta

# Load .env from project root (single source of truth)
from dotenv import load_dotenv
_services_dir = Path(__file__).parent.parent
_project_root = _services_dir.parent
load_dotenv(_project_root / ".env", override=True)

from prefect import flow, task, get_run_logger
from prefect.artifacts import create_markdown_artifact


@dataclass
class LearningStatsResult:
    """Result from stats collection task"""
    sql_agent_queries: int = 0
    code_agent_validations: int = 0
    patterns_learned: int = 0
    corrections_pending: int = 0
    duration_seconds: float = 0.0
    success: bool = True
    errors: List[str] = field(default_factory=list)


@dataclass
class AccuracyResult:
    """Result from accuracy analysis task"""
    code_agent_accuracy: float = 0.0
    false_positives: int = 0
    false_negatives: int = 0
    total_validations: int = 0
    duration_seconds: float = 0.0
    success: bool = True
    errors: List[str] = field(default_factory=list)


@dataclass
class CorrectionResult:
    """Result from correction processing task"""
    corrections_processed: int = 0
    corrections_verified: int = 0
    patterns_created: int = 0
    duration_seconds: float = 0.0
    success: bool = True
    errors: List[str] = field(default_factory=list)


@task(
    name="collect_learning_stats",
    description="Collect learning statistics from agent databases",
    retries=2,
    retry_delay_seconds=10,
    tags=["agent-learning", "stats"]
)
async def collect_learning_stats_task() -> LearningStatsResult:
    """
    Collect learning statistics from the agent learning database.

    Returns:
        LearningStatsResult with current counts
    """
    logger = get_run_logger()
    start_time = time.time()
    result = LearningStatsResult()

    try:
        logger.info("Collecting agent learning statistics...")

        from services.agent_learning_service import get_learning_service
        mongodb_uri = os.environ.get("MONGODB_URI", "mongodb://localhost:27017")
        learning_service = await get_learning_service(
            mongodb_uri=mongodb_uri,
            database_name="agent_learning"
        )

        # Get query count
        result.sql_agent_queries = await learning_service._db["agent_query_history"].count_documents({})
        logger.info(f"SQL Agent queries: {result.sql_agent_queries}")

        # Get validation count
        result.code_agent_validations = await learning_service._db["agent_validation_results"].count_documents({})
        logger.info(f"Code Agent validations: {result.code_agent_validations}")

        # Get patterns count
        result.patterns_learned = await learning_service._db["agent_learned_patterns"].count_documents({"active": True})
        logger.info(f"Active patterns: {result.patterns_learned}")

        # Get pending corrections
        result.corrections_pending = await learning_service._db["agent_corrections"].count_documents({"verified": False})
        logger.info(f"Pending corrections: {result.corrections_pending}")

        result.success = True

    except Exception as e:
        logger.error(f"Stats collection failed: {e}")
        result.errors.append(str(e))
        result.success = False

    result.duration_seconds = time.time() - start_time
    return result


@task(
    name="analyze_accuracy",
    description="Analyze code agent validation accuracy",
    retries=2,
    retry_delay_seconds=10,
    tags=["agent-learning", "accuracy"]
)
async def analyze_accuracy_task(days: int = 30) -> AccuracyResult:
    """
    Analyze code agent validation accuracy over time.

    Args:
        days: Number of days to analyze

    Returns:
        AccuracyResult with accuracy metrics
    """
    logger = get_run_logger()
    start_time = time.time()
    result = AccuracyResult()

    try:
        logger.info(f"Analyzing code agent accuracy (last {days} days)...")

        from services.agent_learning_service import get_learning_service
        mongodb_uri = os.environ.get("MONGODB_URI", "mongodb://localhost:27017")
        learning_service = await get_learning_service(
            mongodb_uri=mongodb_uri,
            database_name="agent_learning"
        )

        accuracy_stats = await learning_service.get_validation_accuracy(days=days)

        result.total_validations = accuracy_stats.get("total", 0)
        result.code_agent_accuracy = accuracy_stats.get("accuracy", 0)
        result.false_positives = accuracy_stats.get("false_positives", 0)
        result.false_negatives = accuracy_stats.get("false_negatives", 0)

        logger.info(f"Accuracy: {result.code_agent_accuracy:.2%}")
        logger.info(f"False positives: {result.false_positives}")
        logger.info(f"False negatives: {result.false_negatives}")

        result.success = True

    except Exception as e:
        logger.error(f"Accuracy analysis failed: {e}")
        result.errors.append(str(e))
        result.success = False

    result.duration_seconds = time.time() - start_time
    return result


@task(
    name="process_corrections",
    description="Process pending user corrections",
    retries=1,
    retry_delay_seconds=30,
    tags=["agent-learning", "corrections"]
)
async def process_corrections_task() -> CorrectionResult:
    """
    Process pending user corrections and create patterns.

    Returns:
        CorrectionResult with processing counts
    """
    logger = get_run_logger()
    start_time = time.time()
    result = CorrectionResult()

    try:
        logger.info("Processing pending corrections...")

        from services.agent_learning_service import get_learning_service
        mongodb_uri = os.environ.get("MONGODB_URI", "mongodb://localhost:27017")
        learning_service = await get_learning_service(
            mongodb_uri=mongodb_uri,
            database_name="agent_learning"
        )

        # Get verified corrections that haven't been learned from
        corrections = await learning_service.get_corrections_for_learning(
            verified_only=True,
            limit=100
        )

        for correction in corrections:
            try:
                # Create pattern from correction
                await learning_service.add_learned_pattern(
                    pattern_type="sql_generation",
                    input_pattern=correction["natural_language"],
                    output_pattern=correction["corrected_sql"],
                    database=correction.get("database"),
                    confidence=1.0,  # User-verified = high confidence
                    source="user_correction"
                )

                # Mark correction as learned
                await learning_service._db["agent_corrections"].update_one(
                    {"_id": correction["_id"]},
                    {"$set": {"learned_from": True}}
                )

                result.patterns_created += 1
                result.corrections_processed += 1

            except Exception as e:
                logger.warning(f"Failed to process correction: {e}")
                result.errors.append(str(e))

        logger.info(f"Processed {result.corrections_processed} corrections, created {result.patterns_created} patterns")
        result.success = True

    except Exception as e:
        logger.error(f"Correction processing failed: {e}")
        result.errors.append(str(e))
        result.success = False

    result.duration_seconds = time.time() - start_time
    return result


@task(
    name="generate_learning_report",
    description="Generate learning statistics report artifact",
    tags=["agent-learning", "reporting"]
)
async def generate_report_task(
    stats: LearningStatsResult,
    accuracy: AccuracyResult,
    corrections: CorrectionResult
) -> str:
    """
    Generate a markdown report of agent learning statistics.

    Returns:
        Markdown report content
    """
    logger = get_run_logger()
    logger.info("Generating agent learning report...")

    report = f"""# Agent Learning Report

**Generated:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}

## SQL Agent Statistics

| Metric | Value |
|--------|-------|
| Total Queries Processed | {stats.sql_agent_queries} |
| Active Patterns | {stats.patterns_learned} |
| Pending Corrections | {stats.corrections_pending} |

## Code Agent Accuracy

| Metric | Value |
|--------|-------|
| Total Validations | {accuracy.total_validations} |
| Accuracy | {accuracy.code_agent_accuracy:.2%} |
| False Positives | {accuracy.false_positives} |
| False Negatives | {accuracy.false_negatives} |

## Correction Processing

| Metric | Value |
|--------|-------|
| Corrections Processed | {corrections.corrections_processed} |
| New Patterns Created | {corrections.patterns_created} |

## Performance

| Task | Duration |
|------|----------|
| Stats Collection | {stats.duration_seconds:.2f}s |
| Accuracy Analysis | {accuracy.duration_seconds:.2f}s |
| Correction Processing | {corrections.duration_seconds:.2f}s |

## Recommendations

"""

    # Add recommendations based on stats
    if accuracy.false_positive_rate > 0.1 if accuracy.total_validations > 0 else False:
        report += "- ⚠️ High false positive rate - consider reviewing security patterns\n"

    if accuracy.false_negative_rate > 0.05 if accuracy.total_validations > 0 else False:
        report += "- ⚠️ False negatives detected - review validation logic\n"

    if stats.corrections_pending > 10:
        report += f"- 📝 {stats.corrections_pending} corrections pending verification\n"

    if stats.patterns_learned < 10 and stats.sql_agent_queries > 100:
        report += "- 📈 Low pattern count relative to queries - consider extracting more patterns\n"

    if not (accuracy.false_positives or accuracy.false_negatives or stats.corrections_pending > 10):
        report += "✅ System is performing well\n"

    # Create artifact
    await create_markdown_artifact(
        key="agent-learning-report",
        markdown=report,
        description="Agent Learning Statistics Report"
    )

    logger.info("Report generated and artifact created")
    return report


@flow(
    name="agent_learning_flow",
    description="Analyze and process agent learning data",
    retries=1,
    retry_delay_seconds=60
)
async def run_agent_learning_flow():
    """
    Main flow for agent learning analysis and processing.

    Steps:
    1. Collect learning statistics
    2. Analyze code agent accuracy
    3. Process pending corrections
    4. Generate report artifact
    """
    logger = get_run_logger()
    start_time = time.time()

    logger.info("Starting Agent Learning Flow")
    logger.info("=" * 50)

    # Step 1: Collect stats
    stats = await collect_learning_stats_task()

    # Step 2: Analyze accuracy
    accuracy = await analyze_accuracy_task(days=30)

    # Step 3: Process corrections
    corrections = await process_corrections_task()

    # Step 4: Generate report
    report = await generate_report_task(stats, accuracy, corrections)

    total_duration = time.time() - start_time
    logger.info("=" * 50)
    logger.info(f"Agent Learning Flow completed in {total_duration:.2f}s")

    return {
        "success": stats.success and accuracy.success and corrections.success,
        "stats": {
            "queries": stats.sql_agent_queries,
            "patterns": stats.patterns_learned,
            "accuracy": accuracy.code_agent_accuracy
        },
        "duration_seconds": total_duration
    }


# Export for scheduled flows
__all__ = ["run_agent_learning_flow"]
