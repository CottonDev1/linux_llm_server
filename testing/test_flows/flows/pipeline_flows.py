"""Prefect flows for Python pipeline tests."""
import subprocess
import time
from typing import Dict, Any, List
from dataclasses import dataclass
from pathlib import Path
import sys

from prefect import flow, task, get_run_logger
from prefect.artifacts import create_markdown_artifact

TESTING_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(TESTING_ROOT))

from config.settings import settings


@dataclass
class PipelineTestResult:
    """Result from a pipeline test run."""
    pipeline_name: str
    passed: int = 0
    failed: int = 0
    skipped: int = 0
    duration_seconds: float = 0.0
    exit_code: int = 0
    error_message: str = ""


@task(name="run_pipeline_tests", retries=1, tags=["pipeline", "pytest"])
def run_pipeline_tests(pipeline_name: str, test_dir: str) -> PipelineTestResult:
    """Run pytest for a specific pipeline."""
    logger = get_run_logger()
    start_time = time.time()

    cmd = [
        "python", "-m", "pytest",
        str(TESTING_ROOT / "pipelines" / test_dir),
        "-v", "--tb=short",
        f"--timeout={settings.pytest_timeout}"
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=settings.pytest_timeout + 60,
            cwd=str(TESTING_ROOT),
        )

        # Parse pytest output for counts
        test_result = PipelineTestResult(
            pipeline_name=pipeline_name,
            exit_code=result.returncode,
            duration_seconds=time.time() - start_time,
        )

        # Simple parsing - look for "X passed, Y failed"
        output = result.stdout + result.stderr
        if "passed" in output:
            import re
            match = re.search(r'(\d+) passed', output)
            if match:
                test_result.passed = int(match.group(1))
            match = re.search(r'(\d+) failed', output)
            if match:
                test_result.failed = int(match.group(1))
            match = re.search(r'(\d+) skipped', output)
            if match:
                test_result.skipped = int(match.group(1))

        logger.info(f"{pipeline_name}: {test_result.passed} passed, {test_result.failed} failed")
        return test_result

    except subprocess.TimeoutExpired:
        return PipelineTestResult(
            pipeline_name=pipeline_name,
            failed=1,
            error_message="Timeout",
            exit_code=124,
            duration_seconds=time.time() - start_time,
        )
    except Exception as e:
        return PipelineTestResult(
            pipeline_name=pipeline_name,
            failed=1,
            error_message=str(e),
            exit_code=1,
            duration_seconds=time.time() - start_time,
        )


# Individual pipeline flows
@flow(name="sql-pipeline-tests", log_prints=True)
def sql_pipeline_test_flow() -> Dict[str, Any]:
    """Run SQL pipeline backend tests."""
    result = run_pipeline_tests("SQL Pipeline", "sql")
    return {"success": result.failed == 0, "result": result}


@flow(name="audio-pipeline-tests", log_prints=True)
def audio_pipeline_test_flow() -> Dict[str, Any]:
    """Run audio pipeline backend tests."""
    result = run_pipeline_tests("Audio Pipeline", "audio")
    return {"success": result.failed == 0, "result": result}


@flow(name="query-pipeline-tests", log_prints=True)
def query_pipeline_test_flow() -> Dict[str, Any]:
    """Run query/RAG pipeline backend tests."""
    result = run_pipeline_tests("Query Pipeline", "query")
    return {"success": result.failed == 0, "result": result}


@flow(name="git-pipeline-tests", log_prints=True)
def git_pipeline_test_flow() -> Dict[str, Any]:
    """Run git pipeline backend tests."""
    result = run_pipeline_tests("Git Pipeline", "git")
    return {"success": result.failed == 0, "result": result}


@flow(name="code-flow-pipeline-tests", log_prints=True)
def code_flow_pipeline_test_flow() -> Dict[str, Any]:
    """Run code flow pipeline backend tests."""
    result = run_pipeline_tests("Code Flow Pipeline", "code_flow")
    return {"success": result.failed == 0, "result": result}


@flow(name="code-assistance-pipeline-tests", log_prints=True)
def code_assistance_pipeline_test_flow() -> Dict[str, Any]:
    """Run code assistance pipeline backend tests."""
    result = run_pipeline_tests("Code Assistance Pipeline", "code_assistance")
    return {"success": result.failed == 0, "result": result}


@flow(name="document-pipeline-tests", log_prints=True)
def document_pipeline_test_flow() -> Dict[str, Any]:
    """Run document pipeline backend tests."""
    result = run_pipeline_tests("Document Pipeline", "document_agent")
    return {"success": result.failed == 0, "result": result}


@flow(name="all-pipeline-tests", log_prints=True)
def all_pipelines_test_flow() -> Dict[str, Any]:
    """Run all pipeline tests sequentially."""
    logger = get_run_logger()

    pipelines = [
        ("sql", "SQL"),
        ("audio", "Audio"),
        ("query", "Query"),
        ("git", "Git"),
        ("code_flow", "Code Flow"),
        ("code_assistance", "Code Assistance"),
        ("document_agent", "Document"),
        ("shared", "Shared"),
    ]

    results = []
    for test_dir, name in pipelines:
        logger.info(f"Running {name} Pipeline tests...")
        result = run_pipeline_tests(f"{name} Pipeline", test_dir)
        results.append(result)

    total_passed = sum(r.passed for r in results)
    total_failed = sum(r.failed for r in results)

    logger.info("=" * 60)
    logger.info("All Pipeline Tests Complete")
    logger.info(f"Total: {total_passed} passed, {total_failed} failed")
    logger.info("=" * 60)

    return {
        "success": total_failed == 0,
        "total_passed": total_passed,
        "total_failed": total_failed,
        "results": results,
    }


__all__ = [
    "sql_pipeline_test_flow",
    "audio_pipeline_test_flow",
    "query_pipeline_test_flow",
    "git_pipeline_test_flow",
    "code_flow_pipeline_test_flow",
    "code_assistance_pipeline_test_flow",
    "document_pipeline_test_flow",
    "all_pipelines_test_flow",
    "shared_test_flow",
]


@flow(name="shared-tests", log_prints=True)
def shared_test_flow() -> Dict[str, Any]:
    """Run shared/common tests (embeddings, streaming, vector search)."""
    result = run_pipeline_tests("Shared Tests", "shared")
    return {"success": result.failed == 0, "result": result}


# Update __all__ to include shared_test_flow
