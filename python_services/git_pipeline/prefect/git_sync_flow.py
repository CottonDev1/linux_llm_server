"""
Prefect Git Sync Pipeline
=========================

Prefect workflow definitions for git repository synchronization with
Roslyn code analysis and vector database import.

Features:
- Pull repositories from git remote
- Run Roslyn code analysis on changed files
- Import analysis results to MongoDB vector database
- Built-in retries with exponential backoff
- Detailed logging and artifact tracking
- Prefect metrics and event emission
- Visual progress in Prefect dashboard

Flows:
- git_sync_flow: Sync a single repository
- git_sync_all_flow: Sync all configured repositories
"""

import asyncio
import os
import sys
import time
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime

from prefect import flow, task, get_run_logger
from prefect.artifacts import create_markdown_artifact
from prefect.events import emit_event

# Add path for imports
_services_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
if _services_path not in sys.path:
    sys.path.insert(0, _services_path)

# Configuration
GIT_ROOT = os.getenv("GIT_ROOT", r"C:\Projects\Git")
PROJECT_ROOT = os.getenv("PROJECT_ROOT", r"C:\Projects\llm_website")


# ============================================================================
# Result Dataclasses
# ============================================================================

@dataclass
class PullTaskResult:
    """Result from git pull task"""
    repo_name: str
    repo_path: str
    success: bool = False
    has_changes: bool = False
    output: str = ""
    error: str = ""


@dataclass
class AnalysisTaskResult:
    """Result from Roslyn analysis task"""
    repo_name: str
    success: bool = False
    entity_count: int = 0
    file_count: int = 0
    duration_seconds: float = 0.0
    output_file: str = ""
    error: str = ""


@dataclass
class ImportTaskResult:
    """Result from vector import task"""
    repo_name: str
    success: bool = False
    documents_imported: int = 0
    documents_updated: int = 0
    duration_seconds: float = 0.0
    error: str = ""


@dataclass
class SyncFlowResult:
    """Complete result from sync flow"""
    repo_name: str
    success: bool = False
    has_changes: bool = False
    pull_result: Optional[PullTaskResult] = None
    analysis_result: Optional[AnalysisTaskResult] = None
    import_result: Optional[ImportTaskResult] = None
    total_duration_seconds: float = 0.0
    error: str = ""


# ============================================================================
# Tasks
# ============================================================================

@task(
    name="git_pull",
    description="Pull latest changes from git remote",
    retries=2,
    retry_delay_seconds=10,
    tags=["git", "pull"]
)
async def git_pull_task(
    repo_path: str,
    repo_name: str
) -> PullTaskResult:
    """
    Pull latest changes from a git repository.

    Args:
        repo_path: Path to the git repository
        repo_name: Name of the repository

    Returns:
        PullTaskResult with pull status
    """
    logger = get_run_logger()
    logger.info(f"Pulling repository: {repo_name}")

    result = PullTaskResult(repo_name=repo_name, repo_path=repo_path)

    if not os.path.exists(repo_path):
        result.error = f"Repository path not found: {repo_path}"
        logger.error(result.error)
        return result

    try:
        from git_service.git_service import GitService
        git_service = GitService(git_root=GIT_ROOT)

        pull_result = await git_service.pull_repository_async(repo_path)

        result.success = pull_result.success
        result.output = pull_result.output
        result.has_changes = not pull_result.is_already_up_to_date
        result.error = pull_result.error or ""

        if result.success:
            status = "has changes" if result.has_changes else "already up to date"
            logger.info(f"Pull successful for {repo_name}: {status}")
        else:
            logger.error(f"Pull failed for {repo_name}: {result.error}")

    except Exception as e:
        result.error = str(e)
        logger.error(f"Pull failed for {repo_name}: {e}")

    return result


@task(
    name="roslyn_analysis",
    description="Run Roslyn code analysis on repository",
    retries=1,
    retry_delay_seconds=30,
    tags=["roslyn", "analysis"]
)
async def roslyn_analysis_task(
    repo_path: str,
    repo_name: str
) -> AnalysisTaskResult:
    """
    Run Roslyn analysis on a repository.

    Args:
        repo_path: Path to the git repository
        repo_name: Name of the repository

    Returns:
        AnalysisTaskResult with analysis status
    """
    logger = get_run_logger()
    logger.info(f"Running Roslyn analysis for {repo_name}")

    result = AnalysisTaskResult(repo_name=repo_name)
    start_time = time.time()

    try:
        from git_pipeline.services.roslyn_service import RoslynService
        from git_pipeline.models.pipeline_models import PipelineConfig

        config = PipelineConfig(
            git_root=GIT_ROOT,
            project_root=PROJECT_ROOT
        )
        roslyn_service = RoslynService(config=config)

        analysis_result = await roslyn_service.analyze_repository(
            repo_path=repo_path,
            repo_name=repo_name
        )

        result.success = analysis_result.success
        result.entity_count = analysis_result.entity_count
        result.file_count = analysis_result.file_count
        result.output_file = analysis_result.output_file or ""
        result.duration_seconds = analysis_result.duration_seconds
        result.error = analysis_result.error or ""

        if result.success:
            logger.info(
                f"Analysis complete for {repo_name}: "
                f"{result.entity_count} entities from {result.file_count} files"
            )
        else:
            logger.error(f"Analysis failed for {repo_name}: {result.error}")

    except Exception as e:
        result.error = str(e)
        result.duration_seconds = time.time() - start_time
        logger.error(f"Analysis failed for {repo_name}: {e}")

    return result


@task(
    name="vector_import",
    description="Import analysis results to vector database",
    retries=2,
    retry_delay_seconds=15,
    tags=["mongodb", "import", "vectors"]
)
async def vector_import_task(
    repo_name: str,
    db_name: str,
    analysis_file: str
) -> ImportTaskResult:
    """
    Import analysis results to MongoDB vector database.

    Args:
        repo_name: Name of the repository
        db_name: Database/collection prefix
        analysis_file: Path to analysis JSON file

    Returns:
        ImportTaskResult with import status
    """
    logger = get_run_logger()
    logger.info(f"Importing analysis results for {repo_name}")

    result = ImportTaskResult(repo_name=repo_name)
    start_time = time.time()

    if not analysis_file or not os.path.exists(analysis_file):
        result.error = "Analysis file not found"
        logger.warning(f"Analysis file not found for {repo_name}, skipping import")
        return result

    try:
        from git_pipeline.services.roslyn_service import RoslynService
        from git_pipeline.services.code_import_service import CodeImportService
        from git_pipeline.models.pipeline_models import PipelineConfig

        config = PipelineConfig(
            git_root=GIT_ROOT,
            project_root=PROJECT_ROOT
        )

        # Parse analysis file
        roslyn_service = RoslynService(config=config)
        analysis_data = roslyn_service._parse_analysis_output(analysis_file, repo_name)

        if not analysis_data:
            result.error = "Failed to parse analysis file"
            logger.error(f"Failed to parse analysis file for {repo_name}")
            return result

        # Import to MongoDB
        import_service = CodeImportService(config=config)
        import_result = await import_service.import_analysis(
            analysis=analysis_data,
            db_name=db_name
        )

        result.success = import_result.success
        result.documents_imported = import_result.documents_imported
        result.documents_updated = import_result.documents_updated
        result.duration_seconds = import_result.duration_seconds
        result.error = import_result.error or ""

        if result.success:
            logger.info(
                f"Import complete for {repo_name}: "
                f"{result.documents_imported} imported, {result.documents_updated} updated"
            )
        else:
            logger.error(f"Import failed for {repo_name}: {result.error}")

        # Cleanup analysis file
        try:
            os.unlink(analysis_file)
            logger.debug(f"Cleaned up analysis file: {analysis_file}")
        except Exception as e:
            logger.warning(f"Failed to cleanup analysis file: {e}")

    except Exception as e:
        result.error = str(e)
        result.duration_seconds = time.time() - start_time
        logger.error(f"Import failed for {repo_name}: {e}")

    return result


# ============================================================================
# Flows
# ============================================================================

@flow(
    name="git-sync-pipeline",
    description="Sync a git repository with Roslyn analysis and vector import",
    retries=1,
    retry_delay_seconds=60
)
async def git_sync_flow(
    repo_name: str,
    repo_path: str,
    db_name: str,
    force_analysis: bool = False
) -> Dict[str, Any]:
    """
    Git Sync Pipeline - Sync a single repository.

    This flow:
    1. Pulls latest changes from git
    2. Runs Roslyn analysis if changes detected (or force_analysis=True)
    3. Imports analysis results to vector database
    4. Creates Prefect artifacts for tracking

    Args:
        repo_name: Name of the repository
        repo_path: Path to the git repository
        db_name: Database/collection prefix for vector storage
        force_analysis: Run analysis even if no changes

    Returns:
        Dict with complete sync results
    """
    logger = get_run_logger()
    flow_start = time.time()

    logger.info(f"Starting git sync for {repo_name}")

    result = SyncFlowResult(repo_name=repo_name)

    # Step 1: Git pull
    pull_result = await git_pull_task(repo_path, repo_name)
    result.pull_result = pull_result
    result.has_changes = pull_result.has_changes

    if not pull_result.success:
        result.error = f"Pull failed: {pull_result.error}"
        result.total_duration_seconds = time.time() - flow_start

        # Emit failure event
        emit_event(
            event="git.sync.failed",
            resource={"prefect.resource.id": "git-sync-pipeline"},
            payload={
                "repository": repo_name,
                "stage": "pull",
                "error": result.error
            }
        )

        return _result_to_dict(result)

    # Check if we should run analysis
    should_analyze = pull_result.has_changes or force_analysis

    if not should_analyze:
        logger.info(f"{repo_name}: No changes, skipping analysis")
        result.success = True
        result.total_duration_seconds = time.time() - flow_start

        # Create artifact for no-changes case
        await create_markdown_artifact(
            key=f"git-sync-{repo_name}",
            markdown=f"""
## Git Sync - {repo_name}

- **Status**: No changes
- **Duration**: {result.total_duration_seconds:.2f}s
- **Message**: Repository already up to date
            """,
            description=f"Git sync for {repo_name}"
        )

        return _result_to_dict(result)

    # Step 2: Roslyn analysis
    analysis_result = await roslyn_analysis_task(repo_path, repo_name)
    result.analysis_result = analysis_result

    if not analysis_result.success:
        result.error = f"Analysis failed: {analysis_result.error}"
        # Continue to import if we have partial results
        if not analysis_result.output_file:
            result.total_duration_seconds = time.time() - flow_start
            emit_event(
                event="git.sync.failed",
                resource={"prefect.resource.id": "git-sync-pipeline"},
                payload={
                    "repository": repo_name,
                    "stage": "analysis",
                    "error": result.error
                }
            )
            return _result_to_dict(result)

    # Step 3: Vector import
    import_result = await vector_import_task(
        repo_name=repo_name,
        db_name=db_name,
        analysis_file=analysis_result.output_file
    )
    result.import_result = import_result

    if not import_result.success:
        result.error = f"Import failed: {import_result.error}"
    else:
        result.success = True

    result.total_duration_seconds = time.time() - flow_start

    # Create summary artifact
    await create_markdown_artifact(
        key=f"git-sync-{repo_name}",
        markdown=f"""
## Git Sync Complete - {repo_name}

### Overview
- **Status**: {"Success" if result.success else "Failed"}
- **Total Duration**: {result.total_duration_seconds:.2f}s
- **Has Changes**: {result.has_changes}

### Stage Results
| Stage | Status | Details |
|-------|--------|---------|
| Pull | {"OK" if pull_result.success else "FAIL"} | {"Has changes" if pull_result.has_changes else "No changes"} |
| Analysis | {"OK" if analysis_result.success else "FAIL"} | {analysis_result.entity_count} entities from {analysis_result.file_count} files |
| Import | {"OK" if import_result.success else "FAIL"} | {import_result.documents_imported} imported, {import_result.documents_updated} updated |

{"### Error" + chr(10) + result.error if result.error else ""}
        """,
        description=f"Git sync summary for {repo_name}"
    )

    # Emit completion event
    emit_event(
        event="git.sync.completed",
        resource={"prefect.resource.id": "git-sync-pipeline"},
        payload={
            "repository": repo_name,
            "success": result.success,
            "has_changes": result.has_changes,
            "entities_extracted": analysis_result.entity_count,
            "documents_imported": import_result.documents_imported,
            "duration_seconds": result.total_duration_seconds
        }
    )

    return _result_to_dict(result)


@flow(
    name="git-sync-all-pipeline",
    description="Sync all configured git repositories",
    retries=0
)
async def git_sync_all_flow(
    force_analysis: bool = False
) -> Dict[str, Any]:
    """
    Git Sync All Pipeline - Sync all configured repositories.

    This flow iterates through all configured repositories and syncs each one.

    Args:
        force_analysis: Run analysis even if no changes

    Returns:
        Dict with results for all repositories
    """
    logger = get_run_logger()
    flow_start = time.time()

    from git_pipeline.models.pipeline_models import DEFAULT_REPOSITORIES

    logger.info(f"Starting git sync for {len(DEFAULT_REPOSITORIES)} repositories")

    results = []
    success_count = 0
    failed_count = 0
    no_changes_count = 0

    for repo_config in DEFAULT_REPOSITORIES:
        if not repo_config.enabled:
            logger.info(f"Skipping disabled repository: {repo_config.name}")
            continue

        result = await git_sync_flow(
            repo_name=repo_config.name,
            repo_path=repo_config.path,
            db_name=repo_config.db_name,
            force_analysis=force_analysis
        )

        results.append(result)

        if result.get("success"):
            if result.get("has_changes"):
                success_count += 1
            else:
                no_changes_count += 1
        else:
            failed_count += 1

    total_duration = time.time() - flow_start

    # Create summary artifact
    await create_markdown_artifact(
        key="git-sync-all-summary",
        markdown=f"""
# Git Sync All Complete

## Summary
- **Total Repositories**: {len(results)}
- **Successful (with changes)**: {success_count}
- **No Changes**: {no_changes_count}
- **Failed**: {failed_count}
- **Total Duration**: {total_duration:.2f}s

## Repository Results
| Repository | Status | Changes | Entities | Imported |
|------------|--------|---------|----------|----------|
{chr(10).join([f"| {r['repo_name']} | {'OK' if r['success'] else 'FAIL'} | {'Yes' if r.get('has_changes') else 'No'} | {r.get('analysis_result', {}).get('entity_count', 0)} | {r.get('import_result', {}).get('documents_imported', 0)} |" for r in results])}
        """,
        description="Git sync all summary"
    )

    # Emit completion event
    emit_event(
        event="git.sync_all.completed",
        resource={"prefect.resource.id": "git-sync-all-pipeline"},
        payload={
            "total_repositories": len(results),
            "successful": success_count,
            "no_changes": no_changes_count,
            "failed": failed_count,
            "duration_seconds": total_duration
        }
    )

    return {
        "success": failed_count == 0,
        "total_repositories": len(results),
        "successful": success_count,
        "no_changes": no_changes_count,
        "failed": failed_count,
        "total_duration_seconds": total_duration,
        "results": results
    }


# ============================================================================
# Helper Functions
# ============================================================================

def _result_to_dict(result: SyncFlowResult) -> Dict[str, Any]:
    """Convert SyncFlowResult to dictionary."""
    return {
        "repo_name": result.repo_name,
        "success": result.success,
        "has_changes": result.has_changes,
        "error": result.error,
        "total_duration_seconds": result.total_duration_seconds,
        "pull_result": {
            "success": result.pull_result.success,
            "has_changes": result.pull_result.has_changes,
            "output": result.pull_result.output[:200] if result.pull_result else "",
            "error": result.pull_result.error if result.pull_result else ""
        } if result.pull_result else None,
        "analysis_result": {
            "success": result.analysis_result.success,
            "entity_count": result.analysis_result.entity_count,
            "file_count": result.analysis_result.file_count,
            "duration_seconds": result.analysis_result.duration_seconds,
            "error": result.analysis_result.error
        } if result.analysis_result else None,
        "import_result": {
            "success": result.import_result.success,
            "documents_imported": result.import_result.documents_imported,
            "documents_updated": result.import_result.documents_updated,
            "duration_seconds": result.import_result.duration_seconds,
            "error": result.import_result.error
        } if result.import_result else None
    }


# ============================================================================
# Synchronous Entry Points
# ============================================================================

def run_git_sync(
    repo_name: str,
    repo_path: str,
    db_name: str,
    force_analysis: bool = False
) -> Dict[str, Any]:
    """
    Run git sync flow synchronously.

    Args:
        repo_name: Repository name
        repo_path: Repository path
        db_name: Database prefix
        force_analysis: Force analysis even if no changes

    Returns:
        Dict with sync results
    """
    return asyncio.run(git_sync_flow(
        repo_name=repo_name,
        repo_path=repo_path,
        db_name=db_name,
        force_analysis=force_analysis
    ))


def run_git_sync_all(force_analysis: bool = False) -> Dict[str, Any]:
    """
    Run git sync all flow synchronously.

    Args:
        force_analysis: Force analysis even if no changes

    Returns:
        Dict with results for all repositories
    """
    return asyncio.run(git_sync_all_flow(force_analysis=force_analysis))


# ============================================================================
# CLI Entry Point
# ============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Git Sync Pipeline")
    parser.add_argument("--all", action="store_true", help="Sync all repositories")
    parser.add_argument("--repo", type=str, help="Repository name to sync")
    parser.add_argument("--force", action="store_true", help="Force analysis")

    args = parser.parse_args()

    if args.all:
        result = run_git_sync_all(force_analysis=args.force)
        print(f"Sync all result: {result}")
    elif args.repo:
        from git_pipeline.models.pipeline_models import DEFAULT_REPOSITORIES

        repo_config = next(
            (r for r in DEFAULT_REPOSITORIES if r.name.lower() == args.repo.lower()),
            None
        )
        if repo_config:
            result = run_git_sync(
                repo_name=repo_config.name,
                repo_path=repo_config.path,
                db_name=repo_config.db_name,
                force_analysis=args.force
            )
            print(f"Sync result: {result}")
        else:
            print(f"Repository not found: {args.repo}")
    else:
        parser.print_help()
