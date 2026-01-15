"""Bulk audio routes for batch audio processing with concurrent job support."""
from fastapi import APIRouter, Request, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import os
import uuid
import asyncio
from datetime import datetime

from log_service import log_pipeline, log_error

router = APIRouter(prefix="/audio/bulk", tags=["Bulk Audio Processing"])


# ============================================================================
# Models
# ============================================================================

class BulkAudioRequest(BaseModel):
    """Request model for bulk audio processing"""
    source_directory: str
    output_directory: Optional[str] = None
    language: str = "auto"
    recursive: bool = True
    max_files: int = 100
    delay_between_files: float = 1.0
    default_metadata: Optional[Dict[str, Any]] = None


class BulkAudioStatusResponse(BaseModel):
    """Response model for bulk audio status"""
    success: bool
    job_id: Optional[str] = None
    is_processing: bool = False
    source_directory: Optional[str] = None
    total_files: int = 0
    processed: int = 0
    successful: int = 0
    failed: int = 0
    current_file: Optional[str] = None
    current_step: Optional[str] = None
    progress_percent: float = 0.0
    errors: List[Dict[str, Any]] = []
    results: Optional[List[Dict[str, Any]]] = None
    created_at: Optional[str] = None
    completed_at: Optional[str] = None


class BulkJobListResponse(BaseModel):
    """Response model for listing bulk processing jobs"""
    success: bool
    active_jobs: List[str]
    total_jobs: int
    jobs: Dict[str, Dict[str, Any]]


# ============================================================================
# Job State Management (replaces single global state)
# ============================================================================

# Dictionary of job states keyed by job_id (enables concurrent jobs)
_bulk_jobs: Dict[str, Dict[str, Any]] = {}

# Set of active job IDs for quick lookup
_active_jobs: set = set()

# Lock for thread-safe job state updates
_jobs_lock = asyncio.Lock()

# Maximum number of concurrent bulk jobs (configurable)
MAX_CONCURRENT_JOBS = 4


def _create_job_state(job_id: str, source_directory: str) -> Dict[str, Any]:
    """Create a new job state dictionary."""
    return {
        "job_id": job_id,
        "is_processing": True,
        "source_directory": source_directory,
        "total_files": 0,
        "processed": 0,
        "successful": 0,
        "failed": 0,
        "current_file": "Scanning directory...",
        "current_step": "Scanning directory...",
        "progress_percent": 0,
        "errors": [],
        "results": [],
        "created_at": datetime.utcnow().isoformat(),
        "completed_at": None,
        "cancel_requested": False
    }


# ============================================================================
# Helper Functions
# ============================================================================

async def process_bulk_audio_files(
    job_id: str,
    source_directory: str,
    language: str,
    recursive: bool,
    max_files: int,
    delay_between_files: float,
    user_ip: str
):
    """Background task to process audio files for a specific job."""
    global _bulk_jobs, _active_jobs

    # Get job state
    job_state = _bulk_jobs.get(job_id)
    if not job_state:
        log_error("BULK_AUDIO", user_ip, f"Job {job_id} not found in job state")
        return

    try:
        from audio_pipeline import get_audio_analysis_service

        audio_service = get_audio_analysis_service()

        # Scan for audio files
        job_state["current_step"] = "Scanning directory..."
        audio_extensions = {'.wav', '.mp3', '.m4a', '.flac', '.ogg', '.wma', '.aac'}
        audio_files = []

        if recursive:
            for root, dirs, files in os.walk(source_directory):
                for f in files:
                    if os.path.splitext(f)[1].lower() in audio_extensions:
                        audio_files.append(os.path.join(root, f))
                        if len(audio_files) >= max_files:
                            break
                if len(audio_files) >= max_files:
                    break
        else:
            for f in os.listdir(source_directory):
                full_path = os.path.join(source_directory, f)
                if os.path.isfile(full_path) and os.path.splitext(f)[1].lower() in audio_extensions:
                    audio_files.append(full_path)
                    if len(audio_files) >= max_files:
                        break

        job_state["total_files"] = len(audio_files)

        if len(audio_files) == 0:
            job_state["is_processing"] = False
            job_state["current_step"] = "No audio files found"
            job_state["completed_at"] = datetime.utcnow().isoformat()
            _active_jobs.discard(job_id)
            return

        # Process each file
        for i, audio_path in enumerate(audio_files):
            # Check for cancellation request
            if job_state.get("cancel_requested"):
                job_state["current_step"] = "Cancelled"
                job_state["is_processing"] = False
                job_state["completed_at"] = datetime.utcnow().isoformat()
                _active_jobs.discard(job_id)
                log_pipeline("BULK_AUDIO", user_ip, f"Job {job_id} cancelled by user")
                return

            filename = os.path.basename(audio_path)
            job_state["current_file"] = filename
            job_state["current_step"] = "Transcribing..."
            job_state["processed"] = i
            job_state["progress_percent"] = int((i / len(audio_files)) * 100)

            try:
                # Process audio file
                result = await audio_service.analyze_audio(audio_path, language)

                if result.get("success"):
                    job_state["successful"] += 1
                    job_state["results"].append({
                        "filename": filename,
                        "success": True,
                        "transcription": result.get("transcription"),
                        "raw_transcription": result.get("raw_transcription"),
                        "transcription_summary": result.get("transcription_summary"),
                        "emotions": result.get("emotions"),
                        "audio_metadata": result.get("audio_metadata"),
                        "language": result.get("language")
                    })
                else:
                    job_state["failed"] += 1
                    job_state["errors"].append({
                        "filename": filename,
                        "error": result.get("error", "Unknown error")
                    })
                    job_state["results"].append({
                        "filename": filename,
                        "success": False,
                        "error": result.get("error", "Unknown error")
                    })

            except Exception as e:
                job_state["failed"] += 1
                job_state["errors"].append({
                    "filename": filename,
                    "error": str(e)
                })
                job_state["results"].append({
                    "filename": filename,
                    "success": False,
                    "error": str(e)
                })

            # Rate limiting delay
            if delay_between_files > 0 and i < len(audio_files) - 1:
                await asyncio.sleep(delay_between_files)

        # Processing complete
        job_state["processed"] = len(audio_files)
        job_state["progress_percent"] = 100
        job_state["is_processing"] = False
        job_state["current_step"] = "Complete"
        job_state["current_file"] = None
        job_state["completed_at"] = datetime.utcnow().isoformat()

        _active_jobs.discard(job_id)

        log_pipeline("BULK_AUDIO", user_ip,
                    f"Job {job_id} complete: {job_state['successful']} successful, {job_state['failed']} failed")

    except Exception as e:
        job_state["is_processing"] = False
        job_state["current_step"] = "Failed"
        job_state["errors"].append({"error": str(e)})
        job_state["completed_at"] = datetime.utcnow().isoformat()
        _active_jobs.discard(job_id)
        log_error("BULK_AUDIO", user_ip, f"Job {job_id} failed", str(e))


# ============================================================================
# Endpoints
# ============================================================================

@router.post("/start")
async def start_bulk_audio_processing(
    request: Request,
    bulk_request: BulkAudioRequest,
    background_tasks: BackgroundTasks
):
    """
    Start bulk audio processing on a directory of audio files.

    Now supports concurrent jobs - each request creates a new job with a unique ID.
    Multiple jobs can run simultaneously (up to MAX_CONCURRENT_JOBS).

    Args:
        source_directory: Directory containing audio files to process
        output_directory: Optional directory for saving report
        language: Language code ('auto', 'en', 'zh', etc.)
        recursive: Whether to scan subdirectories
        max_files: Maximum files to process in one batch
        delay_between_files: Rate limiting delay in seconds
        default_metadata: Default metadata for all files

    Returns:
        JSON with job_id and status_url for tracking progress
    """
    user_ip = request.client.host if request.client else "Unknown"

    global _bulk_jobs, _active_jobs

    try:
        # Check concurrent job limit
        if len(_active_jobs) >= MAX_CONCURRENT_JOBS:
            raise HTTPException(
                status_code=429,
                detail=f"Maximum concurrent jobs ({MAX_CONCURRENT_JOBS}) reached. "
                       f"Active jobs: {list(_active_jobs)}. Use /audio/bulk/jobs to view or cancel jobs."
            )

        # Validate source directory exists
        if not os.path.exists(bulk_request.source_directory):
            raise HTTPException(
                status_code=400,
                detail=f"Source directory not found: {bulk_request.source_directory}"
            )

        # Generate unique job ID
        job_id = str(uuid.uuid4())

        log_pipeline("BULK_AUDIO", user_ip, f"Starting job {job_id}: {bulk_request.source_directory}")

        # Create new job state
        job_state = _create_job_state(job_id, bulk_request.source_directory)
        _bulk_jobs[job_id] = job_state
        _active_jobs.add(job_id)

        # Run processing in background with job_id
        background_tasks.add_task(
            process_bulk_audio_files,
            job_id,
            bulk_request.source_directory,
            bulk_request.language,
            bulk_request.recursive,
            bulk_request.max_files,
            bulk_request.delay_between_files,
            user_ip
        )

        return {
            "success": True,
            "message": "Bulk audio processing started",
            "job_id": job_id,
            "status_url": f"/audio/bulk/status/{job_id}",
            "active_jobs": len(_active_jobs),
            "max_concurrent_jobs": MAX_CONCURRENT_JOBS
        }

    except HTTPException:
        raise
    except Exception as e:
        log_error("BULK_AUDIO", user_ip, "Failed to start bulk audio processing", str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status/{job_id}", response_model=BulkAudioStatusResponse)
async def get_bulk_audio_status_by_id(request: Request, job_id: str):
    """
    Get the status of a specific bulk audio processing job.

    Args:
        job_id: Unique job identifier returned from /start endpoint
    """
    user_ip = request.client.host if request.client else "Unknown"

    global _bulk_jobs

    try:
        # Check if job exists
        if job_id not in _bulk_jobs:
            raise HTTPException(
                status_code=404,
                detail=f"Job {job_id} not found. Use /audio/bulk/jobs to list available jobs."
            )

        job_state = _bulk_jobs[job_id]

        total = job_state.get("total_files", 0)
        processed = job_state.get("processed", 0)
        progress = (processed / total * 100) if total > 0 else 0.0

        return BulkAudioStatusResponse(
            success=True,
            job_id=job_id,
            is_processing=job_state.get("is_processing", False),
            source_directory=job_state.get("source_directory"),
            total_files=total,
            processed=processed,
            successful=job_state.get("successful", 0),
            failed=job_state.get("failed", 0),
            current_file=job_state.get("current_file"),
            current_step=job_state.get("current_step"),
            progress_percent=round(progress, 1),
            errors=job_state.get("errors", []),
            results=job_state.get("results", []),
            created_at=job_state.get("created_at"),
            completed_at=job_state.get("completed_at")
        )

    except HTTPException:
        raise
    except Exception as e:
        log_error("BULK_AUDIO", user_ip, f"Failed to get status for job {job_id}", str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status", response_model=BulkAudioStatusResponse)
async def get_bulk_audio_status(request: Request):
    """
    Get the status of the most recently started bulk audio processing job.

    For backwards compatibility with existing clients.
    Use /status/{job_id} to get status of a specific job.
    """
    user_ip = request.client.host if request.client else "Unknown"

    global _bulk_jobs, _active_jobs

    try:
        # If there are active jobs, return the most recent one
        if _active_jobs:
            # Get the most recently created active job
            active_job_states = [
                (job_id, _bulk_jobs[job_id])
                for job_id in _active_jobs
                if job_id in _bulk_jobs
            ]
            if active_job_states:
                # Sort by created_at descending
                active_job_states.sort(
                    key=lambda x: x[1].get("created_at", ""),
                    reverse=True
                )
                job_id, job_state = active_job_states[0]

                total = job_state.get("total_files", 0)
                processed = job_state.get("processed", 0)
                progress = (processed / total * 100) if total > 0 else 0.0

                return BulkAudioStatusResponse(
                    success=True,
                    job_id=job_id,
                    is_processing=job_state.get("is_processing", False),
                    source_directory=job_state.get("source_directory"),
                    total_files=total,
                    processed=processed,
                    successful=job_state.get("successful", 0),
                    failed=job_state.get("failed", 0),
                    current_file=job_state.get("current_file"),
                    current_step=job_state.get("current_step"),
                    progress_percent=round(progress, 1),
                    errors=job_state.get("errors", []),
                    results=job_state.get("results", []),
                    created_at=job_state.get("created_at"),
                    completed_at=job_state.get("completed_at")
                )

        # If no active jobs, check for any completed jobs
        if _bulk_jobs:
            # Get the most recently completed job
            job_states = list(_bulk_jobs.items())
            job_states.sort(
                key=lambda x: x[1].get("created_at", ""),
                reverse=True
            )
            job_id, job_state = job_states[0]

            total = job_state.get("total_files", 0)
            processed = job_state.get("processed", 0)
            progress = (processed / total * 100) if total > 0 else 0.0

            return BulkAudioStatusResponse(
                success=True,
                job_id=job_id,
                is_processing=job_state.get("is_processing", False),
                source_directory=job_state.get("source_directory"),
                total_files=total,
                processed=processed,
                successful=job_state.get("successful", 0),
                failed=job_state.get("failed", 0),
                current_file=job_state.get("current_file"),
                current_step=job_state.get("current_step"),
                progress_percent=round(progress, 1),
                errors=job_state.get("errors", []),
                results=job_state.get("results", []),
                created_at=job_state.get("created_at"),
                completed_at=job_state.get("completed_at")
            )

        # No jobs at all
        return BulkAudioStatusResponse(
            success=True,
            is_processing=False,
            total_files=0,
            processed=0,
            successful=0,
            failed=0,
            progress_percent=0.0,
            errors=[],
            results=[]
        )

    except Exception as e:
        log_error("BULK_AUDIO", user_ip, "Failed to get bulk status", str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/jobs", response_model=BulkJobListResponse)
async def list_bulk_jobs(request: Request):
    """
    List all bulk processing jobs (active and completed).

    Returns job IDs, status, and summary for each job.
    """
    user_ip = request.client.host if request.client else "Unknown"

    global _bulk_jobs, _active_jobs

    try:
        jobs_summary = {}
        for job_id, job_state in _bulk_jobs.items():
            jobs_summary[job_id] = {
                "source_directory": job_state.get("source_directory"),
                "is_processing": job_state.get("is_processing", False),
                "progress_percent": job_state.get("progress_percent", 0),
                "total_files": job_state.get("total_files", 0),
                "successful": job_state.get("successful", 0),
                "failed": job_state.get("failed", 0),
                "current_step": job_state.get("current_step"),
                "created_at": job_state.get("created_at"),
                "completed_at": job_state.get("completed_at")
            }

        return BulkJobListResponse(
            success=True,
            active_jobs=list(_active_jobs),
            total_jobs=len(_bulk_jobs),
            jobs=jobs_summary
        )

    except Exception as e:
        log_error("BULK_AUDIO", user_ip, "Failed to list bulk jobs", str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/jobs/{job_id}")
async def cancel_bulk_job(request: Request, job_id: str):
    """
    Cancel a running bulk processing job.

    The job will be stopped after the current file completes processing.
    """
    user_ip = request.client.host if request.client else "Unknown"

    global _bulk_jobs, _active_jobs

    try:
        if job_id not in _bulk_jobs:
            raise HTTPException(
                status_code=404,
                detail=f"Job {job_id} not found"
            )

        job_state = _bulk_jobs[job_id]

        if not job_state.get("is_processing"):
            return {
                "success": True,
                "message": f"Job {job_id} is not currently processing",
                "job_id": job_id,
                "status": job_state.get("current_step", "Unknown")
            }

        # Request cancellation (the background task will check this flag)
        job_state["cancel_requested"] = True

        log_pipeline("BULK_AUDIO", user_ip, f"Cancel requested for job {job_id}")

        return {
            "success": True,
            "message": f"Cancellation requested for job {job_id}. Job will stop after current file.",
            "job_id": job_id
        }

    except HTTPException:
        raise
    except Exception as e:
        log_error("BULK_AUDIO", user_ip, f"Failed to cancel job {job_id}", str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/jobs")
async def cleanup_completed_jobs(request: Request, keep_active: bool = True):
    """
    Clean up completed jobs from memory.

    Args:
        keep_active: If True, only removes completed jobs. If False, removes all jobs.
    """
    user_ip = request.client.host if request.client else "Unknown"

    global _bulk_jobs, _active_jobs

    try:
        if keep_active:
            # Remove only completed jobs
            completed_job_ids = [
                job_id for job_id, state in _bulk_jobs.items()
                if not state.get("is_processing", False)
            ]
            for job_id in completed_job_ids:
                del _bulk_jobs[job_id]

            removed_count = len(completed_job_ids)
        else:
            # Remove all jobs (cancel active ones first)
            for job_id in list(_active_jobs):
                if job_id in _bulk_jobs:
                    _bulk_jobs[job_id]["cancel_requested"] = True

            removed_count = len(_bulk_jobs)
            _bulk_jobs.clear()
            _active_jobs.clear()

        log_pipeline("BULK_AUDIO", user_ip, f"Cleaned up {removed_count} jobs")

        return {
            "success": True,
            "message": f"Removed {removed_count} jobs",
            "remaining_jobs": len(_bulk_jobs),
            "active_jobs": len(_active_jobs)
        }

    except Exception as e:
        log_error("BULK_AUDIO", user_ip, "Failed to cleanup jobs", str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/scan")
async def scan_audio_directory(
    request: Request,
    source_directory: str = None,
    recursive: bool = True,
    max_files: int = 100
):
    """
    Scan a directory and return list of audio files without processing.

    Python manages its own configured directory. If source_directory is not provided,
    uses the configured AUDIO_BULK_DIR.

    Useful for previewing what files would be processed.
    """
    user_ip = request.client.host if request.client else "Unknown"

    try:
        # Use provided directory or default to configured bulk directory
        from config import AUDIO_ANALYSIS_CONFIG

        if not source_directory:
            # Get from environment or default
            import os
            from pathlib import Path
            source_directory = os.getenv("AUDIO_BULK_FOLDER_PATH", "./audio-files")
            if not Path(source_directory).is_absolute():
                # Make relative to project root
                source_directory = str(Path(__file__).parent.parent.parent / source_directory)

        if not os.path.exists(source_directory):
            raise HTTPException(
                status_code=400,
                detail=f"Directory not found: {source_directory}"
            )

        supported_formats = {'.wav', '.mp3', '.flac', '.m4a', '.ogg', '.webm', '.aac'}
        files = []
        total_size = 0

        # Scan for audio files
        if recursive:
            for root, dirs, filenames in os.walk(source_directory):
                for filename in filenames:
                    ext = os.path.splitext(filename)[1].lower()
                    if ext in supported_formats:
                        file_path = os.path.join(root, filename)
                        file_size = os.path.getsize(file_path)
                        total_size += file_size
                        files.append({
                            "filename": filename,
                            "filepath": file_path,
                            "format": ext,
                            "size_bytes": file_size,
                            "size_mb": round(file_size / (1024 * 1024), 2)
                        })
                        if len(files) >= max_files:
                            break
                if len(files) >= max_files:
                    break
        else:
            for filename in os.listdir(source_directory):
                file_path = os.path.join(source_directory, filename)
                if os.path.isfile(file_path):
                    ext = os.path.splitext(filename)[1].lower()
                    if ext in supported_formats:
                        file_size = os.path.getsize(file_path)
                        total_size += file_size
                        files.append({
                            "filename": filename,
                            "filepath": file_path,
                            "format": ext,
                            "size_bytes": file_size,
                            "size_mb": round(file_size / (1024 * 1024), 2)
                        })
                        if len(files) >= max_files:
                            break

        log_pipeline("BULK_AUDIO", user_ip, f"Scanned directory: found {len(files)} audio files")

        return {
            "success": True,
            "source_directory": source_directory,
            "total_files": len(files),
            "total_size_mb": round(total_size / (1024 * 1024), 2),
            "files": files,
            "truncated": len(files) >= max_files
        }

    except HTTPException:
        raise
    except Exception as e:
        log_error("BULK_AUDIO", user_ip, "Directory scan failed", str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/history")
async def get_bulk_processing_history(request: Request, limit: int = 10):
    """
    Get the history of bulk audio processing runs.
    Note: Pipeline history tracking has been migrated to Prefect.
    """
    user_ip = request.client.host if request.client else "Unknown"

    log_pipeline("BULK_AUDIO", user_ip, "Bulk history endpoint called")

    # Return empty history - pipeline tracking now uses Prefect
    return {
        "success": True,
        "total_runs": 0,
        "runs": [],
        "message": "Use Prefect dashboard for pipeline history"
    }
