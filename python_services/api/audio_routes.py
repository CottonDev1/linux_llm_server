"""
Audio Analysis API Routes

FastAPI router for audio analysis endpoints.
Uses the modular audio_pipeline services.
"""

import os
import json
import tempfile
import asyncio
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any

from fastapi import APIRouter, File, UploadFile, Body, HTTPException, Request, Header, Query
from fastapi.responses import StreamingResponse, FileResponse

# Import models from audio_pipeline
from audio_pipeline.models.analysis_models import (
    AudioAnalysisResponse,
    AudioMetadata,
    EmotionResult,
    AudioEventResult,
)
from audio_pipeline.models.metadata_models import (
    CallMetadata,
    CallContentAnalysis,
)
from audio_pipeline.models.storage_models import (
    AudioStoreRequest,
    AudioSearchRequest,
    AudioStatsResponse,
)
from audio_pipeline.models.bulk_models import (
    BulkScanRequest,
    BulkScanResponse,
    BulkAudioRequest,
    BulkAudioStatusResponse,
)

# Import services from audio_pipeline
from audio_pipeline.services.audio_analysis_service import get_audio_analysis_service

# Import configuration
from config import AUDIO_ANALYSIS_CONFIG

# Create router
router = APIRouter(prefix="/audio", tags=["Audio Analysis"])

# Pending analyses directory
PENDING_DIR = Path(__file__).parent.parent / "pending_analyses"

# Audio directory for bulk processing and streaming
# Get from environment or use default
AUDIO_BULK_DIR = Path(os.getenv("AUDIO_BULK_FOLDER_PATH", "./audio-files"))
if not AUDIO_BULK_DIR.is_absolute():
    # Make relative to project root (parent of python_services)
    AUDIO_BULK_DIR = Path(__file__).parent.parent.parent / AUDIO_BULK_DIR

# Allowed audio file extensions for streaming
ALLOWED_AUDIO_EXTENSIONS = {".mp3", ".wav", ".m4a", ".flac", ".ogg", ".webm", ".aac"}

# MIME type mapping for audio files
AUDIO_MIME_TYPES = {
    ".mp3": "audio/mpeg",
    ".wav": "audio/wav",
    ".m4a": "audio/mp4",
    ".flac": "audio/flac",
    ".ogg": "audio/ogg",
    ".webm": "audio/webm",
    ".aac": "audio/aac"
}


def log_pipeline(category: str, user_ip: str, message: str, details: Any = None):
    """Log pipeline activity (placeholder - will be replaced with actual logger)"""
    print(f"[{category}] {user_ip}: {message} - {details}")


def log_error(category: str, user_ip: str, message: str, error: str = None):
    """Log error (placeholder - will be replaced with actual logger)"""
    print(f"[{category}] ERROR {user_ip}: {message} - {error}")


@router.post("/upload")
async def upload_audio_file(
    request: Request,
    file: UploadFile = File(..., description="Audio file (WAV, MP3, FLAC, M4A, OGG)")
):
    """
    Upload an audio file for processing.
    Returns the temporary file path for subsequent analysis.

    NOTE: Files are saved to temporary directory only. No shared file system coupling.
    """
    user_ip = request.client.host if request.client else "Unknown"

    audio_service = get_audio_analysis_service()

    # Validate format
    if not audio_service.validate_format(file.filename or ""):
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported format. Supported: {', '.join(audio_service.get_supported_formats())}"
        )

    # Check file size
    content = await file.read()
    file_size_mb = len(content) / (1024 * 1024)
    max_size = audio_service.get_max_file_size_mb()

    if file_size_mb > max_size:
        raise HTTPException(
            status_code=400,
            detail=f"File too large. Maximum size: {max_size}MB"
        )

    # Save to temp file (no shared directory coupling)
    suffix = os.path.splitext(file.filename or "")[1] or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    log_pipeline("AUDIO", user_ip, "Audio file uploaded to temp",
                 {"filename": file.filename, "size_mb": round(file_size_mb, 2)})

    return {
        "success": True,
        "temp_path": tmp_path,
        "filename": file.filename,
        "size_mb": round(file_size_mb, 2)
    }


@router.post("/analyze", response_model=AudioAnalysisResponse)
async def analyze_audio(
    request: Request,
    audio_path: str = Body(..., description="Path to audio file (from upload)"),
    original_filename: str = Body(None, description="Original filename (for metadata parsing)"),
    language: str = Body("auto", description="Language code or 'auto' for detection"),
    summary_length_threshold: int = Body(0, description="Summary threshold in seconds (0=disabled)")
):
    """
    Analyze audio file using SenseVoice.
    Extracts transcription, emotions, and audio events.
    """
    user_ip = request.client.host if request.client else "Unknown"

    log_pipeline("AUDIO", user_ip, "Starting audio analysis",
                 {"audio_path": audio_path, "language": language})

    try:
        audio_service = get_audio_analysis_service()
        result = await audio_service.analyze_audio(audio_path, language, original_filename)

        if not result.get("success"):
            raise HTTPException(status_code=500, detail=result.get("error", "Analysis failed"))

        # Clean up temp file
        try:
            if os.path.exists(audio_path):
                os.unlink(audio_path)
        except:
            pass

        # Save analysis as pending JSON file
        PENDING_DIR.mkdir(exist_ok=True)
        filename_base = original_filename or Path(audio_path).name
        json_filename = Path(filename_base).stem + ".json"
        pending_file_path = PENDING_DIR / json_filename

        result["metadata"] = result.get("metadata", {})
        result["metadata"]["filename"] = filename_base
        if result.get("audio_metadata"):
            result["audio_metadata"]["original_filename"] = filename_base

        with open(pending_file_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        log_pipeline("AUDIO", user_ip, "Audio analysis completed",
                     {"language": result.get("language"), "transcription_length": len(result.get("transcription", ""))})

        # Convert to response model
        return AudioAnalysisResponse(
            success=result["success"],
            transcription=result["transcription"],
            transcription_summary=result.get("transcription_summary"),
            raw_transcription=result["raw_transcription"],
            emotions=EmotionResult(**result["emotions"]),
            audio_events=AudioEventResult(**result["audio_events"]),
            language=result["language"],
            audio_metadata=AudioMetadata(**result["audio_metadata"]),
            call_metadata=CallMetadata(**result["call_metadata"]) if result.get("call_metadata") else None,
            call_content=CallContentAnalysis(**result["call_content"]) if result.get("call_content") else None,
            error=result.get("error")
        )

    except HTTPException:
        raise
    except Exception as e:
        log_error("AUDIO", user_ip, "Audio analysis failed", str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/analyze-stream")
async def analyze_audio_stream(
    request: Request,
    audio_path: str = Body(..., description="Path to audio file (from upload)"),
    original_filename: str = Body(None, description="Original filename (for metadata parsing)"),
    language: str = Body("auto", description="Language code or 'auto' for detection")
):
    """
    Analyze audio file with streaming progress updates.
    Returns Server-Sent Events (SSE) with progress messages.
    """
    user_ip = request.client.host if request.client else "Unknown"

    async def generate_progress():
        progress_queue = asyncio.Queue()
        analysis_result = {"result": None, "error": None}

        async def progress_callback(step: str, message: str):
            await progress_queue.put({"step": step, "message": message})

        async def run_analysis():
            try:
                audio_service = get_audio_analysis_service()
                result = await audio_service.analyze_audio(
                    audio_path, language, original_filename, progress_callback
                )
                analysis_result["result"] = result
            except Exception as e:
                analysis_result["error"] = str(e)
            finally:
                await progress_queue.put(None)

        try:
            yield f"data: {json.dumps({'type': 'progress', 'step': 'init', 'message': 'Initializing audio analysis...'})}\n\n"
            await asyncio.sleep(0.1)

            audio_service = get_audio_analysis_service()

            yield f"data: {json.dumps({'type': 'progress', 'step': 'load', 'message': 'Loading audio file...'})}\n\n"
            await asyncio.sleep(0.1)

            # Start analysis in background
            analysis_task = asyncio.create_task(run_analysis())

            # Read progress updates
            while True:
                try:
                    progress = await asyncio.wait_for(progress_queue.get(), timeout=120.0)
                    if progress is None:
                        break
                    yield f"data: {json.dumps({'type': 'progress', 'step': progress['step'], 'message': progress['message']})}\n\n"
                except asyncio.TimeoutError:
                    yield f"data: {json.dumps({'type': 'progress', 'step': 'waiting', 'message': 'Still processing...'})}\n\n"

            await analysis_task

            if analysis_result["error"]:
                yield f"data: {json.dumps({'type': 'error', 'message': analysis_result['error']})}\n\n"
                return

            result = analysis_result["result"]
            if not result or not result.get("success"):
                error_msg = result.get('error', 'Analysis failed') if result else 'Analysis failed'
                yield f"data: {json.dumps({'type': 'error', 'message': error_msg})}\n\n"
                return

            # Clean up temp file
            try:
                if os.path.exists(audio_path):
                    os.unlink(audio_path)
            except:
                pass

            # Save as pending
            PENDING_DIR.mkdir(exist_ok=True)
            filename_base = original_filename or Path(audio_path).name
            json_filename = Path(filename_base).stem + ".json"
            pending_file_path = PENDING_DIR / json_filename

            result["metadata"] = result.get("metadata", {})
            result["metadata"]["filename"] = filename_base

            with open(pending_file_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)

            yield f"data: {json.dumps({'type': 'progress', 'step': 'save', 'message': f'Saved to pending: {json_filename}'})}\n\n"
            yield f"data: {json.dumps({'type': 'complete', 'result': result})}\n\n"

            log_pipeline("AUDIO", user_ip, "Streaming audio analysis completed", json_filename)

        except Exception as e:
            log_error("AUDIO", user_ip, "Streaming audio analysis failed", str(e))
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

    return StreamingResponse(
        generate_progress(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


@router.get("/settings/pending-path")
async def get_pending_path(request: Request):
    """Get the path where pending analysis files are saved."""
    return {
        "success": True,
        "pending_path": str(PENDING_DIR.resolve())
    }


@router.get("/pending")
async def list_pending_analyses(request: Request):
    """List all pending audio analysis JSON files."""
    user_ip = request.client.host if request.client else "Unknown"

    try:
        if not PENDING_DIR.exists():
            return {"success": True, "pending_files": [], "total_count": 0}

        pending_files = []
        for json_file in PENDING_DIR.glob("*.json"):
            try:
                stat = json_file.stat()
                pending_files.append({
                    "filename": json_file.name,
                    "created_at": datetime.fromtimestamp(stat.st_ctime).isoformat(),
                    "modified_at": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                    "size_bytes": stat.st_size
                })
            except Exception as e:
                log_error("AUDIO", user_ip, f"Failed to get stats for {json_file.name}", str(e))

        pending_files.sort(key=lambda x: x["created_at"], reverse=True)

        return {
            "success": True,
            "pending_files": pending_files,
            "total_count": len(pending_files)
        }

    except Exception as e:
        log_error("AUDIO", user_ip, "Failed to list pending analyses", str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/pending/{filename}")
async def get_pending_analysis(filename: str, request: Request):
    """Get a specific pending audio analysis JSON file."""
    user_ip = request.client.host if request.client else "Unknown"

    if ".." in filename or "/" in filename or "\\" in filename:
        raise HTTPException(status_code=400, detail="Invalid filename")

    try:
        file_path = PENDING_DIR / filename

        if not file_path.exists():
            raise HTTPException(status_code=404, detail=f"Pending analysis not found: {filename}")

        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        return {"success": True, "filename": filename, "data": data}

    except HTTPException:
        raise
    except Exception as e:
        log_error("AUDIO", user_ip, f"Failed to get pending analysis {filename}", str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/pending/{filename}")
async def delete_pending_analysis(filename: str, request: Request):
    """Delete a specific pending audio analysis JSON file."""
    user_ip = request.client.host if request.client else "Unknown"

    if ".." in filename or "/" in filename or "\\" in filename:
        raise HTTPException(status_code=400, detail="Invalid filename")

    try:
        file_path = PENDING_DIR / filename

        if not file_path.exists():
            raise HTTPException(status_code=404, detail=f"Pending analysis not found: {filename}")

        file_path.unlink()

        return {"success": True, "message": f"Pending analysis deleted: {filename}"}

    except HTTPException:
        raise
    except Exception as e:
        log_error("AUDIO", user_ip, f"Failed to delete pending analysis {filename}", str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stream/{filename}")
async def stream_audio_file(
    filename: str,
    request: Request,
    range_header: Optional[str] = Header(None, alias="Range")
):
    """
    Stream an audio file from the configured audio directory.
    Supports Range requests for seeking in audio players.

    Security:
    - Only allows files from configured audio directory
    - Validates filename to prevent path traversal
    - Only allows whitelisted audio file extensions

    Args:
        filename: Name of the audio file to stream
        range_header: HTTP Range header for partial content requests

    Returns:
        StreamingResponse with audio content
    """
    user_ip = request.client.host if request.client else "Unknown"

    # Security: Prevent path traversal attacks
    if ".." in filename or "/" in filename or "\\" in filename:
        log_error("AUDIO", user_ip, f"Path traversal attempt blocked: {filename}")
        raise HTTPException(status_code=400, detail="Invalid filename")

    # Validate file extension
    file_ext = os.path.splitext(filename)[1].lower()
    if file_ext not in ALLOWED_AUDIO_EXTENSIONS:
        log_error("AUDIO", user_ip, f"Invalid file extension: {file_ext}")
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type. Allowed: {', '.join(ALLOWED_AUDIO_EXTENSIONS)}"
        )

    # Construct full file path
    file_path = AUDIO_BULK_DIR / filename

    # Security: Ensure the resolved path is within the audio directory
    try:
        file_path = file_path.resolve()
        audio_dir_resolved = AUDIO_BULK_DIR.resolve()

        if not str(file_path).startswith(str(audio_dir_resolved)):
            log_error("AUDIO", user_ip, f"Path escape attempt blocked: {filename}")
            raise HTTPException(status_code=403, detail="Access denied")
    except Exception as e:
        log_error("AUDIO", user_ip, f"Path resolution failed for {filename}", str(e))
        raise HTTPException(status_code=400, detail="Invalid file path")

    # Check if file exists
    if not file_path.exists():
        log_error("AUDIO", user_ip, f"File not found: {filename}")
        raise HTTPException(status_code=404, detail=f"Audio file not found: {filename}")

    # Get file size
    file_size = file_path.stat().st_size

    # Get MIME type
    mime_type = AUDIO_MIME_TYPES.get(file_ext, "application/octet-stream")

    # Handle Range requests for seeking
    if range_header:
        try:
            # Parse Range header (format: "bytes=start-end")
            range_match = range_header.replace("bytes=", "").split("-")
            start = int(range_match[0]) if range_match[0] else 0
            end = int(range_match[1]) if range_match[1] else file_size - 1

            # Validate range
            if start >= file_size or end >= file_size or start > end:
                raise HTTPException(
                    status_code=416,
                    detail="Requested range not satisfiable",
                    headers={"Content-Range": f"bytes */{file_size}"}
                )

            # Calculate content length
            content_length = end - start + 1

            # Stream partial content
            def iterfile():
                with open(file_path, "rb") as f:
                    f.seek(start)
                    remaining = content_length
                    chunk_size = 8192

                    while remaining > 0:
                        chunk = f.read(min(chunk_size, remaining))
                        if not chunk:
                            break
                        remaining -= len(chunk)
                        yield chunk

            log_pipeline("AUDIO", user_ip, f"Streaming audio (partial): {filename}",
                        {"range": f"{start}-{end}/{file_size}"})

            return StreamingResponse(
                iterfile(),
                media_type=mime_type,
                status_code=206,  # Partial Content
                headers={
                    "Content-Range": f"bytes {start}-{end}/{file_size}",
                    "Accept-Ranges": "bytes",
                    "Content-Length": str(content_length),
                    "Cache-Control": "public, max-age=3600"
                }
            )
        except ValueError as e:
            log_error("AUDIO", user_ip, f"Invalid Range header: {range_header}", str(e))
            raise HTTPException(status_code=400, detail="Invalid Range header")

    # No Range header - stream entire file
    def iterfile():
        with open(file_path, "rb") as f:
            chunk_size = 8192
            while chunk := f.read(chunk_size):
                yield chunk

    log_pipeline("AUDIO", user_ip, f"Streaming audio (full): {filename}",
                {"size_mb": round(file_size / (1024 * 1024), 2)})

    return StreamingResponse(
        iterfile(),
        media_type=mime_type,
        headers={
            "Accept-Ranges": "bytes",
            "Content-Length": str(file_size),
            "Cache-Control": "public, max-age=3600"
        }
    )


@router.get("/stream-file")
async def stream_audio_by_filepath(
    request: Request,
    filepath: str = Query(..., description="Full path to the audio file"),
    range_header: Optional[str] = Header(None, alias="Range")
):
    """
    Stream an audio file from a full filepath.
    Supports Range requests for seeking in audio players.
    Used for directory-polled files where we have the full path.

    Args:
        filepath: Full path to the audio file on the server
        range_header: HTTP Range header for partial content requests

    Returns:
        StreamingResponse with audio content
    """
    user_ip = request.client.host if request.client else "Unknown"

    # Convert to Path object
    file_path = Path(filepath)

    # Validate file extension
    file_ext = file_path.suffix.lower()
    if file_ext not in ALLOWED_AUDIO_EXTENSIONS:
        log_error("AUDIO", user_ip, f"Invalid file extension: {file_ext}")
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type. Allowed: {', '.join(ALLOWED_AUDIO_EXTENSIONS)}"
        )

    # Check if file exists
    if not file_path.exists():
        log_error("AUDIO", user_ip, f"File not found: {filepath}")
        raise HTTPException(status_code=404, detail=f"Audio file not found: {filepath}")

    # Get file size
    file_size = file_path.stat().st_size

    # Get MIME type
    mime_type = AUDIO_MIME_TYPES.get(file_ext, "application/octet-stream")

    # Handle Range requests for seeking
    if range_header:
        try:
            # Parse Range header (format: "bytes=start-end")
            range_match = range_header.replace("bytes=", "").split("-")
            start = int(range_match[0]) if range_match[0] else 0
            end = int(range_match[1]) if range_match[1] else file_size - 1

            # Validate range
            if start >= file_size or end >= file_size or start > end:
                raise HTTPException(
                    status_code=416,
                    detail="Requested range not satisfiable",
                    headers={"Content-Range": f"bytes */{file_size}"}
                )

            # Calculate content length
            content_length = end - start + 1

            # Stream partial content
            def iterfile():
                with open(file_path, "rb") as f:
                    f.seek(start)
                    remaining = content_length
                    chunk_size = 8192

                    while remaining > 0:
                        chunk = f.read(min(chunk_size, remaining))
                        if not chunk:
                            break
                        remaining -= len(chunk)
                        yield chunk

            log_pipeline("AUDIO", user_ip, f"Streaming audio by path (partial): {file_path.name}",
                        {"range": f"{start}-{end}/{file_size}"})

            return StreamingResponse(
                iterfile(),
                media_type=mime_type,
                status_code=206,  # Partial Content
                headers={
                    "Content-Range": f"bytes {start}-{end}/{file_size}",
                    "Accept-Ranges": "bytes",
                    "Content-Length": str(content_length),
                    "Cache-Control": "public, max-age=3600"
                }
            )
        except ValueError as e:
            log_error("AUDIO", user_ip, f"Invalid Range header: {range_header}", str(e))
            raise HTTPException(status_code=400, detail="Invalid Range header")

    # No Range header - stream entire file
    def iterfile():
        with open(file_path, "rb") as f:
            chunk_size = 8192
            while chunk := f.read(chunk_size):
                yield chunk

    log_pipeline("AUDIO", user_ip, f"Streaming audio by path (full): {file_path.name}",
                {"size_mb": round(file_size / (1024 * 1024), 2)})

    return StreamingResponse(
        iterfile(),
        media_type=mime_type,
        headers={
            "Accept-Ranges": "bytes",
            "Content-Length": str(file_size),
            "Cache-Control": "public, max-age=3600"
        }
    )


@router.delete("/delete-file")
async def delete_audio_file(
    request: Request,
    file_path: str = Body(..., embed=True, description="Full path to the audio file to delete")
):
    """
    Delete an audio file from the file system.

    Python service manages its own file system operations.
    Node.js delegates file deletion requests to this endpoint.

    Args:
        file_path: Full path to the audio file

    Returns:
        Success message if deletion was successful
    """
    user_ip = request.client.host if request.client else "Unknown"

    try:
        file_path_obj = Path(file_path)

        # Validate file exists
        if not file_path_obj.exists():
            log_error("AUDIO", user_ip, f"File not found: {file_path}")
            raise HTTPException(status_code=404, detail="File not found")

        # Validate it's a file (not a directory)
        if not file_path_obj.is_file():
            log_error("AUDIO", user_ip, f"Path is not a file: {file_path}")
            raise HTTPException(status_code=400, detail="Path is not a file")

        # Validate it's an audio file
        file_ext = file_path_obj.suffix.lower()
        if file_ext not in ALLOWED_AUDIO_EXTENSIONS:
            log_error("AUDIO", user_ip, f"Not an audio file: {file_path}")
            raise HTTPException(
                status_code=400,
                detail=f"Not an audio file. Allowed: {', '.join(ALLOWED_AUDIO_EXTENSIONS)}"
            )

        # Attempt to delete
        file_path_obj.unlink()

        log_pipeline("AUDIO", user_ip, f"Deleted audio file: {file_path}")

        return {
            "success": True,
            "message": "File deleted successfully",
            "file_path": file_path
        }

    except PermissionError as e:
        log_error("AUDIO", user_ip, f"Permission denied deleting file: {file_path}", str(e))
        raise HTTPException(status_code=403, detail="Permission denied: Cannot delete file")
    except HTTPException:
        raise
    except Exception as e:
        log_error("AUDIO", user_ip, f"Failed to delete file: {file_path}", str(e))
        raise HTTPException(status_code=500, detail=str(e))


# Note: Additional endpoints for /store, /search, /stats, /bulk/*, /staff-metrics, etc.
# would be added here following the same pattern. They are in the original main.py
# and can be migrated incrementally.


def get_audio_router() -> APIRouter:
    """Get the audio router for inclusion in main app."""
    return router
