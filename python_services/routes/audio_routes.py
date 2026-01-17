"""Audio routes for audio transcription and analysis."""
from fastapi import APIRouter, Request, Query, HTTPException, Body, File, UploadFile
from fastapi.responses import StreamingResponse
from typing import Optional, List, Dict, Any
from pathlib import Path
from pydantic import BaseModel
import os
import json
import tempfile
from datetime import datetime
import asyncio

from data_models import (
    AudioStoreRequest, AudioSearchRequest
)
from mongodb import get_mongodb_service
from log_service import log_pipeline, log_error

router = APIRouter(prefix="/audio", tags=["Audio Analysis"])

# Directory for unanalyzed uploaded files (persists across restarts)
UNANALYZED_DIR = Path(__file__).parent.parent / "unanalyzed_uploads"
UNANALYZED_DIR.mkdir(exist_ok=True)


class TicketMatchRequest(BaseModel):
    """Request model for finding tickets that match a phone call"""
    extension: Optional[str] = None  # Staff extension who took the call
    phone_number: Optional[str] = None  # Caller's phone number
    call_datetime: Optional[str] = None  # ISO format datetime of the call
    subject_keywords: Optional[List[str]] = None  # Keywords from call subject
    customer_name: Optional[str] = None  # Customer name from call
    time_window_minutes: int = 60  # How many minutes after call to search


@router.post("/upload")
async def upload_audio_file(
    request: Request,
    file: UploadFile = File(..., description="Audio file (WAV, MP3, FLAC, M4A, OGG)")
):
    """
    Upload an audio file for processing.
    Returns the temporary file path for subsequent analysis.
    """
    user_ip = request.client.host if request.client else "Unknown"

    from audio_pipeline import get_audio_analysis_service
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

    # Save to unanalyzed folder with original filename (for persistence)
    original_filename = file.filename or f"upload_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
    # Ensure unique filename by adding timestamp if file exists
    save_path = UNANALYZED_DIR / original_filename
    if save_path.exists():
        name, ext = os.path.splitext(original_filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        original_filename = f"{name}_{timestamp}{ext}"
        save_path = UNANALYZED_DIR / original_filename

    with open(save_path, "wb") as f:
        f.write(content)

    log_pipeline("AUDIO", user_ip, "Audio file uploaded",
                 original_filename,
                 details={"size_mb": round(file_size_mb, 2)})

    return {
        "success": True,
        "filepath": str(save_path),
        "filename": original_filename,
        "size_mb": round(file_size_mb, 2)
    }


@router.get("/unanalyzed")
async def list_unanalyzed_files(request: Request):
    """
    List all unanalyzed uploaded audio files.
    Returns filename, size, and upload time for each.
    """
    user_ip = request.client.host if request.client else "Unknown"

    try:
        # Supported audio extensions
        audio_extensions = {'.wav', '.mp3', '.m4a', '.flac', '.ogg', '.wma'}

        unanalyzed_files = []
        for file_path in UNANALYZED_DIR.iterdir():
            if file_path.is_file() and file_path.suffix.lower() in audio_extensions:
                try:
                    stat = file_path.stat()
                    unanalyzed_files.append({
                        "filename": file_path.name,
                        "filepath": str(file_path),
                        "size_mb": round(stat.st_size / (1024 * 1024), 2),
                        "uploaded_at": datetime.fromtimestamp(stat.st_mtime).isoformat()
                    })
                except Exception as e:
                    log_error("AUDIO", user_ip, f"Failed to get stats for {file_path.name}: {str(e)}")

        # Sort by upload time (newest first)
        unanalyzed_files.sort(key=lambda x: x["uploaded_at"], reverse=True)

        log_pipeline("AUDIO", user_ip, "Listed unanalyzed files", details={"count": len(unanalyzed_files)})

        return {
            "success": True,
            "files": unanalyzed_files,
            "total_count": len(unanalyzed_files)
        }

    except Exception as e:
        log_error("AUDIO", user_ip, "Failed to list unanalyzed files", str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/unanalyzed/{filename}")
async def delete_unanalyzed_file(filename: str, request: Request):
    """
    Delete an unanalyzed audio file by filename.
    """
    user_ip = request.client.host if request.client else "Unknown"

    # Security: prevent path traversal attacks
    if ".." in filename or "/" in filename or "\\" in filename:
        raise HTTPException(status_code=400, detail="Invalid filename")

    try:
        file_path = UNANALYZED_DIR / filename

        if not file_path.exists():
            raise HTTPException(status_code=404, detail=f"File not found: {filename}")

        file_path.unlink()
        log_pipeline("AUDIO", user_ip, "Deleted unanalyzed file", filename)

        return {
            "success": True,
            "message": f"Deleted: {filename}"
        }

    except HTTPException:
        raise
    except Exception as e:
        log_error("AUDIO", user_ip, f"Failed to delete unanalyzed file {filename}", str(e))
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
        import time
        start_time = time.time()

        # Use an async queue to receive progress updates from the analysis
        progress_queue = asyncio.Queue()
        analysis_result = {"result": None, "error": None}

        def get_elapsed():
            """Get elapsed time since start"""
            return round(time.time() - start_time, 1)

        async def progress_callback(step: str, message: str):
            """Callback that puts progress messages on the queue"""
            await progress_queue.put({"step": step, "message": message, "elapsed": get_elapsed()})

        async def run_analysis():
            """Run the analysis in a background task"""
            try:
                from audio_pipeline import get_audio_analysis_service
                audio_service = get_audio_analysis_service()
                result = await audio_service.analyze_audio(
                    audio_path, language, original_filename, progress_callback
                )
                analysis_result["result"] = result
            except Exception as e:
                analysis_result["error"] = str(e)
            finally:
                # Signal completion by putting None on the queue
                await progress_queue.put(None)

        try:
            # Get GPU info for display
            gpu_info = "CPU"
            try:
                import torch
                if torch.cuda.is_available():
                    gpu_id = int(os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0])
                    gpu_name = torch.cuda.get_device_name(0)  # Get first visible device
                    # Shorten common GPU names
                    if "GeForce" in gpu_name:
                        gpu_name = gpu_name.replace("NVIDIA GeForce ", "")
                    gpu_info = f"GPU {gpu_id}: {gpu_name}"
            except Exception:
                pass

            # Progress: Starting
            yield f"data: {json.dumps({'type': 'progress', 'step': 'init', 'message': f'Initializing on {gpu_info}...', 'gpu': gpu_info, 'elapsed': get_elapsed()})}\n\n"
            await asyncio.sleep(0.1)

            from audio_pipeline import get_audio_analysis_service
            audio_service = get_audio_analysis_service()

            # Progress: Loading audio
            yield f"data: {json.dumps({'type': 'progress', 'step': 'load', 'message': 'Loading audio file...', 'elapsed': get_elapsed()})}\n\n"
            await asyncio.sleep(0.1)

            # Get audio metadata first to show duration
            duration = 0
            if hasattr(audio_service, '_get_audio_metadata'):
                metadata = audio_service._get_audio_metadata(audio_path)
                duration = metadata.get('duration_seconds', 0)
                duration_str = f"{int(duration // 60)}m {int(duration % 60)}s"
                yield f"data: {json.dumps({'type': 'progress', 'step': 'metadata', 'message': f'Audio loaded: {duration_str} duration', 'elapsed': get_elapsed()})}\n\n"
            await asyncio.sleep(0.1)

            # Progress: Transcribing
            yield f"data: {json.dumps({'type': 'progress', 'step': 'transcribe', 'message': 'Transcribing audio with SenseVoice...', 'elapsed': get_elapsed()})}\n\n"

            # Check if chunking will be used
            if hasattr(audio_service, 'vad_available') and not audio_service.vad_available:
                if duration > 25:
                    chunks = int(duration / 25) + 1
                    yield f"data: {json.dumps({'type': 'progress', 'step': 'transcribe', 'message': f'Will process {chunks} audio chunks (long recording)...', 'elapsed': get_elapsed()})}\n\n"

            # Start the analysis in a background task
            analysis_task = asyncio.create_task(run_analysis())

            # Read progress updates from the queue and yield them as SSE events
            while True:
                try:
                    # Wait for progress with a timeout
                    progress = await asyncio.wait_for(progress_queue.get(), timeout=120.0)

                    if progress is None:
                        # Analysis complete
                        break

                    # Yield progress event with timing info
                    elapsed = progress.get('elapsed', get_elapsed())
                    yield f"data: {json.dumps({'type': 'progress', 'step': progress['step'], 'message': progress['message'], 'elapsed': elapsed})}\n\n"

                except asyncio.TimeoutError:
                    yield f"data: {json.dumps({'type': 'progress', 'step': 'waiting', 'message': 'Still processing...', 'elapsed': get_elapsed()})}\n\n"

            # Wait for the task to complete (should already be done)
            await analysis_task

            # Check for errors
            if analysis_result["error"]:
                yield f"data: {json.dumps({'type': 'error', 'message': analysis_result['error']})}\n\n"
                return

            result = analysis_result["result"]
            if not result or not result.get("success"):
                error_msg = result.get('error', 'Analysis failed') if result else 'Analysis failed'
                yield f"data: {json.dumps({'type': 'error', 'message': error_msg})}\n\n"
                return

            # Progress: Emotions
            emotions = result.get("emotions", {}).get("detected", [])
            primary = result.get("emotions", {}).get("primary", "NEUTRAL")
            yield f"data: {json.dumps({'type': 'progress', 'step': 'emotions', 'message': f'Emotions detected: {primary} ({len(emotions)} total)', 'elapsed': get_elapsed()})}\n\n"
            await asyncio.sleep(0.1)

            # Clean up temp file ONLY if it's in our upload directory
            # Do NOT delete files that were polled from user directories
            try:
                audio_path_obj = Path(audio_path).resolve()
                unanalyzed_dir_resolved = UNANALYZED_DIR.resolve()
                # Only delete if the file is inside our unanalyzed_uploads directory
                # Use string comparison with normalized paths for reliability
                audio_path_str = str(audio_path_obj)
                unanalyzed_dir_str = str(unanalyzed_dir_resolved)
                is_in_unanalyzed = audio_path_str.startswith(unanalyzed_dir_str + os.sep)

                print(f"File cleanup check: audio_path={audio_path_str}")
                print(f"File cleanup check: unanalyzed_dir={unanalyzed_dir_str}")
                print(f"File cleanup check: is_in_unanalyzed={is_in_unanalyzed}")

                if is_in_unanalyzed:
                    if os.path.exists(audio_path):
                        os.unlink(audio_path)
                        print(f"Cleaned up uploaded temp file: {audio_path}")
                        yield f"data: {json.dumps({'type': 'progress', 'step': 'cleanup', 'message': 'Removed source file from uploads'})}\n\n"
                    else:
                        print(f"File already removed: {audio_path}")
                else:
                    print(f"Preserved original file (not in upload dir): {audio_path}")
            except Exception as e:
                print(f"Cleanup error: {e}")
                import traceback
                traceback.print_exc()

            # Save as pending
            pending_dir = Path(__file__).parent.parent / "audio-files"
            pending_dir.mkdir(exist_ok=True)
            filename_base = original_filename or Path(audio_path).name
            json_filename = Path(filename_base).stem + ".json"
            pending_file_path = pending_dir / json_filename

            result["metadata"] = result.get("metadata", {})
            result["metadata"]["filename"] = filename_base
            if result.get("audio_metadata"):
                result["audio_metadata"]["original_filename"] = filename_base

            with open(pending_file_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)

            total_time = get_elapsed()
            yield f"data: {json.dumps({'type': 'progress', 'step': 'save', 'message': f'Saved to pending: {json_filename}', 'elapsed': total_time})}\n\n"

            # Final result - include timing and GPU info
            result["processing_info"] = {
                "total_time_seconds": total_time,
                "gpu": gpu_info
            }
            yield f"data: {json.dumps({'type': 'complete', 'result': result, 'total_time': total_time, 'gpu': gpu_info})}\n\n"

            log_pipeline("AUDIO", user_ip, "Streaming audio analysis completed", json_filename, details={"time": total_time, "gpu": gpu_info})

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


@router.get("/pending")
async def list_pending_analyses(request: Request):
    """
    List all pending audio analysis JSON files.
    Returns filename, created_at timestamp, and file size for each.
    """
    user_ip = request.client.host if request.client else "Unknown"

    try:
        pending_dir = Path(__file__).parent.parent / "audio-files"

        if not pending_dir.exists():
            return {"success": True, "pending_files": []}

        pending_files = []
        for json_file in pending_dir.glob("*.json"):
            try:
                stat = json_file.stat()
                pending_files.append({
                    "filename": json_file.name,
                    "created_at": datetime.fromtimestamp(stat.st_ctime).isoformat(),
                    "modified_at": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                    "size_bytes": stat.st_size
                })
            except Exception as e:
                log_error("AUDIO", user_ip, f"Failed to get stats for {json_file.name}: {str(e)}")

        # Sort by created time (newest first)
        pending_files.sort(key=lambda x: x["created_at"], reverse=True)

        log_pipeline("AUDIO", user_ip, "Listed pending analyses", details={"count": len(pending_files)})

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
    """
    Get a specific pending audio analysis JSON file by filename.
    Returns the full analysis data.
    """
    user_ip = request.client.host if request.client else "Unknown"

    # Security: prevent path traversal attacks
    if ".." in filename or "/" in filename or "\\" in filename:
        raise HTTPException(status_code=400, detail="Invalid filename")

    try:
        pending_dir = Path(__file__).parent.parent / "audio-files"
        file_path = pending_dir / filename

        if not file_path.exists():
            raise HTTPException(status_code=404, detail=f"Pending analysis not found: {filename}")

        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        log_pipeline("AUDIO", user_ip, "Retrieved pending analysis", filename)

        return {
            "success": True,
            "filename": filename,
            "data": data
        }

    except HTTPException:
        raise
    except Exception as e:
        log_error("AUDIO", user_ip, f"Failed to get pending analysis {filename}", str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/pending/{filename}")
async def delete_pending_analysis(filename: str, request: Request):
    """
    Delete a specific pending audio analysis JSON file by filename.
    """
    user_ip = request.client.host if request.client else "Unknown"

    # Security: prevent path traversal attacks
    if ".." in filename or "/" in filename or "\\" in filename:
        raise HTTPException(status_code=400, detail="Invalid filename")

    try:
        pending_dir = Path(__file__).parent.parent / "audio-files"
        file_path = pending_dir / filename

        if not file_path.exists():
            raise HTTPException(status_code=404, detail=f"Pending analysis not found: {filename}")

        file_path.unlink()

        log_pipeline("AUDIO", user_ip, "Deleted pending analysis", filename)

        return {
            "success": True,
            "message": f"Pending analysis deleted: {filename}"
        }

    except HTTPException:
        raise
    except Exception as e:
        log_error("AUDIO", user_ip, f"Failed to delete pending analysis {filename}", str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stream/{filename}")
async def stream_audio_file(
    filename: str,
    request: Request
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

    Returns:
        StreamingResponse with audio content
    """
    user_ip = request.client.host if request.client else "Unknown"

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

    # Get audio directory from environment
    audio_dir_path = os.getenv("AUDIO_BULK_FOLDER_PATH", "./audio-files")
    if not os.path.isabs(audio_dir_path):
        # Make relative to project root (parent of python_services)
        audio_dir = Path(__file__).parent.parent.parent / audio_dir_path
    else:
        audio_dir = Path(audio_dir_path)

    # Construct full file path
    file_path = audio_dir / filename

    # Security: Ensure the resolved path is within the audio directory
    try:
        file_path = file_path.resolve()
        audio_dir_resolved = audio_dir.resolve()

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

    # Get Range header if present
    range_header = request.headers.get("range")

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


@router.post("/store")
async def store_audio_analysis(
    request: Request,
    audio_data: AudioStoreRequest
):
    """
    Store audio analysis with embeddings for semantic search.
    If pending_filename is provided, deletes the pending JSON file after successful save.
    Requires customer_support_staff to be set - calls cannot be saved without staff assignment.
    """
    user_ip = request.client.host if request.client else "Unknown"

    # Validate required fields - customer_support_staff is mandatory
    if not audio_data.customer_support_staff or not audio_data.customer_support_staff.strip():
        log_error("AUDIO", user_ip, "Store rejected: customer_support_staff is required", audio_data.filename)
        raise HTTPException(
            status_code=400,
            detail="Customer Support Staff is required. Please select a staff member before saving."
        )

    log_pipeline("AUDIO", user_ip, "Storing audio analysis",
                 audio_data.filename,
                 details={
                     "customer_support_staff": audio_data.customer_support_staff,
                     "ewr_customer": audio_data.ewr_customer,
                     "mood": audio_data.mood,
                     "outcome": audio_data.outcome,
                     "pending_filename": audio_data.pending_filename
                 })

    try:
        mongodb = get_mongodb_service()

        # Convert Pydantic models to dicts
        result = await mongodb.store_audio_analysis(
            customer_support_staff=audio_data.customer_support_staff,
            ewr_customer=audio_data.ewr_customer,
            mood=audio_data.mood,
            outcome=audio_data.outcome,
            filename=audio_data.filename,
            transcription=audio_data.transcription,
            raw_transcription=audio_data.raw_transcription,
            emotions=audio_data.emotions.dict(),
            audio_events=audio_data.audio_events.dict(),
            language=audio_data.language,
            audio_metadata=audio_data.audio_metadata.dict(),
            transcription_summary=audio_data.transcription_summary,
            # New fields for call metadata (parsed from filename)
            call_metadata=audio_data.call_metadata.dict() if audio_data.call_metadata else None,
            # New fields for LLM-analyzed content
            call_content=audio_data.call_content.dict() if audio_data.call_content else None,
            # Related ticket IDs from EWRCentral
            related_ticket_ids=audio_data.related_ticket_ids or [],
            # Speaker diarization results
            speaker_diarization=audio_data.speaker_diarization.dict() if audio_data.speaker_diarization else None
        )

        log_pipeline("AUDIO", user_ip, "Audio analysis stored",
                     details={"analysis_id": result.get("analysis_id")})

        # Delete pending JSON file if provided
        if audio_data.pending_filename:
            try:
                # Security: prevent path traversal
                filename = audio_data.pending_filename
                if ".." not in filename and "/" not in filename and "\\" not in filename:
                    pending_dir = Path(__file__).parent.parent / "audio-files"
                    pending_file = pending_dir / filename
                    if pending_file.exists():
                        pending_file.unlink()
                        log_pipeline("AUDIO", user_ip, "Deleted pending analysis file", filename)
                    else:
                        log_error("AUDIO", user_ip, f"Pending file not found: {filename}")
                else:
                    log_error("AUDIO", user_ip, f"Invalid pending filename: {filename}")
            except Exception as e:
                # Don't fail the entire operation if pending file deletion fails
                log_error("AUDIO", user_ip, f"Failed to delete pending file: {str(e)}")

        return result

    except Exception as e:
        log_error("AUDIO", user_ip, "Failed to store audio analysis", str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/search")
async def search_audio_analyses(
    request: Request,
    search_params: AudioSearchRequest
):
    """
    Search audio analyses using semantic similarity with optional filters.
    """
    user_ip = request.client.host if request.client else "Unknown"

    log_pipeline("AUDIO", user_ip, "Searching audio analyses",
                 search_params.query,
                 details={
                     "limit": search_params.limit,
                     "filters": {
                         "customer_support_staff": search_params.customer_support_staff,
                         "ewr_customer": search_params.ewr_customer,
                         "mood": search_params.mood,
                         "outcome": search_params.outcome,
                         "emotion": search_params.emotion,
                         "language": search_params.language
                     }
                 })

    try:
        mongodb = get_mongodb_service()

        results = await mongodb.search_audio_analyses(
            query=search_params.query,
            limit=search_params.limit,
            customer_support_staff=search_params.customer_support_staff,
            ewr_customer=search_params.ewr_customer,
            mood=search_params.mood,
            outcome=search_params.outcome,
            emotion=search_params.emotion,
            language=search_params.language
        )

        log_pipeline("AUDIO", user_ip, "Audio search completed",
                     details={"results_count": len(results)})

        return {
            "success": True,
            "results": results,
            "count": len(results)
        }

    except Exception as e:
        log_error("AUDIO", user_ip, "Audio search failed", str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats/summary")
async def get_audio_stats(request: Request):
    """
    Get audio analysis statistics.
    """
    user_ip = request.client.host if request.client else "Unknown"

    try:
        mongodb = get_mongodb_service()
        stats = await mongodb.get_audio_stats()

        log_pipeline("AUDIO", user_ip, "Retrieved audio statistics")

        return stats

    except Exception as e:
        log_error("AUDIO", user_ip, "Failed to retrieve audio statistics", str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats/by-staff")
async def get_audio_stats_by_staff(request: Request):
    """
    Get audio analysis statistics grouped by customer support staff.
    Returns aggregated mood counts for each staff member.
    """
    user_ip = request.client.host if request.client else "Unknown"

    try:
        mongodb = get_mongodb_service()

        # Get collection (uses same collection as mongodb_service)
        from config import COLLECTION_AUDIO_ANALYSIS
        collection = mongodb.db[COLLECTION_AUDIO_ANALYSIS]

        # Aggregation pipeline to group by staff and count emotions
        # Note: emotions.primary contains specific emotions (HAPPY, SAD, ANGRY, etc.)
        pipeline = [
            {
                "$group": {
                    "_id": "$customer_support_staff",
                    "total_calls": {"$sum": 1},
                    "moods": {"$push": "$emotions.primary"}
                }
            },
            {
                "$project": {
                    "_id": 0,
                    "staff_name": "$_id",
                    "total_calls": 1,
                    "mood_counts": {
                        "HAPPY": {
                            "$size": {
                                "$filter": {
                                    "input": "$moods",
                                    "as": "m",
                                    "cond": {"$eq": ["$$m", "HAPPY"]}
                                }
                            }
                        },
                        "SAD": {
                            "$size": {
                                "$filter": {
                                    "input": "$moods",
                                    "as": "m",
                                    "cond": {"$eq": ["$$m", "SAD"]}
                                }
                            }
                        },
                        "ANGRY": {
                            "$size": {
                                "$filter": {
                                    "input": "$moods",
                                    "as": "m",
                                    "cond": {"$eq": ["$$m", "ANGRY"]}
                                }
                            }
                        },
                        "NEUTRAL": {
                            "$size": {
                                "$filter": {
                                    "input": "$moods",
                                    "as": "m",
                                    "cond": {"$eq": ["$$m", "NEUTRAL"]}
                                }
                            }
                        },
                        "FEARFUL": {
                            "$size": {
                                "$filter": {
                                    "input": "$moods",
                                    "as": "m",
                                    "cond": {"$eq": ["$$m", "FEARFUL"]}
                                }
                            }
                        },
                        "DISGUSTED": {
                            "$size": {
                                "$filter": {
                                    "input": "$moods",
                                    "as": "m",
                                    "cond": {"$eq": ["$$m", "DISGUSTED"]}
                                }
                            }
                        },
                        "SURPRISED": {
                            "$size": {
                                "$filter": {
                                    "input": "$moods",
                                    "as": "m",
                                    "cond": {"$eq": ["$$m", "SURPRISED"]}
                                }
                            }
                        }
                    }
                }
            },
            {"$sort": {"total_calls": -1}}
        ]

        results = await collection.aggregate(pipeline).to_list(length=None)

        log_pipeline("AUDIO", user_ip, "Retrieved staff statistics",
                     details={"staff_count": len(results)})

        return {
            "success": True,
            "staff": results,
            "total_staff": len(results)
        }

    except Exception as e:
        log_error("AUDIO", user_ip, "Failed to retrieve staff statistics", str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/lookup-staff/{extension}")
async def lookup_staff_by_extension(extension: str, request: Request):
    """
    Lookup customer support staff name by phone extension from EWRCentral database.
    Uses the CentralUsers table to find the staff member associated with the extension.
    """
    user_ip = request.client.host if request.client else "Unknown"

    log_pipeline("AUDIO", user_ip, "Looking up staff by extension", extension)

    try:
        import pymssql

        # Connect to EWRSQLPROD with domain credentials
        connection = pymssql.connect(
            server='EWRSQLPROD',
            database='EWRCentral',
            user='EWR\\chad.walker',
            password='6454@@Christina',
            port='1433'
        )

        cursor = connection.cursor(as_dict=True)

        # Query to find staff by extension
        # The column is OfficePhoneExtension, try exact match and trimmed versions
        query = """
            SELECT TOP 1
                cu.CentralUserID,
                cu.FirstName,
                cu.LastName,
                cu.FirstName + ' ' + cu.LastName AS FullName,
                cu.OfficeEmailAddress AS Email,
                LTRIM(RTRIM(cu.OfficePhoneExtension)) AS PhoneExtension
            FROM CentralUsers cu
            WHERE cu.IsActive = 1
              AND (
                  LTRIM(RTRIM(cu.OfficePhoneExtension)) = %s
                  OR LTRIM(RTRIM(cu.OfficePhoneExtension)) = %s
              )
            ORDER BY cu.LastUpdateUTC DESC
        """

        # Try the extension as-is and with leading zeros stripped
        ext_stripped = extension.lstrip('0') if extension else extension
        cursor.execute(query, (extension, ext_stripped))

        result = cursor.fetchone()
        cursor.close()
        connection.close()

        if result:
            log_pipeline("AUDIO", user_ip, "Staff found for extension",
                         details={"extension": extension, "staff": result['FullName']})
            return {
                "success": True,
                "found": True,
                "extension": extension,
                "staff_name": result['FullName'],
                "first_name": result['FirstName'],
                "last_name": result['LastName'],
                "email": result.get('Email'),
                "user_id": result['CentralUserID']
            }
        else:
            log_pipeline("AUDIO", user_ip, "No staff found for extension", extension)
            return {
                "success": True,
                "found": False,
                "extension": extension,
                "staff_name": None,
                "message": f"No staff member found for extension {extension}"
            }

    except Exception as e:
        log_error("AUDIO", user_ip, f"Failed to lookup staff by extension {extension}", str(e))
        raise HTTPException(status_code=500, detail=f"Database lookup failed: {str(e)}")


@router.get("/customer-support-staff")
async def get_customer_support_staff(request: Request):
    """
    Get all active Customer Support staff members from EWRCentral database.
    Returns list of staff in the Customer Support department/role.
    """
    user_ip = request.client.host if request.client else "Unknown"

    log_pipeline("AUDIO", user_ip, "Fetching Customer Support staff list")

    try:
        import pymssql

        # Connect to EWRSQLPROD with domain credentials
        connection = pymssql.connect(
            server='EWRSQLPROD',
            database='EWRCentral',
            user='EWR\\chad.walker',
            password='6454@@Christina',
            port='1433'
        )

        cursor = connection.cursor(as_dict=True)

        # Query to get all active users who could be Customer Support
        # Filter by role/department if there's a specific field, otherwise get all active users with phone extensions
        query = """
            SELECT
                cu.CentralUserID,
                cu.FirstName,
                cu.LastName,
                cu.FirstName + ' ' + cu.LastName AS FullName,
                cu.OfficeEmailAddress AS Email,
                LTRIM(RTRIM(cu.OfficePhoneExtension)) AS PhoneExtension,
                cu.Title AS JobTitle
            FROM CentralUsers cu
            WHERE cu.IsActive = 1
              AND cu.OfficePhoneExtension IS NOT NULL
              AND LTRIM(RTRIM(cu.OfficePhoneExtension)) <> ''
            ORDER BY cu.LastName, cu.FirstName
        """

        cursor.execute(query)
        results = cursor.fetchall()
        cursor.close()
        connection.close()

        staff_list = [
            {
                "id": r['CentralUserID'],
                "name": r['FullName'],
                "first_name": r['FirstName'],
                "last_name": r['LastName'],
                "email": r['Email'],
                "extension": r['PhoneExtension'],
                "title": r['JobTitle']
            }
            for r in results
        ]

        log_pipeline("AUDIO", user_ip, f"Found {len(staff_list)} Customer Support staff members")

        return {
            "success": True,
            "staff": staff_list,
            "count": len(staff_list)
        }

    except Exception as e:
        log_error("AUDIO", user_ip, "Failed to fetch Customer Support staff", str(e))
        raise HTTPException(status_code=500, detail=f"Database lookup failed: {str(e)}")


@router.post("/match-tickets")
async def find_matching_tickets(req: TicketMatchRequest, request: Request):
    """
    Find tickets in EWRCentral that might correspond to a phone call.

    Matches based on:
    - Staff member (extension -> AddCentralUserID)
    - Caller phone number (-> CustomerContactPhoneNumber)
    - Time window after call (-> AddTicketDate)
    - Subject keywords (-> TicketTitle)
    - Customer name (-> CustomerContactName)

    Returns ranked list of potential matching tickets.
    """
    user_ip = request.client.host if request.client else "Unknown"

    log_pipeline("AUDIO", user_ip, "Searching for matching tickets", {
        "extension": req.extension,
        "phone": req.phone_number,
        "datetime": req.call_datetime
    })

    try:
        import pymssql
        from datetime import timedelta
        import re

        connection = pymssql.connect(
            server='EWRSQLPROD',
            database='EWRCentral',
            user='EWR\\chad.walker',
            password='6454@@Christina',
            port='1433'
        )
        cursor = connection.cursor(as_dict=True)

        # Build dynamic query based on provided criteria
        conditions = []
        params = []

        # Get staff user ID from extension
        staff_user_id = None
        if req.extension:
            cursor.execute("""
                SELECT CentralUserID, FirstName, LastName
                FROM CentralUsers
                WHERE LTRIM(RTRIM(OfficePhoneExtension)) = %s AND IsActive = 1
            """, (req.extension,))
            staff = cursor.fetchone()
            if staff:
                staff_user_id = staff['CentralUserID']
                conditions.append("ct.AddCentralUserID = %s")
                params.append(staff_user_id)

        # Parse call datetime and create time window
        call_dt = None
        if req.call_datetime:
            try:
                # Try parsing ISO format
                call_dt = datetime.fromisoformat(req.call_datetime.replace('Z', '+00:00'))
            except:
                try:
                    # Try common formats
                    call_dt = datetime.strptime(req.call_datetime, "%Y%m%d-%H%M%S")
                except:
                    pass

        if call_dt:
            # Search from call time to call time + window
            end_dt = call_dt + timedelta(minutes=req.time_window_minutes)
            conditions.append("ct.AddTicketDate >= %s AND ct.AddTicketDate <= %s")
            params.extend([call_dt, end_dt])

        # Normalize phone number for matching (remove non-digits)
        if req.phone_number:
            phone_digits = re.sub(r'\D', '', req.phone_number)
            if len(phone_digits) >= 7:
                # Match last 7+ digits
                conditions.append("""
                    (REPLACE(REPLACE(REPLACE(REPLACE(ct.CustomerContactPhoneNumber, '(', ''), ')', ''), '-', ''), ' ', '')
                     LIKE %s)
                """)
                params.append(f'%{phone_digits[-10:]}')  # Last 10 digits

        # Build the main query
        where_clause = " AND ".join(conditions) if conditions else "1=1"

        query = f"""
            SELECT TOP 10
                ct.CentralTicketID,
                ct.TicketTitle,
                ct.Note,
                ct.AddTicketDate,
                ct.CustomerContactName,
                ct.CustomerContactPhoneNumber,
                cc.CompanyName,
                cu.FirstName AS UserFirstName,
                cu.FirstName + ' ' + cu.LastName AS CreatedBy,
                tt.Description AS TicketType,
                ts.Description AS TicketStatus
            FROM CentralTickets ct
            LEFT JOIN CentralCompanies cc ON ct.CentralCompanyID = cc.CentralCompanyID
            LEFT JOIN CentralUsers cu ON ct.AddCentralUserID = cu.CentralUserID
            LEFT JOIN Types tt ON ct.TicketTypeID = tt.TypeID
            LEFT JOIN Types ts ON ct.TicketStatusTypeID = ts.TypeID
            WHERE {where_clause}
            ORDER BY ct.AddTicketDate DESC
        """

        cursor.execute(query, tuple(params))
        tickets = cursor.fetchall()

        # Score and rank tickets based on additional criteria
        scored_tickets = []
        for ticket in tickets:
            score = 0
            match_reasons = []

            # Phone number match (strongest signal)
            if req.phone_number and ticket.get('CustomerContactPhoneNumber'):
                phone_digits = re.sub(r'\D', '', req.phone_number)
                ticket_phone = re.sub(r'\D', '', ticket['CustomerContactPhoneNumber'] or '')
                if phone_digits[-7:] == ticket_phone[-7:]:
                    score += 50
                    match_reasons.append("phone_number_match")

            # Staff match
            if staff_user_id:
                score += 20
                match_reasons.append("staff_match")

            # Time window match (closer = better)
            if call_dt and ticket.get('AddTicketDate'):
                ticket_dt = ticket['AddTicketDate']
                if hasattr(ticket_dt, 'replace'):
                    # Remove timezone for comparison
                    minutes_after = (ticket_dt.replace(tzinfo=None) - call_dt.replace(tzinfo=None)).total_seconds() / 60
                    if 0 <= minutes_after <= 30:
                        score += 30
                        match_reasons.append("created_within_30min")
                    elif 0 <= minutes_after <= 60:
                        score += 15
                        match_reasons.append("created_within_60min")

            # Subject keyword match
            if req.subject_keywords and ticket.get('TicketTitle'):
                title_lower = ticket['TicketTitle'].lower()
                for keyword in req.subject_keywords:
                    if keyword.lower() in title_lower:
                        score += 10
                        match_reasons.append(f"keyword_match:{keyword}")

            # Customer name match
            if req.customer_name and ticket.get('CustomerContactName'):
                if req.customer_name.lower() in ticket['CustomerContactName'].lower():
                    score += 25
                    match_reasons.append("customer_name_match")

            # Convert datetime for JSON serialization
            ticket_dict = dict(ticket)
            if ticket_dict.get('AddTicketDate'):
                ticket_dict['AddTicketDate'] = ticket_dict['AddTicketDate'].isoformat()

            scored_tickets.append({
                "ticket": ticket_dict,
                "match_score": score,
                "match_reasons": match_reasons
            })

        # Sort by score descending
        scored_tickets.sort(key=lambda x: x['match_score'], reverse=True)

        cursor.close()
        connection.close()

        # Determine best match
        best_match = None
        if scored_tickets and scored_tickets[0]['match_score'] >= 50:
            best_match = scored_tickets[0]

        return {
            "success": True,
            "search_criteria": {
                "extension": req.extension,
                "staff_user_id": staff_user_id,
                "phone_number": req.phone_number,
                "call_datetime": req.call_datetime,
                "time_window_minutes": req.time_window_minutes,
                "subject_keywords": req.subject_keywords,
                "customer_name": req.customer_name
            },
            "total_matches": len(scored_tickets),
            "best_match": best_match,
            "matches": scored_tickets
        }

    except Exception as e:
        log_error("AUDIO", user_ip, "Failed to search for matching tickets", str(e))
        raise HTTPException(status_code=500, detail=f"Ticket search failed: {str(e)}")


@router.get("/{analysis_id}")
async def get_audio_analysis(
    analysis_id: str,
    request: Request
):
    """
    Get audio analysis by ID.
    """
    user_ip = request.client.host if request.client else "Unknown"

    try:
        mongodb = get_mongodb_service()
        result = await mongodb.get_audio_analysis(analysis_id)

        if not result:
            raise HTTPException(status_code=404, detail="Audio analysis not found")

        log_pipeline("AUDIO", user_ip, "Retrieved audio analysis",
                     analysis_id)

        return result

    except HTTPException:
        raise
    except Exception as e:
        log_error("AUDIO", user_ip, "Failed to retrieve audio analysis", str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/{analysis_id}")
async def update_audio_analysis(
    analysis_id: str,
    request: Request
):
    """
    Update audio analysis metadata by ID.
    Allows updating customer_support_staff, ewr_customer, mood, and outcome fields.
    """
    user_ip = request.client.host if request.client else "Unknown"

    try:
        # Parse request body
        body = await request.json()

        mongodb = get_mongodb_service()

        # Build update dict with only allowed fields
        update_fields = {}
        allowed_fields = ['customer_support_staff', 'ewr_customer', 'mood', 'outcome', 'related_ticket_ids']

        for field in allowed_fields:
            if field in body:
                update_fields[field] = body[field]

        if not update_fields:
            raise HTTPException(status_code=400, detail="No valid fields to update")

        # Update the document
        result = await mongodb.update_audio_analysis(analysis_id, update_fields)

        if not result:
            raise HTTPException(status_code=404, detail="Audio analysis not found")

        log_pipeline("AUDIO", user_ip, "Updated audio analysis",
                     analysis_id)

        return {
            "success": True,
            "message": "Audio analysis updated successfully",
            "updated_fields": update_fields
        }

    except HTTPException:
        raise
    except Exception as e:
        log_error("AUDIO", user_ip, "Failed to update audio analysis", str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{analysis_id}")
async def delete_audio_analysis(
    analysis_id: str,
    request: Request
):
    """
    Delete audio analysis by ID.
    """
    user_ip = request.client.host if request.client else "Unknown"

    try:
        mongodb = get_mongodb_service()
        result = await mongodb.delete_audio_analysis(analysis_id)

        if not result.get("success"):
            raise HTTPException(status_code=404, detail="Audio analysis not found")

        log_pipeline("AUDIO", user_ip, "Deleted audio analysis",
                     analysis_id)

        return result

    except HTTPException:
        raise
    except Exception as e:
        log_error("AUDIO", user_ip, "Failed to delete audio analysis", str(e))
        raise HTTPException(status_code=500, detail=str(e))
