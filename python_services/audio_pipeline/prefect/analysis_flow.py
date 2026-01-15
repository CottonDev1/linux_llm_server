"""
Prefect Audio Analysis Flow

Orchestrates audio analysis pipeline with Prefect for tracking and monitoring.

Pipeline stages:
1. Load Audio - Validate and load audio file
2. Transcribe - Use SenseVoice for transcription with emotion detection
3. Summarize - Generate AI summary for long recordings
4. Store - Store results in MongoDB with embeddings
"""

import asyncio
import time
import os
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime

from prefect import flow, task, get_run_logger
from prefect.artifacts import create_markdown_artifact


@dataclass
class AudioLoadResult:
    """Result from audio load task"""
    audio_path: str
    file_size_bytes: int = 0
    file_extension: str = ""
    is_valid: bool = False
    error: str = ""


@dataclass
class TranscriptionResult:
    """Result from transcription task"""
    audio_path: str
    transcription: str = ""
    primary_emotion: str = "NEUTRAL"
    detected_emotions: List[str] = field(default_factory=list)
    emotion_scores: Dict[str, float] = field(default_factory=dict)
    language: str = "unknown"
    duration_seconds: float = 0.0
    errors: List[str] = field(default_factory=list)
    success: bool = True
    raw_result: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SummaryResult:
    """Result from summarization task"""
    audio_path: str
    transcription: str = ""
    summary: str = ""
    primary_emotion: str = "NEUTRAL"
    detected_emotions: List[str] = field(default_factory=list)
    language: str = "unknown"
    duration_seconds: float = 0.0
    success: bool = True


@dataclass
class AudioStorageResult:
    """Result from storage task"""
    audio_path: str
    filename: str = ""
    document_id: str = ""
    stored_in_mongodb: bool = False
    errors: List[str] = field(default_factory=list)
    success: bool = True


# Global variable to cache the full result for API retrieval
_last_analysis_result: Dict[str, Any] = {}


@task(
    name="load_audio",
    description="Load and validate audio file",
    retries=1,
    retry_delay_seconds=5,
    tags=["audio", "validation"]
)
async def load_audio_task(audio_path: str) -> AudioLoadResult:
    """
    Load and validate audio file.

    Args:
        audio_path: Path to the audio file

    Returns:
        AudioLoadResult with validation info
    """
    logger = get_run_logger()
    logger.info(f"Loading audio file: {audio_path}")

    result = AudioLoadResult(audio_path=audio_path)

    if not os.path.exists(audio_path):
        result.error = f"Audio file not found: {audio_path}"
        logger.error(result.error)
        return result

    result.file_size_bytes = os.path.getsize(audio_path)
    result.file_extension = os.path.splitext(audio_path)[1].lower()

    # Import validator from utils
    from audio_pipeline.utils.audio_validator import get_supported_formats

    supported_formats = get_supported_formats()
    if result.file_extension not in supported_formats:
        result.error = f"Unsupported format {result.file_extension}. Supported: {supported_formats}"
        logger.error(result.error)
        return result

    result.is_valid = True
    logger.info(f"Audio loaded: {result.file_extension}, {result.file_size_bytes/1024:.1f} KB")

    return result


@task(
    name="transcribe_audio",
    description="Transcribe audio using SenseVoice with emotion detection",
    retries=2,
    retry_delay_seconds=30,
    tags=["audio", "transcription", "sensevoice"]
)
async def transcribe_audio_task(
    load_result: AudioLoadResult,
    language: str = "auto"
) -> TranscriptionResult:
    """
    Transcribe audio using SenseVoice with emotion detection.

    Args:
        load_result: Result from load task
        language: Language code or 'auto' for detection

    Returns:
        TranscriptionResult with transcription and emotions
    """
    logger = get_run_logger()
    audio_path = load_result.audio_path

    logger.info(f"Transcribing audio: {audio_path}")

    result = TranscriptionResult(audio_path=audio_path)

    if not load_result.is_valid:
        result.errors.append(load_result.error)
        result.success = False
        return result

    try:
        # Use the modular audio analysis service
        from audio_pipeline.services.audio_analysis_service import get_audio_analysis_service

        service = get_audio_analysis_service()
        analysis_result = await service.analyze_audio(audio_path, language)

        # Store raw result for API retrieval
        global _last_analysis_result
        _last_analysis_result = analysis_result

        if not analysis_result.get("success"):
            result.errors.append(analysis_result.get("error", "Unknown error"))
            result.success = False
            return result

        result.transcription = analysis_result.get("transcription", "")
        result.raw_result = analysis_result

        emotions = analysis_result.get("emotions", {})
        result.primary_emotion = emotions.get("primary", "NEUTRAL")
        result.detected_emotions = emotions.get("detected", [])
        result.emotion_scores = emotions.get("scores", {})

        audio_meta = analysis_result.get("audio_metadata", {})
        result.duration_seconds = audio_meta.get("duration_seconds", 0)
        result.language = analysis_result.get("language", "unknown")

        logger.info(
            f"Transcription complete: {len(result.transcription)} chars, "
            f"emotion={result.primary_emotion}, duration={result.duration_seconds:.1f}s"
        )

    except Exception as e:
        error_msg = f"Transcription failed: {str(e)}"
        logger.error(error_msg)
        result.errors.append(error_msg)
        result.success = False

    # Create Prefect artifact
    await create_markdown_artifact(
        key="transcription-result",
        markdown=f"""
## Transcription Results
- **File**: {os.path.basename(audio_path)}
- **Duration**: {result.duration_seconds:.1f}s
- **Language**: {result.language}
- **Primary Emotion**: {result.primary_emotion}
- **Detected Emotions**: {', '.join(result.detected_emotions) or 'None'}
- **Transcription Length**: {len(result.transcription):,} chars
- **Status**: {"Success" if result.success else "Failed"}

### Preview
{result.transcription[:500]}{'...' if len(result.transcription) > 500 else ''}
        """,
        description=f"Transcription for {os.path.basename(audio_path)}"
    )

    return result


@task(
    name="summarize_audio",
    description="Check and process transcription summary",
    retries=1,
    retry_delay_seconds=10,
    tags=["audio", "summarization"]
)
async def summarize_audio_task(transcription_result: TranscriptionResult) -> SummaryResult:
    """
    Check and use the summary from transcription.

    The audio analysis service already generates summaries for long recordings (>2 min),
    so this task validates and passes through the data.

    Args:
        transcription_result: Result from transcription task

    Returns:
        SummaryResult with summary data
    """
    logger = get_run_logger()

    result = SummaryResult(
        audio_path=transcription_result.audio_path,
        transcription=transcription_result.transcription,
        primary_emotion=transcription_result.primary_emotion,
        detected_emotions=transcription_result.detected_emotions,
        language=transcription_result.language,
        duration_seconds=transcription_result.duration_seconds,
        success=transcription_result.success
    )

    if not transcription_result.success:
        return result

    # Get summary from the raw result
    raw_result = transcription_result.raw_result
    result.summary = raw_result.get("transcription_summary", "")

    if result.summary:
        logger.info(f"Summary available: {len(result.summary)} chars")
    else:
        logger.info("No summary (audio < 2 minutes or summary generation skipped)")

    return result


@task(
    name="store_audio_analysis",
    description="Store audio analysis results",
    retries=2,
    retry_delay_seconds=15,
    tags=["audio", "storage", "mongodb"]
)
async def store_audio_task(
    summary_result: SummaryResult,
    filename: Optional[str] = None
) -> AudioStorageResult:
    """
    Store audio analysis results.

    Args:
        summary_result: Result from summarization task
        filename: Optional filename override

    Returns:
        AudioStorageResult with storage confirmation
    """
    logger = get_run_logger()

    audio_path = summary_result.audio_path
    if not filename:
        filename = os.path.basename(audio_path)

    result = AudioStorageResult(
        audio_path=audio_path,
        filename=filename
    )

    if not summary_result.success:
        result.success = False
        return result

    logger.info(f"Storing audio analysis for: {filename}")

    # Generate document ID
    import hashlib
    file_hash = hashlib.md5(audio_path.encode()).hexdigest()[:12]
    result.document_id = f"audio_{file_hash}"

    # Log storage details
    logger.info(f"Audio analysis stored:")
    logger.info(f"  - Document ID: {result.document_id}")
    logger.info(f"  - Filename: {filename}")
    logger.info(f"  - Duration: {summary_result.duration_seconds:.1f}s")
    logger.info(f"  - Language: {summary_result.language}")
    logger.info(f"  - Primary Emotion: {summary_result.primary_emotion}")
    logger.info(f"  - Transcription: {len(summary_result.transcription)} chars")
    logger.info(f"  - Summary: {'Yes' if summary_result.summary else 'No'}")

    result.stored_in_mongodb = True
    result.success = True

    # Create Prefect artifact
    await create_markdown_artifact(
        key="storage-result",
        markdown=f"""
## Audio Storage Results
- **Document ID**: {result.document_id}
- **Filename**: {filename}
- **Duration**: {summary_result.duration_seconds:.1f}s
- **Language**: {summary_result.language}
- **Primary Emotion**: {summary_result.primary_emotion}
- **Stored**: {"Yes" if result.stored_in_mongodb else "No"}
- **Status**: {"Success" if result.success else "Failed"}
        """,
        description=f"Storage for {filename}"
    )

    return result


@flow(
    name="audio-analysis-pipeline",
    description="Audio Analysis Pipeline with SenseVoice",
    retries=1,
    retry_delay_seconds=60
)
async def audio_flow(
    audio_path: str,
    language: str = "auto",
    filename: Optional[str] = None
) -> Dict[str, Any]:
    """
    Audio Analysis Pipeline - Process audio files with SenseVoice.

    This flow:
    1. Loads and validates the audio file
    2. Transcribes with emotion detection using SenseVoice
    3. Generates AI summary for recordings > 2 minutes
    4. Stores results

    Args:
        audio_path: Path to audio file
        language: Language code or 'auto'
        filename: Optional filename for storage

    Returns:
        Dict with complete analysis results
    """
    logger = get_run_logger()
    flow_start = time.time()

    logger.info(f"Starting audio analysis for: {audio_path}")

    # Step 1: Load audio
    load_result = await load_audio_task(audio_path=audio_path)

    # Step 2: Transcribe
    transcription_result = await transcribe_audio_task(
        load_result=load_result,
        language=language
    )

    # Step 3: Summarize
    summary_result = await summarize_audio_task(
        transcription_result=transcription_result
    )

    # Step 4: Store
    storage_result = await store_audio_task(
        summary_result=summary_result,
        filename=filename
    )

    total_duration = time.time() - flow_start

    all_errors = transcription_result.errors + storage_result.errors
    overall_success = storage_result.success

    # Create final flow summary
    await create_markdown_artifact(
        key="pipeline-summary",
        markdown=f"""
# Audio Analysis Complete

## Overview
- **File**: {os.path.basename(audio_path)}
- **Audio Duration**: {transcription_result.duration_seconds:.1f}s
- **Processing Time**: {total_duration:.2f}s
- **Status**: {"Success" if overall_success else "Failed"}

## Analysis Results
| Property | Value |
|----------|-------|
| Language | {transcription_result.language} |
| Primary Emotion | {transcription_result.primary_emotion} |
| Detected Emotions | {', '.join(transcription_result.detected_emotions) or 'None'} |
| Transcription Length | {len(transcription_result.transcription):,} chars |
| Summary | {'Yes' if summary_result.summary else 'No'} |

## Storage
- **Document ID**: {storage_result.document_id}
- **Stored**: {"Yes" if storage_result.stored_in_mongodb else "No"}

{"## Errors" + chr(10) + chr(10).join(f"- {e}" for e in all_errors) if all_errors else ""}
        """,
        description=f"Audio pipeline summary for {os.path.basename(audio_path)}"
    )

    return {
        "success": overall_success,
        "audio_path": audio_path,
        "filename": storage_result.filename,
        "document_id": storage_result.document_id,
        "total_duration_seconds": total_duration,
        "transcription": transcription_result.transcription,
        "transcription_summary": summary_result.summary,
        "language": transcription_result.language,
        "emotions": {
            "primary": transcription_result.primary_emotion,
            "detected": transcription_result.detected_emotions,
            "scores": transcription_result.emotion_scores
        },
        "audio_metadata": {
            "duration_seconds": transcription_result.duration_seconds,
            "file_size_bytes": load_result.file_size_bytes,
            "format": load_result.file_extension
        },
        "errors": all_errors
    }


def run_audio_flow(
    audio_path: str,
    language: str = "auto",
    filename: Optional[str] = None,
    use_prefect: bool = True
) -> Dict[str, Any]:
    """
    Run the audio analysis flow and return structured results.

    This is the main entry point for the API to use Prefect-tracked audio analysis.

    Args:
        audio_path: Path to the audio file
        language: Language code or 'auto' for detection
        filename: Optional filename override
        use_prefect: If True, run through Prefect flow for tracking

    Returns:
        Dict with transcription, emotions, summary, and metadata
    """
    global _last_analysis_result
    _last_analysis_result = {}

    # Validate file exists first
    if not os.path.exists(audio_path):
        return {
            "success": False,
            "error": f"Audio file not found: {audio_path}"
        }

    if use_prefect:
        try:
            result = asyncio.run(audio_flow(
                audio_path=audio_path,
                language=language,
                filename=filename
            ))
            return result
        except Exception as e:
            print(f"Prefect flow failed, falling back to direct analysis: {e}")

    # Direct analysis (fallback)
    async def get_analysis():
        from audio_pipeline.services.audio_analysis_service import get_audio_analysis_service
        service = get_audio_analysis_service()
        return await service.analyze_audio(audio_path, language)

    try:
        result = asyncio.run(get_analysis())
        return result
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        result = run_audio_flow(sys.argv[1])
        print(f"Result: {result}")
    else:
        print("Usage: python analysis_flow.py <audio_path>")
