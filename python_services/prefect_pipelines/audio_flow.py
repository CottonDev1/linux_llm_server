"""
Prefect Audio Analysis Pipeline

A thin wrapper around the actual AudioAnalysisService that provides Prefect tracking,
artifacts, and observability without duplicating any pipeline logic.

IMPORTANT: This flow uses the ACTUAL audio_pipeline module.
All audio analysis logic lives in audio_pipeline/ - this file only provides:
1. Prefect flow/task decorators for tracking
2. Artifact creation for the Prefect UI
3. Timing metrics collection

This ensures that testing the Prefect flow tests the real production code.
"""

import asyncio
import time
import os
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime

from prefect import flow, task, get_run_logger
from prefect.artifacts import create_markdown_artifact

# Add ffmpeg to PATH for audio processing (required for pydub)
_ffmpeg_path = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "tools", "ffmpeg", "bin"
)
if os.path.exists(_ffmpeg_path) and _ffmpeg_path not in os.environ.get("PATH", ""):
    os.environ["PATH"] = _ffmpeg_path + os.pathsep + os.environ.get("PATH", "")


# =============================================================================
# Data Classes for Results
# =============================================================================

@dataclass
class AudioPipelineResult:
    """Result from the actual audio pipeline."""
    success: bool = False
    audio_path: str = ""
    filename: str = ""
    transcription: str = ""
    summary: str = ""
    primary_emotion: str = "NEUTRAL"
    detected_emotions: List[str] = field(default_factory=list)
    emotion_scores: Dict[str, float] = field(default_factory=dict)
    language: str = "unknown"
    duration_seconds: float = 0.0
    error: Optional[str] = None
    raw_result: Dict[str, Any] = field(default_factory=dict)
    processing_time_ms: float = 0.0


# =============================================================================
# Helper Functions
# =============================================================================

def sanitize_artifact_key(text: str) -> str:
    """Sanitize text for use in artifact keys."""
    import re
    return re.sub(r'[^a-zA-Z0-9_-]', '-', text.lower())[:50]


# =============================================================================
# Main Pipeline Task - Calls the ACTUAL Audio Pipeline
# =============================================================================

@task(
    name="execute_audio_pipeline",
    description="Execute the actual audio analysis pipeline (not a copy)",
    retries=1,
    retry_delay_seconds=30,
    tags=["audio", "pipeline", "production"]
)
async def execute_audio_pipeline_task(
    audio_path: str,
    language: str = "auto",
    filename: Optional[str] = None,
) -> AudioPipelineResult:
    """
    Execute the actual audio pipeline - NOT a reimplementation.

    This task is a thin wrapper that calls AudioAnalysisService.analyze_audio().
    All audio analysis logic lives in audio_pipeline/.
    """
    logger = get_run_logger()
    start_time = time.time()

    logger.info(f"Calling actual AudioAnalysisService.analyze_audio()")
    logger.info(f"Audio path: {audio_path}, Language: {language}")

    result = AudioPipelineResult(
        audio_path=audio_path,
        filename=filename or os.path.basename(audio_path)
    )

    try:
        # Import and use the ACTUAL pipeline
        from audio_pipeline import get_audio_analysis_service, validate_audio_file

        # Validate audio file first
        validation = validate_audio_file(audio_path)
        if not validation.get("valid", False):
            result.success = False
            result.error = validation.get("error", "Invalid audio file")
            logger.error(f"Validation failed: {result.error}")
            return result

        # Get the service and run analysis
        service = get_audio_analysis_service()
        analysis_result = await service.analyze_audio(audio_path, language)

        # Store raw result
        result.raw_result = analysis_result

        if not analysis_result.get("success", False):
            result.success = False
            result.error = analysis_result.get("error", "Analysis failed")
            logger.error(f"Analysis failed: {result.error}")
            return result

        # Map results
        result.success = True
        result.transcription = analysis_result.get("transcription", "")
        result.summary = analysis_result.get("transcription_summary", "")

        # Emotions
        emotions = analysis_result.get("emotions", {})
        result.primary_emotion = emotions.get("primary", "NEUTRAL")
        result.detected_emotions = emotions.get("detected", [])
        result.emotion_scores = emotions.get("scores", {})

        # Metadata
        audio_meta = analysis_result.get("audio_metadata", {})
        result.duration_seconds = audio_meta.get("duration_seconds", 0)
        result.language = analysis_result.get("language", "unknown")

        logger.info(
            f"Pipeline completed: {len(result.transcription)} chars, "
            f"emotion={result.primary_emotion}, duration={result.duration_seconds:.1f}s"
        )

    except Exception as e:
        result.success = False
        result.error = str(e)
        logger.error(f"Pipeline error: {e}")

    result.processing_time_ms = (time.time() - start_time) * 1000

    return result


# =============================================================================
# Main Flow
# =============================================================================

@flow(
    name="audio-analysis-pipeline",
    description="Audio Analysis Pipeline - Wrapper around actual AudioAnalysisService for Prefect tracking",
    retries=1,
    retry_delay_seconds=60
)
async def audio_flow(
    audio_path: str,
    language: str = "auto",
    filename: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Audio Analysis Pipeline Flow - Tests the ACTUAL production pipeline.

    This flow is a thin wrapper that:
    1. Calls the real AudioAnalysisService.analyze_audio() method
    2. Creates Prefect artifacts for observability
    3. Returns results in a consistent format

    All audio analysis logic lives in audio_pipeline/.
    This ensures that Prefect tests verify the actual production code.

    Args:
        audio_path: Path to audio file
        language: Language code or "auto" for detection
        filename: Optional filename override

    Returns:
        Dict with transcription, emotions, summary, and metadata
    """
    logger = get_run_logger()
    flow_start = time.time()

    logger.info(f"Starting Audio Analysis Pipeline (using actual AudioAnalysisService)")
    logger.info(f"Audio: {audio_path}")

    # Execute the actual pipeline
    result = await execute_audio_pipeline_task(
        audio_path=audio_path,
        language=language,
        filename=filename
    )

    total_time_ms = (time.time() - flow_start) * 1000

    # Create artifact for Prefect UI
    artifact_content = f"""
## Audio Analysis Pipeline Result

**File**: {result.filename}
**Path**: {result.audio_path}
**Success**: {result.success}
**Processing Time**: {total_time_ms:.1f}ms

### Audio Metadata
- **Duration**: {result.duration_seconds:.1f}s
- **Language**: {result.language}

### Emotions
- **Primary**: {result.primary_emotion}
- **Detected**: {', '.join(result.detected_emotions) or 'None'}

### Transcription ({len(result.transcription):,} chars)
```
{result.transcription[:500]}{'...' if len(result.transcription) > 500 else ''}
```

{f"### Summary\\n{result.summary}" if result.summary else ""}

{f"### Error\\n{result.error}" if result.error else ""}
"""

    await create_markdown_artifact(
        key=f"audio-pipeline-{sanitize_artifact_key(result.filename)[:20]}-{int(time.time())}",
        markdown=artifact_content,
        description=f"Audio analysis for: {result.filename}"
    )

    # Return result dict
    return {
        "success": result.success,
        "audio_path": result.audio_path,
        "filename": result.filename,
        "transcription": result.transcription,
        "summary": result.summary,
        "primary_emotion": result.primary_emotion,
        "detected_emotions": result.detected_emotions,
        "emotion_scores": result.emotion_scores,
        "language": result.language,
        "duration_seconds": result.duration_seconds,
        "error": result.error,
        "timing": {
            "total_ms": total_time_ms,
            "pipeline_ms": result.processing_time_ms,
        },
        "processing_time_ms": total_time_ms,
    }


# Backwards compatibility alias
def run_audio_flow(audio_path: str, language: str = "auto", filename: Optional[str] = None) -> Dict[str, Any]:
    """Synchronous wrapper for backwards compatibility."""
    return asyncio.run(audio_flow(audio_path, language, filename))


# =============================================================================
# CLI Entry Point
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run Audio Analysis Pipeline")
    parser.add_argument("--audio", required=True, help="Path to audio file")
    parser.add_argument("--language", default="auto", help="Language code or 'auto'")
    parser.add_argument("--filename", help="Override filename")

    args = parser.parse_args()

    result = asyncio.run(audio_flow(
        audio_path=args.audio,
        language=args.language,
        filename=args.filename
    ))

    print(f"\nResult: {result}")
