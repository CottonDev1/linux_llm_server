"""
Audio Analysis Pipeline

A modular pipeline for audio transcription, emotion detection, and analysis.
Follows the same structure as sql_pipeline for consistency.

Components:
- models/: Pydantic models for requests/responses
- services/: Business logic (transcription, emotion, summarization, etc.)
- prefect/: Prefect workflow definitions
- utils/: Utility functions (validation, chunking, format conversion)

Usage:
    from audio_pipeline import get_audio_analysis_service

    service = get_audio_analysis_service()
    result = await service.analyze_audio(audio_path, language="auto")
"""

# Services - main entry points
from audio_pipeline.services.audio_analysis_service import (
    AudioAnalysisService,
    get_audio_analysis_service,
    get_audio_service,  # Backwards compatibility
)
from audio_pipeline.services.transcription_service import (
    TranscriptionService,
    get_transcription_service,
)
from audio_pipeline.services.emotion_service import (
    EmotionService,
    get_emotion_service,
)
from audio_pipeline.services.summarization_service import (
    SummarizationService,
    get_summarization_service,
)
from audio_pipeline.services.metadata_service import (
    MetadataService,
    get_metadata_service,
    parse_call_filename,
)

# Models
from audio_pipeline.models.analysis_models import (
    AudioAnalysisRequest,
    AudioAnalysisResponse,
    AudioMetadata,
    EmotionResult,
    AudioEventResult,
)
from audio_pipeline.models.metadata_models import (
    CallMetadata,
    CallContentAnalysis,
    CustomerLookup,
)
from audio_pipeline.models.storage_models import (
    AudioStoreRequest,
    AudioSearchRequest,
    AudioAnalysisDocument,
)
from audio_pipeline.models.bulk_models import (
    BulkScanRequest,
    BulkScanResponse,
    BulkProcessingStatus,
    BulkAudioRequest,
    BulkAudioStatusResponse,
)

# Utils
from audio_pipeline.utils.audio_validator import (
    validate_audio_file,
    get_supported_formats,
)
from audio_pipeline.utils.format_converter import (
    convert_to_wav,
)

__all__ = [
    # Services
    "AudioAnalysisService",
    "get_audio_analysis_service",
    "get_audio_service",
    "TranscriptionService",
    "get_transcription_service",
    "EmotionService",
    "get_emotion_service",
    "SummarizationService",
    "get_summarization_service",
    "MetadataService",
    "get_metadata_service",
    "parse_call_filename",
    # Analysis models
    "AudioAnalysisRequest",
    "AudioAnalysisResponse",
    "AudioMetadata",
    "EmotionResult",
    "AudioEventResult",
    # Metadata models
    "CallMetadata",
    "CallContentAnalysis",
    "CustomerLookup",
    # Storage models
    "AudioStoreRequest",
    "AudioSearchRequest",
    "AudioAnalysisDocument",
    # Bulk models
    "BulkScanRequest",
    "BulkScanResponse",
    "BulkProcessingStatus",
    "BulkAudioRequest",
    "BulkAudioStatusResponse",
    # Utils
    "validate_audio_file",
    "get_supported_formats",
    "convert_to_wav",
]
