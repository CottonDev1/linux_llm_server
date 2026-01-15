"""
Audio Pipeline Test Fixtures
=============================

Shared fixtures and utilities for audio pipeline tests.

Provides:
- Mock audio service instances
- Sample audio data generators
- Audio file fixtures (WAV, MP3 simulation)
- Emotion/event tag test data
- Speaker diarization test data

Prefect Variables (configurable in Prefect UI):
    - audio_file_path: Path to audio file for analysis tests
    - audio_support_name: Customer support personnel name for records
    - audio_customer_name: Customer name for records
"""

import os
import sys
import pytest
import tempfile
import asyncio
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, MagicMock, AsyncMock, patch
from dataclasses import dataclass, field

# Add python_services to path for imports
PYTHON_SERVICES = Path(__file__).parent.parent.parent.parent / "python_services"
sys.path.insert(0, str(PYTHON_SERVICES))

# Try to import Prefect variables, fall back to defaults if not available
try:
    from prefect.variables import Variable
    PREFECT_AVAILABLE = True
except ImportError:
    PREFECT_AVAILABLE = False


def get_prefect_variable(name: str, default: str) -> str:
    """Get a Prefect variable value, falling back to default if not available."""
    if not PREFECT_AVAILABLE:
        return default
    try:
        value = Variable.get(name)
        return value if value is not None else default
    except Exception:
        return default


# Prefect variable defaults
PREFECT_DEFAULTS = {
    "audio_file_path": "/data/projects/llm_website/testing_data/audio/sample.wav",
    "audio_support_name": "John Smith",
    "audio_customer_name": "Jane Doe",
}


# =============================================================================
# Test Data Constants
# =============================================================================

# SenseVoice emotion tags
EMOTION_TAGS = ["HAPPY", "SAD", "ANGRY", "NEUTRAL", "FEARFUL", "DISGUSTED", "SURPRISED"]

# Audio event tags
EVENT_TAGS = ["Speech", "BGM", "Applause", "Laughter", "Cry", "Cough", "Sneeze", "Breath"]

# Sample transcription with tags
SAMPLE_RAW_TRANSCRIPTION = (
    "<|en|><|NEUTRAL|><|Speech|>Hello, this is customer support. "
    "<|HAPPY|>How can I help you today? "
    "<|ANGRY|>I've been waiting for 30 minutes! "
    "<|NEUTRAL|><|Speech|>I apologize for the wait. Let me look into that for you."
)

SAMPLE_CLEAN_TRANSCRIPTION = (
    "Hello, this is customer support. "
    "How can I help you today? "
    "I've been waiting for 30 minutes! "
    "I apologize for the wait. Let me look into that for you."
)


# =============================================================================
# Audio Data Fixtures
# =============================================================================


@dataclass
class MockAudioData:
    """Mock audio data for testing."""
    waveform: Any = None
    sample_rate: int = 16000
    duration_seconds: float = 30.0
    channels: int = 1
    format: str = "wav"

    def __post_init__(self):
        """Generate mock waveform if not provided."""
        if self.waveform is None:
            try:
                import numpy as np
                import torch
                # Generate simple sine wave
                samples = int(self.sample_rate * self.duration_seconds)
                t = np.linspace(0, self.duration_seconds, samples)
                audio = np.sin(2 * np.pi * 440 * t).astype(np.float32)
                self.waveform = torch.from_numpy(audio).unsqueeze(0)
            except ImportError:
                # Fallback to list if numpy/torch not available
                self.waveform = [[0.0] * int(self.sample_rate * self.duration_seconds)]


@pytest.fixture
def mock_audio_data() -> MockAudioData:
    """Fixture providing mock audio data for tests."""
    return MockAudioData()


@pytest.fixture
def mock_audio_data_long() -> MockAudioData:
    """Fixture providing long audio data (>25 seconds) for chunking tests."""
    return MockAudioData(duration_seconds=120.0)


@pytest.fixture
def mock_audio_data_short() -> MockAudioData:
    """Fixture providing short audio data (<25 seconds) for direct processing."""
    return MockAudioData(duration_seconds=10.0)


# =============================================================================
# Speaker Segment Fixtures
# =============================================================================


@dataclass
class MockSpeakerSegment:
    """Mock speaker segment for diarization tests."""
    speaker: str
    start_time: float
    end_time: float
    text: str = ""

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time


@pytest.fixture
def sample_speaker_segments() -> List[MockSpeakerSegment]:
    """Fixture providing sample speaker segments for diarization tests."""
    return [
        MockSpeakerSegment("Caller 1", 0.0, 5.5, "Hello, this is customer support."),
        MockSpeakerSegment("Caller 2", 5.5, 10.0, "Hi, I need help with my account."),
        MockSpeakerSegment("Caller 1", 10.0, 18.0, "Sure, I can help you with that. Let me look up your account."),
        MockSpeakerSegment("Caller 2", 18.0, 25.0, "Thank you, my account number is 12345."),
        MockSpeakerSegment("Caller 1", 25.0, 30.0, "I found your account. What seems to be the issue?"),
    ]


@pytest.fixture
def sample_speaker_segments_single() -> List[MockSpeakerSegment]:
    """Fixture providing single-speaker segments."""
    return [
        MockSpeakerSegment("Caller 1", 0.0, 30.0, "This is a monologue with only one speaker."),
    ]


@pytest.fixture
def sample_speaker_segments_multiple() -> List[MockSpeakerSegment]:
    """Fixture providing multi-speaker segments (3+ speakers)."""
    return [
        MockSpeakerSegment("Caller 1", 0.0, 5.0, "Welcome to the conference call."),
        MockSpeakerSegment("Caller 2", 5.0, 10.0, "Thanks for having us."),
        MockSpeakerSegment("Caller 3", 10.0, 15.0, "Happy to be here."),
        MockSpeakerSegment("Caller 1", 15.0, 20.0, "Let's begin the meeting."),
        MockSpeakerSegment("Caller 2", 20.0, 25.0, "I have a question."),
        MockSpeakerSegment("Caller 3", 25.0, 30.0, "Go ahead."),
    ]


# =============================================================================
# Transcription Fixtures
# =============================================================================


@dataclass
class MockTranscriptionResult:
    """Mock transcription result for testing."""
    raw_text: str
    audio_metadata: Dict[str, Any]
    chunk_count: int = 1


@pytest.fixture
def sample_transcription_result() -> MockTranscriptionResult:
    """Fixture providing sample transcription result."""
    return MockTranscriptionResult(
        raw_text=SAMPLE_RAW_TRANSCRIPTION,
        audio_metadata={
            "duration_seconds": 45.0,
            "sample_rate": 16000,
            "channels": 1,
            "format": "mp3",
            "file_size_bytes": 720000
        },
        chunk_count=2
    )


@pytest.fixture
def sample_transcription_result_short() -> MockTranscriptionResult:
    """Fixture providing short transcription result (no chunking)."""
    return MockTranscriptionResult(
        raw_text="<|en|><|NEUTRAL|><|Speech|>Short call about account question.",
        audio_metadata={
            "duration_seconds": 15.0,
            "sample_rate": 16000,
            "channels": 1,
            "format": "wav",
            "file_size_bytes": 480000
        },
        chunk_count=1
    )


# =============================================================================
# Emotion Parsing Fixtures
# =============================================================================


@pytest.fixture
def emotion_test_cases() -> List[Tuple[str, List[str]]]:
    """
    Fixture providing emotion tag test cases.

    Returns:
        List of (raw_text, expected_emotions) tuples
    """
    return [
        ("<|HAPPY|>I'm so excited!", ["HAPPY"]),
        ("<|SAD|>This is disappointing.", ["SAD"]),
        ("<|ANGRY|>This is unacceptable!", ["ANGRY"]),
        ("<|NEUTRAL|>Just a normal statement.", ["NEUTRAL"]),
        ("<|FEARFUL|>I'm worried about this.", ["FEARFUL"]),
        ("<|DISGUSTED|>That's gross.", ["DISGUSTED"]),
        ("<|SURPRISED|>Wow, I didn't expect that!", ["SURPRISED"]),
        ("<|HAPPY|><|SAD|>Mixed feelings today.", ["HAPPY", "SAD"]),
        ("<|EMO_UNKNOWN|>Unclear emotion.", ["NEUTRAL"]),  # Maps to NEUTRAL
        ("<|EMO_UNKOWN|>Typo in model output.", ["NEUTRAL"]),  # Typo also maps to NEUTRAL
        ("No emotion tags here.", []),
    ]


@pytest.fixture
def event_test_cases() -> List[Tuple[str, List[str]]]:
    """
    Fixture providing audio event tag test cases.

    Returns:
        List of (raw_text, expected_events) tuples
    """
    return [
        ("<|Speech|>Normal speaking.", ["Speech"]),
        ("<|BGM|>Music playing.", ["BGM"]),
        ("<|Applause|>Clapping.", ["Applause"]),
        ("<|Laughter|>Ha ha ha.", ["Laughter"]),
        ("<|Cry|>Sobbing.", ["Cry"]),
        ("<|Cough|>*cough*", ["Cough"]),
        ("<|Sneeze|>*achoo*", ["Sneeze"]),
        ("<|Breath|>*breathing*", ["Breath"]),
        ("<|Speech|><|Laughter|>Laughing while talking.", ["Speech", "Laughter"]),
        ("No event tags here.", []),
    ]


@pytest.fixture
def language_test_cases() -> List[Tuple[str, Optional[str]]]:
    """
    Fixture providing language tag test cases.

    Returns:
        List of (raw_text, expected_language) tuples
    """
    return [
        ("<|en|>English text.", "en"),
        ("<|zh|>Chinese text.", "zh"),
        ("<|ja|>Japanese text.", "ja"),
        ("<|ko|>Korean text.", "ko"),
        ("<|es|>Spanish text.", "es"),
        ("No language tag.", None),
    ]


# =============================================================================
# Mock Service Fixtures
# =============================================================================


@pytest.fixture
def mock_sensevoice_model():
    """Fixture providing mock SenseVoice model."""
    mock_model = MagicMock()

    def mock_generate(input, language="auto", use_itn=True, **kwargs):
        """Mock generate method returning realistic transcription result."""
        return [{
            "text": SAMPLE_RAW_TRANSCRIPTION
        }]

    mock_model.generate = mock_generate
    return mock_model


@pytest.fixture
def mock_pyannote_pipeline():
    """Fixture providing mock pyannote diarization pipeline."""
    mock_pipeline = MagicMock()

    class MockDiarization:
        """Mock diarization output."""
        def itertracks(self, yield_label=False):
            tracks = [
                (MagicMock(start=0.0, end=5.5), None, "SPEAKER_00"),
                (MagicMock(start=5.5, end=10.0), None, "SPEAKER_01"),
                (MagicMock(start=10.0, end=18.0), None, "SPEAKER_00"),
                (MagicMock(start=18.0, end=25.0), None, "SPEAKER_01"),
                (MagicMock(start=25.0, end=30.0), None, "SPEAKER_00"),
            ]
            for track in tracks:
                yield track

    mock_pipeline.return_value = MockDiarization()
    return mock_pipeline


@pytest.fixture
def mock_llm_response():
    """Fixture providing mock LLM response for call content analysis."""
    return {
        "subject": "Customer account access issue",
        "outcome": "Resolved",
        "customer_name": "John Smith"
    }


# =============================================================================
# File Path Fixtures
# =============================================================================


@pytest.fixture
def ringcentral_filename_samples() -> List[Tuple[str, Dict]]:
    """
    Fixture providing RingCentral filename parsing test cases.

    Returns:
        List of (filename, expected_metadata) tuples
    """
    return [
        # With extension (outgoing)
        (
            "20251104-133555_302_(252)792-8686_Outgoing_Auto_2243071124051.mp3",
            {
                "parsed": True,
                "call_date": "2025-11-04",
                "call_time": "13:35:55",
                "extension": "302",
                "phone_number": "(252)792-8686",
                "direction": "Outgoing",
                "auto_flag": "Auto",
                "recording_id": "2243071124051"
            }
        ),
        # Without extension (incoming)
        (
            "20251121-141152_(469)906-0558_Incoming_Auto_2254843027051.mp3",
            {
                "parsed": True,
                "call_date": "2025-11-21",
                "call_time": "14:11:52",
                "extension": None,
                "phone_number": "(469)906-0558",
                "direction": "Incoming",
                "auto_flag": "Auto",
                "recording_id": "2254843027051"
            }
        ),
        # Invalid filename
        (
            "random_audio_file.mp3",
            {
                "parsed": False,
                "call_date": None,
                "call_time": None,
                "extension": None,
                "phone_number": None,
                "direction": None,
                "auto_flag": None,
                "recording_id": None
            }
        ),
    ]


@pytest.fixture
def temp_audio_file(tmp_path) -> str:
    """
    Fixture providing a temporary audio file path.

    Creates a minimal WAV file for testing file operations.
    """
    audio_path = tmp_path / "test_audio.wav"

    # Create minimal WAV file
    try:
        import wave
        import struct

        with wave.open(str(audio_path), 'w') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(16000)

            # 1 second of silence
            for _ in range(16000):
                wav_file.writeframes(struct.pack('<h', 0))
    except Exception:
        # If wave module fails, create empty file
        audio_path.touch()

    return str(audio_path)


@pytest.fixture
def temp_mp3_file(tmp_path) -> str:
    """
    Fixture providing a temporary MP3 file path.

    Note: Creates an empty file since actual MP3 creation requires ffmpeg.
    Tests should mock the actual audio loading.
    """
    audio_path = tmp_path / "test_audio.mp3"
    audio_path.touch()
    return str(audio_path)


# =============================================================================
# Progress Callback Fixtures
# =============================================================================


@dataclass
class ProgressTracker:
    """Tracks progress callback invocations."""
    calls: List[Tuple[str, str]] = field(default_factory=list)

    async def callback(self, step: str, message: str):
        """Async callback that records invocations."""
        self.calls.append((step, message))

    def get_steps(self) -> List[str]:
        """Get list of step names."""
        return [step for step, _ in self.calls]

    def get_messages(self) -> List[str]:
        """Get list of messages."""
        return [msg for _, msg in self.calls]

    def has_step(self, step: str) -> bool:
        """Check if a specific step was called."""
        return step in self.get_steps()

    def reset(self):
        """Clear recorded calls."""
        self.calls = []


@pytest.fixture
def progress_tracker() -> ProgressTracker:
    """Fixture providing progress callback tracker."""
    return ProgressTracker()


# =============================================================================
# Analysis Result Fixtures
# =============================================================================


@pytest.fixture
def sample_analysis_result() -> Dict[str, Any]:
    """Fixture providing sample audio analysis result."""
    return {
        "success": True,
        "transcription": SAMPLE_CLEAN_TRANSCRIPTION,
        "transcription_summary": "Customer called about account access issue. Support resolved the problem.",
        "raw_transcription": SAMPLE_RAW_TRANSCRIPTION,
        "emotions": {
            "primary": "NEUTRAL",
            "detected": ["NEUTRAL", "HAPPY", "ANGRY"],
            "timestamps": []
        },
        "audio_events": {
            "detected": ["Speech"],
            "timestamps": []
        },
        "language": "en",
        "audio_metadata": {
            "duration_seconds": 45.0,
            "sample_rate": 16000,
            "channels": 1,
            "format": "mp3",
            "file_size_bytes": 720000
        },
        "call_metadata": {
            "parsed": True,
            "call_date": "2025-11-04",
            "call_time": "13:35:55",
            "extension": "302",
            "phone_number": "(252)792-8686",
            "direction": "Outgoing",
            "auto_flag": "Auto",
            "recording_id": "2243071124051"
        },
        "call_content": {
            "subject": "Account access issue",
            "outcome": "Resolved",
            "customer_name": "John Smith",
            "confidence": 0.85,
            "analysis_model": "Llama-3.2-3B-Instruct"
        },
        "customer_lookup": {"found": False}
    }


@pytest.fixture
def sample_error_result() -> Dict[str, Any]:
    """Fixture providing sample error analysis result."""
    return {
        "success": False,
        "error": "Audio file corrupted or unsupported format",
        "transcription": "",
        "transcription_summary": None,
        "raw_transcription": "",
        "emotions": {"primary": "NEUTRAL", "detected": [], "timestamps": []},
        "audio_events": {"detected": [], "timestamps": []},
        "language": "auto",
        "audio_metadata": {
            "duration_seconds": 0.0,
            "sample_rate": 16000,
            "channels": 1,
            "format": "mp3",
            "file_size_bytes": 0
        },
        "call_metadata": {"parsed": False},
        "call_content": {
            "subject": None,
            "outcome": None,
            "customer_name": None,
            "confidence": 0.0,
            "analysis_model": ""
        },
        "customer_lookup": {"found": False}
    }


# =============================================================================
# Prefect Variable-Based Fixtures
# =============================================================================


@dataclass
class AudioTestConfig:
    """Audio test configuration from Prefect variables."""
    file_path: str
    support_name: str
    customer_name: str

    def get_analysis_request(self, include_transcription: bool = True) -> Dict[str, Any]:
        """Get an audio analysis request dictionary."""
        return {
            "filePath": self.file_path,
            "metadata": {
                "supportPersonnel": self.support_name,
                "customerName": self.customer_name,
            },
            "options": {
                "includeTranscription": include_transcription,
                "includeDiarization": True,
                "includeEmotionAnalysis": True,
                "includeSentiment": True,
            }
        }


@pytest.fixture(scope="session")
def audio_test_config() -> AudioTestConfig:
    """
    Get audio test configuration from Prefect variables.

    These values can be changed in the Prefect UI under Variables:
    - audio_file_path: Path to the audio file to analyze
    - audio_support_name: Support personnel name
    - audio_customer_name: Customer name

    Falls back to default test values if Prefect is not available.
    """
    return AudioTestConfig(
        file_path=get_prefect_variable("audio_file_path", PREFECT_DEFAULTS["audio_file_path"]),
        support_name=get_prefect_variable("audio_support_name", PREFECT_DEFAULTS["audio_support_name"]),
        customer_name=get_prefect_variable("audio_customer_name", PREFECT_DEFAULTS["audio_customer_name"]),
    )


@pytest.fixture
def audio_file_path(audio_test_config: AudioTestConfig) -> str:
    """Get audio file path from Prefect variables."""
    return audio_test_config.file_path


@pytest.fixture
def audio_support_name(audio_test_config: AudioTestConfig) -> str:
    """Get support personnel name from Prefect variables."""
    return audio_test_config.support_name


@pytest.fixture
def audio_customer_name(audio_test_config: AudioTestConfig) -> str:
    """Get customer name from Prefect variables."""
    return audio_test_config.customer_name


@pytest.fixture
def valid_audio_analysis_request(audio_test_config: AudioTestConfig) -> Dict[str, Any]:
    """Standard valid audio analysis request using Prefect variables."""
    return audio_test_config.get_analysis_request()


@pytest.fixture
def prefect_transcription_request(audio_test_config: AudioTestConfig) -> Dict[str, Any]:
    """Transcription-only request using Prefect variables."""
    return {
        "filePath": audio_test_config.file_path,
        "metadata": {
            "supportPersonnel": audio_test_config.support_name,
            "customerName": audio_test_config.customer_name,
        },
        "options": {
            "includeTranscription": True,
            "includeDiarization": False,
            "includeEmotionAnalysis": False,
        }
    }


@pytest.fixture
def prefect_diarization_request(audio_test_config: AudioTestConfig) -> Dict[str, Any]:
    """Diarization request using Prefect variables."""
    return {
        "filePath": audio_test_config.file_path,
        "metadata": {
            "supportPersonnel": audio_test_config.support_name,
            "customerName": audio_test_config.customer_name,
        },
        "options": {
            "includeDiarization": True,
            "speakerLabels": ["Support", "Customer"],
        }
    }
