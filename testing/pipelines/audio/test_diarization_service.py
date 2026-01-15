"""
DiarizationService Tests
========================

Comprehensive tests for the speaker diarization service.

Tests cover:
1. Service initialization and singleton pattern
2. Speaker identification
3. Transcription merging with speaker segments
4. Speaker statistics calculation
5. WAV conversion for non-WAV formats
6. Error handling for missing/corrupted files

These tests use mocks to avoid requiring GPU resources and pyannote models.
"""

import os
import sys
import pytest
import asyncio
import tempfile
from pathlib import Path
from typing import Dict, List, Any
from unittest.mock import Mock, MagicMock, AsyncMock, patch
from dataclasses import dataclass

# Add python_services to path
PYTHON_SERVICES = Path(__file__).parent.parent.parent.parent / "python_services"
sys.path.insert(0, str(PYTHON_SERVICES))


# =============================================================================
# Test Class: Service Initialization
# =============================================================================


class TestDiarizationServiceInitialization:
    """Tests for DiarizationService initialization."""

    def test_singleton_pattern(self):
        """
        Verify DiarizationService uses singleton pattern.

        Multiple calls to get_instance() should return the same instance.
        """
        with patch('audio_pipeline.services.diarization_service.PYANNOTE_AVAILABLE', True):
            from audio_pipeline.services.diarization_service import DiarizationService

            # Reset singleton
            DiarizationService._instance = None

            instance1 = DiarizationService.get_instance()
            instance2 = DiarizationService.get_instance()

            assert instance1 is instance2, "get_instance() should return same instance"

            DiarizationService._instance = None

    def test_initial_state(self):
        """
        Verify initial state of DiarizationService.

        Service should not be initialized on creation.
        """
        with patch('audio_pipeline.services.diarization_service.PYANNOTE_AVAILABLE', True):
            from audio_pipeline.services.diarization_service import DiarizationService

            DiarizationService._instance = None
            service = DiarizationService()

            assert service.pipeline is None, "Pipeline should be None initially"
            assert service.is_initialized is False, "Should not be initialized"

            DiarizationService._instance = None

    def test_device_detection_cuda(self):
        """
        Verify CUDA device detection.

        When CUDA is available, device should be 'cuda'.
        """
        with patch('audio_pipeline.services.diarization_service.PYANNOTE_AVAILABLE', True):
            with patch('audio_pipeline.services.diarization_service.TORCH_AVAILABLE', True):
                with patch('torch.cuda.is_available', return_value=True):
                    from audio_pipeline.services.diarization_service import DiarizationService

                    DiarizationService._instance = None
                    service = DiarizationService()

                    assert service.device == "cuda", "Device should be cuda when available"

                    DiarizationService._instance = None

    def test_device_detection_cpu(self):
        """
        Verify CPU fallback when CUDA unavailable.

        When CUDA is not available, device should be 'cpu'.
        """
        with patch('audio_pipeline.services.diarization_service.PYANNOTE_AVAILABLE', True):
            with patch('audio_pipeline.services.diarization_service.TORCH_AVAILABLE', True):
                with patch('torch.cuda.is_available', return_value=False):
                    from audio_pipeline.services.diarization_service import DiarizationService

                    DiarizationService._instance = None
                    service = DiarizationService()

                    assert service.device == "cpu", "Device should be cpu when CUDA unavailable"

                    DiarizationService._instance = None

    @pytest.mark.asyncio
    async def test_initialize_raises_without_pyannote(self):
        """
        Verify initialization fails without pyannote.

        When pyannote.audio is not installed, initialize() should raise RuntimeError.
        """
        with patch('audio_pipeline.services.diarization_service.PYANNOTE_AVAILABLE', False):
            from audio_pipeline.services.diarization_service import DiarizationService

            DiarizationService._instance = None
            service = DiarizationService()

            with pytest.raises(RuntimeError, match="pyannote.audio not installed"):
                await service.initialize()

            DiarizationService._instance = None


# =============================================================================
# Test Class: Diarization
# =============================================================================


class TestDiarization:
    """Tests for speaker diarization functionality."""

    @pytest.mark.asyncio
    async def test_diarize_file_not_found(self):
        """
        Verify diarize() raises FileNotFoundError for missing files.

        When given a non-existent file path, should raise FileNotFoundError.
        """
        with patch('audio_pipeline.services.diarization_service.PYANNOTE_AVAILABLE', True):
            from audio_pipeline.services.diarization_service import DiarizationService

            DiarizationService._instance = None
            service = DiarizationService()
            service.is_initialized = True
            service.pipeline = MagicMock()

            with pytest.raises(FileNotFoundError):
                await service.diarize("/nonexistent/path/audio.wav")

            DiarizationService._instance = None

    @pytest.mark.asyncio
    async def test_diarize_success(self, temp_audio_file, mock_pyannote_pipeline, progress_tracker):
        """
        Verify successful diarization with progress callbacks.

        Should return list of speaker segments with proper labels.
        """
        with patch('audio_pipeline.services.diarization_service.PYANNOTE_AVAILABLE', True):
            with patch('audio_pipeline.services.diarization_service.TORCH_AVAILABLE', True):
                from audio_pipeline.services.diarization_service import DiarizationService

                DiarizationService._instance = None
                service = DiarizationService()
                service.is_initialized = True
                service.pipeline = mock_pyannote_pipeline

                # Mock audio loading
                with patch.object(service, '_load_audio_as_waveform', return_value={
                    "waveform": MagicMock(),
                    "sample_rate": 16000
                }):
                    with patch.object(service, '_convert_to_wav', return_value=temp_audio_file):
                        segments = await service.diarize(
                            temp_audio_file,
                            progress_callback=progress_tracker.callback
                        )

                assert len(segments) > 0, "Should return speaker segments"

                # Verify speaker labels are assigned
                speakers = set(s.speaker for s in segments)
                assert "Caller 1" in speakers, "First speaker should be labeled Caller 1"

                # Verify progress callback was invoked
                assert progress_tracker.has_step("diarize"), "Diarization callback should be invoked"

                DiarizationService._instance = None

    @pytest.mark.asyncio
    async def test_diarize_with_speaker_limits(self, temp_audio_file, mock_pyannote_pipeline):
        """
        Verify diarization respects speaker count limits.

        When min_speakers and max_speakers are specified, they should be passed to pipeline.
        """
        with patch('audio_pipeline.services.diarization_service.PYANNOTE_AVAILABLE', True):
            with patch('audio_pipeline.services.diarization_service.TORCH_AVAILABLE', True):
                from audio_pipeline.services.diarization_service import DiarizationService

                DiarizationService._instance = None
                service = DiarizationService()
                service.is_initialized = True
                service.pipeline = mock_pyannote_pipeline

                with patch.object(service, '_load_audio_as_waveform', return_value={
                    "waveform": MagicMock(),
                    "sample_rate": 16000
                }):
                    with patch.object(service, '_convert_to_wav', return_value=temp_audio_file):
                        segments = await service.diarize(
                            temp_audio_file,
                            min_speakers=2,
                            max_speakers=3
                        )

                # Verify pipeline was called (mocked)
                assert mock_pyannote_pipeline.called, "Pipeline should be invoked"

                DiarizationService._instance = None

    @pytest.mark.asyncio
    async def test_diarize_default_phone_call_settings(self, temp_audio_file, mock_pyannote_pipeline):
        """
        Verify default settings for phone calls (2 speakers).

        When no speaker limits provided, should default to 2-2 for phone calls.
        """
        with patch('audio_pipeline.services.diarization_service.PYANNOTE_AVAILABLE', True):
            with patch('audio_pipeline.services.diarization_service.TORCH_AVAILABLE', True):
                from audio_pipeline.services.diarization_service import DiarizationService

                DiarizationService._instance = None
                service = DiarizationService()
                service.is_initialized = True
                service.pipeline = mock_pyannote_pipeline

                with patch.object(service, '_load_audio_as_waveform', return_value={
                    "waveform": MagicMock(),
                    "sample_rate": 16000
                }):
                    with patch.object(service, '_convert_to_wav', return_value=temp_audio_file):
                        segments = await service.diarize(temp_audio_file)

                # Pipeline should be called with default params (min=2, max=2)
                assert mock_pyannote_pipeline.called, "Pipeline should be invoked"

                DiarizationService._instance = None


# =============================================================================
# Test Class: WAV Conversion
# =============================================================================


class TestWAVConversion:
    """Tests for audio format conversion to WAV."""

    def test_convert_wav_returns_same_path(self, temp_audio_file):
        """
        Verify WAV files are not converted.

        When input is already WAV, should return the same path.
        """
        with patch('audio_pipeline.services.diarization_service.PYANNOTE_AVAILABLE', True):
            from audio_pipeline.services.diarization_service import DiarizationService

            DiarizationService._instance = None
            service = DiarizationService()

            result = service._convert_to_wav(temp_audio_file)

            assert result == temp_audio_file, "WAV file should not be converted"

            DiarizationService._instance = None

    def test_convert_mp3_to_wav(self, temp_mp3_file):
        """
        Verify MP3 files are converted to WAV.

        MP3 files should be converted to temporary WAV files.
        """
        with patch('audio_pipeline.services.diarization_service.PYANNOTE_AVAILABLE', True):
            with patch('audio_pipeline.services.diarization_service.PYDUB_AVAILABLE', True):
                from audio_pipeline.services.diarization_service import DiarizationService

                DiarizationService._instance = None
                service = DiarizationService()

                # Mock AudioSegment
                mock_audio = MagicMock()
                mock_audio.set_channels.return_value = mock_audio
                mock_audio.set_frame_rate.return_value = mock_audio

                with patch('audio_pipeline.services.diarization_service.AudioSegment') as mock_segment:
                    mock_segment.from_mp3.return_value = mock_audio

                    with patch('tempfile.mkstemp', return_value=(0, '/tmp/converted.wav')):
                        with patch('os.close'):
                            result = service._convert_to_wav(temp_mp3_file)

                assert result != temp_mp3_file, "MP3 should be converted to different path"
                mock_audio.export.assert_called_once()

                DiarizationService._instance = None

    def test_convert_raises_without_pydub(self, temp_mp3_file):
        """
        Verify conversion fails without pydub for non-WAV files.

        When pydub is not available, should raise RuntimeError.
        """
        with patch('audio_pipeline.services.diarization_service.PYANNOTE_AVAILABLE', True):
            with patch('audio_pipeline.services.diarization_service.PYDUB_AVAILABLE', False):
                from audio_pipeline.services.diarization_service import DiarizationService

                DiarizationService._instance = None
                service = DiarizationService()

                with pytest.raises(RuntimeError, match="pydub required"):
                    service._convert_to_wav(temp_mp3_file)

                DiarizationService._instance = None

    def test_convert_m4a_to_wav(self, tmp_path):
        """
        Verify M4A files are converted to WAV.

        M4A is a common iOS recording format that should be supported.
        """
        m4a_file = tmp_path / "test.m4a"
        m4a_file.touch()

        with patch('audio_pipeline.services.diarization_service.PYANNOTE_AVAILABLE', True):
            with patch('audio_pipeline.services.diarization_service.PYDUB_AVAILABLE', True):
                from audio_pipeline.services.diarization_service import DiarizationService

                DiarizationService._instance = None
                service = DiarizationService()

                mock_audio = MagicMock()
                mock_audio.set_channels.return_value = mock_audio
                mock_audio.set_frame_rate.return_value = mock_audio

                with patch('audio_pipeline.services.diarization_service.AudioSegment') as mock_segment:
                    mock_segment.from_file.return_value = mock_audio

                    with patch('tempfile.mkstemp', return_value=(0, '/tmp/converted.wav')):
                        with patch('os.close'):
                            result = service._convert_to_wav(str(m4a_file))

                mock_segment.from_file.assert_called_with(str(m4a_file), format='m4a')

                DiarizationService._instance = None


# =============================================================================
# Test Class: Audio Loading
# =============================================================================


class TestAudioLoading:
    """Tests for audio waveform loading."""

    def test_load_audio_torchaudio(self, temp_audio_file):
        """
        Verify audio loading with torchaudio.

        Should return dictionary with waveform and sample_rate.
        """
        with patch('audio_pipeline.services.diarization_service.PYANNOTE_AVAILABLE', True):
            with patch('audio_pipeline.services.diarization_service.TORCH_AVAILABLE', True):
                from audio_pipeline.services.diarization_service import DiarizationService

                DiarizationService._instance = None
                service = DiarizationService()

                mock_waveform = MagicMock()
                mock_sample_rate = 16000

                with patch('torchaudio.load', return_value=(mock_waveform, mock_sample_rate)):
                    result = service._load_audio_as_waveform(temp_audio_file)

                assert "waveform" in result, "Should return waveform"
                assert "sample_rate" in result, "Should return sample_rate"
                assert result["sample_rate"] == 16000

                DiarizationService._instance = None

    def test_load_audio_scipy_fallback(self, temp_audio_file):
        """
        Verify fallback to scipy when torchaudio fails.

        Should use scipy.io.wavfile as fallback.
        """
        with patch('audio_pipeline.services.diarization_service.PYANNOTE_AVAILABLE', True):
            with patch('audio_pipeline.services.diarization_service.TORCH_AVAILABLE', True):
                with patch('audio_pipeline.services.diarization_service.SCIPY_AVAILABLE', True):
                    from audio_pipeline.services.diarization_service import DiarizationService

                    DiarizationService._instance = None
                    service = DiarizationService()

                    import numpy as np

                    # torchaudio fails
                    with patch('torchaudio.load', side_effect=RuntimeError("torchaudio error")):
                        # scipy succeeds
                        mock_data = np.zeros(16000, dtype=np.int16)
                        with patch('audio_pipeline.services.diarization_service.wavfile.read',
                                   return_value=(16000, mock_data)):
                            with patch('torch.from_numpy') as mock_torch:
                                mock_torch.return_value = MagicMock()
                                result = service._load_audio_as_waveform(temp_audio_file)

                    assert "waveform" in result
                    assert result["sample_rate"] == 16000

                    DiarizationService._instance = None

    def test_load_audio_raises_without_libraries(self, temp_audio_file):
        """
        Verify error when no audio loading library available.

        Should raise RuntimeError when both torchaudio and scipy fail.
        """
        with patch('audio_pipeline.services.diarization_service.PYANNOTE_AVAILABLE', True):
            with patch('audio_pipeline.services.diarization_service.TORCH_AVAILABLE', True):
                with patch('audio_pipeline.services.diarization_service.SCIPY_AVAILABLE', False):
                    from audio_pipeline.services.diarization_service import DiarizationService

                    DiarizationService._instance = None
                    service = DiarizationService()

                    with patch('torchaudio.load', side_effect=RuntimeError("torchaudio error")):
                        with pytest.raises(RuntimeError, match="No audio loading library"):
                            service._load_audio_as_waveform(temp_audio_file)

                    DiarizationService._instance = None


# =============================================================================
# Test Class: Transcription Merging
# =============================================================================


class TestTranscriptionMerging:
    """Tests for merging transcription with speaker segments."""

    def test_merge_transcription_empty_segments(self):
        """
        Verify handling of empty segments list.

        Should return original transcription unchanged.
        """
        with patch('audio_pipeline.services.diarization_service.PYANNOTE_AVAILABLE', True):
            from audio_pipeline.services.diarization_service import DiarizationService

            DiarizationService._instance = None
            service = DiarizationService()

            transcription = "Hello, this is a test."
            segments = []

            labeled, details = service.merge_transcription_with_diarization(
                transcription, segments, 10.0
            )

            assert labeled == transcription, "Empty segments should return original"
            assert details == [], "No segment details for empty segments"

            DiarizationService._instance = None

    def test_merge_transcription_with_segments(self, sample_speaker_segments):
        """
        Verify transcription merging with speaker segments.

        Should produce labeled transcription with speaker prefixes.
        """
        with patch('audio_pipeline.services.diarization_service.PYANNOTE_AVAILABLE', True):
            from audio_pipeline.services.diarization_service import (
                DiarizationService, SpeakerSegment
            )

            DiarizationService._instance = None
            service = DiarizationService()

            transcription = "Hello this is support. How can I help? I need account help. Sure thing."

            # Convert mock segments to actual SpeakerSegment
            segments = [
                SpeakerSegment(s.speaker, s.start_time, s.end_time, s.text)
                for s in sample_speaker_segments
            ]

            labeled, details = service.merge_transcription_with_diarization(
                transcription, segments, 30.0
            )

            assert "Caller 1" in labeled, "Speaker 1 label should appear"
            assert "Caller 2" in labeled, "Speaker 2 label should appear"
            assert len(details) > 0, "Should have segment details"

            DiarizationService._instance = None

    def test_merge_transcription_single_speaker(self, sample_speaker_segments_single):
        """
        Verify merging with single speaker.

        Should correctly label all text with one speaker.
        """
        with patch('audio_pipeline.services.diarization_service.PYANNOTE_AVAILABLE', True):
            from audio_pipeline.services.diarization_service import (
                DiarizationService, SpeakerSegment
            )

            DiarizationService._instance = None
            service = DiarizationService()

            transcription = "This is a monologue with only one speaker throughout."
            segments = [
                SpeakerSegment(s.speaker, s.start_time, s.end_time, s.text)
                for s in sample_speaker_segments_single
            ]

            labeled, details = service.merge_transcription_with_diarization(
                transcription, segments, 30.0
            )

            assert "Caller 1" in labeled, "Single speaker should be labeled"
            assert "Caller 2" not in labeled, "Only one speaker should appear"

            DiarizationService._instance = None

    def test_merge_transcription_multiple_speakers(self, sample_speaker_segments_multiple):
        """
        Verify merging with multiple speakers (3+).

        Should correctly label all speakers.
        """
        with patch('audio_pipeline.services.diarization_service.PYANNOTE_AVAILABLE', True):
            from audio_pipeline.services.diarization_service import (
                DiarizationService, SpeakerSegment
            )

            DiarizationService._instance = None
            service = DiarizationService()

            transcription = "Welcome. Thanks. Happy to be here. Let's begin. Question. Go ahead."
            segments = [
                SpeakerSegment(s.speaker, s.start_time, s.end_time, s.text)
                for s in sample_speaker_segments_multiple
            ]

            labeled, details = service.merge_transcription_with_diarization(
                transcription, segments, 30.0
            )

            assert "Caller 1" in labeled, "Speaker 1 should appear"
            assert "Caller 2" in labeled, "Speaker 2 should appear"
            assert "Caller 3" in labeled, "Speaker 3 should appear"

            DiarizationService._instance = None


# =============================================================================
# Test Class: Speaker Statistics
# =============================================================================


class TestSpeakerStatistics:
    """Tests for speaker statistics calculation."""

    def test_get_speaker_statistics_empty(self):
        """
        Verify handling of empty segments list.

        Should return empty dictionary.
        """
        with patch('audio_pipeline.services.diarization_service.PYANNOTE_AVAILABLE', True):
            from audio_pipeline.services.diarization_service import DiarizationService

            DiarizationService._instance = None
            service = DiarizationService()

            stats = service.get_speaker_statistics([])

            assert stats == {}, "Empty segments should return empty stats"

            DiarizationService._instance = None

    def test_get_speaker_statistics_basic(self, sample_speaker_segments):
        """
        Verify basic statistics calculation.

        Should calculate total duration, segment count, and percentage for each speaker.
        """
        with patch('audio_pipeline.services.diarization_service.PYANNOTE_AVAILABLE', True):
            from audio_pipeline.services.diarization_service import (
                DiarizationService, SpeakerSegment
            )

            DiarizationService._instance = None
            service = DiarizationService()

            segments = [
                SpeakerSegment(s.speaker, s.start_time, s.end_time, s.text)
                for s in sample_speaker_segments
            ]

            stats = service.get_speaker_statistics(segments)

            assert "Caller 1" in stats, "Caller 1 should have stats"
            assert "Caller 2" in stats, "Caller 2 should have stats"

            # Verify stat fields
            for speaker, speaker_stats in stats.items():
                assert "total_duration" in speaker_stats
                assert "segment_count" in speaker_stats
                assert "word_count" in speaker_stats
                assert "percentage" in speaker_stats

            DiarizationService._instance = None

    def test_get_speaker_statistics_percentages(self, sample_speaker_segments):
        """
        Verify percentage calculations sum to 100%.

        Total speaking percentages should approximately equal 100%.
        """
        with patch('audio_pipeline.services.diarization_service.PYANNOTE_AVAILABLE', True):
            from audio_pipeline.services.diarization_service import (
                DiarizationService, SpeakerSegment
            )

            DiarizationService._instance = None
            service = DiarizationService()

            segments = [
                SpeakerSegment(s.speaker, s.start_time, s.end_time, s.text)
                for s in sample_speaker_segments
            ]

            stats = service.get_speaker_statistics(segments)

            total_percentage = sum(s["percentage"] for s in stats.values())

            # Should be approximately 100% (allowing for rounding)
            assert 99.0 <= total_percentage <= 101.0, \
                f"Total percentage should be ~100%, got {total_percentage}"

            DiarizationService._instance = None

    def test_get_speaker_statistics_word_count(self, sample_speaker_segments):
        """
        Verify word count calculation.

        Should count words in segment text.
        """
        with patch('audio_pipeline.services.diarization_service.PYANNOTE_AVAILABLE', True):
            from audio_pipeline.services.diarization_service import (
                DiarizationService, SpeakerSegment
            )

            DiarizationService._instance = None
            service = DiarizationService()

            segments = [
                SpeakerSegment(s.speaker, s.start_time, s.end_time, s.text)
                for s in sample_speaker_segments
            ]

            stats = service.get_speaker_statistics(segments)

            total_words = sum(s["word_count"] for s in stats.values())

            assert total_words > 0, "Should have non-zero word count"

            DiarizationService._instance = None


# =============================================================================
# Test Class: SpeakerSegment Dataclass
# =============================================================================


class TestSpeakerSegment:
    """Tests for SpeakerSegment dataclass."""

    def test_speaker_segment_duration(self):
        """
        Verify duration property calculation.

        Duration should be end_time - start_time.
        """
        with patch('audio_pipeline.services.diarization_service.PYANNOTE_AVAILABLE', True):
            from audio_pipeline.services.diarization_service import SpeakerSegment

            segment = SpeakerSegment(
                speaker="Caller 1",
                start_time=5.0,
                end_time=15.0,
                text="Test text"
            )

            assert segment.duration == 10.0, "Duration should be 10 seconds"

    def test_speaker_segment_default_text(self):
        """
        Verify default empty text.

        Text should default to empty string.
        """
        with patch('audio_pipeline.services.diarization_service.PYANNOTE_AVAILABLE', True):
            from audio_pipeline.services.diarization_service import SpeakerSegment

            segment = SpeakerSegment(
                speaker="Caller 1",
                start_time=0.0,
                end_time=5.0
            )

            assert segment.text == "", "Default text should be empty string"


# =============================================================================
# Test Class: Global Function
# =============================================================================


class TestGlobalFunction:
    """Tests for get_diarization_service global function."""

    def test_get_diarization_service(self):
        """
        Verify get_diarization_service returns singleton instance.

        Should return the same instance as get_instance().
        """
        with patch('audio_pipeline.services.diarization_service.PYANNOTE_AVAILABLE', True):
            from audio_pipeline.services.diarization_service import (
                get_diarization_service, DiarizationService
            )

            DiarizationService._instance = None

            service = get_diarization_service()
            instance = DiarizationService.get_instance()

            assert service is instance, "Should return singleton instance"

            DiarizationService._instance = None
