"""
TranscriptionService Tests
==========================

Comprehensive tests for the audio transcription service using SenseVoice.

Tests cover:
1. Service initialization and singleton pattern
2. Audio transcription (transcribe method)
3. Chunking logic with different audio lengths
4. Parallel chunk processing
5. MP3/WAV loading
6. Metadata extraction
7. Error handling

These tests use mocks to avoid requiring GPU resources and SenseVoice models.
"""

import os
import sys
import pytest
import asyncio
import tempfile
from pathlib import Path
from typing import Dict, List, Any, Tuple
from unittest.mock import Mock, MagicMock, AsyncMock, patch
from dataclasses import dataclass

# Add python_services to path
PYTHON_SERVICES = Path(__file__).parent.parent.parent.parent / "python_services"
sys.path.insert(0, str(PYTHON_SERVICES))


# =============================================================================
# Test Class: Service Initialization
# =============================================================================


class TestTranscriptionServiceInitialization:
    """Tests for TranscriptionService initialization."""

    def test_singleton_pattern(self):
        """
        Verify TranscriptionService uses singleton pattern.

        Multiple calls to get_instance() should return the same instance.
        """
        with patch('audio_pipeline.services.transcription_service.FUNASR_AVAILABLE', True):
            from audio_pipeline.services.transcription_service import TranscriptionService

            TranscriptionService._instance = None

            instance1 = TranscriptionService.get_instance()
            instance2 = TranscriptionService.get_instance()

            assert instance1 is instance2, "get_instance() should return same instance"

            TranscriptionService._instance = None

    def test_initial_state(self):
        """
        Verify initial state of TranscriptionService.

        Service should not be initialized on creation.
        """
        with patch('audio_pipeline.services.transcription_service.FUNASR_AVAILABLE', True):
            from audio_pipeline.services.transcription_service import TranscriptionService

            TranscriptionService._instance = None
            service = TranscriptionService()

            assert service.model is None, "Model should be None initially"
            assert service.is_initialized is False, "Should not be initialized"
            assert service.vad_available is False, "VAD should not be available initially"

            TranscriptionService._instance = None

    def test_device_detection_cuda(self):
        """
        Verify CUDA device detection.

        When CUDA is available, device should be 'cuda:0'.
        """
        with patch('audio_pipeline.services.transcription_service.FUNASR_AVAILABLE', True):
            with patch('torch.cuda.is_available', return_value=True):
                from audio_pipeline.services.transcription_service import TranscriptionService

                TranscriptionService._instance = None
                service = TranscriptionService()

                assert service.device == "cuda:0", "Device should be cuda:0 when available"

                TranscriptionService._instance = None

    def test_device_detection_cpu(self):
        """
        Verify CPU fallback when CUDA unavailable.

        When CUDA is not available, device should be 'cpu'.
        """
        with patch('audio_pipeline.services.transcription_service.FUNASR_AVAILABLE', True):
            with patch('torch.cuda.is_available', return_value=False):
                from audio_pipeline.services.transcription_service import TranscriptionService

                TranscriptionService._instance = None
                service = TranscriptionService()

                assert service.device == "cpu", "Device should be cpu when CUDA unavailable"

                TranscriptionService._instance = None

    def test_windows_detection(self):
        """
        Verify Windows platform detection.

        On Windows, is_windows should be True and VAD unavailable.
        """
        with patch('audio_pipeline.services.transcription_service.FUNASR_AVAILABLE', True):
            with patch('sys.platform', 'win32'):
                from audio_pipeline.services.transcription_service import TranscriptionService

                TranscriptionService._instance = None
                service = TranscriptionService()

                assert service.is_windows is True, "Should detect Windows platform"

                TranscriptionService._instance = None

    def test_parallel_settings(self):
        """
        Verify parallel processing settings.

        Should have parallel_enabled and max_parallel_chunks configured.
        """
        with patch('audio_pipeline.services.transcription_service.FUNASR_AVAILABLE', True):
            from audio_pipeline.services.transcription_service import TranscriptionService

            TranscriptionService._instance = None
            service = TranscriptionService()

            assert hasattr(service, 'parallel_enabled'), "Should have parallel_enabled setting"
            assert hasattr(service, 'max_parallel_chunks'), "Should have max_parallel_chunks setting"
            assert service.max_parallel_chunks >= 1, "Should allow at least 1 parallel chunk"

            TranscriptionService._instance = None

    @pytest.mark.asyncio
    async def test_initialize_raises_without_funasr(self):
        """
        Verify initialization fails without funasr.

        When funasr is not installed, initialize() should raise RuntimeError.
        """
        with patch('audio_pipeline.services.transcription_service.FUNASR_AVAILABLE', False):
            from audio_pipeline.services.transcription_service import TranscriptionService

            TranscriptionService._instance = None
            service = TranscriptionService()

            with pytest.raises(RuntimeError, match="funasr"):
                await service.initialize()

            TranscriptionService._instance = None

    @pytest.mark.asyncio
    async def test_initialize_raises_without_model_path(self):
        """
        Verify initialization fails without SENSEVOICE_MODEL_PATH.

        When environment variable is not set, should raise RuntimeError.
        """
        with patch('audio_pipeline.services.transcription_service.FUNASR_AVAILABLE', True):
            with patch.dict(os.environ, {}, clear=True):
                from audio_pipeline.services.transcription_service import TranscriptionService

                TranscriptionService._instance = None
                service = TranscriptionService()

                with pytest.raises(RuntimeError, match="SENSEVOICE_MODEL_PATH"):
                    await service.initialize()

                TranscriptionService._instance = None


# =============================================================================
# Test Class: Transcription
# =============================================================================


class TestTranscription:
    """Tests for the transcribe method."""

    @pytest.mark.asyncio
    async def test_transcribe_file_not_found(self):
        """
        Verify transcribe() raises FileNotFoundError for missing files.

        When given a non-existent file path, should raise FileNotFoundError.
        """
        with patch('audio_pipeline.services.transcription_service.FUNASR_AVAILABLE', True):
            from audio_pipeline.services.transcription_service import TranscriptionService

            TranscriptionService._instance = None
            service = TranscriptionService()
            service.is_initialized = True
            service.model = MagicMock()

            with pytest.raises(FileNotFoundError):
                await service.transcribe("/nonexistent/path/audio.wav")

            TranscriptionService._instance = None

    @pytest.mark.asyncio
    async def test_transcribe_short_audio_success(self, temp_audio_file, mock_sensevoice_model):
        """
        Verify successful transcription of short audio.

        Short audio (<25s) should be processed directly without chunking.
        """
        with patch('audio_pipeline.services.transcription_service.FUNASR_AVAILABLE', True):
            from audio_pipeline.services.transcription_service import TranscriptionService

            TranscriptionService._instance = None
            service = TranscriptionService()
            service.is_initialized = True
            service.model = mock_sensevoice_model
            service.vad_available = False

            with patch.object(service, 'get_audio_metadata', return_value={
                "duration_seconds": 10.0,
                "sample_rate": 16000,
                "channels": 1,
                "format": "wav",
                "file_size_bytes": 320000
            }):
                result = await service.transcribe(temp_audio_file)

            assert "raw_text" in result, "Should return raw_text"
            assert "audio_metadata" in result, "Should return audio_metadata"
            assert result["chunk_count"] == 1, "Short audio should have 1 chunk"

            TranscriptionService._instance = None

    @pytest.mark.asyncio
    async def test_transcribe_with_language(self, temp_audio_file, mock_sensevoice_model):
        """
        Verify transcription with specified language.

        Language parameter should be passed to model.
        """
        with patch('audio_pipeline.services.transcription_service.FUNASR_AVAILABLE', True):
            from audio_pipeline.services.transcription_service import TranscriptionService

            TranscriptionService._instance = None
            service = TranscriptionService()
            service.is_initialized = True
            service.model = mock_sensevoice_model
            service.vad_available = False

            with patch.object(service, 'get_audio_metadata', return_value={
                "duration_seconds": 10.0,
                "sample_rate": 16000,
                "channels": 1,
                "format": "wav",
                "file_size_bytes": 320000
            }):
                result = await service.transcribe(temp_audio_file, language="en")

            # Verify model was called with language parameter
            mock_sensevoice_model.generate.assert_called()
            call_kwargs = mock_sensevoice_model.generate.call_args
            assert call_kwargs[1]['language'] == "en" or 'en' in str(call_kwargs)

            TranscriptionService._instance = None

    @pytest.mark.asyncio
    async def test_transcribe_empty_result_raises(self, temp_audio_file):
        """
        Verify handling of empty transcription result.

        When model returns empty result, should raise ValueError.
        """
        with patch('audio_pipeline.services.transcription_service.FUNASR_AVAILABLE', True):
            from audio_pipeline.services.transcription_service import TranscriptionService

            TranscriptionService._instance = None
            service = TranscriptionService()
            service.is_initialized = True
            service.model = MagicMock()
            service.model.generate.return_value = []  # Empty result
            service.vad_available = False

            with patch.object(service, 'get_audio_metadata', return_value={
                "duration_seconds": 10.0,
                "sample_rate": 16000,
                "channels": 1,
                "format": "wav",
                "file_size_bytes": 320000
            }):
                with pytest.raises(ValueError, match="No transcription result"):
                    await service.transcribe(temp_audio_file)

            TranscriptionService._instance = None


# =============================================================================
# Test Class: Audio Metadata Extraction
# =============================================================================


class TestAudioMetadata:
    """Tests for audio metadata extraction."""

    def test_get_audio_metadata_wav(self, temp_audio_file):
        """
        Verify metadata extraction for WAV files.

        Should extract duration, sample rate, channels, and format.
        """
        with patch('audio_pipeline.services.transcription_service.FUNASR_AVAILABLE', True):
            with patch('audio_pipeline.services.transcription_service.TORCHAUDIO_AVAILABLE', True):
                from audio_pipeline.services.transcription_service import TranscriptionService

                TranscriptionService._instance = None
                service = TranscriptionService()

                mock_info = MagicMock()
                mock_info.num_frames = 48000
                mock_info.sample_rate = 16000
                mock_info.num_channels = 1

                with patch('torchaudio.info', return_value=mock_info):
                    metadata = service.get_audio_metadata(temp_audio_file)

                assert metadata["duration_seconds"] == 3.0
                assert metadata["sample_rate"] == 16000
                assert metadata["channels"] == 1
                assert metadata["format"] == "wav"

                TranscriptionService._instance = None

    def test_get_audio_metadata_mp3_mutagen(self, temp_mp3_file):
        """
        Verify metadata extraction for MP3 files using mutagen.

        Mutagen should be preferred for MP3 files.
        """
        with patch('audio_pipeline.services.transcription_service.FUNASR_AVAILABLE', True):
            with patch('audio_pipeline.services.transcription_service.MUTAGEN_AVAILABLE', True):
                from audio_pipeline.services.transcription_service import TranscriptionService

                TranscriptionService._instance = None
                service = TranscriptionService()

                mock_mp3 = MagicMock()
                mock_mp3.info.length = 120.5
                mock_mp3.info.sample_rate = 44100
                mock_mp3.info.channels = 2

                with patch('audio_pipeline.services.transcription_service.MP3', return_value=mock_mp3):
                    metadata = service.get_audio_metadata(temp_mp3_file)

                assert metadata["duration_seconds"] == 120.5
                assert metadata["sample_rate"] == 44100
                assert metadata["channels"] == 2
                assert metadata["format"] == "mp3"

                TranscriptionService._instance = None

    def test_get_audio_metadata_file_size(self, temp_audio_file):
        """
        Verify file size is included in metadata.

        Should return file size in bytes.
        """
        with patch('audio_pipeline.services.transcription_service.FUNASR_AVAILABLE', True):
            with patch('audio_pipeline.services.transcription_service.TORCHAUDIO_AVAILABLE', True):
                from audio_pipeline.services.transcription_service import TranscriptionService

                TranscriptionService._instance = None
                service = TranscriptionService()

                mock_info = MagicMock()
                mock_info.num_frames = 16000
                mock_info.sample_rate = 16000
                mock_info.num_channels = 1

                with patch('torchaudio.info', return_value=mock_info):
                    metadata = service.get_audio_metadata(temp_audio_file)

                assert "file_size_bytes" in metadata
                assert metadata["file_size_bytes"] >= 0

                TranscriptionService._instance = None


# =============================================================================
# Test Class: Audio Loading
# =============================================================================


class TestAudioLoading:
    """Tests for audio waveform loading."""

    def test_load_mp3_with_pydub(self, temp_mp3_file):
        """
        Verify MP3 loading with pydub.

        MP3 files should be loaded using pydub for better Windows compatibility.
        """
        with patch('audio_pipeline.services.transcription_service.FUNASR_AVAILABLE', True):
            with patch('audio_pipeline.services.transcription_service.PYDUB_AVAILABLE', True):
                from audio_pipeline.services.transcription_service import TranscriptionService

                TranscriptionService._instance = None
                service = TranscriptionService()

                mock_audio = MagicMock()
                mock_audio.channels = 1
                mock_audio.frame_rate = 16000
                mock_audio.get_array_of_samples.return_value = [0] * 16000
                mock_audio.set_channels.return_value = mock_audio
                mock_audio.set_frame_rate.return_value = mock_audio

                with patch('audio_pipeline.services.transcription_service.AudioSegment') as mock_segment:
                    mock_segment.from_mp3.return_value = mock_audio
                    with patch('audio_pipeline.services.transcription_service.np.array', return_value=MagicMock()):
                        with patch('torch.from_numpy') as mock_torch:
                            mock_torch.return_value.unsqueeze.return_value = MagicMock()
                            waveform, sample_rate = service._load_audio_as_waveform(temp_mp3_file)

                mock_segment.from_mp3.assert_called_once()
                assert sample_rate == 16000

                TranscriptionService._instance = None

    def test_load_wav_with_torchaudio(self, temp_audio_file):
        """
        Verify WAV loading with torchaudio.

        WAV files should be loaded using torchaudio.
        """
        with patch('audio_pipeline.services.transcription_service.FUNASR_AVAILABLE', True):
            with patch('audio_pipeline.services.transcription_service.TORCHAUDIO_AVAILABLE', True):
                from audio_pipeline.services.transcription_service import TranscriptionService

                TranscriptionService._instance = None
                service = TranscriptionService()

                mock_waveform = MagicMock()
                mock_waveform.shape = (1, 16000)

                with patch('torchaudio.load', return_value=(mock_waveform, 16000)):
                    waveform, sample_rate = service._load_audio_as_waveform(temp_audio_file)

                assert sample_rate == 16000

                TranscriptionService._instance = None

    def test_load_stereo_to_mono_conversion(self, temp_audio_file):
        """
        Verify stereo audio is converted to mono.

        Stereo files should be converted to mono for transcription.
        """
        with patch('audio_pipeline.services.transcription_service.FUNASR_AVAILABLE', True):
            with patch('audio_pipeline.services.transcription_service.TORCHAUDIO_AVAILABLE', True):
                from audio_pipeline.services.transcription_service import TranscriptionService

                TranscriptionService._instance = None
                service = TranscriptionService()

                # Stereo waveform (2 channels)
                mock_waveform = MagicMock()
                mock_waveform.shape = (2, 16000)

                with patch('torchaudio.load', return_value=(mock_waveform, 16000)):
                    with patch('torch.mean', return_value=MagicMock()) as mock_mean:
                        service._load_audio_as_waveform(temp_audio_file)

                # Should call torch.mean for stereo to mono conversion
                mock_mean.assert_called()

                TranscriptionService._instance = None

    def test_load_audio_resampling(self, temp_audio_file):
        """
        Verify audio is resampled to 16kHz.

        Audio at different sample rates should be resampled to 16kHz.
        """
        with patch('audio_pipeline.services.transcription_service.FUNASR_AVAILABLE', True):
            with patch('audio_pipeline.services.transcription_service.TORCHAUDIO_AVAILABLE', True):
                from audio_pipeline.services.transcription_service import TranscriptionService

                TranscriptionService._instance = None
                service = TranscriptionService()

                mock_waveform = MagicMock()
                mock_waveform.shape = (1, 44100)  # 44.1kHz audio

                mock_resampler = MagicMock()
                mock_resampler.return_value = MagicMock()

                with patch('torchaudio.load', return_value=(mock_waveform, 44100)):
                    with patch('torchaudio.transforms.Resample', return_value=mock_resampler):
                        waveform, sample_rate = service._load_audio_as_waveform(temp_audio_file)

                # Resampler should be created for 44100 -> 16000
                assert sample_rate == 16000

                TranscriptionService._instance = None


# =============================================================================
# Test Class: Audio Chunking
# =============================================================================


class TestAudioChunking:
    """Tests for audio chunking logic."""

    def test_chunk_audio_creates_correct_chunks(self, mock_audio_data_long):
        """
        Verify audio chunking creates correct number of chunks.

        Long audio should be split into overlapping chunks.
        """
        with patch('audio_pipeline.services.transcription_service.FUNASR_AVAILABLE', True):
            from audio_pipeline.services.transcription_service import TranscriptionService

            TranscriptionService._instance = None
            service = TranscriptionService()

            chunks = service._chunk_audio(
                mock_audio_data_long.waveform,
                mock_audio_data_long.sample_rate,
                chunk_duration=25,
                overlap=2
            )

            # 120 seconds with 25s chunks and 23s step (25-2) should give ~6 chunks
            assert len(chunks) >= 5, "Should create multiple chunks"

            TranscriptionService._instance = None

    def test_chunk_audio_overlap(self, mock_audio_data_long):
        """
        Verify chunks overlap correctly.

        Adjacent chunks should overlap by the specified amount.
        """
        with patch('audio_pipeline.services.transcription_service.FUNASR_AVAILABLE', True):
            from audio_pipeline.services.transcription_service import TranscriptionService

            TranscriptionService._instance = None
            service = TranscriptionService()

            chunks = service._chunk_audio(
                mock_audio_data_long.waveform,
                mock_audio_data_long.sample_rate,
                chunk_duration=25,
                overlap=2
            )

            # Check overlap between first two chunks
            if len(chunks) >= 2:
                _, start1, end1 = chunks[0]
                _, start2, end2 = chunks[1]

                overlap = end1 - start2
                assert overlap > 0, "Chunks should overlap"
                assert abs(overlap - 2) < 0.1, "Overlap should be approximately 2 seconds"

            TranscriptionService._instance = None

    def test_chunk_audio_short_audio_single_chunk(self, mock_audio_data_short):
        """
        Verify short audio creates single chunk.

        Audio shorter than chunk duration should not be split.
        """
        with patch('audio_pipeline.services.transcription_service.FUNASR_AVAILABLE', True):
            from audio_pipeline.services.transcription_service import TranscriptionService

            TranscriptionService._instance = None
            service = TranscriptionService()

            chunks = service._chunk_audio(
                mock_audio_data_short.waveform,
                mock_audio_data_short.sample_rate,
                chunk_duration=25,
                overlap=2
            )

            assert len(chunks) == 1, "Short audio should create single chunk"

            TranscriptionService._instance = None

    def test_chunk_audio_timing(self, mock_audio_data_long):
        """
        Verify chunk timing is correct.

        Each chunk should have valid start and end times.
        """
        with patch('audio_pipeline.services.transcription_service.FUNASR_AVAILABLE', True):
            from audio_pipeline.services.transcription_service import TranscriptionService

            TranscriptionService._instance = None
            service = TranscriptionService()

            chunks = service._chunk_audio(
                mock_audio_data_long.waveform,
                mock_audio_data_long.sample_rate,
                chunk_duration=25,
                overlap=2
            )

            for i, (waveform, start_time, end_time) in enumerate(chunks):
                assert start_time >= 0, f"Chunk {i}: start_time should be >= 0"
                assert end_time > start_time, f"Chunk {i}: end_time should be > start_time"
                assert end_time - start_time <= 25, f"Chunk {i}: duration should be <= 25s"

            TranscriptionService._instance = None


# =============================================================================
# Test Class: Parallel Processing
# =============================================================================


class TestParallelProcessing:
    """Tests for parallel chunk processing."""

    @pytest.mark.asyncio
    async def test_transcribe_chunks_parallel(self, progress_tracker):
        """
        Verify parallel chunk transcription.

        Multiple chunks should be processed concurrently.
        """
        with patch('audio_pipeline.services.transcription_service.FUNASR_AVAILABLE', True):
            from audio_pipeline.services.transcription_service import TranscriptionService

            TranscriptionService._instance = None
            service = TranscriptionService()
            service.is_initialized = True
            service.model = MagicMock()
            service.model.generate.return_value = [{"text": "chunk text"}]

            chunk_data = [
                {'index': 0, 'temp_path': '/tmp/chunk0.wav', 'start_time': 0.0, 'end_time': 25.0},
                {'index': 1, 'temp_path': '/tmp/chunk1.wav', 'start_time': 23.0, 'end_time': 48.0},
                {'index': 2, 'temp_path': '/tmp/chunk2.wav', 'start_time': 46.0, 'end_time': 60.0},
            ]

            results = await service._transcribe_chunks_parallel(
                chunk_data, "auto", progress_tracker.callback
            )

            assert len(results) == 3, "Should return results for all chunks"

            # Results should be indexed correctly
            indices = [idx for idx, _ in results]
            assert sorted(indices) == [0, 1, 2], "All chunk indices should be present"

            TranscriptionService._instance = None

    @pytest.mark.asyncio
    async def test_transcribe_chunks_sequential_fallback(self, progress_tracker):
        """
        Verify sequential chunk transcription (fallback).

        When parallel is disabled, chunks should be processed sequentially.
        """
        with patch('audio_pipeline.services.transcription_service.FUNASR_AVAILABLE', True):
            from audio_pipeline.services.transcription_service import TranscriptionService

            TranscriptionService._instance = None
            service = TranscriptionService()
            service.is_initialized = True
            service.model = MagicMock()
            service.model.generate.return_value = [{"text": "chunk text"}]

            chunk_data = [
                {'index': 0, 'temp_path': '/tmp/chunk0.wav', 'start_time': 0.0, 'end_time': 25.0},
                {'index': 1, 'temp_path': '/tmp/chunk1.wav', 'start_time': 23.0, 'end_time': 48.0},
            ]

            results = await service._transcribe_chunks_sequential(
                chunk_data, "auto", progress_tracker.callback
            )

            assert len(results) == 2, "Should return results for all chunks"

            # Verify progress callbacks were invoked
            assert len(progress_tracker.calls) >= 2, "Should have progress callbacks"

            TranscriptionService._instance = None

    def test_transcribe_single_chunk_success(self):
        """
        Verify single chunk transcription.

        Single chunk should be transcribed correctly.
        """
        with patch('audio_pipeline.services.transcription_service.FUNASR_AVAILABLE', True):
            from audio_pipeline.services.transcription_service import TranscriptionService

            TranscriptionService._instance = None
            service = TranscriptionService()
            service.is_initialized = True
            service.model = MagicMock()
            service.model.generate.return_value = [{"text": "<|en|><|NEUTRAL|>Test text"}]

            idx, text = service._transcribe_single_chunk("/tmp/test.wav", "auto", 0)

            assert idx == 0, "Should return correct index"
            assert "Test text" in text, "Should return transcribed text"

            TranscriptionService._instance = None

    def test_transcribe_single_chunk_error(self):
        """
        Verify error handling in single chunk transcription.

        Errors should be captured and returned as error text.
        """
        with patch('audio_pipeline.services.transcription_service.FUNASR_AVAILABLE', True):
            from audio_pipeline.services.transcription_service import TranscriptionService

            TranscriptionService._instance = None
            service = TranscriptionService()
            service.is_initialized = True
            service.model = MagicMock()
            service.model.generate.side_effect = RuntimeError("Model error")

            idx, text = service._transcribe_single_chunk("/tmp/test.wav", "auto", 0)

            assert idx == 0, "Should return correct index"
            assert "TRANSCRIPTION_ERROR" in text, "Should return error marker"

            TranscriptionService._instance = None


# =============================================================================
# Test Class: Temp File Handling
# =============================================================================


class TestTempFileHandling:
    """Tests for temporary file creation and cleanup."""

    def test_save_chunk_to_temp(self):
        """
        Verify chunk saving to temporary file.

        Should create a temporary WAV file.
        """
        with patch('audio_pipeline.services.transcription_service.FUNASR_AVAILABLE', True):
            with patch('audio_pipeline.services.transcription_service.SOUNDFILE_AVAILABLE', True):
                from audio_pipeline.services.transcription_service import TranscriptionService

                TranscriptionService._instance = None
                service = TranscriptionService()

                mock_waveform = MagicMock()
                mock_waveform.squeeze.return_value.numpy.return_value = [0.0] * 16000

                with patch('tempfile.mkstemp', return_value=(0, '/tmp/chunk_test.wav')):
                    with patch('os.close'):
                        with patch('audio_pipeline.services.transcription_service.sf.write'):
                            path = service._save_chunk_to_temp(mock_waveform, 16000)

                assert path.endswith('.wav'), "Should create WAV file"

                TranscriptionService._instance = None

    def test_save_chunk_to_temp_fallback_torchaudio(self):
        """
        Verify fallback to torchaudio when soundfile unavailable.

        Should use torchaudio.save when soundfile is not available.
        """
        with patch('audio_pipeline.services.transcription_service.FUNASR_AVAILABLE', True):
            with patch('audio_pipeline.services.transcription_service.SOUNDFILE_AVAILABLE', False):
                from audio_pipeline.services.transcription_service import TranscriptionService

                TranscriptionService._instance = None
                service = TranscriptionService()

                mock_waveform = MagicMock()
                mock_waveform.squeeze.return_value.numpy.return_value = [0.0] * 16000

                with patch('tempfile.mkstemp', return_value=(0, '/tmp/chunk_test.wav')):
                    with patch('os.close'):
                        with patch('torchaudio.save') as mock_save:
                            path = service._save_chunk_to_temp(mock_waveform, 16000)

                mock_save.assert_called_once()

                TranscriptionService._instance = None


# =============================================================================
# Test Class: Chunked Processing with Progress
# =============================================================================


class TestChunkedProcessingWithProgress:
    """Tests for chunked processing with progress callbacks."""

    @pytest.mark.asyncio
    async def test_transcribe_with_chunking_progress(
        self, temp_audio_file, mock_sensevoice_model, progress_tracker
    ):
        """
        Verify progress callbacks during chunked processing.

        Should invoke callbacks at each stage.
        """
        with patch('audio_pipeline.services.transcription_service.FUNASR_AVAILABLE', True):
            with patch('audio_pipeline.services.transcription_service.PYDUB_AVAILABLE', True):
                from audio_pipeline.services.transcription_service import TranscriptionService

                TranscriptionService._instance = None
                service = TranscriptionService()
                service.is_initialized = True
                service.model = mock_sensevoice_model
                service.vad_available = False

                # Mock for long audio that triggers chunking
                with patch.object(service, 'get_audio_metadata', return_value={
                    "duration_seconds": 60.0,  # Long enough for chunking
                    "sample_rate": 16000,
                    "channels": 1,
                    "format": "mp3",
                    "file_size_bytes": 960000
                }):
                    # Mock audio loading and chunking
                    mock_waveform = MagicMock()
                    mock_waveform.shape = (1, 960000)

                    with patch.object(service, '_load_audio_as_waveform',
                                      return_value=(mock_waveform, 16000)):
                        with patch.object(service, '_chunk_audio', return_value=[
                            (MagicMock(), 0.0, 25.0),
                            (MagicMock(), 23.0, 48.0),
                            (MagicMock(), 46.0, 60.0),
                        ]):
                            with patch.object(service, '_save_chunk_to_temp', return_value='/tmp/test.wav'):
                                with patch('os.unlink'):
                                    result = await service._transcribe_with_chunking(
                                        temp_audio_file,
                                        "auto",
                                        progress_tracker.callback
                                    )

                # Verify progress callbacks were invoked
                assert progress_tracker.has_step("chunk"), "Should have chunk progress"

                TranscriptionService._instance = None


# =============================================================================
# Test Class: Global Function
# =============================================================================


class TestGlobalFunction:
    """Tests for get_transcription_service global function."""

    def test_get_transcription_service(self):
        """
        Verify get_transcription_service returns singleton instance.

        Should return the same instance as get_instance().
        """
        with patch('audio_pipeline.services.transcription_service.FUNASR_AVAILABLE', True):
            from audio_pipeline.services.transcription_service import (
                get_transcription_service, TranscriptionService
            )

            TranscriptionService._instance = None

            service = get_transcription_service()
            instance = TranscriptionService.get_instance()

            assert service is instance, "Should return singleton instance"

            TranscriptionService._instance = None
