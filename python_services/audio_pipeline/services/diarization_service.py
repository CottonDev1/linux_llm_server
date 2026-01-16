"""
Speaker Diarization Service

Handles speaker diarization using pyannote.audio to identify
different speakers in audio recordings.
"""

import os
import tempfile
from typing import Optional, List, Dict, Any, Tuple
from pathlib import Path
from dataclasses import dataclass

# pyannote.audio for speaker diarization
try:
    from pyannote.audio import Pipeline
    PYANNOTE_AVAILABLE = True
except ImportError:
    PYANNOTE_AVAILABLE = False
    print("Warning: pyannote.audio not installed. Speaker diarization will be disabled.")

# Audio processing
try:
    from pydub import AudioSegment
    import numpy as np
    PYDUB_AVAILABLE = True
except ImportError:
    PYDUB_AVAILABLE = False

try:
    import torch
    import torchaudio
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from scipy.io import wavfile
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


@dataclass
class SpeakerSegment:
    """Represents a segment of speech from a specific speaker"""
    speaker: str
    start_time: float
    end_time: float
    text: str = ""

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time


class DiarizationService:
    """
    Service for speaker diarization using pyannote.audio.

    Identifies different speakers in audio recordings and provides
    timestamped segments for each speaker.
    """

    _instance: Optional['DiarizationService'] = None

    # Hugging Face token for pyannote models (loaded from environment)
    HF_TOKEN = os.environ.get("HF_TOKEN", "")

    def __init__(self):
        self.pipeline = None
        self.is_initialized = False
        self.device = "cuda" if self._check_cuda() else "cpu"

    @classmethod
    def get_instance(cls) -> 'DiarizationService':
        """Get singleton instance"""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _check_cuda(self) -> bool:
        """Check if CUDA is available"""
        if not TORCH_AVAILABLE:
            return False
        try:
            return torch.cuda.is_available()
        except:
            return False

    async def initialize(self):
        """Initialize the pyannote diarization pipeline"""
        if self.is_initialized:
            return

        if not PYANNOTE_AVAILABLE:
            raise RuntimeError(
                "pyannote.audio not installed. Install with: pip install pyannote.audio"
            )

        print(f"Initializing pyannote speaker diarization on device: {self.device}")

        try:
            # Load the pyannote speaker diarization pipeline
            # Note: 'token' parameter replaced deprecated 'use_auth_token' in newer versions
            self.pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                token=self.HF_TOKEN
            )

            # Move to GPU if available
            if self.device == "cuda":
                self.pipeline.to(torch.device("cuda"))

            self.is_initialized = True
            print("pyannote speaker diarization initialized successfully")

        except Exception as e:
            print(f"Error initializing pyannote diarization: {e}")
            raise

    def _convert_to_wav(self, audio_path: str) -> str:
        """
        Convert audio file to WAV format if needed.
        pyannote works best with WAV files.

        Args:
            audio_path: Path to audio file

        Returns:
            Path to WAV file (original or converted)
        """
        file_ext = Path(audio_path).suffix.lower()

        if file_ext == '.wav':
            return audio_path

        if not PYDUB_AVAILABLE:
            raise RuntimeError("pydub required for non-WAV audio conversion")

        print(f"Converting {file_ext} to WAV for diarization...")

        # Load and convert
        if file_ext == '.mp3':
            audio = AudioSegment.from_mp3(audio_path)
        elif file_ext == '.m4a':
            audio = AudioSegment.from_file(audio_path, format='m4a')
        elif file_ext == '.flac':
            audio = AudioSegment.from_file(audio_path, format='flac')
        elif file_ext == '.ogg':
            audio = AudioSegment.from_ogg(audio_path)
        else:
            audio = AudioSegment.from_file(audio_path)

        # Convert to mono 16kHz for optimal diarization
        audio = audio.set_channels(1).set_frame_rate(16000)

        # Save to temp file
        temp_fd, temp_path = tempfile.mkstemp(suffix='.wav')
        os.close(temp_fd)
        audio.export(temp_path, format='wav')

        return temp_path

    def _load_audio_as_waveform(self, audio_path: str) -> Dict[str, Any]:
        """
        Load audio file as a waveform dictionary for pyannote.
        This bypasses torchcodec issues by loading audio manually.

        Args:
            audio_path: Path to audio file (WAV format)

        Returns:
            Dictionary with 'waveform' (torch.Tensor) and 'sample_rate' (int)
        """
        if not TORCH_AVAILABLE:
            raise RuntimeError("torch/torchaudio required for audio loading")

        try:
            # Try torchaudio first (preferred)
            waveform, sample_rate = torchaudio.load(audio_path)
            return {"waveform": waveform, "sample_rate": sample_rate}
        except Exception as e:
            print(f"torchaudio load failed: {e}, trying scipy...")

        # Fallback to scipy + manual conversion
        if SCIPY_AVAILABLE:
            sample_rate, data = wavfile.read(audio_path)

            # Convert to float32 and normalize
            if data.dtype == np.int16:
                data = data.astype(np.float32) / 32768.0
            elif data.dtype == np.int32:
                data = data.astype(np.float32) / 2147483648.0
            elif data.dtype == np.uint8:
                data = (data.astype(np.float32) - 128) / 128.0

            # Ensure 2D (channel, time) format
            if len(data.shape) == 1:
                data = data.reshape(1, -1)  # Mono: add channel dimension
            else:
                data = data.T  # Stereo: transpose to (channel, time)

            waveform = torch.from_numpy(data)
            return {"waveform": waveform, "sample_rate": sample_rate}

        raise RuntimeError("No audio loading library available (torchaudio or scipy)")

    async def diarize(
        self,
        audio_path: str,
        min_speakers: Optional[int] = None,
        max_speakers: Optional[int] = None,
        progress_callback=None
    ) -> List[SpeakerSegment]:
        """
        Perform speaker diarization on an audio file.

        Args:
            audio_path: Path to audio file
            min_speakers: Minimum expected number of speakers
            max_speakers: Maximum expected number of speakers
            progress_callback: Optional async callback(step, message)

        Returns:
            List of SpeakerSegment objects with speaker labels and timestamps
        """
        if not self.is_initialized:
            await self.initialize()

        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        if progress_callback:
            await progress_callback("diarize", "Starting speaker diarization...")

        # Convert to WAV if needed
        wav_path = self._convert_to_wav(audio_path)
        temp_created = wav_path != audio_path

        try:
            print(f"Running speaker diarization on: {audio_path}")

            # Configure diarization parameters
            diarization_params = {}
            if min_speakers is not None:
                diarization_params['min_speakers'] = min_speakers
            if max_speakers is not None:
                diarization_params['max_speakers'] = max_speakers

            # For phone calls, typically 2 speakers
            if not diarization_params:
                diarization_params = {'min_speakers': 2, 'max_speakers': 2}

            # Load audio as waveform to bypass torchcodec issues
            audio_input = self._load_audio_as_waveform(wav_path)

            # Run diarization with waveform input
            diarization_output = self.pipeline(audio_input, **diarization_params)

            # Extract the annotation from DiarizeOutput (pyannote 4.0+ format)
            if hasattr(diarization_output, 'speaker_diarization'):
                diarization = diarization_output.speaker_diarization
            else:
                # Fallback for older versions
                diarization = diarization_output

            # Convert to SpeakerSegment list
            segments = []
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                segment = SpeakerSegment(
                    speaker=speaker,
                    start_time=turn.start,
                    end_time=turn.end
                )
                segments.append(segment)

            # Sort by start time
            segments.sort(key=lambda s: s.start_time)

            # Rename speakers to caller labels (Caller 1, Caller 2, etc.)
            speaker_map = {}
            for seg in segments:
                if seg.speaker not in speaker_map:
                    speaker_map[seg.speaker] = f"Caller {len(speaker_map) + 1}"
                seg.speaker = speaker_map[seg.speaker]

            print(f"Diarization complete: {len(segments)} segments, {len(speaker_map)} speakers")

            if progress_callback:
                await progress_callback(
                    "diarize",
                    f"Found {len(speaker_map)} speakers in {len(segments)} segments"
                )

            return segments

        finally:
            # Clean up temp file if we created one
            if temp_created and os.path.exists(wav_path):
                try:
                    os.unlink(wav_path)
                except:
                    pass

    def merge_transcription_with_diarization(
        self,
        transcription: str,
        segments: List[SpeakerSegment],
        audio_duration: float
    ) -> Tuple[str, List[Dict]]:
        """
        Merge transcription text with speaker diarization segments.

        This uses a simple approach: split transcription by time proportionally
        and assign to speaker segments.

        Args:
            transcription: Full transcription text
            segments: List of SpeakerSegment from diarization
            audio_duration: Total audio duration in seconds

        Returns:
            Tuple of (speaker-labeled transcription, segment details)
        """
        if not segments:
            return transcription, []

        # Split transcription into words/sentences
        # Use sentence-like splitting (by punctuation)
        import re
        sentences = re.split(r'(?<=[.!?])\s+', transcription.strip())
        if not sentences or (len(sentences) == 1 and not sentences[0]):
            sentences = [transcription]

        # Calculate time per character (rough estimation)
        total_chars = len(transcription)
        if total_chars == 0:
            return transcription, []

        chars_per_second = total_chars / audio_duration if audio_duration > 0 else 1

        # Assign sentences to speaker segments based on timing
        labeled_parts = []
        segment_details = []
        current_char_pos = 0

        for segment in segments:
            # Find sentences that fall within this segment's time range
            segment_start_char = int(segment.start_time * chars_per_second)
            segment_end_char = int(segment.end_time * chars_per_second)

            # Extract text for this segment
            segment_text = transcription[segment_start_char:segment_end_char].strip()

            if segment_text:
                labeled_parts.append(f"{segment.speaker}: {segment_text}")
                segment.text = segment_text
                segment_details.append({
                    "speaker": segment.speaker,
                    "start_time": segment.start_time,
                    "end_time": segment.end_time,
                    "text": segment_text
                })

        # If simple char-based approach didn't work well, fall back to sentence-based
        if not labeled_parts or len(labeled_parts) < len(segments) // 2:
            labeled_parts = []
            segment_details = []

            # Distribute sentences across segments by time
            sentence_idx = 0
            for segment in segments:
                segment_duration = segment.duration
                segment_proportion = segment_duration / audio_duration if audio_duration > 0 else 0

                # Estimate number of sentences for this segment
                sentences_for_segment = max(1, int(len(sentences) * segment_proportion))

                segment_sentences = []
                for _ in range(sentences_for_segment):
                    if sentence_idx < len(sentences):
                        segment_sentences.append(sentences[sentence_idx])
                        sentence_idx += 1

                if segment_sentences:
                    segment_text = ' '.join(segment_sentences)
                    labeled_parts.append(f"{segment.speaker}: {segment_text}")
                    segment.text = segment_text
                    segment_details.append({
                        "speaker": segment.speaker,
                        "start_time": segment.start_time,
                        "end_time": segment.end_time,
                        "text": segment_text
                    })

            # Handle remaining sentences
            while sentence_idx < len(sentences):
                if segment_details:
                    segment_details[-1]["text"] += ' ' + sentences[sentence_idx]
                    labeled_parts[-1] += ' ' + sentences[sentence_idx]
                sentence_idx += 1

        labeled_transcription = '\n\n'.join(labeled_parts)

        return labeled_transcription, segment_details

    def get_speaker_statistics(
        self,
        segments: List[SpeakerSegment]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Calculate speaking statistics for each speaker.

        Args:
            segments: List of SpeakerSegment

        Returns:
            Dict mapping speaker name to statistics
        """
        stats = {}

        for segment in segments:
            if segment.speaker not in stats:
                stats[segment.speaker] = {
                    "total_duration": 0.0,
                    "segment_count": 0,
                    "word_count": 0
                }

            stats[segment.speaker]["total_duration"] += segment.duration
            stats[segment.speaker]["segment_count"] += 1
            if segment.text:
                stats[segment.speaker]["word_count"] += len(segment.text.split())

        # Calculate percentages
        total_duration = sum(s["total_duration"] for s in stats.values())
        for speaker_stats in stats.values():
            if total_duration > 0:
                speaker_stats["percentage"] = round(
                    speaker_stats["total_duration"] / total_duration * 100, 1
                )
            else:
                speaker_stats["percentage"] = 0

        return stats


def get_diarization_service() -> DiarizationService:
    """Get the singleton diarization service instance"""
    return DiarizationService.get_instance()
