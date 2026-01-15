"""
Format Converter

Converts audio files between formats.
"""

import os
import tempfile
from pathlib import Path
from typing import Optional, Tuple

# Audio processing libraries
try:
    from pydub import AudioSegment
    PYDUB_AVAILABLE = True
except ImportError:
    PYDUB_AVAILABLE = False

try:
    import torchaudio
    import torch
    TORCHAUDIO_AVAILABLE = True
except ImportError:
    TORCHAUDIO_AVAILABLE = False


class FormatConverter:
    """
    Converts audio files between formats.

    Primary use case: Converting MP3 and other formats to WAV
    for processing by transcription models.
    """

    # Target sample rate for transcription
    TARGET_SAMPLE_RATE = 16000

    def __init__(self):
        self.pydub_available = PYDUB_AVAILABLE
        self.torchaudio_available = TORCHAUDIO_AVAILABLE

    def convert_to_wav(
        self,
        input_path: str,
        output_path: Optional[str] = None,
        sample_rate: int = TARGET_SAMPLE_RATE,
        mono: bool = True
    ) -> Tuple[str, bool]:
        """
        Convert audio file to WAV format.

        Args:
            input_path: Path to input audio file
            output_path: Optional output path (creates temp file if None)
            sample_rate: Target sample rate (default 16000)
            mono: Convert to mono if True

        Returns:
            Tuple of (output_path, success)
        """
        input_ext = Path(input_path).suffix.lower()

        # If already WAV, check if conversion is needed
        if input_ext == '.wav' and not mono:
            return input_path, True

        # Create output path if not provided
        if output_path is None:
            fd, output_path = tempfile.mkstemp(suffix='.wav')
            os.close(fd)

        try:
            # Use pydub for MP3 (more reliable on Windows)
            if input_ext == '.mp3' and self.pydub_available:
                return self._convert_with_pydub(
                    input_path, output_path, sample_rate, mono
                )

            # Use torchaudio for other formats
            if self.torchaudio_available:
                return self._convert_with_torchaudio(
                    input_path, output_path, sample_rate, mono
                )

            # Fallback to pydub if available
            if self.pydub_available:
                return self._convert_with_pydub(
                    input_path, output_path, sample_rate, mono
                )

            raise RuntimeError("No audio processing library available (pydub or torchaudio)")

        except Exception as e:
            print(f"Format conversion failed: {e}")
            # Clean up temp file on failure
            if os.path.exists(output_path) and output_path != input_path:
                try:
                    os.unlink(output_path)
                except:
                    pass
            return input_path, False

    def _convert_with_pydub(
        self,
        input_path: str,
        output_path: str,
        sample_rate: int,
        mono: bool
    ) -> Tuple[str, bool]:
        """Convert using pydub"""
        input_ext = Path(input_path).suffix.lower().replace('.', '')

        # Load audio
        if input_ext == 'mp3':
            audio = AudioSegment.from_mp3(input_path)
        elif input_ext == 'wav':
            audio = AudioSegment.from_wav(input_path)
        elif input_ext == 'flac':
            audio = AudioSegment.from_file(input_path, format='flac')
        elif input_ext in ['m4a', 'aac']:
            audio = AudioSegment.from_file(input_path, format='m4a')
        elif input_ext == 'ogg':
            audio = AudioSegment.from_ogg(input_path)
        else:
            audio = AudioSegment.from_file(input_path)

        # Convert to mono
        if mono and audio.channels > 1:
            audio = audio.set_channels(1)

        # Resample
        if audio.frame_rate != sample_rate:
            audio = audio.set_frame_rate(sample_rate)

        # Export as WAV
        audio.export(output_path, format='wav')
        print(f"Converted {input_path} to WAV using pydub")
        return output_path, True

    def _convert_with_torchaudio(
        self,
        input_path: str,
        output_path: str,
        sample_rate: int,
        mono: bool
    ) -> Tuple[str, bool]:
        """Convert using torchaudio"""
        waveform, orig_sample_rate = torchaudio.load(input_path)

        # Convert to mono
        if mono and waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # Resample
        if orig_sample_rate != sample_rate:
            resampler = torchaudio.transforms.Resample(orig_sample_rate, sample_rate)
            waveform = resampler(waveform)

        # Save as WAV
        torchaudio.save(output_path, waveform, sample_rate)
        print(f"Converted {input_path} to WAV using torchaudio")
        return output_path, True

    def get_audio_info(self, file_path: str) -> dict:
        """
        Get audio file information.

        Args:
            file_path: Path to audio file

        Returns:
            Dict with audio info (duration, sample_rate, channels)
        """
        info = {
            "duration_seconds": 0.0,
            "sample_rate": 0,
            "channels": 0,
            "format": Path(file_path).suffix.lower().replace('.', '')
        }

        try:
            if self.torchaudio_available:
                audio_info = torchaudio.info(file_path)
                info["duration_seconds"] = audio_info.num_frames / audio_info.sample_rate
                info["sample_rate"] = audio_info.sample_rate
                info["channels"] = audio_info.num_channels
            elif self.pydub_available:
                audio = AudioSegment.from_file(file_path)
                info["duration_seconds"] = len(audio) / 1000.0
                info["sample_rate"] = audio.frame_rate
                info["channels"] = audio.channels
        except Exception as e:
            print(f"Could not get audio info: {e}")

        return info


# Singleton instance
_converter: Optional[FormatConverter] = None


def get_converter() -> FormatConverter:
    """Get the singleton converter instance"""
    global _converter
    if _converter is None:
        _converter = FormatConverter()
    return _converter


def convert_to_wav(
    input_path: str,
    output_path: Optional[str] = None,
    sample_rate: int = 16000,
    mono: bool = True
) -> Tuple[str, bool]:
    """
    Convenience function to convert audio to WAV.

    Args:
        input_path: Path to input audio file
        output_path: Optional output path
        sample_rate: Target sample rate
        mono: Convert to mono if True

    Returns:
        Tuple of (output_path, success)
    """
    return get_converter().convert_to_wav(input_path, output_path, sample_rate, mono)
