"""
Audio Validator

Validates audio files for format, size, and readability.
"""

import os
from pathlib import Path
from typing import List, Dict, Optional, Tuple


# Supported audio formats
SUPPORTED_FORMATS = [".wav", ".mp3", ".flac", ".m4a", ".ogg", ".webm", ".aac"]

# Maximum file size in MB
MAX_FILE_SIZE_MB = 100

# Minimum file size in bytes (empty or corrupt files)
MIN_FILE_SIZE_BYTES = 1024  # 1KB


class AudioValidator:
    """
    Validates audio files before processing.

    Checks:
    - File existence
    - File format
    - File size limits
    - Basic file readability
    """

    def __init__(
        self,
        supported_formats: List[str] = None,
        max_size_mb: int = MAX_FILE_SIZE_MB,
        min_size_bytes: int = MIN_FILE_SIZE_BYTES
    ):
        self.supported_formats = supported_formats or SUPPORTED_FORMATS
        self.max_size_mb = max_size_mb
        self.min_size_bytes = min_size_bytes

    def validate(self, file_path: str) -> Tuple[bool, Optional[str]]:
        """
        Validate an audio file.

        Args:
            file_path: Path to the audio file

        Returns:
            Tuple of (is_valid, error_message)
        """
        # Check existence
        if not os.path.exists(file_path):
            return False, f"File not found: {file_path}"

        if not os.path.isfile(file_path):
            return False, f"Path is not a file: {file_path}"

        # Check format
        ext = Path(file_path).suffix.lower()
        if ext not in self.supported_formats:
            return False, f"Unsupported format {ext}. Supported: {', '.join(self.supported_formats)}"

        # Check file size
        file_size = os.path.getsize(file_path)

        if file_size < self.min_size_bytes:
            return False, f"File too small ({file_size} bytes). Minimum: {self.min_size_bytes} bytes"

        max_size_bytes = self.max_size_mb * 1024 * 1024
        if file_size > max_size_bytes:
            size_mb = file_size / (1024 * 1024)
            return False, f"File too large ({size_mb:.1f}MB). Maximum: {self.max_size_mb}MB"

        # Check readability
        try:
            with open(file_path, 'rb') as f:
                # Read first few bytes to verify file is readable
                header = f.read(16)
                if len(header) < 4:
                    return False, "File appears to be corrupt or empty"
        except IOError as e:
            return False, f"Cannot read file: {e}"

        return True, None

    def validate_batch(self, file_paths: List[str]) -> Dict[str, Tuple[bool, Optional[str]]]:
        """
        Validate multiple audio files.

        Args:
            file_paths: List of file paths

        Returns:
            Dict mapping file path to (is_valid, error_message)
        """
        results = {}
        for path in file_paths:
            results[path] = self.validate(path)
        return results

    def get_file_info(self, file_path: str) -> Dict:
        """
        Get basic file information.

        Args:
            file_path: Path to the audio file

        Returns:
            Dict with file info
        """
        if not os.path.exists(file_path):
            return {"exists": False}

        stat = os.stat(file_path)
        return {
            "exists": True,
            "path": file_path,
            "filename": os.path.basename(file_path),
            "extension": Path(file_path).suffix.lower(),
            "size_bytes": stat.st_size,
            "size_mb": stat.st_size / (1024 * 1024),
            "modified_time": stat.st_mtime,
        }


# Singleton instance
_validator: Optional[AudioValidator] = None


def get_validator() -> AudioValidator:
    """Get the singleton validator instance"""
    global _validator
    if _validator is None:
        _validator = AudioValidator()
    return _validator


def validate_audio_file(file_path: str) -> Tuple[bool, Optional[str]]:
    """
    Convenience function to validate an audio file.

    Args:
        file_path: Path to the audio file

    Returns:
        Tuple of (is_valid, error_message)
    """
    return get_validator().validate(file_path)


def get_supported_formats() -> List[str]:
    """Get list of supported audio formats"""
    return SUPPORTED_FORMATS.copy()
