"""
Metadata Service

Parses call metadata from RingCentral recording filenames.
"""

import re
from typing import Dict, Optional
from pathlib import Path


class MetadataService:
    """
    Service for parsing call metadata from recording filenames.

    Supports RingCentral filename patterns:
    1. With extension (outgoing): yyyymmdd-hhmmss_EXT_PHONE_DIRECTION_AUTO_RECORDINGID.mp3
       Example: 20251104-133555_302_(252)792-8686_Outgoing_Auto_2243071124051.mp3
    2. Without extension (incoming): yyyymmdd-hhmmss_PHONE_DIRECTION_AUTO_RECORDINGID.mp3
       Example: 20251121-141152_(469)906-0558_Incoming_Auto_2254843027051.mp3
    """

    _instance: Optional['MetadataService'] = None

    # Pattern with extension (outgoing calls)
    PATTERN_WITH_EXT = r'^(\d{4})(\d{2})(\d{2})-(\d{2})(\d{2})(\d{2})_(\d+)_(\([^)]+\)[^_]+|[^_]+)_(Incoming|Outgoing)_([^_]+)_(\d+)$'

    # Pattern without extension (incoming calls)
    PATTERN_NO_EXT = r'^(\d{4})(\d{2})(\d{2})-(\d{2})(\d{2})(\d{2})_(\([^)]+\)[^_]+|[^_]+)_(Incoming|Outgoing)_([^_]+)_(\d+)$'

    @classmethod
    def get_instance(cls) -> 'MetadataService':
        """Get singleton instance"""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def parse_filename(self, filepath: str) -> Dict:
        """
        Parse RingCentral call recording filename to extract metadata.

        Args:
            filepath: Path to the audio file

        Returns:
            Dict with parsed fields and 'parsed' flag indicating success.
        """
        result = {
            "call_date": None,
            "call_time": None,
            "extension": None,
            "phone_number": None,
            "direction": None,
            "auto_flag": None,
            "recording_id": None,
            "parsed": False
        }

        # Get just the filename from the path
        filename = Path(filepath).stem  # Remove extension

        # Try pattern with extension first
        match = re.match(self.PATTERN_WITH_EXT, filename)
        if match:
            year, month, day, hour, minute, second, ext, phone, direction, auto, recording_id = match.groups()
            result["call_date"] = f"{year}-{month}-{day}"
            result["call_time"] = f"{hour}:{minute}:{second}"
            result["extension"] = ext
            result["phone_number"] = phone
            result["direction"] = direction
            result["auto_flag"] = auto
            result["recording_id"] = recording_id
            result["parsed"] = True
        else:
            # Try pattern without extension (typically incoming calls)
            match = re.match(self.PATTERN_NO_EXT, filename)
            if match:
                year, month, day, hour, minute, second, phone, direction, auto, recording_id = match.groups()
                result["call_date"] = f"{year}-{month}-{day}"
                result["call_time"] = f"{hour}:{minute}:{second}"
                result["extension"] = None  # No extension for incoming calls
                result["phone_number"] = phone
                result["direction"] = direction
                result["auto_flag"] = auto
                result["recording_id"] = recording_id
                result["parsed"] = True

        return result

    def normalize_phone_number(self, phone: str) -> str:
        """
        Normalize phone number to digits only.

        Args:
            phone: Phone number in any format

        Returns:
            Normalized phone number (digits only)
        """
        if not phone:
            return ""
        return re.sub(r'\D', '', phone)

    def format_phone_display(self, phone: str) -> str:
        """
        Format phone number for display.

        Args:
            phone: Normalized phone number

        Returns:
            Formatted phone number for display
        """
        digits = self.normalize_phone_number(phone)
        if len(digits) == 10:
            return f"({digits[:3]}){digits[3:6]}-{digits[6:]}"
        elif len(digits) == 11 and digits[0] == '1':
            return f"+1 ({digits[1:4]}){digits[4:7]}-{digits[7:]}"
        return phone


def get_metadata_service() -> MetadataService:
    """Get the singleton metadata service instance"""
    return MetadataService.get_instance()


def parse_call_filename(filepath: str) -> Dict:
    """
    Convenience function to parse call filename.

    Args:
        filepath: Path to the audio file

    Returns:
        Dict with parsed metadata
    """
    return get_metadata_service().parse_filename(filepath)
