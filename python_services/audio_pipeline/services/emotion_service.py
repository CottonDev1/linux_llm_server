"""
Emotion Service

Parses emotion and audio event tags from SenseVoice transcription output.
"""

import re
from typing import Dict, List, Tuple, Optional


class EmotionService:
    """
    Service for parsing emotion and audio event tags from SenseVoice output.

    SenseVoice embeds special tags in transcription:
    - <|EMOTION|> for emotions (e.g., <|HAPPY|>, <|SAD|>)
    - <|EVENT|> for audio events (e.g., <|Laughter|>, <|BGM|>)
    - <|LANG|> for language codes
    """

    # Emotion tags supported by SenseVoice
    EMOTION_TAGS = [
        "HAPPY", "SAD", "ANGRY", "NEUTRAL",
        "FEARFUL", "DISGUSTED", "SURPRISED"
    ]

    # Audio event tags supported by SenseVoice
    EVENT_TAGS = [
        "Speech", "BGM", "Applause", "Laughter",
        "Cry", "Cough", "Sneeze", "Breath"
    ]

    _instance: Optional['EmotionService'] = None

    @classmethod
    def get_instance(cls) -> 'EmotionService':
        """Get singleton instance"""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def parse_tags(self, text: str, keep_emotion_tags: bool = True) -> Tuple[str, Dict]:
        """
        Parse emotion and event tags from SenseVoice output.

        Args:
            text: Raw transcription with tags
            keep_emotion_tags: If True, preserve emotion tags in output text (default True)

        Returns:
            Tuple of (processed_text, metadata_dict)
        """
        metadata = {
            "emotions": [],
            "audio_events": [],
            "language": None,
            "raw_transcription": text
        }

        # Extract emotion tags
        # SenseVoice can return: HAPPY, SAD, ANGRY, NEUTRAL, FEARFUL, DISGUSTED, SURPRISED
        # It may also return EMO_UNKNOWN or EMO_UNKOWN (typo in model) when emotion is unclear
        # We map both variants to NEUTRAL for consistency
        emotion_pattern = r'<\|(HAPPY|SAD|ANGRY|NEUTRAL|FEARFUL|DISGUSTED|SURPRISED|EMO_UNKNOWN|EMO_UNKOWN)\|>'
        emotions = re.findall(emotion_pattern, text)
        # Map EMO_UNKNOWN and EMO_UNKOWN (typo) to NEUTRAL for consistency
        emotions = ['NEUTRAL' if e in ('EMO_UNKNOWN', 'EMO_UNKOWN') else e for e in emotions]
        metadata["emotions"] = list(set(emotions))  # Remove duplicates

        # Extract audio event tags
        event_pattern = r'<\|(Speech|BGM|Applause|Laughter|Cry|Cough|Sneeze|Breath)\|>'
        events = re.findall(event_pattern, text)
        metadata["audio_events"] = list(set(events))

        # Extract language tag
        lang_pattern = r'<\|([a-z]{2})\|>'
        lang_match = re.search(lang_pattern, text)
        if lang_match:
            metadata["language"] = lang_match.group(1)

        # Process text based on keep_emotion_tags setting
        if keep_emotion_tags:
            # Keep emotion tags but format them nicely, remove language/event tags
            processed_text = text
            # Remove language tags
            processed_text = re.sub(r'<\|[a-z]{2}\|>', '', processed_text)
            # Remove audio event tags (Speech, BGM, etc.)
            processed_text = re.sub(event_pattern, '', processed_text)
            # Format emotion tags to be more readable: <|HAPPY|> -> [HAPPY]
            processed_text = re.sub(r'<\|(HAPPY|SAD|ANGRY|NEUTRAL|FEARFUL|DISGUSTED|SURPRISED)\|>', r'[\1]', processed_text)
            # Remove EMO_UNKNOWN variants
            processed_text = re.sub(r'<\|(EMO_UNKNOWN|EMO_UNKOWN)\|>', '', processed_text)
            processed_text = processed_text.strip()
        else:
            # Remove all tags to get clean transcription
            processed_text = re.sub(r'<\|[^|]+\|>', '', text)
            processed_text = processed_text.strip()

        return processed_text, metadata

    def get_clean_text(self, text: str) -> str:
        """
        Get transcription with all tags removed.

        Args:
            text: Raw transcription with tags

        Returns:
            Clean text with all tags stripped
        """
        clean_text = re.sub(r'<\|[^|]+\|>', '', text)
        return clean_text.strip()

    # Emotion severity ranking (worst/most concerning to best)
    # Used to determine the primary emotion - we want to surface the worst emotion
    EMOTION_SEVERITY = {
        "ANGRY": 1,      # Worst - indicates conflict or frustration
        "DISGUSTED": 2,  # Very negative reaction
        "FEARFUL": 3,    # Concern or worry
        "SAD": 4,        # Negative but less intense
        "SURPRISED": 5,  # Could indicate unexpected issues
        "NEUTRAL": 6,    # Normal/baseline
        "HAPPY": 7,      # Best - positive interaction
    }

    def get_primary_emotion(self, emotions: List[str]) -> str:
        """
        Determine primary emotion from list of detected emotions.
        Returns the WORST (most severe) emotion to surface potential issues.

        Args:
            emotions: List of detected emotions

        Returns:
            Primary emotion string (the worst/most concerning emotion detected)
        """
        if not emotions:
            return "NEUTRAL"

        # Find the emotion with the lowest severity score (worst)
        worst_emotion = "NEUTRAL"
        worst_score = self.EMOTION_SEVERITY.get("NEUTRAL", 6)

        for emotion in emotions:
            emotion_upper = emotion.upper()
            score = self.EMOTION_SEVERITY.get(emotion_upper, 6)
            if score < worst_score:
                worst_score = score
                worst_emotion = emotion_upper

        return worst_emotion

    def merge_emotions(self, emotion_lists: List[List[str]]) -> List[str]:
        """
        Merge emotion lists from multiple chunks.

        Args:
            emotion_lists: List of emotion lists from each chunk

        Returns:
            Merged list of unique emotions
        """
        all_emotions = set()
        for emotions in emotion_lists:
            all_emotions.update(emotions)
        return list(all_emotions)

    def merge_events(self, event_lists: List[List[str]]) -> List[str]:
        """
        Merge audio event lists from multiple chunks.

        Args:
            event_lists: List of event lists from each chunk

        Returns:
            Merged list of unique events
        """
        all_events = set()
        for events in event_lists:
            all_events.update(events)
        return list(all_events)


def get_emotion_service() -> EmotionService:
    """Get the singleton emotion service instance"""
    return EmotionService.get_instance()
