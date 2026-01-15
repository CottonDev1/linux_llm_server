"""
EmotionService Tests
====================

Comprehensive tests for the emotion and audio event parsing service.

Tests cover:
1. Service initialization and singleton pattern
2. Emotion tag parsing and formatting
3. Audio event tag parsing
4. Language tag extraction
5. Clean text extraction (tag removal)
6. EMO_UNKNOWN handling
7. Emotion merging from multiple chunks
8. Audio event merging from multiple chunks

These tests are lightweight as EmotionService is a pure parsing service
without external dependencies.
"""

import os
import sys
import pytest
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# Add python_services to path
PYTHON_SERVICES = Path(__file__).parent.parent.parent.parent / "python_services"
sys.path.insert(0, str(PYTHON_SERVICES))


# =============================================================================
# Test Class: Service Initialization
# =============================================================================


class TestEmotionServiceInitialization:
    """Tests for EmotionService initialization."""

    def test_singleton_pattern(self):
        """
        Verify EmotionService uses singleton pattern.

        Multiple calls to get_instance() should return the same instance.
        """
        from audio_pipeline.services.emotion_service import EmotionService

        EmotionService._instance = None

        instance1 = EmotionService.get_instance()
        instance2 = EmotionService.get_instance()

        assert instance1 is instance2, "get_instance() should return same instance"

        EmotionService._instance = None

    def test_emotion_tags_constant(self):
        """
        Verify EMOTION_TAGS constant contains all SenseVoice emotions.

        Should include 7 basic emotions.
        """
        from audio_pipeline.services.emotion_service import EmotionService

        expected_emotions = ["HAPPY", "SAD", "ANGRY", "NEUTRAL", "FEARFUL", "DISGUSTED", "SURPRISED"]

        assert EmotionService.EMOTION_TAGS == expected_emotions, \
            "Should contain all 7 SenseVoice emotions"

    def test_event_tags_constant(self):
        """
        Verify EVENT_TAGS constant contains all SenseVoice audio events.

        Should include 8 audio event types.
        """
        from audio_pipeline.services.emotion_service import EmotionService

        expected_events = ["Speech", "BGM", "Applause", "Laughter", "Cry", "Cough", "Sneeze", "Breath"]

        assert EmotionService.EVENT_TAGS == expected_events, \
            "Should contain all 8 SenseVoice audio events"


# =============================================================================
# Test Class: Emotion Tag Parsing
# =============================================================================


class TestEmotionTagParsing:
    """Tests for emotion tag extraction."""

    def test_parse_happy_emotion(self):
        """
        Verify parsing of HAPPY emotion tag.

        Should correctly identify HAPPY emotion.
        """
        from audio_pipeline.services.emotion_service import EmotionService

        service = EmotionService.get_instance()

        text = "<|HAPPY|>I'm so excited about this!"
        _, metadata = service.parse_tags(text)

        assert "HAPPY" in metadata["emotions"], "Should detect HAPPY emotion"

    def test_parse_sad_emotion(self):
        """
        Verify parsing of SAD emotion tag.

        Should correctly identify SAD emotion.
        """
        from audio_pipeline.services.emotion_service import EmotionService

        service = EmotionService.get_instance()

        text = "<|SAD|>This is really disappointing."
        _, metadata = service.parse_tags(text)

        assert "SAD" in metadata["emotions"], "Should detect SAD emotion"

    def test_parse_angry_emotion(self):
        """
        Verify parsing of ANGRY emotion tag.

        Should correctly identify ANGRY emotion.
        """
        from audio_pipeline.services.emotion_service import EmotionService

        service = EmotionService.get_instance()

        text = "<|ANGRY|>This is unacceptable!"
        _, metadata = service.parse_tags(text)

        assert "ANGRY" in metadata["emotions"], "Should detect ANGRY emotion"

    def test_parse_neutral_emotion(self):
        """
        Verify parsing of NEUTRAL emotion tag.

        Should correctly identify NEUTRAL emotion.
        """
        from audio_pipeline.services.emotion_service import EmotionService

        service = EmotionService.get_instance()

        text = "<|NEUTRAL|>Just a normal statement."
        _, metadata = service.parse_tags(text)

        assert "NEUTRAL" in metadata["emotions"], "Should detect NEUTRAL emotion"

    def test_parse_fearful_emotion(self):
        """
        Verify parsing of FEARFUL emotion tag.

        Should correctly identify FEARFUL emotion.
        """
        from audio_pipeline.services.emotion_service import EmotionService

        service = EmotionService.get_instance()

        text = "<|FEARFUL|>I'm really worried about this."
        _, metadata = service.parse_tags(text)

        assert "FEARFUL" in metadata["emotions"], "Should detect FEARFUL emotion"

    def test_parse_disgusted_emotion(self):
        """
        Verify parsing of DISGUSTED emotion tag.

        Should correctly identify DISGUSTED emotion.
        """
        from audio_pipeline.services.emotion_service import EmotionService

        service = EmotionService.get_instance()

        text = "<|DISGUSTED|>That's really gross."
        _, metadata = service.parse_tags(text)

        assert "DISGUSTED" in metadata["emotions"], "Should detect DISGUSTED emotion"

    def test_parse_surprised_emotion(self):
        """
        Verify parsing of SURPRISED emotion tag.

        Should correctly identify SURPRISED emotion.
        """
        from audio_pipeline.services.emotion_service import EmotionService

        service = EmotionService.get_instance()

        text = "<|SURPRISED|>Wow, I didn't expect that!"
        _, metadata = service.parse_tags(text)

        assert "SURPRISED" in metadata["emotions"], "Should detect SURPRISED emotion"

    def test_parse_multiple_emotions(self):
        """
        Verify parsing of multiple emotion tags.

        Should correctly identify all emotions in text.
        """
        from audio_pipeline.services.emotion_service import EmotionService

        service = EmotionService.get_instance()

        text = "<|HAPPY|>Good news! <|SAD|>But also some bad news. <|ANGRY|>And this is frustrating!"
        _, metadata = service.parse_tags(text)

        assert "HAPPY" in metadata["emotions"], "Should detect HAPPY"
        assert "SAD" in metadata["emotions"], "Should detect SAD"
        assert "ANGRY" in metadata["emotions"], "Should detect ANGRY"

    def test_parse_no_emotions(self):
        """
        Verify handling of text with no emotion tags.

        Should return empty emotions list.
        """
        from audio_pipeline.services.emotion_service import EmotionService

        service = EmotionService.get_instance()

        text = "Just plain text without any emotion tags."
        _, metadata = service.parse_tags(text)

        assert metadata["emotions"] == [], "Should return empty list for no emotions"

    def test_parse_removes_duplicate_emotions(self):
        """
        Verify duplicate emotions are removed.

        Same emotion appearing multiple times should only appear once.
        """
        from audio_pipeline.services.emotion_service import EmotionService

        service = EmotionService.get_instance()

        text = "<|HAPPY|>First happy. <|HAPPY|>Second happy. <|HAPPY|>Third happy."
        _, metadata = service.parse_tags(text)

        assert metadata["emotions"].count("HAPPY") == 1, "Should deduplicate emotions"


# =============================================================================
# Test Class: EMO_UNKNOWN Handling
# =============================================================================


class TestEmoUnknownHandling:
    """Tests for EMO_UNKNOWN tag handling."""

    def test_parse_emo_unknown_maps_to_neutral(self):
        """
        Verify EMO_UNKNOWN is mapped to NEUTRAL.

        SenseVoice sometimes returns EMO_UNKNOWN for unclear emotions.
        """
        from audio_pipeline.services.emotion_service import EmotionService

        service = EmotionService.get_instance()

        text = "<|EMO_UNKNOWN|>Unclear emotional content."
        _, metadata = service.parse_tags(text)

        assert "NEUTRAL" in metadata["emotions"], "EMO_UNKNOWN should map to NEUTRAL"
        assert "EMO_UNKNOWN" not in metadata["emotions"], "EMO_UNKNOWN should not be in list"

    def test_parse_emo_unkown_typo_maps_to_neutral(self):
        """
        Verify EMO_UNKOWN (typo) is mapped to NEUTRAL.

        SenseVoice model has a typo in EMO_UNKNOWN output.
        """
        from audio_pipeline.services.emotion_service import EmotionService

        service = EmotionService.get_instance()

        text = "<|EMO_UNKOWN|>Typo in model output."
        _, metadata = service.parse_tags(text)

        assert "NEUTRAL" in metadata["emotions"], "EMO_UNKOWN typo should map to NEUTRAL"
        assert "EMO_UNKOWN" not in metadata["emotions"], "EMO_UNKOWN should not be in list"

    def test_parse_emo_unknown_with_other_emotions(self):
        """
        Verify EMO_UNKNOWN handling with other emotions.

        Other emotions should be preserved while EMO_UNKNOWN maps to NEUTRAL.
        """
        from audio_pipeline.services.emotion_service import EmotionService

        service = EmotionService.get_instance()

        text = "<|HAPPY|>Happy part. <|EMO_UNKNOWN|>Unknown part. <|SAD|>Sad part."
        _, metadata = service.parse_tags(text)

        assert "HAPPY" in metadata["emotions"], "HAPPY should be preserved"
        assert "SAD" in metadata["emotions"], "SAD should be preserved"
        assert "NEUTRAL" in metadata["emotions"], "EMO_UNKNOWN should become NEUTRAL"


# =============================================================================
# Test Class: Audio Event Parsing
# =============================================================================


class TestAudioEventParsing:
    """Tests for audio event tag extraction."""

    def test_parse_speech_event(self):
        """
        Verify parsing of Speech event tag.

        Should correctly identify Speech event.
        """
        from audio_pipeline.services.emotion_service import EmotionService

        service = EmotionService.get_instance()

        text = "<|Speech|>Normal talking here."
        _, metadata = service.parse_tags(text)

        assert "Speech" in metadata["audio_events"], "Should detect Speech event"

    def test_parse_bgm_event(self):
        """
        Verify parsing of BGM (Background Music) event tag.

        Should correctly identify BGM event.
        """
        from audio_pipeline.services.emotion_service import EmotionService

        service = EmotionService.get_instance()

        text = "<|BGM|>Music playing in background."
        _, metadata = service.parse_tags(text)

        assert "BGM" in metadata["audio_events"], "Should detect BGM event"

    def test_parse_applause_event(self):
        """
        Verify parsing of Applause event tag.

        Should correctly identify Applause event.
        """
        from audio_pipeline.services.emotion_service import EmotionService

        service = EmotionService.get_instance()

        text = "<|Applause|>Clapping sounds."
        _, metadata = service.parse_tags(text)

        assert "Applause" in metadata["audio_events"], "Should detect Applause event"

    def test_parse_laughter_event(self):
        """
        Verify parsing of Laughter event tag.

        Should correctly identify Laughter event.
        """
        from audio_pipeline.services.emotion_service import EmotionService

        service = EmotionService.get_instance()

        text = "<|Laughter|>Ha ha ha!"
        _, metadata = service.parse_tags(text)

        assert "Laughter" in metadata["audio_events"], "Should detect Laughter event"

    def test_parse_cry_event(self):
        """
        Verify parsing of Cry event tag.

        Should correctly identify Cry event.
        """
        from audio_pipeline.services.emotion_service import EmotionService

        service = EmotionService.get_instance()

        text = "<|Cry|>Sobbing sounds."
        _, metadata = service.parse_tags(text)

        assert "Cry" in metadata["audio_events"], "Should detect Cry event"

    def test_parse_cough_event(self):
        """
        Verify parsing of Cough event tag.

        Should correctly identify Cough event.
        """
        from audio_pipeline.services.emotion_service import EmotionService

        service = EmotionService.get_instance()

        text = "<|Cough|>*cough cough*"
        _, metadata = service.parse_tags(text)

        assert "Cough" in metadata["audio_events"], "Should detect Cough event"

    def test_parse_sneeze_event(self):
        """
        Verify parsing of Sneeze event tag.

        Should correctly identify Sneeze event.
        """
        from audio_pipeline.services.emotion_service import EmotionService

        service = EmotionService.get_instance()

        text = "<|Sneeze|>*achoo*"
        _, metadata = service.parse_tags(text)

        assert "Sneeze" in metadata["audio_events"], "Should detect Sneeze event"

    def test_parse_breath_event(self):
        """
        Verify parsing of Breath event tag.

        Should correctly identify Breath event.
        """
        from audio_pipeline.services.emotion_service import EmotionService

        service = EmotionService.get_instance()

        text = "<|Breath|>*breathing heavily*"
        _, metadata = service.parse_tags(text)

        assert "Breath" in metadata["audio_events"], "Should detect Breath event"

    def test_parse_multiple_events(self):
        """
        Verify parsing of multiple audio event tags.

        Should correctly identify all events in text.
        """
        from audio_pipeline.services.emotion_service import EmotionService

        service = EmotionService.get_instance()

        text = "<|Speech|>Talking <|Laughter|>laughing <|Cough|>coughing"
        _, metadata = service.parse_tags(text)

        assert "Speech" in metadata["audio_events"]
        assert "Laughter" in metadata["audio_events"]
        assert "Cough" in metadata["audio_events"]

    def test_parse_no_events(self):
        """
        Verify handling of text with no event tags.

        Should return empty audio_events list.
        """
        from audio_pipeline.services.emotion_service import EmotionService

        service = EmotionService.get_instance()

        text = "Just plain text without any event tags."
        _, metadata = service.parse_tags(text)

        assert metadata["audio_events"] == [], "Should return empty list for no events"


# =============================================================================
# Test Class: Language Tag Parsing
# =============================================================================


class TestLanguageTagParsing:
    """Tests for language tag extraction."""

    def test_parse_english_language(self):
        """
        Verify parsing of English language tag.

        Should correctly identify 'en' language.
        """
        from audio_pipeline.services.emotion_service import EmotionService

        service = EmotionService.get_instance()

        text = "<|en|>English text here."
        _, metadata = service.parse_tags(text)

        assert metadata["language"] == "en", "Should detect English language"

    def test_parse_chinese_language(self):
        """
        Verify parsing of Chinese language tag.

        Should correctly identify 'zh' language.
        """
        from audio_pipeline.services.emotion_service import EmotionService

        service = EmotionService.get_instance()

        text = "<|zh|>Chinese text here."
        _, metadata = service.parse_tags(text)

        assert metadata["language"] == "zh", "Should detect Chinese language"

    def test_parse_japanese_language(self):
        """
        Verify parsing of Japanese language tag.

        Should correctly identify 'ja' language.
        """
        from audio_pipeline.services.emotion_service import EmotionService

        service = EmotionService.get_instance()

        text = "<|ja|>Japanese text here."
        _, metadata = service.parse_tags(text)

        assert metadata["language"] == "ja", "Should detect Japanese language"

    def test_parse_korean_language(self):
        """
        Verify parsing of Korean language tag.

        Should correctly identify 'ko' language.
        """
        from audio_pipeline.services.emotion_service import EmotionService

        service = EmotionService.get_instance()

        text = "<|ko|>Korean text here."
        _, metadata = service.parse_tags(text)

        assert metadata["language"] == "ko", "Should detect Korean language"

    def test_parse_spanish_language(self):
        """
        Verify parsing of Spanish language tag.

        Should correctly identify 'es' language.
        """
        from audio_pipeline.services.emotion_service import EmotionService

        service = EmotionService.get_instance()

        text = "<|es|>Spanish text here."
        _, metadata = service.parse_tags(text)

        assert metadata["language"] == "es", "Should detect Spanish language"

    def test_parse_no_language(self):
        """
        Verify handling of text with no language tag.

        Should return None for language.
        """
        from audio_pipeline.services.emotion_service import EmotionService

        service = EmotionService.get_instance()

        text = "Text without language tag."
        _, metadata = service.parse_tags(text)

        assert metadata["language"] is None, "Should return None for no language"


# =============================================================================
# Test Class: Clean Text Extraction
# =============================================================================


class TestCleanTextExtraction:
    """Tests for clean text extraction (tag removal)."""

    def test_get_clean_text_removes_all_tags(self):
        """
        Verify all tags are removed from text.

        Should return text with no tag markers.
        """
        from audio_pipeline.services.emotion_service import EmotionService

        service = EmotionService.get_instance()

        text = "<|en|><|HAPPY|><|Speech|>Hello world!"
        clean_text = service.get_clean_text(text)

        assert clean_text == "Hello world!", "All tags should be removed"
        assert "<|" not in clean_text, "No tag markers should remain"
        assert "|>" not in clean_text, "No tag closers should remain"

    def test_get_clean_text_preserves_content(self):
        """
        Verify content is preserved when removing tags.

        Text content should be unchanged except for tag removal.
        """
        from audio_pipeline.services.emotion_service import EmotionService

        service = EmotionService.get_instance()

        text = "<|en|>This is important <|ANGRY|> content that should <|Speech|> be preserved."
        clean_text = service.get_clean_text(text)

        assert "important" in clean_text
        assert "content" in clean_text
        assert "preserved" in clean_text

    def test_get_clean_text_handles_empty_input(self):
        """
        Verify handling of empty input.

        Should return empty string for empty input.
        """
        from audio_pipeline.services.emotion_service import EmotionService

        service = EmotionService.get_instance()

        clean_text = service.get_clean_text("")

        assert clean_text == "", "Should return empty string for empty input"

    def test_get_clean_text_handles_only_tags(self):
        """
        Verify handling of text with only tags.

        Should return empty string when text is only tags.
        """
        from audio_pipeline.services.emotion_service import EmotionService

        service = EmotionService.get_instance()

        text = "<|en|><|NEUTRAL|><|Speech|>"
        clean_text = service.get_clean_text(text)

        assert clean_text == "", "Should return empty string for tags-only text"


# =============================================================================
# Test Class: Parse Tags with Keep Emotion Tags
# =============================================================================


class TestParseTagsWithKeepEmotionTags:
    """Tests for parse_tags with keep_emotion_tags parameter."""

    def test_parse_tags_keep_emotions_true(self):
        """
        Verify emotion tags are formatted when keep_emotion_tags=True.

        Emotion tags should be converted to [EMOTION] format.
        """
        from audio_pipeline.services.emotion_service import EmotionService

        service = EmotionService.get_instance()

        text = "<|en|><|HAPPY|><|Speech|>Hello world!"
        processed_text, _ = service.parse_tags(text, keep_emotion_tags=True)

        assert "[HAPPY]" in processed_text, "Emotion should be formatted as [HAPPY]"
        assert "<|en|>" not in processed_text, "Language tag should be removed"
        assert "<|Speech|>" not in processed_text, "Event tag should be removed"

    def test_parse_tags_keep_emotions_false(self):
        """
        Verify all tags are removed when keep_emotion_tags=False.

        All tags including emotions should be removed.
        """
        from audio_pipeline.services.emotion_service import EmotionService

        service = EmotionService.get_instance()

        text = "<|en|><|HAPPY|><|Speech|>Hello world!"
        processed_text, _ = service.parse_tags(text, keep_emotion_tags=False)

        assert "[HAPPY]" not in processed_text, "Formatted emotion should not appear"
        assert "<|HAPPY|>" not in processed_text, "Raw emotion tag should not appear"
        assert processed_text == "Hello world!", "Only content should remain"


# =============================================================================
# Test Class: Primary Emotion
# =============================================================================


class TestPrimaryEmotion:
    """Tests for primary emotion determination."""

    def test_get_primary_emotion_single(self):
        """
        Verify primary emotion with single emotion.

        Should return the only emotion in list.
        """
        from audio_pipeline.services.emotion_service import EmotionService

        service = EmotionService.get_instance()

        primary = service.get_primary_emotion(["HAPPY"])

        assert primary == "HAPPY", "Should return single emotion"

    def test_get_primary_emotion_multiple(self):
        """
        Verify primary emotion with multiple emotions.

        Should return the first emotion in list.
        """
        from audio_pipeline.services.emotion_service import EmotionService

        service = EmotionService.get_instance()

        primary = service.get_primary_emotion(["HAPPY", "SAD", "ANGRY"])

        assert primary == "HAPPY", "Should return first emotion"

    def test_get_primary_emotion_empty(self):
        """
        Verify primary emotion with empty list.

        Should return NEUTRAL as default.
        """
        from audio_pipeline.services.emotion_service import EmotionService

        service = EmotionService.get_instance()

        primary = service.get_primary_emotion([])

        assert primary == "NEUTRAL", "Should return NEUTRAL for empty list"


# =============================================================================
# Test Class: Emotion Merging
# =============================================================================


class TestEmotionMerging:
    """Tests for merging emotions from multiple chunks."""

    def test_merge_emotions_single_list(self):
        """
        Verify merging of single emotion list.

        Should return the same emotions.
        """
        from audio_pipeline.services.emotion_service import EmotionService

        service = EmotionService.get_instance()

        merged = service.merge_emotions([["HAPPY", "SAD"]])

        assert "HAPPY" in merged
        assert "SAD" in merged

    def test_merge_emotions_multiple_lists(self):
        """
        Verify merging of multiple emotion lists.

        Should return all unique emotions.
        """
        from audio_pipeline.services.emotion_service import EmotionService

        service = EmotionService.get_instance()

        merged = service.merge_emotions([
            ["HAPPY", "SAD"],
            ["ANGRY", "NEUTRAL"],
            ["HAPPY", "SURPRISED"]
        ])

        assert "HAPPY" in merged, "HAPPY should be in merged"
        assert "SAD" in merged, "SAD should be in merged"
        assert "ANGRY" in merged, "ANGRY should be in merged"
        assert "NEUTRAL" in merged, "NEUTRAL should be in merged"
        assert "SURPRISED" in merged, "SURPRISED should be in merged"
        assert len(merged) == 5, "Should have 5 unique emotions"

    def test_merge_emotions_empty_lists(self):
        """
        Verify merging with empty lists.

        Should handle empty lists gracefully.
        """
        from audio_pipeline.services.emotion_service import EmotionService

        service = EmotionService.get_instance()

        merged = service.merge_emotions([[], [], []])

        assert merged == [], "Should return empty list for empty inputs"

    def test_merge_emotions_removes_duplicates(self):
        """
        Verify duplicates are removed when merging.

        Same emotion in multiple lists should appear once.
        """
        from audio_pipeline.services.emotion_service import EmotionService

        service = EmotionService.get_instance()

        merged = service.merge_emotions([
            ["HAPPY", "HAPPY"],
            ["HAPPY"],
            ["HAPPY", "SAD"]
        ])

        assert merged.count("HAPPY") <= 1 or (merged.count("HAPPY") == 1), \
            "HAPPY should appear at most once in merged list"


# =============================================================================
# Test Class: Event Merging
# =============================================================================


class TestEventMerging:
    """Tests for merging events from multiple chunks."""

    def test_merge_events_single_list(self):
        """
        Verify merging of single event list.

        Should return the same events.
        """
        from audio_pipeline.services.emotion_service import EmotionService

        service = EmotionService.get_instance()

        merged = service.merge_events([["Speech", "Laughter"]])

        assert "Speech" in merged
        assert "Laughter" in merged

    def test_merge_events_multiple_lists(self):
        """
        Verify merging of multiple event lists.

        Should return all unique events.
        """
        from audio_pipeline.services.emotion_service import EmotionService

        service = EmotionService.get_instance()

        merged = service.merge_events([
            ["Speech"],
            ["Laughter", "BGM"],
            ["Speech", "Cough"]
        ])

        assert "Speech" in merged
        assert "Laughter" in merged
        assert "BGM" in merged
        assert "Cough" in merged

    def test_merge_events_empty_lists(self):
        """
        Verify merging with empty lists.

        Should handle empty lists gracefully.
        """
        from audio_pipeline.services.emotion_service import EmotionService

        service = EmotionService.get_instance()

        merged = service.merge_events([[], [], []])

        assert merged == [], "Should return empty list for empty inputs"


# =============================================================================
# Test Class: Raw Transcription Preservation
# =============================================================================


class TestRawTranscriptionPreservation:
    """Tests for raw transcription preservation in metadata."""

    def test_raw_transcription_preserved(self):
        """
        Verify raw transcription is preserved in metadata.

        Original text with all tags should be stored in metadata.
        """
        from audio_pipeline.services.emotion_service import EmotionService

        service = EmotionService.get_instance()

        text = "<|en|><|HAPPY|><|Speech|>Hello world!"
        _, metadata = service.parse_tags(text)

        assert metadata["raw_transcription"] == text, "Raw transcription should be preserved"


# =============================================================================
# Test Class: Global Function
# =============================================================================


class TestGlobalFunction:
    """Tests for get_emotion_service global function."""

    def test_get_emotion_service(self):
        """
        Verify get_emotion_service returns singleton instance.

        Should return the same instance as get_instance().
        """
        from audio_pipeline.services.emotion_service import (
            get_emotion_service, EmotionService
        )

        EmotionService._instance = None

        service = get_emotion_service()
        instance = EmotionService.get_instance()

        assert service is instance, "Should return singleton instance"

        EmotionService._instance = None


# =============================================================================
# Test Class: Complete Parsing Scenarios
# =============================================================================


class TestCompleteParsingScenarios:
    """Tests for complete parsing scenarios with multiple tag types."""

    def test_full_sensevoice_output(self):
        """
        Verify parsing of complete SenseVoice output.

        Should correctly parse all tag types in realistic output.
        """
        from audio_pipeline.services.emotion_service import EmotionService

        service = EmotionService.get_instance()

        # Realistic SenseVoice output
        text = ("<|en|><|NEUTRAL|><|Speech|>Hello, this is customer support. "
                "<|HAPPY|>How can I help you today? "
                "<|ANGRY|>I've been waiting for 30 minutes! "
                "<|NEUTRAL|><|Speech|>I apologize for the wait.")

        processed_text, metadata = service.parse_tags(text)

        # Verify emotions
        assert "NEUTRAL" in metadata["emotions"]
        assert "HAPPY" in metadata["emotions"]
        assert "ANGRY" in metadata["emotions"]

        # Verify events
        assert "Speech" in metadata["audio_events"]

        # Verify language
        assert metadata["language"] == "en"

        # Verify raw transcription
        assert metadata["raw_transcription"] == text

    def test_parse_with_special_characters(self):
        """
        Verify parsing with special characters in text.

        Tags should be parsed correctly even with special characters nearby.
        """
        from audio_pipeline.services.emotion_service import EmotionService

        service = EmotionService.get_instance()

        text = "<|en|><|HAPPY|>$100 payment! #success @customer"
        processed_text, metadata = service.parse_tags(text, keep_emotion_tags=False)

        assert "HAPPY" in metadata["emotions"]
        assert "$100" in processed_text
        assert "#success" in processed_text
        assert "@customer" in processed_text

    def test_parse_with_punctuation(self):
        """
        Verify parsing with various punctuation.

        Punctuation should be preserved in clean text.
        """
        from audio_pipeline.services.emotion_service import EmotionService

        service = EmotionService.get_instance()

        text = "<|en|><|NEUTRAL|>Hello! How are you? I'm fine, thanks."
        clean_text = service.get_clean_text(text)

        assert "!" in clean_text
        assert "?" in clean_text
        assert "," in clean_text
        assert "'" in clean_text
