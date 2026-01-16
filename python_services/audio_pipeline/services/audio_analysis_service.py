"""
Audio Analysis Service

Main orchestration service that coordinates all audio analysis components.
Uses asyncio.gather for parallel execution of independent operations.
"""

import os
import asyncio
from typing import Optional, Dict, Any, List
from pathlib import Path

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
from audio_pipeline.services.content_analysis_service import (
    ContentAnalysisService,
    get_content_analysis_service,
)
from audio_pipeline.services.metadata_service import (
    MetadataService,
    get_metadata_service,
)
from audio_pipeline.services.database_service import (
    DatabaseService,
    get_database_service,
)
from audio_pipeline.services.diarization_service import (
    DiarizationService,
    get_diarization_service,
)


class AudioAnalysisService:
    """
    Main service for audio analysis.

    Orchestrates:
    - Transcription (SenseVoice)
    - Emotion/event detection
    - LLM summarization
    - Call content analysis
    - Metadata parsing
    - Database lookups
    """

    _instance: Optional['AudioAnalysisService'] = None

    def __init__(self):
        self.transcription_service = get_transcription_service()
        self.emotion_service = get_emotion_service()
        self.summarization_service = get_summarization_service()
        self.content_analysis_service = get_content_analysis_service()
        self.metadata_service = get_metadata_service()
        self.database_service = get_database_service()
        self.diarization_service = get_diarization_service()

    @classmethod
    def get_instance(cls) -> 'AudioAnalysisService':
        """Get singleton instance"""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    async def initialize(self):
        """Initialize underlying services"""
        await self.transcription_service.initialize()

    @property
    def is_initialized(self) -> bool:
        """Check if service is initialized"""
        return self.transcription_service.is_initialized

    def get_supported_formats(self) -> List[str]:
        """Get list of supported audio formats"""
        return [".wav", ".mp3", ".flac", ".m4a", ".ogg"]

    def validate_format(self, filename: str) -> bool:
        """Check if file format is supported"""
        ext = Path(filename).suffix.lower()
        return ext in self.get_supported_formats()

    def get_max_file_size_mb(self) -> int:
        """Get maximum file size in MB"""
        return 100

    async def analyze_audio(
        self,
        audio_path: str,
        language: Optional[str] = "auto",
        original_filename: Optional[str] = None,
        progress_callback=None,
        enable_diarization: bool = True,
        min_speakers: Optional[int] = 2,
        max_speakers: Optional[int] = 2
    ) -> Dict[str, Any]:
        """
        Analyze audio file - main entry point.

        Uses parallel execution for independent operations:
        - Phase 1: Transcription (blocking)
        - Phase 2: Diarization + Summarization + Metadata parsing (parallel)
        - Phase 3: Staff lookup + Customer lookup (parallel, after metadata)
        - Phase 4: Content analysis (after staff lookup)

        Args:
            audio_path: Path to audio file
            language: Language code or 'auto'
            original_filename: Original filename for metadata parsing
            progress_callback: Optional async callback(step, message)
            enable_diarization: Enable speaker diarization (default True)
            min_speakers: Minimum expected speakers (default 2 for phone calls)
            max_speakers: Maximum expected speakers (default 2 for phone calls)

        Returns:
            Complete analysis result dict
        """
        if not self.is_initialized:
            await self.initialize()

        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        try:
            # ================================================================
            # Phase 1: Transcription (blocking - all other steps depend on this)
            # ================================================================
            transcription_result = await self.transcription_service.transcribe(
                audio_path, language, progress_callback
            )

            raw_text = transcription_result.get("raw_text", "")
            audio_metadata = transcription_result.get("audio_metadata", {})
            duration = audio_metadata.get("duration_seconds", 0)

            # Parse emotions and events (synchronous, fast)
            # Keep emotion tags in the processed text for management view
            text_with_emotions, parsed_metadata = self.emotion_service.parse_tags(raw_text, keep_emotion_tags=True)
            # Also get clean text without any tags for LLM analysis
            clean_text = self.emotion_service.get_clean_text(raw_text)
            primary_emotion = self.emotion_service.get_primary_emotion(
                parsed_metadata.get("emotions", [])
            )

            if progress_callback:
                text_len = len(clean_text) if clean_text else 0
                await progress_callback("transcribe_done", f"Transcription complete ({text_len} characters)")

            # Parse filename for call metadata (synchronous, fast)
            filename_to_parse = original_filename if original_filename else audio_path
            call_metadata = self.metadata_service.parse_filename(filename_to_parse)
            if call_metadata.get("parsed"):
                print(f"Parsed call metadata: date={call_metadata['call_date']}, ext={call_metadata['extension']}")
                if progress_callback:
                    await progress_callback(
                        "metadata",
                        f"Parsed call: {call_metadata['direction']} from {call_metadata['phone_number']}"
                    )

            # ================================================================
            # Phase 2: Parallel operations (independent of each other)
            # - Diarization
            # - Summarization
            # - Staff lookup
            # - Customer lookup
            # ================================================================
            async def run_diarization():
                """Run speaker diarization."""
                if not enable_diarization or duration < 5:
                    return [], None, {}

                try:
                    if progress_callback:
                        await progress_callback("diarize", "Running speaker diarization...")

                    segments = await self.diarization_service.diarize(
                        audio_path,
                        min_speakers=min_speakers,
                        max_speakers=max_speakers,
                        progress_callback=progress_callback
                    )

                    if segments:
                        # Use emotion-tagged text for speaker transcription
                        speaker_trans, _ = self.diarization_service.merge_transcription_with_diarization(
                            text_with_emotions, segments, duration
                        )
                        stats = self.diarization_service.get_speaker_statistics(segments)

                        print(f"Speaker diarization complete: {len(segments)} segments")
                        if progress_callback:
                            await progress_callback(
                                "diarize_done",
                                f"Identified {len(stats)} speakers in {len(segments)} segments"
                            )
                        return segments, speaker_trans, stats

                    return [], None, {}
                except Exception as e:
                    print(f"Speaker diarization failed (non-critical): {e}")
                    if progress_callback:
                        await progress_callback("diarize_error", f"Diarization skipped: {str(e)[:50]}")
                    return [], None, {}

            async def run_summarization():
                """Generate summary for long transcriptions."""
                if not self.summarization_service.should_summarize(duration):
                    return None

                if progress_callback:
                    await progress_callback("summary", "Generating transcription summary...")
                return await self.summarization_service.generate_summary(
                    clean_text,
                    parsed_metadata.get("emotions", []),
                    duration
                )

            async def run_staff_lookup():
                """Look up staff name from extension."""
                if not call_metadata.get("extension"):
                    return None
                try:
                    staff = await self.database_service.lookup_staff_by_extension(
                        call_metadata["extension"]
                    )
                    if staff:
                        print(f"Staff member identified: {staff} (ext: {call_metadata['extension']})")
                    return staff
                except Exception as e:
                    print(f"Staff lookup failed (non-critical): {e}")
                    return None

            async def run_customer_lookup():
                """Look up customer by phone number."""
                if not (call_metadata.get("phone_number") and call_metadata.get("direction") == "Incoming"):
                    return {"found": False, "customer_name": None, "company_name": None}

                try:
                    customer_info = await self.database_service.lookup_customer_by_phone(
                        call_metadata["phone_number"]
                    )
                    if customer_info and customer_info.get("found"):
                        print(f"Customer identified: {customer_info.get('customer_name')}")
                        if progress_callback:
                            await progress_callback(
                                "customer",
                                f"Customer identified: {customer_info.get('customer_name')}"
                            )
                        return customer_info
                except Exception as e:
                    print(f"Customer lookup failed (non-critical): {e}")

                return {"found": False, "customer_name": None, "company_name": None}

            # Run Phase 2 operations in parallel
            print("Starting parallel Phase 2 operations: diarization, summarization, lookups...")
            phase2_results = await asyncio.gather(
                run_diarization(),
                run_summarization(),
                run_staff_lookup(),
                run_customer_lookup(),
                return_exceptions=True
            )

            # Unpack Phase 2 results with error handling
            diarization_result = phase2_results[0] if not isinstance(phase2_results[0], Exception) else ([], None, {})
            transcription_summary = phase2_results[1] if not isinstance(phase2_results[1], Exception) else None
            staff_name = phase2_results[2] if not isinstance(phase2_results[2], Exception) else None
            customer_lookup = phase2_results[3] if not isinstance(phase2_results[3], Exception) else {"found": False, "customer_name": None, "company_name": None}

            speaker_segments, speaker_transcription, speaker_statistics = diarization_result

            # Log any exceptions from Phase 2
            for i, result in enumerate(phase2_results):
                if isinstance(result, Exception):
                    step_names = ["diarization", "summarization", "staff_lookup", "customer_lookup"]
                    print(f"Phase 2 {step_names[i]} failed with exception: {result}")

            # ================================================================
            # Phase 2.5: LLM-based speaker separation (fallback if pyannote unavailable)
            # ================================================================
            llm_speaker_separation = None
            if not speaker_segments and clean_text and len(clean_text.strip()) >= 50:
                print("Pyannote diarization unavailable, using LLM-based speaker separation...")
                if progress_callback:
                    await progress_callback("speaker_sep", "Separating speakers with LLM...")
                try:
                    llm_speaker_separation = await self.content_analysis_service.separate_speakers(
                        clean_text, duration, {"primary": primary_emotion, "detected": parsed_metadata.get("emotions", [])}
                    )
                    if llm_speaker_separation and llm_speaker_separation.get("segments"):
                        # Convert LLM segments to speaker_segments format
                        speaker_segments = []
                        for seg in llm_speaker_separation["segments"]:
                            from audio_pipeline.services.diarization_service import SpeakerSegment
                            speaker_segments.append(SpeakerSegment(
                                speaker=seg["speaker"],
                                start_time=0,
                                end_time=0,
                                text=seg["text"]
                            ))
                        speaker_transcription = llm_speaker_separation.get("formatted_transcript")
                        speaker_statistics = {
                            "Caller 1": {"segment_count": len([s for s in speaker_segments if s.speaker == "Caller 1"]), "word_count": 0},
                            "Caller 2": {"segment_count": len([s for s in speaker_segments if s.speaker == "Caller 2"]), "word_count": 0}
                        }
                        if progress_callback:
                            await progress_callback("speaker_sep_done", f"Identified {llm_speaker_separation.get('num_speakers', 0)} speakers")
                except Exception as e:
                    print(f"LLM speaker separation failed (non-critical): {e}")

            # ================================================================
            # Phase 3: Content analysis (depends on staff_name from Phase 2)
            # ================================================================
            call_content = {
                "subject": None,
                "outcome": None,
                "customer_name": None,
                "confidence": 0.0,
                "analysis_model": ""
            }
            if clean_text and len(clean_text.strip()) >= 50:
                print("Analyzing call content with LLM...")
                if progress_callback:
                    await progress_callback("llm", "Analyzing call content with LLM...")
                call_content = await self.content_analysis_service.analyze_call_content(
                    clean_text, duration, staff_name
                )
                if progress_callback:
                    subject = call_content.get("subject", "")
                    subject_short = subject[:40] + "..." if subject and len(subject) > 40 else subject
                    await progress_callback("llm_done", f"LLM analysis complete: {subject_short}")

            # Use text with emotion tags for the main transcription
            # Emotion tags are formatted as [HAPPY], [SAD], etc. for readability
            formatted_transcription = text_with_emotions
            print(f"Transcription includes emotion tags: {len(parsed_metadata.get('emotions', []))} emotions detected")

            # Build response
            # Use speaker-labeled transcription if available, otherwise use emotion-tagged text
            final_transcription = speaker_transcription if speaker_transcription else formatted_transcription

            return {
                "success": True,
                "transcription": final_transcription,
                "transcription_plain": formatted_transcription,
                "transcription_summary": transcription_summary,
                "raw_transcription": raw_text,
                "emotions": {
                    "primary": primary_emotion,
                    "detected": parsed_metadata.get("emotions", []),
                    "timestamps": []
                },
                "audio_events": {
                    "detected": parsed_metadata.get("audio_events", []),
                    "timestamps": []
                },
                "language": parsed_metadata.get("language") or language,
                "audio_metadata": audio_metadata,
                "call_metadata": call_metadata,
                "call_content": call_content,
                "customer_lookup": customer_lookup,
                "speaker_diarization": {
                    "enabled": enable_diarization,
                    "segments": [
                        {
                            "speaker": seg.speaker,
                            "start_time": seg.start_time,
                            "end_time": seg.end_time,
                            "text": seg.text
                        } for seg in speaker_segments
                    ] if speaker_segments else [],
                    "statistics": speaker_statistics,
                    "num_speakers": len(speaker_statistics) if speaker_statistics else 0
                }
            }

        except Exception as e:
            print(f"Audio analysis error: {e}")
            filename_to_parse = original_filename if original_filename else audio_path
            return {
                "success": False,
                "error": str(e),
                "transcription": "",
                "transcription_plain": "",
                "transcription_summary": None,
                "raw_transcription": "",
                "emotions": {"primary": "NEUTRAL", "detected": [], "timestamps": []},
                "audio_events": {"detected": [], "timestamps": []},
                "language": language,
                "audio_metadata": self.transcription_service.get_audio_metadata(audio_path) if os.path.exists(audio_path) else {},
                "call_metadata": self.metadata_service.parse_filename(filename_to_parse),
                "call_content": {"subject": None, "outcome": None, "customer_name": None, "confidence": 0.0, "analysis_model": ""},
                "customer_lookup": {"found": False, "customer_name": None, "company_name": None},
                "speaker_diarization": {"enabled": enable_diarization, "segments": [], "statistics": {}, "num_speakers": 0}
            }

    async def check_llm_available(self) -> bool:
        """Check if LLM is running and available"""
        try:
            import httpx
            llm_url = os.getenv("LLAMACPP_HOST", "http://localhost:8081")
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{llm_url}/v1/models")
                if response.status_code == 200:
                    data = response.json()
                    models = data.get('data', [])
                    return len(models) > 0
        except Exception:
            pass
        return False


def get_audio_analysis_service() -> AudioAnalysisService:
    """Get the singleton audio analysis service instance"""
    return AudioAnalysisService.get_instance()


# Backwards compatibility alias
def get_audio_service() -> AudioAnalysisService:
    """Backwards compatibility alias for get_audio_analysis_service"""
    return get_audio_analysis_service()
