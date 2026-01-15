"""
Summarization Service

Generates LLM-powered summaries for long transcriptions.
Now uses TracedLLMClient for automatic monitoring and tracing.
"""

import os
from typing import Optional, List

# Default summarization threshold (seconds)
DEFAULT_SUMMARIZATION_THRESHOLD = 120  # 2 minutes

# Try to import TracedLLMClient
try:
    from llm.integration import summarize_transcription, generate_text
    TRACED_LLM_AVAILABLE = True
except ImportError:
    TRACED_LLM_AVAILABLE = False
    print("Warning: llm.integration not available. Using fallback httpx.")
    try:
        import httpx
        HTTPX_AVAILABLE = True
    except ImportError:
        HTTPX_AVAILABLE = False
        print("Warning: httpx not installed. Transcription summarization will be disabled.")


class SummarizationService:
    """
    Service for generating LLM-powered summaries of transcriptions.

    Only generates summaries for transcriptions from audio > 2 minutes.
    Now uses TracedLLMClient for automatic monitoring.
    """

    _instance: Optional['SummarizationService'] = None

    def __init__(self):
        self.llm_url = os.getenv("LLAMACPP_HOST", "http://localhost:8081")
        self.threshold_seconds = DEFAULT_SUMMARIZATION_THRESHOLD

    @classmethod
    def get_instance(cls) -> 'SummarizationService':
        """Get singleton instance"""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def set_threshold(self, seconds: int):
        """Set the summarization threshold in seconds"""
        self.threshold_seconds = seconds

    def should_summarize(self, duration_seconds: float) -> bool:
        """
        Check if transcription should be summarized based on duration.

        Args:
            duration_seconds: Audio duration in seconds

        Returns:
            True if should summarize
        """
        return duration_seconds >= self.threshold_seconds

    async def generate_summary(
        self,
        transcription: str,
        emotions: List[str],
        duration_seconds: float,
        user_id: str = None,
        call_id: str = None
    ) -> Optional[str]:
        """
        Generate a summary for long transcriptions using LLM.

        Args:
            transcription: The clean transcription text
            emotions: List of detected emotions
            duration_seconds: Duration of the audio in seconds
            user_id: Optional user ID for tracing
            call_id: Optional call/audio file ID for tracing

        Returns:
            Summary string if successful, None otherwise
        """
        # Only summarize if duration exceeds threshold
        if not self.should_summarize(duration_seconds):
            return None

        if not transcription or len(transcription.strip()) < 100:
            return None

        emotions_str = ", ".join(emotions) if emotions else "neutral"

        # Try TracedLLMClient first
        if TRACED_LLM_AVAILABLE:
            return await self._generate_summary_traced(
                transcription, emotions_str, duration_seconds, user_id, call_id
            )

        # Fallback to direct httpx
        return await self._generate_summary_httpx(
            transcription, emotions_str, duration_seconds
        )

    async def _generate_summary_traced(
        self,
        transcription: str,
        emotions_str: str,
        duration_seconds: float,
        user_id: str = None,
        call_id: str = None
    ) -> Optional[str]:
        """Generate summary using TracedLLMClient."""
        prompt = f"""Summarize this customer support call transcription in 2-3 sentences.
Focus on: the main issue discussed, the resolution or outcome, and overall customer sentiment.

Detected emotions: {emotions_str}
Duration: {duration_seconds:.0f} seconds

Transcription:
{transcription[:4000]}

Provide a concise summary:"""

        try:
            response = await generate_text(
                prompt=prompt,
                operation="summarize_transcription",
                pipeline="audio",
                user_id=user_id,
                max_tokens=200,
                temperature=0.3,
                stop=["\n\n"],
                tags=["audio", "summarization", "transcription"],
                context_dict={"document_id": call_id} if call_id else None,
            )

            if response.success and response.text:
                summary = response.text.strip()
                print(f"Generated transcription summary ({len(summary)} chars) [TRACED]")
                return summary

        except Exception as e:
            print(f"Traced summarization failed: {e}")

        return None

    async def _generate_summary_httpx(
        self,
        transcription: str,
        emotions_str: str,
        duration_seconds: float
    ) -> Optional[str]:
        """Fallback: Generate summary using direct httpx."""
        if not HTTPX_AVAILABLE:
            print("httpx not available, skipping summarization")
            return None

        import httpx

        prompt = f"""Summarize this customer support call transcription in 2-3 sentences.
Focus on: the main issue discussed, the resolution or outcome, and overall customer sentiment.

Detected emotions: {emotions_str}
Duration: {duration_seconds:.0f} seconds

Transcription:
{transcription[:4000]}

Provide a concise summary:"""

        try:
            timeout_config = httpx.Timeout(
                connect=10.0,
                read=60.0,
                write=10.0,
                pool=10.0
            )

            async with httpx.AsyncClient(timeout=timeout_config) as client:
                response = await client.post(
                    f"{self.llm_url}/v1/completions",
                    json={
                        "prompt": prompt,
                        "max_tokens": 200,
                        "temperature": 0.3,
                        "stop": ["\n\n"]
                    }
                )
                response.raise_for_status()
                data = response.json()
                summary = data.get('choices', [{}])[0].get('text', '').strip()

                if summary:
                    print(f"Generated transcription summary ({len(summary)} chars) [HTTPX]")
                    return summary

        except Exception as e:
            print(f"Transcription summarization failed: {e}")

        return None

    async def format_transcript(
        self,
        transcription: str,
        user_id: str = None,
        call_id: str = None
    ) -> str:
        """
        Format a raw transcription using LLM to make it readable.

        Adds paragraph breaks, speaker identification where possible,
        and general formatting to avoid a wall of text.

        Args:
            transcription: The raw transcription text
            user_id: Optional user ID for tracing
            call_id: Optional call/audio file ID for tracing

        Returns:
            Formatted transcription string
        """
        if not transcription or len(transcription.strip()) < 50:
            return transcription

        # Try TracedLLMClient first
        if TRACED_LLM_AVAILABLE:
            return await self._format_transcript_traced(transcription, user_id, call_id)

        # Fallback to direct httpx
        return await self._format_transcript_httpx(transcription)

    async def _format_transcript_traced(
        self,
        transcription: str,
        user_id: str = None,
        call_id: str = None
    ) -> str:
        """Format transcript using TracedLLMClient."""
        text_to_format = transcription[:8000] if len(transcription) > 8000 else transcription

        prompt = f"""Format this phone call transcription to make it readable.

Instructions:
- Add paragraph breaks between different topics or when speakers change
- If you can identify different speakers (customer vs support), label them as "Customer:" and "Support:" on new lines
- Keep the original words and meaning, just improve formatting
- Add line breaks every 2-3 sentences to avoid walls of text
- Do NOT summarize or remove content - keep ALL the original text
- Do NOT add commentary or explanations

Raw transcription:
{text_to_format}

Formatted transcription:"""

        try:
            response = await generate_text(
                prompt=prompt,
                operation="format_transcript",
                pipeline="audio",
                user_id=user_id,
                max_tokens=4000,
                temperature=0.2,
                stop=["---", "```", "\n\nRaw transcription:", "\n\nInstructions:"],
                tags=["audio", "formatting", "transcription"],
                context_dict={"document_id": call_id} if call_id else None,
            )

            if response.success and response.text:
                content = response.text.strip()
                if len(content) > 50:
                    print(f"Formatted transcription ({len(content)} chars) [TRACED]")
                    return content

            print("LLM returned insufficient formatted content, using original")
            return transcription

        except Exception as e:
            print(f"Traced transcript formatting failed: {e}, using original")
            return transcription

    async def _format_transcript_httpx(self, transcription: str) -> str:
        """Fallback: Format transcript using direct httpx."""
        if not HTTPX_AVAILABLE:
            print("httpx not available, returning raw transcription")
            return transcription

        import httpx

        text_to_format = transcription[:8000] if len(transcription) > 8000 else transcription

        prompt = f"""Format this phone call transcription to make it readable.

Instructions:
- Add paragraph breaks between different topics or when speakers change
- If you can identify different speakers (customer vs support), label them as "Customer:" and "Support:" on new lines
- Keep the original words and meaning, just improve formatting
- Add line breaks every 2-3 sentences to avoid walls of text
- Do NOT summarize or remove content - keep ALL the original text
- Do NOT add commentary or explanations

Raw transcription:
{text_to_format}

Formatted transcription:"""

        try:
            timeout_config = httpx.Timeout(connect=10.0, read=180.0, write=10.0, pool=10.0)

            async with httpx.AsyncClient(timeout=timeout_config) as client:
                response = await client.post(
                    f"{self.llm_url}/v1/completions",
                    json={
                        "prompt": prompt,
                        "max_tokens": 4000,
                        "temperature": 0.2,
                        "stop": ["---", "```", "\n\nRaw transcription:", "\n\nInstructions:"]
                    }
                )
                response.raise_for_status()
                data = response.json()
                content = data.get('choices', [{}])[0].get('text', '').strip()

                if content and len(content) > 50:
                    print(f"Formatted transcription ({len(content)} chars) [HTTPX]")
                    return content
                else:
                    print("LLM returned insufficient formatted content, using original")
                    return transcription

        except httpx.TimeoutException:
            print("Transcript formatting timed out, using original")
            return transcription
        except Exception as e:
            print(f"Transcript formatting failed: {e}, using original")
            return transcription


def get_summarization_service() -> SummarizationService:
    """Get the singleton summarization service instance"""
    return SummarizationService.get_instance()
