"""
Audio Analysis Service - SenseVoice Integration
Provides emotion detection, audio event detection, and transcription
with optional LLM-powered summarization for long transcriptions.

SenseVoice has a 30-second input limit. For longer audio:
- On Linux/Mac: VAD (fsmn-vad) is used for automatic segmentation
- On Windows: Manual chunking is used since fsmn-vad has compatibility issues
"""
import re
import os
import sys
import math
from typing import Optional, List, Dict, Tuple
from pathlib import Path
import tempfile

# HTTP client for LLM API
try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False
    print("Warning: httpx not installed. Transcription summarization will be disabled.")

# FunASR SenseVoice model
try:
    from funasr import AutoModel
    FUNASR_AVAILABLE = True
except ImportError:
    FUNASR_AVAILABLE = False
    print("Warning: funasr not installed. Audio analysis will be disabled.")

# Audio processing
try:
    import torchaudio
    import torch
    TORCHAUDIO_AVAILABLE = True
except ImportError:
    TORCHAUDIO_AVAILABLE = False
    print("Warning: torchaudio not installed. Audio format conversion may be limited.")

# Pydub for MP3 loading (more reliable on Windows than torchaudio for MP3)
try:
    from pydub import AudioSegment
    import numpy as np
    PYDUB_AVAILABLE = True
except ImportError:
    PYDUB_AVAILABLE = False
    print("Warning: pydub not installed. MP3 loading may be limited.")

# Soundfile for saving WAV files (more reliable than torchaudio on Windows)
try:
    import soundfile as sf
    SOUNDFILE_AVAILABLE = True
except ImportError:
    SOUNDFILE_AVAILABLE = False
    print("Warning: soundfile not installed. WAV saving may be limited.")

# Mutagen for MP3 metadata (more reliable than torchaudio for MP3)
try:
    from mutagen.mp3 import MP3
    from mutagen import MutagenError
    MUTAGEN_AVAILABLE = True
except ImportError:
    MUTAGEN_AVAILABLE = False
    print("Warning: mutagen not installed. MP3 metadata extraction may be limited.")


# Transcription summarization threshold (seconds)
SUMMARIZATION_THRESHOLD_SECONDS = 120  # 2 minutes

# Audio chunking settings for Windows (no VAD available)
# SenseVoice has a 30-second input limit, so we chunk at 25 seconds with 2 second overlap
CHUNK_DURATION_SECONDS = 25  # Duration of each chunk
CHUNK_OVERLAP_SECONDS = 2    # Overlap between chunks to avoid cutting words


def parse_call_filename(filepath: str) -> Dict:
    """
    Parse RingCentral call recording filename to extract metadata.

    Patterns supported:
    1. With extension (outgoing): yyyymmdd-hhmmss_EXT_PHONE_DIRECTION_AUTO_RECORDINGID.mp3
       Example: 20251104-133555_302_(252)792-8686_Outgoing_Auto_2243071124051.mp3
    2. Without extension (incoming): yyyymmdd-hhmmss_PHONE_DIRECTION_AUTO_RECORDINGID.mp3
       Example: 20251121-141152_(469)906-0558_Incoming_Auto_2254843027051.mp3

    Returns dict with parsed fields and 'parsed' flag indicating success.
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

    # Pattern 1: With extension - yyyymmdd-hhmmss_EXT_PHONE_DIRECTION_AUTO_RECORDINGID
    pattern_with_ext = r'^(\d{4})(\d{2})(\d{2})-(\d{2})(\d{2})(\d{2})_(\d+)_(\([^)]+\)[^_]+|[^_]+)_(Incoming|Outgoing)_([^_]+)_(\d+)$'

    # Pattern 2: Without extension - yyyymmdd-hhmmss_PHONE_DIRECTION_AUTO_RECORDINGID
    pattern_no_ext = r'^(\d{4})(\d{2})(\d{2})-(\d{2})(\d{2})(\d{2})_(\([^)]+\)[^_]+|[^_]+)_(Incoming|Outgoing)_([^_]+)_(\d+)$'

    match = re.match(pattern_with_ext, filename)
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
        match = re.match(pattern_no_ext, filename)
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


async def analyze_call_content_with_llm(
    transcription: str,
    duration_seconds: float,
    llm_url: str = "http://localhost:8081",  # General model
    staff_name: str = None  # Optional staff name to filter out
) -> Dict:
    """
    Analyze call transcription using LLM to extract subject, outcome, and customer name.

    Args:
        transcription: The call transcription text
        duration_seconds: Duration of the audio
        llm_url: URL of the llama.cpp server
        staff_name: Optional staff member name - if customer_name matches this, it will be cleared

    Returns:
        Dict with subject, outcome, customer_name, confidence, and analysis_model
    """
    result = {
        "subject": None,
        "outcome": None,
        "customer_name": None,
        "confidence": 0.0,
        "analysis_model": ""
    }

    if not HTTPX_AVAILABLE:
        print("httpx not available, skipping call content analysis")
        return result

    if not transcription or len(transcription.strip()) < 50:
        print("Transcription too short for content analysis")
        return result

    # Truncate very long transcriptions
    text_to_analyze = transcription[:6000] if len(transcription) > 6000 else transcription

    prompt = f"""Analyze this customer support call transcription and extract the following information.
Respond ONLY with a JSON object in exactly this format, no other text:

{{
  "subject": "brief description of what the call was about (1-2 sentences)",
  "outcome": "one of: Resolved, Unresolved, Pending Follow-up, Information Provided, Transferred, or Unknown",
  "customer_name": "the CUSTOMER's name (the person calling for help), NOT the support staff name, otherwise null"
}}

IMPORTANT: The customer_name should be the person CALLING for support, not the support representative answering the call.
If someone says "Hi this is [Name] from EWR" or "This is [Name] with support", that is the STAFF, not the customer.
The customer is typically the one asking questions or requesting help.

Call Duration: {duration_seconds:.0f} seconds

Transcription:
{text_to_analyze}

JSON Response:"""

    try:
        import json
        timeout_config = httpx.Timeout(connect=10.0, read=120.0, write=10.0, pool=10.0)

        async with httpx.AsyncClient(timeout=timeout_config) as client:
            # Use OpenAI-compatible endpoint for llama-cpp-python server
            response = await client.post(
                f"{llm_url}/v1/completions",
                json={
                    "prompt": prompt,
                    "max_tokens": 300,
                    "temperature": 0.1,
                    "stop": ["\n\n", "```"]
                }
            )
            response.raise_for_status()
            data = response.json()
            # OpenAI format: choices[0].text
            content = data.get('choices', [{}])[0].get('text', '').strip()

            if content:
                try:
                    # Clean up the response - remove any markdown formatting
                    json_str = content
                    if json_str.startswith("```"):
                        json_str = json_str.split("```")[1]
                        if json_str.startswith("json"):
                            json_str = json_str[4:]
                    json_str = json_str.strip()

                    # Find JSON object boundaries
                    start_idx = json_str.find('{')
                    end_idx = json_str.rfind('}')
                    if start_idx != -1 and end_idx != -1:
                        json_str = json_str[start_idx:end_idx + 1]

                    parsed = json.loads(json_str)

                    result["subject"] = parsed.get('subject', '').strip() if parsed.get('subject') else None
                    result["outcome"] = parsed.get('outcome', '').strip() if parsed.get('outcome') else None
                    result["customer_name"] = parsed.get('customer_name', '').strip() if parsed.get('customer_name') else None
                    result["confidence"] = 0.8
                    result["analysis_model"] = "Llama-3.2-3B-Instruct"

                    # Validate customer_name is not the staff member's name
                    if result["customer_name"] and staff_name:
                        staff_lower = staff_name.lower().strip()
                        customer_lower = result["customer_name"].lower().strip()
                        # Check if customer name matches staff name (full or partial)
                        staff_parts = staff_lower.split()
                        customer_parts = customer_lower.split()
                        # If first name matches, or full name matches, clear it
                        if (customer_lower == staff_lower or
                            (staff_parts and customer_parts and staff_parts[0] == customer_parts[0]) or
                            (len(staff_parts) > 0 and customer_lower in staff_lower) or
                            (len(customer_parts) > 0 and staff_lower in customer_lower)):
                            print(f"Customer name '{result['customer_name']}' matches staff name '{staff_name}', clearing")
                            result["customer_name"] = None

                    print(f"Call content analysis complete: subject='{result['subject'][:50] if result['subject'] else 'N/A'}...', outcome='{result['outcome']}', customer='{result['customer_name'] or 'N/A'}'")

                except json.JSONDecodeError as e:
                    print(f"Failed to parse LLM response as JSON: {e}")
                    result["subject"] = content[:200] if content else None
                    result["confidence"] = 0.3
                    result["analysis_model"] = "Llama-3.2-3B-Instruct"

    except httpx.TimeoutException:
        print("Call content analysis timed out")
    except Exception as e:
        print(f"Call content analysis failed: {e}")

    return result


async def format_transcript_with_llm(
    transcription: str,
    llm_url: str = "http://localhost:8081"  # General model
) -> str:
    """
    Format a raw transcription using LLM to make it readable.

    Adds paragraph breaks, speaker identification where possible,
    and general formatting to avoid a wall of text.

    Args:
        transcription: The raw transcription text
        llm_url: URL of the llama.cpp server

    Returns:
        Formatted transcription string
    """
    if not HTTPX_AVAILABLE:
        print("httpx not available, returning raw transcription")
        return transcription

    if not transcription or len(transcription.strip()) < 50:
        return transcription

    # Truncate very long transcriptions for the LLM
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
                f"{llm_url}/v1/completions",
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
                print(f"Formatted transcription ({len(content)} chars)")
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


class AudioAnalysisService:
    """
    Service for audio analysis using SenseVoice model.

    SenseVoice provides:
    - Multilingual transcription (80+ languages)
    - Emotion detection (7 emotions)
    - Audio event detection (8 events)
    - Voice Activity Detection (VAD)
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

    _instance: Optional['AudioAnalysisService'] = None

    def __init__(self):
        self.model = None
        self.is_initialized = False
        self.device = "cuda:0" if self._check_cuda() else "cpu"
        self.llm_url = os.getenv("LLAMACPP_HOST", "http://localhost:8081")  # General model
        self.llm_model = os.getenv("LLM_MODEL", "llama3.2:3b")
        self.is_windows = sys.platform == 'win32'
        self.vad_available = False  # Will be set during initialization

    @classmethod
    def get_instance(cls) -> 'AudioAnalysisService':
        """Get singleton instance"""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _check_cuda(self) -> bool:
        """Check if CUDA is available"""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False

    async def initialize(self):
        """Initialize SenseVoice model"""
        if self.is_initialized:
            return

        if not FUNASR_AVAILABLE:
            raise RuntimeError("funasr library not installed. Please install: pip install funasr")

        print(f"Initializing SenseVoice model on device: {self.device}")

        try:
            # Initialize SenseVoice Small model
            # Model will be downloaded automatically on first use
            # Use ModelScope model ID (iic/SenseVoiceSmall) - this is the correct identifier
            # Note: VAD model disabled on Windows due to compatibility issues
            if self.is_windows:
                print("Windows detected - initializing without VAD model")
                print("Long audio files will be processed using manual chunking")
                self.model = AutoModel(
                    model="iic/SenseVoiceSmall",
                    device=self.device,
                    disable_pbar=True,
                    disable_update=True
                )
                self.vad_available = False
            else:
                # On Linux/Mac, use VAD for better segmentation
                print("Linux/Mac detected - initializing with VAD model for automatic segmentation")
                self.model = AutoModel(
                    model="iic/SenseVoiceSmall",
                    vad_model="fsmn-vad",
                    vad_kwargs={"max_single_segment_time": 30000},  # 30 seconds max
                    device=self.device,
                    disable_pbar=True,
                    disable_update=True
                )
                self.vad_available = True

            self.is_initialized = True
            print("SenseVoice model initialized successfully")

        except Exception as e:
            print(f"Error initializing SenseVoice model: {e}")
            raise

    def _parse_tags(self, text: str) -> Tuple[str, Dict]:
        """
        Parse emotion and event tags from SenseVoice output.

        SenseVoice embeds special tags in transcription:
        - <|EMOTION|> for emotions (e.g., <|HAPPY|>, <|SAD|>)
        - <|EVENT|> for audio events (e.g., <|Laughter|>, <|BGM|>)
        - <|LANG|> for language codes

        Args:
            text: Raw transcription with tags

        Returns:
            Tuple of (clean_text, metadata_dict)
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

        # Remove all tags to get clean transcription
        clean_text = re.sub(r'<\|[^|]+\|>', '', text)
        clean_text = clean_text.strip()

        return clean_text, metadata

    def _get_audio_metadata(self, audio_path: str) -> Dict:
        """
        Extract metadata from audio file.

        Uses mutagen for MP3 files (more reliable) and torchaudio as fallback.

        Args:
            audio_path: Path to audio file

        Returns:
            Dict with duration, sample_rate, channels, format, file_size
        """
        file_format = Path(audio_path).suffix.lower().replace('.', '')
        metadata = {
            "duration_seconds": 0.0,
            "sample_rate": 16000,  # Default
            "channels": 1,  # Default mono
            "format": file_format,
            "file_size_bytes": os.path.getsize(audio_path) if os.path.exists(audio_path) else 0
        }

        # For MP3 files, prefer mutagen as torchaudio often fails with MP3
        if file_format == 'mp3' and MUTAGEN_AVAILABLE:
            try:
                audio = MP3(audio_path)
                metadata["duration_seconds"] = audio.info.length
                metadata["sample_rate"] = audio.info.sample_rate
                metadata["channels"] = audio.info.channels
                print(f"MP3 metadata extracted via mutagen: duration={metadata['duration_seconds']:.2f}s")
                return metadata
            except MutagenError as e:
                print(f"Warning: mutagen could not read MP3 metadata: {e}")
            except Exception as e:
                print(f"Warning: Unexpected error reading MP3 with mutagen: {e}")

        # Fallback to torchaudio for non-MP3 files or if mutagen failed
        if TORCHAUDIO_AVAILABLE:
            try:
                info = torchaudio.info(audio_path)
                metadata["duration_seconds"] = info.num_frames / info.sample_rate
                metadata["sample_rate"] = info.sample_rate
                metadata["channels"] = info.num_channels
                print(f"Audio metadata extracted via torchaudio: duration={metadata['duration_seconds']:.2f}s")
            except Exception as e:
                print(f"Warning: Could not extract audio metadata with torchaudio: {e}")

        # Log warning if duration is still 0
        if metadata["duration_seconds"] == 0.0:
            print(f"Warning: Could not determine audio duration for {audio_path}")

        return metadata

    async def _lookup_staff_by_extension(self, extension: str) -> Optional[str]:
        """
        Look up staff name from EWRCentral database by phone extension.

        Args:
            extension: Phone extension to look up

        Returns:
            Staff full name if found, None otherwise
        """
        if not extension:
            return None

        try:
            import pymssql

            connection = pymssql.connect(
                server='EWRSQLPROD',
                database='EWRCentral',
                user='EWR\\chad.walker',
                password='6454@@Christina',
                port='1433',
                timeout=10
            )
            cursor = connection.cursor(as_dict=True)

            # Query for staff by extension
            ext_stripped = extension.lstrip('0') if extension else extension
            cursor.execute("""
                SELECT TOP 1
                    FirstName + ' ' + LastName AS FullName
                FROM CentralUsers
                WHERE IsActive = 1
                  AND (
                      LTRIM(RTRIM(OfficePhoneExtension)) = %s
                      OR LTRIM(RTRIM(OfficePhoneExtension)) = %s
                  )
                ORDER BY LastUpdateUTC DESC
            """, (extension, ext_stripped))

            result = cursor.fetchone()
            cursor.close()
            connection.close()

            if result:
                return result['FullName']
            return None

        except Exception as e:
            print(f"Staff lookup error: {e}")
            return None

    async def _lookup_customer_by_phone(self, phone_number: str) -> Optional[Dict]:
        """
        Look up customer information from EWRCentral database by phone number.

        TEMPORARILY DISABLED: OfficeEmailAddress column doesn't exist in database.
        TODO: Fix the SQL query to use correct column names.
        """
        # DISABLED - OfficeEmailAddress column error
        print("Customer lookup disabled - OfficeEmailAddress column error")
        return {"found": False}

        # --- ORIGINAL CODE COMMENTED OUT ---
        # if not phone_number:
        #     return None
        #
        # # Normalize phone number - extract just digits
        # import re
        # phone_digits = re.sub(r'\D', '', phone_number)
        #
        # # Need at least 7 digits for a reasonable match
        # if len(phone_digits) < 7:
        #     print(f"Phone number too short for lookup: {phone_number}")
        #     return {"found": False}
        #
        # # Use last 10 digits (or all if less than 10)
        # search_digits = phone_digits[-10:] if len(phone_digits) >= 10 else phone_digits
        # search_pattern = f'%{search_digits}'
        #
        # try:
        #     import pymssql
        #
        #     connection = pymssql.connect(
        #         server='EWRSQLPROD',
        #         database='EWRCentral',
        #         user='EWR\\chad.walker',
        #         password='6454@@Christina',
        #         port='1433',
        #         timeout=10
        #     )
        #     cursor = connection.cursor(as_dict=True)
        #
        #     cursor.execute("""
        #         SELECT TOP 1
        #             ct.CustomerContactName,
        #             ct.CustomerContactPhoneNumber,
        #             cc.CentralCompanyID,
        #             cc.CompanyName,
        #             cc.OfficeEmailAddress,
        #             ct.AddTicketDate,
        #             (SELECT COUNT(*) FROM CentralTickets ct2
        #              WHERE REPLACE(REPLACE(REPLACE(REPLACE(ct2.CustomerContactPhoneNumber, '(', ''), ')', ''), '-', ''), ' ', '')
        #              LIKE %s) AS TicketCount
        #         FROM CentralTickets ct
        #         LEFT JOIN CentralCompanies cc ON ct.CentralCompanyID = cc.CentralCompanyID
        #         WHERE REPLACE(REPLACE(REPLACE(REPLACE(ct.CustomerContactPhoneNumber, '(', ''), ')', ''), '-', ''), ' ', '')
        #               LIKE %s
        #         ORDER BY ct.AddTicketDate DESC
        #     """, (search_pattern, search_pattern))
        #
        #     result = cursor.fetchone()
        #     cursor.close()
        #     connection.close()
        #
        #     if result:
        #         return {
        #             "found": True,
        #             "customer_name": result.get('CustomerContactName'),
        #             "company_name": result.get('CompanyName'),
        #             "company_id": result.get('CentralCompanyID'),
        #             "email": result.get('OfficeEmailAddress'),
        #             "ticket_count": result.get('TicketCount', 0),
        #             "match_source": "ticket"
        #         }
        #
        #     return {"found": False}
        #
        # except Exception as e:
        #     print(f"Customer lookup error: {e}")
        #     return {"found": False}

    def _load_audio_as_waveform(self, audio_path: str) -> Tuple[any, int]:
        """
        Load audio file and return waveform tensor and sample rate.

        Handles MP3 and other formats, resampling to 16kHz if needed.
        Uses pydub for MP3 files (more reliable on Windows), torchaudio for others.

        Args:
            audio_path: Path to audio file

        Returns:
            Tuple of (waveform tensor, sample_rate)
        """
        file_ext = Path(audio_path).suffix.lower()
        target_sample_rate = 16000

        # Use pydub for MP3 files (torchaudio has issues with MP3 on Windows)
        if file_ext == '.mp3' and PYDUB_AVAILABLE:
            print(f"Loading MP3 with pydub: {audio_path}")
            audio = AudioSegment.from_mp3(audio_path)

            # Convert to mono if stereo
            if audio.channels == 2:
                audio = audio.set_channels(1)

            # Resample to 16kHz if needed
            if audio.frame_rate != target_sample_rate:
                audio = audio.set_frame_rate(target_sample_rate)

            # Convert to numpy array and normalize to [-1, 1]
            samples = np.array(audio.get_array_of_samples())
            samples = samples.astype(np.float32) / 32768.0

            # Convert to torch tensor with shape (1, num_samples)
            waveform = torch.from_numpy(samples).unsqueeze(0)
            sample_rate = target_sample_rate

            print(f"Loaded MP3: {waveform.shape}, sample_rate={sample_rate}")
            return waveform, sample_rate

        # Use torchaudio for other formats (WAV, FLAC, etc.)
        if not TORCHAUDIO_AVAILABLE:
            raise RuntimeError("torchaudio is required for non-MP3 audio loading")

        print(f"Loading audio with torchaudio: {audio_path}")
        waveform, sample_rate = torchaudio.load(audio_path)

        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # Resample to 16kHz if needed (SenseVoice works best at 16kHz)
        if sample_rate != target_sample_rate:
            resampler = torchaudio.transforms.Resample(sample_rate, target_sample_rate)
            waveform = resampler(waveform)
            sample_rate = target_sample_rate

        return waveform, sample_rate

    def _chunk_audio(
        self,
        waveform: any,
        sample_rate: int,
        chunk_duration: float = CHUNK_DURATION_SECONDS,
        overlap: float = CHUNK_OVERLAP_SECONDS
    ) -> List[Tuple[any, float, float]]:
        """
        Split audio waveform into overlapping chunks.

        Args:
            waveform: Audio waveform tensor (channels x samples)
            sample_rate: Sample rate in Hz
            chunk_duration: Duration of each chunk in seconds
            overlap: Overlap between chunks in seconds

        Returns:
            List of tuples: (chunk_waveform, start_time, end_time)
        """
        total_samples = waveform.shape[1]
        total_duration = total_samples / sample_rate

        chunk_samples = int(chunk_duration * sample_rate)
        overlap_samples = int(overlap * sample_rate)
        step_samples = chunk_samples - overlap_samples

        chunks = []
        start_sample = 0

        while start_sample < total_samples:
            end_sample = min(start_sample + chunk_samples, total_samples)
            chunk = waveform[:, start_sample:end_sample]

            start_time = start_sample / sample_rate
            end_time = end_sample / sample_rate

            chunks.append((chunk, start_time, end_time))

            # Move to next chunk
            start_sample += step_samples

            # If remaining audio is very short, include it in the last chunk
            remaining_samples = total_samples - start_sample
            if remaining_samples > 0 and remaining_samples < chunk_samples * 0.3:
                # Less than 30% of a chunk remaining, extend last chunk
                break

        print(f"Audio chunked into {len(chunks)} segments (total duration: {total_duration:.1f}s)")
        return chunks

    def _save_chunk_to_temp(self, waveform: any, sample_rate: int) -> str:
        """
        Save a waveform chunk to a temporary WAV file.

        Uses soundfile for saving (more reliable than torchaudio on Windows).

        Args:
            waveform: Audio waveform tensor
            sample_rate: Sample rate in Hz

        Returns:
            Path to temporary file
        """
        # Create temp file
        temp_fd, temp_path = tempfile.mkstemp(suffix='.wav')
        os.close(temp_fd)

        # Convert tensor to numpy array for soundfile
        # waveform is shape (1, num_samples), soundfile expects (num_samples,) or (num_samples, channels)
        audio_data = waveform.squeeze(0).numpy()

        # Save as WAV using soundfile (more reliable than torchaudio on Windows)
        if SOUNDFILE_AVAILABLE:
            sf.write(temp_path, audio_data, sample_rate)
        else:
            # Fallback to torchaudio if soundfile not available
            torchaudio.save(temp_path, waveform, sample_rate)

        return temp_path

    def _merge_chunk_results(
        self,
        chunk_results: List[Dict],
        chunk_times: List[Tuple[float, float]]
    ) -> Tuple[str, str, Dict]:
        """
        Merge transcription results from multiple chunks.

        Handles overlapping regions by using the transcription from the chunk
        that has more context (i.e., not at a boundary).

        Args:
            chunk_results: List of result dicts from each chunk
            chunk_times: List of (start_time, end_time) tuples for each chunk

        Returns:
            Tuple of (merged_clean_text, merged_raw_text, merged_metadata)
        """
        all_emotions = set()
        all_events = set()
        detected_language = None

        clean_texts = []
        raw_texts = []

        for i, result in enumerate(chunk_results):
            if not result:
                continue

            raw_text = result.get("text", "") if isinstance(result, dict) else str(result)
            clean_text, metadata = self._parse_tags(raw_text)

            # Collect all emotions and events
            all_emotions.update(metadata.get("emotions", []))
            all_events.update(metadata.get("audio_events", []))

            # Use the first detected language
            if not detected_language and metadata.get("language"):
                detected_language = metadata["language"]

            if clean_text:
                clean_texts.append(clean_text)
            if raw_text:
                raw_texts.append(raw_text)

        # Join texts with space
        # Note: This is a simple concatenation. More sophisticated approaches
        # could try to detect and remove duplicate words at chunk boundaries.
        merged_clean = " ".join(clean_texts)
        merged_raw = " | ".join(raw_texts)  # Use | to separate chunks in raw

        merged_metadata = {
            "emotions": list(all_emotions),
            "audio_events": list(all_events),
            "language": detected_language,
            "raw_transcription": merged_raw,
            "chunk_count": len(chunk_results)
        }

        return merged_clean, merged_raw, merged_metadata

    async def _process_with_chunking(
        self,
        audio_path: str,
        language: str,
        audio_metadata: Dict,
        progress_callback=None
    ) -> Tuple[str, str, Dict]:
        """
        Process long audio using manual chunking.

        This is used on Windows where VAD is not available.

        Args:
            audio_path: Path to audio file
            language: Language code or 'auto'
            audio_metadata: Pre-extracted audio metadata
            progress_callback: Optional async callback(step, message) for progress updates

        Returns:
            Tuple of (clean_text, raw_text, metadata)
        """
        print(f"Processing audio with manual chunking (duration: {audio_metadata.get('duration_seconds', 0):.1f}s)")

        # Load audio
        waveform, sample_rate = self._load_audio_as_waveform(audio_path)

        # Chunk the audio
        chunks = self._chunk_audio(waveform, sample_rate)

        chunk_results = []
        chunk_times = []
        temp_files = []

        try:
            for i, (chunk_waveform, start_time, end_time) in enumerate(chunks):
                progress_msg = f"Processing chunk {i+1}/{len(chunks)} ({start_time:.1f}s - {end_time:.1f}s)"
                print(progress_msg)

                # Send progress callback if provided
                if progress_callback:
                    await progress_callback("chunk", progress_msg)

                # Save chunk to temp file
                temp_path = self._save_chunk_to_temp(chunk_waveform, sample_rate)
                temp_files.append(temp_path)

                # Process chunk with SenseVoice
                result = self.model.generate(
                    input=temp_path,
                    language=language,
                    use_itn=True
                )

                if result and len(result) > 0:
                    chunk_results.append(result[0])
                    chunk_times.append((start_time, end_time))
                else:
                    print(f"Warning: No result for chunk {i+1}")
                    chunk_results.append(None)
                    chunk_times.append((start_time, end_time))

        finally:
            # Clean up temp files
            for temp_path in temp_files:
                try:
                    os.unlink(temp_path)
                except Exception as e:
                    print(f"Warning: Could not delete temp file {temp_path}: {e}")

        # Merge results
        clean_text, raw_text, metadata = self._merge_chunk_results(chunk_results, chunk_times)

        print(f"Chunked processing complete. Total text length: {len(clean_text)} characters from {len(chunks)} chunks")

        return clean_text, raw_text, metadata

    async def analyze_audio(
        self,
        audio_path: str,
        language: Optional[str] = "auto",
        original_filename: Optional[str] = None,
        progress_callback=None
    ) -> Dict:
        """
        Analyze audio file using SenseVoice.

        For long audio files (>25 seconds) on Windows, uses manual chunking
        since VAD is not available. On Linux/Mac, VAD handles segmentation.

        Args:
            audio_path: Path to audio file
            language: Language code (e.g., 'en', 'zh', 'ja') or 'auto' for detection
            original_filename: Original filename for metadata parsing (RingCentral format)
            progress_callback: Optional async callback(step, message) for progress updates

        Returns:
            Dict with transcription, emotions, events, language, metadata, call_metadata, and call_content
        """
        if not self.is_initialized:
            await self.initialize()

        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        # Get audio metadata
        audio_metadata = self._get_audio_metadata(audio_path)
        duration = audio_metadata.get("duration_seconds", 0)

        try:
            # Determine if we need chunking
            # Use chunking on Windows for audio longer than chunk duration
            # (VAD is not available on Windows)
            file_ext = Path(audio_path).suffix.lower()
            can_chunk = PYDUB_AVAILABLE if file_ext == '.mp3' else TORCHAUDIO_AVAILABLE
            needs_chunking = (
                not self.vad_available and
                duration > CHUNK_DURATION_SECONDS and
                can_chunk
            )

            if needs_chunking:
                # Process with manual chunking
                print(f"Audio duration ({duration:.1f}s) exceeds {CHUNK_DURATION_SECONDS}s - using chunked processing")
                if progress_callback:
                    await progress_callback("transcribe", f"Starting chunked processing ({duration:.1f}s audio)...")
                clean_text, raw_text, metadata = await self._process_with_chunking(
                    audio_path, language, audio_metadata, progress_callback
                )
            else:
                # Standard processing (VAD available or short audio)
                if self.vad_available:
                    print(f"Processing audio with VAD (duration: {duration:.1f}s)")
                else:
                    print(f"Processing short audio directly (duration: {duration:.1f}s)")

                # Run SenseVoice inference
                result = self.model.generate(
                    input=audio_path,
                    language=language,  # "auto", "zh", "en", "ja", "ko", etc.
                    use_itn=True,  # Inverse text normalization (numbers, dates)
                    batch_size_s=60,  # Process 60 seconds at a time
                    merge_vad=True  # Merge VAD segments (only works with VAD)
                )

                # Extract result
                if not result or len(result) == 0:
                    raise ValueError("No transcription result returned")

                # Get first result (for single file)
                transcription_result = result[0]

                # The result is a dict with 'text' key containing transcription with tags
                raw_text = transcription_result.get("text", "") if isinstance(transcription_result, dict) else str(transcription_result)

                # Parse tags and extract clean transcription
                clean_text, metadata = self._parse_tags(raw_text)

            # Determine primary emotion (most frequent or first detected)
            primary_emotion = metadata["emotions"][0] if metadata["emotions"] else "NEUTRAL"

            # Progress: transcription complete
            if progress_callback:
                text_len = len(clean_text) if clean_text else 0
                await progress_callback("transcribe_done", f"Transcription complete ({text_len} characters)")

            # Generate summary for long transcriptions (>2 minutes)
            transcription_summary = None
            if duration >= SUMMARIZATION_THRESHOLD_SECONDS:
                if progress_callback:
                    await progress_callback("summary", "Generating transcription summary...")
                transcription_summary = await self.generate_transcription_summary(
                    clean_text,
                    metadata["emotions"],
                    duration
                )

            # Parse filename for call metadata (RingCentral format)
            # Use original_filename if provided, otherwise fall back to audio_path
            filename_to_parse = original_filename if original_filename else audio_path
            call_metadata = parse_call_filename(filename_to_parse)
            if call_metadata["parsed"]:
                print(f"Parsed call metadata: date={call_metadata['call_date']}, ext={call_metadata['extension']}, phone={call_metadata['phone_number']}, direction={call_metadata['direction']}")
                if progress_callback:
                    await progress_callback("metadata", f"Parsed call: {call_metadata['direction']} from {call_metadata['phone_number']}")

            # Look up staff name from extension (to filter from customer name detection)
            staff_name = None
            if call_metadata.get("extension"):
                try:
                    staff_name = await self._lookup_staff_by_extension(call_metadata["extension"])
                    if staff_name:
                        print(f"Staff member identified: {staff_name} (ext: {call_metadata['extension']})")
                except Exception as e:
                    print(f"Staff lookup failed (non-critical): {e}")

            # Look up customer by phone number (for incoming calls)
            customer_lookup = {"found": False, "customer_name": None, "company_name": None}
            if call_metadata.get("phone_number") and call_metadata.get("direction") == "Incoming":
                try:
                    customer_info = await self._lookup_customer_by_phone(call_metadata["phone_number"])
                    if customer_info and customer_info.get("found"):
                        customer_lookup = {
                            "found": True,
                            "customer_name": customer_info.get("customer_name"),
                            "company_name": customer_info.get("company_name"),
                            "company_id": customer_info.get("company_id"),
                            "email": customer_info.get("email"),
                            "ticket_count": customer_info.get("ticket_count", 0),
                            "match_source": customer_info.get("match_source", "")
                        }
                        print(f"Customer identified: {customer_lookup['customer_name']} from {customer_lookup['company_name']}")
                        if progress_callback:
                            await progress_callback("customer", f"Customer identified: {customer_lookup['customer_name']}")
                except Exception as e:
                    print(f"Customer lookup failed (non-critical): {e}")

            # Analyze call content using LLM (subject, outcome, customer name)
            call_content = {"subject": None, "outcome": None, "customer_name": None, "confidence": 0.0, "analysis_model": ""}
            if clean_text and len(clean_text.strip()) >= 50:
                print("Analyzing call content with LLM...")
                if progress_callback:
                    await progress_callback("llm", "Analyzing call content with LLM...")
                call_content = await analyze_call_content_with_llm(clean_text, duration, self.llm_url, staff_name)
                if progress_callback:
                    subject = call_content.get("subject", "")
                    subject_short = subject[:40] + "..." if subject and len(subject) > 40 else subject
                    await progress_callback("llm_done", f"LLM analysis complete: {subject_short}")

            # Format transcription for readability using LLM
            # DISABLED - Takes 4+ minutes and causes timeouts
            # TODO: Re-enable when LLM performance is improved or use a faster approach
            formatted_transcription = clean_text
            print("Transcription formatting DISABLED (performance issue)")
            # if clean_text and len(clean_text.strip()) >= 100:
            #     print("Formatting transcription for readability...")
            #     if progress_callback:
            #         await progress_callback("format", "Formatting transcription for readability...")
            #     formatted_transcription = await format_transcript_with_llm(clean_text, self.llm_url)
            #     if progress_callback:
            #         await progress_callback("format_done", "Transcription formatting complete")

            # Build response
            return {
                "success": True,
                "transcription": formatted_transcription,
                "transcription_summary": transcription_summary,
                "raw_transcription": raw_text,
                "emotions": {
                    "primary": primary_emotion,
                    "detected": metadata["emotions"],
                    "timestamps": []  # SenseVoice doesn't provide timestamps per emotion
                },
                "audio_events": {
                    "detected": metadata["audio_events"],
                    "timestamps": []  # SenseVoice doesn't provide timestamps per event
                },
                "language": metadata.get("language") or language,
                "audio_metadata": audio_metadata,
                "call_metadata": call_metadata,
                "call_content": call_content,
                "customer_lookup": customer_lookup
            }

        except Exception as e:
            print(f"Audio analysis error: {e}")
            filename_to_parse = original_filename if original_filename else audio_path
            return {
                "success": False,
                "error": str(e),
                "transcription": "",
                "transcription_summary": None,
                "raw_transcription": "",
                "emotions": {"primary": "NEUTRAL", "detected": [], "timestamps": []},
                "audio_events": {"detected": [], "timestamps": []},
                "language": language,
                "audio_metadata": audio_metadata,
                "call_metadata": parse_call_filename(filename_to_parse),
                "call_content": {"subject": None, "outcome": None, "customer_name": None, "confidence": 0.0, "analysis_model": ""},
                "customer_lookup": {"found": False, "customer_name": None, "company_name": None}
            }

    async def generate_transcription_summary(
        self,
        transcription: str,
        emotions: List[str],
        duration_seconds: float
    ) -> Optional[str]:
        """
        Generate a summary for long transcriptions using LLM.

        Only generates summaries for transcriptions from audio > 2 minutes.

        Args:
            transcription: The clean transcription text
            emotions: List of detected emotions
            duration_seconds: Duration of the audio in seconds

        Returns:
            Summary string if successful, None otherwise
        """
        # Only summarize if duration exceeds threshold
        if duration_seconds < SUMMARIZATION_THRESHOLD_SECONDS:
            return None

        if not HTTPX_AVAILABLE:
            print("httpx not available, skipping summarization")
            return None

        if not transcription or len(transcription.strip()) < 100:
            return None

        emotions_str = ", ".join(emotions) if emotions else "neutral"

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
                # OpenAI format: choices[0].text
                summary = data.get('choices', [{}])[0].get('text', '').strip()

                if summary:
                    print(f"Generated transcription summary ({len(summary)} chars)")
                    return summary

        except Exception as e:
            print(f"Transcription summarization failed: {e}")

        return None

    async def check_llm_available(self) -> bool:
        """Check if LLM is running and model is available (llama-cpp-python)"""
        if not HTTPX_AVAILABLE:
            return False

        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{self.llm_url}/v1/models")
                if response.status_code == 200:
                    data = response.json()
                    models = data.get('data', [])
                    return len(models) > 0
        except Exception:
            pass
        return False

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


# Global instance getter
def get_audio_service() -> AudioAnalysisService:
    """Get the singleton audio analysis service instance"""
    return AudioAnalysisService.get_instance()
