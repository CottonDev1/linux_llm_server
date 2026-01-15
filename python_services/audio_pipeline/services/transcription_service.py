"""
Transcription Service

Handles audio transcription using SenseVoice model with support for
long audio chunking and parallel processing.
"""

import os
import sys
import tempfile
import asyncio
from typing import Optional, List, Dict, Tuple, Any
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

# FunASR SenseVoice model
try:
    from funasr import AutoModel
    FUNASR_AVAILABLE = True
except ImportError:
    FUNASR_AVAILABLE = False
    print("Warning: funasr not installed. Audio transcription will be disabled.")

# Audio processing
try:
    import torchaudio
    import torch
    TORCHAUDIO_AVAILABLE = True
except ImportError:
    TORCHAUDIO_AVAILABLE = False
    print("Warning: torchaudio not installed. Audio format conversion may be limited.")

# Pydub for MP3 loading
try:
    from pydub import AudioSegment
    import numpy as np
    PYDUB_AVAILABLE = True
except ImportError:
    PYDUB_AVAILABLE = False
    print("Warning: pydub not installed. MP3 loading may be limited.")

# Soundfile for saving WAV files
try:
    import soundfile as sf
    SOUNDFILE_AVAILABLE = True
except ImportError:
    SOUNDFILE_AVAILABLE = False
    print("Warning: soundfile not installed. WAV saving may be limited.")

# Mutagen for MP3 metadata
try:
    from mutagen.mp3 import MP3
    from mutagen import MutagenError
    MUTAGEN_AVAILABLE = True
except ImportError:
    MUTAGEN_AVAILABLE = False
    print("Warning: mutagen not installed. MP3 metadata extraction may be limited.")


# Audio chunking settings for Windows (no VAD available)
CHUNK_DURATION_SECONDS = 25  # Duration of each chunk
CHUNK_OVERLAP_SECONDS = 2    # Overlap between chunks

# Parallel processing settings
MAX_PARALLEL_CHUNKS = 4  # Max chunks to process concurrently (GPU memory limited)
PARALLEL_ENABLED = True  # Enable parallel chunk processing


class TranscriptionService:
    """
    Service for audio transcription using SenseVoice model.

    SenseVoice provides:
    - Multilingual transcription (80+ languages)
    - Emotion detection (7 emotions)
    - Audio event detection (8 events)
    - Voice Activity Detection (VAD)

    For long audio (>25 seconds) on Windows, uses manual chunking
    since VAD is not available.
    """

    _instance: Optional['TranscriptionService'] = None

    def __init__(self):
        self.model = None
        self.is_initialized = False
        self.device = "cuda:0" if self._check_cuda() else "cpu"
        self.is_windows = sys.platform == 'win32'
        self.vad_available = False
        self.parallel_enabled = PARALLEL_ENABLED
        self.max_parallel_chunks = MAX_PARALLEL_CHUNKS
        self._executor = None  # Lazy-initialized ThreadPoolExecutor

    @classmethod
    def get_instance(cls) -> 'TranscriptionService':
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

        # Get model path from environment variable
        model_path_env = os.environ.get("SENSEVOICE_MODEL_PATH")
        if not model_path_env:
            raise RuntimeError(
                "\n" + "="*70 + "\n"
                "ERROR: SENSEVOICE_MODEL_PATH environment variable not set.\n\n"
                "To fix this:\n"
                "1. Download the SenseVoice model from:\n"
                "   https://modelscope.cn/models/iic/SenseVoiceSmall\n\n"
                "2. Extract to: <project_root>/models/SenseVoiceSmall/\n\n"
                "3. Add to your .env file:\n"
                "   SENSEVOICE_MODEL_PATH=./models/SenseVoiceSmall\n"
                + "="*70
            )

        # Resolve relative path from project root
        project_root = Path(__file__).parent.parent.parent.parent
        if model_path_env.startswith("./"):
            model_path = project_root / model_path_env[2:]
        else:
            model_path = Path(model_path_env)

        # Verify model exists - check for model.pt which is the main weights file
        model_weights_file = model_path / "model.pt"
        if not model_weights_file.exists():
            raise RuntimeError(
                "\n" + "="*70 + "\n"
                f"ERROR: SenseVoice model not found at: {model_path}\n\n"
                f"Missing required file: {model_weights_file}\n\n"
                "To fix this:\n"
                "1. Download the SenseVoice model from:\n"
                "   https://modelscope.cn/models/iic/SenseVoiceSmall\n\n"
                "2. Extract ALL files to:\n"
                f"   {model_path}/\n\n"
                "3. Ensure model.pt exists in that directory\n"
                + "="*70
            )

        print(f"Using SenseVoice model from: {model_path}")

        # Set offline mode to prevent any network access
        os.environ["MODELSCOPE_OFFLINE"] = "1"

        try:
            if self.is_windows:
                print("Windows detected - initializing without VAD model")
                print("Long audio files will be processed using manual chunking")
                self.model = AutoModel(
                    model=str(model_path),
                    device=self.device,
                    disable_pbar=True,
                    disable_update=True
                )
                self.vad_available = False
            else:
                print("Linux/Mac detected - initializing with VAD model")
                self.model = AutoModel(
                    model=str(model_path),
                    vad_model="fsmn-vad",
                    vad_kwargs={"max_single_segment_time": 30000},
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

    def get_audio_metadata(self, audio_path: str) -> Dict:
        """
        Extract metadata from audio file.

        Args:
            audio_path: Path to audio file

        Returns:
            Dict with duration, sample_rate, channels, format, file_size
        """
        file_format = Path(audio_path).suffix.lower().replace('.', '')
        metadata = {
            "duration_seconds": 0.0,
            "sample_rate": 16000,
            "channels": 1,
            "format": file_format,
            "file_size_bytes": os.path.getsize(audio_path) if os.path.exists(audio_path) else 0
        }

        # For MP3 files, prefer mutagen
        if file_format == 'mp3' and MUTAGEN_AVAILABLE:
            try:
                audio = MP3(audio_path)
                metadata["duration_seconds"] = audio.info.length
                metadata["sample_rate"] = audio.info.sample_rate
                metadata["channels"] = audio.info.channels
                print(f"MP3 metadata extracted via mutagen: duration={metadata['duration_seconds']:.2f}s")
                return metadata
            except Exception as e:
                print(f"Warning: mutagen could not read MP3 metadata: {e}")

        # Fallback to torchaudio
        if TORCHAUDIO_AVAILABLE:
            try:
                info = torchaudio.info(audio_path)
                metadata["duration_seconds"] = info.num_frames / info.sample_rate
                metadata["sample_rate"] = info.sample_rate
                metadata["channels"] = info.num_channels
                print(f"Audio metadata extracted via torchaudio: duration={metadata['duration_seconds']:.2f}s")
            except Exception as e:
                print(f"Warning: Could not extract audio metadata: {e}")

        return metadata

    def _load_audio_as_waveform(self, audio_path: str) -> Tuple[Any, int]:
        """
        Load audio file and return waveform tensor and sample rate.

        Args:
            audio_path: Path to audio file

        Returns:
            Tuple of (waveform tensor, sample_rate)
        """
        file_ext = Path(audio_path).suffix.lower()
        target_sample_rate = 16000

        # Use pydub for MP3 files
        if file_ext == '.mp3' and PYDUB_AVAILABLE:
            print(f"Loading MP3 with pydub: {audio_path}")
            audio = AudioSegment.from_mp3(audio_path)

            if audio.channels == 2:
                audio = audio.set_channels(1)

            if audio.frame_rate != target_sample_rate:
                audio = audio.set_frame_rate(target_sample_rate)

            samples = np.array(audio.get_array_of_samples())
            samples = samples.astype(np.float32) / 32768.0

            # torch is imported at module level with torchaudio
            waveform = torch.from_numpy(samples).unsqueeze(0)
            return waveform, target_sample_rate

        # Use torchaudio for other formats
        if not TORCHAUDIO_AVAILABLE:
            raise RuntimeError("torchaudio is required for non-MP3 audio loading")

        print(f"Loading audio with torchaudio: {audio_path}")
        waveform, sample_rate = torchaudio.load(audio_path)

        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        if sample_rate != target_sample_rate:
            resampler = torchaudio.transforms.Resample(sample_rate, target_sample_rate)
            waveform = resampler(waveform)

        return waveform, target_sample_rate

    def _chunk_audio(
        self,
        waveform: Any,
        sample_rate: int,
        chunk_duration: float = CHUNK_DURATION_SECONDS,
        overlap: float = CHUNK_OVERLAP_SECONDS
    ) -> List[Tuple[Any, float, float]]:
        """
        Split audio waveform into overlapping chunks.

        Args:
            waveform: Audio waveform tensor
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

            start_sample += step_samples

            remaining_samples = total_samples - start_sample
            if remaining_samples > 0 and remaining_samples < chunk_samples * 0.3:
                break

        print(f"Audio chunked into {len(chunks)} segments (total duration: {total_duration:.1f}s)")
        return chunks

    def _save_chunk_to_temp(self, waveform: Any, sample_rate: int) -> str:
        """
        Save a waveform chunk to a temporary WAV file.

        Args:
            waveform: Audio waveform tensor
            sample_rate: Sample rate in Hz

        Returns:
            Path to temporary file
        """
        temp_fd, temp_path = tempfile.mkstemp(suffix='.wav')
        os.close(temp_fd)

        audio_data = waveform.squeeze(0).numpy()

        if SOUNDFILE_AVAILABLE:
            sf.write(temp_path, audio_data, sample_rate)
        else:
            torchaudio.save(temp_path, waveform, sample_rate)

        return temp_path

    async def transcribe(
        self,
        audio_path: str,
        language: str = "auto",
        progress_callback=None
    ) -> Dict[str, Any]:
        """
        Transcribe audio file.

        Args:
            audio_path: Path to audio file
            language: Language code or 'auto'
            progress_callback: Optional async callback(step, message)

        Returns:
            Dict with raw transcription result
        """
        if not self.is_initialized:
            await self.initialize()

        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        # Get audio metadata
        audio_metadata = self.get_audio_metadata(audio_path)
        duration = audio_metadata.get("duration_seconds", 0)

        # Determine if chunking is needed
        file_ext = Path(audio_path).suffix.lower()
        can_chunk = PYDUB_AVAILABLE if file_ext == '.mp3' else TORCHAUDIO_AVAILABLE
        needs_chunking = (
            not self.vad_available and
            duration > CHUNK_DURATION_SECONDS and
            can_chunk
        )

        if needs_chunking:
            print(f"Audio duration ({duration:.1f}s) exceeds {CHUNK_DURATION_SECONDS}s - using chunked processing")
            if progress_callback:
                await progress_callback("transcribe", f"Starting chunked processing ({duration:.1f}s audio)...")
            return await self._transcribe_with_chunking(audio_path, language, progress_callback)
        else:
            if self.vad_available:
                print(f"Processing audio with VAD (duration: {duration:.1f}s)")
            else:
                print(f"Processing short audio directly (duration: {duration:.1f}s)")

            result = self.model.generate(
                input=audio_path,
                language=language,
                use_itn=True,
                batch_size_s=60,
                merge_vad=True
            )

            if not result or len(result) == 0:
                raise ValueError("No transcription result returned")

            transcription_result = result[0]
            raw_text = transcription_result.get("text", "") if isinstance(transcription_result, dict) else str(transcription_result)

            return {
                "raw_text": raw_text,
                "audio_metadata": audio_metadata,
                "chunk_count": 1
            }

    def _transcribe_single_chunk(self, temp_path: str, language: str, chunk_idx: int) -> Tuple[int, str]:
        """
        Transcribe a single chunk synchronously (for use in ThreadPoolExecutor).

        Args:
            temp_path: Path to temporary audio file
            language: Language code
            chunk_idx: Index of the chunk for ordering

        Returns:
            Tuple of (chunk_idx, transcribed_text)
        """
        try:
            result = self.model.generate(
                input=temp_path,
                language=language,
                use_itn=True
            )

            if result and len(result) > 0:
                chunk_result = result[0]
                raw_text = chunk_result.get("text", "") if isinstance(chunk_result, dict) else str(chunk_result)
                return (chunk_idx, raw_text)
            else:
                print(f"Warning: No result for chunk {chunk_idx + 1}")
                return (chunk_idx, "")
        except Exception as e:
            print(f"Error transcribing chunk {chunk_idx + 1}: {e}")
            return (chunk_idx, f"[TRANSCRIPTION_ERROR: {str(e)[:50]}]")

    async def _transcribe_with_chunking(
        self,
        audio_path: str,
        language: str,
        progress_callback=None
    ) -> Dict[str, Any]:
        """
        Transcribe long audio using manual chunking with parallel processing.

        Uses a semaphore to limit concurrent GPU operations and process
        chunks in parallel batches for improved performance.

        Args:
            audio_path: Path to audio file
            language: Language code or 'auto'
            progress_callback: Optional async callback

        Returns:
            Dict with merged transcription result
        """
        audio_metadata = self.get_audio_metadata(audio_path)
        duration = audio_metadata.get('duration_seconds', 0)
        print(f"Processing audio with {'parallel' if self.parallel_enabled else 'sequential'} chunking (duration: {duration:.1f}s)")

        waveform, sample_rate = self._load_audio_as_waveform(audio_path)
        chunks = self._chunk_audio(waveform, sample_rate)

        temp_files = []
        chunk_data = []

        try:
            # Phase 1: Pre-create all temporary files (I/O can be done upfront)
            for i, (chunk_waveform, start_time, end_time) in enumerate(chunks):
                temp_path = self._save_chunk_to_temp(chunk_waveform, sample_rate)
                temp_files.append(temp_path)
                chunk_data.append({
                    'index': i,
                    'temp_path': temp_path,
                    'start_time': start_time,
                    'end_time': end_time
                })

            if progress_callback:
                await progress_callback("chunk", f"Prepared {len(chunks)} chunks for processing")

            # Phase 2: Transcribe chunks
            if self.parallel_enabled and len(chunks) > 1:
                # Parallel processing with semaphore for GPU resource management
                results = await self._transcribe_chunks_parallel(
                    chunk_data, language, progress_callback
                )
            else:
                # Sequential fallback
                results = await self._transcribe_chunks_sequential(
                    chunk_data, language, progress_callback
                )

            # Phase 3: Sort results by chunk index and merge
            results.sort(key=lambda x: x[0])
            raw_texts = [text for _, text in results if text and not text.startswith("[TRANSCRIPTION_ERROR")]

        finally:
            # Cleanup temp files
            for temp_path in temp_files:
                try:
                    os.unlink(temp_path)
                except Exception as e:
                    print(f"Warning: Could not delete temp file {temp_path}: {e}")

        merged_raw = " | ".join(raw_texts)
        print(f"Chunked processing complete. {len(chunks)} chunks processed ({'parallel' if self.parallel_enabled else 'sequential'})")

        return {
            "raw_text": merged_raw,
            "audio_metadata": audio_metadata,
            "chunk_count": len(chunks)
        }

    async def _transcribe_chunks_parallel(
        self,
        chunk_data: List[Dict],
        language: str,
        progress_callback=None
    ) -> List[Tuple[int, str]]:
        """
        Transcribe chunks in parallel using ThreadPoolExecutor with semaphore.

        The semaphore limits concurrent GPU operations to prevent OOM errors.

        Args:
            chunk_data: List of chunk metadata dicts
            language: Language code
            progress_callback: Optional async callback

        Returns:
            List of (chunk_idx, transcribed_text) tuples
        """
        # Lazy-initialize executor
        if self._executor is None:
            self._executor = ThreadPoolExecutor(max_workers=self.max_parallel_chunks)

        # Semaphore to limit concurrent GPU operations
        semaphore = asyncio.Semaphore(self.max_parallel_chunks)
        loop = asyncio.get_event_loop()
        results = []
        completed_count = 0
        total_chunks = len(chunk_data)

        async def process_chunk(chunk: Dict) -> Tuple[int, str]:
            nonlocal completed_count
            async with semaphore:
                idx = chunk['index']
                temp_path = chunk['temp_path']
                start_time = chunk['start_time']
                end_time = chunk['end_time']

                # Run blocking transcription in thread pool
                result = await loop.run_in_executor(
                    self._executor,
                    self._transcribe_single_chunk,
                    temp_path,
                    language,
                    idx
                )

                completed_count += 1
                progress_msg = f"Processed chunk {completed_count}/{total_chunks} ({start_time:.1f}s - {end_time:.1f}s)"
                print(progress_msg)

                if progress_callback:
                    await progress_callback("chunk", progress_msg)

                return result

        # Process all chunks concurrently (semaphore limits actual parallelism)
        print(f"Starting parallel transcription with {self.max_parallel_chunks} workers")
        tasks = [process_chunk(chunk) for chunk in chunk_data]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Handle any exceptions
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                print(f"Chunk {i} failed with exception: {result}")
                processed_results.append((i, f"[TRANSCRIPTION_ERROR: {str(result)[:50]}]"))
            else:
                processed_results.append(result)

        return processed_results

    async def _transcribe_chunks_sequential(
        self,
        chunk_data: List[Dict],
        language: str,
        progress_callback=None
    ) -> List[Tuple[int, str]]:
        """
        Transcribe chunks sequentially (fallback method).

        Args:
            chunk_data: List of chunk metadata dicts
            language: Language code
            progress_callback: Optional async callback

        Returns:
            List of (chunk_idx, transcribed_text) tuples
        """
        results = []
        total_chunks = len(chunk_data)

        for chunk in chunk_data:
            idx = chunk['index']
            temp_path = chunk['temp_path']
            start_time = chunk['start_time']
            end_time = chunk['end_time']

            progress_msg = f"Processing chunk {idx + 1}/{total_chunks} ({start_time:.1f}s - {end_time:.1f}s)"
            print(progress_msg)

            if progress_callback:
                await progress_callback("chunk", progress_msg)

            result = self._transcribe_single_chunk(temp_path, language, idx)
            results.append(result)

        return results


def get_transcription_service() -> TranscriptionService:
    """Get the singleton transcription service instance"""
    return TranscriptionService.get_instance()
