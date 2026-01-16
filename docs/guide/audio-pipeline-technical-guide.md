# Audio Pipeline Technical Guide

> **Version:** 1.0
> **Last Updated:** January 2026
> **Author:** EWR Development Team

---

## Table of Contents

1. [Overview](#1-overview)
2. [Pipeline Flowchart](#2-pipeline-flowchart)
   - 2.1 [High-Level Architecture](#21-high-level-architecture)
   - 2.2 [Detailed Processing Flow](#22-detailed-processing-flow)
   - 2.3 [Data Flow Diagram](#23-data-flow-diagram)
3. [Services](#3-services)
   - 3.1 [AudioAnalysisService](#31-audioanalysisservice)
   - 3.2 [TranscriptionService](#32-transcriptionservice)
   - 3.3 [EmotionService](#33-emotionservice)
   - 3.4 [DiarizationService](#34-diarizationservice)
   - 3.5 [ContentAnalysisService](#35-contentanalysisservice)
   - 3.6 [SummarizationService](#36-summarizationservice)
   - 3.7 [MetadataService](#37-metadataservice)
   - 3.8 [DatabaseService](#38-databaseservice)
4. [Data Models](#4-data-models)
   - 4.1 [Analysis Models](#41-analysis-models)
   - 4.2 [Metadata Models](#42-metadata-models)
   - 4.3 [Storage Models](#43-storage-models)
   - 4.4 [Bulk Processing Models](#44-bulk-processing-models)
5. [Utilities](#5-utilities)
   - 5.1 [AudioValidator](#51-audiovalidator)
   - 5.2 [FormatConverter](#52-formatconverter)
6. [API Routes](#6-api-routes)
   - 6.1 [Core Analysis Endpoints](#61-core-analysis-endpoints)
   - 6.2 [Storage & Search Endpoints](#62-storage--search-endpoints)
   - 6.3 [Statistics Endpoints](#63-statistics-endpoints)
   - 6.4 [Bulk Processing Endpoints](#64-bulk-processing-endpoints)
7. [Python Packages](#7-python-packages)
   - 7.1 [Core Audio Libraries](#71-core-audio-libraries)
   - 7.2 [Machine Learning](#72-machine-learning)
   - 7.3 [Database & Storage](#73-database--storage)
   - 7.4 [HTTP & Async](#74-http--async)
8. [Configuration](#8-configuration)
9. [Database Schema](#9-database-schema)

---

## 1. Overview

The Audio Pipeline is a comprehensive system for processing, transcribing, and analyzing audio recordings from customer support calls. It provides:

- **Automatic Transcription** using SenseVoice AI model
- **Speaker Diarization** to separate Caller 1 (Support) and Caller 2 (Customer)
- **Emotion Detection** identifying 7 emotions in speech
- **Content Analysis** extracting call subject, outcome, and customer information
- **Semantic Search** with vector embeddings for finding similar calls

### Directory Structure

```
python_services/audio_pipeline/
├── __init__.py              # Package exports
├── services/                # Core business logic
│   ├── audio_analysis_service.py
│   ├── transcription_service.py
│   ├── emotion_service.py
│   ├── diarization_service.py
│   ├── content_analysis_service.py
│   ├── summarization_service.py
│   ├── metadata_service.py
│   └── database_service.py
├── models/                  # Pydantic data models
│   ├── analysis_models.py
│   ├── metadata_models.py
│   ├── storage_models.py
│   └── bulk_models.py
└── utils/                   # Utility functions
    ├── audio_validator.py
    └── format_converter.py
```

---

## 2. Pipeline Flowchart

### 2.1 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           AUDIO PIPELINE ARCHITECTURE                        │
└─────────────────────────────────────────────────────────────────────────────┘

    ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
    │  Audio File  │────▶│   Upload &   │────▶│  Validation  │
    │  (MP3/WAV)   │     │   Storage    │     │  & Format    │
    └──────────────┘     └──────────────┘     └──────────────┘
                                                      │
                                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              PHASE 1: TRANSCRIPTION                          │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │  SenseVoice Model → Raw Text with Emotion/Event Tags                   │ │
│  │  [HAPPY][Speech] Hello, thank you for calling...                       │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
                                                      │
                                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         PHASE 2: PARALLEL PROCESSING                         │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐             │
│  │ Speaker         │  │ Emotion         │  │ Summarization   │             │
│  │ Diarization     │  │ Parsing         │  │ (if >2 min)     │             │
│  │ (LLM Fallback)  │  │                 │  │                 │             │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘             │
│  ┌─────────────────┐  ┌─────────────────┐                                  │
│  │ Metadata        │  │ Staff Lookup    │                                  │
│  │ Parsing         │  │ (EWRCentral)    │                                  │
│  └─────────────────┘  └─────────────────┘                                  │
└─────────────────────────────────────────────────────────────────────────────┘
                                                      │
                                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          PHASE 3: CONTENT ANALYSIS                           │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │  LLM Analysis → Subject, Outcome, Customer Name                        │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
                                                      │
                                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                            PHASE 4: STORAGE                                  │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐             │
│  │ MongoDB         │  │ Vector          │  │ Pending JSON    │             │
│  │ Document        │  │ Embeddings      │  │ (Review Queue)  │             │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Detailed Processing Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         DETAILED PROCESSING FLOW                             │
└─────────────────────────────────────────────────────────────────────────────┘

                              ┌─────────────────┐
                              │   Audio Input   │
                              │ .mp3/.wav/.m4a  │
                              └────────┬────────┘
                                       │
                                       ▼
                        ┌──────────────────────────┐
                        │   AudioValidator.validate │
                        │   - Check format         │
                        │   - Check size (≤100MB)  │
                        │   - Check readability    │
                        └────────────┬─────────────┘
                                     │
                          ┌──────────┴──────────┐
                          │                     │
                          ▼                     ▼
                    ┌──────────┐          ┌──────────┐
                    │  Valid   │          │ Invalid  │
                    └────┬─────┘          └────┬─────┘
                         │                     │
                         ▼                     ▼
              ┌─────────────────────┐    ┌──────────┐
              │ FormatConverter     │    │  Error   │
              │ Convert to 16kHz    │    │ Response │
              │ mono WAV            │    └──────────┘
              └──────────┬──────────┘
                         │
                         ▼
              ┌─────────────────────┐
              │ TranscriptionService │
              │ ┌─────────────────┐ │
              │ │ Check duration  │ │
              │ │ >25s? Chunk it  │ │
              │ └────────┬────────┘ │
              │          ▼          │
              │ ┌─────────────────┐ │
              │ │ SenseVoice      │ │
              │ │ Transcription   │ │
              │ │ + Emotions      │ │
              │ │ + Audio Events  │ │
              │ └─────────────────┘ │
              └──────────┬──────────┘
                         │
                         ▼
              ┌─────────────────────┐
              │   EmotionService    │
              │ Parse tags:         │
              │ <|HAPPY|> → [HAPPY] │
              │ <|Speech|> → Speech │
              └──────────┬──────────┘
                         │
           ┌─────────────┼─────────────┐
           │             │             │
           ▼             ▼             ▼
    ┌────────────┐ ┌────────────┐ ┌────────────┐
    │ Diarization│ │ Summarize  │ │ Metadata   │
    │ Service    │ │ Service    │ │ Service    │
    │            │ │            │ │            │
    │ pyannote OR│ │ LLM call   │ │ Parse      │
    │ LLM-based  │ │ if >2 min  │ │ filename   │
    └─────┬──────┘ └─────┬──────┘ └─────┬──────┘
          │              │              │
          └──────────────┼──────────────┘
                         │
                         ▼
              ┌─────────────────────┐
              │   DatabaseService   │
              │ Lookup staff by     │
              │ phone extension     │
              │ from EWRCentral     │
              └──────────┬──────────┘
                         │
                         ▼
              ┌─────────────────────┐
              │ ContentAnalysisService│
              │ LLM extracts:       │
              │ - Call subject      │
              │ - Outcome           │
              │ - Customer name     │
              └──────────┬──────────┘
                         │
                         ▼
              ┌─────────────────────┐
              │  Save to Pending    │
              │  JSON file for      │
              │  human review       │
              └──────────┬──────────┘
                         │
                         ▼
              ┌─────────────────────┐
              │  Human Review &     │
              │  Staff Assignment   │
              └──────────┬──────────┘
                         │
                         ▼
              ┌─────────────────────┐
              │  MongoDB Storage    │
              │  + Vector Embedding │
              └─────────────────────┘
```

### 2.3 Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              DATA FLOW                                       │
└─────────────────────────────────────────────────────────────────────────────┘

INPUT                    PROCESSING                         OUTPUT
─────                    ──────────                         ──────

Audio File ─────────────▶ TranscriptionService
  │                              │
  │ Format: MP3/WAV/M4A         │ Raw text with tags
  │ Size: ≤100MB                │ "<|HAPPY|><|Speech|>Hello..."
  │ Duration: Any               │
  │                              ▼
  │                       EmotionService
  │                              │
  │                              │ Parsed metadata:
  │                              │ {
  │                              │   "emotions": ["HAPPY"],
  │                              │   "events": ["Speech"],
  │                              │   "clean_text": "Hello..."
  │                              │ }
  │                              ▼
Filename ───────────────▶ MetadataService
  │                              │
  │ Pattern:                    │ Call metadata:
  │ 20251209-123034_302_        │ {
  │ (843)858-0749_Outgoing_     │   "call_date": "2025-12-09",
  │ Auto_2265804682051.mp3      │   "extension": "302",
  │                              │   "phone_number": "(843)858-0749",
  │                              │   "direction": "Outgoing"
  │                              │ }
  │                              ▼
Extension ──────────────▶ DatabaseService (EWRCentral)
                                 │
                                 │ Staff info:
                                 │ {
                                 │   "staff_name": "Sarah Johnson",
                                 │   "email": "sarah@ewr.com"
                                 │ }
                                 ▼
Transcription ──────────▶ ContentAnalysisService (LLM)
                                 │
                                 │ Analysis:
                                 │ {
                                 │   "subject": "Internet connectivity issue",
                                 │   "outcome": "Issue Resolved",
                                 │   "customer_name": "John Smith"
                                 │ }
                                 ▼
Clean Text ─────────────▶ DiarizationService / LLM Speaker Sep
                                 │
                                 │ Speaker segments:
                                 │ [
                                 │   {"speaker": "Caller 1", "text": "Hello..."},
                                 │   {"speaker": "Caller 2", "text": "Hi..."}
                                 │ ]
                                 ▼
                          ┌─────────────────┐
                          │ Final Result    │
                          │                 │
                          │ - Transcription │
                          │ - Summary       │
                          │ - Emotions      │
                          │ - Speakers      │
                          │ - Call Content  │
                          │ - Metadata      │
                          └─────────────────┘
                                 │
                                 ▼
                          ┌─────────────────┐
                          │    MongoDB      │
                          │  audio_analyses │
                          └─────────────────┘
```

---

## 3. Services

### 3.1 AudioAnalysisService

**File:** `audio_pipeline/services/audio_analysis_service.py`

**Purpose:** Main orchestration service that coordinates all audio analysis components.

**Pattern:** Singleton (via `get_instance()`)

| Method | Parameters | Returns | Description |
|--------|------------|---------|-------------|
| `analyze_audio()` | `audio_path`, `language`, `original_filename`, `progress_callback`, `enable_diarization`, `min_speakers`, `max_speakers` | `Dict[str, Any]` | Main entry point for analysis |
| `get_supported_formats()` | - | `List[str]` | Returns supported formats |
| `validate_format()` | `filename` | `bool` | Validates file extension |
| `get_max_file_size_mb()` | - | `int` | Returns max file size (100) |
| `initialize()` | - | `None` | Initializes all services |

**Dependencies:**
- TranscriptionService
- EmotionService
- SummarizationService
- ContentAnalysisService
- MetadataService
- DatabaseService
- DiarizationService

---

### 3.2 TranscriptionService

**File:** `audio_pipeline/services/transcription_service.py`

**Purpose:** Handles audio transcription using the SenseVoice model with support for long audio chunking.

**Pattern:** Singleton

| Method | Parameters | Returns | Description |
|--------|------------|---------|-------------|
| `transcribe()` | `audio_path`, `language`, `progress_callback` | `Tuple[str, Dict]` | Transcribes audio file |
| `initialize()` | - | `None` | Loads SenseVoice model |
| `_check_cuda()` | - | `bool` | Detects CUDA availability |
| `_chunk_audio()` | `audio_path`, `chunk_duration`, `overlap` | `List[str]` | Splits long audio |
| `get_audio_metadata()` | `audio_path` | `Dict` | Extracts audio properties |

**Configuration:**
```python
CHUNK_DURATION_SECONDS = 25
CHUNK_OVERLAP_SECONDS = 2
MAX_PARALLEL_CHUNKS = 4
```

**Features:**
- 80+ language support
- Automatic language detection
- Long audio chunking (>25 seconds)
- GPU acceleration (CUDA)
- Emotion tag detection
- Audio event detection

---

### 3.3 EmotionService

**File:** `audio_pipeline/services/emotion_service.py`

**Purpose:** Parses emotion and audio event tags from SenseVoice output.

**Pattern:** Singleton

| Method | Parameters | Returns | Description |
|--------|------------|---------|-------------|
| `parse_tags()` | `raw_text`, `preserve_tags` | `Tuple[str, Dict]` | Extracts tags from text |
| `get_clean_text()` | `raw_text` | `str` | Returns text without tags |
| `get_primary_emotion()` | `emotions` | `str` | Gets dominant emotion |

**Supported Emotions:**
| Emotion | Tag | Display |
|---------|-----|---------|
| Happy | `<\|HAPPY\|>` | `[HAPPY]` |
| Sad | `<\|SAD\|>` | `[SAD]` |
| Angry | `<\|ANGRY\|>` | `[ANGRY]` |
| Neutral | `<\|NEUTRAL\|>` | `[NEUTRAL]` |
| Fearful | `<\|FEARFUL\|>` | `[FEARFUL]` |
| Disgusted | `<\|DISGUSTED\|>` | `[DISGUSTED]` |
| Surprised | `<\|SURPRISED\|>` | `[SURPRISED]` |

**Supported Audio Events:**
- Speech, BGM, Applause, Laughter, Cry, Cough, Sneeze, Breath

---

### 3.4 DiarizationService

**File:** `audio_pipeline/services/diarization_service.py`

**Purpose:** Speaker diarization using pyannote.audio to identify different speakers.

**Pattern:** Singleton

| Method | Parameters | Returns | Description |
|--------|------------|---------|-------------|
| `initialize()` | - | `None` | Loads pyannote pipeline |
| `diarize()` | `audio_path`, `min_speakers`, `max_speakers`, `progress_callback` | `List[SpeakerSegment]` | Performs speaker segmentation |
| `merge_transcription_with_diarization()` | `transcription`, `segments`, `duration` | `Tuple[str, List[Dict]]` | Merges text with speakers |
| `get_speaker_statistics()` | `segments` | `Dict` | Calculates per-speaker stats |

**SpeakerSegment Dataclass:**
```python
@dataclass
class SpeakerSegment:
    speaker: str        # "Caller 1", "Caller 2"
    start_time: float   # Seconds
    end_time: float     # Seconds
    text: str = ""      # Assigned text

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time
```

---

### 3.5 ContentAnalysisService

**File:** `audio_pipeline/services/content_analysis_service.py`

**Purpose:** Analyzes call transcription using LLM to extract subject, outcome, and customer name.

**Pattern:** Singleton

| Method | Parameters | Returns | Description |
|--------|------------|---------|-------------|
| `analyze_call_content()` | `transcription`, `duration_seconds`, `staff_name` | `Dict` | LLM content extraction |
| `separate_speakers()` | `transcription`, `duration_seconds`, `emotions_data` | `Dict` | LLM speaker separation |
| `_is_same_person()` | `name1`, `name2` | `bool` | Name matching helper |

**Outcome Categories:**
- Issue Resolved
- Issue Unresolved
- Issue Logged in Central
- Undetermined

**LLM Configuration:**
- Endpoint: `LLAMACPP_HOST` (default: `http://localhost:8081`)
- Timeout: 120 seconds (content), 180 seconds (speaker separation)
- Max tokens: 300 (content), 4000 (speaker separation)

---

### 3.6 SummarizationService

**File:** `audio_pipeline/services/summarization_service.py`

**Purpose:** Generates LLM-powered summaries for long transcriptions.

**Pattern:** Singleton

| Method | Parameters | Returns | Description |
|--------|------------|---------|-------------|
| `generate_summary()` | `transcription`, `duration` | `Optional[str]` | Creates summary if duration > threshold |
| `should_summarize()` | `duration` | `bool` | Checks if summarization needed |
| `set_threshold()` | `seconds` | `None` | Configure duration threshold |

**Configuration:**
- Default threshold: 120 seconds
- Minimum text length: 100 characters

---

### 3.7 MetadataService

**File:** `audio_pipeline/services/metadata_service.py`

**Purpose:** Parses call metadata from RingCentral recording filenames.

**Pattern:** Singleton

| Method | Parameters | Returns | Description |
|--------|------------|---------|-------------|
| `parse_filename()` | `filepath` | `Dict` | Extracts metadata from filename |
| `normalize_phone_number()` | `phone` | `str` | Strips non-digit characters |

**Filename Pattern:**
```
20251209-123034_302_(843)858-0749_Outgoing_Auto_2265804682051.mp3
│        │      │   │             │        │    │
│        │      │   │             │        │    └── Recording ID
│        │      │   │             │        └── Auto flag
│        │      │   │             └── Direction (Incoming/Outgoing)
│        │      │   └── Phone number
│        │      └── Extension (staff)
│        └── Time (HHMMSS)
└── Date (YYYYMMDD)
```

---

### 3.8 DatabaseService

**File:** `audio_pipeline/services/database_service.py`

**Purpose:** Database lookups for staff and customer information from EWRCentral.

**Pattern:** Singleton

| Method | Parameters | Returns | Description |
|--------|------------|---------|-------------|
| `lookup_staff_by_extension()` | `extension` | `Optional[str]` | Finds staff by phone extension |
| `lookup_customer_by_phone()` | `phone` | `Dict` | Finds customer by phone (disabled) |

**Database Connection:**
- Server: EWRSQLPROD
- Database: EWRCentral
- Table: CentralUsers
- Timeout: 10 seconds

---

## 4. Data Models

### 4.1 Analysis Models

**File:** `audio_pipeline/models/analysis_models.py`

#### AudioMetadata
```python
class AudioMetadata(BaseModel):
    duration_seconds: float
    sample_rate: int
    channels: int
    format: str
    file_size_bytes: int
    original_filename: Optional[str] = None
```

#### EmotionResult
```python
class EmotionResult(BaseModel):
    primary: str                      # Dominant emotion
    detected: List[str]               # All emotions found
    timestamps: List[EmotionTimestamp] = []
```

#### AudioEventResult
```python
class AudioEventResult(BaseModel):
    detected: List[str]               # All events found
    timestamps: List[AudioEventTimestamp] = []
```

#### SpeakerDiarizationResult
```python
class SpeakerDiarizationResult(BaseModel):
    enabled: bool
    segments: List[SpeakerSegment]
    statistics: Dict[str, Any]
    num_speakers: int
```

---

### 4.2 Metadata Models

**File:** `audio_pipeline/models/metadata_models.py`

#### CallMetadata
```python
class CallMetadata(BaseModel):
    call_date: Optional[str]      # YYYY-MM-DD
    call_time: Optional[str]      # HH:MM:SS
    extension: Optional[str]      # Staff phone extension
    phone_number: Optional[str]   # Customer phone
    direction: Optional[str]      # Incoming/Outgoing
    auto_flag: Optional[str]
    recording_id: Optional[str]
    parsed: bool = False
```

#### CallContentAnalysis
```python
class CallContentAnalysis(BaseModel):
    subject: Optional[str]
    outcome: Optional[str]
    customer_name: Optional[str]
    confidence: float = 0.0
    analysis_model: str = ""
```

---

### 4.3 Storage Models

**File:** `audio_pipeline/models/storage_models.py`

#### AudioStoreRequest
```python
class AudioStoreRequest(BaseModel):
    customer_support_staff: str    # Required
    ewr_customer: str
    mood: str
    outcome: str
    filename: str
    transcription: str
    transcription_summary: Optional[str]
    raw_transcription: str
    emotions: EmotionResult
    audio_events: AudioEventResult
    language: str
    audio_metadata: AudioMetadata
    call_metadata: Optional[CallMetadata]
    call_content: Optional[CallContentAnalysis]
    speaker_diarization: Optional[SpeakerDiarizationResult]
    related_ticket_ids: List[int] = []
    pending_filename: Optional[str]
```

#### AudioSearchRequest
```python
class AudioSearchRequest(BaseModel):
    query: Optional[str]
    customer_support_staff: Optional[str]
    ewr_customer: Optional[str]
    mood: Optional[str]
    outcome: Optional[str]
    emotion: Optional[str]
    language: Optional[str]
    limit: int = Field(default=10, ge=1, le=100)
```

---

### 4.4 Bulk Processing Models

**File:** `audio_pipeline/models/bulk_models.py`

#### BulkAudioRequest
```python
class BulkAudioRequest(BaseModel):
    folder_path: str
    recursive: bool = False
    file_patterns: List[str] = ["*.mp3", "*.wav"]
    delay_between_files: float = 2.0
    auto_store: bool = False
```

#### BulkProcessingStatus
```python
class BulkProcessingStatus(BaseModel):
    is_running: bool
    current_file: Optional[str]
    current_index: int
    total_files: int
    processed_files: int
    failed_files: int
    skipped_files: int
    start_time: Optional[datetime]
    elapsed_seconds: float
    estimated_remaining_seconds: Optional[float]
    errors: List[Dict]
```

---

## 5. Utilities

### 5.1 AudioValidator

**File:** `audio_pipeline/utils/audio_validator.py`

**Purpose:** Validates audio files before processing.

| Method | Parameters | Returns | Description |
|--------|------------|---------|-------------|
| `validate()` | `filepath` | `Tuple[bool, Optional[str]]` | Full validation |
| `get_supported_formats()` | - | `List[str]` | Returns format list |

**Validation Checks:**
- File exists
- Extension in whitelist
- File size: 1KB - 100MB
- File is readable

**Supported Formats:**
`.wav`, `.mp3`, `.flac`, `.m4a`, `.ogg`, `.webm`, `.aac`

---

### 5.2 FormatConverter

**File:** `audio_pipeline/utils/format_converter.py`

**Purpose:** Converts audio files to WAV format for processing.

| Method | Parameters | Returns | Description |
|--------|------------|---------|-------------|
| `convert_to_wav()` | `input_path`, `output_path`, `sample_rate`, `mono` | `Tuple[str, bool]` | Converts to WAV |

**Configuration:**
- Target sample rate: 16000 Hz
- Target channels: Mono (1)
- Uses pydub for MP3, torchaudio for others

---

## 6. API Routes

### 6.1 Core Analysis Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/audio/upload` | Upload audio file for processing |
| `POST` | `/audio/analyze-stream` | Streaming SSE analysis with progress |
| `GET` | `/audio/unanalyzed` | List unanalyzed uploaded files |
| `DELETE` | `/audio/unanalyzed/{filename}` | Delete unanalyzed file |
| `GET` | `/audio/pending` | List pending analysis files |
| `GET` | `/audio/pending/{filename}` | Get specific pending analysis |
| `DELETE` | `/audio/pending/{filename}` | Delete pending analysis |
| `GET` | `/audio/stream/{filename}` | Stream audio file |

### 6.2 Storage & Search Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/audio/store` | Store analysis to MongoDB |
| `POST` | `/audio/search` | Semantic search analyses |
| `GET` | `/audio/{analysis_id}` | Get analysis by ID |
| `PUT` | `/audio/{analysis_id}` | Update analysis metadata |
| `DELETE` | `/audio/{analysis_id}` | Delete analysis |

### 6.3 Statistics Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/audio/stats/summary` | Overall statistics |
| `GET` | `/audio/stats/by-staff` | Stats by staff member |
| `GET` | `/audio/staff-metrics/{name}` | Staff performance metrics |
| `GET` | `/audio/customer-support-staff` | List all support staff |
| `GET` | `/audio/lookup-staff/{extension}` | Lookup staff by extension |
| `POST` | `/audio/match-tickets` | Find matching tickets |

### 6.4 Bulk Processing Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/audio/bulk/scan` | Scan directory for audio files |
| `POST` | `/audio/bulk/process` | Start bulk processing |
| `GET` | `/audio/bulk/status/{job_id}` | Get job status |
| `GET` | `/audio/bulk/jobs` | List all jobs |
| `POST` | `/audio/bulk/cancel/{job_id}` | Cancel job |

---

## 7. Python Packages

### 7.1 Core Audio Libraries

| Package | Version | Purpose |
|---------|---------|---------|
| `funasr` | ^1.0.0 | SenseVoice transcription model |
| `pydub` | ^0.25.0 | MP3/audio format conversion |
| `soundfile` | ^0.12.0 | WAV file I/O |
| `mutagen` | ^1.47.0 | MP3 metadata extraction |
| `pyannote-audio` | ^3.1.0 | Speaker diarization |

**funasr (FunASR)**
- Provides SenseVoice model for transcription
- Supports 80+ languages
- Includes emotion and audio event detection
- Optimized for GPU inference

**pydub**
- Cross-platform audio manipulation
- MP3 loading/saving (requires ffmpeg)
- Sample rate conversion
- Channel manipulation

**soundfile**
- Low-level WAV file I/O
- NumPy array conversion
- Lossless audio processing

**pyannote-audio**
- Neural network-based speaker diarization
- Pre-trained models from HuggingFace
- Supports 2+ speaker detection

---

### 7.2 Machine Learning

| Package | Version | Purpose |
|---------|---------|---------|
| `torch` | ^2.0.0 | Deep learning framework |
| `torchaudio` | ^2.0.0 | Audio processing with PyTorch |
| `numpy` | ^1.24.0 | Numerical operations |
| `scipy` | (via torchaudio) | Scientific computing |

**torch (PyTorch)**
- GPU acceleration (CUDA)
- Model inference
- Tensor operations
- Memory management

**torchaudio**
- Audio loading/saving
- Resampling
- Spectral transformations
- Integration with torch tensors

---

### 7.3 Database & Storage

| Package | Version | Purpose |
|---------|---------|---------|
| `motor` | ^3.3.0 | Async MongoDB driver |
| `pymongo` | ^4.6.0 | Sync MongoDB driver |
| `pymssql` | ^2.2.0 | SQL Server connectivity |

**motor**
- Async MongoDB operations
- Connection pooling
- GridFS support

**pymssql**
- FreeTDS-based SQL Server connection
- Windows authentication support
- Parameterized queries

---

### 7.4 HTTP & Async

| Package | Version | Purpose |
|---------|---------|---------|
| `httpx` | ^0.25.0 | Async HTTP client |
| `nest_asyncio` | ^1.5.0 | Nested async context |
| `pydantic` | ^2.5.0 | Data validation |

**httpx**
- Async HTTP requests
- Connection pooling
- Timeout management
- Used for LLM API calls

**pydantic**
- Data model validation
- JSON serialization
- Type coercion
- Configuration management

---

## 8. Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `SENSEVOICE_MODEL_PATH` | - | Path to SenseVoice model |
| `LLAMACPP_HOST` | `http://localhost:8081` | LLM endpoint |
| `CUDA_VISIBLE_DEVICES` | - | GPU selection |
| `MONGODB_URI` | `mongodb://localhost:27017` | MongoDB connection |

### File Paths

| Path | Purpose |
|------|---------|
| `./unanalyzed_uploads/` | Uploaded files awaiting analysis |
| `./audio-files/` | Pending analysis JSON files |
| `./models/SenseVoiceSmall/` | SenseVoice model files |

### Limits

| Setting | Value |
|---------|-------|
| Max file size | 100 MB |
| Min file size | 1 KB |
| Chunk duration | 25 seconds |
| Chunk overlap | 2 seconds |
| Max parallel chunks | 4 |
| Summarization threshold | 120 seconds |
| Max concurrent bulk jobs | 4 |

---

## 9. Database Schema

### MongoDB Collection: `audio_analyses`

```javascript
{
    "_id": ObjectId,

    // Staff & Customer
    "customer_support_staff": String,    // Required
    "ewr_customer": String,

    // Call Classification
    "mood": String,                      // Negative/Positive/Neutral
    "outcome": String,                   // Issue Resolved/Unresolved/etc

    // File Info
    "filename": String,

    // Transcription
    "transcription": String,             // Speaker-labeled
    "transcription_summary": String,     // LLM summary (if applicable)
    "raw_transcription": String,         // With emotion tags

    // Emotions
    "emotions": {
        "primary": String,
        "detected": [String],
        "timestamps": [{
            "emotion": String,
            "start_time": Number,
            "end_time": Number
        }]
    },

    // Audio Events
    "audio_events": {
        "detected": [String],
        "timestamps": [...]
    },

    // Language
    "language": String,

    // Audio Metadata
    "audio_metadata": {
        "duration_seconds": Number,
        "sample_rate": Number,
        "channels": Number,
        "format": String,
        "file_size_bytes": Number,
        "original_filename": String
    },

    // Call Metadata (from filename)
    "call_metadata": {
        "call_date": String,
        "call_time": String,
        "extension": String,
        "phone_number": String,
        "direction": String,
        "recording_id": String,
        "parsed": Boolean
    },

    // LLM Analysis
    "call_content": {
        "subject": String,
        "outcome": String,
        "customer_name": String,
        "confidence": Number,
        "analysis_model": String
    },

    // Speaker Diarization
    "speaker_diarization": {
        "enabled": Boolean,
        "segments": [{
            "speaker": String,
            "start_time": Number,
            "end_time": Number,
            "text": String
        }],
        "statistics": Object,
        "num_speakers": Number
    },

    // Related Data
    "related_ticket_ids": [Number],
    "linked_ticket_id": Number,

    // Vector Search
    "embedding_text": String,
    "embedding": [Number],              // 384-dim vector

    // Timestamps
    "created_at": ISODate,
    "updated_at": ISODate,
    "analyzed_by": String,
    "analysis_version": String
}
```

---

## Appendix: Quick Reference

### Supported Audio Formats
`.wav`, `.mp3`, `.flac`, `.m4a`, `.ogg`, `.webm`, `.aac`

### Emotion Tags
`HAPPY`, `SAD`, `ANGRY`, `NEUTRAL`, `FEARFUL`, `DISGUSTED`, `SURPRISED`

### Audio Event Tags
`Speech`, `BGM`, `Applause`, `Laughter`, `Cry`, `Cough`, `Sneeze`, `Breath`

### Call Outcomes
`Issue Resolved`, `Issue Unresolved`, `Issue Logged in Central`, `Undetermined`

### Service Ports
| Service | Port |
|---------|------|
| Python FastAPI | 8001 |
| MongoDB | 27017 |
| LLM General | 8081 |
| LLM SQL | 8080 |
| LLM Code | 8082 |
| LLM Embedding | 8083 |
