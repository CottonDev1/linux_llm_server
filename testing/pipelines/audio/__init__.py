"""
Audio Pipeline Tests
====================

Comprehensive test suite for the audio analysis pipeline.

Test modules:
- test_audio_storage: MongoDB storage for audio analyses
- test_audio_retrieval: Querying and retrieving audio analyses
- test_audio_generation: LLM-powered analysis (summarization, content analysis)
- test_audio_e2e: End-to-end pipeline tests
- test_audio_analysis_service: Core AudioAnalysisService orchestration tests
- test_diarization_service: Speaker detection and diarization tests
- test_transcription_service: SenseVoice transcription and chunking tests
- test_emotion_service: Emotion/event tag parsing tests

Fixtures (conftest.py):
- Mock audio data generators
- Speaker segment fixtures
- Emotion/event test cases
- Mock SenseVoice and pyannote services
- Progress callback tracking

All tests use local llama.cpp endpoints (port 8081) and MongoDB (EWRSPT-AI:27018).
"""
