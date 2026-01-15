"""
Code Assistance Pipeline E2E Tests
==================================

Comprehensive test suite for the code assistance pipeline providing 80%+ coverage.

Test Modules:
- test_code_assistance_e2e.py - High-level E2E integration tests
- test_code_assistance_pipeline.py - Pipeline orchestration tests
- test_code_retriever.py - Code retrieval service tests
- test_context_builder.py - Context building service tests
- test_response_generator.py - LLM response generation tests
- test_feedback_service.py - Feedback handling tests
- test_code_assistance_endpoints.py - FastAPI endpoint tests
- test_streaming.py - SSE streaming tests

Additional modules:
- test_code_assistance_generation.py - LLM generation tests
- test_code_assistance_retrieval.py - MongoDB retrieval tests
- test_code_assistance_storage.py - MongoDB storage tests

Usage:
    # Run all code assistance tests
    pytest testing/pipelines/code_assistance/ -v

    # Run specific test file
    pytest testing/pipelines/code_assistance/test_code_assistance_pipeline.py -v

    # Run tests requiring services
    pytest testing/pipelines/code_assistance/ -v -m "requires_mongodb or requires_llm"

    # Run only fast unit tests (mocked)
    pytest testing/pipelines/code_assistance/ -v -m "not e2e and not slow"
"""
