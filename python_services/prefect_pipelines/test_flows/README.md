# Prefect Pipeline Test Flows

This directory contains Prefect-integrated test flows for validating pipeline functionality.

## Overview

Each pipeline has a corresponding test flow that runs categorized tests through Prefect:

| Test Flow | Pipeline | Tests Location |
|-----------|----------|----------------|
| `sql_pipeline_test_flow.py` | SQL Generation | `tests/pipelines/sql/` |
| `audio_pipeline_test_flow.py` | Audio Processing | `tests/pipelines/audio/` |
| `document_pipeline_test_flow.py` | Document Processing | `tests/pipelines/document/` |
| `query_pipeline_test_flow.py` | Query Processing | `tests/pipelines/query/` |
| `git_pipeline_test_flow.py` | Git Operations | `tests/pipelines/git/` |

## Test Categories

Each pipeline test flow runs 4 test categories:

1. **Storage** - Data persistence and retrieval
2. **Retrieval** - Search, matching, and context injection
3. **Generation** - Output generation and formatting
4. **E2E** - End-to-end pipeline validation

## Running Tests

### Via Prefect Flow (Recommended)

```bash
# SQL Pipeline
./venv/Scripts/python.exe prefect_pipelines/test_flows/sql_pipeline_test_flow.py \
  --mongodb-uri "mongodb://EWRSPT-AI:27017" \
  --llm-endpoint "http://localhost:8080"

# With custom timeout
./venv/Scripts/python.exe prefect_pipelines/test_flows/sql_pipeline_test_flow.py \
  --mongodb-uri "mongodb://EWRSPT-AI:27017" \
  --timeout 600
```

### Direct pytest

```bash
# Run all SQL tests
./venv/Scripts/python.exe -m pytest tests/pipelines/sql/ -v

# Run specific category
./venv/Scripts/python.exe -m pytest tests/pipelines/sql/test_sql_storage.py -v

# Run with coverage
./venv/Scripts/python.exe -m pytest tests/pipelines/sql/ -v --cov=services
```

## Configuration

Test flows use `TestFlowConfig` for configuration:

```python
from prefect_pipelines.test_flows.base_test_flow import TestFlowConfig

config = TestFlowConfig(
    mongodb_uri="mongodb://EWRSPT-AI:27017",
    mongodb_database="llm_website",
    llm_sql_endpoint="http://localhost:8080",
    llm_general_endpoint="http://localhost:8081",
    timeout_seconds=300,
    cleanup_after_test=True,
)
```

### Environment Variables

Tests receive configuration via environment variables:

| Variable | Description |
|----------|-------------|
| `MONGODB_URI` | MongoDB connection URI |
| `MONGODB_DATABASE` | Database name |
| `LLAMACPP_SQL_HOST` | SQL LLM endpoint |
| `LLAMACPP_HOST` | General LLM endpoint |
| `TEST_CLEANUP` | Cleanup test data (true/false) |

**Important**: Only local LLM endpoints (localhost/127.0.0.1) are allowed.

## Test Results

### Prefect Artifacts

Test flows create markdown artifacts with:
- Pass/fail counts by category
- Duration metrics
- Detailed output

View at: http://localhost:4200

### Console Output

```
Result: {
  'pipeline': 'sql',
  'success': True,
  'metrics': {
    'total': 4,
    'passed': 4,
    'failed': 0,
    'categories': {
      'storage': {'passed': 1, 'failed': 0},
      'retrieval': {'passed': 1, 'failed': 0},
      'generation': {'passed': 1, 'failed': 0},
      'e2e': {'passed': 1, 'failed': 0}
    }
  }
}
```

## Creating New Tests

### Directory Structure

```
tests/pipelines/<pipeline>/
├── __init__.py
├── conftest.py              # Shared fixtures
├── test_<pipeline>_storage.py
├── test_<pipeline>_retrieval.py
├── test_<pipeline>_generation.py
└── test_<pipeline>_e2e.py
```

### Test Naming Convention

Test files must follow the pattern:
```
test_<pipeline>_<category>.py
```

Examples:
- `test_sql_storage.py`
- `test_audio_retrieval.py`
- `test_document_e2e.py`

### Writing Tests

```python
import pytest
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))


class TestMyFeature:
    """Tests for feature X."""

    def test_basic_functionality(self, sample_fixture):
        """Test basic functionality."""
        assert sample_fixture is not None

    @pytest.mark.asyncio
    async def test_async_operation(self, mongodb_uri):
        """Test async operation."""
        try:
            # Test code
            assert True
        except Exception as e:
            pytest.skip(f"Service not available: {e}")
```

## SQL Pipeline Tests

The SQL pipeline tests cover:

### Storage (11 tests)
- Schema document structure
- Foreign key storage
- Sample values
- Related tables
- Stored procedures metadata

### Retrieval (14 tests)
- RelationshipGraph building
- FK-based JOIN generation
- Schema formatting
- Hybrid retrieval
- Rule matching

### Generation (13 tests)
- SQL generator initialization
- Schema formatting
- Business rules injection
- Prompt building
- SQL validation

### E2E (14 tests)
- Pipeline component integration
- Schema context injection
- Relationship graph in pipeline
- Error handling
- Performance baselines

## Shared Components

### base_test_flow.py

- `TestFlowConfig` - Configuration dataclass
- `run_pytest_module` - Task to run pytest
- `run_test_category` - Task to run category tests
- `create_test_artifact` - Task to create result artifacts

### test_flow_utils.py

- `TestStatus` - Enum for test status
- `TestResult` - Dataclass for individual results
- `TestMetrics` - Aggregated metrics
- `ProgressTracker` - Progress tracking
- `create_test_report_markdown` - Report generation

## Troubleshooting

### Tests Not Found

```
ERROR: file or directory not found: tests/pipelines/sql/test_sql_storage.py
```

Ensure test files exist in `tests/pipelines/<pipeline>/` with correct naming.

### MongoDB Connection Failed

```
SKIPPED: MongoDB not available
```

Check MongoDB is running and URI is correct.

### LLM Endpoint Validation Error

```
ValueError: Only local endpoints allowed
```

Ensure LLM endpoint is localhost or 127.0.0.1.
