# Agent Tests Catalog

This document catalogs all tests for agents in the llm_website project.

## Overview

| Metric | Count |
|--------|-------|
| Total Test Files | 2 |
| Total Test Classes | 17 |
| Total Test Methods | 71 |
| Async Tests | 38 |
| Sync Tests | 33 |

## Directory Structure

```
/agents/tests/
├── conftest.py              # Shared pytest configuration and fixtures
├── __init__.py              # Package marker
├── TEST_CATALOG.md          # This file
├── ewr_code_agent/          # Code Agent tests
│   ├── __init__.py
│   └── test_sql_performance.py
├── ewr_task_agent/          # Task Agent tests
│   ├── __init__.py
│   └── test_schema_change.py
└── shared/                  # Multi-agent tests
    └── __init__.py
```

## Quick Run Commands

```bash
# Run all agent tests
cd /mnt/c/Projects/llm_website/agents
python -m pytest tests/ -v

# Run Code Agent tests only
python -m pytest tests/ewr_code_agent/ -v

# Run Task Agent tests only
python -m pytest tests/ewr_task_agent/ -v

# Run with coverage
python -m pytest tests/ -v --cov=. --cov-report=html

# Run specific test class
python -m pytest tests/ewr_code_agent/test_sql_performance.py::TestCodeAgentSQLPerformance -v

# Run async tests only
python -m pytest tests/ -v -k "asyncio"
```

---

## Test Files

### ewr_code_agent/test_sql_performance.py

**Category:** Unit Tests
**Agent:** ewr_code_agent (Code Agent)
**Description:** Tests SQL anti-pattern detection, complexity scoring, and recommendations.

| Test Class | Test Count | Description |
|------------|------------|-------------|
| `TestIssueSeverity` | 1 | Tests for IssueSeverity enum values |
| `TestPerformanceIssue` | 2 | Tests for PerformanceIssue model creation and defaults |
| `TestPerformanceReport` | 5 | Tests for PerformanceReport model and properties |
| `TestCodeAgentSQLPerformance` | 13 | SQL anti-pattern detection (async) |
| `TestCodeAgentComplexityScoring` | 9 | SQL complexity scoring algorithm |
| `TestCodeAgentCostEstimation` | 5 | Query cost estimation |
| `TestCodeAgentIndexRecommendations` | 4 | Index recommendation generation |
| `TestCodeAgentTableExtraction` | 6 | SQL table extraction from queries |

**Total: 8 classes, 39 tests**

#### Test Details

| Test | Purpose |
|------|---------|
| `TestIssueSeverity::test_severity_values` | Verify INFO, WARNING, ERROR enum values |
| `TestPerformanceIssue::test_performance_issue_creation` | Create issue with all fields |
| `TestPerformanceIssue::test_performance_issue_defaults` | Verify default values |
| `TestPerformanceReport::test_performance_report_creation` | Create report with fields |
| `TestPerformanceReport::test_performance_report_defaults` | Verify report defaults |
| `TestPerformanceReport::test_has_critical_issues_false` | No errors = not critical |
| `TestPerformanceReport::test_has_critical_issues_true` | Has ERROR = critical |
| `TestPerformanceReport::test_issue_count` | Count issues correctly |
| `TestCodeAgentSQLPerformance::test_detect_select_star` | Detect SELECT * anti-pattern |
| `TestCodeAgentSQLPerformance::test_detect_select_star_with_join` | SELECT * with JOIN |
| `TestCodeAgentSQLPerformance::test_no_select_star_with_columns` | Explicit columns OK |
| `TestCodeAgentSQLPerformance::test_detect_missing_where_on_large_table` | Missing WHERE on large table |
| `TestCodeAgentSQLPerformance::test_no_missing_where_with_filter` | WHERE exists = OK |
| `TestCodeAgentSQLPerformance::test_detect_date_function_in_where` | YEAR() in WHERE |
| `TestCodeAgentSQLPerformance::test_detect_multiple_date_functions` | Multiple date functions |
| `TestCodeAgentSQLPerformance::test_detect_no_row_limit` | Missing TOP/LIMIT |
| `TestCodeAgentSQLPerformance::test_no_row_limit_with_top` | TOP present = OK |
| `TestCodeAgentSQLPerformance::test_no_row_limit_with_aggregation` | Aggregation OK |
| `TestCodeAgentSQLPerformance::test_detect_cursor_usage` | CURSOR detected |
| `TestCodeAgentSQLPerformance::test_detect_nolock_hint` | NOLOCK hint detected |
| `TestCodeAgentSQLPerformance::test_detect_function_in_where` | Function in WHERE |
| `TestCodeAgentComplexityScoring::test_simple_query_complexity` | Simple query = low |
| `TestCodeAgentComplexityScoring::test_query_with_where_complexity` | WHERE adds complexity |
| `TestCodeAgentComplexityScoring::test_query_with_join_complexity` | JOIN adds complexity |
| `TestCodeAgentComplexityScoring::test_query_with_multiple_joins` | Multiple JOINs |
| `TestCodeAgentComplexityScoring::test_query_with_group_by_complexity` | GROUP BY |
| `TestCodeAgentComplexityScoring::test_query_with_subquery_complexity` | Subquery |
| `TestCodeAgentComplexityScoring::test_query_with_cte_complexity` | CTE/WITH |
| `TestCodeAgentComplexityScoring::test_query_with_window_function` | OVER() |
| `TestCodeAgentComplexityScoring::test_complexity_capped_at_ten` | Max 10 |
| `TestCodeAgentCostEstimation::test_low_cost_simple_query` | Simple = low |
| `TestCodeAgentCostEstimation::test_high_cost_large_table_no_filter` | Large + no filter = high |
| `TestCodeAgentCostEstimation::test_medium_cost_large_table_with_filter` | Large + filter = medium |
| `TestCodeAgentCostEstimation::test_high_cost_high_complexity` | High complexity = high |
| `TestCodeAgentCostEstimation::test_medium_cost_moderate_complexity` | Medium complexity |
| `TestCodeAgentIndexRecommendations::test_index_recommendation_equality` | Equality filter |
| `TestCodeAgentIndexRecommendations::test_index_recommendation_range` | Range filter |
| `TestCodeAgentIndexRecommendations::test_index_recommendation_order_by` | ORDER BY |
| `TestCodeAgentIndexRecommendations::test_max_five_recommendations` | Max 5 recommendations |
| `TestCodeAgentTableExtraction::test_extract_from_table` | FROM clause |
| `TestCodeAgentTableExtraction::test_extract_join_table` | JOIN table |
| `TestCodeAgentTableExtraction::test_extract_table_with_schema` | dbo.Table |
| `TestCodeAgentTableExtraction::test_extract_table_with_brackets` | [dbo].[Table] |
| `TestCodeAgentTableExtraction::test_extract_update_table` | UPDATE statement |
| `TestCodeAgentTableExtraction::test_extract_insert_table` | INSERT statement |

---

### ewr_task_agent/test_schema_change.py

**Category:** Unit Tests
**Agent:** ewr_task_agent (Task Agent)
**Description:** Tests git-based schema change detection and cache invalidation workflow.

| Test Class | Test Count | Description |
|------------|------------|-------------|
| `TestSchemaChangeType` | 1 | Tests for SchemaChangeType enum |
| `TestSchemaObjectType` | 1 | Tests for SchemaObjectType enum |
| `TestSchemaChange` | 2 | Tests for SchemaChange model |
| `TestSchemaChangeResult` | 6 | Tests for SchemaChangeResult model and properties |
| `TestTaskAgentSchemaChangeWorkflow` | 3 | Schema change workflow tests (async) |
| `TestTaskAgentDDLParsing` | 8 | DDL statement parsing (async) |
| `TestTaskAgentGetChangedFiles` | 3 | Git diff file extraction (async) |
| `TestTaskAgentCacheInvalidation` | 2 | Cache invalidation (async) |
| `TestTaskAgentWorkflowIntegration` | 3 | End-to-end workflow (async) |

**Total: 9 classes, 32 tests**

#### Test Details

| Test | Purpose |
|------|---------|
| `TestSchemaChangeType::test_change_type_values` | CREATE, ALTER, DROP values |
| `TestSchemaObjectType::test_object_type_values` | TABLE, COLUMN, INDEX, etc. |
| `TestSchemaChange::test_schema_change_creation` | Create with all fields |
| `TestSchemaChange::test_schema_change_defaults` | Verify defaults |
| `TestSchemaChangeResult::test_schema_change_result_creation` | Create result |
| `TestSchemaChangeResult::test_schema_change_result_defaults` | Verify defaults |
| `TestSchemaChangeResult::test_has_changes_true` | Has changes |
| `TestSchemaChangeResult::test_has_changes_false` | No changes |
| `TestSchemaChangeResult::test_has_destructive_changes_true` | DROP = destructive |
| `TestSchemaChangeResult::test_has_destructive_changes_false` | CREATE = not destructive |
| `TestTaskAgentSchemaChangeWorkflow::test_workflow_no_changes` | No SQL changes |
| `TestTaskAgentSchemaChangeWorkflow::test_workflow_with_changes` | SQL file changes |
| `TestTaskAgentSchemaChangeWorkflow::test_workflow_custom_compare_ref` | Custom git ref |
| `TestTaskAgentDDLParsing::test_parse_create_table` | CREATE TABLE |
| `TestTaskAgentDDLParsing::test_parse_alter_table` | ALTER TABLE |
| `TestTaskAgentDDLParsing::test_parse_drop_table` | DROP TABLE |
| `TestTaskAgentDDLParsing::test_parse_create_index` | CREATE INDEX |
| `TestTaskAgentDDLParsing::test_parse_create_procedure` | CREATE PROCEDURE |
| `TestTaskAgentDDLParsing::test_parse_create_view` | CREATE VIEW |
| `TestTaskAgentDDLParsing::test_parse_bracket_notation` | [dbo].[Table] |
| `TestTaskAgentDDLParsing::test_parse_no_ddl` | Non-DDL SQL |
| `TestTaskAgentGetChangedFiles::test_get_sql_files` | Get .sql files |
| `TestTaskAgentGetChangedFiles::test_get_migration_files` | Migration files |
| `TestTaskAgentGetChangedFiles::test_get_files_git_error` | Handle git error |
| `TestTaskAgentCacheInvalidation::test_invalidate_caches_success` | Cache invalidation |
| `TestTaskAgentCacheInvalidation::test_trigger_reindex_success` | Trigger reindex |
| `TestTaskAgentWorkflowIntegration::test_complete_workflow_with_table_changes` | Full workflow |
| `TestTaskAgentWorkflowIntegration::test_workflow_error_handling` | Error handling |
| `TestTaskAgentWorkflowIntegration::test_workflow_timestamps` | Timestamps recorded |

---

## Adding New Tests

When adding new agent tests, follow these guidelines:

1. **Location**: Place tests in `/agents/tests/<agent_name>/`
2. **Naming**: Use `test_<feature>.py` naming convention
3. **Imports**: Use the imports set up by `conftest.py` - no need for path manipulation
4. **Docstrings**: Add docstrings to test classes and methods
5. **Async**: Use `@pytest.mark.asyncio` for async tests
6. **Update Catalog**: Add new tests to this document

### Example Test File

```python
"""
Tests for <agent_name> <feature>.

Location: /agents/tests/<agent_name>/test_<feature>.py
Run: cd /agents && python -m pytest tests/<agent_name>/ -v
"""

import pytest
from <agent_package> import <Agent>


class TestFeature:
    """Tests for <feature>."""

    @pytest.fixture
    def agent(self):
        """Create an agent for testing."""
        return <Agent>()

    def test_example(self, agent):
        """Test example functionality."""
        assert agent is not None

    @pytest.mark.asyncio
    async def test_async_example(self, agent):
        """Test async functionality."""
        result = await agent.some_method()
        assert result is not None
```

---

## Changelog

- 2024-12-22: Initial centralization of agent tests
  - Moved tests from individual agent directories to `/agents/tests/`
  - Created unified `conftest.py` for shared fixtures
  - Created this catalog document
