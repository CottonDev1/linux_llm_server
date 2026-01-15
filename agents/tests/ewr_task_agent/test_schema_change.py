"""
Unit tests for Phase 3 Task Agent Schema Change Detection.

Tests git-based schema change detection and cache invalidation workflow.

Location: /agents/tests/ewr_task_agent/test_schema_change.py
Run: cd /agents && python -m pytest tests/ewr_task_agent/ -v
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime

from ewr_task_agent import TaskAgent
from ewr_task_agent.models import (
    SchemaChangeType,
    SchemaObjectType,
    SchemaChange,
    SchemaChangeResult,
    ShellResult,
    ShellType,
)


class TestSchemaChangeType:
    """Tests for SchemaChangeType enum."""

    def test_change_type_values(self):
        """Test change type enum values."""
        assert SchemaChangeType.CREATE.value == "create"
        assert SchemaChangeType.ALTER.value == "alter"
        assert SchemaChangeType.DROP.value == "drop"


class TestSchemaObjectType:
    """Tests for SchemaObjectType enum."""

    def test_object_type_values(self):
        """Test object type enum values."""
        assert SchemaObjectType.TABLE.value == "table"
        assert SchemaObjectType.COLUMN.value == "column"
        assert SchemaObjectType.INDEX.value == "index"
        assert SchemaObjectType.PROCEDURE.value == "procedure"
        assert SchemaObjectType.VIEW.value == "view"
        assert SchemaObjectType.TRIGGER.value == "trigger"


class TestSchemaChange:
    """Tests for SchemaChange model."""

    def test_schema_change_creation(self):
        """Test creating a schema change."""
        change = SchemaChange(
            file_path="migrations/001.sql",
            change_type=SchemaChangeType.CREATE,
            object_type=SchemaObjectType.TABLE,
            object_name="NewTable",
            details="Created new table"
        )

        assert change.file_path == "migrations/001.sql"
        assert change.change_type == SchemaChangeType.CREATE
        assert change.object_type == SchemaObjectType.TABLE
        assert change.object_name == "NewTable"

    def test_schema_change_defaults(self):
        """Test schema change default values."""
        change = SchemaChange(
            file_path="test.sql",
            change_type=SchemaChangeType.ALTER,
            object_type=SchemaObjectType.COLUMN,
            object_name="TestColumn"
        )

        assert change.details == ""
        assert change.line_number is None


class TestSchemaChangeResult:
    """Tests for SchemaChangeResult model."""

    def test_schema_change_result_creation(self):
        """Test creating a schema change result."""
        result = SchemaChangeResult(
            success=True,
            changed_files=["file1.sql", "file2.sql"],
            affected_tables=["Table1", "Table2"]
        )

        assert result.success is True
        assert len(result.changed_files) == 2
        assert len(result.affected_tables) == 2

    def test_schema_change_result_defaults(self):
        """Test schema change result defaults."""
        result = SchemaChangeResult(success=True)

        assert len(result.changed_files) == 0
        assert len(result.changes) == 0
        assert len(result.affected_tables) == 0
        assert result.cache_invalidated is False
        assert result.reindex_triggered is False
        assert result.compare_ref == "HEAD~1"

    def test_has_changes_true(self):
        """Test has_changes property when changes exist."""
        result = SchemaChangeResult(
            success=True,
            changes=[
                SchemaChange(
                    file_path="test.sql",
                    change_type=SchemaChangeType.CREATE,
                    object_type=SchemaObjectType.TABLE,
                    object_name="NewTable"
                )
            ]
        )

        assert result.has_changes is True

    def test_has_changes_false(self):
        """Test has_changes property when no changes."""
        result = SchemaChangeResult(success=True)

        assert result.has_changes is False

    def test_has_destructive_changes_true(self):
        """Test has_destructive_changes with DROP."""
        result = SchemaChangeResult(
            success=True,
            changes=[
                SchemaChange(
                    file_path="test.sql",
                    change_type=SchemaChangeType.DROP,
                    object_type=SchemaObjectType.TABLE,
                    object_name="OldTable"
                )
            ]
        )

        assert result.has_destructive_changes is True

    def test_has_destructive_changes_false(self):
        """Test has_destructive_changes without DROP."""
        result = SchemaChangeResult(
            success=True,
            changes=[
                SchemaChange(
                    file_path="test.sql",
                    change_type=SchemaChangeType.CREATE,
                    object_type=SchemaObjectType.TABLE,
                    object_name="NewTable"
                )
            ]
        )

        assert result.has_destructive_changes is False


class TestTaskAgentSchemaChangeWorkflow:
    """Tests for TaskAgent schema change workflow."""

    @pytest.fixture
    def agent(self):
        """Create a task agent for testing."""
        agent = TaskAgent()
        return agent

    @pytest.fixture
    def mock_shell_result_no_changes(self):
        """Mock shell result with no changes."""
        return ShellResult(
            command="git diff",
            shell_type=ShellType.BASH,
            exit_code=0,
            stdout="",
            stderr=""
        )

    @pytest.fixture
    def mock_shell_result_with_changes(self):
        """Mock shell result with SQL file changes."""
        return ShellResult(
            command="git diff",
            shell_type=ShellType.BASH,
            exit_code=0,
            stdout="migrations/001_create_table.sql\nmigrations/002_alter_table.sql\n",
            stderr=""
        )

    @pytest.mark.asyncio
    async def test_workflow_no_changes(self, agent):
        """Test workflow with no SQL file changes."""
        # Mock the shell execution
        agent.execute_shell = AsyncMock(return_value=ShellResult(
            command="git diff",
            shell_type=ShellType.BASH,
            exit_code=0,
            stdout="",
            stderr=""
        ))

        result = await agent.schema_change_workflow()

        assert result.success is True
        assert len(result.changed_files) == 0
        assert not result.has_changes

    @pytest.mark.asyncio
    async def test_workflow_with_changes(self, agent):
        """Test workflow with SQL file changes."""
        # Mock getting changed files
        async def mock_execute_shell(cmd, **kwargs):
            if "--name-only" in cmd:
                return ShellResult(
                    command=cmd,
                    shell_type=ShellType.BASH,
                    exit_code=0,
                    stdout="migrations/create_table.sql\n",
                    stderr=""
                )
            elif "git diff" in cmd:
                return ShellResult(
                    command=cmd,
                    shell_type=ShellType.BASH,
                    exit_code=0,
                    stdout="+CREATE TABLE NewTable (ID INT)",
                    stderr=""
                )
            return ShellResult(
                command=cmd,
                shell_type=ShellType.BASH,
                exit_code=0,
                stdout="",
                stderr=""
            )

        agent.execute_shell = mock_execute_shell

        result = await agent.schema_change_workflow()

        assert result.success is True
        assert len(result.changed_files) == 1
        assert "migrations/create_table.sql" in result.changed_files

    @pytest.mark.asyncio
    async def test_workflow_custom_compare_ref(self, agent):
        """Test workflow with custom compare reference."""
        agent.execute_shell = AsyncMock(return_value=ShellResult(
            command="git diff",
            shell_type=ShellType.BASH,
            exit_code=0,
            stdout="",
            stderr=""
        ))

        result = await agent.schema_change_workflow(compare_ref="HEAD~5")

        assert result.compare_ref == "HEAD~5"


class TestTaskAgentDDLParsing:
    """Tests for DDL parsing in schema change detection."""

    @pytest.fixture
    def agent(self):
        """Create a task agent for testing."""
        agent = TaskAgent()
        return agent

    @pytest.mark.asyncio
    async def test_parse_create_table(self, agent):
        """Test parsing CREATE TABLE statement."""
        diff_content = "+CREATE TABLE NewTable (ID INT PRIMARY KEY)"

        agent.execute_shell = AsyncMock(return_value=ShellResult(
            command="git diff",
            shell_type=ShellType.BASH,
            exit_code=0,
            stdout=diff_content,
            stderr=""
        ))

        changes = await agent._parse_schema_changes(".", "test.sql", "HEAD~1")

        assert len(changes) == 1
        assert changes[0].change_type == SchemaChangeType.CREATE
        assert changes[0].object_type == SchemaObjectType.TABLE
        assert changes[0].object_name == "NewTable"

    @pytest.mark.asyncio
    async def test_parse_alter_table(self, agent):
        """Test parsing ALTER TABLE statement."""
        diff_content = "+ALTER TABLE ExistingTable ADD COLUMN NewColumn VARCHAR(100)"

        agent.execute_shell = AsyncMock(return_value=ShellResult(
            command="git diff",
            shell_type=ShellType.BASH,
            exit_code=0,
            stdout=diff_content,
            stderr=""
        ))

        changes = await agent._parse_schema_changes(".", "test.sql", "HEAD~1")

        # Should detect both ALTER TABLE and ADD COLUMN
        assert len(changes) >= 1
        assert any(c.change_type == SchemaChangeType.ALTER for c in changes)

    @pytest.mark.asyncio
    async def test_parse_drop_table(self, agent):
        """Test parsing DROP TABLE statement."""
        diff_content = "+DROP TABLE OldTable"

        agent.execute_shell = AsyncMock(return_value=ShellResult(
            command="git diff",
            shell_type=ShellType.BASH,
            exit_code=0,
            stdout=diff_content,
            stderr=""
        ))

        changes = await agent._parse_schema_changes(".", "test.sql", "HEAD~1")

        assert len(changes) == 1
        assert changes[0].change_type == SchemaChangeType.DROP
        assert changes[0].object_type == SchemaObjectType.TABLE
        assert changes[0].object_name == "OldTable"

    @pytest.mark.asyncio
    async def test_parse_create_index(self, agent):
        """Test parsing CREATE INDEX statement."""
        diff_content = "+CREATE INDEX IX_Table_Column ON MyTable (Column1)"

        agent.execute_shell = AsyncMock(return_value=ShellResult(
            command="git diff",
            shell_type=ShellType.BASH,
            exit_code=0,
            stdout=diff_content,
            stderr=""
        ))

        changes = await agent._parse_schema_changes(".", "test.sql", "HEAD~1")

        assert len(changes) == 1
        assert changes[0].change_type == SchemaChangeType.CREATE
        assert changes[0].object_type == SchemaObjectType.INDEX
        assert "IX_Table_Column" in changes[0].object_name

    @pytest.mark.asyncio
    async def test_parse_create_procedure(self, agent):
        """Test parsing CREATE PROCEDURE statement."""
        diff_content = "+CREATE PROCEDURE sp_GetData AS SELECT * FROM Table"

        agent.execute_shell = AsyncMock(return_value=ShellResult(
            command="git diff",
            shell_type=ShellType.BASH,
            exit_code=0,
            stdout=diff_content,
            stderr=""
        ))

        changes = await agent._parse_schema_changes(".", "test.sql", "HEAD~1")

        assert len(changes) == 1
        assert changes[0].change_type == SchemaChangeType.CREATE
        assert changes[0].object_type == SchemaObjectType.PROCEDURE
        assert "sp_GetData" in changes[0].object_name

    @pytest.mark.asyncio
    async def test_parse_create_view(self, agent):
        """Test parsing CREATE VIEW statement."""
        diff_content = "+CREATE VIEW vw_Data AS SELECT * FROM Table"

        agent.execute_shell = AsyncMock(return_value=ShellResult(
            command="git diff",
            shell_type=ShellType.BASH,
            exit_code=0,
            stdout=diff_content,
            stderr=""
        ))

        changes = await agent._parse_schema_changes(".", "test.sql", "HEAD~1")

        assert len(changes) == 1
        assert changes[0].change_type == SchemaChangeType.CREATE
        assert changes[0].object_type == SchemaObjectType.VIEW
        assert "vw_Data" in changes[0].object_name

    @pytest.mark.asyncio
    async def test_parse_bracket_notation(self, agent):
        """Test parsing DDL with bracket notation."""
        diff_content = "+CREATE TABLE [dbo].[NewTable] (ID INT)"

        agent.execute_shell = AsyncMock(return_value=ShellResult(
            command="git diff",
            shell_type=ShellType.BASH,
            exit_code=0,
            stdout=diff_content,
            stderr=""
        ))

        changes = await agent._parse_schema_changes(".", "test.sql", "HEAD~1")

        assert len(changes) == 1
        # Brackets should be stripped
        assert "[" not in changes[0].object_name
        assert "]" not in changes[0].object_name

    @pytest.mark.asyncio
    async def test_parse_no_ddl(self, agent):
        """Test parsing diff with no DDL changes."""
        diff_content = "+SELECT * FROM Table"

        agent.execute_shell = AsyncMock(return_value=ShellResult(
            command="git diff",
            shell_type=ShellType.BASH,
            exit_code=0,
            stdout=diff_content,
            stderr=""
        ))

        changes = await agent._parse_schema_changes(".", "test.sql", "HEAD~1")

        assert len(changes) == 0


class TestTaskAgentGetChangedFiles:
    """Tests for getting changed SQL files from git."""

    @pytest.fixture
    def agent(self):
        """Create a task agent for testing."""
        return TaskAgent()

    @pytest.mark.asyncio
    async def test_get_sql_files(self, agent):
        """Test getting .sql files from git diff."""
        agent.execute_shell = AsyncMock(return_value=ShellResult(
            command="git diff",
            shell_type=ShellType.BASH,
            exit_code=0,
            stdout="schema.sql\nmigrations/001.sql\n",
            stderr=""
        ))

        files = await agent._get_changed_sql_files(".", "HEAD~1")

        assert len(files) == 2
        assert "schema.sql" in files
        assert "migrations/001.sql" in files

    @pytest.mark.asyncio
    async def test_get_migration_files(self, agent):
        """Test getting migration files."""
        agent.execute_shell = AsyncMock(return_value=ShellResult(
            command="git diff",
            shell_type=ShellType.BASH,
            exit_code=0,
            stdout="migrations/up.sql\nmigrations/down.sql\n",
            stderr=""
        ))

        files = await agent._get_changed_sql_files(".", "HEAD~1")

        assert len(files) == 2

    @pytest.mark.asyncio
    async def test_get_files_git_error(self, agent):
        """Test handling git error gracefully."""
        agent.execute_shell = AsyncMock(return_value=ShellResult(
            command="git diff",
            shell_type=ShellType.BASH,
            exit_code=1,
            stdout="",
            stderr="fatal: not a git repository"
        ))

        files = await agent._get_changed_sql_files(".", "HEAD~1")

        assert len(files) == 0


class TestTaskAgentCacheInvalidation:
    """Tests for cache invalidation notifications."""

    @pytest.fixture
    def agent(self):
        """Create a task agent for testing."""
        return TaskAgent()

    @pytest.mark.asyncio
    async def test_invalidate_caches_success(self, agent):
        """Test cache invalidation returns True on success."""
        result = await agent._invalidate_affected_caches(["Table1", "Table2"])

        assert result is True

    @pytest.mark.asyncio
    async def test_trigger_reindex_success(self, agent):
        """Test reindex trigger returns True on success."""
        result = await agent._trigger_reindex(["Table1", "Table2"])

        assert result is True


class TestTaskAgentWorkflowIntegration:
    """Integration tests for complete workflow."""

    @pytest.fixture
    def agent(self):
        """Create a task agent for testing."""
        return TaskAgent()

    @pytest.mark.asyncio
    async def test_complete_workflow_with_table_changes(self, agent):
        """Test complete workflow detecting table changes."""
        call_count = 0

        async def mock_execute_shell(cmd, **kwargs):
            nonlocal call_count
            call_count += 1

            if "--name-only" in cmd:
                return ShellResult(
                    command=cmd,
                    shell_type=ShellType.BASH,
                    exit_code=0,
                    stdout="migrations/create_users.sql\n",
                    stderr=""
                )
            else:
                return ShellResult(
                    command=cmd,
                    shell_type=ShellType.BASH,
                    exit_code=0,
                    stdout="+CREATE TABLE Users (ID INT PRIMARY KEY, Name VARCHAR(100))",
                    stderr=""
                )

        agent.execute_shell = mock_execute_shell

        result = await agent.schema_change_workflow()

        assert result.success is True
        assert result.has_changes is True
        assert len(result.affected_tables) > 0
        assert "Users" in result.affected_tables
        assert result.cache_invalidated is True
        assert result.reindex_triggered is True

    @pytest.mark.asyncio
    async def test_workflow_error_handling(self, agent):
        """Test workflow handles errors gracefully."""
        agent.execute_shell = AsyncMock(side_effect=Exception("Git error"))

        result = await agent.schema_change_workflow()

        assert result.success is False
        assert result.error is not None
        assert "Git error" in result.error

    @pytest.mark.asyncio
    async def test_workflow_timestamps(self, agent):
        """Test workflow records timestamps."""
        agent.execute_shell = AsyncMock(return_value=ShellResult(
            command="git diff",
            shell_type=ShellType.BASH,
            exit_code=0,
            stdout="",
            stderr=""
        ))

        result = await agent.schema_change_workflow()

        assert result.started_at is not None
        assert result.completed_at is not None
        assert result.started_at <= result.completed_at
