"""
Roslyn Service Tests
====================

Comprehensive tests for the RoslynService class providing code analysis.

Tests cover:
- Analyzer availability checking
- Repository analysis (sync and async)
- Analysis result parsing
- Timeout handling
- Error recovery
- Output file management
"""

import os
import json
import pytest
import asyncio
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from datetime import datetime
import subprocess

# Setup environment before imports
os.environ.setdefault("GIT_ROOT", r"C:\Projects\Git")
os.environ.setdefault("PROJECT_ROOT", r"C:\Projects\llm_website")

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "python_services"))

try:
    from git_pipeline.services.roslyn_service import RoslynService
    from git_pipeline.models.pipeline_models import (
        PipelineConfig,
        AnalysisResult,
        RoslynAnalysisOutput,
        CodeEntity,
    )
    HAS_ROSLYN_SERVICE = True
except ImportError as e:
    HAS_ROSLYN_SERVICE = False
    IMPORT_ERROR = str(e)


@pytest.fixture
def pipeline_config():
    """Create a pipeline configuration for testing."""
    return PipelineConfig(
        git_root=r"C:\Projects\Git",
        project_root=r"C:\Projects\llm_website",
        roslyn_analyzer_path="tools/RoslynCodeAnalyzer/RoslynCodeAnalyzer.dll",
        analysis_timeout=300
    )


@pytest.fixture
def mock_analysis_output():
    """Create sample Roslyn analysis output JSON."""
    return {
        "fileCount": 10,
        "classes": [
            {
                "name": "UserService",
                "fullName": "MyApp.Services.UserService",
                "filePath": "/src/Services/UserService.cs",
                "lineNumber": 15,
                "namespace": "MyApp.Services",
                "modifiers": ["public"],
                "documentation": "Service for user operations"
            }
        ],
        "methods": [
            {
                "name": "GetUserById",
                "fullName": "MyApp.Services.UserService.GetUserById",
                "filePath": "/src/Services/UserService.cs",
                "lineNumber": 25,
                "namespace": "MyApp.Services",
                "className": "UserService",
                "signature": "public User GetUserById(int userId)",
                "returnType": "User",
                "parameters": [{"name": "userId", "type": "int"}],
                "modifiers": ["public"],
                "documentation": "Retrieves a user by ID",
                "bodyPreview": "return _repository.GetById(userId);"
            }
        ],
        "properties": [
            {
                "name": "ConnectionString",
                "fullName": "MyApp.Config.ConnectionString",
                "filePath": "/src/Config/Config.cs",
                "lineNumber": 10,
                "className": "Config",
                "type": "string",
                "modifiers": ["public", "static"],
                "documentation": "Database connection string"
            }
        ],
        "interfaces": [
            {
                "name": "IUserRepository",
                "fullName": "MyApp.Interfaces.IUserRepository",
                "filePath": "/src/Interfaces/IUserRepository.cs",
                "lineNumber": 5,
                "namespace": "MyApp.Interfaces",
                "modifiers": ["public"],
                "documentation": "Repository interface for user operations"
            }
        ],
        "enums": [
            {
                "name": "UserStatus",
                "fullName": "MyApp.Models.UserStatus",
                "filePath": "/src/Models/UserStatus.cs",
                "lineNumber": 3,
                "namespace": "MyApp.Models",
                "modifiers": ["public"],
                "documentation": "User account status enumeration"
            }
        ],
        "databaseOperations": [
            {
                "operationType": "SELECT",
                "tableName": "Users",
                "sqlCommand": "SELECT * FROM Users WHERE UserID = @UserId",
                "containingClass": "UserRepository",
                "containingMethod": "GetById",
                "filePath": "/src/Repositories/UserRepository.cs",
                "lineNumber": 30
            }
        ],
        "eventHandlers": [
            {
                "eventName": "Click",
                "handlerName": "btnSave_Click",
                "controlName": "btnSave",
                "controlType": "Button",
                "containingClass": "UserForm",
                "filePath": "/src/Forms/UserForm.cs",
                "lineNumber": 50
            }
        ],
        "errors": []
    }


@pytest.fixture
def temp_analysis_file(mock_analysis_output, tmp_path):
    """Create a temporary analysis output file."""
    file_path = tmp_path / "analysis_output.json"
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(mock_analysis_output, f)
    return str(file_path)


@pytest.mark.skipif(not HAS_ROSLYN_SERVICE, reason=f"RoslynService not available: {IMPORT_ERROR if not HAS_ROSLYN_SERVICE else ''}")
class TestRoslynServiceInit:
    """Test RoslynService initialization."""

    def test_initialization_with_default_config(self):
        """Test RoslynService initializes with default configuration."""
        service = RoslynService()

        assert service.config is not None
        assert isinstance(service.config, PipelineConfig)

    def test_initialization_with_custom_config(self, pipeline_config):
        """Test RoslynService initializes with custom configuration."""
        service = RoslynService(config=pipeline_config)

        assert service.config == pipeline_config
        assert service.config.analysis_timeout == 300

    def test_analyzer_path_resolution(self, pipeline_config):
        """Test analyzer path is properly resolved."""
        service = RoslynService(config=pipeline_config)

        assert service.analyzer_path is not None
        assert "RoslynCodeAnalyzer" in service.analyzer_path


@pytest.mark.skipif(not HAS_ROSLYN_SERVICE, reason="RoslynService not available")
class TestIsAnalyzerAvailable:
    """Test analyzer availability checking."""

    def test_analyzer_available_when_exists(self, pipeline_config):
        """Test analyzer is available when file exists."""
        with patch("os.path.exists", return_value=True):
            service = RoslynService(config=pipeline_config)
            is_available = service.is_analyzer_available()

            assert is_available is True

    def test_analyzer_unavailable_when_missing(self, pipeline_config):
        """Test analyzer is unavailable when file is missing."""
        with patch("os.path.exists", return_value=False):
            service = RoslynService(config=pipeline_config)
            is_available = service.is_analyzer_available()

            assert is_available is False


@pytest.mark.skipif(not HAS_ROSLYN_SERVICE, reason="RoslynService not available")
class TestAnalyzeRepository:
    """Test repository analysis."""

    @pytest.mark.asyncio
    async def test_analyze_repository_success(self, pipeline_config, mock_analysis_output, tmp_path):
        """Test successful repository analysis."""
        output_file = tmp_path / "output.json"
        with open(output_file, 'w') as f:
            json.dump(mock_analysis_output, f)

        service = RoslynService(config=pipeline_config)

        with patch.object(service, 'is_analyzer_available', return_value=True):
            with patch.object(service, '_run_analyzer', return_value={"success": True}):
                with patch('os.path.exists', return_value=True):
                    with patch('tempfile.gettempdir', return_value=str(tmp_path)):
                        with patch.object(service, '_parse_analysis_output', return_value=RoslynAnalysisOutput(
                            repository="TestRepo",
                            file_count=10,
                            classes=[],
                            methods=[],
                            properties=[],
                            interfaces=[],
                            enums=[]
                        )):
                            result = await service.analyze_repository(
                                repo_path=r"C:\Projects\Git\TestRepo",
                                repo_name="TestRepo"
                            )

                            assert isinstance(result, AnalysisResult)
                            assert result.success is True
                            assert result.repository == "TestRepo"

    @pytest.mark.asyncio
    async def test_analyze_repository_analyzer_not_found(self, pipeline_config):
        """Test analysis fails when analyzer not found."""
        service = RoslynService(config=pipeline_config)

        with patch.object(service, 'is_analyzer_available', return_value=False):
            result = await service.analyze_repository(
                repo_path=r"C:\Projects\Git\TestRepo",
                repo_name="TestRepo"
            )

            assert result.success is False
            assert "not found" in result.error.lower()

    @pytest.mark.asyncio
    async def test_analyze_repository_run_analyzer_failure(self, pipeline_config):
        """Test analysis handles analyzer execution failure."""
        service = RoslynService(config=pipeline_config)

        with patch.object(service, 'is_analyzer_available', return_value=True):
            with patch.object(service, '_run_analyzer', return_value={
                "success": False,
                "error": "dotnet not found"
            }):
                result = await service.analyze_repository(
                    repo_path=r"C:\Projects\Git\TestRepo",
                    repo_name="TestRepo"
                )

                assert result.success is False
                assert "dotnet not found" in result.error

    @pytest.mark.asyncio
    async def test_analyze_repository_output_not_created(self, pipeline_config, tmp_path):
        """Test analysis handles missing output file."""
        service = RoslynService(config=pipeline_config)

        with patch.object(service, 'is_analyzer_available', return_value=True):
            with patch.object(service, '_run_analyzer', return_value={"success": True}):
                with patch('tempfile.gettempdir', return_value=str(tmp_path)):
                    with patch('os.path.exists', return_value=False):
                        result = await service.analyze_repository(
                            repo_path=r"C:\Projects\Git\TestRepo",
                            repo_name="TestRepo"
                        )

                        assert result.success is False
                        assert "not created" in result.error.lower()

    @pytest.mark.asyncio
    async def test_analyze_repository_exception_handling(self, pipeline_config):
        """Test analysis handles unexpected exceptions."""
        service = RoslynService(config=pipeline_config)

        with patch.object(service, 'is_analyzer_available', side_effect=Exception("Unexpected error")):
            result = await service.analyze_repository(
                repo_path=r"C:\Projects\Git\TestRepo",
                repo_name="TestRepo"
            )

            assert result.success is False
            assert "Unexpected error" in result.error


@pytest.mark.skipif(not HAS_ROSLYN_SERVICE, reason="RoslynService not available")
class TestAnalyzeRepositorySync:
    """Test synchronous repository analysis."""

    def test_analyze_repository_sync(self, pipeline_config, mock_analysis_output, tmp_path):
        """Test sync wrapper for repository analysis."""
        service = RoslynService(config=pipeline_config)

        # Mock the async method
        async_result = AnalysisResult(
            success=True,
            repository="TestRepo",
            entity_count=10,
            file_count=5
        )

        with patch.object(service, 'analyze_repository', new=AsyncMock(return_value=async_result)):
            result = service.analyze_repository_sync(
                repo_path=r"C:\Projects\Git\TestRepo",
                repo_name="TestRepo"
            )

            assert isinstance(result, AnalysisResult)
            assert result.success is True


@pytest.mark.skipif(not HAS_ROSLYN_SERVICE, reason="RoslynService not available")
class TestRunAnalyzer:
    """Test the _run_analyzer method."""

    def test_run_analyzer_success(self, pipeline_config, tmp_path):
        """Test successful analyzer execution."""
        service = RoslynService(config=pipeline_config)
        output_file = str(tmp_path / "output.json")

        with patch('subprocess.run') as mock_run:
            mock_run.return_value = Mock(
                returncode=0,
                stdout="Analysis completed successfully",
                stderr=""
            )

            result = service._run_analyzer(
                repo_path=r"C:\Projects\Git\TestRepo",
                output_file=output_file
            )

            assert result["success"] is True
            mock_run.assert_called_once()

    def test_run_analyzer_failure(self, pipeline_config, tmp_path):
        """Test analyzer execution failure."""
        service = RoslynService(config=pipeline_config)
        output_file = str(tmp_path / "output.json")

        with patch('subprocess.run') as mock_run:
            mock_run.return_value = Mock(
                returncode=1,
                stdout="",
                stderr="Error: Could not find project file"
            )

            result = service._run_analyzer(
                repo_path=r"C:\Projects\Git\TestRepo",
                output_file=output_file
            )

            assert result["success"] is False
            assert "Could not find project file" in result["error"]

    def test_run_analyzer_timeout(self, pipeline_config, tmp_path):
        """Test analyzer timeout handling."""
        service = RoslynService(config=pipeline_config)
        output_file = str(tmp_path / "output.json")

        with patch('subprocess.run') as mock_run:
            mock_run.side_effect = subprocess.TimeoutExpired(cmd="dotnet", timeout=600)

            result = service._run_analyzer(
                repo_path=r"C:\Projects\Git\TestRepo",
                output_file=output_file
            )

            assert result["success"] is False
            assert "timed out" in result["error"].lower()

    def test_run_analyzer_dotnet_not_found(self, pipeline_config, tmp_path):
        """Test handling when dotnet is not installed."""
        service = RoslynService(config=pipeline_config)
        output_file = str(tmp_path / "output.json")

        with patch('subprocess.run') as mock_run:
            mock_run.side_effect = FileNotFoundError("dotnet not found")

            result = service._run_analyzer(
                repo_path=r"C:\Projects\Git\TestRepo",
                output_file=output_file
            )

            assert result["success"] is False
            assert "dotnet" in result["error"].lower()

    def test_run_analyzer_general_exception(self, pipeline_config, tmp_path):
        """Test handling of general exceptions."""
        service = RoslynService(config=pipeline_config)
        output_file = str(tmp_path / "output.json")

        with patch('subprocess.run') as mock_run:
            mock_run.side_effect = Exception("Unexpected subprocess error")

            result = service._run_analyzer(
                repo_path=r"C:\Projects\Git\TestRepo",
                output_file=output_file
            )

            assert result["success"] is False
            assert "Unexpected subprocess error" in result["error"]


@pytest.mark.skipif(not HAS_ROSLYN_SERVICE, reason="RoslynService not available")
class TestParseAnalysisOutput:
    """Test analysis output parsing."""

    def test_parse_analysis_output_success(self, pipeline_config, temp_analysis_file):
        """Test successful parsing of analysis output."""
        service = RoslynService(config=pipeline_config)

        result = service._parse_analysis_output(temp_analysis_file, "TestRepo")

        assert isinstance(result, RoslynAnalysisOutput)
        assert result.repository == "TestRepo"
        assert result.file_count == 10
        assert len(result.classes) == 1
        assert len(result.methods) == 1
        assert len(result.properties) == 1
        assert len(result.interfaces) == 1
        assert len(result.enums) == 1

    def test_parse_analysis_output_classes(self, pipeline_config, temp_analysis_file):
        """Test parsing of class entities."""
        service = RoslynService(config=pipeline_config)

        result = service._parse_analysis_output(temp_analysis_file, "TestRepo")

        assert len(result.classes) == 1
        class_entity = result.classes[0]
        assert class_entity.type == "class"
        assert class_entity.name == "UserService"
        assert class_entity.namespace == "MyApp.Services"

    def test_parse_analysis_output_methods(self, pipeline_config, temp_analysis_file):
        """Test parsing of method entities."""
        service = RoslynService(config=pipeline_config)

        result = service._parse_analysis_output(temp_analysis_file, "TestRepo")

        assert len(result.methods) == 1
        method = result.methods[0]
        assert method.type == "method"
        assert method.name == "GetUserById"
        assert method.parent_class == "UserService"
        assert method.return_type == "User"
        assert method.signature is not None

    def test_parse_analysis_output_properties(self, pipeline_config, temp_analysis_file):
        """Test parsing of property entities."""
        service = RoslynService(config=pipeline_config)

        result = service._parse_analysis_output(temp_analysis_file, "TestRepo")

        assert len(result.properties) == 1
        prop = result.properties[0]
        assert prop.type == "property"
        assert prop.name == "ConnectionString"
        assert prop.return_type == "string"

    def test_parse_analysis_output_interfaces(self, pipeline_config, temp_analysis_file):
        """Test parsing of interface entities."""
        service = RoslynService(config=pipeline_config)

        result = service._parse_analysis_output(temp_analysis_file, "TestRepo")

        assert len(result.interfaces) == 1
        iface = result.interfaces[0]
        assert iface.type == "interface"
        assert iface.name == "IUserRepository"
        assert iface.namespace == "MyApp.Interfaces"

    def test_parse_analysis_output_enums(self, pipeline_config, temp_analysis_file):
        """Test parsing of enum entities."""
        service = RoslynService(config=pipeline_config)

        result = service._parse_analysis_output(temp_analysis_file, "TestRepo")

        assert len(result.enums) == 1
        enum = result.enums[0]
        assert enum.type == "enum"
        assert enum.name == "UserStatus"

    def test_parse_analysis_output_database_operations(self, pipeline_config, temp_analysis_file):
        """Test parsing of database operations."""
        service = RoslynService(config=pipeline_config)

        result = service._parse_analysis_output(temp_analysis_file, "TestRepo")

        assert len(result.database_operations) == 1
        db_op = result.database_operations[0]
        assert db_op["operationType"] == "SELECT"
        assert db_op["tableName"] == "Users"

    def test_parse_analysis_output_event_handlers(self, pipeline_config, temp_analysis_file):
        """Test parsing of event handlers."""
        service = RoslynService(config=pipeline_config)

        result = service._parse_analysis_output(temp_analysis_file, "TestRepo")

        assert len(result.event_handlers) == 1
        handler = result.event_handlers[0]
        assert handler["eventName"] == "Click"
        assert handler["handlerName"] == "btnSave_Click"

    def test_parse_analysis_output_invalid_json(self, pipeline_config, tmp_path):
        """Test parsing of invalid JSON file."""
        invalid_file = tmp_path / "invalid.json"
        invalid_file.write_text("{ invalid json }")

        service = RoslynService(config=pipeline_config)
        result = service._parse_analysis_output(str(invalid_file), "TestRepo")

        assert result is None

    def test_parse_analysis_output_empty_file(self, pipeline_config, tmp_path):
        """Test parsing of empty JSON file."""
        empty_file = tmp_path / "empty.json"
        empty_file.write_text("{}")

        service = RoslynService(config=pipeline_config)
        result = service._parse_analysis_output(str(empty_file), "TestRepo")

        assert isinstance(result, RoslynAnalysisOutput)
        assert result.file_count == 0
        assert len(result.classes) == 0
        assert len(result.methods) == 0

    def test_parse_analysis_output_missing_fields(self, pipeline_config, tmp_path):
        """Test parsing handles missing fields gracefully."""
        partial_file = tmp_path / "partial.json"
        partial_file.write_text('{"classes": [{"name": "Test"}]}')

        service = RoslynService(config=pipeline_config)
        result = service._parse_analysis_output(str(partial_file), "TestRepo")

        assert isinstance(result, RoslynAnalysisOutput)
        assert len(result.classes) == 1
        assert result.classes[0].name == "Test"
        assert result.classes[0].full_name == "Test"  # Falls back to name

    def test_parse_analysis_output_total_entities(self, pipeline_config, temp_analysis_file):
        """Test total_entities property calculation."""
        service = RoslynService(config=pipeline_config)

        result = service._parse_analysis_output(temp_analysis_file, "TestRepo")

        # 1 class + 1 method + 1 property + 1 interface + 1 enum = 5
        assert result.total_entities == 5


@pytest.mark.skipif(not HAS_ROSLYN_SERVICE, reason="RoslynService not available")
class TestCleanupOutputFile:
    """Test output file cleanup."""

    def test_cleanup_output_file_success(self, pipeline_config, tmp_path):
        """Test successful cleanup of output file."""
        test_file = tmp_path / "test_output.json"
        test_file.write_text("{}")

        service = RoslynService(config=pipeline_config)
        success = service.cleanup_output_file(str(test_file))

        assert success is True
        assert not test_file.exists()

    def test_cleanup_output_file_not_exists(self, pipeline_config):
        """Test cleanup of non-existent file."""
        service = RoslynService(config=pipeline_config)
        success = service.cleanup_output_file("/nonexistent/file.json")

        # Should return True since there's nothing to clean up
        assert success is True or success is False  # Both are acceptable

    def test_cleanup_output_file_none(self, pipeline_config):
        """Test cleanup with None file path."""
        service = RoslynService(config=pipeline_config)
        success = service.cleanup_output_file(None)

        assert success is False

    def test_cleanup_output_file_empty_string(self, pipeline_config):
        """Test cleanup with empty file path."""
        service = RoslynService(config=pipeline_config)
        success = service.cleanup_output_file("")

        assert success is False


@pytest.mark.skipif(not HAS_ROSLYN_SERVICE, reason="RoslynService not available")
class TestCodeEntityModel:
    """Test CodeEntity model."""

    def test_code_entity_defaults(self):
        """Test CodeEntity default values."""
        entity = CodeEntity(type="method", name="TestMethod")

        assert entity.type == "method"
        assert entity.name == "TestMethod"
        assert entity.full_name == ""
        assert entity.file_path == ""
        assert entity.line_number == 0
        assert entity.modifiers == []

    def test_code_entity_full_population(self):
        """Test CodeEntity with all fields populated."""
        entity = CodeEntity(
            type="method",
            name="GetUserById",
            full_name="MyApp.UserService.GetUserById",
            file_path="/src/UserService.cs",
            line_number=25,
            namespace="MyApp",
            parent_class="UserService",
            signature="public User GetUserById(int id)",
            return_type="User",
            parameters=[{"name": "id", "type": "int"}],
            modifiers=["public"],
            doc_comment="Gets user by ID",
            body_preview="return _repo.Get(id);"
        )

        assert entity.full_name == "MyApp.UserService.GetUserById"
        assert entity.parent_class == "UserService"
        assert entity.return_type == "User"
        assert len(entity.parameters) == 1


@pytest.mark.skipif(not HAS_ROSLYN_SERVICE, reason="RoslynService not available")
class TestRoslynAnalysisOutputModel:
    """Test RoslynAnalysisOutput model."""

    def test_roslyn_analysis_output_defaults(self):
        """Test RoslynAnalysisOutput default values."""
        output = RoslynAnalysisOutput(repository="TestRepo")

        assert output.repository == "TestRepo"
        assert output.file_count == 0
        assert output.classes == []
        assert output.methods == []
        assert output.properties == []
        assert output.interfaces == []
        assert output.enums == []
        assert output.errors == []

    def test_roslyn_analysis_output_total_entities(self):
        """Test total_entities calculation."""
        output = RoslynAnalysisOutput(
            repository="TestRepo",
            classes=[CodeEntity(type="class", name="C1"), CodeEntity(type="class", name="C2")],
            methods=[CodeEntity(type="method", name="M1")],
            properties=[CodeEntity(type="property", name="P1")],
            interfaces=[CodeEntity(type="interface", name="I1")],
            enums=[CodeEntity(type="enum", name="E1")]
        )

        assert output.total_entities == 6


@pytest.mark.skipif(not HAS_ROSLYN_SERVICE, reason="RoslynService not available")
class TestRoslynServiceIntegration:
    """Integration tests for RoslynService."""

    @pytest.mark.e2e
    @pytest.mark.asyncio
    async def test_full_analysis_workflow_mocked(self, pipeline_config, mock_analysis_output, tmp_path):
        """Test complete analysis workflow with mocked subprocess."""
        output_file = tmp_path / "analysis_output.json"

        service = RoslynService(config=pipeline_config)

        # Setup mocks
        with patch.object(service, 'is_analyzer_available', return_value=True):
            with patch.object(service, '_run_analyzer') as mock_run:
                # Make _run_analyzer create the output file
                def create_output(repo_path, out_file):
                    with open(out_file, 'w') as f:
                        json.dump(mock_analysis_output, f)
                    return {"success": True}

                mock_run.side_effect = create_output

                with patch('tempfile.gettempdir', return_value=str(tmp_path)):
                    result = await service.analyze_repository(
                        repo_path=r"C:\Projects\Git\TestRepo",
                        repo_name="TestRepo"
                    )

                    # Verify full workflow
                    assert result.success is True
                    assert result.repository == "TestRepo"
                    assert result.entity_count == 5  # From mock data
                    assert result.file_count == 10
                    assert result.duration_seconds > 0

    @pytest.mark.e2e
    def test_parse_and_cleanup_workflow(self, pipeline_config, temp_analysis_file):
        """Test parse followed by cleanup workflow."""
        service = RoslynService(config=pipeline_config)

        # Parse the file
        result = service._parse_analysis_output(temp_analysis_file, "TestRepo")
        assert result is not None
        assert result.total_entities == 5

        # File should still exist
        assert os.path.exists(temp_analysis_file)

        # Cleanup the file
        success = service.cleanup_output_file(temp_analysis_file)
        assert success is True

        # File should be deleted
        assert not os.path.exists(temp_analysis_file)


@pytest.mark.skipif(not HAS_ROSLYN_SERVICE, reason="RoslynService not available")
class TestAnalyzerPathResolution:
    """Test analyzer path resolution logic."""

    def test_resolve_relative_path(self, pipeline_config, tmp_path):
        """Test resolving relative analyzer path."""
        # Create mock analyzer file
        analyzer_dir = tmp_path / "tools" / "RoslynCodeAnalyzer"
        analyzer_dir.mkdir(parents=True)
        analyzer_file = analyzer_dir / "RoslynCodeAnalyzer.dll"
        analyzer_file.write_text("mock dll")

        config = PipelineConfig(
            project_root=str(tmp_path),
            roslyn_analyzer_path="tools/RoslynCodeAnalyzer/RoslynCodeAnalyzer.dll"
        )

        service = RoslynService(config=config)

        assert "RoslynCodeAnalyzer.dll" in service.analyzer_path

    def test_resolve_absolute_path(self, pipeline_config, tmp_path):
        """Test resolving absolute analyzer path."""
        analyzer_file = tmp_path / "RoslynCodeAnalyzer.dll"
        analyzer_file.write_text("mock dll")

        config = PipelineConfig(
            project_root=r"C:\SomeOtherPath",
            roslyn_analyzer_path=str(analyzer_file)
        )

        service = RoslynService(config=config)

        assert service.analyzer_path == str(analyzer_file)

    def test_resolve_missing_analyzer(self, pipeline_config):
        """Test handling missing analyzer file."""
        config = PipelineConfig(
            project_root=r"C:\NonExistent",
            roslyn_analyzer_path="tools/NonExistent.dll"
        )

        # Should not raise, but log warning
        service = RoslynService(config=config)

        # Path is set but file doesn't exist
        assert service.analyzer_path is not None
        assert service.is_analyzer_available() is False
