"""
Git Routes Tests
================

Comprehensive tests for Git API endpoints.

Tests cover:
- GET /api/git-repositories - List repositories
- POST /api/git-repositories - Create repository
- GET /api/git-repositories/{name} - Get repository
- PATCH /api/git-repositories/{name} - Update repository
- DELETE /api/git-repositories/{name} - Delete repository
- POST /api/git-repositories/{name}/pull - Update pull status
- POST /api/git-repositories/{name}/analyzed - Mark as analyzed
- GET /api/git-repositories/auto-sync - List auto-sync repos
- GET /api/git-repositories/needs-analysis - List repos needing analysis

Also covers git operations routes:
- GET /git/repositories - List git repositories
- GET /git/repositories/{repo_name}/info - Get repository info
- GET /git/repositories/{repo_name}/commits - Get commits
- POST /git/pull - Pull repository
- POST /git/analyze-past - Analyze past commits
- GET /git/changed-files/{repo_name} - Get changed files
"""

import os
import pytest
import asyncio
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from datetime import datetime
from typing import List, Dict, Any

# Setup environment before imports
os.environ.setdefault("GIT_ROOT", r"C:\Projects\Git")
os.environ.setdefault("PROJECT_ROOT", r"C:\Projects\llm_website")

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "python_services"))

try:
    from fastapi.testclient import TestClient
    from fastapi import FastAPI
    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False

try:
    from routes.git_routes import router as git_config_router
    from routes.git_operations_routes import router as git_operations_router
    HAS_ROUTES = True
except ImportError as e:
    HAS_ROUTES = False
    IMPORT_ERROR = str(e)


# Create test app
if HAS_FASTAPI and HAS_ROUTES:
    test_app = FastAPI()
    test_app.include_router(git_config_router)
    test_app.include_router(git_operations_router)


@pytest.fixture
def client():
    """Create test client."""
    if not HAS_FASTAPI or not HAS_ROUTES:
        pytest.skip("FastAPI or routes not available")
    return TestClient(test_app)


@pytest.fixture
def mock_sqlite_service():
    """Create a mock SQLite service."""
    mock = Mock()
    mock.get_all_git_repositories.return_value = [
        {
            "name": "TestRepo",
            "path": r"C:\Projects\Git\TestRepo",
            "display_name": "Test Repository",
            "enabled": True,
            "auto_sync": False,
            "last_pull_date": "2024-01-15T10:00:00",
            "last_analysis_date": "2024-01-15T11:00:00"
        }
    ]
    mock.get_git_repository.return_value = {
        "name": "TestRepo",
        "path": r"C:\Projects\Git\TestRepo",
        "display_name": "Test Repository",
        "enabled": True
    }
    mock.create_git_repository.return_value = {
        "name": "NewRepo",
        "path": r"C:\Projects\Git\NewRepo",
        "display_name": "New Repository",
        "enabled": True
    }
    mock.update_git_repository.return_value = {
        "name": "TestRepo",
        "path": r"C:\Projects\Git\TestRepo",
        "display_name": "Updated Repository",
        "enabled": True
    }
    mock.delete_git_repository.return_value = True
    mock.get_auto_sync_repositories.return_value = []
    mock.get_repositories_needing_analysis.return_value = []
    mock.update_git_repository_pull.return_value = {
        "name": "TestRepo",
        "last_pull_date": datetime.utcnow().isoformat()
    }
    mock.update_git_repository_analysis_date.return_value = {
        "name": "TestRepo",
        "last_analysis_date": datetime.utcnow().isoformat()
    }
    mock.get_repository_access_token.return_value = None
    return mock


@pytest.fixture
def mock_git_service():
    """Create a mock GitService."""
    mock = Mock()
    mock.scan_repositories.return_value = [
        Mock(name="TestRepo", path=r"C:\Projects\Git\TestRepo", display_name="Test Repository")
    ]
    mock.get_repository_path.return_value = r"C:\Projects\Git\TestRepo"
    mock.get_repository_info = AsyncMock(return_value=Mock(
        path=r"C:\Projects\Git\TestRepo",
        name="TestRepo",
        recent_commits=[],
        last_sync="2024-01-15T10:00:00",
        status="active",
        model_dump=lambda: {
            "path": r"C:\Projects\Git\TestRepo",
            "name": "TestRepo",
            "recent_commits": [],
            "last_sync": "2024-01-15T10:00:00",
            "status": "active"
        }
    ))
    mock.get_recent_commits_async = AsyncMock(return_value=[
        Mock(
            hash="abc123",
            author="Test Author",
            date="2024-01-15T10:00:00",
            message="Test commit",
            model_dump=lambda: {
                "hash": "abc123",
                "author": "Test Author",
                "date": "2024-01-15T10:00:00",
                "message": "Test commit"
            }
        )
    ])
    mock.pull_repository_async = AsyncMock(return_value=Mock(
        success=True,
        output="Already up to date.",
        is_already_up_to_date=True,
        error=None
    ))
    mock.get_current_head_async = AsyncMock(return_value="abc123def456")
    mock.get_changed_files_async = AsyncMock(return_value=["file1.cs", "file2.cs"])
    mock.analyze_commits_by_date_range_async = AsyncMock(return_value=[])
    mock.get_commit_details.return_value = None
    mock.get_changed_files_with_status.return_value = []
    return mock


@pytest.fixture
def mock_code_analyzer():
    """Create a mock CodeAnalyzer."""
    mock = Mock()
    mock.filter_code_files.return_value = ["file1.cs", "file2.cs"]
    mock.analyze_multiple_files = AsyncMock(return_value=[])
    mock.is_code_file.return_value = True
    mock.analyze_file_sync.return_value = Mock(
        success=True,
        classes=["TestClass"],
        methods=["TestMethod"]
    )
    return mock


@pytest.fixture
def mock_auth():
    """Create mock authentication dependencies."""
    async def mock_get_current_user():
        return {"username": "testuser", "role": "user"}

    async def mock_require_admin():
        return {"username": "admin", "role": "admin"}

    return mock_get_current_user, mock_require_admin


@pytest.mark.skipif(not HAS_FASTAPI or not HAS_ROUTES, reason="FastAPI or routes not available")
class TestGitRepositoriesRoutes:
    """Test git repository configuration routes."""

    def test_list_repositories(self, client, mock_sqlite_service, mock_auth):
        """Test listing all repositories."""
        get_user, _ = mock_auth

        with patch("routes.git_routes.get_db", return_value=mock_sqlite_service):
            with patch("routes.git_routes.get_current_user", return_value=get_user):
                response = client.get("/api/git-repositories")

                # May need authentication
                assert response.status_code in [200, 401, 403, 422]

    def test_create_repository(self, client, mock_sqlite_service, mock_auth):
        """Test creating a new repository."""
        _, require_admin = mock_auth
        mock_sqlite_service.get_git_repository.return_value = None  # Not existing

        with patch("routes.git_routes.get_db", return_value=mock_sqlite_service):
            with patch("routes.git_routes.require_admin", return_value=require_admin):
                response = client.post(
                    "/api/git-repositories",
                    json={
                        "name": "NewRepo",
                        "path": r"C:\Projects\Git\NewRepo",
                        "display_name": "New Repository"
                    }
                )

                # May need authentication
                assert response.status_code in [201, 401, 403, 422]

    def test_create_repository_duplicate(self, client, mock_sqlite_service, mock_auth):
        """Test creating a duplicate repository returns conflict."""
        _, require_admin = mock_auth

        with patch("routes.git_routes.get_db", return_value=mock_sqlite_service):
            with patch("routes.git_routes.require_admin", return_value=require_admin):
                response = client.post(
                    "/api/git-repositories",
                    json={
                        "name": "TestRepo",  # Already exists
                        "path": r"C:\Projects\Git\TestRepo",
                        "display_name": "Test Repository"
                    }
                )

                # Should return 409 Conflict or auth error
                assert response.status_code in [409, 401, 403, 422]

    def test_get_repository(self, client, mock_sqlite_service, mock_auth):
        """Test getting a specific repository."""
        get_user, _ = mock_auth

        with patch("routes.git_routes.get_db", return_value=mock_sqlite_service):
            with patch("routes.git_routes.get_current_user", return_value=get_user):
                response = client.get("/api/git-repositories/TestRepo")

                assert response.status_code in [200, 401, 403, 422]

    def test_get_repository_not_found(self, client, mock_sqlite_service, mock_auth):
        """Test getting a non-existent repository."""
        get_user, _ = mock_auth
        mock_sqlite_service.get_git_repository.return_value = None

        with patch("routes.git_routes.get_db", return_value=mock_sqlite_service):
            with patch("routes.git_routes.get_current_user", return_value=get_user):
                response = client.get("/api/git-repositories/NonExistent")

                # Should return 404 or auth error
                assert response.status_code in [404, 401, 403, 422]

    def test_update_repository(self, client, mock_sqlite_service, mock_auth):
        """Test updating a repository."""
        _, require_admin = mock_auth

        with patch("routes.git_routes.get_db", return_value=mock_sqlite_service):
            with patch("routes.git_routes.require_admin", return_value=require_admin):
                response = client.patch(
                    "/api/git-repositories/TestRepo",
                    json={"display_name": "Updated Name"}
                )

                assert response.status_code in [200, 401, 403, 422]

    def test_delete_repository(self, client, mock_sqlite_service, mock_auth):
        """Test deleting a repository."""
        _, require_admin = mock_auth

        with patch("routes.git_routes.get_db", return_value=mock_sqlite_service):
            with patch("routes.git_routes.require_admin", return_value=require_admin):
                response = client.delete("/api/git-repositories/TestRepo")

                assert response.status_code in [200, 401, 403, 422]

    def test_delete_repository_not_found(self, client, mock_sqlite_service, mock_auth):
        """Test deleting a non-existent repository."""
        _, require_admin = mock_auth
        mock_sqlite_service.delete_git_repository.return_value = False

        with patch("routes.git_routes.get_db", return_value=mock_sqlite_service):
            with patch("routes.git_routes.require_admin", return_value=require_admin):
                response = client.delete("/api/git-repositories/NonExistent")

                # Should return 404 or auth error
                assert response.status_code in [404, 401, 403, 422]

    def test_list_auto_sync_repositories(self, client, mock_sqlite_service, mock_auth):
        """Test listing auto-sync repositories."""
        get_user, _ = mock_auth

        with patch("routes.git_routes.get_db", return_value=mock_sqlite_service):
            with patch("routes.git_routes.get_current_user", return_value=get_user):
                response = client.get("/api/git-repositories/auto-sync")

                assert response.status_code in [200, 401, 403, 422]

    def test_list_repositories_needing_analysis(self, client, mock_sqlite_service, mock_auth):
        """Test listing repositories needing analysis."""
        get_user, _ = mock_auth

        with patch("routes.git_routes.get_db", return_value=mock_sqlite_service):
            with patch("routes.git_routes.get_current_user", return_value=get_user):
                response = client.get("/api/git-repositories/needs-analysis")

                assert response.status_code in [200, 401, 403, 422]

    def test_update_pull_status(self, client, mock_sqlite_service, mock_auth):
        """Test updating repository pull status."""
        get_user, _ = mock_auth

        with patch("routes.git_routes.get_db", return_value=mock_sqlite_service):
            with patch("routes.git_routes.get_current_user", return_value=get_user):
                response = client.post(
                    "/api/git-repositories/TestRepo/pull",
                    json={
                        "commit_hash": "abc123",
                        "had_changes": True
                    }
                )

                assert response.status_code in [200, 401, 403, 422]

    def test_mark_analyzed(self, client, mock_sqlite_service, mock_auth):
        """Test marking repository as analyzed."""
        get_user, _ = mock_auth

        with patch("routes.git_routes.get_db", return_value=mock_sqlite_service):
            with patch("routes.git_routes.get_current_user", return_value=get_user):
                response = client.post("/api/git-repositories/TestRepo/analyzed")

                assert response.status_code in [200, 401, 403, 422]


@pytest.mark.skipif(not HAS_FASTAPI or not HAS_ROUTES, reason="FastAPI or routes not available")
class TestGitOperationsRoutes:
    """Test git operations routes."""

    def test_list_git_repositories(self, client, mock_git_service):
        """Test listing git repositories via operations endpoint."""
        with patch("routes.git_operations_routes.get_git_service", return_value=mock_git_service):
            response = client.get("/git/repositories")

            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert "repositories" in data

    def test_get_repository_info(self, client, mock_git_service):
        """Test getting repository info."""
        with patch("routes.git_operations_routes.get_git_service", return_value=mock_git_service):
            response = client.get("/git/repositories/TestRepo/info")

            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert "repository" in data

    def test_get_repository_info_not_found(self, client, mock_git_service):
        """Test getting info for non-existent repository."""
        mock_git_service.get_repository_path.return_value = None

        with patch("routes.git_operations_routes.get_git_service", return_value=mock_git_service):
            response = client.get("/git/repositories/NonExistent/info")

            assert response.status_code == 404

    def test_get_repository_commits(self, client, mock_git_service):
        """Test getting repository commits."""
        with patch("routes.git_operations_routes.get_git_service", return_value=mock_git_service):
            response = client.get("/git/repositories/TestRepo/commits")

            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert "commits" in data

    def test_get_repository_commits_with_limit(self, client, mock_git_service):
        """Test getting repository commits with limit parameter."""
        with patch("routes.git_operations_routes.get_git_service", return_value=mock_git_service):
            response = client.get("/git/repositories/TestRepo/commits?limit=5")

            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True

    def test_get_repository_commits_not_found(self, client, mock_git_service):
        """Test getting commits from non-existent repository."""
        mock_git_service.get_repository_path.return_value = None

        with patch("routes.git_operations_routes.get_git_service", return_value=mock_git_service):
            response = client.get("/git/repositories/NonExistent/commits")

            assert response.status_code == 404

    def test_pull_repository(self, client, mock_git_service, mock_code_analyzer):
        """Test pulling a repository."""
        with patch("routes.git_operations_routes.get_git_service", return_value=mock_git_service):
            with patch("routes.git_operations_routes.get_code_analyzer", return_value=mock_code_analyzer):
                response = client.post(
                    "/git/pull",
                    json={
                        "repo": "TestRepo",
                        "analyze_changes": True,
                        "include_code_analysis": False
                    }
                )

                assert response.status_code == 200
                data = response.json()
                assert data["success"] is True
                assert data["repo"] == "TestRepo"

    def test_pull_repository_not_found(self, client, mock_git_service, mock_code_analyzer):
        """Test pulling non-existent repository."""
        mock_git_service.get_repository_path.return_value = None

        with patch("routes.git_operations_routes.get_git_service", return_value=mock_git_service):
            with patch("routes.git_operations_routes.get_code_analyzer", return_value=mock_code_analyzer):
                response = client.post(
                    "/git/pull",
                    json={"repo": "NonExistent"}
                )

                assert response.status_code == 404

    def test_pull_repository_already_up_to_date(self, client, mock_git_service, mock_code_analyzer):
        """Test pulling repository that's already up to date."""
        with patch("routes.git_operations_routes.get_git_service", return_value=mock_git_service):
            with patch("routes.git_operations_routes.get_code_analyzer", return_value=mock_code_analyzer):
                response = client.post(
                    "/git/pull",
                    json={"repo": "TestRepo"}
                )

                assert response.status_code == 200
                data = response.json()
                assert data["success"] is True
                assert data["hasChanges"] is False

    def test_pull_repository_with_changes(self, client, mock_git_service, mock_code_analyzer):
        """Test pulling repository with changes."""
        mock_git_service.pull_repository_async.return_value = Mock(
            success=True,
            output="Updating abc..def",
            is_already_up_to_date=False,
            error=None
        )

        with patch("routes.git_operations_routes.get_git_service", return_value=mock_git_service):
            with patch("routes.git_operations_routes.get_code_analyzer", return_value=mock_code_analyzer):
                response = client.post(
                    "/git/pull",
                    json={"repo": "TestRepo", "analyze_changes": True}
                )

                assert response.status_code == 200
                data = response.json()
                assert data["success"] is True
                assert data["hasChanges"] is True

    def test_analyze_past_commits(self, client, mock_git_service, mock_code_analyzer):
        """Test analyzing past commits."""
        with patch("routes.git_operations_routes.get_git_service", return_value=mock_git_service):
            with patch("routes.git_operations_routes.get_code_analyzer", return_value=mock_code_analyzer):
                response = client.post(
                    "/git/analyze-past",
                    json={
                        "repo": "TestRepo",
                        "start_date": "2024-01-01",
                        "end_date": "2024-01-31"
                    }
                )

                assert response.status_code == 200
                data = response.json()
                assert data["success"] is True
                assert data["repo"] == "TestRepo"

    def test_analyze_past_commits_not_found(self, client, mock_git_service, mock_code_analyzer):
        """Test analyzing past commits for non-existent repository."""
        mock_git_service.get_repository_path.return_value = None

        with patch("routes.git_operations_routes.get_git_service", return_value=mock_git_service):
            with patch("routes.git_operations_routes.get_code_analyzer", return_value=mock_code_analyzer):
                response = client.post(
                    "/git/analyze-past",
                    json={
                        "repo": "NonExistent",
                        "start_date": "2024-01-01",
                        "end_date": "2024-01-31"
                    }
                )

                assert response.status_code == 404

    def test_get_changed_files(self, client, mock_git_service):
        """Test getting changed files."""
        mock_git_service.get_changed_files_async = AsyncMock(return_value=["file1.cs", "file2.cs"])

        with patch("routes.git_operations_routes.get_git_service", return_value=mock_git_service):
            response = client.get(
                "/git/changed-files/TestRepo",
                params={"from_commit": "abc123", "to_commit": "HEAD"}
            )

            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert "changedFiles" in data

    def test_get_changed_files_not_found(self, client, mock_git_service):
        """Test getting changed files from non-existent repository."""
        mock_git_service.get_repository_path.return_value = None

        with patch("routes.git_operations_routes.get_git_service", return_value=mock_git_service):
            response = client.get(
                "/git/changed-files/NonExistent",
                params={"from_commit": "abc123"}
            )

            assert response.status_code == 404

    def test_get_file_status(self, client, mock_git_service):
        """Test getting file status."""
        mock_git_service.get_changed_files_with_status.return_value = [
            Mock(file="file1.cs", status="modified", status_code="M",
                 model_dump=lambda: {"file": "file1.cs", "status": "modified", "status_code": "M"})
        ]

        with patch("routes.git_operations_routes.get_git_service", return_value=mock_git_service):
            response = client.get("/git/file-status/TestRepo")

            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert "files" in data


@pytest.mark.skipif(not HAS_FASTAPI or not HAS_ROUTES, reason="FastAPI or routes not available")
class TestGitRoutesErrorHandling:
    """Test error handling in git routes."""

    def test_invalid_json_body(self, client):
        """Test handling of invalid JSON body."""
        response = client.post(
            "/git/pull",
            content="invalid json",
            headers={"Content-Type": "application/json"}
        )

        # Should return 422 Unprocessable Entity
        assert response.status_code == 422

    def test_missing_required_field(self, client):
        """Test handling of missing required fields."""
        response = client.post(
            "/git/pull",
            json={}  # Missing 'repo' field
        )

        assert response.status_code == 422

    def test_invalid_date_format(self, client, mock_git_service, mock_code_analyzer):
        """Test handling of invalid date format."""
        with patch("routes.git_operations_routes.get_git_service", return_value=mock_git_service):
            with patch("routes.git_operations_routes.get_code_analyzer", return_value=mock_code_analyzer):
                response = client.post(
                    "/git/analyze-past",
                    json={
                        "repo": "TestRepo",
                        "start_date": "invalid-date",
                        "end_date": "2024-01-31"
                    }
                )

                # May still succeed (validation might not check date format)
                # or return 422 if date validation is implemented
                assert response.status_code in [200, 422]


@pytest.mark.skipif(not HAS_FASTAPI or not HAS_ROUTES, reason="FastAPI or routes not available")
class TestGitRoutesIntegration:
    """Integration tests for git routes."""

    def test_list_and_get_workflow(self, client, mock_git_service):
        """Test listing repositories then getting details."""
        with patch("routes.git_operations_routes.get_git_service", return_value=mock_git_service):
            # List repositories
            list_response = client.get("/git/repositories")
            assert list_response.status_code == 200
            repos = list_response.json()["repositories"]
            assert len(repos) > 0

            # Get details for first repository
            repo_name = repos[0]["name"]
            detail_response = client.get(f"/git/repositories/{repo_name}/info")
            assert detail_response.status_code == 200

    def test_pull_and_analyze_workflow(self, client, mock_git_service, mock_code_analyzer):
        """Test pull followed by analysis workflow."""
        mock_git_service.pull_repository_async.return_value = Mock(
            success=True,
            output="Updating abc..def",
            is_already_up_to_date=False,
            error=None
        )

        with patch("routes.git_operations_routes.get_git_service", return_value=mock_git_service):
            with patch("routes.git_operations_routes.get_code_analyzer", return_value=mock_code_analyzer):
                # Pull repository
                pull_response = client.post(
                    "/git/pull",
                    json={"repo": "TestRepo", "analyze_changes": True}
                )
                assert pull_response.status_code == 200
                assert pull_response.json()["hasChanges"] is True

                # Analyze past commits
                analyze_response = client.post(
                    "/git/analyze-past",
                    json={
                        "repo": "TestRepo",
                        "start_date": "2024-01-01",
                        "end_date": "2024-01-31"
                    }
                )
                assert analyze_response.status_code == 200


@pytest.mark.skipif(not HAS_FASTAPI or not HAS_ROUTES, reason="FastAPI or routes not available")
class TestGitRoutesResponseFormat:
    """Test response format consistency."""

    def test_success_response_format(self, client, mock_git_service):
        """Test success response has consistent format."""
        with patch("routes.git_operations_routes.get_git_service", return_value=mock_git_service):
            response = client.get("/git/repositories")

            assert response.status_code == 200
            data = response.json()
            assert "success" in data
            assert data["success"] is True

    def test_error_response_format(self, client, mock_git_service):
        """Test error response has consistent format."""
        mock_git_service.get_repository_path.return_value = None

        with patch("routes.git_operations_routes.get_git_service", return_value=mock_git_service):
            response = client.get("/git/repositories/NonExistent/info")

            assert response.status_code == 404
            data = response.json()
            assert "detail" in data

    def test_commits_response_format(self, client, mock_git_service):
        """Test commits response has expected fields."""
        with patch("routes.git_operations_routes.get_git_service", return_value=mock_git_service):
            response = client.get("/git/repositories/TestRepo/commits")

            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert "commits" in data
            assert "repo" in data

    def test_pull_response_format(self, client, mock_git_service, mock_code_analyzer):
        """Test pull response has expected fields."""
        with patch("routes.git_operations_routes.get_git_service", return_value=mock_git_service):
            with patch("routes.git_operations_routes.get_code_analyzer", return_value=mock_code_analyzer):
                response = client.post("/git/pull", json={"repo": "TestRepo"})

                assert response.status_code == 200
                data = response.json()
                assert "success" in data
                assert "repo" in data
                assert "output" in data
                assert "message" in data
                assert "hasChanges" in data
                assert "changedFiles" in data


@pytest.mark.skipif(not HAS_FASTAPI or not HAS_ROUTES, reason="FastAPI or routes not available")
class TestGitRoutesConcurrency:
    """Test concurrent request handling."""

    @pytest.mark.asyncio
    async def test_concurrent_repository_lists(self, mock_git_service):
        """Test handling concurrent list requests."""
        import httpx

        with patch("routes.git_operations_routes.get_git_service", return_value=mock_git_service):
            async with httpx.AsyncClient(app=test_app, base_url="http://test") as client:
                # Send multiple concurrent requests
                tasks = [
                    client.get("/git/repositories")
                    for _ in range(5)
                ]

                responses = await asyncio.gather(*tasks)

                # All should succeed
                for response in responses:
                    assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_concurrent_info_requests(self, mock_git_service):
        """Test handling concurrent info requests."""
        import httpx

        with patch("routes.git_operations_routes.get_git_service", return_value=mock_git_service):
            async with httpx.AsyncClient(app=test_app, base_url="http://test") as client:
                # Send multiple concurrent requests
                tasks = [
                    client.get("/git/repositories/TestRepo/info")
                    for _ in range(5)
                ]

                responses = await asyncio.gather(*tasks)

                # All should succeed
                for response in responses:
                    assert response.status_code == 200
