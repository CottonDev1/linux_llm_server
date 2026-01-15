"""
Unit tests for Phase 3 Code Agent SQL Performance Analysis.

Tests SQL anti-pattern detection, complexity scoring, and recommendations.

Location: /agents/tests/ewr_code_agent/test_sql_performance.py
Run: cd /agents && python -m pytest tests/ewr_code_agent/ -v
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from ewr_code_agent import CodeAgent
from ewr_code_agent.models import (
    IssueSeverity,
    PerformanceIssue,
    PerformanceReport,
)


class TestIssueSeverity:
    """Tests for IssueSeverity enum."""

    def test_severity_values(self):
        """Test severity enum values."""
        assert IssueSeverity.INFO.value == "info"
        assert IssueSeverity.WARNING.value == "warning"
        assert IssueSeverity.ERROR.value == "error"


class TestPerformanceIssue:
    """Tests for PerformanceIssue model."""

    def test_performance_issue_creation(self):
        """Test creating a performance issue."""
        issue = PerformanceIssue(
            severity=IssueSeverity.WARNING,
            issue_type="select_star",
            message="SELECT * retrieves all columns",
            location="SELECT clause",
            suggestion="Replace with explicit column list"
        )

        assert issue.severity == IssueSeverity.WARNING
        assert issue.issue_type == "select_star"
        assert "SELECT *" in issue.message

    def test_performance_issue_defaults(self):
        """Test performance issue default values."""
        issue = PerformanceIssue(
            issue_type="test",
            message="Test message"
        )

        assert issue.severity == IssueSeverity.WARNING
        assert issue.location == ""
        assert issue.suggestion is None
        assert issue.line_number is None


class TestPerformanceReport:
    """Tests for PerformanceReport model."""

    def test_performance_report_creation(self):
        """Test creating a performance report."""
        report = PerformanceReport(
            sql="SELECT * FROM Table",
            complexity_score=3.0,
            estimated_cost="low"
        )

        assert report.sql == "SELECT * FROM Table"
        assert report.complexity_score == 3.0
        assert report.estimated_cost == "low"

    def test_performance_report_defaults(self):
        """Test performance report default values."""
        report = PerformanceReport(sql="SELECT 1")

        assert report.complexity_score == 0.0
        assert report.estimated_cost == "unknown"
        assert len(report.issues) == 0
        assert len(report.suggestions) == 0
        assert len(report.index_recommendations) == 0

    def test_has_critical_issues_false(self):
        """Test has_critical_issues with no errors."""
        report = PerformanceReport(
            sql="SELECT 1",
            issues=[
                PerformanceIssue(
                    severity=IssueSeverity.WARNING,
                    issue_type="test",
                    message="warning"
                )
            ]
        )

        assert report.has_critical_issues is False

    def test_has_critical_issues_true(self):
        """Test has_critical_issues with errors."""
        report = PerformanceReport(
            sql="SELECT 1",
            issues=[
                PerformanceIssue(
                    severity=IssueSeverity.ERROR,
                    issue_type="critical",
                    message="error"
                )
            ]
        )

        assert report.has_critical_issues is True

    def test_issue_count(self):
        """Test issue count property."""
        report = PerformanceReport(
            sql="SELECT 1",
            issues=[
                PerformanceIssue(issue_type="1", message="1"),
                PerformanceIssue(issue_type="2", message="2"),
                PerformanceIssue(issue_type="3", message="3"),
            ]
        )

        assert report.issue_count == 3


class TestCodeAgentSQLPerformance:
    """Tests for CodeAgent SQL performance analysis."""

    @pytest.fixture
    def agent(self):
        """Create a code agent for testing."""
        agent = CodeAgent()
        return agent

    @pytest.mark.asyncio
    async def test_detect_select_star(self, agent):
        """Test detection of SELECT * anti-pattern."""
        sql = "SELECT * FROM CentralTickets"

        report = await agent.analyze_sql_performance(sql)

        assert any(i.issue_type == "select_star" for i in report.issues)

    @pytest.mark.asyncio
    async def test_detect_select_star_with_join(self, agent):
        """Test detection of SELECT * with JOIN."""
        sql = "SELECT * FROM CentralTickets t JOIN Users u ON t.UserID = u.UserID"

        report = await agent.analyze_sql_performance(sql)

        assert any(i.issue_type == "select_star" for i in report.issues)

    @pytest.mark.asyncio
    async def test_no_select_star_with_columns(self, agent):
        """Test no SELECT * detection with explicit columns."""
        sql = "SELECT TicketID, Subject, Status FROM CentralTickets"

        report = await agent.analyze_sql_performance(sql)

        assert not any(i.issue_type == "select_star" for i in report.issues)

    @pytest.mark.asyncio
    async def test_detect_missing_where_on_large_table(self, agent):
        """Test detection of missing WHERE on large table."""
        sql = "SELECT * FROM CentralTickets"
        large_tables = ["CentralTickets"]

        report = await agent.analyze_sql_performance(sql, large_tables=large_tables)

        assert any(i.issue_type == "missing_where" for i in report.issues)

    @pytest.mark.asyncio
    async def test_no_missing_where_with_filter(self, agent):
        """Test no missing WHERE detection when filter exists."""
        sql = "SELECT * FROM CentralTickets WHERE Status = 'Open'"
        large_tables = ["CentralTickets"]

        report = await agent.analyze_sql_performance(sql, large_tables=large_tables)

        assert not any(i.issue_type == "missing_where" for i in report.issues)

    @pytest.mark.asyncio
    async def test_detect_date_function_in_where(self, agent):
        """Test detection of date functions in WHERE clause."""
        sql = "SELECT * FROM CentralTickets WHERE YEAR(AddTicketDate) = 2024"

        report = await agent.analyze_sql_performance(sql)

        assert any(i.issue_type == "date_function_in_where" for i in report.issues)

    @pytest.mark.asyncio
    async def test_detect_multiple_date_functions(self, agent):
        """Test detection of multiple date functions."""
        sql = "SELECT * FROM CentralTickets WHERE YEAR(AddTicketDate) = 2024 AND MONTH(AddTicketDate) = 12"

        report = await agent.analyze_sql_performance(sql)

        issues = [i for i in report.issues if i.issue_type == "date_function_in_where"]
        assert len(issues) >= 1

    @pytest.mark.asyncio
    async def test_detect_no_row_limit(self, agent):
        """Test detection of missing row limit."""
        sql = "SELECT TicketID, Subject FROM CentralTickets WHERE Status = 'Open'"

        report = await agent.analyze_sql_performance(sql)

        assert any(i.issue_type == "no_row_limit" for i in report.issues)

    @pytest.mark.asyncio
    async def test_no_row_limit_with_top(self, agent):
        """Test no row limit detection with TOP."""
        sql = "SELECT TOP 100 TicketID, Subject FROM CentralTickets WHERE Status = 'Open'"

        report = await agent.analyze_sql_performance(sql)

        assert not any(i.issue_type == "no_row_limit" for i in report.issues)

    @pytest.mark.asyncio
    async def test_no_row_limit_with_aggregation(self, agent):
        """Test no row limit detection with aggregation."""
        sql = "SELECT Status, COUNT(*) FROM CentralTickets GROUP BY Status"

        report = await agent.analyze_sql_performance(sql)

        # Aggregation queries don't need row limits
        assert not any(i.issue_type == "no_row_limit" for i in report.issues)

    @pytest.mark.asyncio
    async def test_detect_cursor_usage(self, agent):
        """Test detection of CURSOR usage."""
        sql = """
        DECLARE @cursor CURSOR
        SET @cursor = CURSOR FOR SELECT TicketID FROM CentralTickets
        """

        report = await agent.analyze_sql_performance(sql)

        assert any(i.issue_type == "cursor_usage" for i in report.issues)

    @pytest.mark.asyncio
    async def test_detect_nolock_hint(self, agent):
        """Test detection of NOLOCK hint."""
        sql = "SELECT * FROM CentralTickets WITH (NOLOCK)"

        report = await agent.analyze_sql_performance(sql)

        assert any(i.issue_type == "nolock_hint" for i in report.issues)

    @pytest.mark.asyncio
    async def test_detect_function_in_where(self, agent):
        """Test detection of function in WHERE clause."""
        sql = "SELECT * FROM CentralTickets WHERE UPPER(Subject) = 'TEST'"

        report = await agent.analyze_sql_performance(sql)

        assert any(i.issue_type == "function_in_where" for i in report.issues)


class TestCodeAgentComplexityScoring:
    """Tests for SQL complexity scoring."""

    @pytest.fixture
    def agent(self):
        """Create a code agent for testing."""
        return CodeAgent()

    def test_simple_query_complexity(self, agent):
        """Test complexity of simple query."""
        sql = "SELECT * FROM Table"

        score = agent._calculate_sql_complexity(sql)

        assert 1.0 <= score <= 3.0

    def test_query_with_where_complexity(self, agent):
        """Test complexity with WHERE clause."""
        sql = "SELECT * FROM Table WHERE Status = 'Open'"

        score = agent._calculate_sql_complexity(sql)

        # SELECT + WHERE = 1.0 + 0.5 = 1.5
        assert score >= 1.5

    def test_query_with_join_complexity(self, agent):
        """Test complexity with JOIN."""
        sql = "SELECT * FROM Table1 JOIN Table2 ON Table1.ID = Table2.ID"

        score = agent._calculate_sql_complexity(sql)

        # SELECT + JOIN = 1.0 + 1.0 = 2.0
        assert score >= 2.0

    def test_query_with_multiple_joins(self, agent):
        """Test complexity with multiple JOINs."""
        sql = """
        SELECT * FROM Table1
        JOIN Table2 ON Table1.ID = Table2.ID
        JOIN Table3 ON Table2.ID = Table3.ID
        JOIN Table4 ON Table3.ID = Table4.ID
        """

        score = agent._calculate_sql_complexity(sql)

        # SELECT + 3 JOINs = 1.0 + 3.0 = 4.0
        assert score >= 4.0

    def test_query_with_group_by_complexity(self, agent):
        """Test complexity with GROUP BY."""
        sql = "SELECT Status, COUNT(*) FROM Table GROUP BY Status"

        score = agent._calculate_sql_complexity(sql)

        # SELECT + GROUP BY = 1.0 + 1.0 = 2.0
        assert score >= 2.0

    def test_query_with_subquery_complexity(self, agent):
        """Test complexity with subquery."""
        sql = "SELECT * FROM Table WHERE ID IN (SELECT ID FROM OtherTable)"

        score = agent._calculate_sql_complexity(sql)

        # SELECT + subquery = 1.0 + 1.5 = 2.5
        assert score >= 2.5

    def test_query_with_cte_complexity(self, agent):
        """Test complexity with CTE."""
        sql = """
        WITH CTE AS (SELECT * FROM Table)
        SELECT * FROM CTE
        """

        score = agent._calculate_sql_complexity(sql)

        # SELECT + CTE = 1.0 + 1.5 = 2.5
        assert score >= 2.5

    def test_query_with_window_function(self, agent):
        """Test complexity with window function."""
        sql = "SELECT *, ROW_NUMBER() OVER(ORDER BY ID) FROM Table"

        score = agent._calculate_sql_complexity(sql)

        # SELECT + OVER() = 1.0 + 1.0 = 2.0
        assert score >= 2.0

    def test_complexity_capped_at_ten(self, agent):
        """Test that complexity is capped at 10."""
        sql = """
        WITH CTE AS (SELECT * FROM Table)
        SELECT * FROM CTE
        JOIN T1 ON CTE.ID = T1.ID
        JOIN T2 ON T1.ID = T2.ID
        JOIN T3 ON T2.ID = T3.ID
        JOIN T4 ON T3.ID = T4.ID
        WHERE Status IN (SELECT Status FROM Other)
        GROUP BY Status
        HAVING COUNT(*) > 5
        ORDER BY Status
        """

        score = agent._calculate_sql_complexity(sql)

        assert score <= 10.0


class TestCodeAgentCostEstimation:
    """Tests for query cost estimation."""

    @pytest.fixture
    def agent(self):
        """Create a code agent for testing."""
        return CodeAgent()

    def test_low_cost_simple_query(self, agent):
        """Test low cost for simple query."""
        sql = "SELECT TOP 10 * FROM SmallTable"
        tables = ["SmallTable"]
        large_tables = ["LargeTable"]

        cost = agent._estimate_query_cost(sql, 2.0, tables, large_tables)

        assert cost == "low"

    def test_high_cost_large_table_no_filter(self, agent):
        """Test high cost for large table without filter."""
        sql = "SELECT * FROM LargeTable"
        tables = ["LargeTable"]
        large_tables = ["LargeTable"]

        cost = agent._estimate_query_cost(sql, 2.0, tables, large_tables)

        assert cost == "high"

    def test_medium_cost_large_table_with_filter(self, agent):
        """Test medium cost for large table with filter."""
        sql = "SELECT * FROM LargeTable WHERE ID = 1"
        tables = ["LargeTable"]
        large_tables = ["LargeTable"]

        cost = agent._estimate_query_cost(sql, 2.0, tables, large_tables)

        assert cost == "medium"

    def test_high_cost_high_complexity(self, agent):
        """Test high cost for high complexity query."""
        sql = "SELECT * FROM SmallTable WHERE 1=1"
        tables = ["SmallTable"]
        large_tables = []

        cost = agent._estimate_query_cost(sql, 8.0, tables, large_tables)

        assert cost == "high"

    def test_medium_cost_moderate_complexity(self, agent):
        """Test medium cost for moderate complexity."""
        sql = "SELECT * FROM SmallTable WHERE 1=1"
        tables = ["SmallTable"]
        large_tables = []

        cost = agent._estimate_query_cost(sql, 5.0, tables, large_tables)

        assert cost == "medium"


class TestCodeAgentIndexRecommendations:
    """Tests for index recommendations."""

    @pytest.fixture
    def agent(self):
        """Create a code agent for testing."""
        return CodeAgent()

    def test_index_recommendation_equality(self, agent):
        """Test index recommendation for equality filter."""
        sql = "SELECT * FROM Table WHERE Status = 'Open'"

        recommendations = agent._generate_index_recommendations(sql)

        assert any("Status" in r for r in recommendations)

    def test_index_recommendation_range(self, agent):
        """Test index recommendation for range filter."""
        sql = "SELECT * FROM Table WHERE CreateDate >= '2024-01-01'"

        recommendations = agent._generate_index_recommendations(sql)

        assert any("CreateDate" in r for r in recommendations)

    def test_index_recommendation_order_by(self, agent):
        """Test index recommendation for ORDER BY."""
        sql = "SELECT * FROM Table ORDER BY Name ASC"

        recommendations = agent._generate_index_recommendations(sql)

        assert any("Name" in r for r in recommendations)

    def test_max_five_recommendations(self, agent):
        """Test that max 5 recommendations are returned."""
        sql = """
        SELECT * FROM Table
        WHERE A = 1 AND B = 2 AND C = 3 AND D = 4 AND E = 5 AND F = 6
        ORDER BY G, H, I
        """

        recommendations = agent._generate_index_recommendations(sql)

        assert len(recommendations) <= 5


class TestCodeAgentTableExtraction:
    """Tests for SQL table extraction."""

    @pytest.fixture
    def agent(self):
        """Create a code agent for testing."""
        return CodeAgent()

    def test_extract_from_table(self, agent):
        """Test extracting table from FROM clause."""
        sql = "SELECT * FROM CentralTickets"

        tables = agent._extract_sql_tables(sql)

        assert "CentralTickets" in tables

    def test_extract_join_table(self, agent):
        """Test extracting table from JOIN clause."""
        sql = "SELECT * FROM Table1 JOIN Table2 ON 1=1"

        tables = agent._extract_sql_tables(sql)

        assert "Table1" in tables
        assert "Table2" in tables

    def test_extract_table_with_schema(self, agent):
        """Test extracting table with schema prefix."""
        sql = "SELECT * FROM dbo.CentralTickets"

        tables = agent._extract_sql_tables(sql)

        assert "dbo.CentralTickets" in tables

    def test_extract_table_with_brackets(self, agent):
        """Test extracting table with bracket notation."""
        sql = "SELECT * FROM [dbo].[CentralTickets]"

        tables = agent._extract_sql_tables(sql)

        assert "dbo.CentralTickets" in tables

    def test_extract_update_table(self, agent):
        """Test extracting table from UPDATE statement."""
        sql = "UPDATE CentralTickets SET Status = 'Closed'"

        tables = agent._extract_sql_tables(sql)

        assert "CentralTickets" in tables

    def test_extract_insert_table(self, agent):
        """Test extracting table from INSERT statement."""
        sql = "INSERT INTO CentralTickets (Subject) VALUES ('Test')"

        tables = agent._extract_sql_tables(sql)

        assert "CentralTickets" in tables
