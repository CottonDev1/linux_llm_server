# Rule Generation Multi-Agent System
"""
Orchestrates multiple AI agents to analyze stored procedures and generate SQL rules.

Components:
- ProcedureAnalyzer: Analyzes stored procedures for semantic patterns
- RuleGenerator: Generates rules from procedure analysis
- RuleValidator: Validates rules before upload
- RuleTester: Tests rules via the Python API
- RuleGenerationOrchestrator: Coordinates the entire pipeline

Usage:
    from rule_generation import RuleGenerationOrchestrator

    orchestrator = RuleGenerationOrchestrator(
        mongodb_uri="mongodb://localhost:27017",
        num_agents=10
    )
    result = orchestrator.run(database_filter="EWRCentral", limit=100)
"""

from .models import (
    ProcedureType, ComplexityTier, ActionType, TemporalScope,
    TableRole, ColumnRole, ImplicitFilter, TableSemantics,
    ColumnSemantics, JoinTemplate, AggregationPattern, TemporalPattern,
    ProcedureAnalysis, GeneratedRule, GeneratedExample,
    ValidationResult, TestResult
)
from .analyzer import ProcedureAnalyzer
from .rule_generator import RuleGenerator
from .validator import RuleValidator
from .tester import RuleTester, SyncRuleTester
from .orchestrator import RuleGenerationOrchestrator

__all__ = [
    # Enums
    'ProcedureType', 'ComplexityTier', 'ActionType', 'TemporalScope',
    'TableRole', 'ColumnRole',
    # Data classes
    'ImplicitFilter', 'TableSemantics', 'ColumnSemantics',
    'JoinTemplate', 'AggregationPattern', 'TemporalPattern',
    'ProcedureAnalysis', 'GeneratedRule', 'GeneratedExample',
    'ValidationResult', 'TestResult',
    # Main classes
    'ProcedureAnalyzer', 'RuleGenerator', 'RuleValidator',
    'RuleTester', 'SyncRuleTester', 'RuleGenerationOrchestrator'
]
