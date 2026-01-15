"""
Rule Generator Agent.
Generates SQL rules and examples from procedure analysis.
"""

import hashlib
from typing import List, Dict, Optional, Tuple
from datetime import datetime
from .models import (
    ProcedureAnalysis, GeneratedRule, GeneratedExample,
    ProcedureType, ComplexityTier, ActionType, TableRole
)


class RuleGenerator:
    """Generates SQL rules from procedure analysis."""

    def __init__(self, database_prefix: str = ""):
        self.database_prefix = database_prefix
        self.generated_count = 0

    def generate_rules(self, analysis: ProcedureAnalysis) -> List[GeneratedRule]:
        """
        Generate rules from procedure analysis.

        Args:
            analysis: ProcedureAnalysis from analyzer

        Returns:
            List of GeneratedRule objects
        """
        rules = []

        # Only generate rules for high-relevance procedures
        if analysis.nlq_relevance < 0.3:
            return rules

        # 1. Generate table relationship rules
        if len(analysis.tables_used) > 1:
            join_rule = self._generate_join_rule(analysis)
            if join_rule:
                rules.append(join_rule)

        # 2. Generate implicit filter rules
        for filter_info in analysis.implicit_filters:
            filter_rule = self._generate_filter_rule(analysis, filter_info)
            if filter_rule:
                rules.append(filter_rule)

        # 3. Generate aggregation rules for reporting procedures
        if analysis.procedure_type == ProcedureType.REPORTING:
            if analysis.aggregation_patterns:
                agg_rule = self._generate_aggregation_rule(analysis)
                if agg_rule:
                    rules.append(agg_rule)

        # 4. Generate temporal pattern rules
        if analysis.temporal_patterns:
            temporal_rule = self._generate_temporal_rule(analysis)
            if temporal_rule:
                rules.append(temporal_rule)

        # 5. Generate entity-specific guidance rule
        if analysis.nlq_relevance >= 0.7:
            guidance_rule = self._generate_entity_guidance_rule(analysis)
            if guidance_rule:
                rules.append(guidance_rule)

        return rules

    def generate_examples(self, analysis: ProcedureAnalysis) -> List[GeneratedExample]:
        """
        Generate example question/SQL pairs from procedure analysis.

        Args:
            analysis: ProcedureAnalysis from analyzer

        Returns:
            List of GeneratedExample objects
        """
        examples = []

        # Only generate examples for high-relevance procedures
        if analysis.nlq_relevance < 0.5:
            return examples

        # Generate examples based on procedure type
        if analysis.procedure_type == ProcedureType.REPORTING:
            # Generate retrieval example
            retrieval_example = self._generate_retrieval_example(analysis)
            if retrieval_example:
                examples.append(retrieval_example)

            # Generate aggregation example if applicable
            if analysis.action_type == ActionType.AGGREGATE:
                agg_example = self._generate_aggregation_example(analysis)
                if agg_example:
                    examples.append(agg_example)

            # Generate temporal example if applicable
            if analysis.temporal_patterns:
                temporal_example = self._generate_temporal_example(analysis)
                if temporal_example:
                    examples.append(temporal_example)

        return examples

    def _generate_rule_id(self, prefix: str, analysis: ProcedureAnalysis) -> str:
        """Generate a unique rule ID."""
        self.generated_count += 1
        hash_input = f"{analysis.procedure_name}_{prefix}_{self.generated_count}"
        hash_suffix = hashlib.md5(hash_input.encode()).hexdigest()[:8]
        return f"{analysis.database}-{prefix}-{hash_suffix}"

    def _generate_join_rule(self, analysis: ProcedureAnalysis) -> Optional[GeneratedRule]:
        """Generate a rule about table joins."""
        if not analysis.join_templates:
            return None

        # Build join guidance text
        join_texts = []
        for jt in analysis.join_templates[:5]:
            join_texts.append(f"- {jt.join_type} JOIN {jt.target_table} ON {jt.condition}")

        rule_text = f"""When querying {analysis.entity_focus} data, use these JOIN patterns:
{chr(10).join(join_texts)}

Primary table: {analysis.primary_table or analysis.tables_used[0].name if analysis.tables_used else 'Unknown'}
"""

        # Extract table names for triggers
        trigger_tables = [t.name for t in analysis.tables_used[:5]]

        return GeneratedRule(
            rule_id=self._generate_rule_id("join", analysis),
            database=analysis.database,
            rule_type="assistance",
            priority="normal",
            description=f"JOIN patterns for {analysis.entity_focus} queries",
            rule_text=rule_text,
            trigger_keywords=analysis.trigger_keywords[:5],
            trigger_tables=trigger_tables,
            source_procedure=analysis.procedure_name,
            confidence=analysis.nlq_relevance,
        )

    def _generate_filter_rule(self, analysis: ProcedureAnalysis,
                               filter_info) -> Optional[GeneratedRule]:
        """Generate a rule about implicit filters."""
        rule_text = f"""When querying {analysis.entity_focus}, apply this filter:
{filter_info.column} {filter_info.operator} {filter_info.value}

Reason: {filter_info.description}
This filter appears in {int(filter_info.frequency * 100)}% of queries.
"""

        return GeneratedRule(
            rule_id=self._generate_rule_id("filter", analysis),
            database=analysis.database,
            rule_type="constraint",
            priority="high",
            description=f"Default filter: {filter_info.description}",
            rule_text=rule_text,
            trigger_keywords=analysis.trigger_keywords[:5],
            trigger_tables=[t.name for t in analysis.tables_used[:3]],
            trigger_columns=[filter_info.column],
            auto_fix={
                "pattern": rf"FROM\s+(\w+{analysis.entity_focus}\w*)\s+(?!WHERE)",
                "replacement": rf"FROM \1 WHERE {filter_info.column} {filter_info.operator} {filter_info.value}"
            } if filter_info.operator == '=' else None,
            source_procedure=analysis.procedure_name,
            confidence=filter_info.frequency,
        )

    def _generate_aggregation_rule(self, analysis: ProcedureAnalysis) -> Optional[GeneratedRule]:
        """Generate a rule about aggregation patterns."""
        if not analysis.aggregation_patterns:
            return None

        agg_texts = []
        for ap in analysis.aggregation_patterns:
            phrases = ", ".join(f'"{p}"' for p in ap.trigger_phrases[:3])
            agg_texts.append(f"- For {phrases}: Use {ap.sql_function}()")

        rule_text = f"""Aggregation patterns for {analysis.entity_focus}:
{chr(10).join(agg_texts)}

Remember to include GROUP BY when using aggregate functions with other columns.
"""

        return GeneratedRule(
            rule_id=self._generate_rule_id("agg", analysis),
            database=analysis.database,
            rule_type="assistance",
            priority="normal",
            description=f"Aggregation patterns for {analysis.entity_focus}",
            rule_text=rule_text,
            trigger_keywords=["how many", "count", "total", "sum", "average"] + analysis.trigger_keywords[:3],
            trigger_tables=[t.name for t in analysis.tables_used[:3]],
            source_procedure=analysis.procedure_name,
            confidence=analysis.nlq_relevance,
        )

    def _generate_temporal_rule(self, analysis: ProcedureAnalysis) -> Optional[GeneratedRule]:
        """Generate a rule about temporal patterns."""
        if not analysis.temporal_patterns:
            return None

        pattern_texts = []
        for tp in analysis.temporal_patterns[:5]:
            pattern_texts.append(f'- "{tp.natural_language}": {tp.sql_template}')

        # Find date columns
        date_columns = [c.name for c in analysis.columns if c.is_temporal][:3]
        date_col_text = ", ".join(date_columns) if date_columns else "AddTicketDate, CreateDate"

        rule_text = f"""Temporal patterns for {analysis.entity_focus}:
{chr(10).join(pattern_texts)}

Use these date columns: {date_col_text}
Always use T-SQL date functions (YEAR, MONTH, DATEADD) not EXTRACT.
"""

        return GeneratedRule(
            rule_id=self._generate_rule_id("temporal", analysis),
            database=analysis.database,
            rule_type="assistance",
            priority="normal",
            description=f"Date filtering patterns for {analysis.entity_focus}",
            rule_text=rule_text,
            trigger_keywords=["today", "yesterday", "last week", "last month", "this year", "last year"] + analysis.trigger_keywords[:3],
            trigger_tables=[t.name for t in analysis.tables_used[:3]],
            trigger_columns=date_columns,
            source_procedure=analysis.procedure_name,
            confidence=analysis.nlq_relevance,
        )

    def _generate_entity_guidance_rule(self, analysis: ProcedureAnalysis) -> Optional[GeneratedRule]:
        """Generate general guidance for querying this entity."""
        # Build comprehensive guidance
        primary_table = analysis.primary_table or (analysis.tables_used[0].name if analysis.tables_used else None)
        if not primary_table:
            return None

        # Identify key columns
        id_cols = [c for c in analysis.columns if c.role.value == 'identifier'][:3]
        measure_cols = [c for c in analysis.columns if c.role.value == 'measure'][:3]
        dim_cols = [c for c in analysis.columns if c.role.value == 'dimension'][:5]

        guidance_parts = [f"Primary table: {primary_table}"]

        if id_cols:
            guidance_parts.append(f"Key columns: {', '.join(c.name for c in id_cols)}")
        if measure_cols:
            guidance_parts.append(f"Measure columns: {', '.join(c.name for c in measure_cols)}")
        if dim_cols:
            guidance_parts.append(f"Descriptive columns: {', '.join(c.name for c in dim_cols)}")

        # Add related tables
        related = [t.name for t in analysis.tables_used if t.name != primary_table][:5]
        if related:
            guidance_parts.append(f"Related tables: {', '.join(related)}")

        rule_text = f"""Guidance for {analysis.entity_focus} queries:

{chr(10).join(guidance_parts)}

Typical query patterns:
- List all: SELECT * FROM {primary_table}
- Count: SELECT COUNT(*) FROM {primary_table}
- By user: JOIN CentralUsers ON {primary_table}.CentralUserID = CentralUsers.CentralUserID
"""

        return GeneratedRule(
            rule_id=self._generate_rule_id("guide", analysis),
            database=analysis.database,
            rule_type="assistance",
            priority="low",
            description=f"General guidance for {analysis.entity_focus} queries",
            rule_text=rule_text,
            trigger_keywords=analysis.trigger_keywords,
            trigger_tables=[primary_table],
            source_procedure=analysis.procedure_name,
            confidence=analysis.nlq_relevance,
        )

    def _generate_retrieval_example(self, analysis: ProcedureAnalysis) -> Optional[GeneratedExample]:
        """Generate a basic retrieval example."""
        primary_table = analysis.primary_table or (analysis.tables_used[0].name if analysis.tables_used else None)
        if not primary_table:
            return None

        # Build a simple SELECT
        columns = [c.name for c in analysis.columns[:5]] if analysis.columns else ['*']
        col_list = ', '.join(columns)

        question = f"Show me all {analysis.entity_focus.lower()}s"
        sql = f"SELECT {col_list} FROM {primary_table}"

        # Add a simple filter if we have implicit filters
        if analysis.implicit_filters:
            f = analysis.implicit_filters[0]
            sql += f" WHERE {f.column} {f.operator} {f.value}"

        return GeneratedExample(
            example_id=self._generate_rule_id("ex-list", analysis),
            database=analysis.database,
            question=question,
            sql=sql,
            source_procedure=analysis.procedure_name,
            tables_used=[primary_table],
            complexity="simple",
            confidence=analysis.nlq_relevance,
        )

    def _generate_aggregation_example(self, analysis: ProcedureAnalysis) -> Optional[GeneratedExample]:
        """Generate an aggregation example."""
        primary_table = analysis.primary_table or (analysis.tables_used[0].name if analysis.tables_used else None)
        if not primary_table:
            return None

        question = f"How many {analysis.entity_focus.lower()}s are there"
        sql = f"SELECT COUNT(*) AS Total{analysis.entity_focus}s FROM {primary_table}"

        if analysis.implicit_filters:
            f = analysis.implicit_filters[0]
            sql += f" WHERE {f.column} {f.operator} {f.value}"

        return GeneratedExample(
            example_id=self._generate_rule_id("ex-count", analysis),
            database=analysis.database,
            question=question,
            sql=sql,
            source_procedure=analysis.procedure_name,
            tables_used=[primary_table],
            complexity="simple",
            confidence=analysis.nlq_relevance,
        )

    def _generate_temporal_example(self, analysis: ProcedureAnalysis) -> Optional[GeneratedExample]:
        """Generate a temporal filtering example."""
        primary_table = analysis.primary_table or (analysis.tables_used[0].name if analysis.tables_used else None)
        if not primary_table:
            return None

        # Find a date column
        date_cols = [c.name for c in analysis.columns if c.is_temporal]
        date_col = date_cols[0] if date_cols else "AddTicketDate"

        question = f"How many {analysis.entity_focus.lower()}s were created last month"
        sql = f"""SELECT COUNT(*) AS Total FROM {primary_table}
WHERE MONTH({date_col}) = MONTH(DATEADD(MONTH, -1, GETDATE()))
  AND YEAR({date_col}) = YEAR(DATEADD(MONTH, -1, GETDATE()))"""

        return GeneratedExample(
            example_id=self._generate_rule_id("ex-temporal", analysis),
            database=analysis.database,
            question=question,
            sql=sql,
            source_procedure=analysis.procedure_name,
            tables_used=[primary_table],
            complexity="aggregate",
            confidence=analysis.nlq_relevance * 0.9,  # Slightly lower confidence
        )
