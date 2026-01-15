"""
Stored Procedure Analyzer Agent.
Analyzes stored procedures and extracts semantic information.
"""

import re
from typing import List, Dict, Optional, Tuple
from .models import (
    ProcedureAnalysis, ProcedureType, ComplexityTier, ActionType,
    TemporalScope, TableSemantics, TableRole, ColumnSemantics, ColumnRole,
    JoinTemplate, ImplicitFilter, AggregationPattern, TemporalPattern
)


class ProcedureAnalyzer:
    """Analyzes stored procedures to extract semantic information."""

    # Procedure name patterns for classification
    REPORTING_PREFIXES = ['get', 'report', 'search', 'find', 'list', 'fetch', 'load', 'select', 'query']
    CRUD_PREFIXES = ['insert', 'update', 'delete', 'save', 'add', 'remove', 'create', 'modify']
    BATCH_PREFIXES = ['batch', 'process', 'sync', 'import', 'export', 'migrate', 'bulk']
    UTILITY_PREFIXES = ['log', 'audit', 'util', 'helper', 'validate', 'check']

    # Table role patterns
    LOOKUP_PATTERNS = ['types', 'status', 'category', 'categories', 'codes', 'lookups']
    AUDIT_PATTERNS = ['audit', 'history', 'log', 'archive']
    BRIDGE_PATTERNS = ['map', 'mapping', 'link', 'xref', 'relation']

    # Column patterns for semantic inference
    ID_SUFFIXES = ['id', 'key', 'code']
    DATE_SUFFIXES = ['date', 'utc', 'datetime', 'time', 'timestamp']
    FLAG_PREFIXES = ['is', 'has', 'can', 'should', 'allow']
    USER_ID_PATTERNS = ['userid', 'centraluserid', 'createdby', 'modifiedby', 'assignedto']

    # Common implicit filters
    COMMON_FILTERS = {
        'IsActive': ('=', 1, 'Only include active records'),
        'IsDeleted': ('=', 0, 'Exclude deleted records'),
        'Status': ('!=', 'Deleted', 'Exclude deleted status'),
    }

    # Temporal pattern mappings
    TEMPORAL_PATTERNS = {
        'today': "CAST({col} AS DATE) = CAST(GETDATE() AS DATE)",
        'yesterday': "CAST({col} AS DATE) = CAST(DATEADD(DAY, -1, GETDATE()) AS DATE)",
        'this week': "{col} >= DATEADD(WEEK, DATEDIFF(WEEK, 0, GETDATE()), 0)",
        'last week': "{col} >= DATEADD(WEEK, DATEDIFF(WEEK, 0, GETDATE()) - 1, 0) AND {col} < DATEADD(WEEK, DATEDIFF(WEEK, 0, GETDATE()), 0)",
        'this month': "MONTH({col}) = MONTH(GETDATE()) AND YEAR({col}) = YEAR(GETDATE())",
        'last month': "MONTH({col}) = MONTH(DATEADD(MONTH, -1, GETDATE())) AND YEAR({col}) = YEAR(DATEADD(MONTH, -1, GETDATE()))",
        'this year': "YEAR({col}) = YEAR(GETDATE())",
        'last year': "YEAR({col}) = YEAR(GETDATE()) - 1",
        'last 7 days': "{col} >= DATEADD(DAY, -7, GETDATE())",
        'last 30 days': "{col} >= DATEADD(DAY, -30, GETDATE())",
        'last 90 days': "{col} >= DATEADD(DAY, -90, GETDATE())",
    }

    # Aggregation patterns
    AGGREGATION_MAPPINGS = {
        ('how many', 'count of', 'number of', 'total count'): ('COUNT', '*'),
        ('how much', 'total amount', 'sum of'): ('SUM', 'amount'),
        ('average', 'avg', 'mean'): ('AVG', 'value'),
        ('maximum', 'max', 'highest', 'most'): ('MAX', 'value'),
        ('minimum', 'min', 'lowest', 'least'): ('MIN', 'value'),
    }

    def __init__(self):
        pass

    def analyze(self, procedure_doc: Dict) -> ProcedureAnalysis:
        """
        Analyze a stored procedure document and extract semantic information.

        Args:
            procedure_doc: MongoDB document from sql_stored_procedures

        Returns:
            ProcedureAnalysis with extracted semantic information
        """
        proc_name = procedure_doc.get('procedure_name', '')
        database = procedure_doc.get('database', '')
        definition = procedure_doc.get('definition', '')
        summary = procedure_doc.get('summary', '')
        tables = procedure_doc.get('tables_affected', [])
        operations = procedure_doc.get('operations', [])
        parameters = procedure_doc.get('parameters', [])
        keywords = procedure_doc.get('keywords', [])

        # Classify procedure
        proc_type = self._classify_procedure_type(proc_name, operations)
        complexity = self._classify_complexity(definition, tables, operations)
        nlq_relevance = self._calculate_nlq_relevance(proc_type, complexity)

        # Extract intent
        action_type = self._infer_action_type(proc_name, operations, definition)
        entity_focus = self._extract_entity_focus(proc_name, tables)
        temporal_scope = self._infer_temporal_scope(definition, parameters)

        # Analyze tables
        table_semantics = [self._analyze_table(t, definition) for t in tables]
        primary_table = self._identify_primary_table(tables, definition)

        # Analyze columns (from SELECT statements)
        columns = self._extract_columns(definition)

        # Extract patterns
        join_templates = self._extract_join_templates(definition)
        implicit_filters = self._extract_implicit_filters(definition)
        aggregation_patterns = self._infer_aggregation_patterns(definition, proc_name)
        temporal_patterns = self._identify_temporal_patterns(definition)

        # Map parameters to natural language
        param_mappings = self._map_parameters(parameters)

        # Generate trigger keywords
        trigger_keywords = self._generate_trigger_keywords(
            proc_name, entity_focus, summary, keywords
        )

        # Generate example questions
        example_questions = self._generate_example_questions(
            proc_name, entity_focus, action_type, parameters
        )

        return ProcedureAnalysis(
            procedure_name=proc_name,
            database=database,
            procedure_type=proc_type,
            complexity_tier=complexity,
            nlq_relevance=nlq_relevance,
            action_type=action_type,
            entity_focus=entity_focus,
            temporal_scope=temporal_scope,
            tables_used=table_semantics,
            primary_table=primary_table,
            columns=columns,
            join_templates=join_templates,
            aggregation_patterns=aggregation_patterns,
            temporal_patterns=temporal_patterns,
            implicit_filters=implicit_filters,
            parameter_mappings=param_mappings,
            trigger_keywords=trigger_keywords,
            example_questions=example_questions,
        )

    def _classify_procedure_type(self, proc_name: str, operations: List[str]) -> ProcedureType:
        """Classify procedure based on name and operations."""
        name_lower = proc_name.lower()

        # Remove common prefixes
        for prefix in ['usp_', 'sp_', 'proc_']:
            if name_lower.startswith(prefix):
                name_lower = name_lower[len(prefix):]
                break

        # Check patterns
        for prefix in self.REPORTING_PREFIXES:
            if name_lower.startswith(prefix):
                return ProcedureType.REPORTING

        for prefix in self.CRUD_PREFIXES:
            if name_lower.startswith(prefix):
                return ProcedureType.CRUD

        for prefix in self.BATCH_PREFIXES:
            if name_lower.startswith(prefix):
                return ProcedureType.BATCH

        for prefix in self.UTILITY_PREFIXES:
            if name_lower.startswith(prefix):
                return ProcedureType.UTILITY

        # Infer from operations
        if operations:
            if 'SELECT' in operations and len(operations) == 1:
                return ProcedureType.REPORTING
            if any(op in operations for op in ['INSERT', 'UPDATE', 'DELETE']):
                return ProcedureType.CRUD

        return ProcedureType.UTILITY

    def _classify_complexity(self, definition: str, tables: List[str], operations: List[str]) -> ComplexityTier:
        """Classify complexity based on SQL content."""
        if not definition:
            return ComplexityTier.SIMPLE

        definition_upper = definition.upper()

        # Multi-step indicators
        multi_step_indicators = [
            'BEGIN TRANSACTION', 'COMMIT', 'ROLLBACK',
            'CREATE TABLE #', 'CREATE TABLE @',  # Temp tables
            'WHILE ', 'CURSOR',
        ]
        if any(ind in definition_upper for ind in multi_step_indicators):
            return ComplexityTier.MULTI_STEP

        # Aggregate indicators
        if re.search(r'\bGROUP\s+BY\b', definition_upper):
            return ComplexityTier.AGGREGATE
        if re.search(r'\b(COUNT|SUM|AVG|MIN|MAX)\s*\(', definition_upper):
            return ComplexityTier.AGGREGATE

        # Multiple tables = more complex
        if len(tables) > 2:
            return ComplexityTier.AGGREGATE

        return ComplexityTier.SIMPLE

    def _calculate_nlq_relevance(self, proc_type: ProcedureType, complexity: ComplexityTier) -> float:
        """Calculate NLQ relevance score (0.0 - 1.0)."""
        base_scores = {
            ProcedureType.REPORTING: 0.9,
            ProcedureType.CRUD: 0.3,
            ProcedureType.BATCH: 0.4,
            ProcedureType.UTILITY: 0.1,
        }

        complexity_modifiers = {
            ComplexityTier.SIMPLE: 0.0,
            ComplexityTier.AGGREGATE: 0.05,
            ComplexityTier.MULTI_STEP: -0.1,
        }

        score = base_scores.get(proc_type, 0.5)
        score += complexity_modifiers.get(complexity, 0.0)

        return max(0.0, min(1.0, score))

    def _infer_action_type(self, proc_name: str, operations: List[str], definition: str) -> ActionType:
        """Infer the primary action type."""
        name_lower = proc_name.lower()
        def_upper = (definition or '').upper()

        if 'search' in name_lower or 'find' in name_lower:
            return ActionType.SEARCH

        if re.search(r'\bGROUP\s+BY\b', def_upper):
            return ActionType.AGGREGATE

        if 'compare' in name_lower or 'diff' in name_lower:
            return ActionType.COMPARE

        if 'trend' in name_lower or 'history' in name_lower:
            return ActionType.TREND

        return ActionType.RETRIEVE

    def _extract_entity_focus(self, proc_name: str, tables: List[str]) -> str:
        """Extract the primary entity (noun) this procedure deals with."""
        # Try to extract from procedure name
        name_lower = proc_name.lower()

        # Remove prefixes
        for prefix in ['usp_', 'sp_', 'proc_', 'get', 'find', 'search', 'list',
                       'insert', 'update', 'delete', 'add', 'remove', 'save']:
            if name_lower.startswith(prefix):
                name_lower = name_lower[len(prefix):]
            name_lower = name_lower.replace(prefix, '')

        # Extract noun from CamelCase
        words = re.findall(r'[A-Z][a-z]+|[a-z]+', proc_name)
        if words:
            # Filter out action verbs
            action_verbs = {'get', 'set', 'find', 'search', 'list', 'add', 'delete',
                           'update', 'insert', 'remove', 'save', 'load', 'fetch', 'by', 'all'}
            nouns = [w for w in words if w.lower() not in action_verbs]
            if nouns:
                return nouns[0]

        # Fall back to first table
        if tables:
            table_name = tables[0].replace('dbo.', '').replace('Central', '')
            return table_name

        return "Unknown"

    def _infer_temporal_scope(self, definition: str, parameters: List[Dict]) -> TemporalScope:
        """Infer the temporal scope of the procedure."""
        if not definition:
            return TemporalScope.CURRENT_STATE

        def_upper = definition.upper()

        # Check for date parameters
        date_params = [p for p in parameters if 'date' in p.get('name', '').lower()]
        if len(date_params) >= 2:  # Start and end date = range
            return TemporalScope.RANGE

        # Check for history patterns
        if 'HISTORY' in def_upper or 'AUDIT' in def_upper:
            return TemporalScope.HISTORICAL

        # Check for date filtering
        if re.search(r'WHERE.*DATE', def_upper):
            return TemporalScope.POINT_IN_TIME

        return TemporalScope.CURRENT_STATE

    def _analyze_table(self, table_name: str, definition: str) -> TableSemantics:
        """Analyze a table and determine its semantic role."""
        name_lower = table_name.lower()

        # Determine role
        role = TableRole.FACT  # Default

        if any(pattern in name_lower for pattern in self.LOOKUP_PATTERNS):
            role = TableRole.LOOKUP
        elif any(pattern in name_lower for pattern in self.AUDIT_PATTERNS):
            role = TableRole.AUDIT
        elif any(pattern in name_lower for pattern in self.BRIDGE_PATTERNS):
            role = TableRole.BRIDGE
        elif name_lower.endswith('s') and not any(
            pattern in name_lower for pattern in ['status', 'address']
        ):
            # Plural tables are often fact tables
            role = TableRole.FACT

        # Infer grain from table name
        grain = self._infer_grain(table_name)

        return TableSemantics(
            name=table_name,
            role=role,
            grain=grain,
        )

    def _identify_primary_table(self, tables: List[str], definition: str) -> Optional[str]:
        """Identify the primary table from the procedure."""
        if not tables:
            return None

        if len(tables) == 1:
            return tables[0]

        # Check FROM clause for the first table
        if definition:
            from_match = re.search(r'FROM\s+(\w+(?:\.\w+)?)', definition, re.IGNORECASE)
            if from_match:
                from_table = from_match.group(1)
                # Check if it matches any of our tables
                for table in tables:
                    if table.lower() == from_table.lower() or table.lower().endswith('.' + from_table.lower()):
                        return table

        # Fall back to first table
        return tables[0]

    def _infer_grain(self, table_name: str) -> str:
        """Infer the grain (row level) of a table."""
        name_lower = table_name.lower()

        if 'ticket' in name_lower:
            return 'per_ticket'
        if 'user' in name_lower:
            return 'per_user'
        if 'company' in name_lower or 'customer' in name_lower:
            return 'per_company'
        if 'order' in name_lower:
            return 'per_order'
        if 'daily' in name_lower or 'day' in name_lower:
            return 'per_day'
        if 'monthly' in name_lower or 'month' in name_lower:
            return 'per_month'

        return 'per_record'

    def _extract_columns(self, definition: str) -> List[ColumnSemantics]:
        """Extract column information from SELECT statements."""
        columns = []
        if not definition:
            return columns

        # Find SELECT columns
        select_match = re.search(r'SELECT\s+(.*?)\s+FROM', definition, re.IGNORECASE | re.DOTALL)
        if not select_match:
            return columns

        select_clause = select_match.group(1)

        # Parse individual columns
        col_parts = re.split(r',(?![^(]*\))', select_clause)  # Split on commas not in parentheses

        for col_part in col_parts:
            col_part = col_part.strip()
            if not col_part or col_part == '*':
                continue

            # Extract column name (handle aliases)
            alias_match = re.search(r'\bAS\s+(\w+)\s*$', col_part, re.IGNORECASE)
            if alias_match:
                col_name = alias_match.group(1)
            else:
                # Get last part after dot
                col_name = col_part.split('.')[-1].split()[-1]

            col_name = col_name.strip('[]')

            # Determine role
            role = self._infer_column_role(col_name, col_part)

            # Check if it's a calculation
            calculation = None
            if re.search(r'\(.*\)', col_part):
                calculation = col_part

            columns.append(ColumnSemantics(
                name=col_name,
                role=role,
                display_name=self._to_display_name(col_name),
                calculation=calculation,
                is_temporal='date' in col_name.lower() or 'time' in col_name.lower(),
            ))

        return columns[:20]  # Limit to first 20 columns

    def _infer_column_role(self, col_name: str, col_expr: str) -> ColumnRole:
        """Infer the semantic role of a column."""
        name_lower = col_name.lower()
        expr_lower = col_expr.lower()

        # Check for aggregations
        if any(agg in expr_lower for agg in ['count(', 'sum(', 'avg(', 'min(', 'max(']):
            return ColumnRole.MEASURE

        # Check patterns
        if any(name_lower.endswith(suffix) for suffix in self.ID_SUFFIXES):
            return ColumnRole.IDENTIFIER

        if any(name_lower.endswith(suffix) for suffix in self.DATE_SUFFIXES):
            return ColumnRole.TIMESTAMP

        if any(name_lower.startswith(prefix) for prefix in self.FLAG_PREFIXES):
            return ColumnRole.FLAG

        if 'status' in name_lower or 'state' in name_lower:
            return ColumnRole.STATUS

        return ColumnRole.DIMENSION

    def _to_display_name(self, column_name: str) -> str:
        """Convert column name to display-friendly format."""
        # Split CamelCase and underscores
        words = re.findall(r'[A-Z][a-z]+|[a-z]+|[A-Z]+(?=[A-Z]|$)', column_name)
        if not words:
            words = column_name.split('_')

        return ' '.join(w.capitalize() for w in words)

    def _extract_join_templates(self, definition: str) -> List[JoinTemplate]:
        """Extract JOIN patterns from definition."""
        templates = []
        if not definition:
            return templates

        # Find JOINs
        join_pattern = r'((?:LEFT|RIGHT|INNER|OUTER|CROSS)?\s*JOIN)\s+(\w+(?:\.\w+)?)\s+(?:AS\s+)?(\w+)?\s+ON\s+([^WHERE|JOIN|ORDER|GROUP]+)'
        matches = re.findall(join_pattern, definition, re.IGNORECASE)

        for match in matches[:10]:  # Limit to 10 joins
            join_type = match[0].strip().upper() or 'INNER JOIN'
            target_table = match[1]
            alias = match[2] or target_table
            condition = match[3].strip()

            # Extract source table from condition
            source_match = re.search(r'(\w+)\.', condition)
            source_table = source_match.group(1) if source_match else ''

            templates.append(JoinTemplate(
                source_table=source_table,
                target_table=target_table,
                join_type=join_type.replace('JOIN', '').strip() or 'INNER',
                condition=condition,
                purpose=f"Join {target_table}",
            ))

        return templates

    def _extract_implicit_filters(self, definition: str) -> List[ImplicitFilter]:
        """Extract implicit filters from WHERE clauses."""
        filters = []
        if not definition:
            return filters

        # Find WHERE clause
        where_match = re.search(r'WHERE\s+(.+?)(?:ORDER|GROUP|HAVING|$)', definition, re.IGNORECASE | re.DOTALL)
        if not where_match:
            return filters

        where_clause = where_match.group(1)

        # Look for common filter patterns
        for col_name, (operator, value, desc) in self.COMMON_FILTERS.items():
            if re.search(rf'\b{col_name}\s*{operator}\s*{value}\b', where_clause, re.IGNORECASE):
                filters.append(ImplicitFilter(
                    column=col_name,
                    operator=operator,
                    value=value,
                    frequency=0.8,  # Assume high frequency if found
                    description=desc,
                ))

        return filters

    def _infer_aggregation_patterns(self, definition: str, proc_name: str) -> List[AggregationPattern]:
        """Infer aggregation patterns from procedure."""
        patterns = []
        if not definition:
            return patterns

        def_upper = definition.upper()

        for phrases, (func, typical_col) in self.AGGREGATION_MAPPINGS.items():
            if func + '(' in def_upper:
                patterns.append(AggregationPattern(
                    trigger_phrases=list(phrases),
                    sql_function=func,
                    typical_group_by=[],  # Would need more analysis
                ))

        return patterns

    def _identify_temporal_patterns(self, definition: str) -> List[TemporalPattern]:
        """Identify temporal patterns that could be useful."""
        patterns = []

        # Find date columns used in WHERE
        date_cols = re.findall(r'(\w+Date|\w+UTC|\w+DateTime)\s*[>=<]', definition or '', re.IGNORECASE)

        if date_cols:
            # Add common temporal patterns
            for nl, sql in list(self.TEMPORAL_PATTERNS.items())[:5]:
                patterns.append(TemporalPattern(
                    natural_language=nl,
                    sql_template=sql,
                ))

        return patterns

    def _map_parameters(self, parameters: List[Dict]) -> Dict[str, Dict]:
        """Map parameters to natural language references."""
        mappings = {}

        for param in parameters:
            param_name = param.get('name', '')
            param_type = param.get('type', '')

            nl_refs = []
            resolution = None

            # Infer natural language references
            name_lower = param_name.lower().replace('@', '')

            if 'date' in name_lower:
                if 'start' in name_lower:
                    nl_refs = ['from', 'starting', 'after', 'since']
                elif 'end' in name_lower:
                    nl_refs = ['to', 'until', 'before', 'ending']
                else:
                    nl_refs = ['on', 'date', 'when']

            elif 'id' in name_lower:
                if 'user' in name_lower:
                    nl_refs = ['user', 'person', 'employee', 'by']
                    resolution = 'lookup_user'
                elif 'ticket' in name_lower:
                    nl_refs = ['ticket', 'issue', 'case']
                elif 'company' in name_lower:
                    nl_refs = ['company', 'customer', 'client']

            elif 'name' in name_lower:
                nl_refs = ['named', 'called']
                resolution = 'fuzzy_match'

            if nl_refs:
                mappings[param_name] = {
                    'nl_phrases': nl_refs,
                    'type': param_type,
                    'resolution': resolution,
                }

        return mappings

    def _generate_trigger_keywords(self, proc_name: str, entity: str,
                                    summary: str, existing_keywords: List[str]) -> List[str]:
        """Generate keywords that should trigger this procedure's rules."""
        keywords = set(existing_keywords)

        # Add entity
        keywords.add(entity.lower())

        # Extract from procedure name
        words = re.findall(r'[A-Z][a-z]+|[a-z]+', proc_name)
        action_verbs = {'get', 'set', 'find', 'search', 'list', 'add', 'delete',
                       'update', 'insert', 'remove', 'save', 'load', 'fetch', 'usp', 'sp'}
        for word in words:
            if word.lower() not in action_verbs and len(word) > 2:
                keywords.add(word.lower())

        # Extract from summary
        if summary:
            # Simple word extraction
            summary_words = re.findall(r'\b[a-zA-Z]{4,}\b', summary)
            for word in summary_words[:5]:
                keywords.add(word.lower())

        return list(keywords)[:15]  # Limit to 15 keywords

    def _generate_example_questions(self, proc_name: str, entity: str,
                                     action: ActionType, parameters: List[Dict]) -> List[str]:
        """Generate example natural language questions."""
        questions = []

        # Base question patterns by action type
        if action == ActionType.RETRIEVE:
            questions.append(f"Show me all {entity}s")
            questions.append(f"Get {entity} information")

        elif action == ActionType.AGGREGATE:
            questions.append(f"How many {entity}s are there")
            questions.append(f"Count of {entity}s")

        elif action == ActionType.SEARCH:
            questions.append(f"Find {entity}s matching...")
            questions.append(f"Search for {entity}s")

        # Add parameter-based questions
        for param in parameters[:3]:
            param_name = param.get('name', '').replace('@', '')
            if 'date' in param_name.lower():
                questions.append(f"{entity}s from last month")
            elif 'id' in param_name.lower():
                questions.append(f"{entity}s for specific user")

        return questions[:5]
