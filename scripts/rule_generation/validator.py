"""
Rule Validator.
Validates generated rules before upload to MongoDB.
"""

import re
from typing import List, Optional
from .models import GeneratedRule, GeneratedExample, ValidationResult


class RuleValidator:
    """Validates rules and examples before database insertion."""

    # Required fields for rules (uses 'type' not 'rule_type' to match existing schema)
    REQUIRED_RULE_FIELDS = ['rule_id', 'database', 'type', 'description', 'rule_text']

    # Required fields for examples
    REQUIRED_EXAMPLE_FIELDS = ['example_id', 'database', 'question', 'sql']

    # Valid rule types
    VALID_RULE_TYPES = ['assistance', 'constraint', 'example']

    # Valid priorities
    VALID_PRIORITIES = ['high', 'normal', 'low']

    # SQL keywords that should appear in valid SQL
    SQL_KEYWORDS = ['SELECT', 'FROM', 'INSERT', 'UPDATE', 'DELETE', 'WITH']

    # Dangerous patterns to check for
    DANGEROUS_PATTERNS = [
        r'\bDROP\s+TABLE\b',
        r'\bDROP\s+DATABASE\b',
        r'\bTRUNCATE\b',
        r'\bDELETE\s+FROM\s+\w+\s*$',  # DELETE without WHERE
        r'\bUPDATE\s+\w+\s+SET\s+.*(?<!WHERE)',  # UPDATE without WHERE (simplified)
        r'--.*DROP',
        r'/\*.*DROP.*\*/',
    ]

    def __init__(self, strict_mode: bool = True):
        """
        Initialize validator.

        Args:
            strict_mode: If True, fail on warnings. If False, only fail on errors.
        """
        self.strict_mode = strict_mode

    def validate_rule(self, rule: GeneratedRule) -> ValidationResult:
        """
        Validate a generated rule.

        Args:
            rule: GeneratedRule to validate

        Returns:
            ValidationResult with validation status
        """
        errors = []
        warnings = []

        # Check required fields
        doc = rule.to_mongodb_doc()
        for field in self.REQUIRED_RULE_FIELDS:
            if not doc.get(field):
                errors.append(f"Missing required field: {field}")

        # Validate rule_id format
        if rule.rule_id:
            if not re.match(r'^[\w\-\.]+$', rule.rule_id):
                errors.append(f"Invalid rule_id format: {rule.rule_id}")

        # Validate rule_type (stored as 'type' in MongoDB doc)
        if rule.rule_type not in self.VALID_RULE_TYPES:
            errors.append(f"Invalid type: {rule.rule_type}")

        # Validate priority
        if rule.priority not in self.VALID_PRIORITIES:
            warnings.append(f"Non-standard priority: {rule.priority}")

        # Validate database name
        if rule.database:
            if not re.match(r'^[\w\.\-]+$', rule.database):
                errors.append(f"Invalid database name: {rule.database}")

        # Validate description length
        if rule.description:
            if len(rule.description) < 10:
                warnings.append("Description is very short")
            if len(rule.description) > 500:
                warnings.append("Description is very long")

        # Validate rule_text
        if rule.rule_text:
            if len(rule.rule_text) < 20:
                warnings.append("Rule text is very short")
            if len(rule.rule_text) > 5000:
                warnings.append("Rule text is very long")

        # Validate keywords
        if not rule.trigger_keywords:
            warnings.append("No trigger keywords defined")
        else:
            for kw in rule.trigger_keywords:
                if len(kw) < 2:
                    warnings.append(f"Very short keyword: {kw}")

        # Validate auto_fix if present
        if rule.auto_fix:
            if 'pattern' not in rule.auto_fix:
                errors.append("auto_fix missing 'pattern'")
            if 'replacement' not in rule.auto_fix:
                errors.append("auto_fix missing 'replacement'")

            # Try to compile the regex
            if 'pattern' in rule.auto_fix:
                try:
                    re.compile(rule.auto_fix['pattern'])
                except re.error as e:
                    errors.append(f"Invalid auto_fix regex: {e}")

        # Validate example if present
        if rule.example_sql:
            sql_validation = self._validate_sql(rule.example_sql)
            errors.extend(sql_validation.errors)
            warnings.extend(sql_validation.warnings)

        # Check confidence
        if rule.confidence < 0.3:
            warnings.append(f"Low confidence: {rule.confidence}")

        # Determine validity
        is_valid = len(errors) == 0
        if self.strict_mode and len(warnings) > 3:
            is_valid = False

        return ValidationResult(
            is_valid=is_valid,
            errors=errors,
            warnings=warnings
        )

    def validate_example(self, example: GeneratedExample) -> ValidationResult:
        """
        Validate a generated example.

        Args:
            example: GeneratedExample to validate

        Returns:
            ValidationResult with validation status
        """
        errors = []
        warnings = []

        # Check required fields
        doc = example.to_mongodb_doc()
        for field in self.REQUIRED_EXAMPLE_FIELDS:
            if not doc.get(field):
                errors.append(f"Missing required field: {field}")

        # Validate example_id format
        if example.example_id:
            if not re.match(r'^[\w\-\.]+$', example.example_id):
                errors.append(f"Invalid example_id format: {example.example_id}")

        # Validate question
        if example.question:
            if len(example.question) < 10:
                warnings.append("Question is very short")
            if len(example.question) > 500:
                warnings.append("Question is very long")
            if not example.question.strip():
                errors.append("Question is empty or whitespace")

        # Validate SQL
        if example.sql:
            sql_validation = self._validate_sql(example.sql)
            errors.extend(sql_validation.errors)
            warnings.extend(sql_validation.warnings)
        else:
            errors.append("SQL is empty")

        # Check confidence
        if example.confidence < 0.3:
            warnings.append(f"Low confidence: {example.confidence}")

        # Determine validity
        is_valid = len(errors) == 0
        if self.strict_mode and len(warnings) > 3:
            is_valid = False

        return ValidationResult(
            is_valid=is_valid,
            errors=errors,
            warnings=warnings
        )

    def _validate_sql(self, sql: str) -> ValidationResult:
        """
        Validate SQL syntax (basic checks).

        Args:
            sql: SQL string to validate

        Returns:
            ValidationResult
        """
        errors = []
        warnings = []

        if not sql or not sql.strip():
            errors.append("SQL is empty")
            return ValidationResult(is_valid=False, errors=errors, warnings=warnings)

        sql_upper = sql.upper()

        # Check for at least one SQL keyword
        has_keyword = any(kw in sql_upper for kw in self.SQL_KEYWORDS)
        if not has_keyword:
            errors.append("SQL doesn't contain any recognized SQL keywords")

        # Check for dangerous patterns
        for pattern in self.DANGEROUS_PATTERNS:
            if re.search(pattern, sql, re.IGNORECASE):
                errors.append(f"Dangerous SQL pattern detected: {pattern}")

        # Check for balanced parentheses
        if sql.count('(') != sql.count(')'):
            errors.append("Unbalanced parentheses in SQL")

        # Check for PostgreSQL syntax (should be T-SQL)
        postgres_patterns = [
            r'\bEXTRACT\s*\(',
            r'\bLIMIT\s+\d+',
            r'::\w+',  # Type casting
            r'\bILIKE\b',
        ]
        for pattern in postgres_patterns:
            if re.search(pattern, sql, re.IGNORECASE):
                warnings.append(f"Possible PostgreSQL syntax detected: {pattern}")

        # Check for common T-SQL best practices
        if 'SELECT *' in sql_upper and 'EXISTS' not in sql_upper:
            warnings.append("SELECT * should be avoided; list specific columns")

        # Check for missing semicolon (optional but good practice)
        if not sql.strip().endswith(';'):
            pass  # Not an error, just noting

        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings
        )

    def validate_batch(self, rules: List[GeneratedRule],
                        examples: List[GeneratedExample]) -> dict:
        """
        Validate a batch of rules and examples.

        Args:
            rules: List of rules to validate
            examples: List of examples to validate

        Returns:
            Dict with validation summary
        """
        results = {
            'rules': {
                'total': len(rules),
                'valid': 0,
                'invalid': 0,
                'errors': [],
                'warnings': []
            },
            'examples': {
                'total': len(examples),
                'valid': 0,
                'invalid': 0,
                'errors': [],
                'warnings': []
            }
        }

        # Validate rules
        for rule in rules:
            result = self.validate_rule(rule)
            if result.is_valid:
                results['rules']['valid'] += 1
            else:
                results['rules']['invalid'] += 1
                for error in result.errors:
                    results['rules']['errors'].append(f"{rule.rule_id}: {error}")

            for warning in result.warnings:
                results['rules']['warnings'].append(f"{rule.rule_id}: {warning}")

        # Validate examples
        for example in examples:
            result = self.validate_example(example)
            if result.is_valid:
                results['examples']['valid'] += 1
            else:
                results['examples']['invalid'] += 1
                for error in result.errors:
                    results['examples']['errors'].append(f"{example.example_id}: {error}")

            for warning in result.warnings:
                results['examples']['warnings'].append(f"{example.example_id}: {warning}")

        return results
