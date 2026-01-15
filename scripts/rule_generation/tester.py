"""
Rule Tester.
Tests generated rules by making queries through the Python API.
"""

import asyncio
import aiohttp
from typing import List, Dict, Optional
from datetime import datetime
from .models import GeneratedRule, GeneratedExample, TestResult


class RuleTester:
    """Tests rules by querying the SQL pipeline API."""

    def __init__(self, api_base_url: str = "http://localhost:8001"):
        """
        Initialize tester.

        Args:
            api_base_url: Base URL for the Python API
        """
        self.api_base_url = api_base_url
        self.query_endpoint = f"{api_base_url}/api/sql/query"
        self.test_results: List[TestResult] = []

    async def test_rule(self, rule: GeneratedRule,
                        test_question: Optional[str] = None) -> TestResult:
        """
        Test a rule by querying the API.

        Args:
            rule: Rule to test
            test_question: Optional custom question. If not provided, uses rule's example.

        Returns:
            TestResult with success/failure info
        """
        # Determine question to use
        question = test_question or rule.example_question
        if not question:
            # Generate a question from trigger keywords
            if rule.trigger_keywords:
                question = f"Show me {' '.join(rule.trigger_keywords[:3])}"
            else:
                return TestResult(
                    rule_id=rule.rule_id,
                    test_question="",
                    success=False,
                    error="No test question available"
                )

        start_time = datetime.now()

        try:
            async with aiohttp.ClientSession() as session:
                payload = {
                    "question": question,
                    "database": rule.database,
                    "execute": False  # Don't execute, just generate SQL
                }

                async with session.post(
                    self.query_endpoint,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=60)
                ) as response:
                    elapsed_ms = (datetime.now() - start_time).total_seconds() * 1000

                    if response.status == 200:
                        result = await response.json()
                        return TestResult(
                            rule_id=rule.rule_id,
                            test_question=question,
                            success=True,
                            generated_sql=result.get("sql"),
                            execution_result=result.get("result"),
                            response_time_ms=elapsed_ms
                        )
                    else:
                        error_text = await response.text()
                        return TestResult(
                            rule_id=rule.rule_id,
                            test_question=question,
                            success=False,
                            error=f"API returned {response.status}: {error_text}",
                            response_time_ms=elapsed_ms
                        )

        except asyncio.TimeoutError:
            return TestResult(
                rule_id=rule.rule_id,
                test_question=question,
                success=False,
                error="Request timed out after 60 seconds",
                response_time_ms=60000
            )
        except Exception as e:
            return TestResult(
                rule_id=rule.rule_id,
                test_question=question,
                success=False,
                error=str(e),
                response_time_ms=(datetime.now() - start_time).total_seconds() * 1000
            )

    async def test_example(self, example: GeneratedExample) -> TestResult:
        """
        Test an example by querying the API and comparing results.

        Args:
            example: Example to test

        Returns:
            TestResult with success/failure info
        """
        start_time = datetime.now()

        try:
            async with aiohttp.ClientSession() as session:
                payload = {
                    "question": example.question,
                    "database": example.database,
                    "execute": False
                }

                async with session.post(
                    self.query_endpoint,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=60)
                ) as response:
                    elapsed_ms = (datetime.now() - start_time).total_seconds() * 1000

                    if response.status == 200:
                        result = await response.json()
                        generated_sql = result.get("sql", "")

                        # Check if the generated SQL uses the same tables
                        tables_match = self._check_tables_used(
                            generated_sql,
                            example.tables_used
                        )

                        return TestResult(
                            rule_id=example.example_id,
                            test_question=example.question,
                            success=tables_match,
                            generated_sql=generated_sql,
                            execution_result={
                                "expected_sql": example.sql,
                                "tables_match": tables_match
                            },
                            response_time_ms=elapsed_ms
                        )
                    else:
                        error_text = await response.text()
                        return TestResult(
                            rule_id=example.example_id,
                            test_question=example.question,
                            success=False,
                            error=f"API returned {response.status}: {error_text}",
                            response_time_ms=elapsed_ms
                        )

        except asyncio.TimeoutError:
            return TestResult(
                rule_id=example.example_id,
                test_question=example.question,
                success=False,
                error="Request timed out after 60 seconds",
                response_time_ms=60000
            )
        except Exception as e:
            return TestResult(
                rule_id=example.example_id,
                test_question=example.question,
                success=False,
                error=str(e),
                response_time_ms=(datetime.now() - start_time).total_seconds() * 1000
            )

    def _check_tables_used(self, sql: str, expected_tables: List[str]) -> bool:
        """
        Check if the SQL uses the expected tables.

        Args:
            sql: Generated SQL
            expected_tables: Tables that should be used

        Returns:
            True if at least one expected table is found
        """
        if not sql or not expected_tables:
            return False

        sql_upper = sql.upper()
        for table in expected_tables:
            if table.upper() in sql_upper:
                return True
        return False

    async def test_batch(self, rules: List[GeneratedRule],
                         examples: List[GeneratedExample],
                         batch_size: int = 5) -> Dict:
        """
        Test a batch of rules and examples.

        Args:
            rules: Rules to test
            examples: Examples to test
            batch_size: Number of concurrent tests

        Returns:
            Dict with test summary
        """
        results = {
            'rules': {
                'total': len(rules),
                'passed': 0,
                'failed': 0,
                'results': []
            },
            'examples': {
                'total': len(examples),
                'passed': 0,
                'failed': 0,
                'results': []
            },
            'avg_response_time_ms': 0
        }

        total_time = 0
        total_tests = 0

        # Test rules in batches
        for i in range(0, len(rules), batch_size):
            batch = rules[i:i + batch_size]
            tasks = [self.test_rule(rule) for rule in batch]
            batch_results = await asyncio.gather(*tasks)

            for result in batch_results:
                results['rules']['results'].append(result)
                if result.success:
                    results['rules']['passed'] += 1
                else:
                    results['rules']['failed'] += 1
                total_time += result.response_time_ms
                total_tests += 1

        # Test examples in batches
        for i in range(0, len(examples), batch_size):
            batch = examples[i:i + batch_size]
            tasks = [self.test_example(ex) for ex in batch]
            batch_results = await asyncio.gather(*tasks)

            for result in batch_results:
                results['examples']['results'].append(result)
                if result.success:
                    results['examples']['passed'] += 1
                else:
                    results['examples']['failed'] += 1
                total_time += result.response_time_ms
                total_tests += 1

        if total_tests > 0:
            results['avg_response_time_ms'] = total_time / total_tests

        return results

    async def run_checkpoint_test(self, rules: List[GeneratedRule],
                                   examples: List[GeneratedExample],
                                   checkpoint_every: int = 25) -> Dict:
        """
        Run tests at checkpoints (every N rules).

        Args:
            rules: All rules generated so far
            examples: All examples generated so far
            checkpoint_every: Run test every N items

        Returns:
            Dict with checkpoint test results
        """
        total_items = len(rules) + len(examples)

        if total_items < checkpoint_every:
            return {'checkpoint_reached': False, 'total_items': total_items}

        # Get items to test (sample from recent additions)
        sample_rules = rules[-min(5, len(rules)):]
        sample_examples = examples[-min(5, len(examples)):]

        test_results = await self.test_batch(sample_rules, sample_examples)

        return {
            'checkpoint_reached': True,
            'total_items': total_items,
            'test_results': test_results
        }


class SyncRuleTester:
    """Synchronous wrapper for RuleTester."""

    def __init__(self, api_base_url: str = "http://localhost:8001"):
        self.tester = RuleTester(api_base_url)

    def test_rule(self, rule: GeneratedRule,
                  test_question: Optional[str] = None) -> TestResult:
        """Test a single rule synchronously."""
        return asyncio.run(self.tester.test_rule(rule, test_question))

    def test_example(self, example: GeneratedExample) -> TestResult:
        """Test a single example synchronously."""
        return asyncio.run(self.tester.test_example(example))

    def test_batch(self, rules: List[GeneratedRule],
                   examples: List[GeneratedExample],
                   batch_size: int = 5) -> Dict:
        """Test a batch synchronously."""
        return asyncio.run(self.tester.test_batch(rules, examples, batch_size))
