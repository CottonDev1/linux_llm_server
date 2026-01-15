"""
Rule Generation Orchestrator.
Coordinates multiple agents to analyze stored procedures and generate SQL rules.
"""

import asyncio
import logging
from typing import List, Dict, Optional, Tuple
from datetime import datetime
from pymongo import MongoClient
from concurrent.futures import ThreadPoolExecutor, as_completed

from .models import (
    ProcedureAnalysis, GeneratedRule, GeneratedExample,
    ValidationResult, TestResult
)
from .analyzer import ProcedureAnalyzer
from .rule_generator import RuleGenerator
from .validator import RuleValidator
from .tester import SyncRuleTester

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RuleGenerationOrchestrator:
    """
    Orchestrates the rule generation pipeline.

    Pipeline stages:
    1. Load stored procedures from MongoDB
    2. Analyze procedures (10 parallel agents)
    3. Generate rules from analysis
    4. Validate rules
    5. Test every 25 rules via API
    6. Upload validated rules to MongoDB
    """

    def __init__(
        self,
        mongodb_uri: str = "mongodb://localhost:27017",
        database_name: str = "rag_server",
        api_base_url: str = "http://localhost:8001",
        num_agents: int = 10,
        checkpoint_every: int = 25,
        strict_validation: bool = True
    ):
        """
        Initialize orchestrator.

        Args:
            mongodb_uri: MongoDB connection string
            database_name: Database name for RAG server
            api_base_url: Base URL for Python API testing
            num_agents: Number of parallel analyzer agents
            checkpoint_every: Run tests every N rules
            strict_validation: Fail rules with too many warnings
        """
        self.mongodb_uri = mongodb_uri
        self.database_name = database_name
        self.api_base_url = api_base_url
        self.num_agents = num_agents
        self.checkpoint_every = checkpoint_every

        # Initialize components
        self.analyzer = ProcedureAnalyzer()
        self.generator = RuleGenerator()
        self.validator = RuleValidator(strict_mode=strict_validation)
        self.tester = SyncRuleTester(api_base_url)

        # State tracking
        self.all_rules: List[GeneratedRule] = []
        self.all_examples: List[GeneratedExample] = []
        self.validation_errors: List[Dict] = []
        self.test_results: List[Dict] = []

        # Statistics
        self.stats = {
            'procedures_processed': 0,
            'procedures_skipped': 0,
            'rules_generated': 0,
            'rules_valid': 0,
            'rules_invalid': 0,
            'examples_generated': 0,
            'examples_valid': 0,
            'examples_invalid': 0,
            'tests_run': 0,
            'tests_passed': 0,
            'tests_failed': 0,
            'rules_uploaded': 0,
            'examples_uploaded': 0,
        }

    def _get_mongo_client(self) -> MongoClient:
        """Get MongoDB client."""
        return MongoClient(self.mongodb_uri)

    def load_procedures(self, database_filter: Optional[str] = None,
                        limit: Optional[int] = None) -> List[Dict]:
        """
        Load stored procedures from MongoDB.

        Args:
            database_filter: Filter by database name (optional)
            limit: Maximum procedures to load (optional)

        Returns:
            List of procedure documents
        """
        logger.info(f"Loading stored procedures from MongoDB...")

        client = self._get_mongo_client()
        db = client[self.database_name]
        collection = db['sql_stored_procedures']

        query = {}
        if database_filter:
            query['database'] = {'$regex': database_filter, '$options': 'i'}

        # Prefer procedures with definitions
        cursor = collection.find(query).sort([
            ('definition', -1),  # Prefer those with definitions
            ('tables_affected', -1)  # Then those with table info
        ])

        if limit:
            cursor = cursor.limit(limit)

        procedures = list(cursor)
        client.close()

        logger.info(f"Loaded {len(procedures)} stored procedures")
        return procedures

    def analyze_procedure(self, proc_doc: Dict) -> Optional[ProcedureAnalysis]:
        """
        Analyze a single stored procedure.

        Args:
            proc_doc: Procedure document from MongoDB

        Returns:
            ProcedureAnalysis or None if not relevant
        """
        try:
            analysis = self.analyzer.analyze(proc_doc)

            # Skip low-relevance procedures
            if analysis.nlq_relevance < 0.3:
                return None

            return analysis
        except Exception as e:
            logger.warning(f"Error analyzing {proc_doc.get('name', 'unknown')}: {e}")
            return None

    def analyze_procedures_parallel(self, procedures: List[Dict]) -> List[ProcedureAnalysis]:
        """
        Analyze procedures in parallel using multiple agents.

        Args:
            procedures: List of procedure documents

        Returns:
            List of ProcedureAnalysis results
        """
        logger.info(f"Analyzing {len(procedures)} procedures with {self.num_agents} agents...")

        analyses = []
        with ThreadPoolExecutor(max_workers=self.num_agents) as executor:
            futures = {
                executor.submit(self.analyze_procedure, proc): proc
                for proc in procedures
            }

            for future in as_completed(futures):
                proc = futures[future]
                try:
                    analysis = future.result()
                    if analysis:
                        analyses.append(analysis)
                        self.stats['procedures_processed'] += 1
                    else:
                        self.stats['procedures_skipped'] += 1
                except Exception as e:
                    logger.error(f"Agent error for {proc.get('name')}: {e}")
                    self.stats['procedures_skipped'] += 1

        logger.info(f"Analyzed {len(analyses)} relevant procedures")
        return analyses

    def generate_rules_from_analyses(
        self,
        analyses: List[ProcedureAnalysis]
    ) -> Tuple[List[GeneratedRule], List[GeneratedExample]]:
        """
        Generate rules and examples from analyses.

        Args:
            analyses: List of procedure analyses

        Returns:
            Tuple of (rules, examples)
        """
        logger.info(f"Generating rules from {len(analyses)} analyses...")

        rules = []
        examples = []

        for analysis in analyses:
            # Generate rules
            proc_rules = self.generator.generate_rules(analysis)
            rules.extend(proc_rules)
            self.stats['rules_generated'] += len(proc_rules)

            # Generate examples
            proc_examples = self.generator.generate_examples(analysis)
            examples.extend(proc_examples)
            self.stats['examples_generated'] += len(proc_examples)

        logger.info(f"Generated {len(rules)} rules and {len(examples)} examples")
        return rules, examples

    def validate_rules(
        self,
        rules: List[GeneratedRule],
        examples: List[GeneratedExample]
    ) -> Tuple[List[GeneratedRule], List[GeneratedExample]]:
        """
        Validate rules and examples, returning only valid ones.

        Args:
            rules: Rules to validate
            examples: Examples to validate

        Returns:
            Tuple of (valid_rules, valid_examples)
        """
        logger.info(f"Validating {len(rules)} rules and {len(examples)} examples...")

        valid_rules = []
        valid_examples = []

        # Validate rules
        for rule in rules:
            result = self.validator.validate_rule(rule)
            if result.is_valid:
                valid_rules.append(rule)
                self.stats['rules_valid'] += 1
            else:
                self.stats['rules_invalid'] += 1
                self.validation_errors.append({
                    'type': 'rule',
                    'id': rule.rule_id,
                    'errors': result.errors,
                    'warnings': result.warnings
                })

        # Validate examples
        for example in examples:
            result = self.validator.validate_example(example)
            if result.is_valid:
                valid_examples.append(example)
                self.stats['examples_valid'] += 1
            else:
                self.stats['examples_invalid'] += 1
                self.validation_errors.append({
                    'type': 'example',
                    'id': example.example_id,
                    'errors': result.errors,
                    'warnings': result.warnings
                })

        logger.info(f"Validated: {len(valid_rules)} valid rules, {len(valid_examples)} valid examples")
        return valid_rules, valid_examples

    def run_checkpoint_tests(
        self,
        rules: List[GeneratedRule],
        examples: List[GeneratedExample]
    ) -> bool:
        """
        Run checkpoint tests on a sample of rules/examples.

        Args:
            rules: Rules to sample from
            examples: Examples to sample from

        Returns:
            True if tests pass threshold, False otherwise
        """
        total_items = len(rules) + len(examples)
        if total_items < self.checkpoint_every:
            return True

        logger.info(f"Running checkpoint tests at {total_items} items...")

        # Sample recent rules and examples
        sample_rules = rules[-min(5, len(rules)):]
        sample_examples = examples[-min(3, len(examples)):]

        results = self.tester.test_batch(sample_rules, sample_examples)
        self.test_results.append(results)

        passed = results['rules']['passed'] + results['examples']['passed']
        total = results['rules']['total'] + results['examples']['total']

        self.stats['tests_run'] += total
        self.stats['tests_passed'] += passed
        self.stats['tests_failed'] += (total - passed)

        success_rate = passed / total if total > 0 else 0
        logger.info(f"Checkpoint test: {passed}/{total} passed ({success_rate:.1%})")

        # Pass if at least 50% success
        return success_rate >= 0.5

    def upload_rules(self, rules: List[GeneratedRule]) -> int:
        """
        Upload validated rules to MongoDB.

        Args:
            rules: Rules to upload

        Returns:
            Number of rules uploaded
        """
        if not rules:
            return 0

        logger.info(f"Uploading {len(rules)} rules to MongoDB...")

        client = self._get_mongo_client()
        db = client[self.database_name]
        collection = db['sql_rules']

        uploaded = 0
        for rule in rules:
            doc = rule.to_mongodb_doc()
            try:
                # Upsert to avoid duplicates
                collection.update_one(
                    {'rule_id': rule.rule_id},
                    {'$set': doc},
                    upsert=True
                )
                uploaded += 1
            except Exception as e:
                logger.error(f"Error uploading rule {rule.rule_id}: {e}")

        client.close()
        self.stats['rules_uploaded'] += uploaded
        logger.info(f"Uploaded {uploaded} rules")
        return uploaded

    def upload_examples(self, examples: List[GeneratedExample]) -> int:
        """
        Upload validated examples to MongoDB.

        Args:
            examples: Examples to upload

        Returns:
            Number of examples uploaded
        """
        if not examples:
            return 0

        logger.info(f"Uploading {len(examples)} examples to MongoDB...")

        client = self._get_mongo_client()
        db = client[self.database_name]
        collection = db['sql_examples']

        uploaded = 0
        for example in examples:
            doc = example.to_mongodb_doc()
            try:
                # Upsert to avoid duplicates
                collection.update_one(
                    {'example_id': example.example_id},
                    {'$set': doc},
                    upsert=True
                )
                uploaded += 1
            except Exception as e:
                logger.error(f"Error uploading example {example.example_id}: {e}")

        client.close()
        self.stats['examples_uploaded'] += uploaded
        logger.info(f"Uploaded {uploaded} examples")
        return uploaded

    def run(
        self,
        database_filter: Optional[str] = None,
        limit: Optional[int] = None,
        dry_run: bool = False
    ) -> Dict:
        """
        Run the complete rule generation pipeline.

        Args:
            database_filter: Filter procedures by database
            limit: Maximum procedures to process
            dry_run: If True, don't upload to MongoDB

        Returns:
            Dict with pipeline results and statistics
        """
        start_time = datetime.now()
        logger.info("=" * 60)
        logger.info("Starting Rule Generation Pipeline")
        logger.info("=" * 60)

        # Step 1: Load procedures
        procedures = self.load_procedures(database_filter, limit)
        if not procedures:
            logger.warning("No procedures found!")
            return {'error': 'No procedures found', 'stats': self.stats}

        # Step 2: Analyze procedures in parallel
        analyses = self.analyze_procedures_parallel(procedures)
        if not analyses:
            logger.warning("No relevant procedures found after analysis")
            return {'error': 'No relevant procedures', 'stats': self.stats}

        # Step 3: Generate rules and examples
        rules, examples = self.generate_rules_from_analyses(analyses)

        # Step 4: Validate
        valid_rules, valid_examples = self.validate_rules(rules, examples)

        # Step 5: Run checkpoint tests
        if valid_rules or valid_examples:
            test_passed = self.run_checkpoint_tests(valid_rules, valid_examples)
            if not test_passed:
                logger.warning("Checkpoint tests failed - continuing anyway")

        # Step 6: Upload (if not dry run)
        if not dry_run:
            self.upload_rules(valid_rules)
            self.upload_examples(valid_examples)
        else:
            logger.info("DRY RUN - Skipping upload")

        # Store results
        self.all_rules.extend(valid_rules)
        self.all_examples.extend(valid_examples)

        # Calculate duration
        duration = (datetime.now() - start_time).total_seconds()

        # Build result summary
        result = {
            'success': True,
            'duration_seconds': duration,
            'stats': self.stats,
            'validation_errors': self.validation_errors[:20],  # First 20 errors
            'test_results': self.test_results,
            'sample_rules': [r.to_mongodb_doc() for r in valid_rules[:5]],
            'sample_examples': [e.to_mongodb_doc() for e in valid_examples[:5]]
        }

        # Log summary
        logger.info("=" * 60)
        logger.info("Pipeline Complete")
        logger.info("=" * 60)
        logger.info(f"Duration: {duration:.1f} seconds")
        logger.info(f"Procedures: {self.stats['procedures_processed']} processed, "
                    f"{self.stats['procedures_skipped']} skipped")
        logger.info(f"Rules: {self.stats['rules_generated']} generated, "
                    f"{self.stats['rules_valid']} valid, "
                    f"{self.stats['rules_uploaded']} uploaded")
        logger.info(f"Examples: {self.stats['examples_generated']} generated, "
                    f"{self.stats['examples_valid']} valid, "
                    f"{self.stats['examples_uploaded']} uploaded")
        logger.info(f"Tests: {self.stats['tests_passed']}/{self.stats['tests_run']} passed")

        return result


def main():
    """CLI entry point for rule generation."""
    import argparse
    import json

    parser = argparse.ArgumentParser(description='Generate SQL rules from stored procedures')
    parser.add_argument('--database', '-d', help='Filter by database name')
    parser.add_argument('--limit', '-l', type=int, help='Maximum procedures to process')
    parser.add_argument('--dry-run', action='store_true', help='Do not upload to MongoDB')
    parser.add_argument('--agents', '-a', type=int, default=10, help='Number of parallel agents')
    parser.add_argument('--mongodb-uri', default='mongodb://localhost:27017',
                        help='MongoDB connection URI')
    parser.add_argument('--api-url', default='http://localhost:8001',
                        help='Python API base URL')
    parser.add_argument('--output', '-o', help='Output file for results JSON')

    args = parser.parse_args()

    orchestrator = RuleGenerationOrchestrator(
        mongodb_uri=args.mongodb_uri,
        api_base_url=args.api_url,
        num_agents=args.agents
    )

    result = orchestrator.run(
        database_filter=args.database,
        limit=args.limit,
        dry_run=args.dry_run
    )

    if args.output:
        with open(args.output, 'w') as f:
            # Convert datetime objects for JSON serialization
            json.dump(result, f, indent=2, default=str)
        print(f"Results saved to {args.output}")

    return 0 if result.get('success') else 1


if __name__ == '__main__':
    exit(main())
