"""
EWR SP Analysis Agent
=====================

Analyzes stored procedures to generate high-quality training data for text-to-SQL systems.

Core Workflow:
1. Generate diverse NL questions from SP name/definition using LLM
2. Create test SQL queries to validate each question
3. Execute validation queries and verify results
4. Use LLM-as-Judge to score question/result alignment
5. Store validated training data in MongoDB with embeddings

Features:
- Multi-prompt diversity for question generation
- Chain-of-thought test query generation
- LLM-as-Judge validation scoring
- Token bucket rate limiting
- Batch processing with concurrency control
- MongoDB integration with embeddings
"""

import asyncio
import time
import re
import hashlib
import uuid
from typing import List, Dict, Any, Optional
from datetime import datetime
import aiohttp

from ewr_agent_core import (
    BaseAgent,
    AgentType,
    AgentCapability,
    TaskResult,
    TaskStatus,
    AgentConfig,
)

from .models import (
    SPAnalysisResult,
    GeneratedQuestion,
    TestQuery,
    ValidationResult,
    QuestionDifficulty,
    ValidationStatus,
    TrainingExample,
    BatchAnalysisResult,
)


# ============================================================================
# Stored Procedure Class
# ============================================================================

class StoredProcedure:
    """Simple dataclass for stored procedure info."""
    def __init__(self, name: str, database: str, schema: str = "dbo",
                 definition: str = "", parameters: List[Dict[str, Any]] = None):
        self.name = name
        self.database = database
        self.schema = schema
        self.definition = definition
        self.parameters = parameters or []


# ============================================================================
# Token Bucket Rate Limiter
# ============================================================================

class TokenBucket:
    """Token bucket for rate limiting LLM API calls."""

    def __init__(self, capacity: int = 10, refill_rate: float = 1.0):
        """
        Args:
            capacity: Maximum tokens in bucket
            refill_rate: Tokens added per second
        """
        self.capacity = capacity
        self.refill_rate = refill_rate
        self.tokens = float(capacity)
        self.last_refill = time.time()
        self._lock = asyncio.Lock()

    async def acquire(self, tokens: int = 1) -> None:
        """Acquire tokens, waiting if necessary."""
        async with self._lock:
            while self.tokens < tokens:
                await self._refill()
                if self.tokens < tokens:
                    wait_time = (tokens - self.tokens) / self.refill_rate
                    await asyncio.sleep(wait_time)

            self.tokens -= tokens

    async def _refill(self) -> None:
        """Refill tokens based on elapsed time."""
        now = time.time()
        elapsed = now - self.last_refill
        new_tokens = elapsed * self.refill_rate
        self.tokens = min(self.capacity, self.tokens + new_tokens)
        self.last_refill = now


# ============================================================================
# SP Analysis Agent
# ============================================================================

class SPAnalysisAgent(BaseAgent):
    """
    Stored Procedure Analysis Agent.

    Generates training data from stored procedures using:
    - QuestionGenerator: Multi-prompt LLM question generation
    - TestQueryGenerator: Chain-of-thought test SQL creation
    - ValidationExecutor: Query execution and validation
    - ResultEmbedder: Embedding generation for search
    """

    def __init__(
        self,
        config: AgentConfig = None,
        mongodb_uri: str = "mongodb://localhost:27017",
        mongodb_database: str = "rag_db",
        llm_url: str = "http://localhost:11434",
        llm_model: str = "qwen2.5-coder:7b",
        rate_limit_capacity: int = 10,
        rate_limit_per_second: float = 2.0,
        **kwargs
    ):
        super().__init__(config, **kwargs)

        # MongoDB configuration
        self.mongodb_uri = mongodb_uri
        self.mongodb_database = mongodb_database
        self.collection_name = "sp_analysis_training_data"
        self._mongo_client = None
        self._mongo_db = None
        self._mongo_collection = None

        # LLM configuration
        self.llm_url = llm_url.rstrip('/')
        self.llm_model = llm_model

        # Rate limiting
        self.rate_limiter = TokenBucket(
            capacity=rate_limit_capacity,
            refill_rate=rate_limit_per_second
        )

        # HTTP session
        self._http_session: Optional[aiohttp.ClientSession] = None

    @property
    def agent_type(self) -> AgentType:
        return AgentType.CUSTOM

    @property
    def capabilities(self) -> List[AgentCapability]:
        return [
            AgentCapability.TRAINING_DATA,
            AgentCapability.SQL_GENERATE,
            AgentCapability.SQL_VALIDATE,
        ]

    async def _initialize(self) -> None:
        """Initialize MongoDB connection and HTTP session."""
        # Import motor here to avoid import errors
        from motor.motor_asyncio import AsyncIOMotorClient

        # Connect to MongoDB
        self.logger.info(f"Connecting to MongoDB: {self.mongodb_uri}")
        self._mongo_client = AsyncIOMotorClient(self.mongodb_uri)
        self._mongo_db = self._mongo_client[self.mongodb_database]
        self._mongo_collection = self._mongo_db[self.collection_name]

        # Create indexes
        await self._create_indexes()

        # Create HTTP session
        self._http_session = aiohttp.ClientSession()

        self.logger.info("SP Analysis Agent initialized")

    async def stop(self) -> None:
        """Clean up resources."""
        if self._http_session:
            await self._http_session.close()

        if self._mongo_client:
            self._mongo_client.close()

        await super().stop()

    async def _create_indexes(self) -> None:
        """Create MongoDB indexes for efficient queries."""
        try:
            await self._mongo_collection.create_index("sp_name")
            await self._mongo_collection.create_index("database")
            await self._mongo_collection.create_index("created_at")
            await self._mongo_collection.create_index("validation_score")
            await self._mongo_collection.create_index(
                [("database", 1), ("sp_name", 1)],
                unique=False
            )
            self.logger.info("MongoDB indexes created")
        except Exception as e:
            self.logger.warning(f"Failed to create indexes: {e}")

    async def handle_task(self, task: Dict[str, Any]) -> TaskResult:
        """Handle incoming analysis tasks."""
        task_id = task.get("task_id", "unknown")
        task_type = task.get("task_type", "")
        params = task.get("params", {})

        try:
            if task_type == "analyze_sp":
                sp_data = params.get("stored_procedure")
                if not sp_data:
                    raise ValueError("Missing stored_procedure in params")

                sp = StoredProcedure(**sp_data)
                result = await self.analyze_stored_procedure(sp)

                return TaskResult(
                    task_id=task_id,
                    status=TaskStatus.COMPLETED,
                    result=result.model_dump()
                )

            elif task_type == "batch_analyze":
                procedures_data = params.get("procedures", [])
                batch_size = params.get("batch_size", 10)

                procedures = [StoredProcedure(**p) for p in procedures_data]
                result = await self.batch_analyze(procedures, batch_size)

                return TaskResult(
                    task_id=task_id,
                    status=TaskStatus.COMPLETED,
                    result=result.model_dump()
                )

            else:
                return TaskResult(
                    task_id=task_id,
                    status=TaskStatus.FAILED,
                    error=f"Unknown task type: {task_type}"
                )

        except Exception as e:
            self.logger.error(f"Task failed: {e}", exc_info=True)
            return TaskResult(
                task_id=task_id,
                status=TaskStatus.FAILED,
                error=str(e)
            )

    # ========================================================================
    # Core Analysis Methods
    # ========================================================================

    async def analyze_stored_procedure(
        self,
        sp: StoredProcedure,
        question_count: int = 3
    ) -> SPAnalysisResult:
        """
        Analyze a single stored procedure and generate training data.

        Args:
            sp: StoredProcedure to analyze
            question_count: Number of questions to generate

        Returns:
            SPAnalysisResult with generated questions and validation
        """
        self.logger.info(f"Analyzing {sp.name}")
        start_time = time.time()

        result = SPAnalysisResult(
            sp_name=sp.name,
            sp_schema=sp.schema,
            sp_definition=sp.definition,
            sp_parameters=sp.parameters
        )

        try:
            # Generate questions
            questions = await self.generate_questions(sp, count=question_count)
            result.questions = questions

            # Validate each question
            for question in questions:
                try:
                    # Generate test query
                    test_query = await self.generate_test_query(sp, question)
                    result.test_queries.append(test_query)

                    # Execute and validate
                    validation = await self.validate_question(question, test_query)
                    result.validations.append(validation)

                    # Update counts
                    if validation.status == ValidationStatus.PASSED:
                        result.passed_count += 1
                        # Store as training example
                        await self._store_training_example(sp, question, test_query, validation)
                    else:
                        result.failed_count += 1

                except Exception as e:
                    self.logger.error(f"Failed to validate question: {e}")
                    result.errors.append(f"Question validation failed: {str(e)}")
                    result.failed_count += 1

        except Exception as e:
            result.errors.append(f"Analysis failed: {str(e)}")
            self.logger.error(f"Analysis failed for {sp.name}: {e}")

        result.analysis_duration_ms = int((time.time() - start_time) * 1000)
        return result

    async def generate_questions(
        self,
        sp: StoredProcedure,
        count: int = 3
    ) -> List[GeneratedQuestion]:
        """
        Generate NL questions from a stored procedure using multi-prompt diversity.

        Uses multiple prompts with different perspectives to generate diverse questions.

        Args:
            sp: StoredProcedure to analyze
            count: Number of questions to generate

        Returns:
            List of GeneratedQuestion objects
        """
        questions = []

        # Use multiple prompt strategies for diversity
        prompts = self._get_question_generation_prompts(sp, count)

        for i, prompt in enumerate(prompts[:count]):
            await self.rate_limiter.acquire()

            try:
                response = await self._call_llm(prompt, temperature=0.8)
                question_text = self._extract_question(response)

                if question_text:
                    question_id = str(uuid.uuid4())
                    questions.append(GeneratedQuestion(
                        id=question_id,
                        question=question_text,
                        sp_name=sp.name,
                        sp_parameters={},  # Could extract specific parameter values
                        category=self._classify_question(question_text),
                        difficulty=self._estimate_difficulty(question_text, sp),
                        confidence=0.8  # Default confidence
                    ))

            except Exception as e:
                self.logger.error(f"Failed to generate question {i}: {e}")

        return questions

    async def generate_test_query(
        self,
        sp: StoredProcedure,
        question: GeneratedQuestion
    ) -> TestQuery:
        """
        Generate a test SQL query to validate the question using chain-of-thought.

        Args:
            sp: StoredProcedure context
            question: GeneratedQuestion to validate

        Returns:
            TestQuery with SQL and expected behavior
        """
        await self.rate_limiter.acquire()

        prompt = self._get_test_query_prompt(sp, question)
        response = await self._call_llm(prompt, temperature=0.3)

        # Parse response
        sql = self._extract_sql(response)

        return TestQuery(
            question_id=question.id,
            sql=sql,
            executed=False,
            execution_time_ms=0,
            row_count=0
        )

    async def validate_question(
        self,
        question: GeneratedQuestion,
        test_query: TestQuery
    ) -> ValidationResult:
        """
        Use LLM-as-Judge to validate question-result alignment.

        Args:
            question: GeneratedQuestion to validate
            test_query: TestQuery with validation SQL

        Returns:
            ValidationResult with quality score
        """
        await self.rate_limiter.acquire()

        # Build validation prompt
        prompt = self._get_validation_prompt(question, test_query)
        response = await self._call_llm(prompt, temperature=0.2)

        # Parse LLM judge response
        status = self._parse_validation_status(response)
        score = self._parse_numeric_score(response)
        notes = self._extract_validation_notes(response)

        return ValidationResult(
            question_id=question.id,
            test_query_id=str(uuid.uuid4()),
            status=status,
            sp_row_count=0,  # Would need actual execution
            query_row_count=0,
            column_match_ratio=score,
            row_match_ratio=score,
            validation_notes=[notes] if notes else []
        )

    async def batch_analyze(
        self,
        procedures: List[StoredProcedure],
        batch_size: int = 10,
        concurrency: int = 3
    ) -> BatchAnalysisResult:
        """
        Process multiple stored procedures with rate limiting.

        Args:
            procedures: List of StoredProcedure objects
            batch_size: Questions per procedure
            concurrency: Max concurrent analyses

        Returns:
            BatchAnalysisResult with summary statistics
        """
        start_time = time.time()

        result = BatchAnalysisResult(
            database=procedures[0].database if procedures else "",
            started_at=datetime.utcnow()
        )

        semaphore = asyncio.Semaphore(concurrency)

        async def analyze_with_limit(sp: StoredProcedure):
            async with semaphore:
                return await self.analyze_stored_procedure(sp, question_count=batch_size)

        # Process all procedures
        tasks = [analyze_with_limit(sp) for sp in procedures]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Aggregate results
        for sp_result in results:
            if isinstance(sp_result, Exception):
                result.procedures_failed += 1
                result.errors.append(str(sp_result))
            elif isinstance(sp_result, SPAnalysisResult):
                result.procedures_analyzed += 1
                result.sp_results.append(sp_result)
                result.total_questions_generated += len(sp_result.questions)
                result.total_questions_validated += sp_result.passed_count
                result.total_questions_failed += sp_result.failed_count

        result.duration_ms = int((time.time() - start_time) * 1000)
        result.completed_at = datetime.utcnow()

        # Build summary
        result.summary = {
            "success_rate": f"{(result.procedures_analyzed / len(procedures) * 100):.1f}%",
            "avg_questions_per_sp": result.total_questions_generated / max(result.procedures_analyzed, 1),
            "validation_rate": f"{(result.total_questions_validated / max(result.total_questions_generated, 1) * 100):.1f}%"
        }

        return result

    # ========================================================================
    # LLM Prompts
    # ========================================================================

    def _get_question_generation_prompts(
        self,
        sp: StoredProcedure,
        count: int
    ) -> List[str]:
        """
        Generate multiple diverse prompts for question generation.

        Multi-prompt strategy ensures diversity in generated questions.
        """
        base_info = f"""
Stored Procedure: {sp.name}
Database: {sp.database}
Schema: {sp.schema}

Definition:
{sp.definition[:2000]}

Parameters:
{self._format_parameters(sp.parameters)}
"""

        # Prompt 1: Business user perspective
        prompt1 = f"""You are a business analyst working with a database system.

{base_info}

Generate a natural language question that a business user might ask that would be answered by calling this stored procedure.

The question should be:
- Clear and specific
- In plain English (no SQL terminology)
- Focused on business value
- Answerable by this procedure

Output only the question, nothing else."""

        # Prompt 2: Technical perspective
        prompt2 = f"""You are a database developer analyzing stored procedures.

{base_info}

Generate a technical question that would require executing this stored procedure to answer.

The question should:
- Reference specific data operations
- Be precise and unambiguous
- Match the procedure's purpose
- Use domain terminology

Output only the question, nothing else."""

        # Prompt 3: Reporting perspective
        prompt3 = f"""You are a data analyst creating reports.

{base_info}

Generate a reporting question that this stored procedure would help answer.

The question should:
- Focus on data retrieval or analysis
- Be specific about what data is needed
- Match the procedure's output
- Be suitable for a dashboard or report

Output only the question, nothing else."""

        prompts = [prompt1, prompt2, prompt3]

        # Add more generic prompts if needed
        while len(prompts) < count:
            generic = f"""Analyze this stored procedure and generate a natural language question it could answer:

{base_info}

Output only the question, nothing else."""
            prompts.append(generic)

        return prompts

    def _get_test_query_prompt(
        self,
        sp: StoredProcedure,
        question: GeneratedQuestion
    ) -> str:
        """Generate chain-of-thought prompt for test query generation."""
        return f"""Generate a SQL query to validate that the following question can be answered by the stored procedure.

Stored Procedure: {sp.name}

Question: {question.question}

Use chain-of-thought reasoning:

1. ANALYZE: What data does the question ask for?
2. PROCEDURE: What does {sp.name} return?
3. ALIGNMENT: Do they match?
4. SQL: Write a query that would validate this (e.g., EXEC {sp.name} with test parameters)

Output format:
ANALYSIS: [your analysis]
PROCEDURE: [what it does]
ALIGNMENT: [yes/no with explanation]
SQL:
```sql
[your validation query]
```"""

    def _get_validation_prompt(
        self,
        question: GeneratedQuestion,
        test_query: TestQuery
    ) -> str:
        """Generate LLM-as-Judge prompt for validation."""
        return f"""You are an expert judge evaluating the quality of text-to-SQL training examples.

Question: {question.question}

Validation Query:
{test_query.sql}

Rate the alignment between the question and query on this scale:
- PASSED: Perfect match, clear, unambiguous
- PARTIAL: Strong match, minor improvements possible
- FAILED: Misaligned or unclear

Provide your rating in this format:
STATUS: [PASSED/PARTIAL/FAILED]
SCORE: [0.0-1.0]
NOTES: [brief explanation]"""

    # ========================================================================
    # LLM Communication
    # ========================================================================

    async def _call_llm(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 1000
    ) -> str:
        """
        Call llama.cpp API for LLM generation.

        Args:
            prompt: Input prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate

        Returns:
            Generated text
        """
        url = f"{self.llm_url}/completion"

        payload = {
            "prompt": prompt,
            "temperature": temperature,
            "n_predict": max_tokens,
            "stop": ["</s>", "USER:", "SYSTEM:"]
        }

        try:
            async with self._http_session.post(url, json=payload, timeout=60) as resp:
                resp.raise_for_status()
                data = await resp.json()
                return data.get("content", "").strip()

        except Exception as e:
            self.logger.error(f"LLM API call failed: {e}")
            raise

    # ========================================================================
    # Response Parsing
    # ========================================================================

    def _extract_question(self, response: str) -> str:
        """Extract question from LLM response."""
        # Remove common prefixes
        response = re.sub(r'^(Question:|Q:)\s*', '', response, flags=re.IGNORECASE)

        # Take first line if multiline
        lines = response.strip().split('\n')
        question = lines[0].strip()

        # Ensure it ends with question mark
        if question and not question.endswith('?'):
            question += '?'

        return question

    def _extract_sql(self, response: str) -> str:
        """Extract SQL from response."""
        # Try code block first
        sql_match = re.search(r'```sql\n(.*?)\n```', response, re.DOTALL | re.IGNORECASE)
        if sql_match:
            return sql_match.group(1).strip()

        # Try generic code block
        code_match = re.search(r'```\n?(.*?)\n?```', response, re.DOTALL)
        if code_match:
            return code_match.group(1).strip()

        # Look for SQL: prefix
        sql_line = re.search(r'SQL:\s*(.+)', response, re.DOTALL | re.IGNORECASE)
        if sql_line:
            return sql_line.group(1).strip()

        return ""

    def _parse_validation_status(self, response: str) -> ValidationStatus:
        """Parse validation status from judge response."""
        status_match = re.search(r'STATUS:\s*(\w+)', response, re.IGNORECASE)
        if status_match:
            status_str = status_match.group(1).upper()
            if status_str in ValidationStatus.__members__:
                return ValidationStatus[status_str]

        return ValidationStatus.PENDING

    def _parse_numeric_score(self, response: str) -> float:
        """Parse numeric score from judge response."""
        score_match = re.search(r'SCORE:\s*([\d.]+)', response)
        if score_match:
            return float(score_match.group(1))

        # Default based on status
        return 0.7

    def _extract_validation_notes(self, response: str) -> str:
        """Extract reasoning from judge response."""
        notes_match = re.search(r'NOTES:\s*(.+)', response, re.DOTALL)
        if notes_match:
            return notes_match.group(1).strip()

        return ""

    def _classify_question(self, question_text: str) -> str:
        """Classify question type based on content."""
        question_lower = question_text.lower()

        if any(word in question_lower for word in ['how many', 'count', 'total', 'number of']):
            return "aggregation"
        elif any(word in question_lower for word in ['list', 'show', 'get', 'find']):
            return "retrieval"
        elif any(word in question_lower for word in ['update', 'insert', 'delete', 'modify']):
            return "modification"
        elif any(word in question_lower for word in ['when', 'date', 'time']):
            return "temporal"
        else:
            return "general"

    def _estimate_difficulty(self, question_text: str, sp: StoredProcedure) -> QuestionDifficulty:
        """Estimate question difficulty."""
        # Simple heuristic based on question length and SP complexity
        param_count = len(sp.parameters)
        question_length = len(question_text.split())

        if param_count > 5 or question_length > 20:
            return QuestionDifficulty.HARD
        elif param_count > 2 or question_length > 12:
            return QuestionDifficulty.MEDIUM
        else:
            return QuestionDifficulty.EASY

    # ========================================================================
    # Storage
    # ========================================================================

    async def _store_training_example(
        self,
        sp: StoredProcedure,
        question: GeneratedQuestion,
        test_query: TestQuery,
        validation: ValidationResult
    ) -> None:
        """Store validated training example in MongoDB."""
        try:
            # Generate unique ID
            content_hash = hashlib.sha256(
                f"{sp.name}:{question.question}".encode()
            ).hexdigest()[:16]

            example = TrainingExample(
                id=content_hash,
                question=question.question,
                sql=test_query.sql,
                database=sp.database,
                tables=[],  # Would need to parse from SQL
                columns=[],  # Would need to parse from SQL
                difficulty=question.difficulty,
                category=question.category,
                source_sp=sp.name,
                validation_score=validation.column_match_ratio
            )

            # Insert or update
            await self._mongo_collection.update_one(
                {"_id": content_hash},
                {"$set": example.model_dump()},
                upsert=True
            )

            self.logger.info(f"Stored training example: {content_hash}")

        except Exception as e:
            self.logger.error(f"Failed to store training example: {e}")

    # ========================================================================
    # Utilities
    # ========================================================================

    def _format_parameters(self, parameters: List[Dict[str, Any]]) -> str:
        """Format parameter list for prompts."""
        if not parameters:
            return "None"

        lines = []
        for param in parameters:
            direction = "OUTPUT" if param.get("is_output") else "INPUT"
            lines.append(f"  - {param['name']} ({param['type']}) [{direction}]")

        return "\n".join(lines)
