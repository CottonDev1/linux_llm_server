"""
EWR Stored Procedure Analysis Agent Models
==========================================

Pydantic v2 models for stored procedure analysis, question generation,
validation, and training data export.
"""

from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field


class QuestionDifficulty(str, Enum):
    """
    Difficulty levels for generated questions.

    Used to categorize training examples based on complexity.
    """
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"
    EXPERT = "expert"


class ValidationStatus(str, Enum):
    """
    Status of question-to-result validation.

    Indicates whether a generated question produces results that
    align with the stored procedure output.
    """
    PENDING = "pending"
    PASSED = "passed"
    FAILED = "failed"
    PARTIAL = "partial"  # Some columns match
    SKIPPED = "skipped"  # Could not be validated


class SPAnalysisConfig(BaseModel):
    """
    Configuration for the Stored Procedure Analysis Agent.

    Defines database connection settings, LLM parameters, and
    analysis behavior options.

    Attributes:
        database: Target database name containing stored procedures.
        server: Database server hostname or IP address.
        username: Optional database username for authentication.
        password: Optional database password for authentication.
        llm_model: LLM model identifier for question generation.
        llm_backend: LLM backend type (llamacpp, openai).
        llm_base_url: Base URL for the LLM API endpoint.
        llm_timeout: Timeout in seconds for LLM requests.
        questions_per_sp: Number of questions to generate per stored procedure.
        validate_questions: Whether to validate generated questions against SP results.
        include_parameters: Whether to include parameter variations in questions.
        max_concurrent: Maximum concurrent stored procedures to analyze.
        output_format: Output format for training data (json, jsonl, csv).
    """
    database: str = Field(
        default="",
        description="Target database name containing stored procedures"
    )
    server: str = Field(
        default="localhost",
        description="Database server hostname or IP address"
    )
    username: Optional[str] = Field(
        default=None,
        description="Database username for authentication"
    )
    password: Optional[str] = Field(
        default=None,
        description="Database password for authentication"
    )
    llm_model: str = Field(
        default="qwen2.5-coder:7b",
        description="LLM model identifier for question generation"
    )
    llm_backend: str = Field(
        default="llamacpp",
        description="LLM backend type (llamacpp, openai)"
    )
    llm_base_url: Optional[str] = Field(
        default=None,
        description="Base URL for the LLM API endpoint"
    )
    llm_timeout: int = Field(
        default=120,
        ge=10,
        le=600,
        description="Timeout in seconds for LLM requests"
    )
    questions_per_sp: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of questions to generate per stored procedure"
    )
    validate_questions: bool = Field(
        default=True,
        description="Whether to validate generated questions against SP results"
    )
    include_parameters: bool = Field(
        default=True,
        description="Whether to include parameter variations in questions"
    )
    max_concurrent: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Maximum concurrent stored procedures to analyze"
    )
    output_format: str = Field(
        default="json",
        description="Output format for training data (json, jsonl, csv)"
    )


class GeneratedQuestion(BaseModel):
    """
    A natural language question generated from stored procedure analysis.

    Represents a single question that, when converted to SQL, should
    produce results similar to the stored procedure output.

    Attributes:
        id: Unique identifier for the question.
        question: The natural language question text.
        sp_name: Source stored procedure name.
        sp_parameters: Parameters used when analyzing the SP.
        inferred_sql: SQL query inferred from the question (for validation).
        difficulty: Estimated difficulty level of the question.
        category: Question category (e.g., aggregation, filter, join).
        confidence: LLM's confidence score for this question (0.0-1.0).
        generated_at: Timestamp when the question was generated.
        metadata: Additional metadata about the question generation.
    """
    id: str = Field(
        default="",
        description="Unique identifier for the question"
    )
    question: str = Field(
        ...,
        min_length=10,
        description="The natural language question text"
    )
    sp_name: str = Field(
        ...,
        description="Source stored procedure name"
    )
    sp_parameters: Dict[str, Any] = Field(
        default_factory=dict,
        description="Parameters used when analyzing the SP"
    )
    inferred_sql: Optional[str] = Field(
        default=None,
        description="SQL query inferred from the question"
    )
    difficulty: QuestionDifficulty = Field(
        default=QuestionDifficulty.MEDIUM,
        description="Estimated difficulty level of the question"
    )
    category: str = Field(
        default="general",
        description="Question category (e.g., aggregation, filter, join)"
    )
    confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="LLM's confidence score for this question"
    )
    generated_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="Timestamp when the question was generated"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata about the question generation"
    )


class TestQuery(BaseModel):
    """
    A test query generated to validate a question-to-SP alignment.

    Contains the SQL query derived from the natural language question
    and execution results for comparison with the stored procedure.

    Attributes:
        question_id: Reference to the GeneratedQuestion being tested.
        sql: The SQL query generated from the natural language question.
        executed: Whether the query was successfully executed.
        execution_time_ms: Query execution time in milliseconds.
        row_count: Number of rows returned by the query.
        column_names: List of column names in the result set.
        sample_rows: Sample of rows for comparison (limited for efficiency).
        error: Error message if query execution failed.
        executed_at: Timestamp when the query was executed.
    """
    question_id: str = Field(
        ...,
        description="Reference to the GeneratedQuestion being tested"
    )
    sql: str = Field(
        ...,
        description="The SQL query generated from the natural language question"
    )
    executed: bool = Field(
        default=False,
        description="Whether the query was successfully executed"
    )
    execution_time_ms: int = Field(
        default=0,
        ge=0,
        description="Query execution time in milliseconds"
    )
    row_count: int = Field(
        default=0,
        ge=0,
        description="Number of rows returned by the query"
    )
    column_names: List[str] = Field(
        default_factory=list,
        description="List of column names in the result set"
    )
    sample_rows: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Sample of rows for comparison"
    )
    error: Optional[str] = Field(
        default=None,
        description="Error message if query execution failed"
    )
    executed_at: Optional[datetime] = Field(
        default=None,
        description="Timestamp when the query was executed"
    )


class ValidationResult(BaseModel):
    """
    Result of validating a question against stored procedure results.

    Compares the query results from the generated question's SQL
    with the actual stored procedure output to determine alignment.

    Attributes:
        question_id: Reference to the GeneratedQuestion being validated.
        test_query_id: Reference to the TestQuery used for validation.
        status: Overall validation status.
        sp_row_count: Number of rows from stored procedure execution.
        query_row_count: Number of rows from generated query execution.
        column_match_ratio: Ratio of matching columns (0.0-1.0).
        row_match_ratio: Ratio of matching rows (0.0-1.0).
        matching_columns: List of columns that match between SP and query.
        missing_columns: Columns in SP result but not in query result.
        extra_columns: Columns in query result but not in SP result.
        data_type_mismatches: Columns with different data types.
        value_discrepancies: Sample of value differences found.
        validation_notes: Notes explaining the validation result.
        validated_at: Timestamp when validation was performed.
    """
    question_id: str = Field(
        ...,
        description="Reference to the GeneratedQuestion being validated"
    )
    test_query_id: Optional[str] = Field(
        default=None,
        description="Reference to the TestQuery used for validation"
    )
    status: ValidationStatus = Field(
        default=ValidationStatus.PENDING,
        description="Overall validation status"
    )
    sp_row_count: int = Field(
        default=0,
        ge=0,
        description="Number of rows from stored procedure execution"
    )
    query_row_count: int = Field(
        default=0,
        ge=0,
        description="Number of rows from generated query execution"
    )
    column_match_ratio: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Ratio of matching columns"
    )
    row_match_ratio: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Ratio of matching rows"
    )
    matching_columns: List[str] = Field(
        default_factory=list,
        description="List of columns that match between SP and query"
    )
    missing_columns: List[str] = Field(
        default_factory=list,
        description="Columns in SP result but not in query result"
    )
    extra_columns: List[str] = Field(
        default_factory=list,
        description="Columns in query result but not in SP result"
    )
    data_type_mismatches: Dict[str, Dict[str, str]] = Field(
        default_factory=dict,
        description="Columns with different data types"
    )
    value_discrepancies: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Sample of value differences found"
    )
    validation_notes: List[str] = Field(
        default_factory=list,
        description="Notes explaining the validation result"
    )
    validated_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="Timestamp when validation was performed"
    )


class SPAnalysisResult(BaseModel):
    """
    Complete analysis result for a single stored procedure.

    Contains all generated questions, test queries, and validation
    results from analyzing one stored procedure.

    Attributes:
        sp_name: Name of the analyzed stored procedure.
        sp_schema: Schema name of the stored procedure.
        sp_definition: Full T-SQL definition of the stored procedure.
        sp_parameters: List of stored procedure parameters with types.
        tables_referenced: Tables referenced by the stored procedure.
        columns_referenced: Columns referenced by the stored procedure.
        questions: List of generated natural language questions.
        test_queries: List of test queries executed for validation.
        validations: List of validation results.
        passed_count: Number of questions that passed validation.
        failed_count: Number of questions that failed validation.
        analysis_duration_ms: Total analysis time in milliseconds.
        analyzed_at: Timestamp when analysis was performed.
        errors: List of errors encountered during analysis.
        warnings: List of warnings from the analysis process.
    """
    sp_name: str = Field(
        ...,
        description="Name of the analyzed stored procedure"
    )
    sp_schema: str = Field(
        default="dbo",
        description="Schema name of the stored procedure"
    )
    sp_definition: Optional[str] = Field(
        default=None,
        description="Full T-SQL definition of the stored procedure"
    )
    sp_parameters: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="List of stored procedure parameters with types"
    )
    tables_referenced: List[str] = Field(
        default_factory=list,
        description="Tables referenced by the stored procedure"
    )
    columns_referenced: List[str] = Field(
        default_factory=list,
        description="Columns referenced by the stored procedure"
    )
    questions: List[GeneratedQuestion] = Field(
        default_factory=list,
        description="List of generated natural language questions"
    )
    test_queries: List[TestQuery] = Field(
        default_factory=list,
        description="List of test queries executed for validation"
    )
    validations: List[ValidationResult] = Field(
        default_factory=list,
        description="List of validation results"
    )
    passed_count: int = Field(
        default=0,
        ge=0,
        description="Number of questions that passed validation"
    )
    failed_count: int = Field(
        default=0,
        ge=0,
        description="Number of questions that failed validation"
    )
    analysis_duration_ms: int = Field(
        default=0,
        ge=0,
        description="Total analysis time in milliseconds"
    )
    analyzed_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="Timestamp when analysis was performed"
    )
    errors: List[str] = Field(
        default_factory=list,
        description="List of errors encountered during analysis"
    )
    warnings: List[str] = Field(
        default_factory=list,
        description="List of warnings from the analysis process"
    )


class TrainingExample(BaseModel):
    """
    Final training data format for embedding and fine-tuning.

    Represents a validated question-SQL pair ready for use in
    training Text-to-SQL models.

    Attributes:
        id: Unique identifier for the training example.
        question: The natural language question.
        sql: The validated SQL query.
        database: Target database name.
        tables: Tables involved in the query.
        columns: Columns referenced in the query.
        difficulty: Difficulty level of the example.
        category: Query category (select, join, aggregation, etc.).
        source_sp: Name of the source stored procedure.
        validation_score: Validation confidence score (0.0-1.0).
        created_at: Timestamp when the example was created.
        metadata: Additional metadata for the example.
    """
    id: str = Field(
        default="",
        description="Unique identifier for the training example"
    )
    question: str = Field(
        ...,
        min_length=10,
        description="The natural language question"
    )
    sql: str = Field(
        ...,
        min_length=10,
        description="The validated SQL query"
    )
    database: str = Field(
        ...,
        description="Target database name"
    )
    tables: List[str] = Field(
        default_factory=list,
        description="Tables involved in the query"
    )
    columns: List[str] = Field(
        default_factory=list,
        description="Columns referenced in the query"
    )
    difficulty: QuestionDifficulty = Field(
        default=QuestionDifficulty.MEDIUM,
        description="Difficulty level of the example"
    )
    category: str = Field(
        default="general",
        description="Query category (select, join, aggregation, etc.)"
    )
    source_sp: Optional[str] = Field(
        default=None,
        description="Name of the source stored procedure"
    )
    validation_score: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Validation confidence score"
    )
    created_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="Timestamp when the example was created"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata for the example"
    )


class BatchAnalysisResult(BaseModel):
    """
    Results from batch processing multiple stored procedures.

    Aggregates analysis results across multiple stored procedures
    and provides summary statistics.

    Attributes:
        database: Database that was analyzed.
        procedures_analyzed: Number of stored procedures analyzed.
        procedures_failed: Number of procedures that failed analysis.
        total_questions_generated: Total questions generated across all SPs.
        total_questions_validated: Total questions that passed validation.
        total_questions_failed: Total questions that failed validation.
        training_examples: List of validated training examples.
        sp_results: Individual analysis results per stored procedure.
        duration_ms: Total batch processing time in milliseconds.
        started_at: Timestamp when batch processing started.
        completed_at: Timestamp when batch processing completed.
        errors: List of errors encountered during batch processing.
        summary: Summary statistics for the batch.
    """
    database: str = Field(
        ...,
        description="Database that was analyzed"
    )
    procedures_analyzed: int = Field(
        default=0,
        ge=0,
        description="Number of stored procedures analyzed"
    )
    procedures_failed: int = Field(
        default=0,
        ge=0,
        description="Number of procedures that failed analysis"
    )
    total_questions_generated: int = Field(
        default=0,
        ge=0,
        description="Total questions generated across all SPs"
    )
    total_questions_validated: int = Field(
        default=0,
        ge=0,
        description="Total questions that passed validation"
    )
    total_questions_failed: int = Field(
        default=0,
        ge=0,
        description="Total questions that failed validation"
    )
    training_examples: List[TrainingExample] = Field(
        default_factory=list,
        description="List of validated training examples"
    )
    sp_results: List[SPAnalysisResult] = Field(
        default_factory=list,
        description="Individual analysis results per stored procedure"
    )
    duration_ms: int = Field(
        default=0,
        ge=0,
        description="Total batch processing time in milliseconds"
    )
    started_at: Optional[datetime] = Field(
        default=None,
        description="Timestamp when batch processing started"
    )
    completed_at: Optional[datetime] = Field(
        default=None,
        description="Timestamp when batch processing completed"
    )
    errors: List[str] = Field(
        default_factory=list,
        description="List of errors encountered during batch processing"
    )
    summary: Dict[str, Any] = Field(
        default_factory=dict,
        description="Summary statistics for the batch"
    )
