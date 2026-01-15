# EWR Stored Procedure Analysis Agent

A specialized AI agent for analyzing stored procedures and generating NL->SQL training data. Part of the EWR Agent Framework.

## Overview

The EWR SP Analysis Agent automates the generation of Text-to-SQL training data by:

1. **Extracting** stored procedures from databases
2. **Analyzing** SP definitions to understand their purpose and output
3. **Generating** natural language questions that would produce similar results
4. **Validating** questions by comparing query results with SP output
5. **Exporting** validated examples as training data

This approach leverages existing database knowledge encoded in stored procedures to create high-quality, domain-specific training examples.

## Key Features

- **Automated Question Generation**: Uses LLMs to generate natural language questions from SP analysis
- **Result Validation**: Validates generated questions by comparing query results with SP output
- **Batch Processing**: Analyze multiple stored procedures concurrently
- **Training Export**: Export validated examples in various formats (JSON, JSONL, CSV)
- **Prefect Integration**: Orchestrate analysis workflows with Prefect
- **MongoDB Storage**: Store analysis results in MongoDB for querying and retrieval

## Installation

### Basic Installation

```bash
pip install ewr-sp-analysis-agent
```

### Development Installation

```bash
pip install ewr-sp-analysis-agent[dev]
```

## Dependencies

### Core Dependencies

- **ewr-agent-core** (>=1.0.0): Base agent framework with LLM integration
- **motor** (>=3.0.0): Async MongoDB driver for result storage
- **aiohttp** (>=3.8.0): Async HTTP client for LLM API calls
- **prefect** (>=2.14.0): Workflow orchestration

### System Requirements

- Python 3.10 or higher
- Running LLM backend (llama.cpp, Ollama, or OpenAI)
- MongoDB instance (optional, for result storage)
- SQL Server with stored procedures to analyze

## Usage

### Python API

#### Basic Setup

```python
from ewr_sp_analysis_agent import SPAnalysisAgent, SPAnalysisConfig

# Create configuration
config = SPAnalysisConfig(
    database="EWRCentral",
    server="localhost",
    llm_model="qwen2.5-coder:7b",
    questions_per_sp=5,
    validate_questions=True
)

# Initialize agent
agent = SPAnalysisAgent(config=config)
await agent.start()
```

#### Analyze a Single Stored Procedure

```python
# Analyze one stored procedure
result = await agent.analyze_procedure("usp_GetTicketsByDate")

print(f"SP: {result.sp_name}")
print(f"Questions generated: {len(result.questions)}")
print(f"Passed validation: {result.passed_count}")
print(f"Failed validation: {result.failed_count}")

# View generated questions
for question in result.questions:
    print(f"\nQ: {question.question}")
    print(f"Difficulty: {question.difficulty}")
    print(f"Confidence: {question.confidence}")
```

#### Batch Analysis

```python
# Analyze multiple stored procedures
batch_result = await agent.batch_analyze(
    procedures=["usp_GetTickets", "usp_GetUsers", "usp_GetOrders"],
    max_concurrent=3
)

print(f"Procedures analyzed: {batch_result.procedures_analyzed}")
print(f"Total questions: {batch_result.total_questions_generated}")
print(f"Validated: {batch_result.total_questions_validated}")

# Export training examples
training_data = batch_result.training_examples
```

#### Export Training Data

```python
import json

# Export to JSON
training_json = {
    "database": batch_result.database,
    "examples": [
        {
            "id": ex.id,
            "question": ex.question,
            "sql": ex.sql,
            "tables": ex.tables,
            "difficulty": ex.difficulty.value,
            "category": ex.category
        }
        for ex in batch_result.training_examples
    ]
}

with open("training_data.json", "w") as f:
    json.dump(training_json, f, indent=2)
```

### CLI Usage

```bash
# Analyze a single stored procedure
ewr-sp-analysis analyze usp_GetTicketsByDate --database=EWRCentral

# Batch analyze with pattern matching
ewr-sp-analysis batch "usp_Get*" --database=EWRCentral --output=training.json

# List stored procedures
ewr-sp-analysis list --database=EWRCentral

# Export training data
ewr-sp-analysis export --database=EWRCentral --format=jsonl --output=training.jsonl
```

## Models

### SPAnalysisConfig

Configuration for the agent including database connection, LLM settings, and analysis parameters.

```python
config = SPAnalysisConfig(
    database="EWRCentral",
    server="localhost",
    username="sa",
    password="password",
    llm_model="qwen2.5-coder:7b",
    llm_backend="llamacpp",
    llm_timeout=120,
    questions_per_sp=5,
    validate_questions=True,
    include_parameters=True,
    max_concurrent=3,
    output_format="json"
)
```

### GeneratedQuestion

A natural language question generated from SP analysis.

```python
question = GeneratedQuestion(
    id="q-001",
    question="How many tickets were created today?",
    sp_name="usp_GetDailyTicketCount",
    difficulty=QuestionDifficulty.EASY,
    category="aggregation",
    confidence=0.95
)
```

### ValidationResult

Result of validating a question against SP results.

```python
validation = ValidationResult(
    question_id="q-001",
    status=ValidationStatus.PASSED,
    column_match_ratio=1.0,
    row_match_ratio=0.98,
    matching_columns=["Count", "Date"]
)
```

### TrainingExample

Final training data format for embedding.

```python
example = TrainingExample(
    id="ex-001",
    question="How many tickets were created today?",
    sql="SELECT COUNT(*) AS Count FROM CentralTickets WHERE CAST(AddTicketDate AS DATE) = CAST(GETDATE() AS DATE)",
    database="EWRCentral",
    tables=["CentralTickets"],
    difficulty=QuestionDifficulty.EASY,
    category="aggregation",
    source_sp="usp_GetDailyTicketCount",
    validation_score=0.98
)
```

## Workflow

### Analysis Pipeline

1. **Extract SP Definition**: Retrieve stored procedure T-SQL from database
2. **Parse SP Structure**: Identify tables, columns, parameters, and output shape
3. **Generate Questions**: Use LLM to create natural language questions
4. **Convert Questions to SQL**: Generate SQL from each question
5. **Execute and Compare**: Run both SP and generated SQL, compare results
6. **Validate Alignment**: Score how well query results match SP output
7. **Export Valid Examples**: Save validated question-SQL pairs

### Validation Criteria

Questions pass validation when:

- Column names match (exact or semantic match)
- Row counts are within acceptable threshold
- Data types are compatible
- Sample values align

## Configuration

### Environment Variables

```bash
# Database connection
export SP_AGENT_SERVER="localhost"
export SP_AGENT_DATABASE="EWRCentral"
export SP_AGENT_USER="sa"
export SP_AGENT_PASSWORD="password"

# LLM configuration
export LLM_BACKEND="llamacpp"
export LLM_MODEL="qwen2.5-coder:7b"
export LLM_BASE_URL="http://localhost:8080"

# MongoDB (optional)
export MONGODB_URI="mongodb://localhost:27017"
export MONGODB_DATABASE="sp_analysis"
```

### Recommended LLM Models

For question generation:

- **qwen2.5-coder:7b**: Good balance of speed and quality
- **sqlcoder:7b**: Specialized for SQL understanding
- **deepseek-coder-v2:16b**: Higher quality, slower

## Integration with EWR Agent Framework

This agent integrates with the broader EWR Agent Framework:

- Uses `ewr-agent-core` for base functionality
- Shares models with `ewr-sql-agent` for SQL operations
- Can be orchestrated with Prefect alongside other agents
- Exports training data compatible with embedding pipelines

## License

MIT License - see LICENSE file for details.

## Version

Current version: 1.0.0
