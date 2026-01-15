# SP Analysis Pipeline

## Overview

The SP Analysis Pipeline is a Prefect-based workflow that automatically generates training data for SQL query generation from stored procedures in MongoDB. It creates natural language questions paired with SQL queries to improve RAG-based text-to-SQL systems.

## Architecture

```
┌─────────────────┐
│ Fetch SP Batch  │
│  (MongoDB)      │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Generate NL     │
│ Questions (LLM) │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Create Test     │
│ SQL Queries     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Validate &      │
│ Store Training  │
│ Data (MongoDB)  │
└─────────────────┘
```

## Pipeline Components

### 1. Tasks

#### `fetch_sp_batch`
- **Purpose**: Fetch stored procedures from MongoDB in batches
- **Retries**: 2 with 30s delay
- **Output**: `SPBatchResult` with SP documents and pagination info

**Parameters**:
- `database`: Database name to query
- `batch_size`: Number of SPs per batch (default: 10)
- `offset`: Pagination offset
- `mongodb_uri`: MongoDB connection string

#### `generate_nl_questions`
- **Purpose**: Generate natural language questions using LLM
- **Retries**: 1 with 60s delay
- **Output**: `QuestionGenResult` with generated questions

**Parameters**:
- `sp`: Stored procedure document from MongoDB
- `question_count`: Number of questions to generate (default: 3)
- `llm_host`: LLM API endpoint
- `model`: Model to use (default: `qwen2.5-coder:7b`)

**Question Generation Strategy**:
- Uses SP summary, parameters, and table references as context
- Generates realistic business-focused questions
- Returns varied phrasings for different user intents
- Temperature set to 0.7 for creative variation

#### `create_test_queries`
- **Purpose**: Create EXEC statements for testing
- **Retries**: 1 with 30s delay
- **Output**: `TestQueryResult` with SQL queries

**Parameters**:
- `sp`: Stored procedure document
- `questions`: Generated questions from previous task

**Query Structure**:
```sql
EXEC schema.ProcedureName @Param1 = NULL, @Param2 = NULL
```

#### `validate_and_store`
- **Purpose**: Validate and store training examples
- **Retries**: 2 with 30s delay
- **Output**: `ValidationResult` with validation scores

**Parameters**:
- `sp`: Stored procedure document
- `questions`: Generated questions
- `queries`: Test queries
- `mongodb_uri`: MongoDB connection string

**Validation Scoring** (0-1 scale):
- **Semantic alignment** (0.4): Question aligns with SP purpose
- **Query correctness** (0.3): SQL syntax is valid
- **Parameter coverage** (0.3): Parameters are appropriately matched

**Threshold**: Examples with score >= 0.6 are stored

### 2. Flows

#### `analyze_single_sp`
Analyzes a single stored procedure by ID.

**Usage**:
```python
from prefect_pipelines.sp_analysis_flow import analyze_single_sp

result = await analyze_single_sp(
    sp_id="507f1f77bcf86cd799439011",
    database="EWRCentral",
    question_count=3
)
```

#### `analyze_sp_batch`
Batch processes multiple stored procedures.

**Usage**:
```python
from prefect_pipelines.sp_analysis_flow import analyze_sp_batch

result = await analyze_sp_batch(
    database="EWRCentral",
    batch_size=10,
    max_sps=100,
    question_count=3
)
```

**Parameters**:
- `database`: Database to process
- `batch_size`: SPs per batch (default: 10)
- `max_sps`: Maximum SPs to process (0 = all, default: 100)
- `question_count`: Questions per SP (default: 3)
- `llm_host`: LLM endpoint (default: http://localhost:11434)
- `model`: LLM model (default: qwen2.5-coder:7b)

#### `scheduled_sp_analysis`
Scheduled daily run across multiple databases.

**Usage**:
```python
from prefect_pipelines.sp_analysis_flow import scheduled_sp_analysis

result = await scheduled_sp_analysis(
    databases=["EWRCentral", "EWRReporting"],
    batch_size=10,
    max_sps_per_db=100,
    question_count=3
)
```

**Schedule**: Daily at 2 AM (configurable via CronSchedule)

### 3. Data Structures

#### SPBatchResult
```python
@dataclass
class SPBatchResult:
    database: str
    batch_number: int
    offset: int
    sps_fetched: int
    sps_total: int
    has_more: bool
    duration_seconds: float
    success: bool
    errors: List[str]
```

#### QuestionGenResult
```python
@dataclass
class QuestionGenResult:
    sp_id: str
    sp_name: str
    questions_generated: int
    questions: List[Dict[str, Any]]
    llm_calls: int
    tokens_used: int
    duration_seconds: float
    success: bool
    errors: List[str]
```

#### ValidationResult
```python
@dataclass
class ValidationResult:
    sp_id: str
    sp_name: str
    questions_validated: int
    training_examples_created: int
    validation_score: float
    stored: bool
    duration_seconds: float
    success: bool
    errors: List[str]
```

## MongoDB Collections

### Input Collection
**Collection**: `sql_stored_procedures`

**Required Fields**:
- `_id`: ObjectId
- `database`: Normalized database name
- `procedure_name`: SP name
- `schema`: Schema (default: dbo)
- `definition`: SQL definition
- `parameters`: List of parameter objects
- `summary`: LLM-generated summary
- `tables_referenced`: List of table names

### Output Collection
**Collection**: `sql_examples`

**Stored Fields**:
```javascript
{
  sp_id: ObjectId,
  sp_name: String,
  database: String,
  schema: String,
  question: String,             // Natural language question
  sql: String,                  // EXEC statement
  parameters: Object,           // Parameter values
  validation_score: Number,     // 0-1 score
  generated_at: ISODate,
  source: "sp_analysis_pipeline",
  active: Boolean
}
```

## Prefect Artifacts

The pipeline creates markdown artifacts for tracking:

### Batch Summary Artifact
**Key**: `sp-analysis-batch-summary`

**Contains**:
- Processing metrics (SPs processed, success rate)
- Generated assets (questions, training examples)
- Average validation scores
- Duration statistics

### Scheduled Summary Artifact
**Key**: `sp-analysis-scheduled-summary`

**Contains**:
- Per-database results
- Aggregate metrics across all databases
- Overall validation scores

## Error Handling

### Retry Strategy
- **fetch_sp_batch**: 2 retries, 30s delay
- **generate_nl_questions**: 1 retry, 60s delay
- **create_test_queries**: 1 retry, 30s delay
- **validate_and_store**: 2 retries, 30s delay

### Error Propagation
- Errors are logged to Prefect logger
- Failed SPs are counted but don't stop batch processing
- Error messages stored in result dataclasses
- Artifacts include error summaries

## Performance Considerations

### Resource Management
- **Batch Size**: Default 10 SPs per batch (configurable)
- **Max SPs**: Limit total processing to avoid resource exhaustion
- **Sequential Processing**: SPs processed one-at-a-time within batches
- **LLM Rate Limiting**: Built into LLM service layer

### Memory Efficiency
- MongoDB cursors with projection (only required fields)
- Batch-based pagination
- Immediate storage after validation (no large in-memory accumulation)

### Estimated Metrics
- **Time per SP**: ~15-30 seconds (3 questions)
  - LLM generation: 10-20s
  - Query creation: 1-2s
  - Validation/storage: 2-5s
- **Batch of 10 SPs**: ~3-5 minutes
- **100 SPs**: ~25-50 minutes

## Deployment

### Manual Deployment

```python
from prefect_pipelines.sp_analysis_flow import scheduled_sp_analysis
from prefect.deployments import Deployment
from prefect.server.schemas.schedules import CronSchedule

deployment = Deployment.build_from_flow(
    flow=scheduled_sp_analysis,
    name="sp-analysis-daily",
    schedule=CronSchedule(cron="0 2 * * *"),  # 2 AM daily
    work_queue_name="sp-analysis",
    parameters={
        "databases": ["EWRCentral", "EWRReporting"],
        "batch_size": 10,
        "max_sps_per_db": 100,
        "question_count": 3
    }
)

deployment.apply()
```

### CLI Deployment

```bash
prefect deployment build \
  prefect_pipelines/sp_analysis_flow.py:scheduled_sp_analysis \
  --name "sp-analysis-daily" \
  --cron "0 2 * * *" \
  --param databases='["EWRCentral", "EWRReporting"]' \
  --param batch_size=10 \
  --param max_sps_per_db=100 \
  --param question_count=3 \
  --work-queue "sp-analysis" \
  --apply
```

## Monitoring

### Prefect Dashboard
- **Flow runs**: View execution history
- **Task runs**: Individual task status and logs
- **Artifacts**: Generated markdown reports
- **Work queues**: Monitor sp-analysis queue

### Key Metrics to Monitor
- **Success rate**: Successful analyses / Total SPs processed
- **Validation scores**: Average score across training examples
- **Generation rate**: Training examples created per hour
- **Error rate**: Failed SPs / Total SPs processed

### Alerting Thresholds
- Success rate < 80%: Investigate LLM or MongoDB issues
- Avg validation score < 0.6: Review question generation prompts
- Error rate > 20%: Check system resources and connectivity

## Troubleshooting

### Common Issues

#### LLM Timeout
**Symptom**: `generate_nl_questions` fails with timeout

**Solutions**:
- Reduce `question_count` (3 → 2)
- Use faster model (`qwen2.5-coder:1.5b`)
- Increase timeout in LLM service config

#### MongoDB Connection Errors
**Symptom**: `fetch_sp_batch` fails to connect

**Solutions**:
- Verify MongoDB is running: `mongo --host EWRSPT-AI:27017`
- Check network connectivity to VM
- Validate `mongodb_uri` parameter

#### Low Validation Scores
**Symptom**: Most examples score < 0.6

**Solutions**:
- Improve SP summaries (run summarization pipeline)
- Enhance question generation prompt
- Review parameter extraction logic

#### Memory Issues
**Symptom**: Pipeline crashes with OOM

**Solutions**:
- Reduce `batch_size` (10 → 5)
- Lower `max_sps` limit
- Increase system memory allocation

## Example Outputs

### Generated Question Examples

For SP: `dbo.GetCentralTicketsByDateRange`

1. "Show me all tickets created in the last 7 days"
2. "What tickets were added between January 1st and today?"
3. "Get ticket details for tickets created this month"

### Generated SQL Examples

```sql
EXEC dbo.GetCentralTicketsByDateRange
  @StartDate = NULL,
  @EndDate = NULL

EXEC dbo.GetCentralTicketsByDateRange
  @StartDate = '2024-01-01',
  @EndDate = GETDATE()
```

### Training Data Example

```json
{
  "sp_id": "507f1f77bcf86cd799439011",
  "sp_name": "GetCentralTicketsByDateRange",
  "database": "ewrcentral",
  "schema": "dbo",
  "question": "Show me all tickets created in the last 7 days",
  "sql": "EXEC dbo.GetCentralTicketsByDateRange @StartDate = NULL, @EndDate = NULL",
  "parameters": {
    "@StartDate": null,
    "@EndDate": null
  },
  "validation_score": 0.85,
  "generated_at": "2024-12-15T06:30:00Z",
  "source": "sp_analysis_pipeline",
  "active": true
}
```

## Integration with RAG System

### How Training Data is Used

1. **Semantic Search**: Questions are embedded and indexed
2. **Query Matching**: User questions matched against training examples
3. **SQL Generation**: Matched examples guide LLM SQL generation
4. **Validation**: Generated SQL compared against validated patterns

### Continuous Improvement Loop

```
User Query → RAG Lookup → SQL Generation → Execution → Feedback
                ↑                                          ↓
                └──────────── Update Training Data ────────┘
```

## Future Enhancements

### Planned Improvements
- [ ] Entity extraction for parameter values (NER)
- [ ] Semantic similarity scoring for validation
- [ ] Multi-model ensemble for question generation
- [ ] Automatic parameter type inference
- [ ] Question quality scoring with fine-tuned classifier
- [ ] A/B testing of different generation prompts

### Advanced Features
- [ ] Cross-database SP pattern detection
- [ ] Automatic synonym expansion for questions
- [ ] SQL query execution validation (optional)
- [ ] User feedback integration loop
- [ ] Adaptive batch sizing based on system load

## References

- [Prefect Documentation](https://docs.prefect.io/)
- [MongoDB Async Motor](https://motor.readthedocs.io/)
- [Project CLAUDE.md](../.claude/CLAUDE.md)
- [SQL Rules Guide](../../docs/SQL_RULES_GUIDE.md)
