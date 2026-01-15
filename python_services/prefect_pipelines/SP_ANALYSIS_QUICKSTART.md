# SP Analysis Pipeline - Quick Start Guide

## What is it?

The SP Analysis Pipeline automatically generates training data for your SQL query generation system by:

1. **Fetching** stored procedures from MongoDB
2. **Generating** natural language questions using LLM
3. **Creating** test SQL queries (EXEC statements)
4. **Validating** and storing high-quality training examples

## 5-Minute Quick Start

### 1. Prerequisites Check

```bash
# Verify MongoDB is accessible
mongo --host EWRSPT-AI:27017 --eval "db.adminCommand('ping')"

# Verify LLM service is running
curl http://localhost:11434/api/version

# Verify stored procedures exist in MongoDB
mongo EWRSPT-AI:27017/rag_data --eval "db.sql_stored_procedures.countDocuments({})"
```

### 2. Run Your First Analysis

```python
import asyncio
from prefect_pipelines.sp_analysis_flow import analyze_sp_batch

async def quick_test():
    result = await analyze_sp_batch(
        database="EWRCentral",
        batch_size=5,
        max_sps=10,
        question_count=3
    )
    print(f"Success: {result['success']}")
    print(f"Training examples created: {result['total_training_examples']}")

asyncio.run(quick_test())
```

### 3. View Results

```python
# Check what was created
from mongodb_service import MongoDBService
import asyncio

async def view_results():
    mongodb = MongoDBService()
    await mongodb.connect()

    # Get a sample training example
    example = await mongodb.db["sql_examples"].find_one(
        {"database": "ewrcentral"},
        sort=[("generated_at", -1)]
    )

    print(f"Question: {example['question']}")
    print(f"SQL: {example['sql']}")
    print(f"Validation Score: {example['validation_score']}")

asyncio.run(view_results())
```

## Common Use Cases

### Use Case 1: Generate Training Data for New Database

```python
# Process all SPs in a database
await analyze_sp_batch(
    database="MyNewDatabase",
    batch_size=10,
    max_sps=0,  # 0 = process all
    question_count=3
)
```

### Use Case 2: Update Training Data for Specific SP

```python
# Regenerate questions for one SP
await analyze_single_sp(
    sp_id="507f1f77bcf86cd799439011",  # Replace with actual ID
    database="EWRCentral",
    question_count=5  # Generate more questions
)
```

### Use Case 3: Scheduled Daily Updates

```python
# Set up daily processing
from prefect.deployments import Deployment
from prefect.server.schemas.schedules import CronSchedule
from prefect_pipelines.sp_analysis_flow import scheduled_sp_analysis

deployment = Deployment.build_from_flow(
    flow=scheduled_sp_analysis,
    name="daily-sp-analysis",
    schedule=CronSchedule(cron="0 2 * * *"),  # 2 AM daily
    parameters={
        "databases": ["EWRCentral", "EWRReporting"],
        "max_sps_per_db": 100
    }
)
deployment.apply()
```

## Understanding the Output

### Training Example Structure

Each successful analysis creates training examples in MongoDB:

```javascript
{
  sp_id: ObjectId("..."),
  sp_name: "GetTicketsByDate",
  database: "ewrcentral",
  question: "Show me all tickets created today",  // Natural language
  sql: "EXEC dbo.GetTicketsByDate @Date = NULL",  // SQL query
  validation_score: 0.85,                          // Quality score (0-1)
  generated_at: ISODate("2024-12-15T06:00:00Z"),
  source: "sp_analysis_pipeline",
  active: true
}
```

### Validation Scores

- **0.8-1.0**: Excellent quality, high confidence
- **0.6-0.8**: Good quality, usable
- **< 0.6**: Not stored (below threshold)

### What Makes a High-Quality Example?

1. **Semantic alignment** (40%): Question clearly maps to SP purpose
2. **SQL correctness** (30%): Valid EXEC syntax with SP name
3. **Parameter coverage** (30%): All parameters accounted for

## Performance Expectations

### Processing Speed (approximate)

| Task | Time per SP | Notes |
|------|-------------|-------|
| Question Generation | 10-20s | LLM call (varies by model) |
| Query Creation | 1-2s | Fast, rule-based |
| Validation & Storage | 2-5s | MongoDB operations |
| **Total** | **15-30s** | Per SP (3 questions) |

### Recommended Batch Sizes

| Scenario | Batch Size | Max SPs | Duration |
|----------|------------|---------|----------|
| Quick Test | 5 | 10 | ~2-5 min |
| Small Database | 10 | 100 | ~25-50 min |
| Large Database | 20 | 1000 | ~4-8 hours |
| Production Daily | 10 | 100/db | ~1-2 hours |

## Troubleshooting

### "No questions generated"

**Cause**: LLM timeout or poor SP summary

**Fix**:
```python
# Use smaller, faster model
await analyze_sp_batch(
    database="EWRCentral",
    model="qwen2.5-coder:1.5b",  # Faster
    question_count=2              # Fewer questions
)
```

### "Low validation scores"

**Cause**: SP summaries missing or poor quality

**Fix**:
```bash
# Run SP summarization first
cd python_services
python sql_extraction/procedure_summarizer.py
```

### "MongoDB connection failed"

**Cause**: MongoDB not running or network issue

**Fix**:
```bash
# Check MongoDB status
ssh chad@EWRSPT-AI
systemctl status mongodb

# Test connection
mongo --host EWRSPT-AI:27017 --eval "db.runCommand({ping:1})"
```

## Monitoring

### Check Progress in Prefect Dashboard

```bash
# Start Prefect server (if not running)
prefect server start

# Visit dashboard
# http://localhost:4200
```

### Query Training Data Statistics

```python
from mongodb_service import MongoDBService

async def get_stats():
    mongodb = MongoDBService()
    await mongodb.connect()
    collection = mongodb.db["sql_examples"]

    total = await collection.count_documents({"active": True})
    high_quality = await collection.count_documents({
        "validation_score": {"$gte": 0.8},
        "active": True
    })

    print(f"Total examples: {total}")
    print(f"High quality (>=0.8): {high_quality}")
    print(f"Quality rate: {high_quality/total*100:.1f}%")
```

## Best Practices

### 1. Start Small
- Test with `max_sps=10` first
- Verify quality before scaling up
- Check validation scores

### 2. Use Appropriate Models
- **Fast testing**: `qwen2.5-coder:1.5b`
- **Production**: `qwen2.5-coder:7b`
- **High quality**: `sqlcoder:7b` or `deepseek-coder-v2:16b`

### 3. Monitor Quality
- Target avg validation score > 0.7
- Review low-scoring examples
- Improve SP summaries if needed

### 4. Schedule Intelligently
- Run during off-peak hours (2-4 AM)
- Limit max_sps to manage load
- Process different databases on different days if needed

## Integration with RAG System

The training examples are automatically used by your SQL generation system:

1. **User asks**: "Show me tickets from last week"
2. **RAG searches**: `sql_examples` collection for similar questions
3. **Finds**: "Show me all tickets created in the last 7 days"
4. **Uses**: Matching SQL as template for generation
5. **Generates**: Appropriate query with correct parameters

## Next Steps

1. **Read full documentation**: `SP_ANALYSIS_README.md`
2. **Try examples**: `examples/sp_analysis_example.py`
3. **Set up scheduling**: Deploy for automated runs
4. **Monitor quality**: Review validation scores and adjust
5. **Iterate**: Improve prompts based on results

## Getting Help

- Full documentation: `SP_ANALYSIS_README.md`
- Usage examples: `examples/sp_analysis_example.py`
- Prefect docs: https://docs.prefect.io/
- Project CLAUDE.md: `../.claude/CLAUDE.md`

## Files Reference

```
prefect_pipelines/
├── sp_analysis_flow.py          # Main pipeline (this is what runs)
├── SP_ANALYSIS_README.md        # Full documentation
├── SP_ANALYSIS_QUICKSTART.md    # This file
└── examples/
    └── sp_analysis_example.py   # Runnable examples
```

---

**Ready to start?** Copy the "Run Your First Analysis" code above and execute it!
