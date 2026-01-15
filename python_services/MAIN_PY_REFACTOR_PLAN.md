# main.py Refactoring Plan

## Current State
- **File size**: 5,617 lines
- **Target size**: ~200-300 lines (app setup, lifespan, router includes)
- **Pattern**: Extract route groups into separate router modules in `routes/` directory

## Refactoring Strategy

Each phase extracts a logical group of routes into a dedicated router file using FastAPI's `APIRouter`. The router is then included in main.py with a single line.

### File Structure After Refactoring
```
python_services/
├── main.py                          # ~200 lines (app setup only)
├── routes/
│   ├── __init__.py
│   ├── status_routes.py             # Phase 1
│   ├── document_routes.py           # Phase 2
│   ├── code_context_routes.py       # Phase 3
│   ├── sql_knowledge_routes.py      # Phase 4
│   ├── sql_rag_routes.py            # Phase 5
│   ├── sql_validation_routes.py     # Phase 6
│   ├── sql_extraction_routes.py     # Phase 7
│   ├── sql_summarization_routes.py  # Phase 8
│   ├── pipeline_routes.py           # Phase 9
│   ├── git_routes.py                # Phase 10
│   ├── admin_routes.py              # Phase 11
│   ├── roslyn_routes.py             # Phase 12
│   ├── feedback_routes.py           # Phase 13
│   ├── category_routes.py           # Phase 14
│   ├── audio_routes.py              # Phase 15
│   ├── audio_metrics_routes.py      # Phase 16
│   ├── bulk_audio_routes.py         # Phase 17
│   └── sql_auth_routes.py           # Phase 18
```

---

## Phase 1: Status Routes (~50 lines)
**Lines**: 357-413
**New file**: `routes/status_routes.py`

### Endpoints to extract:
- `GET /status` - System status
- `POST /status/refresh` - Refresh status
- `GET /` - Root endpoint

### Testing after Phase 1:
```bash
# Start the service
cd /mnt/c/Projects/llm_website/python_services && python main.py

# Test endpoints
curl http://localhost:8001/status
curl -X POST http://localhost:8001/status/refresh
curl http://localhost:8001/
```

---

## Phase 2: Document Routes (~430 lines)
**Lines**: 420-854
**New file**: `routes/document_routes.py`

### Endpoints to extract:
- `POST /documents` - Store document
- `POST /documents/upload` - Upload document file
- `GET /documents/search` - Search documents
- `GET /documents/total-count` - Get count
- `GET /documents/aggregate` - Aggregate query
- `GET /documents/by-title` - Get by title
- `DELETE /documents/by-title` - Delete by title
- `GET /documents/{document_id}` - Get by ID
- `GET /documents` - List documents
- `PATCH /documents/{document_id}` - Update metadata
- `DELETE /documents/{document_id}` - Delete document
- `GET /documents/stats/summary` - Get stats

### Testing after Phase 2:
```bash
curl http://localhost:8001/documents/stats/summary
curl "http://localhost:8001/documents/search?query=test&limit=5"
curl http://localhost:8001/documents/total-count
```

---

## Phase 3: Code Context Routes (~130 lines)
**Lines**: 859-981
**New file**: `routes/code_context_routes.py`

### Endpoints to extract:
- `POST /code-context` - Store code context
- `POST /code-context/bulk` - Bulk store
- `GET /code-context/search` - Search
- `GET /code-context/{document_id}` - Get by ID
- `DELETE /code-context/{document_id}` - Delete
- `GET /code-context/stats/summary` - Stats
- `DELETE /code-context/project/{project_name}` - Delete project

### Testing after Phase 3:
```bash
curl http://localhost:8001/code-context/stats/summary
curl "http://localhost:8001/code-context/search?query=test"
```

---

## Phase 4: SQL Knowledge Routes (~50 lines)
**Lines**: 987-1035
**New file**: `routes/sql_knowledge_routes.py`

### Endpoints to extract:
- `POST /sql-knowledge` - Store SQL knowledge
- `GET /sql-knowledge/search` - Search
- `GET /sql-knowledge/stats/summary` - Stats

### Testing after Phase 4:
```bash
curl http://localhost:8001/sql-knowledge/stats/summary
```

---

## Phase 5: SQL RAG Routes (~400 lines)
**Lines**: 1043-1439
**New file**: `routes/sql_rag_routes.py`

### Endpoints to extract:
- `POST /sql/examples` - Store example
- `GET /sql/examples/search` - Search examples
- `POST /sql/failed-queries` - Store failed query
- `GET /sql/failed-queries/search` - Search failed
- `POST /sql/corrections` - Store correction
- `GET /sql/corrections/search` - Search corrections
- `GET /sql/corrections/pending` - Get pending
- `POST /sql/corrections/{id}/validate` - Validate
- `POST /sql/corrections/{id}/promote` - Promote
- `GET /sql/corrections/stats` - Correction stats
- `POST /sql/schema-context` - Store schema
- `GET /sql/schema-context/search` - Search schema
- `GET /sql/schema-context/{database}/{table}` - Get table schema
- `POST /sql/stored-procedures` - Store procedure
- `GET /sql/stored-procedures/search` - Search procedures
- `GET /sql/comprehensive-context` - Get comprehensive context
- `GET /sql/rag-stats` - RAG stats
- `GET /sql/database-stats/{database}` - DB stats
- `GET /sql/pipeline-stats` - Pipeline stats

### Testing after Phase 5:
```bash
curl http://localhost:8001/sql/rag-stats
curl "http://localhost:8001/sql/examples/search?query=tickets"
```

---

## Phase 6: SQL Validation Routes (~200 lines)
**Lines**: 1445-1627
**New file**: `routes/sql_validation_routes.py`

### Endpoints to extract:
- `POST /sql/validate` - Validate SQL
- `POST /sql/validate-and-fix` - Validate and fix
- `GET /sql/validator/stats` - Validator stats
- `POST /sql/validator/refresh` - Refresh cache
- `GET /sql/validator/columns/{database}/{table}` - Get columns
- `GET /sql/validator/tables/{database}` - Get tables

### Testing after Phase 6:
```bash
curl http://localhost:8001/sql/validator/stats
curl http://localhost:8001/sql/validator/tables/EWRCentral
```

---

## Phase 7: SQL Extraction Routes (~200 lines)
**Lines**: 1633-1828
**New file**: `routes/sql_extraction_routes.py`

### Endpoints to extract:
- `POST /extract/database` - Extract database
- `POST /extract/from-config` - Extract from config
- `POST /extract/stored-procedures` - Extract procedures
- `GET /extract/status/{job_id}` - Get job status

### Testing after Phase 7:
```bash
curl http://localhost:8001/extract/status/test-job-id
```

---

## Phase 8: SQL Summarization Routes (~180 lines)
**Lines**: 1834-2032
**New file**: `routes/sql_summarization_routes.py`

### Endpoints to extract:
- `POST /summarize/stored-procedures` - Summarize procedures
- `POST /summarize/schemas` - Summarize schemas
- `GET /summarize/status` - Get status

### Testing after Phase 8:
```bash
curl http://localhost:8001/summarize/status
```

---

## Phase 9: RAG Pipeline Routes (~260 lines)
**Lines**: 2037-2290
**New file**: `routes/pipeline_routes.py`

### Endpoints to extract:
- `GET /pipeline/status/{database}` - Pipeline status
- `POST /pipeline/run` - Run pipeline
- `POST /pipeline/summarize` - Summarize
- `POST /pipeline/embed` - Embed

### Testing after Phase 9:
```bash
curl http://localhost:8001/pipeline/status/EWRCentral
```

---

## Phase 10: Git Routes (~320 lines)
**Lines**: 2324-2643
**New file**: `routes/git_routes.py`

### Endpoints to extract:
- `GET /git/repositories` - List repos
- `GET /git/repositories/{repo}/info` - Repo info
- `GET /git/repositories/{repo}/commits` - Commits
- `POST /git/pull` - Pull repos
- `POST /git/analyze-past` - Analyze past
- `POST /git/analyze-commit-impact` - Analyze impact
- `GET /git/changed-files/{repo}` - Changed files
- `GET /git/file-status/{repo}` - File status

### Testing after Phase 10:
```bash
curl http://localhost:8001/git/repositories
```

---

## Phase 11: Admin Routes (~170 lines)
**Lines**: 2648-2995
**New file**: `routes/admin_routes.py`

### Endpoints to extract:
- `GET /admin/db-stats` - DB stats
- `GET /admin/dashboard-stats` - Dashboard stats
- `GET /api/projects` - List projects
- `GET /admin/git/repositories` - Admin git repos
- `POST /admin/git/pull-all` - Pull all repos
- `POST /admin/test-mongodb` - Test MongoDB
- `POST /admin/reconnect-mongodb` - Reconnect
- `POST /admin/mongodb-reconnect` - Reconnect (alias)

### Testing after Phase 11:
```bash
curl http://localhost:8001/admin/db-stats
curl http://localhost:8001/admin/dashboard-stats
```

---

## Phase 12: Roslyn Routes (~370 lines)
**Lines**: 3004-3373
**New file**: `routes/roslyn_routes.py`

### Endpoints to extract:
- `POST /roslyn/analyze` - Analyze code
- `GET /roslyn/search/methods` - Search methods
- `GET /roslyn/search/classes` - Search classes
- `GET /roslyn/search/event-handlers` - Search handlers
- `GET /roslyn/call-chain` - Call chain
- `GET /roslyn/stats` - Stats
- `POST /roslyn/analyze-targeted` - Targeted analysis
- `POST /roslyn/analyze-full` - Full analysis
- `GET /roslyn/callgraph-status/{project}` - Callgraph status

### Testing after Phase 12:
```bash
curl http://localhost:8001/roslyn/stats
```

---

## Phase 13: Feedback Routes (~200 lines)
**Lines**: 3384-3576
**New file**: `routes/feedback_routes.py`

### Endpoints to extract:
- `POST /feedback` - Submit feedback
- `GET /feedback/stats` - Feedback stats
- `GET /feedback/low-performing` - Low performing docs

### Testing after Phase 13:
```bash
curl http://localhost:8001/feedback/stats
```

---

## Phase 14: Category Routes (~220 lines)
**Lines**: 3581-3797
**New file**: `routes/category_routes.py`

### Endpoints to extract:
- `GET /categories/{type}` - List categories
- `POST /categories/{type}` - Create category
- `PUT /categories/{type}/{id}` - Update category
- `DELETE /categories/{type}/{id}` - Delete category

### Testing after Phase 14:
```bash
curl http://localhost:8001/categories/department
```

---

## Phase 15: Audio Analysis Routes (~1140 lines)
**Lines**: 3802-4942
**New file**: `routes/audio_routes.py`

### Endpoints to extract:
- `POST /audio/upload` - Upload audio
- `POST /audio/analyze-stream` - Stream analysis
- `GET /audio/pending` - Pending analyses
- `GET /audio/pending/{filename}` - Get pending
- `DELETE /audio/pending/{filename}` - Delete pending
- `GET /audio/stream/{filename}` - Stream audio
- `POST /audio/store` - Store analysis
- `POST /audio/search` - Search analyses
- `GET /audio/{id}` - Get analysis
- `PUT /audio/{id}` - Update analysis
- `DELETE /audio/{id}` - Delete analysis
- `GET /audio/stats/summary` - Stats
- `GET /audio/stats/by-staff` - Staff stats
- `GET /audio/lookup-staff/{extension}` - Lookup staff
- `POST /audio/match-tickets` - Match tickets

### Testing after Phase 15:
```bash
curl http://localhost:8001/audio/stats/summary
curl http://localhost:8001/audio/pending
```

---

## Phase 16: Audio Staff Metrics Routes (~200 lines)
**Lines**: 4966-5149
**New file**: `routes/audio_metrics_routes.py`

### Endpoints to extract:
- `GET /audio/staff-metrics/{staff_name}` - Staff metrics

### Testing after Phase 16:
```bash
curl http://localhost:8001/audio/staff-metrics/TestUser
```

---

## Phase 17: Bulk Audio Processing Routes (~320 lines)
**Lines**: 5195-5527
**New file**: `routes/bulk_audio_routes.py`

### Endpoints to extract:
- `POST /audio/bulk/start` - Start bulk processing
- `GET /audio/bulk/status` - Bulk status
- `POST /audio/bulk/scan` - Scan for audio
- `GET /audio/bulk/history` - Processing history

### Testing after Phase 17:
```bash
curl http://localhost:8001/audio/bulk/status
curl http://localhost:8001/audio/bulk/history
```

---

## Phase 18: SQL Auth Routes (~50 lines)
**Lines**: 5540-5581
**New file**: `routes/sql_auth_routes.py`

### Endpoints to extract:
- `POST /api/sql/list-databases` - List databases

### Testing after Phase 18:
```bash
curl -X POST http://localhost:8001/api/sql/list-databases
```

---

## Final Phase: Cleanup main.py

After all phases, main.py should contain only:
1. Imports
2. App configuration
3. CORS middleware setup
4. Lifespan function (startup/shutdown)
5. Router includes
6. `if __name__ == "__main__"` block

### Final main.py structure:
```python
"""Python Data Services API"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

# Route imports
from routes.status_routes import router as status_router
from routes.document_routes import router as document_router
# ... etc

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup logic
    yield
    # Shutdown logic

app = FastAPI(lifespan=lifespan, ...)

# CORS
app.add_middleware(CORSMiddleware, ...)

# Include routers
app.include_router(status_router)
app.include_router(document_router)
# ... etc

if __name__ == "__main__":
    uvicorn.run(...)
```

---

## Router Template

Use this template for each new router file:

```python
"""
[Route Group Name] Routes

[Brief description]
"""
from fastapi import APIRouter, HTTPException, Query, Request
from typing import Optional, List

from mongodb import get_mongodb_service
# Add other imports as needed

router = APIRouter(tags=["[Tag Name]"])


@router.get("/endpoint")
async def endpoint_name():
    """Endpoint description"""
    pass
```

---

## Notes

1. **Test after EACH phase** - Don't batch multiple phases
2. **Commit after EACH phase** - Allows easy rollback
3. **Keep imports minimal** - Only import what each router needs
4. **Preserve existing behavior** - No functional changes during refactoring
5. **Update any imports** - If other files import from main.py, update them
