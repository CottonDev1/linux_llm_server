"""
SP Analysis API Routes
======================

API endpoints for stored procedure analysis and training data generation.
These endpoints support analyzing stored procedures with LLMs to generate
training questions and SQL examples for fine-tuning.

Features:
- Single SP analysis with question generation
- Batch analysis with progress tracking
- Training data export in multiple formats
- Prefect flow integration for orchestration
"""

import logging
import os
import uuid
import asyncio
from typing import Optional, Dict, Any, List
from datetime import datetime
from fastapi import APIRouter, HTTPException, Query, BackgroundTasks
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/sp-analysis", tags=["SP Analysis"])


# =============================================================================
# Request/Response Models
# =============================================================================

class BatchAnalysisRequest(BaseModel):
    """Request for batch analysis of stored procedures."""
    database: str = Field(..., description="Database name to analyze")
    server: str = Field(default="CHAD-PC", description="SQL Server hostname")
    batch_size: int = Field(default=10, description="Number of SPs to analyze per batch", ge=1, le=100)
    max_sps: int = Field(default=100, description="Maximum number of SPs to analyze", ge=1, le=1000)
    question_count: int = Field(default=3, description="Number of questions to generate per SP", ge=1, le=10)
    schema_filter: Optional[str] = Field(None, description="Filter to specific schema (e.g., 'dbo')")
    sp_name_pattern: Optional[str] = Field(None, description="Filter SPs by name pattern (SQL LIKE syntax)")
    model: str = Field(default="qwen2.5-coder:1.5b", description="LLM model to use for analysis")


class AnalysisResponse(BaseModel):
    """Response from SP analysis."""
    success: bool
    sp_id: Optional[str] = None
    sp_name: Optional[str] = None
    database: str
    questions_generated: int = 0
    processing_time_ms: int = 0
    error: Optional[str] = None
    training_examples: List[Dict[str, Any]] = Field(default_factory=list)


class BatchAnalysisResponse(BaseModel):
    """Response from batch analysis request."""
    success: bool
    job_id: str
    status: str
    message: str
    total_sps: int = 0
    estimated_duration_minutes: float = 0.0


class JobStatusResponse(BaseModel):
    """Job status response."""
    job_id: str
    status: str  # pending, running, completed, failed
    progress: float = 0.0  # 0.0 to 1.0
    stage: str = "queued"
    total_sps: int = 0
    processed_sps: int = 0
    failed_sps: int = 0
    total_questions: int = 0
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error: Optional[str] = None
    results: Optional[Dict[str, Any]] = None


class TrainingDataListResponse(BaseModel):
    """Response for listing training data."""
    success: bool
    total_count: int
    returned_count: int
    limit: int
    offset: int
    data: List[Dict[str, Any]]


class TrainingDataResponse(BaseModel):
    """Response for single training example."""
    success: bool
    id: str
    database: str
    sp_name: str
    question: str
    sql: str
    created_at: datetime
    metadata: Dict[str, Any]


class ExportRequest(BaseModel):
    """Request for exporting training data."""
    format: str = Field(default="jsonl", description="Export format: jsonl, csv, parquet")
    database: Optional[str] = Field(None, description="Filter by database")
    start_date: Optional[datetime] = Field(None, description="Filter by creation date (start)")
    end_date: Optional[datetime] = Field(None, description="Filter by creation date (end)")
    min_quality_score: Optional[float] = Field(None, description="Filter by quality score", ge=0.0, le=1.0)


class ExportResponse(BaseModel):
    """Response from export request."""
    success: bool
    format: str
    file_path: str
    record_count: int
    file_size_bytes: int
    message: str


class FlowTriggerRequest(BaseModel):
    """Request to trigger a Prefect flow."""
    flow_name: str = Field(..., description="Name of the flow to trigger")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Flow parameters")
    wait_for_completion: bool = Field(default=False, description="Wait for flow to complete")


class FlowTriggerResponse(BaseModel):
    """Response from flow trigger."""
    success: bool
    flow_run_id: str
    flow_name: str
    status: str
    message: str
    flow_url: Optional[str] = None


class FlowListResponse(BaseModel):
    """Response for listing available flows."""
    success: bool
    flows: List[Dict[str, Any]]


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    sp_analysis_available: bool
    mongodb_connected: bool
    llm_available: bool
    prefect_available: bool
    details: Dict[str, Any] = Field(default_factory=dict)


class StatsResponse(BaseModel):
    """Statistics response."""
    total_sps_analyzed: int
    total_questions_generated: int
    databases_analyzed: List[str]
    jobs_completed: int
    jobs_failed: int
    avg_questions_per_sp: float
    last_analysis: Optional[datetime] = None


# =============================================================================
# Global State
# =============================================================================

# Job tracking (in-memory for now, could move to Redis/MongoDB)
_jobs: Dict[str, Dict[str, Any]] = {}
_jobs_lock = asyncio.Lock()

# Service instances (lazy initialized)
_mongodb_service = None
_llm_service = None


# =============================================================================
# Helper Functions
# =============================================================================

async def get_mongodb_service():
    """Get or create MongoDB service singleton."""
    global _mongodb_service
    if _mongodb_service is None:
        try:
            from mongodb import get_mongodb_service as get_mongo
            _mongodb_service = await get_mongo()
        except Exception as e:
            logger.error(f"Failed to initialize MongoDB service: {e}")
            raise HTTPException(status_code=503, detail="MongoDB service unavailable")
    return _mongodb_service


async def get_llm_service():
    """Get or create LLM service singleton."""
    global _llm_service
    if _llm_service is None:
        try:
            from services.llm_service import LLMService
            _llm_service = LLMService()
        except Exception as e:
            logger.error(f"Failed to initialize LLM service: {e}")
            raise HTTPException(status_code=503, detail="LLM service unavailable")
    return _llm_service


async def create_job(job_data: Dict[str, Any]) -> str:
    """Create a new job for tracking."""
    job_id = str(uuid.uuid4())
    async with _jobs_lock:
        _jobs[job_id] = {
            "job_id": job_id,
            "status": "pending",
            "stage": "queued",
            "progress": 0.0,
            "total_sps": job_data.get("total_sps", 0),
            "processed_sps": 0,
            "failed_sps": 0,
            "total_questions": 0,
            "started_at": None,
            "completed_at": None,
            "error": None,
            "results": None,
            **job_data
        }
    logger.info(f"Created job {job_id}")
    return job_id


async def update_job(job_id: str, **kwargs):
    """Update job status."""
    async with _jobs_lock:
        if job_id in _jobs:
            _jobs[job_id].update(kwargs)
            # Update progress based on processed/total if available
            if "processed_sps" in kwargs or "total_sps" in kwargs:
                total = _jobs[job_id].get("total_sps", 0)
                processed = _jobs[job_id].get("processed_sps", 0)
                if total > 0:
                    _jobs[job_id]["progress"] = processed / total
            logger.debug(f"Updated job {job_id}: {kwargs}")


async def get_job(job_id: str) -> Optional[Dict[str, Any]]:
    """Get job status."""
    async with _jobs_lock:
        return _jobs.get(job_id)


# =============================================================================
# Health and Status Endpoints
# =============================================================================

@router.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check for SP analysis service.

    Checks:
    - MongoDB connection
    - LLM service availability
    - Prefect availability (optional)
    """
    details = {}
    mongodb_ok = False
    llm_ok = False
    prefect_ok = False

    # Check MongoDB
    try:
        mongo = await get_mongodb_service()
        mongodb_ok = mongo is not None
        details["mongodb"] = "connected"
    except Exception as e:
        details["mongodb"] = f"error: {str(e)}"

    # Check LLM
    try:
        llm = await get_llm_service()
        llm_ok = llm is not None
        details["llm"] = "available"
    except Exception as e:
        details["llm"] = f"error: {str(e)}"

    # Check Prefect (optional)
    try:
        import httpx
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get("http://localhost:4200/api/health")
            prefect_ok = response.status_code == 200
            details["prefect"] = "available"
    except Exception as e:
        details["prefect"] = f"unavailable: {str(e)}"
        prefect_ok = False  # Prefect is optional

    overall_status = "healthy" if (mongodb_ok and llm_ok) else "degraded"

    return HealthResponse(
        status=overall_status,
        sp_analysis_available=mongodb_ok and llm_ok,
        mongodb_connected=mongodb_ok,
        llm_available=llm_ok,
        prefect_available=prefect_ok,
        details=details
    )


@router.get("/stats", response_model=StatsResponse)
async def get_stats():
    """
    Get SP analysis statistics.

    Returns aggregate statistics about analyzed stored procedures
    and generated training data.
    """
    try:
        mongo = await get_mongodb_service()

        # Get collection
        training_data = mongo.db["sp_training_data"]

        # Count documents
        total_count = await training_data.count_documents({})

        # Get distinct databases
        databases = await training_data.distinct("database")

        # Count jobs
        async with _jobs_lock:
            completed_jobs = sum(1 for j in _jobs.values() if j["status"] == "completed")
            failed_jobs = sum(1 for j in _jobs.values() if j["status"] == "failed")

        # Get last analysis timestamp
        last_doc = await training_data.find_one(
            sort=[("created_at", -1)],
            projection={"created_at": 1}
        )
        last_analysis = last_doc["created_at"] if last_doc else None

        # Calculate distinct SPs analyzed
        distinct_sps = await training_data.distinct("sp_name")
        total_sps = len(distinct_sps)

        # Calculate average questions per SP
        avg_questions = total_count / total_sps if total_sps > 0 else 0.0

        return StatsResponse(
            total_sps_analyzed=total_sps,
            total_questions_generated=total_count,
            databases_analyzed=databases,
            jobs_completed=completed_jobs,
            jobs_failed=failed_jobs,
            avg_questions_per_sp=avg_questions,
            last_analysis=last_analysis
        )

    except Exception as e:
        logger.error(f"Failed to get stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# Analysis Endpoints
# =============================================================================

@router.post("/analyze/{sp_id}", response_model=AnalysisResponse)
async def analyze_single(
    sp_id: str,
    question_count: int = Query(default=3, ge=1, le=10, description="Number of questions to generate"),
    model: str = Query(default="qwen2.5-coder:1.5b", description="LLM model to use")
):
    """
    Analyze a single stored procedure by ID.

    Generates training questions and SQL examples for the specified
    stored procedure. The SP must already be indexed in MongoDB.

    Args:
        sp_id: MongoDB document ID of the stored procedure
        question_count: Number of questions to generate (1-10)
        model: LLM model to use for analysis

    Returns:
        AnalysisResponse with generated training examples
    """
    start_time = datetime.utcnow()

    try:
        mongo = await get_mongodb_service()
        llm = await get_llm_service()

        # Get SP from MongoDB
        from bson import ObjectId
        sp_doc = await mongo.db["stored_procedures"].find_one({"_id": ObjectId(sp_id)})

        if not sp_doc:
            raise HTTPException(status_code=404, detail=f"Stored procedure {sp_id} not found")

        # TODO: Integrate with actual SP analysis service
        # For now, placeholder response
        logger.info(f"Analyzing SP: {sp_doc.get('name')} from {sp_doc.get('database')}")

        # Simulate analysis (replace with actual implementation)
        training_examples = []
        for i in range(question_count):
            training_examples.append({
                "question": f"Example question {i+1} for {sp_doc.get('name')}",
                "sql": f"EXEC {sp_doc.get('name')} /* params */",
                "metadata": {
                    "sp_id": sp_id,
                    "sp_name": sp_doc.get("name"),
                    "database": sp_doc.get("database"),
                    "model": model
                }
            })

        # Store in MongoDB
        if training_examples:
            await mongo.db["sp_training_data"].insert_many([
                {
                    **example,
                    "created_at": datetime.utcnow(),
                    "quality_score": 0.8  # Placeholder
                }
                for example in training_examples
            ])

        end_time = datetime.utcnow()
        processing_time = int((end_time - start_time).total_seconds() * 1000)

        return AnalysisResponse(
            success=True,
            sp_id=sp_id,
            sp_name=sp_doc.get("name"),
            database=sp_doc.get("database"),
            questions_generated=len(training_examples),
            processing_time_ms=processing_time,
            training_examples=training_examples
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Analysis failed for {sp_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/analyze-batch", response_model=BatchAnalysisResponse)
async def analyze_batch(
    request: BatchAnalysisRequest,
    background_tasks: BackgroundTasks
):
    """
    Start batch analysis of stored procedures.

    Analyzes multiple stored procedures from a database in batches,
    generating training questions for each. Processing happens in
    the background and can be monitored via the job status endpoint.

    Args:
        request: Batch analysis configuration
        background_tasks: FastAPI background tasks

    Returns:
        BatchAnalysisResponse with job_id for tracking
    """
    try:
        # Estimate duration (rough: 2 seconds per SP per question)
        estimated_duration = (request.max_sps * request.question_count * 2) / 60.0

        # Create job
        job_id = await create_job({
            "database": request.database,
            "server": request.server,
            "batch_size": request.batch_size,
            "total_sps": request.max_sps,
            "question_count": request.question_count,
            "model": request.model,
            "schema_filter": request.schema_filter,
            "sp_name_pattern": request.sp_name_pattern
        })

        # Queue background task
        background_tasks.add_task(
            _process_batch_analysis,
            job_id,
            request
        )

        return BatchAnalysisResponse(
            success=True,
            job_id=job_id,
            status="pending",
            message="Batch analysis queued for processing",
            total_sps=request.max_sps,
            estimated_duration_minutes=estimated_duration
        )

    except Exception as e:
        logger.error(f"Failed to start batch analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def _process_batch_analysis(job_id: str, request: BatchAnalysisRequest):
    """Background task for batch analysis."""
    try:
        await update_job(job_id, status="running", stage="initializing", started_at=datetime.utcnow())

        mongo = await get_mongodb_service()

        # Build query filter
        query_filter = {"database": request.database}
        if request.schema_filter:
            query_filter["schema"] = request.schema_filter
        if request.sp_name_pattern:
            query_filter["name"] = {"$regex": request.sp_name_pattern, "$options": "i"}

        # Get SPs to analyze
        await update_job(job_id, stage="fetching_procedures")
        sps = await mongo.db["stored_procedures"].find(query_filter).limit(request.max_sps).to_list(None)

        actual_count = len(sps)
        await update_job(job_id, total_sps=actual_count, stage="analyzing")

        logger.info(f"Job {job_id}: Found {actual_count} SPs to analyze")

        # Process in batches
        processed = 0
        failed = 0
        total_questions = 0

        for i in range(0, len(sps), request.batch_size):
            batch = sps[i:i + request.batch_size]

            for sp_doc in batch:
                try:
                    # TODO: Call actual SP analysis service
                    # For now, simulate
                    await asyncio.sleep(0.1)  # Simulate processing

                    processed += 1
                    total_questions += request.question_count

                    await update_job(
                        job_id,
                        processed_sps=processed,
                        total_questions=total_questions,
                        stage=f"analyzing ({processed}/{actual_count})"
                    )

                except Exception as e:
                    logger.error(f"Failed to analyze SP {sp_doc.get('name')}: {e}")
                    failed += 1
                    await update_job(job_id, failed_sps=failed)

        # Mark complete
        await update_job(
            job_id,
            status="completed",
            stage="completed",
            progress=1.0,
            completed_at=datetime.utcnow(),
            results={
                "processed": processed,
                "failed": failed,
                "total_questions": total_questions
            }
        )

        logger.info(f"Job {job_id} completed: {processed} SPs, {total_questions} questions")

    except Exception as e:
        logger.error(f"Job {job_id} failed: {e}")
        await update_job(
            job_id,
            status="failed",
            stage="error",
            error=str(e),
            completed_at=datetime.utcnow()
        )


@router.get("/jobs/{job_id}", response_model=JobStatusResponse)
async def get_job_status(job_id: str):
    """
    Get status of an analysis job.

    Returns current status, progress, and results (if completed)
    for a batch analysis job.

    Args:
        job_id: Job ID from batch analysis request

    Returns:
        JobStatusResponse with current status
    """
    job = await get_job(job_id)

    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    return JobStatusResponse(**job)


# =============================================================================
# Training Data Endpoints
# =============================================================================

@router.get("/training-data", response_model=TrainingDataListResponse)
async def list_training_data(
    database: Optional[str] = Query(None, description="Filter by database"),
    limit: int = Query(100, ge=1, le=1000, description="Max results to return"),
    offset: int = Query(0, ge=0, description="Offset for pagination")
):
    """
    List generated training data.

    Returns paginated list of training examples generated from
    stored procedure analysis.

    Args:
        database: Optional database filter
        limit: Maximum results to return (1-1000)
        offset: Pagination offset

    Returns:
        TrainingDataListResponse with training examples
    """
    try:
        mongo = await get_mongodb_service()

        # Build query
        query = {}
        if database:
            query["database"] = database

        # Get total count
        total_count = await mongo.db["sp_training_data"].count_documents(query)

        # Get paginated results
        cursor = mongo.db["sp_training_data"].find(query).skip(offset).limit(limit)
        results = await cursor.to_list(None)

        # Convert ObjectId to string
        for doc in results:
            doc["id"] = str(doc.pop("_id"))

        return TrainingDataListResponse(
            success=True,
            total_count=total_count,
            returned_count=len(results),
            limit=limit,
            offset=offset,
            data=results
        )

    except Exception as e:
        logger.error(f"Failed to list training data: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/training-data/{id}", response_model=TrainingDataResponse)
async def get_training_example(id: str):
    """
    Get a specific training example.

    Args:
        id: Training example ID

    Returns:
        TrainingDataResponse with example details
    """
    try:
        mongo = await get_mongodb_service()
        from bson import ObjectId

        doc = await mongo.db["sp_training_data"].find_one({"_id": ObjectId(id)})

        if not doc:
            raise HTTPException(status_code=404, detail=f"Training example {id} not found")

        return TrainingDataResponse(
            success=True,
            id=str(doc["_id"]),
            database=doc.get("database", ""),
            sp_name=doc.get("metadata", {}).get("sp_name", ""),
            question=doc.get("question", ""),
            sql=doc.get("sql", ""),
            created_at=doc.get("created_at", datetime.utcnow()),
            metadata=doc.get("metadata", {})
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get training example {id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/training-data/export", response_model=ExportResponse)
async def export_training_data(request: ExportRequest):
    """
    Export training data for fine-tuning.

    Exports training examples in the specified format for use
    in LLM fine-tuning pipelines.

    Supported formats:
    - jsonl: JSON Lines format (one example per line)
    - csv: Comma-separated values
    - parquet: Apache Parquet format

    Args:
        request: Export configuration

    Returns:
        ExportResponse with file path and metadata
    """
    try:
        mongo = await get_mongodb_service()

        # Build query
        query = {}
        if request.database:
            query["database"] = request.database
        if request.start_date:
            query.setdefault("created_at", {})["$gte"] = request.start_date
        if request.end_date:
            query.setdefault("created_at", {})["$lte"] = request.end_date
        if request.min_quality_score:
            query["quality_score"] = {"$gte": request.min_quality_score}

        # Get data
        cursor = mongo.db["sp_training_data"].find(query)
        results = await cursor.to_list(None)

        # Generate export file
        export_dir = os.path.join(os.path.dirname(__file__), "..", "exports")
        os.makedirs(export_dir, exist_ok=True)

        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        filename = f"sp_training_data_{timestamp}.{request.format}"
        file_path = os.path.join(export_dir, filename)

        # Export based on format
        if request.format == "jsonl":
            import json
            with open(file_path, "w", encoding="utf-8") as f:
                for doc in results:
                    doc["_id"] = str(doc["_id"])
                    f.write(json.dumps(doc) + "\n")

        elif request.format == "csv":
            import csv
            with open(file_path, "w", newline="", encoding="utf-8") as f:
                if results:
                    fieldnames = ["question", "sql", "database", "sp_name", "created_at"]
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    for doc in results:
                        writer.writerow({
                            "question": doc.get("question", ""),
                            "sql": doc.get("sql", ""),
                            "database": doc.get("database", ""),
                            "sp_name": doc.get("metadata", {}).get("sp_name", ""),
                            "created_at": doc.get("created_at", "")
                        })

        elif request.format == "parquet":
            import pandas as pd
            df = pd.DataFrame(results)
            df.to_parquet(file_path, index=False)

        else:
            raise HTTPException(status_code=400, detail=f"Unsupported format: {request.format}")

        file_size = os.path.getsize(file_path)

        return ExportResponse(
            success=True,
            format=request.format,
            file_path=file_path,
            record_count=len(results),
            file_size_bytes=file_size,
            message=f"Exported {len(results)} training examples to {filename}"
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Export failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/training-data/{id}")
async def delete_training_example(id: str):
    """
    Delete a training example.

    Args:
        id: Training example ID to delete

    Returns:
        Success message
    """
    try:
        mongo = await get_mongodb_service()
        from bson import ObjectId

        result = await mongo.db["sp_training_data"].delete_one({"_id": ObjectId(id)})

        if result.deleted_count == 0:
            raise HTTPException(status_code=404, detail=f"Training example {id} not found")

        return {
            "success": True,
            "id": id,
            "message": "Training example deleted"
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Delete failed for {id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# Prefect Integration Endpoints
# =============================================================================

@router.post("/run-flow", response_model=FlowTriggerResponse)
async def trigger_flow(request: FlowTriggerRequest):
    """
    Trigger a Prefect flow.

    Triggers a Prefect flow for orchestrated SP analysis workflows.

    Args:
        request: Flow trigger configuration

    Returns:
        FlowTriggerResponse with flow run details
    """
    try:
        import httpx

        # Trigger flow via Prefect API
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"http://localhost:4200/api/deployments/{request.flow_name}/create_flow_run",
                json={
                    "parameters": request.parameters,
                    "state": {
                        "type": "SCHEDULED",
                        "name": "Scheduled"
                    }
                }
            )

            if response.status_code != 201:
                raise HTTPException(
                    status_code=response.status_code,
                    detail=f"Failed to trigger flow: {response.text}"
                )

            data = response.json()
            flow_run_id = data.get("id", "")

            return FlowTriggerResponse(
                success=True,
                flow_run_id=flow_run_id,
                flow_name=request.flow_name,
                status="scheduled",
                message="Flow triggered successfully",
                flow_url=f"http://localhost:4200/flow-runs/flow-run/{flow_run_id}"
            )

    except httpx.RequestError as e:
        logger.error(f"Prefect connection error: {e}")
        raise HTTPException(
            status_code=503,
            detail="Prefect server unavailable. Make sure Prefect is running on port 4200"
        )
    except Exception as e:
        logger.error(f"Failed to trigger flow: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/flows", response_model=FlowListResponse)
async def list_flows():
    """
    List available Prefect flows.

    Returns list of available SP analysis flows that can be triggered.

    Returns:
        FlowListResponse with available flows
    """
    try:
        import httpx

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get("http://localhost:4200/api/deployments")

            if response.status_code != 200:
                raise HTTPException(
                    status_code=response.status_code,
                    detail=f"Failed to list flows: {response.text}"
                )

            data = response.json()

            # Filter to SP analysis flows
            sp_flows = [
                {
                    "name": flow.get("name", ""),
                    "description": flow.get("description", ""),
                    "id": flow.get("id", ""),
                    "tags": flow.get("tags", [])
                }
                for flow in data
                if "sp-analysis" in flow.get("tags", []) or "sp_analysis" in flow.get("name", "").lower()
            ]

            return FlowListResponse(
                success=True,
                flows=sp_flows
            )

    except httpx.RequestError as e:
        logger.error(f"Prefect connection error: {e}")
        # Return empty list instead of error (Prefect is optional)
        return FlowListResponse(
            success=False,
            flows=[]
        )
    except Exception as e:
        logger.error(f"Failed to list flows: {e}")
        raise HTTPException(status_code=500, detail=str(e))
