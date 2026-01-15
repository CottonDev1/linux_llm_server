"""
Agent API Routes
FastAPI endpoints for the background task agent.
"""

from typing import Optional, Dict, Any
from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel

from agent import get_agent_service, TaskExecutor
from agent.models import (
    TaskCreate, TaskType, TaskPriority, TaskStatus,
    AgentStatus, TaskResult
)

router = APIRouter(prefix="/agent", tags=["Agent"])


# Request/Response models for API

class ShellCommandRequest(BaseModel):
    """Request to execute a shell command"""
    command: str
    working_dir: Optional[str] = None
    timeout: int = 300
    async_exec: bool = False


class PowerShellCommandRequest(BaseModel):
    """Request to execute a PowerShell command"""
    command: str
    working_dir: Optional[str] = None
    timeout: int = 300
    async_exec: bool = False


class GitSyncRequest(BaseModel):
    """Request to sync git repositories"""
    repository: Optional[str] = None  # None = sync all
    async_exec: bool = True


class SQLChainRequest(BaseModel):
    """Request for chain-of-thought SQL generation"""
    question: str
    database: str
    context: Optional[Dict[str, Any]] = None
    async_exec: bool = False


class TaskResponse(BaseModel):
    """Response for task submission"""
    task_id: str
    status: str
    message: str


# Endpoints

@router.get("/status", response_model=AgentStatus)
async def get_agent_status():
    """Get agent status and statistics"""
    agent = await get_agent_service()
    return agent.get_status()


@router.get("/health")
async def health_check():
    """Run health checks on all services"""
    agent = await get_agent_service()
    task = TaskCreate(type=TaskType.HEALTH_CHECK)
    result = await agent.execute_task(task)
    return result.result if result.result else {"status": "error", "error": result.error}


@router.post("/shell", response_model=Dict[str, Any])
async def execute_shell(request: ShellCommandRequest):
    """
    Execute a shell command.

    If async_exec=True, returns immediately with task_id.
    If async_exec=False, waits for completion and returns result.
    """
    agent = await get_agent_service()

    task = TaskCreate(
        type=TaskType.SHELL,
        command=request.command,
        params={"working_dir": request.working_dir} if request.working_dir else None,
        timeout=request.timeout
    )

    if request.async_exec:
        submitted = agent.submit_task(task)
        return {
            "task_id": submitted.task_id,
            "status": "queued",
            "message": "Task submitted to queue"
        }
    else:
        result = await agent.execute_task(task)
        return {
            "task_id": result.task_id,
            "status": result.status.value,
            "exit_code": result.exit_code,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "duration_ms": result.duration_ms,
            "error": result.error
        }


@router.post("/powershell", response_model=Dict[str, Any])
async def execute_powershell(request: PowerShellCommandRequest):
    """
    Execute a PowerShell command.

    If async_exec=True, returns immediately with task_id.
    If async_exec=False, waits for completion and returns result.
    """
    agent = await get_agent_service()

    task = TaskCreate(
        type=TaskType.POWERSHELL,
        command=request.command,
        params={"working_dir": request.working_dir} if request.working_dir else None,
        timeout=request.timeout
    )

    if request.async_exec:
        submitted = agent.submit_task(task)
        return {
            "task_id": submitted.task_id,
            "status": "queued",
            "message": "Task submitted to queue"
        }
    else:
        result = await agent.execute_task(task)
        return {
            "task_id": result.task_id,
            "status": result.status.value,
            "exit_code": result.exit_code,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "duration_ms": result.duration_ms,
            "error": result.error
        }


@router.post("/git/sync", response_model=Dict[str, Any])
async def git_sync(request: GitSyncRequest):
    """
    Sync git repositories.

    If repository is specified, sync only that repository.
    If repository is None, sync all configured repositories.
    """
    agent = await get_agent_service()

    if request.repository:
        task = TaskCreate(
            type=TaskType.GIT_PULL,
            params={"repository": request.repository}
        )
    else:
        task = TaskCreate(type=TaskType.GIT_SYNC)

    if request.async_exec:
        submitted = agent.submit_task(task)
        return {
            "task_id": submitted.task_id,
            "status": "queued",
            "message": "Git sync task submitted"
        }
    else:
        result = await agent.execute_task(task)
        return {
            "task_id": result.task_id,
            "status": result.status.value,
            "result": result.result,
            "duration_ms": result.duration_ms,
            "error": result.error
        }


@router.post("/sql/chain", response_model=Dict[str, Any])
async def sql_chain_of_thought(request: SQLChainRequest):
    """
    Generate SQL using chain-of-thought reasoning.

    Breaks down the question into steps:
    1. Understand the question
    2. Identify tables/columns
    3. Plan query structure
    4. Generate SQL
    5. Validate and refine
    """
    agent = await get_agent_service()

    task = TaskCreate(
        type=TaskType.SQL_CHAIN,
        params={
            "question": request.question,
            "database": request.database,
            "context": request.context
        }
    )

    if request.async_exec:
        submitted = agent.submit_task(task)
        return {
            "task_id": submitted.task_id,
            "status": "queued",
            "message": "SQL chain-of-thought task submitted"
        }
    else:
        result = await agent.execute_task(task)
        return {
            "task_id": result.task_id,
            "status": result.status.value,
            "result": result.result,
            "duration_ms": result.duration_ms,
            "error": result.error
        }


@router.get("/task/{task_id}")
async def get_task_status(task_id: str):
    """Get status of a task"""
    agent = await get_agent_service()
    status = agent.get_task_status(task_id)

    if not status:
        raise HTTPException(status_code=404, detail="Task not found")

    return status


@router.get("/task/{task_id}/result")
async def get_task_result(task_id: str):
    """Get result of a completed task"""
    agent = await get_agent_service()
    result = agent.get_task_result(task_id)

    if not result:
        # Check if still running
        status = agent.get_task_status(task_id)
        if status:
            return {"status": status["status"], "message": "Task not yet completed"}
        raise HTTPException(status_code=404, detail="Task not found")

    return result.dict()


@router.delete("/task/{task_id}")
async def cancel_task(task_id: str):
    """Cancel a pending or running task"""
    agent = await get_agent_service()
    cancelled = agent.cancel_task(task_id)

    if not cancelled:
        raise HTTPException(status_code=404, detail="Task not found or already completed")

    return {"task_id": task_id, "status": "cancelled"}


@router.get("/queue")
async def get_queue_status():
    """Get current queue status"""
    agent = await get_agent_service()
    status = agent.get_status()

    return {
        "queue_size": status.queue_size,
        "running_tasks": status.current_tasks,
        "tasks_completed": status.tasks_completed,
        "tasks_failed": status.tasks_failed
    }
