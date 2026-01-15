"""
EWR Task Agent
==============

Specialized agent for shell execution, Git operations, and workflow automation.

Capabilities:
- SHELL_EXEC: Execute shell commands (bash, sh)
- POWERSHELL_EXEC: Execute PowerShell commands
- GIT_OPS: Git repository operations (status, commit, push, etc.)
- WORKFLOW: Execute multi-step workflows

Usage:
    from ewr_task_agent import TaskAgent

    agent = TaskAgent(name="task-agent")
    await agent.start()

    # Execute a shell command
    result = await agent.execute_shell("ls -la")

    # Execute PowerShell
    result = await agent.execute_powershell("Get-Process")

    # Git operations
    status = await agent.git_status("/path/to/repo")
"""

from .agent import TaskAgent
from .models import (
    ShellCommand,
    ShellResult,
    GitStatus,
    GitCommit,
    WorkflowStep,
    WorkflowResult,
)

__version__ = "1.0.0"

__all__ = [
    "TaskAgent",
    "ShellCommand",
    "ShellResult",
    "GitStatus",
    "GitCommit",
    "WorkflowStep",
    "WorkflowResult",
]
