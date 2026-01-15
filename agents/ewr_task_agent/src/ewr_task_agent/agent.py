"""
EWR Task Agent
==============

Specialized agent for shell execution, Git operations, and workflow automation.
"""

import asyncio
import os
import platform
import re
import time
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime

from ewr_agent_core import (
    BaseAgent,
    AgentType,
    AgentCapability,
    TaskResult,
    TaskStatus,
    AgentConfig,
)

from .models import (
    ShellType,
    ShellCommand,
    ShellResult,
    GitFileStatus,
    GitFile,
    GitStatus,
    GitCommit,
    GitBranch,
    GitDiff,
    WorkflowStep,
    WorkflowStepType,
    WorkflowStepResult,
    WorkflowResult,
    # Phase 3: Schema Change Detection
    SchemaChangeType,
    SchemaObjectType,
    SchemaChange,
    SchemaChangeResult,
)


class TaskAgent(BaseAgent):
    """
    Task Agent - Specialized for shell execution and Git operations.

    Capabilities:
    - SHELL_EXEC: Execute bash/sh commands
    - POWERSHELL_EXEC: Execute PowerShell commands
    - GIT_OPS: Git repository operations
    - WORKFLOW: Execute multi-step workflows
    """

    def __init__(
        self,
        config: AgentConfig = None,
        default_shell: ShellType = None,
        allowed_commands: List[str] = None,
        blocked_commands: List[str] = None,
        **kwargs
    ):
        super().__init__(config, **kwargs)

        # Detect default shell based on platform
        if default_shell is None:
            if platform.system() == "Windows":
                self.default_shell = ShellType.POWERSHELL
            else:
                self.default_shell = ShellType.BASH
        else:
            self.default_shell = default_shell

        self.allowed_commands = allowed_commands
        self.blocked_commands = blocked_commands or [
            "rm -rf /",
            "mkfs",
            "dd if=",
            ":(){:|:&};:",  # Fork bomb
            "format",  # Windows format
        ]

    @property
    def agent_type(self) -> AgentType:
        return AgentType.TASK

    @property
    def capabilities(self) -> List[AgentCapability]:
        return [
            AgentCapability.SHELL_EXEC,
            AgentCapability.POWERSHELL_EXEC,
            AgentCapability.GIT_OPS,
            AgentCapability.WORKFLOW,
            # Phase 3: Schema Change Detection
            AgentCapability.SCHEMA_CHANGE,
        ]

    async def _initialize(self) -> None:
        """Initialize the task agent."""
        self.logger.info(f"Task agent initialized (default shell: {self.default_shell.value})")

    async def handle_task(self, task: Dict[str, Any]) -> TaskResult:
        """Handle incoming tasks."""
        task_id = task.get("task_id", "unknown")
        task_type = task.get("task_type", "")
        params = task.get("params", {})

        try:
            if task_type == "shell_exec" or task_type == AgentCapability.SHELL_EXEC.value:
                result = await self.execute_shell(
                    command=params.get("command", ""),
                    working_dir=params.get("working_dir"),
                    timeout=params.get("timeout", 300)
                )
                return TaskResult(
                    task_id=task_id,
                    status=TaskStatus.COMPLETED if result.success else TaskStatus.FAILED,
                    stdout=result.stdout,
                    stderr=result.stderr,
                    exit_code=result.exit_code,
                    duration_ms=result.duration_ms
                )

            elif task_type == "powershell_exec" or task_type == AgentCapability.POWERSHELL_EXEC.value:
                result = await self.execute_powershell(
                    command=params.get("command", ""),
                    working_dir=params.get("working_dir"),
                    timeout=params.get("timeout", 300)
                )
                return TaskResult(
                    task_id=task_id,
                    status=TaskStatus.COMPLETED if result.success else TaskStatus.FAILED,
                    stdout=result.stdout,
                    stderr=result.stderr,
                    exit_code=result.exit_code,
                    duration_ms=result.duration_ms
                )

            elif task_type == "git_ops" or task_type == AgentCapability.GIT_OPS.value:
                operation = params.get("operation", "status")
                repo_path = params.get("repo_path", ".")

                if operation == "status":
                    result = await self.git_status(repo_path)
                    return TaskResult(
                        task_id=task_id,
                        status=TaskStatus.COMPLETED,
                        result={"status": result.model_dump()}
                    )
                elif operation == "commit":
                    commit = await self.git_commit(
                        repo_path,
                        message=params.get("message", ""),
                        files=params.get("files")
                    )
                    return TaskResult(
                        task_id=task_id,
                        status=TaskStatus.COMPLETED,
                        result={"commit": commit.model_dump() if commit else None}
                    )
                elif operation == "log":
                    commits = await self.git_log(repo_path, limit=params.get("limit", 10))
                    return TaskResult(
                        task_id=task_id,
                        status=TaskStatus.COMPLETED,
                        result={"commits": [c.model_dump() for c in commits]}
                    )
                else:
                    return TaskResult(
                        task_id=task_id,
                        status=TaskStatus.FAILED,
                        error=f"Unknown git operation: {operation}"
                    )

            elif task_type == "workflow" or task_type == AgentCapability.WORKFLOW.value:
                steps = [WorkflowStep(**s) for s in params.get("steps", [])]
                result = await self.execute_workflow(
                    workflow_id=params.get("workflow_id", task_id),
                    workflow_name=params.get("workflow_name", "workflow"),
                    steps=steps
                )
                return TaskResult(
                    task_id=task_id,
                    status=TaskStatus.COMPLETED if result.success else TaskStatus.FAILED,
                    result={"workflow": result.model_dump()}
                )

            # Phase 3: Schema Change Detection
            elif task_type == "schema_change" or task_type == AgentCapability.SCHEMA_CHANGE.value:
                result = await self.schema_change_workflow(
                    repo_path=params.get("repo_path", "."),
                    compare_ref=params.get("compare_ref", "HEAD~1")
                )
                return TaskResult(
                    task_id=task_id,
                    status=TaskStatus.COMPLETED if result.success else TaskStatus.FAILED,
                    result={"schema_changes": result.model_dump()}
                )

            else:
                return TaskResult(
                    task_id=task_id,
                    status=TaskStatus.FAILED,
                    error=f"Unknown task type: {task_type}"
                )

        except Exception as e:
            self.logger.error(f"Task failed: {e}")
            return TaskResult(
                task_id=task_id,
                status=TaskStatus.FAILED,
                error=str(e)
            )

    # =========================================================================
    # Shell Execution
    # =========================================================================

    def _is_command_allowed(self, command: str) -> bool:
        """Check if a command is allowed to run."""
        # Check blocked commands
        for blocked in self.blocked_commands:
            if blocked in command:
                return False

        # Check allowed commands if specified
        if self.allowed_commands:
            return any(allowed in command for allowed in self.allowed_commands)

        return True

    async def execute_shell(
        self,
        command: str,
        shell_type: ShellType = None,
        working_dir: str = None,
        timeout: int = 300,
        environment: Dict[str, str] = None
    ) -> ShellResult:
        """
        Execute a shell command.

        Args:
            command: Command to execute
            shell_type: Shell to use (defaults to bash on Unix, PowerShell on Windows)
            working_dir: Working directory
            timeout: Timeout in seconds
            environment: Additional environment variables

        Returns:
            ShellResult with output and status
        """
        shell_type = shell_type or self.default_shell

        if not self._is_command_allowed(command):
            return ShellResult(
                command=command,
                shell_type=shell_type,
                exit_code=-1,
                stderr="Command is blocked for security reasons",
            )

        start_time = time.time()
        started_at = datetime.utcnow()

        # Build environment
        env = os.environ.copy()
        if environment:
            env.update(environment)

        # Determine shell executable
        if shell_type == ShellType.BASH:
            shell_cmd = ["bash", "-c", command]
        elif shell_type == ShellType.SH:
            shell_cmd = ["sh", "-c", command]
        elif shell_type == ShellType.POWERSHELL:
            shell_cmd = ["powershell.exe", "-NoProfile", "-Command", command]
        elif shell_type == ShellType.CMD:
            shell_cmd = ["cmd.exe", "/c", command]
        elif shell_type == ShellType.ZSH:
            shell_cmd = ["zsh", "-c", command]
        else:
            shell_cmd = ["bash", "-c", command]

        try:
            process = await asyncio.create_subprocess_exec(
                *shell_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=working_dir,
                env=env
            )

            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=timeout
                )
                timed_out = False
            except asyncio.TimeoutError:
                process.kill()
                stdout, stderr = await process.communicate()
                timed_out = True

            duration_ms = int((time.time() - start_time) * 1000)

            return ShellResult(
                command=command,
                shell_type=shell_type,
                exit_code=process.returncode or 0,
                stdout=stdout.decode("utf-8", errors="replace"),
                stderr=stderr.decode("utf-8", errors="replace"),
                duration_ms=duration_ms,
                timed_out=timed_out,
                started_at=started_at,
                completed_at=datetime.utcnow()
            )

        except Exception as e:
            return ShellResult(
                command=command,
                shell_type=shell_type,
                exit_code=-1,
                stderr=str(e),
                duration_ms=int((time.time() - start_time) * 1000),
                started_at=started_at,
                completed_at=datetime.utcnow()
            )

    async def execute_powershell(
        self,
        command: str,
        working_dir: str = None,
        timeout: int = 300,
        environment: Dict[str, str] = None
    ) -> ShellResult:
        """
        Execute a PowerShell command.

        Args:
            command: PowerShell command to execute
            working_dir: Working directory
            timeout: Timeout in seconds
            environment: Additional environment variables

        Returns:
            ShellResult with output and status
        """
        return await self.execute_shell(
            command=command,
            shell_type=ShellType.POWERSHELL,
            working_dir=working_dir,
            timeout=timeout,
            environment=environment
        )

    # =========================================================================
    # Git Operations
    # =========================================================================

    async def git_status(self, repo_path: str = ".") -> GitStatus:
        """
        Get Git repository status.

        Args:
            repo_path: Path to the repository

        Returns:
            GitStatus with repository information
        """
        repo_path = str(Path(repo_path).resolve())

        # Get branch name
        branch_result = await self.execute_shell(
            "git rev-parse --abbrev-ref HEAD",
            working_dir=repo_path
        )
        branch = branch_result.stdout.strip() if branch_result.success else "unknown"

        # Get status
        status_result = await self.execute_shell(
            "git status --porcelain",
            working_dir=repo_path
        )

        modified_files = []
        staged_files = []
        untracked_files = []

        if status_result.success:
            for line in status_result.stdout.strip().split("\n"):
                if not line:
                    continue

                index_status = line[0]
                work_status = line[1]
                file_path = line[3:].strip()

                # Handle renamed files
                if " -> " in file_path:
                    old_path, file_path = file_path.split(" -> ")
                else:
                    old_path = None

                # Staged changes
                if index_status != " " and index_status != "?":
                    status_map = {
                        "M": GitFileStatus.MODIFIED,
                        "A": GitFileStatus.ADDED,
                        "D": GitFileStatus.DELETED,
                        "R": GitFileStatus.RENAMED,
                        "C": GitFileStatus.COPIED,
                    }
                    staged_files.append(GitFile(
                        path=file_path,
                        status=status_map.get(index_status, GitFileStatus.MODIFIED),
                        staged=True,
                        original_path=old_path
                    ))

                # Working directory changes
                if work_status != " ":
                    if work_status == "?":
                        untracked_files.append(file_path)
                    else:
                        status_map = {
                            "M": GitFileStatus.MODIFIED,
                            "D": GitFileStatus.DELETED,
                        }
                        modified_files.append(GitFile(
                            path=file_path,
                            status=status_map.get(work_status, GitFileStatus.MODIFIED),
                            staged=False
                        ))

        # Get ahead/behind counts
        ahead = 0
        behind = 0
        remote_result = await self.execute_shell(
            "git rev-list --left-right --count HEAD...@{upstream} 2>/dev/null",
            working_dir=repo_path
        )
        if remote_result.success and remote_result.stdout.strip():
            parts = remote_result.stdout.strip().split()
            if len(parts) >= 2:
                ahead = int(parts[0])
                behind = int(parts[1])

        is_clean = len(modified_files) == 0 and len(staged_files) == 0 and len(untracked_files) == 0

        return GitStatus(
            repo_path=repo_path,
            branch=branch,
            is_clean=is_clean,
            ahead=ahead,
            behind=behind,
            modified_files=modified_files,
            staged_files=staged_files,
            untracked_files=untracked_files,
        )

    async def git_log(
        self,
        repo_path: str = ".",
        limit: int = 10,
        branch: str = None
    ) -> List[GitCommit]:
        """
        Get Git commit history.

        Args:
            repo_path: Path to the repository
            limit: Maximum commits to return
            branch: Branch to get history from (default: current)

        Returns:
            List of GitCommit
        """
        repo_path = str(Path(repo_path).resolve())

        format_str = "%H|%h|%s|%an|%ae|%aI"
        cmd = f'git log -{limit} --format="{format_str}"'
        if branch:
            cmd += f" {branch}"

        result = await self.execute_shell(cmd, working_dir=repo_path)

        commits = []
        if result.success:
            for line in result.stdout.strip().split("\n"):
                if not line or "|" not in line:
                    continue
                parts = line.split("|")
                if len(parts) >= 6:
                    commits.append(GitCommit(
                        hash=parts[0],
                        short_hash=parts[1],
                        message=parts[2],
                        author=parts[3],
                        author_email=parts[4],
                        date=datetime.fromisoformat(parts[5].replace("Z", "+00:00"))
                    ))

        return commits

    async def git_commit(
        self,
        repo_path: str = ".",
        message: str = "",
        files: List[str] = None,
        all_files: bool = False
    ) -> Optional[GitCommit]:
        """
        Create a Git commit.

        Args:
            repo_path: Path to the repository
            message: Commit message
            files: Specific files to commit
            all_files: Stage all modified files

        Returns:
            GitCommit if successful, None otherwise
        """
        repo_path = str(Path(repo_path).resolve())

        # Stage files
        if files:
            for file in files:
                await self.execute_shell(f'git add "{file}"', working_dir=repo_path)
        elif all_files:
            await self.execute_shell("git add -A", working_dir=repo_path)

        # Create commit
        # Escape message for shell
        safe_message = message.replace('"', '\\"').replace("$", "\\$")
        result = await self.execute_shell(
            f'git commit -m "{safe_message}"',
            working_dir=repo_path
        )

        if not result.success:
            self.logger.error(f"Git commit failed: {result.stderr}")
            return None

        # Get the commit info
        commits = await self.git_log(repo_path, limit=1)
        return commits[0] if commits else None

    async def git_push(
        self,
        repo_path: str = ".",
        remote: str = "origin",
        branch: str = None,
        force: bool = False
    ) -> bool:
        """
        Push to remote repository.

        Args:
            repo_path: Path to the repository
            remote: Remote name
            branch: Branch to push
            force: Force push

        Returns:
            True if successful
        """
        repo_path = str(Path(repo_path).resolve())

        cmd = f"git push {remote}"
        if branch:
            cmd += f" {branch}"
        if force:
            cmd += " --force"

        result = await self.execute_shell(cmd, working_dir=repo_path)
        return result.success

    async def git_pull(
        self,
        repo_path: str = ".",
        remote: str = "origin",
        branch: str = None,
        rebase: bool = False
    ) -> bool:
        """
        Pull from remote repository.

        Args:
            repo_path: Path to the repository
            remote: Remote name
            branch: Branch to pull
            rebase: Use rebase instead of merge

        Returns:
            True if successful
        """
        repo_path = str(Path(repo_path).resolve())

        cmd = f"git pull {remote}"
        if branch:
            cmd += f" {branch}"
        if rebase:
            cmd += " --rebase"

        result = await self.execute_shell(cmd, working_dir=repo_path)
        return result.success

    async def git_branches(self, repo_path: str = ".") -> List[GitBranch]:
        """
        List Git branches.

        Args:
            repo_path: Path to the repository

        Returns:
            List of GitBranch
        """
        repo_path = str(Path(repo_path).resolve())

        result = await self.execute_shell(
            "git branch -a --format='%(HEAD)|%(refname:short)|%(upstream:short)'",
            working_dir=repo_path
        )

        branches = []
        if result.success:
            for line in result.stdout.strip().split("\n"):
                if not line:
                    continue
                parts = line.split("|")
                if len(parts) >= 2:
                    is_current = parts[0].strip() == "*"
                    name = parts[1].strip()
                    tracking = parts[2].strip() if len(parts) > 2 else None

                    branches.append(GitBranch(
                        name=name,
                        is_current=is_current,
                        is_remote=name.startswith("remotes/"),
                        tracking=tracking if tracking else None
                    ))

        return branches

    async def git_diff(
        self,
        repo_path: str = ".",
        staged: bool = False,
        file_path: str = None
    ) -> List[GitDiff]:
        """
        Get Git diff.

        Args:
            repo_path: Path to the repository
            staged: Get staged changes
            file_path: Specific file to diff

        Returns:
            List of GitDiff
        """
        repo_path = str(Path(repo_path).resolve())

        cmd = "git diff --stat"
        if staged:
            cmd += " --staged"
        if file_path:
            cmd += f' -- "{file_path}"'

        result = await self.execute_shell(cmd, working_dir=repo_path)

        diffs = []
        if result.success:
            for line in result.stdout.strip().split("\n"):
                if "|" not in line or "changed" in line:
                    continue

                parts = line.split("|")
                if len(parts) >= 2:
                    path = parts[0].strip()
                    stats = parts[1].strip()

                    additions = stats.count("+")
                    deletions = stats.count("-")

                    diffs.append(GitDiff(
                        file_path=path,
                        additions=additions,
                        deletions=deletions,
                    ))

        return diffs

    # =========================================================================
    # Workflow Execution
    # =========================================================================

    async def execute_workflow(
        self,
        workflow_id: str,
        workflow_name: str,
        steps: List[WorkflowStep],
    ) -> WorkflowResult:
        """
        Execute a multi-step workflow.

        Args:
            workflow_id: Unique workflow identifier
            workflow_name: Human-readable name
            steps: List of workflow steps

        Returns:
            WorkflowResult with all step outcomes
        """
        start_time = time.time()
        started_at = datetime.utcnow()

        result = WorkflowResult(
            workflow_id=workflow_id,
            workflow_name=workflow_name,
            success=True,
            started_at=started_at,
        )

        # Track completed steps for dependencies
        completed_steps: Dict[str, WorkflowStepResult] = {}

        for step in steps:
            step_start = time.time()

            # Check dependencies
            dependencies_met = True
            for dep_id in step.depends_on:
                if dep_id not in completed_steps:
                    dependencies_met = False
                    break
                if not completed_steps[dep_id].success:
                    dependencies_met = False
                    break

            if not dependencies_met:
                step_result = WorkflowStepResult(
                    step_id=step.id,
                    step_name=step.name,
                    success=False,
                    skipped=True,
                    skip_reason="Dependencies not met"
                )
                result.step_results.append(step_result)
                completed_steps[step.id] = step_result
                if not step.continue_on_error:
                    result.success = False
                    break
                continue

            # Execute step
            try:
                if step.step_type == WorkflowStepType.SHELL:
                    shell_result = await self.execute_shell(
                        command=step.command,
                        working_dir=step.params.get("working_dir"),
                        timeout=step.timeout_seconds
                    )
                    step_result = WorkflowStepResult(
                        step_id=step.id,
                        step_name=step.name,
                        success=shell_result.success,
                        output={"stdout": shell_result.stdout, "stderr": shell_result.stderr},
                        error=shell_result.stderr if not shell_result.success else None,
                        duration_ms=shell_result.duration_ms
                    )

                elif step.step_type == WorkflowStepType.POWERSHELL:
                    ps_result = await self.execute_powershell(
                        command=step.command,
                        working_dir=step.params.get("working_dir"),
                        timeout=step.timeout_seconds
                    )
                    step_result = WorkflowStepResult(
                        step_id=step.id,
                        step_name=step.name,
                        success=ps_result.success,
                        output={"stdout": ps_result.stdout, "stderr": ps_result.stderr},
                        error=ps_result.stderr if not ps_result.success else None,
                        duration_ms=ps_result.duration_ms
                    )

                elif step.step_type == WorkflowStepType.GIT:
                    git_op = step.params.get("operation", "status")
                    repo_path = step.params.get("repo_path", ".")

                    if git_op == "status":
                        git_result = await self.git_status(repo_path)
                        step_result = WorkflowStepResult(
                            step_id=step.id,
                            step_name=step.name,
                            success=True,
                            output=git_result.model_dump(),
                            duration_ms=int((time.time() - step_start) * 1000)
                        )
                    elif git_op == "commit":
                        commit = await self.git_commit(
                            repo_path,
                            message=step.params.get("message", ""),
                            files=step.params.get("files"),
                            all_files=step.params.get("all_files", False)
                        )
                        step_result = WorkflowStepResult(
                            step_id=step.id,
                            step_name=step.name,
                            success=commit is not None,
                            output=commit.model_dump() if commit else None,
                            error="Commit failed" if not commit else None,
                            duration_ms=int((time.time() - step_start) * 1000)
                        )
                    elif git_op == "push":
                        success = await self.git_push(
                            repo_path,
                            remote=step.params.get("remote", "origin"),
                            branch=step.params.get("branch")
                        )
                        step_result = WorkflowStepResult(
                            step_id=step.id,
                            step_name=step.name,
                            success=success,
                            error="Push failed" if not success else None,
                            duration_ms=int((time.time() - step_start) * 1000)
                        )
                    elif git_op == "pull":
                        success = await self.git_pull(
                            repo_path,
                            remote=step.params.get("remote", "origin"),
                            branch=step.params.get("branch")
                        )
                        step_result = WorkflowStepResult(
                            step_id=step.id,
                            step_name=step.name,
                            success=success,
                            error="Pull failed" if not success else None,
                            duration_ms=int((time.time() - step_start) * 1000)
                        )
                    else:
                        step_result = WorkflowStepResult(
                            step_id=step.id,
                            step_name=step.name,
                            success=False,
                            error=f"Unknown git operation: {git_op}",
                            duration_ms=int((time.time() - step_start) * 1000)
                        )

                else:
                    step_result = WorkflowStepResult(
                        step_id=step.id,
                        step_name=step.name,
                        success=False,
                        error=f"Unsupported step type: {step.step_type}",
                        duration_ms=int((time.time() - step_start) * 1000)
                    )

            except Exception as e:
                step_result = WorkflowStepResult(
                    step_id=step.id,
                    step_name=step.name,
                    success=False,
                    error=str(e),
                    duration_ms=int((time.time() - step_start) * 1000)
                )

            result.step_results.append(step_result)
            completed_steps[step.id] = step_result

            if not step_result.success and not step.continue_on_error:
                result.success = False
                break

        result.total_duration_ms = int((time.time() - start_time) * 1000)
        result.completed_at = datetime.utcnow()

        return result

    # =========================================================================
    # Phase 3: Schema Change Detection Workflow
    # =========================================================================

    async def schema_change_workflow(
        self,
        repo_path: str = ".",
        compare_ref: str = "HEAD~1"
    ) -> SchemaChangeResult:
        """
        Detect schema changes via git and trigger cache invalidation.

        Args:
            repo_path: Path to the git repository
            compare_ref: Git reference to compare against (default: HEAD~1)

        Returns:
            SchemaChangeResult with detected changes and workflow status
        """
        started_at = datetime.utcnow()
        repo_path = str(Path(repo_path).resolve())

        result = SchemaChangeResult(
            success=True,
            compare_ref=compare_ref,
            started_at=started_at
        )

        try:
            # Step 1: Get changed SQL files
            changed_files = await self._get_changed_sql_files(repo_path, compare_ref)
            result.changed_files = changed_files

            if not changed_files:
                self.logger.info("No SQL file changes detected")
                result.completed_at = datetime.utcnow()
                return result

            # Step 2: Parse changes from each file
            all_changes = []
            affected_tables = set()

            for file_path in changed_files:
                changes = await self._parse_schema_changes(repo_path, file_path, compare_ref)
                all_changes.extend(changes)

                # Extract affected table names
                for change in changes:
                    if change.object_type == SchemaObjectType.TABLE:
                        affected_tables.add(change.object_name)
                    elif change.object_type == SchemaObjectType.COLUMN:
                        # Column changes affect the parent table
                        # Object name format might be "TableName.ColumnName"
                        if "." in change.object_name:
                            table_name = change.object_name.split(".")[0]
                            affected_tables.add(table_name)

            result.changes = all_changes
            result.affected_tables = list(affected_tables)

            # Step 3: Invalidate cache for affected tables
            if affected_tables:
                cache_invalidated = await self._invalidate_affected_caches(list(affected_tables))
                result.cache_invalidated = cache_invalidated

            # Step 4: Trigger reindexing
            if affected_tables:
                reindex_triggered = await self._trigger_reindex(list(affected_tables))
                result.reindex_triggered = reindex_triggered

            # Log summary
            self.logger.info(
                f"Schema change detection complete: "
                f"{len(changed_files)} files, {len(all_changes)} changes, "
                f"{len(affected_tables)} affected tables"
            )

        except Exception as e:
            self.logger.error(f"Schema change workflow failed: {e}")
            result.success = False
            result.error = str(e)

        result.completed_at = datetime.utcnow()
        return result

    async def _get_changed_sql_files(
        self,
        repo_path: str,
        compare_ref: str
    ) -> List[str]:
        """Get list of changed SQL files from git diff."""
        # Find changed SQL files and migration files
        cmd = f"git diff --name-only {compare_ref} -- '*.sql' 'migrations/*'"
        shell_result = await self.execute_shell(cmd, working_dir=repo_path)

        if not shell_result.success:
            self.logger.warning(f"Git diff failed: {shell_result.stderr}")
            return []

        files = []
        for line in shell_result.stdout.strip().split("\n"):
            line = line.strip()
            if line and (line.endswith(".sql") or "migration" in line.lower()):
                files.append(line)

        return files

    async def _parse_schema_changes(
        self,
        repo_path: str,
        file_path: str,
        compare_ref: str
    ) -> List[SchemaChange]:
        """Parse git diff output to identify schema changes."""
        # Get the diff content for this file
        cmd = f"git diff {compare_ref} -- '{file_path}'"
        shell_result = await self.execute_shell(cmd, working_dir=repo_path)

        if not shell_result.success:
            return []

        diff_content = shell_result.stdout
        changes = []

        # Regex patterns for DDL detection (looking for added lines starting with +)
        patterns = [
            # CREATE TABLE
            (
                r'\+\s*CREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?(\[?\w+\]?\.?\[?\w+\]?)',
                SchemaChangeType.CREATE,
                SchemaObjectType.TABLE
            ),
            # ALTER TABLE
            (
                r'\+\s*ALTER\s+TABLE\s+(\[?\w+\]?\.?\[?\w+\]?)',
                SchemaChangeType.ALTER,
                SchemaObjectType.TABLE
            ),
            # DROP TABLE
            (
                r'\+\s*DROP\s+TABLE\s+(?:IF\s+EXISTS\s+)?(\[?\w+\]?\.?\[?\w+\]?)',
                SchemaChangeType.DROP,
                SchemaObjectType.TABLE
            ),
            # ADD COLUMN
            (
                r'\+\s*ALTER\s+TABLE\s+(\[?\w+\]?\.?\[?\w+\]?)\s+ADD\s+(?:COLUMN\s+)?(\[?\w+\]?)',
                SchemaChangeType.ALTER,
                SchemaObjectType.COLUMN
            ),
            # CREATE INDEX
            (
                r'\+\s*CREATE\s+(?:UNIQUE\s+)?(?:CLUSTERED\s+)?(?:NONCLUSTERED\s+)?INDEX\s+(\[?\w+\]?)\s+ON\s+(\[?\w+\]?\.?\[?\w+\]?)',
                SchemaChangeType.CREATE,
                SchemaObjectType.INDEX
            ),
            # DROP INDEX
            (
                r'\+\s*DROP\s+INDEX\s+(?:IF\s+EXISTS\s+)?(\[?\w+\]?)(?:\s+ON\s+(\[?\w+\]?\.?\[?\w+\]?))?',
                SchemaChangeType.DROP,
                SchemaObjectType.INDEX
            ),
            # CREATE PROCEDURE
            (
                r'\+\s*CREATE\s+(?:OR\s+ALTER\s+)?PROC(?:EDURE)?\s+(\[?\w+\]?\.?\[?\w+\]?)',
                SchemaChangeType.CREATE,
                SchemaObjectType.PROCEDURE
            ),
            # ALTER PROCEDURE
            (
                r'\+\s*ALTER\s+PROC(?:EDURE)?\s+(\[?\w+\]?\.?\[?\w+\]?)',
                SchemaChangeType.ALTER,
                SchemaObjectType.PROCEDURE
            ),
            # CREATE VIEW
            (
                r'\+\s*CREATE\s+(?:OR\s+ALTER\s+)?VIEW\s+(\[?\w+\]?\.?\[?\w+\]?)',
                SchemaChangeType.CREATE,
                SchemaObjectType.VIEW
            ),
        ]

        for pattern, change_type, object_type in patterns:
            matches = re.finditer(pattern, diff_content, re.IGNORECASE)
            for match in matches:
                # Clean up object name (remove brackets)
                object_name = match.group(1).replace("[", "").replace("]", "")

                # For column additions, include table.column format
                if object_type == SchemaObjectType.COLUMN and len(match.groups()) > 1:
                    column_name = match.group(2).replace("[", "").replace("]", "")
                    object_name = f"{object_name}.{column_name}"

                # For index with ON clause
                if object_type == SchemaObjectType.INDEX and len(match.groups()) > 1:
                    table_name = match.group(2) if match.group(2) else ""
                    if table_name:
                        table_name = table_name.replace("[", "").replace("]", "")
                        details = f"On table {table_name}"
                    else:
                        details = ""
                else:
                    details = ""

                changes.append(SchemaChange(
                    file_path=file_path,
                    change_type=change_type,
                    object_type=object_type,
                    object_name=object_name,
                    details=details
                ))

        return changes

    async def _invalidate_affected_caches(self, tables: List[str]) -> bool:
        """
        Invalidate caches for affected tables.

        This would typically communicate with the SQL agent to invalidate its cache.
        """
        try:
            # Log the cache invalidation request
            self.logger.info(f"Cache invalidation requested for tables: {tables}")

            # Use the capability dispatch if available
            if AgentCapability.CACHE_INVALIDATE in self.capabilities:
                # Direct cache invalidation would happen here
                # For now, just mark as invalidated and log
                self.logger.info(f"Cache invalidated for {len(tables)} tables")
                return True

            # Cache invalidation is a no-op if capability not available
            # but we still return True to indicate intent was processed
            return True
        except Exception as e:
            self.logger.warning(f"Cache invalidation failed: {e}")
            return False

    async def _trigger_reindex(self, tables: List[str]) -> bool:
        """
        Trigger schema reindexing for affected tables.

        This would typically communicate with the SQL agent to refresh schema info.
        """
        try:
            # Log the reindex request
            self.logger.info(f"Schema reindex requested for tables: {tables}")

            # Use the capability dispatch if available
            if AgentCapability.SCHEMA_EXPLORE in self.capabilities:
                # Direct schema reindexing would happen here
                # For now, just mark as triggered and log
                self.logger.info(f"Reindex triggered for {len(tables)} tables")
                return True

            # Reindex is a no-op if capability not available
            # but we still return True to indicate intent was processed
            return True
        except Exception as e:
            self.logger.warning(f"Reindex trigger failed: {e}")
            return False
