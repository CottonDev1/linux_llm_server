"""
Base Pipeline Classes

Provides the foundational classes for building step-based pipelines.
Based on the SQL pipeline pattern for consistency.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, AsyncGenerator
from enum import Enum
import time
import logging


class PipelineStatus(Enum):
    """Status of pipeline execution."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CACHED = "cached"
    STOPPED = "stopped"


@dataclass
class PipelineContext:
    """
    Shared state passed between pipeline steps.

    This is the data container that flows through the pipeline,
    with each step reading from and writing to it.

    Note: Subclasses should extend this for pipeline-specific context.
    """
    # Control flow
    should_stop: bool = False
    skip_remaining: bool = False
    use_cached: bool = False

    # Metadata
    step_timings: Dict[str, float] = field(default_factory=dict)
    errors: List[Dict[str, str]] = field(default_factory=list)
    status: PipelineStatus = PipelineStatus.PENDING


@dataclass
class StepResult:
    """
    Result from a single pipeline step.

    Attributes:
        success: Whether the step completed successfully
        data: Optional data produced by the step
        error: Error message if step failed
        should_stop: Whether to stop pipeline execution
        skip_remaining: Whether to skip remaining steps
    """
    success: bool
    data: Optional[Any] = None
    error: Optional[str] = None
    should_stop: bool = False
    skip_remaining: bool = False


class PipelineStep(ABC):
    """
    Base class for all pipeline steps.

    Each step is a discrete unit of work in the pipeline.
    Steps should be:
    - Independent: Minimal coupling with other steps
    - Testable: Easy to unit test in isolation
    - Reusable: Configurable for different contexts
    - Observable: Log progress and errors

    Example:
        class MyStep(PipelineStep):
            name = "my_step"

            async def execute(self, context: PipelineContext) -> StepResult:
                try:
                    # Do work
                    result = await self.do_something(context)
                    return StepResult(success=True, data=result)
                except Exception as e:
                    return StepResult(success=False, error=str(e))
    """

    name: str = "base_step"

    def __init__(self):
        """Initialize the pipeline step."""
        self.logger = logging.getLogger(f"pipeline.{self.name}")

    @abstractmethod
    async def execute(self, context: PipelineContext) -> StepResult:
        """
        Execute this step.

        Args:
            context: Shared pipeline context

        Returns:
            StepResult with execution outcome
        """
        pass

    async def on_error(self, context: PipelineContext, error: Exception) -> StepResult:
        """
        Handle errors in this step.

        Override to implement custom error handling logic.

        Args:
            context: Shared pipeline context
            error: The exception that occurred

        Returns:
            StepResult with error information
        """
        self.logger.error(f"Step {self.name} failed: {error}")
        return StepResult(
            success=False,
            error=str(error),
            should_stop=False  # Default: continue pipeline
        )

    async def before_execute(self, context: PipelineContext) -> None:
        """
        Hook called before execute().

        Override for setup logic like validation, resource allocation, etc.
        """
        pass

    async def after_execute(self, context: PipelineContext, result: StepResult) -> None:
        """
        Hook called after execute().

        Override for cleanup logic like releasing resources, logging, etc.
        """
        pass


class Pipeline:
    """
    Base pipeline orchestrator.

    Executes a sequence of PipelineStep instances, managing:
    - Sequential execution
    - Error handling
    - Timing/metrics
    - Early termination
    - Streaming updates

    Example:
        steps = [Step1(), Step2(), Step3()]
        pipeline = Pipeline(steps)

        context = PipelineContext(query="test")
        result = await pipeline.execute(context)
    """

    def __init__(self, steps: List[PipelineStep], config: Optional[Dict] = None):
        """
        Initialize the pipeline.

        Args:
            steps: List of PipelineStep instances to execute
            config: Optional configuration dictionary
        """
        self.steps = steps
        self.config = config or {}
        self.logger = logging.getLogger(f"pipeline.{self.__class__.__name__}")

    async def execute(self, context: PipelineContext) -> PipelineContext:
        """
        Execute all steps in sequence.

        Args:
            context: Pipeline context with input data

        Returns:
            Updated context with results from all steps
        """
        context.status = PipelineStatus.RUNNING

        for step in self.steps:
            if context.should_stop or context.skip_remaining:
                self.logger.info(f"Pipeline stopping early at step {step.name}")
                break

            start_time = time.time()

            try:
                # Before hook
                await step.before_execute(context)

                # Execute step
                self.logger.debug(f"Executing step: {step.name}")
                result = await step.execute(context)

                # Record timing
                elapsed = time.time() - start_time
                context.step_timings[step.name] = elapsed

                # After hook
                await step.after_execute(context, result)

                # Handle result
                if not result.success:
                    context.errors.append({
                        "step": step.name,
                        "error": result.error,
                        "timestamp": time.time()
                    })

                    if result.should_stop:
                        self.logger.warning(f"Step {step.name} requested stop")
                        context.should_stop = True
                        break

                if result.skip_remaining:
                    self.logger.info(f"Step {step.name} requested skip remaining")
                    context.skip_remaining = True
                    break

            except Exception as e:
                # Error handling
                self.logger.error(f"Step {step.name} raised exception: {e}")

                error_result = await step.on_error(context, e)

                context.errors.append({
                    "step": step.name,
                    "error": str(e),
                    "timestamp": time.time()
                })

                elapsed = time.time() - start_time
                context.step_timings[step.name] = elapsed

                if error_result.should_stop:
                    context.should_stop = True
                    break

        # Set final status
        if context.errors:
            context.status = PipelineStatus.FAILED
        elif context.use_cached:
            context.status = PipelineStatus.CACHED
        else:
            context.status = PipelineStatus.COMPLETED

        return context

    async def execute_streaming(
        self,
        context: PipelineContext
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Execute pipeline with streaming updates.

        Yields progress events for each step, useful for SSE streaming.

        Args:
            context: Pipeline context with input data

        Yields:
            Dict events with step progress information
        """
        context.status = PipelineStatus.RUNNING

        yield {
            "event": "pipeline_start",
            "timestamp": time.time(),
            "steps": [step.name for step in self.steps]
        }

        for step in self.steps:
            if context.should_stop or context.skip_remaining:
                yield {
                    "event": "pipeline_stopped",
                    "step": step.name,
                    "timestamp": time.time()
                }
                break

            start_time = time.time()

            yield {
                "event": "step_start",
                "step": step.name,
                "timestamp": start_time
            }

            try:
                # Before hook
                await step.before_execute(context)

                # Execute step
                result = await step.execute(context)

                # Record timing
                elapsed = time.time() - start_time
                context.step_timings[step.name] = elapsed

                # After hook
                await step.after_execute(context, result)

                # Yield completion event
                yield {
                    "event": "step_complete",
                    "step": step.name,
                    "success": result.success,
                    "elapsed_ms": int(elapsed * 1000),
                    "data": result.data,
                    "timestamp": time.time()
                }

                # Handle errors
                if not result.success:
                    context.errors.append({
                        "step": step.name,
                        "error": result.error,
                        "timestamp": time.time()
                    })

                    yield {
                        "event": "step_error",
                        "step": step.name,
                        "error": result.error,
                        "timestamp": time.time()
                    }

                    if result.should_stop:
                        context.should_stop = True
                        break

                if result.skip_remaining:
                    context.skip_remaining = True
                    break

            except Exception as e:
                # Error handling
                error_result = await step.on_error(context, e)

                elapsed = time.time() - start_time
                context.step_timings[step.name] = elapsed

                context.errors.append({
                    "step": step.name,
                    "error": str(e),
                    "timestamp": time.time()
                })

                yield {
                    "event": "step_error",
                    "step": step.name,
                    "error": str(e),
                    "elapsed_ms": int(elapsed * 1000),
                    "timestamp": time.time()
                }

                if error_result.should_stop:
                    context.should_stop = True
                    break

        # Set final status
        if context.errors:
            context.status = PipelineStatus.FAILED
        elif context.use_cached:
            context.status = PipelineStatus.CACHED
        else:
            context.status = PipelineStatus.COMPLETED

        # Final event
        yield {
            "event": "pipeline_complete",
            "status": context.status.value,
            "total_time_ms": int(sum(context.step_timings.values()) * 1000),
            "errors": context.errors,
            "timestamp": time.time()
        }

    def get_step(self, name: str) -> Optional[PipelineStep]:
        """
        Get a step by name.

        Args:
            name: Step name

        Returns:
            PipelineStep instance or None if not found
        """
        for step in self.steps:
            if step.name == name:
                return step
        return None

    def insert_step(self, step: PipelineStep, after: Optional[str] = None) -> None:
        """
        Insert a step into the pipeline.

        Args:
            step: Step to insert
            after: Name of step to insert after (None = insert at beginning)
        """
        if after is None:
            self.steps.insert(0, step)
        else:
            for i, s in enumerate(self.steps):
                if s.name == after:
                    self.steps.insert(i + 1, step)
                    return
            raise ValueError(f"Step {after} not found")

    def remove_step(self, name: str) -> bool:
        """
        Remove a step from the pipeline.

        Args:
            name: Name of step to remove

        Returns:
            True if removed, False if not found
        """
        for i, step in enumerate(self.steps):
            if step.name == name:
                self.steps.pop(i)
                return True
        return False
