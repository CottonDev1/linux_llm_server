"""
EWR Document Agent - Retry Utilities
=====================================

Retry decorator and utilities for robust document processing.

Features:
- Exponential backoff with jitter
- Configurable retry strategies
- Per-operation retry configurations
"""

import asyncio
import random
import logging
from functools import wraps
from typing import Callable, TypeVar, Tuple, Type, Optional, Any
from dataclasses import dataclass

T = TypeVar('T')
logger = logging.getLogger(__name__)


@dataclass
class RetryConfig:
    """Configuration for retry behavior."""
    max_retries: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    jitter: float = 0.1
    retryable_exceptions: Tuple[Type[Exception], ...] = (Exception,)

    def calculate_delay(self, attempt: int) -> float:
        """Calculate delay for a given attempt with exponential backoff and jitter."""
        delay = min(
            self.base_delay * (self.exponential_base ** attempt),
            self.max_delay
        )
        # Add jitter to prevent thundering herd
        jitter_range = delay * self.jitter
        delay += random.uniform(-jitter_range, jitter_range)
        return max(0, delay)


def with_retry(config: Optional[RetryConfig] = None):
    """
    Decorator for async functions with retry logic.

    Usage:
        @with_retry(RetryConfig(max_retries=3))
        async def embed_chunk(chunk: DocumentChunk):
            ...

    Args:
        config: Retry configuration. Uses defaults if not provided.

    Returns:
        Decorated function with retry behavior.
    """
    if config is None:
        config = RetryConfig()

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> T:
            last_exception: Optional[Exception] = None

            for attempt in range(config.max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except config.retryable_exceptions as e:
                    last_exception = e

                    if attempt == config.max_retries:
                        logger.error(
                            f"Function {func.__name__} failed after "
                            f"{config.max_retries} retries: {e}"
                        )
                        raise

                    delay = config.calculate_delay(attempt)
                    logger.warning(
                        f"Attempt {attempt + 1}/{config.max_retries + 1} failed "
                        f"for {func.__name__}, retrying in {delay:.2f}s: {e}"
                    )

                    await asyncio.sleep(delay)

            # Should not reach here, but just in case
            if last_exception:
                raise last_exception
            raise RuntimeError("Unexpected state in retry logic")

        return wrapper
    return decorator


def with_retry_sync(config: Optional[RetryConfig] = None):
    """
    Decorator for synchronous functions with retry logic.

    Usage:
        @with_retry_sync(RetryConfig(max_retries=3))
        def read_file(path: str):
            ...
    """
    import time

    if config is None:
        config = RetryConfig()

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            last_exception: Optional[Exception] = None

            for attempt in range(config.max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except config.retryable_exceptions as e:
                    last_exception = e

                    if attempt == config.max_retries:
                        logger.error(
                            f"Function {func.__name__} failed after "
                            f"{config.max_retries} retries: {e}"
                        )
                        raise

                    delay = config.calculate_delay(attempt)
                    logger.warning(
                        f"Attempt {attempt + 1}/{config.max_retries + 1} failed "
                        f"for {func.__name__}, retrying in {delay:.2f}s: {e}"
                    )

                    time.sleep(delay)

            if last_exception:
                raise last_exception
            raise RuntimeError("Unexpected state in retry logic")

        return wrapper
    return decorator


# ============================================================================
# Pre-configured Retry Strategies
# ============================================================================

# For embedding generation (may timeout on slow LLM)
EMBEDDING_RETRY = RetryConfig(
    max_retries=3,
    base_delay=2.0,
    max_delay=30.0,
    exponential_base=2.0,
    retryable_exceptions=(ConnectionError, TimeoutError, OSError),
)

# For MongoDB operations
STORAGE_RETRY = RetryConfig(
    max_retries=5,
    base_delay=1.0,
    max_delay=60.0,
    exponential_base=2.0,
    retryable_exceptions=(ConnectionError, TimeoutError, OSError),
)

# For file operations
FILE_RETRY = RetryConfig(
    max_retries=3,
    base_delay=0.5,
    max_delay=5.0,
    exponential_base=2.0,
    retryable_exceptions=(IOError, OSError, PermissionError),
)

# For HTTP requests
HTTP_RETRY = RetryConfig(
    max_retries=3,
    base_delay=1.0,
    max_delay=30.0,
    exponential_base=2.0,
    retryable_exceptions=(ConnectionError, TimeoutError, OSError),
)


# ============================================================================
# Utility Functions
# ============================================================================

async def retry_async(
    func: Callable[..., Any],
    *args,
    config: Optional[RetryConfig] = None,
    **kwargs
) -> Any:
    """
    Execute an async function with retry logic.

    Usage:
        result = await retry_async(
            some_async_func,
            arg1, arg2,
            config=EMBEDDING_RETRY,
            kwarg1=value1
        )
    """
    config = config or RetryConfig()
    decorated = with_retry(config)(func)
    return await decorated(*args, **kwargs)


def retry_sync(
    func: Callable[..., Any],
    *args,
    config: Optional[RetryConfig] = None,
    **kwargs
) -> Any:
    """
    Execute a synchronous function with retry logic.

    Usage:
        result = retry_sync(
            some_func,
            arg1, arg2,
            config=FILE_RETRY,
            kwarg1=value1
        )
    """
    config = config or RetryConfig()
    decorated = with_retry_sync(config)(func)
    return decorated(*args, **kwargs)


class RetryContext:
    """
    Context manager for retry operations with manual control.

    Usage:
        async with RetryContext(EMBEDDING_RETRY) as ctx:
            while ctx.should_retry():
                try:
                    result = await some_operation()
                    ctx.success()
                    break
                except Exception as e:
                    await ctx.handle_error(e)
    """

    def __init__(self, config: Optional[RetryConfig] = None):
        self.config = config or RetryConfig()
        self.attempt = 0
        self.last_error: Optional[Exception] = None
        self._succeeded = False

    def should_retry(self) -> bool:
        """Check if we should attempt another retry."""
        return not self._succeeded and self.attempt <= self.config.max_retries

    def success(self):
        """Mark operation as successful."""
        self._succeeded = True

    async def handle_error(self, error: Exception):
        """Handle an error and prepare for retry if applicable."""
        self.last_error = error

        if not isinstance(error, self.config.retryable_exceptions):
            raise error

        if self.attempt >= self.config.max_retries:
            logger.error(f"Max retries ({self.config.max_retries}) exceeded: {error}")
            raise error

        delay = self.config.calculate_delay(self.attempt)
        logger.warning(
            f"Attempt {self.attempt + 1} failed, retrying in {delay:.2f}s: {error}"
        )
        await asyncio.sleep(delay)
        self.attempt += 1

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        # Don't suppress exceptions
        return False
