"""
LLM Service - Async wrapper for llama.cpp API with streaming support.

This service provides:
- Async text generation via llama.cpp
- Server-Sent Events (SSE) streaming for real-time responses
- Request deduplication and caching
- Configurable model, temperature, and timeout
- Retry logic for transient failures

Design Rationale:
-----------------
The LLMService is designed as an async singleton to efficiently manage connections
to the llama.cpp API. By using aiohttp for HTTP operations, we achieve non-blocking
I/O that scales well under concurrent load. The caching layer prevents redundant
API calls for identical prompts within a configurable TTL window.

Architecture:
- Singleton pattern ensures shared connection pool across requests
- Async generators enable memory-efficient streaming of large responses
- LRU-style cache with TTL for deduplication without unbounded memory growth
"""

import asyncio
import hashlib
import json
import logging
import os
import re
import time
from core.log_utils import log_info
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple
from collections import OrderedDict

import aiohttp
from aiohttp import ClientTimeout

logger = logging.getLogger(__name__)


def parse_prometheus_metrics(metrics_text: str) -> Dict[str, Any]:
    """
    Parse Prometheus metrics format text from llama.cpp /metrics endpoint.

    Returns a dictionary with key performance metrics.
    """
    result = {
        "tps": 0.0,
        "requests": 0,
        "tokensProcessed": 0,
        "promptTokensTotal": 0,
        "generatedTokensTotal": 0,
        "requestsProcessing": 0,
        "slotsIdle": 0,
        "slotsProcessing": 0,
        "context": 0,
        "kvCacheTokens": 0,
        "kvCacheUsed": 0,
    }

    lines = metrics_text.split('\n')
    for line in lines:
        if line.startswith('#') or not line.strip():
            continue

        # llamacpp_tokens_second - tokens per second (TPS)
        if line.startswith('llamacpp_tokens_second'):
            match = re.search(r'llamacpp_tokens_second\s+([\d.]+)', line)
            if match:
                result["tps"] = float(match.group(1))

        # llamacpp_requests_processing - active requests being processed
        elif line.startswith('llamacpp_requests_processing'):
            match = re.search(r'llamacpp_requests_processing\s+(\d+)', line)
            if match:
                result["requestsProcessing"] = int(match.group(1))

        # llamacpp_prompt_tokens_total - total prompt tokens processed
        elif line.startswith('llamacpp_prompt_tokens_total'):
            match = re.search(r'llamacpp_prompt_tokens_total\s+(\d+)', line)
            if match:
                result["promptTokensTotal"] = int(match.group(1))

        # llamacpp_tokens_predicted_total - total tokens generated
        elif line.startswith('llamacpp_tokens_predicted_total'):
            match = re.search(r'llamacpp_tokens_predicted_total\s+(\d+)', line)
            if match:
                result["generatedTokensTotal"] = int(match.group(1))

        # llamacpp_kv_cache_tokens - current KV cache token count
        elif line.startswith('llamacpp_kv_cache_tokens'):
            match = re.search(r'llamacpp_kv_cache_tokens\s+(\d+)', line)
            if match:
                result["kvCacheTokens"] = int(match.group(1))

        # llamacpp_kv_cache_used_cells - KV cache cells in use
        elif line.startswith('llamacpp_kv_cache_used_cells'):
            match = re.search(r'llamacpp_kv_cache_used_cells\s+(\d+)', line)
            if match:
                result["kvCacheUsed"] = int(match.group(1))

        # llamacpp_requests_total - total requests handled
        elif line.startswith('llamacpp_requests_total'):
            match = re.search(r'llamacpp_requests_total\s+(\d+)', line)
            if match:
                result["requests"] = int(match.group(1))

        # llamacpp_slots_idle - idle inference slots
        elif line.startswith('llamacpp_slots_idle'):
            match = re.search(r'llamacpp_slots_idle\s+(\d+)', line)
            if match:
                result["slotsIdle"] = int(match.group(1))

        # llamacpp_slots_processing - active inference slots
        elif line.startswith('llamacpp_slots_processing'):
            match = re.search(r'llamacpp_slots_processing\s+(\d+)', line)
            if match:
                result["slotsProcessing"] = int(match.group(1))

        # llamacpp_n_ctx - context window size
        elif line.startswith('llamacpp_n_ctx'):
            match = re.search(r'llamacpp_n_ctx\s+(\d+)', line)
            if match:
                result["context"] = int(match.group(1))

    # Calculate total tokens processed
    result["tokensProcessed"] = result["promptTokensTotal"] + result["generatedTokensTotal"]

    return result


@dataclass
class LLMConfig:
    """
    Configuration for LLM service with multi-endpoint support.

    Multi-Model Architecture:
    -------------------------
    For multi-user websites, running separate llama.cpp instances per model type
    provides better concurrency and allows specialized models to run in parallel.

    When use_dedicated_endpoints=True (default), each model type routes to its
    own llama.cpp server instance. This enables:
        - Parallel inference across model types
        - No model swapping overhead
        - Independent scaling per workload

    When use_dedicated_endpoints=False, all requests go to 'host' and rely on
    the model parameter (legacy single-instance mode).
    """
    # Primary endpoints (multi-instance architecture)
    host: str = "http://localhost:8081"           # General model (chat, summarization)
    sql_host: str = "http://localhost:8081"       # SQL model (sqlcoder2 on 8081 for testing)
    code_host: str = "http://localhost:8082"      # Code model (optional)

    # Model names (used for logging and single-instance fallback)
    model: str = "qwen2.5-7b-instruct"
    sql_model: str = "sqlcoder2"
    code_model: str = "qwen2.5-coder"

    # Generation parameters
    temperature: float = 0.0
    timeout: int = 120  # seconds
    max_retries: int = 3
    retry_delay: float = 1.0
    cache_ttl: int = 300  # seconds
    cache_max_size: int = 100
    context_window: int = 8192
    max_tokens: int = 512  # Max tokens for completion response

    # Feature flags
    use_dedicated_endpoints: bool = True  # Use separate endpoints per model type


@dataclass
class GenerationResult:
    """Result from LLM generation."""
    success: bool
    response: str = ""
    error: Optional[str] = None
    token_usage: Dict[str, int] = field(default_factory=lambda: {
        "prompt_tokens": 0,
        "response_tokens": 0,
        "total_tokens": 0
    })
    model: str = ""
    generation_time_ms: int = 0
    cached: bool = False


@dataclass
class StreamChunk:
    """A chunk of streamed response."""
    content: str
    done: bool = False
    error: Optional[str] = None
    token_usage: Optional[Dict[str, int]] = None


class LLMMetricsTracker:
    """
    Tracks LLM performance metrics internally.

    Since llama-cpp-python doesn't expose Prometheus metrics, we track
    metrics based on API responses flowing through our service.

    Metrics tracked per endpoint:
    - Total requests
    - Total tokens (prompt + generated)
    - Tokens per second (rolling average)
    - Active requests (concurrent)
    - Last update timestamp
    """

    def __init__(self, window_size: int = 60):
        """
        Initialize metrics tracker.

        Args:
            window_size: Number of seconds for rolling TPS calculation
        """
        self._lock = asyncio.Lock()
        self._window_size = window_size

        # Per-endpoint metrics: {endpoint_name: {...}}
        self._metrics: Dict[str, Dict[str, Any]] = {
            "sql": self._create_empty_metrics(),
            "general": self._create_empty_metrics(),
            "code": self._create_empty_metrics(),
        }

        # Rolling window for TPS calculation: {endpoint: [(timestamp, tokens), ...]}
        self._token_windows: Dict[str, List[Tuple[float, int]]] = {
            "sql": [],
            "general": [],
            "code": [],
        }

    def _create_empty_metrics(self) -> Dict[str, Any]:
        """Create empty metrics structure for an endpoint."""
        return {
            "requests": 0,
            "promptTokensTotal": 0,
            "generatedTokensTotal": 0,
            "tokensProcessed": 0,
            "requestsProcessing": 0,
            "tps": 0.0,
            "lastUpdate": None,
        }

    def _get_endpoint_name(self, use_sql_model: bool, use_code_model: bool) -> str:
        """Determine endpoint name based on model type."""
        if use_sql_model:
            return "sql"
        elif use_code_model:
            return "code"
        else:
            return "general"

    async def record_request_start(self, use_sql_model: bool = False, use_code_model: bool = False):
        """Record that a request has started (increment active count)."""
        endpoint = self._get_endpoint_name(use_sql_model, use_code_model)
        async with self._lock:
            self._metrics[endpoint]["requestsProcessing"] += 1

    async def record_request_complete(
        self,
        use_sql_model: bool = False,
        use_code_model: bool = False,
        prompt_tokens: int = 0,
        generated_tokens: int = 0,
        generation_time_ms: int = 0,
    ):
        """
        Record a completed request with token counts.

        Args:
            use_sql_model: Whether this was a SQL model request
            use_code_model: Whether this was a code model request
            prompt_tokens: Number of prompt tokens
            generated_tokens: Number of generated tokens
            generation_time_ms: Time taken for generation in milliseconds
        """
        endpoint = self._get_endpoint_name(use_sql_model, use_code_model)
        now = time.time()
        total_tokens = prompt_tokens + generated_tokens

        async with self._lock:
            metrics = self._metrics[endpoint]
            metrics["requests"] += 1
            metrics["promptTokensTotal"] += prompt_tokens
            metrics["generatedTokensTotal"] += generated_tokens
            metrics["tokensProcessed"] += total_tokens
            metrics["requestsProcessing"] = max(0, metrics["requestsProcessing"] - 1)
            metrics["lastUpdate"] = now

            # Add to rolling window for TPS calculation
            self._token_windows[endpoint].append((now, generated_tokens))

            # Clean old entries from window
            cutoff = now - self._window_size
            self._token_windows[endpoint] = [
                (t, tokens) for t, tokens in self._token_windows[endpoint] if t > cutoff
            ]

            # Calculate TPS from window
            window = self._token_windows[endpoint]
            if len(window) >= 2:
                time_span = window[-1][0] - window[0][0]
                if time_span > 0:
                    total_window_tokens = sum(tokens for _, tokens in window)
                    metrics["tps"] = round(total_window_tokens / time_span, 2)
            elif generation_time_ms > 0 and generated_tokens > 0:
                # Single request TPS
                metrics["tps"] = round(generated_tokens / (generation_time_ms / 1000), 2)

    async def get_metrics(self, endpoint: Optional[str] = None) -> Dict[str, Any]:
        """
        Get current metrics for one or all endpoints.

        Args:
            endpoint: Specific endpoint name, or None for all

        Returns:
            Metrics dictionary
        """
        async with self._lock:
            if endpoint:
                return self._metrics.get(endpoint, self._create_empty_metrics()).copy()
            return {k: v.copy() for k, v in self._metrics.items()}

    async def reset_metrics(self, endpoint: Optional[str] = None):
        """Reset metrics for one or all endpoints."""
        async with self._lock:
            if endpoint:
                self._metrics[endpoint] = self._create_empty_metrics()
                self._token_windows[endpoint] = []
            else:
                for ep in self._metrics:
                    self._metrics[ep] = self._create_empty_metrics()
                    self._token_windows[ep] = []


# Global metrics tracker instance
_metrics_tracker: Optional[LLMMetricsTracker] = None


def get_metrics_tracker() -> LLMMetricsTracker:
    """Get or create the global metrics tracker."""
    global _metrics_tracker
    if _metrics_tracker is None:
        _metrics_tracker = LLMMetricsTracker()
    return _metrics_tracker


class LLMCache:
    """
    Simple LRU cache with TTL for LLM responses.

    Design: Uses OrderedDict for O(1) access and LRU eviction.
    TTL ensures stale responses don't persist indefinitely.
    """

    def __init__(self, max_size: int = 100, ttl: int = 300):
        self.max_size = max_size
        self.ttl = ttl
        self._cache: OrderedDict[str, Tuple[GenerationResult, float]] = OrderedDict()
        self._lock = asyncio.Lock()

    def _make_key(self, prompt: str, system: str, model: str, temperature: float) -> str:
        """Create a unique cache key from request parameters."""
        key_data = f"{prompt}|{system}|{model}|{temperature}"
        return hashlib.sha256(key_data.encode()).hexdigest()

    async def get(self, prompt: str, system: str, model: str, temperature: float) -> Optional[GenerationResult]:
        """Get cached result if valid."""
        key = self._make_key(prompt, system, model, temperature)
        async with self._lock:
            if key not in self._cache:
                return None

            result, timestamp = self._cache[key]

            # Check TTL
            if time.time() - timestamp > self.ttl:
                del self._cache[key]
                return None

            # Move to end (most recently used)
            self._cache.move_to_end(key)

            # Mark as cached
            result.cached = True
            return result

    async def set(self, prompt: str, system: str, model: str, temperature: float, result: GenerationResult):
        """Cache a result."""
        key = self._make_key(prompt, system, model, temperature)
        async with self._lock:
            # Evict oldest if at capacity
            while len(self._cache) >= self.max_size:
                self._cache.popitem(last=False)

            self._cache[key] = (result, time.time())

    async def clear(self):
        """Clear all cached entries."""
        async with self._lock:
            self._cache.clear()

    @property
    def size(self) -> int:
        """Current cache size."""
        return len(self._cache)


class LLMService:
    """
    Async service for LLM operations via llama.cpp API.

    Features:
    - Non-blocking async operations
    - SSE streaming support
    - Request caching with deduplication
    - Automatic retries with exponential backoff
    - Configurable models and parameters

    Usage:
        service = await get_llm_service()
        result = await service.generate(prompt="Hello", system="Be helpful")

        # Or with streaming
        async for chunk in service.generate_stream(prompt="Hello"):
            print(chunk.content, end="", flush=True)
    """

    _instance: Optional["LLMService"] = None

    def __init__(self, config: Optional[LLMConfig] = None):
        self.config = config or self._load_config()
        self._session: Optional[aiohttp.ClientSession] = None
        self._cache = LLMCache(
            max_size=self.config.cache_max_size,
            ttl=self.config.cache_ttl
        )
        self._initialized = False
        self._pending_requests: Dict[str, asyncio.Event] = {}
        self._pending_results: Dict[str, GenerationResult] = {}

    @classmethod
    async def get_instance(cls) -> "LLMService":
        """Get or create singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
            await cls._instance.initialize()
        return cls._instance

    def _load_config(self) -> LLMConfig:
        """Load configuration from environment variables."""
        return LLMConfig(
            # Multi-endpoint configuration
            host=os.getenv("LLAMACPP_HOST", "http://localhost:8081"),
            sql_host=os.getenv("LLAMACPP_SQL_HOST", "http://localhost:8081"),
            code_host=os.getenv("LLAMACPP_CODE_HOST", "http://localhost:8082"),

            # Model names (for logging/fallback)
            model=os.getenv("LLAMACPP_MODEL", "qwen2.5-7b-instruct"),
            sql_model=os.getenv("LLAMACPP_SQL_MODEL", "sqlcoder2"),
            code_model=os.getenv("LLAMACPP_CODE_MODEL", "qwen2.5-coder"),

            # Generation parameters
            temperature=float(os.getenv("LLM_TEMPERATURE", "0.0")),
            timeout=int(os.getenv("LLM_TIMEOUT", "120")),
            max_retries=int(os.getenv("LLM_MAX_RETRIES", "3")),
            cache_ttl=int(os.getenv("LLM_CACHE_TTL", "300")),
            context_window=int(os.getenv("LLM_CONTEXT_WINDOW", "8192")),

            # Feature flags
            use_dedicated_endpoints=os.getenv("LLAMACPP_USE_DEDICATED_ENDPOINTS", "true").lower() == "true",
        )

    async def initialize(self):
        """Initialize the service and HTTP session."""
        if self._initialized:
            return

        timeout = ClientTimeout(total=self.config.timeout)
        self._session = aiohttp.ClientSession(timeout=timeout)
        self._initialized = True
        log_info("LLM Service", f"Initialized with host={self.config.host}, model={self.config.model}")

    async def close(self):
        """Close the HTTP session and cleanup resources."""
        if self._session and not self._session.closed:
            await self._session.close()
            # Give the event loop time to finalize the connection cleanup
            # This prevents "Unclosed client session" warnings
            await asyncio.sleep(0.25)
            self._session = None
        self._initialized = False
        logger.info("LLMService closed")

    async def fetch_metrics(self, endpoint: str) -> Optional[Dict[str, Any]]:
        """
        Fetch and parse Prometheus metrics from a llama.cpp server.

        Args:
            endpoint: Base URL of the llama.cpp server (e.g., "http://localhost:8080")

        Returns:
            Dictionary of parsed metrics, or None if fetch fails
        """
        try:
            async with self._session.get(f"{endpoint}/metrics", timeout=3) as response:
                if response.status == 200:
                    metrics_text = await response.text()
                    return parse_prometheus_metrics(metrics_text)
                else:
                    logger.debug(f"Metrics endpoint returned {response.status} for {endpoint}")
                    return None
        except Exception as e:
            logger.debug(f"Failed to fetch metrics from {endpoint}: {e}")
            return None

    async def health_check(self) -> Dict[str, Any]:
        """
        Check if llama.cpp services are available and collect performance metrics.

        Returns comprehensive health and metrics data for each llama.cpp endpoint,
        including TPS, token counts, slot utilization, and context info.

        Metrics are tracked internally since llama-cpp-python doesn't expose
        a Prometheus /metrics endpoint.
        """
        results = {
            "healthy": True,
            "use_dedicated_endpoints": self.config.use_dedicated_endpoints,
            "endpoints": {}
        }

        endpoints_to_check = [
            ("general", self.config.host),
            ("sql", self.config.sql_host),
        ]

        # Only check code endpoint if it's different from general
        if self.config.code_host != self.config.host:
            endpoints_to_check.append(("code", self.config.code_host))

        # Get internal metrics from our tracker
        tracker = get_metrics_tracker()
        all_metrics = await tracker.get_metrics()

        for name, endpoint in endpoints_to_check:
            try:
                async with self._session.get(f"{endpoint}/v1/models", timeout=5) as response:
                    if response.status == 200:
                        data = await response.json()
                        models = [m.get("id") for m in data.get("data", [])]

                        # Use internal metrics tracker (llama-cpp-python doesn't have /metrics)
                        metrics = all_metrics.get(name)

                        results["endpoints"][name] = {
                            "healthy": True,
                            "url": endpoint,
                            "models": models[:5],
                            "metrics": metrics,
                        }
                    else:
                        results["endpoints"][name] = {
                            "healthy": False,
                            "url": endpoint,
                            "error": f"HTTP {response.status}",
                            "metrics": None,
                        }
                        results["healthy"] = False
            except Exception as e:
                results["endpoints"][name] = {
                    "healthy": False,
                    "url": endpoint,
                    "error": str(e),
                    "metrics": None,
                }
                results["healthy"] = False

        return results

    def _make_dedup_key(self, prompt: str, system: str, model: str) -> str:
        """Create key for request deduplication."""
        key_data = f"{prompt}|{system}|{model}"
        return hashlib.sha256(key_data.encode()).hexdigest()[:16]

    def _get_endpoint(self, use_sql_model: bool = False, use_code_model: bool = False) -> str:
        """
        Get the appropriate endpoint based on model type.

        In multi-instance mode (use_dedicated_endpoints=True), routes to:
            - sql_host for SQL models
            - code_host for code models
            - host for general models

        In single-instance mode, always returns host.
        """
        if not self.config.use_dedicated_endpoints:
            return self.config.host

        if use_sql_model:
            return self.config.sql_host
        elif use_code_model:
            return self.config.code_host
        else:
            return self.config.host

    async def get_generation_progress(self, use_sql_model: bool = False, use_code_model: bool = False) -> dict:
        """
        Get current generation progress from llama.cpp /slots endpoint.

        Returns:
            dict with:
                - is_processing: bool
                - n_decoded: int (tokens generated so far)
                - max_tokens: int (maximum tokens to generate)
                - progress_pct: float (0-100)
        """
        endpoint = self._get_endpoint(use_sql_model, use_code_model)
        try:
            async with self._session.get(f"{endpoint}/slots", timeout=ClientTimeout(total=2)) as response:
                if response.status == 200:
                    slots = await response.json()
                    if slots and len(slots) > 0:
                        slot = slots[0]
                        is_processing = slot.get("is_processing", False)
                        max_tokens = slot.get("params", {}).get("max_tokens", 512)
                        next_token = slot.get("next_token", [{}])
                        n_decoded = next_token[0].get("n_decoded", 0) if next_token else 0
                        progress_pct = (n_decoded / max_tokens * 100) if max_tokens > 0 else 0
                        return {
                            "is_processing": is_processing,
                            "n_decoded": n_decoded,
                            "max_tokens": max_tokens,
                            "progress_pct": min(progress_pct, 100),
                        }
        except Exception as e:
            logger.debug(f"Failed to get generation progress: {e}")

        return {"is_processing": False, "n_decoded": 0, "max_tokens": 0, "progress_pct": 0}

    async def generate(
        self,
        prompt: str,
        system: str = "",
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        use_cache: bool = True,
        use_sql_model: bool = False,
        use_code_model: bool = False,
        stop_sequences: Optional[List[str]] = None,
    ) -> GenerationResult:
        """
        Generate text using llama.cpp.

        Args:
            prompt: User prompt
            system: System prompt for context
            model: Override model (uses configured model if None)
            temperature: Override temperature
            max_tokens: Override max tokens for completion (default: 512)
            use_cache: Whether to use response cache
            use_sql_model: Use the dedicated SQL model
            use_code_model: Use the dedicated code model
            stop_sequences: List of strings that stop generation when encountered

        Returns:
            GenerationResult with response or error

        Design Notes:
        - Request deduplication prevents duplicate API calls for identical requests
        - Caching reduces latency for repeated queries
        - Retry logic handles transient LLM failures
        """
        if not self._initialized:
            await self.initialize()

        # Determine endpoint based on model type
        if self.config.use_dedicated_endpoints:
            if use_sql_model:
                endpoint = self.config.sql_host
                resolved_model = self.config.sql_model
            elif use_code_model:
                endpoint = self.config.code_host
                resolved_model = self.config.code_model
            else:
                endpoint = self.config.host
                resolved_model = model or self.config.model
        else:
            # Legacy single-endpoint mode
            endpoint = self.config.host
            resolved_model = model or (self.config.sql_model if use_sql_model else self.config.model)

        resolved_temp = temperature if temperature is not None else self.config.temperature
        resolved_max_tokens = max_tokens or self.config.max_tokens

        # Check cache first
        if use_cache:
            cached = await self._cache.get(prompt, system, resolved_model, resolved_temp)
            if cached:
                logger.debug(f"Cache hit for prompt: {prompt[:50]}...")
                return cached

        # Request deduplication - if same request is in flight, wait for it
        dedup_key = self._make_dedup_key(prompt, system, resolved_model)
        if dedup_key in self._pending_requests:
            logger.debug(f"Waiting for duplicate request: {dedup_key}")
            await self._pending_requests[dedup_key].wait()
            if dedup_key in self._pending_results:
                return self._pending_results[dedup_key]

        # Mark request as in-flight
        self._pending_requests[dedup_key] = asyncio.Event()

        start_time = time.time()
        result = GenerationResult(success=False, model=resolved_model)

        try:
            # Combine system and prompt for OpenAI-compatible endpoint
            full_prompt = f"{system}\n\n{prompt}" if system else prompt
            payload = {
                "prompt": full_prompt,
                "max_tokens": resolved_max_tokens,
                "temperature": resolved_temp,
            }

            # Add stop sequences to prevent runaway generation
            if stop_sequences:
                payload["stop"] = stop_sequences

            logger.debug(f"Using endpoint: {endpoint} (sql_model={use_sql_model}, code_model={use_code_model})")

            # Retry loop with exponential backoff
            last_error = None
            for attempt in range(self.config.max_retries):
                try:
                    async with self._session.post(
                        f"{endpoint}/v1/completions",
                        json=payload,
                    ) as response:
                        if response.status == 200:
                            data = await response.json()

                            result.success = True
                            # OpenAI format: choices[0].text
                            result.response = data.get("choices", [{}])[0].get("text", "").strip()
                            usage = data.get("usage", {})
                            result.token_usage = {
                                "prompt_tokens": usage.get("prompt_tokens", 0),
                                "response_tokens": usage.get("completion_tokens", 0),
                                "total_tokens": usage.get("total_tokens", 0),
                            }
                            result.generation_time_ms = int((time.time() - start_time) * 1000)

                            # Cache successful result
                            if use_cache:
                                await self._cache.set(prompt, system, resolved_model, resolved_temp, result)

                            break
                        else:
                            last_error = f"HTTP {response.status}: {await response.text()}"
                            logger.warning(f"LLM request failed (attempt {attempt + 1}): {last_error}")

                except asyncio.TimeoutError:
                    last_error = "Request timeout"
                    logger.warning(f"LLM timeout (attempt {attempt + 1})")
                except aiohttp.ClientError as e:
                    last_error = str(e)
                    logger.warning(f"LLM client error (attempt {attempt + 1}): {e}")

                # Wait before retry (exponential backoff)
                if attempt < self.config.max_retries - 1:
                    await asyncio.sleep(self.config.retry_delay * (2 ** attempt))

            if not result.success:
                result.error = last_error or "Unknown error"
                logger.error(f"LLM generation failed after {self.config.max_retries} attempts: {result.error}")

        finally:
            # Complete deduplication
            self._pending_results[dedup_key] = result
            self._pending_requests[dedup_key].set()

            # Record metrics for monitoring
            tracker = get_metrics_tracker()
            await tracker.record_request_complete(
                use_sql_model=use_sql_model,
                use_code_model=use_code_model,
                prompt_tokens=result.token_usage.get("prompt_tokens", 0),
                generated_tokens=result.token_usage.get("response_tokens", 0),
                generation_time_ms=result.generation_time_ms,
            )

            # Cleanup after short delay
            async def cleanup():
                await asyncio.sleep(1.0)
                self._pending_requests.pop(dedup_key, None)
                self._pending_results.pop(dedup_key, None)

            asyncio.create_task(cleanup())

        return result

    async def generate_stream(
        self,
        prompt: str,
        system: str = "",
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        use_sql_model: bool = False,
        use_code_model: bool = False,
        stop_sequences: Optional[List[str]] = None,
    ) -> AsyncGenerator[StreamChunk, None]:
        """
        Generate text with streaming response.

        Yields StreamChunk objects as they arrive from llama.cpp.
        This is ideal for Server-Sent Events (SSE) endpoints.

        Args:
            prompt: User prompt
            system: System prompt
            model: Override model
            temperature: Override temperature
            max_tokens: Override max tokens for completion
            use_sql_model: Use the dedicated SQL model
            use_code_model: Use the dedicated code model
            stop_sequences: List of strings that stop generation when encountered

        Yields:
            StreamChunk with partial response content

        Design Notes:
        - Uses async generators for memory-efficient streaming
        - Yields chunks immediately as they arrive
        - Final chunk has done=True with token usage stats
        """
        if not self._initialized:
            await self.initialize()

        # Determine endpoint based on model type
        if self.config.use_dedicated_endpoints:
            if use_sql_model:
                endpoint = self.config.sql_host
                resolved_model = self.config.sql_model
            elif use_code_model:
                endpoint = self.config.code_host
                resolved_model = self.config.code_model
            else:
                endpoint = self.config.host
                resolved_model = model or self.config.model
        else:
            # Legacy single-endpoint mode
            endpoint = self.config.host
            resolved_model = model or (self.config.sql_model if use_sql_model else self.config.model)

        resolved_temp = temperature if temperature is not None else self.config.temperature
        resolved_max_tokens = max_tokens or self.config.max_tokens

        # Combine system and prompt for OpenAI-compatible endpoint
        full_prompt = f"{system}\n\n{prompt}" if system else prompt
        payload = {
            "prompt": full_prompt,
            "max_tokens": resolved_max_tokens,
            "temperature": resolved_temp,
            "stream": True,
        }

        # Add stop sequences to prevent runaway generation
        if stop_sequences:
            payload["stop"] = stop_sequences

        start_time = time.time()
        tracker = get_metrics_tracker()
        total_tokens = 0
        prompt_tokens = 0

        try:
            async with self._session.post(
                f"{endpoint}/v1/completions",
                json=payload,
            ) as response:
                if response.status != 200:
                    yield StreamChunk(
                        content="",
                        done=True,
                        error=f"HTTP {response.status}: {await response.text()}"
                    )
                    return

                # Stream response - OpenAI SSE format
                async for line in response.content:
                    if not line:
                        continue

                    line_str = line.decode("utf-8").strip()
                    if not line_str.startswith("data:"):
                        continue

                    data_str = line_str[5:].strip()
                    if data_str == "[DONE]":
                        # Record metrics on stream completion
                        generation_time_ms = int((time.time() - start_time) * 1000)
                        await tracker.record_request_complete(
                            use_sql_model=use_sql_model,
                            use_code_model=use_code_model,
                            prompt_tokens=prompt_tokens,
                            generated_tokens=total_tokens,
                            generation_time_ms=generation_time_ms,
                        )
                        yield StreamChunk(
                            content="",
                            done=True,
                            token_usage={
                                "prompt_tokens": prompt_tokens,
                                "response_tokens": total_tokens,
                                "total_tokens": prompt_tokens + total_tokens,
                            }
                        )
                        return

                    try:
                        data = json.loads(data_str)
                        choices = data.get("choices", [])
                        if choices:
                            text = choices[0].get("text", "")
                            finish_reason = choices[0].get("finish_reason")
                            total_tokens += 1

                            if finish_reason:
                                usage = data.get("usage", {})
                                prompt_tokens = usage.get("prompt_tokens", 0)
                                # Record metrics on stream completion
                                generation_time_ms = int((time.time() - start_time) * 1000)
                                await tracker.record_request_complete(
                                    use_sql_model=use_sql_model,
                                    use_code_model=use_code_model,
                                    prompt_tokens=prompt_tokens,
                                    generated_tokens=usage.get("completion_tokens", total_tokens),
                                    generation_time_ms=generation_time_ms,
                                )
                                yield StreamChunk(
                                    content=text,
                                    done=True,
                                    token_usage={
                                        "prompt_tokens": prompt_tokens,
                                        "response_tokens": usage.get("completion_tokens", total_tokens),
                                        "total_tokens": usage.get("total_tokens", total_tokens),
                                    }
                                )
                            else:
                                yield StreamChunk(content=text, done=False)
                    except json.JSONDecodeError:
                        continue

        except asyncio.TimeoutError:
            yield StreamChunk(content="", done=True, error="Request timeout")
        except aiohttp.ClientError as e:
            yield StreamChunk(content="", done=True, error=str(e))
        except Exception as e:
            logger.error(f"Streaming error: {e}")
            yield StreamChunk(content="", done=True, error=str(e))

    async def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            "cache_size": self._cache.size,
            "cache_max_size": self.config.cache_max_size,
            "cache_ttl": self.config.cache_ttl,
        }

    async def clear_cache(self):
        """Clear the response cache."""
        await self._cache.clear()
        logger.info("LLM cache cleared")


# Singleton accessor
_llm_service: Optional[LLMService] = None


async def get_llm_service() -> LLMService:
    """Get or create the global LLM service instance."""
    global _llm_service
    if _llm_service is None:
        _llm_service = LLMService()
        await _llm_service.initialize()
    return _llm_service


async def close_llm_service():
    """Close the global LLM service."""
    global _llm_service
    if _llm_service:
        await _llm_service.close()
        _llm_service = None
