"""
LLM Endpoint configuration and health checking.
"""
from typing import Dict
import httpx

from .models import LLMEndpoint, Pipeline


# Pipeline to endpoint mapping
PIPELINE_ENDPOINTS: Dict[Pipeline, LLMEndpoint] = {
    Pipeline.SQL: LLMEndpoint.SQL,
    Pipeline.AUDIO: LLMEndpoint.GENERAL,
    Pipeline.QUERY: LLMEndpoint.GENERAL,
    Pipeline.GIT: LLMEndpoint.CODE,
    Pipeline.CODE_FLOW: LLMEndpoint.CODE,
    Pipeline.CODE_ASSISTANCE: LLMEndpoint.CODE,
    Pipeline.DOCUMENT_AGENT: LLMEndpoint.GENERAL,
}

# Model names by endpoint (update based on your actual models)
ENDPOINT_MODELS: Dict[LLMEndpoint, str] = {
    LLMEndpoint.SQL: "qwen2.5-7b-sql",
    LLMEndpoint.GENERAL: "qwen2.5-7b-general",
    LLMEndpoint.CODE: "qwen2.5-7b-code",
}


def get_endpoint_for_pipeline(pipeline: Pipeline) -> str:
    """Get the LLM endpoint URL for a pipeline."""
    return PIPELINE_ENDPOINTS[pipeline].value


def get_model_for_endpoint(endpoint: str) -> str:
    """Get the model name for an endpoint."""
    for ep, model in ENDPOINT_MODELS.items():
        if ep.value == endpoint:
            return model
    return "unknown"


async def check_endpoint_health(endpoint: str, timeout: float = 5.0) -> bool:
    """Check if an LLM endpoint is healthy."""
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            # llama.cpp health endpoint
            response = await client.get(f"{endpoint}/health")
            return response.status_code == 200
    except Exception:
        return False


async def get_all_endpoint_health() -> Dict[str, bool]:
    """Check health of all endpoints."""
    health = {}
    for endpoint in LLMEndpoint:
        health[endpoint.value] = await check_endpoint_health(endpoint.value)
    return health
