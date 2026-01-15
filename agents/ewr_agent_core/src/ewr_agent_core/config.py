"""
EWR Agent Core Configuration
============================

Configuration management for agents with support for:
- Environment variables
- YAML/JSON config files
- Programmatic configuration
"""

import os
from pathlib import Path
from typing import Optional, Dict, Any, Union
from pydantic import BaseModel, Field
import yaml
import json

from .models import LLMConfig, AgentType


class AgentConfig(BaseModel):
    """
    Complete configuration for an agent.

    Configuration is loaded from (in order of precedence):
    1. Programmatic values passed to constructor
    2. Environment variables (EWR_AGENT_*)
    3. Config file (~/.ewr-agents/config.yaml or specified path)
    4. Default values
    """

    # Agent identity
    agent_type: AgentType = AgentType.CUSTOM
    name: str = "Agent"
    version: str = "1.0.0"
    description: str = ""

    # LLM configuration
    llm_backend: str = Field(default="llamacpp", description="LLM backend: llamacpp, openai, anthropic")
    llm_model: str = Field(default="qwen2.5-coder:1.5b", description="Model name")
    llm_base_url: Optional[str] = Field(default=None, description="Base URL for LLM API")
    llm_api_key: Optional[str] = Field(default=None, description="API key for LLM")
    llm_temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    llm_max_tokens: int = Field(default=4096, ge=1)
    llm_timeout: int = Field(default=60, ge=1)

    # Registry configuration
    registry_enabled: bool = Field(default=True, description="Register with agent registry")
    registry_url: Optional[str] = Field(default=None, description="URL of remote registry")

    # Communication
    broker_enabled: bool = Field(default=True, description="Enable message broker")
    heartbeat_interval: int = Field(default=10, ge=1, description="Heartbeat interval in seconds")

    # Execution
    max_concurrent_tasks: int = Field(default=1, ge=1)
    task_timeout: int = Field(default=300, ge=1)

    # Logging
    log_level: str = Field(default="INFO")
    log_file: Optional[str] = None

    # Custom metadata
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @property
    def llm_config(self) -> LLMConfig:
        """Get LLM configuration as LLMConfig object."""
        return LLMConfig(
            backend=self.llm_backend,
            model=self.llm_model,
            base_url=self.llm_base_url,
            api_key=self.llm_api_key,
            temperature=self.llm_temperature,
            max_tokens=self.llm_max_tokens,
            timeout_seconds=self.llm_timeout,
        )

    @classmethod
    def from_env(cls, prefix: str = "EWR_AGENT") -> "AgentConfig":
        """
        Load configuration from environment variables.

        Environment variables should be prefixed with EWR_AGENT_ (or custom prefix).
        Example: EWR_AGENT_LLM_BACKEND=openai
        """
        env_mapping = {
            "agent_type": f"{prefix}_TYPE",
            "name": f"{prefix}_NAME",
            "llm_backend": f"{prefix}_LLM_BACKEND",
            "llm_model": f"{prefix}_LLM_MODEL",
            "llm_base_url": f"{prefix}_LLM_BASE_URL",
            "llm_api_key": f"{prefix}_LLM_API_KEY",
            "llm_temperature": f"{prefix}_LLM_TEMPERATURE",
            "llm_max_tokens": f"{prefix}_LLM_MAX_TOKENS",
            "llm_timeout": f"{prefix}_LLM_TIMEOUT",
            "registry_enabled": f"{prefix}_REGISTRY_ENABLED",
            "registry_url": f"{prefix}_REGISTRY_URL",
            "broker_enabled": f"{prefix}_BROKER_ENABLED",
            "heartbeat_interval": f"{prefix}_HEARTBEAT_INTERVAL",
            "max_concurrent_tasks": f"{prefix}_MAX_CONCURRENT_TASKS",
            "task_timeout": f"{prefix}_TASK_TIMEOUT",
            "log_level": f"{prefix}_LOG_LEVEL",
            "log_file": f"{prefix}_LOG_FILE",
        }

        values = {}
        for field, env_var in env_mapping.items():
            env_value = os.environ.get(env_var)
            if env_value is not None:
                # Convert types
                if field in ("llm_temperature",):
                    values[field] = float(env_value)
                elif field in ("llm_max_tokens", "llm_timeout", "heartbeat_interval",
                              "max_concurrent_tasks", "task_timeout"):
                    values[field] = int(env_value)
                elif field in ("registry_enabled", "broker_enabled"):
                    values[field] = env_value.lower() in ("true", "1", "yes")
                else:
                    values[field] = env_value

        return cls(**values)

    @classmethod
    def from_file(cls, path: Union[str, Path]) -> "AgentConfig":
        """
        Load configuration from a YAML or JSON file.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path) as f:
            if path.suffix in (".yaml", ".yml"):
                data = yaml.safe_load(f)
            elif path.suffix == ".json":
                data = json.load(f)
            else:
                raise ValueError(f"Unsupported config file format: {path.suffix}")

        return cls(**data)

    def to_file(self, path: Union[str, Path]) -> None:
        """Save configuration to a YAML or JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = self.model_dump(exclude_none=True)

        with open(path, "w") as f:
            if path.suffix in (".yaml", ".yml"):
                yaml.safe_dump(data, f, default_flow_style=False)
            else:
                json.dump(data, f, indent=2)


def load_config(
    config_path: Optional[Union[str, Path]] = None,
    env_prefix: str = "EWR_AGENT",
    **overrides
) -> AgentConfig:
    """
    Load agent configuration with the following precedence:
    1. Keyword argument overrides
    2. Environment variables
    3. Config file
    4. Defaults

    Args:
        config_path: Path to config file (optional)
        env_prefix: Prefix for environment variables
        **overrides: Direct configuration overrides

    Returns:
        AgentConfig instance

    Example:
        # Load from defaults + env vars
        config = load_config()

        # Load from file + env vars + overrides
        config = load_config(
            config_path="~/.ewr-agents/config.yaml",
            llm_model="gpt-4"
        )
    """
    # Start with defaults
    config_data = {}

    # Load from file if specified
    if config_path:
        path = Path(config_path).expanduser()
        if path.exists():
            file_config = AgentConfig.from_file(path)
            config_data.update(file_config.model_dump(exclude_none=True))
    else:
        # Try default config locations
        default_paths = [
            Path.home() / ".ewr-agents" / "config.yaml",
            Path.home() / ".ewr-agents" / "config.json",
            Path.cwd() / ".ewr-agents.yaml",
            Path.cwd() / ".ewr-agents.json",
        ]
        for default_path in default_paths:
            if default_path.exists():
                file_config = AgentConfig.from_file(default_path)
                config_data.update(file_config.model_dump(exclude_none=True))
                break

    # Override with environment variables
    env_config = AgentConfig.from_env(env_prefix)
    env_data = env_config.model_dump(exclude_none=True)
    # Only use env values that differ from defaults
    default_config = AgentConfig()
    for key, value in env_data.items():
        if value != getattr(default_config, key, None):
            config_data[key] = value

    # Apply explicit overrides
    config_data.update({k: v for k, v in overrides.items() if v is not None})

    return AgentConfig(**config_data)


def get_default_config_path() -> Path:
    """Get the default configuration directory path."""
    return Path.home() / ".ewr-agents"


def ensure_config_dir() -> Path:
    """Ensure the configuration directory exists and return its path."""
    config_dir = get_default_config_path()
    config_dir.mkdir(parents=True, exist_ok=True)
    return config_dir
