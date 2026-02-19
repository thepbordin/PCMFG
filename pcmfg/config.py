"""Configuration management for PCMFG.

Supports loading configuration from:
1. CLI arguments (highest priority)
2. Config file (YAML)
3. Environment variables
4. Built-in defaults (lowest priority)
"""

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field


class LLMConfig(BaseModel):
    """LLM provider configuration."""

    provider: Literal["openai", "anthropic"] = "openai"
    model: str = "gpt-4o"
    temperature: float = Field(default=0.3, ge=0.0, le=2.0)
    max_tokens: int = Field(default=4096, ge=100)


class ProcessingConfig(BaseModel):
    """Text processing configuration."""

    beat_detection: Literal["automatic", "length", "chapter"] = "automatic"
    beat_length: int = Field(default=500, ge=100, description="Target words per beat")
    min_beat_length: int = Field(default=200, ge=50, description="Minimum words per beat")
    max_chunk_tokens: int = Field(
        default=3000, ge=500, description="Maximum tokens per LLM chunk"
    )


class OutputConfig(BaseModel):
    """Output configuration."""

    formats: list[Literal["json", "csv", "png"]] = Field(default=["json", "png"])
    include_stats: bool = True
    dpi: int = Field(default=300, ge=72, description="Image DPI for PNG output")


class Config(BaseModel):
    """Root configuration model."""

    llm: LLMConfig = Field(default_factory=LLMConfig)
    processing: ProcessingConfig = Field(default_factory=ProcessingConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)


# Default config paths to search (in order)
CONFIG_SEARCH_PATHS = [
    Path("./pcmfg_config.yaml"),
    Path("./pcmfg_config.yml"),
    Path.home() / ".pcmfg" / "config.yaml",
]


def load_config(config_path: str | Path | None = None) -> Config:
    """Load configuration from file or return defaults.

    Args:
        config_path: Explicit path to config file. If None, searches default locations.

    Returns:
        Config object with loaded or default values.
    """
    if config_path is not None:
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        return _load_config_from_file(path)

    # Search default locations
    for search_path in CONFIG_SEARCH_PATHS:
        if search_path.exists():
            return _load_config_from_file(search_path)

    # Return defaults if no config file found
    return Config()


def _load_config_from_file(path: Path) -> Config:
    """Load configuration from a YAML file.

    Args:
        path: Path to the YAML config file.

    Returns:
        Config object with loaded values.
    """
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    return Config(**data)


def merge_cli_overrides(config: Config, **overrides: str | int | float | bool | None) -> Config:
    """Merge CLI argument overrides into configuration.

    Args:
        config: Base configuration object.
        **overrides: Key-value pairs to override. Keys should match Config fields
                     (e.g., "llm__model", "output__dpi").

    Returns:
        New Config object with overrides applied.
    """
    config_dict = config.model_dump()

    for key, value in overrides.items():
        if value is None:
            continue

        # Support nested keys with double underscore (e.g., "llm__model")
        parts = key.split("__")
        if len(parts) == 1:
            if key in config_dict:
                config_dict[key] = value
        elif len(parts) == 2:
            section, field = parts
            if section in config_dict and field in config_dict[section]:
                config_dict[section][field] = value

    return Config(**config_dict)
