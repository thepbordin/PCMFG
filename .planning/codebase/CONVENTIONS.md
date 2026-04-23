# Coding Conventions

**Analysis Date:** 2026-04-23

## Language & Runtime

- **Python 3.12** (see `.python-version` and `pyproject.toml` `requires-python`)
- **Package manager**: `uv` (lockfile: `uv.lock` present)
- **Build backend**: `hatchling`

## File Naming

- **Source files**: `snake_case.py` — e.g., `world_builder.py`, `emotion_extractor.py`, `text_processing.py`
- **Test files**: `test_<module>.py` — e.g., `test_phase1.py`, `test_synthesizer.py`, `test_text_processing.py`
- **Fixture files**: descriptive names in `tests/fixtures/` — e.g., `sample_texts.py`
- **Module directories**: `snake_case` — e.g., `phase1/`, `phase3/`, `utils/`, `llm/`, `analysis/`
- **Private helpers**: prefixed with underscore — e.g., `_split_sentences()`, `_parse_chapter_filename()`, `_load_config_from_file()`
- **Constants**: `UPPER_SNAKE_CASE` — e.g., `WORLD_BUILDER_SYSTEM_PROMPT`, `BASE_EMOTIONS`, `ROMANCE_AXES`, `EMOTION_CONFIG`, `CHECKPOINT_VERSION`

## Naming Patterns

**Functions:**
- `snake_case` — e.g., `estimate_tokens()`, `clean_text()`, `chunk_text_by_length()`, `should_process_chunk()`, `impute_missing_emotions()`
- Verb-first for actions: `build`, `extract`, `create`, `compute`, `load`, `save`, `parse`, `normalize`, `validate`
- Boolean-returning functions use `is_`, `has_`, or `should_` prefix — e.g., `has_valid_checkpoint()`, `should_process_chunk()`

**Classes:**
- `PascalCase` — e.g., `PCMFGAnalyzer`, `WorldBuilder`, `EmotionExtractor`, `Synthesizer`, `EmotionPlotter`, `CheckpointManager`
- Data models (Pydantic): `PascalCase` — e.g., `DirectedEmotionScores`, `ChunkAnalysis`, `WorldBuilderOutput`, `AnalysisResult`, `CheckpointData`
- Named tuples: `PascalCase` — e.g., `ValidationWarning`, `ProcessingResult`

**Variables:**
- `snake_case` — e.g., `main_pairing`, `chunk_text`, `last_known_state`, `total_chunks`
- Type aliases use `PascalCase` — e.g., `EmotionScore`, `AxisValue`, `EmotionTimeSeriesDict`

**Type annotations:**
- Use `X | Y` union syntax (not `Union[X, Y]`) — e.g., `str | None`, `list[ChunkAnalysis]`, `dict[str, list[str]]`
- Use `list[X]`, `dict[K, V]`, `tuple[X, Y]` (not `List`, `Dict`, `Tuple` from typing) — e.g., `list[str]`, `dict[str, Any]`
- Use `collections.abc.Callable`, `collections.abc.Iterator` (not `typing.Callable`, `typing.Iterator`) — though `typing.Any`, `typing.Literal`, `typing.Protocol`, `typing.Generic`, `typing.TypeVar`, `typing.TypedDict` are used from `typing`

## Code Style

**Formatter:** Black
- Line length: 88 characters
- Target Python: 3.12
- Config in `pyproject.toml` `[tool.black]`

**Linter:** Ruff
- Line length: 88
- Target Python: 3.12
- Enabled rules: `E` (pycodestyle errors), `F` (pyflakes), `I` (isort), `N` (pep8-naming), `W` (pycodestyle warnings), `UP` (pyupgrade)
- Config in `pyproject.toml` `[tool.ruff]`

**Type checker:** mypy
- `disallow_untyped_defs = true` — all functions must have type annotations
- `warn_return_any = true`
- `warn_unused_configs = true`
- Config in `pyproject.toml` `[tool.mypy]`

**No pre-commit hooks** configured (no `.pre-commit-config.yaml` found).

## Import Organization

Observed ordering (implicit via Ruff's `I` rule):
1. Standard library (`os`, `json`, `re`, `logging`, `pathlib`, `datetime`, `abc`, `enum`, `dataclasses`, `collections.abc`)
2. Third-party (`pydantic`, `openai`, `anthropic`, `click`, `rich`, `yaml`, `tenacity`, `numpy`, `pandas`, `matplotlib`, `sklearn`)
3. Local package (`pcmfg.*`)

```python
# Example from pcmfg/analyzer.py:
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from pcmfg.checkpoint import CheckpointData, CheckpointManager, compute_text_hash
from pcmfg.config import Config
from pcmfg.llm.anthropic_client import AnthropicLLMClient
from pcmfg.llm.openai_client import OpenAIClient
from pcmfg.models.schemas import (
    AnalysisMetadata,
    AnalysisResult,
    ChunkAnalysis,
    WorldBuilderOutput,
)
from pcmfg.phase1.emotion_extractor import EmotionExtractor, should_process_chunk
from pcmfg.phase1.world_builder import WorldBuilder, WorldBuilderError
from pcmfg.phase3.synthesizer import Synthesizer
from pcmfg.utils.text_processing import (
    chunk_text_by_chapter,
    chunk_text_by_length,
    chunk_text_by_paragraph,
    clean_text,
    estimate_tokens,
    get_strategic_sample,
)
```

**Lazy imports**: Used for heavy or optional dependencies inside functions:
```python
# Inside method body:
from pcmfg.utils.parallel import ParallelProcessor
import numpy as np
from pcmfg.utils.novel_loader import NovelLoader, get_novel_info
```

**Path aliases**: None — all imports use fully qualified `pcmfg.*` paths.

## Module Structure

Every package directory contains an `__init__.py`. The root `pcmfg/__init__.py` exports the public API:

```python
__all__ = [
    "__version__",
    "PCMFGAnalyzer",
    "analyze",
    "Config",
    "load_config",
    "AnalysisResult",
]
```

**Sub-packages do NOT re-export** — consumers import directly from the module, e.g., `from pcmfg.phase1.world_builder import WorldBuilder`.

## Data Models

Use **Pydantic v2** `BaseModel` for all data validation:
```python
class DirectedEmotionScores(BaseModel):
    Joy: EmotionScore = Field(default=1, description="Happiness, pleasure, delight")
    Trust: EmotionScore = Field(default=1, description="Safety, reliance, vulnerability")
    # ...
    model_config = ConfigDict(frozen=True)
```

- Use `Field()` with `description` for documentation
- Use `Field(default=...)` for defaults
- Use `Field(ge=..., le=...)` for numeric constraints
- Use `Annotated` for reusable type aliases — e.g., `EmotionScore = Annotated[int, Field(ge=1, le=5)]`
- Use `ConfigDict(frozen=True)` for immutable models
- Use `default_factory=list` for mutable defaults

Use `NamedTuple` for lightweight data containers:
```python
class ValidationWarning(NamedTuple):
    chunk_id: int
    warning_type: str
    message: str
```

Use `dataclass` for simple data holders:
```python
@dataclass
class ProcessingResult(Generic[R]):
    index: int
    result: R | None
    error: Exception | None
    success: bool
```

## Error Handling

**Custom exception hierarchy** in `pcmfg/llm/base.py`:
```python
class LLMError(Exception): ...
class LLMRateLimitError(LLMError): ...
class LLMAPIError(LLMError): ...
class LLMResponseParseError(LLMError): ...
```

**Domain-specific exceptions** defined in their respective modules:
- `WorldBuilderError(Exception)` in `pcmfg/phase1/world_builder.py`
- `EmotionExtractionError(Exception)` in `pcmfg/phase1/emotion_extractor.py`

**Pattern: Wrap and re-raise** — LLM client methods catch provider-specific errors and raise domain exceptions:
```python
except RateLimitError as e:
    raise LLMRateLimitError(f"OpenAI rate limit exceeded: {e}")
except APIError as e:
    raise LLMAPIError(f"OpenAI API error: {e}")
```

**Pattern: Graceful degradation** — `EmotionExtractor.extract()` catches LLM errors and returns a default chunk instead of propagating:
```python
except (LLMAPIError, LLMResponseParseError) as e:
    return self._create_default_chunk(chunk_id, position, str(e))
```

**Pattern: Validation with default fallback** — `PCMFGAnalyzer._run_world_builder()` catches `WorldBuilderError` and returns a minimal default:
```python
except WorldBuilderError as e:
    return WorldBuilderOutput(
        main_pairing=default_pairing,
        aliases={...},
        world_guidelines=[],
        mermaid_graph="",
    )
```

**Never use bare `except:`** — always specify exception types.

## Logging

**Framework**: Python standard `logging` module.

**Pattern: Module-level logger**:
```python
logger = logging.getLogger(__name__)
```

Used consistently across all modules. Log levels:
- `logger.debug()` — chunk-level details (e.g., "Skipping chunk 5: no character names found")
- `logger.info()` — phase-level progress (e.g., "Phase 1: World Builder", "Processing chunks: 42/100")
- `logger.warning()` — quality issues (e.g., "Invalid emotion scores detected", "Checkpoint version mismatch")
- `logger.error()` — failures (e.g., "World builder failed", "Error processing item 5")

**CLI logging setup** in `pcmfg/cli.py`:
```python
def setup_logging(debug: bool = False) -> None:
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
```

## Docstrings

**Every module has a module-level docstring** describing its purpose.

**Every public function and class has a Google-style docstring** with:
- Brief summary line
- Blank line
- Description (when needed)
- `Args:` section with type and description
- `Returns:` section with type and description
- `Raises:` section (when applicable)

```python
def chunk_text_by_length(
    text: str,
    max_tokens: int = 3000,
    min_chunk_tokens: int = 200,
) -> list[str]:
    """Split text into chunks by word count for LLM processing.

    Attempts to split at sentence boundaries when possible.

    Args:
        text: Input text string.
        max_tokens: Maximum tokens per chunk.
        min_chunk_tokens: Minimum tokens per chunk (avoid over-fragmentation).

    Returns:
        List of text chunks.
    """
```

**JSDoc/TSDoc**: Not applicable (Python project).

## Function Design

**Size**: Functions tend to be moderate length (20-60 lines). Methods can be longer for orchestration logic (e.g., `_extract_emotions_with_checkpoint` is ~80 lines). Single-responsibility is favored — helper functions are extracted for subtasks.

**Parameters**: Use keyword arguments with sensible defaults. Required parameters come first, optional parameters follow.

**Return values**: Always typed. Return `None` explicitly only for void methods. Use `| None` for optional returns — e.g., `WorldBuilderOutput | None`.

**Private methods**: Prefixed with `_` — e.g., `_create_llm_client()`, `_parse_response()`, `_create_default_chunk()`, `_chunk_text()`, `_split_sentences()`.

## Concurrency

**Pattern: ThreadPoolExecutor for I/O-bound parallelism** in `pcmfg/utils/parallel.py`:
- `ParallelProcessor` — synchronous parallel processing using `ThreadPoolExecutor`
- `AsyncParallelProcessor` — async version using `asyncio.Semaphore`
- `process_in_batches()` — sequential batch processing for rate limiting
- Concurrency controlled by `max_concurrency` config (default: 5)

## Configuration

**Pydantic-based config** in `pcmfg/config.py`:
- Nested models: `Config` → `LLMConfig`, `ProcessingConfig`, `OutputConfig`
- Loads from YAML file, environment variables, or uses defaults
- CLI overrides merged with `merge_cli_overrides()` using `__` separator for nested keys

## Protocol-Based Interfaces

**LLM client interface** uses `typing.Protocol` (not ABC):
```python
class BaseLLMClient(Protocol):
    @abstractmethod
    def call(self, system_prompt: str, user_prompt: str) -> str: ...
    @abstractmethod
    def call_json(self, system_prompt: str, user_prompt: str) -> dict: ...
```

This allows any object implementing `call()` and `call_json()` to be used as an LLM client.

## Retry Logic

**Use `tenacity` for retrying LLM calls**:
```python
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    reraise=True,
)
def call(self, system_prompt: str, user_prompt: str) -> str:
    ...
```

Both `OpenAIClient` and `AnthropicLLMClient` use identical retry decorators.

## File Encoding

Always use `encoding="utf-8"` when opening files:
```python
with open(path, encoding="utf-8") as f:
    data = yaml.safe_load(f)
```

---

*Convention analysis: 2026-04-23*
