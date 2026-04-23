# Testing Patterns

**Analysis Date:** 2026-04-23

## Test Framework

**Runner:**
- pytest >= 7.4.0
- Config: `pyproject.toml` `[tool.pytest.ini_options]`

**Assertion Library:**
- pytest's built-in `assert` statements
- No additional assertion libraries (no `hypothesis`, no `pytest-asyncio`)

**Coverage:**
- pytest-cov >= 4.1.0 (available in dev dependencies)
- No coverage thresholds enforced in config

**Run Commands:**
```bash
pytest                        # Run all tests
pytest tests/test_phase1.py   # Run specific test file
pytest tests/test_phase1.py::TestWorldBuilder::test_build_success  # Run specific test
pytest -x                     # Stop on first failure
pytest -v                     # Verbose output
pytest --cov=pcmfg            # Run with coverage
```

## Test File Organization

**Location:**
- Tests are in a separate `tests/` directory at project root (not co-located)
- Test fixtures live in `tests/fixtures/`

**Naming:**
- Files: `test_<module>.py` — mirrors the source module being tested
- Classes: `Test<ClassOrFunction>` — e.g., `TestWorldBuilder`, `TestPCMFGAnalyzer`, `TestCleanText`
- Methods: `test_<behavior>` — e.g., `test_build_success`, `test_empty_string`, `test_forward_fill_carries_state`

**Directory structure:**
```
tests/
├── __init__.py
├── conftest.py                    # Shared fixtures for all tests
├── fixtures/
│   └── sample_texts.py            # Static text fixtures
├── test_analyzer.py               # Tests for pcmfg/analyzer.py
├── test_checkpoint.py             # Tests for pcmfg/checkpoint.py
├── test_phase1.py                 # Tests for pcmfg/phase1/world_builder.py
├── test_phase2.py                 # Tests for pcmfg/phase2/normalizer.py
├── test_phase3.py                 # Tests for pcmfg/phase3/axis_mapper.py
├── test_synthesizer.py            # Tests for pcmfg/phase3/synthesizer.py
├── test_text_processing.py        # Tests for pcmfg/utils/text_processing.py
```

## Test Structure

**Class-based test organization** is the primary pattern:
```python
class TestWorldBuilder:
    """Tests for the WorldBuilder class."""

    def test_init(self, mock_llm_client) -> None:
        """Test WorldBuilder initialization."""
        builder = WorldBuilder(mock_llm_client)
        assert builder.llm_client == mock_llm_client

    def test_build_success(self, mock_llm_client, sample_text: str) -> None:
        """Test successful world building."""
        builder = WorldBuilder(mock_llm_client)
        result = builder.build(sample_text)
        assert isinstance(result, WorldBuilderOutput)
```

**Test method conventions:**
- Every test method has a docstring: `"""Test <specific behavior>."""`
- Return type annotation: `-> None`
- Test methods accept fixtures as parameters
- One assertion concept per test, but multiple `assert` statements are acceptable for verifying related properties

## Fixtures

**Shared fixtures** defined in `tests/conftest.py`:

```python
@pytest.fixture
def sample_config() -> Config:
    """Create a sample configuration."""
    return Config()

@pytest.fixture
def mock_llm_client() -> MagicMock:
    """Create a mock LLM client."""
    client = MagicMock()
    client.call.return_value = "Mock response"
    client.call_json.return_value = {
        "main_pairing": ["Alice", "Bob"],
        "aliases": {"Alice": ["Ali"], "Bob": ["Robert"]},
        "world_guidelines": ["They met at a ball."],
        "mermaid_graph": "graph TD\n    A[Alice] -->|loves| B[Bob]",
    }
    return client

@pytest.fixture
def sample_world_builder_output() -> WorldBuilderOutput:
    """Create a sample world builder output."""
    return WorldBuilderOutput(
        main_pairing=["Alice", "Bob"],
        aliases={"Alice": ["Ali"], "Bob": ["Robert"]},
        world_guidelines=["They met at a ball.", "Alice initially disliked Bob."],
        mermaid_graph="graph TD\n    A[Alice] -->|loves| B[Bob]",
    )

@pytest.fixture
def sample_emotion_scores() -> DirectedEmotionScores: ...

@pytest.fixture
def sample_directed_emotion(sample_emotion_scores) -> DirectedEmotion: ...

@pytest.fixture
def sample_chunk_analysis(sample_directed_emotion) -> ChunkAnalysis: ...

@pytest.fixture
def sample_axes_time_series() -> AxesTimeSeries: ...

@pytest.fixture
def sample_text() -> str:
    """Create sample romance text for testing."""
    return """
    Alice stood at the edge of the ballroom...
    """
```

**Local fixtures** defined within test classes when scoped:
```python
class TestSynthesizer:
    @pytest.fixture
    def world(self) -> WorldBuilderOutput:
        return WorldBuilderOutput(...)

    @pytest.fixture
    def sample_chunks(self) -> list[ChunkAnalysis]:
        return [...]
```

**Static text fixtures** in `tests/fixtures/sample_texts.py`:
```python
SIMPLE_ROMANCE = """They met at the ball..."""
EMOTIONAL_ARC = """Chapter 1: The Meeting..."""
NO_PAIRING = """The weather was pleasant that day..."""
INTENSE_EMOTIONS = """She hated him with every fiber of her being..."""
```

## Mocking

**Framework:** `unittest.mock.MagicMock` (stdlib, no `pytest-mock` dependency)

**Pattern: Mock LLM client** — The primary mock pattern is replacing LLM clients:
```python
from unittest.mock import MagicMock

# Fixture returns a MagicMock with pre-configured responses
mock_llm_client.call_json.return_value = {...}
mock_llm_client.call.side_effect = LLMAPIError("API failed")
```

**Pattern: Mock side effects for sequential calls** — The analyzer calls `call_json()` multiple times (once for world builder, once per chunk). Use `side_effect` with a list:
```python
mock_llm_client.call_json.side_effect = [
    # First call: World builder response
    {"main_pairing": ["Alice", "Bob"], ...},
    # Second call: Emotion extractor response
    {"chunk_id": 0, "directed_emotions": [...], ...},
]
```

**Pattern: Patch internal methods** — Used to bypass LLM client creation:
```python
from unittest.mock import patch

def test_init_with_config(self, sample_config: Config) -> None:
    with patch.object(PCMFGAnalyzer, "_create_llm_client") as mock_create:
        mock_create.return_value = MagicMock()
        analyzer = PCMFGAnalyzer(config=sample_config)
```

**Pattern: Patch at module level** — For convenience function testing:
```python
def test_analyze_function(self, sample_text: str) -> None:
    with patch("pcmfg.analyzer.PCMFGAnalyzer") as mock_analyzer_class:
        mock_analyzer = MagicMock()
        mock_analyzer_class.return_value = mock_analyzer
        mock_analyzer.analyze.return_value = MagicMock(spec=AnalysisResult)
        result = analyze(sample_text, provider="openai")
```

**What to mock:**
- LLM clients (`OpenAIClient`, `AnthropicLLMClient`) — mock `call()` and `call_json()` methods
- Internal creation methods (`_create_llm_client`) — to avoid real API key validation
- Module-level imports — to isolate the unit under test

**What NOT to mock:**
- Pydantic models — use real instances
- Utility functions (`clean_text`, `estimate_tokens`, `chunk_text_by_length`) — test them directly
- Synthesizer — pure Python, no external dependencies

## Common Test Patterns

**Default values testing:**
```python
def test_default_values(self) -> None:
    output = WorldBuilderOutput(main_pairing=["A", "B"])
    assert output.aliases == {}
    assert output.core_conflict == ""
```

**Error/exception testing:**
```python
def test_build_with_too_few_characters(self, mock_llm_client, sample_text) -> None:
    mock_llm_client.call_json.return_value = {"main_pairing": ["Alice"], ...}
    builder = WorldBuilder(mock_llm_client)
    with pytest.raises(WorldBuilderError):
        builder.build(sample_text)
```

**Boundary condition testing:**
```python
def test_empty_string(self) -> None:
    assert estimate_tokens("") == 0

def test_short_text(self) -> None:
    text = "Hello world"
    chunks = chunk_text_by_length(text)
    assert len(chunks) == 1
```

**Edge case testing:**
```python
def test_empty_chunks(self) -> None:
    imputed = impute_missing_emotions([], ["Alice", "Bob"])
    assert imputed == []

def test_insufficient_pairing(self) -> None:
    imputed = impute_missing_emotions([chunk], ["Alice"])
    assert len(imputed) == 1
```

**Property/state verification:**
```python
def test_forward_fill_carries_state(self) -> None:
    # Setup chunks where chunk 0 has both directions, chunk 1 only has A→B
    ...
    # Verify chunk 1's B→A carries forward from chunk 0
    assert b_to_a_chunk1.scores.Joy == 4  # Carried from chunk 0
    assert b_to_a_chunk1.scores.Fear == 3  # Carried from chunk 0
```

**Numeric precision testing:**
```python
def test_compute_axes_mixed_emotions(self) -> None:
    scores = DirectedEmotionScores(
        Joy=3, Trust=2,  # Intimacy: (3+2)/2 = 2.5
        Arousal=2, Anticipation=3,  # Passion: (2+3+3)/3 = 2.67
    )
    result = mapper.compute_axes(scores)
    assert result.intimacy == 2.5
    assert 2.6 <= result.passion <= 2.7
```

**Pydantic validation testing:**
```python
def test_main_pairing_length_validation(self) -> None:
    WorldBuilderOutput(main_pairing=["A", "B"])  # Should work
    with pytest.raises(Exception):
        WorldBuilderOutput(main_pairing=["A"])   # Too few
    with pytest.raises(Exception):
        WorldBuilderOutput(main_pairing=["A", "B", "C"])  # Too many
```

**File system testing:**
```python
def test_save_and_load_checkpoint(self, tmp_path: Path) -> None:
    manager = CheckpointManager(checkpoint_dir=tmp_path)
    data = CheckpointData(source_hash="test123", ...)
    saved_path = manager.save_checkpoint(data)
    assert saved_path.exists()
    loaded = manager.load_checkpoint("test123")
    assert loaded is not None
```

Uses pytest's built-in `tmp_path` fixture for file I/O tests.

## Test Types

**Unit Tests:**
- The majority of tests are unit tests
- Test individual functions, classes, and methods in isolation
- Mock external dependencies (LLM clients, file I/O via `tmp_path`)
- Located in `tests/test_*.py`

**Integration Tests:**
- `tests/test_analyzer.py::TestPCMFGAnalyzer::test_analyze_simple_text` — tests the full pipeline with mocked LLM
- No end-to-end tests with real LLM calls

**E2E Tests:**
- Not present — all LLM interactions are mocked

## Test Coverage Summary

| Source Module | Test File | Coverage Level |
|---|---|---|
| `pcmfg/models/schemas.py` | `tests/test_phase1.py` (partial) | Model validation tested via WorldBuilderOutput |
| `pcmfg/phase1/world_builder.py` | `tests/test_phase1.py` | High — success, error, edge cases |
| `pcmfg/phase1/emotion_extractor.py` | `tests/test_synthesizer.py` (chunk filtering) | Partial — `should_process_chunk` tested |
| `pcmfg/phase2/normalizer.py` | `tests/test_phase2.py` | High — validation, justification, hallucination checks |
| `pcmfg/phase3/synthesizer.py` | `tests/test_synthesizer.py` | High — imputation, timeseries building |
| `pcmfg/phase3/axis_mapper.py` | `tests/test_phase3.py` | High — all axis computations |
| `pcmfg/analyzer.py` | `tests/test_analyzer.py` | Medium — init, basic analyze, chunking |
| `pcmfg/checkpoint.py` | `tests/test_checkpoint.py` | High — save, load, delete, list |
| `pcmfg/utils/text_processing.py` | `tests/test_text_processing.py` | High — all chunking functions, cleaning, sampling |
| `pcmfg/config.py` | Not directly tested | None — used via fixtures |
| `pcmfg/llm/*.py` | Not directly tested | None — tested indirectly via mock in analyzer tests |
| `pcmfg/cli.py` | Not tested | None |
| `pcmfg/visualization/plotter.py` | Not tested | None |
| `pcmfg/utils/novel_loader.py` | Not tested | None |
| `pcmfg/utils/parallel.py` | Not tested | None |
| `pcmfg/analysis/feature_extractor.py` | Not tested | None |
| `pcmfg/analysis/clusterer.py` | Not tested | None |
| `pcmfg/analysis/plotter.py` | Not tested | None |

## Coverage Gaps

**Untested modules** (priority: High):
- `pcmfg/cli.py` — CLI commands, no integration tests
- `pcmfg/visualization/plotter.py` — all plotting logic
- `pcmfg/utils/novel_loader.py` — file loading from directory structure

**Untested modules** (priority: Medium):
- `pcmfg/llm/openai_client.py` and `pcmfg/llm/anthropic_client.py` — retry logic, error translation
- `pcmfg/utils/parallel.py` — parallel processing, async processor
- `pcmfg/config.py` — config loading, merging, file discovery

**Untested modules** (priority: Low):
- `pcmfg/analysis/feature_extractor.py` — feature extraction strategies
- `pcmfg/analysis/clusterer.py` — clustering algorithms
- `pcmfg/analysis/plotter.py` — cluster visualization

---

*Testing analysis: 2026-04-23*
