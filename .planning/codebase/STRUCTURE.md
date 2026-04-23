# Codebase Structure

**Analysis Date:** 2026-04-23

## Directory Layout

```
PCMFG/                          # Project root
├── main.py                     # Thin CLI entry point (delegates to pcmfg.cli)
├── pyproject.toml              # Project metadata, dependencies, tool configs
├── uv.lock                     # Dependency lockfile (uv package manager)
├── .python-version             # Python version pinning
├── pcmfg_config.yaml           # Active configuration (YAML)
├── pcmfg_config.example.yaml   # Example configuration template
├── CLAUDE.md                   # AI assistant technical guide
├── README.md                   # Project documentation
├── .env.example                # Environment variable template
├── .env                        # Environment variables (secrets — DO NOT read)
├── run_clustering_example.py   # Standalone clustering demo script
│
├── pcmfg/                      # Main package
│   ├── __init__.py             # Public API exports (PCMFGAnalyzer, analyze, Config, AnalysisResult)
│   ├── cli.py                  # Click CLI with subcommands
│   ├── analyzer.py             # Pipeline orchestrator (PCMFGAnalyzer)
│   ├── config.py               # Pydantic config models + loader
│   ├── checkpoint.py           # Checkpoint save/resume for long analyses
│   │
│   ├── models/                 # Pydantic data schemas
│   │   ├── __init__.py
│   │   └── schemas.py          # All data models (DirectedEmotion, ChunkAnalysis, etc.)
│   │
│   ├── llm/                    # LLM provider abstraction
│   │   ├── __init__.py
│   │   ├── base.py             # BaseLLMClient protocol, error types, JSON parser
│   │   ├── openai_client.py    # OpenAI implementation with retry
│   │   └── anthropic_client.py # Anthropic implementation with retry
│   │
│   ├── phase1/                 # Phase 1: World Builder + Emotion Extractor
│   │   ├── __init__.py         # Exports WorldBuilder, EmotionExtractor
│   │   ├── world_builder.py    # Agent 1: narrative context extraction
│   │   └── emotion_extractor.py # Agent 2: directed emotion scoring per chunk
│   │
│   ├── phase2/                 # Phase 2: Validation/Normalization
│   │   ├── __init__.py         # Exports EmotionNormalizer
│   │   └── normalizer.py       # Score validation, hallucination detection
│   │
│   ├── phase3/                 # Phase 3: Deterministic synthesis
│   │   ├── __init__.py         # Exports Synthesizer, AxisMapper
│   │   ├── synthesizer.py      # Forward fill + time-series building
│   │   └── axis_mapper.py      # (Deprecated) Maps emotions to romance axes
│   │
│   ├── utils/                  # Shared utilities
│   │   ├── __init__.py
│   │   ├── text_processing.py  # Chunking, token estimation, cleaning, sampling
│   │   ├── parallel.py         # ParallelProcessor, AsyncParallelProcessor
│   │   └── novel_loader.py     # NovelLoader for structured directory input
│   │
│   ├── visualization/          # Emotion trajectory plotting
│   │   ├── __init__.py
│   │   └── plotter.py          # EmotionPlotter (timeseries, comparison, gap)
│   │
│   └── analysis/               # Clustering and feature extraction
│       ├── __init__.py         # Exports clustering and feature classes
│       ├── feature_extractor.py # FeatureExtractor (raw, delta, statistical, windowed)
│       ├── clusterer.py        # SceneClusterer, TrajectoryClusterer
│       └── plotter.py          # Cluster visualization (PCA, timeline, radar)
│
├── tests/                      # Test suite
│   ├── __init__.py
│   ├── conftest.py             # Shared pytest fixtures
│   ├── fixtures/
│   │   └── sample_texts.py     # Test text data
│   ├── test_phase1.py          # World Builder + Emotion Extractor tests
│   ├── test_phase2.py          # Normalizer tests
│   ├── test_phase3.py          # Synthesizer tests
│   ├── test_synthesizer.py     # Additional synthesizer tests
│   ├── test_analyzer.py        # Pipeline orchestrator tests
│   ├── test_text_processing.py # Chunking and text utils tests
│   └── test_checkpoint.py      # Checkpoint save/resume tests
│
├── .pcmfg_checkpoints/         # Runtime checkpoint storage (generated)
├── apps/                       # Empty — reserved for future app code
├── novels/                     # Sample novel directories for input
├── .planning/                  # GSD planning directory
│   └── codebase/               # Codebase analysis documents
│
└── output*/                    # Analysis output directories (generated, various names)
```

## Directory Purposes

**`pcmfg/`** — Main Python package containing all library code. Organized by pipeline phase (`phase1/`, `phase2/`, `phase3/`) with shared infrastructure in `llm/`, `models/`, `utils/`, `visualization/`, and `analysis/`.

**`pcmfg/models/`** — Pydantic schemas that define the data contract for the entire pipeline. All other modules import from here. Single file `schemas.py` contains all models.

**`pcmfg/llm/`** — Provider-agnostic LLM abstraction. `base.py` defines the protocol and error types. Implementations in `openai_client.py` and `anthropic_client.py` wrap SDK calls with retry logic.

**`pcmfg/phase1/`** — Contains both the World Builder (Agent 1) and Emotion Extractor (Agent 2), both of which are LLM-dependent. Despite the name "phase1", the Emotion Extractor performs what the pipeline documents call "Phase 2" work.

**`pcmfg/phase2/`** — Optional validation/normalization pass. Contains `EmotionNormalizer` for score validation and hallucination detection. Not called in the main pipeline flow by default.

**`pcmfg/phase3/`** — Deterministic Python processing. `Synthesizer` is the active component; `AxisMapper` is deprecated but retained for backward compatibility.

**`pcmfg/utils/`** — Shared infrastructure used across phases. `text_processing.py` is the most critical utility (chunking strategies, token estimation, strategic sampling). `parallel.py` provides generic concurrency primitives.

**`pcmfg/visualization/`** — Matplotlib-based plotting for emotional trajectory visualization. Produces PNG files.

**`pcmfg/analysis/`** — Post-analysis clustering module. Independent of the main pipeline — operates on completed `AnalysisResult` objects. Requires `scikit-learn`.

**`tests/`** — Pytest test suite. Tests are organized by module/pipeline phase. `conftest.py` provides shared fixtures (mock LLM client, sample schemas, sample text).

**`novels/`** — Input data directory. Contains novel text files organized in collection-folder structures that `NovelLoader` can parse.

**`.pcmfg_checkpoints/`** — Runtime directory for checkpoint JSON files. Created automatically during analysis. Named `checkpoint_{hash}.json`.

## Key File Locations

**Entry Points:**
- `main.py`: Thin wrapper that calls `pcmfg.cli:main()` when run directly
- `pcmfg/cli.py`: Click CLI with subcommands `analyze`, `analyze-novel`, `load`, `checkpoints`, `version`
- `pcmfg/__init__.py`: Public Python API — `PCMFGAnalyzer`, `analyze()`, `Config`, `AnalysisResult`

**Configuration:**
- `pcmfg/config.py`: Pydantic config models (`Config`, `LLMConfig`, `ProcessingConfig`, `OutputConfig`) and loader functions
- `pcmfg_config.yaml`: Active YAML configuration (loaded by default)
- `pcmfg_config.example.yaml`: Documented template for configuration
- `.env`: Environment variables for API keys (secrets)
- `.env.example`: Template showing required env vars

**Core Logic:**
- `pcmfg/analyzer.py`: Pipeline orchestrator (`PCMFGAnalyzer` class)
- `pcmfg/phase1/world_builder.py`: Agent 1 — World Builder
- `pcmfg/phase1/emotion_extractor.py`: Agent 2 — Emotion Extractor
- `pcmfg/phase3/synthesizer.py`: Phase 3 — Forward fill + time-series synthesis
- `pcmfg/checkpoint.py`: Checkpoint manager for resumable analysis

**Data Models:**
- `pcmfg/models/schemas.py`: All Pydantic models (`DirectedEmotionScores`, `DirectedEmotion`, `ChunkAnalysis`, `WorldBuilderOutput`, `EmotionTimeSeries`, `AnalysisResult`)

**LLM Clients:**
- `pcmfg/llm/base.py`: Protocol and error types
- `pcmfg/llm/openai_client.py`: OpenAI implementation
- `pcmfg/llm/anthropic_client.py`: Anthropic implementation

**Testing:**
- `tests/conftest.py`: Shared pytest fixtures
- `tests/test_phase1.py`: World Builder and Emotion Extractor tests
- `tests/test_phase2.py`: Normalizer tests
- `tests/test_phase3.py`: Synthesizer tests
- `tests/test_analyzer.py`: Full pipeline orchestrator tests
- `tests/fixtures/sample_texts.py`: Test text data

## Naming Conventions

**Files:**
- Snake_case throughout: `emotion_extractor.py`, `text_processing.py`, `novel_loader.py`
- Test files prefixed with `test_`: `test_phase1.py`, `test_analyzer.py`
- Fixture modules in `fixtures/` subdirectory

**Directories:**
- Lowercase snake_case: `phase1/`, `phase2/`, `phase3/`, `visualization/`, `analysis/`
- `__init__.py` in every package directory

**Classes:**
- PascalCase: `PCMFGAnalyzer`, `WorldBuilder`, `EmotionExtractor`, `Synthesizer`, `EmotionPlotter`, `CheckpointManager`
- Protocol classes: `BaseLLMClient` (not suffixed with "Protocol")
- Error classes: PascalCase suffixed with "Error": `WorldBuilderError`, `EmotionExtractionError`, `LLMError`

**Functions:**
- Snake_case: `analyze_file()`, `should_process_chunk()`, `normalize_character_name()`, `build_emotion_timeseries()`
- Private functions prefixed with `_`: `_parse_response()`, `_create_default_chunk()`, `_chunk_text()`, `_split_sentences()`
- Module-level convenience functions: `analyze()` (in `analyzer.py`), `load_novel_from_directory()` (in `novel_loader.py`)

**Constants:**
- UPPER_SNAKE_CASE: `BASE_EMOTIONS`, `ROMANCE_AXES`, `EMOTION_CONFIG`, `CHECKPOINT_VERSION`, `SKLEARN_AVAILABLE`
- Config search paths: `CONFIG_SEARCH_PATHS`

**Pydantic Models:**
- PascalCase: `DirectedEmotionScores`, `ChunkAnalysis`, `WorldBuilderOutput`, `AnalysisResult`
- Type aliases: PascalCase with "Type" suffix: `EmotionScore`, `AxisValue`, `FeatureType`
- TypedDicts: PascalCase with "Dict" suffix: `EmotionTimeSeriesDict`

## Where to Add New Code

**New LLM Provider (e.g., Google Gemini, local LLM):**
- Implementation: `pcmfg/llm/{provider}_client.py`
- Follow pattern from `pcmfg/llm/openai_client.py` — implement `call()` and `call_json()`, use `tenacity` retry, map to domain error types
- Register in `pcmfg/analyzer.py:_create_llm_client()` — add new `elif` branch
- Add provider name to `pcmfg/config.py:LLMConfig.provider` Literal type

**New Chunking Strategy:**
- Implementation: Add new function in `pcmfg/utils/text_processing.py` (e.g., `chunk_text_by_scene()`)
- Register in `pcmfg/analyzer.py:_chunk_text()` — add new `elif` branch
- Add mode name to `pcmfg/config.py:ProcessingConfig.beat_detection` Literal type

**New Visualization Type:**
- Implementation: Add new method to `pcmfg/visualization/plotter.py` `EmotionPlotter` class
- Call from `pcmfg/cli.py:_export_results()` if it should be auto-generated

**New Emotion/Score Dimension:**
- Add to `BASE_EMOTIONS` list in `pcmfg/models/schemas.py`
- Add field to `DirectedEmotionScores` model
- Add entry to `EMOTION_CONFIG` in `pcmfg/visualization/plotter.py`
- Update Agent 2 system prompt in `pcmfg/phase1/emotion_extractor.py:build_emotion_extractor_system_prompt()`

**New Clustering Algorithm:**
- Add to `ClusteringAlgorithm` enum in `pcmfg/analysis/clusterer.py`
- Add implementation in `SceneClusterer.cluster()` method

**New CLI Subcommand:**
- Add new `@cli.command()` function in `pcmfg/cli.py`

**New Test:**
- Unit test for a specific module: `tests/test_{module_name}.py`
- Shared fixtures: `tests/conftest.py`
- Test data: `tests/fixtures/`

## Special Directories

**`.pcmfg_checkpoints/`:**
- Purpose: Stores checkpoint JSON files for resumable analysis
- Generated: Yes (created automatically during analysis with checkpoints)
- Committed: No (listed in `.gitignore`)

**`novels/`:**
- Purpose: Contains input novel text files organized in collection-folder structures
- Generated: No (user-provided input data)
- Committed: Partially (sample data may be committed)

**`apps/`:**
- Purpose: Reserved for future application code
- Generated: No
- Committed: Yes (currently empty)

**`output*/`:**
- Purpose: Analysis output directories (JSON results, CSV data, PNG plots, markdown reports)
- Generated: Yes (created during analysis)
- Committed: No (various output directories with different names)

**`.venv/`:**
- Purpose: Python virtual environment
- Generated: Yes
- Committed: No

---

*Structure analysis: 2026-04-23*
