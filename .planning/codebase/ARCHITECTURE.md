# Architecture

**Analysis Date:** 2026-04-23

## Pattern Overview

**Overall:** 3-Phase Sequential Pipeline with Checkpoint Recovery

**Key Characteristics:**
- Three distinct phases executed sequentially: World Building (LLM) → Emotion Extraction (LLM loop) → Synthesis (deterministic Python)
- Protocol-based LLM abstraction enabling OpenAI and Anthropic as interchangeable backends
- Checkpoint/resume support for long-running analyses (Phase 1 and Phase 2 are checkpointable)
- Parallel processing of text chunks during Phase 2 using thread pool concurrency
- Token-efficiency optimization: chunks without relevant character names are skipped entirely (no LLM call)
- Pydantic models enforce data validation at every pipeline boundary

## Layers

**CLI / Entry Layer:**
- Purpose: Command-line interface, user interaction, output formatting
- Location: `pcmfg/cli.py`, `main.py`
- Contains: Click commands (`analyze`, `analyze-novel`, `load`, `checkpoints`, `version`), Rich console output, result export logic (JSON, CSV, PNG), stats report generation
- Depends on: `PCMFGAnalyzer`, `Config`, `EmotionPlotter`, `NovelLoader`
- Used by: End users via `pcmfg` CLI command or `python main.py`

**Orchestration Layer:**
- Purpose: Coordinates the full 3-phase pipeline, manages checkpointing, delegates chunk processing
- Location: `pcmfg/analyzer.py`
- Contains: `PCMFGAnalyzer` class with `analyze()`, `analyze_file()`, `analyze_with_checkpoint()`, convenience `analyze()` function
- Depends on: `WorldBuilder`, `EmotionExtractor`, `Synthesizer`, `CheckpointManager`, `ParallelProcessor`, `Config`, LLM clients
- Used by: CLI layer, direct Python API consumers

**LLM Abstraction Layer:**
- Purpose: Provider-agnostic interface for LLM API calls with retry logic and JSON response parsing
- Location: `pcmfg/llm/`
- Contains: `BaseLLMClient` protocol, `OpenAIClient`, `AnthropicLLMClient`, shared error types (`LLMError`, `LLMRateLimitError`, `LLMAPIError`, `LLMResponseParseError`), `parse_json_response()` utility
- Depends on: `openai` SDK, `anthropic` SDK, `tenacity` for retry
- Used by: `WorldBuilder`, `EmotionExtractor` (Phase 1 and Phase 2 agents)

**Phase 1 — World Builder (LLM Agent):**
- Purpose: Extract narrative scaffolding from strategic text sample (beginning/middle/end)
- Location: `pcmfg/phase1/world_builder.py`
- Contains: `WorldBuilder` class with `build()`, system prompt template, `WorldBuilderError`
- Depends on: LLM client, `WorldBuilderOutput` schema
- Used by: `PCMFGAnalyzer._run_world_builder()`

**Phase 2 — Emotion Extractor (LLM Agent Loop):**
- Purpose: Extract directed 9-base-emotion scores from each text chunk
- Location: `pcmfg/phase1/emotion_extractor.py`
- Contains: `EmotionExtractor` class with `extract()`, `should_process_chunk()` token-efficiency filter, `normalize_character_name()` alias resolver, system prompt builder
- Depends on: LLM client, `WorldBuilderOutput` (for context), `ChunkAnalysis` schema
- Used by: `PCMFGAnalyzer._extract_emotions()`, `PCMFGAnalyzer._extract_emotions_with_checkpoint()`

**Phase 2 — Normalizer (Validation Pass):**
- Purpose: Validate emotion scores, check justification quality, detect hallucinations
- Location: `pcmfg/phase2/normalizer.py`
- Contains: `EmotionNormalizer` with `validate_scores()`, `check_justification()`, `check_hallucination()`, `aggregate_bidirectional()`
- Depends on: `ChunkAnalysis`, `DirectedEmotionScores` schemas
- Used by: Optional quality pass between Phase 2 extraction and Phase 3 synthesis

**Phase 3 — Synthesis (Deterministic Python):**
- Purpose: Forward-fill missing emotion directions, build raw emotion time-series
- Location: `pcmfg/phase3/synthesizer.py`
- Contains: `Synthesizer` class with `synthesize()`, `impute_missing_emotions()`, `build_emotion_timeseries()`
- Depends on: `ChunkAnalysis`, `WorldBuilderOutput`, `AnalysisResult` schemas
- Used by: `PCMFGAnalyzer` as final pipeline step

**Phase 3 — Axis Mapper (Deprecated):**
- Purpose: Map raw 9 emotions to 4 romance axes (intimacy, passion, hostility, anxiety)
- Location: `pcmfg/phase3/axis_mapper.py`
- Contains: `AxisMapper` with `compute_axes()`, `map_chunk()`, `map_chunks()`
- Depends on: `DirectedEmotionScores`, `AxisValues` schemas
- Used by: Legacy code only; main pipeline now outputs raw emotions

**Visualization Layer:**
- Purpose: Generate matplotlib plots for emotional trajectories
- Location: `pcmfg/visualization/plotter.py`
- Contains: `EmotionPlotter` with `plot_timeseries()` (9×2 grid), `plot_directional_comparison()`, `plot_emotion_gap()`, `plot_comparison()`, `export_data()`
- Depends on: `EmotionTimeSeries`, `AxesTimeSeries` schemas, `matplotlib`, `numpy`
- Used by: CLI layer for PNG output

**Analysis / Clustering Layer:**
- Purpose: Feature extraction and clustering for discovering emotional narrative patterns
- Location: `pcmfg/analysis/`
- Contains: `FeatureExtractor` (raw, delta, statistical, windowed features), `SceneClusterer` (K-Means, DBSCAN, Hierarchical), `TrajectoryClusterer`, cluster visualization plots
- Depends on: `AnalysisResult` schema, `scikit-learn`, `matplotlib`, `numpy`
- Used by: `run_clustering_example.py`, direct Python API consumers

**Data Model Layer:**
- Purpose: Pydantic schemas for type-safe data validation across the pipeline
- Location: `pcmfg/models/schemas.py`
- Contains: `DirectedEmotionScores`, `DirectedEmotion`, `ChunkAnalysis`, `WorldBuilderOutput`, `EmotionTimeSeries`, `AxesTimeSeries` (deprecated), `AnalysisMetadata`, `AnalysisResult`, `BASE_EMOTIONS` constant, `ROMANCE_AXES` constant
- Depends on: `pydantic`
- Used by: Every layer in the system

**Utility Layer:**
- Purpose: Shared infrastructure for text processing, parallel execution, novel loading
- Location: `pcmfg/utils/`
- Contains: `text_processing.py` (chunking strategies, token estimation, cleaning, strategic sampling), `parallel.py` (`ParallelProcessor`, `AsyncParallelProcessor`, `process_in_batches`), `novel_loader.py` (`NovelLoader` for structured directory input)
- Depends on: Standard library, `concurrent.futures`
- Used by: Orchestration layer, CLI layer

**Checkpoint Layer:**
- Purpose: Persist and restore analysis progress for resumable long-running jobs
- Location: `pcmfg/checkpoint.py`
- Contains: `CheckpointManager`, `CheckpointData` (Pydantic model), `compute_text_hash()`, `list_checkpoints()`
- Depends on: `ChunkAnalysis`, `WorldBuilderOutput` schemas
- Used by: `PCMFGAnalyzer.analyze_with_checkpoint()`, CLI `checkpoints` command

**Configuration Layer:**
- Purpose: Load and merge configuration from YAML files, CLI args, and defaults
- Location: `pcmfg/config.py`
- Contains: `Config` (root), `LLMConfig`, `ProcessingConfig`, `OutputConfig` (all Pydantic), `load_config()`, `merge_cli_overrides()`
- Depends on: `pydantic`, `pyyaml`
- Used by: CLI layer, `PCMFGAnalyzer`

## Data Flow

**Standard Analysis (`pcmfg analyze`):**

1. CLI parses arguments, loads `Config` from YAML/env/defaults, merges CLI overrides
2. `PCMFGAnalyzer` is created with config; LLM client is instantiated (OpenAI or Anthropic)
3. Input text is cleaned via `clean_text()`
4. **Phase 1:** Strategic sample extracted from text (beginning 40%, middle 30%, end 30%) → `WorldBuilder.build()` calls LLM once → `WorldBuilderOutput` returned (main pairing, aliases, core conflict, world guidelines, mermaid graph)
5. **Phase 2:** `EmotionExtractor` is created with World Builder context; text is chunked (chapter/paragraph/length mode); each chunk is checked via `should_process_chunk()` against aliases (skip if no character names); remaining chunks are processed in parallel via `ParallelProcessor` (ThreadPoolExecutor) calling LLM per chunk → `list[ChunkAnalysis]`
6. **Phase 3:** `Synthesizer.synthesize()` forward-fills missing directions (B→A when B absent), builds `EmotionTimeSeries` for A→B and B→A → `AnalysisResult`
7. Results exported as JSON, CSV, and PNG plots by CLI layer

**Novel Analysis (`pcmfg analyze-novel`):**

1. `NovelLoader` reads structured directory (collection folders → chapter .txt files)
2. Chapters filtered by range (`--start`/`--end`), side stories (chapter ≥ 1000) excluded
3. Chapters merged into single text
4. Pipeline runs with checkpoint support (`analyze_with_checkpoint`)
5. Checkpoints saved after Phase 1 and every 10 chunks during Phase 2
6. On completion, checkpoint deleted

**State Management:**
- Pipeline state flows through function return values (no global mutable state)
- Checkpoint state is persisted to `.pcmfg_checkpoints/checkpoint_{hash}.json`
- LLM client state is encapsulated in client instances (API key, model config)
- Configuration is immutable after load (Pydantic frozen models)

## Key Abstractions

**LLM Client Protocol (`BaseLLMClient`):**
- Purpose: Decouple pipeline logic from specific LLM providers
- Examples: `pcmfg/llm/openai_client.py`, `pcmfg/llm/anthropic_client.py`
- Pattern: Python `Protocol` with `call()` and `call_json()` methods; implementations use `tenacity` retry decorator; provider-specific errors mapped to domain errors (`LLMRateLimitError`, `LLMAPIError`, `LLMResponseParseError`)

**Directed Emotion:**
- Purpose: Core data unit representing one character's feelings toward another
- Examples: `pcmfg/models/schemas.py` (`DirectedEmotion`, `DirectedEmotionScores`)
- Pattern: Source/target pair with 9 integer scores (1-5 scale); frozen Pydantic model; default all-1s baseline

**Chunk Analysis:**
- Purpose: Per-chunk analysis result linking narrative position to emotions
- Examples: `pcmfg/models/schemas.py` (`ChunkAnalysis`)
- Pattern: Sequential `chunk_id`, float `position` (0.0-1.0), POV character, characters present, list of `DirectedEmotion`, scene summary

**World Builder Output:**
- Purpose: Context envelope for downstream emotion extraction
- Examples: `pcmfg/models/schemas.py` (`WorldBuilderOutput`)
- Pattern: Main pairing (2 names), alias dictionary, core conflict string, world guidelines list, mermaid graph string

**Parallel Processor:**
- Purpose: Generic concurrency wrapper for I/O-bound operations
- Examples: `pcmfg/utils/parallel.py` (`ParallelProcessor`, `AsyncParallelProcessor`)
- Pattern: Generic over input/output types; uses `ThreadPoolExecutor` with configurable max concurrency; returns ordered `ProcessingResult` list with success/error tracking

## Entry Points

**CLI Entry Point:**
- Location: `pcmfg/cli.py` (function `main()`)
- Triggers: `pcmfg` command (registered via `pyproject.toml` scripts), or `python main.py`
- Responsibilities: Click group with subcommands (`analyze`, `analyze-novel`, `load`, `checkpoints`, `version`), logging setup, config loading, progress display via Rich, result export

**Python API Entry Point:**
- Location: `pcmfg/analyzer.py` (class `PCMFGAnalyzer`, function `analyze()`)
- Triggers: Direct import `from pcmfg import PCMFGAnalyzer, analyze`
- Responsibilities: Full pipeline orchestration with or without checkpoints

**Clustering Entry Point:**
- Location: `run_clustering_example.py`
- Triggers: `python run_clustering_example.py`
- Responsibilities: Demonstrates feature extraction and scene clustering workflow

## Error Handling

**Strategy:** Domain-specific exception hierarchy with graceful degradation

**Patterns:**
- LLM errors are mapped to domain types: `LLMRateLimitError` → retry (via tenacity), `LLMAPIError` → fail, `LLMResponseParseError` → retry parse with fallback
- Phase 1 failure returns a default `WorldBuilderOutput` with placeholder pairing ("Character A", "Character B") instead of crashing
- Phase 2 chunk extraction failures return default chunks (all-1s scores) with error justification, allowing pipeline to continue
- CLI catches `ValueError` (config errors) and generic `Exception` (analysis errors) with Rich-formatted error messages
- Pydantic `ValidationError` on LLM responses triggers fallback to default chunks

**Cross-Cutting Concerns**

**Logging:** Python `logging` module; configured in CLI via `setup_logging()` (DEBUG or INFO level); each module has `logger = logging.getLogger(__name__)`

**Validation:** Pydantic models enforce type safety at all boundaries; `EmotionNormalizer` provides optional quality validation (score range, justification quality, hallucination detection)

**Authentication:** API keys loaded from environment variables (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`) via `.env` file (`python-dotenv`); keys are required at client construction time

---

*Architecture analysis: 2026-04-23*
