# External Integrations

**Analysis Date:** 2026-04-23

## APIs & External Services

**LLM Providers (required - at least one):**

- **OpenAI API** - Primary LLM for world building and emotion extraction
  - SDK/Client: `openai` package (`pcmfg/llm/openai_client.py`)
  - Auth: `OPENAI_API_KEY` environment variable
  - Models: gpt-4o (default), gpt-4o-mini, gpt-4-turbo
  - Features: JSON mode (`response_format={"type": "json_object"}`), chat completions
  - Custom base URL support: Yes - via `llm.base_url` config (supports OpenRouter, Azure OpenAI, vLLM, Ollama)
  - Retry: 3 attempts with exponential backoff (4s-10s) via `tenacity`

- **Anthropic Claude API** - Alternative LLM provider
  - SDK/Client: `anthropic` package (`pcmfg/llm/anthropic_client.py`)
  - Auth: `ANTHROPIC_API_KEY` environment variable
  - Models: claude-3-5-sonnet-20241022 (default), claude-3-opus-20240229, claude-3-haiku-20240307
  - Features: Messages API with system prompt, JSON response parsing
  - Retry: 3 attempts with exponential backoff (4s-10s) via `tenacity`

**LLM Usage Pattern:**
- Phase 1 (World Builder): 1 LLM call per analysis (strategic sample of text)
- Phase 2 (Emotion Extractor): 1 LLM call per text chunk (many calls, parallelized)
- Phase 3 (Synthesis): No LLM calls (pure Python)
- Token efficiency: Chunks with no character names are skipped entirely

## Data Storage

**Databases:**
- None. PCMFG is a stateless CLI tool with no database.

**File Storage (local filesystem only):**
- Input: Text files (.txt), novel directories (structured folder hierarchy)
- Output: JSON analysis results, PNG visualizations, CSV data, Markdown reports
- Default output directory: `./output/`
- Checkpoints: `.pcmfg_checkpoints/` directory (JSON files for resumable analysis)

**Caching:**
- Checkpoint system: Saves Phase 1 and Phase 2 progress to disk (`pcmfg/checkpoint.py`)
  - Checkpoint files: `.pcmfg_checkpoints/checkpoint_<hash>.json`
  - Hash-based validation: SHA-256 of source text (first 8 chars)
  - Auto-cleanup: Checkpoints deleted on successful analysis completion
- No LLM response caching (each analysis re-runs all LLM calls)

## Authentication & Identity

**Auth Provider:**
- API key authentication (no OAuth, no user accounts)
- Keys loaded from environment variables via `python-dotenv`

**Implementation:**
- `pcmfg/llm/openai_client.py`: Reads `OPENAI_API_KEY` from env
- `pcmfg/llm/anthropic_client.py`: Reads `ANTHROPIC_API_KEY` from env
- Both raise `ValueError` if API key not found
- API keys never logged or stored in output files

## Monitoring & Observability

**Error Tracking:**
- None (no Sentry, no external error tracking)

**Logs:**
- Python `logging` module throughout (`pcmfg/cli.py` configures logging)
- Log format: `%(asctime)s - %(name)s - %(levelname)s - %(message)s`
- Log level: INFO by default, DEBUG with `--debug` flag
- Rich console for user-facing output (tables, progress spinners)

**Custom Error Hierarchy:**
- `LLMError` → base (`pcmfg/llm/base.py`)
  - `LLMRateLimitError` → rate limit exceeded
  - `LLMAPIError` → API call failure
  - `LLMResponseParseError` → JSON parse failure
- `WorldBuilderError` → Phase 1 extraction failure (`pcmfg/phase1/world_builder.py`)
- `EmotionExtractionError` → Phase 2 extraction failure (`pcmfg/phase1/emotion_extractor.py`)

## CI/CD & Deployment

**Hosting:**
- Not deployed as a service. Local CLI tool only.

**CI Pipeline:**
- None detected. No GitHub Actions, no CI configuration files.

**Distribution:**
- Installable via `pip install .` (hatchling build backend)
- CLI entry point: `pcmfg` command registered in `pyproject.toml`
- Version: 0.1.0

## Environment Configuration

**Required env vars (at least one LLM key):**
- `OPENAI_API_KEY` - OpenAI API key
- `ANTHROPIC_API_KEY` - Anthropic API key

**Optional env vars (override YAML config):**
- `LLM_MODEL` - Model name override
- `LLM_TEMPERATURE` - Temperature override (default: 0.3)
- `LLM_MAX_TOKENS` - Max tokens override (default: 1000)
- `LLM_MAX_RETRIES` - Retry attempts (default: 3)
- `BEAT_DETECTION` - Chunking strategy (automatic/length/chapter)
- `OUTPUT_DIR` - Output directory (default: ./output)
- `OUTPUT_FORMATS` - Output formats (default: json,png)
- `DEBUG` - Enable debug mode (default: false)
- `LOG_LEVEL` - Log level (default: INFO)
- `MAX_CONCURRENT_REQUESTS` - Parallel API calls (default: 3)

**Secrets location:**
- `.env` file (git-ignored, template in `.env.example`)
- Environment variables directly
- `.env` is listed in `.gitignore`

## Webhooks & Callbacks

**Incoming:**
- None

**Outgoing:**
- None

**Data Flow:**
```
Local text file → PCMFG CLI → LLM API (OpenAI/Anthropic) → Local output files
```
All data stays local except LLM API calls.

## Internal Integrations

**Novel Loader:**
- Reads structured directory layouts of novel chapters (`pcmfg/utils/novel_loader.py`)
- Expects: `novel_dir/collection_N/NNN_Chapter Title.txt`
- CLI command: `pcmfg load <novel_dir>` to merge and inspect novels

**Clustering Analysis (optional):**
- Post-processing module (`pcmfg/analysis/`) for clustering emotional patterns
- Uses scikit-learn: KMeans, DBSCAN, AgglomerativeClustering, PCA, t-SNE
- Standalone script: `run_clustering_example.py`
- Feature extraction strategies: RAW, DELTA, STATISTICAL, COMBINED, WINDOWED

**Parallel Processing:**
- `pcmfg/utils/parallel.py` - ThreadPoolExecutor for concurrent LLM API calls
- Configurable concurrency: `processing.max_concurrency` (default: 5)
- Async alternative: `AsyncParallelProcessor` using asyncio (defined but not used in main pipeline)

---

*Integration audit: 2026-04-23*
