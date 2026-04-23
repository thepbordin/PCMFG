# Technology Stack

**Analysis Date:** 2026-04-23

## Languages

**Primary:**
- Python 3.12 - Entire codebase (pipeline, CLI, models, visualization, analysis)

**Secondary:**
- YAML - Configuration files (`pcmfg_config.yaml`)
- JSON - LLM prompts/responses, checkpoint data, analysis output
- Mermaid.js - Relationship graph syntax (embedded in JSON output)

## Runtime

**Environment:**
- Python >=3.12 (specified in `pyproject.toml`)
- Python 3.12 (pinned in `.python-version`)

**Package Manager:**
- uv (lockfile: `uv.lock` present)
- Build backend: hatchling (specified in `pyproject.toml`)

**Virtual Environment:**
- `.venv/` directory (git-ignored)

## Frameworks

**Core:**
- Pydantic 2.x - Data validation and serialization for all schemas (`pcmfg/models/schemas.py`)
- Click 8.x - CLI framework for `pcmfg` command (`pcmfg/cli.py`)
- Rich 13.x - Terminal UI: progress bars, tables, colored output (`pcmfg/cli.py`)

**LLM Integration:**
- openai >=1.0.0 - OpenAI API client (`pcmfg/llm/openai_client.py`)
- anthropic >=0.18.0 - Anthropic Claude API client (`pcmfg/llm/anthropic_client.py`)
- tenacity >=8.2.0 - Retry logic with exponential backoff for API calls (`pcmfg/llm/openai_client.py`, `pcmfg/llm/anthropic_client.py`)

**Data Processing:**
- pandas >=2.0.0 - CSV export, data manipulation (`pcmfg/visualization/plotter.py`)
- numpy >=1.24.0 - Numerical operations for time-series, feature extraction, clustering (`pcmfg/phase3/synthesizer.py`, `pcmfg/analysis/`)
- scikit-learn >=1.3.0 - Clustering (KMeans, DBSCAN, Agglomerative), PCA, t-SNE, StandardScaler (`pcmfg/analysis/clusterer.py`, `pcmfg/analysis/plotter.py`)

**Visualization:**
- matplotlib >=3.7.0 - Plotting emotion trajectories, gap analysis, cluster visualizations (`pcmfg/visualization/plotter.py`, `pcmfg/analysis/plotter.py`)

**Configuration:**
- python-dotenv >=1.0.0 - `.env` file loading (`pcmfg/cli.py`)
- pyyaml >=6.0 - YAML config file parsing (`pcmfg/config.py`)

**Testing:**
- pytest >=7.4.0 - Test runner (`pyproject.toml`)
- pytest-cov >=4.1.0 - Coverage reporting (`pyproject.toml`)

**Code Quality (dev):**
- black >=23.0.0 - Code formatter (line-length 88, target py312)
- ruff >=0.1.0 - Linter (rules: E, F, I, N, W, UP; line-length 88)
- mypy >=1.5.0 - Static type checking (disallow_untyped_defs, warn_return_any)
- pre-commit >=3.3.0 - Git hooks

**Build:**
- hatchling - Build backend (`pyproject.toml`)

## Key Dependencies

**Critical (required at runtime):**
- `pydantic` - All data models and validation; project cannot run without it
- `openai` or `anthropic` - At least one LLM client required for analysis
- `tenacity` - Retry logic for API resilience
- `click` - CLI entry point
- `rich` - Terminal output
- `matplotlib` - Visualization generation
- `numpy` - Time-series computation
- `pandas` - CSV export
- `pyyaml` - Config loading
- `python-dotenv` - Environment variable loading

**Optional (clustering features):**
- `scikit-learn` - Gracefully handled with `SKLEARN_AVAILABLE` flag in `pcmfg/analysis/clusterer.py` and `pcmfg/analysis/plotter.py`

**Dev-only:**
- `pytest`, `pytest-cov`, `black`, `ruff`, `mypy`, `pre-commit`

## Configuration

**Environment:**
- Three-layer config system: CLI args > YAML config > env vars > defaults
- Config file: `pcmfg_config.yaml` (searched in `./pcmfg_config.yaml`, `./pcmfg_config.yml`, `~/.pcmfg/config.yaml`)
- Env vars: `.env` file loaded via `python-dotenv` in `pcmfg/cli.py`
- Example configs: `pcmfg_config.example.yaml`, `.env.example`

**Build:**
- `pyproject.toml` - Project metadata, dependencies, tool configs (black, ruff, mypy, pytest)
- `uv.lock` - Dependency lockfile for uv package manager

**Key configs in `pyproject.toml`:**
- `[tool.black]`: line-length=88, target-version=py312
- `[tool.ruff]`: line-length=88, select=["E","F","I","N","W","UP"]
- `[tool.mypy]`: python_version=3.12, disallow_untyped_defs=true
- `[tool.pytest.ini_options]`: testpaths=["tests"], python_files=["test_*.py"]
- `[project.scripts]`: `pcmfg = "pcmfg.cli:main"` (CLI entry point)

## Platform Requirements

**Development:**
- Python 3.12+
- uv package manager (lockfile present)
- LLM API key (OpenAI or Anthropic)
- No database, no Docker, no external services required

**Production:**
- Local execution only (no server deployment)
- CLI tool: `pcmfg analyze <file>` or `pcmfg analyze-novel <dir>`
- Output: JSON files, PNG plots, CSV data, Markdown reports
- Checkpoint support: `.pcmfg_checkpoints/` directory for resumable analysis

**Target users:**
- Researchers analyzing romantic narratives
- CLI tool, not a web service

---

*Stack analysis: 2026-04-23*
