# PCMFG Runner Guide

## Quick Start

```bash
# 1. Install dependencies
uv sync

# 2. Set up API key
cp .env.example .env
# Edit .env with your OPENAI_API_KEY or ANTHROPIC_API_KEY

# 3. Analyze a single file
python main.py analyze novels/you/story.txt -o output/you -f json --stats

# 4. Analyze a novel directory (multi-chapter)
python main.py analyze-novel novels/first -o output/first
```

---

## Configuration

All config lives in `pcmfg_config.yaml`. CLI flags override config values.

### LLM Provider (`llm` section)

```yaml
llm:
  provider: "openai"          # "openai" or "anthropic"
  model: "gpt-4o"             # any model name
  temperature: 0.3             # 0.0 = deterministic, 1.0 = creative
  max_tokens: 4096
  base_url: null               # custom endpoint (OpenRouter, vLLM, etc.)
```

### Chunking (`processing` section)

Five chunking modes via `beat_detection`:

| Mode | Description | Config keys |
|---|---|---|
| `automatic` | Chapter-based first, falls back to length | (none) |
| `chapter` | Split on "Chapter N" markers | `max_chunk_tokens` |
| `length` | Split by word count | `beat_length`, `min_beat_length` |
| `paragraph` | One paragraph per chunk | `max_chunk_tokens` |
| `paragraphs` | Group N paragraphs per chunk | `paragraphs_per_chunk`, `max_chunk_tokens` |

```yaml
processing:
  beat_detection: "paragraphs"       # chunking mode
  paragraphs_per_chunk: 3            # for "paragraphs" mode (2-10)
  beat_length: 500                   # for "length" mode (target words)
  min_beat_length: 200               # minimum words per chunk
  max_chunk_tokens: 3000             # hard cap per LLM call
```

### Emotion Carry-Over

When enabled, each chunk's LLM prompt includes the previous chunk's emotion scores, POV, and scene summary. This produces smoother emotional trajectories.

```yaml
processing:
  emotion_carryover: true    # include previous chunk state as context
```

**Trade-off**: Carry-over requires sequential processing (one LLM call at a time). Disable for parallel speed:

```yaml
processing:
  emotion_carryover: false   # chunks processed in parallel
  max_concurrency: 5         # parallel API calls
```

### Output (`output` section)

```yaml
output:
  formats: ["json", "png"]   # "json", "csv", "png"
  include_stats: true        # generate analysis_report.md
  dpi: 300                   # image resolution
```

### World Builder Hint

For non-romantic stories, guide the world builder:

```yaml
processing:
  world_builder_hint: "This is a dark thriller, not a romance. Focus on power dynamics and fear."
```

---

## Running

### Single file analysis

```bash
python main.py analyze INPUT_FILE -o OUTPUT_DIR [options]

# Basic
python main.py analyze novels/you/story.txt -o output/you

# With stats report and plots
python main.py analyze novels/you/story.txt -o output/you -f json --stats

# CSV output, no plots
python main.py analyze novels/you/story.txt -o output/you -f csv --no-plot

# Override model/provider
python main.py analyze novels/you/story.txt -o output/you -p openai -m gpt-4o-mini

# Use a different config file
python main.py analyze novels/you/story.txt -o output/you -c my_config.yaml
```

### Novel directory analysis (with checkpoints)

```bash
python main.py analyze-novel NOVEL_DIR [options]

# Full novel
python main.py analyze-novel novels/first -o output/first

# Chapter range
python main.py analyze-novel novels/first -o output/first --start 1 --end 50

# Start fresh (ignore checkpoint)
python main.py analyze-novel novels/first -o output/first --no-resume
```

### Checkpoints

Resume interrupted runs automatically with `analyze-novel`. Checkpoints are stored in `.pcmfg_checkpoints/`.

```bash
# List all checkpoints
python main.py checkpoints
```

### Parallel vs Sequential

| Config | Processing | Speed | Use case |
|---|---|---|---|
| `emotion_carryover: true` | Sequential | ~30s/chunk | Best quality, smooth trajectories |
| `emotion_carryover: false` | Parallel (up to `max_concurrency`) | ~6s/chunk with 5 workers | Quick exploration, large corpora |

Switch between them by editing `pcmfg_config.yaml`:

```yaml
# Fast parallel mode
processing:
  emotion_carryover: false
  max_concurrency: 5

# Quality sequential mode
processing:
  emotion_carryover: true
```

### Running Multiple Novels in Parallel (shell)

When `emotion_carryover: false`, run multiple novels simultaneously:

```bash
python main.py analyze novels/you/story.txt -o output/you &
python main.py analyze novels/titanic/story.txt -o output/titanic &
python main.py analyze novels/lalaland/story.txt -o output/lalaland &
wait
```

When `emotion_carryover: true`, run them sequentially or use `nohup`:

```bash
nohup python main.py analyze novels/you/story.txt -o output/you > /tmp/you.log 2>&1 &
```

### Debug Mode

```bash
python main.py --debug analyze novels/you/story.txt -o output/you
```

---

## Input Format

Place stories as `.txt` or `.md` files in `novels/<name>/story.txt`.

```
novels/
  you/story.txt
  titanic/story.txt
  lalaland/story.txt
  the-three-little-pigs/story.md
```

For multi-chapter novels, use the `analyze-novel` command with a directory of chapter files.

---

## Output Structure

```
output/<novel_name>/
  emotional_trajectory.json              # Full analysis result (all data)
  emotional_trajectory.png               # Side-by-side A->B and B->A plots
  emotional_directional_comparison.png    # Both directions overlaid
  emotional_gap_analysis.png             # A-B emotional difference
  analysis_report.md                     # Statistical summary with trends
```

### Key JSON Fields

- `metadata` — source, model, chunk count, date
- `world_builder` — main pairing, aliases, core conflict, world guidelines
- `chunks[]` — per-chunk analysis with directed emotions and justifications
- `timeseries` — A_to_B and B_to_A raw emotion time-series
- `axes` — computed romance axes (intimacy, passion, hostility, anxiety)

---

## Post-Analysis: DTW Clustering

After analyzing 2+ novels, run cross-narrative DTW clustering:

```python
from pcmfg.models.schemas import AnalysisResult
from pcmfg.analysis import NarrativeNormalizer, DTWClusterer
import json

# Load results
results = []
for path in ["output/you/emotional_trajectory.json", "output/titanic/emotional_trajectory.json"]:
    with open(path) as f:
        results.append(AnalysisResult(**json.load(f)))

# Normalize to uniform 100-point grid
normalizer = NarrativeNormalizer(n_points=100)
trajectories = normalizer.normalize_all(results)

# Cluster by emotional arc shape
clusterer = DTWClusterer(n_clusters=3, metric="dtw", sakoe_chiba_radius=0.2)
cluster_result = clusterer.cluster(trajectories)

print(cluster_result.assignments)       # which narrative -> which cluster
print(cluster_result.silhouette_score)  # cluster quality (-1 to 1)
print(cluster_result.distance_matrix)   # pairwise DTW distances
```

Or run via the existing script for per-novel scene clustering:

```bash
python run_clustering_example.py output/you/emotional_trajectory.json output/you/clusters
```

---

## Per-Novel Scene Clustering

Cluster individual scenes (chunks) within a single narrative by emotional profile:

```bash
python run_clustering_example.py output/you/emotional_trajectory.json [output_dir]
```

Outputs:
- `scene_cluster_scatter_2d.png` — PCA scatter of scenes
- `scene_cluster_timeline.png` — cluster assignments over narrative position
- `scene_cluster_comparison.png` — average emotion bars per cluster
- `scene_cluster_radar.png` — radar chart per cluster

---

## Troubleshooting

| Problem | Solution |
|---|---|
| `No such option: --debug` | Put `--debug` before the command: `python main.py --debug analyze ...` |
| `Analysis failed: 'DirectedEmotionScores' object is not subscriptable` | Run from latest source (fixed in emotion_extractor.py) |
| Chapters not detected | Text needs "Chapter N" markers. Use `beat_detection: "length"` or `"paragraphs"` instead |
| Too many chunks | Increase `beat_length` or `paragraphs_per_chunk` |
| Too few chunks | Decrease `beat_length` or `paragraphs_per_chunk` |
| API rate limits | Decrease `max_concurrency` or increase `temperature` |
| Out of memory on long texts | Decrease `world_builder_sample_tokens` |
| Titanic / large texts timeout | Use `nohup` and `emotion_carryover: false` for parallel mode |
