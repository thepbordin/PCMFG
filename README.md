# PCFMG

**Please Care My Feeling Graph** — A computational romance narrative mining system that extracts and visualizes emotional trajectories from romantic literature.

## Overview

PCFMG analyzes romantic narratives to extract emotional patterns and output them as raw time-series data. Using large language models and psychological frameworks (Russell's Circumplex Model and Plutchik's Wheel of Emotions), it transforms text into quantitative emotional data that reveals how romantic relationships evolve over time.

### What PCFMG Does

- **Phase 1 (World Builder)**: Uses LLM Agent 1 to identify main pairing, aliases, core conflict, world guidelines, and create a Mermaid relationship graph
- **Phase 2 (Emotion Extraction)**: Uses LLM Agent 2 in an iterative loop to extract directed emotions (9 base emotions) using a strict 1-5 scale
- **Phase 3 (Synthesis)**: Deterministic Python processing with forward-fill imputation and time-series generation

**Key Difference**: PCFMG outputs **raw 9 base emotions** as time-series, preserving maximum granularity for downstream analysis. It does NOT aggregate emotions into derived axes.

The result is a time-series visualization showing how each of the 9 emotions evolves throughout the narrative for both directions of the relationship.

## The Psychological Framework

PCFMG combines two established psychological models:

### Russell's Circumplex Model

Emotions exist in a 2D space defined by:
- **Valence** (horizontal): Pleasure/displeasure, positive/negative
- **Arousal** (vertical): Activation/deactivation, high/low energy

```
        High Arousal
             ↑
             |
    Fear ←—— + ——→ Joy
             |
Anger ←—— + ——→ Sadness
             ↓
        Low Arousal
```

### Plutchik's Wheel of Emotions

PCFMG uses an **Extended Plutchik Model** with 9 base emotions:
- Joy, Trust, Fear, Surprise, Sadness, Disgust, Anger, Anticipation, **Arousal**

The addition of **Arousal** (physical lust, romantic desire, sexual tension) specifically captures romantic intensity not present in the original model.

### Directed Emotions

A key innovation in PCFMG is the concept of "directed emotions" — emotions that one character feels **toward** another specific character. This creates the relational dimension necessary for analyzing romance.

**Example**: "Alice felt joy toward Bob" (positive valence, high arousal)

## Architecture

PCFMG operates as a 3-phase pipeline:

```
┌─────────────────────────────────────────────────────────────────┐
│                         INPUT TEXT                                │
│                    (Romantic Narrative)                           │
└────────────────────────────┬────────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────────┐
│                 PHASE 1: WORLD BUILDER (Agent 1)                  │
│                        (LLM-Based)                                │
│  • Identify main pairing (2 central characters)                  │
│  • Extract character aliases (nicknames, titles)                 │
│  • Extract core conflict (single sentence)                       │
│  • Generate world guidelines (discrete facts)                    │
│  • Create Mermaid relationship graph                             │
│  • Output: WorldBuilderOutput JSON                               │
└────────────────────────────┬────────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────────┐
│              PHASE 2: EMOTION EXTRACTION (Agent 2)                │
│                   (LLM Loop over chunks)                          │
│  • Split text into chunks (e.g., 500 words)                      │
│  • Skip chunks with no relevant characters (token efficiency)    │
│  • Extract directed emotions (A→B and B→A separately)            │
│  • Score 9 base emotions per direction (1-5 scale)              │
│  • Include justification quotes for validation                   │
│  • Output: List[ChunkAnalysis]                                   │
└────────────────────────────┬────────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────────┐
│                 PHASE 3: SYNTHESIS (Python)                       │
│                   (Deterministic - No LLM)                        │
│  • Forward fill missing emotion data (.ffill())                  │
│  • Build raw emotion time-series for A→B                         │
│  • Build raw emotion time-series for B→A                         │
│  • Generate visualization (9 emotions × 2 directions)            │
│  • Output: JSON + PNG                                             │
└────────────────────────────┬────────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────────┐
│                         OUTPUT                                    │
│  • JSON file with raw emotion time-series                        │
│  • Matplotlib visualization                                       │
│  • Optional: Statistical analysis report                         │
└─────────────────────────────────────────────────────────────────┘
```

## The 9 Base Emotions

PCFMG tracks 9 raw emotions, preserving maximum data granularity:

| Emotion | Description | Romantic Context |
|---------|-------------|------------------|
| **Joy** | Happiness, pleasure, delight | Moments of happiness in the relationship |
| **Trust** | Safety, reliance, vulnerability | Emotional openness and reliance |
| **Fear** | Panic, dread, terror, anxiety | Fear of rejection, loss, or revelation |
| **Surprise** | Astonishment, shock | Unexpected revelations or behaviors |
| **Sadness** | Grief, sorrow, despair | Loss, longing, disappointment |
| **Disgust** | Revulsion, aversion, contempt | Moral objections, physical aversion |
| **Anger** | Fury, rage, frustration | Conflicts, betrayals, misunderstandings |
| **Anticipation** | Looking forward to, expecting | Hope, planning, waiting |
| **Arousal** | Physical lust, romantic desire, sexual tension | Physical attraction, romantic tension |

### The 1-5 Scoring System

PCFMG uses a strict scoring rubric where **1 is the baseline/neutral**:

| Score | Description |
|-------|-------------|
| **1 (Baseline)** | No evidence of this emotion. Polite, functional, or entirely absent. This is the default. |
| **2 (Mild)** | A brief, subtle hint or low-energy flicker of the emotion. |
| **3 (Moderate)** | Clear, undeniable presence of the emotion. |
| **4 (Strong)** | Emotion heavily drives the character's actions or thoughts. High physiological arousal. |
| **5 (Extreme)** | Overwhelming, consuming saturation of the emotion. Maximum intensity. |

**Critical**: All emotions default to 1 unless explicit text proves otherwise. Normal conversation between characters typically scores all 1s.

### Directed Emotions (A→B vs B→A)

A fundamental principle in PCFMG is that **A→B is NOT the same as B→A**:

- Each direction is scored independently
- Only score what is explicitly in the text (dialogue, internal monologue, physical action)
- If A thinks about B while B is absent, ONLY output A→B
- Forward fill is used to carry forward last known emotional state when a character is absent

## Features

- **Multi-format Input**: Accepts plain text files, PDFs, and EPUBs
- **Configurable LLM Backend**: Supports OpenAI GPT and Anthropic Claude
- **Token Efficiency**: Skips LLM calls for chunks with no relevant characters
- **Forward Fill Imputation**: Handles missing emotion data gracefully
- **Bidirectional Tracking**: Tracks both A→B and B→A emotions separately
- **Justification Quotes**: Every score is backed by textual evidence
- **Export Options**: JSON, CSV, PNG, PDF formats

## Installation

### Prerequisites

- Python 3.12 or higher
- An API key for OpenAI or Anthropic Claude

### Setup with uv (Recommended)

```bash
# Clone the repository
git clone https://github.com/yourusername/PCMFG.git
cd PCMFG

# Install dependencies using uv
pip install uv
uv sync

# Copy environment template
cp .env.example .env

# Edit .env and add your API key
nano .env  # or use your preferred editor
```

### Setup with pip

```bash
# Clone the repository
git clone https://github.com/yourusername/PCMFG.git
cd PCMFG

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -e .
```

## Quick Start

### 1. Configure Environment

Create a `.env` file in the project root:

```bash
# Required: Choose one LLM provider
OPENAI_API_KEY=sk-your-openai-key-here
# or
ANTHROPIC_API_KEY=sk-ant-your-anthropic-key-here

# Optional: Model selection
LLM_MODEL=gpt-4o  # or claude-3-5-sonnet-20241022

# Optional: OpenAI Base URL (for custom endpoints)
OPENAI_BASE_URL=https://api.openai.com/v1

# Optional: Output directory
OUTPUT_DIR=./output
```

### 2. Analyze a Narrative

```bash
python main.py analyze "path/to/romance_novel.txt" --output ./results
```

### 3. View the Results

Results will be saved to the specified output directory:

```
results/
├── emotional_trajectory.json    # Raw time-series data
├── emotional_trajectory.png     # Visualization
├── emotional_directional_comparison.png
├── emotional_gap_analysis.png
└── clusters/                    # Clustering analysis (optional)
    ├── scene_cluster_scatter_2d.png
    ├── scene_cluster_timeline.png
    ├── scene_cluster_comparison.png
    └── scene_cluster_radar.png
```

To generate cluster analysis, run:

```bash
python run_clustering_example.py results/emotional_trajectory.json
```

## Usage Examples

### Basic Usage

```bash
# Analyze a single text file
python main.py analyze pride_and_prejudice.txt

# Specify output directory
python main.py analyze novel.txt --output ./my_results

# Use specific LLM model
python main.py analyze novel.txt --model claude-3-5-sonnet-20241022
```

### Advanced Options

```bash
# Process multiple files
python main.py analyze *.txt --batch

# Custom chunk length
python main.py analyze novel.txt --chunk-length 500

# Include statistical report
python main.py analyze novel.txt --stats

# Export to CSV instead of JSON
python main.py analyze novel.txt --format csv
```

### Python API Usage

```python
from pcmfg import PCMFGAnalyzer

# Initialize analyzer
analyzer = PCMFGAnalyzer(
    api_key="your-api-key",
    model="gpt-4o"
)

# Load and analyze text
with open("novel.txt", "r") as f:
    text = f.read()

# Run the 3-phase pipeline
results = analyzer.analyze(text)

# Access raw emotion time-series
a_to_b = results.timeseries["A_to_B"]
print(f"Joy trajectory: {a_to_b.Joy}")
print(f"Trust trajectory: {a_to_b.Trust}")

# Generate visualization
results.plot("output.png")

# Export data
results.export_json("trajectory.json")
```

## Output Format

### JSON Structure

```json
{
  "metadata": {
    "source": "novel.txt",
    "analysis_date": "2025-01-15T10:30:00Z",
    "model": "gpt-4o",
    "total_chunks": 47
  },
  "world_builder": {
    "main_pairing": ["Elizabeth Bennet", "Fitzwilliam Darcy"],
    "aliases": {
      "Elizabeth Bennet": ["Elizabeth", "Lizzy", "Miss Bennet", "Lizzie"],
      "Fitzwilliam Darcy": ["Darcy", "Mr. Darcy"]
    },
    "core_conflict": "Elizabeth's prejudice clashes with Darcy's pride until they overcome their misconceptions.",
    "world_guidelines": [
      "Fact 1: The Bennet family has no fortune and five daughters to marry off.",
      "Fact 2: Darcy initially appears proud and dismissive at the ball.",
      "Fact 3: Wickham spreads lies about Darcy cheating him out of an inheritance."
    ],
    "mermaid_graph": "graph TD\\n    Elizabeth[Elizabeth Bennet] -->|Initially dislikes| Darcy[Mr. Darcy]\\n    Darcy -->|Secretly attracted| Elizabeth"
  },
  "chunks": [
    {
      "chunk_id": 0,
      "position": 0.0,
      "chunk_main_pov": "Elizabeth",
      "characters_present": ["Elizabeth", "Darcy", "Bingley"],
      "directed_emotions": [
        {
          "source": "Elizabeth",
          "target": "Darcy",
          "scores": {
            "Joy": 1,
            "Trust": 1,
            "Fear": 1,
            "Surprise": 1,
            "Sadness": 1,
            "Disgust": 3,
            "Anger": 2,
            "Anticipation": 1,
            "Arousal": 1
          },
          "justification_quote": "She told the story...with great energy among her friends"
        }
      ],
      "scene_summary": "Elizabeth meets Darcy at the Meryton ball."
    }
  ],
  "timeseries": {
    "A_to_B": {
      "Joy": [1.0, 1.0, 1.5, 2.0, 3.0, 4.0],
      "Trust": [1.0, 1.0, 1.0, 1.5, 2.0, 3.5],
      "Fear": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
      "Surprise": [1.0, 2.0, 1.0, 1.0, 1.0, 1.0],
      "Sadness": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
      "Disgust": [3.0, 2.5, 2.0, 1.5, 1.0, 1.0],
      "Anger": [2.0, 2.0, 1.5, 1.0, 1.0, 1.0],
      "Anticipation": [1.0, 1.0, 1.0, 1.5, 2.0, 3.0],
      "Arousal": [1.0, 1.0, 1.0, 1.5, 2.5, 4.0]
    },
    "B_to_A": {
      "Joy": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
      "Trust": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
      "Fear": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
      "Surprise": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
      "Sadness": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
      "Disgust": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
      "Anger": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
      "Anticipation": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
      "Arousal": [3.0, 3.0, 3.5, 4.0, 4.5, 5.0]
    }
  }
}
```

### Visualization

The default output visualizes the raw 9 emotions over time for both relationship directions:

```
Elizabeth → Darcy                          Darcy → Elizabeth
┌─────────────────────────────┐            ┌─────────────────────────────┐
│ Joy     ────────            │            │ Joy     ────────            │
│ Trust   ────────            │            │ Trust   ────────            │
│ Fear    ────────            │            │ Fear    ────────            │
│ Surprise ────────           │            │ Surprise ────────           │
│ Sadness ────────            │            │ Sadness ────────            │
│ Disgust ▀▀▀▄▄▄────          │            │ Disgust ────────            │
│ Anger   ▄▄▄▄▄────           │            │ Anger   ────────            │
│ Anticip. ────────           │            │ Anticip. ────────           │
│ Arousal ────────            │            │ Arousal ▄▄▀▀▀▀▀▀            │
└─────────────────────────────┘            └─────────────────────────────┘
        Narrative Progression                      Narrative Progression

Note: Y-axis shows 1-5 scale where 1 is baseline/neutral
```

## Configuration Options

Configuration can be provided via command-line arguments or a config file (`pcmfg_config.yaml`):

```yaml
# pcmfg_config.yaml
llm:
  provider: "openai"  # or "anthropic"
  model: "gpt-4o"
  temperature: 0.3
  max_tokens: 4096
  base_url: null  # Optional: custom API endpoint (e.g., OpenRouter, Azure, local LLM)

processing:
  beat_detection: "automatic"  # automatic, length, or chapter
  beat_length: 500  # Target words per beat (for length mode)
  min_beat_length: 200  # Minimum words per beat
  max_chunk_tokens: 3000  # Max tokens per LLM chunk
  world_builder_sample_tokens: 8000  # Max tokens for world builder (strategic sampling)
  max_concurrency: 5  # Concurrent API calls

output:
  formats: ["json", "png"]  # Supported: json, csv, png
  include_stats: true
  dpi: 300
```

### World Builder Strategic Sampling

The `world_builder_sample_tokens` config controls how much text is sampled for world building. For long novels, the system uses **strategic sampling** from three sections:

- **Beginning** (~40%): Character introductions and initial setup
- **Middle** (~30%): Plot development and conflicts
- **End** (~30%): Resolution and relationship outcomes

This ensures the World Builder understands the complete narrative arc, not just the opening chapters.

```bash
python main.py analyze novel.txt --config pcmfg_config.yaml
```

## Token Efficiency

PCFMG includes a token-saving optimization: before calling Agent 2 for a chunk, it checks if the chunk contains any character names from the aliases list. If not, the LLM call is skipped entirely, saving API costs.

```python
def should_process_chunk(chunk_text: str, aliases: dict) -> bool:
    """Skip chunks with no relevant character names."""
    all_names = set()
    for main_name, alias_list in aliases.items():
        all_names.add(main_name)
        all_names.update(alias_list)
    
    return any(name.lower() in chunk_text.lower() for name in all_names)
```

## Forward Fill Imputation

When a character is absent from a scene, Agent 2 only outputs the direction for the present character. Phase 3 uses forward fill to carry forward the last known emotional state:

```python
def impute_missing_emotions(chunks: list) -> list:
    """
    Forward fill missing emotion directions.
    If chunk 5 has A→B but not B→A, carry forward B→A from chunk 4.
    """
    last_known = {}  # "source->target": DirectedEmotion
    
    for chunk in chunks:
        # Store current
        for emotion in chunk.directed_emotions:
            key = f"{emotion.source}->{emotion.target}"
            last_known[key] = emotion
        
        # Impute missing for main pairing
        # ...
```

## Clustering Analysis

PCFMG includes a clustering module for discovering emotional patterns and scene archetypes in narrative data. This enables deeper insights into the emotional structure of stories.

### Overview

The clustering module (`pcmfg.analysis`) provides tools to:

1. **Extract features** from emotional time-series data
2. **Cluster scenes** by their emotional profile
3. **Visualize** cluster assignments and emotional archetypes

### Feature Extraction

The `FeatureExtractor` transforms emotional time-series into feature vectors suitable for clustering:

| Feature Type | Description | Dimensions |
|--------------|-------------|------------|
| `RAW` | Direct emotion scores per chunk | 18 (9 emotions × 2 directions) |
| `DELTA` | Change between consecutive chunks | 18 |
| `STATISTICAL` | Mean, std, min, max, range aggregations | 90 |
| `COMBINED` | Raw + cumulative mean + trend | 54 |
| `WINDOWED` | Rolling window statistics | 36 |

### Clustering Algorithms

The `SceneClusterer` supports multiple algorithms:

| Algorithm | Best For | Parameters |
|-----------|----------|------------|
| **K-Means** | Spherical clusters, fast | `n_clusters`, `random_state` |
| **DBSCAN** | Outlier detection, irregular shapes | `eps`, `min_samples` |
| **Hierarchical** | Nested patterns, dendrograms | `n_clusters` |

### Usage

#### Command Line

```bash
# Run clustering on an analysis result (outputs to same directory)
python run_clustering_example.py output/emotional_trajectory.json

# Specify custom output directory
python run_clustering_example.py output/emotional_trajectory.json custom_output/
```

#### Python API

```python
from pcmfg.analysis import (
    FeatureExtractor,
    FeatureType,
    SceneClusterer,
    save_cluster_plots,
)
from pcmfg.models.schemas import AnalysisResult

# Load analysis result
result = AnalysisResult.parse_file("output/emotional_trajectory.json")

# Extract features
extractor = FeatureExtractor(feature_type=FeatureType.RAW)
features = extractor.extract(result)

# Find optimal number of clusters
clusterer = SceneClusterer(n_clusters=4)
scores = clusterer.find_optimal_k(features, k_range=(2, 6))
best_k = max(scores, key=scores.get)

# Cluster with optimal k
clusterer = SceneClusterer(n_clusters=best_k)
cluster_result = clusterer.cluster(features, interpret=True)

# Access results
print(f"Clusters: {cluster_result.n_clusters}")
print(f"Silhouette: {cluster_result.silhouette_score:.3f}")
print(cluster_result.cluster_interpretations)

# Save visualizations to the same output directory
save_cluster_plots(features, cluster_result, output_dir="output/clusters")
```

### Output Structure

Cluster visualizations are saved to a `clusters/` subdirectory within the analysis output:

```
output/
├── emotional_trajectory.json       # Raw time-series data
├── emotional_trajectory.png        # Emotion visualization
├── emotional_directional_comparison.png
├── emotional_gap_analysis.png
└── clusters/                       # Clustering output
    ├── scene_cluster_scatter_2d.png
    ├── scene_cluster_timeline.png
    ├── scene_cluster_comparison.png
    └── scene_cluster_radar.png
```

### Visualization Types

| Plot | Description |
|------|-------------|
| **Scatter 2D** | PCA projection of scenes, colored by cluster |
| **Timeline** | Cluster assignments over narrative progression |
| **Comparison** | Bar chart comparing emotions across clusters |
| **Radar** | Spider/radar charts showing cluster emotion profiles |

### Cluster Interpretation

The clusterer automatically generates human-readable interpretations based on dominant emotions:

```
Cluster 0: A_to_B: Trust(3.1), Arousal(3.1), Joy(3.0) | B_to_A: ...
Cluster 1: A_to_B: baseline/neutral | B_to_A: baseline/neutral
Cluster 2: A_to_B: Anger(4.0), Fear(3.0) | B_to_A: ...
```

### Silhouette Score

The silhouette score measures clustering quality (-1 to 1):

| Score | Quality |
|-------|---------|
| 0.7+ | Excellent |
| 0.5-0.7 | Good |
| 0.25-0.5 | Fair |
| <0.25 | Poor |

The `find_optimal_k()` method uses silhouette analysis to suggest the best number of clusters.

### Cross-Novel Comparison

Use `TrajectoryClusterer` to compare emotional trajectories across multiple novels:

```python
from pcmfg.analysis import TrajectoryClusterer

# Cluster multiple novels by their overall emotional profile
clusterer = TrajectoryClusterer(n_clusters=3)
results = [load_result(f) for f in novel_files]
trajectory_result = clusterer.cluster_multiple(results)
```

## Contributing

Contributions are welcome! Areas of interest:

- **Additional Emotion Models**: Integration with other psychological frameworks
- **New Input Formats**: Support for more file types (DOCX, HTML, etc.)
- **Visualization Enhancements**: Interactive plots, 3D visualizations
- **Performance**: Optimization for large texts
- **Testing**: Unit tests, integration tests, benchmarking

### Development Setup

```bash
# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Format code
black pcmfg/
ruff pcmfg/
```

## License

MIT License — See LICENSE file for details.

## Acknowledgments

- **James Russell** — Circumplex Model of Affect
- **Robert Plutchik** — Wheel of Emotions
- The computational literary analysis community

## Contact

For questions, issues, or suggestions, please open a GitHub issue.

---

**PCFMG**: Because every love story deserves a graph.
