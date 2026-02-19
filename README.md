# PCFMG

**Please Care My Feeling Graph** — A computational romance narrative mining system that extracts and visualizes emotional trajectories from romantic literature.

## Overview

PCFMG analyzes romantic narratives to extract emotional patterns and plot them on multi-dimensional axes. Using large language models and psychological frameworks (Russell's Circumplex Model and Plutchik's Wheel of Emotions), it transforms text into quantitative emotional data that reveals how romantic relationships evolve over time.

### What PCFMG Does

- **Phase 1**: Uses LLMs to identify story beats, main pairing, aliases, world guidelines, and extract directed emotions (9 base emotions) using a strict 1-5 scale
- **Phase 2**: Normalizes emotions based on the 1-5 baseline system (1=neutral, 5=extreme)
- **Phase 3**: Computes four romance axes (Intimacy, Passion, Hostility, Anxiety) from the base emotion scores

The result is a time-series visualization showing how each relationship dimension changes throughout the narrative.

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
│                        PHASE 1: Extraction                        │
│                     (LLM-Based Analysis)                          │
│  • Split text into story beats/scenes                            │
│  • Identify characters and their relationships                   │
│  • Extract directed emotions for each beat                       │
│  • Output: Raw emotion list with metadata                        │
└────────────────────────────┬────────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────────┐
│                        PHASE 2: Normalization                     │
│                    (Baseline Calibration)                         │
│  • Normalize all emotions to 1-5 scale                           │
│  • Positive emotions: 1 (neutral) to 5 (euphoric)               │
│  • Negative emotions: -1 (neutral) to -5 (agonizing)            │
│  • Apply valence adjustment based on emotion type                │
└────────────────────────────┬────────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────────┐
│                        PHASE 3: Axis Mapping                      │
│                   (Romance Dimensions)                            │
│  • Map normalized emotions to 4 axes:                            │
│    - Intimacy: Closeness, trust, vulnerability                  │
│    - Passion: Desire, excitement, romantic intensity             │
│    - Hostility: Anger, resentment, conflict                      │
│    - Anxiety: Fear, uncertainty, tension                         │
│  • Generate time-series data for visualization                   │
└────────────────────────────┬────────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────────┐
│                         OUTPUT                                    │
│  • JSON file with time-series data                               │
│  • Matplotlib visualization (4-panel plot)                       │
│  • Optional: Statistical analysis report                         │
└─────────────────────────────────────────────────────────────────┘
```

## The Four Romance Axes

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

### Romance Axes Mapping

| Axis | Description | Base Emotions (Computation) |
|------|-------------|------------------|
| **Intimacy** | Emotional closeness, trust, vulnerability, connection | Primary: Trust, Joy |
| **Passion** | Romantic desire, excitement, physical attraction, intensity | Primary: Arousal, Anticipation, Joy |
| **Hostility** | Anger, resentment, conflict, dislike, antagonism | Primary: Anger, Disgust, Sadness |
| **Anxiety** | Fear, uncertainty, tension, worry, insecurity | Primary: Fear, Surprise (negative), Sadness |

Each emotion is mapped to one or more axes based on:
1. Its valence (positive/negative)
2. Its arousal level (high/low)
3. Its relevance to romantic relationships

## Features

- **Multi-format Input**: Accepts plain text files, PDFs, and EPUBs
- **Configurable LLM Backend**: Supports OpenAI GPT and Anthropic Claude
- **Batch Processing**: Analyze multiple narratives in parallel
- **Interactive Visualization**: Zoom, pan, and explore emotional trajectories
- **Export Options**: JSON, CSV, PNG, PDF formats
- **Statistical Analysis**: Correlation matrices, trend analysis, anomaly detection

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
├── emotional_trajectory.png     # 4-panel visualization
└── analysis_report.md           # Human-readable summary
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

# Custom beat detection (scene boundaries)
python main.py analyze novel.txt --beat-length 500

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
          "justification_quote": "She told the story...with great energy among her friends; for she had a lively, playful disposition, which delighted in anything ridiculous."
        }
      ],
      "scene_summary": "Elizabeth meets Darcy at the Meryton ball and takes offense at his refusal to dance."
    }
  ],
  "axes": {
    "intimacy": [1.0, 1.2, 1.5, 2.1, 3.0, 3.5],
    "passion": [1.0, 1.0, 1.2, 1.8, 2.5, 4.2],
    "hostility": [4.0, 3.5, 2.5, 1.5, 0.8, 0.3],
    "anxiety": [1.0, 1.5, 2.2, 2.5, 1.2, 0.5]
  }
}
```

### Visualization

The default output is a 4-panel matplotlib figure showing each axis over time:

```
┌─────────────────────┬─────────────────────┐
│    Intimacy         │    Passion          │
│  ─────────────      │  ─────────────      │
│    5  ┌───┐         │    5       ┌─┐      │
│    4  │   │         │    4       │ │      │
│    3  │   └─┐       │    3   ┌───┘ │      │
│    2  │     └───┐   │    2   │     └──┐   │
│    1  ──────────   │    1───┘        └───│
│       ──────       │       ──────       │
├─────────────────────┼─────────────────────┤
│    Hostility        │    Anxiety          │
│  ─────────────      │  ─────────────      │
│    5  ┌─┐           │    5   ┌───┐        │
│    4  │ └──┐        │    4   │   │        │
│    3  │    └──┐     │    3   │   └──┐     │
│    2  │       └──┐  │    2   │      └──┐  │
│    1  ─────────    │    1───┘         └──│
│       ──────       │       ──────       │
└─────────────────────┴─────────────────────┘

Note: Y-axis shows 1-5 scale where 1 is baseline/neutral
```

## Configuration Options

Configuration can be provided via command-line arguments or a config file:

```yaml
# pcmfg_config.yaml
llm:
  provider: "openai"  # or "anthropic"
  model: "gpt-4o"
  temperature: 0.3
  max_tokens: 4096

processing:
  beat_detection: "automatic"  # or "length", "chapter"
  beat_length: 500  # words per beat
  min_beat_length: 200

output:
  formats: ["json", "png", "csv"]
  include_stats: true
  dpi: 300
```

```bash
python main.py analyze novel.txt --config pcmfg_config.yaml
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

For questions, issues, or suggestions, please open a GitHub issue or contact [your-email@example.com].

---

**PCFMG**: Because every love story deserves a graph.
