# PCFMG Technical Guide for AI Assistants

**Purpose**: This document provides technical guidance for AI assistants (Claude, GPT, etc.) working on the PCFMG codebase.

---

## Project Mission

PCFMG (Please Care My Feeling Graph) is a computational romance narrative mining system that:

1. **Extracts** world context and directed emotions using two specialized LLM agents:
   - Agent 1 (World Builder): Identifies main pairing, aliases, world guidelines, and relationship graph
   - Agent 2 (Base Emotion Extractor): Scores 9 base emotions using strict 1-5 scale
2. **Normalizes** all emotions to a quantifiable 1-5 baseline where 1=neutral
3. **Computes** four romance-specific axes from base emotions: Intimacy, Passion, Hostility, Anxiety

The goal is to transform unstructured romantic narratives into structured time-series data that reveals how relationships evolve over time.

---

## Architecture Deep-Dive

### The 3-Phase Pipeline

```
INPUT TEXT (romantic narrative)
    ↓
┌─────────────────────────────────────────────┐
│ PHASE 1: EXTRACTION                          │
│ - Story beat detection                       │
│ - Character identification                   │
│ - Directed emotion extraction (LLM)          │
│ Output: List of (beat, emotions) tuples      │
└─────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────┐
│ PHASE 2: NORMALIZATION                      │
│ - Valence determination (pos/neg)            │
│ - Intensity scaling to 1-5                   │
│ - Baseline calibration                       │
│ Output: Normalized emotion values            │
└─────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────┐
│ PHASE 3: AXIS MAPPING                       │
│ - Map to Intimacy/Passion/Hostility/Anxiety  │
│ - Aggregate by beat                          │
│ - Generate time-series                       │
│ Output: JSON + visualization                 │
└─────────────────────────────────────────────┘
```

### Phase 1: Extraction (Two-Agent LLM Architecture)

**Input**: Raw text of a romantic narrative

PCFMG uses **two specialized agents** for extraction:

#### Agent 1: World Builder

**Purpose**: Extract narrative scaffolding and world context

**Tasks**:
1. Identify the **main pairing** (two central characters of the romance)
2. Extract **aliases** for all main and secondary characters (nicknames, titles, last names)
3. Generate **world guidelines** - discrete facts about core conflict, status quo, and backstory
4. Create **Mermaid.js graph** mapping relationships between main pairing and key secondary characters

**Output Schema**:
```json
{
  "main_pairing": ["Full Name 1", "Full Name 2"],
  "aliases": {
    "Full Name 1": ["Alias A", "Alias B", "Title"],
    "Full Name 2": ["Alias C", "Alias D"]
  },
  "world_guidelines": [
    "Fact 1: They were forced into a political marriage.",
    "Fact 2: Character A lost his memory in an accident.",
    "Fact 3: Character B is terrified of Character A regaining his memory."
  ],
  "mermaid_graph": "graph TD\\n    A[Character A] -->|Married| B[Character B]\\n    B -->|Afraid of| A"
}
```

#### Agent 2: Base Emotion Extractor

**Purpose**: Extract directed emotions scored on 9 base emotions

**Input from Agent 1**:
- Main pairing and aliases
- World guidelines (for context)
- Mermaid graph (optional visualization)

**Tasks**:
1. Identify the **chunk_main_pov** (whose perspective we're in)
2. List **characters_present** in the scene
3. Score **directed emotions** (Source → Target) between main pairing ONLY
4. Provide **justification quotes** for the highest scores

**Critical Rules**:
- A → B is NOT the same as B → A
- Only score directions with **explicit textual evidence** (dialogue, internal monologue, physical action)
- If A thinks about B while B is absent, ONLY output A → B direction
- **Default all scores to 1** unless text proves otherwise

**Output Schema**:
```json
{
  "chunk_id": 0,
  "chunk_main_pov": "Character Name",
  "characters_present": ["Character A", "Character B"],
  "directed_emotions": [
    {
      "source": "Name of Source",
      "target": "Name of Target",
      "scores": {
        "Joy": <int 1-5>,
        "Trust": <int 1-5>,
        "Fear": <int 1-5>,
        "Surprise": <int 1-5>,
        "Sadness": <int 1-5>,
        "Disgust": <int 1-5>,
        "Anger": <int 1-5>,
        "Anticipation": <int 1-5>,
        "Arousal": <int 1-5>
      },
      "justification_quote": "Exact text quote proving the highest active scores"
    }
  ],
  "scene_summary": "One brief sentence summarizing the action"
}
```

### Phase 2: Normalization

**Input**: Raw emotion scores (1-5) from Agent 2

**Note**: In PCFMG's architecture, Agent 2 already outputs normalized 1-5 scores. Phase 2 is primarily a **validation and quality check** phase rather than mathematical normalization.

**Tasks**:
1. **Validate** that all scores are within 1-5 range
2. **Check** that justification quotes support the scores
3. **Flag** anomalies (e.g., all 5s, inconsistent patterns)
4. **Aggregate** bidirectional scores for each chunk

**The 1-5 Baseline System**:
- **1 (Baseline)**: No evidence of emotion. Polite, functional, or absent. **This is the default.**
- **2 (Mild)**: Brief, subtle hint or low-energy flicker
- **3 (Moderate)**: Clear, undeniable presence
- **4 (Strong)**: Heavily drives actions/thoughts, high physiological arousal
- **5 (Extreme)**: Overwhelming, consuming saturation

**Quality Checks**:
```python
def validate_emotion_scores(emotion_data: dict) -> bool:
    """Validate that emotion scores follow the strict rubric."""
    for emotion_direction in emotion_data["directed_emotions"]:
        scores = emotion_direction["scores"]
        for emotion, score in scores.items():
            if not isinstance(score, int) or score < 1 or score > 5:
                return False
            # Warn if too many 5s (possible LLM hallucination)
            if score == 5:
                log_high_score_warning(emotion, emotion_direction["justification_quote"])
    return True
```

**Bidirectional Aggregation**:
```python
def aggregate_bidirectional_emotions(
    a_to_b: dict[str, int],
    b_to_a: dict[str, int]
) -> dict[str, float]:
    """Aggregate A→B and B→A scores into overall chunk emotion state."""
    aggregated = {}
    for emotion in EMOTION_LIST:
        # Average the two directions
        a_score = a_to_b.get(emotion, 1)
        b_score = b_to_a.get(emotion, 1)
        aggregated[emotion] = (a_score + b_score) / 2.0
    return aggregated
```

### Phase 3: Axis Mapping

**The Four Romance Axes**:

| Axis | Base Emotions (Computation) | Formula |
|------|---------------------------|---------|
| **Intimacy** | Trust, Joy | `(Trust + Joy) / 2` |
| **Passion** | Arousal, Anticipation, Joy | `(Arousal + Anticipation + Joy) / 3` |
| **Hostility** | Anger, Disgust, Sadness | `(Anger + Disgust + Sadness) / 3` |
| **Anxiety** | Fear, Surprise, Sadness | `(Fear + Surprise + Sadness) / 3` |

**Mapping Strategy**:
1. Use the aggregated emotion scores from Phase 2
2. Apply weighted averaging formulas for each axis
3. Handle edge cases (e.g., missing character in scene)

**Implementation**:
```python
from typing import TypedDict

class EmotionScores(TypedDict):
    Joy: int
    Trust: int
    Fear: int
    Surprise: int
    Sadness: int
    Disgust: int
    Anger: int
    Anticipation: int
    Arousal: int

class AxisScores(TypedDict):
    intimacy: float
    passion: float
    hostility: float
    anxiety: float

def compute_axes(emotions: EmotionScores) -> AxisScores:
    """
    Compute the four romance axes from base emotion scores.

    Args:
        emotions: Dictionary of 9 base emotion scores (1-5)

    Returns:
        Dictionary of 4 axis scores (1-5)
    """
    return {
        "intimacy": (emotions["Trust"] + emotions["Joy"]) / 2.0,
        "passion": (emotions["Arousal"] + emotions["Anticipation"] + emotions["Joy"]) / 3.0,
        "hostility": (emotions["Anger"] + emotions["Disgust"] + emotions["Sadness"]) / 3.0,
        "anxiety": (emotions["Fear"] + emotions["Surprise"] + emotions["Sadness"]) / 3.0,
    }

# Handle bidirectional case (A→B vs B→A)
def compute_relationship_axes(
    a_to_b: EmotionScores,
    b_to_a: EmotionScores
) -> tuple[AxisScores, AxisScores]:
    """
    Compute axes for both directions of the relationship.

    Returns:
        Tuple of (A_feels_toward_B, B_feels_toward_A)
    """
    return compute_axes(a_to_b), compute_axes(b_to_a)
```

---

## Key Concepts

### Directed Emotions

A **directed emotion** is an emotion that one character (source) feels toward another specific character (target).

**Critical Property**: Directed emotions enable relationship modeling. Without the directionality, we only have emotional atmosphere, not relationship dynamics.

**Examples**:
- "Alice felt joy toward Bob" → Directed (Alice→Bob)
- "The room was filled with tension" → Not directed (atmosphere)
- "She hated him" → Directed (implicit she→him)

### The 9 Base Emotions (Extended Plutchik Model)

PCFMG uses Plutchik's 8 emotions plus an added 9th emotion specific to romantic literature:

1. **Joy** (Happiness, pleasure, delight)
2. **Trust** (Safety, reliance, vulnerability)
3. **Fear** (Panic, dread, terror, anxiety)
4. **Surprise** (Astonishment, shock)
5. **Sadness** (Grief, sorrow, despair)
6. **Disgust** (Revulsion, aversion, contempt)
7. **Anger** (Fury, rage, frustration)
8. **Anticipation** (Looking forward to, expecting, plotting)
9. **Arousal** (Physical lust, romantic desire, sexual tension) ← **Added for romance**

**Why Arousal?** Romantic narratives have a unique dimension of physical/sexual tension that standard emotion models don't capture. This distinguishes "close friendship" from "romantic passion."

**Emotion Scoring** (Strict 1-5 Scale):
- All emotions default to **1 (baseline)**
- Score 2-5 only with explicit textual evidence
- Normal conversation = all 1s

### Baseline Logic

**The "Baseline is 1" Rule**:

In PCFMG, **1 is the neutral baseline**, not 0. This is critical:

- **1**: No evidence of emotion. Polite, functional interaction. This is the DEFAULT.
- **2-5**: Increasing intensity of emotion

**Why 1 instead of 0?**
- Romantic narratives inherently have emotional charge
- "0" would imply complete absence, which is rare in social interactions
- "1" represents "baseline human presence" - characters acknowledge each other without emotional loading

**Example**:
```
"Good morning," she said.    → All emotions = 1 (baseline)
"Good morning!" she beamed.  → Joy = 3, others = 1
"Good morning," she spat.    → Anger = 3, Disgust = 2, others = 1
```

---

## Code Organization

### Expected Directory Structure

```
PCMFG/
├── pcmfg/
│   ├── __init__.py
│   ├── cli.py                 # Command-line interface
│   ├── analyzer.py            # Main orchestration
│   ├── phase1/
│   │   ├── __init__.py
│   │   ├── beat_detector.py   # Text segmentation
│   │   ├── character_extractor.py
│   │   └── emotion_extractor.py  # LLM-based extraction
│   ├── phase2/
│   │   ├── __init__.py
│   │   ├── normalizer.py
│   │   └── valence_classifier.py
│   ├── phase3/
│   │   ├── __init__.py
│   │   ├── axis_mapper.py
│   │   └── aggregator.py
│   ├── visualization/
│   │   ├── __init__.py
│   │   └── plotter.py
│   ├── models/
│   │   ├── __init__.py
│   │   └── schemas.py         # Pydantic models
│   └── utils/
│       ├── __init__.py
│       └── text_processing.py
├── tests/
│   ├── test_phase1.py
│   ├── test_phase2.py
│   ├── test_phase3.py
│   └── fixtures/
├── main.py
├── pyproject.toml
├── README.md
└── CLAUDE.md
```

### Module Responsibilities

**`analyzer.py`**: Orchestrates the full pipeline
```python
class PCMFGAnalyzer:
    def __init__(self, llm_client, config):
        self.phase1 = Phase1Extractor(llm_client)
        self.phase2 = Phase2Normalizer()
        self.phase3 = Phase3Mapper()

    def analyze(self, text: str) -> AnalysisResult:
        beats = self.phase1.extract_beats(text)
        emotions = self.phase1.extract_emotions(beats)
        normalized = self.phase2.normalize(emotions)
        axes = self.phase3.map_to_axes(normalized)
        return AnalysisResult(axes)
```

**`schemas.py`**: Pydantic models for validation
```python
from pydantic import BaseModel, Field
from typing import Dict, List

class DirectedEmotionScores(BaseModel):
    """9 base emotion scores for a directed emotion (Source → Target)."""
    Joy: int = Field(ge=1, le=5)
    Trust: int = Field(ge=1, le=5)
    Fear: int = Field(ge=1, le=5)
    Surprise: int = Field(ge=1, le=5)
    Sadness: int = Field(ge=1, le=5)
    Disgust: int = Field(ge=1, le=5)
    Anger: int = Field(ge=1, le=5)
    Anticipation: int = Field(ge=1, le=5)
    Arousal: int = Field(ge=1, le=5)

class DirectedEmotion(BaseModel):
    """A directed emotion with scores and justification."""
    source: str
    target: str
    scores: DirectedEmotionScores
    justification_quote: str

class ChunkAnalysis(BaseModel):
    """Analysis result for a text chunk."""
    chunk_id: int
    chunk_main_pov: str
    characters_present: List[str]
    directed_emotions: List[DirectedEmotion]
    scene_summary: str

class WorldBuilderOutput(BaseModel):
    """Output from Agent 1 (World Builder)."""
    main_pairing: List[str]
    aliases: Dict[str, List[str]]
    world_guidelines: List[str]
    mermaid_graph: str

class AxisValues(BaseModel):
    """Computed romance axis values."""
    intimacy: float = Field(ge=1, le=5)
    passion: float = Field(ge=1, le=5)
    hostility: float = Field(ge=1, le=5)
    anxiety: float = Field(ge=1, le=5)

class AnalysisResult(BaseModel):
    """Complete analysis result."""
    metadata: Dict[str, str | int]
    world_builder: WorldBuilderOutput
    chunks: List[ChunkAnalysis]
    axes: Dict[str, List[float]]
```

---

## Development Guidelines

### Code Style

- Use **type hints** everywhere
- Follow PEP 8 (4 spaces, max line length 88)
- Use **dataclasses** for simple data containers
- Use **Pydantic** for validation
- Prefer **composition** over inheritance

### Error Handling

```python
# DO: Specific exceptions
class EmotionExtractionError(Exception):
    """Raised when LLM fails to extract emotions."""

class TextTooShortError(ValueError):
    """Raised when input text is too short."""

# DON'T: Bare except
try:
    risky_operation()
except:  # Bad
    pass

# DO: Explicit error recovery
try:
    result = llm_call()
except RateLimitError:
    time.sleep(60)
    result = llm_call()
except APIError as e:
    raise EmotionExtractionError(f"LLM API failed: {e}")
```

### LLM Integration

**Retry Logic**:
```python
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10)
)
def call_llm(prompt: str) -> str:
    # Implementation
    pass
```

**Token Management**:
```python
def estimate_tokens(text: str) -> int:
    return len(text.split()) * 1.3  # Rough estimate

def chunk_text(text: str, max_tokens: int = 3000) -> list[str]:
    """Split text into chunks that fit in LLM context."""
    words = text.split()
    chunks = []
    current_chunk = []
    current_count = 0

    for word in words:
        current_chunk.append(word)
        current_count += 1.3
        if current_count >= max_tokens:
            chunks.append(" ".join(current_chunk))
            current_chunk = []
            current_count = 0

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks
```

---

## Testing Approach

### Unit Tests

```python
import pytest
from pcmfg.phase2.normalizer import EmotionNormalizer

def test_normalize_positive_emotion():
    normalizer = EmotionNormalizer()
    result = normalizer.normalize("joy")
    assert result > 0
    assert result <= 5

def test_normalize_negative_emotion():
    normalizer = EmotionNormalizer()
    result = normalizer.normalize("anger")
    assert result < 0
    assert result >= -5

@pytest.mark.parametrize("emotion,expected_range", [
    ("joy", (3, 5)),
    ("contentment", (1, 3)),
    ("anger", (-5, -3)),
    ("annoyance", (-3, -1)),
])
def test_emotion_falls_in_expected_range(emotion, expected_range):
    normalizer = EmotionNormalizer()
    result = normalizer.normalize(emotion)
    assert expected_range[0] <= result <= expected_range[1]
```

### Integration Tests

```python
def test_full_pipeline_with_simple_text():
    text = "Alice loved Bob. Bob hated Alice."
    analyzer = PCMFGAnalyzer(mock_llm_client)

    result = analyzer.analyze(text)

    assert len(result.axes) == 4
    assert "intimacy" in result.axes
    assert "passion" in result.axes
    assert "hostility" in result.axes
    assert "anxiety" in result.axes
```

### Fixtures

```python
# tests/fixtures/sample_texts.py
SIMPLE_ROMANCE = """
They met at the ball. She felt drawn to him immediately.
He thought she was beautiful. They danced all night.
"""

COMPLEX_ROMANCE = """
[Full page from a published novel]
"""
```

---

## Important Constraints and Safeguards

### 1. LLM Reliability

**Issue**: LLMs can hallucinate or produce inconsistent results.

**Safeguards**:
- Use structured JSON output with validation
- Implement confidence thresholds
- Cross-validate with multiple runs for critical analyses
- Provide fallback heuristics when LLM fails

```python
def extract_emotions_with_fallback(text: str) -> list[DirectedEmotion]:
    try:
        return llm_extract_emotions(text)
    except LLMError:
        # Fallback to keyword-based detection
        return keyword_extract_emotions(text)
```

### 2. Text Length Limits

**Issue**: LLMs have token limits.

**Safeguards**:
- Chunk text intelligently (by scenes, not arbitrary cuts)
- Track beat position for reassembly
- Warn user if text exceeds practical limits

### 3. Cultural Bias

**Issue**: Emotion labels are culturally dependent.

**Safeguards**:
- Document Western psychological framework assumption
- Allow custom emotion mappings
- Support multiple language models (different cultural training)

### 4. Privacy

**Issue**: User texts may be sensitive.

**Safeguards**:
- Never log input texts
- Clear .env warnings about API key handling
- Local-only processing when possible

---

## Common Tasks Reference

### Adding a New LLM Provider

```python
# 1. Create new client class
class AnthropicClient(BaseLLMClient):
    def call(self, prompt: str) -> str:
        # Implementation
        pass

# 2. Register in factory
LLM_PROVIDERS = {
    "openai": OpenAIClient,
    "anthropic": AnthropicClient,
}

# 3. Update CLI
# cli.py: Add --provider option
```

### Adding a New Output Format

```python
# 1. Create exporter function
def export_csv(result: AnalysisResult, path: str):
    import pandas as pd
    df = pd.DataFrame(result.axes)
    df.to_csv(path, index=False)

# 2. Register in analyzer
EXPORTERS = {
    "json": export_json,
    "csv": export_csv,
    "png": export_plot,
}
```

### Debugging Emotion Extraction

```python
# Add debug mode
DEBUG = True

if DEBUG:
    print(f"Prompt: {prompt}")
    print(f"Raw response: {response}")
    print(f"Parsed emotions: {emotions}")
```

---

## Prompt Templates

### Agent 1: World Builder

```python
agent_1_system_prompt = """
You are an expert literary analyst, data structurer, and world-builder. Your task is to analyze a romance novel's text (or summary) and extract the core narrative scaffolding, relationship dynamics, and world rules.

### YOUR TASK
Read the provided text. Identify the primary characters, their aliases, the fundamental rules of their situation, and map their relationships using Mermaid.js syntax. Output your findings STRICTLY as a valid JSON object. Do not include markdown formatting like ```json in the output.

### EXTRACTION RULES
1. "main_pairing": The TWO central characters of the romance.
2. "aliases": A comprehensive dictionary mapping the main and key secondary characters to all their nicknames, titles, and last names used in the text (e.g., "Elizabeth": ["Lizzy", "Miss Bennet"]).
3. "world_guidelines": A list of discrete facts outlining the core conflict, the current status quo, and vital backstory. Break complex lore into simple, individual bullet points.
4. "mermaid_graph": Create a Mermaid.js flowchart (graph TD) mapping the relationships between the main pairing and key secondary characters. Use labeled arrows to define the relationship (e.g., A -->|Political Marriage| B; B -->|Secretly Hates| C).

### REQUIRED JSON SCHEMA
{
  "main_pairing": ["Full Name 1", "Full Name 2"],
  "aliases": {
    "Full Name 1": ["Alias A", "Alias B", "Title"],
    "Full Name 2": ["Alias C", "Alias D"]
  },
  "world_guidelines": [
    "Fact 1: They were forced into a political marriage.",
    "Fact 2: Character A lost his memory in an accident.",
    "Fact 3: Character B is terrified of Character A regaining his memory."
  ],
  "mermaid_graph": "graph TD\\n    A[Character A] -->|Married| B[Character B]\\n    B -->|Afraid of| A"
}
"""

agent_1_user_prompt = "Analyze the following text and extract the world information:\\n\\n{text}"
```

### Agent 2: Base Emotion Extractor

```python
agent_2_system_prompt = f"""
You are an expert computational literary analyst extracting granular, directed emotional data from a romance novel.

### CONTEXT
* Main Pairing: {agent_1_json['main_pairing']}
* Aliases: {agent_1_json['aliases']}
* Core Conflict: {agent_1_json.get('core_conflict', 'Not specified')}

### YOUR TASK
Analyze the provided text chunk and output STRICTLY in JSON.
1. Identify the "chunk_main_pov" (whose perspective we are in, or the focal character).
2. List all "characters_present" in the scene.
3. Map the DIRECTED emotions (Source -> Target) between the Main Pairing ONLY.
   - A -> B is NOT the same as B -> A.
   - Only score a direction if there is explicit textual evidence (dialogue, internal monologue, or physical action).
   - If A is thinking about B while B is absent, ONLY output the A -> B direction. Do not guess B's unwritten feelings.

### THE 9 BASE EMOTIONS (EXTENDED PLUTCHIK MODEL)
Score the Source's feelings toward the Target on each of these 9 metrics:
1. Joy (Happiness, pleasure, delight)
2. Trust (Safety, reliance, vulnerability)
3. Fear (Panic, dread, terror, anxiety)
4. Surprise (Astonishment, shock)
5. Sadness (Grief, sorrow, despair)
6. Disgust (Revulsion, aversion, contempt)
7. Anger (Fury, rage, frustration)
8. Anticipation (Looking forward to, expecting, plotting)
9. Arousal (Physical lust, romantic desire, sexual tension)

### SCORING RUBRIC (STRICT DEFAULT TO 1)
Assume 1 (Neutral/None) for ALL emotions unless explicit text proves otherwise. Normal conversation is all 1s.
* 1 (None/Baseline): No evidence of this emotion. Polite, functional, or entirely absent.
* 2 (Mild): A brief, subtle hint or low-energy flicker of the emotion.
* 3 (Moderate): Clear, undeniable presence of the emotion.
* 4 (Strong): Emotion heavily drives the character's actions or thoughts. High physiological arousal.
* 5 (Extreme): Overwhelming, consuming saturation of the emotion. Maximum intensity.

### REQUIRED JSON SCHEMA
{{
  "chunk_id": {chunk_id},
  "chunk_main_pov": "Name of POV character",
  "characters_present": ["Name 1", "Name 2"],
  "directed_emotions": [
    {{
      "source": "Name of Source Character",
      "target": "Name of Target Character",
      "scores": {{
        "Joy": <int 1-5>,
        "Trust": <int 1-5>,
        "Fear": <int 1-5>,
        "Surprise": <int 1-5>,
        "Sadness": <int 1-5>,
        "Disgust": <int 1-5>,
        "Anger": <int 1-5>,
        "Anticipation": <int 1-5>,
        "Arousal": <int 1-5>
      }},
      "justification_quote": "Exact text quote proving the highest active scores for this direction."
    }}
  ],
  "scene_summary": "One brief sentence summarizing the action."
}}
"""

agent_2_user_prompt = "Analyze the following text chunk:\\n\\n{text_chunk}"
```

---

## JSON Schemas

### Agent 1 Output Schema (World Builder)

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "properties": {
    "main_pairing": {
      "type": "array",
      "items": {"type": "string"},
      "minItems": 2,
      "maxItems": 2,
      "description": "The two central characters of the romance"
    },
    "aliases": {
      "type": "object",
      "description": "Character name to aliases mapping",
      "additionalProperties": {
        "type": "array",
        "items": {"type": "string"}
      }
    },
    "world_guidelines": {
      "type": "array",
      "items": {"type": "string"},
      "description": "Discrete facts about the world"
    },
    "mermaid_graph": {
      "type": "string",
      "description": "Mermaid.js relationship graph"
    }
  },
  "required": ["main_pairing", "aliases", "world_guidelines", "mermaid_graph"]
}
```

### Agent 2 Output Schema (Base Emotion Extractor)

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "properties": {
    "chunk_id": {"type": "integer"},
    "chunk_main_pov": {"type": "string"},
    "characters_present": {
      "type": "array",
      "items": {"type": "string"}
    },
    "directed_emotions": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "source": {"type": "string"},
          "target": {"type": "string"},
          "scores": {
            "type": "object",
            "properties": {
              "Joy": {"type": "integer", "minimum": 1, "maximum": 5},
              "Trust": {"type": "integer", "minimum": 1, "maximum": 5},
              "Fear": {"type": "integer", "minimum": 1, "maximum": 5},
              "Surprise": {"type": "integer", "minimum": 1, "maximum": 5},
              "Sadness": {"type": "integer", "minimum": 1, "maximum": 5},
              "Disgust": {"type": "integer", "minimum": 1, "maximum": 5},
              "Anger": {"type": "integer", "minimum": 1, "maximum": 5},
              "Anticipation": {"type": "integer", "minimum": 1, "maximum": 5},
              "Arousal": {"type": "integer", "minimum": 1, "maximum": 5}
            },
            "required": ["Joy", "Trust", "Fear", "Surprise", "Sadness", "Disgust", "Anger", "Anticipation", "Arousal"]
          },
          "justification_quote": {"type": "string"}
        },
        "required": ["source", "target", "scores", "justification_quote"]
      }
    },
    "scene_summary": {"type": "string"}
  },
  "required": ["chunk_id", "chunk_main_pov", "characters_present", "directed_emotions", "scene_summary"]
}
```

### Final Output Schema (Complete Analysis)

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "properties": {
    "metadata": {
      "type": "object",
      "properties": {
        "source": {"type": "string"},
        "analysis_date": {"type": "string", "format": "date-time"},
        "model": {"type": "string"},
        "total_chunks": {"type": "integer"}
      }
    },
    "world_builder": {
      "type": "object",
      "properties": {
        "main_pairing": {"type": "array", "items": {"type": "string"}},
        "aliases": {"type": "object"},
        "world_guidelines": {"type": "array", "items": {"type": "string"}},
        "mermaid_graph": {"type": "string"}
      }
    },
    "chunks": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "chunk_id": {"type": "integer"},
          "position": {"type": "number", "minimum": 0, "maximum": 1},
          "chunk_main_pov": {"type": "string"},
          "characters_present": {"type": "array", "items": {"type": "string"}},
          "directed_emotions": {"type": "array"},
          "scene_summary": {"type": "string"}
        }
      }
    },
    "axes": {
      "type": "object",
      "properties": {
        "intimacy": {"type": "array", "items": {"type": "number", "minimum": 1, "maximum": 5}},
        "passion": {"type": "array", "items": {"type": "number", "minimum": 1, "maximum": 5}},
        "hostility": {"type": "array", "items": {"type": "number", "minimum": 1, "maximum": 5}},
        "anxiety": {"type": "array", "items": {"type": "number", "minimum": 1, "maximum": 5}}
      },
      "required": ["intimacy", "passion", "hostility", "anxiety"]
    }
  },
  "required": ["metadata", "world_builder", "chunks", "axes"]
}
```

---

## Quick Reference: AI Assistant Commands

When working on PCFMG, you can use these commands:

```bash
# Run the full pipeline
python main.py analyze path/to/text.txt

# Run with debug output
python main.py analyze text.txt --debug

# Run tests
pytest

# Run specific test
pytest tests/test_phase2.py::test_normalize_emotion

# Format code
black pcmfg/
ruff check pcmfg/

# Type check
mypy pcmfg/
```

---

**Remember**: The goal is to make invisible emotional dynamics visible. Every design decision should serve that mission.
