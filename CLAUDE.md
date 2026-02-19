# PCFMG Technical Guide for AI Assistants

**Purpose**: This document provides technical guidance for AI assistants (Claude, GPT, etc.) working on the PCFMG codebase.

---

## Project Mission

PCFMG (Please Care My Feeling Graph) is a computational romance narrative mining system that:

1. **Extracts** world context and directed emotions using two specialized LLM agents:
   - Agent 1 (World Builder): Identifies main pairing, aliases, core conflict, world guidelines, and relationship graph
   - Agent 2 (Base Emotion Extractor): Scores 9 base emotions using strict 1-5 scale
2. **Iteratively extracts** directed emotions from text chunks with context from Agent 1
3. **Synthesizes** raw emotion time-series data using deterministic Python (no LLM)

**Key Difference**: PCFMG outputs **raw 9 base emotions** as time-series, NOT aggregated romance axes. The raw data preserves maximum granularity for downstream analysis.

The goal is to transform unstructured romantic narratives into structured time-series data that reveals how relationships evolve over time.

---

## Architecture Deep-Dive

### The 3-Phase Pipeline

```
INPUT TEXT (romantic narrative)
    ↓
┌─────────────────────────────────────────────┐
│ PHASE 1: WORLD BUILDER (Agent 1 - LLM)       │
│ - Identify main pairing                      │
│ - Extract character aliases                  │
│ - Extract core conflict                      │
│ - Generate world guidelines                  │
│ - Create Mermaid relationship graph          │
│ Output: WorldBuilderOutput JSON              │
└─────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────┐
│ PHASE 2: EMOTION EXTRACTION (Agent 2 - LLM)  │
│ - Iterate over text chunks                   │
│ - Skip chunks with no relevant characters    │
│ - Extract directed emotions (A→B, B→A)       │
│ - Score 9 base emotions per direction        │
│ Output: List[ChunkAnalysis]                  │
└─────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────┐
│ PHASE 3: SYNTHESIS (Deterministic Python)    │
│ - Forward fill missing emotion data          │
│ - Build time-series for each emotion         │
│ - Generate visualization                     │
│ Output: JSON + visualization                 │
└─────────────────────────────────────────────┘
```

### Phase 1: World Builder (Agent 1 - LLM)

**Input**: Novel summary, synopsis, or the first few chapters

**Purpose**: Establish context and rules to prevent downstream hallucinations and catch name variations.

**Tasks**:
1. Identify the **main pairing** (two central characters of the romance)
2. Extract **aliases** for all main and secondary characters (nicknames, titles, last names)
3. Extract **core conflict** - a single sentence describing the central tension
4. Generate **world guidelines** - discrete facts about status quo and backstory
5. Create **Mermaid.js graph** mapping relationships between main pairing and key secondary characters

**Output Schema**:
```json
{
  "main_pairing": ["Full Name 1", "Full Name 2"],
  "aliases": {
    "Full Name 1": ["Alias A", "Alias B", "Title"],
    "Full Name 2": ["Alias C", "Alias D"]
  },
  "core_conflict": "A single sentence describing the central romantic tension.",
  "world_guidelines": [
    "Fact 1: They were forced into a political marriage.",
    "Fact 2: Character A lost his memory in an accident.",
    "Fact 3: Character B is terrified of Character A regaining his memory."
  ],
  "mermaid_graph": "graph TD\\n    A[Character A] -->|Married| B[Character B]\\n    B -->|Afraid of| A"
}
```

### Phase 2: Emotion Extraction (Agent 2 - LLM Loop)

**Input**: 
- Rolling text chunks (e.g., paragraph by paragraph or 500-word windows)
- Agent 1's context (main_pairing, aliases, core_conflict, world_guidelines)

**Purpose**: Extract directed emotions for each chunk, scoring only what is explicitly on the page.

**Token Efficiency Optimization**: Before calling Agent 2, check if the chunk contains any names from the aliases list. If not, **skip the LLM call entirely** to save API costs.

**Tasks**:
1. Identify the **chunk_main_pov** (whose perspective we're in)
2. List **characters_present** in the scene
3. Score **directed emotions** (Source → Target) between main pairing ONLY
4. Provide **justification quotes** for the highest scores

**Critical Rules**:
- A → B is NOT the same as B → A
- Only score directions with **explicit textual evidence** (dialogue, internal monologue, physical action)
- If A thinks about B while B is absent, ONLY output A → B direction. Do not guess B's unwritten feelings.
- **Default all scores to 1** unless text proves otherwise

**Output Schema per Chunk**:
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

### Phase 3: Synthesis (Deterministic Python)

**No LLM is used in this phase** - standard Python (`pandas`, `numpy`, `matplotlib`) handles all computation.

#### Step 3A: Missing Data Imputation (Forward Fill)

Because Agent 2 only scores what is on the page, the reverse direction (B → A) will often be null when B is absent from a scene.

**Solution**: Use `.ffill()` (forward fill) to carry forward the last known emotional state until the character appears again.

```python
def impute_missing_emotions(chunks: list[ChunkAnalysis]) -> list[ChunkAnalysis]:
    """
    Forward fill missing emotion directions.
    
    If chunk 5 has A→B but not B→A, carry forward B→A from chunk 4.
    """
    last_known_state = {}  # key: "source->target", value: DirectedEmotion
    
    for chunk in chunks:
        # Store current directions
        for emotion in chunk.directed_emotions:
            key = f"{emotion.source}->{emotion.target}"
            last_known_state[key] = emotion
        
        # Impute missing directions for main pairing
        # If A→B exists but B→A doesn't, use last known B→A
        # ...implementation
```

#### Step 3B: Build Raw Emotion Time-Series

**Output the raw 9 emotions as separate time-series**, not aggregated axes:

```python
class EmotionTimeSeries(TypedDict):
    """Raw emotion time-series for a directed relationship."""
    Joy: list[float]
    Trust: list[float]
    Fear: list[float]
    Surprise: list[float]
    Sadness: list[float]
    Disgust: list[float]
    Anger: list[float]
    Anticipation: list[float]
    Arousal: list[float]

def build_emotion_timeseries(
    chunks: list[ChunkAnalysis],
    source: str,
    target: str
) -> EmotionTimeSeries:
    """Build time-series for Source→Target emotions across all chunks."""
    timeseries = {emotion: [] for emotion in EMOTION_LIST}
    
    for chunk in chunks:
        # Find the directed emotion for this pair
        emotion = find_directed_emotion(chunk, source, target)
        if emotion:
            for e in EMOTION_LIST:
                timeseries[e].append(emotion.scores[e])
        else:
            # After forward fill, this should be rare
            for e in EMOTION_LIST:
                timeseries[e].append(1.0)  # baseline
    
    return timeseries
```

#### Step 3C: Graphing

Plot the narrative timeline (X-axis) against the 1-5 emotional intensity (Y-axis) for each of the 9 base emotions, generating the final visual "heartbeat" of the novel.

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

## Output Format

### Final Output Schema (Complete Analysis)

The output contains **raw emotion time-series** for both directions of the relationship:

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
          "justification_quote": "She told the story...with great energy among her friends; for she had a lively, playful disposition, which delighted in anything ridiculous."
        }
      ],
      "scene_summary": "Elizabeth meets Darcy at the Meryton ball and takes offense at his refusal to dance."
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

---

## Code Organization

### Expected Directory Structure

```
PCMFG/
├── pcmfg/
│   ├── __init__.py
│   ├── cli.py                 # Command-line interface
│   ├── analyzer.py            # Main orchestration
│   ├── config.py              # Configuration management
│   ├── phase1/
│   │   ├── __init__.py
│   │   ├── world_builder.py   # Agent 1: World Builder
│   │   └── emotion_extractor.py  # Agent 2: Emotion Extractor
│   ├── phase2/
│   │   ├── __init__.py
│   │   └── (deprecated)       # Phase 2 is now the Agent 2 loop
│   ├── phase3/
│   │   ├── __init__.py
│   │   ├── synthesizer.py     # Forward fill & time-series building
│   │   └── plotter.py         # Visualization
│   ├── llm/
│   │   ├── __init__.py
│   │   ├── base.py            # LLM client protocol
│   │   ├── openai_client.py   # OpenAI implementation
│   │   └── anthropic_client.py # Anthropic implementation
│   ├── models/
│   │   ├── __init__.py
│   │   └── schemas.py         # Pydantic models
│   └── utils/
│       ├── __init__.py
│       ├── text_processing.py # Chunking utilities
│       └── alias_filter.py    # Token-efficient chunk filtering
├── tests/
│   ├── test_phase1.py
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
        self.world_builder = WorldBuilder(llm_client)
        self.emotion_extractor = EmotionExtractor(llm_client)
        self.synthesizer = Synthesizer()

    def analyze(self, text: str) -> AnalysisResult:
        # Phase 1: World Builder
        world = self.world_builder.build(text)
        
        # Phase 2: Emotion Extraction (iterative)
        chunks = self.emotion_extractor.extract_all(text, world)
        
        # Phase 3: Synthesis (deterministic Python)
        result = self.synthesizer.synthesize(chunks, world)
        
        return result
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
    position: float  # 0.0-1.0
    chunk_main_pov: str
    characters_present: List[str]
    directed_emotions: List[DirectedEmotion]
    scene_summary: str

class WorldBuilderOutput(BaseModel):
    """Output from Agent 1 (World Builder)."""
    main_pairing: List[str]
    aliases: Dict[str, List[str]]
    core_conflict: str
    world_guidelines: List[str]
    mermaid_graph: str

class EmotionTimeSeries(BaseModel):
    """Raw emotion time-series for a directed relationship."""
    Joy: List[float]
    Trust: List[float]
    Fear: List[float]
    Surprise: List[float]
    Sadness: List[float]
    Disgust: List[float]
    Anger: List[float]
    Anticipation: List[float]
    Arousal: List[float]

class AnalysisResult(BaseModel):
    """Complete analysis result."""
    metadata: Dict[str, str | int]
    world_builder: WorldBuilderOutput
    chunks: List[ChunkAnalysis]
    timeseries: Dict[str, EmotionTimeSeries]  # "A_to_B" and "B_to_A"
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
3. "core_conflict": A single sentence describing the central romantic tension or obstacle between the main pairing.
4. "world_guidelines": A list of discrete facts outlining the current status quo and vital backstory. Break complex lore into simple, individual bullet points.
5. "mermaid_graph": Create a Mermaid.js flowchart (graph TD) mapping the relationships between the main pairing and key secondary characters. Use labeled arrows to define the relationship (e.g., A -->|Political Marriage| B; B -->|Secretly Hates| C).

### REQUIRED JSON SCHEMA
{
  "main_pairing": ["Full Name 1", "Full Name 2"],
  "aliases": {
    "Full Name 1": ["Alias A", "Alias B", "Title"],
    "Full Name 2": ["Alias C", "Alias D"]
  },
  "core_conflict": "A single sentence describing the central romantic tension.",
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
* Core Conflict: {agent_1_json['core_conflict']}
* World Guidelines: {agent_1_json['world_guidelines']}

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
    "core_conflict": {
      "type": "string",
      "description": "A single sentence describing the central romantic tension"
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
  "required": ["main_pairing", "aliases", "core_conflict", "world_guidelines", "mermaid_graph"]
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
        "core_conflict": {"type": "string"},
        "world_guidelines": {"type": "array", "items": {"type": "string"}},
        "mermaid_graph": {"type": "string"}
      },
      "required": ["main_pairing", "aliases", "core_conflict", "world_guidelines", "mermaid_graph"]
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
    "timeseries": {
      "type": "object",
      "description": "Raw emotion time-series for each direction",
      "properties": {
        "A_to_B": {
          "type": "object",
          "properties": {
            "Joy": {"type": "array", "items": {"type": "number", "minimum": 1, "maximum": 5}},
            "Trust": {"type": "array", "items": {"type": "number", "minimum": 1, "maximum": 5}},
            "Fear": {"type": "array", "items": {"type": "number", "minimum": 1, "maximum": 5}},
            "Surprise": {"type": "array", "items": {"type": "number", "minimum": 1, "maximum": 5}},
            "Sadness": {"type": "array", "items": {"type": "number", "minimum": 1, "maximum": 5}},
            "Disgust": {"type": "array", "items": {"type": "number", "minimum": 1, "maximum": 5}},
            "Anger": {"type": "array", "items": {"type": "number", "minimum": 1, "maximum": 5}},
            "Anticipation": {"type": "array", "items": {"type": "number", "minimum": 1, "maximum": 5}},
            "Arousal": {"type": "array", "items": {"type": "number", "minimum": 1, "maximum": 5}}
          }
        },
        "B_to_A": {
          "type": "object",
          "properties": {
            "Joy": {"type": "array", "items": {"type": "number", "minimum": 1, "maximum": 5}},
            "Trust": {"type": "array", "items": {"type": "number", "minimum": 1, "maximum": 5}},
            "Fear": {"type": "array", "items": {"type": "number", "minimum": 1, "maximum": 5}},
            "Surprise": {"type": "array", "items": {"type": "number", "minimum": 1, "maximum": 5}},
            "Sadness": {"type": "array", "items": {"type": "number", "minimum": 1, "maximum": 5}},
            "Disgust": {"type": "array", "items": {"type": "number", "minimum": 1, "maximum": 5}},
            "Anger": {"type": "array", "items": {"type": "number", "minimum": 1, "maximum": 5}},
            "Anticipation": {"type": "array", "items": {"type": "number", "minimum": 1, "maximum": 5}},
            "Arousal": {"type": "array", "items": {"type": "number", "minimum": 1, "maximum": 5}}
          }
        }
      }
    }
  },
  "required": ["metadata", "world_builder", "chunks", "timeseries"]
}
```

---

## Key Safeguards & Developer Notes

### 1. Strict JSON Enforcement

Both LLMs must use structured outputs (JSON schema) so the Python pipeline does not crash during parsing.

### 2. Token Efficiency

By resolving `aliases` in Phase 1, the Python script can pre-filter chunks. If a text chunk contains **zero names from the aliases list**, the script **skips Agent 2 entirely**, saving massive API costs.

```python
def should_process_chunk(chunk_text: str, aliases: dict[str, list[str]]) -> bool:
    """Check if chunk contains any character names."""
    all_names = set()
    for main_name, alias_list in aliases.items():
        all_names.add(main_name)
        all_names.update(alias_list)
    
    chunk_lower = chunk_text.lower()
    return any(name.lower() in chunk_lower for name in all_names)
```

### 3. LLM Reliability

**Issue**: LLMs can hallucinate or produce inconsistent results.

**Safeguards**:
- Use structured JSON output with validation
- Implement confidence thresholds
- Cross-validate with multiple runs for critical analyses

### 4. Privacy

**Issue**: User texts may be sensitive.

**Safeguards**:
- Never log input texts
- Clear .env warnings about API key handling
- Local-only processing when possible

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
pytest tests/test_phase1.py::test_world_builder

# Format code
black pcmfg/
ruff check pcmfg/

# Type check
mypy pcmfg/
```

---

**Remember**: The goal is to make invisible emotional dynamics visible. Every design decision should serve that mission.
