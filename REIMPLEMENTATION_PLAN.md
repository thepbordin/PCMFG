# PCFMG Re-Implementation Plan

**Purpose**: This document outlines the detailed plan to re-implement PCFMG according to the new specification based on the Source of Truth (`.claude/agents/` instructions).

---

## Executive Summary

### Key Changes Required

| Current Implementation | New Specification | Priority |
|-----------------------|-------------------|----------|
| Outputs 4 aggregated axes (Intimacy, Passion, Hostility, Anxiety) | Output raw 9 emotions as time-series | **CRITICAL** |
| No core_conflict field in WorldBuilderOutput | Add core_conflict to Agent 1 output | **HIGH** |
| No forward fill imputation | Implement .ffill() for missing directions | **HIGH** |
| No chunk-skipping optimization | Skip chunks with no relevant characters | **MEDIUM** |
| Phase 2 is "Normalization" | Phase 2 is "Emotion Extraction" (Agent 2 loop) | **MEDIUM** |
| Phase 3 computes axes | Phase 3 is deterministic synthesis | **MEDIUM** |

---

## Phase-by-Phase Implementation Plan

### Phase 1: Update World Builder (Agent 1)

**File**: `pcmfg/phase1/world_builder.py`

#### 1.1 Update Pydantic Schema

**Current**:
```python
class WorldBuilderOutput(BaseModel):
    main_pairing: list[str]
    aliases: dict[str, list[str]]
    world_guidelines: list[str]
    mermaid_graph: str
```

**New**:
```python
class WorldBuilderOutput(BaseModel):
    main_pairing: list[str] = Field(min_length=2, max_length=2)
    aliases: dict[str, list[str]]
    core_conflict: str  # NEW FIELD
    world_guidelines: list[str]
    mermaid_graph: str
```

#### 1.2 Update System Prompt

**Changes**:
1. Add `core_conflict` to extraction rules
2. Update JSON schema in prompt to include `core_conflict`

**New Prompt Section**:
```python
"""
### EXTRACTION RULES
1. "main_pairing": The TWO central characters of the romance.
2. "aliases": A comprehensive dictionary mapping the main and key secondary 
   characters to all their nicknames, titles, and last names.
3. "core_conflict": A single sentence describing the central romantic tension 
   or obstacle between the main pairing.
4. "world_guidelines": A list of discrete facts outlining the current status 
   quo and vital backstory.
5. "mermaid_graph": Create a Mermaid.js flowchart mapping relationships.

### REQUIRED JSON SCHEMA
{
  "main_pairing": ["Full Name 1", "Full Name 2"],
  "aliases": {...},
  "core_conflict": "A single sentence describing the central romantic tension.",
  "world_guidelines": [...],
  "mermaid_graph": "..."
}
"""
```

#### 1.3 Tasks

- [ ] Update `WorldBuilderOutput` schema in `models/schemas.py`
- [ ] Update Agent 1 system prompt in `world_builder.py`
- [ ] Update Agent 2 to inject `core_conflict` into context
- [ ] Add validation for `core_conflict` (non-empty string)

---

### Phase 2: Update Emotion Extractor (Agent 2)

**File**: `pcmfg/phase1/emotion_extractor.py`

#### 2.1 Add Chunk Filtering (Token Efficiency)

**New Function**:
```python
def should_process_chunk(chunk_text: str, aliases: dict[str, list[str]]) -> bool:
    """
    Check if chunk contains any character names from aliases.
    Returns False to skip LLM call (saves tokens).
    """
    all_names: set[str] = set()
    for main_name, alias_list in aliases.items():
        all_names.add(main_name)
        all_names.update(alias_list)
    
    chunk_lower = chunk_text.lower()
    return any(name.lower() in chunk_lower for name in all_names)
```

**Integration**:
```python
class EmotionExtractor:
    def extract_all(
        self, 
        chunks: list[str], 
        world: WorldBuilderOutput
    ) -> list[ChunkAnalysis]:
        results = []
        
        for i, chunk_text in enumerate(chunks):
            # TOKEN EFFICIENCY: Skip chunks with no relevant characters
            if not should_process_chunk(chunk_text, world.aliases):
                logger.info(f"Skipping chunk {i}: no character names found")
                continue
            
            result = self.extract_chunk(i, chunk_text, world)
            results.append(result)
        
        return results
```

#### 2.2 Update Agent 2 System Prompt

**Changes**:
1. Include `core_conflict` in CONTEXT section
2. Include `world_guidelines` list in CONTEXT section

**New Prompt Section**:
```python
def build_emotion_extractor_system_prompt(world: WorldBuilderOutput) -> str:
    return f"""
### CONTEXT
* Main Pairing: {world.main_pairing}
* Aliases: {world.aliases}
* Core Conflict: {world.core_conflict}
* World Guidelines: {world.world_guidelines}

### YOUR TASK
...
"""
```

#### 2.3 Tasks

- [ ] Add `should_process_chunk()` function
- [ ] Integrate chunk filtering in `extract_all()`
- [ ] Update Agent 2 system prompt to include `core_conflict` and `world_guidelines`
- [ ] Add logging for skipped chunks

---

### Phase 3: Implement Synthesizer (Deterministic Python)

**New File**: `pcmfg/phase3/synthesizer.py`

This is the major re-implementation. The current `axis_mapper.py` computes aggregated axes - this needs to be replaced with raw emotion time-series synthesis.

#### 3.1 Remove Axis Mapper

**Current `axis_mapper.py` computes**:
```python
intimacy = (Trust + Joy) / 2.0
passion = (Arousal + Anticipation + Joy) / 3.0
hostility = (Anger + Disgust + Sadness) / 3.0
anxiety = (Fear + Surprise + Sadness) / 3.0
```

**New `synthesizer.py` outputs**:
- Raw 9 emotion time-series for A→B
- Raw 9 emotion time-series for B→A
- NO aggregated axes

#### 3.2 Forward Fill Implementation

```python
from typing import TypedDict

EMOTION_LIST = [
    "Joy", "Trust", "Fear", "Surprise", 
    "Sadness", "Disgust", "Anger", "Anticipation", "Arousal"
]

class EmotionTimeSeries(TypedDict):
    Joy: list[float]
    Trust: list[float]
    Fear: list[float]
    Surprise: list[float]
    Sadness: list[float]
    Disgust: list[float]
    Anger: list[float]
    Anticipation: list[float]
    Arousal: list[float]


def impute_missing_emotions(
    chunks: list[ChunkAnalysis],
    main_pairing: list[str]
) -> list[ChunkAnalysis]:
    """
    Forward fill missing emotion directions.
    
    Because Agent 2 only scores what is on the page, the reverse direction
    (B → A) will often be null when B is absent from a scene.
    
    Solution: Use .ffill() to carry forward the last known emotional state.
    """
    char_a, char_b = main_pairing[0], main_pairing[1]
    
    # Initialize last known states with baseline (all 1s)
    baseline_scores = {emotion: 1 for emotion in EMOTION_LIST}
    
    last_known: dict[str, DirectedEmotionScores] = {
        f"{char_a}->{char_b}": DirectedEmotionScores(**baseline_scores),
        f"{char_b}->{char_a}": DirectedEmotionScores(**baseline_scores),
    }
    
    imputed_chunks = []
    
    for chunk in chunks:
        # Track which directions are present in this chunk
        present_directions = set()
        
        for emotion in chunk.directed_emotions:
            key = f"{emotion.source}->{emotion.target}"
            present_directions.add(key)
            last_known[key] = emotion.scores
        
        # Build new directed_emotions with forward-filled missing directions
        new_directed_emotions = list(chunk.directed_emotions)
        
        for direction in [f"{char_a}->{char_b}", f"{char_b}->{char_a}"]:
            if direction not in present_directions:
                # Create imputed emotion from last known state
                source, target = direction.split("->")
                imputed = DirectedEmotion(
                    source=source,
                    target=target,
                    scores=last_known[direction],
                    justification_quote="[FORWARD FILLED - character absent from scene]"
                )
                new_directed_emotions.append(imputed)
        
        # Create new chunk with imputed data
        imputed_chunk = ChunkAnalysis(
            chunk_id=chunk.chunk_id,
            position=chunk.position,
            chunk_main_pov=chunk.chunk_main_pov,
            characters_present=chunk.characters_present,
            directed_emotions=new_directed_emotions,
            scene_summary=chunk.scene_summary
        )
        imputed_chunks.append(imputed_chunk)
    
    return imputed_chunks
```

#### 3.3 Time-Series Builder

```python
def build_emotion_timeseries(
    chunks: list[ChunkAnalysis],
    source: str,
    target: str
) -> EmotionTimeSeries:
    """
    Build time-series for Source→Target emotions across all chunks.
    
    Returns raw 9 emotion arrays.
    """
    timeseries: EmotionTimeSeries = {emotion: [] for emotion in EMOTION_LIST}
    
    for chunk in chunks:
        # Find the directed emotion for this pair
        emotion = None
        for de in chunk.directed_emotions:
            if de.source == source and de.target == target:
                emotion = de
                break
        
        if emotion:
            for e in EMOTION_LIST:
                score = getattr(emotion.scores, e)
                timeseries[e].append(float(score))
        else:
            # Fallback to baseline (should not happen after forward fill)
            for e in EMOTION_LIST:
                timeseries[e].append(1.0)
    
    return timeseries
```

#### 3.4 Main Synthesizer Class

```python
class Synthesizer:
    """Phase 3: Deterministic Python synthesis (no LLM)."""
    
    def synthesize(
        self,
        chunks: list[ChunkAnalysis],
        world: WorldBuilderOutput
    ) -> AnalysisResult:
        """
        Synthesize raw emotion time-series from extracted chunks.
        
        Steps:
        1. Forward fill missing emotion directions
        2. Build time-series for A→B
        3. Build time-series for B→A
        """
        char_a, char_b = world.main_pairing[0], world.main_pairing[1]
        
        # Step 1: Forward fill
        imputed_chunks = impute_missing_emotions(chunks, world.main_pairing)
        
        # Step 2 & 3: Build time-series
        a_to_b = build_emotion_timeseries(imputed_chunks, char_a, char_b)
        b_to_a = build_emotion_timeseries(imputed_chunks, char_b, char_a)
        
        # Build result
        return AnalysisResult(
            metadata={...},
            world_builder=world,
            chunks=imputed_chunks,
            timeseries={
                "A_to_B": EmotionTimeSeries(**a_to_b),
                "B_to_A": EmotionTimeSeries(**b_to_a),
            }
        )
```

#### 3.5 Tasks

- [ ] Create new `synthesizer.py` file
- [ ] Implement `impute_missing_emotions()` with forward fill
- [ ] Implement `build_emotion_timeseries()`
- [ ] Implement `Synthesizer` class
- [ ] Deprecate or remove `axis_mapper.py`
- [ ] Update `EmotionTimeSeries` schema in `models/schemas.py`
- [ ] Update `AnalysisResult` schema to use `timeseries` instead of `axes`

---

### Phase 4: Update Schemas

**File**: `pcmfg/models/schemas.py`

#### 4.1 Update WorldBuilderOutput

```python
class WorldBuilderOutput(BaseModel):
    """Output from Agent 1 (World Builder)."""
    main_pairing: list[str] = Field(min_length=2, max_length=2)
    aliases: dict[str, list[str]]
    core_conflict: str = Field(min_length=1)  # NEW
    world_guidelines: list[str]
    mermaid_graph: str
```

#### 4.2 Add EmotionTimeSeries

```python
class EmotionTimeSeries(BaseModel):
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
```

#### 4.3 Update AnalysisResult

**Current**:
```python
class AnalysisResult(BaseModel):
    axes: Dict[str, List[float]]  # intimacy, passion, hostility, anxiety
```

**New**:
```python
class AnalysisResult(BaseModel):
    timeseries: Dict[str, EmotionTimeSeries]  # "A_to_B" and "B_to_A"
```

#### 4.4 Tasks

- [ ] Update `WorldBuilderOutput` with `core_conflict`
- [ ] Add `EmotionTimeSeries` model
- [ ] Update `AnalysisResult` to use `timeseries` instead of `axes`
- [ ] Remove `AxisValues` model (no longer needed)

---

### Phase 5: Update Analyzer

**File**: `pcmfg/analyzer.py`

#### 5.1 Update Pipeline Flow

**Current**:
```python
class PCMFGAnalyzer:
    def analyze(self, text: str) -> AnalysisResult:
        # Phase 1
        world = self.world_builder.build(text)
        emotions = self.emotion_extractor.extract_all(chunks, world)
        
        # Phase 2 (Validation/Normalization)
        normalized = self.normalizer.normalize(emotions)
        
        # Phase 3 (Axis Mapping)
        axes = self.axis_mapper.map_to_axes(normalized)
        
        return AnalysisResult(axes=axes)
```

**New**:
```python
class PCMFGAnalyzer:
    def analyze(self, text: str) -> AnalysisResult:
        # Phase 1: World Builder
        world = self.world_builder.build(text)
        
        # Phase 2: Emotion Extraction (with chunk filtering)
        chunks = self.text_processor.chunk_text(text)
        emotions = self.emotion_extractor.extract_all(chunks, world)
        
        # Phase 3: Synthesis (forward fill + time-series)
        result = self.synthesizer.synthesize(emotions, world)
        
        return result
```

#### 5.2 Tasks

- [ ] Remove normalizer from pipeline
- [ ] Remove axis_mapper from pipeline
- [ ] Add synthesizer to pipeline
- [ ] Update imports

---

### Phase 6: Update Visualization

**File**: `pcmfg/visualization/plotter.py`

#### 6.1 Current Visualization

- 4-panel plot showing Intimacy, Passion, Hostility, Anxiety

#### 6.2 New Visualization

**Option A: 9×2 Grid (Recommended)**
- 9 rows (one per emotion)
- 2 columns (A→B and B→A)
- Each subplot shows the emotion trajectory

**Option B: Side-by-Side Comparison**
- Left: All 9 emotions for A→B
- Right: All 9 emotions for B→A

**Implementation**:
```python
def plot_emotion_timeseries(result: AnalysisResult, output_path: str):
    """Plot raw 9 emotion time-series for both directions."""
    fig, axes = plt.subplots(9, 2, figsize=(12, 18))
    
    a_to_b = result.timeseries["A_to_B"]
    b_to_a = result.timeseries["B_to_A"]
    x = range(len(a_to_b.Joy))
    
    for i, emotion in enumerate(EMOTION_LIST):
        # A→B (left column)
        axes[i, 0].plot(x, getattr(a_to_b, emotion))
        axes[i, 0].set_title(f"{emotion} (A→B)")
        axes[i, 0].set_ylim(0.5, 5.5)
        
        # B→A (right column)
        axes[i, 1].plot(x, getattr(b_to_a, emotion))
        axes[i, 1].set_title(f"{emotion} (B→A)")
        axes[i, 1].set_ylim(0.5, 5.5)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
```

#### 6.3 Tasks

- [ ] Redesign visualization for 9×2 grid
- [ ] Update plotter to use `timeseries` instead of `axes`
- [ ] Update color scheme and labels
- [ ] Update legend to show direction

---

### Phase 7: Update Tests

#### 7.1 Tests to Update

| Test File | Changes |
|-----------|---------|
| `test_phase1.py` | Add test for `core_conflict` extraction |
| `test_phase2.py` | Deprecate (Phase 2 is now part of Phase 1) |
| `test_phase3.py` | Rewrite for `Synthesizer` instead of `AxisMapper` |

#### 7.2 New Tests Required

```python
# test_synthesizer.py

def test_forward_fill_imputation():
    """Test that missing directions are forward filled."""
    # Create chunks where B→A is missing in chunk 2
    # Verify B→A in chunk 2 equals B→A in chunk 1
    pass

def test_timeseries_building():
    """Test that time-series are correctly built."""
    # Verify each emotion array has correct length
    # Verify values match extracted scores
    pass

def test_baseline_initialization():
    """Test that forward fill starts with baseline (all 1s)."""
    # First chunk with only A→B should have B→A as all 1s
    pass
```

```python
# test_chunk_filtering.py

def test_skip_chunk_no_characters():
    """Test that chunks with no character names are skipped."""
    pass

def test_process_chunk_with_characters():
    """Test that chunks with character names are processed."""
    pass
```

#### 7.3 Tasks

- [ ] Update `test_phase1.py` with `core_conflict` tests
- [ ] Create `test_synthesizer.py` with forward fill tests
- [ ] Create `test_chunk_filtering.py` with token efficiency tests
- [ ] Deprecate `test_phase2.py` (normalization no longer separate)
- [ ] Update integration tests for new output format

---

## Migration Checklist

### Breaking Changes

1. **Output Format**: `axes` → `timeseries`
   - Downstream consumers expecting `axes.intimacy` will break
   - Migration: Consumers must now access `timeseries.A_to_B.Trust`, etc.

2. **WorldBuilderOutput**: New `core_conflict` field
   - Old JSON files won't have this field
   - Migration: Add default empty string for backward compatibility

3. **Phase Numbering**: Phase 2 is now "Emotion Extraction" not "Normalization"
   - Documentation references need updating

### Migration Steps

1. [ ] Update all code to use new schemas
2. [ ] Update all prompts in `world_builder.py` and `emotion_extractor.py`
3. [ ] Create `synthesizer.py` and deprecate `axis_mapper.py`
4. [ ] Update `analyzer.py` pipeline
5. [ ] Update `plotter.py` visualization
6. [ ] Update all tests
7. [ ] Update CLI output format descriptions
8. [ ] Test with sample novel to verify end-to-end

---

## File Changes Summary

| File | Action | Priority |
|------|--------|----------|
| `models/schemas.py` | Update (add core_conflict, EmotionTimeSeries, update AnalysisResult) | CRITICAL |
| `phase1/world_builder.py` | Update (add core_conflict to prompt and schema) | HIGH |
| `phase1/emotion_extractor.py` | Update (add chunk filtering, update prompt) | HIGH |
| `phase3/synthesizer.py` | **CREATE NEW** (forward fill + time-series) | CRITICAL |
| `phase3/axis_mapper.py` | **DEPRECATE/REMOVE** | HIGH |
| `phase2/normalizer.py` | **DEPRECATE/REMOVE** | MEDIUM |
| `analyzer.py` | Update (remove normalizer/axis_mapper, add synthesizer) | HIGH |
| `visualization/plotter.py` | Update (9×2 grid for raw emotions) | MEDIUM |
| `tests/test_phase1.py` | Update (add core_conflict tests) | MEDIUM |
| `tests/test_synthesizer.py` | **CREATE NEW** | HIGH |
| `tests/test_chunk_filtering.py` | **CREATE NEW** | MEDIUM |
| `tests/test_phase2.py` | **DEPRECATE/REMOVE** | LOW |
| `tests/test_phase3.py` | **REPLACE** with test_synthesizer.py | HIGH |

---

## Estimated Effort

| Phase | Effort | Complexity |
|-------|--------|------------|
| Phase 1: World Builder | 1-2 hours | Low |
| Phase 2: Emotion Extractor | 2-3 hours | Medium |
| Phase 3: Synthesizer | 4-6 hours | High |
| Phase 4: Schemas | 1-2 hours | Low |
| Phase 5: Analyzer | 1-2 hours | Low |
| Phase 6: Visualization | 2-3 hours | Medium |
| Phase 7: Tests | 3-4 hours | Medium |

**Total**: ~14-22 hours

---

## Success Criteria

1. **Agent 1** outputs `core_conflict` field
2. **Agent 2** skips chunks with no character names (logged)
3. **Phase 3** implements forward fill for missing directions
4. **Output** contains raw 9 emotion time-series (no aggregated axes)
5. **Visualization** shows 9×2 grid of emotion trajectories
6. **All tests** pass
7. **End-to-end** analysis works with sample novel

---

## Notes

- The `normalizer.py` can be kept for validation purposes (checking 1-5 range, justification quality) but should not be a separate pipeline phase
- Forward fill should start with baseline (all 1s), not null
- The `core_conflict` field helps Agent 2 understand the central tension without needing to infer it
- Token efficiency from chunk filtering can save significant API costs for long novels
