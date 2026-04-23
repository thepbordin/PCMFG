"""Pydantic schemas for PCMFG data validation.

This module defines all data models used throughout the PCMFG pipeline:
- DirectedEmotionScores: 9 base emotion scores (1-5 scale)
- DirectedEmotion: A directed emotion with source, target, scores, and justification
- ChunkAnalysis: Analysis result for a text chunk
- WorldBuilderOutput: Output from Agent 1 (World Builder)
- AxisValues: Computed romance axis values
- AnalysisResult: Complete analysis output
"""

from datetime import datetime, timezone
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field

# =============================================================================
# Emotion Score Types
# =============================================================================

# Emotion score is always an integer from 1 to 5
EmotionScore = Annotated[int, Field(ge=1, le=5)]

# Axis values are floats from 1.0 to 5.0
AxisValue = Annotated[float, Field(ge=1.0, le=5.0)]


# =============================================================================
# Base Emotion Models
# =============================================================================

BASE_EMOTIONS = [
    "Joy",
    "Trust",
    "Fear",
    "Surprise",
    "Sadness",
    "Disgust",
    "Anger",
    "Anticipation",
    "Arousal",
]

ROMANCE_AXES = ["intimacy", "passion", "hostility", "anxiety"]


class DirectedEmotionScores(BaseModel):
    """9 base emotion scores for a directed emotion (Source → Target).

    All scores use a strict 1-5 scale where:
    - 1 (Baseline): No evidence of emotion. Polite, functional, or absent.
    - 2 (Mild): Brief, subtle hint or low-energy flicker.
    - 3 (Moderate): Clear, undeniable presence.
    - 4 (Strong): Heavily drives actions/thoughts, high physiological arousal.
    - 5 (Extreme): Overwhelming, consuming saturation.
    """

    Joy: EmotionScore = Field(default=1, description="Happiness, pleasure, delight")
    Trust: EmotionScore = Field(
        default=1, description="Safety, reliance, vulnerability"
    )
    Fear: EmotionScore = Field(default=1, description="Panic, dread, terror, anxiety")
    Surprise: EmotionScore = Field(default=1, description="Astonishment, shock")
    Sadness: EmotionScore = Field(default=1, description="Grief, sorrow, despair")
    Disgust: EmotionScore = Field(
        default=1, description="Revulsion, aversion, contempt"
    )
    Anger: EmotionScore = Field(default=1, description="Fury, rage, frustration")
    Anticipation: EmotionScore = Field(
        default=1, description="Looking forward to, expecting, plotting"
    )
    Arousal: EmotionScore = Field(
        default=1, description="Physical lust, romantic desire, sexual tension"
    )

    model_config = ConfigDict(frozen=True)


class DirectedEmotion(BaseModel):
    """A directed emotion with source, target, scores, and justification.

    Represents emotions that one character (source) feels toward another (target).
    Directionality is critical: A → B is NOT the same as B → A.
    """

    source: str = Field(description="Character feeling the emotion")
    target: str = Field(description="Character receiving the emotion")
    scores: DirectedEmotionScores = Field(description="9 base emotion scores")
    justification_quote: str = Field(
        description="Exact text quote proving the highest active scores"
    )


# =============================================================================
# Chunk Analysis Models
# =============================================================================


class ChunkAnalysis(BaseModel):
    """Analysis result for a text chunk.

    Each chunk represents a story beat or scene segment with:
    - Position in the narrative (0.0 to 1.0)
    - POV character
    - Characters present in the scene
    - Directed emotions between the main pairing
    - Brief scene summary
    """

    chunk_id: int = Field(description="Sequential chunk identifier")
    position: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Position in narrative (0.0-1.0)"
    )
    chunk_main_pov: str = Field(description="POV character for this chunk")
    characters_present: list[str] = Field(
        default_factory=list, description="Characters present in this scene"
    )
    directed_emotions: list[DirectedEmotion] = Field(
        default_factory=list, description="Directed emotions between main pairing"
    )
    scene_summary: str = Field(default="", description="Brief scene summary")


# =============================================================================
# World Builder Output (Agent 1)
# =============================================================================


class WorldBuilderOutput(BaseModel):
    """Output from Agent 1 (World Builder).

    Extracts narrative scaffolding and world context:
    - Main pairing identification
    - Character aliases
    - Core conflict (single sentence describing central tension)
    - World guidelines (facts about the story world)
    - Relationship graph (Mermaid.js syntax)
    """

    main_pairing: list[str] = Field(
        min_length=2,
        max_length=2,
        default=["Character A", "Character B"],
        description="The two central characters of the romance",
    )
    aliases: dict[str, list[str]] = Field(
        default_factory=dict, description="Character name to aliases mapping"
    )
    core_conflict: str = Field(
        default="",
        description="A single sentence describing the central romantic tension",
    )
    world_guidelines: list[str] = Field(
        default_factory=list, description="Discrete facts about the world"
    )
    mermaid_graph: str = Field(default="", description="Mermaid.js relationship graph")


# =============================================================================
# Axis Values (Phase 3 Output) - DEPRECATED
# =============================================================================


class AxisValues(BaseModel):
    """Computed romance axis values - DEPRECATED.

    This class is kept for backward compatibility but is no longer used
    in the main pipeline. The new output uses raw 9 emotion time-series.
    """

    intimacy: AxisValue = Field(description="Emotional closeness, trust, vulnerability")
    passion: AxisValue = Field(description="Romantic desire, excitement, intensity")
    hostility: AxisValue = Field(description="Anger, resentment, conflict")
    anxiety: AxisValue = Field(description="Fear, uncertainty, tension")

    model_config = ConfigDict(frozen=True)


class AxesTimeSeries(BaseModel):
    """Time series data for all four romance axes - DEPRECATED."""

    intimacy: list[float] = Field(default_factory=list)
    passion: list[float] = Field(default_factory=list)
    hostility: list[float] = Field(default_factory=list)
    anxiety: list[float] = Field(default_factory=list)


# =============================================================================
# Emotion Time-Series (New Phase 3 Output)
# =============================================================================


class EmotionTimeSeries(BaseModel):
    """Raw emotion time-series for a directed relationship.

    Contains the 9 base emotion values over time for one direction
    (e.g., A→B or B→A). All values use the 1-5 scale where 1 is baseline.
    """

    Joy: list[float] = Field(default_factory=list, description="Happiness trajectory")
    Trust: list[float] = Field(default_factory=list, description="Trust trajectory")
    Fear: list[float] = Field(default_factory=list, description="Fear trajectory")
    Surprise: list[float] = Field(
        default_factory=list, description="Surprise trajectory"
    )
    Sadness: list[float] = Field(default_factory=list, description="Sadness trajectory")
    Disgust: list[float] = Field(default_factory=list, description="Disgust trajectory")
    Anger: list[float] = Field(default_factory=list, description="Anger trajectory")
    Anticipation: list[float] = Field(
        default_factory=list, description="Anticipation trajectory"
    )
    Arousal: list[float] = Field(default_factory=list, description="Arousal trajectory")

    model_config = ConfigDict(frozen=True)


# =============================================================================
# Analysis Metadata
# =============================================================================


class AnalysisMetadata(BaseModel):
    """Metadata for an analysis run."""

    source: str = Field(default="", description="Source file or text identifier")
    analysis_date: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When the analysis was run",
    )
    model: str = Field(default="", description="LLM model used")
    total_chunks: int = Field(default=0, description="Total number of chunks analyzed")
    provider: str = Field(default="", description="LLM provider (openai/anthropic)")


# =============================================================================
# Complete Analysis Result
# =============================================================================


class AnalysisResult(BaseModel):
    """Complete analysis result from the PCMFG pipeline.

    Contains:
    - Metadata about the analysis run
    - World builder output (main pairing, aliases, core_conflict, etc.)
    - Chunk-by-chunk analysis results
    - Raw emotion time-series for both directions (A→B and B→A)
    """

    metadata: AnalysisMetadata = Field(default_factory=AnalysisMetadata)
    world_builder: WorldBuilderOutput = Field(default_factory=WorldBuilderOutput)
    chunks: list[ChunkAnalysis] = Field(default_factory=list)
    timeseries: dict[str, EmotionTimeSeries] = Field(
        default_factory=dict,
        description="Raw emotion time-series for each direction (A_to_B, B_to_A)",
    )
    # Deprecated: kept for backward compatibility
    axes: AxesTimeSeries = Field(default_factory=AxesTimeSeries)


# =============================================================================
# Normalized Trajectory (Post-Pipeline Processing)
# =============================================================================


class NormalizedTrajectory(BaseModel):
    """A single emotion trajectory resampled to a uniform grid.

    Represents one emotion's trajectory for one direction (e.g., A→B Joy)
    resampled to a uniform [0.0, 1.0] grid with N points. Used for
    cross-narrative comparison regardless of source length.
    """

    source: str = Field(description="Source narrative identifier")
    main_pairing: list[str] = Field(
        description="Character names [Character A, Character B]"
    )
    direction: str = Field(description="'A_to_B' or 'B_to_A'")
    emotion: str = Field(description="Emotion name from BASE_EMOTIONS")
    x: list[float] = Field(description="Uniform grid points on [0.0, 1.0]")
    y: list[float] = Field(description="Resampled emotion values (integers 1-5)")
    original_length: int = Field(description="Original number of data points")
    n_points: int = Field(description="Normalized grid size")

    model_config = ConfigDict(frozen=True)


# =============================================================================
# Interesting Section Detection Models (Post-Pipeline Analysis)
# =============================================================================


class DiscordResult(BaseModel):
    """A detected discord — the most unique emotional moment in the narrative."""

    index: int = Field(description="Index in the timeseries where discord starts")
    position: float = Field(
        ge=0.0, le=1.0, description="Narrative position (0.0-1.0)"
    )
    chunk_id: int = Field(description="Chunk ID at this position")
    distance: float = Field(
        ge=0.0, description="Matrix Profile distance to nearest neighbor"
    )
    window_size: int = Field(description="Window size used for detection")


class SegmentationResult(BaseModel):
    """A semantic segmentation change point (story act boundary)."""

    index: int = Field(description="Index in the timeseries where change occurs")
    position: float = Field(
        ge=0.0, le=1.0, description="Narrative position (0.0-1.0)"
    )
    chunk_id: int = Field(description="Chunk ID at this boundary")
    regime_label: str = Field(
        description="Label like 'Intro', 'Conflict', 'Resolution'"
    )


class MotifPair(BaseModel):
    """A pair of similar 18D sequences (recurring trope)."""

    index_a: int = Field(description="Start index of first occurrence")
    index_b: int = Field(description="Start index of second occurrence")
    position_a: float = Field(
        ge=0.0, le=1.0, description="Narrative position of first"
    )
    position_b: float = Field(
        ge=0.0, le=1.0, description="Narrative position of second"
    )
    chunk_id_a: int = Field(description="Chunk ID of first occurrence")
    chunk_id_b: int = Field(description="Chunk ID of second occurrence")
    distance: float = Field(
        ge=0.0, description="Pairwise distance between subsequences"
    )


class GapValue(BaseModel):
    """Per-emotion gap at a single timestamp."""

    emotion: str = Field(description="Emotion name from BASE_EMOTIONS")
    a_to_b: float = Field(description="A->B value")
    b_to_a: float = Field(description="B->A value")
    gap: float = Field(description="A->B minus B->A")


class GapAtTimestamp(BaseModel):
    """All 9 emotion gaps at a specific timestamp."""

    index: int = Field(description="Timeseries index")
    position: float = Field(ge=0.0, le=1.0, description="Narrative position")
    chunk_id: int = Field(description="Chunk ID")
    gaps: list[GapValue] = Field(description="Per-emotion gap values")
    dominant_gap_emotion: str = Field(
        description="Emotion with largest absolute gap"
    )
    dominant_gap_value: float = Field(
        description="Value of the largest absolute gap"
    )


class InterestingSectionReport(BaseModel):
    """Complete report from Interesting Section Detection."""

    source: str = Field(description="Source narrative identifier")
    main_pairing: list[str] = Field(description="Character names")
    window_size: int = Field(description="Window size used for MP computation")
    n_chunks: int = Field(description="Number of chunks in the narrative")
    discords: list[DiscordResult] = Field(
        default_factory=list, description="Top-K detected discords"
    )
    segments: list[SegmentationResult] = Field(
        default_factory=list, description="Semantic change points"
    )
    motifs: list[MotifPair] = Field(
        default_factory=list, description="Top-K recurring motif pairs"
    )
    gaps: list[GapAtTimestamp] = Field(
        default_factory=list, description="Gap analysis at interesting points"
    )
    matrix_profile_distances: list[float] = Field(
        default_factory=list,
        description="Full matrix profile distance array (for visualization)",
    )
