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
    Trust: EmotionScore = Field(default=1, description="Safety, reliance, vulnerability")
    Fear: EmotionScore = Field(default=1, description="Panic, dread, terror, anxiety")
    Surprise: EmotionScore = Field(default=1, description="Astonishment, shock")
    Sadness: EmotionScore = Field(default=1, description="Grief, sorrow, despair")
    Disgust: EmotionScore = Field(default=1, description="Revulsion, aversion, contempt")
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
    world_guidelines: list[str] = Field(
        default_factory=list, description="Discrete facts about the world"
    )
    mermaid_graph: str = Field(default="", description="Mermaid.js relationship graph")


# =============================================================================
# Axis Values (Phase 3 Output)
# =============================================================================


class AxisValues(BaseModel):
    """Computed romance axis values.

    Four axes computed from base emotions:
    - Intimacy: (Trust + Joy) / 2
    - Passion: (Arousal + Anticipation + Joy) / 3
    - Hostility: (Anger + Disgust + Sadness) / 3
    - Anxiety: (Fear + Surprise + Sadness) / 3
    """

    intimacy: AxisValue = Field(description="Emotional closeness, trust, vulnerability")
    passion: AxisValue = Field(description="Romantic desire, excitement, intensity")
    hostility: AxisValue = Field(description="Anger, resentment, conflict")
    anxiety: AxisValue = Field(description="Fear, uncertainty, tension")

    model_config = ConfigDict(frozen=True)


class AxesTimeSeries(BaseModel):
    """Time series data for all four romance axes."""

    intimacy: list[float] = Field(default_factory=list)
    passion: list[float] = Field(default_factory=list)
    hostility: list[float] = Field(default_factory=list)
    anxiety: list[float] = Field(default_factory=list)


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
    - World builder output (main pairing, aliases, etc.)
    - Chunk-by-chunk analysis results
    - Time series data for the four romance axes
    """

    metadata: AnalysisMetadata = Field(default_factory=AnalysisMetadata)
    world_builder: WorldBuilderOutput = Field(default_factory=WorldBuilderOutput)
    chunks: list[ChunkAnalysis] = Field(default_factory=list)
    axes: AxesTimeSeries = Field(default_factory=AxesTimeSeries)
