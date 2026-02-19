"""Shared pytest fixtures for PCMFG tests."""

from typing import Any
from unittest.mock import MagicMock

import pytest

from pcmfg.config import Config
from pcmfg.models.schemas import (
    AxesTimeSeries,
    ChunkAnalysis,
    DirectedEmotion,
    DirectedEmotionScores,
    WorldBuilderOutput,
)


@pytest.fixture
def sample_config() -> Config:
    """Create a sample configuration."""
    return Config()


@pytest.fixture
def mock_llm_client() -> MagicMock:
    """Create a mock LLM client."""
    client = MagicMock()
    client.call.return_value = "Mock response"
    client.call_json.return_value = {
        "main_pairing": ["Alice", "Bob"],
        "aliases": {"Alice": ["Ali"], "Bob": ["Robert"]},
        "world_guidelines": ["They met at a ball."],
        "mermaid_graph": "graph TD\n    A[Alice] -->|loves| B[Bob]",
    }
    return client


@pytest.fixture
def sample_world_builder_output() -> WorldBuilderOutput:
    """Create a sample world builder output."""
    return WorldBuilderOutput(
        main_pairing=["Alice", "Bob"],
        aliases={"Alice": ["Ali"], "Bob": ["Robert"]},
        world_guidelines=["They met at a ball.", "Alice initially disliked Bob."],
        mermaid_graph="graph TD\n    A[Alice] -->|loves| B[Bob]",
    )


@pytest.fixture
def sample_emotion_scores() -> DirectedEmotionScores:
    """Create sample emotion scores."""
    return DirectedEmotionScores(
        Joy=3,
        Trust=2,
        Fear=1,
        Surprise=1,
        Sadness=1,
        Disgust=1,
        Anger=1,
        Anticipation=2,
        Arousal=2,
    )


@pytest.fixture
def sample_directed_emotion(sample_emotion_scores: DirectedEmotionScores) -> DirectedEmotion:
    """Create a sample directed emotion."""
    return DirectedEmotion(
        source="Alice",
        target="Bob",
        scores=sample_emotion_scores,
        justification_quote="Alice smiled at Bob across the room.",
    )


@pytest.fixture
def sample_chunk_analysis(
    sample_directed_emotion: DirectedEmotion,
) -> ChunkAnalysis:
    """Create a sample chunk analysis."""
    # Create reverse emotion (Bob -> Alice)
    reverse_scores = DirectedEmotionScores(Joy=2, Trust=2)
    reverse_emotion = DirectedEmotion(
        source="Bob",
        target="Alice",
        scores=reverse_scores,
        justification_quote="Bob noticed Alice from across the room.",
    )

    return ChunkAnalysis(
        chunk_id=0,
        position=0.0,
        chunk_main_pov="Alice",
        characters_present=["Alice", "Bob"],
        directed_emotions=[sample_directed_emotion, reverse_emotion],
        scene_summary="Alice and Bob meet at the ball.",
    )


@pytest.fixture
def sample_axes_time_series() -> AxesTimeSeries:
    """Create sample axes time series data."""
    return AxesTimeSeries(
        intimacy=[1.0, 1.5, 2.0, 2.5, 3.0],
        passion=[1.0, 1.2, 1.5, 2.0, 2.5],
        hostility=[2.0, 1.8, 1.5, 1.2, 1.0],
        anxiety=[1.5, 1.3, 1.2, 1.1, 1.0],
    )


@pytest.fixture
def sample_text() -> str:
    """Create sample romance text for testing."""
    return """
    Alice stood at the edge of the ballroom, watching the dancers swirl past.
    She had no intention of participating tonight—until she saw him.

    Bob was handsome in a sharp, angular way, his dark hair swept back from
    his forehead. He moved through the crowd with easy confidence, nodding
    at acquaintances, stopping to exchange pleasantries.

    Their eyes met across the room. Alice felt a jolt of something—was it
    annoyance? Interest? She couldn't tell. But when he started walking
    toward her, she found herself unable to look away.

    "Miss Alice," he said, bowing slightly. "Would you do me the honor?"

    She should say no. She wanted to say no. But instead she heard herself
    reply, "I suppose one dance wouldn't hurt."

    As they moved across the floor together, Alice realized with growing
    surprise that she was actually enjoying herself. Perhaps this evening
    wouldn't be so terrible after all.
    """
