"""Shared pytest fixtures for PCMFG tests."""

from datetime import datetime, timezone
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest

from pcmfg.analysis.dtw_clusterer import DTWClusterResult
from pcmfg.config import Config
from pcmfg.models.schemas import (
    AnalysisMetadata,
    AnalysisResult,
    AxesTimeSeries,
    BASE_EMOTIONS,
    ChunkAnalysis,
    DirectedEmotion,
    DirectedEmotionScores,
    EmotionTimeSeries,
    NormalizedTrajectory,
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


@pytest.fixture
def sample_analysis_result(
    sample_world_builder_output: WorldBuilderOutput,
) -> AnalysisResult:
    """Create a sample AnalysisResult with 5 chunks and full timeseries.

    Simulates a short narrative with varying emotional dynamics:
    - Chunks at non-uniform positions (0.0, 0.15, 0.35, 0.7, 1.0)
    - A_to_B: Joy increases 1→3→2→4→5, Anger decreases 3→2→1→1→1
    - B_to_A: Trust stays 1→1→2→3→4, Arousal 1→1→1→2→3
    """
    chunks = [
        ChunkAnalysis(
            chunk_id=0,
            position=0.0,
            chunk_main_pov="Alice",
            characters_present=["Alice", "Bob"],
            directed_emotions=[
                DirectedEmotion(
                    source="Alice",
                    target="Bob",
                    scores=DirectedEmotionScores(Joy=1, Anger=3, Anticipation=2),
                    justification_quote="Alice scowled at Bob.",
                ),
                DirectedEmotion(
                    source="Bob",
                    target="Alice",
                    scores=DirectedEmotionScores(Trust=1, Arousal=1),
                    justification_quote="Bob nodded politely.",
                ),
            ],
            scene_summary="Alice and Bob meet coldly.",
        ),
        ChunkAnalysis(
            chunk_id=1,
            position=0.15,
            chunk_main_pov="Alice",
            characters_present=["Alice", "Bob"],
            directed_emotions=[
                DirectedEmotion(
                    source="Alice",
                    target="Bob",
                    scores=DirectedEmotionScores(Joy=3, Anger=2, Anticipation=2),
                    justification_quote="Alice found Bob surprisingly interesting.",
                ),
                DirectedEmotion(
                    source="Bob",
                    target="Alice",
                    scores=DirectedEmotionScores(Trust=1, Arousal=1),
                    justification_quote="Bob watched her carefully.",
                ),
            ],
            scene_summary="A guarded conversation.",
        ),
        ChunkAnalysis(
            chunk_id=2,
            position=0.35,
            chunk_main_pov="Alice",
            characters_present=["Alice", "Bob"],
            directed_emotions=[
                DirectedEmotion(
                    source="Alice",
                    target="Bob",
                    scores=DirectedEmotionScores(Joy=2, Anger=1, Anticipation=3),
                    justification_quote="Alice felt uncertain about Bob.",
                ),
                DirectedEmotion(
                    source="Bob",
                    target="Alice",
                    scores=DirectedEmotionScores(Trust=2, Arousal=1),
                    justification_quote="Bob began to trust her.",
                ),
            ],
            scene_summary="Tension and uncertainty.",
        ),
        ChunkAnalysis(
            chunk_id=3,
            position=0.7,
            chunk_main_pov="Alice",
            characters_present=["Alice", "Bob"],
            directed_emotions=[
                DirectedEmotion(
                    source="Alice",
                    target="Bob",
                    scores=DirectedEmotionScores(Joy=4, Anger=1, Anticipation=3),
                    justification_quote="Alice realized she cared for Bob.",
                ),
                DirectedEmotion(
                    source="Bob",
                    target="Alice",
                    scores=DirectedEmotionScores(Trust=3, Arousal=2),
                    justification_quote="Bob felt drawn to Alice.",
                ),
            ],
            scene_summary="Growing affection.",
        ),
        ChunkAnalysis(
            chunk_id=4,
            position=1.0,
            chunk_main_pov="Alice",
            characters_present=["Alice", "Bob"],
            directed_emotions=[
                DirectedEmotion(
                    source="Alice",
                    target="Bob",
                    scores=DirectedEmotionScores(
                        Joy=5, Anger=1, Anticipation=1, Arousal=4
                    ),
                    justification_quote="Alice kissed Bob passionately.",
                ),
                DirectedEmotion(
                    source="Bob",
                    target="Alice",
                    scores=DirectedEmotionScores(Trust=4, Arousal=3),
                    justification_quote="Bob held her close.",
                ),
            ],
            scene_summary="Romantic climax.",
        ),
    ]

    return AnalysisResult(
        metadata=AnalysisMetadata(
            source="test_novel.txt",
            analysis_date=datetime(2026, 1, 15, tzinfo=timezone.utc),
            model="gpt-4o",
            total_chunks=5,
            provider="openai",
        ),
        world_builder=sample_world_builder_output,
        chunks=chunks,
        timeseries={
            "A_to_B": EmotionTimeSeries(
                Joy=[1.0, 3.0, 2.0, 4.0, 5.0],
                Trust=[1.0, 1.0, 1.0, 2.0, 3.0],
                Fear=[1.0, 1.0, 2.0, 1.0, 1.0],
                Surprise=[1.0, 2.0, 1.0, 1.0, 1.0],
                Sadness=[1.0, 1.0, 1.0, 1.0, 1.0],
                Disgust=[1.0, 1.0, 1.0, 1.0, 1.0],
                Anger=[3.0, 2.0, 1.0, 1.0, 1.0],
                Anticipation=[2.0, 2.0, 3.0, 3.0, 1.0],
                Arousal=[1.0, 1.0, 1.0, 2.0, 4.0],
            ),
            "B_to_A": EmotionTimeSeries(
                Joy=[1.0, 1.0, 1.0, 2.0, 3.0],
                Trust=[1.0, 1.0, 2.0, 3.0, 4.0],
                Fear=[1.0, 1.0, 1.0, 1.0, 1.0],
                Surprise=[1.0, 1.0, 1.0, 1.0, 1.0],
                Sadness=[1.0, 1.0, 1.0, 1.0, 1.0],
                Disgust=[1.0, 1.0, 1.0, 1.0, 1.0],
                Anger=[1.0, 1.0, 1.0, 1.0, 1.0],
                Anticipation=[1.0, 1.0, 2.0, 2.0, 3.0],
                Arousal=[1.0, 1.0, 1.0, 2.0, 3.0],
            ),
        },
    )


@pytest.fixture
def sample_normalized_trajectories_multi() -> list[NormalizedTrajectory]:
    """Create normalized trajectories from 3 narratives with different emotional arcs.

    Narrative 1 "rising_romance.txt": Joy A2B rises 1→5, Trust B2A rises 1→4
    Narrative 2 "enemies_to_lovers.txt": Anger A2B drops 4→1, Joy rises 1→4
    Narrative 3 "slow_burn.txt": All emotions near baseline 1→2 (flat arc)

    Each narrative produces 18 NormalizedTrajectory objects
    (9 emotions x 2 directions).
    """
    trajectories: list[NormalizedTrajectory] = []

    # Narrative 1: Rising romance
    for emotion in BASE_EMOTIONS:
        if emotion == "Joy":
            y_a2b = [1.0] * 25 + [2.0] * 25 + [3.0] * 25 + [4.0] * 25
            y_b2a = [1.0] * 50 + [2.0] * 50
        elif emotion == "Trust":
            y_a2b = [1.0] * 50 + [2.0] * 25 + [3.0] * 25
            y_b2a = [1.0] * 25 + [2.0] * 25 + [3.0] * 25 + [4.0] * 25
        elif emotion == "Arousal":
            y_a2b = [1.0] * 60 + [2.0] * 20 + [3.0] * 10 + [4.0] * 10
            y_b2a = [1.0] * 70 + [2.0] * 30
        else:
            y_a2b = [1.0] * 100
            y_b2a = [1.0] * 100

        x = [float(i) / 99.0 for i in range(100)]
        trajectories.append(
            NormalizedTrajectory(
                source="rising_romance.txt",
                main_pairing=["Alice", "Bob"],
                direction="A_to_B",
                emotion=emotion,
                x=x,
                y=y_a2b,
                original_length=10,
                n_points=100,
            )
        )
        trajectories.append(
            NormalizedTrajectory(
                source="rising_romance.txt",
                main_pairing=["Alice", "Bob"],
                direction="B_to_A",
                emotion=emotion,
                x=x,
                y=y_b2a,
                original_length=10,
                n_points=100,
            )
        )

    # Narrative 2: Enemies to lovers
    for emotion in BASE_EMOTIONS:
        if emotion == "Anger":
            y_a2b = [4.0] * 25 + [3.0] * 25 + [2.0] * 25 + [1.0] * 25
            y_b2a = [3.0] * 50 + [2.0] * 50
        elif emotion == "Joy":
            y_a2b = [1.0] * 50 + [2.0] * 25 + [3.0] * 25
            y_b2a = [1.0] * 60 + [2.0] * 40
        elif emotion == "Disgust":
            y_a2b = [3.0] * 25 + [2.0] * 50 + [1.0] * 25
            y_b2a = [1.0] * 100
        else:
            y_a2b = [1.0] * 100
            y_b2a = [1.0] * 100

        x = [float(i) / 99.0 for i in range(100)]
        trajectories.append(
            NormalizedTrajectory(
                source="enemies_to_lovers.txt",
                main_pairing=["Eve", "Frank"],
                direction="A_to_B",
                emotion=emotion,
                x=x,
                y=y_a2b,
                original_length=12,
                n_points=100,
            )
        )
        trajectories.append(
            NormalizedTrajectory(
                source="enemies_to_lovers.txt",
                main_pairing=["Eve", "Frank"],
                direction="B_to_A",
                emotion=emotion,
                x=x,
                y=y_b2a,
                original_length=12,
                n_points=100,
            )
        )

    # Narrative 3: Slow burn (flat baseline arc)
    for emotion in BASE_EMOTIONS:
        y_a2b = [1.0] * 80 + [2.0] * 20
        y_b2a = [1.0] * 80 + [2.0] * 20

        x = [float(i) / 99.0 for i in range(100)]
        trajectories.append(
            NormalizedTrajectory(
                source="slow_burn.txt",
                main_pairing=["Grace", "Henry"],
                direction="A_to_B",
                emotion=emotion,
                x=x,
                y=y_a2b,
                original_length=8,
                n_points=100,
            )
        )
        trajectories.append(
            NormalizedTrajectory(
                source="slow_burn.txt",
                main_pairing=["Grace", "Henry"],
                direction="B_to_A",
                emotion=emotion,
                x=x,
                y=y_b2a,
                original_length=8,
                n_points=100,
            )
        )

    return trajectories


@pytest.fixture
def sample_normalized_trajectories_missing_direction() -> list[NormalizedTrajectory]:
    """Create normalized trajectories where one narrative is missing B_to_A direction.

    Used to test D-13: missing direction filled with baseline (all 1s).
    """
    trajectories: list[NormalizedTrajectory] = []
    x = [float(i) / 99.0 for i in range(100)]

    # Narrative with only A_to_B (no B_to_A)
    for emotion in BASE_EMOTIONS:
        y = [float(i % 3 + 1) for i in range(100)]
        trajectories.append(
            NormalizedTrajectory(
                source="one_direction.txt",
                main_pairing=["A", "B"],
                direction="A_to_B",
                emotion=emotion,
                x=x,
                y=y,
                original_length=5,
                n_points=100,
            )
        )

    return trajectories


@pytest.fixture
def sample_dtw_cluster_result(
    sample_normalized_trajectories_multi: list[NormalizedTrajectory],
) -> DTWClusterResult:
    """Create a sample DTWClusterResult with 2 clusters.

    Cluster 0: rising_romance.txt + slow_burn.txt
    Cluster 1: enemies_to_lovers.txt

    Barycenters are shape (100, 18) numpy arrays with values derived
    from the constituent trajectories.
    """
    sources = ["rising_romance.txt", "enemies_to_lovers.txt", "slow_burn.txt"]

    # Group trajectories by source for barycenter computation
    grouped: dict[str, dict[tuple[str, str], list[float]]] = {}
    for traj in sample_normalized_trajectories_multi:
        if traj.source not in grouped:
            grouped[traj.source] = {}
        grouped[traj.source][(traj.direction, traj.emotion)] = list(traj.y)

    # Build barycenters: mean of cluster members
    n_points = 100

    # Cluster 0: rising_romance + slow_burn
    bary_0 = np.zeros((n_points, 18), dtype=np.float64)
    for emotion_idx, emotion in enumerate(BASE_EMOTIONS):
        for dir_offset, direction in enumerate(["A_to_B", "B_to_A"]):
            col = emotion_idx * 2 + dir_offset
            vals_0 = np.array(grouped["rising_romance.txt"][(direction, emotion)])
            vals_1 = np.array(grouped["slow_burn.txt"][(direction, emotion)])
            bary_0[:, col] = (vals_0 + vals_1) / 2.0

    # Cluster 1: enemies_to_lovers
    bary_1 = np.zeros((n_points, 18), dtype=np.float64)
    for emotion_idx, emotion in enumerate(BASE_EMOTIONS):
        for dir_offset, direction in enumerate(["A_to_B", "B_to_A"]):
            col = emotion_idx * 2 + dir_offset
            bary_1[:, col] = np.array(
                grouped["enemies_to_lovers.txt"][(direction, emotion)]
            )

    # Distance matrix: 3x3 symmetric
    distance_matrix = np.array(
        [
            [0.0, 3.0, 1.0],
            [3.0, 0.0, 4.0],
            [1.0, 4.0, 0.0],
        ],
        dtype=np.float64,
    )

    return DTWClusterResult(
        assignments={
            "rising_romance.txt": 0,
            "enemies_to_lovers.txt": 1,
            "slow_burn.txt": 0,
        },
        barycenters=[bary_0, bary_1],
        distance_matrix=distance_matrix,
        n_clusters=2,
        metric="dtw",
        sakoe_chiba_radius=2,
        cluster_sizes={"0": 2, "1": 1},
        silhouette_score=0.5,
        sources=sources,
    )
