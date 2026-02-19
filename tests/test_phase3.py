"""Tests for Phase 3 - Axis Mapper."""

import pytest

from pcmfg.models.schemas import (
    AxesTimeSeries,
    AxisValues,
    ChunkAnalysis,
    DirectedEmotion,
    DirectedEmotionScores,
)
from pcmfg.phase3.axis_mapper import AxisMapper


class TestAxisMapper:
    """Tests for the AxisMapper class."""

    def test_init(self) -> None:
        """Test mapper initialization."""
        mapper = AxisMapper()
        assert mapper is not None

    def test_compute_axes_baseline(self) -> None:
        """Test axis computation with baseline scores (all 1s)."""
        mapper = AxisMapper()
        scores = DirectedEmotionScores()  # All 1s

        result = mapper.compute_axes(scores)

        assert isinstance(result, AxisValues)
        # All formulas should result in 1.0 when all inputs are 1
        assert result.intimacy == 1.0
        assert result.passion == 1.0
        assert result.hostility == 1.0
        assert result.anxiety == 1.0

    def test_compute_axes_high_intimacy(self) -> None:
        """Test axis computation with high intimacy emotions."""
        mapper = AxisMapper()
        # High Joy and Trust -> High Intimacy
        scores = DirectedEmotionScores(Joy=5, Trust=5)

        result = mapper.compute_axes(scores)

        # Intimacy = (Trust + Joy) / 2 = (5 + 5) / 2 = 5.0
        assert result.intimacy == 5.0
        # Passion also includes Joy, so it should be elevated
        assert result.passion > 1.0

    def test_compute_axes_high_hostility(self) -> None:
        """Test axis computation with high hostility emotions."""
        mapper = AxisMapper()
        # High Anger, Disgust, Sadness -> High Hostility
        scores = DirectedEmotionScores(Anger=5, Disgust=5, Sadness=5)

        result = mapper.compute_axes(scores)

        # Hostility = (Anger + Disgust + Sadness) / 3 = (5 + 5 + 5) / 3 = 5.0
        assert result.hostility == 5.0
        # Anxiety also includes Sadness, so it should be elevated
        assert result.anxiety > 1.0

    def test_compute_axes_high_passion(self) -> None:
        """Test axis computation with high passion emotions."""
        mapper = AxisMapper()
        # High Arousal, Anticipation, Joy -> High Passion
        scores = DirectedEmotionScores(Arousal=5, Anticipation=5, Joy=5)

        result = mapper.compute_axes(scores)

        # Passion = (Arousal + Anticipation + Joy) / 3 = 5.0
        assert result.passion == 5.0
        # Intimacy also includes Joy, so it should be elevated
        assert result.intimacy > 1.0

    def test_compute_axes_high_anxiety(self) -> None:
        """Test axis computation with high anxiety emotions."""
        mapper = AxisMapper()
        # High Fear, Surprise, Sadness -> High Anxiety
        scores = DirectedEmotionScores(Fear=5, Surprise=5, Sadness=5)

        result = mapper.compute_axes(scores)

        # Anxiety = (Fear + Surprise + Sadness) / 3 = 5.0
        assert result.anxiety == 5.0
        # Hostility also includes Sadness, so it should be elevated
        assert result.hostility > 1.0

    def test_compute_axes_mixed_emotions(self) -> None:
        """Test axis computation with mixed emotions."""
        mapper = AxisMapper()
        scores = DirectedEmotionScores(
            Joy=3, Trust=2,  # Intimacy: (3+2)/2 = 2.5
            Arousal=2, Anticipation=3,  # Passion: (2+3+3)/3 = 2.67
            Anger=2, Disgust=1, Sadness=1,  # Hostility: (2+1+1)/3 = 1.33
            Fear=1, Surprise=2,  # Anxiety: (1+2+1)/3 = 1.33
        )

        result = mapper.compute_axes(scores)

        assert result.intimacy == 2.5
        assert 2.6 <= result.passion <= 2.7
        assert 1.3 <= result.hostility <= 1.4
        assert 1.3 <= result.anxiety <= 1.4

    def test_map_chunk(self, sample_chunk_analysis: ChunkAnalysis) -> None:
        """Test mapping a chunk to axes."""
        mapper = AxisMapper()

        result = mapper.map_chunk(sample_chunk_analysis)

        assert isinstance(result, AxisValues)
        # Values should be within 1-5 range
        assert 1.0 <= result.intimacy <= 5.0
        assert 1.0 <= result.passion <= 5.0
        assert 1.0 <= result.hostility <= 5.0
        assert 1.0 <= result.anxiety <= 5.0

    def test_map_chunk_empty_emotions(self) -> None:
        """Test mapping a chunk with no emotions."""
        mapper = AxisMapper()

        chunk = ChunkAnalysis(
            chunk_id=0,
            chunk_main_pov="A",
            characters_present=["A"],
            directed_emotions=[],
            scene_summary="Empty scene",
        )

        result = mapper.map_chunk(chunk)

        # Should return baseline values
        assert result.intimacy == 1.0
        assert result.passion == 1.0
        assert result.hostility == 1.0
        assert result.anxiety == 1.0

    def test_map_chunks(self, sample_chunk_analysis: ChunkAnalysis) -> None:
        """Test mapping multiple chunks to time series."""
        mapper = AxisMapper()

        # Create multiple chunks
        chunk2 = ChunkAnalysis(
            chunk_id=1,
            position=0.5,
            chunk_main_pov="Bob",
            characters_present=["Alice", "Bob"],
            directed_emotions=[
                DirectedEmotion(
                    source="Alice",
                    target="Bob",
                    scores=DirectedEmotionScores(Joy=4, Trust=3),
                    justification_quote="Test",
                ),
                DirectedEmotion(
                    source="Bob",
                    target="Alice",
                    scores=DirectedEmotionScores(Joy=3, Trust=4),
                    justification_quote="Test",
                ),
            ],
            scene_summary="Second scene",
        )

        chunks = [sample_chunk_analysis, chunk2]
        result = mapper.map_chunks(chunks)

        assert isinstance(result, AxesTimeSeries)
        assert len(result.intimacy) == 2
        assert len(result.passion) == 2
        assert len(result.hostility) == 2
        assert len(result.anxiety) == 2

    def test_compute_directional_axes(self, sample_chunk_analysis: ChunkAnalysis) -> None:
        """Test computing axes for each direction separately."""
        mapper = AxisMapper()

        a_to_b, b_to_a = mapper.compute_directional_axes(sample_chunk_analysis)

        assert a_to_b is not None
        assert b_to_a is not None
        assert isinstance(a_to_b, AxisValues)
        assert isinstance(b_to_a, AxisValues)

    def test_compute_directional_axes_single_direction(self) -> None:
        """Test computing axes with only one direction."""
        mapper = AxisMapper()

        emotion = DirectedEmotion(
            source="A",
            target="B",
            scores=DirectedEmotionScores(Joy=3),
            justification_quote="Test",
        )
        chunk = ChunkAnalysis(
            chunk_id=0,
            chunk_main_pov="A",
            characters_present=["A", "B"],
            directed_emotions=[emotion],
            scene_summary="Single direction",
        )

        a_to_b, b_to_a = mapper.compute_directional_axes(chunk)

        assert a_to_b is not None
        assert b_to_a is None

    def test_axis_values_range(self) -> None:
        """Test that axis values are always in 1-5 range."""
        mapper = AxisMapper()

        # Test with various score combinations
        test_cases = [
            DirectedEmotionScores(Joy=1, Trust=1),
            DirectedEmotionScores(Joy=5, Trust=5),
            DirectedEmotionScores(Joy=3, Trust=3, Arousal=3),
            DirectedEmotionScores(Anger=5, Disgust=5, Sadness=5),
        ]

        for scores in test_cases:
            result = mapper.compute_axes(scores)
            assert 1.0 <= result.intimacy <= 5.0, f"Intimacy out of range: {result.intimacy}"
            assert 1.0 <= result.passion <= 5.0, f"Passion out of range: {result.passion}"
            assert 1.0 <= result.hostility <= 5.0, f"Hostility out of range: {result.hostility}"
            assert 1.0 <= result.anxiety <= 5.0, f"Anxiety out of range: {result.anxiety}"
