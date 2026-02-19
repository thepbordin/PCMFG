"""Tests for Phase 2 - Normalizer."""

import pytest

from pcmfg.models.schemas import (
    ChunkAnalysis,
    DirectedEmotion,
    DirectedEmotionScores,
)
from pcmfg.phase2.normalizer import EmotionNormalizer, ValidationWarning


class TestEmotionNormalizer:
    """Tests for the EmotionNormalizer class."""

    def test_init(self) -> None:
        """Test normalizer initialization."""
        normalizer = EmotionNormalizer()
        assert normalizer.max_fives_allowed == 3

        normalizer_custom = EmotionNormalizer(max_fives_allowed=5)
        assert normalizer_custom.max_fives_allowed == 5

    def test_validate_scores_valid(self, sample_chunk_analysis: ChunkAnalysis) -> None:
        """Test validation of valid scores."""
        normalizer = EmotionNormalizer()
        assert normalizer.validate_scores(sample_chunk_analysis) is True

    def test_validate_scores_all_ones(self) -> None:
        """Test validation of all-1 scores (baseline)."""
        normalizer = EmotionNormalizer()

        emotion = DirectedEmotion(
            source="A",
            target="B",
            scores=DirectedEmotionScores(),  # All defaults to 1
            justification_quote="Normal interaction",
        )
        chunk = ChunkAnalysis(
            chunk_id=0,
            chunk_main_pov="A",
            characters_present=["A", "B"],
            directed_emotions=[emotion],
            scene_summary="A normal conversation",
        )

        assert normalizer.validate_scores(chunk) is True

    def test_check_justification_with_high_scores(self) -> None:
        """Test justification check with high scores."""
        normalizer = EmotionNormalizer()

        # High scores without justification
        high_scores = DirectedEmotionScores(Joy=4, Trust=3)
        emotion = DirectedEmotion(
            source="A",
            target="B",
            scores=high_scores,
            justification_quote="",  # Empty justification
        )
        chunk = ChunkAnalysis(
            chunk_id=0,
            chunk_main_pov="A",
            characters_present=["A", "B"],
            directed_emotions=[emotion],
            scene_summary="Test",
        )

        warnings = normalizer.check_justification(chunk)
        assert len(warnings) == 1
        assert warnings[0].warning_type == "missing_justification"

    def test_check_justification_with_justification(self) -> None:
        """Test justification check with proper justification."""
        normalizer = EmotionNormalizer()

        high_scores = DirectedEmotionScores(Joy=4, Trust=3)
        emotion = DirectedEmotion(
            source="A",
            target="B",
            scores=high_scores,
            justification_quote="She smiled warmly at him.",
        )
        chunk = ChunkAnalysis(
            chunk_id=0,
            chunk_main_pov="A",
            characters_present=["A", "B"],
            directed_emotions=[emotion],
            scene_summary="Test",
        )

        warnings = normalizer.check_justification(chunk)
        assert len(warnings) == 0

    def test_check_hallucination_many_fives(self) -> None:
        """Test hallucination check with many 5s."""
        normalizer = EmotionNormalizer(max_fives_allowed=3)

        # 5 scores of 5
        many_fives = DirectedEmotionScores(
            Joy=5, Trust=5, Fear=5, Surprise=5, Sadness=1,
            Disgust=1, Anger=1, Anticipation=1, Arousal=1,
        )
        emotion = DirectedEmotion(
            source="A",
            target="B",
            scores=many_fives,
            justification_quote="Test",
        )
        chunk = ChunkAnalysis(
            chunk_id=0,
            chunk_main_pov="A",
            characters_present=["A", "B"],
            directed_emotions=[emotion],
            scene_summary="Test",
        )

        warnings = normalizer.check_hallucination(chunk)
        assert len(warnings) == 1
        assert warnings[0].warning_type == "potential_hallucination"

    def test_check_hallucination_acceptable_fives(self) -> None:
        """Test hallucination check with acceptable number of 5s."""
        normalizer = EmotionNormalizer(max_fives_allowed=3)

        # Only 2 scores of 5
        few_fives = DirectedEmotionScores(
            Joy=5, Trust=5, Fear=1, Surprise=1, Sadness=1,
            Disgust=1, Anger=1, Anticipation=1, Arousal=1,
        )
        emotion = DirectedEmotion(
            source="A",
            target="B",
            scores=few_fives,
            justification_quote="Test",
        )
        chunk = ChunkAnalysis(
            chunk_id=0,
            chunk_main_pov="A",
            characters_present=["A", "B"],
            directed_emotions=[emotion],
            scene_summary="Test",
        )

        warnings = normalizer.check_hallucination(chunk)
        assert len(warnings) == 0

    def test_aggregate_bidirectional(self, sample_chunk_analysis: ChunkAnalysis) -> None:
        """Test bidirectional emotion aggregation."""
        normalizer = EmotionNormalizer()

        aggregated = normalizer.aggregate_bidirectional(
            sample_chunk_analysis.directed_emotions
        )

        # Should return averaged scores
        assert isinstance(aggregated, DirectedEmotionScores)
        # Joy was 3 in Alice->Bob and 2 in Bob->Alice, average = 2.5 -> rounded
        assert aggregated.Joy in [2, 3]

    def test_aggregate_bidirectional_empty(self) -> None:
        """Test aggregation with empty emotion list."""
        normalizer = EmotionNormalizer()

        aggregated = normalizer.aggregate_bidirectional([])

        # Should return default scores (all 1s)
        assert aggregated.Joy == 1
        assert aggregated.Trust == 1

    def test_normalize(self, sample_chunk_analysis: ChunkAnalysis) -> None:
        """Test the normalize method."""
        normalizer = EmotionNormalizer()

        result = normalizer.normalize(sample_chunk_analysis)

        # Should return the same chunk (validation only)
        assert result == sample_chunk_analysis

    def test_normalize_all(self, sample_chunk_analysis: ChunkAnalysis) -> None:
        """Test normalize_all method."""
        normalizer = EmotionNormalizer()

        chunks = [sample_chunk_analysis, sample_chunk_analysis]
        results = normalizer.normalize_all(chunks)

        assert len(results) == 2
        assert all(isinstance(c, ChunkAnalysis) for c in results)

    def test_get_aggregated_scores(self, sample_chunk_analysis: ChunkAnalysis) -> None:
        """Test get_aggregated_scores method."""
        normalizer = EmotionNormalizer()

        scores = normalizer.get_aggregated_scores(sample_chunk_analysis)

        assert isinstance(scores, DirectedEmotionScores)
