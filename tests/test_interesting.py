"""Tests for Interesting Section Detection module.

Tests the full pipeline: converter, discord discovery, FLUSS segmentation,
motif discovery, gap analysis, and the orchestrating detector.
"""

from datetime import datetime, timezone

import numpy as np
import pytest

from pcmfg.models.schemas import (
    AnalysisMetadata,
    AnalysisResult,
    BASE_EMOTIONS,
    ChunkAnalysis,
    DirectedEmotion,
    DirectedEmotionScores,
    EmotionTimeSeries,
    InterestingSectionReport,
    WorldBuilderOutput,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_scores(**overrides: int) -> DirectedEmotionScores:
    """Create emotion scores with optional overrides (defaults all 1)."""
    return DirectedEmotionScores(**overrides)


def _make_analysis_result_30() -> AnalysisResult:
    """Create a 30-chunk AnalysisResult with deliberate patterns.

    Patterns:
    - Discord at chunks 14-16: sudden spike in all negative emotions
    - Motif at chunks 4-7 and 22-25: nearly identical "argument" pattern
      (high Anger, low Joy, moderate Fear)
    - Structural shift around chunk 10: transition from "conflict" pattern
      (high Anger) to "resolution" pattern (high Joy, Trust)
    - Gap at climax (14-16): A feels intense Fear/Sadness, B feels Arousal/Joy
    """
    chunks: list[ChunkAnalysis] = []

    # Base emotional profiles for each section
    for i in range(30):
        position = i / 29.0

        if i < 10:
            # Section 1: Conflict pattern (Anger high, Joy low)
            a_scores = _make_scores(Joy=1, Anger=3 + (i % 2), Trust=1, Anticipation=2)
            b_scores = _make_scores(Joy=1, Anger=2, Trust=1)
        elif i < 14:
            # Section 2: Transition (emotions shift toward positive)
            a_scores = _make_scores(Joy=2 + (i - 10), Anger=2 - (i - 10) // 2, Trust=2)
            b_scores = _make_scores(Joy=2, Trust=2 + (i - 10))
        elif 14 <= i <= 16:
            # Section 3: Climax/Discord — sudden spike
            a_scores = _make_scores(Joy=1, Fear=5, Sadness=5, Anger=4, Surprise=5, Disgust=4)
            b_scores = _make_scores(Joy=4, Arousal=5, Anticipation=3, Fear=1)
        elif i < 22:
            # Section 4: Aftermath (recovering)
            a_scores = _make_scores(Joy=2, Trust=2, Fear=2, Sadness=2)
            b_scores = _make_scores(Joy=3, Trust=3, Arousal=2)
        else:
            # Section 5: Resolution (positive)
            a_scores = _make_scores(Joy=4, Trust=4, Arousal=3, Anticipation=3)
            b_scores = _make_scores(Joy=4, Trust=4, Arousal=3)

        # Inject motif pattern at chunks 4-7 (same as 22-25)
        if 4 <= i <= 7:
            a_scores = _make_scores(Joy=1, Anger=4, Fear=3, Trust=1, Anticipation=2)
            b_scores = _make_scores(Joy=1, Anger=3, Fear=2, Trust=1)
        if 22 <= i <= 25:
            a_scores = _make_scores(Joy=1, Anger=4, Fear=3, Trust=1, Anticipation=2)
            b_scores = _make_scores(Joy=1, Anger=3, Fear=2, Trust=1)

        chunks.append(
            ChunkAnalysis(
                chunk_id=i,
                position=position,
                chunk_main_pov="Alice",
                characters_present=["Alice", "Bob"],
                directed_emotions=[
                    DirectedEmotion(
                        source="Alice",
                        target="Bob",
                        scores=a_scores,
                        justification_quote=f"Chunk {i} A->B",
                    ),
                    DirectedEmotion(
                        source="Bob",
                        target="Alice",
                        scores=b_scores,
                        justification_quote=f"Chunk {i} B->A",
                    ),
                ],
                scene_summary=f"Scene {i}",
            )
        )

    # Build timeseries from chunk directed_emotions
    a_to_b_data: dict[str, list[float]] = {e: [] for e in BASE_EMOTIONS}
    b_to_a_data: dict[str, list[float]] = {e: [] for e in BASE_EMOTIONS}

    for chunk in chunks:
        for de in chunk.directed_emotions:
            target = a_to_b_data if de.source == "Alice" else b_to_a_data
            for emotion in BASE_EMOTIONS:
                target[emotion].append(float(getattr(de.scores, emotion)))

    return AnalysisResult(
        metadata=AnalysisMetadata(
            source="test_30chunk.txt",
            analysis_date=datetime(2026, 4, 24, tzinfo=timezone.utc),
            model="test",
            total_chunks=30,
        ),
        world_builder=WorldBuilderOutput(
            main_pairing=["Alice", "Bob"],
            aliases={"Alice": ["Ali"], "Bob": ["Robert"]},
            world_guidelines=["Test narrative with deliberate patterns."],
        ),
        chunks=chunks,
        timeseries={
            "A_to_B": EmotionTimeSeries(**a_to_b_data),
            "B_to_A": EmotionTimeSeries(**b_to_a_data),
        },
    )


@pytest.fixture
def result_30() -> AnalysisResult:
    """30-chunk AnalysisResult with deliberate emotional patterns."""
    return _make_analysis_result_30()


# ---------------------------------------------------------------------------
# Converter Tests
# ---------------------------------------------------------------------------


class TestConverter:
    """Test result_to_18d conversion."""

    def test_shape_is_T_x_18(self, result_30: AnalysisResult) -> None:
        from pcmfg.interesting.converter import result_to_18d

        T_18d, positions, chunk_ids = result_to_18d(result_30)
        assert T_18d.shape == (30, 18)
        assert len(positions) == 30
        assert len(chunk_ids) == 30

    def test_column_ordering(self, result_30: AnalysisResult) -> None:
        from pcmfg.interesting.converter import result_to_18d

        T_18d, _, _ = result_to_18d(result_30)
        # Chunk 14 (climax): A->B Fear=5, so col for Fear_A2B = 2*2 = 4
        fear_a2b_col = 2 * 2  # emotion_idx=2 (Fear), dir_offset=0
        assert T_18d[14, fear_a2b_col] == 5.0
        # B->A Arousal=5 at climax, col = 8*2+1 = 17
        arousal_b2a_col = 8 * 2 + 1
        assert T_18d[14, arousal_b2a_col] == 5.0

    def test_positions_from_chunks(self, result_30: AnalysisResult) -> None:
        from pcmfg.interesting.converter import result_to_18d

        _, positions, _ = result_to_18d(result_30)
        assert positions[0] == pytest.approx(0.0)
        assert positions[-1] == pytest.approx(1.0)
        assert positions[15] == pytest.approx(15 / 29.0)

    def test_chunk_ids_match(self, result_30: AnalysisResult) -> None:
        from pcmfg.interesting.converter import result_to_18d

        _, _, chunk_ids = result_to_18d(result_30)
        assert chunk_ids == list(range(30))

    def test_missing_timeseries_raises(self) -> None:
        from pcmfg.interesting.converter import result_to_18d

        result = AnalysisResult()
        with pytest.raises(ValueError, match="A_to_B"):
            result_to_18d(result)

    def test_empty_timeseries_raises(self) -> None:
        from pcmfg.interesting.converter import result_to_18d

        result = AnalysisResult(
            timeseries={"A_to_B": EmotionTimeSeries(), "B_to_A": EmotionTimeSeries()}
        )
        with pytest.raises(ValueError, match="empty"):
            result_to_18d(result)


# ---------------------------------------------------------------------------
# Discord Discovery Tests
# ---------------------------------------------------------------------------


class TestDiscordDiscovery:
    """Test discover_discords."""

    def test_finds_climax_discord(self, result_30: AnalysisResult) -> None:
        from pcmfg.interesting.converter import result_to_18d
        from pcmfg.interesting.discord import discover_discords
        import stumpy

        T_18d, positions, chunk_ids = result_to_18d(result_30)
        T_1d = np.linalg.norm(T_18d, axis=1)
        m = 3
        mp = stumpy.stump(T_1d, m)
        mp_distances = mp[:, 0].astype(float)
        mp_distances = np.nan_to_num(mp_distances, nan=0.0, posinf=0.0, neginf=0.0)

        discords = discover_discords(mp_distances, m, positions, chunk_ids, k=3)
        assert len(discords) <= 3
        # The climax at chunks 14-16 should be among top discords
        discord_indices = {d.index for d in discords}
        # At least one discord should be near the climax
        assert any(12 <= idx <= 16 for idx in discord_indices), (
            f"Expected discord near climax (14-16), got {discord_indices}"
        )

    def test_no_overlap_in_discords(self, result_30: AnalysisResult) -> None:
        from pcmfg.interesting.converter import result_to_18d
        from pcmfg.interesting.discord import discover_discords
        import stumpy

        T_18d, positions, chunk_ids = result_to_18d(result_30)
        T_1d = np.linalg.norm(T_18d, axis=1)
        m = 3
        mp = stumpy.stump(T_1d, m)
        mp_distances = mp[:, 0].astype(float)
        mp_distances = np.nan_to_num(mp_distances, nan=0.0, posinf=0.0, neginf=0.0)

        discords = discover_discords(mp_distances, m, positions, chunk_ids, k=5)
        for i in range(len(discords)):
            for j in range(i + 1, len(discords)):
                assert abs(discords[i].index - discords[j].index) >= m

    def test_empty_mp_returns_empty(self) -> None:
        from pcmfg.interesting.discord import discover_discords

        result = discover_discords(
            np.array([]), 3, np.array([]), [], k=3
        )
        assert result == []

    def test_sorted_by_distance_descending(self, result_30: AnalysisResult) -> None:
        from pcmfg.interesting.converter import result_to_18d
        from pcmfg.interesting.discord import discover_discords
        import stumpy

        T_18d, positions, chunk_ids = result_to_18d(result_30)
        T_1d = np.linalg.norm(T_18d, axis=1)
        m = 3
        mp = stumpy.stump(T_1d, m)
        mp_distances = mp[:, 0].astype(float)
        mp_distances = np.nan_to_num(mp_distances, nan=0.0, posinf=0.0, neginf=0.0)

        discords = discover_discords(mp_distances, m, positions, chunk_ids, k=5)
        for i in range(len(discords) - 1):
            assert discords[i].distance >= discords[i + 1].distance


# ---------------------------------------------------------------------------
# Gap Analysis Tests
# ---------------------------------------------------------------------------


class TestGapAnalysis:
    """Test compute_gaps."""

    def test_gap_computation_correct(self, result_30: AnalysisResult) -> None:
        from pcmfg.interesting.gap import compute_gaps

        # At chunk 14 (climax): A->B Fear=5, B->A Fear=1 → gap = 4.0
        gaps = compute_gaps(result_30, [14])
        assert len(gaps) == 1
        fear_gap = next(g for g in gaps[0].gaps if g.emotion == "Fear")
        assert fear_gap.a_to_b == 5.0
        assert fear_gap.b_to_a == 1.0
        assert fear_gap.gap == pytest.approx(4.0)

    def test_dominant_gap_identified(self, result_30: AnalysisResult) -> None:
        from pcmfg.interesting.gap import compute_gaps

        gaps = compute_gaps(result_30, [14])
        # At climax: Fear gap = 5-1 = 4, Sadness gap = 5-1 = 4, etc.
        assert gaps[0].dominant_gap_emotion in ("Fear", "Sadness")
        assert gaps[0].dominant_gap_value >= 4.0

    def test_all_indices_mode(self, result_30: AnalysisResult) -> None:
        from pcmfg.interesting.gap import compute_gaps

        gaps = compute_gaps(result_30, [], all_indices=True)
        assert len(gaps) == 30

    def test_zero_gap_at_baseline(self) -> None:
        from pcmfg.interesting.gap import compute_gaps

        result = AnalysisResult(
            chunks=[
                ChunkAnalysis(chunk_id=0, position=0.0, chunk_main_pov="A"),
            ],
            timeseries={
                "A_to_B": EmotionTimeSeries(Joy=[1.0], Trust=[1.0], Fear=[1.0], Surprise=[1.0], Sadness=[1.0], Disgust=[1.0], Anger=[1.0], Anticipation=[1.0], Arousal=[1.0]),
                "B_to_A": EmotionTimeSeries(Joy=[1.0], Trust=[1.0], Fear=[1.0], Surprise=[1.0], Sadness=[1.0], Disgust=[1.0], Anger=[1.0], Anticipation=[1.0], Arousal=[1.0]),
            },
        )
        gaps = compute_gaps(result, [0])
        assert len(gaps) == 1
        for gv in gaps[0].gaps:
            assert gv.gap == pytest.approx(0.0)

    def test_missing_timeseries_returns_empty(self) -> None:
        from pcmfg.interesting.gap import compute_gaps

        result = AnalysisResult()
        gaps = compute_gaps(result, [0])
        assert gaps == []


# ---------------------------------------------------------------------------
# Detector Integration Tests
# ---------------------------------------------------------------------------


class TestDetectorIntegration:
    """Test the full InterestingSectionDetector pipeline."""

    def test_full_detection_returns_report(self, result_30: AnalysisResult) -> None:
        from pcmfg.interesting.detector import InterestingSectionDetector

        detector = InterestingSectionDetector(
            n_discords=3,
            n_regimes=3,
            n_motifs=2,
        )
        report = detector.detect(result_30)

        assert isinstance(report, InterestingSectionReport)
        assert report.n_chunks == 30
        assert report.source == "test_30chunk.txt"
        assert report.main_pairing == ["Alice", "Bob"]
        assert report.window_size > 0

    def test_discords_detected(self, result_30: AnalysisResult) -> None:
        from pcmfg.interesting.detector import InterestingSectionDetector

        detector = InterestingSectionDetector(n_discords=3)
        report = detector.detect(result_30)

        assert len(report.discords) >= 1
        for d in report.discords:
            assert d.distance > 0
            assert 0.0 <= d.position <= 1.0

    def test_segments_detected(self, result_30: AnalysisResult) -> None:
        from pcmfg.interesting.detector import InterestingSectionDetector

        detector = InterestingSectionDetector(n_regimes=3)
        report = detector.detect(result_30)

        assert len(report.segments) >= 1
        for s in report.segments:
            assert s.regime_label in (
                "Intro", "Conflict", "Resolution",
                "Act 1: Setup", "Act 2A: Rising", "Act 2B: Crisis", "Act 3: Resolution",
            )

    def test_gaps_at_interesting_points(self, result_30: AnalysisResult) -> None:
        from pcmfg.interesting.detector import InterestingSectionDetector

        detector = InterestingSectionDetector(n_discords=3, n_regimes=3)
        report = detector.detect(result_30)

        assert len(report.gaps) >= 1
        for g in report.gaps:
            assert len(g.gaps) == 9

    def test_serialization_roundtrip(self, result_30: AnalysisResult) -> None:
        from pcmfg.interesting.detector import InterestingSectionDetector

        detector = InterestingSectionDetector(n_discords=2)
        report = detector.detect(result_30)

        data = report.model_dump(mode="json")
        restored = InterestingSectionReport.model_validate(data)

        assert restored.source == report.source
        assert restored.n_chunks == report.n_chunks
        assert len(restored.discords) == len(report.discords)
        assert len(restored.matrix_profile_distances) == len(
            report.matrix_profile_distances
        )

    def test_short_timeseries_handled(self) -> None:
        from pcmfg.interesting.detector import InterestingSectionDetector

        result = AnalysisResult(
            chunks=[
                ChunkAnalysis(chunk_id=i, position=i / 4.0, chunk_main_pov="A")
                for i in range(5)
            ],
            timeseries={
                "A_to_B": EmotionTimeSeries(
                    Joy=[1.0, 2.0, 3.0, 2.0, 1.0],
                    Trust=[1.0, 1.0, 1.0, 1.0, 1.0],
                    Fear=[1.0, 1.0, 1.0, 1.0, 1.0],
                    Surprise=[1.0, 1.0, 1.0, 1.0, 1.0],
                    Sadness=[1.0, 1.0, 1.0, 1.0, 1.0],
                    Disgust=[1.0, 1.0, 1.0, 1.0, 1.0],
                    Anger=[1.0, 1.0, 1.0, 1.0, 1.0],
                    Anticipation=[1.0, 1.0, 1.0, 1.0, 1.0],
                    Arousal=[1.0, 1.0, 1.0, 1.0, 1.0],
                ),
                "B_to_A": EmotionTimeSeries(
                    Joy=[1.0, 1.0, 1.0, 1.0, 1.0],
                    Trust=[1.0, 1.0, 1.0, 1.0, 1.0],
                    Fear=[1.0, 1.0, 1.0, 1.0, 1.0],
                    Surprise=[1.0, 1.0, 1.0, 1.0, 1.0],
                    Sadness=[1.0, 1.0, 1.0, 1.0, 1.0],
                    Disgust=[1.0, 1.0, 1.0, 1.0, 1.0],
                    Anger=[1.0, 1.0, 1.0, 1.0, 1.0],
                    Anticipation=[1.0, 1.0, 1.0, 1.0, 1.0],
                    Arousal=[1.0, 1.0, 1.0, 1.0, 1.0],
                ),
            },
        )

        detector = InterestingSectionDetector()
        report = detector.detect(result)
        assert isinstance(report, InterestingSectionReport)
        assert report.n_chunks == 5

    def test_very_short_timeseries_empty_report(self) -> None:
        from pcmfg.interesting.detector import InterestingSectionDetector

        result = AnalysisResult(
            chunks=[
                ChunkAnalysis(chunk_id=0, position=0.0, chunk_main_pov="A"),
                ChunkAnalysis(chunk_id=1, position=1.0, chunk_main_pov="A"),
            ],
            timeseries={
                "A_to_B": EmotionTimeSeries(
                    Joy=[1.0, 1.0], Trust=[1.0, 1.0], Fear=[1.0, 1.0],
                    Surprise=[1.0, 1.0], Sadness=[1.0, 1.0], Disgust=[1.0, 1.0],
                    Anger=[1.0, 1.0], Anticipation=[1.0, 1.0], Arousal=[1.0, 1.0],
                ),
                "B_to_A": EmotionTimeSeries(
                    Joy=[1.0, 1.0], Trust=[1.0, 1.0], Fear=[1.0, 1.0],
                    Surprise=[1.0, 1.0], Sadness=[1.0, 1.0], Disgust=[1.0, 1.0],
                    Anger=[1.0, 1.0], Anticipation=[1.0, 1.0], Arousal=[1.0, 1.0],
                ),
            },
        )

        detector = InterestingSectionDetector()
        report = detector.detect(result)
        assert report.discords == []
        assert report.n_chunks == 2

    def test_mp_distances_populated(self, result_30: AnalysisResult) -> None:
        from pcmfg.interesting.detector import InterestingSectionDetector

        detector = InterestingSectionDetector()
        report = detector.detect(result_30)

        assert len(report.matrix_profile_distances) > 0
        assert all(isinstance(d, float) for d in report.matrix_profile_distances)
