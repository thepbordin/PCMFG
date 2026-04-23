"""Tests for PCMFG narrative normalizer.

Tests the normalization of variable-length emotion time-series to a
uniform [0.0, 1.0] grid using nearest-neighbor interpolation.
"""

import numpy as np
import pytest
from pydantic import ValidationError

from pcmfg.models.schemas import (
    BASE_EMOTIONS,
    AnalysisResult,
    AnalysisMetadata,
    EmotionTimeSeries,
    NormalizedTrajectory,
    WorldBuilderOutput,
    ChunkAnalysis,
)


# --- NORM-01: Grid range and configurable points ---


class TestNorm01GridAndPoints:
    """NORM-01: Map time-series to [0.0, 1.0] grid with configurable resampling points."""

    def test_normalize_grid_range(self, sample_analysis_result):
        """Output x values span [0.0, 1.0]."""
        from pcmfg.analysis.normalizer import NarrativeNormalizer

        normalizer = NarrativeNormalizer(n_points=100)
        trajectories = normalizer.normalize(sample_analysis_result)

        for traj in trajectories:
            assert traj.x[0] == 0.0, f"First grid point must be 0.0, got {traj.x[0]}"
            assert traj.x[-1] == pytest.approx(1.0), f"Last grid point must be ~1.0, got {traj.x[-1]}"
            assert all(traj.x[i] <= traj.x[i + 1] for i in range(len(traj.x) - 1)), "Grid must be monotonically increasing"

    def test_n_points_configurable(self, sample_analysis_result):
        """n_points parameter controls output length."""
        from pcmfg.analysis.normalizer import NarrativeNormalizer

        for n in [10, 50, 100, 200]:
            normalizer = NarrativeNormalizer(n_points=n)
            trajectories = normalizer.normalize(sample_analysis_result)
            for traj in trajectories:
                assert len(traj.x) == n, f"Expected {n} grid points, got {len(traj.x)}"
                assert len(traj.y) == n, f"Expected {n} values, got {len(traj.y)}"

    def test_n_points_minimum(self):
        """n_points must be >= 2."""
        from pcmfg.analysis.normalizer import NarrativeNormalizer

        with pytest.raises(ValueError):
            NarrativeNormalizer(n_points=1)

        with pytest.raises(ValueError):
            NarrativeNormalizer(n_points=0)


# --- NORM-02: Event order preserved ---


class TestNorm02EventOrder:
    """NORM-02: Event order preserved after resampling."""

    def test_event_order_preserved(self, sample_analysis_result):
        """Monotonic emotion changes in input produce monotonic changes in output.

        A_to_B Joy goes 1->3->2->4->5 (non-monotonic).
        After resampling, the general shape must be preserved:
        - First third should have lower Joy than last third (overall upward trend).
        """
        from pcmfg.analysis.normalizer import NarrativeNormalizer

        normalizer = NarrativeNormalizer(n_points=100)
        trajectories = normalizer.normalize(sample_analysis_result)

        joy_a2b = next(t for t in trajectories if t.direction == "A_to_B" and t.emotion == "Joy")

        first_quarter = np.mean(joy_a2b.y[:20])
        last_quarter = np.mean(joy_a2b.y[-20:])
        assert last_quarter >= first_quarter, (
            f"Joy should trend upward: first quarter avg={first_quarter}, "
            f"last quarter avg={last_quarter}"
        )


# --- NORM-03: Variable-length input ---


class TestNorm03VariableLength:
    """NORM-03: Handle variable-length input sequences."""

    def test_variable_length_input(self):
        """Narratives with 12 and 47 chunks produce same-length output."""
        from pcmfg.analysis.normalizer import NarrativeNormalizer

        normalizer = NarrativeNormalizer(n_points=100)

        result_12 = self._make_result(n_chunks=12)
        trajectories_12 = normalizer.normalize(result_12)

        result_47 = self._make_result(n_chunks=47)
        trajectories_47 = normalizer.normalize(result_47)

        assert len(trajectories_12) == len(trajectories_47) == 18
        for t12, t47 in zip(trajectories_12, trajectories_47):
            assert len(t12.x) == len(t47.x) == 100

    def test_single_point_input(self):
        """A narrative with only 1 chunk is handled gracefully."""
        from pcmfg.analysis.normalizer import NarrativeNormalizer

        normalizer = NarrativeNormalizer(n_points=50)
        result = self._make_result(n_chunks=1)
        trajectories = normalizer.normalize(result)

        assert len(trajectories) == 18
        for traj in trajectories:
            assert len(traj.y) == 50
            assert all(v == traj.y[0] for v in traj.y), "Single-point input should repeat the same value"

    def test_empty_timeseries_direction(self):
        """Missing direction in timeseries dict is handled gracefully."""
        from pcmfg.analysis.normalizer import NarrativeNormalizer

        normalizer = NarrativeNormalizer(n_points=50)
        result = AnalysisResult(
            metadata=AnalysisMetadata(source="test.txt", total_chunks=3),
            world_builder=WorldBuilderOutput(main_pairing=["A", "B"]),
            chunks=[
                ChunkAnalysis(chunk_id=i, position=i / 2.0, chunk_main_pov="A")
                for i in range(3)
            ],
            timeseries={
                "A_to_B": EmotionTimeSeries(
                    Joy=[1.0, 2.0, 3.0],
                    Trust=[1.0, 1.0, 2.0],
                    Fear=[1.0, 1.0, 1.0],
                    Surprise=[1.0, 1.0, 1.0],
                    Sadness=[1.0, 1.0, 1.0],
                    Disgust=[1.0, 1.0, 1.0],
                    Anger=[1.0, 1.0, 1.0],
                    Anticipation=[1.0, 1.0, 1.0],
                    Arousal=[1.0, 1.0, 1.0],
                ),
            },
        )
        trajectories = normalizer.normalize(result)
        assert len(trajectories) == 9
        assert all(t.direction == "A_to_B" for t in trajectories)

    @staticmethod
    def _make_result(n_chunks: int) -> AnalysisResult:
        """Helper to create AnalysisResult with n_chunks."""
        positions = np.linspace(0.0, 1.0, n_chunks).tolist()
        chunks = [
            ChunkAnalysis(
                chunk_id=i,
                position=positions[i],
                chunk_main_pov="Alice",
                characters_present=["Alice", "Bob"],
            )
            for i in range(n_chunks)
        ]
        emotion_values = np.random.RandomState(42).randint(1, 6, size=(n_chunks, 9)).astype(float)

        ts_data = {}
        for direction in ["A_to_B", "B_to_A"]:
            ts_data[direction] = EmotionTimeSeries(**{
                emotion: emotion_values[:, j].tolist()
                for j, emotion in enumerate(BASE_EMOTIONS)
            })

        return AnalysisResult(
            metadata=AnalysisMetadata(source=f"test_{n_chunks}_chunks.txt", total_chunks=n_chunks),
            world_builder=WorldBuilderOutput(main_pairing=["Alice", "Bob"]),
            chunks=chunks,
            timeseries=ts_data,
        )


# --- NORM-04: Nearest-neighbor interpolation ---


class TestNorm04NearestNeighbor:
    """NORM-04: Use nearest-neighbor interpolation for ordinal scores."""

    def test_output_values_are_integers(self, sample_analysis_result):
        """All output y values are integers 1-5."""
        from pcmfg.analysis.normalizer import NarrativeNormalizer

        normalizer = NarrativeNormalizer(n_points=100)
        trajectories = normalizer.normalize(sample_analysis_result)

        for traj in trajectories:
            for val in traj.y:
                assert val == int(val), f"Value {val} is not an integer"
                assert 1 <= val <= 5, f"Value {val} out of range [1, 5]"

    def test_no_fractional_values(self, sample_analysis_result):
        """No fractional values like 2.3 or 3.7 in output."""
        from pcmfg.analysis.normalizer import NarrativeNormalizer

        normalizer = NarrativeNormalizer(n_points=100)
        trajectories = normalizer.normalize(sample_analysis_result)

        for traj in trajectories:
            for val in traj.y:
                assert val == round(val), f"Fractional value found: {val}"

    def test_values_from_original_data(self, sample_analysis_result):
        """Every output value exists in the original input."""
        from pcmfg.analysis.normalizer import NarrativeNormalizer

        normalizer = NarrativeNormalizer(n_points=100)
        trajectories = normalizer.normalize(sample_analysis_result)

        for traj in trajectories:
            ts = sample_analysis_result.timeseries[traj.direction]
            original_values = set(getattr(ts, traj.emotion))
            for val in traj.y:
                assert val in original_values, (
                    f"Output value {val} not found in original values {original_values}"
                )


# --- INTG-01: Backward compatibility ---


class TestIntg01BackwardCompatibility:
    """INTG-01: Existing clusterer imports still work unchanged."""

    def test_clusterer_imports_unchanged(self):
        """SceneClusterer and TrajectoryClusterer still importable."""
        from pcmfg.analysis.clusterer import ClusterResult, SceneClusterer, TrajectoryClusterer

        assert SceneClusterer is not None
        assert TrajectoryClusterer is not None
        assert ClusterResult is not None

    def test_feature_extractor_imports_unchanged(self):
        """FeatureExtractor and related types still importable."""
        from pcmfg.analysis.feature_extractor import (
            ExtractedFeatures,
            FeatureExtractor,
            FeatureType,
            SceneFeatures,
        )

        assert FeatureExtractor is not None
        assert FeatureType is not None

    def test_existing_models_unchanged(self):
        """AnalysisResult, EmotionTimeSeries, ChunkAnalysis unchanged."""
        from pcmfg.models.schemas import (
            AnalysisResult,
            ChunkAnalysis,
            EmotionTimeSeries,
        )

        fields = set(AnalysisResult.model_fields.keys())
        assert "metadata" in fields
        assert "world_builder" in fields
        assert "chunks" in fields
        assert "timeseries" in fields


# --- INTG-02: Accept AnalysisResult from JSON ---


class TestIntg02FromJson:
    """INTG-02: Accept existing AnalysisResult JSON files as input."""

    def test_from_analysis_result_json(self, sample_analysis_result):
        """Round-trip: AnalysisResult -> JSON -> parse -> normalize."""
        import json

        from pcmfg.analysis.normalizer import NarrativeNormalizer

        json_str = sample_analysis_result.model_dump_json()
        parsed = AnalysisResult.model_validate_json(json_str)

        normalizer = NarrativeNormalizer(n_points=50)
        trajectories = normalizer.normalize(parsed)

        assert len(trajectories) == 18
        for traj in trajectories:
            assert len(traj.y) == 50
            assert traj.source == "test_novel.txt"


# --- INTG-03: tslearn importable ---


class TestIntg03TslearnAvailable:
    """INTG-03: tslearn is importable."""

    def test_tslearn_importable(self):
        """tslearn can be imported."""
        import tslearn  # noqa: F401

        assert hasattr(tslearn, "__version__")
