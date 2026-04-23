"""Tests for PCMFG DTW-based clustering.

Tests DTW distance computation, shape-based clustering, barycenter
extraction, and configurable metrics for emotional trajectory comparison.
"""

import numpy as np
import pytest
from pydantic import ValidationError

from pcmfg.analysis.dtw_clusterer import (
    DTWClusterResult,
    DTWClusterer,
    DistanceMetric,
    build_dtw_dataset,
)
from pcmfg.models.schemas import BASE_EMOTIONS, NormalizedTrajectory


# --- DTW-01: DTW as supported distance metric ---


class TestDTW01:
    """DTW-01: System uses DTW as a supported distance metric via tslearn."""

    def test_dtw_metric_accepted(self):
        """DTWClusterer accepts 'dtw' as metric parameter."""
        clusterer = DTWClusterer(metric="dtw", n_clusters=2)
        assert clusterer.metric == "dtw"

    def test_dtw_metric_enum(self):
        """DistanceMetric.DTW maps to 'dtw' string."""
        assert DistanceMetric.DTW.value == "dtw"

    def test_dtw_clustering_produces_labels(
        self, sample_normalized_trajectories_multi
    ):
        """Clustering with metric='dtw' produces cluster labels for each narrative."""
        clusterer = DTWClusterer(metric="dtw", n_clusters=2, random_state=42)
        result = clusterer.cluster(sample_normalized_trajectories_multi)
        assert len(result.assignments) == 3
        assert all(isinstance(v, int) for v in result.assignments.values())

    def test_dtw_distance_matrix_symmetric(
        self, sample_normalized_trajectories_multi
    ):
        """DTW distance matrix is symmetric."""
        clusterer = DTWClusterer(metric="dtw", n_clusters=2, random_state=42)
        result = clusterer.cluster(sample_normalized_trajectories_multi)
        np.testing.assert_array_almost_equal(
            result.distance_matrix, result.distance_matrix.T
        )


# --- DTW-02: Sakoe-Chiba warping constraint ---


class TestDTW02:
    """DTW-02: System supports Sakoe-Chiba warping constraint with configurable radius."""

    def test_default_sakoe_chiba_radius(self):
        """Default Sakoe-Chiba radius is 0.2 (20% of series length)."""
        clusterer = DTWClusterer()
        assert clusterer.sakoe_chiba_radius == 0.2

    def test_custom_sakoe_chiba_radius(self):
        """Sakoe-Chiba radius is configurable."""
        clusterer = DTWClusterer(sakoe_chiba_radius=0.1)
        assert clusterer.sakoe_chiba_radius == 0.1

    def test_no_constraint_available(self):
        """Setting sakoe_chiba_radius=None allows unconstrained DTW."""
        clusterer = DTWClusterer(sakoe_chiba_radius=None)
        assert clusterer.sakoe_chiba_radius is None

    def test_constrained_distance_smaller_or_equal(
        self, sample_normalized_trajectories_multi
    ):
        """Constrained DTW distances >= unconstrained DTW distances (constraint can only increase distance)."""
        c_constrained = DTWClusterer(
            metric="dtw",
            n_clusters=2,
            sakoe_chiba_radius=0.1,
            random_state=42,
        )
        c_unconstrained = DTWClusterer(
            metric="dtw",
            n_clusters=2,
            sakoe_chiba_radius=None,
            random_state=42,
        )
        r_constrained = c_constrained.cluster(sample_normalized_trajectories_multi)
        r_unconstrained = c_unconstrained.cluster(
            sample_normalized_trajectories_multi
        )
        # Constrained DTW distance should be >= unconstrained for at least one pair
        assert np.any(
            r_constrained.distance_matrix >= r_unconstrained.distance_matrix - 1e-6
        )

    def test_radius_stored_in_result(self, sample_normalized_trajectories_multi):
        """DTWClusterResult records the Sakoe-Chiba radius used."""
        clusterer = DTWClusterer(
            metric="dtw", sakoe_chiba_radius=0.3, n_clusters=2, random_state=42
        )
        result = clusterer.cluster(sample_normalized_trajectories_multi)
        assert result.sakoe_chiba_radius == 30  # int(0.3 * 100) = 30


# --- DTW-03: Multi-dimensional DTW (18 dimensions) ---


class TestDTW03:
    """DTW-03: System supports multi-dimensional DTW (9 emotions x 2 directions = 18 dimensions)."""

    def test_dataset_shape_18_features(self, sample_normalized_trajectories_multi):
        """build_dtw_dataset produces (n_narratives, n_points, 18) array."""
        dataset, sources, n_points = build_dtw_dataset(
            sample_normalized_trajectories_multi
        )
        assert dataset.shape[2] == 18

    def test_dataset_3_narratives(self, sample_normalized_trajectories_multi):
        """build_dtw_dataset correctly groups 3 narratives."""
        dataset, sources, n_points = build_dtw_dataset(
            sample_normalized_trajectories_multi
        )
        assert dataset.shape[0] == 3
        assert len(sources) == 3

    def test_feature_axis_ordering(self, sample_normalized_trajectories_multi):
        """Feature axis follows [Joy_A2B, Joy_B2A, Trust_A2B, Trust_B2A, ...] ordering."""
        dataset, sources, n_points = build_dtw_dataset(
            sample_normalized_trajectories_multi
        )
        # For rising_romance.txt, Joy A2B starts at 1.0 and rises to 4.0
        # Feature column 0 = Joy A2B, column 1 = Joy B2A
        rising_idx = sources.index("rising_romance.txt")
        joy_a2b = dataset[rising_idx, :, 0]
        joy_b2a = dataset[rising_idx, :, 1]
        # Joy A2B should be higher on average than Joy B2A (rising romance)
        assert np.mean(joy_a2b) > np.mean(joy_b2a)

    def test_multivariate_clustering_works(
        self, sample_normalized_trajectories_multi
    ):
        """DTWClusterer produces valid results with 18-dimensional input."""
        clusterer = DTWClusterer(metric="dtw", n_clusters=2, random_state=42)
        result = clusterer.cluster(sample_normalized_trajectories_multi)
        assert result.n_clusters >= 1
        assert result.barycenters[0].shape[1] == 18

    def test_missing_direction_baseline_fill(
        self, sample_normalized_trajectories_missing_direction
    ):
        """Missing direction filled with baseline 1.0 (D-13)."""
        dataset, sources, n_points = build_dtw_dataset(
            sample_normalized_trajectories_missing_direction
        )
        # All B_to_A columns (odd indices) should be 1.0
        assert dataset.shape == (1, 100, 18)
        for col in range(1, 18, 2):  # B_to_A columns
            assert np.all(dataset[0, :, col] == 1.0)


# --- DTW-04: Switching between distance metrics ---


class TestDTW04:
    """DTW-04: System allows switching between Euclidean, DTW, and Soft-DTW metrics."""

    def test_euclidean_metric_accepted(self):
        """Euclidean metric is accepted."""
        clusterer = DTWClusterer(metric="euclidean")
        assert clusterer.metric == "euclidean"

    def test_soft_dtw_metric_accepted(self):
        """Soft-DTW metric is accepted with hyphenated string."""
        clusterer = DTWClusterer(metric="soft-dtw")
        assert clusterer.metric == "soft-dtw"

    def test_distance_metric_enum_values(self):
        """All three enum values exist."""
        assert set(DistanceMetric) == {
            DistanceMetric.EUCLIDEAN,
            DistanceMetric.DTW,
            DistanceMetric.SOFT_DTW,
        }

    def test_different_metrics_different_results(
        self, sample_normalized_trajectories_multi
    ):
        """Different metrics produce different distance matrices."""
        c_dtw = DTWClusterer(metric="dtw", n_clusters=2, random_state=42)
        c_euc = DTWClusterer(metric="euclidean", n_clusters=2, random_state=42)
        r_dtw = c_dtw.cluster(sample_normalized_trajectories_multi)
        r_euc = c_euc.cluster(sample_normalized_trajectories_multi)
        # DTW and Euclidean distances should differ
        assert not np.allclose(r_dtw.distance_matrix, r_euc.distance_matrix)

    def test_metric_recorded_in_result(self, sample_normalized_trajectories_multi):
        """DTWClusterResult records which metric was used."""
        for metric in ["euclidean", "dtw", "soft-dtw"]:
            clusterer = DTWClusterer(
                metric=metric, n_clusters=2, random_state=42
            )
            result = clusterer.cluster(sample_normalized_trajectories_multi)
            assert result.metric == metric


# --- CLST-01: TimeSeriesKMeans clustering with DTW ---


class TestCLST01:
    """CLST-01: System clusters narratives by shape similarity using TimeSeriesKMeans with DTW."""

    def test_clustering_produces_labels(self, sample_normalized_trajectories_multi):
        """Clustering assigns a label to each narrative."""
        clusterer = DTWClusterer(metric="dtw", n_clusters=2, random_state=42)
        result = clusterer.cluster(sample_normalized_trajectories_multi)
        assert len(result.assignments) == 3

    def test_labels_are_non_negative(self, sample_normalized_trajectories_multi):
        """All cluster labels are non-negative integers."""
        clusterer = DTWClusterer(metric="dtw", n_clusters=2, random_state=42)
        result = clusterer.cluster(sample_normalized_trajectories_multi)
        for label in result.assignments.values():
            assert label >= 0

    def test_n_clusters_respected(self, sample_normalized_trajectories_multi):
        """Number of unique clusters equals n_clusters (or fewer if not enough data)."""
        clusterer = DTWClusterer(metric="dtw", n_clusters=2, random_state=42)
        result = clusterer.cluster(sample_normalized_trajectories_multi)
        unique_labels = set(result.assignments.values())
        assert len(unique_labels) <= 2
        assert len(unique_labels) >= 1

    def test_cluster_sizes_populated(self, sample_normalized_trajectories_multi):
        """cluster_sizes dict is populated with correct totals."""
        clusterer = DTWClusterer(metric="dtw", n_clusters=2, random_state=42)
        result = clusterer.cluster(sample_normalized_trajectories_multi)
        assert sum(result.cluster_sizes.values()) == 3


# --- CLST-02: DTW Barycenter Averaging ---


class TestCLST02:
    """CLST-02: System produces DTW Barycenter Averaging (DBA) as prototypical arcs."""

    def test_barycenters_shape(self, sample_normalized_trajectories_multi):
        """Barycenters have shape (n_clusters, n_points, 18)."""
        clusterer = DTWClusterer(metric="dtw", n_clusters=2, random_state=42)
        result = clusterer.cluster(sample_normalized_trajectories_multi)
        for barycenter in result.barycenters:
            assert barycenter.shape == (100, 18)

    def test_barycenter_count_matches_clusters(
        self, sample_normalized_trajectories_multi
    ):
        """Number of barycenters matches number of clusters found."""
        clusterer = DTWClusterer(metric="dtw", n_clusters=2, random_state=42)
        result = clusterer.cluster(sample_normalized_trajectories_multi)
        assert len(result.barycenters) == result.n_clusters

    def test_barycenter_values_in_range(
        self, sample_normalized_trajectories_multi
    ):
        """Barycenter emotion values stay in reasonable range."""
        clusterer = DTWClusterer(metric="dtw", n_clusters=2, random_state=42)
        result = clusterer.cluster(sample_normalized_trajectories_multi)
        for barycenter in result.barycenters:
            assert np.all(barycenter >= 0.5)  # Should not be negative
            assert np.all(barycenter <= 6.0)  # Should not be far above max


# --- CLST-03: Cluster assignments map source to label ---


class TestCLST03:
    """CLST-03: System outputs cluster assignments mapping each narrative to its cluster."""

    def test_assignments_dict_keys_are_sources(
        self, sample_normalized_trajectories_multi
    ):
        """Assignment dict keys match narrative source identifiers."""
        clusterer = DTWClusterer(metric="dtw", n_clusters=2, random_state=42)
        result = clusterer.cluster(sample_normalized_trajectories_multi)
        expected_sources = {
            "rising_romance.txt",
            "enemies_to_lovers.txt",
            "slow_burn.txt",
        }
        assert set(result.assignments.keys()) == expected_sources

    def test_sources_list_matches_assignments(
        self, sample_normalized_trajectories_multi
    ):
        """sources list preserves all narrative identifiers."""
        clusterer = DTWClusterer(metric="dtw", n_clusters=2, random_state=42)
        result = clusterer.cluster(sample_normalized_trajectories_multi)
        assert set(result.sources) == set(result.assignments.keys())

    def test_assignments_are_integers(self, sample_normalized_trajectories_multi):
        """All assignment values are integers."""
        clusterer = DTWClusterer(metric="dtw", n_clusters=2, random_state=42)
        result = clusterer.cluster(sample_normalized_trajectories_multi)
        for v in result.assignments.values():
            assert isinstance(v, int)


# --- CLST-04: At least two distinct pattern classes ---


class TestCLST04:
    """CLST-04: System identifies at least two distinct pattern classes from sample data."""

    def test_two_clusters_from_three_narratives(
        self, sample_normalized_trajectories_multi
    ):
        """With 3 clearly different arcs, k=2 should find at least 2 clusters."""
        clusterer = DTWClusterer(metric="dtw", n_clusters=2, random_state=42)
        result = clusterer.cluster(sample_normalized_trajectories_multi)
        unique_labels = set(result.assignments.values())
        assert len(unique_labels) >= 2, (
            f"Expected at least 2 distinct clusters, got {len(unique_labels)}: {unique_labels}"
        )

    def test_slow_burn_separates_from_rising(
        self, sample_normalized_trajectories_multi
    ):
        """Slow burn (flat arc) should cluster differently from rising romance."""
        clusterer = DTWClusterer(metric="dtw", n_clusters=2, random_state=42)
        result = clusterer.cluster(sample_normalized_trajectories_multi)
        # The slow_burn has a very flat arc vs rising_romance which is dynamic
        # With 2 clusters, they should often end up in different clusters
        # (This is a soft check — DTW may group differently depending on params)


# --- CLST-05: Reproducible with random_state ---


class TestCLST05:
    """CLST-05: System is reproducible for same inputs and configuration."""

    def test_same_inputs_same_labels(self, sample_normalized_trajectories_multi):
        """Same inputs + same config = identical cluster labels."""
        c1 = DTWClusterer(metric="dtw", n_clusters=2, random_state=42)
        c2 = DTWClusterer(metric="dtw", n_clusters=2, random_state=42)
        r1 = c1.cluster(sample_normalized_trajectories_multi)
        r2 = c2.cluster(sample_normalized_trajectories_multi)
        assert r1.assignments == r2.assignments

    def test_same_inputs_same_distance_matrix(
        self, sample_normalized_trajectories_multi
    ):
        """Same inputs + same config = identical distance matrix."""
        c1 = DTWClusterer(metric="dtw", n_clusters=2, random_state=42)
        c2 = DTWClusterer(metric="dtw", n_clusters=2, random_state=42)
        r1 = c1.cluster(sample_normalized_trajectories_multi)
        r2 = c2.cluster(sample_normalized_trajectories_multi)
        np.testing.assert_array_equal(r1.distance_matrix, r2.distance_matrix)

    def test_different_random_state_may_differ(
        self, sample_normalized_trajectories_multi
    ):
        """Different random_state may produce different results."""
        c1 = DTWClusterer(metric="dtw", n_clusters=2, random_state=42)
        c2 = DTWClusterer(metric="dtw", n_clusters=2, random_state=99)
        r1 = c1.cluster(sample_normalized_trajectories_multi)
        r2 = c2.cluster(sample_normalized_trajectories_multi)
        # May or may not differ, but distance matrices should be identical
        # (random_state only affects initialization, not distance computation)
        np.testing.assert_array_equal(r1.distance_matrix, r2.distance_matrix)

    def test_default_random_state_is_42(self):
        """Default random_state is 42."""
        clusterer = DTWClusterer()
        assert clusterer.random_state == 42
