"""DTW-based clustering for emotional trajectory shape similarity.

Clusters narratives by emotional arc shape using Dynamic Time Warping (DTW)
distance via tslearn. Supports multivariate trajectories (9 emotions x 2
directions = 18 dimensions) with configurable warping constraints.

This module is purely additive — existing SceneClusterer, TrajectoryClusterer,
and Phase 1-3 pipeline remain unchanged.
"""

import logging
from collections import defaultdict
from enum import Enum

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field

from pcmfg.models.schemas import BASE_EMOTIONS, NormalizedTrajectory

logger = logging.getLogger(__name__)

# Optional dependency guard (follows clusterer.py:28-35 pattern)
try:
    from tslearn.clustering import TimeSeriesKMeans
    from tslearn.metrics import cdist_dtw

    TSLEARN_AVAILABLE = True
except ImportError:
    TSLEARN_AVAILABLE = False
    logger.warning(
        "tslearn not installed. DTW clustering features disabled. "
        "Install with: pip install tslearn"
    )


class DistanceMetric(str, Enum):
    """Available distance metrics for DTW clustering."""

    EUCLIDEAN = "euclidean"
    DTW = "dtw"
    SOFT_DTW = "soft-dtw"


class DTWClusterResult(BaseModel):
    """Result of DTW-based clustering analysis.

    Contains cluster assignments, barycenters, distance matrix, and
    quality metrics for shape-based narrative clustering.
    """

    assignments: dict[str, int] = Field(
        description="Map of source narrative identifier to cluster label"
    )
    barycenters: list[NDArray[np.float64]] = Field(
        description="DTW Barycenter Averaging (DBA) prototypes per cluster, "
        "each shape (n_points, 18)"
    )
    distance_matrix: NDArray[np.float64] = Field(
        description="Pairwise distance matrix, shape (n_narratives, n_narratives)"
    )
    n_clusters: int = Field(description="Number of clusters")
    metric: str = Field(
        description="Distance metric used (euclidean, dtw, or soft-dtw)"
    )
    sakoe_chiba_radius: int | None = Field(
        description="Sakoe-Chiba warping radius used (None if unconstrained)"
    )
    cluster_sizes: dict[str, int] = Field(
        default_factory=dict,
        description="Number of narratives per cluster"
    )
    silhouette_score: float | None = Field(
        default=None,
        description="Silhouette score (-1 to 1, higher is better)"
    )
    sources: list[str] = Field(
        description="Ordered list of source narrative identifiers"
    )

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)


def build_dtw_dataset(
    trajectories: list[NormalizedTrajectory],
) -> tuple[NDArray[np.float64], list[str], int]:
    """Convert flat NormalizedTrajectory list to tslearn (n, sz, 18) format.

    Groups trajectories by source narrative, then stacks the 18 emotion
    arrays (9 emotions x 2 directions) into a single matrix per narrative.

    Feature axis ordering (D-02): For each of the 9 BASE_EMOTIONS,
    first A_to_B then B_to_A:
    [Joy_A2B, Joy_B2A, Trust_A2B, Trust_B2A, Fear_A2B, Fear_B2A, ...]

    Args:
        trajectories: Flat list of NormalizedTrajectory objects from
            NarrativeNormalizer.normalize_all(). One per (direction, emotion)
            pair per narrative.

    Returns:
        Tuple of (dataset, sources, n_points) where:
        - dataset: numpy array of shape (n_narratives, n_points, 18)
        - sources: ordered list of narrative identifiers
        - n_points: number of resampling points per trajectory

    Raises:
        ValueError: If trajectories list is empty.
    """
    if not trajectories:
        raise ValueError("Cannot build DTW dataset from empty trajectory list")

    # Group by source narrative
    grouped: dict[str, dict[tuple[str, str], NDArray[np.float64]]] = (
        defaultdict(dict)
    )

    for traj in trajectories:
        key = (traj.direction, traj.emotion)
        grouped[traj.source][key] = np.array(traj.y, dtype=np.float64)

    # Determine n_points from first trajectory's y values
    first_values = next(iter(next(iter(grouped.values())).values()))
    n_points = len(first_values)
    n_narratives = len(grouped)

    # Build dataset matrix
    dataset = np.zeros((n_narratives, n_points, 18), dtype=np.float64)
    sources: list[str] = []

    for i, (source, traj_map) in enumerate(sorted(grouped.items())):
        sources.append(source)
        for emotion_idx, emotion in enumerate(BASE_EMOTIONS):
            for dir_offset, direction in enumerate(["A_to_B", "B_to_A"]):
                feature_col = emotion_idx * 2 + dir_offset
                key = (direction, emotion)
                if key in traj_map:
                    dataset[i, :, feature_col] = traj_map[key]
                else:
                    # D-13: Missing direction filled with baseline (all 1s)
                    dataset[i, :, feature_col] = 1.0
                    logger.warning(
                        "Missing %s %s for source '%s', filled with baseline",
                        direction,
                        emotion,
                        source,
                    )

    return dataset, sources, n_points


class DTWClusterer:
    """Cluster narratives by emotional trajectory shape using DTW distance.

    Uses tslearn's TimeSeriesKMeans with DTW metric for shape-based
    clustering. Operates on NormalizedTrajectory output from
    NarrativeNormalizer — does NOT modify existing pipeline.

    Example:
        >>> clusterer = DTWClusterer(n_clusters=3, metric="dtw")
        >>> result = clusterer.cluster(normalized_trajectories)
        >>> print(result.assignments)
    """

    def __init__(
        self,
        metric: str | DistanceMetric = DistanceMetric.DTW,
        n_clusters: int = 3,
        sakoe_chiba_radius: float | None = 0.2,
        random_state: int = 42,
        max_iter: int = 50,
        n_init: int = 1,
        max_iter_barycenter: int = 100,
        soft_dtw_gamma: float = 1.0,
    ) -> None:
        """Initialize the DTW clusterer.

        Args:
            metric: Distance metric — "euclidean", "dtw", or "soft-dtw".
            n_clusters: Number of clusters to find.
            sakoe_chiba_radius: Sakoe-Chiba constraint as fraction of series
                length (e.g., 0.2 = 20%). None for unconstrained DTW.
            random_state: Random seed for reproducibility.
            max_iter: Max iterations for k-means.
            n_init: Number of random initializations.
            max_iter_barycenter: Max iterations for DBA barycenter update.
            soft_dtw_gamma: Smoothing parameter for soft-DTW metric.
        """
        if not TSLEARN_AVAILABLE:
            raise ImportError(
                "tslearn is required for DTW clustering. "
                "Install with: pip install tslearn"
            )
        self.metric = metric.value if isinstance(metric, DistanceMetric) else metric
        self.n_clusters = n_clusters
        self.sakoe_chiba_radius = sakoe_chiba_radius
        self.random_state = random_state
        self.max_iter = max_iter
        self.n_init = n_init
        self.max_iter_barycenter = max_iter_barycenter
        self.soft_dtw_gamma = soft_dtw_gamma

    def cluster(
        self, trajectories: list[NormalizedTrajectory]
    ) -> DTWClusterResult:
        """Cluster narratives by emotional arc shape using DTW distance.

        Args:
            trajectories: Flat list of NormalizedTrajectory objects from
                NarrativeNormalizer.normalize_all().

        Returns:
            DTWClusterResult with cluster assignments, barycenters,
            distance matrix, and quality metrics.

        Raises:
            ValueError: If fewer narratives than n_clusters.
        """
        from scipy.spatial.distance import cdist as scipy_cdist
        from sklearn.metrics import silhouette_score as sklearn_silhouette_score

        # Build tslearn-compatible dataset
        dataset, sources, n_points = build_dtw_dataset(trajectories)
        n_narratives = len(sources)

        # Validate input count
        if n_narratives < self.n_clusters:
            raise ValueError(
                f"Need at least {self.n_clusters} narratives for "
                f"n_clusters={self.n_clusters}, got {n_narratives}"
            )

        # --- Compute distance matrix ---
        if self.metric == "dtw":
            radius: int | None = None
            if self.sakoe_chiba_radius is not None:
                radius = int(self.sakoe_chiba_radius * n_points)

            if radius is not None and radius > 0:
                distance_matrix = cdist_dtw(
                    dataset,
                    global_constraint="sakoe_chiba",
                    sakoe_chiba_radius=radius,
                )
            else:
                distance_matrix = cdist_dtw(dataset)

        elif self.metric == "euclidean":
            dataset_flat = dataset.reshape(n_narratives, -1)
            distance_matrix = scipy_cdist(dataset_flat, dataset_flat)

        elif self.metric == "soft-dtw":
            radius_soft: int | None = None
            if self.sakoe_chiba_radius is not None:
                radius_soft = int(self.sakoe_chiba_radius * n_points)

            if radius_soft is not None and radius_soft > 0:
                distance_matrix = cdist_dtw(
                    dataset,
                    global_constraint="sakoe_chiba",
                    sakoe_chiba_radius=radius_soft,
                )
            else:
                distance_matrix = cdist_dtw(dataset)

        else:
            raise ValueError(
                f"Unknown metric: {self.metric}. "
                f"Use 'euclidean', 'dtw', or 'soft-dtw'."
            )

        # --- Build metric_params for TimeSeriesKMeans ---
        tslearn_metric: str = self.metric
        metric_params: dict[str, object] | None = None

        if self.metric == "soft-dtw":
            tslearn_metric = "softdtw"  # CRITICAL: tslearn uses "softdtw"
            metric_params = {"gamma": self.soft_dtw_gamma}
        elif self.metric == "dtw":
            radius_km: int | None = None
            if self.sakoe_chiba_radius is not None:
                radius_km = int(self.sakoe_chiba_radius * n_points)
            if radius_km is not None and radius_km > 0:
                metric_params = {
                    "global_constraint": "sakoe_chiba",
                    "sakoe_chiba_radius": radius_km,
                }

        # --- Fit TimeSeriesKMeans ---
        km = TimeSeriesKMeans(
            n_clusters=self.n_clusters,
            metric=tslearn_metric,
            metric_params=metric_params,
            random_state=self.random_state,
            max_iter=self.max_iter,
            n_init=self.n_init,
            max_iter_barycenter=self.max_iter_barycenter,
        )
        km.fit(dataset)

        labels = km.labels_
        centers = km.cluster_centers_

        # --- Build cluster assignments (source → label) ---
        assignments: dict[str, int] = {
            source: int(label) for source, label in zip(sources, labels)
        }

        # --- Build cluster sizes ---
        cluster_sizes: dict[str, int] = {}
        for label in sorted(set(labels.tolist())):
            cluster_sizes[f"Cluster {label}"] = int(np.sum(labels == label))

        # --- Compute silhouette score (safe) ---
        unique_labels = set(labels.tolist())
        sil_score: float | None = None
        if len(unique_labels) >= 2 and len(labels) > len(unique_labels):
            try:
                if self.metric in ("dtw", "soft-dtw"):
                    sil_score = float(
                        sklearn_silhouette_score(
                            distance_matrix,
                            labels,
                            metric="precomputed",
                        )
                    )
                else:
                    X_flat = dataset.reshape(len(labels), -1)
                    sil_score = float(sklearn_silhouette_score(X_flat, labels))
            except Exception:
                sil_score = None

        # --- Extract barycenters ---
        barycenters: list[NDArray[np.float64]] = []
        for i in range(len(unique_labels)):
            barycenters.append(centers[i])

        # --- Determine stored radius ---
        stored_radius: int | None = None
        if self.sakoe_chiba_radius is not None:
            stored_radius = int(self.sakoe_chiba_radius * n_points)

        return DTWClusterResult(
            assignments=assignments,
            barycenters=barycenters,
            distance_matrix=distance_matrix,
            n_clusters=len(unique_labels),
            metric=self.metric,
            sakoe_chiba_radius=stored_radius,
            cluster_sizes=cluster_sizes,
            silhouette_score=sil_score,
            sources=sources,
        )
