"""Clustering module for emotional time-series analysis.

Provides clustering algorithms to discover patterns in emotional narratives:

1. SceneClusterer: Cluster individual scenes/chunks by emotional profile
2. TrajectoryClusterer: Cluster emotional trajectory patterns across novels

Supports multiple clustering algorithms:
- K-Means: Fast, interpretable, good for spherical clusters
- DBSCAN: Density-based, good for outliers and irregular shapes
- Hierarchical: Dendrogram visualization, good for nested patterns
"""

import logging
from enum import Enum
from typing import Any

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, Field

from pcmfg.analysis.feature_extractor import ExtractedFeatures, FeatureExtractor
from pcmfg.models.schemas import BASE_EMOTIONS, AnalysisResult

logger = logging.getLogger(__name__)

# Try to import sklearn, provide helpful error if not available
try:
    from sklearn.cluster import DBSCAN, KMeans, AgglomerativeClustering
    from sklearn.metrics import silhouette_score
    from sklearn.preprocessing import StandardScaler

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning(
        "scikit-learn not installed. Clustering features disabled. "
        "Install with: pip install scikit-learn"
    )


class ClusteringAlgorithm(str, Enum):
    """Available clustering algorithms."""

    KMEANS = "kmeans"
    DBSCAN = "dbscan"
    HIERARCHICAL = "hierarchical"


class ClusterResult(BaseModel):
    """Result of clustering analysis."""

    n_clusters: int = Field(description="Number of clusters found")
    labels: list[int] = Field(description="Cluster label for each sample")
    cluster_centers: list[list[float]] | None = Field(
        default=None, description="Cluster centroids (for K-Means)"
    )
    silhouette_score: float | None = Field(
        default=None, description="Silhouette score (-1 to 1, higher is better)"
    )
    cluster_sizes: dict[str, int] = Field(
        default_factory=dict, description="Number of samples per cluster"
    )
    cluster_interpretations: dict[str, str] = Field(
        default_factory=dict, description="Human-readable cluster descriptions"
    )


class SceneClusterer:
    """Cluster scenes/chunks by their emotional profile.

    Discovers emotional "scene archetypes" such as:
    - Neutral/baseline scenes
    - Conflict scenes (high anger/disgust)
    - Tender moments (high joy/trust)
    - Anxious anticipation scenes
    - Passion peaks (high arousal)

    Example:
        >>> extractor = FeatureExtractor(FeatureType.RAW)
        >>> features = extractor.extract(analysis_result)
        >>> clusterer = SceneClusterer(n_clusters=5)
        >>> result = clusterer.cluster(features)
        >>> print(result.cluster_interpretations)
    """

    def __init__(
        self,
        algorithm: ClusteringAlgorithm = ClusteringAlgorithm.KMEANS,
        n_clusters: int = 5,
        random_state: int = 42,
        dbscan_eps: float = 0.5,
        dbscan_min_samples: int = 3,
        scale_features: bool = True,
    ) -> None:
        """Initialize the scene clusterer.

        Args:
            algorithm: Clustering algorithm to use.
            n_clusters: Number of clusters (for K-Means/Hierarchical).
            random_state: Random seed for reproducibility.
            dbscan_eps: Epsilon for DBSCAN (neighborhood radius).
            dbscan_min_samples: Min samples for DBSCAN core points.
            scale_features: Whether to standardize features before clustering.
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError(
                "scikit-learn is required for clustering. "
                "Install with: pip install scikit-learn"
            )

        self.algorithm = algorithm
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.dbscan_eps = dbscan_eps
        self.dbscan_min_samples = dbscan_min_samples
        self.scale_features = scale_features
        self._scaler: StandardScaler | None = None

    def cluster(
        self,
        features: ExtractedFeatures,
        interpret: bool = True,
    ) -> ClusterResult:
        """Cluster scenes by emotional features.

        Args:
            features: Extracted features from analysis result.
            interpret: Generate human-readable cluster interpretations.

        Returns:
            ClusterResult with cluster assignments and interpretations.
        """
        X = features.to_numpy()

        # Scale features if requested
        if self.scale_features:
            self._scaler = StandardScaler()
            X = self._scaler.fit_transform(X)

        # Run clustering
        if self.algorithm == ClusteringAlgorithm.KMEANS:
            model = KMeans(
                n_clusters=self.n_clusters,
                random_state=self.random_state,
                n_init=10,
            )
            labels = model.fit_predict(X)
            centers = model.cluster_centers_
            if self._scaler is not None:
                centers = self._scaler.inverse_transform(centers)
        elif self.algorithm == ClusteringAlgorithm.DBSCAN:
            model = DBSCAN(
                eps=self.dbscan_eps,
                min_samples=self.dbscan_min_samples,
            )
            labels = model.fit_predict(X)
            centers = None
        elif self.algorithm == ClusteringAlgorithm.HIERARCHICAL:
            model = AgglomerativeClustering(n_clusters=self.n_clusters)
            labels = model.fit_predict(X)
            centers = None
        else:
            raise ValueError(f"Unknown algorithm: {self.algorithm}")

        # Compute silhouette score (only if more than 1 cluster and no noise points)
        unique_labels = set(labels) - {-1}  # Remove noise label
        sil_score: float | None = None
        if len(unique_labels) > 1 and len(labels) > len(unique_labels):
            try:
                sil_score = float(silhouette_score(X, labels))
            except Exception:
                sil_score = None

        # Compute cluster sizes
        cluster_sizes: dict[str, int] = {}
        for label in sorted(set(labels)):
            name = f"Cluster {label}" if label >= 0 else "Noise"
            cluster_sizes[name] = int(np.sum(labels == label))

        # Generate interpretations
        interpretations: dict[str, str] = {}
        if interpret and centers is not None:
            interpretations = self._interpret_clusters(
                centers, list(unique_labels), features.feature_names
            )
        elif interpret:
            # For algorithms without explicit centers, compute mean per cluster
            interpretations = self._interpret_clusters_from_data(
                X, labels, features.feature_names
            )

        return ClusterResult(
            n_clusters=len(unique_labels),
            labels=labels.tolist(),
            cluster_centers=centers.tolist() if centers is not None else None,
            silhouette_score=sil_score,
            cluster_sizes=cluster_sizes,
            cluster_interpretations=interpretations,
        )

    def _interpret_clusters(
        self,
        centers: NDArray[np.float64],
        labels: list[int],
        feature_names: list[str],
    ) -> dict[str, str]:
        """Generate human-readable interpretations for each cluster.

        Analyzes cluster centroids to identify dominant emotional patterns.
        """
        interpretations = {}

        for i, label in enumerate(labels):
            if label < 0:
                continue

            centroid = centers[i]
            description = self._describe_centroid(centroid, feature_names)
            interpretations[f"Cluster {label}"] = description

        return interpretations

    def _interpret_clusters_from_data(
        self,
        X: NDArray[np.float64],
        labels: NDArray[np.int64],
        feature_names: list[str],
    ) -> dict[str, str]:
        """Generate interpretations by computing cluster means from data."""
        interpretations = {}
        unique_labels = sorted(set(labels) - {-1})

        for label in unique_labels:
            mask = labels == label
            centroid = X[mask].mean(axis=0)

            # Inverse transform if scaled
            if self._scaler is not None:
                centroid = self._scaler.inverse_transform([centroid])[0]

            description = self._describe_centroid(centroid, feature_names)
            interpretations[f"Cluster {label}"] = description

        if -1 in labels:
            noise_count = int(np.sum(labels == -1))
            interpretations["Noise"] = f"Outlier scenes ({noise_count} samples)"

        return interpretations

    def _describe_centroid(
        self, centroid: NDArray[np.float64], feature_names: list[str]
    ) -> str:
        """Generate a description of a cluster centroid."""
        # Build feature -> value mapping
        feature_values = {
            name: val for name, val in zip(feature_names, centroid)
        }

        # Analyze A→B and B→A separately
        descriptions = []

        for direction in ["A_to_B", "B_to_A"]:
            direction_scores = {}
            for emotion in BASE_EMOTIONS:
                key = f"{direction}_{emotion}"
                if key in feature_values:
                    direction_scores[emotion] = feature_values[key]

            if direction_scores:
                # Find dominant emotions (score > 2)
                high_emotions = [
                    (e, s) for e, s in direction_scores.items() if s > 2.0
                ]
                high_emotions.sort(key=lambda x: x[1], reverse=True)

                if high_emotions:
                    top_emotions = [f"{e}({s:.1f})" for e, s in high_emotions[:3]]
                    descriptions.append(f"{direction}: {', '.join(top_emotions)}")
                else:
                    descriptions.append(f"{direction}: baseline/neutral")

        return " | ".join(descriptions) if descriptions else "Unknown pattern"

    def find_optimal_k(
        self,
        features: ExtractedFeatures,
        k_range: tuple[int, int] = (2, 10),
    ) -> dict[int, float]:
        """Find optimal number of clusters using silhouette analysis.

        Args:
            features: Extracted features.
            k_range: Range of k values to test (inclusive).

        Returns:
            Dictionary mapping k to silhouette score.
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn required")

        X = features.to_numpy()
        if self.scale_features:
            scaler = StandardScaler()
            X = scaler.fit_transform(X)

        scores = {}
        for k in range(k_range[0], k_range[1] + 1):
            if k >= len(X):
                continue

            model = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
            labels = model.fit_predict(X)

            try:
                score = silhouette_score(X, labels)
                scores[k] = score
            except Exception:
                continue

        return scores


class TrajectoryClusterer:
    """Cluster emotional trajectories across multiple novels.

    Compares the overall emotional arc of different narratives to find
    similar relationship patterns and narrative structures.

    Example:
        >>> # Compare trajectories across multiple novels
        >>> results = [analyzer.analyze(novel) for novel in novels]
        >>> clusterer = TrajectoryClusterer()
        >>> cluster_result = clusterer.cluster_multiple(results)
    """

    def __init__(
        self,
        algorithm: ClusteringAlgorithm = ClusteringAlgorithm.HIERARCHICAL,
        n_clusters: int = 3,
        random_state: int = 42,
        scale_features: bool = True,
    ) -> None:
        """Initialize the trajectory clusterer.

        Args:
            algorithm: Clustering algorithm (Hierarchical recommended for interpretability).
            n_clusters: Number of clusters.
            random_state: Random seed.
            scale_features: Whether to standardize features.
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError(
                "scikit-learn is required for clustering. "
                "Install with: pip install scikit-learn"
            )

        self.algorithm = algorithm
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.scale_features = scale_features
        self._scaler: StandardScaler | None = None

    def cluster_multiple(
        self,
        results: list[AnalysisResult],
    ) -> ClusterResult:
        """Cluster multiple novels by their emotional trajectory.

        Args:
            results: List of analysis results from different novels.

        Returns:
            ClusterResult with cluster assignments for each novel.
        """
        # Extract statistical features from each novel
        extractor = FeatureExtractor()
        features_list = []
        sources = []

        for result in results:
            features = extractor._extract_statistical_features(result)
            features_list.append(features[0])  # Single row per novel
            sources.append(result.metadata.source)

        X = np.array(features_list)

        # Scale if requested
        if self.scale_features:
            self._scaler = StandardScaler()
            X = self._scaler.fit_transform(X)

        # Cluster
        if self.algorithm == ClusteringAlgorithm.KMEANS:
            model = KMeans(
                n_clusters=min(self.n_clusters, len(results)),
                random_state=self.random_state,
                n_init=10,
            )
            labels = model.fit_predict(X)
            centers = model.cluster_centers_
            if self._scaler is not None:
                centers = self._scaler.inverse_transform(centers)
        elif self.algorithm == ClusteringAlgorithm.HIERARCHICAL:
            model = AgglomerativeClustering(
                n_clusters=min(self.n_clusters, len(results))
            )
            labels = model.fit_predict(X)
            centers = None
        else:
            model = DBSCAN(eps=0.5, min_samples=2)
            labels = model.fit_predict(X)
            centers = None

        # Compute silhouette if possible
        unique_labels = set(labels) - {-1}
        sil_score: float | None = None
        if len(unique_labels) > 1 and len(labels) > len(unique_labels):
            try:
                sil_score = float(silhouette_score(X, labels))
            except Exception:
                sil_score = None

        # Build cluster sizes
        cluster_sizes: dict[str, int] = {}
        for label in sorted(set(labels)):
            name = f"Cluster {label}" if label >= 0 else "Noise"
            cluster_sizes[name] = int(np.sum(labels == label))

        return ClusterResult(
            n_clusters=len(unique_labels),
            labels=labels.tolist(),
            cluster_centers=centers.tolist() if centers is not None else None,
            silhouette_score=sil_score,
            cluster_sizes=cluster_sizes,
            cluster_interpretations={},
        )

    def get_cluster_members(
        self,
        sources: list[str],
        labels: list[int],
    ) -> dict[int, list[str]]:
        """Get the novels belonging to each cluster.

        Args:
            sources: List of source identifiers.
            labels: Cluster labels from clustering.

        Returns:
            Dictionary mapping cluster ID to list of source names.
        """
        members: dict[int, list[str]] = {}
        for source, label in zip(sources, labels):
            if label not in members:
                members[label] = []
            members[label].append(source)
        return members
