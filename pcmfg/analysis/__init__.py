"""Clustering analysis module for PCMFG.

This module provides tools for extracting features from emotional time-series
data and clustering them to discover narrative patterns and emotional archetypes.

Main components:
- FeatureExtractor: Transform emotional time-series into feature vectors
- SceneClusterer: Cluster scenes/chunks by emotional profile
- TrajectoryClusterer: Cluster emotional trajectory patterns
- Visualization: Cluster plotting functions
"""

from pcmfg.analysis.clusterer import ClusterResult, SceneClusterer, TrajectoryClusterer
from pcmfg.analysis.feature_extractor import (
    ExtractedFeatures,
    FeatureExtractor,
    FeatureType,
    SceneFeatures,
)
from pcmfg.analysis.plotter import (
    plot_cluster_comparison,
    plot_cluster_emotions_radar,
    plot_clusters_2d,
    plot_clusters_timeline,
    save_cluster_plots,
)

__all__ = [
    # Feature extraction
    "FeatureExtractor",
    "FeatureType",
    "SceneFeatures",
    "ExtractedFeatures",
    # Clustering
    "SceneClusterer",
    "TrajectoryClusterer",
    "ClusterResult",
    # Visualization
    "plot_clusters_2d",
    "plot_clusters_timeline",
    "plot_cluster_comparison",
    "plot_cluster_emotions_radar",
    "save_cluster_plots",
]
