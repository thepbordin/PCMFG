"""Cluster visualization module for PCMFG.

Provides plotting functions for clustering results:
- 2D scatter plot with PCA dimensionality reduction
- Time-series view colored by cluster
- Cluster centroid comparison
"""

import logging
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from numpy.typing import NDArray

from pcmfg.analysis.clusterer import ClusterResult
from pcmfg.analysis.feature_extractor import ExtractedFeatures

logger = logging.getLogger(__name__)

# Import sklearn components
try:
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    from sklearn.preprocessing import StandardScaler

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("scikit-learn not installed. Visualization features disabled.")


# Color palette for clusters
CLUSTER_COLORS = [
    "#3498db",  # Blue
    "#e74c3c",  # Red
    "#2ecc71",  # Green
    "#9b59b6",  # Purple
    "#f39c12",  # Orange
    "#1abc9c",  # Teal
    "#e91e63",  # Pink
    "#00bcd4",  # Cyan
    "#795548",  # Brown
    "#607d8b",  # Blue Gray
]

NOISE_COLOR = "#95a5a6"  # Gray for noise/outliers


def plot_clusters_2d(
    features: ExtractedFeatures,
    cluster_result: ClusterResult,
    method: Literal["pca", "tsne"] = "pca",
    title: str = "Scene Clustering",
    show_labels: bool = False,
    figsize: tuple[float, float] = (10, 8),
) -> Figure:
    """Create a 2D scatter plot of clustered scenes.

    Uses PCA or t-SNE to reduce feature dimensions to 2D for visualization.

    Args:
        features: Extracted features.
        cluster_result: Clustering result with labels.
        method: Dimensionality reduction method ('pca' or 'tsne').
        title: Plot title.
        show_labels: Show chunk IDs as labels.
        figsize: Figure size.

    Returns:
        Matplotlib Figure object.
    """
    if not SKLEARN_AVAILABLE:
        raise ImportError("scikit-learn required for visualization")

    X = features.to_numpy()
    labels = np.array(cluster_result.labels)

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Reduce to 2D
    if method == "pca":
        reducer = PCA(n_components=2, random_state=42)
        X_2d = reducer.fit_transform(X_scaled)
        explained_var = reducer.explained_variance_ratio_
        xlabel = f"PC1 ({explained_var[0]*100:.1f}% variance)"
        ylabel = f"PC2 ({explained_var[1]*100:.1f}% variance)"
    else:  # tsne
        reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, len(X) - 1))
        X_2d = reducer.fit_transform(X_scaled)
        xlabel = "t-SNE 1"
        ylabel = "t-SNE 2"

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Get unique clusters
    unique_labels = sorted(set(labels))

    # Plot each cluster
    for i, label in enumerate(unique_labels):
        mask = labels == label
        color = NOISE_COLOR if label < 0 else CLUSTER_COLORS[label % len(CLUSTER_COLORS)]

        cluster_name = f"Cluster {label}" if label >= 0 else "Noise"
        ax.scatter(
            X_2d[mask, 0],
            X_2d[mask, 1],
            c=color,
            label=f"{cluster_name} ({np.sum(mask)})",
            alpha=0.7,
            s=100,
            edgecolors="white",
            linewidths=0.5,
        )

        # Show chunk IDs as labels
        if show_labels:
            for j, (x, y) in enumerate(X_2d[mask]):
                chunk_idx = np.where(mask)[0][j]
                ax.annotate(
                    str(chunk_idx),
                    (x, y),
                    fontsize=8,
                    ha="center",
                    va="center",
                    color="white",
                    fontweight="bold",
                )

    # Plot centroids if available
    if cluster_result.cluster_centers is not None:
        centers = np.array(cluster_result.cluster_centers)
        centers_scaled = scaler.transform(centers)
        if method == "pca":
            centers_2d = reducer.transform(centers_scaled)
        else:
            # t-SNE doesn't have transform, skip centroids
            centers_2d = None

        if centers_2d is not None:
            for i, (cx, cy) in enumerate(centers_2d):
                ax.scatter(
                    cx,
                    cy,
                    c=CLUSTER_COLORS[i % len(CLUSTER_COLORS)],
                    marker="X",
                    s=300,
                    edgecolors="black",
                    linewidths=2,
                    zorder=10,
                )

    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(loc="best", framealpha=0.9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_clusters_timeline(
    features: ExtractedFeatures,
    cluster_result: ClusterResult,
    title: str = "Cluster Timeline",
    figsize: tuple[float, float] = (14, 5),
) -> Figure:
    """Create a timeline view showing cluster assignments over narrative position.

    Args:
        features: Extracted features with position information.
        cluster_result: Clustering result with labels.
        title: Plot title.
        figsize: Figure size.

    Returns:
        Matplotlib Figure object.
    """
    labels = np.array(cluster_result.labels)
    positions = [sf.position for sf in features.features]

    fig, ax = plt.subplots(figsize=figsize)

    # Create colored segments
    unique_labels = sorted(set(labels))

    for label in unique_labels:
        mask = labels == label
        color = NOISE_COLOR if label < 0 else CLUSTER_COLORS[label % len(CLUSTER_COLORS)]
        cluster_name = f"Cluster {label}" if label >= 0 else "Noise"

        ax.scatter(
            [positions[i] for i in range(len(mask)) if mask[i]],
            [labels[i] for i in range(len(mask)) if mask[i]],
            c=color,
            label=cluster_name,
            s=100,
            alpha=0.8,
            edgecolors="white",
            linewidths=0.5,
        )

    # Connect points with lines
    sorted_indices = np.argsort(positions)
    sorted_positions = [positions[i] for i in sorted_indices]
    sorted_labels = [labels[i] for i in sorted_indices]

    ax.plot(sorted_positions, sorted_labels, "k-", alpha=0.3, linewidth=1)

    ax.set_xlabel("Narrative Position", fontsize=12)
    ax.set_ylabel("Cluster", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_ylim(-1.5, max(unique_labels) + 0.5)
    ax.set_xlim(-0.02, 1.02)
    ax.set_yticks(sorted(unique_labels))
    ax.set_yticklabels([f"Cluster {l}" if l >= 0 else "Noise" for l in unique_labels])
    ax.grid(True, alpha=0.3, axis="x")

    plt.tight_layout()
    return fig


def plot_cluster_emotions_radar(
    features: ExtractedFeatures,
    cluster_result: ClusterResult,
    emotions: list[str] | None = None,
    title: str = "Cluster Emotion Profiles",
    figsize: tuple[float, float] = (12, 8),
) -> Figure:
    """Create radar charts showing emotion profiles for each cluster.

    Args:
        features: Extracted features.
        cluster_result: Clustering result.
        emotions: List of emotions to show (default: all base emotions).
        title: Plot title.
        figsize: Figure size.

    Returns:
        Matplotlib Figure object.
    """
    from matplotlib.patches import Patch

    if emotions is None:
        emotions = ["Joy", "Trust", "Fear", "Anger", "Sadness", "Arousal"]

    labels = np.array(cluster_result.labels)
    unique_labels = sorted(set(l for l in unique_labels if l >= 0)) if (unique_labels := set(labels)) else []

    n_clusters = len(unique_labels)
    if n_clusters == 0:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "No clusters to display", ha="center", va="center")
        return fig

    # Calculate subplot grid
    n_cols = min(3, n_clusters)
    n_rows = (n_clusters + n_cols - 1) // n_cols

    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=figsize, subplot_kw=dict(projection="polar")
    )
    if n_clusters == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if hasattr(axes, "flatten") else [axes]

    # Compute average emotions per cluster
    for idx, label in enumerate(unique_labels):
        ax = axes[idx]
        mask = labels == label

        # Get feature values for this cluster
        emotion_values = {e: [] for e in emotions}
        for i, sf in enumerate(features.features):
            if mask[i]:
                for j, fname in enumerate(sf.feature_names):
                    if fname.startswith("A_to_B_"):
                        emotion = fname.replace("A_to_B_", "")
                        if emotion in emotions:
                            emotion_values[emotion].append(sf.feature_vector[j])

        # Compute averages
        avg_values = [np.mean(emotion_values[e]) if emotion_values[e] else 1.0 for e in emotions]

        # Setup radar chart
        angles = np.linspace(0, 2 * np.pi, len(emotions), endpoint=False).tolist()
        values = avg_values + [avg_values[0]]  # Close the loop
        angles += angles[:1]

        ax.plot(angles, values, "o-", linewidth=2, color=CLUSTER_COLORS[label % len(CLUSTER_COLORS)])
        ax.fill(angles, values, alpha=0.25, color=CLUSTER_COLORS[label % len(CLUSTER_COLORS)])

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(emotions, fontsize=9)
        ax.set_ylim(0, 5)
        ax.set_yticks([1, 2, 3, 4, 5])
        ax.set_yticklabels(["1", "2", "3", "4", "5"], fontsize=8)
        ax.set_title(f"Cluster {label}\n({np.sum(mask)} scenes)", fontsize=11, fontweight="bold")

    # Hide empty subplots
    for idx in range(n_clusters, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    return fig


def plot_cluster_comparison(
    features: ExtractedFeatures,
    cluster_result: ClusterResult,
    title: str = "Cluster Comparison",
    figsize: tuple[float, float] = (14, 6),
) -> Figure:
    """Create a bar chart comparing average emotions across clusters.

    Args:
        features: Extracted features.
        cluster_result: Clustering result.
        title: Plot title.
        figsize: Figure size.

    Returns:
        Matplotlib Figure object.
    """
    labels = np.array(cluster_result.labels)
    unique_labels = sorted(set(l for l in set(labels) if l >= 0))

    if not unique_labels:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "No clusters to display", ha="center", va="center")
        return fig

    # Emotions to compare
    emotions = ["Joy", "Trust", "Fear", "Anger", "Sadness", "Arousal"]

    # Compute averages per cluster
    cluster_averages: dict[int, dict[str, float]] = {}
    for label in unique_labels:
        mask = labels == label
        cluster_averages[label] = {}

        for emotion in emotions:
            values = []
            for i, sf in enumerate(features.features):
                if mask[i]:
                    for j, fname in enumerate(sf.feature_names):
                        if fname == f"A_to_B_{emotion}":
                            values.append(sf.feature_vector[j])
            cluster_averages[label][emotion] = np.mean(values) if values else 1.0

    # Create grouped bar chart
    fig, ax = plt.subplots(figsize=figsize)

    x = np.arange(len(emotions))
    width = 0.8 / len(unique_labels)

    for i, label in enumerate(unique_labels):
        values = [cluster_averages[label][e] for e in emotions]
        bars = ax.bar(
            x + i * width,
            values,
            width,
            label=f"Cluster {label}",
            color=CLUSTER_COLORS[label % len(CLUSTER_COLORS)],
            alpha=0.8,
        )

        # Add value labels on bars
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.1,
                f"{val:.1f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    ax.set_xlabel("Emotion", fontsize=12)
    ax.set_ylabel("Average Score", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xticks(x + width * (len(unique_labels) - 1) / 2)
    ax.set_xticklabels(emotions)
    ax.set_ylim(0, 5.5)
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    return fig


def save_cluster_plots(
    features: ExtractedFeatures,
    cluster_result: ClusterResult,
    output_dir: str | Path,
    prefix: str = "cluster",
    formats: list[str] | None = None,
) -> list[Path]:
    """Generate and save all cluster visualization plots.

    Args:
        features: Extracted features.
        cluster_result: Clustering result.
        output_dir: Directory to save plots.
        prefix: Filename prefix.
        formats: Output formats (default: ['png']).

    Returns:
        List of saved file paths.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if formats is None:
        formats = ["png"]

    saved_files = []

    # 2D scatter plot
    fig = plot_clusters_2d(features, cluster_result, method="pca")
    for fmt in formats:
        path = output_dir / f"{prefix}_scatter_2d.{fmt}"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        saved_files.append(path)
    plt.close(fig)

    # Timeline plot
    fig = plot_clusters_timeline(features, cluster_result)
    for fmt in formats:
        path = output_dir / f"{prefix}_timeline.{fmt}"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        saved_files.append(path)
    plt.close(fig)

    # Emotion comparison
    fig = plot_cluster_comparison(features, cluster_result)
    for fmt in formats:
        path = output_dir / f"{prefix}_comparison.{fmt}"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        saved_files.append(path)
    plt.close(fig)

    # Radar charts
    fig = plot_cluster_emotions_radar(features, cluster_result)
    for fmt in formats:
        path = output_dir / f"{prefix}_radar.{fmt}"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        saved_files.append(path)
    plt.close(fig)

    logger.info(f"Saved {len(saved_files)} cluster plots to {output_dir}")
    return saved_files
