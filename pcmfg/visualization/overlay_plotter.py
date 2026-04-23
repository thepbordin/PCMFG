"""Overlay visualization plotter for normalized emotional trajectories.

Creates matplotlib visualizations that overlay multiple normalized emotion
trajectories on the same axes for cross-narrative comparison. Supports
optional cluster coloring from DTWClusterResult with barycenter overlays.

Follows the EmotionPlotter pattern from visualization/plotter.py:
- Class-based with dpi/figsize in __init__
- Save-to-file via fig.savefig() + plt.close(fig)
- 9x2 grid layout (emotions x directions)
- Y-axis [0.5, 5.5] with integer ticks, X-axis [0, 1]
"""

import logging
from collections import defaultdict
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from pcmfg.analysis.dtw_clusterer import DTWClusterResult
from pcmfg.analysis.plotter import CLUSTER_COLORS
from pcmfg.models.schemas import BASE_EMOTIONS, NormalizedTrajectory

logger = logging.getLogger(__name__)

# Direction labels for subplots
DIRECTIONS = ["A_to_B", "B_to_A"]
DIRECTION_LABELS = {"A_to_B": "A → B", "B_to_A": "B → A"}


class NarrativeOverlayPlotter:
    """Creates overlay plots comparing emotional trajectories across narratives.

    Generates matplotlib figures showing multiple normalized emotion
    trajectories on the same axes, with optional cluster coloring and
    barycenter overlays from DTW clustering results.

    Methods save PNG files to disk (not return Figure objects), matching
    the EmotionPlotter convention.
    """

    def __init__(self, dpi: int = 300, figsize: tuple[float, float] = (16, 20)) -> None:
        """Initialize the plotter.

        Args:
            dpi: Image resolution (dots per inch).
            figsize: Figure size in inches (width, height).
        """
        self.dpi = dpi
        self.figsize = figsize

    def _group_trajectories(
        self, trajectories: list[NormalizedTrajectory]
    ) -> dict[str, dict[tuple[str, str], NDArray[np.float64]]]:
        """Group flat trajectory list by source narrative.

        Same grouping pattern as build_dtw_dataset() in dtw_clusterer.py.
        Returns dict mapping source -> dict[(direction, emotion)] -> y array.

        Args:
            trajectories: Flat list of NormalizedTrajectory objects.

        Returns:
            Nested dict: source -> (direction, emotion) -> numpy y-array.
        """
        grouped: dict[str, dict[tuple[str, str], NDArray[np.float64]]] = (
            defaultdict(dict)
        )

        for traj in trajectories:
            key = (traj.direction, traj.emotion)
            grouped[traj.source][key] = np.array(traj.y, dtype=np.float64)

        return dict(grouped)

    def _unpack_barycenter(
        self, barycenter: NDArray[np.float64]
    ) -> dict[tuple[str, str], NDArray[np.float64]]:
        """Unpack 18-dim barycenter into per-emotion, per-direction arrays.

        Uses the same column ordering as build_dtw_dataset():
        For each of the 9 BASE_EMOTIONS, first A_to_B then B_to_A.
        Column index = emotion_idx * 2 + dir_offset.

        Args:
            barycenter: Array of shape (n_points, 18).

        Returns:
            Dict mapping (direction, emotion) -> array of shape (n_points,).
        """
        unpacked: dict[tuple[str, str], NDArray[np.float64]] = {}

        for emotion_idx, emotion in enumerate(BASE_EMOTIONS):
            for dir_offset, direction in enumerate(DIRECTIONS):
                feature_col = emotion_idx * 2 + dir_offset
                unpacked[(direction, emotion)] = barycenter[:, feature_col]

        return unpacked

    def _plot_subplot_overlay(
        self,
        ax: plt.Axes,
        y_arrays: list[NDArray[np.float64]],
        colors: list[str],
        alpha: float,
        barycenter_y: NDArray[np.float64] | None,
        x: NDArray[np.float64],
        emotion_label: str,
    ) -> None:
        """Render multiple lines + optional barycenter on one subplot.

        Args:
            ax: Matplotlib axes to plot on.
            y_arrays: List of y-value arrays (one per narrative).
            colors: List of color strings (one per narrative).
            alpha: Alpha value for member lines.
            barycenter_y: Optional barycenter y-values for this subplot.
            x: Shared x-axis array (narrative progress).
            emotion_label: Label for the subplot title.
        """
        # Plot member narrative lines
        for y_values, color in zip(y_arrays, colors):
            ax.plot(x, y_values, color=color, linewidth=1, alpha=alpha)

        # Plot barycenter if provided (D-11, D-12)
        if barycenter_y is not None:
            markevery = max(1, len(x) // 10)
            ax.plot(
                x,
                barycenter_y,
                color="black",
                linewidth=3.5,
                marker="D",
                markersize=6,
                markevery=markevery,
                alpha=1.0,
                zorder=10,
                label="Barycenter",
            )

        # Axis formatting (matches EmotionPlotter convention)
        ax.set_ylim(0.5, 5.5)
        ax.set_yticks([1, 2, 3, 4, 5])
        ax.set_xlim(0, 1)
        ax.axhline(y=1, color="gray", linestyle="--", linewidth=1, alpha=0.7)
        ax.grid(True, alpha=0.3)
        ax.set_title(emotion_label, fontsize=10)

    def _resolve_colors_and_alpha(
        self,
        sources: list[str],
        cluster_result: DTWClusterResult | None,
    ) -> tuple[list[str], float]:
        """Determine per-narrative colors and alpha based on cluster result.

        Args:
            sources: Ordered list of source narrative identifiers.
            cluster_result: Optional DTWClusterResult for cluster coloring.

        Returns:
            Tuple of (color_list, alpha).
        """
        if cluster_result is not None:
            colors = [
                CLUSTER_COLORS[cluster_result.assignments[src] % len(CLUSTER_COLORS)]
                for src in sources
            ]
            alpha = 0.3
        else:
            cmap = plt.get_cmap("tab10")
            colors = [cmap(i % 10) for i in range(len(sources))]
            alpha = 0.7

        return colors, alpha

    def _add_cluster_legend(
        self,
        fig: plt.Figure,
        cluster_result: DTWClusterResult,
    ) -> None:
        """Add cluster legend entries to figure.

        Legends show cluster labels only (D-09), e.g., "Cluster 0 (5 narratives)".

        Args:
            fig: Matplotlib figure to add legend to.
            cluster_result: DTWClusterResult with cluster_sizes.
        """
        from matplotlib.lines import Line2D

        legend_handles = []
        for cluster_label, size in sorted(
            cluster_result.cluster_sizes.items(), key=lambda x: int(x[0])
        ):
            label_idx = int(cluster_label)
            color = CLUSTER_COLORS[label_idx % len(CLUSTER_COLORS)]
            handle = Line2D(
                [0],
                [0],
                color=color,
                linewidth=2,
                label=f"Cluster {cluster_label} ({size} narratives)",
            )
            legend_handles.append(handle)

        # Add barycenter legend entry
        bary_handle = Line2D(
            [0],
            [0],
            color="black",
            linewidth=3.5,
            marker="D",
            markersize=6,
            label="Barycenter",
        )
        legend_handles.append(bary_handle)

        fig.legend(
            handles=legend_handles,
            loc="upper left",
            bbox_to_anchor=(1.02, 1),
            fontsize=9,
            framealpha=0.9,
        )

    def plot_overlay(
        self,
        trajectories: list[NormalizedTrajectory],
        output_path: str | Path,
        cluster_result: DTWClusterResult | None = None,
        title: str = "Narrative Overlay",
    ) -> None:
        """Plot 9x2 grid (emotions x directions) with all narratives overlaid.

        Creates a figure with 9 rows (one per BASE_EMOTION) and 2 columns
        (A→B and B→A). Each subplot overlays all narrative lines for that
        emotion+direction combination.

        When cluster_result is provided: colors by CLUSTER_COLORS, alpha=0.3,
        barycenter shown, legend shows cluster labels.
        When None: tab10 colors, alpha=0.7, no cluster legend.

        Args:
            trajectories: List of NormalizedTrajectory objects.
            output_path: Path to save the output PNG.
            cluster_result: Optional DTWClusterResult for cluster coloring.
            title: Title for the overall figure.
        """
        grouped = self._group_trajectories(trajectories)
        sources = sorted(grouped.keys())

        if len(sources) < 2:
            logger.warning(
                "Overlay with %d narrative — comparison is limited",
                len(sources),
            )

        colors, alpha = self._resolve_colors_and_alpha(sources, cluster_result)

        # Determine x-axis from first trajectory
        first_traj = next(iter(grouped[sources[0]].values()))
        x = np.linspace(0, 1, len(first_traj))

        # Unpack barycenters if cluster_result provided
        barycenters: dict[int, dict[tuple[str, str], NDArray[np.float64]]] = {}
        if cluster_result is not None:
            for cluster_idx, bary in enumerate(cluster_result.barycenters):
                barycenters[cluster_idx] = self._unpack_barycenter(bary)

        # Build source-to-color-index mapping for barycenter color
        source_cluster_map: dict[str, int] = {}
        if cluster_result is not None:
            for src in sources:
                source_cluster_map[src] = cluster_result.assignments[src]

        fig, axs = plt.subplots(9, 2, figsize=self.figsize)
        fig.suptitle(title, fontsize=16, fontweight="bold")

        for i, emotion in enumerate(BASE_EMOTIONS):
            for j, direction in enumerate(DIRECTIONS):
                ax = axs[i, j]
                key = (direction, emotion)

                # Collect y-arrays for each narrative
                y_arrays: list[NDArray[np.float64]] = []
                subplot_colors: list[str] = []
                for src_idx, src in enumerate(sources):
                    if key in grouped[src]:
                        y_arrays.append(grouped[src][key])
                        subplot_colors.append(colors[src_idx])

                # Get barycenter for this subplot
                barycenter_y: NDArray[np.float64] | None = None
                if cluster_result is not None and barycenters:
                    # Show the first matching barycenter for this subplot
                    # In practice, each cluster has one barycenter
                    for cluster_idx in range(cluster_result.n_clusters):
                        if cluster_idx in barycenters and key in barycenters[cluster_idx]:
                            # Find the dominant cluster for this subplot
                            # Use the first source's cluster
                            pass

                # Use the cluster barycenter matching the first source's cluster
                if cluster_result is not None and sources:
                    first_cluster = cluster_result.assignments[sources[0]]
                    if first_cluster in barycenters and key in barycenters[first_cluster]:
                        barycenter_y = barycenters[first_cluster][key]

                label = f"{emotion} ({DIRECTION_LABELS[direction]})"
                self._plot_subplot_overlay(
                    ax, y_arrays, subplot_colors, alpha, barycenter_y, x, label
                )

        if cluster_result is not None:
            self._add_cluster_legend(fig, cluster_result)

        plt.tight_layout(rect=[0, 0, 1, 0.96])

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)

    def plot_emotion(
        self,
        trajectories: list[NormalizedTrajectory],
        emotion: str,
        output_path: str | Path,
        cluster_result: DTWClusterResult | None = None,
    ) -> None:
        """Plot single emotion with both A→B and B→A subplots.

        Creates a 1x2 subplot figure for one emotion, showing all narratives
        overlaid for both directions.

        Args:
            trajectories: List of NormalizedTrajectory objects.
            emotion: Emotion name from BASE_EMOTIONS.
            output_path: Path to save the output PNG.
            cluster_result: Optional DTWClusterResult for cluster coloring.
        """
        grouped = self._group_trajectories(trajectories)
        sources = sorted(grouped.keys())

        if len(sources) < 2:
            logger.warning(
                "Overlay with %d narrative — comparison is limited",
                len(sources),
            )

        colors, alpha = self._resolve_colors_and_alpha(sources, cluster_result)

        first_traj = next(iter(grouped[sources[0]].values()))
        x = np.linspace(0, 1, len(first_traj))

        barycenters: dict[int, dict[tuple[str, str], NDArray[np.float64]]] = {}
        if cluster_result is not None:
            for cluster_idx, bary in enumerate(cluster_result.barycenters):
                barycenters[cluster_idx] = self._unpack_barycenter(bary)

        fig, axs = plt.subplots(1, 2, figsize=(16, 5))
        fig.suptitle(f"{emotion} Overlay", fontsize=14, fontweight="bold")

        for j, direction in enumerate(DIRECTIONS):
            ax = axs[j]
            key = (direction, emotion)

            y_arrays: list[NDArray[np.float64]] = []
            subplot_colors: list[str] = []
            for src_idx, src in enumerate(sources):
                if key in grouped[src]:
                    y_arrays.append(grouped[src][key])
                    subplot_colors.append(colors[src_idx])

            barycenter_y: NDArray[np.float64] | None = None
            if cluster_result is not None and sources:
                first_cluster = cluster_result.assignments[sources[0]]
                if first_cluster in barycenters and key in barycenters[first_cluster]:
                    barycenter_y = barycenters[first_cluster][key]

            label = f"{emotion} ({DIRECTION_LABELS[direction]})"
            self._plot_subplot_overlay(
                ax, y_arrays, subplot_colors, alpha, barycenter_y, x, label
            )

        if cluster_result is not None:
            self._add_cluster_legend(fig, cluster_result)

        plt.tight_layout(rect=[0, 0, 1, 0.96])

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)

    def plot_direction(
        self,
        trajectories: list[NormalizedTrajectory],
        direction: str,
        output_path: str | Path,
        cluster_result: DTWClusterResult | None = None,
    ) -> None:
        """Plot all 9 emotions for one direction (A→B or B→A).

        Creates a 9x1 subplot figure with all emotions for the specified
        direction, showing all narratives overlaid.

        Args:
            trajectories: List of NormalizedTrajectory objects.
            direction: Either "A_to_B" or "B_to_A".
            output_path: Path to save the output PNG.
            cluster_result: Optional DTWClusterResult for cluster coloring.
        """
        grouped = self._group_trajectories(trajectories)
        sources = sorted(grouped.keys())

        if len(sources) < 2:
            logger.warning(
                "Overlay with %d narrative — comparison is limited",
                len(sources),
            )

        colors, alpha = self._resolve_colors_and_alpha(sources, cluster_result)

        first_traj = next(iter(grouped[sources[0]].values()))
        x = np.linspace(0, 1, len(first_traj))

        barycenters: dict[int, dict[tuple[str, str], NDArray[np.float64]]] = {}
        if cluster_result is not None:
            for cluster_idx, bary in enumerate(cluster_result.barycenters):
                barycenters[cluster_idx] = self._unpack_barycenter(bary)

        fig, axs = plt.subplots(9, 1, figsize=(10, 20))
        dir_label = DIRECTION_LABELS.get(direction, direction)
        fig.suptitle(f"Overlay ({dir_label})", fontsize=14, fontweight="bold")

        for i, emotion in enumerate(BASE_EMOTIONS):
            ax = axs[i]
            key = (direction, emotion)

            y_arrays: list[NDArray[np.float64]] = []
            subplot_colors: list[str] = []
            for src_idx, src in enumerate(sources):
                if key in grouped[src]:
                    y_arrays.append(grouped[src][key])
                    subplot_colors.append(colors[src_idx])

            barycenter_y: NDArray[np.float64] | None = None
            if cluster_result is not None and sources:
                first_cluster = cluster_result.assignments[sources[0]]
                if first_cluster in barycenters and key in barycenters[first_cluster]:
                    barycenter_y = barycenters[first_cluster][key]

            label = f"{emotion} ({dir_label})"
            self._plot_subplot_overlay(
                ax, y_arrays, subplot_colors, alpha, barycenter_y, x, label
            )

        if cluster_result is not None:
            self._add_cluster_legend(fig, cluster_result)

        plt.tight_layout(rect=[0, 0, 1, 0.96])

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)
