"""Visualization for Interesting Section Detection results.

Plots the Matrix Profile with annotated discords, segment boundaries,
motif pairs, and gap analysis heatmaps.
"""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from pcmfg.models.schemas import (
    BASE_EMOTIONS,
    InterestingSectionReport,
)

logger = logging.getLogger(__name__)


def plot_interesting_report(
    report: InterestingSectionReport,
    output_path: Path,
    title: str | None = None,
    dpi: int = 150,
) -> None:
    """Generate a multi-panel visualization of the detection report.

    Panels:
    1. Matrix Profile with discord and segment annotations
    2. Gap analysis heatmap at interesting points

    Args:
        report: The detection report to visualize.
        output_path: Where to save the plot.
        title: Optional title for the plot.
        dpi: Resolution for saved figure.
    """
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), height_ratios=[1, 1])

    if title:
        fig.suptitle(title, fontsize=14, fontweight="bold")

    _plot_matrix_profile(axes[0], report)
    _plot_gap_heatmap(axes[1], report)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved interesting section plot to %s", output_path)


def _plot_matrix_profile(ax: plt.Axes, report: InterestingSectionReport) -> None:
    """Plot Matrix Profile distances with discord/segment annotations."""
    mp = report.matrix_profile_distances
    x = np.arange(len(mp))

    ax.plot(x, mp, color="#2196F3", linewidth=1.0, alpha=0.8, label="Matrix Profile")

    # Annotate discords
    for i, d in enumerate(report.discords):
        if d.index < len(mp):
            ax.axvspan(
                d.index,
                d.index + d.window_size,
                alpha=0.2,
                color="red",
                label="Discord" if i == 0 else None,
            )
            ax.annotate(
                f"#{i+1}",
                xy=(d.index, d.distance),
                fontsize=8,
                color="red",
                fontweight="bold",
            )

    # Annotate segments
    for seg in report.segments:
        ax.axvline(seg.index, color="green", linestyle="--", alpha=0.7, linewidth=1.5)
        ax.annotate(
            seg.regime_label,
            xy=(seg.index, ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 1),
            fontsize=8,
            color="green",
            fontweight="bold",
            rotation=90,
            va="top",
        )

    ax.set_title("Matrix Profile — Discord Discovery & Segmentation")
    ax.set_xlabel("Chunk Index")
    ax.set_ylabel("Distance to Nearest Neighbor")
    ax.legend(loc="upper right", fontsize=8)


def _plot_gap_heatmap(ax: plt.Axes, report: InterestingSectionReport) -> None:
    """Plot gap analysis as a heatmap."""
    if not report.gaps:
        ax.text(0.5, 0.5, "No gap data", ha="center", va="center")
        ax.set_title("Gap Analysis — No Data")
        return

    # Build gap matrix: rows = timestamps, cols = emotions
    emotions = list(BASE_EMOTIONS)
    gap_matrix = np.zeros((len(report.gaps), len(emotions)))

    for i, gap_ts in enumerate(report.gaps):
        for j, emotion in enumerate(emotions):
            for gv in gap_ts.gaps:
                if gv.emotion == emotion:
                    gap_matrix[i, j] = gv.gap
                    break

    im = ax.imshow(
        gap_matrix.T,
        aspect="auto",
        cmap="RdBu_r",
        vmin=-4,
        vmax=4,
        interpolation="nearest",
    )

    ax.set_yticks(range(len(emotions)))
    ax.set_yticklabels(emotions, fontsize=8)
    ax.set_xticks(range(len(report.gaps)))
    ax.set_xticklabels(
        [f"{g.position:.2f}" for g in report.gaps],
        fontsize=7,
        rotation=45,
    )
    ax.set_xlabel("Narrative Position")
    ax.set_title("Gap Analysis — A→B minus B→A (red = A higher, blue = B higher)")

    plt.colorbar(im, ax=ax, shrink=0.8, label="Gap Value")
