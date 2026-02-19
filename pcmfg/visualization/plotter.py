"""Emotion visualization plotter for PCMFG.

Creates matplotlib visualizations of emotional trajectories
across the four romance axes.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from pcmfg.models.schemas import AxesTimeSeries


class EmotionPlotter:
    """Creates visualization plots for PCMFG analysis results.

    Generates a 4-panel matplotlib figure showing each romance axis
    over the course of the narrative.
    """

    def __init__(self, dpi: int = 300, figsize: tuple[float, float] = (12, 10)) -> None:
        """Initialize the plotter.

        Args:
            dpi: Image resolution (dots per inch).
            figsize: Figure size in inches (width, height).
        """
        self.dpi = dpi
        self.figsize = figsize

        # Color scheme for each axis
        self.colors = {
            "intimacy": "#2ecc71",  # Green
            "passion": "#e74c3c",  # Red
            "hostility": "#e67e22",  # Orange
            "anxiety": "#9b59b6",  # Purple
        }

    def plot_axes(
        self,
        axes: AxesTimeSeries,
        output_path: str | Path,
        title: str = "Emotional Trajectory",
    ) -> None:
        """Plot the four romance axes as a 4-panel figure.

        Args:
            axes: Time series data for all four axes.
            output_path: Path to save the output image.
            title: Title for the overall figure.
        """
        fig, axs = plt.subplots(2, 2, figsize=self.figsize)
        fig.suptitle(title, fontsize=16, fontweight="bold")

        # Flatten axes for easier iteration
        axs_flat = axs.flatten()

        axis_data = [
            ("Intimacy", axes.intimacy, "Closeness, trust, vulnerability"),
            ("Passion", axes.passion, "Desire, excitement, intensity"),
            ("Hostility", axes.hostility, "Anger, resentment, conflict"),
            ("Anxiety", axes.anxiety, "Fear, uncertainty, tension"),
        ]

        for ax, (name, values, subtitle) in zip(axs_flat, axis_data):
            self._plot_single_axis(ax, name.lower(), values, name, subtitle)

        plt.tight_layout()

        # Save the figure
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)

    def _plot_single_axis(
        self,
        ax: plt.Axes,
        axis_key: str,
        values: list[float],
        title: str,
        subtitle: str,
    ) -> None:
        """Plot a single axis on a subplot.

        Args:
            ax: Matplotlib axes to plot on.
            axis_key: Key for color lookup.
            values: Time series values.
            title: Axis title.
            subtitle: Axis subtitle/description.
        """
        color = self.colors.get(axis_key, "#3498db")

        # X-axis: chunk positions (0 to 1)
        x = np.linspace(0, 1, len(values)) if values else [0]

        # Plot the line
        ax.plot(x, values, color=color, linewidth=2, marker="o", markersize=4)

        # Fill area under the curve
        ax.fill_between(x, values, alpha=0.3, color=color)

        # Add baseline reference line at y=1
        ax.axhline(y=1, color="gray", linestyle="--", linewidth=1, alpha=0.7, label="Baseline")

        # Set y-axis limits (1-5 scale)
        ax.set_ylim(0.5, 5.5)
        ax.set_yticks([1, 2, 3, 4, 5])
        ax.set_yticklabels(["1\n(Baseline)", "2", "3", "4", "5\n(Extreme)"])

        # Set x-axis
        ax.set_xlim(0, 1)
        ax.set_xlabel("Narrative Progress")

        # Title and labels
        ax.set_title(f"{title}\n({subtitle})", fontsize=11)
        ax.set_ylabel("Intensity")

        # Grid
        ax.grid(True, alpha=0.3)

    def plot_comparison(
        self,
        axes_a: AxesTimeSeries,
        axes_b: AxesTimeSeries,
        output_path: str | Path,
        label_a: str = "Analysis A",
        label_b: str = "Analysis B",
        title: str = "Emotional Trajectory Comparison",
    ) -> None:
        """Plot two analyses side by side for comparison.

        Args:
            axes_a: First analysis time series.
            axes_b: Second analysis time series.
            output_path: Path to save the output image.
            label_a: Label for first analysis.
            label_b: Label for second analysis.
            title: Title for the overall figure.
        """
        fig, axs = plt.subplots(2, 2, figsize=self.figsize)
        fig.suptitle(title, fontsize=16, fontweight="bold")

        axs_flat = axs.flatten()

        axis_data = [
            ("Intimacy", axes_a.intimacy, axes_b.intimacy),
            ("Passion", axes_a.passion, axes_b.passion),
            ("Hostility", axes_a.hostility, axes_b.hostility),
            ("Anxiety", axes_a.anxiety, axes_b.anxiety),
        ]

        for ax, (name, values_a, values_b) in zip(axs_flat, axis_data):
            self._plot_comparison_axis(ax, name.lower(), values_a, values_b, name, label_a, label_b)

        # Add legend
        handles = [
            plt.Line2D([0], [0], color="#3498db", linewidth=2, label=label_a),
            plt.Line2D([0], [0], color="#e74c3c", linewidth=2, label=label_b),
        ]
        fig.legend(handles=handles, loc="upper right", fontsize=10)

        plt.tight_layout()

        # Save the figure
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)

    def _plot_comparison_axis(
        self,
        ax: plt.Axes,
        axis_key: str,
        values_a: list[float],
        values_b: list[float],
        title: str,
        label_a: str,
        label_b: str,
    ) -> None:
        """Plot comparison for a single axis.

        Args:
            ax: Matplotlib axes to plot on.
            axis_key: Key for color lookup.
            values_a: First analysis values.
            values_b: Second analysis values.
            title: Axis title.
            label_a: Label for first analysis.
            label_b: Label for second analysis.
        """
        x_a = np.linspace(0, 1, len(values_a)) if values_a else [0]
        x_b = np.linspace(0, 1, len(values_b)) if values_b else [0]

        # Plot both lines
        ax.plot(x_a, values_a, color="#3498db", linewidth=2, label=label_a)
        ax.plot(x_b, values_b, color="#e74c3c", linewidth=2, label=label_b)

        # Baseline
        ax.axhline(y=1, color="gray", linestyle="--", linewidth=1, alpha=0.7)

        # Formatting
        ax.set_ylim(0.5, 5.5)
        ax.set_yticks([1, 2, 3, 4, 5])
        ax.set_xlim(0, 1)
        ax.set_xlabel("Narrative Progress")
        ax.set_title(title, fontsize=12)
        ax.set_ylabel("Intensity")
        ax.grid(True, alpha=0.3)

    def export_data(
        self,
        axes: AxesTimeSeries,
        output_path: str | Path,
        format: str = "csv",
    ) -> None:
        """Export axes data to CSV or JSON.

        Args:
            axes: Time series data for all four axes.
            output_path: Path to save the output file.
            format: Output format ("csv" or "json").
        """
        import json

        import pandas as pd

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if format == "csv":
            df = pd.DataFrame(
                {
                    "position": np.linspace(0, 1, len(axes.intimacy)),
                    "intimacy": axes.intimacy,
                    "passion": axes.passion,
                    "hostility": axes.hostility,
                    "anxiety": axes.anxiety,
                }
            )
            df.to_csv(output_path, index=False)

        elif format == "json":
            data = {
                "intimacy": axes.intimacy,
                "passion": axes.passion,
                "hostility": axes.hostility,
                "anxiety": axes.anxiety,
            }
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
