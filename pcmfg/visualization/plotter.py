"""Emotion visualization plotter for PCMFG.

Creates matplotlib visualizations of emotional trajectories
using raw 9 base emotions for both relationship directions.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from pcmfg.models.schemas import AxesTimeSeries, EmotionTimeSeries

# Emotion display configuration
EMOTION_CONFIG = {
    "Joy": {"color": "#2ecc71", "description": "Happiness, pleasure, delight"},
    "Trust": {"color": "#3498db", "description": "Safety, reliance, vulnerability"},
    "Fear": {"color": "#9b59b6", "description": "Panic, dread, terror, anxiety"},
    "Surprise": {"color": "#f39c12", "description": "Astonishment, shock"},
    "Sadness": {"color": "#34495e", "description": "Grief, sorrow, despair"},
    "Disgust": {"color": "#27ae60", "description": "Revulsion, aversion, contempt"},
    "Anger": {"color": "#e74c3c", "description": "Fury, rage, frustration"},
    "Anticipation": {
        "color": "#1abc9c",
        "description": "Looking forward to, expecting",
    },
    "Arousal": {"color": "#e91e63", "description": "Physical lust, romantic desire"},
}

EMOTION_LIST = list(EMOTION_CONFIG.keys())


class EmotionPlotter:
    """Creates visualization plots for PCMFG analysis results.

    Generates matplotlib figures showing raw emotion trajectories
    for both relationship directions (A→B and B→A).
    """

    def __init__(self, dpi: int = 300, figsize: tuple[float, float] = (16, 20)) -> None:
        """Initialize the plotter.

        Args:
            dpi: Image resolution (dots per inch).
            figsize: Figure size in inches (width, height).
        """
        self.dpi = dpi
        self.figsize = figsize

    def plot_timeseries(
        self,
        timeseries: dict[str, EmotionTimeSeries],
        output_path: str | Path,
        title: str = "Emotional Trajectory",
        main_pairing: list[str] | None = None,
    ) -> None:
        """Plot raw 9 emotion time-series as a 9×2 grid.

        Args:
            timeseries: Dictionary with "A_to_B" and "B_to_A" EmotionTimeSeries.
            output_path: Path to save the output image.
            title: Title for the overall figure.
            main_pairing: Optional [Character A, Character B] names for labels.
        """
        a_to_b = timeseries.get("A_to_B")
        b_to_a = timeseries.get("B_to_A")

        if a_to_b is None and b_to_a is None:
            raise ValueError("At least one time-series direction must be provided")

        # Determine labels
        if main_pairing and len(main_pairing) >= 2:
            label_a_to_b = f"{main_pairing[0]} → {main_pairing[1]}"
            label_b_to_a = f"{main_pairing[1]} → {main_pairing[0]}"
        else:
            label_a_to_b = "A → B"
            label_b_to_a = "B → A"

        # Create 9×2 grid (9 emotions, 2 directions)
        fig, axs = plt.subplots(9, 2, figsize=self.figsize)
        fig.suptitle(title, fontsize=16, fontweight="bold")

        for i, emotion in enumerate(EMOTION_LIST):
            # Left column: A→B
            if a_to_b is not None:
                values = getattr(a_to_b, emotion, [])
                self._plot_single_emotion(
                    axs[i, 0],
                    emotion,
                    values,
                    f"{emotion} ({label_a_to_b})",
                )
            else:
                axs[i, 0].text(0.5, 0.5, "No data", ha="center", va="center")
                axs[i, 0].set_title(f"{emotion} ({label_a_to_b})")

            # Right column: B→A
            if b_to_a is not None:
                values = getattr(b_to_a, emotion, [])
                self._plot_single_emotion(
                    axs[i, 1],
                    emotion,
                    values,
                    f"{emotion} ({label_b_to_a})",
                )
            else:
                axs[i, 1].text(0.5, 0.5, "No data", ha="center", va="center")
                axs[i, 1].set_title(f"{emotion} ({label_b_to_a})")

        plt.tight_layout(rect=[0, 0, 1, 0.96])  # Make room for suptitle

        # Save the figure
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)

    def _plot_single_emotion(
        self,
        ax: plt.Axes,
        emotion: str,
        values: list[float],
        title: str,
    ) -> None:
        """Plot a single emotion trajectory.

        Args:
            ax: Matplotlib axes to plot on.
            emotion: Emotion name for color lookup.
            values: Time series values.
            title: Plot title.
        """
        config = EMOTION_CONFIG.get(emotion, {"color": "#3498db", "description": ""})
        color = config["color"]

        # Handle empty data
        if not values:
            ax.text(
                0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes
            )
            ax.set_title(title, fontsize=10)
            ax.grid(True, alpha=0.3)
            return

        # X-axis: chunk positions (0 to 1)
        x = np.linspace(0, 1, len(values))

        # Plot the line
        ax.plot(x, values, color=color, linewidth=2, marker="o", markersize=3)

        # Fill area under the curve
        ax.fill_between(x, values, alpha=0.3, color=color)

        # Add baseline reference line at y=1
        ax.axhline(
            y=1, color="gray", linestyle="--", linewidth=1, alpha=0.7, label="Baseline"
        )

        # Set y-axis limits (1-5 scale)
        ax.set_ylim(0.5, 5.5)
        ax.set_yticks([1, 2, 3, 4, 5])
        ax.set_yticklabels(["1\n(Baseline)", "2", "3", "4", "5\n(Extreme)"], fontsize=8)

        # Set x-axis
        ax.set_xlim(0, 1)
        ax.set_xlabel("Narrative Progress", fontsize=8)

        # Title and labels
        ax.set_title(title, fontsize=10)
        ax.set_ylabel("Intensity", fontsize=8)

        # Grid
        ax.grid(True, alpha=0.3)

    def plot_directional_comparison(
        self,
        timeseries: dict[str, EmotionTimeSeries],
        output_path: str | Path,
        title: str = "Emotional Trajectory: Directional Comparison",
        main_pairing: list[str] | None = None,
    ) -> None:
        """Plot A→B vs B→A on same plot for each emotion to show relationship dynamics.

        This visualization reveals the asymmetry in emotions between characters,
        highlighting areas of mutual feeling vs one-sided emotion.

        Args:
            timeseries: Dictionary with "A_to_B" and "B_to_A" EmotionTimeSeries.
            output_path: Path to save the output image.
            title: Title for the overall figure.
            main_pairing: Optional [Character A, Character B] names for labels.
        """
        a_to_b = timeseries.get("A_to_B")
        b_to_a = timeseries.get("B_to_A")

        if a_to_b is None and b_to_a is None:
            raise ValueError("At least one direction must have time-series data")

        # Create labels
        if main_pairing and len(main_pairing) >= 2:
            label_a_to_b = f"{main_pairing[0]} → {main_pairing[1]}"
            label_b_to_a = f"{main_pairing[1]} → {main_pairing[0]}"
        else:
            label_a_to_b = "A → B"
            label_b_to_a = "B → A"

        # Create 3×3 grid for 9 emotions
        fig, axs = plt.subplots(3, 3, figsize=(16, 14))
        fig.suptitle(title, fontsize=16, fontweight="bold")

        axs_flat = axs.flatten()

        # Color scheme: warm for A→B, cool for B→A
        color_a_to_b = "#e74c3c"  # Red/warm
        color_b_to_a = "#3498db"  # Blue/cool

        for i, emotion in enumerate(EMOTION_LIST):
            ax = axs_flat[i]
            config = EMOTION_CONFIG[emotion]

            # Get values for both directions
            values_a_to_b = getattr(a_to_b, emotion, []) if a_to_b else []
            values_b_to_a = getattr(b_to_a, emotion, []) if b_to_a else []

            # X-axis: narrative progress
            x_a = np.linspace(0, 1, len(values_a_to_b)) if values_a_to_b else []
            x_b = np.linspace(0, 1, len(values_b_to_a)) if values_b_to_a else []

            # Plot A→B direction
            if len(values_a_to_b) > 0:
                ax.plot(
                    x_a,
                    values_a_to_b,
                    color=color_a_to_b,
                    linewidth=2.5,
                    marker="o",
                    markersize=3,
                    label=label_a_to_b,
                    alpha=0.9,
                )

            # Plot B→A direction
            if len(values_b_to_a) > 0:
                ax.plot(
                    x_b,
                    values_b_to_a,
                    color=color_b_to_a,
                    linewidth=2.5,
                    marker="s",
                    markersize=3,
                    label=label_b_to_a,
                    alpha=0.9,
                )

            # Baseline reference line
            ax.axhline(
                y=1,
                color="gray",
                linestyle="--",
                linewidth=1.5,
                alpha=0.7,
                label="Baseline",
            )

            # Formatting
            ax.set_ylim(0.5, 5.5)
            ax.set_yticks([1, 2, 3, 4, 5])
            ax.set_yticklabels(
                ["1\n(Baseline)", "2", "3", "4", "5\n(Extreme)"], fontsize=8
            )
            ax.set_xlim(0, 1)
            ax.set_xlabel("Narrative Progress", fontsize=9)
            ax.set_title(
                f"{emotion}\n({config['description']})",
                fontsize=11,
                fontweight="bold",
            )
            ax.set_ylabel("Intensity", fontsize=9)
            ax.grid(True, alpha=0.3)
            ax.legend(loc="upper right", fontsize=8, framealpha=0.9)

        plt.tight_layout(rect=[0, 0, 1, 0.96])

        # Save the figure
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)

    def plot_emotion_gap(
        self,
        timeseries: dict[str, EmotionTimeSeries],
        output_path: str | Path,
        title: str = "Emotional Gap Analysis",
        main_pairing: list[str] | None = None,
    ) -> None:
        """Plot the difference (gap) between A→B and B→A for each emotion.

        Positive values = A feels more than B
        Negative values = B feels more than A
        Zero = mutual/equal feeling

        Args:
            timeseries: Dictionary with "A_to_B" and "B_to_A" EmotionTimeSeries.
            output_path: Path to save the output image.
            title: Title for the overall figure.
            main_pairing: Optional [Character A, Character B] names for labels.
        """
        a_to_b = timeseries.get("A_to_B")
        b_to_a = timeseries.get("B_to_A")

        if a_to_b is None or b_to_a is None:
            raise ValueError("Both directions must have time-series data")

        # Create 3×3 grid for 9 emotions
        fig, axs = plt.subplots(3, 3, figsize=(16, 14))
        fig.suptitle(title, fontsize=16, fontweight="bold")

        axs_flat = axs.flatten()

        # Color scheme
        color_positive = "#e74c3c"  # Red - A feels more
        color_negative = "#3498db"  # Blue - B feels more

        for i, emotion in enumerate(EMOTION_LIST):
            ax = axs_flat[i]
            config = EMOTION_CONFIG[emotion]

            values_a = np.array(getattr(a_to_b, emotion, []))
            values_b = np.array(getattr(b_to_a, emotion, []))

            if len(values_a) == 0 or len(values_b) == 0:
                ax.text(
                    0.5,
                    0.5,
                    "No data",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
                ax.set_title(f"{emotion}", fontsize=11, fontweight="bold")
                continue

            # Calculate gap (A - B)
            min_len = min(len(values_a), len(values_b))
            gap = values_a[:min_len] - values_b[:min_len]
            x = np.linspace(0, 1, min_len)

            # Plot gap as bars
            colors = [color_positive if g > 0 else color_negative for g in gap]
            ax.bar(x, gap, width=1 / min_len, color=colors, alpha=0.7, edgecolor="none")

            # Zero line
            ax.axhline(y=0, color="black", linewidth=1.5, linestyle="-")

            # Fill regions
            ax.fill_between(x, gap, 0, where=(gap > 0), alpha=0.3, color=color_positive)
            ax.fill_between(x, gap, 0, where=(gap < 0), alpha=0.3, color=color_negative)

            # Formatting
            ax.set_ylim(-4, 4)
            ax.set_xlim(0, 1)
            ax.set_xlabel("Narrative Progress", fontsize=9)
            ax.set_title(
                f"{emotion}\n({config['description']})",
                fontsize=11,
                fontweight="bold",
            )
            ax.set_ylabel("Gap (A→B - B→A)", fontsize=9)
            ax.grid(True, alpha=0.3, axis="y")

            # Add labels for interpretation
            if main_pairing and len(main_pairing) >= 2:
                ax.text(
                    0.02,
                    3.5,
                    f"{main_pairing[0]} feels more",
                    fontsize=7,
                    color=color_positive,
                    alpha=0.8,
                )
                ax.text(
                    0.02,
                    -3.5,
                    f"{main_pairing[1]} feels more",
                    fontsize=7,
                    color=color_negative,
                    alpha=0.8,
                )

        # Add legend
        legend_elements = [
            plt.Rectangle(
                (0, 0),
                1,
                1,
                fc=color_positive,
                alpha=0.7,
                label=f"{main_pairing[0]} feels more"
                if main_pairing
                else "A feels more",
            ),
            plt.Rectangle(
                (0, 0),
                1,
                1,
                fc=color_negative,
                alpha=0.7,
                label=f"{main_pairing[1]} feels more"
                if main_pairing
                else "B feels more",
            ),
        ]
        fig.legend(handles=legend_elements, loc="upper right", fontsize=10)

        plt.tight_layout(rect=[0, 0, 1, 0.96])

        # Save the figure
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)

    def plot_comparison(
        self,
        timeseries_a: dict[str, EmotionTimeSeries],
        timeseries_b: dict[str, EmotionTimeSeries],
        output_path: str | Path,
        label_a: str = "Analysis A",
        label_b: str = "Analysis B",
        title: str = "Emotional Trajectory Comparison",
    ) -> None:
        """Plot two analyses side by side for comparison.

        Args:
            timeseries_a: First analysis time-series.
            timeseries_b: Second analysis time-series.
            output_path: Path to save the output image.
            label_a: Label for first analysis.
            label_b: Label for second analysis.
            title: Title for the overall figure.
        """
        # Use A→B from both analyses for comparison
        a_to_b_a = timeseries_a.get("A_to_B")
        a_to_b_b = timeseries_b.get("A_to_B")

        if a_to_b_a is None and a_to_b_b is None:
            raise ValueError("At least one analysis must have A_to_B time-series")

        # Create 3×3 grid for 9 emotions
        fig, axs = plt.subplots(3, 3, figsize=(15, 12))
        fig.suptitle(title, fontsize=16, fontweight="bold")

        axs_flat = axs.flatten()

        for i, emotion in enumerate(EMOTION_LIST):
            ax = axs_flat[i]
            config = EMOTION_CONFIG[emotion]
            color = config["color"]

            # Plot both analyses
            if a_to_b_a is not None:
                values_a = getattr(a_to_b_a, emotion, [])
                x_a = np.linspace(0, 1, len(values_a)) if values_a else [0]
                ax.plot(x_a, values_a, color="#3498db", linewidth=2, label=label_a)

            if a_to_b_b is not None:
                values_b = getattr(a_to_b_b, emotion, [])
                x_b = np.linspace(0, 1, len(values_b)) if values_b else [0]
                ax.plot(x_b, values_b, color="#e74c3c", linewidth=2, label=label_b)

            # Baseline
            ax.axhline(y=1, color="gray", linestyle="--", linewidth=1, alpha=0.7)

            # Formatting
            ax.set_ylim(0.5, 5.5)
            ax.set_yticks([1, 2, 3, 4, 5])
            ax.set_xlim(0, 1)
            ax.set_xlabel("Narrative Progress", fontsize=8)
            ax.set_title(f"{emotion}\n({config['description']})", fontsize=10)
            ax.set_ylabel("Intensity", fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)

        plt.tight_layout(rect=[0, 0, 1, 0.96])

        # Save the figure
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)

    def plot_axes(
        self,
        axes: AxesTimeSeries,
        output_path: str | Path,
        title: str = "Emotional Trajectory (Axes)",
    ) -> None:
        """Plot the four romance axes as a 4-panel figure (DEPRECATED).

        This method is kept for backward compatibility but the main
        visualization now uses raw emotions via plot_timeseries().

        Args:
            axes: Time series data for all four axes.
            output_path: Path to save the output image.
            title: Title for the overall figure.
        """
        fig, axs = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(
            f"{title} (DEPRECATED - use raw emotions)", fontsize=14, fontweight="bold"
        )

        # Flatten axes for easier iteration
        axs_flat = axs.flatten()

        axis_data = [
            ("Intimacy", axes.intimacy, "Closeness, trust, vulnerability"),
            ("Passion", axes.passion, "Desire, excitement, intensity"),
            ("Hostility", axes.hostility, "Anger, resentment, conflict"),
            ("Anxiety", axes.anxiety, "Fear, uncertainty, tension"),
        ]

        colors = {
            "intimacy": "#2ecc71",
            "passion": "#e74c3c",
            "hostility": "#e67e22",
            "anxiety": "#9b59b6",
        }

        for ax, (name, values, subtitle) in zip(axs_flat, axis_data):
            color = colors.get(name.lower(), "#3498db")

            # Skip plotting if no data
            if not values:
                ax.text(
                    0.5,
                    0.5,
                    "No data",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
                ax.set_title(f"{name}\n({subtitle})", fontsize=11)
                ax.grid(True, alpha=0.3)
                continue

            # X-axis: chunk positions (0 to 1)
            x = np.linspace(0, 1, len(values))

            # Plot the line
            ax.plot(x, values, color=color, linewidth=2, marker="o", markersize=4)

            # Fill area under the curve
            ax.fill_between(x, values, alpha=0.3, color=color)

            # Add baseline reference line at y=1
            ax.axhline(
                y=1,
                color="gray",
                linestyle="--",
                linewidth=1,
                alpha=0.7,
                label="Baseline",
            )

            # Set y-axis limits (1-5 scale)
            ax.set_ylim(0.5, 5.5)
            ax.set_yticks([1, 2, 3, 4, 5])
            ax.set_yticklabels(["1\n(Baseline)", "2", "3", "4", "5\n(Extreme)"])

            # Set x-axis
            ax.set_xlim(0, 1)
            ax.set_xlabel("Narrative Progress")

            # Title and labels
            ax.set_title(f"{name}\n({subtitle})", fontsize=11)
            ax.set_ylabel("Intensity")

            # Grid
            ax.grid(True, alpha=0.3)

        plt.tight_layout()

        # Save the figure
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)

    def export_data(
        self,
        timeseries: dict[str, EmotionTimeSeries],
        output_path: str | Path,
        format: str = "csv",
    ) -> None:
        """Export time-series data to CSV or JSON.

        Args:
            timeseries: Dictionary with "A_to_B" and "B_to_A" EmotionTimeSeries.
            output_path: Path to save the output file.
            format: Output format ("csv" or "json").
        """
        import json

        import pandas as pd

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        a_to_b = timeseries.get("A_to_B")
        b_to_a = timeseries.get("B_to_A")

        if format == "csv":
            data = {"position": []}

            # Get length from first available time-series
            length = 0
            if a_to_b is not None and a_to_b.Joy:
                length = len(a_to_b.Joy)
            elif b_to_a is not None and b_to_a.Joy:
                length = len(b_to_a.Joy)

            data["position"] = list(np.linspace(0, 1, length)) if length > 0 else []

            # Add A_to_B data
            if a_to_b is not None:
                for emotion in EMOTION_LIST:
                    data[f"A_to_B_{emotion}"] = getattr(a_to_b, emotion, [])

            # Add B_to_A data
            if b_to_a is not None:
                for emotion in EMOTION_LIST:
                    data[f"B_to_A_{emotion}"] = getattr(b_to_a, emotion, [])

            df = pd.DataFrame(data)
            df.to_csv(output_path, index=False)

        elif format == "json":
            json_data = {}

            if a_to_b is not None:
                json_data["A_to_B"] = {
                    emotion: getattr(a_to_b, emotion, []) for emotion in EMOTION_LIST
                }

            if b_to_a is not None:
                json_data["B_to_A"] = {
                    emotion: getattr(b_to_a, emotion, []) for emotion in EMOTION_LIST
                }

            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(json_data, f, indent=2)
