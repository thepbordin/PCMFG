import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from pcmfg.models.schemas import BASE_EMOTIONS, NormalizedTrajectory
from pcmfg.analysis.dtw_clusterer import build_dtw_dataset, TSLEARN_AVAILABLE

if TSLEARN_AVAILABLE:
    from tslearn.metrics import dtw_path

logger = logging.getLogger(__name__)

class DTWAlignmentPlotter:
    """Plots the DTW alignment path between two narratives."""

    def __init__(self, dpi: int = 300, figsize: tuple[float, float] = (12, 6)):
        self.dpi = dpi
        self.figsize = figsize

    def plot_alignment(
        self,
        trajectories: list[NormalizedTrajectory],
        source_1: str,
        source_2: str,
        emotion: str = "Joy",
        direction: str = "A_to_B",
        output_path: str | Path = "dtw_alignment.png",
        offset: float = 4.0,
        colors: tuple[str, str] = ("#3498db", "#2ecc71"),
    ) -> None:
        """Plots the 2D alignment path for a specific emotion between 2 stories."""
        if not TSLEARN_AVAILABLE:
            logger.error("tslearn not installed. Cannot plot DTW alignment.")
            return

        dataset, sources, n_points = build_dtw_dataset(trajectories)
        
        if source_1 not in sources or source_2 not in sources:
            logger.error(f"Source not found. Available: {sources}")
            return
            
        s1_idx = sources.index(source_1)
        s2_idx = sources.index(source_2)
        
        data_1 = dataset[s1_idx]
        data_2 = dataset[s2_idx]
        
        # Calculate full 18-dimensional DTW path
        path, distance = dtw_path(data_1, data_2)
        
        # Find column index for the selected emotion and direction
        e_idx = BASE_EMOTIONS.index(emotion)
        dir_offset = 0 if direction == "A_to_B" else 1
        col_idx = e_idx * 2 + dir_offset
        
        y1 = data_1[:, col_idx]
        y2 = data_2[:, col_idx]
        
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Plot source 1 on top, source 2 shifted down by offset
        ax.plot(y1, color=colors[0], label=f"{source_1} ({emotion} {direction})", linewidth=2)
        y2_shifted = y2 - offset
        ax.plot(y2_shifted, color=colors[1], label=f"{source_2} ({emotion} {direction})", linewidth=2)
        
        # Draw dotted alignment lines
        for (i, j) in path:
            # Draw every point for accuracy, or perhaps every 2nd or 3rd to avoid dense clutter
            if i % 3 == 0 or j % 3 == 0:
                ax.plot([i, j], [y1[i], y2_shifted[j]], color="gray", linestyle=":", alpha=0.4, zorder=-1)
                
        plt.title(f"DTW Alignment: {source_1} vs {source_2}\n(Warped across 18 dims, showing {emotion} {direction})")
        plt.xlabel("Narrative Timeline (%)")
        plt.ylabel("Emotion Intensity (1-5)")
        # Hide Y ticks because of the offset
        ax.set_yticks([])
        
        plt.legend(loc="upper right")
        plt.tight_layout()
        
        plt.savefig(output_path, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved DTW alignment plot to {output_path}")

