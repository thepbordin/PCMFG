"""Narrative normalizer for cross-narrative emotional trajectory comparison.

Resamples variable-length PCMFG emotion time-series to a uniform N-point
grid on [0.0, 1.0] using nearest-neighbor interpolation. This enables
direct comparison of emotional arcs from narratives of different lengths.

Uses numpy.searchsorted (not scipy.interpolate.interp1d which is now
legacy in SciPy v1.17+) for nearest-neighbor resampling with non-uniform
original positions.
"""

import logging

import numpy as np
from numpy.typing import NDArray

from pcmfg.models.schemas import (
    BASE_EMOTIONS,
    AnalysisResult,
    NormalizedTrajectory,
)

logger = logging.getLogger(__name__)


def resample_nearest(
    x_original: NDArray[np.float64],
    y_original: NDArray[np.float64],
    n_points: int = 100,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Resample using nearest-neighbor interpolation.

    All output y values are present in y_original (no interpolation artifacts).
    Handles non-uniform x_original positions.

    Args:
        x_original: Original x-coordinates (chunk positions), shape (M,).
            Must be sorted in ascending order.
        y_original: Original y-values (emotion scores), shape (M,).
        n_points: Number of points in the uniform output grid.

    Returns:
        Tuple of (x_grid, y_resampled) where x_grid is uniform [0, 1].

    Raises:
        ValueError: If x_original is empty.
    """
    if len(x_original) == 0:
        raise ValueError("Cannot resample empty array")

    # Edge case: single point — repeat for all grid points
    if len(x_original) == 1:
        x_grid = np.linspace(0.0, 1.0, n_points)
        return x_grid, np.full(n_points, y_original[0])

    # Compute halfway boundaries between consecutive original points
    x_bds = x_original[:-1] / 2.0 + x_original[1:] / 2.0

    # Uniform target grid
    x_grid = np.linspace(0.0, 1.0, n_points)

    # Find which original bin each target point falls into
    indices = np.searchsorted(x_bds, x_grid, side="left")
    indices = np.clip(indices, 0, len(x_original) - 1)

    return x_grid, y_original[indices]


class NarrativeNormalizer:
    """Resample emotion time-series to a uniform N-point grid on [0.0, 1.0].

    Consumes existing AnalysisResult objects and produces NormalizedTrajectory
    objects — one per (direction, emotion) pair. For a typical result with
    2 directions and 9 emotions, normalize() returns 18 trajectories.

    Uses nearest-neighbor interpolation to preserve integer emotion scores
    (1-5 scale). No fractional values are introduced.

    Example:
        >>> normalizer = NarrativeNormalizer(n_points=100)
        >>> trajectories = normalizer.normalize(analysis_result)
        >>> joy_a2b = next(t for t in trajectories
        ...                 if t.direction == "A_to_B" and t.emotion == "Joy")
        >>> len(joy_a2b.y)  # 100
    """

    def __init__(self, n_points: int = 100) -> None:
        """Initialize the normalizer.

        Args:
            n_points: Number of resampling points on [0.0, 1.0] grid.
                Must be >= 2. Default 100.

        Raises:
            ValueError: If n_points < 2.
        """
        if n_points < 2:
            raise ValueError(f"n_points must be >= 2, got {n_points}")
        self.n_points = n_points

    def normalize(self, result: AnalysisResult) -> list[NormalizedTrajectory]:
        """Normalize a single AnalysisResult into per-emotion, per-direction trajectories.

        Produces one NormalizedTrajectory for each (direction, emotion) pair.
        For a result with 2 directions and 9 emotions, returns 18 trajectories.

        Args:
            result: PCMFG analysis result with chunks, timeseries, and metadata.

        Returns:
            List of NormalizedTrajectory objects, one per (direction, emotion).
        """
        trajectories: list[NormalizedTrajectory] = []
        directions = ["A_to_B", "B_to_A"]

        # Extract positions from chunks
        positions = np.array(
            [chunk.position for chunk in result.chunks], dtype=np.float64
        )

        if len(positions) == 0:
            logger.warning("AnalysisResult has no chunks, returning empty list")
            return []

        # Sort by position to handle potential non-monotonic ordering
        sort_indices = np.argsort(positions)
        positions_sorted = positions[sort_indices]

        for direction in directions:
            if direction not in result.timeseries:
                logger.debug(
                    "Direction '%s' not in timeseries, skipping", direction
                )
                continue

            ts = result.timeseries[direction]

            for emotion in BASE_EMOTIONS:
                values = np.array(getattr(ts, emotion), dtype=np.float64)

                # Sort values to match sorted positions
                values_sorted = values[sort_indices]

                # Resample using nearest-neighbor
                x_grid, y_resampled = resample_nearest(
                    positions_sorted, values_sorted, self.n_points
                )

                trajectories.append(
                    NormalizedTrajectory(
                        source=result.metadata.source,
                        main_pairing=result.world_builder.main_pairing,
                        direction=direction,
                        emotion=emotion,
                        x=x_grid.tolist(),
                        y=y_resampled.tolist(),
                        original_length=len(result.chunks),
                        n_points=self.n_points,
                    )
                )

        return trajectories

    def normalize_all(
        self, results: list[AnalysisResult]
    ) -> list[NormalizedTrajectory]:
        """Normalize multiple AnalysisResults into a flat list.

        Args:
            results: List of PCMFG analysis results.

        Returns:
            Flat list of NormalizedTrajectory objects from all results.
        """
        all_trajectories: list[NormalizedTrajectory] = []
        for result in results:
            all_trajectories.extend(self.normalize(result))
        return all_trajectories
