"""FLUSS Semantic Segmentation — identify story act boundaries.

Uses the Matrix Profile to find regime change points where the emotional
pattern permanently shifts, marking structural boundaries like
Intro → Conflict → Resolution.
"""

import logging

import numpy as np
from numpy.typing import NDArray

from pcmfg.models.schemas import SegmentationResult

logger = logging.getLogger(__name__)

_REGIME_LABELS_3 = ["Intro", "Conflict", "Resolution"]
_REGIME_LABELS_4 = ["Act 1: Setup", "Act 2A: Rising", "Act 2B: Crisis", "Act 3: Resolution"]
_REGIME_LABELS_5 = [
    "Act 1: Setup",
    "Act 2A: Rising",
    "Act 2B: Midpoint",
    "Act 2C: Crisis",
    "Act 3: Resolution",
]


def compute_segmentation(
    mp: NDArray[np.float64],
    window_size: int,
    positions: NDArray[np.float64],
    chunk_ids: list[int],
    n_regimes: int = 3,
) -> list[SegmentationResult]:
    """Compute semantic segmentation using FLUSS on the Matrix Profile.

    Args:
        mp: Full Matrix Profile array from stumpy.stump (columns: distance, idx, left, right).
        window_size: Window size used for MP computation.
        positions: Narrative positions for each timeseries index.
        chunk_ids: Chunk IDs for each timeseries index.
        n_regimes: Number of structural segments to detect.

    Returns:
        List of SegmentationResult marking act boundaries.
    """
    try:
        import stumpy
    except ImportError as e:
        raise ImportError(
            "stumpy is required for segmentation. Install with: pip install stumpy"
        ) from e

    n = len(mp)
    if n == 0:
        logger.warning("Empty Matrix Profile — no segmentation possible")
        return []

    # fluss expects the nearest neighbor index column (column 1) as a 1D array
    mp_indices = mp[:, 1].astype(np.int64)

    L = window_size
    result = stumpy.fluss(mp_indices, L=L, n_regimes=n_regimes - 1)

    # fluss returns (cac, change_points) tuple
    if isinstance(result, tuple):
        _, change_points = result
    else:
        change_points = result

    if len(change_points) == 0:
        logger.info("FLUSS found no change points with n_regimes=%d", n_regimes)
        return []

    # Select regime labels
    if n_regimes <= 3:
        labels = _REGIME_LABELS_3
    elif n_regimes == 4:
        labels = _REGIME_LABELS_4
    elif n_regimes >= 5:
        labels = _REGIME_LABELS_5
    else:
        labels = [f"Segment {i+1}" for i in range(n_regimes)]

    # change_points is a 1D array of indices
    cp_list = sorted(int(cp) for cp in np.array(change_points).flatten())

    results: list[SegmentationResult] = []
    for i, cp in enumerate(cp_list):
        idx = min(cp, len(positions) - 1)
        regime_idx = min(i + 1, len(labels) - 1)
        results.append(
            SegmentationResult(
                index=cp,
                position=float(positions[idx]),
                chunk_id=chunk_ids[idx] if idx < len(chunk_ids) else cp,
                regime_label=labels[regime_idx],
            )
        )

    logger.info(
        "Found %d change points for %d regimes", len(results), n_regimes
    )
    return results
