"""Discord Discovery — find the most unique emotional moments.

Uses Matrix Profile distances to identify top-K discords: subsequences
with the highest distance to their nearest neighbor, representing the
narrative's emotional climaxes or plot twists.
"""

import logging
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from pcmfg.models.schemas import DiscordResult

logger = logging.getLogger(__name__)


def discover_discords(
    mp_distances: NDArray[np.float64],
    window_size: int,
    positions: NDArray[np.float64],
    chunk_ids: list[int],
    k: int = 5,
) -> list[DiscordResult]:
    """Find top-K discords from Matrix Profile distances.

    Args:
        mp_distances: 1D array of Matrix Profile distances (length T - m + 1).
        window_size: The window size used for MP computation.
        positions: Narrative positions for each timeseries index.
        chunk_ids: Chunk IDs for each timeseries index.
        k: Number of top discords to return.

    Returns:
        List of DiscordResult sorted by distance descending.
    """
    n = len(mp_distances)
    if n == 0:
        logger.warning("Empty Matrix Profile — no discords to discover")
        return []

    # Sort indices by distance descending
    sorted_indices = np.argsort(mp_distances)[::-1]

    results: list[DiscordResult] = []
    used_indices: set[int] = set()

    for idx in sorted_indices:
        if len(results) >= k:
            break

        idx_int = int(idx)

        # Skip overlapping windows — ensure no overlap with already-selected discords
        if any(abs(idx_int - u) < window_size for u in used_indices):
            continue

        used_indices.add(idx_int)

        # Map MP index to original timeseries index (MP index = start of window)
        position = float(positions[idx_int]) if idx_int < len(positions) else 0.0
        cid = chunk_ids[idx_int] if idx_int < len(chunk_ids) else idx_int

        results.append(
            DiscordResult(
                index=idx_int,
                position=position,
                chunk_id=cid,
                distance=float(mp_distances[idx_int]),
                window_size=window_size,
            )
        )

    logger.info("Discovered %d discords (requested %d)", len(results), k)
    return results
