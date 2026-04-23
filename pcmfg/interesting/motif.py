"""Motif Discovery — find recurring emotional patterns (tropes).

Identifies Top-K motif pairs: 18D subsequences with the lowest pairwise
distance, representing recurring narrative patterns like "Argument & Make-up".
"""

import logging

import numpy as np
from numpy.typing import NDArray

from pcmfg.models.schemas import MotifPair

logger = logging.getLogger(__name__)


def discover_motifs(
    T_1d: NDArray[np.float64],
    mp_distances: NDArray[np.float64],
    window_size: int,
    positions: NDArray[np.float64],
    chunk_ids: list[int],
    k: int = 3,
    max_distance: float | None = None,
) -> list[MotifPair]:
    """Find top-K recurring motif pairs from the aggregated timeseries.

    Uses stumpy.motifs on the 1D aggregated signal and its Matrix Profile
    to find the most similar subsequences.

    Args:
        T_1d: 1D aggregated timeseries (L2 norm of 18D per timestep).
        mp_distances: 1D array of Matrix Profile distances.
        window_size: Window size used for MP computation.
        positions: Narrative positions for each timeseries index.
        chunk_ids: Chunk IDs for each timeseries index.
        k: Number of top motif groups to return.
        max_distance: Maximum distance threshold. None = auto.

    Returns:
        List of MotifPair sorted by distance ascending.
    """
    try:
        import stumpy
    except ImportError as e:
        raise ImportError(
            "stumpy is required for motif discovery. Install with: pip install stumpy"
        ) from e

    n_mp = len(mp_distances)
    if n_mp == 0:
        logger.warning("Empty Matrix Profile — no motifs to discover")
        return []

    distance_idx_pairs = stumpy.motifs(
        T_1d,
        mp_distances,
        max_distance=max_distance,
        max_matches=k + 1,
    )

    if distance_idx_pairs is None or len(distance_idx_pairs) == 0:
        logger.info("No motif groups found")
        return []

    # distance_idx_pairs is a list of (distance, index) arrays
    # Each element is one motif group: distances and their matching indices
    results: list[MotifPair] = []

    for group in distance_idx_pairs:
        if len(results) >= k:
            break

        # group is a 2D array: each row is [distance, index]
        if len(group) < 2:
            continue

        indices = [int(row[1]) for row in group]

        # Take the first non-overlapping pair
        for i_a in range(len(indices)):
            found = False
            for i_b in range(i_a + 1, len(indices)):
                idx_a = indices[i_a]
                idx_b = indices[i_b]

                if abs(idx_a - idx_b) < window_size:
                    continue

                pos_a = float(positions[idx_a]) if idx_a < len(positions) else 0.0
                pos_b = float(positions[idx_b]) if idx_b < len(positions) else 0.0
                cid_a = chunk_ids[idx_a] if idx_a < len(chunk_ids) else idx_a
                cid_b = chunk_ids[idx_b] if idx_b < len(chunk_ids) else idx_b

                # Distance from the motif group
                distance = float(group[i_b][0])

                results.append(
                    MotifPair(
                        index_a=idx_a,
                        index_b=idx_b,
                        position_a=pos_a,
                        position_b=pos_b,
                        chunk_id_a=cid_a,
                        chunk_id_b=cid_b,
                        distance=distance,
                    )
                )
                found = True
                break
            if found:
                break

    results.sort(key=lambda m: m.distance)
    logger.info("Discovered %d motif pairs (requested %d)", len(results), k)
    return results
