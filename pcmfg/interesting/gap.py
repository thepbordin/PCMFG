"""Gap Analysis — compute directional emotion imbalance at interesting points.

Calculates A→B minus B→A for each of the 9 emotions at detected peaks/discords,
revealing one-sided feelings and emotional asymmetry.
"""

import logging

from pcmfg.models.schemas import (
    BASE_EMOTIONS,
    AnalysisResult,
    GapAtTimestamp,
    GapValue,
)

logger = logging.getLogger(__name__)


def compute_gaps(
    result: AnalysisResult,
    interesting_indices: list[int],
    all_indices: bool = False,
) -> list[GapAtTimestamp]:
    """Compute per-emotion gaps at interesting timestamps.

    Args:
        result: Complete analysis result with timeseries data.
        interesting_indices: Indices to compute gaps at (from discords/segments).
        all_indices: If True, compute gaps at every timestamp instead.

    Returns:
        List of GapAtTimestamp with per-emotion gap values.
    """
    a_to_b = result.timeseries.get("A_to_B")
    b_to_a = result.timeseries.get("B_to_A")

    if a_to_b is None or b_to_a is None:
        logger.warning("Missing timeseries directions — cannot compute gaps")
        return []

    n_chunks = len(a_to_b.Joy)

    if all_indices:
        indices = list(range(n_chunks))
    else:
        indices = sorted(set(interesting_indices))

    results: list[GapAtTimestamp] = []

    for idx in indices:
        if idx < 0 or idx >= n_chunks:
            continue

        gaps: list[GapValue] = []
        max_gap_emotion = ""
        max_gap_value = 0.0

        for emotion in BASE_EMOTIONS:
            a_values = getattr(a_to_b, emotion, [])
            b_values = getattr(b_to_a, emotion, [])

            a_val = float(a_values[idx]) if idx < len(a_values) else 1.0
            b_val = float(b_values[idx]) if idx < len(b_values) else 1.0
            gap = a_val - b_val

            gaps.append(
                GapValue(emotion=emotion, a_to_b=a_val, b_to_a=b_val, gap=gap)
            )

            if abs(gap) > abs(max_gap_value):
                max_gap_emotion = emotion
                max_gap_value = gap

        position = 0.0
        chunk_id = idx
        if result.chunks and idx < len(result.chunks):
            position = result.chunks[idx].position
            chunk_id = result.chunks[idx].chunk_id
        elif n_chunks > 1:
            position = idx / (n_chunks - 1)

        results.append(
            GapAtTimestamp(
                index=idx,
                position=position,
                chunk_id=chunk_id,
                gaps=gaps,
                dominant_gap_emotion=max_gap_emotion,
                dominant_gap_value=max_gap_value,
            )
        )

    logger.info("Computed gaps at %d timestamps", len(results))
    return results
