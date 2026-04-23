"""Convert AnalysisResult to 18D numpy array for Matrix Profile computation.

Builds a (T, 18) array from the timeseries in AnalysisResult, where T is the
number of chunks. Column ordering matches build_dtw_dataset in dtw_clusterer.py:
[Joy_A2B, Joy_B2A, Trust_A2B, Trust_B2A, ..., Arousal_A2B, Arousal_B2A].
"""

import logging

import numpy as np
from numpy.typing import NDArray

from pcmfg.models.schemas import BASE_EMOTIONS, AnalysisResult

logger = logging.getLogger(__name__)


def result_to_18d(
    result: AnalysisResult,
) -> tuple[NDArray[np.float64], NDArray[np.float64], list[int]]:
    """Convert AnalysisResult timeseries to an 18D numpy array.

    Args:
        result: Complete analysis result with timeseries data.

    Returns:
        Tuple of:
        - T_18d: array of shape (T, 18) with interleaved A_to_B/B_to_A columns
        - positions: array of shape (T,) with narrative positions (0.0-1.0)
        - chunk_ids: list of chunk IDs matching each row
    """
    a_to_b = result.timeseries.get("A_to_B")
    b_to_a = result.timeseries.get("B_to_A")

    if a_to_b is None:
        raise ValueError("AnalysisResult missing 'A_to_B' timeseries")
    if b_to_a is None:
        raise ValueError("AnalysisResult missing 'B_to_A' timeseries")

    # Determine length from first emotion
    a_joy = a_to_b.Joy
    if not a_joy:
        raise ValueError("Timeseries is empty — no emotion data to analyze")
    n_chunks = len(a_joy)

    T_18d = np.ones((n_chunks, 18), dtype=np.float64)
    positions = np.zeros(n_chunks, dtype=np.float64)
    chunk_ids: list[int] = []

    for emotion_idx, emotion in enumerate(BASE_EMOTIONS):
        a_values = getattr(a_to_b, emotion, [])
        b_values = getattr(b_to_a, emotion, [])

        for t in range(n_chunks):
            col_a = emotion_idx * 2
            col_b = emotion_idx * 2 + 1
            if t < len(a_values):
                T_18d[t, col_a] = float(a_values[t])
            if t < len(b_values):
                T_18d[t, col_b] = float(b_values[t])

    # Build position and chunk_id arrays
    if result.chunks:
        for i, chunk in enumerate(result.chunks[:n_chunks]):
            positions[i] = chunk.position
            chunk_ids.append(chunk.chunk_id)
    else:
        # Fallback: uniform positions, sequential IDs
        for i in range(n_chunks):
            positions[i] = i / max(n_chunks - 1, 1)
            chunk_ids.append(i)

    return T_18d, positions, chunk_ids
