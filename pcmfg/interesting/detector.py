"""InterestingSectionDetector — orchestrates all detection components.

Computes the Matrix Profile once and feeds it to discord discovery,
FLUSS segmentation, motif discovery, and gap analysis. Returns a single
InterestingSectionReport with all results.
"""

import logging

import numpy as np
from numpy.typing import NDArray

from pcmfg.models.schemas import AnalysisResult, InterestingSectionReport

from .converter import result_to_18d
from .discord import discover_discords
from .gap import compute_gaps
from .motif import discover_motifs
from .segmentation import compute_segmentation

logger = logging.getLogger(__name__)

try:
    import stumpy

    STUMPY_AVAILABLE = True
except ImportError:
    STUMPY_AVAILABLE = False
    logger.warning(
        "stumpy not installed. Interesting section detection disabled. "
        "Install with: pip install stumpy"
    )


class InterestingSectionDetector:
    """Detect interesting narrative sections from emotional timeseries.

    Runs Matrix Profile-based analysis to find emotional climaxes,
    story act boundaries, recurring tropes, and directional imbalances.
    Operates on AnalysisResult — no LLM calls needed.

    Example::

        detector = InterestingSectionDetector(n_discords=5, n_regimes=3)
        report = detector.detect(analysis_result)
    """

    def __init__(
        self,
        window_size: int | None = None,
        n_discords: int = 5,
        n_regimes: int = 3,
        n_motifs: int = 3,
        compute_all_gaps: bool = False,
    ) -> None:
        if not STUMPY_AVAILABLE:
            raise ImportError(
                "stumpy is required for InterestingSectionDetector. "
                "Install with: pip install stumpy"
            )

        self.window_size = window_size
        self.n_discords = n_discords
        self.n_regimes = n_regimes
        self.n_motifs = n_motifs
        self.compute_all_gaps = compute_all_gaps

    def detect(self, result: AnalysisResult) -> InterestingSectionReport:
        """Run full interesting section detection pipeline.

        Args:
            result: Complete analysis result from the PCMFG pipeline.

        Returns:
            InterestingSectionReport with all detection results.
        """
        # Step 1: Convert to 18D array
        T_18d, positions, chunk_ids = result_to_18d(result)
        n_chunks = T_18d.shape[0]

        # Step 2: Compute window size
        m = self.window_size or max(5, n_chunks // 5)
        if n_chunks < 2 * m:
            m = max(3, n_chunks // 3)
            logger.warning(
                "Timeseries too short for window_size=%s, reduced to %d",
                self.window_size,
                m,
            )

        if n_chunks < 3:
            logger.warning(
                "Timeseries has only %d points — returning empty report", n_chunks
            )
            return InterestingSectionReport(
                source=result.metadata.source or "unknown",
                main_pairing=result.world_builder.main_pairing[:2],
                window_size=m,
                n_chunks=n_chunks,
            )

        # Step 3: Compute Matrix Profile
        # Aggregate 18D to 1D via L2 norm (Euclidean magnitude) per timestep.
        # This preserves the total emotional "intensity" while giving a 1D
        # signal compatible with stump, fluss, and motifs.
        T_1d = np.linalg.norm(T_18d, axis=1)
        mp = stumpy.stump(T_1d, m)
        mp_distances = mp[:, 0].astype(float)

        # Replace inf/nan with 0 for downstream processing
        mp_distances = _clean_distances(mp_distances)

        # Step 4: Discord Discovery
        discords = discover_discords(
            mp_distances, m, positions, chunk_ids, k=self.n_discords
        )

        # Step 5: FLUSS Segmentation (needs full MP array)
        segments = compute_segmentation(
            mp, m, positions, chunk_ids, n_regimes=self.n_regimes
        )

        # Step 6: Motif Discovery (uses 1D signal + 1D MP distances)
        motifs = discover_motifs(
            T_1d, mp_distances, m, positions, chunk_ids, k=self.n_motifs
        )

        # Step 7: Gap Analysis at interesting points
        interesting_indices = [d.index for d in discords] + [
            s.index for s in segments
        ]
        gaps = compute_gaps(
            result, interesting_indices, all_indices=self.compute_all_gaps
        )

        report = InterestingSectionReport(
            source=result.metadata.source or "unknown",
            main_pairing=result.world_builder.main_pairing[:2],
            window_size=m,
            n_chunks=n_chunks,
            discords=discords,
            segments=segments,
            motifs=motifs,
            gaps=gaps,
            matrix_profile_distances=mp_distances.tolist(),
        )

        logger.info(
            "Detection complete: %d discords, %d segments, %d motifs, %d gap points",
            len(discords),
            len(segments),
            len(motifs),
            len(gaps),
        )

        return report


def _clean_distances(distances: NDArray) -> NDArray:
    """Replace inf and nan values in Matrix Profile distances with 0."""
    import numpy as np

    cleaned = distances.copy()
    cleaned[np.isinf(cleaned)] = 0.0
    cleaned[np.isnan(cleaned)] = 0.0
    return cleaned
