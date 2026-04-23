"""Interesting Section Detection for PCMFG.

Uses Matrix Profile algorithms (via stumpy) to detect the most interesting
narrative sections from 18D emotion timeseries:
- Discord Discovery: emotional climaxes and unique moments
- FLUSS Segmentation: story act boundaries
- Motif Discovery: recurring emotional patterns (tropes)
- Gap Analysis: directional emotion imbalance

Example::

    from pcmfg.interesting import InterestingSectionDetector

    detector = InterestingSectionDetector(n_discords=5, n_regimes=3)
    report = detector.detect(analysis_result)
"""

from pcmfg.interesting.converter import result_to_18d
from pcmfg.interesting.detector import InterestingSectionDetector
from pcmfg.interesting.discord import discover_discords
from pcmfg.interesting.gap import compute_gaps
from pcmfg.interesting.motif import discover_motifs
from pcmfg.interesting.plotter import plot_interesting_report
from pcmfg.interesting.segmentation import compute_segmentation

__all__ = [
    "InterestingSectionDetector",
    "result_to_18d",
    "discover_discords",
    "compute_segmentation",
    "discover_motifs",
    "compute_gaps",
    "plot_interesting_report",
]
