"""Phase 3: Synthesis module for PCMFG."""

from pcmfg.phase3.axis_mapper import AxisMapper
from pcmfg.phase3.synthesizer import (
    Synthesizer,
    build_emotion_timeseries,
    impute_missing_emotions,
)

__all__ = [
    "AxisMapper",
    "Synthesizer",
    "impute_missing_emotions",
    "build_emotion_timeseries",
]
