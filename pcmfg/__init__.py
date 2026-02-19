"""PCMFG - Please Care My Feeling Graph.

A computational romance narrative mining system that extracts and visualizes
emotional trajectories from romantic literature.
"""

from pcmfg.analyzer import PCMFGAnalyzer, analyze
from pcmfg.config import Config, load_config
from pcmfg.models.schemas import AnalysisResult

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "PCMFGAnalyzer",
    "analyze",
    "Config",
    "load_config",
    "AnalysisResult",
]
