"""Feature extraction for emotional time-series clustering.

This module transforms PCMFG analysis results into feature vectors suitable
for clustering. It supports multiple feature extraction strategies:

1. Raw emotion vectors (9-dim per direction)
2. Delta vectors (emotion change between chunks)
3. Statistical aggregations (mean, std, min, max, range)
4. Combined features (raw + statistical)
"""

from enum import Enum
from typing import Any

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, Field

from pcmfg.models.schemas import (
    BASE_EMOTIONS,
    AnalysisResult,
    ChunkAnalysis,
    DirectedEmotion,
    EmotionTimeSeries,
)


class FeatureType(str, Enum):
    """Types of features that can be extracted."""

    RAW = "raw"  # Raw 9 emotion scores (or 18 for both directions)
    DELTA = "delta"  # Change in emotions between consecutive chunks
    STATISTICAL = "statistical"  # Statistical aggregations
    COMBINED = "combined"  # Raw + statistical features
    WINDOWED = "windowed"  # Rolling window features


class SceneFeatures(BaseModel):
    """Features extracted from a single scene/chunk."""

    chunk_id: int = Field(description="ID of the chunk")
    position: float = Field(description="Position in narrative (0.0-1.0)")
    feature_vector: list[float] = Field(description="Feature vector")
    feature_names: list[str] = Field(description="Names of features")
    pov: str = Field(default="", description="POV character")
    characters_present: list[str] = Field(default_factory=list)
    scene_summary: str = Field(default="")


class ExtractedFeatures(BaseModel):
    """Complete feature set extracted from an analysis result."""

    source: str = Field(description="Source file/text identifier")
    feature_type: FeatureType = Field(description="Type of features extracted")
    main_pairing: list[str] = Field(description="Main character pairing")
    features: list[SceneFeatures] = Field(description="Features per chunk")
    feature_matrix: list[list[float]] = Field(description="2D matrix of features")
    feature_names: list[str] = Field(description="Names of all features")

    def to_numpy(self) -> NDArray[np.float64]:
        """Convert feature matrix to numpy array."""
        return np.array(self.feature_matrix, dtype=np.float64)


class FeatureExtractor:
    """Extract features from PCMFG analysis results for clustering.

    Supports multiple feature extraction strategies to capture different
    aspects of emotional dynamics:

    - RAW: Direct emotion scores per chunk (9 dims per direction)
    - DELTA: Changes between consecutive chunks
    - STATISTICAL: Aggregated statistics (mean, std, min, max)
    - COMBINED: Raw emotions + statistical features
    - WINDOWED: Rolling window statistics
    """

    def __init__(
        self,
        feature_type: FeatureType = FeatureType.RAW,
        include_both_directions: bool = True,
        window_size: int = 3,
    ) -> None:
        """Initialize the feature extractor.

        Args:
            feature_type: Type of features to extract.
            include_both_directions: Include A→B and B→A (True) or just A→B (False).
            window_size: Window size for WINDOWED feature type.
        """
        self.feature_type = feature_type
        self.include_both_directions = include_both_directions
        self.window_size = window_size

    def extract(self, result: AnalysisResult) -> ExtractedFeatures:
        """Extract features from an analysis result.

        Args:
            result: PCMFG analysis result.

        Returns:
            ExtractedFeatures with feature vectors for each chunk.
        """
        if self.feature_type == FeatureType.RAW:
            features, names = self._extract_raw_features(result)
        elif self.feature_type == FeatureType.DELTA:
            features, names = self._extract_delta_features(result)
        elif self.feature_type == FeatureType.STATISTICAL:
            features, names = self._extract_statistical_features(result)
        elif self.feature_type == FeatureType.COMBINED:
            features, names = self._extract_combined_features(result)
        elif self.feature_type == FeatureType.WINDOWED:
            features, names = self._extract_windowed_features(result)
        else:
            raise ValueError(f"Unknown feature type: {self.feature_type}")

        # Build SceneFeatures objects
        scene_features = []
        for i, chunk in enumerate(result.chunks):
            scene_features.append(
                SceneFeatures(
                    chunk_id=chunk.chunk_id,
                    position=chunk.position,
                    feature_vector=features[i].tolist(),
                    feature_names=names,
                    pov=chunk.chunk_main_pov,
                    characters_present=chunk.characters_present,
                    scene_summary=chunk.scene_summary,
                )
            )

        return ExtractedFeatures(
            source=result.metadata.source,
            feature_type=self.feature_type,
            main_pairing=result.world_builder.main_pairing,
            features=scene_features,
            feature_matrix=features.tolist(),
            feature_names=names,
        )

    def _extract_raw_features(
        self, result: AnalysisResult
    ) -> tuple[NDArray[np.float64], list[str]]:
        """Extract raw emotion scores as features.

        Returns:
            Tuple of (feature matrix, feature names).
        """
        chunks = result.chunks
        n_chunks = len(chunks)

        if self.include_both_directions:
            n_features = 18  # 9 emotions × 2 directions
            feature_names = []
            for direction in ["A_to_B", "B_to_A"]:
                for emotion in BASE_EMOTIONS:
                    feature_names.append(f"{direction}_{emotion}")
        else:
            n_features = 9
            feature_names = [f"A_to_B_{e}" for e in BASE_EMOTIONS]

        features = np.zeros((n_chunks, n_features))

        for i, chunk in enumerate(chunks):
            # Get emotion vectors for each direction
            a_to_b = self._get_direction_scores(chunk, result.world_builder.main_pairing)
            b_to_a = self._get_direction_scores(
                chunk,
                result.world_builder.main_pairing,
                reverse=True,
            )

            if self.include_both_directions:
                features[i] = np.concatenate([a_to_b, b_to_a])
            else:
                features[i] = a_to_b

        return features, feature_names

    def _extract_delta_features(
        self, result: AnalysisResult
    ) -> tuple[NDArray[np.float64], list[str]]:
        """Extract emotion changes (deltas) as features.

        First chunk uses raw values, subsequent chunks use differences.

        Returns:
            Tuple of (feature matrix, feature names).
        """
        raw_features, names = self._extract_raw_features(result)

        # Compute deltas (first row stays the same)
        deltas = np.zeros_like(raw_features)
        deltas[0] = raw_features[0]
        deltas[1:] = np.diff(raw_features, axis=0)

        # Update feature names
        delta_names = [f"delta_{n}" for n in names]

        return deltas, delta_names

    def _extract_statistical_features(
        self, result: AnalysisResult
    ) -> tuple[NDArray[np.float64], list[str]]:
        """Extract statistical aggregations of emotion time-series.

        Computes mean, std, min, max, range for each emotion direction.

        Returns:
            Tuple of (feature matrix with one row, feature names).
        """
        timeseries = result.timeseries

        if self.include_both_directions:
            directions = ["A_to_B", "B_to_A"]
        else:
            directions = ["A_to_B"]

        stats = []
        feature_names = []

        for direction in directions:
            if direction not in timeseries:
                continue
            ts = timeseries[direction]
            for emotion in BASE_EMOTIONS:
                values = getattr(ts, emotion)
                arr = np.array(values)

                # Compute statistics
                stats.extend(
                    [
                        np.mean(arr),
                        np.std(arr),
                        np.min(arr),
                        np.max(arr),
                        np.max(arr) - np.min(arr),  # range
                    ]
                )

                # Add feature names
                stat_names = ["mean", "std", "min", "max", "range"]
                for stat in stat_names:
                    feature_names.append(f"{direction}_{emotion}_{stat}")

        # Return as single-row matrix
        features = np.array([stats])

        return features, feature_names

    def _extract_combined_features(
        self, result: AnalysisResult
    ) -> tuple[NDArray[np.float64], list[str]]:
        """Extract raw + per-chunk statistical features.

        Returns:
            Tuple of (feature matrix, feature names).
        """
        chunks = result.chunks
        n_chunks = len(chunks)

        # Build cumulative statistics
        if self.include_both_directions:
            directions = ["A_to_B", "B_to_A"]
            n_base = 18
        else:
            directions = ["A_to_B"]
            n_base = 9

        # Additional features: cumulative mean and trend
        n_features = n_base + 2 * len(directions) * len(BASE_EMOTIONS)

        feature_names = []
        for direction in directions:
            for emotion in BASE_EMOTIONS:
                feature_names.append(f"{direction}_{emotion}")
        for direction in directions:
            for emotion in BASE_EMOTIONS:
                feature_names.append(f"{direction}_{emotion}_cummean")
        for direction in directions:
            for emotion in BASE_EMOTIONS:
                feature_names.append(f"{direction}_{emotion}_trend")

        features = np.zeros((n_chunks, n_features))

        # Get time-series data
        ts_data: dict[str, dict[str, list[float]]] = {}
        for direction in directions:
            if direction in result.timeseries:
                ts = result.timeseries[direction]
                ts_data[direction] = {e: getattr(ts, e) for e in BASE_EMOTIONS}
            else:
                ts_data[direction] = {e: [1.0] * n_chunks for e in BASE_EMOTIONS}

        for i, chunk in enumerate(chunks):
            # Raw features
            a_to_b = self._get_direction_scores(chunk, result.world_builder.main_pairing)
            b_to_a = self._get_direction_scores(
                chunk, result.world_builder.main_pairing, reverse=True
            )

            if self.include_both_directions:
                raw = np.concatenate([a_to_b, b_to_a])
            else:
                raw = a_to_b

            # Cumulative mean features
            cummeans = []
            for direction in directions:
                for emotion in BASE_EMOTIONS:
                    values = ts_data[direction][emotion][: i + 1]
                    cummeans.append(np.mean(values))

            # Trend features (slope of last window_size points)
            trends = []
            for direction in directions:
                for emotion in BASE_EMOTIONS:
                    values = ts_data[direction][emotion][: i + 1]
                    if len(values) >= 2:
                        # Simple linear trend
                        x = np.arange(len(values))
                        trend = np.polyfit(x, values, 1)[0]
                    else:
                        trend = 0.0
                    trends.append(trend)

            features[i] = np.concatenate([raw, cummeans, trends])

        return features, feature_names

    def _extract_windowed_features(
        self, result: AnalysisResult
    ) -> tuple[NDArray[np.float64], list[str]]:
        """Extract rolling window statistics.

        For each chunk, computes mean and std over the previous window_size chunks.

        Returns:
            Tuple of (feature matrix, feature names).
        """
        chunks = result.chunks
        n_chunks = len(chunks)

        if self.include_both_directions:
            directions = ["A_to_B", "B_to_A"]
        else:
            directions = ["A_to_B"]

        # Features: window mean and std for each emotion direction
        n_features = 2 * len(directions) * len(BASE_EMOTIONS)

        feature_names = []
        for direction in directions:
            for emotion in BASE_EMOTIONS:
                feature_names.append(f"{direction}_{emotion}_win_mean")
        for direction in directions:
            for emotion in BASE_EMOTIONS:
                feature_names.append(f"{direction}_{emotion}_win_std")

        features = np.zeros((n_chunks, n_features))

        # Get time-series data
        ts_data: dict[str, dict[str, list[float]]] = {}
        for direction in directions:
            if direction in result.timeseries:
                ts = result.timeseries[direction]
                ts_data[direction] = {e: getattr(ts, e) for e in BASE_EMOTIONS}
            else:
                ts_data[direction] = {e: [1.0] * n_chunks for e in BASE_EMOTIONS}

        for i in range(n_chunks):
            start = max(0, i - self.window_size + 1)
            window = slice(start, i + 1)

            means = []
            stds = []
            for direction in directions:
                for emotion in BASE_EMOTIONS:
                    values = ts_data[direction][emotion][window]
                    means.append(np.mean(values))
                    stds.append(np.std(values) if len(values) > 1 else 0.0)

            features[i] = means + stds

        return features, feature_names

    def _get_direction_scores(
        self,
        chunk: ChunkAnalysis,
        main_pairing: list[str],
        reverse: bool = False,
    ) -> NDArray[np.float64]:
        """Get emotion scores for a specific direction from a chunk.

        Args:
            chunk: Chunk to extract from.
            main_pairing: [Character A, Character B].
            reverse: If False, get A→B; if True, get B→A.

        Returns:
            Array of 9 emotion scores.
        """
        if reverse:
            source = main_pairing[1]
            target = main_pairing[0]
        else:
            source = main_pairing[0]
            target = main_pairing[1]

        # Find matching directed emotion
        for emotion in chunk.directed_emotions:
            if emotion.source == source and emotion.target == target:
                return np.array([emotion.scores.__dict__[e] for e in BASE_EMOTIONS])

        # Default to all 1s if not found
        return np.ones(len(BASE_EMOTIONS))

    def extract_from_chunks(
        self, chunks: list[ChunkAnalysis], main_pairing: list[str]
    ) -> tuple[NDArray[np.float64], list[str]]:
        """Extract raw features directly from chunks without full AnalysisResult.

        Useful for incremental feature extraction.

        Args:
            chunks: List of chunk analyses.
            main_pairing: [Character A, Character B].

        Returns:
            Tuple of (feature matrix, feature names).
        """
        n_chunks = len(chunks)

        if self.include_both_directions:
            n_features = 18
            feature_names = []
            for direction in ["A_to_B", "B_to_A"]:
                for emotion in BASE_EMOTIONS:
                    feature_names.append(f"{direction}_{emotion}")
        else:
            n_features = 9
            feature_names = [f"A_to_B_{e}" for e in BASE_EMOTIONS]

        features = np.zeros((n_chunks, n_features))

        for i, chunk in enumerate(chunks):
            a_to_b = self._get_direction_scores(chunk, main_pairing)
            b_to_a = self._get_direction_scores(chunk, main_pairing, reverse=True)

            if self.include_both_directions:
                features[i] = np.concatenate([a_to_b, b_to_b])
            else:
                features[i] = a_to_b

        return features, feature_names
