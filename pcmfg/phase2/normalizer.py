"""Emotion Normalizer for PCMFG Phase 2.

Validates and normalizes emotion scores:
- Validates 1-5 score range
- Checks justification quality
- Aggregates bidirectional emotions
- Flags potential hallucinations
"""

import logging
from typing import NamedTuple

from pcmfg.models.schemas import (
    BASE_EMOTIONS,
    ChunkAnalysis,
    DirectedEmotion,
    DirectedEmotionScores,
)

logger = logging.getLogger(__name__)


class ValidationWarning(NamedTuple):
    """A validation warning for a chunk analysis."""

    chunk_id: int
    warning_type: str
    message: str


class EmotionNormalizer:
    """Validates and normalizes emotion scores from Phase 1.

    The normalizer performs quality checks:
    1. Validates all scores are in 1-5 range
    2. Checks justification quotes for high scores
    3. Warns about potential hallucinations (too many 5s)
    4. Aggregates bidirectional emotions for axis computation
    """

    def __init__(self, max_fives_allowed: int = 3) -> None:
        """Initialize normalizer.

        Args:
            max_fives_allowed: Maximum number of 5s allowed before flagging.
        """
        self.max_fives_allowed = max_fives_allowed

    def validate_scores(self, chunk: ChunkAnalysis) -> bool:
        """Validate that all emotion scores are in valid range.

        Args:
            chunk: Chunk analysis to validate.

        Returns:
            True if all scores are valid (1-5).
        """
        for emotion in chunk.directed_emotions:
            scores = emotion.scores
            for emotion_name in BASE_EMOTIONS:
                score = getattr(scores, emotion_name, None)
                if score is None or not isinstance(score, int) or score < 1 or score > 5:
                    return False
        return True

    def check_justification(self, chunk: ChunkAnalysis) -> list[ValidationWarning]:
        """Check justification quality for high-scoring emotions.

        Args:
            chunk: Chunk analysis to check.

        Returns:
            List of validation warnings.
        """
        warnings: list[ValidationWarning] = []

        for emotion in chunk.directed_emotions:
            # Count high scores (>= 3)
            high_scores = []
            for emotion_name in BASE_EMOTIONS:
                score = getattr(emotion.scores, emotion_name)
                if score >= 3:
                    high_scores.append(emotion_name)

            # Check if justification exists for high scores
            if high_scores and not emotion.justification_quote.strip():
                warnings.append(
                    ValidationWarning(
                        chunk_id=chunk.chunk_id,
                        warning_type="missing_justification",
                        message=(
                            f"High scores {high_scores} for {emotion.source} → {emotion.target} "
                            f"lack justification quote"
                        ),
                    )
                )

        return warnings

    def check_hallucination(self, chunk: ChunkAnalysis) -> list[ValidationWarning]:
        """Check for potential LLM hallucinations (too many extreme scores).

        Args:
            chunk: Chunk analysis to check.

        Returns:
            List of validation warnings.
        """
        warnings: list[ValidationWarning] = []

        for emotion in chunk.directed_emotions:
            five_count = sum(
                1 for e in BASE_EMOTIONS if getattr(emotion.scores, e) == 5
            )

            if five_count > self.max_fives_allowed:
                warnings.append(
                    ValidationWarning(
                        chunk_id=chunk.chunk_id,
                        warning_type="potential_hallucination",
                        message=(
                            f"{emotion.source} → {emotion.target} has {five_count} "
                            f"scores of 5 (max allowed: {self.max_fives_allowed})"
                        ),
                    )
                )

        return warnings

    def aggregate_bidirectional(
        self, emotions: list[DirectedEmotion]
    ) -> DirectedEmotionScores:
        """Aggregate bidirectional emotions into a single score set.

        Averages A→B and B→A scores for each emotion.

        Args:
            emotions: List of directed emotions (typically 2 directions).

        Returns:
            Aggregated emotion scores (averaged).
        """
        if not emotions:
            return DirectedEmotionScores()

        # Sum up all scores
        totals: dict[str, float] = {e: 0.0 for e in BASE_EMOTIONS}
        counts: dict[str, int] = {e: 0 for e in BASE_EMOTIONS}

        for emotion in emotions:
            for emotion_name in BASE_EMOTIONS:
                score = getattr(emotion.scores, emotion_name)
                totals[emotion_name] += score
                counts[emotion_name] += 1

        # Calculate averages
        averaged = {}
        for emotion_name in BASE_EMOTIONS:
            if counts[emotion_name] > 0:
                averaged[emotion_name] = round(totals[emotion_name] / counts[emotion_name])
            else:
                averaged[emotion_name] = 1

        return DirectedEmotionScores(**averaged)

    def normalize(self, chunk: ChunkAnalysis) -> ChunkAnalysis:
        """Normalize a chunk analysis (validation pass).

        Performs all quality checks and logs warnings.

        Args:
            chunk: Chunk analysis to normalize.

        Returns:
            The same chunk (validation only, no modification).
        """
        # Validate scores
        if not self.validate_scores(chunk):
            logger.warning(
                f"Chunk {chunk.chunk_id}: Invalid emotion scores detected"
            )

        # Check justifications
        justification_warnings = self.check_justification(chunk)
        for warning in justification_warnings:
            logger.warning(f"Chunk {chunk.chunk_id}: {warning.message}")

        # Check for hallucinations
        hallucination_warnings = self.check_hallucination(chunk)
        for warning in hallucination_warnings:
            logger.warning(f"Chunk {chunk.chunk_id}: {warning.message}")

        return chunk

    def normalize_all(self, chunks: list[ChunkAnalysis]) -> list[ChunkAnalysis]:
        """Normalize all chunks.

        Args:
            chunks: List of chunk analyses.

        Returns:
            List of normalized chunks.
        """
        return [self.normalize(chunk) for chunk in chunks]

    def get_aggregated_scores(self, chunk: ChunkAnalysis) -> DirectedEmotionScores:
        """Get aggregated emotion scores for a chunk.

        Args:
            chunk: Chunk analysis.

        Returns:
            Aggregated (averaged) emotion scores.
        """
        return self.aggregate_bidirectional(chunk.directed_emotions)
