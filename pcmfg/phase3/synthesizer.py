"""Synthesizer for PCMFG Phase 3.

Deterministic Python processing (no LLM) that:
- Forward fills missing emotion data
- Builds raw emotion time-series for both directions (A→B and B→A)
- Generates the final analysis output
"""

import logging
from typing import TypedDict

from pcmfg.models.schemas import (
    BASE_EMOTIONS,
    AnalysisMetadata,
    AnalysisResult,
    ChunkAnalysis,
    DirectedEmotion,
    DirectedEmotionScores,
    EmotionTimeSeries,
    WorldBuilderOutput,
)

logger = logging.getLogger(__name__)


class EmotionTimeSeriesDict(TypedDict):
    """TypedDict for building emotion time-series."""

    Joy: list[float]
    Trust: list[float]
    Fear: list[float]
    Surprise: list[float]
    Sadness: list[float]
    Disgust: list[float]
    Anger: list[float]
    Anticipation: list[float]
    Arousal: list[float]


def create_baseline_scores() -> DirectedEmotionScores:
    """Create baseline emotion scores (all 1s)."""
    return DirectedEmotionScores()


def impute_missing_emotions(
    chunks: list[ChunkAnalysis],
    main_pairing: list[str],
) -> list[ChunkAnalysis]:
    """Forward fill missing emotion directions.

    Because Agent 2 only scores what is on the page, the reverse direction
    (B → A) will often be null when B is absent from a scene.

    Solution: Use forward fill to carry forward the last known emotional state.

    Args:
        chunks: List of chunk analyses from Agent 2.
        main_pairing: The two main characters [A, B].

    Returns:
        List of chunks with forward-filled missing directions.
    """
    if len(main_pairing) < 2:
        logger.warning("Main pairing has fewer than 2 characters, skipping imputation")
        return chunks

    char_a, char_b = main_pairing[0], main_pairing[1]

    # Initialize last known states with baseline (all 1s)
    baseline = create_baseline_scores()
    last_known: dict[str, DirectedEmotionScores] = {
        f"{char_a}->{char_b}": baseline,
        f"{char_b}->{char_a}": baseline,
    }

    imputed_chunks: list[ChunkAnalysis] = []

    for chunk in chunks:
        # Track which directions are present in this chunk
        present_directions: set[str] = set()

        # Filter and update last known states from current chunk
        # Only consider emotions between main pairing characters
        filtered_emotions: list[DirectedEmotion] = []
        for emotion in chunk.directed_emotions:
            # Check if this emotion is between main pairing
            is_a_to_b = (
                emotion.source == char_a and emotion.target == char_b
            ) or (
                # Also check case-insensitive and partial matches
                char_a.lower() in emotion.source.lower()
                and char_b.lower() in emotion.target.lower()
            )
            is_b_to_a = (
                emotion.source == char_b and emotion.target == char_a
            ) or (
                char_b.lower() in emotion.source.lower()
                and char_a.lower() in emotion.target.lower()
            )

            if is_a_to_b:
                key = f"{char_a}->{char_b}"
                present_directions.add(key)
                last_known[key] = emotion.scores
                # Create normalized emotion
                filtered_emotions.append(
                    DirectedEmotion(
                        source=char_a,
                        target=char_b,
                        scores=emotion.scores,
                        justification_quote=emotion.justification_quote,
                    )
                )
            elif is_b_to_a:
                key = f"{char_b}->{char_a}"
                present_directions.add(key)
                last_known[key] = emotion.scores
                # Create normalized emotion
                filtered_emotions.append(
                    DirectedEmotion(
                        source=char_b,
                        target=char_a,
                        scores=emotion.scores,
                        justification_quote=emotion.justification_quote,
                    )
                )

        # Build new directed_emotions with forward-filled missing directions
        new_directed_emotions: list[DirectedEmotion] = filtered_emotions

        # Check and impute each direction
        for direction in [f"{char_a}->{char_b}", f"{char_b}->{char_a}"]:
            if direction not in present_directions:
                # Create imputed emotion from last known state
                source, target = direction.split("->")
                imputed = DirectedEmotion(
                    source=source,
                    target=target,
                    scores=last_known[direction],
                    justification_quote="[FORWARD FILLED - character absent from scene]",
                )
                new_directed_emotions.append(imputed)
                logger.debug(f"Chunk {chunk.chunk_id}: Forward filled {direction}")

        # Create new chunk with imputed data
        imputed_chunk = ChunkAnalysis(
            chunk_id=chunk.chunk_id,
            position=chunk.position,
            chunk_main_pov=chunk.chunk_main_pov,
            characters_present=chunk.characters_present,
            directed_emotions=new_directed_emotions,
            scene_summary=chunk.scene_summary,
        )
        imputed_chunks.append(imputed_chunk)

    logger.info(f"Forward fill complete for {len(imputed_chunks)} chunks")
    return imputed_chunks


def build_emotion_timeseries(
    chunks: list[ChunkAnalysis],
    source: str,
    target: str,
) -> EmotionTimeSeriesDict:
    """Build time-series for Source→Target emotions across all chunks.

    Args:
        chunks: List of chunk analyses (should be imputed).
        source: Source character name.
        target: Target character name.

    Returns:
        Dictionary with 9 emotion time-series arrays.
    """
    # Initialize time-series arrays
    timeseries: EmotionTimeSeriesDict = {emotion: [] for emotion in BASE_EMOTIONS}

    for chunk in chunks:
        # Find the directed emotion for this pair
        emotion = None
        for de in chunk.directed_emotions:
            if de.source == source and de.target == target:
                emotion = de
                break

        if emotion:
            for e in BASE_EMOTIONS:
                score = getattr(emotion.scores, e)
                timeseries[e].append(float(score))
        else:
            # Fallback to baseline (should not happen after forward fill)
            for e in BASE_EMOTIONS:
                timeseries[e].append(1.0)

    return timeseries


class Synthesizer:
    """Phase 3: Deterministic Python synthesis (no LLM).

    Processes extracted emotion data to produce:
    - Forward-filled emotion data for missing directions
    - Raw emotion time-series for both relationship directions
    """

    def synthesize(
        self,
        chunks: list[ChunkAnalysis],
        world: WorldBuilderOutput,
        metadata: AnalysisMetadata | None = None,
    ) -> AnalysisResult:
        """Synthesize raw emotion time-series from extracted chunks.

        Steps:
        1. Forward fill missing emotion directions
        2. Build time-series for A→B
        3. Build time-series for B→A

        Args:
            chunks: List of chunk analyses from Agent 2.
            world: World builder output with main pairing.
            metadata: Optional metadata for the analysis.

        Returns:
            AnalysisResult with timeseries data.
        """
        logger.info(f"Synthesizing {len(chunks)} chunks")

        if len(world.main_pairing) < 2:
            logger.error("Main pairing has fewer than 2 characters")
            raise ValueError("Main pairing must have exactly 2 characters")

        char_a, char_b = world.main_pairing[0], world.main_pairing[1]

        # Step 1: Forward fill missing directions
        imputed_chunks = impute_missing_emotions(chunks, world.main_pairing)

        # Step 2 & 3: Build time-series for both directions
        a_to_b_dict = build_emotion_timeseries(imputed_chunks, char_a, char_b)
        b_to_a_dict = build_emotion_timeseries(imputed_chunks, char_b, char_a)

        # Convert to EmotionTimeSeries models
        a_to_b = EmotionTimeSeries(**a_to_b_dict)
        b_to_a = EmotionTimeSeries(**b_to_a_dict)

        logger.info(
            f"Built time-series: A→B has {len(a_to_b.Joy)} data points, "
            f"B→A has {len(b_to_a.Joy)} data points"
        )

        # Build result
        return AnalysisResult(
            metadata=metadata or AnalysisMetadata(total_chunks=len(imputed_chunks)),
            world_builder=world,
            chunks=imputed_chunks,
            timeseries={
                "A_to_B": a_to_b,
                "B_to_A": b_to_a,
            },
        )

    def synthesize_from_result(
        self,
        result: AnalysisResult,
    ) -> AnalysisResult:
        """Re-synthesize an existing result (e.g., to add forward fill).

        Args:
            result: Existing analysis result.

        Returns:
            New AnalysisResult with synthesized timeseries.
        """
        return self.synthesize(
            chunks=result.chunks,
            world=result.world_builder,
            metadata=result.metadata,
        )
