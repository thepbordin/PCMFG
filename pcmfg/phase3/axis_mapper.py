"""Axis Mapper for PCMFG Phase 3.

Maps base emotion scores to four romance axes:
- Intimacy: (Trust + Joy) / 2
- Passion: (Arousal + Anticipation + Joy) / 3
- Hostility: (Anger + Disgust + Sadness) / 3
- Anxiety: (Fear + Surprise + Sadness) / 3
"""

from pcmfg.models.schemas import (
    AxesTimeSeries,
    AxisValues,
    ChunkAnalysis,
    DirectedEmotionScores,
)


class AxisMapper:
    """Maps base emotions to romance axes.

    The four romance axes are:
    - Intimacy: Emotional closeness, trust, vulnerability
    - Passion: Romantic desire, excitement, intensity
    - Hostility: Anger, resentment, conflict
    - Anxiety: Fear, uncertainty, tension
    """

    def compute_axes(self, emotions: DirectedEmotionScores) -> AxisValues:
        """Compute the four romance axes from base emotion scores.

        Formulas:
        - Intimacy: (Trust + Joy) / 2
        - Passion: (Arousal + Anticipation + Joy) / 3
        - Hostility: (Anger + Disgust + Sadness) / 3
        - Anxiety: (Fear + Surprise + Sadness) / 3

        Args:
            emotions: Directed emotion scores (9 base emotions).

        Returns:
            AxisValues with computed axis values (1-5 scale).
        """
        # Intimacy: Trust and Joy
        intimacy = (emotions.Trust + emotions.Joy) / 2.0

        # Passion: Arousal, Anticipation, and Joy
        passion = (emotions.Arousal + emotions.Anticipation + emotions.Joy) / 3.0

        # Hostility: Anger, Disgust, and Sadness
        hostility = (emotions.Anger + emotions.Disgust + emotions.Sadness) / 3.0

        # Anxiety: Fear, Surprise, and Sadness
        anxiety = (emotions.Fear + emotions.Surprise + emotions.Sadness) / 3.0

        return AxisValues(
            intimacy=round(intimacy, 2),
            passion=round(passion, 2),
            hostility=round(hostility, 2),
            anxiety=round(anxiety, 2),
        )

    def map_chunk(self, chunk: ChunkAnalysis) -> AxisValues:
        """Map a chunk's emotions to romance axes.

        Aggregates bidirectional emotions and computes axes.

        Args:
            chunk: Chunk analysis with directed emotions.

        Returns:
            AxisValues for this chunk.
        """
        if not chunk.directed_emotions:
            # Default to baseline (all 1s -> all axes = 1.0)
            return AxisValues(intimacy=1.0, passion=1.0, hostility=1.0, anxiety=1.0)

        # Aggregate bidirectional emotions
        aggregated = self._aggregate_emotions(chunk.directed_emotions)

        return self.compute_axes(aggregated)

    def map_chunks(self, chunks: list[ChunkAnalysis]) -> AxesTimeSeries:
        """Map all chunks to romance axes time series.

        Args:
            chunks: List of chunk analyses.

        Returns:
            AxesTimeSeries with time series data for all four axes.
        """
        intimacy_series: list[float] = []
        passion_series: list[float] = []
        hostility_series: list[float] = []
        anxiety_series: list[float] = []

        for chunk in chunks:
            axis_values = self.map_chunk(chunk)
            intimacy_series.append(axis_values.intimacy)
            passion_series.append(axis_values.passion)
            hostility_series.append(axis_values.hostility)
            anxiety_series.append(axis_values.anxiety)

        return AxesTimeSeries(
            intimacy=intimacy_series,
            passion=passion_series,
            hostility=hostility_series,
            anxiety=anxiety_series,
        )

    def _aggregate_emotions(self, emotions: list) -> DirectedEmotionScores:
        """Aggregate multiple directed emotions into average scores.

        Args:
            emotions: List of DirectedEmotion objects.

        Returns:
            DirectedEmotionScores with averaged values.
        """
        if not emotions:
            return DirectedEmotionScores()

        # Sum all scores
        totals = {
            "Joy": 0,
            "Trust": 0,
            "Fear": 0,
            "Surprise": 0,
            "Sadness": 0,
            "Disgust": 0,
            "Anger": 0,
            "Anticipation": 0,
            "Arousal": 0,
        }

        for emotion in emotions:
            scores = emotion.scores
            totals["Joy"] += scores.Joy
            totals["Trust"] += scores.Trust
            totals["Fear"] += scores.Fear
            totals["Surprise"] += scores.Surprise
            totals["Sadness"] += scores.Sadness
            totals["Disgust"] += scores.Disgust
            totals["Anger"] += scores.Anger
            totals["Anticipation"] += scores.Anticipation
            totals["Arousal"] += scores.Arousal

        # Calculate averages
        count = len(emotions)
        return DirectedEmotionScores(
            Joy=round(totals["Joy"] / count),
            Trust=round(totals["Trust"] / count),
            Fear=round(totals["Fear"] / count),
            Surprise=round(totals["Surprise"] / count),
            Sadness=round(totals["Sadness"] / count),
            Disgust=round(totals["Disgust"] / count),
            Anger=round(totals["Anger"] / count),
            Anticipation=round(totals["Anticipation"] / count),
            Arousal=round(totals["Arousal"] / count),
        )

    def compute_directional_axes(
        self, chunk: ChunkAnalysis
    ) -> tuple[AxisValues | None, AxisValues | None]:
        """Compute axes for each direction separately (A→B and B→A).

        Args:
            chunk: Chunk analysis with directed emotions.

        Returns:
            Tuple of (A→B axes, B→A axes). Either may be None if direction not present.
        """
        if len(chunk.directed_emotions) < 2:
            # Only one direction available
            if chunk.directed_emotions:
                return (self.compute_axes(chunk.directed_emotions[0].scores), None)
            return (None, None)

        # Assume first two emotions are A→B and B→A
        a_to_b = self.compute_axes(chunk.directed_emotions[0].scores)
        b_to_a = self.compute_axes(chunk.directed_emotions[1].scores)

        return (a_to_b, b_to_a)
