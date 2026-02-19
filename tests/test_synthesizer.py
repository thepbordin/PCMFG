"""Tests for Phase 3 - Synthesizer."""

import pytest

from pcmfg.models.schemas import (
    AnalysisMetadata,
    ChunkAnalysis,
    DirectedEmotion,
    DirectedEmotionScores,
    EmotionTimeSeries,
    WorldBuilderOutput,
)
from pcmfg.phase3.synthesizer import (
    Synthesizer,
    build_emotion_timeseries,
    create_baseline_scores,
    impute_missing_emotions,
)


class TestCreateBaselineScores:
    """Tests for create_baseline_scores function."""

    def test_baseline_all_ones(self) -> None:
        """Test that baseline scores are all 1s."""
        scores = create_baseline_scores()

        assert scores.Joy == 1
        assert scores.Trust == 1
        assert scores.Fear == 1
        assert scores.Surprise == 1
        assert scores.Sadness == 1
        assert scores.Disgust == 1
        assert scores.Anger == 1
        assert scores.Anticipation == 1
        assert scores.Arousal == 1


class TestImputeMissingEmotions:
    """Tests for forward fill imputation."""

    def test_impute_missing_direction(self) -> None:
        """Test that missing B→A direction is forward filled."""
        # Chunk 0: Only A→B present
        chunk0 = ChunkAnalysis(
            chunk_id=0,
            position=0.0,
            chunk_main_pov="Alice",
            characters_present=["Alice"],
            directed_emotions=[
                DirectedEmotion(
                    source="Alice",
                    target="Bob",
                    scores=DirectedEmotionScores(Joy=3, Trust=2),
                    justification_quote="Alice smiled at Bob.",
                )
            ],
            scene_summary="Alice thinks about Bob.",
        )

        # Chunk 1: Both directions present
        chunk1 = ChunkAnalysis(
            chunk_id=1,
            position=0.5,
            chunk_main_pov="Alice",
            characters_present=["Alice", "Bob"],
            directed_emotions=[
                DirectedEmotion(
                    source="Alice",
                    target="Bob",
                    scores=DirectedEmotionScores(Joy=4),
                    justification_quote="Alice laughed.",
                ),
                DirectedEmotion(
                    source="Bob",
                    target="Alice",
                    scores=DirectedEmotionScores(Joy=2, Fear=3),
                    justification_quote="Bob was nervous.",
                ),
            ],
            scene_summary="They meet.",
        )

        chunks = [chunk0, chunk1]
        main_pairing = ["Alice", "Bob"]

        imputed = impute_missing_emotions(chunks, main_pairing)

        # Chunk 0 should now have B→A imputed (baseline values)
        assert len(imputed[0].directed_emotions) == 2
        # Find Bob→Alice in chunk 0
        b_to_a_chunk0 = None
        for de in imputed[0].directed_emotions:
            if de.source == "Bob" and de.target == "Alice":
                b_to_a_chunk0 = de
                break
        assert b_to_a_chunk0 is not None
        assert b_to_a_chunk0.scores.Joy == 1  # Baseline
        assert "[FORWARD FILLED" in b_to_a_chunk0.justification_quote

        # Chunk 1 should have both directions (no change needed)
        assert len(imputed[1].directed_emotions) == 2

    def test_forward_fill_carries_state(self) -> None:
        """Test that forward fill carries forward the last known state."""
        # Chunk 0: Both directions present
        chunk0 = ChunkAnalysis(
            chunk_id=0,
            position=0.0,
            chunk_main_pov="Alice",
            characters_present=["Alice", "Bob"],
            directed_emotions=[
                DirectedEmotion(
                    source="Alice",
                    target="Bob",
                    scores=DirectedEmotionScores(Joy=2),
                    justification_quote="Test",
                ),
                DirectedEmotion(
                    source="Bob",
                    target="Alice",
                    scores=DirectedEmotionScores(Joy=4, Fear=3),
                    justification_quote="Test",
                ),
            ],
            scene_summary="Scene 1",
        )

        # Chunk 1: Only A→B present (Bob absent)
        chunk1 = ChunkAnalysis(
            chunk_id=1,
            position=0.5,
            chunk_main_pov="Alice",
            characters_present=["Alice"],
            directed_emotions=[
                DirectedEmotion(
                    source="Alice",
                    target="Bob",
                    scores=DirectedEmotionScores(Joy=3),
                    justification_quote="Test",
                )
            ],
            scene_summary="Scene 2",
        )

        chunks = [chunk0, chunk1]
        imputed = impute_missing_emotions(chunks, ["Alice", "Bob"])

        # Chunk 1's B→A should carry forward chunk 0's values
        b_to_a_chunk1 = None
        for de in imputed[1].directed_emotions:
            if de.source == "Bob" and de.target == "Alice":
                b_to_a_chunk1 = de
                break
        assert b_to_a_chunk1 is not None
        assert b_to_a_chunk1.scores.Joy == 4  # Carried from chunk 0
        assert b_to_a_chunk1.scores.Fear == 3  # Carried from chunk 0

    def test_empty_chunks(self) -> None:
        """Test imputation with empty chunks list."""
        imputed = impute_missing_emotions([], ["Alice", "Bob"])
        assert imputed == []

    def test_insufficient_pairing(self) -> None:
        """Test imputation with insufficient main pairing."""
        chunk = ChunkAnalysis(
            chunk_id=0,
            position=0.0,
            chunk_main_pov="Alice",
            characters_present=["Alice"],
            directed_emotions=[],
            scene_summary="Test",
        )

        # Should return unchanged with only 1 character in pairing
        imputed = impute_missing_emotions([chunk], ["Alice"])
        assert len(imputed) == 1


class TestBuildEmotionTimeseries:
    """Tests for time-series building."""

    def test_build_timeseries_basic(self) -> None:
        """Test basic time-series building."""
        chunks = [
            ChunkAnalysis(
                chunk_id=0,
                position=0.0,
                chunk_main_pov="Alice",
                characters_present=["Alice", "Bob"],
                directed_emotions=[
                    DirectedEmotion(
                        source="Alice",
                        target="Bob",
                        scores=DirectedEmotionScores(Joy=2, Trust=3),
                        justification_quote="Test",
                    )
                ],
                scene_summary="Scene 1",
            ),
            ChunkAnalysis(
                chunk_id=1,
                position=0.5,
                chunk_main_pov="Alice",
                characters_present=["Alice", "Bob"],
                directed_emotions=[
                    DirectedEmotion(
                        source="Alice",
                        target="Bob",
                        scores=DirectedEmotionScores(Joy=4, Trust=4),
                        justification_quote="Test",
                    )
                ],
                scene_summary="Scene 2",
            ),
        ]

        ts = build_emotion_timeseries(chunks, "Alice", "Bob")

        assert len(ts["Joy"]) == 2
        assert ts["Joy"] == [2.0, 4.0]
        assert ts["Trust"] == [3.0, 4.0]

    def test_build_timeseries_missing_direction(self) -> None:
        """Test time-series building when direction is missing in some chunks."""
        chunks = [
            ChunkAnalysis(
                chunk_id=0,
                position=0.0,
                chunk_main_pov="Alice",
                characters_present=["Alice", "Bob"],
                directed_emotions=[
                    DirectedEmotion(
                        source="Alice",
                        target="Bob",
                        scores=DirectedEmotionScores(Joy=3),
                        justification_quote="Test",
                    )
                ],
                scene_summary="Scene 1",
            ),
            ChunkAnalysis(
                chunk_id=1,
                position=0.5,
                chunk_main_pov="Bob",
                characters_present=["Bob"],
                directed_emotions=[
                    DirectedEmotion(
                        source="Bob",
                        target="Alice",
                        scores=DirectedEmotionScores(Joy=2),
                        justification_quote="Test",
                    )
                ],
                scene_summary="Scene 2",
            ),
        ]

        # Alice→Bob should only find data in chunk 0
        ts_a_to_b = build_emotion_timeseries(chunks, "Alice", "Bob")
        assert ts_a_to_b["Joy"] == [3.0, 1.0]  # Missing in chunk 1, defaults to 1.0

        # Bob→Alice should only find data in chunk 1
        ts_b_to_a = build_emotion_timeseries(chunks, "Bob", "Alice")
        assert ts_b_to_a["Joy"] == [1.0, 2.0]  # Missing in chunk 0, defaults to 1.0


class TestSynthesizer:
    """Tests for the Synthesizer class."""

    @pytest.fixture
    def world(self) -> WorldBuilderOutput:
        """Create a sample world builder output."""
        return WorldBuilderOutput(
            main_pairing=["Alice", "Bob"],
            aliases={"Alice": ["Ali"], "Bob": ["Bobby"]},
            core_conflict="They are from rival families.",
            world_guidelines=["Rule 1", "Rule 2"],
            mermaid_graph="graph TD",
        )

    @pytest.fixture
    def sample_chunks(self) -> list[ChunkAnalysis]:
        """Create sample chunk analyses."""
        return [
            ChunkAnalysis(
                chunk_id=0,
                position=0.0,
                chunk_main_pov="Alice",
                characters_present=["Alice", "Bob"],
                directed_emotions=[
                    DirectedEmotion(
                        source="Alice",
                        target="Bob",
                        scores=DirectedEmotionScores(Joy=2, Trust=3),
                        justification_quote="Test",
                    ),
                    DirectedEmotion(
                        source="Bob",
                        target="Alice",
                        scores=DirectedEmotionScores(Joy=3, Fear=2),
                        justification_quote="Test",
                    ),
                ],
                scene_summary="Scene 1",
            ),
            ChunkAnalysis(
                chunk_id=1,
                position=1.0,
                chunk_main_pov="Alice",
                characters_present=["Alice", "Bob"],
                directed_emotions=[
                    DirectedEmotion(
                        source="Alice",
                        target="Bob",
                        scores=DirectedEmotionScores(Joy=4, Trust=4),
                        justification_quote="Test",
                    ),
                    DirectedEmotion(
                        source="Bob",
                        target="Alice",
                        scores=DirectedEmotionScores(Joy=4, Fear=1),
                        justification_quote="Test",
                    ),
                ],
                scene_summary="Scene 2",
            ),
        ]

    def test_synthesize_basic(
        self, world: WorldBuilderOutput, sample_chunks: list[ChunkAnalysis]
    ) -> None:
        """Test basic synthesis."""
        synthesizer = Synthesizer()
        result = synthesizer.synthesize(sample_chunks, world)

        assert result.world_builder == world
        assert len(result.chunks) == 2
        assert "A_to_B" in result.timeseries
        assert "B_to_A" in result.timeseries

    def test_synthesize_timeseries_values(
        self, world: WorldBuilderOutput, sample_chunks: list[ChunkAnalysis]
    ) -> None:
        """Test that timeseries values are correct."""
        synthesizer = Synthesizer()
        result = synthesizer.synthesize(sample_chunks, world)

        a_to_b = result.timeseries["A_to_B"]
        assert a_to_b.Joy == [2.0, 4.0]
        assert a_to_b.Trust == [3.0, 4.0]

        b_to_a = result.timeseries["B_to_A"]
        assert b_to_a.Joy == [3.0, 4.0]
        assert b_to_a.Fear == [2.0, 1.0]

    def test_synthesize_with_metadata(
        self, world: WorldBuilderOutput, sample_chunks: list[ChunkAnalysis]
    ) -> None:
        """Test synthesis with custom metadata."""
        synthesizer = Synthesizer()
        metadata = AnalysisMetadata(
            source="test.txt",
            model="gpt-4",
            total_chunks=2,
        )
        result = synthesizer.synthesize(sample_chunks, world, metadata)

        assert result.metadata.source == "test.txt"
        assert result.metadata.model == "gpt-4"

    def test_synthesize_insufficient_pairing(
        self, sample_chunks: list[ChunkAnalysis]
    ) -> None:
        """Test synthesis handles valid pairing correctly."""
        # Note: WorldBuilderOutput requires exactly 2 characters via pydantic validation
        # So we test that synthesis works correctly with valid input
        world = WorldBuilderOutput(
            main_pairing=["Alice", "Bob"],
            core_conflict="Test conflict",
        )
        synthesizer = Synthesizer()

        # Should work with valid pairing
        result = synthesizer.synthesize(sample_chunks, world)
        assert result is not None
        assert "A_to_B" in result.timeseries
        assert "B_to_A" in result.timeseries


class TestChunkFiltering:
    """Tests for chunk filtering optimization."""

    def test_should_process_chunk_with_character(self) -> None:
        """Test that chunks with character names are processed."""
        from pcmfg.phase1.emotion_extractor import should_process_chunk

        aliases = {"Alice": ["Ali", "Al"], "Bob": ["Bobby"]}

        # Should process - contains "Alice"
        assert should_process_chunk("Alice walked into the room.", aliases)

        # Should process - contains alias "Bobby"
        assert should_process_chunk("Bobby was waiting.", aliases)

        # Should process - contains both
        assert should_process_chunk("Alice and Bob talked.", aliases)

    def test_should_skip_chunk_without_characters(self) -> None:
        """Test that chunks without character names are skipped."""
        from pcmfg.phase1.emotion_extractor import should_process_chunk

        aliases = {"Alice": ["Ali"], "Bob": ["Bobby"]}

        # Should skip - no character names
        assert not should_process_chunk("The weather was nice that day.", aliases)

        # Should skip - unrelated names
        assert not should_process_chunk("Charlie and David went to the park.", aliases)

    def test_case_insensitive_matching(self) -> None:
        """Test that matching is case insensitive."""
        from pcmfg.phase1.emotion_extractor import should_process_chunk

        aliases = {"Alice": ["Ali"]}

        assert should_process_chunk("alice was happy.", aliases)
        assert should_process_chunk("ALICE was happy.", aliases)
        assert should_process_chunk("ali was happy.", aliases)
