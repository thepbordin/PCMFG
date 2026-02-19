"""Tests for the main PCMFG Analyzer."""

from unittest.mock import MagicMock, patch

import pytest

from pcmfg.analyzer import PCMFGAnalyzer, analyze
from pcmfg.config import Config
from pcmfg.models.schemas import (
    AnalysisResult,
    ChunkAnalysis,
    DirectedEmotion,
    DirectedEmotionScores,
    WorldBuilderOutput,
)


class TestPCMFGAnalyzer:
    """Tests for the PCMFGAnalyzer class."""

    def test_init_with_config(self, sample_config: Config) -> None:
        """Test analyzer initialization with config."""
        with patch.object(PCMFGAnalyzer, "_create_llm_client") as mock_create:
            mock_create.return_value = MagicMock()
            analyzer = PCMFGAnalyzer(config=sample_config)
            assert analyzer.config == sample_config

    def test_init_with_llm_client(self, mock_llm_client: MagicMock) -> None:
        """Test analyzer initialization with pre-configured LLM client."""
        analyzer = PCMFGAnalyzer(llm_client=mock_llm_client)
        assert analyzer.llm_client == mock_llm_client

    def test_analyze_simple_text(
        self, mock_llm_client: MagicMock, sample_text: str
    ) -> None:
        """Test analyzing simple text."""
        # Set up mock responses
        mock_llm_client.call_json.side_effect = [
            # World builder response
            {
                "main_pairing": ["Alice", "Bob"],
                "aliases": {"Alice": ["Ali"], "Bob": ["Robert"]},
                "core_conflict": "They are from different worlds.",
                "world_guidelines": ["They met at a ball."],
                "mermaid_graph": "graph TD\n    A --> B",
            },
            # Emotion extractor response
            {
                "chunk_id": 0,
                "chunk_main_pov": "Alice",
                "characters_present": ["Alice", "Bob"],
                "directed_emotions": [
                    {
                        "source": "Alice",
                        "target": "Bob",
                        "scores": {
                            "Joy": 3,
                            "Trust": 2,
                            "Fear": 1,
                            "Surprise": 1,
                            "Sadness": 1,
                            "Disgust": 1,
                            "Anger": 1,
                            "Anticipation": 2,
                            "Arousal": 2,
                        },
                        "justification_quote": "She felt drawn to him.",
                    },
                    {
                        "source": "Bob",
                        "target": "Alice",
                        "scores": {
                            "Joy": 2,
                            "Trust": 1,
                            "Fear": 1,
                            "Surprise": 1,
                            "Sadness": 1,
                            "Disgust": 1,
                            "Anger": 1,
                            "Anticipation": 1,
                            "Arousal": 2,
                        },
                        "justification_quote": "He thought she was beautiful.",
                    },
                ],
                "scene_summary": "They meet at the ball.",
            },
        ]

        analyzer = PCMFGAnalyzer(llm_client=mock_llm_client)
        result = analyzer.analyze(sample_text, source="test.txt")

        assert isinstance(result, AnalysisResult)
        assert result.metadata.source == "test.txt"
        assert result.metadata.total_chunks >= 1
        assert len(result.world_builder.main_pairing) == 2
        # Check timeseries instead of axes (new output format)
        assert "A_to_B" in result.timeseries
        assert "B_to_A" in result.timeseries
        assert len(result.timeseries["A_to_B"].Joy) >= 1

    def test_analyze_creates_all_phases(
        self, mock_llm_client: MagicMock, sample_text: str
    ) -> None:
        """Test that analyze creates all phase processors."""
        mock_llm_client.call_json.return_value = {
            "main_pairing": ["A", "B"],
            "aliases": {},
            "core_conflict": "Test conflict.",
            "world_guidelines": [],
            "mermaid_graph": "",
        }

        analyzer = PCMFGAnalyzer(llm_client=mock_llm_client)

        # Verify phase processors are created
        assert analyzer.world_builder is not None
        assert analyzer.synthesizer is not None

    def test_chunk_text_automatic(self, mock_llm_client: MagicMock) -> None:
        """Test text chunking in automatic mode."""
        config = Config()
        config.processing.beat_detection = "automatic"

        with patch.object(PCMFGAnalyzer, "_create_llm_client") as mock_create:
            mock_create.return_value = mock_llm_client
            analyzer = PCMFGAnalyzer(config=config)

            # Long text should be chunked
            long_text = "Paragraph one.\n\n" * 100
            chunks = analyzer._chunk_text(long_text)

            assert len(chunks) >= 1

    def test_chunk_text_by_length(self, mock_llm_client: MagicMock) -> None:
        """Test text chunking by length."""
        config = Config()
        config.processing.beat_detection = "length"
        config.processing.beat_length = 50

        with patch.object(PCMFGAnalyzer, "_create_llm_client") as mock_create:
            mock_create.return_value = mock_llm_client
            analyzer = PCMFGAnalyzer(config=config)

            # Create text longer than beat_length
            long_text = " ".join(["word"] * 200)
            chunks = analyzer._chunk_text(long_text)

            assert len(chunks) >= 1

    def test_get_text_sample(self, mock_llm_client: MagicMock) -> None:
        """Test text sampling for world builder."""
        with patch.object(PCMFGAnalyzer, "_create_llm_client") as mock_create:
            mock_create.return_value = mock_llm_client
            analyzer = PCMFGAnalyzer()

            long_text = " ".join(["word"] * 5000)
            sample = analyzer._get_text_sample(long_text, 1000)

            # Sample should be shorter than full text
            assert len(sample.split()) < len(long_text.split())


class TestConvenienceFunction:
    """Tests for the convenience analyze function."""

    def test_analyze_function(self, sample_text: str) -> None:
        """Test the convenience analyze function."""
        with patch("pcmfg.analyzer.PCMFGAnalyzer") as mock_analyzer_class:
            mock_analyzer = MagicMock()
            mock_analyzer_class.return_value = mock_analyzer
            mock_analyzer.analyze.return_value = MagicMock(spec=AnalysisResult)

            result = analyze(sample_text, provider="openai")

            assert result is not None
            mock_analyzer.analyze.assert_called_once()


class TestAnalysisResult:
    """Tests for the AnalysisResult model."""

    def test_default_values(self) -> None:
        """Test AnalysisResult default values."""
        result = AnalysisResult()
        assert result.metadata is not None
        assert result.world_builder is not None
        assert result.chunks == []
        assert result.axes is not None

    def test_model_dump(self) -> None:
        """Test model serialization."""
        result = AnalysisResult(
            world_builder=WorldBuilderOutput(main_pairing=["A", "B"]),
            chunks=[
                ChunkAnalysis(
                    chunk_id=0,
                    chunk_main_pov="A",
                    characters_present=["A", "B"],
                    directed_emotions=[],
                    scene_summary="Test",
                )
            ],
        )

        data = result.model_dump()

        assert "world_builder" in data
        assert "chunks" in data
        assert "axes" in data
