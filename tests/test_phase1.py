"""Tests for Phase 1 - World Builder."""

import pytest

from pcmfg.phase1.world_builder import WorldBuilder, WorldBuilderError, WORLD_BUILDER_SYSTEM_PROMPT
from pcmfg.models.schemas import WorldBuilderOutput


class TestWorldBuilder:
    """Tests for the WorldBuilder class."""

    def test_init(self, mock_llm_client) -> None:
        """Test WorldBuilder initialization."""
        builder = WorldBuilder(mock_llm_client)
        assert builder.llm_client == mock_llm_client

    def test_build_success(self, mock_llm_client, sample_text: str) -> None:
        """Test successful world building."""
        builder = WorldBuilder(mock_llm_client)
        result = builder.build(sample_text)

        assert isinstance(result, WorldBuilderOutput)
        assert len(result.main_pairing) == 2
        assert isinstance(result.aliases, dict)
        assert isinstance(result.world_guidelines, list)
        assert isinstance(result.mermaid_graph, str)

    def test_build_with_custom_response(self, mock_llm_client, sample_text: str) -> None:
        """Test world building with custom LLM response."""
        mock_llm_client.call_json.return_value = {
            "main_pairing": ["Alice", "Bob"],
            "aliases": {"Alice": ["Ali", "Al"], "Bob": ["Robert", "Bobby"]},
            "world_guidelines": [
                "They met at a ball.",
                "Alice initially disliked Bob.",
                "They eventually fell in love.",
            ],
            "mermaid_graph": "graph TD\n    A[Alice] -->|loves| B[Bob]",
        }

        builder = WorldBuilder(mock_llm_client)
        result = builder.build(sample_text)

        assert result.main_pairing == ["Alice", "Bob"]
        assert "Alice" in result.aliases
        assert len(result.world_guidelines) == 3

    def test_build_with_missing_fields(self, mock_llm_client, sample_text: str) -> None:
        """Test world building with missing optional fields."""
        mock_llm_client.call_json.return_value = {
            "main_pairing": ["Alice", "Bob"],
            # Missing aliases, world_guidelines, mermaid_graph
        }

        builder = WorldBuilder(mock_llm_client)
        result = builder.build(sample_text)

        assert result.main_pairing == ["Alice", "Bob"]
        assert result.aliases == {}
        assert result.world_guidelines == []
        assert result.mermaid_graph == ""

    def test_build_with_too_many_characters(self, mock_llm_client, sample_text: str) -> None:
        """Test world building with more than 2 main characters."""
        mock_llm_client.call_json.return_value = {
            "main_pairing": ["Alice", "Bob", "Charlie"],
            "aliases": {},
            "world_guidelines": [],
            "mermaid_graph": "",
        }

        builder = WorldBuilder(mock_llm_client)
        result = builder.build(sample_text)

        # Should take only first 2
        assert len(result.main_pairing) == 2
        assert result.main_pairing == ["Alice", "Bob"]

    def test_build_with_too_few_characters(self, mock_llm_client, sample_text: str) -> None:
        """Test world building with fewer than 2 main characters."""
        mock_llm_client.call_json.return_value = {
            "main_pairing": ["Alice"],
            "aliases": {},
            "world_guidelines": [],
            "mermaid_graph": "",
        }

        builder = WorldBuilder(mock_llm_client)
        with pytest.raises(WorldBuilderError):
            builder.build(sample_text)

    def test_build_llm_error(self, mock_llm_client, sample_text: str) -> None:
        """Test world building when LLM fails."""
        from pcmfg.llm.base import LLMAPIError

        mock_llm_client.call_json.side_effect = LLMAPIError("API failed")

        builder = WorldBuilder(mock_llm_client)
        with pytest.raises(WorldBuilderError):
            builder.build(sample_text)

    def test_system_prompt_contains_required_elements(self) -> None:
        """Test that the system prompt has all required elements."""
        assert "main_pairing" in WORLD_BUILDER_SYSTEM_PROMPT
        assert "aliases" in WORLD_BUILDER_SYSTEM_PROMPT
        assert "world_guidelines" in WORLD_BUILDER_SYSTEM_PROMPT
        assert "mermaid_graph" in WORLD_BUILDER_SYSTEM_PROMPT
        assert "JSON" in WORLD_BUILDER_SYSTEM_PROMPT


class TestWorldBuilderOutput:
    """Tests for the WorldBuilderOutput model."""

    def test_default_values(self) -> None:
        """Test default values for WorldBuilderOutput."""
        output = WorldBuilderOutput(main_pairing=["A", "B"])
        assert output.aliases == {}
        assert output.world_guidelines == []
        assert output.mermaid_graph == ""

    def test_valid_output(self) -> None:
        """Test valid WorldBuilderOutput creation."""
        output = WorldBuilderOutput(
            main_pairing=["Alice", "Bob"],
            aliases={"Alice": ["Ali"]},
            world_guidelines=["They met at a ball."],
            mermaid_graph="graph TD",
        )
        assert output.main_pairing == ["Alice", "Bob"]

    def test_main_pairing_length_validation(self) -> None:
        """Test that main_pairing must have exactly 2 elements."""
        # Should work with 2
        WorldBuilderOutput(main_pairing=["A", "B"])

        # Should fail with 1
        with pytest.raises(Exception):  # Pydantic ValidationError
            WorldBuilderOutput(main_pairing=["A"])

        # Should fail with 3
        with pytest.raises(Exception):  # Pydantic ValidationError
            WorldBuilderOutput(main_pairing=["A", "B", "C"])
