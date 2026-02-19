"""Tests for text processing utilities."""

import pytest

from pcmfg.utils.text_processing import (
    chunk_text_by_chapter,
    chunk_text_by_length,
    clean_text,
    estimate_tokens,
    get_strategic_sample,
)


class TestEstimateTokens:
    """Tests for estimate_tokens function."""

    def test_empty_string(self) -> None:
        """Test token estimation for empty string."""
        assert estimate_tokens("") == 0

    def test_simple_text(self) -> None:
        """Test token estimation for simple text."""
        text = "Hello world this is a test"
        # 6 words * 1.3 = 7.8 -> 7
        assert estimate_tokens(text) == 7

    def test_longer_text(self) -> None:
        """Test token estimation for longer text."""
        text = " ".join(["word"] * 100)
        # 100 words * 1.3 = 130
        assert estimate_tokens(text) == 130


class TestCleanText:
    """Tests for clean_text function."""

    def test_remove_control_characters(self) -> None:
        """Test removal of control characters."""
        text = "Hello\x00World"
        assert clean_text(text) == "HelloWorld"

    def test_normalize_line_endings(self) -> None:
        """Test normalization of line endings."""
        text = "Hello\r\nWorld\rTest"
        assert clean_text(text) == "Hello\nWorld\nTest"

    def test_remove_excessive_blank_lines(self) -> None:
        """Test removal of excessive blank lines."""
        text = "Hello\n\n\n\n\nWorld"
        assert clean_text(text) == "Hello\n\nWorld"

    def test_strip_whitespace(self) -> None:
        """Test stripping of leading/trailing whitespace."""
        text = "   Hello World   "
        assert clean_text(text) == "Hello World"


class TestChunkTextByLength:
    """Tests for chunk_text_by_length function."""

    def test_empty_text(self) -> None:
        """Test chunking empty text."""
        assert chunk_text_by_length("") == []

    def test_short_text(self) -> None:
        """Test chunking short text."""
        text = "Hello world"
        chunks = chunk_text_by_length(text)
        assert len(chunks) == 1
        assert chunks[0] == "Hello world"

    def test_long_text(self) -> None:
        """Test chunking long text."""
        # Use paragraphs to ensure chunking happens
        text = "\n\n".join([" ".join(["word"] * 500) for _ in range(3)])
        chunks = chunk_text_by_length(text, max_tokens=500)
        assert len(chunks) > 1

    def test_respect_min_chunk_size(self) -> None:
        """Test that chunks respect minimum size."""
        text = " ".join(["word"] * 100)
        chunks = chunk_text_by_length(text, max_tokens=50, min_chunk_tokens=30)
        # All chunks should have at least min_chunk_tokens worth of words
        for chunk in chunks:
            if chunk:  # Skip empty chunks
                word_count = len(chunk.split())
                assert word_count >= 20  # min_chunk_tokens / 1.3 approx


class TestChunkTextByChapter:
    """Tests for chunk_text_by_chapter function."""

    def test_no_chapters(self) -> None:
        """Test chunking text without chapter markers."""
        text = "Just some text without chapters."
        chunks = chunk_text_by_chapter(text)
        # Should fall back to length-based chunking
        assert len(chunks) >= 1

    def test_with_chapters(self) -> None:
        """Test chunking text with chapter markers."""
        text = """
Chapter 1

This is the first chapter. It has some content.

Chapter 2

This is the second chapter. More content here.
"""
        chunks = chunk_text_by_chapter(text)
        assert len(chunks) == 2

    def test_long_chapter_split(self) -> None:
        """Test that long chapters are split."""
        # Create a long chapter with multiple paragraphs
        paragraphs = "\n\n".join([" ".join(["word"] * 500) for _ in range(10)])
        text = f"Chapter 1\n\n{paragraphs}"
        chunks = chunk_text_by_chapter(text, max_tokens=500)
        # Should be split into multiple chunks
        assert len(chunks) > 1


class TestGetStrategicSample:
    """Tests for get_strategic_sample function."""

    def test_short_text_returns_full(self) -> None:
        """Test that short text is returned in full."""
        text = "Hello world this is a short text."
        result = get_strategic_sample(text, max_tokens=100)
        assert result == text

    def test_long_text_is_sampled(self) -> None:
        """Test that long text is strategically sampled."""
        # Create a long text
        text = " ".join(["word"] * 10000)
        result = get_strategic_sample(text, max_tokens=1000)

        # Result should be shorter than original
        assert len(result.split()) < len(text.split())

        # Should contain section markers
        assert "[BEGINNING]" in result
        assert "[MIDDLE]" in result
        assert "[END]" in result

    def test_contains_beginning_section(self) -> None:
        """Test that sample contains beginning of text."""
        text = "STARTMARKER " + " ".join(["word"] * 5000) + " ENDMARKER"
        result = get_strategic_sample(text, max_tokens=1000)

        # Beginning should contain the start marker
        assert "STARTMARKER" in result

    def test_contains_end_section(self) -> None:
        """Test that sample contains end of text."""
        text = "STARTMARKER " + " ".join(["word"] * 5000) + " ENDMARKER"
        result = get_strategic_sample(text, max_tokens=1000)

        # End should contain the end marker
        assert "ENDMARKER" in result

    def test_sections_separated_by_delimiter(self) -> None:
        """Test that sections are separated by delimiter."""
        text = " ".join(["word"] * 10000)
        result = get_strategic_sample(text, max_tokens=2000)

        # Should have section separators
        assert "---" in result

    def test_respects_max_tokens(self) -> None:
        """Test that result respects max_tokens limit."""
        text = " ".join(["word"] * 10000)
        max_tokens = 1000
        result = get_strategic_sample(text, max_tokens=max_tokens)

        # Result should be under max_tokens (with some tolerance for markers)
        result_tokens = estimate_tokens(result)
        # Allow 20% tolerance for section markers
        assert result_tokens < max_tokens * 1.2

    def test_exact_text_at_limit(self) -> None:
        """Test text that is exactly at the limit."""
        # Create text that's close to the limit
        words_needed = int(8000 / 1.3)  # ~6154 words for 8000 tokens
        text = " ".join(["word"] * words_needed)
        result = get_strategic_sample(text, max_tokens=8000)

        # Should return full text since it's under the limit
        assert result == text
