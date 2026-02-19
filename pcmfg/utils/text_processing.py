"""Text processing utilities for PCMFG.

This module provides functions for:
- Token estimation
- Text chunking for LLM context windows
- Text cleaning and normalization
"""

import re
from collections.abc import Iterator


def estimate_tokens(text: str) -> int:
    """Estimate the number of tokens in a text string.

    Uses a simple heuristic: words * 1.3 (accounts for subword tokenization).

    Args:
        text: Input text string.

    Returns:
        Estimated token count.
    """
    word_count = len(text.split())
    return int(word_count * 1.3)


def clean_text(text: str) -> str:
    """Clean and normalize text for processing.

    Operations:
    - Remove excessive whitespace
    - Normalize line endings
    - Remove control characters

    Args:
        text: Input text string.

    Returns:
        Cleaned text string.
    """
    # Remove control characters except newlines and tabs
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]", "", text)

    # Normalize line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Remove excessive blank lines (more than 2 consecutive)
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Remove trailing/leading whitespace from each line
    lines = [line.rstrip() for line in text.split("\n")]
    text = "\n".join(lines)

    # Remove leading/trailing whitespace from entire text
    text = text.strip()

    return text


def chunk_text_by_length(
    text: str,
    max_tokens: int = 3000,
    min_chunk_tokens: int = 200,
) -> list[str]:
    """Split text into chunks by word count for LLM processing.

    Attempts to split at sentence boundaries when possible.

    Args:
        text: Input text string.
        max_tokens: Maximum tokens per chunk.
        min_chunk_tokens: Minimum tokens per chunk (avoid over-fragmentation).

    Returns:
        List of text chunks.
    """
    if not text:
        return []

    # Clean text first
    text = clean_text(text)

    # Estimate max words per chunk
    max_words = int(max_tokens / 1.3)
    min_words = int(min_chunk_tokens / 1.3)

    # Split into paragraphs first
    paragraphs = text.split("\n\n")

    chunks: list[str] = []
    current_chunk: list[str] = []
    current_word_count = 0

    for para in paragraphs:
        para_words = len(para.split())

        # If single paragraph exceeds max, split by sentences
        if para_words > max_words:
            # Flush current chunk
            if current_chunk:
                chunk_text = "\n\n".join(current_chunk)
                if len(chunk_text.split()) >= min_words:
                    chunks.append(chunk_text)
                current_chunk = []
                current_word_count = 0

            # Split paragraph by sentences
            sentences = _split_sentences(para)
            for sentence in sentences:
                sent_words = len(sentence.split())
                if current_word_count + sent_words > max_words:
                    if current_chunk:
                        chunks.append(" ".join(current_chunk))
                    current_chunk = [sentence]
                    current_word_count = sent_words
                else:
                    current_chunk.append(sentence)
                    current_word_count += sent_words

        # Check if adding this paragraph would exceed limit
        elif current_word_count + para_words > max_words:
            # Flush current chunk if it meets minimum
            if current_chunk:
                chunk_text = "\n\n".join(current_chunk)
                if len(chunk_text.split()) >= min_words:
                    chunks.append(chunk_text)
            current_chunk = [para]
            current_word_count = para_words

        else:
            current_chunk.append(para)
            current_word_count += para_words

    # Don't forget the last chunk
    if current_chunk:
        chunk_text = "\n\n".join(current_chunk)
        chunks.append(chunk_text)

    return chunks


def chunk_text_by_chapter(
    text: str,
    max_tokens: int = 3000,
    chapter_pattern: str = r"(?i)^chapter\s+\d+",
) -> list[str]:
    """Split text into chunks by chapter markers.

    Args:
        text: Input text string.
        max_tokens: Maximum tokens per chunk (for very long chapters).
        chapter_pattern: Regex pattern to match chapter headings.

    Returns:
        List of text chunks, one per chapter.
    """
    if not text:
        return []

    text = clean_text(text)
    lines = text.split("\n")

    # Find chapter boundaries
    chapter_starts: list[int] = []
    for i, line in enumerate(lines):
        if re.match(chapter_pattern, line.strip()):
            chapter_starts.append(i)

    # If no chapters found, fall back to length-based chunking
    if not chapter_starts:
        return chunk_text_by_length(text, max_tokens)

    # Extract chapter chunks
    chunks: list[str] = []
    chapter_starts.append(len(lines))  # Add end boundary

    for i, start in enumerate(chapter_starts[:-1]):
        end = chapter_starts[i + 1]
        chapter_lines = lines[start:end]
        chapter_text = "\n".join(chapter_lines).strip()

        if chapter_text:
            # If chapter is too long, split it further
            if estimate_tokens(chapter_text) > max_tokens:
                sub_chunks = chunk_text_by_length(chapter_text, max_tokens)
                chunks.extend(sub_chunks)
            else:
                chunks.append(chapter_text)

    return chunks


def chunk_text_by_scenes(
    text: str,
    max_tokens: int = 3000,
    scene_markers: list[str] | None = None,
) -> list[str]:
    """Split text into chunks by scene markers.

    Scene markers include blank lines, asterisks, or other indicators.

    Args:
        text: Input text string.
        max_tokens: Maximum tokens per chunk.
        scene_markers: Custom scene marker patterns.

    Returns:
        List of text chunks.
    """
    if scene_markers is None:
        scene_markers = [r"\*{3,}", r"#\*#", r"-{3,}"]

    if not text:
        return []

    text = clean_text(text)

    # Build regex pattern for scene markers
    pattern = "|".join(f"({marker})" for marker in scene_markers)

    # Split by scene markers
    parts = re.split(pattern, text)

    # Filter out empty parts and marker matches
    chunks: list[str] = []
    current_chunk = ""

    for part in parts:
        if not part or not part.strip():
            continue

        # Check if this is a scene marker
        if any(re.fullmatch(marker, part.strip()) for marker in scene_markers):
            if current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = ""
        else:
            if estimate_tokens(current_chunk + part) > max_tokens:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = part
            else:
                current_chunk += "\n\n" + part

    if current_chunk:
        chunks.append(current_chunk.strip())

    # Fall back to length-based if no scene markers found
    if not chunks:
        return chunk_text_by_length(text, max_tokens)

    return chunks


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences.

    Args:
        text: Input text string.

    Returns:
        List of sentences.
    """
    # Basic sentence splitting (handles common cases)
    # This is intentionally simple - for production, consider using nltk or spacy
    sentence_endings = r"(?<=[.!?])\s+(?=[A-Z])"
    sentences = re.split(sentence_endings, text)
    return [s.strip() for s in sentences if s.strip()]


def iter_chunks(text: str, max_tokens: int = 3000) -> Iterator[tuple[int, str]]:
    """Iterate over text chunks with their indices.

    Args:
        text: Input text string.
        max_tokens: Maximum tokens per chunk.

    Yields:
        Tuples of (chunk_index, chunk_text).
    """
    chunks = chunk_text_by_length(text, max_tokens)
    for i, chunk in enumerate(chunks):
        yield i, chunk


def get_strategic_sample(text: str, max_tokens: int = 8000) -> str:
    """Get a strategic sample of text for world building.

    Samples from beginning, middle, and end of text to capture:
    - Character introductions (beginning)
    - Plot development and conflicts (middle)
    - Resolution and relationship outcomes (end)

    Args:
        text: Full input text.
        max_tokens: Maximum tokens for the combined sample.

    Returns:
        Strategic text sample.
    """
    total_tokens = estimate_tokens(text)

    # If text is short enough, use it all
    if total_tokens <= max_tokens:
        return text

    # Calculate how much to allocate to each section
    # Weight: 40% beginning, 30% middle, 30% end
    section_tokens = max_tokens // 3
    beginning_tokens = int(section_tokens * 1.2)  # Give beginning slightly more

    words = text.split()
    total_words = len(words)

    # Calculate word positions for each section
    words_per_token = total_words / total_tokens
    beginning_words = int(beginning_tokens * words_per_token)
    middle_words = int(section_tokens * words_per_token)
    end_words = int(section_tokens * words_per_token)

    # Extract sections
    beginning_start = 0
    beginning_end = min(beginning_words, total_words)

    middle_start = total_words // 2 - middle_words // 2
    middle_end = min(middle_start + middle_words, total_words)

    end_start = max(0, total_words - end_words)
    end_end = total_words

    # Build sample with section markers
    sections = []

    # Beginning section
    if beginning_end > 0:
        beginning_text = " ".join(words[beginning_start:beginning_end])
        sections.append(f"[BEGINNING]\n{beginning_text}")

    # Middle section
    if middle_end > middle_start:
        middle_text = " ".join(words[middle_start:middle_end])
        sections.append(f"[MIDDLE]\n{middle_text}")

    # End section
    if end_end > end_start and end_start > middle_end:
        end_text = " ".join(words[end_start:end_end])
        sections.append(f"[END]\n{end_text}")

    return "\n\n---\n\n".join(sections)
