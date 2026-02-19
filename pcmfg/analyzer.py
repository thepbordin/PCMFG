"""Main PCMFG analyzer orchestrating the 3-phase pipeline.

This module provides the main entry point for analyzing romantic narratives
through the complete PCMFG pipeline:

1. Phase 1: Extraction - World building and emotion extraction
2. Phase 2: Normalization - Validation and aggregation
3. Phase 3: Axis Mapping - Romance axis computation
"""

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from pcmfg.config import Config
from pcmfg.llm.anthropic_client import AnthropicLLMClient
from pcmfg.llm.openai_client import OpenAIClient
from pcmfg.models.schemas import (
    AnalysisMetadata,
    AnalysisResult,
    ChunkAnalysis,
    WorldBuilderOutput,
)
from pcmfg.phase1.emotion_extractor import EmotionExtractor
from pcmfg.phase1.world_builder import WorldBuilder, WorldBuilderError
from pcmfg.phase2.normalizer import EmotionNormalizer
from pcmfg.phase3.axis_mapper import AxisMapper
from pcmfg.utils.text_processing import (
    chunk_text_by_chapter,
    chunk_text_by_length,
    clean_text,
    estimate_tokens,
)

logger = logging.getLogger(__name__)


class PCMFGAnalyzer:
    """Main orchestrator for the PCMFG analysis pipeline.

    Coordinates all three phases:
    - Phase 1: Extraction (World Builder + Emotion Extractor)
    - Phase 2: Normalization (validation)
    - Phase 3: Axis Mapping (romance axes)
    """

    def __init__(
        self,
        llm_client: Any | None = None,
        config: Config | None = None,
        api_key: str | None = None,
        provider: Literal["openai", "anthropic"] | None = None,
        model: str | None = None,
    ) -> None:
        """Initialize the PCMFG analyzer.

        Args:
            llm_client: Pre-configured LLM client. If provided, other LLM args are ignored.
            config: Configuration object. If None, uses defaults.
            api_key: API key for LLM provider (overrides config).
            provider: LLM provider ("openai" or "anthropic").
            model: Model name to use.
        """
        self.config = config or Config()

        # Set up LLM client
        if llm_client is not None:
            self.llm_client = llm_client
        else:
            self.llm_client = self._create_llm_client(api_key, provider, model)

        # Initialize phase processors
        self.world_builder = WorldBuilder(self.llm_client)
        self.normalizer = EmotionNormalizer()
        self.axis_mapper = AxisMapper()

        # Emotion extractor is created after world building
        self.emotion_extractor: EmotionExtractor | None = None

    def _create_llm_client(
        self,
        api_key: str | None,
        provider: Literal["openai", "anthropic"] | None,
        model: str | None,
    ) -> Any:
        """Create an LLM client based on configuration.

        Args:
            api_key: API key override.
            provider: Provider override.
            model: Model name override.

        Returns:
            Configured LLM client.
        """
        # Use overrides or fall back to config
        actual_provider = provider or self.config.llm.provider
        actual_model = model or self.config.llm.model
        temperature = self.config.llm.temperature
        max_tokens = self.config.llm.max_tokens

        if actual_provider == "openai":
            return OpenAIClient(
                api_key=api_key,
                model=actual_model,
                temperature=temperature,
                max_tokens=max_tokens,
                base_url=self.config.llm.base_url,
            )
        elif actual_provider == "anthropic":
            return AnthropicLLMClient(
                api_key=api_key,
                model=actual_model,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        else:
            raise ValueError(f"Unknown LLM provider: {actual_provider}")

    def analyze(self, text: str, source: str = "") -> AnalysisResult:
        """Analyze a romantic narrative text.

        Runs the complete 3-phase pipeline:
        1. Extraction: World building + emotion extraction
        2. Normalization: Validation
        3. Axis Mapping: Compute romance axes

        Args:
            text: Input text to analyze.
            source: Source identifier (filename, etc.).

        Returns:
            AnalysisResult with complete analysis data.
        """
        logger.info("Starting PCMFG analysis")

        # Clean input text
        text = clean_text(text)

        # Phase 1: Extraction
        logger.info("Phase 1: Extraction")
        world = self._run_world_builder(text)
        self.emotion_extractor = EmotionExtractor(self.llm_client, world)
        chunks = self._extract_emotions(text)

        # Phase 2: Normalization
        logger.info("Phase 2: Normalization")
        chunks = self.normalizer.normalize_all(chunks)

        # Phase 3: Axis Mapping
        logger.info("Phase 3: Axis Mapping")
        axes = self.axis_mapper.map_chunks(chunks)

        # Build metadata
        metadata = AnalysisMetadata(
            source=source,
            analysis_date=datetime.now(timezone.utc),
            model=self.config.llm.model,
            total_chunks=len(chunks),
            provider=self.config.llm.provider,
        )

        logger.info(f"Analysis complete: {len(chunks)} chunks processed")

        return AnalysisResult(
            metadata=metadata,
            world_builder=world,
            chunks=chunks,
            axes=axes,
        )

    def _run_world_builder(self, text: str) -> WorldBuilderOutput:
        """Run the world builder agent.

        For very long texts, uses a sample to extract world info.

        Args:
            text: Full input text.

        Returns:
            WorldBuilderOutput with extracted world info.
        """
        # Use first ~3000 tokens for world building
        max_sample_tokens = 3000
        if estimate_tokens(text) > max_sample_tokens:
            # Take first portion of text
            sample_text = self._get_text_sample(text, max_sample_tokens)
        else:
            sample_text = text

        try:
            return self.world_builder.build(sample_text)
        except WorldBuilderError as e:
            logger.error(f"World builder failed: {e}")
            # Return minimal world builder output
            return WorldBuilderOutput(
                main_pairing=["Character A", "Character B"],
                aliases={},
                world_guidelines=[],
                mermaid_graph="",
            )

    def _get_text_sample(self, text: str, max_tokens: int) -> str:
        """Get a sample of text up to max_tokens.

        Args:
            text: Full text.
            max_tokens: Maximum tokens for sample.

        Returns:
            Text sample.
        """
        max_words = int(max_tokens / 1.3)
        words = text.split()
        return " ".join(words[:max_words])

    def _extract_emotions(self, text: str) -> list[ChunkAnalysis]:
        """Extract emotions from all text chunks.

        Uses parallel processing with configurable concurrency.

        Args:
            text: Full input text.

        Returns:
            List of ChunkAnalysis objects.
        """
        from pcmfg.utils.parallel import ParallelProcessor, ProcessingResult

        # Chunk text based on config
        chunks_text = self._chunk_text(text)
        total_chunks = len(chunks_text)

        # emotion_extractor is guaranteed to be set after _run_world_builder
        assert self.emotion_extractor is not None
        extractor = self.emotion_extractor  # Local reference for type safety

        # Create processor function
        def process_chunk(args: tuple[str, int, float]) -> ChunkAnalysis:
            chunk_text, chunk_id, position = args
            return extractor.extract(chunk_text, chunk_id, position)

        # Prepare chunk data
        chunk_items = [
            (chunk_text, i, i / total_chunks if total_chunks > 0 else 0.0)
            for i, chunk_text in enumerate(chunks_text)
        ]

        # Progress tracking
        completed_count = 0

        def on_progress(completed: int, total: int) -> None:
            nonlocal completed_count
            if completed > completed_count:
                logger.info(f"Processing chunks: {completed}/{total}")
                completed_count = completed

        def on_error(index: int, error: Exception) -> None:
            logger.error(f"Failed to process chunk {index}: {error}")

        # Process in parallel
        processor = ParallelProcessor(
            process_fn=process_chunk,
            max_concurrency=self.config.processing.max_concurrency,
            on_progress=on_progress,
            on_error=on_error,
        )

        results = processor.process(chunk_items)

        # Extract successful results, create defaults for failures
        chunk_analyses: list[ChunkAnalysis] = []
        for result in results:
            if result.success and result.result is not None:
                chunk_analyses.append(result.result)
            else:
                # Create default chunk for failed extraction
                default_chunk = self.emotion_extractor._create_default_chunk(
                    result.index,
                    result.index / total_chunks if total_chunks > 0 else 0.0,
                    str(result.error) if result.error else "Unknown error",
                )
                chunk_analyses.append(default_chunk)

        return chunk_analyses

    def _chunk_text(self, text: str) -> list[str]:
        """Split text into chunks based on config settings.

        Args:
            text: Input text.

        Returns:
            List of text chunks.
        """
        beat_detection = self.config.processing.beat_detection
        max_tokens = self.config.processing.max_chunk_tokens

        if beat_detection == "chapter":
            return chunk_text_by_chapter(text, max_tokens)
        elif beat_detection == "length":
            return chunk_text_by_length(
                text,
                max_tokens=max_tokens,
                min_chunk_tokens=self.config.processing.min_beat_length,
            )
        else:  # automatic
            # Try chapter-based first, fall back to length-based
            chunks = chunk_text_by_chapter(text, max_tokens)
            if len(chunks) <= 1:
                # No chapters found, use length-based
                chunks = chunk_text_by_length(
                    text,
                    max_tokens=max_tokens,
                    min_chunk_tokens=self.config.processing.min_beat_length,
                )
            return chunks

    def analyze_file(self, file_path: str | Path) -> AnalysisResult:
        """Analyze a text file.

        Args:
            file_path: Path to the text file.

        Returns:
            AnalysisResult with complete analysis data.
        """
        file_path = Path(file_path)

        with open(file_path, encoding="utf-8") as f:
            text = f.read()

        return self.analyze(text, source=str(file_path))


def analyze(
    text: str,
    provider: Literal["openai", "anthropic"] = "openai",
    model: str | None = None,
    config: Config | None = None,
) -> AnalysisResult:
    """Convenience function to analyze text with minimal setup.

    Args:
        text: Input text to analyze.
        provider: LLM provider to use.
        model: Model name (optional).
        config: Configuration object (optional).

    Returns:
        AnalysisResult with complete analysis data.
    """
    config = config or Config()
    if model:
        config.llm.model = model
    config.llm.provider = provider

    analyzer = PCMFGAnalyzer(config=config)
    return analyzer.analyze(text)
