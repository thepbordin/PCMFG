"""Main PCMFG analyzer orchestrating the 3-phase pipeline.

This module provides the main entry point for analyzing romantic narratives
through the complete PCMFG pipeline:

1. Phase 1: World Builder (Agent 1 - LLM)
2. Phase 2: Emotion Extraction (Agent 2 - LLM loop)
3. Phase 3: Synthesis (Deterministic Python - forward fill + time-series)
"""

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from pcmfg.checkpoint import CheckpointData, CheckpointManager, compute_text_hash
from pcmfg.config import Config
from pcmfg.llm.anthropic_client import AnthropicLLMClient
from pcmfg.llm.openai_client import OpenAIClient
from pcmfg.models.schemas import (
    AnalysisMetadata,
    AnalysisResult,
    ChunkAnalysis,
    WorldBuilderOutput,
)
from pcmfg.phase1.emotion_extractor import EmotionExtractor, should_process_chunk
from pcmfg.phase1.world_builder import WorldBuilder, WorldBuilderError
from pcmfg.phase3.synthesizer import Synthesizer
from pcmfg.utils.text_processing import (
    chunk_text_by_chapter,
    chunk_text_by_length,
    clean_text,
    estimate_tokens,
    get_strategic_sample,
)

logger = logging.getLogger(__name__)


class PCMFGAnalyzer:
    """Main orchestrator for the PCMFG analysis pipeline.

    Coordinates all three phases:
    - Phase 1: World Builder (Agent 1 - LLM)
    - Phase 2: Emotion Extraction (Agent 2 - LLM loop)
    - Phase 3: Synthesis (Deterministic Python)
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
        self.synthesizer = Synthesizer()

        # Emotion extractor is created after world building
        self.emotion_extractor: EmotionExtractor | None = None

        # Checkpoint manager
        self.checkpoint_manager = CheckpointManager()

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
        1. Phase 1: World Builder (Agent 1)
        2. Phase 2: Emotion Extraction (Agent 2 loop)
        3. Phase 3: Synthesis (forward fill + time-series)

        Args:
            text: Input text to analyze.
            source: Source identifier (filename, etc.).

        Returns:
            AnalysisResult with complete analysis data including raw emotion time-series.
        """
        logger.info("Starting PCMFG analysis")

        # Clean input text
        text = clean_text(text)

        # Phase 1: World Builder (Agent 1)
        logger.info("Phase 1: World Builder")
        world = self._run_world_builder(text)
        self.emotion_extractor = EmotionExtractor(self.llm_client, world)

        # Phase 2: Emotion Extraction (Agent 2 loop)
        logger.info("Phase 2: Emotion Extraction")
        chunks = self._extract_emotions(text)

        # Build metadata
        metadata = AnalysisMetadata(
            source=source,
            analysis_date=datetime.now(timezone.utc),
            model=self.config.llm.model,
            total_chunks=len(chunks),
            provider=self.config.llm.provider,
        )

        # Phase 3: Synthesis (forward fill + time-series)
        logger.info("Phase 3: Synthesis")
        result = self.synthesizer.synthesize(chunks, world, metadata)

        logger.info(f"Analysis complete: {len(chunks)} chunks processed")

        return result

    def analyze_with_checkpoint(
        self,
        text: str,
        source: str = "",
        resume: bool = True,
        checkpoint_interval: int = 10,
    ) -> AnalysisResult:
        """Analyze text with checkpoint support for resumable processing.

        Saves checkpoints after Phase 1 and periodically during Phase 2.
        Can resume from checkpoint if interrupted.

        Args:
            text: Input text to analyze.
            source: Source identifier (filename, etc.).
            resume: If True, attempt to resume from existing checkpoint.
            checkpoint_interval: Save checkpoint every N chunks during Phase 2.

        Returns:
            AnalysisResult with complete analysis data.

        Raises:
            ValueError: If checkpoint exists but resume=False.
        """
        # Clean input text
        text = clean_text(text)
        source_hash = compute_text_hash(text)

        # Try to load existing checkpoint
        checkpoint_data = None
        if resume:
            checkpoint_data = self.checkpoint_manager.load_checkpoint(source_hash)
            if checkpoint_data:
                logger.info(
                    f"Resuming from checkpoint: Phase 1={checkpoint_data.phase1_complete}, "
                    f"Phase 2 chunks={checkpoint_data.processed_chunks}/{checkpoint_data.total_chunks}"
                )

        # Initialize checkpoint data if needed
        if checkpoint_data is None:
            checkpoint_data = CheckpointData(
                source_hash=source_hash,
                source_file=source,
                config=self.config.model_dump(),
            )

        # Phase 1: World Builder (Agent 1)
        world: WorldBuilderOutput
        if checkpoint_data.phase1_complete and checkpoint_data.world_builder:
            world = self.checkpoint_manager.load_world_builder(checkpoint_data)
            if world is None:
                logger.warning(
                    "Failed to load world builder from checkpoint, re-running Phase 1"
                )
                world = self._run_world_builder(text)
                self._save_phase1_checkpoint(checkpoint_data, source, world)
            else:
                logger.info("Phase 1: World Builder (from checkpoint)")
        else:
            logger.info("Phase 1: World Builder")
            world = self._run_world_builder(text)
            self._save_phase1_checkpoint(checkpoint_data, source, world)

        self.emotion_extractor = EmotionExtractor(self.llm_client, world)

        # Phase 2: Emotion Extraction (Agent 2 loop) with checkpointing
        logger.info("Phase 2: Emotion Extraction")
        chunks = self._extract_emotions_with_checkpoint(
            text, checkpoint_data, checkpoint_interval
        )

        # Build metadata
        metadata = AnalysisMetadata(
            source=source,
            analysis_date=datetime.now(timezone.utc),
            model=self.config.llm.model,
            total_chunks=len(chunks),
            provider=self.config.llm.provider,
        )

        # Phase 3: Synthesis (forward fill + time-series)
        logger.info("Phase 3: Synthesis")
        result = self.synthesizer.synthesize(chunks, world, metadata)

        # Clean up checkpoint on success
        self.checkpoint_manager.delete_checkpoint(source_hash)

        logger.info(f"Analysis complete: {len(chunks)} chunks processed")

        return result

    def _save_phase1_checkpoint(
        self,
        checkpoint_data: CheckpointData,
        source: str,
        world: WorldBuilderOutput,
    ) -> None:
        """Save checkpoint after Phase 1 completion."""
        checkpoint_data.source_file = source
        checkpoint_data.phase1_complete = True
        checkpoint_data.world_builder = world.model_dump()
        self.checkpoint_manager.save_checkpoint(checkpoint_data)

    def _extract_emotions_with_checkpoint(
        self,
        text: str,
        checkpoint_data: CheckpointData,
        checkpoint_interval: int = 10,
    ) -> list[ChunkAnalysis]:
        """Extract emotions with periodic checkpoint saves.

        Args:
            text: Full input text.
            checkpoint_data: Checkpoint data to update.
            checkpoint_interval: Save checkpoint every N chunks.

        Returns:
            List of ChunkAnalysis objects.
        """
        from pcmfg.utils.parallel import ParallelProcessor

        # Chunk text based on config
        chunks_text = self._chunk_text(text)
        total_chunks = len(chunks_text)

        # Load previously processed chunks
        existing_chunks = self.checkpoint_manager.load_chunks(checkpoint_data)
        processed_ids = {c.chunk_id for c in existing_chunks}

        if processed_ids:
            logger.info(
                f"Resuming with {len(processed_ids)} previously processed chunks"
            )

        # emotion_extractor is guaranteed to be set after _run_world_builder
        assert self.emotion_extractor is not None
        extractor = self.emotion_extractor
        world = extractor.world

        # Track skipped chunks for logging
        skipped_count = 0

        # Create processor function with chunk filtering
        def process_chunk(args: tuple[str, int, float]) -> ChunkAnalysis | None:
            nonlocal skipped_count
            chunk_text, chunk_id, position = args

            # Skip already processed chunks
            if chunk_id in processed_ids:
                return None

            # Token efficiency: skip chunks with no relevant characters
            if not should_process_chunk(chunk_text, world.aliases):
                logger.debug(f"Skipping chunk {chunk_id}: no character names found")
                skipped_count += 1
                return None

            return extractor.extract(chunk_text, chunk_id, position)

        # Prepare chunk data (filter out already processed)
        chunk_items = [
            (chunk_text, i, i / total_chunks if total_chunks > 0 else 0.0)
            for i, chunk_text in enumerate(chunks_text)
            if i not in processed_ids
        ]

        logger.info(
            f"Processing {len(chunk_items)} remaining chunks (total: {total_chunks})"
        )

        # Progress tracking
        completed_count = len(processed_ids)
        chunks_since_checkpoint = 0

        def on_progress(completed: int, total: int) -> None:
            nonlocal completed_count, chunks_since_checkpoint
            actual_completed = completed + len(processed_ids)
            if actual_completed > completed_count:
                logger.info(f"Processing chunks: {actual_completed}/{total_chunks}")
                completed_count = actual_completed
                chunks_since_checkpoint += 1

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

        if skipped_count > 0:
            logger.info(f"Skipped {skipped_count} chunks (no character names)")

        # Build final chunk list: existing + newly processed
        chunk_analyses = list(existing_chunks)

        for result in results:
            if result.success and result.result is not None:
                chunk_analyses.append(result.result)
            elif result.index not in processed_ids:
                # Create default chunk for failed/skipped extraction
                actual_index = result.index
                position = actual_index / total_chunks if total_chunks > 0 else 0.0
                error_msg = (
                    str(result.error)
                    if result.error
                    else "Skipped (no character names)"
                )
                default_chunk = extractor._create_default_chunk(
                    actual_index,
                    position,
                    error_msg,
                )
                chunk_analyses.append(default_chunk)

            # Save checkpoint periodically
            if chunks_since_checkpoint >= checkpoint_interval:
                self.checkpoint_manager.save_phase2_progress(
                    checkpoint_data, chunk_analyses, total_chunks
                )
                chunks_since_checkpoint = 0

        # Sort by chunk_id to maintain order
        chunk_analyses.sort(key=lambda c: c.chunk_id)

        # Save final checkpoint for Phase 2
        self.checkpoint_manager.save_phase2_progress(
            checkpoint_data, chunk_analyses, total_chunks
        )

        return chunk_analyses

    def _run_world_builder(self, text: str) -> WorldBuilderOutput:
        """Run the world builder agent.

        For very long texts, uses strategic sampling (beginning, middle, end)
        to capture full story arc context.

        Args:
            text: Full input text.

        Returns:
            WorldBuilderOutput with extracted world info.
        """
        # Use strategic sampling for world building
        max_sample_tokens = self.config.processing.world_builder_sample_tokens
        sample_text = get_strategic_sample(text, max_sample_tokens)

        logger.info(
            f"World builder sample: {estimate_tokens(sample_text)} tokens "
            f"(from {estimate_tokens(text)} total)"
        )

        try:
            return self.world_builder.build(sample_text)
        except WorldBuilderError as e:
            logger.error(f"World builder failed: {e}")
            # Return minimal world builder output with default aliases
            # IMPORTANT: Include main_pairing as aliases so chunk filtering works
            default_pairing = ["Character A", "Character B"]
            return WorldBuilderOutput(
                main_pairing=default_pairing,
                aliases={
                    default_pairing[0]: [default_pairing[0]],
                    default_pairing[1]: [default_pairing[1]],
                },
                world_guidelines=[],
                mermaid_graph="",
            )

    def _get_text_sample(self, text: str, max_tokens: int) -> str:
        """Get a sample of text up to max_tokens (deprecated, use get_strategic_sample).

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
        Also implements token efficiency optimization by skipping chunks
        that don't contain any character names from the aliases list.

        Args:
            text: Full input text.

        Returns:
            List of ChunkAnalysis objects.
        """
        from pcmfg.utils.parallel import ParallelProcessor

        # Chunk text based on config
        chunks_text = self._chunk_text(text)
        total_chunks = len(chunks_text)

        # emotion_extractor is guaranteed to be set after _run_world_builder
        assert self.emotion_extractor is not None
        extractor = self.emotion_extractor  # Local reference for type safety
        world = extractor.world

        # Track skipped chunks for logging
        skipped_count = 0

        # Create processor function with chunk filtering
        def process_chunk(args: tuple[str, int, float]) -> ChunkAnalysis | None:
            nonlocal skipped_count
            chunk_text, chunk_id, position = args

            # Token efficiency: skip chunks with no relevant characters
            if not should_process_chunk(chunk_text, world.aliases):
                logger.debug(f"Skipping chunk {chunk_id}: no character names found")
                skipped_count += 1
                # Return None for skipped chunks - we'll create default chunks later
                return None

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

        if skipped_count > 0:
            logger.info(f"Skipped {skipped_count} chunks (no character names)")

        # Extract successful results, create defaults for failures/skipped
        chunk_analyses: list[ChunkAnalysis] = []
        for result in results:
            if result.success and result.result is not None:
                chunk_analyses.append(result.result)
            else:
                # Create default chunk for failed/skipped extraction
                position = result.index / total_chunks if total_chunks > 0 else 0.0
                error_msg = (
                    str(result.error)
                    if result.error
                    else "Skipped (no character names)"
                )
                default_chunk = extractor._create_default_chunk(
                    result.index,
                    position,
                    error_msg,
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
