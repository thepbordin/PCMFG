"""Checkpoint management for PCMFG analysis.

Provides checkpoint save/resume functionality to allow long-running
analysis jobs to be interrupted and resumed without losing progress.

Checkpoint Structure:
- Phase 1 (World Builder): Saved after completion
- Phase 2 (Emotion Extraction): Saved after each chunk batch
- Phase 3 (Synthesis): Not checkpointed (fast, deterministic)
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from pcmfg.models.schemas import (
    ChunkAnalysis,
    WorldBuilderOutput,
)

logger = logging.getLogger(__name__)

CHECKPOINT_VERSION = 1
CHECKPOINT_FILE = "pcmfg_checkpoint.json"


class CheckpointData(BaseModel):
    """Checkpoint data structure."""

    version: int = CHECKPOINT_VERSION
    created_at: str = ""
    updated_at: str = ""
    source_file: str = ""
    source_hash: str = ""  # First 8 chars of hash for validation

    # Phase 1: World Builder
    phase1_complete: bool = False
    world_builder: dict[str, Any] | None = None

    # Phase 2: Emotion Extraction
    phase2_complete: bool = False
    chunks: list[dict[str, Any]] = []
    total_chunks: int = 0
    processed_chunks: int = 0

    # Config snapshot
    config: dict[str, Any] = {}


def compute_text_hash(text: str) -> str:
    """Compute a hash of text for checkpoint validation.

    Args:
        text: Input text.

    Returns:
        First 8 characters of hash.
    """
    import hashlib

    return hashlib.sha256(text.encode()).hexdigest()[:8]


class CheckpointManager:
    """Manages checkpoint save/load for PCMFG analysis."""

    def __init__(self, checkpoint_dir: Path | str = ".pcmfg_checkpoints") -> None:
        """Initialize checkpoint manager.

        Args:
            checkpoint_dir: Directory to store checkpoint files.
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def get_checkpoint_path(self, source_hash: str) -> Path:
        """Get checkpoint file path for a source.

        Args:
            source_hash: Hash of source text.

        Returns:
            Path to checkpoint file.
        """
        return self.checkpoint_dir / f"checkpoint_{source_hash}.json"

    def save_checkpoint(self, data: CheckpointData) -> Path:
        """Save checkpoint data to file.

        Args:
            data: Checkpoint data to save.

        Returns:
            Path to saved checkpoint file.
        """
        data.updated_at = datetime.now(timezone.utc).isoformat()
        checkpoint_path = self.get_checkpoint_path(data.source_hash)

        with open(checkpoint_path, "w", encoding="utf-8") as f:
            f.write(data.model_dump_json(indent=2))

        logger.info(f"Checkpoint saved: {checkpoint_path}")
        return checkpoint_path

    def load_checkpoint(self, source_hash: str) -> CheckpointData | None:
        """Load checkpoint data from file.

        Args:
            source_hash: Hash of source text.

        Returns:
            CheckpointData if exists and valid, None otherwise.
        """
        checkpoint_path = self.get_checkpoint_path(source_hash)

        if not checkpoint_path.exists():
            return None

        try:
            with open(checkpoint_path, encoding="utf-8") as f:
                raw_data = json.load(f)

            data = CheckpointData(**raw_data)

            # Validate version
            if data.version != CHECKPOINT_VERSION:
                logger.warning(
                    f"Checkpoint version mismatch: {data.version} != {CHECKPOINT_VERSION}"
                )
                return None

            logger.info(f"Checkpoint loaded: {checkpoint_path}")
            return data

        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            return None

    def delete_checkpoint(self, source_hash: str) -> None:
        """Delete checkpoint file.

        Args:
            source_hash: Hash of source text.
        """
        checkpoint_path = self.get_checkpoint_path(source_hash)
        if checkpoint_path.exists():
            checkpoint_path.unlink()
            logger.info(f"Checkpoint deleted: {checkpoint_path}")

    def has_valid_checkpoint(self, source_hash: str) -> bool:
        """Check if a valid checkpoint exists.

        Args:
            source_hash: Hash of source text.

        Returns:
            True if valid checkpoint exists.
        """
        return self.load_checkpoint(source_hash) is not None

    def save_phase1(
        self,
        source_hash: str,
        source_file: str,
        world_builder: WorldBuilderOutput,
        config: dict[str, Any],
    ) -> Path:
        """Save checkpoint after Phase 1 completion.

        Args:
            source_hash: Hash of source text.
            source_file: Source file path.
            world_builder: World builder output.
            config: Configuration snapshot.

        Returns:
            Path to saved checkpoint.
        """
        data = CheckpointData(
            source_hash=source_hash,
            source_file=source_file,
            config=config,
            phase1_complete=True,
            world_builder=world_builder.model_dump(),
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        return self.save_checkpoint(data)

    def save_phase2_progress(
        self,
        data: CheckpointData,
        chunks: list[ChunkAnalysis],
        total_chunks: int,
    ) -> Path:
        """Save checkpoint during Phase 2 progress.

        Args:
            data: Existing checkpoint data.
            chunks: List of processed chunks.
            total_chunks: Total number of chunks.

        Returns:
            Path to saved checkpoint.
        """
        data.chunks = [c.model_dump() for c in chunks]
        data.total_chunks = total_chunks
        data.processed_chunks = len(chunks)
        data.phase2_complete = len(chunks) >= total_chunks
        return self.save_checkpoint(data)

    def load_world_builder(self, data: CheckpointData) -> WorldBuilderOutput | None:
        """Load world builder from checkpoint.

        Args:
            data: Checkpoint data.

        Returns:
            WorldBuilderOutput if available, None otherwise.
        """
        if not data.phase1_complete or data.world_builder is None:
            return None

        try:
            return WorldBuilderOutput(**data.world_builder)
        except Exception as e:
            logger.error(f"Failed to load world builder from checkpoint: {e}")
            return None

    def load_chunks(self, data: CheckpointData) -> list[ChunkAnalysis]:
        """Load chunks from checkpoint.

        Args:
            data: Checkpoint data.

        Returns:
            List of ChunkAnalysis objects.
        """
        chunks = []
        for chunk_data in data.chunks:
            try:
                chunks.append(ChunkAnalysis(**chunk_data))
            except Exception as e:
                logger.warning(f"Failed to load chunk from checkpoint: {e}")
        return chunks


def list_checkpoints(checkpoint_dir: Path | str = ".pcmfg_checkpoints") -> list[dict]:
    """List all available checkpoints.

    Args:
        checkpoint_dir: Directory containing checkpoints.

    Returns:
        List of checkpoint info dictionaries.
    """
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.exists():
        return []

    checkpoints = []
    for cp_file in checkpoint_dir.glob("checkpoint_*.json"):
        try:
            with open(cp_file, encoding="utf-8") as f:
                data = json.load(f)
            checkpoints.append(
                {
                    "path": str(cp_file),
                    "source_file": data.get("source_file", "unknown"),
                    "source_hash": data.get("source_hash", "unknown"),
                    "phase1_complete": data.get("phase1_complete", False),
                    "phase2_complete": data.get("phase2_complete", False),
                    "processed_chunks": data.get("processed_chunks", 0),
                    "total_chunks": data.get("total_chunks", 0),
                    "updated_at": data.get("updated_at", "unknown"),
                }
            )
        except Exception:
            continue

    return checkpoints
