"""Tests for checkpoint functionality."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from pcmfg.checkpoint import (
    CheckpointData,
    CheckpointManager,
    compute_text_hash,
    list_checkpoints,
)
from pcmfg.models.schemas import WorldBuilderOutput


class TestComputeTextHash:
    """Tests for compute_text_hash function."""

    def test_consistent_hash(self) -> None:
        """Test that same text produces same hash."""
        text = "Hello world"
        hash1 = compute_text_hash(text)
        hash2 = compute_text_hash(text)
        assert hash1 == hash2

    def test_different_text_different_hash(self) -> None:
        """Test that different texts produce different hashes."""
        hash1 = compute_text_hash("Hello world")
        hash2 = compute_text_hash("Goodbye world")
        assert hash1 != hash2

    def test_hash_length(self) -> None:
        """Test that hash is 8 characters."""
        text = "Some text"
        hash_result = compute_text_hash(text)
        assert len(hash_result) == 8


class TestCheckpointData:
    """Tests for CheckpointData model."""

    def test_default_values(self) -> None:
        """Test default values for CheckpointData."""
        data = CheckpointData()
        assert data.version == 1
        assert data.phase1_complete is False
        assert data.phase2_complete is False
        assert data.chunks == []
        assert data.world_builder is None

    def test_custom_values(self) -> None:
        """Test custom values for CheckpointData."""
        data = CheckpointData(
            source_hash="abc123",
            source_file="test.txt",
            phase1_complete=True,
            world_builder={"main_pairing": ["A", "B"], "aliases": {}},
        )
        assert data.source_hash == "abc123"
        assert data.phase1_complete is True


class TestCheckpointManager:
    """Tests for CheckpointManager class."""

    def test_save_and_load_checkpoint(self, tmp_path: Path) -> None:
        """Test saving and loading a checkpoint."""
        manager = CheckpointManager(checkpoint_dir=tmp_path)

        data = CheckpointData(
            source_hash="test123",
            source_file="test.txt",
            phase1_complete=True,
        )

        # Save checkpoint
        saved_path = manager.save_checkpoint(data)
        assert saved_path.exists()

        # Load checkpoint
        loaded = manager.load_checkpoint("test123")
        assert loaded is not None
        assert loaded.source_hash == "test123"
        assert loaded.phase1_complete is True

    def test_load_nonexistent_checkpoint(self, tmp_path: Path) -> None:
        """Test loading a checkpoint that doesn't exist."""
        manager = CheckpointManager(checkpoint_dir=tmp_path)
        loaded = manager.load_checkpoint("nonexistent")
        assert loaded is None

    def test_delete_checkpoint(self, tmp_path: Path) -> None:
        """Test deleting a checkpoint."""
        manager = CheckpointManager(checkpoint_dir=tmp_path)

        data = CheckpointData(source_hash="delete_me")
        manager.save_checkpoint(data)

        # Verify it exists
        assert manager.has_valid_checkpoint("delete_me")

        # Delete it
        manager.delete_checkpoint("delete_me")

        # Verify it's gone
        assert not manager.has_valid_checkpoint("delete_me")

    def test_save_phase1(self, tmp_path: Path) -> None:
        """Test saving Phase 1 checkpoint."""
        manager = CheckpointManager(checkpoint_dir=tmp_path)

        world = WorldBuilderOutput(
            main_pairing=["Alice", "Bob"],
            aliases={"Alice": ["Ali"], "Bob": ["Bobby"]},
            core_conflict="They hate each other.",
            world_guidelines=["Rule 1"],
            mermaid_graph="graph TD",
        )

        path = manager.save_phase1(
            source_hash="test123",
            source_file="novel.txt",
            world_builder=world,
            config={"llm": {"model": "gpt-4"}},
        )

        assert path.exists()

        # Verify can load
        loaded = manager.load_checkpoint("test123")
        assert loaded is not None
        assert loaded.phase1_complete is True
        assert loaded.world_builder is not None
        assert loaded.world_builder["main_pairing"] == ["Alice", "Bob"]

    def test_load_world_builder(self, tmp_path: Path) -> None:
        """Test loading world builder from checkpoint."""
        manager = CheckpointManager(checkpoint_dir=tmp_path)

        world = WorldBuilderOutput(
            main_pairing=["Alice", "Bob"],
            aliases={},
            core_conflict="",
            world_guidelines=[],
            mermaid_graph="",
        )

        manager.save_phase1("test123", "novel.txt", world, {})

        data = manager.load_checkpoint("test123")
        assert data is not None

        loaded_world = manager.load_world_builder(data)
        assert loaded_world is not None
        assert loaded_world.main_pairing == ["Alice", "Bob"]


class TestListCheckpoints:
    """Tests for list_checkpoints function."""

    def test_empty_directory(self, tmp_path: Path) -> None:
        """Test listing checkpoints in empty directory."""
        checkpoints = list_checkpoints(checkpoint_dir=tmp_path)
        assert checkpoints == []

    def test_list_multiple_checkpoints(self, tmp_path: Path) -> None:
        """Test listing multiple checkpoints."""
        manager = CheckpointManager(checkpoint_dir=tmp_path)

        # Create multiple checkpoints
        for i in range(3):
            data = CheckpointData(
                source_hash=f"hash{i}",
                source_file=f"novel{i}.txt",
                phase1_complete=True,
                processed_chunks=i * 10,
                total_chunks=100,
            )
            manager.save_checkpoint(data)

        checkpoints = list_checkpoints(checkpoint_dir=tmp_path)
        assert len(checkpoints) == 3

        # Verify checkpoint info
        sources = {cp["source_file"] for cp in checkpoints}
        assert "novel0.txt" in sources
        assert "novel1.txt" in sources
        assert "novel2.txt" in sources
