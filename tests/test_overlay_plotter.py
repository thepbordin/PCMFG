"""Tests for NarrativeOverlayPlotter - TDD RED phase."""

import pytest
from pathlib import Path

from pcmfg.visualization.overlay_plotter import NarrativeOverlayPlotter


class TestNarrativeOverlayPlotterInit:
    """Tests for NarrativeOverlayPlotter initialization and core methods."""

    def test_class_instantiates_with_defaults(self) -> None:
        """NarrativeOverlayPlotter can be created with default parameters."""
        plotter = NarrativeOverlayPlotter()
        assert plotter.dpi == 300
        assert plotter.figsize == (16, 20)

    def test_class_instantiates_with_custom_params(self) -> None:
        """NarrativeOverlayPlotter accepts custom dpi and figsize."""
        plotter = NarrativeOverlayPlotter(dpi=150, figsize=(10, 12))
        assert plotter.dpi == 150
        assert plotter.figsize == (10, 12)

    def test_group_trajectories_groups_by_source(self) -> None:
        """_group_trajectories correctly groups flat trajectory list by source."""
        from pcmfg.models.schemas import NormalizedTrajectory

        trajectories = [
            NormalizedTrajectory(
                source="story_a.txt",
                main_pairing=["A", "B"],
                direction="A_to_B",
                emotion="Joy",
                x=[0.0, 0.5, 1.0],
                y=[1.0, 2.0, 3.0],
                original_length=3,
                n_points=3,
            ),
            NormalizedTrajectory(
                source="story_a.txt",
                main_pairing=["A", "B"],
                direction="B_to_A",
                emotion="Joy",
                x=[0.0, 0.5, 1.0],
                y=[1.0, 1.0, 2.0],
                original_length=3,
                n_points=3,
            ),
            NormalizedTrajectory(
                source="story_b.txt",
                main_pairing=["C", "D"],
                direction="A_to_B",
                emotion="Joy",
                x=[0.0, 0.5, 1.0],
                y=[3.0, 2.0, 1.0],
                original_length=3,
                n_points=3,
            ),
        ]

        plotter = NarrativeOverlayPlotter()
        grouped = plotter._group_trajectories(trajectories)

        assert "story_a.txt" in grouped
        assert "story_b.txt" in grouped
        assert ("A_to_B", "Joy") in grouped["story_a.txt"]
        assert ("B_to_A", "Joy") in grouped["story_a.txt"]
        assert ("A_to_B", "Joy") in grouped["story_b.txt"]

    def test_unpack_barycenter_correct_column_ordering(self) -> None:
        """_unpack_barycenter uses correct emotion_idx*2+dir_offset column indexing."""
        import numpy as np

        plotter = NarrativeOverlayPlotter()
        # Create a barycenter where each column has a unique value
        n_points = 5
        barycenter = np.zeros((n_points, 18), dtype=np.float64)
        for col in range(18):
            barycenter[:, col] = float(col + 1)

        unpacked = plotter._unpack_barycenter(barycenter)

        # Joy (idx=0): A_to_B col=0 (val=1), B_to_A col=1 (val=2)
        import numpy.testing as npt

        npt.assert_array_equal(unpacked[("A_to_B", "Joy")], [1.0, 1.0, 1.0, 1.0, 1.0])
        npt.assert_array_equal(unpacked[("B_to_A", "Joy")], [2.0, 2.0, 2.0, 2.0, 2.0])

        # Trust (idx=1): A_to_B col=2 (val=3), B_to_A col=3 (val=4)
        npt.assert_array_equal(unpacked[("A_to_B", "Trust")], [3.0, 3.0, 3.0, 3.0, 3.0])
        npt.assert_array_equal(unpacked[("B_to_A", "Trust")], [4.0, 4.0, 4.0, 4.0, 4.0])

        # Arousal (idx=8): A_to_B col=16 (val=17), B_to_A col=17 (val=18)
        npt.assert_array_equal(
            unpacked[("A_to_B", "Arousal")], [17.0, 17.0, 17.0, 17.0, 17.0]
        )
        npt.assert_array_equal(
            unpacked[("B_to_A", "Arousal")], [18.0, 18.0, 18.0, 18.0, 18.0]
        )
