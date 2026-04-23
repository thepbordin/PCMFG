"""Tests for NarrativeOverlayPlotter - VIS-01, VIS-02, VIS-03, VIS-04."""

import os

import pytest
from pathlib import Path

from pcmfg.models.schemas import BASE_EMOTIONS
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


class TestVIS01:
    """VIS-01: Overlay multiple normalized trajectories on same axes."""

    def test_overlay_creates_file(
        self,
        sample_normalized_trajectories_multi: list,
        tmp_path: Path,
    ) -> None:
        """plot_overlay() creates a PNG file at output_path."""
        plotter = NarrativeOverlayPlotter()
        output = tmp_path / "overlay.png"
        plotter.plot_overlay(sample_normalized_trajectories_multi, output)

        assert output.exists(), "Overlay PNG file should be created"
        assert output.stat().st_size > 0, "Overlay PNG file should not be empty"

    def test_overlay_multiple_narratives(
        self,
        sample_normalized_trajectories_multi: list,
        tmp_path: Path,
    ) -> None:
        """plot_overlay() with 3 trajectories produces a non-empty file."""
        plotter = NarrativeOverlayPlotter()
        output = tmp_path / "overlay_multi.png"
        plotter.plot_overlay(sample_normalized_trajectories_multi, output)

        assert output.exists()
        assert output.stat().st_size > 1000, "File should have reasonable size for 3 narratives"

    def test_overlay_with_cluster_coloring(
        self,
        sample_normalized_trajectories_multi: list,
        sample_dtw_cluster_result: object,
        tmp_path: Path,
    ) -> None:
        """plot_overlay() with cluster_result uses CLUSTER_COLORS and creates file."""
        from pcmfg.analysis.plotter import CLUSTER_COLORS

        plotter = NarrativeOverlayPlotter()
        output = tmp_path / "overlay_clustered.png"
        plotter.plot_overlay(
            sample_normalized_trajectories_multi,
            output,
            cluster_result=sample_dtw_cluster_result,
        )

        assert output.exists()
        assert output.stat().st_size > 0

    def test_overlay_without_cluster_uses_tab10(
        self,
        sample_normalized_trajectories_multi: list,
        tmp_path: Path,
    ) -> None:
        """plot_overlay() without cluster_result uses default tab10 coloring."""
        plotter = NarrativeOverlayPlotter()
        output = tmp_path / "overlay_tab10.png"
        plotter.plot_overlay(sample_normalized_trajectories_multi, output)

        assert output.exists()
        assert output.stat().st_size > 0


class TestVIS02:
    """VIS-02: Per-emotion overlay plots for all 9 base emotions."""

    def test_plot_emotion_creates_file(
        self,
        sample_normalized_trajectories_multi: list,
        tmp_path: Path,
    ) -> None:
        """plot_emotion('Joy', ...) creates a PNG file."""
        plotter = NarrativeOverlayPlotter()
        output = tmp_path / "overlay_joy.png"
        plotter.plot_emotion(sample_normalized_trajectories_multi, "Joy", output)

        assert output.exists()
        assert output.stat().st_size > 0

    def test_plot_all_emotions(
        self,
        sample_normalized_trajectories_multi: list,
        tmp_path: Path,
    ) -> None:
        """Can call plot_emotion for all 9 BASE_EMOTIONS without error."""
        plotter = NarrativeOverlayPlotter()

        for emotion in BASE_EMOTIONS:
            output = tmp_path / f"overlay_{emotion.lower()}.png"
            plotter.plot_emotion(
                sample_normalized_trajectories_multi, emotion, output
            )
            assert output.exists(), f"Emotion plot for {emotion} should exist"
            assert output.stat().st_size > 0


class TestVIS03:
    """VIS-03: Per-direction overlay plots for asymmetry analysis."""

    def test_plot_direction_creates_file(
        self,
        sample_normalized_trajectories_multi: list,
        tmp_path: Path,
    ) -> None:
        """plot_direction('A_to_B', ...) creates a PNG file."""
        plotter = NarrativeOverlayPlotter()
        output = tmp_path / "overlay_atob.png"
        plotter.plot_direction(
            sample_normalized_trajectories_multi, "A_to_B", output
        )

        assert output.exists()
        assert output.stat().st_size > 0

    def test_plot_direction_btoa(
        self,
        sample_normalized_trajectories_multi: list,
        tmp_path: Path,
    ) -> None:
        """plot_direction('B_to_A', ...) creates a PNG file."""
        plotter = NarrativeOverlayPlotter()
        output = tmp_path / "overlay_btoa.png"
        plotter.plot_direction(
            sample_normalized_trajectories_multi, "B_to_A", output
        )

        assert output.exists()
        assert output.stat().st_size > 0


class TestVIS04ClusterAndPlotAll:
    """VIS-04: Cluster-specific visualization and plot_all suite generation."""

    def test_cluster_plot_creates_file(
        self,
        sample_normalized_trajectories_multi: list,
        sample_dtw_cluster_result: object,
        tmp_path: Path,
    ) -> None:
        """plot_cluster(cluster_id=0) creates a PNG file for the specified cluster."""
        plotter = NarrativeOverlayPlotter()
        output = tmp_path / "overlay_cluster_0.png"
        plotter.plot_cluster(
            sample_normalized_trajectories_multi,
            sample_dtw_cluster_result,
            cluster_id=0,
            output_path=output,
        )

        assert output.exists(), "Cluster 0 plot should be created"
        assert output.stat().st_size > 0, "Cluster 0 plot should not be empty"

    def test_cluster_plot_shows_barycenter(
        self,
        sample_normalized_trajectories_multi: list,
        sample_dtw_cluster_result: object,
        tmp_path: Path,
    ) -> None:
        """plot_cluster exercises barycenter logic and creates file for both clusters."""
        plotter = NarrativeOverlayPlotter()
        for cid in [0, 1]:
            output = tmp_path / f"overlay_cluster_{cid}.png"
            plotter.plot_cluster(
                sample_normalized_trajectories_multi,
                sample_dtw_cluster_result,
                cluster_id=cid,
                output_path=output,
            )
            assert output.exists(), f"Cluster {cid} plot should exist"
            assert output.stat().st_size > 0

    def test_cluster_invalid_id_raises(
        self,
        sample_normalized_trajectories_multi: list,
        sample_dtw_cluster_result: object,
        tmp_path: Path,
    ) -> None:
        """plot_cluster(cluster_id=99) raises ValueError for out-of-range id."""
        import numpy as np

        plotter = NarrativeOverlayPlotter()
        output = tmp_path / "overlay_cluster_99.png"

        with pytest.raises(ValueError, match="cluster_id 99 out of range"):
            plotter.plot_cluster(
                sample_normalized_trajectories_multi,
                sample_dtw_cluster_result,
                cluster_id=99,
                output_path=output,
            )

    def test_cluster_zero_members_raises(
        self,
        sample_normalized_trajectories_multi: list,
        tmp_path: Path,
    ) -> None:
        """plot_cluster raises ValueError when cluster has 0 members."""
        import numpy as np

        from pcmfg.analysis.dtw_clusterer import DTWClusterResult

        # Create cluster result with one cluster that has NO matching sources
        bary = np.zeros((100, 18), dtype=np.float64)
        empty_result = DTWClusterResult(
            assignments={},
            barycenters=[bary],
            distance_matrix=np.zeros((1, 1)),
            n_clusters=1,
            metric="dtw",
            sakoe_chiba_radius=2,
            cluster_sizes={"0": 0},
            silhouette_score=0.0,
            sources=[],
        )

        plotter = NarrativeOverlayPlotter()
        output = tmp_path / "overlay_cluster_empty.png"

        with pytest.raises(ValueError, match="has no members"):
            plotter.plot_cluster(
                sample_normalized_trajectories_multi,
                empty_result,
                cluster_id=0,
                output_path=output,
            )

    def test_plot_all_generates_suite(
        self,
        sample_normalized_trajectories_multi: list,
        sample_dtw_cluster_result: object,
        tmp_path: Path,
    ) -> None:
        """plot_all() with cluster_result returns list of Paths, all files exist."""
        plotter = NarrativeOverlayPlotter()
        generated = plotter.plot_all(
            sample_normalized_trajectories_multi,
            tmp_path,
            cluster_result=sample_dtw_cluster_result,
        )

        assert isinstance(generated, list)
        assert len(generated) == 14, (
            f"Expected 14 files (grid+9emotions+2directions+2clusters), got {len(generated)}"
        )
        for path in generated:
            assert path.exists(), f"Expected file {path} to exist"

    def test_plot_all_without_cluster(
        self,
        sample_normalized_trajectories_multi: list,
        tmp_path: Path,
    ) -> None:
        """plot_all() without cluster_result generates 12 files (no cluster PNGs)."""
        plotter = NarrativeOverlayPlotter()
        generated = plotter.plot_all(
            sample_normalized_trajectories_multi,
            tmp_path,
        )

        assert isinstance(generated, list)
        assert len(generated) == 12, (
            f"Expected 12 files (grid+9emotions+2directions), got {len(generated)}"
        )
        for path in generated:
            assert path.exists(), f"Expected file {path} to exist"

    def test_plot_all_with_cluster_file_count(
        self,
        sample_normalized_trajectories_multi: list,
        sample_dtw_cluster_result: object,
        tmp_path: Path,
    ) -> None:
        """plot_all() with cluster_result generates 12 + N_cluster files."""
        plotter = NarrativeOverlayPlotter()
        generated = plotter.plot_all(
            sample_normalized_trajectories_multi,
            tmp_path,
            cluster_result=sample_dtw_cluster_result,
        )

        # 2 clusters in sample_dtw_cluster_result
        assert len(generated) == 14
