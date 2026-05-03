"""Tests for utils_behavior.sleap.tracks (Sleap_Tracks wrapper)."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from utils_behavior.sleap import Sleap_Tracks


@pytest.fixture
def mock_cv2_capture():
    """Mock cv2.VideoCapture so Sleap_Tracks can pretend a video exists."""
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    # Make .get return 30 fps for CAP_PROP_FPS and 100 frames for CAP_PROP_FRAME_COUNT.
    # cv2.CAP_PROP_FPS == 5, cv2.CAP_PROP_FRAME_COUNT == 7 (they're ints).
    mock_cap.get.side_effect = lambda prop: {5: 30.0, 7: 100}.get(prop, 0.0)

    with patch("utils_behavior.sleap.tracks.cv2.VideoCapture", return_value=mock_cap):
        yield mock_cap


class TestSleapTracksInit:
    def test_reads_node_names(self, fake_sleap_h5, mock_cv2_capture):
        tracks = Sleap_Tracks(
            fake_sleap_h5, object_type="fly", smoothed_tracks=False
        )
        assert tracks.node_names == ["head", "thorax", "abdomen"]

    def test_reads_edges(self, fake_sleap_h5, mock_cv2_capture):
        tracks = Sleap_Tracks(fake_sleap_h5, smoothed_tracks=False)
        assert tracks.edge_names == [["head", "thorax"], ["thorax", "abdomen"]]
        assert tracks.edges_idx == [[0, 1], [1, 2]]

    def test_fps_from_mocked_video(self, fake_sleap_h5, mock_cv2_capture):
        tracks = Sleap_Tracks(fake_sleap_h5, smoothed_tracks=False)
        assert tracks.fps == 30.0

    def test_path_is_pathlib(self, fake_sleap_h5, mock_cv2_capture):
        tracks = Sleap_Tracks(str(fake_sleap_h5), smoothed_tracks=False)
        assert isinstance(tracks.path, Path)

    def test_object_count_matches_tracks(
        self, fake_sleap_h5, sample_tracks_array, mock_cv2_capture
    ):
        tracks = Sleap_Tracks(fake_sleap_h5, smoothed_tracks=False)
        assert len(tracks.objects) == sample_tracks_array.shape[0]


class TestTracksData:
    def test_dataset_columns(self, fake_sleap_h5, mock_cv2_capture):
        tracks = Sleap_Tracks(fake_sleap_h5, smoothed_tracks=False)
        obj = tracks.objects[0]
        df = obj.dataset
        expected_cols = {
            "frame",
            "time",
            "object",
            "x_head",
            "y_head",
            "x_thorax",
            "y_thorax",
            "x_abdomen",
            "y_abdomen",
        }
        assert expected_cols.issubset(set(df.columns))

    def test_frame_count(self, fake_sleap_h5, sample_tracks_array, mock_cv2_capture):
        tracks = Sleap_Tracks(fake_sleap_h5, smoothed_tracks=False)
        # sample fixture uses 100 frames
        assert len(tracks.objects[0].dataset) == sample_tracks_array.shape[-1]

    def test_time_is_frame_over_fps(self, fake_sleap_h5, mock_cv2_capture):
        tracks = Sleap_Tracks(fake_sleap_h5, smoothed_tracks=False)
        df = tracks.objects[0].dataset
        np.testing.assert_allclose(df["time"].values, df["frame"].values / 30.0)

    def test_object_label_uses_object_type(self, fake_sleap_h5, mock_cv2_capture):
        tracks = Sleap_Tracks(
            fake_sleap_h5, object_type="fly", smoothed_tracks=False
        )
        assert (tracks.objects[0].dataset["object"] == "fly_1").all()
        assert (tracks.objects[1].dataset["object"] == "fly_2").all()

    def test_node_property_returns_xy_pairs(
        self, fake_sleap_h5, mock_cv2_capture, sample_tracks_array
    ):
        tracks = Sleap_Tracks(fake_sleap_h5, smoothed_tracks=False)
        obj = tracks.objects[0]
        head_xy = obj.head  # dynamic attribute
        assert len(head_xy) == sample_tracks_array.shape[-1]
        # Each entry is an (x, y) tuple of floats
        assert all(isinstance(pt, tuple) and len(pt) == 2 for pt in head_xy[:5])


class TestFilterData:
    def test_restricts_time_range(self, fake_sleap_h5, mock_cv2_capture):
        tracks = Sleap_Tracks(fake_sleap_h5, smoothed_tracks=False)
        # All frames exist (100 frames at 30 fps => ~3.33 s of data)
        tracks.filter_data(time_range=(0.5, 1.5))
        for obj in tracks.objects:
            assert obj.dataset["time"].min() >= 0.5
            assert obj.dataset["time"].max() <= 1.5

    def test_open_ended_start(self, fake_sleap_h5, mock_cv2_capture):
        tracks = Sleap_Tracks(fake_sleap_h5, smoothed_tracks=False)
        tracks.filter_data(time_range=(None, 1.0))
        for obj in tracks.objects:
            assert obj.dataset["time"].max() <= 1.0

    def test_open_ended_end(self, fake_sleap_h5, mock_cv2_capture):
        tracks = Sleap_Tracks(fake_sleap_h5, smoothed_tracks=False)
        tracks.filter_data(time_range=(1.0, None))
        for obj in tracks.objects:
            assert obj.dataset["time"].min() >= 1.0


class TestSmoothing:
    def test_smoothed_flag_changes_coordinates(
        self, fake_sleap_h5, mock_cv2_capture
    ):
        raw = Sleap_Tracks(fake_sleap_h5, smoothed_tracks=False)
        smoothed = Sleap_Tracks(
            fake_sleap_h5, smoothed_tracks=True, smoothing_params=(11, 1)
        )
        raw_x = raw.objects[0].dataset["x_head"].values
        smoothed_x = smoothed.objects[0].dataset["x_head"].values
        # Smoothing should reduce variance for noisy uniform data
        assert np.var(smoothed_x) < np.var(raw_x)


class TestPublicExports:
    def test_top_level_import(self):
        """Sleap_Tracks and generate_annotated_frame should be importable from the subpackage."""
        from utils_behavior.sleap import Sleap_Tracks, generate_annotated_frame

        assert callable(generate_annotated_frame)
        assert isinstance(Sleap_Tracks, type)
