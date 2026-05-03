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


class TestGetShades:
    def test_returns_requested_count(self):
        from utils_behavior.sleap.tracks import get_shades

        assert len(get_shades("red", 5)) == 5
        assert len(get_shades("blue", 3)) == 3
        assert len(get_shades("green", 7)) == 7

    def test_red_shades_keep_red_at_255(self):
        from utils_behavior.sleap.tracks import get_shades

        for r, _g, _b in get_shades("red", 5):
            assert r == 255

    def test_blue_shades_keep_blue_at_255(self):
        from utils_behavior.sleap.tracks import get_shades

        for _r, _g, b in get_shades("blue", 5):
            assert b == 255

    def test_green_shades_keep_green_at_255(self):
        from utils_behavior.sleap.tracks import get_shades

        for _r, g, _b in get_shades("green", 5):
            assert g == 255

    def test_unknown_color_returns_empty(self):
        from utils_behavior.sleap.tracks import get_shades

        # The function returns whatever it appended for unknown colors,
        # which is nothing — so we get an empty list.
        assert get_shades("magenta", 5) == []

    def test_first_shade_is_pure_color(self):
        from utils_behavior.sleap.tracks import get_shades

        # i=0 means the lerp is 0 and the other channels collapse to 0
        assert get_shades("red", 4)[0] == (255, 0, 0)
        assert get_shades("green", 4)[0] == (0, 255, 0)
        assert get_shades("blue", 4)[0] == (0, 0, 255)


class TestCombinedSleapTracks:
    def test_combines_datasets_from_multiple_tracks(
        self, fake_sleap_h5, mock_cv2_capture
    ):
        from utils_behavior.sleap.tracks import CombinedSleapTracks

        a = Sleap_Tracks(fake_sleap_h5, object_type="fly", smoothed_tracks=False)
        b = Sleap_Tracks(fake_sleap_h5, object_type="ball", smoothed_tracks=False)

        combined = CombinedSleapTracks("/tmp/dummy.mp4", [a, b])
        assert combined.dataset is not None
        # Each tracks contributes 2 objects * 100 frames = 200 rows; total 400.
        assert len(combined.dataset) == len(a.objects[0].dataset) * (
            len(a.objects) + len(b.objects)
        )

    def test_assigns_one_color_per_tracks(self, fake_sleap_h5, mock_cv2_capture):
        from utils_behavior.sleap.tracks import CombinedSleapTracks

        a = Sleap_Tracks(fake_sleap_h5, smoothed_tracks=False)
        b = Sleap_Tracks(fake_sleap_h5, smoothed_tracks=False)

        combined = CombinedSleapTracks("/tmp/dummy.mp4", [a, b])
        # Two BGR colors, one per Sleap_Tracks object
        assert set(combined.colors.keys()) == {a, b}
        for color in combined.colors.values():
            assert len(color) == 3
            assert all(0 <= c <= 255 for c in color)


class TestSleapTracksEdgeCases:
    def test_handles_string_path(self, fake_sleap_h5, mock_cv2_capture):
        # Should accept str or Path indistinguishably
        t1 = Sleap_Tracks(str(fake_sleap_h5), smoothed_tracks=False)
        t2 = Sleap_Tracks(fake_sleap_h5, smoothed_tracks=False)
        assert t1.node_names == t2.node_names

    def test_filter_data_outside_range_yields_empty(
        self, fake_sleap_h5, mock_cv2_capture
    ):
        tracks = Sleap_Tracks(fake_sleap_h5, smoothed_tracks=False)
        tracks.filter_data(time_range=(1000.0, 2000.0))
        for obj in tracks.objects:
            assert len(obj.dataset) == 0

    def test_debug_mode_prints_summary(
        self, fake_sleap_h5, mock_cv2_capture, capsys
    ):
        Sleap_Tracks(fake_sleap_h5, smoothed_tracks=False, debug=True)
        captured = capsys.readouterr()
        assert "Loaded SLEAP tracking file" in captured.out
        assert "Nodes" in captured.out

    def test_pickling_roundtrip(self, fake_sleap_h5, mock_cv2_capture):
        """The Object inner class defines __getstate__/__setstate__ for pickling."""
        import pickle

        tracks = Sleap_Tracks(fake_sleap_h5, smoothed_tracks=False)
        obj = tracks.objects[0]
        head_before = list(obj.head)

        roundtrip = pickle.loads(pickle.dumps(obj))
        head_after = list(roundtrip.head)

        # Coordinates after a pickle roundtrip should match exactly
        assert head_before == head_after
        assert roundtrip.node_names == obj.node_names
