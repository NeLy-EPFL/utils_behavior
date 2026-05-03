"""Tests for utils_behavior.utils."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from utils_behavior import utils


class TestPlatformPaths:
    @patch("utils_behavior.utils.platform.system", return_value="Darwin")
    def test_get_labserver_macos(self, _mock):
        assert utils.get_labserver() == Path("/Volumes/upramdya_files")

    @patch("utils_behavior.utils.platform.system", return_value="Linux")
    def test_get_labserver_linux(self, _mock):
        assert utils.get_labserver() == Path("/mnt/upramdya_files")

    @patch("utils_behavior.utils.platform.system", return_value="Windows")
    def test_get_labserver_unsupported_raises(self, _mock):
        with pytest.raises(ValueError, match="Unsupported platform"):
            utils.get_labserver()

    @patch("utils_behavior.utils.platform.system", return_value="Darwin")
    def test_get_data_server_macos(self, _mock):
        assert utils.get_data_server() == Path("/Volumes/upramdya/data")

    @patch("utils_behavior.utils.platform.system", return_value="Linux")
    def test_get_data_server_linux(self, _mock):
        assert utils.get_data_server() == Path("/mnt/upramdya_data")

    @patch("utils_behavior.utils.platform.system", return_value="Windows")
    def test_get_data_server_unsupported_raises(self, _mock):
        with pytest.raises(ValueError, match="Unsupported platform"):
            utils.get_data_server()


class TestGetDataPath:
    def test_default_setup(self):
        assert utils.get_data_path() == Path(
            "/mnt/upramdya_data/MD/MultiMazeRecorder/Videos"
        )

    def test_explicit_setup(self):
        assert utils.get_data_path(setup="mazerecorder") == Path(
            "/mnt/upramdya_data/MD/MultiMazeRecorder/Videos"
        )


class TestGetFolders:
    def test_returns_all_folders_when_no_keywords(self, tmp_path):
        (tmp_path / "alpha").mkdir()
        (tmp_path / "beta").mkdir()
        (tmp_path / "alpha_test").mkdir()
        # Files in the same dir should also be returned (function does not
        # filter to folders only — that's the documented current behavior)
        (tmp_path / "note.txt").write_text("x")

        items = utils.get_folders(tmp_path)
        names = sorted(p.name for p in items)
        assert names == ["alpha", "alpha_test", "beta", "note.txt"]

    def test_filters_by_single_keyword(self, tmp_path):
        (tmp_path / "alpha").mkdir()
        (tmp_path / "beta").mkdir()
        (tmp_path / "alpha_extra").mkdir()

        items = utils.get_folders(tmp_path, keywords=["alpha"])
        names = sorted(p.name for p in items)
        assert names == ["alpha", "alpha_extra"]

    def test_filters_by_multiple_keywords_uses_AND(self, tmp_path):
        (tmp_path / "alpha_2024").mkdir()
        (tmp_path / "alpha_2023").mkdir()
        (tmp_path / "beta_2024").mkdir()

        items = utils.get_folders(tmp_path, keywords=["alpha", "2024"])
        names = sorted(p.name for p in items)
        assert names == ["alpha_2024"]

    def test_keywords_are_case_insensitive(self, tmp_path):
        (tmp_path / "AlphaExperiment").mkdir()
        items = utils.get_folders(tmp_path, keywords=["ALPHA"])
        assert len(items) == 1


class TestFrame2Time:
    def test_frames_to_seconds(self, capsys):
        # frame=300, fps=30 -> 10 seconds
        result = utils.frame2time(300, 30)
        assert result == 10
        # The function also prints the value; verify it didn't crash on print
        capsys.readouterr()

    def test_frames_to_clock(self, capsys):
        # 3661 seconds ~ 1h 1m 1s; at fps=1, frame 3661 is 3661 seconds
        result = utils.frame2time(3661, 1, clockformat=True)
        assert result == (1, 1, 1)
        capsys.readouterr()

    def test_seconds_to_frames(self, capsys):
        # reverse=True: time in seconds -> frames
        result = utils.frame2time(10, 30, reverse=True)
        assert result == 300
        capsys.readouterr()

    def test_clock_string_to_frames(self, capsys):
        # reverse=True + clockformat=True: parse "HH:MM:SS" string
        result = utils.frame2time("01:00:00", 30, reverse=True, clockformat=True)
        assert result == 3600 * 30
        capsys.readouterr()


class TestAddNote:
    def test_creates_note_file_when_missing(self, tmp_path, capsys):
        fpath = tmp_path / "video.mp4"
        fpath.touch()
        utils.add_note(fpath, "first note")

        notes = tmp_path / "notes.txt"
        assert notes.exists()
        assert notes.read_text() == "first note\n"

    def test_appends_to_existing_note_file(self, tmp_path, capsys):
        fpath = tmp_path / "video.mp4"
        fpath.touch()
        notes = tmp_path / "notes.txt"
        notes.write_text("existing note\n")

        utils.add_note(fpath, "second note")

        assert notes.read_text() == "existing note\nsecond note\n"


class TestNotifyMe:
    def test_no_op_when_url_unset(self, monkeypatch, capsys):
        monkeypatch.delenv("IFTTT_URL", raising=False)
        # Should not raise; should print a friendly notice
        utils.notify_me()
        captured = capsys.readouterr()
        # The function prints "no IFTTT URL" guidance — just make sure we got
        # SOME output rather than an exception
        assert captured.out != ""

    def test_posts_when_url_set(self, monkeypatch):
        monkeypatch.setenv("IFTTT_URL", "https://example.invalid/trigger")
        with patch("utils_behavior.utils.requests.post") as mock_post:
            utils.notify_me()
            mock_post.assert_called_once_with("https://example.invalid/trigger")
