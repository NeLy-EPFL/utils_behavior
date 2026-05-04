"""Shared pytest fixtures for utils_behavior tests."""
from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest


@pytest.fixture
def sample_tracks_array() -> np.ndarray:
    """A deterministic synthetic ``tracks`` array shaped like SLEAP's H5 export.

    SLEAP's H5 ``tracks`` dataset has shape ``(n_tracks, 2, n_nodes, n_frames)``
    where axis 1 is (x, y). We generate 2 tracks with 3 nodes and 100 frames.
    """
    rng = np.random.default_rng(seed=0)
    n_tracks, n_nodes, n_frames = 2, 3, 100
    arr = rng.uniform(0.0, 500.0, size=(n_tracks, 2, n_nodes, n_frames))
    return arr.astype(np.float64)


@pytest.fixture
def fake_sleap_h5(tmp_path: Path, sample_tracks_array: np.ndarray) -> Path:
    """Write a minimal but valid SLEAP-style ``.h5`` file to ``tmp_path``.

    Contains the fields Sleap_Tracks reads at init:
    - ``node_names``   : bytes array of node names
    - ``edge_names``   : bytes array of (src, dst) node name pairs
    - ``edge_inds``    : int array of (src, dst) node indices
    - ``tracks``       : float array shaped (n_tracks, 2, n_nodes, n_frames)
    - ``video_path``   : scalar bytes string (filename only, no real video needed)
    """
    h5_path = tmp_path / "predictions.h5"
    node_names = [b"head", b"thorax", b"abdomen"]
    edge_names = np.array(
        [[b"head", b"thorax"], [b"thorax", b"abdomen"]], dtype="S"
    )
    edge_inds = np.array([[0, 1], [1, 2]], dtype=np.int64)

    # A fake video path — Sleap_Tracks will try to open it via cv2 and just
    # set fps=None on failure, which is fine for the metadata-only tests here.
    fake_video = tmp_path / "dummy.mp4"

    with h5py.File(h5_path, "w") as f:
        f.create_dataset("node_names", data=np.array(node_names, dtype="S"))
        f.create_dataset("edge_names", data=edge_names)
        f.create_dataset("edge_inds", data=edge_inds)
        f.create_dataset("tracks", data=sample_tracks_array)
        f.create_dataset(
            "video_path", data=np.bytes_(str(fake_video)), dtype=h5py.string_dtype()
        )

    return h5_path
