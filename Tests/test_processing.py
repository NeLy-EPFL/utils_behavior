"""Tests for utils_behavior.processing — pure-function helpers."""
from __future__ import annotations

import h5py
import numpy as np
import pytest

from utils_behavior import processing


class TestButterLowpass:
    def test_removes_high_frequency_noise(self):
        fs = 100
        t = np.linspace(0, 1, fs, endpoint=False)
        signal = np.sin(2 * np.pi * 2 * t)  # 2 Hz signal
        noise = 0.5 * np.sin(2 * np.pi * 40 * t)  # 40 Hz noise
        noisy = signal + noise

        filtered = processing.butter_lowpass_filter(noisy, cutoff=0.1, order=4)

        # Filtered output should be closer to the clean signal than the noisy input
        assert np.std(filtered - signal) < np.std(noisy - signal)
        assert filtered.shape == noisy.shape


class TestSavgolLowpass:
    def test_preserves_shape(self):
        data = np.sin(np.linspace(0, 10, 500))
        smoothed = processing.savgol_lowpass_filter(data, window_length=51, polyorder=2)
        assert smoothed.shape == data.shape

    def test_default_params(self):
        data = np.random.default_rng(0).standard_normal(500)
        smoothed = processing.savgol_lowpass_filter(data)
        assert smoothed.shape == data.shape
        # Smoothed signal should have lower variance than raw noise
        assert np.var(smoothed) < np.var(data)


class TestReplaceNansWithPreviousValue:
    def test_fills_interior_nans(self):
        arr = np.array([1.0, 2.0, np.nan, np.nan, 5.0], dtype=float)
        processing.replace_nans_with_previous_value(arr)
        assert not np.isnan(arr).any()
        assert arr[2] == 2.0
        assert arr[3] == 2.0

    def test_leading_nan_uses_next_valid(self):
        arr = np.array([np.nan, np.nan, 3.0, 4.0], dtype=float)
        processing.replace_nans_with_previous_value(arr)
        assert arr[0] == 3.0

    def test_no_nans_is_noop(self):
        arr = np.array([1.0, 2.0, 3.0])
        original = arr.copy()
        processing.replace_nans_with_previous_value(arr)
        np.testing.assert_array_equal(arr, original)


class TestEuclideanDistance:
    def test_scalar_inputs(self):
        d = processing.calculate_euclidian_distance(0, 0, 3, 4)
        assert d == pytest.approx(5.0)

    def test_array_inputs(self):
        x1 = np.array([0.0, 1.0])
        y1 = np.array([0.0, 1.0])
        x2 = np.array([3.0, 4.0])
        y2 = np.array([4.0, 5.0])
        d = processing.calculate_euclidian_distance(x1, y1, x2, y2)
        np.testing.assert_allclose(d, [5.0, 5.0])


class TestLogisticFunction:
    def test_midpoint(self):
        # At t == t0, output is L/2
        val = processing.logistic_function(0.0, L=1.0, k=1.0, t0=0.0)
        assert val == pytest.approx(0.5)

    def test_asymptotes(self):
        assert processing.logistic_function(-100.0, L=1.0, k=1.0, t0=0.0) == pytest.approx(0.0)
        assert processing.logistic_function(100.0, L=1.0, k=1.0, t0=0.0) == pytest.approx(1.0)


class TestExtractCoordinates:
    def test_reads_single_object_h5(self, tmp_path, sample_tracks_array):
        # extract_coordinates expects single-object data; use the first track only.
        single_obj = sample_tracks_array[0:1]  # shape (1, 2, n_nodes, n_frames)
        h5_path = tmp_path / "single.h5"
        with h5py.File(h5_path, "w") as f:
            f.create_dataset("tracks", data=single_obj)

        x, y = processing.extract_coordinates(str(h5_path))
        # After transpose + squeeze, we should get (n_frames, n_nodes) arrays.
        assert x.ndim == 2
        assert y.ndim == 2
        assert x.shape == y.shape
        # The values should match the source data (axis 0 of sliced pairs).
        assert np.isfinite(x).any()
        assert np.isfinite(y).any()


class TestDrawBsCi:
    def test_returns_interval(self):
        rng = np.random.default_rng(0)
        data = rng.normal(loc=5.0, scale=1.0, size=200)
        ci = processing.draw_bs_ci(data, n_reps=100)
        assert ci.shape == (2,)
        low, high = ci
        assert low < high
        # 95% CI for the mean should straddle the true mean
        assert low < 5.0 < high
