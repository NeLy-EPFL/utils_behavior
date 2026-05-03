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

    def test_uses_custom_func(self):
        # Should accept any reduction function (e.g. np.median).
        rng = np.random.default_rng(0)
        data = rng.normal(loc=0.0, scale=1.0, size=200)
        ci = processing.draw_bs_ci(data, func=np.median, n_reps=100)
        assert ci.shape == (2,)
        assert ci[0] < ci[1]

    def test_constant_data_yields_tight_interval(self):
        data = np.full(100, 7.0)
        ci = processing.draw_bs_ci(data, n_reps=50)
        # All bootstrap means of constant data are exactly 7.0
        np.testing.assert_allclose(ci, [7.0, 7.0])


class TestDrawBsRep:
    def test_returns_scalar_under_func(self):
        rng = np.random.default_rng(0)
        data = np.arange(50, dtype=float)
        rep = processing.draw_bs_rep(data, np.mean, rng)
        assert np.isfinite(rep)
        # The mean of any resample should be within the data's [min, max] range
        assert data.min() <= rep <= data.max()


class TestCheby1Lowpass:
    def test_smooths_high_freq_noise(self):
        fs = 100
        t = np.linspace(0, 1, fs, endpoint=False)
        signal_clean = np.sin(2 * np.pi * 2 * t)  # 2 Hz signal
        noise = 0.5 * np.sin(2 * np.pi * 40 * t)  # 40 Hz noise
        noisy = signal_clean + noise

        filtered = processing.cheby1_lowpass_filter(noisy, cutoff=0.1, order=4, rp=1.0)

        assert filtered.shape == noisy.shape
        # Filtered output is closer to clean than noisy is
        assert np.std(filtered - signal_clean) < np.std(noisy - signal_clean)


class TestComputeEffectSize:
    def test_returns_mean_difference(self):
        # interval here means a (low, high) confidence interval
        interval1 = (4.0, 6.0)  # mean = 5
        interval2 = (1.0, 3.0)  # mean = 2
        effect, _ = processing.compute_effect_size(interval1, interval2)
        assert effect == pytest.approx(3.0)

    def test_returns_effect_interval(self):
        interval1 = (4.0, 6.0)
        interval2 = (1.0, 3.0)
        _, eff_interval = processing.compute_effect_size(interval1, interval2)
        # interval_min = interval1[1] - interval2[0] = 6-1 = 5
        # interval_max = interval1[0] - interval2[1] = 4-3 = 1
        assert eff_interval == (5.0, 1.0)

    def test_zero_effect(self):
        # Identical intervals -> zero effect, zero-width interval
        eff, eff_int = processing.compute_effect_size((0.0, 4.0), (0.0, 4.0))
        assert eff == pytest.approx(2.0 - 2.0)
        assert eff_int[0] == 4.0
        assert eff_int[1] == -4.0


class TestPreprocessData:
    @pytest.fixture
    def sample_df(self):
        rng = np.random.default_rng(0)
        n = 200
        return pd.DataFrame(
            {
                "time": np.linspace(0, 10, n),
                "distance_ball_0": rng.normal(0, 1, n),
                "Brain region": rng.choice(["A", "B"], n),
                "fly": rng.choice(["fly1", "fly2", "fly3"], n),
            }
        )

    def test_returns_dataframe_with_expected_columns(self, sample_df):
        out = processing.preprocess_data(sample_df, bins=10)
        for col in (
            "time_bin",
            "Brain region",
            "fly",
            "avg_distance_ball_0",
            "median_distance_ball_0",
        ):
            assert col in out.columns

    def test_bin_count_caps_unique_bins(self, sample_df):
        out = processing.preprocess_data(sample_df, bins=5)
        assert out["time_bin"].nunique() <= 5

    def test_no_binning_uses_raw_time(self, sample_df):
        out = processing.preprocess_data(sample_df, bins=None)
        # Without binning, time_bin equals the raw time column for input rows
        assert out["time_bin"].nunique() <= len(sample_df["time"].unique())


class TestComputePermutationTest:
    @pytest.fixture
    def grouped_df(self):
        rng = np.random.default_rng(42)
        rows = []
        for tb in range(5):
            for grp, mu in [("Control", 0.0), ("Treated", 1.5)]:
                vals = rng.normal(mu, 0.3, size=20)
                for v in vals:
                    rows.append({"time_bin": tb, "Brain region": grp, "metric": v})
        return pd.DataFrame(rows)

    def test_required_columns_validated(self):
        df = pd.DataFrame({"time_bin": [0], "Brain region": ["A"]})
        with pytest.raises(ValueError, match="Missing required columns"):
            processing.compute_permutation_test(df, metric="metric")

    def test_returns_expected_keys(self, grouped_df):
        out = processing.compute_permutation_test(
            grouped_df, metric="metric", n_permutations=50
        )
        assert set(out.keys()) >= {
            "observed_diff",
            "p_values",
            "p_values_corrected",
            "significant_timepoints",
            "time_bins",
        }
        assert out["observed_diff"].shape == (5,)
        assert out["p_values"].shape == (5,)

    def test_detects_clear_separation(self, grouped_df):
        # With strong separation between Control (0) and Treated (1.5),
        # the observed differences should be large and at least one bin should
        # come back significant after FDR correction.
        out = processing.compute_permutation_test(
            grouped_df, metric="metric", n_permutations=200
        )
        assert np.all(np.abs(out["observed_diff"]) > 0.5)
        assert (out["p_values_corrected"] < 0.05).any()
