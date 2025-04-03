from scipy import signal
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import h5py
import pandas as pd
from typing import Dict, List, Union
from statsmodels.stats.multitest import multipletests

# Compute low-pass filtered data


def butter_lowpass_filter(data, cutoff, order):
    """
    Apply a Butterworth low-pass filter to the data. This is a useful filter for removing high-frequency noise from data,
    but it can smooth out sharp edges in the data.

    Args:
        data (np.ndarray): The data to be filtered.
        cutoff (float): The cutoff frequency in normalized frequency.
        order (int): The order of the filter.

    Returns:
        y (np.ndarray): The filtered data.
    """
    # Get the filter coefficients
    b, a = signal.butter(order, cutoff, btype="low", analog=False)
    # Apply the filter to the data
    y = signal.filtfilt(b, a, data)

    return y


def savgol_lowpass_filter(data, window_length=221, polyorder=1):
    """
    Apply a Savitzky-Golay low-pass filter to the data. This is useful for removing high-frequency noise from data while
    preserving sharp edges in the data.

    Args:
        data (np.ndarray): The data to be filtered.
        window_length (int): The length of the filter window (i.e., the number of coefficients). Must be an odd integer.
        polyorder (int): The order of the polynomial used to fit the samples.

    Returns:
        y (np.ndarray): The filtered data.
    """

    # Apply the Savitzky-Golay filter
    y = signal.savgol_filter(data, window_length, polyorder)
    return y


def cheby1_lowpass_filter(data, cutoff, order, rp):
    """
    Apply a Chebyshev Type I low-pass filter to the data. This filter minimizes the worst-case error between the ideal
    and actual frequency response.

    Args:
        data (np.ndarray): The data to be filtered.
        cutoff (float): The cutoff frequency in normalized frequency.
        order (int): The order of the filter.
        rp (float): The maximum ripple allowed below unity gain in the passband (specified in decibels).

    Returns:
        y (np.ndarray): The filtered data.
    """

    # Get the filter coefficients
    b, a = signal.cheby1(order, rp, cutoff, btype="low", analog=False)
    # Apply the filter to the data
    y = signal.filtfilt(b, a, data)
    return y


# Compute bootstrapped confidence interval

rg = np.random.default_rng()


def draw_bs_rep(data, func, rg):
    """
    Compute a bootstrap replicate from data.

    Args:
        data (np.ndarray): The data to draw samples from.
        func (callable): The function to apply to the bootstrap samples.
        rg (np.random.Generator): The random number generator.

    Returns:
        float: The bootstrap replicate.
    """

    bs_sample = rg.choice(data, size=len(data))
    return func(bs_sample)


def draw_bs_ci(data, func=np.mean, rg=rg, n_reps=300, show_progress=False):
    """
    Sample bootstrap multiple times and compute confidence interval.

    Args:
        data (np.ndarray): The data to draw samples from.
        func (callable, optional): The function to apply to the bootstrap samples. Defaults to np.mean.
        rg (np.random.Generator): The random number generator.
        n_reps (int, optional): The number of bootstrap replicates. Defaults to 300.
        show_progress (bool, optional): Whether to show the progress bar. Defaults to True.

    Returns:
        np.ndarray: The lower and upper confidence interval bounds.
    """

    with ThreadPoolExecutor() as executor:
        if show_progress:
            bs_reps = list(
                tqdm(
                    executor.map(lambda _: draw_bs_rep(data, func, rg), range(n_reps)),
                    total=n_reps,
                )
            )
        else:
            bs_reps = list(
                executor.map(lambda _: draw_bs_rep(data, func, rg), range(n_reps))
            )
    conf_int = np.percentile(bs_reps, [2.5, 97.5])
    return conf_int


def compute_effect_size(interval1, interval2):
    """This function takes two sets of intervals and computes the effect size between them.

    Returns:
        tuple: the average effect size and the interval of the effect size.
    Args:
        interval1 (tuple): The first set of intervals.
        interval1 (tuple): The second set of intervals.
    """

    # Compute the effect size by getting the central tendency of the two intervals
    effect_size = np.mean(interval1) - np.mean(interval2)

    # Compute the interval of the effect size
    interval_min = interval1[1] - interval2[0]
    interval_max = interval1[0] - interval2[1]

    effect_size_interval = (interval_min, interval_max)

    return effect_size, effect_size_interval


def extract_coordinates(h5_file):
    """
    Extracts the x and y coordinates from a h5 file. Only works for single object tracking. For skeleton tracking,
    use extract_skeleton.

    Args:
        h5_file (str): The path to the h5 file.

    Returns:
        tuple: Two np.ndarrays representing the x and y coordinates.
    """

    with h5py.File(h5_file, "r") as f:
        locs = f["tracks"][:].T
        y = locs[:, :, 1, :].squeeze()
        x = locs[:, :, 0, :].squeeze()
    return x, y


def calculate_euclidian_distance(x1, y1, x2, y2):
    """
    Calculate the Euclidean distance between two sets of x and y coordinates.

    Args:
        x1 (np.ndarray): The x coordinates of the first set.
        y1 (np.ndarray): The y coordinates of the first set.
        x2 (np.ndarray): The x coordinates of the second set.
        y2 (np.ndarray): The y coordinates of the second set.

    Returns:
        np.ndarray: The Euclidean distance between the two sets of coordinates.
    """
    return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def logistic_function(t, L, k, t0):
    """
    Logistic function.

    Parameters
    ----------
    t : array-like
        Time.
    L : float
        Maximum value (plateau).
    k : float
        Growth rate (steepness of the curve).
    t0 : float
        Midpoint (time at which the curve reaches half of L).

    Returns
    -------
    array-like
        Logistic function values.
    """
    return L / (1 + np.exp(-k * (t - t0)))


def replace_nans_with_previous_value(arr):
    """
    Replace NaN values with the previous value in a numpy array. If the first value is NaN, it is replaced with the
    next non-NaN value.

    Parameters:
    arr (numpy.ndarray): The input array with potential NaN values.

    Returns:
    None. The function modifies the array in-place.
    """
    # Ensure arr is a numpy array
    arr = np.asarray(arr)

    # Check if the first value is NaN
    if np.isnan(arr[0]):
        # Find the next non-NaN value
        next_val_index = next((i for i, x in enumerate(arr) if not np.isnan(x)), None)
        if next_val_index is not None:
            arr[0] = arr[next_val_index]

    # Replace NaNs with the previous value
    for i in range(1, len(arr)):
        if np.isnan(arr[i]):
            arr[i] = arr[i - 1]


def preprocess_data(
    data: pd.DataFrame,
    time_col: str = "time",
    value_col: str = "distance_ball_0",
    group_col: str = "Brain region",
    subject_col: str = "fly",
    bins: Union[int, List[float], None] = 10,
) -> pd.DataFrame:
    """
    Prepares a simplified dataset by aggregating time-series data into bins or using raw time values.

    Args:
        data: Input DataFrame containing time-series data
        time_col: Name of the time column (default: 'time')
        value_col: Name of the value column to aggregate (default: 'distance_ball_0')
        group_col: Name of the grouping column (e.g. experimental groups) (default: 'Brain region')
        subject_col: Name of the subject ID column (default: 'fly')
        bins: Number of bins, bin edges, or None to use raw time values (default: 10)

    Returns:
        DataFrame with aggregated statistics per time bin/point, group, and subject

    Example:
        >>> preprocess_data(df, time_col='timestamp', value_col='velocity', bins=20)
    """
    df = data.copy()

    if bins is None:
        # Use raw time values without binning
        time_bins = df[time_col]
    else:
        # Create time bins using pandas cut
        time_bins = pd.cut(df[time_col], bins=bins, labels=False)

    df["time_bin"] = time_bins

    agg_funcs = {
        "avg": ("mean", f"avg_{value_col}"),
        "median": ("median", f"median_{value_col}"),
    }

    processed = (
        df.groupby(["time_bin", group_col, subject_col], observed=True)[value_col]
        .agg([agg_funcs[k][0] for k in agg_funcs])
        .rename(columns={v[0]: v[1] for k, v in agg_funcs.items()})
        .reset_index()
    )

    return processed


def compute_permutation_test(
    data: pd.DataFrame,
    metric: str,
    group_col: str = "Brain region",
    control_group: str = "Control",
    n_permutations: int = 1000,
    progress: bool = False,
) -> Dict[str, np.ndarray]:
    """
    Performs permutation testing between experimental groups and control group.

    Args:
        data: Input DataFrame containing preprocessed data
        metric: Name of the metric column to compare (e.g. 'avg_distance_ball_0')
        group_col: Name of the grouping column (default: 'Brain region')
        control_group: Name of the control group (default: 'Control')
        n_permutations: Number of permutations to perform (default: 1000)
        progress: Show progress bar (default: False)

    Returns:
        Dictionary containing:
        - observed_diff: Array of observed differences
        - p_values: Array of raw p-values
        - p_values_corrected: FDR-corrected p-values
        - significant_timepoints: Indices of significant timepoints
        - time_bins: Array of time bins

    Example:
        >>> results = compute_permutation_test(df, metric='avg_velocity', control_group='Placebo')
    """
    # Input validation
    required_cols = ["time_bin", group_col, metric]
    missing = [col for col in required_cols if col not in data]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Split data
    focal_mask = data[group_col] != control_group
    focal_data = data[focal_mask]
    control_data = data[~focal_mask]

    # Initialize results storage
    time_bins = np.sort(data["time_bin"].unique())
    results = {
        "observed_diff": np.zeros(len(time_bins)),
        "p_values": np.ones(len(time_bins)),
        "p_values_corrected": np.ones(len(time_bins)),
        "significant_timepoints": np.array([]),
        "time_bins": time_bins,
    }

    # Main permutation loop
    iterator = enumerate(time_bins)
    if progress:
        iterator = tqdm(iterator, total=len(time_bins), desc="Processing time bins")

    for i, tb in iterator:
        focal = focal_data[focal_data["time_bin"] == tb][metric].dropna()
        control = control_data[control_data["time_bin"] == tb][metric].dropna()

        if focal.empty or control.empty:
            continue

        # Calculate observed difference
        obs_diff = np.mean(focal) - np.mean(control)
        results["observed_diff"][i] = obs_diff

        # Permutation test
        combined = np.concatenate([focal, control])
        n_focal = len(focal)

        perm_diffs = np.empty(n_permutations)
        for p in range(n_permutations):
            np.random.shuffle(combined)
            perm_diffs[p] = np.mean(combined[:n_focal]) - np.mean(combined[n_focal:])

        pval = (np.abs(perm_diffs) >= np.abs(obs_diff)).mean()
        results["p_values"][i] = pval

    # Multiple testing correction
    _, pvals_corrected, _, _ = multipletests(results["p_values"], method="fdr_bh")
    results["p_values_corrected"] = pvals_corrected
    results["significant_timepoints"] = np.where(pvals_corrected < 0.05)[0]

    return results
