from scipy import signal
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import h5py
import warnings
import pandas as pd

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


def permutation_test(observed, random, n_permutations=1000):
    """
    Perform a permutation test to compare the means of two groups.

    This function calculates the observed mean difference between two groups and performs a permutation test to determine the significance of the observed difference. The permutation test involves shuffling the combined data and recalculating the mean difference for a specified number of permutations.

    Args:
        observed (pd.DataFrame): A DataFrame containing the observed data.
        random (pd.DataFrame): A DataFrame containing the random data.
        n_permutations (int, optional): The number of permutations to perform. Default is 1000.

    Returns:
        tuple: A tuple containing:
            - observed_diff (pd.Series): The observed mean difference between the two groups at each time point.
            - p_values (np.ndarray): The p-values for each time point, representing the proportion of permuted mean differences that are as extreme as the observed difference.

    Example:
        >>> observed = pd.DataFrame([[1, 2, 3], [4, 5, 6]])
        >>> random = pd.DataFrame([[7, 8, 9], [10, 11, 12]])
        >>> observed_diff, p_values = permutation_test(observed, random, n_permutations=1000)
    """
    combined = pd.concat([observed, random], axis=1)
    observed_diff = observed.mean(axis=1, skipna=True) - random.mean(
        axis=1, skipna=True
    )
    perm_diffs = []

    for i in range(n_permutations):
        # Shuffle columns and split back into two groups
        permuted_df = combined.sample(frac=1, axis=1, replace=False, random_state=i)
        perm_group1 = permuted_df.iloc[:, : observed.shape[1]]
        perm_group2 = permuted_df.iloc[:, observed.shape[1] :]

        # Calculate mean difference of permuted groups at each time point
        perm_diff = perm_group1.mean(axis=1, skipna=True) - perm_group2.mean(
            axis=1, skipna=True
        )
        perm_diffs.append(perm_diff.values)

    # Calculate p-values: proportion of permuted mean differences that are as extreme as observed
    perm_diffs = np.array(perm_diffs)
    p_values = np.array(
        [
            np.mean(np.abs(perm_diffs[:, i]) >= np.abs(observed_diff[i]))
            for i in range(len(observed_diff))
        ]
    )

    return observed_diff, p_values
