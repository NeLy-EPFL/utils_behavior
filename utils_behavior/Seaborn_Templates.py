# In this script we generate a set of styling and plotting templates for Seaborn


# Importing necessary libraries
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
import os
import re

from . import Processing

# Define styling templates
styling_templates = {
    "default": {
        "general": {
            "figure.figsize": (10, 6),
            "axes.titlesize": 16,
            "axes.labelsize": 14,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 12,
            "font.size": 12,
            "axes.grid": True,
            "grid.linestyle": "--",
            "grid.color": "gray",
            "axes.facecolor": "#EAEAF2",
            "axes.edgecolor": "white",
            "grid.color": "white",
        },
        "boxplot": {
            "linewidth": 2.5,
            "showfliers": False,
            "notch": True,
        },
        "violinplot": {
            "linewidth": 2.5,
        },
        "stripplot": {
            "jitter": 0.2,
            "alpha": 0.7,
        },
        "bs_notch": True,
        # Add more plot-specific parameters as needed
    },
    "large": {
        "general": {
            "figure.figsize": (14, 8),
            "axes.titlesize": 20,
            "axes.labelsize": 18,
            "xtick.labelsize": 16,
            "ytick.labelsize": 16,
            "legend.fontsize": 16,
            "font.size": 16,
            "axes.grid": True,
            "grid.linestyle": "--",
            "grid.color": "gray",
        },
        "boxplot": {
            "linewidth": 3,
        },
        "violinplot": {
            "linewidth": 3,
        },
        # Add more plot-specific parameters as needed
    },
    # Add more templates as needed
}


def clean_data(data, metric, groupby=None):
    """Clean the data by removing NaN values for the given metric and converting it to numeric.

    Args:
        data (pandas DataFrame): The dataset to clean.
        metric (str): The metric to clean.
        groupby (str): The column to group by if any.
    """

    if groupby:
        # Filter out groups where the metric column is all NaN
        data = data.groupby(groupby).filter(lambda x: x[metric].notna().any())
    else:
        # Filter out groups where the metric column is all NaN
        data = data[data[metric].notna()]

    # Clean the data by removing NaN values for this metric
    data = data.dropna(subset=[metric])

    # Convert metric to numeric
    data[metric] = pd.to_numeric(data[metric], errors="coerce")

    return data


def compute_controls_bs_ci(data, metric, control_label, kdims):
    """
    Compute a 95% bootstrap confidence interval for a given metric for the control group.

    Args:
        data (pandas DataFrame): The dataset to compute the confidence interval from.
        metric (str): The metric to compute the confidence interval for.
        control_label (str): The label of the control group.
        kdims (str): The key dimension to group by.

    Returns:
        np.ndarray: The lower and upper confidence interval bounds.
    """

    # Find the label corresponding to the control group
    control_label_full = data[data[kdims].str.startswith(control_label)].iloc[0][kdims]

    # Filter the data for the control group
    control_data = data[data[kdims] == control_label_full]

    # Drop rows with NaN values in the metric column
    control_data = control_data.dropna(subset=[metric])

    # Compute the bootstrap confidence interval for the given metric
    ci = Processing.draw_bs_ci(control_data[metric].values)

    return ci


def compute_group_bs_ci(data, metric, groupby):
    """
    Compute bootstrapped confidence intervals for each group.

    Args:
        data (pandas DataFrame): The dataset to compute the confidence intervals from.
        metric (str): The metric to compute the confidence intervals for.
        groupby (str): The column to group by.

    Returns:
        dict: A dictionary with group labels as keys and confidence intervals as values.
    """
    ci_dict = {}
    grouped_data = data.groupby(groupby)

    for group, group_data in grouped_data:
        group_data = group_data.dropna(subset=[metric])
        ci = Processing.draw_bs_ci(group_data[metric].values)
        ci_dict[group] = ci

    return ci_dict


def compute_hline_values(data, metric, hline_method, control_label, kdims):
    """
    Compute the values for the control area in the jitter boxplot. The values can be computed using either a bootstrap confidence interval or the 25% and 75% quantiles of the control group.

    Args:
        data (pandas DataFrame): The dataset to compute the values from.
        metric (str): The metric to compute the values for.
        hline_method (str): The method to compute the values. Either 'bootstrap' or 'boxplot'.
        control_label (str): The label of the control group.
        kdims (str): The key dimension to group by.

    """
    # Find the label corresponding to the control group
    control_label_full = data[data[kdims].str.startswith(control_label)].iloc[0][kdims]

    if hline_method == "bootstrap":
        return compute_controls_bs_ci(data, metric, control_label_full, kdims)
    elif hline_method == "boxplot":
        # Filter the data for the control group
        control_data = data[data[kdims] == control_label_full]

        # Calculate 25% and 75% quantiles for the control group
        return (
            control_data[metric].quantile(0.25),
            control_data[metric].quantile(0.75),
        )
    else:
        raise ValueError(
            "Invalid hline_method. Choose either 'bootstrap' or 'boxplot'."
        )


def sort_data(data, group_by, metric, sort_by="median"):
    """
    Sort the data by the median of the metric for each group in group_by.

    Args:
        data (pandas DataFrame): The dataset to sort.
        group_by (str or list): The column(s) to group by.
        metric (str): The metric to sort by.
        sort_by (str): The method to sort the data. Either 'median' or 'original'.
    """
    if isinstance(group_by, list):
        # Calculate the median for each group
        median_values = data.groupby(group_by)[metric].median()

        # Sort the groups by their median
        group_order = median_values.groupby(group_by[0]).median().sort_values().index

        # Within each group, sort the subgroups by their median
        subgroup_order_within_group = median_values.groupby(group_by[0]).apply(
            lambda x: x.sort_values().index.get_level_values(group_by[1])
        )

        # Create a new category type for the groups with the calculated order
        data[group_by[0]] = pd.Categorical(
            data[group_by[0]], categories=group_order, ordered=True
        )

        # Create a list to hold the correct order of subgroups across all groups
        correct_order_global = []

        # For each group, add the subgroup order to the global list
        for group in group_order:
            correct_order_global.extend(subgroup_order_within_group[group])

        # Convert the subgroups to a categorical type with the global order
        data[group_by[1]] = pd.Categorical(
            data[group_by[1]], categories=correct_order_global, ordered=True
        )

        # Now you can sort
        data.sort_values(by=group_by, inplace=True)

    elif isinstance(group_by, str):
        if sort_by == "median":
            median_values = data.groupby(group_by)[metric].median().sort_values()
            data["median"] = data[group_by].map(median_values)
            data = data.sort_values("median")
        else:
            # Return the original order with a warning message
            print(
                "Invalid sort_by option. No sorting applied. The data will be displayed in the original order."
            )

    return data


def compute_sample_size(data, metric, kdims, groupby=None):
    """
    Compute the sample size for each group and update the DataFrame with the sample size and label columns.

    Args:
        data (pandas DataFrame): The dataset to compute the sample size from.
        metric (str): The metric to filter out rows with NAs.
        kdims (str): The key dimension to group by.
        groupby (str, optional): Additional column to group by. Defaults to None.

    Returns:
        pandas DataFrame: The updated DataFrame with sample size and label columns.
        str: The updated kdims to use the new label column.
    """

    # Determine the grouping columns based on the presence of groupby
    grouping_columns = [kdims] if groupby is None else [kdims, groupby]

    # Filter out rows with NAs in the metric column
    data_filtered = data.dropna(subset=[metric])

    # Group by the determined columns and calculate the size of each group
    sample_size = (
        data_filtered.groupby(grouping_columns).size().reset_index(name="size")
    )

    # Ensure kdims is a string
    data["kdims_str"] = data[kdims].astype(str)

    # Merge sample_size back to the original data to align indices
    data = data.merge(sample_size, on=grouping_columns, how="left")

    # Fill NA values in the size column with 0
    data["size"] = data["size"].fillna(0).astype(int)

    # Create the label column
    data["label"] = data["kdims_str"] + " (n=" + data["size"].astype(str) + ")"

    # Update kdims to use the new label column
    kdims = "label"

    return data, kdims


def sns_plot(
    data,
    metric,
    kdims,
    plot_type="scatter",
    sort_by=None,
    group_by=None,
    colorby=None,
    hline_method="bootstrap",
    control_label=None,
    show=True,
    savepath=None,
    style_template="default",
    **kwargs,
):
    """
    Generate a seaborn plot with the given data and metric.

    Args:
        data (pandas DataFrame): The dataset to plot.
        metric (str): The metric to plot.
        kdims (str): The key dimension to group by.
        plot_type (str, optional): The type of plot to create ('scatter', 'boxplot', 'violinplot', 'jitterboxplot'). Defaults to 'scatter'.
        sort_by (str, optional): The method to sort the data. Defaults to None.
        group_by (str, optional): The column to group by. Defaults to None.
        hline_method (str, optional): The method to compute the hline values ('bootstrap' or 'boxplot'). Defaults to 'bootstrap'.
        control_label (str, optional): The label of the control group. Defaults to None.
        show (bool, optional): Whether to show the plot. Defaults to True.
        savepath (str, optional): The path to save the plot. Defaults to None.
        **kwargs: Additional keyword arguments to pass to the Seaborn plotting function.

    Returns:
        matplotlib Axes: The plot axes.
    """

    # Apply general styling template
    if style_template in styling_templates:
        plt.rcParams.update(styling_templates[style_template]["general"])
    else:
        raise ValueError(
            f"Invalid style_template. Choose from {list(styling_templates.keys())}"
        )

    # Clean the data by removing NaN values for the given metric
    data = clean_data(data, metric)

    # Get the labels and sample size for each group
    data, kdims = compute_sample_size(data, metric, kdims, groupby=group_by)

    # Sort the data if specified
    if sort_by:
        data = sort_data(data, group_by=kdims, metric=metric, sort_by=sort_by)

    # Apply plot-specific parameters
    plot_specific_params = styling_templates[style_template].get(plot_type, {})
    kwargs.update(plot_specific_params)

    # Check for bs_notch parameter
    bs_notch = styling_templates[style_template].get("bs_notch", False)
    if bs_notch:
        ci_dict = compute_group_bs_ci(data, metric, kdims)

    # Make the plot
    if plot_type == "scatter":
        ax = sns.stripplot(data=data, x=kdims, y=metric, hue=colorby, **kwargs)
    elif plot_type == "boxplot":
        ax = sns.boxplot(data=data, x=kdims, y=metric, hue=colorby, **kwargs)
        if bs_notch:
            for i, artist in enumerate(ax.artists):
                group = artist.get_label()
                if group in ci_dict:
                    lower, upper = ci_dict[group]
                    median = data[data[kdims] == group][metric].median()
                    artist.set_edgecolor("black")
                    artist.set_linewidth(2)
                    ax.plot(
                        [i - 0.2, i + 0.2],
                        [lower, lower],
                        color="black",
                        linestyle="--",
                    )
                    ax.plot(
                        [i - 0.2, i + 0.2],
                        [upper, upper],
                        color="black",
                        linestyle="--",
                    )
                    ax.plot([i, i], [lower, upper], color="black", linestyle="--")
    elif plot_type == "violinplot":
        ax = sns.violinplot(data=data, x=kdims, y=metric, hue=colorby, **kwargs)
    elif plot_type == "jitterboxplot":
        # Apply boxplot-specific parameters
        boxplot_params = styling_templates[style_template].get("boxplot", {})
        ax = sns.boxplot(
            data=data, x=kdims, y=metric, hue=colorby, **{**kwargs, **boxplot_params}
        )
        if bs_notch:
            for i, artist in enumerate(ax.artists):
                group = artist.get_label()
                if group in ci_dict:
                    lower, upper = ci_dict[group]
                    median = data[data[kdims] == group][metric].median()
                    artist.set_edgecolor("black")
                    artist.set_linewidth(2)
                    ax.plot(
                        [i - 0.2, i + 0.2],
                        [lower, lower],
                        color="black",
                        linestyle="--",
                    )
                    ax.plot(
                        [i - 0.2, i + 0.2],
                        [upper, upper],
                        color="black",
                        linestyle="--",
                    )
                    ax.plot([i, i], [lower, upper], color="black", linestyle="--")

        # Apply stripplot-specific parameters
        stripplot_params = styling_templates[style_template].get("stripplot", {})
        sns.stripplot(
            data=data,
            x=kdims,
            y=metric,
            hue=colorby,
            color="black",
            dodge=True if group_by else False,
            ax=ax,
            **{**kwargs, **stripplot_params},
        )

        handles, labels = plt.gca().get_legend_handles_labels()
        plt.legend(
            handles[0 : len(handles) // 2], labels[0 : len(labels) // 2], title=colorby
        )
    else:
        raise ValueError(
            "Invalid plot type. Choose either 'scatter', 'boxplot', 'violinplot', or 'jitterboxplot'."
        )

    # Compute and add hline if specified
    if hline_method and control_label:
        hline_values = compute_hline_values(
            data,
            metric=metric,
            hline_method=hline_method,
            control_label=control_label,
            kdims=kdims,
        )
        for hline in hline_values:
            plt.axhline(y=hline, color="r", linestyle="--")

    if show:
        plt.show()

    if savepath:
        plt.savefig(savepath)

    return ax
