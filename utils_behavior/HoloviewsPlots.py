from bokeh.models import HoverTool

import numpy as np
import holoviews as hv
from . import Utils
from . import Processing
from .HoloviewsTemplates import hv_main, pooled_opts
import datetime
from icecream import ic
import pandas as pd
import colorcet as cc
from bokeh.palettes import Category10
from bokeh.palettes import all_palettes

# Custom Jitterboxplot function


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


def compute_controls_bs_ci(data, metric):
    """
    Compute a 95% bootstrap confidence interval for a given metric for flies with specified genotypes. The default usage is to compute the confidence interval for the control genotypes, which are the ones set as default in the genotypes argument. Currently excluded Genotypes that are technically controls : , "TNTxZ2035", "TNTxM6", "TNTxM7"

    Args:
        data (pandas DataFrame): The dataset to compute the confidence interval from.
        metric (str): The metric to compute the confidence interval for.
        ctrl_label (str): The label of the control group.
        kdims (str): The column name to filter by.

    Returns:
        np.ndarray: The lower and upper confidence interval bounds.
    """

    # Drop rows with NaN values in the metric column
    data = data.dropna(subset=[metric])

    # Compute the bootstrap confidence interval for the given metric
    ci = Processing.draw_bs_ci(data[metric].values)

    return ci


def compute_hline_values(data, metric, hline_method):
    """
    Compute the values for the control area in the jitter boxplot. The values can be computed using either a bootstrap confidence interval or the 25% and 75% quantiles of the control group.

    Args:
        data (pandas DataFrame): The dataset to compute the values from.
        metric (str): The metric to compute the values for.
        hline_method (str): The method to compute the values. Either 'bootstrap' or 'boxplot'.

    """
    if hline_method == "bootstrap":
        return compute_controls_bs_ci(data, metric)
    elif hline_method == "boxplot":
        # Calculate 25% and 75% quantiles for the control group
        return (
            data[metric].quantile(0.25),
            data[metric].quantile(0.75),
        )
    else:
        raise ValueError(
            "Invalid hline_method. Choose either 'bootstrap' or 'boxplot'."
        )


def sort_data(data, group_by, metric, sort_by="default"):
    """
    Sort the data by the median of the metric for each group in group_by.

    Args:
        data (pandas DataFrame): The dataset to sort.
        group_by (str or list): The column(s) to group by.
        metric (str): The metric to sort by.
        sort_by (str): The method to sort the data. Either 'median', 'original', or 'default'.
    """
    if isinstance(group_by, list):
        if sort_by == "median":
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

        elif sort_by == "default":
            # Sort by kdims in the original order
            data.sort_values(by=group_by, inplace=True)

        else:
            # Return the original order with a warning message
            print(
                "Invalid sort_by option. No sorting applied. The data will be displayed in the original order."
            )

    elif isinstance(group_by, str):
        if sort_by == "median":
            median_values = data.groupby(group_by)[metric].median().sort_values()
            data["median"] = data[group_by].map(median_values)
            data = data.sort_values("median")
        elif sort_by == "default":
            # Sort by kdims in the original order
            data.sort_values(by=group_by, inplace=True)
        else:
            # Return the original order with a warning message
            print(
                "Invalid sort_by option. No sorting applied. The data will be displayed in the original order."
            )

    return data


def create_jitterboxplot(
    data,
    metric,
    kdims,
    plot_options,
    y_min,
    y_max,
    hover,
    metadata=None,
    scatter_color="black",
    control=False,
    colorby=None,
):
    """
    Create a jitter boxplot with a scatterplot overlay for a given group.

    Args:
        data (pandas DataFrame): The dataset to create the plot from.
        metric (str): The metric to plot.
        kdims (str): The column to plot on the x-axis.
        plot_options (dict): The plot options to use.
        y_min (float): The minimum value for the y-axis.
        y_max (float): The maximum value for the y-axis.
        hover (HoverTool): The hover tool to use.
        metadata (list): The metadata columns to include in the tooltips.
        scatter_color (str): The color to use for the scatterplot.
        control (bool): Whether the group is the control group.
        colorby (str): The column to color by.
    """
    boxplot = hv.BoxWhisker(
        data=data,
        vdims=metric,
        kdims=[kdims],
    ).opts(**plot_options["boxwhisker"], ylim=(y_min, y_max))

    scatterplot = hv.Scatter(
        data=data,
        vdims=[metric]
        + (
            [colorby]
            if colorby is not None
            and colorby not in (metadata if metadata is not None else [])
            else []
        )
        + (metadata if metadata is not None else [])
        # + ["fly"]
        ,
        kdims=[kdims],
    ).opts(
        **plot_options["scatter"],
        color=scatter_color,
        tools=[hover],
        ylim=(y_min, y_max),
    )

    color_column = colorby if colorby else kdims
    unique_values = data[color_column].unique()

    # Get the colormap name from the plot options
    cmap_name = plot_options["scatter"]["cmap"]

    # Get the colormap from bokeh.palettes
    cmap = all_palettes[cmap_name][10]  # Adjust the number as needed

    color_mapping = {
        value: cmap[i % len(cmap)] for i, value in enumerate(unique_values)
    }

    scatterplot = scatterplot.opts(color=hv.dim(color_column).categorize(color_mapping))

    if control:
        boxplot = boxplot.opts(box_line_color="green")
        scatterplot = scatterplot.opts(color="green")

    return boxplot, scatterplot


def create_single_jitterboxplot(
    data,
    kdims,
    metric,
    groupby,
    control=None,
    sort_by=None,
    scale_max=False,
    metadata=None,
    hline=None,
    plot_options=hv_main,
    colorby=None,
    debug=False,
):
    data = clean_data(data, metric, groupby)

    # If sort_by is set to 'median', sort the data by the median of the metric for each group in groupby
    if sort_by == "median":
        data = sort_data(data, groupby, metric, sort_by)

    # Pre-calculate common values
    y_max = data[metric].quantile(0.95) if scale_max else data[metric].max()
    y_min = data[metric].min()

    # Ensure control is a list
    if control and not isinstance(control, list):
        control = [control]

    # Use control argument to get control_data
    if control and control is not None:
        control_data = data[data[kdims].isin(control)]
        if debug:
            print(f"Control data size: {len(control_data)}")  # Debug print
    else:
        control_data = None

    hline_values = None  # Initialize hline_values
    if control and hline:  # Changed hline_values to hline
        hline_values = compute_hline_values(
            control_data,
            metric,
            hline,
        )

    tooltips = [
        # ("Fly", "@fly"),
        (metric.capitalize(), f"@{metric}"),
    ]
    if metadata is not None:
        tooltips.extend([(var.capitalize(), f"@{var}") for var in metadata])

    hover = HoverTool(tooltips=tooltips)

    y_max = data[metric].quantile(0.95) if scale_max else data[metric].max()
    y_min = data[metric].min()

    boxplot, scatterplot = create_jitterboxplot(
        data=data,
        metric=metric,
        kdims=kdims,
        plot_options=plot_options,
        colorby=colorby,
        y_min=y_min,
        y_max=y_max,
        hover=hover,
        metadata=metadata,
        scatter_color=kdims,
    )

    if hline_values is not None:
        hv_hline = hv.HSpan(hline_values[0], hline_values[1]).opts(
            fill_alpha=0.2, color="red"
        )

        jitter_boxplot = (boxplot * scatterplot * hv_hline).opts(
            ylabel=f"{metric}", **plot_options["plot"]
        )
    else:
        jitter_boxplot = (boxplot * scatterplot).opts(
            ylabel=f"{metric}", **plot_options["plot"]
        )

    return jitter_boxplot


def create_groupby_jitterboxplots(
    data,
    kdims,
    metric,
    groupby,
    control=None,
    sort_by=None,
    scale_max=False,
    metadata=None,
    hline=None,
    plot_options=hv_main,
    colorby=None,
    layout=False,
    layout_cols=2,  # Default to 2 columns
    debug=False,
):
    data = clean_data(data, metric, groupby)

    # If groupby is a list, create an interaction term
    if isinstance(groupby, list):
        data["interaction"] = data[groupby].apply(lambda x: "-".join(x), axis=1)
        groupby = "interaction"

    # Pre-calculate common values
    y_max = data[metric].quantile(0.95) if scale_max else data[metric].max()
    y_min = data[metric].min()

    # Ensure control is a list
    if control and not isinstance(control, list):
        control = [control]

    # Use control argument to get control_data
    if control and control is not None:
        control_data = data[data[kdims].isin(control)]
        if debug:
            print(f"Control data size: {len(control_data)}")  # Debug print
    else:
        control_data = None

    # Get the group value for the control group
    if control:
        control_group = control_data[groupby].unique()[0]
        if debug:
            print(f"Control group: {control_group}")  # Debug print

    hline_values = None  # Initialize hline_values
    if control and hline:  # Changed hline_values to hline
        hline_values = compute_hline_values(
            control_data,
            metric,
            hline,
        )

    plots = {}
    for group in data[groupby].unique():
        if debug:
            print(f"Processing group: {group}")  # Debug print
        # Use list comprehension for tooltips
        tooltips = [
            # ("Fly", "@fly"),
            (metric.capitalize(), f"@{metric}"),
        ]
        if metadata is not None:
            tooltips.extend([(var.capitalize(), f"@{var}") for var in metadata])

        hover = HoverTool(tooltips=tooltips)
        group_data = data[data[groupby] == group]

        group_data = sort_data(group_data, kdims, metric, sort_by)

        boxplot, scatterplot = create_jitterboxplot(
            data=group_data,
            metric=metric,
            kdims=kdims,
            plot_options=plot_options,
            colorby=colorby,
            y_min=y_min,
            y_max=y_max,
            hover=hover,
            metadata=metadata,
            scatter_color=kdims,
        )

        if control and group != control_group:
            control_boxplot, control_scatterplot = create_jitterboxplot(
                data=control_data,
                metric=metric,
                kdims=kdims,
                plot_options=plot_options,
                y_min=y_min,
                y_max=y_max,
                hover=hover,
                metadata=metadata,
                scatter_color="green",
            )

        if hline_values is not None:
            hv_hline = hv.HSpan(hline_values[0], hline_values[1]).opts(
                fill_alpha=0.2, color="red"
            )

        if control and group != control_group:
            if hline_values is not None:
                plot = (
                    hv_hline
                    * boxplot
                    * scatterplot
                    * control_boxplot
                    * control_scatterplot
                ).opts(ylabel=f"{metric}", title=str(group), **plot_options["plot"])
            else:
                plot = (
                    boxplot * scatterplot * control_boxplot * control_scatterplot
                ).opts(ylabel=f"{metric}", title=str(group), **plot_options["plot"])
        else:
            if hline_values is not None:
                plot = (hv_hline * boxplot * scatterplot).opts(
                    ylabel=f"{metric}", title=str(group), **plot_options["plot"]
                )
            else:
                plot = (boxplot * scatterplot).opts(
                    ylabel=f"{metric}", title=str(group), **plot_options["plot"]
                )

        plots[group] = plot

    # Create a Layout or a HoloMap with the jitter boxplots for each group
    if layout:
        cols = layout_cols
        jitter_boxplot = hv.Layout(plots.values()).cols(cols).opts(shared_axes=False)
    else:
        jitter_boxplot = hv.HoloMap(plots, kdims=[groupby])

    return jitter_boxplot


def create_pooled_jitterboxplot(
    data,
    metric,
    kdims,
    sort_by,
    control=None,
    hline=None,
    metadata=None,
    scale_max=False,
    plot_options=pooled_opts,
    groupby=None,
    colorby=None,
):

    data = clean_data(data, metric)

    # Sort the data
    data = sort_data(data, groupby, metric, sort_by)

    # Ensure control is a list if it's not None
    if control is not None and not isinstance(control, list):
        control = [control]

    # Filter control data only if control is not None
    if control is not None:
        control_data = data[data[kdims].isin(control)]
    else:
        control_data = pd.DataFrame()  # Empty DataFrame if control is None

    hline_values = None  # Initialize hline_values
    if control and hline:  # Changed hline_values to hline
        hline_values = compute_hline_values(
            control_data,
            metric,
            hline,
        )

    # Get the limits for the y axis
    y_min = data[metric].min()

    if scale_max:
        y_max = data[metric].quantile(0.95)
    else:
        y_max = data[metric].max()

    # Create the hover tool
    tooltips = [
        # ("Fly", "@fly"),
        (metric.capitalize(), f"@{metric}"),
    ]

    hover = HoverTool(tooltips=tooltips)

    # Determine the column to use for coloring
    if colorby is None and groupby is not None:
        color_column = groupby
    elif colorby == kdims:
        color_column = "label"
    else:
        color_column = colorby

    # Modify plot_options for scatter and boxwhisker
    scatter_cmap = plot_options["scatter"]["cmap"]
    box_colors = hv.Cycle(scatter_cmap)

    plot_options["scatter"]["color"] = "black"
    plot_options["boxwhisker"]["box_fill_color"] = box_colors

    # Create the BoxWhisker plot and the Scatter plot
    if color_column:
        groups = data[color_column].unique()

        boxplot = hv.Overlay(
            [
                hv.BoxWhisker(
                    data[data[color_column] == group],
                    kdims=kdims,
                    vdims=metric,
                ).opts(**plot_options["boxwhisker"])
                for group in groups
            ]
        )

        scatterplot = hv.Scatter(
            data=data,
            vdims=[metric] + (metadata if metadata is not None else []),
            kdims=[kdims],
        ).opts(**plot_options["scatter"], tools=[hover], ylim=(y_min, y_max))

    else:
        boxplot = hv.BoxWhisker(data, kdims=kdims, vdims=metric).opts(
            **plot_options["boxwhisker"]
        )

        scatterplot = hv.Scatter(
            data=data,
            vdims=[metric] + (metadata if metadata is not None else []),
            kdims=[kdims],
        ).opts(**plot_options["scatter"], tools=[hover], ylim=(y_min, y_max))

    # Create the hline
    if hline_values is not None:
        hv_hline = hv.HSpan(hline_values[0], hline_values[1]).opts(
            fill_alpha=0.2, color="red"
        )
        # Create the final plot
        jitter_boxplot = (
            (hv_hline * boxplot * scatterplot)
            .opts(ylabel=f"{metric}", **plot_options["plot"])
            .opts(show_grid=False)
        )
    else:
        # Create the final plot without hline
        jitter_boxplot = (
            (boxplot * scatterplot)
            .opts(ylabel=f"{metric}", **plot_options["plot"])
            .opts(show_grid=False)
        )

    return jitter_boxplot


def jitter_boxplot(
    data,
    metric,
    kdims,
    sort_by="median",
    control=None,
    hline=None,
    metadata=None,
    scale_max=False,
    plot_options=None,
    groupby=None,
    colorby=None,
    render="single",
    sample_size=True,
    layout_cols=(4, 1),
):
    if sample_size:
        # Determine the grouping columns based on the presence of groupby
        grouping_columns = [kdims] if groupby is None else [kdims, groupby]

        # Filter out rows with NAs in the metric column
        data_filtered = data.dropna(subset=[metric])

        # Group by the determined columns and calculate the size of each group
        sample_size = data_filtered.groupby(grouping_columns).size().reset_index(name="size")

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

    if plot_options is None:
        if render == "pooled":
            plot_options = pooled_opts
        else:
            plot_options = hv_main

    if render == "pooled":
        return create_pooled_jitterboxplot(
            data,
            metric,
            kdims,
            sort_by,
            control,
            hline,
            metadata,
            scale_max,
            plot_options,
            groupby,
            colorby=colorby,
        )
    elif render == "grouped":
        return create_groupby_jitterboxplots(
            data,
            kdims,
            metric,
            groupby,
            control,
            sort_by,
            scale_max,
            metadata,
            hline,
            plot_options,
            colorby=colorby,
            layout=False,
        )
    elif render == "layout":
        return create_groupby_jitterboxplots(
            data,
            kdims,
            metric,
            groupby,
            control,
            sort_by,
            scale_max,
            metadata,
            hline,
            plot_options,
            colorby=colorby,
            layout=True,
            layout_cols=layout_cols,
        )
    elif render == "single":
        return create_single_jitterboxplot(
            data,
            kdims,
            metric,
            groupby,
            control,
            sort_by,
            scale_max,
            metadata,
            hline,
            plot_options,
            colorby,
        )
    else:
        raise ValueError(
            "Invalid render option. Choose from 'pooled', 'grouped', or 'layout'."
        )


# TODO: SOlve bad colorby with pooled plots.
# TODO: handle the case where colorby is passed to the groupby jitterboxplot function
# TODO: add saving , showing and outpath options to the jitterboxplot functions
# TODO: better variable names.


def histograms(
    data,
    metric,
    categories,
    bins=None,
    xlabel=None,
    plot_options=hv_main,
    orientation="vertical",
):
    # Get the color map from the plot options
    unique_values = data[categories].unique()
    color_map = plot_options["hist"]["cmap"]

    cmap = all_palettes[color_map][10]

    # Get the color palette from the color map
    color_palette = {
        value: cmap[i % len(cmap)] for i, value in enumerate(unique_values)
    }

    def hist(data, bins=bins):
        """Compute bins and edges for histogram using Freedman-Diaconis rule"""
        h = 2 * (np.percentile(data, 75) - np.percentile(data, 25)) / np.cbrt(len(data))

        if bins is None:
            bins = int(np.ceil((data.max() - data.min()) / h))
        else:
            bins = bins

        return np.histogram(data, bins=bins)

    hists = [
        hv.Histogram(hist(group[metric]), kdims=metric, vdims="count", label=label)
        .opts(**plot_options["hist"])
        .opts(
            color=color_palette[label],
            title=label,
            xlabel=metric if xlabel is None else xlabel,
        )
        for label, group in data.groupby(categories, sort=False)
    ]

    if orientation == "horizontal":
        return hv.Layout(hists).cols(len(hists))
    else:
        return hv.Layout(hists).cols(1)
