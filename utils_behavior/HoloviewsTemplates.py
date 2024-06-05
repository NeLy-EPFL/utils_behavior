from bokeh.models import HoverTool

import holoviews as hv
from . import Utils
from . import Processing
import datetime
from icecream import ic
import pandas as pd

# My main template

hv_main = {
    "boxwhisker": {
        "box_fill_color": None,
        "box_line_color": "black",
        "outlier_fill_color": None,
        "outlier_line_color": None,
        "framewise": True,
    },
    "scatter": {
        "jitter": 0.3,
        # "color": "label",
        "alpha": 0.5,
        "size": 6,
        "cmap": "Category10",
        "framewise": True,
    },
    "plot": {
        "width": 750,
        "height": 500,
        "show_legend": False,
        "xlabel": "",
        "invert_axes": True,
        "show_grid": True,
        "fontscale": 1.5,
        "title": "",
    },
}

# hv_slides is optimised for nice rendering on a ppt or keynote presentation. Essentially plots are bigger and lines are thicker to make it more readable.

hv_slides = {
    "boxwhisker": {
        "box_fill_color": None,
        "box_line_color": "black",
        "outlier_fill_color": None,
        "outlier_line_color": None,
        "box_line_width": 2,
        "whisker_line_width": 2,
        "framewise": True,
    },
    "scatter": {
        "jitter": 0.3,
        # "color": "label",
        "alpha": 0.7,
        "size": 8,
        "cmap": "Category10",
        "framewise": True,
    },
    "plot": {
        "width": 1500,
        "height": 1000,
        "show_legend": False,
        "xlabel": "",
        "invert_axes": True,
        "show_grid": True,
        "fontscale": 2.5,
        "title": "",
    },
}

hv_irene = {
    "boxwhisker": {
        "box_fill_color": None,
        "box_line_color": "black",
        "outlier_fill_color": None,
        "outlier_line_color": None,
        "framewise": True,
    },
    "scatter": {
        "jitter": 0.3,
        "alpha": 0.5,
        "size": 6,
        "cmap": "Category10",
        "framewise": True,
    },
    "plot": {
        "width": 500,
        "height": 500,
        "show_legend": False,
        "xlabel": "",
        "invert_axes": True,
        "show_grid": True,
        "fontscale": 1,
        "title": "",
        "active_tools": [],
    },
}

# Custom Jitterboxplot function


def clean_data(data, vdim, groupby=None):

    if groupby:
        # Filter out groups where the vdim column is all NaN
        data = data.groupby(groupby).filter(lambda x: x[vdim].notna().any())
    else:
        # Filter out groups where the vdim column is all NaN
        data = data[data[vdim].notna()]

    # Clean the data by removing NaN values for this metric
    data = data.dropna(subset=[vdim])

    # Convert vdim to numeric
    data[vdim] = pd.to_numeric(data[vdim], errors="coerce")

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


def compute_hline_values(data, vdim, hline_method):
    if hline_method == "bootstrap":
        return compute_controls_bs_ci(data, vdim)
    elif hline_method == "boxplot":
        # Calculate 25% and 75% quantiles for the control group
        return (
            data[vdim].quantile(0.25),
            data[vdim].quantile(0.75),
        )
    else:
        raise ValueError(
            "Invalid hline_method. Choose either 'bootstrap' or 'boxplot'."
        )


def create_jitterboxplot(
    data,
    vdim,
    kdims,
    plot_options,
    y_min,
    y_max,
    hover,
    metadata=None,
    scatter_color="black",
    control=False,
):
    """Helper function to create a plot."""
    scatterplot = hv.Scatter(
        data=data,
        vdims=[vdim] + (metadata if metadata is not None else []) + ["fly"],
        kdims=[kdims],
    ).opts(
        **plot_options["scatter"],
        color=scatter_color,
        tools=[hover],
        ylim=(y_min, y_max),
    )

    boxplot = hv.BoxWhisker(
        data=data,
        vdims=vdim,
        kdims=[kdims],
    ).opts(**plot_options["boxwhisker"], ylim=(y_min, y_max))

    if control:
        boxplot = boxplot.opts(box_line_color="green")
        scatterplot = scatterplot.opts(color="green")

    return boxplot, scatterplot


def create_groupby_plots(
    data,
    kdims,
    vdim,
    groupby,
    control=None,
    sort_by=None,
    scale_max=False,
    metadata=None,
    hline=None,  # Changed hline_values to hline
    plot_options=hv_main,
    layout=False,
):
    data = clean_data(data, vdim, groupby)

    # Pre-calculate common values
    y_max = data[vdim].quantile(0.95) if scale_max else data[vdim].max()
    y_min = data[vdim].min()

    # Ensure control is a list
    if control and not isinstance(control, list):
        control = [control]

    # Use control argument to get control_data
    if control:
        control_data = data[data[kdims].isin(control)]
    else:
        control_data = None

    # Get the group value for the control group
    if control:
        control_group = control_data[groupby].unique()[0]
        print(control_group)

    hline_values = None  # Initialize hline_values
    if control and hline:  # Changed hline_values to hline
        hline_values = compute_hline_values(
            control_data,
            vdim,
            hline,
        )

    plots = {}
    for group in data[groupby].unique():
        # Use list comprehension for tooltips
        tooltips = [
            ("Fly", "@fly"),
            (vdim.capitalize(), f"@{vdim}"),
        ]
        if metadata is not None:
            tooltips.extend([(var.capitalize(), f"@{var}") for var in metadata])

        hover = HoverTool(tooltips=tooltips)
        group_data = data[data[groupby] == group]

        if sort_by == "median":
            median_values = group_data.groupby(kdims)[vdim].median().sort_values()
            group_data["median"] = group_data[kdims].map(median_values)
            group_data = group_data.sort_values("median")

        else:
            # Return the original order with a warning message
            print(
                "Invalid sort_by option. No sorting applied. The data will be displayed in the original order."
            )

        boxplot, scatterplot = create_jitterboxplot(
            data=group_data,
            vdim=vdim,
            kdims=kdims,
            plot_options=plot_options,
            y_min=y_min,
            y_max=y_max,
            hover=hover,
            metadata=metadata,
            scatter_color=kdims,
        )

        if control and group != control_group:
            control_boxplot, control_scatterplot = create_jitterboxplot(
                data=control_data,
                vdim=vdim,
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
                ).opts(ylabel=f"{vdim}", **plot_options["plot"])
            else:
                plot = (
                    boxplot * scatterplot * control_boxplot * control_scatterplot
                ).opts(ylabel=f"{vdim}", **plot_options["plot"])
        else:
            if hline_values is not None:
                plot = (hv_hline * boxplot * scatterplot).opts(
                    ylabel=f"{vdim}", **plot_options["plot"]
                )
            else:
                plot = (boxplot * scatterplot).opts(
                    ylabel=f"{vdim}", **plot_options["plot"]
                )

        plots[group] = plot

    # Create a Layout or a HoloMap with the jitter boxplots for each group
    if layout:
        jitter_boxplot = hv.Layout(plots).cols(2)
    else:
        jitter_boxplot = hv.HoloMap(plots, kdims=[groupby])

    return jitter_boxplot


def jitter_boxplot(
    data,
    vdim,
    metadata,
    folder,
    name=None,
    kdims="Genotype",
    groupby="Brain region",
    bs_controls=True,
    plot_options=hv_main,
    show=True,
    save=False,
    outpath=None,
    sort_by=None,
    hline_method="bootstrap",
    readme=None,
    layout=False,
):
    # Filter out groups where the vdim column is all NaN
    data = data.groupby(groupby).filter(lambda x: x[vdim].notna().any())

    # Clean the data by removing NaN values for this metric
    data = data.dropna(subset=[vdim])

    # Convert vdim to numeric
    data[vdim] = pd.to_numeric(data[vdim], errors="coerce")

    # Get the metadata for the tooltips
    tooltips = [
        ("Fly", "@fly"),
        (vdim.capitalize(), f"@{vdim}"),
    ]

    # Add the metadata to the tooltips
    for var in metadata:
        tooltips.append((var.capitalize(), f"@{var}"))

    hover = HoverTool(tooltips=tooltips)

    # Compute the bootstrap confidence interval for the metric
    if hline_method == "bootstrap":
        if bs_controls:
            hline_values = compute_controls_bs_ci(data, vdim, kdims=kdims)
        else:
            hline_values = None
    elif hline_method == "boxplot":
        # Calculate 25% and 75% quantiles for the control group
        control_data = data[data["Genotype"] == "TNTxZ2018"]
        hline_values = (
            control_data[vdim].quantile(0.25),
            control_data[vdim].quantile(0.75),
        )
    else:
        raise ValueError(
            "Invalid hline_method. Choose either 'bootstrap' or 'boxplot'."
        )

    # Get the limits for the y axis
    y_min = data[vdim].min()
    # For y_max, use the 95th percentile of the data
    y_max = data[vdim].max()
    # y_max = data[vdim].quantile(0.95)

    # Group the data by groupby and 'label'
    grouped_data = data.groupby([groupby, kdims])

    # Define a function that creates a BoxWhisker and a Scatter plot for a given group

    def create_plots(region):
        region_data = data[data[groupby] == region]

        # If sort_by is set to 'median', sort the region_data by the median of vdim grouped by label
        if sort_by == "median":
            median_values = region_data.groupby(kdims)[vdim].median().sort_values()
            region_data["median"] = region_data[kdims].map(median_values)
            region_data = region_data.sort_values("median")

        boxplot = hv.BoxWhisker(
            data=region_data,
            vdims=vdim,
            kdims=[kdims],
        ).opts(**plot_options["boxwhisker"], ylim=(y_min, y_max))

        scatterplot = hv.Scatter(
            data=region_data,
            vdims=[vdim] + metadata + ["fly"],
            kdims=[kdims],
        ).opts(
            **plot_options["scatter"], color=kdims, tools=[hover], ylim=(y_min, y_max)
        )

        if region != "Control":
            control_boxplot = hv.BoxWhisker(
                data=control_data,
                vdims=vdim,
                kdims=[kdims],
            ).opts(box_fill_color=None, box_line_color="green", ylim=(y_min, y_max))

            control_scatterplot = hv.Scatter(
                data=control_data,
                vdims=[vdim] + metadata + ["fly"],
                kdims=[kdims],
            ).opts(
                alpha=0.5,
                jitter=0.3,
                size=6,
                color="black",
                tools=[hover],
                ylim=(y_min, y_max),
            )

        if hline_values is not None:
            # Create an Area plot for the confidence interval
            hv_hline = hv.HSpan(hline_values[0], hline_values[1]).opts(
                fill_alpha=0.2, color="red"
            )

        if region != "Control":
            if hline_values is not None:
                return (
                    hv_hline
                    * boxplot
                    * scatterplot
                    * control_boxplot
                    * control_scatterplot
                ).opts(ylabel=f"{vdim}", **plot_options["plot"])
            else:
                return (
                    boxplot * scatterplot * control_boxplot * control_scatterplot
                ).opts(ylabel=f"{vdim}", **plot_options["plot"])
        else:
            if hline_values is not None:
                return (hv_hline * boxplot * scatterplot).opts(
                    ylabel=f"{vdim}", **plot_options["plot"]
                )
            else:
                return (boxplot * scatterplot).opts(
                    ylabel=f"{vdim}", **plot_options["plot"]
                )

    # If layout is True, create a Layout with the jitter boxplots for each group
    if layout:
        jitter_boxplot = hv.Layout(
            {region: create_plots(region) for region in data[groupby].unique()}
        ).cols(2)

    else:

        # Create a HoloMap
        jitter_boxplot = hv.HoloMap(
            {region: create_plots(region) for region in data[groupby].unique()},
            kdims=[groupby],
        )

    if readme is not None:
        # Add the readme text to the plot
        readme_text = hv.Text(0, 0, readme).opts()
        jitter_boxplot = jitter_boxplot + readme_text

    if show:
        hv.render(jitter_boxplot)
    if save:
        if outpath is None:
            now = datetime.datetime.now()  # get current date and time
            date_time = now.strftime("%Y%m%d_%H%M")  # format as a string

            if name:
                filename = name
            else:
                filename = vdim

            output_path = (
                Utils.get_labserver()
                / "Experimental_data"
                / "MultiMazeRecorder"
                / "Plots"
                / folder
                / f"{filename}_{date_time}.html"
            )

        hv.save(jitter_boxplot, output_path)

    return jitter_boxplot
