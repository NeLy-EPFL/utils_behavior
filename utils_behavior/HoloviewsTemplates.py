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
    "hist": {
        "cmap": "Category10",
        "fill_alpha": 0.5,
        "frame_height": 500,
        "frame_width": 750,
        "line_alpha": 0,
        "show_grid": True,
        "title": "HoloViews",
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
    "hist": {
        "cmap": "Category10",
        "fill_alpha": 0.5,
        "height": 1000,
        "width": 1500,
        "fontscale": 2.5,
        "line_alpha": 0,
        "show_grid": True,
        "title": "HoloViews",
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

# Pooled jitterboxplots : These templates are optimised for bigger plots where all the data is plotted at once instead of as subplots

pooled_opts = {
    "boxwhisker": {
        # "box_fill_color": None,
        # "box_line_color": "black",
        "outlier_fill_color": None,
        "outlier_line_color": None,
        "framewise": True,
    },
    "scatter": {
        "jitter": 0.15,
        "color": "black",
        "alpha": 0.8,
        "size": 2,
        "cmap": "Category10",
        "framewise": True,
    },
    "plot": {
        "width": 1100,
        "height": 1423,
        "show_legend": False,
        "xlabel": "",
        "invert_axes": True,
        "show_grid": True,
        "fontscale": 1,
        "title": "",
    },
}

# A template optimised for jupyter notebooks

hv_jupyter = {
    "boxwhisker": {
        "box_fill_color": None,
        "box_line_color": "black",
        "outlier_fill_color": None,
        "outlier_line_color": None,
        "framewise": True,
        "width": 600,
        "height": 400,
    },
    "scatter": {
        "jitter": 0.3,
        "alpha": 0.5,
        "size": 6,
        "cmap": "Category10",
        "framewise": True,
        "width": 600,
        "height": 400,
    },
    "plot": {
        "width": 600,
        "height": 400,
        "show_legend": True,
        "xlabel": "",
        "invert_axes": False,
        "show_grid": True,
        "fontscale": 1.2,
    },
    "hist": {
        "cmap": "Category10",
        "fill_alpha": 0.5,
        "frame_height": 400,
        "frame_width": 600,
        "line_alpha": 0,
        "show_grid": True,
    },
}