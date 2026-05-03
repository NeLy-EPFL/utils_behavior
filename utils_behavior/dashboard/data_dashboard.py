"""
Data Exploration Dashboard

This script creates an interactive dashboard for exploring datasets using Shiny for Python.

Usage:
    1. From the command line:
       python data_explorer.py path/to/your/data.csv

    2. From another Python script:
       from data_explorer import create_app, run_app

       # With a DataFrame
       df = pd.DataFrame(...)
       app = create_app(df)
       run_app(app)

       # Or using the main function
       from data_explorer import main
       main("path/to/your/data.csv")  # With CSV file
       main(df)  # With DataFrame

Features:
    - Automatically detects numeric and categorical columns
    - Supports various plot types: scatter, box, violin, histogram, bar
    - Allows grouping by categorical variables
    - Option to show marginal plots for scatter plots

Requirements:
    - Python 3.6+
    - pandas
    - seaborn
    - matplotlib
    - shiny

For more information on usage and features, run:
    python data_explorer.py --help
"""

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from shiny import App, Inputs, Outputs, Session, reactive, render, req, ui, run_app
import argparse
import sys
import io
import base64
import pyperclip

# Run the app in the browser
import nest_asyncio

# Apply the nest_asyncio patch
nest_asyncio.apply()

sns.set_theme()


def create_dashboard(data: pd.DataFrame):
    """Create a Shiny app for data exploration.

    Args:
        data (pd.DataFrame): Input dataset to explore.

    Returns:
        App: A Shiny app instance.
    """
    # Identify numeric and categorical columns
    numeric_cols = data.select_dtypes(include=["float64", "int64"]).columns.tolist()
    cat_cols = data.select_dtypes(include=["object", "category"]).columns.tolist()

    app_ui = ui.page_sidebar(
        ui.sidebar(
            ui.input_selectize("xvar", "X variable", numeric_cols + cat_cols),
            ui.input_selectize("yvar", "Y variable", numeric_cols + cat_cols),
            ui.input_selectize("group_var", "Group by", cat_cols, multiple=True),
            ui.hr(),
            ui.input_select(
                "plot_type",
                "Plot type",
                ["scatter", "box", "violin", "histogram"],
            ),
            ui.input_switch("show_margins", "Show marginal plots", value=True),
            ui.hr(),
            ui.input_select("save_format", "Save format", ["png", "svg"]),
            ui.download_button("save_plot", "Save Plot"),
            ui.hr(),
            ui.input_select(
                "export_format", "Export format", ["Python script", "Clipboard"]
            ),
            ui.input_action_button("export_code", "Export Source Code"),
        ),
        ui.card(
            ui.output_plot("main_plot"),
        ),
        ui.output_text("export_message"),
    )

    def server(input: Inputs, output: Outputs, session: Session):
        @reactive.Calc
        def filtered_df() -> pd.DataFrame:
            return data

        @reactive.Calc
        def generate_plot():
            fig, ax = plt.subplots(figsize=(10, 6))

            x = input.xvar()
            y = input.yvar()
            group = input.group_var() if input.group_var() else None

            if input.plot_type() == "scatter":
                if input.show_margins():
                    g = sns.jointplot(
                        data=filtered_df(), x=x, y=y, hue=group[0] if group else None
                    )
                    return g.figure
                else:
                    sns.scatterplot(
                        data=filtered_df(),
                        x=x,
                        y=y,
                        hue=group[0] if group else None,
                        ax=ax,
                    )
            elif input.plot_type() == "box":
                sns.boxplot(
                    data=filtered_df(),
                    x=x,
                    y=y,
                    hue=group[0] if group else None,
                    fliersize=0,
                    gap=0.1,
                    ax=ax,
                )

                dodge = True if (group and x != group[0]) else False

                sns.stripplot(
                    data=filtered_df(),
                    x=x,
                    y=y,
                    hue=group[0] if group else None,
                    color="black",
                    jitter=True,
                    alpha=0.7,
                    dodge=dodge,
                    ax=ax,
                )
            elif input.plot_type() == "violin":
                sns.violinplot(
                    data=filtered_df(), x=x, y=y, hue=group[0] if group else None, ax=ax
                )
            elif input.plot_type() == "histogram":
                sns.histplot(
                    data=filtered_df(), x=x, hue=group[0] if group else None, ax=ax
                )

            plt.title(f"{input.plot_type().capitalize()} Plot: {y} vs {x}")
            plt.tight_layout()
            return fig

        @output
        @render.plot
        def main_plot():
            return generate_plot()

        @render.download(filename=lambda: f"plot.{input.save_format()}")
        def save_plot():
            img_bytes = io.BytesIO()
            generate_plot().savefig(img_bytes, format=input.save_format())
            img_bytes.seek(0)
            return img_bytes

        @reactive.Effect
        @reactive.event(input.export_code)
        def export_source_code():
            code = f"""
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Assuming you have your data in a DataFrame called 'data'
x = "{input.xvar()}"
y = "{input.yvar()}"
group = {input.group_var() if input.group_var() else None}

fig, ax = plt.subplots(figsize=(10, 6))

if "{input.plot_type()}" == "scatter":
    {"g = sns.jointplot(data=data, x=x, y=y, hue=group[0] if group else None)" if input.show_margins() else "sns.scatterplot(data=data, x=x, y=y, hue=group[0] if group else None, ax=ax)"}
elif "{input.plot_type()}" == "box":
    sns.boxplot(data=data, x=x, y=y, hue=group[0] if group else None, fliersize=0, gap=0.1, ax=ax)
    dodge = True if (group and x != group[0]) else False
    sns.stripplot(data=data, x=x, y=y, hue=group[0] if group else None, color="black", jitter=True, alpha=0.7, dodge=dodge, ax=ax)
elif "{input.plot_type()}" == "violin":
    sns.violinplot(data=data, x=x, y=y, hue=group[0] if group else None, ax=ax)
elif "{input.plot_type()}" == "histogram":
    sns.histplot(data=data, x=x, hue=group[0] if group else None, ax=ax)

plt.title(f"{input.plot_type().capitalize()} Plot: {{y}} vs {{x}}")
plt.tight_layout()
plt.show()
"""
            if input.export_format() == "Python script":
                with open("plot_code.py", "w") as f:
                    f.write(code)
                ui.notification_show(
                    "Python script saved as 'plot_code.py'", duration=3
                )
            else:
                pyperclip.copy(code)
                ui.notification_show("Code copied to clipboard", duration=3)

    return App(app_ui, server)


def main(input_data):
    """
    Main function to run the data exploration dashboard.

    Args:
        input_data (str or pd.DataFrame): Either a path to a CSV/Feather file or a pandas DataFrame.

    Raises:
        SystemExit: If there's an error reading the file or if the input is invalid.
    """

    if isinstance(input_data, str):
        # If input is a string, assume it's a file path
        try:
            if input_data.endswith(".csv"):
                data = pd.read_csv(input_data)
            elif input_data.endswith(".feather"):
                data = pd.read_feather(input_data)
            else:
                print("Unsupported file format. Please provide a CSV or Feather file.")
                sys.exit(1)
        except Exception as e:
            print(f"Error reading file: {e}")
            sys.exit(1)
    elif isinstance(input_data, pd.DataFrame):
        # If input is already a DataFrame, use it directly
        data = input_data
    else:
        print(
            "Invalid input. Please provide either a CSV/Feather file path or a pandas DataFrame."
        )
        sys.exit(1)

    app = create_dashboard(data)
    run_app(app)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Interactive Data Exploration Dashboard",
        epilog="Example: python data_explorer.py path/to/your/data.csv",
    )
    parser.add_argument(
        "input", help="Path to the CSV file containing the dataset to explore"
    )
    parser.add_argument(
        "-p",
        "--port",
        type=int,
        default=8000,
        help="Port to run the Shiny app on (default: 8000)",
    )
    args = parser.parse_args()

    main(args.input)
