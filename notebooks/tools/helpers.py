import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import random
import pandas as pd
from matplotlib.ticker import MaxNLocator


def random_hex():
    """
    Generates a random hex color code.

    The function creates a random color by generating random values for red, green, and blue components.

    Returns:
    -------
    str
        A random hex color code in the format '#RRGGBB'.
    """
    random_color = "#{:06x}".format(random.randint(0, 0xFFFFFF))
    return random_color


def get_column_uniques(df, column_name):
    """
    Returns a list with unique values in a column.

    Parameters:
    ----------
    df : pandas.DataFrame
        The input DataFrame containing the column.
    column_name : str
        The name of the column to extract unique values from.

    Returns:
    -------
    list
        A list containing unique values found in the specified column.

    Raises:
    ------
    ValueError
        If the specified column does not exist in the DataFrame.
    """
    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' does not exist in the DataFrame")

    unique_values = df[column_name].str.split(",\s*").explode().unique()
    return unique_values.tolist()


def get_column_uniques_count(df, column_name):
    """
    Returns a pandas Series object containing the count of occurrences for each unique value in a column.

    Parameters:
    ----------
    df : pandas.DataFrame
        The input DataFrame containing the column.
    column_name : str
        The name of the column for which to count unique values.

    Returns:
    -------
    pandas.Series
        A Series object with counts of unique values.

    Raises:
    ------
    ValueError
        If the specified column does not exist in the DataFrame.
    """
    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' does not exist in the DataFrame")

    unique_values = df[column_name].str.split(",\s*").explode()
    value_counts = unique_values.value_counts()
    return value_counts


def plot_grouped_by_category_barh_charts(grouped_df, column_name, title, ylabel):
    """
    Plots multiple horizontal bar charts based on a set of 'groups'.

    Each group is plotted on a separate chart.

    Parameters:
    ----------
    grouped_df : pandas GroupBy object
        The grouped DataFrame by some categorical variable.
    column_name : str
        The name of the column containing categorical data to be plotted.

    Returns:
    -------
    None
    """
    for group_name in grouped_df.groups.keys():
        group_df = grouped_df.get_group(group_name)
        counts = get_column_uniques_count(group_df, column_name)

        plt.figure()
        ax = counts.plot(kind="barh")

        ax.set_xlabel("Count")
        ax.set_ylabel(ylabel)
        plt.title(f"{title} \n {group_name}", fontsize=14, fontweight="bold")
        for index, value in enumerate(counts):
            ax.text(value, index, str(value), va="center")

        plt.show()


def barh_chart_unique_values(df, group_column, value_column, title, ylabel):
    """
    Plots horizontal bar charts for each unique value in group_column.

    Counts occurrences of value_column within each group.

    Parameters:
    ----------
    df : pandas.DataFrame
        The input DataFrame.
    group_column : str
        The name of the column containing categorical values for grouping.
    value_column : str
        The name of the column for which to count occurrences within each group.

    Returns:
    -------
    None
    """
    # Ensure there are no NaN values in the group_column
    df = df.dropna(subset=[group_column])

    unique_values = get_column_uniques(df, group_column)

    for value in unique_values:
        counts = df[df[group_column].str.contains(value, na=False)][
            value_column
        ].value_counts()

        plt.figure()
        ax = counts.plot(kind="barh")
        plt.title(f"{title} \n {value}", fontsize=14, fontweight="bold")
        ax.set_xlabel("Count")
        ax.set_ylabel(ylabel)

        for index, val in enumerate(counts):
            ax.text(val, index, str(val), va="center")

        plt.show()


def print_unique_normalized_values_by_group(df, group_column, value_column, title):
    """
    Plots horizontal bar charts for each unique value in group_column.

    Counts occurrences of value_column within each group.

    Parameters:
    ----------
    df : pandas.DataFrame
        The input DataFrame.
    group_column : str
        The name of the column containing categorical values for grouping.
    value_column : str
        The name of the column for which to count occurrences within each group.

    Returns:
    -------
    None
    """

    unique_values = get_column_uniques(df, group_column)

    for value in unique_values:
        counts = df[df[group_column].str.contains(value)][value_column].value_counts(
            normalize=True
        )
        percentages = (counts * 100).astype(int).astype(str) + "%"

        print(f"{title} - {value}")
        print("=" * 20)
        print(percentages)
        print("\n" * 3)


def make_vertical_grouped_chart(df, g1, g2, col, labels, config):
    """
    Creates a vertical grouped bar chart comparing values between two groups.

    Args:
        df (pandas.DataFrame): The DataFrame containing the data.
        g1 (pandas.DataFrame): A subset of the DataFrame representing the first group.
        g2 (pandas.DataFrame): A subset of the DataFrame representing the second group.
        col (str): The name of the column to compare values for.
        labels (list): A list of unique values from the column to use as labels.
        config (dict): A configuration dictionary with keys:
            - title (str): The title of the chart.
            - g1_label (str): The label for the first group's bars.
            - g2_label (str): The label for the second group's bars.
            - xlabel (str): The label for the x-axis.
            - ylabel (str): The label for the y-axis.

    Returns:
        None
    """

    def get_uniques_col_count(group_df, column):
        return group_df[column].value_counts().to_dict()

    def get_column_uniques(dataframe, column):
        return dataframe[column].unique()

    g1_count = get_uniques_col_count(g1, col)
    g2_count = get_uniques_col_count(g2, col)

    # Values
    g1_val = [g1_count.get(i, 0) for i in labels]
    g2_val = [g2_count.get(i, 0) for i in labels]

    x = np.arange(len(labels))  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots(figsize=(10, 6))  # Adjust size as needed
    rects1 = ax.bar(x - width / 2, g1_val, width, label=config.get("g1_label", ""))
    rects2 = ax.bar(x + width / 2, g2_val, width, label=config.get("g2_label", ""))

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_xlabel(config.get("xlabel", ""))
    ax.set_title(config.get("title", ""))
    ax.set_ylabel(config.get("ylabel", ""))
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    # Ensure y-axis has integer values
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    def annotate_bars(rects, values):
        for rect, value in zip(rects, values):
            height = rect.get_height()
            if height != 0:
                ax.annotate(
                    f"{value}",
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                )

    annotate_bars(rects1, g1_val)
    annotate_bars(rects2, g2_val)

    fig.tight_layout()
    plt.show()


def make_horizontal_grouped_chart(df, g1, g2, col, labels, config):
    """Creates a horizontal grouped bar chart comparing values between two groups.

    Args:
        df (pandas.DataFrame): The DataFrame containing the data.
        g1 (pandas.DataFrame): A subset of the DataFrame representing the first group.
        g2 (pandas.DataFrame): A subset of the DataFrame representing the second group.
        col (str): The name of the column to compare values for.
        labels (list): A list of unique values from the column to use as labels.
        config (dict): A configuration dictionary with keys:
            - title (str): The title of the chart.
            - c1_label (str): The label for the first group's bars.
            - c2_label (str): The label for the second group's bars.
            - xlabel (str): The label for the x-axis.
            - ylabel (str): The label for the y-axis.

    Raises:
        ValueError: If the specified column does not exist in the DataFrame.

    Example:
        >>> df = pd.DataFrame({'exp_en_IT': ['A', 'B', 'A', 'C', 'B'], 'gender': ['MAN', 'WOMAN', 'MAN', 'MAN', 'WOMAN']})
        >>> gen = df.groupby('gender')
        >>> group_config = {
        ...     'title': "exp_en_IT by Gender",
        ...     'c1_label': "MAN",
        ...     'c2_label': "WOMAN",
        ...     'xlabel': "Count",
        ...     'ylabel': "exp_en_IT level"
        ... }
        >>> make_horizontal_grouped_chart(df, gen.get_group("MAN"), gen.get_group("WOMAN"), "exp_en_IT", df["exp_en_IT"].unique(), group_config)
    """

    def get_uniques_col_count(group, column):
        return group[column].value_counts().to_dict()

    def get_column_uniques(dataframe, column):
        return dataframe[column].unique()

    g1_count = get_uniques_col_count(g1, col)
    g2_count = get_uniques_col_count(g2, col)
    labels = get_column_uniques(df, col)

    g1_val = [int(g1_count.get(i, 0)) for i in labels]
    g2_val = [int(g2_count.get(i, 0)) for i in labels]

    x = np.arange(len(labels))  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.barh(x - width / 2, g1_val, width, label=config.get("c1_label", ""))
    rects2 = ax.barh(x + width / 2, g2_val, width, label=config.get("c2_label", ""))

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_xlabel(config.get("xlabel", ""))
    ax.set_title(config.get("title", ""))
    ax.set_ylabel(config.get("ylabel", ""))
    ax.set_yticks(x)
    ax.set_yticklabels(labels)
    ax.legend()

    def annotate_bars(rects, values):
        for rect, value in zip(rects, values):
            width = rect.get_width()
            ax.annotate(
                f"{value}",
                xy=(width, rect.get_y() + rect.get_height() / 2),
                xytext=(3, 0),  # 3 points horizontal offset
                textcoords="offset points",
                ha="left",
                va="center",
            )

    annotate_bars(rects1, g1_val)
    annotate_bars(rects2, g2_val)

    fig.tight_layout()
    plt.show()


def grouped_grid_barh_chart(df_grouped, column, title, nrows, ncols, color="blue"):
    """
    Creates a grid of horizontal bar charts for each group in a grouped DataFrame.

    Parameters:
    df_grouped (pd.core.groupby.generic.DataFrameGroupBy): Grouped DataFrame
    column (str): Column name to be plotted in each chart
    title (str): Title of each chart plus the group name
    nrows (int): Number of rows in the grid
    ncols (int): Number of columns in the grid
    color (str): Color of the bars in the charts
    """
    plt.figure(figsize=(ncols * 5, nrows * 3), dpi=80)

    for k, (group_name, group_data) in enumerate(df_grouped):
        plt.subplot(nrows, ncols, k + 1)
        group_data[column].value_counts().plot(
            kind="barh",
            title=f"{title} \n {group_name} | {group_data.shape[0]}",
            color=color,
        )

    plt.subplots_adjust(
        left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.4, hspace=0.9
    )
    plt.show()


def create_grouped_dataframe_list(df_list, groupby_column):
    """
    Creates a grouped DataFrame from a list of DataFrames.

    Parameters:
    df_list (list of pd.DataFrame): List of DataFrames to be grouped.
    groupby_column (str): Column name to group by.

    Returns:
    pd.core.groupby.generic.DataFrameGroupBy: Grouped DataFrame.
    """
    # Concatenate all DataFrames into one
    concatenated_df = pd.concat(df_list)

    # Group by the specified column
    grouped_df = concatenated_df.groupby(groupby_column)

    return grouped_df


def grouped_grid_pie_chart(df_grouped, column, title, nrows, ncols, autopct="%1.1f%%"):
    """
    Creates a grid of pie charts for each group in a grouped DataFrame using default color palette.

    Parameters:
    df_grouped (pd.core.groupby.generic.DataFrameGroupBy): Grouped DataFrame
    column (str): Column name to be plotted in each pie chart
    title (str): Base title for each chart plus the group name
    nrows (int): Number of rows in the grid
    ncols (int): Number of columns in the grid
    autopct (str or callable, optional): Format string or callable function for autopct parameter
    """
    fig, axs = plt.subplots(
        nrows=nrows, ncols=ncols, figsize=(ncols * 4, nrows * 3), dpi=80
    )

    for ax, (group_name, group_data) in zip(axs.flat, df_grouped):
        values = group_data[column].value_counts()

        # Use default color palette
        ax.pie(values, labels=values.index, autopct=autopct)
        ax.set_title(
            f"{title} \n {group_name} | Total: {group_data.shape[0]}", fontweight="bold"
        )

    # Adjust layout to add space between subplots
    plt.subplots_adjust(
        left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.4, hspace=0.5
    )
    plt.show()


def chart_pie_list(grouped_df, column, title, explode=None, autopct="%1.1f%%"):
    """
    Create pie charts for each group in a grouped DataFrame.

    Parameters:
    grouped_df (pd.core.groupby.generic.DataFrameGroupBy): Grouped DataFrame
    column (str): Column name to plot in each pie chart
    explode (list, optional): List of explode values for pie charts
    autopct (str or callable, optional): Format string or callable function for autopct parameter
    """
    for k, (group_name, group_data) in enumerate(grouped_df):
        aux_df = group_data[column].value_counts(normalize=True)
        plt.figure(figsize=(6, 6))  # Adjust figure size as needed
        plt.pie(aux_df, labels=aux_df.index, autopct=autopct, explode=explode)
        plt.title(f"{title} \n {group_name} \n Total: {group_data.shape[0]}")
        plt.ylabel("")
        plt.show()


def barh_chart_count(df, column_name, title, xlabel):
    """
    Plots a horizontal bar chart for the value counts of a specified column.

    Parameters:
    ----------
    df : pandas.DataFrame
        The input DataFrame.
    column_name : str
        The name of the column to plot value counts for.
    title : str
        The title of the chart.
    xlabel : str
        The label for the x-axis.

    Returns:
    -------
    None
    """
    counts = df[column_name].value_counts()

    plt.figure()
    ax = counts.plot(kind="barh")
    plt.title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(column_name)
    ax.xaxis.set_major_locator(
        MaxNLocator(integer=True)
    )  # Ensure numeric values are integers

    for index, val in enumerate(counts):
        ax.text(val, index, str(val), va="center")

    plt.show()


def barh_chart_normal_count(
    df, column_name, title="Work mode total count", xlabel="count", ylabel="Work Mode"
):
    fig = plt.figure(figsize=(9, 6))

    wm_plot = (
        df[column_name]
        .value_counts()
        .plot(
            kind="barh",
            title=title,
            xlabel=xlabel,
            ylabel=ylabel,
        )
    )
    # wm_plot.set_xlabel("Total")
    # wm_plot.set_ylabel("Job title.")

    for k, v in enumerate(get_column_uniques_count(df, column_name)):
        wm_plot.annotate(v, (v, k), va="center")

    plt.show()


def uniques_count_to_dataframe(series, top_n=10):
    """
    Converts a Series object containing unique counts into a DataFrame with specified column names,
    excluding the first value and including 'role' as the first column.

    Parameters:
    - series (pd.Series): The Series object to convert.
    - top_n (int): The number of top entries to include in the DataFrame.

    Returns:
    - pd.DataFrame: The resulting DataFrame with columns 'role' and 'count'.
    """
    # Convert the Series to a DataFrame, exclude the first value, and reset the index
    df = series.iloc[1 : top_n + 1].reset_index()

    # Rename the columns
    df.columns = ["role", "count"]

    return df
