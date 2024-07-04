import numpy as np
from collections import Counter
import matplotlib.pyplot as plt


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


def plot_grouped_barh_charts(grouped_df, column_name):
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
        ax = counts.plot(
            kind="barh", title=f"{column_name.capitalize()} for {group_name}"
        )

        for index, value in enumerate(counts):
            ax.text(value, index, str(value), va="center")

        plt.show()


def chart_unique_values(df, group_column, value_column):
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
        counts = df[df[group_column].str.contains(value)][value_column].value_counts()

        plt.figure()
        ax = counts.plot(kind="barh", title=f"Importance of formal education - {value}")

        ax.set_xlabel("Count")
        ax.set_ylabel(value_column)

        for index, val in enumerate(counts):
            ax.text(val, index, str(val), va="center")

        plt.show()
