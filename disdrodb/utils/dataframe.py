#!/usr/bin/env python3

# -----------------------------------------------------------------------------.
# Copyright (c) 2021-2023 DISDRODB developers
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
# -----------------------------------------------------------------------------.
"""Dataframe utilities."""
import numpy as np
import pandas as pd


def log_arange(start, stop, log_step=0.1, base=10):
    """
    Return numbers spaced evenly on a log scale (similar to np.arange but in log space).

    Parameters
    ----------
    start : float
        The starting value of the sequence (must be > 0).
    stop : float
        The end value of the sequence (must be > 0).
    log_step : float
        The step size in log-space (default is 0.1).
    base : float
        The logarithmic base (default is 10).

    Returns
    -------
    np.ndarray
        Array of values spaced in log scale.
    """
    if start <= 0 or stop <= 0:
        raise ValueError("Both start and stop must be > 0 for log spacing.")

    log_start = np.log(start) / np.log(base)
    log_stop = np.log(stop) / np.log(base)

    log_values = np.arange(log_start, log_stop, log_step)
    return base**log_values


def compute_1d_histogram(df, column, variables=None, bins=10, labels=None, prefix_name=True, include_quantiles=False):
    """Compute conditional univariate statistics.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe
    column : str
        Column name to be binned.
    variables : str or list, optional
        Column names for which conditional statistics will be computed.
        If None, only counts are computed.
    bins : int or array-like
        Number of bins or bin edges.
    labels : array-like, optional
        Labels for the column bins. If None, uses bin centers.

    Returns
    -------
    pandas.DataFrame
    """
    # Copy data
    df = df.copy()

    # Ensure `variables` is a list of variables
    # - If no variable specified, create dummy variable
    if variables is None:
        variables = ["dummy"]
        df["dummy"] = np.ones(df[column].shape)
        variables_specified = False
    elif isinstance(variables, str):
        variables = [variables]
        variables_specified = True
    elif isinstance(variables, list):
        variables_specified = True
    else:
        raise TypeError("`variables` must be a string, list of strings, or None.")
    variables = np.unique(variables)

    # Handle column binning
    if isinstance(bins, int):
        bins = np.linspace(df[column].min(), df[column].max(), bins + 1)

    # Drop rows where any of the key columns have NaN
    df = df.dropna(subset=[column, *variables])

    if len(df) == 0:
        raise ValueError("No valid data points after removing NaN values")

    # Create binned columns with explicit handling of out-of-bounds values
    df[f"{column}_binned"] = pd.cut(df[column], bins=bins, include_lowest=True)

    # Create complete IntervalIndex for both dimensions
    intervals = df[f"{column}_binned"].cat.categories

    # Create IntervalIndex with all possible combinations
    full_index = pd.Index(intervals, name=f"{column}_binned")

    # Define grouping object
    df_grouped = df.groupby([f"{column}_binned"], observed=False)

    # Compute statistics for specified variables
    variables_stats = []
    for i, var in enumerate(variables):
        # Prepare prefix
        prefix = f"{var}_" if prefix_name and variables_specified else ""

        # Define statistics to compute
        if variables_specified:
            # Compute quantiles
            quantiles = [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
            df_stats_quantiles = df_grouped[var].quantile(quantiles).unstack(level=-1)
            df_stats_quantiles.columns = [f"{prefix}Q{int(q*100)}" for q in df_stats_quantiles.columns]
            df_stats_quantiles = df_stats_quantiles.rename(
                columns={
                    f"{prefix}Q50": f"{prefix}median",
                },
            )
            # Define other stats to compute
            list_stats = [
                (f"{prefix}std", "std"),
                (f"{prefix}min", "min"),
                (f"{prefix}max", "max"),
                (f"{prefix}mad", lambda s: np.median(np.abs(s - np.median(s)))),
            ]
            if i == 0:
                list_stats.append(("count", "count"))
        else:
            list_stats = [("count", "count")]

        # Compute statistics
        df_stats = df_grouped[var].agg(list_stats)

        # Compute other variable statistics
        if variables_specified:
            df_stats[f"{prefix}range"] = df_stats[f"{prefix}max"] - df_stats[f"{prefix}min"]
            df_stats[f"{prefix}iqr"] = df_stats_quantiles[f"{prefix}Q75"] - df_stats_quantiles[f"{prefix}Q25"]
            df_stats[f"{prefix}ipr80"] = df_stats_quantiles[f"{prefix}Q90"] - df_stats_quantiles[f"{prefix}Q10"]
            df_stats[f"{prefix}ipr90"] = df_stats_quantiles[f"{prefix}Q95"] - df_stats_quantiles[f"{prefix}Q5"]
            df_stats[f"{prefix}ipr98"] = df_stats_quantiles[f"{prefix}Q99"] - df_stats_quantiles[f"{prefix}Q1"]
            if include_quantiles:
                df_stats = pd.concat((df_stats, df_stats_quantiles), axis=1)
            else:
                df_stats[f"{prefix}median"] = df_stats_quantiles[f"{prefix}median"]
        variables_stats.append(df_stats)

    # Combine all statistics into a single DataFrame
    df_stats = pd.concat(variables_stats, axis=1)

    # Reindex to include all interval combinations
    df_stats = df_stats.reindex(full_index)

    # Determine bin centers
    centers = intervals.mid

    # Use provided labels if available
    coords = labels if labels is not None else centers

    # Reset index and add coordinates/labels
    df_stats = df_stats.reset_index()
    df_stats[f"{column}"] = pd.Categorical(df_stats[f"{column}_binned"].map(dict(zip(intervals, coords, strict=False))))
    df_stats = df_stats.drop(columns=f"{column}_binned")

    return df_stats


def compute_2d_histogram(
    df,
    x,
    y,
    variables=None,
    x_bins=10,
    y_bins=10,
    x_labels=None,
    y_labels=None,
    prefix_name=True,
    include_quantiles=False,
):
    """Compute conditional bivariate statistics.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe
    x : str
        Column name for x-axis binning (will be rounded to integers)
    y : str
        Column name for y-axis binning
    variables : str or list, optional
        Column names for which statistics will be computed.
        If None, only counts are computed.
    x_bins : int or array-like
        Number of bins or bin edges for x
    y_bins : int or array-like
        Number of bins or bin edges for y
    x_labels : array-like, optional
        Labels for x bins. If None, uses bin centers
    y_labels : array-like, optional
        Labels for y bins. If None, uses bin centers

    Returns
    -------
    xarray.Dataset
        Dataset with dimensions corresponding to binned variables and
        data variables for each statistic
    """
    # # If polars, cast to pandas
    # if isinstance(df, pl.DataFrame):
    #     df = df.to_pandas()

    # Copy data
    df = df.copy()

    # Ensure `variables` is a list of variables
    # - If no variable specified, create dummy variable
    if variables is None:
        variables = ["dummy"]
        df["dummy"] = np.ones(df[x].shape)
        variables_specified = False
    elif isinstance(variables, str):
        variables = [variables]
        variables_specified = True
    elif isinstance(variables, list):
        variables_specified = True
    else:
        raise TypeError("`variables` must be a string, list of strings, or None.")
    variables = np.unique(variables)

    # Handle x-axis binning
    if isinstance(x_bins, int):
        x_bins = np.linspace(df[x].min(), df[x].max(), x_bins + 1)
    # Handle y-axis binning
    if isinstance(y_bins, int):
        y_bins = np.linspace(df[y].min(), df[y].max(), y_bins + 1)

    # Drop rows where any of the key columns have NaN
    df = df.dropna(subset=[x, y, *variables])

    if len(df) == 0:
        raise ValueError("No valid data points after removing NaN values")

    # Create binned columns with explicit handling of out-of-bounds values
    df[f"{x}_binned"] = pd.cut(df[x], bins=x_bins, include_lowest=True)
    df[f"{y}_binned"] = pd.cut(df[y], bins=y_bins, include_lowest=True)

    # Create complete IntervalIndex for both dimensions
    x_intervals = df[f"{x}_binned"].cat.categories
    y_intervals = df[f"{y}_binned"].cat.categories

    # Create MultiIndex with all possible combinations
    full_index = pd.MultiIndex.from_product([x_intervals, y_intervals], names=[f"{x}_binned", f"{y}_binned"])

    # Define grouping object
    df_grouped = df.groupby([f"{x}_binned", f"{y}_binned"], observed=False)

    # Compute statistics for specified variables
    variables_stats = []
    for i, var in enumerate(variables):
        # Prepare prefix
        prefix = f"{var}_" if prefix_name and variables_specified else ""

        # Define statistics to compute
        if variables_specified:
            # Compute quantiles
            quantiles = [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
            df_stats_quantiles = df_grouped[var].quantile(quantiles).unstack(level=-1)
            df_stats_quantiles.columns = [f"{prefix}Q{int(q*100)}" for q in df_stats_quantiles.columns]
            df_stats_quantiles = df_stats_quantiles.rename(
                columns={
                    f"{prefix}Q50": f"{prefix}median",
                },
            )
            # Define other stats to compute
            list_stats = [
                (f"{prefix}std", "std"),
                (f"{prefix}min", "min"),
                (f"{prefix}max", "max"),
                (f"{prefix}mad", lambda s: np.median(np.abs(s - np.median(s)))),
            ]
            if i == 0:
                list_stats.append(("count", "count"))
        else:
            list_stats = [("count", "count")]

        # Compute statistics
        df_stats = df_grouped[var].agg(list_stats)

        # Compute other variable statistics
        if variables_specified:
            df_stats[f"{prefix}range"] = df_stats[f"{prefix}max"] - df_stats[f"{prefix}min"]
            df_stats[f"{prefix}iqr"] = df_stats_quantiles[f"{prefix}Q75"] - df_stats_quantiles[f"{prefix}Q25"]
            df_stats[f"{prefix}ipr80"] = df_stats_quantiles[f"{prefix}Q90"] - df_stats_quantiles[f"{prefix}Q10"]
            df_stats[f"{prefix}ipr90"] = df_stats_quantiles[f"{prefix}Q95"] - df_stats_quantiles[f"{prefix}Q5"]
            df_stats[f"{prefix}ipr98"] = df_stats_quantiles[f"{prefix}Q99"] - df_stats_quantiles[f"{prefix}Q1"]
            if include_quantiles:
                df_stats = pd.concat((df_stats, df_stats_quantiles), axis=1)
            else:
                df_stats[f"{prefix}median"] = df_stats_quantiles[f"{prefix}median"]
        variables_stats.append(df_stats)

    # Combine all statistics into a single DataFrame
    df_stats = pd.concat(variables_stats, axis=1)

    # Reindex to include all interval combinations
    df_stats = df_stats.reindex(full_index)

    # Determine coordinates
    x_centers = x_intervals.mid
    y_centers = y_intervals.mid

    # Use provided labels if available
    x_coords = x_labels if x_labels is not None else x_centers
    y_coords = y_labels if y_labels is not None else y_centers

    # Reset index and set new coordinates
    df_stats = df_stats.reset_index()
    df_stats[f"{x}"] = pd.Categorical(df_stats[f"{x}_binned"].map(dict(zip(x_intervals, x_coords, strict=False))))
    df_stats[f"{y}"] = pd.Categorical(df_stats[f"{y}_binned"].map(dict(zip(y_intervals, y_coords, strict=False))))

    # Set new MultiIndex with coordinates
    df_stats = df_stats.set_index([f"{x}", f"{y}"])
    df_stats = df_stats.drop(columns=[f"{x}_binned", f"{y}_binned"])

    # Convert to dataset
    ds = df_stats.to_xarray()

    # Transpose arrays
    ds = ds.transpose(y, x)
    return ds
