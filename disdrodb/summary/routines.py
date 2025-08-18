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
"""Utilities to create summary statistics."""
import os
import subprocess
import tempfile
from shutil import which

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.colors import ListedColormap, LogNorm, Normalize
from matplotlib.gridspec import GridSpec
from scipy.optimize import curve_fit

import disdrodb
from disdrodb.api.path import define_station_dir
from disdrodb.constants import DIAMETER_DIMENSION, VELOCITY_DIMENSION
from disdrodb.l2.empirical_dsd import get_drop_average_velocity
from disdrodb.l2.event import group_timesteps_into_event
from disdrodb.scattering import RADAR_OPTIONS
from disdrodb.utils.dataframe import compute_2d_histogram, log_arange
from disdrodb.utils.manipulations import (
    get_diameter_bin_edges,
    resample_drop_number_concentration,
    unstack_radar_variables,
)
from disdrodb.utils.warnings import suppress_warnings
from disdrodb.utils.yaml import write_yaml
from disdrodb.viz import compute_dense_lines, max_blend_images, to_rgba

####-----------------------------------------------------------------
#### PDF Latex Utilities


def is_latex_engine_available() -> bool:
    """
    Determine whether the Tectonic TeX/LaTeX engine is installed and accessible.

    Returns
    -------
    bool
        True if tectonic is found, False otherwise.
    """
    return which("tectonic") is not None


def save_table_to_pdf(
    df: pd.DataFrame,
    filepath: str,
    index=True,
    caption=None,
    fontsize: str = r"\tiny",
    orientation: str = "landscape",
) -> None:
    r"""Render a pandas DataFrame as a well-formatted table in PDF via LaTeX.

    Parameters
    ----------
    df : pd.DataFrame
        The data to render.
    filepath : str
        File path where write the final PDF (e.g. '<...>/table.pdf').
    caption : str, optional
        LaTeX caption for the table environment.
    fontsize : str, optional
        LaTeX font-size command to wrap the table (e.g. '\\small').
        The default is '\\tiny'.
    orientation : {'portrait', 'landscape'}
        Page orientation. If 'landscape', the table will be laid out horizontally.
        The default is 'landscape'.
    """
    # Export table to LaTeX
    table_tex = df.to_latex(
        index=index,
        longtable=True,
        caption=caption,
        label=None,
        escape=False,
    )

    # Define LaTeX document
    opts = "a4paper"
    doc = [
        f"\\documentclass[{opts}]{{article}}",
        "\\usepackage[margin=0.1in]{geometry}",
        # Reduce column separation
        "\\setlength{\\tabcolsep}{3pt}",
        "\\usepackage{booktabs}",
        "\\usepackage{longtable}",
        "\\usepackage{caption}",
        "\\captionsetup[longtable]{font=tiny}",
        "\\usepackage{pdflscape}",
        "\\begin{document}",
        # Remove page numbers
        "\\pagestyle{empty}",
    ]

    if orientation.lower() == "landscape":
        doc.append("\\begin{landscape}")

    doc.append(f"{{{fontsize}\n{table_tex}\n}}")

    if orientation.lower() == "landscape":
        doc.append("\\end{landscape}")

    doc.append("\\end{document}")
    document = "\n".join(doc)

    # Compile with pdflatex in a temp dir
    with tempfile.TemporaryDirectory() as td:
        tex_path = os.path.join(td, "table.tex")
        with open(tex_path, "w", encoding="utf-8") as f:
            f.write(document)
        for _ in range(2):
            subprocess.run(
                [
                    "tectonic",
                    "--outdir",
                    td,
                    tex_path,
                ],
                cwd=td,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=True,
            )
        # Move result
        os.replace(os.path.join(td, "table.pdf"), filepath)


####-----------------------------------------------------------------
#### Tables summaries


def create_table_rain_summary(df):
    """Create rainy table summary."""
    # Initialize dictionary
    table = {}

    # Keep rows with R > 0
    df = df[df["R"] > 0]

    # Number of years, months, days, minutes
    if df.index.name == "time":
        df = df.reset_index()
    time = np.sort(np.asanyarray(df["time"]))

    # Define start_time and end_time
    start_time = pd.Timestamp(time[0])
    end_time = pd.Timestamp(time[-1])

    # Define years and years-month coverage
    start_year = start_time.year
    start_month = start_time.month_name()
    end_year = end_time.year
    end_month = end_time.month_name()
    if start_year == end_year:
        years_coverage = f"{start_year}"
        years_month_coverage = f"{start_month[0:3]}-{end_month[0:3]} {start_year}"
    else:
        years_coverage = f"{start_year} - {end_year}"
        years_month_coverage = f"{start_month[0:3]} {start_year} - {end_month[0:3]} {end_year}"
    table["years_coverage"] = years_coverage
    table["years_month_coverage"] = years_month_coverage

    # Rainy minutes statistics
    table["n_rainy_minutes"] = len(df["R"])
    table["n_rainy_minutes_<0.1"] = df["R"].between(0, 0.1, inclusive="right").sum().item()
    table["n_rainy_minutes_0.1_1"] = df["R"].between(0.1, 1, inclusive="right").sum().item()
    table["n_rainy_minutes_1_10"] = df["R"].between(1, 10, inclusive="right").sum().item()
    table["n_rainy_minutes_10_25"] = df["R"].between(10, 25, inclusive="right").sum().item()
    table["n_rainy_minutes_25_50"] = df["R"].between(25, 50, inclusive="right").sum().item()
    table["n_rainy_minutes_50_100"] = df["R"].between(50, 100, inclusive="right").sum().item()
    table["n_rainy_minutes_100_200"] = df["R"].between(100, 200, inclusive="right").sum().item()
    table["n_rainy_minutes_>200"] = np.sum(df["R"] > 200).item()

    # Minutes with larger Dmax
    table["n_minutes_Dmax_>7"] = np.sum(df["Dmax"] > 7).item()
    table["n_minutes_Dmax_>8"] = np.sum(df["Dmax"] > 8).item()
    table["n_minutes_Dmax_>9"] = np.sum(df["Dmax"] > 9).item()
    return table


def create_table_dsd_summary(df):
    """Create table with integral DSD parameters statistics."""
    # Define additional variables
    df["log10(Nw)"] = np.log10(df["Nw"])
    df["log10(Nt)"] = np.log10(df["Nt"])

    # List of variables to summarize
    variables = ["W", "R", "Z", "D50", "Dm", "sigma_m", "Dmax", "Nw", "log10(Nw)", "Nt", "log10(Nt)"]

    # Define subset dataframe
    df_subset = df[variables]

    # Prepare summary DataFrame
    stats_cols = [
        "MIN",
        "Q1",
        "Q5",
        "Q10",
        "Q50",
        "Q90",
        "Q95",
        "Q99",
        "MAX",
        "MEAN",
        "STD",
        "MAD",
        "SKEWNESS",
        "KURTOSIS",
    ]
    df_stats = pd.DataFrame(index=variables, columns=stats_cols)

    # Compute DSD integral parameters statistics
    df_stats["MIN"] = df_subset.min()
    df_stats["Q1"] = df_subset.quantile(0.01)
    df_stats["Q5"] = df_subset.quantile(0.05)
    df_stats["Q10"] = df_subset.quantile(0.10)
    df_stats["Q50"] = df_subset.median()
    df_stats["Q90"] = df_subset.quantile(0.90)
    df_stats["Q95"] = df_subset.quantile(0.95)
    df_stats["Q99"] = df_subset.quantile(0.99)
    df_stats["MAX"] = df_subset.max()
    df_stats["MEAN"] = df_subset.mean()
    df_stats["STD"] = df_subset.std()
    df_stats["MAD"] = df_subset.apply(lambda x: np.mean(np.abs(x - x.mean())))
    df_stats["SKEWNESS"] = df_subset.skew()
    df_stats["KURTOSIS"] = df_subset.kurt()

    # Round statistics
    df_stats = df_stats.astype(float).round(2)
    return df_stats


def create_table_events_summary(df):
    """Creata table with events statistics."""
    # Event file
    # - Events are separated by 1 hour or more rain-free periods in rain rate time series.
    # - The events that are less than 'min_duration' minutes or the rain total is less than 0.1 mm
    #   are not reported.
    event_settings = {
        "neighbor_min_size": 2,
        "neighbor_time_interval": "5MIN",
        "event_max_time_gap": "1H",
        "event_min_duration": "5MIN",
        "event_min_size": 3,
    }
    # Keep rows with R > 0
    df = df[df["R"] > 0]

    # Number of years, months, days, minutes
    if df.index.name == "time":
        df = df.reset_index()
    timesteps = np.sort(np.asanyarray(df["time"]))

    # Define event list
    event_list = group_timesteps_into_event(
        timesteps=timesteps,
        neighbor_min_size=event_settings["neighbor_min_size"],
        neighbor_time_interval=event_settings["neighbor_time_interval"],
        event_max_time_gap=event_settings["event_max_time_gap"],
        event_min_duration=event_settings["event_min_duration"],
        event_min_size=event_settings["event_min_size"],
    )

    # Create dataframe with statistics for each event
    events_stats = []
    rain_thresholds = [0.1, 1, 5, 10, 20, 50, 100]
    for event in event_list:
        # Retrieve event start_time and end_time
        start, end = event["start_time"], event["end_time"]
        # Retrieve event dataframe
        df_event = df[(df["time"] >= start) & (df["time"] <= end)]
        # Initialize event record
        event_stats = {
            # Event time info
            "start_time": start,
            "end_time": end,
            "duration": int((end - start) / np.timedelta64(1, "m")),
            # Rainy minutes above thresholds
            **{f"rainy_minutes_>{thr}": int((df_event["R"] > thr).sum()) for thr in rain_thresholds},
            # Total precipitation (mm)
            "P_total": df_event["P"].sum(),
            # R statistics
            "mean_R": df_event["R"].mean(),
            "median_R": df_event["R"].median(),
            "max_R": df_event["R"].max(),
            # DSD statistics
            "max_Dmax": df_event["Dmax"].max(),
            "mean_Dm": df_event["Dm"].mean(),
            "median_Dm": df_event["Dm"].median(),
            "max_Dm": df_event["Dm"].max(),
            "mean_sigma_m": df_event["sigma_m"].mean(),
            "median_sigma_m": df_event["sigma_m"].median(),
            "max_sigma_m": df_event["sigma_m"].max(),
            "mean_W": df_event["W"].mean(),
            "median_W": df_event["W"].median(),
            "max_W": df_event["W"].max(),
            "max_Z": df_event["Z"].max(),
            "mean_Nbins": int(df_event["Nbins"].mean()),
            "max_Nbins": int(df_event["Nbins"].max()),
            # TODO in future:
            # - rain_detected = True
            # - snow_detected = True
            # - hail_detected = True
        }
        events_stats.append(event_stats)

    df_events = pd.DataFrame.from_records(events_stats)
    return df_events


def prepare_latex_table_dsd_summary(df):
    """Prepare a DataFrame with DSD statistics for LaTeX table output."""
    df = df.copy()
    # Round float columns to nearest integer, leave ints unchanged
    float_cols = df.select_dtypes(include=["float"]).columns
    df[float_cols] = df[float_cols].astype(float).round(decimals=2).astype(str)
    # Rename
    rename_dict = {
        "W": r"$W\,[\mathrm{g}\,\mathrm{m}^{-3}]$",  # [g/m3]
        "R": r"$R\,[\mathrm{mm}\,\mathrm{h}^{-1}]$",  # [mm/hr]
        "Z": r"$Z\,[\mathrm{dBZ}]$",  # [dBZ]
        "D50": r"$D_{50}\,[\mathrm{mm}]$",  # [mm]
        "Dm": r"$D_{m}\,[\mathrm{mm}]$",  # [mm]
        "sigma_m": r"$\sigma_{m}\,[\mathrm{mm}]$",  # [mm]
        "Dmax": r"$D_{\max}\,[\mathrm{mm}]$",  # [mm]
        "Nw": r"$N_{w}\,[\mathrm{mm}^{-1}\,\mathrm{m}^{-3}]$",  # [mm$^{-1}$ m$^{-3}$]
        "log10(Nw)": r"$\log_{10}(N_{w})$",  # [$\log_{10}$(mm$^{-1}$ m$^{-3}$)]
        "Nt": r"$N_{t}\,[\mathrm{m}^{-3}]$",  # [m$^{-3}$]
        "log10(Nt)": r"$\log_{10}(N_{t})$",  # [log10(m$^{-3}$)]
    }
    df = df.rename(index=rename_dict)
    return df


def prepare_latex_table_events_summary(df):
    """Prepare a DataFrame with events statistics for LaTeX table output."""
    df = df.copy()
    # Round datetime to minutes
    df["start_time"] = df["start_time"].dt.strftime("%Y-%m-%d %H:%M")
    df["end_time"] = df["end_time"].dt.strftime("%Y-%m-%d %H:%M")
    # Round float columns to nearest integer, leave ints unchanged
    float_cols = df.select_dtypes(include=["float"]).columns
    df[float_cols] = df[float_cols].astype(float).round(decimals=2).astype(str)
    # Rename
    rename_dict = {
        "start_time": r"Start",
        "end_time": r"End",
        "duration": r"Min.",
        "rainy_minutes_>0.1": r"R>0.1",
        "rainy_minutes_>1": r"R>1",
        "rainy_minutes_>5": r"R>5",
        # 'rainy_minutes_>10':    r'R>10',
        # 'rainy_minutes_>20':    r'R>20',
        "rainy_minutes_>50": r"R>50",
        # 'rainy_minutes_>100':   r'R>100',
        "P_total": r"$P_{\mathrm{tot}} [mm]$",
        "mean_R": r"$R_{\mathrm{mean}}$",
        "median_R": r"$R_{\mathrm{median}}$",
        "max_R": r"$R_{\max}$",
        "max_Dmax": r"$D_{\max}$",
        "mean_Dm": r"$D_{m,\mathrm{mean}}$",
        "median_Dm": r"$D_{m,\mathrm{median}}$",
        "max_Dm": r"$D_{m,\max}$",
        "mean_sigma_m": r"$\sigma_{m,\mathrm{mean}}$",
        "median_sigma_m": r"$\sigma_{m,\mathrm{median}}$",
        "max_sigma_m": r"$\sigma_{m,\max}$",
        "mean_W": r"$W_{\mathrm{mean}}$",
        "median_W": r"$W_{\mathrm{median}}$",
        "max_W": r"$W_{\max}$",
        "max_Z": r"$Z_{\max}$",
        "mean_Nbins": r"$N_{\mathrm{bins},\mathrm{mean}}$",
        "max_Nbins": r"$N_{\mathrm{bins},\max}$",
    }
    df = df[list(rename_dict)]
    df = df.rename(columns=rename_dict)
    return df


####-------------------------------------------------------------------
#### Powerlaw routines


def fit_powerlaw(x, y, xbins, quantile=0.5, min_counts=10, x_in_db=False):
    """
    Fit a power-law relationship ``y = a * x**b`` to binned median values.

    This function bins ``x`` into intervals defined by ``xbins``, computes the
    median of ``y`` in each bin (robust to outliers), and fits a power-law model
    using the Levenberg-Marquardt algorithm. Optionally, ``x`` can be converted
    from decibel units to linear scale automatically before fitting.

    Parameters
    ----------
    x : array_like
        Independent variable values. Must be positive and finite after filtering.
    y : array_like
        Dependent variable values. Must be positive and finite after filtering.
    xbins : array_like
        Bin edges for grouping ``x`` values (passed to ``pandas.cut``).
    quantile : float, optional
      Quantile of ``y`` to compute in each bin (between 0 and 1).
      For example: 0.5 = median, 0.25 = lower quartile, 0.75 = upper quartile.
      Default is 0.5 (median)
    x_in_db : bool, optional
        If True, converts ``x`` values from decibels (dB) to linear scale using
        :func:`disdrodb.idecibel`. Default is False.

    Returns
    -------
    params : tuple of float
        Estimated parameters ``(a, b)`` of the power-law relationship.
    params_std : tuple of float
        One standard deviation uncertainties ``(a_std, b_std)`` estimated from
        the covariance matrix of the fit.

    Notes
    -----
    - This implementation uses median statistics within bins, which reduces
      the influence of outliers.
    - Both ``x`` and ``y`` are filtered to retain only positive, finite values
      before binning.
    - Fitting is performed on the bin centers (midpoints between bin edges).

    See Also
    --------
    predict_from_powerlaw : Predict values from the fitted power-law parameters.
    inverse_powerlaw_parameters : Compute parameters of the inverse power law.

    Examples
    --------
    >>> import numpy as np
    >>> x = np.linspace(1, 50, 200)
    >>> y = 2 * x**1.5 + np.random.normal(0, 5, size=x.size)
    >>> xbins = np.arange(0, 60, 5)
    >>> (a, b), (a_std, b_std) = fit_powerlaw(x, y, xbins)
    """
    # Set min_counts to 0 during pytest execution in order to test the summary routine
    if os.environ.get("PYTEST_CURRENT_TEST"):
        min_counts = 0

    # Ensure numpy array
    x = np.asanyarray(x)
    y = np.asanyarray(y)

    # Ensure values > 0 and finite
    valid_values = (x > 0) & (y > 0) & np.isfinite(x) & np.isfinite(y)
    x = x[valid_values]
    y = y[valid_values]

    # Define dataframe
    df_data = pd.DataFrame({"x": x, "y": y})

    # Alternative code
    # from disdrodb.utils.dataframe import compute_1d_histogram
    # df_agg = compute_1d_histogram(
    #     df=df_data,
    #     column="x",
    #     variables="y",
    #     bins=xbins,
    #     prefix_name=True,
    #     include_quantiles=False
    # )
    # df_agg["count"] # instead of N

    # - Keep only data within bin range
    df_data = df_data[(df_data["x"] >= xbins[0]) & (df_data["x"] < xbins[-1])]
    # - Define bins
    df_data["x_bins"] = pd.cut(df_data["x"], bins=xbins, right=False)
    # - Remove data outside specified bins
    df_data = df_data[df_data["x_bins"].cat.codes != -1]

    # Derive median y values at various x points
    # - This typically remove outliers
    df_agg = df_data.groupby(by="x_bins", observed=True)["y"].agg(
        y=lambda s: s.quantile(quantile),
        n="count",
        mad=lambda s: (s - s.median()).abs().median(),
    )
    df_agg["x"] = np.array([iv.left + (iv.right - iv.left) / 2 for iv in df_agg.index])

    # If input is in decibel, convert to linear scale
    if x_in_db:
        df_agg["x"] = disdrodb.idecibel(df_agg["x"])

    # Remove bins with less than n counts
    df_agg = df_agg[df_agg["n"] > min_counts]
    if len(df_agg) < 5:
        raise ValueError("Not enough data to fit a power law.")

    # Estimate sigma based on MAD
    sigma = df_agg["mad"]

    # Fit the data
    with suppress_warnings():
        (a, b), pcov = curve_fit(
            lambda x, a, b: a * np.power(x, b),
            df_agg["x"],
            df_agg["y"],
            method="lm",
            sigma=sigma,
            absolute_sigma=True,
            maxfev=10_000,  # max n iterations
        )
        (a_std, b_std) = np.sqrt(np.diag(pcov))

    # Return the parameters and their standard deviation
    return (float(a), float(b)), (float(a_std), float(b_std))


def predict_from_powerlaw(x, a, b):
    """
    Predict values from a power-law relationship ``y = a * x**b``.

    Parameters
    ----------
    x : array_like
        Independent variable values.
    a : float
        Power-law coefficient.
    b : float
        Power-law exponent.

    Returns
    -------
    y : ndarray
        Predicted dependent variable values.

    Notes
    -----
    This function does not check for invalid (negative or zero) ``x`` values.
    Ensure that ``x`` is compatible with the model before calling.
    """
    return a * np.power(x, b)


def inverse_powerlaw_parameters(a, b):
    """
    Compute parameters of the inverse power-law relationship.

    Given a model ``y = a * x**b``, this returns parameters ``(A, B)``
    such that the inverse relation ``x = A * y**B`` holds.

    Parameters
    ----------
    a : float
        Power-law coefficient in ``y = a * x**b``.
    b : float
        Power-law exponent in ``y = a * x**b``.

    Returns
    -------
    A : float
        Coefficient of the inverse power-law model.
    B : float
        Exponent of the inverse power-law model.
    """
    A = 1 / (a ** (1 / b))
    B = 1 / b
    return A, B


def predict_from_inverse_powerlaw(x, a, b):
    """
    Predict values from the inverse power-law relationship.

    Given parameters ``a`` and ``b`` from ``x = a * y**b``, this function computes
    ``y`` given ``x``.

    Parameters
    ----------
    x : array_like
        Values of ``x`` (independent variable in the original power law).
    a : float
        Power-law coefficient of the inverse power-law model.
    b : float
        Power-law exponent of the inverse power-law model.

    Returns
    -------
    y : ndarray
        Predicted dependent variable values.
    """
    return (x ** (1 / b)) / (a ** (1 / b))


####-------------------------------------------------------------------
#### Drop spectrum plots


def plot_drop_spectrum(drop_number, norm=None, add_colorbar=True, title="Drop Spectrum"):
    """Plot the drop spectrum."""
    cmap = plt.get_cmap("Spectral_r").copy()
    cmap.set_under("none")
    if norm is None:
        norm = LogNorm(vmin=1, vmax=None)

    p = drop_number.plot.pcolormesh(
        x=DIAMETER_DIMENSION,
        y=VELOCITY_DIMENSION,
        cmap=cmap,
        extend="max",
        norm=norm,
        add_colorbar=add_colorbar,
        cbar_kwargs={"label": "Number of particles"},
    )
    p.axes.set_yticks([])
    p.axes.set_yticklabels([])
    p.axes.set_xlabel("Diamenter [mm]")
    p.axes.set_ylabel("Fall velocity [m/s]")
    p.axes.set_title(title)
    return p


def plot_raw_and_filtered_spectrums(
    raw_drop_number,
    drop_number,
    theoretical_average_velocity,
    measured_average_velocity=None,
    norm=None,
    figsize=(8, 4),
    dpi=300,
):
    """Plot raw and filtered drop spectrum."""
    # Drop number matrix
    cmap = plt.get_cmap("Spectral_r").copy()
    cmap.set_under("none")

    if norm is None:
        norm = LogNorm(1, None)

    fig = plt.figure(figsize=figsize, dpi=dpi)
    gs = GridSpec(1, 2, width_ratios=[1, 1.15], wspace=0.05)  # More space for ax2
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])

    raw_drop_number.plot.pcolormesh(
        x=DIAMETER_DIMENSION,
        y=VELOCITY_DIMENSION,
        ax=ax1,
        cmap=cmap,
        norm=norm,
        extend="max",
        add_colorbar=False,
    )
    theoretical_average_velocity.plot(ax=ax1, c="k", linestyle="dashed")
    if measured_average_velocity is not None:
        measured_average_velocity.plot(ax=ax1, c="k", linestyle="dotted")
    ax1.set_xlabel("Diamenter [mm]")
    ax1.set_ylabel("Fall velocity [m/s]")
    ax1.set_title("Raw Spectrum")
    drop_number.plot.pcolormesh(
        x=DIAMETER_DIMENSION,
        y=VELOCITY_DIMENSION,
        cmap=cmap,
        extend="max",
        ax=ax2,
        norm=norm,
        cbar_kwargs={"label": "Number of particles"},
    )
    theoretical_average_velocity.plot(ax=ax2, c="k", linestyle="dashed", label="Theoretical velocity")
    if measured_average_velocity is not None:
        measured_average_velocity.plot(ax=ax2, c="k", linestyle="dotted", label="Measured average velocity")
    ax2.set_yticks([])
    ax2.set_yticklabels([])
    ax2.set_xlabel("Diamenter [mm]")
    ax2.set_ylabel("")
    ax2.set_title("Filtered Spectrum")
    ax2.legend(loc="lower right", frameon=False)
    return fig


####-------------------------------------------------------------------
#### N(D) Climatological plots


def create_nd_dataframe(ds, variables=None):
    """Create pandas Dataframe with N(D) data."""
    # Define variables to select
    if isinstance(variables, str):
        variables = [variables]
    variables = [] if variables is None else variables
    variables = ["drop_number_concentration", "Nw", "diameter_bin_center", "Dm", "D50", "R", *variables]
    variables = np.unique(variables).tolist()

    # Retrieve stacked N(D) dataframe
    ds_stack = ds[variables].stack(  # noqa: PD013
        dim={"obs": ["time", "diameter_bin_center"]},
    )
    # Drop coordinates
    coords_to_drop = [
        "velocity_method",
        "sample_interval",
        *RADAR_OPTIONS,
    ]
    df_nd = ds_stack.to_dataframe().drop(columns=coords_to_drop, errors="ignore")
    df_nd["D"] = df_nd["diameter_bin_center"]
    df_nd["N(D)"] = df_nd["drop_number_concentration"]
    df_nd = df_nd[df_nd["R"] != 0]
    df_nd = df_nd[df_nd["N(D)"] != 0]

    # Compute normalized density
    df_nd["D/D50"] = df_nd["D"] / df_nd["D50"]
    df_nd["D/Dm"] = df_nd["D"] / df_nd["Dm"]
    df_nd["N(D)/Nw"] = df_nd["N(D)"] / df_nd["Nw"]
    df_nd["log10[N(D)/Nw]"] = np.log10(df_nd["N(D)/Nw"])
    return df_nd


def plot_normalized_dsd_density(df_nd, x="D/D50", figsize=(8, 8), dpi=300):
    """Plot normalized DSD N(D)/Nw ~ D/D50 (or D/Dm) density."""
    ds_stats = compute_2d_histogram(
        df_nd,
        x=x,
        y="N(D)/Nw",
        x_bins=np.arange(0, 4, 0.025),
        y_bins=log_arange(1e-5, 50, log_step=0.1, base=10),
    )
    cmap = plt.get_cmap("Spectral_r").copy()
    cmap.set_under(alpha=0)
    norm = LogNorm(1, None)

    ds_stats = ds_stats.isel({"N(D)/Nw": ds_stats["N(D)/Nw"] > 0})

    fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)
    p = ds_stats["count"].plot.pcolormesh(
        x=x,
        y="N(D)/Nw",
        ax=ax,
        vmin=1,
        cmap=cmap,
        norm=norm,
        extend="max",
        yscale="log",
    )
    ax.set_ylim(1e-5, 20)
    ax.set_xlim(0, 4)
    ax.set_xlabel(f"{x} [-]")
    ax.set_ylabel(r"$N(D)/N_w$ [-]")
    ax.set_title("Normalized DSD")
    return p


def plot_dsd_density(df_nd, diameter_bin_edges, figsize=(8, 8), dpi=300):
    """Plot N(D) ~ D density."""
    ds_stats = compute_2d_histogram(
        df_nd,
        x="D",
        y="N(D)",
        x_bins=diameter_bin_edges,
        y_bins=log_arange(0.1, 20_000, log_step=0.1, base=10),
    )
    cmap = plt.get_cmap("Spectral_r").copy()
    cmap.set_under(alpha=0)
    norm = LogNorm(1, None)
    fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)
    p = ds_stats["count"].plot.pcolormesh(x="D", y="N(D)", ax=ax, cmap=cmap, norm=norm, extend="max", yscale="log")
    ax.set_xlim(0, 8)
    ax.set_ylim(1, 20_000)
    ax.set_xlabel(r"$D$ [mm]")
    ax.set_ylabel(r"$N(D)$ [m$^{-3}$ mm$^{-1}$]")
    ax.set_title("DSD")
    return p


def plot_dsd_with_dense_lines(ds, figsize=(8, 8), dpi=300):
    """Plot N(D) ~ D using dense lines."""
    # Define intervals for rain rates
    r_bins = [0, 2, 5, 10, 50, 100, 500]

    # Define N(D) bins and diameter bin edeges
    y_bins = log_arange(1, 20_000, log_step=0.025, base=10)
    diameter_bin_edges = np.arange(0, 8, 0.01)

    # Resample N(D) to high resolution !
    # - quadratic, pchip
    da = resample_drop_number_concentration(
        ds["drop_number_concentration"],
        diameter_bin_edges=diameter_bin_edges,
        method="linear",
    )
    ds_resampled = xr.Dataset(
        {
            "R": ds["R"],
            "drop_number_concentration": da,
        },
    )

    # Define diameter bin edges to compute dense lines
    x_bins = da.disdrodb.diameter_bin_edges

    # Define discrete colormap (one color per rain-interval):
    cmap_list = [
        plt.get_cmap("Reds"),
        plt.get_cmap("Oranges"),
        plt.get_cmap("Purples"),
        plt.get_cmap("Greens"),
        plt.get_cmap("Blues"),
        plt.get_cmap("Grays"),
    ]
    cmap_list = [ListedColormap(cmap(np.arange(0, cmap.N))[-40:]) for cmap in cmap_list]

    # Compute dense lines
    dict_rgb = {}
    for i in range(0, len(r_bins) - 1):
        # Define dataset subset
        idx_rain_interval = np.logical_and(ds_resampled["R"] >= r_bins[i], ds_resampled["R"] < r_bins[i + 1])
        da = ds_resampled.isel(time=idx_rain_interval)["drop_number_concentration"]

        # Retrieve dense lines
        da_dense_lines = compute_dense_lines(
            da=da,
            coord="diameter_bin_center",
            x_bins=x_bins,
            y_bins=y_bins,
            normalization="max",
        )
        # Define cmap
        cmap = cmap_list[i]
        # Map colors and transparency
        # da_rgb = to_rgba(da_dense_lines, cmap=cmap, scaling="linear")
        # da_rgb = to_rgba(da_dense_lines, cmap=cmap, scaling="exp")
        # da_rgb = to_rgba(da_dense_lines, cmap=cmap, scaling="log")
        da_rgb = to_rgba(da_dense_lines, cmap=cmap, scaling="sqrt")

        dict_rgb[i] = da_rgb

    # Blend images with max-alpha
    ds_rgb = xr.concat(dict_rgb.values(), dim="r_class")
    da_blended = max_blend_images(ds_rgb, dim="r_class")

    # Prepare legend handles
    handles = []
    labels = []
    for i in range(len(r_bins) - 1):
        color = cmap_list[i](0.8)  # pick a representative color from each cmap
        handle = mlines.Line2D([], [], color=color, alpha=0.6, linewidth=2)
        label = f"[{r_bins[i]} - {r_bins[i+1]}]"
        handles.append(handle)
        labels.append(label)

    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)

    p = ax.pcolormesh(
        da_blended["diameter_bin_center"],
        da_blended["drop_number_concentration"],
        da_blended.data,
    )

    # Set axis scale and limits
    ax.set_yscale("log")
    ax.set_xlim(0, 8)
    ax.set_ylim(1, 20_000)

    # Add axis labels and title
    ax.set_xlabel(r"$D$ [mm]")
    ax.set_ylabel(r"$N(D)$ [m$^{-3}$ mm$^{-1}$]")
    ax.set_title("DSD")

    # Add legend with title
    ax.legend(handles, labels, title="Rain rate [mm/hr]", loc="upper right")

    # Return figure
    return p


####-------------------------------------------------------------------
#### DSD parameters plots


def define_lognorm_max_value(value):
    """Round up to next nice number: 90->100, 400->500, 1200->2000."""
    if value <= 0:
        return 1
    magnitude = 10 ** np.floor(np.log10(value))
    first_digit = value / magnitude
    nice_value = 1 if first_digit <= 1 else 2 if first_digit <= 2 else 5 if first_digit <= 5 else 10
    return nice_value * magnitude


def plot_dsd_params_relationships(df, add_nt=False, dpi=300):
    """Create a figure illustrating the relationships between DSD parameters."""
    # TODO: option to use D50 instead of Dm

    # Compute the required datasets for the plots
    # - Dm vs Nw
    ds_Dm_Nw_stats = compute_2d_histogram(
        df,
        x="Dm",
        y="Nw",
        variables=["R", "W", "Nt"],
        x_bins=np.arange(0, 8, 0.1),
        y_bins=log_arange(10, 1_000_000, log_step=0.05, base=10),
    )

    # - Dm vs LWC (W)
    ds_Dm_LWC_stats = compute_2d_histogram(
        df,
        x="Dm",
        y="W",
        variables=["R", "Nw", "Nt"],
        x_bins=np.arange(0, 8, 0.1),
        y_bins=log_arange(0.01, 10, log_step=0.05, base=10),
    )

    # - Dm vs R
    ds_Dm_R_stats = compute_2d_histogram(
        df,
        x="Dm",
        y="R",
        variables=["Nw", "W", "Nt"],
        x_bins=np.arange(0, 8, 0.1),
        y_bins=log_arange(0.1, 500, log_step=0.05, base=10),
    )

    # - Dm vs Nt
    ds_Dm_Nt_stats = compute_2d_histogram(
        df,
        x="Dm",
        y="Nt",
        variables=["R", "W", "Nw"],
        x_bins=np.arange(0, 8, 0.1),
        y_bins=log_arange(1, 100_000, log_step=0.05, base=10),
    )

    # Define different colormaps for each column
    cmap_counts = plt.get_cmap("viridis")
    cmap_lwc = plt.get_cmap("YlOrRd")
    cmap_r = plt.get_cmap("Blues")
    cmap_nt = plt.get_cmap("Greens")

    # Define normalizations for each variable
    norm_counts = LogNorm(1, None)
    norm_lwc = LogNorm(0.01, 10)
    norm_r = LogNorm(0.1, 500)
    # norm_nw = LogNorm(10, 100000)
    norm_nt = LogNorm(1, 10000)

    # Define axis limits
    dm_lim = (0.3, 6)
    nw_lim = (10, 1_000_000)
    lwc_lim = (0.01, 10)
    r_lim = (0.1, 500)
    nt_lim = (1, 100_000)

    # Define figure size
    if add_nt:
        figsize = (16, 16)
        nrows = 4
        height_ratios = [0.2, 1, 1, 1, 1]
    else:
        figsize = (16, 12)
        nrows = 3
        height_ratios = [0.2, 1, 1, 1]

    # Create figure with 4x4 subplots
    fig = plt.figure(figsize=figsize, dpi=dpi)
    gs = fig.add_gridspec(nrows + 1, 4, height_ratios=height_ratios, hspace=0.05, wspace=0.02)
    axes = np.empty((nrows + 1, 4), dtype=object)

    # Create colorbar axes in the bottom row of the gridspec
    cbar_axes = [fig.add_subplot(gs[0, j]) for j in range(4)]

    # Create axes for the grid
    for i in range(1, nrows + 1):
        for j in range(4):
            axes[i, j] = fig.add_subplot(gs[i, j])

    # Create empty subplot for diagonal elements (when y-axis variable matches column variable)
    # axes[2, 1].set_visible(False)
    # axes[3, 2].set_visible(False)
    # axes[4, 3].set_visible(False)

    ####-------------------------------------------------------------------.
    #### Dm vs Nw
    #### - Counts
    im_counts = ds_Dm_Nw_stats["count"].plot.pcolormesh(
        x="Dm",
        y="Nw",
        cmap=cmap_counts,
        norm=norm_counts,
        extend="max",
        yscale="log",
        add_colorbar=False,
        ax=axes[1, 0],
    )
    axes[1, 0].set_ylabel(r"$N_w$ [mm$^{-1}$ m$^{-3}$]")
    axes[1, 0].set_xlim(*dm_lim)
    axes[1, 0].set_ylim(nw_lim)

    #### - LWC
    im_lwc = ds_Dm_Nw_stats["W_median"].plot.pcolormesh(
        x="Dm",
        y="Nw",
        cmap=cmap_lwc,
        norm=norm_lwc,
        extend="both",
        yscale="log",
        add_colorbar=False,
        ax=axes[1, 1],
    )
    axes[1, 1].set_xlim(*dm_lim)
    axes[1, 1].set_ylim(nw_lim)

    #### - R
    im_r = ds_Dm_Nw_stats["R_median"].plot.pcolormesh(
        x="Dm",
        y="Nw",
        cmap=cmap_r,
        norm=norm_r,
        extend="both",
        yscale="log",
        add_colorbar=False,
        ax=axes[1, 2],
    )
    axes[1, 2].set_xlim(*dm_lim)
    axes[1, 2].set_ylim(nw_lim)
    axes[1, 2].set_yticklabels([])

    #### - Nt
    im_nt = ds_Dm_Nw_stats["Nt_median"].plot.pcolormesh(
        x="Dm",
        y="Nw",
        cmap=cmap_nt,
        norm=norm_nt,
        extend="both",
        yscale="log",
        add_colorbar=False,
        ax=axes[1, 3],
    )
    axes[1, 3].set_xlim(*dm_lim)
    axes[1, 3].set_ylim(nw_lim)
    axes[1, 3].set_yticklabels([])

    ####-------------------------------------------------------------------.
    #### Dm vs LWC
    #### - Counts
    ds_Dm_LWC_stats["count"].plot.pcolormesh(
        x="Dm",
        y="W",
        cmap=cmap_counts,
        norm=norm_counts,
        extend="max",
        yscale="log",
        add_colorbar=False,
        ax=axes[2, 0],
    )
    axes[2, 0].set_ylabel(r"LWC [g/m³]")
    axes[2, 0].set_xlim(*dm_lim)
    axes[2, 0].set_ylim(lwc_lim)

    #### - LWC
    # - Empty (diagonal where y-axis is W) - handled above in the loop
    ds_Dm_LWC_stats["R_median"].plot.pcolormesh(
        x="Dm",
        y="W",
        cmap=cmap_r,
        norm=norm_r,
        alpha=0,  # fully transparent
        extend="both",
        yscale="log",
        add_colorbar=False,
        ax=axes[2, 1],
    )
    axes[2, 1].set_xlim(*dm_lim)
    axes[2, 1].set_ylim(lwc_lim)
    axes[2, 1].set_yticklabels([])

    #### - R
    ds_Dm_LWC_stats["R_median"].plot.pcolormesh(
        x="Dm",
        y="W",
        cmap=cmap_r,
        norm=norm_r,
        extend="both",
        yscale="log",
        add_colorbar=False,
        ax=axes[2, 2],
    )
    axes[2, 2].set_xlim(*dm_lim)
    axes[2, 2].set_ylim(lwc_lim)
    axes[2, 2].set_yticklabels([])

    #### - Nt
    im_nt = ds_Dm_LWC_stats["Nt_median"].plot.pcolormesh(
        x="Dm",
        y="W",
        cmap=cmap_nt,
        norm=norm_nt,
        extend="both",
        yscale="log",
        add_colorbar=False,
        ax=axes[2, 3],
    )
    axes[2, 3].set_xlim(*dm_lim)
    axes[2, 3].set_ylim(lwc_lim)
    axes[2, 3].set_yticklabels([])

    ####-------------------------------------------------------------------.
    #### Dm vs R
    #### - Counts
    ds_Dm_R_stats["count"].plot.pcolormesh(
        x="Dm",
        y="R",
        cmap=cmap_counts,
        norm=norm_counts,
        extend="max",
        yscale="log",
        add_colorbar=False,
        ax=axes[3, 0],
    )
    axes[3, 0].set_ylabel(r"$R$ [mm h$^{-1}$]")
    axes[3, 0].set_xlim(*dm_lim)
    axes[3, 0].set_ylim(r_lim)

    #### - LWC
    im_lwc = ds_Dm_R_stats["W_median"].plot.pcolormesh(
        x="Dm",
        y="R",
        cmap=cmap_lwc,
        norm=norm_lwc,
        extend="both",
        yscale="log",
        add_colorbar=False,
        ax=axes[3, 1],
    )
    axes[3, 1].set_xlim(*dm_lim)
    axes[3, 1].set_ylim(r_lim)
    axes[3, 1].set_yticklabels([])

    #### - R
    # - Empty (diagonal where y-axis is R) - handled above in the loop
    #### - Nt
    ds_Dm_R_stats["Nt_median"].plot.pcolormesh(
        x="Dm",
        y="R",
        cmap=cmap_nt,
        norm=norm_nt,
        alpha=0,  # fully transparent
        extend="both",
        yscale="log",
        add_colorbar=False,
        ax=axes[3, 2],
    )
    axes[3, 2].set_xlim(*dm_lim)
    axes[3, 2].set_ylim(r_lim)
    axes[3, 2].set_yticklabels([])

    #### - Nt
    ds_Dm_R_stats["Nt_median"].plot.pcolormesh(
        x="Dm",
        y="R",
        cmap=cmap_nt,
        norm=norm_nt,
        extend="both",
        yscale="log",
        add_colorbar=False,
        ax=axes[3, 3],
    )
    axes[3, 3].set_xlim(*dm_lim)
    axes[3, 3].set_ylim(r_lim)
    axes[3, 3].set_yticklabels([])

    ####-------------------------------------------------------------------.
    #### Dm vs Nt
    if add_nt:
        #### - Counts
        ds_Dm_Nt_stats["count"].plot.pcolormesh(
            x="Dm",
            y="Nt",
            cmap=cmap_counts,
            norm=norm_counts,
            extend="max",
            yscale="log",
            add_colorbar=False,
            ax=axes[4, 0],
        )
        axes[4, 0].set_ylabel(r"$N_t$ [m$^{-3}$]")
        axes[4, 0].set_xlabel(r"$D_m$ [mm]")
        axes[4, 0].set_xlim(*dm_lim)
        axes[4, 0].set_ylim(nt_lim)

        #### - LWC
        ds_Dm_Nt_stats["W_median"].plot.pcolormesh(
            x="Dm",
            y="Nt",
            cmap=cmap_lwc,
            norm=norm_lwc,
            extend="both",
            yscale="log",
            add_colorbar=False,
            ax=axes[4, 1],
        )
        axes[4, 1].set_xlabel(r"$D_m$ [mm]")
        axes[4, 1].set_xlim(*dm_lim)
        axes[4, 1].set_ylim(nt_lim)
        axes[4, 1].set_yticklabels([])

        #### - R
        ds_Dm_Nt_stats["R_median"].plot.pcolormesh(
            x="Dm",
            y="Nt",
            cmap=cmap_r,
            norm=norm_r,
            extend="both",
            yscale="log",
            add_colorbar=False,
            ax=axes[4, 2],
        )
        axes[4, 2].set_xlabel(r"$D_m$ [mm]")
        axes[4, 2].set_xlim(*dm_lim)
        axes[4, 2].set_ylim(nt_lim)
        axes[4, 2].set_yticklabels([])

        #### - Nt
        # - Empty plot - handled above in the loop
        ds_Dm_Nt_stats["R_median"].plot.pcolormesh(
            x="Dm",
            y="Nt",
            cmap=cmap_r,
            norm=norm_r,
            alpha=0,  # fully transparent
            extend="both",
            yscale="log",
            add_colorbar=False,
            ax=axes[4, 2],
        )
        axes[4, 3].set_xlabel(r"$D_m$ [mm]")
        axes[4, 3].set_xlim(*dm_lim)
        axes[4, 3].set_ylim(nt_lim)
        axes[4, 3].set_yticklabels([])

    ####-------------------------------------------------------------------.
    #### Finalize figure
    # Remove x ticks and labels for all but bottom row
    for i in range(1, nrows):
        for j in range(4):
            if axes[i, j].get_visible():
                axes[i, j].set_xticklabels([])
                axes[i, j].set_xticks([])
                axes[i, j].set_xlabel("")

    # Remove y ticks and labels for all but left row
    for i in range(1, nrows + 1):
        for j in range(1, 4):
            if axes[i, j].get_visible():
                axes[i, j].set_yticks([])
                axes[i, j].set_yticklabels([])
                axes[i, j].set_ylabel("")

    # -------------------------------------------------.
    # Add colorbars
    # - Counts colorbar
    cbar1 = plt.colorbar(im_counts, cax=cbar_axes[0], orientation="horizontal", extend="both")
    cbar1.set_label("Counts", fontweight="bold")
    cbar1.ax.xaxis.set_label_position("top")
    cbar1.ax.set_aspect(0.25)
    # - LWC colorbar
    cbar2 = plt.colorbar(im_lwc, cax=cbar_axes[1], orientation="horizontal", extend="both")
    cbar2.set_label("Median LWC [g/m³]", fontweight="bold")
    cbar2.ax.xaxis.set_label_position("top")
    cbar2.ax.set_aspect(0.25)
    # - R colorbar
    cbar3 = plt.colorbar(im_r, cax=cbar_axes[2], orientation="horizontal", extend="both")
    cbar3.set_label("Median R [mm/h]", fontweight="bold")
    cbar3.ax.xaxis.set_label_position("top")
    cbar3.ax.set_aspect(0.3)
    # - Nt colorbar
    cbar4 = plt.colorbar(im_nt, cax=cbar_axes[3], orientation="horizontal", extend="both")
    cbar4.set_label("Median $N_t$ [m$^{-3}$]", fontweight="bold")
    cbar4.ax.xaxis.set_label_position("top")
    cbar4.ax.set_aspect(0.3)

    # -------------------------------------------------.
    # Return figure
    return fig


def plot_dsd_params_density(df, log_dm=False, lwc=True, log_normalize=False, figsize=(10, 10), dpi=300):
    """Generate a figure with various DSD relationships.

    All histograms are computed first, then normalized, and finally plotted together.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing DSD parameters (Dm, Nt, Nw, LWC/W, R, sigma_m, M2, M3, M4, M6)
    log_dm : bool, optional
        If True, use linear scale for Dm axes. If False, use log scale. Default is True.
    lwc : bool, optional
        If True, use Liquid Water Content (W). If False, use Rain Rate (R). Default is True.
    figsize : tuple, optional
        Figure size (width, height) in inches. Default is (18, 18).

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object containing all subplots
    axes : numpy.ndarray
        Array of all subplot axes
    """
    # TODO: option to use D50 instead of Dm

    # Common parameters
    cmap = plt.get_cmap("Spectral_r").copy()
    norm = Normalize(0, 1)  # Normalized data goes from 0 to 1

    # Define the water variable based on lwc flag
    df["LWC"] = df["W"]
    water_var = "LWC" if lwc else "R"
    water_label = "LWC [g/m³]" if lwc else "R [mm/h]"

    log_step = 0.05
    linear_step = 0.1

    # Dm range and scale settings
    if not log_dm:
        dm_bins = np.arange(0, 8, linear_step)
        dm_scale = None
        dm_lim = (0, 6)
        dm_ticklabels = [0, 2, 4, 6]
    else:
        dm_bins = log_arange(0.1, 10, log_step=log_step, base=10)
        dm_scale = "log"
        dm_lim = (0.3, 6)
        dm_ticklabels = [0.5, 1, 2, 5]

    # Nt and Nw range
    nt_bins = log_arange(1, 100_000, log_step=log_step, base=10)
    nw_bins = log_arange(1, 100_000, log_step=log_step, base=10)
    nw_lim = (10, 1_000_000)
    nt_lim = (1, 100_000)

    # Water range
    if lwc:
        water_bins = log_arange(0.001, 10, log_step=log_step, base=10)
        water_lim = (0.005, 10)
    else:
        water_bins = log_arange(0.1, 500, log_step=log_step, base=10)
        water_lim = (0.1, 500)

    # Define sigma_m bins
    sigma_bins = np.arange(0, 4, linear_step / 2)
    sigma_lim = (0, 3)

    # Compute all histograms first
    # 1. Dm vs Nt
    ds_stats_dm_nt = compute_2d_histogram(
        df,
        x="Dm",
        y="Nt",
        x_bins=dm_bins,
        y_bins=nt_bins,
    )

    # 2. Dm vs Nw
    ds_stats_dm_nw = compute_2d_histogram(
        df,
        x="Dm",
        y="Nw",
        x_bins=dm_bins,
        y_bins=nw_bins,
    )

    # 3. Dm vs LWC/R
    ds_stats_dm_w = compute_2d_histogram(
        df,
        x="Dm",
        y=water_var,
        x_bins=dm_bins,
        y_bins=water_bins,
    )

    # 4. LWC/R vs Nt
    ds_stats_w_nt = compute_2d_histogram(
        df,
        x=water_var,
        y="Nt",
        x_bins=water_bins,
        y_bins=nt_bins,
    )

    # 5. LWC/R vs Nw
    ds_stats_w_nw = compute_2d_histogram(
        df,
        x=water_var,
        y="Nw",
        x_bins=water_bins,
        y_bins=nw_bins,
    )

    # 6. LWC/R vs sigma_m
    ds_stats_w_sigma = compute_2d_histogram(
        df,
        x=water_var,
        y="sigma_m",
        x_bins=water_bins,
        y_bins=sigma_bins,
    )

    # 7. M2 vs M4
    ds_stats_m2_m4 = compute_2d_histogram(
        df,
        x="M2",
        y="M4",
        x_bins=log_arange(1, 10_000, log_step=log_step, base=10),
        y_bins=log_arange(1, 40_000, log_step=log_step, base=10),
    )

    # 8. M3 vs M6
    ds_stats_m3_m6 = compute_2d_histogram(
        df,
        x="M3",
        y="M6",
        x_bins=log_arange(1, 10_000, log_step=log_step, base=10),
        y_bins=log_arange(0.1, 1000_000, log_step=log_step, base=10),
    )

    # 9. M2 vs M6
    ds_stats_m2_m6 = compute_2d_histogram(
        df,
        x="M2",
        y="M6",
        x_bins=log_arange(1, 10_000, log_step=log_step, base=10),
        y_bins=log_arange(0.1, 1000_000, log_step=log_step, base=10),
    )

    # Define normalization
    def max_normalize(ds):
        return ds["count"].where(ds["count"] > 0) / ds["count"].max().item()

    def log_max_normalize(ds):
        counts = ds["count"].where(ds["count"] > 0)
        log_counts = np.log10(counts)
        max_log = float(log_counts.max().item())
        return log_counts / max_log

    normalizer = log_max_normalize if log_normalize else max_normalize

    # Normalize all histograms
    ds_stats_dm_nt["normalized"] = normalizer(ds_stats_dm_nt)
    ds_stats_dm_nw["normalized"] = normalizer(ds_stats_dm_nw)
    ds_stats_dm_w["normalized"] = normalizer(ds_stats_dm_w)
    ds_stats_w_nt["normalized"] = normalizer(ds_stats_w_nt)
    ds_stats_w_nw["normalized"] = normalizer(ds_stats_w_nw)
    ds_stats_w_sigma["normalized"] = normalizer(ds_stats_w_sigma)
    ds_stats_m2_m4["normalized"] = normalizer(ds_stats_m2_m4)
    ds_stats_m3_m6["normalized"] = normalizer(ds_stats_m3_m6)
    ds_stats_m2_m6["normalized"] = normalizer(ds_stats_m2_m6)

    # Set up figure and axes
    fig, axes = plt.subplots(3, 3, figsize=figsize, dpi=dpi)
    fig.subplots_adjust(hspace=0.05, wspace=0.35)

    # COLUMN 1: All plots with Dm as x-axis
    # 1. Dm vs Nt (0,0)
    _ = ds_stats_dm_nt["normalized"].plot.pcolormesh(
        x="Dm",
        y="Nt",
        cmap=cmap,
        norm=norm,
        extend="max",
        xscale=dm_scale,
        yscale="log",
        add_colorbar=False,
        ax=axes[0, 0],
    )
    axes[0, 0].set_xlabel("")  # Hide x labels except for bottom row
    axes[0, 0].set_ylabel(r"$N_t$ [m$^{-3}$]")
    axes[0, 0].set_xlim(*dm_lim)
    axes[0, 0].set_ylim(*nt_lim)
    axes[0, 0].set_title(r"$D_m$ vs $N_t$")

    # 2. Dm vs Nw (1,0)
    _ = ds_stats_dm_nw["normalized"].plot.pcolormesh(
        x="Dm",
        y="Nw",
        cmap=cmap,
        norm=norm,
        extend="max",
        xscale=dm_scale,
        yscale="log",
        add_colorbar=False,
        ax=axes[1, 0],
    )
    axes[1, 0].set_xlabel("")  # Hide x labels except for bottom row
    axes[1, 0].set_ylabel(r"$N_w$ [mm$^{-1}$ m$^{-3}$]")
    axes[1, 0].set_xlim(*dm_lim)
    axes[1, 0].set_ylim(*nw_lim)
    axes[1, 0].set_title(r"$D_m$ vs $N_w$")

    # 3. Dm vs LWC/R (2,0)
    _ = ds_stats_dm_w["normalized"].plot.pcolormesh(
        x="Dm",
        y=water_var,
        cmap=cmap,
        norm=norm,
        extend="max",
        xscale=dm_scale,
        yscale="log",
        add_colorbar=False,
        ax=axes[2, 0],
    )
    axes[2, 0].set_xlabel(r"$D_m$ [mm]")
    axes[2, 0].set_ylabel(water_label)
    axes[2, 0].set_xlim(*dm_lim)
    if lwc:
        axes[2, 0].set_ylim(*water_lim)
        axes[2, 0].set_yticks([0.01, 0.1, 0.5, 1, 5])
        axes[2, 0].set_yticklabels(["0.01", "0.1", "0.5", "1", "5"])
    else:
        axes[2, 0].set_ylim(*water_lim)
    axes[2, 0].set_title(f"$D_m$ vs {water_var}")

    axes[2, 0].set_xticks(dm_ticklabels)
    axes[2, 0].set_xticklabels([str(v) for v in dm_ticklabels])

    # COLUMN 2: All plots with LWC/R as x-axis
    # 4. LWC/R vs Nt (0,1)
    _ = ds_stats_w_nt["normalized"].plot.pcolormesh(
        x=water_var,
        y="Nt",
        cmap=cmap,
        norm=norm,
        extend="max",
        xscale="log",
        yscale="log",
        add_colorbar=False,
        ax=axes[0, 1],
    )
    axes[0, 1].set_xlabel("")  # Hide x labels except for bottom row
    axes[0, 1].set_ylabel(r"$N_t$ [m$^{-3}$]")
    if lwc:
        axes[0, 1].set_xlim(*water_lim)
        axes[0, 1].set_xticks([0.01, 0.1, 1, 10])
        axes[0, 1].set_xticklabels(["0.01", "0.1", "1", "10"])
    else:
        axes[0, 1].set_xlim(*water_lim)
    axes[0, 1].set_ylim(*nt_lim)
    axes[0, 1].set_title(f"{water_var} vs $N_t$")

    # 5. LWC/R vs Nw (1,1)
    _ = ds_stats_w_nw["normalized"].plot.pcolormesh(
        x=water_var,
        y="Nw",
        cmap=cmap,
        norm=norm,
        extend="max",
        xscale="log",
        yscale="log",
        add_colorbar=False,
        ax=axes[1, 1],
    )
    axes[1, 1].set_xlabel("")  # Hide x labels except for bottom row
    axes[1, 1].set_ylabel(r"$N_w$ [mm$^{-1}$ m$^{-3}$]")
    if lwc:
        axes[1, 1].set_xlim(*water_lim)
        axes[1, 1].set_xticks([0.01, 0.1, 1, 10])
        axes[1, 1].set_xticklabels(["0.01", "0.1", "1", "10"])
    else:
        axes[1, 1].set_xlim(*water_lim)
    axes[1, 1].set_ylim(*nw_lim)
    axes[1, 1].set_title(f"{water_var} vs $N_w$")

    # 6. LWC/R vs sigma_m (2,1)
    _ = ds_stats_w_sigma["normalized"].plot.pcolormesh(
        x=water_var,
        y="sigma_m",
        cmap=cmap,
        norm=norm,
        extend="max",
        xscale="log",
        add_colorbar=False,
        ax=axes[2, 1],
    )
    axes[2, 1].set_xlabel(water_label)
    axes[2, 1].set_ylabel(r"$\sigma_m$ [mm]")
    if lwc:
        axes[2, 1].set_xlim(*water_lim)
        axes[2, 1].set_xticks([0.01, 0.1, 1, 10])
        axes[2, 1].set_xticklabels(["0.01", "0.1", "1", "10"])
    else:
        axes[2, 1].set_xlim(*water_lim)
    axes[2, 1].set_ylim(*sigma_lim)

    axes[2, 1].set_title(rf"{water_var} vs $\sigma_m$")

    # COLUMN 3: Moment relationships
    # 7. M2 vs M4 (0,2)
    _ = ds_stats_m2_m4["normalized"].plot.pcolormesh(
        x="M2",
        y="M4",
        cmap=cmap,
        norm=norm,
        extend="max",
        xscale="log",
        yscale="log",
        add_colorbar=False,
        ax=axes[0, 2],
    )
    axes[0, 2].set_xlabel("")  # Hide x labels except for bottom row
    axes[0, 2].set_ylabel(r"M4 [m$^{-3}$ mm$^{4}$]")
    axes[0, 2].set_xlim(1, 10_000)
    axes[0, 2].set_ylim(1, 40_000)
    axes[0, 2].set_title(r"M2 vs M4")

    # 8. M3 vs M6 (1,2)
    _ = ds_stats_m3_m6["normalized"].plot.pcolormesh(
        x="M3",
        y="M6",
        cmap=cmap,
        norm=norm,
        extend="max",
        xscale="log",
        yscale="log",
        add_colorbar=False,
        ax=axes[1, 2],
    )
    axes[1, 2].set_xlabel("")  # Hide x labels except for bottom row
    axes[1, 2].set_ylabel(r"M6 [m$^{-3}$ mm$^{6}$]")
    axes[1, 2].set_xlim(1, 10_000)
    axes[1, 2].set_ylim(0.1, 1000_000)
    axes[1, 2].set_title(r"M3 vs M6")

    # 9. M2 vs M6 (2,2)
    _ = ds_stats_m2_m6["normalized"].plot.pcolormesh(
        x="M2",
        y="M6",
        cmap=cmap,
        norm=norm,
        extend="max",
        xscale="log",
        yscale="log",
        add_colorbar=False,
        ax=axes[2, 2],
    )
    axes[2, 2].set_xlabel(r"M* [m$^{-3}$ mm$^{*}$]")
    axes[2, 2].set_ylabel(r"M6 [m$^{-3}$ mm$^{6}$]")
    axes[2, 2].set_xlim(1, 10_000)
    axes[2, 2].set_ylim(0.1, 1000_000)
    axes[2, 2].set_title(r"M2 vs M6")

    # Remove x-axis ticks and ticklabels for all but bottom row
    for i in range(2):
        for j in range(3):
            axes[i, j].set_xticklabels([])
            axes[i, j].tick_params(axis="x", which="both", bottom=False)

    # Add subplot titles as text in top left corner of each plot
    title_bbox_dict = {
        "facecolor": "white",
        "alpha": 0.7,
        "edgecolor": "none",
        "pad": 1,
    }
    for ax in axes.flatten():
        # Add text in top left corner with some padding
        ax.text(
            0.03,
            0.95,
            ax.get_title(),
            transform=ax.transAxes,
            fontsize=11,
            #  fontweight='bold',
            ha="left",
            va="top",
            bbox=title_bbox_dict,
        )
        ax.set_title("")

    return fig


def plot_dmax_relationships(df, diameter_bin_edges, dmax="Dmax", diameter_max=10, norm_vmax=None, dpi=300):
    """
    Plot 2x2 subplots showing relationships between Dmax and precipitation parameters.

    Parameters
    ----------
    df : DataFrame
        Input dataframe containing the precipitation data
    dmax : str, default "Dmax"
        Column name for maximum diameter
    vmax : float, default 10
        Maximum value for Dmax axis limits
    dpi : int, default 300
        Resolution for the figure
    """
    # Compute 2D histograms
    # - Dmax-R
    ds_stats_dmax_r = compute_2d_histogram(
        df,
        x=dmax,
        y="R",
        x_bins=diameter_bin_edges,
        y_bins=log_arange(0.1, 500, log_step=0.05, base=10),
    )
    # - Dmax-Nw
    ds_stats_dmax_nw = compute_2d_histogram(
        df,
        x=dmax,
        y="Nw",
        x_bins=diameter_bin_edges,
        y_bins=log_arange(10, 1_000_000, log_step=0.05, base=10),
    )
    # - Dmax-Nt
    ds_stats_dmax_nt = compute_2d_histogram(
        df,
        x=dmax,
        y="Nt",
        x_bins=diameter_bin_edges,
        y_bins=log_arange(1, 100_000, log_step=0.05, base=10),
    )
    # - Dmax-Dm
    ds_stats_dmax_dm = compute_2d_histogram(
        df,
        x=dmax,
        y="Dm",
        variables=["R", "Nw", "sigma_m"],
        x_bins=diameter_bin_edges,
        y_bins=np.arange(0, 8, 0.05),
    )

    # Define vmax for counts
    if norm_vmax:
        norm_vmax = define_lognorm_max_value(ds_stats_dmax_r["count"].max().item())

    # Define plotting parameters
    cmap = plt.get_cmap("Spectral_r").copy()
    cmap.set_under(alpha=0)
    norm = LogNorm(1, norm_vmax)

    # Create figure with 2x2 subplots
    figsize = (8, 6)
    fig = plt.figure(figsize=figsize, dpi=dpi)

    # Create main gridspec with larger space between plots and colorbar
    # - Horizontal colorbar
    main_gs = fig.add_gridspec(2, 1, height_ratios=[1, 0.20], hspace=0.15)

    # - Vertical colorbar
    # main_gs = fig.add_gridspec(1, 2, width_ratios=[1, 0.20], wspace=0.15)

    # Create nested gridspec for the 2x2 subplots with smaller internal spacing
    subplots_gs = main_gs[0].subgridspec(2, 2, hspace=0.05, wspace=0.05)

    # Create the 2x2 subplot grid
    axes = np.array(
        [
            [fig.add_subplot(subplots_gs[0, 0]), fig.add_subplot(subplots_gs[0, 1])],
            [fig.add_subplot(subplots_gs[1, 0]), fig.add_subplot(subplots_gs[1, 1])],
        ],
    )

    # - Dmax vs R (top-left)
    ax1 = axes[0, 0]
    p1 = ds_stats_dmax_r["count"].plot.pcolormesh(
        x=dmax,
        y="R",
        cmap=cmap,
        norm=norm,
        extend="max",
        yscale="log",
        add_colorbar=False,
        ax=ax1,
    )
    ax1.set_xlabel(r"$D_{max}$ [mm]")
    ax1.set_ylabel(r"$R$ [mm h$^{-1}$]")
    ax1.set_xlim(0.2, diameter_max)
    ax1.set_ylim(0.1, 500)

    # - Dmax vs Nw (top-right)
    ax2 = axes[0, 1]
    _ = ds_stats_dmax_nw["count"].plot.pcolormesh(
        x=dmax,
        y="Nw",
        cmap=cmap,
        norm=norm,
        extend="max",
        yscale="log",
        add_colorbar=False,
        ax=ax2,
    )
    ax2.set_xlabel(r"$D_{max}$ [mm]")
    ax2.set_ylabel(r"$N_w$ [mm$^{-1}$ m$^{-3}$]")
    ax2.set_xlim(0.2, diameter_max)
    ax2.set_ylim(10, 1_000_000)

    # - Dmax vs Nt (bottom-left)
    ax3 = axes[1, 0]
    _ = ds_stats_dmax_nt["count"].plot.pcolormesh(
        x=dmax,
        y="Nt",
        cmap=cmap,
        norm=norm,
        extend="max",
        yscale="log",
        add_colorbar=False,
        ax=ax3,
    )
    ax3.set_xlabel(r"$D_{max}$ [mm]")
    ax3.set_ylabel(r"$N_t$ [m$^{-3}$]")
    ax3.set_xlim(0.2, diameter_max)
    ax3.set_ylim(1, 100_000)

    # - Dmax vs Dm (bottom-right)
    ax4 = axes[1, 1]
    _ = ds_stats_dmax_dm["count"].plot.pcolormesh(
        x=dmax,
        y="Dm",
        cmap=cmap,
        norm=norm,
        extend="max",
        add_colorbar=False,
        ax=ax4,
    )
    ax4.set_xlabel(r"$D_{max}$ [mm]")
    ax4.set_ylabel(r"$D_m$ [mm]")
    ax4.set_xlim(0.2, diameter_max)
    ax4.set_ylim(0, 6)

    # Remove xaxis labels and ticklables labels on first row
    for ax in axes[0, :]:  # First row (both columns)
        ax.set_xlabel("")
        ax.set_xticks([])
        ax.set_xticklabels([])
        ax.tick_params(axis="x", which="both", bottom=True, top=False, labelbottom=False)

    # Move y-axis of second column to the right
    for ax in axes[:, 1]:  # Second column (both rows)
        ax.yaxis.tick_right()
        ax.yaxis.set_label_position("right")

    # Add titles as legends in upper corners
    title_bbox_dict = {
        "facecolor": "white",
        "alpha": 0.7,
        "edgecolor": "none",
        "pad": 1,
    }
    axes[0, 0].text(
        0.05,
        0.95,
        r"$D_{max}$ vs $R$",
        transform=axes[0, 0].transAxes,
        fontsize=12,
        verticalalignment="top",
        bbox=title_bbox_dict,
    )

    axes[0, 1].text(
        0.05,
        0.95,
        r"$D_{max}$ vs $N_w$",
        transform=axes[0, 1].transAxes,
        fontsize=12,
        verticalalignment="top",
        bbox=title_bbox_dict,
    )

    axes[1, 0].text(
        0.05,
        0.95,
        r"$D_{max}$ vs $N_t$",
        transform=axes[1, 0].transAxes,
        fontsize=12,
        verticalalignment="top",
        bbox=title_bbox_dict,
    )

    axes[1, 1].text(
        0.05,
        0.95,
        r"$D_{max}$ vs $D_m$",
        transform=axes[1, 1].transAxes,
        fontsize=12,
        verticalalignment="top",
        bbox=title_bbox_dict,
    )

    # Add colorbar
    cax = fig.add_subplot(main_gs[1])
    # - Horizontal colorbar
    cbar = fig.colorbar(p1, cax=cax, extend="max", orientation="horizontal")
    cbar.set_label("Counts", labelpad=10)
    cbar.ax.set_aspect(0.1)
    cbar.ax.xaxis.set_label_position("top")

    # - Vertical colorbar
    # cbar = fig.colorbar(p2, cax=cax, extend="max")
    # cbar.set_label('Count', rotation=270, labelpad=10)
    # cbar.ax.set_aspect(10)
    return fig


####-------------------------------------------------------------------
#### Radar plots


def _define_coeff_string(a):
    # - Format a coefficient as m * 10^{e}
    m_str, e_str = f"{a:.2e}".split("e")
    m, e = float(m_str), int(e_str)
    # Build coefficient string
    a_str = f"{a:.2f}" if e >= -1 else f"{m:.2f} \\times 10^{{{e}}}"
    return a_str


def get_symbol_str(symbol, pol=""):
    """Generate symbol string with optional polarization subscript.

    Parameters
    ----------
    symbol : str
        The base symbol (e.g., 'A', 'Z', 'z')
    pol : str, optional
        Polarization identifier (e.g., 'H', 'V')

    Returns
    -------
    str
        LaTeX formatted symbol string
    """
    if pol:
        return rf"{symbol}_{{\mathrm{{{pol}}}}}"
    return symbol


def plot_A_R(
    df,
    a,
    r,
    cmap=None,
    norm=None,
    add_colorbar=True,
    add_fit=True,
    pol="",
    title=None,
    ax=None,
    figsize=(8, 8),
    dpi=300,
    legend_fontsize=14,
):
    """Create a 2D histogram of A vs R."""
    # Define a_min and a_max
    a_min = 0.001
    a_max = 10
    a_bins = log_arange(a_min, a_max, log_step=0.025, base=10)
    rlims = (0.1, 500)
    r_bins = log_arange(*rlims, log_step=0.025, base=10)

    # Compute 2D histogram
    ds_stats = compute_2d_histogram(
        df,
        x=r,
        y=a,
        x_bins=r_bins,
        y_bins=a_bins,
    )

    # Define colormap and norm
    if cmap is None:
        cmap = plt.get_cmap("Spectral_r").copy()
        cmap.set_under(alpha=0)
    norm = LogNorm(1, None) if norm is None else norm

    # Define ticks and ticklabels
    r_ticks = [0.1, 1, 10, 50, 100, 500]
    a_ticks = [0.001, 0.01, 0.1, 0.5, 1, 5]  # Adapt on a_max

    # Define A symbol
    a_symbol = get_symbol_str("A", pol)

    # Set default title if not provided
    if title is None:
        title = rf"${a_symbol}$ vs $R$"

    # Create figure if ax is None
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)

    # Plot 2D histogram
    p = ds_stats["count"].plot.pcolormesh(
        x=r,
        y=a,
        ax=ax,
        cmap=cmap,
        norm=norm,
        add_colorbar=add_colorbar,
        extend="max",
        xscale="log",
        yscale="log",
        cbar_kwargs={"label": "Counts"} if add_colorbar else {},
    )
    ax.set_xlabel(r"$R$ [mm h$^{-1}$]")
    ax.set_ylabel(rf"${a_symbol}$ [dB km$^{{-1}}$]")
    ax.set_ylim(a_min, a_max)
    ax.set_xlim(*rlims)
    ax.set_xticks(r_ticks)
    ax.set_xticklabels([str(v) for v in r_ticks])
    ax.set_yticks(a_ticks)
    ax.set_yticklabels([str(v) for v in a_ticks])
    ax.set_title(title)
    if add_fit:
        # Fit powerlaw k = a * R ** b
        (a_c, b), _ = fit_powerlaw(x=df[r], y=df[a], xbins=r_bins, x_in_db=False)
        # Invert for R = A * k ** B
        A_c, B = inverse_powerlaw_parameters(a_c, b)
        # Define legend title
        a_str = _define_coeff_string(a_c)
        A_str = _define_coeff_string(A_c)
        legend_str = rf"${a_symbol} =  {a_str} \, R^{{{b:.2f}}}$" "\n" rf"$R = {A_str} \, {a_symbol}^{{{B:.2f}}}$"
        # Get power law predictions
        x_pred = np.arange(*rlims)
        r_pred = predict_from_powerlaw(x_pred, a=a_c, b=b)
        # Add fitted power law
        ax.plot(x_pred, r_pred, linestyle="dashed", color="black")
        # Add legend
        legend_bbox_dict = {"facecolor": "white", "edgecolor": "black", "alpha": 0.7}
        ax.text(
            0.05,
            0.95,
            legend_str,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=legend_fontsize,
            bbox=legend_bbox_dict,
        )
    return p


def plot_A_Z(
    df,
    a,
    z,
    cmap=None,
    norm=None,
    add_colorbar=True,
    add_fit=True,
    pol="",
    title=None,
    ax=None,
    a_lim=(0.0001, 10),
    z_lim=(0, 70),
    figsize=(8, 8),
    dpi=300,
    legend_fontsize=14,
):
    """Create a 2D histogram of A vs Z."""
    # Define bins
    a_bins = log_arange(*a_lim, log_step=0.025, base=10)
    z_bins = np.arange(*z_lim, 0.5)

    # Compute 2d histogram
    ds_stats = compute_2d_histogram(
        df,
        x=z,
        y=a,
        x_bins=z_bins,
        y_bins=a_bins,
    )

    # Define colormap and norm
    if cmap is None:
        cmap = plt.get_cmap("Spectral_r").copy()
        cmap.set_under(alpha=0)
    if norm is None:
        norm = LogNorm(1, None)

    # Ticks
    a_ticks = [0.001, 0.01, 0.1, 0.5, 1, 5]

    # Define symbols
    a_symbol = get_symbol_str("A", pol)
    z_symbol = get_symbol_str("Z", pol)
    z_lower_symbol = get_symbol_str("z", pol)

    # Set default title if not provided
    if title is None:
        title = rf"${a_symbol}$ vs ${z_symbol}$"

    # Create figure if ax is None
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)

    # Plot 2D histogram
    p = ds_stats["count"].plot.pcolormesh(
        x=z,
        y=a,
        ax=ax,
        cmap=cmap,
        norm=norm,
        add_colorbar=add_colorbar,
        extend="max",
        xscale=None,
        yscale="log",
        cbar_kwargs={"label": "Counts"} if add_colorbar else {},
    )
    ax.set_xlabel(rf"${z_symbol}$ [dBZ]")
    ax.set_ylabel(rf"${a_symbol}$ [dB km$^{{-1}}$]")
    ax.set_xlim(*z_lim)
    ax.set_ylim(*a_lim)
    ax.set_yticks(a_ticks)
    ax.set_yticklabels([str(v) for v in a_ticks])
    ax.set_title(title)

    # Fit and plot the power law
    if add_fit:
        # Fit powerlaw k = a * Z ** b  (Z in dBZ -> x_in_db=True)
        (a_c, b), _ = fit_powerlaw(
            x=df[z],
            y=df[a],
            xbins=z_bins,
            x_in_db=True,
        )
        # Invert for Z = A * k ** B
        A_c, B = inverse_powerlaw_parameters(a_c, b)
        # Legend text
        a_str = _define_coeff_string(a_c)
        A_str = _define_coeff_string(A_c)
        legend_str = (
            rf"${a_symbol} = {a_str} \, {z_lower_symbol}^{{{b:.2f}}}$"
            "\n"
            rf"${z_lower_symbol} = {A_str} \, {a_symbol}^{{{B:.2f}}}$"
        )
        # Predictions
        x_pred = np.arange(*z_lim)
        x_pred_linear = disdrodb.idecibel(x_pred)  # convert to linear for prediction
        y_pred = predict_from_powerlaw(x_pred_linear, a=a_c, b=b)
        ax.plot(x_pred, y_pred, linestyle="dashed", color="black")
        # Add legend
        legend_bbox_dict = {"facecolor": "white", "edgecolor": "black", "alpha": 0.7}
        ax.text(
            0.05,
            0.95,
            legend_str,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=legend_fontsize,
            bbox=legend_bbox_dict,
        )
    return p


def plot_A_KDP(
    df,
    a,
    kdp,
    log_a=True,
    log_kdp=False,
    a_lim=(0.001, 10),
    kdp_lim=None,
    pol="",
    ax=None,
    cmap=None,
    norm=None,
    add_colorbar=True,
    add_fit=True,
    title=None,
    figsize=(8, 8),
    dpi=300,
    legend_fontsize=14,
):
    """Create a 2D histogram of k(H/V) vs KDP."""
    # Bins & limits for a
    if log_a:
        a_bins = log_arange(*a_lim, log_step=0.025, base=10)
        yscale = "log"
        a_ticks = [0.001, 0.01, 0.1, 0.5, 1, 5]
    else:
        a_bins = np.arange(a_lim[0], a_lim[1], 0.01)
        yscale = None
        a_ticks = None

    # Bins & limits for KDP
    if log_kdp:
        kdp_lim = (0.05, 10) if kdp_lim is None else kdp_lim
        kdp_bins = log_arange(*kdp_lim, log_step=0.05, base=10)
        xscale = "log"
        kdp_ticks = [0.05, 0.1, 0.5, 1, 5, 10]
    else:
        kdp_lim = (0, 8) if kdp_lim is None else kdp_lim
        kdp_bins = np.arange(kdp_lim[0], kdp_lim[1], 0.1)
        xscale = None
        kdp_ticks = None

    # Compute 2D histogram
    ds_stats = compute_2d_histogram(
        df,
        x=kdp,
        y=a,
        x_bins=kdp_bins,
        y_bins=a_bins,
    )

    # Colormap & norm
    if cmap is None:
        cmap = plt.get_cmap("Spectral_r").copy()
        cmap.set_under(alpha=0)
    if norm is None:
        norm = LogNorm(1, None)

    # Define symbols
    a_symbol = get_symbol_str("A", pol)

    # Set default title if not provided
    if title is None:
        title = rf"${a_symbol}$ vs $K_{{\mathrm{{DP}}}}$"

    # Create figure if ax is None
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)

    # Plot 2D histogram
    p = ds_stats["count"].plot.pcolormesh(
        x=kdp,
        y=a,
        ax=ax,
        cmap=cmap,
        norm=norm,
        add_colorbar=add_colorbar,
        extend="max",
        xscale=xscale,
        yscale=yscale,
        cbar_kwargs={"label": "Counts"} if add_colorbar else {},
    )
    ax.set_xlabel(r"$K_{\mathrm{DP}}$ [deg km$^{-1}$]")
    ax.set_ylabel(rf"${a_symbol}$ [dB km$^{{-1}}$]")
    ax.set_xlim(*kdp_lim)
    ax.set_ylim(*a_lim)
    if kdp_ticks is not None:
        ax.set_xticks(kdp_ticks)
        ax.set_xticklabels([str(v) for v in kdp_ticks])
    if a_ticks is not None:
        ax.set_yticks(a_ticks)
        ax.set_yticklabels([str(v) for v in a_ticks])
    ax.set_title(title)

    # Fit and overlay power law: k = a * KDP^b
    if add_fit:
        (a_c, b), _ = fit_powerlaw(
            x=df[kdp],
            y=df[a],
            xbins=kdp_bins,
            x_in_db=False,
        )
        # Invert: KDP = A * k^B
        A_c, B = inverse_powerlaw_parameters(a_c, b)

        a_str = _define_coeff_string(a_c)
        A_str = _define_coeff_string(A_c)
        legend_str = (
            rf"${a_symbol} = {a_str}\,K_{{\mathrm{{DP}}}}^{{{b:.2f}}}$"
            "\n"
            rf"$K_{{\mathrm{{DP}}}} = {A_str}\,{a_symbol}^{{{B:.2f}}}$"
        )

        # Predictions along KDP axis
        if log_kdp:
            x_pred = np.logspace(np.log10(kdp_lim[0]), np.log10(kdp_lim[1]), 400)
        else:
            x_pred = np.arange(kdp_lim[0], kdp_lim[1], 0.05)
        y_pred = predict_from_powerlaw(x_pred, a=a_c, b=b)

        ax.plot(x_pred, y_pred, linestyle="dashed", color="black")
        # Add legend
        legend_bbox_dict = {"facecolor": "white", "edgecolor": "black", "alpha": 0.7}
        ax.text(
            0.05,
            0.95,
            legend_str,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=legend_fontsize,
            bbox=legend_bbox_dict,
        )

    return p


def plot_R_Z(
    df,
    z,
    r,
    cmap=None,
    norm=None,
    add_colorbar=True,
    add_fit=True,
    pol="",
    title=None,
    ax=None,
    figsize=(8, 8),
    dpi=300,
    legend_fontsize=14,
):
    """Create a 2D histogram of Z vs R."""
    # Define axis limits
    z_lims = (0, 70)
    r_lims = (0.1, 500)

    # Compute 2d histogram
    ds_stats = compute_2d_histogram(
        df,
        x=z,
        y=r,
        x_bins=np.arange(*z_lims, 0.5),
        y_bins=log_arange(*r_lims, log_step=0.05, base=10),
    )

    # Define colormap and norm
    if cmap is None:
        cmap = plt.get_cmap("Spectral_r").copy()
        cmap.set_under(alpha=0)
    norm = LogNorm(1, None) if norm is None else norm

    # Define rain ticks and ticklabels
    r_ticks = [0.1, 1, 10, 50, 100, 500]

    # Define symbols
    z_symbol = get_symbol_str("Z", pol)
    z_lower_symbol = get_symbol_str("z", pol)

    # Set default title if not provided
    if title is None:
        title = rf"${z_symbol}$ vs $R$"

    # Create figure if ax is None
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)

    # Plot 2D histogram
    p = ds_stats["count"].plot.pcolormesh(
        x=z,
        y=r,
        ax=ax,
        cmap=cmap,
        norm=norm,
        add_colorbar=add_colorbar,
        extend="max",
        yscale="log",
        cbar_kwargs={"label": "Counts"} if add_colorbar else {},
    )
    ax.set_ylabel(r"$R$ [mm h$^{-1}$]")
    ax.set_xlabel(rf"${z_symbol}$ [dBZ]")
    ax.set_xlim(*z_lims)
    ax.set_ylim(*r_lims)
    ax.set_yticks(r_ticks)
    ax.set_yticklabels([str(v) for v in r_ticks])
    ax.set_title(title)

    # Fit and plot the powerlaw
    if add_fit:
        # Fit powerlaw R = a * z ** b
        (a, b), _ = fit_powerlaw(x=df[z], y=df[r], xbins=np.arange(10, 50, 1), x_in_db=True)
        # Invert for z = A * R ** B
        A, B = inverse_powerlaw_parameters(a, b)
        # Define legend title
        a_str = _define_coeff_string(a)
        A_str = _define_coeff_string(A)
        legend_str = (
            rf"$R = {a_str} \, {z_lower_symbol}^{{{b:.2f}}}$" "\n" rf"${z_lower_symbol} = {A_str} \, R^{{{B:.2f}}}$"
        )
        # Get power law predictions
        x_pred = np.arange(*z_lims)
        x_pred_linear = disdrodb.idecibel(x_pred)
        r_pred = predict_from_powerlaw(x_pred_linear, a=a, b=b)
        # Add fitted powerlaw
        ax.plot(x_pred, r_pred, linestyle="dashed", color="black")
        # Add legend
        legend_bbox_dict = {"facecolor": "white", "edgecolor": "black", "alpha": 0.7}
        ax.text(
            0.05,
            0.95,
            legend_str,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=legend_fontsize,
            bbox=legend_bbox_dict,
        )
    return p


def plot_R_KDP(
    df,
    kdp,
    r,
    kdp_lim=None,
    r_lim=None,
    cmap=None,
    norm=None,
    add_colorbar=True,
    log_scale=False,
    add_fit=True,
    title=None,
    ax=None,
    figsize=(8, 8),
    dpi=300,
    legend_fontsize=14,
):
    """Create a 2D histogram of KDP vs R."""
    # Define bins
    if not log_scale:
        kdp_lim = (0, 8) if kdp_lim is None else kdp_lim
        r_lim = (0, 200) if r_lim is None else r_lim
        xbins = np.arange(*kdp_lim, 0.1)
        ybins = np.arange(*r_lim, 1)
        xscale = None
        yscale = None
    else:
        kdp_lim = (0.1, 10) if kdp_lim is None else kdp_lim
        r_lim = (0.1, 500) if r_lim is None else r_lim
        xbins = log_arange(*kdp_lim, log_step=0.05, base=10)
        ybins = log_arange(*r_lim, log_step=0.05, base=10)
        xscale = "log"
        yscale = "log"

    # Compute 2d histogram
    ds_stats = compute_2d_histogram(
        df,
        x=kdp,
        y=r,
        x_bins=xbins,
        y_bins=ybins,
        # y_bins=log_arange(0.1, 500, log_step=0.05, base=10),
    )

    # Define colormap and norm
    if cmap is None:
        cmap = plt.get_cmap("Spectral_r").copy()
        cmap.set_under(alpha=0)
    norm = LogNorm(1, None) if norm is None else norm

    # Set default title if not provided
    if title is None:
        title = r"$K_{\mathrm{DP}}$ vs $R$"

    # Create figure if ax is None
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)

    # Plot 2D histogram
    p = ds_stats["count"].plot.pcolormesh(
        x=kdp,
        y=r,
        ax=ax,
        cmap=cmap,
        norm=norm,
        add_colorbar=add_colorbar,
        xscale=xscale,
        yscale=yscale,
        extend="max",
        cbar_kwargs={"label": "Counts"} if add_colorbar else {},
    )
    ax.set_ylabel(r"$R$ [mm h$^{-1}$]")
    ax.set_xlabel(r"$K_{\mathrm{DP}}$ [deg km$^{-1}$]")
    ax.set_xlim(*kdp_lim)
    ax.set_ylim(*r_lim)
    ax.set_title(title)

    # Fit and plot the power law
    if add_fit:
        # Fit powerlaw R = a * KDP ** b
        (a, b), _ = fit_powerlaw(x=df[kdp], y=df[r], xbins=xbins, x_in_db=False)
        # Invert for KDP = A * R ** B
        A, B = inverse_powerlaw_parameters(a, b)
        # Define legend title
        a_str = _define_coeff_string(a)
        A_str = _define_coeff_string(A)
        legend_str = (
            rf"$R = {a_str} \, K_{{\mathrm{{DP}}}}^{{{b:.2f}}}$"
            "\n"
            rf"$K_{{\mathrm{{DP}}}} = {A_str} \, R^{{{B:.2f}}}$"
        )
        # Get power law predictions
        x_pred = np.arange(*kdp_lim)
        r_pred = predict_from_powerlaw(x_pred, a=a, b=b)
        # Add fitted line
        ax.plot(x_pred, r_pred, linestyle="dashed", color="black")
        # Add legend
        legend_bbox_dict = {"facecolor": "white", "edgecolor": "black", "alpha": 0.7}
        ax.text(
            0.05,
            0.95,
            legend_str,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=legend_fontsize,
            bbox=legend_bbox_dict,
        )
    return p


def plot_ZDR_Z(
    df,
    z,
    zdr,
    zdr_lim=(0, 2.5),
    z_lim=(0, 70),
    cmap=None,
    norm=None,
    add_colorbar=True,
    add_fit=True,
    pol="",
    title=None,
    ax=None,
    figsize=(8, 8),
    dpi=300,
    legend_fontsize=14,
):
    """Create a 2D histogram of Zdr vs Z."""
    # Compute 2d histogram
    ds_stats = compute_2d_histogram(
        df,
        x=z,
        y=zdr,
        x_bins=np.arange(*z_lim, 0.5),
        y_bins=np.arange(*zdr_lim, 0.025),
    )

    # Define colormap and norm
    if cmap is None:
        cmap = plt.get_cmap("Spectral_r").copy()
        cmap.set_under(alpha=0)
    norm = LogNorm(1, None) if norm is None else norm

    # Define symbols
    z_symbol = get_symbol_str("Z", pol)
    z_lower_symbol = get_symbol_str("z", pol)

    # Set default title if not provided
    if title is None:
        title = rf"$Z_{{\mathrm{{DR}}}}$ vs ${z_symbol}$"

    # Create figure if ax is None
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)

    # Plot 2D histogram
    p = ds_stats["count"].plot.pcolormesh(
        x=z,
        y=zdr,
        ax=ax,
        cmap=cmap,
        norm=norm,
        add_colorbar=add_colorbar,
        extend="max",
        cbar_kwargs={"label": "Counts"} if add_colorbar else {},
    )
    ax.set_xlabel(rf"${z_symbol}$ [dBZ]")
    ax.set_ylabel(r"$Z_{DR}$ [dB]")
    ax.set_xlim(*z_lim)
    ax.set_ylim(*zdr_lim)
    ax.set_title(title)

    # Fit and plot the power law
    if add_fit:
        # Fit powerlaw ZDR = a * Z ** b
        (a, b), _ = fit_powerlaw(
            x=df[z],
            y=df[zdr],
            xbins=np.arange(5, 40, 1),
            x_in_db=True,
        )
        # Invert for Z = A * ZDR ** B
        A, B = inverse_powerlaw_parameters(a, b)
        # Define legend title
        a_str = _define_coeff_string(a)
        A_str = _define_coeff_string(A)
        legend_str = (
            rf"$Z_{{\mathrm{{DR}}}} = {a_str} \, {z_lower_symbol}^{{{b:.2f}}}$"
            "\n"
            rf"${z_lower_symbol} = {A_str} \, Z_{{\mathrm{{DR}}}}^{{{B:.2f}}}$"
        )
        # Get power law predictions
        x_pred = np.arange(0, 70)
        x_pred_linear = disdrodb.idecibel(x_pred)
        r_pred = predict_from_powerlaw(x_pred_linear, a=a, b=b)
        # Add fitted line
        ax.plot(x_pred, r_pred, linestyle="dashed", color="black")
        # Add legend
        legend_bbox_dict = {"facecolor": "white", "edgecolor": "black", "alpha": 0.7}
        ax.text(
            0.05,
            0.95,
            legend_str,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=legend_fontsize,
            bbox=legend_bbox_dict,
        )
    return p


def plot_KDP_Z(
    df,
    kdp,
    z,
    z_lim=(0, 70),
    log_kdp=False,
    kdp_lim=None,
    cmap=None,
    norm=None,
    add_colorbar=True,
    add_fit=True,
    pol="",
    title=None,
    ax=None,
    figsize=(8, 8),
    dpi=300,
    legend_fontsize=14,
):
    """Create a 2D histogram of KDP vs Z."""
    # Bins & limits
    z_bins = np.arange(*z_lim, 0.5)
    if log_kdp:
        kdp_lim = (0.01, 10) if kdp_lim is None else kdp_lim
        kdp_bins = log_arange(*kdp_lim, log_step=0.05, base=10)
        yscale = "log"
        kdp_ticks = [0.01, 0.1, 0.5, 1, 5, 10]
    else:
        kdp_lim = (0, 10) if kdp_lim is None else kdp_lim
        kdp_bins = np.arange(*kdp_lim, 0.1)
        yscale = None
        kdp_ticks = None

    # Compute 2D histogram
    ds_stats = compute_2d_histogram(
        df,
        x=z,
        y=kdp,
        x_bins=z_bins,
        y_bins=kdp_bins,
    )

    # Colormap & norm
    if cmap is None:
        cmap = plt.get_cmap("Spectral_r").copy()
        cmap.set_under(alpha=0)
    if norm is None:
        norm = LogNorm(1, None)

    # Define symbols
    z_symbol = get_symbol_str("Z", pol)
    z_lower_symbol = get_symbol_str("z", pol)

    # Set default title if not provided
    if title is None:
        title = rf"$K_{{\mathrm{{DP}}}}$ vs ${z_symbol}$"

    # Create figure if ax is None
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)

    # Plot 2D histogram
    p = ds_stats["count"].plot.pcolormesh(
        x=z,
        y=kdp,
        ax=ax,
        cmap=cmap,
        norm=norm,
        add_colorbar=add_colorbar,
        extend="max",
        yscale=yscale,
        cbar_kwargs={"label": "Counts"} if add_colorbar else {},
    )
    ax.set_xlabel(rf"${z_symbol}$ [dBZ]")
    ax.set_ylabel(r"$K_{\mathrm{DP}}$ [deg km$^{-1}$]")
    ax.set_xlim(*z_lim)
    ax.set_ylim(*kdp_lim)
    if kdp_ticks is not None:
        ax.set_yticks(kdp_ticks)
        ax.set_yticklabels([str(v) for v in kdp_ticks])
    ax.set_title(title)

    # Fit and overlay power law
    if add_fit:
        # Fit: KDP = a * Z^b   (Z in dBZ → x_in_db=True)
        (a, b), _ = fit_powerlaw(
            x=df[z],
            y=df[kdp],
            xbins=np.arange(15, 50),
            x_in_db=True,
        )
        # Invert: Z = A * KDP^B
        A, B = inverse_powerlaw_parameters(a, b)

        # Define legend title
        a_str = _define_coeff_string(a)
        A_str = _define_coeff_string(A)
        legend_str = (
            rf"$K_{{\mathrm{{DP}}}} = {a_str}\,{z_lower_symbol}^{{{b:.2f}}}$"
            "\n"
            rf"${z_lower_symbol} = {A_str}\,K_{{\mathrm{{DP}}}}^{{{B:.2f}}}$"
        )

        # Get power law predictions
        x_pred = np.arange(*z_lim)
        x_pred_linear = disdrodb.idecibel(x_pred)
        y_pred = predict_from_powerlaw(x_pred_linear, a=a, b=b)
        # Add fitted power law
        ax.plot(x_pred, y_pred, linestyle="dashed", color="black")
        # Add legend
        legend_bbox_dict = {"facecolor": "white", "edgecolor": "black", "alpha": 0.7}
        ax.text(
            0.05,
            0.95,
            legend_str,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=legend_fontsize,
            bbox=legend_bbox_dict,
        )

    return p


def plot_ADP_KDP_ZDR(
    df,
    adp,
    kdp,
    zdr,
    y_lim=(0, 0.015),
    zdr_lim=(0, 6),
    cmap=None,
    norm=None,
    add_colorbar=True,
    title=None,
    ax=None,
    figsize=(8, 8),
    dpi=300,
):
    """Create a 2D histogram of ADP/KDP vs ZDR.

    References
    ----------
    Ryzhkov, A., P. Zhang, and J. Hu, 2025.
    Suggested Modifications for the S-Band Polarimetric Radar Rainfall Estimation Algorithm.
    J. Hydrometeor., 26, 1053-1062. https://doi.org/10.1175/JHM-D-25-0014.1.
    """
    # Compute ADP/KDP
    df["ADP/KDP"] = df[adp] / df[kdp]

    # Bins & limits
    y_bins = np.arange(y_lim[0], y_lim[1], (y_lim[1] - y_lim[0]) / 200)
    zdr_bins = np.arange(zdr_lim[0], zdr_lim[1] + 0.025, 0.025)

    # Compute 2D histogram
    ds_stats = compute_2d_histogram(
        df,
        x=zdr,
        y="ADP/KDP",
        x_bins=zdr_bins,
        y_bins=y_bins,
    )

    # Colormap & norm
    if cmap is None:
        cmap = plt.get_cmap("Spectral_r").copy()
        cmap.set_under(alpha=0)
    if norm is None:
        norm = LogNorm(1, None)

    # Set default title if not provided
    if title is None:
        title = r"$A_{\mathrm{DP}} / K_{\mathrm{DP}}$ vs $Z_{\mathrm{DR}}$"

    # Create figure if ax is None
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)

    # Plot 2D histogram
    p = ds_stats["count"].plot.pcolormesh(
        x=zdr,
        y="ADP/KDP",
        ax=ax,
        cmap=cmap,
        norm=norm,
        add_colorbar=add_colorbar,
        extend="max",
        cbar_kwargs={"label": "Counts"} if add_colorbar else {},
    )
    ax.set_xlabel(r"$Z_{\mathrm{DR}}$ [dB]")
    ax.set_ylabel(r"$A_{\mathrm{DP}}  /  K_{\mathrm{DP}}$ [dB deg$^{{-1}}$]")
    ax.set_xlim(*zdr_lim)
    ax.set_ylim(*y_lim)
    ax.set_title(title)

    return p


def plot_A_KDP_ZDR(
    df,
    a,
    kdp,
    zdr,
    y_lim=(0, 0.05),
    zdr_lim=(0, 3),
    cmap=None,
    norm=None,
    add_colorbar=True,
    pol="",
    title=None,
    ax=None,
    figsize=(8, 8),
    dpi=300,
):
    """Create a 2D histogram of k/KDP vs ZDR.

    References
    ----------
    Ryzhkov, A., P. Zhang, and J. Hu, 2025.
    Suggested Modifications for the S-Band Polarimetric Radar Rainfall Estimation Algorithm.
    J. Hydrometeor., 26, 1053-1062. https://doi.org/10.1175/JHM-D-25-0014.1.
    """
    # Compute A/KDP
    df["A/KDP"] = df[a] / df[kdp]

    # Bins & limits
    y_bins = np.arange(y_lim[0], y_lim[1], (y_lim[1] - y_lim[0]) / 200)
    x_bins = np.arange(zdr_lim[0], zdr_lim[1] + 0.025, 0.025)

    # Compute 2D histogram
    ds_stats = compute_2d_histogram(
        df,
        x=zdr,
        y="A/KDP",
        x_bins=x_bins,
        y_bins=y_bins,
    )

    # Colormap & norm
    if cmap is None:
        cmap = plt.get_cmap("Spectral_r").copy()
        cmap.set_under(alpha=0)
    if norm is None:
        norm = LogNorm(1, None)

    # Define symbols
    a_symbol = get_symbol_str("A", pol)

    # Set default title if not provided
    if title is None:
        title = rf"${a_symbol} / K_{{\mathrm{{DP}}}}$ vs $Z_{{\mathrm{{DR}}}}$"

    # Create figure if ax is None
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)

    # Plot 2D histogram
    p = ds_stats["count"].plot.pcolormesh(
        x=zdr,
        y="A/KDP",
        ax=ax,
        cmap=cmap,
        norm=norm,
        add_colorbar=add_colorbar,
        extend="max",
        cbar_kwargs={"label": "Counts"} if add_colorbar else {},
    )
    ax.set_xlabel(r"$Z_{\mathrm{DR}}$ [dB]")
    ax.set_ylabel(rf"${a_symbol} / K_{{\mathrm{{DP}}}}$ [dB/deg]")
    ax.set_xlim(*zdr_lim)
    ax.set_ylim(*y_lim)
    ax.set_title(title)

    return p


def plot_KDP_Z_ZDR(
    df,
    kdp,
    z,
    zdr,
    y_lim=None,
    zdr_lim=(0, 5),
    z_linear=True,
    cmap=None,
    norm=None,
    add_colorbar=True,
    title=None,
    ax=None,
    figsize=(8, 8),
    dpi=300,
):
    """Create a 2D histogram of (KDP/Z) vs ZDR with log-scale y-axis (no fit)."""
    # Define y limits and KDP/Z
    if z_linear:
        df["KDP/Z"] = df[kdp] / disdrodb.idecibel(df[z])
        y_lim = (1e-6, 1e-3) if y_lim is None else y_lim
        y_label = r"$K_{\mathrm{DP}} / Z$ [deg km$^{-1}$ / mm$^6$ m$^{-3}$]"

    else:
        df["KDP/Z"] = df[kdp] / df[z]
        y_lim = (1e-5, 1e-1) if y_lim is None else y_lim
        y_label = r"$K_{\mathrm{DP}} / Z$ [deg km$^{-1}$ / dBZ]"

    # Define bins
    y_bins = log_arange(y_lim[0], y_lim[1], log_step=0.025, base=10)
    x_bins = np.arange(zdr_lim[0], zdr_lim[1] + 0.025, 0.025)

    # Compute 2D histogram
    ds_stats = compute_2d_histogram(
        df,
        x=zdr,
        y="KDP/Z",
        x_bins=x_bins,
        y_bins=y_bins,
    )

    # Colormap & norm
    if cmap is None:
        cmap = plt.get_cmap("Spectral_r").copy()
        cmap.set_under(alpha=0)
    if norm is None:
        norm = LogNorm(1, None)

    # Set default title if not provided
    if title is None:
        title = r"$K_{\mathrm{DP}}/Z$ vs $Z_{\mathrm{DR}}$"

    # Create figure if ax is None
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)

    # Plot 2D histogram
    p = ds_stats["count"].plot.pcolormesh(
        x=zdr,
        y="KDP/Z",
        ax=ax,
        cmap=cmap,
        norm=norm,
        add_colorbar=add_colorbar,
        extend="max",
        yscale="log",
        cbar_kwargs={"label": "Counts"} if add_colorbar else {},
    )
    ax.set_xlabel(r"$Z_{\mathrm{DR}}$ [dB]")
    ax.set_ylabel(y_label)
    ax.set_xlim(*zdr_lim)
    ax.set_ylim(*y_lim)
    ax.set_title(title)
    return p


def plot_KED_R(
    df,
    log_r=True,
    log_ked=False,
    add_fit=True,
    cmap=None,
    norm=None,
    add_colorbar=True,
    title=None,
    ax=None,
    legend_fontsize=14,
    figsize=(8, 8),
    dpi=300,
):
    """Create a 2D histogram of KED vs R."""
    if log_r:
        r_bins = log_arange(0.1, 500, log_step=0.05, base=10)
        r_lims = (0.1, 500)
        r_ticks = [0.1, 1, 10, 50, 100, 500]
        xscale = "log"
    else:
        r_bins = np.arange(0, 500, step=2)
        r_lims = (0, 500)
        r_ticks = None
        xscale = "linear"
    if log_ked:
        ked_bins = log_arange(1, 50, log_step=0.025, base=10)
        ked_lims = (1, 50)
        ked_ticks = [1, 10, 50]
        yscale = "log"
    else:
        ked_bins = np.arange(0, 50, step=1)
        ked_lims = (0, 50)
        ked_ticks = None
        yscale = "linear"

    # Compute 2d histogram
    ds_stats = compute_2d_histogram(
        df,
        x="R",
        y="KED",
        x_bins=r_bins,
        y_bins=ked_bins,
    )

    # Define colormap and norm
    if cmap is None:
        cmap = plt.get_cmap("Spectral_r").copy()
        cmap.set_under(alpha=0)
    norm = LogNorm(1, None) if norm is None else norm

    # Set default title if not provided
    if title is None:
        title = "KED vs R"

    # Create figure if ax is None
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)

    # Plot 2D histogram
    p = ds_stats["count"].plot.pcolormesh(
        x="R",
        y="KED",
        ax=ax,
        cmap=cmap,
        norm=norm,
        add_colorbar=add_colorbar,
        extend="max",
        xscale=xscale,
        yscale=yscale,
        cbar_kwargs={"label": "Counts"} if add_colorbar else {},
    )
    ax.set_xlabel(r"$R$ [mm h$^{-1}$]")
    ax.set_ylabel(r"KED [J m$^{-2}$ mm$^{-1}$]")
    ax.set_xlim(*r_lims)
    ax.set_ylim(*ked_lims)
    if r_ticks is not None:
        ax.set_xticks(r_ticks)
        ax.set_xticklabels([str(v) for v in r_ticks])
    if ked_ticks is not None:
        ax.set_yticks(ked_ticks)
        ax.set_yticklabels([str(v) for v in ked_ticks])
    ax.set_title("KED vs R")
    # Fit and plot a powerlaw
    if add_fit:
        # Fit a power law KED = a * R**b
        (a, b), _ = fit_powerlaw(
            x=df["R"],
            y=df["KED"],
            xbins=r_bins,
            x_in_db=False,
        )
        # Invert for R = A * KED**B
        A, B = inverse_powerlaw_parameters(a, b)
        # Define legend string
        a_str = _define_coeff_string(a)
        A_str = _define_coeff_string(A)
        legend_str = rf"$\mathrm{{KED}} = {a_str}\,R^{{{b:.2f}}}$" "\n" rf"$R = {A_str}\,\mathrm{{KED}}^{{{B:.2f}}}$"
        # Get power law predictions
        x_pred = np.arange(r_lims[0], r_lims[1])
        y_pred = predict_from_powerlaw(x_pred, a=a, b=b)
        # Add fitted powerlaw
        ax.plot(x_pred, y_pred, linestyle="dashed", color="black")
        # Add legend
        legend_bbox_dict = {"facecolor": "white", "edgecolor": "black", "alpha": 0.7}
        ax.text(
            0.05,
            0.95,
            legend_str,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=legend_fontsize,
            bbox=legend_bbox_dict,
        )

    return p


def plot_KEF_R(
    df,
    log_r=True,
    log_kef=True,
    add_fit=True,
    cmap=None,
    norm=None,
    add_colorbar=True,
    title=None,
    ax=None,
    legend_fontsize=14,
    figsize=(8, 8),
    dpi=300,
):
    """Create a 2D histogram of KEF vs R."""
    if log_r:
        r_bins = log_arange(0.1, 500, log_step=0.05, base=10)
        r_lims = (0.1, 500)
        r_ticks = [0.1, 1, 10, 50, 100, 500]
        xscale = "log"
    else:
        r_bins = np.arange(0, 500, step=2)
        r_lims = (0, 500)
        r_ticks = None
        xscale = "linear"
    if log_kef:
        kef_bins = log_arange(0.1, 10_000, log_step=0.05, base=10)
        kef_lims = (0.1, 10_000)
        kef_ticks = [0.1, 1, 10, 100, 1000, 10000]
        yscale = "log"
    else:
        kef_bins = np.arange(0, 5000, step=50)
        kef_lims = (0, 5000)
        kef_ticks = None
        yscale = "linear"

    # Compute 2d histogram
    ds_stats = compute_2d_histogram(
        df,
        x="R",
        y="KEF",
        x_bins=r_bins,
        y_bins=kef_bins,
    )

    # Define colormap and norm
    if cmap is None:
        cmap = plt.get_cmap("Spectral_r").copy()
        cmap.set_under(alpha=0)
    norm = LogNorm(1, None) if norm is None else norm

    # Set default title if not provided
    if title is None:
        title = "KEF vs R"

    # Create figure if ax is None
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)

    # Plot 2D histogram
    p = ds_stats["count"].plot.pcolormesh(
        x="R",
        y="KEF",
        ax=ax,
        cmap=cmap,
        norm=norm,
        add_colorbar=add_colorbar,
        extend="max",
        xscale=xscale,
        yscale=yscale,
        cbar_kwargs={"label": "Counts"} if add_colorbar else {},
    )
    ax.set_xlabel(r"$R$ [mm h$^{-1}$]")
    ax.set_ylabel(r"KEF [J m$^{-2}$ h$^{-1}$]")
    ax.set_xlim(*r_lims)
    ax.set_ylim(*kef_lims)
    if r_ticks is not None:
        ax.set_xticks(r_ticks)
        ax.set_xticklabels([str(v) for v in r_ticks])
    if kef_ticks is not None:
        ax.set_yticks(kef_ticks)
        ax.set_yticklabels([str(v) for v in kef_ticks])
    ax.set_title(title)

    # Fit and plot the power law
    # - Alternative fit model: a + I *(1 - b*exp(c*I))  (a is upper limit)
    if add_fit:
        # Fit power law KEF = a * R ** b
        (a, b), _ = fit_powerlaw(
            x=df["R"],
            y=df["KEF"],
            xbins=r_bins,
            x_in_db=False,
        )
        # Invert parameters for R = A * KEF ** B
        A, B = inverse_powerlaw_parameters(a, b)
        # Define legend string
        a_str = _define_coeff_string(a)
        A_str = _define_coeff_string(A)
        legend_str = rf"$\mathrm{{KEF}} = {a_str}\,R^{{{b:.2f}}}$" "\n" rf"$R = {A_str}\,\mathrm{{KEF}}^{{{B:.2f}}}$"
        # Get power law predictions
        x_pred = np.arange(*r_lims)
        kef_pred = predict_from_powerlaw(x_pred, a=a, b=b)
        # Add fitted powerlaw
        ax.plot(x_pred, kef_pred, linestyle="dashed", color="black")
        # Add legend
        legend_bbox_dict = {"facecolor": "white", "edgecolor": "black", "alpha": 0.7}
        ax.text(
            0.05,
            0.95,
            legend_str,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=legend_fontsize,
            bbox=legend_bbox_dict,
        )
    return p


def plot_KEF_Z(
    df,
    z="Z",
    log_kef=True,
    add_fit=True,
    pol="",
    cmap=None,
    norm=None,
    add_colorbar=True,
    title=None,
    ax=None,
    legend_fontsize=14,
    figsize=(8, 8),
    dpi=300,
):
    """Create a 2D histogram of KEF vs Z."""
    # Define limits and bins
    z_lims = (0, 70)
    z_bins = np.arange(*z_lims, step=1)

    if log_kef:
        kef_lims = (0.1, 10_000)
        kef_bins = log_arange(*kef_lims, log_step=0.05, base=10)
        kef_ticks = [0.1, 1, 10, 100, 1000, 10000]
        yscale = "log"
    else:
        kef_lims = (0, 5000)
        kef_bins = np.arange(*kef_lims, step=50)
        kef_ticks = None
        yscale = "linear"

    # Compute 2d histogram
    ds_stats = compute_2d_histogram(
        df,
        x=z,
        y="KEF",
        x_bins=z_bins,
        y_bins=kef_bins,
    )

    # Define colormap and norm
    if cmap is None:
        cmap = plt.get_cmap("Spectral_r").copy()
        cmap.set_under(alpha=0)
    norm = LogNorm(1, None) if norm is None else norm

    # Define symbols
    z_symbol = get_symbol_str("Z", pol)
    z_lower_symbol = get_symbol_str("z", pol)

    # Set default title if not provided
    if title is None:
        title = rf"KEF vs ${z_symbol}$"

    # Create figure if ax is None
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)

    # Plot 2D histogram
    p = ds_stats["count"].plot.pcolormesh(
        x=z,
        y="KEF",
        ax=ax,
        cmap=cmap,
        norm=norm,
        add_colorbar=add_colorbar,
        extend="max",
        yscale=yscale,
        cbar_kwargs={"label": "Counts"} if add_colorbar else {},
    )
    ax.set_xlabel(rf"${z_symbol}$ [dB]")
    ax.set_ylabel(r"KEF [$J$ m$^{-2}$ h$^{-1}$]")
    ax.set_xlim(*z_lims)
    ax.set_ylim(*kef_lims)
    if kef_ticks is not None:
        ax.set_yticks(kef_ticks)
        ax.set_yticklabels([str(v) for v in kef_ticks])
    ax.set_title(title)

    # Fit and plot the powerlaw
    if add_fit:
        # Fit power law KEF = a * Z ** b
        (a, b), _ = fit_powerlaw(
            x=df[z],
            y=df["KEF"],
            xbins=z_bins,
            x_in_db=True,
        )
        # Invert parameters for Z = A * KEF ** B
        A, B = inverse_powerlaw_parameters(a, b)
        # Define legend string
        a_str = _define_coeff_string(a)
        A_str = _define_coeff_string(A)
        legend_str = (
            rf"$\mathrm{{KEF}} = {a_str}\;{z_lower_symbol}^{{{b:.2f}}}$"
            "\n"
            rf"${z_lower_symbol} = {A_str}\;\mathrm{{KEF}}^{{{B:.2f}}}$"
        )
        # Get power law predictions
        x_pred = np.arange(*z_lims)
        x_pred_linear = disdrodb.idecibel(x_pred)
        kef_pred = predict_from_powerlaw(x_pred_linear, a=a, b=b)
        # Add fitted powerlaw
        ax.plot(x_pred, kef_pred, linestyle="dashed", color="black")
        # Add legend
        legend_bbox_dict = {"facecolor": "white", "edgecolor": "black", "alpha": 0.7}
        ax.text(
            0.05,
            0.95,
            legend_str,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=legend_fontsize,
            bbox=legend_bbox_dict,
        )

    return p


def plot_TKE_Z(
    df,
    z="Z",
    log_tke=True,
    add_fit=True,
    cmap=None,
    norm=None,
    add_colorbar=True,
    title=None,
    ax=None,
    legend_fontsize=14,
    figsize=(8, 8),
    dpi=300,
):
    """Create a 2D histogram of TKE vs Z."""
    z_bins = np.arange(0, 70, step=1)
    z_lims = (0, 70)
    if log_tke:
        tke_bins = log_arange(0.01, 500, log_step=0.05, base=10)
        tke_lims = (0.01, 200)
        tke_ticks = [0.01, 0.1, 1, 10, 100, 200]
        yscale = "log"
    else:
        tke_bins = np.arange(0, 200, step=1)
        tke_lims = (0, 200)
        tke_ticks = None
        yscale = "linear"

    # Compute 2d histogram
    ds_stats = compute_2d_histogram(
        df,
        x=z,
        y="TKE",
        x_bins=z_bins,
        y_bins=tke_bins,
    )

    # Define colormap and norm
    if cmap is None:
        cmap = plt.get_cmap("Spectral_r").copy()
        cmap.set_under(alpha=0)
    norm = LogNorm(1, None) if norm is None else norm

    # Set default title if not provided
    if title is None:
        title = "TKE vs Z"

    # Create figure if ax is None
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)

    # Plot 2D histogram
    p = ds_stats["count"].plot.pcolormesh(
        x=z,
        y="TKE",
        ax=ax,
        cmap=cmap,
        norm=norm,
        add_colorbar=add_colorbar,
        extend="max",
        yscale=yscale,
        cbar_kwargs={"label": "Counts"} if add_colorbar else {},
    )
    ax.set_xlabel(r"$Z$ [dB]")
    ax.set_ylabel(r"TKE [$J$ m$^{-2}$]")
    ax.set_xlim(*z_lims)
    ax.set_ylim(*tke_lims)
    if tke_ticks is not None:
        ax.set_yticks(tke_ticks)
        ax.set_yticklabels([str(v) for v in tke_ticks])
    ax.set_title(title)

    # Fit and plot the powerlaw
    if add_fit:
        # Fit power law TKE = a * Z ** b
        (a, b), _ = fit_powerlaw(
            x=df[z],
            y=df["TKE"],
            xbins=z_bins,
            x_in_db=True,
        )
        # Invert parameters for Z = A * KEF ** B
        A, B = inverse_powerlaw_parameters(a, b)
        # Define legend string
        a_str = _define_coeff_string(a)
        A_str = _define_coeff_string(A)
        legend_str = rf"$\mathrm{{TKE}} = {a_str}\;z^{{{b:.2f}}}$" "\n" rf"$z = {A_str}\;\mathrm{{TKE}}^{{{B:.2f}}}$"
        # Get power law predictions
        x_pred = np.arange(*z_lims)
        x_pred_linear = disdrodb.idecibel(x_pred)
        y_pred = predict_from_powerlaw(x_pred_linear, a=a, b=b)
        # Add fitted powerlaw
        ax.plot(x_pred, y_pred, linestyle="dashed", color="black")
        # Add legend
        legend_bbox_dict = {"facecolor": "white", "edgecolor": "black", "alpha": 0.7}
        ax.text(
            0.05,
            0.95,
            legend_str,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=legend_fontsize,
            bbox=legend_bbox_dict,
        )

    return p


####-----------------------------------------------------------------------.
#### Radar and Kinetic Energy Summary figures


def plot_radar_relationships(df, band):
    """Create 3x3 multipanel figure with radar relationships."""
    # Check band
    if band not in {"X", "C", "S"}:
        raise ValueError("Plotting function developed only for bands: 'X', 'C', 'S'.")

    # Define columns
    z = f"DBZH_{band}"
    zdr = f"ZDR_{band}"
    kdp = f"KDP_{band}"
    a = f"AH_{band}"
    adp = f"ADP_{band}"

    # Define limits
    adp_kdp_ylim_dict = {
        "X": (0.0, 0.05),
        "C": (0.0, 0.05),
        "S": (0.0, 0.015),
    }
    adp_kdp_ylim = adp_kdp_ylim_dict[band]

    a_ylim_dict = {
        "S": (0.00001, 1),
        "C": (0.0001, 10),
        "X": (0.0001, 10),
    }
    a_ylim = a_ylim_dict[band]

    # Define plotting settings
    add_colorbar = False
    norm = LogNorm(1, None)
    legend_fontsize = 12

    # Initialize figure
    fig = plt.figure(figsize=(10, 12), dpi=300)  # Slightly taller to accommodate colorbar
    # fig.suptitle(f'C-band Polarimetric Radar Variables Relationships', fontsize=16, y=0.96)

    # Create gridspec with space for colorbar at bottom
    gs = GridSpec(
        4,
        3,
        figure=fig,
        height_ratios=[1, 1, 1, 0.05],
        hspace=0.35,
        wspace=0.35,
        left=0.05,
        right=0.95,
        top=0.93,
        bottom=0.08,
    )

    # Create subplots using gridspec
    axes = []
    for i in range(3):
        for j in range(3):
            ax = fig.add_subplot(gs[i, j])
            axes.append(ax)

    # Flatten axes for easier indexing
    ax = np.array(axes).flatten()

    # - R vs Z_H
    p = plot_R_Z(
        df,
        z=z,
        r="R",
        pol="H",
        norm=norm,
        add_colorbar=add_colorbar,
        legend_fontsize=legend_fontsize,
        ax=ax[0],
    )

    # - Define norm for other plots
    norm = p.norm

    # - R vs K_DP
    plot_R_KDP(
        df,
        kdp=kdp,
        r="R",
        log_scale=True,
        legend_fontsize=legend_fontsize,
        norm=norm,
        add_colorbar=add_colorbar,
        ax=ax[1],
    )

    # - Z_DR vs Z_H
    plot_ZDR_Z(
        df,
        z=z,
        zdr=zdr,
        pol="H",
        legend_fontsize=legend_fontsize,
        norm=norm,
        add_colorbar=add_colorbar,
        ax=ax[2],
    )

    # - A_H vs Z_H
    plot_A_Z(
        df,
        a=a,
        z=z,
        pol="H",
        legend_fontsize=legend_fontsize,
        norm=norm,
        add_colorbar=add_colorbar,
        a_lim=a_ylim,
        ax=ax[3],
    )

    # - A_H vs K_DP
    plot_A_KDP(
        df,
        a=a,
        kdp=kdp,
        pol="H",
        legend_fontsize=legend_fontsize,
        norm=norm,
        add_colorbar=add_colorbar,
        ax=ax[4],
    )
    # plot_A_KDP(df, a=a, kdp=kdp, log_a=True, log_kdp=True,
    #            legend_fontsize=legend_fontsize, norm=norm, add_colorbar=add_colorbar, ax=ax[4]))

    # - A_H vs R
    plot_A_R(df, a=a, r="R", pol="H", legend_fontsize=legend_fontsize, norm=norm, add_colorbar=add_colorbar, ax=ax[5])

    # - K_DP vs Z_H
    plot_KDP_Z(
        df,
        kdp=kdp,
        z=z,
        pol="H",
        legend_fontsize=legend_fontsize,
        norm=norm,
        add_colorbar=add_colorbar,
        log_kdp=True,
        ax=ax[6],
    )

    # - A_DP/K_DP vs Z_DR
    plot_ADP_KDP_ZDR(df, adp=adp, kdp=kdp, zdr=zdr, norm=norm, add_colorbar=add_colorbar, y_lim=adp_kdp_ylim, ax=ax[7])
    # plot_A_KDP_ZDR(df, a=a, kdp=kdp, zdr=zdr, y_lim=(0, 0.3), norm=norm, add_colorbar=add_colorbar)

    # - K_DP/Z vs Z_DR
    p = plot_KDP_Z_ZDR(df, kdp=kdp, z=z, zdr=zdr, norm=norm, add_colorbar=add_colorbar, z_linear=False, ax=ax[8])
    # plot_KDP_Z_ZDR(df, kdp=kdp, z=z, zdr=zdr, norm=norm, add_colorbar=add_colorbar, z_linear=True, ax=ax[8])

    # - Add colorbar
    cax = fig.add_subplot(gs[3, :])  # Spans all columns in the bottom row
    cbar = plt.colorbar(p, cax=cax, orientation="horizontal", extend="max", extendfrac=0.025)
    cbar.ax.set_aspect(0.1)
    cbar.set_label("Counts", fontsize=12, labelpad=6)
    cbar.ax.tick_params(labelsize=12)
    cbar.ax.xaxis.set_label_position("top")
    return fig


def plot_kinetic_energy_relationships(df):
    """Create a 2x2 multipanel figure showing kinetic energy relationships."""
    # Define plotting settings
    add_colorbar = False
    norm = LogNorm(1, None)
    legend_fontsize = 12
    # Initialize figure
    fig = plt.figure(figsize=(9, 10), dpi=300)

    # Create gridspec with space for colorbar at bottom
    gs = GridSpec(
        3,
        2,
        figure=fig,
        height_ratios=[1, 1, 0.05],
        hspace=0.3,
        wspace=0.25,
        left=0.05,
        right=0.95,
        top=0.93,
        bottom=0.08,
    )

    # Create subplots using gridspec
    axes = []
    for i in range(2):
        for j in range(2):
            ax = fig.add_subplot(gs[i, j])
            axes.append(ax)

    # Flatten axes for easier indexing
    ax = np.array(axes).flatten()

    # Plot the specific functions you requested:

    # KED vs R (linear KED)
    p = plot_KED_R(df, norm=norm, legend_fontsize=legend_fontsize, add_colorbar=add_colorbar, ax=ax[0])

    # Define norm for other plots based on first plot
    norm = p.norm

    # KEF vs R
    plot_KEF_R(df, norm=norm, legend_fontsize=legend_fontsize, add_colorbar=add_colorbar, ax=ax[1])

    # KEF vs Z_H
    plot_KEF_Z(df, z="Z", norm=norm, legend_fontsize=legend_fontsize, add_colorbar=add_colorbar, ax=ax[2])

    # TKE vs Z_H
    p_last = plot_TKE_Z(df, z="Z", norm=norm, legend_fontsize=legend_fontsize, add_colorbar=add_colorbar, ax=ax[3])

    # Add colorbar at the bottom
    cax = fig.add_subplot(gs[2, :])  # Spans all columns in the bottom row
    cbar = plt.colorbar(p_last, cax=cax, orientation="horizontal", extend="max", extendfrac=0.025)
    cbar.ax.set_aspect(0.1)
    cbar.set_label("Counts", fontsize=12, labelpad=10)
    cbar.ax.tick_params(labelsize=12)
    cbar.ax.xaxis.set_label_position("top")

    return fig


####-----------------------------------------------------------------------.
#### Summary routine


def define_filename(prefix, extension, data_source, campaign_name, station_name):
    """Define filename for summary files."""
    if extension in ["png", "jpeg"]:
        filename = f"Figure.{prefix}.{data_source}.{campaign_name}.{station_name}.{extension}"
    if extension in ["csv", "parquet", "pdf", "yaml", "yml"]:
        filename = f"Table.{prefix}.{data_source}.{campaign_name}.{station_name}.{extension}"
    if extension in ["nc"]:
        filename = f"Dataset.{prefix}.{data_source}.{campaign_name}.{station_name}.{extension}"
    return filename


def create_l2_dataframe(ds):
    """Create pandas Dataframe for L2 analysis."""
    # - Drop array variables and convert to pandas
    df = ds.drop_dims([DIAMETER_DIMENSION, VELOCITY_DIMENSION]).to_pandas()
    # - Drop coordinates
    coords_to_drop = ["velocity_method", "sample_interval", *RADAR_OPTIONS]
    df = df.drop(columns=coords_to_drop, errors="ignore")
    # - Drop rows with missing rain
    df = df[df["R"] > 0]
    return df


def prepare_summary_dataset(ds, velocity_method="fall_velocity", source="drop_number"):
    """Prepare the L2E or L2M dataset to be converted to a dataframe."""
    # Select fall velocity method
    if "velocity_method" in ds.dims:
        ds = ds.sel(velocity_method=velocity_method)

    # Select first occurrence of radars options (except frequency)
    for dim in RADAR_OPTIONS:
        if dim in ds.dims and dim != "frequency":
            ds = ds.isel({dim: 0})

    # Unstack frequency dimension
    ds = unstack_radar_variables(ds)

    # For kinetic energy variables, select source="drop_number"
    if "source" in ds.dims:
        ds = ds.sel(source=source)

    # Select only timesteps with R > 0
    # - We save R with 2 decimals accuracy ... so 0.01 is the smallest value
    rainy_timesteps = np.logical_and(ds["Rm"].compute() >= 0.01, ds["R"].compute() >= 0.01)
    ds = ds.isel(time=ds["Rm"].compute() >= rainy_timesteps)
    return ds


def generate_station_summary(ds, summary_dir_path, data_source, campaign_name, station_name):
    """Generate station summary using L2E dataset."""
    ####---------------------------------------------------------------------.
    #### Prepare dataset
    ds = prepare_summary_dataset(ds)

    # Ensure all data are in memory
    ds = ds.compute()

    ####---------------------------------------------------------------------.
    #### Create drop spectrum figures and statistics
    # Compute sum of raw and filtered spectrum over time
    raw_drop_number = ds["raw_drop_number"].sum(dim="time")
    drop_number = ds["drop_number"].sum(dim="time")

    # Define theoretical and measured average velocity
    theoretical_average_velocity = ds["fall_velocity"].mean(dim="time")
    measured_average_velocity = get_drop_average_velocity(drop_number)

    # Save raw and filtered spectrum over time & theoretical and measured average fall velocity
    ds_stats = xr.Dataset()
    ds_stats["raw_drop_number"] = raw_drop_number
    ds_stats["drop_number"] = raw_drop_number
    ds_stats["theoretical_average_velocity"] = theoretical_average_velocity
    ds_stats["measured_average_velocity"] = measured_average_velocity
    filename = define_filename(
        prefix="SpectrumStats",
        extension="nc",
        data_source=data_source,
        campaign_name=campaign_name,
        station_name=station_name,
    )
    ds_stats.to_netcdf(os.path.join(summary_dir_path, filename))

    # Create figures with raw and filtered spectrum
    # - Raw
    filename = define_filename(
        prefix="SpectrumRaw",
        extension="png",
        data_source=data_source,
        campaign_name=campaign_name,
        station_name=station_name,
    )
    p = plot_drop_spectrum(raw_drop_number, title="Raw Drop Spectrum")
    p.figure.savefig(os.path.join(summary_dir_path, filename))
    plt.close()

    # - Filtered
    filename = define_filename(
        prefix="SpectrumFiltered",
        extension="png",
        data_source=data_source,
        campaign_name=campaign_name,
        station_name=station_name,
    )
    p = plot_drop_spectrum(drop_number, title="Filtered Drop Spectrum")
    p.figure.savefig(os.path.join(summary_dir_path, filename))
    plt.close()

    # Create figure comparing raw and filtered spectrum
    filename = define_filename(
        prefix="SpectrumSummary",
        extension="png",
        data_source=data_source,
        campaign_name=campaign_name,
        station_name=station_name,
    )

    fig = plot_raw_and_filtered_spectrums(
        raw_drop_number=raw_drop_number,
        drop_number=drop_number,
        theoretical_average_velocity=theoretical_average_velocity,
        measured_average_velocity=measured_average_velocity,
    )
    fig.savefig(os.path.join(summary_dir_path, filename))
    plt.close()

    ####---------------------------------------------------------------------.
    #### Create L2E 1MIN dataframe
    df = create_l2_dataframe(ds)

    # Define diameter bin edges
    diameter_bin_edges = get_diameter_bin_edges(ds)

    # ---------------------------------------------------------------------.
    #### Save L2E 1MIN Parquet
    l2e_parquet_filename = f"L2E.1MIN.PARQUET.{data_source}.{campaign_name}.{station_name}.parquet"
    l2e_parquet_filepath = os.path.join(summary_dir_path, l2e_parquet_filename)
    df.to_parquet(l2e_parquet_filepath, engine="pyarrow", compression="snappy")

    #### ---------------------------------------------------------------------.
    #### Create table with rain summary
    table_rain_summary = create_table_rain_summary(df)
    table_rain_summary_filename = f"Station_Summary.{data_source}.{campaign_name}.{station_name}.yaml"
    table_rain_summary_filepath = os.path.join(summary_dir_path, table_rain_summary_filename)
    write_yaml(table_rain_summary, filepath=table_rain_summary_filepath)

    # ---------------------------------------------------------------------.
    #### Creata table with events summary
    table_events_summary = create_table_events_summary(df)
    # - Save table as csv
    table_events_summary_csv_filename = f"Events_Summary.{data_source}.{campaign_name}.{station_name}.csv"
    table_events_summary_csv_filepath = os.path.join(summary_dir_path, table_events_summary_csv_filename)
    table_events_summary.to_csv(table_events_summary_csv_filepath)
    # - Save table as pdf
    if is_latex_engine_available():
        table_events_summary_pdf_filename = f"Events_Summary.{data_source}.{campaign_name}.{station_name}.pdf"
        table_events_summary_pdf_filepath = os.path.join(summary_dir_path, table_events_summary_pdf_filename)
        save_table_to_pdf(
            df=prepare_latex_table_events_summary(table_events_summary),
            filepath=table_events_summary_pdf_filepath,
            index=True,
            caption="Events Summary",
            orientation="landscape",
        )

    # ---------------------------------------------------------------------.
    #### Create table with integral DSD parameters statistics
    table_dsd_summary = create_table_dsd_summary(df)
    # - Save table as csv
    table_dsd_summary_csv_filename = f"DSD_Summary.{data_source}.{campaign_name}.{station_name}.csv"
    table_dsd_summary_csv_filepath = os.path.join(summary_dir_path, table_dsd_summary_csv_filename)
    table_dsd_summary.to_csv(table_dsd_summary_csv_filepath)
    # - Save table as pdf
    if is_latex_engine_available():
        table_dsd_summary_pdf_filename = f"DSD_Summary.{data_source}.{campaign_name}.{station_name}.pdf"
        table_dsd_summary_pdf_filepath = os.path.join(summary_dir_path, table_dsd_summary_pdf_filename)
        save_table_to_pdf(
            df=prepare_latex_table_dsd_summary(table_dsd_summary),
            index=True,
            filepath=table_dsd_summary_pdf_filepath,
            caption="DSD Summary",
            orientation="portrait",  # "landscape",
        )

    #### ---------------------------------------------------------------------.
    #### Create L2E RADAR Summary Plots
    # Summary plots at X, C, S bands
    if "DBZH_X" in df:
        filename = define_filename(
            prefix="Radar_Band_X",
            extension="png",
            data_source=data_source,
            campaign_name=campaign_name,
            station_name=station_name,
        )
        fig = plot_radar_relationships(df, band="X")
        fig.savefig(os.path.join(summary_dir_path, filename))
    if "DBZH_C" in df:
        filename = define_filename(
            prefix="Radar_Band_C",
            extension="png",
            data_source=data_source,
            campaign_name=campaign_name,
            station_name=station_name,
        )
        fig = plot_radar_relationships(df, band="C")
        fig.savefig(os.path.join(summary_dir_path, filename))
    if "DBZH_S" in df:
        filename = define_filename(
            prefix="Radar_Band_S",
            extension="png",
            data_source=data_source,
            campaign_name=campaign_name,
            station_name=station_name,
        )
        fig = plot_radar_relationships(df, band="S")
        fig.savefig(os.path.join(summary_dir_path, filename))

    # ---------------------------------------------------------------------.
    #### - Create Z-R figure
    filename = define_filename(
        prefix="Z-R",
        extension="png",
        data_source=data_source,
        campaign_name=campaign_name,
        station_name=station_name,
    )

    p = plot_R_Z(df, z="Z", r="R", title=r"$Z$ vs $R$")
    p.figure.savefig(os.path.join(summary_dir_path, filename))
    plt.close()

    #### ---------------------------------------------------------------------.
    #### Create L2E Kinetic Energy Summary Plots
    filename = define_filename(
        prefix="KineticEnergy",
        extension="png",
        data_source=data_source,
        campaign_name=campaign_name,
        station_name=station_name,
    )
    fig = plot_kinetic_energy_relationships(df)
    fig.savefig(os.path.join(summary_dir_path, filename))

    #### ---------------------------------------------------------------------.
    #### Create L2E DSD Parameters summary plots
    #### - Create DSD parameters density figures with LWC
    filename = define_filename(
        prefix="DSD_Params_Density_with_LWC_LinearDm_MaxNormalized",
        extension="png",
        data_source=data_source,
        campaign_name=campaign_name,
        station_name=station_name,
    )
    fig = plot_dsd_params_density(df, log_dm=False, lwc=True, log_normalize=False)
    fig.savefig(os.path.join(summary_dir_path, filename))
    plt.close()

    filename = define_filename(
        prefix="DSD_Params_Density_with_LWC_LogDm_MaxNormalized",
        extension="png",
        data_source=data_source,
        campaign_name=campaign_name,
        station_name=station_name,
    )
    fig = plot_dsd_params_density(df, log_dm=True, lwc=True, log_normalize=False)
    fig.savefig(os.path.join(summary_dir_path, filename))
    plt.close()

    filename = define_filename(
        prefix="DSD_Params_Density_with_LWC_LinearDm_LogNormalized",
        extension="png",
        data_source=data_source,
        campaign_name=campaign_name,
        station_name=station_name,
    )
    fig = plot_dsd_params_density(df, log_dm=False, lwc=True, log_normalize=True)
    fig.savefig(os.path.join(summary_dir_path, filename))
    plt.close()

    filename = define_filename(
        prefix="DSD_Params_Density_with_LWC_LogDm_LogNormalized",
        extension="png",
        data_source=data_source,
        campaign_name=campaign_name,
        station_name=station_name,
    )
    fig = plot_dsd_params_density(df, log_dm=True, lwc=True, log_normalize=True)
    fig.savefig(os.path.join(summary_dir_path, filename))
    plt.close()

    ###------------------------------------------------------------------------.
    #### - Create DSD parameters density figures with R
    filename = define_filename(
        prefix="DSD_Params_Density_with_R_LinearDm_MaxNormalized",
        extension="png",
        data_source=data_source,
        campaign_name=campaign_name,
        station_name=station_name,
    )
    fig = plot_dsd_params_density(df, log_dm=False, lwc=False, log_normalize=False)
    fig.savefig(os.path.join(summary_dir_path, filename))
    plt.close()

    filename = define_filename(
        prefix="DSD_Params_Density_with_R_LogDm_MaxNormalized",
        extension="png",
        data_source=data_source,
        campaign_name=campaign_name,
        station_name=station_name,
    )
    fig = plot_dsd_params_density(df, log_dm=True, lwc=False, log_normalize=False)
    fig.savefig(os.path.join(summary_dir_path, filename))
    plt.close()

    filename = define_filename(
        prefix="DSD_Params_Density_with_R_LinearDm_LogNormalized",
        extension="png",
        data_source=data_source,
        campaign_name=campaign_name,
        station_name=station_name,
    )
    fig = plot_dsd_params_density(df, log_dm=False, lwc=False, log_normalize=True)
    fig.savefig(os.path.join(summary_dir_path, filename))
    plt.close()

    filename = define_filename(
        prefix="DSD_Params_Density_with_R_LogDm_LogNormalized",
        extension="png",
        data_source=data_source,
        campaign_name=campaign_name,
        station_name=station_name,
    )
    fig = plot_dsd_params_density(df, log_dm=True, lwc=False, log_normalize=True)
    fig.savefig(os.path.join(summary_dir_path, filename))
    plt.close()

    ###------------------------------------------------------------------------.
    #### - Create DSD parameters relationship figures
    filename = define_filename(
        prefix="DSD_Params_Relations",
        extension="png",
        data_source=data_source,
        campaign_name=campaign_name,
        station_name=station_name,
    )
    fig = plot_dsd_params_relationships(df, add_nt=True)
    fig.savefig(os.path.join(summary_dir_path, filename))
    plt.close()

    ###------------------------------------------------------------------------.
    #### - Create Dmax relationship figures
    filename = define_filename(
        prefix="DSD_Dmax_Relations",
        extension="png",
        data_source=data_source,
        campaign_name=campaign_name,
        station_name=station_name,
    )
    fig = plot_dmax_relationships(df, diameter_bin_edges=diameter_bin_edges, dmax="Dmax", diameter_max=10)
    fig.savefig(os.path.join(summary_dir_path, filename))
    plt.close()

    #### ---------------------------------------------------------------------.
    #### Create L2E QC summary plots
    # TODO:

    ####------------------------------------------------------------------------.
    #### Create N(D) densities
    df_nd = create_nd_dataframe(ds)

    #### - Plot N(D) vs D
    filename = define_filename(
        prefix="N(D)",
        extension="png",
        data_source=data_source,
        campaign_name=campaign_name,
        station_name=station_name,
    )
    p = plot_dsd_density(df_nd, diameter_bin_edges=diameter_bin_edges)
    p.figure.savefig(os.path.join(summary_dir_path, filename))
    plt.close()

    #### - Plot N(D) vs D with dense lines
    filename = define_filename(
        prefix="N(D)_DenseLines",
        extension="png",
        data_source=data_source,
        campaign_name=campaign_name,
        station_name=station_name,
    )
    p = plot_dsd_with_dense_lines(ds)
    p.figure.savefig(os.path.join(summary_dir_path, filename))
    plt.close()

    #### - Plot N(D)/Nw vs D/Dm
    filename = define_filename(
        prefix="N(D)_Normalized",
        extension="png",
        data_source=data_source,
        campaign_name=campaign_name,
        station_name=station_name,
    )
    p = plot_normalized_dsd_density(df_nd)
    p.figure.savefig(os.path.join(summary_dir_path, filename))
    plt.close()


####------------------------------------------------------------------------.
#### Wrappers


def create_station_summary(data_source, campaign_name, station_name, parallel=False, data_archive_dir=None):
    """Create summary figures and tables for a disdrometer station."""
    # Print processing info
    print(f"Creation of station summary for {data_source} {campaign_name} {station_name} has started.")

    # Define station summary directory
    summary_dir_path = define_station_dir(
        product="SUMMARY",
        data_source=data_source,
        campaign_name=campaign_name,
        station_name=station_name,
        data_archive_dir=data_archive_dir,
        check_exists=False,
    )
    os.makedirs(summary_dir_path, exist_ok=True)

    # Load L2E 1MIN dataset
    ds = disdrodb.open_dataset(
        data_archive_dir=data_archive_dir,
        data_source=data_source,
        campaign_name=campaign_name,
        station_name=station_name,
        product="L2E",
        product_kwargs={"rolling": False, "sample_interval": 60},
        parallel=parallel,
        chunks=-1,
    )

    # Generate station summary figures and table
    generate_station_summary(
        ds=ds,
        summary_dir_path=summary_dir_path,
        data_source=data_source,
        campaign_name=campaign_name,
        station_name=station_name,
    )

    print(f"Creation of station summary for {data_source} {campaign_name} {station_name} has terminated.")

    # -------------------------------------------------------------------------------------------------.
