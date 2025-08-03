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
"""Utilities to create summary statistics."""
import os
import subprocess
import tempfile
from shutil import which

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.colors import LogNorm, Normalize
from matplotlib.gridspec import GridSpec

import disdrodb
from disdrodb import DIAMETER_DIMENSION, VELOCITY_DIMENSION
from disdrodb.l2.empirical_dsd import get_drop_average_velocity
from disdrodb.l2.event import group_timesteps_into_event
from disdrodb.scattering import RADAR_OPTIONS, available_radar_bands
from disdrodb.utils.dataframe import compute_2d_histogram, log_arange
from disdrodb.utils.manipulations import get_diameter_bin_edges, unstack_radar_variables
from disdrodb.utils.yaml import write_yaml


def is_latex_engine_available() -> bool:
    """
    Determine whether the Tectonic TeX/LaTeX engine is installed and accessible.

    Returns
    -------
    bool
        True if tectonic is found, False otherwise.
    """
    return which("tectonic") is not None


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


def plot_drop_spectrum(drop_number, title="Drop Spectrum"):
    """Plot the drop spectrum."""
    cmap = plt.get_cmap("Spectral_r")
    cmap.set_under("none")
    norm = LogNorm(vmin=1, vmax=None)

    p = drop_number.plot.pcolormesh(
        x=DIAMETER_DIMENSION,
        y=VELOCITY_DIMENSION,
        cmap=cmap,
        extend="max",
        norm=norm,
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
    norm=True,
    figsize=(8, 4),
    dpi=300,
):
    """Plot raw and filtered drop spectrum."""
    # Drop number matrix
    cmap = plt.get_cmap("Spectral_r")
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


def plot_normalized_dsd_density(df_nd, figsize=(8, 8), dpi=300):
    """Plot normalized DSD N(D)/Nw ~ D/Dm density."""
    df_nd["D/Dm"] = df_nd["D"] / df_nd["Dm"]
    df_nd["N(D)/Nw"] = df_nd["N(D)"] / df_nd["Nw"]
    ds_stats = compute_2d_histogram(
        df_nd,
        x="D/Dm",
        y="N(D)/Nw",
        x_bins=np.arange(0, 4, 0.025),
        y_bins=log_arange(1e-5, 50, log_step=0.1, base=10),
    )
    cmap = plt.get_cmap("Spectral_r")
    cmap.set_under(alpha=0)
    norm = LogNorm(1, None)

    ds_stats = ds_stats.isel({"N(D)/Nw": ds_stats["N(D)/Nw"] > 0})

    fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)
    _ = ds_stats["count"].plot.pcolormesh(
        x="D/Dm",
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
    ax.set_xlabel(r"$D/D_m$ [-]")
    ax.set_ylabel(r"$N(D)/N_w$ [-]")
    ax.set_title("Normalized DSD")
    return fig


def plot_dsd_density(df_nd, diameter_bin_edges, figsize=(8, 8), dpi=300):
    """Plot N(D) ~ D density."""
    ds_stats = compute_2d_histogram(
        df_nd,
        x="D",
        y="N(D)",
        x_bins=diameter_bin_edges,
        y_bins=log_arange(0.1, 20_000, log_step=0.1, base=10),
    )
    cmap = plt.get_cmap("Spectral_r")
    cmap.set_under(alpha=0)
    norm = LogNorm(1, None)
    fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)
    _ = ds_stats["count"].plot.pcolormesh(x="D", y="N(D)", ax=ax, cmap=cmap, norm=norm, extend="max", yscale="log")
    ax.set_xlim(0, 8)
    ax.set_ylim(1, 20_000)
    ax.set_xlabel(r"$D$ [mm]")
    ax.set_ylabel(r"$N(D)$ [m$^{-3}$ mm$^{-1}$]")
    ax.set_title("DSD")
    return fig


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
    cmap = plt.get_cmap("Spectral_r")
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
    bbox_dict = {
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
            bbox=bbox_dict,
        )
        ax.set_title("")

    return fig


def plot_k_R(df, var, title=r"$R$ vs $k$", figsize=(8, 8), dpi=300):
    """Create a 2D histogram of k vs R."""
    # Define kmin and kmax
    k_min = 0.001
    k_max = 10

    # Compute 2d histogram
    ds_stats = compute_2d_histogram(
        df,
        x="Rm",
        y=var,
        y_bins=log_arange(k_min, k_max, log_step=0.025, base=10),
        x_bins=log_arange(0.1, 500, log_step=0.025, base=10),
    )
    # Define plot options
    cmap = plt.get_cmap("Spectral_r")
    cmap.set_under(alpha=0)
    norm = LogNorm(1, None)
    r_ticks = [0.1, 1, 10, 50, 100, 500]
    k_ticks = [0.001, 0.01, 0.1, 0.5, 1, 5]  # Adapt on k_max

    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)
    _ = ds_stats["count"].plot.pcolormesh(
        x="Rm",
        y=var,
        ax=ax,
        cmap=cmap,
        norm=norm,
        extend="max",
        xscale="log",
        yscale="log",
        cbar_kwargs={"label": "Counts"},
    )
    ax.set_xlabel(r"$R$ [mm h$^{-1}$]")
    ax.set_ylabel(r"k [dB/km]")
    ax.set_ylim(k_min, k_max)
    ax.set_xlim(0.1, 500)
    ax.set_xticks(r_ticks)
    ax.set_xticklabels([str(v) for v in r_ticks])
    ax.set_yticks(k_ticks)
    ax.set_yticklabels([str(v) for v in k_ticks])
    ax.set_title(title)
    return fig


def plot_Z_R(df, figsize=(8, 8), dpi=300, z="Z", r="R", title=r"$Z$ vs $R$"):
    """Create a 2D histogram of Z vs R."""
    # Compute 2d histogram
    ds_stats = compute_2d_histogram(
        df,
        x=z,
        y=r,
        x_bins=np.arange(0, 70, 0.5),
        y_bins=log_arange(0.1, 500, log_step=0.05, base=10),
    )
    # Define plot options
    cmap = plt.get_cmap("Spectral_r")
    cmap.set_under(alpha=0)
    norm = LogNorm(1, None)
    r_ticks = [0.1, 1, 10, 50, 100, 500]

    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)
    _ = ds_stats["count"].plot.pcolormesh(
        x=z,
        y=r,
        ax=ax,
        cmap=cmap,
        norm=norm,
        extend="max",
        yscale="log",
        cbar_kwargs={"label": "Counts"},
    )
    ax.set_ylabel(r"$R$ [mm h$^{-1}$]")
    ax.set_xlabel(r"Z [dB]")
    ax.set_xlim(0, 70)
    ax.set_ylim(0.1, 500)
    ax.set_yticks(r_ticks)
    ax.set_yticklabels([str(v) for v in r_ticks])
    ax.set_title(title)
    return fig


def plot_KED_R(df, log_r=True, log_ked=False, figsize=(8, 8), dpi=300):
    """Create a 2D histogram of KED vs R."""
    # KED J mm-1 m-2 vs R
    # TODO: Fit Model:
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
        ked_bins = log_arange(0.1, 50, log_step=0.05, base=10)
        ked_lims = (0.1, 50)
        ked_ticks = [0.1, 1, 10, 50]
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

    # Define plot options
    cmap = plt.get_cmap("Spectral_r")
    cmap.set_under(alpha=0)
    norm = LogNorm(0.5, None)

    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)
    _ = ds_stats["count"].plot.pcolormesh(
        x="R",
        y="KED",
        ax=ax,
        cmap=cmap,
        norm=norm,
        extend="max",
        xscale=xscale,
        yscale=yscale,
        cbar_kwargs={"label": "Counts"},
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
    return fig


def plot_KEF_R(df, log_r=True, log_kef=True, figsize=(8, 8), dpi=300):
    """Create a 2D histogram of KEF vs R."""
    # KEF J m-2 h-1 vs R
    # TODO: Fit Model: a + I *(1 - b*exp(c*I))  (a is upper limit)
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

    # Define plot options
    cmap = plt.get_cmap("Spectral_r")
    cmap.set_under(alpha=0)
    norm = LogNorm(0.5, None)

    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)
    _ = ds_stats["count"].plot.pcolormesh(
        x="R",
        y="KEF",
        ax=ax,
        cmap=cmap,
        norm=norm,
        extend="max",
        xscale=xscale,
        yscale=yscale,
        cbar_kwargs={"label": "Counts"},
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
    ax.set_title("KEF vs R")
    return fig


def plot_KEF_Z(df, figsize=(8, 8), dpi=300, z="Z", log_kef=True):
    """Create a 2D histogram of KEF vs Z."""
    # KEF J m-2 h-1 vs Z
    # TODO: Fit relationship
    z_bins = np.arange(0, 70, step=1)
    z_lims = (0, 70)
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
        x=z,
        y="KEF",
        x_bins=z_bins,
        y_bins=kef_bins,
    )

    # Define plot options
    cmap = plt.get_cmap("Spectral_r")
    cmap.set_under(alpha=0)
    norm = LogNorm(0.5, None)

    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)
    _ = ds_stats["count"].plot.pcolormesh(
        x=z,
        y="KEF",
        ax=ax,
        cmap=cmap,
        norm=norm,
        extend="max",
        yscale=yscale,
        cbar_kwargs={"label": "Counts"},
    )
    ax.set_xlabel(r"$Z$ [dB]")
    ax.set_ylabel(r"KEF [$J$ m$^{-2}$ h$^{-1}$]")
    ax.set_xlim(*z_lims)
    ax.set_ylim(*kef_lims)
    if kef_ticks is not None:
        ax.set_yticks(kef_ticks)
        ax.set_yticklabels([str(v) for v in kef_ticks])
    ax.set_title("KEF - Z")
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


def create_l2e_dataframe(ds):
    """Create pandas Dataframe for L2E analysis."""
    # - Drop array variables and convert to pandas
    df = ds.drop_dims([DIAMETER_DIMENSION, VELOCITY_DIMENSION]).to_pandas()
    # - Drop coordinates
    coords_to_drop = ["velocity_method", "sample_interval", *RADAR_OPTIONS]
    df = df.drop(columns=coords_to_drop, errors="ignore")
    # - Drop rows with missing rain
    df = df[df["R"] > 0]
    return df


def create_nd_dataframe(ds):
    """Create pandas Dataframe with N(D) data."""
    # Retrieve stacked N(D) dataframe
    ds_stack = ds[["drop_number_concentration", "Nw", "diameter_bin_center", "Dm", "R"]].stack(
        dim={"obs": ["time", "diameter_bin_center"]},
    )
    # - Drop coordinates
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
    return df_nd


def create_station_summary(data_source, campaign_name, station_name, data_archive_dir=None):
    """Create summary figures and tables for a disdrometer station."""
    # Define station summary directory
    summary_dir_path = "/tmp/"

    # ---------------------------------------------------------------------.
    #### Load L2E 1MIN dataset
    ds = disdrodb.open_dataset(
        data_archive_dir=data_archive_dir,
        data_source=data_source,
        campaign_name=campaign_name,
        station_name=station_name,
        product="L2E",
        product_kwargs={"rolling": False, "sample_interval": 60},
        parallel=True,
        chunks=-1,
    )
    # Select fall velocity method
    if "velocity_method" in ds.dims:
        ds = ds.sel(velocity_method="fall_velocity")

    # Select first occurrence of radars options (except frequency)
    for dim in RADAR_OPTIONS:
        if dim in ds.dims and dim != "frequency":
            ds = ds.isel({dim: 0})

    # Unstack frequency dimension
    ds = unstack_radar_variables(ds)

    # For kinetic energy variables, select source="drop_number"
    if "source" in ds.dims:
        ds = ds.sel(source="drop_number")

    # Put all data into memory
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
        prefix="RawSpectrum",
        extension="png",
        data_source=data_source,
        campaign_name=campaign_name,
        station_name=station_name,
    )
    p = plot_drop_spectrum(raw_drop_number, title="Raw Drop Spectrum")
    p.figure.savefig(os.path.join(summary_dir_path, filename))

    # - Filtered
    filename = define_filename(
        prefix="FilteredSpectrum",
        extension="png",
        data_source=data_source,
        campaign_name=campaign_name,
        station_name=station_name,
    )
    p = plot_drop_spectrum(drop_number, title="Filtered Drop Spectrum")
    p.figure.savefig(os.path.join(summary_dir_path, filename))

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

    ####---------------------------------------------------------------------.
    #### Create L2E 1MIN dataframe
    df = create_l2e_dataframe(ds)

    # ---------------------------------------------------------------------.
    #### Save L2E 1MIN Parquet
    l2e_parquet_filename = f"L2E.1MIN.PARQUET.{data_source}.{campaign_name}.{station_name}.parquet"
    l2e_parquet_filepath = os.path.join(summary_dir_path, l2e_parquet_filename)
    df.to_parquet(l2e_parquet_filepath, engine="pyarrow", compression="snappy")

    # ---------------------------------------------------------------------.
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

    # ---------------------------------------------------------------------.
    #### Create L2E DSD Parameters summary plots
    #### - Create Z-R figure
    filename = define_filename(
        prefix="Z-R",
        extension="png",
        data_source=data_source,
        campaign_name=campaign_name,
        station_name=station_name,
    )

    fig = plot_Z_R(df, z="Z", r="R", title=r"$Z$ vs $R$")
    fig.savefig(os.path.join(summary_dir_path, filename))
    ###------------------------------------------------------------------------.
    #### - Create R-KED figure
    filename = define_filename(
        prefix="R-KED",
        extension="png",
        data_source=data_source,
        campaign_name=campaign_name,
        station_name=station_name,
    )
    fig = plot_KED_R(df, log_r=True, log_ked=False)
    fig.savefig(os.path.join(summary_dir_path, filename))

    ###------------------------------------------------------------------------.
    #### - Create R-KEF figure
    filename = define_filename(
        prefix="R-KEF",
        extension="png",
        data_source=data_source,
        campaign_name=campaign_name,
        station_name=station_name,
    )
    fig = plot_KEF_R(df, log_r=True, log_kef=True)
    fig.savefig(os.path.join(summary_dir_path, filename))

    ###------------------------------------------------------------------------.
    #### - Create Z-KEF figure
    filename = define_filename(
        prefix="Z-KEF",
        extension="png",
        data_source=data_source,
        campaign_name=campaign_name,
        station_name=station_name,
    )
    fig = plot_KEF_Z(df, log_kef=True)
    fig.savefig(os.path.join(summary_dir_path, filename))

    ###------------------------------------------------------------------------.
    #### - Create R-AH figures
    radar_bands = available_radar_bands()
    for radar_band in radar_bands:
        var = f"AH_{radar_band}"
        if var in df:
            filename = define_filename(
                prefix=f"R-k_Band_{radar_band}",
                extension="png",
                data_source=data_source,
                campaign_name=campaign_name,
                station_name=station_name,
            )
            fig = plot_k_R(df, var=var, title=f"R vs k (at {radar_band} band)", figsize=(8, 8), dpi=300)
            fig.savefig(os.path.join(summary_dir_path, filename))

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

    filename = define_filename(
        prefix="DSD_Params_Density_with_LWC_LogDm_MaxNormalized",
        extension="png",
        data_source=data_source,
        campaign_name=campaign_name,
        station_name=station_name,
    )
    fig = plot_dsd_params_density(df, log_dm=True, lwc=True, log_normalize=False)
    fig.savefig(os.path.join(summary_dir_path, filename))

    filename = define_filename(
        prefix="DSD_Params_Density_with_LWC_LinearDm_LogNormalized",
        extension="png",
        data_source=data_source,
        campaign_name=campaign_name,
        station_name=station_name,
    )
    fig = plot_dsd_params_density(df, log_dm=False, lwc=True, log_normalize=True)
    fig.savefig(os.path.join(summary_dir_path, filename))

    filename = define_filename(
        prefix="DSD_Params_Density_with_LWC_LogDm_LogNormalized",
        extension="png",
        data_source=data_source,
        campaign_name=campaign_name,
        station_name=station_name,
    )
    fig = plot_dsd_params_density(df, log_dm=True, lwc=True, log_normalize=True)
    fig.savefig(os.path.join(summary_dir_path, filename))

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

    filename = define_filename(
        prefix="DSD_Params_Density_with_R_LogDm_MaxNormalized",
        extension="png",
        data_source=data_source,
        campaign_name=campaign_name,
        station_name=station_name,
    )
    fig = plot_dsd_params_density(df, log_dm=True, lwc=False, log_normalize=False)
    fig.savefig(os.path.join(summary_dir_path, filename))

    filename = define_filename(
        prefix="DSD_Params_Density_with_R_LinearDm_LogNormalized",
        extension="png",
        data_source=data_source,
        campaign_name=campaign_name,
        station_name=station_name,
    )
    fig = plot_dsd_params_density(df, log_dm=False, lwc=False, log_normalize=True)
    fig.savefig(os.path.join(summary_dir_path, filename))

    filename = define_filename(
        prefix="DSD_Params_Density_with_R_LogDm_LogNormalized",
        extension="png",
        data_source=data_source,
        campaign_name=campaign_name,
        station_name=station_name,
    )
    fig = plot_dsd_params_density(df, log_dm=True, lwc=False, log_normalize=True)
    fig.savefig(os.path.join(summary_dir_path, filename))

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

    # ---------------------------------------------------------------------.
    #### Create L2E QC summary plots
    # TODO:

    ####------------------------------------------------------------------------.
    #### Create N(D) densities
    df_nd = create_nd_dataframe(ds)

    #### - Plot N(D)/Nw vs D/Dm
    filename = define_filename(
        prefix="N(D)_Normalized",
        extension="png",
        data_source=data_source,
        campaign_name=campaign_name,
        station_name=station_name,
    )
    fig = plot_normalized_dsd_density(df_nd)
    fig.savefig(os.path.join(summary_dir_path, filename))

    #### - Plot N(D) vs D
    filename = define_filename(
        prefix="N(D)",
        extension="png",
        data_source=data_source,
        campaign_name=campaign_name,
        station_name=station_name,
    )
    diameter_bin_edges = get_diameter_bin_edges(ds)
    plot_dsd_density(df_nd, diameter_bin_edges=diameter_bin_edges)
    fig.savefig(os.path.join(summary_dir_path, filename))

    # ---------------------------------------------------------------------.
