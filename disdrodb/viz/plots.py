# -----------------------------------------------------------------------------.
# Copyright (c) 2021-2026 DISDRODB developers
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
"""DISDRODB Plotting Tools."""

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psutil
import xarray as xr
from matplotlib.colors import BoundaryNorm, ListedColormap, LogNorm, Normalize
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator

from disdrodb.constants import DIAMETER_DIMENSION, VELOCITY_DIMENSION
from disdrodb.l2.empirical_dsd import get_drop_average_velocity
from disdrodb.l2.processing import get_mask_contour, get_spectrum_mask_boundary

# TODO FIX: XARRAY PCOLORMESH IS CURRENTLY INACCURATE

# IMPROVEMENTS
# plot_filtering_boundary()
# plot_spectrum(add theoreticall_fall_velocity,
#               add_tolerance_fraction=0.5,
#               add_tolerance_=0.5)
# Add plot_raw_and_filtered_spectra(animation=True, legend_variables on first axes)

# TODO: plot_l0_quicklook
## - plot_l0_quicklook   # remap weather codes to hydrometeor_type, find R if available


####-------------------------------------------------------------------------------------------------------


def get_precipitation_legend_style(name):
    """Return (colors_dict, labels_dict) for supported categorical variables.

    Supported:
        - rain_type
        - precipitation_type
        - hydrometeor_type
    """
    if name == "rain_type":
        colors = {
            0: "white",
            1: "dodgerblue",  # stratiform
            2: "orangered",  # convective
        }
        labels = {
            0: "No precipitation",
            1: "Stratiform",
            2: "Convective",
        }

    elif name == "precipitation_type":
        colors = {
            -2: "lightgray",  # undefined
            -1: "white",  # no_precip
            0: "#0050b5",  # rainfall (deep blue)
            1: "#00bcd4",  # snowfall (cyan)
            2: "#8e44ad",  # mixed_phase (purple)
        }
        labels = {
            -2: "Undefined",
            -1: "No precipitation",
            0: "Rainfall",
            1: "Snowfall",
            2: "Mixed phase",
        }

    elif name == "hydrometeor_type":
        colors = {
            -2: "white",  # no_hydrometeor
            -1: "lightgray",  # undefined
            0: "white",  # no_precipitation
            # 🌧 Liquid family (blue gradient)
            1: "#4a90e2",  # drizzle (light blue)
            2: "#0057d9",  # drizzle + rain (strong blue)
            3: "#002f8a",  # rain (deep royal blue)
            # 1: "#6baed6",   # drizzle (light blue)
            # 2: "#2171b5",   # drizzle + rain (medium blue)
            # 3: "#08306b",   # rain (deep navy blue)
            # 🟣 Mixed
            4: "#9b59b6",  # mixed (purple)
            # ❄️ Frozen family (teal / aqua gradient — shifted hue)
            5: "#1abc6c",  # snow (green-teal)
            6: "#9be7c4",  # snow grains (pale mint)
            # 5: "#1abc9c",   # snow (teal)
            # 6: "#76eec6",   # snow grains (light aqua)
            # 🟠 Ice family
            7: "#f39c12",  # ice pellets
            8: "#e67e22",  # graupel
            # 🔴 Severe
            9: "#c0392b",  # hail
            # 1: "lightskyblue",     # drizzle
            # 2: "dodgerblue",       # drizzle_and_rain
            # 3: "royalblue",        # rain
            # 4: "mediumorchid",     # mixed
            # 5: "deepskyblue",      # snow
            # 6: "paleturquoise",    # snow_grains
            # 7: "darkorange",       # ice_pellets
            # 8: "orange",           # graupel
            # 9: "red",              # hail
        }
        labels = {
            -2: "No hydrometeor",
            -1: "Undefined",
            0: "No precipitation",
            1: "Drizzle",
            2: "Drizzle + Rain",
            3: "Rain",
            4: "Mixed",
            5: "Snow",
            6: "Snow grains",
            7: "Ice pellets",
            8: "Graupel",
            9: "Hail",
        }

    else:
        raise ValueError(f"Unsupported categorical variable: {name}")

    return colors, labels


####-------------------------------------------------------------------------------------------------------
#### N(D) and n(D) visualizations


L0_ND_VARIABLES = ["raw_drop_concentration"]
L1_ND_VARIABLES = ["raw_particle_counts"]
L2_ND_VARIABLES = ["drop_number_concentration", "drop_counts", "raw_drop_number_concentration", "raw_drop_counts"]
ND_VARIABLES = [*L2_ND_VARIABLES, *L1_ND_VARIABLES, *L0_ND_VARIABLES]

ND_LABEL_DICT = {
    # L0
    "raw_spectrum": "$n_{raw}(D,V)$ [#]",  # FUTURE
    "raw_drop_concentration": "$N_{raw}(D)$ [# $m^{-3} mm^{-1}$]",  # FUTURE: raw_particle_number_concentration
    # L1
    # raw_spectrum
    "raw_particle_counts": "$n_{raw}(D)$ [#]",
    # L2
    "raw_drop_number": "$n_{raw}(D,V)$ [#]",  # FUTURE: raw spectrum
    "drop_number": "$n(D,V)$ [#]",  # FUTURE: spectrum
    "raw_drop_counts": "$n_{raw}(D)$ [#]",
    "drop_counts": "$n(D)$ [#]",
    "raw_drop_number_concentration": "$N_{raw}(D)$ [# $m^{-3} mm^{-1}$]",
    "drop_number_concentration": "$N(D)$ [# $m^{-3} mm^{-1}$]",
}

ND_TITLE_DICT = {
    # L0
    "raw_spectrum": "Raw spectrum n(D,V)",
    "raw_drop_concentration": "Raw particle number concentration N(D)",  # FUTURE: raw_particle_number_concentration
    # L1
    "raw_particle_counts": "Raw particle counts n(D)",
    # L2
    "raw_drop_number": "Raw spectrum n(D,V)",
    "drop_number": "Spectrum n(D,V)",
    "raw_drop_counts": "Raw drop counts n(D)",
    "drop_counts": "Drop counts n(D)",
    "raw_number_concentration": "Raw drop number concentration N(D)",
    "drop_number_concentration": "Drop number concentration N(D)",
}


def _get_nd_labels(variable_name, da=None):
    """Get appropriate labels based on the variable type.

    Parameters
    ----------
    variable_name : str
        Name of the variable.
    da : xr.DataArray, optional
        DataArray to extract units from attributes if variable is generic.

    Returns
    -------
    dict
        Dictionary with 'ylabel', 'cbar_label', and 'title' keys.
    """
    # Retrieve for DISDRODB N(D) or n(D) variables
    if variable_name in ND_LABEL_DICT:
        return {
            "ylabel": ND_LABEL_DICT[variable_name],
            "cbar_label": ND_LABEL_DICT[variable_name],
            "title": ND_TITLE_DICT[variable_name],
        }

    # Generic fallback - try to extract units from DataArray attributes
    units = None
    if da is not None and hasattr(da, "attrs") and "units" in da.attrs:
        units = da.attrs["units"]
        # Handle dimensionless units
        if units in {"1", ""}:
            units = "#"

    variable_name = variable_name.replace("_", " ").capitalize()
    if units:
        ylabel = f"{variable_name} [{units}]"
        cbar_label = f"{variable_name} [{units}]"
    else:
        ylabel = variable_name
        cbar_label = variable_name
    return {
        "ylabel": ylabel,
        "cbar_label": cbar_label,
        "title": f"{variable_name} distribution",
    }


def _single_plot_nd_distribution(
    data,
    diameter,
    diameter_bin_width,
    variable_name="drop_number_concentration",
    ax=None,
    yscale="linear",
):
    if ax is None:
        fig, ax = plt.subplots(1, 1)
    labels = _get_nd_labels(variable_name, da=data)
    ax.bar(
        diameter,
        data,
        width=diameter_bin_width,
        edgecolor="darkgray",
        color="lightgray",
        label="Data",
    )
    ax.set_xlim(diameter[0] - diameter_bin_width[0] / 2, None)
    ax.set_title(labels["title"])
    ax.set_xlabel("Drop diameter (mm)")
    ax.set_ylabel(labels["ylabel"])
    ax.set_yscale(yscale)
    return ax


def _check_has_diameter_dims(da, diameter_dim):
    if diameter_dim not in da.dims:
        raise ValueError(f"The DataArray must have dimension '{DIAMETER_DIMENSION}'.")
    if "diameter_bin_width" not in da.coords:
        raise ValueError("The DataArray must have coordinate 'diameter_bin_width'.")
    return da


def get_dataset_nd_variable_name(ds, variables=None):
    """Return N(D) or n(D) variable name present in the xarray.Dataset."""
    variables = ND_VARIABLES if variables is None else variables
    # Search for candidate variables
    variable = None
    for var in variables:
        if var in ds:
            variable = var
            break
    if variable is None:
        raise ValueError(
            "Any n(D) or N(D) variable found in the dataset. "
            f"Searched for any of {variables} variables. "
            "Please specify the variable explicitly.",
        )
    return variable


def get_nd_variable(xr_obj, variable=None, diameter_dim=DIAMETER_DIMENSION):
    """Return N(D), n(d) DataArray.

    If N(D) or n(d) not available, derive n(d) from n(D,V).

    Parameters
    ----------
    xr_obj : xr.Dataset or xr.DataArray
        Input xarray object.
    variable : str, optional
         Variable name to extract from the xarray object.
        If xr_obj is a DataArray, will return the DataArray and its name directly'.
        If xr_obj is a Dataset, if None, will search for candidate variables in order:
        ['drop_number_concentration', 'drop_counts', 'raw_drop_counts',
         'raw_particle_counts', 'raw_drop_concentration'].

    Returns
    -------
    tuple
        (DataArray, variable_name)
    """
    if not isinstance(xr_obj, (xr.Dataset, xr.DataArray)):
        raise TypeError("Expecting xarray object as input.")

    if isinstance(xr_obj, xr.DataArray):
        # If DataArray provided, use it directly
        da = xr_obj
    else:
        # xr.Dataset provided
        if variable is None:
            variable = get_dataset_nd_variable_name(xr_obj, variables=[*ND_VARIABLES, "raw_drop_number"])
        elif variable not in xr_obj:
            raise ValueError(f"The dataset does not include {variable=}.")

        # Extract DataArray
        da = xr_obj[variable]

        # Deal with raw_drop_number (in future raw_spectrum)
        # --> Compute raw_particle_counts on the fly
        if variable == "raw_drop_number" and VELOCITY_DIMENSION in da.dims:
            da = da.sum(dim=VELOCITY_DIMENSION)
            da.name = "raw_particle_counts"

    if VELOCITY_DIMENSION in da.dims:
        raise ValueError("N(D) must not have the velocity dimension.")
    da = _check_has_diameter_dims(da, diameter_dim=diameter_dim)
    return da


def plot_nd(
    xr_obj,
    variable=None,
    cmap=None,
    norm=None,
    yscale="linear",
    ax=None,
    velocity_method="theoretical_velocity",
):
    """Plot drop number concentration N(D) or drop counts n(D) timeseries.

    Parameters
    ----------
    xr_obj : xr.Dataset or xr.DataArray
        Input xarray object containing drop data.
    variable : str, optional
        Variable name to plot. If None and xr_obj is a Dataset,
        will search for candidate variables in order:
        ['drop_number_concentration', 'raw_particle_counts', 'drop_counts'].
        If xr_obj is a DataArray, it will be plotted directly.
    cmap : matplotlib colormap, optional
        Colormap to use for the plot.
    norm : matplotlib normalization, optional
        Normalization for the colormap.
    yscale : str, optional
        Scale for y-axis ('linear' or 'log'). Default is 'linear'.
    ax : matplotlib axes, optional
        Axes to plot on.
    velocity_method : str, optional
        If the dataset has a velocity_method dimension, select the method to use for plotting.
        The default is "theoretical_velocity".

    Returns
    -------
    matplotlib axes or plot object
    """
    # Select velocity_method=0 if velocity_method dimension exists (e.g. for L2E products)
    if "velocity_method" in xr_obj.dims:
        xr_obj = xr_obj.sel(velocity_method=velocity_method)

    # Retrieve N(D) or n(D)
    da_nd = get_nd_variable(xr_obj, variable=variable)
    da_nd = da_nd.compute()

    # Check not empty object
    if da_nd.size == 0:
        raise ValueError("No data to plot.")

    # Retrieve label
    variable_name = da_nd.name
    labels = _get_nd_labels(variable_name, da=da_nd)

    # Check only time and diameter dimensions are specified
    if "time" not in da_nd.dims:
        ax = _single_plot_nd_distribution(
            data=da_nd.isel(velocity_method=0, missing_dims="ignore"),
            diameter=(
                xr_obj["diameter_bin_center"] if isinstance(xr_obj, xr.Dataset) else da_nd["diameter_bin_center"]
            ),
            diameter_bin_width=(
                xr_obj["diameter_bin_width"] if isinstance(xr_obj, xr.Dataset) else da_nd["diameter_bin_width"]
            ),
            variable_name=variable_name,
            yscale=yscale,
            ax=ax,
        )
        return ax

    # Regularize input if sample_interval is available to ensure consistent time steps
    if "sample_interval" in da_nd.coords:
        da_nd = da_nd.disdrodb.regularize()

    # Set 0 values to np.nan
    da_nd = da_nd.where(da_nd > 0)

    # Define cmap an norm
    if cmap is None:
        cmap = plt.get_cmap("Spectral_r").copy()

    if norm is None:
        vmin = np.maximum(da_nd.min().item(), 1e-1)  # 0 is set to np.nan before
        norm = Normalize() if np.isnan(vmin) else LogNorm(vmin, None)

    # Plot N(D) or drop counts
    cbar_kwargs = {"label": labels["cbar_label"]}
    p = da_nd.plot.pcolormesh(x="time", norm=norm, cmap=cmap, extend="max", cbar_kwargs=cbar_kwargs, ax=ax)
    p.axes.set_title(labels["title"])
    p.axes.set_ylabel("Drop diameter (mm)")

    # Improve time axis ticks/labels ---
    locator = mdates.AutoDateLocator(minticks=4, maxticks=8)
    formatter = mdates.ConciseDateFormatter(locator)  # compact, avoids repetition

    p.axes.xaxis.set_major_locator(locator)
    p.axes.xaxis.set_major_formatter(formatter)

    # Nice rotation/alignment if still dense
    p.axes.figure.autofmt_xdate(rotation=30, ha="right")

    # Optional: avoid clipping of labels
    p.axes.figure.tight_layout()

    return p


def plot_l1_nd_quicklook(xr_obj, precipitation_type="precipitation_type", **kwargs):
    """Define L1 DSD default quicklook."""
    fig = plot_nd_quicklook(xr_obj, precipitation_type=precipitation_type, **kwargs)
    return fig


def plot_l2_nd_quicklook(
    xr_obj,
    secondary_var="R",
    secondary_label=None,
    secondary_ylim=None,
    secondary_hlines=None,
    secondary_yscale=None,
    secondary_linestyle=":",
    **kwargs,
):
    """Define L2 DSD default quicklook."""
    if secondary_var == "R" and secondary_var in xr_obj:
        secondary_ylim = secondary_ylim if secondary_ylim is not None else (0.1, 200)
        secondary_yscale = secondary_yscale if secondary_yscale is not None else "log"
        secondary_label = secondary_label if secondary_label is not None else r"R [$mm hr^{-1}$]"
        secondary_hlines = secondary_hlines if secondary_hlines is not None else (1, 10, 100)

    fig = plot_nd_quicklook(
        xr_obj,
        secondary_var=secondary_var,
        secondary_ylim=secondary_ylim,
        secondary_yscale=secondary_yscale,
        secondary_linestyle=secondary_linestyle,
        secondary_hlines=secondary_hlines,
        **kwargs,
    )
    return fig


def plot_nd_quicklook(
    xr_obj,
    # Plot layout
    hours_per_slice=3,
    max_rows=6,
    aligned=True,
    verbose=False,
    # Spectrum options
    variable=None,
    cbar_label=None,
    cmap=None,
    norm=None,
    # Diameter axis options
    d_dim=DIAMETER_DIMENSION,
    d_lim=(0.3, 5.5),
    d_label="Diameter [mm]",
    # Colorbar options
    cbar_as_legend=True,
    cbar_xpos=0.73,
    cbar_width=0.25,
    # Secondary time-series overlay options
    secondary_var=None,
    secondary_ylim=None,
    secondary_yscale="linear",
    secondary_color="black",
    secondary_alpha=1,
    secondary_linewidth=1,
    secondary_linestyle="-",
    secondary_label=None,
    secondary_hlines=None,
    # Precipitation type options
    precipitation_type=None,
    precipitation_legend_fontsize=7,
    precipitation_legend_colors=None,
    precipitation_legend_labels=None,
    precipitation_legend_ncol=None,
    precipitation_legend_height=0.3,  # inches
    # Diameter axis variables
    add_dm=True,
    add_sigma_m=True,
    sigma_label=r"$2\sigma_m$",
    sigma_linewidth=0.5,
    dm_linewidth=0.5,
    # Figure options
    dpi=300,
):
    """Display multi-rows quicklook of N(D)."""
    from pycolorbar.utils.mpl_legend import add_fancybox, get_tightbbox_position

    # Figure settings
    plt.rcParams.update(
        {
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
        },
    )

    legend_fontsize = 8
    title_fontsize = 8
    side_title_fontsize = 6
    axis_label_fontsize = 8
    cbar_fontsize = 9
    cbar_labelpad = 6
    cbar_ypos = 0.7
    time_ticklabel_pad = 2

    # ------------------------------------------------------------------------.
    # Ensure to create Dataset if input is DataArray
    if isinstance(xr_obj, xr.Dataset):
        ds = xr_obj
    elif isinstance(xr_obj, xr.DataArray):
        variable = xr_obj.name
        variable = "unknown DSD" if variable is None else variable
        ds = xr_obj.to_dataset(name=variable)
        # Set options to False
        precipitation_type = None
        secondary_var = None
        add_dm = False
        add_sigma_m = False

    else:
        raise TypeError("Expecting xarray.Dataset or xarray.DataArray as input.")

    # ------------------------------------------------------------------------.
    # Validate generic secondary axis and precipitation classification options
    if secondary_var is not None:
        if not isinstance(secondary_var, str):
            raise TypeError("secondary_var must be a string or None.")
        if secondary_var not in ds:
            raise ValueError(f"{secondary_var} not found in dataset.")

    if secondary_ylim is not None and not (isinstance(secondary_ylim, (tuple, list)) and len(secondary_ylim) == 2):
        raise ValueError("secondary_ylim must be a tuple/list of length 2.")

    if secondary_hlines is not None and not isinstance(secondary_hlines, (list, tuple)):
        raise TypeError("secondary_hlines must be a list or tuple.")

    if precipitation_type is not None:
        if not isinstance(precipitation_type, str):
            raise TypeError("precipitation_type must be a string or None.")
        if precipitation_type not in ds:
            raise ValueError(f"{precipitation_type} not found in dataset.")

    # ------------------------------------------------------------------------.
    # Disable overlays when required variables are missing
    if "Dm" not in ds:
        add_dm = False
    if "sigma_m" not in ds:
        add_sigma_m = False

    # ------------------------------------------------------------------------.
    # Select velocity_method
    if "velocity_method" in ds.dims:
        velocity_method = ds["velocity_method"].to_numpy()[0]
        print(f"Selecting velocity_method '{velocity_method}")
        ds = ds.sel(velocity_method=velocity_method)

    # ------------------------------------------------------------------------.
    # Derive N(D) variable
    da_nd = get_nd_variable(ds, variable=variable, diameter_dim=d_dim)
    variable = da_nd.name
    ds[da_nd.name] = da_nd  # might have computed n(d) on-the-fly from N(D,V)

    # ------------------------------------------------------------------------.
    # Define precipitation type classification (colors and legend)
    add_precipitation_legend = precipitation_type is not None
    if add_precipitation_legend:
        if precipitation_legend_colors is None:
            precipitation_legend_colors, precipitation_legend_labels = get_precipitation_legend_style(
                precipitation_type,
            )
        if precipitation_legend_ncol is None:
            precipitation_legend_ncol = len(precipitation_legend_labels)

    # ------------------------------------------------------------------------.
    # Colormap & normalization
    if cmap is None:
        cmap = plt.get_cmap("Spectral_r").copy()
        cmap.set_under("none")
    if norm is None:
        norm = LogNorm(vmin=1, vmax=10_000)

    # ------------------------------------------------------------------------.
    # Define cbar label
    if cbar_label is None:
        if variable in ND_LABEL_DICT:
            cbar_label = ND_LABEL_DICT[variable]
        else:
            units = ds[variable].attrs.get("units", "-")
            cbar_label = f"{variable} [{units}]"

    # ------------------------------------------------------------------------.
    # Calculate event duration
    duration = ds.disdrodb.end_time - ds.disdrodb.start_time

    # ------------------------------------------------------------------------.
    # Define temporal slices
    # - Align to closest <hours_per_slice> time
    # - For hours_per_slice=3 --> 00, 03, 06, ...
    time = ds["time"].to_index()
    t_start = time[0]
    t_end = time[-1]
    if aligned:
        aligned_start = t_start.floor(f"{hours_per_slice}h")
        aligned_end = t_end.ceil(f"{hours_per_slice}h")
        # Create time bins
        time_bins = pd.date_range(
            start=aligned_start,
            end=aligned_end,
            freq=f"{hours_per_slice}h",
        )
    else:
        # Create time bins
        time_bins = pd.date_range(
            start=t_start,
            end=t_end + pd.Timedelta(f"{hours_per_slice}h"),
            freq=f"{hours_per_slice}h",
        )

    n_total_slices = len(time_bins) - 1
    n_slices = min(n_total_slices, max_rows)

    # ------------------------------------------------------------------------.
    # Print info on event quicklook
    if verbose:
        print("=== N(D) Event Quicklook ===")
        print(f"Dataset time span : {t_start} → {t_end}")
        print(f"Slice length      : {hours_per_slice} h")
        print(f"Plotted slices    : {n_slices}/{n_total_slices}")
        if n_total_slices > max_rows:
            last_plotted_end = time_bins[max_rows]
            print(f"Unplotted period  : {last_plotted_end} → {t_end}")

    # ------------------------------------------------------------------------.
    # Regularize dataset to match bin start_time and end_time
    ds = ds.disdrodb.regularize(
        start_time=time_bins[0],
        end_time=time_bins[-1],
        fill_value=np.nan,
    )

    # Check at least 2 timesteps are available
    if ds.sizes["time"] < 2:
        raise ValueError("Dataset must have at least 2 time steps for quicklook.")

    # Enforce legend colorbar for n_slices 1 and 2
    if n_slices <= 2:
        cbar_as_legend = True

    ####-----------------------------------------------------------------------.
    #### Define figure with GridSpec
    # - If cbar_as_legend=False: reserve extra row for colorbar
    # - If cbar_as_legend=True: no extra row needed
    # - If add_precipitation_legend True, add extra row for precipitation classification legend

    # Define number of extra rows
    extra_rows = 1 if (not cbar_as_legend) else 0
    extra_rows += 1 if add_precipitation_legend else 0

    # Define figure size
    fig_width = 6.9
    subplot_height = 1.9
    fig_height = subplot_height * n_slices + (precipitation_legend_height if add_precipitation_legend else 0)
    figsize = (fig_width, fig_height)

    # Define height ratios
    hspace = 0.15
    height_ratios = [1] * n_slices
    if not cbar_as_legend:
        cbar_height_ratio = 0.2 if n_slices == 3 else 0.15  # more subplots = relatively smaller colorbar row
        height_ratios.append(cbar_height_ratio)

    if add_precipitation_legend:
        height_ratio_precipitation_legend = precipitation_legend_height / subplot_height
        height_ratios.append(height_ratio_precipitation_legend)
        hspace = 0.2

    # Create figure
    fig = plt.figure(figsize=figsize, dpi=dpi)
    gs = GridSpec(
        nrows=n_slices + extra_rows,
        ncols=1,
        figure=fig,
        height_ratios=height_ratios,
        hspace=hspace,
    )

    axes = [fig.add_subplot(gs[i, 0]) for i in range(n_slices)]
    ax_sec_for_legend = None

    # ------------------------------------------------------------------------.
    #### - Plot each slice
    for i in range(n_slices):
        # Extract dataset slice
        t0 = time_bins[i]
        t1 = time_bins[i + 1]
        ds_slice = ds.sel(time=slice(t0, t1))
        da_nd = ds_slice[variable]

        # Define plot ax
        ax = axes[i]

        # Plot N(D)
        p = da_nd.plot.pcolormesh(
            ax=ax,
            x="time",
            y=d_dim,
            norm=norm,
            cmap=cmap,
            shading="auto",
            add_colorbar=False,
        )
        # Always remove xarray default title
        ax.set_title("")

        #### - Overlay Dm
        if add_dm:
            ds_slice["Dm"].plot(
                ax=ax,
                x="time",
                color="black",
                linestyle="-",
                linewidth=dm_linewidth,
                label="$D_m$",
            )

        #### - Overlay sigma_m
        if add_sigma_m:
            (ds_slice["sigma_m"] * 2).plot(
                ax=ax,
                x="time",
                color="black",
                linestyle="--",
                linewidth=sigma_linewidth,
                label=sigma_label,
            )

        # Remove xarray default title
        ax.set_title("")

        # Add axis labels - remove ylabel from all individual axes
        ax.set_xlabel("")
        ax.set_ylabel("")

        #### - Add generic secondary time series on a twin axis
        if secondary_var is not None:
            ax_sec = ax.twinx()
            ds_slice[secondary_var].plot(
                ax=ax_sec,
                x="time",
                color=secondary_color,
                alpha=secondary_alpha,
                linewidth=secondary_linewidth,
                linestyle=secondary_linestyle,
                label=secondary_var,
            )
            # Always remove xarray default title
            ax_sec.set_title("")
            ax_sec.set_yscale(secondary_yscale)
            # Display playnumbers instead of scientific notation
            yticks = ax_sec.get_yticks()
            ytick_labels = [f"{t:g}" for t in yticks]
            ax_sec.set_yticks(yticks)
            ax_sec.set_yticklabels(ytick_labels)
            if secondary_ylim is not None:
                ax_sec.set_ylim(secondary_ylim)

            # Remove ylabel from all individual axes
            ax_sec.set_ylabel("")
            ax_sec.tick_params(axis="y", labelcolor=secondary_color)

            if secondary_hlines is not None:
                for y in secondary_hlines:
                    ax_sec.axhline(
                        y=y,
                        color="gray",
                        alpha=0.2,
                        linewidth=1,
                        linestyle="-",
                        zorder=0,
                    )

            if ax_sec_for_legend is None:
                ax_sec_for_legend = ax_sec

        ax.set_ylim(*d_lim)
        ax.tick_params(axis="x", pad=time_ticklabel_pad)
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))

        #### - Add precipitation classification strip at the top of the subplot as inset axes
        if add_precipitation_legend:
            # Get classification values for this time slice
            precip_type = ds_slice[precipitation_type]

            # Create inset axes at the top (sharing x-axis with main plot)
            # [x0, y0, width, height] in axes coordinates
            ax_precip = ax.inset_axes([0, 0.95, 1, 0.05], sharex=ax)

            # Define colormap and norm
            # - Sort category values
            values = np.array(sorted(precipitation_legend_colors.keys()))

            # - Extract colors in same order
            colors = [precipitation_legend_colors[v] for v in values]

            # - Create colormap
            cmap_precip = ListedColormap(colors)

            # - Build automatic boundaries
            # Works even for negative or non-consecutive values
            boundaries = np.concatenate(
                [
                    [values[0] - 0.5],
                    (values[:-1] + values[1:]) / 2,
                    [values[-1] + 0.5],
                ],
            )
            norm_precip = BoundaryNorm(boundaries, cmap_precip.N)

            # Plot 1-pixel-high strip
            t = precip_type["time"].to_numpy()
            dt = np.diff(t)  # timedelta64
            t_edges = np.concatenate(
                [
                    [t[0] - dt[0] / 2],
                    t[:-1] + dt / 2,
                    [t[-1] + dt[-1] / 2],
                ],
            )

            ax_precip.pcolormesh(
                t_edges,
                [0, 1],
                precip_type.to_numpy()[np.newaxis, :],
                cmap=cmap_precip,
                norm=norm_precip,
                shading="flat",
            )
            # Add 'axis' line
            ax_precip.axhline(
                y=0,
                color="black",
                linewidth=1,
                alpha=1.0,
            )

            # Remove ticks and ticklabels
            ax_precip.set_yticks([])
            ax_precip.xaxis.set_visible(False)
            for spine in ax_precip.spines.values():
                spine.set_visible(False)

    # Add time xlabel
    if precipitation_type is None:
        axes[n_slices - 1].set_xlabel("Time (UTC)", fontsize=axis_label_fontsize)

    # ------------------------------------------------------------------------.
    #### - Add title
    # Format duration as "XhYmin" or "Xmin"
    total_minutes = int(duration.total_seconds() / 60)
    hours = total_minutes // 60
    minutes = total_minutes % 60
    duration_str = f"{hours}H{minutes:02d}MIN" if hours > 0 else f"{minutes}MIN"

    # Format title based on whether dates are the same
    t_start_dt = time_bins[0]
    t_end_dt = time_bins[n_slices]
    if t_start_dt.date() == t_end_dt.date():
        # Same date: show "YYYY-MM-DD HH:MM - HH:MM UTC"
        title_str = f"{t_start_dt.strftime('%Y-%m-%d %H:%M')} - {t_end_dt.strftime('%H:%M')} UTC"
    else:
        # Different dates: show full datetime for both
        title_str = f"{t_start_dt.strftime('%Y-%m-%d %H:%M')} - {t_end_dt.strftime('%Y-%m-%d %H:%M')} UTC"

    # Set center title
    axes[0].set_title(
        title_str,
        fontsize=title_fontsize,
        fontweight="bold",
        loc="center",
    )

    # Add right title with event duration
    axes[0].set_title(
        duration_str,
        fontsize=side_title_fontsize,
        loc="right",
    )

    # ------------------------------------------------------------------------.
    #### - Add centered y-labels in the middle of the figure (closer to axes)
    fig.text(
        0.08,
        0.5,
        d_label,
        va="center",
        rotation="vertical",
        fontsize=axis_label_fontsize,
    )

    if secondary_var is not None:
        if secondary_label is None:
            units = ds[secondary_var].attrs.get("units", "")
            secondary_label = f"{secondary_var} [{units}]" if units else secondary_var
        fig.text(
            0.945,
            0.5,
            secondary_label,
            va="center",
            rotation="vertical",
            fontsize=axis_label_fontsize,
            color=secondary_color,
        )

    # ------------------------------------------------------------------------.
    #### - Add legend
    # Collect legend handles from both axes
    handles, labels = axes[0].get_legend_handles_labels()

    if secondary_var is not None and ax_sec_for_legend is not None:
        handles_sec, labels_sec = ax_sec_for_legend.get_legend_handles_labels()
        handles += handles_sec
        labels += labels_sec

    if np.any([secondary_var is not None, add_sigma_m, add_dm]):
        axes[0].legend(
            handles,
            labels,
            loc="upper left",
            bbox_to_anchor=(0, 0.98),
            fontsize=legend_fontsize,
            frameon=True,
            fancybox=False,
            edgecolor="black",
        )

    # ------------------------------------------------------------------------.
    #### - Add colorbar
    if cbar_as_legend:
        # Add colorbar as a legend in the last subplot with background box
        cax = axes[-1].inset_axes([cbar_xpos, cbar_ypos, cbar_width, 0.10])  # [x, y, width, height] in axes coords

        # # Raise z-order so the colorbar is on top and fancybox behind
        fancybox_zorder = cax.get_zorder() + 1
        cax.set_zorder(cax.get_zorder() + 2)

        cbar = fig.colorbar(
            p,
            cax=cax,
            orientation="horizontal",
            extend="max",
        )
        # Move label above colorbar
        cbar.ax.set_xlabel(cbar_label, fontsize=cbar_fontsize, labelpad=cbar_labelpad)
        cbar.ax.xaxis.set_label_position("top")
        cbar.ax.tick_params(labelsize=9)

        # Add white box with edge behind colorbar
        fancy_bbox = get_tightbbox_position(cax)
        add_fancybox(
            ax=axes[-1],
            bbox=fancy_bbox,
            pad=0,
            fc="white",
            ec="none",
            lw=0.5,
            alpha=0.9,
            shape="square",
            zorder=fancybox_zorder,
        )

    else:
        # Add colorbar as separate subplot at bottom
        cbar_pad = 0.05 * (3 / n_slices)
        cbar_fraction = 0.03 * (3 / n_slices)

        cbar = fig.colorbar(
            p,
            ax=axes,
            orientation="horizontal",
            pad=cbar_pad,
            fraction=cbar_fraction,
            extend="max",
        )
        cbar.set_label(cbar_label, fontsize=cbar_fontsize)

    #### Add classification legend
    if add_precipitation_legend:
        legend_patches = [
            Patch(facecolor=precipitation_legend_colors[k], edgecolor="black", label=precipitation_legend_labels[k])
            for k in sorted(precipitation_legend_colors.keys())
        ]

        # Select last row in GridSpec
        precip_ax = fig.add_subplot(gs[-1, 0])
        precip_ax.axis("off")

        precip_ax.legend(
            handles=legend_patches,
            loc="center left",
            bbox_to_anchor=(-0.01, 0.5),
            ncol=precipitation_legend_ncol,
            frameon=False,
            fontsize=precipitation_legend_fontsize,
            columnspacing=1.2,
            handlelength=0.8,
        )

    # Return figure
    return fig


####-------------------------------------------------------------------------------------------------------
#### Spectra visualizations


def _check_has_diameter_and_velocity_dims(da):
    if DIAMETER_DIMENSION not in da.dims or VELOCITY_DIMENSION not in da.dims:
        raise ValueError(f"The DataArray must have both '{DIAMETER_DIMENSION}' and '{VELOCITY_DIMENSION}' dimensions.")
    return da


def _get_spectrum_variable(xr_obj, variable):
    if not isinstance(xr_obj, (xr.Dataset, xr.DataArray)):
        raise TypeError("Expecting xarray object as input.")
    if VELOCITY_DIMENSION not in xr_obj.dims:
        raise ValueError("2D spectrum not available.")
    if isinstance(xr_obj, xr.Dataset):
        if variable not in xr_obj:
            raise ValueError(f"The dataset do not include {variable=}.")
        xr_obj = xr_obj[variable]
    xr_obj = _check_has_diameter_and_velocity_dims(xr_obj)
    return xr_obj


def plot_spectrum_evolution(
    ds,
    legend_variables=None,
    legend_ncol=1,
    xlim=None,
    ylim=None,
    plot_hc_rain_mask_boundary=False,
    **plot_kwargs,
):
    """Plot the evolution of disdrodb spectra over time.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset containing 'time' dimension and 'disdrodb'.
    legend_variables : list of str, optional
        Dataset variables to display in the legend.
    legend_ncol : int, optional
        Number of legend entries per row (horizontal layout).
    xlim, ylim : tuple, optional
        Axis limits passed to matplotlib.
    plot_kwargs : dict
        Additional keyword arguments passed to plot_spectrum().
    """
    # Check timestep available
    if "time" not in ds.dims or ds.sizes["time"] == 0:
        raise ValueError("No timesteps available.")

    # Define legend formatting
    # --> Define decimals per variable
    decimals = {}
    if legend_variables is not None:
        for var in legend_variables:
            if var not in ds:
                raise KeyError(f"Variable '{var}' not found in dataset")

            ds[var] = ds[var].compute()  # Ensure variable is loaded in memory

            values = ds[var].to_numpy()

            # Remove NaNs
            values = values[np.isfinite(values)]

            # Integer-like → 0 decimals
            if np.allclose(values, np.round(values)):
                decimals[var] = 0
            else:
                decimals[var] = 2

    # Precompute hc rain mask contour
    if plot_hc_rain_mask_boundary:
        contour = get_spectrum_mask_boundary(
            ds,
            above_velocity_tolerance=2,
            below_velocity_fraction=None,
            below_velocity_tolerance=3,
            maintain_drops_smaller_than=1,  # 1,   # 2
            maintain_drops_slower_than=2.5,  # 2.5, # 3
            maintain_smallest_drops=False,
            fall_velocity_model="Beard1976",
        )

    # Loop over time
    for i in range(ds.sizes["time"]):
        ds_i = ds.isel(time=i)

        # Define figure
        fig, ax = plt.subplots()

        # Plot spectrum
        plot_spectrum(ds_i, ax=ax, plot_hc_rain_mask_boundary=False, **plot_kwargs)

        if plot_hc_rain_mask_boundary:
            plot_contour(contour, ax=ax, color="black", linestyle="--")

        # Set title
        title_str = pd.to_datetime(ds_i["time"].to_numpy()).strftime("%Y-%m-%d %H:%M:%S")
        ax.set_title(title_str)

        # Add legend
        if legend_variables is not None:
            handles = []
            labels = []
            for var in legend_variables:
                value = ds_i[var].item()
                dec = decimals[var]
                label = f"{var}: NaN" if np.isnan(value) else f"{var}: {value:.{dec}f}"
                # Invisible handle
                handles.append(Line2D([], [], linestyle="none"))
                labels.append(label)
            ax.legend(
                handles,
                labels,
                loc="upper left",
                ncol=legend_ncol,
                frameon=True,
                handlelength=0,
                handletextpad=0.0,
                columnspacing=1.2,
            )

        # Set limits if specified
        if xlim is not None:
            plt.xlim(xlim)
        if ylim is not None:
            plt.ylim(ylim)

        plt.show()


def plot_spectrum(
    xr_obj,
    variable="raw_drop_number",
    ax=None,
    cmap=None,
    norm=None,
    extend="max",
    add_colorbar=True,
    cbar_kwargs=None,
    title=None,
    plot_hc_rain_mask_boundary=False,
    **plot_kwargs,
):
    """Plot the spectrum.

    Parameters
    ----------
    xr_obj : xarray.Dataset or xarray.DataArray
        Input xarray object. If Dataset, the variable to plot must be specified.
        If DataArray, it must have both diameter and velocity dimensions.
    variable : str
        Name of the variable to plot if xr_obj is a Dataset.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, uses current axes or creates a new one.
    cmap : matplotlib.colors.Colormap, optional
        Colormap to use. If None, uses 'Spectral_r' with 'under' set to 'none'.
    norm : matplotlib.colors.Normalize, optional
        Normalization for colormap. If None, uses LogNorm with vmin=1.
    extend : str, optional
        Whether to draw arrows on the colorbar to indicate out-of-range values.
        Valid options are 'neither', 'min', 'max', 'both'.
        Default is 'max'.
    add_colorbar : bool, optional
        Whether to add a colorbar. Default is True.
    cbar_kwargs : dict, optional
        Additional keyword arguments for colorbar. If None, uses {'label': 'Number of particles '}.
    title : str, optional
        Title of the plot. If not provided, defaults to the timestep or time range of the spectrum.
    **plot_kwargs : dict
        Additional keyword arguments passed to xarray's plot.pcolormesh method.

    Notes
    -----
    If the input DataArray has a time dimension, it is summed over time before plotting
    unless FacetGrid options (e.g., col, row) are specified in plot_kwargs.

    If FacetGrid options are used, the plot will create a grid of subplots for each time slice.

    To create a FacetGrid plot, use:

      ds.isel(time=slice(0, 9)).disdrodb.plot_spectrum(col="time", col_wrap=3)

    """
    # Retrieve spectrum
    drop_number = _get_spectrum_variable(xr_obj, variable)

    # Check if FacetGrid
    is_facetgrid = "col" in plot_kwargs or "row" in plot_kwargs

    # Check not empty object
    if drop_number.size == 0:
        raise ValueError("No data to plot.")

    # Define start_time and end_time if time coordinate is present
    drop_number = drop_number.squeeze()
    if "time" in drop_number.dims:
        start_time = pd.to_datetime(drop_number.disdrodb.start_time).strftime("%Y-%m-%d %H:%M:%S")
        end_time = pd.to_datetime(drop_number.disdrodb.end_time).strftime("%Y-%m-%d %H:%M:%S")
    else:
        start_time = None
        end_time = None

    # Sum over time dimension if still present
    # - Unless FacetGrid options in plot_kwargs
    if "time" in drop_number.dims and not is_facetgrid:
        drop_number = drop_number.sum(dim="time")
        if title is None:
            title = f"{start_time} - {end_time}" if start_time is not None else ""
    elif title is None:
        title = f"{start_time}" if start_time is not None else ""

    # Define default cbar_kwargs if not specified
    if cbar_kwargs is None:
        cbar_kwargs = {"label": "Number of particles"}

    # Define cmap and norm
    if cmap is None:
        cmap = plt.get_cmap("Spectral_r").copy()
        cmap.set_under("none")

    if norm is None:
        norm = LogNorm(vmin=1, vmax=None) if drop_number.sum() > 0 else None

    # Remove cbar_kwargs if add_colorbar=False
    if not add_colorbar:
        cbar_kwargs = None

    # Plot
    p = drop_number.plot.pcolormesh(
        ax=ax,
        x=DIAMETER_DIMENSION,
        y=VELOCITY_DIMENSION,
        cmap=cmap,
        extend=extend,
        norm=norm,
        add_colorbar=add_colorbar,
        cbar_kwargs=cbar_kwargs,
        **plot_kwargs,
    )

    if plot_hc_rain_mask_boundary:
        contour = get_spectrum_mask_boundary(
            xr_obj,
            above_velocity_tolerance=2,
            below_velocity_fraction=None,
            below_velocity_tolerance=3,
            maintain_drops_smaller_than=1,  # 1,   # 2
            maintain_drops_slower_than=2.5,  # 2.5, # 3
            maintain_smallest_drops=False,
            fall_velocity_model="Beard1976",
        )
        plot_contour(contour, ax=ax, color="black", linestyle="--")

    if not is_facetgrid:
        p.axes.set_xlabel("Diamenter [mm]")
        p.axes.set_ylabel("Fall velocity [m/s]")
        p.axes.set_title(title)
    else:
        p.set_axis_labels("Diameter [mm]", "Fall velocity [m/s]")
    return p


def plot_raw_and_filtered_spectra(
    ds,
    cmap=None,
    norm=None,
    extend="max",
    add_theoretical_average_velocity=True,
    add_measured_average_velocity=True,
    figsize=(6.9, 3.2),
    dpi=300,
):
    """Plot raw and filtered drop spectrum."""
    # Retrieve spectrum arrays
    drop_number = _get_spectrum_variable(ds, variable="drop_number")
    if "time" in drop_number.dims:
        drop_number = drop_number.sum(dim="time")
    drop_number = drop_number.compute()

    raw_drop_number = _get_spectrum_variable(ds, variable="raw_drop_number")
    if "time" in raw_drop_number.dims:
        raw_drop_number = raw_drop_number.sum(dim="time")
    raw_drop_number = raw_drop_number.compute()

    # Compute theoretical and measured average velocity if asked
    if add_theoretical_average_velocity:
        theoretical_average_velocity = ds["fall_velocity"]
        if "time" in theoretical_average_velocity.dims:
            theoretical_average_velocity = theoretical_average_velocity.mean(dim="time")
    if add_measured_average_velocity and VELOCITY_DIMENSION in drop_number.dims:
        measured_average_velocity = get_drop_average_velocity(drop_number)

    # Define norm if not specified
    if norm is None:
        norm = LogNorm(1, raw_drop_number.max())

    # Initialize figure
    fig = plt.figure(figsize=figsize, dpi=dpi)
    gs = GridSpec(1, 2, width_ratios=[1, 1.15], wspace=0.05)  # More space for ax2
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])

    # Plot raw_drop_number
    plot_spectrum(raw_drop_number, ax=ax1, cmap=cmap, norm=norm, extend=extend, add_colorbar=False, title="")

    # Add velocities if asked
    if add_theoretical_average_velocity:
        theoretical_average_velocity.plot(ax=ax1, c="k", linestyle="dashed")
    if add_measured_average_velocity and VELOCITY_DIMENSION in drop_number.dims:
        measured_average_velocity.plot(ax=ax1, c="k", linestyle="dotted")

    # Improve plot appearance
    ax1.set_xlabel("Diamenter [mm]")
    ax1.set_ylabel("Fall velocity [m/s]")
    ax1.set_title("Raw Spectrum")

    # Plot drop_number
    plot_spectrum(drop_number, ax=ax2, cmap=cmap, norm=norm, extend=extend, add_colorbar=True, title="")

    # Add velocities if asked
    if add_theoretical_average_velocity:
        theoretical_average_velocity.plot(ax=ax2, c="k", linestyle="dashed", label="Theoretical velocity")
    if add_measured_average_velocity and VELOCITY_DIMENSION in drop_number.dims:
        measured_average_velocity.plot(ax=ax2, c="k", linestyle="dotted", label="Measured average velocity")

    # Improve plot appearance
    ax2.set_yticks([])
    ax2.set_yticklabels([])
    ax2.set_xlabel("Diamenter [mm]")
    ax2.set_ylabel("")
    ax2.set_title("Filtered Spectrum")

    # Add legend
    if add_theoretical_average_velocity or add_measured_average_velocity:
        ax2.legend(loc="lower right", frameon=False)

    return fig


####---------------------------------------------------------------------------.
#### Mask utilities


def plot_contour(contour, ax=None, **kwargs):
    """Plot contour [X,Y]."""
    if ax is None:
        fig, ax = plt.subplots(1, 1)
    label = kwargs.pop("label", None)
    for i, seg in enumerate(contour):
        if label is not None and i == len(contour) - 1:
            kwargs["label"] = label
        ax.plot(seg[:, 0], seg[:, 1], **kwargs)


def plot_mask_contour(mask, ax=None, **kwargs):
    """Plot mask contour."""
    contour = get_mask_contour(mask)
    if ax is None:
        fig, ax = plt.subplots(1, 1)
    label = kwargs.pop("label", None)
    for i, seg in enumerate(contour):
        if label is not None and i == len(contour) - 1:
            kwargs["label"] = label
        ax.plot(seg[:, 0], seg[:, 1], **kwargs)


####-------------------------------------------------------------------------------------------------------
#### DenseLines


def normalize_array(arr, method="max"):
    """Normalize a NumPy array according to the chosen method.

    Parameters
    ----------
    arr : numpy.ndarray
        Input array.
    method : str
        Normalization method. Options:

        - 'max'  : Divide by the maximum value.
        - 'minmax': Scale to [0, 1] range.
        - 'zscore': Standardize to mean 0, std 1.
        - 'log'  : Apply log10 transform (shifted if min <= 0).
        - 'none' : No normalization (return original array).


    Returns
    -------
    numpy.ndarray
        Normalized array.
    """
    arr = np.asarray(arr, dtype=float)

    if method == "max":
        max_val = np.nanmax(arr)
        return arr / max_val if max_val != 0 else arr

    if method == "minmax":
        min_val = np.nanmin(arr)
        max_val = np.nanmax(arr)
        return (arr - min_val) / (max_val - min_val) if max_val != min_val else np.zeros_like(arr)

    if method == "zscore":
        mean_val = np.nanmean(arr)
        std_val = np.nanstd(arr)
        return (arr - mean_val) / std_val if std_val != 0 else np.zeros_like(arr)

    if method == "log":
        min_val = np.nanmin(arr)
        shifted = arr - min_val + 1e-12  # Shift to avoid log(0) or log of negative
        return np.log10(shifted)

    if method == "none":
        return arr

    raise ValueError(f"Unknown normalization method: {method}")


def _np_to_rgba_alpha(arr, cmap="viridis", cmap_norm=None, scaling="linear"):
    """Convert a numpy array to an RGBA array with alpha based on array value.

    Parameters
    ----------
    arr : numpy.ndarray
        arr of counts or frequencies.
    cmap : str or Colormap, optional
        Matplotlib colormap to use for RGB channels.
    cmap_norm: matplotlib.colors.Norm
        Norm to be used to scale data before assigning cmap colors.
        The default is Normalize(vmin, vmax).
    scaling : str, optional
        Scaling type for alpha mapping:

        - "linear"   : min-max normalization
        - "log"      : logarithmic normalization (positive values only)
        - "sqrt"     : square-root (power-law with exponent=0.5)
        - "exp"      : exponential scaling
        - "quantile" : percentile-based scaling
        - "none"     : full opacity (alpha=1)


    Returns
    -------
    rgba : 3D numpy array (ny, nx, 4)
        RGBA array.
    """
    # Ensure numpy array
    arr = np.asarray(arr, dtype=float)
    # Define mask with NaN pixel
    mask_na = np.isnan(arr)
    # Retrieve array shape
    ny, nx = arr.shape

    # Define colormap norm
    if cmap_norm is None:
        cmap_norm = Normalize(vmin=np.nanmin(arr), vmax=np.nanmax(arr))

    # Define alpha
    if scaling == "linear":
        norm = Normalize(vmin=np.nanmin(arr), vmax=np.nanmax(arr))
        alpha = norm(arr)
    elif scaling == "log":
        vals = np.where(arr > 0, arr, np.nan)  # mask non-positive
        norm = LogNorm(vmin=np.nanmin(vals), vmax=np.nanmax(vals))
        alpha = norm(arr)
        alpha = np.nan_to_num(alpha, nan=0.0)
    elif scaling == "sqrt":
        alpha = np.sqrt(np.clip(arr, 0, None) / np.nanmax(arr))
    elif scaling == "exp":
        normed = np.clip(arr / np.nanmax(arr), 0, 1)
        alpha = np.expm1(normed) / np.expm1(1)
    elif scaling == "quantile":
        flat = arr.ravel()
        ranks = np.argsort(np.argsort(flat))  # rankdata without scipy
        alpha = ranks / (len(flat) - 1)
        alpha = alpha.reshape(arr.shape)
    elif scaling == "none":
        alpha = np.ones_like(arr, dtype=float)
    else:
        raise ValueError(f"Unknown scaling type: {scaling}")

    # Map values to colors
    cmap = plt.get_cmap(cmap).copy()
    rgba = cmap(cmap_norm(arr))

    # Set alpha channel
    alpha[mask_na] = 0  # where input was NaN
    rgba[..., -1] = np.clip(alpha, 0, 1)
    return rgba


def to_rgba(obj, cmap="viridis", norm=None, scaling="none"):
    """Map a xarray DataArray (or numpy array) to RGBA with optional alpha-scaling."""
    input_is_xarray = False
    if isinstance(obj, xr.DataArray):
        # Define template for RGBA DataArray
        da_rgba = obj.copy()
        da_rgba = da_rgba.expand_dims({"rgba": 4}).transpose(..., "rgba")
        input_is_xarray = True

        # Extract numpy array
        obj = obj.to_numpy()

    # Apply transparency
    arr = _np_to_rgba_alpha(obj, cmap=cmap, cmap_norm=norm, scaling=scaling)

    # Return xarray.DataArray
    if input_is_xarray:
        da_rgba.data = arr
        return da_rgba
    # Or numpy array otherwise
    return arr


def max_blend_images(ds_rgb, dim):
    """Max blend a RGBA DataArray across a samples dimensions."""
    # Ensure dimension to blend in first position
    ds_rgb = ds_rgb.transpose(dim, ...)
    # Extract numpy array
    stack = ds_rgb.data
    # Extract alpha array
    alphas = stack[..., 3]
    # Select the winning RGBA per pixel  # (N, H, W)
    idx = np.argmax(alphas, axis=0)  # (H, W), index of image with max alpha
    idx4 = np.repeat(idx[np.newaxis, ..., np.newaxis], 4, axis=-1)  # (1, H, W, 4)
    out = np.take_along_axis(stack, idx4, axis=0)[0]  # (H, W, 4)
    # Create output RGBA array
    da = ds_rgb.isel({dim: 0}).copy()
    da.data = out
    return da


def _create_denseline_grid(indices, ny, nx, nsamples):
    # Assign 1 when line pass in a bin
    valid = (indices >= 0) & (indices < ny)
    s_idx, x_idx = np.nonzero(valid)
    y_idx = indices[valid]

    # ----------------------------------------------
    ### Vectorized code with high memory footprint because of 3D array

    # # Create 3D array with hits
    # grid_3d = np.zeros((nsamples, ny, nx), dtype=np.int64)
    # grid_3d[s_idx, y_idx, x_idx] = 1

    # # Normalize by columns
    # col_sums = grid_3d.sum(axis=1, keepdims=True)
    # col_sums[col_sums == 0] = 1  # Avoid division by zero
    # grid_3d = grid_3d / col_sums

    # # Sum over samples
    # grid = grid_3d.sum(axis=0)

    # # Free memory
    # del grid_3d

    # ----------------------------------------------
    ## Vectorized alternative with much lower memory footprint

    # Count hits per (sample, y, x)
    grid = np.zeros((ny, nx), dtype=np.float64)

    # Compute per-sample-per-column counts
    col_counts = np.zeros((nsamples, nx), dtype=np.int64)
    np.add.at(col_counts, (s_idx, x_idx), 1)

    # Define weights to normalize contributions, avoiding division by zero
    # - Weight = 1 / (# hits per column, per sample)
    col_counts[col_counts == 0] = 1
    weights = 1.0 / col_counts[s_idx, x_idx]

    # Accumulate weighted contributions
    np.add.at(grid, (y_idx, x_idx), weights)

    # Return 2D grid
    return grid


def _compute_block_size(ny, nx, dtype=np.float64, safety_margin=2e9):
    """Compute maximum block size given available memory."""
    avail_mem = psutil.virtual_memory().available - safety_margin

    # Constant cost for final grid
    base = ny * nx * np.dtype(dtype).itemsize

    # Per-sample cost (worst case, includes col_counts + indices + weights)
    per_sample = nx * 40

    max_block = (avail_mem - base) // per_sample
    return max(1, int(max_block))


def compute_dense_lines(
    da: xr.DataArray,
    coord: str,
    x_bins: list,
    y_bins: list,
    normalization="max",
):
    """
    Compute a 2D density-of-lines histogram from an xarray.DataArray.

    Parameters
    ----------
    da : xarray.DataArray
        Input data array. One of its dimensions (named by ``coord``) is taken
        as the horizontal coordinate. All other dimensions are collapsed into
        “series,” so that each combination of the remaining dimension values
        produces one 1D line along ``coord``.
    coord : str
        The name of the coordinate/dimension of the DataArray to bin over.
        ``da.coords[coord]`` must be a 1D numeric array (monotonic is recommended).
    x_bins : array-like
        Bin edges to bin the coordinate/dimension with shape (nx+1,).
        Must be monotonically increasing.
        The number of x-bins will be ``nx = len(x_bins) - 1``.
    y_bins : array-like
        Bin edges for the DataArray values with shape (ny+1,).
        Must be monotonically increasing.
        The number of y-bins will be ``ny = len(y_bins) - 1``.
    normalization : bool, optional
        If 'none', returns the raw histogram.
        By default, the function normalize the histogram by its global maximum ('max').
        Log-normalization ('log') is also available.

    Returns
    -------
    xarray.DataArray
        2D histogram of shape ``(ny, nx)``. Dimensions are ``('y', 'x')``, where:

        - ``x``: the bin-center coordinate of ``x_bins`` (length ``nx``)
        - ``y``: the bin-center coordinate of ``y_bins`` (length ``ny``)

        Each element ``out.values[y_i, x_j]`` is the count (or normalized count) of how
        many “series-values” from ``da`` fell into the rectangular bin
        ``x_bins[j] ≤ x_value < x_bins[j+1]`` and
        ``y_bins[i] ≤ data_value < y_bins[i+1]``.

    References
    ----------
    Moritz, D., Fisher, D. (2018).
    Visualizing a Million Time Series with the Density Line Chart
    https://doi.org/10.48550/arXiv.1808.06019
    """
    # Check DataArray name
    if da.name is None or da.name == "":
        raise ValueError("The DataArray must have a name.")

    # Validate x_bins and y_bins
    x_bins = np.asarray(x_bins)
    y_bins = np.asarray(y_bins)
    if x_bins.ndim != 1 or x_bins.size < 2:
        raise ValueError("`x_bins` must be a 1D array with at least two edges.")
    if y_bins.ndim != 1 or y_bins.size < 2:
        raise ValueError("`y_bins` must be a 1D array with at least two edges.")
    if not np.all(np.diff(x_bins) > 0):
        raise ValueError("`x_bins` must be strictly increasing.")
    if not np.all(np.diff(y_bins) > 0):
        raise ValueError("`y_bins` must be strictly increasing.")

    # Verify that `coord` exists as either a dimension or a coordinate
    if coord not in (list(da.coords) + list(da.dims)):
        raise ValueError(f"'{coord}' is not a dimension or coordinate of the DataArray.")
    if coord not in da.dims:
        if da[coord].ndim != 1:
            raise ValueError(f"Coordinate '{coord}' must be 1D. Instead has dimensions {da[coord].dims}")
        x_dim = da[coord].dims[0]
    else:
        x_dim = coord

    # Extract the coordinate array
    x_values = (x_bins[0:-1] + x_bins[1:]) / 2

    # Extract the array (samples, x)
    other_dims = [d for d in da.dims if d != x_dim]
    if len(other_dims) == 1:
        arr = da.transpose(*other_dims, x_dim).to_numpy()
    else:
        arr = da.stack({"sample": other_dims}).transpose("sample", x_dim).to_numpy()  # noqa PD013

    # Define y bins center
    y_center = (y_bins[0:-1] + y_bins[1:]) / 2

    # Prepare the 2D count grid of shape (ny, nx)
    # - ny correspond tot he value of the timeseries at nx points
    nx = len(x_bins) - 1
    ny = len(y_bins) - 1
    nsamples = arr.shape[0]

    # For each (series, x-index), find which y-bin it falls into:
    # - np.searchsorted(y_bins, value) gives the insertion index in y_bins;
    #   --> subtracting 1 yields the bin index.
    # If a value is not in y_bins, searchsorted returns 0, so idx = -1
    # If a valueis NaN, the indices value will be ny
    indices = np.searchsorted(y_bins, arr) - 1  # (samples, nx)

    # Compute unormalized DenseLines grid
    # grid = _create_denseline_grid(
    #     indices=indices,
    #     ny=ny,
    #     nx=nx,
    #     nsamples=nsamples
    # )

    # Compute unormalized DenseLines grid by blocks to avoid running out of memory
    # - Define block size based on available RAM memory
    block = _compute_block_size(ny=ny, nx=nx, dtype=np.float64, safety_margin=4e9)
    list_grid = []
    for i in range(0, nsamples, block):
        block_start_idx = i
        block_end_idx = min(i + block, nsamples)
        block_indices = indices[block_start_idx:block_end_idx, :]
        block_nsamples = block_end_idx - block_start_idx
        block_grid = _create_denseline_grid(indices=block_indices, ny=ny, nx=nx, nsamples=block_nsamples)
        list_grid.append(block_grid)

    grid_3d = np.stack(list_grid, axis=0)

    # Finalize sum over samples
    grid = grid_3d.sum(axis=0)

    # Normalize grid
    grid = normalize_array(grid, method=normalization)

    # Create DataArray
    name = da.name
    out = xr.DataArray(grid, dims=[name, coord], coords={coord: (coord, x_values), name: (name, y_center)})

    # Mask values which are 0 with NaN
    out = out.where(out > 0)

    # Return 2D histogram
    return out
