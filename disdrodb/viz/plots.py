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

from disdrodb.constants import DIAMETER_DIMENSION, VELOCITY_DIMENSION
from disdrodb.l2.empirical_dsd import get_drop_average_velocity
from disdrodb.utils.time import ensure_sample_interval_in_seconds, regularize_dataset

####-------------------------------------------------------------------------------------------------------
#### N(D) visualizations


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
    if variable_name == "drop_number_concentration":
        return {
            "ylabel": "N(D) [m-3 mm-1]",
            "cbar_label": "N(D) [m-3 mm-1]",
            "title": "Drop number concentration (N(D))",
        }
    if variable_name in ["raw_particle_counts", "raw_drop_counts", "drop_counts"]:
        variable_name = variable_name.replace("_", " ").capitalize()
        return {
            "ylabel": f"{variable_name} [#]",
            "cbar_label": f"{variable_name} [#]",
            "title": f"{variable_name} distribution",
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


def _check_has_diameter_dims(da):
    if DIAMETER_DIMENSION not in da.dims:
        raise ValueError(f"The DataArray must have dimension '{DIAMETER_DIMENSION}'.")
    if "diameter_bin_width" not in da.coords:
        raise ValueError("The DataArray must have coordinate 'diameter_bin_width'.")
    return da


def _get_nd_variable(xr_obj, variable=None):
    """Get N(D) or drop counts variable from xarray object.

    Parameters
    ----------
    xr_obj : xr.Dataset or xr.DataArray
        Input xarray object.
    variable : str, optional
         Variable name to extract from the xarray object.
        If xr_obj is a DataArray, will return the DataArray and its name directly'.
        If xr_obj is a Dataset, if None, will search for candidate variables in order:
        ['drop_number_concentration', 'drop_counts', 'raw_drop_counts', 'raw_particle_counts'].

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
        variable_name = da.name
    else:
        # xr.Dataset provided
        if variable is None:
            # Search for candidate variables
            l1_variables = ["raw_particle_counts"]
            l2e_variables = ["drop_number_concentration", "drop_counts", "raw_drop_counts"]
            candidate_variables = [*l2e_variables, *l1_variables]
            for var in candidate_variables:
                if var in xr_obj:
                    variable = var
                    break
            if variable is None:
                raise ValueError(
                    f"None of the candidate variables {candidate_variables} found in the dataset. "
                    "Please specify the variable explicitly.",
                )
        elif variable not in xr_obj:
            raise ValueError(f"The dataset does not include {variable=}.")

        da = xr_obj[variable]
        variable_name = variable

    if VELOCITY_DIMENSION in da.dims:
        raise ValueError("N(D) must not have the velocity dimension.")
    da = _check_has_diameter_dims(da)
    return da, variable_name


def plot_nd(xr_obj, variable=None, cmap=None, norm=None, yscale="linear", ax=None):
    """Plot drop number concentration N(D) or drop counts timeseries.

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

    Returns
    -------
    matplotlib axes or plot object
    """
    da_nd, variable_name = _get_nd_variable(xr_obj, variable=variable)
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

    # Regularize input
    da_nd = da_nd.compute()
    da_nd = da_nd.disdrodb.regularize()

    # Set 0 values to np.nan
    da_nd = da_nd.where(da_nd > 0)

    # Define cmap an norm
    if cmap is None:
        cmap = plt.get_cmap("Spectral_r").copy()

    vmin = da_nd.min().item()
    norm = LogNorm(vmin, None) if norm is None else norm

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


def plot_nd_quicklook(
    ds,
    # Plot layout
    hours_per_slice=5,
    max_rows=6,
    aligned=True,
    verbose=True,
    # Spectrum options
    variable="drop_number_concentration",
    cbar_label="N(D) [# m⁻³ mm⁻¹]",
    cmap=None,
    norm=None,
    d_lim=(0.3, 5),
    # Colorbar options
    cbar_as_legend=True,
    # Rain type options
    add_rain_type=None,
    rain_type_colors=None,
    # Add Dm and sigma_m
    add_dm=True,
    add_sigma_m=True,
    # Rain rate options
    add_r=True,
    r_lim=(0.1, 200),
    r_scale="log",
    r_color="black",
    r_alpha=1,
    r_linewidth=1.8,
    r_linestyle=":",
    # Figure options
    dpi=300,
):
    """Display multi-rows quicklook of N(D)."""
    from pycolorbar.utils.mpl_legend import add_fancybox, get_tightbbox_position

    # Colormap & normalization
    if cmap is None:
        cmap = plt.get_cmap("Spectral_r").copy()
        cmap.set_under("none")
    if norm is None:
        norm = LogNorm(vmin=1, vmax=10_000)

    # Compute event Rmax, Ptot, and event duration
    r_max = ds["R"].max().item()
    p_tot = ds["P"].sum().item()

    # Calculate event duration
    duration = ds.disdrodb.end_time - ds.disdrodb.start_time

    # ---------------------------
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

    # Print info on event quicklook
    if verbose:
        print("=== N(D) Event Quicklook ===")
        print(f"Dataset time span : {t_start} → {t_end}")
        print(f"Slice length      : {hours_per_slice} h")
        print(f"Plotted slices    : {n_slices}/{n_total_slices}")
        if n_total_slices > max_rows:
            last_plotted_end = time_bins[max_rows]
            print(f"Unplotted period  : {last_plotted_end} → {t_end}")

    # Regularize dataset to match bin start_time and end_time
    sample_interval = ensure_sample_interval_in_seconds(ds["sample_interval"].to_numpy()).item()
    ds = regularize_dataset(
        ds,
        freq=f"{sample_interval}s",
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

    # ------------------------------------------------------------
    #### - Define figure with GridSpec
    # - If cbar_as_legend=False: reserve extra row for colorbar
    # - If cbar_as_legend=True: no extra row needed
    fig = plt.figure(figsize=(14, 2.8 * n_slices), dpi=dpi)

    if cbar_as_legend:
        # No extra row for colorbar
        gs = GridSpec(
            nrows=n_slices,
            ncols=1,
            figure=fig,
            hspace=0.15,
        )
        axes = [fig.add_subplot(gs[i, 0]) for i in range(n_slices)]
    else:
        # Extra row for colorbar at bottom
        # Scale colorbar row height based on number of subplots
        # - More subplots = relatively smaller colorbar row
        cbar_height_ratio = 0.2 if n_slices == 3 else 0.15

        height_ratios = [1] * n_slices + [cbar_height_ratio]
        gs = GridSpec(
            nrows=n_slices + 1,
            ncols=1,
            figure=fig,
            height_ratios=height_ratios,
            hspace=0.15,
        )
        axes = [fig.add_subplot(gs[i, 0]) for i in range(n_slices)]

    # ---------------------------------------------------------------
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
            y="diameter_bin_center",
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
                linewidth=1.2,
                label="$D_m$",
            )

        #### - Overlay sigma_m
        if add_sigma_m:
            (ds_slice["sigma_m"] * 2).plot(
                ax=ax,
                x="time",
                color="black",
                linestyle="--",
                linewidth=1.2,
                label=r"$2\sigma_m$",
            )

        # Remove xarray default title
        ax.set_title("")

        # Add axis labels - remove ylabel from all individual axes
        ax.set_xlabel("")
        ax.set_ylabel("")

        #### - Add rain rate on secondary axis
        if add_r:
            ax_r = ax.twinx()
            ds_slice["R"].plot(
                ax=ax_r,
                x="time",
                color=r_color,
                alpha=r_alpha,
                linewidth=r_linewidth,
                linestyle=r_linestyle,
                label="R",
            )
            # Always remove xarray default title
            ax_r.set_title("")
            ax_r.set_yscale(r_scale)
            # Display playnumbers instead of scientific notation
            ax_r.set_ylim(r_lim)
            yticks = ax_r.get_yticks()
            ytick_labels = [f"{t:g}" for t in yticks]
            ax_r.set_yticks(yticks)
            ax_r.set_yticklabels(ytick_labels)

            # Enforce ylim
            ax_r.set_ylim(r_lim)
            # Remove ylabel from all individual axes
            ax_r.set_ylabel("")
            ax_r.tick_params(axis="y", labelcolor=r_color)

            # Add horizontal reference lines at 1, 10, and 100 mm/h
            for r_ref in [1, 10, 100]:
                ax_r.axhline(y=r_ref, color="gray", alpha=0.2, linewidth=1, linestyle="-", zorder=0)

        ax.set_ylim(*d_lim)

        #### - Add rain type strip at the top of the subplot as inset axes
        if add_rain_type and add_rain_type in ds_slice:
            # Get rain_type values for this time slice
            rain_type = ds_slice[add_rain_type]

            # Define colors for rain types (0: transparent, 1: stratiform, 2: convective)
            if rain_type_colors is None:
                rain_type_colors = {
                    0: "none",  # No precipitation
                    1: "dodgerblue",  # Stratiform
                    2: "orangered",  # Convective
                }

            # Create inset axes at the top (sharing x-axis with main plot)
            # [x0, y0, width, height] in axes coordinates
            ax_rain = ax.inset_axes([0, 0.95, 1, 0.05], sharex=ax)

            # Define colormap and norm
            colors = [rain_type_colors.get(i, "white") for i in range(3)]
            cmap_rain = ListedColormap(colors)
            norm_rain = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], cmap_rain.N)

            # Plot 1-pixel-high strip
            t = rain_type["time"].to_numpy()
            dt = np.diff(t)  # timedelta64
            t_edges = np.concatenate(
                [
                    [t[0] - dt[0] / 2],
                    t[:-1] + dt / 2,
                    [t[-1] + dt[-1] / 2],
                ],
            )

            ax_rain.pcolormesh(
                t_edges,
                [0, 1],
                rain_type.to_numpy()[np.newaxis, :],
                cmap=cmap_rain,
                norm=norm_rain,
                shading="flat",
            )
            # Add 'axis' line
            ax_rain.axhline(
                y=0,
                color="black",
                linewidth=0.8,
                alpha=1.0,
            )

            # Remove ticks and ticklabels
            ax_rain.set_yticks([])
            ax_rain.xaxis.set_visible(False)
            for spine in ax_rain.spines.values():
                spine.set_visible(False)

    axes[n_slices - 1].set_xlabel("Time (UTC)", fontsize=12)

    # ---------------------------------------------------------------
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
        fontsize=14,
        fontweight="bold",
        loc="center",
    )

    # Add left title with Rmax and Ptot
    left_title_str = f"$R_{{max}}$={r_max:.1f} mm/h  $P_{{tot}}$={p_tot:.1f} mm"
    axes[0].set_title(
        left_title_str,
        fontsize=13,
        loc="left",
    )

    # Add right title with event duration
    axes[0].set_title(
        duration_str,
        fontsize=13,
        loc="right",
    )

    # --------------------------------------------------------------
    #### - Add centered y-labels in the middle of the figure (closer to axes)
    fig.text(
        0.09,
        0.5,
        "Diameter [mm]",
        va="center",
        rotation="vertical",
        fontsize=15,
    )

    if add_r:
        fig.text(
            0.94,
            0.5,
            "Rain rate [mm h$^{-1}$]",
            va="center",
            rotation="vertical",
            fontsize=15,
            color=r_color,
        )

    # --------------------------------------------------------------
    #### - Add legend
    # Collect legend handles from both axes
    handles, labels = ax.get_legend_handles_labels()

    if add_r:
        handles_r, labels_r = ax_r.get_legend_handles_labels()
        handles += handles_r
        labels += labels_r

    axes[0].legend(
        handles,
        labels,
        loc="upper left",
        bbox_to_anchor=(0, 0.98),
        fontsize=12,
        frameon=True,
        fancybox=False,
        edgecolor="black",
    )

    # --------------------------------------------------------------
    #### - Add colorbar
    if cbar_as_legend:
        # Add colorbar as a legend in the last subplot with background box
        cax = axes[-1].inset_axes([0.73, 0.70, 0.25, 0.12])  # [x, y, width, height] in axes coords

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
        cbar.ax.set_xlabel(cbar_label, fontsize=12, labelpad=8)
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
        cbar.set_label(cbar_label, fontsize=11)

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


def plot_spectrum_evolution(ds, legend_variables=None, legend_ncol=1, xlim=None, ylim=None, **plot_kwargs):
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
    # Define legend formatting
    # --> Define decimals per variable
    decimals = {}
    if legend_variables is not None:
        for var in legend_variables:
            if var not in ds:
                raise KeyError(f"Variable '{var}' not found in dataset")

            values = ds[var].to_numpy()

            # Remove NaNs
            values = values[np.isfinite(values)]

            # Integer-like → 0 decimals
            if np.allclose(values, np.round(values)):
                decimals[var] = 0
            else:
                decimals[var] = 2

    # Loop over time
    for i in range(ds.sizes["time"]):
        ds_i = ds.isel(time=i)

        # Define figure
        fig, ax = plt.subplots()

        # Plot spectrum
        plot_spectrum(ds_i, ax=ax, **plot_kwargs)

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
    title="Drop Spectrum",
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
        Title of the plot. Default is 'Drop Spectrum'.
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

    # Sum over time dimension if still present
    # - Unless FacetGrid options in plot_kwargs
    if "time" in drop_number.dims and not is_facetgrid:
        drop_number = drop_number.sum(dim="time")

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
    figsize=(8, 4),
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
