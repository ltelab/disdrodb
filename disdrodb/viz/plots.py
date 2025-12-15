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
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psutil
import xarray as xr
from matplotlib.colors import LogNorm, Normalize
from matplotlib.gridspec import GridSpec

from disdrodb.constants import DIAMETER_DIMENSION, VELOCITY_DIMENSION
from disdrodb.l2.empirical_dsd import get_drop_average_velocity
from disdrodb.utils.time import ensure_sample_interval_in_seconds, regularize_dataset

####-------------------------------------------------------------------------------------------------------
#### N(D) visualizations


def _single_plot_nd_distribution(drop_number_concentration, diameter, diameter_bin_width):
    fig, ax = plt.subplots(1, 1)
    ax.bar(
        diameter,
        drop_number_concentration,
        width=diameter_bin_width,
        edgecolor="darkgray",
        color="lightgray",
        label="Data",
    )
    ax.set_title("Drop number concentration (N(D))")
    ax.set_xlabel("Drop diameter (mm)")
    ax.set_ylabel("N(D) [m-3 mm-1]")
    return ax


def _check_has_diameter_dims(da):
    if DIAMETER_DIMENSION not in da.dims:
        raise ValueError(f"The DataArray must have dimension '{DIAMETER_DIMENSION}'.")
    if "diameter_bin_width" not in da.coords:
        raise ValueError("The DataArray must have coordinate 'diameter_bin_width'.")
    return da


def _get_nd_variable(xr_obj, variable):
    if not isinstance(xr_obj, (xr.Dataset, xr.DataArray)):
        raise TypeError("Expecting xarray object as input.")
    if isinstance(xr_obj, xr.Dataset):
        if variable not in xr_obj:
            raise ValueError(f"The dataset do not include {variable=}.")
        xr_obj = xr_obj[variable]
    if VELOCITY_DIMENSION in xr_obj.dims:
        raise ValueError("N(D) must no have the velocity dimension.")
    xr_obj = _check_has_diameter_dims(xr_obj)
    return xr_obj


def plot_nd(xr_obj, variable="drop_number_concentration", cmap=None, norm=None):
    """Plot drop number concentration N(D) timeseries."""
    da_nd = _get_nd_variable(xr_obj, variable=variable)

    # Check only time and diameter dimensions are specified
    if "time" not in da_nd.dims:
        ax = _single_plot_nd_distribution(
            drop_number_concentration=da_nd.isel(velocity_method=0, missing_dims="ignore"),
            diameter=xr_obj["diameter_bin_center"],
            diameter_bin_width=xr_obj["diameter_bin_width"],
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

    # Plot N(D)
    cbar_kwargs = {"label": "N(D) [m-3 mm-1]"}
    p = da_nd.plot.pcolormesh(x="time", norm=norm, cmap=cmap, extend="max", cbar_kwargs=cbar_kwargs)
    p.axes.set_title("Drop number concentration N(D)")
    p.axes.set_ylabel("Drop diameter (mm)")
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
    # R options
    add_r=True,
    r_lim=(0.1, 50),
    r_scale="log",
    r_color="tab:blue",
    r_linewidth=1.2,
):
    """Display multi-rows quicklook of N(D)."""
    # Colormap & normalization
    if cmap is None:
        cmap = plt.get_cmap("Spectral_r").copy()
        cmap.set_under("none")
    if norm is None:
        norm = LogNorm(vmin=1, vmax=10_000)

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
            print(f"Unplotted period  : {last_plotted_end} → {aligned_end}")

    # Regularize dataset to match bin start_time and end_time
    sample_interval = ensure_sample_interval_in_seconds(ds["sample_interval"].to_numpy()).item()
    ds = regularize_dataset(ds, freq=f"{sample_interval}s", start_time=time_bins[0], end_time=time_bins[-1])

    # Define figure
    fig, axes = plt.subplots(
        nrows=n_slices,
        ncols=1,
        figsize=(14, 2.8 * n_slices),
        sharex=False,
        constrained_layout=True,
    )

    if n_slices == 1:
        axes = [axes]

    # Plot each slice
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

        # Overlay Dm
        ds_slice["Dm"].plot(
            ax=ax,
            x="time",
            color="black",
            linestyle="--",
            linewidth=1.2,
            label="Dm",
        )

        # Add axis labels and title
        ax.set_xlabel("")
        ax.set_ylabel("Diameter [mm]")
        ax.set_title(f"{t0:%H:%M} - {t1:%H:%M} UTC")

        if i == 0:
            ax.legend(loc="upper right")

        # Add rain rate on secondary axis
        if add_r:
            ax_r = ax.twinx()
            ds_slice["R"].plot(
                ax=ax_r,
                x="time",
                color=r_color,
                linewidth=r_linewidth,
                label="R",
            )
            ax_r.set_ylim(r_lim)
            ax_r.set_yscale(r_scale)
            ax_r.set_ylabel("Rain rate [mm h$^{-1}$]", color="tab:blue")
            ax_r.tick_params(axis="y", labelcolor="tab:blue")
            ax_r.set_title("")

        ax.set_ylim(*d_lim)

    axes[-1].set_xlabel("Time (UTC)")
    # ---------------------------
    # Shared colorbar
    # ---------------------------
    cbar = fig.colorbar(
        p,
        ax=axes,
        orientation="horizontal",
        pad=0.02,
        fraction=0.03,
        extend="max",
    )
    cbar.set_label(cbar_label)


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
    cmap : Colormap, optional
        Colormap to use. If None, uses 'Spectral_r' with 'under' set to 'none'.
    norm : matplotlib.colors.Normalize, optional
        Normalization for colormap. If None, uses LogNorm with vmin=1.
    extend : {'neither', 'both', 'min', 'max'}, optional
        Whether to draw arrows on the colorbar to indicate out-of-range values.
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
    - If the input DataArray has a time dimension, it is summed over time before plotting
        unless FacetGrid options (e.g., col, row) are specified in plot_kwargs.
    - If FacetGrid options are used, the plot will create a grid of subplots for each time slice.
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
    arr : np.ndarray
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
    np.ndarray
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
    x_bins : array_like of shape (nx+1,)
        Bin edges to bin the coordinate/dimension.
        Must be monotonically increasing.
        The number of x-bins will be ``nx = len(x_bins) - 1``.
    y_bins : array_like of shape (ny+1,)
        Bin edges for the DataArray values.
        Must be monotonically increasing.
        The number of y-bins will be ``ny = len(y_bins) - 1``.
    normalization : bool, optional
        If 'none', returns the raw histogram.
        By default, the function normalize the histogram by its global maximum ('max').
        Log-normalization ('log') is also available.

    Returns
    -------
    xr.DataArray
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
