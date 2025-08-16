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
"""DISDRODB Plotting Tools."""
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.colors import LogNorm, Normalize


def plot_nd(ds, var="drop_number_concentration", cmap=None, norm=None):
    """Plot drop number concentration N(D) timeseries."""
    # Check inputs
    if var not in ds:
        raise ValueError(f"{var} is not a xarray Dataset variable!")
    # Check only time and diameter dimensions are specified
    # TODO: DIAMETER_DIMENSION, "time"

    # Select N(D)
    ds_var = ds[[var]].compute()

    # Regularize input
    ds_var = ds_var.disdrodb.regularize()

    # Set 0 values to np.nan
    ds_var = ds_var.where(ds_var[var] > 0)

    # Define cmap an norm
    if cmap is None:
        cmap = plt.get_cmap("Spectral_r").copy()

    vmin = ds_var[var].min().item()
    norm = LogNorm(vmin, None) if norm is None else norm

    # Plot N(D)
    p = ds_var[var].plot.pcolormesh(x="time", norm=norm, cmap=cmap)
    p.axes.set_title("Drop number concentration (N(D))")
    p.axes.set_ylabel("Drop diameter (mm)")
    return p


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
        arr = da.stack({"sample": other_dims}).transpose("sample", x_dim).to_numpy()

    # Define y bins center
    y_center = (y_bins[0:-1] + y_bins[1:]) / 2

    # Prepare the 2D count grid of shape (ny, nx)
    # - ny correspond tot he value of the timeseries at nx points
    nx = len(x_bins) - 1
    ny = len(y_bins) - 1
    nsamples = arr.shape[0]
    grid = np.zeros((ny, nx), dtype=float)

    # For each (series, x-index), find which y-bin it falls into:
    # - np.searchsorted(y_bins, value) gives the insertion index in y_bins;
    #   --> subtracting 1 yields the bin index.
    # If a value is not in y_bins, searchsorted returns 0, so idx = -1
    indices = np.searchsorted(y_bins, arr) - 1  # (samples, nx)

    # Assign 1 when line pass in a bin
    valid = (indices >= 0) & (indices < ny)
    s_idx, x_idx = np.nonzero(valid)
    y_idx = indices[valid]
    grid_3d = np.zeros((nsamples, ny, nx), dtype=int)
    grid_3d[s_idx, y_idx, x_idx] = 1

    # Normalize by columns
    col_sums = grid_3d.sum(axis=1, keepdims=True)
    col_sums[col_sums == 0] = 1  # Avoid division by zero
    grid_3d = grid_3d / col_sums

    # Normalize over samples
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
