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
"""Include functions helping for DISDRODB product manipulations."""

import numpy as np
import xarray as xr
from scipy.interpolate import PchipInterpolator

from disdrodb.constants import DIAMETER_DIMENSION, VELOCITY_DIMENSION
from disdrodb.utils.xarray import unstack_datarray_dimension


def define_diameter_datarray(bounds, dim="diameter_bin_center"):
    """Define diameter DataArray."""
    bounds = np.asarray(bounds, dtype=float)
    diameters_bin_lower = bounds[:-1]
    diameters_bin_upper = bounds[1:]
    diameters_bin_width = diameters_bin_upper - diameters_bin_lower
    diameters_bin_center = diameters_bin_lower + diameters_bin_width / 2
    da = xr.DataArray(
        diameters_bin_center,
        dims=dim,
        coords={
            "diameter_bin_width": (dim, diameters_bin_width),
            "diameter_bin_lower": (dim, diameters_bin_lower),
            "diameter_bin_upper": (dim, diameters_bin_upper),
            dim: (dim, diameters_bin_center),
        },
    )
    return da


def define_velocity_datarray(bounds, dim="velocity_bin_center"):
    """Define velocity DataArray."""
    velocitys_bin_lower = bounds[:-1]
    velocitys_bin_upper = bounds[1:]
    velocitys_bin_width = velocitys_bin_upper - velocitys_bin_lower
    velocitys_bin_center = velocitys_bin_lower + velocitys_bin_width / 2
    da = xr.DataArray(
        velocitys_bin_center,
        dims=dim,
        coords={
            "velocity_bin_width": (dim, velocitys_bin_width),
            "velocity_bin_lower": (dim, velocitys_bin_lower),
            "velocity_bin_upper": (dim, velocitys_bin_upper),
            dim: (dim, velocitys_bin_center),
        },
    )
    return da


def define_diameter_array(diameter_min=0, diameter_max=10, diameter_spacing=0.05):
    """
    Define an array of diameters and their corresponding bin properties.

    Parameters
    ----------
    diameter_min : float, optional
        The minimum diameter value. The default value is 0 mm.
    diameter_max : float, optional
        The maximum diameter value. The default value is 10 mm.
    diameter_spacing : float, optional
        The spacing between diameter values. The default value is 0.05 mm.

    Returns
    -------
    xarray.DataArray
        A DataArray containing the center of each diameter bin, with coordinates for
        the bin width, lower bound, upper bound, and center.

    """
    diameters_bounds = np.arange(diameter_min, diameter_max + diameter_spacing / 2, step=diameter_spacing)
    return define_diameter_datarray(diameters_bounds)


def define_velocity_array(velocity_min=0, velocity_max=10, velocity_spacing=0.05):
    """
    Define an array of velocities and their corresponding bin properties.

    Parameters
    ----------
    velocity_min : float, optional
        The minimum velocity value. The default value is 0 mm.
    velocity_max : float, optional
        The maximum velocity value. The default value is 10 mm.
    velocity_spacing : float, optional
        The spacing between velocity values. The default value is 0.05 mm.

    Returns
    -------
    xarray.DataArray
        A DataArray containing the center of each velocity bin, with coordinates for
        the bin width, lower bound, upper bound, and center.

    """
    velocitys_bounds = np.arange(velocity_min, velocity_max + velocity_spacing / 2, step=velocity_spacing)
    return define_velocity_datarray(velocitys_bounds)


def filter_diameter_bins(ds, minimum_diameter=None, maximum_diameter=None):
    """
    Filter the dataset to include only diameter bins within specified bounds.

    Parameters
    ----------
    ds : xarray.Dataset
        The dataset containing diameter bin data.
    minimum_diameter : float, optional
        The minimum diameter to be included, in millimeters.
        Defaults to the minimum value in `ds["diameter_bin_lower"]`.
    maximum_diameter : float, optional
        The maximum diameter to be included, in millimeters.
        Defaults to the maximum value in `ds["diameter_bin_upper"]`.

    Returns
    -------
    xarray.Dataset
        The filtered dataset containing only the specified diameter bins.
    """
    # Put data into memory
    ds["diameter_bin_lower"] = ds["diameter_bin_lower"].compute()
    ds["diameter_bin_upper"] = ds["diameter_bin_upper"].compute()

    # Initialize default arguments
    if minimum_diameter is None:
        minimum_diameter = ds["diameter_bin_lower"].min().item()
    if maximum_diameter is None:
        maximum_diameter = ds["diameter_bin_upper"].max().item()

    # Select bins which overlap the specified diameters
    valid_indices = np.logical_and(
        ds["diameter_bin_upper"] > minimum_diameter,
        ds["diameter_bin_lower"] < maximum_diameter,
    )
    ds = ds.isel({DIAMETER_DIMENSION: valid_indices})

    if ds.sizes[DIAMETER_DIMENSION] == 0:
        msg = f"Filtering using {minimum_diameter=} removes all diameter bins."
        raise ValueError(msg)
    return ds


def filter_velocity_bins(ds, minimum_velocity=None, maximum_velocity=None):
    """
    Filter the dataset to include only velocity bins within specified bounds.

    Parameters
    ----------
    ds : xarray.Dataset
        The dataset containing velocity bin data.
    minimum_velocity : float, optional
        The minimum velocity to include in the filter, in meters per second.
        Defaults to the minimum value in `ds["velocity_bin_lower"]`.
    maximum_velocity : float, optional
        The maximum velocity to include in the filter, in meters per second.
        Defaults to the maximum value in `ds["velocity_bin_upper"]`.

    Returns
    -------
    xarray.Dataset
        The filtered dataset containing only the specified velocity bins.
    """
    # Put data into memory
    ds["velocity_bin_lower"] = ds["velocity_bin_lower"].compute()
    ds["velocity_bin_upper"] = ds["velocity_bin_upper"].compute()

    # Initialize default arguments
    if minimum_velocity is None:
        minimum_velocity = ds["velocity_bin_lower"].min().item()
    if maximum_velocity is None:
        maximum_velocity = ds["velocity_bin_upper"].max().item()

    # Select bins which overlap the specified velocities
    valid_indices = np.logical_and(
        ds["velocity_bin_upper"] > minimum_velocity,
        ds["velocity_bin_lower"] < maximum_velocity,
    )

    ds = ds.isel({VELOCITY_DIMENSION: valid_indices})
    if ds.sizes[VELOCITY_DIMENSION] == 0:
        msg = f"Filtering using {minimum_velocity=} removes all velocity bins."
        raise ValueError(msg)
    return ds


def get_diameter_bin_edges(ds):
    """Retrieve diameter bin edges."""
    bin_edges = np.append(ds["diameter_bin_lower"].to_numpy(), ds["diameter_bin_upper"].to_numpy()[-1])
    return bin_edges


def get_velocity_bin_edges(ds):
    """Retrieve velocity bin edges."""
    bin_edges = np.append(ds["velocity_bin_lower"].to_numpy(), ds["velocity_bin_upper"].to_numpy()[-1])
    return bin_edges


def convert_from_decibel(x):
    """Convert dB to unit."""
    return np.power(10.0, 0.1 * x)  # x/10


def convert_to_decibel(x):
    """Convert unit to dB."""
    return 10 * np.log10(x)


def unstack_radar_variables(ds):
    """Unstack radar variables."""
    from disdrodb.scattering import RADAR_VARIABLES

    for var in RADAR_VARIABLES:
        if var in ds:
            ds_unstack = unstack_datarray_dimension(ds[var], dim="frequency", prefix="", suffix="_")
            ds.update(ds_unstack)
            ds = ds.drop_vars(var)
    if "frequency" in ds.dims:
        ds = ds.drop_dims("frequency")
    return ds


def get_diameter_coords_dict_from_bin_edges(diameter_bin_edges):
    """Get dictionary with all relevant diameter coordinates."""
    if np.size(diameter_bin_edges) < 2:
        raise ValueError("Expecting at least 2 values defining bin edges.")
    diameter_bin_center = diameter_bin_edges[:-1] + np.diff(diameter_bin_edges) / 2
    diameter_bin_width = np.diff(diameter_bin_edges)
    diameter_bin_lower = diameter_bin_edges[:-1]
    diameter_bin_upper = diameter_bin_edges[1:]
    coords_dict = {
        "diameter_bin_center": (DIAMETER_DIMENSION, diameter_bin_center),
        "diameter_bin_width": (DIAMETER_DIMENSION, diameter_bin_width),
        "diameter_bin_lower": (DIAMETER_DIMENSION, diameter_bin_lower),
        "diameter_bin_upper": (DIAMETER_DIMENSION, diameter_bin_upper),
    }
    return coords_dict


def infer_dsd_sample_dim(da, diameter_dim):
    """Infer the sample dimension.

    Typically "time", but could also be sample_id for example.
    """
    sample_dim_candidates = set(da.dims) - {diameter_dim}
    if len(sample_dim_candidates) != 1:
        raise ValueError(
            f"Could not infer sample dimension from dims={da.dims}."
            f"Please adapt the function for your array layout.",
        )
    sample_dim = list(sample_dim_candidates)[0]  # noqa
    return sample_dim


def _sample_along_dim(xr_obj, dim, n=None, random_state=None):
    """
    Randomly subsample a xarray object along its sample dimension.

    Parameters
    ----------
    da : xarray.DataArray or xarray.Dataset
    n : int or None, optional
        Number of samples to keep. If None, keep all.
    dim : str or None, optional
        Dimension along which to sample. If None, infer it as the only dimension
        different from `coord_dim`.
    random_state : int or None, optional
        Seed for reproducible sampling.

    Returns
    -------
    xr.DataArray
    """
    if n is None:
        return xr_obj

    size = xr_obj.sizes.get(dim, 0)
    if size == 0 or n >= size:
        return xr_obj

    rng = np.random.default_rng(random_state)
    idx = np.sort(rng.choice(size, size=n, replace=False))
    return xr_obj.isel({dim: idx})


def xr_sample(xr_obj, dim, n, random_state=19922207, cat_var=None, category_order=None):
    """Sample an xarray object."""
    # Checks
    if not isinstance(xr_obj, (xr.DataArray, xr.Dataset)):
        raise ValueError("Expecting xarray object.")

    if cat_var is not None and cat_var not in xr_obj:
        raise ValueError(f"{cat_var} is not a xr.Dataset variable.")

    if cat_var is not None and category_order is None:
        category_order = np.unique(xr_obj[cat_var])

    # -------------------------------------------------------------------------.
    # Sampling
    if cat_var is None:
        return _sample_along_dim(
            xr_obj,
            n=n,
            dim=dim,
            random_state=random_state,
        )
    dict_cat = {}
    for i, cat in enumerate(category_order):
        mask = xr_obj[cat_var] == cat
        ds_subset = xr_obj.isel({dim: mask})
        if ds_subset.sizes.get(dim, 0) == 0:
            continue

        ds_subset = _sample_along_dim(
            ds_subset,
            n=n,
            dim=dim,
            random_state=None if random_state is None else random_state + i,
        )
        dict_cat[cat] = ds_subset

    if len(dict_cat) == 0:
        raise ValueError("No data available after category-wise sampling.")

    ds = xr.concat(
        dict_cat.values(),
        dim=dim,
        join="outer",
        coords="minimal",
        compat="override",
    )
    return ds


####---------------------------------------------------------------------------.
#### Resampling low-level functions


def _prepare_resampling_inputs(y_src, d_src, d_dst, dD_src, dD_dst):
    """Sanitize and align source and destination arrays for conservative remapping."""
    # Ensure numpy array input
    y_src = np.asarray(y_src, dtype=float)
    d_src = np.asarray(d_src, dtype=float)
    d_dst = np.asarray(d_dst, dtype=float)
    dD_src = np.asarray(dD_src, dtype=float)
    dD_dst = np.asarray(dD_dst, dtype=float)

    # Ensure only positive values
    y_src = np.where(np.isfinite(y_src), y_src, 0.0)
    y_src = np.clip(y_src, 0.0, None)

    # Identify bins with invalid values
    valid_src = np.isfinite(d_src) & np.isfinite(dD_src) & (dD_src > 0)
    valid_dst = np.isfinite(d_dst) & np.isfinite(dD_dst) & (dD_dst > 0)

    # If not valid values, return None flag --> zero array returned
    if np.count_nonzero(valid_src) == 0 or np.count_nonzero(valid_dst) == 0:
        return None

    # Keep only bin with valid values
    y_src = y_src[valid_src]
    d_src = d_src[valid_src]
    dD_src = dD_src[valid_src]

    # Sort by diameter
    order = np.argsort(d_src)
    y_src = y_src[order]
    d_src = d_src[order]
    dD_src = dD_src[order]

    # Return dictionary with relevant info quality-checked
    return {
        "y_src": y_src,
        "d_src": d_src,
        "d_dst": d_dst,
        "dD_src": dD_src,
        "dD_dst": dD_dst,
        "valid_dst": valid_dst,
        "d_dst_valid": d_dst[valid_dst],
        "dD_dst_valid": dD_dst[valid_dst],
    }


def _assemble_resampled_output(y_dst_valid, d_dst, valid_dst):
    """Populate invalid destination bins with zero after remapping."""
    y_dst = np.zeros_like(d_dst, dtype=float)
    y_dst[valid_dst] = y_dst_valid
    return np.clip(y_dst, 0.0, None)


def _conservative_counts_remapping(n_src, d_src, d_dst, dD_src, dD_dst):
    """First-order conservative remapping of bin-integrated counts between diameter grids.

    Preserves total counts and redistributes them by geometric overlap.
    """
    # Prepare input data
    prepared = _prepare_resampling_inputs(n_src, d_src, d_dst, dD_src, dD_dst)
    if prepared is None:
        return np.zeros_like(d_dst, dtype=float)

    n_src = prepared["y_src"]
    d_src = prepared["d_src"]
    dD_src = prepared["dD_src"]
    d_dst_valid = prepared["d_dst_valid"]
    dD_dst_valid = prepared["dD_dst_valid"]

    # Source edges
    src_left = d_src - 0.5 * dD_src
    src_right = d_src + 0.5 * dD_src

    # Destination edges
    dst_left = d_dst_valid - 0.5 * dD_dst_valid
    dst_right = d_dst_valid + 0.5 * dD_dst_valid

    # Overlap matrix (Ns, N)
    overlap = np.minimum(src_right[:, None], dst_right[None, :]) - np.maximum(src_left[:, None], dst_left[None, :])
    overlap = np.clip(overlap, 0.0, None)

    # Redistributes only fraction of the bin count that overlaps each destination bin
    n_dst_valid = (n_src[:, None] * overlap / dD_src[:, None]).sum(axis=0)

    # Return resampled counts
    return _assemble_resampled_output(n_dst_valid, prepared["d_dst"], prepared["valid_dst"])


def _conservative_density_remapping(y_src, d_src, d_dst, dD_src, dD_dst):
    """First-order conservative remapping of bin densities between diameter grids.

    Preserves total number concentration.
    """
    # Prepare input data
    prepared = _prepare_resampling_inputs(y_src, d_src, d_dst, dD_src, dD_dst)
    if prepared is None:
        return np.zeros_like(d_dst, dtype=float)

    y_src = prepared["y_src"]
    d_src = prepared["d_src"]
    dD_src = prepared["dD_src"]
    d_dst_valid = prepared["d_dst_valid"]
    dD_dst_valid = prepared["dD_dst_valid"]

    # Source edges
    src_left = d_src - 0.5 * dD_src
    src_right = d_src + 0.5 * dD_src

    # Destination edges
    dst_left = d_dst_valid - 0.5 * dD_dst_valid
    dst_right = d_dst_valid + 0.5 * dD_dst_valid

    # Overlap matrix (Ns, Nd)
    overlap = np.minimum(src_right[:, None], dst_right[None, :]) - np.maximum(src_left[:, None], dst_left[None, :])
    overlap = np.clip(overlap, 0.0, None)

    # Integrate density over the overlap length to get destination bin counts
    n_dst = (y_src[:, None] * overlap).sum(axis=0)

    # Divide by destination bin width to go back to density
    y_dst_valid = n_dst / dD_dst_valid

    # Return resampled counts
    return _assemble_resampled_output(y_dst_valid, prepared["d_dst"], prepared["valid_dst"])


def _log_pchip_conservative_density_remapping(y_src, d_src, d_dst, dD_src, dD_dst):
    """Smooth remapping in log-space, scaled to conserve global integrated number."""
    prepared = _prepare_resampling_inputs(y_src, d_src, d_dst, dD_src, dD_dst)
    if prepared is None:
        return np.zeros_like(d_dst, dtype=float)

    y_src = prepared["y_src"]
    d_src = prepared["d_src"]
    dD_src = prepared["dD_src"]
    d_dst_valid = prepared["d_dst_valid"]
    dD_dst_valid = prepared["dD_dst_valid"]

    n_src_total = np.sum(y_src * dD_src)
    if n_src_total <= 0:
        return np.zeros_like(d_dst, dtype=float)

    positive = y_src > 0
    if np.count_nonzero(positive) < 2:
        y_dst_valid = _conservative_density_remapping(y_src, d_src, d_dst_valid, dD_src, dD_dst_valid)
        return _assemble_resampled_output(y_dst_valid, prepared["d_dst"], prepared["valid_dst"])

    log_interp = PchipInterpolator(
        d_src[positive],
        np.log(y_src[positive]),
        extrapolate=False,
    )
    y_dst_valid = np.exp(log_interp(d_dst_valid))
    y_dst_valid = np.where(np.isfinite(y_dst_valid), y_dst_valid, 0.0)

    # Keep signal only where destination bins overlap support of positive source bins.
    src_left = d_src - 0.5 * dD_src
    src_right = d_src + 0.5 * dD_src
    support_left = np.min(src_left[positive])
    support_right = np.max(src_right[positive])
    dst_left = d_dst_valid - 0.5 * dD_dst_valid
    dst_right = d_dst_valid + 0.5 * dD_dst_valid
    overlaps_support = (dst_right > support_left) & (dst_left < support_right)

    # Fill boundary half-bins that overlap support but lie outside interpolation center range.
    first_center = d_src[positive][0]
    last_center = d_src[positive][-1]
    first_value = y_src[positive][0]
    last_value = y_src[positive][-1]
    left_boundary = overlaps_support & (d_dst_valid < first_center)
    right_boundary = overlaps_support & (d_dst_valid > last_center)
    y_dst_valid[left_boundary] = first_value
    y_dst_valid[right_boundary] = last_value
    y_dst_valid[~overlaps_support] = 0.0

    n_dst_total = np.sum(y_dst_valid * dD_dst_valid)
    if n_dst_total > 0:
        y_dst_valid *= n_src_total / n_dst_total
    return _assemble_resampled_output(y_dst_valid, prepared["d_dst"], prepared["valid_dst"])


def _log_pchip_conservative_counts_remapping(n_src, d_src, d_dst, dD_src, dD_dst):
    """Smooth conservative remapping of bin-integrated counts via count density."""
    prepared = _prepare_resampling_inputs(n_src, d_src, d_dst, dD_src, dD_dst)
    if prepared is None:
        return np.zeros_like(d_dst, dtype=float)

    # Smooth interpolation method should be applied to an intensive quantity (not extensive)
    # --> Convert counts to density
    y_src = prepared["y_src"] / prepared["dD_src"]

    # Resample density
    y_dst_valid = _log_pchip_conservative_density_remapping(
        y_src,
        prepared["d_src"],
        prepared["d_dst_valid"],
        prepared["dD_src"],
        prepared["dD_dst_valid"],
    )
    # Convert back to counts
    n_dst_valid = y_dst_valid * prepared["dD_dst_valid"]
    return _assemble_resampled_output(n_dst_valid, prepared["d_dst"], prepared["valid_dst"])


def resample_counts(
    da_counts,
    d_src,
    d_dst,
    dim,
    new_dim,
    dD_src,
    dD_dst,
    method="constant",
):
    """Conservatively resample bin-integrated counts between diameter grids.

    Parameters
    ----------
    da_counts : xr.DataArray
        Bin-integrated counts defined on ``dim``.
    d_src : xr.DataArray
        Source diameter centers associated with ``da_counts``.
    d_dst : xr.DataArray
        Destination diameter centers associated with ``new_dim``.
    dim : str
        Source diameter dimension.
    new_dim : str
        Destination diameter dimension.
    dD_src : xr.DataArray
        Source bin widths defined on ``dim``.
    dD_dst : xr.DataArray
        Destination bin widths defined on ``new_dim``.
    method : {"constant", "log_pchip"} or callable, optional
        Remapping strategy used within ``xr.apply_ufunc``.

        - ``"constant"``: first-order conservative remapping in count space.
          Use this for coarsening and when strict robustness is preferred.
        - ``"log_pchip"``: smooth conservative remapping obtained by applying
          log-PCHIP to the implied count density ``counts / dD`` and then
          converting back to counts. Use this mainly for refinement to finer bins.

        If coarsening is detected (destination has fewer bins than source), the
        method is forced to ``"constant"`` to avoid shape artifacts.

        If callable, it must have signature
        ``f(n_src, d_src, d_dst, dD_src, dD_dst)`` and return remapped destination counts.

    Returns
    -------
    xr.DataArray
        Resampled counts on ``new_dim``.

    Notes
    -----
    The remapping preserves total counts over the remapped support and assumes
    counts are piecewise-uniform inside each source bin.
    """
    if len(d_dst) < len(d_src) and method != "constant":  # coarsening
        print("Resampling method set to 'constant' for coarsening.")
        method = "constant"

    da_counts = da_counts.where(da_counts > 0, 0)

    methods = {
        "constant": _conservative_counts_remapping,
        "log_pchip": _log_pchip_conservative_counts_remapping,
    }
    if callable(method):
        resampling_func = method
    else:
        if method not in methods:
            valid_methods = ", ".join(methods)
            msg = f"Unknown {method!r}. Valid options are: {valid_methods}."
            raise ValueError(msg)
        resampling_func = methods[method]

    da_counts_new = xr.apply_ufunc(
        resampling_func,
        da_counts,
        d_src,
        d_dst,
        dD_src,
        dD_dst,
        input_core_dims=[[dim], [dim], [new_dim], [dim], [new_dim]],
        output_core_dims=[[new_dim]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[float],
    )

    name = da_counts.name
    da_counts_new = da_counts_new.assign_coords({new_dim: d_dst})
    da_counts_new.name = name if name is not None else "counts"
    return da_counts_new


def resample_density(
    da_density,
    d_src,
    d_dst,
    dim,
    new_dim,
    dD_src,
    dD_dst,
    method="log_pchip",
):
    """Conservatively resample a density between diameter grids.

    The remapping preserves integrated quantity (``sum(density * bin_width)``)
    over the remapped support.

    Parameters
    ----------
    da_density : xr.DataArray
        Density defined per unit diameter.
    d_src : xr.DataArray
        Source diameter centers (can be 2D: time, D).
    d_dst : xr.DataArray
        Destination diameter centers (1D).
    dim : str
        Source diameter dimension.
    new_dim : str
        Destination dimension name.
    dD_src : xr.DataArray
        Source bin widths (same dim as dim).
    dD_dst : xr.DataArray
        Destination bin widths (same dim as new_dim).
    method : str or callable
        Remapping strategy used within ``xr.apply_ufunc``.

        If str, available methods are:

        - ``"constant"``: first-order conservative remapping (piecewise
          constant inside each source bin). Use this for coarsening and when
          strict robustness is preferred.
        - ``"log_pchip"``: conservative smooth remapping using PCHIP in
          log-density space. Use this mainly for refinement to finer bins.

        If coarsening is detected (destination has fewer bins than source), the
        method is forced to ``"constant"`` to avoid shape artifacts.

        If callable, it must have signature
        ``f(y_src, d_src, d_dst, dD_src, dD_dst)`` and return remapped destination density.

    Returns
    -------
    xr.DataArray
        Remapped density conserving integrated amount.

    Notes
    -----
    For ``drop_number`` counts already integrated per bin, use
    :func:`resample_drop_counts` instead of this function.
    """
    if len(d_dst) < len(d_src) and method != "constant":  # coarsening
        print("Resampling method set to 'constant' for coarsening.")
        method = "constant"

    da_density = da_density.where(da_density > 0, 0)

    methods = {
        "constant": _conservative_density_remapping,
        "log_pchip": _log_pchip_conservative_density_remapping,
    }
    if callable(method):
        resampling_func = method
    else:
        if method not in methods:
            valid_methods = ", ".join(methods)
            msg = f"Unknown {method!r}. Valid options are: {valid_methods}."
            raise ValueError(msg)
        resampling_func = methods[method]

    da_density_new = xr.apply_ufunc(
        resampling_func,
        da_density,
        d_src,
        d_dst,
        dD_src,
        dD_dst,
        input_core_dims=[[dim], [dim], [new_dim], [dim], [new_dim]],
        output_core_dims=[[new_dim]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[float],
    )

    # Assign coordinate
    name = da_density.name
    da_density_new = da_density_new.assign_coords({new_dim: d_dst})
    da_density_new.name = name if name is not None else "drop_number_concentration"
    return da_density_new


####---------------------------------------------------------------------------.
#### Resampling wrappers for n(D) and N(D)


def resample_drop_counts(drop_counts, diameter_bin_edges, method="constant"):
    """Conservatively resample bin-integrated drop counts to new diameter bins.

    Parameters
    ----------
    drop_counts : xr.DataArray
        Bin-integrated counts (for example ``drop_counts`` or ``drop_number``) defined on
        ``diameter_bin_center`` and carrying ``diameter_bin_width`` coordinates.
        Additional dimensions (for example ``time`` or ``velocity_bin_center``)
        are supported and vectorized.
    diameter_bin_edges : array-like
        Destination diameter bin edges. The destination bins do not need to be
        aligned with source bins.
    method : {"constant", "log_pchip"} or callable, optional
        Remapping method passed to :func:`resample_counts`.

        - ``"constant"``: first-order conservative remapping in count space.
          Recommended for coarsening and robust for any grid change.
        - ``"log_pchip"``: smooth conservative remapping based on the implied
          count density. Recommended for refinement/interpolation to finer bins
          when the source spectrum is sufficiently resolved.

        If coarsening is requested, :func:`resample_counts` automatically
        switches to ``"constant"``.

    Returns
    -------
    xr.DataArray
        Resampled counts on destination ``diameter_bin_center`` with updated
        ``diameter_bin_width``, ``diameter_bin_lower`` and
        ``diameter_bin_upper`` coordinates.
    """
    da_dst_d_bin_centers = define_diameter_datarray(diameter_bin_edges, dim="d_new")
    da_resampled = resample_counts(
        da_counts=drop_counts,
        d_src=drop_counts["diameter_bin_center"],
        d_dst=da_dst_d_bin_centers,
        dim="diameter_bin_center",
        new_dim="d_new",
        dD_src=drop_counts["diameter_bin_width"],
        dD_dst=da_dst_d_bin_centers["diameter_bin_width"],
        method=method,
    )
    da_resampled = da_resampled.rename({"d_new": "diameter_bin_center"})
    da_resampled.name = drop_counts.name if drop_counts.name is not None else "drop_counts"
    return da_resampled


def resample_drop_number_concentration(
    drop_number_concentration,
    diameter_bin_edges,
    method="log_pchip",
):
    """Resample drop number concentration ``N(D)`` to new diameter bins.

    The remapping is conservative with respect to integrated number
    concentration (that is, ``sum(N(D) * dD)``).

    Parameters
    ----------
    drop_number_concentration : xr.DataArray
        Drop number concentration defined per unit diameter on
        ``diameter_bin_center`` and carrying ``diameter_bin_width``.
    diameter_bin_edges : array-like
        Destination diameter bin edges.
    method : {"constant", "log_pchip"} or callable, optional
        Remapping method passed to :func:`resample_density`.

        - ``"constant"``: first-order conservative remapping. Recommended for
          coarsening and robust for any grid change.
        - ``"log_pchip"``: smooth conservative remapping in log-space.
          Recommended for refinement/interpolation to finer bins when positive
          source spectra are sufficiently resolved.

        If coarsening is requested, :func:`resample_density` automatically
        switches to ``"constant"``.

    Returns
    -------
    xr.DataArray
        Resampled ``N(D)`` on destination ``diameter_bin_center`` with updated
        diameter-bin coordinates.
    """
    da_dst_d_bin_centers = define_diameter_datarray(diameter_bin_edges, dim="d_new")
    da_resampled = resample_density(
        da_density=drop_number_concentration,
        d_src=drop_number_concentration["diameter_bin_center"],
        d_dst=da_dst_d_bin_centers,
        dim="diameter_bin_center",
        new_dim="d_new",
        dD_src=drop_number_concentration["diameter_bin_width"],
        dD_dst=da_dst_d_bin_centers["diameter_bin_width"],
        method=method,
    )
    da_resampled = da_resampled.rename({"d_new": "diameter_bin_center"})
    return da_resampled


####---------------------------------------------------------------------------.
#### Double Moment Normalization utilities


def remap_to_diameter(
    da,
    d_src,
    d_dst,
    dim,
    new_dim,
    method="linear",
):
    """Remap DataArray from source to destination diameter coordinate.

    Parameters
    ----------
    da : xr.DataArray
        DataArray with dimension `dim` and typically another dim (e.g., time).
    d_src : xr.DataArray
        Source diameter coordinate (can be 2D, e.g., D/Dm (time, D)).
        Must share dimensions with da.
    d_dst : xr.DataArray
        1D target coordinate.
    dim : str
        Original diameter dimension.
    new_dim : str
        Name of output diameter dimension.
    method : {"linear", "pchip"}
        Interpolation method used for remapping.

    Returns
    -------
    xr.DataArray
    """
    if method not in {"linear", "pchip"}:
        msg = f"Unknown {method!r}. Valid options are: linear, pchip."
        raise ValueError(msg)

    def _interp_1d_linear(x_new, x_old, y_old):
        valid = np.isfinite(x_old) & np.isfinite(y_old)
        if np.count_nonzero(valid) == 0:
            return np.full_like(x_new, np.nan, dtype=float)

        x_old = np.asarray(x_old[valid], dtype=float)
        y_old = np.asarray(y_old[valid], dtype=float)
        order = np.argsort(x_old)
        x_old = x_old[order]
        y_old = y_old[order]

        # np.interp expects increasing xp. Duplicate xp values are collapsed.
        x_old_unique, unique_idx = np.unique(x_old, return_index=True)
        y_old_unique = y_old[unique_idx]
        return np.interp(x_new, x_old_unique, y_old_unique, left=np.nan, right=np.nan)

    def _interp_1d_pchip(x_new, x_old, y_old):
        valid = np.isfinite(x_old) & np.isfinite(y_old)
        if np.count_nonzero(valid) < 2:
            return _interp_1d_linear(x_new, x_old, y_old)

        x_old = np.asarray(x_old[valid], dtype=float)
        y_old = np.asarray(y_old[valid], dtype=float)
        order = np.argsort(x_old)
        x_old = x_old[order]
        y_old = y_old[order]

        # PCHIP requires strictly increasing x values.
        x_old_unique, unique_idx = np.unique(x_old, return_index=True)
        y_old_unique = y_old[unique_idx]
        if x_old_unique.size < 2:
            return _interp_1d_linear(x_new, x_old_unique, y_old_unique)

        pchip = PchipInterpolator(x_old_unique, y_old_unique, extrapolate=False)
        return pchip(x_new)

    interp_func = _interp_1d_linear if method == "linear" else _interp_1d_pchip

    da_out = xr.apply_ufunc(
        interp_func,
        d_dst,
        d_src,
        da,
        input_core_dims=[[new_dim], [dim], [dim]],
        output_core_dims=[[new_dim]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[float],
    )

    da_out = da_out.assign_coords({new_dim: d_dst})
    return da_out


def compute_normalized_dsd_datarray(
    ds,
    Nc="Nw",
    Dc="Dm",
    d_min=0,
    d_max=6,
    d_step=0.001,
    method="linear",
    diameter_dim="diameter_bin_center",
    dsd_var="drop_number_concentration",
):
    """Compute normalized DSD and remap to regular D/Dc dimension."""
    # Infer sample dimension (typically time)
    sample_id = infer_dsd_sample_dim(ds[dsd_var], diameter_dim=diameter_dim)

    # Compute Normalized DSD and normalized diameter
    ds["N(D)/Nc"] = ds[dsd_var] / ds[Nc]
    ds["D/Dc"] = ds[diameter_dim] / ds[Dc]
    ds["dD/Dc"] = ds["diameter_bin_width"] / ds[Dc]

    ds["D/Dc"] = ds["D/Dc"].transpose(diameter_dim, sample_id)
    ds["N(D)/Nc"] = ds["N(D)/Nc"].transpose(diameter_dim, sample_id)

    # Define normalized diameter coordinate
    da_normalized_diameter = define_diameter_datarray(np.arange(d_min, d_max, d_step), dim="D/Dc")

    # Map N(D)/Nc value for each D/Dc to regular D/Dc array
    da_dsd_norm = remap_to_diameter(
        da=ds["N(D)/Nc"],
        d_src=ds["D/Dc"],
        d_dst=da_normalized_diameter,
        dim=diameter_dim,
        new_dim="D/Dc",
        method=method,
    )

    da_dsd_norm = da_dsd_norm.assign_coords({"diameter_bin_width": da_normalized_diameter["diameter_bin_width"]})
    da_dsd_norm.name = "N(D)/Nc"
    return da_dsd_norm


# def interpolate_drop_number_concentration(drop_number_concentration, diameter_bin_edges, method="linear"):
#     """Interpolate drop number concentration N(D) DataArray to high resolution diameter bins.

#     This should be done only for visualization purposes as it change the distribution moments.
#     """
#     diameters_bin_center = diameter_bin_edges[:-1] + np.diff(diameter_bin_edges) / 2

#     da = drop_number_concentration.interp(coords={"diameter_bin_center": diameters_bin_center}, method=method)
#     coords_dict = get_diameter_coords_dict_from_bin_edges(diameter_bin_edges)
#     da = da.assign_coords(coords_dict)
#     return da
