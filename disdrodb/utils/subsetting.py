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
"""This module contains functions for subsetting and aligning DISDRODB products."""

import numpy as np
from xarray.core.utils import either_dict_or_kwargs

from disdrodb.constants import DIAMETER_DIMENSION, VELOCITY_DIMENSION


def is_1d_non_dimensional_coord(xr_obj, coord):
    """Checks if a coordinate is a 1d, non-dimensional coordinate."""
    if coord not in xr_obj.coords:
        return False
    if xr_obj[coord].ndim != 1:
        return False
    is_1d_dim_coord = xr_obj[coord].dims[0] == coord
    return not is_1d_dim_coord


def _get_dim_of_1d_non_dimensional_coord(xr_obj, coord):
    """Get the dimension of a 1D non-dimension coordinate."""
    if not is_1d_non_dimensional_coord(xr_obj, coord):
        raise ValueError(f"'{coord}' is not a dimension or a 1D non-dimensional coordinate.")
    dim = xr_obj[coord].dims[0]
    return dim


def _get_dim_isel_on_non_dim_coord_from_isel(xr_obj, coord, isel_indices):
    """Get dimension and isel_indices related to a 1D non-dimension coordinate.

    Parameters
    ----------
    xr_obj : (xr.Dataset, xr.DataArray)
        A xarray object.
    coord : str
        Name of the coordinate wishing to subset with .sel
    isel_indices : (str, int, float, list, np.array)
        Coordinate indices wishing to be selected.

    Returns
    -------
    dim : str
        Dimension related to the 1D non-dimension coordinate.
    isel_indices : (int, list, slice)
        Indices for index-based selection.
    """
    dim = _get_dim_of_1d_non_dimensional_coord(xr_obj, coord)
    return dim, isel_indices


def _get_dim_isel_indices_from_isel_indices(xr_obj, key, indices, method="dummy"):  # noqa
    """Return the dimension and isel_indices related to the dimension position indices of a coordinate."""
    # Non-dimensional coordinate case
    if key not in xr_obj.dims:
        key, indices = _get_dim_isel_on_non_dim_coord_from_isel(xr_obj, coord=key, isel_indices=indices)
    return key, indices


def _get_isel_indices_from_sel_indices(xr_obj, coord, sel_indices, method):
    """Get isel_indices corresponding to sel_indices."""
    da_coord = xr_obj[coord]
    dim = da_coord.dims[0]
    da_coord = da_coord.assign_coords({"isel_indices": (dim, np.arange(0, da_coord.size))})
    da_subset = da_coord.swap_dims({dim: coord}).sel({coord: sel_indices}, method=method)
    isel_indices = da_subset["isel_indices"].data
    return isel_indices


def _get_dim_isel_on_non_dim_coord_from_sel(xr_obj, coord, sel_indices, method):
    """
    Return the dimension and isel_indices related to a 1D non-dimension coordinate.

    Parameters
    ----------
    xr_obj : (xr.Dataset, xr.DataArray)
        A xarray object.
    coord : str
        Name of the coordinate wishing to subset with .sel
    sel_indices : (str, int, float, list, np.array)
        Coordinate values wishing to be selected.

    Returns
    -------
    dim : str
        Dimension related to the 1D non-dimension coordinate.
    isel_indices : numpy.ndarray
        Indices for index-based selection.
    """
    dim = _get_dim_of_1d_non_dimensional_coord(xr_obj, coord)
    isel_indices = _get_isel_indices_from_sel_indices(xr_obj, coord=coord, sel_indices=sel_indices, method=method)
    return dim, isel_indices


def _get_dim_isel_indices_from_sel_indices(xr_obj, key, indices, method):
    """Return the dimension and isel_indices related to values of a coordinate."""
    # Dimension case
    if key in xr_obj.dims:
        if key not in xr_obj.coords:
            raise ValueError(f"Can not subset with disdrodb.sel the dimension '{key}' if it is not also a coordinate.")
        isel_indices = _get_isel_indices_from_sel_indices(xr_obj, coord=key, sel_indices=indices, method=method)
    # Non-dimensional coordinate case
    else:
        key, isel_indices = _get_dim_isel_on_non_dim_coord_from_sel(
            xr_obj,
            coord=key,
            sel_indices=indices,
            method=method,
        )
    return key, isel_indices


def _get_dim_isel_indices_function(func):
    func_dict = {
        "sel": _get_dim_isel_indices_from_sel_indices,
        "isel": _get_dim_isel_indices_from_isel_indices,
    }
    return func_dict[func]


def _subset(xr_obj, indexers=None, func="isel", drop=False, method=None, **indexers_kwargs):
    """Perform selection with isel or isel."""
    # Retrieve indexers
    indexers = either_dict_or_kwargs(indexers, indexers_kwargs, func)
    # Get function returning isel_indices
    get_dim_isel_indices = _get_dim_isel_indices_function(func)
    # Define isel_dict
    isel_dict = {}
    for key, indices in indexers.items():
        key, isel_indices = get_dim_isel_indices(xr_obj, key=key, indices=indices, method=method)
        if key in isel_dict:
            raise ValueError(f"Multiple indexers point to the '{key}' dimension.")
        isel_dict[key] = isel_indices

    # Subset and update area
    xr_obj = xr_obj.isel(isel_dict, drop=drop)
    return xr_obj


def isel(xr_obj, indexers=None, drop=False, **indexers_kwargs):
    """Perform index-based dimension selection."""
    return _subset(xr_obj, indexers=indexers, func="isel", drop=drop, **indexers_kwargs)


def sel(xr_obj, indexers=None, drop=False, method=None, **indexers_kwargs):
    """Perform value-based coordinate selection.

    Slices are treated as inclusive of both the start and stop values, unlike normal Python indexing.
    The disdrodb `sel` method is empowered to:

    - slice by disdrodb-id strings !
    - slice by any xarray coordinate value !

    You can use string shortcuts for datetime coordinates (e.g., '2000-01' to select all values in January 2000).
    """
    return _subset(xr_obj, indexers=indexers, func="sel", drop=drop, method=method, **indexers_kwargs)


def align(*args):
    """Align DISDRODB products over time, velocity and diameter dimensions."""
    list_xr_obj = args

    # Check input
    if len(list_xr_obj) <= 1:
        raise ValueError("At least two xarray object are required for alignment.")

    # Define dimensions used for alignment
    dims_to_align = ["time", DIAMETER_DIMENSION, VELOCITY_DIMENSION]

    # Check which dimensions and coordinates are available across all datasets
    coords = [coord for coord in dims_to_align if all(coord in xr_obj.coords for xr_obj in list_xr_obj)]
    if not coords:
        raise ValueError("No common coordinates found among the input datasets for alignment.")

    # Start with the input datasets
    list_aligned = list(list_xr_obj)

    # Loop over the dimensions which are available
    for coord in coords:
        # Retrieve list of coordinate values
        list_coord_values = [xr_obj[coord].data for xr_obj in list_aligned]

        # Retrieve intersection of coordinates values
        # - np.atleast_1d ensure that the dimension is not dropped if only 1 value
        # - np.intersect1d returns the sorted array of common unique elements
        common_values = list_coord_values[0]
        for coord_values in list_coord_values[1:]:
            common_values = np.intersect1d(common_values, coord_values)
        sel_indices = np.atleast_1d(common_values)

        # Check there are common coordinate values
        if len(sel_indices) == 0:
            raise ValueError(f"No common {coord} values across input objects.")

        # Subset dataset
        new_list_aligned = [sel(xr_obj, {coord: sel_indices}) for xr_obj in list_aligned]
        list_aligned = new_list_aligned

    return list_aligned
