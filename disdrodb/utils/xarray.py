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
"""Xarray utilities."""
import numpy as np
import xarray as xr
from xarray.core import dtypes


def xr_get_last_valid_idx(da_condition, dim, fill_value=None):
    """
    Get the index of the last True value along a specified dimension in an xarray DataArray.

    This function finds the last index along the given dimension where the condition is True.
    If all values are False or NaN along that dimension, the function returns ``fill_value``.

    Parameters
    ----------
    da_condition : xarray.DataArray
        A boolean DataArray where True indicates valid or desired values.
        Should have the dimension specified in `dim`.
    dim : str
        The name of the dimension along which to find the last True index.
    fill_value : int or float
        The fill value when all values are False or NaN along the specified dimension.
        The default value is ``dim_size - 1``.

    Returns
    -------
    last_idx : xarray.DataArray
        An array containing the index of the last True value along the specified dimension.
        If all values are False or NaN, the corresponding entry in `last_idx` will be NaN.

    Notes
    -----
    The function works by reversing the DataArray along the specified dimension and using
    `argmax` to find the first True value in the reversed array. It then calculates the
    corresponding index in the original array. To handle cases where all values are False
    or NaN (and `argmax` would return 0), the function checks if there is any True value
    along the dimension and assigns NaN to `last_idx` where appropriate.

    Examples
    --------
    >>> import xarray as xr
    >>> da = xr.DataArray([[False, False, True], [False, False, False]], dims=["time", "my_dimension"])
    >>> last_idx = xr_get_last_valid_idx(da, "my_dimension")
    >>> print(last_idx)
    <xarray.DataArray (time: 2)>
    array([2., nan])
    Dimensions without coordinates: time

    In this example, for the first time step, the last True index is 2.
    For the second time step, all values are False, so the function returns NaN.

    """
    # Check input is a boolean array
    if not np.issubdtype(da_condition.dtype, np.bool_):
        raise ValueError("`da_condition` must be a boolean DataArray.")

    # Get the size of the 'dim' dimension
    dim_size = da_condition.sizes[dim]

    # Define default fillvalue
    if fill_value is None:
        fill_value = dim_size - 1

    # Reverse the mask along 'dim'
    da_condition_reversed = da_condition.isel({dim: slice(None, None, -1)})

    # Check if there is any True value along the dimension for each slice
    has_true = da_condition.any(dim=dim)

    # Find the first non-zero index in the reversed array
    last_idx_from_end = da_condition_reversed.argmax(dim=dim)

    # Calculate the last True index in the original array
    last_idx = xr.where(
        has_true,
        dim_size - last_idx_from_end - 1,
        fill_value,
    )
    return last_idx


def define_dataarray_fill_value(da):
    """Define the fill value for a numerical xarray.DataArray."""
    if np.issubdtype(da.dtype, np.floating):
        return dtypes.NA
    if np.issubdtype(da.dtype, np.integer):
        if "_FillValue" in da.attrs:
            return da.attrs["_FillValue"]
        if "_FillValue" in da.encoding:
            return da.encoding["_FillValue"]
        return np.iinfo(da.dtype).max
    return None


def define_dataarray_fill_value_dictionary(da):
    """Define fill values for numerical variables and coordinates of a xarray.DataArray.

    Return a dict of fill values:
      - floating → NaN
      - integer → ds[var].attrs["_FillValue"] if present, else np.iinfo(dtype).max
    """
    fill_value_dict = {}
    # Add fill value of DataArray
    fill_value_array = define_dataarray_fill_value(da)
    if fill_value_array is not None:
        fill_value_dict[da.name] = fill_value_array
    # Add fill value of coordinates
    fill_value_dict.update(define_dataset_fill_value_dictionary(da.coords))
    # Return fill value dictionary
    return fill_value_dict


def define_dataset_fill_value_dictionary(ds):
    """Define fill values for numerical variables and coordinates of a xarray.Dataset.

    Return a dict of per-variable fill values:
      - floating --> NaN
      - integer --> ds[var].attrs["_FillValue"] if present, else the maximum allowed number.
    """
    fill_value_dict = {}
    # Retrieve fill values for numerical variables and coordinates
    for var in list(ds.variables):
        array_fill_value = define_dataarray_fill_value(ds[var])
        if array_fill_value is not None:
            fill_value_dict[var] = array_fill_value
    # Return fill value dictionary
    return fill_value_dict


def define_fill_value_dictionary(xr_obj):
    """Define fill values for numerical variables and coordinates of a xarray object.

    Return a dict of per-variable fill values:
      - floating --> NaN
      - integer --> ds[var].attrs["_FillValue"] if present, else the maximum allowed number.
    """
    if isinstance(xr_obj, xr.Dataset):
        return define_dataset_fill_value_dictionary(xr_obj)
    return define_dataarray_fill_value_dictionary(xr_obj)


####-----------------------------------------------------------------------------------
#### Diameter and Velocity Coordinates


def remove_diameter_coordinates(xr_obj):
    """Drop diameter coordinates from xarray object."""
    from disdrodb import DIAMETER_COORDS

    return xr_obj.drop_vars(DIAMETER_COORDS, errors="ignore")


def remove_velocity_coordinates(xr_obj):
    """Drop velocity coordinates from xarray object."""
    from disdrodb import VELOCITY_COORDS

    return xr_obj.drop_vars(VELOCITY_COORDS, errors="ignore")
