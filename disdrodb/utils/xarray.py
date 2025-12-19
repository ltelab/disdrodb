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
"""Xarray utilities."""

import numpy as np
import xarray as xr
from xarray.core import dtypes

from disdrodb.constants import DIAMETER_COORDS, VELOCITY_COORDS


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


def _np_remap_numeric_array(arr, remapping_dict, fill_value=np.nan):
    # Define conditions
    conditions = [arr == i for i in remapping_dict]
    # Define choices corresponding to conditions
    choices = remapping_dict.values()
    # Apply np.select to transform the array
    return np.select(conditions, choices, default=fill_value)


def _dask_remap_numeric_array(arr, remapping_dict, fill_value=np.nan):
    import dask.array

    return dask.array.map_blocks(_np_remap_numeric_array, arr, remapping_dict, fill_value, dtype=arr.dtype)


def remap_numeric_array(arr, remapping_dict, fill_value=np.nan):
    """Remap the values of a numeric array."""
    if hasattr(arr, "chunks"):
        return _dask_remap_numeric_array(arr, remapping_dict, fill_value=fill_value)
    return _np_remap_numeric_array(arr, remapping_dict, fill_value=fill_value)


def xr_remap_numeric_array(da, remapping_dict, fill_value=np.nan):
    """Remap values of a xr.DataArray."""
    output = da.copy()
    output.data = remap_numeric_array(da.data, remapping_dict, fill_value=fill_value)
    return output


####-------------------------------------------------------------------
#### Unstacking dimension


def _check_coord_handling(coord_handling):
    if coord_handling not in {"keep", "drop", "unstack"}:
        raise ValueError("coord_handling must be one of 'keep', 'drop', or 'unstack'.")
    return coord_handling


def _unstack_coordinates(xr_obj, dim, prefix, suffix):
    # Identify coordinates that share the target dimension
    coords_with_dim = _get_non_dimensional_coordinates(xr_obj, dim=dim)
    ds = xr.Dataset()
    for coord_name in coords_with_dim:
        coord_da = xr_obj[coord_name]
        # Split the coordinate DataArray along the target dimension, drop coordinate and merge
        split_ds = unstack_datarray_dimension(coord_da, coord_handling="drop", dim=dim, prefix=prefix, suffix=suffix)
        ds.update(split_ds)
    return ds


def _handle_unstack_non_dim_coords(ds, source_xr_obj, coord_handling, dim, prefix, suffix):
    # Deal with coordinates sharing the target dimension
    if coord_handling == "keep":
        return ds
    if coord_handling == "unstack":
        ds_coords = _unstack_coordinates(source_xr_obj, dim=dim, prefix=prefix, suffix=suffix)
        ds.update(ds_coords)
    # Remove non dimensional coordinates (unstack and drop coord_handling)
    ds = ds.drop_vars(_get_non_dimensional_coordinates(ds, dim=dim))
    return ds


def _get_non_dimensional_coordinates(xr_obj, dim):
    return [coord_name for coord_name, coord_da in xr_obj.coords.items() if dim in coord_da.dims and coord_name != dim]


def unstack_datarray_dimension(da, dim, coord_handling="keep", prefix="", suffix=""):
    """
    Split a DataArray along a specified dimension into a Dataset with separate prefixed and suffixed variables.

    Parameters
    ----------
    da : xarray.DataArray
        The DataArray to split.
    dim : str
        The dimension along which to split the DataArray.
    coord_handling : str, optional
        Option to handle coordinates sharing the target dimension.
        Choices are 'keep', 'drop', or 'unstack'. Defaults to 'keep'.
    prefix : str, optional
        String to prepend to each new variable name.
    suffix : str, optional
        String to append to each new variable name.

    Returns
    -------
    xarray.Dataset
        A Dataset with each variable split along the specified dimension.
        The Dataset variables are named  "{prefix}{name}{suffix}{dim_value}".
        Coordinates sharing the target dimension are handled based on `coord_handling`.
    """
    # Retrieve DataArray name
    name = da.name
    coord_handling = _check_coord_handling(coord_handling)

    # Unstack variables
    ds = da.to_dataset(dim=dim)
    rename_dict = {dim_value: f"{prefix}{name}{suffix}{dim_value}" for dim_value in list(ds.data_vars)}
    ds = ds.rename_vars(rename_dict)
    # Deal with coordinates sharing the target dimension
    return _handle_unstack_non_dim_coords(
        ds=ds,
        source_xr_obj=da,
        coord_handling=coord_handling,
        dim=dim,
        prefix=prefix,
        suffix=suffix,
    )


####--------------------------------------------------------------------------
#### Fill Values Utilities


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
    return xr_obj.drop_vars(DIAMETER_COORDS, errors="ignore")


def remove_velocity_coordinates(xr_obj):
    """Drop velocity coordinates from xarray object."""
    return xr_obj.drop_vars(VELOCITY_COORDS, errors="ignore")
