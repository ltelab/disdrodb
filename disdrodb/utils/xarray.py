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
import pandas as pd
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
        The default is ``dim_size - 1``.

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


def get_dataset_start_end_time(ds: xr.Dataset):
    """Retrieves dataset starting and ending time.

    Parameters
    ----------
    ds  : xarray.Dataset
        Input dataset

    Returns
    -------
    tuple
        (``starting_time``, ``ending_time``)

    """
    starting_time = ds["time"].to_numpy()[0]
    ending_time = ds["time"].to_numpy()[-1]
    return (starting_time, ending_time)


def regularize_dataset(ds: xr.Dataset, freq: str, time_dim="time", method=None, fill_value=dtypes.NA):
    """
    Regularize a dataset across time dimension with uniform resolution.

    Parameters
    ----------
    ds  : xarray.Dataset
        xarray Dataset.
    time_dim : str, optional
        The time dimension in the xr.Dataset. The default is ``"time"``.
    freq : str
        The ``freq`` string to pass to ``pd.date_range`` to define the new time coordinates.
        Examples: ``freq="2min"``.
    method : str, optional
        Method to use for filling missing timesteps.
        If ``None``, fill with ``fill_value``. The default is ``None``.
        For other possible methods, see https://docs.xarray.dev/en/stable/generated/xarray.Dataset.reindex.html
    fill_value : float, optional
        Fill value to fill missing timesteps. The default is ``dtypes.NA``.

    Returns
    -------
    ds_reindexed  : xarray.Dataset
        Regularized dataset.

    """
    start = ds[time_dim].to_numpy()[0]
    end = ds[time_dim].to_numpy()[-1]
    new_time_index = pd.date_range(start=pd.to_datetime(start), end=pd.to_datetime(end), freq=freq)

    # Regularize dataset and fill with NA values
    ds_reindexed = ds.reindex(
        {"time": new_time_index},
        method=method,  # do not fill gaps
        # tolerance=tolerance,  # mismatch in seconds
        fill_value=fill_value,
    )
    return ds_reindexed


def remove_diameter_coordinates(xr_obj):
    """Drop diameter coordinates from xarray object."""
    from disdrodb import DIAMETER_COORDS

    return xr_obj.drop_vars(DIAMETER_COORDS, errors="ignore")


def remove_velocity_coordinates(xr_obj):
    """Drop velocity coordinates from xarray object."""
    from disdrodb import VELOCITY_COORDS

    return xr_obj.drop_vars(VELOCITY_COORDS, errors="ignore")
