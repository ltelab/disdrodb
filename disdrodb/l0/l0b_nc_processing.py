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
"""Functions to process DISDRODB raw netCDF files into DISDRODB L0B netCDF files."""

import copy
import logging

import numpy as np
import xarray as xr

from disdrodb.l0.l0b_processing import finalize_dataset
from disdrodb.l0.standards import (
    get_bin_coords_dict,
    get_data_range_dict,
    get_nan_flags_dict,
    get_valid_names,
    get_valid_values_dict,
    get_valid_variable_names,
)
from disdrodb.utils.logger import (
    # log_warning,
    # log_debug,
    log_error,
    log_info,
)

logger = logging.getLogger(__name__)
####--------------------------------------------------------------------------.
#### L0B Raw netCDFs Preprocessing


def _check_dict_names_validity(dict_names, sensor_name):
    """Check dict_names dictionary values validity."""
    valid_names = get_valid_names(sensor_name)
    keys = np.array(list(dict_names.keys()))
    values = np.array(list(dict_names.values()))
    # Get invalid keys
    invalid_keys = keys[np.isin(values, valid_names, invert=True)]
    if len(invalid_keys) > 0:
        # Report invalid keys and raise error
        invalid_dict = {k: dict_names[k] for k in invalid_keys}
        msg = f"The following dict_names values are not valid: {invalid_dict}"
        log_error(logger=logger, msg=msg, verbose=False)
        raise ValueError(msg)
    return None


def _get_dict_names_variables(dict_names, sensor_name):
    """Get DISDRODB variables specified in dict_names."""
    possible_variables = get_valid_variable_names(sensor_name)
    dictionary_names = list(dict_names.values())
    variables = [name for name in dictionary_names if name in possible_variables]
    return variables


def _get_missing_variables(ds, dict_names, sensor_name):
    """Get list of missing variables in the dataset."""
    expected_vars = set(_get_dict_names_variables(dict_names, sensor_name))
    dataset_vars = set(ds.data_vars)
    missing_vars = expected_vars.difference(dataset_vars)
    return missing_vars


def rename_dataset(ds, dict_names):
    """Rename Dataset variables, coordinates and dimensions."""
    # Get dataset variables, coordinates and dimensions of the dataset
    ds_vars = list(ds.data_vars)
    ds_dims = list(ds.dims)
    ds_coords = list(ds.coords)
    # Possible keys
    possible_keys = ds_vars + ds_coords + ds_dims
    # Get keys that are dimensions but not coordinates
    rename_dim_keys = [dim for dim in ds_dims if dim not in ds_coords]
    # Get rename keys (coords + variables)
    rename_keys = [k for k in possible_keys if k not in rename_dim_keys]
    # Get rename dictionary
    # - Remove keys which are missing from the dataset
    rename_dict = {k: v for k, v in dict_names.items() if k in rename_keys}
    # Rename dataset
    ds = ds.rename(rename_dict)
    # Rename dimensions
    rename_dim_dict = {k: v for k, v in dict_names.items() if k in rename_dim_keys}
    ds = ds.rename_dims(rename_dim_dict)
    return ds


def subset_dataset(ds, dict_names, sensor_name):
    """Subset Dataset with expected variables."""
    # Get valid variable names
    possible_variables = get_valid_variable_names(sensor_name)
    # Get variables availables in the dict_names and dataset
    dataset_variables = list(ds.data_vars)
    dictionary_names = list(dict_names.values())
    # Get subset variables
    subset_variables = []
    for var in dataset_variables:
        if var in dictionary_names and var in possible_variables:
            subset_variables.append(var)
    # Subset the dataset
    ds = ds[subset_variables]
    return ds


def add_dataset_missing_variables(ds, missing_vars, sensor_name):
    """Add missing Dataset variables as nan DataArrays."""
    from disdrodb.l0.standards import get_variables_dimension

    # Get dimension of each variables
    var_dims_dict = get_variables_dimension(sensor_name)
    # Attach a nan DataArray to the Dataset for each missing variable
    for var in missing_vars:
        # Get variable dimension
        dims = var_dims_dict[var]
        # Retrieve expected shape
        expected_shape = [ds.dims[dim] for dim in dims]
        # Create DataArray
        arr = np.zeros(expected_shape) * np.nan
        da = xr.DataArray(arr, dims=dims)
        # Attach to dataset
        ds[var] = da
    return ds


def preprocess_raw_netcdf(ds, dict_names, sensor_name):
    """This function preprocess raw netCDF to improve compatibility with DISDRODB standards.

    This function checks validity of the dict_names, rename and subset the data accordingly.
    If some variables specified in the dict_names are missing, it adds a NaN DataArray !

    Parameters
    ----------
    ds : xr.Dataset
        Raw netCDF to be converted to DISDRODB standards.
    dict_names : dict
        Dictionary mapping raw netCDF variables/coordinates/dimension names
        to DISDRODB standards.
    sensor_name : str
        Sensor name.

    Returns
    -------
    ds : xr.Dataset
        xarray Dataset with DISDRODB-compliant variable naming conventions.

    """
    # Check variable_dict has valid values
    # - Check valid DISDRODB variables + dimensions + coords
    _check_dict_names_validity(dict_names=dict_names, sensor_name=sensor_name)

    # Rename dataset variables and coordinates
    ds = rename_dataset(ds=ds, dict_names=dict_names)

    # Subset dataset with expected variables
    ds = subset_dataset(ds=ds, dict_names=dict_names, sensor_name=sensor_name)

    # If missing variables, infill with NaN array
    missing_vars = _get_missing_variables(ds, dict_names, sensor_name)
    if len(missing_vars) > 0:
        ds = add_dataset_missing_variables(ds=ds, missing_vars=missing_vars, sensor_name=sensor_name)

    # Update the coordinates for (diameter and velocity)
    coords = get_bin_coords_dict(sensor_name)
    ds = ds.assign_coords(coords)

    # Return dataset
    return ds


def replace_custom_nan_flags(ds, dict_nan_flags, verbose=False):
    """Set values corresponding to nan_flags to np.nan.

    This function must be used in a reader, if necessary.

    Parameters
    ----------
    df : xr.Dataset
        Input xarray dataset
    dict_nan_flags : dict
        Dictionary with nan flags value to set as np.nan

    Returns
    -------
    xr.Dataset
        Dataset without nan_flags values.
    """
    # Loop over the needed variable, and replace nan_flags values with np.nan
    for var, nan_flags in dict_nan_flags.items():
        # If the variable is in the dataframe
        if var in ds:
            # Get occurrence of nan_flags
            is_a_nan_flag = ds[var].isin(nan_flags)
            n_nan_flags_values = np.sum(is_a_nan_flag.data)
            if n_nan_flags_values > 0:
                msg = f"In variable {var}, {n_nan_flags_values} values were nan_flags and were replaced to np.nan."
                log_info(logger=logger, msg=msg, verbose=verbose)
                # Replace with np.nan
                ds[var] = ds[var].where(~is_a_nan_flag)
    # Return dataset
    return ds


def replace_nan_flags(ds, sensor_name, verbose):
    """Set values corresponding to nan_flags to np.nan.

    Parameters
    ----------
    ds : xr.Dataset
        Input xarray dataset
    dict_nan_flags : dict
        Dictionary with nan flags value to set as np.nan
    verbose : bool
        Whether to verbose the processing.

    Returns
    -------
    xr.Dataset
        Dataset without nan_flags values.
    """
    # Get dictionary of nan flags
    dict_nan_flags = get_nan_flags_dict(sensor_name)
    # Replace nan flags with nan
    ds = replace_custom_nan_flags(ds, dict_nan_flags, verbose=verbose)
    return ds


def set_nan_outside_data_range(ds, sensor_name, verbose):
    """Set values outside the data range as np.nan.

    Parameters
    ----------
    ds : xr.Dataset
        Input xarray dataset
    sensor_name : str
        Name of the sensor.
    verbose : bool
        Whether to verbose the processing.

    Returns
    -------
    xr.Dataset
        Dataset without values outside the expected data range.
    """
    # Get dictionary of data_range
    dict_data_range = get_data_range_dict(sensor_name)
    # Loop over the variable with a defined data_range
    for var, data_range in dict_data_range.items():
        # If the variable is in the dataframe
        if var in ds:
            # Get min and max value
            min_val = data_range[0]
            max_val = data_range[1]
            # Check within data range or already np.nan
            is_valid = (ds[var] >= min_val) & (ds[var] <= max_val) | np.isnan(ds[var])
            # If there are values outside the data range, set to np.nan
            n_invalid = np.sum(~is_valid.data)
            if n_invalid > 0:
                msg = f"{n_invalid} {var} values were outside the data range and were set to np.nan."
                log_info(logger=logger, msg=msg, verbose=verbose)
                ds[var] = ds[var].where(is_valid)  # set not valid to np.nan
    # Return dataset
    return ds


def set_nan_invalid_values(ds, sensor_name, verbose):
    """Set invalid (class) values to np.nan.

    Parameters
    ----------
    ds : xr.Dataset
        Input xarray dataset
    sensor_name : str
        Name of the sensor.
    verbose : bool
        Whether to verbose the processing.

    Returns
    -------
    xr.Dataset
        Dataset without invalid values.
    """
    # Get dictionary of valid values
    dict_valid_values = get_valid_values_dict(sensor_name)
    # Loop over the variable with a defined data_range
    for var, valid_values in dict_valid_values.items():
        # If the variable is in the dataframe
        if var in ds:
            # Get array with occurrence of correct values (or already np.nan)
            is_valid_values = ds[var].isin(valid_values) | np.isnan(ds[var])
            # If invalid values are present, replace with np.nan
            n_invalid_values = np.sum(~is_valid_values.data)
            if n_invalid_values > 0:
                msg = f"{n_invalid_values} {var} values were invalid and were replaced to np.nan."
                log_info(logger=logger, msg=msg, verbose=verbose)
                ds[var] = ds[var].where(is_valid_values)  # set not valid to np.nan

    # Return dataset
    return ds


def create_l0b_from_raw_nc(
    ds,
    dict_names,
    ds_sanitizer_fun,
    sensor_name,
    verbose,
    attrs,
):
    """Convert a raw xr.Dataset into a DISDRODB L0B netCDF.

    Parameters
    ----------
    ds : xr.Dataset
        xr.Dataset
    dict_names : dict
        Dictionary mapping raw netCDF variables/coordinates/dimension names
        to DISDRODB standards.
    ds_sanitizer_fun : function
        Sanitizer function to do ad-hoc processing of the xr.Dataset.
    attrs: dict
        Global metadata to attach as global attributes to the xr.Dataset.
    sensor_name : str
        Name of the sensor.
    verbose : bool
        Whether to verbose the processing.


    Returns
    -------
    xr.Dataset
        L0B xr.Dataset
    """
    # Preprocess netcdf
    ds = preprocess_raw_netcdf(ds=ds, dict_names=dict_names, sensor_name=sensor_name)

    # Add CRS and geolocation information
    attrs = copy.deepcopy(attrs)
    coords = {}
    geolocation_vars = ["latitude", "longitude", "altitude"]
    for var in geolocation_vars:
        if var not in ds:
            coords[var] = attrs[var]
        _ = attrs.pop(var)
    ds = ds.assign_coords(coords)

    # Add global attributes
    ds.attrs = attrs

    # Apply dataset sanitizer function
    ds = ds_sanitizer_fun(ds)

    # Replace nan flags values with np.nans
    ds = replace_nan_flags(ds, sensor_name=sensor_name, verbose=verbose)

    # Set values outside the data range to np.nan
    ds = set_nan_outside_data_range(ds, sensor_name=sensor_name, verbose=verbose)

    # Replace invalid values with np.nan
    ds = set_nan_invalid_values(ds, sensor_name=sensor_name, verbose=verbose)

    # Finalize dataset
    ds = finalize_dataset(ds, sensor_name=sensor_name)

    # Return dataset
    return ds
