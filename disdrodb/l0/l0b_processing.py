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
"""Functions to process DISDRODB L0A files into DISDRODB L0B netCDF files."""

import logging
import os

import numpy as np
import pandas as pd
import xarray as xr

from disdrodb.l0.check_standards import (
    _check_raw_fields_available,
    check_l0b_standards,
)
from disdrodb.l0.standards import (
    # get_valid_coordinates_names,
    get_bin_coords_dict,
    get_data_range_dict,
    get_dims_size_dict,
    get_l0b_cf_attrs_dict,
    get_l0b_encodings_dict,
    get_raw_array_dims_order,
    get_raw_array_nvalues,
)
from disdrodb.utils.attrs import (
    set_coordinate_attributes,
    set_disdrodb_attrs,
)
from disdrodb.utils.directories import create_directory, remove_if_exists
from disdrodb.utils.encoding import set_encodings
from disdrodb.utils.logger import (
    # log_warning,
    # log_debug,
    log_info,
)
from disdrodb.utils.time import ensure_sorted_by_time

logger = logging.getLogger(__name__)


####--------------------------------------------------------------------------.
#### L0B Raw Precipitation Spectrum Processing


def infer_split_str(string: str) -> str:
    """Infer the delimiter inside a string.

    Parameters
    ----------
    string : str
        Input string.

    Returns
    -------
    str
        Inferred delimiter.
    """
    if not isinstance(string, str):
        raise TypeError("infer_split_str expects a string")
    if len(string) > 0:
        valid_delims = [";", ","]  # here we can add others if needed [|, ... ]
        counts = np.array([string.count(delim) for delim in valid_delims])
        idx_delimiter = np.argmax(counts)
        # If don't find the delimiter, set to None
        # --> The array will not be split, and then raise an error later on
        split_str = None if counts[idx_delimiter] == 0 else valid_delims[idx_delimiter]
    else:
        split_str = None  # ''.split(None) output []
    return split_str


def _replace_empty_strings_with_zeros(values):
    values[np.char.str_len(values) == 0] = "0"
    return values


def _format_string_array(string: str, n_values: int) -> np.array:
    """Split a string with multiple numbers separated by a delimiter into an 1D array.

        e.g. : _format_string_array("2,44,22,33", 4) will return [ 2. 44. 22. 33.]

    If empty string ("") --> Return an arrays of zeros
    If the list length is not n_values -> Return an arrays of np.nan

    The function strip potential delimiters at start and end before splitting.

    Parameters
    ----------
    string : str
        Input string
    n_values : int
        Expected length of the output array.

    Returns
    -------
    np.array
        array of float
    """
    split_str = infer_split_str(string)
    values = np.array(string.strip(split_str).split(split_str))

    # -------------------------------------------------------------------------.
    ## Assumptions !!!
    # If empty list --> Assume no precipitation recorded. Return an arrays of zeros
    if len(values) == 0:
        values = np.zeros(n_values)
        return values

    # -------------------------------------------------------------------------.
    # If the length is not as expected --> Assume data corruption
    # --> Return an array with nan
    if len(values) != n_values:
        values = np.zeros(n_values) * np.nan
    else:
        # Ensure string type
        values = values.astype("str")
        # Replace '' with 0
        values = _replace_empty_strings_with_zeros(values)
        # Replace "-9.999" with 0
        values = np.char.replace(values, "-9.999", "0")
        # Cast values to float type
        # --> Note: the disk encoding is specified in the l0b_encodings.yml
        values = values.astype(float)
    return values


def _reshape_raw_spectrum(
    arr: np.array,
    dims_order: list,
    dims_size_dict: dict,
    n_timesteps: int,
) -> np.array:
    """Reshape the raw spectrum to a 2D+time array.

    The array has dimensions ["time"] + dims_order

    Parameters
    ----------
    arr : np.array
        Input array.
    dims_order : list
        The order of dimension in the raw spectrum.

    Examples
    --------
        - OTT PARSIVEL spectrum [v1d1 ... v1d32, v2d1, ..., v2d32]
        --> dims_order = ["diameter_bin_center", "velocity_bin_center"]
        - Thies LPM spectrum [v1d1 ... v20d1, v1d2, ..., v20d2]
        --> dims_order = ["velocity_bin_center", "diameter_bin_center"]
    dims_size_dict : dict
        Dictionary with the number of bins for each dimension.
        For PARSIVEL and PARSIVEL2:
        {"diameter_bin_center": 32, "velocity_bin_center": 32}
        For LPM
        {"diameter_bin_center": 22, "velocity_bin_center": 20}
        For PWS100
        {"diameter_bin_center": 34, "velocity_bin_center": 34}
    n_timesteps : int
        Number of timesteps.

    Returns
    -------
    np.array
        Output array.

    Raises
    ------
    ValueError
        Impossible to reshape the raw_spectrum matrix
    """
    # Define output dimensions
    dims = ["time", *dims_order]
    # Retrieve reshaping dimensions as function of dimension order
    reshape_dims = [n_timesteps] + [dims_size_dict[dim] for dim in dims_order]
    try:
        arr = arr.reshape(reshape_dims)
    except Exception as e:
        msg = f"Impossible to reshape the raw_spectrum matrix. The error is: \n {e}"
        raise ValueError(msg)
    return arr, dims


def retrieve_l0b_arrays(
    df: pd.DataFrame,
    sensor_name: str,
    logger=None,
    verbose: bool = False,
) -> dict:
    """Retrieves the L0B data matrix.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe
    sensor_name : str
        Name of the sensor

    Returns
    -------
    dict
        Dictionary with data arrays.

    """
    msg = "Retrieval of L0B data arrays started."
    log_info(logger=logger, msg=msg, verbose=verbose)
    # ----------------------------------------------------------.
    # Check L0 raw field availability
    _check_raw_fields_available(df=df, sensor_name=sensor_name, logger=logger, verbose=verbose)

    # Retrieve the number of values expected for each array
    n_values_dict = get_raw_array_nvalues(sensor_name=sensor_name)

    # Retrieve the dimension order for each raw array
    # - For the raw spectrum (raw_drop_number), it controls the way data are reshaped !
    dims_order_dict = get_raw_array_dims_order(sensor_name=sensor_name)

    # Retrieve number of bins for each dimension
    dims_size_dict = get_dims_size_dict(sensor_name=sensor_name)

    # Retrieve number of timesteps
    n_timesteps = df.shape[0]

    # Retrieve available arrays
    dict_data = {}
    unavailable_keys = []
    for key, n_values in n_values_dict.items():
        # Check key is available in dataframe
        if key not in df.columns:
            unavailable_keys.append(key)
            continue

        # Ensure is a string
        df_series = df[key].astype(str)

        # Get a numpy array for each row and then stack
        list_arr = df_series.apply(_format_string_array, n_values=n_values)
        arr = np.stack(list_arr, axis=0)

        # Retrieve dimensions
        dims_order = dims_order_dict[key]

        # For key='raw_drop_number', if 2D spectrum, reshape to 2D matrix
        # Example:
        # - This applies i.e for PARSIVEL*, LPM, PWS100
        # - This does not apply to RD80
        if key == "raw_drop_number" and len(dims_order) == 2:
            arr, dims = _reshape_raw_spectrum(
                arr=arr,
                dims_order=dims_order,
                dims_size_dict=dims_size_dict,
                n_timesteps=n_timesteps,
            )
        else:
            # Otherwise just define the dimensions of the array
            dims = ["time", *dims_order]

        # Define dictionary to pass to xr.Dataset
        dict_data[key] = (dims, arr)

    # -------------------------------------------------------------------------.
    # Log
    msg = "Retrieval of L0B data arrays ended."
    log_info(logger=logger, msg=msg, verbose=verbose)
    # Return
    return dict_data


####--------------------------------------------------------------------------.
#### L0B Coords and attributes


def _convert_object_variables_to_string(ds: xr.Dataset) -> xr.Dataset:
    """Convert variables with ``object`` dtype to ``string``.

    Parameters
    ----------
    ds  : xarray.Dataset
        Input dataset.

    Returns
    -------
    xarray.Dataset
        Output dataset.
    """
    for var in ds.data_vars:
        if pd.api.types.is_object_dtype(ds[var]):
            ds[var] = ds[var].astype(str)
    return ds


def _set_variable_attributes(ds: xr.Dataset, sensor_name: str) -> xr.Dataset:
    """Set attributes to each ``xr.Dataset`` variable.

    Parameters
    ----------
    ds  : xarray.Dataset
        Input dataset.
    sensor_name : str
        Name of the sensor.

    Returns
    -------
    ds
        xr.Dataset.
    """
    # Retrieve attributes dictionaries
    cf_attrs_dict = get_l0b_cf_attrs_dict(sensor_name)
    data_range_dict = get_data_range_dict(sensor_name)

    # Assign attributes to each variable
    for var in ds.data_vars:
        ds[var].attrs = cf_attrs_dict[var]
        if var in data_range_dict:
            ds[var].attrs["valid_min"] = data_range_dict[var][0]
            ds[var].attrs["valid_max"] = data_range_dict[var][1]
    return ds


def _set_dataset_attrs(ds, sensor_name):
    """Set variable and coordinates attributes."""
    # - Add netCDF variable attributes
    # --> Attributes: long_name, units, descriptions, valid_min, valid_max
    ds = _set_variable_attributes(ds=ds, sensor_name=sensor_name)
    # - Add netCDF coordinate attributes
    ds = set_coordinate_attributes(ds=ds)
    #  - Set DISDRODB global attributes
    ds = set_disdrodb_attrs(ds=ds, product="L0B")
    return ds


def add_dataset_crs_coords(ds):
    """Add the CRS coordinate to the xr.Dataset."""
    # TODO: define CF-compliant CRS !
    # - CF compliant
    # - wkt
    # - add grid_mapping name
    # -->
    # attrs["EPSG"] = 4326
    # attrs["proj4_string"] = "+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs"
    ds = ds.assign_coords({"crs": ["WGS84"]})
    return ds


####--------------------------------------------------------------------------.
#### L0B Raw DataFrame Preprocessing


def _define_dataset_variables(df, sensor_name, logger=None, verbose=False):
    """Define DISDRODB L0B netCDF variables."""
    # Preprocess raw_spectrum, diameter and velocity arrays if available
    raw_fields = ["raw_drop_concentration", "raw_drop_average_velocity", "raw_drop_number"]
    if np.any(np.isin(raw_fields, df.columns)):
        # Retrieve dictionary of raw data matrices for xarray Dataset
        data_vars = retrieve_l0b_arrays(df, sensor_name=sensor_name, logger=logger, verbose=verbose)
    else:
        raise ValueError("No raw fields available.")

    # Define other disdrometer 'auxiliary' variables varying over time dimension
    # - Includes time
    # - Includes longitude and latitude for moving sensors
    valid_core_fields = [
        "raw_drop_concentration",
        "raw_drop_average_velocity",
        "raw_drop_number",
    ]
    aux_columns = df.columns[np.isin(df.columns, valid_core_fields, invert=True)]
    aux_data_vars = {column: (["time"], df[column].to_numpy()) for column in aux_columns}
    data_vars.update(aux_data_vars)
    return data_vars


def create_l0b_from_l0a(
    df: pd.DataFrame,
    metadata: dict,
    logger=None,
    verbose: bool = False,
) -> xr.Dataset:
    """Transform the L0A dataframe to the L0B xr.Dataset.

    Parameters
    ----------
    df : pandas.DataFrame
        DISDRODB L0A dataframe.
        The raw drop number spectrum is reshaped to a 2D(+time) array.
        The raw drop concentration and velocity are reshaped to 1D(+time) arrays.
    metadata : dict
        DISDRODB station metadata.
        To use this function outside the DISDRODB routines, the dictionary must
        contain the fields: ``sensor_name``, ``latitude``, ``longitude``, ``altitude``, ``platform_type``.
    verbose : bool, optional
        Whether to verbose the processing. The default value is ``False``.

    Returns
    -------
    xarray.Dataset
        DISDRODB L0B dataset.

    Raises
    ------
    ValueError
        Error if the DISDRODB L0B xarray dataset can not be created.
    """
    # Retrieve sensor name
    metadata = metadata.copy()
    sensor_name = metadata["sensor_name"]

    # Define Dataset variables and coordinates
    data_vars = _define_dataset_variables(df, sensor_name=sensor_name, logger=logger, verbose=verbose)

    # Create xarray Dataset
    ds = xr.Dataset(data_vars=data_vars)
    ds = finalize_dataset(ds, sensor_name=sensor_name, metadata=metadata)
    return ds


####--------------------------------------------------------------------------.
#### L0B netCDF4 Writer


def set_geolocation_coordinates(ds, metadata):
    """Add geolocation coordinates to dataset."""
    # Assumption
    # - If coordinate is present in L0A, overrides the one specified in the attributes
    # - If a station is fixed, discard the coordinates in the DISDRODB reader !

    # Assign geolocation coordinates to dataset
    coords = ["latitude", "longitude", "altitude"]
    for coord in coords:
        # If coordinate not present, add it from dictionary
        if coord not in ds:
            ds = ds.assign_coords({coord: metadata.pop(coord, np.nan)})
        # Else if set coordinates the variable in the dataset (present in the raw data)
        else:
            ds = ds.set_coords(coord)
            _ = metadata.pop(coord, None)

    # Set -9999 flag value to np.nan
    for coord in coords:
        ds[coord] = xr.where(ds[coord] == -9999, np.nan, ds[coord])

    # Set attributes without geolocation coordinates
    ds.attrs = metadata
    return ds


def finalize_dataset(ds, sensor_name, metadata):
    """Finalize DISDRODB L0B Dataset."""
    # Ensure sorted by time
    ds = ensure_sorted_by_time(ds)

    # Set diameter and velocity bin coordinates
    ds = ds.assign_coords(get_bin_coords_dict(sensor_name=sensor_name))

    # Set geolocation coordinates and attributes
    ds = set_geolocation_coordinates(ds, metadata=metadata)

    # Add dataset CRS coordinate
    ds = add_dataset_crs_coords(ds)

    # Set netCDF dimension order
    ds = ds.transpose("time", "diameter_bin_center", ...)

    # Add netCDF variable and coordinate attributes
    ds = _set_dataset_attrs(ds, sensor_name)

    # Ensure variables with dtype object are converted to string
    ds = _convert_object_variables_to_string(ds)

    # Check L0B standards
    check_l0b_standards(ds)
    return ds


def set_l0b_encodings(ds: xr.Dataset, sensor_name: str):
    """Apply the L0B encodings to the xarray Dataset.

    Parameters
    ----------
    ds  : xarray.Dataset
        Input xarray dataset.
    sensor_name : str
        Name of the sensor.

    Returns
    -------
    xarray.Dataset
        Output xarray dataset.
    """
    encoding_dict = get_l0b_encodings_dict(sensor_name)
    ds = set_encodings(ds=ds, encoding_dict=encoding_dict)
    return ds


def write_l0b(ds: xr.Dataset, filepath: str, force=False) -> None:
    """Save the xarray dataset into a NetCDF file.

    Parameters
    ----------
    ds  : xarray.Dataset
        Input xarray dataset.
    filepath : str
        Output file path.
    sensor_name : str
        Name of the sensor.
    force : bool, optional
        Whether to overwrite existing data.
        If ``True``, overwrite existing data into destination directories.
        If ``False``, raise an error if there are already data into destination directories. This is the default.
    """
    # Create station directory if does not exist
    create_directory(os.path.dirname(filepath))

    # Check if the file already exists
    # - If force=True --> Remove it
    # - If force=False --> Raise error
    remove_if_exists(filepath, force=force)

    # Get sensor name from dataset
    sensor_name = ds.attrs.get("sensor_name")

    # Set encodings
    ds = set_l0b_encodings(ds=ds, sensor_name=sensor_name)

    # Write netcdf
    ds.to_netcdf(filepath, engine="netcdf4")


####--------------------------------------------------------------------------.
