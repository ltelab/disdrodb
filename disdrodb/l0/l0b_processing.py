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
    get_coords_attrs_dict,
    get_data_range_dict,
    get_dims_size_dict,
    get_l0b_cf_attrs_dict,
    get_l0b_encodings_dict,
    get_raw_array_dims_order,
    get_raw_array_nvalues,
    get_time_encoding,
    set_disdrodb_attrs,
)
from disdrodb.utils.directories import create_directory, remove_if_exists
from disdrodb.utils.logger import (
    # log_warning,
    # log_debug,
    log_error,
    log_info,
)

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
        if counts[idx_delimiter] == 0:
            split_str = None
        else:
            split_str = valid_delims[idx_delimiter]
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

        Examples:
        - OTT Parsivel spectrum [v1d1 ... v1d32, v2d1, ..., v2d32]
        --> dims_order = ["diameter_bin_center", "velocity_bin_center"]
        - Thies LPM spectrum [v1d1 ... v20d1, v1d2, ..., v20d2]
        --> dims_order = ["velocity_bin_center", "diameter_bin_center"]
    dims_size_dict : dict
        Dictionary with the number of bins for each dimension.
        For OTT_Parsivel:
        {"diameter_bin_center": 32, "velocity_bin_center": 32}
        For This_LPM
        {"diameter_bin_center": 22, "velocity_bin_center": 20}
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
    dims = ["time"] + dims_order
    # Retrieve reshaping dimensions as function of dimension order
    reshape_dims = [n_timesteps] + [dims_size_dict[dim] for dim in dims_order]
    try:
        arr = arr.reshape(reshape_dims)
    except Exception as e:
        msg = f"Impossible to reshape the raw_spectrum matrix. The error is: \n {e}"
        log_error(logger=logger, msg=msg, verbose=False)
        raise ValueError(msg)
    return arr, dims


def retrieve_l0b_arrays(
    df: pd.DataFrame,
    sensor_name: str,
    verbose: bool = False,
) -> dict:
    """Retrieves the L0B data matrix.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    sensor_name : str
        Name of the sensor

    Returns
    -------
    dict
        Dictionary with data arrays.

    """

    msg = " - Retrieval of L0B data arrays started."
    log_info(logger=logger, msg=msg, verbose=verbose)
    # ----------------------------------------------------------.
    # Check L0 raw field availability
    _check_raw_fields_available(df=df, sensor_name=sensor_name)

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
        # - This applies i.e for OTT_Parsivel* and Thies_LPM
        # - This does not apply to RD_80
        if key == "raw_drop_number" and len(dims_order) == 2:
            arr, dims = _reshape_raw_spectrum(
                arr=arr,
                dims_order=dims_order,
                dims_size_dict=dims_size_dict,
                n_timesteps=n_timesteps,
            )
        else:
            # Otherwise just define the dimensions of the array
            dims = ["time"] + dims_order

        # Define dictionary to pass to xr.Dataset
        dict_data[key] = (dims, arr)

    # -------------------------------------------------------------------------.
    # Log
    msg = " - Retrieval of L0B data arrays ended."
    log_info(logger=logger, msg=msg, verbose=verbose)
    # Return
    return dict_data


####--------------------------------------------------------------------------.
#### L0B Coords and attributes


def _convert_object_variables_to_string(ds: xr.Dataset) -> xr.Dataset:
    """Convert variables with object dtype to string.

    Parameters
    ----------
    ds : xr.Dataset
        Input dataset.

    Returns
    -------
    xr.Dataset
        Output dataset.
    """
    for var in ds.data_vars:
        if pd.api.types.is_object_dtype(ds[var]):
            ds[var] = ds[var].astype(str)
    return ds


def _set_variable_attributes(ds: xr.Dataset, sensor_name: str) -> xr.Dataset:
    """Set attributes to each xr.Dataset variable.

    Parameters
    ----------
    ds : xr.Dataset
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


def _set_attrs_dict(ds, attrs_dict):
    for var in attrs_dict.keys():
        if var in ds:
            ds[var].attrs.update(attrs_dict[var])
    return ds


def _set_coordinate_attributes(ds):
    # Get attributes dictionary
    attrs_dict = get_coords_attrs_dict(ds)
    # Set attributes
    ds = _set_attrs_dict(ds, attrs_dict)
    return ds


def _set_dataset_attrs(ds, sensor_name):
    """Set variable and coordinates attributes."""
    # - Add netCDF variable attributes
    # --> Attributes: long_name, units, descriptions, valid_min, valid_max
    ds = _set_variable_attributes(ds=ds, sensor_name=sensor_name)
    # - Add netCDF coordinate attributes
    ds = _set_coordinate_attributes(ds=ds)
    #  - Set DISDRODB global attributes
    ds = set_disdrodb_attrs(ds=ds, product="L0B")
    return ds


def add_dataset_crs_coords(ds):
    "Add the CRS coordinate to the xr.Dataset"
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


def _define_dataset_variables(df, sensor_name, verbose):
    """Define DISDRODB L0B netCDF variables."""
    # Preprocess raw_spectrum, diameter and velocity arrays if available
    raw_fields = ["raw_drop_concentration", "raw_drop_average_velocity", "raw_drop_number"]
    if np.any(np.isin(raw_fields, df.columns)):
        # Retrieve dictionary of raw data matrices for xarray Dataset
        data_vars = retrieve_l0b_arrays(df, sensor_name, verbose=verbose)
    else:
        raise ValueError("No raw fields available.")

    # Define other disdrometer 'auxiliary' variables varying over time dimension
    valid_core_fields = [
        "raw_drop_concentration",
        "raw_drop_average_velocity",
        "raw_drop_number",
        "time",
        # longitude and latitude too for moving sensors
    ]
    aux_columns = df.columns[np.isin(df.columns, valid_core_fields, invert=True)]
    aux_data_vars = {column: (["time"], df[column].values) for column in aux_columns}
    data_vars.update(aux_data_vars)

    # Add key "time"
    # - Is dropped in _define_coordinates !
    data_vars["time"] = df["time"].values

    return data_vars


def _define_coordinates(data_vars, attrs, sensor_name):
    """Define DISDRODB L0B netCDF coordinates."""
    # Note: attrs and data_vars are modified in place !

    # - Diameter and velocity
    coords = get_bin_coords_dict(sensor_name=sensor_name)

    # - Geolocation + Time
    geolocation_vars = ["time", "latitude", "longitude", "altitude"]
    for var in geolocation_vars:
        if var in data_vars:
            coords[var] = data_vars[var]
            _ = data_vars.pop(var)
            _ = attrs.pop(var, None)
        else:
            coords[var] = attrs[var]
            _ = attrs.pop(var)
    return coords


def create_l0b_from_l0a(
    df: pd.DataFrame,
    attrs: dict,
    verbose: bool = False,
) -> xr.Dataset:
    """Transform the L0A dataframe to the L0B xr.Dataset.

    Parameters
    ----------
    df : pd.DataFrame
        DISDRODB L0A dataframe.
    attrs : dict
        Station metadata.
    verbose : bool, optional
        Whether to verbose the processing.
        The default is False.

    Returns
    -------
    xr.Dataset
        DISDRODB L0B dataset.

    Raises
    ------
    ValueError
        Error if the DISDRODB L0B xarray dataset can not be created.
    """
    # Retrieve sensor name
    attrs = attrs.copy()
    sensor_name = attrs["sensor_name"]
    # -----------------------------------------------------------.
    # Define Dataset variables and coordinates
    data_vars = _define_dataset_variables(df, sensor_name=sensor_name, verbose=verbose)

    # -----------------------------------------------------------.
    # Define coordinates for xarray Dataset
    # - attrs and data_vars are modified in place !
    coords = _define_coordinates(data_vars, attrs=attrs, sensor_name=sensor_name)

    # -----------------------------------------------------------
    # Create xarray Dataset
    ds = xr.Dataset(
        data_vars=data_vars,
        coords=coords,
        attrs=attrs,
    )
    ds = finalize_dataset(ds, sensor_name=sensor_name)

    # -----------------------------------------------------------
    return ds


####--------------------------------------------------------------------------.
#### L0B netCDF4 Writer


def finalize_dataset(ds, sensor_name):
    """Finalize DISDRODB L0B Dataset."""
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


def sanitize_encodings_dict(encoding_dict: dict, ds: xr.Dataset) -> dict:
    """Ensure chunk size to be smaller than the array shape.

    Parameters
    ----------
    encoding_dict : dict
        Dictionary containing the encoding to write DISDRODB L0B netCDFs.
    ds : xr.Dataset
        Input dataset.

    Returns
    -------
    dict
        Encoding dictionary.
    """
    for var in ds.data_vars:
        shape = ds[var].shape
        chunks = encoding_dict[var]["chunksizes"]
        if chunks is not None:
            chunks = [shape[i] if chunks[i] > shape[i] else chunks[i] for i in range(len(chunks))]
            encoding_dict[var]["chunksizes"] = chunks
    return encoding_dict


def rechunk_dataset(ds: xr.Dataset, encoding_dict: dict) -> xr.Dataset:
    """Coerce the dataset arrays to have the chunk size specified in the encoding dictionary.

    Parameters
    ----------
    ds : xr.Dataset
        Input xarray dataset
    encoding_dict : dict
        Dictionary containing the encoding to write the xarray dataset as a netCDF.

    Returns
    -------
    xr.Dataset
        Output xarray dataset
    """

    for var in ds.data_vars:
        chunks = encoding_dict[var].pop("chunksizes")
        dims = list(ds[var].dims)
        chunks_dict = dict(zip(dims, chunks))
        if chunks is not None:
            ds[var] = ds[var].chunk(chunks_dict)
    return ds


def set_encodings(ds: xr.Dataset, sensor_name: str) -> xr.Dataset:
    """Apply the encodings to the xarray Dataset.

    Parameters
    ----------
    ds : xr.Dataset
        Input xarray dataset.
    sensor_name : str
        Name of the sensor.

    Returns
    -------
    xr.Dataset
        Output xarray dataset.
    """
    # Get encoding dictionary
    encoding_dict = get_l0b_encodings_dict(sensor_name)
    encoding_dict = {k: encoding_dict[k] for k in ds.data_vars}

    # Ensure chunksize smaller than the array shape
    encoding_dict = sanitize_encodings_dict(encoding_dict, ds)

    # Rechunk variables for fast writing !
    # - This pop the chunksize argument from the encoding dict !
    ds = rechunk_dataset(ds, encoding_dict)

    # Set time encoding
    ds["time"].encoding.update(get_time_encoding())

    # Set the variable encodings
    for var in ds.data_vars:
        ds[var].encoding.update(encoding_dict[var])

    return ds


def write_l0b(ds: xr.Dataset, filepath: str, force=False) -> None:
    """Save the xarray dataset into a NetCDF file.

    Parameters
    ----------
    ds : xr.Dataset
        Input xarray dataset.
    filepath : str
        Output file path.
    sensor_name : str
        Name of the sensor.
    force : bool, optional
        Whether to overwrite existing data.
        If True, overwrite existing data into destination directories.
        If False, raise an error if there are already data into destination directories. This is the default.
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
    ds = set_encodings(ds=ds, sensor_name=sensor_name)

    # Write netcdf
    ds.to_netcdf(filepath, engine="netcdf4")


####--------------------------------------------------------------------------.
