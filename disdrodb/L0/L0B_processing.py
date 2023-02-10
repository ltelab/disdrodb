#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# -----------------------------------------------------------------------------.
# Copyright (c) 2021-2022 DISDRODB developers
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

# File content
# - Functions to convert L0A Apache parquet files to L0B netCDF files

# -----------------------------------------------------------------------------.
import os
import logging
import numpy as np
import pandas as pd
import xarray as xr
from disdrodb.L0.check_standards import (
    check_sensor_name,
    check_L0B_standards,
    _check_raw_fields_available,
)
from disdrodb.L0.io import _remove_if_exists, _create_directory
from disdrodb.L0.standards import (
    get_diameter_bin_center,
    get_diameter_bin_lower,
    get_diameter_bin_upper,
    get_diameter_bin_width,
    get_velocity_bin_center,
    get_velocity_bin_lower,
    get_velocity_bin_upper,
    get_velocity_bin_width,
    get_raw_field_nbins,
    get_raw_field_dim_order,
    get_raw_spectrum_ndims,
    get_L0B_encodings_dict,
    get_time_encoding,
    get_valid_names,
    get_valid_variable_names,
    get_valid_dimension_names,
    # get_valid_coordinates_names,
    get_coords_attrs_dict,
    set_disdrodb_attrs,
)
from disdrodb.utils.logger import (
    log_info,
    # log_warning,
    # log_debug,
    log_error,
)

logger = logging.getLogger(__name__)


####--------------------------------------------------------------------------.
#### L0B Raw Text Spectrum Processing
# def get_drop_concentration(arr):
#     # TODO
#     logger.info("Computing raw_drop_concentration from raw spectrum.")
#     return arr[:, :, 0]


# def get_drop_average_velocity(arr):
#     # TODO
#     logger.info("Computing raw_drop_average_velocity from raw spectrum.")
#     return arr[:, 0, :]


def infer_split_str(string: str) -> str:
    """Infer the delimeter inside a string.

    Parameters
    ----------
    string : str
        Input string.

    Returns
    -------
    str
        Inferred delimiter.
    """
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


def format_string_array(string: str, n_values: int) -> np.array:
    """Split a string with multiple numbers separated by a delimiter into an 1D array.

        e.g. : format_string_array("2,44,22,33", 4) will return [ 2. 44. 22. 33.]

    If empty string ("") --> Assume no precipitation recorded
    If the list length is not n_values or n_values+1 --> Set np.nan

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
    values = np.array(string.split(split_str))

    # -------------------------------------------------------------------------.
    ## Assumptions !!!
    # If empty list --> Assume no precipitation recorded
    if len(values) == 0:
        values = np.zeros(n_values)
        return values

    # If the length of the array is + 1 than the expected, but the last character of
    #  the string is a delimiter --> Drop the last array value
    if len(values) == (n_values + 1):
        if string[-1] == split_str:
            values = np.delete(values, -1)

    # -------------------------------------------------------------------------.
    # If the length is not as expected --> Assume data corruption
    # --> Return an array with nan
    if len(values) != n_values:
        values = np.zeros(n_values) * np.nan
    else:
        # Ensure string type
        values = values.astype("str")
        # Replace '' with 0
        values[values == ""] = "0"
        # Replace "-9.999" with 0
        values = np.char.replace(values, "-9.999", "0")
        # Cast values to float type
        # --> Note: the disk encoding is specified in the L0B_encodings.yml
        values = values.astype(float)
    return values


def reshape_raw_spectrum_to_2D(
    arr: np.array, n_bins_dict: dict, n_timesteps: int, verbose: bool = False
) -> np.array:
    """Reshape the raw spectrum to 2D.

    Parameters
    ----------
    arr : np.array
        Input array.
    n_bins_dict : dict
        Raw field number of bins.
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

    try:
        arr = arr.reshape(
            n_timesteps,
            n_bins_dict["raw_drop_concentration"],
            n_bins_dict["raw_drop_average_velocity"],
        )
    except Exception as e:
        msg = f"Impossible to reshape the raw_spectrum matrix. The error is: \n {e}"
        log_error(logger=logger, msg=msg, verbose=False)
        raise ValueError(msg)
    return arr


def retrieve_L0B_arrays(
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

    msg = " - Retrieval of L0B data matrix started."
    log_info(logger=logger, msg=msg, verbose=verbose)
    # ----------------------------------------------------------.
    # Check L0 raw field availability
    _check_raw_fields_available(df=df, sensor_name=sensor_name)

    # Retrieve raw fields matrix bins dictionary
    n_bins_dict = get_raw_field_nbins(sensor_name=sensor_name)

    # Retrieve dimension order dictionary
    dims_dict = get_raw_field_dim_order(sensor_name)

    # Retrieve dimension of the raw_drop_number field
    n_dim_spectrum = get_raw_spectrum_ndims(sensor_name)

    # Retrieve number of timesteps
    n_timesteps = df.shape[0]

    # Retrieve available arrays
    dict_data = {}
    unavailable_keys = []
    for key, n_bins in n_bins_dict.items():
        # Check key is available in dataframe
        if key not in df.columns:
            unavailable_keys.append(key)
            continue

        # Ensure is a string
        df_series = df[key].astype(str)

        # Get a numpy array for each row and then stack
        list_arr = df_series.apply(format_string_array, n_values=n_bins)
        arr = np.stack(list_arr, axis=0)

        # For key='raw_drop_number', if 2D ... reshape to 2D matrix
        # - This applies i.e for OTT_Parsivels and ThiesLPM
        # - This does not apply to RD80
        if key == "raw_drop_number" and n_dim_spectrum == 2:
            arr = reshape_raw_spectrum_to_2D(
                arr, n_bins_dict, n_timesteps, verbose=verbose
            )

        # Define dictionary to pass to xr.Dataset
        dims_order = ["time"] + dims_dict[key]
        dict_data[key] = (dims_order, arr)

    # -------------------------------------------------------------------------.
    # Log
    msg = " - Retrieval of L0B data matrices finished."
    log_info(logger=logger, msg=msg, verbose=verbose)
    # Return
    return dict_data


####--------------------------------------------------------------------------.
#### L0B Coords and attributes


def get_bin_coords(sensor_name: str) -> dict:
    """Retrieve diameter (and velocity) bin coordinates.

    Parameters
    ----------
    sensor_name : str
        Name of the sensor.

    Returns
    -------
    dict
        Dictionary with coordinate arrays.
    """

    check_sensor_name(sensor_name=sensor_name)
    coords = {}
    # Retrieve diameter coords
    coords["diameter_bin_center"] = get_diameter_bin_center(sensor_name=sensor_name)
    coords["diameter_bin_lower"] = (
        ["diameter_bin_center"],
        get_diameter_bin_lower(sensor_name=sensor_name),
    )
    coords["diameter_bin_upper"] = (
        ["diameter_bin_center"],
        get_diameter_bin_upper(sensor_name=sensor_name),
    )
    coords["diameter_bin_width"] = (
        ["diameter_bin_center"],
        get_diameter_bin_width(sensor_name=sensor_name),
    )
    # Retrieve velocity coords (if available)
    if get_velocity_bin_center(sensor_name=sensor_name) is not None:
        coords["velocity_bin_center"] = (
            ["velocity_bin_center"],
            get_velocity_bin_center(sensor_name=sensor_name),
        )
        coords["velocity_bin_lower"] = (
            ["velocity_bin_center"],
            get_velocity_bin_lower(sensor_name=sensor_name),
        )
        coords["velocity_bin_upper"] = (
            ["velocity_bin_center"],
            get_velocity_bin_upper(sensor_name=sensor_name),
        )
        coords["velocity_bin_width"] = (
            ["velocity_bin_center"],
            get_velocity_bin_width(sensor_name=sensor_name),
        )
    return coords


def convert_object_variables_to_string(ds: xr.Dataset) -> xr.Dataset:
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


def set_variable_attributes(ds: xr.Dataset, sensor_name: str) -> xr.Dataset:
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
    from disdrodb.L0.standards import (
        get_description_dict,
        get_units_dict,
        get_long_name_dict,
        get_data_range_dict,
    )

    # Retrieve attributes dictionaries
    description_dict = get_description_dict(sensor_name)
    units_dict = get_units_dict(sensor_name)
    long_name_dict = get_long_name_dict(sensor_name)
    data_range_dict = get_data_range_dict(sensor_name)

    # Assign attributes to each variable
    for var in ds.data_vars:
        ds[var].attrs = {}
        ds[var].attrs["description"] = description_dict[var]
        ds[var].attrs["units"] = units_dict[var]
        ds[var].attrs["long_name"] = long_name_dict[var]
        if var in data_range_dict:
            ds[var].attrs["valid_min"] = data_range_dict[var][0]
            ds[var].attrs["valid_max"] = data_range_dict[var][1]
    return ds


def _set_attrs_dict(ds, attrs_dict):
    for var in attrs_dict.keys():
        if var in ds:
            ds[var].attrs.update(attrs_dict[var])


def set_coordinate_attributes(ds):
    # Get attributes dictionary
    attrs_dict = get_coords_attrs_dict(ds)
    # Set attributes
    _set_attrs_dict(ds, attrs_dict)
    return ds


def set_dataset_attrs(ds, sensor_name):
    """Set variable and coordinates attributes."""
    # - Add netCDF variable attributes
    # --> Attributes: long_name, units, descriptions, valid_min, valid_max
    ds = set_variable_attributes(ds=ds, sensor_name=sensor_name)

    # - Add netCDF coordinate attributes
    ds = set_coordinate_attributes(ds=ds)

    #  - Set DISDRODB global attributes
    ds = set_disdrodb_attrs(ds=ds, product_level="L0B")
    return ds


####--------------------------------------------------------------------------.
#### L0B Raw DataFrame Preprocessing


def create_L0B_from_L0A(
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
        Wheter to verbose the processing.
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
    sensor_name = attrs["sensor_name"]
    # -----------------------------------------------------------.
    # Preprocess raw_spectrum, diameter and velocity arrays if available
    if np.any(
        np.isin(
            ["raw_drop_concentration", "raw_drop_average_velocity", "raw_drop_number"],
            df.columns,
        )
    ):
        # Retrieve dictionary of raw data matrices for xarray Dataset
        data_vars = retrieve_L0B_arrays(df, sensor_name, verbose=verbose)
    else:
        data_vars = {}
    # -----------------------------------------------------------.
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

    # -----------------------------------------------------------.
    # Define coordinates for xarray Dataset
    # - Diameter and velocity
    coords = get_bin_coords(sensor_name=sensor_name)
    # - Time
    coords["time"] = df["time"].values
    # - CRS
    coords["crs"] = attrs["crs"]
    # - Geolocation
    geolocation_vars = ["latitude", "longitude", "altitude"]
    for var in geolocation_vars:
        if var in data_vars:
            coords[var] = data_vars[var]
            _ = data_vars.pop(var)
        else:
            coords[var] = attrs[var]
            _ = attrs.pop(var)

    # -----------------------------------------------------------
    # Create xarray Dataset
    try:
        ds = xr.Dataset(
            data_vars=data_vars,
            coords=coords,
            attrs=attrs,
        )
    except Exception as e:
        msg = f"Error in the creation of L1 xarray Dataset. The error is: \n {e}"
        log_error(logger=logger, msg=msg, verbose=False)
        raise ValueError(msg)

    # Ensure variables with dtype object are converted to string
    ds = convert_object_variables_to_string(ds)

    # Set netCDF dimension order
    ds = ds.transpose("time", "diameter_bin_center", ...)

    # Add netCDF variable and coordinate attributes
    ds = set_dataset_attrs(ds, sensor_name)

    # Check L0B standards
    check_L0B_standards(ds)

    # -----------------------------------------------------------
    return ds


####--------------------------------------------------------------------------.
#### L0B netCDF4 Writer


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
            chunks = [
                shape[i] if chunks[i] > shape[i] else chunks[i]
                for i in range(len(chunks))
            ]
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
        if chunks is not None:
            ds[var] = ds[var].chunk(chunks)
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
    encoding_dict = get_L0B_encodings_dict(sensor_name)
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


def write_L0B(ds: xr.Dataset, fpath: str, force=False) -> None:
    """Save the xarray dataset into a NetCDF file.

    Parameters
    ----------
    ds : xr.Dataset
        Input xarray dataset.
    fpath : str
        Output file path.
    sensor_name : str
        Name of the sensor.
    force : bool, optional
        Whether to overwrite existing data.
        If True, overwrite existing data into destination directories.
        If False, raise an error if there are already data into destination directories. This is the default.
    """
    # Create station directory if does not exist
    _create_directory(os.path.dirname(fpath))

    # Check if the file already exists
    # - If force=True --> Remove it
    # - If force=False --> Raise error
    _remove_if_exists(fpath, force=force)

    # Get sensor name from dataset
    sensor_name = ds.attrs.get("sensor_name")

    # Set encodings
    ds = set_encodings(ds=ds, sensor_name=sensor_name)

    # Write netcdf
    ds.to_netcdf(fpath, engine="netcdf4")


####--------------------------------------------------------------------------.
#### L0B Raw netCDFs Preprocessing


def _check_dict_names_validity(dict_names, sensor_name):
    """Check dict_names dictionary values validity."""
    valid_names = get_valid_names(sensor_name)
    keys = np.array(list(dict_names.keys()))
    values = np.array(list(dict_names.values()))
    # Get unvalid keys
    unvalid_keys = keys[np.isin(values, valid_names, invert=True)]
    if len(unvalid_keys) > 0:
        # Report unvalid keys and raise error
        unvalid_dict = {k: dict_names[k] for k in unvalid_keys}
        msg = f"The following dict_names values are not valid: {unvalid_dict}"
        log_error(logger=logger, msg=msg, verbose=False)
        raise ValueError(msg)


def _get_dict_names_variables(dict_names, sensor_name):
    """Get DISDRODB variables specified in dict_names."""
    possible_variables = get_valid_variable_names(sensor_name)
    dictionary_names = list(dict_names.values())
    variables = [name for name in dictionary_names if name in possible_variables]
    return variables


def _get_dict_names_dimensions(dict_names, sensor_name):
    """Get DISDRODB dimensions specified in dict_names."""
    possible_dims = get_valid_dimension_names(sensor_name)
    dictionary_names = list(dict_names.values())
    dims = [name for name in dictionary_names if name in possible_dims]
    return dims


def _get_dict_dims(dict_names, sensor_name):
    dims = _get_dict_names_dimensions(dict_names, sensor_name)
    dict_dims = {k: v for k, v in dict_names.items() if v in dims}
    return dict_dims


def rename_dataset_dimension(ds, dict_names, sensor_name):
    """Rename Dataset dimensions."""
    dict_dims = _get_dict_dims(dict_names, sensor_name)
    ds = ds.rename_dims(dict_dims)
    return ds


def rename_dataset(ds, dict_names):
    """Rename Dataset variables and coordinates."""
    # Get valid variables, coordinates, dimensions of the dataset
    ds_keys = list(ds.data_vars) + list(ds.coords) + list(ds.dims)
    # Get valid dictionary
    # - Remove keys which are missing from the dataset
    valid_dict = {k: v for k, v in dict_names.items() if k in ds_keys}
    # Rename dataset
    ds = ds.rename(valid_dict)
    return ds


def subset_dataset(ds, dict_names, sensor_name):
    # Get valid variable names
    possible_variables = get_valid_variable_names(sensor_name)
    # Get variables availables in the dict_names and dataset
    dataset_variables = list(ds.data_vars)
    dictionary_names = list(dict_names.values())
    subset_variables = [
        var
        for var in dataset_variables
        if var in dictionary_names and var in possible_variables
    ]
    # Subset the dataset
    ds = ds[subset_variables]
    return ds


def add_dataset_missing_variables(ds, missing_vars, sensor_name):
    """Add missing Dataset variables as nan DataArrays."""
    from disdrodb.L0.standards import get_variables_dimension

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

    # Rename dataset dimensions
    rename_dataset_dimension

    # Rename dataset variables and coordinates
    ds = rename_dataset(ds=ds, dict_names=dict_names)

    # Subset dataset with expected variables
    ds = subset_dataset(ds=ds, dict_names=dict_names, sensor_name=sensor_name)

    # If missing variables, infill with NaN array
    expected_vars = set(_get_dict_names_variables(dict_names, sensor_name))
    dataset_vars = set(ds.data_vars)
    missing_vars = expected_vars.difference(dataset_vars)
    if len(missing_vars) > 0:
        ds = add_dataset_missing_variables(
            ds=ds, missing_vars=missing_vars, sensor_name=sensor_name
        )

    # Update the coordinates for (diameter and velocity)
    coords = get_bin_coords(sensor_name)
    ds = ds.assign_coords(coords)

    # Return dataset
    return ds


def process_raw_nc(
    filepath,
    dict_names,
    ds_sanitizer_fun,
    sensor_name,
    verbose,
    attrs,
):
    """Read and convert a raw netCDF into a DISDRODB L0B netCDF.

    Parameters
    ----------
    filepath : str
        netCDF file path.
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
        Wheter to verbose the processing.


    Returns
    -------
    xr.Dataset
        L0B xr.Dataset
    """
    # Open the netCDF
    ds = xr.open_dataset(filepath)

    # Preprocess netcdf
    ds = preprocess_raw_netcdf(ds=ds, dict_names=dict_names, sensor_name=sensor_name)

    # Add CRS and geolocation information
    attrs = attrs.copy()
    coords = {}
    coords["crs"] = attrs["crs"]
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

    # Ensure variables with dtype object are converted to string
    ds = convert_object_variables_to_string(ds)

    # Set netCDF dimension order
    ds = ds.transpose("time", "diameter_bin_center", ...)

    # Add netCDF variable and coordinate attributes
    ds = set_dataset_attrs(ds, sensor_name)

    # Check L0B standards
    check_L0B_standards(ds)

    # Return dataset
    return ds


####--------------------------------------------------------------------------.
