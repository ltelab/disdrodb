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
"""Functions to process DISDRODB L0A files into DISDRODB L0B netCDF files."""

import logging

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
from disdrodb.utils.coords import add_dataset_crs_coords
from disdrodb.utils.encoding import set_encodings
from disdrodb.utils.logger import log_info
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


def replace_empty_strings_with_zeros(values):
    """Replace empty comma separated strings with '0'."""
    values[np.char.str_len(values) == 0] = "0"
    return values


def format_string_array(string: str, n_values: int) -> np.array:
    """Split a string with multiple numbers separated by a delimiter into an 1D array.

        e.g. : format_string_array("2,44,22,33", 4) will return [ 2. 44. 22. 33.]

    If empty string ("") or "" --> Return an arrays of zeros
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
    # Check for empty string or "0" case
    # - Assume no precipitation recorded. Return an arrays of zeros
    if string in {"", "0"}:
        values = np.zeros(n_values)
        return values

    # Check for NaN case
    # - Assume no data available. Return an arrays of NaN
    if string == "NaN":
        values = np.zeros(n_values) * np.nan
        return values

    # Retrieve list of values
    split_str = infer_split_str(string)
    values = np.array(string.strip(split_str).split(split_str))

    # If the length is not as expected --> Assume data corruption
    # --> Return an array with nan
    if len(values) != n_values:
        values = np.zeros(n_values) * np.nan
        return values

    # Otherwise sanitize the list of value
    # Ensure string type
    values = values.astype("str")
    # Replace '' with 0
    values = replace_empty_strings_with_zeros(values)
    # Replace "-9.999" with 0
    values = np.char.replace(values, "-9.999", "0")
    # Cast values to float type
    # --> Note: the disk encoding is specified in the l0b_encodings.yml
    values = values.astype(float)
    return values


def reshape_raw_spectrum(
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

        # Ensure is a string, get a numpy array for each row and then stack
        # - Option 1: Clear but lot of copies
        # df_series = df[key].astype(str)
        # list_arr = df_series.apply(format_string_array, n_values=n_values)
        # arr = np.stack(list_arr, axis=0)

        # - Option 2: still copies
        # arr = np.vstack(format_string_array(s, n_values=n_values) for s in df_series.astype(str))

        # - Option 3: more memory efficient
        n_timesteps = len(df[key])
        arr = np.empty((n_timesteps, n_values), dtype=float)  # preallocates
        for i, s in enumerate(df[key].astype(str)):
            arr[i, :] = format_string_array(s, n_values=n_values)

        # Retrieve dimensions
        dims_order = dims_order_dict[key]

        # For key='raw_drop_number', if 2D spectrum, reshape to 2D matrix
        # Example:
        # - This applies i.e for PARSIVEL*, LPM, PWS100
        # - This does not apply to RD80
        if key == "raw_drop_number" and len(dims_order) == 2:
            arr, dims = reshape_raw_spectrum(
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


def ensure_valid_geolocation(ds: xr.Dataset, coord: str, errors: str = "ignore") -> xr.Dataset:
    """Ensure valid geolocation coordinates.

    'altitude' must be >= 0, 'latitude' must be within [-90, 90] and
    'longitude' within [-180, 180].

    It can deal with coordinates varying with time.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset containing the coordinate.
    coord : str
        Name of the coordinate variable to validate.
    errors : {"ignore", "raise", "coerce"}, default "ignore"
        - "ignore": nothing is done.
        - "raise" : raise ValueError if invalid values are found.
        - "coerce": out-of-range values are replaced with NaN.

    Returns
    -------
    xr.Dataset
        Dataset with validated coordinate values.
    """
    # Define coordinates ranges
    ranges = {
        "altitude": (0, np.inf),
        "latitude": (-90, 90),
        "longitude": (-180, 180),  # used only for "raise"/"coerce"
    }

    # Check coordinate is available and correctly defined.
    if coord not in ds:
        raise ValueError(f"Coordinate '{coord}' not found in dataset.")
    if coord not in list(ranges):
        raise ValueError(f"Valid geolocation coordinates are: {list(ranges)}.")

    # Validate coordinate
    vmin, vmax = ranges[coord]
    invalid = (ds[coord] < vmin) | (ds[coord] > vmax)
    invalid = invalid.compute()

    # Deal within invalid errors
    if errors == "raise" and invalid.any():
        raise ValueError(f"{coord} out of range {vmin}-{vmax}.")
    if errors == "coerce":
        ds[coord] = ds[coord].where(~invalid)
    return ds


def convert_object_variables_to_string(ds: xr.Dataset) -> xr.Dataset:
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


def set_variable_attributes(ds: xr.Dataset, sensor_name: str) -> xr.Dataset:
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


####--------------------------------------------------------------------------.
#### L0B Raw DataFrame Preprocessing


def _define_dataset_variables(df, sensor_name, logger=None, verbose=False):
    """Define DISDRODB L0B netCDF array variables."""
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


def generate_l0b(
    df: pd.DataFrame,
    metadata: dict,
    logger=None,
    verbose: bool = False,
) -> xr.Dataset:
    """Transform the DISDRODB L0A dataframe to the DISDRODB L0B xr.Dataset.

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
        # Else ensure coord is a dataset coordinates
        else:
            ds = ds.set_coords(coord)
            _ = metadata.pop(coord, None)

    # Set -9999 flag value to np.nan
    for coord in coords:
        ds[coord] = xr.where(ds[coord] == -9999, np.nan, ds[coord])

    # Ensure valid geolocation coordinates
    for coord in coords:
        ds = ensure_valid_geolocation(ds=ds, coord=coord, errors="coerce")

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
    # --> Required for correct encoding !
    ds = ds.transpose("time", "diameter_bin_center", ...)

    # Ensure variables with dtype object are converted to string
    ds = convert_object_variables_to_string(ds)

    # Add netCDF variable and coordinate attributes
    # - Add variable attributes: long_name, units, descriptions, valid_min, valid_max
    ds = set_variable_attributes(ds=ds, sensor_name=sensor_name)
    # - Add netCDF coordinate attributes
    ds = set_coordinate_attributes(ds=ds)
    #  - Set DISDRODB global attributes
    ds = set_disdrodb_attrs(ds=ds, product="L0B")

    # Check L0B standards
    check_l0b_standards(ds)

    # Set L0B encodings
    ds = set_l0b_encodings(ds=ds, sensor_name=sensor_name)
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
    encodings_dict = get_l0b_encodings_dict(sensor_name)
    ds = set_encodings(ds=ds, encodings_dict=encodings_dict)
    return ds


####--------------------------------------------------------------------------.
