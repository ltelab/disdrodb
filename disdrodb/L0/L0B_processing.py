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
import yaml
import logging
import numpy as np
import pandas as pd
import dask.array as da
import xarray as xr

from disdrodb.L0.check_standards import (
    check_sensor_name,
    check_L0B_standards,
    # check_array_lengths_consistency,
)
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
)

logger = logging.getLogger(__name__)


# def get_drop_concentration(arr):
#     # TODO
#     logger.info("Computing raw_drop_concentration from raw spectrum.")
#     return arr[:, :, 0]


# def get_drop_average_velocity(arr):
#     # TODO
#     logger.info("Computing raw_drop_average_velocity from raw spectrum.")
#     return arr[:, 0, :]


def check_L0_raw_fields_available(df, sensor_name):
    n_bins_dict = get_raw_field_nbins(sensor_name=sensor_name)
    raw_vars = np.array(list(n_bins_dict.keys()))
    missing_vars = raw_vars[np.isin(raw_vars, list(df.columns), invert=True)]
    if len(missing_vars) > 0:
        raise ValueError(f"The following L0 raw fields are missing: {missing_vars}")


def infer_split_str(string):
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


def format_string_array(string, n_values):
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


def reshape_raw_spectrum_to_2D(arr, n_bins_dict, n_timesteps):
    try:
        arr = arr.reshape(
            n_timesteps,
            n_bins_dict["raw_drop_concentration"],
            n_bins_dict["raw_drop_average_velocity"],
        )
    except Exception as e:
        msg = f"Impossible to reshape the raw_spectrum matrix. The error is: \n {e}"
        logger.error(msg)
        print(msg)
        raise ValueError(msg)
    return arr


def retrieve_L0B_arrays(df, sensor_name, lazy=True, verbose=False):
    # Log
    msg = " - Retrieval of L0B data matrix started."
    if verbose:
        print(msg)
    logger.info(msg)
    # ----------------------------------------------------------.
    # Check L0 raw field availability
    # check_L0_raw_fields_available(df, sensor_name)

    # Retrieve raw fields matrix bins dictionary
    n_bins_dict = get_raw_field_nbins(sensor_name=sensor_name)

    # Retrieve dimension order dictionary
    dims_dict = get_raw_field_dim_order(sensor_name)

    # Retrieve dimension of the raw_drop_number field
    n_dim_spectrum = get_raw_spectrum_ndims(sensor_name)

    # Retrieve number of timesteps
    if lazy:
        n_timesteps = df.shape[0].compute()
    else:
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

        # Get a numpy array for each row
        list_arr = df_series.apply(format_string_array, n_values=n_bins)

        # Create the array
        if lazy:
            arr = da.stack(list_arr, axis=0)
        else:
            arr = np.stack(list_arr, axis=0)

        # For key='raw_drop_number', if 2D ... reshape to 2D matrix
        # - This applies i.e for OTT_Parsivels and ThiesLPM
        # - This does not apply to RD80
        if key == "raw_drop_number" and n_dim_spectrum == 2:
            arr = reshape_raw_spectrum_to_2D(arr, n_bins_dict, n_timesteps)

        # Define dictionary to pass to xr.Dataset
        dims_order = ["time"] + dims_dict[key]
        dict_data[key] = (dims_order, arr)

    # -------------------------------------------------------------------------.
    # Retrieve unavailable keys from the raw spectrum field
    # TODO: This should be performed when the xarray object is created !
    # if len(unavailable_keys) > 0:
    #     if "raw_drop_number" not in list(dict_data.keys()):
    #         raise ValueError(
    #             """The raw spectrum is required to compute the unavailables
    #             'raw_drop_concentration' and 'raw_drop_average_velocity' fields."""
    #         )
    #     if "raw_drop_concentration" in unavailable_keys and n_dim_spectrum == 2:
    #         # TODO: can this be computed for RD80 ?
    #         dict_data["raw_drop_concentration"] = get_drop_concentration(
    #             dict_data["raw_drop_number"]
    #         )
    #     if "raw_drop_average_velocity" in unavailable_keys and n_dim_spectrum == 2:
    #         dict_data["raw_drop_average_velocity"] = get_drop_average_velocity(
    #             dict_data["raw_drop_number"]
    #         )
    # -------------------------------------------------------------------------.
    # Log
    msg = " - Retrieval of L0B data matrices finished."
    if verbose:
        print(msg)
    logger.info(msg)

    # Return
    return dict_data


def get_coords(sensor_name):
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


def convert_object_variables_to_string(ds):
    for var in ds.data_vars:
        if pd.api.types.is_object_dtype(ds[var]):
            ds[var] = ds[var].astype(str)
    return ds


def create_L0B_from_L0A(df, attrs, lazy=True, verbose=False):
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
        data_vars = retrieve_L0B_arrays(df, sensor_name, lazy=lazy, verbose=verbose)
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
    if lazy:
        aux_data_vars = {
            column: (["time"], df[column].to_dask_array(lengths=True))
            for column in aux_columns
        }
    else:
        aux_data_vars = {
            column: (["time"], df[column].values) for column in aux_columns
        }
    data_vars.update(aux_data_vars)

    # -----------------------------------------------------------.
    # Drop lat/lon array if present (TODO: in future L0 should not contain it)
    if "latitude" in data_vars:
        _ = data_vars.pop("latitude")
    if "longitude" in data_vars:
        _ = data_vars.pop("longitude")
    if "altitude" in data_vars:
        _ = data_vars.pop("altitude")

    # -----------------------------------------------------------.
    # Define coordinates for xarray Dataset
    coords = get_coords(sensor_name=sensor_name)
    coords["time"] = df["time"].values
    coords["crs"] = attrs["crs"]
    if "latitude" in data_vars:
        coords["latitude"] = data_vars["latitude"]
        _ = data_vars.pop("latitude")
    else:
        coords["latitude"] = attrs["latitude"]
    if "longitude" in data_vars:
        coords["longitude"] = data_vars["longitude"]
        _ = data_vars.pop("longitude")
    else:
        coords["longitude"] = attrs["longitude"]
    if "altitude" in data_vars:
        coords["altitude"] = data_vars["altitude"]
        _ = data_vars.pop("altitude")
    else:
        coords["altitude"] = attrs["altitude"]

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
        logger.error(msg)
        raise ValueError(msg)

    # Ensure variables with dtype object are converted to string
    ds = convert_object_variables_to_string(ds)

    # -----------------------------------------------------------
    # Check L0B standards
    check_L0B_standards(ds)

    # -----------------------------------------------------------
    # TODO: Replace NA flags

    # -----------------------------------------------------------
    return ds


####--------------------------------------------------------------------------.
#### Writers
def sanitize_encodings_dict(encoding_dict, ds):
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


def rechunk_dataset(ds, encoding_dict):
    for var in ds.data_vars:
        chunks = encoding_dict[var].pop("chunksizes")
        if chunks is not None:
            ds[var] = ds[var].chunk(chunks)
    return ds


def set_encodings(ds, sensor_name):
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


def write_L0B(ds, fpath, sensor_name):
    # Ensure directory exist
    os.makedirs(os.path.dirname(fpath), exist_ok=True)

    # Set encodings
    ds = set_encodings(ds, sensor_name)

    # Write netcdf
    ds.to_netcdf(fpath, engine="netcdf4")  # , encoding=encoding_dict)


####--------------------------------------------------------------------------.
#### Summary statistics
def create_summary_statistics(ds, processed_dir, station_id, sensor_name):
    """Create L0 summary statistics and save it into the station info YAML file."""
    ###-----------------------------------------------------------------------.
    # Initialize dictionary
    stats_dict = {}

    # Infer the sampling interval looking at the difference between timesteps
    dt, counts = np.unique(np.diff(ds.time.values), return_counts=True)
    dt_most_frequent = dt[np.argmax(counts)]
    dt_most_frequent = dt_most_frequent.astype("m8[s]")
    inferred_sampling_interval = dt_most_frequent.astype(int)
    stats_dict["inferred_sampling_interval"] = inferred_sampling_interval

    # Number of years, months, days, minutes
    time = ds.time.values
    n_timesteps = len(time)
    n_minutes = inferred_sampling_interval / 60 * n_timesteps
    n_hours = n_minutes / 60
    n_days = n_hours / 24

    stats_dict["n_timesteps"] = n_timesteps
    stats_dict["n_minutes"] = n_minutes
    stats_dict["n_hours"] = n_hours
    stats_dict["n_days"] = n_days

    # Add start_time and end_time
    start_time = pd.DatetimeIndex(time[[0]])
    end_time = pd.DatetimeIndex(time[[-1]])
    years = np.unique([start_time.year, end_time.year])
    if len(years) == 1:
        years_coverage = str(years[0])
    else:
        years_coverage = str(years[0]) + "-" + str(years[-1])

    stats_dict["years_coverage"] = years_coverage
    stats_dict["start_time"] = start_time[0].isoformat()
    stats_dict["end_time"] = end_time[0].isoformat()

    ###-----------------------------------------------------------------------.
    # TODO: Create and save image with temporal coverage
    # --> Colored using quality flag from sensor_status if available ?

    ###-----------------------------------------------------------------------.
    # TODO STATISTICS
    # --> Requiring deriving stats from raw spectrum

    # diameter_min, diameter_max, diameter_sum

    # Total rain events

    # Total rainy minutes

    # Total dry minutes

    # Number of dry/rainy minutes

    ###-----------------------------------------------------------------------.
    # Save to info.yaml
    info_path = os.path.join(processed_dir, "info", station_id + ".yml")
    with open(info_path, "w") as f:
        yaml.dump(stats_dict, f, sort_keys=False)

    return None
