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
    check_array_lengths_consistency,
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
    get_L0B_encodings_dict, 
)

logger = logging.getLogger(__name__)


def get_drop_concentration(arr):
    # TODO
    logger.info("Computing raw_drop_concentration from raw spectrum.")
    return arr[:, :, 0]


def get_drop_average_velocity(arr):
    # TODO
    logger.info("Computing raw_drop_average_velocity from raw spectrum.")
    return arr[:, 0, :]


def check_L0_raw_fields_available(df, sensor_name):
    n_bins_dict = get_raw_field_nbins(sensor_name=sensor_name)
    raw_vars = np.array(list(n_bins_dict.keys()))
    missing_vars = raw_vars[np.isin(raw_vars, list(df.columns), invert=True)]
    if len(missing_vars) > 0:
        raise ValueError(f"The following L0 raw fields are missing: {missing_vars}")


def convert_L0_raw_fields_arr_flags(arr, key):
    # TODO: raw_drop_concentration and raw_drop_average_velocity --> -9.999, has floating number
    pass
    return arr


def set_raw_fields_arr_dtype(arr, key):
    if key == "raw_drop_number":
        arr = arr.astype(int)
    else:
        arr = arr.astype(float)
    return arr


def reshape_L0_raw_drop_number_matrix_to_2D(arr, n_bins_dict, n_timesteps):
    try:
        arr = arr.reshape(n_timesteps, n_bins_dict["raw_drop_concentration"], n_bins_dict["raw_drop_average_velocity"])
    except Exception as e:
        msg = f"Impossible to reshape the raw_spectrum matrix. The error is: \n {e}"
        logger.error(msg)
        print(msg)
        raise ValueError(msg)
    return arr


def retrieve_L1_raw_arrays(df, sensor_name, lazy=True, verbose=False):
    # Log
    msg = " - Retrieval of L1 data matrix started."
    if verbose:
        print(msg)
    logger.info(msg)
    # ----------------------------------------------------------.
    # Check L0 raw field availability
    # check_L0_raw_fields_available(df, sensor_name)
    # Retrieve raw fields matrix bins dictionary
    n_bins_dict = get_raw_field_nbins(sensor_name=sensor_name)
    # Retrieve number of timesteps
    if lazy:
        n_timesteps = df.shape[0].compute()
    else:
        n_timesteps = df.shape[0]


    if sensor_name in ['OTT_Parsivel', 'OTT_Parsivel2']:
        split_str = ','
        # Found a campaing (MELBOURNE_2007_OTT) with different different divider, this is a temporary solution
        try:
            head = df.head(1)
            head = head['raw_drop_concentration']
            if head.find(',') == -1:
                    split_str = head[6]
        except KeyError:
            msg = "Something wrong with divider for L0B matrix, default divider is ',', tried to parse with {}".format(split_str)
            if verbose:
                print(msg)
            logger.info(msg)
    if sensor_name in ['Thies_LPM']:
        split_str = ';'
    
    # Retrieve available arrays
    dict_data = {}
    unavailable_keys = []
    for key, n_bins in n_bins_dict.items():
        # Check key is available in dataframe
        if key not in df.columns:
            unavailable_keys.append(key)
            continue
        # Parse the string splitting at ,
        df_series = df[key].astype(str).str.split(split_str)
        # Create array
        if lazy:
            arr = da.stack(df_series, axis=0)
        else:
            arr = np.stack(df_series, axis=0)
        # Remove '' at the last array position
        arr = arr[:, 0 : n_bins_dict[key]]
        # Deal with flag values (-9.9999)
        arr = convert_L0_raw_fields_arr_flags(arr, key=key)
        # Set dtype of the matrix
        arr = set_raw_fields_arr_dtype(arr, key=key)
        # For key='raw_drop_number', reshape to 2D matrix
        if key == "raw_drop_number":
            arr = reshape_L0_raw_drop_number_matrix_to_2D(arr, n_bins_dict, n_timesteps)
        # Add array to dictionary
        dict_data[key] = arr

    # Retrieve unavailable keys from raw spectrum
    if len(unavailable_keys) > 0:
        if "raw_drop_number" not in list(dict_data.keys()):
            raise ValueError(
                "The raw spectrum is required to compute unavaible N_D and N_V."
            )
        if "raw_drop_concentration" in unavailable_keys:
            dict_data["raw_drop_concentration"] = get_drop_concentration(dict_data["raw_drop_number"])
        if "raw_drop_average_velocity" in unavailable_keys:
            dict_data["raw_drop_average_velocity"] = get_drop_average_velocity(dict_data["raw_drop_number"])

    # Log
    msg = " - Retrieval of L1 data matrix finished."
    if verbose:
        print(msg)
    logger.info(msg)
    # Return
    return dict_data


def get_coords(sensor_name):
    check_sensor_name(sensor_name=sensor_name)
    coords = {}
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


def create_L0B_from_L0A(df, attrs, lazy=True, verbose=False):
    # Retrieve sensor name
    sensor_name = attrs["sensor_name"]
    # -----------------------------------------------------------.
    # Preprocess raw_spectrum, diameter and velocity arrays if available
    if np.any(np.isin(["raw_drop_concentration", "raw_drop_average_velocity", "raw_drop_number"], df.columns)):
        # Check dataframe row consistency
        df = check_array_lengths_consistency(
            df, sensor_name=sensor_name, lazy=lazy, verbose=verbose
        )
        # Retrieve raw data matrices
        dict_data = retrieve_L1_raw_arrays(
            df, sensor_name, lazy=lazy, verbose=verbose
        )
        # Define raw data matrix variables for xarray Dataset
        data_vars = {
            "raw_drop_concentration": (["time", "diameter_bin_center"], dict_data["raw_drop_concentration"]),
            "raw_drop_average_velocity": (["time", "velocity_bin_center"], dict_data["raw_drop_average_velocity"]),
            "raw_drop_number": (
                ["time", "diameter_bin_center", "velocity_bin_center"],
                dict_data["raw_drop_number"],
            ),
        }
    else:
        data_vars = {}
    # -----------------------------------------------------------.
    # Define other disdrometer 'auxiliary' variables varying over time dimension
    aux_columns = df.columns[
        np.isin(df.columns, ["raw_drop_concentration", "raw_drop_average_velocity", "raw_drop_number", "time"], invert=True)
    ]
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
        chunks = encoding_dict[var]["chunksizes"]
        if chunks is not None:
            ds[var] = ds[var].chunk(chunks)
    return ds


def write_L0B(ds, fpath, sensor_name):
    # Ensure directory exist 
    os.makedirs(os.path.dirname(fpath), exist_ok=True)
    
    # Get encoding dictionary
    encoding_dict = get_L0B_encodings_dict(sensor_name)
    encoding_dict = {k: encoding_dict[k] for k in ds.data_vars}

    # Ensure chunksize smaller than the array shape)
    encoding_dict = sanitize_encodings_dict(encoding_dict, ds)

    # Rechunk variables for fast writing !
    ds = rechunk_dataset(ds, encoding_dict)

    # Write netcdf
    ds.to_netcdf(fpath, engine="netcdf4", encoding=encoding_dict)


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
    dt_most_frequent = dt_most_frequent.astype('m8[s]')
    inferred_sampling_interval = dt_most_frequent.astype(int)
    stats_dict['inferred_sampling_interval'] = inferred_sampling_interval

    # Number of years, months, days, minutes 
    time = ds.time.values
    n_timesteps = len(time)
    n_minutes = inferred_sampling_interval/60*n_timesteps
    n_hours = n_minutes/60 
    n_days = n_hours/24

    stats_dict['n_timesteps'] = n_timesteps
    stats_dict['n_minutes'] = n_minutes
    stats_dict['n_hours'] = n_hours
    stats_dict['n_days'] = n_days

    # Add start_time and end_time
    start_time = pd.DatetimeIndex(time[[0]])
    end_time = pd.DatetimeIndex(time[[-1]])
    years = np.unique([start_time.year, end_time.year])
    if len(years) == 1:
        years_coverage = str(years[0])
    else: 
        years_coverage = str(years[0]) + "-" + str(years[-1])
        
    stats_dict['years_coverage'] = years_coverage
    stats_dict['start_time'] = start_time[0].isoformat()
    stats_dict['end_time'] = end_time[0].isoformat()

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
