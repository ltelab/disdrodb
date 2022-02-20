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
# - Functions to convert L0 Apache parquet files to L1 netCDF or Zarr files

# -----------------------------------------------------------------------------.
import logging
import zarr
import numpy as np
import dask.array as da
import dask.dataframe as dd
import xarray as xr

from disdrodb.check_standards import check_sensor_name
from disdrodb.check_standards import check_L1_standards
from disdrodb.check_standards import check_array_lengths_consistency

from disdrodb.standards import get_diameter_bin_center
from disdrodb.standards import get_diameter_bin_lower
from disdrodb.standards import get_diameter_bin_upper
from disdrodb.standards import get_diameter_bin_width
from disdrodb.standards import get_velocity_bin_center
from disdrodb.standards import get_velocity_bin_lower
from disdrodb.standards import get_velocity_bin_upper
from disdrodb.standards import get_velocity_bin_width
from disdrodb.standards import get_raw_field_nbins

logger = logging.getLogger(__name__)


def get_fieldn_from_raw_spectrum(arr):
    # TODO
    logger.info("Computing fieldn from raw spectrum.")
    return arr[:, :, 0]


def get_fieldv_from_raw_spectrum(arr):
    # TODO
    logger.info("Computing fieldv from raw spectrum.")
    return arr[:, 0, :]


def check_L0_raw_fields_available(df, sensor_name):
    n_bins_dict = get_raw_field_nbins(sensor_name=sensor_name)
    raw_vars = np.array(list(n_bins_dict.keys()))
    missing_vars = raw_vars[np.isin(raw_vars, list(df.columns), invert=True)]
    if len(missing_vars) > 0:
        raise ValueError(f"The following L0 raw fields are missing: {missing_vars}")


def convert_L0_raw_fields_arr_flags(arr, key):
    # TODO: FieldN and FieldV --> -9.999, has floating number
    pass
    return arr


def set_raw_fields_arr_dtype(arr, key):
    if key == "RawData":
        arr = arr.astype(int)
    else:
        arr = arr.astype(float)
    return arr


def reshape_L0_raw_datamatrix_to_2D(arr, n_bins_dict, n_timesteps):
    try:
        arr = arr.reshape(n_timesteps, n_bins_dict["FieldN"], n_bins_dict["FieldV"])
    except Exception as e:
        msg = f"Impossible to reshape the raw_spectrum matrix. The error is: \n {e}"
        logger.error(msg)
        print(msg)
        raise ValueError(msg)
    return arr


def retrieve_L1_raw_data_matrix(df, sensor_name, lazy=True, verbose=False):
    # Log
    msg = " - Retrieval of L1 data matrix started."
    if verbose:
        print(msg)
    logger.info(msg)
    # ----------------------------------------------------------.
    # Check L0 raw field availability
    check_L0_raw_fields_available(df, sensor_name)
    # Retrieve raw fields matrix bins dictionary
    n_bins_dict = get_raw_field_nbins(sensor_name=sensor_name)
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
        # Parse the string splitting at ,
        df_series = df[key].astype(str).str.split(",")
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
        # For key='RawData', reshape to 2D matrix
        if key == "RawData":
            arr = reshape_L0_raw_datamatrix_to_2D(arr, n_bins_dict, n_timesteps)
        # Add array to dictionary
        dict_data[key] = arr

    # Retrieve unavailable keys from raw spectrum
    if len(unavailable_keys) > 0:
        if "RawData" not in list(dict_data.keys()):
            raise ValueError(
                "The raw spectrum is required to compute unavaible N_D and N_V."
            )
        if "FieldN" in unavailable_keys:
            dict_data["FieldN"] = get_fieldn_from_raw_spectrum(dict_data["RawData"])
        if "FieldV" in unavailable_keys:
            dict_data["FieldV"] = get_fieldv_from_raw_spectrum(dict_data["RawData"])

    # Log
    msg = " - Retrieval of L1 data matrix finished."
    if verbose:
        print(msg)
    logger.info(msg)
    # Return
    return dict_data


def get_L1_coords(sensor_name):
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


def create_L1_dataset_from_L0(df, attrs, lazy=True, verbose=False):
    # Retrieve sensor name
    sensor_name = attrs["sensor_name"]
    # -----------------------------------------------------------.
    # Preprocess raw_spectrum, diameter and velocity arrays if available
    if np.any(np.isin(["FieldN", "FieldV", "RawData"], df.columns)):
        # Check dataframe row consistency
        df = check_array_lengths_consistency(
            df, sensor_name=sensor_name, lazy=lazy, verbose=verbose
        )
        # Retrieve raw data matrices
        dict_data = retrieve_L1_raw_data_matrix(
            df, sensor_name, lazy=lazy, verbose=verbose
        )
        # Define raw data matrix variables for xarray Dataset
        data_vars = {
            "FieldN": (["time", "diameter_bin_center"], dict_data["FieldN"]),
            "FieldV": (["time", "velocity_bin_center"], dict_data["FieldV"]),
            "RawData": (
                ["time", "diameter_bin_center", "velocity_bin_center"],
                dict_data["RawData"],
            ),
        }
    else:
        data_vars = {}
    # -----------------------------------------------------------.
    # Define other disdrometer 'auxiliary' variables varying over time dimension
    aux_columns = df.columns[
        np.isin(df.columns, ["FieldN", "FieldV", "RawData", "time"], invert=True)
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
    coords = get_L1_coords(sensor_name=sensor_name)
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
    # Check L1 standards
    check_L1_standards(ds)

    # -----------------------------------------------------------
    # TODO: Replace NA flags

    # -----------------------------------------------------------
    # TODO: Add L1 encoding

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


def write_L1_to_netcdf(ds, fpath, sensor_name):
    from disdrodb.standards import get_L1_netcdf_encoding_dict

    # Get encoding dictionary
    encoding_dict = get_L1_netcdf_encoding_dict(sensor_name)
    encoding_dict = {k: encoding_dict[k] for k in ds.data_vars}

    # Ensure chunksize smaller than the array shape)
    encoding_dict = sanitize_encodings_dict(encoding_dict, ds)

    # Rechunk variables for fast writing !
    ds = rechunk_dataset(ds, encoding_dict)

    # Write netcdf
    ds.to_netcdf(fpath, engine="netcdf4", encoding=encoding_dict)


def write_L1_to_zarr(ds, fpath, sensor_name):
    pass
    # ds = rechunk_L1_dataset(ds, sensor_name=sensor_name)
    # zarr_encoding_dict = get_L1_zarr_encodings_standards(sensor_name=sensor_name)
    # ds.to_zarr(fpath, encoding=zarr_encoding_dict, mode="w")
    return None


####--------------------------------------------------------------------------.
#### L1 Summary statistics
def create_L1_summary_statistics(ds, processed_dir, station_id, sensor_name):
    # TODO[GG]
    pass
    return
