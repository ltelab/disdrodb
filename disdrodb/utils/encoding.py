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
"""DISDRODB netCDF4 encoding utilities."""
import os

import numpy as np
import xarray as xr

from disdrodb.utils.yaml import read_yaml

EPOCH = "seconds since 1970-01-01 00:00:00"


def get_encodings_dict():
    """Get encoding dictionary for DISDRODB product variables and coordinates."""
    import disdrodb

    configs_path = os.path.join(disdrodb.__root_path__, "disdrodb", "etc", "configs")
    encodings_dict = read_yaml(os.path.join(configs_path, "encodings.yaml"))
    return encodings_dict


def set_encodings(ds: xr.Dataset, encodings_dict: dict) -> xr.Dataset:
    """Apply the encodings to the xarray Dataset.

    Parameters
    ----------
    ds  : xarray.Dataset
        Input xarray dataset.
    encodings_dict : dict
        Dictionary with encodings specifications.

    Returns
    -------
    xarray.Dataset
        Output xarray dataset.
    """
    # TODO: CHANGE CHUNKSIZES SPECIFICATION USING {<DIM>: <CHUNKSIZE>} INSTEAD OF LIST
    # --> Then unwrap to list of chunksizes here

    # Subset encoding dictionary
    # - Here below encodings_dict contains only keys (variables) within the dataset
    encodings_dict = {var: encodings_dict[var] for var in ds.data_vars if var in encodings_dict}

    # Ensure chunksize smaller than the array shape
    encodings_dict = sanitize_encodings_dict(encodings_dict, ds)

    # Rechunk variables for fast writing !
    # - This pop the chunksize argument from the encoding dict !
    ds = rechunk_dataset(ds, encodings_dict)

    # Set time encoding
    if "time" in ds:
        ds["time"] = ds["time"].dt.floor("s")  # ensure no sub-second values
        ds["time"] = ds["time"].astype("datetime64[s]")
        ds["time"].encoding.update(get_time_encoding())

    # Set the variable encodings
    for var, encoding in encodings_dict.items():
        ds[var].encoding.update(encoding)

    # Ensure no deprecated "missing_value" attribute
    # - When source dataset is netcdf (i.e. ARM)
    for var in list(ds.variables):
        _ = ds[var].encoding.pop("missing_value", None)

    return ds


def sanitize_encodings_dict(encodings_dict: dict, ds: xr.Dataset) -> dict:
    """Ensure chunk size to be smaller than the array shape.

    Parameters
    ----------
    encodings_dict : dict
        Dictionary containing the variable encodings.
    ds  : xarray.Dataset
        Input dataset.

    Returns
    -------
    dict
        Encoding dictionary.
    """
    for var in ds.data_vars:
        if var in encodings_dict:
            shape = ds[var].shape
            chunks = encodings_dict[var].get("chunksizes", None)
            if chunks is not None:
                chunks = [shape[i] if chunks[i] > shape[i] else chunks[i] for i in range(len(chunks))]
                encodings_dict[var]["chunksizes"] = chunks
    return encodings_dict


def rechunk_dataset(ds: xr.Dataset, encodings_dict: dict) -> xr.Dataset:
    """Coerce the dataset arrays to have the chunk size specified in the encoding dictionary.

    Parameters
    ----------
    ds  : xarray.Dataset
        Input xarray dataset
    encodings_dict : dict
        Dictionary containing the encoding to write the xarray dataset as a netCDF.

    Returns
    -------
    xarray.Dataset
        Output xarray dataset
    """
    for var in ds.data_vars:
        if var in encodings_dict:
            chunks = encodings_dict[var].get("chunksizes", None)  # .pop("chunksizes", None)
            if chunks is not None:
                dims = list(ds[var].dims)
                chunks_dict = dict(zip(dims, chunks))
                ds[var] = ds[var].chunk(chunks_dict)
                ds[var].encoding["chunksizes"] = chunks
    return ds


def get_time_encoding() -> dict:
    """Create time encoding.

    Returns
    -------
    dict
        Time encoding.
    """
    encoding = {}
    encoding["dtype"] = "int64"  # if float trailing sub-seconds values
    encoding["fillvalue"] = np.iinfo(np.int64).max
    encoding["units"] = EPOCH
    encoding["calendar"] = "proleptic_gregorian"
    return encoding
