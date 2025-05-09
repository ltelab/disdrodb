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
import xarray as xr

EPOCH = "seconds since 1970-01-01 00:00:00"


def set_encodings(ds: xr.Dataset, encoding_dict: dict) -> xr.Dataset:
    """Apply the encodings to the xarray Dataset.

    Parameters
    ----------
    ds  : xarray.Dataset
        Input xarray dataset.
    encoding_dict : dict
        Dictionary with encoding specifications.

    Returns
    -------
    xarray.Dataset
        Output xarray dataset.
    """
    # Subset encoding dictionary
    # - Here below encoding_dict contains only keys (variables) within the dataset
    encoding_dict = {var: encoding_dict[var] for var in ds.data_vars if var in encoding_dict}

    # Ensure chunksize smaller than the array shape
    encoding_dict = sanitize_encodings_dict(encoding_dict, ds)

    # Rechunk variables for fast writing !
    # - This pop the chunksize argument from the encoding dict !
    ds = rechunk_dataset(ds, encoding_dict)

    # Set time encoding
    ds["time"].encoding.update(get_time_encoding())

    # Set the variable encodings
    for var, encoding in encoding_dict.items():
        ds[var].encoding.update(encoding)

    # Ensure no deprecated "missing_value" attribute
    # - When source dataset is netcdf (i.e. ARM)
    for var in list(ds.variables):
        _ = ds[var].encoding.pop("missing_value", None)

    return ds


def sanitize_encodings_dict(encoding_dict: dict, ds: xr.Dataset) -> dict:
    """Ensure chunk size to be smaller than the array shape.

    Parameters
    ----------
    encoding_dict : dict
        Dictionary containing the variable encodings.
    ds  : xarray.Dataset
        Input dataset.

    Returns
    -------
    dict
        Encoding dictionary.
    """
    for var in ds.data_vars:
        if var in encoding_dict:
            shape = ds[var].shape
            chunks = encoding_dict[var].get("chunksizes", None)
            if chunks is not None:
                chunks = [shape[i] if chunks[i] > shape[i] else chunks[i] for i in range(len(chunks))]
                encoding_dict[var]["chunksizes"] = chunks
    return encoding_dict


def rechunk_dataset(ds: xr.Dataset, encoding_dict: dict) -> xr.Dataset:
    """Coerce the dataset arrays to have the chunk size specified in the encoding dictionary.

    Parameters
    ----------
    ds  : xarray.Dataset
        Input xarray dataset
    encoding_dict : dict
        Dictionary containing the encoding to write the xarray dataset as a netCDF.

    Returns
    -------
    xarray.Dataset
        Output xarray dataset
    """
    for var in ds.data_vars:
        if var in encoding_dict:
            chunks = encoding_dict[var].pop("chunksizes", None)
            if chunks is not None:
                dims = list(ds[var].dims)
                chunks_dict = dict(zip(dims, chunks))
                ds[var] = ds[var].chunk(chunks_dict)
    return ds


def get_time_encoding() -> dict:
    """Create time encoding.

    Returns
    -------
    dict
        Time encoding.
    """
    encoding = {}
    encoding["units"] = EPOCH
    encoding["calendar"] = "proleptic_gregorian"
    return encoding
