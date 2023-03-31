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

import logging
import numpy as np
import pandas as pd
import xarray as xr
from typing import Tuple
from disdrodb.utils.logger import log_info, log_warning, log_error

logger = logging.getLogger(__name__)


####---------------------------------------------------------------------------.
def _sort_datasets_by_dim(list_ds: list, fpaths: str, dim: str = "time") -> Tuple[list, list]:
    """Sort a list of xarray.Dataset and corresponding file paths by the starting value of a specified dimension.

    Parameters
    ----------
    fpaths : list
        List of netCDFs file paths.
    list_ds : list
        List of xarray Dataset.
    dim : str, optional
        Dimension name. The default is "time".

    Returns
    -------
    tuple
        Tuple of sorted list of xarray datasets and sorted list of file paths.
    """
    start_values = [ds[dim].values[0] for ds in list_ds]
    sorted_idx = np.argsort(start_values)
    sorted_list_ds = [list_ds[i] for i in sorted_idx]
    sorted_fpaths = [fpaths[i] for i in sorted_idx]
    return sorted_list_ds, sorted_fpaths


def _get_dim_values_index(list_ds: list, dim: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Get list and dataset indices associated to the dimension values."""
    dim_values = np.concatenate([ds[dim].values for ds in list_ds])
    list_index = np.concatenate([np.ones(len(ds[dim])) * i for i, ds in enumerate(list_ds)])
    list_index = list_index.astype(int)
    ds_index = np.concatenate([np.arange(0, len(ds[dim])) for i, ds in enumerate(list_ds)])
    return dim_values, list_index, ds_index


def _get_non_monotonic_indices_to_remove(dim_values: np.ndarray) -> np.ndarray:
    """Returns the indices that cause a non-monotonic increasing series of values.

    Assume that duplicated values, if present, occurs consecutively !
    """
    diff_dim_values = np.diff(dim_values)
    indices_decreasing = np.where(diff_dim_values.astype(float) <= 0)[0] + 1
    if len(indices_decreasing) == 0:
        return []
    idx_start_decreasing = indices_decreasing[0]
    idx_restart_increase = np.max(np.where(dim_values <= dim_values[idx_start_decreasing - 1])[0])
    idx_to_remove = np.arange(idx_start_decreasing, idx_restart_increase + 1)
    return idx_to_remove


def _get_duplicated_indices(x, keep="first"):
    """Return the indices to remove for duplicated values in x such that there is only one value occurence.

    Parameters
    ----------
    x :  np.array
        Array of values.
    keep : str, optional
        The value to keep, either 'first', 'last' or False.
        The default is 'first'.
        ‘first’ : Mark duplicates as True except for the first occurrence.
        ‘last’ : Mark duplicates as True except for the last occurrence.
        False : Mark all duplicates as True.

    Returns
    -------
    np.array
        Array of indices to remove.
    """
    # Check 'keep' argument
    # if not isinstance(keep, str):
    #     raise TypeError("`keep` must be a string. Either first or last.")
    # if not np.isin(keep, ["first", "last"]):
    #     raise ValueError("Invalid value for argument keep. Only 'first' and 'last' are accepted.")
    # # Get

    # x_indices = np.arange(len(x))
    # unique_values, unique_counts = np.unique(x, return_counts=True)
    # duplicated_values = unique_values[unique_counts > 1]

    # duplicated_indices = np.array([], dtype=np.int32)
    # if keep == 'first':
    #     for value in duplicated_values:
    #         indices = np.where(x == value)[0]
    #         duplicated_indices = np.concatenate([duplicated_indices, indices[1:]])
    # elif keep == 'last':
    #     indices = np.where(x == value)[0]
    #     duplicated_indices = np.concatenate([duplicated_indices, indices[:-1]])
    # return duplicated_indices

    # Get duplicate indices
    idx_duplicated = pd.Index(x).duplicated(keep=keep)
    return np.where(idx_duplicated)[0]


def _get_bad_info_dict(
    idx_to_remove: np.ndarray,
    list_index: np.ndarray,
    dim_values: np.ndarray,
    ds_index: np.ndarray,
) -> Tuple[dict, dict]:
    """Return two dictionaries mapping, for each dataset, the bad values and indices to remove.

    Parameters
    ----------
    idx_to_remove : np.ndarray
        Indices to be removed to ensure monotonic dimension.
    list_index : np.ndarray
        Indices corresponding to the file in the `list_ds` parameter.
    ds_index : np.ndarray
        Indices corresponding to the dataset dimension index in the `list_ds` parameter.

    Returns
    -------
    dict
        A dictionary mapping the dimension values to remove for each file.
    dict
        A dictionary mapping the dataset dimension indices to remove for each file.
    """
    list_index_bad = list_index[idx_to_remove]
    ds_index_bad = ds_index[idx_to_remove]
    dim_values_bad = dim_values[idx_to_remove]
    # Retrieve dictionary with the bad values in each dataset
    dict_ds_bad_values = {k: dim_values_bad[np.where(list_index_bad == k)[0]] for k in np.unique(list_index_bad)}
    # Retrieve dictionary with the index with the bad values in each dataset
    dict_ds_bad_idx = {k: ds_index_bad[np.where(list_index_bad == k)[0]] for k in np.unique(list_index_bad)}
    return dict_ds_bad_values, dict_ds_bad_idx


def _remove_dataset_bad_values(list_ds, fpaths, dict_ds_bad_idx, dim):
    """Remove portions of xarray Datasets corresponding to duplicated values.

    Parameters
    ----------
    list_ds : list
        List of xarray Dataset.
    dict_ds_bad_idx : dict
        Dictionary with the dimension indices corresponding to bad values in each xarray Dataset.

    Returns
    -------

    list_ds : list
        List of xarray Dataset without bad values.
    """
    list_index_valid = list(range(len(list_ds)))
    for list_index_bad, bad_idx in dict_ds_bad_idx.items():
        # Get dataset
        ds = list_ds[list_index_bad]
        # If resulting in a empty dataset, drop index from list_index_valid
        if len(bad_idx) == len(list_ds[list_index_bad][dim]):
            list_index_valid.remove(list_index_bad)
        # Remove unvalid indices
        list_ds[list_index_bad] = ds.drop_isel({dim: bad_idx})

    # Keep non-empty datasets
    new_list_ds = [list_ds[idx] for idx in list_index_valid]
    new_fpaths = [fpaths[idx] for idx in list_index_valid]
    return new_list_ds, new_fpaths


def ensure_unique_dimension_values(list_ds: list, fpaths: str, dim: str = "time", verbose: bool = False) -> list:
    """Ensure that a list of xr.Dataset has non duplicated dimension values.

    Parameters
    ----------
    list_ds : list
        List of xarray Dataset.
    fpaths : list
        List of netCDFs file paths.
    dim : str, optional
        Dimension name.
        The default is "time".

    Returns
    -------
    list
        List of xarray Dataset.
    list
        List of netCDFs file paths.
    """
    # Reorder the files and filepaths by the starting dimension value (time)
    sorted_list_ds, sorted_fpaths = _sort_datasets_by_dim(list_ds=list_ds, fpaths=fpaths, dim=dim)

    # Get the datasets dimension values array (and associated list_ds/xr.Dataset indices)
    dim_values, list_index, ds_index = _get_dim_values_index(list_ds, dim=dim)

    # Get duplicated indices
    idx_duplicated = _get_duplicated_indices(dim_values, keep="first")

    # Remove duplicated indices
    if len(idx_duplicated) > 0:
        # Retrieve dictionary providing bad values and indexes for each dataset
        dict_ds_bad_values, dict_ds_bad_idx = _get_bad_info_dict(
            idx_to_remove=idx_duplicated,
            list_index=list_index,
            dim_values=dim_values,
            ds_index=ds_index,
        )

        # Report for each dataset, the duplicates values occuring
        for list_index_bad, bad_values in dict_ds_bad_values.items():
            # Retrieve dataset filepath
            fpath = fpaths[list_index_bad]
            # If all values inside the file are duplicated, report it
            if len(bad_values) == len(list_ds[list_index_bad][dim]):
                msg = (
                    f"{fpath} is excluded from concatenation. All {dim} values are already present in some other file."
                )
                log_warning(logger=logger, msg=msg, verbose=verbose)
            else:
                if np.issubdtype(bad_values.dtype, np.datetime64):
                    bad_values = bad_values.astype("M8[s]")
                msg = f"In {fpath}, dropping {dim} values {bad_values} to avoid duplicated {dim} values."
                log_warning(logger=logger, msg=msg, verbose=verbose)

        # Remove duplicated values
        list_ds, fpaths = _remove_dataset_bad_values(
            list_ds=list_ds, fpaths=fpaths, dict_ds_bad_idx=dict_ds_bad_idx, dim=dim
        )
    return list_ds, fpaths


def ensure_monotonic_dimension(list_ds: list, fpaths: str, dim: str = "time", verbose: bool = False) -> list:
    """Ensure that a list of xr.Dataset has a monotonic increasing (non duplicated) dimension values.

    Parameters
    ----------
    list_ds : list
        List of xarray Dataset.
    fpaths : list
        List of netCDFs file paths.
    dim : str, optional
        Dimension name.
        The default is "time".

    Returns
    -------
    list
        List of xarray Dataset.
    list
        List of netCDFs file paths.
    """
    # Reorder the files and filepaths by the starting dimension value (time)
    # TODO: should maybe also split by non-continuous time ...
    sorted_list_ds, sorted_fpaths = _sort_datasets_by_dim(list_ds=list_ds, fpaths=fpaths, dim=dim)

    # Get the datasets dimension values array (and associated list_ds/xr.Dataset indices)
    dim_values, list_index, ds_index = _get_dim_values_index(list_ds, dim=dim)

    # Identify the indices to remove to ensure monotonic values
    idx_to_remove = _get_non_monotonic_indices_to_remove(dim_values)

    # Remove indices causing the values to be non-monotonic increasing
    if len(idx_to_remove) > 0:
        # Retrieve dictionary providing bad values and indexes for each dataset
        dict_ds_bad_values, dict_ds_bad_idx = _get_bad_info_dict(
            idx_to_remove=idx_to_remove,
            list_index=list_index,
            dim_values=dim_values,
            ds_index=ds_index,
        )

        # Report for each dataset, the values to be dropped
        for list_index_bad, bad_values in dict_ds_bad_values.items():
            # Retrieve dataset filepath
            fpath = fpaths[list_index_bad]
            # If all values inside the file shoudl be dropped, report it
            if len(bad_values) == len(list_ds[list_index_bad][dim]):
                msg = (
                    f"{fpath} is excluded from concatenation. All {dim} values cause the dimension to be non-monotonic."
                )
                log_warning(logger=logger, msg=msg, verbose=verbose)
            else:
                if np.issubdtype(bad_values.dtype, np.datetime64):
                    bad_values = bad_values.astype("M8[s]")
                msg = f"In {fpath}, dropping {dim} values {bad_values} to ensure monotonic {dim} dimension."
                log_warning(logger=logger, msg=msg, verbose=verbose)

        # Remove duplicated values
        list_ds, fpaths = _remove_dataset_bad_values(
            list_ds=list_ds, fpaths=fpaths, dict_ds_bad_idx=dict_ds_bad_idx, dim=dim
        )
        # Iterative check
        list_ds, fpathsa = ensure_monotonic_dimension(list_ds=list_ds, fpaths=fpaths, dim=dim)

    return list_ds, fpaths


# ds_index = [0,1,2,3,0,1,2,3,4]
# list_index = [0,0,0,0,1, 1, 1,1, 1]
# dim_values = [0,1,5,5,5, 5, 6,7,8]
# list_index = np.array(list_index)
# dim_values = np.array(dim_values)
# ds_index = np.array(ds_index)


####---------------------------------------------------------------------------
def get_list_ds(fpaths: str) -> list:
    """Get list of xarray datasets from file paths.

    Parameters
    ----------
    fpaths : list
        List of netCDFs file paths.

    Returns
    -------
    list
        List of xarray datasets.
    """
    import xarray as xr

    list_ds = []
    for fpath in fpaths:
        # This context manager is required to avoid random HDF locking
        # - cache=True: store data in memory to avoid reading back from disk
        # --> but LRU cache might cause the netCDF to not be closed !
        with xr.open_dataset(fpath, cache=False) as data:
            ds = data.load()
        list_ds.append(ds)
    return list_ds


# def get_list_ds(fpaths: str) -> list:
#     """Get list of xarray datasets from file paths.

#     Parameters
#     ----------
#     fpaths : list
#         List of netCDFs file paths.

#     Returns
#     -------
#     list
#         List of xarray datasets.
#     """
#     # WARNING: READING IN PARALLEL USING MULTIPROCESS CAUSE HDF LOCK ERRORS
#     @dask.delayed
#     def open_dataset_delayed(fpath):
#         import os

#         os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
#         import xarray as xr

#         # This context manager is required to avoid random HDF locking
#         # - cache=True: store data in memory to avoid reading back from disk
#         # --> but LRU cache might cause the netCDF to not be closed !
#         with xr.open_dataset(fpath, cache=False) as data:
#             ds = data.load()
#         return ds

#     list_ds_delayed = []
#     for fpath in fpaths:
#         list_ds_delayed.append(open_dataset_delayed(fpath))
#     list_ds = dask.compute(list_ds_delayed)[0]
#     return list_ds


####---------------------------------------------------------------------------


def _concatenate_datasets(list_ds, dim="time", verbose=False):
    try:
        msg = "Start concatenating with xr.concat."
        log_info(logger=logger, msg=msg, verbose=verbose)

        ds = xr.concat(list_ds, dim="time", coords="minimal", compat="override")

        msg = "Concatenation with xr.concat has been successful."
        log_info(logger=logger, msg=msg, verbose=verbose)

    except Exception as e:
        msg = f"Concatenation with xr.concat failed. Error is {e}."
        log_error(logger=logger, msg=msg, verbose=False)
        raise ValueError(msg)
    return ds


def xr_concat_datasets(fpaths: str, verbose=False) -> xr.Dataset:
    """Concat xr.Dataset in a robust and parallel way.

    1. It checks for time dimension monotonicity

    Parameters
    ----------
    fpaths : list
        List of netCDFs file paths.

    Returns
    -------
    xr.Dataset
        A single xarray dataset.

    Raises
    ------
    ValueError
        Error if the merging/concatenation operations can not be achieved.

    """

    # --------------------------------------.
    # Open xr.Dataset lazily in parallel using dask delayed
    list_ds = get_list_ds(fpaths)

    # --------------------------------------.
    # Ensure time dimension contains no duplicated values
    list_ds, fpaths = ensure_unique_dimension_values(list_ds=list_ds, fpaths=fpaths, dim="time", verbose=verbose)

    # Ensure time dimension is monotonic increasingly
    list_ds, fpaths = ensure_monotonic_dimension(list_ds=list_ds, fpaths=fpaths, dim="time", verbose=verbose)

    # --------------------------------------.
    # Concatenate all netCDFs
    ds = _concatenate_datasets(list_ds=list_ds, dim="time", verbose=verbose)

    # --------------------------------------.
    # Return xr.Dataset
    return ds
