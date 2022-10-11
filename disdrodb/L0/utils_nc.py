#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 23 14:02:12 2022

@author: ghiggi
"""
import dask
import logging
import numpy as np
import xarray as xr

logger = logging.getLogger(__name__)


def ensure_monotonic_dimension(fpaths: str, list_ds: list, dim: str = "time") -> list:
    """Ensure that a list of xr.Dataset has a monotonic increasing dimension.

    Parameters
    ----------
    fpaths : list
        List of netCDFs file paths.
    list_ds : list
        List of xarray Dataset.
    dim : str, optional
        Dimension name.
        The default is "time".

    Returns
    -------
    list
        List of xarray datasets.
    """

    # Get dimension values array (and associated list_ds/xr.Dataset indices)
    dim_values = np.concatenate([ds[dim].values for ds in list_ds])
    list_index = np.concatenate(
        [np.ones(len(ds[dim])) * i for i, ds in enumerate(list_ds)]
    )
    list_index = list_index.astype(int)
    ds_index = np.concatenate(
        [np.arange(0, len(ds[dim])) for i, ds in enumerate(list_ds)]
    )

    # Identify index where start decreasing or duplicated
    diff_dim_values = np.diff(dim_values)
    indices_decreasing = np.where(diff_dim_values.astype(float) <= 0)[0] + 1

    if len(indices_decreasing) > 0:
        idx_start_decreasing = indices_decreasing[0]  # or duplicate

        # Find last timestep that is lower or equal to that timestep
        idx_restart_increase = np.max(
            np.where(dim_values <= dim_values[idx_start_decreasing - 1])[0]
        )

        # Indices to remove
        idx_to_remove = np.arange(idx_start_decreasing, idx_restart_increase + 1)
        list_index_bad = list_index[idx_to_remove]
        ds_index_bad = ds_index[idx_to_remove]
        dim_values_bad = dim_values[idx_to_remove]

        # Get dictionary of values/idx which are duplicated (or lead to no monotonic dimension)
        dict_ds_bad_values = {
            k: dim_values_bad[np.where(list_index_bad == k)[0]]
            for k in np.unique(list_index_bad)
        }
        dict_ds_bad_idx = {
            k: ds_index_bad[np.where(list_index_bad == k)[0]]
            for k in np.unique(list_index_bad)
        }

        # Print message
        for ds_idx_bad, bad_values in dict_ds_bad_values.items():
            fpath = fpaths[ds_idx_bad]
            msg = f"In {fpath}, dropping {dim} values {bad_values} to ensure monotonic {dim} dimension."
            logger.debug(msg)

        # Remove non-unique and not  da
        for ds_idx_bad, bad_idx in dict_ds_bad_idx.items():
            ds = list_ds[ds_idx_bad]
            list_ds[ds_idx_bad] = ds.drop_isel({dim: bad_idx})

        # Iterative check
        list_ds = ensure_monotonic_dimension(fpaths, list_ds, dim=dim)

    # Return list of xr.Dataset with monotonic dimension
    return list_ds


# ds_index = [0,1,2,3,0,1,2,3,4]
# list_index = [0,0,0,0,1, 1, 1,1, 1]
# dim_values = [0,1,5,5,5, 5, 6,7,8]
# list_index = np.array(list_index)
# dim_values = np.array(dim_values)
# ds_index = np.array(ds_index)


def get_common_coords(list_ds: list) -> set:
    """Get the common set of coordinates across xarray datasets.

    Parameters
    ----------
    list_ds : list
        List of xarray datasets.

    Returns
    -------
    set
        The common set of coordinates.
    """
    # Get common data vars
    coords_ref = set(list_ds[0].coords)
    for ds in list_ds:
        coords_ref = set(ds.coords).intersection(coords_ref)
    return coords_ref


def get_common_vars(list_ds: list) -> tuple:
    """Get the common set of variables across xarray datasets.

    Parameters
    ----------
    list_ds : list
        List of xarray datasets

    Returns
    -------
    tuple
        (The common set of variables, Dictionary to collect problems)
    """
    # Initialize variables in first file
    vars_init = set(list_ds[0].data_vars)
    n_vars_init = len(vars_init)
    # Initialize common variable reference
    common_vars_ref = set(list_ds[0].data_vars)
    # Initialize dictionary to collect problems
    dict_problems = {}
    dict_problems["missing_versus_first"] = {}
    dict_problems["additional_versus_first"] = {}
    dict_problems["evolution"] = {}
    dict_problems["missing_versus_first"]["ref"] = vars_init
    dict_problems["additional_versus_first"]["ref"] = vars_init
    dict_problems["evolution"]["ref"] = vars_init
    # Loop over all datasets
    for i, ds in enumerate(list_ds):
        # Extract current dataset variable info
        vars_ds = set(ds.data_vars)
        n_vars = len(vars_ds)
        # Check if missing variable compared to first
        vars_intersection_vs_first = vars_ds.intersection(vars_init)
        # Collect information on missing variables (compared to first)
        if len(vars_intersection_vs_first) != n_vars_init:
            if len(vars_init.difference(vars_ds)) > 0:
                dict_problems["missing_versus_first"][i] = vars_init.difference(vars_ds)
            if len(vars_ds.difference(vars_init)) > 0:
                dict_problems["additional_versus_first"][i] = vars_ds.difference(
                    vars_init
                )
        # Check if missing an additional variable compared to past common variables
        common_vars_ref_new = vars_ds.intersection(common_vars_ref)
        # If missing, collect information on additional missing variable
        if len(common_vars_ref_new) != len(common_vars_ref):
            dict_problems["evolution"][i] = common_vars_ref.difference(
                common_vars_ref_new
            )

        # Redefine variables common to all datasets
        common_vars_ref = common_vars_ref_new

    # ---------------------------------------------.
    # Return results
    return common_vars_ref, dict_problems


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

    @dask.delayed
    def open_dataset_delayed(fpath):
        ds = xr.open_dataset(fpath, chunks={"time": -1}, engine="netcdf4")
        return ds

    list_ds_delayed = []
    for fpath in fpaths:
        list_ds_delayed.append(open_dataset_delayed(fpath))
    list_ds = dask.compute(list_ds_delayed)[0]
    return list_ds


def ensure_constant_coords(list_ds: list, coords: list) -> tuple:
    """Check coordinates are invariant across xarray datasets.

    Parameters
    ----------
    list_ds : list
        List of xarray datasets.
    coords : list
        List of coordinates names.

    Returns
    -------
    tuple
        (List of xarray datasets, Dictionary to collect problems)
    """
    dict_problems = {}
    for coord in coords:
        ref_set = set(list_ds[0][coord].values)
        dict_coord = {}
        dict_coord["ref"] = ref_set
        for i, ds in enumerate(list_ds):
            values = set(ds[coord].values)
            if ref_set != values:
                dict_coord[i] = values
                list_ds[i] = ds.assign_coords({coord: list(ref_set)})
        dict_problems[coord] = dict_coord
    return list_ds, dict_problems


def xr_concat_datasets(fpaths: str) -> xr.Dataset:
    """Concat xr.Dataset in a robust and parallel way.

    1. It checks for time dimension monotonicity
    2. It checks by default for a minimum common set of variables
    Note: client = Client(processes=True) to execute it fast !


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
    # Ensure time dimension is monotonic increasingly
    list_ds = ensure_monotonic_dimension(fpaths, list_ds, dim="time")

    # --------------------------------------.
    # Ensure coordinate outside time do not vary
    coords_var = get_common_coords(list_ds)
    coords_without_time = coords_var.difference(
        set(["time", "latitude", "longitude", "lat", "lon"])
    )
    list_ds, dict_problems = ensure_constant_coords(list_ds, coords=coords_without_time)
    # --------------------------------------.
    # - Log the possible problems
    for coord in coords_without_time:
        dict_problem = dict_problems[coord]
        ref = dict_problem.pop("ref")
        n_problems = len(dict_problem)
        n_files = len(list_ds)
        if len(dict_problem) != 0:
            msg = f"Difference found in {n_problems}/{n_files} files."
            logger.debug(msg)
            msg = f"In the first file, the coordinate {coord} has values {ref}."
            logger.debug(msg)
            for i, values in dict_problem.items():
                fpath = fpaths[i]
                msg = f"In {fpath}, the coordinate {coord} has values {values}."
                logger.debug(msg)
                logger.info(
                    "Such coordinates have been replaced with the one observed in the first file."
                )

    # --------------------------------------.
    # Get set of common_vars
    common_vars, dict_problems = get_common_vars(list_ds)

    # --------------------------------------.
    # Log the missing variables compared to first file
    dict_problem = dict_problems["missing_versus_first"]
    n_problems = len(dict_problem)
    n_files = len(list_ds)
    ref = dict_problem.pop("ref")
    if len(dict_problem) != 0:
        msg = f"Difference found in {n_problems}/{n_files} files."
        for i, vars in dict_problem.items():
            fpath = fpaths[i]
            msg = f"At file {i}/{n_files} ({fpath}), the variables {vars} are missing (compared to first file)."
            logger.debug(msg)

    # --------------------------------------.
    # Log the additional variables compared to first file
    dict_problem = dict_problems["additional_versus_first"]
    n_problems = len(dict_problem)
    n_files = len(list_ds)
    ref = dict_problem.pop("ref")
    if len(dict_problem) != 0:
        msg = f"Difference found in {n_problems}/{n_files} files."
        for i, vars in dict_problem.items():
            fpath = fpaths[i]
            msg = f"At file {i}/{n_files} ({fpath}), there are the variables {vars} which are missing in the first file."
            logger.debug(msg)

    # --------------------------------------.
    # Log the progressively shrinkage of common variables
    dict_problem = dict_problems["evolution"]
    n_problems = len(dict_problem)
    n_files = len(list_ds)
    ref = dict_problem.pop("ref")
    if len(dict_problem) != 0:
        msg = "Here we report the file which led to shrinkage of the set of common variables."
        logger.debug(msg)
        for i, removed_vars in dict_problem.items():
            fpath = fpaths[i]
            for var in removed_vars:
                msg = f"At file {i}/{n_files} ({fpath}), the {var} is removed from the common set of variables."
                logger.debug(msg)

    # --------------------------------------.
    # Concat/merge all netCDFs
    # - If there are common variables, use xr.concat
    if len(common_vars) > 0:
        # Ensure common set of variables across xr.Datasets
        for i, ds in enumerate(list_ds):
            list_ds[i] = ds[common_vars]
        # Try concatenating
        try:
            logger.info("Start concatenating with xr.concat.")
            ds = xr.concat(list_ds, dim="time", coords="minimal", compat="override")
            logger.info("Concatenation with xr.concat has been successful.")
        except:
            msg = "Concatenation with xr.concat failed."
            logger.error(msg)
            raise ValueError(msg)
    # - Otherwise use xr.merge
    else:
        try:
            logger.info("Start concatenating with xr.merge.")
            ds = xr.merge(
                list_ds, compat="override", join="outer", combine_attrs="override"
            )
            logger.info("Concatenation with xr.merge has been successful.")
        except:
            msg = "Concatenation with xr.merge failed."
            logger.error(msg)
            raise ValueError(msg)
    # --------------------------------------.
    # Return xr.Dataset
    return ds
