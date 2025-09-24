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
"""Routines to list and open DISDRODB products."""
import datetime
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional

import numpy as np

from disdrodb.api.checks import (
    check_filepaths,
    check_start_end_time,
    get_current_utc_time,
)
from disdrodb.api.info import get_start_end_time_from_filepaths
from disdrodb.api.path import (
    define_campaign_dir,
    define_data_dir,
    define_metadata_dir,
    define_station_dir,
)
from disdrodb.l0.l0_reader import define_readers_directory
from disdrodb.utils.dict import extract_product_kwargs
from disdrodb.utils.directories import list_files
from disdrodb.utils.logger import (
    log_info,
)

####----------------------------------------------------------------------------------
#### DISDRODB Search Product Files


def filter_filepaths(filepaths, debugging_mode):
    """Filter out filepaths if ``debugging_mode=True``."""
    if debugging_mode:
        max_files = min(3, len(filepaths))
        filepaths = filepaths[0:max_files]
    return filepaths


def is_within_time_period(l_start_time, l_end_time, start_time, end_time):
    """Assess which files are within the start and end time."""
    # - Case 1
    #     s               e
    #     |               |
    #   ---------> (-------->)
    idx_select1 = np.logical_and(l_start_time <= start_time, l_end_time > start_time)
    # - Case 2
    #     s               e
    #     |               |
    #          ---------(-.)
    idx_select2 = np.logical_and(l_start_time >= start_time, l_end_time <= end_time)
    # - Case 3
    #     s               e
    #     |               |
    #                -------------
    idx_select3 = np.logical_and(l_start_time < end_time, l_end_time > end_time)
    # - Get idx where one of the cases occur
    idx_select = np.logical_or.reduce([idx_select1, idx_select2, idx_select3])
    return idx_select


def filter_by_time(filepaths, start_time=None, end_time=None):
    """Filter filepaths by start_time and end_time.

    Parameters
    ----------
    filepaths : list
        List of filepaths.
    start_time : datetime.datetime
        Start time.
        If ``None``, will be set to 1997-01-01.
    end_time : datetime.datetime
        End time.
        If ``None`` will be set to current UTC time.

    Returns
    -------
    filepaths : list
        List of valid filepaths.
        If no valid filepaths, returns an empty list !

    """
    # -------------------------------------------------------------------------.
    # Check filepaths
    if isinstance(filepaths, type(None)):
        return []
    filepaths = check_filepaths(filepaths)
    if len(filepaths) == 0:
        return []

    # -------------------------------------------------------------------------.
    # Check start_time and end_time
    if start_time is None:
        start_time = datetime.datetime(1978, 1, 1, 0, 0, 0)  # Dummy start
    if end_time is None:
        end_time = get_current_utc_time()  # Current time
    start_time, end_time = check_start_end_time(start_time, end_time)

    # -------------------------------------------------------------------------.
    # - Retrieve start_time and end_time of GPM granules
    l_start_time, l_end_time = get_start_end_time_from_filepaths(filepaths)

    # -------------------------------------------------------------------------.
    # Select granules with data within the start and end time
    idx_select = is_within_time_period(l_start_time, l_end_time, start_time=start_time, end_time=end_time)
    return np.array(filepaths)[idx_select].tolist()


def find_files(
    data_source,
    campaign_name,
    station_name,
    product,
    debugging_mode: bool = False,
    data_archive_dir: Optional[str] = None,
    metadata_archive_dir: Optional[str] = None,
    glob_pattern=None,
    start_time=None,
    end_time=None,
    **product_kwargs,
):
    """Retrieve DISDRODB product files for a give station.

    Parameters
    ----------
    data_source : str
        The name of the institution (for campaigns spanning multiple countries) or
        the name of the country (for campaigns or sensor networks within a single country).
        Must be provided in UPPER CASE.
    campaign_name : str
        The name of the campaign. Must be provided in UPPER CASE.
    station_name : str
        The name of the station.
    product : str
        The name DISDRODB product.
    debugging_mode : bool, optional
        If ``True``, it select maximum 3 files for debugging purposes.
        The default value is ``False``.
    data_archive_dir : str, optional
        The base directory of DISDRODB, expected in the format ``<...>/DISDRODB``.
        If not specified, the path specified in the DISDRODB active configuration will be used.
    glob_pattern: str, optional
        Glob pattern to search for raw data files. The default is "*".
        The argument is used only if product="RAW".

    Other Parameters
    ----------------
    temporal_resolution : str, optional
        The temporal resolution of the product (e.g., "1MIN", "10MIN", "1H").
        It must be specified only for product L1, L2E and L2M !
    model_name : str
        The model name of the statistical distribution for the DSD.
        It must be specified only for product L2M !

    Returns
    -------
    filepaths : list
        List of file paths.

    """
    from disdrodb.metadata import read_station_metadata

    # Retrieve data directory
    data_dir = define_data_dir(
        data_archive_dir=data_archive_dir,
        data_source=data_source,
        campaign_name=campaign_name,
        station_name=station_name,
        product=product,
        # Product options
        **product_kwargs,
    )
    # For the DISDRODB RAW product, retrieve glob_pattern from metadata if not specified
    if product == "RAW" and glob_pattern is None:
        metadata = read_station_metadata(
            data_source=data_source,
            campaign_name=campaign_name,
            station_name=station_name,
            metadata_archive_dir=metadata_archive_dir,
        )
        glob_pattern = metadata.get("raw_data_glob_pattern", "")

    # For the others DISDRODB products, define the correct glob pattern
    if product != "RAW":
        glob_pattern = "*.parquet" if product == "L0A" else "*.nc"

    # Retrieve files
    filepaths = list_files(data_dir, glob_pattern=glob_pattern, recursive=True)

    # Filter out filepaths if debugging_mode=True
    filepaths = filter_filepaths(filepaths, debugging_mode=debugging_mode)

    # If no file available, raise error
    if len(filepaths) == 0:
        msg = f"No {product} files are available in {data_dir}. Run {product} processing first."
        raise ValueError(msg)

    # Filter files by start_time and end_time
    if product != "RAW":
        filepaths = filter_by_time(filepaths=filepaths, start_time=start_time, end_time=end_time)
        if len(filepaths) == 0:
            msg = f"No {product} files are available between {start_time} and {end_time}."
            raise ValueError(msg)

    # Sort filepaths
    filepaths = sorted(filepaths)
    return filepaths


####----------------------------------------------------------------------------------
#### DISDRODB Open Product Files


def _open_raw_files(filepaths, data_source, campaign_name, station_name, metadata_archive_dir):
    """Open raw files to DISDRODB L0A or L0B format.

    Raw text files are opened into a DISDRODB L0A pandas Dataframe.
    Raw netCDF files are opened into a DISDRODB L0B xarray Dataset.
    """
    from disdrodb.issue import read_station_issue
    from disdrodb.l0 import generate_l0a, generate_l0b_from_nc, get_station_reader
    from disdrodb.metadata import read_station_metadata

    # Read station metadata
    metadata = read_station_metadata(
        data_source=data_source,
        campaign_name=campaign_name,
        station_name=station_name,
        metadata_archive_dir=metadata_archive_dir,
    )
    sensor_name = metadata["sensor_name"]

    # Read station issue YAML file
    try:
        issue_dict = read_station_issue(
            data_source=data_source,
            campaign_name=campaign_name,
            station_name=station_name,
            metadata_archive_dir=metadata_archive_dir,
        )
    except Exception:
        issue_dict = None

    # Get reader
    reader = get_station_reader(
        data_source=data_source,
        campaign_name=campaign_name,
        station_name=station_name,
        metadata_archive_dir=metadata_archive_dir,
    )
    # Return DISDRODB L0A dataframe if raw text files
    if metadata["raw_data_format"] == "txt":
        df = generate_l0a(
            filepaths=filepaths,
            reader=reader,
            sensor_name=sensor_name,
            issue_dict=issue_dict,
            verbose=False,
        )
        return df

    # Return DISDRODB L0B dataframe if raw netCDF files
    ds = generate_l0b_from_nc(
        filepaths=filepaths,
        reader=reader,
        sensor_name=sensor_name,
        metadata=metadata,
        issue_dict=issue_dict,
        verbose=False,
    )
    return ds


def filter_dataset_by_time(ds, start_time=None, end_time=None):
    """Subset an xarray.Dataset by time, robust to duplicated/non-monotonic indices.

    NOTE: ds.sel(time=slice(start_time, end_time)) fails in presence of duplicated
    timesteps because time 'index is not monotonic increasing or decreasing'.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset with a `time` coordinate.
    start_time : np.datetime64 or None
        Inclusive start bound. If None, no lower bound is applied.
    end_time : np.datetime64 or None
        Inclusive end bound. If None, no upper bound is applied.

    Returns
    -------
    xr.Dataset
        Subset dataset with the same ordering of timesteps (duplicates preserved).
    """
    time = ds["time"].to_numpy()
    mask = np.ones(time.shape, dtype=bool)
    if start_time is not None:
        mask &= time >= np.array(start_time, dtype="datetime64[ns]")
    if end_time is not None:
        mask &= time <= np.array(end_time, dtype="datetime64[ns]")
    return ds.isel(time=np.where(mask)[0])


def open_netcdf_files(
    filepaths,
    chunks=-1,
    start_time=None,
    end_time=None,
    variables=None,
    parallel=False,
    compute=True,
    **open_kwargs,
):
    """Open DISDRODB netCDF files using xarray.

    Using combine="nested" and join="outer" ensure that duplicated timesteps are not overwritten!
    """
    import xarray as xr

    # Ensure variables is a list
    if variables is not None and isinstance(variables, str):
        variables = [variables]
    # Define preprocessing function for parallel opening
    preprocess = (lambda ds: ds[variables]) if parallel and variables is not None else None

    # Open netcdf
    ds = xr.open_mfdataset(
        filepaths,
        chunks=chunks,
        combine="nested",
        join="outer",
        concat_dim="time",
        engine="netcdf4",
        parallel=parallel,
        preprocess=preprocess,
        compat="no_conflicts",
        combine_attrs="override",
        coords="different",  # maybe minimal?
        decode_timedelta=False,
        cache=False,
        autoclose=True,
        **open_kwargs,
    )
    # - Subset variables
    if variables is not None and preprocess is None:
        ds = ds[variables]
    # - Subset time
    if start_time is not None or end_time is not None:
        ds = filter_dataset_by_time(ds, start_time=start_time, end_time=end_time)
    # - If compute=True, load in memory and close connections to files
    if compute:
        dataset = ds.compute()
        ds.close()
        dataset.close()
        del ds
    else:
        dataset = ds
    return dataset


def open_dataset(
    data_source,
    campaign_name,
    station_name,
    product,
    product_kwargs=None,
    debugging_mode: bool = False,
    data_archive_dir: Optional[str] = None,
    metadata_archive_dir: Optional[str] = None,
    chunks=-1,
    parallel=False,
    compute=False,
    start_time=None,
    end_time=None,
    variables=None,
    **open_kwargs,
):
    """Retrieve DISDRODB product files for a give station.

    Parameters
    ----------
    data_source : str
        The name of the institution (for campaigns spanning multiple countries) or
        the name of the country (for campaigns or sensor networks within a single country).
        Must be provided in UPPER CASE.
    campaign_name : str
        The name of the campaign. Must be provided in UPPER CASE.
    station_name : str
        The name of the station.
    product : str
        The name DISDRODB product.
    debugging_mode : bool, optional
        If ``True``, it select maximum 3 files for debugging purposes.
        The default value is ``False``.
    data_archive_dir : str, optional
        The base directory of DISDRODB, expected in the format ``<...>/DISDRODB``.
        If not specified, the path specified in the DISDRODB active configuration will be used.
    **product_kwargs : optional
        DISDRODB product options
        It must be specified only for product L1, L2E and L2M products !
        For L1, L2E and L2M products, temporal_resolution is required
        FOr L2M product, model_name is required.
    **open_kwargs : optional
        Additional keyword arguments passed to ``xarray.open_mfdataset()``.

    Returns
    -------
    xarray.Dataset

    """
    from disdrodb.l0.l0a_processing import read_l0a_dataframe

    # Extract product kwargs from open_kwargs
    product_kwargs = extract_product_kwargs(open_kwargs, product=product)

    # List product files
    filepaths = find_files(
        data_archive_dir=data_archive_dir,
        metadata_archive_dir=metadata_archive_dir,
        data_source=data_source,
        campaign_name=campaign_name,
        station_name=station_name,
        product=product,
        debugging_mode=debugging_mode,
        start_time=start_time,
        end_time=end_time,
        **product_kwargs,
    )

    # Open RAW files
    # - For raw txt files return DISDRODB L0A dataframe
    # - For raw netCDF files return DISDRODB L0B dataframe
    if product == "RAW":
        obj = _open_raw_files(
            filepaths=filepaths,
            data_source=data_source,
            campaign_name=campaign_name,
            station_name=station_name,
            metadata_archive_dir=metadata_archive_dir,
        )
        return obj

    # Open L0A Parquet files
    if product == "L0A":
        return read_l0a_dataframe(filepaths)

    # Open DISDRODB netCDF files using xarray
    ds = open_netcdf_files(
        filepaths=filepaths,
        chunks=chunks,
        start_time=start_time,
        end_time=end_time,
        variables=variables,
        parallel=parallel,
        compute=compute,
        **open_kwargs,
    )
    return ds


####----------------------------------------------------------------------------------
#### DISDRODB Remove Product Files


def remove_product(
    data_archive_dir,
    product,
    data_source,
    campaign_name,
    station_name,
    logger=None,
    verbose=True,
    **product_kwargs,
):
    """Remove all product files of a specific station."""
    if product.upper() == "RAW":
        raise ValueError("Removal of 'RAW' files is not allowed.")
    data_dir = define_data_dir(
        data_archive_dir=data_archive_dir,
        product=product,
        data_source=data_source,
        campaign_name=campaign_name,
        station_name=station_name,
        **product_kwargs,
    )
    log_info(logger=logger, msg="Removal of {product} files started.", verbose=verbose)
    shutil.rmtree(data_dir)
    log_info(logger=logger, msg="Removal of {product} files ended.", verbose=verbose)


####--------------------------------------------------------------------------.
#### Open directories


def open_file_explorer(path):
    """Open the native file-browser showing 'path'."""
    p = Path(path).resolve()
    if not p.exists():
        raise FileNotFoundError(f"{p} does not exist")

    if sys.platform.startswith("win"):
        # Windows
        os.startfile(str(p))
    elif sys.platform == "darwin":
        # macOS
        subprocess.run(["open", str(p)], check=False)
    else:
        # Linux (most desktop environments)
        subprocess.run(["xdg-open", str(p)], check=False)


def open_logs_directory(
    data_source,
    campaign_name,
    station_name=None,  # noqa
    data_archive_dir=None,
):
    """Open the DISDRODB Data Archive logs directory of a station."""
    from disdrodb.configs import get_data_archive_dir

    data_archive_dir = get_data_archive_dir(data_archive_dir)
    campaign_dir = define_campaign_dir(
        archive_dir=data_archive_dir,
        product="L0A",
        data_source=data_source,
        campaign_name=campaign_name,
        check_exists=True,
    )
    logs_dir = os.path.join(campaign_dir, "logs")
    open_file_explorer(logs_dir)


def open_product_directory(
    product,
    data_source,
    campaign_name,
    station_name,
    data_archive_dir=None,
):
    """Open the DISDRODB Data Archive station product directory."""
    from disdrodb.configs import get_data_archive_dir

    data_archive_dir = get_data_archive_dir(data_archive_dir)
    station_dir = define_station_dir(
        data_archive_dir=data_archive_dir,
        product=product,
        data_source=data_source,
        campaign_name=campaign_name,
        station_name=station_name,
        check_exists=True,
    )
    open_file_explorer(station_dir)


def open_metadata_directory(
    data_source,
    campaign_name,
    station_name=None,  # noqa
    metadata_archive_dir=None,
):
    """Open the DISDRODB Metadata Archive station(s) metadata directory."""
    from disdrodb.configs import get_metadata_archive_dir

    metadata_archive_dir = get_metadata_archive_dir(metadata_archive_dir)
    metadata_dir = define_metadata_dir(
        metadata_archive_dir=metadata_archive_dir,
        data_source=data_source,
        campaign_name=campaign_name,
        check_exists=True,
    )
    open_file_explorer(metadata_dir)


def open_readers_directory():
    """Open the disdrodb software readers directory."""
    readers_directory = define_readers_directory()

    open_file_explorer(readers_directory)


def open_metadata_archive(
    metadata_archive_dir=None,
):
    """Open the DISDRODB Metadata Archive."""
    from disdrodb.configs import get_metadata_archive_dir

    metadata_archive_dir = get_metadata_archive_dir(metadata_archive_dir)
    open_file_explorer(metadata_archive_dir)


def open_data_archive(
    data_archive_dir=None,
):
    """Open the DISDRODB Data Archive."""
    from disdrodb.configs import get_data_archive_dir

    data_archive_dir = get_data_archive_dir(data_archive_dir)
    open_file_explorer(data_archive_dir)
