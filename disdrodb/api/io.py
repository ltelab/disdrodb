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
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional

from disdrodb.api.path import (
    define_campaign_dir,
    define_data_dir,
    define_metadata_dir,
    define_station_dir,
)
from disdrodb.l0.l0_reader import define_readers_directory
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


def find_files(
    data_source,
    campaign_name,
    station_name,
    product,
    debugging_mode: bool = False,
    data_archive_dir: Optional[str] = None,
    glob_pattern="*",
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
    sample_interval : int, optional
        The sampling interval in seconds of the product.
        It must be specified only for product L2E and L2M !
    rolling : bool, optional
        Whether the dataset has been resampled by aggregating or rolling.
        It must be specified only for product L2E and L2M !
    model_name : str
        The model name of the statistical distribution for the DSD.
        It must be specified only for product L2M !

    Returns
    -------
    filepaths : list
        List of file paths.

    """
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

    # Define or check the specified glob pattern
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

    # Sort filepaths
    filepaths = sorted(filepaths)
    return filepaths


####----------------------------------------------------------------------------------
#### DISDRODB Open Product Files
def open_dataset(
    data_source,
    campaign_name,
    station_name,
    product,
    product_kwargs=None,
    debugging_mode: bool = False,
    data_archive_dir: Optional[str] = None,
    parallel=False,
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
    sample_interval : int, optional
        The sampling interval in seconds of the product.
        It must be specified only for product L2E and L2M !
    rolling : bool, optional
        Whether the dataset has been resampled by aggregating or rolling.
        It must be specified only for product L2E and L2M !
    model_name : str
        The model name of the statistical distribution for the DSD.
        It must be specified only for product L2M !
    debugging_mode : bool, optional
        If ``True``, it select maximum 3 files for debugging purposes.
        The default value is ``False``.
    data_archive_dir : str, optional
        The base directory of DISDRODB, expected in the format ``<...>/DISDRODB``.
        If not specified, the path specified in the DISDRODB active configuration will be used.

    Returns
    -------
    xarray.Dataset

    """
    import xarray as xr

    from disdrodb.l0.l0a_processing import read_l0a_dataframe

    # Check product validity
    if product == "RAW":
        raise ValueError("It's not possible to open the raw data with this function.")
    product_kwargs = product_kwargs if product_kwargs else {}

    # List product files
    filepaths = find_files(
        data_archive_dir=data_archive_dir,
        data_source=data_source,
        campaign_name=campaign_name,
        station_name=station_name,
        product=product,
        debugging_mode=debugging_mode,
        **product_kwargs,
    )

    # Open L0A Parquet files
    if product == "L0A":
        return read_l0a_dataframe(filepaths)

    # Open DISDRODB netCDF files using xarray
    # - TODO: parallel option and add closers !
    # - decode_timedelta -- > sample_interval not decoded to timedelta !
    # list_ds = [xr.open_dataset(fpath, decode_timedelta=False, **open_kwargs) for fpath in filepaths]
    # ds = xr.concat(list_ds, dim="time")
    ds = xr.open_mfdataset(
        filepaths,
        engine="netcdf4",
        combine="nested",  # 'by_coords',
        concat_dim="time",
        decode_timedelta=False,
        parallel=parallel,
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
    if logger is not None:
        log_info(logger=logger, msg="Removal of {product} files started.", verbose=verbose)
    shutil.rmtree(data_dir)
    if logger is not None:
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
