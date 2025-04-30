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

import shutil
from typing import Optional

import xarray as xr

from disdrodb.api.path import define_data_dir
from disdrodb.utils.directories import list_files
from disdrodb.utils.logger import (
    log_info,
)


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
    base_dir: Optional[str] = None,
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
        The default is ``False``.
    base_dir : str, optional
        The base directory of DISDRODB, expected in the format ``<...>/DISDRODB``.
        If not specified, the path specified in the DISDRODB active configuration will be used.

    Returns
    -------
    filepaths : list
        List of file paths.

    """
    # Retrieve data directory
    data_dir = define_data_dir(
        base_dir=base_dir,
        data_source=data_source,
        campaign_name=campaign_name,
        station_name=station_name,
        product=product,
        # Product options
        **product_kwargs,
    )

    # Define glob pattern
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


def open_dataset(
    data_source,
    campaign_name,
    station_name,
    product,
    product_kwargs=None,
    debugging_mode: bool = False,
    base_dir: Optional[str] = None,
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
        The default is ``False``.
    base_dir : str, optional
        The base directory of DISDRODB, expected in the format ``<...>/DISDRODB``.
        If not specified, the path specified in the DISDRODB active configuration will be used.

    Returns
    -------
    xr.Dataset

    """
    # Check product validity
    if product == "RAW":
        raise ValueError("It's not possible to open the raw data with this function.")
    product_kwargs = product_kwargs if product_kwargs else {}
    # List product files
    filepaths = find_files(
        base_dir=base_dir,
        data_source=data_source,
        campaign_name=campaign_name,
        station_name=station_name,
        product=product,
        debugging_mode=debugging_mode,
        **product_kwargs,
    )
    # Open L0A Parquet files
    if product == "L0A":
        # TODO: with pandas?
        raise NotImplementedError

    # Open DISDRODB netCDF files using xarray
    # - TODO: parallel option and add closers !
    # - decode_timedelta -- > sample_interval not decoded to timedelta !
    list_ds = [xr.open_dataset(fpath, decode_timedelta=False, **open_kwargs) for fpath in filepaths]
    ds = xr.concat(list_ds, dim="time")
    return ds


####----------------------------------------------------------------------------------
#### DISDRODB Product Removal


def remove_product(
    base_dir,
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
        base_dir=base_dir,
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
