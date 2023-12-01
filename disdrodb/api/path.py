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
"""Define paths within the DISDRODB infrastructure."""

import os

import pandas as pd
import xarray as xr

from disdrodb.api.info import infer_campaign_name_from_path
from disdrodb.configs import get_base_dir
from disdrodb.utils.directories import check_directory_exists

####--------------------------------------------------------------------------.
#### Paths from BASE_DIR


def get_disdrodb_path(
    base_dir,
    product,
    data_source="",
    campaign_name="",
    check_exists=True,
):
    """Return the directory in the DISDRODB infrastructure.

    If data_source and campaign_name are not specified it return the product directory.
    If data_source is specified, it returns the data_source directory.
    If campaign_source is specified, it returns the campaign_name directory.

    Parameters
    ----------
    base_dir : str
        The disdrodb base directory
    product : str
        The DISDRODB product. It can be "RAW", "L0A", or "L0B".
    data_source : str, optional
        The data source. Must be specified if campaign_name is specified.
    campaign_name : str, optional
        The campaign_name.
    check_exists : bool, optional
        Whether to check if the directory exists. By default True.

    Returns
    -------
    dir_path : str
        Directory path
    """
    from disdrodb.api.checks import check_base_dir

    # Check base_dir validity
    base_dir = check_base_dir(base_dir)
    if len(campaign_name) > 0:
        if len(data_source) == 0:
            raise ValueError("If campaign_name is specified, data_source must be specified.")

    # Get directory
    if product.upper() == "RAW":
        dir_path = os.path.join(base_dir, "Raw", data_source, campaign_name)
    else:
        dir_path = os.path.join(base_dir, "Processed", data_source, campaign_name)
    if check_exists:
        check_directory_exists(dir_path)
    return dir_path


def define_campaign_dir(
    product,
    data_source,
    campaign_name,
    base_dir=None,
    check_exists=False,
):
    """Return the campaign directory in the DISDRODB infrastructure.

    Parameters
    ----------
    product : str
        The DISDRODB product. It can be "RAW", "L0A", or "L0B".
    data_source : str
        The data source. Must be specified if campaign_name is specified.
    campaign_name : str
        The campaign_name.
    base_dir : str, optional
        The base directory of DISDRODB, expected in the format ``<...>/DISDRODB``.
        If not specified, the path specified in the DISDRODB active configuration will be used.
    check_exists : bool, optional
        Whether to check if the directory exists. By default False.

    Returns
    -------
    station_dir : str
        Station data directory path
    """
    base_dir = get_base_dir(base_dir)
    campaign_dir = get_disdrodb_path(
        base_dir=base_dir,
        product=product,
        data_source=data_source,
        campaign_name=campaign_name,
        check_exists=check_exists,
    )
    return str(campaign_dir)


def define_station_dir(
    product,
    data_source,
    campaign_name,
    station_name,
    base_dir=None,
    check_exists=False,
):
    """Return the station data directory in the DISDRODB infrastructure.

    Parameters
    ----------
    product : str
        The DISDRODB product. It can be "RAW", "L0A", or "L0B".
    data_source : str
        The data source.
    campaign_name : str
        The campaign name.
    station_name : str
        The station name.
    base_dir : str, optional
        The base directory of DISDRODB, expected in the format ``<...>/DISDRODB``.
        If not specified, the path specified in the DISDRODB active configuration will be used.
    check_exists : bool, optional
        Whether to check if the directory exists. By default False.

    Returns
    -------
    station_dir : str
        Station data directory path
    """
    base_dir = get_base_dir(base_dir)
    campaign_dir = get_disdrodb_path(
        base_dir=base_dir,
        product=product,
        data_source=data_source,
        campaign_name=campaign_name,
        check_exists=check_exists,
    )
    if product.upper() == "RAW":
        station_dir = os.path.join(campaign_dir, "data", station_name)
    else:
        station_dir = os.path.join(campaign_dir, product, station_name)
    if check_exists:
        check_directory_exists(station_dir)
    return str(station_dir)


def define_metadata_dir(
    product,
    data_source,
    campaign_name,
    base_dir=None,
    check_exists=False,
):
    """Return the metadata directory in the DISDRODB infrastructure.

    Parameters
    ----------
    product : str
        The DISDRODB product. It can be "RAW", "L0A", or "L0B".
    data_source : str
        The data source.
    campaign_name : str
        The campaign name.
    base_dir : str, optional
        The base directory of DISDRODB, expected in the format ``<...>/DISDRODB``.
        If not specified, the path specified in the DISDRODB active configuration will be used.
    check_exists : bool, optional
        Whether to check if the directory exists. By default False.

    Returns
    -------
    metadata_dir : str
        Station data directory path
    """
    base_dir = get_base_dir(base_dir)
    campaign_dir = define_campaign_dir(
        base_dir=base_dir,
        product=product,
        data_source=data_source,
        campaign_name=campaign_name,
        check_exists=check_exists,
    )
    metadata_dir = os.path.join(campaign_dir, "metadata")
    if check_exists:
        check_directory_exists(metadata_dir)
    return str(metadata_dir)


def define_issue_dir(
    data_source,
    campaign_name,
    base_dir=None,
    check_exists=False,
):
    """Return the issue directory in the DISDRODB infrastructure.

    Parameters
    ----------
    data_source : str
        The data source.
    campaign_name : str
        The campaign name.
    base_dir : str, optional
        The base directory of DISDRODB, expected in the format ``<...>/DISDRODB``.
        If not specified, the path specified in the DISDRODB active configuration will be used.
    check_exists : bool, optional
        Whether to check if the directory exists. By default False.

    Returns
    -------
    issue_dir : str
        Station data directory path
    """
    base_dir = get_base_dir(base_dir)
    campaign_dir = define_campaign_dir(
        base_dir=base_dir,
        product="RAW",
        data_source=data_source,
        campaign_name=campaign_name,
        check_exists=check_exists,
    )
    issue_dir = os.path.join(campaign_dir, "issue")
    if check_exists:
        check_directory_exists(issue_dir)
    return str(issue_dir)


def define_metadata_filepath(
    product,
    data_source,
    campaign_name,
    station_name,
    base_dir=None,
    check_exists=False,
):
    """Return the station metadata filepath in the DISDRODB infrastructure.

    Parameters
    ----------
    product : str
        The DISDRODB product. It can be "RAW", "L0A", or "L0B".
    data_source : str
        The data source.
    campaign_name : str
        The campaign name.
    station_name : str
        The station name.
    base_dir : str, optional
        The base directory of DISDRODB, expected in the format ``<...>/DISDRODB``.
        If not specified, the path specified in the DISDRODB active configuration will be used.
    check_exists : bool, optional
        Whether to check if the directory exists. By default False.

    Returns
    -------
    metadata_dir : str
        Station data directory path
    """
    base_dir = get_base_dir(base_dir)
    metadata_dir = define_metadata_dir(
        base_dir=base_dir,
        product=product,
        data_source=data_source,
        campaign_name=campaign_name,
        check_exists=False,
    )
    metadata_filepath = os.path.join(metadata_dir, f"{station_name}.yml")
    if check_exists and not os.path.exists(metadata_filepath):
        raise ValueError(f"The metadata file for {station_name} at {metadata_filepath} does not exists.")

    return str(metadata_filepath)


def define_issue_filepath(
    data_source,
    campaign_name,
    station_name,
    base_dir=None,
    check_exists=False,
):
    """Return the station issue filepath in the DISDRODB infrastructure.

    Parameters
    ----------
    data_source : str
        The data source.
    campaign_name : str
        The campaign name.
    station_name : str
        The station name.
    base_dir : str, optional
        The base directory of DISDRODB, expected in the format ``<...>/DISDRODB``.
        If not specified, the path specified in the DISDRODB active configuration will be used.
    check_exists : bool, optional
        Whether to check if the directory exists. By default False.

    Returns
    -------
    issue_dir : str
        Station data directory path
    """
    base_dir = get_base_dir(base_dir)
    issue_dir = define_issue_dir(
        base_dir=base_dir,
        data_source=data_source,
        campaign_name=campaign_name,
        check_exists=False,
    )
    issue_filepath = os.path.join(issue_dir, f"{station_name}.yml")
    if check_exists and not os.path.exists(issue_filepath):
        raise ValueError(f"The issue file for {station_name} at {issue_filepath} does not exists.")

    return str(issue_filepath)


def define_config_dir(product):
    """Define the config directory path of a given DISDRODB product."""
    from disdrodb import __root_path__

    if product.upper() in ["RAW", "L0A", "L0B"]:
        dir_name = "l0"
    else:
        raise NotImplementedError(f"Product {product} not implemented.")
    config_dir = os.path.join(__root_path__, "disdrodb", dir_name, "configs")
    return config_dir


####--------------------------------------------------------------------------.
#### Directory/Filepaths L0A and L0B products


def define_l0a_station_dir(processed_dir: str, station_name: str) -> str:
    """Define L0A directory.

    Parameters
    ----------
    processed_dir : str
        Path of the processed directory
    station_name : str
        Name of the station

    Returns
    -------
    str
        L0A directory path.
    """
    station_dir = os.path.join(processed_dir, "L0A", station_name)
    return station_dir


def define_l0b_station_dir(processed_dir: str, station_name: str) -> str:
    """Define L0B directory.

    Parameters
    ----------
    processed_dir : str
        Path of the processed directory
    station_name : int
        Name of the station

    Returns
    -------
    str
        Path of the L0B directory
    """
    station_dir = os.path.join(processed_dir, "L0B", station_name)
    return station_dir


def define_l0a_filename(df, processed_dir, station_name: str) -> str:
    """Define L0A file name.

    Parameters
    ----------
    df : pd.DataFrame
        L0A DataFrame
    processed_dir : str
        Path of the processed directory
    station_name : str
        Name of the station

    Returns
    -------
    str
        L0A file name.
    """
    from disdrodb.l0.standards import PRODUCT_VERSION
    from disdrodb.utils.pandas import get_dataframe_start_end_time

    starting_time, ending_time = get_dataframe_start_end_time(df)
    starting_time = pd.to_datetime(starting_time).strftime("%Y%m%d%H%M%S")
    ending_time = pd.to_datetime(ending_time).strftime("%Y%m%d%H%M%S")
    campaign_name = infer_campaign_name_from_path(processed_dir).replace(".", "-")
    version = PRODUCT_VERSION
    filename = f"L0A.{campaign_name}.{station_name}.s{starting_time}.e{ending_time}.{version}.parquet"
    return filename


def define_l0b_filename(ds, processed_dir, station_name: str) -> str:
    """Define L0B file name.

    Parameters
    ----------
    ds : xr.Dataset
        L0B xarray Dataset
    processed_dir : str
        Path of the processed directory
    station_name : str
        Name of the station

    Returns
    -------
    str
        L0B file name.
    """
    from disdrodb.l0.standards import PRODUCT_VERSION
    from disdrodb.utils.xarray import get_dataset_start_end_time

    starting_time, ending_time = get_dataset_start_end_time(ds)
    starting_time = pd.to_datetime(starting_time).strftime("%Y%m%d%H%M%S")
    ending_time = pd.to_datetime(ending_time).strftime("%Y%m%d%H%M%S")
    campaign_name = infer_campaign_name_from_path(processed_dir).replace(".", "-")
    version = PRODUCT_VERSION
    filename = f"L0B.{campaign_name}.{station_name}.s{starting_time}.e{ending_time}.{version}.nc"
    return filename


def define_l0a_filepath(df: pd.DataFrame, processed_dir: str, station_name: str) -> str:
    """Define L0A file path.

    Parameters
    ----------
    df : pd.DataFrame
        L0A DataFrame.
    processed_dir : str
        Path of the processed directory.
    station_name : str
        Name of the station.

    Returns
    -------
    str
        L0A file path.
    """
    filename = define_l0a_filename(df=df, processed_dir=processed_dir, station_name=station_name)
    station_dir = define_l0a_station_dir(processed_dir=processed_dir, station_name=station_name)
    filepath = os.path.join(station_dir, filename)
    return filepath


def define_l0b_filepath(ds: xr.Dataset, processed_dir: str, station_name: str, l0b_concat=False) -> str:
    """Define L0B file path.

    Parameters
    ----------
    ds : xr.Dataset
        L0B xarray Dataset.
    processed_dir : str
        Path of the processed directory.
    station_name : str
        ID of the station
    l0b_concat : bool
        If False, the file is specified inside the station directory.
        If True, the file is specified outside the station directory.

    Returns
    -------
    str
        L0B file path.
    """
    station_dir = define_l0b_station_dir(processed_dir, station_name)
    filename = define_l0b_filename(ds, processed_dir, station_name)
    if l0b_concat:
        product_dir = os.path.dirname(station_dir)
        filepath = os.path.join(product_dir, filename)
    else:
        filepath = os.path.join(station_dir, filename)
    return filepath


####--------------------------------------------------------------------------.
