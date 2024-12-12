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
from typing import Optional

import pandas as pd

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

    If ``data_source`` and ``campaign_name`` are not specified it return the product directory.

    If ``data_source`` is specified, it returns the ``data_source`` directory.

    If ``campaign_source`` is specified, it returns the ``campaign_name`` directory.

    Parameters
    ----------
    base_dir : str
        The disdrodb base directory
    product : str
        The DISDRODB product. It can be ``"RAW"``, ``"L0A"``, or ``"L0B"``.
    data_source : str, optional
        The data source. Must be specified if ``campaign_name`` is specified.
    campaign_name : str, optional
        The campaign name.
    check_exists : bool, optional
        Whether to check if the directory exists. By default ``True``.

    Returns
    -------
    dir_path : str
        Directory path
    """
    from disdrodb.api.checks import check_base_dir

    # Check base_dir validity
    base_dir = check_base_dir(base_dir)
    if len(campaign_name) > 0 and len(data_source) == 0:
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
        The DISDRODB product. It can be ``"RAW"``, ``"L0A"``, or ``"L0B"``.
    data_source : str
        The data source. Must be specified if ``campaign_name`` is specified.
    campaign_name : str
        The campaign name.
    base_dir : str, optional
        The base directory of DISDRODB, expected in the format ``<...>/DISDRODB``.
        If not specified, the path specified in the DISDRODB active configuration will be used.
    check_exists : bool, optional
        Whether to check if the directory exists. By default ``False``.

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
        The DISDRODB product. It can be ``"RAW"``, ``"L0A"``, or ``"L0B"``.
    data_source : str
        The data source.
    campaign_name : str
        The campaign name.
    base_dir : str, optional
        The base directory of DISDRODB, expected in the format ``<...>/DISDRODB``.
        If not specified, the path specified in the DISDRODB active configuration will be used.
    check_exists : bool, optional
        Whether to check if the directory exists. By default ``False``.

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
        Whether to check if the directory exists. By default ``False``.

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
        Whether to check if the directory exists. By default ``False``.

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
        Whether to check if the directory exists. By default ``False``.

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


def check_sample_interval(sample_interval):
    """Check sample_interval argument validity."""
    if not isinstance(sample_interval, int):
        raise ValueError("'sample_interval' must be an integer.")


def check_distribution(distribution):
    """Check distribution argument validity."""
    valid_distributions = ["gamma", "normalized_gamma", "lognormal", "exponential"]
    if distribution not in valid_distributions:
        raise ValueError(f"Invalid 'distribution' {distribution}. Valid values are {valid_distributions}")


def check_rolling(rolling):
    """Check rolling argument validity."""
    if not isinstance(rolling, bool):
        raise ValueError("'rolling' must be a boolean.")


def define_product_dir_tree(
    product,
    distribution=None,
    sample_interval=None,
    rolling=None,
):
    """Return the product directory tree.

    Parameters
    ----------
    product : str
        The DISDRODB product. It can be ``"RAW"``, ``"L0A"``, or ``"L0B"``.
    sample_interval : int, optional
        The sampling interval in seconds of the product.
        It must be specified only for product L2E and L2M !
    rolling : bool, optional
        Whether the dataset has been resampled by aggregating or rolling.
        It must be specified only for product L2E and L2M !
    distribution : str
        The model of the statistical distribution for the DSD.
        It must be specified only for product L2M !

    Returns
    -------
    data_dir : str
        Station data directory path
    """
    if product.upper() == "RAW":
        return ""
    if product.upper() in ["L0A", "L0B", "L0C", "L1"]:
        return product
    if product == "L2E":
        check_rolling(rolling)
        check_sample_interval(sample_interval)
        sample_interval_acronym = get_sample_interval_acronym(seconds=sample_interval, rolling=rolling)
        return os.path.join(product, sample_interval_acronym)
    if product == "L2M":
        check_rolling(rolling)
        check_sample_interval(sample_interval)
        check_distribution(distribution)
        sample_interval_acronym = get_sample_interval_acronym(seconds=sample_interval, rolling=rolling)
        distribution_acronym = get_distribution_acronym(distribution)
        return os.path.join(product, distribution_acronym, sample_interval_acronym)
    raise ValueError(f"The product {product} is not defined.")


def define_station_dir_new(
    product,
    data_source,
    campaign_name,
    station_name,
    base_dir=None,
    check_exists=False,
):  # TODO: IN FUTURE without product --> campaign_dir/station_name/product !
    """Return the station data directory in the DISDRODB infrastructure.

    Parameters
    ----------
    product : str
        The DISDRODB product. It can be ``"RAW"``, ``"L0A"``, or ``"L0B"``.
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
        Whether to check if the directory exists. By default ``False``.

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
        station_dir = os.path.join(campaign_dir, station_name, "data")
    if check_exists:
        check_directory_exists(station_dir)
    return str(station_dir)


def define_data_dir_new(
    product,
    data_source,
    campaign_name,
    station_name,
    distribution=None,
    sample_interval=None,
    rolling=None,
    base_dir=None,
    check_exists=False,
):
    """Return the station data directory in the DISDRODB infrastructure.

    Parameters
    ----------
    product : str
        The DISDRODB product. It can be ``"RAW"``, ``"L0A"``, or ``"L0B"``.
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
        Whether to check if the directory exists. By default ``False``.

    Returns
    -------
    station_dir : str
        Station data directory path
    """
    station_dir = define_station_dir_new(
        base_dir=base_dir,
        product=product,
        data_source=data_source,
        campaign_name=campaign_name,
        station_name=station_name,
        check_exists=check_exists,
    )
    product_dir_tree = define_product_dir_tree(
        product=product,
        distribution=distribution,
        sample_interval=sample_interval,
        rolling=rolling,
    )
    data_dir = os.path.join(station_dir, product_dir_tree)
    if check_exists:
        check_directory_exists(data_dir)
    return str(data_dir)


def define_logs_dir(
    product,
    data_source,
    campaign_name,
    station_name,
    distribution=None,
    sample_interval=None,
    rolling=None,
    base_dir=None,
    check_exists=False,
):
    """Return the station log directory in the DISDRODB infrastructure.

    Parameters
    ----------
    product : str
        The DISDRODB product. It can be ``"RAW"``, ``"L0A"``, or ``"L0B"``.
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
        Whether to check if the directory exists. By default ``False``.

    Returns
    -------
    station_dir : str
        Station data directory path
    """
    # station_dir = define_station_dir_new(
    #     base_dir=base_dir,
    #     product=product,
    #     data_source=data_source,
    #     campaign_name=campaign_name,
    #     check_exists=check_exists,
    # )
    campaign_dir = define_campaign_dir(
        base_dir=base_dir,
        product=product,
        data_source=data_source,
        campaign_name=campaign_name,
        check_exists=check_exists,
    )
    product_dir_tree = define_product_dir_tree(
        product=product,
        distribution=distribution,
        sample_interval=sample_interval,
        rolling=rolling,
    )
    logs_dir = os.path.join(campaign_dir, "logs", "files", product_dir_tree, station_name)
    if check_exists:
        check_directory_exists(logs_dir)
    return str(logs_dir)


def define_data_dir(
    product,
    data_source,
    campaign_name,
    station_name,
    distribution=None,
    sample_interval=None,
    rolling=None,
    base_dir=None,
    check_exists=False,
):
    """Return the station data directory in the DISDRODB infrastructure.

    Parameters
    ----------
    product : str
        The DISDRODB product. It can be ``"RAW"``, ``"L0A"``, or ``"L0B"``.
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
        Whether to check if the directory exists. By default ``False``.
    sample_interval : int, optional
        The sampling interval in seconds of the product.
        It must be specified only for product L2E and L2M !
    rolling : bool, optional
        Whether the dataset has been resampled by aggregating or rolling.
        It must be specified only for product L2E and L2M !
    distribution : str
        The model of the statistical distribution for the DSD.
        It must be specified only for product L2M !

    Returns
    -------
    data_dir : str
        Station data directory path
    """
    station_dir = define_station_dir(
        base_dir=base_dir,
        product=product,
        data_source=data_source,
        campaign_name=campaign_name,
        station_name=station_name,
        check_exists=check_exists,
    )
    if product.upper() in ["RAW", "L0A", "L0B", "L0C", "L1"]:
        data_dir = station_dir
    elif product == "L2E":
        check_rolling(rolling)
        check_sample_interval(sample_interval)
        sample_interval_acronym = get_sample_interval_acronym(seconds=sample_interval, rolling=rolling)
        data_dir = os.path.join(station_dir, sample_interval_acronym)
    elif product == "L2M":
        check_rolling(rolling)
        check_sample_interval(sample_interval)
        check_distribution(distribution)
        sample_interval_acronym = get_sample_interval_acronym(seconds=sample_interval, rolling=rolling)
        distribution_acronym = get_distribution_acronym(distribution)
        data_dir = os.path.join(station_dir, distribution_acronym, sample_interval_acronym)
    else:
        raise ValueError("TODO")  # CHECK Product on top !`
    if check_exists:
        check_directory_exists(data_dir)
    return str(data_dir)


def define_product_dir(campaign_dir: str, product: str) -> str:
    """Define product directory."""
    # TODO: this currently only works for L0A and L0B. Should be removed !
    # - Raw: <campaign>/data/<...>
    # - Processed: <campaign>/L0A/L0B>
    if product.upper() == "RAW":
        product_dir = os.path.join(campaign_dir, "data")
    else:
        product_dir = os.path.join(campaign_dir, product)
    return product_dir


def define_station_dir(
    product,
    data_source,
    campaign_name,
    station_name,
    base_dir=None,
    check_exists=False,
):  # TODO: IN FUTURE without product --> campaign_dir/station_name/product !
    """Return the station data directory in the DISDRODB infrastructure.

    Parameters
    ----------
    product : str
        The DISDRODB product. It can be ``"RAW"``, ``"L0A"``, or ``"L0B"``.
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
        Whether to check if the directory exists. By default ``False``.

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


####--------------------------------------------------------------------------.
#### Filenames for DISDRODB products


def get_distribution_acronym(distribution):
    """Define DISDRODB L2M distribution acronym."""
    acronym_dict = {
        "lognorm": "LOGNORM",
        "normalized_gamma": "NGAMMA",
        "gamma": "GAMMA",
        "exponential": "EXP",
    }
    return acronym_dict[distribution]


def get_sample_interval_acronym(seconds, rolling=False):
    """
    Convert a duration in seconds to a readable string format (e.g., "1H30", "1D2H").

    Parameters
    ----------
    - seconds (int): The time duration in seconds.

    Returns
    -------
    - str: The duration as a string in a format like "30S", "1MIN30S", "1H30MIN", or "1D2H".
    """
    timedelta = pd.Timedelta(seconds=seconds)
    components = timedelta.components

    parts = []
    if components.days > 0:
        parts.append(f"{components.days}D")
    if components.hours > 0:
        parts.append(f"{components.hours}H")
    if components.minutes > 0:
        parts.append(f"{components.minutes}MIN")
    if components.seconds > 0:
        parts.append(f"{components.seconds}S")
    sample_interval_acronym = "".join(parts)
    # Prefix with ROLL if rolling=True
    if rolling:
        sample_interval_acronym = f"ROLL{sample_interval_acronym}"
    return sample_interval_acronym


def define_filename(
    product: str,
    campaign_name: str,
    station_name: str,
    # L2E option
    sample_interval: Optional[int] = None,
    rolling: Optional[bool] = None,
    # L2M option
    distribution: Optional[str] = None,
    # Filename options
    obj=None,
    add_version=True,
    add_time_period=True,
    add_extension=True,
    # Prefix
    prefix="",
    suffix="",
) -> str:
    """Define DISDRODB products filename.

    Parameters
    ----------
    obj  : xarray.Dataset or pandas.DataFrame
        xarray Dataset or pandas DataFrame.
        Required if add_time_period = True.
    campaign_name : str
       Name of the campaign.
    station_name : str
       Name of the station.
    sample_interval : int, optional
        The sampling interval in seconds of the product.
        It must be specified only for product L2E and L2M !
    rolling : bool, optional
        Whether the dataset has been resampled by aggregating or rolling.
        It must be specified only for product L2E and L2M !
    distribution : str
        The model of the statistical distribution for the DSD.
        It must be specified only for product L2M !

    Returns
    -------
    str
        L0B file name.
    """
    from disdrodb import PRODUCT_VERSION
    from disdrodb.utils.pandas import get_dataframe_start_end_time
    from disdrodb.utils.xarray import get_dataset_start_end_time

    # -----------------------------------------.
    # Define product acronym
    product_acronym = f"{product}"
    if product in ["L2E", "L2M"]:
        sample_interval_acronym = get_sample_interval_acronym(seconds=sample_interval)
        if rolling:
            sample_interval_acronym = f"ROLL{sample_interval_acronym}"
        product_acronym = f"L2E.{sample_interval_acronym}"
    if product in ["L2M"]:
        distribution_acronym = get_distribution_acronym(distribution)
        product_acronym = f"L2M_{distribution_acronym}.{sample_interval_acronym}"

    # -----------------------------------------.
    # Define base filename
    filename = f"{product_acronym}.{campaign_name}.{station_name}"

    # -----------------------------------------.
    # Add prefix
    if prefix != "":
        filename = f"{prefix}.{filename}"

    # -----------------------------------------.
    # Add time period information
    if add_time_period:
        if product == "L0A":
            starting_time, ending_time = get_dataframe_start_end_time(obj)
        else:
            starting_time, ending_time = get_dataset_start_end_time(obj)
        starting_time = pd.to_datetime(starting_time).strftime("%Y%m%d%H%M%S")
        ending_time = pd.to_datetime(ending_time).strftime("%Y%m%d%H%M%S")
        filename = f"{filename}.s{starting_time}.e{ending_time}"

    # -----------------------------------------.
    # Add product version
    if add_version:
        filename = f"{filename}.{PRODUCT_VERSION}"

    # -----------------------------------------.
    # Add product extension
    if add_extension:
        filename = f"{filename}.parquet" if product == "L0A" else f"{filename}.nc"

    # -----------------------------------------.
    # Add suffix
    if suffix != "":
        filename = f"{filename}.{suffix}"
    return filename


def define_l0a_filename(df, campaign_name: str, station_name: str) -> str:
    """Define L0A file name.

    Parameters
    ----------
    df : pandas.DataFrame
        L0A DataFrame.
    campaign_name : str
        Name of the campaign.
    station_name : str
        Name of the station.

    Returns
    -------
    str
        L0A file name.
    """
    from disdrodb import PRODUCT_VERSION
    from disdrodb.utils.pandas import get_dataframe_start_end_time

    starting_time, ending_time = get_dataframe_start_end_time(df)
    starting_time = pd.to_datetime(starting_time).strftime("%Y%m%d%H%M%S")
    ending_time = pd.to_datetime(ending_time).strftime("%Y%m%d%H%M%S")
    version = PRODUCT_VERSION
    filename = f"L0A.{campaign_name}.{station_name}.s{starting_time}.e{ending_time}.{version}.parquet"
    return filename


def define_l0b_filename(ds, campaign_name: str, station_name: str) -> str:
    """Define L0B file name.

    Parameters
    ----------
    ds  : xarray.Dataset
        L0B xarray Dataset.
    campaign_name : str
        Name of the campaign.
    station_name : str
        Name of the station.

    Returns
    -------
    str
        L0B file name.
    """
    from disdrodb import PRODUCT_VERSION
    from disdrodb.utils.xarray import get_dataset_start_end_time

    starting_time, ending_time = get_dataset_start_end_time(ds)
    starting_time = pd.to_datetime(starting_time).strftime("%Y%m%d%H%M%S")
    ending_time = pd.to_datetime(ending_time).strftime("%Y%m%d%H%M%S")
    version = PRODUCT_VERSION
    filename = f"L0B.{campaign_name}.{station_name}.s{starting_time}.e{ending_time}.{version}.nc"
    return filename


def define_l0c_filename(ds, campaign_name: str, station_name: str) -> str:
    """Define L0C file name.

    Parameters
    ----------
    ds  : xarray.Dataset
        L0B xarray Dataset
    campaign_name : str
        Name of the campaign
    station_name : str
        Name of the station

    Returns
    -------
    str
        L0B file name.
    """
    from disdrodb import PRODUCT_VERSION
    from disdrodb.utils.xarray import get_dataset_start_end_time

    starting_time, ending_time = get_dataset_start_end_time(ds)
    starting_time = pd.to_datetime(starting_time).strftime("%Y%m%d%H%M%S")
    ending_time = pd.to_datetime(ending_time).strftime("%Y%m%d%H%M%S")
    version = PRODUCT_VERSION
    filename = f"L0C.{campaign_name}.{station_name}.s{starting_time}.e{ending_time}.{version}.nc"
    return filename


def define_l1_filename(ds, campaign_name, station_name: str) -> str:
    """Define L1 file name.

    Parameters
    ----------
    ds  : xarray.Dataset
        L1 xarray Dataset
    processed_dir : str
        Path of the processed directory
    station_name : str
        Name of the station

    Returns
    -------
    str
        L0B file name.
    """
    from disdrodb import PRODUCT_VERSION
    from disdrodb.utils.xarray import get_dataset_start_end_time

    starting_time, ending_time = get_dataset_start_end_time(ds)
    starting_time = pd.to_datetime(starting_time).strftime("%Y%m%d%H%M%S")
    ending_time = pd.to_datetime(ending_time).strftime("%Y%m%d%H%M%S")
    version = PRODUCT_VERSION
    filename = f"L1.{campaign_name}.{station_name}.s{starting_time}.e{ending_time}.{version}.nc"
    return filename


def define_l2e_filename(ds, campaign_name: str, station_name: str, sample_interval: int, rolling: bool) -> str:
    """Define L2E file name.

    Parameters
    ----------
    ds  : xarray.Dataset
        L1 xarray Dataset
    processed_dir : str
        Path of the processed directory
    station_name : str
        Name of the station

    Returns
    -------
    str
        L0B file name.
    """
    from disdrodb import PRODUCT_VERSION
    from disdrodb.utils.xarray import get_dataset_start_end_time

    sample_interval_acronym = get_sample_interval_acronym(seconds=sample_interval)
    if rolling:
        sample_interval_acronym = f"ROLL{sample_interval_acronym}"
    starting_time, ending_time = get_dataset_start_end_time(ds)
    starting_time = pd.to_datetime(starting_time).strftime("%Y%m%d%H%M%S")
    ending_time = pd.to_datetime(ending_time).strftime("%Y%m%d%H%M%S")
    version = PRODUCT_VERSION
    filename = (
        f"L2E.{sample_interval_acronym}.{campaign_name}.{station_name}.s{starting_time}.e{ending_time}.{version}.nc"
    )
    return filename


def define_l2m_filename(
    ds,
    campaign_name: str,
    station_name: str,
    distribution: str,
    sample_interval: int,
    rolling: bool,
) -> str:
    """Define L2M file name.

    Parameters
    ----------
    ds  : xarray.Dataset
        L1 xarray Dataset
    processed_dir : str
        Path of the processed directory
    station_name : str
        Name of the station

    Returns
    -------
    str
        L0B file name.
    """
    from disdrodb import PRODUCT_VERSION
    from disdrodb.utils.xarray import get_dataset_start_end_time

    distribution_acronym = get_distribution_acronym(distribution)
    sample_interval_acronym = get_sample_interval_acronym(seconds=sample_interval)
    if rolling:
        sample_interval_acronym = f"ROLL{sample_interval_acronym}"
    starting_time, ending_time = get_dataset_start_end_time(ds)
    starting_time = pd.to_datetime(starting_time).strftime("%Y%m%d%H%M%S")
    ending_time = pd.to_datetime(ending_time).strftime("%Y%m%d%H%M%S")
    version = PRODUCT_VERSION
    filename = (
        f"L2M_{distribution_acronym}.{sample_interval_acronym}.{campaign_name}."
        + f"{station_name}.s{starting_time}.e{ending_time}.{version}.nc"
    )
    return filename
