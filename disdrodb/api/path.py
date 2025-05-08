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

from disdrodb.configs import get_data_archive_dir, get_metadata_archive_dir
from disdrodb.utils.directories import check_directory_exists
from disdrodb.utils.time import (
    ensure_sample_interval_in_seconds,
    get_file_start_end_time,
    seconds_to_acronym,
)

####--------------------------------------------------------------------------.
#### DISDRODB Metadata and Data Archive directories and file paths


def define_disdrodb_path(
    archive_dir,
    product,
    data_source="",
    campaign_name="",
    check_exists=True,
):
    """Return the directory path in the DISDRODB Metadata and Data Archive.

    If ``product="METADATA"``, it returns the path in the DISDRODB Metadata Archive.
    Otherwise, it returns the path in the DISDRODB Data Archive.

    If ``data_source`` and ``campaign_name`` are not specified it return the product directory.

    If ``data_source`` is specified, it returns the ``data_source`` directory.

    If ``campaign_source`` is specified, it returns the ``campaign_name`` directory.

    Parameters
    ----------
    archive_dir : str
        The DISDRODB archive directory
    product : str
        The DISDRODB product. See ``disdrodb.available_products()``.
        If "METADATA" is specified, it returns the path in the DISDRODB Metadata Archive.
    data_source : str, optional
        The data source. Must be specified if ``campaign_name`` is specified.
    campaign_name : str, optional
        The campaign name.
    check_exists : bool, optional
        Whether to check if the directory exists. The default value is ``True``.

    Returns
    -------
    dir_path : str
        Directory path
    """
    from disdrodb import ARCHIVE_VERSION

    if len(campaign_name) > 0 and len(data_source) == 0:
        raise ValueError("If campaign_name is specified, data_source must be specified.")

    # Get directory
    if product.upper() == "METADATA":
        dir_path = os.path.join(archive_dir, "METADATA", data_source, campaign_name)
    elif product.upper() == "RAW":
        dir_path = os.path.join(archive_dir, "RAW", data_source, campaign_name)
    else:
        dir_path = os.path.join(archive_dir, ARCHIVE_VERSION, data_source, campaign_name)
    if check_exists:
        check_directory_exists(dir_path)
    return dir_path


def define_data_source_dir(
    archive_dir,
    product,
    data_source,
    check_exists=False,
):
    """Return the data source directory in the DISDRODB infrastructure.

    If ``product="METADATA"``, it returns the path in the DISDRODB Metadata Archive.
    Otherwise, it returns the path in the DISDRODB Data Archive.

    Parameters
    ----------
    product : str
        The DISDRODB product. See ``disdrodb.available_products()``.
        If "METADATA" is specified, it returns the path in the DISDRODB Metadata Archive.
    data_source : str
        The data source.
    archive_dir : str, optional
        The base directory of DISDRODB, expected in the format ``<...>/DISDRODB``.
        If not specified, the path specified in the DISDRODB active configuration will be used.
    check_exists : bool, optional
        Whether to check if the directory exists. The default value is ``False``.

    Returns
    -------
    station_dir : str
        Station data directory path
    """
    data_source_dir = define_disdrodb_path(
        archive_dir=archive_dir,
        product=product,
        data_source=data_source,
        check_exists=check_exists,
    )
    return str(data_source_dir)


def define_campaign_dir(
    archive_dir,
    product,
    data_source,
    campaign_name,
    check_exists=False,
):
    """Return the campaign directory in the DISDRODB infrastructure.

    If ``product="METADATA"``, it returns the path in the DISDRODB Metadata Archive.
    Otherwise, it returns the path in the DISDRODB Data Archive.

    Parameters
    ----------
    product : str
        The DISDRODB product. See ``disdrodb.available_products()``.
        If "METADATA" is specified, it returns the path in the DISDRODB Metadata Archive.
    data_source : str
        The data source. Must be specified if ``campaign_name`` is specified.
    campaign_name : str
        The campaign name.
    archive_dir : str, optional
        The base directory of DISDRODB, expected in the format ``<...>/DISDRODB``.
        If not specified, the path specified in the DISDRODB active configuration will be used.
    check_exists : bool, optional
        Whether to check if the directory exists. The default value is ``False``.

    Returns
    -------
    station_dir : str
        Station data directory path
    """
    campaign_dir = define_disdrodb_path(
        archive_dir=archive_dir,
        product=product,
        data_source=data_source,
        campaign_name=campaign_name,
        check_exists=check_exists,
    )
    return str(campaign_dir)


def define_metadata_dir(
    data_source,
    campaign_name,
    metadata_archive_dir=None,
    check_exists=False,
):
    """Return the metadata directory in the DISDRODB infrastructure.

    Parameters
    ----------
    data_source : str
        The data source.
    campaign_name : str
        The campaign name.
    data_archive_dir : str, optional
        The base directory of DISDRODB, expected in the format ``<...>/DISDRODB``.
        If not specified, the path specified in the DISDRODB active configuration will be used.
    check_exists : bool, optional
        Whether to check if the directory exists. The default value is ``False``.

    Returns
    -------
    metadata_archive_dir : str
        Station data directory path
    """
    metadata_archive_dir = get_metadata_archive_dir(metadata_archive_dir)
    campaign_dir = define_campaign_dir(
        archive_dir=metadata_archive_dir,
        data_source=data_source,
        product="METADATA",
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
    metadata_archive_dir=None,
    check_exists=False,
):
    """Return the issue directory in the DISDRODB infrastructure.

    Parameters
    ----------
    data_source : str
        The data source.
    campaign_name : str
        The campaign name.
    data_archive_dir : str, optional
        The base directory of DISDRODB, expected in the format ``<...>/DISDRODB``.
        If not specified, the path specified in the DISDRODB active configuration will be used.
    check_exists : bool, optional
        Whether to check if the directory exists. The default value is ``False``.

    Returns
    -------
    issue_dir : str
        Station data directory path
    """
    metadata_archive_dir = get_metadata_archive_dir(metadata_archive_dir)
    campaign_dir = define_campaign_dir(
        archive_dir=metadata_archive_dir,
        product="METADATA",
        data_source=data_source,
        campaign_name=campaign_name,
        check_exists=check_exists,
    )
    issue_dir = os.path.join(campaign_dir, "issue")
    if check_exists:
        check_directory_exists(issue_dir)
    return str(issue_dir)


def define_metadata_filepath(
    data_source,
    campaign_name,
    station_name,
    metadata_archive_dir=None,
    check_exists=False,
):
    """Return the station metadata filepath in the DISDRODB infrastructure.

    Parameters
    ----------
    data_source : str
        The data source.
    campaign_name : str
        The campaign name.
    station_name : str
        The station name.
    data_archive_dir : str, optional
        The base directory of DISDRODB, expected in the format ``<...>/DISDRODB``.
        If not specified, the path specified in the DISDRODB active configuration will be used.
    check_exists : bool, optional
        Whether to check if the directory exists. The default value is ``False``.

    Returns
    -------
    metadata_archive_dir : str
        Station data directory path
    """
    metadata_dir = get_metadata_archive_dir(metadata_archive_dir)
    metadata_dir = define_metadata_dir(
        metadata_archive_dir=metadata_archive_dir,
        data_source=data_source,
        campaign_name=campaign_name,
        check_exists=False,
    )
    metadata_filepath = os.path.join(metadata_dir, f"{station_name}.yml")
    if check_exists and not os.path.exists(metadata_filepath):
        raise ValueError(
            f"The station {data_source} {campaign_name} {station_name} does not exist. "
            f"No metadata file is available at {metadata_filepath}.",
        )

    return str(metadata_filepath)


def define_issue_filepath(
    data_source,
    campaign_name,
    station_name,
    metadata_archive_dir=None,
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
    data_archive_dir : str, optional
        The base directory of DISDRODB, expected in the format ``<...>/DISDRODB``.
        If not specified, the path specified in the DISDRODB active configuration will be used.
    check_exists : bool, optional
        Whether to check if the directory exists. The default value is ``False``.

    Returns
    -------
    issue_dir : str
        Station data directory path
    """
    metadata_archive_dir = get_metadata_archive_dir(metadata_archive_dir)
    issue_dir = define_issue_dir(
        metadata_archive_dir=metadata_archive_dir,
        data_source=data_source,
        campaign_name=campaign_name,
        check_exists=False,
    )
    issue_filepath = os.path.join(issue_dir, f"{station_name}.yml")
    if check_exists and not os.path.exists(issue_filepath):
        raise ValueError(f"The issue file for {station_name} at {issue_filepath} does not exists.")
    return str(issue_filepath)


####--------------------------------------------------------------------------.
#### DISDRODB software configuration directory


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


def define_file_folder_path(obj, data_dir, folder_partitioning):
    """
    Define the folder path where saving a file based on the dataset's starting time.

    Parameters
    ----------
    ds : xarray.Dataset or pandas.DataFrame
        The object containing time information.
    data_dir : str
        Directory within the DISDRODB Data Archive where DISDRODB product files are to be saved.
    folder_partitioning : str or None
        Define the subdirectory structure where saving files.
        Allowed values are:
          - None: Files are saved directly in data_dir.
          - "year": Files are saved under a subdirectory for the year.
          - "year/month": Files are saved under subdirectories for year and month.
          - "year/month/day": Files are saved under subdirectories for year, month and day
          - "year/month_name": Files are stored under subdirectories by year and month name
          - "year/quarter": Files are saved under subdirectories for year and quarter.

    Returns
    -------
    str
        A complete directory path where the file should be saved.
    """
    from disdrodb.api.checks import check_folder_partitioning

    # Validate the folder partition parameter.
    check_folder_partitioning(folder_partitioning)

    # Retrieve the starting time from the dataset.
    starting_time, _ = get_file_start_end_time(obj)

    # Build the folder path based on the chosen partition scheme.
    if folder_partitioning == "":
        return data_dir
    if folder_partitioning == "year":
        year = str(starting_time.year)
        return os.path.join(data_dir, year)
    if folder_partitioning == "year/month":
        year = str(starting_time.year)
        month = str(starting_time.month).zfill(2)
        return os.path.join(data_dir, year, month)
    if folder_partitioning == "year/month/day":
        year = str(starting_time.year)
        month = str(starting_time.month).zfill(2)
        day = str(starting_time.day).zfill(2)
        return os.path.join(data_dir, year, month, day)
    if folder_partitioning == "year/month_name":
        year = str(starting_time.year)
        month = str(starting_time.month_name())
        return os.path.join(data_dir, year, month)
    if folder_partitioning == "year/quarter":
        year = str(starting_time.year)
        # Calculate quarter: months 1-3 => Q1, 4-6 => Q2, etc.
        quarter = (starting_time.month - 1) // 3 + 1
        quarter_dir = f"Q{quarter}"
        return os.path.join(data_dir, year, quarter_dir)
    raise NotImplementedError(f"Unrecognized '{folder_partitioning}' folder partitioning scheme.")


def define_product_dir_tree(
    product,
    **product_kwargs,
):
    """Return the product directory tree.

    Parameters
    ----------
    product : str
       The DISDRODB product. See ``disdrodb.available_products()``.
    sample_interval : int, optional
        The sampling interval in seconds of the product.
        It must be specified only for product L2E and L2M !
    rolling : bool, optional
        Whether the dataset has been resampled by aggregating or rolling.
        It must be specified only for product L2E and L2M !
    model_name : str
        The custom model name of the fitted statistical distribution.
        It must be specified only for product L2M !

    Returns
    -------
    data_dir : str
        Station data directory path
    """
    from disdrodb.api.checks import check_product, check_product_kwargs, check_rolling, check_sample_interval

    product = check_product(product)
    product_kwargs = check_product_kwargs(product, product_kwargs)
    if product.upper() == "RAW":
        return ""
    if product.upper() in ["L0A", "L0B", "L0C", "L1"]:
        return ""
    if product == "L2E":
        rolling = product_kwargs.get("rolling")
        sample_interval = product_kwargs.get("sample_interval")
        check_rolling(rolling)
        check_sample_interval(sample_interval)
        sample_interval_acronym = define_accumulation_acronym(seconds=sample_interval, rolling=rolling)
        return os.path.join(sample_interval_acronym)
    if product == "L2M":
        rolling = product_kwargs.get("rolling")
        sample_interval = product_kwargs.get("sample_interval")
        model_name = product_kwargs.get("model_name")
        check_rolling(rolling)
        check_sample_interval(sample_interval)
        sample_interval_acronym = define_accumulation_acronym(seconds=sample_interval, rolling=rolling)
        return os.path.join(model_name, sample_interval_acronym)
    raise ValueError(f"The product {product} is not defined.")


def define_logs_dir(
    product,
    data_source,
    campaign_name,
    station_name,
    data_archive_dir=None,
    check_exists=False,
    **product_kwargs,
):
    """Return the station log directory in the DISDRODB infrastructure.

    Parameters
    ----------
    product : str
       The DISDRODB product. See ``disdrodb.available_products()``.
    data_source : str
        The data source.
    campaign_name : str
        The campaign name.
    station_name : str
        The station name.
    data_archive_dir : str, optional
        The base directory of DISDRODB, expected in the format ``<...>/DISDRODB``.
        If not specified, the path specified in the DISDRODB active configuration will be used.
    check_exists : bool, optional
        Whether to check if the directory exists. The default value is ``False``.

    Returns
    -------
    station_dir : str
        Station data directory path
    """
    campaign_dir = define_campaign_dir(
        archive_dir=data_archive_dir,
        product=product,
        data_source=data_source,
        campaign_name=campaign_name,
        check_exists=check_exists,
    )
    product_dir_tree = define_product_dir_tree(
        product=product,
        **product_kwargs,
    )
    logs_dir = os.path.join(campaign_dir, "logs", "files", product, product_dir_tree, station_name)
    if check_exists:
        check_directory_exists(logs_dir)
    return str(logs_dir)


def define_station_dir(
    product,
    data_source,
    campaign_name,
    station_name,
    data_archive_dir=None,
    check_exists=False,
):
    """Return the station product directory in the DISDRODB infrastructure.

    Parameters
    ----------
    product : str
       The DISDRODB product. See ``disdrodb.available_products()``.
    data_source : str
        The data source.
    campaign_name : str
        The campaign name.
    station_name : str
        The station name.
    data_archive_dir : str, optional
        The base directory of DISDRODB, expected in the format ``<...>/DISDRODB``.
        If not specified, the path specified in the DISDRODB active configuration will be used.
    check_exists : bool, optional
        Whether to check if the directory exists. The default value is ``False``.

    Returns
    -------
    station_dir : str
        Station data directory path
    """
    data_archive_dir = get_data_archive_dir(data_archive_dir)
    campaign_dir = define_disdrodb_path(
        archive_dir=data_archive_dir,
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


def define_data_dir(
    product,
    data_source,
    campaign_name,
    station_name,
    data_archive_dir=None,
    check_exists=False,
    **product_kwargs,
):
    """Return the station product data directory in the DISDRODB infrastructure.

    Parameters
    ----------
    product : str
       The DISDRODB product. See ``disdrodb.available_products()``.
    data_source : str
        The data source.
    campaign_name : str
        The campaign name.
    station_name : str
        The station name.
    data_archive_dir : str, optional
        The base directory of DISDRODB, expected in the format ``<...>/DISDRODB``.
        If not specified, the path specified in the DISDRODB active configuration will be used.
    check_exists : bool, optional
        Whether to check if the directory exists. The default value is ``False``.
    sample_interval : int, optional
        The sampling interval in seconds of the product.
        It must be specified only for product L2E and L2M !
    rolling : bool, optional
        Whether the dataset has been resampled by aggregating or rolling.
        It must be specified only for product L2E and L2M !
    model_name : str
        The name of the fitted statistical distribution for the DSD.
        It must be specified only for product L2M !

    Returns
    -------
    data_dir : str
        Station data directory path
    """
    from disdrodb.api.checks import check_product, check_product_kwargs

    product = check_product(product)
    product_kwargs = check_product_kwargs(product, product_kwargs)
    # Define station directory
    station_dir = define_station_dir(
        data_archive_dir=data_archive_dir,
        product=product,
        data_source=data_source,
        campaign_name=campaign_name,
        station_name=station_name,
        check_exists=check_exists,
    )  # <product>/<station_name>
    # Define product directory directory (i.e. for L2E and L2M)
    product_dir_tree = define_product_dir_tree(
        product,
        **product_kwargs,
    )
    # Define data directory
    data_dir = os.path.join(station_dir, product_dir_tree)
    # Check if directory exists
    if check_exists:
        check_directory_exists(data_dir)
    return str(data_dir)


####--------------------------------------------------------------------------.
#### Filenames for DISDRODB products


def define_accumulation_acronym(seconds, rolling):
    """Define the accumulation acronnym.

    Prefix the accumulation interval acronym with ROLL if rolling=True.
    """
    accumulation_acronym = seconds_to_acronym(seconds)
    if rolling:
        accumulation_acronym = f"ROLL{accumulation_acronym}"
    return accumulation_acronym


####--------------------------------------------------------------------------.
#### Filenames for DISDRODB products


def define_filename(
    product: str,
    campaign_name: str,
    station_name: str,
    # Filename options
    obj=None,
    add_version=True,
    add_time_period=True,
    add_extension=True,
    # Prefix
    prefix="",
    suffix="",
    # Product options
    **product_kwargs,
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
    model_name : str
        The model name of the fitted statistical distribution for the DSD.
        It must be specified only for product L2M !

    Returns
    -------
    str
        L0B file name.
    """
    from disdrodb import ARCHIVE_VERSION
    from disdrodb.api.checks import check_product, check_product_kwargs

    product = check_product(product)
    product_kwargs = check_product_kwargs(product, product_kwargs)

    # -----------------------------------------.
    # TODO: Define sample_interval_acronym
    # - ADD sample_interval_acronym also to L0A and L0B
    # - Add sample_interval_acronym also to L0C and L1

    # -----------------------------------------.
    # Define product acronym
    product_acronym = f"{product}"
    if product in ["L2E", "L2M"]:
        rolling = product_kwargs.get("rolling")
        sample_interval = product_kwargs.get("sample_interval")
        sample_interval_acronym = define_accumulation_acronym(seconds=sample_interval, rolling=rolling)
        product_acronym = f"L2E.{sample_interval_acronym}"
    if product in ["L2M"]:
        model_name = product_kwargs.get("model_name")
        product_acronym = f"L2M_{model_name}.{sample_interval_acronym}"

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
        starting_time, ending_time = get_file_start_end_time(obj)
        starting_time = starting_time.strftime("%Y%m%d%H%M%S")
        ending_time = ending_time.strftime("%Y%m%d%H%M%S")
        filename = f"{filename}.s{starting_time}.e{ending_time}"

    # -----------------------------------------.
    # Add product version
    if add_version:
        filename = f"{filename}.{ARCHIVE_VERSION}"

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
    from disdrodb import ARCHIVE_VERSION

    starting_time, ending_time = get_file_start_end_time(df)
    starting_time = starting_time.strftime("%Y%m%d%H%M%S")
    ending_time = ending_time.strftime("%Y%m%d%H%M%S")
    version = ARCHIVE_VERSION
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
    from disdrodb import ARCHIVE_VERSION

    starting_time, ending_time = get_file_start_end_time(ds)
    starting_time = starting_time.strftime("%Y%m%d%H%M%S")
    ending_time = ending_time.strftime("%Y%m%d%H%M%S")
    version = ARCHIVE_VERSION
    filename = f"L0B.{campaign_name}.{station_name}.s{starting_time}.e{ending_time}.{version}.nc"
    return filename


def define_l0c_filename(ds, campaign_name: str, station_name: str) -> str:
    """Define L0C file name.

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
    from disdrodb import ARCHIVE_VERSION

    # TODO: add sample_interval as argument
    sample_interval = int(ensure_sample_interval_in_seconds(ds["sample_interval"]).data.item())
    sample_interval_acronym = define_accumulation_acronym(sample_interval, rolling=False)
    starting_time, ending_time = get_file_start_end_time(ds)
    starting_time = starting_time.strftime("%Y%m%d%H%M%S")
    ending_time = ending_time.strftime("%Y%m%d%H%M%S")
    version = ARCHIVE_VERSION
    filename = (
        f"L0C.{sample_interval_acronym}.{campaign_name}.{station_name}.s{starting_time}.e{ending_time}.{version}.nc"
    )
    return filename


def define_l1_filename(ds, campaign_name, station_name: str) -> str:
    """Define L1 file name.

    Parameters
    ----------
    ds  : xarray.Dataset
        L1 xarray Dataset.
    campaign_name : str
        Name of the campaign.
    station_name : str
        Name of the station.

    Returns
    -------
    str
        L1 file name.
    """
    from disdrodb import ARCHIVE_VERSION

    # TODO: add sample_interval as argument
    sample_interval = int(ensure_sample_interval_in_seconds(ds["sample_interval"]).data.item())
    sample_interval_acronym = define_accumulation_acronym(sample_interval, rolling=False)
    starting_time, ending_time = get_file_start_end_time(ds)
    starting_time = starting_time.strftime("%Y%m%d%H%M%S")
    ending_time = ending_time.strftime("%Y%m%d%H%M%S")
    version = ARCHIVE_VERSION
    filename = (
        f"L1.{sample_interval_acronym}.{campaign_name}.{station_name}.s{starting_time}.e{ending_time}.{version}.nc"
    )
    return filename


def define_l2e_filename(ds, campaign_name: str, station_name: str, sample_interval: int, rolling: bool) -> str:
    """Define L2E file name.

    Parameters
    ----------
    ds  : xarray.Dataset
        L1 xarray Dataset
    campaign_name : str
        Name of the campaign.
    station_name : str
        Name of the station

    Returns
    -------
    str
        L0B file name.
    """
    from disdrodb import ARCHIVE_VERSION

    sample_interval_acronym = define_accumulation_acronym(seconds=sample_interval, rolling=rolling)
    starting_time, ending_time = get_file_start_end_time(ds)
    starting_time = starting_time.strftime("%Y%m%d%H%M%S")
    ending_time = ending_time.strftime("%Y%m%d%H%M%S")
    version = ARCHIVE_VERSION
    filename = (
        f"L2E.{sample_interval_acronym}.{campaign_name}.{station_name}.s{starting_time}.e{ending_time}.{version}.nc"
    )
    return filename


def define_l2m_filename(
    ds,
    campaign_name: str,
    station_name: str,
    sample_interval: int,
    rolling: bool,
    model_name: str,
) -> str:
    """Define L2M file name.

    Parameters
    ----------
    ds  : xarray.Dataset
        L1 xarray Dataset
    campaign_name : str
        Name of the campaign.
    station_name : str
        Name of the station

    Returns
    -------
    str
        L0B file name.
    """
    from disdrodb import ARCHIVE_VERSION

    sample_interval_acronym = define_accumulation_acronym(seconds=sample_interval, rolling=rolling)
    starting_time, ending_time = get_file_start_end_time(ds)
    starting_time = starting_time.strftime("%Y%m%d%H%M%S")
    ending_time = ending_time.strftime("%Y%m%d%H%M%S")
    version = ARCHIVE_VERSION
    filename = (
        f"L2M_{model_name}.{sample_interval_acronym}.{campaign_name}."
        + f"{station_name}.s{starting_time}.e{ending_time}.{version}.nc"
    )
    return filename
