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
"""DISDRODB Checks Functions."""
import datetime
import difflib
import logging
import os
import re
import sys
import warnings

import numpy as np

from disdrodb.api.path import (
    define_data_dir,
    define_issue_dir,
    define_issue_filepath,
    define_metadata_filepath,
)
from disdrodb.constants import PRODUCTS, PRODUCTS_ARGUMENTS
from disdrodb.utils.directories import (
    ensure_string_path,
    list_directories,
    list_files,
)

logger = logging.getLogger(__name__)


def check_path(path: str) -> None:
    """Check if a path exists.

    Parameters
    ----------
    path : str
        Path to check.

    Raises
    ------
    FileNotFoundError
        If the path does not exist.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"The path {path} does not exist. Please check the path.")


def check_url(url: str) -> bool:
    """Check url.

    Parameters
    ----------
    url : str
        URL to check.

    Returns
    -------
    bool
        ``True`` if url well formatted, ``False`` if not well formatted.
    """
    regex = r"^(https?:\/\/)?(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*)$"  # noqa: E501
    return re.match(regex, url)


def check_path_is_a_directory(dir_path, path_name=""):
    """Check that the path exists and is directory."""
    dir_path = ensure_string_path(dir_path, msg="Provide {path_name} as a string", accepth_pathlib=True)
    if not os.path.exists(dir_path):
        raise ValueError(f"{path_name} {dir_path} directory does not exist.")
    if not os.path.isdir(dir_path):
        raise ValueError(f"{path_name} {dir_path} is not a directory.")


def check_directories_inside(dir_path):
    """Check there are directories inside the specified ``dir_path``."""
    dir_paths = list_directories(dir_path, recursive=False)
    if len(dir_paths) == 0:
        raise ValueError(f"There are not directories within {dir_path}")


def check_data_archive_dir(data_archive_dir: str):
    """Raise an error if the path does not end with ``DISDRODB``."""
    data_archive_dir = str(data_archive_dir)  # convert Pathlib to string
    data_archive_dir = os.path.normpath(data_archive_dir)
    if not data_archive_dir.endswith("DISDRODB"):
        raise ValueError(f"The path {data_archive_dir} does not end with DISDRODB. Please check the path.")
    return data_archive_dir


def check_metadata_archive_dir(metadata_archive_dir: str):
    """Raise an error if the path does not end with ``DISDRODB``."""
    metadata_archive_dir = str(metadata_archive_dir)  # convert Pathlib to string
    metadata_archive_dir = os.path.normpath(metadata_archive_dir)
    if not metadata_archive_dir.endswith("DISDRODB"):
        raise ValueError(f"The path {metadata_archive_dir} does not end with DISDRODB. Please check the path.")
    return metadata_archive_dir


def check_scattering_table_dir(scattering_table_dir: str):
    """Raise an error if the directory does not exist."""
    scattering_table_dir = str(scattering_table_dir)  # convert Pathlib to string
    scattering_table_dir = os.path.normpath(scattering_table_dir)
    if not os.path.exists(scattering_table_dir):
        raise ValueError(f"The DISDRODB T-Matrix scattering tables directory {scattering_table_dir} does not exists.")
    return scattering_table_dir


def check_measurement_interval(measurement_interval):
    """Check measurement interval validity."""
    if isinstance(measurement_interval, str) and measurement_interval == "":
        raise ValueError("measurement_interval' must be specified as an integer value.")
    if isinstance(measurement_interval, type(None)):
        raise ValueError("measurement_interval' can not be None.")
    if isinstance(measurement_interval, str) and not measurement_interval.isdigit():
        raise ValueError("measurement_interval' is not a positive digit.")
    return int(measurement_interval)


def check_measurement_intervals(measurement_intervals):
    """Check measurement interval.

    Can be a list. It must be a positive natural number
    """
    if isinstance(measurement_intervals, (int, float, str)):
        measurement_intervals = [measurement_intervals]
    measurement_intervals = [check_measurement_interval(v) for v in measurement_intervals]
    return measurement_intervals


def check_sample_interval(sample_interval):
    """Check sample_interval argument validity."""
    if not isinstance(sample_interval, int) or isinstance(sample_interval, bool):
        raise TypeError("'sample_interval' must be an integer.")


def check_rolling(rolling):
    """Check rolling argument validity."""
    if not isinstance(rolling, bool):
        raise TypeError("'rolling' must be a boolean.")


def check_folder_partitioning(folder_partitioning):
    """
    Check if the given folder partitioning scheme is valid.

    Parameters
    ----------
    folder_partitioning : str or None
        Defines the subdirectory structure based on the dataset's start time.
        Allowed values are:
          - "" or None: No additional subdirectories, files are saved directly in dir.
          - "year": Files are stored under a subdirectory for the year (<dir>/2025).
          - "year/month": Files are stored under subdirectories by year and month (<dir>/2025/04).
          - "year/month/day": Files are stored under subdirectories by year, month and day (<dir>/2025/04/01).
          - "year/month_name": Files are stored under subdirectories by year and month name (<dir>/2025/April).
          - "year/quarter": Files are stored under subdirectories by year and quarter (<dir>/2025/Q2).

    Returns
    -------
    folder_partitioning
        The verified folder partitioning scheme.
    """
    valid_options = ["", "year", "year/month", "year/month/day", "year/month_name", "year/quarter"]
    if folder_partitioning is None:
        folder_partitioning = ""
    if folder_partitioning not in valid_options:
        raise ValueError(
            f"Invalid folder_partitioning scheme '{folder_partitioning}'. Valid options are: {valid_options}.",
        )
    return folder_partitioning


def check_sensor_name(sensor_name: str) -> None:
    """Check sensor name.

    Parameters
    ----------
    sensor_name : str
        Name of the sensor.
    product : str
        DISDRODB product.

    Raises
    ------
    TypeError
        Error if ``sensor_name`` is not a string.
    ValueError
        Error if the input sensor name has not been found in the list of available sensors.
    """
    from disdrodb.api.configs import available_sensor_names

    sensor_names = available_sensor_names()
    if not isinstance(sensor_name, str):
        raise TypeError("'sensor_name' must be a string.")
    if sensor_name not in sensor_names:
        msg = f"'{sensor_name}' is not a valid sensor_name. Valid values are {sensor_names}."
        raise ValueError(msg)


def check_campaign_name(campaign_name):
    """Check the campaign name is upper case !."""
    upper_campaign_name = campaign_name.upper()
    if campaign_name != upper_campaign_name:
        msg = f"The campaign directory name {campaign_name} must be defined uppercase: {upper_campaign_name}"
        logger.error(msg)
        raise ValueError(msg)


def check_data_source(data_source):
    """Check the data_source name is upper case !."""
    upper_data_source = data_source.upper()
    if data_source != upper_data_source:
        msg = f"The data source directory name {data_source} must be defined uppercase: {upper_data_source}"
        logger.error(msg)
        raise ValueError(msg)


def check_product(product):
    """Check DISDRODB product."""
    if not isinstance(product, str):
        raise TypeError("`product` must be a string.")
    valid_products = PRODUCTS
    if product.upper() not in valid_products:
        msg = f"Valid `products` are {valid_products}."
        logger.error(msg)
        raise ValueError(msg)
    return product


def check_product_kwargs(product, product_kwargs):
    """Validate that product_kwargs for a given product contains exactly the required parameters.

    Parameters
    ----------
    product : str
        The product name (e.g., "L2E", "L2M").
    product_kwargs : dict
        Keyword arguments provided for this product.

    Returns
    -------
    dict
        The validated product_kwargs.

    Raises
    ------
    ValueError
        If required arguments are missing or if there are unexpected extra arguments.
    """
    required = set(PRODUCTS_ARGUMENTS.get(product, []))
    provided = set(product_kwargs.keys())
    missing = required - provided
    extra = provided - required
    if missing and extra:
        raise ValueError(
            f"For product '{product}', missing arguments: {sorted(missing)}, " f"unexpected arguments: {sorted(extra)}",
        )
    if missing:
        raise ValueError(f"For product '{product}', missing arguments: {sorted(missing)}")
    if extra:
        raise ValueError(f"For product '{product}', unexpected arguments: {sorted(extra)}")
    return product_kwargs


def select_required_product_kwargs(product, product_kwargs):
    """Select the required product arguments."""
    required = set(PRODUCTS_ARGUMENTS.get(product, []))
    provided = set(product_kwargs.keys())
    missing = required - provided
    # If missing, raise error
    if missing:
        raise ValueError(f"For product '{product}', missing arguments: {sorted(missing)}")
    # Else return just required arguments
    # --> e.g. for L0 no product arguments
    return {k: product_kwargs[k] for k in required}


def _check_fields(fields):
    if fields is None:  # isinstance(fields, type(None)):
        return fields
    # Ensure is a list
    if isinstance(fields, str):
        fields = [fields]
    # Remove duplicates
    fields = np.unique(np.array(fields))
    return fields


def check_data_sources(data_sources):
    """Check DISDRODB data sources."""
    return _check_fields(data_sources)


def check_campaign_names(campaign_names):
    """Check DISDRODB campaign names."""
    return _check_fields(campaign_names)


def check_station_names(station_names):
    """Check DISDRODB station names."""
    return _check_fields(station_names)


def check_invalid_fields_policy(invalid_fields):
    """Check invalid fields policy."""
    if invalid_fields not in ["raise", "warn", "ignore"]:
        raise ValueError(
            f"Invalid value for invalid_fields: {invalid_fields}. " "Valid values are 'raise', 'warn', or 'ignore'.",
        )
    return invalid_fields


def check_valid_fields(fields, available_fields, field_name, invalid_fields_policy="raise"):
    """Check if fields are valid."""
    if fields is None:
        return fields
    if isinstance(fields, str):
        fields = [fields]
    fields = np.unique(np.array(fields))
    invalid_fields_policy = check_invalid_fields_policy(invalid_fields_policy)

    # Check for invalid fields
    fields = np.array(fields)
    is_valid = np.isin(fields, available_fields)
    invalid_fields_values = fields[~is_valid].tolist()
    fields = fields[is_valid].tolist()

    # If invalid fields, suggest corrections using difflib
    if invalid_fields_values:

        # Format invalid fields nicely (avoid single-element lists)
        if len(invalid_fields_values) == 1:
            invalid_fields_str = f"'{invalid_fields_values[0]}'"
        else:
            invalid_fields_str = f"{invalid_fields_values}"

        # Prepare suggestion string
        suggestions = []
        for invalid in invalid_fields_values:
            matches = difflib.get_close_matches(invalid, available_fields, n=1, cutoff=0.4)
            if matches:
                suggestions.append(f"Did you mean '{matches[0]}' instead of '{invalid}'?")
        suggestion_msg = " " + " ".join(suggestions) if suggestions else ""

    # Error handling for invalid fields were found
    if invalid_fields_policy == "warn" and invalid_fields_values:
        msg = f"Ignoring invalid {field_name}: {invalid_fields_str}.{suggestion_msg}"
        warnings.warn(msg, UserWarning, stacklevel=2)
    elif invalid_fields_policy == "raise" and invalid_fields_values:
        msg = f"These {field_name} do not exist: {invalid_fields_str}.{suggestion_msg}"
        raise ValueError(msg)
    else:  # "ignore" silently drop invalid entries
        pass
    # If no valid fields left, raise error
    if len(fields) == 0:
        raise ValueError(f"All specified {field_name} do not exist !.")
    return fields


def check_station_inputs(
    data_source,
    campaign_name,
    station_name,
    metadata_archive_dir=None,
):
    """Check validity of stations inputs."""
    import disdrodb

    # Check data source
    valid_data_sources = disdrodb.available_data_sources(metadata_archive_dir=metadata_archive_dir)
    if data_source not in valid_data_sources:
        matches = difflib.get_close_matches(data_source, valid_data_sources, n=1, cutoff=0.4)
        suggestion = f"Did you mean '{matches[0]}'?" if matches else ""
        raise ValueError(f"DISDRODB does not include a data source named {data_source}. {suggestion}")

    # Check campaign name
    valid_campaigns = disdrodb.available_campaigns(data_sources=data_source, metadata_archive_dir=metadata_archive_dir)
    if campaign_name not in valid_campaigns:
        matches = difflib.get_close_matches(campaign_name, valid_campaigns, n=1, cutoff=0.4)
        suggestion = f"Did you mean campaign '{matches[0]}' ?" if matches else ""
        raise ValueError(
            f"The {data_source} data source does not include a campaign named {campaign_name}. {suggestion}",
        )

    # Check station name
    valid_stations = disdrodb.available_stations(
        data_sources=data_source,
        campaign_names=campaign_name,
        metadata_archive_dir=metadata_archive_dir,
        return_tuple=False,
    )
    if station_name not in valid_stations:
        matches = difflib.get_close_matches(station_name, valid_stations, n=1, cutoff=0.4)
        suggestion = f"Did you mean station '{matches[0]}'?" if matches else ""
        raise ValueError(
            f"The {data_source} {campaign_name} campaign does not have a station named {station_name}. {suggestion}",
        )


def has_available_data(
    data_source,
    campaign_name,
    station_name,
    product,
    data_archive_dir=None,
    # Product Options
    **product_kwargs,
):
    """Return ``True`` if data are available for the given product and station."""
    # Define product directory
    data_dir = define_data_dir(
        product=product,
        data_archive_dir=data_archive_dir,
        data_source=data_source,
        campaign_name=campaign_name,
        station_name=station_name,
        # Directory options
        check_exists=False,
        # Product Options
        **product_kwargs,
    )
    # If the product directory does not exists, return False
    if not os.path.isdir(data_dir):
        return False

    # If no files, return False
    filepaths = list_files(data_dir, recursive=True)
    nfiles = len(filepaths)
    return nfiles >= 1


def check_data_availability(
    product,
    data_source,
    campaign_name,
    station_name,
    data_archive_dir=None,
    # Product Options
    **product_kwargs,
):
    """Check the station product data directory has files inside. If not, raise an error."""
    if not has_available_data(
        product=product,
        data_archive_dir=data_archive_dir,
        data_source=data_source,
        campaign_name=campaign_name,
        station_name=station_name,
        # Product Options
        **product_kwargs,
    ):
        msg = f"The {product} station data directory of {data_source} {campaign_name} {station_name} is empty !"
        raise ValueError(msg)


def check_metadata_file(metadata_archive_dir, data_source, campaign_name, station_name, check_validity=True):
    """Check existence of a valid metadata YAML file. If does not exists, raise an error."""
    from disdrodb.metadata.checks import check_station_metadata

    metadata_filepath = define_metadata_filepath(
        metadata_archive_dir=metadata_archive_dir,
        data_source=data_source,
        campaign_name=campaign_name,
        station_name=station_name,
    )
    # Check existence
    if not os.path.exists(metadata_filepath):
        msg = (
            f"The metadata YAML file of {data_source} {campaign_name} {station_name} does not exist at"
            f" {metadata_filepath}."
        )
        raise ValueError(msg)

    # Check validity
    if check_validity:
        check_station_metadata(
            metadata_archive_dir=metadata_archive_dir,
            data_source=data_source,
            campaign_name=campaign_name,
            station_name=station_name,
        )
    return metadata_filepath


def check_issue_dir(data_source, campaign_name, metadata_archive_dir=None):
    """Check existence of the issue directory. If does not exists, raise an error."""
    issue_dir = define_issue_dir(
        metadata_archive_dir=metadata_archive_dir,
        data_source=data_source,
        campaign_name=campaign_name,
        check_exists=False,
    )
    if not os.path.exists(issue_dir) or not os.path.isdir(issue_dir):
        msg = f"The issue directory does not exist at {issue_dir}."
        logger.error(msg)
    return issue_dir


def check_issue_file(data_source, campaign_name, station_name, metadata_archive_dir=None):
    """Check existence of a valid issue YAML file. If does not exists, raise an error."""
    from disdrodb.issue.checks import check_issue_compliance
    from disdrodb.issue.writer import create_station_issue

    _ = check_issue_dir(
        metadata_archive_dir=metadata_archive_dir,
        data_source=data_source,
        campaign_name=campaign_name,
    )
    issue_filepath = define_issue_filepath(
        metadata_archive_dir=metadata_archive_dir,
        data_source=data_source,
        campaign_name=campaign_name,
        station_name=station_name,
        check_exists=False,
    )
    # Check existence. If not, create one !
    if not os.path.exists(issue_filepath):
        create_station_issue(
            metadata_archive_dir=metadata_archive_dir,
            data_source=data_source,
            campaign_name=campaign_name,
            station_name=station_name,
        )

    # Check validity
    check_issue_compliance(
        metadata_archive_dir=metadata_archive_dir,
        data_source=data_source,
        campaign_name=campaign_name,
        station_name=station_name,
    )
    return issue_filepath


def check_filepaths(filepaths):
    """Ensure filepaths is a list of string."""
    if isinstance(filepaths, str):
        filepaths = [filepaths]
    if not isinstance(filepaths, list):
        raise TypeError("Expecting a list of filepaths.")
    return filepaths


def get_current_utc_time():
    """Get current UTC time."""
    if sys.version_info >= (3, 11):
        return datetime.datetime.now(datetime.UTC).replace(tzinfo=None)
    return datetime.datetime.utcnow()


def check_start_end_time(start_time, end_time):
    """Check start_time and end_time value validity."""
    start_time = check_time(start_time)
    end_time = check_time(end_time)

    # Check start_time and end_time are chronological
    if start_time > end_time:
        raise ValueError("Provide 'start_time' occurring before of 'end_time'.")
    # Check start_time and end_time are in the past
    if start_time > get_current_utc_time():
        raise ValueError("Provide 'start_time' occurring in the past.")
    if end_time > get_current_utc_time():
        raise ValueError("Provide 'end_time' occurring in the past.")
    return (start_time, end_time)


def check_time(time):
    """Check time validity.

    It returns a :py:class:`datetime.datetime` object to seconds precision.

    Parameters
    ----------
    time : datetime.datetime, datetime.date, numpy.datetime64 or str
        Time object.
        Accepted types: ``datetime.datetime``, ``datetime.date``, ``numpy.datetime64`` or ``str``.
        If string type, it expects the isoformat ``YYYY-MM-DD hh:mm:ss``.

    Returns
    -------
    time: datetime.datetime

    """
    if not isinstance(time, (datetime.datetime, datetime.date, np.datetime64, np.ndarray, str)):
        raise TypeError(
            "Specify time with datetime.datetime objects or a " "string of format 'YYYY-MM-DD hh:mm:ss'.",
        )

    # If numpy array with datetime64 (and size=1)
    if isinstance(time, np.ndarray):
        if np.issubdtype(time.dtype, np.datetime64):
            if time.size == 1:
                time = time[0].astype("datetime64[s]").tolist()
            else:
                raise ValueError("Expecting a single timestep!")
        else:
            raise ValueError("The numpy array does not have a numpy.datetime64 dtype!")

    # If np.datetime64, convert to datetime.datetime
    if isinstance(time, np.datetime64):
        time = time.astype("datetime64[s]").tolist()
    # If datetime.date, convert to datetime.datetime
    if not isinstance(time, (datetime.datetime, str)):
        time = datetime.datetime(time.year, time.month, time.day, 0, 0, 0)
    if isinstance(time, str):
        try:
            time = datetime.datetime.fromisoformat(time)
        except ValueError:
            raise ValueError("The time string must have format 'YYYY-MM-DD hh:mm:ss'")
    # If datetime object carries timezone that is not UTC, raise error
    if time.tzinfo is not None:
        if str(time.tzinfo) != "UTC":
            raise ValueError("The datetime object must be in UTC timezone if timezone is given.")
        # If UTC, strip timezone information
        time = time.replace(tzinfo=None)
    return time
