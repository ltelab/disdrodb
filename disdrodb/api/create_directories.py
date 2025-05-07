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
"""Tools to create RAW, L0A and L0B DISDRODB directories."""

# L0A and L0B from raw NC: create_l0_directory_structure(...)
# L0B: create_product_directory(...)

import logging
import os
import shutil

from disdrodb.api.checks import (
    check_campaign_name,
    check_data_archive_dir,
    check_data_availability,
    check_data_source,
    check_issue_file,
    check_metadata_file,
    check_product,
    has_available_data,
    select_required_product_kwargs,
)
from disdrodb.api.path import (
    define_data_dir,
    define_issue_dir,
    define_logs_dir,
    define_metadata_dir,
    define_metadata_filepath,
)
from disdrodb.configs import get_data_archive_dir, get_metadata_archive_dir
from disdrodb.utils.directories import (
    create_directory,
    remove_if_exists,
)

logger = logging.getLogger(__name__)

####--------------------------------------------------------------------------------.
#### DISDRODB Products Directories


def ensure_empty_data_dir(data_dir, force):
    """Remove the content of the data_dir directory."""
    # If force=True, remove all the directory content
    if force:
        shutil.rmtree(data_dir)
        # Recreate the directory
        create_directory(data_dir)
    else:
        msg = f"The product directory {data_dir} already contains files and force=False."
        logger.error(msg)
        raise ValueError(msg)


def create_l0_directory_structure(
    data_archive_dir,
    metadata_archive_dir,
    data_source,
    campaign_name,
    station_name,
    force,
    product,
):
    """Create directory structure for the first L0 DISDRODB product.

    If the input data are raw text files, use ``product = "L0A"``
    If the input data are raw netCDF files, use ``product = "L0B"``

    ``product = "L0A"`` will call ``run_l0a``.
    ``product = "L0B"`` will call ``run_l0b_nc``.
    """
    from disdrodb.configs import get_data_archive_dir, get_metadata_archive_dir

    # Retrieve the DISDRODB Metadata Archive directory
    data_archive_dir = get_data_archive_dir(data_archive_dir)
    metadata_archive_dir = get_metadata_archive_dir(metadata_archive_dir)

    # Check <DATA_SOURCE> and <CAMPAIGN_NAME> are upper case
    check_campaign_name(campaign_name)
    check_data_source(data_source)

    # Check raw station data are available
    check_data_availability(
        product="RAW",
        data_archive_dir=data_archive_dir,
        data_source=data_source,
        campaign_name=campaign_name,
        station_name=station_name,
    )

    # Check there is a valid metadata YAML file
    check_metadata_file(
        metadata_archive_dir=metadata_archive_dir,
        data_source=data_source,
        campaign_name=campaign_name,
        station_name=station_name,
    )

    # Check there is valid issue YAML file
    check_issue_file(
        metadata_archive_dir=metadata_archive_dir,
        data_source=data_source,
        campaign_name=campaign_name,
        station_name=station_name,
    )

    # Define product output data directory
    data_dir = define_data_dir(
        product=product,
        data_archive_dir=data_archive_dir,
        data_source=data_source,
        campaign_name=campaign_name,
        station_name=station_name,
    )

    # Create product output data directory (if it doesn't exist)
    create_directory(data_dir)

    # Check if product files are already available
    available_data = has_available_data(
        product=product,
        data_archive_dir=data_archive_dir,
        data_source=data_source,
        campaign_name=campaign_name,
        station_name=station_name,
    )

    # If product files are already available:
    # - If force=True, remove all data inside the product directory
    # - If force=False, raise an error
    if available_data:
        ensure_empty_data_dir(data_dir, force=force)

    return data_dir


def create_product_directory(
    data_source,
    campaign_name,
    station_name,
    product,
    force,
    data_archive_dir=None,
    metadata_archive_dir=None,
    # Product Options
    **product_kwargs,
):
    """Initialize the directory structure for a DISDRODB product.

    If product files already exists:
    - If ``force=True``, it remove all existing data inside the product directory.
    - If ``force=False``, it raise an error.
    """
    # NOTE: ``force=False`` behaviour could be changed to enable updating of missing files.
    # This would require also adding code to check whether a downstream file already exist.

    from disdrodb.api.search import get_required_product

    # Get DISDRODB base directory
    data_archive_dir = get_data_archive_dir(data_archive_dir)
    metadata_archive_dir = get_metadata_archive_dir(metadata_archive_dir)

    # Check inputs
    check_product(product)

    # Determine required product
    required_product = get_required_product(product)

    # Check station data is available in the previous product level
    required_product_kwargs = select_required_product_kwargs(required_product, product_kwargs)
    check_data_availability(
        product=required_product,
        data_archive_dir=data_archive_dir,
        data_source=data_source,
        campaign_name=campaign_name,
        station_name=station_name,
        # Product options
        **required_product_kwargs,
    )

    # Check metadata file is available
    check_metadata_file(
        metadata_archive_dir=metadata_archive_dir,
        data_source=data_source,
        campaign_name=campaign_name,
        station_name=station_name,
    )

    # Define product output data directory
    data_dir = define_data_dir(
        product=product,
        data_archive_dir=data_archive_dir,
        data_source=data_source,
        campaign_name=campaign_name,
        station_name=station_name,
        # Product options
        **product_kwargs,
    )

    # Create product output data directory (if it doesn't exist)
    create_directory(data_dir)

    # Check if product files are already available
    available_data = has_available_data(
        product=product,
        data_archive_dir=data_archive_dir,
        data_source=data_source,
        campaign_name=campaign_name,
        station_name=station_name,
        # Product options
        **product_kwargs,
    )

    # If product files are already available:
    # - If force=True, remove all data inside the product directory
    # - If force=False, raise an error
    if available_data:
        ensure_empty_data_dir(data_dir, force=force)

    # Return product directory
    return data_dir


def create_logs_directory(
    product,
    data_source,
    campaign_name,
    station_name,
    data_archive_dir=None,
    # Product options
    **product_kwargs,
):
    """Initialize the logs directory structure for a DISDRODB product."""
    # Define logs directory
    logs_dir = define_logs_dir(
        product=product,
        data_archive_dir=data_archive_dir,
        data_source=data_source,
        campaign_name=campaign_name,
        station_name=station_name,
        # Product options
        **product_kwargs,
    )

    # Ensure empty log directory
    if os.path.isdir(logs_dir):
        shutil.rmtree(logs_dir)

    # Create logs directory
    os.makedirs(logs_dir, exist_ok=True)

    # Return logs directory
    return logs_dir


####--------------------------------------------------------------------------------.
#### DISDRODB Station Initialization


def create_data_directory(data_archive_dir, product, data_source, campaign_name, station_name, **product_kwargs):
    """Create station product data directory."""
    data_dir = define_data_dir(
        data_archive_dir=data_archive_dir,
        product=product,
        data_source=data_source,
        campaign_name=campaign_name,
        station_name=station_name,
        check_exists=False,
        **product_kwargs,
    )
    if not os.path.exists(data_dir):
        os.makedirs(data_dir, exist_ok=True)
    return str(data_dir)


def create_metadata_directory(metadata_archive_dir, data_source, campaign_name):
    """Create metadata directory."""
    metadata_dir = define_metadata_dir(
        metadata_archive_dir=metadata_archive_dir,
        data_source=data_source,
        campaign_name=campaign_name,
        check_exists=False,
    )
    if not os.path.exists(metadata_dir):
        os.makedirs(metadata_dir, exist_ok=True)
    return str(metadata_dir)


def create_issue_directory(metadata_archive_dir, data_source, campaign_name):
    """Create issue directory."""
    issue_dir = define_issue_dir(
        metadata_archive_dir=metadata_archive_dir,
        data_source=data_source,
        campaign_name=campaign_name,
        check_exists=False,
    )
    if not os.path.exists(issue_dir):
        os.makedirs(issue_dir, exist_ok=True)
    return str(issue_dir)


def create_initial_station_structure(
    data_source,
    campaign_name,
    station_name,
    data_archive_dir=None,
    metadata_archive_dir=None,
):
    """Create the DISDRODB Data and Metadata Archive structure for a single station."""
    from disdrodb.issue.writer import create_station_issue
    from disdrodb.metadata.writer import create_station_metadata

    data_archive_dir = get_data_archive_dir(data_archive_dir)
    metadata_archive_dir = get_metadata_archive_dir(metadata_archive_dir)

    # Check if already been defined
    # - Check presence of metadata file
    metadata_filepath = define_metadata_filepath(
        metadata_archive_dir=metadata_archive_dir,
        data_source=data_source,
        campaign_name=campaign_name,
        station_name=station_name,
    )
    if os.path.exists(metadata_filepath):
        raise ValueError(
            f"The DISDRODB Metadata Archive has already a metadata file "
            f"for {data_source} {campaign_name} {station_name} at '{metadata_filepath}'. "
            "You might have already initialized the directory structure for such station !",
        )

    # -----------------------.
    # Create station directory in the DISDRODB Data Archive
    data_dir = create_data_directory(
        data_archive_dir=data_archive_dir,
        product="RAW",
        data_source=data_source,
        campaign_name=campaign_name,
        station_name=station_name,
    )

    # -----------------------.
    # Create issue and metadata files in the DISDRODB Metadata Archive
    # - Create /metadata and /issue directories in the DISDRODB Metadata Archive
    _ = create_metadata_directory(
        metadata_archive_dir=metadata_archive_dir,
        data_source=data_source,
        campaign_name=campaign_name,
    )
    _ = create_issue_directory(
        metadata_archive_dir=metadata_archive_dir,
        data_source=data_source,
        campaign_name=campaign_name,
    )

    # - Add an empty/default metadata file (to be filled by data contributor)
    create_station_metadata(
        metadata_archive_dir=metadata_archive_dir,
        data_source=data_source,
        campaign_name=campaign_name,
        station_name=station_name,
    )
    # - Add an empty issue file (to be filled by data contributor if necessary)
    create_station_issue(
        metadata_archive_dir=metadata_archive_dir,
        data_source=data_source,
        campaign_name=campaign_name,
        station_name=station_name,
    )

    # --------------------------------------------------------------------------.
    # Report next steps to contribute data to DISDRODB
    print("The DISDRODB Metadata and Data Archive directories have been initialized.")
    print("To contribute your data to DISDRODB:")
    print(f"1. Place you raw data within the '{data_dir}' directory.")
    print(f"2. Fill the metadata fields of the '{metadata_filepath}' file.")


####--------------------------------------------------------------------------------.
#### DISDRODB upload/download testing
def create_test_archive(
    test_data_archive_dir,
    data_source,
    campaign_name,
    station_name,
    data_archive_dir=None,
    metadata_archive_dir=None,
    force=False,
):
    """Create test DISDRODB Archive for a single existing station.

    This function is used to make a copy of metadata and issue files of a stations.
    This enable to then test data download and DISDRODB processing.
    """
    # Check metadata repository is available
    metadata_archive_dir = get_metadata_archive_dir(metadata_archive_dir)

    # Check test_data_archive_dir is not equal to true data_archive_dir
    test_data_archive_dir = check_data_archive_dir(test_data_archive_dir)
    if test_data_archive_dir == get_data_archive_dir(data_archive_dir):
        raise ValueError(
            "Provide a test_data_archive_dir directory different from the true DISDRODB Data Archive directory !",
        )

    # Create test DISDRODB base directory
    remove_if_exists(test_data_archive_dir, force=force)
    os.makedirs(test_data_archive_dir, exist_ok=True)

    tree = f"{data_source} {campaign_name} {station_name}"
    print(
        f"The test DISDRODB Data Archive for {tree} has been set up at {test_data_archive_dir} !",
    )
    return test_data_archive_dir
