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
"""Tools to create Raw, L0A and L0B DISDRODB directories."""

# L0A and L0B from raw NC: create_l0_directory_structure(raw_dir, processed_dir)
# L0B: create_product_directory(processed_dir)

import logging
import os
import shutil

from disdrodb.api.checks import (
    check_base_dir,
    check_campaign_name,
    check_data_availability,
    check_data_source,
    check_issue_file,
    check_metadata_file,
    check_processed_dir,
    check_product,
    check_raw_dir,
    has_available_data,
)
from disdrodb.api.info import (
    infer_campaign_name_from_path,
    infer_data_source_from_path,
    infer_disdrodb_tree_path_components,
)
from disdrodb.api.path import (
    define_campaign_dir,
    define_data_dir,
    define_issue_dir,
    define_logs_dir,
    define_metadata_dir,
    define_metadata_filepath,
    define_station_dir,
)
from disdrodb.configs import get_base_dir, get_metadata_dir
from disdrodb.utils.directories import (
    create_directory,
    create_required_directory,
    remove_if_exists,
)

logger = logging.getLogger(__name__)


#### DISDRODB Products directories
def _check_data_source_consistency(raw_dir: str, processed_dir: str) -> str:
    """Check that ``raw_dir`` and ``processed_dir`` have same ``data_source``.

    Parameters
    ----------
    raw_dir : str
        Path of the raw campaign directory
    processed_dir : str
        Path of the processed campaign directory

    Returns
    -------
    str
        ``data_source`` in capital letter.

    Raises
    ------
    ValueError
        Error if the ``data_source`` of the two directory paths does not match.
    """
    raw_data_source = infer_data_source_from_path(raw_dir)
    processed_data_source = infer_data_source_from_path(processed_dir)
    if raw_data_source != processed_data_source:
        msg = f"'raw_dir' and 'processed_dir' must ends with same <DATA_SOURCE>: {raw_data_source}"
        logger.error(msg)
        raise ValueError(msg)
    return raw_data_source.upper()


def _check_campaign_name_consistency(raw_dir: str, processed_dir: str) -> str:
    """Check that ``raw_dir`` and ``processed_dir`` have same campaign_name.

    Parameters
    ----------
    raw_dir : str
        Path of the raw campaign directory
    processed_dir : str
        Path of the processed campaign directory

    Returns
    -------
    str
        ``campaign_name`` in capital letter.

    Raises
    ------
    ValueError
        Error if the ``campaign_name`` of the two directory paths does not match.
    """
    raw_campaign_name = infer_campaign_name_from_path(raw_dir)
    processed_campaign_name = infer_campaign_name_from_path(processed_dir)
    if raw_campaign_name != processed_campaign_name:
        msg = f"'raw_dir' and 'processed_dir' must ends with same <CAMPAIGN_NAME>: {raw_campaign_name}"
        logger.error(msg)
        raise ValueError(msg)
    return raw_campaign_name.upper()


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
    raw_dir,
    processed_dir,
    metadata_dir,
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
    from disdrodb.configs import get_metadata_dir

    # Retrieve the DISDRODB Metadata Archive directory
    metadata_dir = get_metadata_dir(metadata_dir)

    # Retrieve base_dir data_source and campaign_name
    base_dir, _, data_source, campaign_name = infer_disdrodb_tree_path_components(raw_dir)

    # Check <DATA_SOURCE> and <CAMPAIGN_NAME> are upper case
    check_campaign_name(campaign_name)
    check_data_source(data_source)

    # Check input-output directories
    raw_dir = check_raw_dir(raw_dir=raw_dir)
    processed_dir = check_processed_dir(processed_dir=processed_dir)

    # Check consistent data_source and campaign name
    _ = _check_data_source_consistency(raw_dir=raw_dir, processed_dir=processed_dir)
    _ = _check_campaign_name_consistency(raw_dir=raw_dir, processed_dir=processed_dir)

    # Check there is data in the station directory
    check_data_availability(
        product="RAW",
        base_dir=base_dir,
        data_source=data_source,
        campaign_name=campaign_name,
        station_name=station_name,
    )

    # Check there is a valid metadata YAML file
    # check_metadata_file(
    #     metadata_dir=metadata_dir,
    #     data_source=data_source,
    #     campaign_name=campaign_name,
    #     station_name=station_name,
    # )

    # Check there is valid issue YAML file
    check_issue_file(
        metadata_dir=metadata_dir,
        data_source=data_source,
        campaign_name=campaign_name,
        station_name=station_name,
    )

    # Create required directories (if they don't exist)
    create_required_directory(processed_dir, dir_name="metadata")
    create_required_directory(processed_dir, dir_name="info")

    # Define and create product directory
    data_dir = define_data_dir(
        product=product,
        base_dir=base_dir,
        data_source=data_source,
        campaign_name=campaign_name,
        station_name=station_name,
    )

    # Create required directory (if it doesn't exist)
    create_directory(data_dir)

    # Check if product files are already available
    available_data = has_available_data(
        product=product,
        base_dir=base_dir,
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
    base_dir=None,
    metadata_dir=None,
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
    base_dir = get_base_dir(base_dir)
    metadata_dir = get_metadata_dir(metadata_dir)

    # Check inputs
    check_product(product)

    # Determine required product
    required_product = get_required_product(product)

    # Check station data is available in the previous product level
    check_data_availability(
        product=required_product,
        base_dir=base_dir,
        data_source=data_source,
        campaign_name=campaign_name,
        station_name=station_name,
        # Product options
        **product_kwargs,
    )

    # Check metadata file is available
    check_metadata_file(
        metadata_dir=metadata_dir,
        data_source=data_source,
        campaign_name=campaign_name,
        station_name=station_name,
    )

    # Define product output directory
    data_dir = define_data_dir(
        product=product,
        base_dir=base_dir,
        data_source=data_source,
        campaign_name=campaign_name,
        station_name=station_name,
        # Product options
        **product_kwargs,
    )

    # Create required directory (if it doesn't exist)
    create_directory(data_dir)

    # Check if product files are already available
    available_data = has_available_data(
        product=product,
        base_dir=base_dir,
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
    base_dir=None,
    # Option for L2E
    sample_interval=None,
    rolling=None,
    # Option for L2M
    model_name=None,
):
    """Initialize the logs directory structure for a DISDRODB product."""
    # Define logs directory
    logs_dir = define_logs_dir(
        product=product,
        base_dir=base_dir,
        data_source=data_source,
        campaign_name=campaign_name,
        station_name=station_name,
        # Option for L2E
        sample_interval=sample_interval,
        rolling=rolling,
        # Option for L2M
        model_name=model_name,
    )

    # Ensure empty log directory
    if os.path.isdir(logs_dir):
        shutil.rmtree(logs_dir)

    # Create logs directory
    os.makedirs(logs_dir, exist_ok=True)

    # Return logs directory
    return logs_dir


#### DISDRODB Station Initialization


def _create_station_directories(
    base_dir,
    metadata_dir,
    data_source,
    campaign_name,
    station_name,
):
    """Create the ``/metadata``, ``/issue`` and ``/data/<station_name>`` directories of a station."""
    # Create DISDRODB Data Archive Directory Structure
    _ = create_station_directory(
        base_dir=base_dir,
        product="RAW",
        data_source=data_source,
        campaign_name=campaign_name,
        station_name=station_name,
    )
    # Create DISDRODB Metadata Archive Directory Structure
    _ = create_metadata_directory(
        metadata_dir=metadata_dir,
        data_source=data_source,
        campaign_name=campaign_name,
    )
    _ = create_issue_directory(metadata_dir=metadata_dir, data_source=data_source, campaign_name=campaign_name)


def create_metadata_directory(metadata_dir, data_source, campaign_name):
    """Create metadata directory."""
    metadata_dir = define_metadata_dir(
        metadata_dir=metadata_dir,
        data_source=data_source,
        campaign_name=campaign_name,
        check_exists=False,
    )
    if not os.path.exists(metadata_dir):
        os.makedirs(metadata_dir, exist_ok=True)
    return str(metadata_dir)


def create_station_directory(base_dir, product, data_source, campaign_name, station_name):
    """Create station data directory."""
    station_dir = define_station_dir(
        base_dir=base_dir,
        product=product,
        data_source=data_source,
        campaign_name=campaign_name,
        station_name=station_name,
        check_exists=False,
    )
    if not os.path.exists(station_dir):
        os.makedirs(station_dir, exist_ok=True)
    return str(station_dir)


def create_data_directory(base_dir, product, data_source, campaign_name, station_name, **product_kwargs):
    """Create station product data directory."""
    data_dir = define_data_dir(
        base_dir=base_dir,
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


def create_issue_directory(metadata_dir, data_source, campaign_name):
    """Create issue directory."""
    issue_dir = define_issue_dir(
        metadata_dir=metadata_dir,
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
    base_dir=None,
    metadata_dir=None,
):
    """Create the DISDRODB Data and Metadata Archive structure for a single station."""
    from disdrodb.issue.writer import create_station_issue
    from disdrodb.metadata.writer import create_station_metadata

    base_dir = get_base_dir(base_dir)
    metadata_dir = get_metadata_dir(metadata_dir)

    # Check if already been defined
    # - Check presence of metadata file
    metadata_filepath = define_metadata_filepath(
        metadata_dir=metadata_dir,
        data_source=data_source,
        campaign_name=campaign_name,
        station_name=station_name,
    )
    if os.path.exists(metadata_filepath):
        raise ValueError(
            f"A metadata file already exists at {metadata_filepath}. "
            "The station is already part of the DISDRODB Archive or "
            "or you already initialized the directory structure for the station !",
        )

    # Create directory structure (/metadata, /issue and /data/<station_name>)
    _create_station_directories(
        base_dir=base_dir,
        metadata_dir=metadata_dir,
        data_source=data_source,
        campaign_name=campaign_name,
        station_name=station_name,
    )

    # Add default station metadata file
    create_station_metadata(
        metadata_dir=metadata_dir,
        data_source=data_source,
        campaign_name=campaign_name,
        station_name=station_name,
    )
    # Add default station issue file
    create_station_issue(
        metadata_dir=metadata_dir,
        data_source=data_source,
        campaign_name=campaign_name,
        station_name=station_name,
    )

    # Report location of the campaign directory
    campaign_dir = define_campaign_dir(
        base_dir=base_dir,
        data_source=data_source,
        campaign_name=campaign_name,
        product="RAW",
    )
    print(f"Initial station directory structure created at: {campaign_dir}")


#### DISDRODB upload/download testing
def create_test_archive(
    test_base_dir,
    data_source,
    campaign_name,
    station_name,
    base_dir=None,
    metadata_dir=None,
    force=False,
):
    """Create test DISDRODB Archive for a single existing station.

    This function is used to make a copy of metadata and issue files of a stations.
    This enable to then test data download and DISDRODB processing.
    """
    # Check metadata repository is available
    metadata_dir = get_metadata_dir(metadata_dir)

    # Check test_base_dir is not equal to true base_dir
    test_base_dir = check_base_dir(test_base_dir)
    if test_base_dir == get_base_dir(base_dir):
        raise ValueError("Provide a test_base_dir directory different from the true DISDRODB Data Archive directory !")

    # Create test DISDRODB base directory
    remove_if_exists(test_base_dir, force=force)
    os.makedirs(test_base_dir, exist_ok=True)

    tree = f"{data_source} {campaign_name} {station_name}"
    print(
        f"The test DISDRODB Data Archive for {tree} has been set up at {test_base_dir} !",
    )

    # TODO: REFACTOR_STRUCTURE LIKELY UNNECESSARY HERE BELOW !
    # Create directories (/metadata, /issue and /data/<station_name>)
    # _create_station_directories(
    #     base_dir=test_base_dir,
    #     metadata_dir=metadata_dir,
    #     data_source=data_source,
    #     campaign_name=campaign_name,
    #     station_name=station_name,
    # )
    # # Copy metadata and issue files in the test archive
    # src_metadata_fpath = define_metadata_filepath(
    #     data_source=data_source,
    #     campaign_name=campaign_name,
    #     station_name=station_name,
    #     metadata_dir=metadata_dir,
    # )
    # dst_metadata_fpath = define_metadata_filepath(
    #     data_source=data_source,
    #     campaign_name=campaign_name,
    #     station_name=station_name,
    #     metadata_dir=test_metadata_dir,
    # )
    # src_issue_fpath = define_issue_filepath(
    #     data_source=data_source,
    #     campaign_name=campaign_name,
    #     station_name=station_name,
    #     metadata_dir=metadata_dir,
    # )
    # dst_issue_fpath = define_issue_filepath(
    #     data_source=data_source,
    #     campaign_name=campaign_name,
    #     station_name=station_name,
    #     metadata_dir=test_metadata_dir,
    # )
    # copy_file(src_issue_fpath, dst_issue_fpath)
    # copy_file(src_metadata_fpath, dst_metadata_fpath)
