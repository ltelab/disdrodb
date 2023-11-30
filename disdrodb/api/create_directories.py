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
# L0B: create_directory_structure(processed_dir)

import logging
import os
import shutil

from disdrodb.api.checks import (
    check_metadata_file,
    check_processed_dir,
    check_product,
    check_raw_dir,
    check_station_has_data,
    has_available_station_files,
)
from disdrodb.api.info import (
    infer_campaign_name_from_path,
    infer_data_source_from_path,
    infer_disdrodb_tree_path_components,
)
from disdrodb.api.path import (
    define_campaign_dir,
    define_issue_dir,
    define_issue_filepath,
    define_metadata_dir,
    define_metadata_filepath,
    define_station_dir,
)
from disdrodb.configs import get_base_dir
from disdrodb.utils.directories import (
    check_directory_exists,
    copy_file,
    create_required_directory,
    remove_if_exists,
)

logger = logging.getLogger(__name__)


#### DISDRODB Products directories
def _check_data_source_consistency(raw_dir: str, processed_dir: str) -> str:
    """Check that 'raw_dir' and 'processed_dir' have same data_source.

    Parameters
    ----------
    raw_dir : str
        Path of the raw campaign directory
    processed_dir : str
        Path of the processed campaign directory

    Returns
    -------
    str
        data_source in capital letter.

    Raises
    ------
    ValueError
        Error if the data_source of the two directory paths does not match.
    """
    raw_data_source = infer_data_source_from_path(raw_dir)
    processed_data_source = infer_data_source_from_path(processed_dir)
    if raw_data_source != processed_data_source:
        msg = f"'raw_dir' and 'processed_dir' must ends with same <DATA_SOURCE>: {raw_data_source}"
        logger.error(msg)
        raise ValueError(msg)
    return raw_data_source.upper()


def _check_campaign_name_consistency(raw_dir: str, processed_dir: str) -> str:
    """Check that 'raw_dir' and 'processed_dir' have same campaign_name.

    Parameters
    ----------
    raw_dir : str
        Path of the raw campaign directory
    processed_dir : str
        Path of the processed campaign directory

    Returns
    -------
    str
        Campaign name in capital letter.

    Raises
    ------
    ValueError
        Error if the campaign_name of the two directory paths does not match.
    """
    raw_campaign_name = infer_campaign_name_from_path(raw_dir)
    processed_campaign_name = infer_campaign_name_from_path(processed_dir)
    if raw_campaign_name != processed_campaign_name:
        msg = f"'raw_dir' and 'processed_dir' must ends with same <CAMPAIGN_NAME>: {raw_campaign_name}"
        logger.error(msg)
        raise ValueError(msg)
    return raw_campaign_name.upper()


def _copy_station_metadata(
    data_source: str, campaign_name: str, station_name: str, base_dir: str = None, check_validity: bool = False
) -> None:
    """Copy the station YAML file from the raw_dir/metadata into processed_dir/metadata

    Parameters
    ----------
    raw_dir : str
        Path of the raw campaign directory
    processed_dir : str
        Path of the processed campaign directory

    Raises
    ------
    ValueError
        Error if the copy fails.
    """
    # Check the raw metadata YAML file exists
    raw_metadata_filepath = check_metadata_file(
        product="RAW",
        base_dir=base_dir,
        data_source=data_source,
        campaign_name=campaign_name,
        station_name=station_name,
        check_validity=check_validity,
    )
    # Define the destination filepath
    processed_metadata_filepath = define_metadata_filepath(
        product="L0A",
        base_dir=base_dir,
        data_source=data_source,
        campaign_name=campaign_name,
        station_name=station_name,
    )
    # Copy the metadata file
    copy_file(
        src_filepath=raw_metadata_filepath,
        dst_filepath=processed_metadata_filepath,
    )
    return None


def _check_pre_existing_station_data(
    data_source: str, campaign_name: str, station_name: str, product: str, base_dir=None, force=False
):
    """Check for pre-existing station data.

    - If force=True, remove all data inside the station directory.
    - If force=False, raise error.

    NOTE: force=False behaviour could be changed to enable updating of missing files.
         This would require also adding code to check whether a downstream file already exist.
    """
    # Check if there are available data
    available_data = has_available_station_files(
        product=product,
        base_dir=base_dir,
        data_source=data_source,
        campaign_name=campaign_name,
        station_name=station_name,
    )
    # Define the station directory path
    station_dir = define_station_dir(
        product=product,
        base_dir=base_dir,
        data_source=data_source,
        campaign_name=campaign_name,
        station_name=station_name,
    )
    # If the station data are already present:
    # - If force=True, remove all data inside the station directory
    # - If force=False, raise error
    if available_data:
        # Check is a directory
        check_directory_exists(station_dir)
        # If force=True, remove all the content
        if force:
            # Remove all station directory content
            shutil.rmtree(station_dir)
        else:
            msg = f"The station directory {station_dir} already exists and force=False."
            logger.error(msg)
            raise ValueError(msg)


def create_l0_directory_structure(
    raw_dir,
    processed_dir,
    station_name,
    force,
    product,
    verbose=False,
):
    """Create directory structure for the first L0 DISDRODB product.

    If the input data are raw text files --> product = "L0A"    (run_l0a)
    If the input data are raw netCDF files --> product = "L0B"  (run_l0b_nc)
    """
    # Check inputs
    raw_dir = check_raw_dir(raw_dir=raw_dir, station_name=station_name)
    processed_dir = check_processed_dir(processed_dir=processed_dir)

    # Check consistent data_source and campaign name
    _ = _check_data_source_consistency(raw_dir=raw_dir, processed_dir=processed_dir)
    _ = _check_campaign_name_consistency(raw_dir=raw_dir, processed_dir=processed_dir)

    # Retrieve components
    base_dir, product_type, data_source, campaign_name = infer_disdrodb_tree_path_components(processed_dir)

    # Check station data are available
    check_station_has_data(
        product="RAW",
        base_dir=base_dir,
        data_source=data_source,
        campaign_name=campaign_name,
        station_name=station_name,
    )

    # Create required directories (if they don't exist)
    create_required_directory(processed_dir, dir_name="metadata")
    create_required_directory(processed_dir, dir_name="info")
    create_required_directory(processed_dir, dir_name=product)

    # Copy the station metadata
    _copy_station_metadata(
        base_dir=base_dir,
        data_source=data_source,
        campaign_name=campaign_name,
        station_name=station_name,
    )
    # Remove <product>/<station> directory if force=True
    _check_pre_existing_station_data(
        product=product,
        base_dir=base_dir,
        data_source=data_source,
        campaign_name=campaign_name,
        station_name=station_name,
        force=force,
    )
    # Create the <product>/<station> directory
    create_required_directory(os.path.join(processed_dir, product), dir_name=station_name)


def create_directory_structure(processed_dir, product, station_name, force):
    """Create directory structure for L0B and higher DISDRODB products."""
    # Check inputs
    check_product(product)
    processed_dir = check_processed_dir(processed_dir=processed_dir)

    base_dir, product_type, data_source, campaign_name = infer_disdrodb_tree_path_components(processed_dir)

    # Determine required product
    if product == "L0B":
        required_product = "L0A"
    else:
        raise NotImplementedError("product {product} not yet implemented.")

    # Check station is available in the previous product level
    check_station_has_data(
        product=required_product,
        base_dir=base_dir,
        data_source=data_source,
        campaign_name=campaign_name,
        station_name=station_name,
    )

    # Check metadata file is available
    check_metadata_file(
        product=required_product,
        base_dir=base_dir,
        data_source=data_source,
        campaign_name=campaign_name,
        station_name=station_name,
    )

    # Create required directory (if it doesn't exist)
    create_required_directory(processed_dir, dir_name=product)

    # Remove <product>/<station_name> directory if force=True
    _check_pre_existing_station_data(
        product=product,
        base_dir=base_dir,
        data_source=data_source,
        campaign_name=campaign_name,
        station_name=station_name,
        force=force,
    )


#### DISDRODB Station Initialization


def _create_station_directories(
    data_source,
    campaign_name,
    station_name,
    product="RAW",
    base_dir=None,
):
    """Create the /metadata, /issue and /data/<station_name> directories of a station."""
    # Create directory structure
    _ = create_station_directory(
        base_dir=base_dir,
        product=product,
        data_source=data_source,
        campaign_name=campaign_name,
        station_name=station_name,
    )
    _ = create_metadata_directory(
        base_dir=base_dir, product=product, data_source=data_source, campaign_name=campaign_name
    )

    if product.upper() == "RAW":
        _ = create_issue_directory(base_dir=base_dir, data_source=data_source, campaign_name=campaign_name)


def create_metadata_directory(base_dir, product, data_source, campaign_name):
    """Create metadata directory."""
    metadata_dir = define_metadata_dir(
        base_dir=base_dir,
        product=product,
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


def create_issue_directory(base_dir, data_source, campaign_name):
    issue_dir = define_issue_dir(
        base_dir=base_dir,
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
):
    """Create the DISDRODB Data Archive structure for a single station."""
    from disdrodb.issue.writer import create_station_issue
    from disdrodb.metadata.writer import create_station_metadata

    # Check if already been defined
    # - Check presence of metadata file
    metadata_filepath = define_metadata_filepath(
        base_dir=base_dir,
        data_source=data_source,
        campaign_name=campaign_name,
        station_name=station_name,
        product="RAW",
        check_exists=False,
    )
    if os.path.exists(metadata_filepath):
        raise ValueError(
            f"A metadata file already exists at {metadata_filepath}. "
            "The station is already part of the DISDRODB Archive or "
            "or you already initialized the directory structure for the station !"
        )

    # Create directory structure (/metadata, /issue and /data/<station_name>)
    _create_station_directories(
        base_dir=base_dir,
        data_source=data_source,
        campaign_name=campaign_name,
        station_name=station_name,
        product="RAW",
    )

    # Add default station metadata file
    create_station_metadata(
        base_dir=base_dir,
        data_source=data_source,
        campaign_name=campaign_name,
        station_name=station_name,
    )
    # Add default station issue file
    create_station_issue(
        base_dir=base_dir,
        data_source=data_source,
        campaign_name=campaign_name,
        station_name=station_name,
    )

    # Report location of the campaign directory
    campaign_dir = define_campaign_dir(
        base_dir=base_dir, data_source=data_source, campaign_name=campaign_name, product="RAW"
    )
    print(f"Initial station directory structure created at: {campaign_dir}")


#### DISDRODB upload/download testing
def create_test_archive(test_base_dir, data_source, campaign_name, station_name, base_dir=None, force=False):
    """Create test DISDRODB Archive for a single existing station.

    This function is used to make a copy of metadata and issue files of a stations.
    This enable to then test data download and DISDRODB processing.
    """
    # Check test_base_dir is not equal to true base_dir
    if test_base_dir == get_base_dir(base_dir):
        raise ValueError("Provide a test_base_dir directory different from the true DISDRODB base directory !")

    # Create test DISDRODB base directory
    remove_if_exists(test_base_dir, force=force)
    os.makedirs(test_base_dir, exist_ok=True)

    # Create directories (/metadata, /issue and /data/<station_name>)
    _create_station_directories(
        base_dir=test_base_dir,
        product="RAW",
        data_source=data_source,
        campaign_name=campaign_name,
        station_name=station_name,
    )
    # Copy metadata and issue files in the test archive
    src_metadata_fpath = define_metadata_filepath(
        product="RAW",
        data_source=data_source,
        campaign_name=campaign_name,
        station_name=station_name,
        base_dir=base_dir,
    )
    dst_metadata_fpath = define_metadata_filepath(
        product="RAW",
        data_source=data_source,
        campaign_name=campaign_name,
        station_name=station_name,
        base_dir=test_base_dir,
    )
    src_issue_fpath = define_issue_filepath(
        data_source=data_source,
        campaign_name=campaign_name,
        station_name=station_name,
        base_dir=base_dir,
    )
    dst_issue_fpath = define_issue_filepath(
        data_source=data_source, campaign_name=campaign_name, station_name=station_name, base_dir=test_base_dir
    )
    copy_file(src_issue_fpath, dst_issue_fpath)
    copy_file(src_metadata_fpath, dst_metadata_fpath)
    print(f"The test DISDRODB archive for {data_source} {campaign_name} {station_name} has been set up !")
