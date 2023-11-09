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
import logging
import os
import shutil
from disdrodb.api.info import infer_campaign_name_from_path, infer_data_source_from_path
from disdrodb.utils.directories import (
    check_directory_exist,
    copy_file,
    create_required_directory,
)


logger = logging.getLogger(__name__)


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


def _copy_station_metadata(raw_dir: str, processed_dir: str, station_name: str) -> None:
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
    # Get src and dst metadata directory
    raw_metadata_dir = os.path.join(raw_dir, "metadata")
    processed_metadata_dir = os.path.join(processed_dir, "metadata")
    # Retrieve the metadata fpath in the raw directory
    metadata_fname = f"{station_name}.yml"
    raw_metadata_fpath = os.path.join(raw_metadata_dir, metadata_fname)
    # Check the metadata exists
    if not os.path.isfile(raw_metadata_fpath):
        raise ValueError(f"No metadata available for {station_name} at {raw_metadata_fpath}")
    # Define the destination fpath
    processed_metadata_fpath = os.path.join(processed_metadata_dir, os.path.basename(raw_metadata_fpath))
    # Copy the metadata file
    copy_file(src_fpath=raw_metadata_fpath, dst_fpath=processed_metadata_fpath)
    return None


def _check_pre_existing_station_data(campaign_dir, product, station_name, force=False):
    """Check for pre-existing station data.

    - If force=True, remove all data inside the station folder.
    - If force=False, raise error.

    NOTE: force=False behaviour could be changed to enable updating of missing files.
         This would require also adding code to check whether a downstream file already exist.
    """
    from disdrodb.api.io import _get_list_stations_with_data

    # Get list of available stations
    list_stations = _get_list_stations_with_data(product=product, campaign_dir=campaign_dir)

    # Check if station data are already present
    station_already_present = station_name in list_stations

    # Define the station directory path
    station_dir = os.path.join(campaign_dir, product, station_name)

    # If the station data are already present:
    # - If force=True, remove all data inside the station folder
    # - If force=False, raise error
    if station_already_present:
        # Check is a directory
        check_directory_exist(station_dir)
        # If force=True, remove all the content
        if force:
            # Remove all station directory content
            shutil.rmtree(station_dir)
        else:
            msg = f"The station directory {station_dir} already exists and force=False."
            logger.error(msg)
            raise ValueError(msg)


def create_initial_directory_structure(
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
    from disdrodb.api.io import _get_list_stations_with_data
    from disdrodb.l0.check_directories import check_processed_dir, check_raw_dir

    # Check inputs
    raw_dir = check_raw_dir(raw_dir=raw_dir, verbose=verbose)
    processed_dir = check_processed_dir(processed_dir=processed_dir)

    # Check consistent data_source and campaign name
    _ = _check_data_source_consistency(raw_dir=raw_dir, processed_dir=processed_dir)
    _ = _check_campaign_name_consistency(raw_dir=raw_dir, processed_dir=processed_dir)

    # Get list of available stations (at raw level)
    list_stations = _get_list_stations_with_data(product="RAW", campaign_dir=raw_dir)

    # Check station is available
    if station_name not in list_stations:
        raise ValueError(f"No data available for station {station_name}. Available stations: {list_stations}.")

    # Create required directory (if they don't exists)
    create_required_directory(processed_dir, dir_name="metadata")
    create_required_directory(processed_dir, dir_name="info")
    create_required_directory(processed_dir, dir_name=product)

    # Copy the station metadata
    _copy_station_metadata(raw_dir=raw_dir, processed_dir=processed_dir, station_name=station_name)

    # Remove <product>/<station> directory if force=True
    _check_pre_existing_station_data(
        campaign_dir=processed_dir,
        product=product,
        station_name=station_name,
        force=force,
    )


def create_directory_structure(processed_dir, product, station_name, force, verbose=False):
    """Create directory structure for L0B and higher DISDRODB products."""
    from disdrodb.api.checks import check_product
    from disdrodb.api.io import _get_list_stations_with_data
    from disdrodb.l0.check_directories import check_processed_dir

    # Check inputs
    check_product(product)
    processed_dir = check_processed_dir(processed_dir=processed_dir)

    # Check station is available in the target processed_dir directory
    if product == "L0B":
        required_product = "L0A"
        list_stations = _get_list_stations_with_data(product=required_product, campaign_dir=processed_dir)
    else:
        raise NotImplementedError("product {product} not yet implemented.")

    if station_name not in list_stations:
        raise ValueError(
            f"No {required_product} data available for station {station_name}. Available stations: {list_stations}."
        )

    # Create required directory (if they don't exists)
    create_required_directory(processed_dir, dir_name=product)

    # Remove <product>/<station_name> directory if force=True
    _check_pre_existing_station_data(
        campaign_dir=processed_dir,
        product=product,
        station_name=station_name,
        force=force,
    )
