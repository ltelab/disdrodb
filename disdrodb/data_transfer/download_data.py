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
"""Routines to download data from the DISDRODB Decentralized Data Archive."""

import logging
import os
import shutil
from typing import Optional, Union

import click
import pooch
import tqdm

from disdrodb.api.path import define_metadata_filepath, define_station_dir
from disdrodb.configs import get_data_archive_dir, get_metadata_archive_dir
from disdrodb.metadata import get_list_metadata
from disdrodb.utils.compression import unzip_file
from disdrodb.utils.directories import is_empty_directory
from disdrodb.utils.yaml import read_yaml


def click_download_archive_options(function: object):
    """Click command line options for DISDRODB archive download.

    Parameters
    ----------
    function : object
        Function.
    """
    function = click.option(
        "--data_sources",
        type=str,
        show_default=True,
        default="",
        help="""Data source name (eg : EPFL). If not provided (None),
    all data sources will be downloaded.
    Multiple data sources can be specified by separating them with spaces.
    """,
    )(function)
    function = click.option(
        "--campaign_names",
        type=str,
        show_default=True,
        default="",
        help="""Name of the campaign (eg :  EPFL_ROOF_2012).
    If not provided (None), all campaigns will be downloaded.
    Multiple campaign names can be specified by separating them with spaces.
    """,
    )(function)
    function = click.option(
        "--station_names",
        type=str,
        show_default=True,
        default="",
        help="""Station name. If not provided (None), all stations will be downloaded.
    Multiple station names  can be specified by separating them with spaces.

    """,
    )(function)
    return function


def click_download_options(function: object):
    """Click command line options for DISDRODB download.

    Parameters
    ----------
    function : object
        Function.
    """
    function = click.option(
        "-f",
        "--force",
        type=bool,
        show_default=True,
        default=False,
        help="Force overwriting",
    )(function)

    return function


def download_archive(
    data_sources: Optional[Union[str, list[str]]] = None,
    campaign_names: Optional[Union[str, list[str]]] = None,
    station_names: Optional[Union[str, list[str]]] = None,
    force: bool = False,
    data_archive_dir: Optional[str] = None,
    metadata_archive_dir: Optional[str] = None,
):
    """Download DISDRODB stations with the ``disdrodb_data_url`` in the metadata.

    Parameters
    ----------
    data_sources : str or list of str, optional
        Data source name (eg : EPFL).
        If not provided (``None``), all data sources will be downloaded.
        The default value is ``data_source=None``.
    campaign_names : str or list of str, optional
        Campaign name (eg :  EPFL_ROOF_2012).
        If not provided (``None``), all campaigns will be downloaded.
        The default value is ``campaign_name=None``.
    station_names : str or list of str, optional
        Station name.
        If not provided (``None``), all stations will be downloaded.
        The default value is ``station_name=None``.
    force : bool, optional
        If ``True``, overwrite the already existing raw data file.
        The default value is ``False``.
    data_archive_dir : str (optional)
        DISDRODB Data Archive directory. Format: ``<...>/DISDRODB``.
        If ``None`` (the default), the disdrodb config variable ``data_archive_dir`` is used.
    """
    # Retrieve the DISDRODB Metadata and Data Archive Directories
    data_archive_dir = get_data_archive_dir(data_archive_dir)
    metadata_archive_dir = get_metadata_archive_dir(metadata_archive_dir)

    # Select only metadata_filepaths with specified disdrodb_data_url
    metadata_filepaths = get_list_metadata(
        metadata_archive_dir=metadata_archive_dir,
        data_sources=data_sources,
        campaign_names=campaign_names,
        station_names=station_names,
        product=None,  # --> Search in DISDRODB Metadata Archive
        available_data=True,  # --> Select only metadata with disdrodb_data_url
        raise_error_if_empty=False,  # Do not raise error if no matching metadata file found
        invalid_fields_policy="raise",  # Raise error if invalid filtering criteria are specified
    )

    # Return early if no data to download
    if len(metadata_filepaths) == 0:
        print("No available data to download from the online DISDRODB Decentralized Archive.")
        return

    # Try to download the data
    # - It will download data only if the disdrodb_data_url is specified !
    for metadata_filepath in metadata_filepaths:
        metadata = read_yaml(metadata_filepath)
        data_source = metadata["data_source"]
        campaign_name = metadata["campaign_name"]
        station_name = metadata["station_name"]
        try:
            download_station(
                data_source=data_source,
                campaign_name=campaign_name,
                station_name=station_name,
                data_archive_dir=data_archive_dir,
                metadata_archive_dir=metadata_archive_dir,
                force=force,
            )
        except Exception:
            print(" - Download error: {e}")
            print(" ")


def download_station(
    data_source: str,
    campaign_name: str,
    station_name: str,
    force: bool = False,
    data_archive_dir: Optional[str] = None,
    metadata_archive_dir: Optional[str] = None,
) -> None:
    """
    Download data of a single DISDRODB station from the DISDRODB remote repository.

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
    data_archive_dir : str, optional
        The base directory of DISDRODB, expected in the format ``<...>/DISDRODB``.
        If not specified, the path specified in the DISDRODB active configuration will be used.
    force: bool, optional
        If ``True``, overwrite the already existing raw data file.
        The default value is ``False``.
    data_archive_dir : str (optional)
        DISDRODB Data Archive directory. Format: ``<...>/DISDRODB``.
        If ``None`` (the default), the disdrodb config variable ``data_archive_dir`` is used.
    """
    print(f"Start download of {data_source} {campaign_name} {station_name} station data")
    # Retrieve the DISDRODB Metadata and Data Archive Directories
    data_archive_dir = get_data_archive_dir(data_archive_dir)
    metadata_archive_dir = get_metadata_archive_dir(metadata_archive_dir)
    # Define metadata_filepath
    metadata_filepath = define_metadata_filepath(
        metadata_archive_dir=metadata_archive_dir,
        data_source=data_source,
        campaign_name=campaign_name,
        station_name=station_name,
        check_exists=True,
    )
    # Download data
    _download_station_data(metadata_filepath, data_archive_dir=data_archive_dir, force=force)


def _is_valid_disdrodb_data_url(disdrodb_data_url):
    """Check if it is a valid disdrodb_data_url."""
    return isinstance(disdrodb_data_url, str) and len(disdrodb_data_url) > 10


def _extract_station_files(zip_filepath, station_dir):
    """Extract files from the station.zip file and remove the station.zip file."""
    unzip_file(filepath=zip_filepath, dest_path=station_dir)
    if os.path.exists(zip_filepath):
        os.remove(zip_filepath)


def _download_station_data(metadata_filepath: str, data_archive_dir: str, force: bool = False) -> None:
    """Download and unzip the station data .

    Parameters
    ----------
    metadata_filepaths : str
        Metadata file path.
    force : bool, optional
        If ``True``, delete existing files and redownload it. The default value is ``False``.

    """
    # Open metadata file
    metadata_dict = read_yaml(metadata_filepath)
    # Retrieve station information
    data_archive_dir = get_data_archive_dir(data_archive_dir)
    data_source = metadata_dict["data_source"]
    campaign_name = metadata_dict["campaign_name"]
    station_name = metadata_dict["station_name"]
    station_name = check_consistent_station_name(metadata_filepath, station_name)
    # Define the destination local filepath path
    station_dir = define_station_dir(
        data_archive_dir=data_archive_dir,
        data_source=data_source,
        campaign_name=campaign_name,
        station_name=station_name,
        product="RAW",
    )
    # Check DISDRODB data url
    disdrodb_data_url = metadata_dict.get("disdrodb_data_url", None)
    if not _is_valid_disdrodb_data_url(disdrodb_data_url):
        raise ValueError(f"Invalid disdrodb_data_url '{disdrodb_data_url}' for station {station_name}")
    # Download file
    zip_filepath = _download_file_from_url(disdrodb_data_url, dst_dir=station_dir, force=force)
    # Extract the stations files from the downloaded station.zip file
    _extract_station_files(zip_filepath, station_dir=station_dir)


def check_consistent_station_name(metadata_filepath, station_name):
    """Check consistent station_name between YAML file name and metadata key."""
    # Check consistent station name
    expected_station_name = os.path.basename(metadata_filepath).replace(".yml", "")
    if station_name and str(station_name) != str(expected_station_name):
        raise ValueError(f"Inconsistent station_name values in the {metadata_filepath} file. Download aborted.")
    return station_name


def _download_file_from_url(url: str, dst_dir: str, force: bool = False) -> str:
    """Download station zip file into the DISDRODB station data directory.

    Parameters
    ----------
    url : str
        URL of the file to download.
    dst_dir : str
        Local directory where to download the file (DISDRODB station data directory).
    force : bool, optional
        Overwrite the raw data file if already existing. The default value is ``False``.

    Returns
    -------
    dst_filepath
        Path of the downloaded file.
    to_unzip
        Flag that specify if the download station zip file must be unzipped.
    """
    dst_filename = os.path.basename(dst_dir) + ".zip"
    dst_filepath = os.path.join(dst_dir, dst_filename)
    os.makedirs(dst_dir, exist_ok=True)
    if not is_empty_directory(dst_dir):
        if force:
            shutil.rmtree(dst_dir)
            os.makedirs(dst_dir)  # station directory
        else:
            raise ValueError(
                f"There are already raw files within the DISDRODB Data Archive at {dst_dir}. Download is suspended. "
                "Use force=True to force the download and overwrite existing raw files.",
            )

    os.makedirs(dst_dir, exist_ok=True)

    # Grab Pooch's logger and remember its current level
    logger = pooch.get_logger()
    orig_level = logger.level
    # Silence INFO messages (including the SHA256 print)
    logger.setLevel(logging.WARNING)
    # Define pooch downloader
    downloader = pooch.HTTPDownloader(progressbar=True)
    # Download the file
    pooch.retrieve(url=url, known_hash=None, path=dst_dir, fname=dst_filename, downloader=downloader, progressbar=tqdm)
    # Restore the previous logging level
    logger.setLevel(orig_level)
    return dst_filepath
