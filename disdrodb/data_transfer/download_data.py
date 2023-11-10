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

import os
import shutil
from typing import List, Optional, Union

import click
import pooch
import tqdm

from disdrodb.api.info import infer_disdrodb_tree_path
from disdrodb.configs import get_base_dir
from disdrodb.metadata import get_list_metadata
from disdrodb.utils.compression import _unzip_file
from disdrodb.utils.directories import is_empty_directory
from disdrodb.utils.yaml import read_yaml


def click_download_option(function: object):
    """Click command line options for DISDRODB archive download transfer.
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
    function = click.option(
        "-f",
        "--force",
        type=bool,
        show_default=True,
        default=True,
        help="Force overwriting",
    )(function)
    function = click.option(
        "--base_dir",
        type=str,
        show_default=True,
        default=None,
        help="DISDRODB base directory",
    )(function)
    return function


def download_archive(
    data_sources: Optional[Union[str, List[str]]] = None,
    campaign_names: Optional[Union[str, List[str]]] = None,
    station_names: Optional[Union[str, List[str]]] = None,
    force: bool = False,
    base_dir: Optional[str] = None,
):
    """Get all YAML files that contain the 'disdrodb_data_url' key
    and download the data locally.

    Parameters
    ----------
    data_sources : str or list of str, optional
        Data source name (eg : EPFL).
        If not provided (None), all data sources will be downloaded.
        The default is data_source=None.
    campaign_names : str or list of str, optional
        Campaign name (eg :  EPFL_ROOF_2012).
        If not provided (None), all campaigns will be downloaded.
        The default is campaign_name=None.
    station_names : str or list of str, optional
        Station name.
        If not provided (None), all stations will be downloaded.
        The default is station_name=None.
    force : bool, optional
        If True, overwrite the already existing raw data file.
        The default is False.
    base_dir : str (optional)
        Base directory of DISDRODB. Format: <...>/DISDRODB
        If None (the default), the disdrodb config variable 'dir' is used.

    """
    # Retrieve the requested metadata
    base_dir = get_base_dir(base_dir)
    metadata_filepaths = get_list_metadata(
        base_dir=base_dir,
        data_sources=data_sources,
        campaign_names=campaign_names,
        station_names=station_names,
        with_stations_data=False,
    )
    # Try to download the data
    # - It will download data only if the disdrodb_data_url is specified !
    for metadata_filepath in metadata_filepaths:
        try:
            _download_station_data(metadata_filepath, force)
        except Exception as e:
            station_dir = infer_disdrodb_tree_path(metadata_filepath).replace("metadata", "data").replace(".yml", "")
            print(f"ERROR during downloading the station {station_dir}: {e}")
            print(" ")


def _extract_station_files(zip_filepath, station_dir):
    """Extract files from the station.zip file and remove the station.zip file."""
    _unzip_file(filepath=zip_filepath, dest_path=station_dir)
    if os.path.exists(zip_filepath):
        os.remove(zip_filepath)


def _download_station_data(metadata_filepath: str, force: bool = False) -> None:
    """Download and unzip the station data .

    Parameters
    ----------
    metadata_filepaths : str
        Metadata file path.
    force : bool, optional
        force download, by default False

    """
    disdrodb_data_url, station_dir = _get_station_url_and_dir_path(metadata_filepath)
    if isinstance(disdrodb_data_url, str) and disdrodb_data_url != "":
        # Download file
        zip_filepath, to_unzip = _download_file_from_url(disdrodb_data_url, dst_dir=station_dir, force=force)
        # Extract the stations files from the downloaded station.zip file
        if to_unzip:
            _extract_station_files(zip_filepath, station_dir=station_dir)


def _get_valid_station_name(metadata_filepath, metadata_dict):
    """Check consistent station_name between YAML file name and metadata key."""
    # Check consistent station name
    expected_station_name = os.path.basename(metadata_filepath).replace(".yml", "")
    station_name = metadata_dict.get("station_name")
    if station_name and str(station_name) != str(expected_station_name):
        raise ValueError(f"Inconsistent station_name values in the {metadata_filepath} file. Download aborted.")
    return station_name


def _get_station_url_and_dir_path(metadata_filepath: str) -> tuple:
    """Return the station's remote url and the local destination directory path.

    Parameters
    ----------
    metadata_filepath : str
        Path to the metadata YAML file.

    Returns
    -------
    disdrodb_data_url, station_dir
        Tuple containing the remote url and the DISDRODB station directory path.
    """
    metadata_dict = read_yaml(metadata_filepath)
    station_name = _get_valid_station_name(metadata_filepath, metadata_dict)
    disdrodb_data_url = metadata_dict.get("disdrodb_data_url", None)
    # Define the destination local filepath path
    data_dir = os.path.dirname(metadata_filepath).replace("metadata", "data")
    station_dir = os.path.join(data_dir, station_name)
    return disdrodb_data_url, station_dir


def _download_file_from_url(url: str, dst_dir: str, force: bool = False) -> str:
    """Download station zip file into the DISDRODB station data directory.

    Parameters
    ----------
    url : str
        URL of the file to download.
    dst_dir : str
        Local directory where to download the file (DISDRODB station data directory).
    force : bool, optional
        Overwrite the raw data file if already existing, by default False.

    Returns
    -------
    dst_filepath
        Path of the downloaded file.
    to_unzip
        Flag that specify if the download station zip file must be unzipped.
    """
    filename = os.path.basename(url)
    dst_filepath = os.path.join(dst_dir, filename)
    os.makedirs(dst_dir, exist_ok=True)
    if not is_empty_directory(dst_dir):
        if force:
            shutil.rmtree(dst_dir)
            os.makedirs(dst_dir)  # station directory
        else:
            print(f"There are already files within {dst_dir}. Skipping the station data download.")
            to_unzip = False
            return dst_filepath, to_unzip
    os.makedirs(dst_dir, exist_ok=True)
    downloader = pooch.HTTPDownloader(progressbar=True)
    pooch.retrieve(url=url, known_hash=None, path=dst_dir, fname=filename, downloader=downloader, progressbar=tqdm)
    to_unzip = True
    return dst_filepath, to_unzip
