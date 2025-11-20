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
import subprocess
import urllib.parse
from typing import Optional, Union

import click
import pooch
import tqdm

from disdrodb.api.path import define_metadata_filepath, define_station_dir
from disdrodb.configs import get_data_archive_dir, get_metadata_archive_dir
from disdrodb.metadata import get_list_metadata
from disdrodb.utils.compression import unzip_file
from disdrodb.utils.directories import is_empty_directory, remove_file_or_directories
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
        If ``True``, delete existing files and re-download raw data files.
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
        except Exception as e:
            msg = e.args[0] if e.args else str(e)
            print(f" - Download error: {msg}")
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
        If ``True``, remove existing data and re-download.
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
    download_station_data(metadata_filepath, data_archive_dir=data_archive_dir, force=force)


def _is_valid_disdrodb_data_url(disdrodb_data_url):
    """Check if it is a valid disdrodb_data_url."""
    return isinstance(disdrodb_data_url, str) and len(disdrodb_data_url) > 10


def _extract_station_files(zip_filepath, station_dir):
    """Extract files from the station.zip file and remove the station.zip file."""
    unzip_file(filepath=zip_filepath, dest_path=station_dir)
    if os.path.exists(zip_filepath):
        os.remove(zip_filepath)


def check_consistent_station_name(metadata_filepath, station_name):
    """Check consistent station_name between YAML file name and metadata key."""
    # Check consistent station name
    expected_station_name = os.path.basename(metadata_filepath).replace(".yml", "")
    if station_name and str(station_name) != str(expected_station_name):
        raise ValueError(f"Inconsistent station_name values in the {metadata_filepath} file. Download aborted.")
    return station_name


def download_station_data(metadata_filepath: str, data_archive_dir: str, force: bool = False, verbose=True) -> None:
    """Download and unzip the station data.

    Parameters
    ----------
    metadata_filepaths : str
        Metadata file path.
    data_archive_dir : str (optional)
        DISDRODB Data Archive directory. Format: ``<...>/DISDRODB``.
        If ``None`` (the default), the disdrodb config variable ``data_archive_dir`` is used.
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
    # Define the path to the station RAW data directory
    station_dir = define_station_dir(
        data_archive_dir=data_archive_dir,
        data_source=data_source,
        campaign_name=campaign_name,
        station_name=station_name,
        product="RAW",
    )
    # Check DISDRODB data url
    disdrodb_data_url = metadata_dict.get("disdrodb_data_url", "")
    if disdrodb_data_url == "":
        raise ValueError(f"{campaign_name} {station_name} station data are not yet publicly available.")
    if not _is_valid_disdrodb_data_url(disdrodb_data_url):
        raise ValueError(f"Invalid disdrodb_data_url '{disdrodb_data_url}' for station {station_name}")

    # Remove existing station directory if force=True
    if force and os.path.exists(station_dir):
        print(f" - Removing existing station data at {station_dir}.")
        remove_file_or_directories(station_dir)

    # Download files
    # - Option 1: Download ZIP file containing all station raw data
    zip_repos = ["https://zenodo.org/", "https://cloudnet.fmi.fi/", "https://data.dtu.dk/"]
    if any(disdrodb_data_url.startswith(repo) for repo in zip_repos):
        download_zip_file(url=disdrodb_data_url, dst_dir=station_dir)

    # - Option 2: Recursive download from a web server via HTTP or HTTPS.
    elif disdrodb_data_url.startswith("http"):
        download_web_server_data(url=disdrodb_data_url, dst_dir=station_dir, verbose=verbose)
        # - Retry to be more sure that all data have been downloaded
        download_web_server_data(url=disdrodb_data_url, dst_dir=station_dir, verbose=verbose)

    # - Option 3: Recursive download from a ftp server
    elif disdrodb_data_url.startswith("ftp"):
        download_ftp_server_data(url=disdrodb_data_url, dst_dir=station_dir, verbose=verbose)
        # - Retry to be more sure that all data have been downloaded
        download_ftp_server_data(url=disdrodb_data_url, dst_dir=station_dir, verbose=verbose)

    else:
        raise NotImplementedError(f"Open a GitHub Issue to enable the download of data from {disdrodb_data_url}.")


####--------------------------------------------------------------------.
#### Download from Web Server via HTTP or HTTPS


def download_web_server_data(url: str, dst_dir: str, verbose=True) -> None:
    """Download data from a web server via HTTP or HTTPS.

    Use the system's wget command to recursively download all files and subdirectories
    under the given HTTPS “directory” URL. Works on both Windows and Linux, provided
    that wget is installed and on the PATH.

    1. Ensure wget is available.
    2. Normalize URL to end with '/'.
    3. Compute cut-dirs so that only the last segment of the path remains locally.
    4. Build and run the wget command.

    Parameters
    ----------
    url : str
        HTTPS URL pointing to webserver folder. Example: "https://ruisdael.citg.tudelft.nl/parsivel/PAR001_Cabauw/"
    dst_dir : str
         Local directory where to download the file (DISDRODB station data directory).
    verbose : bool, optional
        Print wget output (default is True).
    """
    # 1. Ensure wget exists
    ensure_wget_available()

    # 2. Normalize URL
    url = ensure_trailing_slash(url)

    # 3. Compute cut-dirs so that only the last URL segment remains locally
    cut_dirs = compute_cut_dirs(url)

    # 4. Create destination directory if needed
    os.makedirs(dst_dir, exist_ok=True)

    # 5. Build wget command
    cmd = build_webserver_wget_command(url, cut_dirs=cut_dirs, dst_dir=dst_dir, verbose=verbose)

    # 6. Run wget command
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        raise subprocess.CalledProcessError(
            returncode=e.returncode,
            cmd=e.cmd,
            output=e.output,
            stderr=e.stderr,
        )


def ensure_wget_available() -> None:
    """Raise FileNotFoundError if 'wget' is not on the system PATH."""
    if shutil.which("wget") is None:
        raise FileNotFoundError("The WGET software was not found. Please install WGET or add it to PATH.")


def ensure_trailing_slash(url: str) -> str:
    """Return `url` guaranteed to end with a slash."""
    return url if url.endswith("/") else url.rstrip("/") + "/"


def compute_cut_dirs(url: str) -> int:
    """Compute the wget cut_dirs value to download directly in `dst_dir`.

    Given a URL ending with '/', compute the total number of path segments.
    By returning len(segments), we strip away all of them—so that files
    within that final directory land directly in `dst_dir` without creating
    an extra subfolder.
    """
    parsed = urllib.parse.urlparse(url)
    path = parsed.path.strip("/")  # remove leading/trailing '/'
    segments = path.split("/") if path else []
    return len(segments)


def build_webserver_wget_command(url: str, cut_dirs: int, dst_dir: str, verbose: bool) -> list[str]:
    """Construct the wget command list for subprocess.run.

    Notes
    -----
    The following wget arguments are used
      - -q         : quiet mode (no detailed progress)
      - -r         : recursive
      - -np        : no parent
      - -nH        : no host directories
      - --timestamping: download missing files or when remote version is newer
      - --cut-dirs : strip all but the last path segment from the remote path
      - -P dst_dir : download into `dst_dir`
      - url
    """
    cmd = ["wget"]
    if not verbose:
        cmd.append("-q")
    cmd += [
        "-r",
        "-np",
        "-nH",
        "--reject",
        "index.html*",  # avoid to download Apache autoindex index.html
        f"--cut-dirs={cut_dirs}",
        # Downloads just new data without re-downloading existing files
        "--timestamping",  # -N
    ]

    # Define source and destination directory
    cmd += [
        "-P",
        dst_dir,
        url,
    ]
    return cmd


####--------------------------------------------------------------------.
#### Download from FTP Server


def build_ftp_server_wget_command(
    url: str,
    cut_dirs: int,
    dst_dir: str,
    verbose: bool,
) -> list[str]:
    """Construct the wget command list for FTP recursive download.

    Parameters
    ----------
    url : str
        FTP URL to download from.
    cut_dirs : int
        Number of leading path components to strip.
    dst_dir : str
        Local destination directory.
    verbose : bool
        If False, suppress wget output (-q).
    """
    cmd = ["wget"]  # base command

    if not verbose:
        cmd.append("-q")  # quiet mode --> no output except errors

    cmd += [
        "-r",  # recursive --> traverse into subdirectories
        "-np",  # no parent --> don't ascend to higher-level dirs
        "-nH",  # no host dirs --> avoid creating ftp.example.com/ locally
        f"--cut-dirs={cut_dirs}",  # strip N leading path components
        "--timestamping",  # download if remote file is newer
        "-P",  # specify local destination directory
        dst_dir,
        f"ftp://anonymous:disdrodb@{url}",  # target FTP URL
    ]
    return cmd


def download_ftp_server_data(url: str, dst_dir: str, verbose: bool = True) -> None:
    """Download data from an FTP server with anonymous login.

    Parameters
    ----------
    url : str
        FTP server URL pointing to a folder. Example: "ftp://ftp.example.com/path/to/data/"
    dst_dir : str
         Local directory where to download the file (DISDRODB station data directory).
    verbose : bool, optional
        Print wget output (default is True).
    """
    ensure_wget_available()

    # Ensure trailing slash
    url = ensure_trailing_slash(url)

    # Compute cut-dirs so files land directly in dst_dir
    cut_dirs = compute_cut_dirs(url)

    # Make destination directory
    os.makedirs(dst_dir, exist_ok=True)

    # Build wget command
    cmd = build_ftp_server_wget_command(
        url,
        cut_dirs=cut_dirs,
        dst_dir=dst_dir,
        verbose=verbose,
    )
    # Run wget
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        raise subprocess.CalledProcessError(
            returncode=e.returncode,
            cmd=e.cmd,
            output=e.output,
            stderr=e.stderr,
        )


####--------------------------------------------------------------------.
#### Download from Zenodo


def download_zip_file(url, dst_dir):
    """Download zip file from zenodo and extract station raw data."""
    # Download zip file
    zip_filepath = _download_file_from_url(url, dst_dir=dst_dir)
    # Extract the stations files from the downloaded station.zip file
    _extract_station_files(zip_filepath, station_dir=dst_dir)


def _download_file_from_url(url: str, dst_dir: str) -> str:
    """Download station zip file into the DISDRODB station data directory.

    Parameters
    ----------
    url : str
        URL of the file to download.
    dst_dir : str
        Local directory where to download the file (DISDRODB station data directory).

    Returns
    -------
    dst_filepath
        Path of the downloaded file.
    to_unzip
        Flag that specify if the download station zip file must be unzipped.
    """
    dst_filename = os.path.basename(dst_dir) + ".zip"
    dst_filepath = os.path.join(dst_dir, dst_filename)
    # Ensure destination directory exists and is empty
    os.makedirs(dst_dir, exist_ok=True)
    if not is_empty_directory(dst_dir):
        raise ValueError(
            "ZIP archive download is aborted."
            f"There are already raw files within the DISDRODB Data Archive at {dst_dir}."
            "Use force=True if you wish to remove existing files and redownload the station archive.",
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
