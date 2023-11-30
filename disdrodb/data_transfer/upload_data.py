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
"""Routines to upload data to the DISDRODB Decentralized Data Archive."""

from typing import List, Optional

import click

from disdrodb.api.path import define_metadata_filepath
from disdrodb.data_transfer.zenodo import upload_station_to_zenodo
from disdrodb.metadata import get_list_metadata
from disdrodb.utils.yaml import read_yaml


def click_upload_options(function: object):
    function = click.option(
        "--platform",
        type=click.Choice(["zenodo", "sandbox.zenodo"], case_sensitive=False),
        show_default=True,
        default="sandbox.zenodo",
        help="Name of remote platform. If not provided (None), the default platform is Zenodo.",
    )(function)
    function = click.option(
        "-f",
        "--force",
        type=bool,
        show_default=True,
        default=False,
        help="Force uploading even if data already exists on another remote location.",
    )(function)
    return function


def click_upload_archive_options(function: object):
    """Click command line options for DISDRODB archive upload.

    Parameters
    ----------
    function: object
        Function.
    """
    function = click.option(
        "--data_sources",
        type=str,
        show_default=True,
        default="",
        help="""Data source name (eg: EPFL). If not provided (None),
    all data sources will be uploaded.
    Multiple data sources can be specified by separating them with spaces.
    """,
    )(function)
    function = click.option(
        "--campaign_names",
        type=str,
        show_default=True,
        default="",
        help="""Name of the campaign (eg:  EPFL_ROOF_2012).
    If not provided (None), all campaigns will be uploaded.
    Multiple campaign names can be specified by separating them with spaces.
    """,
    )(function)
    function = click.option(
        "--station_names",
        type=str,
        show_default=True,
        default="",
        help="""Station name. If not provided (None), all stations will be uploaded.
    Multiple station names  can be specified by separating them with spaces.
    """,
    )(function)
    return function


def _check_if_upload(metadata_filepath: str, force: bool):
    """Check if data must be uploaded."""
    if not force:
        disdrodb_data_url = read_yaml(metadata_filepath).get("disdrodb_data_url", "")
        if isinstance(disdrodb_data_url, str) and len(disdrodb_data_url) > 1:
            raise ValueError(f"'force' is False and {metadata_filepath} has already a 'disdrodb_data_url' specified.")


def _filter_already_uploaded(metadata_filepaths: List[str], force: bool) -> List[str]:
    """Filter metadata files that already have a remote url specified."""
    filtered = []
    for metadata_filepath in metadata_filepaths:
        try:
            _check_if_upload(metadata_filepath, force=force)
            filtered.append(metadata_filepath)
        except Exception:
            msg = (
                f"'force' is False and {metadata_filepath} has already a 'disdrodb_data_url' specified. Skipping data"
                " upload ..."
            )
            print(msg)
    return filtered


def _check_valid_platform(platform):
    """Check upload platform validity."""
    valid_platform = ["zenodo", "sandbox.zenodo"]
    if platform not in valid_platform:
        raise NotImplementedError(f"Invalid platform {platform}. Valid platforms are {valid_platform}.")


def upload_station(
    data_source: str,
    campaign_name: str,
    station_name: str,
    platform: Optional[str] = "sandbox.zenodo",
    force: bool = False,
    base_dir: Optional[str] = None,
) -> None:
    """
    Upload data from a single DISDRODB station on a remote repository.

    This function also automatically update the disdrodb_data url in the metadata file.

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
    base_dir : str, optional
        The base directory of DISDRODB, expected in the format ``<...>/DISDRODB``.
        If not specified, the path specified in the DISDRODB active configuration will be used.
    platform: str, optional
        Name of the remote platform.
        The default platform is "sandbox.zenodo"
    force: bool, optional
        If True, upload the data and overwrite the disdrodb_data_url.
        The default is force=False.

    """
    _check_valid_platform(platform)

    # Define metadata_filepath
    metadata_filepath = define_metadata_filepath(
        data_source=data_source,
        campaign_name=campaign_name,
        station_name=station_name,
        base_dir=base_dir,
        product="RAW",
        check_exists=True,
    )
    # Check if data must be uploaded
    _check_if_upload(metadata_filepath, force=force)

    print(f"Start uploading of {data_source} {campaign_name} {station_name}")
    # Upload the data
    if platform == "zenodo":
        upload_station_to_zenodo(metadata_filepath, sandbox=False)

    else:  # platform == "sandbox.zenodo":  # Only for testing purposes, not available through CLI
        upload_station_to_zenodo(metadata_filepath, sandbox=True)


def upload_archive(
    platform: Optional[str] = None,
    force: bool = False,
    base_dir: Optional[str] = None,
    **kwargs,
) -> None:
    """Find all stations containing local data and upload them to a remote repository.

    Parameters
    ----------
    platform: str, optional
        Name of the remote platform.
        If not provided (None), the default platform is Zenodo.
        The default is platform=None.
    force: bool, optional
        If True, upload even if data already exists on another remote location.
        The default is force=False.
    base_dir : str (optional)
        Base directory of DISDRODB. Format: <...>/DISDRODB
        If None (the default), the disdrodb config variable 'dir' is used.

    Other Parameters
    ----------------

    data_sources: str or list of str, optional
        Data source name (eg: EPFL).
        If not provided (None), all data sources will be uploaded.
        The default is data_source=None.
    campaign_names: str or list of str, optional
        Campaign name (eg:  EPFL_ROOF_2012).
        If not provided (None), all campaigns will be uploaded.
        The default is campaign_name=None.
    station_names: str or list of str, optional
        Station name.
        If not provided (None), all stations will be uploaded.
        The default is station_name=None.
    """
    _check_valid_platform(platform)

    # Get list metadata
    metadata_filepaths = get_list_metadata(
        **kwargs,
        base_dir=base_dir,
        with_stations_data=True,
    )
    # If force=False, keep only metadata without disdrodb_data_url
    if not force:
        metadata_filepaths = _filter_already_uploaded(metadata_filepaths, force=force)

    # Check there are some stations to upload
    if len(metadata_filepaths) == 0:
        print("There is no remaining data to upload.")
        return

    # Upload station data
    for metadata_filepath in metadata_filepaths:
        metadata = read_yaml(metadata_filepath)
        data_source = metadata["data_source"]
        campaign_name = metadata["campaign_name"]
        station_name = metadata["station_name"]
        try:
            upload_station(
                base_dir=base_dir,
                data_source=data_source,
                campaign_name=campaign_name,
                station_name=station_name,
                platform=platform,
                force=force,
            )
        except Exception as e:
            print(f"{e}")

    print("All data have been uploaded. Please review your data depositions and publish it when ready.")
