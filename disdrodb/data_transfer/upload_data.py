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

import os
from typing import List, Optional

import click

from disdrodb.api.metadata import get_list_metadata
from disdrodb.utils.compression import _zip_dir
from disdrodb.utils.yaml import read_yaml, write_yaml
from disdrodb.utils.zenodo import _create_zenodo_deposition, _upload_file_to_zenodo


def click_upload_option(function: object):
    """Click command line options for DISDRODB archive upload transfer.

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
        help="""Data source folder name (eg: EPFL). If not provided (None),
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
    function = click.option(
        "--platform",
        type=click.Choice(["zenodo"], case_sensitive=False),
        show_default=True,
        default="",
        help="Name of remote platform. If not provided (None), the default platform is Zenodo.",
    )(function)
    function = click.option(
        "-f",
        "--force",
        type=bool,
        show_default=True,
        default=True,
        help="Force uploading even if data already exists on another remote location.",
    )(function)
    function = click.option(
        "--base_dir",
        type=str,
        show_default=True,
        default=None,
        help="DISDRODB root directory",
    )(function)
    return function


def _filter_already_uploaded(metadata_fpaths: List[str]) -> List[str]:
    """Filter metadata files that already have a remote url specified."""

    filtered = []

    for metadata_fpath in metadata_fpaths:
        metadata_dict = read_yaml(metadata_fpath)
        if metadata_dict.get("data_url"):
            print(f"{metadata_fpath} already has a remote url specified. Skipping.")
            continue
        filtered.append(metadata_fpath)

    return filtered


def _upload_data_to_zenodo(metadata_fpaths: List[str], sandbox: bool = False) -> None:
    """Upload data to Zenodo.

    Parameters
    ----------
    metadata_fpaths: list of str
        List of metadata file paths.
    sandbox: bool
        If True, upload to Zenodo sandbox for testing purposes.
    """

    deposition_id, bucket_url = _create_zenodo_deposition(sandbox)
    zenodo_host = "sandbox.zenodo.org" if sandbox else "zenodo.org"
    deposition_url = f"https://{zenodo_host}/deposit/{deposition_id}"
    print(f"Zenodo deposition created: {deposition_url}.")

    for metadata_fpath in metadata_fpaths:
        remote_path = _upload_station_data_to_zenodo(metadata_fpath, bucket_url)
        _update_metadata_with_zenodo_url(metadata_fpath, deposition_id, remote_path, sandbox)

    print("Data uploaded. Please review your deposition an publish it when ready.")


def _generate_data_remote_path(metadata_fpath: str) -> str:
    """Generate data remote path from a metadata path.

    metadata_fpath has the form "base_dir/Raw/data_source/campaign_name/metadata/station_name.yml".
    The remote path has the form "data_source/campaign_name/station_name".

    Parameters
    ----------
    metadata_fpath: str
        Metadata file path.
    """

    remote_path = os.path.normpath(metadata_fpath)
    # Remove up to "Raw/"
    remote_path = remote_path.split("Raw" + os.sep)[1]
    # Remove "/metadata"
    remote_path = remote_path.replace(os.sep + "metadata", "")
    # Remove trailing ".yml"
    remote_path = os.path.splitext(remote_path)[0]

    return remote_path


def _upload_station_data_to_zenodo(metadata_fpath: str, bucket_url: str) -> str:
    """Zip and upload station data to Zenodo.

    Update the metadata file with the remote url, and zip the data directory before uploading.

    Parameters
    ----------
    metadata_fpath: str
        Metadata file path.
    bucket_url: str
        Zenodo bucket url.
    """

    remote_path = _generate_data_remote_path(metadata_fpath)
    remote_url = f"{bucket_url}/{remote_path}.zip"
    temp_zip_path = _archive_station_data(metadata_fpath)

    _upload_file_to_zenodo(temp_zip_path, remote_url)

    os.remove(temp_zip_path)

    return remote_path


def _archive_station_data(metadata_fpath: str) -> str:
    """Archive station data.

    Parameters
    ----------
    metadata_fpath: str
        Metadata file path.
    """

    data_path = metadata_fpath.replace("metadata", "data")
    data_path = os.path.splitext(data_path)[0]  # remove trailing ".yml"
    temp_zip_path = _zip_dir(data_path)

    return temp_zip_path


def _update_metadata_with_zenodo_url(
    metadata_fpath: str, deposition_id: int, remote_path: str, sandbox: bool = False
) -> None:
    """Update metadata with Zenodo zip file url.

    Parameters
    ----------
    metadata_fpath: str
        Metadata file path.
    deposition_id: int
        Zenodo deposition id.
    remote_path: str
        Remote path of the zip file.
    sandbox: bool
        If True, set reference to Zenodo sandbox for testing purposes.
    """
    zenodo_host = "sandbox.zenodo.org" if sandbox else "zenodo.org"
    metadata_dict = read_yaml(metadata_fpath)
    metadata_dict["data_url"] = f"https://{zenodo_host}/record/{deposition_id}/files/{remote_path}.zip"
    write_yaml(metadata_dict, metadata_fpath)


def upload_disdrodb_archives(
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
        Data source folder name (eg: EPFL).
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

    metadata_fpaths = get_list_metadata(
        **kwargs,
        base_dir=base_dir,
        with_stations_data=True,
    )

    if not force:
        metadata_fpaths = _filter_already_uploaded(metadata_fpaths)

    if len(metadata_fpaths) == 0:
        print("There is no data fulfilling the criteria.")
        return

    if platform == "zenodo":
        _upload_data_to_zenodo(metadata_fpaths)

    elif platform == "sandbox.zenodo":  # Only for testing purposes, not available through CLI
        _upload_data_to_zenodo(metadata_fpaths, sandbox=True)
