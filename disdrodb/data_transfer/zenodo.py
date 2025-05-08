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
"""DISDRODB Zenodo utility."""

import json
import os

import requests

from disdrodb.configs import get_zenodo_token
from disdrodb.utils.yaml import read_yaml, write_yaml


def _check_http_response(
    response: requests.Response,
    expected_status_code: int,
    task_description: str,
) -> None:
    """Check the Zenodo HTTP request response status code and raise an error if not the expected one."""
    if response.status_code == expected_status_code:
        return

    error_message = f"Error {task_description}: {response.status_code}"
    data = response.json()

    if "message" in data:
        error_message += f" {data['message']}"

    if "errors" in data:
        for sub_data in data["errors"]:
            error_message += f"\n- {sub_data['field']}: {sub_data['message']}"

    raise ValueError(error_message)


def _create_zenodo_deposition(sandbox) -> tuple[int, str]:
    """Create a new Zenodo deposition and get the deposit information.

    At every function call, the deposit_id and bucket url will change !

    Parameters
    ----------
    sandbox : bool
        If ``True``, create the deposit on Zenodo Sandbox for testing purposes.
        If ``False``, create the deposit on Zenodo.

    Returns
    -------
    deposit_id, bucket_url : Tuple[int, str]
        Zenodo deposition ID and bucket URL.

    """
    access_token = get_zenodo_token(sandbox=sandbox)

    # Define Zenodo deposition url
    host = "sandbox.zenodo.org" if sandbox else "zenodo.org"
    deposit_url = f"https://{host}/api/deposit/depositions"

    # Create a new deposition
    # url = f"{deposit_url}?access_token={access_token}"
    params = {"access_token": access_token}
    headers = {"Content-Type": "application/json"}
    response = requests.post(deposit_url, params=params, json={}, headers=headers)
    _check_http_response(response, 201, task_description="Creation of Zenodo deposition")

    # Get deposition ID and bucket URL
    data = response.json()
    deposit_id = data["id"]
    bucket_url = data["links"]["bucket"]
    deposit_url = f"{deposit_url}/{deposit_id}"
    return deposit_id, deposit_url, bucket_url


def _define_disdrodb_data_url(zenodo_host, deposit_id, filename):
    return f"https://{zenodo_host}/records/{deposit_id}/files/{filename}?download=1"


def _upload_file_to_zenodo(filepath: str, metadata_filepath: str, sandbox: bool) -> None:
    """Upload a file to a Zenodo bucket."""
    # Read metadata
    metadata = read_yaml(metadata_filepath)
    data_source = metadata["data_source"]
    campaign_name = metadata["campaign_name"]

    # Define Zenodo bucket url
    deposit_id, deposit_url, bucket_url = _create_zenodo_deposition(sandbox=sandbox)

    # Define remote filename and remote url
    # --> <data_source>-<campaign_name>-<station_name>.zip !
    filename = os.path.basename(filepath)
    filename = f"{data_source}-{campaign_name}-{filename}"
    remote_url = f"{bucket_url}/{filename}"

    # Define access tokens
    access_token = get_zenodo_token(sandbox)
    params = {"access_token": access_token}

    ###----------------------------------------------------------.
    # Upload data
    with open(filepath, "rb") as f:
        response = requests.put(remote_url, data=f, params=params)
    host_name = "Zenodo Sandbox" if sandbox else "Zenodo"
    _check_http_response(response, 201, f"Upload of {filepath} to {host_name}.")

    ###----------------------------------------------------------.
    # Add zenodo metadata
    headers = {"Content-Type": "application/json"}
    zenodo_metadata = _define_zenodo_metadata(metadata)
    response = requests.put(deposit_url, params=params, data=json.dumps(zenodo_metadata), headers=headers)
    _check_http_response(response, 200, "Add Zenodo metadata deposit.")

    ###----------------------------------------------------------.
    # Define disdrodb data url
    zenodo_host = "sandbox.zenodo.org" if sandbox else "zenodo.org"
    disdrodb_data_url = _define_disdrodb_data_url(zenodo_host, deposit_id, filename)

    # Define Zenodo url to review and publish the uploaded data
    review_url = f"https://{zenodo_host}/uploads/{deposit_id}"

    ###----------------------------------------------------------.
    print(f" - Please review your data deposition at {review_url} and publish it when ready !")
    print(f"   The direct link to download station data is {disdrodb_data_url}")
    return disdrodb_data_url


def _define_creators_list(metadata):
    """Try to define Zenodo creator list from DISDRODB metadata."""
    try:
        import re

        list_names = re.split(";|,", metadata["authors"])
        list_identifier = re.split(";|,", metadata["authors_url"])
        list_affiliation = re.split(";|,", metadata["institution"])

        # Check identifier fields match the number of specified authors
        # - If not, set identifier to ""
        if len(list_names) != len(list_identifier):
            list_identifier = [""] * len(list_names)

        # If affiliation is only one --> Assume is the same for everyone
        if len(list_affiliation) == 1:
            list_affiliation = list_affiliation * len(list_names)
        # If more than one affiliation, but does not match list of authors, set to ""
        if len(list_affiliation) > 1 and len(list_affiliation) != len(list_names):
            list_affiliation = [""] * len(list_names)

        # Create info dictionary of each author
        list_creator_dict = []
        for name, identifier, affiliation in zip(list_names, list_identifier, list_affiliation):
            creator_dict = {}
            creator_dict["name"] = name.strip()
            creator_dict["orcid"] = identifier.strip()
            creator_dict["affiliation"] = affiliation.strip()
            list_creator_dict.append(creator_dict)
    except Exception:
        list_creator_dict = []
    return list_creator_dict


def _define_zenodo_metadata(metadata):
    """Define Zenodo metadata from DISDRODB metadata."""
    data_source = metadata["data_source"]
    campaign_name = metadata["campaign_name"]
    station_name = metadata["station_name"]
    name = f"{data_source} {campaign_name} {station_name}"
    description = f"Disdrometer measurements of the {name} station. "
    description += "This dataset is part of the DISDRODB project. "
    description += "Station metadata are available at "
    description += f"https://github.com/ltelab/DISDRODB-METADATA/blob/main/DISDRODB/METADATA/{data_source}/{campaign_name}/metadata/{station_name}.yml . "  # noqa: E501
    description += "The software to easily process and standardize the raw data into netCDF files is available at "
    description += "https://github.com/ltelab/disdrodb ."

    zenodo_metadata = {
        "metadata": {
            "title": f"{name} disdrometer station data",
            "upload_type": "dataset",
            "description": description,
            "creators": _define_creators_list(metadata),
        },
    }
    return zenodo_metadata


def _update_metadata_with_zenodo_url(metadata_filepath: str, disdrodb_data_url: str) -> None:
    """Update metadata with Zenodo zip file url.

    Parameters
    ----------
    metadata_filepath: str
        Path to the station metadata file.
    disdrodb_data_url: str
        Remote URL where the station data are stored.
    """
    metadata_dict = read_yaml(metadata_filepath)
    metadata_dict["disdrodb_data_url"] = disdrodb_data_url
    write_yaml(metadata_dict, metadata_filepath)


def upload_station_to_zenodo(metadata_filepath: str, station_zip_filepath: str, sandbox: bool = True) -> str:
    """Zip station data, upload data to Zenodo and update the metadata disdrodb_data_url.

    Parameters
    ----------
    metadata_filepath: str
        Path to the station metadata file.
    station_zip_filepath: str
        Path to the zip file containing the station data.
    sandbox: bool
        If ``True``, upload to Zenodo Sandbox (for testing purposes).
        If ``False``, upload to Zenodo.
    """
    # Upload the station data zip file on Zenodo
    # - After upload, it removes the zip file !
    print(" - Uploading station data")
    try:
        disdrodb_data_url = _upload_file_to_zenodo(
            filepath=station_zip_filepath,
            metadata_filepath=metadata_filepath,
            sandbox=sandbox,
        )
        os.remove(station_zip_filepath)
    except Exception as e:
        os.remove(station_zip_filepath)
        raise ValueError(f"{station_zip_filepath} The upload on Zenodo has failed: {e}.")

    # Add the disdrodb_data_url information to the metadata
    print(" - The station metadata 'disdrodb_data_url' key has been updated with the remote url")
    _update_metadata_with_zenodo_url(metadata_filepath=metadata_filepath, disdrodb_data_url=disdrodb_data_url)
