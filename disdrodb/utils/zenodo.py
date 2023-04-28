import os
import requests
from typing import Tuple


def _get_zenodo_access_token(sandbox: bool = False) -> str:
    """Get Zenodo access token from environment variable or from user input."""

    access_token = os.environ.get("ZENODO_ACCESS_TOKEN")

    if not access_token:
        host = "sandbox.zenodo.org" if sandbox else "zenodo.org"
        print("Enter your Zenodo access token.")
        print(f"If you don't have one, create one on https://{host}/account/settings/applications/tokens/new/")
        access_token = input("")
        os.environ["ZENODO_ACCESS_TOKEN"] = access_token

    return access_token


def _check_http_response(
    response: requests.Response,
    expected_status_code: int,
    task_description: str,
) -> None:
    """Check the HTTP response status code and raise an error if not the expected one."""

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


def _create_zenodo_deposition(sandbox=False) -> Tuple[int, str]:
    """Create a new Zenodo deposition and get the deposition infos."""

    access_token = _get_zenodo_access_token(sandbox)
    host = "sandbox.zenodo.org" if sandbox else "zenodo.org"
    deposition_url = f"https://{host}/api/deposit/depositions"

    # Create a new deposition
    url = f"{deposition_url}?access_token={access_token}"
    headers = {"Content-Type": "application/json"}
    response = requests.post(url, json={}, headers=headers)
    _check_http_response(response, 201, "creating Zenodo deposition")

    # Get deposition ID and bucket URL
    data = response.json()
    deposition_id = data["id"]
    bucket_url = data["links"]["bucket"]

    return deposition_id, bucket_url


def _upload_file_to_zenodo(file_path: str, remote_url: str) -> None:
    """Upload a file to Zenodo into a bucket."""

    access_token = _get_zenodo_access_token()
    params = {"access_token": access_token}

    with open(file_path, "rb") as f:
        response = requests.put(remote_url, data=f, params=params)

    _check_http_response(response, 200, "uploading {file_path} to Zenodo")
