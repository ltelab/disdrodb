# #!/usr/bin/env python3

# # -----------------------------------------------------------------------------.
# # Copyright (c) 2021-2026 DISDRODB developers
# #
# # This program is free software: you can redistribute it and/or modify
# # it under the terms of the GNU General Public License as published by
# # the Free Software Foundation, either version 3 of the License, or
# # (at your option) any later version.
# #
# # This program is distributed in the hope that it will be useful,
# # but WITHOUT ANY WARRANTY; without even the implied warranty of
# # MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# # GNU General Public License for more details.
# #
# # You should have received a copy of the GNU General Public License
# # along with this program.  If not, see <http://www.gnu.org/licenses/>.
# # -----------------------------------------------------------------------------.
"""Test DISDRODB zenodo utility."""

import os

import pytest

from disdrodb.data_transfer.zenodo import (
    _check_http_response,
    _define_creators_list,
    _define_disdrodb_data_url,
    upload_station_to_zenodo,
)


class MockResponse:
    """Create Mock Request Response."""

    def __init__(self, status_code, json_data=None):
        self.status_code = status_code
        self._json_data = json_data or {}

    def json(self):
        return self._json_data


class Test_Check_Http_Response:
    """Test check_http_response behaviour."""

    def test_check_http_response_success(self):
        """Test case where the status code is as expected."""
        response = MockResponse(200)
        try:
            _check_http_response(response, 200, "test task")
        except ValueError:
            pytest.fail("Unexpected ValueError raised")

    def test_check_http_response_error_message(self):
        """Test case where the status code is different and an error message is included."""
        response = MockResponse(400, {"message": "Bad request"})
        with pytest.raises(ValueError) as exc_info:
            _check_http_response(response, 200, "test task")
        assert "Error test task: 400 Bad request" in str(exc_info.value)

    def test_check_http_response_detailed_errors(self):
        """Test case with detailed errors."""
        response = MockResponse(400, {"errors": [{"field": "title", "message": "Required"}]})
        with pytest.raises(ValueError) as exc_info:
            _check_http_response(response, 200, "test task")
        assert "Error test task: 400\n- title: Required" in str(exc_info.value)


def test_define_disdrodb_data_url():
    """Test Zenodo disdrodb_data_url."""
    zenodo_host = "zenodo.org"
    deposit_id = "123456"
    filename = "testfile.txt"

    expected_url = f"https://{zenodo_host}/records/{deposit_id}/files/{filename}?download=1"
    actual_url = _define_disdrodb_data_url(zenodo_host, deposit_id, filename)
    assert actual_url == expected_url, "URL does not match expected format"


class Test_Define_Creators_List:
    """Test define_creators_list."""

    def test_when_fields_number_correspond(self):
        metadata = {
            "authors": "John Doe; Jane Smith",
            "authors_url": "http://example.com/john, http://example.com/jane",
            "institution": "University A,  University B",
        }
        expected_result = [
            {"name": "John Doe", "orcid": "http://example.com/john", "affiliation": "University A"},
            {"name": "Jane Smith", "orcid": "http://example.com/jane", "affiliation": "University B"},
        ]

        assert _define_creators_list(metadata) == expected_result

    @pytest.mark.parametrize("key", ["authors", "authors_url", "institution"])
    def test_empty_key(self, key):
        metadata = {
            "authors": "John Doe",
            "authors_url": "http://example.com/john",
            "institution": "University A",
        }
        metadata[key] = ""
        key_value_mapping = {"authors": "name", "authors_url": "orcid", "institution": "affiliation"}
        expected_result = [
            {"name": "John Doe", "orcid": "http://example.com/john", "affiliation": "University A"},
        ]
        expected_result[0][key_value_mapping[key]] = ""
        assert _define_creators_list(metadata) == expected_result

    def test_all_empty_key(self):
        metadata = {
            "authors": "",
            "authors_url": "",
            "institution": "",
        }
        expected_result = [
            {"name": "", "orcid": "", "affiliation": ""},
        ]
        assert _define_creators_list(metadata) == expected_result

    def test_institution_is_replicated(self):
        metadata = {
            "authors": "John Doe; Jane Smith",
            "authors_url": "http://example.com/john, http://example.com/jane",
            "institution": "University A",
        }
        expected_result = [
            {"name": "John Doe", "orcid": "http://example.com/john", "affiliation": "University A"},
            {"name": "Jane Smith", "orcid": "http://example.com/jane", "affiliation": "University A"},
        ]

        assert _define_creators_list(metadata) == expected_result

    def test_identifiers_number_mismatch(self):
        metadata = {
            "authors": "John Doe; Jane Smith",
            "authors_url": "http://example.com/john",
            "institution": "University A",
        }
        expected_result = [
            {"name": "John Doe", "orcid": "", "affiliation": "University A"},
            {"name": "Jane Smith", "orcid": "", "affiliation": "University A"},
        ]
        assert _define_creators_list(metadata) == expected_result

    def test_institution_number_mismatch(self):
        metadata = {
            "authors": "John Doe; Jane Smith; Scooby Doo",
            "authors_url": "http://example.com/john",
            "institution": "University A, University B",
        }
        expected_result = [
            {"name": "John Doe", "orcid": "", "affiliation": ""},
            {"name": "Jane Smith", "orcid": "", "affiliation": ""},
            {"name": "Scooby Doo", "orcid": "", "affiliation": ""},
        ]
        assert _define_creators_list(metadata) == expected_result

    def test_when_key_is_missing(self):
        metadata = {"authors": "John Doe;Jane Smith", "institution": "University A, University B"}
        assert _define_creators_list(metadata) == []


def test_if_upload_raise_error_remove_zip_file(tmp_path, mocker):
    """Test that temporary zip file are removed if something fail !."""
    # Create a dummy file at station_zip_filepath
    station_name = "40"
    station_zip_fpath = str(tmp_path / f"{station_name}.zip")
    with open(station_zip_fpath, "w") as f:
        f.write("dummy content")
    assert os.path.exists(station_zip_fpath)

    # Mock stuffs
    mocker.patch("disdrodb.utils.compression._zip_dir", return_value=station_zip_fpath)
    mocker.patch(
        "disdrodb.data_transfer.zenodo._upload_file_to_zenodo",
        side_effect=Exception("Whatever error occurred"),
    )

    # Test it remove the file if something fail
    with pytest.raises(ValueError):
        upload_station_to_zenodo(metadata_filepath=f"{station_name}.yml", station_zip_filepath=station_zip_fpath)

    assert not os.path.exists(station_zip_fpath)
