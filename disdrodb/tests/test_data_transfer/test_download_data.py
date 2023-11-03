# #!/usr/bin/env python3

# # -----------------------------------------------------------------------------.
# # Copyright (c) 2021-2023 DISDRODB developers
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
# """Test DISDRODB download utility."""

import os

import pytest

from disdrodb.data_transfer.download_data import (
    _download_file_from_url,
    _is_empty_directory,
)
from disdrodb.utils.yaml import write_yaml


class TestIsEmptyDirectory:
    def test_non_existent_directory(self):
        with pytest.raises(OSError, match=r".* does not exist."):
            _is_empty_directory("non_existent_directory")

    def test_non_directory_path(self, tmp_path):
        # Create a temporary file
        file_path = tmp_path / "test_file.txt"
        file_path.write_text("This is a test file.")
        with pytest.raises(OSError, match=r".* is not a directory."):
            _is_empty_directory(str(file_path))

    def test_empty_directory(self, tmp_path):
        # `tmp_path` is a pytest fixture that provides a temporary directory unique to the test invocation
        assert _is_empty_directory(tmp_path)

    def test_non_empty_directory(self, tmp_path):
        # Create a temporary file inside the temporary directory
        file_path = tmp_path / "test_file.txt"
        file_path.write_text("This is a test file.")
        assert not _is_empty_directory(tmp_path)


def test_download_file_from_url(tmp_path):
    # DUBUG
    # tmp_path = "/tmp/empty_2"
    # os.makedirs(tmp_path)

    # Test download case when empty directory
    url = "https://httpbin.org/stream-bytes/1024"
    _download_file_from_url(url, tmp_path, force=False)
    filename = os.path.basename(url)  # README.md
    filepath = os.path.join(tmp_path, filename)
    assert os.path.isfile(filepath)

    # Test download case when directory is not empty and force=False --> avoid download
    url = "https://httpbin.org/stream-bytes/1025"
    _download_file_from_url(url, tmp_path, force=False)
    filename = os.path.basename(url)  # README.md
    filepath = os.path.join(tmp_path, filename)
    assert not os.path.isfile(filepath)

    # Test download case when directory is not empty and force=True --> it download
    url = "https://httpbin.org/stream-bytes/1026"
    _download_file_from_url(url, tmp_path, force=True)
    filename = os.path.basename(url)  # README.md
    filepath = os.path.join(tmp_path, filename)
    assert os.path.isfile(filepath)


def create_fake_metadata_file(
    tmp_path,
    data_source="data_source",
    campaign_name="campaign_name",
    station_name="station_name",
    with_url: bool = True,
):
    metadata_dir_path = tmp_path / "DISDRODB" / "Raw" / data_source / campaign_name / "metadata"
    metadata_dir_path.mkdir(parents=True)
    metadata_fpath = os.path.join(metadata_dir_path, f"{station_name}.yml")
    # Define fake metadata dictionary
    yaml_dict = {}
    yaml_dict["station_name"] = station_name
    if with_url:
        raw_github_path = "https://raw.githubusercontent.com"
        disdro_repo_path = f"{raw_github_path}/ltelab/disdrodb/main"
        test_data_path = "disdrodb/tests/data/test_data_download/station_files.zip"
        disdrodb_data_url = f"{disdro_repo_path}/{test_data_path}"
        yaml_dict["disdrodb_data_url"] = disdrodb_data_url
    # Write fake yaml file in temp folder
    write_yaml(yaml_dict, metadata_fpath)
    assert os.path.exists(metadata_fpath)
    return metadata_fpath


# def test_download_station_data(tmp_path):
#     # DUBUG
#     # from pathlib import Path
#     # tmp_path = Path("/tmp/empty_3")
#     # os.makedirs(tmp_path)

#     station_name = "station_name"
#     metadata_fpath = create_fake_metadata_file(tmp_path, station_name=station_name, with_url=True)
#     station_dir_path = metadata_fpath.replace("metadata", "data").replace(".yml", "")
#     _download_station_data(metadata_fpath=metadata_fpath, force=True)
#     # Assert files in the zip file have been unzipped
#     assert os.path.isfile(os.path.join(station_dir_path, "station_file1.txt"))
#     # Assert inner zipped files are not unzipped !
#     assert os.path.isfile(os.path.join(station_dir_path, "station_file2.zip"))
#     # Assert inner directories are there
#     assert os.path.isdir(os.path.join(station_dir_path, "2020"))
#     # Assert zip file has been removed
#     assert not os.path.exists(os.path.join(station_dir_path, "station_files.zip"))
