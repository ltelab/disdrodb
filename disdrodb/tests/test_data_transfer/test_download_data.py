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

from disdrodb.data_transfer.download_data import _download_file_from_url

# from disdrodb.data_transfer.download_data import _download_station_data
# from disdrodb.tests.conftest import create_fake_metadata_file


def test_download_file_from_url(tmp_path):
    # DUBUG
    # tmp_path = "/tmp/empty_2"
    # os.makedirs(tmp_path)

    # Test download case when empty directory
    # url = "https://raw.githubusercontent.com/ltelab/disdrodb/main/README.md"
    url = "https://httpbin.org/stream-bytes/1024"
    _download_file_from_url(url, tmp_path, force=False)
    filename = os.path.basename(url)  # README.md
    filepath = os.path.join(tmp_path, filename)
    assert os.path.isfile(filepath)

    # Test download case when directory is not empty and force=False --> avoid download
    # url = "https://raw.githubusercontent.com/ltelab/disdrodb/main/CODE_OF_CONDUCT.md"
    url = "https://httpbin.org/stream-bytes/1025"
    _download_file_from_url(url, tmp_path, force=False)
    filename = os.path.basename(url)  # README.md
    filepath = os.path.join(tmp_path, filename)
    assert not os.path.isfile(filepath)

    # Test download case when directory is not empty and force=True --> it download
    # url = "https://raw.githubusercontent.com/ltelab/disdrodb/main/CODE_OF_CONDUCT.md"
    url = "https://httpbin.org/stream-bytes/1026"
    _download_file_from_url(url, tmp_path, force=True)
    filename = os.path.basename(url)  # README.md
    filepath = os.path.join(tmp_path, filename)
    assert os.path.isfile(filepath)


# def test_download_station_data(tmp_path):
#     # DEBUG
#     # from pathlib import Path
#     # tmp_path = Path("/tmp/empty_3")
#     # os.makedirs(tmp_path)

#     # Define metadata
#     metadata_dict = {}
#     raw_github_path = "https://raw.githubusercontent.com"
#     disdro_repo_path = f"{raw_github_path}/ltelab/disdrodb/main"
#     test_data_path = "disdrodb/tests/data/test_data_download/station_files.zip"
#     disdrodb_data_url = f"{disdro_repo_path}/{test_data_path}"
#     metadata_dict["disdrodb_data_url"] = disdrodb_data_url
#     # Create metadata file
#     base_dir = tmp_path / "DISDRODB"
#     metadata_fpath = create_fake_metadata_file(base_dir=base_dir, metadata_dict=metadata_dict)
#     # Download data
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
