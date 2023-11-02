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
"""Test DISDRODB download utility."""

import os

import pytest
import yaml

from disdrodb.data_transfer import download_data


def create_fake_metadata_file(temp_path, data_source, campaign_name, station_name, with_url: bool = True):
    subfolder_path = temp_path / "DISDRODB" / "Raw" / data_source / campaign_name / "metadata"
    subfolder_path.mkdir(parents=True)
    # create a fake yaml file in temp folder
    with open(os.path.join(subfolder_path, f"{station_name}.yml"), "w") as f:
        yaml_dict = {}
        if with_url:
            yaml_dict["data_url"] = "https://www.example.com"
        yaml_dict["station_name"] = "station_name"

        yaml.dump(yaml_dict, f)

    assert os.path.exists(os.path.join(subfolder_path, f"{station_name}.yml"))


@pytest.mark.parametrize("url", ["https://raw.githubusercontent.com/ltelab/disdrodb/main/README.md"])
def test_download_file_from_url(url, tmp_path):
    download_data._download_file_from_url(url, tmp_path)
    assert os.path.isfile(os.path.join(tmp_path, os.path.basename(url))) is True
