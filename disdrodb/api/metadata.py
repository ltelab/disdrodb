#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# -----------------------------------------------------------------------------.
# Copyright (c) 2021-2022 DISDRODB developers
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
import os
import glob
import yaml
from disdrodb.api.io import _get_disdrodb_directory


def read_station_metadata(
    disdrodb_dir, product_level, data_source, campaign_name, station_name
):
    """Open the station metadata YAML file into a dictionary."""
    # Retrieve campaign directory
    campaign_dir = _get_disdrodb_directory(
        disdrodb_dir=disdrodb_dir,
        product_level=product_level,
        data_source=data_source,
        campaign_name=campaign_name,
        check_exist=True,
    )
    # Define metadata filepath
    fpath = os.path.join(campaign_dir, "metadata", f"{station_name}.yml")

    # Check the file exists
    if not os.path.exists(fpath):
        raise ValueError(
            f"The metadata file for {station_name} at {fpath} does not exists."
        )

    # Read the metadata file
    with open(fpath, "r") as f:
        dictionary = yaml.safe_load(f)
    return dictionary


def get_metadata_list(disdrodb_dir):
    """Get the list of metadata filepaths in the DISDRODB Raw directory."""
    fpaths = glob.glob(os.path.join(disdrodb_dir, "Raw/*/*/metadata/*.yml"))
    return fpaths
