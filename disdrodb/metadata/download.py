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
"""Routine to download the DISDRODB Metadata Archive from GitHub."""
import io
import os
import urllib.request
import zipfile


def download_metadata_archive(root_dir):
    """Download the DISDRODB Metadata Archive.

    It returns the DISDRODB Metadata Archive directory.
    """
    # Define DISDRODB Metadata Archive GitHub URL
    archive_zip_url = "https://github.com/ltelab/disdrodb-data/archive/refs/heads/main.zip"

    # Download archive to disk
    resp = urllib.request.urlopen(archive_zip_url)
    archive_data = resp.read()

    # Unpack archive
    with zipfile.ZipFile(io.BytesIO(archive_data)) as zf:
        zf.extractall(root_dir)

    # Check "disdrodb-data-main" directory has been download
    if os.listdir(root_dir) != ["disdrodb-data-main"]:
        raise ValueError("The DISDRODB Metadata Archive hosted on Github could not been download !")

    # Define metadata directory
    metadata_dir = str(os.path.join(root_dir, "disdrodb-data-main", "DISDRODB"))
    return metadata_dir
