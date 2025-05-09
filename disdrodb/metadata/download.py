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
import shutil
import urllib.request
import zipfile


def download_metadata_archive(directory_path, force=False):
    """Download the DISDRODB Metadata Archive to the specified directory.

    Parameters
    ----------
    directory_path : str
        The directory path where the DISDRODB-METADATA directory will be downloaded.
    force : bool, optional
        If ``True``, the existing DISDRODB-METADATA directory will be removed
        and a new one will be downloaded. The default value is ``False``.

    Returns
    -------
    metadata_archive_dir
        The DISDRODB Metadata Archive directory path.
    """
    # Define DISDRODB Metadata Archive GitHub URL
    archive_zip_url = "https://github.com/ltelab/DISDRODB-METADATA/archive/refs/heads/main.zip"

    # Download archive to disk
    resp = urllib.request.urlopen(archive_zip_url)
    archive_data = resp.read()

    # Unpack archive
    with zipfile.ZipFile(io.BytesIO(archive_data)) as zf:
        zf.extractall(directory_path)

    # Check the archive has been download
    extracted_dir = os.path.join(directory_path, "DISDRODB-METADATA-main")
    if not os.path.isdir(extracted_dir):
        raise ValueError(
            "The DISDRODB Metadata Archive hosted on GitHub could not be downloaded!",
        )

    # Define target directory for the metadata archive
    target_dir = os.path.join(directory_path, "DISDRODB-METADATA")

    # Handle existing target directory
    if os.path.exists(target_dir):
        if force:
            shutil.rmtree(target_dir)
        else:
            raise FileExistsError(
                f"A DISDRODB Metadata Archive already exists at '{target_dir}'. Use force=True to update it.",
            )

    # Rename extracted directory to target
    shutil.move(extracted_dir, target_dir)

    # Define metadata archive directory
    metadata_archive_dir = os.path.join(target_dir, "DISDRODB")

    print("The DISDRODB Metadata Archive has been download successfully.")
    print(f"The DISDRODB Metadata Archive directory path is {metadata_archive_dir}")
    return metadata_archive_dir
