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
"""DISDRODB raw data compression utility."""

import bz2
import gzip
import os
import shutil
import tempfile
import zipfile
from typing import Optional

from disdrodb.api.checks import check_data_archive_dir
from disdrodb.api.path import define_station_dir
from disdrodb.configs import get_data_archive_dir
from disdrodb.utils.directories import list_files
from disdrodb.utils.yaml import read_yaml

COMPRESSION_OPTIONS = {
    "zip": ".zip",
    "gzip": ".gz",
    "bzip2": ".bz2",
}


def unzip_file(filepath: str, dest_path: str) -> None:
    """Unzip a file into a directory.

    Parameters
    ----------
    filepath : str
        Path of the file to unzip.
    dest_path : str
        Path of the destination directory.
    """
    with zipfile.ZipFile(filepath, "r") as zip_ref:
        zip_ref.extractall(dest_path)


def _zip_dir(dir_path: str) -> str:
    """Zip a directory into a file located in the same directory.

    Parameters
    ----------
    dir_path : str
        Path of the directory to zip.

    Returns
    -------
    str
        Path of the zip archive.
    """
    output_path_without_extension = os.path.join(tempfile.gettempdir(), os.path.basename(dir_path))
    output_path = output_path_without_extension + ".zip"
    shutil.make_archive(output_path_without_extension, "zip", dir_path)
    return output_path


def check_consistent_station_name(metadata_filepath, station_name):
    """Check consistent station_name between YAML file name and metadata key."""
    # Check consistent station name
    expected_station_name = os.path.basename(metadata_filepath).replace(".yml", "")
    if station_name and str(station_name) != str(expected_station_name):
        raise ValueError(f"Inconsistent station_name values in the {metadata_filepath} file. Download aborted.")
    return station_name


def archive_station_data(metadata_filepath: str, data_archive_dir: str) -> str:
    """Archive station data into a zip file for subsequent data upload.

    It create a zip file into a temporary directory !

    Parameters
    ----------
    metadata_filepath: str
        Metadata file path.

    """
    # Open metadata file
    metadata_dict = read_yaml(metadata_filepath)
    # Retrieve station information
    data_archive_dir = get_data_archive_dir(data_archive_dir)
    data_source = metadata_dict["data_source"]
    campaign_name = metadata_dict["campaign_name"]
    station_name = metadata_dict["station_name"]
    station_name = check_consistent_station_name(metadata_filepath, station_name)
    # Define the destination local filepath path
    station_dir = define_station_dir(
        data_archive_dir=data_archive_dir,
        data_source=data_source,
        campaign_name=campaign_name,
        station_name=station_name,
        product="RAW",
    )
    # Zip station directory
    station_zip_filepath = _zip_dir(station_dir)
    return station_zip_filepath


def compress_station_files(
    data_archive_dir: str,
    data_source: str,
    campaign_name: str,
    station_name: str,
    method: str = "gzip",
    skip: bool = True,
) -> None:
    """Compress each raw file of a station.

    Parameters
    ----------
    data_archive_dir : str
        DISDRODB Data Archive directory
    data_source : str
        Name of data source of interest.
    campaign_name : str
        Name of the campaign of interest.
    station_name : str
        Station name of interest.
    method : str
        Compression method. ``"zip"``, ``"gzip"`` or ``"bzip2"``.
    skip : bool
        Whether to raise an error if a file is already compressed.
        If ``True``, it does not raise an error and try to compress the other files.
        If ``False``, it raise an error and stop the compression routine.
        The default value is ``True``.
    """
    if method not in COMPRESSION_OPTIONS:
        raise ValueError(f"Invalid compression method {method}. Valid methods are {list(COMPRESSION_OPTIONS.keys())}")

    data_archive_dir = check_data_archive_dir(data_archive_dir)
    station_dir = define_station_dir(
        data_archive_dir=data_archive_dir,
        product="RAW",
        data_source=data_source,
        campaign_name=campaign_name,
        station_name=station_name,
        check_exists=False,
    )
    if not os.path.isdir(station_dir):
        raise ValueError(f"Station data directory {station_dir} does not exist.")

    # Get list of files inside the station directory (in all nested directories)
    filepaths = list_files(station_dir, glob_pattern="*", recursive=True)
    for filepath in filepaths:
        _ = _compress_file(filepath, method, skip=skip)

    print(f"All files of {data_source} {campaign_name} {station_name} have been compressed.")
    print("Please now remember to update the glob_pattern of the reader ¨!")


def _compress_file(filepath: str, method: str, skip: bool) -> str:
    """Compress a file and delete the original.

    If the file is already compressed, it is not compressed again.

    Parameters
    ----------
    filepath : str
        Path of the file to compress.
    method : str
        Compression method. ``None``, ``"zip"``, ``"gzip"`` or ``"bzip2"``.
    skip : bool
        Whether to raise an error if a file is already compressed.
        If ``True``, it does not raise an error return the input filepath.
        If ``False``, it raise an error.

    Returns
    -------
    str
        Path of the compressed file. Same as input if no compression.
    """
    if filepath.endswith(".nc") or filepath.endswith(".netcdf4"):
        raise ValueError("netCDF files must be not compressed !")

    if _check_file_compression(filepath) is not None:
        if skip:
            print(f"File {filepath} is already compressed. Skipping.")
            return filepath
        raise ValueError(f"File {filepath} is already compressed !")

    extension = COMPRESSION_OPTIONS[method]
    archive_name = os.path.basename(filepath) + extension
    compressed_filepath = os.path.join(os.path.dirname(filepath), archive_name)
    compress_file_function = {
        "zip": _compress_file_zip,
        "gzip": _compress_file_gzip,
        "bzip2": _compress_file_bzip2,
    }[method]

    compress_file_function(filepath, compressed_filepath)
    os.remove(filepath)

    return compressed_filepath


def _check_file_compression(filepath: str) -> Optional[str]:
    """Check the method used to compress a raw text file.

    From https://stackoverflow.com/questions/13044562/python-mechanism-to-identify-compressed-file-type-and-uncompress

    Parameters
    ----------
    filepath : str
        Path of the file to check.

    Returns
    -------
    Optional[str]
        Compression method. ``None``, ``"zip"``, ``"gzip"`` or ``"bzip2"``.

    """
    magic_dict = {
        b"\x1f\x8b\x08": "gzip",
        b"\x42\x5a\x68": "bzip2",
        b"\x50\x4b\x03\x04": "zip",
    }
    with open(filepath, "rb") as f:
        file_start = f.read(4)
        for magic, filetype in magic_dict.items():
            if file_start.startswith(magic):
                return filetype

    return None


def _compress_file_zip(filepath: str, compressed_filepath: str) -> None:
    """Compress a single file into a zip archive.

    Parameters
    ----------
    filepath : str
        Path of the file to compress.

    compressed_filepath : str
        Path of the compressed file.

    """
    with zipfile.ZipFile(compressed_filepath, "w", compression=zipfile.ZIP_DEFLATED) as zipf:
        zipf.write(filepath, os.path.basename(filepath))


def _compress_file_gzip(filepath: str, compressed_filepath: str) -> None:
    """Compress a single file into a gzip archive.

    Parameters
    ----------
    filepath : str
        Path of the file to compress.

    compressed_filepath : str
        Path of the compressed file.

    """
    with open(filepath, "rb") as f_in, gzip.open(compressed_filepath, "wb") as f_out:
        f_out.writelines(f_in)


def _compress_file_bzip2(filepath: str, compressed_filepath: str) -> None:
    """Compress a single file into a bzip2 archive.

    Parameters
    ----------
    filepath : str
        Path of the file to compress.

    compressed_filepath : str
        Path of the compressed file.

    """
    with open(filepath, "rb") as f_in, bz2.open(compressed_filepath, "wb") as f_out:
        f_out.writelines(f_in)
