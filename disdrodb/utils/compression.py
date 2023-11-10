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

from disdrodb.api.checks import check_base_dir
from disdrodb.api.io import define_station_dir
from disdrodb.utils.directories import list_files


def _unzip_file(filepath: str, dest_path: str) -> None:
    """Unzip a file into a directory

    Parameters

    ----------
    filepath : str
        Path of the file to unzip
    dest_path : str
        Path of the destination directory
    """

    with zipfile.ZipFile(filepath, "r") as zip_ref:
        zip_ref.extractall(dest_path)


def _zip_dir(dir_path: str) -> str:
    """Zip a directory into a file located in the same directory.

    Parameters
    ----------
    dir_path : str
        Path of the directory to zip

    Returns
    -------
    str
        Path of the zip archive
    """

    output_path_without_extension = os.path.join(tempfile.gettempdir(), os.path.basename(dir_path))
    output_path = output_path_without_extension + ".zip"
    shutil.make_archive(output_path_without_extension, "zip", dir_path)
    return output_path


def compress_station_files(base_dir: str, data_source: str, campaign_name: str, station_name: str, method: str) -> None:
    """Compress all files of a station.

    Parameters
    ----------
    base_dir : str
        Base directory of DISDRODB
    data_source : str
        Name of data source of interest.
    campaign_name : str
        Name of the campaign of interest.
    station_name : str
        Station name of interest.
    method : str
        Compression method. "zip", "gzip" or "bzip2".

    """

    base_dir = check_base_dir(base_dir)
    station_dir = define_station_dir(
        base_dir=base_dir,
        product="RAW",
        data_source=data_source,
        campaign_name=campaign_name,
        station_name=station_name,
        check_exists=False,
    )
    if not os.path.isdir(station_dir):
        print(f"Station data directory {station_dir} does not exist. Skipping.")
        return

    # use glob to get list of files recursively
    filepaths = list_files(station_dir, glob_pattern="*", recursive=True)

    for filepath in filepaths:
        if os.path.isfile(filepath):
            _compress_file(filepath, method)


def _compress_file(filepath: str, method: str) -> str:
    """Compress a file and delete the original.

    If the file is already compressed, it is not compressed again.

    Parameters
    ----------
    filepath : str
        Path of the file to compress.

    method : str
        Compression method. None, "zip", "gzip" or "bzip2".


    Returns
    -------
    str
        Path of the compressed file. Same as input if no compression.
    """

    if _check_file_compression(filepath) is not None:
        print(f"File {filepath} is already compressed. Skipping.")
        return filepath

    valid_extensions = {
        "zip": ".zip",
        "gzip": ".gz",
        "bzip2": ".bz2",
    }

    if method not in valid_extensions:
        raise ValueError(f"Invalid compression method {method}. Valid methods are {list(valid_extensions.keys())}")

    extension = valid_extensions[method]
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
    """Check the method used to compress a file.

    From https://stackoverflow.com/questions/13044562/python-mechanism-to-identify-compressed-file-type-and-uncompress

    Parameters
    ----------
    filepath : str
        Path of the file to check.


    Returns
    -------
    Optional[str]
        Compression method. None, "zip", "gzip" or "bzip2".

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

    with open(filepath, "rb") as f_in:
        with gzip.open(compressed_filepath, "wb") as f_out:
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

    with open(filepath, "rb") as f_in:
        with bz2.open(compressed_filepath, "wb") as f_out:
            f_out.writelines(f_in)


def archive_station_data(metadata_filepath: str) -> str:
    """Archive station data into a zip file (based on metadata filepath).

    It create a zip file into a temporary directory !

    Parameters
    ----------
    metadata_filepath: str
        Metadata file path.

    Returns
    -------
    station_zip_filepath
        Filepath of the zip file containing the station's data.
    """

    station_data_path = metadata_filepath.replace("metadata", "data")
    station_data_path = os.path.splitext(station_data_path)[0]  # remove trailing ".yml"
    station_zip_filepath = _zip_dir(station_data_path)
    return station_zip_filepath
