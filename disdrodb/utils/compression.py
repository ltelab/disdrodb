import bz2
import glob
import gzip
import os
import shutil
import tempfile
import zipfile
from typing import Optional

from ..api.checks import check_disdrodb_dir


def _unzip_file(file_path: str, dest_path: str) -> None:
    """Unzip a file into a folder

    Parameters

    ----------
    file_path : str
        Path of the file to unzip
    dest_path : str
        Path of the destination folder
    """

    with zipfile.ZipFile(file_path, "r") as zip_ref:
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


def compress_station_files(
    disdrodb_dir: str, data_source: str, campaign_name: str, station_name: str, method: str
) -> None:
    """Compress all files of a station.

    Parameters
    ----------
    disdrodb_dir : str
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

    check_disdrodb_dir(str(disdrodb_dir))
    data_dir = os.path.join(disdrodb_dir, "Raw", data_source, campaign_name, "data", station_name)

    if not os.path.isdir(data_dir):
        print(f"Station data directory {data_dir} does not exist. Skipping.")
        return

    # use glob to get list of files recursively
    files = glob.glob(os.path.join(data_dir, "**"), recursive=True)

    for file_path in files:
        if os.path.isfile(file_path):
            _compress_file(file_path, method)


def _compress_file(file_path: str, method: str) -> str:
    """Compress a file and delete the original.

    If the file is already compressed, it is not compressed again.

    Parameters
    ----------
    file_path : str
        Path of the file to compress.

    method : str
        Compression method. None, "zip", "gzip" or "bzip2".


    Returns
    -------
    str
        Path of the compressed file. Same as input if no compression.
    """

    if _check_file_compression(file_path) is not None:
        print(f"File {file_path} is already compressed. Skipping.")
        return file_path

    valid_extensions = {
        "zip": ".zip",
        "gzip": ".gz",
        "bzip2": ".bz2",
    }

    if method not in valid_extensions:
        raise ValueError(f"Invalid compression method {method}. Valid methods are {list(valid_extensions.keys())}")

    extension = valid_extensions[method]
    archive_name = os.path.basename(file_path) + extension
    compressed_file_path = os.path.join(os.path.dirname(file_path), archive_name)
    compress_file_function = {
        "zip": _compress_file_zip,
        "gzip": _compress_file_gzip,
        "bzip2": _compress_file_bzip2,
    }[method]

    compress_file_function(file_path, compressed_file_path)
    os.remove(file_path)

    return compressed_file_path


def _check_file_compression(file_path: str) -> Optional[str]:
    """Check the method used to compress a file.

    From https://stackoverflow.com/questions/13044562/python-mechanism-to-identify-compressed-file-type-and-uncompress

    Parameters
    ----------
    file_path : str
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

    with open(file_path, "rb") as f:
        file_start = f.read(4)
        for magic, filetype in magic_dict.items():
            if file_start.startswith(magic):
                return filetype

    return None


def _compress_file_zip(file_path: str, compressed_file_path: str) -> None:
    """Compress a single file into a zip archive.

    Parameters
    ----------
    file_path : str
        Path of the file to compress.

    compressed_file_path : str
        Path of the compressed file.

    """

    with zipfile.ZipFile(compressed_file_path, "w", compression=zipfile.ZIP_DEFLATED) as zipf:
        zipf.write(file_path, os.path.basename(file_path))


def _compress_file_gzip(file_path: str, compressed_file_path: str) -> None:
    """Compress a single file into a gzip archive.

    Parameters
    ----------
    file_path : str
        Path of the file to compress.

    compressed_file_path : str
        Path of the compressed file.

    """

    with open(file_path, "rb") as f_in:
        with gzip.open(compressed_file_path, "wb") as f_out:
            f_out.writelines(f_in)


def _compress_file_bzip2(file_path: str, compressed_file_path: str) -> None:
    """Compress a single file into a bzip2 archive.

    Parameters
    ----------
    file_path : str
        Path of the file to compress.

    compressed_file_path : str
        Path of the compressed file.

    """

    with open(file_path, "rb") as f_in:
        with bz2.open(compressed_file_path, "wb") as f_out:
            f_out.writelines(f_in)
