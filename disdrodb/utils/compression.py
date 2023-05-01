import bz2
import gzip
import os
import tempfile
import zipfile
from typing import Optional


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


def _zip_dir(dir_path: str, files_compression: Optional[str] = None) -> str:
    """Zip a directory into a file located in the same directory.

    Parameters
    ----------
    dir_path : str
        Path of the directory to zip
    files_compression : str
        Compression method for individual files inside directory

    Returns
    -------
    str
        Path of the zip archive
    """
    output_path = os.path.join(tempfile.gettempdir(), os.path.basename(dir_path) + ".zip")

    with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(dir_path):
            for file in files:
                file_path = os.path.join(root, file)
                file_path_in_archive = os.path.relpath(file_path, dir_path)
                compressed_file_path = _compress_file(file_path, files_compression)

                if compressed_file_path != file_path:
                    # Change file_path_in_archive to have compression extension
                    file_path_in_archive += os.path.splitext(compressed_file_path)[1]

                zipf.write(
                    compressed_file_path,
                    file_path_in_archive,
                )

                if compressed_file_path != file_path:
                    os.remove(compressed_file_path)

    return output_path


def _compress_file(file_path: str, method: Optional[str] = None) -> str:
    """Compress a file into a temporary file.

    Parameters
    ----------
    file_path : str
        Path of the file to compress
    method : str
        Compression method. None, "zip", "gzip" or "bzip2"

    Returns
    -------
    str
        Path of the compressed file. Same as input if no compression.
    """

    if method is None:
        return file_path

    try:
        extension = {
            "zip": ".zip",
            "gzip": ".gz",
            "bzip2": ".bz2",
        }[method]

    except KeyError:
        raise ValueError(f'Unknown compression method "{method}"')

    archive_name = os.path.basename(file_path) + extension
    compressed_file_path = os.path.join(tempfile.gettempdir(), archive_name)
    compress_file_function = {
        "zip": _compress_file_zip,
        "gzip": _compress_file_gzip,
        "bzip2": _compress_file_bzip2,
    }[method]

    compress_file_function(file_path, compressed_file_path)

    return compressed_file_path


def _compress_file_zip(file_path: str, compressed_file_path: str) -> None:
    """Compress a single file into a zip archive.

    Parameters
    ----------
    file_path : str
        Path of the file to compress
    compressed_file_path : str
        Path of the compressed file
    """

    with zipfile.ZipFile(compressed_file_path, "w", compression=zipfile.ZIP_DEFLATED) as zipf:
        zipf.write(file_path, os.path.basename(file_path))


def _compress_file_gzip(file_path: str, compressed_file_path: str) -> None:
    """Compress a single file into a gzip archive.

    Parameters
    ----------
    file_path : str
        Path of the file to compress
    compressed_file_path : str
        Path of the compressed file
    """

    with open(file_path, "rb") as f_in:
        with gzip.open(compressed_file_path, "wb") as f_out:
            f_out.writelines(f_in)


def _compress_file_bzip2(file_path: str, compressed_file_path: str) -> None:
    """Compress a single file into a bzip2 archive.

    Parameters
    ----------
    file_path : str
        Path of the file to compress
    compressed_file_path : str
        Path of the compressed file
    """

    with open(file_path, "rb") as f_in:
        with bz2.open(compressed_file_path, "wb") as f_out:
            f_out.writelines(f_in)
