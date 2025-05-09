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
"""Define utilities for Directory/File Checks/Creation/Deletion."""

import glob
import logging
import os
import pathlib
import shutil
from typing import Union

from disdrodb.utils.list import flatten_list
from disdrodb.utils.logger import log_info

logger = logging.getLogger(__name__)


def ensure_string_path(path, msg, accepth_pathlib=False):
    """Ensure that the path is a string."""
    valid_types = (str, pathlib.PurePath) if accepth_pathlib else str
    if not isinstance(path, valid_types):
        raise TypeError(msg)
    return str(path)


def contains_netcdf_or_parquet_files(dir_path: str) -> bool:
    """Check (recursively) if a directory has any Parquet or netCDF file.

    os.walk under the hood uses os.scandir
    os.walk file generator + any() avoid use of while loop

    The function returns True as soon as one file is found (short-circuit)^; False otherwise.
    """
    suffixes = (".nc", ".parquet")
    return any(fname.endswith(suffixes) for _, _, files in os.walk(dir_path) for fname in files)


def contains_files(dir_path: str) -> bool:
    """Check (recursively) if a directory contains any file.

    os.walk under the hood uses os.scandir
    os.walk file generator + any() avoid use of while loop

    The function returns True as soon as one file is found (short-circuit); False otherwise.
    """
    return any(fname for _, _, files in os.walk(dir_path) for fname in files)


def check_glob_pattern(pattern: str) -> None:
    """Check if glob pattern is a string and is a valid pattern.

    Parameters
    ----------
    pattern : str
        String to be checked.
    """
    if not isinstance(pattern, str):
        raise TypeError("Expect pattern as a string.")
    if pattern[0] == "/":
        raise ValueError("glob_pattern should not start with /")
    if "//" in pattern:
        raise ValueError("glob_pattern expects path with single separators: /, not //")
    if "\\" in pattern:
        raise ValueError("glob_pattern expects path separators to be /, not \\")
    return pattern


def check_glob_patterns(patterns: Union[str, list]) -> list:
    """Check if glob patterns are valids."""
    if not isinstance(patterns, (str, list)):
        raise ValueError("'glob_patterns' must be a str or list of strings.")
    if isinstance(patterns, str):
        patterns = [patterns]
    patterns = [check_glob_pattern(pattern) for pattern in patterns]
    return patterns


def _recursive_glob(dir_path, glob_pattern):
    # ** search for in zero or all subdirectories recursively

    dir_path = pathlib.Path(dir_path)
    return [str(path) for path in dir_path.rglob(glob_pattern)]


def _list_paths(dir_path, glob_pattern, recursive=False):
    """Return a list of filepaths and directory paths based on a single glob pattern."""
    # If glob pattern has separators, disable recursive option
    if "/" in glob_pattern and "**" not in glob_pattern:
        recursive = False
    # Search paths
    if not recursive:
        return glob.glob(os.path.join(dir_path, glob_pattern))
    return _recursive_glob(dir_path, glob_pattern)


def list_paths(dir_path, glob_pattern, recursive=False):
    """Return a list of filepaths and directory paths.

    This function accept also a list of glob patterns !
    """
    # Check validity of glob pattern(s)
    glob_patterns = check_glob_patterns(glob_pattern)
    # Search path for specified glob patterns
    paths = flatten_list(
        [
            _list_paths(dir_path=dir_path, glob_pattern=glob_pattern, recursive=recursive)
            for glob_pattern in glob_patterns
        ],
    )
    return paths


def list_files(dir_path, glob_pattern, recursive=False):
    """Return a list of filepaths (exclude directory paths)."""
    paths = list_paths(dir_path, glob_pattern, recursive=recursive)
    filepaths = [f for f in paths if os.path.isfile(f)]
    return filepaths


def list_directories(dir_path, glob_pattern, recursive=False):
    """Return a list of directory paths (exclude file paths)."""
    paths = list_paths(dir_path, glob_pattern, recursive=recursive)
    dir_paths = [f for f in paths if os.path.isdir(f)]
    return dir_paths


def count_files(dir_path, glob_pattern, recursive=False):
    """Return the number of files (exclude directories)."""
    return len(list_files(dir_path, glob_pattern, recursive=recursive))


def count_directories(dir_path, glob_pattern, recursive=False):
    """Return the number of files (exclude directories)."""
    return len(list_directories(dir_path, glob_pattern, recursive=recursive))


def check_directory_exists(dir_path):
    """Check if the directory exists."""
    if not os.path.exists(dir_path):
        raise ValueError(f"{dir_path} directory does not exist.")
    if not os.path.isdir(dir_path):
        raise ValueError(f"{dir_path} is not a directory.")


def create_directory(path: str, exist_ok=True) -> None:
    """Create a directory at the provided path."""
    path = ensure_string_path(path, msg="'path' must be a string", accepth_pathlib=True)
    try:
        os.makedirs(path, exist_ok=exist_ok)
    except Exception as e:
        dir_path = os.path.dirname(path)
        dir_name = os.path.basename(path)
        msg = f"Can not create directory {dir_name} inside {dir_path}. Error: {e}"
        raise FileNotFoundError(msg)


def create_required_directory(dir_path, dir_name, exist_ok=True):
    """Create directory ``dir_name`` inside the ``dir_path`` directory."""
    dir_path = ensure_string_path(dir_path, msg="'path' must be a string", accepth_pathlib=True)
    new_dir_path = os.path.join(dir_path, dir_name)
    create_directory(path=new_dir_path, exist_ok=exist_ok)


def is_empty_directory(path):
    """Check if a directory path is empty.

    Return ``False`` if path is a file or non-empty directory.
    If the path does not exist, raise an error.
    """
    if not os.path.exists(path):
        raise OSError(f"{path} does not exist.")
    if not os.path.isdir(path):
        return False

    paths = os.listdir(path)
    return len(paths) == 0


def _remove_file_or_directories(path, logger=None):
    """Return the file/directory or subdirectories tree of ``path``.

    Use this function with caution.
    """
    # If file
    if os.path.isfile(path):
        os.remove(path)
        log_info(logger, msg=f"Deleted the file {path}")
    # If empty directory
    elif is_empty_directory(path):
        os.rmdir(path)
        log_info(logger, msg=f"Deleted the empty directory {path}")
    # If not empty directory
    else:
        shutil.rmtree(path)
        log_info(logger, msg=f"Deleted directories within {path}")


def remove_if_exists(path: str, force: bool = False, logger=None) -> None:
    """Remove file or directory if exists and ``force=True``.

    If ``force=False``, it raises an error.
    """
    # If the path does not exist, do nothing
    if not os.path.exists(path):
        return

    # If the path exists and force=False, raise Error
    if not force:
        msg = f"--force is False and a file already exists at: {path}"
        raise ValueError(msg)

    # If force=True, remove the file/directory or subdirectories and files !
    try:
        _remove_file_or_directories(path, logger=logger)
    except Exception as e:
        msg = f"Can not delete file(s) at {path}. The error is: {e}"
        raise ValueError(msg)


def copy_file(src_filepath, dst_filepath):
    """Copy a file from a location to another."""
    filename = os.path.basename(src_filepath)
    dst_dir = os.path.dirname(dst_filepath)
    try:
        shutil.copy(src_filepath, dst_filepath)
        msg = f"{filename} copied at {dst_filepath}."
        logger.info(msg)
    except Exception as e:
        msg = f"Something went wrong when copying {filename} into {dst_dir}.\n The error is: {e}."
        raise ValueError(msg)


def remove_path_trailing_slash(path: str) -> str:
    r"""
    Removes a trailing slash or backslash from a file path if it exists.

    This function ensures that the provided file path is normalized by removing
    any trailing directory separator characters (``'/'`` or ``'\\'``).
    This is useful for maintaining consistency in path strings and for
    preparing paths for operations that may not expect a trailing slash.

    Parameters
    ----------
    path : str
        The file path to normalize.

    Returns
    -------
    str
        The normalized path without a trailing slash.

    Raises
    ------
    TypeError
        If the input path is not a string.

    Examples
    --------
    >>> remove_trailing_slash("some/path/")
    'some/path'
    >>> remove_trailing_slash("another\\path\\")
    'another\\path'
    """
    path = ensure_string_path(path, msg="Expecting a string 'path'", accepth_pathlib=True)
    # Remove trailing slash or backslash (if present)
    path = path.rstrip("/\\")
    return path
