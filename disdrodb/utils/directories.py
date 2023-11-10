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

logger = logging.getLogger(__name__)


def ensure_string_path(path, msg, accepth_pathlib=False):
    if accepth_pathlib:
        valid_types = (str, pathlib.PurePath)
    else:
        valid_types = str
    if not isinstance(path, valid_types):
        raise TypeError(msg)
    return str(path)


def list_files(glob_pattern):
    """Return a list of filepaths (exclude directory paths)."""
    paths = glob.glob(glob_pattern)
    filepaths = [f for f in paths if os.path.isfile(f)]
    return filepaths


def list_directories(glob_pattern):
    """Return a list of directory paths (exclude file paths)."""
    paths = glob.glob(glob_pattern)
    dir_paths = [f for f in paths if os.path.isdir(f)]
    return dir_paths


def count_files(glob_pattern):
    """Return the number of files (exclude directories)."""
    return len(list_files(glob_pattern))


def check_directory_exists(dir_path):
    """Check if the directory exist."""
    if not os.path.exists(dir_path):
        raise ValueError(f"{dir_path} directory does not exist.")
    if not os.path.isdir(dir_path):
        raise ValueError(f"{dir_path} is not a directory.")


def create_directory(path: str, exist_ok=True) -> None:
    """Create a directory at the provided path."""
    path = ensure_string_path(path, msg="'path' must be a string", accepth_pathlib=True)
    try:
        os.makedirs(path, exist_ok=exist_ok)
        logger.debug(f"Created directory {path}.")
    except Exception as e:
        dir_name = os.path.basename(path)
        msg = f"Can not create folder {dir_name} inside <path>. Error: {e}"
        logger.exception(msg)
        raise FileNotFoundError(msg)


def create_required_directory(dir_path, dir_name):
    """Create directory <dir_name> inside the <dir_path> directory."""
    try:
        new_dir_path = os.path.join(dir_path, dir_name)
        os.makedirs(new_dir_path, exist_ok=True)
    except Exception as e:
        msg = f"Can not create folder {dir_name} at {new_dir_path}. Error: {e}"
        logger.exception(msg)
        raise FileNotFoundError(msg)


def is_empty_directory(path):
    """Check if a directory path is empty.

    Return False if path is a file or non-empty directory.
    If the path does not exist, raise an error.
    """
    if not os.path.exists(path):
        raise OSError(f"{path} does not exist.")
    if not os.path.isdir(path):
        return False

    list_files = os.listdir(path)
    if len(list_files) == 0:
        return True
    else:
        return False


def _remove_file_or_directories(path):
    """Return the file/directory or subdirectories tree of 'path'.

    Use this function with caution.
    """
    # If file
    if os.path.isfile(path):
        os.remove(path)
        logger.info(f"Deleted the file {path}")
    # If empty directory
    elif is_empty_directory(path):
        os.rmdir(path)
        logger.info(f"Deleted the empty directory {path}")
    # If not empty directory
    else:
        shutil.rmtree(path)
        logger.info(f"Deleted directories within {path}")
    return None


def remove_if_exists(path: str, force: bool = False) -> None:
    """Remove file or directory if exists and force=True.

    If force=False --> Raise error
    """
    # If the path does not exist, do nothing
    if not os.path.exists(path):
        return None

    # If the path exists and force=False, raise Error
    if not force:
        msg = f"--force is False and a file already exists at: {path}"
        logger.error(msg)
        raise ValueError(msg)

    # If force=True, remove the file/directory or subdirectories and files !
    try:
        _remove_file_or_directories(path)
    except Exception as e:
        msg = f"Can not delete file(s) at {path}. The error is: {e}"
        logger.error(msg)
        raise ValueError(msg)


def copy_file(src_fpath, dst_fpath):
    """Copy a file from a location to another."""
    filename = os.path.basename(src_fpath)
    dst_dir = os.path.dirname(dst_fpath)
    try:
        shutil.copy(src_fpath, dst_fpath)
        msg = f"{filename} copied at {dst_fpath}."
        logger.info(msg)
    except Exception as e:
        msg = f"Something went wrong when copying {filename} into {dst_dir}.\n The error is: {e}."
        logger.error(msg)
        raise ValueError(msg)


def remove_path_trailing_slash(path: str) -> str:
    """
    Removes a trailing slash or backslash from a file path if it exists.

    This function ensures that the provided file path is normalized by removing
    any trailing directory separator characters ('/' or '\\'). This is useful for
    maintaining consistency in path strings and for preparing paths for operations
    that may not expect a trailing slash.

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
