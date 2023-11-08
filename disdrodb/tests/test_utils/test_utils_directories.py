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
"""Test directories utility."""

import os 
import pytest

# import platform

from disdrodb.utils.directories import (
    # ensure_string_path,
    check_directory_exist,
    create_directory,
    # create_required_directory,
    # copy_file,
    is_empty_directory,
    remove_if_exists,
    remove_path_trailing_slash,
)


def test_check_directory_exist(tmp_path):
    # Check when is a directory 
    assert check_directory_exist(tmp_path) is None
    
    # Check raise error when path is a file   
    file_path = tmp_path / "test_file.txt"
    file_path.write_text("This is a test file.")
    with pytest.raises(ValueError):
           check_directory_exist(file_path)
           
    # Check raise error when unexisting path
    with pytest.raises(ValueError):
           check_directory_exist("unexisting_path")


class TestIsEmptyDirectory:
    
    def test_non_existent_directory(self):
        with pytest.raises(OSError, match=r".* does not exist."):
            is_empty_directory("non_existent_directory")

    def test_file_path(self, tmp_path):
        # Create a temporary file
        file_path = tmp_path / "test_file.txt"
        file_path.write_text("This is a test file.")
        assert not is_empty_directory(str(file_path))

    def test_empty_directory(self, tmp_path):
        # `tmp_path` is a pytest fixture that provides a temporary directory unique to the test invocation
        assert is_empty_directory(tmp_path)

    def test_non_empty_directory(self, tmp_path):
        # Create a temporary file inside the temporary directory
        file_path = tmp_path / "test_file.txt"
        file_path.write_text("This is a test file.")
        assert not is_empty_directory(tmp_path)
        


def test_create_directory(tmp_path):
    temp_folder = os.path.join(tmp_path, "temp_folder")
    create_directory(temp_folder)
    if os.path.exists(temp_folder):
        res = True
    else:
        res = False
    assert res


# @pytest.mark.skipif(platform.system() == "Windows", reason="This test does not run on Windows")
def test_remove_if_exists_empty_directory(tmp_path):
    
    tmp_directory = os.path.join(tmp_path, "temp_folder")

    # Create empty folder if not exists
    if not os.path.exists(tmp_directory):
        create_directory(tmp_directory)

    # Check it raise an error if force=False
    with pytest.raises(ValueError): 
        remove_if_exists(tmp_directory, force=False)
    
    # Check it removes the folder
    remove_if_exists(tmp_directory, force=True)

    # Test the removal
    assert not os.path.exists(tmp_directory)


def test_remove_if_exists_file(tmp_path):
    
    file_path = tmp_path / "test_file.txt"
    file_path.write_text("This is a test file.")
        
    # Check it raise an error if force=False
    with pytest.raises(ValueError): 
        remove_if_exists(file_path, force=False)
    
    # Check it removes the folder
    remove_if_exists(file_path, force=True)

    # Test the removal
    assert not os.path.exists(file_path)


def test_remove_if_exists_with_shutil(tmp_path):    
    from pathlib import Path
    tmp_path = Path("/tmp/test211")
    tmp_path.mkdir()
    
    tmp_sub_directory = tmp_path / "subfolder"
    tmp_sub_directory.mkdir()
    tmp_filepath = tmp_sub_directory / "test_file.txt"
    tmp_filepath.write_text("This is a test file.")
    
    # Create empty folder if not exists
    if not os.path.exists(tmp_sub_directory):
        create_directory(tmp_sub_directory)
    
    # Check it raise an error if force=False
    with pytest.raises(ValueError): 
        remove_if_exists(tmp_path, force=False)
    
    # Check it removes the folder
    remove_if_exists(tmp_path, force=True)

    # Test the removal
    assert not os.path.exists(tmp_path)
    assert not os.path.exists(tmp_sub_directory)
    assert not os.path.exists(tmp_filepath)
    

def test_remove_path_trailing_slash():
    path_dir_windows_in = "\\DISDRODB\\Processed\\DATA_SOURCE\\CAMPAIGN_NAME\\"
    path_dir_windows_out = "\\DISDRODB\\Processed\\DATA_SOURCE\\CAMPAIGN_NAME"
    assert remove_path_trailing_slash(path_dir_windows_in) == path_dir_windows_out

    path_dir_linux_in = "/DISDRODB/Processed/DATA_SOURCE/CAMPAIGN_NAME/"
    path_dir_linux_out = "/DISDRODB/Processed/DATA_SOURCE/CAMPAIGN_NAME"
    assert remove_path_trailing_slash(path_dir_linux_in) == path_dir_linux_out