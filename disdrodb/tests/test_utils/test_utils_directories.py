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
import platform

import pytest
import pathlib
from disdrodb.utils.directories import (
    ensure_string_path,
    check_directory_exists,
    count_directories,
    count_files,
    create_directory,
    create_required_directory,
    copy_file,
    is_empty_directory,
    list_directories,
    list_files,
    remove_if_exists,
    remove_path_trailing_slash,
)


def test_list_count_files(tmp_path):
    # Set up test environment
    ext = "txt"
    dir1 = tmp_path / "dir1"
    dir1.mkdir()

    dir1_dummy = tmp_path / "dir1_dummy"
    dir1_dummy.mkdir()

    dir2 = dir1 / "dir2"
    dir2.mkdir()

    dir2_dummy = dir1 / "dir2_dummy"
    dir2_dummy.mkdir()

    file1 = tmp_path / f"file1.{ext}"
    file2 = tmp_path / f"file2.{ext}"
    file3 = tmp_path / "file3.ANOTHER"

    file4 = dir1 / f"file4.{ext}"
    file5 = dir1 / "file5.ANOTHER"

    file6 = dir2 / f"file6.{ext}"

    file1.touch()
    file2.touch()
    file3.touch()
    file4.touch()
    file5.touch()
    file6.touch()

    glob_pattern = "*"
    expected_files = [file1, file2, file3]
    assert set(list_files(tmp_path, glob_pattern, recursive=False)) == set(map(str, expected_files))

    glob_pattern = os.path.join("*", "*")
    expected_files = [file4, file5]
    assert set(list_files(tmp_path, glob_pattern, recursive=False)) == set(map(str, expected_files))
    assert count_files(tmp_path, glob_pattern, recursive=False) == len(expected_files)

    glob_pattern = f"*.{ext}"
    expected_files = [file1, file2]
    assert set(list_files(tmp_path, glob_pattern, recursive=False)) == set(map(str, expected_files))
    assert count_files(tmp_path, glob_pattern, recursive=False) == len(expected_files)

    glob_pattern = os.path.join("*", f"*.{ext}")
    expected_files = [file4]
    assert set(list_files(tmp_path, glob_pattern, recursive=False)) == set(map(str, expected_files))
    assert count_files(tmp_path, glob_pattern, recursive=False) == len(expected_files)

    glob_pattern = f"*.{ext}"
    expected_files = [file1, file2, file4, file6]
    assert set(list_files(tmp_path, glob_pattern, recursive=True)) == set(map(str, expected_files))
    assert count_files(tmp_path, glob_pattern, recursive=True) == len(expected_files)

    glob_pattern = os.path.join("*", f"*.{ext}")
    expected_files = [file4, file6]
    assert set(list_files(tmp_path, glob_pattern, recursive=True)) == set(map(str, expected_files))
    assert count_files(tmp_path, glob_pattern, recursive=True) == len(expected_files)


def test_list_and_count_directories(tmp_path):
    # Set up test environment
    dir1 = tmp_path / "dir1"
    dir1.mkdir()

    dir2 = dir1 / "dir2"
    dir2.mkdir()

    # Adding some files to check if they are excluded
    file1 = tmp_path / "file1.txt"
    file1.touch()
    file2 = dir1 / "file2.txt"
    file2.touch()

    # Non-recursive tests
    glob_pattern = "*"
    expected_dirs = [dir1]
    assert set(list_directories(tmp_path, glob_pattern, recursive=False)) == set(map(str, expected_dirs))
    assert count_directories(tmp_path, glob_pattern, recursive=False) == len(expected_dirs)

    glob_pattern = os.path.join("dir1", "*")
    expected_dirs = [dir2]
    assert set(list_directories(tmp_path, glob_pattern, recursive=False)) == set(map(str, expected_dirs))
    assert count_directories(tmp_path, glob_pattern, recursive=False) == len(expected_dirs)

    # Recursive tests
    glob_pattern = "*"
    expected_dirs = [dir1, dir2]
    assert set(list_directories(tmp_path, glob_pattern, recursive=True)) == set(map(str, expected_dirs))
    assert count_directories(tmp_path, glob_pattern, recursive=True) == len(expected_dirs)

    glob_pattern = os.path.join("**", "*")
    expected_dirs = [dir1, dir2]
    assert set(list_directories(tmp_path, glob_pattern, recursive=True)) == set(map(str, expected_dirs))
    assert count_directories(tmp_path, glob_pattern, recursive=True) == len(expected_dirs)


def test_check_directory_exists(tmp_path):
    # Check when is a directory
    assert check_directory_exists(tmp_path) is None

    # Check raise error when path is a file
    filepath = tmp_path / "test_file.txt"
    filepath.write_text("This is a test file.")
    with pytest.raises(ValueError):
        check_directory_exists(filepath)

    # Check raise error when unexisting path
    with pytest.raises(ValueError):
        check_directory_exists("unexisting_path")


class Test_Ensure_String_Path: 
    """Test ensure_string_path."""
    
    def test_ensure_string_path_valid_string(self):
        path = "some/path"
        assert ensure_string_path(path, "Invalid path") == path
    
    def test_ensure_string_path_valid_pathlib(self):
        path = pathlib.PurePath("some/path")
        assert ensure_string_path(path, "Invalid path", accepth_pathlib=True) == str(path)
    
    def test_ensure_string_path_invalid_pathlib(self):
        path = pathlib.PurePath("some/path")
        with pytest.raises(TypeError) as excinfo:
            ensure_string_path(path, "Invalid path", accepth_pathlib=False)
        assert "Invalid path" in str(excinfo.value)
    
    @pytest.mark.parametrize("invalid_input", [123, ["list"], {"dict": "value"}])
    def test_ensure_string_path_invalid_type(self, invalid_input):
        with pytest.raises(TypeError) as excinfo:
            ensure_string_path(invalid_input, "Invalid path")
        assert "Invalid path" in str(excinfo.value)
    
    def test_ensure_string_path_custom_error_message(self):
        path = 123
        custom_msg = "Path must be a string or PurePath"
        with pytest.raises(TypeError) as excinfo:
            ensure_string_path(path, custom_msg)
        assert custom_msg in str(excinfo.value)
        
    
class TestIsEmptyDirectory:
    def test_non_existent_directory(self):
        with pytest.raises(OSError, match=r".* does not exist."):
            is_empty_directory("non_existent_directory")

    def test_filepath(self, tmp_path):
        # Create a temporary file
        filepath = tmp_path / "test_file.txt"
        filepath.write_text("This is a test file.")
        assert not is_empty_directory(str(filepath))

    def test_empty_directory(self, tmp_path):
        # `tmp_path` is a pytest fixture that provides a temporary directory unique to the test invocation
        assert is_empty_directory(tmp_path)

    def test_non_empty_directory(self, tmp_path):
        # Create a temporary file inside the temporary directory
        filepath = tmp_path / "test_file.txt"
        filepath.write_text("This is a test file.")
        assert not is_empty_directory(tmp_path)


def test_create_directory(tmp_path, mocker):
    temp_dir = os.path.join(tmp_path, "temp_dir")
    create_directory(temp_dir)
    assert os.path.exists(temp_dir)
        
    # Test does not raise error if the directory already exists
    create_directory(temp_dir)
    assert os.path.exists(temp_dir)
        
    # Test capture possible error
    mocker.patch("os.makedirs", side_effect=Exception("Unexpected error"))
    with pytest.raises(FileNotFoundError):
        create_directory(temp_dir)


def test_create_required_directory(tmp_path, mocker):
    diirectory_name = "desired_directory_name"
    temp_dir = os.path.join(tmp_path, diirectory_name)
    create_required_directory(tmp_path, diirectory_name)
    assert os.path.exists(temp_dir)
 
    # Test does not raise error if the directory already exists
    create_required_directory(tmp_path, diirectory_name)
    assert os.path.exists(temp_dir)
        
    # Test capture possible error
    mocker.patch("os.makedirs", side_effect=Exception("Unexpected error"))
    with pytest.raises(FileNotFoundError):
        create_required_directory(tmp_path, diirectory_name)
        

class Test_Remove_If_Exists:
    """Test remove_if_exists."""
    
    def test_remove_if_exists_capture_errors(self, tmp_path, mocker):
        tmp_directory = os.path.join(tmp_path, "temp_dir")
        create_directory(tmp_directory)
    
        mocker.patch("disdrodb.utils.directories._remove_file_or_directories", side_effect=Exception("Unexpected error"))
        with pytest.raises(ValueError):
            remove_if_exists(tmp_directory, force=True)
            
    
    def test_remove_if_exists_empty_directory(self, tmp_path):
        tmp_directory = os.path.join(tmp_path, "temp_dir")
    
        # Create empty directory if not exists
        if not os.path.exists(tmp_directory):
            create_directory(tmp_directory)
    
        # Check it raise an error if force=False
        with pytest.raises(ValueError):
            remove_if_exists(tmp_directory, force=False)
    
        # Check it removes the directory
        remove_if_exists(tmp_directory, force=True)
    
        # Test the removal
        assert not os.path.exists(tmp_directory)
    
    
    def test_remove_if_exists_file(self, tmp_path):
        filepath = tmp_path / "test_file.txt"
        filepath.write_text("This is a test file.")
    
        # Check it raise an error if force=False
        with pytest.raises(ValueError):
            remove_if_exists(filepath, force=False)
    
        # Check it removes the directory
        remove_if_exists(filepath, force=True)
    
        # Test the removal
        assert not os.path.exists(filepath)
    
    
    @pytest.mark.skipif(platform.system() == "Windows", reason="This test does not run on Windows")
    def test_remove_if_exists_with_shutil(self, tmp_path):
        tmp_sub_directory = tmp_path / "subdirectory"
        tmp_sub_directory.mkdir()
        tmp_filepath = tmp_sub_directory / "test_file.txt"
        tmp_filepath.write_text("This is a test file.")
    
        # Create empty directory if not exists
        if not os.path.exists(tmp_sub_directory):
            create_directory(tmp_sub_directory)
    
        # Check it raise an error if force=False
        with pytest.raises(ValueError):
            remove_if_exists(tmp_path, force=False)
    
        # Check it removes the directory
        remove_if_exists(tmp_path, force=True)
    
        # Test the removal
        assert not os.path.exists(tmp_path)
        assert not os.path.exists(tmp_sub_directory)
        assert not os.path.exists(tmp_filepath)


def test_remove_path_trailing_slash():
    path_windows_in = "\\DISDRODB\\Processed\\DATA_SOURCE\\CAMPAIGN_NAME\\"
    path_windows_out = "\\DISDRODB\\Processed\\DATA_SOURCE\\CAMPAIGN_NAME"
    assert remove_path_trailing_slash(path_windows_in) == path_windows_out

    path_linux_in = "/DISDRODB/Processed/DATA_SOURCE/CAMPAIGN_NAME/"
    path_linux_out = "/DISDRODB/Processed/DATA_SOURCE/CAMPAIGN_NAME"
    assert remove_path_trailing_slash(path_linux_in) == path_linux_out
    
class Test_Copy_File:
    """Test copy_file."""
    
    def test_copy_file_success(self, tmp_path, mocker):
        # Create source and destination files
        src_file = tmp_path / "src.txt"
        dst_file = tmp_path / "dst.txt"
        src_file.write_text("test content")
    
        # Test copying file
        copy_file(str(src_file), str(dst_file))
        assert dst_file.read_text() == "test content"
    
    def test_copy_file_nonexistent_source(self, mocker):
        with pytest.raises(ValueError):
            copy_file("nonexistent_file.txt", "destination.txt")

    def test_copy_file_invalid_destination(self, tmp_path, mocker):
        src_file = tmp_path / "source.txt"
        with open(src_file, "w") as f:
            f.write("test content")
    
        with pytest.raises(ValueError):
            copy_file(src_file, "/invalid/path/destination.txt")
