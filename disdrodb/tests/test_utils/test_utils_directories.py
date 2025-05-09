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
import pathlib
import platform

import pytest

from disdrodb.utils.directories import (
    check_directory_exists,
    check_glob_pattern,
    check_glob_patterns,
    contains_files,
    contains_netcdf_or_parquet_files,
    copy_file,
    count_directories,
    count_files,
    create_directory,
    create_required_directory,
    ensure_string_path,
    is_empty_directory,
    list_directories,
    list_files,
    list_paths,
    remove_if_exists,
    remove_path_trailing_slash,
)

# import pathlib
# tmp_path = pathlib.Path("/tmp/5")
# tmp_path.mkdir()


class TestContainsFiles:
    def test_empty_directory(self, tmp_path):
        """Should return False when directory contains no files."""
        root = tmp_path / "empty"
        root.mkdir()
        assert not contains_files(str(root))

    def test_top_level_file(self, tmp_path):
        """Should return True when there is a file at the top level."""
        root = tmp_path / "dir"
        root.mkdir()
        f = root / "file.txt"
        f.write_text("data")
        assert contains_files(str(root))

    def test_nested_file(self, tmp_path):
        """Should return True when files exist in nested subdirectories."""
        root = tmp_path / "dir"
        sub = root / "sub"
        sub.mkdir(parents=True)
        f = sub / "file.txt"
        f.write_text("data")
        assert contains_files(str(root))

    def test_only_empty_subdirs(self, tmp_path):
        """Should return False when only empty subdirectories are present."""
        root = tmp_path / "dir"
        sub = root / "sub"
        sub.mkdir(parents=True)
        assert not contains_files(str(root))


class TestContainsNetcdfOrParquetFiles:
    def test_empty_directory(self, tmp_path):
        """Should return False when directory contains no files."""
        empty = tmp_path / "empty"
        empty.mkdir()
        assert not contains_netcdf_or_parquet_files(str(empty))

    def test_only_other_files(self, tmp_path):
        """Should return False when files exist but none are .nc or .parquet."""
        root = tmp_path / "dir"
        root.mkdir()
        (root / "file.txt").write_text("")
        (root / "data.csv").write_text("")
        assert not contains_netcdf_or_parquet_files(str(root))

    def test_netcdf_file_top_level(self, tmp_path):
        """Should return True when a .nc file exists at the top level."""
        root = tmp_path / "dir"
        root.mkdir()
        (root / "sample.nc").write_text("")
        assert contains_netcdf_or_parquet_files(str(root))

    def test_parquet_file_top_level(self, tmp_path):
        """Should return True when a .parquet file exists at the top level."""
        root = tmp_path / "dir"
        root.mkdir()
        (root / "sample.parquet").write_text("")
        assert contains_netcdf_or_parquet_files(str(root))

    def test_netcdf_file_nested(self, tmp_path):
        """Should return True when a .nc file exists in a nested subdirectory."""
        root = tmp_path / "dir"
        sub = root / "sub"
        sub.mkdir(parents=True)
        (sub / "nested.nc").write_text("")
        assert contains_netcdf_or_parquet_files(str(root))

    def test_parquet_file_nested(self, tmp_path):
        """Should return True when a .parquet file exists in a nested subdirectory."""
        root = tmp_path / "dir"
        sub = root / "sub"
        sub.mkdir(parents=True)
        (sub / "nested.parquet").write_text("")
        assert contains_netcdf_or_parquet_files(str(root))


class TestCheckGlobPattern:
    def test_non_string_input(self):
        """Should raise TypeError when pattern is not a string."""
        with pytest.raises(TypeError, match="Expect pattern as a string."):
            check_glob_pattern(1)

    def test_pattern_starts_with_slash(self):
        """Should raise ValueError when pattern starts with a slash."""
        with pytest.raises(ValueError, match="glob_pattern should not start with /"):
            check_glob_pattern("/1")

    def test_duplicate_separators(self):
        """Should raise ValueError on duplicate path separators '//'."""
        with pytest.raises(ValueError, match="glob_pattern expects path with single separators: /, not //"):
            check_glob_pattern("path//with//duplicate//separators")

    def test_pattern_with_single_backslash(self):
        """Should raise ValueError when pattern uses single backslashes as separators."""
        with pytest.raises(ValueError, match="glob_pattern expects path separators to be /, not"):
            check_glob_pattern(r"path\window\style\*")

    def test_pattern_with_escaped_backslashes(self):
        """Should raise ValueError when pattern uses escaped backslashes as separators."""
        with pytest.raises(ValueError, match="glob_pattern expects path separators to be /, not "):
            check_glob_pattern(r"path\\window\\style\\*")

    def test_valid_pattern(self):
        """Should return the pattern unchanged when valid."""
        assert check_glob_pattern("*") == "*"


class TestCheckGlobPatterns:
    def test_non_list_or_string(self):
        """Should raise ValueError when patterns is neither str nor list."""
        with pytest.raises(ValueError, match="'glob_patterns' must be a str or list of strings."):
            check_glob_patterns(123)

    def test_single_string_input(self):
        """Should wrap single string into a list and validate its pattern."""
        assert check_glob_patterns("*") == ["*"]

    def test_list_of_strings(self):
        """Should return list of validated patterns when given a list of strings."""
        input_patterns = ["*", "data/*.csv"]
        assert check_glob_patterns(input_patterns) == ["*", "data/*.csv"]

    def test_list_with_invalid_pattern(self):
        """Should raise ValueError if any pattern in the list is invalid."""
        with pytest.raises(ValueError):
            check_glob_patterns(["valid", "/invalid"])


# import pathlib
# tmp_path = pathlib.Path("/tmp/8")
# tmp_path.mkdir()


class TestListPaths:

    def test_list_paths_non_recursive(self, tmp_path):
        """Should return file paths for single-string glob pattern."""
        f1 = tmp_path / "a.csv"
        f2 = tmp_path / "b.csv"
        f1.write_text("")
        f2.write_text("")
        # Files to ignore
        nested = tmp_path / "a"
        nested.mkdir(parents=True)
        f3 = nested / "c.csv"
        f3.write_text("")
        # Test results
        result = list_paths(tmp_path, glob_pattern="*.csv", recursive=False)
        assert set(result) == {str(f1), str(f2)}

    def test_list_paths_multiple_patterns(self, tmp_path):
        """Should return combined paths for multiple glob patterns."""
        f1 = tmp_path / "a.txt"
        f2 = tmp_path / "b.csv"
        f1.write_text("")
        f2.write_text("")
        result = list_paths(tmp_path, glob_pattern=["*.txt", "*.csv"], recursive=False)
        assert set(result) == {str(f1), str(f2)}

    def test_list_paths_recursive(self, tmp_path):
        """Should list matches in all subdirectories (when pattern has no slash)."""
        nested = tmp_path / "sub"
        nested.mkdir(parents=True)
        f1 = tmp_path / "file1.csv"
        f2 = nested / "file2.csv"
        f1.write_text("")
        f2.write_text("")
        # Test results
        result = list_paths(tmp_path, glob_pattern="*.csv", recursive=True)
        assert set(result) == {str(f1), str(f2)}

    def test_list_paths_disable_recursive_on_slash(self, tmp_path):
        """Should disable recursive search when pattern contains slash."""
        sub = tmp_path / "sub"
        sub.mkdir()
        f1 = sub / "file1.csv"
        f1.write_text("")
        # Files to ignore
        nested = sub / "nested"
        nested.mkdir()
        f2 = nested / "file2.csv"
        f2.write_text("")
        # Test results
        result = list_paths(tmp_path, glob_pattern="sub/*.csv", recursive=True)
        assert result == [str(f1)]

    @pytest.mark.parametrize("recursive", [True, False])
    def test_list_paths_single_level_pattern(self, tmp_path, recursive):
        """Should list files in multiple subdirectories with a single-level pattern.

        Recursive is set to False with such type of glob pattern.
        """
        sub1 = tmp_path / "sub1"
        sub2 = tmp_path / "sub2"
        sub1.mkdir()
        sub2.mkdir()
        f1 = sub1 / "file1.csv"
        f2 = sub2 / "file2.csv"
        f1.write_text("")
        f2.write_text("")
        # File to ignore
        nested = tmp_path / "a" / "b"
        nested.mkdir(parents=True)
        f3 = nested / "file3.csv"
        f3.write_text("")
        # Test results
        result = list_paths(tmp_path, "*/*.csv", recursive=recursive)
        assert set(result) == {str(f1), str(f2)}

    @pytest.mark.parametrize("recursive", [True, False])
    def test_list_paths_two_level_pattern(self, tmp_path, recursive):
        """Should list files in multiple subdirectories with two-level pattern.

        Recursive is set to False with such type of glob pattern.
        """
        sub1 = tmp_path / "sub1" / "a"
        sub2 = tmp_path / "sub2" / "b"
        sub1.mkdir(parents=True)
        sub2.mkdir(parents=True)
        f1 = sub1 / "file1.csv"
        f2 = sub2 / "file2.csv"
        f1.write_text("")
        f2.write_text("")
        # File to ignore
        nested = tmp_path / "sub1" / "c" / "other"
        nested.mkdir(parents=True)
        f3 = nested / "file3.csv"
        f3.write_text("")
        f4 = tmp_path / "file4.csv"
        f4.write_text("")
        # Test results
        result = list_paths(tmp_path, "*/*/*.csv", recursive=recursive)
        assert set(result) == {str(f1), str(f2)}

    def test_list_paths_in_nested_dirs_with_wildcard_pattern(self, tmp_path):
        """Should find files matching pattern recursively in all subdirectoriess with wildcards."""
        base_path = tmp_path / "data"
        nested = base_path / "a" / "b"
        nested.mkdir(parents=True)
        f1 = base_path / "file1.txt"
        f2 = nested / "file2.txt"
        f1.write_text("")
        f2.write_text("")
        result = list_paths(tmp_path, glob_pattern="data/**/*.txt", recursive=True)
        assert set(result) == {str(f1), str(f2)}


class TestListAndCountDirectories:
    def test_non_recursive_list_and_count(self, tmp_path):
        """Should list and count only top-level directories with a simple pattern."""
        dir1 = tmp_path / "dir1"
        dir1.mkdir()
        dir2 = dir1 / "dir2"
        dir2.mkdir()

        # Files to be ignored
        (tmp_path / "file1.txt").touch()
        (dir1 / "file2.txt").touch()

        # Test results
        dirs = list_directories(tmp_path, "*", recursive=False)
        count = count_directories(tmp_path, "*", recursive=False)
        assert set(dirs) == {str(dir1)}
        assert count == 1

        dirs = list_directories(tmp_path, "dir1/*", recursive=False)
        count = count_directories(tmp_path, "dir1/*", recursive=False)
        assert set(dirs) == {str(dir2)}
        assert count == 1

    def test_recursive_list_and_count(self, tmp_path):
        """Should list and count all directories recursively with wildcard patterns."""
        dir1 = tmp_path / "dir1"
        dir1.mkdir()
        dir2 = dir1 / "dir2"
        dir2.mkdir()

        # Files to be ignored
        (tmp_path / "file1.txt").touch()
        (dir1 / "file2.txt").touch()

        # Test results
        dirs = list_directories(tmp_path, "*", recursive=True)
        count = count_directories(tmp_path, "*", recursive=True)
        assert set(dirs) == {str(dir1), str(dir2)}
        assert count == 2

        dirs = list_directories(tmp_path, "**/*", recursive=True)
        count = count_directories(tmp_path, "**/*", recursive=True)
        assert set(dirs) == {str(dir1), str(dir2)}
        assert count == 2


# import pathlib
# tmp_path = pathlib.Path("/tmp/11")
# tmp_path.mkdir()


class TestListAndCountFiles:
    def test_list_top_level_all_files(self, tmp_path):
        """Should list all files at the top level with '*' pattern."""
        ext = "txt"
        # Setup dirs and files
        dir1 = tmp_path / "dir1"
        dir1.mkdir()
        dir1_dummy = tmp_path / "dir1_dummy"
        dir1_dummy.mkdir()
        file1 = tmp_path / f"file1.{ext}"
        file2 = tmp_path / f"file2.{ext}"
        file3 = tmp_path / "file3.ANOTHER"
        for f in (file1, file2, file3):
            f.touch()

        result = list_files(tmp_path, "*", recursive=False)
        assert set(result) == {str(file1), str(file2), str(file3)}
        assert count_files(tmp_path, "*", recursive=False) == 3

    def test_list_non_recursive_subdir_all(self, tmp_path):
        """Should list only files in a subdirectory with '*/*' pattern non-recursively."""
        ext = "txt"
        dir1 = tmp_path / "dir1"
        dir1.mkdir()
        dir2 = dir1 / "dir2"
        dir2.mkdir()
        file1 = dir1 / f"file1.{ext}"
        file2 = dir1 / "file2.ANOTHER"
        for f in (file1, file2):
            f.touch()

        result = list_files(tmp_path, "*/*", recursive=False)
        assert set(result) == {str(file1), str(file2)}
        assert count_files(tmp_path,"*/*", recursive=False) == 2

    def test_list_non_recursive_extension_filter(self, tmp_path):
        """Should list only files matching extension with '*.ext' pattern non-recursively."""
        ext = "txt"
        file1 = tmp_path / f"file1.{ext}"
        file2 = tmp_path / f"file2.{ext}"
        file3 = tmp_path / "file3.ANOTHER"
        for f in (file1, file2, file3):
            f.touch()

        result = list_files(tmp_path, f"*.{ext}", recursive=False)
        assert set(result) == {str(file1), str(file2)}
        assert count_files(tmp_path, f"*.{ext}", recursive=False) == 2

    def test_list_non_recursive_subdir_extension(self, tmp_path):
        """Should list only files matching extension in subdirectories non-recursively."""
        ext = "txt"
        dir1 = tmp_path / "dir1"
        dir1.mkdir()
        file1 = dir1 / f"file4.{ext}"
        file2 = dir1 / "file5.ANOTHER"
        file1.touch()
        file2.touch()

        pattern = f"*/*.{ext}"
        result = list_files(tmp_path, pattern, recursive=False)
        assert set(result) == {str(file1)}
        assert count_files(tmp_path, pattern, recursive=False) == 1

    def test_list_recursive_extension(self, tmp_path):
        """Should list files matching extension recursively across directories."""
        ext = "txt"
        dir1 = tmp_path / "dir1"
        dir1.mkdir()
        dir2 = dir1 / "dir2"
        dir2.mkdir()
        file1 = tmp_path / f"file1.{ext}"
        file2 = tmp_path / f"file2.{ext}"
        file3 = dir1 / f"file3.{ext}"
        file4 = dir2 / f"file4.{ext}"
        for f in (file1, file2, file3, file4):
            f.touch()

        result = list_files(tmp_path, f"*.{ext}", recursive=True)
        assert set(result) == {str(file1), str(file2), str(file3), str(file4)}
        assert count_files(tmp_path, f"*.{ext}", recursive=True) == 4

    def test_list_recursive_subdir_extension(self, tmp_path):
        """Should list files matching extension in subdirectories recursively."""
        ext = "txt"
        dir1 = tmp_path / "dir1"
        dir1.mkdir()
        dir2 = dir1 / "dir2"
        dir2.mkdir()
        file1 = dir1 / f"file1.{ext}"
        file2 = dir2 / f"file2.{ext}"
        file1.touch()
        file2.touch()

        pattern =f"**/*.{ext}"
        result = list_files(tmp_path, pattern, recursive=True)
        assert set(result) == {str(file1), str(file2)}
        assert count_files(tmp_path, pattern, recursive=True) == 2

        pattern = f"*/*.{ext}"
        result = list_files(tmp_path, pattern, recursive=True)
        assert set(result) == {str(file1)}
        assert count_files(tmp_path, pattern, recursive=True) == 1




class TestCheckDirectoryExists:
    def test_directory_path(self, tmp_path):
        """Should pass when path is an existing directory."""
        # No exception expected
        assert check_directory_exists(tmp_path) is None

    def test_file_path_raises(self, tmp_path):
        """Should raise ValueError when path is a file, not a directory."""
        filepath = tmp_path / "test_file.txt"
        filepath.write_text("This is a test file.")
        with pytest.raises(ValueError):
            check_directory_exists(filepath)

    def test_nonexistent_path_raises(self):
        """Should raise ValueError when path does not exist."""
        with pytest.raises(ValueError):
            check_directory_exists("unexisting_path")


class TestEnsureStringPath:
    def test_ensure_string_path_valid_string(self):
        """Should return the original string when a valid string path is provided."""
        path = "some/path"
        assert ensure_string_path(path, "Invalid path") == path

    def test_ensure_string_path_valid_pathlib(self):
        """Should convert a PurePath to string when accepth_pathlib=True."""
        path = pathlib.PurePath("some/path")
        assert ensure_string_path(path, "Invalid path", accepth_pathlib=True) == str(path)

    def test_ensure_string_path_invalid_pathlib(self):
        """Should raise TypeError for PurePath input when accepth_pathlib=False."""
        path = pathlib.PurePath("some/path")
        with pytest.raises(TypeError) as excinfo:
            ensure_string_path(path, "Invalid path", accepth_pathlib=False)
        assert "Invalid path" in str(excinfo.value)

    @pytest.mark.parametrize("invalid_input", [123, ["list"], {"dict": "value"}])
    def test_ensure_string_path_invalid_type(self, invalid_input):
        """Should raise TypeError for inputs that are not str or PurePath."""
        with pytest.raises(TypeError) as excinfo:
            ensure_string_path(invalid_input, "Invalid path")
        assert "Invalid path" in str(excinfo.value)

    def test_ensure_string_path_custom_error_message(self):
        """Should include custom message when raising TypeError on invalid input."""
        path = 123
        custom_msg = "Path must be a string or PurePath"
        with pytest.raises(TypeError) as excinfo:
            ensure_string_path(path, custom_msg)
        assert custom_msg in str(excinfo.value)


class TestCreateDirectory:
    def test_create_new_directory(self, tmp_path):
        """Should create a new directory when it does not exist."""
        temp_dir = os.path.join(tmp_path, "temp_dir")
        create_directory(temp_dir)
        assert os.path.exists(temp_dir)

    def test_existing_directory_no_error(self, tmp_path):
        """Should not raise an error if the directory already exists."""
        temp_dir = os.path.join(tmp_path, "temp_dir")
        create_directory(temp_dir)
        create_directory(temp_dir)
        assert os.path.exists(temp_dir)

    def test_capture_os_makedirs_error(self, tmp_path, mocker):
        """Should raise FileNotFoundError when os.makedirs fails unexpectedly."""
        temp_dir = os.path.join(tmp_path, "temp_dir")
        mocker.patch("os.makedirs", side_effect=Exception("Unexpected error"))
        with pytest.raises(FileNotFoundError):
            create_directory(temp_dir)


class TestCreateRequiredDirectory:
    def test_create_new_required_directory(self, tmp_path):
        """Should create the required directory within base path when it does not exist."""
        directory_name = "desired_directory_name"
        result_path = os.path.join(tmp_path, directory_name)
        create_required_directory(tmp_path, directory_name)
        assert os.path.exists(result_path)

    def test_existing_required_directory_no_error(self, tmp_path):
        """Should not raise an error if the required directory already exists."""
        directory_name = "desired_directory_name"
        result_path = os.path.join(tmp_path, directory_name)
        create_required_directory(tmp_path, directory_name)
        create_required_directory(tmp_path, directory_name)
        assert os.path.exists(result_path)

    def test_capture_os_makedirs_error(self, tmp_path, mocker):
        """Should raise FileNotFoundError when os.makedirs fails unexpectedly."""
        directory_name = "desired_directory_name"
        mocker.patch("os.makedirs", side_effect=Exception("Unexpected error"))
        with pytest.raises(FileNotFoundError):
            create_required_directory(tmp_path, directory_name)


class TestIsEmptyDirectory:
    def test_non_existent_directory(self):
        """Should raise OSError when directory does not exist."""
        with pytest.raises(OSError, match=r".* does not exist."):
            is_empty_directory("non_existent_directory")

    def test_filepath(self, tmp_path):
        """Should return False when path is a file, not a directory."""
        filepath = tmp_path / "test_file.txt"
        filepath.write_text("This is a test file.")
        assert not is_empty_directory(str(filepath))

    def test_empty_directory(self, tmp_path):
        """Should return True for an existing empty directory."""
        assert is_empty_directory(tmp_path)

    def test_non_empty_directory(self, tmp_path):
        """Should return False when directory contains files."""
        filepath = tmp_path / "test_file.txt"
        filepath.write_text("This is a test file.")
        assert not is_empty_directory(tmp_path)


class TestRemoveIfExists:
    def test_capture_errors(self, tmp_path, mocker):
        """Should wrap underlying removal errors into ValueError when force=True."""
        tmp_directory = os.path.join(tmp_path, "temp_dir")
        create_directory(tmp_directory)

        mocker.patch(
            "disdrodb.utils.directories._remove_file_or_directories",
            side_effect=Exception("Unexpected error"),
        )
        with pytest.raises(ValueError):
            remove_if_exists(tmp_directory, force=True)

    def test_inexisting_path(self, tmp_path):
        """Test do nothing when path does not exists."""
        remove_if_exists(tmp_path / "inexisting", force=True)

    def test_empty_directory_behaviour(self, tmp_path):
        """Should raise without force and remove directory when force=True."""
        tmp_directory = os.path.join(tmp_path, "temp_dir")
        create_directory(tmp_directory)

        with pytest.raises(ValueError):
            remove_if_exists(tmp_directory, force=False)

        remove_if_exists(tmp_directory, force=True)
        assert not os.path.exists(tmp_directory)

    def test_file_behaviour(self, tmp_path):
        """Should raise without force and remove a file when force=True."""
        filepath = tmp_path / "test_file.txt"
        filepath.write_text("This is a test file.")

        with pytest.raises(ValueError):
            remove_if_exists(str(filepath), force=False)

        remove_if_exists(str(filepath), force=True)
        assert not os.path.exists(filepath)

    @pytest.mark.skipif(platform.system() == "Windows", reason="This test does not run on Windows")
    def test_with_shutil(self, tmp_path):
        """Should remove nested directories and files recursively when force=True."""
        tmp_sub_directory = tmp_path / "subdirectory"
        tmp_sub_directory.mkdir()
        tmp_filepath = tmp_sub_directory / "test_file.txt"
        tmp_filepath.write_text("This is a test file.")

        with pytest.raises(ValueError):
            remove_if_exists(str(tmp_path), force=False)

        remove_if_exists(str(tmp_path), force=True)
        assert not os.path.exists(tmp_path)
        assert not os.path.exists(tmp_sub_directory)
        assert not os.path.exists(tmp_filepath)


class TestCopyFile:
    def test_copy_file_success(self, tmp_path):
        """Should copy contents from source to destination correctly."""
        src_file = tmp_path / "src.txt"
        dst_file = tmp_path / "dst.txt"
        src_file.write_text("test content")

        copy_file(str(src_file), str(dst_file))
        assert dst_file.read_text() == "test content"

    def test_copy_file_nonexistent_source(self):
        """Should raise ValueError when source file does not exist."""
        with pytest.raises(ValueError):
            copy_file("nonexistent_file.txt", "destination.txt")

    def test_copy_file_invalid_destination(self, tmp_path):
        """Should raise ValueError when destination path is invalid."""
        src_file = tmp_path / "source.txt"
        src_file.write_text("test content")

        invalid_dst = "/invalid/path/destination.txt"
        with pytest.raises(ValueError):
            copy_file(str(src_file), invalid_dst)


class TestRemovePathTrailingSlash:
    def test_windows_path(self):
        """Should strip trailing slash from a Windows path string."""
        path_windows_in = "\\DISDRODB\\ARCHIVE_VERSION\\DATA_SOURCE\\CAMPAIGN_NAME\\"
        path_windows_out = "\\DISDRODB\\ARCHIVE_VERSION\\DATA_SOURCE\\CAMPAIGN_NAME"
        assert remove_path_trailing_slash(path_windows_in) == path_windows_out

    def test_linux_path(self):
        """Should strip trailing slash from a Linux/Unix path string."""
        path_linux_in = "/DISDRODB/ARCHIVE_VERSION/DATA_SOURCE/CAMPAIGN_NAME/"
        path_linux_out = "/DISDRODB/ARCHIVE_VERSION/DATA_SOURCE/CAMPAIGN_NAME"
        assert remove_path_trailing_slash(path_linux_in) == path_linux_out
