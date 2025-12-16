# -----------------------------------------------------------------------------.
# Copyright (c) 2021-2026 DISDRODB developers
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
"""Test CLI utilities."""

import subprocess

import pytest

from disdrodb.utils.cli import (  # adjust import to your actual module
    execute_cmd,
    parse_archive_dir,
    parse_arg_to_list,
    parse_empty_string_and_none,
)


class TestExecuteCmd:
    def test_runs_successfully(self, capsys):
        """It runs a shell command and prints its output."""
        execute_cmd("echo hello")
        captured = capsys.readouterr()
        assert "hello" in captured.out

    def test_raises_called_process_error_when_failed(self):
        """It raises CalledProcessError if command fails and raise_error=True."""
        with pytest.raises(subprocess.CalledProcessError):
            execute_cmd("exit 1", raise_error=True)

    def test_no_raise_when_failed_if_raise_error_false(self):
        """It does not raise if command fails and raise_error=False."""
        # Should not raise, even if command fails
        execute_cmd("exit 1", raise_error=False)


class TestParseEmptyStringAndNone:
    def test_returns_none_for_empty_or_None_strings(self):
        """It returns None for '' and 'None' strings."""
        assert parse_empty_string_and_none("") is None
        assert parse_empty_string_and_none("None") is None

    def test_returns_none_for_python_none(self):
        """It returns None if input is already None."""
        assert parse_empty_string_and_none(None) is None

    def test_returns_value_for_other_strings(self):
        """It returns the string unchanged for other values."""
        assert parse_empty_string_and_none("abc") == "abc"
        assert parse_empty_string_and_none("123") == "123"


class TestParseArgToList:
    def test_returns_none_for_empty_and_None(self):
        """It returns None for '' and 'None'."""
        assert parse_arg_to_list("") is None
        assert parse_arg_to_list("None") is None

    def test_returns_list_for_single_value(self):
        """It wraps a single string in a list."""
        assert parse_arg_to_list("var") == ["var"]

    def test_returns_list_for_multiple_values(self):
        """It splits a space-separated string into a list."""
        assert parse_arg_to_list("a b c") == ["a", "b", "c"]

    def test_removes_extra_spaces(self):
        """It removes empty entries caused by multiple spaces."""
        assert parse_arg_to_list("a   b") == ["a", "b"]

    def test_accepts_list_as_input(self):
        """It leaves lists unchanged."""
        inp = ["a", "b"]
        assert parse_arg_to_list(inp) == inp


class TestParseArchiveDir:
    def test_returns_none_for_empty_and_None(self):
        """It returns None for '' and 'None' strings."""
        assert parse_archive_dir("") is None
        assert parse_archive_dir("None") is None

    def test_returns_value_for_other_strings(self):
        """It returns the directory path unchanged for other strings."""
        assert parse_archive_dir("/path/to/data/archive") == "/path/to/data/archive"
