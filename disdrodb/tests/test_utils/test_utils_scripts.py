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
"""Test DISDRODB scripts utility."""

from disdrodb.utils.scripts import parse_arg_to_list


def test_parse_arg_to_list_empty_string():
    """Test parse_arg_to_list() with an empty string."""
    args = ""
    expected_output = None
    assert parse_arg_to_list(args) == expected_output


def test_parse_arg_to_list_single_variable():
    """Test parse_arg_to_list() with a single variable."""
    args = "variable"
    expected_output = ["variable"]
    assert parse_arg_to_list(args) == expected_output


def test_parse_arg_to_list_multiple_variables():
    """Test parse_arg_to_list() with multiple variables."""
    args = "variable1 variable2"
    expected_output = ["variable1", "variable2"]
    assert parse_arg_to_list(args) == expected_output


def test_parse_arg_to_list_extra_spaces():
    """Test parse_arg_to_list() with extra spaces between variables."""
    args = "  variable1    variable2  "
    expected_output = ["variable1", "variable2"]
    assert parse_arg_to_list(args) == expected_output


def test_parse_arg_to_list_none():
    """Test parse_arg_to_list() with None input."""
    args = None
    expected_output = None
    assert parse_arg_to_list(args) == expected_output

    args = "None"
    expected_output = None
    assert parse_arg_to_list(args) == expected_output


def test_parse_arg_to_list_other_types():
    """Test parse_arg_to_list() with other types of input."""
    args = 123
    expected_output = 123
    assert parse_arg_to_list(args) == expected_output


def test_parse_arg_to_list_empty_list():
    """Test parse_arg_to_list() with an empty list."""
    args = []
    expected_output = []
    assert parse_arg_to_list(args) == expected_output
