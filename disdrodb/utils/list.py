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
"""Utilities to work with lists."""


def flatten_list(nested_list):
    """Flatten a nested list into a single-level list."""
    if isinstance(nested_list, list) and len(nested_list) == 0:
        return nested_list
    # If list is already flat, return as is to avoid flattening to chars
    if isinstance(nested_list, list) and not isinstance(nested_list[0], list):
        return nested_list
    return [item for sublist in nested_list for item in sublist] if isinstance(nested_list, list) else [nested_list]
