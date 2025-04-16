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
"""Test decorators."""
import pytest
from disdrodb.utils.decorators import (
    check_software_availability,
)


def test_check_software_availability_decorator():
    """Test check_software_availability_decorator raise ImportError."""

    @check_software_availability(software="dummy_package", conda_package="dummy_package")
    def dummy_function(a, b=1):
        return a, b

    with pytest.raises(ImportError):
        dummy_function()

    @check_software_availability(software="numpy", conda_package="numpy")
    def dummy_function(a, b=1):
        return a, b

    assert dummy_function(2, b=3) == (2, 3)