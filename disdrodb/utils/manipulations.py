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
"""Include functions helping for DISDRODB product manipulations."""

import numpy as np


def get_diameter_bin_edges(ds):
    """Retrieve diameter bin edges."""
    bin_edges = np.append(ds["diameter_bin_lower"].compute().data, ds["diameter_bin_upper"].compute().data[-1])
    return bin_edges


def convert_from_decibel(x):
    """Convert dB to unit."""
    return np.power(10.0, x / 10)


def convert_to_decibel(x):
    """Convert unit to dB."""
    return 10 * np.log10(x)
