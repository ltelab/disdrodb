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
"""Test code to infer altitude from lat/lon using OpenTopoData API."""
import numpy as np

from disdrodb.metadata.geolocation import infer_altitude, infer_altitudes


def test_infer_altitude():
    """Test infer_altitude."""
    # Over ocean return None
    altitude = infer_altitude(latitude=0, longitude=0, dem="aster30m")
    assert altitude is None

    # Over land
    altitude = infer_altitude(latitude=42, longitude=15, dem="aster30m")
    assert altitude == 8.0


def test_infer_altitudes():
    """Test infer_altitudes."""
    altitudes = infer_altitudes(lats=[42, 43], lons=[15, 16])
    np.testing.assert_allclose(altitudes, [8.0, 0.0])

    altitudes = infer_altitudes(lats=[42, 0], lons=[15, 0])
    np.testing.assert_allclose(altitudes, [8.0, np.nan])
