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
"""Test DISDRODB coordinates utilities."""

import xarray as xr

from disdrodb.utils.coords import add_dataset_crs_coords


def test_add_dataset_crs_coords():
    """Test add_dataset_crs_coords."""
    # Create example dataset
    ds = xr.Dataset(
        {
            "var1": xr.DataArray([1, 2, 3], dims="time"),
            "lat": xr.DataArray([0, 1, 2], dims="time"),
            "lon": xr.DataArray([0, 1, 2], dims="time"),
        },
    )
    # Call the function and check the output
    ds_out = add_dataset_crs_coords(ds)
    assert "crs" in ds_out.coords
    assert ds_out["crs"].to_numpy() == "WGS84"
    # assert "crs_wkt" in ds_out["crs"].attrs
    # assert ds_out["crs"].attrs["epsg_code"] == "EPSG:4326"
