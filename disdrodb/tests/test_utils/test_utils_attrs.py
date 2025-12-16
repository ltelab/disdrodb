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
"""Test DISDRODB netCDF4 attributes utilities."""

import xarray as xr

from disdrodb.utils.attrs import set_attrs, set_coordinate_attributes


def test_set_attrs():
    ds = xr.Dataset({"var1": xr.DataArray([1, 2, 3], dims="time")})
    attrs_dict = {"var1": {"attr1": "value1"}}
    ds = set_attrs(ds, attrs_dict)
    assert ds["var1"].attrs["attr1"] == "value1"

    attrs_dict = {"var2": {"attr1": "value1"}}
    ds = set_attrs(ds, attrs_dict)
    assert "var2" not in ds

    attrs_dict = {"var1": {"attr1": "value1"}, "var2": {"attr2": "value2"}}
    ds = set_attrs(ds, attrs_dict)
    assert ds["var1"].attrs["attr1"] == "value1"
    assert "var2" not in ds


def test_set_coordinate_attributes():
    # Create example dataset
    ds = xr.Dataset(
        {
            "var1": xr.DataArray([1, 2, 3], dims="time"),
            "lat": xr.DataArray([0, 1, 2], dims="time"),
            "lon": xr.DataArray([0, 1, 2], dims="time"),
        },
    )
    ds.lat.attrs["units"] = "degrees_north"
    ds.lon.attrs["units"] = "degrees_east"

    # Call the function and check the output
    ds_out = set_coordinate_attributes(ds)
    assert "units" in ds_out["lat"].attrs
    assert ds_out["lat"].attrs["units"] == "degrees_north"
    assert "units" in ds_out["lon"].attrs
    assert ds_out["lon"].attrs["units"] == "degrees_east"
    assert "units" not in ds_out["var1"].attrs
