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
# along with this progra  If not, see <http://www.gnu.org/licenses/>.
# -----------------------------------------------------------------------------.
"""Testing ENV routines."""
import logging

import numpy as np
import xarray as xr

from disdrodb.constants import GEOLOCATION_COORDS
from disdrodb.l1_env.routines import (
    DEFAULT_GEOLOCATION,
    get_default_environment_dataset,
    load_env_dataset,
)


def test_get_default_environment_dataset_values():
    """Test get_default_environment_dataset returns expected constants with correct values."""
    ds = get_default_environment_dataset()
    assert np.isclose(ds["sea_level_air_pressure"].item(), 101325)
    assert np.isclose(ds["gas_constant_dry_air"].item(), 287.04)
    assert np.isclose(ds["lapse_rate"].item(), 0.0065)
    assert np.isclose(ds["relative_humidity"].item(), 0.95)
    assert np.isclose(ds["temperature"].item(), 288.15)  # 15°C
    assert np.isclose(ds["water_density"].item(), 1000)


class TestLoadEnvDataset:
    """Unit tests for ENV core functions."""

    def test_assign_geolocation_with_valid_scalar_coordinates(self):
        """Test _assign_geolocation keeps valid scalar coordinates unchanged."""
        ds = xr.Dataset({coord: xr.DataArray(0) for coord in GEOLOCATION_COORDS})
        ds_env = load_env_dataset(ds)
        for coord in GEOLOCATION_COORDS:
            assert np.isclose(ds_env[coord].item(), 0)

    def test_assign_geolocation_with_invalid_scalar_coordinates(self):
        """Test _assign_geolocation replaces invalid scalar coordinates with defaults."""
        ds = xr.Dataset({coord: xr.DataArray(-1000) for coord in GEOLOCATION_COORDS})
        ds_env = load_env_dataset(ds)
        for coord in GEOLOCATION_COORDS:
            assert np.isclose(ds_env[coord].item(), DEFAULT_GEOLOCATION[coord])

    def test_assign_geolocation_with_missing_coords(self, caplog):
        """Test _assign_geolocation assigns defaults when coordinates are missing and logs warnings."""
        ds = xr.Dataset()  # no coords

        logger = logging.getLogger("test_logger")
        with caplog.at_level(logging.WARNING, logger="test_logger"):
            ds_env = load_env_dataset(ds, logger=logger)

        for coord in GEOLOCATION_COORDS:
            assert np.isclose(ds_env[coord].item(), DEFAULT_GEOLOCATION[coord])
        assert any("not available" in rec.message for rec in caplog.records)

    def test_assign_geolocation_infill_missing_values_in_time_varying_coordinates(self):
        """Test _assign_geolocation fills NaN values over time with ffill/bfill."""
        time = np.arange(5)
        lat_values = [np.nan, 10, np.nan, np.nan, 20]
        ds = xr.Dataset(
            {
                "latitude": ("time", lat_values),
                "longitude": ("time", [5, np.nan, np.nan, 15, 20]),
                "altitude": ("time", [0, 1, np.nan, 2, 3]),
            },
            coords={"time": time},
        )

        ds_env = load_env_dataset(ds)

        # forward/backward filling should remove NaNs
        assert not np.isnan(ds_env["latitude"]).any()
        assert not np.isnan(ds_env["longitude"]).any()
        assert not np.isnan(ds_env["altitude"]).any()
        np.testing.assert_allclose(ds_env["latitude"].to_numpy(), np.array([10.0, 10.0, 10.0, 10.0, 20.0]))

    def test_load_env_dataset_defaults(self):
        """Test load_env_dataset without input dataset assigns default environment + geolocation."""
        ds_env = load_env_dataset()
        for coord in GEOLOCATION_COORDS:
            assert np.isclose(ds_env[coord].item(), DEFAULT_GEOLOCATION[coord])
        assert "sea_level_air_pressure" in ds_env
        assert "latitude" in ds_env
        assert "altitude" in ds_env

    def test_load_env_dataset_with_dataset(self):
        """Test load_env_dataset with input dataset assigns missing geolocation."""
        ds = xr.Dataset({"latitude": xr.DataArray(50)})
        ds_env = load_env_dataset(ds=ds)
        assert np.isclose(ds_env["latitude"].item(), 50)
        assert "sea_level_air_pressure" in ds_env
        assert "altitude" in ds_env
