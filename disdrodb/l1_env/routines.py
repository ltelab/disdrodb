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
"""Core functions for DISDRODB ENV production."""

import xarray as xr


def get_default_environment_dataset():
    """Define defaults values for the ENV dataset."""
    ds_env = xr.Dataset()
    ds_env["sea_level_air_pressure"] = 101_325
    ds_env["gas_constant_dry_air"] = 287.04
    ds_env["lapse_rate"] = 0.0065
    ds_env["relative_humidity"] = 0.95  # Value between 0 and 1 !
    ds_env["temperature"] = 20 + 273.15
    return ds_env


def load_env_dataset(ds):
    """Load the ENV dataset."""
    # TODO - Retrieve relative_humidity and temperature from L1-ENV
    ds_env = get_default_environment_dataset()
    ds_env = ds_env.assign_coords({"altitude": ds["altitude"], "latitude": ds["latitude"]})
    return ds_env
