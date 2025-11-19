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
import numpy as np
import xarray as xr

from disdrodb.constants import GEOLOCATION_COORDS
from disdrodb.l0.l0b_processing import ensure_valid_geolocation
from disdrodb.utils.logger import log_warning

DEFAULT_GEOLOCATION = {
    "latitude": 46.159346,
    "longitude": 8.774586,
    "altitude": 0,
}


def get_default_environment_dataset():
    """Set International Standard Atmosphere values for the default ENV dataset."""
    ds_env = xr.Dataset()
    ds_env["sea_level_air_pressure"] = 101_325  # Pa # sea level
    ds_env["gas_constant_dry_air"] = 287.04  # J kg⁻¹ K⁻¹
    ds_env["lapse_rate"] = 0.0065  # K m⁻¹  (6.5 deg/km)
    ds_env["relative_humidity"] = 0.95  # 0-1 !
    ds_env["temperature"] = 15 + 273.15  # K
    ds_env["water_density"] = 1000  # kg m⁻³   (T == 10 --> 999.7, T == 20 --> 998.2)
    # air density = 1.225 kg m⁻³ (if RH = 0) using retrieve_air_density(ds_env)
    return ds_env


def _assign_geolocation(ds_src, dst_dst, logger=None):
    dict_coords = {}
    for coord in GEOLOCATION_COORDS:
        if coord in ds_src:
            # Check geolocation validity
            ds_src = ensure_valid_geolocation(ds_src, coord=coord, errors="coerce")
            # Assign valid geolocation (or default one if invalid)
            if "time" not in ds_src[coord].dims:
                dict_coords[coord] = ds_src[coord] if not np.isnan(ds_src[coord]) else DEFAULT_GEOLOCATION[coord]
            else:  # If coordinates varies over time, infill NaN over time with forward and backward filling
                dict_coords[coord] = ds_src[coord].ffill(dim="time").bfill(dim="time")
        else:
            dict_coords[coord] = DEFAULT_GEOLOCATION[coord]
            log_warning(
                logger=logger,
                msg=f"{coord} not available. Setting {coord}={DEFAULT_GEOLOCATION[coord]}",
                verbose=False,
            )

    # Assign geolocation
    dst_dst = dst_dst.assign_coords(dict_coords)
    return dst_dst


def load_env_dataset(ds=None, logger=None):
    """Load the ENV dataset."""
    # TODO: Retrieve relative_humidity, lapse_rate and temperature from DISDRODB-ENV product

    # Load default environment dataset
    ds_env = get_default_environment_dataset()

    # Assign geolocation if input dataset provided
    if ds is not None:
        ds_env = _assign_geolocation(ds_src=ds, dst_dst=ds_env, logger=logger)
    # Otherwise add default geolocation
    else:
        ds_env = ds_env.assign_coords(DEFAULT_GEOLOCATION)
    return ds_env
