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
"""DISDRODB physics wrapper functions."""
from disdrodb.physics.atmosphere import (
    get_air_pressure_at_height, 
    get_air_dynamic_viscosity, 
    get_vapor_actual_pressure, 
    get_air_density,
)

####---------------------------------------------------------------------------.
#### Wrappers
def retrieve_air_pressure(ds_env):
    """Retrieve air pressure."""
    if "air_pressure" in ds_env:
        return ds_env["air_pressure"]
    air_pressure = get_air_pressure_at_height(
        altitude=ds_env["altitude"],
        latitude=ds_env["latitude"],
        temperature=ds_env["temperature"],
        sea_level_air_pressure=ds_env["sea_level_air_pressure"],
        lapse_rate=ds_env["lapse_rate"],
    )
    return air_pressure


def retrieve_air_dynamic_viscosity(ds_env):
    """Retrieve air dynamic viscosity."""
    air_viscosity = get_air_dynamic_viscosity(ds_env["temperature"])
    return air_viscosity


def retrieve_air_density(ds_env):
    """Retrieve air density."""
    temperature = ds_env["temperature"]
    relative_humidity = ds_env["relative_humidity"]
    air_pressure = retrieve_air_pressure(ds_env)
    vapor_pressure = get_vapor_actual_pressure(
        relative_humidity=relative_humidity,
        temperature=temperature,
    )
    air_density = get_air_density(
        temperature=temperature,
        air_pressure=air_pressure,
        vapor_pressure=vapor_pressure,
    )
    return air_density
