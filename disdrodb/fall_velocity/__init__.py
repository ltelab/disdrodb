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
"""DISDRODB fall velocity module."""

from disdrodb.fall_velocity.graupel import (
    available_graupel_fall_velocity_models,
    get_graupel_fall_velocity,
    get_graupel_fall_velocity_model,
)
from disdrodb.fall_velocity.hail import (
    available_hail_fall_velocity_models,
    get_hail_fall_velocity,
    get_hail_fall_velocity_model,
)
from disdrodb.fall_velocity.rain import (
    available_rain_fall_velocity_models,
    get_rain_fall_velocity,
    get_rain_fall_velocity_from_ds,
    get_rain_fall_velocity_model,
)

__all__ = [
    "available_graupel_fall_velocity_models",
    "available_hail_fall_velocity_models",
    "available_rain_fall_velocity_models",
    "get_graupel_fall_velocity",
    "get_graupel_fall_velocity_model",
    "get_hail_fall_velocity",
    "get_hail_fall_velocity_model",
    "get_rain_fall_velocity",
    "get_rain_fall_velocity_from_ds",
    "get_rain_fall_velocity_model",
]
