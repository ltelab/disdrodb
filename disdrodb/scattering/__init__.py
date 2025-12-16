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
"""Implement PSD scattering routines."""

from disdrodb.scattering.axis_ratio import available_axis_ratio_models, get_axis_ratio_model
from disdrodb.scattering.permittivity import available_permittivity_models, get_refractive_index
from disdrodb.scattering.routines import (
    RADAR_OPTIONS,
    RADAR_VARIABLES,
    available_radar_bands,
    get_radar_parameters,
    load_scatterer,
)

__all__ = [
    "RADAR_OPTIONS",
    "RADAR_VARIABLES",
    "available_axis_ratio_models",
    "available_permittivity_models",
    "available_radar_bands",
    "get_axis_ratio_model",
    "get_radar_parameters",
    "get_refractive_index",
    "load_scatterer",
]
