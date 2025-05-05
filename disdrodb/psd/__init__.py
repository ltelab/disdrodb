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
"""Implement PSD model and fitting routines."""


from disdrodb.psd.fitting import estimate_model_parameters
from disdrodb.psd.models import (
    ExponentialPSD,
    GammaPSD,
    LognormalPSD,
    NormalizedGammaPSD,
    available_psd_models,
    create_psd,
)

__all__ = [
    "ExponentialPSD",
    "GammaPSD",
    "LognormalPSD",
    "NormalizedGammaPSD",
    "available_psd_models",
    "create_psd",
    "estimate_model_parameters",
]
