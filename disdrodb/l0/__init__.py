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
"""DISDRODB L0 software."""
from disdrodb.l0.l0_reader import available_readers, get_reader, get_station_reader
from disdrodb.l0.l0a_processing import generate_l0a
from disdrodb.l0.l0b_nc_processing import generate_l0b_from_nc
from disdrodb.l0.l0b_processing import generate_l0b

__all__ = [
    "available_readers",
    "generate_l0a",
    "generate_l0b",
    "generate_l0b_from_nc",
    "get_reader",
    "get_station_reader",
]
