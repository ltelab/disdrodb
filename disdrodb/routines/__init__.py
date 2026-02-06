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
"""DISDRODB L0 software."""

from disdrodb.routines.options import get_model_options, get_product_options
from disdrodb.routines.wrappers import (
    create_summary,
    create_summary_station,
    run,
    run_l0,
    run_l0_station,
    run_l0a,
    run_l0a_station,
    run_l0b,
    run_l0b_station,
    run_l0c,
    run_l0c_station,
    run_l1,
    run_l1_station,
    run_l2e,
    run_l2e_station,
    run_l2m,
    run_l2m_station,
    run_station,
)

__all__ = [
    "create_summary",
    "create_summary_station",
    "get_model_options",
    "get_product_options",
    "run",
    "run_l0",
    "run_l0_station",
    "run_l0a",
    "run_l0a_station",
    "run_l0b",
    "run_l0b_station",
    "run_l0c",
    "run_l0c_station",
    "run_l1",
    "run_l1_station",
    "run_l2e",
    "run_l2e_station",
    "run_l2m",
    "run_l2m_station",
    "run_station",
]
