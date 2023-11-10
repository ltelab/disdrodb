#!/usr/bin/env python3

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
"""Pandas utility."""

import pandas as pd


def get_dataframe_start_end_time(df: pd.DataFrame):
    """Retrieves dataframe starting and ending time.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe

    Returns
    -------
    tuple
        (starting_time, ending_time)

    """

    starting_time = df["time"].iloc[0]
    ending_time = df["time"].iloc[-1]
    return (starting_time, ending_time)
