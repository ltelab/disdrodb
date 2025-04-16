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
"""Testing time utilities."""
import datetime

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from disdrodb.utils.time import get_dataframe_start_end_time, get_dataset_start_end_time


def test_get_dataframe_start_end_time():
    start_date = datetime.datetime(2019, 3, 26, 0, 0, 0)
    end_date = datetime.datetime(2021, 2, 8, 0, 0, 0)
    df = pd.DataFrame({"time": pd.date_range(start=start_date, end=end_date)})
    starting_time, ending_time = get_dataframe_start_end_time(df)
    assert isinstance(starting_time, pd.Timestamp)
    assert isinstance(ending_time, pd.Timestamp)
    assert starting_time == start_date
    assert ending_time == end_date


def test_get_dataset_start_end_time():
    times = pd.date_range("2023-01-01", periods=10, freq="D")
    data = np.random.rand(10, 2, 2)  # Random data for the sake of example
    ds = xr.Dataset({"my_data": (("time", "x", "y"), data)}, coords={"time": times})
    expected_start_time = ds["time"].to_numpy()[0]
    expected_end_time = ds["time"].to_numpy()[-1]

    starting_time, ending_time = get_dataset_start_end_time(ds)

    assert isinstance(starting_time, pd.Timestamp)
    assert isinstance(ending_time, pd.Timestamp)

    assert starting_time == expected_start_time
    assert ending_time == expected_end_time

    # Test raise if empty dataset
    empty_ds = xr.Dataset()
    with pytest.raises(KeyError):
        get_dataset_start_end_time(empty_ds)
