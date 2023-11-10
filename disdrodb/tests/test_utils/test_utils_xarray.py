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
"""Test Xarray utility."""

import pytest
import xarray as xr
import numpy as np
import pandas as pd
from disdrodb.utils.xarray import get_dataset_start_end_time  


def create_test_dataset():
    """Create a mock xarray.Dataset for testing."""
    times = pd.date_range('2023-01-01', periods=10, freq='D')
    data = np.random.rand(10, 2, 2)  # Random data for the sake of example
    ds = xr.Dataset({'my_data': (('time', 'x', 'y'), data)},
                    coords={'time': times})
    return ds


def test_get_dataset_start_end_time():
    ds = create_test_dataset()
    expected_start_time = ds["time"].values[0]
    expected_end_time = ds["time"].values[-1]

    start_time, end_time = get_dataset_start_end_time(ds)

    assert start_time == expected_start_time
    assert end_time == expected_end_time
    
    # Test raise if empty dataset
    empty_ds = xr.Dataset()
    with pytest.raises(KeyError):
        get_dataset_start_end_time(empty_ds)
