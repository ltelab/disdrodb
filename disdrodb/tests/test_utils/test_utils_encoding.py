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
"""Test DISDRODB netCDF4 encoding utilities."""
import pytest
import numpy as np
import xarray as xr
from disdrodb.utils.encoding import rechunk_dataset, sanitize_encodings_dict, get_time_encoding


def test_rechunk_dataset():
    # Create a sample xarray dataset
    data = {
        "a": (["x", "y"], [[1, 2, 3], [4, 5, 6]]),
        "b": (["x", "y"], [[7, 8, 9], [10, 11, 12]]),
    }
    coords = {"x": [0, 1], "y": [0, 1, 2]}
    ds = xr.Dataset(data, coords=coords)

    # Define the encoding dictionary
    encoding_dict = {"a": {"chunksizes": (1, 2)}, "b": {"chunksizes": (2, 1)}}

    # Test the rechunk_dataset function
    ds_rechunked = rechunk_dataset(ds, encoding_dict)
    assert ds_rechunked["a"].chunks == ((1, 1), (2, 1))
    assert ds_rechunked["b"].chunks == ((2,), (1, 1, 1))
    
    

@pytest.fixture()
def encoding_dict_1():
    # create a test encoding dictionary
    return {
        "var1": {"dtype": "float32", "chunksizes": (10, 10, 10)},
        "var2": {"dtype": "int16", "chunksizes": (5, 5, 5)},
        "var3": {"dtype": "float64", "chunksizes": (100, 100, 100)},
    }


@pytest.fixture()
def encoding_dict_2():
    # create a test encoding dictionary
    return {
        "var1": {"dtype": "float32", "chunksizes": (100, 100, 100)},
        "var2": {"dtype": "int16", "chunksizes": (100, 100, 100)},
        "var3": {"dtype": "float64", "chunksizes": (100, 100, 100)},
    }


@pytest.fixture()
def ds():
    # create a test xr.Dataset
    data = {
        "var1": (["time", "x", "y"], np.random.random((10, 20, 30))),
        "var2": (["time", "x", "y"], np.random.randint(0, 10, size=(10, 20, 30))),
        "var3": (["time", "x", "y"], np.random.random((10, 20, 30))),
    }
    coords = {"time": np.arange(10), "x": np.arange(20), "y": np.arange(30)}
    return xr.Dataset(data, coords)


def test_sanitize_encodings_dict(encoding_dict_1, encoding_dict_2, ds):
    result = sanitize_encodings_dict(encoding_dict_1, ds)

    assert isinstance(result, dict)

    # Test that the dictionary contains the same keys as the input dictionary
    assert set(result.keys()) == set(encoding_dict_1.keys())

    # Test that the chunk sizes in the returned dictionary are smaller than or equal to the corresponding array shapes
    # in the dataset
    for var in result:
        assert tuple(result[var]["chunksizes"]) <= ds[var].shape

    result = sanitize_encodings_dict(encoding_dict_2, ds)

    assert isinstance(result, dict)

    # Test that the dictionary contains the same keys as the input dictionary
    assert set(result.keys()) == set(encoding_dict_2.keys())

    # Test that the chunk sizes in the returned dictionary are smaller than or equal to the corresponding array shapes
    # in the dataset
    for var in result:
        assert tuple(result[var]["chunksizes"]) <= ds[var].shape


def test_get_time_encoding():
    assert isinstance(get_time_encoding(), dict)