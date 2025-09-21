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
"""Test DISDRODB product writer."""
import os

import numpy as np
import xarray as xr

from disdrodb.l0.l0b_processing import set_l0b_encodings
from disdrodb.tests.fake_datasets import create_template_dataset  #  create_template_l2e_dataset
from disdrodb.utils.writer import finalize_product, write_product


def test_finalize_product():
    """Test finalizing product do not raise error and add relevant attributes."""
    product = "L1"
    ds = create_template_dataset()

    ds = finalize_product(ds, product=product)
    assert isinstance(ds, xr.Dataset)
    assert ds.attrs["disdrodb_product"] == product
    assert "disdrodb_processing_date" in ds.attrs
    assert "disdrodb_product_version" in ds.attrs
    assert "disdrodb_software_version" in ds.attrs
    assert "Conventions" in ds.attrs
    assert "time_coverage_start" in ds.attrs
    assert "time_coverage_end" in ds.attrs


def test_write_product(tmp_path):
    """Test write DISDROB product."""
    ds = create_template_dataset()
    filepath = os.path.join(tmp_path, "test.nc")
    write_product(ds, filepath=filepath)

    ds_in = xr.open_dataset(filepath, decode_timedelta=False)
    xr.testing.assert_equal(ds, ds_in)


def test_correct_chunks_encoding(tmp_path):
    """Test DISDRODB correctly chunks the netCDF files."""
    # Create dataset with raw_drop_number variable with following dimensions
    # - (time: 10000, diameter_bin_center=2, velocity_bin_center=2)
    time = np.datetime64("2000-01-01T00:00") + np.arange(10_000) * np.timedelta64(
        1,
        "m",
    )  # larger than current chunksize
    velocity_bin_center = np.arange(2)
    diameter_bin_center = np.arange(2)

    # Create random data for drop_number
    drop_number_data = np.random.randint(0, 100, size=(len(time), len(velocity_bin_center), len(diameter_bin_center)))

    # Create dataset with drop_number
    ds = xr.Dataset(
        data_vars={
            "raw_drop_number": (("time", "diameter_bin_center", "velocity_bin_center"), drop_number_data),
        },
        coords={
            "time": time,
            "velocity_bin_center": velocity_bin_center,
            "diameter_bin_center": diameter_bin_center,
        },
    )
    # Apply encodings
    ds = set_l0b_encodings(ds=ds, sensor_name="PARSIVEL")

    # Check chunksizes is added by set_l0b_encodings
    assert "chunksizes" in ds["raw_drop_number"].encoding

    # Retrieve time chunksize
    chunksize = ds["raw_drop_number"].encoding["chunksizes"][0]

    # Save file by chunks
    filepath = os.path.join(tmp_path, "chunked.nc")
    write_product(ds, filepath)

    # Open file with file chunks using chunks={}
    ds_in = xr.open_dataset(filepath, chunks={}, decode_timedelta=False)

    # Check all time chunks correspond to what specified in encodings (except last)
    # - chunks format ((time_chunk, time_chunk), (...), (...))
    chunks = ds_in["raw_drop_number"].chunks
    assert np.all(np.array(chunks[0][:-1]) == chunksize)
