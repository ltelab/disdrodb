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

import xarray as xr

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
