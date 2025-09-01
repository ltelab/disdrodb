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
"""Test manipulation utilities."""
import numpy as np
import xarray as xr

import disdrodb
from disdrodb.tests.fake_datasets import create_template_l2e_dataset
from disdrodb.utils.manipulations import (
    convert_from_decibel,
    convert_to_decibel,
    get_diameter_bin_edges,
    resample_drop_number_concentration,
    unstack_radar_variables,
)


def test_get_diameter_bin_edges():
    """It retrieves correct diameter bin edges from dataset."""
    ds = create_template_l2e_dataset()
    edges = get_diameter_bin_edges(ds)
    expected = np.append(
        ds["diameter_bin_lower"].values,
        ds["diameter_bin_upper"].values[-1],
    )
    np.testing.assert_array_equal(edges, expected)
    np.testing.assert_array_equal(edges, ds.disdrodb.diameter_bin_edges)  # disdrodb accessor


def test_convert_from_and_to_decibel_inverse():
    """It ensures convert_from_decibel and convert_to_decibel are inverses."""
    values_db = np.array([0.0, 10.0, 20.0])
    values_lin = convert_from_decibel(values_db)
    back_to_db = convert_to_decibel(values_lin)

    np.testing.assert_allclose(values_lin, np.array([1.0, 10.0, 100.0]))
    np.testing.assert_allclose(back_to_db, values_db, rtol=1e-10)
    np.testing.assert_allclose(disdrodb.idecibel(values_db), values_lin, rtol=1e-10)
    np.testing.assert_allclose(disdrodb.decibel(values_lin), values_db, rtol=1e-10)


def test_unstack_radar_variables():
    """It unstacks radar variables and removes frequency dimension."""
    # Build dataset with radar variable and frequency dimension
    ds = create_template_l2e_dataset()
    ds["DBZH"] = xr.ones_like(ds["drop_number_concentration"]).expand_dims({"frequency": [1, 2]})
    ds_unstacked = unstack_radar_variables(ds)

    # Check unstacked dataset
    assert "DBZH" not in ds_unstacked
    assert "frequency" not in ds_unstacked.dims

    assert any(var.startswith("DBZH") for var in ds_unstacked.data_vars), "Expect new variables named DBZH_<freq>"


def test_resample_drop_number_concentration_linear():
    """It resamples drop_number_concentration onto higher resolution bins."""
    ds = create_template_l2e_dataset()

    new_edges = np.linspace(0, 10, 50)
    da_resampled = resample_drop_number_concentration(
        ds["drop_number_concentration"],
        diameter_bin_edges=new_edges,
        method="linear",
    )

    # New coordinates should exist
    for coord in ["diameter_bin_width", "diameter_bin_lower", "diameter_bin_upper"]:
        assert coord in da_resampled.coords

    # Shape consistency: new bin centers = len(edges)-1
    assert da_resampled.sizes["diameter_bin_center"] == len(new_edges) - 1
