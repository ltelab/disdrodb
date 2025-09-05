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
import pytest
import xarray as xr

import disdrodb
from disdrodb.constants import DIAMETER_DIMENSION
from disdrodb.tests.fake_datasets import create_template_l2e_dataset
from disdrodb.utils.manipulations import (
    convert_from_decibel,
    convert_to_decibel,
    get_diameter_bin_edges,
    get_diameter_coords_dict_from_bin_edges,
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


class TestGetDiameterCoordsDictFromBinEdges:
    """Test suite for get_diameter_coords_dict_from_bin_edges."""

    def test_simple_bin_edges(self):
        """Check correctness for a small set of bin edges."""
        edges = np.array([0.0, 1.0, 2.0])
        coords = get_diameter_coords_dict_from_bin_edges(edges)

        # Expected values
        expected_center = np.array([0.5, 1.5])
        expected_width = np.array([1.0, 1.0])
        expected_lower = np.array([0.0, 1.0])
        expected_upper = np.array([1.0, 2.0])

        np.testing.assert_array_equal(coords["diameter_bin_center"][1], expected_center)
        np.testing.assert_array_equal(coords["diameter_bin_width"][1], expected_width)
        np.testing.assert_array_equal(coords["diameter_bin_lower"][1], expected_lower)
        np.testing.assert_array_equal(coords["diameter_bin_upper"][1], expected_upper)

    def test_increasing_bin_edges_with_variable_widths(self):
        """Check correctness when bin widths are not uniform."""
        edges = np.array([0.0, 0.5, 2.0, 3.5])
        coords = get_diameter_coords_dict_from_bin_edges(edges)

        expected_center = np.array([0.25, 1.25, 2.75])
        expected_width = np.array([0.5, 1.5, 1.5])
        expected_lower = np.array([0.0, 0.5, 2.0])
        expected_upper = np.array([0.5, 2.0, 3.5])

        np.testing.assert_array_equal(coords["diameter_bin_center"][1], expected_center)
        np.testing.assert_array_equal(coords["diameter_bin_width"][1], expected_width)
        np.testing.assert_array_equal(coords["diameter_bin_lower"][1], expected_lower)
        np.testing.assert_array_equal(coords["diameter_bin_upper"][1], expected_upper)

    def test_minimum_two_edges(self):
        """Check that two edges produce a single bin."""
        edges = np.array([1.0, 2.0])
        coords = get_diameter_coords_dict_from_bin_edges(edges)

        np.testing.assert_array_equal(coords["diameter_bin_center"][1], [1.5])
        np.testing.assert_array_equal(coords["diameter_bin_width"][1], [1.0])
        np.testing.assert_array_equal(coords["diameter_bin_lower"][1], [1.0])
        np.testing.assert_array_equal(coords["diameter_bin_upper"][1], [2.0])

    def test_invalid_bin_edges_raises(self):
        """Check that fewer than two edges raises an error."""
        edges = np.array([1.0])  # not enough to form a bin
        with pytest.raises(ValueError):
            get_diameter_coords_dict_from_bin_edges(edges)

    def test_usable_as_xarray_coords(self):
        """Check that the dictionary can be passed directly to xarray.Dataset."""
        edges = np.array([0.0, 1.0, 2.0])
        coords = get_diameter_coords_dict_from_bin_edges(edges)

        ds = xr.Dataset(coords=coords)
        assert "diameter_bin_center" in ds.coords
        assert "diameter_bin_lower" in ds.coords
        assert DIAMETER_DIMENSION in ds.dims
        assert ds.sizes[DIAMETER_DIMENSION] == 2


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
