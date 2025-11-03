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
from disdrodb.constants import DIAMETER_DIMENSION, VELOCITY_DIMENSION
from disdrodb.tests.fake_datasets import create_template_l2e_dataset
from disdrodb.utils.manipulations import (
    convert_from_decibel,
    convert_to_decibel,
    filter_diameter_bins,
    filter_velocity_bins,
    get_diameter_bin_edges,
    get_diameter_coords_dict_from_bin_edges,
    resample_drop_number_concentration,
    unstack_radar_variables,
)


def create_test_dataset():
    """Create a small synthetic dataset with diameter and velocity bins."""
    ds = xr.Dataset(
        {
            "diameter_bin_lower": ("diameter", [0.5, 1.0, 2.0, 3.0]),
            "diameter_bin_upper": ("diameter", [1.0, 2.0, 3.0, 4.0]),
            "velocity_bin_lower": ("velocity", [0.0, 2.0, 5.0, 10.0]),
            "velocity_bin_upper": ("velocity", [2.0, 5.0, 10.0, 15.0]),
            "drop_number": (("diameter", "velocity"), np.ones((4, 4))),
        },
    )
    ds = ds.rename({"diameter": DIAMETER_DIMENSION, "velocity": VELOCITY_DIMENSION})
    ds = ds.set_coords(["diameter_bin_lower", "diameter_bin_upper", "velocity_bin_lower", "velocity_bin_upper"])
    return ds


class TestFilterDiameterBins:
    """Test units for filter_diameter_bins."""

    def test_filter_diameter_bins_with_bounds(self):
        """Test filter_diameter_bins keeps bins overlapping specified min/max diameters."""
        ds = create_test_dataset()
        ds_filtered = filter_diameter_bins(ds, minimum_diameter=1.5, maximum_diameter=3.5)
        assert np.all(ds_filtered["diameter_bin_lower"].to_numpy() >= 1.0)
        assert np.all(ds_filtered["diameter_bin_upper"].to_numpy() <= 4.0)

    def test_filter_diameter_bins_boundaries_inclusion(self):
        """Test filter_diameter_bins excludes bins that only touch the min/max boundaries."""
        ds = create_test_dataset()
        ds_filtered = filter_diameter_bins(ds, minimum_diameter=1.0, maximum_diameter=3.0)
        np.testing.assert_allclose(ds_filtered["diameter_bin_lower"].to_numpy(), [1, 2])
        np.testing.assert_allclose(ds_filtered["diameter_bin_upper"].to_numpy(), [2, 3])

    def test_filter_diameter_bins_without_arguments(self):
        """Test filter_diameter_bins without arguments returns input dataset."""
        ds = create_test_dataset()
        ds_filtered = filter_diameter_bins(ds)
        assert ds_filtered.sizes[DIAMETER_DIMENSION] == ds.sizes[DIAMETER_DIMENSION]  # all bins kept

    def test_filter_diameter_bins_raise_error_if_filter_everything(self):
        """Test filter_diameter_bins raise error if filter everything."""
        ds = create_test_dataset()
        with pytest.raises(ValueError):
            filter_diameter_bins(ds, minimum_diameter=1000)


class TestFilterVelocityBins:
    """Test units for filter_velocity_bins."""

    def test_filter_velocity_bins_with_bounds(self):
        """Test filter_velocity_bins keeps bins overlapping specified min/max velocities."""
        ds = create_test_dataset()
        ds_filtered = filter_velocity_bins(ds, minimum_velocity=1, maximum_velocity=6)
        assert np.all(ds_filtered["velocity_bin_lower"].to_numpy() < 6)
        assert np.all(ds_filtered["velocity_bin_upper"].to_numpy() > 1)

    def test_filter_velocity_bins_boundaries_inclusion(self):
        """Test filter_velocity_bins excludes bins that only touch the min/max boundaries."""
        ds = create_test_dataset()
        ds_filtered = filter_velocity_bins(ds, minimum_velocity=2.0, maximum_velocity=10.0)
        np.testing.assert_allclose(ds_filtered["velocity_bin_lower"].to_numpy(), [2, 5])
        np.testing.assert_allclose(ds_filtered["velocity_bin_upper"].to_numpy(), [5, 10])

    def test_filter_velocity_bins_without_arguments(self):
        """Test filter_velocity_bins without arguments returns input dataset."""
        ds = create_test_dataset()
        ds_filtered = filter_velocity_bins(ds)
        assert ds_filtered.sizes[VELOCITY_DIMENSION] == ds.sizes[VELOCITY_DIMENSION]  # all bins kept

    def test_filter_velocity_bins_raise_error_if_filter_everything(self):
        """Test filter_velocity_bins raise error if filter everything."""
        ds = create_test_dataset()
        with pytest.raises(ValueError):
            filter_velocity_bins(ds, minimum_velocity=1000)


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
