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
"""Test manipulation utilities."""

import numpy as np
import pytest
import xarray as xr
from scipy.interpolate import PchipInterpolator

import disdrodb
from disdrodb.constants import DIAMETER_DIMENSION, VELOCITY_DIMENSION
from disdrodb.tests.fake_datasets import create_template_l2e_dataset
from disdrodb.utils.manipulations import (
    convert_from_decibel,
    convert_to_decibel,
    define_diameter_array,
    define_diameter_datarray,
    define_velocity_array,
    define_velocity_datarray,
    filter_diameter_bins,
    filter_velocity_bins,
    get_diameter_bin_edges,
    get_diameter_coords_dict_from_bin_edges,
    remap_to_diameter,
    resample_density,
    resample_drop_number_concentration,
    unstack_radar_variables,
)


def test_define_diameter_datarray_basic():
    """Test define_diameter_datarrayrray creation."""
    bounds = np.array([0, 1, 2])
    da = define_diameter_datarray(bounds)

    expected_centers = np.array([0.5, 1.5])
    expected_widths = np.array([1, 1])

    assert isinstance(da, xr.DataArray)
    assert da.dims == ("diameter_bin_center",)
    np.testing.assert_array_equal(da.values, expected_centers)
    np.testing.assert_array_equal(da.coords["diameter_bin_width"], expected_widths)
    np.testing.assert_array_equal(da.coords["diameter_bin_lower"], bounds[:-1])
    np.testing.assert_array_equal(da.coords["diameter_bin_upper"], bounds[1:])


def test_define_velocity_datarray():
    """Test define_velocity_datarray creation."""
    bounds = np.array([0, 2, 4])
    da = define_velocity_datarray(bounds)

    expected_centers = np.array([1, 3])
    expected_widths = np.array([2, 2])

    assert isinstance(da, xr.DataArray)
    assert da.dims == ("velocity_bin_center",)

    np.testing.assert_array_equal(da.values, expected_centers)
    np.testing.assert_array_equal(da.coords["velocity_bin_width"], expected_widths)
    np.testing.assert_array_equal(da.coords["velocity_bin_lower"], bounds[:-1])
    np.testing.assert_array_equal(da.coords["velocity_bin_upper"], bounds[1:])


class TestDefineDiameterArray:
    def test_define_diameter_array_defaults(self):
        """Test diameter array with default arguments."""
        da = define_diameter_array()

        # Expected number of bins: (10 - 0) / 0.05 = 200
        assert len(da) == 200
        assert np.isclose(da.values[0], 0.025)  # center of first bin
        assert np.isclose(da.values[-1], 9.975)  # center of last bin

    def test_define_diameter_array_custom(self):
        """Test diameter array with custom spacing."""
        da = define_diameter_array(0, 2, 1)
        expected_centers = np.array([0.5, 1.5])
        np.testing.assert_array_equal(da.values, expected_centers)


class TestDefineVelocityArray:
    def test_define_velocity_array_defaults(self):
        """Test velocity array with default arguments."""
        da = define_velocity_array()
        assert len(da) == 200
        assert np.isclose(da.values[0], 0.025)
        assert np.isclose(da.values[-1], 9.975)

    def test_define_velocity_array_custom(self):
        """Test velocity array with custom spacing."""
        da = define_velocity_array(0, 4, 2)
        expected_centers = np.array([1, 3])
        np.testing.assert_array_equal(da.values, expected_centers)


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


def test_resample_drop_number_concentration():
    """It resamples drop_number_concentration onto higher resolution bins."""
    ds = create_template_l2e_dataset()

    new_edges = np.linspace(0, 10, 50)
    da_resampled = resample_drop_number_concentration(
        ds["drop_number_concentration"],
        diameter_bin_edges=new_edges,
    )

    # New coordinates should exist
    for coord in ["diameter_bin_width", "diameter_bin_lower", "diameter_bin_upper"]:
        assert coord in da_resampled.coords

    # Shape consistency: new bin centers = len(edges)-1
    assert da_resampled.sizes["diameter_bin_center"] == len(new_edges) - 1

    # Assert that total drop number concentration is conserved (within numerical precision)
    Nt_original = (ds["drop_number_concentration"] * ds["diameter_bin_width"]).sum(dim="diameter_bin_center")
    Nt_resampled = (da_resampled * da_resampled.coords["diameter_bin_width"]).sum(dim="diameter_bin_center")
    np.testing.assert_allclose(Nt_original.to_numpy(), Nt_resampled.to_numpy(), rtol=1e-6)


def test_resample_density_log_pchip_conservative_irregular_bins():
    """It remaps irregular bins smoothly in log-space while conserving total number."""
    # Irregular source grid with contiguous edges
    d_src = xr.DataArray(
        [0.2, 0.55, 1.05, 1.9, 3.2, 4.8],
        dims=("diameter_bin_center",),
    )
    dD_src = xr.DataArray(
        [0.2, 0.5, 0.5, 1.2, 1.4, 1.8],
        dims=("diameter_bin_center",),
    )
    y_src = xr.DataArray(
        [3.0e4, 1.2e4, 4.0e3, 9.0e2, 1.5e2, 8.0],
        dims=("diameter_bin_center",),
        coords={"diameter_bin_center": d_src},
    )

    # Regular destination grid covering the same support
    d_dst_edges = np.arange(0.1, 5.7 + 0.1, 0.1)
    d_dst = define_diameter_datarray(d_dst_edges, dim="d_new")
    dD_dst = d_dst["diameter_bin_width"]

    da_constant = resample_density(
        da_density=y_src,
        d_src=d_src,
        d_dst=d_dst,
        dim="diameter_bin_center",
        new_dim="d_new",
        dD_src=dD_src,
        dD_dst=dD_dst,
        method="constant",
    )
    da_smooth = resample_density(
        da_density=y_src,
        d_src=d_src,
        d_dst=d_dst,
        dim="diameter_bin_center",
        new_dim="d_new",
        dD_src=dD_src,
        dD_dst=dD_dst,
        method="log_pchip",
    )

    nt_src = float((y_src * dD_src).sum())
    nt_smooth = float((da_smooth * dD_dst).sum())
    np.testing.assert_allclose(nt_smooth, nt_src, rtol=1e-12, atol=1e-12)
    assert np.all(np.isfinite(da_smooth))
    assert np.all(da_smooth >= 0)

    # The smooth method should not reproduce staircase-like constant segments.
    n_unique_constant = np.unique(np.round(da_constant.to_numpy(), decimals=8)).size
    n_unique_smooth = np.unique(np.round(da_smooth.to_numpy(), decimals=8)).size
    assert n_unique_smooth > n_unique_constant


def test_resample_density_raises_on_unknown_method():
    """It raises if an unsupported remapping method is requested."""
    da_density = xr.DataArray([10.0, 5.0], dims=("diameter_bin_center",))
    d_src = xr.DataArray([0.25, 0.75], dims=("diameter_bin_center",))
    dD_src = xr.DataArray([0.5, 0.5], dims=("diameter_bin_center",))
    d_dst = define_diameter_datarray(np.array([0.0, 0.5, 1.0]), dim="d_new")

    with pytest.raises(ValueError):
        resample_density(
            da_density=da_density,
            d_src=d_src,
            d_dst=d_dst,
            dim="diameter_bin_center",
            new_dim="d_new",
            dD_src=dD_src,
            dD_dst=d_dst["diameter_bin_width"],
            method="not_a_method",
        )


def test_resample_density_log_pchip_keeps_edge_half_bins():
    """It keeps non-zero values in destination bins overlapping source edge half-bins."""
    d_src = xr.DataArray([1.0, 2.0], dims=("diameter_bin_center",))
    dD_src = xr.DataArray([1.0, 1.0], dims=("diameter_bin_center",))
    y_src = xr.DataArray([100.0, 10.0], dims=("diameter_bin_center",), coords={"diameter_bin_center": d_src})

    # Destination bins overlap source support [0.5, 2.5] but have centers outside [1.0, 2.0] at the edges.
    d_dst = define_diameter_datarray(np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]), dim="d_new")
    dD_dst = d_dst["diameter_bin_width"]

    da_smooth = resample_density(
        da_density=y_src,
        d_src=d_src,
        d_dst=d_dst,
        dim="diameter_bin_center",
        new_dim="d_new",
        dD_src=dD_src,
        dD_dst=dD_dst,
        method="log_pchip",
    )

    dst_left = d_dst.to_numpy() - 0.5 * dD_dst.to_numpy()
    dst_right = d_dst.to_numpy() + 0.5 * dD_dst.to_numpy()
    overlap_support = (dst_right > 0.5) & (dst_left < 2.5)
    no_overlap_support = ~overlap_support

    assert np.all(da_smooth.to_numpy()[overlap_support] > 0)
    assert np.allclose(da_smooth.to_numpy()[no_overlap_support], 0)

    nt_src = float((y_src * dD_src).sum())
    nt_smooth = float((da_smooth * dD_dst).sum())
    np.testing.assert_allclose(nt_smooth, nt_src, rtol=1e-12, atol=1e-12)


def test_remap_to_diameter_pchip():
    """It remaps with shape-preserving cubic interpolation when method='pchip'."""
    d_src = xr.DataArray([0.0, 1.0, 2.0, 3.0], dims=("diameter_bin_center",))
    d_dst = xr.DataArray([0.5, 1.5, 2.5], dims=("D/Dc",))
    da = xr.DataArray([0.0, 1.0, 4.0, 9.0], dims=("diameter_bin_center",))

    da_out = remap_to_diameter(
        da=da,
        d_src=d_src,
        d_dst=d_dst,
        dim="diameter_bin_center",
        new_dim="D/Dc",
        method="pchip",
    )

    expected = PchipInterpolator(d_src.to_numpy(), da.to_numpy(), extrapolate=False)(d_dst.to_numpy())
    np.testing.assert_allclose(da_out.to_numpy(), expected, rtol=1e-12, atol=1e-12)


def test_remap_to_diameter_raises_on_unknown_method():
    """It raises if an unsupported interpolation method is requested."""
    d_src = xr.DataArray([0.0, 1.0], dims=("diameter_bin_center",))
    d_dst = xr.DataArray([0.5], dims=("D/Dc",))
    da = xr.DataArray([1.0, 2.0], dims=("diameter_bin_center",))

    with pytest.raises(ValueError):
        remap_to_diameter(
            da=da,
            d_src=d_src,
            d_dst=d_dst,
            dim="diameter_bin_center",
            new_dim="D/Dc",
            method="not_a_method",
        )
