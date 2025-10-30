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
# along with this progra  If not, see <http://www.gnu.org/licenses/>.
# -----------------------------------------------------------------------------.
"""Testing filtering utilities."""
import numpy as np
import pytest
import xarray as xr

from disdrodb.constants import DIAMETER_DIMENSION, VELOCITY_DIMENSION
from disdrodb.l1.filters import (
    define_rain_spectrum_mask,
    filter_diameter_bins,
    filter_velocity_bins,
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


class TestDefineRainDropSpectrumMask:
    """Unit tests for diameter/velocity filters and spectrum mask definition."""

    def test_define_spectrum_mask_with_fraction(self):
        """Test spectrum mask creation with velocity fraction thresholds."""
        ds = create_test_dataset()
        fall_velocity = xr.DataArray([3.0, 6.0, 12.0, 20.0], dims=[DIAMETER_DIMENSION])
        ds_mask = define_rain_spectrum_mask(
            ds["drop_number"],
            fall_velocity,
            above_velocity_fraction=0.2,
            below_velocity_fraction=0.2,
        )
        assert isinstance(ds_mask, xr.DataArray)
        assert ds_mask.dtype == bool
        assert ds_mask.shape == ds["drop_number"].shape
        expected_mask = np.array(
            [
                [False, True, False, False],
                [False, True, True, False],
                [False, False, True, True],
                [False, False, False, False],
            ],
        )
        np.testing.assert_allclose(ds_mask.to_numpy(), expected_mask)

    def test_define_spectrum_mask_with_tolerance(self):
        """Test spectrum mask creation with velocity tolerance thresholds."""
        ds = create_test_dataset()
        fall_velocity = xr.DataArray([3.0, 6.0, 12.0, 20.0], dims=[DIAMETER_DIMENSION])
        ds_mask = define_rain_spectrum_mask(
            ds["drop_number"],
            fall_velocity,
            above_velocity_tolerance=1.0,
            below_velocity_tolerance=1.0,
        )
        assert isinstance(ds_mask, xr.DataArray)
        assert ds_mask.dtype == bool
        assert ds_mask.shape == ds["drop_number"].shape
        expected_mask = np.array(
            [
                [False, True, False, False],
                [False, False, True, False],
                [False, False, False, True],
                [False, False, False, False],
            ],
        )
        np.testing.assert_allclose(ds_mask.to_numpy(), expected_mask)

    def test_define_spectrum_without_arguments(self):
        """Test spectrum mask without arguments returns True array."""
        ds = create_test_dataset()
        fall_velocity = xr.DataArray([3.0, 6.0, 12.0, 20.0], dims=[DIAMETER_DIMENSION])
        ds_mask = define_rain_spectrum_mask(
            ds["drop_number"],
            fall_velocity,
        )
        assert isinstance(ds_mask, xr.DataArray)
        assert ds_mask.dtype == bool
        assert ds_mask.shape == ds["drop_number"].shape
        assert ds_mask.to_numpy().all()  # all True

    def test_define_rain_spectrum_mask_conflicting_args(self):
        """Test spectrum mask raises error if both fraction and tolerance are given."""
        ds = create_test_dataset()
        fall_velocity = xr.DataArray([3.0, 6.0, 12.0, 20.0], dims=[DIAMETER_DIMENSION])

        with pytest.raises(ValueError):
            define_rain_spectrum_mask(
                ds["drop_number"],
                fall_velocity,
                above_velocity_fraction=0.1,
                above_velocity_tolerance=1.0,
            )

        with pytest.raises(ValueError):
            define_rain_spectrum_mask(
                ds["drop_number"],
                fall_velocity,
                below_velocity_fraction=0.1,
                below_velocity_tolerance=1.0,
            )

    def test_define_rain_spectrum_mask_keep_smallest(self):
        """Test spectrum mask retains smallest drops when maintain_smallest_drops=True."""
        ds = create_test_dataset()
        fall_velocity = xr.DataArray([3.0, 6.0, 12.0, 20.0], dims=[DIAMETER_DIMENSION])
        ds_mask = define_rain_spectrum_mask(
            ds["drop_number"],
            fall_velocity,
            above_velocity_fraction=0.1,
            below_velocity_fraction=0.1,
            maintain_smallest_drops=True,
            small_diameter_threshold=1.5,
            small_velocity_threshold=3.0,
        )
        assert isinstance(ds_mask, xr.DataArray)
        assert ds_mask.dtype == bool
        assert ds_mask.shape == ds["drop_number"].shape
        assert ds_mask.to_numpy()[0, 0]  # because keeps drops with D<1.5 and V<3
        expected_mask = np.array(
            [
                [True, True, False, False],
                [False, False, True, False],
                [False, False, False, True],
                [False, False, False, False],
            ],
        )
        np.testing.assert_allclose(ds_mask.to_numpy(), expected_mask)

    def test_define_rain_spectrum_mask_with_fixed_fall_velocity(self):
        """Test spectrum mask works with fall_velocity not varying with time."""
        ds = create_test_dataset()
        ds = ds.expand_dims({"time": 3})

        drop_number = ds["drop_number"]

        fall_velocity = xr.DataArray([3.0, 6.0, 12.0, 20.0], dims=[DIAMETER_DIMENSION])

        ds_mask = define_rain_spectrum_mask(
            drop_number,
            fall_velocity,
            above_velocity_fraction=0.2,
            below_velocity_fraction=0.2,
        )
        assert isinstance(ds_mask, xr.DataArray)
        assert "time" not in ds_mask.dims

    def test_define_rain_spectrum_mask_with_fall_velocity_varying_with_time(self):
        """Test spectrum mask works with fall_velocity varying with time."""
        ds = create_test_dataset()
        ds = ds.expand_dims({"time": 3})

        drop_number = ds["drop_number"]

        fall_velocity = xr.DataArray([3.0, 6.0, 12.0, 20.0], dims=[DIAMETER_DIMENSION])
        fall_velocity = fall_velocity.expand_dims({"time": 3})

        ds_mask = define_rain_spectrum_mask(
            drop_number,
            fall_velocity,
            above_velocity_fraction=0.2,
            below_velocity_fraction=0.2,
        )
        assert isinstance(ds_mask, xr.DataArray)
        assert "time" in ds_mask.dims
        assert ds_mask.sizes["time"] == 3

    def test_define_rain_spectrum_mask_with_dask_array(self):
        """Test spectrum mask works correctly with xarray DataArray backed by dask array."""
        ds = create_test_dataset()
        ds = ds.expand_dims({"time": 3})
        fall_velocity = xr.DataArray([3.0, 6.0, 12.0, 20.0], dims=[DIAMETER_DIMENSION])
        fall_velocity = fall_velocity.expand_dims({"time": 3})

        drop_number = ds["drop_number"].chunk({"time": 1})
        fall_velocity = fall_velocity.chunk({"time": 1})

        ds_mask = define_rain_spectrum_mask(
            drop_number,
            fall_velocity,
            above_velocity_fraction=0.2,
            below_velocity_fraction=0.2,
            maintain_smallest_drops=True,
            small_diameter_threshold=1,  # 1.0 upper inclusive !
            small_velocity_threshold=2.0,  # 2.0 upper inclusive !
        )

        assert isinstance(ds_mask, xr.DataArray)
        assert hasattr(ds_mask, "chunks")

        expected_mask = np.array(
            [
                [True, True, False, False],
                [False, True, True, False],
                [False, False, True, True],
                [False, False, False, False],
            ],
        )
        np.testing.assert_allclose(ds_mask.isel(time=0).compute().to_numpy(), expected_mask)
