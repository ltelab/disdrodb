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
# along with this progra  If not, see <http://www.gnu.org/licenses/>.
# -----------------------------------------------------------------------------.
"""Testing resampling utilities."""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from disdrodb.constants import DIAMETER_DIMENSION
from disdrodb.l1.resampling import (
    add_sample_interval,
    define_window_size,
    resample_dataset,
)


class TestAddSampleInterval:
    def test_adds_sample_interval_coordinate(self):
        """Adds the sample_interval coordinate to the dataset."""
        ds = xr.Dataset()
        result = add_sample_interval(ds, 2)
        assert "sample_interval" in result.coords

    def test_sets_coordinate_values(self):
        """Sets the correct value for the sample_interval coordinate."""
        ds = xr.Dataset()
        result = add_sample_interval(ds, 7)
        assert result["sample_interval"].item() == 7

    def test_sample_interval_attributes(self):
        """Sets description, long_name, and units on the coordinate."""
        ds = xr.Dataset()
        result = add_sample_interval(ds, 3)
        var = result["sample_interval"]
        assert var.attrs["description"] == "Sample interval"
        assert var.attrs["long_name"] == "Sample interval"
        assert var.attrs["units"] == "seconds"

    def test_sets_as_coordinate(self):
        """Marks sample_interval as a coordinate, not just a variable."""
        ds = xr.Dataset()
        result = add_sample_interval(ds, 1)
        # xarray caches coords in .coords
        assert "sample_interval" in result.coords
        # and not in .data_vars
        assert "sample_interval" not in result.data_vars

    def test_updates_measurement_interval_attribute(self):
        """Updates the dataset's measurement_interval attribute."""
        ds = xr.Dataset(attrs={"measurement_interval": 60})
        result = add_sample_interval(ds, 120)
        assert result.attrs["measurement_interval"] == 120

    def test_float_sample_interval_casts_to_int(self):
        """Casts a float sample_interval to int for measurement_interval."""
        ds = xr.Dataset()
        result = add_sample_interval(ds, 60.0)
        assert isinstance(result.attrs["measurement_interval"], int)
        assert result.attrs["measurement_interval"] == int(60.0)

    def test_original_dataset_unmodified(self):
        """Does not add sample_interval to the original dataset object."""
        ds = xr.Dataset()
        ds_copy = ds.copy()
        _ = add_sample_interval(ds, 4)
        # original ds_copy still has no sample_interval
        assert "sample_interval" not in ds_copy.coords


class TestDefineWindowSize:
    def test_exact_multiple_returns_correct_size(self):
        """Returns correct window size when accumulation is exact multiple."""
        assert define_window_size(60, 300) == 5

    def test_different_intervals_return_correct_size(self):
        """Handles other valid sampling and accumulation intervals."""
        assert define_window_size(120, 600) == 5
        assert define_window_size(30, 90) == 3

    def test_accumulation_equals_sampling_returns_one(self):
        """Returns 1 when accumulation interval equals sampling interval."""
        assert define_window_size(10, 10) == 1

    def test_zero_accumulation_returns_zero(self):
        """Returns 0 when accumulation interval is zero."""
        assert define_window_size(5, 0) == 0

    def test_non_multiple_raises_value_error(self):
        """Raises ValueError if accumulation is not a multiple of sampling."""
        with pytest.raises(ValueError):
            define_window_size(60, 250)

    def test_zero_sampling_raises_zero_division_error(self):
        """Raises ZeroDivisionError when sampling interval is zero."""
        with pytest.raises(ZeroDivisionError):
            define_window_size(0, 100)

    def test_large_intervals(self):
        """Correctly computes window size for large interval values."""
        assert define_window_size(1, 10**6) == 10**6
        assert define_window_size(250, 250_000) == 1000


class TestResampleDataset:

    def test_case_sample_interval_equal_accumulation_interval(self):
        """Test it returns regularized input dataset with sample_interval coordinate and flags."""
        ds = xr.Dataset(coords={"time": pd.date_range("2000-01-01 00:00:00", periods=5, freq="1min")})
        out = resample_dataset(ds, sample_interval=60, temporal_resolution="ROLL1MIN")
        # Test sample_interval coordinate is added
        assert "sample_interval" in out.coords
        assert out["sample_interval"].item() == 60
        # Test flags set correctly
        assert out.attrs["disdrodb_rolled_product"] == "False"
        assert out.attrs["disdrodb_aggregated_product"] == "False"

    def test_case_when_accumulation_interval_not_multiple_sample_interval(self):
        """Test raises ValueError if accumulation_interval is not a multiple of sample_interval."""
        ds = xr.Dataset(coords={"time": pd.date_range("2000-01-01", periods=3, freq="2min")})
        with pytest.raises(ValueError):
            resample_dataset(ds, sample_interval=120, temporal_resolution="ROLL200S")

    def test_case_when_accumulation_interval_smaller_than_sample_interval(self):
        """Test raises ValueError if accumulation_interval is less than sample_interval."""
        ds = xr.Dataset(coords={"time": pd.date_range("2000-01-01", periods=3, freq="1min")})
        with pytest.raises(ValueError):
            resample_dataset(ds, sample_interval=60, temporal_resolution="30S")

    def test_non_rolling_resample_aggregates_correctly(self):
        """Test aggregates correctly when rolling=False."""
        # Create 1 minute timeseries
        times = pd.date_range("2000-01-01", periods=4, freq="1min").astype("datetime64[s]")
        ds = xr.Dataset(
            {
                "fall_velocity": ("time", [10, 20, 30, 40]),
                "drop_number": ("time", [1, 2, 3, 4]),
                "raw_drop_number": ("time", [2, 3, 4, 5]),
                "Dmin": ("time", [5, 6, 7, 8]),
                "Dmax": ("time", [8, 9, 10, 11]),
            },
            coords={"time": times},
        )
        # Resample to 2 minutes timeseries
        out = resample_dataset(ds, sample_interval=60, temporal_resolution="120S")

        # Check expected timesteps
        # 00:00:00 and 00:02:00 (start of accumulated intervals)
        expected_times = pd.to_datetime(["2000-01-01T00:00:00", "2000-01-01T00:02:00"])
        assert list(out["time"].data) == list(expected_times)

        # Check mean, sum, and custom var raw_drop_number
        np.testing.assert_allclose(out["fall_velocity"], [15, 35])
        np.testing.assert_allclose(out["drop_number"], [3, 7])
        np.testing.assert_allclose(out["raw_drop_number"], [5, 9])  # sum of two entries each
        np.testing.assert_allclose(out["Dmin"], [5, 7])
        np.testing.assert_allclose(out["Dmax"], [9, 11])

        # Check flags and coordinates
        assert out.attrs["disdrodb_rolled_product"] == "False"
        assert out.attrs["disdrodb_aggregated_product"] == "True"
        assert out.attrs["measurement_interval"] == 120
        assert "sample_interval" in out.coords
        assert out["sample_interval"].item() == 120

    def test_rolling_resample_aggregates_correctly(self):
        """Test aggregates correctly when rolling=True."""
        times = pd.date_range("2000-01-01", periods=4, freq="1min").astype("datetime64[s]")
        ds = xr.Dataset(
            {
                "fall_velocity": ("time", [10, 20, 30, 40]),
                "drop_number": ("time", [1, 1, 1, 1]),
                "raw_drop_number": ("time", [2, 3, 4, 5]),
            },
            coords={"time": times},
        )
        # Resample to 2 minutes timeseries
        out = resample_dataset(ds, sample_interval=60, temporal_resolution="ROLL120S")

        # Window size of 2
        # - Output time indicate start of the measurement interval !
        expected_times = pd.to_datetime(
            ["2000-01-01T00:00:00", "2000-01-01T00:01:00", "2000-01-01T00:02:00"],
        )
        assert list(out["time"].data) == list(expected_times)
        np.testing.assert_allclose(out["fall_velocity"], [15, 25, 35])
        np.testing.assert_allclose(out["drop_number"], [2, 2, 2])
        np.testing.assert_allclose(out["raw_drop_number"], [5, 7, 9])

        assert out.attrs["disdrodb_rolled_product"] == "True"
        assert out.attrs["disdrodb_aggregated_product"] == "True"
        assert out.attrs["measurement_interval"] == 120
        assert "sample_interval" in out.coords
        assert out["sample_interval"].item() == 120

    def test_resample_accessor_method(self):
        """Test disdrodb resample accessor method."""
        ds = xr.Dataset(coords={"time": pd.date_range("2000-01-01 00:00:00", periods=5, freq="1min")})
        ds = ds.assign_coords({"sample_interval": 60})
        out = ds.disdrodb.resample(temporal_resolution="ROLL2MIN")
        # Test sample_interval coordinate is added
        assert "sample_interval" in out.coords
        assert out["sample_interval"].item() == 120
        # Test flags set correctly
        assert out.attrs["disdrodb_rolled_product"] == "True"
        assert out.attrs["disdrodb_aggregated_product"] == "True"

    def test_qc_resampling_flag(self):
        """Test qc_resampling flag."""
        # Test when all timesteps are available and sample_interval=accumulation_interval
        ds = xr.Dataset(coords={"time": pd.date_range("2000-01-01 00:00:00", periods=5, freq="1min")})
        # - rolling=True
        ds_out = resample_dataset(ds, sample_interval=60, temporal_resolution="ROLL60S")
        assert np.all(ds_out["qc_resampling"].data == 0)
        # - rolling=False
        ds_out = resample_dataset(ds, sample_interval=60, temporal_resolution="60S")
        assert np.all(ds_out["qc_resampling"].data == 0)

        # Test when all timesteps are available
        ds = xr.Dataset(coords={"time": pd.date_range("2000-01-01 00:00:00", periods=6, freq="1min")})
        ds["drop_number"] = xr.ones_like(ds["time"], dtype=int)
        # - rolling=False
        ds_out = resample_dataset(ds, sample_interval=60, temporal_resolution="2MIN")
        assert np.all(ds_out["qc_resampling"].data == 0)
        assert ds_out.sizes["time"] == 3
        # - rolling=True
        ds_out = resample_dataset(ds, sample_interval=60, temporal_resolution="ROLL2MIN")
        assert np.all(ds_out["qc_resampling"].data == 0)
        assert ds_out.sizes["time"] == 5  #  6 -1

        # Test when some timesteps are not available
        ds = xr.Dataset(coords={"time": pd.date_range("2000-01-01 00:00:00", periods=6, freq="1min")})
        ds = ds.isel(time=[0, 4, 5])  # 1, 2, 3 missing

        # - rolling=True
        ds_out = resample_dataset(ds, sample_interval=60, temporal_resolution="ROLL2MIN")
        np.testing.assert_allclose(ds_out["qc_resampling"].data, [0.5, 1.0, 1.0, 0.5, 0.0])

        # - rolling=False
        ds_out = resample_dataset(ds, sample_interval=60, temporal_resolution="2MIN")
        np.testing.assert_allclose(ds_out["qc_resampling"].data, [0.5, 1.0, 0.0])

    def test_drop_number_with_nan(self):
        """Test spectrum with NaN are processed correctly."""
        # Times at 30-second intervals
        times = pd.date_range("2020-01-01", periods=8, freq="60s")

        # drop_number has NaNs in between
        drop_number = np.ones((8, 3))
        drop_number[2, 1] = np.nan  # idx 3 valid data
        # idx 4:5 missing
        drop_number[6:8, :] = np.nan  # idx 6:8 NaN

        ds = xr.Dataset(
            {
                "drop_number": (("time", DIAMETER_DIMENSION), drop_number),
            },
            coords={"time": times},
        )
        ds = ds.isel(time=[0, 1, 2, 3, 6, 7])  # Remove 4 and 5

        # Test case with no resampling (sample_interval == accumulation_interval)
        # - Expect return same dataset as input (no filtering)
        ds_out = resample_dataset(ds, sample_interval=60, temporal_resolution="1MIN")
        assert ds_out.sizes["time"] == 6
        np.testing.assert_allclose(ds_out["drop_number"].data, ds["drop_number"].data)

        # Test case with resampling
        ds_out = resample_dataset(ds, sample_interval=60, temporal_resolution="2MIN")
        assert ds_out.sizes["time"] == 4
        assert (ds_out["drop_number"].isel(time=0) == 2).all()

        assert (ds_out["drop_number"].isel(time=1) == 1).all()  # timestep with NaN is ignored in the sum
        assert not np.isnan(ds_out["drop_number"].isel(time=1)).any()

        assert np.isnan(ds_out["drop_number"].isel(time=2)).all()  # missing timesteps lead to NaN array

        assert np.isnan(ds_out["drop_number"].isel(time=3)).all()  # timesteps with only NaN lead to NaN array

    def test_raise_error_if_no_valid_timesteps(self):
        """Test raise error if all spectrums are NaN."""
        times = pd.date_range("2020-01-01", periods=3, freq="30s")

        # All drop_number values invalid
        ds = xr.Dataset(
            {
                "drop_number": ("time", [np.nan, np.nan, np.nan]),
            },
            coords={"time": times},
        )

        # Expect ValueError due to no valid timesteps
        with pytest.raises(ValueError, match="No timesteps with valid spectrum"):
            resample_dataset(ds, sample_interval=30, temporal_resolution="1MIN")
