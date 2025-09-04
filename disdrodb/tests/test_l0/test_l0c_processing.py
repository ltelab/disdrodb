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
"""Test DISDRODB L0C processing routines."""
import logging

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from disdrodb.l0.l0c_processing import (
    check_timesteps_regularity,
    create_l0c_datasets,
    drop_timesteps_with_invalid_sample_interval,
    get_problematic_timestep_indices,
    has_same_value_over_time,
    regularize_timesteps,
    remove_duplicated_timesteps,
    split_dataset_by_sampling_intervals,
)


class TestDropTimestepsWithInvalidSampleInterval:
    def test_no_invalid_intervals(self):
        """If all intervals are valid, dataset should remain unchanged."""
        times = pd.date_range("2023-01-01", periods=3, freq="1h")
        ds = xr.Dataset(
            {"sample_interval": ("time", [60, 60, 60])},
            coords={"time": times},
        )
        measurement_intervals = [60]

        ds_out = drop_timesteps_with_invalid_sample_interval(ds, measurement_intervals, verbose=False)

        xr.testing.assert_identical(ds_out, ds)

    def test_some_invalid_intervals(self):
        """Invalid intervals should be dropped and only valid remain."""
        times = pd.date_range("2023-01-01", periods=4, freq="1h")
        ds = xr.Dataset(
            {"sample_interval": ("time", [60, 59, 60, 62])},
            coords={"time": times},
        )
        measurement_intervals = [60]

        ds_out = drop_timesteps_with_invalid_sample_interval(ds, measurement_intervals, verbose=False)

        ds_expected = ds.isel(time=[0, 2])  # keep only valid
        xr.testing.assert_identical(ds_out, ds_expected)

    def test_all_invalid_intervals(self):
        """If all intervals invalid, return dataset with no time steps."""
        times = pd.date_range("2023-01-01", periods=3, freq="1h")
        ds = xr.Dataset(
            {"sample_interval": ("time", [59, 61, 62])},
            coords={"time": times},
        )
        measurement_intervals = [60]

        ds_out = drop_timesteps_with_invalid_sample_interval(ds, measurement_intervals, verbose=False)

        # Expect empty dataset along time
        assert ds_out.dims["time"] == 0

    def test_mixed_valid_intervals(self, capsys):
        """Invalid timesteps should trigger warnings if verbose=True."""
        times = pd.date_range("2023-01-01", periods=3, freq="1h")
        ds = xr.Dataset(
            {"sample_interval": ("time", [60, 120, 60])},
            coords={"time": times},
        )
        measurement_intervals = [60]

        ds_out = drop_timesteps_with_invalid_sample_interval(ds, measurement_intervals, verbose=True)

        # Only timesteps 0 and 2 remain
        assert np.all(ds_out["sample_interval"].to_numpy() == [60, 60])

        # Check warning was printed
        captured = capsys.readouterr()
        assert "Unexpected sampling interval (120 s)" in captured.out


class TestSplitDatasetBySamplingIntervals:

    def test_unsorted_time_is_sorted_first(self):
        """Unsorted dataset should be internally sorted by time before splitting."""
        times = pd.to_datetime(
            ["2023-01-01 00:02:00", "2023-01-01 00:00:00", "2023-01-01 00:01:00"],  # deliberately out of order
        )
        ds = xr.Dataset(coords={"time": times})

        result = split_dataset_by_sampling_intervals(ds, measurement_intervals=[60])
        # Keys should contain 60 only
        assert list(result.keys()) == [60]
        # Dataset should be sorted
        sorted_times = np.sort(times.values)
        np.testing.assert_array_equal(result[60].time.values, sorted_times)

    def test_trailing_seconds_case(self):
        """Intervals slightly off (e.g. 59, 61) should be rounded to expected ones."""
        times = pd.to_datetime(
            [
                "2023-01-01 00:00:00",
                "2023-01-01 00:00:59",  # ~60s
                "2023-01-01 00:02:01",  # ~62s
                "2023-01-01 00:03:00",
            ],
        )
        ds = xr.Dataset(coords={"time": times})

        result = split_dataset_by_sampling_intervals(ds, measurement_intervals=[60])
        assert 60 in result
        assert len(result[60].time) == 4

        result = split_dataset_by_sampling_intervals(ds, measurement_intervals=[60, 180])
        assert 60 in result
        assert len(result[60].time) == 4

    def test_no_valid_intervals_raises(self):
        """If none of the deltas match measurement_intervals, raise ValueError."""
        times = pd.date_range("2023-01-01", periods=5, freq="30s")
        ds = xr.Dataset(coords={"time": times})

        with pytest.raises(ValueError, match="Impossible to identify timesteps"):
            split_dataset_by_sampling_intervals(ds, measurement_intervals=[60, 180])

        # This currently passes (because e.g continuous trailing seconds could raise an error)
        split_dataset_by_sampling_intervals(ds, measurement_intervals=[60])

    def test_small_block_is_dropped_if_multiple_measurement_intervals(self):
        """Short segments (<min_block_size) are removed if multiple intervals are present."""
        times = pd.to_datetime(
            [
                "2023-01-01 00:00:00",
                "2023-01-01 00:01:00",  # 60s
                "2023-01-01 00:02:00",
                "2023-01-01 00:05:00",  # 180s jump
                "2023-01-01 00:08:00",
            ],
        )
        ds = xr.Dataset(coords={"time": times})

        with pytest.raises(ValueError, match="No blocks"):
            split_dataset_by_sampling_intervals(ds, measurement_intervals=[60, 180], min_block_size=4)

    def test_small_block_is_kept_if_unique_interval(self):
        """Test short segments not removed if only one measurements interval is observed in the data."""
        times = pd.to_datetime(
            [
                "2023-01-01 00:00:00",
                "2023-01-01 00:01:00",  # 60s
                "2023-01-01 00:02:00",
            ],
        )
        ds = xr.Dataset(coords={"time": times})
        dict_ds = split_dataset_by_sampling_intervals(ds, measurement_intervals=[60, 180], min_block_size=4)
        assert 60 in dict_ds
        assert dict_ds[60].sizes["time"] == 3

    def test_two_intervals_split_time_is_end(self):
        """Should split into two datasets with distinct sampling intervals."""
        times = pd.to_datetime(
            [
                "2023-01-01 00:00:00",
                "2023-01-01 00:01:00",  # 5x 60s
                "2023-01-01 00:02:00",
                "2023-01-01 00:03:00",
                "2023-01-01 00:04:00",
                "2023-01-01 00:05:00",
                "2023-01-01 00:08:00",  # 2x 180s jump
                "2023-01-01 00:11:00",
            ],
        )
        ds = xr.Dataset(coords={"time": times})

        dict_ds = split_dataset_by_sampling_intervals(ds, measurement_intervals=[60, 180], time_is_end_interval=True)
        assert set(dict_ds.keys()) == {60}  # because only 2 consecutive 180s samples
        # First segment has 5 consecutive samples (60s)
        assert len(dict_ds[60].time) == 6
        assert "2023-01-01T00:05:00" in dict_ds[60]["time"].astype("M8[s]").astype(str)

        times = pd.to_datetime(
            [
                "2023-01-01 00:00:00",
                "2023-01-01 00:01:00",  # 5x 60s
                "2023-01-01 00:02:00",
                "2023-01-01 00:03:00",
                "2023-01-01 00:04:00",
                "2023-01-01 00:05:00",
                "2023-01-01 00:08:00",  # 5x 180s jump
                "2023-01-01 00:11:00",
                "2023-01-01 00:14:00",
                "2023-01-01 00:17:00",
                "2023-01-01 00:20:00",
            ],
        )
        times = times[::-1]  # just ensure that also if not ordered ... it works
        ds = xr.Dataset(coords={"time": times})
        dict_ds = split_dataset_by_sampling_intervals(ds, measurement_intervals=[60, 180], time_is_end_interval=True)
        assert set(dict_ds.keys()) == {60, 180}
        # First segment has 5 consecutive samples (60s)
        assert len(dict_ds[60].time) == 6
        assert "2023-01-01T00:05:00" in dict_ds[60]["time"].astype("M8[s]").astype(str)

        # Second segment has 2 samples (180s)
        assert len(dict_ds[180].time) == 5
        assert "2023-01-01T00:05:00" not in dict_ds[180]["time"].astype("M8[s]").astype(str)

        # Check returned timesteps are sorted
        np.testing.assert_array_equal(dict_ds[60].time.values, np.sort(dict_ds[60].time.values))
        np.testing.assert_array_equal(dict_ds[180].time.values, np.sort(dict_ds[180].time.values))

    def test_two_intervals_split_time_is_start(self):
        """Should split into two datasets with distinct sampling intervals."""
        times = pd.to_datetime(
            [
                "2023-01-01 00:00:00",
                "2023-01-01 00:01:00",  # 5x 60s
                "2023-01-01 00:02:00",
                "2023-01-01 00:03:00",
                "2023-01-01 00:04:00",
                "2023-01-01 00:05:00",
                "2023-01-01 00:08:00",  # 2x 180s jump
                "2023-01-01 00:11:00",
            ],
        )
        ds = xr.Dataset(coords={"time": times})

        dict_ds = split_dataset_by_sampling_intervals(ds, measurement_intervals=[60, 180], time_is_end_interval=False)
        assert set(dict_ds.keys()) == {60}  # because only 2 consecutive 180s samples
        # First segment has 5 consecutive samples (60s)
        assert len(dict_ds[60].time) == 5
        assert "2023-01-01T00:05:00" not in dict_ds[60]["time"].astype("M8[s]").astype(str)

        times = pd.to_datetime(
            [
                "2023-01-01 00:00:00",
                "2023-01-01 00:01:00",  # 5x 60s
                "2023-01-01 00:02:00",
                "2023-01-01 00:03:00",
                "2023-01-01 00:04:00",
                "2023-01-01 00:05:00",
                "2023-01-01 00:08:00",  # 5x 180s jump
                "2023-01-01 00:11:00",
                "2023-01-01 00:14:00",
                "2023-01-01 00:17:00",
                "2023-01-01 00:20:00",
            ],
        )
        times = times[::-1]  # just ensure that also if not ordered ... it works
        ds = xr.Dataset(coords={"time": times})
        dict_ds = split_dataset_by_sampling_intervals(ds, measurement_intervals=[60, 180], time_is_end_interval=False)
        assert set(dict_ds.keys()) == {60, 180}
        # First segment has 5 consecutive samples (60s)
        assert len(dict_ds[60].time) == 5
        assert "2023-01-01T00:05:00" not in dict_ds[60]["time"].astype("M8[s]").astype(str)

        # Second segment has 2 samples (180s)
        assert len(dict_ds[180].time) == 6
        assert "2023-01-01T00:05:00" in dict_ds[180]["time"].astype("M8[s]").astype(str)

        # Check returned timesteps are sorted
        np.testing.assert_array_equal(dict_ds[60].time.values, np.sort(dict_ds[60].time.values))
        np.testing.assert_array_equal(dict_ds[180].time.values, np.sort(dict_ds[180].time.values))

    def test_outlier_first_timestep(self):
        """Should split into two datasets with distinct sampling intervals."""
        times = pd.to_datetime(
            [
                "2022-01-01 00:00:00",  # BAD
                "2023-01-01 00:01:00",  # 5x 60s
                "2023-01-01 00:02:00",
                "2023-01-01 00:03:00",
                "2023-01-01 00:04:00",
                "2023-01-01 00:05:00",
                "2023-01-01 00:08:00",  # 2x 180s jump
                "2023-01-01 00:11:00",
                "2023-01-01 00:14:00",
                "2023-01-01 00:17:00",
                "2023-01-01 00:20:00",
            ],
        )
        ds = xr.Dataset(coords={"time": times})

        dict_ds = split_dataset_by_sampling_intervals(ds, measurement_intervals=[60, 180], time_is_end_interval=True)
        assert set(dict_ds.keys()) == {60, 180}  # because only 2 consecutive 180s samples
        # First segment has 5 consecutive samples (60s)
        assert len(dict_ds[60].time) == 6
        # Test outlier first timesteps not discarded
        assert "2022-01-01T00:00:00" in dict_ds[60]["time"].astype("M8[s]").astype(str)

    def test_outlier_middle_timestep(self):
        """Should split into two datasets with distinct sampling intervals."""
        times = pd.to_datetime(
            [
                "2023-01-01 00:00:00",
                "2023-01-01 00:01:00",  # 5x 60s
                "2023-01-01 00:02:00",
                "2023-01-01 00:03:00",
                "2023-01-01 00:04:00",
                "2023-01-01 00:05:00",
                "2024-01-01 00:05:00",  # outlier
                "2025-01-01 00:08:00",  # 2x 180s jump
                "2025-01-01 00:11:00",
                "2025-01-01 00:14:00",
                "2025-01-01 00:17:00",
                "2025-01-01 00:20:00",
            ],
        )
        ds = xr.Dataset(coords={"time": times})

        dict_ds = split_dataset_by_sampling_intervals(ds, measurement_intervals=[60, 180], time_is_end_interval=True)
        assert set(dict_ds.keys()) == {60, 180}  # because only 2 consecutive 180s samples
        # First segment has 5 consecutive samples (60s)
        assert len(dict_ds[60].time) == 7
        # Test outlier first timesteps not discarded
        assert "2024-01-01T00:05:00" in dict_ds[60]["time"].astype("M8[s]").astype(
            str,
        )  # this will be flagged in qc_time
        assert "2025-01-01T00:08:00" not in dict_ds[60]["time"].astype("M8[s]").astype(str)

    def test_multiple_changing_segments(self):
        """Test it concatenates all segments correctly when measurement interval changing multiple times."""
        times = pd.to_datetime(
            [
                "2023-01-01 00:00:00",
                "2023-01-01 00:01:00",  # 5x 60s
                "2023-01-01 00:02:00",
                "2023-01-01 00:03:00",
                "2023-01-01 00:04:00",
                "2023-01-01 00:05:00",
                "2024-01-01 00:08:00",  # 2x 180s jump
                "2024-01-01 00:11:00",
                "2024-01-01 00:14:00",
                "2024-01-01 00:17:00",
                "2024-01-01 00:20:00",
                "2025-01-01 00:00:00",
                "2025-01-01 00:01:00",  # 5x 60s
                "2025-01-01 00:02:00",
                "2025-01-01 00:03:00",
                "2025-01-01 00:04:00",
                "2025-01-01 00:05:00",
            ],
        )
        ds = xr.Dataset(coords={"time": times})

        dict_ds = split_dataset_by_sampling_intervals(ds, measurement_intervals=[60, 180], time_is_end_interval=True)
        assert set(dict_ds.keys()) == {60, 180}  # because only 2 consecutive 180s samples
        # First segment has 5 consecutive samples (60s)
        assert len(dict_ds[60].time) == 12
        # Test outlier first timesteps not discarded
        assert "2023-01-01T00:05:00" in dict_ds[60]["time"].astype("M8[s]").astype(str)
        assert "2025-01-01T00:00:00" in dict_ds[60]["time"].astype("M8[s]").astype(str)
        assert "2024-01-01T00:08:00" not in dict_ds[60]["time"].astype("M8[s]").astype(str)
        assert "2024-01-01T00:20:00" not in dict_ds[60]["time"].astype("M8[s]").astype(str)

        assert len(dict_ds[180].time) == 5
        # Test outlier first timesteps not discarded
        assert "2023-01-01T00:05:00" not in dict_ds[180]["time"].astype("M8[s]").astype(str)
        assert "2025-01-01T00:00:00" not in dict_ds[180]["time"].astype("M8[s]").astype(str)
        assert "2024-01-01T00:08:00" in dict_ds[180]["time"].astype("M8[s]").astype(str)
        assert "2024-01-01T00:20:00" in dict_ds[180]["time"].astype("M8[s]").astype(str)


class TestHasSameValueOverTime:
    def test_all_equal_values(self):
        """Should return True when all values are identical."""
        da = xr.DataArray([1, 1, 1, 1], dims=["time"])
        assert has_same_value_over_time(da) is True

    def test_varying_values(self):
        """Should return False when values differ over time."""
        da = xr.DataArray([1, 2, 1, 1], dims=["time"])
        assert has_same_value_over_time(da) is False

    def test_all_nans(self):
        """All-NaN series is considered constant."""
        da = xr.DataArray([np.nan, np.nan, np.nan], dims=["time"])
        assert has_same_value_over_time(da) is True

    def test_some_nans_but_equal_values(self):
        """If both first and later values are NaN at the same positions, return True."""
        da = xr.DataArray([[np.nan, 5], [np.nan, 5], [np.nan, 5]], dims=["time", "x"])
        assert has_same_value_over_time(da) is True

    def test_some_nans_and_different_values(self):
        """If values differ at non-NaN positions, return False."""
        da = xr.DataArray([[np.nan, 5], [np.nan, 7]], dims=["time", "x"])
        assert has_same_value_over_time(da) is False

    def test_multidimensional_dataarray(self):
        """Works on multi-d arrays (all timesteps must match the first one)."""
        da = xr.DataArray(np.ones((3, 2, 2)), dims=["time", "lat", "lon"])
        assert has_same_value_over_time(da) is True
        da[1, 0, 0] = 2  # introduce a difference
        assert has_same_value_over_time(da) is False

    def test_single_timestep(self):
        """A single timestep trivially satisfies the condition."""
        da = xr.DataArray([42], dims=["time"])
        assert has_same_value_over_time(da) is True

    def test_all_infs(self):
        """All inf values are considered constant."""
        da = xr.DataArray([np.inf, np.inf, np.inf], dims=["time"])
        assert has_same_value_over_time(da) is True

    def test_mixed_inf_and_values(self):
        """Different inf values vs finite values should return False."""
        da = xr.DataArray([np.inf, 1, np.inf], dims=["time"])
        assert has_same_value_over_time(da) is False

    def test_pos_and_neg_inf(self):
        """+inf and -inf are not equal."""
        da = xr.DataArray([np.inf, -np.inf], dims=["time"])
        assert has_same_value_over_time(da) is False


class TestRemoveDuplicatedTimesteps:
    def test_no_duplicates(self):
        """Dataset without duplicate timesteps should be unchanged."""
        time = np.arange(3)
        ds = xr.Dataset(
            {
                "raw_drop_number": ("time", [1, 2, 3]),
                "var": ("time", [10, 20, 30]),
            },
            coords={"time": time},
        )
        result = remove_duplicated_timesteps(ds)
        xr.testing.assert_identical(result, ds)

    def test_equal_duplicates_keep_first(self):
        """Equal duplicate timesteps should be reduced to one occurrence."""
        time = [0, 0, 1]
        ds = xr.Dataset(
            {
                "raw_drop_number": ("time", [5, 5, 10]),
                "var": ("time", [7, 7, 8]),
            },
            coords={"time": time},
        )
        result = remove_duplicated_timesteps(ds)
        assert result.sizes["time"] == 2
        np.testing.assert_array_equal(result["time"].values, [0, 1])
        np.testing.assert_array_equal(result["raw_drop_number"].values, [5, 10])

    def test_different_raw_drop_number_dropped(self):
        """Duplicates with different raw_drop_number are dropped entirely."""
        time = [0, 0, 1]
        ds = xr.Dataset(
            {
                "raw_drop_number": ("time", [5, 6, 10]),
                "var": ("time", [7, 7, 8]),
            },
            coords={"time": time},
        )
        result = remove_duplicated_timesteps(ds)
        # timestep 0 is fully dropped
        assert result.sizes["time"] == 1
        np.testing.assert_array_equal(result["time"].values, [1])

    def test_other_var_diff_ensure_true(self):
        """When other vars differ and ensure_variables_equality=True, drop them."""
        time = [0, 0, 1]
        ds = xr.Dataset(
            {
                "raw_drop_number": ("time", [5, 5, 10]),
                "var": ("time", [7, 8, 9]),  # differ in duplicate
            },
            coords={"time": time},
        )
        result = remove_duplicated_timesteps(ds, ensure_variables_equality=True)
        # timestep 0 dropped, only time=1 remains
        np.testing.assert_array_equal(result["time"].values, [1])

    def test_other_var_diff_ensure_false(self):
        """When other vars differ but ensure_variables_equality=False, keep raw_drop_number duplicates."""
        time = [0, 0, 1]
        ds = xr.Dataset(
            {
                "raw_drop_number": ("time", [5, 5, 10]),
                "var": ("time", [7, 8, 9]),  # differ in duplicate
            },
            coords={"time": time},
        )
        result = remove_duplicated_timesteps(ds, ensure_variables_equality=False)
        # keep first duplicate of time=0, plus time=1
        np.testing.assert_array_equal(result["time"].values, [0, 1])
        np.testing.assert_array_equal(result["raw_drop_number"].values, [5, 10])


class TestRegularizeTimesteps:
    def test_sorted_regular_dataset(self):
        """Regular timesteps remain unchanged with zero QC flags."""
        sample_interval = 60
        freq = "1min"
        timesteps = pd.date_range("2021-01-01T00:00:00", periods=5, freq=freq)
        arr = np.arange(5)
        ds = xr.Dataset({"data": ("time", arr)}, coords={"time": timesteps})
        ds_out = regularize_timesteps(ds, sample_interval=sample_interval)
        # times and values unchanged
        np.testing.assert_array_equal(ds_out["time"].to_numpy(), timesteps.to_numpy())
        np.testing.assert_array_equal(ds_out["data"].to_numpy(), arr)
        # QC flags all zero
        assert "time_qc" in ds_out.coords
        assert np.all(ds_out["time_qc"].to_numpy() == 0)

    def test_unsorted_regular_dataset(self):
        """Unsorted regular timesteps is sorted by time and has zero QC flags."""
        sample_interval = 60
        freq = "1min"
        timesteps = pd.date_range("2021-01-01T00:00:00", periods=3, freq=freq)
        arr = np.array([0, 1, 2])
        ds = xr.Dataset({"data": ("time", arr)}, coords={"time": timesteps[::-1]})
        ds_out = regularize_timesteps(ds, sample_interval=sample_interval)
        # Output times sorted ascending
        np.testing.assert_array_equal(ds_out["time"].to_numpy(), timesteps.to_numpy())
        # QC flags all zero
        assert "time_qc" in ds_out.coords
        assert np.all(ds_out["time_qc"].to_numpy() == 0)

    def test_missing_timesteps_quality_flags(self):
        """Missing timesteps produce correct QC flags for prev/next missing."""
        base = pd.Timestamp("2021-01-01T00:00:00")
        timesteps = np.array([base, base + pd.Timedelta(seconds=60)], dtype="datetime64[ns]")
        ds = xr.Dataset({"data": ("time", [0, 1])}, coords={"time": timesteps})
        ds_out = regularize_timesteps(ds, sample_interval=30)
        qc = ds_out["time_qc"].to_numpy()
        # First index missing next -> flag 2, second missing previous -> flag 1
        assert qc[0] == 2
        assert qc[1] == 1

    def test_missing_timesteps_every_two(self):
        """Missing timesteps around each timestep."""
        times = pd.date_range("2023-01-01", periods=4, freq="60s")
        ds = xr.Dataset(
            {
                "raw_drop_number": ("time", np.arange(4)),
                "data": ("time", np.random.rand(4)),
            },
            coords={"time": times},
        )
        ds_out = regularize_timesteps(ds, sample_interval=30)
        qc = ds_out["time_qc"].to_numpy()
        np.testing.assert_allclose(qc, [2.0, 3.0, 3.0, 1.0])

    def test_constant_trailing_seconds_correction(self):
        """Trailing seconds are adjusted and flag still kept to 0."""
        times = np.arange(
            np.datetime64("2021-01-01T00:00:00"),
            np.datetime64("2021-01-01T00:05:00"),
            np.timedelta64(1, "m"),
        )
        timesteps = times + np.array(1).astype("timedelta64[s]")  # XX:XX:01 ...
        arr = np.arange(timesteps.size)
        ds = xr.Dataset({"data": ("time", arr)}, coords={"time": timesteps})
        ds_out = regularize_timesteps(ds, sample_interval=60)
        # values unchanged
        np.testing.assert_array_equal(ds_out["data"].to_numpy(), arr)
        # timesteps corrected for trailing seconds
        expected_timesteps = times.astype("datetime64[s]")
        np.testing.assert_array_equal(ds_out["time"].to_numpy(), expected_timesteps)
        # QC flags all zero
        assert "time_qc" in ds_out.coords
        assert np.all(ds_out["time_qc"].to_numpy() == 0)

    def test_varying_trailing_seconds_correction(self):
        """Trailing seconds are adjusted and flag still kept to 0."""
        base = pd.Timestamp("2021-01-01T00:00:00")
        offsets = np.array([0, 29, 59, 92, 118]).astype("timedelta64[s]")
        timesteps = base + offsets
        arr = np.arange(timesteps.size)
        ds = xr.Dataset({"data": ("time", arr)}, coords={"time": timesteps})
        ds_out = regularize_timesteps(ds, sample_interval=30)
        # values unchanged
        np.testing.assert_array_equal(ds_out["data"].to_numpy(), arr)
        # timesteps corrected for trailing seconds
        expected_timesteps = base + np.array([0, 30, 60, 90, 120]).astype("timedelta64[s]")
        np.testing.assert_array_equal(ds_out["time"].to_numpy(), expected_timesteps)
        # QC flags all zero
        assert "time_qc" in ds_out.coords
        assert np.all(ds_out["time_qc"].to_numpy() == 0)

    def test_single_duplicate_identified_and_dropped(self):
        """First occurrence of duplicate timesteps is dropped and flag correctly."""
        base = pd.Timestamp("2021-01-01T00:00:00")
        offsets = np.array([0, 30, 30, 60]).astype("timedelta64[s]")
        timesteps = base + offsets
        arr = np.arange(timesteps.size)
        ds = xr.Dataset({"data": ("time", arr)}, coords={"time": timesteps})
        # Robust=True raise error
        with pytest.raises(ValueError):
            regularize_timesteps(ds, sample_interval=30, robust=True)
        # Otherwise works correctly
        ds_out = regularize_timesteps(ds, sample_interval=30)

        # One duplicate dropped -> length 3
        assert ds_out.sizes["time"] == 3
        # Times are unique and ordered
        expected_timesteps = base + np.array([0, 30, 60]).astype("timedelta64[s]")
        np.testing.assert_array_equal(ds_out["time"].to_numpy(), expected_timesteps)
        # Values array has changed: 1 (first duplicate has been dropped)
        np.testing.assert_array_equal(ds_out["data"].to_numpy(), [0, 2, 3])
        # QC flags: only middle flagged as dropped duplicate (5)
        np.testing.assert_array_equal(ds_out["time_qc"].to_numpy(), [0, 5, 0])

    def test_multiple_duplicates_identified_and_droppped(self):
        """First occurrences of duplicates timesteps are dropped and flag correctly."""
        base = pd.Timestamp("2021-01-01T00:00:00")
        # Three duplicates of the same timestamp
        offsets = np.array([0, 30, 30, 30, 60]).astype("timedelta64[s]")
        timesteps = base + offsets
        arr = np.arange(timesteps.size)
        ds = xr.Dataset({"data": ("time", arr)}, coords={"time": timesteps})

        # Robust=True raise error
        with pytest.raises(ValueError):
            regularize_timesteps(ds, sample_interval=30, robust=True)
        # Otherwise works correctly
        ds_out = regularize_timesteps(ds, sample_interval=30, robust=False)

        # Two duplicate were dropped -> length 3
        assert ds_out.sizes["time"] == 3
        # Times are unique and ordered
        expected_timesteps = base + np.array([0, 30, 60]).astype("timedelta64[s]")
        np.testing.assert_array_equal(ds_out["time"].to_numpy(), expected_timesteps)
        # Values array has changed: 1 (first two duplicates have been dropped)
        np.testing.assert_array_equal(ds_out["data"].to_numpy(), [0, 3, 4])
        # QC flags: only middle flagged as dropped duplicate (5)
        np.testing.assert_array_equal(ds_out["time_qc"].to_numpy(), [0, 5, 0])

    def test_duplicate_moved_to_missing_previous(self):
        """Test duplicate timestep moved to previous missing timestep."""
        base = pd.Timestamp("2021-01-01T00:00:00")
        # Value after 50 seconds can be moved to missing 30 seconds
        offsets = np.array([0, 50, 60, 90]).astype("timedelta64[s]")
        timesteps = base + offsets
        arr = np.arange(timesteps.size)
        ds = xr.Dataset({"data": ("time", arr)}, coords={"time": timesteps})

        # Robust=True do not raise error
        ds_out = regularize_timesteps(ds, sample_interval=30, robust=True)
        ds_out = regularize_timesteps(ds, sample_interval=30, robust=False, verbose=True)
        # Two duplicate were dropped -> length 3
        assert ds_out.sizes["time"] == 4
        # Times are unique and ordered
        expected_timesteps = base + np.array([0, 30, 60, 90]).astype("timedelta64[s]")
        np.testing.assert_array_equal(ds_out["time"].to_numpy(), expected_timesteps)
        # Values array has not changed
        np.testing.assert_array_equal(ds_out["data"].to_numpy(), [0, 1, 2, 3])
        # QC flags: timestep moved get flag 4
        np.testing.assert_array_equal(ds_out["time_qc"].to_numpy(), [0, 4, 0, 0])  # OR [(2), 4, (1), 0]

    def test_duplicate_moved_to_missing_next(self):
        """Test duplicate timestep moved to previous missing timestep."""
        base = pd.Timestamp("2021-01-01T00:00:00")
        # Value +44 seconds can be moved to missing +60 seconds
        offsets = np.array([0, 30, 44, 90]).astype("timedelta64[s]")
        timesteps = base + offsets
        arr = np.arange(timesteps.size)
        ds = xr.Dataset({"data": ("time", arr)}, coords={"time": timesteps})

        # Robust=True do not raise error
        ds_out = regularize_timesteps(ds, sample_interval=30, robust=True)
        ds_out = regularize_timesteps(ds, sample_interval=30, robust=False, verbose=True)
        # Two duplicate were dropped -> length 3
        assert ds_out.sizes["time"] == 4
        # Times are unique and ordered
        expected_timesteps = base + np.array([0, 30, 60, 90]).astype("timedelta64[s]")
        np.testing.assert_array_equal(ds_out["time"].to_numpy(), expected_timesteps)
        # Values array has not changed
        np.testing.assert_array_equal(ds_out["data"].to_numpy(), [0, 1, 2, 3])
        # QC flags: timestep moved get flag 4
        np.testing.assert_array_equal(ds_out["time_qc"].to_numpy(), [0, 0, 4, 0])  # OR: [0, (2), 4, (1)]

    def test_triple_duplicate_moved_to_missing_previous_and_next(self):
        """Test 3 duplicated timestep moved to previous and next missing timesteps."""
        base = pd.Timestamp("2021-01-01T00:00:00")
        # Value +46 seconds can be moved to missing +30 seconds
        # Value +114 seconds can be moved to missing +120 seconds
        offsets = np.array([0, 46, 60, 74, 120]).astype("timedelta64[s]")
        timesteps = base + offsets
        arr = np.arange(timesteps.size)
        ds = xr.Dataset({"data": ("time", arr)}, coords={"time": timesteps})

        # Robust=True do not raise error
        ds_out = regularize_timesteps(ds, sample_interval=30, robust=True)
        ds_out = regularize_timesteps(ds, sample_interval=30, robust=False, verbose=True)

        # Two duplicate were dropped -> length 3
        assert ds_out.sizes["time"] == 5
        # Times are unique and ordered
        expected_timesteps = base + np.array([0, 30, 60, 90, 120]).astype("timedelta64[s]")
        np.testing.assert_array_equal(ds_out["time"].to_numpy(), expected_timesteps)
        # Values array has not changed
        np.testing.assert_array_equal(ds_out["data"].to_numpy(), [0, 1, 2, 3, 4])
        # QC flags: only middle flagged as dropped duplicate
        np.testing.assert_array_equal(ds_out["time_qc"].to_numpy(), [0, 4, 0, 4, 0])  # OR [(2), 4, (4), 4, (1)]


class TestGetProblematicTimestepIndices:
    def test_no_missing_timesteps(self):
        """Continuous timesteps yield no problematic indices."""
        timesteps = pd.date_range("2021-01-01", periods=5, freq="1s")
        prev, nxt, isol = get_problematic_timestep_indices(timesteps, 1)
        assert isinstance(prev, np.ndarray)
        assert prev.size == 0
        assert isinstance(nxt, np.ndarray)
        assert nxt.size == 0
        assert isinstance(isol, np.ndarray)
        assert isol.size == 0

    def test_prev_and_next_missing_separate(self):
        """Timesteps missing previous or next neighbors reported correctly."""
        timesteps = pd.to_datetime(
            [
                "2021-01-01 00:00:00",
                "2021-01-01 00:00:01",
                "2021-01-01 00:00:03",
                "2021-01-01 00:00:04",
            ],
        )
        prev, nxt, isol = get_problematic_timestep_indices(timesteps, 1)
        # index 2 misses previous, index 1 misses next, no isolated
        np.testing.assert_array_equal(prev, np.array([2]))
        np.testing.assert_array_equal(nxt, np.array([1]))
        assert isol.size == 0

    def test_isolated_missing_timesteps(self):
        """Timesteps missing both previous and next are isolated."""
        timesteps = pd.to_datetime(
            [
                "2021-01-01 00:00:00",
                "2021-01-01 00:00:02",
                "2021-01-01 00:00:04",
                "2021-01-01 00:00:05",
            ],
        )
        prev, nxt, isol = get_problematic_timestep_indices(timesteps, 1)
        # index 1 is isolated, others split
        np.testing.assert_array_equal(isol, np.array([1]))
        np.testing.assert_array_equal(prev, np.array([2]))
        np.testing.assert_array_equal(nxt, np.array([0]))

    def test_invalid_timesteps_type_raises(self):
        """Non-datetime timesteps input raises TypeError."""
        with pytest.raises(TypeError):
            get_problematic_timestep_indices([1, 2, 3], 1)


class TestCheckTimestepsRegularity:
    def test_all_regular(self, caplog):
        """No warnings if all timesteps match sample_interval."""
        times = pd.to_datetime([0, 60, 120, 180], unit="s")
        ds = xr.Dataset({"raw_drop_number": ("time", np.arange(len(times)))}, coords={"time": times})

        result = check_timesteps_regularity(ds, sample_interval=60, verbose=True)
        assert isinstance(result, xr.Dataset)
        assert "warning" not in caplog.text.lower()

    def test_sample_interval_fraction_low(self, caplog):
        """Warn if expected interval occurs less than 60%."""
        times = pd.to_datetime([0, 60, 150, 200, 500, 800], unit="s")  # intervals [60, 60, ...]
        times = pd.to_datetime([0, 50, 130, 190], unit="s")  # intervals [50, 70, 60]
        ds = xr.Dataset({"raw_drop_number": ("time", np.arange(len(times)))}, coords={"time": times})

        logger = logging.getLogger("test_logger")
        with caplog.at_level(logging.WARNING):
            check_timesteps_regularity(ds, sample_interval=60, verbose=True, logger=logger)
        assert "expected (sampling) interval is 60 s and occurs 1/4 times" in caplog.text

    def test_most_frequent_interval_differs(self, caplog):
        """Warn if most frequent interval ≠ expected interval."""
        times = pd.to_datetime([0, 60, 70, 80, 90], unit="s")  # intervals [60, 10, 10, 10]
        ds = xr.Dataset({"raw_drop_number": ("time", np.arange(len(times)))}, coords={"time": times})

        logger = logging.getLogger("test_logger")
        with caplog.at_level(logging.WARNING):
            check_timesteps_regularity(ds, sample_interval=60, verbose=True, logger=logger)
            assert "most frequent time interval between observations is 10 s" in caplog.text

    def test_unexpected_interval_frequent(self, caplog):
        """Warn if unexpected intervals occur >5%."""
        times = pd.to_datetime([0, 60, 120, 130, 140], unit="s")
        ds = xr.Dataset({"raw_drop_number": ("time", np.arange(len(times)))}, coords={"time": times})

        logger = logging.getLogger("test_logger")
        with caplog.at_level(logging.WARNING):
            check_timesteps_regularity(ds, sample_interval=60, verbose=True, logger=logger)
            assert "The following time intervals between observations occur frequently: 10 s" in caplog.text

    def test_unsorted_dataset(self):
        """Unsorted dataset should be sorted internally without errors."""
        times = pd.to_datetime([0, 180, 120, 60], unit="s")
        ds = xr.Dataset({"raw_drop_number": ("time", np.arange(len(times)))}, coords={"time": times})

        result = check_timesteps_regularity(ds, sample_interval=60)
        diffs = np.diff(result["time"].values).astype("timedelta64[s]").astype(int)
        assert np.all(diffs >= 0)


class TestCreateL0cDatasets:
    def test_successful_processing_single_interval(self, mocker):
        """Test successful creation of L0C dataset with single measurement interval."""
        # Mock dataset with regular timesteps
        times = pd.date_range("2023-01-01", periods=10, freq="60s")
        ds = xr.Dataset(
            {
                "raw_drop_number": ("time", np.arange(10)),
                "data": ("time", np.random.rand(10)),
            },
            coords={"time": times},
        )

        # Mock open_netcdf_files
        mocker.patch("disdrodb.l0.l0c_processing.open_netcdf_files", return_value=ds)
        mocker.patch("disdrodb.l0.l0c_processing.set_l0b_encodings", side_effect=lambda ds, **kwargs: ds)
        mocker.patch("disdrodb.l0.l0c_processing.set_disdrodb_attrs", side_effect=lambda ds, **kwargs: ds)

        event_info = {
            "start_time": np.datetime64("2023-01-01T00:00:00"),
            "end_time": np.datetime64("2023-01-02T23:59:00"),
            "filepaths": ["file1.nc"],
        }

        result = create_l0c_datasets(
            event_info=event_info,
            measurement_intervals=[60],
            sensor_name="PARSIVEL",
        )

        assert isinstance(result, dict)
        assert 60 in result
        assert isinstance(result[60], xr.Dataset)

    def test_successful_processing_single_interval_with_missing_observations(self, mocker):
        """Test successful creation of L0C dataset when every timestep miss the neighbours."""
        # Mock dataset with regular timesteps
        times = pd.date_range("2023-01-01", periods=10, freq="60s")
        ds = xr.Dataset(
            {
                "raw_drop_number": ("time", np.arange(10)),
                "data": ("time", np.random.rand(10)),
            },
            coords={"time": times},
        )

        # Mock open_netcdf_files
        mocker.patch("disdrodb.l0.l0c_processing.open_netcdf_files", return_value=ds)
        mocker.patch("disdrodb.l0.l0c_processing.set_l0b_encodings", side_effect=lambda ds, **kwargs: ds)
        mocker.patch("disdrodb.l0.l0c_processing.set_disdrodb_attrs", side_effect=lambda ds, **kwargs: ds)

        event_info = {
            "start_time": np.datetime64("2023-01-01T00:00:00"),
            "end_time": np.datetime64("2023-01-02T23:59:00"),
            "filepaths": ["file1.nc"],
        }

        result = create_l0c_datasets(
            event_info=event_info,
            measurement_intervals=[30],
            sensor_name="PARSIVEL",
        )

        assert isinstance(result, dict)
        assert 30 in result
        assert isinstance(result[30], xr.Dataset)

    def test_too_few_timesteps_after_invalid_sample_interval_removal(self, mocker):
        """Test error raised when too few timesteps remain after invalid interval removal."""
        # Mock dataset with mostly invalid sample intervals
        times = pd.date_range("2023-01-01", periods=2, freq="60s")
        ds = xr.Dataset(
            {
                "raw_drop_number": ("time", [1, 2]),
                "sample_interval": ("time", [30, 45]),  # Invalid intervals
                "data": ("time", [1.0, 2.0]),
            },
            coords={"time": times},
        )

        mocker.patch("disdrodb.l0.l0c_processing.open_netcdf_files", return_value=ds)

        event_info = {
            "start_time": np.datetime64("2023-01-01T00:00:00"),
            "end_time": np.datetime64("2023-01-02T00:01:00"),
            "filepaths": ["file1.nc"],
        }

        with pytest.raises(ValueError, match="timesteps left after removing those with unexpected sample interval."):
            create_l0c_datasets(
                event_info=event_info,
                measurement_intervals=[60],
                sensor_name="PARSIVEL",
            )

    def test_too_few_timesteps_after_duplicate_removal(self, mocker):
        """Test error raised when too few timesteps remain after duplicate removal."""
        # Mock dataset with duplicates having different raw_drop_number
        times = pd.date_range("2023-01-01", periods=2, freq="60s")
        ds = xr.Dataset(
            {
                "raw_drop_number": ("time", [1, 2]),  # Different values
                "data": ("time", [1.0, 2.0]),
            },
            coords={"time": times},
        )

        mocker.patch("disdrodb.l0.l0c_processing.open_netcdf_files", return_value=ds)

        event_info = {
            "start_time": np.datetime64("2023-01-01T00:00:00"),
            "end_time": np.datetime64("2023-01-02T00:00:00"),
            "filepaths": ["file1.nc"],
        }

        with pytest.raises(ValueError, match="timesteps left after removing duplicated"):
            create_l0c_datasets(
                event_info=event_info,
                measurement_intervals=[60],
                sensor_name="PARSIVEL",
            )

    def test_multiple_measurement_intervals(self, mocker):
        """Test successful creation with multiple measurement intervals."""
        # Mock dataset with two different sampling intervals
        times = pd.to_datetime(
            [
                "2023-01-01T00:00:00",
                "2023-01-01T00:01:00",  # 5x 60s
                "2023-01-01T00:02:00",
                "2023-01-01T00:03:00",
                "2023-01-01T00:04:00",
                "2023-01-01T00:05:00",
                "2023-01-01T00:08:00",  # 5x 180s
                "2023-01-01T00:11:00",
                "2023-01-01T00:14:00",
                "2023-01-01T00:17:00",
                "2023-01-01T00:20:00",
            ],
        )
        mock_ds = xr.Dataset(
            {
                "raw_drop_number": ("time", np.arange(11)),
                "sample_interval": ("time", [60, 60, 60, 60, 60, 60, 180, 180, 180, 180, 180]),
                "data": ("time", np.random.rand(11)),
            },
            coords={"time": times},
        )

        mocker.patch("disdrodb.l0.l0c_processing.open_netcdf_files", return_value=mock_ds)
        mocker.patch("disdrodb.l0.l0c_processing.set_l0b_encodings", side_effect=lambda ds, **kwargs: ds)
        mocker.patch("disdrodb.l0.l0c_processing.set_disdrodb_attrs", side_effect=lambda ds, **kwargs: ds)

        event_info = {
            "start_time": np.datetime64("2023-01-01T00:00:00"),
            "end_time": np.datetime64("2023-01-01T00:20:00"),
            "filepaths": ["file1.nc"],
        }

        result = create_l0c_datasets(
            event_info=event_info,
            measurement_intervals=[60, 180],
            sensor_name="PARSIVEL",
        )

        assert isinstance(result, dict)
        assert 60 in result
        assert 180 in result
        assert len(result) == 2
        assert isinstance(result[60], xr.Dataset)
        assert isinstance(result[180], xr.Dataset)
