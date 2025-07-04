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
"""Testing time utilities."""
import datetime

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from disdrodb.utils.time import (
    acronym_to_seconds,
    check_freq,
    ensure_sample_interval_in_seconds,
    ensure_sorted_by_time,
    generate_time_blocks,
    get_dataframe_start_end_time,
    get_dataset_start_end_time,
    get_file_start_end_time,
    get_problematic_timestep_indices,
    get_resampling_information,
    infer_sample_interval,
    regularize_dataset,
    regularize_timesteps,
    seconds_to_acronym,
)


class TestGetObjectStartEndTime:
    """Test the get_object_start_end_time functionalities."""

    def test_get_dataframe_start_end_time(self):
        """Test the get_dataframe_start_end_time function."""
        start_date = datetime.datetime(2019, 3, 26, 0, 0, 0)
        end_date = datetime.datetime(2021, 2, 8, 0, 0, 0)
        df = pd.DataFrame({"time": pd.date_range(start=start_date, end=end_date)})
        starting_time, ending_time = get_dataframe_start_end_time(df)

        assert isinstance(starting_time, pd.Timestamp)
        assert isinstance(ending_time, pd.Timestamp)
        assert starting_time == start_date
        assert ending_time == end_date

    def test_get_file_start_end_time_with_dataframe(self):
        """Test the get_file_start_end_time with a dataframe."""
        start_date = datetime.datetime(2019, 3, 26, 0, 0, 0)
        end_date = datetime.datetime(2021, 2, 8, 0, 0, 0)
        df = pd.DataFrame({"time": pd.date_range(start=start_date, end=end_date)})
        starting_time, ending_time = get_file_start_end_time(df)

        assert isinstance(starting_time, pd.Timestamp)
        assert isinstance(ending_time, pd.Timestamp)
        assert starting_time == start_date
        assert ending_time == end_date

    def test_get_dataset_start_end_time(self):
        """Test the get_dataset_start_end_time function."""
        # Create a sample xarray Dataset
        times = pd.date_range("2023-01-01", periods=10, freq="D")
        data = np.random.rand(10, 2, 2)  # Random data for the sake of example
        ds = xr.Dataset({"my_data": (("time", "x", "y"), data)}, coords={"time": times})
        expected_start_time = ds["time"].to_numpy()[0]
        expected_end_time = ds["time"].to_numpy()[-1]
        starting_time, ending_time = get_dataset_start_end_time(ds)

        assert isinstance(starting_time, pd.Timestamp)
        assert isinstance(ending_time, pd.Timestamp)
        assert starting_time == expected_start_time
        assert ending_time == expected_end_time

        # Test raise if empty dataset
        empty_ds = xr.Dataset()
        with pytest.raises(KeyError):
            get_dataset_start_end_time(empty_ds)

    def test_get_file_start_end_time_with_dataset(self):
        """Test the get_file_start_end_time with a dataset."""
        # Create a sample xarray dataset
        times = pd.date_range("2023-01-01", periods=10, freq="D")
        data = np.random.rand(10, 2, 2)  # Random data for the sake of example
        ds = xr.Dataset({"my_data": (("time", "x", "y"), data)}, coords={"time": times})
        expected_start_time = ds["time"].to_numpy()[0]
        expected_end_time = ds["time"].to_numpy()[-1]
        starting_time, ending_time = get_file_start_end_time(ds)
        assert isinstance(starting_time, pd.Timestamp)
        assert isinstance(ending_time, pd.Timestamp)
        assert starting_time == expected_start_time
        assert ending_time == expected_end_time

    def test_get_file_start_end_time_with_bad_type(self):
        """Test the get_file_start_end_time with a unexpected object."""
        with pytest.raises(TypeError):
            get_file_start_end_time("dummy_object")


class TestRegularizeDataset:
    """Test the regularize_dataset functionality."""

    def test_regularize_dataset(self):
        """Test regularizing an xarray Dataset with an irregular time coordinate."""
        # Create a dataset with irregular daily timestamps (missing 2023-01-02)
        timesteps = pd.to_datetime(["2023-01-01", "2023-01-03", "2023-01-04"])
        float_data = np.ones((3, 2, 2)) * 2.0
        integer_data = np.ones((3, 2, 2), dtype="uint8")
        str_data = np.ones((3, 2, 2), dtype="str")
        dict_vars = {
            "float_data": (("time", "x", "y"), float_data),
            "int_data": (("time", "x", "y"), integer_data),
            "float_with_fillvalue": (("time", "x", "y"), float_data),
            "int_with_fillvalue": (("time", "x", "y"), integer_data),
            "str_data": (("time", "x", "y"), str_data),
        }
        dict_coords = {
            "time": timesteps,
            "float_coord": ("time", np.ones(timesteps.shape)),
            "int_coord": ("time", np.ones(timesteps.shape, dtype="int8")),
            "int_coord_with_fillvalue": ("time", np.ones(timesteps.shape, dtype="int8")),
            "str_coord": ("time", np.ones(timesteps.shape, dtype="str")),
            "time_coord": ("time", timesteps),
        }
        ds = xr.Dataset(dict_vars, coords=dict_coords)
        ds["float_with_fillvalue"].attrs["_FillValue"] = -1
        ds["int_with_fillvalue"].encoding["_FillValue"] = 0
        ds["int_coord_with_fillvalue"].attrs["_FillValue"] = 0

        # Regularize the dataset with daily frequency
        ds_reg = regularize_dataset(ds, freq="D")

        # Expected time coordinate: daily from the first to the last timestamp
        expected_times = pd.date_range(start="2023-01-01", end="2023-01-04", freq="D")
        np.testing.assert_array_equal(ds_reg["time"].to_numpy(), expected_times.to_numpy())

        # Original data should be aligned; missing dates should have NaNs
        expected_shape = (expected_times.size, 2, 2)
        assert ds_reg["float_data"].shape == expected_shape

        # Check that newly inserted values are the expected ones
        time_index = ds_reg.indexes["time"]
        missing_idx = np.where(time_index == pd.Timestamp("2023-01-02"))[0][0]

        # Float array --> Set NaN
        assert np.all(np.isnan(ds_reg["float_data"].isel(time=missing_idx).to_numpy()))
        # Float coordinates --> Set to NaN
        assert np.all(np.isnan(ds_reg["float_coord"].isel(time=missing_idx).to_numpy()))
        # Float array with FillValue --> Set to NaN
        assert np.all(np.isnan(ds_reg["float_with_fillvalue"].isel(time=missing_idx).to_numpy()))
        # Integer array --> Set to 255 (max value of uint8)
        assert np.all(ds_reg["int_data"].isel(time=missing_idx).to_numpy() == 255)
        # Integer array with FillValue --> Set to specified fillvalue 0
        assert np.all(ds_reg["int_with_fillvalue"].isel(time=missing_idx).to_numpy() == 0)
        # Integer coordinate with FillValue --> Set to specified fillvalue 0
        assert np.all(ds_reg["int_coord_with_fillvalue"].isel(time=missing_idx).to_numpy() == 0)
        # Integer coordinates --> Set to 127 (max value of int8)
        assert np.all(ds_reg["int_coord"].isel(time=missing_idx).to_numpy() == 127)
        # Str array --> Set to NaN and object dtype
        assert str(ds_reg["str_coord"].to_numpy().dtype) == "object"
        assert np.all(ds_reg["str_coord"].isel(time=missing_idx).to_numpy().astype(str) == "nan")
        # Str coordinate --> Set to NaN and object dtype
        assert str(ds_reg["str_coord"].to_numpy().dtype) == "object"
        assert np.all(ds_reg["str_coord"].isel(time=missing_idx).to_numpy().astype(str) == "nan")
        # Datetime coordinate --> Set to NaT
        assert np.all(np.isnat(ds_reg["time_coord"].isel(time=missing_idx).to_numpy()))

    def test_regularize_dataset_with_unsorted_time_coordinate(self):
        """Test regularizing an xarray Dataset with an irregular unsorted time coordinate."""
        timesteps = pd.to_datetime(["2023-01-01", "2023-01-04", "2023-01-03"])
        float_data = np.arange(0, 3 * 2).reshape(3, 2)
        dict_vars = {
            "float_data": (("time", "x"), float_data),
        }
        dict_coords = {
            "time": timesteps,
        }
        ds = xr.Dataset(dict_vars, coords=dict_coords)
        assert np.all(ds["time"].to_numpy() == timesteps)

        # Regularize the dataset
        ds_reg = regularize_dataset(ds, freq="D")

        # Check expected sorted time
        expected_times = pd.date_range(start="2023-01-01", end="2023-01-04", freq="D")
        np.testing.assert_array_equal(ds_reg["time"].to_numpy(), expected_times.to_numpy())

        # Check reordering time does not messed up original values
        np.testing.assert_array_equal(
            ds_reg.sel(time="2023-01-04")["float_data"].to_numpy(),
            ds.sel(time="2023-01-04")["float_data"].to_numpy(),
        )

    def test_regularize_dataset_with_datarray(self):
        """Test regularizing an xarray DataArray."""
        # Create a dataset with irregular daily timestamps (missing 2023-01-02)
        timesteps = pd.to_datetime(["2023-01-01", "2023-01-03", "2023-01-04"])
        integer_data = np.ones((3, 2), dtype="uint8")
        dict_vars = {
            "int_data": (("time", "x"), integer_data),
        }
        dict_coords = {
            "time": timesteps,
            "float_coord": ("time", np.ones(timesteps.shape)),
            "int_coord": ("time", np.ones(timesteps.shape, dtype="int8")),
            "int_coord_with_fillvalue": ("time", np.ones(timesteps.shape, dtype="int8")),
            "str_coord": ("time", np.ones(timesteps.shape, dtype="str")),
            "time_coord": ("time", timesteps),
        }
        ds = xr.Dataset(dict_vars, coords=dict_coords)
        ds["int_coord_with_fillvalue"].attrs["_FillValue"] = 0

        # Regularize the dataset with daily frequency
        da_reg = regularize_dataset(ds["int_data"], freq="D")

        # Expected time coordinate: daily from the first to the last timestamp
        expected_times = pd.date_range(start="2023-01-01", end="2023-01-04", freq="D")
        np.testing.assert_array_equal(da_reg["time"].to_numpy(), expected_times.to_numpy())

        # Original data should be aligned; missing dates should have NaNs
        expected_shape = (expected_times.size, 2)
        assert da_reg.shape == expected_shape

        # Check that newly inserted values are the expected ones
        time_index = da_reg.indexes["time"]
        missing_idx = np.where(time_index == pd.Timestamp("2023-01-02"))[0][0]

        # Integer array --> Set to 255 (max value of uint8)
        assert np.all(da_reg.isel(time=missing_idx).to_numpy() == 255)
        # Integer coordinate with FillValue --> Set to specified fillvalue 0
        assert np.all(da_reg["int_coord_with_fillvalue"].isel(time=missing_idx).to_numpy() == 0)
        # Integer coordinates --> Set to 127 (max value of int8)
        assert np.all(da_reg["int_coord"].isel(time=missing_idx).to_numpy() == 127)
        # Float coordinates --> Set to NaN
        assert np.all(np.isnan(da_reg["float_coord"].isel(time=missing_idx).to_numpy()))
        # Str coordinate --> Set to NaN and object dtype
        assert str(da_reg["str_coord"].to_numpy().dtype) == "object"
        assert np.all(da_reg["str_coord"].isel(time=missing_idx).to_numpy().astype(str) == "nan")
        # Datetime coordinate --> Set to NaT
        assert np.all(np.isnat(da_reg["time_coord"].isel(time=missing_idx).to_numpy()))

    def test_regularize_dataset_raise_error_with_incompatible_frequency(self):
        """Test raise error if frequency incompatible with existing timesteps."""
        # Create a dataset with irregular irregular hourly timestamps
        timesteps = pd.to_datetime(["2023-01-01 00:00", "2023-01-01 02:00", "2023-01-01 05:00"])
        float_data = np.ones((3, 2))
        dict_vars = {
            "float_data": (("time", "x"), float_data),
        }
        dict_coords = {
            "time": timesteps,
        }
        ds = xr.Dataset(dict_vars, coords=dict_coords)

        # Regularize the dataset with daily frequency
        with pytest.raises(ValueError):
            regularize_dataset(ds, freq="2h")

    def test_regularize_dataset_raise_error_with_duplicated_timesteps(self):
        """Test raise error if duplicated timesteps are present."""
        # Create a dataset with duplicated timestamps
        timesteps = pd.to_datetime(["2023-01-01 00:00", "2023-01-01 00:00", "2023-01-01 05:00"])
        dict_vars = {
            "float_data": (("time", "x"), np.ones((3, 2))),
        }
        dict_coords = {
            "time": timesteps,
        }
        ds = xr.Dataset(dict_vars, coords=dict_coords)

        # Check raise error because of duplicated timesteps
        with pytest.raises(ValueError):
            regularize_dataset(ds, freq="1h")


class TestEnsureSortedByTime:
    """Test the ensure_sorted_by_time functionality."""

    def test_dataframe_unsorted(self):
        """Test that an unsorted DataFrame is sorted by time."""
        # Create a DataFrame with unsorted time
        times = pd.to_datetime(["2023-01-03", "2023-01-01", "2023-01-02"])
        df = pd.DataFrame({"time": times, "value": [3, 1, 2]})
        # Ensure unsorted order initially
        assert not df["time"].is_monotonic_increasing

        sorted_df = ensure_sorted_by_time(df)
        # Check that the resulting DataFrame is sorted
        assert sorted_df["time"].is_monotonic_increasing
        # Check that data was correctly reordered
        expected_times = pd.to_datetime(["2023-01-01", "2023-01-02", "2023-01-03"])
        pd.testing.assert_series_equal(
            sorted_df["time"].reset_index(drop=True),
            pd.Series(expected_times),
            check_names=False,
        )

    def test_dataframe_already_sorted(self):
        """Test that a sorted DataFrame remains unchanged."""
        times = pd.date_range(start="2023-01-01", periods=3, freq="D")
        df = pd.DataFrame({"time": times, "value": [1, 2, 3]})
        result_df = ensure_sorted_by_time(df)
        # Data should remain in the same order
        pd.testing.assert_frame_equal(result_df.reset_index(drop=True), df.reset_index(drop=True))

    def test_dataset_unsorted(self):
        """Test that an unsorted xarray Dataset is sorted by its time coordinate."""
        # Create an xarray Dataset with unsorted time coordinate
        unsorted_times = pd.to_datetime(["2023-01-03", "2023-01-01", "2023-01-02"])
        data = np.arange(3)
        ds = xr.Dataset({"data": (("time",), data)}, coords={"time": unsorted_times})
        # Ensure unsorted order initially
        assert not np.all(np.diff(ds["time"].to_numpy()) >= np.timedelta64(0, "s"))

        ds_sorted = ensure_sorted_by_time(ds)
        sorted_times = ds_sorted["time"].to_numpy()

        # Check that the time coordinate is sorted
        assert np.all(np.diff(sorted_times) >= np.timedelta64(0, "s"))
        # Check that data is correctly reordered (assuming sorting by time)
        expected_times = pd.to_datetime(["2023-01-01", "2023-01-02", "2023-01-03"]).to_numpy()
        np.testing.assert_array_equal(sorted_times, expected_times)

    def test_dataset_already_sorted(self):
        """Test that an already sorted xarray Dataset remains unchanged."""
        sorted_times = pd.date_range(start="2023-01-01", periods=3, freq="D")
        data = np.arange(3)
        ds = xr.Dataset({"data": (("time",), data)}, coords={"time": sorted_times})
        ds_sorted = ensure_sorted_by_time(ds)
        # The time coordinate should be identical
        np.testing.assert_array_equal(ds_sorted["time"].to_numpy(), ds["time"].to_numpy())

    def test_invalid_type(self):
        """Test that passing an unsupported type raises a TypeError."""
        with pytest.raises(TypeError):
            ensure_sorted_by_time("invalid_object")


class TestEnsureSampleIntervalInSeconds:
    """Test the ensure_sample_interval_in_seconds functionality."""

    def test_integers_sample_interval(self):
        """Test passing integers values return same value."""
        assert ensure_sample_interval_in_seconds(60) == 60
        assert ensure_sample_interval_in_seconds(1) == 1
        assert ensure_sample_interval_in_seconds(np.int32(1)) == 1
        assert ensure_sample_interval_in_seconds(np.int64(1)) == 1
        assert ensure_sample_interval_in_seconds(np.array(1)) == 1
        assert np.all(ensure_sample_interval_in_seconds(np.array([1, 1])) == np.array([1, 1]))
        assert np.all(
            ensure_sample_interval_in_seconds(xr.DataArray(np.array([2, 2]), dims="time")).data == np.array([2, 2]),
        )

    def test_float_sample_interval(self):
        """Test floating values as input."""
        # Whole number accepted
        assert ensure_sample_interval_in_seconds(4.0) == 4

        # Floating number with decimals, raise error
        with pytest.raises(TypeError):
            ensure_sample_interval_in_seconds(4.5)

    def test_float_xarray_dataarray_whole_numbers(self):
        """Test a xarray.DataArray with floating values."""
        # Test only whole numbers
        sample_interval = xr.DataArray([4.0, 5.0], dims="time")
        interval = ensure_sample_interval_in_seconds(sample_interval)
        assert isinstance(interval, xr.DataArray)
        assert interval.dtype.kind in ("i", "u")
        np.testing.assert_array_equal(interval.to_numpy(), np.array([4, 5], dtype=int))

        # Test only whole numbers and NaN
        sample_interval = xr.DataArray([4.0, np.nan], dims="time")
        interval = ensure_sample_interval_in_seconds(sample_interval)
        assert isinstance(interval, xr.DataArray)
        assert interval.dtype.kind in ("f")
        np.testing.assert_array_equal(interval.to_numpy(), np.array([4.0, np.nan]))

        # Test only NaN
        sample_interval = xr.DataArray([np.nan, np.nan], dims="time")
        interval = ensure_sample_interval_in_seconds(sample_interval)
        assert isinstance(interval, xr.DataArray)
        assert interval.dtype.kind in ("f")
        np.testing.assert_array_equal(interval.to_numpy(), np.array([np.nan, np.nan]))

        # Test invalid sampling interval
        with pytest.raises(TypeError):
            ensure_sample_interval_in_seconds(xr.DataArray([4.0, 4.5], dims="time"))

        with pytest.raises(TypeError):
            ensure_sample_interval_in_seconds(xr.DataArray([np.nan, 4.5], dims="time"))

    def test_float_numpy_array_whole_numbers(self):
        """Test a numpy.ndarray with floating values."""
        # Test only whole numbers
        sample_interval = np.array([4.0, 5.0])
        interval = ensure_sample_interval_in_seconds(sample_interval)
        assert interval.dtype.kind in ("i", "u")
        np.testing.assert_array_equal(interval, np.array([4, 5], dtype=int))

        # Test only whole numbers and NaN
        sample_interval = np.array([4.0, np.nan])
        interval = ensure_sample_interval_in_seconds(sample_interval)
        assert interval.dtype.kind in ("f")
        np.testing.assert_array_equal(interval, np.array([4.0, np.nan], dtype=float))

        # Test only NaN
        sample_interval = np.array([np.nan, np.nan])
        interval = ensure_sample_interval_in_seconds(sample_interval)
        assert interval.dtype.kind in ("f")
        np.testing.assert_array_equal(interval, np.array([np.nan, np.nan]))

        # Test invalid sampling interval
        with pytest.raises(TypeError):
            ensure_sample_interval_in_seconds(np.array([4.0, 4.5]))

        with pytest.raises(TypeError):
            ensure_sample_interval_in_seconds(np.array([np.nan, 4.5]))

    def test_numpy_timedelta(self):
        """Test a a np.timedelta64 object."""
        assert ensure_sample_interval_in_seconds(np.timedelta64(60, "s")) == 60
        assert ensure_sample_interval_in_seconds(np.timedelta64(1, "m")) == 60

    def test_numpy_array_timedelta(self):
        """Test a numpy array with dtype timedelta64."""
        # Create a numpy array with timedelta64 dtype (without NaT)
        sample_interval = pd.date_range(start="2023-01-01", periods=5, freq="60s").diff().to_numpy()[1:]
        interval = ensure_sample_interval_in_seconds(sample_interval)
        assert np.all(interval == 60)

        # Create a numpy array with timedelta64 dtype (without NaT)
        sample_interval = pd.date_range(start="2023-01-01", periods=5, freq="1min").diff().to_numpy()[1:]
        interval = ensure_sample_interval_in_seconds(sample_interval)
        assert np.all(interval == 60)

        # Create a numpy array with timedelta64 dtype (with NaT)
        sample_interval = pd.date_range(start="2023-01-01", periods=5, freq="60s").diff().to_numpy()
        interval = ensure_sample_interval_in_seconds(sample_interval)
        assert np.isnan(interval[0])
        assert np.all(interval[1:] == 60)

    def test_xarray_datarray_timedelta(self):
        """Test an xarray DataArray with dtype timedelta64.."""
        # Create a xarray.DataArray with timedelta64 dtype (without NaT)
        sample_interval = pd.date_range(start="2023-01-01", periods=5, freq="60s").diff().to_numpy()[1:]
        sample_interval = xr.DataArray(sample_interval, dims="time")
        interval = ensure_sample_interval_in_seconds(sample_interval)
        assert isinstance(interval, xr.DataArray)
        assert np.all(interval.to_numpy() == 60)

        # Create a xarray.DataArray with timedelta64 dtype (without NaT)
        sample_interval = pd.date_range(start="2023-01-01", periods=5, freq="60s").diff().to_numpy()
        sample_interval = xr.DataArray(sample_interval, dims="time")
        interval = ensure_sample_interval_in_seconds(sample_interval)
        assert isinstance(interval, xr.DataArray)
        assert np.isnan(interval.data[0])
        assert np.all(interval.data[1:] == 60)

    def test_inconsistent_sample_interval_dtype(self):
        """Test that that passing an unsupported type raises a TypeError."""
        with pytest.raises(TypeError):
            ensure_sample_interval_in_seconds("str")
        with pytest.raises(TypeError):
            ensure_sample_interval_in_seconds(None)
        with pytest.raises(TypeError):
            ensure_sample_interval_in_seconds([1])
        with pytest.raises(TypeError):
            ensure_sample_interval_in_seconds(np.nan)
        with pytest.raises(TypeError):
            ensure_sample_interval_in_seconds(np.datetime64("2021-01-01"))


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


class TestSecondsToAcronym:
    def test_zero_seconds(self):
        """Zero seconds returns empty string."""
        assert seconds_to_acronym(0) == ""

    def test_seconds_only(self):
        """Only seconds less than a minute returns e.g. '45S'."""
        assert seconds_to_acronym(45) == "45S"

    def test_minutes_and_seconds(self):
        """Minutes and seconds return e.g. '1MIN30S'."""
        assert seconds_to_acronym(90) == "1MIN30S"

    def test_hours_and_minutes(self):
        """Hours and minutes return e.g. '1H15MIN'."""
        assert seconds_to_acronym(3600 + 900) == "1H15MIN"

    def test_hours_only(self):
        """Exact hours return only hours acronym."""
        assert seconds_to_acronym(7200) == "2H"

    def test_days_and_hours(self):
        """Days and hours return e.g. '1D2H'."""
        assert seconds_to_acronym(86400 + 7200) == "1D2H"

    def test_full_components(self):
        """Days, hours, minutes, and seconds returns '1D1H1MIN1S'."""
        total = 86400 + 3600 + 60 + 1
        assert seconds_to_acronym(total) == "1D1H1MIN1S"

    def test_invalid_type(self):
        """Non-integer input raises an error."""
        with pytest.raises((TypeError, ValueError)):
            seconds_to_acronym("60")


def test_acronym_to_seconds():
    """Test acronym_to_seconds."""
    assert acronym_to_seconds("30S") == 30


class TestGetResamplingInformation:
    def test_single_seconds(self):
        """Single seconds acronym returns correct seconds and non-rolling flag."""
        seconds, rolling = get_resampling_information("30S")
        assert seconds == 30
        assert rolling is False

    def test_single_minutes(self):
        """Single minutes acronym returns correct seconds and non-rolling flag."""
        seconds, rolling = get_resampling_information("5MIN")
        assert seconds == 5 * 60
        assert rolling is False

    def test_hours_and_minutes(self):
        """Composite hours and minutes acronym computes proper seconds and non-rolling flag."""
        seconds, rolling = get_resampling_information("2H30MIN")
        assert seconds == 2 * 3600 + 30 * 60
        assert rolling is False

    def test_full_components(self):
        """Composite days, hours, minutes, and seconds computes proper seconds."""
        seconds, rolling = get_resampling_information("1D2H15MIN30S")
        assert seconds == 1 * 86400 + 2 * 3600 + 15 * 60 + 30
        assert rolling is False

    def test_rolling_prefix_simple(self):
        """Rolling prefix with single unit returns correct seconds and rolling flag."""
        seconds, rolling = get_resampling_information("ROLL1H")
        assert seconds == 3600
        assert rolling is True

    def test_rolling_prefix_composite(self):
        """Rolling prefix with composite units returns correct seconds and rolling flag."""
        seconds, rolling = get_resampling_information("ROLL15MIN45S")
        assert seconds == 15 * 60 + 45
        assert rolling is True

    def test_zero_seconds(self):
        """Zero seconds acronym returns zero and non-rolling flag."""
        seconds, rolling = get_resampling_information("0S")
        assert seconds == 0
        assert rolling is False

    def test_order_independence(self):
        """Unit order in acronym does not affect computed seconds."""
        s1, _ = get_resampling_information("1H30MIN")
        s2, _ = get_resampling_information("30MIN1H")
        assert s1 == s2

    def test_invalid_missing_unit(self):
        """Acronym missing unit part raises ValueError."""
        with pytest.raises(ValueError):
            get_resampling_information("30")

    def test_invalid_unknown_unit(self):
        """Acronym with unknown unit raises ValueError."""
        with pytest.raises(ValueError):
            get_resampling_information("10X")

    def test_invalid_partial(self):
        """Acronym with partial valid and invalid parts raises ValueError."""
        with pytest.raises(ValueError):
            get_resampling_information("1H30")

    def test_empty_string(self):
        """Empty acronym string raises ValueError."""
        with pytest.raises(ValueError):
            get_resampling_information("")


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

    def test_trailing_seconds_coorection(self):
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


class TestInferSampleInterval:

    def test_uniform_interval(self):
        """Return correct interval for uniformly sampled times."""
        base = np.datetime64("2025-04-28T00:00:00")
        # generate times every 10 seconds
        times = base + np.arange(0, 50, 10).astype("timedelta64[s]")
        ds = xr.Dataset(coords={"time": times})
        assert infer_sample_interval(ds) == 10
        assert infer_sample_interval(ds, robust=True, verbose=True) == 10

    def test_jittered_interval_rounding(self):
        """Round jittered interval to the nearest consistent sample interval."""
        base = np.datetime64("2025-04-28T00:00:00")
        # interval: 10, 9, 11 seconds → all round to 10
        offsets = np.array([0, 10, 19, 30]).astype("timedelta64[s]")
        times = base + offsets
        ds = xr.Dataset(coords={"time": times})
        assert infer_sample_interval(ds) == 10
        assert infer_sample_interval(ds, robust=True, verbose=True) == 10

    def test_exact_duplicate_timesteps(self):
        """Test behaviour with exact duplicated timesteps."""
        base = np.datetime64("2025-04-28T00:00:00")
        # times: 0,10,10,20,30 → one duplicate
        offsets = np.array([0, 10, 10, 20, 30]).astype("timedelta64[s]")
        ds = xr.Dataset(coords={"time": base + offsets})
        # robust=False should not raise error
        assert infer_sample_interval(ds, robust=False, verbose=True) == 10
        # robust=True should not raise error
        assert infer_sample_interval(ds, robust=True, verbose=True) == 10

    def test_likely_duplicate_timesteps(self):
        """Test behaviour with exact duplicated timesteps."""
        base = np.datetime64("2025-04-28T00:00:00")
        # times: 0,10,10,20,30 → one duplicate
        offsets = np.array([0, 10, 11, 20, 30]).astype("timedelta64[s]")
        ds = xr.Dataset(coords={"time": base + offsets})
        # robust=False should not raise error
        assert infer_sample_interval(ds, robust=False, verbose=True) == 10
        # robust=True raise error
        with pytest.raises(ValueError, match="Likely presence of duplicated timesteps"):
            infer_sample_interval(ds, robust=True)

    def test_non_unique_interval_with_relative_large_frequency(self, capsys):
        """Test behaviour when unexpected interval has relative large frequency."""
        base = np.datetime64("2025-04-28T00:00:00")
        # Many 30s and unexpected 10s (with frequency > 20%)
        offsets = np.array([0, 30, 60, 90, 100]).astype("timedelta64[s]")
        ds = xr.Dataset(coords={"time": base + offsets})
        # robust=False should not raise error
        assert infer_sample_interval(ds, robust=False, verbose=True) == 30
        captured = capsys.readouterr()
        assert "unexpected intervals have a frequency greater than 20%" in str(captured.out)
        # robust=True should raise error
        with pytest.raises(ValueError, match="Not unique sampling interval"):
            infer_sample_interval(ds, robust=True)

    def test_non_unique_interval_with_small_confidence(self, capsys):
        """Test behaviour when inferred interval has relative low frequency."""
        # Many 30s (but frequency than 60%) and unexpected 10s
        base = np.datetime64("2025-04-28T00:00:00")
        offsets = np.array([0, 30, 60, 70, 90, 1000]).astype("timedelta64[s]")
        ds = xr.Dataset(coords={"time": base + offsets})
        # robust=False should not raise error
        assert infer_sample_interval(ds, robust=False, verbose=True) == 30
        captured = capsys.readouterr()
        assert "The most frequent sampling interval (30.0 s) has a frequency lower than 60%" in str(captured.out)
        # robust=True should raise error
        with pytest.raises(ValueError, match="Not unique sampling interval"):
            infer_sample_interval(ds, robust=True)


class TestGenerateTimeBlocks:
    def test_none(self):
        """Test 'none' returns the original interval as a single block."""
        start_time = np.datetime64("2022-01-01T05:06:07")
        end_time = np.datetime64("2022-01-02T08:09:10")
        result = generate_time_blocks(start_time, end_time, freq="none")
        expected = np.array([[start_time, end_time]], dtype="datetime64[s]")
        np.testing.assert_array_equal(result, expected)

    def test_hour(self):
        """Test 'hour' splits into full-hour blocks covering the range."""
        start_time = np.datetime64("2022-01-01T10:15:00")
        end_time = np.datetime64("2022-01-01T12:45:00")
        result = generate_time_blocks(start_time, end_time, freq="hour")

        assert result.shape == (3, 2)

        expected_first = np.array(["2022-01-01T10:00:00", "2022-01-01T10:59:59"], dtype="datetime64[s]")
        np.testing.assert_array_equal(result[0], expected_first)

        expected_last = np.array(["2022-01-01T12:00:00", "2022-01-01T12:59:59"], dtype="datetime64[s]")
        np.testing.assert_array_equal(result[-1], expected_last)

    def test_day(self):
        """Test 'day' splits into calendar-day blocks covering the range."""
        start_time = np.datetime64("2022-03-10T05:00:00")
        end_time = np.datetime64("2022-03-12T20:00:00")
        result = generate_time_blocks(start_time, end_time, freq="day")

        assert result.shape == (3, 2)

        expected_first = np.array(["2022-03-10T00:00:00", "2022-03-10T23:59:59"], dtype="datetime64[s]")
        np.testing.assert_array_equal(result[0], expected_first)

        expected_last = np.array(["2022-03-12T00:00:00", "2022-03-12T23:59:59"], dtype="datetime64[s]")
        np.testing.assert_array_equal(result[-1], expected_last)

    def test_month(self):
        """Test 'month' splits into month blocks covering the range."""
        start_time = np.datetime64("2022-01-15")
        end_time = np.datetime64("2022-04-10")
        result = generate_time_blocks(start_time, end_time, freq="month")

        assert result.shape == (4, 2)

        expected_first = np.array(["2022-01-01T00:00:00", "2022-01-31T23:59:59"], dtype="datetime64[s]")
        np.testing.assert_array_equal(result[0], expected_first)

        expected_last = np.array(["2022-04-01T00:00:00", "2022-04-30T23:59:59"], dtype="datetime64[s]")
        np.testing.assert_array_equal(result[-1], expected_last)

    def test_year(self):
        """Test 'year' splits into year blocks covering the range."""
        start_time = np.datetime64("2020-06-01")
        end_time = np.datetime64("2023-02-01")
        result = generate_time_blocks(start_time, end_time, freq="year")

        assert result.shape == (4, 2)

        expected_first = np.array(["2020-01-01T00:00:00", "2020-12-31T23:59:59"], dtype="datetime64[s]")
        np.testing.assert_array_equal(result[0], expected_first)

        expected_last = np.array(["2023-01-01T00:00:00", "2023-12-31T23:59:59"], dtype="datetime64[s]")
        np.testing.assert_array_equal(result[-1], expected_last)

    def test_quarter(self):
        """Test 'quarter' splits into quarter blocks covering the range."""
        start_time = np.datetime64("2022-02-10")
        end_time = np.datetime64("2022-08-20")
        result = generate_time_blocks(start_time, end_time, freq="quarter")

        assert result.shape == (3, 2)

        expected_first = np.array(["2022-01-01T00:00:00", "2022-03-31T23:59:59"], dtype="datetime64[s]")
        np.testing.assert_array_equal(result[0], expected_first)

        expected_last = np.array(["2022-07-01T00:00:00", "2022-09-30T23:59:59"], dtype="datetime64[s]")
        np.testing.assert_array_equal(result[-1], expected_last)

    def test_season(self):
        """Test 'season' splits into meteorological-season blocks covering the range."""
        start_time = np.datetime64("2022-01-01")
        end_time = np.datetime64("2022-12-31")
        result = generate_time_blocks(start_time, end_time, freq="season")
        assert result.shape == (5, 2)

        expected_first = np.array(["2021-12-01T00:00:00", "2022-02-28T23:59:59"], dtype="datetime64[s]")
        np.testing.assert_array_equal(result[0], expected_first)

        expected_last = np.array(["2022-12-01T00:00:00", "2023-02-28T23:59:59"], dtype="datetime64[s]")
        np.testing.assert_array_equal(result[-1], expected_last)


class TestCheckFreq:
    def test_valid_freq_returns_value(self):
        """Valid freq strings should be returned unchanged."""
        for valid in ["none", "year", "season", "quarter", "month", "day", "hour"]:
            assert check_freq(valid) == valid

    def test_non_string_raises_type_error(self):
        """Non-string freq inputs should raise a TypeError."""
        with pytest.raises(TypeError) as excinfo:
            check_freq(123)
        assert "'freq' must be a string." in str(excinfo.value)

    def test_invalid_string_raises_value_error(self):
        """String not in allowed list should raise a ValueError."""
        with pytest.raises(ValueError) as excinfo:
            check_freq("minute")
        assert "'freq' 'minute' is not possible. Must be one of:" in str(excinfo.value)
