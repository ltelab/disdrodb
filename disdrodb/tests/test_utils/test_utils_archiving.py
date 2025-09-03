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
"""Test DISDRODB archiving utilities."""
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from disdrodb.utils import archiving
from disdrodb.utils.archiving import (
    check_freq,
    generate_time_blocks,
    get_files_partitions,
    get_files_per_time_block,
    identify_events,
    identify_time_partitions,
)


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

        # Test XX::00:00 - XX:00:00 case
        # -inclusive_end_time=True:  if end_time is XX:00:00 it includes that hour up to 59:59
        start_time = np.datetime64("2022-03-10T02:00:00")
        end_time = np.datetime64("2022-03-10T03:00:00")
        result = generate_time_blocks(start_time, end_time, freq="hour")
        assert result.shape == (2, 2)
        expected_first = np.array(["2022-03-10T02:00:00", "2022-03-10T02:59:59"], dtype="datetime64[s]")
        np.testing.assert_array_equal(result[0], expected_first)

        expected_last = np.array(["2022-03-10T03:00:00", "2022-03-10T03:59:59"], dtype="datetime64[s]")
        np.testing.assert_array_equal(result[-1], expected_last)
        # - inclusive_end_time=False: if end_time is XX:00:00 should not include that hour
        result = generate_time_blocks(start_time, end_time, freq="hour", inclusive_end_time=False)
        assert result.shape == (1, 2)
        expected_first = np.array(["2022-03-10T02:00:00", "2022-03-10T02:59:59"], dtype="datetime64[s]")
        np.testing.assert_array_equal(result[0], expected_first)

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

        # Test 00:00:00 - 00:00:00 case
        # - inclusive_end_time=True: if end_time is 00:00:00 it includes that day up to 23:59:59
        start_time = np.datetime64("2022-03-10T00:00:00")
        end_time = np.datetime64("2022-03-11T00:00:00")
        result = generate_time_blocks(start_time, end_time, freq="day")
        assert result.shape == (2, 2)
        expected_first = np.array(["2022-03-10T00:00:00", "2022-03-10T23:59:59"], dtype="datetime64[s]")
        np.testing.assert_array_equal(result[0], expected_first)
        expected_last = np.array(["2022-03-11T00:00:00", "2022-03-11T23:59:59"], dtype="datetime64[s]")
        np.testing.assert_array_equal(result[-1], expected_last)
        # - inclusive_end_time=True: if end_time is 00:00:00 it does not includes that day
        result = generate_time_blocks(start_time, end_time, freq="day", inclusive_end_time=False)
        assert result.shape == (1, 2)
        expected_first = np.array(["2022-03-10T00:00:00", "2022-03-10T23:59:59"], dtype="datetime64[s]")
        np.testing.assert_array_equal(result[0], expected_first)

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

        # Test XXX-01-01 - XXX-02-01 case
        # - inclusive_end_time=True: if end_time is XXXX-01-01 it includes that month up to last day of month
        start_time = np.datetime64("2022-01-01")
        end_time = np.datetime64("2022-02-01")
        result = generate_time_blocks(start_time, end_time, freq="month")
        assert result.shape == (2, 2)
        expected_first = np.array(["2022-01-01T00:00:00", "2022-01-31T23:59:59"], dtype="datetime64[s]")
        np.testing.assert_array_equal(result[0], expected_first)
        expected_last = np.array(["2022-02-01T00:00:00", "2022-02-28T23:59:59"], dtype="datetime64[s]")
        np.testing.assert_array_equal(result[-1], expected_last)
        # - inclusive_end_time=False: if end_time is XXXX-01-01 it does not includes that month
        result = generate_time_blocks(start_time, end_time, freq="month", inclusive_end_time=False)
        assert result.shape == (1, 2)
        expected_first = np.array(["2022-01-01T00:00:00", "2022-01-31T23:59:59"], dtype="datetime64[s]")
        np.testing.assert_array_equal(result[0], expected_first)

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


class TestIdentifyEvents:
    def test_returns_expected_dict_list(self, monkeypatch):
        """Check that identify_events returns a list of dicts with expected keys."""

        # --- Monkeypatch open_netcdf_files to return a minimal dataset ---
        def fake_open_netcdf_files(filepaths, variables, parallel, compute):
            times = pd.date_range("2022-01-01", periods=5, freq="H")
            N = [0, 10, 0, 12, 15]  # some below and some above min_drops
            ds = xr.Dataset(
                {
                    "N": ("time", np.array(N)),
                },
                coords={"time": times.to_numpy()},
            )
            return ds

        monkeypatch.setattr(archiving, "open_netcdf_files", fake_open_netcdf_files)

        # --- Run function ---
        events = identify_events(filepaths=["fake1", "fake2"], min_drops=5)

        # --- Assertions ---
        assert isinstance(events, list)
        assert all(isinstance(e, dict) for e in events)
        for e in events:
            assert set(e.keys()) == {"start_time", "end_time", "duration", "n_timesteps"}
            assert isinstance(e["start_time"], np.datetime64)
            assert isinstance(e["end_time"], np.datetime64)
            assert isinstance(e["duration"], np.timedelta64)
            assert isinstance(e["n_timesteps"], (int, np.integer))

    def test_empty_result_when_no_timesteps(self, monkeypatch):
        """Check that identify_events returns [] when no valid rainy timesteps exist."""

        # --- Monkeypatch open_netcdf_files to return empty dataset ---
        def fake_open_netcdf_files(filepaths, variables, parallel, compute):
            times = pd.date_range("2022-01-01", periods=5, freq="H")
            N = [0, 0, 0, 0, 0]  # all below threshold
            ds = xr.Dataset(
                {
                    "N": ("time", np.array(N)),
                },
                coords={"time": times.to_numpy()},
            )
            return ds

        monkeypatch.setattr(archiving, "open_netcdf_files", fake_open_netcdf_files)

        # Test identify_events function return empty list
        events = identify_events(["fake"], min_drops=5)
        assert events == []


def generate_product_filename(start_time, end_time, product="L1", temporal_resolution="1MIN"):
    """Helper to generate DISDRODB products filenames given a numpy.datetime64 start_time/end_time."""
    s = np.datetime_as_string(start_time, unit="s").replace("-", "").replace("T", "").replace(":", "")
    e = np.datetime_as_string(end_time, unit="s").replace("-", "").replace("T", "").replace(":", "")
    return f"{product}.{temporal_resolution}.campaign.station.s{s}.e{e}.V0.nc"
    # 'L1.1MIN.UL.Ljubljana.s20180601120000.e20180701120000.V0.nc',


class TestIdentifyTimePartitions:
    def test_none_frequency(self):
        """'none' returns a single block spanning all files."""
        start_times = np.array([np.datetime64("2017-05-22T15:45:58"), np.datetime64("2018-06-01T12:00:00")])
        end_times = np.array([np.datetime64("2017-07-31T13:29:27"), np.datetime64("2018-07-01T12:00:00")])

        result = identify_time_partitions(start_times, end_times, freq="none")
        assert len(result) == 1
        block = result[0]
        assert block["start_time"] == np.datetime64("2017-05-22T15:45:58")
        assert block["end_time"] == np.datetime64("2018-07-01T12:00:00")

    def test_day_frequency(self):
        """'day' returns each calendar day block overlapping any file."""
        start_times = np.array([np.datetime64("2022-05-01T10:00:00"), np.datetime64("2022-05-05T15:00:00")])
        end_times = np.array([np.datetime64("2022-05-01T12:00:00"), np.datetime64("2022-05-05T17:00:00")])

        result = identify_time_partitions(start_times, end_times, freq="day")

        assert len(result) == 2
        assert result[0] == {
            "start_time": np.datetime64("2022-05-01T00:00:00"),
            "end_time": np.datetime64("2022-05-01T23:59:59"),
        }
        assert result[1] == {
            "start_time": np.datetime64("2022-05-05T00:00:00"),
            "end_time": np.datetime64("2022-05-05T23:59:59"),
        }

    def test_season_frequency(self):
        """'season' returns DJF block starting Dec of previous year when needed."""
        start_times = np.array([np.datetime64("2022-01-01T12:00:00")])
        end_times = np.array([np.datetime64("2022-02-15T12:00:00")])

        result = identify_time_partitions(start_times, end_times, freq="season")

        assert len(result) == 1
        block = result[0]
        assert block["start_time"] == np.datetime64("2021-12-01T00:00:00")
        assert block["end_time"] == np.datetime64("2022-02-28T23:59:59")


# Generate filenames for period on 2017-05-22T00:00:00 - 2017-05-22T00:11:00^
base = np.datetime64("2017-05-22T00:00:00")
filepaths = [
    generate_product_filename(base + np.timedelta64(i * 2, "m"), base + np.timedelta64(i * 2 + 1, "m"))
    for i in range(6)
]


class TestGetFilesPartitions:

    def test_no_overlap(self):
        """Events outside the files range should yield no results."""
        events = [
            {
                "start_time": np.datetime64("2017-05-22T17:30:00"),
                "end_time": np.datetime64("2017-05-22T18:00:00"),
            },
        ]
        out = get_files_partitions(events, filepaths, sample_interval=60, accumulation_interval=60, rolling=False)
        assert out == []

    def test_no_aggregation_case(self):
        """Test case when sample_interval== accumulation_interval."""
        events = [
            {
                "start_time": np.datetime64("2017-05-22T00:00:00"),
                "end_time": np.datetime64("2017-05-22T00:02:00"),
            },
        ]
        out = get_files_partitions(events, filepaths, sample_interval=60, accumulation_interval=60, rolling=False)
        assert len(out) == 1
        info = out[0]
        assert info["start_time"] == np.datetime64("2017-05-22T00:00:00")
        assert info["end_time"] == np.datetime64("2017-05-22T00:02:00")
        assert len(info["filepaths"]) == 2

        # Extend end_time by 180 seconds
        # --> Include last file based on its end time
        out = get_files_partitions(events, filepaths, sample_interval=60, accumulation_interval=180, rolling=False)
        assert len(out) == 1
        info = out[0]
        assert info["start_time"] == np.datetime64("2017-05-22T00:00:00")
        assert info["end_time"] == np.datetime64("2017-05-22T00:05:00")
        assert len(info["filepaths"]) == 3
        assert info["filepaths"][-1] == "L1.1MIN.campaign.station.s20170522000400.e20170522000500.V0.nc"

    def test_single_event_not_rolling_case(self):
        """Test case for rolling=False."""
        events = [
            {
                "start_time": np.datetime64("2017-05-22T00:00:00"),
                "end_time": np.datetime64("2017-05-22T00:02:00"),
            },
        ]
        # Extend end_time by 120 seconds
        # --> Include last file based on its start time
        out = get_files_partitions(events, filepaths, sample_interval=60, accumulation_interval=120, rolling=False)
        assert len(out) == 1
        info = out[0]
        assert info["start_time"] == np.datetime64("2017-05-22T00:00:00")
        assert info["end_time"] == np.datetime64("2017-05-22T00:04:00")
        assert len(info["filepaths"]) == 3
        assert info["filepaths"][-1] == "L1.1MIN.campaign.station.s20170522000400.e20170522000500.V0.nc"

        # Extend end_time by 180 seconds
        # --> Include last file based on its end time
        out = get_files_partitions(events, filepaths, sample_interval=60, accumulation_interval=180, rolling=False)
        assert len(out) == 1
        info = out[0]
        assert info["start_time"] == np.datetime64("2017-05-22T00:00:00")
        assert info["end_time"] == np.datetime64("2017-05-22T00:05:00")
        assert len(info["filepaths"]) == 3
        assert info["filepaths"][-1] == "L1.1MIN.campaign.station.s20170522000400.e20170522000500.V0.nc"

    def test_single_event_rolling_case(self):
        """Test case for rolling=True."""
        events = [
            {
                "start_time": np.datetime64("2017-05-22T00:00:00"),
                "end_time": np.datetime64("2017-05-22T00:02:00"),
            },
        ]
        # Extend end_time by 120 seconds
        # --> Include last file based on its start time
        out = get_files_partitions(events, filepaths, sample_interval=60, accumulation_interval=120, rolling=True)
        assert len(out) == 1
        info = out[0]
        assert info["start_time"] == np.datetime64("2017-05-22T00:00:00")
        assert info["end_time"] == np.datetime64("2017-05-22T00:04:00")
        assert len(info["filepaths"]) == 3
        assert info["filepaths"][-1] == "L1.1MIN.campaign.station.s20170522000400.e20170522000500.V0.nc"

        # Extend end_time by 180 seconds
        # --> Include last file based on its end time
        out = get_files_partitions(events, filepaths, sample_interval=60, accumulation_interval=180, rolling=True)
        assert len(out) == 1
        info = out[0]
        assert info["start_time"] == np.datetime64("2017-05-22T00:00:00")
        assert info["end_time"] == np.datetime64("2017-05-22T00:05:00")
        assert len(info["filepaths"]) == 3
        assert info["filepaths"][-1] == "L1.1MIN.campaign.station.s20170522000400.e20170522000500.V0.nc"

    def test_multiple_event(self):
        """Multiple events with forward 1-min extend map to correct file sets."""
        events = [
            {
                "start_time": np.datetime64("2017-05-22T00:00:00"),
                "end_time": np.datetime64("2017-05-22T00:02:00"),
            },
            {
                "start_time": np.datetime64("2017-05-22T00:04:00"),
                "end_time": np.datetime64("2017-05-22T00:05:00"),
            },
            {  # Event without files
                "start_time": np.datetime64("2018-05-22T00:04:00"),
                "end_time": np.datetime64("2018-05-22T00:05:00"),
            },
        ]

        out = get_files_partitions(events, filepaths, sample_interval=60, accumulation_interval=120, rolling=False)
        assert len(out) == 2
        assert out[0]["start_time"] == np.datetime64("2017-05-22T00:00:00")
        assert out[0]["end_time"] == np.datetime64("2017-05-22T00:04:00")

        assert out[1]["start_time"] == np.datetime64("2017-05-22T00:04:00")
        assert out[1]["end_time"] == np.datetime64("2017-05-22T00:07:00")

    def test_empty_events_list(self):
        """Providing no events returns an empty list."""
        out = get_files_partitions([], filepaths, sample_interval=60, accumulation_interval=60, rolling=False)
        assert out == []

    def test_empty_filepaths_list(self):
        """Providing no files always yields no overlaps."""
        events = [
            {
                "start_time": np.datetime64("2017-05-22T00:01:00"),
                "end_time": np.datetime64("2017-05-22T00:01:00"),
            },
        ]
        out = get_files_partitions(events, [], sample_interval=60, accumulation_interval=60, rolling=False)
        assert out == []


class TestGetFilesPerDays:
    """Test suite for get_files_per_time_block."""

    def test_files_grouped_per_day(self):
        """Files spanning multiple days should be grouped correctly by day."""
        filepaths = [
            "L0B.1MIN.LOCARNO_2019.61.s20190713134200.e20190713161000.V0.nc",
            "L0B.1MIN.LOCARNO_2019.61.s20190714144200.e20190715111000.V0.nc",
            "L0B.1MIN.LOCARNO_2019.61.s20190715144200.e20190716111000.V0.nc",
            "L0B.1MIN.LOCARNO_2019.61.s20190716144200.e20190717111000.V0.nc",
        ]

        dict_days = get_files_per_time_block(filepaths)

        # Days covered span from 2019-07-13 to 2019-07-17
        expected_days = [
            "2019-07-13T00:00:00",
            "2019-07-14T00:00:00",
            "2019-07-15T00:00:00",
            "2019-07-16T00:00:00",
            "2019-07-17T00:00:00",
        ]
        assert set(dict_days.keys()) == set(expected_days)

        # Check that each day has the correct file(s)
        assert filepaths[0] in dict_days["2019-07-13T00:00:00"]
        assert len(dict_days["2019-07-13T00:00:00"]) == 1
        assert filepaths[1] in dict_days["2019-07-14T00:00:00"]
        assert filepaths[1] in dict_days["2019-07-15T00:00:00"]
        assert filepaths[2] in dict_days["2019-07-15T00:00:00"]
        assert filepaths[2] in dict_days["2019-07-16T00:00:00"]
        assert filepaths[3] in dict_days["2019-07-16T00:00:00"]
        assert filepaths[3] in dict_days["2019-07-17T00:00:00"]

    def test_empty_list_returns_empty_dict(self):
        """An empty list of filepaths should return an empty dict."""
        assert get_files_per_time_block([]) == {}

    def test_single_file_spanning_multiple_days(self):
        """A single file spanning multiple days should appear in all those days."""
        filepaths = [
            "L0B.1MIN.LOCARNO_2019.61.s20190701000000.e20190702235900.V0.nc",
        ]
        dict_days = get_files_per_time_block(filepaths)

        # Expect to cover July 1, 2, and 3 + previous and next day around midnight
        expected_days = ["2019-06-30T00:00:00", "2019-07-01T00:00:00", "2019-07-02T00:00:00", "2019-07-03T00:00:00"]
        assert set(dict_days.keys()) == set(expected_days)
        for day in expected_days:
            assert filepaths[0] in dict_days[day]
