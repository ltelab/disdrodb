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

from disdrodb.utils.archiving import (
    get_files_partitions,
    identify_time_partitions,
)


def generate_product_filename(start_time, end_time, product="L1", temporal_resolution="1MIN"):
    """Helper to generate DISDRODB products filenames given a numpy.datetime64 start_time/end_time."""
    s = np.datetime_as_string(start_time, unit="s").replace("-", "").replace("T", "").replace(":", "")
    e = np.datetime_as_string(end_time, unit="s").replace("-", "").replace("T", "").replace(":", "")
    return f"{product}.{temporal_resolution}.campaign.station.s{s}.e{e}.V0.nc"
    # 'L1.1MIN.UL.Ljubljana.s20180601120000.e20180701120000.V0.nc',


class TestIdentifyTimePartitions:
    def test_none_frequency(self, monkeypatch):
        """'none' returns a single block spanning all files."""
        filepaths = [
            "L1.1MIN.UL.Ljubljana.s20170522154558.e20170731132927.V0.nc",
            "L1.1MIN.UL.Ljubljana.s20180601120000.e20180701120000.V0.nc",
        ]

        result = identify_time_partitions(filepaths, freq="none")
        assert len(result) == 1
        block = result[0]
        assert block["start_time"] == np.datetime64("2017-05-22T15:45:58")
        assert block["end_time"] == np.datetime64("2018-07-01T12:00:00")

    def test_day_frequency(self, monkeypatch):
        """'day' returns each calendar day block overlapping any file."""
        filepaths = [
            "L1.1MIN.UL.Ljubljana.s20220501100000.e20220501120000.V0.nc",
            "L1.1MIN.UL.Ljubljana.s20220505150000.e20220505170000.V0.nc",
        ]
        result = identify_time_partitions(filepaths, freq="day")

        # Expect two days: May 1 and May 5, 2022
        assert len(result) == 2
        assert result[0] == {
            "start_time": np.datetime64("2022-05-01T00:00:00"),
            "end_time": np.datetime64("2022-05-01T23:59:59"),
        }
        assert result[1] == {
            "start_time": np.datetime64("2022-05-05T00:00:00"),
            "end_time": np.datetime64("2022-05-05T23:59:59"),
        }

    def test_season_frequency(self, monkeypatch):
        """'season' returns DJF block starting Dec of previous year when needed."""
        filepaths = [
            "L1.1MIN.UL.Ljubljana.s20220101120000.e20220215120000.V0.nc",
        ]
        result = identify_time_partitions(filepaths, freq="season")
        # Only the DJF season for 2022 should appear, starting 2021-12-01, ending 2022-02-28
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
