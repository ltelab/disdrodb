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
from disdrodb.l0.l0c_processing import (
    get_files_per_days,
)


class TestGetFilesPerDays:
    """Test suite for get_files_per_days."""

    def test_files_grouped_per_day(self):
        """Files spanning multiple days should be grouped correctly by day."""
        filepaths = [
            "L0B.1MIN.LOCARNO_2019.61.s20190713134200.e20190713161000.V0.nc",
            "L0B.1MIN.LOCARNO_2019.61.s20190714144200.e20190715111000.V0.nc",
            "L0B.1MIN.LOCARNO_2019.61.s20190715144200.e20190716111000.V0.nc",
            "L0B.1MIN.LOCARNO_2019.61.s20190716144200.e20190717111000.V0.nc",
        ]

        dict_days = get_files_per_days(filepaths)

        # Days covered span from 2019-07-13 to 2019-07-17
        expected_days = [
            "2019-07-13",
            "2019-07-14",
            "2019-07-15",
            "2019-07-16",
            "2019-07-17",
        ]
        assert set(dict_days.keys()) == set(expected_days)

        # Check that each day has the correct file(s)
        assert filepaths[0] in dict_days["2019-07-13"]
        assert len(dict_days["2019-07-13"]) == 1
        assert filepaths[1] in dict_days["2019-07-14"]
        assert filepaths[1] in dict_days["2019-07-15"]
        assert filepaths[2] in dict_days["2019-07-15"]
        assert filepaths[2] in dict_days["2019-07-16"]
        assert filepaths[3] in dict_days["2019-07-16"]
        assert filepaths[3] in dict_days["2019-07-17"]

    def test_empty_list_returns_empty_dict(self):
        """An empty list of filepaths should return an empty dict."""
        assert get_files_per_days([]) == {}

    def test_single_file_spanning_multiple_days(self):
        """A single file spanning multiple days should appear in all those days."""
        filepaths = [
            "L0B.1MIN.LOCARNO_2019.61.s20190701000000.e20190702235900.V0.nc",
        ]
        dict_days = get_files_per_days(filepaths)

        # Expect to cover July 1, 2, and 3 + previous and next day around midnight
        expected_days = ["2019-06-30", "2019-07-01", "2019-07-02", "2019-07-03"]
        assert set(dict_days.keys()) == set(expected_days)
        for day in expected_days:
            assert filepaths[0] in dict_days[day]
