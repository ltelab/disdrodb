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
"""Test DISDRODB Input/Output Function."""
import datetime

import pytest

from disdrodb.api.io import filter_by_time, find_files
from disdrodb.tests.conftest import create_fake_raw_data_file

# import pathlib
# tmp_path = pathlib.Path("/tmp/17")


class TestFindFiles:

    def test_find_raw_files(self, tmp_path):
        """Test finding raw files."""
        # Define station info
        data_archive_dir = tmp_path / "data" / "DISDRODB"
        data_source = "DATA_SOURCE"
        campaign_name = "CAMPAIGN_NAME"
        station_name = "STATION_NAME"
        # Define correct glob pattern
        glob_pattern = "*.txt"
        # Test that the function raises an error if no files presenet
        with pytest.raises(ValueError, match="No RAW files are available in"):
            _ = find_files(
                data_archive_dir=data_archive_dir,
                data_source=data_source,
                campaign_name=campaign_name,
                station_name=station_name,
                product="RAW",
                glob_pattern=glob_pattern,
            )

        # Add fake data files
        for filename in ["file1.txt", "file2.txt"]:
            _ = create_fake_raw_data_file(
                data_archive_dir=data_archive_dir,
                product="RAW",
                data_source=data_source,
                campaign_name=campaign_name,
                station_name=station_name,
                filename=filename,
            )

        # Test that the function returns the correct number of files in debugging mode
        filepaths = find_files(
            data_archive_dir=data_archive_dir,
            data_source=data_source,
            campaign_name=campaign_name,
            station_name=station_name,
            product="RAW",
            glob_pattern=glob_pattern,
            debugging_mode=True,
        )
        assert len(filepaths) == 2  # max(2, 3)

        # Add other fake data files
        for filename in ["file3.txt", "file4.txt"]:
            _ = create_fake_raw_data_file(
                data_archive_dir=data_archive_dir,
                product="RAW",
                data_source=data_source,
                campaign_name=campaign_name,
                station_name=station_name,
                filename=filename,
            )
        filepaths = find_files(
            data_archive_dir=data_archive_dir,
            data_source=data_source,
            campaign_name=campaign_name,
            station_name=station_name,
            product="RAW",
            glob_pattern=glob_pattern,
            debugging_mode=True,
        )
        assert len(filepaths) == 3  # 3 when debugging_mode

        # Test that the function returns the correct number of files in normal mode
        filepaths = find_files(
            data_archive_dir=data_archive_dir,
            data_source=data_source,
            campaign_name=campaign_name,
            station_name=station_name,
            product="RAW",
            glob_pattern=glob_pattern,
            debugging_mode=False,
        )
        assert len(filepaths) == 4

        # Test that the function raises an error if the glob_patterns is not a str or list
        with pytest.raises(ValueError, match="'glob_patterns' must be a str or list of strings."):
            find_files(
                data_archive_dir=data_archive_dir,
                data_source=data_source,
                campaign_name=campaign_name,
                station_name=station_name,
                product="RAW",
                glob_pattern=1,
                debugging_mode=False,
            )

        # Test that the function raises an error if no files are found
        with pytest.raises(ValueError, match="No RAW files are available in"):
            find_files(
                data_archive_dir=data_archive_dir,
                data_source=data_source,
                campaign_name=campaign_name,
                station_name=station_name,
                product="RAW",
                glob_pattern="*.csv",
                debugging_mode=False,
            )

        # Test with list of glob patterns
        for filename in ["file_new.csv"]:
            _ = create_fake_raw_data_file(
                data_archive_dir=data_archive_dir,
                product="RAW",
                data_source=data_source,
                campaign_name=campaign_name,
                station_name=station_name,
                filename=filename,
            )
        filepaths = find_files(
            data_archive_dir=data_archive_dir,
            data_source=data_source,
            campaign_name=campaign_name,
            station_name=station_name,
            product="RAW",
            glob_pattern=["*txt", "*.csv"],
            debugging_mode=False,
        )
        assert len(filepaths) == 5

    def test_find_l0a_files(self, tmp_path):
        """Test finding L0A files."""
        # Define station info
        data_archive_dir = tmp_path / "data" / "DISDRODB"
        data_source = "DATA_SOURCE"
        campaign_name = "CAMPAIGN_NAME"
        station_name = "STATION_NAME"
        product = "L0A"

        # Test that the function raises an error if no files presenet
        with pytest.raises(ValueError):
            _ = find_files(
                data_archive_dir=data_archive_dir,
                data_source=data_source,
                campaign_name=campaign_name,
                station_name=station_name,
                product=product,
            )

        # Add fake data files
        filenames = [
            "L0A.1MIN.LOCARNO_2019.61.s20190713134200.e20190714111000.V0.parquet",
            "L0A.1MIN.LOCARNO_2019.61.s20190714144200.e20190715111000.V0.parquet",
        ]
        for filename in filenames:
            _ = create_fake_raw_data_file(
                data_archive_dir=data_archive_dir,
                product=product,
                data_source=data_source,
                campaign_name=campaign_name,
                station_name=station_name,
                filename=filename,
            )

        # Test that the function returns the correct number of files in debugging mode
        filepaths = find_files(
            data_archive_dir=data_archive_dir,
            data_source=data_source,
            campaign_name=campaign_name,
            station_name=station_name,
            product=product,
            debugging_mode=True,
        )
        assert len(filepaths) == 2  # max(2, 3)

        # Test that the function returns the correct number of files in normal mode
        filepaths = find_files(
            data_archive_dir=data_archive_dir,
            data_source=data_source,
            campaign_name=campaign_name,
            station_name=station_name,
            product=product,
        )
        assert len(filepaths) == 2

    def test_find_l0b_files(self, tmp_path):
        """Test finding L0B files."""
        # Define station info
        data_archive_dir = tmp_path / "data" / "DISDRODB"
        data_source = "DATA_SOURCE"
        campaign_name = "CAMPAIGN_NAME"
        station_name = "STATION_NAME"
        product = "L0B"

        # Test that the function raises an error if no files presenet
        with pytest.raises(ValueError):
            _ = find_files(
                data_archive_dir=data_archive_dir,
                data_source=data_source,
                campaign_name=campaign_name,
                station_name=station_name,
                product=product,
            )

        # Add fake data files
        filenames = [
            "L0B.1MIN.LOCARNO_2019.61.s20190713134200.e20190714111000.V0.nc",
            "L0B.1MIN.LOCARNO_2019.61.s20190714144200.e20190715111000.V0.nc",
            "L0B.1MIN.LOCARNO_2019.61.s20190715144200.e20190716111000.V0.nc",
            "L0B.1MIN.LOCARNO_2019.61.s20190716144200.e20190717111000.V0.nc",
        ]
        for filename in filenames:
            _ = create_fake_raw_data_file(
                data_archive_dir=data_archive_dir,
                product=product,
                data_source=data_source,
                campaign_name=campaign_name,
                station_name=station_name,
                filename=filename,
            )

        # Test that the function returns the correct number of files in debugging mode
        filepaths = find_files(
            data_archive_dir=data_archive_dir,
            data_source=data_source,
            campaign_name=campaign_name,
            station_name=station_name,
            product=product,
            debugging_mode=True,
        )
        assert len(filepaths) == 3

        # Test that the function returns the correct number of files in normal mode
        filepaths = find_files(
            data_archive_dir=data_archive_dir,
            data_source=data_source,
            campaign_name=campaign_name,
            station_name=station_name,
            product=product,
        )
        assert len(filepaths) == 4

        # Test it does not return other files except netCDFs
        for filename in [".hidden_file", "dummy.log"]:
            _ = create_fake_raw_data_file(
                data_archive_dir=data_archive_dir,
                product=product,
                data_source=data_source,
                campaign_name=campaign_name,
                station_name=station_name,
                filename=filename,
            )
        filepaths = find_files(
            data_archive_dir=data_archive_dir,
            data_source=data_source,
            campaign_name=campaign_name,
            station_name=station_name,
            product=product,
        )
        assert len(filepaths) == 4

        # Test raise error if there is a netCDF file with bad naming
        _ = create_fake_raw_data_file(
            data_archive_dir=data_archive_dir,
            product=product,
            data_source=data_source,
            campaign_name=campaign_name,
            station_name=station_name,
            filename="dummy.nc",
        )
        with pytest.raises(ValueError, match="dummy.nc can not be parsed"):
            find_files(
                data_archive_dir=data_archive_dir,
                data_source=data_source,
                campaign_name=campaign_name,
                station_name=station_name,
                product=product,
            )


####--------------------------------------------------------------------------.
def test_filter_by_time() -> None:
    """Test filter filepaths."""
    filepaths = [
        "L2E.1MIN.LOCARNO_2019.61.s20190713134200.e20190731111000.V0.nc",
        "L2E.1MIN.LOCARNO_2019.61.s20190801001100.e20190831231600.V0.nc",
        "L2E.1MIN.LOCARNO_2019.61.s20200801001100.e20200831231600.V0.nc",
    ]

    out = filter_by_time(
        filepaths=filepaths,
        start_time=datetime.datetime(2019, 1, 1),
        end_time=datetime.datetime(2019, 12, 31, 23, 59, 59),
    )

    assert len(out) == 2

    # Test None filepaths
    res = filter_by_time(
        filepaths=None,
        start_time=datetime.datetime(2019, 1, 1),
        end_time=datetime.datetime(2019, 12, 31, 23, 59, 59),
    )

    assert res == []

    # Test empty filepath list
    res = filter_by_time(
        filepaths=[],
        start_time=datetime.datetime(2019, 1, 1),
        end_time=datetime.datetime(2019, 12, 31, 23, 59, 59),
    )

    assert res == []

    # Test empty start time
    res = filter_by_time(
        filepaths=filepaths,
        start_time=None,
        end_time=datetime.datetime(2019, 12, 31, 23, 59, 59),
    )

    assert len(res) == 2

    # Test empty end time (should default to utcnow which will technically be
    # in the past by the time it gets to the function)
    res = filter_by_time(
        filepaths=filepaths,
        start_time=datetime.datetime(2019, 1, 1),
        end_time=None,
    )
    assert len(res) == 3

    # Test select when time period requested is within file
    res = filter_by_time(
        filepaths=filepaths,
        start_time=datetime.datetime(2019, 7, 14, 0, 0, 20),
        end_time=datetime.datetime(2019, 7, 15, 0, 0, 30),
    )
    assert len(res) == 1
