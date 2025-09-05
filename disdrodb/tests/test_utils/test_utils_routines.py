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
from datetime import datetime

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from disdrodb.utils.routines import is_possible_product, run_product_generation


class TestIsPossibleProduct:
    """Test suite for is_possible_product function."""

    def test_false_when_rolling_and_equal_intervals(self):
        """Should return False if rolling is True and accumulation == sample interval."""
        assert is_possible_product(accumulation_interval=60, sample_interval=60, rolling=True) is False

    def test_false_when_accumulation_smaller_than_sample(self):
        """Should return False if accumulation interval < sample interval."""
        assert is_possible_product(accumulation_interval=30, sample_interval=60, rolling=False) is False

    def test_false_when_not_multiple(self):
        """Should return False if accumulation interval is not multiple of sample interval."""
        assert is_possible_product(accumulation_interval=70, sample_interval=60, rolling=False) is False

    @pytest.mark.parametrize("rolling", [True, False])
    def test_true_when_multiple_and_valid(self, rolling):
        """Should return True if accumulation is a multiple of sample interval and rules are satisfied."""
        # Example: accumulation 120 is multiple of sample interval 60
        # - if rolling=True, it's still valid because intervals are not equal
        assert is_possible_product(accumulation_interval=120, sample_interval=60, rolling=rolling) is True


class TestRunProductGeneration:
    """Integration-like tests for run_product_generation."""

    def test_success_with_dataset(self, tmp_path):
        """Should run successfully with an xarray.Dataset and return None under pytest."""
        logs_dir = tmp_path / "logs"
        logs_dir.mkdir()

        def core_func(logger):
            # Create minimal dataset with a time coordinate (needed for define_file_folder_path)
            times = np.array([np.datetime64("2023-01-01T00:00:00")])
            ds = xr.Dataset({"var": ("time", [1])}, coords={"time": times})
            return ds

        logger_filepath = run_product_generation(
            product="L0A",
            logs_dir=str(logs_dir),
            logs_filename="test_dataset",
            parallel=False,
            verbose=False,
            folder_partitioning="",  # no partitioning for simplicity
            core_func=core_func,
            core_func_kwargs={},
            pass_logger=True,
        )

        # Under pytest, no file is actually created
        assert logger_filepath is None

    def test_success_with_dataframe(self, tmp_path):
        """Should run successfully with a pandas.DataFrame and return None under pytest."""
        logs_dir = tmp_path / "logs"
        logs_dir.mkdir()

        def core_func(logger):
            df = pd.DataFrame({"time": [datetime(2023, 1, 1)], "value": [42]})
            return df

        logger_filepath = run_product_generation(
            product="L0B",
            logs_dir=str(logs_dir),
            logs_filename="test_dataframe",
            parallel=False,
            verbose=False,
            folder_partitioning="",  # no partitioning for simplicity
            core_func=core_func,
            core_func_kwargs={},
            pass_logger=True,
        )
        assert logger_filepath is None

    def test_failure_core_func(self, tmp_path):
        """Should log error and return None when core_func raises an exception."""
        logs_dir = tmp_path / "logs"
        logs_dir.mkdir()

        def failing_core_func():
            raise RuntimeError("boom")

        logger_filepath = run_product_generation(
            product="L0C",
            logs_dir=str(logs_dir),
            logs_filename="test_failure",
            parallel=False,
            verbose=False,
            folder_partitioning="",
            core_func=failing_core_func,
            core_func_kwargs={},
        )

        # Even when failing, the wrapper should complete cleanly
        assert logger_filepath is None
