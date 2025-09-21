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
"""Test decorators."""
import importlib

import dask
import pytest
from dask.delayed import Delayed

from disdrodb.utils.decorators import (
    check_pytmatrix_availability,
    check_software_availability,
    delayed_if_parallel,
    single_threaded_if_parallel,
)


def test_check_software_availability_decorator():
    """Test check_software_availability_decorator raise ImportError."""

    @check_software_availability(software="dummy_package", conda_package="dummy_package")
    def dummy_function(a, b=1):
        return a, b

    with pytest.raises(ImportError):
        dummy_function(a=2, b=3)

    @check_software_availability(software="numpy", conda_package="numpy")
    def dummy_function(a, b=1):
        return a, b

    assert dummy_function(2, b=3) == (2, 3)


class TestDelayedIfParallel:
    def test_runs_normally_when_parallel_false(self):
        """It runs the function normally if parallel=False."""

        @delayed_if_parallel
        def dummy_func(x, verbose=True, parallel=False):
            return (x, verbose, parallel)

        result = dummy_func(5, parallel=False, verbose=True)
        assert result == (5, True, False)

    def test_returns_delayed_when_parallel_true(self):
        """It returns a dask.delayed object if parallel=True."""

        @delayed_if_parallel
        def dummy_func(x, verbose=True, parallel=False):
            return (x, verbose, parallel)

        result = dummy_func(5, parallel=True, verbose=True)
        assert isinstance(result, Delayed)
        computed = result.compute()

        assert computed == (5, False, True), "verbose must be forced to False"


class TestSingleThreadedIfParallel:
    def test_runs_normally_when_parallel_false(self):
        """It runs normally if parallel=False."""

        @single_threaded_if_parallel
        def dummy_scheduler_func(parallel=False):
            return dask.config.get("scheduler")

        scheduler = dummy_scheduler_func(parallel=False)
        assert scheduler is None or scheduler in ["threads", "synchronous"]

    def test_runs_with_synchronous_when_parallel_true(self):
        """It forces scheduler='synchronous' if parallel=True."""

        @single_threaded_if_parallel
        def dummy_scheduler_func(parallel=False):
            return dask.config.get("scheduler")

        scheduler = dummy_scheduler_func(parallel=True)
        assert scheduler == "synchronous"


class TestCheckPytmatrixAvailability:
    def test_raises_if_pytmatrix_missing(self, monkeypatch):
        """It raises ImportError if pytmatrix is not installed."""
        monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)

        @check_pytmatrix_availability
        def dummy_func(x):
            return x

        with pytest.raises(ImportError):
            dummy_func(1)

    def test_runs_if_pytmatrix_present(self, monkeypatch):
        """It runs normally if pytmatrix is installed."""
        monkeypatch.setattr(importlib.util, "find_spec", lambda name: object())

        @check_pytmatrix_availability
        def dummy_func(x):
            return x

        assert dummy_func(99) == 99
