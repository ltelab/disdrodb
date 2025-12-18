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
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
# -----------------------------------------------------------------------------.
"""Test dask utility."""

import logging
import os
from tempfile import TemporaryDirectory

import pytest
from dask import delayed

from disdrodb.utils.dask import (
    check_parallel_validity,
    close_dask_cluster,
    execute_tasks_safely,
    initialize_dask_cluster,
)


@pytest.fixture(autouse=True)
def reset_dask_scheduler():
    """Force all tests to run with scheduler=None."""
    import dask

    with dask.config.set(scheduler=None):
        yield


def test_check_parallel_validity_no_scheduler(reset_dask_scheduler):
    """When no scheduler is set, it should return the parallel flag unchanged."""
    assert check_parallel_validity(parallel=True) is True
    assert check_parallel_validity(parallel=False) is False


@pytest.mark.parametrize("scheduler", ["synchronous", "threads"])
def test_check_parallel_validity_threads_and_sync(scheduler, reset_dask_scheduler):
    """Always returns False when scheduler is synchronous or threads."""
    import dask

    with dask.config.set({"scheduler": scheduler}):
        assert check_parallel_validity(parallel=True) is False
        assert check_parallel_validity(parallel=False) is False


def test_check_parallel_validity_distributed_thread_workers(reset_dask_scheduler):
    """Distributed cluster with ThreadPool workers should disable parallel computing."""
    from dask.distributed import Client, LocalCluster

    with LocalCluster(n_workers=1, threads_per_worker=2, processes=False) as cluster, Client(cluster):
        result = check_parallel_validity(parallel=True)
        assert result is False  # because all workers share one PID


def test_check_parallel_validity_distributed_process_workers(reset_dask_scheduler):
    """Distributed cluster with Process workers and 1 thread per worker should allow parallel computing."""
    from dask.distributed import Client, LocalCluster

    with LocalCluster(n_workers=2, threads_per_worker=2, processes=True) as cluster, Client(cluster):
        result = check_parallel_validity(parallel=True)
        assert result is False  # because although separate processes, more than 1 thread each


def test_check_parallel_validity_distributed_process_workers_2(reset_dask_scheduler):
    """Distributed cluster with Process workers and 1 thread per worker should allow parallel computing."""
    import dask

    with dask.config.set({"scheduler": "distributed"}):
        cluster, client = initialize_dask_cluster()
        result = check_parallel_validity(parallel=True)
        assert result is True  # because all workers share one PID
        close_dask_cluster(cluster, client)


class TestInitializeDaskCluster:
    """Test initialize_dask_cluster."""

    def test_initialize_and_close_cluster(self, reset_dask_scheduler):
        """Test basic initialization and closing of dask cluster."""
        cluster, client = initialize_dask_cluster()
        try:
            # Basic checks
            assert cluster.scheduler_address
            assert client.status == "running"
            assert os.environ.get("HDF5_USE_FILE_LOCKING") == "FALSE"
        finally:
            close_dask_cluster(cluster, client)
            # After closing, the client should not be running
            assert client.status != "running"

    def test_initialize_with_memory_constraint(self, reset_dask_scheduler):
        """Test setting an excessive memory enforce 1 worker."""
        # Request a huge memory requirement → should fallback to at least 1 worker
        cluster, client = initialize_dask_cluster(minimum_memory="10TB")
        try:
            assert len(cluster.workers) == 1
        finally:
            close_dask_cluster(cluster, client)

    def test_close_dask_cluster_restores_log_level(self, reset_dask_scheduler):
        """Test that closing dask cluster restores original log level."""
        cluster, client = initialize_dask_cluster()
        logger = logging.getLogger()
        original_level = logger.level
        close_dask_cluster(cluster, client)
        assert logger.level == original_level


def test_execute_tasks_safely_with_failures(reset_dask_scheduler):
    """Test execute_tasks_safely."""
    from dask.distributed import Client

    # Dummy functions
    @delayed
    def succeed(x):
        return x * 2

    @delayed
    def fail(x):
        raise ValueError(f"Bad input: {x}")

    # Create tasks
    list_tasks = [succeed(2), fail(3, dask_key_name="fail-my_custom_name"), succeed(5)]

    with TemporaryDirectory() as tmpdir:  # noqa: SIM117
        # Create a local client
        with Client(n_workers=2, threads_per_worker=1, processes=True, memory_limit="512MB"):
            # Run tasks
            results = execute_tasks_safely(list_tasks, parallel=True, logs_dir=tmpdir)

            # Check results length matches tasks
            assert len(results) == 3

            # Assert first list tasks succeeded
            assert results[0] == 4
            assert results[1] == 10

            # Assert last element is the filepath to the file log
            assert "FAILED_DASK_TASKS.log" in results[2]

            # Open log file containing failures info
            with open(results[2]) as f:
                log_content = f.read()

            # Test reports name of the dask task failing
            assert "my_custom_name" in log_content

            # Test reports cause of error
            assert "Bad input: 3" in log_content

            # Test that if only successful tasks, no FAILED_DASK_TASKS.log added
            list_tasks = [succeed(2), succeed(5)]
            results = execute_tasks_safely(list_tasks, parallel=True, logs_dir=tmpdir)
            assert len(results) == 2
            assert results[0] == 4
            assert results[1] == 10
