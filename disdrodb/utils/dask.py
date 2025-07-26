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
"""Utilities for Dask Distributed Computations."""
import logging
import os

import numpy as np


def check_parallel_validity(parallel):
    """Check validity of parallel option given Dask settings."""
    import dask

    scheduler = dask.config.get("scheduler", None)
    if scheduler is None:
        return parallel
    if scheduler in ["synchronous", "threads"]:
        return False
    if scheduler == "distributed":
        from dask.distributed import default_client

        client = default_client()
        info = client.scheduler_info()

        # If ThreadWorker, only 1 pid
        pids = list(client.run(os.getpid).values())
        if len(np.unique(pids)) == 1:
            return False

        # If ProcessWorker
        # - Check single thread per worker to avoid locks
        nthreads_per_process = np.array([v["nthreads"] for v in info["workers"].values()])
        if not np.all(nthreads_per_process == 1):
            print(
                "To open netCDFs in parallel with dask distributed (processes=True), please set threads_per_worker=1 !",
            )
            return False

    # Otherwise let the user choose
    return parallel


def initialize_dask_cluster(minimum_memory=None):
    """Initialize Dask Cluster."""
    import dask
    import psutil

    # Silence dask warnings
    # dask.config.set({"logging.distributed": "error"})
    # Import dask.distributed after setting the config
    from dask.distributed import Client, LocalCluster
    from dask.utils import parse_bytes

    # Set HDF5_USE_FILE_LOCKING to avoid going stuck with HDF
    os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

    # Retrieve the number of processes to run
    available_workers = os.cpu_count() - 2  # if not set, all CPUs minus 2
    num_workers = dask.config.get("num_workers", available_workers)

    # If memory limit specified, ensure correct amount of workers
    if minimum_memory is not None:
        # Compute available memory (in bytes)
        total_memory = psutil.virtual_memory().total
        # Get minimum memory per worker (in bytes)
        minimum_memory = parse_bytes(minimum_memory)
        # Determine number of workers constrained by memory
        maximum_workers_allowed = max(1, total_memory // minimum_memory)
        # Respect both CPU and memory requirements
        num_workers = min(maximum_workers_allowed, num_workers)

    # Create dask.distributed local cluster
    cluster = LocalCluster(
        n_workers=num_workers,
        threads_per_worker=1,
        processes=True,
        # memory_limit='8GB',
        silence_logs=logging.ERROR,
    )
    client = Client(cluster)
    return cluster, client


def close_dask_cluster(cluster, client):
    """Close Dask Cluster."""
    logger = logging.getLogger()
    # Backup current log level
    original_level = logger.level
    logger.setLevel(logging.CRITICAL + 1)  # Set level to suppress all logs
    # Close cluster
    # - Avoid log 'distributed.worker - ERROR - Failed to communicate with scheduler during heartbeat.'
    try:
        cluster.close()
        client.close()
    finally:
        # Restore the original log level
        logger.setLevel(original_level)
