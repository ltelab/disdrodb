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
"""Utilities for Dask Distributed computations."""
import logging
import os


def initialize_dask_cluster():
    """Initialize Dask Cluster."""
    import dask
    from dask.distributed import Client, LocalCluster

    # Set HDF5_USE_FILE_LOCKING to avoid going stuck with HDF
    os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
    # Retrieve the number of process to run
    available_workers = os.cpu_count() - 2  # if not set, all CPUs
    num_workers = dask.config.get("num_workers", available_workers)
    # Silence dask warnings
    dask.config.set({"logging.distributed": "error"})
    # dask.config.set({"distributed.admin.system-monitor.gil.enabled": False})
    # Create dask.distributed local cluster
    cluster = LocalCluster(
        n_workers=num_workers,
        threads_per_worker=1,
        processes=True,
        # memory_limit='8GB',
        # silence_logs=False,
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
