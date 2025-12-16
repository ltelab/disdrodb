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
    # --> If DASK_NUM_WORKERS is not set, use all CPUs minus 2
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
        memory_limit=0,  # this avoid flexible dask memory management
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


def _batch_iterable(iterable, n):
    """Yield successive n-sized chunks from iterable."""
    for i in range(0, len(iterable), n):
        yield iterable[i : i + n]


def execute_tasks_safely(list_tasks, parallel: bool, logs_dir: str, max_tasks_per_batch=5_000):
    """
    Execute Dask tasks and skip failed ones.

    Parameters
    ----------
    list_tasks : list
        List of dask delayed objects or results.
    parallel : bool
        Whether to execute in parallel with Dask or not.
    logs_dir : str
        Directory to store FAILED_TASKS.log.
    max_tasks_per_batch : int or None, optional
     Maximum number of tasks to submit to `client.compute()` at once.
     The default is 5000. Dask struggle if more than 10_000 tasks are submitted.

    Returns
    -------
    list_logs : list
        List of task results. For failed tasks, adds the path
        to FAILED_TASKS.log in place of the result.
    """
    from dask.distributed import get_client

    if not parallel:
        # Non-parallel mode: just return results directly
        return list_tasks

    # Ensure logs_dir exists
    os.makedirs(logs_dir, exist_ok=True)

    # Define file name where to log failed dask tasks
    failed_log_path = os.path.join(logs_dir, "FAILED_DASK_TASKS.log")

    # Ensure we have a Dask client
    try:
        client = get_client()
    except ValueError:
        raise ValueError("No Dask Distributed Client found.")

    all_results = []
    failed_futures = []

    # Batch execution
    task_batches = list(_batch_iterable(list_tasks, max_tasks_per_batch)) if max_tasks_per_batch else [list_tasks]

    for batch in task_batches:
        # Compute tasks (all concurrently)
        # - Runs tasks == num_workers * threads_per_worker (which is 1 for DISDRODB)
        # - If errors occurs in some, skip it
        futures = client.compute(batch)
        results = client.gather(futures, errors="skip")

        # Identify and collect failed futures
        batch_failed = [f for f in futures if f.status != "finished"]
        failed_futures.extend(batch_failed)

        # Collect results from successful tasks
        all_results.extend(results)

    # If no tasks failed, return results
    if not failed_futures:
        return all_results

    # Otherwise define log file listing failed tasks
    with open(failed_log_path, "w") as f:
        for fut in failed_futures:
            err = fut.exception()
            f.write(f"ERROR - DASK TASK FAILURE - Task {fut.key} failed: {err}\n")

    # Append to list of log filepaths (results) the dask failing log
    all_results.append(failed_log_path)
    return all_results
