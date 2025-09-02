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
"""Utilities for DISDRODB processing routines."""
import os
import shutil
import tempfile

from disdrodb.api.path import define_file_folder_path
from disdrodb.utils.logger import (
    close_logger,
    create_logger_file,
    log_error,
    log_info,
)


def run_product_generation(
    product: str,
    logs_dir: str,
    logs_filename: str,
    parallel: bool,
    verbose: bool,
    folder_partitioning: str,
    core_func: callable,
    core_func_kwargs: dict,
    pass_logger=False,
):
    """
    Generic wrapper for DISDRODB product generation.

    Parameters
    ----------
    product : str
        Product name (e.g., "L0A", "L0B", ...).

    logs_dir : str
        Logs directory.
    logs_filename : str
        Logs filename.
    parallel : bool
        Parallel flag (for logger).
    verbose : bool
        Verbose logging flag.
    folder_partitioning : str
        Partitioning scheme.
    core_func : callable
        Function with signature `core_func(logger)` that does the product-specific work.
        Must return an xarray.Dataset or pandas.DataFrame (used to determine log subdir).
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        # Initialize log file
        logger, tmp_logger_filepath = create_logger_file(
            logs_dir=tmpdir,
            filename=logs_filename,
            parallel=parallel,
        )

        # Inform that product creation has started
        log_info(logger, f"{product} processing of {logs_filename} has started.", verbose=verbose)

        # Initialize object
        obj = None  # if None, means the product creation failed

        # Add logger to core_func_kwargs if specified
        if pass_logger:
            core_func_kwargs["logger"] = logger

        # Try product creation
        try:
            # Run product creation
            obj = core_func(**core_func_kwargs)

            # Inform that product creation has ended
            log_info(logger, f"{product} processing of {logs_filename} has ended.", verbose=verbose)

        # Report error if the case
        except Exception as e:
            log_error(logger, f"{type(e).__name__}: {e}", verbose=verbose)

        finally:
            # Close logger
            close_logger(logger)

            # Move log file to final logs directory
            success_flag = obj is not None
            if success_flag:
                logs_dir = define_file_folder_path(obj, dir_path=logs_dir, folder_partitioning=folder_partitioning)
            os.makedirs(logs_dir, exist_ok=True)
            if tmp_logger_filepath is not None:  # (when running pytest, tmp_logger_filepath is None)
                logger_filepath = os.path.join(logs_dir, os.path.basename(tmp_logger_filepath))
                shutil.move(tmp_logger_filepath, logger_filepath)
            else:
                logger_filepath = None

            # Free memory
            del obj

    # Return logger filepath
    return logger_filepath
