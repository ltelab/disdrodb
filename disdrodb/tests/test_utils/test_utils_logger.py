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
"""Test DISDRODB logger utility."""
import logging
import os

import pytest

from disdrodb.utils.logger import (
    close_logger,
    create_file_logger,
    define_summary_log,
    log_debug,
    log_error,
    log_info,
    log_warning,
)


def create_dummy_log_file(filepath, contents):
    """Define helper function to create a dummy log file."""
    with open(filepath, "w") as f:
        f.write(contents)
    return filepath


def test_define_summary_log(tmp_path):
    station_name = "STATION_NAME"
    logs_dir = tmp_path / "PRODUCT" / "logs"
    logs_dir.mkdir(parents=True)

    logs_station_dir = logs_dir / station_name
    logs_station_dir.mkdir(parents=True, exist_ok=True)

    log1_fpath = logs_station_dir / "log1.log"
    log2_fpath = logs_station_dir / "log2.log"

    summary_log_path = logs_dir / f"logs_summary_{station_name}.log"
    problem_log_path = logs_dir / f"logs_problem_{station_name}.log"

    # Create dummy log files
    log_contents1 = (
        "INFO: DUMMY MESSAGE \nProcess has started \nWARNING: Potential issue detected \nNOTHING TO SUMMARIZE \n"
        " Process has ended"
    )
    log_contents2 = "ERROR: Critical failure occurred \n WHATEVER IS IN THE LOG IS COPIED \n "
    log_file1 = create_dummy_log_file(log1_fpath, log_contents1)
    log_file2 = create_dummy_log_file(log2_fpath, log_contents2)

    # Call the function with the list of log files
    list_logs = [str(log_file1), str(log_file2)]
    define_summary_log(list_logs)

    # Check summary log file
    with open(str(summary_log_path)) as f:
        summary_contents = f.read()
    assert "WARNING: Potential issue detected" in summary_contents
    assert "ERROR: Critical failure occurred" in summary_contents
    assert "Process has started" in summary_contents
    assert "Process has ended" in summary_contents
    assert "INFO: DUMMY MESSAGE" not in summary_contents
    assert "NOTHING TO SUMMARIZE" not in summary_contents

    # Check problem log file
    with open(str(problem_log_path)) as f:
        problem_contents = f.read()
    assert "ERROR: Critical failure occurred" in problem_contents
    assert "WHATEVER IS IN THE LOG IS COPIED" in problem_contents
    assert "WARNING: Potential issue detected" not in problem_contents

    # Log file without error is not copied
    assert "Process has started" not in problem_contents
    assert "DUMMY MESSAGE" not in problem_contents


@pytest.fixture
def test_logger():
    logger = logging.getLogger("test_logger")
    logger.setLevel(logging.DEBUG)  # Capture all log levels
    return logger


def test_log_debug(caplog, test_logger, capfd):
    message = "Debug message"
    log_debug(test_logger, message, verbose=True)
    assert caplog.record_tuples == [("test_logger", logging.DEBUG, message)]
    out, _ = capfd.readouterr()
    assert " - Debug message" in out


def test_log_info(caplog, test_logger, capfd):
    message = "Info message"
    log_info(test_logger, message, verbose=True)
    assert caplog.record_tuples == [("test_logger", logging.INFO, message)]
    out, _ = capfd.readouterr()
    assert " - Info message" in out


def test_log_warning(caplog, test_logger, capfd):
    message = "Warning message"
    log_warning(test_logger, message, verbose=True)
    assert caplog.record_tuples == [("test_logger", logging.WARNING, message)]
    out, _ = capfd.readouterr()
    assert " - Warning message" in out


def test_log_error(caplog, test_logger, capfd):
    message = "Error message"
    log_error(test_logger, message, verbose=True)
    assert caplog.record_tuples == [("test_logger", logging.ERROR, message)]
    out, _ = capfd.readouterr()
    assert " - Error message" in out


@pytest.fixture
def log_environment(tmp_path):
    processed_dir = tmp_path / "processed"
    os.makedirs(processed_dir, exist_ok=True)
    product = "test_product"
    station_name = "test_station"
    filename = "test"
    return processed_dir, product, station_name, filename


def test_create_file_logger_paralle_false(log_environment):
    processed_dir, product, station_name, filename = log_environment
    logger = create_file_logger(str(processed_dir), product, station_name, filename, parallel=False)

    assert isinstance(logger, logging.Logger)

    # Check if log file is created
    log_file_path = os.path.join(processed_dir, "logs", product, station_name, f"logs_{filename}.log")
    assert os.path.exists(log_file_path)

    # Test logging
    test_message = "Test logging message"
    logger.info(test_message)

    # Read log file and check if message is there
    with open(log_file_path) as log_file:
        logs = log_file.read()
        assert test_message in logs

    # Close logger
    close_logger(logger)

    # Check if logger is closed
    assert not logger.hasHandlers()


def test_close_logger(log_environment):
    processed_dir, product, station_name, filename = log_environment
    logger = create_file_logger(str(processed_dir), product, station_name, filename, parallel=False)
    close_logger(logger)
    assert not logger.handlers
