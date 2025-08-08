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

from disdrodb.api.path import define_campaign_dir, define_logs_dir
from disdrodb.constants import ARCHIVE_VERSION
from disdrodb.utils.logger import (
    close_logger,
    create_logger_file,
    create_product_logs,
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


def test_create_product_logs(tmp_path):
    test_data_archive_dir = tmp_path / "data" / "DISDRODB"
    data_source = "DATA_SOURCE"
    campaign_name = "CAMPAIGN_NAME"
    station_name = "STATION_NAME"
    product = "L0A"

    # Define directory where logs files are saved
    logs_dir = define_logs_dir(
        product=product,
        data_archive_dir=test_data_archive_dir,
        data_source=data_source,
        campaign_name=campaign_name,
        station_name=station_name,
    )
    os.makedirs(logs_dir, exist_ok=True)

    # Define paths of logs files
    log1_fpath = os.path.join(logs_dir, "log1.log")
    log2_fpath = os.path.join(logs_dir, "log2.log")

    # Define /summary and /problem directory
    campaign_dir = define_campaign_dir(
        archive_dir=test_data_archive_dir,
        product=product,
        data_source=data_source,
        campaign_name=campaign_name,
    )
    logs_summary_dir = os.path.join(campaign_dir, "logs", "summary")
    logs_problem_dir = os.path.join(campaign_dir, "logs", "problems")

    # Define summary and problem filepath
    summary_log_path = os.path.join(logs_summary_dir, f"SUMMARY.{product}.{campaign_name}.{station_name}.log")
    problem_log_path = os.path.join(logs_problem_dir, f"PROBLEMS.{product}.{campaign_name}.{station_name}.log")

    ####-------------------------------------.
    # Create dummy log files
    log_contents1 = (
        "INFO: DUMMY MESSAGE \nProcess has started \nWARNING: Potential issue detected \nNOTHING TO SUMMARIZE \n"
        " Process has ended \n"
    )
    log_contents2 = "ERROR: Critical failure occurred \n WHATEVER IS IN THE LOG IS COPIED \n "
    log_file1 = create_dummy_log_file(log1_fpath, log_contents1)
    log_file2 = create_dummy_log_file(log2_fpath, log_contents2)

    # Call the function with the list of log files
    list_logs = [str(log_file1), str(log_file2)]
    create_product_logs(
        product=product,
        data_source=data_source,
        campaign_name=campaign_name,
        station_name=station_name,
        data_archive_dir=test_data_archive_dir,
        # Logs list
        list_logs=list_logs,
    )

    # Check summary log file
    with open(str(summary_log_path)) as f:
        summary_contents = f.read()

    assert "Process has started" in summary_contents
    assert "Process has ended" in summary_contents
    assert "WARNING: Potential issue detected" in summary_contents
    assert "ERROR: Critical failure occurred" in summary_contents

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


def test_define_summary_log_when_no_problems(tmp_path):
    """Test that not problem log file is created if no errors occurs."""
    test_data_archive_dir = tmp_path / "data" / "DISDRODB"
    data_source = "DATA_SOURCE"
    campaign_name = "CAMPAIGN_NAME"
    station_name = "STATION_NAME"
    product = "L0A"

    # Define directory where logs files are saved
    logs_dir = define_logs_dir(
        product=product,
        data_archive_dir=test_data_archive_dir,
        data_source=data_source,
        campaign_name=campaign_name,
        station_name=station_name,
    )
    os.makedirs(logs_dir, exist_ok=True)

    # Define paths of logs files
    log1_fpath = os.path.join(logs_dir, "log1.log")
    log2_fpath = os.path.join(logs_dir, "log2.log")

    # Define /summary and /problem directory
    campaign_dir = define_campaign_dir(
        archive_dir=test_data_archive_dir,
        product=product,
        data_source=data_source,
        campaign_name=campaign_name,
    )
    logs_summary_dir = os.path.join(campaign_dir, "logs", "summary")
    logs_problem_dir = os.path.join(campaign_dir, "logs", "problems")

    # Define summary and problem filepath
    summary_log_path = os.path.join(logs_summary_dir, f"SUMMARY.{product}.{campaign_name}.{station_name}.log")
    problem_log_path = os.path.join(logs_problem_dir, f"PROBLEMS.{product}.{campaign_name}.{station_name}.log")

    ####-------------------------------------.
    # Check that if no problems, the problems log is not created
    log_contents1 = "INFO: DUMMY MESSAGE \nProcess has started \n Process has ended  \n"
    log_contents2 = "INFO: DUMMY MESSAGE \nProcess has started \n Process has ended  \n"
    log_file1 = create_dummy_log_file(log1_fpath, log_contents1)
    log_file2 = create_dummy_log_file(log2_fpath, log_contents2)
    list_logs = [str(log_file1), str(log_file2)]  # noqa

    # List logs direc
    create_product_logs(
        product=product,
        data_source=data_source,
        campaign_name=campaign_name,
        station_name=station_name,
        data_archive_dir=test_data_archive_dir,
        list_logs=None,  # search for logs based on inputs
    )

    assert os.path.exists(summary_log_path)
    assert not os.path.exists(problem_log_path)


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
    campaign_dir = tmp_path / ARCHIVE_VERSION
    os.makedirs(campaign_dir, exist_ok=True)
    product = "test_product"
    station_name = "test_station"
    filename = "test"
    return campaign_dir, product, station_name, filename


def test_create_logger_file_paralle_false(log_environment):
    campaign_dir, product, station_name, filename = log_environment
    logs_dir = os.path.join(str(campaign_dir), "logs", product, station_name)
    logger, logger_filepath = create_logger_file(logs_dir, filename, parallel=False)

    assert isinstance(logger, logging.Logger)

    # Check if log file is created
    log_file_path = os.path.join(campaign_dir, "logs", product, station_name, f"logs_{filename}.log")
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
    campaign_dir, product, station_name, filename = log_environment
    logs_dir = os.path.join(str(campaign_dir), "logs", product, station_name)
    logger, logger_filepath = create_logger_file(logs_dir, filename, parallel=False)
    close_logger(logger)
    assert not logger.handlers
