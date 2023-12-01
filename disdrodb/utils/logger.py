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
"""DISDRODB logger utility."""

import logging
import os
import re
from asyncio.log import logger


def create_file_logger(processed_dir, product, station_name, filename, parallel):
    """Create file logger."""
    # Create logs directory
    logs_dir = os.path.join(processed_dir, "logs", product, station_name)
    os.makedirs(logs_dir, exist_ok=True)

    # Define logger filepath
    logger_filename = f"logs_{filename}.log"
    logger_filepath = os.path.join(logs_dir, logger_filename)

    # Set logger
    if parallel:
        logger = logging.getLogger(filename)  # does not log submodules logs
    else:
        logger = logging.getLogger()  # root logger

    handler = logging.FileHandler(logger_filepath, mode="w")
    format_type = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    handler.setFormatter(logging.Formatter(format_type))
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    return logger


def close_logger(logger: logger) -> None:
    """Close the logger

    Parameters
    ----------
    logger : logger
        Logger object.
    """
    handlers = logger.handlers[:]
    for handler in handlers:
        handler.flush()
        handler.close()
        logger.removeHandler(handler)


####---------------------------------------------------------------------------.


def log_debug(logger: logger, msg: str, verbose: bool = False) -> None:
    """Include debug entry into log.

    Parameters
    ----------
    logger : logger
        Log object.
    msg : str
        Message.
    verbose : bool, optional
        Whether to verbose the processing.
        The default is False.
    """
    if verbose:
        print(" - " + msg)
    logger.debug(msg)


def log_info(logger: logger, msg: str, verbose: bool = False) -> None:
    """Include info entry into log.

    Parameters
    ----------
    logger : logger
        Log object.
    msg : str
        Message.
    verbose : bool, optional
        Whether to verbose the processing.
        The default is False.
    """
    if verbose:
        print(" - " + msg)
    logger.info(msg)


def log_warning(logger: logger, msg: str, verbose: bool = False) -> None:
    """Include warning entry into log.

    Parameters
    ----------
    logger : logger
        Log object.
    msg : str
        Message.
    verbose : bool, optional
        Whether to verbose the processing.
        The default is False.
    """
    if verbose:
        print(" - " + msg)
    logger.warning(msg)


def log_error(logger: logger, msg: str, verbose: bool = False) -> None:
    """Include error entry into log.

    Parameters
    ----------
    logger : logger
        Log object.
    msg : str
        Message.
    verbose : bool, optional
        Whether to verbose the processing.
        The default is False.
    """
    if verbose:
        print(" - " + msg)
    logger.error(msg)


def _get_logs_dir(list_logs):
    list_logs = sorted(list_logs)
    station_logs_dir = os.path.dirname(list_logs[0])
    station_name = station_logs_dir.split(os.path.sep)[-1]
    logs_dir = os.path.dirname(station_logs_dir)
    return station_name, logs_dir


def _define_station_summary_log_file(list_logs, summary_filepath):
    # Define logs keywords to select lines to copy into the summary log file
    # -- > "has started" and "has ended" is used to copy the line with the filename being processed
    list_keywords = ["has started", "has ended", "WARNING", "ERROR"]  # "DEBUG"
    re_keyword = re.compile("|".join(list_keywords))
    # Filter and concat all logs files
    with open(summary_filepath, "w") as output_file:
        for log_filepath in list_logs:
            with open(log_filepath) as input_file:
                for line in input_file:
                    if re_keyword.search(line):
                        # Write line to output file
                        output_file.write(line)


def _define_station_problem_log_file(list_logs, problem_filepath):
    # - Copy the log of files with warnings and error
    list_keywords = ["ERROR"]  # "WARNING"
    re_keyword = re.compile("|".join(list_keywords))
    any_problem = False
    with open(problem_filepath, "w") as output_file:
        for log_filepath in list_logs:
            log_with_problem = False
            # Check if an error is reported
            with open(log_filepath) as input_file:
                for line in input_file:
                    if re_keyword.search(line):
                        log_with_problem = True
                        any_problem = True
                        break
            # If it is reported, copy the log file in the logs_problem file
            if log_with_problem:
                with open(log_filepath) as input_file:
                    output_file.write(input_file.read())

    # If no problems occurred, remove the logs_problem_<station_name>.log file
    if not any_problem:
        os.remove(problem_filepath)


def define_summary_log(list_logs):
    """Define a station summary and a problems log file from the list of input logs.

    The summary log select only logged lines with root, WARNING and ERROR keywords.
    The problems log file select only logged lines with the ERROR keyword.
    The two log files are saved in the parent directory of the input list_logs.

    Assume logs to be located at:

        /DISDRODB/Processed/<DATA_SOURCE>/<CAMPAIGN_NAME>/logs/<product>/<station_name>/*.log

    """
    # LogCaptureHandler of pytest does not have baseFilename attribute, so it returns None
    if list_logs[0] is None:
        return None

    station_name, logs_dir = _get_logs_dir(list_logs)

    # Define station summary log file name
    summary_filepath = os.path.join(logs_dir, f"logs_summary_{station_name}.log")
    # Define station problem logs file name
    problem_filepath = os.path.join(logs_dir, f"logs_problem_{station_name}.log")
    # Create station summary log file
    _define_station_summary_log_file(list_logs, summary_filepath)
    # Create station ptoblems log file (if no problems, no file)
    _define_station_problem_log_file(list_logs, problem_filepath)
