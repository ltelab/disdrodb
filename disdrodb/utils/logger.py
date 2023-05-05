#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# -----------------------------------------------------------------------------.
# Copyright (c) 2021-2022 DISDRODB developers
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

from asyncio.log import logger
import os
import time
import logging
import re


def create_l0_logger(processed_dir: str, campaign_name: str, verbose: bool = False) -> logger:
    """Create L0 logger.

    Parameters
    ----------
    processed_dir : str
        Path of the processed directory.
    campaign_name : str
        Campaign name.
    verbose : bool, optional
        Whether to verbose the processing.
        The default is False.

    Returns
    -------
    logger
        Logger object.
    """
    # Define log name
    logger_name = "LO_" + "reader_" + campaign_name
    # Create logs directory
    logs_dir = os.path.join(processed_dir, "logs")
    os.makedirs(logs_dir, exist_ok=True)

    # Create logger
    _create_logger(logs_dir, logger_name)
    logger = logging.getLogger(campaign_name)

    # logger = _create_logger(logs_dir, logger_name)
    # -------------------------------------------------.
    # Update logger
    msg = "### Script started ###"
    if verbose:
        print("\n  " + msg + "\n")
    logger.info(msg)
    # -------------------------------------------------.
    # Return logger
    return logger


def _create_logger(log_dir, logger_name):
    # Define log file filepath
    logger_fname = f'{logger_name}_{time.strftime("%d-%m-%Y_%H-%M-%S")}.log'
    logger_fpath = os.path.join(log_dir, logger_fname)
    # -------------------------------------------------------------------------.
    # Define logger format
    format_type = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    # Define logger level
    level = logging.DEBUG

    # Define logging
    logging.basicConfig(format=format_type, level=level, filename=logger_fpath)

    # Retrieve logger
    # logger = logging.getLogger(logger_fpath)
    # return logger
    return None


def close_logger(logger: logger) -> None:
    """Close the logger

    Parameters
    ----------
    logger : logger
        Logger object.
    """
    if not os.environ.get("PYTEST_CURRENT_TEST"):
        handlers = logger.handlers[:]
        for handler in handlers:
            handler.flush()
            handler.close()
            logger.removeHandler(handler)
        return


def create_file_logger(processed_dir, product_level, station_name, filename, parallel):
    ##------------------------------------------------------------------------.
    # Create logs directory
    if os.environ.get("PYTEST_CURRENT_TEST"):
        return None
    logs_dir = os.path.join(processed_dir, "logs", product_level, station_name)
    os.makedirs(logs_dir, exist_ok=True)

    # logger_fname = f'logs_{fname}_{time.strftime("%d-%m-%Y_%H-%M-%S")}.log'
    logger_fname = f"logs_{filename}.log"
    logger_fpath = os.path.join(logs_dir, logger_fname)

    # -------------------------------------------------------------------------.
    # Set logger (TODO: messy with multiprocess)
    if parallel:
        logger = logging.getLogger(filename)  # does not log submodules logs
    else:
        logger = logging.getLogger()  # root logger (messy with multiprocess)

    handler = logging.FileHandler(logger_fpath, mode="w")
    # handler.setLevel(logging.DEBUG)
    format_type = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    handler.setFormatter(logging.Formatter(format_type))
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    return logger


####---------------------------------------------------------------------------.


def define_summary_log(list_logs):
    """Define station summary log file from list of file logs.

    It select only logged lines with root, WARNING and ERROR keywords.
    It write the summary log in the parent directory.
    """

    if os.environ.get("PYTEST_CURRENT_TEST"):
        return None
    list_logs = sorted(list_logs)
    logs_dir = os.path.dirname(list_logs[0])
    station_name = logs_dir.split(os.path.sep)[-1]
    summary_logs_dir = os.path.dirname(logs_dir)
    ####-----------------------------------------------------------------------.
    #### Define summary and problem logs
    # Define summary logs file name
    summary_fpath = os.path.join(summary_logs_dir, f"logs_summary_{station_name}.log")
    # Define logs keywords to select lines to copy into the summary log file
    # -- > "has started" and "has ended" is used to copy the line with the filename being processsed
    list_keywords = ["has started", "has ended", "WARNING", "ERROR"]  # "DEBUG"
    re_keyword = re.compile("|".join(list_keywords))
    # Filter and concat all logs files
    with open(summary_fpath, "w") as output_file:
        for log_fpath in list_logs:
            with open(log_fpath) as input_file:
                for line in input_file:
                    if re_keyword.search(line):
                        # Write line to output file
                        output_file.write(line)
    ####-----------------------------------------------------------------------.
    #### Define problem logs
    # Define problem logs file name
    problem_fpath = os.path.join(summary_logs_dir, f"logs_problem_{station_name}.log")
    # - Copy the log of files with warnings and error
    list_keywords = ["ERROR"]  # "WARNING"
    re_keyword = re.compile("|".join(list_keywords))
    any_problem = False
    with open(problem_fpath, "w") as output_file:
        for log_fpath in list_logs:
            log_with_problem = False
            # Check if a warning or error is reported
            with open(log_fpath) as input_file:
                for line in input_file:
                    if re_keyword.search(line):
                        log_with_problem = True
                        any_problem = True
                        break
            # If it is reported, copy the log file in the logs_problem file
            if log_with_problem:
                with open(log_fpath) as input_file:
                    output_file.write(input_file.read())

    # If no problems occured, remove the logs_problem_<station_name>.log file
    if not any_problem:
        os.remove(problem_fpath)
    return None


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
    logger.debug(msg)
    if verbose:
        print(" - " + msg)


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
    if not os.environ.get("PYTEST_CURRENT_TEST"):
        logger.info(msg)
        if verbose:
            print(" - " + msg)


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
    logger.warning(msg)
    if verbose:
        print(" - " + msg)


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
    logger.error(msg)
    if verbose:
        print(" - " + msg)
