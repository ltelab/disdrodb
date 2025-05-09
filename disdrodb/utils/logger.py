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


def create_logger_file(logs_dir, filename, parallel):
    """Create logger file."""
    # Create logs directory
    os.makedirs(logs_dir, exist_ok=True)

    # Define logger filepath
    logger_filename = f"logs_{filename}.log"
    logger_filepath = os.path.join(logs_dir, logger_filename)

    # Set logger
    # - getLogger() # root logger
    # - getLogger(filename) does not log submodules logs
    logger = logging.getLogger(filename) if parallel else logging.getLogger()

    handler = logging.FileHandler(logger_filepath, mode="w")
    format_type = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    handler.setFormatter(logging.Formatter(format_type))
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)

    # Define logger filepath
    # - LogCaptureHandler of pytest does not have baseFilename attribute --> So set None
    logger_filepath = logger.handlers[0].baseFilename if not os.environ.get("PYTEST_CURRENT_TEST") else None
    return logger, logger_filepath


def close_logger(logger) -> None:
    """Close the logger.

    Parameters
    ----------
    logger : logging.Logger
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
    logger : logging.Logger
        Log object.
    msg : str
        Message.
    verbose : bool, optional
        Whether to verbose the processing.
        The default value is ``False``.
    """
    if verbose:
        print(" - " + msg)
    if logger is not None:
        logger.debug(msg)


def log_info(logger: logger, msg: str, verbose: bool = False) -> None:
    """Include info entry into log.

    Parameters
    ----------
    logger : logging.Logger
        Log object.
    msg : str
        Message.
    verbose : bool, optional
        Whether to verbose the processing.
        The default value is ``False``.
    """
    if verbose:
        print(" - " + msg)
    if logger is not None:
        logger.info(msg)


def log_warning(logger: logger, msg: str, verbose: bool = False) -> None:
    """Include warning entry into log.

    Parameters
    ----------
    logger : logging.Logger
        Log object.
    msg : str
        Message.
    verbose : bool, optional
        Whether to verbose the processing.
        The default value is ``False``.
    """
    if verbose:
        print(" - " + msg)
    if logger is not None:
        logger.warning(msg)


def log_error(logger: logger, msg: str, verbose: bool = False) -> None:
    """Include error entry into log.

    Parameters
    ----------
    logger : logging.Logger
        Log object.
    msg : str
        Message.
    verbose : bool, optional
        Whether to verbose the processing.
        The default value is ``False``.
    """
    if verbose:
        print(" - " + msg)
    if logger is not None:
        logger.error(msg)


####---------------------------------------------------------------------------.
#### SUMMARY LOGS


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
    list_patterns = ["ValueError: Less than 5 timesteps available for day"]
    re_keyword = re.compile("|".join(list_keywords))
    # Compile patterns to ignore, escaping any special regex characters
    re_patterns = re.compile("|".join(map(re.escape, list_patterns))) if list_patterns else None
    # Initialize problem log file
    any_problem = False
    n_files = len(list_logs)
    n_files_with_problems = 0
    with open(problem_filepath, "w") as output_file:
        # Loop over log files and collect problems
        for log_filepath in list_logs:
            log_with_problem = False
            # Check if an error is reported
            with open(log_filepath) as input_file:
                for line in input_file:
                    if re_keyword.search(line):
                        # If the line matches an ignore pattern, skip it
                        if re_patterns and re_patterns.search(line):
                            continue
                        log_with_problem = True
                        n_files_with_problems += 1
                        any_problem = True
                        break
            # If it is reported, copy the log file in the logs_problem file
            if log_with_problem:
                with open(log_filepath) as input_file:
                    output_file.write(input_file.read())

        # Add number of files with problems
        msg = f"SUMMARY: {n_files_with_problems} of {n_files} files had problems."
        output_file.write(msg)

    # If no problems occurred, remove the logs_problem_<station_name>.log file
    if not any_problem:
        os.remove(problem_filepath)


def create_product_logs(
    product,
    data_source,
    campaign_name,
    station_name,
    data_archive_dir=None,
    # Logs list
    list_logs=None,  # If none, list it !
    # Product options
    **product_kwargs,
):
    """Create station summary and station problems log files.

    The summary log selects only logged lines with ``root``, ``WARNING``, and ``ERROR`` keywords.
    The problems log file selects only logged lines with the ``ERROR`` keyword.

    The logs directory structure is the follow:
    /logs
    - /files/<product_acronym>/<station> (same structure as data ... a log for each processed file)
    - /summary
      -->  SUMMARY.<PRODUCT_ACRONYM>.<CAMPAIGN_NAME>.<STATION_NAME>.log
    - /problems
      -->  PROBLEMS.<PRODUCT_ACRONYM>.<CAMPAIGN_NAME>.<STATION_NAME>.log

    Parameters
    ----------
    product : str
        The DISDRODB product.
    data_source : str
        The data source name.
    campaign_name : str
        The campaign name.
    station_name : str
        The station name.
    data_archive_dir : str, optional
        The base directory path. Default is None.
    sample_interval : str, optional
        The sample interval for L2E option. Default is None.
    rolling : str, optional
        The rolling option for L2E. Default is None.
    model_name : str, optional
        The model name for L2M. Default is None.
    list_logs : list, optional
        List of log file paths. If None, the function will list the log files.

    Returns
    -------
    None

    """
    from disdrodb.api.path import define_campaign_dir, define_filename, define_logs_dir
    from disdrodb.utils.directories import list_files

    # --------------------------------------------------------.
    # Search for logs file
    if list_logs is None:
        # Define product logs directory within /files/....
        logs_dir = define_logs_dir(
            product=product,
            data_archive_dir=data_archive_dir,
            data_source=data_source,
            campaign_name=campaign_name,
            station_name=station_name,
            # Product options
            **product_kwargs,
        )
        list_logs = list_files(logs_dir, glob_pattern="*", recursive=True)

    # --------------------------------------------------------.
    # LogCaptureHandler of pytest does not have baseFilename attribute, so it returns None
    if list_logs[0] is None:
        return

    # --------------------------------------------------------.
    # Define /summary and /problem directory
    campaign_dir = define_campaign_dir(
        archive_dir=data_archive_dir,
        product=product,
        data_source=data_source,
        campaign_name=campaign_name,
    )
    logs_summary_dir = os.path.join(campaign_dir, "logs", "summary")
    logs_problem_dir = os.path.join(campaign_dir, "logs", "problems")

    os.makedirs(logs_summary_dir, exist_ok=True)
    os.makedirs(logs_problem_dir, exist_ok=True)

    # --------------------------------------------------------.
    # Define station summary log file name
    summary_filename = define_filename(
        product=product,
        campaign_name=campaign_name,
        station_name=station_name,
        # Filename options
        add_version=False,
        add_time_period=False,
        add_extension=False,
        prefix="SUMMARY",
        suffix="log",
        # Product options
        **product_kwargs,
    )
    summary_filepath = os.path.join(logs_summary_dir, summary_filename)

    # Define station problem logs file name
    problem_filename = define_filename(
        product=product,
        campaign_name=campaign_name,
        station_name=station_name,
        # Filename options
        add_version=False,
        add_time_period=False,
        add_extension=False,
        prefix="PROBLEMS",
        suffix="log",
        # Product options
        **product_kwargs,
    )
    problem_filepath = os.path.join(logs_problem_dir, problem_filename)

    # --------------------------------------------------------.
    # Create summary log file
    _define_station_summary_log_file(list_logs, summary_filepath)

    # Create problem log file (if no problems, no file created)
    _define_station_problem_log_file(list_logs, problem_filepath)

    # --------------------------------------------------------.
    # Remove /problem directory if empty !
    if len(os.listdir(logs_problem_dir)) == 0:
        os.rmdir(logs_problem_dir)
