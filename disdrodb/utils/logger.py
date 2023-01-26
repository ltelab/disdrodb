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


def create_l0_logger(
    processed_dir: str, campaign_name: str, verbose: bool = False
) -> logger:
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
    # Retrieve logger
    logger = logging.getLogger(campaign_name)
    # Update logger
    msg = "### Script started ###"
    if verbose:
        print("\n  " + msg + "\n")
    logger.info(msg)
    # Return logger
    return logger


def _create_logger(log_dir, logger_name):
    # Define log file filepath
    logger_fname = f'{time.strftime("%d-%m-%Y_%H-%M-%S")}_{logger_name}.log'
    logger_fpath = os.path.join(log_dir, logger_fname)
    # -------------------------------------------------------------------------.
    # Define logger format
    format_type = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    # Define logger level
    level = logging.DEBUG

    # Define logging
    logging.basicConfig(format=format_type, level=level, filename=logger_fpath)
    return None


def close_logger(logger: logger) -> None:
    """Close the logger

    Parameters
    ----------
    logger : logger
        Logger object.
    """
    handlers = logger.handlers[:]
    for handler in handlers:
        handler.close()
        logger.removeHandler(handler)
    return


# ----------------------------
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
