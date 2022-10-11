#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 23 18:44:52 2022

@author: ghiggi
"""
import os
import yaml
import logging

logger = logging.getLogger(__name__)


def _write_issue_timestamps_docs(f):
    """Provide template for timestamps.yml"""
    # TODO: Kim adapt to desired format
    f.write(
        "# This file is used to store dates to drop by the reader, the time format used is the isoformat (YYYY-mm-dd HH:MM:SS). \n"
    )
    f.write("# timestamp: list of timestamps \n")
    f.write("# time_period: list of list ranges of dates \n")
    f.write("# Example: \n")
    f.write(
        "# timestamp: ['2018-12-07 14:15','2018-12-07 14:17','2018-12-07 14:19', '2018-12-07 14:25'] \n"
    )
    f.write("# time_period: [['2018-08-01 12:00:00', '2018-08-01 14:00:00'], \n")
    f.write("#               ['2018-08-01 15:44:30', '2018-08-01 15:59:31'], \n")
    f.write("#               ['2018-08-02 12:44:30', '2018-08-02 12:59:31']] \n")


def create_issue_yml(
    fpath: str, timestamp: str = None, time_period: str = None
) -> None:
    """Create issue YAML file.

    Parameters
    ----------
    fpath : str
        timestamps.yml file path.
    timestamp : str, optional
        Timestamp, by default None
    time_period : str, optional
        Timeperiod, by default None
    """

    logger.info(f"Creating issue YAML file at {fpath}")
    with open(fpath, "w") as f:
        # Write template for timestamps.yml
        _write_issue_timestamps_docs(f)

        # Write timestamp if provided
        # TODO: Kim adapt to desired format
        if timestamp is not None:
            timestamp_dict = {"timestamp": timestamp}
            yaml.dump(timestamp_dict, f, default_flow_style=None)

        # Write timestamp if provided
        # TODO: Kim adapt to desired format
        if time_period is not None:
            time_period_dict = {"time_period": time_period}
            yaml.dump(time_period_dict, f, default_flow_style=None)


def read_issue(raw_dir: str, station_id: str) -> dict:
    """Read YAML issue file.

    Parameters
    ----------
    processed_dir : str
        Path of the processed directory
    station_id : int
        Id of the station.

    Returns
    -------
    dict
        Issue dictionary.
    """

    issue_fpath = os.path.join(raw_dir, "issue", station_id + ".yml")
    with open(issue_fpath, "r") as f:
        issue_dict = yaml.safe_load(f)
    # issue_dict = check_issue_compliance(issue_dict)
    return issue_dict


def check_issue_compliance(fpath: str) -> None:
    """Check issue compliance

    Parameters
    ----------
    fpath : str
        File pazh


    """
    pass
