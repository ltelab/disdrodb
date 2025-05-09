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
"""DISDRODB command-line-interface scripts utilities."""

import click


def _execute_cmd(cmd, raise_error=False):
    """Execute command in the terminal, streaming output in python console."""
    from subprocess import PIPE, CalledProcessError, Popen

    with Popen(cmd, shell=True, stdout=PIPE, bufsize=1, universal_newlines=True) as p:
        for line in p.stdout:
            print(line, end="")

    # Raise error if command didn't run successfully
    if p.returncode != 0 and raise_error:
        raise CalledProcessError(p.returncode, p.args)


def _parse_empty_string_and_none(args):
    """Utility to parse argument passed from the command line.

    If ``args = ''``, returns None.
    If ``args = 'None'`` returns None.
    Otherwise return ``args``.
    """
    # If '', set to 'None'
    args = None if args == "" else args
    # - If multiple arguments, split by space
    if isinstance(args, str) and args == "None":
        args = None
    return args


def parse_arg_to_list(args):
    """Utility to pass list to command line scripts.

    If ``args = ''`` returns ``None``.
    If ``args = 'None'`` returns ``None``.
    If ``args = 'variable'`` returns ``[variable]``.
    If ``args = 'variable1 variable2'`` returns ``[variable1, variable2]``.
    """
    # If '' or 'None' --> Set to None
    args = _parse_empty_string_and_none(args)
    # - If multiple arguments, split by space
    if isinstance(args, str):
        # - Split by space
        list_args = args.split(" ")
        # - Remove '' (deal with multi space)
        args = [args for args in list_args if len(args) > 0]
    return args


def parse_archive_dir(archive_dir: str):
    """Utility to parse archive directories provided by command line.

    If ``archive_dir = 'None'`` returns ``None``.
    If ``archive_dir = ''`` returns ``None``.
    """
    # If '', set to 'None'
    return _parse_empty_string_and_none(archive_dir)


def click_station_arguments(function: object):
    """Click command line arguments for DISDRODB station processing.

    Parameters
    ----------
    function : object
        Function.
    """
    function = click.argument("station_name", metavar="<station>")(function)
    function = click.argument("campaign_name", metavar="<CAMPAIGN_NAME>")(function)
    function = click.argument("data_source", metavar="<DATA_SOURCE>")(function)
    return function


def click_data_archive_dir_option(function: object):
    """Click command line argument for DISDRODB ``data_archive_dir``.

    Parameters
    ----------
    function : object
        Function.
    """
    function = click.option(
        "--data_archive_dir",
        type=str,
        show_default=True,
        default=None,
        help="DISDRODB base directory",
    )(function)
    return function


def click_metadata_archive_dir_option(function: object):
    """Click command line argument for DISDRODB ``metadata_archive_dir``.

    Parameters
    ----------
    function : object
        Function.
    """
    function = click.option(
        "--metadata_archive_dir",
        type=str,
        show_default=True,
        default=None,
        help="DISDRODB Metadata Archive Directory",
    )(function)
    return function


def click_stations_options(function: object):
    """Click command line options for DISDRODB archive L0 processing.

    Parameters
    ----------
    function : object
        Function.
    """
    function = click.option(
        "--data_sources",
        type=str,
        show_default=True,
        default="",
        help="DISDRODB data sources to process",
    )(function)
    function = click.option(
        "--campaign_names",
        type=str,
        show_default=True,
        default="",
        help="DISDRODB campaign names to process",
    )(function)
    function = click.option(
        "--station_names",
        type=str,
        show_default=True,
        default="",
        help="DISDRODB station names to process",
    )(function)
    return function


def click_processing_options(function: object):
    """Click command line default parameters for L0 processing options.

    Parameters
    ----------
    function : object
        Function.
    """
    function = click.option(
        "-p",
        "--parallel",
        type=bool,
        show_default=True,
        default=False,
        help="Process files in parallel",
    )(function)
    function = click.option(
        "-d",
        "--debugging_mode",
        type=bool,
        show_default=True,
        default=False,
        help="Switch to debugging mode",
    )(function)
    function = click.option("-v", "--verbose", type=bool, show_default=True, default=True, help="Verbose")(function)
    function = click.option(
        "-f",
        "--force",
        type=bool,
        show_default=True,
        default=False,
        help="Force overwriting",
    )(function)
    return function


def click_remove_l0a_option(function: object):
    """Click command line argument for ``remove_l0a``."""
    function = click.option(
        "--remove_l0a",
        type=bool,
        show_default=True,
        default=False,
        help="If true, remove the L0A files once the L0B processing is terminated.",
    )(function)
    return function


def click_remove_l0b_option(function: object):
    """Click command line argument for ``remove_l0b``."""
    function = click.option(
        "--remove_l0b",
        type=bool,
        show_default=True,
        default=False,
        help="If true, remove the L0B files once the L0C processing is terminated.",
    )(function)
    return function


def click_l0_archive_options(function: object):
    """Click command line arguments for L0 processing archiving of a station.

    Parameters
    ----------
    function : object
        Function.
    """
    function = click.option(
        "--remove_l0b",
        type=bool,
        show_default=True,
        default=False,
        help="If true, remove all source L0B files once L0B concatenation is terminated.",
    )(function)
    function = click.option(
        "--remove_l0a",
        type=bool,
        show_default=True,
        default=False,
        help="If true, remove the L0A files once the L0B processing is terminated.",
    )(function)
    function = click.option(
        "-l0c",
        "--l0c_processing",
        type=bool,
        show_default=True,
        default=True,
        help="Perform L0C processing.",
    )(function)
    function = click.option(
        "-l0b",
        "--l0b_processing",
        type=bool,
        show_default=True,
        default=True,
        help="Perform L0B processing.",
    )(function)
    function = click.option(
        "-l0a",
        "--l0a_processing",
        type=bool,
        show_default=True,
        default=True,
        help="Perform L0A processing.",
    )(function)
    return function
