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
##################################################
## Wrapper to concat L0B files by command lines ##
##################################################
import sys

import click

from disdrodb.l0.routines import click_l0b_concat_options
from disdrodb.utils.scripts import (
    click_base_dir_option,
    click_station_arguments,
    parse_base_dir,
)

sys.tracebacklimit = 0  # avoid full traceback error if occur


@click.command()
@click_station_arguments
@click_l0b_concat_options
@click_base_dir_option
def disdrodb_run_l0b_concat_station(
    # Station arguments
    data_source: str,
    campaign_name: str,
    station_name: str,
    # L0B concat options
    remove_l0b=False,
    verbose=True,
    base_dir: str = None,
):
    """Concatenation all L0B files of a specific DISDRODB station into a single netCDF.

    Parameters
    ----------

    data_source : str
        Institution name (when campaign data spans more than 1 country),
        or country (when all campaigns (or sensor networks) are inside a given country).
        Must be UPPER CASE.
    campaign_name : str
        Campaign name. Must be UPPER CASE.
    station_name : str
        Station name
    remove_l0b : bool
        If true, remove all source L0B files once L0B concatenation is terminated.
        The default is False.
    verbose : bool
        Whether to print detailed processing information into terminal.
        The default is False.
    base_dir : str
        Base directory of DISDRODB
        Format: <...>/DISDRODB
        If not specified, uses path specified in the DISDRODB active configuration.
    """
    from disdrodb.l0.l0_processing import run_l0b_concat_station

    base_dir = parse_base_dir(base_dir)

    run_l0b_concat_station(
        # Station arguments
        data_source=data_source,
        campaign_name=campaign_name,
        station_name=station_name,
        # Processing options
        remove_l0b=remove_l0b,
        verbose=verbose,
        base_dir=base_dir,
    )
