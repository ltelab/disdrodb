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
##################################################
## Wrapper to concat L0B files by command lines ##
##################################################
import sys
import click
from disdrodb.utils.scripts import parse_arg_to_list
from disdrodb.l0.l0_processing import (
    click_l0_stations_options,
    click_l0b_concat_options,
)

sys.tracebacklimit = 0  # avoid full traceback error if occur


@click.command()
@click.argument("disdrodb_dir", metavar="<DISDRODB base directory>")
@click_l0_stations_options
@click_l0b_concat_options
def run_disdrodb_l0b_concat(
    disdrodb_dir,
    data_sources,
    campaign_names,
    station_names,
    remove_l0b=False,
    verbose=True,
):
    """Run the L0B concatenation of available DISDRODB stations.

    This function enable to launch the processing of many DISDRODB stations with a single command.
    From the list of all available DISDRODB stations, it runs the processing of the
    stations matching the provided data_sources, campaign_names and station_names.

    Parameters
    ----------
    disdrodb_dir : str
        Base directory of DISDRODB
        Format: <...>/DISDRODB
    data_sources : str
        Name of data source(s) to process.
        The name(s) must be UPPER CASE.
        If campaign_names and station are not specified, process all stations.
        To specify multiple data sources, write i.e.: --data_sources 'GPM EPFL NCAR'
    campaign_names : str
        Name of the campaign(s) to process.
        The name(s) must be UPPER CASE.
        To specify multiple campaigns, write i.e.: --campaign_names 'IPEX IMPACTS'
    station_names : str
        Station names.
        To specify multiple stations, write i.e.: --station_names 'station1 station2'
    remove_l0b : bool
        If true, remove all source L0B files once L0B concatenation is terminated.
        The default is False.
    verbose : bool
        Whether to print detailed processing information into terminal.
        The default is False.
    """
    from disdrodb.l0.L0B_concat import run_disdrodb_l0b_concat

    # Parse data_sources, campaign_names and station arguments
    data_sources = parse_arg_to_list(data_sources)
    campaign_names = parse_arg_to_list(campaign_names)
    station_names = parse_arg_to_list(station_names)

    # Run concatenation
    run_disdrodb_l0b_concat(
        disdrodb_dir=disdrodb_dir,
        data_sources=data_sources,
        campaign_names=campaign_names,
        station_names=station_names,
        remove_l0b=remove_l0b,
        verbose=verbose,
    )


if __name__ == "__main__":
    run_disdrodb_l0b_concat()
