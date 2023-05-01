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
from disdrodb.l0.l0_processing import (
    click_l0_station_arguments,
    click_l0b_concat_options,
)

sys.tracebacklimit = 0  # avoid full traceback error if occur


@click.command()
@click_l0_station_arguments
@click_l0b_concat_options
def run_disdrodb_l0b_concat_station(
    # Station arguments
    disdrodb_dir,
    data_source,
    campaign_name,
    station_name,
    # L0B concat options
    remove_l0b=False,
    verbose=True,
):
    """Concatenation all L0B files of a specific DISDRODB station into a single netCDF.

    Parameters
    ----------
    disdrodb_dir : str
        Base directory of DISDRODB
        Format: <...>/DISDRODB
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
    """
    from disdrodb.api.io import _get_disdrodb_directory

    from disdrodb.l0.l0b_concat import _concatenate_netcdf_files

    # Retrieve processed_dir
    processed_dir = _get_disdrodb_directory(
        disdrodb_dir=disdrodb_dir,
        product_level="L0B",
        data_source=data_source,
        campaign_name=campaign_name,
        check_exist=True,
    )

    # Run concatenation
    _concatenate_netcdf_files(
        processed_dir=processed_dir,
        station_name=station_name,
        remove=remove_l0b,
        verbose=verbose,
    )

    return None


if __name__ == "__main__":
    run_disdrodb_l0b_concat_station()
