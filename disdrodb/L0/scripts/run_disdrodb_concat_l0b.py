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
import click

# -------------------------------------------------------------------------.
# Click Command Line Interface decorator
@click.command()
@click.argument("disdrodb_dir", metavar="<DISDRODB base directory>")
@click.option(
    "-data_sources",
    "--data_sources",
    type=str,
    show_default=True,
    default=None,
    help="Data source names",
)
@click.option(
    "-campaign_names",
    "--campaign_names",
    type=str,
    show_default=True,
    default=None,
    help="Campaign names",
)
@click.option(
    "-station",
    "--station",
    type=str,
    show_default=True,
    default=None,
    help="Station name",
)
@click.option(
    "-r",
    "--remove",
    type=bool,
    show_default=True,
    default=False,
    help="Remove source L0B netCDFs",
)
@click.option(
    "-v",
    "--verbose",
    type=bool,
    show_default=True,
    default=True,
    help="Verbose processing.",
)
def run_concat_cmd(
    disdrodb_dir, data_sources, campaign_names, station, remove, verbose
):
    """Wrapper to run L0B concatenation on all stations (or a subset of it) from the terminal."""
    from disdrodb.L0.L0B_concat import concatenate_L0B

    # Run concatenation
    concatenate_L0B(
        disdrodb_dir=disdrodb_dir,
        data_sources=data_sources,
        campaign_names=campaign_names,
        station=station,
        remove=remove,
        verbose=verbose,
    )


if __name__ == "__main__":
    run_concat_cmd()
