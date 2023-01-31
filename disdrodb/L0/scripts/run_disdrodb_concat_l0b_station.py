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
@click.argument("data_source", metavar="<data_source>")
@click.argument("campaign_name", metavar="<campaign>")
@click.argument("station", metavar="<station>")
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
def run_concat_cmd(disdrodb_dir, data_source, campaign_name, station, remove, verbose):
    """Wrapper to run concatenation of a single station L0B files from the terminal."""
    from disdrodb.L0.L0B_concat import _concatenate_L0B_station

    # Run concatenation
    _concatenate_L0B_station(
        disdrodb_dir=disdrodb_dir,
        data_source=data_source,
        campaign_name=campaign_name,
        station=station,
        remove=remove,
        verbose=verbose,
    )


if __name__ == "__main__":
    run_concat_cmd()
