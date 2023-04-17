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
####################################################################
## Wrapper to download disdrodb archives by command lines ##
####################################################################
import click


@click.command()
@click.option(
    "--disdrodb_dir",
    required=True,
    type=str,
    show_default=True,
    default="",
    help="Raw data folder path (eg : /home/user/DISDRODB/Raw). Is compulsory.",
)
@click.option(
    "--data_sources",
    multiple=True,
    type=str,
    show_default=True,
    default=[],
    help="""Data source folder name (eg : EPFL). If not provided (None),
    all data sources will be downloaded.
    Many data sources can be provided by duplicating the argument.
    """,
)
@click.option(
    "--campaign_names",
    multiple=True,
    type=str,
    show_default=True,
    default=[],
    help="""Name of the campaign (eg :  EPFL_ROOF_2012).
    If not provided (None), all campaigns will be downloaded.
    Many campaign names can be provided by duplicating the argument.
    """,
)
@click.option(
    "--station_names",
    multiple=True,
    type=str,
    show_default=True,
    default=[],
    help="""Station name. If not provided (None), all stations will be downloaded.
    Many station names can be provided by duplicating the argument.
    """,
)
@click.option("--force", type=bool, help="Overwite existing file ?, default is True")
def download_disdrodb_archive(
    disdrodb_dir=None,
    data_sources=None,
    campaign_names=None,
    station_names=None,
    force=True,
):
    from disdrodb.data_transfer.download_data import download_disdrodb_archives
    from disdrodb.utils.scripts import parse_arg_to_list

    data_sources = parse_arg_to_list(data_sources)
    campaign_names = parse_arg_to_list(campaign_names)
    station_names = parse_arg_to_list(station_names)

    download_disdrodb_archives(disdrodb_dir, data_sources, campaign_names, station_names, force)
