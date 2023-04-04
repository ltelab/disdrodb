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

from disdrodb.data_transfer import download_data


@click.command()
@click.option(
    "--disdrodb_dir",
    required=True,
    help="Raw data folder path (eg : /home/user/DISDRODB/Raw). Is compulsory.",
)
@click.option(
    "--data_source",
    help="Data source folder name (eg : EPFL). If not provided (None), all data sources will be downloaded.",
)
@click.option(
    "--campaign_name",
    help="Name of the campaign (eg :  EPFL_ROOF_2012). If not provided (None), all campaigns will be downloaded.",
)
@click.option(
    "--station_name",
    help="Station name. If not provided (None), all stations will be downloaded.",
)
@click.option("--overwrite", type=bool, help="Overwite existing file ?")
def wraper_download_disdrodb_archive(
    disdrodb_dir=None,
    data_source=None,
    campaign_name=None,
    station_name=None,
    overwrite=False,
):
    download_data.download_disdrodb_archives(disdrodb_dir, data_source, campaign_name, station_name, overwrite)
