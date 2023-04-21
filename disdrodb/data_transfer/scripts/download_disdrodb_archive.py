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

from disdrodb.data_transfer.download_data import click_download_option


@click.command()
@click.argument("disdrodb_dir", metavar="<disdrodb_dir>")
@click_download_option
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
