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
"""Wrapper to download stations from the DISDRODB Decentralized Data Archive."""

import sys
from typing import Optional

import click

from disdrodb.data_transfer.download_data import click_download_archive_options, click_download_options
from disdrodb.utils.cli import (
    click_data_archive_dir_option,
    click_metadata_archive_dir_option,
    parse_archive_dir,
    parse_arg_to_list,
)

sys.tracebacklimit = 0  # avoid full traceback error if occur


@click.command()
@click_download_archive_options
@click_data_archive_dir_option
@click_metadata_archive_dir_option
@click_download_options
def disdrodb_download_archive(
    data_sources: Optional[str] = None,
    campaign_names: Optional[str] = None,
    station_names: Optional[str] = None,
    data_archive_dir: Optional[str] = None,
    metadata_archive_dir: Optional[str] = None,
    force: bool = False,
):
    """Download DISDRODB stations with the ``disdrodb_data_url`` in the metadata.

    Parameters
    ----------
    data_sources : str or list of str, optional
        Data source name (eg : EPFL).
        If not provided (``None``), all data sources will be downloaded.
        The default value is ``data_source=None``.
    campaign_names : str or list of str, optional
        Campaign name (eg :  EPFL_ROOF_2012).
        If not provided (``None``), all campaigns will be downloaded.
        The default value is ``campaign_name=None``.
    station_names : str or list of str, optional
        Station name.
        If not provided (``None``), all stations will be downloaded.
        The default value is ``station_name=None``.
    force : bool, optional
        If ``True``, overwrite the already existing raw data file.
        The default value is ``False``.
    data_archive_dir : str (optional)
        DISDRODB Data Archive directory. Format: ``<...>/DISDRODB``.
        If ``None`` (the default), the disdrodb config variable ``data_archive_dir`` is used.
    """
    from disdrodb.data_transfer.download_data import download_archive

    data_archive_dir = parse_archive_dir(data_archive_dir)
    metadata_archive_dir = parse_archive_dir(metadata_archive_dir)
    data_sources = parse_arg_to_list(data_sources)
    campaign_names = parse_arg_to_list(campaign_names)
    station_names = parse_arg_to_list(station_names)

    download_archive(
        data_archive_dir=data_archive_dir,
        metadata_archive_dir=metadata_archive_dir,
        data_sources=data_sources,
        campaign_names=campaign_names,
        station_names=station_names,
        force=force,
    )
