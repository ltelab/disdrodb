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
"""Routine to download the DISDRODB Metadata Data Archive."""
import sys

import click

from disdrodb.utils.cli import parse_archive_dir

sys.tracebacklimit = 0  # avoid full traceback error if occur


@click.command()
@click.argument("directory_path", metavar="<station>")
@click.option("-f", "--force", type=bool, show_default=True, default=False, help="Force overwriting")
def disdrodb_download_metadata_archive(
    directory_path,
    force: bool = False,
):
    """Download the DISDRODB Metadata Archive to the specified directory.

    Parameters
    ----------
    directory_path : str
        The directory path where the DISDRODB-METADATA directory will be downloaded.
    force : bool, optional
         If ``True``, the existing DISDRODB-METADATA directory will be removed
         and a new one will be downloaded. The default value is ``False``.

    Returns
    -------
    metadata_archive_dir
        The DISDRODB Metadata Archive directory path.
    """
    from disdrodb import download_metadata_archive

    directory_path = parse_archive_dir(directory_path)

    download_metadata_archive(directory_path, force=force)
