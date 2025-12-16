# -----------------------------------------------------------------------------.
# Copyright (c) 2021-2026 DISDRODB developers
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
from pathlib import Path

import click

from disdrodb.utils.cli import parse_archive_dir

sys.tracebacklimit = 0  # avoid full traceback error if occur


@click.command()
@click.argument("directory_path", required=False, metavar="[directory]", type=click.Path())
@click.option("-f", "--force", type=bool, show_default=True, default=False, help="Force overwriting")
def disdrodb_download_metadata_archive(
    directory_path,
    force: bool = False,
):
    """Download the DISDRODB Metadata Archive to the specified directory.

    \b
    Download Options:
        '--force True' removes the existing DISDRODB-METADATA directory and forces re-download.
        The default is --force False. If the DISDRODB-METADATA directory already exists, it raises an error.

    \b
    Examples:
        # Download metadata archive to current directory
        disdrodb_download_metadata_archive

        # Download to specific directory
        disdrodb_download_metadata_archive /path/to/directory

        # Force re-download of existing metadata archive
        disdrodb_download_metadata_archive /path/to/directory --force True

    \b
    Important Notes:
        - Use --force with caution as it will delete the existing metadata archive
    """  # noqa: D301
    from disdrodb import download_metadata_archive

    # Default to current directory if none provided
    directory_path = Path(directory_path or ".").resolve()
    directory_path = parse_archive_dir(str(directory_path))

    # Download metadata archive
    download_metadata_archive(directory_path, force=force)
