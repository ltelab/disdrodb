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
"""Routine to open the DISDRODB Metadata Data Archive."""

import sys
from typing import Optional

import click

from disdrodb.utils.cli import (
    click_metadata_archive_dir_option,
    parse_archive_dir,
)

sys.tracebacklimit = 0  # avoid full traceback error if occur


@click.command()
@click_metadata_archive_dir_option
def disdrodb_open_metadata_archive(
    metadata_archive_dir: Optional[str] = None,
):
    """Open the DISDRODB Metadata Archive directory in the system file explorer.

    Opens the metadata archive directory using the system's default file manager,
    allowing you to browse station metadata files and YAML configurations.

    \b
    Archive Directory:
        --metadata_archive_dir: Custom path to DISDRODB metadata archive
        If not specified, the path from the active DISDRODB configuration is used

    \b
    Examples:
        # Open the metadata archive from active configuration
        disdrodb_open_metadata_archive

        # Open a custom metadata archive directory
        disdrodb_open_metadata_archive --metadata_archive_dir /path/to/DISDRODB-METADATA/DISDRODB

    \b
    Important Notes:
        - Opens the directory in your system's default file manager
        - The metadata archive contains station YAML files and issue reports
        - Useful for manual inspection and verification of station metadata
    """  # noqa: D301
    from disdrodb.api.io import open_metadata_archive

    metadata_archive_dir = parse_archive_dir(metadata_archive_dir)

    open_metadata_archive(metadata_archive_dir=metadata_archive_dir)
