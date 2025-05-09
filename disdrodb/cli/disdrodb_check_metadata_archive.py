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
"""Wrapper to check DISDRODB Metadata Archive Compliance from terminal."""
import sys

import click

from disdrodb.utils.cli import click_metadata_archive_dir_option, parse_archive_dir

sys.tracebacklimit = 0  # avoid full traceback error if occur


@click.command()
@click_metadata_archive_dir_option
@click.option(
    "--raise_error",
    type=bool,
    show_default=True,
    default=True,
    help="Whether to raise error of finish the check",
)
def disdrodb_check_metadata_archive(metadata_archive_dir=None, raise_error=True):
    """Run the DISDRODB Metadata Archive Checks."""
    from disdrodb.metadata.checks import check_metadata_archive

    metadata_archive_dir = parse_archive_dir(metadata_archive_dir)
    check_metadata_archive(metadata_archive_dir=metadata_archive_dir, raise_error=raise_error)
