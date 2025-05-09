#!/usr/bin/env python3

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
"""Routine to print the DISDRODB Data Archive directory."""
import sys

import click

sys.tracebacklimit = 0  # avoid full traceback error if occur


@click.command()
def disdrodb_data_archive_directory():
    """Print the DISDRODB Data Archive directory."""
    import disdrodb

    print("The DISDRODB Data Archive Directory is: ", disdrodb.get_data_archive_dir(None))
