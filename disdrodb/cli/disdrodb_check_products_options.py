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
"""Script to check the validity of the DISDRODB products configuration files."""
import sys

import click

from disdrodb.utils.cli import parse_empty_string_and_none

sys.tracebacklimit = 0  # avoid full traceback error if occur

# -------------------------------------------------------------------------.
# Click Command Line Interface decorator


@click.command()
@click.option(
    "--products_configs_dir",
    type=str,
    show_default=True,
    default=None,
    help="Directory with DISDRODB products configurations files",
)
def disdrodb_check_products_options(products_configs_dir):
    """Validate the DISDRODB products configuration files."""
    from disdrodb.routines.options_validation import validate_products_configurations

    products_configs_dir = parse_empty_string_and_none(products_configs_dir)

    validate_products_configurations(products_configs_dir)
