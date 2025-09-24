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
"""This module contains functions for manipulating dictionaries."""


def extract_product_kwargs(kwargs, product):
    """Infer product kwargs dictionary."""
    from disdrodb.api.checks import check_product
    from disdrodb.constants import PRODUCTS_ARGUMENTS

    check_product(product)
    product_kwargs_keys = set(PRODUCTS_ARGUMENTS.get(product, []))
    return extract_dictionary(kwargs, keys=product_kwargs_keys)


def extract_dictionary(dictionary, keys):
    """Extract a subset of keys from the dictionary, removing them from the input dictionary."""
    return {k: dictionary.pop(k) for k in keys if k in dictionary}
