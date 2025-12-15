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
"""Test list utilities."""

from disdrodb.utils.list import flatten_list


def test_flatten_list() -> None:
    """Test flattening nested lists into lists."""
    assert flatten_list([["single item"]]) == ["single item"]
    assert flatten_list([["double", "item"]]) == ["double", "item"]
    assert flatten_list([]) == [], "Empty list should return empty list"
    assert flatten_list(["single item"]) == ["single item"], "Flat list should return same list"
    assert flatten_list(["double", "item"]) == [
        "double",
        "item",
    ], "Flat list should return same list"
