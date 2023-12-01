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
"""Check DISDRODB configuration files."""
import os

import pytest


def test_disdrodb_config_takes_environment_variable():
    os.environ["DISDRODB_BASE_DIR"] = "MY_BASE_DIR"

    import disdrodb

    assert disdrodb.config.get("base_dir") == "MY_BASE_DIR"


@pytest.mark.parametrize("key", ["base_dir", "zenodo_token", "zenodo_sandbox_token"])
def test_disdrodb_config_donfig(key):
    import disdrodb

    # Assert donfig key context manager
    with disdrodb.config.set({key: "dummy_string"}):
        assert disdrodb.config.get(key) == "dummy_string"

    # # Assert if not initialized, defaults to None
    # assert disdrodb.config.get(key) is None

    # Now initialize
    disdrodb.config.set({key: "dummy_string"})
    assert disdrodb.config.get(key) == "dummy_string"

    # Now try context manager again
    with disdrodb.config.set({key: "new_dummy_string"}):
        assert disdrodb.config.get(key) == "new_dummy_string"
    assert disdrodb.config.get(key) == "dummy_string"
