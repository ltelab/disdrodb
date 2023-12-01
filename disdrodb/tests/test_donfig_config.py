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
import os  # noqa

import pytest
from unittest import mock


def test_disdrodb_config_takes_environment_variable():
    from importlib import reload

    import disdrodb

    with mock.patch.dict("os.environ", {"DISDRODB_BASE_DIR": "/my_path_to/DISDRODB"}):
        reload(disdrodb._config)
        reload(disdrodb)
        assert disdrodb.config.get("base_dir") == "/my_path_to/DISDRODB"


def test_disdrodb_config_takes_config_YAML(tmp_path, mocker):
    from importlib import reload

    import disdrodb

    # Mock to save config YAML at custom location
    config_fpath = str(tmp_path / ".config_disdrodb.yml")
    mocker.patch("disdrodb.configs._define_config_filepath", return_value=config_fpath)

    # Initialize config YAML
    disdrodb.configs.define_disdrodb_configs(base_dir="test_dir/DISDRODB", zenodo_token="test_token")

    reload(disdrodb._config)
    reload(disdrodb)
    assert disdrodb.config.get("base_dir") == "test_dir/DISDRODB"


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
