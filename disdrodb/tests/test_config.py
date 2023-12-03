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


def test_define_disdrodb_configs(tmp_path, mocker):
    import disdrodb

    # Mock to save config YAML at custom location
    mocker.patch("disdrodb.configs._define_config_filepath", return_value=str(tmp_path / ".config_disdrodb.yml"))

    # Define config YAML
    disdrodb.configs.define_disdrodb_configs(base_dir="test_dir", zenodo_token="test_token")
    assert os.path.exists(tmp_path / ".config_disdrodb.yml")


def test_read_disdrodb_configs(tmp_path, mocker):
    from disdrodb.configs import define_disdrodb_configs, read_disdrodb_configs

    # Mock to save config YAML at custom location
    mocker.patch("disdrodb.configs._define_config_filepath", return_value=str(tmp_path / ".config_disdrodb.yml"))

    # Define config YAML
    define_disdrodb_configs(base_dir="test_dir", zenodo_token="test_token")
    assert os.path.exists(tmp_path / ".config_disdrodb.yml")

    # Read config YAML
    config_dict = read_disdrodb_configs()
    assert isinstance(config_dict, dict)


def test_update_disdrodb_configs(tmp_path, mocker):
    import disdrodb
    from disdrodb.utils.yaml import read_yaml

    # Mock to save config YAML at custom location
    config_fpath = str(tmp_path / ".config_disdrodb.yml")
    mocker.patch("disdrodb.configs._define_config_filepath", return_value=config_fpath)

    # Initialize
    disdrodb.configs.define_disdrodb_configs(base_dir="test_dir/DISDRODB", zenodo_token="test_token")
    assert os.path.exists(config_fpath)

    config_dict = read_yaml(config_fpath)
    assert config_dict["base_dir"] == "test_dir/DISDRODB"

    # Update
    disdrodb.configs.define_disdrodb_configs(base_dir="new_test_dir/DISDRODB", zenodo_sandbox_token="new_token")
    assert os.path.exists(config_fpath)
    config_dict = read_yaml(config_fpath)
    assert config_dict["base_dir"] == "new_test_dir/DISDRODB"
    assert config_dict["zenodo_token"] == "test_token"
    assert config_dict["zenodo_sandbox_token"] == "new_token"


def test_get_base_dir():
    import disdrodb
    from disdrodb.configs import get_base_dir

    # Check that if input is not None, return the specified base_dir
    assert get_base_dir(base_dir="test/DISDRODB") == "test/DISDRODB"

    # Check that if no config YAML file specified (base_dir=None), raise error
    with disdrodb.config.set({"base_dir": None}):
        with pytest.raises(ValueError):
            get_base_dir()

    # Set base_dir in the donfig config and check it return it !
    disdrodb.config.set({"base_dir": "another_test_dir/DISDRODB"})
    assert get_base_dir() == "another_test_dir/DISDRODB"

    # Now test that return the one from the temporary disdrodb.config donfig object
    with disdrodb.config.set({"base_dir": "new_test_dir/DISDRODB"}):
        assert get_base_dir() == "new_test_dir/DISDRODB"

    # And check it return the default one
    assert get_base_dir() == "another_test_dir/DISDRODB"


@pytest.mark.parametrize("sandbox,expected_token", [(False, "my_zenodo_token"), (True, "my_sandbox_zenodo_token")])
def test_get_zenodo_token(sandbox, expected_token):
    import disdrodb
    from disdrodb.configs import get_zenodo_token

    disdrodb.config.set({"zenodo_token": None})
    disdrodb.config.set({"zenodo_sandbox_token": None})

    # Check raise error if Zenodo Token  not specified
    with pytest.raises(ValueError):
        get_zenodo_token(sandbox=sandbox)

    disdrodb.config.set({"zenodo_token": "my_zenodo_token"})
    disdrodb.config.set({"zenodo_sandbox_token": "my_sandbox_zenodo_token"})
    assert get_zenodo_token(sandbox=sandbox) == expected_token
