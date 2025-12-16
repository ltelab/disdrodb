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
"""Check DISDRODB configuration files."""

import os  # noqa

import pytest
from unittest import mock


def test_disdrodb_config_takes_environment_variable():
    from importlib import reload

    import disdrodb

    data_archive_dir = os.path.join("my_path_to", "DISDRODB")
    with mock.patch.dict("os.environ", {"DISDRODB_DATA_ARCHIVE_DIR": data_archive_dir}):
        reload(disdrodb._config)
        reload(disdrodb)
        assert disdrodb.config.get("data_archive_dir") == data_archive_dir


def test_disdrodb_config_takes_config_YAML(tmp_path, mocker):
    from importlib import reload

    import disdrodb

    # Mock to save config YAML at custom location
    config_fpath = str(tmp_path / ".config_disdrodb.yml")
    mocker.patch("disdrodb.configs._define_config_filepath", return_value=config_fpath)

    # Initialize config YAML
    data_archive_dir = os.path.join("my_path_to", "DISDRODB")
    disdrodb.configs.define_configs(data_archive_dir=data_archive_dir, zenodo_token="test_token")

    reload(disdrodb._config)
    reload(disdrodb)
    assert disdrodb.config.get("data_archive_dir") == data_archive_dir


@pytest.mark.parametrize("key", ["data_archive_dir", "metadata_archive_dir", "zenodo_token", "zenodo_sandbox_token"])
def test_disdrodb_config_donfig(key):
    import disdrodb

    expected_key = os.path.join("dummy_path", "DISDRODB")
    # Assert donfig key context manager
    with disdrodb.config.set({key: expected_key}):
        assert disdrodb.config.get(key) == expected_key

    # # Assert if not initialized, defaults to None
    # assert disdrodb.config.get(key) is None

    # Now initialize
    disdrodb.config.set({key: expected_key})
    assert disdrodb.config.get(key) == expected_key

    # Now try context manager again
    new_expected_key = os.path.join("new_dummy_path", "DISDRODB")
    with disdrodb.config.set({key: new_expected_key}):
        assert disdrodb.config.get(key) == new_expected_key
    assert disdrodb.config.get(key) == expected_key
