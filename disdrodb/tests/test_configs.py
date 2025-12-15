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
import os

import pytest

from disdrodb.configs import (
    copy_default_products_configs,
    define_configs,
    get_data_archive_dir,
    get_default_products_configs_dir,
    get_folder_partitioning,
    get_metadata_archive_dir,
    get_products_configs_dir,
    get_scattering_table_dir,
    get_zenodo_token,
    read_configs,
)
from disdrodb.utils.yaml import read_yaml


@pytest.fixture
def temporary_home(tmp_path, monkeypatch):
    """Force _define_config_filepath to point into tmp_path."""
    from disdrodb import configs

    config_path = tmp_path / ".config_disdrodb.yml"
    monkeypatch.setattr(configs, "_define_config_filepath", lambda: str(config_path))
    return tmp_path


@pytest.fixture(autouse=True)
def reset_disdrodb_config():
    """Reset disdrodb.config before and after each test to avoid leakage."""
    import disdrodb

    disdrodb.config.clear()
    yield
    disdrodb.config.clear()


class TestDefineConfigs:

    def test_define_configs_creates_file(self, temporary_home, tmp_path):
        """Test define_configs writes a config YAML file."""
        test_dir = str(temporary_home / "DISDRODB-METADATA" / "DISDRODB")
        os.makedirs(test_dir, exist_ok=True)

        define_configs(metadata_archive_dir=test_dir, zenodo_token="test_token")
        assert (temporary_home / ".config_disdrodb.yml").exists()

        config_dict = read_yaml(temporary_home / ".config_disdrodb.yml")
        assert "data_archive_dir" not in config_dict  # unspecified keys are not present
        assert config_dict["metadata_archive_dir"] == test_dir
        assert config_dict["zenodo_token"] == "test_token"

    def test_define_full_configs_file(self, temporary_home, tmp_path):
        """Test define_configs writes a config YAML file."""
        # Define valid config arguments
        data_archive_dir = str(tmp_path / "DISDRODB-METADATA" / "DISDRODB")
        os.makedirs(data_archive_dir, exist_ok=True)
        metadata_archive_dir = str(tmp_path / "DATA" / "DISDRODB")
        os.makedirs(metadata_archive_dir, exist_ok=True)
        scattering_table_dir = str(tmp_path / "scattering")
        os.makedirs(scattering_table_dir, exist_ok=True)
        folder_partitioning = "year"
        zenodo_token = "dummy"
        zenodo_sandbox_token = "dummy"
        products_configs_dir = str(tmp_path / "disdrodb_product_configs")

        define_configs(
            metadata_archive_dir=metadata_archive_dir,
            data_archive_dir=data_archive_dir,
            scattering_table_dir=scattering_table_dir,
            products_configs_dir=products_configs_dir,
            folder_partitioning=folder_partitioning,
            zenodo_token=zenodo_token,
            zenodo_sandbox_token=zenodo_sandbox_token,
        )
        assert (temporary_home / ".config_disdrodb.yml").exists()

        config_dict = read_yaml(temporary_home / ".config_disdrodb.yml")
        assert config_dict["data_archive_dir"] == data_archive_dir
        assert config_dict["metadata_archive_dir"] == metadata_archive_dir
        assert config_dict["zenodo_token"] == zenodo_token
        assert config_dict["zenodo_sandbox_token"] == zenodo_sandbox_token
        assert config_dict["scattering_table_dir"] == scattering_table_dir
        assert config_dict["products_configs_dir"] == products_configs_dir
        assert config_dict["folder_partitioning"] == folder_partitioning

    def test_update_configs_preserves_and_adds(self, temporary_home):
        """Test updating configs keeps old values and adds new ones."""
        config_path = temporary_home / ".config_disdrodb.yml"
        data_archive_dir = str(temporary_home / "DATA" / "DISDRODB")
        new_data_archive_dir = str(temporary_home / "NEW_DATA" / "DISDRODB")

        os.makedirs(data_archive_dir, exist_ok=True)
        os.makedirs(new_data_archive_dir, exist_ok=True)

        # Initial write
        define_configs(data_archive_dir=data_archive_dir, zenodo_token="test_token")
        config_dict = read_yaml(config_path)
        assert config_dict["data_archive_dir"] == data_archive_dir
        assert config_dict["zenodo_token"] == "test_token"

        # Update
        define_configs(data_archive_dir=new_data_archive_dir, zenodo_sandbox_token="new_token")
        config_dict = read_yaml(config_path)
        assert config_dict["data_archive_dir"] == new_data_archive_dir
        assert config_dict["zenodo_token"] == "test_token"
        assert config_dict["zenodo_sandbox_token"] == "new_token"


class TestReadConfigs:

    def test_read_configs_raise_error_if_missing_file(self, temporary_home):
        """Test reading config raises ValueError if file is missing."""
        with pytest.raises(ValueError):
            read_configs()

    def test_read_configs(self, temporary_home):
        """Test reading configs."""
        # Define config YAML
        define_configs(data_archive_dir=os.path.join("test_path_to", "DISDRODB"), zenodo_token="test_token")

        # Read config YAML
        config_dict = read_configs()
        assert isinstance(config_dict, dict)
        assert config_dict["zenodo_token"] == "test_token"


class TestGetDataArchiveDir:

    def test_direct_argument(self, tmp_path):
        """Test returning data archive dir when passed directly."""
        test_dir = str(tmp_path / "DATA" / "DISDRODB")
        os.makedirs(test_dir, exist_ok=True)

        # Check that if input is not None, return the specified data_archive_dir
        assert get_data_archive_dir(data_archive_dir=test_dir) == test_dir

    def test_from_config(self, tmp_path):
        """Test returning metadata archive dir from config object."""
        import disdrodb

        test_dir = str(tmp_path / "DATA" / "DISDRODB")
        os.makedirs(test_dir, exist_ok=True)
        disdrodb.config.set({"data_archive_dir": test_dir})
        assert get_data_archive_dir() == test_dir

    def test_missing_config_raises(self):
        """Test raising ValueError when metadata archive dir missing."""
        import disdrodb

        with disdrodb.config.set({"data_archive_dir": None}), pytest.raises(ValueError):
            get_data_archive_dir()

    def test_temporary_override(self, tmp_path):
        """Test temporary override of data archive dir with context manager."""
        import disdrodb

        test_dir = str(tmp_path / "DATA" / "DISDRODB")
        os.makedirs(test_dir, exist_ok=True)

        # Set data_archive_dir in the donfig config and check it return it !
        disdrodb.config.set({"data_archive_dir": test_dir})
        assert get_data_archive_dir() == test_dir

        # Now test that return the one from the temporary disdrodb.config donfig object
        new_test_dir = str(tmp_path / "NEW_DATA" / "DISDRODB")
        os.makedirs(new_test_dir, exist_ok=True)

        with disdrodb.config.set({"data_archive_dir": new_test_dir}):
            assert get_data_archive_dir() == new_test_dir

        # And check it return the default one
        assert get_data_archive_dir() == test_dir

    def test_raise_error_if_no_path_specified(self, temporary_home):
        """Test define_configs writes a config YAML file."""
        metadata_dir = str(temporary_home / "DISDRODB-METADATA" / "DISDRODB")
        os.makedirs(metadata_dir, exist_ok=True)

        define_configs(metadata_archive_dir=metadata_dir)
        assert (temporary_home / ".config_disdrodb.yml").exists()

        # Test raise error because data_archive not specified
        with pytest.raises(ValueError):
            get_data_archive_dir()

    def test_raise_error_if_no_config_file_specified(self, temporary_home):
        """Test define_configs writes a config YAML file."""
        # Test raise error because the config file is not specified
        with pytest.raises(ValueError):
            get_data_archive_dir()


class TestGetMetadataArchiveDir:

    def test_direct_argument(self, tmp_path):
        """Test returning metadata archive dir when passed directly."""
        test_dir = str(tmp_path / "DISDRODB-METADATA" / "DISDRODB")
        os.makedirs(test_dir, exist_ok=True)
        assert get_metadata_archive_dir(metadata_archive_dir=test_dir) == test_dir

    def test_from_config(self, tmp_path):
        """Test returning metadata archive dir from config object."""
        import disdrodb

        test_dir = str(tmp_path / "DISDRODB-METADATA" / "DISDRODB")
        os.makedirs(test_dir, exist_ok=True)
        disdrodb.config.set({"metadata_archive_dir": test_dir})
        assert get_metadata_archive_dir() == test_dir

    def test_missing_config_raises(self):
        """Test raising ValueError when metadata archive dir missing."""
        import disdrodb
        from disdrodb.configs import get_metadata_archive_dir

        with disdrodb.config.set({"metadata_archive_dir": None}), pytest.raises(ValueError):
            get_metadata_archive_dir()


@pytest.mark.parametrize(
    ("sandbox", "expected_token"),
    [(False, "my_zenodo_token"), (True, "my_sandbox_zenodo_token")],
)
def test_get_zenodo_token(sandbox, expected_token):
    """Test retrieving Zenodo and Sandbox tokens with fallback errors."""
    import disdrodb

    disdrodb.config.set({"zenodo_token": None, "zenodo_sandbox_token": None})
    with pytest.raises(ValueError):
        get_zenodo_token(sandbox=sandbox)

    disdrodb.config.set(
        {
            "zenodo_token": "my_zenodo_token",
            "zenodo_sandbox_token": "my_sandbox_zenodo_token",
        },
    )
    assert get_zenodo_token(sandbox=sandbox) == expected_token


class TestGetScatteringTableDir:
    def test_direct_argument(self, tmp_path):
        """Test returning scattering table dir when passed directly."""
        test_dir = str(tmp_path / "scattering")
        os.makedirs(test_dir, exist_ok=True)
        assert get_scattering_table_dir(scattering_table_dir=test_dir) == test_dir

    def test_from_config(self, tmp_path):
        """Test returning scattering table dir from config object."""
        import disdrodb

        test_dir = str(tmp_path / "scattering")
        os.makedirs(test_dir, exist_ok=True)
        disdrodb.config.set({"scattering_table_dir": test_dir})
        assert get_scattering_table_dir() == test_dir

    def test_missing_config_raises(self):
        """Test raising ValueError when scattering table dir missing."""
        import disdrodb

        with disdrodb.config.set({"scattering_table_dir": None}), pytest.raises(ValueError):
            get_scattering_table_dir()


class TestGetFolderPartitioning:
    def test_valid_partitioning(self):
        """Test returning a valid folder partitioning scheme."""
        import disdrodb

        disdrodb.config.set({"folder_partitioning": "year/month"})
        assert get_folder_partitioning() == "year/month"

    def test_missing_partitioning_raises(self):
        """Test raising ValueError when folder partitioning missing."""
        import disdrodb

        with disdrodb.config.set({"folder_partitioning": None}), pytest.raises(ValueError):
            get_folder_partitioning()


class TestProductsConfigsDirs:

    def test_get_default_products_configs_dir(self):
        """Test getting the default products configs directory."""
        import disdrodb

        default_dir = get_default_products_configs_dir()
        assert default_dir.endswith(os.path.join("etc", "products"))
        assert default_dir.startswith(disdrodb.package_dir)

    def test_get_products_configs_dir_default(self, monkeypatch):
        """Test fallback to default products configs dir if none in config."""
        import disdrodb

        # Force the function to see an empty environment without PYTEST_CURRENT_TEST
        monkeypatch.setattr("os.environ", {})

        disdrodb.config.set({"products_configs_dir": None})
        assert get_products_configs_dir() == get_default_products_configs_dir()

    def test_get_products_configs_dir_custom(self, tmp_path, monkeypatch):
        """Test returning custom products configs dir from config."""
        import disdrodb

        # Force the function to see an empty environment without PYTEST_CURRENT_TEST
        monkeypatch.setattr("os.environ", {})

        custom_dir = str(tmp_path / "custom_products")
        os.makedirs(custom_dir, exist_ok=True)
        disdrodb.config.set({"products_configs_dir": custom_dir})
        assert get_products_configs_dir() == custom_dir


class TestCopyDefaultProductsConfigs:

    def test_copy_success_and_fail(self, tmp_path):
        """Test copying default products configs and raising error if exists."""
        source = get_default_products_configs_dir()
        target = tmp_path / "copied_products"

        copied = copy_default_products_configs(str(target))
        assert os.path.exists(copied)
        assert set(os.listdir(source)) == set(os.listdir(copied))

        # Copy again should raise FileExistsError
        with pytest.raises(FileExistsError):
            copy_default_products_configs(str(target))
