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
"""Test DISDRODB routines options."""
import copy
import datetime as dt
from pathlib import Path

import numpy as np
import pytest
from click.testing import CliRunner
from disdrodb.cli.disdrodb_check_products_options import (
    disdrodb_check_products_options,
)

import disdrodb
from disdrodb.constants import ARCHIVE_VERSION
from disdrodb.routines.options import (
    L0CProcessingOptions,
    L1ProcessingOptions,
    L2ProcessingOptions,
    _define_blocks_offsets,
    check_availability_radar_simulations,
    get_l2m_model_settings_directory,
    get_l2m_model_settings_files,
    get_model_options,
    get_product_custom_options_path,
    get_product_global_options_path,
    get_product_options,
    get_product_options_directory,
    get_product_temporal_resolutions,
)
from disdrodb.tests.test_routines.options_defaults import (
    GAMMA_GS_CONFIG,
    GAMMA_ML_CONFIG,
    L0C_GLOBAL_YAML,
    L1_GLOBAL_YAML,
    L2E_GLOBAL_YAML,
    L2M_GLOBAL_YAML,
    LOGNORMAL_ML_CONFIG,
)
from disdrodb.utils.yaml import write_yaml


class TestProductOptionsDirectory:

    def test_get_product_options_directory_without_sensor(self, tmp_path):
        """Test options directory uses only product when no sensor is given."""
        products_configs_dir = tmp_path / "products"
        product = "L1"

        result = get_product_options_directory(str(products_configs_dir), product)

        assert Path(result) == products_configs_dir / product

    def test_get_product_options_directory_with_sensor(self, tmp_path):
        """Test options directory nests sensor under product when sensor is given."""
        products_configs_dir = tmp_path / "products"
        product = "L1"
        sensor = "PARSIVEL"

        result = get_product_options_directory(str(products_configs_dir), product, sensor_name=sensor)

        assert Path(result) == products_configs_dir / product / sensor


def test_get_l2m_model_settings_directory(tmp_path):
    """Test L2M model settings directory is built under L2M/MODELS."""
    products_configs_dir = tmp_path / "products"

    result = get_l2m_model_settings_directory(str(products_configs_dir))

    assert Path(result) == products_configs_dir / "L2M" / "MODELS"


def test_get_l2m_model_settings_files_lists_yaml_and_yml(tmp_path):
    """Test model settings file discovery returns both .yaml and .yml files."""
    products_configs_dir = tmp_path / "products"
    models_dir = products_configs_dir / "L2M" / "MODELS"
    models_dir.mkdir(parents=True)

    yaml_file = models_dir / "model_a.yaml"
    yml_file = models_dir / "model_b.yml"
    txt_file = models_dir / "ignore.txt"

    yaml_file.write_text("param: 1\n", encoding="utf-8")
    yml_file.write_text("param: 2\n", encoding="utf-8")
    txt_file.write_text("not yaml\n", encoding="utf-8")

    files = get_l2m_model_settings_files(str(products_configs_dir))

    assert set(files) == {yaml_file, yml_file}


class TestProductOptionsPath:

    def test_get_product_global_options_path(self, tmp_path):
        """Test global options path falls back to product-level when sensor file is missing."""
        products_configs_dir = tmp_path / "products"
        product = "L1"
        sensor_name = "PARSIVEL"

        product_dir = products_configs_dir / product
        product_dir.mkdir(parents=True, exist_ok=True)
        product_global = product_dir / "global.yaml"
        product_global.write_text("sensor: false\n", encoding="utf-8")

        result = get_product_global_options_path(
            products_configs_dir=str(products_configs_dir),
            product=product,
            sensor_name=sensor_name,
        )

        assert Path(result) == product_global

    def test_get_product_global_options_path_with_sensor(self, tmp_path):
        """Test global options path prefers sensor-specific file when present."""
        products_configs_dir = tmp_path / "products"
        product = "L1"
        sensor_name = "PARSIVEL"

        sensor_dir = products_configs_dir / product / sensor_name
        product_dir = products_configs_dir / product
        sensor_dir.mkdir(parents=True, exist_ok=True)
        product_dir.mkdir(parents=True, exist_ok=True)

        sensor_global = sensor_dir / "global.yaml"
        product_global = product_dir / "global.yaml"

        sensor_global.write_text("sensor: true\n", encoding="utf-8")
        product_global.write_text("sensor: false\n", encoding="utf-8")

        result = get_product_global_options_path(
            products_configs_dir=products_configs_dir,
            product=product,
            sensor_name=sensor_name,
        )

        assert Path(result) == sensor_global

    def test_get_product_custom_options_path(self, tmp_path):
        """Test custom options path falls back to product-level temporal file when sensor file is missing."""
        products_configs_dir = tmp_path / "products"
        product = "L1"
        temporal_resolution = "1MIN"

        product_dir = products_configs_dir / product
        product_dir.mkdir(parents=True, exist_ok=True)

        # We do NOT create the actual file to test pure path logic.
        result = get_product_custom_options_path(
            products_configs_dir=str(products_configs_dir),
            product=product,
            temporal_resolution=temporal_resolution,
            sensor_name="PARSIVEL",
        )

        expected = str(product_dir / f"{temporal_resolution}.yaml")
        assert result == expected

    def test_get_product_custom_options_path_with_sensor(self, tmp_path):
        """Test custom options path prefers sensor-specific temporal file when present."""
        products_configs_dir = tmp_path / "products"
        product = "L1"
        sensor_name = "PARSIVEL"
        temporal_resolution = "1MIN"

        sensor_dir = products_configs_dir / product / sensor_name
        sensor_dir.mkdir(parents=True, exist_ok=True)

        sensor_custom = sensor_dir / f"{temporal_resolution}.yaml"
        sensor_custom.write_text("sensor: true\n", encoding="utf-8")

        result = get_product_custom_options_path(
            products_configs_dir=str(products_configs_dir),
            product=product,
            temporal_resolution=temporal_resolution,
            sensor_name=sensor_name,
        )

        assert Path(result) == sensor_custom


def test_get_product_temporal_resolutions(tmp_products_configs_dir):
    """Test get_product_temporal_resolutions."""
    l2e_dir = tmp_products_configs_dir / "L2E"
    l2e_dir.mkdir(parents=True)

    temporal_resolutions = ["1MIN", "5MIN", "10MIN", "ROLL10MIN"]
    l2e_options = copy.deepcopy(L2E_GLOBAL_YAML)
    l2e_options["temporal_resolutions"] = temporal_resolutions
    write_yaml(l2e_options, filepath=(l2e_dir / "global.yaml"))

    output = get_product_temporal_resolutions(product="L2E")
    assert isinstance(output, list)
    assert output == temporal_resolutions


class TestGetProductOptions:

    def test_radar_enabled_modified_if_pytmatrix_missing(self, tmp_products_configs_dir, monkeypatch):
        """Test get_product_options check and modify radar_enabled if T-matrix not available."""
        # Create product-level global YAML file
        l2e_dir = tmp_products_configs_dir / "L2E"
        l2e_dir.mkdir(parents=True)

        l2e_options = copy.deepcopy(L2E_GLOBAL_YAML)
        l2e_options["radar_enabled"] = True
        l2e_options["archive_options"]["folder_partitioning"] = "year"

        write_yaml(l2e_options, filepath=(l2e_dir / "global.yaml"))

        # Make pyTMatrix unavailable so radar_enabled should be turned off
        monkeypatch.setattr(disdrodb, "is_pytmatrix_available", lambda: False)

        # Get product options
        options_dict = get_product_options(product="L2E")

        # Assert
        assert options_dict["archive_options"]["folder_partitioning"] == "year"
        assert options_dict["radar_enabled"] is False

    def test_return_global_settings_for_unspecified_resolution(self, tmp_products_configs_dir):
        """Test get_product_options returns global options for unspecified temporal resolution YAML file."""
        # Create product-level global YAML file
        l2e_dir = tmp_products_configs_dir / "L2E"
        l2e_dir.mkdir(parents=True)

        l2e_options = copy.deepcopy(L2E_GLOBAL_YAML)
        write_yaml(l2e_options, filepath=(l2e_dir / "global.yaml"))

        # Get product options
        global_options = get_product_options(product="L2E")
        global_options.pop("temporal_resolutions")

        custom_options = get_product_options(product="L2E", temporal_resolution="20MIN")
        assert global_options == custom_options

    def test_get_product_options_merges_global_and_custom_keys(self, tmp_products_configs_dir, monkeypatch):
        """Test product options merge global and custom nested/flat keys correctly."""
        # Create product-level global YAML file
        l2e_dir = tmp_products_configs_dir / "L2E"
        l2e_dir.mkdir(parents=True)

        l2e_options = copy.deepcopy(L2E_GLOBAL_YAML)
        l2e_options["radar_enabled"] = False
        l2e_options["archive_options"]["folder_partitioning"] = "year"
        l2e_options["product_options"]["maximum_diameter"] = 10  # value in global to be overridden
        l2e_options["product_options"].pop("minimum_nbins")  # no minimum_nbins in global
        l2e_options["global_flat_only"] = 1
        l2e_options["flat"] = 1
        write_yaml(l2e_options, filepath=(l2e_dir / "global.yaml"))

        # Create product-level temporal resolution YAML file
        l2e_1min_options = copy.deepcopy(L2E_GLOBAL_YAML)
        l2e_1min_options.pop("temporal_resolutions")
        l2e_1min_options["archive_options"]["folder_partitioning"] = "year/month"
        l2e_1min_options["archive_options"]["extra_option"] = "extra"
        l2e_1min_options["radar_enabled"] = True

        l2e_1min_options.pop("radar_options")  # no radar options
        l2e_1min_options["product_options"].pop("fall_velocity_model")  # no fall_velocity_model option
        l2e_1min_options["product_options"]["maximum_diameter"] = 15
        l2e_1min_options["custom_flat_only"] = 1
        l2e_1min_options["flat"] = 2
        write_yaml(l2e_1min_options, filepath=(l2e_dir / "1MIN.yaml"))

        # Make pyTMatrix available so radar_enabled stay True
        monkeypatch.setattr(disdrodb, "is_pytmatrix_available", lambda: True)

        # Get product options
        options = get_product_options(
            product="L2E",
            temporal_resolution="1MIN",
        )

        # Temporal_resolutions must have been dropped
        assert "temporal_resolutions" not in options

        # Archive_options are merged, with custom overriding folder_partitioning
        archive = options["archive_options"]
        assert archive["strategy"] == "time_block"
        assert archive["folder_partitioning"] == "year/month"
        assert archive["extra_option"] == "extra"

        # Product_options merged from global -> custom
        product_options = options["product_options"]
        assert product_options["maximum_diameter"] == 15
        assert "fall_velocity_model" in product_options
        assert "minimum_nbins" in product_options

        # Radar_options copied from global
        radar_options = options["radar_options"]
        assert radar_options["num_points"] == 1024

        # Flat keys merged and updated
        assert options["custom_flat_only"] == 1
        assert options["global_flat_only"] == 1
        assert options["flat"] == 2

        assert options["radar_enabled"] is True

    def test_get_product_options_returns_sensor_custom_options(self, tmp_products_configs_dir):
        """Test it ignores product-level 1MIN options if sensor-specific 1MIN options are specified."""
        # Define directories
        l2e_dir = tmp_products_configs_dir / "L2E"
        l2e_dir.mkdir(parents=True)

        sensor_name = "PARSIVEL"
        sensor_dir = l2e_dir / sensor_name
        sensor_dir.mkdir(parents=True)

        # Create product-level global YAML file
        l2e_options = copy.deepcopy(L2E_GLOBAL_YAML)
        l2e_options["radar_enabled"] = True
        l2e_options["archive_options"]["folder_partitioning"] = "year"
        l2e_options["product_options"]["global_level"] = 1
        write_yaml(l2e_options, filepath=(l2e_dir / "global.yaml"))

        # Create product-level temporal resolution YAML file
        l2e_1min_options = copy.deepcopy(L2E_GLOBAL_YAML)
        l2e_1min_options.pop("temporal_resolutions")
        l2e_1min_options["archive_options"]["folder_partitioning"] = "year/month"
        l2e_1min_options["product_options"]["product_level"] = 1
        write_yaml(l2e_1min_options, filepath=(l2e_dir / "1MIN.yaml"))

        # Create sensor-level temporal resolution YAML file
        l2e_parsivel_1min_options = copy.deepcopy(L2E_GLOBAL_YAML)
        l2e_parsivel_1min_options.pop("temporal_resolutions")
        l2e_parsivel_1min_options["archive_options"]["folder_partitioning"] = ""
        l2e_parsivel_1min_options["product_options"]["sensor_level"] = 1

        write_yaml(l2e_parsivel_1min_options, filepath=(sensor_dir / "1MIN.yaml"))

        # Retrieve product options
        options_dict = get_product_options(
            product="L2E",
            temporal_resolution="1MIN",
            sensor_name=sensor_name,
        )

        # Archive_options: sensor-level value
        assert options_dict["archive_options"]["folder_partitioning"] == ""

        # product_options: global < temporal < sensor-specific
        product_options = options_dict["product_options"]
        assert product_options["global_level"] == 1
        assert "product_level" not in product_options
        assert product_options["sensor_level"] == 1

    def test_get_model_options_reads_model_yaml(self, tmp_products_configs_dir):
        """Test model options are read from L2M/MODELS/<model>.yaml."""
        models_dir = tmp_products_configs_dir / "L2M" / "MODELS"
        models_dir.mkdir(parents=True)

        model_name = "test_model"
        model_file = models_dir / f"{model_name}.yaml"
        model_file.write_text("param: 42\n", encoding="utf-8")

        model_options = get_model_options(model_name=model_name)

        assert model_options["param"] == 42


class TestRadarSimulationsAvailability:
    def test_radar_flag_preserved_when_pytmatrix_available(self, monkeypatch):
        """Test radar_enabled stays unchanged when pyTMatrix is available."""
        monkeypatch.setattr(disdrodb, "is_pytmatrix_available", lambda: True)
        options = {"radar_enabled": True, "other": 1}

        result = check_availability_radar_simulations(options)

        assert result is options
        assert result["radar_enabled"] is True
        assert result["other"] == 1

    def test_radar_flag_disabled_when_pytmatrix_not_available(self, monkeypatch):
        """Test radar_enabled is forced to False when pyTMatrix is unavailable."""
        monkeypatch.setattr(disdrodb, "is_pytmatrix_available", lambda: False)
        options = {"radar_enabled": True}

        result = check_availability_radar_simulations(options)

        assert result["radar_enabled"] is False

    def test_no_radar_key_leaves_dict_unchanged(self, monkeypatch):
        """Test options without radar_enabled key are returned unchanged."""
        monkeypatch.setattr(disdrodb, "is_pytmatrix_available", lambda: False)
        options = {"other": 123}

        result = check_availability_radar_simulations(options)

        assert result is options
        assert "radar_enabled" not in result


class TestDefineBlocksOffsets:
    def test_define_blocks_offsets_non_rolling(self):
        """Test offsets are zero for non-rolling temporal resolutions."""
        block_start, block_end = _define_blocks_offsets(
            sample_interval=60,
            temporal_resolution="1MIN",
        )

        assert block_start == 0
        assert block_end == 0

    def test_define_blocks_offsets_rolling_same_interval(self):
        """Test offsets are zero when rolling but sample interval matches accumulation."""
        block_start, block_end = _define_blocks_offsets(
            sample_interval=60,
            temporal_resolution="ROLL1MIN",
        )

        assert block_start == 0
        assert block_end == 0

    def test_define_blocks_offsets_rolling_shorter_sample_interval(self):
        """Test rolling offsets add coverage up to accumulation interval."""
        block_start, block_end = _define_blocks_offsets(
            sample_interval=30,
            temporal_resolution="ROLL1MIN",
        )

        assert block_start == 0
        assert block_end == np.timedelta64(30, "s")


# ---------------------------------------------------------------------------
# Fixtures: temporary products configuration + synthetic product filepaths
# ---------------------------------------------------------------------------


def create_minimal_product_configs(config_dir):
    """Create minimal valid product configuration structure using predefined dictionaries."""
    # Use the predefined dictionaries from the top of the module
    yaml_configs = {
        "L0C": L0C_GLOBAL_YAML,
        "L1": L1_GLOBAL_YAML,
        "L2E": L2E_GLOBAL_YAML,
        "L2M": L2M_GLOBAL_YAML,
    }

    # Write product level global YAML files
    for product, yaml_config in yaml_configs.items():
        product_dir = config_dir / product
        product_dir.mkdir(parents=True, exist_ok=True)
        write_yaml(yaml_config, product_dir / "global.yaml")

    # Create L2M Models using predefined configurations
    models_dir = config_dir / "L2M" / "MODELS"
    models_dir.mkdir(parents=True, exist_ok=True)

    # Create a simplified set of models for minimal config
    minimal_model_configs = {
        "GAMMA_ML": GAMMA_ML_CONFIG,
        "GAMMA_GS": GAMMA_GS_CONFIG,
        "LOGNORMAL_ML": LOGNORMAL_ML_CONFIG,
    }

    for model_name, model_config in minimal_model_configs.items():
        write_yaml(model_config, models_dir / f"{model_name}.yaml")


@pytest.fixture
def l1_filepaths(tmp_path):
    """Create a small set of synthetic L1 files with 1-minute sampling."""
    data_dir = tmp_path / "l1_data"
    data_dir.mkdir()

    filepaths = []
    # Two files on the same day to keep things simple but non-trivial
    for hour in (0, 12):
        start = dt.datetime(2020, 1, 1, hour, 0, 0)
        end = dt.datetime(2020, 1, 1, hour, 59, 59)
        fname = f"L1.1MIN.CAMPAIGN_NAME.STATION_NAME.s{start:%Y%m%d%H%M%S}.e{end:%Y%m%d%H%M%S}.{ARCHIVE_VERSION}.nc"
        path = data_dir / fname
        path.touch()
        filepaths.append(str(path))

    return filepaths


@pytest.fixture
def l2e_filepaths(tmp_path):
    """Create a synthetic L2E file covering a full month."""
    data_dir = tmp_path / "l2e_data"
    data_dir.mkdir()

    start = dt.datetime(2020, 1, 1, 0, 0, 0)
    end = dt.datetime(2020, 1, 31, 23, 59, 59)
    fname = f"L2E.1MIN.CAMPAIGN_NAME.STATION_NAME.s{start:%Y%m%d%H%M%S}.e{end:%Y%m%d%H%M%S}.{ARCHIVE_VERSION}.nc"
    path = data_dir / fname
    path.touch()
    return [str(path)]


@pytest.fixture
def l2m_filepaths(tmp_path):
    """Create a synthetic L2M file for the GAMMA_ML model covering a full month."""
    data_dir = tmp_path / "l2m_data"
    data_dir.mkdir()

    start = dt.datetime(2020, 2, 1, 0, 0, 0)
    end = dt.datetime(2020, 2, 29, 23, 59, 59)
    # Pattern: L2M_<subproduct>.{accumulation_acronym}.campaign.station.s...e....version.format
    fname = f"L2M_GAMMA_ML.1MIN.CAMPAIGN_NAME.STATION_NAME.s{start:%Y%m%d%H%M%S}.e{end:%Y%m%d%H%M%S}.{ARCHIVE_VERSION}.nc"  # noqa
    path = data_dir / fname
    path.touch()
    return [str(path)]


# ---------------------------------------------------------------------------
# Tests for L0CProcessingOptions
# ---------------------------------------------------------------------------


class TestL0CProcessingOptions:
    """Tests for the L0CProcessingOptions helper class."""

    def test_l0c_processing_options_basic(self, tmp_products_configs_dir):
        """Ensure L0CProcessingOptions reads folder partitioning and frequency correctly."""
        create_minimal_product_configs(tmp_products_configs_dir)

        options = L0CProcessingOptions(sensor_name="PARSIVEL2")
        assert options.product == "L0C"
        assert options.folder_partitioning == "year"
        assert options.product_frequency == "day"


# ---------------------------------------------------------------------------
# Tests for L1ProcessingOptions
# ---------------------------------------------------------------------------


class TestL1ProcessingOptions:
    """Tests for the L1ProcessingOptions helper class."""

    def test_l1_temporal_resolutions_filtering(self, tmp_products_configs_dir, l1_filepaths):
        """Ensure L1ProcessingOptions keeps only feasible temporal resolutions."""
        create_minimal_product_configs(tmp_products_configs_dir)
        options = L1ProcessingOptions(
            filepaths=l1_filepaths,
            parallel=False,
            sensor_name="PARSIVEL2",
            temporal_resolutions=None,  # use global config list
        )

        # From the YAML: ["1MIN", "5MIN", "10MIN", "ROLL1MIN"]
        # But with 1-minute sample interval, ROLL1MIN is not allowed (rolling at source interval).
        assert set(options.temporal_resolutions) == {"1MIN", "5MIN", "10MIN"}
        assert "ROLL1MIN" not in options.temporal_resolutions

    def test_l1_folder_partitioning_from_yaml(self, tmp_products_configs_dir, l1_filepaths):
        """Verify L1ProcessingOptions exposes folder partitioning from archive options."""
        create_minimal_product_configs(tmp_products_configs_dir)
        options = L1ProcessingOptions(
            filepaths=l1_filepaths,
            parallel=False,
            sensor_name="PARSIVEL2",
            temporal_resolutions="1MIN",
        )

        assert options.get_folder_partitioning("1MIN") == "year"

        # Assert get_product_options currently return empty dictionary
        assert options.get_product_options("1MIN") == {}

    def test_l1_files_grouped_by_partitions(self, tmp_products_configs_dir, l1_filepaths):
        """Check that L1ProcessingOptions groups L1 files into partitions."""
        create_minimal_product_configs(tmp_products_configs_dir)
        options = L1ProcessingOptions(
            filepaths=l1_filepaths,
            parallel=False,
            sensor_name="PARSIVEL2",
            temporal_resolutions="1MIN",
        )

        partitions = options.group_files_by_temporal_partitions("1MIN")

        # With freq=day and files all in the same day, we expect a single time block.
        assert len(partitions) == 1
        block = partitions[0]

        # All input files should be present in the single block.
        assert {Path(fp).name for fp in block["filepaths"]} == {Path(fp).name for fp in l1_filepaths}

        # Start and end time should be within the expected date if partitions freq is daily.
        assert block["start_time"].date() == block["end_time"].date()


# ---------------------------------------------------------------------------
# Tests for L2ProcessingOptions
# ---------------------------------------------------------------------------


class TestL2ProcessingOptions:
    """Tests for the L2ProcessingOptions helper class."""

    def test_l2e_processing_options_structure(self, tmp_products_configs_dir, l2e_filepaths):
        """Ensure L2E L2ProcessingOptions exposes folder partitioning and product options correctly."""
        create_minimal_product_configs(tmp_products_configs_dir)
        options = L2ProcessingOptions(
            product="L2E",
            filepaths=l2e_filepaths,
            parallel=False,
            temporal_resolution="1MIN",
            sensor_name="PARSIVEL2",
        )

        # Folder partitioning from L2E YAML (empty string)
        assert options.folder_partitioning == ""

        # Product options should contain nested product_options and radar configuration.
        assert "product_options" in options.product_options
        assert "radar_enabled" in options.product_options
        assert "radar_options" in options.product_options

        # Radar should remain disabled as per YAML.
        assert options.product_options["radar_enabled"] is False

        # Files partitions should contain our single monthly block with the file.
        assert len(options.files_partitions) == 1
        block = options.files_partitions[0]
        assert {Path(fp).name for fp in block["filepaths"]} == {
            Path(l2e_filepaths[0]).name,
        }

    def test_l2m_processing_options_include_models(self, tmp_products_configs_dir, l2m_filepaths):
        """Ensure L2M L2ProcessingOptions keeps the models list in product options."""
        create_minimal_product_configs(tmp_products_configs_dir)
        options = L2ProcessingOptions(
            product="L2M",
            filepaths=l2m_filepaths,
            parallel=False,
            temporal_resolution="1MIN",
            sensor_name="PARSIVEL2",
        )

        # Folder partitioning from L2M YAML (empty string)
        assert options.folder_partitioning == ""

        # Models must be preserved in the product options dictionary.
        assert "models" in options.product_options
        assert isinstance(options.product_options["models"], list)
        assert "GAMMA_ML" in options.product_options["models"]

        # Check that the single synthetic file is part of the monthly partition.
        assert len(options.files_partitions) == 1
        block = options.files_partitions[0]
        assert {Path(fp).name for fp in block["filepaths"]} == {
            Path(l2m_filepaths[0]).name,
        }


def test_cli_disdrodb_check_products_options():
    """Test the disdrodb_check_products_options command prints 'successfully'."""
    # Run CLI
    runner = CliRunner()
    result = runner.invoke(disdrodb_check_products_options)

    # Make sure command ran successfully
    assert result.exit_code == 0, result.output
    # Verify expected message appears
    assert "products configurations validated successfully" in result.output.lower()
