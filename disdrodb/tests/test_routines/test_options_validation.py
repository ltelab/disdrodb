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
"""Test DISDRODB product options validation."""

import copy
import os

import numpy as np
import pytest

import disdrodb
from disdrodb import package_dir
from disdrodb.routines.options_validation import (
    ArchiveOptions,
    ArchiveOptionsTimeBlock,
    EventStrategyOptions,
    L2EProductOptions,
    L2MModelConfig,
    L2MProductGlobalConfig,
    L2MProductOptions,
    RadarOptions,
    TemporalResolutionsValidationMixin,
    TimeBlockStrategyOptions,
    _validate_l2m_structure,
    _validate_sensor_subdirectories,
    validate_all_product_yaml_files,
    validate_l2m_model_configs,
    validate_product_configuration_structure,
    validate_products_configurations,
    validate_temporal_resolution_consistency,
)
from disdrodb.tests.test_routines.options_defaults import (
    ARCHIVE_OPTIONS_EVENT,
    ARCHIVE_OPTIONS_TIME_BLOCK,
    GAMMA_GS_CONFIG,
    GAMMA_ML_CONFIG,
    L0C_GLOBAL_YAML,
    L1_GLOBAL_YAML,
    L2E_GLOBAL_YAML,
    L2E_PRODUCT_OPTIONS,
    L2M_GLOBAL_YAML,
    L2M_PRODUCT_OPTIONS,
    LOGNORMAL_ML_CONFIG,
    RADAR_OPTIONS,
)
from disdrodb.utils.yaml import write_yaml


class TestTimeBlockStrategyOptions:
    """Test TimeBlockStrategyOptions validation."""

    def test_valid_freq(self):
        """Test valid frequency validation."""
        options = TimeBlockStrategyOptions(freq="day")
        assert options.freq == "day"

        options = TimeBlockStrategyOptions(freq="month")
        assert options.freq == "month"

        options = TimeBlockStrategyOptions(freq="year")
        assert options.freq == "year"

    def test_invalid_freq(self):
        """Test invalid frequency validation."""
        with pytest.raises(ValueError):
            TimeBlockStrategyOptions(freq="invalid_freq")


class TestEventStrategyOptions:
    """Test EventStrategyOptions validation."""

    def test_valid_event_options(self):
        """Test valid event strategy options."""
        data = copy.deepcopy(ARCHIVE_OPTIONS_EVENT["strategy_options"])
        options = EventStrategyOptions(**data)
        assert options.detection_threshold == 10
        assert options.neighbor_time_interval == "1MIN"

    def test_negative_values_rejected(self):
        """Test that negative values are rejected."""
        data = copy.deepcopy(ARCHIVE_OPTIONS_EVENT["strategy_options"])
        data["detection_threshold"] = -1

        with pytest.raises(ValueError):
            EventStrategyOptions(**data)

    def test_invalid_time_intervals(self):
        """Test invalid time interval validation."""
        data = copy.deepcopy(ARCHIVE_OPTIONS_EVENT["strategy_options"])
        data["neighbor_time_interval"] = "INVALID"

        with pytest.raises(ValueError):
            EventStrategyOptions(**data)


class TestArchiveOptionsTimeBlock:
    """Test ArchiveOptionsTimeBlock validation."""

    def test_valid_time_block_options(self):
        """Test valid time block archive options."""
        data = copy.deepcopy(ARCHIVE_OPTIONS_TIME_BLOCK)
        options = ArchiveOptionsTimeBlock(**data)
        assert options.strategy == "time_block"
        assert options.strategy_options.freq == "day"

    def test_invalid_folder_partitioning(self):
        """Test invalid folder partitioning."""
        data = copy.deepcopy(ARCHIVE_OPTIONS_TIME_BLOCK)
        data["folder_partitioning"] = "INVALID"
        with pytest.raises(ValueError):
            ArchiveOptionsTimeBlock(**data)

    def test_strategy_event_raise_error(self):
        """Test valid time block archive options."""
        data = copy.deepcopy(ARCHIVE_OPTIONS_EVENT)
        with pytest.raises(ValueError):
            ArchiveOptionsTimeBlock(**data)

    def test_extra_key_raise_error(self):
        """Test valid time block archive options."""
        data = copy.deepcopy(ARCHIVE_OPTIONS_TIME_BLOCK)
        data["extra_key"] = "dummy"
        with pytest.raises(ValueError):
            ArchiveOptionsTimeBlock(**data)


class TestArchiveOptions:
    """Test ArchiveOptions validation."""

    def test_time_block_strategy(self):
        """Test time block strategy validation."""
        data = copy.deepcopy(ARCHIVE_OPTIONS_TIME_BLOCK)
        options = ArchiveOptions(**data)
        assert options.strategy == "time_block"
        assert isinstance(options.strategy_options, TimeBlockStrategyOptions)

    def test_event_strategy(self):
        """Test event strategy validation."""
        data = copy.deepcopy(ARCHIVE_OPTIONS_EVENT)
        options = ArchiveOptions(**data)
        assert options.strategy == "event"
        assert isinstance(options.strategy_options, EventStrategyOptions)

    def test_strategy_options_mixed_up(self):
        """Test raise error if strategy options do not correspond to specified strategy."""
        data = copy.deepcopy(ARCHIVE_OPTIONS_EVENT)
        data["strategy_options"] = ARCHIVE_OPTIONS_TIME_BLOCK["strategy_options"]
        with pytest.raises(ValueError):
            ArchiveOptions(**data)


class TestRadarOptions:
    """Test RadarOptions validation."""

    def test_default_radar_options(self):
        """Test default radar options creation."""
        data = copy.deepcopy(RADAR_OPTIONS)
        options = RadarOptions(**data)
        assert isinstance(options.frequency, np.ndarray)
        assert options.num_points == [1024]

    def test_single_values_converted_to_lists(self):
        """Test that single values are converted to lists."""
        data = copy.deepcopy(RADAR_OPTIONS)
        data["frequency"] = "X"
        data["num_points"] = 512
        data["elevation_angle"] = [0, 45]
        options = RadarOptions(**data)
        assert options.frequency == [9.4]  # X-band frequency
        assert options.num_points == [512]
        assert options.axis_ratio_model == ["Thurai2007"]
        assert options.elevation_angle == [0, 45]

    def test_invalid_axis_ratio_model(self):
        """Test invalid axis ratio model validation."""
        data = copy.deepcopy(RADAR_OPTIONS)
        data["axis_ratio_model"] = "INVALID"  # Invalid
        with pytest.raises(ValueError):
            RadarOptions(**data)


class TestL2EProductOptions:
    """Test L2EProductOptions validation."""

    def test_valid_l2e_options(self):
        """Test valid L2E product options."""
        data = copy.deepcopy(L2E_PRODUCT_OPTIONS)
        options = L2EProductOptions(**data)
        assert options.minimum_diameter == 0
        assert options.maximum_diameter == 10

    def test_diameter_range_validation(self):
        """Test diameter range validation."""
        data = copy.deepcopy(L2E_PRODUCT_OPTIONS)
        data["minimum_diameter"] = 0.2
        data["maximum_diameter"] = 0.1  # Less than minimum_diameter
        with pytest.raises(ValueError, match="maximum_diameter must be greater than minimum_diameter"):
            L2EProductOptions(**data)

    def test_velocity_range_validation(self):
        """Test velocity range validation."""
        data = copy.deepcopy(L2E_PRODUCT_OPTIONS)
        data["minimum_velocity"] = 0.5
        data["maximum_velocity"] = 0.1  # Less than minimum_velocity
        with pytest.raises(ValueError, match="maximum_velocity must be greater than minimum_velocity"):
            L2EProductOptions(**data)

    def test_invalid_fall_velocity_model(self):
        """Test invalid fall velocity model."""
        data = copy.deepcopy(L2E_PRODUCT_OPTIONS)
        data["fall_velocity_model"] = "INVALID"
        with pytest.raises(ValueError):
            L2EProductOptions(**data)


class TestL2MProductOptions:
    """Test L2MProductOptions validation."""

    def test_valid_l2m_options(self):
        """Test valid L2M product options."""
        data = copy.deepcopy(L2M_PRODUCT_OPTIONS)
        options = L2MProductOptions(**data)
        assert options.diameter_min == 0
        assert options.diameter_max == 10

    def test_diameter_range_validation(self):
        """Test diameter range validation."""
        data = copy.deepcopy(L2M_PRODUCT_OPTIONS)
        data["diameter_min"] = 0.2
        data["diameter_max"] = 0.1  # Less than minimum_diameter
        with pytest.raises(ValueError, match="diameter_max must be greater than diameter_min"):
            L2MProductOptions(**data)

    def test_diameter_spacing_larger_than_range(self):
        """Test diameter spacing larger than range."""
        data = copy.deepcopy(L2M_PRODUCT_OPTIONS)
        data["diameter_max"] = 9
        data["diameter_spacing"] = 10.0  # Larger than range
        with pytest.raises(ValueError, match="diameter_spacing cannot be larger than the diameter range"):
            L2MProductOptions(**data)


####-------------------------------------------------------------------------.
class TestL2MModelConfig:
    """Test L2MModelConfig validation."""

    def test_valid_l2m_model_config(self):
        """Test valid L2M model configuration."""
        data = copy.deepcopy(GAMMA_ML_CONFIG)
        config = L2MModelConfig(**data)
        assert config.psd_model == "GammaPSD"
        assert config.optimization == "ML"

    def test_invalid_psd_model(self):
        """Test invalid PSD model."""
        data = copy.deepcopy(GAMMA_ML_CONFIG)
        data["psd_model"] = "INVALID"

        with pytest.raises(ValueError, match="Invalid psd_model"):
            L2MModelConfig(**data)

    def test_invalid_optimization_kwargs(self):
        """Test invalid optimization kwargs."""
        data = copy.deepcopy(GAMMA_ML_CONFIG)
        data["optimization_kwargs"]["probability_method"] = "INVALID"

        with pytest.raises(ValueError):
            L2MModelConfig(**data)


####-------------------------------------------------------------------------.
class TestTemporalResolutionsValidationMixin:
    """Test TemporalResolutionsValidationMixin."""

    def test_valid_temporal_resolutions(self):
        """Test valid temporal resolutions validation."""

        class TestModel(TemporalResolutionsValidationMixin):
            pass

        result = TestModel.validate_temporal_resolutions(["1MIN", "5MIN"])
        assert result == ["1MIN", "5MIN"]

    def test_empty_temporal_resolutions(self):
        """Test empty temporal resolutions list."""

        class TestModel(TemporalResolutionsValidationMixin):
            pass

        with pytest.raises(ValueError, match="temporal_resolutions cannot be empty"):
            TestModel.validate_temporal_resolutions([])

    def test_duplicate_temporal_resolutions(self):
        """Test duplicate temporal resolutions."""

        class TestModel(TemporalResolutionsValidationMixin):
            pass

        with pytest.raises(ValueError, match="temporal_resolutions contains duplicates"):
            TestModel.validate_temporal_resolutions(["1MIN", "5MIN", "1MIN"])

    def test_invalid_temporal_resolution(self):
        """Test invalid temporal resolution."""

        class TestModel(TemporalResolutionsValidationMixin):
            pass

        with pytest.raises(ValueError):
            TestModel.validate_temporal_resolutions(["INVALID"])


####-------------------------------------------------------------------------.
class TestValidateProductConfigurationStructure:
    """Test configuration structure validation functions."""

    def test_validate_product_configuration_structure_missing_directory(self):
        """Test validation with missing products config directory."""
        with pytest.raises(FileNotFoundError, match="Products configuration directory not found"):
            validate_product_configuration_structure("/nonexistent")

    def test_validate_product_configuration_structure_missing_product_dir(self, tmp_path):
        """Test validation with missing product directory."""
        # Create products config dir but missing L1 subdirectory
        (tmp_path / "L0C").mkdir()
        # L1 missing
        (tmp_path / "L2E").mkdir()
        (tmp_path / "L2M").mkdir()

        with pytest.raises(FileNotFoundError, match="Required product directory not found"):
            validate_product_configuration_structure(str(tmp_path))

    def test_validate_product_configuration_structure_missing_global_yaml(self, tmp_path):
        """Test validation with missing global.yaml."""
        # Create all product directories but missing global.yaml in L1
        for product in ["L0C", "L1", "L2E", "L2M"]:
            product_dir = tmp_path / product
            product_dir.mkdir()
            if product != "L1":  # Skip L1 to test missing global.yaml
                write_yaml({"test": "config"}, product_dir / "global.yaml")

        # Create L2M/MODELS directory
        models_dir = tmp_path / "L2M" / "MODELS"
        models_dir.mkdir()
        write_yaml({"psd_model": "GammaPSD"}, models_dir / "model.yaml")

        with pytest.raises(FileNotFoundError, match="Required global.yaml file not found"):
            validate_product_configuration_structure(str(tmp_path))

    def test_validate_sensor_subdirectories_invalid_sensor(self, tmp_path):
        """Test validation with invalid sensor subdirectory."""
        product_dir = tmp_path / "L1"
        product_dir.mkdir()
        write_yaml({"test": "config"}, product_dir / "global.yaml")

        # Create invalid sensor directory
        invalid_sensor_dir = product_dir / "INVALID_SENSOR"
        invalid_sensor_dir.mkdir()
        write_yaml({"test": "config"}, invalid_sensor_dir / "config.yaml")

        with pytest.raises(ValueError, match="Invalid sensor directory 'INVALID_SENSOR'"):
            _validate_sensor_subdirectories(product_dir, ["PARSIVEL2"], "L1")

    def test_validate_sensor_subdirectories_no_yaml_files(self, tmp_path):
        """Test validation when sensor directory has no YAML files."""
        product_dir = tmp_path / "L1"
        product_dir.mkdir()

        # Create valid sensor directory but no YAML files
        sensor_dir = product_dir / "PARSIVEL2"
        sensor_dir.mkdir()
        (sensor_dir / "readme.txt").write_text("not yaml")

        with pytest.raises(FileNotFoundError, match="No YAML files found in sensor directory"):
            _validate_sensor_subdirectories(product_dir, ["PARSIVEL2"], "L1")

    def test_validate_l2m_structure_missing_models_dir(self, tmp_path):
        """Test L2M structure validation with missing MODELS directory."""
        l2m_dir = tmp_path / "L2M"
        l2m_dir.mkdir()

        with pytest.raises(FileNotFoundError, match="Required MODELS directory not found"):
            _validate_l2m_structure(l2m_dir)

    def test_validate_l2m_structure_no_yaml_in_models(self, tmp_path):
        """Test L2M structure validation with no YAML files in MODELS."""
        l2m_dir = tmp_path / "L2M"
        models_dir = l2m_dir / "MODELS"
        models_dir.mkdir(parents=True)
        (models_dir / "readme.txt").write_text("not yaml")

        with pytest.raises(FileNotFoundError, match="No YAML model configuration files found"):
            _validate_l2m_structure(l2m_dir)


####-------------------------------------------------------------------------.


class TestL2MProductConfig:
    """Test L2M product configuration.

    Here we use the disdrodb test products configs,
    which includes the GAMMA_ML model.""
    """

    def test_l2m_product_config_valid(self):
        """Test valid L2M product configuration."""
        data = copy.deepcopy(L2M_GLOBAL_YAML)
        data["models"] = ["GAMMA_ML"]

        config = L2MProductGlobalConfig(**data)
        assert config.models == ["GAMMA_ML"]

    def test_l2m_product_config_missing_models(self):
        """Test L2M config with missing model files."""
        data = copy.deepcopy(L2M_GLOBAL_YAML)
        data["models"] = ["NONEXISTENT"]

        with pytest.raises(ValueError, match="L2M model configuration files not found"):
            L2MProductGlobalConfig(**data)

    def test_l2m_product_config_duplicate_models(self):
        """Test L2M config with duplicated model names."""
        data = copy.deepcopy(L2M_GLOBAL_YAML)
        data["models"] = ["GAMMA_ML", "GAMMA_ML"]

        with pytest.raises(ValueError, match="'models' list contains duplicates"):
            L2MProductGlobalConfig(**data)

    def test_l2m_product_config_empty_models(self):
        """Test L2M config with empty models list."""
        data = copy.deepcopy(L2M_GLOBAL_YAML)
        data["models"] = []

        with pytest.raises(ValueError, match="'models' list cannot be empty"):
            L2MProductGlobalConfig(**data)


class TestValidateModelConfigs:

    def test_models_validation_with_valid_configuration(self, tmp_products_configs_dir):
        """Test successful L2M model config validation."""
        create_minimal_product_configs(tmp_products_configs_dir)

        # Should not raise any exceptions
        validate_l2m_model_configs(tmp_products_configs_dir)

    def test_models_validation_with_invalid_configuration(self, tmp_products_configs_dir):
        """Test L2M model config validation with errors."""
        # Create invalid model config
        l2m_dir = tmp_products_configs_dir / "L2M"
        l2m_dir.mkdir()
        models_dir = l2m_dir / "MODELS"
        models_dir.mkdir()

        invalid_config = {
            "psd_model": "INVALID_MODEL",  # Invalid model
            "optimization": "ML",
            "optimization_kwargs": {},
        }
        write_yaml(invalid_config, models_dir / "INVALID_MODEL.yaml")

        with pytest.raises(ValueError, match="L2M model configuration validation failed"):
            validate_l2m_model_configs(tmp_products_configs_dir)


class TestTemporalResolutionConsistency:
    """Test temporal resolution consistency validation."""

    def test_validate_temporal_resolution_consistency_warnings(self, tmp_path, capsys):
        """Test temporal resolution consistency validation print warnings."""
        create_minimal_product_configs(tmp_path)
        create_sensor_specific_configs(tmp_path)

        # This should print WARNINGS due to mismatched temporal resolutions
        validate_temporal_resolution_consistency(tmp_path)
        captured = capsys.readouterr()
        assert "WARNING" in captured.out


####-------------------------------------------------------------------------.


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


def create_sensor_specific_configs(config_dir):
    """Create sensor-specific configurations using predefined dictionaries."""
    # Create L1 PARSIVEL 2 directory
    l1_parsivel_dir = config_dir / "L1" / "PARSIVEL2"
    l1_parsivel_dir.mkdir(parents=True, exist_ok=True)

    # Create L2E PARSIVEL 2 directory
    l2e_parsivel_dir = config_dir / "L2E" / "PARSIVEL2"
    l2e_parsivel_dir.mkdir(parents=True, exist_ok=True)

    # Create L2E PARSIVEL 2 directory
    l2m_parsivel_dir = config_dir / "L2M" / "PARSIVEL2"
    l2m_parsivel_dir.mkdir(parents=True, exist_ok=True)

    # Create L1 sensor-level global YAML file
    l1_parsivel_config = copy.deepcopy(L1_GLOBAL_YAML)
    l1_parsivel_config["temporal_resolutions"] = ["1MIN", "2MIN"]
    l1_parsivel_config["archive_options"] = {
        "strategy": "time_block",
        "strategy_options": {"freq": "year"},
        "folder_partitioning": "year/month/day",
    }
    write_yaml(l1_parsivel_config, l1_parsivel_dir / "global.yaml")

    # Create L1 sensor-level temporal YAML file
    l1_1min_config = {
        "archive_options": {
            "strategy": "time_block",
            "strategy_options": {"freq": "month"},
            "folder_partitioning": "year/month/day",
        },
    }
    write_yaml(l1_1min_config, l1_parsivel_dir / "1MIN.yaml")

    # Create L2E sensor-level global YAML file
    l2e_parsivel_config = copy.deepcopy(L2E_GLOBAL_YAML)
    l2e_parsivel_config["radar_enabled"] = True  # Enable radar for this sensor
    write_yaml(l2e_parsivel_config, l2e_parsivel_dir / "global.yaml")

    # Create L2M sensor-level global YAML file
    l2m_parsivel_config = copy.deepcopy(L2M_GLOBAL_YAML)
    l2m_parsivel_config["temporal_resolutions"] = ["1MIN", "2MIN", "30MIN"]
    write_yaml(l2m_parsivel_config, l2m_parsivel_dir / "global.yaml")


class TestValidateAllProductsYAMLfiles:

    def test_sensor_level_configuration_override_global(self, tmp_products_configs_dir):
        """Test that sensor_level configuration override product-level configuration."""
        create_minimal_product_configs(tmp_products_configs_dir)
        create_sensor_specific_configs(tmp_products_configs_dir)

        # Test that validation passes with hierarchical configs
        validate_all_product_yaml_files(tmp_products_configs_dir)

    def test_raise_error_with_corrupted_yaml(self, tmp_products_configs_dir):
        """Test raise error with corrupted YAML file."""
        # Create minimal structure
        l1_dir = tmp_products_configs_dir / "L1"
        l1_dir.mkdir(parents=True)

        # Write invalid YAML
        (l1_dir / "global.yaml").write_text("invalid: yaml: syntax: [")

        with pytest.raises(Exception):  # ScannerError  # noqa: B017
            validate_all_product_yaml_files(tmp_products_configs_dir)

    def test_raise_error_invalid_global_config(self, tmp_products_configs_dir):
        """Test raise error with invalid product-level global YAML file."""
        create_minimal_product_configs(tmp_products_configs_dir)
        create_sensor_specific_configs(tmp_products_configs_dir)

        # Create invalid YAML file
        l1_dir = tmp_products_configs_dir / "L1"
        l1_dir.mkdir(parents=True, exist_ok=True)
        (l1_dir / "global.yaml").write_text("invalid: dummy")

        with pytest.raises(ValueError):
            validate_all_product_yaml_files(tmp_products_configs_dir)

    def test_raise_error_invalid_sensor_level_global_config(self, tmp_products_configs_dir):
        """Test raise error with invalid sensor-level global YAML file."""
        create_minimal_product_configs(tmp_products_configs_dir)
        create_sensor_specific_configs(tmp_products_configs_dir)

        # Create invalid YAML file
        l1_dir = tmp_products_configs_dir / "L1" / "LPM"
        l1_dir.mkdir(parents=True, exist_ok=True)
        (l1_dir / "global.yaml").write_text("invalid: dummy")

        with pytest.raises(ValueError):
            validate_all_product_yaml_files(tmp_products_configs_dir)

    def test_raise_error_sensor_level_temporal_config(self, tmp_products_configs_dir):
        """Test raise error with invalid sensor-level temporal global YAML file."""
        create_minimal_product_configs(tmp_products_configs_dir)
        create_sensor_specific_configs(tmp_products_configs_dir)

        # Create invalid YAML file
        l1_dir = tmp_products_configs_dir / "L1" / "LPM"
        l1_dir.mkdir(parents=True, exist_ok=True)
        (l1_dir / "1MIN.yaml").write_text("invalid: dummy")
        with pytest.raises(ValueError):
            validate_all_product_yaml_files(tmp_products_configs_dir)


####--------------------------------------------------------------------------------------------------


def test_validate_disdrodb_default_products_configurations(enable_config_validation):
    """Test validation of disdrodb default products configurations."""
    products_configs_dir = os.path.join(package_dir, "etc", "products")
    with disdrodb.config.set({"products_configs_dir": products_configs_dir}):
        validate_products_configurations(products_configs_dir)


def test_validate_disdrodb_tests_products_configurations(enable_config_validation):
    """Test validation of disdrodb default products configurations."""
    products_configs_dir = os.path.join(package_dir, "tests", "products")
    with disdrodb.config.set({"products_configs_dir": products_configs_dir}):
        validate_products_configurations(products_configs_dir)
