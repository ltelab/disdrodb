import os
from pathlib import Path
from typing import Any, Literal, Union

from pydantic import Field, field_validator, model_validator

from disdrodb.api.checks import check_folder_partitioning, check_temporal_resolution
from disdrodb.api.configs import available_sensor_names
from disdrodb.configs import get_products_configs_dir
from disdrodb.fall_velocity.rain import check_rain_fall_velocity_model
from disdrodb.psd.fitting import PSD_MODELS, check_optimization, check_optimization_kwargs
from disdrodb.routines.options import get_l2m_model_settings_files, get_model_options, get_product_options
from disdrodb.scattering.axis_ratio import check_axis_ratio_model
from disdrodb.scattering.permittivity import check_permittivity_model
from disdrodb.scattering.routines import ensure_numerical_frequency
from disdrodb.utils.archiving import check_freq
from disdrodb.utils.pydantic import CustomBaseModel

# -------------------------------------------------------------------------------------------------------.
## Check product options structure (products_configs_dir)
# products_configs_dir : check exists
# Check presence of L0C, L1, L2E and L2M directories
# If inside product directory (L0C-L2E) there is another directory must correspond to a sensor_name
# --> available_sensor_names(), otherwise raise error
# --> Presence of sensor directory is not mandatory
# If sensor directory is specified, there must be yaml file inside, otherwise raise error
# Check there is a global.yaml file inside each product directory
# Check there is the MODEL directory in L2M directory and there are YAML files inside.

# Check global product options YAML files
# Check global product options YAML files per sensor
# Check custom temporal resolution product options YAML files per sensor
# -------------------------------------------------------------------------------------------------------.


def validate_product_configuration_structure(products_configs_dir):
    """
    Validate the DISDRODB products configuration directory structure.

    Parameters
    ----------
    products_configs_dir : str
        Path to the products configuration directory.

    Raises
    ------
    FileNotFoundError
        If required directories or files are missing.
    ValueError
        If directory structure is invalid.
    """
    products_configs_path = Path(products_configs_dir)

    # Check that products_configs_dir exists
    if not products_configs_path.exists():
        raise FileNotFoundError(f"Products configuration directory not found: {products_configs_dir}")

    # Define required product directories
    required_products = ["L0C", "L1", "L2E", "L2M"]
    available_sensors = available_sensor_names()

    # Check presence of required product directories
    for product in required_products:
        product_dir = products_configs_path / product
        if not product_dir.exists():
            raise FileNotFoundError(f"Required product directory not found: {product_dir}")

    # Check for global.yaml in each product directory
    for product in required_products:
        product_dir = products_configs_path / product
        global_yaml = product_dir / "global.yaml"
        if not global_yaml.exists():
            raise FileNotFoundError(f"Required global.yaml file not found in: {product_dir}")

        # Check subdirectories within product directories (L0C-L2E)
        if product in ["L0C", "L1", "L2E", "L2M"]:
            _validate_sensor_subdirectories(product_dir, available_sensors, product)

    # Special validation for L2M directory
    _validate_l2m_structure(products_configs_path / "L2M")


def _validate_sensor_subdirectories(product_dir, available_sensors, product_name):
    """
    Validate sensor subdirectories within a product directory.

    Parameters
    ----------
    product_dir : Path
        Path to the product directory.
    available_sensors : list
        list of available sensor names.
    product_name : str
        Name of the product (for error messages).
    """
    # Get all subdirectories (excluding files)
    subdirs = [d for d in product_dir.iterdir() if d.is_dir()]

    for subdir in subdirs:
        sensor_name = subdir.name

        if product_name == "L2M" and subdir.name == "MODELS":
            continue

        # Check if subdirectory name corresponds to a valid sensor name
        if sensor_name not in available_sensors:
            raise ValueError(
                f"Invalid sensor directory '{sensor_name}' in {product_name}. " f"Must be one of: {available_sensors}",
            )

        # Check that sensor directory contains at least one YAML file
        yaml_files = list(subdir.glob("*.yaml")) + list(subdir.glob("*.yml"))
        if not yaml_files:
            raise FileNotFoundError(
                f"No YAML files found in sensor directory: {subdir}. "
                f"Sensor directories must contain at least one YAML configuration file.",
            )


def _validate_l2m_structure(l2m_dir):
    """
    Validate L2M directory structure including MODELS subdirectory.

    Parameters
    ----------
    l2m_dir : Path
        Path to the L2M directory.
    """
    models_dir = l2m_dir / "MODELS"

    # Check for MODELS directory in L2M
    if not models_dir.exists():
        raise FileNotFoundError(f"Required MODELS directory not found in: {l2m_dir}")

    # Check that MODELS directory contains YAML files
    yaml_files = list(models_dir.glob("*.yaml")) + list(models_dir.glob("*.yml"))
    if not yaml_files:
        raise FileNotFoundError(
            f"No YAML model configuration files found in: {models_dir}. "
            f"MODELS directory must contain at least one model YAML file.",
        )


def validate_temporal_resolution_consistency(products_configs_dir):
    """
    Validate temporal resolution consistency across products for each sensor.

    Raises warnings if:
    - L1 temporal_resolutions doesn't include all L2E and L2M temporal resolutions
    - L2E temporal_resolutions doesn't include all L2M temporal resolutions
    """
    sensor_names = available_sensor_names()
    for sensor_name in sensor_names:
        # Get temporal resolutions for each product
        l1_options = get_product_options(
            product="L1",
            sensor_name=sensor_name,
            products_configs_dir=products_configs_dir,
        )
        l2e_options = get_product_options(
            product="L2E",
            sensor_name=sensor_name,
            products_configs_dir=products_configs_dir,
        )
        l2m_options = get_product_options(
            product="L2M",
            sensor_name=sensor_name,
            products_configs_dir=products_configs_dir,
        )

        l1_temporal_resolutions = set(l1_options.get("temporal_resolutions", []))
        l2e_temporal_resolutions = set(l2e_options.get("temporal_resolutions", []))
        l2m_temporal_resolutions = set(l2m_options.get("temporal_resolutions", []))

        # Check L1 includes all L2E temporal resolutions
        missing_l2e_in_l1 = l2e_temporal_resolutions - l1_temporal_resolutions
        if missing_l2e_in_l1:
            print(
                f"WARNING. Sensor '{sensor_name}': L1 temporal_resolutions {sorted(l1_temporal_resolutions)} "
                f"does not include all L2E temporal resolutions. Missing: {sorted(missing_l2e_in_l1)}. "
                f"L2E temporal_resolutions: {sorted(l2e_temporal_resolutions)}",
            )

        # Check L1 includes all L2M temporal resolutions
        missing_l2m_in_l1 = l2m_temporal_resolutions - l1_temporal_resolutions
        if missing_l2m_in_l1:
            print(
                f"WARNING. Sensor '{sensor_name}': L1 temporal_resolutions {sorted(l1_temporal_resolutions)} "
                f"does not include all L2M temporal resolutions. Missing: {sorted(missing_l2m_in_l1)}. "
                f"L2M temporal_resolutions: {sorted(l2m_temporal_resolutions)}",
            )

        # Check L2E includes all L2M temporal resolutions
        missing_l2m_in_l2e = l2m_temporal_resolutions - l2e_temporal_resolutions
        if missing_l2m_in_l2e:
            print(
                f"WARNING. Sensor '{sensor_name}': L2E temporal_resolutions {sorted(l2e_temporal_resolutions)} "
                f"does not include all L2M temporal resolutions. Missing: {sorted(missing_l2m_in_l2e)}. "
                f"L2M temporal_resolutions: {sorted(l2m_temporal_resolutions)}",
            )


####------------------------------------------------------------------------------------------------


class TimeBlockStrategyOptions(CustomBaseModel):
    """Strategy options for time_block strategy."""

    freq: str = Field(..., description="Frequency for time block partitioning")

    @field_validator("freq")
    @classmethod
    def validate_freq(cls, v):
        """Validate frequency using check_freq function."""
        check_freq(v)
        return v


class EventStrategyOptions(CustomBaseModel):
    """Strategy options for event strategy."""

    variable: str = Field(..., description="Variable to define events")
    detection_threshold: int = Field(..., ge=0, description="Minimum number of drops")
    neighbor_min_size: int = Field(..., ge=0, description="Minimum neighbor size")
    neighbor_time_interval: str = Field(..., description="Neighbor time interval")
    event_max_time_gap: str = Field(..., description="Maximum time gap for events")
    event_min_duration: str = Field(..., description="Minimum event duration")
    event_min_size: int = Field(..., ge=0, description="Minimum event size")

    @field_validator("neighbor_time_interval", "event_max_time_gap", "event_min_duration")
    @classmethod
    def validate_time_intervals(cls, v):
        """Validate time interval strings using check_temporal_resolution."""
        check_temporal_resolution(v)
        return v


class ArchiveOptions(CustomBaseModel):
    """Archive options configuration."""

    strategy: Literal["time_block", "event"] = Field(..., description="Archiving strategy")
    strategy_options: Union[TimeBlockStrategyOptions, EventStrategyOptions] = Field(
        ...,
        description="Strategy-specific options",
    )
    folder_partitioning: str = Field(..., description="Folder partitioning scheme")

    @field_validator("folder_partitioning")
    @classmethod
    def validate_folder_partitioning(cls, v):
        """Validate folder partitioning."""
        check_folder_partitioning(v)
        return v

    @model_validator(mode="after")
    def validate_strategy_options(self):
        """Ensure strategy_options match the selected strategy."""
        expected_type = {
            "time_block": TimeBlockStrategyOptions,
            "event": EventStrategyOptions,
        }[self.strategy]

        if not isinstance(self.strategy_options, expected_type):
            raise ValueError(
                f"{self.strategy} strategy requires {expected_type.__name__}, "
                f"got {type(self.strategy_options).__name__}",
            )

        return self


class ArchiveOptionsTimeBlock(ArchiveOptions):
    """Archive options configuration for L0C and L1 products (time_block only)."""

    @field_validator("strategy", mode="after")
    @classmethod
    def validate_strategy_early(cls, v):
        """Validate that strategy is 'time_block' and fail early if not."""
        if v != "time_block":
            raise ValueError("L0C and L1 products require strategy 'time_block'.")
        return v


class RadarOptions(CustomBaseModel):
    """Radar simulation options."""

    frequency: Union[str, int, float, list[Union[str, int, float]]] = Field(
        ...,
        description="Radar frequency bands or numeric frequency values (in GHz)",
    )
    num_points: Union[int, float, list[Union[int, float]]] = Field(
        ...,
        description="Number of points for T-matrix simulation",
    )
    diameter_max: Union[int, float, list[Union[int, float]]] = Field(
        ...,
        description="Maximum diameter for T-matrix simulation",
    )
    canting_angle_std: Union[int, float, list[Union[int, float]]] = Field(
        ...,
        description="Canting angle standard deviation",
    )
    axis_ratio_model: Union[str, list[str]] = Field(
        ...,
        description="Axis ratio model",
    )
    permittivity_model: Union[str, list[str]] = Field(
        ...,
        description="Permittivity model",
    )
    water_temperature: Union[int, float, list[Union[int, float]]] = Field(
        ...,
        description="Water temperature in Celsius",
    )
    elevation_angle: Union[int, float, list[Union[int, float]]] = Field(
        ...,
        description="Elevation angle in degrees",
    )

    # Normalization: make sure all are lists
    @field_validator(
        "frequency",
        "axis_ratio_model",
        "permittivity_model",
        "num_points",
        "diameter_max",
        "canting_angle_std",
        "water_temperature",
        "elevation_angle",
        mode="before",
    )
    @classmethod
    def ensure_list(cls, v):
        """Normalize single values to lists."""
        if not isinstance(v, list):
            return [v]
        return v

    @field_validator("frequency")
    @classmethod
    def validate_frequency_bands(cls, frequencies):
        """Validate radar frequency bands."""
        return ensure_numerical_frequency(frequencies)

    @field_validator("axis_ratio_model")
    @classmethod
    def validate_axis_ratio_model(cls, axis_ratio_models):
        """Validate axis ratio models."""
        return [check_axis_ratio_model(axis_ratio_model) for axis_ratio_model in axis_ratio_models]

    @field_validator("permittivity_model")
    @classmethod
    def validate_permittivity_model(cls, permittivity_models):
        """Validate permittivity models."""
        return [check_permittivity_model(permittivity_model) for permittivity_model in permittivity_models]


class L2EProductOptions(CustomBaseModel):
    """L2E product-specific options."""

    compute_spectra: bool = Field(..., description="Whether to compute spectra")
    compute_percentage_contribution: bool = Field(
        ...,
        description="Whether to compute percentage contribution",
    )
    minimum_ndrops: int = Field(..., ge=0, description="Minimum number of drops")
    minimum_nbins: int = Field(..., ge=0, description="Minimum number of bins")
    minimum_rain_rate: float = Field(..., ge=0, description="Minimum rain rate threshold")
    fall_velocity_model: str = Field(..., description="Fall velocity model to use")
    minimum_diameter: float = Field(..., ge=0, description="Minimum diameter threshold")
    maximum_diameter: float = Field(..., gt=0, description="Maximum diameter threshold")
    minimum_velocity: float = Field(..., ge=0, description="Minimum velocity threshold")
    maximum_velocity: float = Field(..., gt=0, description="Maximum velocity threshold")
    above_velocity_fraction: Union[float, None] = Field(..., ge=0, le=1, description="Above velocity fraction")
    above_velocity_tolerance: float = Field(..., ge=0, description="Above velocity tolerance")
    below_velocity_fraction: Union[float, None] = Field(..., ge=0, le=1, description="Below velocity fraction")
    below_velocity_tolerance: float = Field(..., ge=0, description="Below velocity tolerance")
    maintain_drops_smaller_than: float = Field(..., ge=0, description="Maintain drops smaller than threshold")
    maintain_drops_slower_than: float = Field(..., ge=0, description="Maintain drops slower than threshold")
    maintain_smallest_drops: bool = Field(..., description="Whether to maintain smallest drops")
    remove_splashing_drops: bool = Field(..., description="Whether to remove splashing drops")

    @model_validator(mode="after")
    def validate_diameter_range(self):
        """Validate that maximum_diameter > minimum_diameter."""
        if self.maximum_diameter <= self.minimum_diameter:
            raise ValueError("maximum_diameter must be greater than minimum_diameter")
        return self

    @model_validator(mode="after")
    def validate_velocity_range(self):
        """Validate that maximum_velocity > minimum_velocity."""
        if self.maximum_velocity <= self.minimum_velocity:
            raise ValueError("maximum_velocity must be greater than minimum_velocity")
        return self

    @field_validator("fall_velocity_model")
    @classmethod
    def validate_fall_velocity_model(cls, fall_velocity_model):
        """Validate fall velocity model."""
        return check_rain_fall_velocity_model(fall_velocity_model)


class L2MProductOptions(CustomBaseModel):
    """L2M product-specific options."""

    fall_velocity_model: str = Field(..., description="Fall velocity model to use")
    diameter_min: float = Field(..., ge=0, description="Minimum diameter threshold")
    diameter_max: float = Field(..., gt=0, description="Maximum diameter threshold")
    diameter_spacing: float = Field(..., gt=0, description="Diameter spacing for grid")
    gof_metrics: bool = Field(..., description="Whether to compute goodness-of-fit metrics")
    minimum_ndrops: int = Field(..., ge=0, description="Minimum number of drops")
    minimum_nbins: int = Field(..., ge=0, description="Minimum number of bins with drops")
    minimum_rain_rate: float = Field(..., ge=0, description="Minimum rain rate threshold")

    @model_validator(mode="after")
    def validate_diameter_range(self):
        """Validate that diameter_max > diameter_min."""
        if self.diameter_max <= self.diameter_min:
            raise ValueError("diameter_max must be greater than diameter_min")
        return self

    @field_validator("fall_velocity_model")
    @classmethod
    def validate_fall_velocity_model(cls, fall_velocity_model):
        """Validate fall velocity model."""
        return check_rain_fall_velocity_model(fall_velocity_model)

    @model_validator(mode="after")
    def validate_diameter_grid(self):
        """Validate diameter grid configuration."""
        # Check that diameter_spacing is reasonable relative to the range
        diameter_range = self.diameter_max - self.diameter_min
        if self.diameter_spacing > diameter_range:
            raise ValueError("diameter_spacing cannot be larger than the diameter range.")
        return self


class TemporalResolutionsValidationMixin:
    """Mixin for temporal resolutions validation."""

    @field_validator("temporal_resolutions")
    @classmethod
    def validate_temporal_resolutions(cls, temporal_resolutions: list[str]):
        """Validate temporal resolutions list."""
        if not temporal_resolutions:
            raise ValueError("temporal_resolutions cannot be empty.")

        for temporal_resolution in temporal_resolutions:
            check_temporal_resolution(temporal_resolution)

        if len(temporal_resolutions) != len(set(temporal_resolutions)):
            raise ValueError("temporal_resolutions contains duplicates.")
        return temporal_resolutions


####------------------------------------------------------------------------------
#### Validation of L2M models settings


class L2MModelConfig(CustomBaseModel):
    """L2M model configuration validation."""

    psd_model: str = Field(..., description="PSD model name")
    optimization: str = Field(..., description="Optimization method")
    optimization_kwargs: dict[str, Any] = Field(..., description="Optimization-specific parameters")

    @field_validator("psd_model")
    @classmethod
    def validate_psd_model(cls, psd_model):
        """Validate psd_model."""
        valid_psd_models = PSD_MODELS
        if psd_model not in valid_psd_models:
            raise ValueError(f"Invalid psd_model '{psd_model}'. Must be one of {valid_psd_models}")
        return psd_model

    @field_validator("optimization")
    @classmethod
    def validate_optimization(cls, optimization):
        """Validate optimization method."""
        return check_optimization(optimization)

    @model_validator(mode="after")
    def validate_optimization_kwargs(self):
        """Validate that optimization_kwargs matches the optimization method."""
        # Use the existing validation function
        check_optimization_kwargs(
            optimization_kwargs=self.optimization_kwargs,
            optimization=self.optimization,
            psd_model=self.psd_model,
        )
        return self


def validate_l2m_model_configs(products_configs_dir: str):
    """
    Validate all L2M model configuration files.

    Parameters
    ----------
    products_configs_dir : str
        Path to products configuration directory.

    Raises
    ------
    ValidationError
        If any L2M model configuration is invalid.
    """
    # Get all L2M model configuration files
    model_settings_files = get_l2m_model_settings_files(products_configs_dir)

    validation_errors = []

    for model_file in model_settings_files:
        model_name = os.path.basename(model_file).replace(".yaml", "")
        try:
            # Load model configuration
            model_config = get_model_options(model_name, products_configs_dir=products_configs_dir)
            # Validate configuration
            L2MModelConfig(**model_config)
        except Exception as e:
            error_msg = f"L2M model '{model_name}' configuration validation failed:\n{e}"
            validation_errors.append(error_msg)

    # Report all validation errors at once
    if validation_errors:
        error_summary = f"\n{'='*80}\n".join(validation_errors)
        raise ValueError(
            f"L2M model configuration validation failed for {len(validation_errors)} model(s):\n\n"
            f"{'='*80}\n{error_summary}\n{'='*80}",
        )

    print("🎉 All L2M models configurations validated successfully!")


####------------------------------------------------------------------------------
#### Validation of DISDRODB products settings


class L1ProductConfig(CustomBaseModel):
    """L0C product configuration model."""

    archive_options: ArchiveOptionsTimeBlock = Field(..., description="Archive configuration options")


class L2EProductConfig(CustomBaseModel):
    """L2E product configuration model."""

    archive_options: ArchiveOptions = Field(..., description="Archive configuration options")
    product_options: L2EProductOptions = Field(..., description="L2E product-specific options")
    radar_enabled: bool = Field(..., description="Whether radar simulation is enabled")
    radar_options: RadarOptions = Field(..., description="Radar simulation options")


class L2MProductConfig(CustomBaseModel):
    """L2M product configuration model."""

    models: list[str] = Field(..., description="list of L2M models to use")
    archive_options: ArchiveOptions = Field(..., description="Archive configuration options")
    product_options: L2MProductOptions = Field(..., description="L2M product-specific options")
    radar_enabled: bool = Field(..., description="Whether radar simulation is enabled")
    radar_options: RadarOptions = Field(..., description="Radar simulation options")

    @field_validator("models", mode="after")
    @classmethod
    def validate_models(cls, models):
        """Validate L2M models list."""
        if not models:
            raise ValueError("'models' list cannot be empty")

        # Check for duplicates
        if len(models) != len(set(models)):
            raise ValueError("'models' list contains duplicates")

        # Retrieve products configuration directory
        products_configs_dir = get_products_configs_dir()

        # Get available model YAML files
        model_settings_paths = get_l2m_model_settings_files(products_configs_dir)

        # Get available model names
        available_models = [os.path.basename(path).replace(".yaml", "") for path in model_settings_paths]

        # Check each requested model has a corresponding YAML file
        missing_models = [model for model in models if model not in available_models]
        if missing_models:
            raise ValueError(
                f"L2M model configuration files not found for: {missing_models}. "
                f"Available models: {sorted(available_models)}",
            )
        return models


class L0CProductGlobalConfig(CustomBaseModel):
    """L0C product configuration model."""

    archive_options: ArchiveOptionsTimeBlock = Field(..., description="Archive configuration options")


class L1ProductGlobalConfig(L1ProductConfig, TemporalResolutionsValidationMixin):
    """L1 product configuration model."""

    temporal_resolutions: list[str] = Field(..., description="list of temporal resolution")


class L2EProductGlobalConfig(L2EProductConfig, TemporalResolutionsValidationMixin):
    """L1 product configuration model."""

    temporal_resolutions: list[str] = Field(..., description="list of temporal resolution")


class L2MProductGlobalConfig(L2MProductConfig, TemporalResolutionsValidationMixin):
    """L2M product configuration model."""

    temporal_resolutions: list[str] = Field(..., description="list of temporal resolutions")


def validate_all_product_yaml_files(products_configs_dir):
    """
    Validate all DISDRODB product YAML configuration files.

    Raises
    ------
    ValidationError
        If any YAML file validation fails with detailed information.
    """
    # Define product validators mapping
    product_global_validators = {
        "L0C": L0CProductGlobalConfig,
        "L1": L1ProductGlobalConfig,
        "L2E": L2EProductGlobalConfig,
        "L2M": L2MProductGlobalConfig,
    }

    # Define custom temporal resolution validators (without temporal_resolutions field)
    custom_temporal_validators = {
        "L1": L1ProductConfig,
        "L2E": L2EProductConfig,
        "L2M": L2MProductConfig,
    }

    products = ["L0C", "L1", "L2E", "L2M"]
    sensor_names = available_sensor_names()

    validation_errors = []

    for product in products:
        # 1. Test global YAML (product-level)
        product_options = get_product_options(product=product, products_configs_dir=products_configs_dir)
        try:
            validator_class = product_global_validators[product]
            validator_class(**product_options)
        except Exception as e:
            error_msg = f"Global {product} configuration validation failed:\n{e}"
            validation_errors.append(error_msg)

        # 2. Test YAML per sensor
        for sensor_name in sensor_names:
            # Test sensor-level global YAML
            product_options = get_product_options(
                product=product,
                sensor_name=sensor_name,
                products_configs_dir=products_configs_dir,
            )
            try:
                validator_class = product_global_validators[product]
                validator_class(**product_options)
            except Exception as e:
                error_msg = f"{product}/{sensor_name} configuration validation failed:\n" f"{e}"
                validation_errors.append(error_msg)
                continue

            # 3. Test custom temporal resolution YAML files for given sensor (not for L0C)
            if "temporal_resolutions" in product_options:
                # Retrieve product validator class
                for temporal_resolution in product_options["temporal_resolutions"]:
                    try:
                        validator_class = custom_temporal_validators[product]
                        custom_product_options = get_product_options(
                            product=product,
                            temporal_resolution=temporal_resolution,
                            sensor_name=sensor_name,
                            products_configs_dir=products_configs_dir,
                        )
                        validator_class(**custom_product_options)
                    except Exception as e:
                        error_msg = (
                            f"{product}/{sensor_name}/{temporal_resolution} configuration validation failed:\n" f"{e}"
                        )
                        validation_errors.append(error_msg)

    # Report all validation errors at once
    if validation_errors:
        error_summary = f"\n{'='*80}\n".join(validation_errors)
        raise ValueError(
            f"YAML configuration validation failed for {len(validation_errors)} file(s):\n\n"
            f"{'='*80}\n{error_summary}\n{'='*80}",
        )

    print("\n🎉 All products configurations validated successfully!")


####-----------------------------------------------------------------------------------------------.
#### Wrapper


def validate_products_configurations(products_configs_dir=None):
    """Validate the DISDRODB products configuration files."""
    import disdrodb

    products_configs_dir = get_products_configs_dir(products_configs_dir=products_configs_dir)

    with disdrodb.config.set({"products_configs_dir": products_configs_dir}):
        # Validate directory structure first
        validate_product_configuration_structure(products_configs_dir)

        # Validate all DISDRODB products global configuration files with pydantic
        validate_all_product_yaml_files(products_configs_dir)

        # Check temporal resolution consistency
        validate_temporal_resolution_consistency(products_configs_dir)

        # Validate L2M model options
        validate_l2m_model_configs(products_configs_dir)


####-----------------------------------------------------------------------------------------------.
