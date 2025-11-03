import os
from pathlib import Path
from disdrodb.api.configs import available_sensor_names

from typing import List, Literal, Union, Dict, Any
from pydantic import BaseModel, Field, model_validator, field_validator
from disdrodb.api.checks import check_temporal_resolution, check_folder_partitioning
from disdrodb.utils.archiving import check_freq


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
        global_yaml = product_dir / "global.yaml"
        if not global_yaml.exists():
            raise FileNotFoundError(f"Required global.yaml file not found in: {product_dir}")
        
        # Check subdirectories within product directories (L0C-L2E)
        if product in ["L0C", "L1", "L2E"]:
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
        List of available sensor names.
    product_name : str
        Name of the product (for error messages).
    """
    # Get all subdirectories (excluding files)
    subdirs = [d for d in product_dir.iterdir() if d.is_dir()]
    
    for subdir in subdirs:
        sensor_name = subdir.name
        
        # Check if subdirectory name corresponds to a valid sensor name
        if sensor_name not in available_sensors:
            raise ValueError(
                f"Invalid sensor directory '{sensor_name}' in {product_name}. "
                f"Must be one of: {available_sensors}"
            )
        
        # Check that sensor directory contains at least one YAML file
        yaml_files = list(subdir.glob("*.yaml")) + list(subdir.glob("*.yml"))
        if not yaml_files:
            raise FileNotFoundError(
                f"No YAML files found in sensor directory: {subdir}. "
                f"Sensor directories must contain at least one YAML configuration file."
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
            f"MODELS directory must contain at least one model YAML file."
        )


class TimeBlockStrategyOptions(BaseModel):
    """Strategy options for time_block strategy."""
    freq: str = Field(..., description="Frequency for time block partitioning")

    @field_validator("freq")
    @classmethod
    def validate_freq(cls, v):
        """Validate frequency using check_freq function."""
        check_freq(v)
        return v


class EventStrategyOptions(BaseModel):
    """Strategy options for event strategy."""
    min_drops: int = Field(..., ge=0, description="Minimum number of drops")
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

class ArchiveOptions(BaseModel):
    """Archive options configuration."""

    strategy: Literal["time_block", "event"] = Field(..., description="Archiving strategy")
    strategy_options: Union[TimeBlockStrategyOptions, EventStrategyOptions] = Field(
        ..., description="Strategy-specific options"
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
        """Validate that strategy_options match the selected strategy."""
        if self.strategy == "time_block":
            if not isinstance(self.strategy_options, TimeBlockStrategyOptions):
                if isinstance(self.strategy_options, dict):
                    try:
                        self.strategy_options = TimeBlockStrategyOptions(**self.strategy_options)
                    except Exception as e:
                        raise ValueError(f"Invalid strategy_options for time_block strategy: {e}")
                else:
                    raise ValueError("time_block strategy requires TimeBlockStrategyOptions")

        elif self.strategy == "event":
            if not isinstance(self.strategy_options, EventStrategyOptions):
                if isinstance(self.strategy_options, dict):
                    try:
                        self.strategy_options = EventStrategyOptions(**self.strategy_options)
                    except Exception as e:
                        raise ValueError(f"Invalid strategy_options for event strategy: {e}")
                else:
                    raise ValueError("event strategy requires EventStrategyOptions")

        return self


class TemporalResolutions(BaseModel):
    """Model for validating a list of temporal resolutions."""
    temporal_resolutions: List[str] = Field(
        ..., description="List of supported temporal resolutions"
    )

    @field_validator("temporal_resolutions")
    @classmethod
    def validate_temporal_resolutions(cls, v: List[str]):
        """Validate each temporal resolution in the list."""
        if not v:
            raise ValueError("temporal_resolutions cannot be empty")

        for temporal_resolution in v:
            check_temporal_resolution(temporal_resolution)

        # Check for duplicates
        if len(v) != len(set(v)):
            raise ValueError("temporal_resolutions contains duplicates")

        return v


class L1ProductConfig(BaseModel):
    """L1 product configuration model."""
    temporal_resolutions: List[str] = Field(..., description="List of temporal resolution")
    archive_options: ArchiveOptions = Field(..., description="Archive configuration options")
    
    @field_validator("temporal_resolutions")
    @classmethod
    def validate_temporal_resolutions(cls, v: List[str]):
       if not v:
           raise ValueError("temporal_resolutions cannot be empty")

       for temporal_resolution in v:
           check_temporal_resolution(temporal_resolution)

       if len(v) != len(set(v)):
           raise ValueError("temporal_resolutions contains duplicates")

       return v
  

product_options = get_product_options(product="L1", sensor_name="PARSIVEL2") # temporal_resolution="1MIN"
L1ProductConfig(**product_options)

# global temporal_resolutions
# custom product: without temporal_resolutions 


def validate_l1_product_options(product_options: str) -> L1ProductConfig:
    """Validate L1 product options."""
    return L1ProductConfig(**product_options)






 


def validate_product_configuration(products_configs_dir):
    """Validate the DISDRODB products configuration files."""
    # Validate directory structure first
    validate_product_configuration_structure(products_configs_dir)
    
    # TODO: Implement validation of DISDRODB products configuration files with pydantic
    # TODO: Raise warning if L1 temporal resolutions does not includes all temporal resolutions of L2 products.
    # TODO: Raise warning if L2E temporal resolutions does not includes all temporal resolutions of L2M products.
    # if stategy_event, check neighbor_time_interval >= sample_interval !
    # if temporal_resolution_to_seconds(neighbor_time_interval) < temporal_resolution_to_seconds(sample_interval):
    #     msg = "'neighbor_time_interval' must be at least equal to the dataset sample interval ({sample_interval})"
    #     raise ValueError(msg)
    
    pass
