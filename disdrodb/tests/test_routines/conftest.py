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
"""Fixtures for testing validation of DISDRODB product options."""
import os
import pytest 


@pytest.fixture
def enable_config_validation():
    """
    Fixture that sets DISDRODB_VALIDATION_FLAG to enable config-based validation.
    
    This allows get_products_configs_dir() to use the configured products_configs_dir
    instead of the default test directory during pytest execution.
    """
    # Set the validation flag
    os.environ["DISDRODB_VALIDATION_FLAG"] = "1"
    
    yield
    
    # Clean up - remove the flag
    if "DISDRODB_VALIDATION_FLAG" in os.environ:
        del os.environ["DISDRODB_VALIDATION_FLAG"]
        
        
@pytest.fixture
def tmp_products_configs_dir(tmp_path, enable_config_validation):
    """Fixture to set up temporary products configuration directory."""
    import disdrodb 
    
    # Store original config
    original_config = disdrodb.config.get("products_configs_dir", None)
    
    # Set temporary config
    disdrodb.config.set({"products_configs_dir": str(tmp_path)})
    
    yield tmp_path
    
    # Restore original config
    disdrodb.config.set({"products_configs_dir": original_config})
