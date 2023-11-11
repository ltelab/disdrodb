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
"""Test DISDRODB configs."""
import os 
import pytest 
from disdrodb.api.configs import ( 
    get_sensor_configs_dir, 
    read_config_file,
    available_sensor_names,
    )

def test_available_sensor_names(): 
    assert len(available_sensor_names()) >= 3


@pytest.mark.parametrize("product", ["L0A"])
@pytest.mark.parametrize("sensor_name", available_sensor_names())
def test_get_sensor_configs_dir(sensor_name, product):
    config_sensor_dir = get_sensor_configs_dir(sensor_name=sensor_name, product=product)
    assert os.path.isdir(config_sensor_dir)
    
    
def test_invalid_sensor_name():
    with pytest.raises(ValueError):
         get_sensor_configs_dir(sensor_name="NOT_EXIST", product="L0A")
         
    with pytest.raises(ValueError):
         read_config_file(sensor_name="OTT_Parsivel", product="L0A", filename="UNEXISTENT.yml")
    