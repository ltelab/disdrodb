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
"""Check DISDRODB L0 configuration files."""
import pytest

from disdrodb.api.configs import available_sensor_names
from disdrodb.l0.check_configs import (
    L0BEncodingSchema,
    check_all_sensors_configs,
    check_bin_consistency,
    check_cf_attributes,
    check_l0a_encoding,
    check_l0b_encoding,
    check_raw_array,
    check_raw_data_format,
    check_variable_consistency,
    check_yaml_files_exists,
)


@pytest.mark.parametrize("sensor_name", available_sensor_names())
def test_check_yaml_files_exists(sensor_name):
    """Check presence of required L0 YAML files for all sensors."""
    check_yaml_files_exists(sensor_name)


@pytest.mark.parametrize("sensor_name", available_sensor_names())
def test_check_variable_consistency(sensor_name):
    """Check variable consistency across L0 YAML files for all sensors."""
    check_variable_consistency(sensor_name)


@pytest.mark.parametrize("sensor_name", available_sensor_names())
def test_check_l0b_encoding(sensor_name):
    """Check validity of L0B encodings for all sensors."""
    check_l0b_encoding(sensor_name)


@pytest.mark.parametrize("sensor_name", available_sensor_names())
def test_check_l0a_encoding(sensor_name):
    """Check validity of L0A encodings for all sensors."""
    check_l0a_encoding(sensor_name)


@pytest.mark.parametrize("sensor_name", available_sensor_names())
def test_check_raw_data_format(sensor_name):
    """Check validity of raw_data_format YAML file for all sensors."""
    check_raw_data_format(sensor_name)


@pytest.mark.parametrize("sensor_name", available_sensor_names())
def test_check_cf_attributes(sensor_name):
    """Check validity of CF_attributes L0 YAML file for all sensors."""
    check_cf_attributes(sensor_name)


@pytest.mark.parametrize("sensor_name", available_sensor_names())
def test_check_bin_consistency(sensor_name):
    """Check consistency of bin tables for all sensors."""
    check_bin_consistency(sensor_name)


@pytest.mark.parametrize("sensor_name", available_sensor_names())
def test_check_raw_array(sensor_name):
    """Check consistency of default chunking for raw arrays for all sensor."""
    check_raw_array(sensor_name)


def test_check_all_sensors_configs():
    """Check validity of all DISDRODB L0 configs file."""
    check_all_sensors_configs()


class TestL0BEncodingSchema:
    """Test NetCDF encoding schema."""

    def test_valid_contiguous_config(self):
        """Test a valid contiguous configuration."""
        config = {
            "contiguous": True,
            "dtype": "float32",
            "zlib": False,
            "complevel": 0,
            "shuffle": False,
            "fletcher32": False,
            "_FillValue": None,
            "chunksizes": None,
        }
        schema = L0BEncodingSchema(**config)
        assert schema.contiguous is True
        assert schema.zlib is False
        assert schema.fletcher32 is False

    def test_valid_non_contiguous_config(self):
        """Test a valid non-contiguous configuration."""
        config = {
            "contiguous": False,
            "dtype": "float32",
            "zlib": True,
            "complevel": 4,
            "shuffle": True,
            "fletcher32": True,
            "_FillValue": None,
            "chunksizes": [100, 200],
        }
        schema = L0BEncodingSchema(**config)
        assert schema.contiguous is False
        assert schema.zlib is True
        assert schema.fletcher32 is True
        assert schema.chunksizes == [100, 200]

    def test_valid_integer_config_with_fillvalue(self):
        """Test valid integer configuration with proper _FillValue."""
        config = {
            "contiguous": False,
            "dtype": "uint16",
            "zlib": True,
            "complevel": 1,
            "shuffle": True,
            "fletcher32": True,
            "_FillValue": 65535,  # np.iinfo('uint16').max
            "chunksizes": [10, 20],
        }
        schema = L0BEncodingSchema(**config)
        assert schema.dtype == "uint16"
        assert schema.FillValue == 65535

    def test_check_chunksizes_and_zlib_missing_chunksizes(self):
        """Test that non-contiguous requires chunksizes."""
        config = {
            "contiguous": False,
            "dtype": "float32",
            "zlib": True,
            "complevel": 4,
            "shuffle": True,
            "fletcher32": True,
            "_FillValue": None,
            "chunksizes": None,  # Missing chunksizes for non-contiguous
        }
        with pytest.raises(ValueError, match="'chunksizes' must be defined if 'contiguous' is False"):
            L0BEncodingSchema(**config)

    def test_check_chunksizes_and_zlib_empty_chunksizes(self):
        """Test that non-contiguous with empty chunksizes fails."""
        config = {
            "contiguous": False,
            "dtype": "float32",
            "zlib": True,
            "complevel": 4,
            "shuffle": True,
            "fletcher32": True,
            "_FillValue": None,
            "chunksizes": [],  # Empty chunksizes for non-contiguous
        }
        with pytest.raises(ValueError, match="'chunksizes' must be defined if 'contiguous' is False"):
            L0BEncodingSchema(**config)

    def test_check_contiguous_and_zlib_conflict(self):
        """Test that contiguous=True with zlib=True fails."""
        config = {
            "contiguous": True,
            "dtype": "float32",
            "zlib": True,  # Should be False when contiguous=True
            "complevel": 0,
            "shuffle": False,
            "fletcher32": False,
            "_FillValue": None,
            "chunksizes": None,
        }
        with pytest.raises(ValueError, match="'zlib' must be set to False if 'contiguous' is True"):
            L0BEncodingSchema(**config)

    def test_check_contiguous_and_fletcher32_conflict(self):
        """Test that contiguous=True with fletcher32=True fails."""
        config = {
            "contiguous": True,
            "dtype": "float32",
            "zlib": False,
            "complevel": 0,
            "shuffle": False,
            "fletcher32": True,  # Should be False when contiguous=True
            "_FillValue": None,
            "chunksizes": None,
        }
        with pytest.raises(ValueError, match="'fletcher32' must be set to False if 'contiguous' is True"):
            L0BEncodingSchema(**config)

    def test_check_integer_fillvalue_missing(self):
        """Test that integer dtypes require _FillValue."""
        config = {
            "contiguous": False,
            "dtype": "int32",
            "zlib": True,
            "complevel": 4,
            "shuffle": True,
            "fletcher32": True,
            "_FillValue": None,  # Missing _FillValue for integer type
            "chunksizes": [100],
        }
        with pytest.raises(ValueError, match="'_FillValue' must be specified for integer dtype 'int32'"):
            L0BEncodingSchema(**config)

    def test_check_integer_fillvalue_wrong_value(self):
        """Test that integer dtypes require correct maximum _FillValue."""
        config = {
            "contiguous": False,
            "dtype": "uint8",
            "zlib": True,
            "complevel": 4,
            "shuffle": True,
            "fletcher32": True,
            "_FillValue": 100,  # Should be 255 for uint8
            "chunksizes": [100],
        }
        with pytest.raises(
            ValueError,
            match="should be set to the maximum allowed value",
        ):
            L0BEncodingSchema(**config)

    @pytest.mark.parametrize(
        ("dtype", "expected_max"),
        [
            ("int8", 127),
            ("int16", 32767),
            ("int32", 2147483647),
            ("int64", 9223372036854775807),
            ("uint8", 255),
            ("uint16", 65535),
            ("uint32", 4294967295),
            ("uint64", 18446744073709551615),
        ],
    )
    def test_integer_dtypes_with_correct_fillvalue(self, dtype, expected_max):
        """Test all integer dtypes with correct _FillValue."""
        config = {
            "contiguous": False,
            "dtype": dtype,
            "zlib": True,
            "complevel": 4,
            "shuffle": True,
            "fletcher32": True,
            "_FillValue": expected_max,
            "chunksizes": [100],
        }
        schema = L0BEncodingSchema(**config)
        assert schema.dtype == dtype
        assert schema.FillValue == expected_max

    @pytest.mark.parametrize("dtype", ["float32", "float64"])
    def test_float_dtypes_no_fillvalue(self, dtype):
        """Test that float dtypes don't require _FillValue."""
        config = {
            "contiguous": False,
            "dtype": dtype,
            "zlib": True,
            "complevel": 4,
            "shuffle": True,
            "fletcher32": True,
            "_FillValue": None,
            "chunksizes": [100],
        }
        schema = L0BEncodingSchema(**config)
        assert schema.dtype == dtype
        assert schema.FillValue is None

    def test_complex_validation_combination(self):
        """Test multiple validation failures at once."""
        config = {
            "contiguous": True,
            "dtype": "int16",
            "zlib": True,  # Invalid: should be False for contiguous
            "complevel": 4,
            "shuffle": False,
            "fletcher32": True,  # Invalid: should be False for contiguous
            "_FillValue": None,  # Invalid: required for integer types
            "chunksizes": None,
        }
        # The first validation error encountered will be raised
        with pytest.raises(ValueError, match="'_FillValue' must be specified for integer dtype 'int16'"):
            L0BEncodingSchema(**config)

    def test_single_integer_chunksize(self):
        """Test configuration with single integer chunksize."""
        config = {
            "contiguous": False,
            "dtype": "float32",
            "zlib": True,
            "complevel": 4,
            "shuffle": True,
            "fletcher32": True,
            "_FillValue": None,
            "chunksizes": 100,  # Single integer instead of list
        }
        schema = L0BEncodingSchema(**config)
        assert schema.chunksizes == 100

    def test_contiguous_true_valid_combination(self):
        """Test valid combination when contiguous=True."""
        config = {
            "contiguous": True,
            "dtype": "float64",
            "zlib": False,  # Required for contiguous
            "complevel": 0,
            "shuffle": False,
            "fletcher32": False,  # Required for contiguous
            "_FillValue": None,
            "chunksizes": None,
        }
        schema = L0BEncodingSchema(**config)
        assert schema.contiguous is True
        assert schema.zlib is False
        assert schema.fletcher32 is False
        assert schema.chunksizes is None
