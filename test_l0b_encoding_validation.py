#!/usr/bin/env python3
"""Test script for L0BEncodingSchema validation."""

import numpy as np
from pydantic import ValidationError

# Import the schema (assuming it works when pydantic is available)
try:
    from disdrodb.l0.check_configs import L0BEncodingSchema

    def test_integer_fillvalue_validation():
        """Test the integer _FillValue validation."""
        # Test 1: Valid case - int8 with max value as _FillValue
        valid_config = {
            "contiguous": False,
            "dtype": "int8",
            "zlib": True,
            "complevel": 1,
            "shuffle": True,
            "fletcher32": True,
            "_FillValue": 127,  # np.iinfo('int8').max
            "chunksizes": [10, 20],
        }

        try:
            schema = L0BEncodingSchema(**valid_config)
            print("✓ Valid int8 config passed")
        except Exception as e:
            print(f"✗ Valid int8 config failed: {e}")

        # Test 2: Invalid case - int8 without _FillValue
        invalid_config_no_fillvalue = {
            "contiguous": False,
            "dtype": "int8",
            "zlib": True,
            "complevel": 1,
            "shuffle": True,
            "fletcher32": True,
            "_FillValue": None,
            "chunksizes": [10, 20],
        }

        try:
            schema = L0BEncodingSchema(**invalid_config_no_fillvalue)
            print("✗ int8 without _FillValue should have failed")
        except ValidationError as e:
            print("✓ int8 without _FillValue correctly failed")
            print(f"  Error: {e}")

        # Test 3: Invalid case - int8 with wrong _FillValue
        invalid_config_wrong_fillvalue = {
            "contiguous": False,
            "dtype": "int8",
            "zlib": True,
            "complevel": 1,
            "shuffle": True,
            "fletcher32": True,
            "_FillValue": 100,  # Should be 127 for int8
            "chunksizes": [10, 20],
        }

        try:
            schema = L0BEncodingSchema(**invalid_config_wrong_fillvalue)
            print("✗ int8 with wrong _FillValue should have failed")
        except ValidationError as e:
            print("✓ int8 with wrong _FillValue correctly failed")
            print(f"  Error: {e}")

        # Test 4: Valid case - uint16 with max value
        valid_uint16_config = {
            "contiguous": False,
            "dtype": "uint16",
            "zlib": True,
            "complevel": 1,
            "shuffle": True,
            "fletcher32": True,
            "_FillValue": 65535,  # np.iinfo('uint16').max
            "chunksizes": [10, 20],
        }

        try:
            schema = L0BEncodingSchema(**valid_uint16_config)
            print("✓ Valid uint16 config passed")
        except Exception as e:
            print(f"✗ Valid uint16 config failed: {e}")

        # Test 5: Valid case - float32 (should not require _FillValue validation)
        float_config = {
            "contiguous": False,
            "dtype": "float32",
            "zlib": True,
            "complevel": 1,
            "shuffle": True,
            "fletcher32": True,
            "_FillValue": None,
            "chunksizes": [10, 20],
        }

        try:
            schema = L0BEncodingSchema(**float_config)
            print("✓ float32 config without _FillValue passed (as expected)")
        except Exception as e:
            print(f"✗ float32 config without _FillValue failed: {e}")

    if __name__ == "__main__":
        print("Testing L0BEncodingSchema integer _FillValue validation...")
        print(f"np.iinfo('int8').max = {np.iinfo('int8').max}")
        print(f"np.iinfo('uint16').max = {np.iinfo('uint16').max}")
        print()
        test_integer_fillvalue_validation()

except ImportError as e:
    print(f"Cannot run test due to import error: {e}")
    print("This is expected if pydantic is not installed.")
