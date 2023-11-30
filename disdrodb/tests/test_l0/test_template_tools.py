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
"""Test DISDRODB L0 template tools."""

import pytest 
import numpy as np 
import pandas as pd 

from disdrodb.l0.template_tools import (
     get_natural_ndigits,
     get_decimal_ndigits,
     get_nchar,
     get_ndigits,
     get_possible_keys,
     str_has_decimal_digits,
     str_is_integer,
     str_is_number,
     arr_has_constant_nchar,
     
     
     infer_column_names,
     print_df_column_names,
     print_df_columns_unique_values,
     print_df_first_n_rows,
     print_df_random_n_rows,
     print_df_summary_stats,
     print_df_with_any_nan_rows,
     print_valid_l0_column_names,
     search_possible_columns,
     check_column_names,

     get_df_columns_unique_values_dict,
     get_field_nchar_dict,
     get_field_ndigits_decimals_dict,
     get_field_ndigits_dict,
     get_field_ndigits_natural_dict,
     get_l0a_dtype,

)

class Test_Arr_Has_Constant_Char:
    """Test arr_has_constant_nchar."""
    
    def test_constant_nchar(self):
        arr = np.array(["abc", "def", "ghi"])
        assert arr_has_constant_nchar(arr)
    
    def test_not_constant_nchar(self):
        arr = np.array(["abc", "de", "fghij"])
        assert not arr_has_constant_nchar(arr)
    
    def test_empty_array(self):
        arr = np.array([], dtype="U")
        assert arr_has_constant_nchar(arr)  
        
        arr = np.array([], dtype="O")
        assert arr_has_constant_nchar(arr)  
        
        arr = np.array([], dtype=int)
        assert arr_has_constant_nchar(arr)  
          
        arr = np.array([])
        assert arr_has_constant_nchar(arr)  
    
    def test_numeric_array(self):
        arr = np.array([123, 456, 789])
        assert arr_has_constant_nchar(arr) 
    
        arr = np.array([1.23, 4.56, 7.89])
        assert arr_has_constant_nchar(arr)
           
            

@pytest.mark.parametrize("test_input, expected", [
    ("123.456", True),
    ("123", True),
    ("-123.456", True),
    ("1e10", True),
    ("abc", False),
    ("123abc", False),
    ("", False),
    ("   ", False)
])
def test_str_is_number(test_input, expected):
    assert str_is_number(test_input) == expected
    
def test_str_is_integer():
    assert str_is_integer("123")
    assert str_is_integer("-123")
    assert not str_is_integer("123.456")
    assert not str_is_integer("abc")
    assert not str_is_integer("123abc")
    assert not str_is_integer("")
    
    

def test_str_has_decimal_digits():
    assert not str_has_decimal_digits("123")
    assert not str_has_decimal_digits("-123")
    assert str_has_decimal_digits("123.456")
    assert str_has_decimal_digits("-123.456")
    assert not str_has_decimal_digits("abc")
    assert not str_has_decimal_digits("123abc")
    assert not str_has_decimal_digits("")


def test_get_decimal_ndigits():
    assert get_decimal_ndigits("123.456") == 3
    assert get_decimal_ndigits("-123.456") == 3
    assert get_decimal_ndigits("123") == 0
    assert get_decimal_ndigits("-123") == 0
    assert get_decimal_ndigits("123.0") == 1
    assert get_decimal_ndigits("-123.0") == 1
    assert get_decimal_ndigits("abc") == 0
    
    
def test_get_natural_ndigits():
    assert get_natural_ndigits("123.456") == 3
    assert get_natural_ndigits("-123.456") == 3 # BUG 4
    assert get_natural_ndigits("123") == 3
    assert get_natural_ndigits("-123") == 3
    assert get_natural_ndigits("abc") == 0
    
def test_get_ndigits():
    assert get_ndigits("123.456") == 6
    assert get_ndigits("-123.456") == 6 # BUG 7
    assert get_ndigits("123") == 3
    assert get_ndigits("-123") == 3
    assert get_ndigits("abc") == 0
    
def test_get_nchar():
    assert get_nchar("123.456") == 7
    assert get_nchar("-123.456") == 8
    assert get_nchar("123") == 3
    assert get_nchar("-123") == 4
    assert get_nchar("abc") == 3
    
def test_get_possible_keys():
    test_dict = {"a": "apple", "b": "banana", "c": "apple"}
    assert get_possible_keys(test_dict, "apple") == {"a", "c"}
    assert get_possible_keys(test_dict, "banana") == {"b"}
    assert get_possible_keys(test_dict, "cherry") == set()
    
    
def test_print_df_column_names(capfd):
    df = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
    print_df_column_names(df)
    out, err = capfd.readouterr()
    assert "Column 0 : A" in out
    assert "Column 1 : B" in out
    
    
def test_print_df_random_n_rows(capfd):
    df = pd.DataFrame({"A": [1, 2, 3, 4, 5], "B": [6, 7, 8, 9, 10]})
    print_df_random_n_rows(df, n=2)
    out, err = capfd.readouterr()
    assert "- Column 0 (A) :" in out
    assert "- Column 1 (B) :" in out

def test_print_df_first_n_rows(capfd):
    # Create a sample DataFrame
    df = pd.DataFrame({
        'A': [1, 2, 3, 4, 5, 6],
        'B': ['a', 'b', 'c', 'd', 'e', 'f']
    })

    # Call the function
    print_df_first_n_rows(df, n=3)

    # Capture the stdout
    out, _ = capfd.readouterr()

    # Assert the output
    expected_output = (
        " - Column 0 ( A ):\n"
        "      [1 2 3 4]\n"
        " - Column 1 ( B ):\n"
        "      ['a' 'b' 'c' 'd']\n"
    )
    assert out == expected_output

def test_print_df_first_n_rows_no_column_names(capfd):
    # Create a sample DataFrame
    df = pd.DataFrame({
        'A': [1, 2, 3, 4, 5, 6],
        'B': ['a', 'b', 'c', 'd', 'e', 'f']
    })

    # Call the function with column_names set to False
    print_df_first_n_rows(df, n=3, column_names=False)

    # Capture the stdout
    out, _ = capfd.readouterr()

    # Assert the output
    expected_output = (
        " - Column 0 :\n"
        "      [1 2 3 4]\n"
        " - Column 1 :\n"
        "      ['a' 'b' 'c' 'd']\n"
    )
    assert out == expected_output
 

