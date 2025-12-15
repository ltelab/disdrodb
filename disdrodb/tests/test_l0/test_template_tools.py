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
"""Test DISDRODB L0 template tools."""

import numpy as np
import pandas as pd
import pytest

from disdrodb.l0.template_tools import (
    _get_possible_keys,
    _has_constant_characters,
    check_column_names,
    get_decimal_ndigits,
    get_df_columns_unique_values_dict,
    get_natural_ndigits,
    get_nchar,
    get_ndigits,
    get_unique_sorted_values,
    infer_column_names,
    print_allowed_column_names,
    print_df_column_names,
    print_df_columns_unique_values,
    print_df_first_n_rows,
    print_df_random_n_rows,
    print_df_summary_stats,
    print_df_with_any_nan_rows,
    str_has_decimal_digits,
    str_is_integer,
    str_is_number,
)


@pytest.mark.parametrize(
    ("test_input", "expected"),
    [
        ("123.456", True),
        ("123", True),
        ("-123.456", True),
        ("1e10", True),
        ("abc", False),
        ("123abc", False),
        ("", False),
        ("   ", False),
    ],
)
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
    assert get_natural_ndigits("-123.456") == 4
    assert get_natural_ndigits("123") == 3
    assert get_natural_ndigits("-123") == 4
    assert get_natural_ndigits("abc") == 0


def test_get_ndigits():
    assert get_ndigits("123.456") == 6
    assert get_ndigits("-123.456") == 7
    assert get_ndigits("123") == 3
    assert get_ndigits("-123") == 4
    assert get_ndigits("abc") == 0


def test_get_nchar():
    assert get_nchar("123.456") == 7
    assert get_nchar("-123.456") == 8
    assert get_nchar("123") == 3
    assert get_nchar("-123") == 4
    assert get_nchar("abc") == 3


def test__get_possible_keys():
    test_dict = {"a": "apple", "b": "banana", "c": "apple"}
    assert _get_possible_keys(test_dict, "apple") == {"a", "c"}
    assert _get_possible_keys(test_dict, "banana") == {"b"}
    assert _get_possible_keys(test_dict, "cherry") == set()


def test_print_df_random_n_rows(capfd):
    df = pd.DataFrame({"A": [1, 2, 3, 4, 5], "B": [6, 7, 8, 9, 10]})
    print_df_random_n_rows(df, n=2)
    out, _ = capfd.readouterr()
    assert "- Column 0 ( A ):" in out
    assert "- Column 1 ( B ):" in out


def test_print_df_random_n_rows_without_column_names(capfd):
    df = pd.DataFrame({"A": [1, 2, 3, 4, 5], "B": [6, 7, 8, 9, 10]})
    print_df_random_n_rows(df, n=2, print_column_names=False)
    out, _ = capfd.readouterr()
    assert "- Column 0 :" in out
    assert "- Column 1 :" in out


def test_print_df_first_n_rows(capfd):
    # Create a sample DataFrame
    df = pd.DataFrame({"A": [1, 2, 3, 4, 5, 6], "B": ["a", "b", "c", "d", "e", "f"]})

    # Call the function
    print_df_first_n_rows(df, n=3)

    # Capture the stdout
    out, _ = capfd.readouterr()

    # Assert the output
    expected_output = " - Column 0 ( A ):\n      [1 2 3 4]\n - Column 1 ( B ):\n      ['a' 'b' 'c' 'd']\n"
    assert out == expected_output


def test_print_df_first_n_rows_without_column_names(capfd):
    # Create a sample DataFrame
    df = pd.DataFrame({"A": [1, 2, 3, 4, 5, 6], "B": ["a", "b", "c", "d", "e", "f"]})

    # Call the function with column_names set to False
    print_df_first_n_rows(df, n=3, print_column_names=False)

    # Capture the stdout
    out, _ = capfd.readouterr()

    # Assert the output
    expected_output = " - Column 0 :\n      [1 2 3 4]\n - Column 1 :\n      ['a' 'b' 'c' 'd']\n"
    assert out == expected_output


def test_print_df_column_names(capfd):
    df = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
    print_df_column_names(df)
    out, _ = capfd.readouterr()
    assert "Column 0 : A" in out
    assert "Column 1 : B" in out


def test_print_allowed_column_names(capfd):
    print_allowed_column_names(sensor_name="PARSIVEL")
    out, _ = capfd.readouterr()
    assert "['air_temperature'," in out


def test_print_invalid_column_indices():
    df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6], "C": [7, 8, 9]})

    with pytest.raises(ValueError):
        print_df_summary_stats(df, column_indices=100)

    with pytest.raises(ValueError):
        print_df_summary_stats(df, column_indices=-1)

    with pytest.raises(TypeError):
        print_df_summary_stats(df, column_indices=1.0)

    with pytest.raises(TypeError):
        print_df_summary_stats(df, column_indices="A")


class Test_Print_Df_Summary_Stats:
    """Test print_df_summary_stats."""

    def test_print_all_columns(self, capfd):
        df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
        print_df_summary_stats(df)
        out, _ = capfd.readouterr()
        assert "Column 0 ( A ):" in out
        assert "Column 1 ( B ):" in out

    @pytest.mark.parametrize("column_indices", [[0, 1], slice(0, 2)])
    def test_print_specific_columns(self, capfd, column_indices):
        df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6], "C": [7, 8, 9]})
        print_df_summary_stats(df, column_indices=column_indices)
        out, _ = capfd.readouterr()
        assert "Column 0 ( A ):" in out
        assert "Column 1 ( B ):" in out
        assert "Column 2 ( C ):" not in out

    def test_non_numeric_columns(self, capfd):
        df = pd.DataFrame({"A": [1, 2, 3], "B": ["a", "b", "c"]})
        print_df_summary_stats(df)
        out, _ = capfd.readouterr()
        assert "Column 0 ( A ):" in out
        assert "Column 1 ( B ):" not in out

    def test_empty_dataframe(self):
        df = pd.DataFrame()
        with pytest.raises(ValueError):
            print_df_summary_stats(df)

    def test_no_numeric_columns(self):
        df = pd.DataFrame({"A": ["a", "b", "c"], "B": ["d", "e", "f"]})
        with pytest.raises(ValueError):
            print_df_summary_stats(df)

    def test_no_numeric_columns_at_specific_column(self):
        df = pd.DataFrame({"A": [1, 2, 3], "B": ["d", "e", "f"]})
        with pytest.raises(ValueError):
            print_df_summary_stats(df, column_indices=1)


class Test_Print_Df_With_Any_Nan_Rows:
    """Test test_print_df_with_any_nan_rows."""

    def test_df_with_nan_and_none_values(self, capfd):
        df = pd.DataFrame(
            {
                "A": [1, 2, None, 4],
                "B": ["a", "b", None, "d"],
                "C": [1, 2, np.nan, 4],
                "D": [1, 2, 3, 4],
            },
        )
        print_df_with_any_nan_rows(df)
        out, _ = capfd.readouterr()
        assert "Column 0 ( A ):\n      [nan]" in out
        assert "Column 1 ( B ):\n      [None]" in out
        assert "Column 2 ( C ):\n      [nan]" in out
        assert "Column 3 ( D ):\n      [3]" in out

    def test_df_without_nan(self, capfd):
        df = pd.DataFrame({"A": [1, 2, 3, 4], "B": ["a", "b", "c", "d"]})
        print_df_with_any_nan_rows(df)
        out, _ = capfd.readouterr()
        # Expecting no output since there are no NaN rows
        assert "The dataframe does not have nan values!" in out

    def test_print_df_with_all_nan_rows(self, capfd):
        df = pd.DataFrame({"A": [None, None, None], "B": [None, None, None]})
        print_df_with_any_nan_rows(df)
        out, _ = capfd.readouterr()
        # Expecting output for all rows since all are NaN
        assert "Column 0 ( A ):\n      [None None None]" in out
        assert "Column 1 ( B ):\n      [None None None]" in out


class TestGetUniqueSortedValues:
    def test_returns_unique_sorted_values(self):
        """Return sorted unique values from a list with duplicates."""
        input_data = ["c", "b", "a", "b", "d"]
        expected = ["a", "b", "c", "d"]
        result = get_unique_sorted_values(input_data)
        assert isinstance(result, list)
        assert result == expected

    def test_handles_nan_in_object_array(self):
        """Convert nan to 'nan' and include in sorted result."""
        arr = np.array(["4610.3512", "4610.3513", np.nan, "4610.3514"], dtype=object)
        result = get_unique_sorted_values(arr)
        expected = ["4610.3512", "4610.3513", "4610.3514", "nan"]
        assert result == expected

    @pytest.mark.parametrize(
        ("input_data", "expected"),
        [
            (["1", "3", "2"], ["1", "2", "3"]),  # already sorted
            ([3.0, 1.0, 2.0, 1.0], [1.0, 2.0, 3.0]),  # mixed numeric types
            (["only"], ["only"]),  # single element
            ([], []),  # empty input
        ],
    )
    def test_various_inputs(self, input_data, expected):
        """Handle varied inputs including numeric, single, and empty."""
        result = get_unique_sorted_values(input_data)
        assert result == expected


def test_print_df_columns_unique_values(capfd):
    df = pd.DataFrame({"A": [1, 2, 2, 3], "B": ["a", "b", "b", "c"]})
    print_df_columns_unique_values(df)
    out, _ = capfd.readouterr()
    assert "Column 0 ( A ):" in out
    assert "[1, 2, 3]" in out
    assert "Column 1 ( B ):" in out
    assert "['a', 'b', 'c']" in out


def test_get_df_columns_unique_values_dict():
    df = pd.DataFrame({"A": [1, 2, 2, 3], "B": ["a", "b", "b", "c"]})
    # Test with column_names
    result = get_df_columns_unique_values_dict(df)
    expected = {"A": [1, 2, 3], "B": ["a", "b", "c"]}
    assert result == expected

    # Test without column_names
    result = get_df_columns_unique_values_dict(df, column_names=False)
    expected = {"Column 0": [1, 2, 3], "Column 1": ["a", "b", "c"]}
    assert result == expected


class Test_Has_Constant_Character:
    """Test _has_constant_characters."""

    def test_constant_nchar(self):
        arr = np.array(["abc", "def", "ghi"])
        assert _has_constant_characters(arr)

    def test_not_constant_nchar(self):
        arr = np.array(["abc", "de", "fghij"])
        assert not _has_constant_characters(arr)

    def test_empty_array(self):
        arr = np.array([], dtype="U")
        assert _has_constant_characters(arr)

        arr = np.array([], dtype="O")
        assert _has_constant_characters(arr)

        arr = np.array([], dtype=int)
        assert _has_constant_characters(arr)

        arr = np.array([])
        assert _has_constant_characters(arr)

    def test_numeric_array(self):
        arr = np.array([123, 456, 789])
        assert _has_constant_characters(arr)

        arr = np.array([1.23, 4.56, 7.89])
        assert _has_constant_characters(arr)


def test_infer_column_names(capfd):
    sensor_name = "PARSIVEL"
    df = pd.DataFrame(
        {
            "0": [123.345, 123.345],  # same number of character
            "1": [12.3456, 1.345],  # not same number characters
        },
    )
    dict_possible_columns = infer_column_names(df=df, sensor_name=sensor_name, row_idx=0)
    assert dict_possible_columns[0] == ["rainfall_amount_absolute_32bit"]
    out, _ = capfd.readouterr()
    assert "WARNING: The number of characters of column 1 values is not constant" in out


def test_check_column_names(capfd):
    sensor_name = "PARSIVEL"

    # Test correct type
    with pytest.raises(TypeError):
        check_column_names(column_names="a string", sensor_name=sensor_name)

    # Test invalid column print a message
    column_names = ["raw_drop_number", "invalid_column", "time"]
    check_column_names(column_names, sensor_name)
    out, _ = capfd.readouterr()
    assert "The following columns do no met the DISDRODB standards: ['invalid_column']" in out
    assert "Please be sure to create the 'time' column" not in out

    # Test m column print a message
    column_names = ["raw_drop_number"]
    check_column_names(column_names, sensor_name)
    out, _ = capfd.readouterr()
    assert "Please be sure to create the 'time' column" in out
