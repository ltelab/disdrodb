from disdrodb.utils.scripts import parse_arg_to_list


def test_parse_arg_to_list_empty_string():
    """Test parse_arg_to_list() with an empty string."""
    args = ""
    expected_output = None
    assert parse_arg_to_list(args) == expected_output


def test_parse_arg_to_list_single_variable():
    """Test parse_arg_to_list() with a single variable."""
    args = "variable"
    expected_output = ["variable"]
    assert parse_arg_to_list(args) == expected_output


def test_parse_arg_to_list_multiple_variables():
    """Test parse_arg_to_list() with multiple variables."""
    args = "variable1 variable2"
    expected_output = ["variable1", "variable2"]
    assert parse_arg_to_list(args) == expected_output


def test_parse_arg_to_list_extra_spaces():
    """Test parse_arg_to_list() with extra spaces between variables."""
    args = "  variable1    variable2  "
    expected_output = ["variable1", "variable2"]
    assert parse_arg_to_list(args) == expected_output


def test_parse_arg_to_list_none():
    """Test parse_arg_to_list() with None input."""
    args = None
    expected_output = None
    assert parse_arg_to_list(args) == expected_output


def test_parse_arg_to_list_other_types():
    """Test parse_arg_to_list() with other types of input."""
    args = 123
    expected_output = 123
    assert parse_arg_to_list(args) == expected_output


def test_parse_arg_to_list_empty_list():
    """Test parse_arg_to_list() with an empty list."""
    args = []
    expected_output = []
    assert parse_arg_to_list(args) == expected_output
