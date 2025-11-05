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
"""Test pydantic custom validation model."""
import pytest
from pydantic import BaseModel, ValidationError, Field, model_validator
from disdrodb.utils.pydantic import CustomBaseModel, format_validation_error


class NestedModel(BaseModel):
    value: int


class ExampleModel(BaseModel):
    name: str
    age: int = Field(..., ge=0)
    nested: NestedModel


class CustomExampleModel(CustomBaseModel):
    name: str
    age: int = Field(..., ge=0)
         

class RootModel(CustomBaseModel):
    """Model that raises a root-level error."""
    value: int

    @model_validator(mode="before")
    def check_values(cls, values):
        # Always trigger a model-level validation error for testing
        raise ValueError("Invalid input value")



class TestFormatValidationError:
    """Tests for format_validation_error utility."""

    def test_regular_field_error(self):
        """Should format a simple type error correctly."""
        with pytest.raises(ValidationError) as exc_info:
            ExampleModel(name=123, age=10, nested={"value": 5})  # name should be str
        formatted = format_validation_error(exc_info.value)
        assert "Field 'name'" in formatted
        assert (
            "Input should be a valid string" in formatted
            or "str type expected" in formatted
        )
        assert "got" in formatted

    def test_missing_field_error(self):
        """Should format missing field error clearly."""
        with pytest.raises(ValueError) as exc_info:
            ExampleModel(age=5, nested={"value": 1})  # missing 'name'
        formatted = format_validation_error(exc_info.value)

        assert "Missing field 'name'" in formatted
        assert "Field required" in formatted

    def test_nested_field_error(self):
        """Should correctly show nested field paths."""
        with pytest.raises(ValidationError) as exc_info:
            ExampleModel(name="ok", age=5, nested={"value": "bad"})  # wrong type in nested
        formatted = format_validation_error(exc_info.value)

        assert "nested.value" in formatted
        assert "got: 'bad'" in formatted

    def test_long_input_truncation(self):
        """Should truncate overly long input values."""
        long_str = "x" * 1000
        with pytest.raises(ValidationError) as exc_info:
            ExampleModel(name="ok", age=long_str, nested={"value": 1})
        formatted = format_validation_error(exc_info.value)

        assert "..." in formatted  # truncated representation
        assert "age" in formatted

    def test_non_validation_error_input(self):
        """Should just convert non-ValidationError input to string."""
        msg = format_validation_error(ValueError("something went wrong"))
        assert msg == "something went wrong"
    
    def test_integration_in_custom_base_model(self):
        """Test CustomBaseModel."""
        with pytest.raises(ValueError) as exc_info:
            CustomExampleModel(name=123, age="bad")  # wrong type in nested
        assert "• Field 'name': Input should be a valid string" in str(exc_info.value)
        assert "• Field 'age': Input should be a valid integer" in str(exc_info.value)

    def test_model_root_error_correctly_formatted(self):
        """Should correctly format a model-level (root) error with 'Value error,' prefix."""
        with pytest.raises(ValueError) as exc_info:
            RootModel(value=-5)  # triggers ValueError in init
        msg = str(exc_info.value)
        assert "Field '<model root>':" not in msg
