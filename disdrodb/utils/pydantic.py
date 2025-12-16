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
"""Definition of pydantic validation custom class."""
from pydantic import BaseModel, ConfigDict, ValidationError


def format_validation_error(validation_error: Exception) -> str:
    """Format a Pydantic ValidationError for better readability."""
    if not isinstance(validation_error, ValidationError):
        return str(validation_error)

    def _shorten(value, max_len=200):
        """Safely truncate long inputs."""
        text = repr(value)
        if len(text) > max_len:
            return text[: max_len - 5] + " ...]"
        return text

    model_name_attr = getattr(validation_error, "title", None)
    model_name = model_name_attr() if callable(model_name_attr) else model_name_attr or "UnknownModel"

    formatted_errors = [f"Validation errors in {model_name}:"]

    for err in validation_error.errors():
        path = ".".join(str(loc) for loc in err["loc"]) or "<model root>"
        msg = err["msg"]
        err_type = err["type"]

        # Handles both "Value error, ..." and "Value error: ..."
        if msg.lower().startswith("value error"):
            msg = msg.split(",", 1)[-1] if "," in msg else msg.split(":", 1)[-1]
            msg = msg.strip()

        # Model-level (root) errors (raise in after or before)
        if path == "<model root>":
            formatted = f"  • {msg}"
        elif err_type == "missing":
            formatted = f"  • Missing field '{path}': {msg}"
        elif "input" in err:
            formatted = f"  • Field '{path}': {msg} (got: {_shorten(err['input'])})"
        else:
            formatted = f"  • Field '{path}': {msg}"

        formatted_errors.append(formatted)

    return "\n".join(formatted_errors)


class CustomBaseModel(BaseModel):
    """Custom pydantic BaseModel.

    Forbid extra keys.
    Hide URLs in error message.
    Simplify error message.
    """

    model_config = ConfigDict(extra="forbid", hide_error_urls=True)

    # Override the standard ValidationError print behavior
    def __init__(self, **data):
        try:
            super().__init__(**data)
        except ValidationError as e:
            formatted = format_validation_error(e)
            # Raise a new simplified exception
            raise ValueError(formatted) from None
