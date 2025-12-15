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
"""Test warnings utility."""
import warnings

from disdrodb.utils.warnings import suppress_warnings


class TestSuppressWarnings:
    def test_suppresses_runtime_and_user_warnings(self):
        """It suppresses RuntimeWarning and UserWarning inside the context."""
        # Collect warnings inside context
        with warnings.catch_warnings(record=True) as caught, suppress_warnings():
            warnings.warn("runtime issue", RuntimeWarning, stacklevel=2)
            warnings.warn("user issue", UserWarning, stacklevel=2)
        assert caught == []  # nothing captured

    def test_does_not_suppress_outside_context(self):
        """It does not suppress warnings outside the context."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.warn("outside", RuntimeWarning, stacklevel=2)
        assert any(isinstance(w.message, RuntimeWarning) for w in caught)

    def test_does_not_suppress_other_warning_types(self):
        """It does not suppress warnings of other types."""
        with warnings.catch_warnings(record=True) as caught, suppress_warnings():
            warnings.warn("dep", DeprecationWarning, stacklevel=2)
        assert any(isinstance(w.message, DeprecationWarning) for w in caught)
