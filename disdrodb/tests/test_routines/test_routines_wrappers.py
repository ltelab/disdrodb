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
"""Test DISDRODB routines wrappers utilities."""

import pytest

from disdrodb.routines.wrappers import validate_processing_flags_and_get_required_product


class TestValidateAndGetRequiredProduct:
    """Test suite for validate_processing_flags_and_get_required_product function."""

    def test_no_processing_levels_enabled_raises_error(self):
        """Test that error is raised when no processing levels are enabled."""
        with pytest.raises(ValueError, match="At least one processing level must be enabled"):
            validate_processing_flags_and_get_required_product(
                l0a_processing=False,
                l0b_processing=False,
                l0c_processing=False,
                l1_processing=False,
                l2e_processing=False,
                l2m_processing=False,
            )

    def test_no_processing_levels_enabled_with_empty_dict_raises_error(self):
        """Test that error is raised when empty dict is passed."""
        with pytest.raises(ValueError, match="At least one processing level must be enabled"):
            validate_processing_flags_and_get_required_product()

    def test_single_level_processing_l0a(self):
        """Test validation with only L0A processing enabled."""
        result = validate_processing_flags_and_get_required_product(l0a_processing=True)
        assert result == "RAW"

    def test_single_level_processing_l0b(self):
        """Test validation with only L0B processing enabled."""
        result = validate_processing_flags_and_get_required_product(l0b_processing=True)
        assert result == "L0A"

    def test_single_level_processing_l1(self):
        """Test validation with only L1 processing enabled."""
        result = validate_processing_flags_and_get_required_product(l1_processing=True)
        assert result == "L0C"

    def test_single_level_processing_l2e(self):
        """Test validation with only L2E processing enabled."""
        result = validate_processing_flags_and_get_required_product(l2e_processing=True)
        assert result == "L1"

    def test_single_level_processing_l2m(self):
        """Test validation with only L2M processing enabled."""
        result = validate_processing_flags_and_get_required_product(l2m_processing=True)
        assert result == "L2E"

    def test_contiguous_processing_chain_l0a_to_l0c(self):
        """Test valid contiguous processing chain from L0A to L0C."""
        result = validate_processing_flags_and_get_required_product(
            l0a_processing=True,
            l0b_processing=True,
            l0c_processing=True,
        )
        assert result == "RAW"

    def test_contiguous_processing_chain_l0b_to_l1(self):
        """Test valid contiguous processing chain from L0B to L1."""
        result = validate_processing_flags_and_get_required_product(
            l0a_processing=False,
            l0b_processing=True,
            l0c_processing=True,
            l1_processing=True,
        )
        assert result == "L0A"

    def test_contiguous_processing_chain_full(self):
        """Test valid full contiguous processing chain."""
        result = validate_processing_flags_and_get_required_product(
            l0a_processing=True,
            l0b_processing=True,
            l0c_processing=True,
            l1_processing=True,
            l2e_processing=True,
            l2m_processing=True,
        )
        assert result == "RAW"

    def test_gap_in_processing_chain_l0a_l0c_raises_error(self):
        """Test that gap between L0A and L0C (missing L0B) raises error."""
        with pytest.raises(ValueError, match="Processing chain has gaps"):
            validate_processing_flags_and_get_required_product(
                l0a_processing=True,
                l0b_processing=False,
                l0c_processing=True,
            )

    def test_gap_in_processing_chain_l0a_l1_raises_error(self):
        """Test that gap between L0A and L1 (missing L0B, L0C) raises error."""
        with pytest.raises(ValueError, match="Processing chain has gaps"):
            validate_processing_flags_and_get_required_product(
                l0a_processing=True,
                l0b_processing=False,
                l0c_processing=False,
                l1_processing=True,
            )

    def test_gap_in_processing_chain_l1_l2m_raises_error(self):
        """Test that gap between L1 and L2M (missing L2E) raises error."""
        with pytest.raises(ValueError, match="Processing chain has gaps"):
            validate_processing_flags_and_get_required_product(
                l1_processing=True,
                l2e_processing=False,
                l2m_processing=True,
            )

    def test_gap_in_middle_of_chain_raises_error(self):
        """Test that gap in middle of processing chain raises error."""
        with pytest.raises(ValueError, match="Processing chain has gaps"):
            validate_processing_flags_and_get_required_product(
                l0a_processing=True,
                l0b_processing=True,
                l0c_processing=False,
                l1_processing=True,
                l2e_processing=True,
            )

    def test_l2e_and_l2m_both_enabled_valid(self):
        """Test that both L2E and L2M enabled is valid."""
        result = validate_processing_flags_and_get_required_product(
            l0a_processing=False,
            l0b_processing=False,
            l0c_processing=False,
            l1_processing=True,
            l2e_processing=True,
            l2m_processing=True,
        )
        assert result == "L0C"

    def test_skip_early_levels_valid(self):
        """Test that skipping early levels (starting from L0B) is valid."""
        result = validate_processing_flags_and_get_required_product(
            l0a_processing=False,
            l0b_processing=False,
            l0c_processing=True,
            l1_processing=True,
        )
        assert result == "L0B"

    def test_skip_last_levels_valid(self):
        """Test that skipping last levels (ending at L0C) is valid."""
        result = validate_processing_flags_and_get_required_product(
            l0a_processing=True,
            l0b_processing=True,
            l0c_processing=True,
            l1_processing=False,
            l2e_processing=False,
            l2m_processing=False,
        )
        assert result == "RAW"
