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
"""Test DISDRODB dictionaries utilities."""
import pytest

from disdrodb.utils.dict import extract_dictionary, extract_product_kwargs


class TestExtractDictionary:
    def test_removes_keys(self):
        """It should extract and remove specified keys from the dictionary."""
        d = {"a": 1, "b": 2, "c": 3}
        result = extract_dictionary(d, keys=["a", "c", "z"])  # "z" not present
        assert result == {"a": 1, "c": 3}
        assert d == {"b": 2}  # keys removed from original dict

    def test_no_matching_keys(self):
        """It should return an empty dict if no keys match."""
        d = {"a": 1, "b": 2}
        result = extract_dictionary(d, keys=["x", "y"])
        assert result == {}
        assert d == {"a": 1, "b": 2}  # unchanged


class TestExtractProductKwargs:
    def test_l0c(self, monkeypatch):
        """It should return an empty dict for product L0C (no kwargs expected)."""
        kwargs = {"foo": 1, "bar": 2}
        result = extract_product_kwargs(kwargs, "L0C")
        assert result == {}
        assert kwargs == {"foo": 1, "bar": 2}

    def test_l1(self, monkeypatch):
        """It should extract only temporal_resolution for product L1."""
        kwargs = {"temporal_resolution": "1MIN", "extra": 42}
        result = extract_product_kwargs(kwargs, "L1")
        assert result == {"temporal_resolution": "1MIN"}
        assert kwargs == {"extra": 42}

    def test_l2e(self, monkeypatch):
        """It should extract only temporal_resolution for product L2E."""
        kwargs = {"temporal_resolution": "5MIN", "foo": "bar"}
        result = extract_product_kwargs(kwargs, "L2E")
        assert result == {"temporal_resolution": "5MIN"}
        assert "temporal_resolution" not in kwargs

    def test_l2m(self, monkeypatch):
        """It should extract temporal_resolution and model_name for product L2M."""
        kwargs = {"temporal_resolution": "10MIN", "model_name": "X", "unused": 1}
        result = extract_product_kwargs(kwargs, "L2M")
        assert result == {"temporal_resolution": "10MIN", "model_name": "X"}
        assert kwargs == {"unused": 1}
        assert "temporal_resolution" not in kwargs
        assert "model_name" not in kwargs

    def test_invalid_product(self, monkeypatch):
        """It should raise ValueError if product is invalid."""
        # Force check_product to raise
        with pytest.raises(ValueError):
            extract_product_kwargs({}, "INVALID_PRODUCT")
