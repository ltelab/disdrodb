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
"""Test 2D LUT routines."""

import numpy as np
import pandas as pd
import pytest

from disdrodb.retrievals.lut import NearestNeighbourLUT2D


class TestNearestNeighbourLUT2D:
    """Test suite for NearestNeighbourLUT2D class."""

    @pytest.fixture
    def sample_df(self):
        """Create a sample DataFrame for testing."""
        return pd.DataFrame(
            {
                "x": [0.0, 1.0, 2.0, 0.0, 1.0, 2.0],
                "y": [0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
                "value1": [10, 20, 30, 40, 50, 60],
                "value2": [100, 200, 300, 400, 500, 600],
            },
        )

    @pytest.fixture
    def lut(self, sample_df):
        """Create a NearestNeighbourLUT2D instance for testing."""
        return NearestNeighbourLUT2D(sample_df, "x", "y", columns=["value1", "value2"])

    def test_init_with_valid_dataframe(self, sample_df):
        """Test initialization with valid DataFrame and parameters."""
        lut = NearestNeighbourLUT2D(sample_df, "x", "y", columns=["value1", "value2"])
        assert lut.x == "x"
        assert lut.y == "y"
        assert lut.columns == ["value1", "value2"]
        assert lut.dtype == np.float32
        assert lut.points.shape == (6, 2)
        assert lut.values.shape == (6, 2)  # noqa: PD011

    def test_init_with_default_columns(self, sample_df):
        """Test initialization with default columns (all columns)."""
        lut = NearestNeighbourLUT2D(sample_df, "x", "y")
        assert set(lut.columns) == {"x", "y", "value1", "value2"}

    def test_init_with_custom_dtype(self, sample_df):
        """Test initialization with custom dtype."""
        lut = NearestNeighbourLUT2D(sample_df, "x", "y", columns=["value1"], dtype=np.float64)
        assert lut.dtype == np.float64
        assert lut.points.dtype == np.float64
        assert lut.values.dtype == np.float64  # noqa: PD011

    def test_init_with_index_reset(self):
        """Test initialization when x is in DataFrame index."""
        df = pd.DataFrame(
            {
                "y": [0.0, 1.0, 2.0],
                "value": [10, 20, 30],
            },
        )
        df = df.set_index(pd.Index([0.0, 1.0, 2.0], name="x"))
        lut = NearestNeighbourLUT2D(df, "x", "y", columns=["value"])
        assert lut.x == "x"
        assert lut.points.shape == (3, 2)

    def test_init_raises_error_for_missing_x_column(self, sample_df):
        """Test that ValueError is raised when x column is missing."""
        with pytest.raises(ValueError, match="x='nonexistent' is not a column of df."):
            NearestNeighbourLUT2D(sample_df, "nonexistent", "y")

    def test_init_raises_error_for_missing_y_column(self, sample_df):
        """Test that ValueError is raised when y column is missing."""
        with pytest.raises(ValueError, match="y='nonexistent' is not a column of df."):
            NearestNeighbourLUT2D(sample_df, "x", "nonexistent")

    def test_init_raises_error_for_missing_value_columns(self, sample_df):
        """Test that ValueError is raised when specified columns are missing."""
        with pytest.raises(ValueError, match="columns not found in DataFrame"):
            NearestNeighbourLUT2D(sample_df, "x", "y", columns=["nonexistent"])

    def test_predict_with_scalar_inputs(self, lut):
        """Test predict method with scalar x and y coordinates."""
        result = lut.predict(0.5, 0.5)
        assert isinstance(result, pd.DataFrame)
        assert result.shape == (1, 4)  # x, y, value1, value2
        assert "x" in result.columns
        assert "y" in result.columns
        assert "value1" in result.columns
        assert "value2" in result.columns

    def test_predict_with_array_inputs(self, lut):
        """Test predict method with array inputs."""
        x_query = [0.5, 1.5]
        y_query = [0.5, 0.5]
        result = lut.predict(x_query, y_query)
        assert result.shape == (2, 4)
        assert np.array_equal(result["x"].values, x_query)
        assert np.array_equal(result["y"].values, y_query)

    def test_predict_with_return_distance(self, lut):
        """Test predict method with return_distance=True."""
        result = lut.predict(0.0, 0.0, return_distance=True)
        assert "distance" in result.columns
        assert result["distance"].iloc[0] == 0.0  # Exact match

    def test_predict_exact_match(self, lut):
        """Test predict returns exact values when query point matches training point."""
        result = lut.predict(1.0, 1.0)
        assert result["value1"].iloc[0] == 50
        assert result["value2"].iloc[0] == 500

    def test_predict_nearest_neighbor(self, lut):
        """Test predict returns nearest neighbor values for intermediate points."""
        result = lut.predict(0.4, 0.4)
        # Should be closest to (0, 0) or (0, 1) or (1, 0)
        assert result["value1"].iloc[0] in [10, 20, 40]

    def test_predict_raises_error_for_mismatched_shapes(self, lut):
        """Test that ValueError is raised when x and y have different shapes."""
        with pytest.raises(ValueError, match="x and y must have the same shape"):
            lut.predict([0.5, 1.5], [0.5])

    def test_predict_column_order(self, lut):
        """Test that predict returns columns in correct order."""
        result = lut.predict(1.0, 1.0)
        expected_columns = ["x", "y", "value1", "value2"]
        assert list(result.columns) == expected_columns

    def test_predict_dict_with_scalar_inputs(self, lut):
        """Test predict_dict method with scalar coordinates."""
        result = lut.predict_dict(1.0, 1.0)
        assert isinstance(result, dict)
        assert result["x"] == 1.0
        assert result["y"] == 1.0
        assert result["value1"] == 50
        assert result["value2"] == 500

    def test_predict_dict_raises_error_for_array_inputs(self, lut):
        """Test that predict_dict raises ValueError for non-scalar inputs."""
        with pytest.raises(ValueError, match="predict_dict accepts only scalars"):
            lut.predict_dict([0.5, 1.5], [0.5, 1.5])

    def test_len(self, lut):
        """Test __len__ method returns correct number of points."""
        assert len(lut) == 6

    def test_save_and_read_lut(self, lut, tmp_path):
        """Test saving and loading lookup table from file."""
        filepath = tmp_path / "test_lut.pkl"
        lut.save_lut(filepath)
        assert filepath.exists()

        loaded_lut = NearestNeighbourLUT2D.read_lut(filepath)
        assert loaded_lut.x == lut.x
        assert loaded_lut.y == lut.y
        assert loaded_lut.columns == lut.columns
        assert np.array_equal(loaded_lut.points, lut.points)
        assert np.array_equal(loaded_lut.values, lut.values)

    def test_save_lut_preserves_functionality(self, lut, tmp_path):
        """Test that loaded LUT produces same predictions as original."""
        filepath = tmp_path / "test_lut.pkl"
        lut.save_lut(filepath)
        loaded_lut = NearestNeighbourLUT2D.read_lut(filepath)

        original_result = lut.predict(0.5, 0.5)
        loaded_result = loaded_lut.predict(0.5, 0.5)

        pd.testing.assert_frame_equal(original_result, loaded_result)

    def test_tree_attribute_exists(self, lut):
        """Test that k-d tree is properly initialized."""
        assert hasattr(lut, "tree")
        assert lut.tree is not None

    def test_multiple_predictions_consistency(self, lut):
        """Test that multiple predictions for same point return consistent results."""
        result1 = lut.predict(0.7, 0.3)
        result2 = lut.predict(0.7, 0.3)
        pd.testing.assert_frame_equal(result1, result2)

    def test_empty_columns_list_uses_all_columns(self):
        """Test initialization with None columns parameter includes all columns."""
        df = pd.DataFrame(
            {
                "x": [0.0, 1.0],
                "y": [0.0, 1.0],
                "val": [10, 20],
            },
        )
        lut = NearestNeighbourLUT2D(df, "x", "y", columns=None)
        assert "val" in lut.columns
        assert "x" in lut.columns
        assert "y" in lut.columns

    def test_predict_with_nan_input_x(self, lut):
        """Test that NaN x-coordinate returns NaN values."""
        result = lut.predict(np.nan, 0.5)
        assert np.isnan(result["value1"].iloc[0])
        assert np.isnan(result["value2"].iloc[0])
        assert np.isnan(result["x"].iloc[0])

    def test_predict_with_nan_input_y(self, lut):
        """Test that NaN y-coordinate returns NaN values."""
        result = lut.predict(0.5, np.nan)
        assert np.isnan(result["value1"].iloc[0])
        assert np.isnan(result["value2"].iloc[0])
        assert np.isnan(result["y"].iloc[0])

    def test_predict_with_inf_input_x(self, lut):
        """Test that inf x-coordinate returns NaN values."""
        result = lut.predict(np.inf, 0.5)
        assert np.isnan(result["value1"].iloc[0])
        assert np.isnan(result["value2"].iloc[0])

    def test_predict_with_inf_input_y(self, lut):
        """Test that inf y-coordinate returns NaN values."""
        result = lut.predict(0.5, np.inf)
        assert np.isnan(result["value1"].iloc[0])
        assert np.isnan(result["value2"].iloc[0])

    def test_predict_with_mixed_valid_invalid_inputs(self, lut):
        """Test that mixed valid/invalid inputs return correct values."""
        x_query = [0.5, np.nan, 1.5]
        y_query = [0.5, 0.5, np.inf]
        result = lut.predict(x_query, y_query)

        # First point should have valid values
        assert not np.isnan(result["value1"].iloc[0])
        # Second point should be NaN
        assert np.isnan(result["value1"].iloc[1])
        # Third point should be NaN
        assert np.isnan(result["value1"].iloc[2])

    def test_predict_with_nan_return_distance(self, lut):
        """Test that NaN inputs return NaN distance when return_distance=True."""
        result = lut.predict(np.nan, 0.5, return_distance=True)
        assert "distance" in result.columns
        assert np.isnan(result["distance"].iloc[0])

    def test_predict_with_max_distance_scalar(self, lut):
        """Test max_distance parameter with scalar value masks distant points."""
        # Query point at (3.0, 3.0) which is far from all training points
        result = lut.predict(3.0, 3.0, max_distance=1.0)
        # Should return NaN because nearest point is > 1.0 away
        assert np.isnan(result["value1"].iloc[0])
        assert np.isnan(result["value2"].iloc[0])

    def test_predict_with_max_distance_scalar_valid(self, lut):
        """Test max_distance parameter allows close points through."""
        # Query point at (0.3, 0.3) which is close to (0, 0)
        result = lut.predict(0.3, 0.3, max_distance=1.0)
        # Should return valid values because nearest point is < 1.0 away
        assert not np.isnan(result["value1"].iloc[0])
        assert not np.isnan(result["value2"].iloc[0])

    def test_predict_with_max_distance_tuple(self, lut):
        """Test max_distance parameter with tuple for univariate thresholds."""
        # Query point at (0.6, 0.2) - close in y but far in x from (0, 0)
        result = lut.predict(0.6, 0.2, max_distance=(0.3, 0.3))
        # Should return NaN because x distance > 0.3
        assert np.isnan(result["value1"].iloc[0])

    def test_predict_with_max_distance_tuple_valid(self, lut):
        """Test max_distance tuple allows points within both thresholds."""
        # Query point at (0.2, 0.2) - close to (0, 0)
        result = lut.predict(0.2, 0.2, max_distance=(0.3, 0.3))
        # Should return valid values
        assert not np.isnan(result["value1"].iloc[0])
        assert result["value1"].iloc[0] == 10  # Nearest to (0, 0)

    def test_predict_with_max_distance_none(self, lut):
        """Test that max_distance=None applies no masking."""
        # Query point far from all training points
        result = lut.predict(100.0, 100.0, max_distance=None)
        # Should return valid values (nearest neighbor values)
        assert not np.isnan(result["value1"].iloc[0])

    def test_predict_with_max_distance_and_return_distance(self, lut):
        """Test max_distance with return_distance=True returns correct distance."""
        result = lut.predict(0.3, 0.3, max_distance=1.0, return_distance=True)
        assert "distance" in result.columns
        assert not np.isnan(result["distance"].iloc[0])
        assert result["distance"].iloc[0] < 1.0

    def test_predict_max_distance_tuple_wrong_length(self, lut):
        """Test that max_distance tuple with wrong length raises ValueError."""
        with pytest.raises(ValueError, match="max_distance tuple must have exactly 2 elements"):
            lut.predict(0.5, 0.5, max_distance=(0.3, 0.3, 0.3))

    def test_predict_with_max_distance_array_of_points(self, lut):
        """Test max_distance with multiple query points masks correctly."""
        x_query = [0.1, 5.0, 0.2]
        y_query = [0.1, 5.0, 0.2]
        result = lut.predict(x_query, y_query, max_distance=1.0)

        # First and third points should be valid (close to origin)
        assert not np.isnan(result["value1"].iloc[0])
        assert not np.isnan(result["value1"].iloc[2])
        # Second point should be NaN (far from all points)
        assert np.isnan(result["value1"].iloc[1])

    def test_predict_max_distance_tuple_asymmetric(self, lut):
        """Test max_distance with asymmetric x and y thresholds."""
        # Point at (0.5, 0.2) - test asymmetric thresholds
        result = lut.predict(0.5, 0.2, max_distance=(0.6, 0.3))
        # Should be valid (within both thresholds)
        assert not np.isnan(result["value1"].iloc[0])

        result2 = lut.predict(0.5, 0.2, max_distance=(0.3, 0.6))
        # Should be NaN (x exceeds threshold)
        assert np.isnan(result2["value1"].iloc[0])

    def test_predict_coordinates_preserved_with_nan_values(self, lut):
        """Test that x and y coordinates are preserved even when values are NaN."""
        result = lut.predict(np.nan, 0.5)
        assert np.isnan(result["x"].iloc[0])
        assert result["y"].iloc[0] == 0.5
