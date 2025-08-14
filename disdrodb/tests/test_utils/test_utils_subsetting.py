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
"""Test dataset subsetting utilities."""
import numpy as np
import pandas as pd
import pytest

from disdrodb.tests.fake_datasets import create_template_dataset


class TestSel:
    """Test disdrodb.sel method."""

    def test_subsetting_diameter(self):
        """Test disdrodb.sel with a diameter coordinate."""
        # Create a disdrodb dataset
        ds = create_template_dataset()

        # Test selecting by coordinate values
        assert not np.all(ds["diameter_bin_lower"].data <= 0.5)
        ds_subset = ds.disdrodb.sel(diameter_bin_lower=slice(0, 0.5))
        assert np.all(ds_subset["diameter_bin_lower"].data <= 0.5)

    def test_subsetting_time(self):
        """Test disdrodb.sel with time coordinate."""
        # Create dataset with time dimension using pandas datetime
        timesteps = pd.to_datetime(
            [
                "2000-01-01T00:00:00",
                "2000-01-01T00:00:01",
                "2000-01-01T00:00:02",
                "2000-01-01T00:00:03",
                "2000-01-01T00:00:04",
                "2000-01-01T00:00:05",
                "2000-01-01T00:00:06",
                "2000-01-01T00:00:07",
            ],
        )

        # Create a disdrodb dataset
        ds = create_template_dataset()

        # Expand to have much more timesteps
        # - Reindex to new time coordinate, filling missing with NaN
        ds = ds.reindex(time=timesteps)

        # Test subsetting with time unordered (as string)
        time_subset = ["2000-01-01T00:00:06", "2000-01-01T00:00:04"]
        ds_subset = ds.disdrodb.sel(time=time_subset)
        expected_times = np.array(time_subset, dtype="M8[ns]")
        np.testing.assert_array_equal(ds_subset["time"].data, expected_times)

        # Test subsetting with time as np.dstetime64
        ds_subset = ds.disdrodb.sel(time=np.array(time_subset, dtype="M8[ns]"))
        np.testing.assert_array_equal(ds_subset["time"].data, expected_times)

        # Test subsetting by time dataArray
        ds_sel = ds.isel(time=slice(0, 2))["time"]
        ds_subset = ds.disdrodb.sel(time=ds_sel)
        expected_subset_times = ds.time.isel(time=slice(0, 2)).data
        np.testing.assert_array_equal(ds_subset["time"].data, expected_subset_times)

        # Test subsetting by time slice (stop is inclusive !)
        ds_subset = ds.disdrodb.sel(time=slice("2000-01-01T00:00:04", "2000-01-01T00:00:06"))
        expected_slice_times = ds.sel(time=slice("2000-01-01T00:00:04", "2000-01-01T00:00:06"))["time"].data
        np.testing.assert_array_equal(ds_subset["time"].data, expected_slice_times)

    def test_raise_error_with_inexisting_coordinate(self):
        """Test disdrodb.sel with an inexisting coordinate."""
        # Create a disdrodb dataset
        ds = create_template_dataset()

        # Test raise error
        with pytest.raises(ValueError):
            ds.disdrodb.sel(inexisting=slice(0, 2))

    def test_raise_error_with_dimension_without_coordinate(self):
        """Test disdrodb.sel with a dimension without coordinate."""
        # Create a disdrodb dataset
        ds = create_template_dataset()
        # Remove time coordinates
        ds = ds.drop_vars("time")

        # Test raise error
        with pytest.raises(ValueError, match="Can not subset with disdrodb.sel the dimension 'time' if it is not"):
            ds.disdrodb.sel(time=slice(0, 2))


class TestIsel:
    """Test disdrodb.isel method."""

    def test_with_1d_coordinate(self):
        """Test disdrodb.isel with a classical 1D coordinate."""
        # Create a disdrodb dataset
        ds = create_template_dataset()

        # Test isel allow subsetting a 1D coordinate
        # Dict
        ds_subset = ds.disdrodb.isel({"diameter_bin_lower": slice(0, 2)})
        np.testing.assert_allclose(ds_subset["diameter_bin_lower"], [0.1, 0.3])
        # Slice
        ds_subset = ds.disdrodb.isel(diameter_bin_lower=slice(0, 2))
        np.testing.assert_allclose(ds_subset["diameter_bin_lower"], [0.1, 0.3])
        # List
        ds_subset = ds.disdrodb.isel(diameter_bin_lower=[3, 2, -1])
        np.testing.assert_allclose(ds_subset["diameter_bin_lower"], [0.7, 0.5, 0.7])
        # Value
        ds_subset = ds.disdrodb.isel(diameter_bin_lower=-1)
        np.testing.assert_allclose(ds_subset["diameter_bin_lower"], [0.7])
        # Out of index
        with pytest.raises(IndexError):
            ds.disdrodb.isel(diameter_bin_lower=10)

    def test_with_2d_coordinate(self):
        """Test disdrodb.isel with a 2D coordinate raise an error."""
        # Create a disdrodb dataset
        ds = create_template_dataset()
        # Add 2D dimensional coordinate
        # - Add extra dim
        ds = ds.expand_dims({"another_dim": np.arange(3)})
        ds = ds.assign_coords(
            coord_2d=(("time", "another_dim"), np.full((ds.sizes["time"], ds.sizes["another_dim"]), np.nan)),
        )
        with pytest.raises(ValueError, match="'coord_2d' is not a dimension or a 1D non-dimensional coordinate."):
            ds.disdrodb.isel(coord_2d=slice(0, 2))

    def test_with_dimension(self):
        """Test disdrodb.isel with a dimension."""
        # Create a disdrodb dataset
        ds = create_template_dataset()

        # Test subset the dimension
        ds_subset = ds.disdrodb.isel(diameter_bin_center=slice(0, 2))
        assert ds_subset.sizes["diameter_bin_center"] == 2
        np.testing.assert_allclose(ds_subset["diameter_bin_center"].to_numpy(), np.array([0.2, 0.4]))


class TestAlign:
    """Test disdrodb.align function."""

    def test_basic_alignment_identical_datasets(self):
        """Test alignment of identical datasets."""
        # Create disdrodb datasets
        ds1 = create_template_dataset()
        ds2 = create_template_dataset()
        ds3 = create_template_dataset()

        # Align the datasets
        aligned_datasets = ds1.disdrodb.align(ds2, ds3)

        # Check that we get the same number of datasets back
        assert len(aligned_datasets) == 3

        # Check that all datasets have the same coordinates
        for ds in aligned_datasets:
            np.testing.assert_array_equal(ds.time.data, ds1.time.data)
            np.testing.assert_array_equal(ds.diameter_bin_center.data, ds1.diameter_bin_center.data)
            np.testing.assert_array_equal(ds.velocity_bin_center.data, ds1.velocity_bin_center.data)

    def test_alignment_with_different_time_coordinates(self):
        """Test alignment with datasets having different time coordinates."""
        # Create disdrodb datasets
        ds1 = create_template_dataset()
        ds2 = create_template_dataset()
        ds3 = create_template_dataset()

        # Modify time coordinates to have partial overlap
        timesteps = pd.to_datetime(
            [
                "2000-01-01T00:00:00",
                "2000-01-01T00:00:01",
                "2000-01-01T00:00:02",
                "2000-01-01T00:00:03",
                "2000-01-01T00:00:04",
            ],
        )
        ds1 = ds1.reindex(time=timesteps[[0, 1, 2]])
        ds2 = ds2.reindex(time=timesteps[[1, 2, 3]])
        ds3 = ds3.reindex(time=timesteps[[0, 2, 4]])

        # Align the datasets
        aligned_datasets = ds1.disdrodb.align(ds2, ds3)

        # Check that all datasets have the common time coordinate (only one is common)
        expected_time = np.array(timesteps[2])
        for ds in aligned_datasets:
            np.testing.assert_array_equal(ds.time.data.astype("M8[s]"), expected_time.astype("M8[s]"))

    def test_alignment_with_different_diameter_coordinates(self):
        """Test alignment with datasets having different diameter coordinates."""
        ds1 = create_template_dataset()
        ds2 = create_template_dataset()

        # Modify diameter coordinates to have partial overlap
        # ds1 has [0.2, 0.4, 0.6, 0.8]
        # ds2 will have [0.4, 0.6, 0.8, 1.0]
        new_diameter = np.array([0.4, 0.6, 0.8, 1.0])
        ds2 = ds2.assign_coords(diameter_bin_center=new_diameter)
        ds2 = ds2.reindex(diameter_bin_center=new_diameter, fill_value=np.nan)

        # Align the datasets
        aligned_datasets = ds1.disdrodb.align(ds2)

        # Check that all datasets have the common diameter coordinates [0.4, 0.6, 0.8]
        expected_diameter = np.array([0.4, 0.6, 0.8])
        for ds in aligned_datasets:
            np.testing.assert_array_equal(ds.diameter_bin_center.data, expected_diameter)

    def test_alignment_with_different_velocity_coordinates(self):
        """Test alignment with datasets having different velocity coordinates."""
        ds1 = create_template_dataset()
        ds2 = create_template_dataset()

        # Modify velocity coordinates to have partial overlap
        # ds1 has [0.2, 0.5, 1.0]
        # ds2 will have [0.5, 1.0, 1.5]
        new_velocity = np.array([0.5, 1.0, 1.5])
        ds2 = ds2.assign_coords(velocity_bin_center=new_velocity)
        ds2 = ds2.reindex(velocity_bin_center=new_velocity, fill_value=np.nan)

        # Align the datasets
        aligned_datasets = ds1.disdrodb.align(ds2)

        # Check that all datasets have the common velocity coordinates [0.5, 1.0]
        expected_velocity = np.array([0.5, 1.0])
        for ds in aligned_datasets:
            np.testing.assert_array_equal(ds.velocity_bin_center.data, expected_velocity)

    def test_alignment_with_missing_dimensions(self):
        """Test alignment when some datasets don't have all dimensions."""
        # Create datasets with and without velocity dimension
        ds1 = create_template_dataset(with_velocity=True)  # Has velocity
        ds2 = create_template_dataset(with_velocity=False)  # No velocity

        # Align the datasets - should only align on time and diameter
        aligned_datasets = ds1.disdrodb.align(ds2)

        # Check that alignment worked on available dimensions
        assert len(aligned_datasets) == 2

        # Both should have same time and diameter coordinates
        np.testing.assert_array_equal(aligned_datasets[0].time.data, aligned_datasets[1].time.data)
        np.testing.assert_array_equal(
            aligned_datasets[0].diameter_bin_center.data,
            aligned_datasets[1].diameter_bin_center.data,
        )

        # Only ds1 should have velocity dimension
        assert "velocity_bin_center" in aligned_datasets[0].coords
        assert "velocity_bin_center" not in aligned_datasets[1].coords

    def test_alignment_no_common_coordinates_error(self):
        """Test that alignment raises error when no common coordinates exist."""
        ds1 = create_template_dataset()
        ds2 = create_template_dataset()

        # Remove all alignment coordinates from ds2
        ds2 = ds2.drop_vars(["time", "diameter_bin_center", "velocity_bin_center"])

        # Should raise ValueError
        with pytest.raises(ValueError, match="No common coordinates found"):
            ds1.disdrodb.align(ds2)

    def test_alignment_no_common_values_error(self):
        """Test that alignment raises error when coordinates exist but have no common values."""
        ds1 = create_template_dataset()
        ds2 = create_template_dataset()

        # Make time coordinates completely different
        ds1 = ds1.assign_coords(time=np.array([0, 1]))
        ds2 = ds2.assign_coords(time=np.array([2, 3]))
        ds1 = ds1.reindex(time=[0, 1], fill_value=np.nan)
        ds2 = ds2.reindex(time=[2, 3], fill_value=np.nan)

        # Should raise ValueError for no common time values
        with pytest.raises(ValueError, match="No common time values"):
            ds1.disdrodb.align(ds2)

    def test_alignment_single_dataset_error(self):
        """Test that alignment raises error with only one dataset."""
        ds1 = create_template_dataset()

        # Should raise ValueError for insufficient datasets
        with pytest.raises(ValueError, match="At least two xarray object are required"):
            ds1.disdrodb.align()

    def test_alignment_preserves_data_variables(self):
        """Test that alignment preserves data variables correctly."""
        ds1 = create_template_dataset()
        ds2 = create_template_dataset()

        # Modify one dataset's data to make it different
        ds2["fall_velocity"] = ds2["fall_velocity"] * 2

        # Align the datasets
        aligned_datasets = ds1.disdrodb.align(ds2)

        # Check that data variables are preserved and different
        assert "fall_velocity" in aligned_datasets[0].data_vars
        assert "fall_velocity" in aligned_datasets[1].data_vars

        # Values should be different (ds2 was multiplied by 2)
        assert not np.array_equal(
            aligned_datasets[0]["fall_velocity"].data,
            aligned_datasets[1]["fall_velocity"].data,
        )

    def test_alignment_with_multiple_dimensions_subset(self):
        """Test alignment when multiple dimensions need subsetting simultaneously."""
        ds1 = create_template_dataset()
        ds2 = create_template_dataset()
        ds3 = create_template_dataset()

        # Create datasets with different overlapping coordinates
        # Time: ds1=[0,1], ds2=[1,2], ds3=[0,1,2] -> common=[1]

        # Modify time coordinates to have partial overlap
        timesteps = pd.to_datetime(
            [
                "2000-01-01T00:00:00",
                "2000-01-01T00:00:01",
                "2000-01-01T00:00:02",
            ],
        )
        ds1 = ds1.reindex(time=timesteps[[0, 1]])
        ds2 = ds2.reindex(time=timesteps[[1, 2]])
        ds3 = ds3.reindex(time=timesteps[[0, 1, 2]])

        # Diameter: keep first 3 bins for ds1, last 3 for ds2, middle 2 for ds3
        ds1 = ds1.isel(diameter_bin_center=slice(0, 3))  # [0.2, 0.4, 0.6]
        ds2 = ds2.isel(diameter_bin_center=slice(1, 4))  # [0.4, 0.6, 0.8]
        ds3 = ds3.isel(diameter_bin_center=slice(1, 3))  # [0.4, 0.6]

        # Align the datasets
        aligned_datasets = ds1.disdrodb.align(ds2, ds3)

        # Check dimensions: time=[1], diameter=[0.4, 0.6]
        expected_time = np.array(timesteps[1])
        expected_diameter = np.array([0.4, 0.6])

        for ds in aligned_datasets:
            np.testing.assert_array_equal(ds.time.data.astype("M8[s]"), expected_time.astype("M8[s]"))
            np.testing.assert_array_equal(ds.diameter_bin_center.data, expected_diameter)
