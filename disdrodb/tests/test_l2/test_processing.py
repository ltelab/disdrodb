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
"""Testing module for L2E processing."""
import pytest
import xarray as xr
import disdrodb.psd
from disdrodb.l2.processing import check_l2e_input_dataset, generate_l2_empirical, generate_l2_model
from disdrodb.tests.fake_datasets import create_template_dataset, create_template_l2e_dataset
from disdrodb import DIAMETER_DIMENSION
from disdrodb.scattering import RADAR_OPTIONS


class TestCheckL2eInputDataset:
    """Test suite for the check_l2e_input_dataset function."""

    def test_valid_dataset(self):
        """Dataset from template should pass validation without errors."""
        ds = create_template_dataset()
        assert isinstance(check_l2e_input_dataset(ds), xr.Dataset)

    def test_missing_variables(self):
        """Missing a required variable should raise ValueError with appropriate message."""
        ds = create_template_dataset()
        ds = ds.drop_vars(["drop_number"])
        with pytest.raises(ValueError, match=r"variables.*drop_number"):
            check_l2e_input_dataset(ds)
  
    def test_missing_coords(self):
        """Missing a required coordinate should raise ValueError with appropriate message."""
        ds = create_template_dataset()
        ds = ds.drop_vars(["diameter_bin_center"])
        with pytest.raises(ValueError, match=r"coords.*diameter_bin_center"):
            check_l2e_input_dataset(ds)
            
    def test_missing_dimension(self):
        """Missing a required dimension should raise ValueError with appropriate message."""
        ds = create_template_dataset()
        ds = ds.rename_dims({DIAMETER_DIMENSION:"dummy_dim"})
        with pytest.raises(ValueError, match=r"dimensions.*"):
            check_l2e_input_dataset(ds)

    def test_missing_attribute(self):
        """Missing a required attribute should raise ValueError with appropriate message."""
        ds = create_template_dataset()
        ds.attrs.pop("sensor_name", None)
        with pytest.raises(ValueError, match=r"attributes.*sensor_name"):
            check_l2e_input_dataset(ds)
    
    def test_unallowed_dimension(self):
        """Missing a required dimension should raise ValueError with appropriate message."""
        ds = create_template_dataset()
        ds = ds.expand_dims({"source": [1, 2]})
        with pytest.raises(ValueError, match=r"dimensions.*"):
            check_l2e_input_dataset(ds)
    
    @pytest.mark.parametrize(
        "dimension",
        ["source", "velocity_method", RADAR_OPTIONS[0]]
    )
    def test_remove_unallowed_dimensions(self, dimension): 
        """Test unallowed dimensions and coordinates are removed silently."""
        ds = create_template_dataset()
        ds = ds.expand_dims({dimension: ["dummy1", "dummy2"]})
        assert dimension in ds.coords
        assert dimension in ds.dims
        ds_out = check_l2e_input_dataset(ds)
        assert dimension not in ds_out.coords
        assert dimension not in ds_out.dims


class TestGenerateL2Empirical:
    """Test the generate_l2_empirical product function."""

    def test_spectrum_with_velocity_dimension(self):
        """Spectrum dataset with velocity dimension should include new dims."""
        ds = create_template_dataset(with_velocity=True)
        ds_out = generate_l2_empirical(ds)
        assert "velocity_method" in ds_out.dims
        assert "source" in ds_out.dims
        assert isinstance(ds_out, xr.Dataset)

    def test_spectrum_without_velocity_dimension(self):
        """Spectrum dataset without velocity dim should not add velocity dims."""
        ds = create_template_dataset(with_velocity=False)
        ds_out = generate_l2_empirical(ds)
        assert "velocity_method" not in ds_out.dims
        assert "source" not in ds_out.dims
        assert isinstance(ds_out, xr.Dataset)

    def test_without_time_dimension(self):
        """Dataset without time dimension should still process correctly."""
        ds = create_template_dataset(with_velocity=True)
        ds = ds.isel(time=0)
        ds_out = generate_l2_empirical(ds)
        assert isinstance(ds_out, xr.Dataset)

    def test_additional_dimension_preserved(self):
        """Additional dimensions should be preserved in output dims."""
        ds = create_template_dataset(with_velocity=True)
        ds = ds.expand_dims({"year": [2012, 2013]})
        ds_out = generate_l2_empirical(ds)
        assert "year" in ds_out.dims
        
    def test_with_lazy_dask_array(self):
        """Test it correctly deals with dask arrays."""
        ds = create_template_dataset(with_velocity=True)
        ds_lazy = ds.chunk({"time": 1})
        ds_out = generate_l2_empirical(ds_lazy)
        # Check returns dask array
        assert isinstance(ds_out, xr.Dataset)
        assert hasattr(ds_out["Dmode"].data, "chunks") 
        assert hasattr(ds_out["D50"].data, "chunks") 
        # Test it can compute without error
        ds_out = ds_out.compute()
        assert isinstance(ds_out, xr.Dataset)
        # Test equaliy with in-memory computing 
        ds_out1 = generate_l2_empirical(ds)
        xr.testing.assert_allclose(ds_out, ds_out1)

    def test_idempotent_l2e_generation(self):
        """Regenerating L2E with L2E dataset should produce identical results."""
        ds = create_template_dataset(with_velocity=True)
        ds_out = generate_l2_empirical(ds)
        ds_out2 = generate_l2_empirical(ds_out)
        xr.testing.assert_allclose(ds_out, ds_out2)


class TestGenerateL2Model:
    """Test the generate_l2_empirical product function."""
    
    @pytest.mark.parametrize(
        "psd_model",
        [disdrodb.psd.available_psd_models()]
    )
    def test_with_in_memory_numpy_array(self, psd_model):
        """Test L2M product generation with in-memory numpy data."""
        ds = create_template_l2e_dataset(with_velocity=True)
        ds_out = generate_l2_model(ds, psd_model=psd_model)
        assert isinstance(ds_out, xr.Dataset)
        
    @pytest.mark.parametrize(
        "psd_model",
        [disdrodb.psd.available_psd_models()]
    )
    def test_with_lazy_dask_array(self, psd_model):
        """Test L2M product generation with lazy dask array data."""
        ds = create_template_l2e_dataset(with_velocity=True)
        ds_lazy = ds.chunk({"time": 1})
        ds_out = generate_l2_model(ds_lazy, psd_model=psd_model)
        # Test it returns dask arrays
        assert isinstance(ds_out, xr.Dataset)
        assert hasattr(ds_out["R"].data, "chunks") 
        assert hasattr(ds_out["kl_divergence"].data, "chunks") 
        
        # Test it can compute without error
        ds_out = ds_out.compute()
        assert isinstance(ds_out, xr.Dataset)
        
        # Test equaliy with in-memory computing 
        ds_out1 = generate_l2_model(ds, psd_model=psd_model)
        xr.testing.assert_allclose(ds_out, ds_out1)

    def test_without_time_dimension(self):
        """Dataset without time dimension should still process correctly."""
        ds = create_template_l2e_dataset(with_velocity=True)
        ds = ds.isel(time=0)
        ds_out = generate_l2_model(ds, psd_model="NormalizedGammaPSD")
        assert isinstance(ds_out, xr.Dataset)

    def test_additional_dimension_preserved(self):
        """Additional dimensions should be preserved in output dims."""
        ds = create_template_l2e_dataset(with_velocity=True)
        ds = ds.expand_dims({"year": [2012, 2013]})
        ds_out = generate_l2_model(ds, psd_model="NormalizedGammaPSD")
        assert "year" in ds_out.dims

    def test_idempotent_generation(self):
        """Regenerating L2M with L2M dataset should produce identical results."""
        ds = create_template_l2e_dataset(with_velocity=True)
        ds_out = generate_l2_model(ds, psd_model="NormalizedGammaPSD")
        ds_out2 = generate_l2_model(ds_out, psd_model="NormalizedGammaPSD")
        xr.testing.assert_allclose(ds_out, ds_out2)
        
    def test_it_accept_only_optimization_kwargs(self):
        """Test it accepts only optimization argument without additional kwargs."""
        ds = create_template_l2e_dataset(with_velocity=True)
        ds_out = generate_l2_model(ds, psd_model="NormalizedGammaPSD", optimization="GS")

        
#------------------------------------------------------------------------------------------.
# Define default PSD model optimization if not specified 
# remove get_default_optimization_options and ensure it works using function defaults
# --> estimate_model_parameters too !


# generate_l2_model(ds, psd_model="GammaPSD")
# generate_l2_model(ds, psd_model="NormalizedGammaPSD")
# generate_l2_model(ds, psd_model="ExponentialPSD")


# generate_l2_model(ds, psd_model="GammaPSD", optimization="ML", optimization_kwargs={})
# generate_l2_model(ds, psd_model="NormalizedGammaPSD", optimization="GS", optimization_kwargs={})
# generate_l2_model(ds, psd_model="ExponentialPSD", optimization="ML", optimization_kwargs={})



 
