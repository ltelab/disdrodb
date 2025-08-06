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
"""Testing PSD fitting."""

import pytest
import xarray as xr

from disdrodb.psd.fitting import estimate_model_parameters
from disdrodb.tests.fake_datasets import create_template_l2e_dataset


class TestEstimateModelParameters:
    """Test the generate_l2_empirical product function."""

    def test_NormalizedGammaPSD_fitting(self):
        """Test Normalized Gamma PSD fitting."""
        ds = create_template_l2e_dataset()

        ds_out = estimate_model_parameters(ds, psd_model="NormalizedGammaPSD", optimization="GS")
        assert ds_out.attrs["disdrodb_psd_model"] == "NormalizedGammaPSD"
        assert "disdrodb_psd_optimization" in ds_out.attrs
        assert "disdrodb_psd_optimization_kwargs" in ds_out.attrs

        # Test raise error
        with pytest.raises(NotImplementedError, match="ML optimization is not available"):
            estimate_model_parameters(ds, psd_model="NormalizedGammaPSD", optimization="ML")

        with pytest.raises(NotImplementedError, match="MOM optimization is not available"):
            estimate_model_parameters(ds, psd_model="NormalizedGammaPSD", optimization="MOM")

    def test_GammaPSD_fitting(self):
        """Test Gamma PSD fitting."""
        ds = create_template_l2e_dataset()

        ds_out = estimate_model_parameters(ds, psd_model="GammaPSD", optimization="GS")

        ds_out = estimate_model_parameters(ds, psd_model="GammaPSD", optimization="ML")

        ds_out = estimate_model_parameters(ds, psd_model="GammaPSD", optimization="MOM")
        assert "mom_method" in ds_out.dims
        assert ds_out.attrs["disdrodb_psd_model"] == "GammaPSD"
        assert "disdrodb_psd_optimization" in ds_out.attrs
        assert "disdrodb_psd_optimization_kwargs" in ds_out.attrs

    def test_LognormalPSD_fitting(self):
        """Test LognormalPSD fitting."""
        ds = create_template_l2e_dataset()

        ds_out = estimate_model_parameters(ds, psd_model="LognormalPSD", optimization="GS")
        ds_out = estimate_model_parameters(ds, psd_model="LognormalPSD", optimization="ML")
        ds_out = estimate_model_parameters(ds, psd_model="LognormalPSD", optimization="MOM")

        assert ds_out.attrs["disdrodb_psd_model"] == "LognormalPSD"
        assert "disdrodb_psd_optimization" in ds_out.attrs
        assert "disdrodb_psd_optimization_kwargs" in ds_out.attrs

    def test_ExponentialPSD_fitting(self):
        """Test ExponentialPSD fitting."""
        ds = create_template_l2e_dataset()

        ds_out = estimate_model_parameters(ds, psd_model="ExponentialPSD", optimization="GS")
        ds_out = estimate_model_parameters(ds, psd_model="ExponentialPSD", optimization="ML")
        ds_out = estimate_model_parameters(ds, psd_model="ExponentialPSD", optimization="MOM")

        assert ds_out.attrs["disdrodb_psd_model"] == "ExponentialPSD"
        assert "disdrodb_psd_optimization" in ds_out.attrs
        assert "disdrodb_psd_optimization_kwargs" in ds_out.attrs

    def test_fitting_without_init_method(self):
        """Test fitting without moment initialization."""
        ds = create_template_l2e_dataset()
        ds_out = estimate_model_parameters(
            ds,
            psd_model="GammaPSD",
            optimization="ML",
            optimization_kwargs={"init_method": None},
        )
        ds_out = estimate_model_parameters(
            ds,
            psd_model="GammaPSD",
            optimization="ML",
            optimization_kwargs={"init_method": "None"},
        )
        ds_out = estimate_model_parameters(
            ds,
            psd_model="GammaPSD",
            optimization="ML",
            optimization_kwargs={"init_method": ["None"]},
        )
        assert isinstance(ds_out, xr.Dataset)

    def test_fitting_with_multiple_init_method(self):
        """Test fitting with multiple initialization methods."""
        ds = create_template_l2e_dataset()
        ds_out = estimate_model_parameters(
            ds,
            psd_model="GammaPSD",
            optimization="ML",
            optimization_kwargs={"init_method": [None, "M234", "M346"]},
        )
        assert "init_method" in ds_out.dims
        assert ds_out.sizes["init_method"] == 3
