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
"""Testing PSD fitting."""

import numpy as np
import pytest
import xarray as xr

from disdrodb.psd.fitting import (
    available_mom_methods,
    available_optimization,
    check_fixed_parameters,
    check_likelihood,
    check_mom_methods,
    check_optimization,
    check_optimizer,
    check_probability_method,
    check_search_space,
    check_search_space_parameters,
    check_truncated_likelihood,
    define_gs_parameters,
    estimate_model_parameters,
    get_exponential_parameters_Zhang2008,
)
from disdrodb.psd.grid_search import (
    TRANSFORMATIONS,
)
from disdrodb.psd.models import available_psd_models, create_psd
from disdrodb.tests.fake_datasets import create_template_l2e_dataset
from disdrodb.utils.manipulations import get_diameter_coords_dict_from_bin_edges

# Define PSD test cases (psd_model, parameters)
CASES = [
    ("LognormalPSD", {"Nt": 800, "mu": 0, "sigma": 1}, ["GS", "ML", "MOM"]),
    ("ExponentialPSD", {"N0": 500, "Lambda": 1}, ["GS", "ML", "MOM"]),
    ("GammaPSD", {"N0": 500, "mu": 1.0, "Lambda": 1}, ["GS", "ML", "MOM"]),
    ("NormalizedGammaPSD", {"Nw": 10_000, "D50": 1, "mu": 2}, ["GS"]),  # ML not implemented
    ("GeneralizedGammaPSD", {"Nt": 800, "mu": 0, "c": 1, "Lambda": 1}, ["GS"]),
    ("NormalizedGeneralizedGammaPSD", {"i": 3, "j": 4, "Nc": 250, "Dc": 3, "mu": 0, "c": 1}, ["GS"]),
]

DEFAULT_ML_KWARGS = {
    "probability_method": "cdf",
    "likelihood": "multinomial",
    "truncated_likelihood": True,
    "optimizer": "Nelder-Mead",
}

# Variations: modify one at a time
ML_KWARGS_ALTERNATIVES = [
    {"probability_method": "pdf"},
    {"likelihood": "poisson"},
    {"truncated_likelihood": False},
    {"optimizer": "L-BFGS-B"},
]

ATOL = {"ML": 0.05, "GS": 0.05}


def _simulate_dataset(psd_model, parameters):
    """Helper to create a synthetic dataset with drop_number_concentration."""
    psd = create_psd(psd_model, parameters)

    diameter_bin_edges = np.linspace(0, 10)
    coords_dict = get_diameter_coords_dict_from_bin_edges(diameter_bin_edges)
    diameters = coords_dict["diameter_bin_center"][1]

    drop_number_concentration = psd(diameters)
    da = xr.DataArray(
        drop_number_concentration,
        dims="diameter_bin_center",
        coords=coords_dict,
    )
    ds = xr.Dataset({"drop_number_concentration": da})
    return ds, psd


class TestGSOptimization:
    """Test PSD parameter estimation using Grid Search (GS)."""

    @pytest.mark.parametrize(("psd_model", "parameters", "optimizations"), CASES)
    @pytest.mark.parametrize("transformation", list(TRANSFORMATIONS))
    # @pytest.mark.parametrize("loss", ["SSE", "MAE", "WD"]) # speed up tests by removing loss
    # @pytest.mark.parametrize("censoring", CENSORING) # speed up tests by removing censoring variations
    def test_gs_on_nd(self, psd_model, parameters, optimizations, transformation):
        """Test PSD parameter estimation using Grid Search (GS) on N(D)."""
        if "GS" not in optimizations:
            pytest.skip(f"GS not available for {psd_model}")

        optimization_settings = {
            "objectives": [
                {
                    "target": "N(D)",
                    "transformation": transformation,
                    "loss": "SSE",
                    "censoring": "none",
                },
            ],
            "return_loss": True,
        }
        if psd_model == "NormalizedGeneralizedGammaPSD":
            optimization_settings["fixed_parameters"] = {"i": 3, "j": 4}

        ds, psd = _simulate_dataset(psd_model, parameters)
        ds_params = estimate_model_parameters(
            ds,
            psd_model=psd.name,
            optimization="GS",
            optimization_settings=optimization_settings,
        )
        assert "cost_function" in ds_params
        ds_params = ds_params.drop_vars("cost_function")
        for var in ds_params.data_vars:
            np.testing.assert_allclose(
                ds_params[var].data,
                parameters[var],
                rtol=0.2,
                atol=ATOL["GS"],
                err_msg=f"GS fitting of {psd_model} using {transformation=} causes inaccurate {var}.",
            )

    @pytest.mark.parametrize("psd_model", available_psd_models())
    def test_gs_with_zeros_nd(self, psd_model):
        """Test PSD parameter estimation using Grid Search (GS) when N(D) is all zeros."""
        optimization_settings = {
            "objectives": [
                {
                    "target": "N(D)",
                    "censoring": "none",
                    "transformation": "identity",
                    "loss": "SSE",
                },
            ],
        }
        if psd_model == "NormalizedGeneralizedGammaPSD":
            optimization_settings["fixed_parameters"] = {"i": 3, "j": 4}

        # Create dataset
        diameter_bin_edges = np.linspace(0, 10)
        coords_dict = get_diameter_coords_dict_from_bin_edges(diameter_bin_edges)
        diameters = coords_dict["diameter_bin_center"][1]
        drop_number_concentration = np.zeros(diameters.shape)
        da_drop_number_concentration = xr.DataArray(
            drop_number_concentration,
            dims="diameter_bin_center",
            coords=coords_dict,
        )
        ds = xr.Dataset({"drop_number_concentration": da_drop_number_concentration})

        # Test output nan values
        ds_params = estimate_model_parameters(
            ds,
            psd_model=psd_model,
            optimization="GS",
            optimization_settings=optimization_settings,
        )
        for var in ds_params.data_vars:
            assert np.isnan(ds_params[var])

    @pytest.mark.parametrize("psd_model", available_psd_models())
    def test_gs_with_nan_nd(self, psd_model):
        """Test PSD parameter estimation using Grid Search (GS) when N(D) has np.nan."""
        optimization_settings = {
            "objectives": [
                {
                    "target": "N(D)",
                    "censoring": "none",
                    "transformation": "identity",
                    "loss": "SSE",
                },
            ],
        }
        if psd_model == "NormalizedGeneralizedGammaPSD":
            optimization_settings["fixed_parameters"] = {"i": 3, "j": 4}

        # Create dataset
        diameter_bin_edges = np.linspace(0, 10)
        coords_dict = get_diameter_coords_dict_from_bin_edges(diameter_bin_edges)
        diameters = coords_dict["diameter_bin_center"][1]
        drop_number_concentration = np.ones(diameters.shape)
        drop_number_concentration[2] = np.nan
        da_drop_number_concentration = xr.DataArray(
            drop_number_concentration,
            dims="diameter_bin_center",
            coords=coords_dict,
        )
        ds = xr.Dataset({"drop_number_concentration": da_drop_number_concentration})

        # Test output nan values
        ds_params = estimate_model_parameters(
            ds,
            psd_model=psd_model,
            optimization="GS",
            optimization_settings=optimization_settings,
        )
        for var in ds_params.data_vars:
            assert np.isnan(ds_params[var])

    @pytest.mark.parametrize("psd_model", available_psd_models())
    def test_gs_with_inf_nd(self, psd_model):
        """Test PSD parameter estimation using Grid Search (GS) when N(D) has np.inf."""
        optimization_settings = {
            "objectives": [
                {
                    "target": "N(D)",
                    "censoring": "none",
                    "transformation": "identity",
                    "loss": "SSE",
                },
            ],
        }
        if psd_model == "NormalizedGeneralizedGammaPSD":
            optimization_settings["fixed_parameters"] = {"i": 3, "j": 4}

        # Create dataset
        diameter_bin_edges = np.linspace(0, 10)
        coords_dict = get_diameter_coords_dict_from_bin_edges(diameter_bin_edges)
        diameters = coords_dict["diameter_bin_center"][1]
        drop_number_concentration = np.ones(diameters.shape)
        drop_number_concentration[2] = np.inf
        da_drop_number_concentration = xr.DataArray(
            drop_number_concentration,
            dims="diameter_bin_center",
            coords=coords_dict,
        )
        ds = xr.Dataset({"drop_number_concentration": da_drop_number_concentration})

        # Test output nan values
        ds_params = estimate_model_parameters(
            ds,
            psd_model=psd_model,
            optimization="GS",
            optimization_settings=optimization_settings,
        )
        for var in ds_params.data_vars:
            assert np.isnan(ds_params[var])

    def test_gs_ngg_without_fixed_parameters_raises(self):
        """Test GS works when fixed_parameters is not specified."""
        psd_model = "NormalizedGeneralizedGammaPSD"
        parameters = {"i": 3, "j": 4, "Nc": 250, "Dc": 3, "mu": 0, "c": 1}
        optimization_settings = {
            "objectives": [
                {
                    "target": "N(D)",
                    "censoring": "none",
                    "transformation": "identity",
                    "loss": "SSE",
                },
            ],
        }
        ds, psd = _simulate_dataset(psd_model, parameters)
        with pytest.raises(ValueError, match="fixed_parameters must include 'i' and 'j' moment orders"):
            estimate_model_parameters(
                ds,
                psd_model=psd.name,
                optimization="GS",
                optimization_settings=optimization_settings,
            )

    def test_gs_with_fixed_parameters_multiobjectives_search_space(self):
        """Test GS works with both fixed_parameters and search_space specified."""
        psd_model = "NormalizedGeneralizedGammaPSD"
        parameters = {"i": 3, "j": 4, "Nc": 250, "Dc": 3, "mu": 0, "c": 1}
        optimization_settings = {
            "objectives": [
                {
                    "target": "N(D)",
                    "censoring": "none",
                    "transformation": "identity",
                    "loss": "SSE",
                    "loss_weight": 0.5,
                },
                {
                    "target": "Z",
                    "censoring": "none",
                    "transformation": "identity",
                    "loss": "AE",
                    "loss_weight": 0.5,
                },
            ],
            "fixed_parameters": {"i": 3, "j": 4},
            "search_space": {
                "mu": {"min": -2, "max": 2, "step": 0.5},
                "c": {"min": 0, "max": 2, "step": 0.5},
            },
        }
        ds, psd = _simulate_dataset(psd_model, parameters)
        ds_params = estimate_model_parameters(
            ds,
            psd_model=psd.name,
            optimization="GS",
            optimization_settings=optimization_settings,
        )
        assert isinstance(ds_params, xr.Dataset)
        for var in ds_params.data_vars:
            # Just verify we got some parameters, not checking accuracy
            assert not np.isnan(ds_params[var].data).all()


@pytest.mark.parametrize(("psd_model", "parameters", "optimizations"), CASES)
class TestMLOptimization:
    """Test PSD parameter estimation using Maximum Likelihood (ML)."""

    @pytest.mark.parametrize("modified_kwarg", ML_KWARGS_ALTERNATIVES)
    def test_ml_estimation(self, psd_model, parameters, optimizations, modified_kwarg):
        if "ML" not in optimizations:
            pytest.skip(f"ML not available for {psd_model}")

        optimization_settings = {**DEFAULT_ML_KWARGS, **modified_kwarg}

        ds, psd = _simulate_dataset(psd_model, parameters)
        ds_params = estimate_model_parameters(
            ds,
            psd_model=psd.name,
            optimization="ML",
            optimization_settings=optimization_settings,
        )

        for var in ds_params.data_vars:
            np.testing.assert_allclose(
                ds_params[var].data,
                parameters[var],
                rtol=0.2,
                atol=ATOL["ML"],
                err_msg=f"ML fitting of {psd_model} using {modified_kwarg} causes inaccurate {var}.",
            )

    def test_ml_estimation_with_mom_init(self, psd_model, parameters, optimizations):
        """Test PSD parameter estimation using Maximum Likelihood (ML) with MOM initialization."""
        if "ML" not in optimizations:
            pytest.skip(f"ML not available for {psd_model}")

        optimization_settings = {**DEFAULT_ML_KWARGS}
        mom_methods = available_mom_methods(psd_model)
        ds, psd = _simulate_dataset(psd_model, parameters)

        for init_method in mom_methods:
            optimization_settings["init_method"] = init_method
            ds_params = estimate_model_parameters(
                ds,
                psd_model=psd.name,
                optimization="ML",
                optimization_settings=optimization_settings,
            )
            for var in ds_params.data_vars:
                np.testing.assert_allclose(
                    ds_params[var].data,
                    parameters[var],
                    rtol=0.2,
                    atol=ATOL["ML"],
                    err_msg=f"ML fitting of {psd_model} using {init_method=} causes inaccurate {var}.",
                )

    def test_ml_with_zeros_nd(self, psd_model, parameters, optimizations):
        """Test PSD parameter estimation using Maximum Likelihood (ML) when N(D) is all zeros."""
        if "ML" not in optimizations:
            pytest.skip(f"ML not available for {psd_model}")

        # Create dataset
        diameter_bin_edges = np.linspace(0, 10)
        coords_dict = get_diameter_coords_dict_from_bin_edges(diameter_bin_edges)
        diameters = coords_dict["diameter_bin_center"][1]
        drop_number_concentration = np.zeros(diameters.shape)
        da_drop_number_concentration = xr.DataArray(
            drop_number_concentration,
            dims="diameter_bin_center",
            coords=coords_dict,
        )
        ds = xr.Dataset({"drop_number_concentration": da_drop_number_concentration})

        # Test output nan values
        ds_params = estimate_model_parameters(ds, psd_model=psd_model, optimization="ML")
        for var in ds_params.data_vars:
            assert np.isnan(ds_params[var])

    def test_ml_with_nan_nd(self, psd_model, parameters, optimizations):
        """Test PSD parameter estimation using Maximum Likelihood (ML) when N(D) has np.nan."""
        if "ML" not in optimizations:
            pytest.skip(f"ML not available for {psd_model}")

        # Create dataset
        diameter_bin_edges = np.linspace(0, 10)
        coords_dict = get_diameter_coords_dict_from_bin_edges(diameter_bin_edges)
        diameters = coords_dict["diameter_bin_center"][1]
        drop_number_concentration = np.ones(diameters.shape)
        drop_number_concentration[2] = np.nan
        da_drop_number_concentration = xr.DataArray(
            drop_number_concentration,
            dims="diameter_bin_center",
            coords=coords_dict,
        )
        ds = xr.Dataset({"drop_number_concentration": da_drop_number_concentration})

        # Test output nan values
        ds_params = estimate_model_parameters(ds, psd_model=psd_model, optimization="ML")
        for var in ds_params.data_vars:
            assert np.isnan(ds_params[var])

    def test_ml_with_inf_nd(self, psd_model, parameters, optimizations):
        """Test PSD parameter estimation using Maximum Likelihood (ML) when N(D) has np.inf."""
        if "ML" not in optimizations:
            pytest.skip(f"ML not available for {psd_model}")

        # Create dataset
        diameter_bin_edges = np.linspace(0, 10)
        coords_dict = get_diameter_coords_dict_from_bin_edges(diameter_bin_edges)
        diameters = coords_dict["diameter_bin_center"][1]
        drop_number_concentration = np.ones(diameters.shape)
        drop_number_concentration[2] = np.inf
        da_drop_number_concentration = xr.DataArray(
            drop_number_concentration,
            dims="diameter_bin_center",
            coords=coords_dict,
        )
        ds = xr.Dataset({"drop_number_concentration": da_drop_number_concentration})

        # Test output nan values
        ds_params = estimate_model_parameters(ds, psd_model=psd_model, optimization="ML")

        for var in ds_params.data_vars:
            assert np.isnan(ds_params[var])


@pytest.mark.parametrize(("psd_model", "parameters", "optimizations"), CASES)
class TestMOMOptimization:
    """Test PSD parameter estimation using Method of Moments (MOM)."""

    def test_mom_estimation(self, psd_model, parameters, optimizations):
        if "MOM" not in optimizations:
            pytest.skip(f"MOM not available for {psd_model}")

        mom_methods = available_mom_methods(psd_model)
        optimization_settings = {}
        optimization_settings["mom_methods"] = mom_methods

        ds, psd = _simulate_dataset(psd_model, parameters)
        ds_params = estimate_model_parameters(
            ds,
            psd_model=psd.name,
            optimization="MOM",
            optimization_settings=optimization_settings,
        )

        assert isinstance(ds_params, xr.Dataset)
        # Just test it runs ... because MOM methods do not provide good values

    def test_mom_with_zeros_nd(self, psd_model, parameters, optimizations):
        """Test PSD parameter estimation using Method of Moments (MOM) when N(D) is all zeros."""
        if "MOM" not in optimizations:
            pytest.skip(f"MOM not available for {psd_model}")

        mom_methods = available_mom_methods(psd_model)
        optimization_settings = {}
        optimization_settings["mom_methods"] = mom_methods

        # Create dataset
        diameter_bin_edges = np.linspace(0, 10)
        coords_dict = get_diameter_coords_dict_from_bin_edges(diameter_bin_edges)
        diameters = coords_dict["diameter_bin_center"][1]
        drop_number_concentration = np.zeros(diameters.shape)
        da_drop_number_concentration = xr.DataArray(
            drop_number_concentration,
            dims="diameter_bin_center",
            coords=coords_dict,
        )
        ds = xr.Dataset({"drop_number_concentration": da_drop_number_concentration})

        # Test output nan values
        ds_params = estimate_model_parameters(
            ds,
            psd_model=psd_model,
            optimization="MOM",
            optimization_settings=optimization_settings,
        )
        for var in ds_params.data_vars:
            assert np.all(np.isnan(ds_params[var]))

    def test_mom_with_nan_nd(self, psd_model, parameters, optimizations):
        """Test PSD parameter estimation using Method of Moments (MOM) when N(D) has np.nan."""
        if "MOM" not in optimizations:
            pytest.skip(f"MOM not available for {psd_model}")

        mom_methods = available_mom_methods(psd_model)
        optimization_settings = {}
        optimization_settings["mom_methods"] = mom_methods

        # Create dataset
        diameter_bin_edges = np.linspace(0, 10)
        coords_dict = get_diameter_coords_dict_from_bin_edges(diameter_bin_edges)
        diameters = coords_dict["diameter_bin_center"][1]
        drop_number_concentration = np.ones(diameters.shape)
        drop_number_concentration[2] = np.nan
        da_drop_number_concentration = xr.DataArray(
            drop_number_concentration,
            dims="diameter_bin_center",
            coords=coords_dict,
        )
        ds = xr.Dataset({"drop_number_concentration": da_drop_number_concentration})

        # Test output nan values
        ds_params = estimate_model_parameters(
            ds,
            psd_model=psd_model,
            optimization="MOM",
            optimization_settings=optimization_settings,
        )
        for var in ds_params.data_vars:
            assert np.all(np.isnan(ds_params[var]))

    def test_mom_with_inf_nd(self, psd_model, parameters, optimizations):
        """Test PSD parameter estimation using Method of Moments (MOM) when N(D) has np.inf."""
        if "MOM" not in optimizations:
            pytest.skip(f"MOM not available for {psd_model}")

        mom_methods = available_mom_methods(psd_model)
        optimization_settings = {}
        optimization_settings["mom_methods"] = mom_methods

        # Create dataset
        diameter_bin_edges = np.linspace(0, 10)
        coords_dict = get_diameter_coords_dict_from_bin_edges(diameter_bin_edges)
        diameters = coords_dict["diameter_bin_center"][1]
        drop_number_concentration = np.ones(diameters.shape)
        drop_number_concentration[2] = np.inf
        da_drop_number_concentration = xr.DataArray(
            drop_number_concentration,
            dims="diameter_bin_center",
            coords=coords_dict,
        )
        ds = xr.Dataset({"drop_number_concentration": da_drop_number_concentration})

        # Test output nan values
        ds_params = estimate_model_parameters(
            ds,
            psd_model=psd_model,
            optimization="MOM",
            optimization_settings=optimization_settings,
        )
        for var in ds_params.data_vars:
            assert np.all(np.isnan(ds_params[var]))


####-------------------------------------------------------------------------------------------------
#### Test checks


def test_available_mom_methods():
    """Test available_mom_methods."""
    mom_methods = available_mom_methods("GammaPSD")
    assert isinstance(mom_methods, list)
    assert "M234" in mom_methods

    # Assert raise error
    with pytest.raises(NotImplementedError, match="NormalizedGammaPSD"):
        available_mom_methods("NormalizedGammaPSD")


@pytest.mark.parametrize("psd_model", available_psd_models())
def test_available_optimization(psd_model):
    """Test available_optimization."""
    optimizations = available_optimization(psd_model)
    assert isinstance(optimizations, list)
    assert "GS" in optimizations


class TestCheckLikelihood:
    """Test suite for check_likelihood."""

    @pytest.mark.parametrize("valid", ["multinomial", "poisson"])
    def test_valid_likelihoods(self, valid):
        """Valid likelihoods should be returned unchanged."""
        assert check_likelihood(valid) == valid

    def test_invalid_likelihood_raises(self):
        """Invalid likelihood should raise ValueError."""
        with pytest.raises(ValueError, match="Invalid 'likelihood'"):
            check_likelihood("INVALID")


class TestCheckTruncatedLikelihood:
    """Test suite for check_truncated_likelihood."""

    @pytest.mark.parametrize("valid", [True, False])
    def test_valid_bool(self, valid):
        """Valid boolean should be returned unchanged."""
        assert check_truncated_likelihood(valid) is valid

    @pytest.mark.parametrize("invalid", ["yes", 1, None, [True]])
    def test_invalid_type_raises(self, invalid):
        """Non-bool values should raise TypeError."""
        with pytest.raises(TypeError, match="Invalid 'truncated_likelihood'"):
            check_truncated_likelihood(invalid)


class TestCheckProbabilityMethod:
    """Test suite for check_probability_method."""

    @pytest.mark.parametrize("valid", ["cdf", "pdf"])
    def test_valid_methods(self, valid):
        """Valid probability methods should be returned unchanged."""
        assert check_probability_method(valid) == valid

    def test_invalid_method_raises(self):
        """Invalid probability method should raise ValueError."""
        with pytest.raises(ValueError, match="Invalid 'probability_method'"):
            check_probability_method("INVALID")


class TestCheckOptimizer:
    """Test suite for check_optimizer."""

    @pytest.mark.parametrize("valid", ["Nelder-Mead", "Powell", "L-BFGS-B"])
    def test_valid_optimizers(self, valid):
        """Valid optimizers should be returned unchanged."""
        assert check_optimizer(valid) == valid

    def test_invalid_optimizer_raises(self):
        """Invalid optimizer should raise ValueError."""
        with pytest.raises(ValueError, match="Invalid 'optimizer'"):
            check_optimizer("INVALID")


class TestCheckMomMethods:
    """Test suite for check_mom_methods."""

    @pytest.mark.parametrize("psd_model", available_psd_models())
    def test_valid_mom_methods(self, psd_model):
        """Valid MOM methods should be returned unchanged."""
        try:
            valid_methods = available_mom_methods(psd_model)
        except Exception:
            pytest.skip(f"No MOM methods for {psd_model}")

        for method in valid_methods:
            result = check_mom_methods(method, psd_model)
            assert result == [method]

    def test_list_input(self):
        """List of methods should be preserved."""
        psd_model = "ExponentialPSD"
        valid_methods = available_mom_methods(psd_model)
        if not valid_methods:
            pytest.skip(f"No MOM methods for {psd_model}")
        result = check_mom_methods(valid_methods, psd_model)
        assert result == valid_methods

    def test_allow_none(self):
        """'None' should be allowed when allow_none=True."""
        psd_model = "ExponentialPSD"
        result = check_mom_methods(None, psd_model, allow_none=True)
        assert result == ["None"]

    def test_invalid_method_raises(self):
        """Invalid methods should raise ValueError."""
        psd_model = "ExponentialPSD"
        with pytest.raises(ValueError, match="Unknown mom_methods"):
            check_mom_methods("INVALID", psd_model)


class TestCheckOptimization:
    """Test suite for check_optimization."""

    @pytest.mark.parametrize("valid", ["ML", "GS", "MOM"])
    def test_valid_optimizations(self, valid):
        """Valid optimizations should be returned unchanged."""
        assert check_optimization(valid) == valid

    def test_invalid_optimization_raises(self):
        """Invalid optimizations should raise ValueError."""
        with pytest.raises(ValueError, match="Invalid 'optimization'"):
            check_optimization("INVALID")


####-------------------------------------------------------------------------------------------------
class TestEstimateModelParameters:
    """Test that estimate_model_parameters runs and returns expected structure."""

    def test_NormalizedGeneralizedGammaPSD_fitting(self):
        """Test Normalized Gamma PSD fitting."""
        ds = create_template_l2e_dataset()

        ds_out = estimate_model_parameters(
            ds,
            psd_model="NormalizedGeneralizedGammaPSD",
            optimization="GS",
            optimization_settings={"fixed_parameters": {"i": 3, "j": 4}},
        )
        assert ds_out.attrs["disdrodb_psd_model"] == "NormalizedGeneralizedGammaPSD"
        assert "disdrodb_psd_optimization" in ds_out.attrs
        assert "disdrodb_psd_optimization_settings" in ds_out.attrs

        # Test raise error
        with pytest.raises(NotImplementedError, match="ML optimization is not available"):
            estimate_model_parameters(ds, psd_model="NormalizedGammaPSD", optimization="ML")

        with pytest.raises(NotImplementedError, match="MOM optimization is not available"):
            estimate_model_parameters(ds, psd_model="NormalizedGammaPSD", optimization="MOM")

    def test_GeneralizedGammaPSD_fitting(self):
        """Test Generalized Gamma PSD fitting."""
        ds = create_template_l2e_dataset()

        ds_out = estimate_model_parameters(ds, psd_model="GeneralizedGammaPSD", optimization="GS")
        assert ds_out.attrs["disdrodb_psd_model"] == "GeneralizedGammaPSD"
        assert "disdrodb_psd_optimization" in ds_out.attrs
        assert "disdrodb_psd_optimization_settings" in ds_out.attrs

        # Test raise error
        with pytest.raises(NotImplementedError, match="ML optimization is not available"):
            estimate_model_parameters(ds, psd_model="NormalizedGammaPSD", optimization="ML")

        with pytest.raises(NotImplementedError, match="MOM optimization is not available"):
            estimate_model_parameters(ds, psd_model="NormalizedGammaPSD", optimization="MOM")

    def test_NormalizedGammaPSD_fitting(self):
        """Test Normalized Gamma PSD fitting."""
        ds = create_template_l2e_dataset()

        ds_out = estimate_model_parameters(ds, psd_model="NormalizedGammaPSD", optimization="GS")
        assert ds_out.attrs["disdrodb_psd_model"] == "NormalizedGammaPSD"
        assert "disdrodb_psd_optimization" in ds_out.attrs
        assert "disdrodb_psd_optimization_settings" in ds_out.attrs

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
        assert "disdrodb_psd_optimization_settings" in ds_out.attrs

    def test_LognormalPSD_fitting(self):
        """Test LognormalPSD fitting."""
        ds = create_template_l2e_dataset()

        ds_out = estimate_model_parameters(ds, psd_model="LognormalPSD", optimization="GS")
        ds_out = estimate_model_parameters(ds, psd_model="LognormalPSD", optimization="ML")
        ds_out = estimate_model_parameters(ds, psd_model="LognormalPSD", optimization="MOM")

        assert ds_out.attrs["disdrodb_psd_model"] == "LognormalPSD"
        assert "disdrodb_psd_optimization" in ds_out.attrs
        assert "disdrodb_psd_optimization_settings" in ds_out.attrs

    def test_ExponentialPSD_fitting(self):
        """Test ExponentialPSD fitting."""
        ds = create_template_l2e_dataset()

        ds_out = estimate_model_parameters(ds, psd_model="ExponentialPSD", optimization="GS")
        ds_out = estimate_model_parameters(ds, psd_model="ExponentialPSD", optimization="ML")
        ds_out = estimate_model_parameters(ds, psd_model="ExponentialPSD", optimization="MOM")

        assert ds_out.attrs["disdrodb_psd_model"] == "ExponentialPSD"
        assert "disdrodb_psd_optimization" in ds_out.attrs
        assert "disdrodb_psd_optimization_settings" in ds_out.attrs

    def test_fitting_without_init_method(self):
        """Test fitting without moment initialization."""
        ds = create_template_l2e_dataset()
        ds_out = estimate_model_parameters(
            ds,
            psd_model="GammaPSD",
            optimization="ML",
            optimization_settings={"init_method": None},
        )
        ds_out = estimate_model_parameters(
            ds,
            psd_model="GammaPSD",
            optimization="ML",
            optimization_settings={"init_method": "None"},
        )
        ds_out = estimate_model_parameters(
            ds,
            psd_model="GammaPSD",
            optimization="ML",
            optimization_settings={"init_method": ["None"]},
        )
        assert isinstance(ds_out, xr.Dataset)

    def test_fitting_with_multiple_init_method(self):
        """Test fitting with multiple initialization methods."""
        ds = create_template_l2e_dataset()
        ds_out = estimate_model_parameters(
            ds,
            psd_model="GammaPSD",
            optimization="ML",
            optimization_settings={"init_method": [None, "M234", "M346"]},
        )
        assert "init_method" in ds_out.dims
        assert ds_out.sizes["init_method"] == 3

    def test_raise_error_if_missing_field(self):
        """Test raise error if drop_number_concentration variable is not available."""
        with pytest.raises(ValueError):
            estimate_model_parameters(
                ds=xr.Dataset(),
                psd_model="GammaPSD",
                optimization="ML",
                optimization_settings={"init_method": None},
            )


####---------------------------------------------------------------------------
#### Test other functions
class TestGetExponentialParametersZhang2008:
    """Test suite for get_exponential_parameters_Zhang2008."""

    def test_known_case(self):
        """Check parameters against manual calculation for known inputs."""
        moment_l = 2.0
        moment_m = 4.0
        l, m = 0, 1  # noqa: E741
        N0, Lambda = get_exponential_parameters_Zhang2008(moment_l, moment_m, l, m)
        np.testing.assert_allclose(Lambda, 0.5)
        np.testing.assert_allclose(N0, 1.0)

    @pytest.mark.parametrize(("l", "m"), [(0, 1), (1, 2), (2, 3)])
    def test_various_moments(self, l, m):  # noqa: E741
        """Test calculation works for multiple (l, m) moment order pairs."""
        moment_l = 5.0
        moment_m = 10.0
        N0, Lambda = get_exponential_parameters_Zhang2008(moment_l, moment_m, l, m)
        assert np.isfinite(N0)
        assert np.isfinite(Lambda)
        assert N0 > 0
        assert Lambda > 0

    def test_invalid_equal_moments(self):
        """Check raises or fails gracefully when l == m."""
        moment_l = 1.0
        moment_m = 2.0
        l = m = 1  # noqa: E741
        with pytest.raises(ValueError):
            get_exponential_parameters_Zhang2008(moment_l, moment_m, l, m)


####---------------------------------------------------------------------------
#### Test check_fixed_parameters


class TestCheckFixedParameters:
    """Test suite for check_fixed_parameters."""

    def test_none_input_allowed(self):
        """Test that None is allowed."""
        for psd_model in ["GammaPSD", "LognormalPSD", "ExponentialPSD"]:
            result = check_fixed_parameters(psd_model, None)
            assert result is None

    def test_none_input_for_ngg_raises(self):
        """Test that None raises error for NormalizedGeneralizedGammaPSD."""
        with pytest.raises(ValueError, match="fixed_parameters must include 'i' and 'j'"):
            check_fixed_parameters("NormalizedGeneralizedGammaPSD", None)

    def test_valid_fixed_parameters(self):
        """Test valid fixed_parameters dictionary."""
        fixed_params = {"mu": 1.5, "Lambda": 2.0}
        result = check_fixed_parameters("GammaPSD", fixed_params)
        assert result == {"mu": 1.5, "Lambda": 2.0}

    def test_fixed_parameters_converted_to_float(self):
        """Test that integer values are converted to float."""
        fixed_params = {"mu": 1, "Lambda": 2}
        result = check_fixed_parameters("GammaPSD", fixed_params)
        assert result["mu"] == 1.0
        assert result["Lambda"] == 2.0
        assert isinstance(result["mu"], float)

    def test_ngg_with_required_moments(self):
        """Test NormalizedGeneralizedGammaPSD with required i and j."""
        fixed_params = {"i": 3, "j": 4, "mu": 0.5}
        result = check_fixed_parameters("NormalizedGeneralizedGammaPSD", fixed_params)
        assert result["i"] == 3.0
        assert result["j"] == 4.0
        assert result["mu"] == 0.5

    def test_ngg_missing_i_raises(self):
        """Test NormalizedGeneralizedGammaPSD raises when 'i' is missing."""
        fixed_params = {"j": 4, "mu": 0.5}
        with pytest.raises(ValueError, match="must include 'i' and 'j'"):
            check_fixed_parameters("NormalizedGeneralizedGammaPSD", fixed_params)

    def test_ngg_missing_j_raises(self):
        """Test NormalizedGeneralizedGammaPSD raises when 'j' is missing."""
        fixed_params = {"i": 3, "mu": 0.5}
        with pytest.raises(ValueError, match="must include 'i' and 'j'"):
            check_fixed_parameters("NormalizedGeneralizedGammaPSD", fixed_params)

    def test_not_dict_raises(self):
        """Test that non-dict input raises ValueError."""
        with pytest.raises(ValueError, match="fixed_parameters must be a dictionary"):
            check_fixed_parameters("GammaPSD", [1, 2, 3])

    def test_string_value_raises(self):
        """Test that string values are not allowed."""
        fixed_params = {"mu": "invalid"}
        with pytest.raises(ValueError, match="strings are not allowed"):
            check_fixed_parameters("GammaPSD", fixed_params)

    def test_array_value_raises(self):
        """Test that array values are not allowed."""
        fixed_params = {"mu": np.array([1, 2])}
        with pytest.raises(ValueError, match="expected scalar"):
            check_fixed_parameters("GammaPSD", fixed_params)

    def test_invalid_parameter_name_raises(self):
        """Test that invalid parameter names raise error."""
        fixed_params = {"invalid_param": 1.0}
        with pytest.raises(ValueError):
            check_fixed_parameters("GammaPSD", fixed_params)


####---------------------------------------------------------------------------
#### Test check_search_space_parameters


class TestCheckSearchSpaceParameters:
    """Test suite for check_search_space_parameters."""

    def test_none_input(self):
        """Test that None returns None."""
        result = check_search_space_parameters(None, "GammaPSD")
        assert result is None

    def test_empty_dict(self):
        """Test that empty dict returns empty dict."""
        result = check_search_space_parameters({}, "GammaPSD")
        assert result == {}

    def test_valid_parameters(self):
        """Test valid search space parameters."""
        search_space = {
            "mu": {"min": 0, "max": 10, "step": 0.5},
            "Lambda": {"min": 0, "max": 5, "step": 0.1},
        }
        result = check_search_space_parameters(search_space, "GammaPSD")
        assert result == search_space

    def test_invalid_parameter_name_raises(self):
        """Test that invalid parameter names raise error."""
        search_space = {
            "invalid_param": {"min": 0, "max": 10, "step": 0.5},
        }
        with pytest.raises(ValueError):
            check_search_space_parameters(search_space, "GammaPSD")

    def test_multiple_valid_parameters(self):
        """Test multiple valid parameters for different models."""
        search_space = {"mu": {"min": 0, "max": 10, "step": 0.5}}
        for psd_model in ["GammaPSD", "GeneralizedGammaPSD"]:
            result = check_search_space_parameters(search_space, psd_model)
            assert result == search_space


####---------------------------------------------------------------------------
#### Test check_search_space


class TestCheckSearchSpace:
    """Test suite for check_search_space."""

    def test_none_input(self):
        """Test that None returns None."""
        result = check_search_space(None)
        assert result is None

    def test_empty_dict_returns_none(self):
        """Test that empty dict returns None."""
        result = check_search_space({})
        assert result is None

    def test_valid_search_space(self):
        """Test valid search space dictionary."""
        search_space = {
            "mu": {"min": 0, "max": 10, "step": 0.5},
            "Lambda": {"min": 0.1, "max": 5, "step": 0.1},
        }
        result = check_search_space(search_space)
        assert result == search_space

    def test_not_dict_raises(self):
        """Test that non-dict input raises ValueError."""
        with pytest.raises(ValueError, match="search_space must be a dictionary"):
            check_search_space([1, 2, 3])

    def test_missing_min_raises(self):
        """Test that missing 'min' key raises error."""
        search_space = {"mu": {"max": 10, "step": 0.5}}
        with pytest.raises(ValueError, match="must be a dict with 'min', 'max', and 'step' keys"):
            check_search_space(search_space)

    def test_missing_max_raises(self):
        """Test that missing 'max' key raises error."""
        search_space = {"mu": {"min": 0, "step": 0.5}}
        with pytest.raises(ValueError, match="must be a dict with 'min', 'max', and 'step' keys"):
            check_search_space(search_space)

    def test_missing_step_raises(self):
        """Test that missing 'step' key raises error."""
        search_space = {"mu": {"min": 0, "max": 10}}
        with pytest.raises(ValueError, match="must be a dict with 'min', 'max', and 'step' keys"):
            check_search_space(search_space)

    def test_min_equal_max_raises(self):
        """Test that min >= max raises error."""
        search_space = {"mu": {"min": 10, "max": 10, "step": 0.5}}
        with pytest.raises(ValueError, match="min .* >= max"):
            check_search_space(search_space)

    def test_min_greater_than_max_raises(self):
        """Test that min > max raises error."""
        search_space = {"mu": {"min": 15, "max": 10, "step": 0.5}}
        with pytest.raises(ValueError, match="min .* >= max"):
            check_search_space(search_space)

    def test_negative_step_raises(self):
        """Test that negative step raises error."""
        search_space = {"mu": {"min": 0, "max": 10, "step": -0.5}}
        with pytest.raises(ValueError, match="step .* must be positive"):
            check_search_space(search_space)

    def test_zero_step_raises(self):
        """Test that zero step raises error."""
        search_space = {"mu": {"min": 0, "max": 10, "step": 0}}
        with pytest.raises(ValueError, match="step .* must be positive"):
            check_search_space(search_space)


####---------------------------------------------------------------------------
#### Test define_gs_parameters


class TestDefineGsParameters:
    """Test suite for define_gs_parameters."""

    def test_none_inputs_returns_empty_dict(self):
        """Test that None inputs return empty dict."""
        result = define_gs_parameters("GammaPSD", fixed_parameters=None, search_space=None)
        assert result == {}

    def test_empty_inputs_returns_empty_dict(self):
        """Test that empty dict inputs return empty dict."""
        result = define_gs_parameters("GammaPSD", fixed_parameters={}, search_space={})
        assert result == {}

    def test_only_fixed_parameters(self):
        """Test with only fixed_parameters specified."""
        fixed_params = {"mu": 1.5, "Lambda": 2.0}
        result = define_gs_parameters("GammaPSD", fixed_parameters=fixed_params, search_space=None)
        assert result["mu"] == 1.5
        assert result["Lambda"] == 2.0

    def test_only_search_space(self):
        """Test with only search_space specified."""
        search_space = {
            "mu": {"min": 0, "max": 3, "step": 1},
            "Lambda": {"min": 0, "max": 2, "step": 1},
        }
        result = define_gs_parameters("GammaPSD", fixed_parameters=None, search_space=search_space)
        np.testing.assert_array_equal(result["mu"], np.arange(0, 4, 1))
        np.testing.assert_array_equal(result["Lambda"], np.arange(0, 3, 1))

    def test_combined_fixed_and_search_space(self):
        """Test with both fixed_parameters and search_space."""
        fixed_params = {"mu": 1.5}
        search_space = {"Lambda": {"min": 0, "max": 2, "step": 1}}
        result = define_gs_parameters("GammaPSD", fixed_parameters=fixed_params, search_space=search_space)
        assert result["mu"] == 1.5
        np.testing.assert_array_equal(result["Lambda"], np.arange(0, 3, 1))

    def test_exponential_psd(self):
        """Test ExponentialPSD with single parameter."""
        search_space = {"Lambda": {"min": 0.5, "max": 2.5, "step": 1}}
        result = define_gs_parameters("ExponentialPSD", fixed_parameters=None, search_space=search_space)
        np.testing.assert_array_equal(result["Lambda"], np.arange(0.5, 3.5, 1))

    def test_lognormal_psd(self):
        """Test LognormalPSD with multiple parameters."""
        fixed_params = {"mu": 0.5}
        search_space = {"sigma": {"min": 0.5, "max": 1.5, "step": 0.5}}
        result = define_gs_parameters("LognormalPSD", fixed_parameters=fixed_params, search_space=search_space)
        assert result["mu"] == 0.5
        np.testing.assert_array_equal(result["sigma"], np.arange(0.5, 2.0, 0.5))

    def test_generalized_gamma_psd(self):
        """Test GeneralizedGammaPSD with multiple parameters."""
        search_space = {
            "mu": {"min": -1, "max": 1, "step": 1},
            "c": {"min": 0.5, "max": 1.5, "step": 0.5},
            "Lambda": {"min": 0.5, "max": 1.5, "step": 0.5},
        }
        result = define_gs_parameters("GeneralizedGammaPSD", fixed_parameters=None, search_space=search_space)
        assert isinstance(result["mu"], np.ndarray)
        assert isinstance(result["c"], np.ndarray)
        assert isinstance(result["Lambda"], np.ndarray)

    def test_ngg_with_required_i_j(self):
        """Test NormalizedGeneralizedGammaPSD with required i and j."""
        fixed_params = {"i": 3, "j": 4, "mu": 0.5}
        search_space = {"c": {"min": 0, "max": 2, "step": 1}}
        result = define_gs_parameters(
            "NormalizedGeneralizedGammaPSD",
            fixed_parameters=fixed_params,
            search_space=search_space,
        )
        assert result["i"] == 3
        assert result["j"] == 4
        assert result["mu"] == 0.5
        np.testing.assert_array_equal(result["c"], np.arange(0, 3, 1))

    def test_ngg_without_fixed_parameters_raises(self):
        """Test NormalizedGeneralizedGammaPSD raises without i and j in fixed_parameters."""
        with pytest.raises(ValueError, match="fixed_parameters must include 'i' and 'j'"):
            define_gs_parameters("NormalizedGeneralizedGammaPSD", fixed_parameters=None, search_space=None)

    def test_float_step_size(self):
        """Test that float step sizes work correctly."""
        search_space = {"mu": {"min": 0, "max": 1, "step": 0.25}}
        result = define_gs_parameters("GammaPSD", fixed_parameters=None, search_space=search_space)
        np.testing.assert_allclose(result["mu"], np.arange(0, 1.25, 0.25))
