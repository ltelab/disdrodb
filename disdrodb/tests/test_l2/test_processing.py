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
"""Testing module for L2E processing."""

import numpy as np
import pytest
import xarray as xr

import disdrodb.l2.processing as l2_processing
import disdrodb.psd
from disdrodb.constants import DIAMETER_DIMENSION, VELOCITY_DIMENSION
from disdrodb.fall_velocity import available_rain_fall_velocity_models
from disdrodb.l2.processing import (
    check_l2e_input_dataset,
    define_rain_spectrum_mask,
    generate_l2e,
    generate_l2m,
    get_mask_contour,
    get_spectrum_mask_boundary,
)
from disdrodb.scattering import RADAR_OPTIONS
from disdrodb.tests.fake_datasets import create_template_dataset, create_template_l2e_dataset
from disdrodb.utils.warnings import suppress_warnings


def create_test_dataset():
    """Create a small synthetic dataset with diameter and velocity bins."""
    ds = xr.Dataset(
        {
            "diameter_bin_lower": ("diameter", [0.5, 1.0, 2.0, 3.0]),
            "diameter_bin_upper": ("diameter", [1.0, 2.0, 3.0, 4.0]),
            "velocity_bin_lower": ("velocity", [0.0, 2.0, 5.0, 10.0]),
            "velocity_bin_upper": ("velocity", [2.0, 5.0, 10.0, 15.0]),
            "drop_number": (("diameter", "velocity"), np.ones((4, 4))),
        },
    )
    ds = ds.rename({"diameter": DIAMETER_DIMENSION, "velocity": VELOCITY_DIMENSION})
    ds = ds.set_coords(["diameter_bin_lower", "diameter_bin_upper", "velocity_bin_lower", "velocity_bin_upper"])
    return ds


def create_test_mask(mask_values):
    """Create a small mask with explicit bin edge coordinates."""
    return xr.DataArray(
        np.asarray(mask_values, dtype=bool),
        dims=(VELOCITY_DIMENSION, DIAMETER_DIMENSION),
        coords={
            VELOCITY_DIMENSION: [0.5, 1.5],
            DIAMETER_DIMENSION: [0.5, 1.5],
            "velocity_bin_lower": (VELOCITY_DIMENSION, [0.0, 10.0]),
            "velocity_bin_upper": (VELOCITY_DIMENSION, [10.0, 20.0]),
            "diameter_bin_lower": (DIAMETER_DIMENSION, [0.0, 1.0]),
            "diameter_bin_upper": (DIAMETER_DIMENSION, [1.0, 2.0]),
        },
    )


def _segment_as_tuple(segment):
    """Convert a segment array to a rounded hashable tuple."""
    return tuple(map(tuple, np.round(np.asarray(segment), 6)))


class TestSpectrumBoundaryExtraction:
    """Unit tests for mask boundary extraction routines."""

    def test_get_mask_contour_returns_four_edges_for_single_true_bin(self):
        """Check contour extraction returns the four boundary edges for one active bin."""
        mask = create_test_mask([[True, False], [False, False]])
        segments = get_mask_contour(mask)
        expected = [
            np.array([[0.0, 0.0], [0.0, 10.0]]),
            np.array([[1.0, 0.0], [1.0, 10.0]]),
            np.array([[0.0, 0.0], [1.0, 0.0]]),
            np.array([[0.0, 10.0], [1.0, 10.0]]),
        ]
        assert len(segments) == 4
        for segment, exp in zip(segments, expected, strict=False):
            np.testing.assert_allclose(segment, exp)

    def test_get_mask_contour_drops_internal_edges_for_adjacent_bins(self):
        """Check contour extraction removes shared edges between neighboring true bins."""
        mask = create_test_mask([[True, True], [False, False]])
        segments = get_mask_contour(mask)
        expected = [
            np.array([[0.0, 0.0], [0.0, 10.0]]),
            np.array([[0.0, 0.0], [1.0, 0.0]]),
            np.array([[0.0, 10.0], [1.0, 10.0]]),
            np.array([[2.0, 0.0], [2.0, 10.0]]),
            np.array([[1.0, 0.0], [2.0, 0.0]]),
            np.array([[1.0, 10.0], [2.0, 10.0]]),
        ]
        internal_edge = np.array([[1.0, 0.0], [1.0, 10.0]])
        assert len(segments) == 6
        for segment, exp in zip(segments, expected, strict=False):
            np.testing.assert_allclose(segment, exp)
        assert not any(np.allclose(segment, internal_edge) for segment in segments)

    def test_get_spectrum_mask_boundary_returns_outer_domain_edges_for_full_mask(self):
        """Check full rain-mask case returns only the outer spectrum boundary segments."""
        ds = create_template_dataset(with_velocity=True)
        segments = get_spectrum_mask_boundary(ds)
        tuple_segments = {_segment_as_tuple(segment) for segment in segments}
        assert len(segments) == 14
        assert ((0.1, 0.0), (0.1, 0.4)) in tuple_segments
        assert ((0.9, 0.6), (0.9, 1.4)) in tuple_segments
        assert ((0.3, 0.0), (0.5, 0.0)) in tuple_segments
        assert ((0.5, 1.4), (0.7, 1.4)) in tuple_segments

    def test_get_spectrum_mask_boundary_forwards_filter_options(self, monkeypatch):
        """Check boundary retrieval forwards velocity filtering arguments to mask creation."""
        ds = create_template_dataset(with_velocity=True)
        records = {}
        sentinel_boundary = [np.array([[0.0, 0.0], [1.0, 1.0]])]

        def fake_load_env_dataset(ds_input):
            records["env_input"] = ds_input
            return xr.Dataset({"dummy_env": xr.DataArray(1)})

        def fake_get_rain_fall_velocity(diameter, model, ds_env):
            records.setdefault("diameter_calls", []).append(diameter.name)
            records.setdefault("models", []).append(model)
            assert "dummy_env" in ds_env
            return xr.ones_like(diameter)

        def fake_define_rain_spectrum_mask(drop_number, **kwargs):
            records["drop_number_name"] = drop_number.name
            records["mask_kwargs"] = kwargs
            return create_test_mask([[True, False], [False, False]])

        def fake_get_mask_contour(mask, x_dim, y_dim):
            records["contour_dims"] = (x_dim, y_dim)
            records["contour_mask_dims"] = mask.dims
            return sentinel_boundary

        monkeypatch.setattr(l2_processing, "load_env_dataset", fake_load_env_dataset)
        monkeypatch.setattr(l2_processing, "get_rain_fall_velocity", fake_get_rain_fall_velocity)
        monkeypatch.setattr(l2_processing, "define_rain_spectrum_mask", fake_define_rain_spectrum_mask)
        monkeypatch.setattr(l2_processing, "get_mask_contour", fake_get_mask_contour)

        output = get_spectrum_mask_boundary(
            ds,
            above_velocity_fraction=0.25,
            below_velocity_tolerance=0.5,
            maintain_drops_smaller_than=0.8,
            maintain_drops_slower_than=1.2,
            maintain_smallest_drops=True,
            fall_velocity_model="Brandes2002",
        )

        assert output == sentinel_boundary
        assert records["env_input"] is ds
        assert records["diameter_calls"] == ["diameter_bin_upper", "diameter_bin_lower"]
        assert records["models"] == ["Brandes2002", "Brandes2002"]
        assert records["drop_number_name"] == "raw_drop_number"
        assert records["mask_kwargs"]["above_velocity_fraction"] == 0.25
        assert records["mask_kwargs"]["below_velocity_tolerance"] == 0.5
        assert records["mask_kwargs"]["maintain_drops_smaller_than"] == 0.8
        assert records["mask_kwargs"]["maintain_drops_slower_than"] == 1.2
        assert records["mask_kwargs"]["maintain_smallest_drops"] is True
        assert records["contour_dims"] == (DIAMETER_DIMENSION, VELOCITY_DIMENSION)
        assert records["contour_mask_dims"] == (VELOCITY_DIMENSION, DIAMETER_DIMENSION)


class TestDefineRainDropSpectrumMask:
    """Unit tests for diameter/velocity filters and spectrum mask definition."""

    def test_define_spectrum_mask_with_fraction(self):
        """Test spectrum mask creation with velocity fraction thresholds."""
        ds = create_test_dataset()
        fall_velocity = xr.DataArray([3.0, 6.0, 12.0, 20.0], dims=[DIAMETER_DIMENSION])
        ds_mask = define_rain_spectrum_mask(
            ds["drop_number"],
            fall_velocity_lower=fall_velocity,
            fall_velocity_upper=fall_velocity,
            above_velocity_fraction=0.2,
            below_velocity_fraction=0.2,
        )
        assert isinstance(ds_mask, xr.DataArray)
        assert ds_mask.dtype == bool
        assert ds_mask.shape == ds["drop_number"].shape
        expected_mask = np.array(
            [
                [False, True, False, False],
                [False, True, True, False],
                [False, False, True, True],
                [False, False, False, False],
            ],
        )
        np.testing.assert_allclose(ds_mask.to_numpy(), expected_mask)

    def test_define_spectrum_mask_with_tolerance(self):
        """Test spectrum mask creation with velocity tolerance thresholds."""
        ds = create_test_dataset()
        fall_velocity = xr.DataArray([3.0, 6.0, 12.0, 20.0], dims=[DIAMETER_DIMENSION])
        ds_mask = define_rain_spectrum_mask(
            ds["drop_number"],
            fall_velocity_lower=fall_velocity,
            fall_velocity_upper=fall_velocity,
            above_velocity_tolerance=1.0,
            below_velocity_tolerance=1.0,
        )
        assert isinstance(ds_mask, xr.DataArray)
        assert ds_mask.dtype == bool
        assert ds_mask.shape == ds["drop_number"].shape
        expected_mask = np.array(
            [
                [False, True, False, False],
                [False, False, True, False],
                [False, False, False, True],
                [False, False, False, False],
            ],
        )
        np.testing.assert_allclose(ds_mask.to_numpy(), expected_mask)

    def test_define_spectrum_without_arguments(self):
        """Test spectrum mask without arguments returns True array."""
        ds = create_test_dataset()
        fall_velocity = xr.DataArray([3.0, 6.0, 12.0, 20.0], dims=[DIAMETER_DIMENSION])
        ds_mask = define_rain_spectrum_mask(
            ds["drop_number"],
            fall_velocity_lower=fall_velocity,
            fall_velocity_upper=fall_velocity,
        )
        assert isinstance(ds_mask, xr.DataArray)
        assert ds_mask.dtype == bool
        assert ds_mask.shape == ds["drop_number"].shape
        assert ds_mask.to_numpy().all()  # all True

    def test_define_rain_spectrum_mask_conflicting_args(self):
        """Test spectrum mask raises error if both fraction and tolerance are given."""
        ds = create_test_dataset()
        fall_velocity = xr.DataArray([3.0, 6.0, 12.0, 20.0], dims=[DIAMETER_DIMENSION])

        with pytest.raises(ValueError):
            define_rain_spectrum_mask(
                ds["drop_number"],
                fall_velocity_lower=fall_velocity,
                fall_velocity_upper=fall_velocity,
                above_velocity_fraction=0.1,
                above_velocity_tolerance=1.0,
            )

        with pytest.raises(ValueError):
            define_rain_spectrum_mask(
                ds["drop_number"],
                fall_velocity_lower=fall_velocity,
                fall_velocity_upper=fall_velocity,
                below_velocity_fraction=0.1,
                below_velocity_tolerance=1.0,
            )

    def test_define_rain_spectrum_mask_keep_smallest(self):
        """Test spectrum mask retains smallest drops when maintain_smallest_drops=True."""
        ds = create_test_dataset()
        fall_velocity = xr.DataArray([3.0, 6.0, 12.0, 20.0], dims=[DIAMETER_DIMENSION])
        ds_mask = define_rain_spectrum_mask(
            ds["drop_number"],
            fall_velocity_lower=fall_velocity,
            fall_velocity_upper=fall_velocity,
            above_velocity_fraction=0.1,
            below_velocity_fraction=0.1,
            maintain_smallest_drops=True,
            maintain_drops_smaller_than=1.5,
            maintain_drops_slower_than=3.0,
        )
        assert isinstance(ds_mask, xr.DataArray)
        assert ds_mask.dtype == bool
        assert ds_mask.shape == ds["drop_number"].shape
        assert ds_mask.to_numpy()[0, 0]  # because keeps drops with D<1.5 and V<3
        expected_mask = np.array(
            [
                [True, True, False, False],
                [False, False, True, False],
                [False, False, False, True],
                [False, False, False, False],
            ],
        )
        np.testing.assert_allclose(ds_mask.to_numpy(), expected_mask)

    def test_define_rain_spectrum_mask_with_fixed_fall_velocity(self):
        """Test spectrum mask works with fall_velocity not varying with time."""
        ds = create_test_dataset()
        ds = ds.expand_dims({"time": 3})

        drop_number = ds["drop_number"]

        fall_velocity = xr.DataArray([3.0, 6.0, 12.0, 20.0], dims=[DIAMETER_DIMENSION])

        ds_mask = define_rain_spectrum_mask(
            drop_number,
            fall_velocity_lower=fall_velocity,
            fall_velocity_upper=fall_velocity,
            above_velocity_fraction=0.2,
            below_velocity_fraction=0.2,
        )
        assert isinstance(ds_mask, xr.DataArray)
        assert "time" not in ds_mask.dims

    def test_define_rain_spectrum_mask_with_fall_velocity_varying_with_time(self):
        """Test spectrum mask works with fall_velocity varying with time."""
        ds = create_test_dataset()
        ds = ds.expand_dims({"time": 3})

        drop_number = ds["drop_number"]

        fall_velocity = xr.DataArray([3.0, 6.0, 12.0, 20.0], dims=[DIAMETER_DIMENSION])
        fall_velocity = fall_velocity.expand_dims({"time": 3})

        ds_mask = define_rain_spectrum_mask(
            drop_number,
            fall_velocity_lower=fall_velocity,
            fall_velocity_upper=fall_velocity,
            above_velocity_fraction=0.2,
            below_velocity_fraction=0.2,
        )
        assert isinstance(ds_mask, xr.DataArray)
        assert "time" in ds_mask.dims
        assert ds_mask.sizes["time"] == 3

    def test_define_rain_spectrum_mask_with_dask_array(self):
        """Test spectrum mask works correctly with xarray DataArray backed by dask array."""
        ds = create_test_dataset()
        ds = ds.expand_dims({"time": 3})
        fall_velocity = xr.DataArray([3.0, 6.0, 12.0, 20.0], dims=[DIAMETER_DIMENSION])
        fall_velocity = fall_velocity.expand_dims({"time": 3})

        drop_number = ds["drop_number"].chunk({"time": 1})
        fall_velocity = fall_velocity.chunk({"time": 1})

        ds_mask = define_rain_spectrum_mask(
            drop_number,
            fall_velocity_lower=fall_velocity,
            fall_velocity_upper=fall_velocity,
            above_velocity_fraction=0.2,
            below_velocity_fraction=0.2,
            maintain_smallest_drops=True,
            maintain_drops_smaller_than=1,  # 1.0 upper inclusive !
            maintain_drops_slower_than=2.0,  # 2.0 upper inclusive !
        )

        assert isinstance(ds_mask, xr.DataArray)
        assert hasattr(ds_mask, "chunks")

        expected_mask = np.array(
            [
                [True, True, False, False],
                [False, True, True, False],
                [False, False, True, True],
                [False, False, False, False],
            ],
        )
        np.testing.assert_allclose(ds_mask.isel(time=0).compute().to_numpy(), expected_mask)


class TestCheckL2eInputDataset:
    """Test suite for the check_l2e_input_dataset function."""

    def test_valid_dataset(self):
        """Dataset from template should pass validation without errors."""
        ds = create_template_dataset()
        assert isinstance(check_l2e_input_dataset(ds), xr.Dataset)

    def test_missing_variables(self):
        """Missing a required variable should raise ValueError with appropriate message."""
        ds = create_template_dataset()
        ds = ds.drop_vars(["raw_drop_number"])
        with pytest.raises(ValueError, match=r"variables.*raw_drop_number"):
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
        ds = ds.rename_dims({DIAMETER_DIMENSION: "dummy_dim"})
        with pytest.raises(ValueError, match=r"dimensions.*"):
            check_l2e_input_dataset(ds)

    def test_missing_attribute(self):
        """Missing a required attribute should raise ValueError with appropriate message."""
        ds = create_template_dataset()
        ds.attrs.pop("sensor_name", None)
        with pytest.raises(ValueError, match=r"attributes.*sensor_name"):
            check_l2e_input_dataset(ds)

    @pytest.mark.parametrize(
        "dimension",
        ["source", "velocity_method", RADAR_OPTIONS[0]],
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
    """Test the generate_l2e product function."""

    @pytest.mark.parametrize("with_velocity", [True, False])
    def test_returns_expected_variables(self, with_velocity):
        """Test generate_l1 returns Dataset with mandatory variables."""
        ds = create_template_dataset(with_velocity=with_velocity)
        ds_l2 = generate_l2e(ds, fall_velocity_model="Brandes2002")
        expected_vars = [
            # L1
            "qc_time",
            "qc_resampling",
            "raw_drop_number",
            # ...
            # L2E
            "fall_velocity",
            "drop_number",
            "drop_counts",
            "Dmin",
            "Dmax",
            "N",
            "Nraw",
            "Nremoved",
            "Nbins",
            "Nbins_missing",
            "Nbins_missing_fraction",
            "Nbins_missing_consecutive",
            "drop_number_concentration",
            "Rm",
            "Pm",
            "Nt",
            "R",
            "P",
            "M0",
            "M1",
            "M2",
            "M3",
            "M4",
            "M5",
            "M6",
            "LWC",
            "Z",
            "Dmode",
            "Dm",
            "sigma_m",
            "Nw",
            "D10",
            "D50",
            "D90",
            "TKE",
            "KEF",
            "KED",
        ]
        for var in expected_vars:
            assert var in ds_l2, f"Missing variable {var}"

    @pytest.mark.parametrize("model", available_rain_fall_velocity_models())
    def test_fall_velocity_is_computed(self, model):
        """Test all fall velocity models produce non-negative fall velocities."""
        ds = create_template_dataset()
        ds_out = generate_l2e(ds, fall_velocity_model=model)
        assert "fall_velocity" in ds_out
        assert np.all(ds_out["fall_velocity"] >= 0)

    def test_diameter_filtering(self):
        """Test diameter filtering."""
        ds = create_template_dataset()
        ds.attrs["sensor_name"] = "LPM"

        # If not PARSIVEL, return the same size
        ds_full = generate_l2e(ds, minimum_diameter=0)  # 0-10 diameter default, 0-12 velocity default
        assert ds_full.sizes["diameter_bin_center"] == ds.sizes["diameter_bin_center"]
        assert ds_full.sizes["velocity_bin_center"] == ds.sizes["velocity_bin_center"]

        # If PARSIVEL, remove first two bins and include only from D>0.2495 ...
        # - In this example, first diameter_bin_upper is 0.3 so it does not discard anything
        ds.attrs["sensor_name"] = "PARSIVEL2"
        ds_full = generate_l2e(ds, minimum_diameter=0)

        assert ds_full.sizes[DIAMETER_DIMENSION] == 4
        assert np.all(ds_full["diameter_bin_upper"] > 0.2495)

        ds_filtered = generate_l2e(ds, minimum_diameter=0.4, maximum_diameter=0.7)  # include 0.7
        assert ds_filtered.sizes[DIAMETER_DIMENSION] == 2
        np.testing.assert_allclose(ds_filtered["diameter_bin_upper"].to_numpy(), np.array([0.5, 0.7]))

        ds_filtered = generate_l2e(ds, minimum_diameter=0.4, maximum_diameter=0.8)  # search in next bin
        assert ds_filtered.sizes[DIAMETER_DIMENSION] == 3
        np.testing.assert_allclose(ds_filtered["diameter_bin_upper"].to_numpy(), np.array([0.5, 0.7, 0.9]))

        ds_filtered = generate_l2e(ds, minimum_diameter=0.3, maximum_diameter=0.8)  # include 0.3
        assert ds_filtered.sizes[DIAMETER_DIMENSION] == 3
        np.testing.assert_allclose(ds_filtered["diameter_bin_upper"].to_numpy(), np.array([0.5, 0.7, 0.9]))
        np.testing.assert_allclose(ds_filtered["diameter_bin_lower"].to_numpy(), np.array([0.3, 0.5, 0.7]))

        # Test filtering everything raise error !
        with pytest.raises(ValueError):
            generate_l2e(ds, minimum_diameter=1000)

    def test_velocity_filtering(self):
        """Test velocity filtering."""
        ds = create_template_dataset()

        ds_full = generate_l2e(ds)  # 0-10 diameter default, 0-12 velocity default
        assert ds_full.sizes[VELOCITY_DIMENSION] == ds.sizes[VELOCITY_DIMENSION]

        ds_filtered = generate_l2e(ds, minimum_velocity=0, maximum_velocity=0.5)
        assert ds_filtered.sizes[VELOCITY_DIMENSION] == 2

        ds_filtered = generate_l2e(ds, minimum_velocity=0, maximum_velocity=0.6)  # <= velocity_bin_upper
        assert ds_filtered.sizes[VELOCITY_DIMENSION] == 2

        ds_filtered = generate_l2e(ds, minimum_velocity=0, maximum_velocity=0.7)  # include also next bin
        assert ds_filtered.sizes[VELOCITY_DIMENSION] == 3

        # Test filtering everything raise error !
        with pytest.raises(ValueError):
            generate_l2e(ds, minimum_velocity=1000)

    def test_dask_allowed(self):
        """Test lazy dataset as input is allowed."""
        ds_lazy = create_template_dataset().chunk({"time": 1})
        ds_l2_lazy = generate_l2e(ds_lazy)
        for var in ds_l2_lazy.data_vars:
            if var in ds_l2_lazy and var != "fall_velocity":
                assert hasattr(ds_l2_lazy[var].data, "chunks")
        xr.testing.assert_allclose(ds_l2_lazy.compute(), generate_l2e(create_template_dataset()))

    def test_preserves_global_attributes(self):
        """Global input attributes are preserved in output L1 dataset."""
        ds = create_template_dataset()
        ds.attrs["dummy"] = "dummy_value"
        assert ds.attrs["sensor_name"] == "PARSIVEL2"

        ds_out = generate_l2e(ds)
        assert "dummy" in ds_out.attrs
        assert ds_out.attrs["sensor_name"] == "PARSIVEL2"

    def test_spectrum_with_velocity_dimension(self):
        """Spectrum dataset with velocity dimension should include new dims."""
        ds = create_template_dataset(with_velocity=True)
        ds_out = generate_l2e(ds)
        assert "velocity_method" in ds_out.dims
        assert "source" in ds_out.dims
        assert isinstance(ds_out, xr.Dataset)

    def test_spectrum_without_velocity_dimension(self):
        """Spectrum dataset without velocity dim should not add velocity dims."""
        ds = create_template_dataset(with_velocity=False)
        ds_out = generate_l2e(ds)
        assert "velocity_method" not in ds_out.dims
        assert "source" not in ds_out.dims
        assert isinstance(ds_out, xr.Dataset)

    def test_without_time_dimension(self):
        """Dataset without time dimension should still process correctly."""
        ds = create_template_dataset(with_velocity=True)
        ds = ds.isel(time=0)
        ds_out = generate_l2e(ds)
        assert isinstance(ds_out, xr.Dataset)

    def test_additional_dimension_preserved(self):
        """Additional dimensions should be preserved in output dims."""
        ds = create_template_dataset(with_velocity=True)
        ds["raw_drop_number"] = ds["raw_drop_number"].expand_dims({"year": [2012, 2013]})
        ds_out = generate_l2e(ds)
        assert "year" in ds_out.dims

    def test_with_lazy_dask_array(self):
        """Test it correctly deals with dask arrays."""
        ds = create_template_dataset(with_velocity=True)
        ds_lazy = ds.chunk({"time": 1})
        ds_out = generate_l2e(ds_lazy)
        # Check returns dask array
        assert isinstance(ds_out, xr.Dataset)
        assert hasattr(ds_out["Dmode"].data, "chunks")
        assert hasattr(ds_out["D50"].data, "chunks")
        # Test it can compute without error
        ds_out = ds_out.compute()
        assert isinstance(ds_out, xr.Dataset)
        # Test equaliy with in-memory computing
        ds_out1 = generate_l2e(ds)
        xr.testing.assert_allclose(ds_out, ds_out1)

    def test_idempotent_l2e_generation(self):
        """Regenerating L2E with L2E dataset should produce identical results."""
        ds = create_template_dataset(with_velocity=True)
        ds_out = generate_l2e(ds)
        ds_out2 = generate_l2e(ds_out)
        xr.testing.assert_allclose(ds_out, ds_out2)


class TestGenerateL2Model:
    """Test the generate_l2e product function."""

    @pytest.mark.parametrize(
        "psd_model",
        disdrodb.psd.available_psd_models(),
    )
    def test_with_in_memory_numpy_array(self, psd_model):
        """Test L2M product generation with in-memory numpy data."""
        ds = create_template_l2e_dataset()
        if psd_model == "NormalizedGeneralizedGammaPSD":
            optimization_settings = {"fixed_parameters": {"i": 3, "j": 4}}
        else:
            optimization_settings = None
        ds_out = generate_l2m(ds, psd_model=psd_model, optimization_settings=optimization_settings)
        assert isinstance(ds_out, xr.Dataset)

    @pytest.mark.parametrize(
        "psd_model",
        disdrodb.psd.available_psd_models(),
    )
    def test_with_lazy_dask_array(self, psd_model):
        """Test L2M product generation with lazy dask array data."""
        ds = create_template_l2e_dataset()
        ds_lazy = ds.chunk({"time": 1})

        if psd_model == "NormalizedGeneralizedGammaPSD":
            optimization_settings = {"fixed_parameters": {"i": 3, "j": 4}}
        else:
            optimization_settings = None

        ds_out = generate_l2m(ds_lazy, psd_model=psd_model, optimization_settings=optimization_settings)

        # Test it returns dask arrays
        assert isinstance(ds_out, xr.Dataset)
        assert hasattr(ds_out["R"].data, "chunks")
        assert hasattr(ds_out["KLDiv"].data, "chunks")

        # Test it can compute without error
        with suppress_warnings():
            ds_out = ds_out.compute()
        assert isinstance(ds_out, xr.Dataset)

        # Test equaliy with in-memory computing
        ds_out1 = generate_l2m(ds, psd_model=psd_model, optimization_settings=optimization_settings)
        xr.testing.assert_allclose(ds_out, ds_out1)

    def test_without_time_dimension(self):
        """Dataset without time dimension should still process correctly."""
        ds = create_template_l2e_dataset()
        ds = ds.isel(time=0)
        ds_out = generate_l2m(ds, psd_model="NormalizedGammaPSD")
        assert isinstance(ds_out, xr.Dataset)

    def test_additional_dimension_preserved(self):
        """Additional dimensions should be preserved in output dims."""
        ds = create_template_l2e_dataset()
        ds["drop_number_concentration"] = ds["drop_number_concentration"].expand_dims({"year": [2012, 2013]})
        ds_out = generate_l2m(ds, psd_model="NormalizedGammaPSD")
        assert "year" in ds_out.dims
        assert "year" in ds_out["mu"].dims

    def test_idempotent_generation(self):
        """Regenerating L2M with L2M dataset should produce identical results."""
        ds = create_template_l2e_dataset()
        ds_out = generate_l2m(ds, psd_model="NormalizedGammaPSD")
        ds_out2 = generate_l2m(ds_out, psd_model="NormalizedGammaPSD")
        xr.testing.assert_allclose(ds_out, ds_out2)

    def test_NormalizedGammaPSD_fitting(self):
        """Test Normalized Gamma PSD fitting."""
        ds = create_template_l2e_dataset()

        ds_out = generate_l2m(ds, psd_model="NormalizedGammaPSD", optimization="GS")
        assert ds_out.attrs["disdrodb_psd_model"] == "NormalizedGammaPSD"
        assert "disdrodb_psd_optimization" in ds_out.attrs
        assert "disdrodb_psd_optimization_settings" in ds_out.attrs

        # Test raise error
        with pytest.raises(NotImplementedError, match="ML optimization is not available"):
            generate_l2m(ds, psd_model="NormalizedGammaPSD", optimization="ML")

        with pytest.raises(NotImplementedError, match="MOM optimization is not available"):
            generate_l2m(ds, psd_model="NormalizedGammaPSD", optimization="MOM")

    def test_GammaPSD_fitting(self):
        """Test Gamma PSD fitting."""
        ds = create_template_l2e_dataset()

        ds_out = generate_l2m(ds, psd_model="GammaPSD", optimization="GS")

        ds_out = generate_l2m(ds, psd_model="GammaPSD", optimization="ML")

        ds_out = generate_l2m(ds, psd_model="GammaPSD", optimization="MOM")
        assert "mom_method" in ds_out.dims
        assert ds_out.attrs["disdrodb_psd_model"] == "GammaPSD"
        assert "disdrodb_psd_optimization" in ds_out.attrs
        assert "disdrodb_psd_optimization_settings" in ds_out.attrs

    def test_LognormalPSD_fitting(self):
        """Test LognormalPSD fitting."""
        ds = create_template_l2e_dataset()

        ds_out = generate_l2m(ds, psd_model="LognormalPSD", optimization="GS", minimum_rain_rate=0)
        ds_out = generate_l2m(ds, psd_model="LognormalPSD", optimization="ML", minimum_rain_rate=0)
        ds_out = generate_l2m(ds, psd_model="LognormalPSD", optimization="MOM", minimum_rain_rate=0)

        assert ds_out.attrs["disdrodb_psd_model"] == "LognormalPSD"
        assert "disdrodb_psd_optimization" in ds_out.attrs
        assert "disdrodb_psd_optimization_settings" in ds_out.attrs

    def test_ExponentialPSD_fitting(self):
        """Test ExponentialPSD fitting."""
        ds = create_template_l2e_dataset()

        ds_out = generate_l2m(ds, psd_model="ExponentialPSD", optimization="GS")
        ds_out = generate_l2m(ds, psd_model="ExponentialPSD", optimization="ML")
        ds_out = generate_l2m(ds, psd_model="ExponentialPSD", optimization="MOM")

        assert ds_out.attrs["disdrodb_psd_model"] == "ExponentialPSD"
        assert "disdrodb_psd_optimization" in ds_out.attrs
        assert "disdrodb_psd_optimization_settings" in ds_out.attrs

    def test_fitting_without_init_method(self):
        """Test fitting without moment initialization."""
        ds = create_template_l2e_dataset()
        ds_out = generate_l2m(
            ds,
            psd_model="GammaPSD",
            optimization="ML",
            optimization_settings={"init_method": None},
        )
        ds_out = generate_l2m(
            ds,
            psd_model="GammaPSD",
            optimization="ML",
            optimization_settings={"init_method": "None"},
        )
        ds_out = generate_l2m(
            ds,
            psd_model="GammaPSD",
            optimization="ML",
            optimization_settings={"init_method": ["None"]},
        )
        assert isinstance(ds_out, xr.Dataset)

    def test_fitting_with_multiple_init_method(self):
        """Test fitting with multiple initialization methods."""
        ds = create_template_l2e_dataset()
        ds_out = generate_l2m(
            ds,
            psd_model="GammaPSD",
            optimization="ML",
            optimization_settings={"init_method": [None, "M234", "M346"]},
        )
        assert "init_method" in ds_out.dims
        assert ds_out.sizes["init_method"] == 3
