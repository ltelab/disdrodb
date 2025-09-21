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
# along with this progra  If not, see <http://www.gnu.org/licenses/>.
# -----------------------------------------------------------------------------.
"""Test L1 processing function."""
import numpy as np
import pytest
import xarray as xr

from disdrodb.constants import DIAMETER_DIMENSION, VELOCITY_DIMENSION
from disdrodb.l1.fall_velocity import available_raindrop_fall_velocity_models
from disdrodb.l1.processing import generate_l1
from disdrodb.tests.fake_datasets import create_template_l0c_dataset


class TestGenerateL1:
    """Test suite for generate_l1 core function."""

    @pytest.mark.parametrize("with_velocity", [True, False])
    def test_returns_expected_variables(self, with_velocity):
        """Test generate_l1 returns Dataset with mandatory variables."""
        ds = create_template_l0c_dataset(with_velocity=with_velocity)
        ds_l1 = generate_l1(ds, fall_velocity_model="Brandes2002")
        expected_vars = [
            "raw_drop_number",
            "fall_velocity",
            "drop_number",
            "drop_counts",
            "Dmin",
            "Dmax",
            "N",
            "Nraw",
            "Nremoved",
            "time_qc",
        ]
        for var in expected_vars:
            assert var in ds_l1, f"Missing variable {var}"

    @pytest.mark.parametrize("model", available_raindrop_fall_velocity_models())
    def test_fall_velocity_is_computed(self, model):
        """Test all fall velocity models produce non-negative fall velocities."""
        ds = create_template_l0c_dataset()
        ds_l1 = generate_l1(ds, fall_velocity_model=model)
        assert "fall_velocity" in ds_l1
        assert np.all(ds_l1["fall_velocity"] >= 0)

    def test_diameter_filtering(self):
        """Test diameter filtering."""
        ds = create_template_l0c_dataset()
        ds.attrs["sensor_name"] = "LPM"

        # If not PARSIVEL, return the same size
        ds_full = generate_l1(ds)  # 0-10 diameter default, 0-12 velocity default
        assert ds_full.sizes == ds.sizes

        # If PARSIVEL, remove first two bins and include only from D>0.2495 ...
        # - In this example, first diameter_bin_upper is 0.3 so it does not discard anything
        ds.attrs["sensor_name"] = "PARSIVEL2"
        ds_full = generate_l1(ds)

        assert ds_full.sizes[DIAMETER_DIMENSION] == 4
        assert np.all(ds_full["diameter_bin_upper"] > 0.2495)

        ds_filtered = generate_l1(ds, minimum_diameter=0.4, maximum_diameter=0.7)  # include 0.7
        assert ds_filtered.sizes[DIAMETER_DIMENSION] == 2
        np.testing.assert_allclose(ds_filtered["diameter_bin_upper"].to_numpy(), np.array([0.5, 0.7]))

        ds_filtered = generate_l1(ds, minimum_diameter=0.4, maximum_diameter=0.8)  # search in next bin
        assert ds_filtered.sizes[DIAMETER_DIMENSION] == 3
        np.testing.assert_allclose(ds_filtered["diameter_bin_upper"].to_numpy(), np.array([0.5, 0.7, 0.9]))

        ds_filtered = generate_l1(ds, minimum_diameter=0.3, maximum_diameter=0.8)  # include 0.3
        assert ds_filtered.sizes[DIAMETER_DIMENSION] == 3
        np.testing.assert_allclose(ds_filtered["diameter_bin_upper"].to_numpy(), np.array([0.5, 0.7, 0.9]))
        np.testing.assert_allclose(ds_filtered["diameter_bin_lower"].to_numpy(), np.array([0.3, 0.5, 0.7]))

        # Test filtering everything raise error !
        with pytest.raises(ValueError):
            generate_l1(ds, minimum_diameter=1000)

    def test_velocity_filtering(self):
        """Test velocity filtering."""
        ds = create_template_l0c_dataset()

        ds_full = generate_l1(ds)  # 0-10 diameter default, 0-12 velocity default
        assert ds_full.sizes[VELOCITY_DIMENSION] == ds.sizes[VELOCITY_DIMENSION]

        ds_filtered = generate_l1(ds, minimum_velocity=0, maximum_velocity=0.5)
        assert ds_filtered.sizes[VELOCITY_DIMENSION] == 2

        ds_filtered = generate_l1(ds, minimum_velocity=0, maximum_velocity=0.6)  # <= velocity_bin_upper
        assert ds_filtered.sizes[VELOCITY_DIMENSION] == 2

        ds_filtered = generate_l1(ds, minimum_velocity=0, maximum_velocity=0.7)  # include also next bin
        assert ds_filtered.sizes[VELOCITY_DIMENSION] == 3

        # Test filtering everything raise error !
        with pytest.raises(ValueError):
            generate_l1(ds, minimum_velocity=1000)

    def test_dask_allowed(self):
        """Test lazy dataset as input is allowed."""
        ds_lazy = create_template_l0c_dataset().chunk({"time": 1})
        ds_l1_lazy = generate_l1(ds_lazy)  # 0-10 dia
        for var in ds_l1_lazy.data_vars:
            if var not in "fall_velocity":  # diameter coordinate is in in memory ..
                assert hasattr(ds_l1_lazy[var].data, "chunks")
        xr.testing.assert_allclose(ds_l1_lazy.compute(), generate_l1(create_template_l0c_dataset()))

    def test_preserves_global_attributes(self):
        """Global input attributes are preserved in output L1 dataset."""
        ds = create_template_l0c_dataset()
        ds.attrs["dummy"] = "dummy_value"
        assert ds.attrs["sensor_name"] == "PARSIVEL2"

        ds_l1 = generate_l1(ds)
        assert "dummy" in ds_l1.attrs
        assert ds_l1.attrs["sensor_name"] == "PARSIVEL2"
