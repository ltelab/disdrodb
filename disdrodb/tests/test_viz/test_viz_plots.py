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
"""Test DISDRODB visualization functions."""

import pytest

from disdrodb.tests.fake_datasets import create_template_l2e_dataset


@pytest.fixture
def l2e_dataset():
    """Create a template L2E dataset for testing visualization functions."""
    return create_template_l2e_dataset()


class TestPlotDsdQuicklook:
    """Test plot_dsd_quicklook function."""

    def test_plot_dsd_quicklook_dataset(self, l2e_dataset):
        """Test plot_dsd_quicklook with xarray Dataset."""
        l2e_dataset.disdrodb.plot_dsd_quicklook()

    def test_plot_dsd_quicklook_dataarray(self, l2e_dataset):
        """Test plot_dsd_quicklook with xarray DataArray."""
        l2e_dataset["raw_drop_number"].disdrodb.plot_dsd_quicklook()


class TestPlotDsd:
    """Test plot_dsd function."""

    def test_plot_dsd_dataset_multiple_timesteps(self, l2e_dataset):
        """Test plot_dsd with Dataset containing multiple timesteps."""
        l2e_dataset.disdrodb.plot_dsd()

    def test_plot_dsd_dataarray_multiple_timesteps(self, l2e_dataset):
        """Test plot_dsd with DataArray containing multiple timesteps."""
        l2e_dataset["drop_number_concentration"].disdrodb.plot_dsd()

    def test_plot_dsd_dataset_single_timestep(self, l2e_dataset):
        """Test plot_dsd with Dataset containing a single timestep."""
        l2e_dataset.isel(time=0).disdrodb.plot_dsd()

    def test_plot_dsd_dataarray_single_timestep(self, l2e_dataset):
        """Test plot_dsd with DataArray containing a single timestep."""
        l2e_dataset["drop_number_concentration"].isel(time=0).disdrodb.plot_dsd()


class TestPlotSpectrum:
    """Test plot_spectrum function."""

    def test_plot_spectrum_dataset_multiple_timesteps(self, l2e_dataset):
        """Test plot_spectrum with Dataset containing multiple timesteps."""
        l2e_dataset.disdrodb.plot_spectrum()

    def test_plot_spectrum_dataarray_multiple_timesteps(self, l2e_dataset):
        """Test plot_spectrum with DataArray containing multiple timesteps."""
        l2e_dataset["raw_drop_number"].disdrodb.plot_spectrum()

    def test_plot_spectrum_dataset_single_timestep(self, l2e_dataset):
        """Test plot_spectrum with Dataset containing a single timestep."""
        l2e_dataset.isel(time=0).disdrodb.plot_spectrum()

    def test_plot_spectrum_dataarray_single_timestep(self, l2e_dataset):
        """Test plot_spectrum with DataArray containing a single timestep."""
        l2e_dataset["raw_drop_number"].isel(time=0).disdrodb.plot_spectrum()
