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
"""This module defines DISDRODB xarray accessors."""
import xarray as xr


class DISDRODB_Base_Accessor:
    """DISDRODB Xarray Base Accessor."""

    def __init__(self, xarray_obj):
        if not isinstance(xarray_obj, (xr.DataArray, xr.Dataset)):
            raise TypeError("The 'gpm' accessor is available only for xarray.Dataset and xarray.DataArray.")
        self._obj = xarray_obj

    @property
    def start_time(self):
        """Return start time."""
        from disdrodb.api.checks import check_time

        if "time" in self._obj.coords:
            start_time = self._obj["time"].to_numpy().min()
        else:
            raise ValueError("Time coordinate not found")
        return check_time(start_time)

    @property
    def end_time(self):
        """Return end time."""
        from disdrodb.api.checks import check_time

        if "time" in self._obj.coords:
            end_time = self._obj["time"].to_numpy().max()
        else:
            raise ValueError("Time coordinate not found")
        return check_time(end_time)

    @property
    def sample_interval(self):
        """Return the sample interval in seconds."""
        from disdrodb.utils.time import ensure_sample_interval_in_seconds

        if "sample_interval" not in self._obj.coords:
            raise ValueError("The sample interval is not specified in the xarray object.")
        return int(ensure_sample_interval_in_seconds(self._obj["sample_interval"].to_numpy()))

    @property
    def diameter_bin_edges(self):
        """Return diameter bin edges."""
        from disdrodb.utils.manipulations import get_diameter_bin_edges

        return get_diameter_bin_edges(self._obj)

    @property
    def velocity_bin_edges(self):
        """Return velocity bin edges."""
        from disdrodb.utils.manipulations import get_velocity_bin_edges

        return get_velocity_bin_edges(self._obj)

    def regularize(self):
        """Regularize timesteps."""
        from disdrodb.utils.time import regularize_dataset

        sample_interval = self._obj.disdrodb.sample_interval
        return regularize_dataset(self._obj, freq=f"{sample_interval}s")

    def isel(self, indexers=None, drop=False, **indexers_kwargs):
        """Perform index-based dimension selection."""
        from disdrodb.utils.subsetting import isel

        return isel(self._obj, indexers=indexers, drop=drop, **indexers_kwargs)

    def sel(self, indexers=None, drop=False, method=None, **indexers_kwargs):
        """Perform value-based coordinate selection."""
        from disdrodb.utils.subsetting import sel

        return sel(self._obj, indexers=indexers, drop=drop, method=method, **indexers_kwargs)

    def align(self, *args):
        """Align DISDRODB products over time, velocity and diameter dimensions."""
        from disdrodb.utils.subsetting import align

        return align(self._obj, *args)

    def plot_spectrum(self, **kwargs):
        """Plot spectrum."""
        from disdrodb.viz.plots import plot_spectrum

        return plot_spectrum(self._obj, **kwargs)

    def plot_nd(self, **kwargs):
        """Plot drop number concentration N(D) timeseries."""
        from disdrodb.viz.plots import plot_nd

        return plot_nd(self._obj, **kwargs)


@xr.register_dataset_accessor("disdrodb")
class DISDRODB_Dataset_Accessor(DISDRODB_Base_Accessor):
    """DISDRODB Xarray Dataset Accessor."""

    def __init__(self, xarray_obj):
        super().__init__(xarray_obj)

    def resample(self, temporal_resolution):
        """Resample a L1 or L2 DISDRODB Product."""
        from disdrodb.l1.resampling import resample_dataset

        sample_interval = self._obj.disdrodb.sample_interval
        ds = resample_dataset(
            self._obj,
            sample_interval=sample_interval,
            temporal_resolution=temporal_resolution,
        )
        return ds

    def plot_raw_and_filtered_spectra(self, **kwargs):
        """Plot the raw and filtered spectra."""
        from disdrodb.viz.plots import plot_raw_and_filtered_spectra

        return plot_raw_and_filtered_spectra(self._obj, **kwargs)


@xr.register_dataarray_accessor("disdrodb")
class DISDRODB_DataArray_Accessor(DISDRODB_Base_Accessor):
    """DISDRODB Xarray DataArray Accessor."""

    def __init__(self, xarray_obj):
        super().__init__(xarray_obj)
