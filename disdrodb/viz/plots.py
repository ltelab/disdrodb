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
"""DISDRODB Plotting Tools."""
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm


def plot_nd(ds, var="drop_number_concentration", cmap=None, norm=None):
    """Plot drop number concentration N(D) timeseries."""
    # Check inputs
    if var not in ds:
        raise ValueError(f"{var} is not a xarray Dataset variable!")
    # Check only time and diameter dimensions are specified
    # TODO: DIAMETER_DIMENSION, "time"

    # Select N(D)
    ds_var = ds[[var]].compute()

    # Regularize input
    ds_var = ds_var.disdrodb.regularize()

    # Set 0 values to np.nan
    ds_var = ds_var.where(ds_var[var] > 0)

    # Define cmap an norm
    if cmap is None:
        cmap = plt.get_cmap("Spectral_r").copy()
    vmin = ds_var[var].min().item()
    norm = LogNorm(vmin, None) if norm is None else norm

    # Plot N(D)
    p = ds_var[var].plot.pcolormesh(x="time", norm=norm, cmap=cmap)
    return p
