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
"""DISDRODB Metadata Plotting Tools."""

import cartopy.feature as cfeature
import matplotlib.pyplot as plt
from cartopy import crs as ccrs

import disdrodb


def add_map_background(ax):
    """Add background to Cartopy.Axes."""
    ax.add_feature(cfeature.LAND, facecolor="0.95", zorder=0)
    ax.add_feature(cfeature.OCEAN, facecolor="white", zorder=0)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.3)
    ax.add_feature(cfeature.BORDERS, linewidth=0.3, linestyle=":")


def plot_stations(df=None, marker_size=10, c="dodgerblue", ax=None, crs_proj=None, figsize=(7, 4.5), dpi=300):
    """Plot stations of DISDRODB Metadata Archive.

    df = disdrodb.read_metadata_archive()
    """
    # Load metadata archive dataframe if not specified
    if df is None:
        df = disdrodb.read_metadata_archive()

    # Initialize map
    if ax is None:
        crs_proj = ccrs.PlateCarree() if crs_proj is None else crs_proj
        fig, ax = plt.subplots(
            subplot_kw={"projection": crs_proj},
            figsize=figsize,
            dpi=dpi,
        )
        add_map_background(ax)

    # Plot stations
    ax.scatter(
        x=df["longitude"],
        y=df["latitude"],
        c=c,
        edgecolor="none",
        marker="o",
        s=marker_size,
        alpha=1,
        transform=ccrs.PlateCarree(),
    )
    return ax.figure
