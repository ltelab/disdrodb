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
"""Plotting utilities for the Double Moment Normalization (DN) approach."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

from disdrodb.psd.models import (
    NormalizedGeneralizedGammaPSD,
)


def plot_loss_function(loss_function, q=(0, 0.8), s=15, cmap="Spectral_r", marker="x", c="black", **kwargs):
    """Plot loss function."""
    # List dimensions
    loss_function = loss_function.squeeze()
    dims = list(loss_function.dims)
    if len(dims) not in [1, 2]:
        raise ValueError("Can plot loss function only if 1D or 2D")

    # 1D loss function
    if len(dims) == 1:
        x = dims[0]
        best_index = np.nanargmin(loss_function)
        best_value = loss_function[x][best_index].item()
        p = loss_function.plot()
        plt.axvline(best_value, c=c, **kwargs)
        plt.xlabel(x.replace("_values", ""))
        plt.ylabel("Loss")
        plt.title("")
        return p[0].figure

    # 2D loss function
    x = dims[1]
    y = dims[0]
    best_index = loss_function.argmin(dim=(y, x))
    best_params = {dim: loss_function[dim].to_numpy()[best_index[dim].item()].item() for dim in [x, y]}

    # Plot loss function
    vmin, vmax = np.nanquantile(loss_function, q=q)
    norm = LogNorm(vmin=vmin, vmax=vmax)
    p = loss_function.plot.pcolormesh(norm=norm, cmap=cmap, cbar_kwargs={"label": "Loss"})
    plt.scatter(best_params[x], best_params[y], marker=marker, s=s, c=c, **kwargs)
    plt.xlabel(x.replace("_values", ""))
    plt.ylabel(y.replace("_values", ""))
    return p.figure


def plot_normalized_dsd_and_ngg_fits(
    ds_dsd_norm_stats,
    c,
    mu,
    i,
    j,
    x_norm,
    xlims,
    ylims,
    add_ng_fits=True,
    cmap="Spectral_r",
    norm=None,
    ax=None,
):
    """Plot normalized DSD and fitted NormalizedGeneralizedGammaPSD."""
    # Define colormap and norm
    cmap = plt.get_cmap(cmap)
    cmap.set_under(alpha=0)
    if norm is None:
        norm = LogNorm(1, None)

    # --------------------------------------------------------------------------.
    # Create figure
    if ax is None:
        fig, ax = plt.subplots()

    # --------------------------------------------------------------------------.
    # Display 2D density counts
    _ = ds_dsd_norm_stats["count"].plot.pcolormesh(
        x=f"D/Dc_{i}_{j}",
        y=f"N(D)/Nc_{i}_{j}",
        cmap=cmap,
        norm=norm,
        extend="max",
        yscale="log",
        cbar_kwargs={"label": "Counts [-]"},
        ax=ax,
    )
    ax.set_xlim(*xlims)
    ax.set_ylim(*ylims)
    ax.set_xlabel(rf"$D/D_{{c,{i},{j}}}$ [-]")
    ax.set_ylabel(rf"$N(D)/N_{{c,{i},{j}}}$ [-]")
    ax.set_title("Normalized DSD")

    # --------------------------------------------------------------------------.
    # Add fitted NormalizedGeneralizedGammaPSD
    model_prefix = "NG" if c == 1 and i == 3 and j == 4 else "NGG"
    nd_norm = NormalizedGeneralizedGammaPSD.normalized_formula(x=x_norm, i=i, j=j, c=c, mu=mu)
    ax.plot(
        x_norm,
        nd_norm,
        color="black",
        linewidth=2.0,
        label=rf"{model_prefix}$(i={i}, \, j={j}, \, \mu={mu:.1f}, \, c={c:.1f})$",
    )

    # --------------------------------------------------------------------------.
    # Add fitted NormalizedGamma with mu 1, 2, 3
    if add_ng_fits:
        nd_norm_mu1 = NormalizedGeneralizedGammaPSD.normalized_formula(x=x_norm, i=i, j=j, c=1, mu=1)
        nd_norm_mu2 = NormalizedGeneralizedGammaPSD.normalized_formula(x=x_norm, i=i, j=j, c=1, mu=2)
        nd_norm_mu3 = NormalizedGeneralizedGammaPSD.normalized_formula(x=x_norm, i=i, j=j, c=1, mu=3)
        ax.plot(
            x_norm,
            nd_norm_mu1,
            linestyle="--",
            color="gray",
            linewidth=1.5,
            label=rf"NG$(i={i},\,j={j},\,\mu=1.0,\,c=1.0)$",
        )

        ax.plot(
            x_norm,
            nd_norm_mu2,
            linestyle="dotted",
            color="gray",
            linewidth=1.5,
            label=rf"NG$(i={i},\,j={j},\,\mu=2.0,\,c=1.0)$",
        )

        ax.plot(
            x_norm,
            nd_norm_mu3,
            linestyle="-.",
            color="gray",
            linewidth=1.5,
            label=rf"NG$(i={i},\,j={j},\,\mu=3.0,\,c=1.0)$",
        )

    # --------------------------------------------------------------------------.
    # Add legend
    ax.legend(
        loc="upper right",
        frameon=False,
        fontsize=8,
        handlelength=2.5,
    )

    # --------------------------------------------------------------------------.
    # Add axis labels
    ax.set_xlabel(rf"$D/D_{{c,{i},{j}}}$")
    ax.set_ylabel(rf"$N(D)/N_{{c,{i},{j}}}$")

    return ax.figure
