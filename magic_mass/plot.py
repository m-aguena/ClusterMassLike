import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

import matplotlib.colors as mcolors

import clmm


def truncate_colormap(cmap_name, min_val=0.0, max_val=1.0, n=100):
    cmap = plt.get_cmap(cmap_name)
    # Extract colors from the specific range (0.0 to 1.0)
    colors = cmap(np.linspace(min_val, max_val, n))
    # Build and return the new truncated colormap
    return mcolors.LinearSegmentedColormap.from_list(f"trunc_{cmap_name}", colors)


####################################
### For mass-concentration plots ###
####################################


def plot_cm_base(m_bins, delta=None):
    fig, ax = plt.subplots(layout="constrained")

    ax.set_xscale("log")
    ax.set_yscale("log")

    subs = r"{\rm crit}"
    if delta is not None:
        subs = rf"{{{delta}_{subs}}}"

    ax.set_xlabel(rf"M$_{subs}$ [M$_\odot$]")
    ax.set_ylabel(rf"$c_{subs}$")

    sliced_cmap = truncate_colormap(
        "magma", min_val=0.0, max_val=((len(m_bins) - 1) / len(m_bins))
    )
    cmap = plt.get_cmap("magma")
    norm = mpl.colors.BoundaryNorm(m_bins, cmap.N)
    colorizer = mpl.colorizer.Colorizer(norm=norm, cmap=sliced_cmap)

    fig.colorbar(
        mpl.colorizer.ColorizingArtist(colorizer),
        ax=ax,
        orientation="vertical",
        label=r"log$_{10}$(M$_{\rm lum}$/M$_\odot$)",
    )

    return fig, ax


def plt_errorbars(ax, x, y, xerr, yerr, colors, **kwargs):
    _kwargs = {**kwargs}
    for i in range(6):
        ax.errorbar(
            x[i],
            y[i],
            xerr[i],
            yerr[i],
            color=colors[i],
            markeredgecolor=colors[i],
            **_kwargs,
        )
        _kwargs["label"] = None


#########################
### For profile plots ###
#########################


def plot_profiles_base(full_table, m_bins, colors):
    fig, axes = plt.subplots(3, 2, sharex=True)

    for i, ax in enumerate(axes.flatten()):
        ax.errorbar(
            full_table["enclosed_mass_radius"][i],
            full_table["ds_t_pc2"][i],
            full_table["ds_err_pc2"][i],
            ls="",
            marker=".",
            markersize=7,
            markeredgewidth=0.0,
            color=colors[i],
            lw=1,
            label=rf"$[{m_bins[i]:.1f}:{m_bins[i + 1]:.1f}]$",
        )

        ax.legend()

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.tick_params(axis="both", which="both", direction="in")

        ax.axvline(full_table["R500_crit"][i], color=colors[i], ls="--")
        ax.axvline(full_table["R200_crit"][i], color=colors[i], ls=":")

    for ax in axes[-1]:
        ax.set_xlabel("R [Mpc]")
        # ax.set_xticks(ax.get_xticks())
        # ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
        # ax.tick_params(axis='x', labelrotation=45)

    axes[1, 0].set_ylabel(r"$\Delta \Sigma$ [M$_\odot$pc$^{-2}$]")

    plt.subplots_adjust(hspace=0, wspace=0.2)

    return fig, axes


def add_profile_analytic(
    full_table, colors, axes, cosmo, delta=500, ds_pred_samples=None, **kwargs
):
    for i, ax in enumerate(axes.flatten()):
        ax.plot(
            full_table["enclosed_mass_radius"][i],
            clmm.compute_excess_surface_density(
                full_table["enclosed_mass_radius"][i],
                full_table[f"M{delta}_crit"][i],
                full_table[f"c{delta}_crit"][i],
                full_table["z_l"][i][0],
                # full_table["z_s"][i],
                cosmo,
                delta_mdef=delta,
                halo_profile_model="nfw",
                massdef="critical",
            ),
            color=colors[i],
            **kwargs,
        )
        if ds_pred_samples is not None:
            ax.fill_between(
                full_table["enclosed_mass_radius"][i],
                np.quantile(ds_pred_samples[i], 0.16, axis=0),
                np.quantile(ds_pred_samples[i], 0.84, axis=0),
                color=colors[i],
                alpha=0.3,
                lw=0,
            )


def add_profile(full_table, colors, axes, concentration_col, func, **kwargs):
    for i, ax in enumerate(axes.flatten()):
        ax.plot(
            full_table["enclosed_mass_radius"][i],
            func(
                full_table["enclosed_mass_radius"][i],
                full_table["M500_crit"][i],
                full_table[concentration_col][i],
                full_table["z_l"][i][0],
                500,
            ),
            color=colors[i],
            **kwargs,
        )


def add_profile_fixm(
    full_table, colors, axes, concentration_col, ds_pred_samples, func, **kwargs
):
    add_profile(full_table, colors, axes, concentration_col, func=func, **kwargs)

    if ds_pred_samples is None:
        return

    for i, ax in enumerate(axes.flatten()):
        ax.fill_between(
            full_table["enclosed_mass_radius"][i],
            np.quantile(ds_pred_samples[i], 0.16, axis=0),
            np.quantile(ds_pred_samples[i], 0.84, axis=0),
            color=colors[i],
            alpha=0.3,
            lw=0,
        )
