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


def plot_profiles_base(
    full_table,
    m_bins,
    colors,
    prof_col="ds_t_pc2",
    prof_col_err="ds_err_pc2",
    ylabel=r"$\Delta \Sigma$ [M$_\odot$pc$^{-2}$]",
):
    fig, axes = plt.subplots(3, 2, sharex=True)

    for i, ax in enumerate(axes.flatten()):
        ax.errorbar(
            full_table["rp"][i],
            full_table[prof_col][i],
            full_table[prof_col_err][i],
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

        ax.axvline(full_table.emass["R500_crit"][i], color=colors[i], ls="--")
        ax.axvline(full_table.emass["R200_crit"][i], color=colors[i], ls=":")

    for ax in axes[-1]:
        ax.set_xlabel("R [Mpc]")
        # ax.set_xticks(ax.get_xticks())
        # ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
        # ax.tick_params(axis='x', labelrotation=45)

    axes[1, 0].set_ylabel(ylabel)

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


######################
### For mcmc plots ###
######################


def quant_err(x):
    return np.array(
        [
            [np.mean(x) - np.quantile(x, 0.16)],
            [np.quantile(x, 0.84) - np.mean(x)],
        ]
    )


def quant_err_vec(x):
    return np.array(
        [
            np.mean(x, axis=-1) - np.quantile(x, 0.16, axis=-1),
            np.quantile(x, 0.84, axis=-1) - np.mean(x, axis=-1),
        ]
    )


def get_chain_mass_and_err(_fit, burnin=2000):
    return (
        10 ** _fit[0, burnin:].mean(),
        10 ** _fit[0, burnin:].mean() * np.log(10) * quant_err(_fit[0, burnin:]),
    )


def get_chain_mass_and_err_vec(_fit, burnin=2000, metric=np.median):
    mass = 10 ** metric(_fit[:, 0, burnin:], axis=-1)
    return (
        mass,
        (mass * np.log(10) * quant_err_vec(_fit[:, 0, burnin:])).T,
    )


def add_mc_fit(ax, fit, burnin, colors, **kwargs):
    _kwargs = {**kwargs}
    for i, (m, merr, c, cerr) in enumerate(
        zip(
            *get_chain_mass_and_err_vec(fit, burnin=burnin),
            fit[:, 1, burnin:].mean(axis=-1),
            quant_err_vec(fit[:, 1, burnin:]).T,
        )
    ):
        ax.errorbar(
            m,
            c,
            cerr[:, None],
            merr[:, None],
            color=colors[i],
            markeredgecolor=colors[i],
            **_kwargs,
        )
        _kwargs["label"] = None


def add_profile_mcmc(axes, full_table, mcmc_fit, burnin, func, colors, **kwargs):
    for i, ax in enumerate(axes.flatten()):
        ax.plot(
            full_table["rp"][i],
            func(
                full_table["rp"][i],
                10 ** mcmc_fit[i, 0, burnin:].mean(),
                mcmc_fit[i, 1, burnin:].mean(),
                full_table["z_l"][i][0],
            ),
            color=colors[i],
            **kwargs,
        )


def add_profile_mcmc_err(axes, full_table, mcmc_fit, burnin, func, colors, **kwargs):
    for i, ax in enumerate(axes.flatten()):
        vals = func(
            full_table["rp"][i][:, None],
            10 ** mcmc_fit[i, 0, burnin:][None, :],
            mcmc_fit[i, 1, burnin:][None, :],
            full_table["z_l"][i][0],
        )
        ax.plot(
            full_table["rp"][i],
            vals.mean(axis=-1),
            color=colors[i],
            **kwargs,
        )
        ax.fill_between(
            full_table["rp"][i],
            np.quantile(vals, 0.16, axis=-1),
            np.quantile(vals, 0.84, axis=-1),
            color=colors[i],
            lw=0,
            alpha=0.3,
        )
