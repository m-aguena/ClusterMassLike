#!/usr/bin/env python
# coding: utf-8

# # Fit Enclosed mass and concentration

# In[1]:


# from dsigma.helpers import dsigma_table
import astropy.units as u
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
from astropy.constants import G
from astropy.cosmology import Planck15
from astropy.table import Table, dstack, vstack


# In[758]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
import sys
sys.path.append("/pbs/home/m/maguena/git_codes/ClusterMassLike/")


# In[ ]:





# In[2]:


# %pip install --upgrade astropy


# In[2]:


# %pip install --upgrade pandas --no-cache-dir


# In[2]:


import pandas


# In[3]:


m_bins = np.array([12.2, 12.5, 12.8, 13.1, 13.4, 13.7, 14.7])


# In[4]:


import matplotlib.colors as mcolors


# 1. Define a helper function to slice the colormap


# In[627]:


def plot_cm_base(delta=None):
    fig, ax = plt.subplots(layout="constrained")

    ax.set_xscale("log")
    ax.set_yscale("log")

    subs = r"{\rm crit}"
    if delta is not None:

        subs = rf"{{{delta}_{subs}}}"

    ax.set_xlabel(rf"M$_{subs}$ [M$_\odot$]")
    ax.set_ylabel(rf"$c_{subs}$")

    fig.colorbar(
        mpl.colorizer.ColorizingArtist(colorizer),
        ax=ax,
        orientation="vertical",
        label=r"log$_{10}$(M$_{\rm lum}$/M$_\odot$)",
    )

    return fig, ax


# In[636]:


def plt_errorbars(ax, x, y, xerr, yerr, **kwargs):
    _kwargs = {**kwargs}
    for i in range(6):
        ax.errorbar(
            x[i],
            y[i],
            xerr[i],
            yerr[i],
            color=colors[i],
            markeredgecolor=colors[i],
            **_kwargs
        )
        _kwargs["label"] = None


# ## Functions for profiles

# In[61]:


import clmm
from clmm.cosmology.ccl import CCLCosmology


# In[570]:


Planck15


# In[62]:


cosmo = CCLCosmology(
    H0=Planck15.H0.value,
    Omega_dm0=Planck15.Om0 - Planck15.Ob0,
    Omega_b0=Planck15.Ob0,
    Omega_k0=Planck15.Ok0,
)


# In[141]:


def nz_normal(*args):
    val = np.random.normal(*args)
    while val < 0:
        val = np.random.normal(*args)
    return val


# In[604]:


def plot_profiles_base():

    fig, axes = plt.subplots(3, 2, sharex=True)

    for i, ax in enumerate(axes.flatten()):
        ax.errorbar(
            mm.radius,
            full_table["ds_t_pc2"][i],
            full_table["ds_err_pc2"][i],
            ls="",
            marker=".",
            markersize=7,
            markeredgewidth=0.0,
            color=colors[i],
            lw=1,
            label=rf"$[{m_bins[i]:.1f}:{m_bins[i+1]:.1f}]$",
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


# ### Basic plots

# In[2]:


lenspath = "/sps/euclid/Users/maguena/working/fgas/data/weak lensing/cluster_coverfraction.fits"
table_l = Table.read(lenspath)
# m_bins = np.array([12.2, 13.1, 13.4, 13.7, 14.0, 14.3, 15])

bins = np.digitize(table_l["mass_lum"], m_bins)
pass


# In[8]:


footprint = np.where((table_l["cf_1mpc"] > 0.5) & (bins > 0))

plt.scatter(
    table_l["IterCenZ"][footprint],
    table_l["mass_lum"][footprint],
    c=bins[footprint],
    cmap=sliced_cmap,
    alpha=0.4,
)

plt.plot()
plt.tick_params(axis="both", direction="in")

plt.xlabel("z")
plt.ylabel(r"log$_{10}$ (M /M$_\odot$)")


# In[9]:


def radius_from_mass(logM, z, overdensity_factor):
    """radius from mass
    computes r_x in Mpc from log10(m_x) for a given critical overdensity factor x
    Parameters
    ----------
    logM: numpy.ndarray
        mass in solar masses
    z: numpy.ndarray
        redshift
    overdensity_factor: int
        overdensity factor

    """
    mass = 10 ** (logM)
    rhoc = Planck15.critical_density(0.2).to(u.solMass / u.Mpc / u.Mpc / u.Mpc).value
    r = (3.0 * mass / (4.0 * np.pi * overdensity_factor * rhoc)) ** (0.33333)
    return r


# In[6]:


# get average mass in each mass bin
m_med = np.array(
    [
        np.median(table_l["mass_lum"][bins == mass_bin + 1])
        for mass_bin in np.arange(len(m_bins) - 1)
    ]
)
print(m_med)
z_med = np.array(
    [
        np.median(table_l["IterCenZ"][bins == mass_bin + 1])
        for mass_bin in np.arange(len(m_bins) - 1)
    ]
)
print(z_med)
r_med = radius_from_mass(m_med, z_med, 200)
print(r_med)
m_mean = np.array(
    [
        np.average(table_l["mass_lum"][bins == mass_bin + 1])
        for mass_bin in np.arange(len(m_bins) - 1)
    ]
)
print(m_mean)
z_mean = np.array(
    [
        np.average(table_l["IterCenZ"][bins == mass_bin + 1])
        for mass_bin in np.arange(len(m_bins) - 1)
    ]
)
print(z_med)
r_mean = radius_from_mass(m_mean, z_mean, 200)
print(r_mean)


# ## Read data

# * here is the list of arrays that we want and their shapes
# (14, 15)["stacked_Dsigma"]
# (14, 15)["stacked_theta"]
# (14, 15)["stacked_variance"]
# (400, 14, 15)["bs_theta"]
# (400, 14, 15)["bs_Dsigma"]
# ['bs_Dsigma', 'bs_theta', 'stacked_Dsigma', 'stacked_theta', 'stacked_variance']

# In[ ]:


# load boostrap table
table_list = []
for i in np.arange(len(m_bins) - 1):
    table_tmp = Table.read(
        f"/sps/lsst/users/lbaumont/software/dsigma_tmp/dsigma/tutorial/bs_hsc_{i}.parquet",
        format="parquet",
    )
    table_tmp["mass_bin"] = i
    table_list.append(table_tmp)
bs_table = dstack(table_list)
# convert from Mpc**2 to pc**2 units
bs_DS = bs_table["ds_t"].T * 10**12
print(bs_DS.shape)


# In[8]:


def read_ds_data_T(data_path_template, **kwargs):
    table_list = []
    for i in np.arange(len(m_bins) - 1):
        table_tmp = Table.read(data_path_template % i, **kwargs)
        table_tmp["mass_bin"] = i
        table_list.append(table_tmp)
    out = dstack(table_list)

    # convert from Mpc**2 to pc**2 units
    out["ds_t_pc2"] = out["ds_t"] * 10**12
    if "ds_err" in out.colnames:
        out["ds_err_pc2"] = out["ds_err"] * 10**12

    return out


# In[6]:


def read_ds_data(data_path_template, **kwargs):
    table_dict = {}
    for i in np.arange(len(m_bins) - 1):
        print(data_path_template % i)
        table_tmp = Table.read(data_path_template % i, **kwargs)
        table_tmp["mass_bin"] = i
        for col in table_tmp.colnames:
            table_dict[col] = table_dict.get(col, []) + [table_tmp[col]]
    # out = Table(list(table_dict.values()), names=list(table_dict.keys()))
    out = Table(table_dict)

    # convert from Mpc**2 to pc**2 units
    out["ds_t_pc2"] = out["ds_t"] * 10**12
    if "ds_err" in out.colnames:
        out["ds_err_pc2"] = out["ds_err"] * 10**12

    return out


# In[7]:


# load boostrap table
bs_table = read_ds_data(
    "/sps/lsst/users/lbaumont/software/dsigma_tmp/dsigma/tutorial/bs_hsc_%d.parquet",
    format="parquet",
)


# In[8]:


get_ipython().run_cell_magic('time', '', '# load regular table\nfull_table = read_ds_data(\n    "/sps/lsst/users/lbaumont/software/dsigma_tmp/dsigma/tutorial/boost_hsc_%d.csv"\n)\nfull_table["ds_t"].T.shape\n')


# ### Plots

# In[19]:


plt.figure()
# Get a magma color for each scatter plot (0–1, one per dataset)
fig, ax = plt.subplots(layout="constrained")

cmap = plt.get_cmap("magma")
norm = mpl.colors.BoundaryNorm(m_bins, cmap.N)

# norm.vmax=14.7


colorizer = mpl.colorizer.Colorizer(norm=norm, cmap=sliced_cmap)
# colorizer.set_clim(4.7)
fig.colorbar(
    mpl.colorizer.ColorizingArtist(colorizer),
    ax=ax,
    orientation="vertical",
    label=r"log$_{10}$(M$_{lum}$/M$\odot$)",
)

colors = [cmap(i / 6) for i in range(7)]  # 6 colors: 0/5, 1/5, ..., 5/5

for i, (theta, DS, errs) in enumerate(
    zip(full_table["rp"], full_table["ds_t_pc2"], full_table["ds_err_pc2"])
):
    color = colors[i % 6]
    ax.errorbar(theta, DS, yerr=errs, color=color, fmt="*")
    ax.set_xscale("log")
    ax.set_yscale("log")
plt.ylim(10**10, 10**15)
plt.xlabel("R [Mpc]")
plt.ylabel(r"$\Delta \Sigma$ [M$_\odot$pc$^{-2}$]")
plt.tick_params(axis="both", which="both", direction="in")
# cmap = cm.get_cmap('magma')(hdulist[1].data[class_key]/num_cl)
pass
# plt.savefig("profiles.pdf", bbox_inches="tight")


# In[82]:


plt.figure()
# Get a magma color for each scatter plot (0–1, one per dataset)
fig, ax = plt.subplots(layout="constrained")

cmap = plt.get_cmap("magma")
norm = mpl.colors.BoundaryNorm(m_bins, cmap.N)

# norm.vmax=14.7


colorizer = mpl.colorizer.Colorizer(norm=norm, cmap=sliced_cmap)
# colorizer.set_clim(4.7)
fig.colorbar(
    mpl.colorizer.ColorizingArtist(colorizer),
    ax=ax,
    orientation="vertical",
    label=r"log$_{10}$(M$_{lum}$/M$\odot$)",
)

colors = [cmap(i / 6) for i in range(7)]  # 6 colors: 0/5, 1/5, ..., 5/5

for i, (theta, DS, errs) in enumerate(
    zip(full_table["rp"].T, full_table["ds_t_pc2"].T, full_table["ds_err_pc2"].T)
):
    color = colors[i % 6]
    ax.errorbar(theta, DS, yerr=errs, color=color, fmt="*")
    ax.set_xscale("log")
    ax.set_yscale("log")
plt.ylim(10**10, 10**15)
plt.xlabel("R [Mpc]")
plt.ylabel(r"$\Delta \Sigma$ [M$_\odot$pc$^{-2}$]")
plt.tick_params(axis="both", which="both", direction="in")
# cmap = cm.get_cmap('magma')(hdulist[1].data[class_key]/num_cl)
pass
# plt.savefig("profiles.pdf", bbox_inches="tight")


# ### Plots new

# In[605]:


plot_profiles_base()


# ## Measure mass

# In[527]:


class MagicMass:
    def __init__(self, radius=None):

        self.phi = np.linspace(0.00001, np.pi / 2, num=200)

        # is this radius definition okay? What are the units?
        self.radius = np.logspace(-1, 0.5, num=10)

        self.dphi = np.diff(self.phi)[0]
        self.rsinphi = self.radius[:, None] / np.sin(self.phi[None, :])

    def measure_gt_mass(self, theta, DS, plot=True):

        if plot:
            plt.plot(self.phi, 1 / np.sin(self.phi), "*")
            plt.show()

        # get delta_sigma values at r=r/sin(phi)
        integrand = np.array(
            [
                np.interp(np.log(self.rsinphi), np.log(theta[i]), DS[i])
                for i in range(DS.shape[0])
            ]
        )

        # rsinphi.max()
        # integrand.shape

        if plot:
            plt.pcolor(np.log10(integrand[-6]))
            plt.colorbar()
            plt.show()

        integral = 4 * np.sum(integrand, axis=-1) * self.dphi

        g = integral * u.solMass / u.Mpc**2
        g = g.to(u.kg / u.m**2) * G

        M = self.r**2 * integral

        return g, M


mm = MagicMass()


# In[10]:


full_table["magic_mass"] = mm.measure_gt_mass(
    theta=full_table["rp"],  # Mpc
    # change units of ds from M_sol*Mpc**2 to M_sol*pc**2
    DS=full_table["ds_t_pc2"],
    # errs = full_table["ds_err_pc2"].T
)[1]


# In[11]:


bs_table["magic_mass"] = np.array(
    [
        mm.measure_gt_mass(theta, DS, plot=False)[1]
        for theta, DS in zip(
            bs_table["rp"].transpose(2, 0, 1), bs_table["ds_t_pc2"].transpose(2, 0, 1)
        )
    ]
).transpose(1, 2, 0)


# In[12]:


bs_table["gt"] = np.array(
    [
        mm.measure_gt_mass(theta, DS, plot=False)[0]
        for theta, DS in zip(
            bs_table["rp"].transpose(2, 0, 1), bs_table["ds_t_pc2"].transpose(2, 0, 1)
        )
    ]
).transpose(1, 2, 0)


# In[13]:


full_table["magic_mass_bs_mean"] = np.nanmean(bs_table["magic_mass"], axis=-1)
full_table["magic_mass_bs_std"] = np.nanstd(bs_table["magic_mass"], axis=-1)


# In[14]:


full_table["gt_bs_mean"] = np.nanmean(bs_table["gt"], axis=-1)
full_table["gt_bs_std"] = np.nanstd(bs_table["gt"], axis=-1)


# ### Plots

# In[417]:


fig, ax = plt.subplots(layout="constrained")


norm = mpl.colors.BoundaryNorm(m_bins, cmap.N)

colorizer = mpl.colorizer.Colorizer(norm=norm, cmap=sliced_cmap)
offset = 0.003
fig.colorbar(
    mpl.colorizer.ColorizingArtist(colorizer),
    ax=ax,
    orientation="vertical",
    label=r"log$_{10}$(M$_{lum}$/M$\odot$)",
)
for i, (mean_m_bs, err_m_bs) in enumerate(
    zip(
        full_table["magic_mass_bs_mean"],
        full_table["magic_mass_bs_std"],
    )
):
    color = colors[i % len(m_bins)]
    ax.errorbar(
        mm.radius * np.power(10, offset + i * shift),
        mean_m_bs,
        yerr=err_m_bs,
        color=color,
        marker="*",
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    # ax.scatter(r_med[i], 10**m_med[i], color=color)

    # plt.plot(r_med[i]/1000, m_med[i]*1e13, "+", color=cmap[i], markersize=20)
plt.xlabel("r [Mpc]")
plt.ylabel(r"Enclosed Lensing Mass [M$\odot$]")
plt.ylim(10**10, 10**15)
plt.tick_params(axis="both", which="both", direction="in")
plt.savefig("figs/enclosed_mass.png", bbox_inches="tight")


# In[ ]:


fig, ax = plt.subplots(layout="constrained")


norm = mpl.colors.BoundaryNorm(m_bins, cmap.N)

colorizer = mpl.colorizer.Colorizer(norm=norm, cmap=sliced_cmap)

fig.colorbar(
    mpl.colorizer.ColorizingArtist(colorizer),
    ax=ax,
    orientation="vertical",
    label=r"log$_{10}$(M$_{lum}$/M$\odot$)",
)

for i in range(theta.shape[0]):
    color = colors[i % len(m_bins)]
    ax.errorbar(
        r * np.power(10, offset + i * shift),
        mean_m_bs[i],
        yerr=err_m_bs[i],
        color=color,
        marker="*",
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    # ax.scatter(r_med[i], 10**m_med[i], color=color)

    # plt.plot(r_med[i]/1000, m_med[i]*1e13, "+", color=cmap[i], markersize=20)
plt.xlabel("r [Mpc]")
plt.ylabel(r"Enclosed Lensing Mass [M$\odot$]")
plt.ylim(10**10, 10**15)
plt.tick_params(axis="both", which="both", direction="in")
plt.savefig("enclosed_mass.pdf", bbox_inches="tight")


# In[ ]:


plt.figure()
for i in range(theta.shape[0]):
    color = colors[i % len(m_bins)]
    plt.errorbar(
        r * np.power(10, offset + i * shift),
        mean_m_bs[i],
        yerr=err_m_bs[i],
        color=color,
        marker="*",
    )
    plt.xscale("log")
    plt.yscale("log")
    # plt.plot(r_med[i]/1000, m_med[i]*1e13, "+", color=cmap[i], markersize=20)
plt.xlabel("r [Mpc]")
plt.ylabel("M [Msol]")
plt.ylim(10**10, 10**15)


# In[18]:


for i in np.arange(8):
    mass_bin = full_table["mass_bin"] == i
    color = colors[i % len(m_bins)]
    plt.errorbar(
        full_table["rp"][mass_bin],
        full_table["ds_t"][mass_bin],
        full_table["ds_err"][mass_bin],
        color=color,
        label=f"mass_bin {i+1}",
    )
    # plt.errorbar(full_table['rp'][mass_bin],full_table['ds_raw'][mass_bin]-full_table['ds'][mass_bin],full_table['ds_err'][mass_bin],label=f'mass_bin {i+1}')
    # plt.legend()
    plt.xlabel("r [mpc]")
    plt.ylabel(r"$\Delta\Sigma_t$")
    plt.xscale("log")


# In[19]:


for i in np.arange(len(m_bins) - 1):
    mass_bin_edge1 = m_bins[i]
    mass_bin_edge2 = m_bins[i + 1]
    mass_bin = full_table["mass_bin"] == 2
    color = colors[i % len(m_bins)]
    plt.errorbar(
        full_table["rp"][mass_bin],
        full_table["ds_x"][mass_bin],
        full_table["ds_x_err"][mass_bin],
        color=color,
        label=f"mass [{mass_bin_edge1},{mass_bin_edge2}]",
    )
    # plt.errorbar(full_table['rp'][mass_bin],full_table['ds_raw'][mass_bin]-full_table['ds'][mass_bin],full_table['ds_err'][mass_bin],label=f'mass_bin {i+1}')
    plt.legend()
    plt.xlabel("r [mpc]")
    plt.ylabel(r"$\Delta\Sigma_x$")
    plt.xscale("log")


# ## Compute M & R $\Delta$
# 
# Then $R_\Delta$ can be obtained by interpolating $\Delta(R)$, and $M_\Delta$ can also be estimated:

# ### Compute $\Delta(R)$
# 
# Using the encompassed masses, estimate $\Delta$ for each radius with:
# 
# $$
# \Delta(R) = \frac{M(R)}{\frac{4\pi}{3} R^3 \rho_{\rm bkg}(z)}
# $$

# In[22]:


def get_delta(mass, radius, rho_bkg):
    return mass / (4 * np.pi / 3 * radius**3 * rho_bkg)


# In[573]:


bs_table["Delatcrit_bkg"] = np.array(
    [
        get_delta(mass, mm.radius, rho_bkg).value
        for mass, rho_bkg in zip(
            bs_table["magic_mass"].transpose(2, 0, 1),
            Planck15.critical_density(bs_table["z_l"])
            .to(u.solMass / u.Mpc**3)
            .transpose(2, 0, 1),
        )
    ]
).transpose(1, 2, 0)


# In[35]:


# Delta critital bkg density
full_table["Delatcrit_bkg"] = get_delta(
    full_table["magic_mass_bs_mean"],
    mm.radius,
    Planck15.critical_density(full_table["z_l"]).to(u.solMass / u.Mpc**3),
).value


# ### Compute M & R
# 
# Then $R_\Delta$ can be obtained by interpolating $\Delta(R)$, and $M_\Delta$ can also be estimated:

# #### Main functions

# In[24]:


from scipy.interpolate import interp1d


# In[117]:


def get_delta_quantity_np(delta, delta_vals, quantity_vals, log=False):
    mask = quantity_vals > 0
    if log:
        return 10 ** np.interp(delta, delta_vals[mask], np.log10(quantity_vals[mask]))
    return np.interp(delta, delta_vals[mask], quantity_vals[mask])


# In[25]:


def get_delta_quantity(delta, delta_vals, quantity_vals, log=False):
    mask = quantity_vals > 0
    if log:
        return 10 ** interp1d(
            delta_vals[mask], np.log10(quantity_vals[mask]), bounds_error=False
        )(delta)
    return interp1d(delta_vals[mask], quantity_vals[mask], bounds_error=False)(delta)


# In[528]:


def add_mrdelta_to_bs_table(bs_table, Delta):
    bs_table[f"R{Delta}_crit"] = np.array(
        [
            [
                get_delta_quantity(Delta, _Delatcrit_bkg, mm.radius, log=True)
                for _Delatcrit_bkg, _vals in zip(Delatcrit_bkg.T, vals.T)
            ]
            for Delatcrit_bkg in bs_table["Delatcrit_bkg"]
        ]
    )
    bs_table[f"M{Delta}_crit"] = np.array(
        [
            [
                get_delta_quantity(Delta, _Delatcrit_bkg, _vals, log=True)
                for _Delatcrit_bkg, _vals in zip(Delatcrit_bkg.T, vals.T)
            ]
            for Delatcrit_bkg, vals in zip(
                bs_table["Delatcrit_bkg"], bs_table["magic_mass"]
            )
        ]
    )
    bs_table[f"M{Delta}_crit"].info.format = "%.4e"


# In[410]:


def add_mrdelta_to_full_table(full_table, Delta):
    full_table[f"R{Delta}_crit"] = [
        get_delta_quantity(Delta, Delatcrit_bkg, mm.radius, log=True)
        for Delatcrit_bkg in full_table["Delatcrit_bkg"]
    ]
    full_table[f"M{Delta}_crit"] = [
        get_delta_quantity(Delta, Delatcrit_bkg, mass_vals, log=True)
        for Delatcrit_bkg, mass_vals in zip(
            full_table["Delatcrit_bkg"], full_table["magic_mass_bs_mean"]
        )
    ]
    full_table[f"M{Delta}_crit"].info.format = "%.4e"


# In[379]:


# Not in use
def get_mr_err(delta, delta_vals, quantity_vals, quantity_errs):

    mask = np.where(quantity_vals > 0)[0][::-1]
    delta_filt = delta_vals[mask]
    errs_filt = quantity_errs[mask]

    print(quantity_vals[mask])
    plt.plot(quantity_vals[mask])
    plt.show()
    plt.plot(delta_filt)
    plt.yscale("log")
    print(delta_filt)

    i = np.digitize(delta, delta_filt) - 1
    ratio = (delta - delta_filt[i]) / (delta_filt[i + 1] - delta_filt[i])

    return np.sqrt(
        ratio**2 * errs_filt[i + 1] ** 2 + (1 - ratio) ** 2 * errs_filt[i] ** 2
    )


# In[411]:


def add_mrdelta_err_to_full_table(full_table, Delta):

    full_table[f"M{Delta}_crit_err"] = [
        get_delta_quantity(Delta, Delatcrit_bkg, mass_vals, log=False)
        for Delatcrit_bkg, mass_vals in zip(
            full_table["Delatcrit_bkg"], full_table["magic_mass_bs_std"]
        )
    ]
    full_table[f"M{Delta}_crit_err"].info.format = "%.4e"


# #### Compute

# In[26]:


get_ipython().run_cell_magic('time', '', 'for Delta in (500, 200):\n    add_mrdelta_to_bs_table(bs_table, Delta)\n')


# In[36]:


for Delta in (500, 200):
    add_mrdelta_to_full_table(full_table, Delta)


# In[ ]:


for Delta in (500, 200):
    add_mrdelta_err_to_full_table(full_table, Delta)


# In[351]:


i = np.nanargmax(bs_table["Rs_chi2"][0])
print(i)
bs_table["Rs_chi2"][0, i], bs_table["Rs"][0, i], bs_table["R500_crit"][0, i], bs_table[
    "c500_crit"
][0, i]


# In[412]:


add_mrdelta_to_full_table(full_table, 2500)
add_mrdelta_err_to_full_table(full_table, 2500)


# ## Get concentration analytically
# 
# To compute the concentration, it is easier to first compute the NFW profile parameter $R_s$.
# It can be obtainerd with the ration of masses in two know values of radii ($R_{200}$ and $R_{500}$ for instance), and solve the equation:
# 
# $$
# \frac{1}{M_{200}}\left[\log\left(\frac{R_{200}}{R_s}+1\right)-\frac{R_{200}}{R_{200}+R_s}\right]=
# \frac{1}{M_{500}}\left[\log\left(\frac{R_{500}}{R_s}+1\right)-\frac{R_{500}}{R_{500}+R_s}\right]
# $$
# 
# 
# Then, the concentration follows as 
# 
# $$c_\Delta = R_\Delta/R_s$$
# 
# 
# A by product of this computasion, is that we can also estimate the normalization of the NFW profile $\rho_0$.
# It can then be computed with any $\Delta$:
# 
# $$
# \rho_0 = \frac{M_{200}}{4\pi R_s^3} \left[\log\left(\frac{R_{200}}{R_s}+1\right)-\frac{R_{200}}{R_{200}+R_s}\right]^{-1}
# $$
# 
# 
# 

# ### Main functions

# In[27]:


from scipy.optimize import fsolve, least_squares


# In[28]:


def nfw_int(x):
    return np.log(1 + x) - (x / (1 + x))


def get_rho0_rs_v0(M1, R1, M2, R2, Rs0=20):

    def func(Rs):
        # if Rs<=0:
        #    return np.nan
        return abs(nfw_int(R1 / Rs) / nfw_int(R2 / Rs) - M1 / M2)

    Rs = fsolve(func, x0=Rs0)

    rho0 = M1 / (4 * np.pi * Rs**3) / nfw_int(R1 / Rs)

    # rho0_2 = M2/(4 * np.pi * Rs**3)/nfw_int(R2/Rs)
    # print(rho0_2/rho0-1)

    print("c1, c2 :", R1 / Rs, R2 / Rs)
    print("diff   :", nfw_int(R1 / Rs) / nfw_int(R2 / Rs) - M1 / M2)

    return Rs, rho0


def get_rho0_rs(M1, R1, M2, R2, Rs0=20):

    # print(M1, R1, M2, R2)

    res = least_squares(
        lambda Rs: np.sqrt((nfw_int(R1 / Rs) / nfw_int(R2 / Rs) - M1 / M2) ** 2),
        x0=Rs0,
        bounds=(0.001, 20),
    )
    # print(res)
    Rs = res.x

    rho0 = M1 / (4 * np.pi * Rs**3) / nfw_int(R1 / Rs)

    # rho0_2 = M2/(4 * np.pi * Rs**3)/nfw_int(R2/Rs)
    # print(rho0_2/rho0-1)

    # print("c1, c2 :", R1 / Rs, R2 / Rs)
    # print("diff   :", nfw_int(R1 / Rs) / nfw_int(R2 / Rs) - M1 / M2)

    return Rs, rho0


def get_rho0_rs_safe(M1, R1, M2, R2, Rs0=20):

    mask = ~(np.isnan(M1) + np.isnan(R1) + np.isnan(M2) + np.isnan(R2))

    Rs = np.full(mask.size, np.nan)
    rho0 = np.full(mask.size, np.nan)

    Rs[mask], rho0[mask] = get_rho0_rs(
        M1[mask], R1[mask], M2[mask], R2[mask], np.array(Rs0)[mask]
    )

    return Rs, rho0


# In[445]:


def add_conc_rs_rho0_to_bs_table(bs_table, Delta1, Delta2):
    bs_table["Rs"], bs_table["rho0"] = np.array(
        [
            get_rho0_rs_safe(m200, r200, m500, r500, Rs0=20 * np.ones(m200.size))
            for m200, r200, m500, r500 in zip(
                np.array(bs_table[f"M{Delta1}_crit"]).T,
                np.array(bs_table[f"R{Delta1}_crit"]).T,
                np.array(bs_table[f"M{Delta2}_crit"]).T,
                np.array(bs_table[f"R{Delta2}_crit"]).T,
            )
        ]
    ).transpose(1, 2, 0)

    # bs_table[f"c{Delta2}_crit"] = bs_table[f"R{Delta2}_crit"] / bs_table["Rs"]
    # bs_table[f"c{Delta1}_crit"] = bs_table[f"R{Delta1}_crit"] / bs_table["Rs"]

    bs_table[f"c{Delta2}_crit"] = bs_table[f"R{Delta2}_crit"] / np.nanmean(
        bs_table["Rs"], axis=0
    )
    bs_table[f"c{Delta1}_crit"] = bs_table[f"R{Delta1}_crit"] / np.nanmean(
        bs_table["Rs"], axis=0
    )

    bs_table["Rs_chi2"] = (
        nfw_int(bs_table[f"R{Delta1}_crit"] / bs_table["Rs"])
        / nfw_int(bs_table[f"R{Delta2}_crit"] / bs_table["Rs"])
        - bs_table[f"M{Delta1}_crit"] / bs_table[f"M{Delta2}_crit"]
    ) ** 2


# In[446]:


def add_conc_rs_rho0_to_full_table(full_table, Delta1, Delta2):

    full_table["Rs"], full_table["rho0"] = get_rho0_rs(
        np.array(full_table[f"M{Delta1}_crit"]),
        np.array(full_table[f"R{Delta1}_crit"]),
        np.array(full_table[f"M{Delta2}_crit"]),
        np.array(full_table[f"R{Delta2}_crit"]),
        Rs0=20 * np.ones(len(full_table)),
    )

    full_table[f"c{Delta2}_crit"] = full_table[f"R{Delta2}_crit"] / full_table["Rs"]
    full_table[f"c{Delta1}_crit"] = full_table[f"R{Delta1}_crit"] / full_table["Rs"]

    full_table["Rs_chi2"] = (
        nfw_int(full_table[f"R{Delta1}_crit"] / full_table["Rs"])
        / nfw_int(full_table[f"R{Delta2}_crit"] / full_table["Rs"])
        - full_table[f"M{Delta1}_crit"] / full_table[f"M{Delta2}_crit"]
    ) ** 2


# In[455]:


def convert_mass(M1, c1, c2):
    return M1 * nfw_int(c2) / nfw_int(c1)


# In[613]:


def add_profile_analytic(axes, delta=500, ds_pred_samples=None, **kwargs):
    for i, ax in enumerate(axes.flatten()):
        ax.plot(
            mm.radius,
            clmm.compute_excess_surface_density(
                mm.radius,
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
                mm.radius,
                np.quantile(ds_pred_samples[i], 0.16, axis=0),
                np.quantile(ds_pred_samples[i], 0.84, axis=0),
                color=colors[i],
                alpha=0.3,
                lw=0,
            )


# #### Errs for full_table

# In[41]:


def get_rs_err(M1, R1, M2, R2, Rs, M1_err, M2_err):
    return (
        Rs
        * nfw_int(R1 / Rs)
        * np.sqrt(
            ((M1_err / M1) ** 2 + (M2_err / M2) ** 2)
            / ((R1 / (Rs + R1)) ** 4 + (M1 / M2) ** 2 * (R2 / (Rs + R2)) ** 4)
        )
    )


# In[42]:


def add_conc_rs_err_to_full_table(full_table, Delta1, Delta2):
    full_table["Rs_err"] = get_rs_err(
        *(
            np.array(v)
            for v in (
                full_table[f"M{Delta1}_crit"],
                full_table[f"R{Delta1}_crit"],
                full_table[f"M{Delta2}_crit"],
                full_table[f"R{Delta2}_crit"],
                full_table["Rs"],
                full_table[f"M{Delta1}_crit_err"],
                full_table[f"M{Delta2}_crit_err"],
            )
        )
    )

    full_table[f"c{Delta2}_crit_err"] = (
        full_table[f"c{Delta2}_crit"] * full_table["Rs_err"] / full_table["Rs"]
    )
    full_table[f"c{Delta1}_crit_err"] = (
        full_table[f"c{Delta1}_crit"] * full_table["Rs_err"] / full_table["Rs"]
    )


# ### Compute

# In[29]:


get_ipython().run_cell_magic('time', '', 'add_conc_rs_rho0_to_bs_table(bs_table, 200, 500)\n')


# In[37]:


add_conc_rs_rho0_to_full_table(full_table, 200, 500)


# In[ ]:


add_conc_rs_err_to_full_table(full_table, 200, 500)


# ### Plot temp

# In[352]:


i = np.nanargmax(bs_table["c500_crit"][0])
print(i)
bs_table["Rs_chi2"][0, i], bs_table["Rs"][0, i], bs_table["R500_crit"][0, i], bs_table[
    "c500_crit"
][0, i]


# In[353]:


plt.scatter(bs_table["c500_crit"], bs_table["Rs_chi2"], marker=".", s=1)
plt.xscale("log")
plt.yscale("log")


# In[33]:


plt.scatter(bs_table["c500_crit"], bs_table["Rs_chi2"], marker=".", s=1)
plt.xscale("log")
plt.yscale("log")


# ### Plot M-c

# In[263]:


np.array(np.log10(full_table["M500_crit"]))


# In[260]:


np.log10(full_table["M500_crit_bs_mean"])


# In[300]:


full_table[
    "Rs",
    "Rs_bs_mean",
    "Rs_bs_std",
    "R500_crit",
    "R500_crit_bs_mean",
    "R500_crit_bs_std",
    "c500_crit",
    "c500_crit_bs_mean",
    "c500_crit_bs_std",
]


# In[277]:


np.nanmedian(bs_table["R500_crit"][0] / bs_table["Rs"][0])


# In[297]:


plt.hist(bs_table["R500_crit"][0], bins=150)


# In[279]:


plt.hist(np.log10(bs_table["R500_crit"][0] / bs_table["Rs"][0]), bins=150)


# In[280]:


((bs_table["R500_crit"] / bs_table["Rs"])[0] < 1e5).sum()


# In[288]:


plt.hist(bs_table["Rs"][0], bins=np.geomspace(1e-41, 0.52, 31))
plt.xscale("log")


# In[295]:


plt.hexbin(
    bs_table["R500_crit"][0],
    bs_table["Rs"][0],
    xscale="log",
    yscale="log",
    mincnt=1,
    gridsize=20,
)
# plt.xscale("log")


# In[283]:


bs_table["Rs"][0][~np.isnan(bs_table["Rs"][0])].min(), bs_table["Rs"][0][
    ~np.isnan(bs_table["Rs"][0])
].max()


# In[306]:


plt.errorbar(
    full_table["M500_crit"],
    full_table["c500_crit"],
    full_table["c500_crit_err"],
    full_table["M500_crit_err"],
    ls="",
    marker=".",
    label=r"$\Delta=500$",
)
plt.errorbar(
    full_table["M500_crit_bs_mean"],
    full_table["c500_crit_bs_mean"],
    full_table["c500_crit_bs_std"],
    full_table["M500_crit_bs_std"],
    ls="",
    marker=".",
    label=r"$BS$",
)
plt.xscale("log")
plt.yscale("log")

plt.xlim(4e11, 0.8e14)
plt.ylim(0, 25)

plt.xlabel(r"$M_{\rm crit}$ [$M_\odot$]")
plt.ylabel(r"$c_{\rm crit}$")
plt.legend()


# In[322]:


plt.errorbar(
    full_table["M500_crit"],
    full_table["c500_crit"],
    full_table["c500_crit_err"],
    full_table["M500_crit_err"],
    ls="",
    marker=".",
    label=r"$f(\rm mean)$",
)
plt.errorbar(
    full_table["M500_crit_bs_mean"],
    full_table["c500_crit_bs_mean"],
    full_table["c500_crit_bs_std"],
    full_table["M500_crit_bs_std"],
    ls="",
    marker=".",
    label=r"$BS$",
)
plt.xscale("log")
plt.yscale("log")

plt.xlim(1e12, 0.8e14)
plt.ylim(0, 100)

plt.xlabel(r"$M_{500_{\rm crit}}$ [$M_\odot$]")
plt.ylabel(r"$c_{500_{\rm crit}}$")
plt.legend()


# In[355]:


plt.errorbar(
    full_table["M500_crit"],
    full_table["c500_crit"],
    full_table["c500_crit_err"],
    full_table["M500_crit_err"],
    ls="",
    marker=".",
    label=r"$f(\rm mean)$",
)
plt.errorbar(
    full_table["M500_crit_bs_mean"],
    full_table["c500_crit_bs_mean"],
    full_table["c500_crit_bs_std"],
    full_table["M500_crit_bs_std"],
    ls="",
    marker=".",
    label=r"$BS$",
)
plt.xscale("log")
plt.yscale("log")

plt.xlim(1e12, 0.8e14)
plt.ylim(0, 100)

plt.xlabel(r"$M_{500_{\rm crit}}$ [$M_\odot$]")
plt.ylabel(r"$c_{500_{\rm crit}}$")
plt.legend()


# In[204]:


plt.errorbar(
    full_table["M500_crit"],
    full_table["c500_crit"],
    full_table["c500_crit_err"],
    full_table["M500_crit_err"],
    ls="",
    marker=".",
    label=r"$\Delta=500$",
)
plt.errorbar(
    full_table["M200_crit"],
    full_table["c200_crit"],
    full_table["c200_crit_err"],
    full_table["M200_crit_err"],
    ls="",
    marker=".",
    label=r"$\Delta=200$",
)
plt.xscale("log")

plt.xlim(1e12, 2e14)
plt.ylim(0, 40)

plt.xlabel(r"$M_{\rm crit}$ [$M_\odot$]")
plt.ylabel(r"$c_{\rm crit}$")
plt.legend()


# In[203]:


plt.errorbar(
    full_table["M500_crit"],
    full_table["c500_crit"],
    full_table["c500_crit_err"],
    full_table["M500_crit_err"],
    ls="",
    marker=".",
    label=r"$\Delta=500$",
)
plt.errorbar(
    full_table["M200_crit"],
    full_table["c200_crit"],
    full_table["c200_crit_err"],
    full_table["M200_crit_err"],
    ls="",
    marker=".",
    label=r"$\Delta=200$",
)
plt.xscale("log")
plt.yscale("log")

plt.xlim(1e12, 2e14)
plt.ylim(0.1, 40)

plt.xlabel(r"$M_{\rm crit}$ [$M_\odot$]")
plt.ylabel(r"$c_{\rm crit}$")
plt.legend()


# In[637]:


fig, ax = plot_cm_base()
plt_errorbars(
    ax,
    full_table["M500_crit"],
    full_table["c500_crit"],
    full_table["c500_crit_err"],
    full_table["M500_crit_err"],
    ls="",
    marker="*",
    markersize=10,
    markeredgewidth=0.0,
    label=r"$\Delta=500$",
    lw=1,
)
plt_errorbars(
    ax,
    full_table["M200_crit"],
    full_table["c200_crit"],
    full_table["c200_crit_err"],
    full_table["M200_crit_err"],
    ls="",
    marker="o",
    label=labels[1],
    lw=1,
    markersize=5,
    markeredgewidth=0.5,
    markerfacecolor="none",
)


ax.set_xlim(4e11, 2e14)
ax.set_ylim(0.1, 200)
ax.legend()

plt.savefig("figs/mass_concentration_mean.png")


# ### Plot profiles

# In[601]:


fig, ax = plt.subplots(layout="constrained")

shift = 1
for i in range(6):
    ax.errorbar(
        mm.radius * shift,
        full_table["ds_t_pc2"][i],
        full_table["ds_err_pc2"][i],
        ls="",
        marker=".",
        markersize=7,
        markeredgewidth=0.0,
        color=colors[i],
        lw=1,
    )

    ax.plot(
        mm.radius * shift,
        clmm.compute_excess_surface_density(
            mm.radius,
            full_table["M200_crit"][i],
            full_table["c200_crit"][i],
            full_table["z_l"][i][0],
            # full_table["z_s"][i],
            cosmo,
            delta_mdef=200,
            halo_profile_model="nfw",
            massdef="critical",
        )
        * 10**0,
        color=colors[i],
    )
    shift *= 1.03


ax.set_xscale("log")
ax.set_yscale("log")

plt.ylim(10**10, 10**15)
plt.xlabel("R [Mpc]")
plt.ylabel(r"$\Delta \Sigma$ [M$_\odot$pc$^{-2}$]")
plt.tick_params(axis="both", which="both", direction="in")

fig.colorbar(
    mpl.colorizer.ColorizingArtist(colorizer),
    ax=ax,
    orientation="vertical",
    label=r"log$_{10}$(M$_{\rm lum}$/M$_\odot$)",
)

# plt.savefig("figs/profiles_fit.png")


# In[574]:


get_ipython().run_cell_magic('time', '', 'ds_pred_samples = [\n    [\n        clmm.compute_excess_surface_density(\n            mm.radius,\n            np.exp(\n                nz_normal(\n                    np.log(full_table["M200_crit"][i]),\n                    full_table["M200_crit_err"][i] / full_table["M200_crit"][i],\n                )\n            ),\n            nz_normal(full_table["c200_crit"][i], full_table["c200_crit_err"][i]),\n            full_table["z_l"][i][0],\n            # full_table["z_s"][i],\n            cosmo,\n            delta_mdef=200,\n            halo_profile_model="nfw",\n            massdef="critical",\n        )\n        for j in range(200)\n    ]\n    for i in range(6)\n]\n')


# In[575]:


get_ipython().run_cell_magic('time', '', 'ds_pred_samples = np.array(\n    [\n        [\n            clmm.compute_excess_surface_density(\n                mm.radius,\n                np.exp(\n                    np.random.normal(\n                        np.log(full_table["M200_crit"][i]),\n                        full_table["M200_crit_err"][i] / full_table["M200_crit"][i],\n                    )\n                ),\n                nz_normal(full_table["c200_crit"][i], full_table["c200_crit_err"][i]),\n                full_table["z_l"][i][0],\n                # full_table["z_s"][i],\n                cosmo,\n                delta_mdef=200,\n                halo_profile_model="nfw",\n                massdef="critical",\n            )\n            for j in range(200)\n        ]\n        for i in range(6)\n    ]\n)\n')


# In[619]:


fig, axes = plot_profiles_base()
add_profile_analytic(axes, delta=200, ds_pred_samples=ds_pred_samples)
plt.savefig("figs/profiles_fit.png")


# In[549]:


get_ipython().run_cell_magic('time', '', 'ds_pred_samples_500 = np.array(\n    [\n        [\n            clmm.compute_excess_surface_density(\n                mm.radius,\n                np.exp(\n                    np.random.normal(\n                        np.log(full_table["M500_crit"][i]),\n                        full_table["M500_crit_err"][i] / full_table["M500_crit"][i],\n                    )\n                ),\n                nz_normal(full_table["c500_crit"][i], full_table["c500_crit_err"][i]),\n                full_table["z_l"][i][0],\n                # full_table["z_s"][i],\n                cosmo,\n                delta_mdef=500,\n                halo_profile_model="nfw",\n                massdef="critical",\n            )\n            for j in range(200)\n        ]\n        for i in range(6)\n    ]\n)\n')


# In[607]:


fig, axes = plot_profiles_base()
add_profile_analytic(axes, ds_pred_samples_500)
plt.savefig("figs/profiles_fit_rdelta.png")


# ## Questions

# Why is z_l 2d?

# In[208]:


bs_table


# ## Compute mean(f)

# In[580]:


for col in (
    "Delatcrit_bkg",
    "R500_crit",
    "M500_crit",
    "R200_crit",
    "M200_crit",
    "Rs",
    "rho0",
    "c500_crit",
    "c200_crit",
):
    full_table[f"{col}_bs_mean"] = np.nanmedian(bs_table[col], axis=-1)
    full_table[f"{col}_bs_std"] = 0.5 * (
        np.nanpercentile(bs_table[col], 84, axis=-1)
        - np.nanpercentile(bs_table[col], 16, axis=-1)
    )


# ## Fit with fixed mass

# In[562]:


import pyccl


class ProfileFit:
    def __init__(self):
        self.biases = {}

    @staticmethod
    def func(radius, mdelta, cdelta, z, delta):

        if cdelta < 0:
            return radius * 0

        return clmm.compute_excess_surface_density(
            radius,
            mdelta,
            cdelta,
            z,
            cosmo,
            delta_mdef=delta,
            halo_profile_model="nfw",
            massdef="critical",
        )

    def bias(self, mdelta, z, delta):

        if delta not in self.biases:
            self.biases[delta] = pyccl.halos.HaloBiasTinker10(mass_def=f"{delta}c")

        return self.biases[delta](cosmo.be_cosmo, mdelta, 1 / (1 + z))

    def just2h(self, radius, mdelta, z, delta):

        return clmm.compute_excess_surface_density_2h(
            radius,
            z,
            cosmo,
            halobias=self.bias(mdelta, z, delta),
        )

    def func2h(self, radius, mdelta, cdelta, z, delta):

        if cdelta < 0:
            return radius * 0

        return self.func(radius, mdelta, cdelta, z, delta) + self.just2h(
            radius, mdelta, z, delta
        )


pf = ProfileFit()


# In[116]:


from scipy.optimize import curve_fit


# In[ ]:


def safe_fit(*args, **kwargs):
    try:
        return curve_fit(*args, **kwargs)
    except:
        return np.array([[None], [None]])


# In[654]:


def add_profile(axes, concentration_col, func=pf.func2h, **kwargs):

    for i, ax in enumerate(axes.flatten()):

        ax.plot(
            mm.radius,
            func(
                mm.radius,
                full_table["M500_crit"][i],
                full_table[concentration_col][i],
                full_table["z_l"][i][0],
                500,
            ),
            color=colors[i],
            **kwargs,
        )


def add_profile_fixm(
    axes, concentration_col, ds_pred_samples, func=pf.func2h, **kwargs
):

    add_profile(axes, concentration_col, func=func, **kwargs)

    if ds_pred_samples is None:
        return

    for i, ax in enumerate(axes.flatten()):

        ax.fill_between(
            mm.radius,
            np.quantile(ds_pred_samples[i], 0.16, axis=0),
            np.quantile(ds_pred_samples[i], 0.84, axis=0),
            color=colors[i],
            alpha=0.3,
            lw=0,
        )


# ### Fit

# In[563]:


for delta in (200, 500):
    full_table[f"c{delta}_crit_fixm"], full_table[f"c{delta}_crit_fixm_err"] = np.array(
        [
            [
                v.flatten()[0]
                for v in curve_fit(
                    lambda radius, cdelta: func(
                        radius,
                        full_table[f"M{delta}_crit"][i],
                        cdelta,
                        full_table["z_l"][i][0],
                        delta,
                    ),
                    mm.radius,
                    full_table["ds_t_pc2"][i],
                    sigma=full_table["ds_err_pc2"][i],
                    p0=10,
                )
            ]
            for i in range(6)
        ]
    ).T


# In[153]:


full_table["M500_crit", "c500_crit", "c500_crit_fixm"]


# In[564]:


fig, ax = plt.subplots(layout="constrained")

labels = [r"$\Delta=500$", r"$\Delta=200$"]
for i in range(6):
    ax.errorbar(
        full_table["M500_crit"][i],
        full_table["c500_crit_fixm"][i],
        full_table["c500_crit_fixm_err"][i],
        full_table["M500_crit_err"][i],
        ls="",
        marker="*",
        markersize=10,
        markeredgewidth=0.0,
        label=labels[0],
        color=colors[i],
        lw=1,
    )
    ax.errorbar(
        full_table["M200_crit"][i],
        full_table["c200_crit_fixm"][i],
        full_table["c200_crit_fixm_err"][i],
        full_table["M200_crit_err"][i],
        ls="",
        marker="o",
        label=labels[1],
        color=colors[i],
        lw=1,
        markersize=5,
        markeredgecolor=colors[i],
        markeredgewidth=0.5,
        markerfacecolor="none",
    )
    labels = [None, None]


ax.set_xscale("log")
ax.set_yscale("log")

ax.set_xlim(4e11, 2e14)
ax.set_ylim(0.1, 200)

ax.set_xlabel(r"M$_{\rm crit}$ [M$_\odot$]")
ax.set_ylabel(r"$c_{\rm crit}$")
ax.legend()


fig.colorbar(
    mpl.colorizer.ColorizingArtist(colorizer),
    ax=ax,
    orientation="vertical",
    label=r"log$_{10}$(M$_{\rm lum}$/M$_\odot$)",
)

# plt.savefig("figs/mass_concentration_mean.png")


# In[148]:


c_fits[500]


# In[639]:


fig, ax = plot_cm_base()
plt_errorbars(
    ax,
    full_table["M500_crit"],
    full_table["c500_crit"],
    full_table["c500_crit_err"],
    full_table["M500_crit_err"],
    ls="",
    marker="*",
    markersize=10,
    markeredgewidth=0.0,
    label="Mass ratios",
    lw=1,
)
plt_errorbars(
    ax,
    full_table["M500_crit"],
    full_table["c500_crit_fixm"],
    full_table["c500_crit_fixm_err"],
    full_table["M500_crit_err"],
    ls="",
    marker="o",
    label="Direct fit",
    lw=1,
    markersize=5,
    markeredgewidth=0.5,
    markerfacecolor="none",
)

ax.set_xlim(4e11, 2e14)
ax.set_ylim(0.1, 200)
ax.legend()
plt.savefig("figs/mass_concentration_mean_fixm.png")


# In[566]:


get_ipython().run_cell_magic('time', '', 'ds_pred_samples_500_fixm = np.array(\n    [\n        [\n            clmm.compute_excess_surface_density(\n                mm.radius,\n                full_table["M500_crit"][i],\n                nz_normal(\n                    full_table["c500_crit_fixm"][i], full_table["c500_crit_fixm_err"][i]\n                ),\n                full_table["z_l"][i][0],\n                # full_table["z_s"][i],\n                cosmo,\n                delta_mdef=500,\n                halo_profile_model="nfw",\n                massdef="critical",\n            )\n            for j in range(200)\n        ]\n        for i in range(6)\n    ]\n)\n')


# In[657]:


fig, axes = plot_profiles_base()
add_profile_analytic(axes)
add_profile_fixm(
    axes, "c500_crit_fixm", ds_pred_samples_500_fixm, func=pf.func, ls="--"
)
plt.savefig("figs/profiles_fit_rdelta_fixm.png")


# ### Add 2h term

# In[568]:


for delta in (200, 500):
    full_table[f"c{delta}_crit_fixm2h"], full_table[f"c{delta}_crit_fixm2h_err"] = (
        np.array(
            [
                [
                    v.flatten()[0]
                    for v in safe_fit(
                        lambda radius, cdelta: pf.func2h(
                            radius,
                            full_table[f"M{delta}_crit"][i],
                            cdelta,
                            full_table["z_l"][i][0],
                            delta,
                        ),
                        mm.radius,
                        full_table["ds_t_pc2"][i],
                        sigma=full_table["ds_err_pc2"][i],
                        p0=10,
                    )
                ]
                for i in range(6)
            ]
        ).T
    )


# In[184]:


full_table  # ["M500_crit", "c500_crit", "c500_crit_fixm"]


# ### Plot M-c

# In[641]:


fig, ax = plot_cm_base()
plt_errorbars(
    ax,
    full_table["M500_crit"],
    full_table["c500_crit_fixm2h"],
    full_table["c500_crit_fixm2h_err"],
    full_table["M500_crit_err"],
    ls="",
    marker="*",
    markersize=10,
    markeredgewidth=0.0,
    label=r"$\Delta=500$",
    lw=1,
)
plt_errorbars(
    ax,
    full_table["M200_crit"],
    full_table["c200_crit_fixm2h"],
    full_table["c200_crit_fixm2h_err"],
    full_table["M200_crit_err"],
    ls="",
    marker="o",
    label=r"$\Delta=200$",
    lw=1,
    markersize=5,
    markeredgewidth=0.5,
    markerfacecolor="none",
)


ax.set_xlim(4e11, 2e14)
ax.set_ylim(0.1, 200)
ax.legend()
# plt.savefig("figs/mass_concentration_mean.png")


# In[643]:


fig, ax = plot_cm_base()
plt_errorbars(
    ax,
    full_table["M500_crit"],
    full_table["c500_crit"],
    full_table["c500_crit_err"],
    full_table["M500_crit_err"],
    ls="",
    marker="*",
    markersize=10,
    markeredgewidth=0.0,
    label="Mass ratios",
    lw=1,
)
plt_errorbars(
    ax,
    full_table["M500_crit"],
    full_table["c500_crit_fixm2h"],
    full_table["c500_crit_fixm2h_err"],
    full_table["M500_crit_err"],
    ls="",
    marker="o",
    label="Direct fit",
    lw=1,
    markersize=5,
    markeredgewidth=0.5,
    markerfacecolor="none",
)


ax.set_xlim(4e11, 2e14)
ax.set_ylim(0.1, 200)
ax.legend()

plt.savefig("figs/mass_concentration_mean_fixm2h.png")


# ### Plot profiles

# In[571]:


get_ipython().run_cell_magic('time', '', 'ds_pred_samples_500_fixm2h = np.array(\n    [\n        [\n            pf.func2h(\n                mm.radius,\n                full_table["M500_crit"][i],\n                nz_normal(\n                    full_table["c500_crit_fixm2h"][i],\n                    full_table["c500_crit_fixm2h_err"][i],\n                ),\n                full_table["z_l"][i][0],\n                500,\n            )\n            for j in range(200)\n        ]\n        for i in range(6)\n    ]\n)\n')


# In[652]:


fig, axes = plot_profiles_base()
add_profile_fixm(axes, "c500_crit_fixm2h", ds_pred_samples_500_fixm2h)
for ax in axes.flatten():
    ax.set_ylim(ax.get_ylim())
# 1 halo term
add_profile(axes, "c500_crit_fixm2h", func=pf.func, ls="--")
plt.savefig("figs/profiles_fit_rdelta_fixm2h.png")


# ## MCMC Fit mass and concentration

# ### Functions

# In[45]:


import emcee


# In[581]:


class MCMCFit:

    def __init__(
        self, delta, nwalkers, use_prior=False, clim=(0.1, 20), logmlim=(8, 100)
    ):

        self.delta = delta
        self.nwalkers = nwalkers
        self.use_prior = use_prior
        self.clim = clim
        self.logmlim = logmlim

        self.samplers = [
            emcee.EnsembleSampler(
                nwalkers=self.nwalkers,
                ndim=2,
                log_prob_fn=self.log_like,
                kwargs={"i": i},
            )
            for i in range(6)
        ]

    def theo(self, logmdelta, cdelta):

        return pf.func2h(
            mm.radius,
            10**logmdelta,
            cdelta,
            full_table["z_l"][i][0],
            self.delta,
        )

    def prior(self, logmdelta):
        return (
            -0.5
            * (
                (logmdelta * np.log(10) - np.log(full_table[f"M{delta}_crit"][i]))
                / (
                    full_table[f"M{delta}_crit_err"][i]
                    / full_table[f"M{delta}_crit"][i]
                )
            )
            ** 2
        )

    def log_like(self, parameters, i=0):
        logmdelta, cdelta = parameters

        if cdelta <= self.clim[0]:
            return -np.inf
        if cdelta > self.clim[1]:
            return -np.inf
        if logmdelta <= self.logmlim[0]:
            return -np.inf
        if logmdelta > self.logmlim[1]:
            return np.inf

        diff = self.theo(logmdelta, cdelta) - full_table["ds_t_pc2"][i]
        sig = full_table["ds_err_pc2"][i]

        lnlike = -0.5 * ((diff / sig) ** 2).sum()

        if self.use_prior:
            lnlike += self.prior(logmdelta)

        return lnlike

    def run_mcmc(self, p0=None, nchain=100):

        if p0 is None:
            p0 = np.random.normal([13, 4], [0.1, 0.1], (self.nwalkers, 2))

        for i in range(6):
            self.samplers[i].run_mcmc(p0, nchain)

        self.fit = np.array(
            [
                sampler.get_chain().reshape(nchain * self.nwalkers, 2).T
                for sampler in self.samplers
            ]
        )

        self.lnlike = np.array(
            [sampler.get_log_prob().flatten() for sampler in self.samplers]
        )

    def get_vstack_res(self):
        return np.vstack([self.fit.transpose(1, 0, 2), [self.lnlike]]).transpose(
            1, 0, 2
        )


# In[253]:


def plot_chain(mass_fit):

    n = len(mass_fit) - 1
    fig, axes = plt.subplots(n, n)

    for i, axl in enumerate(axes):
        for j in range(i):
            axl[j].scatter(mass_fit[j], mass_fit[i], s=1)

        # axl[i].scatter(mass_fit[i], np.exp(mass_fit[-1])/np.exp(mass_fit[-1]).max())
        axl[i].scatter(mass_fit[i], np.exp(mass_fit[-1] - mass_fit[-1].max()), s=1)
        # axl[i].scatter(mass_fit[i], mass_fit[-1])

        for ax in axl[i + 1 :]:
            ax.axis("off")

    for ax in axes[:, 1:].flatten():
        ax.set_yticklabels([])
    for ax in axes[:-1, :].flatten():
        ax.set_xticklabels([])


# In[254]:


def plot_like(mass_fit):
    fig, axes = plt.subplots(len(mass_fit))
    for i in range(len(mass_fit)):
        axes[i].plot(mass_fit[i])


# In[346]:


quant_err = lambda x: np.array(
    [
        [np.quantile(x, 0.5) - np.quantile(x, 0.16)],
        [np.quantile(x, 0.84) - np.quantile(x, 0.5)],
    ]
)


# In[362]:


quant_err = lambda x: np.array(
    [
        [np.mean(x) - np.quantile(x, 0.16)],
        [np.quantile(x, 0.84) - np.mean(x)],
    ]
)


# In[364]:


quant_err_vec = lambda x: np.array(
    [
        np.mean(x, axis=-1) - np.quantile(x, 0.16, axis=-1),
        np.quantile(x, 0.84, axis=-1) - np.mean(x, axis=-1),
    ]
)


# In[491]:


def get_chain_mass_and_err(_fit, burnin=2000):
    return (
        10 ** _fit[0, burnin:].mean(),
        10 ** _fit[0, burnin:].mean() * np.log(10) * quant_err(_fit[0, burnin:]),
    )


# In[752]:


def get_chain_mass_and_err_vec(_fit, burnin=2000, metric=np.median):
    mass = 10 ** metric(_fit[:, 0, burnin:], axis=-1)
    return (
        mass,
        (mass * np.log(10) * quant_err_vec(_fit[:, 0, burnin:])).T,
    )


# In[737]:


get_chain_mass_and_err_vec(mcmc_fit_500.fit, burnin=2000)


# In[735]:


np.median(mcmc_fit_500.fit[:, 0, 2000:], axis=-1)


# In[727]:


np.mean(mcmc_fit_500.fit[0, 0, 2000:])


# In[723]:


for i in range(6):
    plot_like(mcmc_fit_500.get_vstack_res()[i, :, 2000:])
    plot_chain(mcmc_fit_500.get_vstack_res()[i, :, 2000:])


# In[722]:


mcmc_fit_500_prior.fit[0, 0, 2000:].mean()


# In[484]:


def plt_mc(ax, fit, burnin, **kwargs):
    mass, mass_err = get_chain_mass_and_err(fit, burnin)
    ax.errorbar(
        mass, fit[1, burnin:].mean(), quant_err(fit[1, burnin:]), mass_err, **kwargs
    )


# In[713]:


def add_mc_fit(ax, fit, burnin, **kwargs):
    _kwargs = {**kwargs}
    for i, (m, merr, c, cerr) in enumerate(
        zip(
            *get_chain_mass_and_err_vec(fit, burnin=burnin),
            fit[:, 1, burnin:].mean(axis=-1),
            quant_err_vec(fit[:, 1, burnin:]).T
        )
    ):
        ax.errorbar(
            m,
            c,
            cerr[:, None],
            merr[:, None],
            color=colors[i],
            markeredgecolor=colors[i],
            **_kwargs
        )
        _kwargs["label"] = None


# In[592]:


def add_profile_mcmc(axes, mcmc_fit, burnin, **kwargs):
    for i, ax in enumerate(axes.flatten()):
        ax.plot(
            mm.radius,
            func2h(
                mm.radius,
                10 ** mcmc_fit[i, 0, burnin:].mean(),
                mcmc_fit[i, 1, burnin:].mean(),
                full_table["z_l"][i][0],
                500,
            ),
            color=colors[i],
            **kwargs,
        )


# ### Compute fit

# In[582]:


mcmc_fit_500 = MCMCFit(delta=500, nwalkers=32, clim=(0.1, 100))


# In[583]:


get_ipython().run_cell_magic('time', '', 'mcmc_fit_500.run_mcmc(p0=np.random.normal([13, 4], [0.1, 0.1], (32, 2)), nchain=1000)\n')


# In[584]:


for i in range(6):
    plot_like(mcmc_fit_500.get_vstack_res()[i, :, 2000:])
    plot_chain(mcmc_fit_500.get_vstack_res()[i, :, 2000:])


# In[610]:


fig, axes = plot_profiles_base()
add_profile_fixm(axes, "c500_crit_fixm2h", ds_pred_samples_500_fixm2h)
# for ax in axes.flatten():
#    ax.set_ylim(ax.get_ylim())
add_profile_analytic(axes, ls="--")
add_profile_mcmc(axes, mcmc_fit_500.fit, burnin=2000, ls=":")
plt.savefig("figs/profiles_fit_rdelta_fixm2h_mcmc.png")


# In[ ]:





# ### Compute fit with mass prior

# In[586]:


mcmc_fit_500_prior = MCMCFit(delta=500, nwalkers=32)


# In[587]:


get_ipython().run_cell_magic('time', '', 'mcmc_fit_500_prior.run_mcmc(p0, 1000)\n')


# In[588]:


for i in range(6):
    plot_like(mcmc_fit_500_prior.get_vstack_res()[i, :, 2000:])
    plot_chain(mcmc_fit_500_prior.get_vstack_res()[i, :, 2000:])


# In[719]:


for i in range(6):
    plot_like(mcmc_fit_500_prior.get_vstack_res()[i, :, 2000:])
    plot_chain(mcmc_fit_500_prior.get_vstack_res()[i, :, 2000:])


# In[372]:


fig, ax = plt.subplots(layout="constrained")

labels = ["Mass ratios", "Conc. fit", "Mass & conc. fit", "Mass & conc. (m_prior)"]
for i in range(6):
    """
    ax.errorbar(
        full_table["M500_crit"][i],
        full_table["c500_crit"][i],
        full_table["c500_crit_err"][i],
        full_table["M500_crit_err"][i],
        ls="",
        marker="*",
        markersize=10,
        markeredgewidth=0.0,
        label=labels[0],
        color=colors[i],
        lw=1,
    )
    """
    ax.errorbar(
        full_table["M500_crit"][i],
        full_table["c500_crit_fixm2h"][i],
        full_table["c500_crit_fixm2h_err"][i],
        full_table["M500_crit_err"][i],
        ls="",
        marker="o",
        label=labels[1],
        color=colors[i],
        lw=1,
        markersize=5,
        markeredgecolor=colors[i],
        markeredgewidth=0.5,
        markerfacecolor="none",
    )
    ax.errorbar(
        10 ** _fit[i, 0, 2000:].mean(),
        _fit[i, 1, 2000:].mean(),
        _fit[i, 1, 2000:].std(),
        10 ** _fit[i, 0, 2000:].mean() * np.log(10) * _fit[i, 0, 2000:].std(),
        ls="",
        marker="^",
        label=labels[2],
        color=colors[i],
        lw=1,
        markersize=10,
        markeredgecolor=colors[i],
        markeredgewidth=0.5,
        markerfacecolor="none",
    )
    ax.errorbar(
        10 ** _fit2[i, 0, 2000:].mean(),
        _fit2[i, 1, 2000:].mean(),
        _fit2[i, 1, 2000:].std(),
        10 ** _fit2[i, 0, 2000:].mean() * np.log(10) * _fit2[i, 0, 2000:].std(),
        ls="",
        marker="v",
        label=labels[3],
        color=colors[i],
        lw=1,
        markersize=10,
        markeredgecolor=colors[i],
        markeredgewidth=0.5,
        markerfacecolor="none",
    )
    labels = [None, None, None, None]


ax.set_xscale("log")
ax.set_yscale("log")

ax.set_xlim(1e11, 2e14)
ax.set_ylim(0.1, 200)

ax.set_xlabel(r"M$_{\rm crit}$ [M$_\odot$]")
ax.set_ylabel(r"$c_{\rm crit}$")
ax.legend()


fig.colorbar(
    mpl.colorizer.ColorizingArtist(colorizer),
    ax=ax,
    orientation="vertical",
    label=r"log$_{10}$(M$_{\rm lum}$/M$_\odot$)",
)

plt.savefig("figs/mass_concentration_mean_fixm2h_mcmc2.png")


# In[739]:


fig, ax = plot_cm_base()

"""
plt_errorbars(
    ax,
    full_table["M500_crit"],
    full_table["c500_crit"],
    full_table["c500_crit_err"],
    full_table["M500_crit_err"],
    ls="",
    marker="o",
        ls="",
        marker="*",
        markersize=10,
        markeredgewidth=0.0,
        label="Mass ratios",
        lw=1,
)
"""
plt_errorbars(
    ax,
    full_table["M500_crit"],
    full_table["c500_crit_fixm2h"],
    full_table["c500_crit_fixm2h_err"],
    full_table["M500_crit_err"],
    ls="",
    marker="o",
    label="Conc. fit",
    lw=1,
    markersize=5,
    markeredgewidth=0.5,
    markerfacecolor="none",
)

add_mc_fit(
    ax,
    mcmc_fit_500.fit,
    2000,
    ls="",
    marker="^",
    label="Mass & conc. fit",
    lw=1,
    markersize=10,
    markeredgewidth=0.5,
    markerfacecolor="none",
)


add_mc_fit(
    ax,
    mcmc_fit_500_prior.fit,
    2000,
    ls="",
    marker="v",
    label="Mass & conc. (m_prior)",
    lw=1,
    markersize=10,
    markeredgewidth=0.5,
    markerfacecolor="none",
)


ax.set_xlim(1e11, 2e14)
ax.set_ylim(0.1, 200)

ax.legend()


# plt.savefig("figs/mass_concentration_mean_fixm2h_mcmc2.png")

# plt.savefig("figs/mass_concentration_mean_fixm2h_mcmc2.png")


# In[611]:


fig, axes = plot_profiles_base()
add_profile(axes, "c500_crit_fixm2h", ds_pred_samples_500_fixm2h)
# add_profile_analytic(axes,ls="--")
add_profile_mcmc(axes, mcmc_fit_500_prior.fit, burnin=2000, ls=":")
plt.savefig("figs/profiles_fit_rdelta_fixm2h_mcmc2.png")


# #### tmp

# In[332]:


_fit[2, 1, 2000:].std(), _fit2[2, 1, 2000:].std()


# In[355]:


quant_err(_fit[2, 1, 2000:]), quant_err(_fit2[2, 1, 2000:])


# In[333]:


plt.plot(_fit[2, 1, 2000:])
plt.plot(_fit2[2, 1, 2000:])


# In[369]:


quant_err_vec(_fit[:, 1, 2000:]).shape


# In[370]:


np.transpose(
    [
        quant_err_vec(_fit[:, 1, 2000:]).mean(axis=0),
        quant_err_vec(_fit2[:, 1, 2000:]).mean(axis=0),
        full_table["M500_crit_bs_std"],
    ]
)


# In[371]:


np.transpose(
    [
        quant_err_vec(_fit[:, 0, 2000:]).mean(axis=0),
        quant_err_vec(_fit2[:, 0, 2000:]).mean(axis=0),
        full_table["M500_crit_err"] / full_table["M500_crit"],
    ]
)


# ### Compute fit with mass prior delta=200

# In[434]:


mcmc_fit_200_prior = MCMCFit(
    delta=200,
    nwalkers=32,
    use_prior=True,
    clim=(0.1, 100),
)


# In[435]:


get_ipython().run_cell_magic('time', '', 'mcmc_fit_200_prior.run_mcmc(p0, 100)\n')


# In[440]:


for i in range(6):
    plot_like(mcmc_fit_200_prior.get_vstack_res()[i, :, 2000:])
    plot_chain(mcmc_fit_200_prior.get_vstack_res()[i, :, 2000:])


# In[479]:


fig, ax = plt.subplots(layout="constrained")

labels = [
    "Mass ratios",
    "Conc. fit",
    "Mass & conc. (fit with 200)",
    "Mass & conc. (fit with 500)",
]
for i in range(6):
    plt_mc(
        ax,
        mcmc_fit_200_prior.fit[i],
        ls="",
        marker="^",
        label=labels[2],
        color=colors[i],
        lw=1,
        markersize=10,
        markeredgecolor=colors[i],
        markeredgewidth=0.5,
        markerfacecolor="none",
    )
    """
    plt_mc(
        ax,
        mcmc_fit_500_prior.fit[i],
        ls="",
        marker="v",
        label=labels[3],
        color=colors[i],
        lw=1,
        markersize=10,
        markeredgecolor=colors[i],
        markeredgewidth=0.5,
        markerfacecolor="none",
    )
    """
    labels = [None, None, None, None]


ax.set_xscale("log")
ax.set_yscale("log")

# ax.set_xlim(1e11, 2e14)
# ax.set_ylim(0.1, 200)

ax.set_xlabel(r"M$_{\rm crit}$ [M$_\odot$]")
ax.set_ylabel(r"$c_{\rm crit}$")
ax.legend()


fig.colorbar(
    mpl.colorizer.ColorizingArtist(colorizer),
    ax=ax,
    orientation="vertical",
    label=r"log$_{10}$(M$_{\rm lum}$/M$_\odot$)",
)

# plt.savefig("figs/mass_concentration_mean_fixm2h_mcmc2.png")


# In[466]:


fig, ax = plt.subplots(layout="constrained")

labels = [
    "Mass ratios",
    "Conc. fit",
    "Mass & conc. (fit with 200)",
    "Mass & conc. (fit with 500)",
]
for i in range(6):
    plt_mc(
        ax,
        np.array(
            [
                # mcmc_fit_200_prior.fit[i, 0]
                # + np.log10(
                #    nfw_int(full_table["c500_crit"][i])
                #    / nfw_int(full_table["c200_crit"][i])
                # ),
                mcmc_fit_200_prior.fit[i, 0]
                # + np.log10(mcmc_fit_500_prior.fit[i, 1] / mcmc_fit_200_prior.fit[i, 1]),
                + np.log10(full_table["c500_crit"][i] / mcmc_fit_200_prior.fit[i, 1]),
                mcmc_fit_200_prior.fit[i, 1]
                / full_table["R200_crit"][i]
                * full_table["R500_crit"][i],
            ]
        ),
        ls="",
        marker="^",
        label=labels[2],
        color=colors[i],
        lw=1,
        markersize=10,
        markeredgecolor=colors[i],
        markeredgewidth=0.5,
        markerfacecolor="none",
    )
    plt_mc(
        ax,
        mcmc_fit_500_prior.fit[i],
        ls="",
        marker="v",
        label=labels[3],
        color=colors[i],
        lw=1,
        markersize=10,
        markeredgecolor=colors[i],
        markeredgewidth=0.5,
        markerfacecolor="none",
    )
    labels = [None, None, None, None]


ax.set_xscale("log")
ax.set_yscale("log")

# ax.set_xlim(1e11, 2e14)
# ax.set_ylim(0.1, 200)

ax.set_xlabel(r"M$_{500_{\rm crit}}$ [M$_\odot$]")
ax.set_ylabel(r"$c_{500_{\rm crit}}$")
ax.legend()


fig.colorbar(
    mpl.colorizer.ColorizingArtist(colorizer),
    ax=ax,
    orientation="vertical",
    label=r"log$_{10}$(M$_{\rm lum}$/M$_\odot$)",
)

# plt.savefig("figs/mass_concentration_mean_fixm2h_mcmc2.png")


# ### Compute fit with mass prior delta=2500

# In[429]:


mcmc_fit_2500_prior = MCMCFit(
    delta=2500,
    nwalkers=32,
    use_prior=True,
    clim=(0.1, 100),
)


# In[430]:


get_ipython().run_cell_magic('time', '', 'mcmc_fit_2500_prior.run_mcmc(p0, 100)\n')


# In[441]:


for i in range(6):
    plot_like(mcmc_fit_2500_prior.get_vstack_res()[i, :, 2000:])
    plot_chain(mcmc_fit_2500_prior.get_vstack_res()[i, :, 2000:])


# In[433]:


fig, ax = plt.subplots(layout="constrained")

labels = ["Mass ratios", "Conc. fit", "Mass & conc. fit 2500", "Mass & conc. 500"]
for i in range(6):
    plt_mc(
        ax,
        mcmc_fit_2500_prior.fit,
        ls="",
        marker="^",
        label=labels[2],
        color=colors[i],
        lw=1,
        markersize=10,
        markeredgecolor=colors[i],
        markeredgewidth=0.5,
        markerfacecolor="none",
    )
    ax.errorbar(
        10 ** _fit2[i, 0, 2000:].mean(),
        _fit2[i, 1, 2000:].mean(),
        quant_err(_fit2[i, 1, 2000:]),
        10 ** _fit2[i, 0, 2000:].mean() * np.log(10) * quant_err(_fit2[i, 0, 2000:]),
        ls="",
        marker="v",
        label=labels[3],
        color=colors[i],
        lw=1,
        markersize=10,
        markeredgecolor=colors[i],
        markeredgewidth=0.5,
        markerfacecolor="none",
    )
    labels = [None, None, None, None]


ax.set_xscale("log")
ax.set_yscale("log")

ax.set_xlim(1e11, 2e14)
ax.set_ylim(0.1, 200)

ax.set_xlabel(r"M$_{\rm crit}$ [M$_\odot$]")
ax.set_ylabel(r"$c_{\rm crit}$")
ax.legend()


fig.colorbar(
    mpl.colorizer.ColorizingArtist(colorizer),
    ax=ax,
    orientation="vertical",
    label=r"log$_{10}$(M$_{\rm lum}$/M$_\odot$)",
)

# plt.savefig("figs/mass_concentration_mean_fixm2h_mcmc2.png")


# ### Compare approaches

# In[347]:


quant_err(_fit[i, 1, 2000:])


# In[348]:


_fit[i, 1, 2000:].mean()


# In[480]:


fig, ax = plt.subplots(layout="constrained")

labels = ["Mass ratios", "Conc. fit", "Mass & conc. fit", "Mass & conc. (m_prior)"]
for i in range(6):
    ax.errorbar(
        10 ** _fit[i, 0, 2000:].mean(),
        _fit[i, 1, 2000:].mean(),
        quant_err(_fit[i, 1, 2000:]),
        10 ** _fit[i, 0, 2000:].mean() * np.log(10) * quant_err(_fit[i, 0, 2000:]),
        ls="",
        marker=".",
        label=labels[2],
        color=colors[i],
        lw=1,
        markersize=5,
        markeredgecolor=colors[i],
        markeredgewidth=0.5,
        markerfacecolor="none",
    )
    ax.errorbar(
        10 ** _fit2[i, 0, 2000:].mean(),
        _fit2[i, 1, 2000:].mean(),
        quant_err(_fit2[i, 1, 2000:]),
        10 ** _fit2[i, 0, 2000:].mean() * np.log(10) * quant_err(_fit2[i, 0, 2000:]),
        ls="",
        marker="o",
        label=labels[3],
        color=colors[i],
        lw=1,
        markersize=5,
        markeredgecolor=colors[i],
        markeredgewidth=0.5,
        markerfacecolor="none",
    )
    labels = [None, None, None, None]
    ax.axvline(full_table["M500_crit"][i], color=colors[i])


ax.set_xscale("log")
ax.set_yscale("log")

ax.set_xlim(1e11, 2e14)
ax.set_ylim(0.1, 200)

ax.set_xlabel(r"M$_{\rm crit}$ [M$_\odot$]")
ax.set_ylabel(r"$c_{\rm crit}$")
ax.legend()


fig.colorbar(
    mpl.colorizer.ColorizingArtist(colorizer),
    ax=ax,
    orientation="vertical",
    label=r"log$_{10}$(M$_{\rm lum}$/M$_\odot$)",
)

# plt.savefig("figs/mass_concentration_mean_fixm2h_mcmc2.png")


# In[480]:


fig, ax = plt.subplots(layout="constrained")

labels = ["Mass ratios", "Conc. fit", "Mass & conc. fit", "Mass & conc. (m_prior)"]
for i in range(6):
    ax.errorbar(
        10 ** _fit[i, 0, 2000:].mean(),
        _fit[i, 1, 2000:].mean(),
        quant_err(_fit[i, 1, 2000:]),
        10 ** _fit[i, 0, 2000:].mean() * np.log(10) * quant_err(_fit[i, 0, 2000:]),
        ls="",
        marker=".",
        label=labels[2],
        color=colors[i],
        lw=1,
        markersize=5,
        markeredgecolor=colors[i],
        markeredgewidth=0.5,
        markerfacecolor="none",
    )
    ax.errorbar(
        10 ** _fit2[i, 0, 2000:].mean(),
        _fit2[i, 1, 2000:].mean(),
        quant_err(_fit2[i, 1, 2000:]),
        10 ** _fit2[i, 0, 2000:].mean() * np.log(10) * quant_err(_fit2[i, 0, 2000:]),
        ls="",
        marker="o",
        label=labels[3],
        color=colors[i],
        lw=1,
        markersize=5,
        markeredgecolor=colors[i],
        markeredgewidth=0.5,
        markerfacecolor="none",
    )
    labels = [None, None, None, None]
    ax.axvline(full_table["M500_crit"][i], color=colors[i])


ax.set_xscale("log")
ax.set_yscale("log")

ax.set_xlim(1e11, 2e14)
ax.set_ylim(0.1, 200)

ax.set_xlabel(r"M$_{\rm crit}$ [M$_\odot$]")
ax.set_ylabel(r"$c_{\rm crit}$")
ax.legend()


fig.colorbar(
    mpl.colorizer.ColorizingArtist(colorizer),
    ax=ax,
    orientation="vertical",
    label=r"log$_{10}$(M$_{\rm lum}$/M$_\odot$)",
)

# plt.savefig("figs/mass_concentration_mean_fixm2h_mcmc2.png")


# In[499]:


fig, ax = plt.subplots(layout="constrained")


for i in range(6):
    ax.errorbar(
        full_table["M500_crit"][i],
        *get_chain_mass_and_err(mcmc_fit_500.fit[i], 2000),
        xerr=full_table["M500_crit_err"][i],
        ls="",
        marker=".",
        color=colors[i],
        lw=1,
        markersize=5,
        markeredgecolor=colors[i],
        markeredgewidth=0.5,
        markerfacecolor="none",
    )
    ax.errorbar(
        full_table["M500_crit"][i],
        *get_chain_mass_and_err(mcmc_fit_500_prior.fit[i], 2000),
        xerr=full_table["M500_crit_err"][i],
        ls="",
        marker="o",
        color=colors[i],
        lw=1,
        markersize=5,
        markeredgecolor=colors[i],
        markeredgewidth=0.5,
        markerfacecolor="none",
    )


ax.set_xscale("log")
ax.set_yscale("log")

ax.set_xlim(ax.get_xlim())
ax.set_ylim(ax.get_ylim())
ax.set_xlim(left=1e11)

# ax.set_xlim(1e11, 2e14)
# ax.set_ylim(0.1, 200)


diag = (1e11, 1e14)

ax.plot(diag, diag, ls="--", lw=0.5, color="0")

ax.set_xlabel(r"Enclosed M$_{500_{\rm crit}}$ [M$_\odot$]")
ax.set_ylabel(r"Fitted M$_{500_{\rm crit}}$ [M$_\odot$]")


fig.colorbar(
    mpl.colorizer.ColorizingArtist(colorizer),
    ax=ax,
    orientation="vertical",
    label=r"log$_{10}$(M$_{\rm lum}$/M$_\odot$)",
)

# plt.savefig("figs/mass_concentration_mean_fixm2h_mcmc2.png")


# In[517]:


fig, ax = plt.subplots(layout="constrained")


for i in range(6):
    ax.errorbar(
        full_table["M500_crit"][i],
        *get_chain_mass_and_err(mcmc_fit_500.fit[i], 6000),
        xerr=full_table["M500_crit_err"][i],
        ls="",
        marker=".",
        color=colors[i],
        lw=1,
        markersize=5,
        markeredgecolor=colors[i],
        markeredgewidth=0.5,
        markerfacecolor="none",
    )


ax.set_xscale("log")
ax.set_yscale("log")

ax.set_xlim(ax.get_xlim())
ax.set_ylim(ax.get_ylim())
ax.set_xlim(left=1e11)

# ax.set_xlim(1e11, 2e14)
# ax.set_ylim(0.1, 200)


diag = (1e11, 1e14)

ax.plot(diag, diag, ls="--", lw=0.5, color="0")

ax.set_xlabel(r"Enclosed M$_{500_{\rm crit}}$ [M$_\odot$]")
ax.set_ylabel(r"Fitted M$_{500_{\rm crit}}$ [M$_\odot$]")


fig.colorbar(
    mpl.colorizer.ColorizingArtist(colorizer),
    ax=ax,
    orientation="vertical",
    label=r"log$_{10}$(M$_{\rm lum}$/M$_\odot$)",
)

plt.savefig("figs/compare_mass_enclosed_fitted.png")


# In[756]:


fig, ax = plt.subplots(layout="constrained")

plt_errorbars(
    ax,
    full_table["M500_crit"],
    get_chain_mass_and_err_vec(mcmc_fit_500.fit, 6000, metric=np.median)[0],
    get_chain_mass_and_err_vec(mcmc_fit_500.fit, 6000)[1][:,:,None],
    full_table["M500_crit_err"],
    ls="",
    marker=".",
    lw=1,
    markersize=5,
    markeredgewidth=0.5,
    markerfacecolor="none",
)


ax.set_xscale("log")
ax.set_yscale("log")

ax.set_xlim(ax.get_xlim())
ax.set_ylim(ax.get_ylim())
ax.set_xlim(left=1e11)

# ax.set_xlim(1e11, 2e14)
# ax.set_ylim(0.1, 200)


diag = (1e11, 1e14)

ax.plot(diag, diag, ls="--", lw=0.5, color="0")

ax.set_xlabel(r"Enclosed M$_{500_{\rm crit}}$ [M$_\odot$]")
ax.set_ylabel(r"Fitted M$_{500_{\rm crit}}$ [M$_\odot$]")


fig.colorbar(
    mpl.colorizer.ColorizingArtist(colorizer),
    ax=ax,
    orientation="vertical",
    label=r"log$_{10}$(M$_{\rm lum}$/M$_\odot$)",
)

plt.savefig("figs/compare_mass_enclosed_fitted.png")


# # New approach

# In[474]:


m1m2s = np.array([np.meshgrid(m, m) for m in full_table["magic_mass_bs_mean"]])
m1m2s.shape


# In[473]:


r1r2s = np.array([np.meshgrid(m, m) for m in full_table["rp"]])
r1r2s.shape


# In[ ]:


"""
rho0s = [
    [
        get_rho0_rs_safe(m200, r200, m500, r500, Rs0=20 * np.ones(m200.size))
        for m200, r200 in zip
        """

