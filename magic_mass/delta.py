import numpy as np
import matplotlib.pyplot as plt

from scipy.interpolate import interp1d
import astropy.units as u


def get_rdelta(mdelta, delta, z, cosmo):
    return (
        3
        * mdelta
        / (4 * np.pi * cosmo.critical_density(z).to(u.solMass / u.Mpc**3).value * delta)
    ) ** (1 / 3)


def get_delta(mass, radius, rho_bkg):
    return mass / (4 * np.pi / 3 * radius**3 * rho_bkg)


def get_delta_quantity_np(delta, delta_vals, quantity_vals, log=False):
    mask = quantity_vals > 0
    if log:
        return 10 ** np.interp(delta, delta_vals[mask], np.log10(quantity_vals[mask]))
    return np.interp(delta, delta_vals[mask], quantity_vals[mask])


def get_delta_quantity(delta, delta_vals, quantity_vals, log=False):
    mask = quantity_vals > 0
    if log:
        return 10 ** interp1d(
            delta_vals[mask], np.log10(quantity_vals[mask]), bounds_error=False
        )(delta)
    return interp1d(delta_vals[mask], quantity_vals[mask], bounds_error=False)(delta)


def add_mrdelta_to_bs_table(bs_table, Delta):
    bs_table[f"R{Delta}_crit"] = np.array(
        [
            [
                get_delta_quantity(Delta, _Delatcrit_bkg, radius, log=True)
                for _Delatcrit_bkg in Delatcrit_bkg.T
            ]
            for Delatcrit_bkg, radius in zip(
                bs_table["Delatcrit_bkg"], bs_table["enclosed_mass_radius"]
            )
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
        get_delta_quantity(Delta, Delatcrit_bkg, radius, log=True)
        for Delatcrit_bkg, radius in zip(
            full_table["Delatcrit_bkg"], full_table["enclosed_mass_radius"]
        )
    ]
    full_table[f"M{Delta}_crit"] = [
        get_delta_quantity(Delta, Delatcrit_bkg, mass_vals, log=True)
        for Delatcrit_bkg, mass_vals in zip(
            full_table["Delatcrit_bkg"], full_table["magic_mass_bs_mean"]
        )
    ]
    full_table[f"M{Delta}_crit"].info.format = "%.4e"


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


def add_mrdelta_err_to_full_table(full_table, Delta):
    full_table[f"M{Delta}_crit_err"] = [
        get_delta_quantity(Delta, Delatcrit_bkg, mass_vals, log=False)
        for Delatcrit_bkg, mass_vals in zip(
            full_table["Delatcrit_bkg"], full_table["magic_mass_bs_std"]
        )
    ]
    full_table[f"M{Delta}_crit_err"].info.format = "%.4e"
