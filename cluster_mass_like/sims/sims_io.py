from pathlib import Path

import numpy as np
from astropy.table import Table

from . import cat_utils as utils


def load_nz(nz_path):
    nz = Table.read(nz_path)
    z_s_arr = nz["Z_MID"]
    P_zs = 1 / 3 * (nz["BIN2"] + nz["BIN3"] + nz["BIN4"]) / (np.diff(nz["Z_MID"])[0])
    return z_s_arr, P_zs


def load_density_map(root, group_nbr, snapshot_nbr):
    # root = Path("/sps/lsst/users/lbaumont/data/HSC/sims/MassMapsLC30")

    density = np.load(
        root / "density_profile" / f"grnr{group_nbr}_snap{snapshot_nbr}.npy"
    )
    # density in M_sol/(h^-1 kpc)^3
    r_edges = np.logspace(-2, np.log10(5000.0), 500)
    r_mid = 0.5 * (r_edges[:-1] + r_edges[1:])
    assert density.shape == (499,)
    assert r_mid.shape == density.shape
    return density, r_mid


def load_massmap(filepath, pix_size):
    mass_map_per_pixel = np.load(filepath)
    mass_map = mass_map_per_pixel / pix_size**2 * 1e10
    assert mass_map.shape == (500, 500)
    return mass_map


def prepare_groups_catalog(groups_path, m_bins, cosmo, n_rbins=51):
    """adds necessary columns to input group catalog"""
    groups = Table.read(groups_path)

    # add columns for later
    groups.add_column(np.zeros((len(groups), n_rbins)), name="DSig_t")
    groups.add_column(np.zeros(len(groups)), name="Scrit_eff")

    # add optical mass proxy as in the data
    groups["mass_lum"] = utils.optical_mass_proxy(groups["LumBfunc"])

    # fix 360 degree issue
    groups["RA"][np.where(groups["RA"] > 200)] = (
        groups["RA"][np.where(groups["RA"] > 200)] - 360
    )
    groups["BCGRA"][np.where(groups["BCGRA"] > 200)] = (
        groups["BCGRA"][np.where(groups["BCGRA"] > 200)] - 360
    )

    # find miscentering
    groups["dx"], groups["dy"] = utils.get_miscentering(
        groups["RA"],
        groups["DEC"],
        groups["BCGRA"],
        groups["BCGDEC"],
        groups["IterCenZ"],
        cosmo,
    )

    # get mass bins
    groups["mass_bins"] = np.digitize(groups["mass_lum"], m_bins)
    # trim unnecessary bins
    keep_bins = np.where((groups["mass_bins"] > 1) & (groups["mass_bins"] < 7))
    groups = groups[keep_bins]
    return groups


def stack_profiles(profiles, sigma_crit):
    weights = 1 / (sigma_crit) ** 2
    stack_profiles = np.average(profiles, weights=weights)
