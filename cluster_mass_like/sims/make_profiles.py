import astropy.units as u
import numpy as np
from astropy.cosmology import WMAP7 as cosmo
from pathlib import Path

from . import lensing as lensing
from . import sims_io as sims_io


def weighted_nan_avg(param, weights, axis=0):
    """weighted_nan_avg.

    weighted average of param that sets nan-values to zero

    Parameters
    ----------
    param: numpy.ndarray
       quantity to be averaged
    weights: numpy.ndarray
       weights for weighted average
    Returns
    -------
    numpy.ndarray
       averaged quantity that allows for nans
    """
    # prepare terms with the correct output shapes
    params_out = param * np.ones_like(weights)
    weights_out = np.ones_like(param) * weights

    bad_values = np.isnan(params_out * weights_out)

    params_out[bad_values] = 0
    weights_out[bad_values] = 0

    return (params_out * weights_out).sum(axis=axis) / weights_out.sum(axis=axis)


def profile_from_map(
    z_l, mass_map, miscentering, n_rbins, z_s_arr, P_zs, pix_size, l_proj
):

    E1, E2 = lensing.KaiserSquires(mass_map)

    # apply miscentering and get tangential shear

    Et, Ex, R, phi = lensing.getTangential(E1, E2, miscentering, pix_size)

    mass_map = lensing.remove_projected_convergence(z_l, l_proj, cosmo)

    # to save memory, let us project this into wedges with radial bins
    r_bins, gammat_radial, kappa_radial = lensing.compress_gt(
        mass_map, Et, R, phi, n_rbins
    )

    Sigma_inv_ls = lensing.Sigma_crit_inv(z_l, z_s_arr)

    gt_map = (
        Sigma_inv_ls[None, None, :]
        * gammat_radial[:, :, None]
        / (1 - Sigma_inv_ls[None, None, :] * kappa_radial[:, :, None])
    )

    # perform the azimutal average
    s_crit_inv_eff = weighted_nan_avg(Sigma_inv_ls, weights=P_zs)
    dsig_t_zs = np.nanmean(gt_map, axis=1) / s_crit_inv_eff

    # source redshift average
    dsig_t = np.nansum(dsig_t_zs * P_zs[None, :] * np.diff(z_s_arr)[0], axis=1)

    # I like to plot the radial position with the average distance of point in an anulus
    r_centers = 2 / 3 * np.diff(r_bins**3) / np.diff(r_bins**2)

    return dsig_t, 1 / s_crit_inv_eff, r_centers


def get_all_profiles(
    m_bins,
    pix_size=10 / 500,
    n_rbins=51,
    l_proj=5,
    groups_path="/sps/lsst/users/lbaumont/data/HSC/sims/Matched_Halo_Robotham_Mlim.fits",
    nz_path="/sps/lsst/users/lbaumont/data/HSC/nz.fits",
    maps_root=Path("/sps/lsst/users/lbaumont/data/HSC/sims/MassMapsLC30"),
):
    sigma_crit_eff = []
    profiles = []
    groups = sims_io.prepare_groups_catalog(groups_path, m_bins, n_rbins)
    z_s_arr, P_zs = sims_io.load_nz(nz_path)

    for row in groups:
        z_l = row["IterCenZ"][id].value
        miscentering = [row["dx"].value, row["dy"].value]

        # get mass map
        group_nbr = row["grnr"]
        snapshot_nbr = row["snapshot"]
        mass_map = sims_io.load_massmap(
            maps_root / "projected_maps" / f"grnr{group_nbr}_snap{snapshot_nbr}.npy",
            pix_size,
        )

        dsig_t, s_crit_eff, r_centers = profile_from_map(
            z_l, mass_map, miscentering, n_rbins, z_s_arr, P_zs, pix_size, l_proj
        )
        profiles.append(dsig_t)
        sigma_crit_eff.append(s_crit_eff)
    print(r_centers)
    groups["DSig_t"] = np.array(profiles)
    groups["Scrit_eff"] = np.array(sigma_crit_eff)
