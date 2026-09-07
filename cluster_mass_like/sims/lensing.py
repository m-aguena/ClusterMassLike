import numpy as np
from astropy import constants as const


def KaiserSquires(Sigma):

    kappa_tilde = np.fft.fft2(Sigma)

    k = np.fft.fftfreq(kappa_tilde.shape[0])

    oper_1 = (
        -1.0 / (k[:, None] ** 2 + k[None, :] ** 2) * (k[:, None] ** 2 - k[None, :] ** 2)
    )
    oper_2 = -2.0 / (k[:, None] ** 2 + k[None, :] ** 2) * k[:, None] * k[None, :]

    # k=0 is a mass sheat, which does not produce any shear
    oper_1[0, 0] = 0
    oper_2[0, 0] = 0

    e1_tilde = oper_1 * kappa_tilde
    e2_tilde = oper_2 * kappa_tilde

    e1 = np.fft.ifft2(e1_tilde).real
    e2 = np.fft.ifft2(e2_tilde).real

    return e1, e2


def getTangential(e1, e2, center, dx):
    # define a center and get for each pixel the scaled tangential/cross shear,
    # as well as the position angle and distance from the center of that pixel

    # dx can be adjusted to match the pixel size

    n = e1.shape[0]

    xx = np.arange(-n / 2, n / 2) * dx

    XX, YY = np.meshgrid(xx, xx)

    center_1, center_2 = center

    from_cent_1 = XX - center_1
    from_cent_2 = YY - center_2

    angle = -np.sign(from_cent_2) * np.arccos(
        from_cent_1 / np.sqrt(from_cent_1**2 + from_cent_2**2)
    )
    radius = np.sqrt(from_cent_1**2 + from_cent_2**2)

    angle[np.isnan(angle)] = 0

    et = -e1 * np.cos(2 * angle) - e2 * np.sin(2 * angle)
    ex = +e1 * np.sin(2 * angle) - e2 * np.cos(2 * angle)

    return et, ex, radius, angle


def compress_gt(mass_map, Et, R, phi, n_rbins):
    # to save memory, let us project this into wedges with radial bins
    r_bins = np.insert(np.logspace(-2, np.log10(7.08), n_rbins), 0, 0)
    phi_bins = np.linspace(-np.pi, np.pi + 0.01, num=30)

    ii_r = np.digitize(R, r_bins) - 1
    ii_phi = np.digitize(phi, phi_bins) - 1

    _N = np.zeros((len(r_bins) - 1, len(phi_bins) - 1))
    _K = np.zeros((len(r_bins) - 1, len(phi_bins) - 1))
    _GT = np.zeros((len(r_bins) - 1, len(phi_bins) - 1))

    for i_r, i_phi, kk, gg in zip(
        ii_r.flatten(), ii_phi.flatten(), mass_map.flatten(), Et.flatten()
    ):
        _N[i_r, i_phi] += 1
        _K[i_r, i_phi] += kk
        _GT[i_r, i_phi] += gg

    kappa_radial = _K / _N
    gammat_radial = _GT / _N
    return r_bins, gammat_radial, kappa_radial


def remove_projected_convergence(z_l, l_proj, cosmo):
    # the convergence is only given by the excess mass overdensity w.r.t. to the mean density
    # but if we project l_proj = +/- 10 Mpc/h, we accumulate mean matter density in the mass map
    # this mean matter density does not contribute to the convergence in WL
    rho_crit = 2.775362e11  # critical density of the Universe in units of Msol / Mpc2 h -- is actually a constant!

    Omega_M = cosmo.Om0  # adjust to Omega_M of simulations

    # the present matter density is Omega_M*rho_crit, and it scales like (1+z)^3
    rho_M_z = Omega_M * rho_crit * (1 + z_l) ** 3
    floor = (
        2 * l_proj * rho_M_z
    )  # accumulated surface mass density from constant matter density in simulation snapshot

    mass_map -= floor
    # fun fact: the mean matter density is accounted for via the Friedman equation
    # and changes the angular diameter distances
    return mass_map


def Sigma_crit_inv(z_l, z_s_arr, cosmo):
    # compute Sigma_crit_inv(z_s)
    invSigma0 = (4 * np.pi * const.G / const.c**2).to(u.Mpc / u.M_sun).value

    h = cosmo.h  # adjust to simulation
    Ds = cosmo.angular_diameter_distance(z_s_arr).value * h
    Dls = cosmo.angular_diameter_distance_z1z2(z_l, z_s_arr).value * h
    Dl = cosmo.angular_diameter_distance(z_l).value * h
    # following the notation in https://arxiv.org/pdf/2402.08455 eq 3
    # it is of create importance to set the lensing efficiency for foreground sources to 0!

    # DO NOT truncate the source redshift distribution!
    Sigma_inv_ls = invSigma0 * Dl / Ds * np.maximum(0, Dls)  # units Msol / Mpc2 h
    return Sigma_inv_ls
