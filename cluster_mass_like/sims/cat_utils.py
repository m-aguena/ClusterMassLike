import astropy.units as u
import numpy as np


def get_miscentering(ra_true, dec_true, ra_meas, dec_meas, z_lens, cosmo):
    """get miscentering
    given ra and dec of a measured and true detections, compute the miscentering in Mpc/h
    tbd: which direction is RA, and is it along x or y?
    """
    dA_lens = cosmo.angular_diameter_distance(z_lens).value
    h = cosmo.h

    # take geometric mean of dec
    scale = np.sqrt(
        np.cos(dec_meas * u.deg.to(u.rad)) * np.cos(dec_true * u.deg.to(u.rad))
    )
    delta_ra = (ra_meas - ra_true) * u.deg.to(u.rad) * scale

    dx = delta_ra * dA_lens * h

    delta_dec = (dec_meas - dec_true) * u.deg.to(u.rad)

    dy = delta_dec * dA_lens * h

    return dx, dy


def optical_mass_proxy(LumBfunc):
    """compute mass proxy from scaling relation"""
    alpha = 0.94
    beta = -0.24

    lum_r = np.log10(np.array(LumBfunc))

    mass_lum = ((lum_r - 12) - beta) / alpha + 14.0 - 2.0 * np.log10(0.7)
    return mass_lum


def mass_from_density(r_mid, density):
    """mass from density.
    Integrate density profile to get mass profile
    """
    integrand = 4.0 * np.pi * density * r_mid**2

    mass = np.zeros(len(r_mid), dtype=float)
    # cumulative trapezoidal integration
    mass[1:] = np.cumsum(0.5 * (integrand[1:] + integrand[:-1]) * np.diff(r_mid))
    return mass
