from pathlib import Path
import numpy as np
from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt
from astropy.cosmology import WMAP7 as cosmo
import astropy.units as u
from astropy.table import Table, join
import matplotlib.colors as mcolors
from astropy import constants as const


def get_miscentering(ra_true, dec_true, ra_meas, dec_meas, z_lens, cosmo):
    """get miscentering
    given ra and dec of a measured and true detections, compute the miscentering in Mpc/h
    tbd: which direction is RA, and is it along x or y?
    """
    dA_lens=cosmo.angular_diameter_distance(z_lens).value
    h = cosmo.h

    # take geometric mean of dec
    scale = np.sqrt(np.cos(dec_meas*u.deg.to(u.rad)) * np.cos(dec_true*u.deg.to(u.rad)))
    delta_ra = (ra_meas - ra_true)*u.deg.to(u.rad)*scale
     
    dx = delta_ra * dA_lens * h

    delta_dec = (dec_meas - dec_true)*u.deg.to(u.rad)

    dy = delta_dec * dA_lens * h

    return dx, dy

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

def KaiserSquires(Sigma):
    
    kappa_tilde = np.fft.fft2(Sigma)
    
    k = np.fft.fftfreq(kappa_tilde.shape[0])

    oper_1  = - 1./(k[:, None]**2 + k[None, :]**2) * (k[:, None]**2 - k[None, :]**2)
    oper_2  = - 2./(k[:, None]**2 + k[None, :]**2) * k[:, None]*k[None, :]
    
    # k=0 is a mass sheat, which does not produce any shear
    oper_1[0, 0] = 0
    oper_2[0, 0] = 0
    
    e1_tilde = oper_1*kappa_tilde
    e2_tilde = oper_2*kappa_tilde

    e1 = np.fft.ifft2(e1_tilde).real
    e2 = np.fft.ifft2(e2_tilde).real
    
    return e1, e2

def getTangential(e1, e2, center, dx=pix_size):
    
    n = e1.shape[0]
    
    xx = np.arange(-n/2, n/2)*dx

    XX, YY = np.meshgrid(xx, xx)
    
    center_1, center_2 = center
    
    from_cent_1 = XX - center_1
    from_cent_2 = YY - center_2

    angle = -np.sign(from_cent_2)*np.arccos(from_cent_1/np.sqrt(from_cent_1**2+from_cent_2**2))
    radius = np.sqrt(from_cent_1**2+from_cent_2**2)
    
    angle[np.isnan(angle)] = 0
    
    et = - e1*np.cos(2*angle) - e2*np.sin(2*angle)
    ex = + e1*np.sin(2*angle) - e2*np.cos(2*angle)
    
    return et, ex, radius, angle

def get_nz(nz_path):
    nz=Table.read(nz_path)
    z_s_arr = nz['Z_MID']
    P_zs = 1/3*(nz['BIN2']+nz['BIN3']+nz['BIN4'])/(np.diff(nz['Z_MID'])[0])
    return z_s_arr, P_zs

def make_profile(Sigma, radius, rbins):

    drbins = np.diff(np.log(rbins[1:]))[0]
    rbins_c = rbins[1:]*np.exp(-0.5*drbins)
    
    radial_n = np.histogram(radius.flatten(), bins=rbins)[0]
    radial_et = np.histogram(radius.flatten(), bins=rbins, weights=Sigma.flatten())[0]/radial_n
    
    return radial_et, rbins_c

def optical_mass_proxy(LumBfunc):
    """compute mass proxy from scaling relation
    """
    alpha=0.94
    beta=-0.24
        
    lum_r = np.log10(np.array(LumBfunc))
    
    mass_lum = ((lum_r - 12) - beta) / alpha + 14. - 2. * np.log10(0.7)
    return mass_lum

def prepare_groups_catalog(groups_path, m_bins, n_rbins=51):
    """ adds necessary columns to input group catalog
    """
    groups=Table.read(groups_path)

    # add columns for later
    groups.add_column(np.zeros((len(groups),n_rbins)),name='DSig_t')
    groups.add_column(np.zeros(len(groups)),name='Scrit_eff')

    # add optical mass proxy as in the data
    groups['mass_lum'] = optical_mass_proxy(groups['LumBfunc'])

    # fix 360 degree issue
    groups['RA'][np.where(groups['RA'] > 200)]=groups['RA'][np.where(groups['RA'] > 200)]-360
    groups['BCGRA'][np.where(groups['BCGRA'] > 200)]=groups['BCGRA'][np.where(groups['BCGRA'] > 200)]-360

    # find miscentering
    groups['dx'], groups['dy'] = get_miscentering(groups['RA'],groups['DEC'],groups['BCGRA'],groups['BCGDEC'], groups['IterCenZ'], cosmo)

    # get mass bins
    groups['mass_bins'] = np.digitize(groups['mass_lum'], m_bins)
    # trim unnecessary bins
    keep_bins=np.where((groups['mass_bins'] > 1) & (groups['mass_bins'] < 7)) 
    groups = groups[keep_bins]
    return groups

def Sigma_crit_inv(z_l, z_s_arr):
    # compute Sigma_crit_inv(z_s)
    invSigma0 = ( 4*np.pi*const.G/const.c**2 ).to(u.Mpc/u.M_sun).value

    h = cosmo.h  # adjust to simulation
    Ds = cosmo.angular_diameter_distance(z_s_arr).value*h
    Dls = cosmo.angular_diameter_distance_z1z2(z_l, z_s_arr).value*h
    Dl = cosmo.angular_diameter_distance(z_l).value*h
    # following the notation in https://arxiv.org/pdf/2402.08455 eq 3
    # it is of create importance to set the lensing efficiency for foreground sources to 0!

    # DO NOT truncate the source redshift distribution!
    Sigma_inv_ls = invSigma0*Dl/Ds*np.maximum(0, Dls)  # units Msol / Mpc2 h
    return Sigma_inv_ls

def compress_gt(mass_map,Et,R,phi,n_rbins):
    # to save memory, let us project this into wedges with radial bins    
    r_bins = np.insert(np.logspace(-2, np.log10(7.08), n_rbins), 0, 0)
    phi_bins = np.linspace(-np.pi, np.pi+0.01, num=30)
    
    ii_r = np.digitize(R, r_bins)-1
    ii_phi = np.digitize(phi, phi_bins)-1
    
    _N = np.zeros((len(r_bins)-1, len(phi_bins)-1))
    _K = np.zeros((len(r_bins)-1, len(phi_bins)-1))
    _GT = np.zeros((len(r_bins)-1, len(phi_bins)-1))
    
    for i_r, i_phi, kk, gg in zip(ii_r.flatten(), ii_phi.flatten(), mass_map.flatten(), Et.flatten()):
        _N[i_r, i_phi] += 1
        _K[i_r, i_phi] += kk
        _GT[i_r, i_phi] += gg
    
    kappa_radial = _K/_N
    gammat_radial = _GT/_N
    return r_bins, gammat_radial, kappa_radial

def remove_projected_convergence(z_l, l_proj):
    # the convergence is only given by the excess mass overdensity w.r.t. to the mean density
    # but if we project l_proj = +/- 10 Mpc/h, we accumulate mean matter density in the mass map
    # this mean matter density does not contribute to the convergence in WL
    rho_crit = 2.775362e11 # critical density of the Universe in units of Msol / Mpc2 h -- is actually a constant!
    
    Omega_M = cosmo.Om0 # adjust to Omega_M of simulations
    
    # the present matter density is Omega_M*rho_crit, and it scales like (1+z)^3
    rho_M_z = Omega_M*rho_crit*(1+z_l)**3
    floor = 2*l_proj*rho_M_z # accumulated surface mass density from constant matter density in simulation snapshot
    
    mass_map -= floor
      # fun fact: the mean matter density is accounted for via the Friedman equation 
        # and changes the angular diameter distances
    return mass_map
  

def profile_from_map( groups, group_nbr, snapshot_nbr, n_rbins, nz_path, root=Path("/sps/lsst/users/lbaumont/data/HSC/sims/MassMapsLC30"), pix_size=10/500, l_proj=5):
    # read in n(z)
    z_s_arr, P_zs = get_nz(nz_path)
    
    # mass per pixel
    mass_map_per_pixel = np.load(root / "projected_maps" / f"grnr{group_nbr}_snap{snapshot_nbr}.npy")
    mass_map = mass_map_per_pixel/pix_size**2 *1e10
    assert mass_map.shape == (500, 500)

    # use id to get miscentering and lens redshift info from group catalog
    id = (groups['grnr'] == group_nbr) & (groups['snapshot'] == snapshot_nbr)
    z_l = groups['IterCenZ'][id].value
    miscentering = [groups['dx'][id].value,groups['dy'][id].value]

    # apply Kaiser-Squires algorithm to get shear components in units of Msol/Mpc2 h 
    # -- that is scaled by the critical surface mass density of WL

    E1, E2 = KaiserSquires(mass_map)

    # define a center and get for each pixel the scaled tangential/cross shear, 
    # as well as the position angle and distance from the center of that pixel

    # dx can be adjusted to match the pixel size

    Et, Ex, R, phi = getTangential(E1, E2, miscentering)

    mass_map = remove_projected_convergence(z_l,l_proj)
    
    # to save memory, let us project this into wedges with radial bins
    r_bins, gammat_radial, kappa_radial = compress_gt(mass_map,Et,R,phi,n_rbins)
    
    Sigma_inv_ls = Sigma_crit_inv(z_l, z_s_arr)

    gt_map = Sigma_inv_ls[None, None, :]*gammat_radial[:, :, None]/ \
                  (1 - Sigma_inv_ls[None, None, :]*kappa_radial[:, :, None])

    # perform the azimutal average
    s_crit_inv_eff=weighted_nan_avg(Sigma_inv_ls,weights=P_zs)
    dsig_t_zs = np.nanmean(gt_map, axis=1)/s_crit_inv_eff

    # source redshift average
    dsig_t = np.nansum(dsig_t_zs*P_zs[None, :]*np.diff(z_s_arr)[0], axis=1)

    # I like to plot the radial position with the average distance of point in an anulus
    r_centers = 2/3 * np.diff(r_bins**3)/np.diff(r_bins**2)

    return dsig_t, 1/s_crit_inv_eff   

def load_density_map(root,group_nbr, snapshot_nbr):
    root = Path("/sps/lsst/users/lbaumont/data/HSC/sims/MassMapsLC30")

    density = np.load(root / "density_profile" / f"grnr{group_nbr}_snap{snapshot_nbr}.npy")
    # density in M_sol/(h^-1 kpc)^3
    r_edges = np.logspace(-2, np.log10(5000.0), 500)
    r_mid = 0.5 * (r_edges[:-1] + r_edges[1:])
    assert density.shape == (499,)
    assert r_mid.shape == density.shape
    return density, r_mid

def mass_from_density(r_mid,density):
    """mass from density.
    Integrate density profile to get mass profile
    """
    integrand = 4.0 * np.pi * density * r_mid**2

    mass = np.zeros(len(r_mid), dtype=float)
    # cumulative trapezoidal integration
    mass[1:] = np.cumsum(
        0.5 * (integrand[1:] + integrand[:-1]) * np.diff(r_mid)
    )
    return mass
 







    
    


