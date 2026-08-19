import argparse
import numpy as np
from scipy.integrate import simpson
import emcee
import sys
from astropy.table import Table
import time

sys.path.append("/pbs/home/m/maguena/git_codes/ClusterMassLike/")
print(sys.path)


from cluster_mass_like.likelihoods.x_ray.xray_powerlaw import ln_prob_mxray_mtrue  # noqa: E402
from cluster_mass_like.likelihoods.weak_lensing.wl_true_powerlaw import (  # noqa: E402
    ln_prob_mwl_mtrue,
)
from cluster_mass_like.cluster_model.halo_mass_function.ccl import CCLHaloMassFunction  # noqa: E402


class LnProb:
    def __init__(self):
        # number of parameters to fit
        self.npars = 2

        # adding dummy values
        self.parameters_wl = [0.0, 1.0, 1.0]
        self.parameters_xray = [0.0, 1.0, 1.0]

        self.hmf = CCLHaloMassFunction()

        # integration values
        self._integ_logm = np.linspace(13, 16, 51)[:, None]

        # data
        self.sample_logm_wl = None
        self.sample_logm_xray = None
        self.sample_redshift = None
        self.icov = None  # inverse of covariance matrix

        # internal to speed up computations
        self._hmf_tabulated = None

    def prepare(self):
        print("Seting up HMF")
        self.hmf.set_cosmo()
        self.hmf.set_hmf()

        # tabulate hmf
        print("  Tabulating HMF")
        self._hmf_tabulated = self.hmf.dndlnm(
            10**self._integ_logm, self.sample_redshift
        )
        print(self._hmf_tabulated.shape)

    def read_data(self, catalog_path, wl_col, xray_col, z_col):
        """Reads data and returns blah"""
        print("Reading data")
        data = Table.read(catalog_path)
        self.sample_logm_wl = np.log10(data[wl_col])[None, :]
        self.sample_logm_xray = np.log10(data[xray_col])[None, :]
        self.sample_redshift = data[z_col]
        self.icov = np.ones((len(data), len(data)))
        print(f"  {len(data):,} clusters read")

    def lnlike(self):
        # sort out dimentions here, should be (mtrue, nsample)
        _lnlike_kenel = (
            self._hmf_tabulated
            + ln_prob_mwl_mtrue(
                self.sample_logm_wl, self._integ_logm, self.parameters_wl
            )
            + ln_prob_mxray_mtrue(
                self.sample_logm_xray, self._integ_logm, self.parameters_xray
            )
        )

        return np.log(simpson(np.exp(_lnlike_kenel), x=self._integ_logm, axis=0))

    def __call__(self, pars):
        # set parameters
        self.parameters_wl[1] = pars[0]
        self.parameters_wl[1] = pars[1]

        # compute ln(like)

        lnlike_per_cluster = self.lnlike()

        return lnlike_per_cluster @ self.icov @ lnlike_per_cluster


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--catalog",
        "-cat",
        type=str,
        help="Catalog with full path",
        default="/sps/euclid/Users/maguena/working/fgas/data/weak lensing/cluster_cat_newcenters.fits",
    )
    parser.add_argument(
        "--column_wl",
        "-col_wl",
        type=str,
        help="Weak lensing column in the catalog",
        default="MassA",
    )
    parser.add_argument(
        "--column_xray",
        "-col_xray",
        type=str,
        help="x-ray column in the catalog",
        default="mass_r_lum",
    )
    parser.add_argument(
        "--column_z",
        "-col_z",
        type=str,
        help="Redshift column in the catalog",
        default="Zfof_1",
    )
    parser.add_argument(
        "--nwalkers",
        type=int,
        help="emcee number of walkers",
        default=64,
    )
    parser.add_argument(
        "--nsteps",
        type=int,
        help="emcee number of steps",
        default=100,
    )
    args = parser.parse_args()

    # start and prep likelihood

    lnprob = LnProb()
    lnprob.read_data(args.catalog, args.column_wl, args.column_xray, args.column_z)
    t0 = time.time()
    lnprob.prepare()
    print(f"  {time.time() - t0:.2f} sec.")

    # setup for mcmc
    initial_guess = [1, 1]
    initial_spread = [0.1, 0.1]

    # run mcmc

    print("Running MCMC")
    sampler = emcee.EnsembleSampler(args.nwalkers, lnprob.npars, lnprob)
    sampler.run_mcmc(
        np.random.normal(initial_guess, initial_spread, (args.nwalkers, lnprob.npars)),
        args.nsteps,
        progress=True,
    )

    # save
    print("Saving chain")
    chain = sampler.get_chain()
    log_prob = sampler.get_log_prob()
    np.save("mcmc_fit.npy", np.vstack([chain.transpose(2, 0, 1), [log_prob]]))
