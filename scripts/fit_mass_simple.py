import numpy as np
import emcee

from cluster_mass_like.likelihoods.x_ray.xray_simple import ln_prob_mwl
from cluster_mass_like.likelihoods.weak_lensing.wl_simple import ln_prob_mwl_mtrue
from cluster_mass_like.likelihoods.weak_lensing.wl_true_simple import (
    ln_prob_mxray_mtrue,
)


class LnProb:
    def __init__(self):
        # number of parameters to fit
        self.npars = 3

        # adding dummy values
        self.mass_wl = 14.0
        self.mass_xray = 14.0
        self.mass_true = 14.0
        self.parameters_wl = [None]
        self.parameters_wl_tr = [None, None]
        self.parameters_xray = [3.0]

        # data
        self.data = None
        self.icov = None  # inverse of covariance matrix

    def prepare(self):
        pass

    def read_data(self):
        """Reads data and returns blah"""
        self.data = None
        self.icov = None

    def lnlike(self):
        return (
            ln_prob_mwl(self.mass_wl, self.parameters_wl)
            + ln_prob_mwl_mtrue(self.mass_wl, self.mass_true, self.parameters_wl_tr)
            + ln_prob_mxray_mtrue(self.mass_xray, self.mass_true, self.parameters_xray)
        )

    def __call__(self, par1, par2):
        # set parameters
        self.parameters_wl[0] = par1
        self.parameters_wl_tr[1] = par2

        # compute ln(like)

        diff = self.data - self.lnlike()

        return diff @ self.icov @ self.data


if __name__ == "__main__":
    # start and prep likelihood

    lnprob = LnProb()
    lnprob.prepare()
    lnprob.read_data()

    # setup for mcmc

    nwalkers = 64
    nsteps = 100
    initial_guess = [1, 1]
    initial_spread = [0.1, 0.1]

    # run mcmc

    sampler = emcee.EnsembleSampler(nwalkers, lnprob.npars, lnprob)
    sampler.run_mcmc(
        np.random.normal(initial_guess, initial_spread, (nwalkers, lnprob.npars)),
        nsteps,
        progress=True,
    )

    # save
    chain = sampler.get_chain()
    log_prob = sampler.get_log_prob()
    np.save("mcmc_fit.npy", np.vstack([chain, log_prob]))
