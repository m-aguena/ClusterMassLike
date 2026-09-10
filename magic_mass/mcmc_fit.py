import numpy as np
import emcee


class MCMCFit:
    def __init__(
        self,
        full_table,
        delta,
        fit_func,
        cosmo,
        nwalkers,
        use_prior=False,
        clim=(0.1, 20),
        logmlim=(8, 100),
    ):
        self.full_table = full_table
        self.delta = delta
        self.fit_func = fit_func
        self.cosmo = cosmo
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

    def theo(self, logmdelta, cdelta, i):
        return self.fit_func(
            self.full_table.emass["enclosed_mass_radius"][i],
            10**logmdelta,
            cdelta,
            self.full_table["z_l"][i][0],
            self.delta,
            self.cosmo,
        )

    def prior(self, logmdelta, i):
        return (
            -0.5
            * (
                (
                    logmdelta * np.log(10)
                    - np.log(self.full_table[f"M{self.delta}_crit"][i])
                )
                / (
                    self.full_table[f"M{self.delta}_crit_err"][i]
                    / self.full_table[f"M{self.delta}_crit"][i]
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

        diff = self.theo(logmdelta, cdelta, i) - self.full_table["ds_t_pc2"][i]
        sig = self.full_table["ds_err_pc2"][i]

        lnlike = -0.5 * ((diff / sig) ** 2).sum()

        if self.use_prior:
            lnlike += self.prior(logmdelta, i)

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
