import numpy as np

import pyccl
import clmm


def nfw_int(x):
    return np.log(1 + x) - (x / (1 + x))


def convert_mass(M1, c1, c2):
    return M1 * nfw_int(c2) / nfw_int(c1)


class ProfileFit:
    def __init__(self):
        self.biases = {}

    @staticmethod
    def func(radius, mdelta, cdelta, z, delta, cosmo):
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

    def bias(self, mdelta, z, delta, cosmo):
        if delta not in self.biases:
            self.biases[delta] = pyccl.halos.HaloBiasTinker10(mass_def=f"{delta}c")

        return self.biases[delta](cosmo.be_cosmo, mdelta, 1 / (1 + z))

    def just2h(self, radius, mdelta, z, delta, cosmo):
        return clmm.compute_excess_surface_density_2h(
            radius,
            z,
            cosmo,
            halobias=self.bias(mdelta, z, delta),
        )

    def func2h(self, radius, mdelta, cdelta, z, delta, cosmo):
        if cdelta < 0:
            return radius * 0

        return self.func(radius, mdelta, cdelta, z, delta, cosmo) + self.just2h(
            radius, mdelta, z, delta, cosmo
        )
