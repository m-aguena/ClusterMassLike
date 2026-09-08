import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
from astropy.constants import G


class MagicMass:
    def __init__(self, radius=None):
        self.phi = np.linspace(0.00001, np.pi / 2, num=200)

        # is this radius definition okay? What are the units?
        self.radius = np.logspace(-1, 0.5, num=10)

        self.dphi = np.diff(self.phi)[0]
        self.rsinphi = self.radius[:, None] / np.sin(self.phi[None, :])

    def measure_gt_mass(self, theta, DS, plot=True):
        if plot:
            plt.plot(self.phi, 1 / np.sin(self.phi), "*")
            plt.show()

        # get delta_sigma values at r=r/sin(phi)
        integrand = np.array(
            [
                np.interp(np.log(self.rsinphi), np.log(theta[i]), DS[i])
                for i in range(DS.shape[0])
            ]
        )

        # rsinphi.max()
        # integrand.shape

        if plot:
            plt.pcolor(np.log10(integrand[-6]))
            plt.colorbar()
            plt.show()

        integral = 4 * np.sum(integrand, axis=-1) * self.dphi

        g = integral * u.solMass / u.Mpc**2
        g = g.to(u.kg / u.m**2) * G

        M = self.r**2 * integral

        return g, M
