import astropy.units as u
from astropy.table import Table
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

        M = self.radius**2 * integral

        return g, M

    def add_emass_table(self, table, bootstrapped=False, outname="emass"):
        out_table = Table()

        if not bootstrapped:
            out_table["magic_mass"] = self.measure_gt_mass(
                theta=table["rp"],  # Mpc
                # change units of ds from M_sol*Mpc**2 to M_sol*pc**2
                DS=table["ds_t_pc2"],
                # errs = table["ds_err_pc2"].T
            )[1]
            out_table["enclosed_mass_radius"] = (
                np.ones(len(table))[:, None] * self.radius[None, :]
            )

        else:
            out_table["magic_mass"] = np.array(
                [
                    self.measure_gt_mass(theta, DS, plot=False)[1]
                    for theta, DS in zip(
                        table["rp"].transpose(2, 0, 1),
                        table["ds_t_pc2"].transpose(2, 0, 1),
                    )
                ]
            ).transpose(1, 2, 0)

            out_table["gt"] = np.array(
                [
                    self.measure_gt_mass(theta, DS, plot=False)[0]
                    for theta, DS in zip(
                        table["rp"].transpose(2, 0, 1),
                        table["ds_t_pc2"].transpose(2, 0, 1),
                    )
                ]
            ).transpose(1, 2, 0)

            out_table["enclosed_mass_radius"] = (
                np.ones(len(table))[:, None] * self.radius[None, :]
            )

        setattr(table, outname, out_table)
