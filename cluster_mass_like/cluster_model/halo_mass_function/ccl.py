import numpy as np

import pyccl

from .parent_class import HaloMassFunction


class CCLHaloMassFunction(HaloMassFunction):
    def __init__(self, **kwargs):
        self._cosmo = None
        self._hmf = None
        self._cache: dict[tuple[float, float], float] = {}

    def set_hmf(self, **kwargs):
        # set halo mass function
        if "hmf" in kwargs:
            self._hmf = kwargs["hmf"]
        else:
            self._hmf = pyccl.halos.MassFuncBocquet16()
        self._cache = {}

    def set_cosmo(self, **kwargs):
        self._cosmo = pyccl.CosmologyVanillaLCDM(**kwargs)
        self._cache = {}

    def _dndlnm_cache(self, mass, redshift):
        """
        Halo mass function for scalars


        Parameters
        ----------
        mass: float
            Halo mass in Msun
        redshift: float
            Halo redshift

        Return
        ------
        float
            Halo mass function, shape (mass.shape, redshift.shape)
        """

        val = self._hmf_cache.get((mass, redshift))
        if val is None:
            _scale_factor = 1.0 / (1.0 + redshift)
            val = self.halo_mass_function(self.cosmo, mass, _scale_factor)
            self._hmf_cache[(mass, redshift)] = val

        return val

    def dndlnm(self, mass, redshift):
        """
        Halo mass function


        Parameters
        ----------
        mass: np.ndarray
            Halo mass in Msun
        redshift: np.ndarray
            Halo redshift

        Return
        ------
        np.ndarray
            Halo mass function, shape (mass.shape, redshift.shape)
        """
        return np.array(
            [
                [
                    self._dndlnm_cache(m_mdim, z_mdim)
                    for m_mdim, z_mdim in zip(m_zdim, z_zdim)
                ]
                for m_zdim, z_zdim in zip(*np.meshgrid(mass, redshift))
            ]
        )
