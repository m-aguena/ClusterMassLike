class HaloMassFunction:
    def __init__(self, **kwargs):
        self._cosmo = None
        self._hmf = None

    def set_cosmo(self, **kwargs):
        raise NotImplementedError("Function not implemented in parent class")

    def set_hmf(self, **kwargs):
        raise NotImplementedError("Function not implemented in parent class")

    def dndlnm(self, mass, redshift):
        raise NotImplementedError("Function not implemented in parent class")
