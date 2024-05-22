import numpy as np
import random

import sys
sys.path.append('../src')  # Add the 'src' directory to the Python path
from utils import config, constants

class Forcing:
    """Class with some forcing scenarios for barotropic model."""
    def __init__(self, sphere, Si=None):
        """
        Initializes forcing object for barotropic sphere.

        Parameters:
        sphere (Sphere): The sphere object representing the model domain.
        Si (ndarray, optional): Initial state for a stochastic process.
        """
        self.sphere = sphere
        self.Si = Si

    def zero_forcing(self):
        """Zero forcing case."""
        return 0.0

    def evolve_rededdy(self, dt=None, A=1e-11):
        """
        Evolves red eddy forcing represented by an Ornstein-Uhlenbeck process.

        Parameters:
        dt (float, optional): Timestep, required if `Si` is not None.
        A (float, optional): Amplitude of the forcing. Default is 1e-11.

        Returns:
        ndarray: The evolved red eddy forcing in spectral space.
        """
        stir_lat = 40.0  # degrees
        stir_width = 10.0  # degrees
        lat_mask = np.exp(-((np.abs(self.sphere.glats) - stir_lat) / stir_width) ** 2)  # eddy stirring location
        decorr_timescale = 7 * constants.day2sec  # 7 days
        #decorr_timescale = 1e-10

        stirwn = np.where((8 <= self.sphere.specindxn) & (self.sphere.specindxn <= 12), 1, 0)  # force over a set of wavenumbers

        Ai = np.random.normal(0, 1, size=(self.sphere.nspecindx))
        Bi = np.random.normal(0, 1, size=(self.sphere.nspecindx))

        if self.Si is None:  # starting condition
            self.Si = Ai + 1j * Bi
        else:  # evolve condition
            if dt is None:
                raise ValueError("'dt' must be provided if initial 'Si' state is given.")
            self.Si = (Ai + 1j * Bi) * np.sqrt(1 - np.exp(-2 * dt / decorr_timescale)) + np.exp(-dt / decorr_timescale) * self.Si

        return A * self.sphere.to_spectral(self.sphere.to_linear_grid(self.Si * stirwn) * lat_mask)

    def gaussianblob(self, forcing_loc=[50, 160], A=10e-10):
        """
        Gaussian blob forcing timeseries representative of orography or other localized forcing.

        Parameters:
        forcing_loc (list, optional): Location [latitude, longitude] of the Gaussian blob center.
        A (float, optional): Amplitude of the Gaussian blob forcing. Default is 10e-10.

        Returns:
        ndarray: The Gaussian blob forcing in spectral space.
        """
        gauss_forcing = np.zeros(self.sphere.rlons.shape)
        x, y = np.meshgrid(np.linspace(-1, 1, 10), np.linspace(-1, 1, 10))
        d = np.sqrt(x * x + y * y)
        sigma, mu = 0.5, 0.0
        g = np.exp(-((d - mu) ** 2 / (2.0 * sigma ** 2)))  # GAUSSIAN CURVE

        def find_nearest_indx(array, value): return np.abs(array - value).argmin()

        lat_i = find_nearest_indx(self.sphere.glat, forcing_loc[0])
        lon_i = find_nearest_indx(self.sphere.glon, forcing_loc[1])

        gauss_forcing[lat_i:lat_i + 10, lon_i:lon_i + 10] = g * A

        return self.sphere.to_spectral(gauss_forcing)
