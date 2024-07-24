import numpy as np
import random

from utils import config, constants

class Forcing:
    """Class with some forcing scenarios for barotropic model."""
    def __init__(self, sphere, **kwargs):
        """
        Initializes forcing object for barotropic sphere.

        Parameters:
        sphere (Sphere): The sphere object representing the model domain.
        Si (ndarray, optional): Initial state for a stochastic process.
        """
        
        self.sphere = sphere
        
        #forcing type
        self.forcing_type = kwargs.get('forcing_type', config.DEFAULT_forcing_type)
        self.A =            kwargs.get('forcing_A', config.DEFAULT_forcing_A) #forcing amplitude
        
        #for rededdy
        self.Si     = kwargs.get('red_eddy_start', self.gen_Si())
        self.dt     = kwargs.get('dt', config.DEFAULT_dt)
                     
        #for gaussian blob
        self.blob_center = kwargs.get('blob_center', [60, 160])
        

    def gen_Si(self):
        Ai = np.random.normal(0, 1, size=(self.sphere.nspecindx))
        Bi = np.random.normal(0, 1, size=(self.sphere.nspecindx))
        return Ai + 1j * Bi 
            
    def evolve_forcing(self):
        
        if self.forcing_type=='zero_forcing':
            self.force_term = self.zero_forcing()
        elif self.forcing_type=='gaussian_blob':
            self.force_term = self.gaussian_blob()
        elif self.forcing_type=='rededdy':
            self.force_term = self.evolve_rededdy()
        else:
            raise ValueError('forcing type not recognized')
            
        return self.force_term

    def zero_forcing(self):
        """Zero forcing case."""
        return 0.0

    def evolve_rededdy(self):
        """
        Evolves red eddy forcing represented by an Ornstein-Uhlenbeck process.

        Returns:
        ndarray: The evolved red eddy forcing in spectral space.
        """
        stir_lat = 40.0  # degrees
        stir_width = 10.0  # degrees
        lat_mask = np.exp(-((np.abs(self.sphere.glats) - stir_lat) / stir_width) ** 2)  # eddy stirring location
        decorr_timescale = 7 * constants.day2sec  # 7 days

        stirwn = np.where((8 <= self.sphere.specindxn) & (self.sphere.specindxn <= 12), 1, 0)  # force over a set of wavenumbers

        self.Si = self.gen_Si() * np.sqrt(1 - np.exp(-2 * self.dt / decorr_timescale)) + np.exp(-self.dt / decorr_timescale) * self.Si

        return self.A * self.sphere.to_spectral(self.sphere.to_linear_grid(self.Si * stirwn) * lat_mask)

    def gaussian_blob(self):
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

        lat_i = find_nearest_indx(self.sphere.glat, self.blob_center[0])
        lon_i = find_nearest_indx(self.sphere.glon, self.blob_center[1])

        gauss_forcing[lat_i:lat_i + 10, lon_i:lon_i + 10] = g * self.A

        return self.sphere.to_spectral(gauss_forcing)
