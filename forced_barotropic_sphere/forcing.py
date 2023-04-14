import numpy as np
import random


###################################################
# Define physical constants

a = 6371e3         # Radius of the earth in m
Omega = 7.29e-5    # Angular velocity of the earth in rad/s
g00 = 9.81         # Acceleration due to gravity near the surface of the earth in m/s^2
d2r = np.pi / 180. # Factor to convert degrees to radians
r2d = 180. / np.pi # Factor to convert radians to degrees
Tau = 6*86400


### Class which solves a forced vorticity equation using spherical harmonics
class Forcing:
    def __init__(self, sphere, dt, T):
        """
        initializes sphere for barotropic model.
        
        Arguments:
        * nlat (int) : number of latitude points
        * nlon (int) : number of longitude points
        * U (int/float/array) : background zonal mean wind
        * theta0 (float) : equator temperature
        * deltheta (float): amplitude of equator-to-pole gradient
        * rsphere (float) : radius of sphere
        """

        self.sphere = sphere
        self.dt = dt
        self.T = T
        self.Nt = int(self.T / self.dt)

    
    def generate_stocheddy_tseries(self,A=8e-10):
        """Generate a forcing timeseries in grid space of length T"""
        #TODO: generating these t-series of forcing is ugly but will likely be useful for the ensemble cases
        # we want each member to share (some amount) of information about the forcing
        forcing_tseries = np.zeros((self.Nt+2,len(self.sphere.glat),len(self.sphere.glon)))
                                   
        stir_lat = 40. #degrees
        stir_width = 10. #degrees
        
        lat_mask = np.exp(- ((np.abs(self.sphere.glats)-stir_lat)/stir_width)**2 ) #eddy stirring location
        
        wn_forcing = 6
        
        for tt in range(1,self.Nt+1):
            W= np.random.normal(0,1, size = (self.sphere.nspecindx, 2)).view(np.complex128).ravel() #complex white noise
            forcing_tseries[tt,:]= A*lat_mask*self.sphere.to_grid(np.real(W*self.sphere.to_spectral(np.exp(1.j*wn_forcing*self.sphere.rlons))))
            
        self.forcing_tseries=forcing_tseries
    
    def generate_gaussianblob_tseries(self,forcing_loc=[50,160]):
        forcing_tseries = np.zeros((self.Nt+2,len(self.sphere.glat),len(self.sphere.glon)))
        A = 10e-10
  # s**-2
        gauss_forcing = np.zeros(self.sphere.rlons.shape)
        x, y = np.meshgrid(np.linspace(-1,1,10), np.linspace(-1,1,10))
        d = np.sqrt(x*x+y*y)
        sigma, mu = 0.5, 0.0
        g = np.exp(-( (d-mu)**2 / ( 2.0 * sigma**2 ) ) )   # GAUSSIAN CURVE
            
        def find_nearest_indx(array,value): return np.abs(array-value).argmin()

        lat_i = find_nearest_indx(self.sphere.glat,forcing_loc[0])
        lon_i = find_nearest_indx(self.sphere.glon,forcing_loc[1])

        gauss_forcing[lat_i:lat_i+10, lon_i:lon_i+10] = g*A
        
        forcing_tseries[:] = gauss_forcing
        
        self.forcing_tseries=forcing_tseries
        #return forcing_tseries
    
        
    def generate_forcing(self):
        """Generate spectral forcings for the vorticity equation
        
        Returns:
            A forcing in the spectral domain"""
        
        if self.forcing_type == "gaussian":
            # gaussian bubble centered at forcing_loc
            A = 10e-10  # s**-2
            gauss_forcing = np.zeros(self.rlons.shape)
            x, y = np.meshgrid(np.linspace(-1,1,10), np.linspace(-1,1,10))
            d = np.sqrt(x*x+y*y)
            sigma, mu = 0.5, 0.0
            g = np.exp(-( (d-mu)**2 / ( 2.0 * sigma**2 ) ) )   # GAUSSIAN CURVE
            
            def find_nearest_indx(array,value): return np.abs(array-value).argmin()
            
            lat_i = find_nearest_indx(self.glat,self.forcing_loc[0])
            lon_i = find_nearest_indx(self.glon,self.forcing_loc[1])
            
            gauss_forcing[lat_i:lat_i+10, lon_i:lon_i+10] = g*A
            self.forcing_grid = gauss_forcing
            self.forcing_spectral = self.sphere.to_spectral(self.forcing_grid)
            return self.forcing_grid
            
        elif self.forcing_type == "stochastic_eddy":
            #*(self.dt)#**(-0.5)
            stir_lat = 40. #degrees
            stir_width = 10. #degrees

            wn_forcing = 6
            
            #essentially a latitude mask for the eddy stirring location
            #lat_mask = np.exp(- (np.abs(self.glats)-stir_lat)/stir_width )
            lat_mask = np.exp(- ((np.abs(self.glats)-stir_lat)/stir_width)**2 )
            
            W= np.random.normal(0,1, size = (self.nspecindx, 2)).view(np.complex128).ravel()
            #self.white_noise = W*(1.-np.exp(-2*self.dt/stir_memory))**(1/2.) + np.exp(-self.dt/stir_memory)*self.white_noise
            
            F= self.eddy_A*lat_mask*self.sphere.to_grid(np.real(W*self.sphere.to_spectral(np.exp(1.j*wn_forcing*self.rlons))))
            
            self.forcing_spectral = self.sphere.to_spectral(F)
            
            self.forcing_grid = F
            
            return self.forcing_grid
        
        elif self.forcing_type=="chen_eddy":
            A=1e-8*(2*self.dt)**(-0.5)
            stir_lat = 40. #degrees
            stir_width = 10. #degrees
            
            wn_forcing = 6
            #essentially a latitude mask for the eddy stirring location
            lat_mask = np.exp(- (np.abs(self.glats)-stir_lat)/stir_width )
            
            #we'll implement a simple e-folding time for the white noise memory?
            W= np.random.normal(0,1, size = (self.nspecindx, 2)).view(np.complex128).ravel()
            
            self.forcing_grid= A*lat_mask*self.sphere.to_grid(np.real(W*self.sphere.to_spectral(np.exp(1.j*wn_forcing*self.rlons))))
            
            self.forcing_spectral = self.sphere.to_spectral(self.forcing_grid)
            
            return self.forcing_grid

            
        elif self.forcing_type == 'None':
            return 0.
            
        else:
            raise ValueError('Forcing type is not defined, must be in...["gaussian","orography","stochastic_eddy", "None"]')        
        

 