import numpy as np
import random


###################################################
# Define physical constants

a = 6371e3         # Radius of the earth in m
Omega = 7.29e-5    # Angular velocity of the earth in rad/s
g00 = 9.81         # Acceleration due to gravity near the surface of the earth in m/s^2
d2r = np.pi / 180. # Factor to convert degrees to radians
r2d = 180. / np.pi # Factor to convert radians to degrees
d2s = 86400


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
            forcing_tseries[tt,:]= A*lat_mask*self.sphere.to_linear_grid(np.real(W*self.sphere.to_spectral(np.exp(1.j*wn_forcing*self.sphere.rlons))))
            
        self.forcing_tseries=forcing_tseries
        
    def generate_rededdy_start(self):

        Ai= np.random.normal(0, 1, size=(self.sphere.nspecindx))
        Bi= np.random.normal(0, 1, size=(self.sphere.nspecindx))
        
        Si = (Ai+1j*Bi)
        
        return Si
       
    def generate_rededdy_tseries(self, Si, A=1e-11):
        """Generate a forcing timeseries in grid space of length T"""
        #Red Eddys represented by O-U stochastic process
        forcing_tseries = np.zeros((self.Nt+2,len(self.sphere.glat),len(self.sphere.glon)), 'd')
                                   
        stir_lat = 40. #degrees
        stir_width = 10. #degrees
        lat_mask = np.exp(- ((np.abs(self.sphere.glats)-stir_lat)/stir_width)**2 ) #eddy stirring location
        decorr_timescale = 7*d2s #2 days
                
        stirwn = np.where((8<=self.sphere.specindxn) & (self.sphere.specindxn<=12) ,1,0) #force over a set of wavenumbers
        
        
        forcing_tseries[0,:] = self.sphere.to_linear_grid(Si*stirwn)
        for tt in range(1,self.Nt+1):
            Ai= np.random.normal(0, 1, size=(self.sphere.nspecindx))
            Bi= np.random.normal(0, 1, size=(self.sphere.nspecindx))
            
            Si = (Ai+1j*Bi)*(1-np.exp(-2*self.dt/decorr_timescale))**(0.5) + np.exp(-self.dt/decorr_timescale)*Si
            
            forcing_tseries[tt,:]=  self.sphere.to_linear_grid((Si*stirwn))
                    
            
        self.forcing_tseries = A*forcing_tseries*lat_mask[None,:,:]
        return self.forcing_tseries
        
    
    def generate_gaussianblob_tseries(self,forcing_loc=[50,160]):
        """
        Gaussian blob forcing timeseries representative of orography or something
        """
        forcing_tseries = np.zeros((self.Nt+2,len(self.sphere.glat),len(self.sphere.glon)))
        A = 10e-10
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
     
    def generate_zeroforcing_tseries(self):
        """
        Zero forcing case
        """
        forcing_tseries = np.zeros((self.Nt+2,len(self.sphere.glat),len(self.sphere.glon)))
        
        self.forcing_tseries=forcing_tseries
        

 