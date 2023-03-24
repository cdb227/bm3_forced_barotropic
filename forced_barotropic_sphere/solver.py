import numpy as np
import spharm
import random
from tqdm import tqdm
import xarray as xr

from forced_barotropic_sphere.sphere import Sphere
from forced_barotropic_sphere.dynamics import ForcedVorticity

class Solver:
    """
    Solver class which will integrate the vorticity equation
    """
    def __init__(self,  sphere, forcing, ofreq, ics):
        
        self.sphere = sphere
        
        self.dt = forcing.dt
        self.T = forcing.T
        self.Nt = int(self.T / self.dt)        
        self.ts = np.arange(self.Nt) * self.dt
        
        #number of outputs
        self.No = int(self.Nt/ofreq)+1
        self.ofreq = ofreq
        
        self.vortp = ics[0]
        self.thetap = ics[1]
         
        self.dvortp = np.zeros((self.sphere.nlat,self.sphere.nlon, 3), 'd')  
        self.dthetap = np.zeros((self.sphere.nlat,self.sphere.nlon, 3), 'd')
        
        #output to be saved    
        self.vortpo = np.zeros((self.No, self.sphere.nlat,self.sphere.nlon), 'd')
        self.thetapo= np.zeros((self.No, self.sphere.nlat,self.sphere.nlon), 'd')        
        
        self.FVort = ForcedVorticity(self.sphere, self.vortp, self.thetap, forcing.forcing_tseries)
            
    def integrate_dynamics(self, linear=True):
        """
        Integrating function using RK4(?). By default, only the linear terms are considered for integration
        """
        
        self.FVort.linear = linear
        
        j0 = 0
        k0 = 0 
        k = 0
        self.vortpo[k0, :, :] = self.FVort.vortp
        self.thetapo[k0, :, :] = self.FVort.thetap

        # First two forward steps
        self.dthetap[:,:, 0] = self.FVort.theta_tendency()
        self.FVort.thetap = self.FVort.thetap + self.dt * self.dthetap[:,:, 0]
        
        self.dvortp[:,:, 0] = self.FVort.vort_tendency()
        self.FVort.vortp = self.FVort.vortp + self.dt * self.dvortp[:,:, 0]
        
        
        self.dthetap[:,:, 1] = self.FVort.theta_tendency()
        self.FVort.thetap = self.FVort.thetap + self.dt * self.dthetap[:,:, 1]
        
        self.dvortp[:,:, 1] = self.FVort.vort_tendency()
        self.FVort.vortp = self.FVort.vortp + self.dt * self.dvortp[:,:, 1]
    

        eps = 1e-5 # A bit of damping

        i2 = 0
        i1 = 1
        i0 = 2

        #using the same numerical scheme as Peter, RK4 or something?
        for j, t in enumerate(tqdm(self.ts)):

            self.dthetap[:,:, i0] = self.FVort.theta_tendency()
            self.FVort.thetap  = (1 - eps) * self.FVort.thetap + self.dt / 12. * \
                (23. * self.dthetap[:,:, i0] - 16. * self.dthetap[:, :,i1] + 5. * self.dthetap[:,:, i2])

            self.dvortp[:,:, i0] = self.FVort.vort_tendency()
            self.FVort.vortp = (1 - eps) * self.FVort.vortp + self.dt / 12. * \
                (23. * self.dvortp[:,:, i0] - 16. * self.dvortp[:, :,i1] + 5. * self.dvortp[:,:, i2])

            i0, i1, i2 = i2, i0, i1

            k += 1
            if k >= self.ofreq:
                k0 += 1
                self.vortpo[k0, :, :] = self.FVort.vortp
                self.thetapo[k0, :, :] = self.FVort.thetap
                k = 0
                
        crds = [np.linspace(0, self.T, self.No), self.sphere.glat.data[:], self.sphere.glon.data[:]]
        vort = xr.DataArray(self.vortpo, name = 'vort', coords = crds, dims = ['time', 'y', 'x'])
        theta   = xr.DataArray(self.thetapo, name = 'theta',   coords = crds, dims = ['time', 'y', 'x'])

        return self.sphere.to_flow(vort,theta)
    
        
        

 