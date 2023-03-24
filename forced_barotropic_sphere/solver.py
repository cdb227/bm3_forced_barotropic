import numpy as np
import spharm
import random
from tqdm import tqdm
import xarray as xr

from forced_barotropic_sphere.sphere import Sphere
from forced_barotropic_sphere.dynamics import ForcedVorticity

### Class which will integrate the forced vorticity equation and apply stirring to theta field
class Solver:
    def __init__(self,  sphere, tstep, T, ofreq, ics, forcing_tseries):#forcing_type = 'None', forcing_loc=[35,160], eddy_A=8e-10):
        
        self.sphere = sphere
        #self.forcing_type = forcing_type
        #self.forcing_loc = forcing_loc
        #self.eddy_A = eddy_A
        
        self.T = T
        self.Nt = int(self.T / tstep)
        self.dt = tstep
        self.ts = np.arange(self.Nt) * tstep
        
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
        
        self.FVort = ForcedVorticity(self.sphere, self.vortp, self.thetap, self.dt, forcing_tseries)#self.forcing_type, self.forcing_loc, self.eddy_A)
            
    def integrate_dynamics(self, nonlinear=False):
        self.FVort.nonlinear = nonlinear
        
        
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
        #return vort
        return self.sphere.to_flow(vort,theta)
    
        
        

 