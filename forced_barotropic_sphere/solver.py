import numpy as np
from tqdm import tqdm
import xarray as xr

from forced_barotropic_sphere.dynamics import ForcedVorticity

class Solver:
    """
    Solver class which will integrate the vorticity equation
    """
    def __init__(self,  sphere, forcing, ofreq):
        """
        initializes a solver for barotropic model.
        
        Arguments:
        * sphere (Sphere object) : contains spectral methods and grid for integration
        * forcing (Forcing object) : contains timestep information and forcing to be applied to sphere
        * ofreq (int) : return model output every ofreq-th step
        """
        
        self.sphere = sphere
        
        self.dt = forcing.dt
        self.T = forcing.T
        self.Nt = int(self.T / self.dt)        
        self.ts = np.arange(self.Nt) * self.dt
        
        self.No = int(self.Nt/ofreq)+1
        self.ofreq = ofreq
         
        #temporary storage for integration
        self.dvortp = np.zeros((self.sphere.nlat,self.sphere.nlon, 3), 'd')  
        self.dthetap = np.zeros((self.sphere.nlat,self.sphere.nlon, 3), 'd')
        
        #output to be saved    
        self.vortpo = np.zeros((self.No, self.sphere.nlat,self.sphere.nlon), 'd')
        self.thetapo= np.zeros((self.No, self.sphere.nlat,self.sphere.nlon), 'd')        
        
        self.FVort = ForcedVorticity(self.sphere, forcing.forcing_tseries)
            
    def integrate_dynamics(self, linear=True):
        """
        Integrating function using RK4(?). By default, only the linear terms are considered for integration
        """
        
        self.FVort.linear = linear
        
        j0 = 0
        k0 = 0 
        k = 0
        self.vortpo[k0, :, :] = self.sphere.vortp
        self.thetapo[k0, :, :] = self.sphere.thetap

        # First two forward steps
        self.dthetap[:,:, 0] = self.FVort.theta_tendency()
        self.sphere.thetap = self.sphere.thetap + self.dt * self.dthetap[:,:, 0]
        
        self.dvortp[:,:, 0] = self.FVort.vort_tendency()
        self.sphere.vortp = self.sphere.vortp + self.dt * self.dvortp[:,:, 0]
        
        
        self.dthetap[:,:, 1] = self.FVort.theta_tendency()
        self.sphere.thetap = self.sphere.thetap + self.dt * self.dthetap[:,:, 1]
        
        self.dvortp[:,:, 1] = self.FVort.vort_tendency()
        self.sphere.vortp = self.sphere.vortp + self.dt * self.dvortp[:,:, 1]
    

        eps = 1e-5 # A bit of damping

        i2 = 0
        i1 = 1
        i0 = 2

        #using the same numerical scheme as Peter, RK4 or something?
        for j, t in enumerate(tqdm(self.ts)):

            self.dthetap[:,:, i0] = self.FVort.theta_tendency()
            self.sphere.thetap  = (1 - eps) * self.sphere.thetap + self.dt / 12. * \
                (23. * self.dthetap[:,:, i0] - 16. * self.dthetap[:, :,i1] + 5. * self.dthetap[:,:, i2])

            self.dvortp[:,:, i0] = self.FVort.vort_tendency()
            self.sphere.vortp = (1 - eps) * self.sphere.vortp + self.dt / 12. * \
                (23. * self.dvortp[:,:, i0] - 16. * self.dvortp[:, :,i1] + 5. * self.dvortp[:,:, i2])

            i0, i1, i2 = i2, i0, i1

            k += 1
            if k >= self.ofreq:
                k0 += 1
                self.vortpo[k0, :, :] = self.sphere.vortp
                self.thetapo[k0, :, :] = self.sphere.thetap
                k = 0
                
        crds = [np.linspace(0, self.T, self.No), self.sphere.glat.data[:], self.sphere.glon.data[:]]
        vort = xr.DataArray(self.vortpo, name = 'vort', coords = crds, dims = ['time', 'y', 'x'])
        theta   = xr.DataArray(self.thetapo, name = 'theta',   coords = crds, dims = ['time', 'y', 'x'])

        return self.to_flow(vort,theta)
    
    def to_flow(self, vort, theta):
        """
        Compute u, v, vort, theta from vortp, thetap solution
        """
        N, Ny, Nx = vort.shape
        uo   = np.zeros((N, Ny, Nx), 'd')
        vo   = np.zeros((N, Ny, Nx), 'd')
        #vortm,_= self.uv2vrtdiv(self.U,self.V)

        for i in range(N):
            uo[i],vo[i] = self.sphere.vrtdiv2uv(vort[i].values, self.sphere.vortp_div)
            uo[i] = self.sphere.U + uo[i]
            #vo[i] = vo


        crds = [vort.time[:], vort.y[:], vort.x[:]]
        vortp = vort.rename('vortp')
        vort = (vortp + self.sphere.vortm).rename('vort')
        
        thetap = theta.rename('thetap')
        theta = (thetap + self.sphere.thetaeq).rename('theta')
        
        #psi = xr.DataArray(psio, name = 'psi', coords = crds, dims = ['time', 'y', 'x'])
        u   = xr.DataArray(uo, name = 'u',   coords = crds, dims = ['time', 'y', 'x'])
        v   = xr.DataArray(vo, name = 'v',   coords = crds, dims = ['time', 'y', 'x'])
        return xr.Dataset(data_vars = dict(vort=vort, vortp=vortp, u=u, v=v, thetap=thetap, theta=theta))
    
        
        

 