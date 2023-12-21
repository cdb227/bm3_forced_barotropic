import numpy as np
from tqdm import tqdm
import xarray as xr

from forced_barotropic_sphere.dynamics import ForcedVorticity

class Solver:
    """
    Solver class which will integrate the vorticity equation
    """
    def __init__(self,  sphere, forcing, ofreq, fvort_args=None):
        """
        initializes a solver for barotropic model.
        
        Arguments:
        * sphere (Sphere object) : contains spectral methods and grid for integration
        * forcing (Forcing object) : contains timestep information and forcing to be applied to sphere
        * ofreq (int) : return model output every ofreq-th step
        """
        
        self.sphere = sphere
        
        self.dt = forcing.dt                   #integration step
        self.T = forcing.T                     #integration time
        self.Nt = int(self.T / self.dt)        #number of steps   
        self.ts = np.arange(self.Nt) * self.dt #integration times
        
        self.No = int(self.Nt/ofreq)+1         #number of outputs
        self.ofreq = ofreq                     #output frequency
         
        #temporary storage for integration
        self.dvortp = np.zeros((self.sphere.nlat,self.sphere.nlon, 1), 'd')  
        self.vortp = np.zeros((self.sphere.nlat,self.sphere.nlon, 2), 'd')  
        
        
        #output to be saved    
        self.vortpo = np.zeros((self.No, self.sphere.nlat,self.sphere.nlon), 'd')
        self.thetapo= np.zeros((self.No, self.sphere.nlat,self.sphere.nlon), 'd')
        
        
                
        self.nu = 1e-4
        self.diffusion_order=1
        
        damping_coefficient = 1e4
        damping_order = 1
        
        m, n = self.sphere.specindxm, self.sphere.specindxn
        #el = (m + n) * (m + n + 1) / float(self.sphere.s.rsphere) ** 2
#         self.damping = damping_coefficient * \
#                        (el / el[self.sphere._ntrunc]) ** damping_order
        el = ((n) * (n + 1) / float(self.sphere.s.rsphere) ** 2).astype(np.complex64, casting="same_kind")
        self.damping = damping_coefficient * el**damping_order

        
            
        
        self.FVort = ForcedVorticity(self.sphere, forcing.forcing_tseries, **fvort_args)
            
    def integrate_dynamics(self, temp_linear=True, vort_linear=True):
        """
        Integrating function using leapfrog. By default, only the linear terms are considered for integration
        """
        
        self.FVort.temp_linear = temp_linear
        self.FVort.vort_linear = vort_linear
        
        k0 = 0 
        k = 0
        
        #tendencies in spectral space
        dz = np.zeros((self.sphere.nspecindx, 1), dtype='complex')
        
        #vorticity in grid space
        #old, now, new
        z = np.zeros((self.sphere.nlat,self.sphere.nlon, 3), 'd')
        
        #set old, new vorticity to ics
        z[:,:,0] = self.sphere.vortp
        z[:,:,1] = self.sphere.vortp
        
        self.vortpo[k0, :, :] = z[:, :,0]
        
        
        # Robert-Asselin filter strength
        r = 0.04

        i0 = 0
        
        #pointers
        jold,jnow,jnew = 0,1,2
        
        for j, t in enumerate(tqdm(self.ts)):
            
            # Step 1/2: Compute (f + ζ)u and (f + ζ)v on the grid at time t
            #grid space
            u,v = self.sphere.vrtdiv2uv(z[:,:,jnow], np.zeros(z[:,:,jnow].shape)) #divergence-less flow
            
            du =  (self.sphere.f + z[:,:,jnow])*v
            dv = -(self.sphere.f + z[:,:,jnow])*u

            # Step 3: Compute the spectral divergence of du,dv to find dz
            #converts to spectral space
            dz[:,i0], _ = self.sphere.s.getvrtdivspec(du, dv)
                                    
            
            #compute damping
            zs = self.sphere.to_spectral(z)
            coeffs = 1. / (1. + self.damping * self.dt)
            if j==0: coeffs/2. #for dt difference in first step
            #apply damping
            dz[:,i0] = coeffs * (dz[:,i0]  - self.damping * zs[:,jold] )

            
            if j==0: #simple eularian first forward step
                zs[:,jnew] = zs[:, jnow] + self.dt * dz[:,i0]
        
            else: #otherwise leapfrog
                
                #Step 5 Use leapfrog to generate the spectral vorticity ζ(t + ∆t)
                zs[:, jnew]  = zs[:,jold] + 2*self.dt*dz[:,i0]
                
                #5.1 apply robert filter
                zs[:, jnow] += r*(zs[:,jold] - 2*zs[:,jnow])             
                zs[:, jnow] += r*zs[:,jnew]
    
            #finally convert back to grid space and roll
            # 1:   new->old, old->now, now->new
            #-1:   old->new, new->now, now->old
            z = self.sphere.to_grid(zs)
            z = np.roll(z, axis=1, shift=-1)
            
            

            k += 1
        
            if k >= self.ofreq:
                k0 += 1
                self.vortpo[k0, :, :] = z[:, :,jnow]
                k = 0
                
        crds = [np.linspace(0, self.T, self.No), self.sphere.glat.data[:], self.sphere.glon.data[:]]
        vort = xr.DataArray(self.vortpo, name = 'vort', coords = crds, dims = ['time', 'y', 'x'])
        theta   = xr.DataArray(self.thetapo, name = 'theta',   coords = crds, dims = ['time', 'y', 'x'])

        return self.to_flow(vort,theta)
    
    def solve_diffusion_spectral(self, field_spectral, coeff, dt, order=1):
        """:py:meth:`solve_diffusion` with spectral in- and output fields."""
        eigenvalues_op = self.sphere._laplacian_eigenvalues ** order
        return field_spectral / (1. + dt * coeff * eigenvalues_op)
    
    
    
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
    
        
        

 