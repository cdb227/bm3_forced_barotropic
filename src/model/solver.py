import numpy as np
from tqdm import tqdm
import xarray as xr

import sys
sys.path.append('../src')  # Add the 'src' directory to the Python path
from model import forcing
from utils import config, constants


class Solver:
    """
    Solver class which will integrate the vorticity equation
    """
    def __init__(self,  sphere, T, **kwargs):
        """
        initializes a solver for barotropic model.
        
        Arguments:
        * sphere (Sphere object) : contains spectral methods and grid for integration
        * T (int)                : total integration time (seconds) 
        """
        
        self.sphere = sphere
        self.T      = T
        
        #get optional solver arguments
        self.dt     = kwargs.get('dt', config.DEFAULT_dt) #integration timestep
        self.ofreq  = kwargs.get('ofreq', config.DEFAULT_ofreq) #output freq
        
        #get optional dynamics arguments -- check utils/config for defaults.
        self.temp_linear = kwargs.get('temp_linear', config.DEFAULT_temp_linear) #linear dynamics?
        self.vort_linear = kwargs.get('vort_linear', config.DEFAULT_vort_linear)
        self.tau         = kwargs.get('tau', config.DEFAULT_tau) * 1/constants.day2sec #thermal relaxation rate (per day)
        self.rs          = kwargs.get('rs' , config.DEFAULT_rs)  * 1/constants.day2sec #frictional dissipation rate (per day)
        self.r           = kwargs.get('robert_filter', config.DEFAULT_robert_filter)# Robert-Asselin filter strength
        self.nu          = kwargs.get('nu', config.DEFAULT_nu )         # Hyperdiffusion coefficient
        self.diffusion_order = kwargs.get('diffusion_order', config.DEFAULT_diffusion_order) # Hyperdiffusion order
                       
        self.Nt = int(self.T / self.dt)        #number of steps   
        self.ts = np.arange(self.Nt) * self.dt #integration times
        
        self.No = int(self.Nt/self.ofreq)+1         #number of outputs
               
        #use laplacian eigenvalues for damping/diffusion term
        self.damping = self.nu * np.abs(self.sphere._laplacian_eigenvalues) ** self.diffusion_order
                
        self.forcing = forcing.Forcing(sphere, **kwargs)
            
    def integrate_dynamics(self, verbose=False):
        """
        Integrating function using leapfrog. By default, only the linear terms are considered for integration
        """
        #pointers for output      
        k0 = 0 
        k = 0
                
        #vorticity perturbation in spectral space
        #old, now, new
        zs = np.zeros((self.sphere.nspecindx, 3), dtype=np.complex128)

        # Dummy divergence perturbation
        ds = np.zeros((self.sphere.nspecindx, ) , dtype=np.complex128)

        #tendencies in spectral space
        dz = np.zeros((self.sphere.nspecindx, 1), dtype=np.complex128)
        
        #vorticity peturbation output to be saved    
        zo = np.zeros((self.No, self.sphere.nlat,self.sphere.nlon), 'd')

        # Tracer in spectral space
        trs = np.zeros((self.sphere.nspecindx, 3), dtype=np.complex128)

        dtrs = np.zeros((self.sphere.nspecindx, 1), dtype=np.complex128)

        #eventually this will be used for tracer variable, placeholder for now
        so = np.zeros((self.No, self.sphere.nlat,self.sphere.nlon), 'd')

        #set old, now vorticity to ics, used for diffusion
        zs[:, 0] = self.sphere.to_spectral(self.sphere.vortp)
        zs[:, 1] = zs[:, 0]

        # Initialize tracer in the same way
        trs[:, 0] = self.sphere.to_spectral(self.sphere.thetap)
        trs[:, 1] = trs[:, 0]
        
        #save to first step of output
        zo[k0, :, :] = self.sphere.to_linear_grid(zs[:, 0])
        so[k0, :, :] = self.sphere.to_linear_grid(trs[:, 0])
        
        # pointers: j -> state, i -> tendency
        jold,jnow,jnew = 0,1,2
        inow = 0
        
        #main integration loop
        for j, t in enumerate(tqdm(self.ts) if verbose else self.ts):
            # Step 1 & 2: Compute (f + ζ)u and (f + ζ)v on the grid at time t
            #since z = perturbation, this calculates u', v'  
            z = self.sphere.to_quad_grid(zs[:, jnow])
            u,v = self.sphere.vrtdiv2uv(zs[:, jnow], ds, realm='spec', grid='quad') #divergenceless flow

            # Compute advection tendencies
            if self.vort_linear: #Linear contributions fields:
                du =  (z * self.sphere.V + (self.sphere.f + self.sphere.Z) * (self.sphere.V + v))
                dv = -(z * self.sphere.U + (self.sphere.f + self.sphere.Z) * (self.sphere.U + u))

            else: #total fields:
                du =  (self.sphere.f + self.sphere.Z + z)*(v+self.sphere.V)
                dv = -(self.sphere.f + self.sphere.Z + z)*(u+self.sphere.U)

            # Add frictional damping
            du  += -u * self.rs
            dv  += -v * self.rs

            # Step 3: Compute the curl of du, dv to find dzdt in spectral
            dz[:, inow], _ = self.sphere.sq.getvrtdivspec(du, dv, self.sphere._ntrunc)
            dz[:, inow] += self.forcing.evolve_forcing()
            
            #extra-- add coupling from surface?
            #print(self.sphere.sq.getvrtdivspec(du, dv, self.sphere._ntrunc)[0].max(),self.sphere.laplace_spectral(trs[:,jnow]).max() )
            #dz[:, inow] += 1e-1*self.sphere.laplace_spectral(trs[:,jnow])

            # Step 1a: Compute tracer & gradients in grid space
            tr = self.sphere.to_quad_grid(trs[:, jnow])
            dx_tr, dy_tr = self.sphere.gradient(trs[:, jnow], realm = 'spec', grid = 'quad')
                        
            # Compute advection tendencies
            dtr = -u * self.sphere.dxthetam - self.sphere.U * dx_tr \
                  -v * self.sphere.dythetam - self.sphere.V * dy_tr
            if not self.temp_linear: #include if non-linear
                dtr -= u * dx_tr + v * dy_tr

            # Add thermal relaxation to underlying temp field
            dtr += -tr * self.tau

            dtrs[:, inow] = self.sphere.to_spectral(dtr)
                
            if j==0: #for dt difference in first step
                #Step 4: Compute & apply damping 
                #compute:  Z -> (Z-nu*L**n*zold)/(1+nu*2dt*L**n)
                c = 1. / (1. + self.damping * self.dt)

                # Apply damping to tendency
                dz  [:, inow] = c * (dz  [:,inow] - self.damping * zs [:,jold])
                dtrs[:, inow] = c * (dtrs[:,inow] - self.damping * trs[:,jold])

                #Step 5: step forward in time; use eularian forward
                zs [:, jnew] = zs [:, jnow] + self.dt * dz  [:,inow]
                trs[:, jnew] = trs[:, jnow] + self.dt * dtrs[:,inow]
            else:
                #Step 4: Compute & apply damping 
                c = 1. / (1. + self.damping * 2 * self.dt)

                # Apply damping to z tendency
                dz  [:, inow] = c * (dz  [:,inow] - self.damping * zs [:,jold])
                dtrs[:, inow] = c * (dtrs[:,inow] - self.damping * trs[:,jold])
                
                #Step 5: step forward in time; use leapfrog
                zs [:, jnew] = zs [:, jold] + 2 * self.dt * dz  [:,inow]
                trs[:, jnew] = trs[:, jold] + 2 * self.dt * dtrs[:,inow]
                
            #5.1 apply robert filter
            zs [:, jnow] = (1-2*self.r)*zs [:,jnow] + self.r*(zs [:,jnew] + zs [:,jold])
            trs[:, jnow] = (1-2*self.r)*trs[:,jnow] + self.r*(trs[:,jnew] + trs[:,jold])

            k += 1 #add to output
            if k >= self.ofreq:
                k0 += 1
                zo[k0, :, :] = self.sphere.to_linear_grid(zs [:, jnow])
                so[k0, :, :] = self.sphere.to_linear_grid(trs[:, jnow])
                k = 0

            #Step 6: cycle array, now->old, new->now
            jold, jnow, jnew = jnow, jnew, jold
           
                
        #convert to flow and save output        
        crds = [np.linspace(0, self.T, self.No), self.sphere.glat.data[:], self.sphere.glon.data[:]]
        vort = xr.DataArray(zo, name = 'vort', coords = crds, dims = ['time', 'y', 'x'])
        theta   = xr.DataArray(so, name = 'theta',   coords = crds, dims = ['time', 'y', 'x']) #placeholder for now

        return self.to_flow(vort,theta)
   
    def to_flow(self, vort, theta):
        """
        Compute u, v, vort, theta from vortp, thetap solution
        """
        N, Ny, Nx = vort.shape
        uo   = np.zeros((N, Ny, Nx), 'd')
        vo   = np.zeros((N, Ny, Nx), 'd')

        for i in range(N):
            uo[i],vo[i] = self.sphere.vrtdiv2uv(vort[i].values + self.sphere.Z_lin, self.sphere.vortp_div_lin, realm = 'grid', grid='linear')


        crds = [vort.time[:], vort.y[:], vort.x[:]]
        vort = vort.rename('vort')
        vortp = (vort - self.sphere.Z_lin).rename('vortp')
        
        thetap = theta.rename('thetap')
        theta = (thetap + self.sphere.thetaeq_lin).rename('theta')
        
        #psi = xr.DataArray(psio, name = 'psi', coords = crds, dims = ['time', 'y', 'x'])
        u   = xr.DataArray(uo, name = 'u',   coords = crds, dims = ['time', 'y', 'x'])
        v   = xr.DataArray(vo, name = 'v',   coords = crds, dims = ['time', 'y', 'x'])
        
        
        sln = xr.Dataset(data_vars = dict(vort=vort, vortp=vortp, u=u, v=v, thetap=thetap, theta=theta))
        
        sln = sln.assign_coords(x=(((sln.x + 180) % 360) - 180)) # Reassign x coordinates between -180 and 180E
        sln = sln.sortby('x')
        
        #have y go from -90 -> 90
        sln=sln.reindex(y=list(reversed(sln.y))) #have y go from -90 -> 90
        #sln=sln.rename(dict(x='lon',y='lat')) # rename x,y to lon,lat

        return sln
    