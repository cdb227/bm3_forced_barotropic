import numpy as np
from tqdm import tqdm
import xarray as xr

#from forced_barotropic_sphere.dynamics import ForcedVorticity

a = 6371e3         # Radius of the earth in m
Omega = 7.29e-5    # Angular velocity of the earth in rad/s
g00 = 9.81         # Acceleration due to gravity near the surface of the earth in m/s^2
d2r = np.pi / 180. # Factor to convert degrees to radians
r2d = 180. / np.pi # Factor to convert radians to degrees
d2s = 86400      # Factor to convert from days to seconds


class Solver:
    """
    Solver class which will integrate the vorticity equation
    """
    def __init__(self,  sphere, forcing, ofreq, **kwargs):
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
         
        tau = kwargs.get('tau', 0)
        Kappa = kwargs.get('Kappa', 0)
        rs = kwargs.get('rs', 1/7.)
        
        self.tau = tau * 1/d2s      # Thermal relaxation rate
        self.Kappa = Kappa*d2s    # 0.1 day thermal damping
        self.rs = rs * 1/d2s      # frictional dissipation 7 days,

        # Hyperdiffusion coefficient and order
        self.nu = kwargs.get('nu', 0.)
        self.diffusion_order = kwargs.get('diffusion_order', 2)
        
        #print('integrating with: ', 'nu=',self.nu,'diffusion_order=',self.diffusion_order)
        
        # Robert-Asselin filter strength
        self.r = kwargs.get('robert_filter', 0.02)
        
        #use laplacian eigenvalues for damping/diffusion term
        self.Vdamping = self.nu * np.abs(self.sphere._laplacian_eigenvalues) ** self.diffusion_order
        
        self.Tdamping = self.Kappa * np.abs(self.sphere._laplacian_eigenvalues) ** self.diffusion_order
        
        #TODO: integrate forcing
        self.forcing = forcing.forcing_tseries
            
    def integrate_dynamics(self, temp_linear=True, vort_linear=True):
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
        for j, t in enumerate(self.ts):
            # Step 1 & 2: Compute (f + ζ)u and (f + ζ)v on the grid at time t
            #since z = perturbation, this calculates u', v'  
            z = self.sphere.to_quad_grid(zs[:, jnow])
            u,v = self.sphere.vrtdiv2uv(zs[:, jnow], ds, realm='spec', grid='quad') #divergenceless flow

            # Compute advection tendencies
            if vort_linear: #remove nonlinear terms
                #Linear contributions fields:
                du =  (z * self.sphere.V + (self.sphere.f + self.sphere.Z) * (self.sphere.V + v))
                dv = -(z * self.sphere.U + (self.sphere.f + self.sphere.Z) * (self.sphere.U + u))
                #du = self.sphere.f * v
                #dv = -self.sphere.f * u
            else:
                #total fields:
                du =  (self.sphere.f + self.sphere.Z + z)*(v+self.sphere.V)
                dv = -(self.sphere.f + self.sphere.Z + z)*(u+self.sphere.U)

            # Add frictional damping
            du  += -u * self.rs
            dv  += -v * self.rs

            # Step 3: Compute the curl of du, dv to find dzdt in spectral
            dz[:, inow], _ = self.sphere.sq.getvrtdivspec(du, dv, self.sphere._ntrunc)
            dz[:, inow] += self.sphere.to_spectral(self.forcing[j]) #finally, add forcing term in spectral space

            # Step 1a: Compute tracer & gradients in grid space
            tr = self.sphere.to_quad_grid(trs[:, jnow])
            dx_tr, dy_tr = self.sphere.gradient(trs[:, jnow], realm = 'spec', grid = 'quad')

            # Compute advection tendencies
            dtr = -u * self.sphere.dxthetam - self.sphere.U * dx_tr \
                  -v * self.sphere.dythetam - self.sphere.V * dy_tr
            if not temp_linear:
                dtr -= u * dx_tr + v * dy_tr

            # Add thermal relaxation to underlying temp field
            dtr += -tr * self.tau

            dtrs[:, inow] = self.sphere.to_spectral(dtr)
                
            if j==0: #for dt difference in first step
                #Step 4: Compute & apply damping 
                #compute:  Z -> (Z-nu*L**n*zold)/(1+nu*2dt*L**n)
                cV = 1. / (1. + self.Vdamping * self.dt)
                cT = 1. / (1. + self.Tdamping * self.dt)


                # Apply damping to tendency
                dz  [:, inow] = cV * (dz  [:,inow] - self.Vdamping * zs [:,jold])
                dtrs[:, inow] = cT * (dtrs[:,inow] - self.Tdamping * trs[:,jold])

                #Step 5: step forward in time; use eularian forward
                zs [:, jnew] = zs [:, jnow] + self.dt * dz  [:,inow]
                trs[:, jnew] = trs[:, jnow] + self.dt * dtrs[:,inow]
            else:
                #Step 4: Compute & apply damping 
                cV = 1. / (1. + self.Vdamping * 2 * self.dt)
                cT = 1. / (1. + self.Tdamping * 2 * self.dt)

                # Apply damping to z tendency
                dz  [:, inow] = cV * (dz  [:,inow] - self.Vdamping * zs [:,jold])
                dtrs[:, inow] = cT * (dtrs[:,inow] - self.Tdamping * trs[:,jold])
                
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
            #uo[i] = self.sphere.U + uo[i]
            #vo[i] = vo


        crds = [vort.time[:], vort.y[:], vort.x[:]]
        vort = vort.rename('vort')
        vortp = (vort - self.sphere.Z_lin).rename('vortp')
        
        thetap = theta.rename('thetap')
        theta = (thetap + self.sphere.thetaeq_lin).rename('theta')
        
        #psi = xr.DataArray(psio, name = 'psi', coords = crds, dims = ['time', 'y', 'x'])
        u   = xr.DataArray(uo, name = 'u',   coords = crds, dims = ['time', 'y', 'x'])
        v   = xr.DataArray(vo, name = 'v',   coords = crds, dims = ['time', 'y', 'x'])
        return xr.Dataset(data_vars = dict(vort=vort, vortp=vortp, u=u, v=v, thetap=thetap, theta=theta))
    
    
    def solve_diffusion_spectral(self, field_spectral, coeff, dt, order=1):
        """:py:meth:`solve_diffusion` with spectral in- and output fields."""
        eigenvalues_op = self.sphere._laplacian_eigenvalues ** order
        return field_spectral / (1. + dt * coeff * eigenvalues_op)