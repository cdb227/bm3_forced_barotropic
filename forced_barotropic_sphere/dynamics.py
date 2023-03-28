import numpy as np
import random


###################################################
# Define physical constants

a = 6371e3         # Radius of the earth in m
Omega = 7.29e-5    # Angular velocity of the earth in rad/s
g00 = 9.81         # Acceleration due to gravity near the surface of the earth in m/s^2
d2r = np.pi / 180. # Factor to convert degrees to radians
r2d = 180. / np.pi # Factor to convert radians to degrees
d2s = 1/86400      # Factor to convert from days to seconds

class ForcedVorticity:
    """
    Class which contains the right hand side from a forced barotropic sphere
    """
    def __init__(self, sphere, vortp, thetap, forcing_tseries):
        """
        terms used in forced baroptropic vorticity
        
        Arguments:
        * sphere (Sphere object) : contains spectral methods and grid for integration
        * vortp (array; nlat,nlon) : perturbation vorticity field on sphere
        * thetap (array; nlat,nlon) : perturbation temperature field on sphere
        * forcing_tseries (Forcing object) : contains timestep information and forcing to be applied to sphere
        """

        self.sphere = sphere
        
        #nondivergent flow
        self.vortp_div = self.sphere.vortp_div
        
        
        #perturbation vorticity field
        self.vortp = vortp
        
        #mean vorticity field
        self.vortm = self.sphere.vortm
        self.dxvortm,self.dyvortm = self.sphere.gradient(self.vortm)
        
        self.thetap = thetap
        self.Tau_relax = 8*1/d2s #8 days thermal relaxation
        self.Kappa = 0.1*1/d2s # 0.1 day thermal damping
        self.dxthetam, self.dythetam = self.sphere.gradient(self.sphere.thetaeq)
        self.rs = 0.5 * d2s #frictional dissipation 0.5 days,

        
        self.tstep = 0 #keeps track of which index from forcing timseries to be retrieved
        self.forcing_tseries=forcing_tseries
        
        
        #whether to use linear or nonlinear advection
        self.linear = True #linear by default
 
    
    def vort_tendency(self):
        """
        Calculate dvortp/dt of forced barotropic vorticity equation
        Includes an advective term, epsilon term, frictional dissipation and a forcing
        """

        u,v = self.sphere.vrtdiv2uv(self.vortp, self.vortp_div) #pull out u and v from vorticity field at current timestep

        # +++ Dynamics +++ #
        dxvortp,dyvortp= self.sphere.gradient(self.vortp) #find dxzetap, dyzetap
        
        #linear advection term
        # -(U*dzetap/dx)
        Adv = -(self.sphere.U*dxvortp) 
        
        #nonlinear advection term
        # -(u*dzetap/dx + v*dzetap/dy)
        J=0.
        if not self.linear:
            psip = self.sphere.laplace(self.vortp)
            #use arakawa jacobian to deal to numerical instabilities from perturbations
            ##J = - (u*dxvortp + v*dyvortp)
            def KK(a,dadx,b,dbdy): return dadx*dbdy + self.sphere.gradient(a*dbdy)[0] + self.sphere.gradient(b*dadx)[1]
            def K(a,b): return KK(a,self.sphere.gradient(a)[0],b,self.sphere.gradient(b)[1]) # avoids computing da/dx, db/dy twice
            J = (K(self.vortp,psip)-K(psip,self.vortp)) / 3.
        
        #epsilon term
        #-v(beta-U_yy)
        B = -v*(self.sphere.beta-self.dyvortm)
        
        #frictional dissipation term
        #-r_s*zetap
        Diss = -self.rs*self.vortp
        
        #forcing term
        F = self.forcing_tseries[self.tstep,:]
        self.tstep+=1
        
        return Adv+B+Diss+F+J
    
    
    def theta_tendency(self):
        """
        Calculate dthetap/dt from winds forced by barotropic system
        Includes an advective term, thermal relaxation to background state, and hyperdiffusion
        """
        
        u,v = self.sphere.vrtdiv2uv(self.vortp, self.vortp_div)
        
        dxthetap, dythetap = self.sphere.gradient(self.thetap)
        
        
        #linear advection term
        # -(U*dT'/dx + v'*dT/dy)
        Adv =  -(self.sphere.U * dxthetap + v*self.dythetam)
        
        #nonlinear advection term
        # - (u*dT'/dx + v'*dT'/dy)
        J=0.
        if not self.linear:
            psip = self.sphere.laplace(self.vortp)
            #use arakawa jacobian to deal to numerical instabilities from perturbation
            def KK(a,dadx,b,dbdy): return dadx*dbdy + self.sphere.gradient(a*dbdy)[0] + self.sphere.gradient(b*dadx)[1]
            def K(a,b): return KK(a,self.sphere.gradient(a)[0],b,self.sphere.gradient(b)[1]) # avoids computing da/dx, db/dy twice
            J = (K(self.thetap,psip)-K(psip,self.thetap)) / 3.
        
        #relaxation term
        # - (T - Teq)/ Tau
        Rel = - ((self.thetap + self.sphere.thetaeq) - self.sphere.thetaeq)/ self.Tau_relax
        
        #hyper diffusion term, only to smallest resolved spherical harm.
        # -K nabla**8 (T)
        hypdif = np.zeros((self.sphere.nspecindx,2)).view(np.complex128).ravel()
        hypdif[-1] = (self.sphere.laplace_spectral(self.sphere.to_spectral(self.thetap))*4)[-1]
        #print((self.sphere.laplace_spectral(self.sphere.to_spectral(self.thetap))*4))
        #raise ValueError()
        Diff = - self.Kappa * self.sphere.to_grid(hypdif)
        
        return Adv+Rel+Diff+J
    
        



        

 