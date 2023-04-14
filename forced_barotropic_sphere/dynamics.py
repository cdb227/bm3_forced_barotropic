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
    Class which contains the right hand side of vorticity equation from FBS
    """
    def __init__(self, sphere, forcing_tseries):
        """
        terms used in forced baroptropic vorticity
        
        Arguments:
        * sphere (Sphere object) : contains spectral methods and grid for integration
        * forcing_tseries (Forcing object) : contains timestep information and forcing to be applied to sphere
        """

        self.sphere = sphere
        
        self.Tau_relax = 8*1/d2s #8 days thermal relaxation
        self.Kappa = 0.1*1/d2s # 0.1 day thermal damping
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

        u,v = self.sphere.vrtdiv2uv(self.sphere.vortp, self.sphere.vortp_div) #pull out u and v from vorticity field at current timestep

        # +++ Dynamics +++ #
        dxvortp,dyvortp= self.sphere.gradient(self.sphere.vortp) #find dxzetap, dyzetap
        
        #linear advection term
        # -(U*dzetap/dx)
        #Adv = -(self.sphere.U*dxvortp)
        #in Linz et al. (2018) Adv term is prescribed as
        Adv = -12./a * np.gradient(self.sphere.vortp, self.sphere.rlon[1]-self.sphere.rlon[0])[0]
        
        #nonlinear advection term
        # -(u*dzetap/dx + v*dzetap/dy)
        J=0.
        if not self.linear:
            psip = self.sphere.laplace(self.sphere.vortp)
            #use arakawa jacobian to deal to numerical instabilities from perturbations
            ##J = - (u*dxvortp + v*dyvortp)
            def KK(a,dadx,b,dbdy): return dadx*dbdy + self.sphere.gradient(a*dbdy)[0] + self.sphere.gradient(b*dadx)[1]
            def K(a,b): return KK(a,self.sphere.gradient(a)[0],b,self.sphere.gradient(b)[1]) # avoids computing da/dx, db/dy twice
            J = (K(self.sphere.vortp,psip)-K(psip,self.sphere.vortp)) / 3.
        
        #epsilon term
        #-v(beta-U_yy)
        B = -v*(self.sphere.beta-self.sphere.dyvortm)
        
        #frictional dissipation term
        #-r_s*zetap
        Diss = -self.rs*self.sphere.vortp
        
        #forcing term
        F = self.forcing_tseries[self.tstep,:]
        self.tstep+=1
        
        return Adv+B+Diss+F+J
    
    
    def theta_tendency(self):
        """
        Calculate dthetap/dt from winds forced by barotropic system
        Includes an advective term, thermal relaxation to background state, and hyperdiffusion
        """
        
        u,v = self.sphere.vrtdiv2uv(self.sphere.vortp, self.sphere.vortp_div)
        
        dxthetap, dythetap = self.sphere.gradient(self.sphere.thetap)
        
        
        #linear advection term
        # -(U*dT'/dx + v'*dT/dy)
        #Adv =  -(self.sphere.U * dxthetap + v*self.dythetam)
        Adv = v*self.sphere.dythetam # as in Linz et al. (2018)?
        
        #nonlinear advection term
        # - (u*dT'/dx + v'*dT'/dy)
        J=0.
        if not self.linear:
            psip = self.sphere.laplace(self.sphere.vortp)
            #use arakawa jacobian to deal to numerical instabilities from perturbation
            def KK(a,dadx,b,dbdy): return dadx*dbdy + self.sphere.gradient(a*dbdy)[0] + self.sphere.gradient(b*dadx)[1]
            def K(a,b): return KK(a,self.sphere.gradient(a)[0],b,self.sphere.gradient(b)[1]) # avoids computing da/dx, db/dy twice
            J = (K(self.sphere.thetap,psip)-K(psip,self.sphere.thetap)) / 3.
        
        #relaxation term
        # - (T - Teq)/ Tau
        Rel = - ((self.sphere.thetap + self.sphere.thetaeq) - self.sphere.thetaeq)/ self.Tau_relax
        
        #hyper diffusion term, only to smallest resolved spherical harm.
        # -K nabla**8 (T)
        hypdif = np.zeros((self.sphere.nspecindx,2)).view(np.complex128).ravel()
        hypdif[-1] = (self.sphere.laplace_spectral(self.sphere.to_spectral(self.sphere.thetap))*4)[-1]
        Diff = - self.Kappa * self.sphere.to_grid(hypdif)
        
        return Adv+Rel+Diff+J
    
        



        

 