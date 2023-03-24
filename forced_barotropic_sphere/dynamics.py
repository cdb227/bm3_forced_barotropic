import numpy as np
import spharm
import random
from forced_barotropic_sphere.sphere import Sphere
from forced_barotropic_sphere.forcing import Forcing


###################################################
# Define physical constants

a = 6371e3         # Radius of the earth in m
Omega = 7.29e-5    # Angular velocity of the earth in rad/s
g00 = 9.81         # Acceleration due to gravity near the surface of the earth in m/s^2
d2r = np.pi / 180. # Factor to convert degrees to radians
r2d = 180. / np.pi # Factor to convert radians to degrees
rs = 0.5/86400 #frictional dissipation 0.5 days

### Class which solves a forced vorticity equation using spherical harmonics
class ForcedVorticity:
    def __init__(self, sphere, vortp, thetap, forcing_tseries):


        self.sphere = sphere
        
        #nondivergent flow
        self.vortp_div = self.sphere.vortp_div
        
        
        #perturbation vorticity field
        self.vortp = vortp
        
        #mean vorticity field
        self.vortm = self.sphere.vortm
        self.dxvortm,self.dyvortm = self.sphere.gradient(self.vortm)
        
        self.thetap = thetap
        self.Tau_relax = 8*24*60*60 #8 days thermal relaxation
        self.Kappa = 0.1*24*60*60 # 0.1 day thermal damping
        self.dxthetam, self.dythetam = self.sphere.gradient(self.sphere.thetaeq)
        
        #no forcing unless specified
        self.tstep = 0
        self.forcing_tseries=forcing_tseries
        
        
        #whether to use linear or nonlinear advection
        self.linear = True #linear by default
 
    
    def vort_tendency(self):
        """
        Calculate dvortp/dt as in Linz. et al. (2018)
        Includes an advective term, beta term, frictional dissipiation and a wave forcing
        Input: vortp
        Output: dvortp/dt
        """

        u,v = self.sphere.vrtdiv2uv(self.vortp, self.vortp_div) #pull out u and v from vorticity field at current timestep

        # +++ Dynamics +++ #

        dxvortp,dyvortp= self.sphere.gradient(self.vortp) #find dx zetap
        
        #dx
        
        #linear advection term
        # -(U*dzetap/dx + U_yy)
        #prescribed zonal advective speed
        Adv= -(12.*dxvortp) #+ v*self.dyvortm)
        
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
        
        #beta term
        B = -v*self.sphere.beta
        
        #frictional dissipation term
        Diss = -rs*self.vortp
        
        F = self.forcing_tseries[self.tstep,:]
        self.tstep+=1
        
        return Adv+B+Diss+F+J#+self.sphere.to_grid(hypdif)
    
    
    def theta_tendency(self):
        
        u,v = self.sphere.vrtdiv2uv(self.vortp, self.vortp_div)
        
        #we treat the equilibrium temp as the background component?
        #dxthetam, dythetam = self.sphere.gradient(self.sphere.thetaeq)
        
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
    
        



        

 