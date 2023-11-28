import numpy as np
import random


###################################################
# Define physical constants

a = 6371e3         # Radius of the earth in m
Omega = 7.29e-5    # Angular velocity of the earth in rad/s
g00 = 9.81         # Acceleration due to gravity near the surface of the earth in m/s^2
d2r = np.pi / 180. # Factor to convert degrees to radians
r2d = 180. / np.pi # Factor to convert radians to degrees
d2s = 86400      # Factor to convert from days to seconds

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
        
        self.Tau_relax = 8*d2s #8 days thermal relaxation timescale
        self.Kappa = 0.1*d2s # 0.1 day thermal damping
        self.rs = 1/7. * 1/d2s #frictional dissipation 7 days,
        
        self.tstep = 0 #keeps track of which index from forcing timseries to be retrieved
        self.forcing_tseries=forcing_tseries
        
        #whether to use linear or nonlinear advection
        self.temp_linear = True #linear by default
        self.vort_linear = True
    
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
        Adv = -(self.sphere.U*dxvortp)
        
        #nonlinear advection term
        # -(u*dzetap/dx + v*dzetap/dy)
        J=0.
        if not self.vort_linear:
            psip,_ = self.sphere.uv2sfvp(u,v)
            #use arakawa jacobian to deal to numerical instabilities from perturbation
            J = self.sphere.Jacobian(self.sphere.vortp,psip)
        
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
        Adv =  -(self.sphere.U * dxthetap + v*self.sphere.dythetam)
        
        #nonlinear advection term
        # - (u*dT'/dx + v'*dT'/dy)
        J=0.
        if not self.temp_linear:
            psip,_ = self.sphere.uv2sfvp(u,v)
            #use arakawa jacobian to deal to numerical instabilities from perturbation
            J = self.sphere.Jacobian(self.sphere.thetap,psip)
            
        #relaxation term
        # - (T - Teq)/ Tau
        Rel = - ((self.sphere.thetap + self.sphere.thetaeq) - self.sphere.thetaeq)/ self.Tau_relax
        
        #hyper diffusion term, only to smallest resolved spherical harm.
        # -K nabla**8 (T)
        hypdif = np.zeros((self.sphere.nspecindx,2)).view(np.complex128).ravel()
        hypdif[-1] = (self.sphere.laplace_spectral(self.sphere.to_spectral(self.sphere.thetap))*4)[-1]
        Diff = - self.Kappa * self.sphere.to_grid(hypdif)
        
        return Adv+Rel+Diff+J
    
  

## TO DO: SCM
# ## Physical constants for SCM
# a_bar = 0.56 # coalbedo averaged between ice and ocean
# delta_a = 0.48 # difference in coalbedo between ice and ocean
# h_alpha = 0.5 # smoothness of albedo transition (m)
# yr2sec = 3.154e7
# B = 2.83 # dependence of net surface flux on surface temperature (W/m^2/K)
# Fb = 0 # upward heat flux into bottom
# Sm = 100. # downward shortwave annual-mean (W/m^2)
# Sa = 150. # downward shortwave seasonal amplitude (W/m^2)
# Lm = 70. # reference longwave annual-mean (W/m^2)
# La = 41. # reference longwave seasonal amplitude (W/m^2)
# P = 1.*yr2sec # forcing period (yrs)
# phi = 0.15*yr2sec # time between summer solstice and peak of longwave forcing (yrs)
# coHo = 2e8 # heat capacity of ocean mixed layer (J/m^2/K)
# Li = 3e8 # sea ice latent heat of fusion( J/m^3)
# zeta = 0.7 # sea ice thermodynamic scale thickness zeta=ki/B (m)


        
# def entropy_tend( T, t = 0, X=0., h_alpha=0.5):
#     """
#     Run single column model based on entropy (or T) as in Eisenman and Wagner, which contains sea ice
#     Reference: "How Climate Model Complexity Influences Sea Ice Stability"
#     T.J.W. Wagner & I. Eisenman, J Clim 28,10 (2015)
#     """
#     E = (T*coHo >= 0)*(T*coHo) + (T=0)*() 
#     E= T*coHo
#     A_star = ( a_bar+delta_a/2.*np.tanh(E/ (Li*h_alpha) ) ) * ( Sm-Sa*np.cos(2.*np.pi*t/P) ) - ( Lm+La*np.cos(2*np.pi*(t/P-phi/P)) )
#     if E>=0:                    #open ocean condition
#         h_i = 0.
#         T=E/coHo
#     elif (E<0) & (A_star > 0): #"melting" condition
#         h_i = E/-Li
#         T=0.
#     elif (E<0) & (A_star < 0): #"freezing" condition
#         h_i = E/-Li
#         T= (A_star/B) * (1./(1.+zeta/h_i))
        
#     dEdt = A_star-(B*T)+Fb+X 
    
#     dTdt = dEdt/coHo
#     return dEdt


    
        



        

 