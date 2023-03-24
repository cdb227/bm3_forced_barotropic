import numpy as np
import spharm
import random
import xarray as xr


###################################################
# Define physical constants

a = 6371e3         # Radius of the earth in m
Omega = 7.29e-5    # Angular velocity of the earth in rad/s
g00 = 9.81         # Acceleration due to gravity near the surface of the earth in m/s^2
d2r = np.pi / 180. # Factor to convert degrees to radians
r2d = 180. / np.pi # Factor to convert radians to degrees
Tau = 6*86400
#rs = 1/Tau


class Sphere:
    """
    Spectral class for setting up environment for forced batroptropic sphere
    contains routines to convert from lat/lon to sperical harmonics
    """
    def __init__(self, nlat, nlon, U = 0., theta0=300., deltheta= 45.,
                 rsphere=a, legfunc='stored', trunc=None):
        """
        initializes sphere for barotropic model.
        
        Arguments:
        * nlat (int) : number of latitude points
        * nlon (int) : number of longitude points
        * U (int/float/array) : background zonal mean wind
        * theta0 (float) : equator temperature
        * deltheta (float): amplitude of equator-to-pole gradient
        * rsphere (float) : radius of sphere
        """

        self.nlat = nlat
        self.nlon = nlon
        
        #equator is included as a discrete point if nlat is odd
        if self.nlat % 2:
            gridtype = 'gaussian'
        else:
            gridtype = 'regular'

        self.s = spharm.Spharmt(self.nlon, self.nlat, gridtype=gridtype,
                         rsphere=rsphere, legfunc=legfunc)
            
        
        # lat/lon grid in degrees
        self.glon = 360./nlon*np.arange(nlon)
        self.glat = np.linspace(90,-90,self.nlat)
        self.glons,self.glats = np.meshgrid(self.glon,self.glat)

        # lat/lon grid in radians
        self.rlat = np.deg2rad(self.glat)
        self.rlon = np.deg2rad(self.glon)
        self.rlons, self.rlats = np.meshgrid(self.rlon, self.rlat)
        
        #define zonal mean background wind profiles
        if (type(U) == float) | (type(U) == int):
            U=np.ones(self.glats.shape)*U
        self.U = U
        self.V= np.zeros(self.U.shape)


        # Constants
        # Earth's angular velocity
        self.omega = Omega  # unit: s-1
        # Gravitational acceleration
        self.g = g00  # unit: m2/s
        #beta
        self.f =  2.*self.omega*np.sin(self.rlats)
        self.beta = 2.*self.omega*np.cos(self.rlats)/rsphere
        
        #nondivergent flow
        self.vortp_div = np.zeros(self.rlats.shape, dtype = 'd')
        
        #mean vorticity field based on background winds
        self.vortm,_ = self.uv2vrtdiv(self.U,self.V)
        
        #temperature fields
        self.theta0 = theta0
        self.deltheta = deltheta
        #temp. distribution
        self.thetaeq = self.theta0 - deltheta*np.sin(self.rlats)**2
        
        #initial temp of sphere is the equil. temp.
        self.theta = self.thetaeq
        self.dxthetam,self.dythetam  = self.gradient(self.thetaeq)
        
        #nspectral
        #truncation (based on grid)
        self._ntrunc = (self.nlat - 1) if trunc is None else trunc
        self.nspecindx = self.to_spectral(self.rlats).shape[0]
        #index of m,n components for spherical harmonics
        self.specindxm, self.specindxn = spharm.getspecindx(self._ntrunc)
        self._laplacian_eigenvalues = (
                self.specindxn * (1. + self.specindxn) / rsphere / rsphere
                ).astype(np.complex64, casting="same_kind")
        
    
    ##+++spectral transforms+++    
    def to_spectral(self, field_grid):
        """Transform a gridded field into spectral space.
        Parameters:
            field_grid (array): Gridded representation of input field.
        Returns:
            Spectral representation of input field.
        """
        return self.s.grdtospec(field_grid, self._ntrunc)

    def to_grid(self, field_spectral):
        """Transform a spectral field into grid space.
        Parameters:
            field_spectral (array): Spectral representation of input field.
        Returns:
            Gridded representation of input field.
        """
        return self.s.spectogrd(field_spectral)

    def uv2vrtdiv(self, u, v, trunc=None):
        """
        Vortivity and divergence from u and v wind
        Input: u and v (grid)
        Output: vorticity and divergence (grid)
        """

        vrts, divs = self.s.getvrtdivspec(u, v, ntrunc=trunc)
        vrtg = self.s.spectogrd(vrts)
        divg = self.s.spectogrd(divs)
        return(vrtg, divg)

    def uv2sfvp(self, u, v, trunc=None):
        """
        Geostrophic streamfuncion and
        velocity potential from u and v winds
        Input: u and v (grid)
        Output: strf and vel potential (grid)
        """

        psig, chig = self.s.getpsichi(u, v, ntrunc=trunc)
        return(psig, chig)

    def vrtdiv2uv(self, vrt, div, realm='grid', trunc=None):
        """
        # u and v wind from vorticity and divergence
        # Input: vrt, div (either grid or spec)
        # Output: u and v (grid)
        """
        if realm in ['g', 'grid']:
            vrts = self.s.grdtospec(vrt, trunc)
            divs = self.s.grdtospec(div, trunc)
        elif realm in ['s', 'spec', 'spectral']:
            vrts = vrt
            divs = div
        ug, vg = self.s.getuv(vrts, divs)
        return(ug, vg)

    def gradient(self, var, trunc=None):
        """
        Calculate horizontal gradients
        Input: var
        Output: dvar/dx, dvar/dy
        """

        try:
            var = var.filled(fill_value=np.nan)
        except AttributeError:
            pass
        if np.isnan(var).any():
            raise ValueError('var cannot contain missing values')
        try:
            varspec = self.s.grdtospec(var, ntrunc=trunc)
        except ValueError:
            raise ValueError('input field is not compatitble')
        dxvarg, dyvarg = self.s.getgrad(varspec)
        return(dxvarg, dyvarg)
    
    def laplace(self, f):
        """Laplacian of an input field.
        Parameters:
            f (array): 2D input field.
        Returns:
            Gridded Laplacian of **f**.
        """
        return self.to_grid(self.laplace_spectral(self.to_spectral(f)))
    
    def laplace_spectral(self, f):
        """`laplace` with spectral in- and output fields."""
        return -f * self._laplacian_eigenvalues
    
    def to_flow(self, vort, theta):
        """
        Compute u, v, vort, theta from vortp, thetap solution
        """
        N, Ny, Nx = vort.shape
        uo   = np.zeros((N, Ny, Nx), 'd')
        vo   = np.zeros((N, Ny, Nx), 'd')
        #vortm,_= self.uv2vrtdiv(self.U,self.V)

        for i in range(N):
            uo[i],vo[i] = self.vrtdiv2uv(vort[i].values, self.vortp_div)
            uo[i] = self.U + uo[i]
            #vo[i] = vo


        crds = [vort.time[:], vort.y[:], vort.x[:]]
        vortp = vort.rename('vortp')
        vort = (vortp + self.vortm).rename('vort')
        
        thetap = theta.rename('thetap')
        theta = (thetap + self.thetaeq).rename('theta')
        
        #psi = xr.DataArray(psio, name = 'psi', coords = crds, dims = ['time', 'y', 'x'])
        u   = xr.DataArray(uo, name = 'u',   coords = crds, dims = ['time', 'y', 'x'])
        v   = xr.DataArray(vo, name = 'v',   coords = crds, dims = ['time', 'y', 'x'])
        return xr.Dataset(data_vars = dict(vort=vort, vortp=vortp, u=u, v=v, thetap=thetap, theta=theta))
    
    
    ####+++Several possibly useful background flow configurations+++####
    def held_1985(self, A=25., B=30., C=300.):
        """Zonal wind profile similar to that of the upper troposphere.
        Parameters:
            A (number): Coefficient for cos-term.
            B (number): Coefficient for cos³-term.
            C (number): Coefficient for cos⁶sin²-term.

        Introduced by Held (1985), also used by Held and Phillips (1987) and
        Ghinassi et al. (2018).
        """
        cosphi = np.cos(self.rlats)
        sinphi = np.sin(self.rlats)
        u = (A * cosphi - B * cosphi**3 + C * cosphi**6 * sinphi**2)/2.0
        #no meridional wind
        v = np.zeros_like(u)
        vort,_ = self.uv2vrtdiv(u,v)

        self.U = u
        self.V= v
        self.vortm = vort

    def gaussian_jet(self, amplitude=20., center_lat=45., stdev_lat=5.):
        """A bell-shaped, zonally-symmetric zonal jet.
        Parameters:
            amplitude (number): Peak zonal wind at the center of the jet in m/s.
            center_lat (number): Center of the jet in degrees.
            stdev_lat (number): Standard deviation of the jet in degrees.

        A linear wind profile in latitude is added to zero wind speeds at both
        poles.
        """
        u = amplitude * np.exp( -0.5 * (self.glats - center_lat)**2 / stdev_lat**2 )
        # Subtract a linear function to set u=0 at the poles
        u_south = u[-1,0]
        u_north = u[ 0,0]
        u = u - 0.5 * (u_south + u_north) + (u_south - u_north) * self.glats / 180.
        # No meridional wind
        v = np.zeros_like(u)
        vort,_ = self.uv2vrtdiv(u,v)

        self.U = u
        self.V= v
        self.vortm = vort
        

 