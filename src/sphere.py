import numpy as np
import spharm
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
    def __init__(self, M, U = 0., theta0=300., deltheta= 45.,
                 rsphere=a, legfunc='stored', seaice=False, base_state='solid'):
        """
        initializes sphere for barotropic model.
        
        Arguments:
        * M : # of wave numbers to retain in triangular truncation
        * U (int/float/array) : background zonal mean wind
        * theta0 (float) : equator temperature
        * deltheta (float): amplitude of equator-to-pole gradient
        * rsphere (float) : radius of sphere
        """

        # Truncation, linear, and quadratic grids
        self._ntrunc = M
        self.nlon = 2*M + 1
        self.nlat = M + 1

        self.nqlon = 3*M + 1
        self.nqlat = int(np.ceil((3*M + 1)/2))

        # Linear transform grid (for output)
        self.s = spharm.Spharmt(self.nlon, self.nlat, gridtype='gaussian',
                         rsphere=rsphere, legfunc=legfunc)

        # Quadradic transform grid (for numerics to avoid aliasing)
        self.sq = spharm.Spharmt(self.nqlon, self.nqlat, gridtype='gaussian',
                         rsphere=rsphere, legfunc=legfunc)
        
        # lat/lon grid in degrees
        self.glon = 360./self.nlon*np.arange(self.nlon)
        self.glat = spharm.gaussian_lats_wts(self.nlat)[0]
        self.glons,self.glats = np.meshgrid(self.glon,self.glat)

        # lat/lon grid in radians
        self.rlat = np.deg2rad(self.glat)
        self.rlon = np.deg2rad(self.glon)
        self.rlons, self.rlats = np.meshgrid(self.rlon, self.rlat)

        # quad. lat/lon grid in degrees
        self.gqlon = 360./self.nqlon*np.arange(self.nqlon)
        self.gqlat = spharm.gaussian_lats_wts(self.nqlat)[0]
        self.gqlons,self.gqlats = np.meshgrid(self.gqlon,self.gqlat)

        # quad. lat/lon grid in radians
        self.rqlat = np.deg2rad(self.gqlat)
        self.rqlon = np.deg2rad(self.gqlon)
        self.rqlons, self.rqlats = np.meshgrid(self.rqlon, self.rqlat)

        # Constants
        # Earth's angular velocity
        self.omega = Omega  # unit: s-1
        # Gravitational acceleration
        self.g = g00  # unit: m2/s
        #beta
        self.f =  2.*self.omega*np.sin(self.rqlats)
        self.beta = 2.*self.omega*np.cos(self.rqlats)/rsphere
        
        #define zonal mean background wind profiles
        if base_state == 'solid':
            self.solid_body(U = 10)
        elif base_state == 'rest':
            self.solid_body(U = 0)
        elif base_state == 'held85':
            self.held_1985()
        elif base_state == 'gaussian':
            self.gaussian_jet()       
        else:
            raise ValueError('base_state not recognized.')
        
        #nondivergent flow
        self.vortp_div = np.zeros(self.rqlats.shape, dtype = 'd')
        self.vortp_div_lin = np.zeros(self.rlats.shape, dtype = 'd')
        
        #mean vorticity field based on background winds
        self.Z,_     = self.uv2vrtdiv(self.U, self.V, grid = 'quad')
        self.Z_lin,_ = self.uv2vrtdiv(self.U, self.V, grid = 'linear')

        #self.dxvortp,self.dyvortm = self.gradient(self.vortm)
        
        #temperature fields
        self.theta0 = theta0
        self.deltheta = deltheta
        #temp. distribution
        self.thetaeq     = self.theta0 - deltheta*np.sin(self.rqlats)**2
        self.thetaeq_lin = self.theta0 - deltheta*np.sin(self.rlats)**2
        
        #+++Addition of step function in temperature+++#
        sc = 50. #ice edge location
        sw = 1 #transition width (degrees lat)
        sh = 10# temperature difference (K)
        si = sh*np.tanh((self.rlats-np.radians(sc))*(1./np.radians(sw)))
        si[si<0]=0.
        if seaice:
            self.thetaeq = self.thetaeq-si
        
        #initial temp of sphere is the equil. temp.
        self.theta = self.thetaeq
        self.dxthetam, self.dythetam  = self.gradient(self.thetaeq, grid = 'quad')
        
        #perturbation fields (none by default)
        self.vortp = np.zeros(self.gqlats.shape)
        self.thetap= np.zeros(self.gqlats.shape)
        
        #nspectral
        #truncation (based on grid)
        #self._ntrunc = (self.nlat - 1) if trunc is None else trunc
        self.nspecindx = self.to_spectral(self.rlats).shape[0]
        #index of m,n components for spherical harmonics
        self.specindxm, self.specindxn = spharm.getspecindx(self._ntrunc)
        
        #eigenvalues of the laplacian matrix
        self._laplacian_eigenvalues = (
                self.specindxn * (1. + self.specindxn) / rsphere / rsphere
                ).astype(np.complex128, casting="same_kind")
        
        
    def set_ics(self, ics):
        """
        set the perturbation vorticity and temperature field
        Parameters:
            ics (array : (2,nlat,nlon) : initial conditions of vorticity perturbation and temp perturbation, respectively
        """
        self.vortp = ics[0]# + self.vortm
        self.thetap = ics[1]
        
    
    ##+++spectral transforms+++    
    def to_spectral(self, field_grid):
        """Transform a gridded field into spectral space.
        Parameters:
            field_grid (array): Gridded representation of input field.
        Returns:
            Spectral representation of input field.
        """
        if field_grid.shape[0] == self.nlat:
           return self.s.grdtospec(field_grid, self._ntrunc)
        elif field_grid.shape[0] == self.nqlat:
           return self.sq.grdtospec(field_grid, self._ntrunc)
        else:
           raise ValueError('Shape ' + field_grid.shape + ' of gridded field does not match linear or quadratic grid.')

    def to_linear_grid(self, field_spectral):
        """Transform a spectral field into grid space.
        Parameters:
            field_spectral (array): Spectral representation of input field.
        Returns:
            Gridded representation of input field.
        """
        return self.s.spectogrd(field_spectral)

    def to_quad_grid(self, field_spectral):
        """Transform a spectral field into grid space.
        Parameters:
            field_spectral (array): Spectral representation of input field.
        Returns:
            Gridded representation of input field.
        """
        return self.sq.spectogrd(field_spectral)

    def uv2vrtdiv(self, u, v, grid='quad'):
        """
        Vorticity and divergence from u and v wind
        Input: u and v (either linear or quadratic grid)
        Output: vorticity and divergence (grid specified by argument)
        """

        if u.shape[0] == self.nlat:
            vrts, divs = self.s.getvrtdivspec(u, v, ntrunc=self._ntrunc)
        elif u.shape[0] == self.nqlat:
            vrts, divs = self.sq.getvrtdivspec(u, v, ntrunc=self._ntrunc)
        else:
            raise ValueError('Shape ' + u.shape + ' of gridded winds does not match linear or quadratic grid.')

        if grid == 'linear':
            s = self.s
        elif grid == 'quad':
            s = self.sq
        else:
            raise ValueError('grid (%s) must be either "linear" or "quad".')

        vrtg = s.spectogrd(vrts)
        divg = s.spectogrd(divs)
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

    def vrtdiv2uv(self, vrt, div, realm = 'grid', grid = 'linear'):
        """
        # u and v wind from vorticity and divergence
        # Input: vrt, div (either grid or spec)
        # Output: u and v (grid)
        """
        if grid == 'linear':
           s = self.s
        elif grid == 'quad':
           s = self.sq
        else:
           raise ValueError('grid (%s) must be either "linear" or "quad".')

        if realm in ['g', 'grid']:
           vrts = self.to_spectral(vrt)
           divs = self.to_spectral(div)
        elif realm in ['s', 'spec', 'spectral']:
            vrts = vrt
            divs = div
        else:
            raise ValueError("realm (%s) must be either 'grid' or 'spectral'" % realm)

        ug, vg = s.getuv(vrts, divs)
        return(ug, vg)

    def gradient(self, var, realm = 'grid', grid = 'linear'):
        """
        Calculate horizontal gradients
        Input: var
        Output: dvar/dx, dvar/dy
        """
        if grid == 'linear':
           s = self.s
        elif grid == 'quad':
           s = self.sq
        else:
           raise ValueError('grid (%s) must be either "linear" or "quad".')

        if realm in ['g', 'grid']:
            try:
                var = var.filled(fill_value=np.nan)
            except AttributeError:
                pass
            if np.isnan(var).any():
                raise ValueError('var cannot contain missing values')
            try:
                varspec = self.to_spectral(var)
            except ValueError:
                raise ValueError('input field is not compatitble')
        elif realm in ['s', 'spec', 'spectral']:
            varspec = var
        else:
            raise ValueError("realm (%s) must be either 'grid' or 'spectral'" % realm)
        
        return s.getgrad(varspec)
    
    def laplace(self, f, n = 1):
        """Laplacian of an input field.
        Parameters:
            f (array): 2D input field.
            n (integer): power of laplacian to compute; n = 1 corresponds to \nabla^2, n=2 to \nabla^4, etc.
        Returns:
            Gridded Laplacian of **f**.
        """
        return self.to_grid(self.laplace_spectral(self.to_spectral(f), n))
    
    def laplace_spectral(self, f, n = 1):
        """`laplace` with spectral in- and output fields."""
        return -(f**n) * self._laplacian_eigenvalues
    

    def Jacobian(self,A,B):
        """ Returns the Jacobian of two fields (A,B)"""
        #dadx*dbdy + ddx(a*dbdy) + ddy(b*dadx)
        def KK(a,dadx,b,dbdy): return dadx*dbdy + self.gradient(a*dbdy)[0] + self.gradient(b*dadx)[1]
        def K(a,b): return KK(a,self.gradient(a)[0], b,self.gradient(b)[1]) # avoids computing da/dx, db/dy twice
        J = (K(A,B)-K(B,A)) / 3.
        return J
    
    
    ####+++Several possibly useful background flow configurations+++####
    def solid_body(self, U=10.):
        """Zonal wind profile in solid body rotation"""
        u = U*np.cos(self.rqlats)
        v = np.zeros_like(u)

        self.U = u
        self.V = v

    def held_1985(self, A=25., B=30., C=300.):
        """Zonal wind profile similar to that of the upper troposphere.
        Parameters:
            A (number): Coefficient for cos-term.
            B (number): Coefficient for cos³-term.
            C (number): Coefficient for cos⁶sin²-term.

        Introduced by Held (1985), also used by Held and Phillips (1987) and
        Ghinassi et al. (2018).
        """
        cosphi = np.cos(self.rqlats)
        sinphi = np.sin(self.rqlats)
        u = (A * cosphi - B * cosphi**3 + C * cosphi**6 * sinphi**2)/2.0
        #no meridional wind
        v = np.zeros_like(u)
        
        self.U = u
        self.V = v
        
    def gaussian_jet(self, amplitude=20., center_lat=45., stdev_lat=5.):
        """A bell-shaped, zonally-symmetric zonal jet.
        Parameters:
            amplitude (number): Peak zonal wind at the center of the jet in m/s.
            center_lat (number): Center of the jet in degrees.
            stdev_lat (number): Standard deviation of the jet in degrees.

        A linear wind profile in latitude is added to zero wind speeds at both
        poles.
        """
        u = amplitude * np.exp( -0.5 * (self.gqlats - center_lat)**2 / stdev_lat**2 )
        # Subtract a linear function to set u=0 at the poles
        u_south = u[-1,0]
        u_north = u[ 0,0]
        u = u - 0.5 * (u_south + u_north) + (u_south - u_north) * self.gqlats / 180.
        # No meridional wind
        v = np.zeros_like(u)
        
        self.U = u
        self.V = v

 
