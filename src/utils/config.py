import numpy as np

###sphere configs.
DEFAULT_M           = 63       #truncation
DEFAULT_BASE_STATE  = 'solid'  #sphere zonal mean background wind profiles
THETA0              = 300.     #sphere background temp 
DELTHETA            = 45       #sphere background gradient

#integration defaults
DEFAULT_temp_linear  = True    #by default, use model with linear temperature terms
DEFAULT_vort_linear  = True    #by default, use model with linear vorticity terms
DEFAULT_ofreq        = 2       #only save every ofreq-th integration step
DEFAULT_dt           = 1800    #default integration timestep (seconds)

#dynamics defaults
DEFAULT_rs                = 1/7. #frictional dissiptation rate (per day)
DEFAULT_tau               = 1/8. #thermal relaxation rate (per day)
DEFAULT_robert_filter     = 0.02 #Robert-Asselin filter strength
DEFAULT_nu                = 0.   #hyperdiffusion coefficient 
DEFAULT_diffusion_order   = 2    #hyperdiffusion order

#forcing defaults
DEFAULT_forcing_type      = 'zero_forcing'
DEFAULT_forcing_A         = 1e-11

#ensemble defaults
DEFAULT_ENS_SIZE          = 10

###seaice defaults
ICE_LAT             = 60       #ice edge location(degrees lat)
ICE_WIDTH           = 1        #transition width (degrees lat)
ICE_JUMP            = 10       #temperature jump for ice locations (K)



#some other config sets to use for model setup/testing:

def held_1985(st):
    """recreate model from Held 1985: Pseudomomentum and the Orthogonality of Modes in Shear Flows"""
    #first set state into appropriate conditions
    st.set_base_state('held85')
    
    k0=3
    vortp = 1e-5 * np.exp( -0.5 * (st.glats - 45.)**2 / 10**2 ) * np.cos(k0 * st.rlons)
    thetap = np.zeros(vortp.shape)
    st.set_ics([vortp,thetap])
    
    # Turn off frictional dissipation, thermal relaxation, add viscosity values from Held 1985
    params = dict(rs = 0., tau = 0., nu = 1e4, diffusion_order=1, robert_filter=0.01,
                  dt=1800, forcing_type='zero_forcing',
                  temp_linear=True, vort_linear=False)
    
    return params


#TODO: need to check this setup
def held_1987(st):    
    vortp = 1e-5 * np.exp( -0.5 * (st.glats - 45.)**2 / 10**2 -0.5 * (st.glons - 60.)**2 / 10**2 )

    thetap = np.zeros(vortp.shape)
    st.set_ics([vortp,thetap])
    
    # Turn off frictional dissipation, add viscosity values from Held and Phillips 1987
    params = dict(rs = 0., nu = 1e15, diffusion_order=2, dt=1800,
                  forcing_type='zero_forcing',temp_linear=True, vort_linear=False)
    
    return params