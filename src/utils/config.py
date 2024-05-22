###sphere configs.
M                   = 63       #truncation
DEFAULT_BASE_STATE  = 'solid'  #sphere zonal mean background wind profiles
THETA0              = 300.     #sphere background temp 
DELTHETA            = 45       #sphere background gradient

#integration defaults
DEFAULT_temp_linear  = True    #by default, use model with linear temperature terms
DEFAULT_vort_linear  = True    #by default, use model with linear vorticity terms
DEFAULT_ofreq        = 2       #only save every ofreq-th integration step
DEFAULT_dt           = 4000    #default integration timestep (seconds)

#dynamics defaults
DEFAULT_rs                = 1/7. #frictional dissiptation rate (per day)
DEFAULT_tau               = 1/8. #thermal relaxation rate (per day)
DEFAULT_robert_filter     = 0.02 #Robert-Asselin filter strength
DEFAULT_nu                = 0.   #hyperdiffusion coefficient 
DEFAULT_diffusion_order   = 2    #hyperdiffusion order

#forcing defaults
DEFAULT_forcing           = 'rededdy'

#ensemble defaults
DEFAULT_ENS_SIZE          = 10

###seaice configs.
INCLUDE_ICE         = False    #include sea ice in model or not
ICE_LAT             = 50       #ice edge location(degrees lat)
ICE_WIDTH           = 1        #transition width (degrees lat)
ICE_JUMP            = 10       #temperature jump for ice locations (K)
