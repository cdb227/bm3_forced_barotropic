from model.solver import Solver #bm3_barotropic_model packages
from model.sphere import Sphere
from model.forcing import Forcing

from utils import constants, plotting, config
import ensemble_methods as em

import numpy as np
import xarray as xr
import json

import concurrent.futures


#simpel script to generate some ensemble runs/ save to compile for later

dt= 1800
params=dict(nu=5e-18, tau = 1/8, diffusion_order=4, dt=dt, ofreq=6,
            vort_linear=False, theta_linear=False, forcing_type='rededdy')


#produce a climatology-- 10yr sim
#T = 10*365*constants.day2sec
#st = Sphere(base_state='solid')
#st.add_seaice()
#climatology = Solver(st, T= T, **params).integrate_dynamics(verbose=True)
params_str = json.dumps(params)
#climatology['params'] = params_str

# #save every 6 hrs.
#climatology = climatology.assign_coords(time=climatology.time*constants.sec2day).thin({'time':2})
#climatology.to_netcdf('/doppler/data8/bertossa/bm3/climatology_nonlinear_seaice.nc')
#climatology=None

#produce a large ensemble 30-day sim
params['ensemble_size']= 200
params['seaice']= True


def run_ens(sim_id):
    st = Sphere(base_state='solid')
    st.add_seaice()

    T=15*constants.day2sec
    ensemble = em.integrate_ensemble(st=st,T=T,**params)
    
    ensemble = ensemble['theta'] #to save some space, we're only gonna writeout theta for now
    
    params_str = json.dumps(params)

    ensemble['params'] = params_str

    ensemble = ensemble.assign_coords(time=ensemble.time*constants.sec2day)
    ensemble.to_netcdf(f'/doppler/data8/bertossa/bm3/LE_nonlinear_seaice_run{sim_id+30}.nc')
    
    
num_simulations = 50
simulation_ids = list(range(num_simulations))
with concurrent.futures.ProcessPoolExecutor(max_workers=5) as executor:

    # Map the run_simulation function to the list of simulation IDs
    executor.map(run_ens, simulation_ids)
