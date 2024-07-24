import numpy as np
import xarray as xr
import random
from tqdm import tqdm # used for progress tracking
from functools import partial
import time
from cython_routines import cbm_methods as cbm

from model.sphere import Sphere
from model.solver import Solver
from model.forcing import Forcing
from utils import config, constants

import multiprocessing
import gc

def integrate_ensemble(st, T, **kwargs):
    """
    Function to generate an ensemble of forced barotropic model,
    these are initialized with the same forcing term, which decorrelates after about a week.
    """
    M = st._ntrunc
    nlon = 2*M + 1
    nlat = M + 1
    
    n_ens         = kwargs.get('ensemble_size', config.DEFAULT_ENS_SIZE)
    ens_vpert     = kwargs.get('ensemble_vortp',  0.)
    ens_tpert     = kwargs.get('ensemble_thetap', 0.)
    ens_fpert     = kwargs.get('ensemble_forcep', 0.)
    share_forcing = kwargs.get('share_forcing', True)
    seaice        = kwargs.get('seaice'       , False)
        
    slns=[] #storage for each ensemble sln.
    
    if share_forcing:
        Si = Forcing(sphere=st).Si #get starting position for red eddy forcing
        kwargs['red_eddy_start'] = Si #add start, then ensemble members decouple over time
    
    for ee in tqdm(range(n_ens)): #generate each ensemble member individually and run barotropic model
        
        ics_e = np.array([st.vortp  + np.random.normal(loc=0., scale = ens_vpert, size= st.vortp.shape),
                          st.thetap + np.random.normal(loc=0., scale = ens_tpert, size= st.thetap.shape)])
            
        st_e= Sphere(base_state=st.base_state)
        st_e.set_ics(ics_e)
        
        if seaice:
            st_e.add_seaice(**kwargs)
        
        solver = Solver(st_e, T=T, **kwargs)
        slns.append(solver.integrate_dynamics())
        

    slns = xr.concat(slns, dim= "ens_mem")
    slns.assign_coords(ens_mem= range(len(slns.ens_mem)))
    
    return slns



#for some reason calling this from a jupyter notebook causes some problems...
## Define your function to be parallelized
def worker_function(data_subset):
    return cbm.apply_find_bimodality(data_subset)

def parallel_process_data(ensemble, num_processes, save=False, save_str=None):
    if save and (save_str is None):
        raise ValueError("if saving, must provide a str for file name")
    for bb in range(0, ensemble.shape[0], num_processes):
        tst = time.time()                
        # Determine the actual number of processes to be used for the current chunk
        current_num_processes = min(num_processes, ensemble.shape[0] - bb)
        
        print(f'working on {bb}-{bb+current_num_processes} out of {ensemble.shape[0]}')

        pool = multiprocessing.Pool(processes=current_num_processes)

        
        # Create a subset of the ensemble for the current chunk
        print('time to load in :')
        data_brick = ensemble.isel(run=slice(bb, bb + current_num_processes)).values
        print('done loading')
        # Convert the subset to a list of data for each process
        data_chunks = [data_brick[i] for i in range(current_num_processes)]
        
        # Apply the worker function to each chunk in parallel
        results = pool.map(worker_function, data_chunks)
        pool.close()
        pool.join()
        
                # Extract results and deltas from multiprocessing output
        if bb==0:
            bm_results = [result[0] for result in results]
            deltas = [result[1] for result in results]
        else:
            bm_results.extend([result[0] for result in results])
            deltas.extend([result[1] for result in results])
        print('time for one loop,', time.time()-tst)
        del data_brick
        del data_chunks
        gc.collect()
    
    print('done detecting..., converting to xarrays')
    
    bm_results = np.array(bm_results)
    deltas = np.array(deltas)

    
    # Get dimensions and coordinates from ensemble
    dims = [dim for dim in ensemble.dims if dim != 'ens_mem']
    coords = {dim: ensemble[dim].values for dim in dims}
    #print(coords)
    # Create xarray DataArrays with the same dimensions and coordinates as ensemble
    bm_results_xr = xr.DataArray(bm_results, dims=dims, coords=coords)
    deltas_xr = xr.DataArray(deltas, dims=dims, coords=coords)
    
    # Combine the DataArrays into a single Dataset
    combined_dataset = xr.Dataset({'bm_results': bm_results_xr, 'deltas': deltas_xr})
    if save:
        rpath = '/doppler/data8/bertossa/bm3/'
        combined_dataset.to_netcdf(rpath+save_str+'.nc')
    return combined_dataset

        
        

 