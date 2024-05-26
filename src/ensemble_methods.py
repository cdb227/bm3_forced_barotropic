import numpy as np
import xarray as xr
import random

from tqdm import tqdm # used for progress tracking
from functools import partial

from cython_routines import bm_methods as cbm

from model.sphere import Sphere
from model.solver import Solver
from model.forcing import Forcing
from utils import config, constants



#cython C methods
#import sys
#import pyximport
#pyximport.install(setup_args={"script_args" : ["--verbose"]})
#import bm_methods.bm_methods as bm_methods
#import forced_barotropic_sphere.bm_methods as bm_methods

def integrate_ensemble(st, T, **kwargs):
    """
    Function to generate an ensemble of forced barotropic model,
    these are initialized with the same forcing term, which decorrelates after about a week.
    """
    M = st._ntrunc
    nlon = 2*M + 1
    nlat = M + 1
    
    n_ens         = kwargs.get('ensemble_size', config.DEFAULT_ENS_SIZE)
    ens_vpert     = kwargs.get('ensemble_vortp', 0.)
    ens_tpert     = kwargs.get('ensemble_thetap', 0.)
    ens_fpert     = kwargs.get('ensemble_forcep', 0.)
    share_forcing = kwargs.get('share_forcing', True)
        
    slns=[] #storage for each ensemble sln.
    
    if share_forcing:
        Si = Forcing(sphere=st).Si #get starting position for red eddy forcing
        kwargs['red_eddy_start'] = Si #add start, then ensemble members decouple over time
    
    for ee in tqdm(range(n_ens)): #generate each ensemble member individually and run barotropic model
        
        ics_e = np.array([st.vortp  + np.random.normal(loc=0., scale = ens_vpert, size= st.vortp.shape),
                          st.thetap + np.random.normal(loc=0., scale = ens_tpert, size= st.thetap.shape)])
            
        st_e= Sphere(base_state=st.base_state)
        st_e.set_ics(ics_e)
        
        solver = Solver(st_e, T=T, **kwargs)
        slns.append(solver.integrate_dynamics())
        

    slns = xr.concat(slns,dim= "ens_mem")
    
    return slns

##+++bimodal specific functions+++
def fit_KDE(ensemble, pad = False, pbar= None):
    if pbar is not None:
        pbar.update(1)
    """fit a KDE to a distribution and find the critical points"""
    #TODO: check with our bm conditions? dependent on ensemble size, potentially useful for FP/FN experiments, but we'll likely
    #use much larger ensembles for synthetic study
    bw = 0.45730505192732634 * np.std(ensemble)
    roots = cbm.recursive_rootfinding(np.min(ensemble),np.max(ensemble),
                                                tol=bw/10., fargs=(ensemble,bw))
    
    Ms = roots[::2] #maxs
    ms = roots[1::2]#mins
    
    # Pad Ms and ms to fixed lengths to work with ufunc
    if pad:
        Ms_padded = np.full(5, np.nan)
        ms_padded = np.full(5, np.nan)
        Ms_padded[:len(Ms)] = Ms
        ms_padded[:len(ms)] = ms
        return Ms_padded, ms_padded
    else:
        return np.array(Ms), np.array(ms)


    #means,mins = bm_methods.root_finding(ensemble, bw)
    #binary_ensemble = np.zeros(ensemble.shape, dtype=int)
#     bimodal= False
#     if len(means) == 2:# and (4 < len(np.where(ensemble < mins[0])[0]) < 46) \
#     #and ((continuous_pdf(mins[0], ensemble, bw)/continuous_pdf(means[0], ensemble, bw)) < 0.85) \
#     #and ((continuous_pdf(mins[0], ensemble, bw)/continuous_pdf(means[1], ensemble, bw)) < 0.85):
#         #binary_ensemble[ensemble > mins[0]] = 1
#         bimodal=True
#     return bimodal#, binary_ensemble

    
def find_roots(ensemble_da):
    """find all instances in slns in which the ensemble is bimodal using KDE estimation"""
    
    total_steps = ensemble_da.sizes['time'] * ensemble_da.sizes['y'] * ensemble_da.sizes['x']
    pbar = tqdm(total=total_steps, desc="Processing", position=0, leave=True)

    fit_KDE_pbar = partial(fit_KDE, pbar=pbar)
    # Apply the Cython function using xarray.apply_ufunc
    roots = xr.apply_ufunc(
        fit_KDE_pbar,  # the function to apply
        ensemble_ss,           # the input DataArray
        input_core_dims=[['ens_mem']],  # specify dimensions
        output_core_dims=[['maxs'],['mins']], # output dimensions
        vectorize=True,                # whether to vectorize, parallelize isn't really needed here
        kwargs={'pad':True}
    )
    return roots
    
    
#+++ Generate additional noise functions+++
def generate_randomnoise(forcing, pert= 1e-12):
    """Generate small white noise perturbations to a forcing instance to simulate small scale processes for each ensemble mem."""
    return np.random.normal(loc=0., scale = pert, size= (forcing.Nt+2,len(forcing.sphere.glat),len(forcing.sphere.glon)))

        
    
        
        

 